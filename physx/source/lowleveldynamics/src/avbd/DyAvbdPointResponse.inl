// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
namespace
{
bool activePointLock(const AvbdRowConstRef& row)
{
    return (row.header().flags & (eAVBD_ROW_POINT_LOCK | eAVBD_ROW_INACTIVE)) == eAVBD_ROW_POINT_LOCK;
}

void pointJacobian(const AvbdSolverBody& body, const PxVec3& localArm, const PxVec3& axis,
                   AvbdPointResponse::Real (&j)[6])
{
    const PxVec3 angular = body.q.rotate(localArm).cross(axis);
    for (PxU32 d = 0; d < 3; ++d)
    {
        j[d] = axis[d];
        j[d + 3] = angular[d];
    }
}
}

bool AvbdDynamicsContext::pointContactProgram(const PxArray<PxU8>& jointPrograms) const
{
    if (!mArticJoints.empty() || mRows.size() <= mContactRowCount || mContactRowCount != mNpContactRowCount)
        return false;
    bool active = false;
    for (PxU32 r = mContactRowCount; r < mRows.size(); ++r)
    {
        if (mRows[r].header().flags & eAVBD_ROW_INACTIVE)
            continue;
        if (!activePointLock(mRows[r]))
            return false;
        active = true;
    }
    if (!active)
        return false;
    for (PxU32 id = 0; id < mBodies.size(); ++id)
    {
        const auto& body = mBodies[id];
        if (!jointPrograms[id] || !avbdBodyResponds(body))
            continue;
        const PxU32 begin = mRowProgram.blockOffsets[id], end = mRowProgram.blockOffsets[id + 1];
        if (end - begin > AvbdPointResponse::MaxIncidences || !body.core || body.core->lockFlags ||
            !body.invInertia.x || !body.invInertia.y || !body.invInertia.z)
            return false;
        PxU32 axes = 0;
        for (PxU32 s = begin; s < end; ++s)
            axes += activePointLock(mRows[mRowProgram.blockRows[s]]);
        if (axes > 3 * AvbdPointResponse::MaxLinks)
            return false;
    }
    for (PxU32 r = 0; r < mNpContactRowCount; ++r)
    {
        const auto row = mRows[r];
        const auto& h = row.header();
        if (h.flags & eAVBD_ROW_INACTIVE)
            continue;
        if (!(h.flags & eAVBD_ROW_CONTACT) || h.supportCount || row.scalar().contactRadius > 0)
            return false;
        // A contact between two constrained moving endpoints can have a
        // shared first-ring body. Its relative compliance needs both cross terms.
        if (h.bodyA < jointPrograms.size() && h.bodyB < jointPrograms.size() &&
            jointPrograms[h.bodyA] && jointPrograms[h.bodyB] &&
            avbdBodyResponds(mBodies[h.bodyA]) && avbdBodyResponds(mBodies[h.bodyB]))
            return false;
    }
    return true;
}

bool AvbdDynamicsContext::pointEndpointCompliance(PxU32 center, const PxVec3& arm, const PxVec3& normal,
                                                  PxReal& value, bool& coupled)
{
    value = 0;
    if (center >= mBodies.size() || !avbdBodyResponds(mBodies[center]))
        return true;
    const auto& offsets = mRowProgram.blockOffsets;
    const auto& body = mBodies[center];
    value = mDt * mDt * avbdEndpointAxisResponse(body, normal, arm);
    // The program check bounded every joint endpoint. A high-degree free
    // contact body contributes its ordinary spatial mobility in constant time.
    if (offsets[center + 1] - offsets[center] > AvbdPointResponse::MaxIncidences)
        return true;
    PxU32 members[AvbdPointResponse::MaxBodies] = {center}, count = 1;
    bool hasPoint = false;
    const auto slotOf = [&](PxU32 id) {
        PxU32 slot = 0;
        while (slot < count && members[slot] != id)
            ++slot;
        return slot;
    };
    for (PxU32 s = offsets[center]; s < offsets[center + 1]; ++s)
    {
        const auto link = mRows[mRowProgram.blockRows[s]];
        if (!activePointLock(link))
            continue;
        hasPoint = true;
        const PxU32 peer = link.header().bodyA == center ? link.header().bodyB : link.header().bodyA;
        if (peer < mBodies.size() && avbdBodyResponds(mBodies[peer]) && slotOf(peer) == count)
        {
            if (count == AvbdPointResponse::MaxBodies)
                return false;
            members[count++] = peer;
        }
    }
    if (!hasPoint)
        return true;
    // Keep the center and its direct dynamic neighbors mobile. A joint
    // crossing the patch boundary contributes its local diagonal once.
    AvbdPointPatch patch;
    patch.count = count;
    patch.axisCount = 0;
    patch.inverseDt = 1 / mDt;
    for (PxU32 slot = 0; slot < count; ++slot)
    {
        const PxU32 id = members[slot];
        const auto& b = mBodies[id];
        patch.bodies[slot] = {b.massInvH2, b.invInertia, b.qStart};
        for (PxU32 s = offsets[id]; s < offsets[id + 1]; ++s)
        {
            const auto link = mRows[mRowProgram.blockRows[s]];
            if (!activePointLock(link))
                continue;
            const bool first = id == link.header().bodyA;
            const auto peer = first ? link.header().bodyB : link.header().bodyA;
            const PxU32 other = slotOf(peer);
            if (other < slot)
                continue;
            const auto& law = link.scalar();
            PxReal penalty = PxMax(law.rho, b.massInvH2);
            if (peer < mBodies.size() && avbdBodyResponds(mBodies[peer]))
                penalty = PxMax(penalty, mBodies[peer].massInvH2);
            if (patch.axisCount == AvbdPointPatch::MaxAxes)
                return false;
            auto& axis = patch.axes[patch.axisCount++];
            axis.first = slot;
            axis.second = other;
            axis.penalty = penalty;
            axis.normal = law.n;
            axis.angularA = b.q.rotate(first ? law.rA : law.rB).cross(law.n);
            axis.angularB = other == count ? PxVec3(0) :
                mBodies[peer].q.rotate(first ? law.rB : law.rA).cross(law.n);
        }
    }
    AvbdPointResponseArgs args = {&patch, {}, 0, false};
    pointJacobian(body, arm, normal, args.jacobian);
    mKernels.launch(AVBD_POINT_RESPONSE, &args);
    value = args.response;
    coupled = true;
    return args.solved;
}

void AvbdDynamicsContext::conditionPointInterface(PxU32 index)
{
    auto row = mRows[index];
    if (row.header().flags & eAVBD_ROW_INACTIVE)
        return;
    auto& law = row.scalar();
    bool coupled = false;
    PxReal a, b;
    if (!pointEndpointCompliance(row.header().bodyA, law.ogcRA, law.n, a, coupled) ||
        !pointEndpointCompliance(row.header().bodyB, law.ogcRB, law.n, b, coupled))
    {
        PxAtomicExchange(&mLocalDomainFailed, 1);
        return;
    }
    // a + b is relative normal displacement per unit force, in m/N.
    // This conditions the existing unilateral AL row before its first primal.
    if (coupled && a + b > 0)
        law.rho = PxMax(law.rho, PxMin(AVBD_CONTACT_PENALTY_MAX, AVBD_CONTACT_CONDITIONING / (a + b)));
}
