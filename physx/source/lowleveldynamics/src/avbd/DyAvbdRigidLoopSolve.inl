// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
void AvbdDynamicsContext::sampleLoopRows(PxU32 root, PxU32 idx)
{
    const auto &group = mRigidGroups[root];
    auto &block = *mLoopScratch[group.loopSlot];
    const auto s = block.order[idx], id = group.blocks[s];
    const auto &b = mBodies[id];
    auto &H = block.at(idx, idx);
    auto &g = block.g[idx];
    block.clearColumn(idx);
    block.locks[idx] = b.core ? PxU32(b.core->lockFlags) : 0;
    const PxMat33d inertia(avbdOgcDouble(b.inertiaWorldInvH2.column0), avbdOgcDouble(b.inertiaWorldInvH2.column1),
                           avbdOgcDouble(b.inertiaWorldInvH2.column2));
    const auto rotation =
        avbdRotationInertialPotential<double>(avbdNpRotation(b.qHat), avbdNpRotation(b.q), inertia);
    const auto dp = avbdOgcDouble(b.p) - avbdOgcDouble(b.pHat);
    bool scalar = true;
    double energy = .5 * double(b.massInvH2) * dp.magnitudeSquared() + rotation.energy;
    for (unsigned d = 0; d < 3; ++d)
    {
        H[d][d] = b.massInvH2;
        g[d] = double(b.massInvH2) * dp[d];
        g[d + 3] = rotation.gradient[d];
        for (unsigned k = 0; k < 3; ++k)
            H[d + 3][k + 3] = rotation.gaussNewton[d][k];
    }
    const PxReal invH2 = mDt > 0 ? 1 / (mDt * mDt) : 3600;
    for (unsigned slot : block.incidence[idx])
    {
        const auto &entry = mRigidGroupRows[group.rowStart + slot];
        const PxU32 r = entry.row;
        const auto row = mRows[r];
        if (row.header().flags & eAVBD_ROW_INACTIVE)
            continue;
        scalar &=
            !((row.header().flags & eAVBD_ROW_CONTACT) && row.scalar().positionTangent && row.scalar().mu > 0);
        if (scalar)
            energy += rigidRowEnergy(r, entry.representative, mBodies, invH2);
        const auto &sample = block.samples[slot];
        AvbdGroupFactorT<2> factors[3] = {};
        const PxU32 offsets[] = {0, 3};
        for (unsigned j = 0; j < sample.n; ++j)
        {
            const auto member = group.blocks[sample.ids[j]];
            const auto &body = mBodies[member];
            AvbdRigidAssemblyT<PxReal, 2> a = {};
            a.coupledRows = &r;
            a.count = 1;
            a.member = j;
            a.factors = factors;
            a.consistent = true;
            a.factorOffsets = offsets;
            a.lockedMask = body.core ? PxU32(body.core->lockFlags) : 0;
            accumulateRigidConstraintRow(member, r, invH2, a);
            if (!a.consistent)
            {
                block.failed.store(true);
                return;
            }
            auto *dest = member == id ? H : block.peerData[sample.edge].h;
            auto *grad = member == id ? g : block.peerData[sample.edge].g;
            for (unsigned d = 0; d < 6; ++d)
            {
                grad[d] += a.gradient[d];
                for (unsigned k = 0; k < 6; ++k)
                    dest[d][k] += a.hessian[d][k];
            }
        }
        if (sample.n == 2)
            for (const auto &f : factors)
                if (f.curvature)
                    block.cross(sample.ids, f.jacobian, f.curvature);
    }
    block.scalarColumn[s] = scalar;
    block.energy[s] = energy;
}

void AvbdDynamicsContext::prepareBackTrial(unsigned root, unsigned i)
{
    const auto &group = mRigidGroups[root];
    auto &b = *mLoopScratch[group.loopSlot];
    const unsigned s = b.order[i];
    const auto &body = mBodies[group.blocks[s]];
    const PxTransform original(body.p, body.q);
    const auto *x = b.step[s];
    auto &pose = b.fullPose[s];
    pose = original;
    avbdApplyDelta(pose, PxVec3(PxReal(x[0]), PxReal(x[1]), PxReal(x[2])),
                   PxVec3(PxReal(x[3]), PxReal(x[4]), PxReal(x[5])));
    b.fullSane[s] = pose.isSane();
    const auto dp = pose.p - original.p;
    const auto angle = avbdAcceptedRotationDisplacement(original.q, pose.q);
    b.fullStep[s] = PxMax(dp.magnitude() / (getLengthScale() * AVBD_ARTIC_POSITION_REL_TOL),
                          angle.magnitude() / AVBD_ARTIC_ROTATION_TOL);
    auto &delta = b.fullDelta[i];
    for (unsigned d = 0; d < 3; ++d)
    {
        delta[d] = dp[d];
        delta[d + 3] = angle[d];
    }
    const auto &diag = b.at(i, i);
    double slope = 0, quadratic = 0;
    double y[6];
    for (unsigned d = 0; d < 6; ++d)
    {
        slope += b.g[i][d] * delta[d];
        y[d] = delta[d] / diag[d][d];
    }
    for (unsigned e = 0; e < 6; ++e)
        for (unsigned d = 0; d < e; ++d)
            y[d] += diag[e][d] * delta[e];
    for (unsigned j = 0; j < b.nb[i]; ++j)
    {
        const auto high = b.after[i][j];
        const auto &h = b.at(high, i);
        for (unsigned e = 0; e < 6; ++e)
            for (unsigned d = 0; d < 6; ++d)
                y[d] += h[e][d] * b.fullDelta[high][e];
    }
    for (double entry : y)
        quadratic += entry * entry;
    b.fullSlope[i] = slope;
    b.fullQuadratic[i] = quadratic;
}

void AvbdDynamicsContext::solveLoopGroup(PxU32 root, PxReal invH2)
{
    auto &group = mRigidGroups[root];
    ++group.attempts;
    auto &block = *mLoopScratch[group.loopSlot];
    if (block.failed.load(std::memory_order_relaxed))
    {
        PxAtomicExchange(&mLocalDomainFailed, 1);
        return;
    }
    PxU64 profileStart = mStep.profile ? PxTime::getCurrentCounterValue() : 0;
    PxTransform original[AvbdRigidLoop::Bodies], candidate[AvbdRigidLoop::Bodies];
    bool scalar = true;
    double initialEnergy = 0;
    for (unsigned s = 0; s < group.count; ++s)
    {
        const auto &b = mBodies[group.blocks[s]];
        original[s] = PxTransform(b.p, b.q);
        scalar &= block.scalarColumn[s];
        initialEnergy += block.energy[s];
    }
    ++group.kernelSolved;

    auto energy = [&](const PxTransform *states)
    {
        const AvbdRigidPoseView poses = {mBodies, group.blocks, states, group.count};
        double value = 0;
        for (PxU32 s = 0; s < group.count; ++s)
        {
            const auto &b = mBodies[group.blocks[s]];
            const auto dp = avbdOgcDouble(states[s].p) - avbdOgcDouble(b.pHat);
            const PxMat33d inertia(avbdOgcDouble(b.inertiaWorldInvH2.column0),
                                   avbdOgcDouble(b.inertiaWorldInvH2.column1),
                                   avbdOgcDouble(b.inertiaWorldInvH2.column2));
            value +=
                .5 * double(b.massInvH2) * dp.magnitudeSquared() +
                avbdRotationInertialPotential<double>(avbdNpRotation(b.qHat), avbdNpRotation(states[s].q), inertia)
                    .energy;
        }
        for (PxU32 s = 0; s < group.rowCount; ++s)
        {
            const auto &e = mRigidGroupRows[group.rowStart + s];
            value += rigidRowEnergy(e.row, e.representative, poses, invH2);
        }
        return value;
    };
    const double initial = scalar ? initialEnergy : 0;
    double fraction = 1;
    for (PxU32 search = 0; search < 16; ++search, fraction *= .5)
    {
        PxAtomicIncrement(&mStep.localSearchTrials);
        bool valid = true;
        double delta[AvbdRigidLoop::Bodies][6], slope = 0, quadratic = 0;
        if (!search)
        {
            for (unsigned i = 0; i < block.n; ++i)
            {
                const auto s = block.order[i];
                candidate[s] = block.fullPose[s];
                valid &= block.fullSane[s];
                slope += block.fullSlope[i];
                quadratic += block.fullQuadratic[i];
            }
        }
        else
            for (PxU32 s = 0; s < group.count; ++s)
            {
                candidate[s] = original[s];
                const auto *x = block.step[s];
                avbdApplyDelta(candidate[s],
                               PxVec3(PxReal(x[0] * fraction), PxReal(x[1] * fraction), PxReal(x[2] * fraction)),
                               PxVec3(PxReal(x[3] * fraction), PxReal(x[4] * fraction), PxReal(x[5] * fraction)));
                valid &= candidate[s].isSane();
                const auto dp = candidate[s].p - original[s].p;
                const auto angle = avbdAcceptedRotationDisplacement(original[s].q, candidate[s].q);
                for (PxU32 d = 0; d < 3; ++d)
                {
                    delta[s][d] = dp[d];
                    delta[s][d + 3] = angle[d];
                }
                for (PxU32 d = 0; d < 6; ++d)
                    slope += block.g[block.perm[s]][d] * delta[s][d];
            }
        // Pure predicates form one admission conjunction. Reject a failed
        // descent model before paying for the complete geometric certificate.
        if (!valid || slope > 0)
            continue;
        PxU64 phaseStart = mStep.profile ? PxTime::getCurrentCounterValue() : 0;
        const double trial =
            scalar ? energy(candidate) : (!search ? slope + .5 * quadratic : block.model(delta, slope));
        if (mStep.profile)
            group.modelTicks += PxTime::getCurrentCounterValue() - phaseStart;
        if (!std::isfinite(trial) || (scalar ? trial > initial + 1.e-4 * slope : trial > 0))
            continue;
        const AvbdRigidPoseView poses = {mBodies, group.blocks, candidate, group.count};
        if (mStep.profile)
            phaseStart = PxTime::getCurrentCounterValue();
        for (PxU32 s = 0; valid && s < block.geometryRows.size(); ++s)
        {
            const auto &row = mRows[block.geometryRows[s]];
            if (!(row.header().flags & eAVBD_ROW_INACTIVE) &&
                row.header().kernelId == eAVBD_ROW_KERNEL_TETRAHEDRON)
                valid = tetrahedronPath(row, mTetrahedronMaterials[row.header().geometryIndex - 1], mRowSupports,
                                        mBodies, poses);
            if (valid && row.header().kernelId == eAVBD_ROW_KERNEL_NP_CONTACT)
                valid = localNpCandidateAdmissible(row, poses);
        }
        if (mStep.profile)
            group.geometryTicks += PxTime::getCurrentCounterValue() - phaseStart;
        if (!valid)
            continue;
        for (PxU32 s = 0; s < group.count; ++s)
        {
            const PxU32 b = group.blocks[s];
            if (!search)
                mPrimalSteps[b] = block.fullStep[s];
            else
            {
                const auto dp = candidate[s].p - original[s].p;
                const auto angle = avbdAcceptedRotationDisplacement(original[s].q, candidate[s].q);
                mPrimalSteps[b] = PxMax(dp.magnitude() / (getLengthScale() * AVBD_ARTIC_POSITION_REL_TOL),
                                        angle.magnitude() / AVBD_ARTIC_ROTATION_TOL);
            }
            if (candidate[s].p != original[s].p || !(candidate[s].q == original[s].q))
                mPrimalChangedIteration[b] = mStep.iteration + 1;
            mBodies[b].p = candidate[s].p;
            mBodies[b].q = candidate[s].q;
        }
        if (mStep.profile)
            group.searchTicks += PxTime::getCurrentCounterValue() - profileStart;
        ++group.accepted;
        return;
    }
    if (mStep.profile)
        group.searchTicks += PxTime::getCurrentCounterValue() - profileStart;
    PxAtomicIncrement(&mStep.localSearchExhausted);
}
