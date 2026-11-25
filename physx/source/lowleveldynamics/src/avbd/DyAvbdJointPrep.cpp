// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/DyAvbdInternal.h"

#include "DyConstraint.h"
#include "DyConstraintPrep.h"
#include "DyFeatherstoneArticulation.h"
#include "DyIslandManager.h"
#include "PxsIslandSim.h"
#include "PxsSimpleIslandManager.h"
#include "PxConstraint.h"
#include "PxConstraintDesc.h"
#include "foundation/PxHashMap.h"
#include "foundation/PxHashSet.h"


using namespace physx;
using namespace Dy;

void AvbdDynamicsContext::invalidateConstraint(const Constraint* constraint)
{
    // A pool address can be reused by a different constraint before the next
    // DCD epoch. Host invalidation removes its history before that reuse.
    for(PxU32 i=0;i<mJoints.size();++i)
        if(mJoints[i].constraint==constraint) { mJoints.remove(i);return; }
}

void AvbdDynamicsContext::buildJointSources(PxU32 nbIslands,
	const IG::IslandId* islandIds)
{
	PxArray<AvbdJointSrc> prev;
	prev.swap(mJoints);
	PxArray<AvbdJointState> previousStates;
	previousStates.swap(mJointStates);
	// Lookup only: preserve island/row order and the first matching old source.
	// These indices live for this packing call, so removals, sleeping islands
	// and endpoint remapping cannot leave a persistent stale identity cache.
	PxHashMap<const Constraint*, PxU32> previousSources;
	for(PxU32 p = 0; p < prev.size(); ++p)
		previousSources.insert(prev[p].constraint, p);
	AvbdBodyIndex bodyIndices;bodyIndices.build(mBodies);
	PxHashSet<const Constraint*> visited;
	const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();
	for(PxU32 i = 0; i < nbIslands; ++i)
	{
		const IG::Island& island = islandSim.getIsland(islandIds[i]);
		START_ENUMERATING_ISLAND_EDGES(IG::Edge::eCONSTRAINT)
		{
			GET_CURRENT_ISLAND_EDGE
			Constraint* constraint = mIslandManager.getConstraint(edgeId);
			if(!constraint || (!constraint->solverPrep && !constraint->multiBody))
				continue;
			if(constraint->flags & PxConstraintFlag::eDISABLE_CONSTRAINT)
				continue;
			if(constraint->flags & PxConstraintFlag::eBROKEN)
				continue;
			if(constraint->multiBody && !visited.insert(constraint)) continue;
            if(constraint->multiBody)
                for(PxU32 support=0;support<4;++support)
                {
                    PxsBodyCore* core=constraint->multiBody->bodies[support];
                    if(!core || bodyIndices.find(core)!=PX_MAX_U32) continue;
                    if(!core->isKinematic())
                    {
                        PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION,PX_FL,
                            "Material support is missing from the active Host island; step not published.");
                        PxAtomicExchange(&mLocalDomainFailed,1);
                        continue;
                    }
                    AvbdSolverBody boundary;
                    initSolverBody(boundary,*core,NULL,PxVec3(0.0f),mDt);
                    bodyIndices.insert(core,mBodies.size());
                    mBodies.pushBack(boundary);
                }
			AvbdJointSrc src;
			PxMemZero(&src, sizeof(src));
			src.constraint = constraint;
			src.bodyA = bodyIndices.find(constraint->bodyCore0);
			src.bodyB = bodyIndices.find(constraint->bodyCore1);
			src.nbRows = 0;
            if(!constraint->multiBody && !constraint->elasticFramePrep)
            {
                AvbdJointState state;
                PxMemZero(&state,sizeof(state));
                for(PxU32 r=0;r<AVBD_MAX_JOINT_ROWS;++r) state.rho[r]=AVBD_RHO;
                const auto* previous=previousSources.find(constraint);
                if(previous)
                {
                    const AvbdJointSrc& old=prev[previous->second];
                    const AvbdJointState& history=previousStates[old.payloadIndex];
                    src.nbRows=old.nbRows;
                    state.pointLocalA=history.pointLocalA;state.pointLocalB=history.pointLocalB;
                    state.pointLock=history.pointLock;state.pointInitialized=history.pointInitialized;
                    for(PxU32 r=0;r<AVBD_MAX_JOINT_ROWS;++r)
                    {
                        state.lambda[r]=history.lambda[r]*AVBD_ALPHA*AVBD_GAMMA;
                        state.rho[r]=PxClamp(history.rho[r]*AVBD_GAMMA,AVBD_RHO,AVBD_RHO_MAX);
                    }
                }
                src.payloadIndex=mJointStates.size();mJointStates.pushBack(state);
            }
			mJoints.pushBack(src);
			GET_NEXT_ISLAND_EDGE
		}
	}
}

void AvbdDynamicsContext::buildArticJointSources()
{
	PxArray<AvbdArticJointSrc> prev;
	prev.swap(mArticJoints);
	const PxReal h = mDt;
	const PxReal invH2 = (h > 0.0f) ? (1.0f / (h * h)) : 3600.0f;
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		const AvbdSolverBody& child = mBodies[i];
		if(!child.artic || child.linkIndex == 0)
			continue;
		ArticulationData& artData = child.artic->getArticulationData();
		if(child.linkIndex >= artData.getLinkCount())
			continue;
		const ArticulationLink& link = artData.getLink(child.linkIndex);
		if(!link.inboundJoint || link.parent == DY_ARTICULATION_LINK_NONE)
			continue;
		const PxU32 parentBody = findBody(artData.getLink(link.parent).bodyCore);
		if(parentBody == PX_MAX_U32)
			continue;
		AvbdArticJointSrc src;
		PxMemZero(&src, sizeof(src));
		src.joint = link.inboundJoint;
		src.parentBody = parentBody;
		src.childBody = i;
		src.nbRows = 0;
		const PxReal massA = (mBodies[parentBody].invMass > 0.0f)
			? (1.0f / mBodies[parentBody].invMass) : 100.0f;
		const PxReal massB = (child.invMass > 0.0f) ? (1.0f / child.invMass) : 100.0f;
		// demo3d joint.cpp: a hard row starts at PENALTY_MIN and the multiplier
		// carries the load; avbdHardJointPenalty floors the live penalty at the
		// row's own inertial scale so one GS block step is well conditioned.
		// Fixed floors of 1e5/1e6 (1000x the 20 g links' m/h^2) stalled block GS
		// on the mechanism modes of a light loop chain: after 2048 sweeps body
		// gradients of ~30 N remained and their sum pitched the scissor base.
		PX_UNUSED(massA);
		PX_UNUSED(massB);
		PX_UNUSED(invH2);
		for(PxU32 r = 0; r < PxArticulationAxis::eCOUNT; ++r)
			src.rho[r] = AVBD_RHO;
		// demo3d joint.cpp / WebPhysics: a finite-stiffness joint is a spring
		// potential whose penalty is the authored stiffness, implicit in the
		// 6x6.  Capping it (AVBD_2 S_vel=100/h) made the scissor drive 17x
		// softer than authored: the near-massless mechanism rang on that soft
		// spring at +-300 N and its inertial reaction pitched the base.
		for(PxU32 axis = 0; axis < PxArticulationAxis::eCOUNT; ++axis)
		{
			const PxArticulationDrive& drive = src.joint->drives[axis];
			if(src.joint->motion[axis] != PxArticulationMotion::eLOCKED &&
				drive.driveType != PxArticulationDriveType::eNONE &&
				drive.stiffness > 0.0f)
			{
				src.rho[PxArticulationAxis::eCOUNT + axis] =
					PxClamp(drive.stiffness, 1.0f, AVBD_RHO_MAX);
			}
			else
				src.rho[PxArticulationAxis::eCOUNT + axis] = 1.0f;
		}
		for(PxU32 p = 0; p < prev.size(); ++p)
		{
			if(prev[p].joint == src.joint)
			{
				for(PxU32 r = 0; r < AVBD_MAX_ARTIC_ROWS; ++r)
					src.lambda[r] = prev[p].lambda[r] *
						AVBD_ALPHA * AVBD_GAMMA;
				// demo3d Joint::initialize: the penalty is persistent AL state,
				// decayed by gamma and floored; drive slots stay at stiffness.
				for(PxU32 r = 0; r < PxArticulationAxis::eCOUNT; ++r)
					src.rho[r] = PxClamp(prev[p].rho[r] * AVBD_GAMMA,
						AVBD_RHO, AVBD_RHO_MAX);
				break;
			}
		}
        const auto& parent = mBodies[parentBody];
        const PxTransform startA = PxTransform(parent.pStart, parent.qStart) * src.joint->parentPose;
        const PxTransform startB = PxTransform(child.pStart, child.qStart) * src.joint->childPose;
        for (PxU32 axis = 0; axis < PxArticulationAxis::eCOUNT; ++axis)
        {
            if (src.joint->motion[axis] == PxArticulationMotion::eFREE) continue;
            PxVec3 la, aa, lb, ab;
            if (avbdArticFrameAxis(startA, startB, parent.pStart, child.pStart, *src.joint, axis,
                                   la, aa, lb, ab, src.startCoordinate[axis]))
                src.validStartAxes |= 1u << axis;
        }
		mArticJoints.pushBack(src);
	}
}

static void avbdPushArticRow(AvbdRows& rows, AvbdArticJointSrc& src,
	PxU32 parentBody, PxU32 childBody, const PxVec3& jLinA, const PxVec3& jAngA,
	const PxVec3& jLinB, const PxVec3& jAngB, PxReal C, PxReal rho,
	PxReal minL, PxReal maxL, PxU8 flags,
	ArticulationJointCore* joint = NULL, PxU32 axis = 0, PxReal target = 0.0f,
	PxU8 flip = 0)
{
	if(src.nbRows >= AVBD_MAX_ARTIC_ROWS)
		return;
	AvbdRowBuilder row;
	PxMemZero(&row, sizeof(row));
	row.header().bodyA = parentBody;
	row.header().bodyB = childBody;
	row.scalar().jLinA = jLinA;
	row.scalar().jAngA = jAngA;
	row.scalar().jLinB = jLinB;
	row.scalar().jAngB = jAngB;
	row.scalar().C0 = C;
	row.scalar().CLin = C;
	row.scalar().lambda = src.lambda[src.nbRows];
	row.scalar().rho = rho;
	row.scalar().minLambda = minL;
	row.scalar().maxLambda = maxL;
	row.header().flags = flags;
	row.scalar().articJoint = joint;
	row.scalar().articAxis = PxU8(axis);
	row.scalar().articFlip = flip;
	row.scalar().articTarget = target;
	row.scalar().reactionOutput = NULL;
	rows.pushBack(row);
	src.nbRows++;
}

static void avbdPushInactiveArticRow(AvbdRows& rows, AvbdArticJointSrc& src,
	PxU32 parentBody, PxU32 childBody)
{
	src.lambda[src.nbRows] = 0.0f;
	const PxReal rho = src.rho[src.nbRows];
	avbdPushArticRow(rows, src, parentBody, childBody,
		PxVec3(0.0f), PxVec3(0.0f), PxVec3(0.0f), PxVec3(0.0f),
		0.0f, rho, 0.0f, 0.0f, eAVBD_ROW_INACTIVE);
}

void AvbdDynamicsContext::refreshArticJointRows()
{
	for(PxU32 j = 0; j < mArticJoints.size(); ++j)
	{
		AvbdArticJointSrc& src = mArticJoints[j];
		if(!src.joint || src.parentBody >= mBodies.size() || src.childBody >= mBodies.size())
			continue;
		src.nbRows = 0;
		const ArticulationJointCore& joint = *src.joint;
		const AvbdSolverBody& parent = mBodies[src.parentBody];
		const AvbdSolverBody& child = mBodies[src.childBody];
        const PxTransform frameA = PxTransform(parent.p, parent.q) * joint.parentPose;
        const PxTransform frameB = PxTransform(child.p, child.q) * joint.childPose;
		for(PxU32 axis = 0; axis < PxArticulationAxis::eCOUNT; ++axis)
		{
			PxVec3 jLinA, jAngA, jLinB, jAngB;
			PxReal C = 0.0f;
			const bool ok = joint.motion[axis] != PxArticulationMotion::eFREE &&
                avbdArticFrameAxis(frameA, frameB, parent.p, child.p, joint, axis,
				jLinA, jAngA, jLinB, jAngB, C);
			const PxU8 motion = joint.motion[axis];
			if(!ok)
				avbdPushInactiveArticRow(mRows, src, src.parentBody, src.childBody);
			else if(motion == PxArticulationMotion::eLOCKED)
			{
				avbdPushArticRow(mRows, src, src.parentBody, src.childBody,
					jLinA, jAngA, jLinB, jAngB, C, src.rho[axis],
					-PX_MAX_REAL, PX_MAX_REAL, 0, src.joint, axis, 0.0f, 0);
			}
			else if(motion == PxArticulationMotion::eLIMITED)
			{
				const PxReal low = joint.limits[axis].low;
				const PxReal high = joint.limits[axis].high;
				if(C < low)
					avbdPushArticRow(mRows, src, src.parentBody, src.childBody,
						jLinA, jAngA, jLinB, jAngB, C - low, src.rho[axis],
						-PX_MAX_REAL, 0.0f, eAVBD_ROW_UNILATERAL,
						src.joint, axis, low, 0);
				else if(C > high)
					avbdPushArticRow(mRows, src, src.parentBody, src.childBody,
						-jLinA, -jAngA, -jLinB, -jAngB, high - C, src.rho[axis],
						-PX_MAX_REAL, 0.0f, eAVBD_ROW_UNILATERAL,
						src.joint, axis, high, 1);
				else
					avbdPushInactiveArticRow(mRows, src, src.parentBody, src.childBody);
			}
			else
				avbdPushInactiveArticRow(mRows, src, src.parentBody, src.childBody);
			AvbdRowRef compiled = mRows.back();
            if (!(compiled.header().flags & eAVBD_ROW_INACTIVE) && (src.validStartAxes & (1u << axis)))
            {
                const PxReal startC=src.startCoordinate[axis];
                compiled.scalar().C0 = compiled.scalar().articFlip ? compiled.scalar().articTarget - startC : startC - compiled.scalar().articTarget;
            }
		}
		for(PxU32 axis = 0; axis < PxArticulationAxis::eCOUNT; ++axis)
		{
			PxVec3 jLinA, jAngA, jLinB, jAngB;
			PxReal C = 0.0f;
			const PxArticulationDrive& drive = joint.drives[axis];
			const PxU32 slot = PxArticulationAxis::eCOUNT + axis;
			if(joint.motion[axis] != PxArticulationMotion::eLOCKED &&
				drive.driveType != PxArticulationDriveType::eNONE &&
				drive.stiffness > 0.0f &&
				avbdArticFrameAxis(frameA, frameB, parent.p, child.p, joint, axis, jLinA, jAngA, jLinB, jAngB, C))
			{
				const PxReal err = C - joint.targetP[axis];
				avbdPushArticRow(mRows, src, src.parentBody, src.childBody,
					jLinA, jAngA, jLinB, jAngB, err, src.rho[slot],
					-PX_MAX_REAL, PX_MAX_REAL, eAVBD_ROW_DRIVE,
					src.joint, axis, joint.targetP[axis], 0);
			}
			else
				avbdPushInactiveArticRow(mRows, src, src.parentBody, src.childBody);
		}
	}
}

bool AvbdDynamicsContext::refreshJointRows(bool beginStep)
{
	AvbdBodyIndex bodyIndices;if(beginStep)bodyIndices.build(mBodies);
	bool topologyChanged = false;
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		mBodies[i].pLin = mBodies[i].p;
		mBodies[i].qLin = mBodies[i].q;
	}
	mRows.truncate(PxMax(mContactRowCount,mNativeMaterialEnd));
	Px1DConstraint rows[MAX_CONSTRAINT_ROWS];
	for(PxU32 j = 0; j < mJoints.size(); ++j)
	{
		AvbdJointSrc& src = mJoints[j];
		const PxU32 oldNbRows = src.nbRows;
        Constraint& c=*src.constraint;
        if(c.elasticFramePrep)
        {
            if(beginStep)
            {
                PxConstraintElasticFrameData data;
                bool valid=src.bodyA<mBodies.size() && src.bodyB<mBodies.size() &&
                    !mBodies[src.bodyA].isKinematic && !mBodies[src.bodyB].isKinematic &&
                    c.linBreakForce==PX_MAX_F32 && c.angBreakForce==PX_MAX_F32 &&
                    c.minResponseThreshold==0 && c.elasticFramePrep(data,c.constantBlock);
                if(valid)
                {
                    const PxTransform poses[2]={PxTransform(mBodies[src.bodyA].p,mBodies[src.bodyA].q),
                        PxTransform(mBodies[src.bodyB].p,mBodies[src.bodyB].q)};
                    src.payloadIndex=mElasticFrameMaterials.append(data,poses);
                    valid=src.payloadIndex!=PX_MAX_U32;
                }
                if(!valid)
                {
                    PxGetFoundation().error(PxErrorCode::eINVALID_PARAMETER,PX_FL,
                        "Invalid or unsupported elastic-frame material; step not published.");
                    PxAtomicExchange(&mLocalDomainFailed,1);return true;
                }
            }
            if(src.payloadIndex>=mElasticFrameMaterials.size())
            { PxAtomicExchange(&mLocalDomainFailed,1);return true; }
            const bool active=mElasticFrameMaterials[src.payloadIndex].parameters.active();
            if(!active) { src.nbRows=0;topologyChanged|=oldNbRows!=0;continue; }
            AvbdRow row={};
            row.header().bodyA=src.bodyA;row.header().bodyB=src.bodyB;row.header().flags=eAVBD_ROW_DRIVE;
            row.header().kernelId=eAVBD_ROW_KERNEL_ELASTIC_FRAME;row.header().geometryIndex=src.payloadIndex+1;
            src.nbRows=1;topologyChanged|=src.nbRows!=oldNbRows;mRows.pushBack(row);continue;
        }
		// Ordinary 1D and four-support programs retain their preparation path.
        if(c.multiBody)
        {
            if(beginStep)
            {
                PxConstraintTetrahedronData data;
                bool valid=c.multiBody->prepare(data,c.constantBlock) &&
                    PxIsFinite(data.restDeterminant) && data.restDeterminant!=0.0f &&
                    PxIsFinite(data.energyScale) && data.energyScale>0.0f;
                for(PxU32 s=0;s<4;++s) valid=valid && data.localPoints[s].isFinite();
                if(!valid)
                {
                    PxGetFoundation().error(PxErrorCode::eINVALID_PARAMETER,PX_FL,"Invalid tetrahedral material data; step not published.");
                    PxAtomicExchange(&mLocalDomainFailed,1);
                    return true;
                }
                AvbdTetrahedronMaterial material;
                material.inverseRestDeterminant=1.0/double(data.restDeterminant);
                material.energyScale=data.energyScale;
                material.supportStart=mRowSupports.size();
                for(PxU32 s=0;s<4;++s)
                {
                    AvbdRowSupport support;
                    support.block=bodyIndices.find(c.multiBody->bodies[s]);
                    if(c.multiBody->bodies[s] && support.block==PX_MAX_U32)
                    {
                        PxAtomicExchange(&mLocalDomainFailed,1);
                        return true;
                    }
                    support.localPoint=data.localPoints[s]; support.weight=1.0f;
                    mRowSupports.pushBack(support);
                }
                src.payloadIndex=mTetrahedronMaterials.size();
                mTetrahedronMaterials.pushBack(material);
            }
            if(src.payloadIndex>=mTetrahedronMaterials.size())
            {
                PxAtomicExchange(&mLocalDomainFailed,1);
                return true;
            }
            AvbdRow row={};
            row.header().bodyA=row.header().bodyB=PX_MAX_U32;
            row.header().kernelId=eAVBD_ROW_KERNEL_TETRAHEDRON;
            row.header().geometryIndex=src.payloadIndex+1;
            row.header().supportStart=mTetrahedronMaterials[src.payloadIndex].supportStart;
            row.header().supportCount=4;
            row.header().flags=eAVBD_ROW_DRIVE;
            src.nbRows=1;
            topologyChanged|=src.nbRows!=oldNbRows;
            mRows.pushBack(row);
            continue;
        }
		AvbdJointState& state=mJointStates[src.payloadIndex];
		setupConstraintRows(rows, MAX_CONSTRAINT_ROWS);
		// Identity: contact/joint mass-inertia scale is unsupported on AVBD.
		PxConstraintInvMassScale massScale(1.0f, 1.0f, 1.0f, 1.0f);
		PxVec3p bodyAWorldOffset(0.0f);
		PxVec3p cAtW(0.0f), cBtW(0.0f);
		const PxTransform frameA = bodyPose(src.bodyA, c.bodyCore0);
		const PxTransform frameB = bodyPose(src.bodyB, c.bodyCore1);
		const bool extended =
			(c.flags & PxConstraintFlag::eENABLE_EXTENDED_LIMITS) != 0;
		const PxU32 nbRows = (*c.solverPrep)(rows, bodyAWorldOffset,
			MAX_CONSTRAINT_ROWS, massScale, c.constantBlock, frameA, frameB,
			extended, cAtW, cBtW);
		src.nbRows = PxMin(nbRows, AVBD_MAX_JOINT_ROWS);
        if(beginStep)
        {
            state.passiveLock = PxU8(src.nbRows == 6);
            for(PxU32 r = 0; r < src.nbRows && state.passiveLock; ++r)
                state.passiveLock = PxU8(rows[r].solveHint == PxConstraintSolveHint::eEQUALITY &&
                    rows[r].velocityTarget == 0 && !(rows[r].flags & Px1DConstraintFlag::eSPRING));
            if(state.passiveLock && (!c.bodyCore0 || !c.bodyCore1))
                mStep.hasGroundedJoint = true;
        }
		topologyChanged |= src.nbRows != oldNbRows;
		state.body0WorldOffset = bodyAWorldOffset;
		// Finite material coordinates require a stable bilateral spring program.
		// A limit can insert/reorder rows after prediction; never reuse a C0 by
		// slot across that change. Such programs retain their velocity contract.
		bool finiteSprings = src.nbRows > 0;
		for(PxU32 r = 0; r < src.nbRows; ++r)
			finiteSprings &= (rows[r].flags & Px1DConstraintFlag::eSPRING) != 0 &&
				rows[r].mods.spring.stiffness > 0.0f &&
				rows[r].minImpulse < 0.0f && rows[r].maxImpulse > 0.0f;
		if(!finiteSprings || (!beginStep && src.nbRows != oldNbRows))
			for(PxU32 r = 0; r < AVBD_MAX_JOINT_ROWS; ++r)
				if(state.integratedInitialized[r] == 2) state.integratedInitialized[r] = 0;
		if(beginStep && finiteSprings)
			for(PxU32 r = 0; r < src.nbRows; ++r)
			{
				state.integratedC0[r] = rows[r].geometricError;
				state.integratedInitialized[r] = 2;
			}
		for(PxU32 r = 0; r < src.nbRows; ++r)
			finiteSprings &= state.integratedInitialized[r] == 2;
		// A pure point joint is emitted as exactly three bilateral translation
		// rows.  Carrying its three scalar lambdas in solverPrep's rotating frame
		// rotates the physical multiplier between iterations.  Compile the same
		// three-dimensional row space in a fixed world basis instead.  For the
		// fully locked block ConstraintHelper reports cAtW=2*A-B and cBtW=B,
		// hence their midpoint recovers the true A anchor without inspecting the
		// joint extension's private constant block.
		bool pointLock = src.nbRows == 3;
		for(PxU32 r = 0; r < src.nbRows && pointLock; ++r)
		{
			const Px1DConstraint& pr = rows[r];
			const PxU16 excluded = PxU16(Px1DConstraintFlag::eANGULAR_CONSTRAINT) |
				PxU16(Px1DConstraintFlag::eSPRING);
			pointLock = (pr.flags & excluded) == 0 &&
				pr.minImpulse < 0.0f && pr.maxImpulse > 0.0f &&
				PxAbs(pr.linear0.magnitudeSquared() - 1.0f) < 1.0e-3f &&
				(pr.linear0 - pr.linear1).magnitudeSquared() < 1.0e-6f;
		}
		if(pointLock)
		{
			pointLock = PxAbs(rows[0].linear0.dot(rows[1].linear0)) < 1.0e-3f &&
				PxAbs(rows[0].linear0.dot(rows[2].linear0)) < 1.0e-3f &&
				PxAbs(rows[1].linear0.dot(rows[2].linear0)) < 1.0e-3f;
		}
		if(pointLock && !state.pointInitialized)
		{
			// The first compile occurs at the authored joint pose.  Persist these
			// local anchors; later solverPrep calls deliberately move cAtW's
			// stabilized arm as the violation changes.
			const PxVec3 anchorAWorld = PxVec3(cAtW);
			const PxVec3 anchorBWorld = PxVec3(cBtW);
			state.pointLocalA = src.bodyA < mBodies.size() ?
				mBodies[src.bodyA].q.rotateInv(anchorAWorld - mBodies[src.bodyA].p) :
				anchorAWorld;
			state.pointLocalB = src.bodyB < mBodies.size() ?
				mBodies[src.bodyB].q.rotateInv(anchorBWorld - mBodies[src.bodyB].p) :
				anchorBWorld;
		}
		state.pointLock = PxU8(pointLock);
		state.pointInitialized = 1;
		PxVec3 anchorAWorld = state.pointLocalA;
		PxVec3 anchorBWorld = state.pointLocalB;
		if(pointLock && src.bodyA < mBodies.size())
			anchorAWorld = avbdWorldPoint(mBodies[src.bodyA], state.pointLocalA);
		if(pointLock && src.bodyB < mBodies.size())
			anchorBWorld = avbdWorldPoint(mBodies[src.bodyB], state.pointLocalB);
		for(PxU32 r = 0; r < src.nbRows; ++r)
		{
			const Px1DConstraint& row1d = rows[r];
			AvbdRowBuilder row;
			PxMemZero(&row, sizeof(row));
			row.header().bodyA = src.bodyA;
			row.header().bodyB = src.bodyB;
			row.header().flags = 0;
			if(pointLock)
			{
				const PxVec3 axis = r == 0 ? PxVec3(1.0f, 0.0f, 0.0f) :
					(r == 1 ? PxVec3(0.0f, 1.0f, 0.0f) : PxVec3(0.0f, 0.0f, 1.0f));
				row.scalar().n = axis;
				row.scalar().rA = state.pointLocalA;
				row.scalar().rB = state.pointLocalB;
				row.scalar().jLinA = axis;
				row.scalar().jLinB = -axis;
				row.scalar().jAngA = src.bodyA < mBodies.size() ?
					mBodies[src.bodyA].q.rotate(state.pointLocalA).cross(axis) : PxVec3(0.0f);
				row.scalar().jAngB = src.bodyB < mBodies.size() ?
					mBodies[src.bodyB].q.rotate(state.pointLocalB).cross(-axis) : PxVec3(0.0f);
				row.scalar().C0 = axis.dot(anchorAWorld - anchorBWorld);
				row.scalar().CLin = row.scalar().C0;
				row.header().flags = eAVBD_ROW_POINT_LOCK;
			}
			else
			{
				// PhysX J = {linear0, angular0, -linear1, -angular1}.
				row.scalar().jLinA = row1d.linear0;
				row.scalar().jAngA = row1d.angular0;
				row.scalar().jLinB = -row1d.linear1;
				row.scalar().jAngB = -row1d.angular1;
				const bool angularOnly = row1d.linear0.magnitudeSquared() < 1.0e-12f &&
					row1d.linear1.magnitudeSquared() < 1.0e-12f &&
					(row1d.angular0.magnitudeSquared() +
					 row1d.angular1.magnitudeSquared()) > 1.0e-12f;
				const bool angularCoupling = angularOnly && src.nbRows == 1 &&
					row1d.angular0.magnitudeSquared() > 1.0e-12f &&
					row1d.angular1.magnitudeSquared() > 1.0e-12f &&
					row1d.minImpulse < 0.0f && row1d.maxImpulse > 0.0f &&
					(row1d.flags & Px1DConstraintFlag::eSPRING) == 0;
				row.scalar().integratedAngular = PxU8(angularCoupling);
				if(row.scalar().integratedAngular)
				{
					if(!state.integratedInitialized[r])
					{
						state.integratedC0[r] = row1d.geometricError -
							row1d.velocityTarget * mDt;
						state.integratedInitialized[r] = 1;
					}
					row.scalar().C0 = state.integratedC0[r];
					row.scalar().CLin = row.scalar().C0;
				}
				else
				{
					row.scalar().C0 = row1d.geometricError;
					row.scalar().CLin = row1d.geometricError;
				}
			}
			row.scalar().lambda = state.lambda[r];
			row.scalar().mu = 0.0f;
			// A PhysX impulse acts along +J on body A; the AL coefficient is the
			// energy gradient, so the physical force is -J*coeff and the row
			// bounds flip sign.  A limit (minImpulse 0, error > 0 inside) is
			// then coeff <= 0 like a contact.  Mapping [0, inf) unflipped made
			// every cone/angle limit an unbounded centering spring.
			row.scalar().minLambda = -row1d.maxImpulse;
			row.scalar().maxLambda = -row1d.minImpulse;
			if(row1d.minImpulse >= 0.0f && row1d.maxImpulse > 0.0f)
				row.header().flags |= eAVBD_ROW_UNILATERAL;
			if(row1d.flags & Px1DConstraintFlag::eOUTPUT_FORCE)
				row.header().flags |= eAVBD_ROW_OUTPUT_FORCE;
			if(row1d.flags & Px1DConstraintFlag::eSPRING)
			{
				row.header().flags |= eAVBD_ROW_DRIVE;
				const PxReal k = PxMax(row1d.mods.spring.stiffness, 0.0f);
				const PxReal d = PxMax(row1d.mods.spring.damping, 0.0f);
				row.scalar().driveStiffness = k;
				row.scalar().driveDampingOverDt = mDt > 0.0f ? d / mDt : 0.0f;
				row.scalar().driveVelocityTarget = row1d.velocityTarget * mDt;
				if(finiteSprings)
				{
					row.header().kernelId = eAVBD_ROW_KERNEL_FINITE_SPRING;
					row.scalar().C0 = state.integratedC0[r];
				}
				row.scalar().accelerationDrive = PxU8(
					(row1d.flags & Px1DConstraintFlag::eACCELERATION_SPRING) != 0);
				state.rho[r] = k + row.scalar().driveDampingOverDt;
                if(!mStep.commonProgram || mStep.localBudget)row.scalar().lambda = 0.0f;
				if((row1d.flags & Px1DConstraintFlag::eHAS_DRIVE_LIMIT) &&
					(c.flags & PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES) == 0 &&
					mDt > 0.0f)
				{
					row.scalar().minLambda /= mDt;
					row.scalar().maxLambda /= mDt;
				}
				if(state.rho[r] <= 0.0f)
					row.header().flags |= eAVBD_ROW_INACTIVE;
			}
			row.scalar().rho = pointLock ? AVBD_POINT_LOCK_PENALTY : state.rho[r];
			row.scalar().reactionOutput = NULL;
			mRows.pushBack(row);
		}
	}
	refreshArticJointRows();
	return topologyChanged;
}

// A feasible, fully locked tree rooted in the world has no free modes. Its
// inertial gradient can be balanced exactly at the DCD pose. Seed the ordinary
// scalar duals by spatial wrench balance, so finite GS does not have to learn
// a long cantilever's static load anew from positional drift.
void AvbdDynamicsContext::seedAnchoredJointReactions()
{
    if(mLocalDomainFailed) return;

    const PxU32 first = PxMax(mContactRowCount, mNativeMaterialEnd);
    PxArray<PxU32> offsets(mJoints.size()), heads(mBodies.size(), PX_MAX_U32);
    PxArray<PxU32> next(2 * mJoints.size(), PX_MAX_U32);
    PxArray<PxU8> external(mBodies.size(), 0);
    PxArray<PxU32> contactHead(mBodies.size(), PX_MAX_U32), degree(mBodies.size(), 0);
    PxArray<PxU32> nextContact(2*mNpContactRowCount);
    for(PxU32 r = 0; r < first; ++r)
    {
        const auto row = mRows[r];
        const PxU32 ends[] = {row.header().bodyA,row.header().bodyB};
        for(PxU32 side = 0; side < 2; ++side)
        {
            const PxU32 b = ends[side];
            if(b >= mBodies.size()) continue;
            if(r < mNpContactRowCount && !mStep.commonProgram)
            {
                nextContact[2*r+side] = contactHead[b];
                contactHead[b] = 2*r+side;
                ++degree[b];
            }
            else external[b] = 1;
        }
        for(PxU32 i = 0; i < row.header().supportCount; ++i)
        {
            const PxU32 b = mRowSupports[row.header().supportStart+i].block;
            if(b < mBodies.size()) external[b] = 1;
        }
    }
    PxU32 offset = first;
    for(PxU32 j = 0; j < mJoints.size(); ++j)
    {
        const auto& joint = mJoints[j];
        offsets[j] = offset;
        offset += joint.nbRows;
        const PxU32 ends[] = {joint.bodyA, joint.bodyB};
        for(PxU32 s = 0; s < 2; ++s)
            if(ends[s] < heads.size())
            {
                next[2*j+s] = heads[ends[s]];
                heads[ends[s]] = 2*j+s;
            }
    }
    // Only complete small patches are condensed. An incomplete patch cannot
    // infer its boundary load. Other programs keep their usual initial guess.
    struct Link
    {
        PxU32 body, joint, ancestors;
        PxVec3 inertialForce, inertialMoment, load, moment;
        PxMat33 linearInverse, angularInverse, lever;
        PxVec3 linearDual, angularDual;
    };
    const PxU32 limit = 16;
    for(PxU32 root = 0; root < mJoints.size(); ++root)
    {
        const auto& anchor = mJoints[root];
        if(anchor.nbRows != 6 || (anchor.bodyA == PX_MAX_U32) == (anchor.bodyB == PX_MAX_U32)) continue;
        if(anchor.bodyA==PX_MAX_U32 ? anchor.constraint->bodyCore0!=NULL : anchor.constraint->bodyCore1!=NULL) continue;
        Link links[limit];
        links[0].body = PxMin(anchor.bodyA, anchor.bodyB);
        links[0].joint = root;
        links[0].ancestors = 1;
        struct Hit { PxU32 row, incident; PxReal sign; PxVec3 normal, witness; };
        Hit hits[limit];
        PxU32 count = 1, hitCount = 0, other = PX_MAX_U32;
        PxVec3 axis(0);
        bool complete = true;
        for(PxU32 i = 0; i < count && complete; ++i)
        {
            auto& link = links[i];
            const auto& joint = mJoints[link.joint];
            const auto& body = mBodies[link.body];
            if(joint.nbRows != 6 || external[link.body] || !avbdBodyResponds(body) ||
               body.artic || body.core->lockFlags || joint.constraint->multiBody || joint.constraint->elasticFramePrep ||
               !mJointStates[joint.payloadIndex].passiveLock)
            { complete = false; break; }
            for(PxU32 edge=contactHead[link.body]; edge!=PX_MAX_U32; edge=nextContact[edge])
            {
                const auto row = mRows[edge/2];
                const bool sideA = (edge%2)==0;
                const PxU32 free = sideA ? row.header().bodyB : row.header().bodyA;
                if(hitCount == limit || free >= mBodies.size() || (hitCount && free != other))
                { complete = false; break; }
                other = free;
                const PxReal sign = sideA ? 1.f : -1.f;
                hits[hitCount++] = {edge/2, i, sign, sign*row.scalar().n, body.pStart +
                    body.qStart.rotate(sideA ? row.scalar().ogcRA : row.scalar().ogcRB)};
                axis += sign*row.scalar().n;
            }
            if(!complete) break;
            PxMat33 linear(PxZero), angular(PxZero);
            const bool sideA = joint.bodyA == link.body;
            for(PxU32 r = 0; r < 6; ++r)
            {
                const auto row = mRows[offsets[link.joint]+r];
                if(row.header().flags != eAVBD_ROW_OUTPUT_FORCE || !row.hasScalar() ||
                   row.scalar().minLambda != -PX_MAX_REAL || row.scalar().maxLambda != PX_MAX_REAL ||
                   PxAbs(row.scalar().C0) > 8*PX_EPS_F32*(r < 3 ? 1+body.pStart.abs().maxElement() : 1)) { complete = false; break; }
                const auto& data = row.scalar();
                if(joint.bodyA<mBodies.size() && joint.bodyB<mBodies.size())
                {
                    const PxVec3 distance=mBodies[joint.bodyB].pStart-mBodies[joint.bodyA].pStart;
                    const PxVec3 torque=data.jAngA+data.jAngB+distance.cross(data.jLinB);
                    if(!(data.jLinA+data.jLinB).isZero() || torque.magnitude()>16*PX_EPS_F32*
                        (1+distance.magnitude()+data.jAngA.magnitude()+data.jAngB.magnitude()))
                    { complete=false; break; }
                }
                const PxVec3 translation = sideA ? data.jLinA : data.jLinB;
                const PxVec3 rotation = sideA ? data.jAngA : data.jAngB;
                if(r < 3) { linear[r] = translation; link.lever[r] = rotation; }
                else { if(!translation.isZero()) complete = false; angular[r-3] = rotation; }
            }
            if(!complete || PxAbs(linear.getDeterminant()) < 1.e-6f || PxAbs(angular.getDeterminant()) < 1.e-6f)
            { complete = false; break; }
            link.linearInverse = linear.getInverse();
            link.angularInverse = angular.getInverse();
            link.inertialForce = (body.pHat-body.pStart)*body.massInvH2;
            link.inertialMoment = -avbdRotationInertialPotential(body.qHat, body.qStart, body.inertiaWorldInvH2).gradient;
            for(PxU32 edge = heads[link.body]; edge != PX_MAX_U32; edge = next[edge])
            {
                const PxU32 childJoint = edge/2;
                if(childJoint == link.joint) continue;
                const auto& neighbor = mJoints[childJoint];
                const PxU32 child = edge%2 ? neighbor.bodyA : neighbor.bodyB;
                if(child >= mBodies.size() || count == limit) { complete = false; break; }
                for(PxU32 p = 0; p < count; ++p) if(links[p].body == child) complete = false;
                if(!complete) break;
                links[count].body = child;
                links[count].joint = childJoint;
                links[count].ancestors = link.ancestors | (1u << count);
                ++count;
            }
        }
        if(!complete) continue;
        // Each cut sees the external wrenches on its descendant bodies.
        // All cuts read the same immutable data and can reduce independently.
        for(PxU32 k = 0; k < count; ++k)
        {
            auto& link = links[k];
            link.load = link.inertialForce;
            link.moment = link.inertialMoment;
            for(PxU32 j = k+1; j < count; ++j)
            {
                const auto& child = links[j];
                if(!(child.ancestors & (1u << k))) continue;
                link.load += child.inertialForce;
                link.moment += child.inertialMoment +
                    (mBodies[child.body].pStart-mBodies[link.body].pStart).cross(child.inertialForce);
            }
            link.linearDual = link.linearInverse * link.load;
            link.angularDual = link.angularInverse * (link.moment-link.lever*link.linearDual);
            if(!link.linearDual.isFinite() || !link.angularDual.isFinite()) { complete = false; break; }
        }
        if(!complete) continue;
        if(hitCount)
        {
            if(external[other] || heads[other] != PX_MAX_U32 || degree[other] != hitCount ||
               !avbdBodyResponds(mBodies[other]) || mBodies[other].artic || mBodies[other].core->lockFlags ||
               !axis.normalize()) continue;
            const auto& free = mBodies[other];
            const PxReal mass = 1/free.invMass;
            const PxReal twiceEnergy = mass*free.linVel.magnitudeSquared() +
                mDt*mDt*free.angVel.dot(free.inertiaWorldInvH2*free.angVel);
            const PxReal maxNet = mass*free.linVel.magnitude() + PxSqrt(mass*twiceEnergy);
            // For a common separating direction, every Coulomb impulse has
            // d.J >= projection*|J|. Passivity of the free body then bounds
            // the sum of all point-impulse magnitudes, including friction.
            PxReal projection = 1;
            for(PxU32 h=0; hitCount>1 && h<hitCount; ++h)
            {
                const PxReal cosine = axis.dot(hits[h].normal), mu=mRows[hits[h].row].scalar().mu;
                projection = PxMin(projection,(cosine-mu*PxSqrt(PxMax(0.f,1-cosine*cosine)))/PxSqrt(1+mu*mu));
            }
            if(!(projection>0)) continue;
            const PxReal maxImpulse = maxNet/projection*(1+16*PX_EPS_F32);
            if(!PxIsFinite(maxImpulse)) continue;
            const PxU32 saved = mFixedReactions.size();
            for(PxU32 k=0; k<count; ++k)
            {
                const auto& link = links[k];
                const auto& joint = mJoints[link.joint];
                const auto& c = *joint.constraint;
                const bool childA = joint.bodyA == link.body;
                const PxVec3 actor0 = joint.bodyA < mBodies.size() ? mBodies[joint.bodyA].pStart :
                    bodyPose(joint.bodyA,c.bodyCore0).p;
                const PxVec3 anchorPoint = actor0 + mJointStates[joint.payloadIndex].body0WorldOffset;
                const PxVec3 staticMoment = link.moment + (mBodies[link.body].pStart-anchorPoint).cross(link.load);
                PxReal length = 0;
                bool incident = false;
                for(PxU32 h=0; h<hitCount; ++h)
                {
                    if(!(links[hits[h].incident].ancestors & (1u << k))) continue;
                    const PxVec3 arm=hits[h].witness-anchorPoint;
                    length=PxMax(length,arm.magnitude());
                    incident=true;
                    mFixedReactions.pushBack({hits[h].row,c.index,(childA?-1.f:1.f)*hits[h].sign,arm});
                }
                if(link.load.magnitude()*mDt + (incident?maxImpulse:0) >= c.linBreakForce*mDt ||
                   staticMoment.magnitude()*mDt + length*maxImpulse >= c.angBreakForce*mDt)
                { complete=false; break; }
            }
            if(!complete) { mFixedReactions.forceSize_Unsafe(saved); continue; }
        }
        for(PxU32 i = 0; i < count; ++i)
        {
            const auto& link = links[i];
            auto& body = mBodies[link.body];
            body.p = body.pStart;
            body.q = body.qStart;
            if(hitCount)
            {
                // An exactly grounded six-axis tree has zero contact mobility.
                // Keep its Host mass and inertial target for force recovery.
                body.invMass = 0;
                body.invInertia = PxVec3(0);
                body.outputLinVel = body.outputAngVel = PxVec3(0);
                body.linVel = body.angVel = PxVec3(0);
            }
            auto& state = mJointStates[mJoints[link.joint].payloadIndex];
            for(PxU32 r = 0; r < 3; ++r)
            {
                mRows[offsets[link.joint]+r].scalar().lambda = state.lambda[r] = link.linearDual[r];
                mRows[offsets[link.joint]+3+r].scalar().lambda = state.lambda[3+r] = link.angularDual[r];
            }
        }
    }
}
void AvbdDynamicsContext::writeJointDualState()
{
	PxU32 cursor = PxMax(mContactRowCount,mNativeMaterialEnd);
	for(PxU32 j = 0; j < mJoints.size(); ++j)
	{
		AvbdJointSrc& src = mJoints[j];
		if(src.constraint->multiBody || src.constraint->elasticFramePrep) { cursor+=src.nbRows;continue; }
		AvbdJointState& state=mJointStates[src.payloadIndex];
		for(PxU32 r = 0; r < src.nbRows; ++r)
		{
			if(cursor >= mRows.size())
				return;
			const AvbdRowConstRef& row = mRows[cursor++];
			state.lambda[r] = row.scalar().lambda;
			state.rho[r] = row.scalar().rho;
		}
	}
	for(PxU32 j = 0; j < mArticJoints.size(); ++j)
	{
		AvbdArticJointSrc& src = mArticJoints[j];
		for(PxU32 r = 0; r < src.nbRows; ++r)
		{
			if(cursor >= mRows.size())
				return;
			const AvbdRowConstRef& row = mRows[cursor++];
			src.lambda[r] = row.scalar().lambda;
			src.rho[r] = row.scalar().rho;
		}
	}
}

void AvbdDynamicsContext::writeJointConstraintWriteback()
{
	// Host Sc::Scene::checkConstraintBreakage reads this pool.  PGS
	// writeBack1D stores impulses (force * dt) and compares magnitude to
	// linBreakForce * dt.  AVBD lambda is force, so impulse = lambda * dt.
	// Same row program: OUTPUT_FORCE 1D rows + body0WorldOffset, not a
	// second IR or owner enum.
	Cm::PinnableArray<ConstraintWriteback>& pool = getConstraintWriteBackPool();
	const PxReal dt = mDt;
	PxU32 cursor = PxMax(mContactRowCount,mNativeMaterialEnd);
	for(PxU32 j = 0; j < mJoints.size(); ++j)
	{
		const AvbdJointSrc& src = mJoints[j];
		PxVec3 lin(0.0f);
		PxVec3 ang(0.0f);
		for(PxU32 r = 0; r < src.nbRows; ++r)
		{
			if(cursor >= mRows.size())
				return;
			const AvbdRowConstRef& row = mRows[cursor++];
			if(row.header().flags & eAVBD_ROW_OUTPUT_FORCE)
			{
				PxReal force = row.scalar().lambda;
				if((!mStep.commonProgram || mStep.localBudget) && (row.header().flags & eAVBD_ROW_DRIVE))
				{
					PxReal penalty = row.scalar().rho;
					if(row.scalar().accelerationDrive)
					{
						PxReal mobility = 0.0f;
						if(row.header().bodyA < mBodies.size())
						{
							const AvbdSolverBody& b = mBodies[row.header().bodyA];
							const PxVec3 ja = b.qStart.rotateInv(row.scalar().jAngA);
							mobility += b.invMass * row.scalar().jLinA.magnitudeSquared() +
								ja.x*ja.x*b.invInertia.x + ja.y*ja.y*b.invInertia.y +
								ja.z*ja.z*b.invInertia.z;
						}
						if(row.header().bodyB < mBodies.size())
						{
							const AvbdSolverBody& b = mBodies[row.header().bodyB];
							const PxVec3 jb = b.qStart.rotateInv(row.scalar().jAngB);
							mobility += b.invMass * row.scalar().jLinB.magnitudeSquared() +
								jb.x*jb.x*b.invInertia.x + jb.y*jb.y*b.invInertia.y +
								jb.z*jb.z*b.invInertia.z;
						}
						if(mobility > 1.0e-12f && PxIsFinite(mobility))
							penalty /= mobility;
					}
					force = PxMin(row.scalar().maxLambda, PxMax(row.scalar().minLambda,
						penalty * avbdEvalC(row, mBodies)));
				}
				// PGS writeBack1D reports the impulse along +J; the AL
				// coefficient is the gradient, so negate.
				const PxReal impulse = -force * dt;
				lin += row.scalar().jLinA * impulse;
				ang += row.scalar().jAngA * impulse;
			}
		}
		if(!src.constraint->multiBody && !src.constraint->elasticFramePrep)
			ang -= mJointStates[src.payloadIndex].body0WorldOffset.cross(lin);
		if(!src.constraint)
			continue;
		const PxU32 id = src.constraint->index;
		if(id >= pool.size())
			continue;
		ConstraintWriteback& writeback = pool[id];
		writeback.linearImpulse = lin;
		writeback.angularImpulse = ang;
		const bool linBreakable = src.constraint->linBreakForce < PX_MAX_F32;
		const bool angBreakable = src.constraint->angBreakForce < PX_MAX_F32;
		writeback.broken =
			((linBreakable && lin.magnitude() > src.constraint->linBreakForce * dt) ||
			 (angBreakable && ang.magnitude() > src.constraint->angBreakForce * dt))
				? 1u : 0u;
        if(writeback.broken)
        {
            if(src.bodyA<mBodies.size()) mReleasedRigidBodies.pushBack(src.bodyA);
            if(src.bodyB<mBodies.size()) mReleasedRigidBodies.pushBack(src.bodyB);
            for(PxU32 r = cursor-src.nbRows; r < cursor; ++r)
                mRows[r].header().flags |= eAVBD_ROW_INACTIVE;
        }
	}
}
