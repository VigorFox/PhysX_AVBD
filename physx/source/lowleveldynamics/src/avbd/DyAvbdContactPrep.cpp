// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/DyAvbdInternal.h"
#include "foundation/PxHash.h"
#include "avbd/DyAvbdOgc.h"
#include "avbd/DyAvbdContactWarmIndex.h"

#include "DyIslandManager.h"
#include "DyDeformableSurface.h"
#include "PxcNpWorkUnit.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsContactGeometry.h"
#include "PxsIslandSim.h"
#include "PxsSimpleIslandManager.h"
#include "PxContact.h"
#include "geometry/PxTetrahedronMesh.h"
#include <cstdlib>
#include <cstdio>
#include <functional>

using namespace physx;
using namespace Dy;

static bool avbdCompileNpShape(AvbdNpDomain& domain,PxU32 side,
    const PxsContactShapeGeometry& shape,PxU32 body)
{
    if(shape.kind==PxsContactShapeGeometry::eNONE ||
        (shape.kind==PxsContactShapeGeometry::ePLANE && body!=PX_MAX_U32)) return false;
    domain.radius[side]=shape.radius;
    domain.fixedAxis=domain.fixedAxis || shape.kind==PxsContactShapeGeometry::ePLANE;
    domain.fixedPose[side]=shape.shapeToWorld;
    const PxTransform& local=shape.shapeToBody;
    const PxQuatd q=avbdNpRotation(local.q);
    const PxVec3d p=avbdOgcDouble(local.p);
    const PxU32 count=shape.kind==PxsContactShapeGeometry::ePLANE?1:shape.count;
    for(PxU32 i=0;i<count;++i)
    {
        PxVec3d v(0);
        if(shape.kind==PxsContactShapeGeometry::eVERTICES)
        {
            const PxVec3d vertex=avbdOgcDouble(shape.vertices[i]);
            v=avbdOgcDouble(shape.vertexToShape.column0)*vertex.x+
                avbdOgcDouble(shape.vertexToShape.column1)*vertex.y+
                avbdOgcDouble(shape.vertexToShape.column2)*vertex.z;
        }
        else if(shape.kind==PxsContactShapeGeometry::eBOX)
            v=PxVec3d((i&1)?shape.halfExtents.x:-shape.halfExtents.x,
                (i&2)?shape.halfExtents.y:-shape.halfExtents.y,
                (i&4)?shape.halfExtents.z:-shape.halfExtents.z);
        else if(shape.kind==PxsContactShapeGeometry::eCAPSULE)
            v.x=i?shape.halfExtents.x:-shape.halfExtents.x;
        domain.points[side].pushBack(body==PX_MAX_U32?v:p+q.rotate(v));
    }
    return count!=0;
}

void AvbdDynamicsContext::buildContactRows()
{
	AvbdBodyIndex bodyIndices;bodyIndices.build(mBodies);
	PxArray<AvbdContactWarm> prev;
	prev.swap(mContactWarm);
    const AvbdContactWarmIndex warmIndex(prev);
    PxU64 warmCandidates=0,warmGlobalCandidates=0;
	PxArray<PxU8> prevUsed(prev.size());
	for(PxU32 i = 0; i < prevUsed.size(); ++i)
		prevUsed[i] = 0;
	mRows.truncate(0);
    mNpImpulseOutputs.clear();
	mContactPlaneGeometry.clear();
	mContactFrameGeometry.clear();
	mNpDomains.clear();
	mNpDomainUnsupported=mNpDomainPolicyExcluded=mNpDomainUnseparated=0;
	mNpDomainAxisRecovered=0;
	const char* domainMode=std::getenv("PHYSX_NP_GEOMETRY_CERTIFICATES");
	mNpDomainMode=mStep.hasParticleBlocks || (domainMode && domainMode[0]=='2')?2u:(domainMode && domainMode[0]=='1'?1u:0u);
	const PxU32 nb = mContactList.size();
	PxU32 jointRows=0;
    for(const auto& src:mJoints)
        jointRows+=(src.constraint->elasticFramePrep || src.constraint->multiBody)?1:AVBD_MAX_JOINT_ROWS;
	mRows.reserve(nb * 4 + jointRows
		+ mArticJoints.size() * AVBD_MAX_ARTIC_ROWS + 64);
	mContactWarm.reserve(nb * 4 + 16);

	for(PxU32 i = 0; i < nb; ++i)
	{
		const PxsIndexedContactManager& cm = mContactList[i];
		if(!cm.contactManager)
			continue;

		PxcNpWorkUnit& unit = cm.contactManager->getWorkUnit();
		if((unit.mFlags & PxcNpWorkUnitFlag::eSOFT_BODY) || !unit.mRigidCore0 || !unit.mRigidCore1)
			continue;
		PxsContactManagerOutput& output =
			mOutputIterator.getContactManagerOutput(unit.mNpIndex);
        // Bind the host report for the final accepted step.
        if(output.contactForces)
            mNpImpulseOutputs.pushBack({output.contactForces,output.nbContacts});
		if(!output.contactPatches || !output.contactPoints || output.nbContacts == 0)
			continue;
		// NP reports every proximity point inside the inertial query distance
		// (contact offset + one-step reach).  Each point is a speculative
		// unilateral Position-AL row, the same contract PhysX uses for
		// speculative contacts: it carries no force while C > 0 and stops the
		// pair on the linearized plane when the step would cross it.

		const PxU32 ia = contactBody(cm, 0,bodyIndices);
		const PxU32 ib = contactBody(cm, 1,bodyIndices);
		if(ia == PX_MAX_U32 && ib == PX_MAX_U32)
			continue;

		// Iterator, not raw PxContact*: modifiable/extended streams have a
		// different stride. targetVelocity and inv-mass scale are intentionally
		// unread (documented unsupported; smoke PASS unsupported).
		PxContactStreamIterator iter(output.contactPatches, output.contactPoints,
			output.getInternalFaceIndice(), output.nbPatches, output.nbContacts);
		PxU32 streamIndex = 0;
		const PxU32 firstRow=mRows.size();
		bool domainPolicy=(unit.mFlags & (PxcNpWorkUnitFlag::eMODIFIABLE_CONTACT |
			PxcNpWorkUnitFlag::eDISABLE_RESPONSE))==0 && unit.mRestDistance>=0;
		while(iter.hasNextPatch())
		{
			iter.nextPatch();
			if(iter.patch && (iter.patch->internalFlags & PxContactPatch::eFORCE_NO_RESPONSE))
			{
				domainPolicy=false;
				while(iter.hasNextContact())
				{
					iter.nextContact();
					++streamIndex;
				}
				continue;
			}
			PxVec3 n = iter.getContactNormal();
			if(n.normalize() < 1.0e-8f)
			{
				while(iter.hasNextContact())
				{
					iter.nextContact();
					++streamIndex;
				}
				continue;
			}
			PxVec3 t0, t1;
			avbdOrthonormalBasis(n, t0, t1);
			const PxReal mu = PxMax(iter.getStaticFriction(), iter.getDynamicFriction());
			const PxReal rest = PxClamp(iter.getRestitution(), 0.0f, 1.0f);
			const PxU32 materialKey = PxU32(iter.patch->materialIndex0) |
				(PxU32(iter.patch->materialIndex1) << 16);

			while(iter.hasNextContact())
			{
				iter.nextContact();
				const PxU32 idx = streamIndex++;
				PxReal maxImp = iter.getMaxImpulse();
				// Match PhysX contact preparation: the per-point modifier and
				// both public body limits restrict the same normal impulse.
				if(ia<mBodies.size() && mBodies[ia].core)
					maxImp=PxMin(maxImp,mBodies[ia].core->maxContactImpulse);
				if(ib<mBodies.size() && mBodies[ib].core)
					maxImp=PxMin(maxImp,mBodies[ib].core->maxContactImpulse);
				if(maxImp<1.0e15f) domainPolicy=false;
				if(maxImp <= 0.0f)
					continue;
				// C = separation - restDistance (AVBD_2 processedSeparation).
				const PxReal d = iter.getSeparation();

				AvbdRowBuilder row;
				PxMemZero(&row, sizeof(row));
				row.header().bodyA = ia;
				row.header().bodyB = ib;
				row.scalar().n = n;
				row.scalar().t0 = t0;
				row.scalar().t1 = t1;
				// The contact AL Hessian starts at the dimensionless floor; a stiff
				// initial penalty made a contact row one hundred times stiffer than
				// the reference hard-joint block and allowed a heavy projectile to
				// stretch a spherical-joint graph.
				row.scalar().rho = AVBD_CONTACT_PENALTY_MIN;
				row.scalar().rhoT0 = AVBD_CONTACT_PENALTY_MIN;
				row.scalar().rhoT1 = AVBD_CONTACT_PENALTY_MIN;
				row.scalar().lambda = 0.0f;
				row.scalar().lambdaT0 = 0.0f;
				row.scalar().lambdaT1 = 0.0f;
				row.scalar().mu = mu;
				// Passive Coulomb tangents remain Position-AL coordinates even when
				// an endpoint belongs to a hard-joint/articulation program.  Endpoint
				// velocity ownership is compiled later; it must not erase geometry's
				// two tangential coordinates.
				row.scalar().positionTangent = 1;
				row.scalar().restitution = rest;
				row.scalar().minLambda = -PX_MAX_REAL;
				if(maxImp < 1.0e15f && mDt > 0.0f)
					row.scalar().minLambda = -maxImp / mDt;
				row.scalar().maxLambda = 0.0f;
				row.header().flags = PxU8(eAVBD_ROW_UNILATERAL | eAVBD_ROW_CONTACT);
				row.scalar().reactionOutput =
					output.contactForces ? output.contactForces + idx : NULL;

				const PxVec3 point = iter.getContactPoint();
				if(ia != PX_MAX_U32)
					row.scalar().rA = mBodies[ia].q.rotateInv(point - mBodies[ia].p);
				else
					row.scalar().rA = point;

				if(ib != PX_MAX_U32)
					row.scalar().rB = mBodies[ib].q.rotateInv(point - mBodies[ib].p);
				else
					row.scalar().rB = point;
				row.scalar().ogcRA = row.scalar().rA;
				row.scalar().ogcRB = row.scalar().rB;

				// Default rigid NP rows rest at the authored rest distance.
				// The explicit complete-domain trial may compile an offset law
				// below after checking the entire pair's response policy.
				const PxReal rawInitialC0 = d - unit.mRestDistance;
				row.scalar().contactOffset = rawInitialC0;
				PxVec3 normalPoints[2];
				if(output.geometry && output.geometry->normalWitnesses(point,n,d,normalPoints))
				{
					// Separate surface witnesses have the same initial gap and
					// normal Jacobian as the common point, but rotate correctly.
					row.scalar().ogcRA=ia!=PX_MAX_U32?mBodies[ia].q.rotateInv(normalPoints[0]-mBodies[ia].p):normalPoints[0];
					row.scalar().ogcRB=ib!=PX_MAX_U32?mBodies[ib].q.rotateInv(normalPoints[1]-mBodies[ib].p):normalPoints[1];
					row.scalar().contactOffset=-unit.mRestDistance;
				}
				row.scalar().contactRadius = 0.0f;
				if(output.geometry && !(unit.mFlags & PxcNpWorkUnitFlag::eMODIFIABLE_CONTACT))
				{
					const PxU32 ids[2] = {ia, ib};
					for(PxU32 side=0; side<2; ++side)
					{
						const auto& shape = output.geometry->shapes[side];
						if(ids[side]>=mBodies.size() || shape.kind!=PxsContactShapeGeometry::eSPHERE)
							continue;
						// Keep the measured DCD separation. Only the offset of the
						// sphere center from the COM can give a normal force a torque.
						PxVec3& arm = side ? row.scalar().ogcRB : row.scalar().ogcRA;
						const PxVec3 center = shape.shapeToBody.p;
						const PxReal radial = n.dot(mBodies[ids[side]].q.rotate(arm-center));
						row.scalar().contactOffset += side ? -radial : radial;
						arm = center;
						row.scalar().sphereSupports |= PxU8(1u << side);
					}
				}
				row.scalar().contactStiffness = 0.0f;
				if(ia == PX_MAX_U32 || ib == PX_MAX_U32)
				{
					row.scalar().rho *= AVBD_PEN_SCALE_BODY_VS_STATIC;
					row.scalar().rhoT0 *= AVBD_PEN_SCALE_BODY_VS_STATIC;
					row.scalar().rhoT1 *= AVBD_PEN_SCALE_BODY_VS_STATIC;
				}
				bool persistentPointMatched = false;
				PxU32 matchedWarm = PX_MAX_U32;
				PxReal matchedScore = PX_MAX_F32;
                warmGlobalCandidates+=prev.size();
				for(PxU32 w=warmIndex.first(cm.contactManager);w!=PX_MAX_U32;w=warmIndex.next[w])
				{
                    ++warmCandidates;
					if(prevUsed[w] || prev[w].cm != cm.contactManager ||
						prev[w].materialKey != materialKey)
						continue;
					// AVBD_2 persistence gate (matchAvbdPersistentManifold):
					// dual state only survives an exact geometric match of the
					// preceding frame's point. Cached anchors are body-local,
					// re-expressed at the current pose, so persistent contacts
					// on moving bodies still match while a reshuffled manifold
					// does not. Stream-index identity alone re-applies an
					// impact-scale lambda to a different point: a stale 1e6 N
					// force ejects a resting pair at lambda*h/M tens of m/s
					// and the impulse cascades through the pile.
					const PxVec3 oldWorldA = (ia != PX_MAX_U32)
						? mBodies[ia].p + mBodies[ia].q.rotate(prev[w].detectionRA)
						: prev[w].detectionRA;
					const PxVec3 oldWorldB = (ib != PX_MAX_U32)
						? mBodies[ib].p + mBodies[ib].q.rotate(prev[w].detectionRB)
						: prev[w].detectionRB;
					const PxReal score = 0.5f *
						((oldWorldA - point).magnitudeSquared() +
						 (oldWorldB - point).magnitudeSquared());
					const PxReal corr =
						0.025f * PxMax(getLengthScale(), 1.0e-6f);
					const PxReal normalGate = 0.9659258f; // cos(15 deg)
					if(prev[w].n.dot(row.scalar().n) < normalGate ||
						!PxIsFinite(score) || score > corr * corr ||
						score >= matchedScore)
						continue;
					matchedScore = score;
					matchedWarm = w;
				}
				if(matchedWarm != PX_MAX_U32)
				{
					const PxU32 w = matchedWarm;
					prevUsed[w] = 1;
					const PxReal warm = AVBD_ALPHA * AVBD_GAMMA;
					row.scalar().rho = PxMax(row.scalar().rho, PxClamp(prev[w].rho * AVBD_GAMMA,
						AVBD_CONTACT_PENALTY_MIN, AVBD_CONTACT_PENALTY_MAX));
					row.scalar().rhoT0 = PxMax(row.scalar().rhoT0, PxClamp(prev[w].rhoT0 * AVBD_GAMMA,
						AVBD_CONTACT_PENALTY_MIN, AVBD_CONTACT_PENALTY_MAX));
					row.scalar().rhoT1 = PxMax(row.scalar().rhoT1, PxClamp(prev[w].rhoT1 * AVBD_GAMMA,
						AVBD_CONTACT_PENALTY_MIN, AVBD_CONTACT_PENALTY_MAX));
					// A sticking material point owns one inseparable tuple: both local
					// anchors and their normal offset.  Matching uses the fresh detection
					// anchors above; retaining only the dual while rebasing this tuple at
					// every NP refresh erases the elastic displacement stored by Position-AL.
					if(prev[w].stick && mu > 0.0f)
					{
						const PxVec3 oldA = ia != PX_MAX_U32 ?
							mBodies[ia].p + mBodies[ia].q.rotate(prev[w].rA) : prev[w].rA;
						const PxVec3 oldB = ib != PX_MAX_U32 ?
							mBodies[ib].p + mBodies[ib].q.rotate(prev[w].rB) : prev[w].rB;
						const PxVec3 oldDelta = oldA - oldB;
						const PxVec3 tangentDrift = oldDelta - row.scalar().n * oldDelta.dot(row.scalar().n);
						const PxReal corr = 0.025f * PxMax(getLengthScale(), 1.0e-6f);
						if(prev[w].n.dot(row.scalar().n) >= 0.999f &&
							tangentDrift.magnitudeSquared() <= corr * corr)
						{
							row.scalar().rA = prev[w].rA;
							row.scalar().rB = prev[w].rB;
						}
					}
					// demo3d Eq. 19: lambda <- alpha*gamma*lambda through the
					// geometric persistence gate only.
					row.scalar().lambda = PxMin(0.0f, prev[w].lambda * warm);
					const PxVec3 ft = prev[w].t0 * prev[w].lambdaT0
						+ prev[w].t1 * prev[w].lambdaT1;
					row.scalar().lambdaT0 = row.scalar().t0.dot(ft) * warm;
					row.scalar().lambdaT1 = row.scalar().t1.dot(ft) * warm;
					avbdProjectCoulomb(row.scalar().lambda, row.scalar().mu, row.scalar().lambdaT0, row.scalar().lambdaT1);
					persistentPointMatched = true;
				}
				if(!persistentPointMatched)
					row.header().flags = PxU8(row.header().flags | eAVBD_ROW_FRESH);
				// C0 is the signed solve-start continuation.  Clamping a negative
				// value to zero reclassifies every persistent overlap as displacement
				// created by this step; the pose solve then repeats full depenetration
				// and writes that geometric recovery back as impact velocity.  Match
				// the reference contract: retain signed C0, with only a continuous
				// attenuation for a fresh, pathologically deep point.
				PxReal conditionedC0 = rawInitialC0;
				if(!persistentPointMatched)
				{
					const PxReal lengthScale = PxMax(getLengthScale(), 1.0e-6f);
					const PxReal threshold = 0.05f * lengthScale;
					const PxReal maxDepth = 0.20f * lengthScale;
					if(conditionedC0 < -threshold)
					{
						const PxReal t = PxClamp((maxDepth + conditionedC0) /
							(maxDepth - threshold), 0.0f, 1.0f);
						conditionedC0 *= t;
					}
				}
				row.scalar().C0 = conditionedC0;
				PxVec3 initialA = row.scalar().rA;
				PxVec3 initialB = row.scalar().rB;
				if(ia != PX_MAX_U32)
					initialA = mBodies[ia].pStart + mBodies[ia].qStart.rotate(row.scalar().rA);
				if(ib != PX_MAX_U32)
					initialB = mBodies[ib].pStart + mBodies[ib].qStart.rotate(row.scalar().rB);
				const PxVec3 initialDelta = initialA - initialB;
				row.scalar().tangentC0 = initialDelta.dot(row.scalar().t0);
				row.scalar().tangentC1 = initialDelta.dot(row.scalar().t1);
				AvbdContactWarm warm;
				warm.cm = cm.contactManager;
				warm.materialKey = materialKey;
				warm.n = row.scalar().n;
				warm.detectionRA = ia != PX_MAX_U32 ?
					mBodies[ia].q.rotateInv(point - mBodies[ia].p) : point;
				warm.detectionRB = ib != PX_MAX_U32 ?
					mBodies[ib].q.rotateInv(point - mBodies[ib].p) : point;
				warm.rA = row.scalar().rA;
				warm.rB = row.scalar().rB;
				warm.t0 = row.scalar().t0;
				warm.t1 = row.scalar().t1;
				warm.lambda = row.scalar().lambda;
				warm.lambdaT0 = row.scalar().lambdaT0;
				warm.lambdaT1 = row.scalar().lambdaT1;
				warm.rho = row.scalar().rho;
				warm.rhoT0 = row.scalar().rhoT0;
				warm.rhoT1 = row.scalar().rhoT1;
				warm.contactOffset = row.scalar().contactOffset;
				warm.stick = 0;
				mContactWarm.pushBack(warm);
				mRows.pushBack(row);

				if(ia != PX_MAX_U32)
				{
					mBodies[ia].hasContact = 1;
					if(ib == PX_MAX_U32)
						mBodies[ia].hasStaticTouch = 1;
				}
				if(ib != PX_MAX_U32)
				{
					mBodies[ib].hasContact = 1;
					if(ia == PX_MAX_U32)
						mBodies[ib].hasStaticTouch = 1;
				}
			}
		}
		if(!mNpDomainMode) continue;
		if(!domainPolicy) { ++mNpDomainPolicyExcluded;continue; }
		if(firstRow==mRows.size()) continue;
		AvbdNpDomain domain;domain.restDistance=unit.mRestDistance;
		if(!output.geometry ||
			!avbdCompileNpShape(domain,0,output.geometry->shapes[0],ia) ||
			!avbdCompileNpShape(domain,1,output.geometry->shapes[1],ib))
		{ ++mNpDomainUnsupported;continue; }
		const PxTransform poses[2]={ia<mBodies.size()?PxTransform(mBodies[ia].p,mBodies[ia].q):domain.fixedPose[0],
			ib<mBodies.size()?PxTransform(mBodies[ib].p,mBodies[ib].q):domain.fixedPose[1]};
		double best=-DBL_MAX;PxVec3d axis(0);
		for(PxU32 r=firstRow;r<mRows.size();++r)
		{
			domain.normal=avbdOgcDouble(mRows[r].scalar().n).getNormalized();
			for(PxU32 s=0;s<2;++s) if(output.geometry->shapes[s].kind==PxsContactShapeGeometry::ePLANE)
				domain.normal=avbdNpRotation(domain.fixedPose[s].q).rotate(PxVec3d(s?1.0:-1.0,0,0));
			const double gap=avbdNpGap(domain,poses);
			if(gap>best) { best=gap;axis=domain.normal; }
		}
		const bool witnessDirectionSeparated=best>=0;
		if(output.geometry->axisValid && output.geometry->separatingAxis.magnitudeSquared()>0)
		{
			domain.normal=output.geometry->separatingAxis.getNormalized();
			const double gap=avbdNpGap(domain,poses);
			if(gap>best) { best=gap;axis=domain.normal; }
		}
		// Float GJK may lose a separating direction at a nearly touching
		// convex pair. The last publication's double proof axis is only a
		// candidate: test every current support again, without a gap tolerance.
		if(mStep.localBudget && !domain.fixedAxis)
			for(PxU32 w=warmIndex.first(cm.contactManager);w!=PX_MAX_U32;w=warmIndex.next[w])
				if(prev[w].proofAxis.isFinite() && prev[w].proofAxis.magnitudeSquared()>0)
				{
					domain.normal=prev[w].proofAxis;
					const double gap=avbdNpGap(domain,poses);
					if(gap>best){best=gap;axis=domain.normal;}
				}
		domain.normal=axis;
        bool separated=avbdNpDomainPath(domain,poses,poses);
        if(!separated && mStep.localBudget && avbdNpRecoverInitialAxis(domain,poses))
        {
            separated=true;
            best=avbdNpGap(domain,poses);
        }
        if(!separated)
		{
			++mNpDomainUnseparated;
			if(const char* path=std::getenv("PHYSX_AVBD_COMMON_FAILED_NP"))
			if(FILE* file=std::fopen(path,"wb"))
			{
				std::fprintf(file,"{\"reason\":\"initial_axis_unseparated\",\"gap\":%.17g,\"axis\":[%.17g,%.17g,%.17g],\"sides\":[",best,axis.x,axis.y,axis.z);
				for(PxU32 s=0;s<2;++s)
				{
					std::fprintf(file,"%s{\"kind\":%u,\"radius\":%.17g,\"points\":[",s?",":"",PxU32(output.geometry->shapes[s].kind),domain.radius[s]);
					for(PxU32 vertex=0;vertex<domain.points[s].size();++vertex)
					{
						const PxVec3d v=avbdOgcDouble(poses[s].p)+avbdNpRotation(poses[s].q).rotate(domain.points[s][vertex]);
						std::fprintf(file,"%s[%.17g,%.17g,%.17g]",vertex?",":"",v.x,v.y,v.z);
					}
					std::fprintf(file,"]}");
				}
				std::fprintf(file,"]}");std::fclose(file);
			}
			continue;
		}
		if(!witnessDirectionSeparated && best>=0) ++mNpDomainAxisRecovered;
		mNpDomains.pushBack(domain);
		mRows[firstRow].header().kernelId=eAVBD_ROW_KERNEL_NP_CONTACT;
		mRows[firstRow].header().geometryIndex=mNpDomains.size();
		if(mStep.commonProgram && !mStep.localBudget)
		{
			// Replace the reduced witness manifold with the complete normal
			// cone. Each vertex inequality has one force; no second contact pass.
			PxArray<AvbdRowBuilder> source;PxArray<AvbdContactWarm> warmSource;
			bool uniform=true;
			for(PxU32 r=firstRow;r<mRows.size();++r)
			{
				source.pushBack(mRows[r]);warmSource.pushBack(mContactWarm[r]);
				uniform=uniform && mRows[r].scalar().mu==mRows[firstRow].scalar().mu &&
					mContactWarm[r].materialKey==mContactWarm[firstRow].materialKey;
			}
			if(!uniform){++mNpDomainUnsupported;continue;}
			auto& complete=mNpDomains.back();
			// Kinematics may rotate to a prescribed endpoint this step too.
			const bool rotates[2]={ia<mBodies.size(),ib<mBodies.size()};
			avbdNpCompileFeatures(complete,poses,rotates,firstRow);
			mRows.truncate(firstRow);mContactWarm.resize(firstRow);
			for(PxU32 f=0;f<complete.count[0]*complete.count[1];++f)
			{
				PxVec3d arms[2];const double gap=avbdNpFeatureGradient(complete,poses,f,arms);
				const PxVec3d a=avbdOgcDouble(poses[0].p)+arms[0]-axis*domain.radius[0];
				const PxVec3d b=avbdOgcDouble(poses[1].p)+arms[1]+axis*domain.radius[1];
				const PxVec3d location=rotates[0] && !rotates[1]?a:!rotates[0] && rotates[1]?b:(a+b)*.5;
				const PxVec3 point(PxReal(location.x),PxReal(location.y),PxReal(location.z));
				PxU32 nearest=0;double distance=DBL_MAX;
				for(PxU32 r=0;r<source.size();++r)
				{
					const PxVec3 anchor=ia<mBodies.size()?poses[0].transform(source[r].scalar().rA):source[r].scalar().rA;
					const double d=(avbdOgcDouble(anchor)-location).magnitudeSquared();
					if(d<distance){distance=d;nearest=r;}
				}
				AvbdRowBuilder row=source[nearest];AvbdContactWarm warm=warmSource[nearest];
				row.scalar().sphereSupports=0; // regenerated feature anchors below
				row.header().kernelId=eAVBD_ROW_KERNEL_NP_CONTACT;row.header().geometryIndex=mNpDomains.size();
				row.scalar().n=PxVec3(PxReal(axis.x),PxReal(axis.y),PxReal(axis.z));avbdOrthonormalBasis(row.scalar().n,row.scalar().t0,row.scalar().t1);
				row.scalar().rA=ia<mBodies.size()?poses[0].transformInv(point):point;
				row.scalar().rB=ib<mBodies.size()?poses[1].transformInv(point):point;
				row.scalar().ogcRA=row.scalar().rA;row.scalar().ogcRB=row.scalar().rB;row.scalar().C0=row.scalar().contactOffset=PxReal(gap);
				// Do not transfer reduced-manifold multipliers to new feature IDs.
				row.scalar().lambda=row.scalar().lambdaT0=row.scalar().lambdaT1=0;
				warm.n=row.scalar().n;warm.rA=row.scalar().rA;warm.rB=row.scalar().rB;warm.t0=row.scalar().t0;warm.t1=row.scalar().t1;
				warm.lambda=warm.lambdaT0=warm.lambdaT1=0;
				for(PxU32 k=1;k<=5;++k)
				{
					row.scalar().C0=row.scalar().contactOffset=PxReal(gap*5/double(k));
					mRows.pushBack(row);mContactWarm.pushBack(warm);
				}
			}
		}
		if(mNpDomainMode==2)
			for(PxU32 r=firstRow;r<mRows.size();++r)
			{
				// Replace, never add to, the same NP normal's Position-AL law.
				// Use the authored shell, not NP's velocity-inflated query reach.
				mRows[r].scalar().contactRadius=PxMax(0.0f,output.geometry->contactDistance-unit.mRestDistance);
				mRows[r].scalar().contactStiffness=mOgc.contactStiffness;
			}
	}
	mContactRowCount = mRows.size();
    if(mStep.commonProgram && std::getenv("PHYSX_AVBD_COMMON_STATS"))
    {
        PxU64 pairs=0,largest=0;PxU32 side0=0,side1=0;
        for(const auto& domain:mNpDomains)if(domain.rowStart!=PX_MAX_U32)
        {const PxU64 count=PxU64(domain.count[0])*domain.count[1];pairs+=count;
            if(count>largest){largest=count;side0=domain.count[0];side1=domain.count[1];}}
        std::fprintf(stderr,"[AVBD_NP_CARDINALITY] domains=%u pairs=%llu largest=%llu side0=%u side1=%u rows=%u\n",
            mNpDomains.size(),static_cast<unsigned long long>(pairs),static_cast<unsigned long long>(largest),side0,side1,mContactRowCount);
        std::fprintf(stderr,"[AVBD_CONTACT_HISTORY] previous=%u indexedCandidates=%llu globalScanCandidates=%llu indexBytes=%llu\n",
            prev.size(),static_cast<unsigned long long>(warmCandidates),static_cast<unsigned long long>(warmGlobalCandidates),
            static_cast<unsigned long long>(PxU64(warmIndex.next.size())*sizeof(PxU32)));
    }

}

void AvbdDynamicsContext::compileNodalContactRows()
{
#if !PX_SUPPORT_GPU_PHYSX
    mNodalImpactVotes.clear();
    PxArray<const PxsContactManager*> managers;
    PxArray<const PxsDeformableGeometry*> views;
    PxArray<PxU32> starts;
    PxArray<bool*> touching;
    const auto collect=[&](const DeformableBodyCore& core,PxU32 first,bool& hasContact)
    {
        hasContact=false;touching.pushBack(&hasContact);
        views.pushBack(&core.cpuGeometry);starts.pushBack(first);
        for(const auto* cm:core.cpuContactManagers)managers.pushBack(cm);
    };
    for(auto& c:mIslandSurfaceCache)if(c.liveStamp==mIslandSurfaceLiveStamp && c.surface)collect(c.surface->getCore(),c.blockStart,c.hasContact);
    for(auto& c:mIslandSoftCache)if(c.coupled && c.liveStamp==mIslandSoftLiveStamp && c.volume)collect(c.volume->getCore(),c.blockStart,c.hasContact);
    std::sort(managers.begin(),managers.end(),std::less<const PxsContactManager*>());
    struct Source { const PxsContactManagerOutput* output; const PxcNpWorkUnit* unit; };
    PxArray<Source> sources;
    const PxU32 previousCapacity=mNodalFriction.size() ? mNodalFrictionCapacity : 0;
    PxU32 frictionCount=0;
    const auto collectSource=[&](const PxsContactManagerOutput& output,const PxcNpWorkUnit* unit)
    {
        sources.pushBack({&output,unit});
        const auto* contacts=output.geometry?output.geometry->deformableOutput:NULL;
        if(contacts && contacts->valid && contacts->friction>0)frictionCount+=contacts->count;
    };
    const auto pack=[&](const PxsContactManagerOutput& output,const PxcNpWorkUnit* unit)
    {
        const auto* facts=output.geometry;
        if(!facts || !facts->deformableOutput || !facts->deformableOutput->valid)
        {PxAtomicExchange(&mLocalDomainFailed,1);return;}
        if(unit && (unit->mFlags&PxcNpWorkUnitFlag::eDISABLE_RESPONSE))return;
        const auto& contacts=*facts->deformableOutput;
        if(contacts.restitution!=0 || contacts.damping!=0 || (unit &&
            (unit->mFlags&(PxcNpWorkUnitFlag::eMODIFIABLE_CONTACT|PxcNpWorkUnitFlag::eDOMINANCE_0|PxcNpWorkUnitFlag::eDOMINANCE_1))))
        {std::fprintf(stderr,"[AVBD_PARTICLE3] status=UNSUPPORTED reason=np_material_policy\n");PxAtomicExchange(&mLocalDomainFailed,1);return;}
        // An empty, valid NP result has no Jacobian endpoints. An overlapping
        // rigid may already be asleep or belong to another active island.
        if(!contacts.count)return;
        AvbdNodalContact domain={facts,NULL,{PX_MAX_U32,PX_MAX_U32},unit?unit->mRestDistance:0.f};
        if(domain.restDistance<0){PxAtomicExchange(&mLocalDomainFailed,1);return;}
        for(PxU32 s=0;s<2;++s)
        {
            if(facts->deformable[s])
            {
                for(PxU32 v=0;v<views.size();++v)if(views[v]==facts->deformable[s])
                {domain.first[s]=starts[v];*touching[v]=true;break;}
                if(domain.first[s]==PX_MAX_U32){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            }
            else if(unit)
            {
                const auto* core=s?unit->mRigidCore1:unit->mRigidCore0;
                domain.first[s]=findBody(static_cast<const PxsBodyCore*>(core));
                // A non-static endpoint may not silently become an immovable wall.
                const PxU32 flag=s?(PxcNpWorkUnitFlag::eDYNAMIC_BODY1|PxcNpWorkUnitFlag::eARTICULATION_BODY1):
                    (PxcNpWorkUnitFlag::eDYNAMIC_BODY0|PxcNpWorkUnitFlag::eARTICULATION_BODY0);
                if((unit->mFlags&flag) && domain.first[s]==PX_MAX_U32)
                {PxAtomicExchange(&mLocalDomainFailed,1);return;}
                if(domain.first[s]!=PX_MAX_U32 && mBodies[domain.first[s]].core &&
                    mBodies[domain.first[s]].core->maxContactImpulse<1.e32f)
                {std::fprintf(stderr,"[AVBD_PARTICLE3] status=UNSUPPORTED reason=np_force_limit\n");PxAtomicExchange(&mLocalDomainFailed,1);return;}
                if(facts->shapes[s].kind==PxsContactShapeGeometry::ePLANE && domain.first[s]!=PX_MAX_U32)
                {PxAtomicExchange(&mLocalDomainFailed,1);return;}
            }
        }
        const bool volume[2]={facts->deformable[0] && facts->deformable[0]->simulation.arity==4,
                              facts->deformable[1] && facts->deformable[1]->simulation.arity==4};
        // A coherent normal coordinate exists only for stationary parallel supports.
        // Enter at the actor's authored settling speed. Sweeping beyond the NP
        // skin in one step, or contact with a moving partner, invalidates it.
        AvbdIslandSoftCache* planeVolume=NULL;
        for(PxU32 s=0;s<2;++s)if(volume[s])
        {
            auto& cache=mIslandSoftCache[mBlocks[domain.first[s]].objectIndex];
            const auto& other=facts->shapes[1-s];
            if(other.kind!=PxsContactShapeGeometry::ePLANE){cache.planarSupport=false;continue;}
            planeVolume=&cache;
            const PxReal band=PxMax(0.f,facts->contactDistance-domain.restDistance);
            const PxReal settlingSpeed=cache.volume->getCore().settlingThreshold;
            const PxVec3 n=other.shapeToWorld.q.getBasisVector0();
            if(cache.maximumSpeedSquared*mDt*mDt>band*band ||
                (cache.previousSupportNormal!=n && cache.maximumSpeedSquared>settlingSpeed*settlingSpeed) ||
                (!cache.supportNormal.isZero() && cache.supportNormal!=n))
                cache.planarSupport=false;
            cache.supportNormal=n;
            cache.supportFriction=PxMin(cache.supportFriction,contacts.friction);
        }
        const bool volumePair=volume[0] && volume[1] && facts->deformable[0]!=facts->deformable[1];
        const auto fixedSupport=[&](PxU32 s)
        {
            const auto kind=facts->shapes[s].kind;
            return domain.first[s]==PX_MAX_U32 && (kind==PxsContactShapeGeometry::ePLANE || kind==PxsContactShapeGeometry::eBOX);
        };
        const bool project[2]={volumePair || (volume[0] && fixedSupport(1)),volumePair || (volume[1] && fixedSupport(0))};
        if((project[0] || project[1]) && mNodalImpactVotes.empty())
            mNodalImpactVotes.resize(mBlocks.size(),PxI64(0));
        for(PxU32 i=0;i<contacts.count;++i)
        {
            domain.feature=&contacts.contacts[i];const auto& feature=*domain.feature;
            domain.rigidFrame=PX_MAX_U32;
            for(PxU32 s=0;s<2;++s)if(volume[1-s] && domain.first[s]!=PX_MAX_U32 &&
                facts->shapes[s].kind==PxsContactShapeGeometry::eBOX)
            {
                // Only a box face owns a normal that follows its rotation.
                // An edge-edge or tet-face witness retains the shared NP axis.
                const auto axis=facts->shapes[s].shapeToWorld.q.rotateInv(feature.normal);
                if((feature.vertexMask&(1u<<(1-s))) || axis.abs().maxElement()>=1.f-4*PX_EPS_F32)
                    domain.rigidFrame=domain.first[s];
            }
            avbdPrepareNodalProjection(domain);
            domain.separatedTets=volumePair && !feature.vertexMask &&
                facts->deformable[0]->state->mesh.arity==4 && facts->deformable[1]->state->mesh.arity==4 &&
                avbdTetsSeparated(domain,[&](PxU32 s,PxU32 v){return facts->deformable[s]->state->mesh.points[v];});
            AvbdRowBuilder row={};row.bodyA=row.bodyB=PX_MAX_U32;
            row.kernelId=eAVBD_ROW_KERNEL_NODAL_CONTACT;
            row.flags=eAVBD_ROW_CONTACT|eAVBD_ROW_UNILATERAL|eAVBD_ROW_HETEROGENEOUS;
            row.supportStart=mRowSupports.size();row.scalar().n=feature.normal;
            row.scalar().contactOffset=-domain.restDistance;
            row.scalar().minLambda=-PX_MAX_F32;row.scalar().maxLambda=0;
            row.scalar().mu=contacts.friction;row.scalar().positionTangent=1;
            avbdOrthonormalBasis(feature.normal,row.scalar().t0,row.scalar().t1);
            const auto append=[&](PxU32 block,PxReal weight,const PxVec3& local)
            {
                for(PxU32 k=row.supportStart;k<mRowSupports.size();++k)if(mRowSupports[k].block==block)
                {mRowSupports[k].weight+=weight;return;}
                mRowSupports.pushBack({block,weight,local});
            };
            for(PxU32 s=0;s<2;++s)
            {
                const PxReal sign=s?-1.f:1.f;
                if(facts->deformable[s])
                {
                    const auto& mesh=facts->deformable[s]->state->mesh;
                    const bool vertex=(feature.vertexMask&(1u<<s))!=0;
                    const PxU32 count=vertex?1:mesh.arity;
                    for(PxU32 v=0;v<count;++v)
                    {
                        const PxU32 id=vertex?feature.feature[s]:mesh.vertices[count*feature.feature[s]+v];
                        const auto& support=mesh.supports[id];
                        const PxReal weight=sign*(vertex?1.f:feature.weights[s][v]);
                        // All geometry supports are in the CSR, even when this
                        // witness assigns zero force. Coloring sees every read.
                        for(PxU32 k=0;k<support.count;++k)
                            append(domain.first[s]+support.nodes[k],weight*PxReal(support.weights[k]),PxVec3(0));
                    }
                }
                else if(domain.first[s]!=PX_MAX_U32)
                {
                    const auto& shape=facts->shapes[s];
                    const auto local=shape.shapeToBody.transform(shape.shapeToWorld.transformInv(feature.point[s]));
                    append(domain.first[s],sign,local);
                    // Friction samples both velocities at the same DCD world
                    // point. Distinct finite-gap witnesses have different
                    // velocities even under a common rigid rotation.
                    const auto tangent=volume[1-s] ? shape.shapeToBody.transform(
                        shape.shapeToWorld.transformInv(feature.point[1-s])) : local;
                    if(s){row.bodyB=domain.first[s];row.scalar().rB=tangent;row.scalar().ogcRB=local;}
                    else {row.bodyA=domain.first[s];row.scalar().rA=tangent;row.scalar().ogcRA=local;}
                }
                else row.scalar().contactOffset+=sign*feature.normal.dot(feature.point[s]);
            }
            row.supportCount=PxU16(mRowSupports.size()-row.supportStart);
            row.scalar().C0=domain.rigidFrame!=PX_MAX_U32 ? feature.separation-domain.restDistance :
                avbdHeterogeneousRowC(row,mRowSupports,mBlocks,mBodies,contactGeometry());
            PxReal mobility=0,tangentMobility=0,predictedGap=row.scalar().C0;
            PxVec3 relativeTravel(0);
            PxReal maxWeight[2]={};
            domain.loadNodes[0]=domain.loadNodes[1]=PX_MAX_U32;
            for(PxU32 k=row.supportStart;k<mRowSupports.size();++k)
            {
                const auto& support=mRowSupports[k];const auto& block=mBlocks[support.block];
                PxVec3 travel;
                if(block.particle)
                {
                    const PxReal response=support.weight*support.weight*block.particle->invMass;
                    mobility+=response;
                    tangentMobility+=response;
                    travel=block.particle->pHat-block.particle->pStart;
                    if(facts->deformable[0] && facts->deformable[1] &&
                        facts->deformable[0]!=facts->deformable[1] && block.particle->invMass>0)
                    {
                        const PxU32 side=avbdNodalPairSide(domain,support.block);
                        const PxReal scale=PxAbs(support.weight)*PxSqrt(block.particle->invMass);
                        if(scale>maxWeight[side])
                        {maxWeight[side]=scale;domain.loadNodes[side]=support.block;}
                    }
                }
                else
                {
                    const auto& body=mBodies[block.objectIndex];
                    const auto arm=support.block==row.bodyA?row.scalar().rA:row.scalar().rB;
                    const auto normalArm=domain.rigidFrame!=PX_MAX_U32?arm:support.localPoint;
                    mobility+=support.weight*support.weight*avbdEndpointAxisResponse(body,feature.normal,normalArm);
                    if(contacts.friction>0)
                    {
                        tangentMobility+=.5f*support.weight*support.weight*(
                            avbdEndpointAxisResponse(body,row.scalar().t0,arm)+
                            avbdEndpointAxisResponse(body,row.scalar().t1,arm));
                    }
                    const PxQuat rotation=body.qHat*body.qStart.getConjugate();
                    const PxVec3 spin(rotation.x,rotation.y,rotation.z);
                    const PxVec3 startArm=body.qStart.rotate(normalArm);
                    // R(delta)*a-a without subtracting two nearly equal endpoints.
                    travel=(body.pHat-body.pStart)+2.f*spin.cross(spin.cross(startArm)+rotation.w*startArm);
                }
                const PxReal normalStep=feature.normal.dot(travel);
                if(domain.rigidFrame!=PX_MAX_U32)relativeTravel+=travel*support.weight;
                else predictedGap+=support.weight*normalStep;

            }
            if(domain.rigidFrame!=PX_MAX_U32)
            {
                const auto& b=mBodies[domain.rigidFrame];
                predictedGap+=feature.normal.dot(avbdNodalMaterialVector(b.qStart,b.qHat,relativeTravel));
            }
            if(!(mobility>0)){mRowSupports.forceSize_Unsafe(row.supportStart);continue;}
            // Seed the normal from the relative inertial target and its
            // effective mass, keeping the rigid unilateral metric at r=0.
            row.scalar().rho=PxMax(AVBD_CONTACT_PENALTY_MIN,2.f/(mobility*mDt*mDt));
            // A relative tangential displacement of h*v demands an impulse
            // of -v/W. Use the trace on the tangent plane for an isotropic disk.
            row.scalar().rhoT0=row.scalar().rhoT1=tangentMobility>0 ? 1.f/(tangentMobility*mDt*mDt) : 0.f;
            row.scalar().contactRadius=PxMax(0.f,facts->contactDistance-domain.restDistance);
            row.scalar().contactStiffness=mOgc.contactStiffness;
            // Initialize the single shared dual at the inertially loaded
            // contact. For an isolated row d = dHat + h^2 W f; eliminating
            // d gives this scalar resolvent. No pose is changed here.
            const PxReal inverseResponse=1.f/(mobility*mDt*mDt);
            row.scalar().lambda=-avbdOGCNormalResolvent(-inverseResponse*predictedGap,
                inverseResponse,row.scalar().contactRadius,row.scalar().contactStiffness);
            if(planeVolume && planeVolume->planarSupport && predictedGap<row.scalar().C0)
            {
                // Every embedded witness sees the same normal interpolation of
                // its signed barycentric supports. Intersect their scalar ranges.
                const PxReal loadedGap=avbdOGCLoadedGap(-row.scalar().lambda,row.scalar().contactRadius,row.scalar().contactStiffness);
                const PxReal inward=row.scalar().C0-predictedGap;
                const PxReal fraction=PxClamp((row.scalar().C0-loadedGap)/inward,0.f,1.f);
                planeVolume->supportFraction=PxMin(planeVolume->supportFraction,fraction);
            }
            if(row.scalar().contactRadius>0)
                row.scalar().rho=avbdOGCLoadedNormalMetric(mobility*mDt*mDt,-row.scalar().lambda,
                    row.scalar().contactRadius,row.scalar().contactStiffness);
            // The offset potential already carries pressure at a positive
            // gap. A moving finite patch can cross its plane at another
            // corner while the GJK witness still has clearance.
            const bool finitePatch=volumePair || (project[0] && facts->shapes[1].kind==PxsContactShapeGeometry::eBOX) ||
                (project[1] && facts->shapes[0].kind==PxsContactShapeGeometry::eBOX);
            const bool loaded=finitePatch ? row.scalar().lambda<0 : predictedGap<0;
            if(loaded && (project[0] || project[1]))
                mNodalImpactRows.pushBack(mNodalContacts.size());
            domain.frictionSlot=PX_MAX_U32;
            if(contacts.friction>0)
            {
                const auto* view=facts->deformable[facts->deformable[0]?0:1];
                const PxU64 epoch=view->state->epoch;
                const PxU32 hash=PxComputeHash(facts)^PxComputeHash(feature.feature[0])^
                    PxComputeHash(feature.feature[1]+0x9e3779b9u*feature.vertexMask);
                AvbdNodalFriction next={};
                next.pair=facts;next.epoch=epoch;next.vertexMask=feature.vertexMask;
                next.birthEpoch=contacts.birthEpoch;
                next.feature[0]=feature.feature[0];next.feature[1]=feature.feature[1];
                next.normal=feature.normal;next.traction=PxVec3(0);
                PxU32 prior=PX_MAX_U32;
                for(PxU32 probe=0;probe<8 && previousCapacity;++probe)
                {
                    const PxU32 slot=(hash+probe)&(previousCapacity-1);
                    const auto& entry=mPreviousNodalFriction[slot];
                    if(!entry.pair)break;
                    if(entry.pair==facts && entry.vertexMask==feature.vertexMask &&
                        entry.feature[0]==feature.feature[0] && entry.feature[1]==feature.feature[1])
                    {prior=slot;break;}
                }
                if(prior==PX_MAX_U32)prior=mNodalFrictionOverflow.find(mPreviousNodalFriction,next);
                if(prior!=PX_MAX_U32 && avbdReadNodalFriction(mPreviousNodalFriction[prior],next,planeVolume,row.scalar()))
                    avbdProjectCoulomb(row.scalar().lambda,contacts.friction,row.scalar().lambdaT0,row.scalar().lambdaT1);
                for(PxU32 probe=0;probe<8;++probe)
                {
                    const PxU32 slot=(hash+probe)&(mNodalFrictionCapacity-1);
                    if(mNodalFriction[slot].pair)continue;
                    mNodalFriction[slot]=next;domain.frictionSlot=slot;break;
                }
                if(domain.frictionSlot==PX_MAX_U32)
                {
                    domain.frictionSlot=mNodalFriction.size();
                    mNodalFriction.pushBack(next);
                }
            }
            domain.loadWeights[0]=maxWeight[0];domain.loadWeights[1]=maxWeight[1];
            mStep.matchNodal |= maxWeight[0]>0 && maxWeight[1]>0 && row.scalar().lambda<0;
            mNodalContacts.pushBack(domain);row.geometryIndex=mNodalContacts.size();mRows.pushBack(row);
        }
    };
    const PxsContactManager* previous=NULL;
    for(const auto* cm:managers)
    {
        if(!cm){std::fprintf(stderr,"[AVBD_PARTICLE3] status=UNSUPPORTED reason=np_manager_missing\n");PxAtomicExchange(&mLocalDomainFailed,1);continue;}
        if(cm==previous)continue;
        previous=cm;const auto& unit=cm->getWorkUnit();
        if(unit.mNpIndex&PxsContactManagerBase::NEW_CONTACT_MANAGER_MASK){PxAtomicExchange(&mLocalDomainFailed,1);continue;}
        collectSource(mOutputIterator.getContactManagerOutput(unit.mNpIndex),&unit);
    }
    for(const auto* view:views)if(view->selfCollision())
    {if(view->state)collectSource(view->state->selfOutput,NULL);else PxAtomicExchange(&mLocalDomainFailed,1);}
    mPreviousNodalFriction.swap(mNodalFriction);
    mNodalFriction.clear();
    mNodalFrictionOverflow.build(mPreviousNodalFriction,previousCapacity);
    mNodalFrictionCapacity=0;
    if(frictionCount)
    {
        PxU32 capacity=8;
        while(capacity/2<frictionCount && capacity<(1u<<30))capacity*=2;
        mNodalFriction.resize(capacity);
        mNodalFrictionCapacity=capacity;
    }
    for(const auto& source:sources)pack(*source.output,source.unit);
    mNativeMaterialEnd=mRows.size();
#else
    PxAtomicExchange(&mLocalDomainFailed,1);
#endif
}
