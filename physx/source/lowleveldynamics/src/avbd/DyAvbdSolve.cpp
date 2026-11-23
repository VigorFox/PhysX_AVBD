// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Rigid step = demo3d solver.cpp + manifold.cpp, then PhysX bounce only.
// ASCII comments: MSVC C4819 on code page 936.

#include "avbd/DyAvbdInternal.h"
#include "avbd/DyAvbdRigidMetric.h"
#include "avbd/DyAvbdBlockGroup.h"
#include "avbd/DyAvbdCommonResponse.h"
#include "avbd/DyAvbdBoundedResponse.h"
#include "avbd/DyAvbdRowWorkingSet.h"
#include "avbd/DyAvbdCommonResidual.h"
#include "avbd/DyAvbdCurvatureMetric.h"
#include "avbd/DyAvbdNonlinearBlock.h"
#include "avbd/DyAvbdFrameAssembly.h"
#include "avbd/DyAvbdProgramSchedule.h"
#include <cstdio>
#include <string>
#include <cstdlib>
#include "foundation/PxTime.h"
#include "avbd/DyAvbdKernelsScalar.h"
#include "avbd/DyKernelTable.h"
#include "avbd/DyAvbdNonlinearGroup.h"
#include "avbd/DyAvbdLocalWork.h"
#include "DyFeatherstoneArticulation.h"
#include "DySleep.h"

using namespace physx;
using namespace Dy;

// Known translation-invariant local objectives. Other kernels retain their
// ordinary coordinates until the same property has been established for them.
static bool localTranslationInvariant(const AvbdRowConstRef& row)
{
    const auto& h=row.header();
    return h.kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME || h.kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON ||
        h.kernelId==eAVBD_ROW_KERNEL_NP_CONTACT ||
        (h.kernelId==eAVBD_ROW_KERNEL_CONSTRAINT && (h.flags&eAVBD_ROW_CONTACT) && !h.supportCount);
}

template<class Bodies>
static AvbdElasticFrameEvaluation evaluateReferenceFrame(const AvbdRowConstRef& row,
    const AvbdElasticFrameMaterials& materials,const Bodies& bodies,
    const KernelTable& kernels,bool curvature=true)
{
    const PxTransform poses[2]={PxTransform(bodies[row.header().bodyA].p,bodies[row.header().bodyA].q),
        PxTransform(bodies[row.header().bodyB].p,bodies[row.header().bodyB].q)};
    AvbdElasticFrameEvaluation result;
    const auto material=materials[row.header().geometryIndex-1];
    AvbdElasticFrameArgs args={&material,poses,
        curvature?AvbdElasticFrameRequest::Reference:AvbdElasticFrameRequest::ReferenceGradient,&result};
    kernels.launch(AVBD_ELASTIC_FRAME_MATERIAL,&args);return result;
}

template<class Output,class Bodies>
static Output evaluateFrameRequest(const AvbdRowConstRef& row,const AvbdElasticFrameMaterials& materials,
    const Bodies& bodies,const KernelTable& kernels,AvbdElasticFrameRequest request,PxU32 endpoint=2)
{
    const PxTransform poses[2]={PxTransform(bodies[row.header().bodyA].p,bodies[row.header().bodyA].q),
        PxTransform(bodies[row.header().bodyB].p,bodies[row.header().bodyB].q)};
    const auto material=materials[row.header().geometryIndex-1];
    Output result;AvbdElasticFrameArgs args={&material,poses,request,&result,endpoint};
    kernels.launch(AVBD_ELASTIC_FRAME_MATERIAL,&args);return result;
}

template<class Bodies>
static void tetrahedronPoints(const AvbdRowConstRef& row, const PxArray<AvbdRowSupport>& supports,
    const Bodies& bodies, PxVec3d* points)
{
    for(PxU32 s=0;s<4;++s)
    {
        const AvbdRowSupport& support=supports[row.header().supportStart+s];
        const PxVec3 p=support.block<bodies.size() ?
            (support.localPoint.isZero() ? bodies[support.block].p :
                avbdWorldPoint(bodies[support.block],support.localPoint)) : support.localPoint;
        points[s]=PxVec3d(p.x,p.y,p.z);
    }
}

template<class OldBodies,class NewBodies>
static bool tetrahedronPath(const AvbdRowConstRef& row,const AvbdTetrahedronMaterial& material,
    const PxArray<AvbdRowSupport>& supports,const OldBodies& oldPoses,const NewBodies& newPoses)
{
    PxVec3d p[4],q[4]; double errors[4]={};
    tetrahedronPoints(row,supports,oldPoses,p); tetrahedronPoints(row,supports,newPoses,q);
    for(PxU32 s=0;s<4;++s)
    {
        const AvbdRowSupport& support=supports[row.header().supportStart+s];
        // COM anchors follow the translation chord exactly. Their rotations
        // cannot contribute to this path bound, so do not compute four logs.
        if(support.block>=oldPoses.size() || support.localPoint.isZero() ||
            oldPoses[support.block].q==newPoses[support.block].q) continue;
        const double angle=avbdAcceptedRotationDisplacement(oldPoses[support.block].q,newPoses[support.block].q).magnitude();
        errors[s]=double(support.localPoint.magnitude())*angle*angle/8.0;
    }
    return avbdTetrahedronPathAdmissible(material,p,q,errors);
}

AvbdTetrahedronEvaluation AvbdDynamicsContext::evaluateTetrahedron(const AvbdRowConstRef& row,
    PxU32 candidateIndex,const PxTransform* candidate) const
{
    PxVec3d points[4];
    if(candidate)
    {
        const AvbdRigidPoseView poses={mBodies,&candidateIndex,candidate,1};
        tetrahedronPoints(row,mRowSupports,poses,points);
    }
    else tetrahedronPoints(row,mRowSupports,mBodies,points);
    AvbdTetrahedronEvaluation result;
    AvbdTetrahedronArgs args={&mTetrahedronMaterials[row.header().geometryIndex-1],points,&result};
    mKernels.launch(AVBD_TETRAHEDRON_MATERIAL,&args);
    return result;
}

// Start every responding Rigid6 at the common BDF1 inertial target.
// Keep the world rotation chart and public locked axes used by primal.
static bool avbdRigidWarmCandidate(const AvbdSolverBody& b,PxTransform& candidate)
{
	// A log followed by normalized Euler retraction shortens finite rotations.
	// The inertial target is already a pose; preserve a common finite motion.
	candidate.p=b.pHat;
	candidate.q=b.qHat;
	if(b.core && b.core->lockFlags)
	{
		PxVec3 dp=b.pHat-b.p;
		PxQuat relative=b.qHat*b.q.getConjugate();
		if(relative.w<0.0f) relative=-relative;
		// Invert normalize((1,dtheta/2)*q), then project its world axes.
		// A half turn is outside this chart and cannot be silently shortened.
		if(!(relative.w>0.0f)) return false;
		PxVec3 dth=relative.getImaginaryPart()*(2.0f/relative.w);
		if(!dth.isFinite()) return false;
		avbdProjectLockedDelta(b,dp,dth);
		candidate=PxTransform(b.p,b.q);
		avbdApplyDelta(candidate,dp,dth);
	}
	return candidate.p.isFinite() && candidate.q.isFinite();
}

PxVec3 AvbdDynamicsContext::warmParticleTarget(const AvbdBlock& block,PxU32 index) const
{
    const auto& p=*block.particle;
    if(mStep.localBudget && block.particleKernel==eAVBD_PARTICLE_KERNEL_TET)
    {
        const auto& cache=mIslandSoftCache[block.objectIndex];
        if(cache.planarSupport && cache.supportFraction<1)
        {
            // This is a primal seed. Normal feasibility and the Coulomb
            // disk limit the proposed travel; pHat, the dual and BDF remain shared.
            const auto n=cache.supportNormal;
            const auto travel=p.pHat-p.pStart;
            const auto normal=n.dot(travel);
            const auto removed=(1-cache.supportFraction)*normal;
            const auto tangent=travel-n*normal;
            const auto friction=cache.supportFriction*PxMax(0.f,-removed);
            const auto len=tangent.magnitude();
            return p.pHat-n*removed-(len>friction?tangent*(friction/len):tangent);
        }
    }
    if(mStep.localBudget && !mNodalImpactVotes.empty() && mNodalImpactVotes[index])
    {
        const PxI64 vote=mNodalImpactVotes[index];
        const PxU32 contactSide=PX_MAX_U32-PxU32(vote);
        const PxReal delta=PxUnionCast<PxF32>(PxU32(PxU64(vote)>>32));
        return p.pHat+mNodalContacts[contactSide>>1].feature->normal*((contactSide&1)?-delta:delta);
    }
    if(!mStep.localBudget || p.supportedGravityTravel.isZero())return p.pHat;
    const bool hasContact=block.particleKernel==eAVBD_PARTICLE_KERNEL_TRIANGLE ?
        mIslandSurfaceCache[block.objectIndex].hasContact : mIslandSoftCache[block.objectIndex].hasContact;
    // Previous support can balance up to the external gravity displacement.
    // Only the initial iterate uses this estimate. The objective retains pHat.
    return hasContact?p.pHat-p.supportedGravityTravel:p.pHat;
}

template<class Point>
bool AvbdDynamicsContext::warmMaterialAdmissible(const AvbdRowConstRef& row,const Point& point) const
{
    AvbdNodalVec3 x[4];
    for(PxU32 s=0;s<row.header().supportCount;++s)
    {const auto p=point(mRowSupports[row.header().supportStart+s].block);x[s]=AvbdNodalVec3(p.x,p.y,p.z);}
    AvbdNodalShellSampleT<AvbdNodalReal> sample;
    if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_VOLUME)
    {
        const auto& material=mNodalVolumeMaterials[row.header().geometryIndex-1];
        sample=avbdSampleNodalVolume(*material.rest,x,material.mu,material.lambda,PX_MAX_U32);
        AvbdNodalVec3 start[4];
        for(PxU32 s=0;s<4;++s)
        {const auto p=mBlocks[mRowSupports[row.header().supportStart+s].block].particle->p;start[s]=AvbdNodalVec3(p.x,p.y,p.z);}
        sample.valid &= avbdNodalVolumePath(*material.rest,start,x);
    }
    else
    {
        const auto& material=mNodalShellMaterials[row.header().geometryIndex-1];
        sample=avbdSampleNodalShell(*material.rest,x,material.mu,material.lambda,material.bending,PX_MAX_U32);
        if(material.previous && row.header().supportCount==3)
        {
            AvbdNodalVec3 before[3],after[3];
            for(PxU32 s=0;s<3;++s)
            {
                const PxVec3& p=mBlocks[mRowSupports[row.header().supportStart+s].block].particle->pStart;
                before[s]=AvbdNodalVec3(p.x,p.y,p.z);
                after[s]=x[s];
            }
            sample.valid &= avbdMembranePath(before,after);
        }
    }
    return sample.valid;
}

void AvbdDynamicsContext::prepareWarmChecks()
{
    mWarmFailureMasks.clear();
    // Attachments can edit both sides of the predictor. Use their fully
    // assembled candidate in admitWarmstartCandidate. Small programs can be
    // checked inline for less than a 256-row batch per available worker.
    const PxU32 materials=mNodalShellMaterials.size()+mNodalVolumeMaterials.size();
    if(!mStep.localBudget || !mStep.rowProgramValid || !mStep.continuation || mStep.workers<2 ||
        !mAttachmentRows.empty() || !materials || materials+mNodalContacts.size()!=mRows.size() ||
        mRows.size() < AvbdLocalWork::RowLimit*PxMin(mStep.workers,32u))return;

    mWarmPoses.resize(mBodies.size());
    for(PxU32 b=0;b<mBodies.size();++b)
    {
        const auto& body=mBodies[b];
        auto& pose=mWarmPoses[b];
        pose=PxTransform(body.p,body.q);
        if(body.invMass!=0 && !body.isKinematic && !avbdRigidWarmCandidate(body,pose))return;
    }
    mWarmPoints.resize(mBlocks.size()-mBodies.size());
    for(PxU32 b=mBodies.size();b<mBlocks.size();++b)
    {
        const auto& block=mBlocks[b];
        if(!block.particle)return;
        auto& seed=mWarmPoints[b-mBodies.size()];
        seed=block.particle->invMass>0 ? warmParticleTarget(block,b) : block.particle->p;
        if(!seed.isFinite())return;
    }
    mWarmFailureMasks.resize((mRows.size()-1)/32+1);
    mStep.warmCheckedRows=mRows.size();
}

void AvbdDynamicsContext::checkWarmRows(PxU32 maskIndex)
{
    const AvbdRigidPoseView poses={mBodies,NULL,mWarmPoses.begin(),mWarmPoses.size()};
    const auto point=[&](PxU32 b){return mWarmPoints[b-mBodies.size()];};
    const PxU32 start=32*maskIndex,count=PxMin(32u,mRows.size()-start);
    PxU32 failed=0;
    for(PxU32 i=0;i<count;++i)
    {
        const AvbdRowConstRef& row=mRows[start+i];
        const bool valid=row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT ?
            avbdNodalContactPath(mNodalContacts[row.header().geometryIndex-1],point,poses) :
            avbdIsNodalMaterial(row.header().kernelId) && warmMaterialAdmissible(row,point);
        if(!valid)failed |= 1u<<i;
    }
    mWarmFailureMasks[maskIndex]=failed;
}

bool AvbdDynamicsContext::admitWarmstartCandidate(const AvbdProgramSubset* subset,PxReal fraction)
{
	// Candidate storage is private until the compiled hard geometry/material
	// domains accept it. Certified NP pairs additionally admit their complete
	// finite support path. The normal law is selected by the compiled row.
	// The inertial objective and DCD epoch stay intact.
    const bool checked=!subset && fraction==1.f && !mWarmFailureMasks.empty();
    PxArray<PxTransform> inlinePoses;
    PxArray<PxTransform>& poses=checked?mWarmPoses:inlinePoses;
    if(!checked)poses.resize(mBodies.size());
	PxArray<PxU8> selected(mBodies.size(),subset?0:1);
	if(subset)for(PxU32 b:subset->blocks)selected[b]=1;
	// A failed predictor in one component cannot remove the gravity mode of
	// an unrelated elastic component. Test one complete candidate and collect
	// component vetoes in linear scratch; never copy/rescan the island per root.
	PxArray<PxU32> components;PxArray<PxU8> rejected;
    const bool componentVeto=mStep.localBudget && !subset;
    const PxU32 nodalMaterials=mNodalShellMaterials.size()+mNodalVolumeMaterials.size();
    const bool onlyNodal=nodalMaterials && nodalMaterials+mNodalContacts.size()+3*mAttachmentRows.size()==mRows.size();
    const auto prepareComponents=[&]()
	{
        if(!componentVeto || !components.empty())return;
		mRowProgram.components(components,[&](PxU32 b){return b<mBodies.size()?avbdBodyResponds(mBodies[b]):
            mBlocks[b].particle && mBlocks[b].particle->invMass>0;},
			[&](PxU32 r){return !(mRows[r].header().flags&eAVBD_ROW_INACTIVE) ||
				mRows[r].header().kernelId==eAVBD_ROW_KERNEL_NP_CONTACT ||
				mRows[r].header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON;});
		rejected.resize(components.size(),0);
	};
    if(!onlyNodal)prepareComponents();
	const auto rejectBody=[&](PxU32 b)
	{prepareComponents();if(components.empty())return false;rejected[components[b]]=1;return true;};
	const auto rejectRow=[&](PxU32 r)
	{prepareComponents();if(components.empty())return false;mRowProgram.forEachBlock(r,[&](PxU32 b){rejected[components[b]]=1;});return true;};
	const AvbdRigidPoseView bodies={mBodies,NULL,poses.begin(),poses.size()};
	const PxU32 particleStart=mBodies.size();
	const PxU32 particleCount=mBlocks.size()-particleStart;
	PxArray<AvbdBlock> candidateBlocks;
    const bool directNodal=onlyNodal && mStep.localBudget;
	if(particleCount && !directNodal) candidateBlocks=mBlocks;
	const PxArray<AvbdBlock>& blocks=candidateBlocks.empty()?mBlocks:candidateBlocks;
	PxArray<AvbdSoftParticle> particles;
	particles.resize(directNodal?0:particleCount);
    // A rejected plane witness or moving box may change its normal seed.
    // Recheck all affected trajectories before admitting the combined proposal.
    PxArray<PxU32> movingBoxes,changedBlocks,failedNodalRows;
    PxArray<PxU8> boxMarked;
    PxArray<PxVec3> boxShift,normalTravel,inlinePoints;
    PxArray<PxVec3>& initialPoints=checked?mWarmPoints:inlinePoints;
    if(!checked && directNodal)initialPoints.resize(particleCount);
    bool localized=false,canCondition=componentVeto && directNodal;
    const auto candidatePoint=[&](PxU32 b)->PxVec3
    {
        const auto& p=*blocks[b].particle;
        if(!directNodal)return p.p;
        const auto& seed=initialPoints[b-particleStart];
        return localized && p.invMass>0 ? seed+normalTravel[b] : seed;
    };
	for(PxU32 i=0;!checked && i<bodies.size();++i)
	{
		const AvbdSolverBody& b=mBodies[i];
		poses[i]=PxTransform(b.p,b.q);
		if(!selected[i] || b.invMass==0.0f || b.isKinematic) continue;
		if(!avbdRigidWarmCandidate(b,poses[i]))
		{if(!rejectBody(i))return false;canCondition=false;poses[i]=PxTransform(b.p,b.q);}
		if(fraction<1.f)
		{
			poses[i].p=b.p+(poses[i].p-b.p)*fraction;
			PxQuat target=poses[i].q;if(b.q.dot(target)<0)target=-target;
			poses[i].q=(b.q*(1.f-fraction)+target*fraction).getNormalized();
		}
	}
	for(PxU32 i=particleStart;!checked && i<blocks.size();++i)
	{
        if(directNodal)
        {
            const auto* particle=blocks[i].particle;
            if(!particle)continue;
            auto& seed=initialPoints[i-particleStart];
            seed=particle->invMass>0 ? warmParticleTarget(blocks[i],i) : particle->p;
            if(!seed.isFinite())return false;
            continue;
        }
		AvbdBlock& block=candidateBlocks[i];
		if(block.type!=eAVBD_BLOCK_PARTICLE3 || !block.particle) continue;
		AvbdSoftParticle& particle=particles[i-particleStart];
		particle=*block.particle;
		block.particle=&particle;
		if(particle.invMass>0.0f) particle.p=warmParticleTarget(block,i);
		if(!particle.p.isFinite()) return false;
	}
    // Eliminate inertia in the three attachment coordinates. Each independent
    // point has at most five supports. Its proposed impulses sum to zero.
    // The material and NP domain checks below still admit the complete path.
    struct AttachmentSeed { PxU32 first, rigid; PxVec3 force; };
    PxArray<AttachmentSeed> attachmentSeeds;
    for(PxU32 r : mAttachmentRows)
    {
        const auto row=mRows[r];
        const auto& h=row.header();
        const PxU32 last=h.supportStart+h.supportCount-1;
        const PxU32 rigid=h.bodyA;
        const auto& body=mBodies[rigid];
        if(!directNodal || body.isKinematic || body.core->lockFlags) continue;
        bool independent=true;
        if(mAttachmentRows.size()>1)
            for(PxU32 k=h.supportStart;k<=last && independent;++k)
            {
                const PxU32 b=mRowSupports[k].block;
                const auto lo=mRowProgram.blockOffsets[b],hi=mRowProgram.blockOffsets[b+1];
                if(hi-lo>AvbdLocalWork::RowLimit){independent=false;break;}
                for(PxU32 j=lo;j<hi;++j)
                {
                    const PxU32 neighbor=mRowProgram.blockRows[j];
                    if((neighbor<r || neighbor>=r+3) && mRows[neighbor].header().kernelId==eAVBD_ROW_KERNEL_ATTACHMENT)
                    {independent=false;break;}
                }
            }
        if(!independent) continue;
        const PxVec3 arm=poses[rigid].q.rotate(mRowSupports[last].localPoint);
        PxVec3 gap=-arm;
        PxReal inverseMass=body.invMass;
        for(PxU32 k=h.supportStart;k<last;++k)
        {
            const auto& s=mRowSupports[k];
            inverseMass+=s.weight*s.weight*mBlocks[s.block].particle->invMass;
            gap+=s.weight*(initialPoints[s.block-particleStart]-poses[rigid].p);
        }
        PxMat33 mobility(PxIdentity);
        for(PxU32 k=0;k<3;++k)
        {
            const PxVec3 axis=mobility[k];
            mobility[k]=axis*inverseMass+avbdWorldInvInertiaMul(body,arm.cross(axis)).cross(arm);
        }
        const PxVec3 impulse=mobility.getInverse()*gap;
        if(!impulse.isFinite()) return false;
        for(PxU32 k=h.supportStart;k<last;++k)
        {
            const auto& s=mRowSupports[k];
            initialPoints[s.block-particleStart]-=impulse*(s.weight*mBlocks[s.block].particle->invMass);
        }
        avbdApplyDelta(poses[rigid],body.invMass*impulse,avbdWorldInvInertiaMul(body,arm.cross(impulse)));
        attachmentSeeds.pushBack({r,rigid,impulse/(mDt*mDt)});
    }
    // Every expanded inequality of a complete NP domain asks the identical
    // DCD-to-candidate geometry question. Memoize it for THIS immutable pose
    // pair only; test the full domain once, including all its finite supports.
    PxArray<PxU8> testedDomains(mNpDomains.size(),0);
	const PxU32 rowCount=subset?subset->rows.size():mRows.size();
    if(checked)mStep.warmDomainTests+=mNodalContacts.size();
	for(PxU32 slot=0;slot<rowCount;++slot)
	{
        if(checked)
        {
            const PxU32 mask=mWarmFailureMasks[slot/32] & (~0u<<(slot%32));
            if(!mask){slot |= 31u;continue;}
            slot=(slot & ~31u)+PxLowestSetBit(mask);
            if(mStep.profile)++mStep.warmFailedRows;
        }
		const PxU32 i=subset?subset->rows[slot]:slot;
		const AvbdRowConstRef& row=mRows[i];
        if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
        {
            if(!checked)++mStep.warmDomainTests;
            const bool valid=!checked && avbdNodalContactPath(mNodalContacts[row.header().geometryIndex-1],
                candidatePoint,bodies);
            if(!valid && canCondition)
            {
                failedNodalRows.pushBack(i);
                const auto& domain=mNodalContacts[row.header().geometryIndex-1];
                const PxU32 soft=domain.geometry->deformable[0]?0:1,other=1-soft;
                const PxU32 rigid=domain.first[other];
                const auto& shape=domain.geometry->shapes[other];
                bool witnessClear=false;
                if(shape.kind==PxsContactShapeGeometry::eBOX && !domain.feature->vertexMask &&
                    domain.projection==(soft?AvbdNodalContact::StaticSide1:AvbdNodalContact::StaticSide0))
                {
                    // A witness can stay outside the hard boundary even while an
                    // unloaded corner of its finite feature would cross. Reconcile
                    // that geometric discrepancy before the ordinary primal solve.
                    PxReal witness=row.scalar().C0;
                    for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
                    {
                        const auto& entry=mRowSupports[k];
                        witness+=entry.weight*domain.feature->normal.dot(candidatePoint(entry.block)-mBlocks[entry.block].particle->pStart);
                    }
                    witnessClear=witness>=0;
                }
                if(!domain.geometry->deformable[other] && rigid==PX_MAX_U32 && (witnessClear ||
                    (domain.geometry->deformable[soft]->simulation.arity==4 && shape.kind==PxsContactShapeGeometry::ePLANE)))
                {
                    const auto n=soft?-domain.feature->normal:domain.feature->normal;
                    const auto& mesh=domain.geometry->deformable[soft]->state->mesh;
                    const bool vertex=(domain.feature->vertexMask&(1u<<soft))!=0;
                    const PxU32 count=vertex?1u:mesh.arity;
                    const PxReal extent=witnessClear?shape.halfExtents.dot(shape.shapeToWorld.q.rotateInv(n).abs()):0.f;
                    const PxReal clearance=extent+domain.restDistance+shape.radius+domain.geometry->shapes[soft].radius;
                    const auto loadedGap=avbdOGCLoadedGap(-row.scalar().lambda,row.scalar().contactRadius,row.scalar().contactStiffness);
                    // Withdraw only the crossing vertex motion in its mass metric.
                    // Bound the target by the original gap to avoid adding separation.
                    for(PxU32 v=0;v<count;++v)
                    {
                        const PxU32 id=vertex?domain.feature->feature[soft]:mesh.vertices[mesh.arity*domain.feature->feature[soft]+v];
                        const auto point=avbdNodalCollisionPoint(domain,soft,id,candidatePoint);
                        const auto gap=n.dot(point-shape.shapeToWorld.p)-clearance;
                        if(gap>=0)continue;
                        const auto original=mesh.points[id];
                        const PxReal startGap=n.dot(PxVec3(PxReal(original.x),PxReal(original.y),PxReal(original.z))-shape.shapeToWorld.p)-clearance;
                        const PxReal target=PxClamp(startGap,0.f,loadedGap);
                        const auto& embedded=mesh.supports[id];PxReal mobility=0;
                        for(PxU32 k=0;k<embedded.count;++k)
                        {
                            const auto weight=PxReal(embedded.weights[k]);
                            mobility+=weight*weight*mBlocks[domain.first[soft]+embedded.nodes[k]].particle->invMass;
                        }
                        if(!(mobility>0))continue;
                        if(normalTravel.empty())normalTravel.resize(mBlocks.size(),PxVec3(0));
                        for(PxU32 k=0;k<embedded.count;++k)
                        {
                            const PxU32 block=domain.first[soft]+embedded.nodes[k];
                            const auto scale=PxReal(embedded.weights[k])*mBlocks[block].particle->invMass/mobility;
                            const auto correction=n*(scale*(target-gap));auto& current=normalTravel[block];
                            if(correction.magnitudeSquared()>current.magnitudeSquared())
                            {
                                if(current.isZero())changedBlocks.pushBack(block);
                                current=correction;
                            }
                        }
                    }
                }
                if(domain.geometry->deformable[soft]->simulation.arity==4 && !domain.geometry->deformable[other] &&
                    domain.geometry->shapes[other].kind==PxsContactShapeGeometry::eBOX && rigid!=PX_MAX_U32 &&
                    avbdBodyResponds(mBodies[rigid]))
                {
                    if(boxMarked.empty())
                    {
                        boxMarked.resize(mBodies.size(),0);
                        boxShift.resize(mBodies.size(),PxVec3(0));
                    }
                    if(!boxMarked[rigid]){boxMarked[rigid]=1;movingBoxes.pushBack(rigid);}
                    const auto& body=mBodies[rigid];
                    // The mass-metric least-squares translation of this
                    // finite NP patch at the predicted rigid orientation.
                    const auto rotation=poses[rigid].q*body.qStart.getConjugate();
                    PxVec3 delta(0);PxReal mass=1.f/body.invMass;
                    for(PxU32 k=0;k<row.header().supportCount;++k)
                    {
                        const PxU32 id=mRowSupports[row.header().supportStart+k].block;
                        const auto* p=mBlocks[id].particle;
                        if(!p || p->invMass<=0)continue;
                        const auto carried=poses[rigid].p+rotation.rotate(p->pStart-body.pStart);
                        delta+=p->mass*(candidatePoint(id)-carried);
                        mass+=p->mass;
                    }
                    delta/=mass;
                    PxVec3 angular(0);avbdProjectLockedDelta(body,delta,angular);
                    if(delta.magnitudeSquared()>boxShift[rigid].magnitudeSquared())boxShift[rigid]=delta;
                }
            }
            if(!valid && !canCondition && !rejectRow(i))return false;
            continue;
        }
        if(avbdIsNodalMaterial(row.header().kernelId))
        {
            if(checked || !warmMaterialAdmissible(row,candidatePoint))
            {
                if(canCondition)failedNodalRows.pushBack(i);
                else if(!rejectRow(i))return false;
            }
            continue;
        }
        if(mNpDomainMode==2 && row.header().kernelId==eAVBD_ROW_KERNEL_NP_CONTACT)
        {
            if(!row.header().geometryIndex || row.header().geometryIndex>testedDomains.size())return false;
            PxU8& tested=testedDomains[row.header().geometryIndex-1];
            if(!tested)
            {
                ++mStep.warmDomainTests;
                if(mStep.localBudget)
                {
                    const auto& domain=mNpDomains[row.header().geometryIndex-1];
                    const PxU32 ids[2]={row.header().bodyA,row.header().bodyB};PxTransform start[2],end[2];
                    for(PxU32 s=0;s<2;++s)
                    {
                        start[s]=ids[s]<mBodies.size()?PxTransform(mBodies[ids[s]].pStart,mBodies[ids[s]].qStart):domain.fixedPose[s];
                        end[s]=ids[s]<mBodies.size()?PxTransform(bodies[ids[s]].p,bodies[ids[s]].q):domain.fixedPose[s];
                    }
                    bool accepted=false;AvbdNpQuery query;
                    AvbdNpDomainArgs args={&domain,start,end,&accepted,&query};mKernels.launch(AVBD_NP_DOMAIN,&args);
                    if(!accepted && !rejectRow(i))return false;
                }
                else if(!avbdNpCandidateAdmissible(row,mNpDomains,mBodies,bodies,mKernels))return false;
                tested=1;
            }
            else ++mStep.warmDomainReuses;
        }
        if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
        {
            if(!tetrahedronPath(row,mTetrahedronMaterials[row.header().geometryIndex-1],mRowSupports,mBodies,bodies) && !rejectRow(i)) return false;
            continue;
        }
		if((row.header().flags&(eAVBD_ROW_CONTACT|eAVBD_ROW_INACTIVE))!=eAVBD_ROW_CONTACT) continue;
		if(row.header().supportCount)
		{
			if(!avbdHeterogeneousCandidateAdmissible(row,mRowSupports,mBlocks,mBodies,
				blocks,bodies,contactGeometry())) return false;
		}
	}
    mLocalTranslationClosed.clear();
    if(mStep.localBudget && !onlyNodal && !components.empty() && !mBodies.empty())
    {
        PxArray<PxU8> open(components.size(),0),material(components.size(),0);
        // Test the initial affine invariant without moving any pose. A
        // rejected predictor may still have the same total translation (e.g.
        // a zero-total-momentum impact). The bound covers PxReal coordinates;
        // it does not relax contact/material admission or published gates.
        PxArray<PxVec3d> defect(components.size(),PxVec3d(0)),roundoff(components.size(),PxVec3d(0));
        for(PxU32 b=0;b<mBodies.size();++b)if(avbdBodyResponds(mBodies[b]))
        {
            const PxU32 root=components[b];const double mass=1.0/double(mBodies[b].invMass);
            const auto initial=rejected[root]?mBodies[b].p:poses[b].p;
            defect[root]+=(avbdOgcDouble(initial)-avbdOgcDouble(mBodies[b].pHat))*mass;
            for(PxU32 d=0;d<3;++d)roundoff[root][d]+=mass*PX_EPS_F32*(std::abs(double(initial[d]))+std::abs(double(mBodies[b].pHat[d])));
        }
        for(PxU32 b=0;b<components.size();++b)for(PxU32 d=0;d<3;++d)
            open[b] |= std::abs(defect[b][d])>roundoff[b][d];
        for(PxU32 b=0;b<mBodies.size();++b)
            if(!avbdBodyResponds(mBodies[b]) || (mBodies[b].core && (PxU32(mBodies[b].core->lockFlags)&7)))
                open[components[b]]=1;
        for(PxU32 r=0;r<mRows.size();++r)
        {
            const auto& row=mRows[r];const auto& h=row.header();
            if((h.flags&eAVBD_ROW_INACTIVE) && h.kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT && h.kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON)continue;
            const bool supported=localTranslationInvariant(row);
            bool external=!supported;
            if(h.kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME || h.kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
                mRowProgram.forEachBlock(r,[&](PxU32 b){material[components[b]]=1;});
            const auto check=[&](PxU32 b){external |= b>=mBodies.size() || !avbdBodyResponds(mBodies[b]);};
            if(h.supportCount)for(PxU32 k=0;k<h.supportCount;++k)check(mRowSupports[h.supportStart+k].block);
            else {check(h.bodyA);check(h.bodyB);}
            if(external)mRowProgram.forEachBlock(r,[&](PxU32 b){open[components[b]]=1;});
        }
        // A subspace chart is valid only if the bounded group candidates can
        // connect its full dynamic support. Otherwise a high-degree singleton
        // could acquire a permanent artificial translation lock.
        PxArray<PxU32> movable,anchor(components.size(),PX_MAX_U32);
        mRowProgram.components(movable,[&](PxU32 b){return b<mBodies.size() && avbdBodyResponds(mBodies[b]);},
            [&](PxU32 r)
            {
                const auto& h=mRows[r].header();
                if(h.kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT && h.kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME && h.kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON)return false;
                if((h.flags&eAVBD_ROW_INACTIVE) && h.kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT)return false;
                PxU32 count=0,references=0;
                mRowProgram.forEachBlock(r,[&](PxU32 b){if(b<mBodies.size() && avbdBodyResponds(mBodies[b]))
                    {++count;references+=mRowProgram.blockOffsets[b+1]-mRowProgram.blockOffsets[b];}});
                return count>=2 && count<=4 && references<=256;
            });
        for(PxU32 b=0;b<mBodies.size();++b)if(avbdBodyResponds(mBodies[b]))
        {
            const PxU32 root=components[b];
            if(anchor[root]==PX_MAX_U32)anchor[root]=movable[b];
            else if(anchor[root]!=movable[b])open[root]=1;
        }
        mLocalTranslationClosed.resize(mBodies.size(),0);
        bool any=false;
        for(PxU32 b=0;b<mBodies.size();++b)
        {mLocalTranslationClosed[b]=!open[components[b]] && material[components[b]];any |= mLocalTranslationClosed[b]!=0;}
        if(!any)mLocalTranslationClosed.clear();
        if(std::getenv("PHYSX_AVBD_COMMON_STATS"))
        {
            PxU32 count=0,unsupportedRows=0;
            for(PxU8 b:mLocalTranslationClosed)count+=b?1:0;
            for(const auto& row:mRows)if(!(row.header().flags&eAVBD_ROW_INACTIVE) && !localTranslationInvariant(row))++unsupportedRows;
            std::fprintf(stderr,"[AVBD_LOCAL_TRANSLATION] closedBodies=%u unsupportedRows=%u bodies=%u\n",count,unsupportedRows,mBodies.size());
        }
    }
    PxArray<PxU32> unresolved;
    const auto recheckNodal=[&](const PxArray<PxU32>& failed,const PxArray<PxU32>& changed,const auto& onFailure)
    {
        PxArray<PxU8> tested(mRows.size(),0);
        const auto certify=[&](PxU32 index)
        {
            if(tested[index])return;
            tested[index]=1;
            const auto row=mRows[index];
            if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
            {
                ++mStep.warmDomainTests;
                if(!avbdNodalContactPath(mNodalContacts[row.header().geometryIndex-1],candidatePoint,bodies))onFailure(index);
            }
            else if(avbdIsNodalMaterial(row.header().kernelId) && !warmMaterialAdmissible(row,candidatePoint))onFailure(index);
        };
        for(PxU32 index:failed)certify(index);
        for(PxU32 block:changed)
            for(PxU32 j=mRowProgram.blockOffsets[block];j<mRowProgram.blockOffsets[block+1];++j)
                certify(mRowProgram.blockRows[j]);
    };
    if(!movingBoxes.empty() || !changedBlocks.empty())
    {
        if(normalTravel.empty())normalTravel.resize(mBlocks.size(),PxVec3(0));
        for(const PxU32 rigid:movingBoxes)
        {
            poses[rigid].p+=boxShift[rigid];
            changedBlocks.pushBack(rigid);
            const auto& body=mBodies[rigid];
            const auto rotation=poses[rigid].q*body.qStart.getConjugate();
            for(PxU32 j=mRowProgram.blockOffsets[rigid];j<mRowProgram.blockOffsets[rigid+1];++j)
            {
                const auto& h=mRows[mRowProgram.blockRows[j]].header();
                if(h.kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT)continue;
                const auto& contact=mNodalContacts[h.geometryIndex-1];
                const PxU32 side=contact.geometry->deformable[0]?0:1;
                const auto& mesh=contact.geometry->deformable[side]->state->mesh;
                const auto& feature=*contact.feature;
                const bool vertex=(feature.vertexMask&(1u<<side))!=0;
                // Transport the separating direction and material point
                // together for this initial guess. The NP path certificate
                // still tests each complete finite feature.
                const auto n=rotation.rotate(side?-feature.normal:feature.normal);
                for(PxU32 v=0;v<(vertex?1u:mesh.arity);++v)
                {
                    const PxU32 id=vertex?feature.feature[side]:mesh.vertices[mesh.arity*feature.feature[side]+v];
                    const auto& embedded=mesh.supports[id];
                    for(PxU32 k=0;k<embedded.count;++k)
                    {
                        if(embedded.weights[k]==0)continue;
                        const PxU32 block=contact.first[side]+embedded.nodes[k];
                        const auto& p=*mBlocks[block].particle;
                        if(p.invMass<=0)continue;
                        const auto carried=poses[rigid].p+rotation.rotate(p.pStart-body.pStart);
                        const auto inward=n.dot(carried-candidatePoint(block));
                        auto& travel=normalTravel[block];
                        if(inward>0 && inward*inward>travel.magnitudeSquared())
                        {
                            if(travel.isZero())changedBlocks.pushBack(block);
                            travel=n*inward;
                        }
                    }
                }
            }
        }
        localized=true;
        // Unchanged certified rows still describe the identical trajectory.
        // Recheck the union of failed rows and the changed blocks' CSR stars.
        recheckNodal(failedNodalRows,changedBlocks,[&](PxU32 i){unresolved.pushBack(i);});
    }
    else
        unresolved.swap(failedNodalRows);

    // Admit a spatially varying fraction of the inertial predictor near an
    // obstructed sheet patch. The objective, force rows and BDF stay fixed.
    struct Patch
    {
        PxBounds3 bounds=PxBounds3::empty();
        PxReal fraction=1;
    };
    PxArray<Patch> patches;
    const bool affineSheet=canCondition && mBodies.empty() && mNodalVolumeMaterials.empty();
    for(PxU32 i:unresolved)
    {
        if(!rejectRow(i))return false;
        if(!affineSheet)continue;
        const auto row=mRows[i];
        const PxReal f=row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT
            ? avbdSheetBoxPredictorFraction(mNodalContacts[row.header().geometryIndex-1],candidatePoint) : 0.f;
        if(patches.empty())patches.resize(components.size());
        PxBounds3 bounds=PxBounds3::empty();
        mRowProgram.forEachBlock(i,[&](PxU32 b){bounds.include(mBlocks[b].particle->pStart);});
        mRowProgram.forEachBlock(i,[&](PxU32 b)
        {
            if(mBlocks[b].particle->invMass<=0)return;
            Patch& patch=patches[components[b]];
            patch.bounds.include(bounds);patch.fraction=PxMin(patch.fraction,f);
        });
    }
    if(!patches.empty())
    {
        changedBlocks.clear();
        for(PxU32 b=particleStart;b<mBlocks.size();++b)
        {
            if(!mBlocks[b].particle)continue;
            const auto& p=*mBlocks[b].particle;
            const Patch& patch=patches[components[b]];
            if(p.invMass<=0 || patch.fraction<=0 || patch.fraction>=1)continue;
            // Use the plane-stress longitudinal speed and nominal areal
            // density only to shape the seed. Outside one step of elastic
            // travel from the affected region, keep the full inertial motion.
            const auto& sheet=mIslandSurfaceCache[mBlocks[b].objectIndex];
            const PxReal reach=mDt*PxSqrt(sheet.longitudinalAreaModulus*p.invMass);
            const PxVec3 outside=(p.pStart-patch.bounds.maximum).maximum(patch.bounds.minimum-p.pStart).maximum(PxVec3(0));
            const PxReal distance=outside.magnitude();
            if(!(reach>0) || distance>=reach)continue;
            const PxReal t=patch.fraction+(1-patch.fraction)*(distance/reach);
            initialPoints[b-particleStart]=p.pStart+(candidatePoint(b)-p.pStart)*t;
            if(localized)normalTravel[b]=PxVec3(0);
            changedBlocks.pushBack(b);
        }
        if(!changedBlocks.empty())
        {
            for(PxU32 b:changedBlocks)rejected[components[b]]=0;
            // All previously failed rows plus every row reading a changed
            // node must accept the assembled, rounded candidate together.
            recheckNodal(unresolved,changedBlocks,[&](PxU32 i){rejectRow(i);});
        }
    }
    for(const auto& seed : attachmentSeeds)
        if(components.empty() || !rejected[components[seed.rigid]])
            for(PxU32 k=0;k<3;++k) mRows[seed.first+k].scalar().lambda=seed.force[k];
	// Publish the admitted initial iterate. The inertial targets remain fixed.
	PxU32 componentCount=0,rejectedCount=0,acceptedBodies=0,rejectedBodies=0;
	for(PxU32 i=0;i<mBodies.size();++i)
	{
		const bool veto=!components.empty() && rejected[components[i]];
		if(!components.empty() && avbdBodyResponds(mBodies[i]))
		{
			if(components[i]==i){++componentCount;rejectedCount+=veto?1:0;}
			if(veto)++rejectedBodies;else ++acceptedBodies;
		}
		if(!veto){mBodies[i].p=bodies[i].p;mBodies[i].q=bodies[i].q;}
	}
	if(!components.empty() && std::getenv("PHYSX_AVBD_COMMON_STATS"))std::fprintf(stderr,
		"[AVBD_LOCAL_GUESS] components=%u rejectedComponents=%u acceptedBodies=%u rejectedBodies=%u rowTests=%u\n",
		componentCount,rejectedCount,acceptedBodies,rejectedBodies,rowCount);
	for(PxU32 i=particleStart;i<mBlocks.size();++i)
		if(mBlocks[i].type==eAVBD_BLOCK_PARTICLE3 && mBlocks[i].particle &&
            (components.empty() || !rejected[components[i]]))
			mBlocks[i].particle->p=candidatePoint(i);
	return true;
}

void AvbdDynamicsContext::traceLocalMomentum(const char* stage,bool output) const
{
    if(!std::getenv("PHYSX_AVBD_MOMENTUM_TRACE"))return;
    PxVec3d input(0),hat(0),pose(0),velocity(0);double mass=0;
    for(const auto& b:mBodies)if(avbdBodyResponds(b))
    {
        const double m=1.0/double(b.invMass);mass+=m;
        input+=avbdOgcDouble(b.linVel)*m;
        hat+=(avbdOgcDouble(b.pHat)-avbdOgcDouble(b.pStart))*(m/double(mDt));
        pose+=(avbdOgcDouble(b.p)-avbdOgcDouble(b.pStart))*(m/double(mDt));
        if(output)velocity+=avbdOgcDouble(b.outputLinVel)*m;
    }
    for(const auto& block:mBlocks)if(block.particle && block.particle->invMass>0)
    {
        const auto& p=*block.particle;const double m=1.0/double(p.invMass);mass+=m;
        if(!output)input+=avbdOgcDouble(p.vel)*m;
        hat+=(avbdOgcDouble(p.pHat)-avbdOgcDouble(p.pStart))*(m/double(mDt));
        pose+=(avbdOgcDouble(p.p)-avbdOgcDouble(p.pStart))*(m/double(mDt));
        if(output)velocity+=avbdOgcDouble(p.vel)*m;
    }
    std::fprintf(stderr,"[AVBD_MOMENTUM_STAGE] stage=%s iteration=%u mass=%.12g input=%.12g,%.12g,%.12g hat=%.12g,%.12g,%.12g pose=%.12g,%.12g,%.12g output=%.12g,%.12g,%.12g\n",
        stage,mStep.iteration,mass,input.x,input.y,input.z,hat.x,hat.y,hat.z,pose.x,pose.y,pose.z,velocity.x,velocity.y,velocity.z);
}

void AvbdDynamicsContext::warmstartBlocks()
{
    if(mStep.localBudget)
    {
        if(!mStep.rowProgramValid)compileRowProgram();
        if(mLocalDomainFailed || !localCapability()){mLocalDomainFailed=1;return;}
        mStep.rowProgramValid=true;
    }
    const char* mode=std::getenv("PHYSX_AVBD_COMMON_STEP");
    if(mode && std::strcmp(mode,"patch")==0 && !mStep.hasParticleBlocks)
    {
        compileRowProgram();if(mLocalDomainFailed)return;
        PxArray<PxU32> components;const auto responds=[&](PxU32 b){return b<mBodies.size() && avbdBodyResponds(mBodies[b]);};
        mRowProgram.components(components,responds,[&](PxU32 r){return !(mRows[r].header().flags&eAVBD_ROW_INACTIVE);});
        for(PxU32 b=0;b<components.size();++b)if(components[b]==b && responds(b))
        {
            AvbdProgramSubset subset;
            for(PxU32 k=0;k<components.size();++k)if(components[k]==b && responds(k))subset.blocks.pushBack(k);
            subset.collectRows(mRowProgram);PxReal fraction=1.f;bool accepted=false;
            for(PxU32 search=0;search<16;++search,fraction*=.5f)
                if(admitWarmstartCandidate(&subset,fraction)){accepted=true;break;}
            if(std::getenv("PHYSX_AVBD_COMMON_STATS"))std::fprintf(stderr,
                "[AVBD_PATCH_GUESS] blocks=%u fraction=%.9g accepted=%u\n",subset.blocks.size(),double(fraction),unsigned(accepted));
        }
        return;
    }
    // Try the common inertial guess without publishing a partial state.
    // If it is inadmissible, retain the DCD poses as the initial iterate.
    // Projecting pHat block by block can inject elastic energy before the
    // energy-consistent primal solve. Normal GS still solves the same pHat.
    (void)admitWarmstartCandidate();
    traceLocalMomentum("initial");
}

// Heterogeneous (particle-supported) contact normal is the OGC offset
// potential shared with the 3x3 particle blocks. The returned coefficient is
// dE/dd (negative inside the offset shell), in AvbdRow's force sign
// convention. A finite maxImpulse saturates the potential derivative and
// therefore removes the local curvature once the authored limit is reached.
static PxReal avbdJointEndpointMobility(const AvbdSolverBody& b,
	const PxVec3& jLin, const PxVec3& jAng)
{
	if(!avbdBodyResponds(b))
		return 0.0f;
	const PxVec3 jAngLocal = b.qStart.rotateInv(jAng);
	return b.invMass * jLin.magnitudeSquared() +
		jAngLocal.x * jAngLocal.x * b.invInertia.x +
		jAngLocal.y * jAngLocal.y * b.invInertia.y +
		jAngLocal.z * jAngLocal.z * b.invInertia.z;
}

static PxReal avbdDrivePenalty(const AvbdRowConstRef& row,
	const PxArray<AvbdSolverBody>& bodies)
{
	PxReal penalty = row.scalar().rho;
	if(!row.scalar().accelerationDrive)
		return penalty;
	PxReal mobility = 0.0f;
	if(row.header().bodyA < bodies.size())
		mobility += avbdJointEndpointMobility(
			bodies[row.header().bodyA], row.scalar().jLinA, row.scalar().jAngA);
	if(row.header().bodyB < bodies.size())
		mobility += avbdJointEndpointMobility(
			bodies[row.header().bodyB], row.scalar().jLinB, row.scalar().jAngB);
	return mobility > 1.0e-12f && PxIsFinite(mobility) ?
		penalty / mobility : penalty;
}

static PxReal avbdHardJointPenalty(const AvbdRowConstRef& row,
	const PxArray<AvbdSolverBody>& bodies, PxReal invH2,
	PxReal* dualStep = NULL)
{
	if(dualStep)
		*dualStep = row.scalar().rho;
	if(row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_DRIVE |
		eAVBD_ROW_INACTIVE | eAVBD_ROW_SOFT_PAIR))
		return row.scalar().rho;
	if(row.header().flags & eAVBD_ROW_POINT_LOCK)
	{
		// AVBD_2 LocalJointSolve + JointDualPhase form a matched pair:
		// a stiff primal anchor with an inertial-scale, relaxed dual step.
		// Applying the full primal penalty to dual destabilizes accelerated
		// cyclic point-lock graphs. Static endpoints do not contribute mass.
		PxReal minMass = PX_MAX_F32, maxMass = 0.0f;
		const PxU32 endpoints[2] = {row.header().bodyA, row.header().bodyB};
		for(PxU32 side = 0; side < 2; ++side)
		{
			if(endpoints[side] >= bodies.size() || !avbdBodyResponds(bodies[endpoints[side]]))
				continue;
			const PxReal mass = 1.0f / bodies[endpoints[side]].invMass;
			minMass = PxMin(minMass, mass);
			maxMass = PxMax(maxMass, mass);
		}
		if(maxMass > 0.0f)
		{
			const PxReal massH2 = minMass * invH2;
			if(dualStep)
				*dualStep = PxMin(massH2, row.scalar().rho * (row.scalar().rho / (row.scalar().rho + massH2)));
			return PxMax(row.scalar().rho, maxMass * invH2);
		}
	}
	// Other solverPrep/articulation 1D rows retain the demo3d update:
	// one penalty shared by both endpoints and dual, with a spatial effective
	// mass floor 1 / (h^2 J M^-1 J^T). Their adaptive rho is independent of
	// the fixed point-lock policy above.
	PxReal mobility = 0.0f;
	if(row.header().bodyA < bodies.size())
		mobility += avbdJointEndpointMobility(bodies[row.header().bodyA], row.scalar().jLinA, row.scalar().jAngA);
	if(row.header().bodyB < bodies.size())
		mobility += avbdJointEndpointMobility(bodies[row.header().bodyB], row.scalar().jLinB, row.scalar().jAngB);
	PxReal penalty = row.scalar().rho;
	if(PxIsFinite(mobility) && mobility > 1.0e-12f)
	{
		const PxReal massH2 = invH2 / mobility;
		if(PxIsFinite(massH2) && massH2 > 0.0f)
			penalty = PxMax(row.scalar().rho, massH2);
	}
	if(dualStep)
		*dualStep = penalty;
	return penalty;
}

static void initializeRigidMaterial(const AvbdSolverBody& b,PxReal H[6][6],PxReal g[6],const AvbdBodyPose* candidate=NULL)
{
	for(PxU32 rr = 0; rr < 6; ++rr)
	{
		g[rr] = 0.0f;
		for(PxU32 cc = 0; cc < 6; ++cc)
			H[rr][cc] = 0.0f;
	}
	H[0][0] = H[1][1] = H[2][2] = b.massInvH2;
	const AvbdRotationPotential rotation =
		avbdRotationInertialPotential(b.qHat, candidate?candidate->q:b.q, b.inertiaWorldInvH2);
	for(PxU32 rr = 0; rr < 3; ++rr)
		for(PxU32 cc = 0; cc < 3; ++cc)
			H[rr + 3][cc + 3] = rotation.gaussNewton[rr][cc];

	const PxVec3 dp = (candidate?candidate->p:b.p) - b.pHat;
	g[0] = b.massInvH2 * dp.x;
	g[1] = b.massInvH2 * dp.y;
	g[2] = b.massInvH2 * dp.z;

	const PxVec3 ga = rotation.gradient;
	g[3] = ga.x;
	g[4] = ga.y;
	g[5] = ga.z;

}

bool AvbdDynamicsContext::accumulateRigidMaterialRow(PxU32 ri,
    const PxU32* blocks,PxU32 count,AvbdRigidAssembly* assemblies)
{
    const AvbdRowConstRef& row=mRows[ri];
    if(row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME)
    {
        const auto e=evaluateReferenceFrame(row,mElasticFrameMaterials,mBodies,mKernels);
        if(!e.valid){PxAtomicExchange(&mLocalDomainFailed,1);return true;}
        const PxU32 endpoints[2]={row.header().bodyA,row.header().bodyB};
        avbdAssembleElasticFrame(ri,e,endpoints,blocks,count,assemblies);return true;
    }
    if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
    {
        const auto e=evaluateTetrahedron(row);
        if(!e.valid){PxAtomicExchange(&mLocalDomainFailed,1);return true;}
        PxU32 supports[4];PxVec3 arms[4];
        for(PxU32 s=0;s<4;++s)
        {
            const auto& support=mRowSupports[row.header().supportStart+s];supports[s]=support.block;
            arms[s]=support.block<mBodies.size()?mBodies[support.block].q.rotate(support.localPoint):PxVec3(0);
        }
        avbdAssembleTetrahedron(ri,e,supports,arms,blocks,count,assemblies);return true;
    }
    return false;
}

bool AvbdDynamicsContext::compileCommonProgram(AvbdCommonProgram& p,const char*& reason,const AvbdRigidPoseView* candidate,
    const AvbdProgramSubset* subset,bool buildH,bool preserveCurvature)
{
    const AvbdRigidPoseView current={mBodies,NULL,NULL,0};const auto& poses=candidate?*candidate:current;
    reason="invalid_input";
    for(PxU32 b=0;b<mBlocks.size();++b)if(mBlocks[b].type!=eAVBD_BLOCK_RIGID6)
    {reason="particle_material_uncompiled";return false;}
    PxArray<PxU32> index(mBodies.size(),PX_MAX_U32);
    const PxU32 blockCount=subset?subset->blocks.size():mBodies.size();
    const PxU32 rowCount=subset?subset->rows.size():mRows.size();
    if(rowCount>PX_MAX_U32/3){reason="numeric_program_extent";return false;}
    // The row program supplies a finite upper bound. Avoid repeatedly moving
    // the complete contact arrays as their geometric capacity doubles.
    p.dimensions.reserve(blockCount);p.sourceBlocks.reserve(blockCount);p.base.reserve(6*blockCount);
    p.sourceRows.reserve(rowCount);p.mu.reserve(rowCount);p.coulomb.reserve(rowCount);
    p.offset.reserve(3*rowCount);p.lower.reserve(3*rowCount);p.upper.reserve(3*rowCount);p.compliance.reserve(3*rowCount);
    p.J.reserve(2*rowCount);
    for(PxU32 slot=0;slot<blockCount;++slot)
    {
        const PxU32 b=subset?subset->blocks[slot]:slot;if(!avbdBodyResponds(mBodies[b]))continue;
        index[b]=p.dimensions.size();p.dimensions.pushBack(6);p.sourceBlocks.pushBack(b);
        const auto pose=poses[b];
        const auto& body=mBodies[b];const PxVec3d dp=avbdOgcDouble(pose.p)-avbdOgcDouble(body.pHat);
        const PxMat33d inertia(avbdOgcDouble(body.inertiaWorldInvH2.column0),
            avbdOgcDouble(body.inertiaWorldInvH2.column1),avbdOgcDouble(body.inertiaWorldInvH2.column2));
        const auto rotation=avbdRotationInertialPotential<double>(avbdNpRotation(body.qHat),avbdNpRotation(pose.q),inertia);
        for(PxU32 d=0;d<3;++d)
        {
            p.energyPairs.pushBack({dp[d],double(body.massInvH2)*dp[d]});
            p.energyPairs.pushBack({rotation.coordinate[d],rotation.conjugate[d]});
        }
        const PxU32 locked=mBodies[b].core?PxU32(mBodies[b].core->lockFlags):0;
        AvbdCommonHBlock entry={};entry.row=entry.column=index[b];
        for(PxU32 d=0;d<6;++d)
        {
            p.base.pushBack(locked&(1u<<d)?0:-(d<3?double(body.massInvH2)*dp[d]:rotation.gradient[d-3]));
            for(PxU32 k=0;k<6;++k)entry.values[6*d+k]=(locked&((1u<<d)|(1u<<k)))?(d==k?1:0):
                d<3?(d==k?double(body.massInvH2):0):k<3?0:rotation.gaussNewton[d-3][k-3];
        }
        if(buildH)p.H.pushBack(entry);
    }
    AvbdNpPathFeature cachedPath={};AvbdNpPathPolynomial pathPolynomial;PxU32 cachedDomain=PX_MAX_U32,cachedFeature=PX_MAX_U32;
    PxU32 tangentRow=PX_MAX_U32;PxVec3d tangentMotion(0),tangentAngular[2][2];
    for(PxU32 slot=0;slot<rowCount;++slot)
    {
        const PxU32 r=subset?subset->rows[slot]:slot;
        const AvbdRowConstRef& row=mRows[r];if(row.header().flags&eAVBD_ROW_INACTIVE)continue;
        if(row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME || row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
        {
            PxU32 ids[4],count=0;bool overflow=false;
            mRowProgram.forEachBlock(r,[&](PxU32 b)
            {if(b<index.size() && index[b]!=PX_MAX_U32){if(count<4)ids[count++]=b;else overflow=true;}});
            if(overflow){reason="material_support_overflow";return false;}
            const PxU32 rank=row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME?9:1,offsets[2]={0,rank};
            AvbdRigidAssemblyT<double> assembled[4]={};AvbdGroupFactor factors[9]={};
            double frameH[81]={};
            for(PxU32 s=0;s<count;++s)
            {
                auto& a=assembled[s];a.coupledRows=&r;a.count=1;a.member=s;a.factorOffsets=offsets;
                a.factors=factors;a.consistent=true;a.lockedMask=mBodies[ids[s]].core?PxU32(mBodies[ids[s]].core->lockFlags):0;
            }
            if(rank==9)
            {
                const auto e=evaluateReferenceFrame(row,mElasticFrameMaterials,poses,mKernels,buildH);
                if(!e.valid){reason="invalid_frame_material";return false;}
                if(preserveCurvature && buildH && p.metricH.empty())p.metricH=p.H;
                for(PxU32 k=0;k<FINITE_JOINT_ROWS;++k)p.energyPairs.pushBack({e.energyPairs[k][0],e.energyPairs[k][1]});
                const PxU32 ends[2]={row.header().bodyA,row.header().bodyB};avbdAssembleElasticFrame(r,e,ends,ids,count,assembled);
                if(preserveCurvature && buildH)for(PxU32 k=0;k<81;++k)frameH[k]=e.hessian[k];
            }
            else
            {
                PxVec3d points[4];tetrahedronPoints(row,mRowSupports,poses,points);
                AvbdTetrahedronEvaluation e;AvbdTetrahedronArgs args={&mTetrahedronMaterials[row.header().geometryIndex-1],points,&e};
                mKernels.launch(AVBD_TETRAHEDRON_MATERIAL,&args);if(!e.valid){reason="invalid_tet_material";return false;}
                p.materialEnergies.pushBack(e.energy);
                PxU32 supports[4];PxVec3 arms[4];
                for(PxU32 s=0;s<4;++s)
                {const auto& v=mRowSupports[row.header().supportStart+s];supports[s]=v.block;arms[s]=v.block<mBodies.size()?poses[v.block].q.rotate(v.localPoint):PxVec3(0);}
                avbdAssembleTetrahedron(r,e,supports,arms,ids,count,assembled);
            }
            for(PxU32 s=0;s<count;++s)
            {
                if(!assembled[s].consistent){reason="inconsistent_material";return false;}
                for(PxU32 d=0;d<6;++d)if(!(assembled[s].lockedMask&(1u<<d)))p.base[6*index[ids[s]]+d]-=assembled[s].gradient[d];
                for(PxU32 t=0;buildH && t<count;++t)
                {
                    AvbdCommonHBlock entry={};entry.row=index[ids[s]];entry.column=index[ids[t]];
                    // The PSD operator is the numerical metric. In the bounded
                    // native-frame path retain the original signed second
                    // variation in the model, including every cross-body term.
                    for(PxU32 f=0;f<rank;++f)for(PxU32 d=0;d<6;++d)for(PxU32 k=0;k<6;++k)
                        entry.values[6*d+k]+=double(factors[f].curvature)*factors[f].jacobian[s][d]*double(factors[f].jacobian[t][k]);
                    if(!p.metricH.empty())
                    {
                        p.metricH.pushBack(entry);
                        if(rank==9)
                        {
                            for(double& v:entry.values)v=0;
                            const PxU32 ends[2]={row.header().bodyA,row.header().bodyB};
                            for(PxU32 a=0;a<2;++a)if(ends[a]==ids[s])for(PxU32 b=0;b<2;++b)if(ends[b]==ids[t])
                                for(PxU32 d=0;d<6;++d)for(PxU32 k=0;k<6;++k)
                                    if(!(assembled[s].lockedMask&(1u<<d)) && !(assembled[t].lockedMask&(1u<<k)))
                                        entry.values[6*d+k]+=(a && d<3?-1:1)*(b && k<3?-1:1)*
                                            frameH[9*(d<3?d:d+3*a)+(k<3?k:k+3*b)];
                        }
                    }
                    p.H.pushBack(entry);
                }
            }
            continue;
        }
        if(row.header().supportCount || (row.header().kernelId!=eAVBD_ROW_KERNEL_CONSTRAINT &&
            row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT && row.header().kernelId!=eAVBD_ROW_KERNEL_FINITE_SPRING))
        {reason="row_kernel_uncompiled";return false;}
        const bool contact=(row.header().flags&eAVBD_ROW_CONTACT)!=0;
        if(contact && row.scalar().mu>0 && !row.scalar().positionTangent)
        {reason="velocity_contact_uncompiled";return false;}
        const PxU32 group=p.mu.size();const bool friction=contact && row.scalar().mu>0;
        p.sourceRows.pushBack(r);p.mu.pushBack(friction?row.scalar().mu:0);p.coulomb.pushBack(friction?1:0);
        const AvbdNpDomain* domain=row.header().kernelId==eAVBD_ROW_KERNEL_NP_CONTACT?&mNpDomains[row.header().geometryIndex-1]:NULL;
        PxVec3d normalAngular[2];double C=domain?0:avbdEvalC(row,poses);
        if(domain)
        {
            const PxTransform states[2]={row.header().bodyA<poses.size()?PxTransform(poses[row.header().bodyA].p,poses[row.header().bodyA].q):domain->fixedPose[0],
                row.header().bodyB<poses.size()?PxTransform(poses[row.header().bodyB].p,poses[row.header().bodyB].q):domain->fixedPose[1]};
            if(domain->rowStart==PX_MAX_U32)
            {
                PxVec3d arms[2];C=avbdNpGapGradient(*domain,states,arms);
                for(PxU32 s=0;s<2;++s)normalAngular[s]=arms[s].cross(domain->normal*(s?-1.:1.));
            }
            else
            {
                const PxTransform start[2]={row.header().bodyA<mBodies.size()?PxTransform(mBodies[row.header().bodyA].pStart,mBodies[row.header().bodyA].qStart):domain->fixedPose[0],
                    row.header().bodyB<mBodies.size()?PxTransform(mBodies[row.header().bodyB].pStart,mBodies[row.header().bodyB].qStart):domain->fixedPose[1]};
                const PxU32 feature=(r-domain->rowStart)/5,k=(r-domain->rowStart)%5;
                if(cachedDomain!=row.header().geometryIndex)pathPolynomial=AvbdNpPathPolynomial(*domain,start,states);
                if(cachedDomain!=row.header().geometryIndex || cachedFeature!=feature)
                {cachedPath=pathPolynomial.evaluate(*domain,feature);cachedDomain=row.header().geometryIndex;cachedFeature=feature;}
                if(!cachedPath.valid){reason="singular_np_path";return false;}
                C=cachedPath.gap[k];for(PxU32 s=0;s<2;++s)normalAngular[s]=cachedPath.angular[k][s];
            }
        }
        PxVec3d rel(0);
        if(contact)
        {
            // All five path facets at a support point share tangent anchors.
            // Compare the actual inputs so reuse also respects modified rows.
            if(friction)
            {
                const bool reuse=tangentRow!=PX_MAX_U32 && mRows[tangentRow].header().bodyA==row.header().bodyA && mRows[tangentRow].header().bodyB==row.header().bodyB &&
                    mRows[tangentRow].scalar().rA==row.scalar().rA && mRows[tangentRow].scalar().rB==row.scalar().rB && mRows[tangentRow].scalar().t0==row.scalar().t0 && mRows[tangentRow].scalar().t1==row.scalar().t1;
                if(!reuse)
                {
                    tangentRow=r;tangentMotion=PxVec3d(0);
                    const PxU32 ids[2]={row.header().bodyA,row.header().bodyB};const PxVec3 anchors[2]={row.scalar().rA,row.scalar().rB};
                    for(PxU32 s=0;s<2;++s)
                    {
                        tangentAngular[s][0]=tangentAngular[s][1]=PxVec3d(0);
                        if(ids[s]>=poses.size())continue;
                        const auto body=poses[ids[s]];const double sign=s?-1.:1.;
                        tangentMotion+=avbdNpPointMotion(body.p,body.q,body.pStart,body.qStart,anchors[s])*sign;
                        const PxVec3d arm=avbdNpRotation(body.q).rotate(avbdOgcDouble(anchors[s]));
                        tangentAngular[s][0]=arm.cross(avbdOgcDouble(row.scalar().t0)*sign);
                        tangentAngular[s][1]=arm.cross(avbdOgcDouble(row.scalar().t1)*sign);
                    }
                }
                rel=tangentMotion;
            }
            if(!domain)
            {
            PxVec3d points[2]={avbdOgcDouble(row.scalar().ogcRA),avbdOgcDouble(row.scalar().ogcRB)};
            const PxU32 ids[2]={row.header().bodyA,row.header().bodyB};const PxVec3 normals[2]={row.scalar().ogcRA,row.scalar().ogcRB};
            for(PxU32 s=0;s<2;++s)if(ids[s]<poses.size())
            {
                const auto body=poses[ids[s]];
                points[s]=avbdOgcDouble(body.p)+avbdNpRotation(body.q).rotate(avbdOgcDouble(normals[s]));
            }
            C=avbdOgcDouble(row.scalar().n).dot(points[0]-points[1])+double(row.scalar().contactOffset);
            }
        }
        const double stiffness=(row.header().flags&eAVBD_ROW_DRIVE)?double(avbdDrivePenalty(row,mBodies)):0;
        if((row.header().flags&eAVBD_ROW_DRIVE) && !(stiffness>0)) {reason="invalid_drive";return false;}
        // PX_MAX_REAL is PhysX's unbounded sentinel, not an authored finite
        // cap. Treating it as finite multiplies roundoff gap by ~1e38 in the
        // upper-bound complementarity check and can prevent termination.
        auto bound=[](PxReal v)->double
        {return v==PX_MAX_REAL?std::numeric_limits<double>::infinity():
            v==-PX_MAX_REAL?-std::numeric_limits<double>::infinity():double(v);};
        for(PxU32 q=0;q<3;++q)
        {
            p.offset.pushBack(q==0?C:(friction?avbdOgcDouble(q==1?row.scalar().t0:row.scalar().t1).dot(rel):0));
            p.compliance.pushBack(q==0 && stiffness>0?1/stiffness:0);
            p.lower.pushBack(q==0?(contact?0:-bound(row.scalar().maxLambda)):(friction?-std::numeric_limits<double>::infinity():0));
            p.upper.pushBack(q==0?-bound(row.scalar().minLambda):(friction?std::numeric_limits<double>::infinity():0));
        }
        bool valid=true;
        mRowProgram.forEachBlock(r,[&](PxU32 b)
        {
            if(b>=index.size()){valid=false;return;}if(index[b]==PX_MAX_U32)return;
            PxReal value;PxVec3 linear(0),angular(0);
            if(contact?(b!=row.header().bodyA && b!=row.header().bodyB):!linearizeRigidRow(row,b,value,linear,angular,&poses))
            {valid=false;return;}
            AvbdCommonJBlock entry={};entry.row=group;entry.column=index[b];
            const PxU32 locked=mBodies[b].core?PxU32(mBodies[b].core->lockFlags):0;
            for(PxU32 q=0;q<(friction?3u:1u);++q)
            {
                if(contact && (q || !domain))
                {
                    const bool a=row.header().bodyA==b;const PxVec3d n=avbdOgcDouble(q?(q==1?row.scalar().t0:row.scalar().t1):row.scalar().n)*(a?1.:-1.);
                    const PxVec3 anchor=q?(a?row.scalar().rA:row.scalar().rB):(a?row.scalar().ogcRA:row.scalar().ogcRB);
                    const PxVec3d torque=q?tangentAngular[a?0:1][q-1]:avbdNpRotation(poses[b].q).rotate(avbdOgcDouble(anchor)).cross(n);
                    for(PxU32 d=0;d<3;++d)
                    {entry.values[6*q+d]=(locked&(1u<<d))?0:n[d];entry.values[6*q+d+3]=(locked&(1u<<(d+3)))?0:torque[d];}
                    continue;
                }
                for(PxU32 d=0;d<3;++d)
                {entry.values[6*q+d]=(locked&(1u<<d))?0:linear[d];entry.values[6*q+d+3]=(locked&(1u<<(d+3)))?0:angular[d];}
                if(q==0 && domain)
                {
                    const PxU32 side=row.header().bodyA==b?0:1;const PxVec3d n=domain->normal*(side?-1.:1.);
                    const PxVec3d torque=normalAngular[side];
                    for(PxU32 d=0;d<3;++d)
                    {entry.values[d]=(locked&(1u<<d))?0:n[d];entry.values[d+3]=(locked&(1u<<(d+3)))?0:torque[d];}
                }
            }
            p.J.pushBack(entry);
        });
        if(!valid){reason="row_linearization_failed";return false;}
    }
    if(!p.finish()){reason="numeric_program_invalid";return false;}
    if(!p.metricH.empty())for(PxU32 e=0;e<p.H.size();++e)if(p.H[e].row==p.H[e].column)
        if(!avbdRegularizeLocalCurvature(p.H[e].values,p.metricBlock(e).values,p.dimensions[p.H[e].row]))
        {reason="invalid_local_curvature";return false;}

    reason="ok";return true;
}

bool AvbdDynamicsContext::commonCapability()
{
    for(const auto& block:mBlocks)if(block.type!=eAVBD_BLOCK_RIGID6)
    {std::fprintf(stderr,"[AVBD_COMMON_STEP] stage=compile reason=particle_material_uncompiled status=FAIL\n");return false;}
    const char* unsupported=NULL;
    if(mNpDomainUnsupported || mNpDomainUnseparated)unsupported="np_geometry_or_initial_recovery_uncompiled";
    for(const auto& domain:mNpDomains)if(domain.restDistance<0)unsupported="negative_rest_distance_uncompiled";
    for(const auto& row:mRows)if(row.hasScalar() && !(row.header().flags&eAVBD_ROW_INACTIVE) && row.scalar().restitution>0)unsupported="restitution_uncompiled";
    if(unsupported)
    {std::fprintf(stderr,"[AVBD_COMMON_STEP] stage=capability reason=%s status=FAIL\n",unsupported);return false;}
    return true;
}

bool AvbdDynamicsContext::solveCommonCandidate(bool& stationary)
{
    if(!commonCapability())return false;
    PxArray<PxU32> components;const auto responds=[&](PxU32 b)
    {return b<mBodies.size() && avbdBodyResponds(mBodies[b]);};
    const char* mode=std::getenv("PHYSX_AVBD_COMMON_STEP");
    if(mode && std::strcmp(mode,"patch")==0)
    {
        AvbdProgramSchedule schedule;
        schedule.compile(mRowProgram,responds,[&](PxU32 r){return !(mRows[r].header().flags&eAVBD_ROW_INACTIVE);},[&](PxU32 r)
        {return mRows[r].header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME || mRows[r].header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON;});
        stationary=true;PxU32 skipped=0;
        for(PxU32 c=0;c+1<schedule.colorOffsets.size();++c)
            for(PxU32 s=schedule.colorOffsets[c];s<schedule.colorOffsets[c+1];++s)
            {
                const PxU32 g=schedule.colorGroups[s];const auto& subset=schedule.groups[g];bool closed=false;
                const PxU32 idle=mPrimalIdleIteration[subset.blocks[0]];bool unchanged=idle!=0;
                for(PxU32 b:subset.blocks)unchanged=unchanged && mPrimalIdleIteration[b]==idle && mPrimalChangedIteration[b]<idle;
                for(PxU32 r:subset.rows)if(unchanged)
                {
                    const auto kernel=mRows[r].header().kernelId;
                    unchanged=kernel==eAVBD_ROW_KERNEL_ELASTIC_FRAME || kernel==eAVBD_ROW_KERNEL_TETRAHEDRON || kernel==eAVBD_ROW_KERNEL_NP_CONTACT;
                    mRowProgram.forEachBlock(r,[&](PxU32 b){unchanged=unchanged && mPrimalChangedIteration[b]<idle;});
                }
                if(unchanged){++skipped;continue;}
                if(!solveCommonComponent(schedule.groups[g],closed))
                {
                    if(std::getenv("PHYSX_AVBD_COMMON_STATS"))std::fprintf(stderr,
                        "[AVBD_PATCH_FAILURE] iteration=%u group=%u blocks=%u rows=%u firstBlock=%u\n",
                        mStep.iteration,g,schedule.groups[g].blocks.size(),schedule.groups[g].rows.size(),schedule.groups[g].blocks[0]);
                    return false;
                }
                for(PxU32 b:subset.blocks)mPrimalIdleIteration[b]=closed?mStep.iteration+1:0;
                stationary=stationary && closed;
            }
        if(stationary)
        {
            // Conditional patch equilibria alone cannot qualify a collective
            // weak mode. Only near a quiet sweep check the original full-H
            // correction, without a new force solve or a physical update.
            AvbdCommonProgram p;const char* reason;
            if(!compileCommonProgram(p,reason))return false;
            PxArray<double> force(p.offset.size()),tolerance(p.base.size());
            for(PxU32 r=0;r<p.sourceRows.size();++r)
            {const auto& row=mRows[p.sourceRows[r]];force[3*r]=-row.scalar().lambda;force[3*r+1]=-row.scalar().lambdaT0;force[3*r+2]=-row.scalar().lambdaT1;}
            for(PxU32 b=0;b<p.sourceBlocks.size();++b)for(PxU32 d=0;d<6;++d)
                tolerance[6*b+d]=d<3?getLengthScale()*AVBD_ARTIC_POSITION_REL_TOL:AVBD_ARTIC_ROTATION_TOL;
            const AvbdCommonResidual metric(p);stationary=metric.smallRoot(p,force.begin(),tolerance.begin());
            if(stationary)
            {
                AvbdLocalResponse response(p,mKernels);PxArray<double> rhs(p.base.size()),correction(p.base.size());
                p.transpose(force.begin(),rhs.begin());for(PxU32 d=0;d<rhs.size();++d)rhs[d]+=p.base[d];
                if(!response.material(rhs.begin(),correction.begin()))return false;
                for(PxU32 d=0;d<rhs.size();++d)stationary=stationary && std::abs(correction[d])<=tolerance[d];
            }
        }
        writeJointDualState();
        if(std::getenv("PHYSX_AVBD_COMMON_STATS"))std::fprintf(stderr,
            "[AVBD_PATCH_SWEEP] iteration=%u groups=%u colors=%u maxBlocks=%u skipped=%u stationary=%u\n",
            mStep.iteration,schedule.groups.size(),schedule.colorOffsets.size()-1,schedule.maximumBlocks,skipped,unsigned(stationary));
        return true;
    }
    mRowProgram.components(components,responds,[&](PxU32 r){return !(mRows[r].header().flags&eAVBD_ROW_INACTIVE);});
    // A closed, disconnected component with step-constant payloads cannot
    // change while another component iterates. Reuse the existing idle state;
    // solverPrep/artic coordinates conservatively reevaluate after refresh.
    PxArray<PxU8> constant(components.size(),1);
    for(PxU32 r=0;r<mRows.size();++r)
        if(mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT && mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME &&
            mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON)
            mRowProgram.forEachBlock(r,[&](PxU32 b){if(responds(b))constant[components[b]]=0;});
    stationary=true;PxU32 count=0,skipped=0;
    for(PxU32 b=0;b<components.size();++b)if(components[b]==b && responds(b))
    {
        ++count;bool idle=constant[b]!=0;
        for(PxU32 k=0;k<components.size() && idle;++k)if(components[k]==b && responds(k))idle=mPrimalIdleIteration[k]!=0;
        if(idle){++skipped;continue;}
        AvbdProgramSubset subset;
        for(PxU32 k=0;k<components.size();++k)if(components[k]==b && responds(k))subset.blocks.pushBack(k);
        subset.collectRows(mRowProgram);
        bool closed=false;if(!solveCommonComponent(subset,closed))return false;
        stationary=stationary && closed;
        for(PxU32 k=0;k<components.size();++k)if(components[k]==b && responds(k))mPrimalIdleIteration[k]=closed?mStep.iteration+1:0;
    }
    writeJointDualState();
    if(std::getenv("PHYSX_AVBD_COMMON_STATS"))
        std::fprintf(stderr,"[AVBD_COMMON_COMPONENTS] iteration=%u count=%u skipped=%u stationary=%u\n",mStep.iteration,count,skipped,unsigned(stationary));
    return true;
}

// Numerical and candidate scratch for ONE existing row program. The pending
// request permits the Host continuation to yield without retaining stack data.
// Original physical poses and duals remain owned by mBodies/mRows until admission.
namespace physx { namespace Dy {
struct AvbdCommonPublication : public PxUserAllocated
{
    enum Phase {Direction,SecondOrder,Correction,Trial,Admission,DirectionCertificate};
    Phase phase=Direction;
    AvbdProgramSubset subset;
    AvbdCommonProgram p,actual,linear;
    AvbdCommonResidual metric;
    AvbdCommonResponse* reference=NULL;
    AvbdBoundedResponse* linearResponse=NULL;
    AvbdRowWorkingSet workingRows;
    bool expandingRows=false;
    PxArray<double> previous,force,step,trial,motionTolerance,rhs,correction,originalOffset;
    PxArray<double> uncorrectedStep,uncorrectedForce;
    PxArray<PxTransform> candidate;
    AvbdCoulombNcpResult result={},uncorrectedResult={};
    double initial=0,requested=0,fraction=1,lastMerit=std::numeric_limits<double>::infinity();
    double uncorrectedRequested=0,correctionRatio=0;
    PxU32 search=0,secondOrderSearch=PX_MAX_U32,uncorrectedGroups=0,correctionIndex=PX_MAX_U32;
    PxU32 activeGroups=0,materialProducts=0,numericalSweeps=0;
    bool patch=false,bounded=false,solved=false,valid=true,secondOrderAttempted=false;
    bool closed=false,rootSmall=false,finished=false;
    const char* rejection="none";
    AvbdCommonPublication():metric(p){}
    ~AvbdCommonPublication(){PX_DELETE(reference);PX_DELETE(linearResponse);}
};
} }

void AvbdDynamicsContext::clearCommonPublication()
{
    if(!mCommonPublication || mBoundedResponse!=mCommonPublication->linearResponse)PX_DELETE(mBoundedResponse);
    PX_DELETE(mCommonPublication);
    mBoundedResponse=NULL;mCommonPublication=NULL;mBoundedCursor=0;
}

bool AvbdDynamicsContext::beginCommonPublication(const AvbdProgramSubset& subset,bool bounded)
{
    clearCommonPublication();mCommonPublication=PX_NEW(AvbdCommonPublication)();
    auto& s=*mCommonPublication;s.subset=subset;s.bounded=bounded;
    const char* mode=std::getenv("PHYSX_AVBD_COMMON_STEP");s.patch=mode && std::strcmp(mode,"patch")==0;
    const char* reason="ok";auto& p=s.p;
    const PxU64 compileStart=mStep.profile?PxTime::getCurrentCounterValue():0;
    const bool compiled=compileCommonProgram(p,reason,NULL,&subset,true,bounded);
    if(mStep.profile)mStep.compileTicks+=PxTime::getCurrentCounterValue()-compileStart;
    if(!compiled)
    {std::fprintf(stderr,"[AVBD_COMMON_STEP] iteration=%u stage=compile reason=%s status=FAIL\n",mStep.iteration,reason);return false;}
    s.previous.resize(p.offset.size());s.force.resize(p.offset.size());
    for(PxU32 r=0;r<p.sourceRows.size();++r)
    {
        const auto& row=mRows[p.sourceRows[r]];
        s.previous[3*r]=-double(row.scalar().lambda);s.previous[3*r+1]=-double(row.scalar().lambdaT0);s.previous[3*r+2]=-double(row.scalar().lambdaT1);
        for(PxU32 q=0;q<3;++q)s.force[3*r+q]=(s.patch || bounded)?0:s.previous[3*r+q];
    }
    s.motionTolerance.resize(p.base.size());
    for(PxU32 b=0;b<p.sourceBlocks.size();++b)for(PxU32 d=0;d<6;++d)
        s.motionTolerance[6*b+d]=d<3?getLengthScale()*AVBD_ARTIC_POSITION_REL_TOL:AVBD_ARTIC_ROTATION_TOL;
    if(!bounded)
    {
        s.reference=PX_NEW(AvbdCommonResponse)(p,mKernels,s.patch || (mode && std::strcmp(mode,"local")==0));
        if(s.reference->local)s.reference->local->motionTolerance=s.motionTolerance;
    }
    s.metric=AvbdCommonResidual(p);s.initial=s.metric.evaluate(p,s.previous.begin());
    s.candidate.resize(s.patch?p.sourceBlocks.size():mBodies.size());s.trial.resize(s.force.size());
    return true;
}

bool AvbdDynamicsContext::solveCommonComponent(const AvbdProgramSubset& subset,bool& stationary)
{
    if(!beginCommonPublication(subset,false)){clearCommonPublication();return false;}
    auto& s=*mCommonPublication;bool ok=true;
    do
    {
        if(s.phase==AvbdCommonPublication::Correction)s.solved=s.reference->material(s.rhs.begin(),s.correction.begin());
        else s.solved=s.reference->solve(s.step,s.force,s.result,s.patch);
        s.valid=s.reference->valid;s.activeGroups=s.reference->activeGroups;s.materialProducts=s.reference->materialProducts;
        ok=resumeCommonPublication(stationary);
    } while(ok && !s.finished);
    clearCommonPublication();return ok;
}

bool AvbdDynamicsContext::beginBoundedCommon()
{
    if(!commonCapability())return false;
    clearCommonPublication();PxArray<PxU32> components;
    const auto responds=[&](PxU32 b){return b<mBodies.size() && avbdBodyResponds(mBodies[b]);};
    mRowProgram.components(components,responds,[&](PxU32 r){return !(mRows[r].header().flags&eAVBD_ROW_INACTIVE);});
    // CSR only schedules the existing connected row program. A static support
    // is an input, not a bridge that couples independent dynamic searches.
    auto& offsets=mStep.commonOffsets;offsets.resize(components.size()+1);
    for(auto& value:offsets)value=0;
    for(PxU32 b=0;b<components.size();++b)if(responds(b))++offsets[components[b]+1];
    for(PxU32 b=1;b<offsets.size();++b)offsets[b]+=offsets[b-1];
    mStep.commonBlocks.resize(offsets.back());PxArray<PxU32> cursor=offsets;
    for(PxU32 b=0;b<components.size();++b)if(responds(b))mStep.commonBlocks[cursor[components[b]]++]=b;
    mStep.commonConstant.resize(components.size());for(auto& value:mStep.commonConstant)value=1;
    for(PxU32 r=0;r<mRows.size();++r)
        if(mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT && mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME &&
           mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON)
            mRowProgram.forEachBlock(r,[&](PxU32 b){if(responds(b))mStep.commonConstant[components[b]]=0;});
    mStep.commonComponent=0;mStep.commonClosed=true;
    return nextBoundedCommonComponent();
}

bool AvbdDynamicsContext::nextBoundedCommonComponent()
{
    const auto& offsets=mStep.commonOffsets;
    for(;mStep.commonComponent+1<offsets.size();++mStep.commonComponent)
    {
        const PxU32 c=mStep.commonComponent,begin=offsets[c],end=offsets[c+1];if(begin==end)continue;
        bool idle=mStep.commonConstant[c]!=0;
        for(PxU32 i=begin;i<end && idle;++i)idle=mPrimalIdleIteration[mStep.commonBlocks[i]]!=0;
        if(idle)continue;
        AvbdProgramSubset subset;
        for(PxU32 i=begin;i<end;++i)subset.blocks.pushBack(mStep.commonBlocks[i]);
        subset.collectRows(mRowProgram);
        if(!beginCommonPublication(subset,true))return false;
        startBoundedResponse();return true;
    }
    return true;
}

void AvbdDynamicsContext::startBoundedResponse()
{
    const PxU64 setupStart=mStep.profile?PxTime::getCurrentCounterValue():0;
    auto& s=*mCommonPublication;
    if(mBoundedResponse!=s.linearResponse)PX_DELETE(mBoundedResponse);
    mBoundedResponse=NULL;
    if(s.phase==AvbdCommonPublication::Correction || s.phase==AvbdCommonPublication::DirectionCertificate)
    {
        // The original complete PSD metric defines the unchanged engineering
        // certificate; the model operator can retain signed material curvature.
        // Both requests use the identical bounded 3/6 block executor.
        if(s.linear.dimensions.empty())
        {s.linear.dimensions=s.p.dimensions;s.linear.sourceBlocks=s.p.sourceBlocks;
         s.linear.H=s.p.metricH.empty()?s.p.H:s.p.metricH;s.linear.base=s.rhs;s.linear.finish();}
        else s.linear.base=s.rhs;
        if(!s.linearResponse){s.linearResponse=PX_NEW(AvbdBoundedResponse)(s.linear);++mStep.responseBuilds;}
        else
        {
            ++mStep.responseResets;
            if(!s.linearResponse->restartHOnly())s.linearResponse->status=AvbdBoundedResponse::NumericalFailure;
        }
        mBoundedResponse=s.linearResponse;
    }
    else
    {
        if(s.p.mu.empty())mBoundedResponse=PX_NEW(AvbdBoundedResponse)(s.p,s.force.begin());
        else
        {
            auto& rows=s.workingRows;
            const bool expanding=s.expandingRows;
            if(!expanding)rows.initialize(s.p,s.force.begin(),s.previous.begin());
            s.expandingRows=false;
            const bool compiled=rows.compile(s.p,s.force.begin());
            mBoundedResponse=PX_NEW(AvbdBoundedResponse)(rows.active,rows.seed.begin(),&s.p,&rows.map,
                AvbdBoundedResponse::SweepLimit-rows.sweeps,expanding?s.step.begin():NULL,&rows.bounds);
            if(!compiled)mBoundedResponse->status=AvbdBoundedResponse::Unsupported;
        }
        ++mStep.responseBuilds;
    }
    if(mStep.profile)
    {mStep.responseStart=PxTime::getCurrentCounterValue();mStep.setupTicks+=mStep.responseStart-setupStart;}
    mBoundedCursor=0;
}

bool AvbdDynamicsContext::completeBoundedCommon(bool& done,bool& stationary)
{
    if(!mBoundedResponse){done=true;stationary=mStep.commonClosed;writeJointDualState();return true;}
    auto& s=*mCommonPublication;const auto& response=*mBoundedResponse;done=false;
    if(mStep.profile)mStep.responseTicks+=PxTime::getCurrentCounterValue()-mStep.responseStart;
    s.solved=response.status==AvbdBoundedResponse::Converged;
    s.valid=response.status!=AvbdBoundedResponse::NumericalFailure && response.status!=AvbdBoundedResponse::Unsupported;
    s.numericalSweeps+=response.sweeps;
    mStep.boundedSweeps+=response.sweeps;
    if(std::getenv("PHYSX_AVBD_COMMON_STATS"))std::fprintf(stderr,
        "[AVBD_BOUNDED_RESPONSE] iteration=%u phase=%u blocks=%u rows=%u colors=%u sweeps=%u error=%.9g status=%s completeRows=%u expansions=%u\n",
        mStep.iteration,unsigned(s.phase),response.p.dimensions.size(),response.p.mu.size(),response.colorCount,
        response.sweeps,response.error,response.statusName(),response.complete?response.complete->mu.size():response.p.mu.size(),
        response.complete?s.workingRows.expansions:0);
    if(response.complete)
    {
        auto& rows=s.workingRows;rows.sweeps+=response.sweeps;
        if(s.valid)rows.scatter(response.force,s.force);
        if(response.status==AvbdBoundedResponse::RowsRequired && rows.sweeps<AvbdBoundedResponse::SweepLimit && rows.grow(response.requests))
        {
            s.step=response.step;s.expandingRows=true;
            startBoundedResponse();return true;
        }
        if(response.status==AvbdBoundedResponse::RowsRequired)s.solved=false;
    }
    if(!s.solved && s.phase!=AvbdCommonPublication::Correction && s.phase!=AvbdCommonPublication::DirectionCertificate)
        if(const char* path=std::getenv("PHYSX_AVBD_COMMON_FAILED_NCP"))
        {FILE* file=std::fopen(path,"wb");if(file){s.p.write(file,response.complete?&s.force:&response.force,&s.motionTolerance);std::fclose(file);}}
    if(s.phase==AvbdCommonPublication::Correction || s.phase==AvbdCommonPublication::DirectionCertificate)s.correction=response.step;
    else
    {
        s.step=response.step;if(!response.complete)s.force=response.force;s.activeGroups=0;
        for(PxU8 active:response.enabled)s.activeGroups+=active!=0;
        const PxU32 sweeps=response.complete?s.workingRows.sweeps:response.sweeps;
        s.result={s.solved,sweeps,0,sweeps,response.error};
    }
    if(mBoundedResponse!=s.linearResponse)PX_DELETE(mBoundedResponse);
    mBoundedResponse=NULL;
    const PxU64 admissionStart=mStep.profile?PxTime::getCurrentCounterValue():0;
    const bool admitted=resumeCommonPublication(stationary);
    if(mStep.profile)mStep.admissionTicks+=PxTime::getCurrentCounterValue()-admissionStart;
    if(!admitted)return false;
    done=s.finished;
    if(done)
    {
        mStep.commonClosed=mStep.commonClosed && stationary;
        for(PxU32 b:s.subset.blocks)mPrimalIdleIteration[b]=stationary?mStep.iteration+1:0;
        ++mStep.commonComponent;clearCommonPublication();
        if(!nextBoundedCommonComponent())return false;
        done=mBoundedResponse==NULL;stationary=mStep.commonClosed;
        if(done)writeJointDualState();
    }
    else startBoundedResponse();
    return true;
}

bool AvbdDynamicsContext::resumeCommonPublication(bool& stationary)
{
    auto& publication=*mCommonPublication;auto& p=publication.p;const auto& subset=publication.subset;
    auto& previous=publication.previous;auto& force=publication.force;auto& step=publication.step;auto& trial=publication.trial;
    auto& candidate=publication.candidate;auto& motionTolerance=publication.motionTolerance;auto& actual=publication.actual;
    auto& result=publication.result;auto& requested=publication.requested;auto& fraction=publication.fraction;auto& search=publication.search;
    auto& initial=publication.initial;auto& lastMerit=publication.lastMerit;auto& rejection=publication.rejection;
    auto& secondOrderAttempted=publication.secondOrderAttempted;auto& secondOrderSearch=publication.secondOrderSearch;
    auto& uncorrectedStep=publication.uncorrectedStep;auto& uncorrectedForce=publication.uncorrectedForce;
    auto& uncorrectedResult=publication.uncorrectedResult;auto& uncorrectedRequested=publication.uncorrectedRequested;
    auto& uncorrectedGroups=publication.uncorrectedGroups;auto& closed=publication.closed;auto& rootSmall=publication.rootSmall;
    auto& correctionRatio=publication.correctionRatio;auto& correctionIndex=publication.correctionIndex;
    const AvbdCommonResidual& metric=publication.metric;
    const AvbdRigidPoseView poses={mBodies,publication.patch?p.sourceBlocks.begin():NULL,candidate.begin(),candidate.size()};
    stationary=false;const char* reason="ok";
    auto failed=[&](const char* stage)
    {std::fprintf(stderr,"[AVBD_COMMON_STEP] iteration=%u stage=%s reason=%s status=FAIL\n",mStep.iteration,stage,reason);return false;};
    auto captureCandidate=[&]()
    {
    if(const char* path=std::getenv("PHYSX_AVBD_COMMON_FAILED_CANDIDATE"))
    {
        FILE* file=std::fopen(path,"wb");if(file){p.write(file,&previous,&motionTolerance);std::fclose(file);}
        // Diagnostic only: distinguish material PSD projection from the
        // original second variation. No captured operator enters execution.
        const std::string curvaturePath=std::string(path)+".curvature.json";
        file=std::fopen(curvaturePath.c_str(),"wb");
        if(file)
        {
            std::fprintf(file,"{\"frames\":[");bool first=true;
            for(PxU32 r:subset.rows)
            {
                const auto& row=mRows[r];
                if((row.header().flags&eAVBD_ROW_INACTIVE) || row.header().kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME)continue;
                const auto& m=mElasticFrameMaterials[row.header().geometryIndex-1];
                const PxTransform endpoints[2]={PxTransform(mBodies[row.header().bodyA].p,mBodies[row.header().bodyA].q),PxTransform(mBodies[row.header().bodyB].p,mBodies[row.header().bodyB].q)};
                const auto e=avbdEvaluateElasticFrame(m,endpoints,true);
                std::fprintf(file,"%s{\"row\":%u,\"a\":%u,\"b\":%u,\"valid\":%s,\"raw\":[",first?"":",",r,row.header().bodyA,row.header().bodyB,e.valid?"true":"false");first=false;
                for(PxU32 j=0;j<81;++j)std::fprintf(file,"%s%.17g",j?",":"",e.hessian[j]);
                std::fprintf(file,"],\"projected\":[");
                for(PxU32 j=0;j<9;++j)for(PxU32 k=0;k<9;++k)
                {double value=0;for(PxU32 z=0;z<9;++z)value+=double(PxReal(e.root[z][j<6?j:j+3]))*double(PxReal(e.root[z][k<6?k:k+3]));
                    std::fprintf(file,"%s%.17g",j||k?",":"",value);}
                std::fprintf(file,"]}");
            }
            std::fprintf(file,"]}\n");std::fclose(file);
        }
        const std::string detail=std::string(path)+".trial.json";file=std::fopen(detail.c_str(),"wb");
        if(file)
        {
            std::fprintf(file,"{\"actual\":");actual.write(file);
            auto values=[&](const char* name,const PxArray<double>& array)
            {std::fprintf(file,",\"%s\":[",name);for(PxU32 d=0;d<array.size();++d)std::fprintf(file,"%s%.17g",d?",":"",array[d]);std::fprintf(file,"]");};
            values("step",step);values("force",force);values("storedForce",trial);
            PxArray<double> from,to;
            for(PxU32 b:p.sourceBlocks)
            {
                const auto end=poses[b];const auto& body=mBodies[b];
                for(PxU32 d=0;d<3;++d){from.pushBack(body.p[d]);to.pushBack(end.p[d]);}
                const double a[4]={body.q.x,body.q.y,body.q.z,body.q.w},z[4]={end.q.x,end.q.y,end.q.z,end.q.w};
                for(PxU32 d=0;d<4;++d){from.pushBack(a[d]);to.pushBack(z[d]);}
            }
            values("from",from);values("to",to);std::fprintf(file,"}\n");std::fclose(file);
        }
    }
    };
    if(publication.phase==AvbdCommonPublication::Direction)
    {
        if(!publication.solved)
        {
            if(!publication.bounded || !publication.valid){reason="frozen_response_unresolved";return failed("solve");}
            // Qualify an inexact DIRECTION using the existing engineering
            // scales; this is not permission to publish an unfinished frame.
            PxArray<double> tolerance=motionTolerance;
            for(double& value:tolerance)value*=.1;
            if(!metric.smallRoot(p,force.begin(),tolerance.begin(),step.begin()))
            {reason="inexact_direction_root";return failed("certificate");}
            publication.rhs.resize(p.base.size());publication.correction.resize(p.base.size());
            p.hessian(step.begin(),publication.correction.begin());p.transpose(force.begin(),publication.rhs.begin());
            for(PxU32 d=0;d<p.base.size();++d)publication.rhs[d]+=p.base[d]-publication.correction[d];
            publication.phase=AvbdCommonPublication::DirectionCertificate;
            return true;
        }
        requested=0;for(PxU32 d=0;d<step.size();++d)requested=std::max(requested,std::abs(step[d])/motionTolerance[d]);
        publication.phase=AvbdCommonPublication::Trial;
    }
    else if(publication.phase==AvbdCommonPublication::DirectionCertificate)
    {
        if(!publication.solved){reason="inexact_direction_h_unresolved";return failed("certificate");}
        double ratio=0;for(PxU32 d=0;d<publication.correction.size();++d)
            ratio=std::max(ratio,std::abs(publication.correction[d])/(.1*motionTolerance[d]));
        if(!(ratio<=1)){reason="inexact_direction_correction";return failed("certificate");}
        mStep.requiresDomainClosure=true;
        if(std::getenv("PHYSX_AVBD_COMMON_STATS"))std::fprintf(stderr,
            "[AVBD_COMMON_DIRECTION_CERTIFICATE] iteration=%u correction=%.9g strict=FAIL direction=PASS\n",mStep.iteration,ratio);
        requested=0;for(PxU32 d=0;d<step.size();++d)requested=std::max(requested,std::abs(step[d])/motionTolerance[d]);
        publication.phase=AvbdCommonPublication::Trial;
    }
    else if(publication.phase==AvbdCommonPublication::SecondOrder)
    {
        p.offset.swap(publication.originalOffset);
        if(!publication.valid){reason="second_order_material_unresolved";return failed("solve");}
        ++search;fraction=.5;
        if(!publication.solved)
        {
            step.swap(uncorrectedStep);force.swap(uncorrectedForce);result=uncorrectedResult;
            requested=uncorrectedRequested;publication.activeGroups=uncorrectedGroups;
        }
        else
        {
            secondOrderSearch=search;fraction=1;
            requested=0;for(PxU32 d=0;d<step.size();++d)requested=std::max(requested,std::abs(step[d])/motionTolerance[d]);
        }
        publication.phase=AvbdCommonPublication::Trial;
    }
    else if(publication.phase==AvbdCommonPublication::Correction)
    {
        closed=publication.solved;
        for(PxU32 d=0;d<publication.correction.size();++d)
        {const double ratio=std::abs(publication.correction[d])/motionTolerance[d];
            if(ratio>correctionRatio){correctionRatio=ratio;correctionIndex=d;}closed=closed && ratio<=1;}
        if(publication.bounded && !publication.solved){reason="bounded_certificate_unresolved";return failed("certificate");}
        publication.phase=AvbdCommonPublication::Admission;
    }
    for(;search<16;++search,fraction*=.5,publication.phase=AvbdCommonPublication::Trial)
    {
        if(publication.phase==AvbdCommonPublication::Trial)
        {
        // SOC is one full-step candidate, not a replacement direction merely
        // because its shifted linear system converged. If actual admission
        // rejects it, resume backtracking the original coupled pose/force pair.
        if(search>secondOrderSearch)
        {
            step.swap(uncorrectedStep);force.swap(uncorrectedForce);result=uncorrectedResult;
            requested=uncorrectedRequested;publication.activeGroups=uncorrectedGroups;
            secondOrderSearch=PX_MAX_U32;fraction=.5;
        }
        // Inexact roots may point slightly inward at a nearly active plane.
        // Start ordinary backtracking inside its exact affine boundary instead
        // of spending all trials above it. A zero bound is not convergence.
        if(search && search!=secondOrderSearch)
            fraction=PxMin(fraction,.5*p.linearFeasibleFraction(step.begin()));
        for(PxU32 i=0;i<candidate.size();++i)
        {const PxU32 b=publication.patch?p.sourceBlocks[i]:i;candidate[i]=PxTransform(mBodies[b].p,mBodies[b].q);}
        for(PxU32 s=0;s<p.sourceBlocks.size();++s)
        {
            const PxU32 b=p.sourceBlocks[s],i=publication.patch?s:b;PxVec3d dth;
            for(PxU32 d=0;d<3;++d)
            {candidate[i].p[d]=PxReal(double(mBodies[b].p[d])+fraction*step[6*s+d]);dth[d]=fraction*step[6*s+d+3];}
            // Locked coordinates were eliminated in H/J. Round the common
            // retraction once; float renormalization can rotate even a zero step.
            candidate[i].q=avbdStoredRotationIncrement(mBodies[b].q,dth);
        }
        // Select a representable point inside the proposal's FLOAT CELL,
        // reusing the existing lattice publisher. This is quantization, not
        // a new geometric correction or a change to the coupled force.
        for(PxU32 s=0;s<p.sourceBlocks.size();++s)
        {
            const PxU32 b=p.sourceBlocks[s],i=publication.patch?s:b;AvbdOgcProjectionPlanes planes;
            for(PxU32 j=mRowProgram.blockOffsets[b];j<mRowProgram.blockOffsets[b+1];++j)
            {
                const PxU32 r=mRowProgram.blockRows[j];const auto& row=mRows[r];
                if(row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT || (row.header().flags&eAVBD_ROW_INACTIVE))continue;
                const auto& domain=mNpDomains[row.header().geometryIndex-1];
                if(domain.rowStart!=PX_MAX_U32 && r!=domain.rowStart)continue;
                const PxTransform ends[2]={row.header().bodyA<poses.size()?PxTransform(poses[row.header().bodyA].p,poses[row.header().bodyA].q):domain.fixedPose[0],
                    row.header().bodyB<poses.size()?PxTransform(poses[row.header().bodyB].p,poses[row.header().bodyB].q):domain.fixedPose[1]};
                AvbdOgcProjectionPlane plane;plane.n=domain.normal*(row.header().bodyA==b?1.:-1.);
                const PxTransform start[2]={row.header().bodyA<mBodies.size()?PxTransform(mBodies[row.header().bodyA].pStart,mBodies[row.header().bodyA].qStart):domain.fixedPose[0],
                    row.header().bodyB<mBodies.size()?PxTransform(mBodies[row.header().bodyB].pStart,mBodies[row.header().bodyB].qStart):domain.fixedPose[1]};
                const double gap=domain.rowStart==PX_MAX_U32?avbdNpGap(domain,ends):avbdNpPathGap(domain,start,ends);
                plane.b=plane.n.dot(avbdOgcDouble(candidate[i].p)-avbdOgcDouble(mBodies[b].p))-gap;
                planes.pushBack(plane);
            }
            const PxVec3d dp(fraction*step[6*s],fraction*step[6*s+1],fraction*step[6*s+2]);
            PxVec3 stored;
            if(avbdOgcPublishRegionPoint(mBodies[b].p,dp,planes,PX_MAX_F32,stored))candidate[i].p=stored;
        }
        for(PxU32 i=0;i<trial.size();++i)
        {
            // Force the actual publication precision, including /fp:fast.
            volatile PxReal stored=PxReal(previous[i]+fraction*(force[i]-previous[i]));trial[i]=stored;
        }
        bool admissible=true;
        for(PxU32 b:p.sourceBlocks)
        {
            const auto end=poses[b];
            if(!end.p.isFinite() || !end.q.isFinite()){admissible=false;break;}
        }
        for(PxU32 rowIndex:subset.rows)if(admissible && !(mRows[rowIndex].header().flags&eAVBD_ROW_INACTIVE))
        {
            const auto& row=mRows[rowIndex];
            if(row.header().kernelId==eAVBD_ROW_KERNEL_NP_CONTACT)
            {
                const PxU32 first=mNpDomains[row.header().geometryIndex-1].rowStart;
                if(first!=PX_MAX_U32 && rowIndex!=first)continue;
            }
            if(!avbdNpCandidateAdmissible(row,mNpDomains,mBodies,poses,mKernels))
            {
                admissible=false;
                if((search==0 && avbdCommonVerbose()) || (search==15 && std::getenv("PHYSX_AVBD_COMMON_STATS")))
                {
                    const auto& domain=mNpDomains[row.header().geometryIndex-1];PxTransform ends[2],starts[2],dcd[2];
                    const PxU32 ids[2]={row.header().bodyA,row.header().bodyB};
                    for(PxU32 s=0;s<2;++s)
                    {ends[s]=ids[s]<mBodies.size()?PxTransform(poses[ids[s]].p,poses[ids[s]].q):domain.fixedPose[s];
                     starts[s]=ids[s]<mBodies.size()?PxTransform(mBodies[ids[s]].p,mBodies[ids[s]].q):domain.fixedPose[s];
                     dcd[s]=ids[s]<mBodies.size()?PxTransform(mBodies[ids[s]].pStart,mBodies[ids[s]].qStart):domain.fixedPose[s];}
                    std::fprintf(stderr,"[AVBD_COMMON_GEOMETRY] a=%u b=%u domain=%u current=%.17g candidate=%.17g witness=%.17g\n",
                        row.header().bodyA,row.header().bodyB,row.header().geometryIndex,avbdNpGap(domain,starts),avbdNpGap(domain,ends),double(avbdEvalC(row,poses)));
                    if(search==15)
                    {
                        PxArray<double> delta(step.size()),linear(p.offset.size());
                        for(PxU32 s=0;s<p.sourceBlocks.size();++s)
                        {const PxU32 b=p.sourceBlocks[s];const auto end=poses[b];
                            const PxVec3d dp=avbdOgcDouble(end.p)-avbdOgcDouble(mBodies[b].p),dt=avbdAcceptedRotationDisplacementDouble(mBodies[b].q,end.q);
                            for(PxU32 d=0;d<3;++d){delta[6*s+d]=dp[d];delta[6*s+3+d]=dt[d];}}
                        p.jacobian(delta.begin(),linear.begin());double predicted=DBL_MAX;
                        for(PxU32 r=0;r<p.sourceRows.size();++r)if(mRows[p.sourceRows[r]].header().geometryIndex==row.header().geometryIndex)
                            predicted=PxMin(predicted,p.offset[3*r]+linear[3*r]);
                        std::fprintf(stderr,"[AVBD_COMMON_GEOMETRY_END] domain=%u fraction=%.17g currentPath=%.17g proposalPath=%.17g linearPath=%.17g\n",
                            row.header().geometryIndex,fraction,avbdNpPathGap(domain,dcd,starts),avbdNpPathGap(domain,dcd,ends),predicted);
                    }
                }
            }
            if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON &&
                !tetrahedronPath(row,mTetrahedronMaterials[row.header().geometryIndex-1],mRowSupports,mBodies,poses))admissible=false;
        }
        if(!admissible)
        {
            rejection="geometry";
            // A tangent SQP direction may violate a curved boundary by O(d^2)
            // at EVERY step length (Maratos effect). Correct the constraint
            // linearization once, resolving the SAME H/J for pose AND force.
            // This never projects a pose independently of its reaction.
            if(!secondOrderAttempted && search==0)
            {
                AvbdCommonProgram nonlinear;
                if(compileCommonProgram(nonlinear,reason,&poses,&subset,false))
                {
                    PxArray<double> linear(p.offset.size());p.jacobian(step.begin(),linear.begin());
                    publication.originalOffset=p.offset;
                    uncorrectedStep=step;uncorrectedForce=force;uncorrectedResult=result;
                    uncorrectedRequested=requested;uncorrectedGroups=publication.activeGroups;
                    for(PxU32 i=0;i<p.offset.size();++i)p.offset[i]=nonlinear.offset[i]-linear[i];
                    if(publication.patch || publication.bounded)for(double& f:force)f=0;
                    secondOrderAttempted=true;publication.phase=AvbdCommonPublication::SecondOrder;
                    return true; // Resume after the same numerical executor joins.
                }
            }
            continue;
        }
        publication.actual=AvbdCommonProgram();
        if(!compileCommonProgram(actual,reason,&poses,&subset,false)){rejection=reason;continue;}
        lastMerit=metric.evaluate(actual,trial.begin());
        // Engineering closure uses the existing physical displacement scales,
        // not monotone decrease below the float storage floor. Verify BOTH
        // nonlinear stationarity and the original force law at this exact pair.
        publication.closed=false;publication.correctionRatio=0;publication.correctionIndex=PX_MAX_U32;
        publication.rootSmall=metric.smallRoot(actual,trial.begin(),motionTolerance.begin());
        if(publication.rootSmall)
        {
            publication.rhs.resize(actual.base.size());publication.correction.resize(actual.base.size());
            actual.transpose(trial.begin(),publication.rhs.begin());
            for(PxU32 d=0;d<publication.rhs.size();++d)publication.rhs[d]+=actual.base[d];
            publication.phase=AvbdCommonPublication::Correction;
            return true; // Full-H certificate, never a physical correction.
        }
        publication.phase=AvbdCommonPublication::Admission;
        }
        PxArray<double> delta(step.size());
        for(PxU32 s=0;s<p.sourceBlocks.size();++s)
        {
            const PxU32 b=p.sourceBlocks[s];const auto& body=mBodies[b];
            const PxVec3d dp=avbdOgcDouble(poses[b].p)-avbdOgcDouble(body.p),
                dth=avbdAcceptedRotationDisplacementDouble(body.q,poses[b].q);
            for(PxU32 d=0;d<3;++d){delta[6*s+d]=dp[d];delta[6*s+d+3]=dth[d];}
        }
        double work=0,slope=0;
        const bool descent=AvbdCommonResidual::workDescent(p,actual,force.begin(),delta.begin(),work,slope);
        if(avbdCommonVerbose() && (search==0 || search==15 || mStep.iteration>=9))
        {
            std::fprintf(stderr,"[AVBD_COMMON_TRIAL] iteration=%u search=%u fraction=%.9g initial=%.17g residual=%.17g work=%.17g slope=%.17g closed=%u descent=%u\n",
                mStep.iteration,search,fraction,initial,lastMerit,work,slope,unsigned(closed),unsigned(descent));
            std::fprintf(stderr,"[AVBD_COMMON_CLOSURE] rootSmall=%u correction=%.9g body=%u coordinate=%u\n",
                unsigned(rootSmall),correctionRatio,correctionIndex==PX_MAX_U32?PX_MAX_U32:p.sourceBlocks[correctionIndex/6],correctionIndex%6);
            if(search==15 && !descent)
            {
                PxArray<double> linear(p.offset.size());p.jacobian(delta.begin(),linear.begin());
                PxU32 worst=PX_MAX_U32;double defect=-1;
                for(PxU32 i=0;i<p.offset.size();++i)
                {const double error=std::abs(force[i]*(actual.offset[i]-p.offset[i]-linear[i]));
                    if(error>defect){defect=error;worst=i;}}
                if(worst!=PX_MAX_U32)std::fprintf(stderr,"[AVBD_COMMON_DIRECTION] row=%u coordinate=%u C=%.17g actual=%.17g linear=%.17g force=%.17g\n",
                    p.sourceRows[worst/3],worst%3,p.offset[worst],actual.offset[worst]-p.offset[worst],linear[worst],force[worst]);
            }
        }
        if(publication.bounded && !result.converged && closed && std::getenv("PHYSX_AVBD_COMMON_STATS"))
            std::fprintf(stderr,"[AVBD_COMMON_CERTIFIED_INEXACT] iteration=%u rootSmall=%u correction=%.9g strict=FAIL actual=PASS\n",
                mStep.iteration,unsigned(rootSmall),correctionRatio);
        if(!(closed || descent))
        {rejection="nonlinear_work";continue;}
        // Retain the exact unfinished candidate at the outer budget boundary,
        // before Backend commit changes the diagnostic from/to pair.
        if(!closed && mStep.iteration+1>=mStep.iterationLimit)captureCandidate();
        for(PxU32 s=0;s<p.sourceBlocks.size();++s)
        {
            const PxU32 b=p.sourceBlocks[s];
            const auto end=poses[b];
            if(mBodies[b].p!=end.p || !(mBodies[b].q==end.q))mPrimalChangedIteration[b]=mStep.iteration+1;
            mBodies[b].p=end.p;mBodies[b].q=end.q;
        }
        for(PxU32 r=0;r<p.sourceRows.size();++r)
        {
            const PxU32 i=p.sourceRows[r];auto row=mRows[i];
            row.scalar().lambda=-PxReal(trial[3*r]);row.scalar().lambdaT0=-PxReal(trial[3*r+1]);row.scalar().lambdaT1=-PxReal(trial[3*r+2]);
            if(i<mContactWarm.size())
            {auto& warm=mContactWarm[i];warm.lambda=row.scalar().lambda;warm.lambdaT0=row.scalar().lambdaT0;warm.lambdaT1=row.scalar().lambdaT1;}
        }
        stationary=closed && (!publication.patch || requested*fraction<=1);
        publication.finished=true;
        if(avbdCommonVerbose())std::fprintf(stderr,
            "[AVBD_COMMON_STEP] iteration=%u blocks=%u groups=%u fraction=%.9g requested=%.9g initial=%.9g residual=%.9g activeGroups=%u materialProducts=%u secondOrder=%u status=PASS storage=backend_iterate\n",
            mStep.iteration,p.dimensions.size(),p.mu.size(),fraction,requested,initial,lastMerit,publication.activeGroups,publication.materialProducts,unsigned(secondOrderAttempted));
        return true;
    }
    reason=rejection;std::fprintf(stderr,"[AVBD_COMMON_REJECT] requested=%.9g initial=%.9g residual=%.9g\n",requested,initial,lastMerit);
    captureCandidate();
    return failed("accept");
}

// Explicit diagnostic capture only. The compiler owns no pose or dual and
// cannot publish an unverified frozen candidate into the physical step.
void AvbdDynamicsContext::captureCommonProgram()
{
    const char* path=std::getenv("PHYSX_AVBD_COMMON_CAPTURE");if(!path || !path[0])return;
    const char* minimum=std::getenv("PHYSX_AVBD_COMMON_CAPTURE_CONTACTS");
    if(minimum && mContactRowCount<PxU32(std::strtoul(minimum,NULL,10)))return;
    const char* load=std::getenv("PHYSX_AVBD_COMMON_CAPTURE_LOAD");
    if(load)
    {
        double maximum=0;for(PxU32 r=0;r<mContactRowCount;++r)maximum=std::max(maximum,-double(mRows[r].scalar().lambda));
        if(maximum<std::strtod(load,NULL))return;
    }
    static PxI32 claimed=0;if(PxAtomicCompareExchange(&claimed,0,0)!=0)return;
    AvbdCommonProgram p;const char* reason;const bool complete=compileCommonProgram(p,reason);
    AvbdCommonResponse response(p,mKernels);PxArray<double> step,force;
    const char* closing=std::getenv("PHYSX_AVBD_COMMON_CAPTURE_CLOSING");
    if(complete && closing)
    {
        step.resize(p.base.size());PxArray<double> motion(p.offset.size());
        if(response.material(p.base.begin(),step.begin()))
        {
            p.jacobian(step.begin(),motion.begin());double gap=0;
            for(PxU32 r=0;r<p.sourceRows.size();++r)if(mRows[p.sourceRows[r]].header().flags&eAVBD_ROW_CONTACT)
                gap=std::min(gap,p.offset[3*r]+motion[3*r]);
            if(gap>=-std::strtod(closing,NULL))return;
        }
    }
    if(PxAtomicCompareExchange(&claimed,1,0)!=0)return;
    FILE* file=std::fopen(path,"wb");if(!file)return;
    const char* solveOption=std::getenv("PHYSX_AVBD_COMMON_CAPTURE_SOLVE");
    const bool attempted=solveOption && std::atoi(solveOption)!=0;
    std::fprintf(file,"{\"schema\":1,\"complete\":%s,\"reason\":\"%s\",\"sourceRowCount\":%u,\"sourceBlockCount\":%u,\"dt\":%.17g,",
        complete?"true":"false",reason,mRows.size(),mBlocks.size(),double(mDt));
    std::fprintf(file,"\"tableSolveAttempted\":%s,",attempted?"true":"false");
    auto array=[&](const char* name,const auto& values)
    {std::fprintf(file,"\"%s\":[",name);for(PxU32 i=0;i<values.size();++i)
        {const double v=double(values[i]);std::fprintf(file,"%s",i?",":"");
            if(std::isfinite(v))std::fprintf(file,"%.17g",v);else std::fprintf(file,"\"%s\"",std::isnan(v)?"nan":v<0?"-inf":"inf");}
        std::fprintf(file,"],");};
    array("dimensions",p.dimensions);array("sourceBlocks",p.sourceBlocks);array("sourceRows",p.sourceRows);
    array("base",p.base);array("offset",p.offset);array("lower",p.lower);array("upper",p.upper);array("compliance",p.compliance);
    array("mu",p.mu);array("coulomb",p.coulomb);
    auto matrix=[&](const char* name,const auto& entries,PxU32 width)
    {
        std::fprintf(file,"\"%s\":[",name);
        for(PxU32 e=0;e<entries.size();++e)
        {std::fprintf(file,"%s[%u,%u",e?",":"",entries[e].row,entries[e].column);for(PxU32 k=0;k<width;++k)std::fprintf(file,",%.17g",entries[e].values[k]);std::fprintf(file,"]");}
        std::fprintf(file,"]");
    };
    matrix("H",p.H,36);std::fprintf(file,",");matrix("J",p.J,18);std::fprintf(file,"}\n");
    // Preserve a complete input even if a diagnostic solve is interrupted.
    std::fflush(file);
    AvbdCoulombNcpResult result={};const bool solved=complete && attempted && response.solve(step,force,result);
    std::fseek(file,-2,SEEK_END);std::fprintf(file,",");
    array("tableStep",step);array("tableForce",force);
    std::fprintf(file,"\"tableSolved\":%s,\"tableIterations\":%u,\"materialSolves\":%u,\"materialProducts\":%u}\n",
        solved?"true":"false",result.iterations,response.materialSolves,response.materialProducts);
    const bool written=std::ferror(file)==0;std::fclose(file);
    std::fprintf(stderr,"[AVBD_COMMON_CAPTURE] complete=%u written=%u reason=%s blocks=%u groups=%u hBlocks=%u jBlocks=%u tableSolved=%u materialProducts=%u publication=none\n",
        unsigned(complete),unsigned(written),reason,p.dimensions.size(),p.mu.size(),p.H.size(),p.J.size(),unsigned(solved),response.materialProducts);
}

bool AvbdDynamicsContext::linearizeRigidRow(const AvbdRowConstRef& row,PxU32 i,
    PxReal& C,PxVec3& jLin,PxVec3& jAng,const AvbdRigidPoseView* candidate) const
{
    const AvbdRigidPoseView current={mBodies,NULL,NULL,0};const auto& poses=candidate?*candidate:current;
    const auto b=poses[i];
	const bool isA = (row.header().bodyA == i);
	const bool isB = (row.header().bodyB == i);
	if(!isA && !isB)
		return false;

	C=0;
	if(row.header().flags & eAVBD_ROW_POINT_LOCK)
	{
		C = avbdEvalC(row, poses);
		const PxVec3 axis = row.scalar().n;
		if(isA)
		{
			jLin = axis;
			jAng = b.q.rotate(row.scalar().rA).cross(axis);
		}
		else
		{
			jLin = -axis;
			jAng = b.q.rotate(row.scalar().rB).cross(-axis);
		}
	}
	else if(row.scalar().articJoint)
	{
		PxVec3 jLinA, jAngA, jLinB, jAngB;
		if(!avbdEvalArticRow(row, poses, jLinA, jAngA, jLinB, jAngB, C))
			return false;
		jLin = isA ? jLinA : jLinB;
		jAng = isA ? jAngA : jAngB;
	}
	else
	{
		if(row.header().supportCount)
		{
			C = avbdHeterogeneousPositionAl(row) ?
				avbdHeterogeneousAlC(row, mRowSupports, mBlocks, mBodies, contactGeometry()) :
				avbdHeterogeneousRowC(row, mRowSupports, mBlocks, mBodies, contactGeometry());
			jLin = PxVec3(0.0f);
			jAng = PxVec3(0.0f);
			for(PxU32 supportIndex = row.header().supportStart;
				supportIndex < row.header().supportStart + row.header().supportCount;
				++supportIndex)
			{
				if(supportIndex >= mRowSupports.size())
					break;
				const AvbdRowSupport& support = mRowSupports[supportIndex];
				if(support.block != i || support.block >= mBlocks.size())
					continue;
				PxVec3 supportLin, supportAng;
				if(avbdHeterogeneousSupportJacobian(row, support,
					mBlocks[support.block], mBodies, contactGeometry(), mRowSupports, mBlocks, supportLin, supportAng))
				{
					jLin += supportLin;
					jAng += supportAng;
				}
			}
		}
		else
		{
			C = avbdEvalC(row, poses);
			if(row.header().flags & eAVBD_ROW_CONTACT)
			{
				const PxReal sign = isA ? 1.0f : -1.0f;
				const PxVec3 rWorld = isA ? b.q.rotate(row.scalar().ogcRA) : b.q.rotate(row.scalar().ogcRB);
				jLin = row.scalar().n * sign;
				jAng = rWorld.cross(jLin);
			}
			else
			{
				jLin = isA ? row.scalar().jLinA : row.scalar().jLinB;
				jAng = isA ? row.scalar().jAngA : row.scalar().jAngB;
				const PxVec3 positionGradient=avbdRotationChartGradient(b.qLin,b.q,jAng);
				if(row.scalar().integratedAngular)
					jAng=avbdRotationChartGradient(b.qStart,b.q,jAng);
				else if((row.header().flags&eAVBD_ROW_DRIVE) && row.header().kernelId!=eAVBD_ROW_KERNEL_FINITE_SPRING)
				{
					const PxReal total=row.scalar().driveStiffness+row.scalar().driveDampingOverDt;
					jAng=total>0.0f ? (positionGradient*row.scalar().driveStiffness+
						avbdRotationChartGradient(b.qStart,b.q,jAng)*row.scalar().driveDampingOverDt)/total : PxVec3(0.0f);
				}
				else jAng=positionGradient;
			}
		}
	}

	return true;
}

template<class Assembly> void AvbdDynamicsContext::accumulateRigidConstraintRow(PxU32 i,PxU32 ri,
    PxReal invH2,Assembly& destination)
{
    const AvbdSolverBody& b=mBodies[i];const AvbdRowConstRef& row=mRows[ri];
    PxReal (&H)[6][6]=destination.hessian;PxReal (&g)[6]=destination.gradient;
    Assembly* assembly=&destination;
        PxReal C=0;PxVec3 jLin,jAng;
        if(!linearizeRigidRow(row,i,C,jLin,jAng))return;
        if(row.scalar().articJoint)C=avbdArticPrimalC(row,C);
        else if((row.header().flags&eAVBD_ROW_CONTACT) && !row.header().supportCount && !(row.scalar().contactRadius>0))
            C-=AVBD_ALPHA*PxMin(row.scalar().C0,0.0f);
        const bool isA=row.header().bodyA==i;

		PxReal penalty = (row.header().flags & eAVBD_ROW_DRIVE) ?
			avbdDrivePenalty(row, mBodies) :
			avbdHardJointPenalty(row, mBodies, invH2);
		if((row.header().flags & eAVBD_ROW_CONTACT) && !mJoints.empty() &&
			b.invMass > 1.0e-12f)
		{
			// Endpoint conditioning belongs to the block metric. The contact
			// force below must still use the shared row penalty on both ends.
			penalty = PxMax(penalty,
				AVBD_CONTACT_JOINT_BOOST_FRACTION * invH2 / b.invMass);
		}
		const PxReal raw = penalty * C + row.scalar().lambda;
		const bool ogcContact = (row.header().flags & eAVBD_ROW_CONTACT) != 0;
		// Every non-contact row has force bounds (bilateral rows carry
		// +-PX_MAX_REAL).  A saturated row contributes its bound and no
		// Hessian, which is how a joint limit stays inactive while inside.
		const bool saturatedDrive = !ogcContact &&
			(raw < row.scalar().minLambda || raw > row.scalar().maxLambda);
		const PxReal driveClamped = saturatedDrive ?
			PxMin(row.scalar().maxLambda, PxMax(row.scalar().minLambda, raw)) : raw;
		// Default NP rigid contact is Position-AL (demo3d / WebPhysics):
		// Fn = min(0, rho*C + lambda) with lambda carrying the resting load, so
		// the pair rests at the authored rest distance independent of load.  A
		// pure offset potential would rest at r - W/(N k): light bodies float on
		// the shell, heavy stacks sink into it.  The OGC region stays the
		// non-penetration certificate; it is not the load carrier. The explicit
		// NP trial replaces this force with the shared normal envelope below.
		PxReal coeff = ((row.header().flags & eAVBD_ROW_CONTACT) &&
			(row.header().flags & eAVBD_ROW_UNILATERAL)) ? PxMin(0.0f, raw) : driveClamped;
		PxReal hessianCoeff = saturatedDrive ? 0.0f : penalty;
		if(ogcContact && !row.header().supportCount)
		{
			// One normal law for primal, dual, force limits and energy. Using
			// the endpoint metric here gives unequal action/reaction forces.
			const AvbdContactPotential potential=avbdNpContactPotential(row,C);
			coeff=potential.gradient;
			// The envelope curvature is bounded by rho for every gap. Retain
			// that global majorizer, including when Coulomb has no scalar
			// line search; a pointwise log curvature is not such a bound.
		}
		if(ogcContact && row.header().supportCount)
		{
			// Both block dimensions and dual consume the same row potential.
			// Partner mobility belongs in the solve metric, never in its force.
			const AvbdContactPotential potential =
				avbdHeterogeneousContactPotential(row, C);
			coeff = potential.gradient;
			hessianCoeff = potential.curvature;
		}
        // Endpoint conditioning is diagonal-only. Shared NP cross terms use
        // the same row coefficient on both ends, preserving an SPD metric.
        if(assembly) assembly->record(ri,jLin,jAng,ogcContact && !row.header().supportCount?row.scalar().rho:hessianCoeff);
		avbdAddOuter(H, jLin, jAng, hessianCoeff);
		g[0] += jLin.x * coeff;
		g[1] += jLin.y * coeff;
		g[2] += jLin.z * coeff;
		g[3] += jAng.x * coeff;
		g[4] += jAng.y * coeff;
		g[5] += jAng.z * coeff;

		// demo3d updatePrimal always stamps tangential K, sliding or not.
		if((row.header().flags & eAVBD_ROW_CONTACT) && row.scalar().mu > 0.0f &&
			row.scalar().positionTangent)
		{
			const PxVec3 rel = row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT?
                avbdNodalContactSlip(row,mRowSupports,mBlocks,mBodies,contactGeometry()):avbdContactRelDisp(row, mBodies);
			const PxReal sign = isA ? 1.0f : -1.0f;
			const PxVec3 rWorld = isA ? b.q.rotate(row.scalar().rA) : b.q.rotate(row.scalar().rB);
			PxVec3 j0 = row.scalar().t0 * sign,a0 = rWorld.cross(j0);
			PxVec3 j1 = row.scalar().t1 * sign,a1 = rWorld.cross(j1);
            if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
            {
                avbdNodalContactJacobian(row,i,row.scalar().t0,mRowSupports,mBlocks,mBodies,contactGeometry(),j0,a0);
                avbdNodalContactJacobian(row,i,row.scalar().t1,mRowSupports,mBlocks,mBodies,contactGeometry(),j1,a1);
            }
			const PxReal c0 = (1.0f - AVBD_ALPHA) * row.scalar().tangentC0 + row.scalar().t0.dot(rel);
			const PxReal c1 = (1.0f - AVBD_ALPHA) * row.scalar().tangentC1 + row.scalar().t1.dot(rel);
			PxReal tangentPenalty0 = row.scalar().rhoT0;
			PxReal tangentPenalty1 = row.scalar().rhoT1;
			if(!mJoints.empty() && b.invMass > 1.0e-12f)
			{
				const PxReal floor = AVBD_CONTACT_JOINT_BOOST_FRACTION * invH2 / b.invMass;
				tangentPenalty0 = PxMax(tangentPenalty0, floor);
				tangentPenalty1 = PxMax(tangentPenalty1, floor);
			}
			PxReal f0 = row.scalar().rhoT0 * c0 + row.scalar().lambdaT0;
			PxReal f1 = row.scalar().rhoT1 * c1 + row.scalar().lambdaT1;
			const PxReal projection = avbdProjectCoulomb(coeff, row.scalar().mu, f0, f1);
			const PxReal tangentScale = row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT ? projection : 1.f;
			// demo3d stamps the tangential Hessian for both endpoints in stick
			// and slip.  A maximal-coordinate link is not a different material:
			// omitting only its half of the Hessian breaks the shared friction
			// potential and lets cargo drift across a moving link.
			avbdAddOuter(H, j0, a0, tangentPenalty0*tangentScale);
			avbdAddOuter(H, j1, a1, tangentPenalty1*tangentScale);
            if(assembly)
            {
                assembly->record(ri,j0,a0,row.scalar().rhoT0*tangentScale,1);
                assembly->record(ri,j1,a1,row.scalar().rhoT1*tangentScale,2);
            }
			g[0] += j0.x * f0 + j1.x * f1;
			g[1] += j0.y * f0 + j1.y * f1;
			g[2] += j0.z * f0 + j1.z * f1;
			g[3] += a0.x * f0 + a1.x * f1;
			g[4] += a0.y * f0 + a1.y * f1;
			g[5] += a0.z * f0 + a1.z * f1;
		}
}

double AvbdDynamicsContext::localBlockInertialEnergy(PxU32 i,const PxTransform& pose) const
{
    const auto& b=mBodies[i];const auto dp=pose.p-b.pHat;
    return .5f*b.massInvH2*dp.magnitudeSquared()+
        avbdRotationInertialPotential(b.qHat,pose.q,b.inertiaWorldInvH2).energy;
}

bool AvbdDynamicsContext::localRowDomainAdmissible(const AvbdRowConstRef& row,const AvbdRigidPoseView& poses)
{
    if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
    {
        PxU32 visits=0;
        const bool valid=avbdNodalContactPath(mNodalContacts[row.header().geometryIndex-1],
            [&](PxU32 b){return mBlocks[b].particle->p;},poses,mStep.countNodalSolves?&visits:NULL);
        if(mStep.countNodalSolves){PxAtomicIncrement(&mStep.localPrimalQueries);PxAtomicAdd(&mStep.localPrimalPointVisits,PxI64(visits));}
        return valid;
    }
    if(row.header().kernelId==eAVBD_ROW_KERNEL_NP_CONTACT)
        return localNpCandidateAdmissible(row,poses);
    return row.header().kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON ||
        tetrahedronPath(row,mTetrahedronMaterials[row.header().geometryIndex-1],mRowSupports,mBodies,poses);
}

void AvbdDynamicsContext::sampleLocalBlock(const AvbdLocalBlockState& state,PxU32 begin,PxU32 end,
    AvbdLocalPartial& partial,bool trial)
{
    PX_ASSERT(end>=begin && end-begin<=AvbdLocalWork::RowLimit);
    if(end<begin || end-begin>AvbdLocalWork::RowLimit){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    if(!trial)partial=AvbdLocalPartial();else {partial.energy=0;partial.valid=true;}
    const PxU32 i=state.body;const auto& b=mBodies[i];
    const PxTransform& pose=trial?state.candidate:state.original;
    const AvbdRigidPoseView poses={mBodies,&i,&pose,1};
    partial.valid=pose.isSane();auto& block=partial.model;
    const AvbdFrameAssemblyTarget target={block.gradient,block.K,block.M,&partial.energy,1,36};
    AvbdFrameAssemblyPacket frames(mElasticFrameMaterials,mKernels,&i,target);
    const PxReal invH2=1.f/(mDt*mDt);
    for(PxU32 slot=begin;slot<end;++slot)
    {
        const PxU32 r=mRowProgram.blockRows[slot];const auto& row=mRows[r];
        if(partial.valid)partial.valid=localRowDomainAdmissible(row,poses);
        if(trial)
        {if(state.scalar)partial.energy+=rigidRowEnergy(r,i,poses,invH2);continue;}
        if(row.header().flags&eAVBD_ROW_INACTIVE)continue;
        partial.scalar &= !((row.header().flags&eAVBD_ROW_CONTACT) && row.scalar().positionTangent && row.scalar().mu>0);
        // Once any leaf finds non-associated friction, the reduced scalar
        // energies are unused. No false Coulomb energy oracle is introduced.
        if(row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME)
        {
            if(!frames.append(row.header().geometryIndex-1,row.header().bodyA,row.header().bodyB,poses))
            {PxAtomicExchange(&mLocalDomainFailed,1);return;}
            continue;
        }
        if(!frames.flush()){PxAtomicExchange(&mLocalDomainFailed,1);return;}
        if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
        {
            const auto e=evaluateTetrahedron(row);if(!e.valid){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            partial.energy+=e.energy;
            PxVec3d linear(0),angular(0);
            for(PxU32 k=0;k<4;++k)
            {
                const auto& support=mRowSupports[row.header().supportStart+k];if(support.block!=i)continue;
                linear+=e.jacobians[k];angular+=avbdOgcDouble(b.q.rotate(support.localPoint)).cross(e.jacobians[k]);
            }
            const double J[6]={linear.x,linear.y,linear.z,angular.x,angular.y,angular.z};
            for(PxU32 d=0;d<6;++d)
            {block.gradient[d]+=e.gradient*J[d];for(PxU32 k=0;k<6;++k)
                {const double h=e.curvature*J[d]*J[k];block.K[6*d+k]+=h;block.M[6*d+k]+=h;}}
        }
        else
        {
            if(partial.scalar)partial.energy+=rigidRowEnergy(r,i,poses,invH2);
            AvbdRigidAssembly rowBlock={};rowBlock.consistent=true;accumulateRigidConstraintRow(i,r,invH2,rowBlock);
            if(!rowBlock.consistent){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            for(PxU32 d=0;d<6;++d)
            {block.gradient[d]+=rowBlock.gradient[d];for(PxU32 k=0;k<6;++k)
                {block.K[6*d+k]+=rowBlock.hessian[d][k];block.M[6*d+k]+=rowBlock.hessian[d][k];}}
        }
    }
    if(!frames.flush())PxAtomicExchange(&mLocalDomainFailed,1);
}

void AvbdDynamicsContext::directLocalBlock(AvbdLocalBlockState& state,const AvbdLocalPartial& partial)
{
    const auto& b=mBodies[state.body];AvbdNonlinearBlock block=partial.model;
    block.locked=b.core?PxU32(b.core->lockFlags):0;
    if(state.body<mLocalTranslationClosed.size() && mLocalTranslationClosed[state.body])block.locked |= 7;
    const PxMat33d inertia(avbdOgcDouble(b.inertiaWorldInvH2.column0),avbdOgcDouble(b.inertiaWorldInvH2.column1),avbdOgcDouble(b.inertiaWorldInvH2.column2));
    const auto rotation=avbdRotationInertialPotential<double>(avbdNpRotation(b.qHat),avbdNpRotation(b.q),inertia);
    const auto displacement=avbdOgcDouble(b.p)-avbdOgcDouble(b.pHat);
    for(PxU32 d=0;d<3;++d)
    {
        block.K[7*d]+=b.massInvH2;block.M[7*d]+=b.massInvH2;
        block.gradient[d]+=double(b.massInvH2)*displacement[d];block.gradient[d+3]+=rotation.gradient[d];
        for(PxU32 k=0;k<3;++k)
        {block.K[6*(d+3)+k+3]+=rotation.gaussNewton[d][k];block.M[6*(d+3)+k+3]+=rotation.gaussNewton[d][k];}
    }
    AvbdNonlinearBlockArgs args={&block};mKernels.launch(AVBD_NONLINEAR_BLOCK,&args);
    if(!block.solved){PxAtomicExchange(&mLocalDomainFailed,1);state.done=true;return;}
    for(PxU32 d=0;d<3;++d)
    {state.dp[d]=PxReal(block.step[d]);state.dth[d]=PxReal(block.step[d+3]);
        state.gradientLinear[d]=PxReal(block.gradient[d]);state.gradientAngular[d]=PxReal(block.gradient[d+3]);}
    for(PxU32 d=0;d<6;++d)for(PxU32 k=0;k<6;++k)state.metric[d][k]=PxReal(block.M[6*d+k]);
    if(!state.dp.isFinite() || !state.dth.isFinite() || !state.gradientLinear.isFinite() || !state.gradientAngular.isFinite())
    {PxAtomicExchange(&mLocalDomainFailed,1);state.done=true;return;}
    state.scalar=partial.scalar;state.initialFeasible=partial.valid;
    state.initialEnergy=partial.energy+localBlockInertialEnergy(state.body,state.original);
    state.candidate=state.original;avbdApplyDelta(state.candidate,state.dp,state.dth);
}

void AvbdDynamicsContext::finishLocalBlockTrial(AvbdLocalBlockState& state,const AvbdLocalPartial& partial)
{
    PxAtomicIncrement(&mStep.localSearchTrials);
    const auto dp=state.candidate.p-state.original.p;
    const auto dth=avbdAcceptedRotationDisplacement(state.original.q,state.candidate.q);
    const PxReal slope=state.gradientLinear.dot(dp)+state.gradientAngular.dot(dth);
    const double energy=partial.energy+localBlockInertialEnergy(state.body,state.candidate);
    bool modelAccepted=true;
    if(!state.scalar && state.initialFeasible)
    {
        const double delta[6]={dp.x,dp.y,dp.z,dth.x,dth.y,dth.z};double quadratic=0;
        for(PxU32 r=0;r<6;++r)for(PxU32 c=0;c<6;++c)quadratic+=delta[r]*state.metric[r][c]*delta[c];
        modelAccepted=std::isfinite(quadratic) && double(slope)+.5*quadratic<=0;
    }
    if(partial.valid && modelAccepted && (!state.scalar || (!state.initialFeasible && std::isfinite(energy)) ||
        (slope<=0 && std::isfinite(energy) && energy<=state.initialEnergy+1.e-4f*slope)))
    {
        const PxU32 i=state.body;state.done=true;mBodies[i].p=state.candidate.p;mBodies[i].q=state.candidate.q;
        state.result=PxMax(dp.magnitude()/PxMax(getLengthScale(),1.e-6f)/AVBD_ARTIC_POSITION_REL_TOL,dth.magnitude()/AVBD_ARTIC_ROTATION_TOL);
        mPrimalSteps[i]=state.result;
        if(state.candidate.p!=state.original.p || !(state.candidate.q==state.original.q))mPrimalChangedIteration[i]=mStep.iteration+1;
        return;
    }
    if(++state.search==AvbdLocalWork::SearchLimit)
    {
        state.done=true;PxAtomicIncrement(&mStep.localSearchExhausted);
        if(!state.initialFeasible)PxAtomicExchange(&mLocalDomainFailed,1);
        return;
    }
    state.fraction*=.5f;state.candidate=state.original;
    avbdApplyDelta(state.candidate,state.dp*state.fraction,state.dth*state.fraction);
}

PxReal AvbdDynamicsContext::solveLocalRigidBlock(PxU32 i)
{
    AvbdLocalBlockState state;state.body=i;state.original=PxTransform(mBodies[i].p,mBodies[i].q);
    AvbdLocalPartial partial;const PxU32 begin=mRowProgram.blockOffsets[i],end=mRowProgram.blockOffsets[i+1];
    sampleLocalBlock(state,begin,end,partial,false);if(mLocalDomainFailed)return 0;
    return finishRigidAssembly(state,partial);
}

PxReal AvbdDynamicsContext::finishRigidAssembly(AvbdLocalBlockState& state,AvbdLocalPartial& partial)
{
    const PxU32 begin=mRowProgram.blockOffsets[state.body],end=mRowProgram.blockOffsets[state.body+1];
    directLocalBlock(state,partial);
    for(PxU32 search=0;search<AvbdLocalWork::SearchLimit && !state.done && !mLocalDomainFailed;++search)
    {sampleLocalBlock(state,begin,end,partial,true);finishLocalBlockTrial(state,partial);}
    return state.result;
}

void AvbdDynamicsContext::solveLocalWorkJob(PxU32 job)
{
    auto& work=*mLocalWork;
    if(work.phase==AvbdLocalWork::Direction || work.phase==AvbdLocalWork::Accept)
    {
        auto& state=work.states[job];if(state.done)return;
        if(state.body>=mBodies.size())
        {
            auto& trial=work.nodalTrials[job];auto& rows=work.nodal[state.root];
            if(work.phase==AvbdLocalWork::Direction)beginNodalTrial(state.body,rows,trial);
            else
            {
                acceptNodalTrial(state.body,rows,trial,work.nodalEnergy[state.root]);
            }
            state.done=trial.done || (work.phase==AvbdLocalWork::Accept && work.search+1==AvbdLocalWork::SearchLimit);
            return;
        }
        if(work.phase==AvbdLocalWork::Direction)directLocalBlock(state,work.partials[state.root]);
        else finishLocalBlockTrial(state,work.partials[state.root]);
        return;
    }
    const bool leaf=work.phase==AvbdLocalWork::Assemble || work.phase==AvbdLocalWork::Trial;
    const PxU32 id=leaf?work.leaves[job]:work.levels[work.level][job];const auto& node=work.nodes[id];
    const auto& state=work.states[node.state];if(state.done)return;
    if(state.body>=mBodies.size())
    {
        if(work.phase==AvbdLocalWork::Assemble){assembleNodalRows(state.body,node.begin,node.end,work.nodal[id]);return;}
        if(work.phase==AvbdLocalWork::ReduceAssembly)
        {work.nodal[id]=work.nodal[node.left];work.nodal[id].add(work.nodal[node.right]);return;}
        if(work.phase==AvbdLocalWork::Trial)
        {
            sampleNodalTrial(state.body,node.begin,node.end,work.nodalTrials[node.state],work.nodalEnergy[id]);return;
        }
        if(work.phase==AvbdLocalWork::ReduceTrial)
        {
            const auto& a=work.nodalEnergy[node.left];const auto& b=work.nodalEnergy[node.right];
            work.nodalEnergy[id]={a.energy+b.energy,a.valid && b.valid};return;
        }
    }
    if(leaf)sampleLocalBlock(state,node.begin,node.end,work.partials[id],work.phase==AvbdLocalWork::Trial);
    else
    {
        AvbdLocalReductionArgs args={&work.partials[node.left],&work.partials[node.right],&work.partials[id],work.phase==AvbdLocalWork::ReduceTrial};
        mKernels.launch(AVBD_LOCAL_REDUCTION,&args);
    }
}

void AvbdDynamicsContext::solvePrimalBody(PxU32 i, PxReal invH2,
	PxVec3& outDp, PxVec3& outDth, PxVec3& gradientLinear,
	PxVec3& gradientAngular,PxReal (*metric)[6])
{
	outDp = PxVec3(0.0f);
	outDth = PxVec3(0.0f);
	gradientLinear = gradientAngular = PxVec3(0.0f);
	AvbdSolverBody& b = mBodies[i];
	if(b.invMass == 0.0f || b.isKinematic)
		return;



    AvbdRigidAssembly output={};
    PxReal (&H)[6][6]=output.hessian;PxReal (&g)[6]=output.gradient;
    const PxU32 locked=b.core?PxU32(b.core->lockFlags):0;
    output.lockedMask=locked;
    initializeRigidMaterial(b,H,g);
	const PxU32 rowBegin = mRowProgram.blockOffsets[i];
	const PxU32 rowEnd = mRowProgram.blockOffsets[i + 1];
	for(PxU32 slot = rowBegin; slot < rowEnd; ++slot)
	{
		const PxU32 ri = mRowProgram.blockRows[slot];
		AvbdRowRef row = mRows[ri];
		if((row.header().flags & eAVBD_ROW_INACTIVE))
			continue;
        if(accumulateRigidMaterialRow(ri,&i,1,&output))
        { if(mLocalDomainFailed)return;continue; }
        accumulateRigidConstraintRow(i,ri,invH2,output);
	}

    avbdKernelRestrict6(H,g,locked);
	gradientLinear = PxVec3(g[0], g[1], g[2]);
	gradientAngular = PxVec3(g[3], g[4], g[5]);
	for(PxU32 d = 0; d < 6; ++d)
		H[d][d] += AVBD_REG;
	if(metric) PxMemCopy(metric,H,sizeof(H));

	PxReal rhs[6];
	for(PxU32 d = 0; d < 6; ++d)
		rhs[d] = -g[d];
	PxReal x[6];
	bool ok = false;
	AvbdLdlt6Args args;
	args.A = H;
	args.b = rhs;
	args.x = x;
	args.ok = &ok;
	mKernels.launch(AVBD_LDLT6, &args);
	if(!ok)
	{
		return;
	}
	PxVec3 dpSolve(x[0], x[1], x[2]);
	PxVec3 dthSolve(x[3], x[4], x[5]);
	avbdProjectLockedDelta(b, dpSolve, dthSolve);

	outDp = dpSolve;
	outDth = dthSolve;
}

void AvbdDynamicsContext::compileHeterogeneousBlocks()
{
	mBlocks.clear();
	mBlocks.reserve(mBodies.size());
	for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); ++bodyIndex)
	{
		AvbdBlock block;
		block.objectIndex = bodyIndex;
		block.stateIndex = PX_MAX_U32;
		block.particle = NULL;
		block.type = eAVBD_BLOCK_RIGID6;
		block.particleKernel = eAVBD_PARTICLE_KERNEL_NONE;
		mBlocks.pushBack(block);
	}
	for(PxU32 cacheIndex = 0; cacheIndex < mIslandSoftCache.size(); ++cacheIndex)
	{
		AvbdIslandSoftCache& cache = mIslandSoftCache[cacheIndex];
		cache.blockStart = mBlocks.size();
		if(!cache.coupled || cache.liveStamp != mIslandSoftLiveStamp ||
			!cache.particles)
			continue;
		mBlocks.reserve(mBlocks.size() + cache.nbParticles);
		for(PxU32 particleIndex = 0; particleIndex < cache.nbParticles;
			++particleIndex)
		{
			AvbdBlock block;
			block.objectIndex = cacheIndex;
			block.stateIndex = particleIndex;
			block.particle = cache.particles + particleIndex;
			block.type = eAVBD_BLOCK_PARTICLE3;
			block.particleKernel = eAVBD_PARTICLE_KERNEL_TET;
			mBlocks.pushBack(block);
		}
	}
	for(PxU32 cacheIndex = 0; cacheIndex < mIslandSurfaceCache.size(); ++cacheIndex)
	{
		AvbdIslandSurfaceCache& cache = mIslandSurfaceCache[cacheIndex];
		cache.blockStart = mBlocks.size();
		if(cache.liveStamp != mIslandSurfaceLiveStamp || !cache.particles)
			continue;
		mBlocks.reserve(mBlocks.size() + cache.nbParticles);
		for(PxU32 particleIndex = 0; particleIndex < cache.nbParticles;
			++particleIndex)
		{
			AvbdBlock block;
			block.objectIndex = cacheIndex;
			block.stateIndex = particleIndex;
			block.particle = cache.particles + particleIndex;
			block.type = eAVBD_BLOCK_PARTICLE3;
			block.particleKernel = eAVBD_PARTICLE_KERNEL_TRIANGLE;
			mBlocks.pushBack(block);
		}
	}
}

void AvbdDynamicsContext::compileRowProgram()
{
    const bool valid = mRowProgram.compile(mRows.size(), mBlocks.size(),
        [&](PxU32 index, const auto& emit)
    {
        const AvbdRowConstRef& row = mRows[index];
        if((row.header().bodyA != PX_MAX_U32 && row.header().bodyA >= mBodies.size()) ||
            (row.header().bodyB != PX_MAX_U32 && row.header().bodyB >= mBodies.size()) ||
            row.header().supportStart > mRowSupports.size() ||
            row.header().supportCount > mRowSupports.size() - row.header().supportStart) return false;
        // These energy kernels have no multiplier update. Pure membrane
        // history is a final-iteration operation in the same dual phase.
        const auto kernel=row.header().kernelId;
        if(kernel==eAVBD_ROW_KERNEL_NODAL_SHELL)mRowProgram.historyRows.append(index);
        else if(kernel!=eAVBD_ROW_KERNEL_TETRAHEDRON && kernel!=eAVBD_ROW_KERNEL_ELASTIC_FRAME &&
                !avbdIsNodalMaterial(kernel))mRowProgram.dualRows.append(index);
        emit(row.header().bodyA); emit(row.header().bodyB);
        for(PxU32 s = row.header().supportStart; s < row.header().supportStart + row.header().supportCount; ++s)
            emit(mRowSupports[s].block);
        return true;
    },&mSurfacePrefix);
    if(!valid)
    {
        PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION, PX_FL,
            "AVBD row contains an undeclared block or invalid support range; step not published.");
        PxAtomicExchange(&mLocalDomainFailed, 1);
    }
}

void AvbdDynamicsContext::compileVelocityComponents(PxArray<PxU32>& parents) const
{
    mRowProgram.components(parents, [&](PxU32 index)
    {
        const AvbdBlock& block = mBlocks[index];
        return block.type == eAVBD_BLOCK_RIGID6 ? avbdBodyResponds(mBodies[block.objectIndex]) :
            block.particle && block.particle->invMass > 0.0f;
    }, [&](PxU32 row) { return !(mRows[row].header().flags & eAVBD_ROW_INACTIVE); });
}

void AvbdDynamicsContext::clipRigidContactStep(PxU32 i, PxVec3& dp,
	PxVec3& dth) const
{
  // Directional pair trust region.  A core-overlap row retains the soft
  // endpoint's remaining inertial displacement so both endpoints leave the
  // invalid topology coherently.  Once live clearance is positive, row C
  // owns the shell and replaying pHat-p would oppose elastic separation.
  if(i + 1 >= mRowProgram.blockOffsets.size())
    return;
  const PxReal tolerance = 1.0e-6f;
  auto candidateGap = [&](const AvbdRowConstRef& row, PxReal fraction)
  {
    PxTransform candidateBody(mBodies[i].p,mBodies[i].q);
    avbdApplyDelta(candidateBody,dp*fraction,dth*fraction);
    const AvbdRigidPoseView poses={mBodies,&i,&candidateBody,1};
    return avbdHeterogeneousRowC(row,mRowSupports,mBlocks,poses,contactGeometry())+
      PxReal(avbdHeterogeneousAdmissionOffset(row));
  };
  auto rigidJacobian = [&](const AvbdRowConstRef& row, PxVec3& jLin, PxVec3& jAng)
  {
    jLin = PxVec3(0.0f);
    jAng = PxVec3(0.0f);
    for(PxU32 supportIndex = row.header().supportStart;
        supportIndex < row.header().supportStart + row.header().supportCount; ++supportIndex)
    {
      if(supportIndex >= mRowSupports.size()) break;
      const AvbdRowSupport& support = mRowSupports[supportIndex];
      if(support.block != i || support.block >= mBlocks.size())
        continue;
      PxVec3 supportLin, supportAng;
      if(avbdHeterogeneousSupportJacobian(row, support,
          mBlocks[support.block], mBodies, contactGeometry(), mRowSupports, mBlocks, supportLin, supportAng))
      {
        jLin += supportLin;
        jAng += supportAng;
      }
    }
  };
  for(PxU32 slot = mRowProgram.blockOffsets[i];
      slot < mRowProgram.blockOffsets[i + 1]; ++slot)
  {
    if(slot >= mRowProgram.blockRows.size()) break;
    const PxU32 rowIndex = mRowProgram.blockRows[slot];
    if(rowIndex >= mRows.size()) continue;
    const AvbdRowConstRef& row = mRows[rowIndex];
    if((row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_SOFT_PAIR |
        eAVBD_ROW_INACTIVE)) !=
        (eAVBD_ROW_CONTACT | eAVBD_ROW_SOFT_PAIR) || !row.header().supportCount)
      continue;
    const PxReal currentGap = avbdHeterogeneousRowC(row, mRowSupports,
        mBlocks, mBodies, contactGeometry()) +
        PxReal(avbdHeterogeneousAdmissionOffset(row));
    const PxReal lower = PxReal(avbdHeterogeneousAdmissionLower(row,
        double(currentGap)));
    const PxReal rel0 = candidateGap(row, 0.0f);
    const PxReal rel1 = candidateGap(row, 1.0f);
    if(!PxIsFinite(currentGap) || !PxIsFinite(lower) ||
      !PxIsFinite(rel0) || !PxIsFinite(rel1))
    {
      dp = PxVec3(0.0f);
      dth = PxVec3(0.0f);
      return;
    }
    // The contact shell bounds the potential support, not outward motion.
    // Inertial separation must survive warmstart and final BDF unchanged.
    if(rel1 >= lower - tolerance)
      continue;
    if(rel0 > lower + tolerance)
    {
      PxReal lo = 0.0f, hi = 1.0f;
      for(PxU32 iteration = 0; iteration < 10; ++iteration)
      {
        const PxReal mid = 0.5f * (lo + hi);
		if(candidateGap(row, mid) >= lower + tolerance) lo = mid;
        else hi = mid;
      }
      dp *= lo;
      dth *= lo;
      continue;
    }
    PxVec3 jLin, jAng;
    rigidJacobian(row, jLin, jAng);
    const PxReal vn = jLin.dot(dp) + jAng.dot(dth);
    const PxReal metric = jLin.magnitudeSquared() + jAng.magnitudeSquared();
    if(vn < 0.0f && metric > 1.0e-20f)
    {
      const PxReal s = vn / metric;
      dp -= jLin * s;
      dth -= jAng * s;
    }
  }
}

// Final-pose admission follows every candidate transformation.  Checking the
// Newton direction before OGC projection is insufficient: that projection can
// cross another contact, and finite rotation is not its tangent Jacobian.
// Initially overlapping rows may recover, but may not deepen their gap.
bool AvbdDynamicsContext::projectNpCandidate(PxU32 i,PxTransform& candidate) const
{
    if(mStep.localBudget || mNpDomainMode!=2) return true;
    if(i+1>=mRowProgram.blockOffsets.size()) return true;
    AvbdOgcProjectionPlanes planes;
    const PxVec3& anchor=mBodies[i].p;
    const PxVec3d target=avbdOgcDouble(candidate.p)-avbdOgcDouble(anchor);
    for(PxU32 slot=mRowProgram.blockOffsets[i];slot<mRowProgram.blockOffsets[i+1];++slot)
    {
        const AvbdRowConstRef& row=mRows[mRowProgram.blockRows[slot]];
        if(row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT) continue;
        const AvbdNpDomain& domain=mNpDomains[row.header().geometryIndex-1];
        const PxU32 indices[2]={row.header().bodyA,row.header().bodyB};PxTransform poses[2];
        for(PxU32 s=0;s<2;++s)
            poses[s]=indices[s]==i?candidate:(indices[s]<mBodies.size()?
                PxTransform(mBodies[indices[s]].p,mBodies[indices[s]].q):domain.fixedPose[s]);
        AvbdOgcProjectionPlane plane;
        plane.n=domain.normal*(row.header().bodyA==i?1.0:-1.0);
        plane.b=plane.n.dot(target)-avbdNpGap(domain,poses);
        plane.roundoff=0;
        planes.pushBack(plane);
    }
    if(planes.empty()) return true;
    // Geometry restoration must stay in the same public motion subspace as
    // the Newton proposal. Projecting in unrestricted R^3 can move a locked
    // axis even when avbdProjectLockedDelta removed that component above.
    if(mBodies[i].core)
        for(PxU32 axis=0;axis<3;++axis)
            if(mBodies[i].core->lockFlags & PxRigidDynamicLockFlag::Enum(1u<<axis))
            {
                AvbdOgcProjectionPlane plane;
                plane.n=PxVec3d(0);plane.n[axis]=1;plane.b=plane.roundoff=0;
                planes.pushBack(plane);plane.n[axis]=-1;planes.pushBack(plane);
            }
    avbdOgcCompactParallelPlanes(planes);
    PxVec3d projected;
    if(!avbdOgcProjectHalfSpacesExact(planes,target,projected,PxMat33d(PxIdentity))) return false;
    return avbdOgcPublishRegionPoint(anchor,projected,planes,PX_MAX_F32,candidate.p,&anchor);
}

bool AvbdDynamicsContext::rigidContactCandidateAdmissible(PxU32 i,
 const PxTransform& candidate)
{
 if(!candidate.p.isFinite() || !candidate.q.isFinite()) return false;
 if(i + 1 >= mRowProgram.blockOffsets.size()) return true;
 const AvbdRigidPoseView poses = {mBodies,&i,&candidate,1};
 for(PxU32 slot=mRowProgram.blockOffsets[i];slot<mRowProgram.blockOffsets[i+1];++slot)
 {
  const AvbdRowConstRef& row=mRows[mRowProgram.blockRows[slot]];
  if(mNpDomainMode==2 && !(mStep.localBudget?localNpCandidateAdmissible(row,poses):
      avbdNpCandidateAdmissible(row,mNpDomains,mBodies,poses,mKernels))) return false;
  if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
  {
   if(!tetrahedronPath(row,mTetrahedronMaterials[row.header().geometryIndex-1],mRowSupports,mBodies,poses)) return false;
   continue;
  }
  if(!row.header().supportCount || !(row.header().flags&eAVBD_ROW_CONTACT) ||
   (row.header().flags&eAVBD_ROW_INACTIVE)) continue;
  if(!avbdHeterogeneousCandidateAdmissible(row,mRowSupports,mBlocks,mBodies,
   mBlocks,poses,contactGeometry())) return false;
 }
 return true;
}

template<class PoseView>
double AvbdDynamicsContext::rigidRowEnergy(PxU32 index,PxU32 block,
    const PoseView& poses,PxReal invH2) const
{
    const AvbdRowConstRef& row=mRows[index];
    if((row.header().flags&eAVBD_ROW_INACTIVE)) return 0;
    if(row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME)
    {
        const auto e=evaluateFrameRequest<AvbdElasticFrameEnergy>(row,mElasticFrameMaterials,poses,
            mKernels,AvbdElasticFrameRequest::Energy);
        return e.valid?e.energy:double(PX_MAX_F32);
    }
    if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
    {
        PxVec3d points[4];tetrahedronPoints(row,mRowSupports,poses,points);
        AvbdTetrahedronEvaluation e;
        AvbdTetrahedronArgs args={&mTetrahedronMaterials[row.header().geometryIndex-1],points,&e};
        mKernels.launch(AVBD_TETRAHEDRON_MATERIAL,&args);
        return e.valid ? e.energy : double(PX_MAX_F32);
    }
    if(row.header().supportCount && (row.header().flags&eAVBD_ROW_CONTACT))
        return avbdHeterogeneousContactPotential(row,
            avbdHeterogeneousRowC(row,mRowSupports,mBlocks,poses,contactGeometry())).energy;
    if(row.header().kernelId==eAVBD_ROW_KERNEL_ATTACHMENT)
    {
        const PxReal c=avbdHeterogeneousRowC(row,mRowSupports,mBlocks,poses,contactGeometry());
        return c*(row.scalar().lambda+.5f*row.scalar().rho*c);
    }
    const PxReal C=avbdPrimalC(row,poses,getLengthScale());
    PxReal penalty=(row.header().flags&eAVBD_ROW_DRIVE) ? avbdDrivePenalty(row,mBodies) : avbdHardJointPenalty(row,mBodies,invH2);
    PX_UNUSED(block);
    return (row.header().flags&(eAVBD_ROW_CONTACT|eAVBD_ROW_UNILATERAL))==(eAVBD_ROW_CONTACT|eAVBD_ROW_UNILATERAL) ?
        avbdNpContactPotential(row,C).energy :
        avbdBoundedRowEnergy(C,penalty,row.scalar().lambda,row.scalar().minLambda,row.scalar().maxLambda);
}

double AvbdDynamicsContext::rigidBlockEnergy(PxU32 i,
    const PxTransform& candidate,PxReal invH2) const
{
    const AvbdSolverBody& original=mBodies[i];
    const AvbdRigidPoseView poses={mBodies,&i,&candidate,1};
    const PxVec3 dp=candidate.p-original.pHat;
    double energy=0.5f*original.massInvH2*dp.magnitudeSquared()+
        avbdRotationInertialPotential(original.qHat,candidate.q,original.inertiaWorldInvH2).energy;
    if(!mStep.localBudget)energy=PxReal(energy);
    for(PxU32 slot=mRowProgram.blockOffsets[i];slot<mRowProgram.blockOffsets[i+1];++slot)
        // The four-point potential is evaluated in double. Round the sum,
        // not the individual contribution before Armijo admission.
    {
        const double value=rigidRowEnergy(mRowProgram.blockRows[slot],i,poses,invH2);
        energy=mStep.localBudget?energy+value:PxReal(energy+value);
    }
    return energy;
}

namespace {
void avbdLaunchLocalGroup(const KernelTable& table,AvbdNonlinearGroup& block)
{AvbdNonlinearGroupArgs args={&block};table.launch(AVBD_NONLINEAR_GROUP,&args);}
void avbdLaunchLocalGroup(const KernelTable& table,AvbdLocalNonlinearGroup& block)
{AvbdLocalNonlinearGroupArgs args={&block};table.launch(AVBD_LOCAL_NONLINEAR_GROUP,&args);}
}
void AvbdDynamicsContext::solveLocalRigidGroup(PxU32 root,PxReal invH2)
{
    if(mRigidGroups[root].count<=4)solveLocalRigidGroupKernel<4>(root,invH2);
    else solveLocalRigidGroupKernel<16>(root,invH2);
}

template<PxU32 Capacity> void AvbdDynamicsContext::solveLocalRigidGroupKernel(PxU32 root,PxReal invH2)
{
    auto& group=mRigidGroups[root];++group.attempts;
    PxU64 profileStart=mStep.profile?PxTime::getCurrentCounterValue():0;
    AvbdNonlinearGroupT<Capacity> block;block.count=group.count;
    PxTransform original[Capacity],candidate[Capacity];bool scalar=true;
    double initialEnergy=0;
    const AvbdFrameAssemblyTarget target={&block.gradient[0][0],&block.K[0][0][0],&block.M[0][0][0],&initialEnergy,group.count,36*Capacity};
    AvbdFrameAssemblyPacket frames(mElasticFrameMaterials,mKernels,group.blocks,target);
    auto flushFrames=[&]()
    {
        const PxU64 start=mStep.profile?PxTime::getCurrentCounterValue():0;
        const bool valid=frames.flush();
        if(mStep.profile)group.assemblyPartTicks[1]+=PxTime::getCurrentCounterValue()-start;
        return valid;
    };
    for(PxU32 s=0;s<group.count;++s)
    {
        const auto& b=mBodies[group.blocks[s]];original[s]=PxTransform(b.p,b.q);
        block.locked[s]=b.core?PxU32(b.core->lockFlags):0;
        if(group.blocks[s]<mLocalTranslationClosed.size() && mLocalTranslationClosed[group.blocks[s]])block.translationMass[s]=1.0/double(b.invMass);
        const PxMat33d inertia(avbdOgcDouble(b.inertiaWorldInvH2.column0),avbdOgcDouble(b.inertiaWorldInvH2.column1),avbdOgcDouble(b.inertiaWorldInvH2.column2));
        const auto rotation=avbdRotationInertialPotential<double>(avbdNpRotation(b.qHat),avbdNpRotation(b.q),inertia);
        const auto dp=avbdOgcDouble(b.p)-avbdOgcDouble(b.pHat);
        initialEnergy+=.5*double(b.massInvH2)*dp.magnitudeSquared()+rotation.energy;
        for(PxU32 d=0;d<3;++d)
        {
            block.K[s][s][7*d]=block.M[s][s][7*d]=b.massInvH2;
            block.gradient[s][d]=double(b.massInvH2)*dp[d];block.gradient[s][d+3]=rotation.gradient[d];
            for(PxU32 k=0;k<3;++k)block.K[s][s][6*(d+3)+k+3]=block.M[s][s][6*(d+3)+k+3]=rotation.gaussNewton[d][k];
        }
    }
    if(mStep.profile){group.assemblyPartTicks[0]+=PxTime::getCurrentCounterValue()-profileStart;group.assemblyPartRows[0]+=group.count;}
    for(PxU32 slot=0;slot<group.rowCount;++slot)
    {
        const auto& entry=mRigidGroupRows[group.rowStart+slot];const PxU32 r=entry.row;const auto& row=mRows[r];
        if(row.header().flags&eAVBD_ROW_INACTIVE)continue;
        scalar &= !((row.header().flags&eAVBD_ROW_CONTACT) && row.scalar().positionTangent && row.scalar().mu>0);
        if(row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME)
        {
            const PxU64 start=mStep.profile?PxTime::getCurrentCounterValue():0;
            if(!frames.append(row.header().geometryIndex-1,row.header().bodyA,row.header().bodyB,mBodies))
            {PxAtomicExchange(&mLocalDomainFailed,1);return;}
            if(mStep.profile){group.assemblyPartTicks[1]+=PxTime::getCurrentCounterValue()-start;++group.assemblyPartRows[1];}
            ++group.materialEvaluations;group.materialEndpointUses+=entry.supportCount;
            continue;
        }
        if(!flushFrames()){PxAtomicExchange(&mLocalDomainFailed,1);return;}
        const PxU64 rowStart=mStep.profile?PxTime::getCurrentCounterValue():0;
        if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
        {
            const auto e=evaluateTetrahedron(row);if(!e.valid){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            initialEnergy+=e.energy;
            double J[4][6]={};PxU32 members[4],memberCount=0;
            for(PxU32 k=0;k<4;++k)
            {
                const auto& support=mRowSupports[row.header().supportStart+k];
                for(PxU32 s=0;s<group.count;++s)if(support.block==group.blocks[s])
                {
                    PxU32 local=0;while(local<memberCount && members[local]!=s)++local;
                    if(local==memberCount)members[memberCount++]=s;
                    const auto arm=avbdOgcDouble(mBodies[group.blocks[s]].q.rotate(support.localPoint));const auto angular=arm.cross(e.jacobians[k]);
                    for(PxU32 d=0;d<3;++d){J[local][d]+=e.jacobians[k][d];J[local][d+3]+=angular[d];}
                    break;
                }
            }
            for(PxU32 a=0;a<memberCount;++a)for(PxU32 d=0;d<6;++d)
                block.gradient[members[a]][d]+=e.gradient*J[a][d];
            block.addRankOne(members,memberCount,J,e.curvature);
            ++group.materialEvaluations;group.materialEndpointUses+=entry.supportCount;
        }
        else
        {
            if(scalar)initialEnergy+=rigidRowEnergy(r,entry.representative,mBodies,invH2);
            AvbdGroupFactorT<2> factors[3]={};const PxU32 offsets[2]={0,3};
            PxU32 members[2],memberCount=0;
            for(PxU32 s=0;s<group.count;++s)if(row.header().bodyA==group.blocks[s] || row.header().bodyB==group.blocks[s])
            {
                members[memberCount++]=s;
                AvbdRigidAssemblyT<PxReal,2> a={};a.coupledRows=&r;a.count=1;a.member=memberCount-1;a.factors=factors;a.consistent=true;
                a.factorOffsets=offsets;a.lockedMask=block.locked[s];
                accumulateRigidConstraintRow(group.blocks[s],r,invH2,a);
                if(!a.consistent){PxAtomicExchange(&mLocalDomainFailed,1);return;}
                for(PxU32 d=0;d<6;++d)
                {block.gradient[s][d]+=a.gradient[d];for(PxU32 k=0;k<6;++k)
                    {block.K[s][s][6*d+k]+=a.hessian[d][k];block.M[s][s][6*d+k]+=a.hessian[d][k];}}
            }
            if(entry.supportCount>1)for(const auto& f:factors)if(f.curvature!=0)
                block.addRankOne(members,memberCount,f.jacobian,double(f.curvature),true);
        }
        if(mStep.profile)
        {const PxU32 part=row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON?2:3;group.assemblyPartTicks[part]+=PxTime::getCurrentCounterValue()-rowStart;++group.assemblyPartRows[part];}
    }
    if(!flushFrames()){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    if(mStep.profile){const PxU64 now=PxTime::getCurrentCounterValue();group.assemblyTicks+=now-profileStart;profileStart=now;}
    avbdLaunchLocalGroup(mKernels,block);group.products+=block.factorizations;group.rightHandSides+=block.substitutions;
    if(mStep.profile){const PxU64 now=PxTime::getCurrentCounterValue();group.factorTicks+=now-profileStart;profileStart=now;}
    if(!block.solved){PxAtomicExchange(&mLocalDomainFailed,1);return;}++group.kernelSolved;
    auto energy=[&](const PxTransform* states)
    {
        const AvbdRigidPoseView poses={mBodies,group.blocks,states,group.count};double value=0;
        for(PxU32 s=0;s<group.count;++s)
        {
            const auto& b=mBodies[group.blocks[s]];const auto dp=avbdOgcDouble(states[s].p)-avbdOgcDouble(b.pHat);
            const PxMat33d inertia(avbdOgcDouble(b.inertiaWorldInvH2.column0),avbdOgcDouble(b.inertiaWorldInvH2.column1),avbdOgcDouble(b.inertiaWorldInvH2.column2));
            value+=.5*double(b.massInvH2)*dp.magnitudeSquared()+avbdRotationInertialPotential<double>(avbdNpRotation(b.qHat),avbdNpRotation(states[s].q),inertia).energy;
        }
        for(PxU32 s=0;s<group.rowCount;++s)
        {const auto& e=mRigidGroupRows[group.rowStart+s];value+=rigidRowEnergy(e.row,e.representative,poses,invH2);}
        return value;
    };
    const double initial=scalar?initialEnergy:0;double fraction=1;
    for(PxU32 search=0;search<16;++search,fraction*=.5)
    {
        PxAtomicIncrement(&mStep.localSearchTrials);
        bool valid=true;double delta[Capacity][6]={},slope=0,quadratic=0;
        for(PxU32 s=0;s<group.count;++s)
        {
            candidate[s]=original[s];const auto* x=block.step[s];
            avbdApplyDelta(candidate[s],PxVec3(PxReal(x[0]*fraction),PxReal(x[1]*fraction),PxReal(x[2]*fraction)),
                PxVec3(PxReal(x[3]*fraction),PxReal(x[4]*fraction),PxReal(x[5]*fraction)));
            valid &= candidate[s].isSane();const auto dp=candidate[s].p-original[s].p;
            const auto angle=avbdAcceptedRotationDisplacement(original[s].q,candidate[s].q);
            for(PxU32 d=0;d<3;++d){delta[s][d]=dp[d];delta[s][d+3]=angle[d];}
            for(PxU32 d=0;d<6;++d)slope+=block.gradient[s][d]*delta[s][d];
        }
        // Pure predicates form one admission conjunction. Reject a failed
        // descent model before paying for the complete geometric certificate.
        if(!valid || slope>0)continue;
        PxU64 phaseStart=mStep.profile?PxTime::getCurrentCounterValue():0;
        if(!scalar)for(PxU32 i=0;i<group.count;++i)for(PxU32 j=0;j<group.count;++j)
            for(PxU32 d=0;d<6;++d)for(PxU32 k=0;k<6;++k)
                quadratic+=delta[i][d]*(block.metric?block.M:block.K)[i][j][6*d+k]*delta[j][k];
        const double trial=scalar?energy(candidate):slope+.5*quadratic;
        if(mStep.profile)group.modelTicks+=PxTime::getCurrentCounterValue()-phaseStart;
        if(!std::isfinite(trial) || (scalar?trial>initial+1.e-4*slope:trial>0))continue;
        const AvbdRigidPoseView poses={mBodies,group.blocks,candidate,group.count};
        if(mStep.profile)phaseStart=PxTime::getCurrentCounterValue();
        for(PxU32 s=0;valid && s<group.rowCount;++s)
        {
            const auto& row=mRows[mRigidGroupRows[group.rowStart+s].row];
            if(!(row.header().flags&eAVBD_ROW_INACTIVE) && row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
                valid=tetrahedronPath(row,mTetrahedronMaterials[row.header().geometryIndex-1],mRowSupports,mBodies,poses);
            if(valid && row.header().kernelId==eAVBD_ROW_KERNEL_NP_CONTACT)
                valid=localNpCandidateAdmissible(row,poses);
        }
        if(mStep.profile)group.geometryTicks+=PxTime::getCurrentCounterValue()-phaseStart;
        if(!valid)continue;
        for(PxU32 s=0;s<group.count;++s)
        {
            const PxU32 b=group.blocks[s];const auto dp=candidate[s].p-original[s].p;
            const auto angle=avbdAcceptedRotationDisplacement(original[s].q,candidate[s].q);
            mPrimalSteps[b]=PxMax(dp.magnitude()/(getLengthScale()*AVBD_ARTIC_POSITION_REL_TOL),angle.magnitude()/AVBD_ARTIC_ROTATION_TOL);
            if(candidate[s].p!=original[s].p || !(candidate[s].q==original[s].q))mPrimalChangedIteration[b]=mStep.iteration+1;
            mBodies[b].p=candidate[s].p;mBodies[b].q=candidate[s].q;
        }
        if(mStep.profile)group.searchTicks+=PxTime::getCurrentCounterValue()-profileStart;
        ++group.accepted;return;
    }
    if(mStep.profile)group.searchTicks+=PxTime::getCurrentCounterValue()-profileStart;
    PxAtomicIncrement(&mStep.localSearchExhausted);
}

bool AvbdDynamicsContext::solvePrimalRigidGroup(PxU32 root,PxReal invH2)
{
    AvbdRigidGroup& group=mRigidGroups[root];++group.attempts;
    bool scalarObjective=true,sharedContact=false;
    for(PxU32 r=0;r<group.rowCount;++r)
    {
        const AvbdRigidGroupRow& entry=mRigidGroupRows[group.rowStart+r];
        const AvbdRowConstRef& row=mRows[entry.row];
        if(row.header().flags&eAVBD_ROW_INACTIVE) continue;
        if((row.header().flags&eAVBD_ROW_CONTACT) && row.header().supportCount && entry.supportCount>1)
        { ++group.unsupported;return false; }
        if((row.header().flags&eAVBD_ROW_CONTACT) && row.scalar().positionTangent && row.scalar().mu>0) scalarObjective=false;
        if((row.header().flags&eAVBD_ROW_CONTACT) && entry.supportCount>1) sharedContact=true;
    }
    if(!scalarObjective)++group.frictionAttempts;
    if(sharedContact)++group.sharedContactAttempts;
    PxTransform original[4],candidate[4];
    AvbdRigidAssembly assemblies[4];
    PxReal diagonal[4][6][6],gradient[4][6],step[4][6];
    PxInlineArray<AvbdGroupFactor,32> factors;
    const PxU32 factorCount=mRigidGroupFactorOffsets[group.sharedStart+group.sharedCount]-
        mRigidGroupFactorOffsets[group.sharedStart];
    factors.resize(factorCount);
    PxMemZero(factors.begin(),factors.size()*sizeof(AvbdGroupFactor));
    for(PxU32 s=0;s<group.count;++s)
    {
        const PxU32 block=group.blocks[s];original[s]=PxTransform(mBodies[block].p,mBodies[block].q);
        assemblies[s].coupledRows=mRigidGroupSharedRows.begin()+group.sharedStart;
        assemblies[s].count=group.sharedCount;assemblies[s].member=s;
        assemblies[s].factors=factors.begin();assemblies[s].consistent=true;
        assemblies[s].factorOffsets=mRigidGroupFactorOffsets.begin()+group.sharedStart;
        assemblies[s].lockedMask=mBodies[block].core?PxU32(mBodies[block].core->lockFlags):0;
        initializeRigidMaterial(mBodies[block],assemblies[s].hessian,assemblies[s].gradient);
    }
    // Row-major assembly retains each block's row order. Material kernels
    // execute once, then feed every incident endpoint and shared factor.
    for(PxU32 r=0;r<group.rowCount;++r)
    {
        const auto& entry=mRigidGroupRows[group.rowStart+r];const AvbdRowConstRef& row=mRows[entry.row];
        if((row.header().flags&eAVBD_ROW_INACTIVE))continue;
        if(accumulateRigidMaterialRow(entry.row,group.blocks,group.count,assemblies))
        {
            ++group.materialEvaluations;group.materialEndpointUses+=entry.supportCount;
            if(mLocalDomainFailed)return true;
        }
        else mRowProgram.forEachBlock(entry.row,[&](PxU32 block)
        {
            for(PxU32 s=0;s<group.count;++s)if(group.blocks[s]==block)
                accumulateRigidConstraintRow(block,entry.row,invH2,assemblies[s]);
        });
    }
    for(PxU32 s=0;s<group.count;++s)
    {
        auto& a=assemblies[s];
        avbdKernelRestrict6(a.hessian,a.gradient,a.lockedMask);
        for(PxU32 d=0;d<6;++d)a.hessian[d][d]+=AVBD_REG;
        PxMemCopy(diagonal[s],a.hessian,sizeof(diagonal[s]));
        PxMemCopy(gradient[s],a.gradient,sizeof(gradient[s]));
        if(!a.consistent)return false;
    }
    bool solved=false;
    AvbdLinearResponseResult linear={false,0,0,0};
    AvbdGroupSolveArgs args={group.count,diagonal,gradient,factors.begin(),factors.size(),step,&solved,&linear};
    mKernels.launch(AVBD_RIGID_GROUP,&args);
    group.products+=linear.operatorCalls;
    if(!solved) return false;
    ++group.kernelSolved;
    auto energy=[&](const PxTransform* states)
    {
        const AvbdRigidPoseView poses={mBodies,group.blocks,states,group.count};double value=0;
        for(PxU32 s=0;s<group.count;++s)
        {
            const AvbdSolverBody& body=mBodies[group.blocks[s]];
            const PxVec3 dp=states[s].p-body.pHat;
            value+=.5*double(body.massInvH2)*double(dp.magnitudeSquared())+
                avbdRotationInertialPotential(body.qHat,states[s].q,body.inertiaWorldInvH2).energy;
        }
        for(PxU32 r=0;r<group.rowCount;++r)
        {
            const AvbdRigidGroupRow& entry=mRigidGroupRows[group.rowStart+r];
            value+=rigidRowEnergy(entry.row,entry.representative,poses,invH2);
        }
        return value;
    };
    auto admissible=[&]()
    {
        const AvbdRigidPoseView poses={mBodies,group.blocks,candidate,group.count};
        for(PxU32 s=0;s<group.count;++s)
        {
            if(!candidate[s].p.isFinite() || !candidate[s].q.isFinite()) return false;
        }
        for(PxU32 r=0;r<group.rowCount;++r)
        {
            const AvbdRowConstRef& row=mRows[mRigidGroupRows[group.rowStart+r].row];
            if(row.header().flags&eAVBD_ROW_INACTIVE) continue;
            if(mNpDomainMode==2 && !avbdNpCandidateAdmissible(row,mNpDomains,mBodies,poses,mKernels)) return false;
            if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON)
            {
                if(!tetrahedronPath(row,mTetrahedronMaterials[row.header().geometryIndex-1],mRowSupports,mBodies,poses)) return false;
            }
            else if(row.header().supportCount && (row.header().flags&eAVBD_ROW_CONTACT) &&
                !avbdHeterogeneousCandidateAdmissible(row,mRowSupports,mBlocks,mBodies,mBlocks,poses,contactGeometry())) return false;
        }
        return true;
    };
    const double initial=scalarObjective?energy(original):0;PxReal fraction=1;
    for(PxU32 search=0;search<16;++search)
    {
        double slope=0,delta[24]={};
        for(PxU32 s=0;s<group.count;++s)
        {
            candidate[s]=original[s];
            PxVec3 dp(step[s][0]*fraction,step[s][1]*fraction,step[s][2]*fraction);
            PxVec3 dth(step[s][3]*fraction,step[s][4]*fraction,step[s][5]*fraction);
            avbdProjectLockedDelta(mBodies[group.blocks[s]],dp,dth);avbdApplyDelta(candidate[s],dp,dth);
            const PxVec3 accepted=candidate[s].p-original[s].p;
            const PxVec3 angle=avbdAcceptedRotationDisplacement(original[s].q,candidate[s].q);
            for(PxU32 k=0;k<3;++k) slope+=double(gradient[s][k])*accepted[k]+double(gradient[s][k+3])*angle[k];
            for(PxU32 k=0;k<3;++k){delta[6*s+k]=accepted[k];delta[6*s+k+3]=angle[k];}
        }
        if(admissible())
        {
            const double trial=scalarObjective?energy(candidate):avbdGroupModelChange(args,delta);
            // Coulomb is non-associated. This is an assembled model test,
            // not a fictitious global friction energy or an NCP certificate.
            if(PxIsFinite(trial) && (scalarObjective?
                (slope<=0 && trial<=initial+1.e-4*slope):trial<=0))
            {
                for(PxU32 s=0;s<group.count;++s)
                {
                    const PxU32 b=group.blocks[s];
                    const PxReal dp=(candidate[s].p-original[s].p).magnitude();
                    const PxReal angle=avbdAcceptedRotationDisplacement(original[s].q,candidate[s].q).magnitude();
                    mPrimalSteps[b]=PxMax(dp/(getLengthScale()*AVBD_ARTIC_POSITION_REL_TOL),angle/AVBD_ARTIC_ROTATION_TOL);
                    if(candidate[s].p!=original[s].p || !(candidate[s].q==original[s].q)) mPrimalChangedIteration[b]=mStep.iteration+1;
                    mBodies[b].p=candidate[s].p;mBodies[b].q=candidate[s].q;
                }
                ++group.accepted;
                if(!scalarObjective)++group.frictionAccepted;
                if(sharedContact)++group.sharedContactAccepted;
                return true;
            }
        }
        fraction*=.5f;
    }
    return false;
}

PxReal AvbdDynamicsContext::solvePrimalRigidBlock(PxU32 i, PxReal invH2)
{
	if(mStep.localBudget)return solveLocalRigidBlock(i);
	const PxReal invLengthScale = 1.0f / PxMax(getLengthScale(), 1.0e-6f);
	PxVec3 dp, dth, gradientLinear, gradientAngular;
	PxReal metric[6][6]={};
	solvePrimalBody(i, invH2, dp, dth, gradientLinear, gradientAngular,
		mNpDomainMode==2?metric:NULL);
	const AvbdSolverBody& body = mBodies[i];
	const PxTransform original(body.p,body.q);
	// Position-friction currently projects an anisotropic tangent force and
	// has no matching scalar primitive. Preserve that existing program until
	// its variational law is specified; do not invent a false energy oracle.
	bool scalarObjective = true;
	for(PxU32 slot = mRowProgram.blockOffsets[i]; slot < mRowProgram.blockOffsets[i + 1]; ++slot)
	{
		const AvbdRowConstRef& row = mRows[mRowProgram.blockRows[slot]];
		if((row.header().flags & (eAVBD_ROW_CONTACT|eAVBD_ROW_INACTIVE))==eAVBD_ROW_CONTACT && row.scalar().positionTangent && row.scalar().mu > 0.0f)
			scalarObjective = false;
	}
	const double initialEnergy = scalarObjective ? rigidBlockEnergy(i, original, invH2) : 0.0f;
	// A prescribed boundary can make the initial iterate infeasible at the
	// end time. A feasible constrained proposal has priority over its finite
	// unconstrained energy. Every geometric/material check still applies.
	const bool initialFeasible = mNpDomainMode!=2 || rigidContactCandidateAdmissible(i,original);
	PxReal fraction = 1.0f;
	for(PxU32 search = 0; search < (scalarObjective || mNpDomainMode==2 ? 16u : 1u); ++search)
	{
		if(mStep.localBudget)PxAtomicIncrement(&mStep.localSearchTrials);
		PxVec3 trialDp = dp * fraction, trialDth = dth * fraction;
		clipRigidContactStep(i, trialDp, trialDth);
		PxTransform candidate = original;
		avbdApplyDelta(candidate, trialDp, trialDth);
		if(!projectNpCandidate(i,candidate)) { fraction*=0.5f;continue; }
		const PxVec3 acceptedDp = candidate.p - original.p;
		const PxVec3 acceptedDth = avbdAcceptedRotationDisplacement(original.q, candidate.q);
		const PxReal slope = gradientLinear.dot(acceptedDp) + gradientAngular.dot(acceptedDth);
		const double trialEnergy = scalarObjective ? rigidBlockEnergy(i, candidate, invH2) : 0.0f;
        const double sufficientDecrease=mStep.localBudget?initialEnergy+1.0e-4f*slope:
            double(PxReal(initialEnergy)+1.0e-4f*slope);
		bool modelAccepted=true;
		if(!scalarObjective && mNpDomainMode==2 && initialFeasible)
		{
			// Non-associated Coulomb has no claimed scalar energy here. Its
			// assembled positive local model still must decrease after a
			// projection that can radically change the original Newton step.
			const double delta[6]={acceptedDp.x,acceptedDp.y,acceptedDp.z,
				acceptedDth.x,acceptedDth.y,acceptedDth.z};
			double quadratic=0;
			for(PxU32 r=0;r<6;++r) for(PxU32 c=0;c<6;++c)
				quadratic+=delta[r]*metric[r][c]*delta[c];
			modelAccepted=PxIsFinite(quadratic) && double(slope)+.5*quadratic<=0;
		}
		if(rigidContactCandidateAdmissible(i, candidate) &&
			modelAccepted &&
			(!scalarObjective || (!initialFeasible && PxIsFinite(trialEnergy)) || (slope <= 0.0f && PxIsFinite(trialEnergy) &&
			trialEnergy <= sufficientDecrease)))
		{
			mBodies[i].p = candidate.p;
			mBodies[i].q = candidate.q;
			return PxMax(acceptedDp.magnitude() * invLengthScale / AVBD_ARTIC_POSITION_REL_TOL,
				acceptedDth.magnitude() / AVBD_ARTIC_ROTATION_TOL);
		}
		fraction *= 0.5f;
	}
    if(mStep.localBudget)PxAtomicIncrement(&mStep.localSearchExhausted);
    if(!rigidContactCandidateAdmissible(i,original))
    {
        PxGetFoundation().error(PxErrorCode::eDEBUG_INFO,PX_FL,
            "AVBD domain diagnostic: committed Rigid6 state outside contact domain.");
        PxAtomicExchange(&mLocalDomainFailed,1);
    }
	return 0.0f;
}

void AvbdDynamicsContext::applyDual()
{
    applyDualRange(0,mRows.size());writeJointDualState();
}

bool AvbdDynamicsContext::localNpCandidateAdmissible(const AvbdRowConstRef& row,
    const AvbdRigidPoseView& poses)
{
    if(row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT)return true;
    // Use precisely the final publication predicate, including its bounded
    // alternative axes. All members of a coupled group move in one proposal;
    // never project the bodies separately or change the accepted direction.
    AvbdNpQuery query;
    const bool accepted=avbdNpCandidateAdmissible(row,mNpDomains,mBodies,poses,mKernels,&query);
    PxAtomicIncrement(&mStep.localPrimalQueries);
    PxAtomicAdd(&mStep.localPrimalPointVisits,PxI64(query.pointVisits));
    return accepted;
}

void AvbdDynamicsContext::validateLocalRow(PxU32 index)
{
    const auto& row=mRows[index];
    if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
    {
        PxU32 visits=0;
        if(!avbdNodalContactPath(mNodalContacts[row.header().geometryIndex-1],
            [&](PxU32 b){return mBlocks[b].particle->p;},mBodies,mStep.countNodalSolves?&visits:NULL))PxAtomicExchange(&mLocalDomainFailed,1);
        if(mStep.countNodalSolves){PxAtomicIncrement(&mStep.localQueries);PxAtomicAdd(&mStep.localPointVisits,PxI64(visits));}
        return;
    }
    if(row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT)return;
    const auto& domain=mNpDomains[row.header().geometryIndex-1];
    const PxU32 ids[2]={row.header().bodyA,row.header().bodyB};PxTransform start[2],end[2];
    for(PxU32 s=0;s<2;++s)
    {
        start[s]=ids[s]<mBodies.size()?PxTransform(mBodies[ids[s]].pStart,mBodies[ids[s]].qStart):domain.fixedPose[s];
        end[s]=ids[s]<mBodies.size()?PxTransform(mBodies[ids[s]].p,mBodies[ids[s]].q):domain.fixedPose[s];
    }
    bool accepted=false;AvbdNpQuery query;
    AvbdNpDomainArgs args={&domain,start,end,&accepted,&query};mKernels.launch(AVBD_NP_DOMAIN,&args);
    PxAtomicIncrement(&mStep.localQueries);
    PxAtomicAdd(&mStep.localPointVisits,PxI64(query.pointVisits));
    if(accepted && query.axisFrame!=PX_MAX_U32)PxAtomicIncrement(&mStep.localTransportedProofs);
    if(accepted && query.axisCandidates)PxAtomicIncrement(&mStep.localReselectedProofs);
    if(accepted && index<mContactWarm.size())
    {
        PxVec3d axis=query.proofAxis;
        if(query.axisFrame<2)
            axis=avbdNpRotation(end[query.axisFrame].q).rotate(
                avbdNpRotation(start[query.axisFrame].q).rotateInv(domain.normal));
        mContactWarm[index].proofAxis=axis.getNormalized();
    }
    if(!accepted && PxAtomicExchange(&mLocalDomainFailed,1)==0)
    {
        double gap=DBL_MAX;for(double x:query.gap)gap=PxMin(gap,x);
        std::fprintf(stderr,"[AVBD_LOCAL_GEOMETRY] status=REJECTED valid=%u gap=%.17g roundoff=%.17g row=%u\n",
            unsigned(query.valid),gap,query.roundoff,index);
        double witness=DBL_MAX,force=0;
        for(PxU32 r=index;r<mContactRowCount && (r==index || mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT);++r)
        {witness=PxMin(witness,double(avbdPrimalC(mRows[r],mBodies,getLengthScale())));force+=mRows[r].scalar().lambda;}
        std::fprintf(stderr,"[AVBD_LOCAL_WITNESS] gap=%.17g endGap=%.17g force=%.17g points=%u,%u\n",
            witness,avbdNpGap(domain,end),force,query.point[4][0],query.point[4][1]);
        std::fprintf(stderr,"[AVBD_LOCAL_COORDINATE] initialGap=%.17g offset=%.9g radius=%.9g normal=%.9g,%.9g,%.9g axis=%.9g,%.9g,%.9g\n",
            avbdNpGap(domain,start),double(row.scalar().contactOffset),double(row.scalar().contactRadius),double(row.scalar().n.x),double(row.scalar().n.y),double(row.scalar().n.z),domain.normal.x,domain.normal.y,domain.normal.z);
        if(const char* path=std::getenv("PHYSX_AVBD_COMMON_FAILED_NP"))
        if(FILE* file=std::fopen(path,"wb"))
        {
            const auto vector=[&](const PxVec3d& v){std::fprintf(file,"[%.17g,%.17g,%.17g]",v.x,v.y,v.z);};
            const auto pose=[&](const PxTransform& p){std::fprintf(file,"[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",
                double(p.p.x),double(p.p.y),double(p.p.z),double(p.q.x),double(p.q.y),double(p.q.z),double(p.q.w));};
            std::fprintf(file,"{\"reason\":\"local_complete_path\",\"normal\":");vector(domain.normal);
            std::fprintf(file,",\"rest\":%.17g,\"fixedAxis\":%u,\"sides\":[",domain.restDistance,unsigned(domain.fixedAxis));
            for(PxU32 s=0;s<2;++s)
            {
                std::fprintf(file,"%s{\"body\":%u,\"radius\":%.17g,\"start\":",s?",":"",ids[s],domain.radius[s]);pose(start[s]);
                std::fprintf(file,",\"end\":");pose(end[s]);std::fprintf(file,",\"points\":[");
                for(PxU32 i=0;i<domain.points[s].size();++i){if(i)std::fprintf(file,",");vector(domain.points[s][i]);}
                std::fprintf(file,"]}");
            }
            std::fprintf(file,"],\"rows\":[");
            for(PxU32 r=index;r<mContactRowCount && (r==index || mRows[r].header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT);++r)
            {
                const auto& contact=mRows[r];std::fprintf(file,"%s{\"n\":",r==index?"":",");vector(avbdOgcDouble(contact.scalar().n));
                std::fprintf(file,",\"a\":");vector(avbdOgcDouble(contact.scalar().ogcRA));std::fprintf(file,",\"b\":");vector(avbdOgcDouble(contact.scalar().ogcRB));
                std::fprintf(file,",\"offset\":%.9g,\"lambda\":%.9g}",double(contact.scalar().contactOffset),double(contact.scalar().lambda));
            }
            std::fprintf(file,"]}");std::fclose(file);
        }
    }
}

void AvbdDynamicsContext::applyDualRange(PxU32 begin,PxU32 end)
{
	const PxReal invH2 = mDt > 0.0f ? 1.0f / (mDt * mDt) : 0.0f;
	const PxReal beta = AVBD_BETA / PxMax(getLengthScale(), 1.0e-6f);
	for(PxU32 ri = begin; ri < end; ++ri)
	{
		AvbdRowRef row = mRows[ri];
		if((row.header().flags & eAVBD_ROW_INACTIVE))
			continue;
        if(mStep.incidencePrimal && row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_SHELL)
        {
            if(mStep.iteration+1==mStep.iterationLimit)
            {
                const auto& h=row.header();
                auto& material=mNodalShellMaterials[h.geometryIndex-1];
                PxVec3 start[3],endPose[3];
                for(PxU32 i=0;i<3;++i)
                {
                    const auto slot=h.supportStart+i;
                    const auto& p=*mBlocks[mRowSupports[slot].block].particle;
                    start[i]=p.pStart;endPose[i]=p.p;
                    material.previous[i]=mIncidenceDisplacement[slot];
                }
                if(!avbdMembranePath(start,endPose))PxAtomicExchange(&mLocalDomainFailed,1);
            }
            continue;
        }
        if(row.header().kernelId==eAVBD_ROW_KERNEL_TETRAHEDRON || row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME ||
            avbdIsNodalMaterial(row.header().kernelId)) continue;
		PxReal C;
		if(row.header().supportCount)
		{
			if(avbdHeterogeneousPositionAl(row))
				C = avbdHeterogeneousAlC(row, mRowSupports, mBlocks, mBodies, contactGeometry());
			else
			{
				C = avbdHeterogeneousRowC(row, mRowSupports, mBlocks, mBodies, contactGeometry());
			}
		}
		else
			C = avbdPrimalC(row, mBodies, getLengthScale());
		if(row.header().flags & eAVBD_ROW_CONTACT)
		{
			const PxReal oldFn = row.scalar().lambda;
			PxReal Fn;
			if(mStep.localBudget)
            {
                // Same normal envelope and Coulomb disk, inertial-response
                // preconditioned dual ascent after the colored primal joins.
                const auto mobility=[&](const PxVec3& axis,bool tangent=false)
                {
                    double value=0;
                    if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
                    {
                        const auto frame=avbdNodalRigidFrame(row,contactGeometry());
                        const auto current=frame==PX_MAX_U32?axis:avbdNodalWorldAxis(mBodies[frame].qStart,mBodies[frame].q,axis);
                        for(PxU32 k=row.header().supportStart;k<row.header().supportStart+row.header().supportCount;++k)
                        {
                            const auto& support=mRowSupports[k];const auto& block=mBlocks[support.block];
                            if(block.particle){value+=support.weight*support.weight*block.particle->invMass;continue;}
                            const auto& body=mBodies[block.objectIndex];
                            auto arm=(tangent || frame!=PX_MAX_U32)?
                                (support.block==row.header().bodyA?row.scalar().rA:row.scalar().rB):support.localPoint;
                            if(frame!=PX_MAX_U32)arm-=body.q.rotateInv(avbdNodalWitnessTravel(row,mRowSupports,mBlocks,mBodies))*support.weight;
                            value+=support.weight*support.weight*avbdEndpointAxisResponse(body,current,arm);
                        }
                        return value;
                    }
                    if(row.header().bodyA<mBodies.size())
                    {const auto& b=mBodies[row.header().bodyA];value+=avbdJointEndpointMobility(b,axis,b.q.rotate(tangent?row.scalar().rA:row.scalar().ogcRA).cross(axis));}
                    if(row.header().bodyB<mBodies.size())
                    {const auto& b=mBodies[row.header().bodyB];value+=avbdJointEndpointMobility(b,axis,b.q.rotate(tangent?row.scalar().rB:row.scalar().ogcRB).cross(axis));}
                    return value;
                };
                const PxReal target=avbdNpContactPotential(row,C).gradient;
                if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT && row.scalar().contactRadius>0)
                    Fn=target; // The shared normal metric already includes inertial compliance.
                else
                {
                    const double tau=avbdInertialDualStep(row.scalar().rho,mobility(row.scalar().n),mDt);
                    Fn=PxReal(oldFn+(target-oldFn)*(tau/row.scalar().rho));
                }
                if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT && row.scalar().mu==0)
                {
                    row.scalar().lambda=Fn;
                    row.scalar().lambdaT0=row.scalar().lambdaT1=0;
                    continue;
                }
                const PxVec3 rel=row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT?
                    avbdNodalContactSlip(row,mRowSupports,mBlocks,mBodies,contactGeometry()):avbdContactRelDisp(row,mBodies);
                const PxReal c0=(1-AVBD_ALPHA)*row.scalar().tangentC0+row.scalar().t0.dot(rel);
                const PxReal c1=(1-AVBD_ALPHA)*row.scalar().tangentC1+row.scalar().t1.dot(rel);
                const PxReal trial0=row.scalar().lambdaT0+row.scalar().rhoT0*c0,trial1=row.scalar().lambdaT1+row.scalar().rhoT1*c1;
                const bool stick=trial0*trial0+trial1*trial1<=row.scalar().mu*row.scalar().mu*target*target;
                PxReal f0=PxReal(row.scalar().lambdaT0+avbdInertialDualStep(row.scalar().rhoT0,mobility(row.scalar().t0,true),mDt)*c0);
                PxReal f1=PxReal(row.scalar().lambdaT1+avbdInertialDualStep(row.scalar().rhoT1,mobility(row.scalar().t1,true),mDt)*c1);
                avbdProjectCoulomb(Fn,row.scalar().mu,f0,f1);
                row.scalar().lambda=Fn;row.scalar().lambdaT0=f0;row.scalar().lambdaT1=f1;
                if(row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
                {
                    const PxU32 slot=mNodalContacts[row.header().geometryIndex-1].frictionSlot;
                    if(slot!=PX_MAX_U32)
                    {
                        auto& next=mNodalFriction[slot];const auto& law=row.scalar();
                        next.traction=law.t0*f0+law.t1*f1;
                        const auto& c=mNodalContacts[row.header().geometryIndex-1];
                        if(c.rigidFrame!=PX_MAX_U32)
                        {
                            const auto& body=mBodies[c.rigidFrame];
                            next.normal=avbdNodalWorldAxis(body.qStart,body.q,law.n);
                            next.traction=avbdNodalWorldAxis(body.qStart,body.q,next.traction);
                        }
                        const PxU32 soft=c.geometry->deformable[0]?0u:1u;
                        const bool anchored=c.geometry->shapes[1-soft].kind==PxsContactShapeGeometry::ePLANE &&
                            c.geometry->deformable[soft]->simulation.arity==4 && mIslandSoftCache[mBlocks[c.first[soft]].objectIndex].planarSupport;
                        next.stickSlip=anchored && stick && Fn<0?law.t0*(law.tangentC0+law.t0.dot(rel))+
                            law.t1*(law.tangentC1+law.t1.dot(rel)):PxVec3(0);
                    }
                }
                if(ri<mContactWarm.size())
                {
                    auto& warm=mContactWarm[ri];warm.lambda=Fn;warm.lambdaT0=f0;warm.lambdaT1=f1;
                    warm.rho=row.scalar().rho;warm.rhoT0=row.scalar().rhoT0;warm.rhoT1=row.scalar().rhoT1;
                    warm.t0=row.scalar().t0;warm.t1=row.scalar().t1;warm.stick=PxU8(stick);
                }
                continue;
            }
			if(row.header().supportCount && !avbdHeterogeneousPositionAl(row))
			{
				// Heterogeneous OGC normal is a geometric potential: publish its
				// current force for material capacity and reports.
				Fn = avbdHeterogeneousContactPotential(row, C).gradient;
			}
			else
			{
				// demo3d updateDual: lambda <- min(0, rho*C + lambda); the penalty
				// grows only while the row is load-bearing (Eq. 16).
				Fn = row.header().supportCount ?
					avbdHeterogeneousContactPotential(row, C).gradient :
					avbdNpContactPotential(row,C).gradient;
				const PxReal residual=!row.header().supportCount && row.scalar().contactRadius>0 ?
					(Fn-oldFn)/row.scalar().rho:C;
				if(Fn < 0.0f)
					row.scalar().rho = PxMin(AVBD_CONTACT_PENALTY_MAX,
						row.scalar().rho + beta * PxAbs(residual));
			}
			PxReal f0 = 0.0f;
			PxReal f1 = 0.0f;
			PxReal cT0 = 0.0f;
			PxReal cT1 = 0.0f;
			bool frictionStick = false;
			if(row.scalar().mu > 0.0f && row.scalar().positionTangent)
			{
				const PxVec3 rel = row.header().kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT?
                avbdNodalContactSlip(row,mRowSupports,mBlocks,mBodies,contactGeometry()):avbdContactRelDisp(row, mBodies);
				cT0 = (1.0f - AVBD_ALPHA) * row.scalar().tangentC0 + row.scalar().t0.dot(rel);
				cT1 = (1.0f - AVBD_ALPHA) * row.scalar().tangentC1 + row.scalar().t1.dot(rel);
				f0 = row.scalar().rhoT0 * cT0 + row.scalar().lambdaT0;
				f1 = row.scalar().rhoT1 * cT1 + row.scalar().lambdaT1;
				// Stick/slip is classified from the trial dual force.  The
				// projected force is always inside the Coulomb disk and therefore
				// cannot be used as the classifier: doing so grows tangent rho on
				// every sliding sweep and turns an impact manifold into a weld.
				const PxReal preLen = PxSqrt(f0 * f0 + f1 * f1);
				const PxReal boundT = PxAbs(Fn) * row.scalar().mu;
				avbdProjectCoulomb(Fn, row.scalar().mu, f0, f1);
				frictionStick = preLen <= boundT;
				if(frictionStick)
				{
					row.scalar().rhoT0 = PxMin(AVBD_CONTACT_PENALTY_MAX,
						row.scalar().rhoT0 + beta * PxAbs(cT0));
					row.scalar().rhoT1 = PxMin(AVBD_CONTACT_PENALTY_MAX,
						row.scalar().rhoT1 + beta * PxAbs(cT1));
				}
			}
			row.scalar().lambda = Fn;
			row.scalar().lambdaT0 = f0;
			row.scalar().lambdaT1 = f1;
			// A tangent is conditioned only after the Signorini row becomes
			// load-bearing.  This gives a loaded Coulomb coordinate the same
			// dimensionless stiffness as its normal without turning separated
			// speculative contacts into a tangential glue network.
			if(Fn < 0.0f && oldFn >= 0.0f && row.scalar().positionTangent)
				avbdFloorLoadedContactTangentRho(row, mBodies, mDt);
			if(row.scalar().forceWriteback)
				*row.scalar().forceWriteback = -row.scalar().lambda;
			if(ri < mContactWarm.size())
			{
				mContactWarm[ri].lambda = row.scalar().lambda;
				mContactWarm[ri].lambdaT0 = row.scalar().lambdaT0;
				mContactWarm[ri].lambdaT1 = row.scalar().lambdaT1;
				mContactWarm[ri].rho = row.scalar().rho;
				mContactWarm[ri].rhoT0 = row.scalar().rhoT0;
				mContactWarm[ri].rhoT1 = row.scalar().rhoT1;
				mContactWarm[ri].t0 = row.scalar().t0;
				mContactWarm[ri].t1 = row.scalar().t1;
				mContactWarm[ri].stick = PxU8(frictionStick);
			}
		}
		else if(row.header().flags & eAVBD_ROW_DRIVE)
		{
			// A drive is a physical spring/damper potential. A persistent AL
			// multiplier would apply the authored force a second time.
			row.scalar().lambda = 0.0f;
		}
		else
		{
			PxReal dualStep = row.scalar().rho;
			if((row.header().flags & eAVBD_ROW_DRIVE) == 0)
				{
                const PxReal penalty=avbdHardJointPenalty(row,mBodies,invH2,&dualStep);
                if(mStep.localBudget)
                {
                    double mobility=0;
                    if(row.header().bodyA<mBodies.size())mobility+=avbdJointEndpointMobility(mBodies[row.header().bodyA],row.scalar().jLinA,row.scalar().jAngA);
                    if(row.header().bodyB<mBodies.size())mobility+=avbdJointEndpointMobility(mBodies[row.header().bodyB],row.scalar().jLinB,row.scalar().jAngB);
                    dualStep=PxReal(avbdInertialDualStep(penalty,mobility,mDt));
                }
            }
			const bool pointLock = (row.header().flags & eAVBD_ROW_POINT_LOCK) != 0;
			// Point locks use AVBD_2's in-sweep relaxation with the bounded
			// dual step above. Other hard 1D rows retain demo3d lambda += K*C.
			const PxReal retention = pointLock && !mStep.localBudget ? AVBD_POINT_LOCK_DUAL_DECAY : 1.0f;
			PxReal next = row.scalar().lambda * retention + dualStep * C;
			next = PxMin(row.scalar().maxLambda, PxMax(row.scalar().minLambda, next));
			row.scalar().lambda = next;
			// Point-lock rho is fixed. Other hard rows grow it inside the
			// force bounds; angular rows use demo3d's smaller betaAng.
			if(!pointLock && next > row.scalar().minLambda && next < row.scalar().maxLambda)
			{
				const bool angularRow = row.scalar().articJoint ?
					row.scalar().articAxis < PxArticulationAxis::eX :
					(row.scalar().jLinA.magnitudeSquared() < 1.0e-12f &&
					 row.scalar().jLinB.magnitudeSquared() < 1.0e-12f);
				const PxReal rowBeta = angularRow ?
					AVBD_BETA * AVBD_JOINT_BETA_ANGULAR_SCALE : beta;
				row.scalar().rho = PxMin(AVBD_RHO_MAX, row.scalar().rho + rowBeta * PxAbs(C));
			}
		}
	}
}

static PxReal avbdMaterialEndpointResponse(const AvbdSolverBody& b,
	const PxVec3& axis, const PxVec3& arm)
{
	if(!avbdBodyResponds(b))
		return 0.0f;
	const PxVec3 jAngLocal = b.qStart.rotateInv(arm.cross(axis));
	return b.invMass +
		jAngLocal.x * jAngLocal.x * b.invInertia.x +
		jAngLocal.y * jAngLocal.y * b.invInertia.y +
		jAngLocal.z * jAngLocal.z * b.invInertia.z;
}

static PxReal avbdMaterialAxisResponse(const AvbdRowConstRef& row,
	const AvbdMaterialRow& material, const PxArray<AvbdSolverBody>& bodies,
	const PxVec3& axis)
{
	PxReal response = 0.0f;
	if(row.header().bodyA < bodies.size())
		response += avbdMaterialEndpointResponse(bodies[row.header().bodyA], axis,
			material.armA);
	if(row.header().bodyB < bodies.size())
		response += avbdMaterialEndpointResponse(bodies[row.header().bodyB], axis,
			material.armB);
	return PxIsFinite(response) && response > 1.0e-12f ? response : 0.0f;
}

static PxReal avbdMaterialLinearAxisResponse(const AvbdRowConstRef& row,
	const PxArray<AvbdSolverBody>& bodies)
{
	PxReal response = 0.0f;
	if(row.header().bodyA < bodies.size() && avbdBodyResponds(bodies[row.header().bodyA]))
		response += bodies[row.header().bodyA].invMass;
	if(row.header().bodyB < bodies.size() && avbdBodyResponds(bodies[row.header().bodyB]))
		response += bodies[row.header().bodyB].invMass;
	return PxIsFinite(response) && response > 1.0e-12f ? response : 0.0f;
}

static PxReal avbdMaterialCrossResponse(const AvbdRowConstRef& row,
	const AvbdMaterialRow& material, const PxArray<AvbdSolverBody>& bodies,
	const PxVec3& axis0, const PxVec3& axis1)
{
	PxReal response = 0.0f;
	const PxU32 ids[2] = { row.header().bodyA, row.header().bodyB };
	const PxVec3 arms[2] = { material.armA, material.armB };
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		if(ids[endpoint] >= bodies.size() ||
			!avbdBodyResponds(bodies[ids[endpoint]]))
			continue;
		const AvbdSolverBody& b = bodies[ids[endpoint]];
		const PxVec3 j0 = b.qStart.rotateInv(arms[endpoint].cross(axis0));
		const PxVec3 j1 = b.qStart.rotateInv(arms[endpoint].cross(axis1));
		response += b.invMass * axis0.dot(axis1) +
			j0.x * j1.x * b.invInertia.x +
			j0.y * j1.y * b.invInertia.y +
			j0.z * j1.z * b.invInertia.z;
	}
	return PxIsFinite(response) ? response : 0.0f;
}

static void avbdApplyMaterialAxisImpulse(const AvbdRowConstRef& row,
	const AvbdMaterialRow& material, PxArray<AvbdSolverBody>& bodies,
	PxArray<AvbdMaterialVelocityDelta>& velocityDelta,
	const PxVec3& axis, double impulse, bool linearOnly = false)
{
	if(!PxIsFinite(PxReal(impulse)) || PxAbs(PxReal(impulse)) < 1.0e-12f)
		return;
	const PxU32 ids[2] = { row.header().bodyA, row.header().bodyB };
	const PxVec3 arms[2] = { material.armA, material.armB };
	const PxReal signs[2] = { 1.0f, -1.0f };
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		if(ids[endpoint] >= bodies.size())
			continue;
		AvbdSolverBody& b = bodies[ids[endpoint]];
		if(!avbdBodyResponds(b))
			continue;
		const PxVec3 endpointAxis = axis * signs[endpoint];
		velocityDelta[ids[endpoint]].addLinear(endpointAxis,
			impulse * double(b.invMass));
		if(!linearOnly)
			velocityDelta[ids[endpoint]].addAngular(avbdWorldInvInertiaMul(
				b, arms[endpoint].cross(endpointAxis)), impulse);
	}
}


void AvbdDynamicsContext::prepareImpactMaterial(PxReal dt,const PxArray<PxU8>& materialBody)
{
	const PxReal bounceThreshold = getBounceThreshold();
	const PxU32 nbRows = mRows.size();
	mStep.materialPending=true;
    mContactMaterial.begin(mBodies.size());
    auto& velocityDelta=mContactMaterial.delta;
	// Material velocity ownership is an island property, not a scene-wide
	// "has any joint" switch.  Publish the rigid contact components once and
	// mark those containing a bilateral row.  Their pose-derived velocity is
	// the coupled Position-AL certificate; a contact-only M^-1 projection cannot
	// replace one span without the corresponding joint-nullspace mobility.
	// prepareRigidVelocities compiled this finalized row program. Only staged
	// velocities changed since then; reuse the same immutable components.
	const PxArray<PxU32>& componentParent = mMaterialComponentParentScratch;
	PxArray<PxU8>& jointComponent = mMaterialJointComponentScratch;
	jointComponent.resize(mBlocks.size());
	for(PxU32 i = 0; i < jointComponent.size(); ++i) jointComponent[i] = 0;
	for(PxU32 ri = 0; ri < nbRows; ++ri)
	{
		const AvbdRowConstRef& row = mRows[ri];
		if((row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_INACTIVE)) == 0)
			mRowProgram.forEachBlock(ri, [&](PxU32 block)
			{ jointComponent[componentParent[block]] = 1; });
	}
	mMaterialRowsScratch.clear();
	mMaterialRowsScratch.reserve(nbRows);
	for(PxU32 ri = 0; ri < nbRows; ++ri)
	{
		const AvbdRowConstRef& row = mRows[ri];
		if((row.header().flags & eAVBD_ROW_CONTACT) == 0 ||
			(row.header().flags & (eAVBD_ROW_INACTIVE | eAVBD_ROW_SOFT_PAIR)))
			continue;
		const bool dynamicA = row.header().bodyA < mBodies.size() &&
			avbdBodyResponds(mBodies[row.header().bodyA]);
		const bool dynamicB = row.header().bodyB < mBodies.size() &&
			avbdBodyResponds(mBodies[row.header().bodyB]);
		const bool jointCoupledPair = dynamicA && dynamicB &&
			componentParent[row.header().bodyA] == componentParent[row.header().bodyB] &&
			jointComponent[componentParent[row.header().bodyA]] != 0;
		// The legacy joint-island restitution coordinate is translational.  Its
		// approach epoch must use the same linear Jacobian as its response; mixing
		// point angular velocity into the target but not the mobility changes the
		// restitution law.
		const PxReal vnIn = jointCoupledPair ?
			row.scalar().n.dot(mBodies[row.header().bodyA].linVel - mBodies[row.header().bodyB].linVel) :
			avbdContactVn(row, mBodies, true);
		// Restitution is a material event, not a proximity response.  The DCD
		// neighborhood is intentionally wider than the authored contact band for
		// OGC; a point is bounce-eligible only if its incoming normal motion can
		// actually close the detection separation during this step.  This is the
		// same PhysX/TGS eligibility inequality used by the distilled AVBD_2
		// manifold owner, expressed with AvbdRow's signed solve-start separation.
		const PxReal approach = -vnIn;
		const PxReal detectionRate = dt > 0.0f ? row.scalar().contactOffset / dt : PX_MAX_F32;
		const bool impactRow = vnIn < bounceThreshold && approach > detectionRate;
		const bool jointLinearRestitution = jointCoupledPair && impactRow &&
			row.scalar().restitution > 0.0f;
		// The joint component owns its passive material velocity through the
		// Position-AL pose certificate.  A contact-only endpoint in the same row
		// was reset to solve-start velocity, however, so it must still receive its
		// missing Position-AL material certificate below.  The row itself remains
		// pose-owned and
		// does not run a free-body material projection unless it is the authored
		// restitution coordinate.
		const bool jointPoseOwnedMaterial = jointCoupledPair &&
			!jointLinearRestitution;
		const bool reconstructA = dynamicA && row.header().bodyA < materialBody.size() &&
			materialBody[row.header().bodyA];
		const bool reconstructB = dynamicB && row.header().bodyB < materialBody.size() &&
			materialBody[row.header().bodyB];
		const bool finiteNormalBudget = row.scalar().minLambda > -1.0e15f;
		// NP rigid rows are Position-AL: the material NCP consumes the AL load
		// (-lambda dt) as its normal capacity.  A finite maxImpulse restarts
		// from the incoming velocity and lets the projection spend the authored
		// impulse exactly.
		const bool replayCertificate =
			(!dynamicA || reconstructA) && (!dynamicB || reconstructB);
		AvbdMaterialRow material;
		material.rowIndex = ri;
		PxVec3 freshA = row.scalar().ogcRA;
		PxVec3 freshB = row.scalar().ogcRB;
		if(row.header().bodyA < mBodies.size())
			freshA = avbdWorldPoint(mBodies[row.header().bodyA], row.scalar().ogcRA);
		if(row.header().bodyB < mBodies.size())
			freshB = avbdWorldPoint(mBodies[row.header().bodyB], row.scalar().ogcRB);
		const auto condensed = [&](PxU32 index) {
            return index < mBodies.size() && mBodies[index].core && !mBodies[index].isKinematic &&
                !mBodies[index].invMass && mBodies[index].core->inverseMass > 0;
        };
        const PxVec3 materialPoint = condensed(row.header().bodyA) ? freshA :
            (condensed(row.header().bodyB) ? freshB : (freshA + freshB)*.5f);
		material.armA = row.header().bodyA < mBodies.size() ?
			materialPoint - mBodies[row.header().bodyA].p : PxVec3(0.0f);
		material.armB = row.header().bodyB < mBodies.size() ?
			materialPoint - mBodies[row.header().bodyB].p : PxVec3(0.0f);
		material.positionNormalImpulse = finiteNormalBudget && replayCertificate ?
			0.0f : PxMax(0.0f, -row.scalar().lambda * dt);
		material.normalImpulse = 0.0f;
		material.normalCapacity = material.positionNormalImpulse;
		material.tangentImpulse0 = 0.0f;
		material.tangentImpulse1 = 0.0f;
		// A fully contact-only row has the exact mapping
		// v = vMinus + M^-1 J^T (-lambda dt).  For a mixed row the same mapping is
		// exact on its contact-only endpoint, while the joint endpoint keeps the
		// response of the complete coupled pose objective.  The row-level material
		// KKT therefore starts with zero incremental impulse unless both endpoints
		// belong to the contact-only velocity epoch.
		if(replayCertificate && !finiteNormalBudget &&
			(row.header().flags & eAVBD_ROW_FRESH) == 0)
		{
			material.normalImpulse = PxMax(0.0f, -row.scalar().lambda * dt);
			material.tangentImpulse0 = -row.scalar().lambdaT0 * dt;
			material.tangentImpulse1 = -row.scalar().lambdaT1 * dt;
		}
		// A hard-joint island does not yet publish a joint-nullspace material
		// mobility.  AVBD_2 therefore leaves the complete Position-AL response in
		// the pose velocity and closes only the missing dynamic-dynamic bounce
		// coordinate with the two linear inverse masses.  Treating this row as a
		// free spatial contact both replays an AL impulse that cannot be inverted
		// independently and diverts the restitution impulse into the light link's
		// angular modes.
		material.normalLinearOnly = PxU8(jointLinearRestitution);
		material.normalCapacity = PxMax(material.normalCapacity,
			material.positionNormalImpulse + material.normalImpulse);
		avbdProjectCoulomb(material.normalCapacity, row.scalar().mu,
			material.tangentImpulse0, material.tangentImpulse1);
		const bool activeMaterialRow = row.scalar().lambda < 0.0f ||
			avbdEvalC(row, mBodies) <= 0.0f;
		const bool solveRow = impactRow || activeMaterialRow;
		material.normalTarget = impactRow && row.scalar().restitution > 0.0f ?
			-row.scalar().restitution * vnIn : 0.0f;
		material.normalResponse = solveRow ?
			(material.normalLinearOnly ?
				avbdMaterialLinearAxisResponse(row, mBodies) :
				avbdMaterialAxisResponse(row, material, mBodies, row.scalar().n)) : 0.0f;
		if(!replayCertificate && !jointLinearRestitution &&
			(row.header().flags & eAVBD_ROW_FRESH) == 0)
		{
			// Reconstruct the missing endpoint certificate from the final AL force
			// integral itself.  This is the exact Position-AL scatter at the
			// persistent contact anchor; projecting pose velocity point-by-point is
			// not equivalent for a multi-point rigid manifold and loses friction.
			const PxU32 ids[2] = { row.header().bodyA, row.header().bodyB };
			const bool reconstruct[2] = { reconstructA, reconstructB };
			const PxReal signs[2] = { 1.0f, -1.0f };
			const PxReal impulses[3] = {
				PxMax(0.0f, -row.scalar().lambda * dt),
				-row.scalar().lambdaT0 * dt,
				-row.scalar().lambdaT1 * dt
			};
			const PxVec3 axes[3] = { row.scalar().n, row.scalar().t0, row.scalar().t1 };
			const PxU32 componentCount = row.scalar().positionTangent ? 3u : 1u;
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				if(!reconstruct[endpoint] || ids[endpoint] >= mBodies.size())
					continue;
				const AvbdSolverBody& b = mBodies[ids[endpoint]];
				for(PxU32 component = 0; component < componentCount; ++component)
				{
					const PxVec3 localPoint = component == 0 ?
						(endpoint == 0 ? row.scalar().ogcRA : row.scalar().ogcRB) :
						(endpoint == 0 ? row.scalar().rA : row.scalar().rB);
					const PxVec3 persistentArm = b.q.rotate(localPoint);
					const PxReal impulse = impulses[component];
					if(!PxIsFinite(impulse) || PxAbs(impulse) <= 1.0e-12f)
						continue;
					const PxVec3 endpointAxis = axes[component] * signs[endpoint];
					const PxVec3 linearDelta = endpointAxis *
						(impulse * b.invMass);
					const PxVec3 angularDelta = avbdWorldInvInertiaMul(b,
						persistentArm.cross(endpointAxis)) * impulse;
					velocityDelta[ids[endpoint]].addLinear(linearDelta);
					velocityDelta[ids[endpoint]].addAngular(angularDelta);
				}
				material.normalCapacity = PxMax(material.normalCapacity,
					impulses[0]);
			}
		}
		const PxReal totalNormalBudget =
			(finiteNormalBudget && dt > 0.0f) ?
			PxMax(0.0f, -row.scalar().minLambda * dt) : PX_MAX_F32;
		material.maxNormalImpulse = totalNormalBudget == PX_MAX_F32 ?
			PX_MAX_F32 : PxMax(0.0f,
				totalNormalBudget - material.positionNormalImpulse);
		material.solveNormal = PxU8(material.normalResponse > 0.0f &&
			!jointPoseOwnedMaterial);
		material.solveTangent = 0;
		// The free-rigid frontier has an exact M^-1 contact operator.  A hard
		// joint island requires its joint-nullspace mobility; until that Schur
		// operator is assembled, retain the coupled pose response and solve only
		// the material residual, matching the AVBD_2 joint path.
		material.tangentStep = 0.0f;
        material.tangentInverse=PxVec3(0.f);
		if(solveRow && row.scalar().mu > 0.0f && !jointLinearRestitution &&
			!jointPoseOwnedMaterial)
		{
			const PxReal w0 = avbdMaterialAxisResponse(row, material, mBodies, row.scalar().t0);
			const PxReal w1 = avbdMaterialAxisResponse(row, material, mBodies, row.scalar().t1);
			if(w0 > 0.0f && w1 > 0.0f)
			{
				const PxReal w01 = avbdMaterialCrossResponse(row, material, mBodies,
					row.scalar().t0, row.scalar().t1);
				const PxReal trace = w0 + w1;
				const PxReal disc = PxSqrt(PxMax(0.0f,
					(w0 - w1) * (w0 - w1) + 4.0f * w01 * w01));
				const PxReal lipschitz = 0.5f * (trace + disc);
				if(lipschitz > 1.0e-12f)
				{
                    const double det=double(w0)*w1-double(w01)*w01;
                    if(det>0)
                        material.tangentInverse=PxVec3(PxReal(w1/det),PxReal(-w01/det),PxReal(w0/det));
					material.tangentStep = AVBD_BOUNCE_SOR / lipschitz;
					material.solveTangent = 1;
				}
			}
		}
		mMaterialRowsScratch.pushBack(material);
		// Contact-only components restart at solve-start velocity and replay the
		// AL certificate.  A hard-joint component already contains that response
		// in its pose-derived velocity, so only later impulse deltas are scattered.
		if(replayCertificate)
		{
			avbdApplyMaterialAxisImpulse(row, material, mBodies, velocityDelta, row.scalar().n,
				material.normalImpulse);
			avbdApplyMaterialAxisImpulse(row, material, mBodies, velocityDelta, row.scalar().t0,
				material.tangentImpulse0);
			avbdApplyMaterialAxisImpulse(row, material, mBodies, velocityDelta, row.scalar().t1,
				material.tangentImpulse1);
		}
        if(!material.solveNormal && !material.solveTangent) continue;
        // The endpoints' poses and incoming velocity are frozen for this step.
        // The accumulated velocity deltas remain live throughout the solve.
        auto& program=mContactMaterial;
        const PxU32 ids[2]={row.header().bodyA,row.header().bodyB};
        if(program.patches.empty() || program.patches.back().count==AvbdContactMaterial::PointsPerPatch ||
            program.patches.back().body[0]!=ids[0] || program.patches.back().body[1]!=ids[1])
        {
            AvbdMaterialPatch patch={program.points.size(),0,{ids[0],ids[1]},{0,0}};
            for(PxU32 s=0;s<2;++s)
                if(ids[s]<mBodies.size() && avbdBodyResponds(mBodies[ids[s]]))
                    patch.invMass[s]=mBodies[ids[s]].invMass;
            program.patches.pushBack(patch);
        }
        ++program.patches.back().count;
        AvbdMaterialPoint point={};
        point.state=mMaterialRowsScratch.size()-1; point.friction=row.scalar().mu;
        point.axes[0].direction=row.scalar().n;
        point.axes[1].direction=row.scalar().t0;
        point.axes[2].direction=row.scalar().t1;
        const PxVec3 arms[2]={material.armA,material.armB};
        for(PxU32 s=0;s<2;++s)
        {
            if(ids[s]>=mBodies.size() || !mBodies[ids[s]].core) continue;
            const auto& body=mBodies[ids[s]];
            for(PxU32 k=0;k<3;++k)
            {
                auto& axis=point.axes[k];
                const PxVec3 a=axis.direction;
                const PxVec3 j=(k==0 && material.normalLinearOnly) ? PxVec3(0.f) : arms[s].cross(a);
                axis.angular[s]=j;
                axis.angularResponse[s]=avbdWorldInvInertiaMul(body,j);
                const PxVec3& l=body.outputLinVel; const PxVec3& w=body.outputAngVel;
                const double vn=double(a.x)*l.x+double(a.y)*l.y+double(a.z)*l.z+
                    double(j.x)*w.x+double(j.y)*w.y+double(j.z)*w.z;
                axis.baseSpeed += s ? -vn : vn;
            }
        }
        program.points.pushBack(point);
    }
    mContactMaterial.schedule(componentParent,mStep.workers);
}

bool AvbdDynamicsContext::advanceContactMaterial()
{
    auto& p=mContactMaterial;
    while(!p.patches.empty())
    {
        while(p.islandCursor+1<p.islandOffsets.size())
            if(!dispatchRowRanges(MaterialIslandStage,p.islandCursor,p.islandOffsets.size()-1,1)) return false;
        while(p.color+1<p.offsets.size())
        {
            const PxU32 end=p.offsets[p.color+1];
            while(p.cursor<end)
                if(!dispatchRowRanges(MaterialSolveStage,p.cursor,end,AvbdContactMaterial::PatchesPerTask)) return false;
            if(++p.color+1==p.offsets.size()) p.cursor=0;
        }
        while(p.cursor<p.order.size())
            if(!dispatchRowRanges(MaterialResidualStage,p.cursor,p.order.size(),AvbdContactMaterial::PatchesPerTask)) return false;
        AvbdMaterialResidual residual;
        for(const auto& r:p.residual) residual.include(r.error,r.scale);
        ++p.iteration;
        if(p.iteration==AVBD_MATERIAL_MAX_ITERS || residual.error<=AVBD_MATERIAL_REL_TOL*residual.scale)
        {
            if(std::getenv("PHYSX_AVBD_MATERIAL_STATS"))
                std::fprintf(stderr,"[AVBD_MATERIAL_SWEEP] coupled=%u iterations=%u residual=%.9g points=%u patches=%u islandJobs=%u colored=%u\n",
                    unsigned(p.friction),p.iteration,double(residual.error/residual.scale),p.points.size(),p.patches.size(),
                    p.islandOffsets.empty()?0:p.islandOffsets.size()-1,p.order.size());
            if(p.friction) break;
            p.friction=true; p.iteration=0;
        }
        p.color=p.cursor=p.islandCursor=0;
    }
    return true;
}

void AvbdDynamicsContext::publishContactMaterial()
{
    const auto& velocityDelta=mContactMaterial.delta;
    // Recover support wrenches from the very impulse accepted by material.
    // Material rows are appended in increasing NP row order.
    for(const auto& reaction : mFixedReactions)
    {
        PxU32 lo=0, hi=mMaterialRowsScratch.size();
        while(lo<hi)
        {
            const PxU32 mid=lo+(hi-lo)/2;
            if(mMaterialRowsScratch[mid].rowIndex < reaction.row) lo=mid+1; else hi=mid;
        }
        if(lo==mMaterialRowsScratch.size() || mMaterialRowsScratch[lo].rowIndex!=reaction.row) continue;
        const auto& material=mMaterialRowsScratch[lo];
        const auto row=mRows[reaction.row];
        const PxVec3 impulse = reaction.sign*(row.scalar().n*material.normalImpulse +
            row.scalar().t0*material.tangentImpulse0 + row.scalar().t1*material.tangentImpulse1);
        auto& result=getConstraintWriteBackPool()[reaction.writeback];
        result.linearImpulse += impulse;
        result.angularImpulse += reaction.arm.cross(impulse);
    }
// Publish only the converged net response.  Position-AL removal and total
	// material replay can each be very large for a heavy body on a light joint
	// graph; exposing either intermediate to PxReal body state loses the small
	// physical difference that the material KKT actually solved.
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		AvbdSolverBody& b = mBodies[i];
		if(!b.core || !avbdBodyResponds(b))
			continue;
		const PxVec3 linearDelta = velocityDelta[i].linearPx();
		const PxVec3 angularDelta = velocityDelta[i].angularPx();
		if(!linearDelta.isFinite() || !angularDelta.isFinite())
			continue;
		b.outputLinVel += linearDelta;
		b.outputAngVel += angularDelta;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_X) b.outputLinVel.x = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Y) b.outputLinVel.y = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Z) b.outputLinVel.z = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_X) b.outputAngVel.x = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y) b.outputAngVel.y = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z) b.outputAngVel.z = 0.0f;
	}
}

void AvbdDynamicsContext::prepareRigidVelocities(PxReal dt)
{
    if(mBodies.empty() && !mContactRowCount)return;
	const PxReal invH = dt > 0.0f ? 1.0f / dt : 0.0f;
	// Compile one exclusive writeback epoch per dynamic endpoint.  A body whose
	// incident program is contact-only starts the material solve at solve-start
	// velocity and replays its Position-AL certificate.  A joint/soft endpoint
	// keeps pose-derived velocity; mixed contacts replay the same normal and
	// passive-tangent certificate only on their contact-only endpoint.
	PxArray<PxU8>& materialBody = mMaterialBodyScratch;
	PxArray<PxU8>& hasIncidentRow = mMaterialIncidentScratch;
	PxArray<PxU8>& materialTangentBody = mMaterialTangentBodyScratch;
	materialBody.resize(mBlocks.size());
	for(PxU32 i = mBodies.size(); i < mBlocks.size(); ++i) materialBody[i] = 0;
	hasIncidentRow.resize(mBodies.size());
	materialTangentBody.resize(mBodies.size());
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		// A contact-only body normally publishes the OGC pose velocity. The row
		// pass below retains solve-start velocity only for a phase-I feasibility
		// program, whose geometric repair is not physical momentum.
		materialBody[i] = !mStep.commonProgram && mBodies[i].core && mBodies[i].invMass > 0.0f &&
			!mBodies[i].isKinematic && !mBodies[i].artic;
		hasIncidentRow[i] = 0;
		materialTangentBody[i] = 0;
	}
	PxArray<PxU8> released;
    if(!mReleasedRigidBodies.empty())
    {
        released.resize(mBlocks.size(),0);
        for(PxU32 b : mReleasedRigidBodies) released[b]=1;
    }
    for(PxU32 ri = 0; ri < mRows.size(); ++ri)
	{
		const AvbdRowConstRef& row = mRows[ri];
		if(row.header().flags & eAVBD_ROW_INACTIVE) continue;
		mRowProgram.forEachBlock(ri, [&](PxU32 block)
		{
			if(block >= mBodies.size()) return;
			hasIncidentRow[block] = 1;
			if((row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_SOFT_PAIR)) != eAVBD_ROW_CONTACT)
				materialBody[block] = 0;
		});
	}
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		materialBody[i] = PxU8(materialBody[i] && (hasIncidentRow[i] || (!released.empty() && released[i])));
		hasIncidentRow[i] = 0;
	}
	// Velocity epoch belongs to the complete dynamic component. Resetting a
	// contact-only projectile while its spring/joint endpoint retains BDF
	// replaces only half of the coupled impulse and creates net momentum.
	PxArray<PxU32>& epochParent = mMaterialComponentParentScratch;
	compileVelocityComponents(epochParent);
    // A released component has no physical impulse from a broken equality.
    // Restart all its free endpoints at the same incoming momentum epoch.
    for(PxU32 i=0; i<released.size(); ++i)
        if(released[i]) released[epochParent[i]] = 1;
	for(PxU32 i = 0; i < mBlocks.size(); ++i)
		if(!materialBody[i]) materialBody[epochParent[i]] = 0;
	for(PxU32 i = 0; i < mBodies.size(); ++i)
		materialBody[i] = materialBody[epochParent[i]];
	// An ordinary dynamic-dynamic restitution manifold is reconstructed from
	// the Position-AL residual in AVBD_2: geometry already supplied the
	// inelastic normal reaction, and material adds only the bounce/cone delta.
	// Restoring solve-start velocity here would first erase that reaction and
	// then ask a pointwise free-body solve to rediscover the mobility of an
	// attached endpoint.
	for(PxU32 ri = 0; ri < mRows.size(); ++ri)
	{
		const AvbdRowConstRef& row = mRows[ri];
		if((row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_INACTIVE |
			eAVBD_ROW_SOFT_PAIR)) != eAVBD_ROW_CONTACT ||
			row.scalar().restitution <= 0.0f || row.header().bodyA >= mBodies.size() ||
			row.header().bodyB >= mBodies.size() ||
			!avbdBodyResponds(mBodies[row.header().bodyA]) ||
			!avbdBodyResponds(mBodies[row.header().bodyB]) ||
			row.scalar().minLambda > -1.0e15f ||
            (!released.empty() && released[epochParent[row.header().bodyA]]))
			continue;
		materialBody[row.header().bodyA] = 0;
		materialBody[row.header().bodyB] = 0;
	}
	// Solver-level velocity damping is a joint-program stabilization term in
	// the AVBD_2 step contract.  A free/contact-only island does not run that
	// program: damping it merely because it has no material row changes free
	// fall and makes a later impact look artificially stiff.  Reuse the scratch
	// mask for the bodies that actually participate in an active non-contact
	// hard AL rigid row; finite DRIVE rows already dissipate their relative
	// coordinate through authored damping and must preserve common translation.
	// Public per-body damping remains independent below.
	for(PxU32 ri = 0; ri < mRows.size(); ++ri)
	{
		const AvbdRowConstRef& row = mRows[ri];
		if((row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_INACTIVE |
			eAVBD_ROW_SOFT_PAIR)) == eAVBD_ROW_CONTACT && row.scalar().mu > 0.0f &&
			row.scalar().positionTangent)
		{
			if(row.header().bodyA < mBodies.size())
				materialTangentBody[row.header().bodyA] = 1;
			if(row.header().bodyB < mBodies.size())
				materialTangentBody[row.header().bodyB] = 1;
		}
		if(row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_INACTIVE |
			eAVBD_ROW_SOFT_PAIR | eAVBD_ROW_DRIVE))
			continue;
		if(row.header().bodyA < mBodies.size())
			hasIncidentRow[row.header().bodyA] = 1;
		if(row.header().bodyB < mBodies.size())
			hasIncidentRow[row.header().bodyB] = 1;
	}
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		AvbdSolverBody& b = mBodies[i];
		if(!b.core)
			continue;
		if(b.invMass == 0.0f || b.isKinematic)
		{
			if(b.core->fixedBaseLink)
			{
				b.outputLinVel = PxVec3(0.0f);
				b.outputAngVel = PxVec3(0.0f);
			}
			continue;
		}
		PxVec3 lin;
		PxVec3 ang;
		if(materialBody[i])
		{
			lin = b.linVel;
			ang = b.angVel;
		}
		else
			avbdPoseDeltaVel(b, invH, lin, ang);
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_X) lin.x = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Y) lin.y = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Z) lin.z = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_X) ang.x = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y) ang.y = 0.0f;
		if(b.core->lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z) ang.z = 0.0f;

		b.outputLinVel = lin;
		b.outputAngVel = ang;
	}

	// Damping and public speed limits define the material solve's incoming
	// velocity epoch.  Contact complementarity must be the last writer: damping
	// different endpoint types after friction has closed vt immediately creates
	// new slip, while post-material caps discard a required support impulse.
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		AvbdSolverBody& b = mBodies[i];
		if(!b.core || b.invMass == 0.0f || b.isKinematic)
			continue;
		if(!mStep.commonProgram && !materialBody[i] && hasIncidentRow[i] && !materialTangentBody[i])
		{
			b.outputLinVel *= AVBD_VELOCITY_DAMPING;
			b.outputAngVel *= AVBD_ANGULAR_DAMPING;
		}
		// Pose reconstruction includes constraint-generated velocity modes that
		// did not exist when the inertial target was integrated.  Authored body
		// damping therefore belongs at the final velocity boundary as well.
		if(b.core->linearDamping > 0.0f)
		{
			const PxReal decay = 1.0f / (1.0f + b.core->linearDamping * dt);
			b.outputLinVel *= decay;
		}
		if(b.core->angularDamping > 0.0f)
		{
			const PxReal decay = 1.0f / (1.0f + b.core->angularDamping * dt);
			b.outputAngVel *= decay;
		}
		const PxReal linSq = b.outputLinVel.magnitudeSquared();
		if(linSq > b.core->maxLinearVelocitySq && linSq > 1.0e-12f)
		{
			const PxReal scale = PxSqrt(b.core->maxLinearVelocitySq / linSq);
			b.outputLinVel *= scale;
		}
		const PxReal angSq = b.outputAngVel.magnitudeSquared();
		if(angSq > b.core->maxAngularVelocitySq && angSq > 1.0e-12f)
		{
			const PxReal scale = PxSqrt(b.core->maxAngularVelocitySq / angSq);
			b.outputAngVel *= scale;
		}
	}

	if(!mStep.commonProgram) prepareImpactMaterial(dt,materialBody);
}

void AvbdDynamicsContext::finalizeRigidBodies(PxReal dt)
{
	// This is the only Host publication boundary for Rigid6. Both the rigid-only
	// and heterogeneous material programs have already updated output* in the
	// backend, so no intermediate pose/velocity is visible to PxsBodyCore.
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		AvbdSolverBody& b = mBodies[i];
		if(!b.core)
			continue;
		if(!b.invMass && !b.isKinematic && b.core->inverseMass > 0)
        {
            b.invMass = b.core->inverseMass;
            b.invInertia = b.core->inverseInertia;
        }
        if(b.invMass == 0.0f || b.isKinematic)
		{
			if(b.core->fixedBaseLink)
			{
				b.core->linearVelocity = PxVec3(0.0f);
				b.core->angularVelocity = PxVec3(0.0f);
			}
			if(b.artic)
			{
				Cm::SpatialVectorF* mv =
					b.artic->getArticulationData().getMotionVelocities();
				if(mv)
					mv[b.linkIndex] = Cm::SpatialVectorF(
						PxVec3(0.0f), PxVec3(0.0f));
			}
			continue;
		}
		if(b.rigid)
			b.rigid->mLastTransform = b.core->body2World;
		b.core->body2World.p = b.p;
		b.core->body2World.q = b.q;
		b.core->linearVelocity = b.outputLinVel;
		b.core->angularVelocity = b.outputAngVel;
		avbdLockCoreVel(*b.core);
		if(b.artic)
			avbdSyncArticMotionVel(b);
	}

	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		AvbdSolverBody& b = mBodies[i];
		if(!b.core)
			continue;
		if(b.artic && b.linkIndex > 0)
		{
			ArticulationData& artData = b.artic->getArticulationData();
			const ArticulationLink& link = artData.getLink(b.linkIndex);
			if(link.inboundJoint && link.parent != DY_ARTICULATION_LINK_NONE)
			{
				ArticulationJointCoreData& jd = artData.getJointData(b.linkIndex);
				const PxU32 parentIdx = findBody(artData.getLink(link.parent).bodyCore);
				if(jd.nbDof == 1 && jd.jointOffset != PX_MAX_U32 &&
					artData.getJointPositions() && parentIdx < mBodies.size())
				{
					PxVec3 jLinA, jAngA, jLinB, jAngB;
					PxReal C = 0.0f;
					const PxU32 axis = link.inboundJoint->dofIds[0];
					if(axis < PxArticulationAxis::eCOUNT &&
						avbdArticAxis(mBodies[parentIdx], b, *link.inboundJoint, axis,
							jLinA, jAngA, jLinB, jAngB, C))
					{
						artData.getJointPositions()[jd.jointOffset] = C;
						PxReal* jv = artData.getJointVelocities();
						if(jv && mBodies[parentIdx].core)
						{
							const PxVec3 vA = mBodies[parentIdx].core->linearVelocity;
							const PxVec3 wA = mBodies[parentIdx].core->angularVelocity;
							const PxVec3 lin = b.core->linearVelocity;
							const PxVec3 ang = b.core->angularVelocity;
							jv[jd.jointOffset] = jLinA.dot(vA) + jAngA.dot(wA) +
								jLinB.dot(lin) + jAngB.dot(ang);
						}
						link.inboundJoint->jointPos[axis] = C;
					}
				}
			}
		}
		else if(b.rigid && b.invMass > 0.0f && !b.isKinematic)
		{
			const Cm::SpatialVector motion(b.core->linearVelocity,
				b.core->angularVelocity);
			sleepCheck(b.rigid, dt, mEnableStabilization, motion, b.hasStaticTouch);
		}
	}
}
