#include <string>
// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/DyAvbdBoundedResponse.h"
#include "avbd/DyAvbdDynamics.h"
#include "DyAvbdRigidLoop.h"
#include "DyAvbdPointResponse.h"
#include "DyAvbdNonlinearGroup.h"
#include "avbd/DyAvbdColoring.h"
#include "CmTask.h"
#include "task/PxCpuDispatcher.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "avbd/DyAvbdVbd.h"
#include "avbd/DyAvbdInternal.h"
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd/DyAvbdCoupledDomain.h"

#include "common/PxProfileZone.h"
#include "CmSpatialVector.h"
#include "DyBodyCoreIntegrator.h"
#include "DyConstraint.h"
#include "DyConstraintPrep.h"
#include "DyFeatherstoneArticulation.h"
#include "DyArticulationJointCore.h"
#include "DyIslandManager.h"
#include "DySleep.h"
#include "PxcNpWorkUnit.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsDeformableContact.h"
#include "PxsIslandSim.h"
#include "PxsRigidBody.h"
#include "PxsSimpleIslandManager.h"
#include "PxConstraint.h"
#include "PxConstraintDesc.h"
#include "PxContact.h"
#include "foundation/PxMath.h"
#include "foundation/PxTime.h"
#include "foundation/PxThread.h"
#include "avbd/DyAvbdCpuBatch.h"
#include "avbd/DyAvbdLocalWork.h"
#include "avbd/DyAvbdColorTasks.h"
#include "foundation/PxBitUtils.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMemory.h"
#include "foundation/PxArray.h"
#include "foundation/PxSort.h"
#include "foundation/PxHashMap.h"
#include "DyDeformableVolume.h"
#include "DyDeformableSurface.h"
#include "geometry/PxTetrahedronMesh.h"
#include "geometry/PxTriangleMesh.h"
#include "mesh/GuTetrahedronMesh.h"
#include "GuIntersectionTriangleBox.h"
#include "PxActor.h"

using namespace physx;
using namespace Dy;



namespace physx
{
namespace Dy
{
namespace {
// Execution bounds only; every queued task stays inside one color/phase.
enum { LocalQueueDepth=4,LocalTaskLimit=AvbdCpuBatch::TaskLimit*LocalQueueDepth,
    LocalDualBatchLimit=256 };
}
// A continuation fan-in, never queued as an empty dispatcher job. The last
// worker advances from release(); when the caller finishes last it keeps its
// existing loop instead, so synchronous dispatch cannot grow the call stack.
class AvbdAdvanceTask : public PxLightCpuTask, public PxUserAllocated
{
	AvbdDynamicsContext &mContext;

  public:
	AvbdAdvanceTask(AvbdDynamicsContext &context, PxU64 id) : mContext(context) { mContextID = id; }
	void run() PX_OVERRIDE {}
	const char *getName() const PX_OVERRIDE { return "AVBD.colorComplete"; }
	bool completeInline() { return PxAtomicDecrement(&mRefCount) == 0; }
    void reserveWave(PxTaskManager& tm,PxU32 count)
    {
        // The caller owns one lane. Publish all dependencies before the
        // first submission, including on a synchronous user dispatcher.
        PX_ASSERT(mRefCount==0 && count>1);
        mTm=&tm;mCont=NULL;mRefCount=PxI32(count);
    }
	void removeReference() PX_OVERRIDE
	{
		if (completeInline())
		{
			PX_SIMD_GUARD;
			mContext.advancePrimal();
		}
	}
};
class AvbdReadyTask : public Cm::Task, public PxUserAllocated
{
    AvbdDynamicsContext& context;
    PxU32 root=0,first=0,end=0;
public:
    AvbdReadyTask(AvbdDynamicsContext& ctx,PxU64 id):Cm::Task(id),context(ctx){}
    void submit(AvbdAdvanceTask& join,PxU32 r,PxU32 a,PxU32 b)
    {
        root=r;first=a;end=b;mTm=join.getTaskManager();mCont=&join;
        mTm->getCpuDispatcher()->submitTask(*this);
    }
    void runInternal() PX_OVERRIDE
    {if(root==PX_MAX_U32)context.runReadySeed(first,end,true);else context.runReadySiblings(root,first);}
    const char* getName() const PX_OVERRIDE {return "AVBD.sparseBranch";}
};
class AvbdBlockSolveTask : public Cm::Task, public PxUserAllocated
{
	AvbdDynamicsContext &mContext;

  public:
	AvbdBlockSolveTask(AvbdDynamicsContext &context, PxU64 id) : Cm::Task(id), mContext(context) {}
	PxU32 begin=0,end=0,slot=0;
    bool bounded=false,local=false;
    PxU32 localStage=0;
    void runInternal() PX_OVERRIDE
    {if(local)mContext.solveLocalBatch(begin,end,localStage,slot);
     else if(bounded)mContext.solveBoundedBatch(begin,end,slot);else mContext.solvePrimalBatch();}
	const char *getName() const PX_OVERRIDE { return "AVBD.colorBlocks"; }
};
// A stable dispatcher lane addresses the immutable same-color tile program.
// The caller and workers expose up to three more ready tiles each. The same
// fan-in counts every tile before the first submission, including synchronous dispatch.
class AvbdColorLane : public Cm::Task, public PxUserAllocated
{
    AvbdDynamicsContext& mContext;
    const PxU32 mSlot;
public:
    AvbdColorLane(AvbdDynamicsContext& context,PxU64 id,PxU32 slot):Cm::Task(id),mContext(context),mSlot(slot) {}
    void submit(AvbdAdvanceTask& join)
    {
        mTm=join.getTaskManager();mCont=&join;mTm->getCpuDispatcher()->submitTask(*this);
    }
    void runInternal() PX_OVERRIDE { mContext.runColorLane(mSlot); }
    const char* getName() const PX_OVERRIDE { return "AVBD.colorBlocks"; }
};
} // namespace Dy
} // namespace physx


PX_COMPILE_TIME_ASSERT(AVBD_MAX_JOINT_ROWS == MAX_CONSTRAINT_ROWS);

AvbdDynamicsContext::AvbdDynamicsContext(PxcNpMemBlockPool* memBlockPool,
	Cm::FlushPool& taskPool, PxvSimStats& simStats,
	Cm::VirtualAllocatorCallback& allocator, PxsMaterialManager* materialManager,
	IG::SimpleIslandManager& islandManager, PxU64 contextID,
	PxReal maxBiasCoefficient, PxReal lengthScale, PxSceneFlags sceneFlags)
	: DynamicsContextBase(memBlockPool, taskPool, simStats, allocator,
		  materialManager, islandManager, contextID, maxBiasCoefficient,
		  lengthScale, sceneFlags)
	, mIslandSoftLiveStamp(0)
	, mIslandSurfaceLiveStamp(0)
	, mContactRowCount(0)
	, mNpContactRowCount(0)
	, mIterations(AVBD_ITERS_CONTACT)
	, mLengthScale(lengthScale)
	, mGravity(0.0f)
	, mKernels(KernelTable::makeAvbdCpu())
{
	mOgc.contactRadius = 0.04f * lengthScale;
	mOgc.contactStiffness = 1.0e5f;
	mOgc.safetyRelax = 0.45f;
	mOgc.tau = -1.0f;
}

AvbdDynamicsContext::~AvbdDynamicsContext()
{
    for(auto* task:mReadyTasks)PX_DELETE(task);
    for(auto* loop:mLoopScratch)PX_DELETE(loop);
    PX_DELETE(mLocalWork);
    for(auto* work:mWidePool)PX_DELETE(work);
    PX_DELETE(mColorTasks);
    PX_DELETE(mWideNomination);
    clearCommonPublication();
	clearIslandSoftCache();
	clearIslandSurfaceCache();
	for(PxU32 i=0;i<mBlockTasks.size();++i) PX_DELETE(mBlockTasks[i]);
	for(auto* lane:mColorLanes)PX_DELETE(lane);
	PX_DELETE(mColorContinuation);
}

void AvbdDynamicsContext::destroy()
{
	clearIslandSoftCache();
	clearIslandSurfaceCache();
	this->~AvbdDynamicsContext();
	PX_FREE_THIS;
}

namespace
{
bool avbdReadTetIndices(const PxTetrahedronMesh& mesh, PxU32 tet, PxU32 out[4])
{
	if(tet >= mesh.getNbTetrahedrons())
		return false;
	const bool idx16 = mesh.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES;
	if(idx16)
	{
		const PxU16* idx = reinterpret_cast<const PxU16*>(mesh.getTetrahedrons());
		out[0] = idx[tet * 4 + 0];
		out[1] = idx[tet * 4 + 1];
		out[2] = idx[tet * 4 + 2];
		out[3] = idx[tet * 4 + 3];
	}
	else
	{
		const PxU32* idx = reinterpret_cast<const PxU32*>(mesh.getTetrahedrons());
		out[0] = idx[tet * 4 + 0];
		out[1] = idx[tet * 4 + 1];
		out[2] = idx[tet * 4 + 2];
		out[3] = idx[tet * 4 + 3];
	}
	return true;
}

bool avbdReadTriIndices(const PxTriangleMesh& mesh, PxU32 tri, PxU32 out[3])
{
	if(tri >= mesh.getNbTriangles())
		return false;
	if(mesh.getTriangleMeshFlags() & PxTriangleMeshFlag::e16_BIT_INDICES)
	{
		const PxU16* idx = reinterpret_cast<const PxU16*>(mesh.getTriangles());
		out[0] = idx[tri * 3 + 0];
		out[1] = idx[tri * 3 + 1];
		out[2] = idx[tri * 3 + 2];
	}
	else
	{
		const PxU32* idx = reinterpret_cast<const PxU32*>(mesh.getTriangles());
		out[0] = idx[tri * 3 + 0];
		out[1] = idx[tri * 3 + 1];
		out[2] = idx[tri * 3 + 2];
	}
	return true;
}
} // namespace

void AvbdDynamicsContext::clearIslandSoftCache()
{
	for(PxU32 i = 0; i < mIslandSoftCache.size(); ++i)
		avbdFreeIslandSoftBlocks(mIslandSoftCache[i]);
	mIslandSoftCache.forceSize_Unsafe(0);
}

AvbdIslandSoftCache* AvbdDynamicsContext::getIslandSoftCache(DeformableVolume* volume)
{
	for(PxU32 i = 0; i < mIslandSoftCache.size(); ++i)
	{
		if(mIslandSoftCache[i].volume == volume)
			return &mIslandSoftCache[i];
	}
	return NULL;
}

bool AvbdDynamicsContext::anyCoupledDeformableVolume() const
{
	for(PxU32 i = 0; i < mIslandSoftCache.size(); ++i)
	{
		if(mIslandSoftCache[i].coupled &&
			mIslandSoftCache[i].liveStamp == mIslandSoftLiveStamp)
			return true;
	}
	return false;
}

bool AvbdDynamicsContext::anyLiveDeformableSurface() const
{
	for(PxU32 i = 0; i < mIslandSurfaceCache.size(); ++i)
		if(mIslandSurfaceCache[i].liveStamp == mIslandSurfaceLiveStamp)
			return true;
	return false;
}

namespace
{
// The simulation tetrahedra are the AVBD unknowns; collision vertices are an
// embedded view of those unknowns.  This is the Host equivalent of
// PxDeformableVolumeExt::updateEmbeddedCollisionMesh and uses the same cooked
// tet remap/barycentric contract.
bool avbdWriteEmbeddedCollisionPositions(DeformableVolume& volume,
	DeformableVolumeCore& core)
{
	PxTetrahedronMesh* collisionMesh = volume.getCollisionMesh();
	PxTetrahedronMesh* simulationMesh = volume.getSimulationMesh();
	if(!core.positionInvMass || !core.simPositionInvMass || !collisionMesh || !simulationMesh)
		return false;

	const PxU32 nbCollisionVertices = collisionMesh->getNbVertices();
	const PxU32 nbSimulationVertices = simulationMesh->getNbVertices();
	// Equal vertex counts do not imply an identity embedding. Use the cooked
	// mapping for publication, as for contact supports and the SDK extension.

	Gu::DeformableVolumeAuxData* aux =
		static_cast<Gu::DeformableVolumeAuxData*>(volume.getAuxData());
	if(!aux || !aux->mVertsRemapInGridModel || !aux->mVertsBarycentricInGridModel)
		return false;

	const void* tetrahedra = simulationMesh->getTetrahedrons();
	const PxU32 nbTetrahedra = simulationMesh->getNbTetrahedrons();
	if(!tetrahedra || nbTetrahedra == 0)
		return false;
	const bool sixteenBit = simulationMesh->getTetrahedronMeshFlags().isSet(
		PxTetrahedronMeshFlag::e16_BIT_INDICES);
	const PxU16* tets16 = static_cast<const PxU16*>(tetrahedra);
	const PxU32* tets32 = static_cast<const PxU32*>(tetrahedra);

	for(PxU32 i = 0; i < nbCollisionVertices; ++i)
	{
		const PxU32 tet = aux->mVertsRemapInGridModel[i];
		if(tet >= nbTetrahedra)
			return false;
		PxU32 ids[4];
		for(PxU32 j = 0; j < 4; ++j)
		{
			ids[j] = sixteenBit ? PxU32(tets16[4 * tet + j]) : tets32[4 * tet + j];
			if(ids[j] >= nbSimulationVertices)
				return false;
		}
		const PxReal* w = aux->mVertsBarycentricInGridModel + 4 * i;
		PxVec4 embedded = core.simPositionInvMass[ids[0]] * w[0] +
			core.simPositionInvMass[ids[1]] * w[1] +
			core.simPositionInvMass[ids[2]] * w[2] +
			core.simPositionInvMass[ids[3]] * w[3];
		embedded.w = 1.0f;
		core.positionInvMass[i] = embedded;
	}
	return true;
}

} // namespace

void AvbdDynamicsContext::publishIslandSoftCache(AvbdIslandSoftCache& cache)
{
	DeformableVolume* vol = cache.volume;
	if(!vol || !cache.particles)
		return;
	DeformableVolumeCore& core = vol->getCore();
	if(!core.simPositionInvMass || !core.simVelocity)
		return;
	for(PxU32 i = 0; i < cache.nbParticles; ++i)
	{
		AvbdSoftParticle& p = cache.particles[i];
		core.simPositionInvMass[i] = PxVec4(p.p, p.invMass);
		core.simVelocity[i] = PxVec4(p.vel, p.invMass);
	}
	avbdWriteEmbeddedCollisionPositions(*vol, core);
	core.dirtyFlags = PxDeformableVolumeDataFlags(PxDeformableVolumeDataFlag::ePOSITION_INVMASS);
	core.dirtyFlags |= PxDeformableVolumeDataFlag::eSIM_POSITION_INVMASS;
	core.dirtyFlags |= PxDeformableVolumeDataFlag::eSIM_VELOCITY;
}

void AvbdDynamicsContext::pruneIslandSoftCache()
{
	PxU32 w = 0;
	for(PxU32 i = 0; i < mIslandSoftCache.size(); ++i)
	{
		if(mIslandSoftCache[i].liveStamp == mIslandSoftLiveStamp)
		{
			if(w != i)
				mIslandSoftCache[w] = mIslandSoftCache[i];
			++w;
		}
		else
			avbdFreeIslandSoftBlocks(mIslandSoftCache[i]);
	}
	mIslandSoftCache.forceSize_Unsafe(w);
}

// Read-only bounded CSR gather. Reciprocal nominations select disjoint
// two-node patches independently of the worker visitation order.
PxVec4 AvbdDynamicsContext::sampleNodalLoad(PxU32 b,PxU32 first,PxU32 end,bool canPair)
{
    PxReal best=0,weightSum=0;PxVec3 load(0);
    for(PxU32 k=first;k<end;++k)
    {
        const auto row=mRows[mRowProgram.blockRows[k]];
        if(row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT)continue;
        const auto& c=mNodalContacts[row.header().geometryIndex-1];
        if(c.loadNodes[0]==PX_MAX_U32 || c.loadNodes[1]==PX_MAX_U32 || row.scalar().lambda>=0)continue;
        for(PxU32 s=row.header().supportStart;s<row.header().supportStart+row.header().supportCount;++s)
        {
            const auto& support=mRowSupports[s];if(support.block!=b || support.weight==0)continue;
            load+=row.scalar().n*(-row.scalar().lambda*support.weight);
            weightSum+=PxAbs(support.weight);
            const PxU32 side=avbdNodalPairSide(c,b),peer=c.loadNodes[1-side];
            if(!canPair || end-first+(mRowProgram.blockOffsets[peer+1]-mRowProgram.blockOffsets[peer])>256)break;
            const PxReal score=PxAbs(support.weight)*c.loadWeights[1-side]*row.scalar().rho;
            if(score>best){best=score;mNodalPartners[b]=peer;}
            break;
        }
    }
    return PxVec4(load,weightSum);
}

void AvbdDynamicsContext::seedNodalImpact(PxU32 index)
{
    const auto& law=mRows[mNativeMaterialEnd-mNodalContacts.size()+index].scalar();
    const PxReal gap=avbdOGCLoadedGap(-law.lambda,law.contactRadius,law.contactStiffness);
    const auto& contact=mNodalContacts[index];
    const auto nominate=[&](PxU32 block,PxReal displacement)
    {
        const PxU32 direction=(index<<1)|PxU32(displacement<0);
        PxAtomicMax(&mNodalImpactVotes[block],avbdImpactVote(PxAbs(displacement),direction));
    };
    if(contact.geometry->deformable[0] && contact.geometry->deformable[1])
    {
        avbdNominatePairImpact(contact,gap,
            [&](PxU32 block)->const AvbdSoftParticle& {return *mBlocks[block].particle;},nominate);
        return;
    }
    const PxU32 side=contact.geometry->deformable[0]?0:1;
    avbdNominateFixedImpact(contact,side,gap,
        [&](PxU32 block){const auto& p=*mBlocks[block].particle;return p.pHat-p.pStart;},
        [&](PxU32 block,PxReal displacement)
        {
            if(mBlocks[block].particle->invMass>0)nominate(block,side?-displacement:displacement);
        });
}

PxU32 AvbdDynamicsContext::nominationJobs() const
{
    return mBlocks.size()+(mWideNomination?mWideNomination->tree.leaves.size():0);
}

PxU32 AvbdDynamicsContext::nominationRows(PxU32 job) const
{
    if(job<mBlocks.size())
    {
        const PxU32 rows=mRowProgram.blockOffsets[job+1]-mRowProgram.blockOffsets[job];
        return rows>AvbdLocalWork::RowLimit?1:PxMax(1u,rows);
    }
    const auto& tree=mWideNomination->tree;
    const auto& node=tree.nodes[tree.leaves[job-mBlocks.size()]];
    return node.end-node.begin;
}

void AvbdDynamicsContext::nominateNodalPartner(PxU32 job)
{
    if(job<mBlocks.size())
    {
        const PxU32 lo=mRowProgram.blockOffsets[job],hi=mRowProgram.blockOffsets[job+1];
        if(hi-lo<=AvbdLocalWork::RowLimit)mNodalIncidentLoad[job]=sampleNodalLoad(job,lo,hi,true);
        return;
    }
    auto& wide=*mWideNomination;const auto& tree=wide.tree;
    PxU32 id=tree.leaves[job-mBlocks.size()];
    const auto& node=tree.nodes[id];const PxU32 block=tree.states[node.state].body;
    if(wide.reduce(id,sampleNodalLoad(block,node.begin,node.end,false)))
        mNodalIncidentLoad[block]=wide.load[tree.states[node.state].root];
}

void AvbdDynamicsContext::conditionNodalLoad(PxU32 index)
{
    const auto& c=mNodalContacts[index];
    auto row=mRows[mNativeMaterialEnd-mNodalContacts.size()+index];
    if(c.loadNodes[0]==PX_MAX_U32 || c.loadNodes[1]==PX_MAX_U32 || row.scalar().lambda>=0)return;
    const auto& head=row.header();const PxVec3 n=row.scalar().n;
    PxReal gap=row.scalar().C0,response=0,collective=0;
    for(PxU32 k=head.supportStart;k<head.supportStart+head.supportCount;++k)
    {
        const auto& support=mRowSupports[k];const auto& p=*mBlocks[support.block].particle;
        gap+=support.weight*n.dot(p.pHat-p.pStart);
        const auto& load=mNodalIncidentLoad[support.block];
        response+=PxAbs(support.weight)*load.w*p.invMass;
        collective+=support.weight*p.invMass*n.dot(load.getXYZ());
    }
    // Freeze the incident loads and take one scalar resolvent of their
    // common predicted gap using the absolute row-sum Delassus majorant.
    // An isolated or already equilibrated row is unchanged.
    if(!(response>0))return;
    const PxReal stiffness=1.f/(mDt*mDt*response);
    row.scalar().lambda=-avbdOGCNormalResolvent(-row.scalar().lambda-stiffness*(gap+mDt*mDt*collective),stiffness,
        row.scalar().contactRadius,row.scalar().contactStiffness);
    if(row.scalar().contactRadius>0)
        row.scalar().rho=avbdOGCLoadedNormalMetric(mDt*mDt*response,-row.scalar().lambda,
            row.scalar().contactRadius,row.scalar().contactStiffness);
}

PxU32 AvbdDynamicsContext::nodalPartner(PxU32 block) const
{
    if(block>=mNodalPartners.size())return PX_MAX_U32;
    const PxU32 peer=mNodalPartners[block];
    return peer!=PX_MAX_U32 && mNodalPartners[peer]==block?peer:PX_MAX_U32;
}

void AvbdDynamicsContext::bakeMembranePackets(AvbdIslandSurfaceCache& surface)
{
    auto& bake=*surface.membrane;
    const PxU32 width=PxClamp(surface.nbTriangles/32,1u,32u);
    PxArray<PxU32> owner(surface.nbTriangles);
    for(PxU32 first=0;first<surface.nbTriangles;)
    {
        AvbdMembranePacket packet;packet.first=first;
        PxU32 nodes[32],used[32]={};
        while(first+packet.rows<surface.nbTriangles && packet.rows<width)
        {
            const auto& face=surface.triangles[first+packet.rows];
            const PxU32 v[3]={face.p0,face.p1,face.p2};
            PxU32 slot[3],n=packet.nodes,conflicts=0;
            for(PxU32 j=0;j<3;++j)
            {
                PxU32 k=0;while(k<packet.nodes && nodes[k]!=v[j])++k;
                if(k<packet.nodes)conflicts|=used[k];else k=n++;
                slot[j]=k;
            }
            if(n>32)break;
            const PxU32 color=PxLowestSetBit(~conflicts);
            for(PxU32 j=0;j<3;++j){nodes[slot[j]]=v[j];used[slot[j]]|=1u<<color;}
            packet.nodes=PxU8(n);packet.passes=PxMax(packet.passes,PxU8(color+1));
            packet.masks[color]|=1u<<packet.rows;owner[first+packet.rows]=bake.packets.size();++packet.rows;
        }
        for(PxU32 c=0;c<packet.passes;++c)
            packet.parallelFaces=PxMax(packet.parallelFaces,PxU8(PxBitCount(packet.masks[c])));
        bake.packets.pushBack(packet);first+=packet.rows;
    }
    // Each vertex contributes one conflict scope, even when several faces of
    // a packet touch it. Color packets with the most shared supports first.
    const PxU32 count=bake.packets.size();
    AvbdBlockColoring graph;PxArray<PxU32> rank(count,0),order(count),incident;
    PxArray<PxU8> active(count,1);
    for(PxU32 i=0;i<surface.nbParticles;++i)
    {
        const PxU32 b=surface.blockStart+i;incident.clear();
        for(PxU32 k=mRowProgram.blockOffsets[b];k<mRowProgram.blockOffsets[b+1];++k)
        {
            const PxU32 r=mRowProgram.blockRows[k]-bake.firstRow;
            if(r<surface.nbTriangles && (incident.empty() || incident.back()!=owner[r]))incident.pushBack(owner[r]);
        }
        for(PxU32 id:incident)rank[id]+=incident.size()-1;
        graph.addScope(incident.begin(),incident.size(),active);
    }
    for(PxU32 i=0;i<count;++i)order[i]=i;
    std::sort(order.begin(),order.end(),[&](PxU32 a,PxU32 b){return rank[a]!=rank[b]?rank[a]>rank[b]:a<b;});
    graph.build(order,count,bake.offsets,bake.faces,count);
}

bool AvbdDynamicsContext::compileIncidencePrimal()
{
    if(!mStep.localBudget || mNodalShellMaterials.empty())return false;
    for(const auto& m:mNodalShellMaterials)if(!m.previous)return false;
    for(const auto row:mRows)
        if((row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_SHELL || row.header().supportCount!=3) &&
            row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT)return false;
    for(const auto& b:mBodies)if(b.core && b.core->lockFlags)return false;
    mStep.incidencePrimal=true;
    mIncidenceDisplacement.resize(mRowSupports.size(),PxVec3(0));
    mIncidenceRotation.resize(mNodalContacts.size(),PxVec3(0));
    PxArray<PxU32> colors(mRows.size(),PX_MAX_U32);
    const PxU32 first=mNativeMaterialEnd-mNodalContacts.size();
    mIncidenceFacePackets.clear();mIncidenceFacePackets.resize(first,NULL);
    PxU32 count=0;
    for(auto& surface:mIslandSurfaceCache)if(surface.liveStamp==mIslandSurfaceLiveStamp && surface.membrane)
    {
        auto& bake=*surface.membrane;
        if(bake.offsets.empty())bakeMembranePackets(surface);
        count=PxMax(count,bake.offsets.size()-1);
        for(PxU32 c=0;c+1<bake.offsets.size();++c)
            for(PxU32 i=bake.offsets[c];i<bake.offsets[c+1];++i)
            {
                const auto& packet=bake.packets[bake.faces[i]];const PxU32 r=bake.firstRow+packet.first;
                mIncidenceFacePackets[r]=&packet;
                for(PxU32 j=0;j<packet.rows;++j)colors[r+j]=c;
                if(mStep.countNodalSolves)
                {
                    mStep.membranePacketFaces=PxMax(mStep.membranePacketFaces,PxU32(packet.rows));
                    mStep.membranePacketColors=PxMax(mStep.membranePacketColors,PxU32(packet.passes));
                    mStep.membraneParallelFaces=PxMax(mStep.membraneParallelFaces,PxU32(packet.parallelFaces));
                }
            }
    }
    mIncidencePackets.clear();mIncidencePackets.resize(mNodalContacts.size(),{0,0});
    PxArray<PxU32> used(count,PX_MAX_U32);
    PxArray<PxU32> nextRigidColor(mBodies.size(),0);
    for(PxU32 r=first;r<mNativeMaterialEnd;)
    {
        const auto& head=mRows[r].header();PxU32 rows=1,entries=head.supportCount;
        const PxU32 rigid=PxMin(head.bodyA,head.bodyB);
        // Fixed tiles keep candidate construction independent across eight-row
        // segments. A packet never spans two rigid spatial metrics.
        const PxU32 tileEnd=PxMin(mNativeMaterialEnd,first+((r-first)/8+1)*8);
        while(r+rows<tileEnd && rigid<mBodies.size())
        {
            const auto& h=mRows[r+rows].header();
            if(PxMin(h.bodyA,h.bodyB)!=rigid || entries+h.supportCount>32)break;
            entries+=h.supportCount;++rows;
        }
        for(PxU32 j=0;j<rows;++j)mRowProgram.forEachBlock(r+j,[&](PxU32 b)
        {
            if(b==rigid)return;
            for(PxU32 k=mRowProgram.blockOffsets[b];k<mRowProgram.blockOffsets[b+1];++k)
            {
                const auto c=colors[mRowProgram.blockRows[k]];
                if(c!=PX_MAX_U32)used[c]=r;
            }
        });
        PxU32 c=rigid<mBodies.size()?nextRigidColor[rigid]:0;
        while(c<count && used[c]==r)++c;
        if(c==count){used.pushBack(r);++count;}
        if(rigid<mBodies.size())nextRigidColor[rigid]=c+1;
        mIncidencePackets[r-first]={PxU8(rows),PxU8(entries)};
        for(PxU32 j=0;j<rows;++j)colors[r+j]=c;
        r+=rows;
    }
    const auto workItem=[&](PxU32 r){return r<first?mIncidenceFacePackets[r]!=NULL:mIncidencePackets[r-first].rows!=0;};
    mColorOffsets.clear();mColorOffsets.resize(count+1,0);
    for(PxU32 r=0;r<mRows.size();++r)if(workItem(r))++mColorOffsets[colors[r]+1];
    for(PxU32 i=0;i<count;++i)mColorOffsets[i+1]+=mColorOffsets[i];
    PxArray<PxU32> cursor(mColorOffsets);
    mColorBlocks.resize(mColorOffsets.back());
    for(PxU32 r=0;r<mRows.size();++r)if(workItem(r))mColorBlocks[cursor[colors[r]]++]=mBlocks.size()+r;
    mIncidenceTaskShape.resize(mColorBlocks.size());
    for(PxU32 i=0;i<mColorBlocks.size();++i)
    {
        const PxU32 b=mColorBlocks[i],blocks=localItemBlocks(b);
        mIncidenceTaskShape[i]=(blocks<<16)|localItemRows(b,blocks);
    }
    ++mStep.localColorBuilds;return true;
}

PxU32 AvbdDynamicsContext::incidenceHistoryEnd(bool includeScatter) const
{
    const PxU32 leaves=mWideNomination?mWideNomination->tree.leaves.size():0;
    return mBlocks.size()+mNodalContacts.size()+(includeScatter?2u:1u)*leaves;
}

PxU32 AvbdDynamicsContext::incidenceHistoryRows(PxU32 job) const
{
    if(job<mBlocks.size())return mBlocks[job].particle?nominationRows(job):1;
    const PxU32 base=mBlocks.size()+mNodalContacts.size();if(job<base)return 1;
    const auto& tree=mWideNomination->tree;
    const auto& node=tree.nodes[tree.leaves[(job-base)%tree.leaves.size()]];
    return node.end-node.begin;
}

void AvbdDynamicsContext::prepareIncidenceHistory(PxU32 block)
{
    const PxU32 base=mBlocks.size()+mNodalContacts.size();
    if(block>=base)
    {
        auto& wide=*mWideNomination;const auto& tree=wide.tree;
        const PxU32 i=block-base,id=tree.leaves[i%tree.leaves.size()];
        const auto& node=tree.nodes[id];const auto& state=tree.states[node.state];
        if(i<tree.leaves.size())wide.reduce(id,PxVec4(gatherIncidenceHistory(state.body,node.begin,node.end),0));
        else scatterIncidenceHistory(state.body,node.begin,node.end,wide.load[state.root].getXYZ());
        return;
    }
    if(block>=mBlocks.size())
    {
        const auto& h=mRows[mNativeMaterialEnd-mNodalContacts.size()+block-mBlocks.size()].header();
        const PxU32 rigid=PxMin(h.bodyA,h.bodyB);
        if(rigid>=mBodies.size() || mBodies[rigid].invMass<=0 || mBodies[rigid].isKinematic)return;
        const auto& b=mBodies[rigid];
        const PxReal degree=PxReal(mRowProgram.blockOffsets[rigid+1]-mRowProgram.blockOffsets[rigid]);
        for(PxU32 s=h.supportStart;s<h.supportStart+h.supportCount;++s)if(mRowSupports[s].block==rigid)
        {
            mIncidenceDisplacement[s]=(b.p-b.pHat)/degree;
            mIncidenceRotation[h.geometryIndex-1]=avbdQuatLogWorld(b.qHat,b.q)/degree;
        }
        return;
    }
    const PxU32 first=mRowProgram.blockOffsets[block],end=mRowProgram.blockOffsets[block+1];
    if(!mBlocks[block].particle || end==first || end-first>AvbdLocalWork::RowLimit)return;
    scatterIncidenceHistory(block,first,end,gatherIncidenceHistory(block,first,end));
}

PxVec3 AvbdDynamicsContext::gatherIncidenceHistory(PxU32 block,PxU32 first,PxU32 end)
{
    // Change coordinates for the new epoch without moving a node. Each node
    // owns its incidence slots; no atomics, component momentum or extra sweep.
    PxVec3 sum(0);
    for(PxU32 k=first;k<end;++k)
    {
        const auto& h=mRows[mRowProgram.blockRows[k]].header();
        if(h.kernelId!=eAVBD_ROW_KERNEL_NODAL_SHELL)continue;
        const auto& m=mNodalShellMaterials[h.geometryIndex-1];if(!m.warm)continue;
        for(PxU32 i=0;i<3;++i)if(mRowSupports[h.supportStart+i].block==block)
        {mIncidenceDisplacement[h.supportStart+i]=m.previous[i];sum+=m.previous[i];}
    }
    return sum;
}

void AvbdDynamicsContext::scatterIncidenceHistory(PxU32 block,PxU32 first,PxU32 end,const PxVec3& sum)
{
    const auto& p=*mBlocks[block].particle;
    const PxVec3 share=(sum-(p.p-p.pHat))/PxReal(mRowProgram.blockOffsets[block+1]-mRowProgram.blockOffsets[block]);
    for(PxU32 k=first;k<end;++k)
    {
        const auto& h=mRows[mRowProgram.blockRows[k]].header();
        for(PxU32 s=h.supportStart;s<h.supportStart+h.supportCount;++s)
            if(mRowSupports[s].block==block)mIncidenceDisplacement[s]-=share;
    }
}

void AvbdDynamicsContext::solveIncidencePrimal(PxU32 index)
{
    const auto& row=mRows[index].header();
    if(row.kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT){solveIncidenceContact(index);return;}
    const auto& packet=*mIncidenceFacePackets[index];
    for(PxU32 c=0;c<packet.passes;++c)
        for(PxU32 mask=packet.masks[c];mask;mask&=mask-1)solveIncidenceFace(index+PxLowestSetBit(mask));
}

void AvbdDynamicsContext::solveIncidenceFace(PxU32 index)
{
    const auto& row=mRows[index].header();
    const auto& material=mNodalShellMaterials[row.geometryIndex-1];
    using V=AvbdNodalVec3;using R=AvbdNodalReal;
    V x[3],previous[3],weighted[3],force[3],direction[3];R mobility[3],majorant[3];
    const R dt2=R(mDt)*mDt;
    for(PxU32 i=0;i<3;++i)
    {
        const auto slot=row.supportStart+i;
        const auto& p=*mBlocks[mRowSupports[slot].block].particle;
        const auto& displacement=mIncidenceDisplacement[slot];
        x[i]=V(p.p.x,p.p.y,p.p.z);previous[i]=V(displacement.x,displacement.y,displacement.z);mobility[i]=dt2*p.invMass;
        const PxU32 b=mRowSupports[slot].block;
        const R degree=material.warm?R(1):R(mRowProgram.blockOffsets[b+1]-mRowProgram.blockOffsets[b]);
        majorant[i]=degree*mobility[i];weighted[i]=previous[i]*degree;
    }
    AvbdMembranePrimalArgs args={*material.rest,x,majorant,weighted,material.mu,material.lambda,force};
    mKernels.launch(AVBD_MEMBRANE_PRIMAL,&args);
    if(mStep.countNodalSolves)PxAtomicIncrement(&mStep.membranePrimalSolves);
    if(!args.solved)
    {PxAtomicExchange(&mLocalDomainFailed,1);return;}
    for(PxU32 i=0;i<3;++i)direction[i]=force[i]*mobility[i]-previous[i];
    const auto merit=[&](const V (&y)[3])
    {
        auto sample=avbdSampleNodalShell(*material.rest,y,material.mu,material.lambda,material.bending,PX_MAX_U32);
        if(!sample.valid)return std::numeric_limits<R>::infinity();
        for(PxU32 i=0;i<3;++i)if(mobility[i]>0)
            sample.energy+=(previous[i]+(y[i]-x[i])).magnitudeSquared()/(2*mobility[i]);
        return sample.energy;
    };
    const R base=merit(x);
    for(PxU32 trial=0;trial<16;++trial)
    {
        const R fraction=R(1)/R(1u<<trial);
        V y[3];for(PxU32 i=0;i<3;++i)y[i]=x[i]+direction[i]*fraction;
        const V a=x[1]-x[0],b=x[2]-x[0],normal=a.cross(b);
        const V e=y[1]-y[0],f=y[2]-y[0];
        if(!(normal.dot(e.cross(b)+a.cross(f))>0 && normal.dot(e.cross(f))>0))continue;
        if(!(merit(y)<=base))continue;
        for(PxU32 i=0;i<3;++i)
        {
            const auto slot=row.supportStart+i,bid=mRowSupports[slot].block;
            auto& p=*mBlocks[bid].particle;
            const PxVec3 accepted=PxVec3(PxReal(y[i].x),PxReal(y[i].y),PxReal(y[i].z))-p.p;
            p.p+=accepted;mIncidenceDisplacement[slot]+=accepted;
        }
        return;
    }
}

void AvbdDynamicsContext::solveIncidenceContact(PxU32 first)
{
    // Condense disjoint material inertia into each normal/tangent law. The
    // common rigid translation and rotation are eliminated once for this tile.
    // Each leaf contributes independently to the symmetric 6x6 reduction.
    const PxU32 count=mIncidencePackets[mRows[first].header().geometryIndex-1].rows;
    const PxU32 rigid=PxMin(mRows[first].header().bodyA,mRows[first].header().bodyB);
    const bool dynamic=rigid<mBodies.size() && mBodies[rigid].invMass>0 && !mBodies[rigid].isKinematic;
    const PxReal dt2=mDt*mDt;
    struct Leaf
    {
        PxReal soft=0,gap=0,sign=0;
        PxVec3 slip=PxVec3(0),arm=PxVec3(0),oldDp=PxVec3(0),oldAngle=PxVec3(0);
        PxVec3 angle=PxVec3(0);
        PxReal J[3][6]={};
    } leaf[8];
    struct Endpoint {PxU32 id;PxReal weights;PxVec3 delta;};
    Endpoint endpoint[32];PxU32 ends=0;
    PxU8 which[32];PxVec3 increment[32];
    const PxU32 slot0=mRows[first].header().supportStart;
    PxReal A[6][6]={},rhs[6]={},q[6]={};
    for(PxU32 j=0;j<6;++j)A[j][j]=1;
    const PxReal linear=dynamic?mDt*PxSqrt(mBodies[rigid].invMass):0;
    PxVec3 angular(0);
    if(dynamic)for(PxU32 j=0;j<3;++j)angular[j]=mDt*PxSqrt(mBodies[rigid].invInertia[j]);
    for(PxU32 r=0;r<count;++r)
    {
        const auto& h=mRows[first+r].header();
        for(PxU32 s=h.supportStart;s<h.supportStart+h.supportCount;++s)
        {
            const auto& sup=mRowSupports[s];PxU32 k=0;
            while(k<ends && endpoint[k].id!=sup.block)++k;
            if(k==ends)endpoint[ends++]={sup.block,0,PxVec3(0)};
            endpoint[k].weights+=PxAbs(sup.weight);which[s-slot0]=PxU8(k);
        }
    }
    const auto traction=[&](PxU32 r,const PxVec3& motion,PxReal (&pressure)[3],PxReal (&slope)[3])
    {
        const auto row=mRows[first+r];const auto& law=row.scalar();const auto& l=leaf[r];
        const PxReal series=1/(1+law.rho*l.soft),rho=law.rho*series;
        const auto normal=avbdOGCNormalEnvelope(l.gap+motion.x,law.lambda*series,rho,law.contactRadius,law.contactStiffness);
        pressure[0]=normal.force;slope[0]=normal.hessian;
        pressure[1]=pressure[2]=slope[1]=slope[2]=0;
        if(law.mu<=0 || pressure[0]<=0 || law.rhoT0<=0)return;
        slope[1]=law.rhoT0/(1+law.rhoT0*l.soft);slope[2]=law.rhoT1/(1+law.rhoT1*l.soft);
        pressure[1]=-slope[1]*(law.lambdaT0/law.rhoT0+law.t0.dot(l.slip)+motion.y);
        pressure[2]=-slope[2]*(law.lambdaT1/law.rhoT1+law.t1.dot(l.slip)+motion.z);
        const PxReal scale=avbdProjectCoulomb(-pressure[0],law.mu,pressure[1],pressure[2]);
        slope[1]*=scale;slope[2]*=scale;
    };
    for(PxU32 r=0;r<count;++r)
    {
        auto& l=leaf[r];const auto row=mRows[first+r];const auto& h=row.header();const auto& law=row.scalar();
        PxVec3 previous(0);
        for(PxU32 s=h.supportStart;s<h.supportStart+h.supportCount;++s)
        {
            const auto& sup=mRowSupports[s];const PxReal w=sup.weight;const auto& block=mBlocks[sup.block];
            if(block.particle)
            {
                const PxReal weights=endpoint[which[s-slot0]].weights;
                l.soft+=dt2*PxAbs(w)*weights*block.particle->invMass;
                if(w!=0)previous+=mIncidenceDisplacement[s]*(w<0?-weights:weights);
            }
            else if(dynamic)
            {
                l.sign=w;l.arm=mBodies[rigid].q.rotate(sup.localPoint);l.oldDp=mIncidenceDisplacement[s];
                l.oldAngle=mIncidenceRotation[h.geometryIndex-1];
            }
        }
        l.gap=avbdHeterogeneousRowC(row,mRowSupports,mBlocks,mBodies,contactGeometry())-law.n.dot(previous);
        l.slip=avbdNodalContactSlip(row,mRowSupports,mBlocks,mBodies,contactGeometry())-previous;
        if(!dynamic)continue;
        const PxVec3 axes[]={law.n,law.t0,law.t1};PxReal pressure[3],slope[3];
        traction(r,PxVec3(0),pressure,slope);
        const auto old=mBodies[rigid].qStart.rotateInv(l.oldAngle);
        for(PxU32 i=0;i<3;++i)
        {
            rhs[i]-=l.oldDp[i]/linear;if(angular[i]>0)rhs[i+3]-=old[i]/angular[i];
            const auto lever=mBodies[rigid].qStart.rotateInv(l.arm.cross(axes[i]));
            for(PxU32 j=0;j<3;++j){l.J[i][j]=linear*l.sign*axes[i][j];l.J[i][j+3]=angular[j]*l.sign*lever[j];}
        }
        for(PxU32 a=0;a<3;++a)for(PxU32 i=0;i<6;++i)
        {
            rhs[i]+=l.J[a][i]*pressure[a];
            for(PxU32 j=i;j<6;++j)A[i][j]+=slope[a]*l.J[a][i]*l.J[a][j];
        }
    }
    if(dynamic)
    {
        bool solved=false;AvbdLdlt6Args args={A,rhs,q,&solved};mKernels.launch(AVBD_LDLT6,&args);
        if(!solved){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    }
    PxVec3 angle(0);
    for(PxU32 r=0;r<count;++r)
    {
        auto& l=leaf[r];const auto row=mRows[first+r];const auto& h=row.header();const auto& law=row.scalar();
        PxVec3 motion(0);
        for(PxU32 i=0;dynamic && i<3;++i)for(PxU32 j=0;j<6;++j)motion[i]+=l.J[i][j]*q[j];
        PxReal pressure[3],slope[3];traction(r,motion,pressure,slope);
        const auto force=law.n*pressure[0]+law.t0*pressure[1]+law.t1*pressure[2];
        if(!force.isFinite()){PxAtomicExchange(&mLocalDomainFailed,1);return;}
        for(PxU32 s=h.supportStart;s<h.supportStart+h.supportCount;++s)
        {
            const auto& sup=mRowSupports[s];const auto& block=mBlocks[sup.block];const auto load=force*(dt2*sup.weight);
            const PxReal inv=block.particle?block.particle->invMass:(dynamic?mBodies[rigid].invMass:0);
            increment[s-slot0]=load*inv-mIncidenceDisplacement[s];
            endpoint[which[s-slot0]].delta+=increment[s-slot0];
            if(!block.particle && dynamic){l.angle=avbdWorldInvInertiaMul(mBodies[rigid],l.arm.cross(load))-l.oldAngle;angle+=l.angle;}
        }
    }
    if(mStep.countNodalSolves)PxAtomicAdd(&mStep.contactPrimalSolves,PxI32(count));
    for(PxU32 trial=0;trial<16;++trial)
    {
        const PxReal fraction=1.f/PxReal(1u<<trial);
        PxTransform pose(PxIdentity);
        for(PxU32 i=0;i<ends;++i)if(endpoint[i].id==rigid)
        {pose=PxTransform(mBodies[rigid].p,mBodies[rigid].q);avbdApplyDelta(pose,endpoint[i].delta*fraction,angle*fraction);}
        const AvbdRigidPoseView poses={mBodies,&rigid,&pose,rigid<mBodies.size()?1u:0u};
        const auto node=[&](PxU32 b){for(PxU32 i=0;i<ends;++i)if(endpoint[i].id==b)return mBlocks[b].particle->p+endpoint[i].delta*fraction;
            return mBlocks[b].particle->p;};
        bool admitted=true;
        for(PxU32 r=0;r<count && admitted;++r)
            admitted=avbdNodalContactPath(mNodalContacts[mRows[first+r].header().geometryIndex-1],node,poses);
        if(!admitted)continue;
        for(PxU32 r=0;r<count;++r)
        {
            const auto& h=mRows[first+r].header();mIncidenceRotation[h.geometryIndex-1]+=leaf[r].angle*fraction;
            for(PxU32 s=h.supportStart;s<h.supportStart+h.supportCount;++s)mIncidenceDisplacement[s]+=increment[s-slot0]*fraction;
        }
        for(PxU32 i=0;i<ends;++i)
        {
            const auto& e=endpoint[i];
            if(e.id==rigid){mBodies[rigid].p=pose.p;mBodies[rigid].q=pose.q;}
            else mBlocks[e.id].particle->p+=e.delta*fraction;
        }
        return;
    }
}

#include "DyAvbdRigidLoopSchedule.inl"

void AvbdDynamicsContext::compileBlockColors()
{
    const PxU64 profileStart=mStep.profile?PxTime::getCurrentCounterValue():0;
	AvbdBlockColoring graph;
    mLoopColors.clear();mLoopCursor=0;
    if(compileIncidencePrimal())return;
	PxArray<PxU8> writable(mBlocks.size(), 0);
	PxArray<PxU32> order;
	for (PxU32 i = 0; i < mBlocks.size(); ++i)
	{
		const AvbdBlock &block = mBlocks[i];
		writable[i] = block.type == eAVBD_BLOCK_RIGID6 ? (mBodies[block.objectIndex].invMass > 0.0f &&
		                                                  !mBodies[block.objectIndex].isKinematic)
		                                               : (block.particle && block.particle->invMass > 0.0f);
	}
    bool closedChart=false;for(PxU8 b:mLocalTranslationClosed)closedChart |= b!=0;
    PxArray<PxU32> previousGroups;
    if(mStep.localBudget && closedChart && mStep.iteration>0)previousGroups=mRigidGroupRoots;
    PxArray<PxU32> seedRows(mRows.size());for(PxU32 r=0;r<mRows.size();++r)seedRows[r]=r;
    if(mStep.localBudget && closedChart)
    {
        PxArray<PxU32> slots;
        for(PxU32 r=0;r<mRows.size();++r)
        {
            bool closed=false;
            mRowProgram.forEachBlock(r,[&](PxU32 b){closed |= b<mLocalTranslationClosed.size() && mLocalTranslationClosed[b];});
            if(closed)slots.pushBack(r);
        }
        PxU32 seed=0x9e3779b9u*(mStep.iteration+1);
        for(PxU32 i=slots.size();i>1;--i)
        {seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;std::swap(seedRows[slots[i-1]],seedRows[slots[seed%i]]);}
    }
    if(mStep.localBudget && mStep.iteration>0)for(const auto& g:mRigidGroups)
    {
        if(mStep.profile && g.count<=16)
        {mStep.localShape[g.count][0]+=g.attempts;mStep.localShape[g.count][1]+=g.products;mStep.localShape[g.count][2]+=g.rightHandSides;}
        mStep.retiredGroupStats.attempts+=g.attempts;
        mStep.retiredGroupStats.unsupported+=g.unsupported;
        mStep.retiredGroupStats.kernelSolved+=g.kernelSolved;
        mStep.retiredGroupStats.accepted+=g.accepted;
        mStep.retiredGroupStats.products+=g.products;
        mStep.retiredGroupStats.frictionAttempts+=g.frictionAttempts;
        mStep.retiredGroupStats.frictionAccepted+=g.frictionAccepted;
        mStep.retiredGroupStats.sharedContactAttempts+=g.sharedContactAttempts;
        mStep.retiredGroupStats.sharedContactAccepted+=g.sharedContactAccepted;
        mStep.retiredGroupStats.materialEvaluations+=g.materialEvaluations;
        mStep.retiredGroupStats.materialEndpointUses+=g.materialEndpointUses;
        mStep.retiredGroupStats.rightHandSides+=g.rightHandSides;
        mStep.retiredGroupStats.assemblyTicks+=g.assemblyTicks;
        mStep.retiredGroupStats.factorTicks+=g.factorTicks;
        mStep.retiredGroupStats.searchTicks+=g.searchTicks;
        mStep.retiredGroupStats.geometryTicks+=g.geometryTicks;
        mStep.retiredGroupStats.modelTicks+=g.modelTicks;
        for(PxU32 k=0;k<4;++k)
        {mStep.retiredGroupStats.assemblyPartTicks[k]+=g.assemblyPartTicks[k];mStep.retiredGroupStats.assemblyPartRows[k]+=g.assemblyPartRows[k];}
    }
    if(mStep.localBudget)++mStep.localColorBuilds;
    mRigidGroups.clear();mRigidGroupRoots.clear();mRigidGroupRows.clear();mRigidGroupSharedRows.clear();mRigidGroupFactorOffsets.clear();
    mRigidGroupFactorOffsets.pushBack(0);
    bool grouped=false;
    const bool rigidJoints=!mStep.commonProgram && !mStep.localBudget && !mStep.hasParticleBlocks &&
        mTetrahedronMaterials.empty() && mElasticFrameMaterials.empty() && (!mJoints.empty() || !mArticJoints.empty());
    if(rigidJoints || !mTetrahedronMaterials.empty() || !mElasticFrameMaterials.empty())
    {
        mRigidGroups.resize(mBodies.size());mRigidGroupRoots.resize(mBodies.size());
        for(PxU32 i=0;i<mBodies.size();++i)
        { mRigidGroupRoots[i]=i;AvbdRigidGroup g={};g.blocks[0]=i;g.count=1;mRigidGroups[i]=g; }
        PxArray<PxU8> assigned(mBodies.size(),0);
        PxArray<PxU32> scores((mStep.localBudget || rigidJoints)?mBodies.size():0,0);
        PxArray<double> frameStrength(mStep.localBudget?mBodies.size():0,0);
        if(mStep.localBudget)for(const auto& row:mRows)
            if(row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME && !(row.header().flags&eAVBD_ROW_INACTIVE))
            {
                const auto& material=mElasticFrameMaterials[row.header().geometryIndex-1];double strength=0;
                for(PxU32 d=0;d<3;++d)strength=PxMax(strength,material.parameters.stiffness[d]+material.parameters.dampingOverDt[d]);
                for(PxU32 b:{row.header().bodyA,row.header().bodyB})if(b<frameStrength.size())frameStrength[b]=PxMax(frameStrength[b],strength);
            }
        // Classify original row supports once, before the bounded partition
        // pass. Buckets are count/scan/fill metadata, not physical row copies.
        // Each input row has its own classification output on a device.
        PxArray<PxU32> seeds[2][3];
        for(PxU32 index=0;index<seedRows.size();++index)
        {
            const PxU32 r=seedRows[index];const auto& row=mRows[r];
            const PxU32 kernel=row.header().kernelId;
            const PxU32 phase=kernel==eAVBD_ROW_KERNEL_NP_CONTACT?0:1;
            if(phase==0?!mStep.localBudget:
                ((kernel!=eAVBD_ROW_KERNEL_TETRAHEDRON && kernel!=eAVBD_ROW_KERNEL_ELASTIC_FRAME && !(rigidJoints && isLoopEquality(row))) ||
                    (row.header().flags&eAVBD_ROW_INACTIVE)))continue;
            PxU32 count=0;mRowProgram.forEachBlock(r,[&](PxU32 b){count+=b<mBodies.size() && writable[b];});
            if(count>=2 && count<=4)seeds[phase][count-2].pushBack(r);
        }
        // Couple NP links before material links when their penalty dominates
        // the weaker endpoint's translational frame stiffness. Weak proximity
        // links must not fragment a stiff material patch. This is a numerical
        // partition heuristic only: every boundary row stays in the objective.
        for(PxU32 phase=mStep.localBudget?0:1;phase<2;++phase)
        for(PxU32 preference=0;preference<(previousGroups.empty()?1u:2u);++preference)
        for(PxU32 arity=4;arity>=2;--arity) for(PxU32 r:seeds[phase][arity-2])
        {
            const AvbdRowConstRef& row=mRows[r];
            const bool closedLink=row.header().bodyA<mLocalTranslationClosed.size() && row.header().bodyB<mLocalTranslationClosed.size() &&
                mLocalTranslationClosed[row.header().bodyA] && mLocalTranslationClosed[row.header().bodyB];
            if(phase==0 && !closedLink && row.header().bodyA<frameStrength.size() && row.header().bodyB<frameStrength.size() &&
                double(row.scalar().rho)<PxMin(frameStrength[row.header().bodyA],frameStrength[row.header().bodyB]))continue;
            AvbdRigidGroup group={};bool available=true;
            mRowProgram.forEachBlock(r,[&](PxU32 b)
            {
                if(b>=mBodies.size() || !writable[b]) return;
                if(assigned[b]) available=false;
                if(group.count<4) group.blocks[group.count++]=b;else available=false;
            });
            if(!available || group.count!=arity) continue;
            if(!previousGroups.empty())
            {
                bool cut=false;for(PxU32 k=1;k<group.count;++k)cut |= previousGroups[group.blocks[k]]!=previousGroups[group.blocks[0]];
                const bool closed=group.blocks[0]<mLocalTranslationClosed.size() && mLocalTranslationClosed[group.blocks[0]];
                if(closed?cut!=(preference==0):preference!=0)continue;
            }
            PxU32 references=0;
            for(PxU32 s=0;s<group.count;++s)references+=mRowProgram.blockOffsets[group.blocks[s]+1]-mRowProgram.blockOffsets[group.blocks[s]];
            if((mStep.localBudget || rigidJoints) && references>256)continue;
            // Grow bounded frame groups by internal adjacency, not geometry
            // or body class. Prefer a candidate joining several current nodes.
            while((mStep.localBudget || rigidJoints) && group.count<((group.blocks[0]<mLocalTranslationClosed.size() && mLocalTranslationClosed[group.blocks[0]])?16u:4u))
            {
                PxInlineArray<PxU32,256> candidates;
                for(PxU32 s=0;s<group.count;++s)
                    for(PxU32 slot=mRowProgram.blockOffsets[group.blocks[s]];slot<mRowProgram.blockOffsets[group.blocks[s]+1];++slot)
                    {
                        const auto& link=mRows[mRowProgram.blockRows[slot]];
                        if((link.header().kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME && !(rigidJoints && isLoopEquality(link))) || (link.header().flags&eAVBD_ROW_INACTIVE))continue;
                        const PxU32 other=link.header().bodyA==group.blocks[s]?link.header().bodyB:link.header().bodyA;
                        if(other>=mBodies.size() || !writable[other] || assigned[other])continue;
                        bool inside=false;for(PxU32 k=0;k<group.count;++k)inside |= group.blocks[k]==other;
                        if(inside || references+mRowProgram.blockOffsets[other+1]-mRowProgram.blockOffsets[other]>256)continue;
                        if(!scores[other])candidates.pushBack(other);
                        scores[other]+=(!previousGroups.empty() && other<mLocalTranslationClosed.size() && mLocalTranslationClosed[other] && previousGroups[other]!=previousGroups[group.blocks[s]])?4:1;
                    }
                PxU32 best=PX_MAX_U32,score=0;
                for(PxU32 b:candidates){if(scores[b]>score){best=b;score=scores[b];}scores[b]=0;}
                if(best==PX_MAX_U32)break;
                group.blocks[group.count++]=best;references+=mRowProgram.blockOffsets[best+1]-mRowProgram.blockOffsets[best];
            }
            const PxU32 root=group.blocks[0];
            for(PxU32 s=0;s<group.count;++s)
            { assigned[group.blocks[s]]=1;mRigidGroupRoots[group.blocks[s]]=root;mRigidGroups[group.blocks[s]].count=0; }
            mRigidGroups[root]=group;grouped=true;
        }
        if(rigidJoints)grouped=compileRigidLoops(writable,seedRows,scores);
        for(PxU32 i=0;i<mRigidGroups.size();++i)
        {
            AvbdRigidGroup& group=mRigidGroups[i];if(group.count<2) continue;
            PxInlineArray<PxU32,256> rows;
            for(PxU32 s=0;s<group.count;++s)
                for(PxU32 j=mRowProgram.blockOffsets[group.blocks[s]];j<mRowProgram.blockOffsets[group.blocks[s]+1];++j)
                    rows.pushBack(mRowProgram.blockRows[j]);
            if(rows.size()>1) std::sort(rows.begin(),rows.end());
            group.rowStart=mRigidGroupRows.size();
            group.sharedStart=mRigidGroupSharedRows.size();
            for(PxU32 r=0;r<rows.size();++r) if(!r || rows[r]!=rows[r-1])
            {
                AvbdRigidGroupRow entry={rows[r],i,0};
                mRowProgram.forEachBlock(entry.row,[&](PxU32 block)
                {
                    if(block<mRigidGroupRoots.size() && mRigidGroupRoots[block]==i)
                    { if(!entry.supportCount) entry.representative=block;++entry.supportCount; }
                });
                mRigidGroupRows.pushBack(entry);
                if(entry.supportCount>1)
                {
                    mRigidGroupSharedRows.pushBack(entry.row);
                    const AvbdRowConstRef& row=mRows[entry.row];
                    const PxU32 rank=row.header().kernelId==eAVBD_ROW_KERNEL_ELASTIC_FRAME?9:
                        ((row.header().flags&eAVBD_ROW_CONTACT) && row.scalar().positionTangent && row.scalar().mu>0?3:1);
                    mRigidGroupFactorOffsets.pushBack(mRigidGroupFactorOffsets.back()+rank);
                }
            }
            group.rowCount=mRigidGroupRows.size()-group.rowStart;
            group.sharedCount=mRigidGroupSharedRows.size()-group.sharedStart;
            if(group.leafCount)bindRigidLoopRows(i,scores);
        }
    }
    if(mStep.localBudget && std::getenv("PHYSX_AVBD_GROUP_TRACE"))
    {
        std::string trace="[AVBD_LOCAL_PARTITION] iteration="+std::to_string(mStep.iteration)+" closed=";
        for(PxU32 b=0;b<mLocalTranslationClosed.size();++b)if(mLocalTranslationClosed[b])trace+=std::to_string(b)+",";
        trace+=" groups=";
        for(const auto& group:mRigidGroups)if(group.count>1)
        {for(PxU32 k=0;k<group.count;++k)trace+=std::to_string(group.blocks[k])+",";trace+=";";}
        std::fprintf(stderr,"%s\n",trace.c_str());
    }
    grouped |= mStep.matchNodal;
    auto rootOf=[&](PxU32 block)
    {
        if(block<mRigidGroupRoots.size())return mRigidGroupRoots[block];
        return PxMin(block,nodalPartner(block));
    };
	// Compile one conflict-free color program for single-thread and task execution.
	// Finite elastic objectives need no historical hard-joint conflict order.
	// Derive that execution property from the actual 1D row program, without
	// recognizing joint classes or adding a separate softbody schedule.
	PxArray<PxU8> jointPrograms(mBodies.size(), 0);
	for(PxU32 i = mContactRowCount; i < mRows.size(); ++i)
	{
		const AvbdRowConstRef& row = mRows[i];
		const PxU8 property =
			(row.header().flags & (eAVBD_ROW_DRIVE | eAVBD_ROW_INACTIVE)) == eAVBD_ROW_DRIVE ? 1 : 2;
		mRowProgram.forEachBlock(i, [&](PxU32 block)
		{
			if(mBlocks[block].type == eAVBD_BLOCK_RIGID6)
				jointPrograms[mBlocks[block].objectIndex] |= property;
		});
	}
	for(PxU32 i=0;i<mBodies.size();++i)
		jointPrograms[rootOf(i)] |= jointPrograms[i];
	PxU32 reorderableCount = 0;
	for (PxU32 kind = 0; kind < 3; ++kind)
		for (PxU32 i = 0; i < mBlocks.size(); ++i)
		{
			const AvbdBlock& block = mBlocks[i];
			const PxU32 blockKind = block.type != eAVBD_BLOCK_RIGID6 ? 0 :
				(jointPrograms[block.objectIndex] == 1 ? 1 : 2);
			if (writable[i] && rootOf(i)==i && blockKind == kind)
			{
				order.pushBack(i);
				if (kind < 2) ++reorderableCount;
			}
		}
    AvbdRowProgram groupedProgram;
    if(!grouped) graph.addProgram(mRowProgram);
    else
    {
        // Project the canonical CSR directly. Avoid a heap allocation and
        // sort for every row, followed by a second scope-incidence build.
        // Read-only supports are not vertices of the writable conflict graph.
        groupedProgram.compile(mRows.size(),mBlocks.size(),[&](PxU32 r,const auto& emit)
        {
            mRowProgram.forEachBlock(r,[&](PxU32 b){if(writable[b])emit(rootOf(b));});return true;
        });
        graph.addProgram(groupedProgram);
    }
	// The finite policy preserves read/write conflicts, not the historical
	// hard-joint visitation order. A grid must not become a serial wavefront.
    const bool pointNetwork = rigidJoints && !grouped && pointContactProgram(jointPrograms);
    if (pointNetwork && mStep.iteration == 0)
    {
        mStep.iterationLimit = mStep.minimumJointIterations = PxMax(4u, mStep.rigidPositionBudget);
        mStep.pointInterfaceEnd = mNpContactRowCount;
    }
    const bool independentOrder=pointNetwork || mStep.localBudget || (!mStep.hasParticleBlocks && mJoints.empty() && mArticJoints.empty());
	graph.build(order, mBlocks.size(), mColorOffsets, mColorBlocks,
        independentOrder?order.size():reorderableCount);
    if(mStep.profile)mStep.colorTicks+=PxTime::getCurrentCounterValue()-profileStart;
    if(rigidJoints && grouped)bakeLoopWaves();
}

#include "DyAvbdPointResponse.inl"

void AvbdDynamicsContext::preparePrimalSweep()
{
	mPrimalSteps.resize(mBlocks.size());
	if(mStep.iteration==0)
	{
		mPrimalIdleIteration.clear();mPrimalIdleIteration.resize(mBlocks.size(),0);
		mPrimalChangedIteration.clear();mPrimalChangedIteration.resize(mBlocks.size(),0);
		mPrimalIdleSkips.clear();mPrimalIdleSkips.resize(mBlocks.size(),0);
	}
	for (PxU32 i = 0; i < mBlocks.size(); ++i)
	{
		mPrimalSteps[i] = 0.0f;
	}
}

void AvbdDynamicsContext::solveBoundedBatch(PxU32 begin,PxU32 end,PxU32 slot)
{
    PX_COMPILE_TIME_ASSERT(int(AvbdCpuBatch::TileWidth)==int(AvbdBoundedResponse::BatchWidth));
    PX_ASSERT(slot<AvbdCpuBatch::TaskLimit && end>=begin && end-begin<=AvbdCpuBatch::TileWidth*AvbdCpuBatch::TilesPerTask);
    if(slot>=AvbdCpuBatch::TaskLimit || end<begin || end-begin>AvbdCpuBatch::TileWidth*AvbdCpuBatch::TilesPerTask)
    {PxAtomicExchange(&mLocalDomainFailed,1);return;}
    const PxU64 started=mStep.profile?PxTime::getCurrentCounterValue():0;
    PxAtomicMax(&mPeakPrimalTasks,PxAtomicIncrement(&mActivePrimalTasks));
    for(PxU32 cursor=begin;cursor<end;)
    {
        const PxU32 next=cursor+PxMin(PxU32(AvbdCpuBatch::TileWidth),end-cursor);
        AvbdBoundedBatchArgs args={mBoundedResponse,cursor,next,false};
        mKernels.launch(AVBD_BOUNDED_BATCH,&args);
        if(!args.executed)PxAtomicExchange(&mLocalDomainFailed,1);
        cursor=next;
    }
    PxAtomicDecrement(&mActivePrimalTasks);
    if(mStep.profile)mStep.jobTicks[slot]=PxTime::getCurrentCounterValue()-started;
}

bool AvbdDynamicsContext::dispatchBoundedWave()
{
    auto& response=*mBoundedResponse;
    if(mStep.profile && mStep.profileJobs)
    {
        const PxU64 now=PxTime::getCurrentCounterValue();
        mStep.groupTicks+=now-mStep.groupStart;mStep.tailTicks+=now-mStep.callerDone;
        for(PxU32 i=0;i<mStep.profileJobs;++i)
        {mStep.jobTotal+=mStep.jobTicks[i];mStep.jobMaximum=PxMax(mStep.jobMaximum,mStep.jobTicks[i]);}
        mStep.profileJobs=0;
    }
    if(mBoundedCursor==response.count())
    {response.advance();++mStep.boundedWaves;mBoundedCursor=0;return true;}
    const AvbdCpuBatch pack(response.count()-mBoundedCursor,mStep.continuation?mStep.workers:0);
    const PxU32 count=pack.tasks,start=mBoundedCursor;
    mStep.boundedBatches+=pack.tiles;mStep.boundedJobs+=count;++mStep.boundedGroups;
    if(mStep.profile){mStep.profileJobs=count;mStep.groupStart=PxTime::getCurrentCounterValue();}
    if(mStep.continuation && count>1)
    {
        if(!mColorContinuation)mColorContinuation=PX_NEW(AvbdAdvanceTask)(*this,mContextID);
        auto* next=mColorContinuation;next->setContinuation(*mStep.continuation->getTaskManager(),NULL);
        while(mBlockTasks.size()+1<count)mBlockTasks.pushBack(PX_NEW(AvbdBlockSolveTask)(*this,mContextID));
        for(PxU32 i=0;i+1<count;++i)
        {
            auto* task=mBlockTasks[i];task->local=false;task->bounded=true;task->slot=i;task->begin=mBoundedCursor;
            task->end=start+pack.offset(i+1);mBoundedCursor=task->end;
            task->setContinuation(next);++mStep.submittedTasks;task->removeReference();
        }
        const PxU32 begin=mBoundedCursor;mBoundedCursor=start+pack.items;
        solveBoundedBatch(begin,mBoundedCursor,count-1);
        if(mStep.profile)mStep.callerDone=PxTime::getCurrentCounterValue();
        return next->completeInline();
    }
    const PxU32 begin=mBoundedCursor;mBoundedCursor=start+pack.items;
    solveBoundedBatch(begin,mBoundedCursor,0);
    if(mStep.profile)mStep.callerDone=PxTime::getCurrentCounterValue();
    return true;
}

void AvbdDynamicsContext::solvePrimalBatch()
{
	PxAtomicMax(&mPeakPrimalTasks, PxAtomicIncrement(&mActivePrimalTasks));
	for (;;)
	{
		const PxU32 index = PxU32(PxAtomicIncrement(&mColorCursor) - 1);
		if (index >= mColorEnd)
			break;
		solvePrimalBlock(mColorBlocks[index]);
	}
	PxAtomicDecrement(&mActivePrimalTasks);
}

bool AvbdDynamicsContext::localCapability()
{
    const char* reason=NULL;
    if(!mArticJoints.empty())reason="articulation_update_uncompiled";
    else if(mContactRowCount && (mNpDomainMode!=2 || mNpDomainUnsupported ||
        mNpDomainUnseparated || mNpDomainPolicyExcluded))reason="contact_geometry_or_policy_uncompiled";
    for(const auto& domain:mNpDomains)for(PxU32 s=0;s<2;++s)
        if(domain.points[s].empty() || domain.points[s].size()>AvbdNpQuery::PointLimit)reason="np_support_capacity";
    for(const auto& row:mRows)if(!reason && !(row.header().flags&eAVBD_ROW_INACTIVE))
    {
        if(row.header().kernelId!=eAVBD_ROW_KERNEL_CONSTRAINT && row.header().kernelId!=eAVBD_ROW_KERNEL_FINITE_SPRING &&
           row.header().kernelId!=eAVBD_ROW_KERNEL_ELASTIC_FRAME && row.header().kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON &&
           row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT && row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT &&
           row.header().kernelId!=eAVBD_ROW_KERNEL_ATTACHMENT && !avbdIsNodalMaterial(row.header().kernelId))
            reason="row_kernel_uncompiled";
        if(row.header().supportCount && row.header().kernelId!=eAVBD_ROW_KERNEL_TETRAHEDRON &&
            row.header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT && row.header().kernelId!=eAVBD_ROW_KERNEL_ATTACHMENT &&
            !avbdIsNodalMaterial(row.header().kernelId))reason="row_support_uncompiled";
    }
    for(PxU32 b=0;b<mBlocks.size();++b)if(mBlocks[b].type==eAVBD_BLOCK_PARTICLE3)
    {
        if(mBlocks[b].particleKernel!=eAVBD_PARTICLE_KERNEL_TRIANGLE &&
            mBlocks[b].particleKernel!=eAVBD_PARTICLE_KERNEL_TET)reason="nodal_kernel_uncompiled";
        if(mRowProgram.blockOffsets.size()==mBlocks.size()+1 &&
            mRowProgram.blockOffsets[b+1]-mRowProgram.blockOffsets[b]>AvbdLocalWork::RowLimit)
            mStep.hasWideNodes=true;
    }
    if(!reason)return true;
    std::fprintf(stderr,"[AVBD_LOCAL_BUDGET] status=UNSUPPORTED reason=%s\n",reason);
    return false;
}

void AvbdDynamicsContext::solveLocalBatch(PxU32 begin,PxU32 end,PxU32 stage,PxU32 slot)
{
    if(stage==MaterialSolveStage || stage==MaterialResidualStage || stage==MaterialIslandStage)
    {
        PX_ASSERT(end>=begin && end-begin<=AvbdContactMaterial::PatchesPerTask);
        const auto op=stage==MaterialIslandStage ? AvbdMaterialBatch::IndependentIslands :
            stage==MaterialResidualStage ? AvbdMaterialBatch::Residual : AvbdMaterialBatch::Update;
        AvbdMaterialBatch args={mContactMaterial,mMaterialRowsScratch.begin(),begin,end,op};
        mKernels.launch(AVBD_CONTACT_MATERIAL_BATCH,&args);
        return;
    }
    const PxU64 started=mStep.profile?PxTime::getCurrentCounterValue():0;
    const PxU32 limit=stage==SurfacePackStage?PxU32(AvbdLocalWork::RowLimit):stage==1?PxU32(LocalDualBatchLimit):32u;
    PX_ASSERT(end>=begin && end-begin<=limit);
    if(end<begin || end-begin>limit){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    const bool baked=stage==0 && mColorTasks && !mColorTasks->offsets.empty();
    bool leaf=false;
    if(baked)
    {
        const auto& tile=mColorTasks->tiles[begin];
        leaf=tile.split!=PX_MAX_U32;
        if(!leaf){begin=tile.begin;end=tile.end;}
    }
    if(mStep.countNodalSolves)PxAtomicMax(&mPeakPrimalTasks,PxAtomicIncrement(&mActivePrimalTasks));
    if(stage==SurfacePackStage)packSurfaceRows(begin,end);
    else if(stage==PointInterfaceStage)
        for(PxU32 i=begin;i<end;++i)conditionPointInterface(i);
    else if(stage==WarmCheckStage)for(PxU32 i=begin;i<end;++i)checkWarmRows(i);
    else if(stage==NodalImpactStage)for(PxU32 i=begin;i<end;++i)seedNodalImpact(mNodalImpactRows[i]);
    else if(stage==IncidenceHistoryStage)for(PxU32 i=begin;i<end;++i)prepareIncidenceHistory(i);
    else if(stage==NodalLoadStage)for(PxU32 i=begin;i<end;++i)conditionNodalLoad(i);
    else if(stage==NodalMatchStage)for(PxU32 i=begin;i<end;++i)nominateNodalPartner(i);
    else if(stage==3)
    {
        if(mWideFusedDrain)while(drainOneWide()){}
        else if(mLocalWork)for(PxU32 i=begin;i<end;++i)solveLocalWorkJob(*mLocalWork,i);
    }
    else if(stage==2)for(PxU32 i=begin;i<end;++i)
        validateLocalRow(i<mContactRowCount?i:mNativeMaterialEnd-mNodalContacts.size()+i-mContactRowCount);
    else if(stage==1)applyDualRange(begin,end);
    else if(leaf)
    {
        executeColorLeaf(begin);
        if(mStep.profile)
        {
            const auto& t=mColorTasks->tiles[begin];const auto b=mColorTasks->splits[t.split].block;
            mStep.hotItems[slot]={PxTime::getCurrentCounterValue()-started,b<mBodies.size()?0u:1u,t.rows};
        }
    }
    else if(mStep.profile)
    {
        auto& hot=mStep.hotItems[slot];hot={};
        for(PxU32 i=begin;i<end;++i)
        {
            const PxU32 b=mColorBlocks[i];const auto start=PxTime::getCurrentCounterValue();
            solvePrimalBlock(b);
            const auto ticks=PxTime::getCurrentCounterValue()-start;
            if(ticks>=hot.ticks)
                hot={ticks,b>=mBlocks.size()?3u:b<mBodies.size()?0u:nodalPartner(b)!=PX_MAX_U32?2u:1u,
                    localItemRows(b,localItemBlocks(b))};
        }
    }
    else for(PxU32 i=begin;i<end;++i)solvePrimalBlock(mColorBlocks[i]);
    if(mStep.profile)
    {
        const PxU64 ticks=PxTime::getCurrentCounterValue()-started;
        mStep.localWaveTaskStarts[slot]=started;
        mStep.localWaveTaskTicks[slot]=ticks; // Exclusive slot, consumed after fan-in.
        if(mStep.localTimeline)mStep.localWaveTaskThreads[slot]=PxThread::getId();
        PxAtomicAdd(&mStep.localTaskTicks[stage],PxI64(ticks));
    }
    if(mStep.countNodalSolves)PxAtomicDecrement(&mActivePrimalTasks);
}

void AvbdDynamicsContext::finishLocalProfileWave()
{
    if(!mStep.localWaveStart)return;
    const PxU64 span=PxTime::getCurrentCounterValue()-mStep.localWaveStart;
    const PxU32 stage=mStep.localWaveStage;
    PxU64 work=0,longest=0,first=~PxU64(0),last=0;
    PxU64 durations[LocalTaskLimit],loads[32]={};
    for(PxU32 i=0;i<mStep.localWaveTaskCount;++i)
    {
        durations[i]=mStep.localWaveTaskTicks[i];work+=durations[i];longest=PxMax(longest,durations[i]);
        first=PxMin(first,mStep.localWaveTaskStarts[i]);last=PxMax(last,mStep.localWaveTaskStarts[i]+durations[i]);
    }
    // Read-only offline schedule estimate. It never changes task ordering.
    if(stage==0)
    {
        const PxU32 slot=PxU32(std::max_element(durations,durations+mStep.localWaveTaskCount)-durations);
        const auto& h=mStep.hotItems[slot];
        mStep.criticalItemTicks[h.kind]+=h.ticks;mStep.criticalItemRows[h.kind]+=h.rows;++mStep.criticalItemCount[h.kind];
    }
    std::sort(durations,durations+mStep.localWaveTaskCount);
    for(PxU32 i=mStep.localWaveTaskCount;i>0;--i)
    {
        PxU32 lane=0;for(PxU32 j=1;j<mStep.localWaveJobs;++j)if(loads[j]<loads[lane])lane=j;
        loads[lane]+=durations[i-1];
    }
    PxU64 lpt=0;for(PxU32 j=0;j<mStep.localWaveJobs;++j)lpt=PxMax(lpt,loads[j]);
    if(mStep.localTimeline)
    {
        // Collect after fan-in; never log between waves and keep workers awake
        // accidentally. Queued idle capacity is distinct from insufficient width.
        struct Event { PxU64 time;PxI32 ready,active; } events[3*LocalTaskLimit];
        const PxU64 loss=span>lpt?span-lpt:0;
        for(PxU32 i=0;i<mStep.localWaveTaskCount;++i)
        {
            const PxU64 ready=mStep.localWaveTaskReady[i]-mStep.localWaveStart;
            const PxU64 start=mStep.localWaveTaskStarts[i]-mStep.localWaveStart;
            const PxU64 end=start+mStep.localWaveTaskTicks[i];
            events[3*i]={ready,1,0};events[3*i+1]={start,-1,1};events[3*i+2]={end,0,-1};
            if(loss>=mStep.localTraceLoss)
            {mStep.localTrace[i][0]=ready;mStep.localTrace[i][1]=start;mStep.localTrace[i][2]=end;
                mStep.localTrace[i][3]=mStep.localWaveTaskThreads[i];}
        }
        if(loss>=mStep.localTraceLoss)
        {mStep.localTraceLoss=loss;mStep.localTraceSpan=span;mStep.localTraceLpt=lpt;
            mStep.localTraceStage=stage;mStep.localTraceJobs=mStep.localWaveTaskCount;mStep.localTraceLanes=mStep.localWaveJobs;}
        const PxU32 count=3*mStep.localWaveTaskCount;
        std::sort(events,events+count,[](const Event& a,const Event& b){return a.time<b.time;});
        PxI32 ready=0,active=0;PxU64 previous=0;
        for(PxU32 i=0;i<count;++i)
        {
            const PxI32 idle=PxMax(0,PxI32(mStep.localWaveJobs)-active);
            mStep.localReadyIdleTicks[stage]+=(events[i].time-previous)*PxMin(ready,idle);
            mStep.localWidthIdleTicks[stage]+=(events[i].time-previous)*PxMax(0,idle-ready);
            ready+=events[i].ready;active+=events[i].active;previous=events[i].time;
        }
        mStep.localWidthIdleTicks[stage]+=(span-previous)*mStep.localWaveJobs;
    }
    mStep.localLptTicks[stage]+=lpt;
    mStep.localStartupTicks[stage]+=first-mStep.localWaveStart;
    mStep.localDrainTicks[stage]+=mStep.localWaveStart+span-last;
    mStep.localWorkBoundTicks[stage]+=PxMax(work/mStep.localWaveJobs,longest);
    mStep.localLongestTicks[stage]+=longest;
    mStep.localWaveTicks[stage]+=span;mStep.localCapacityTicks[stage]+=span*mStep.localWaveJobs;
    ++mStep.localWaves[stage];mStep.localWaveStart=0;
}

PxU32 AvbdDynamicsContext::localItemBlocks(PxU32 b) const
{
    if(b>=mBlocks.size())
    {
        const auto& h=mRows[b-mBlocks.size()].header();
        return h.kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT?PxU32(mIncidencePackets[h.geometryIndex-1].entries):PxU32(mIncidenceFacePackets[b-mBlocks.size()]->nodes);
    }
    return b<mRigidGroups.size()?PxU32(mRigidGroups[b].count):(nodalPartner(b)!=PX_MAX_U32?2u:1u);
}

PxU32 AvbdDynamicsContext::localItemRows(PxU32 block,PxU32 size) const
{
    if(block>=mBlocks.size())
    {
        const PxU32 r=block-mBlocks.size();
        return r<mIncidenceFacePackets.size()?3u*mIncidenceFacePackets[r]->rows:size;
    }
    PxU32 rows=0;
    for(PxU32 s=0;s<size;++s)
    {
        const PxU32 b=size==1?block:block<mRigidGroups.size()?mRigidGroups[block].blocks[s]:(s?mNodalPartners[block]:block);
        rows+=mRowProgram.blockOffsets[b+1]-mRowProgram.blockOffsets[b];
    }
    return PxMax(1u,rows);
}

PxU32 AvbdDynamicsContext::localColorStart(PxU32 color) const
{
    return mColorTasks && !mColorTasks->offsets.empty()?mColorTasks->offsets[color]:mColorOffsets[color];
}

PxU32 AvbdDynamicsContext::localDualRows() const
{
    return mRowProgram.dualRows.rows+(mStep.incidencePrimal && mStep.iteration+1==mStep.iterationLimit?
        mRowProgram.historyRows.rows:0);
}

void AvbdDynamicsContext::bakeLocalColorTasks()
{
    if(!mStep.localBudget || !mStep.hasParticleBlocks)
    {if(mColorTasks)mColorTasks->clear();return;}
    if(!mColorTasks)mColorTasks=PX_NEW(AvbdColorTasks)();
    auto& plan=*mColorTasks;
    const PxU32 workers=mStep.continuation?PxClamp(mStep.workers,1u,32u):1;
    const bool incidence=mStep.incidencePrimal;
    const auto equal=[](const PxArray<PxU32>& a,const PxArray<PxU32>& b)
    {return a.size()==b.size() && (a.empty() || !std::memcmp(a.begin(),b.begin(),size_t(a.size())*sizeof(PxU32)));};
    if(incidence && plan.workers==workers && equal(plan.incidenceOffsets,mColorOffsets) &&
        equal(plan.incidenceShape,mIncidenceTaskShape))
    {++mStep.reusedColorPlans;return;}
    plan.clear();plan.workers=workers;
    ++mStep.bakedColorPlans;
    for(PxU32 color=0;color+1<mColorOffsets.size();++color)
    {
        const PxU32 first=mColorOffsets[color],end=mColorOffsets[color+1];
        plan.offsets.pushBack(plan.tiles.size());plan.wideOffsets.pushBack(plan.wideBlocks.size());
        PxU64 total=0;
        for(PxU32 i=first;i<end;++i)
        {
            const PxU32 b=mColorBlocks[i],size=localItemBlocks(b),rows=localItemRows(b,size);
            // Above the one-block limit the existing multilevel bounded
            // reduction owns assembly AND trial. Do not truncate its domain.
            if(size>32 || (rows>AvbdLocalWork::RowLimit && size!=1)){plan.clear();return;}
            if(rows>AvbdLocalWork::RowLimit){plan.wideBlocks.pushBack(b);continue;}
            total+=rows;
        }
        const PxU32 perLane=PxU32(PxMin(PxU64(256),(total+workers-1)/workers));
        const PxU32 target=workers==1?256:PxClamp(perLane,16u,256u);
        PxU32 begin=first,blocks=0,refs=0;
        const auto flush=[&](PxU32 stop)
        {
            if(begin<stop)plan.tiles.pushBack({begin,stop,PX_MAX_U32,0,blocks,refs});
            begin=stop;blocks=refs=0;
        };
        for(PxU32 i=first;i<end;++i)
        {
            const PxU32 b=mColorBlocks[i],size=localItemBlocks(b),rows=localItemRows(b,size);
            // A workgroup cooperates only on a single unknown. Paired
            // translations retain their complete off-diagonal Schur blocks.
            // Slice expensive rigid contact assembly or a node longer than
            // a lane's share of this color, up to 16 leaves / 256 rows.
            const bool split=!incidence && workers>1 && size==1 && rows>=2*AvbdColorTasks::LeafRows &&
                (b<mBodies.size() || rows>perLane);
            if(rows>AvbdLocalWork::RowLimit){flush(i);begin=i+1;continue;}
            if(split)
            {
                flush(i);const bool rigid=b<mBodies.size();const PxU32 count=(rows+AvbdColorTasks::LeafRows-1)/AvbdColorTasks::LeafRows;
                const PxU32 part=rigid?plan.rigid.size():plan.nodal.size(),id=plan.splits.size();
                if(rigid)plan.rigid.resize(part+count);else plan.nodal.resize(part+count);
                plan.splits.pushBack({b,part,count,PxI32(count)});
                const PxU32 start=mRowProgram.blockOffsets[b];
                for(PxU32 l=0;l<count;++l)
                {
                    const PxU32 lo=l*AvbdColorTasks::LeafRows,hi=PxMin(rows,lo+AvbdColorTasks::LeafRows);
                    plan.tiles.pushBack({start+lo,start+hi,id,part+l,l?0u:1u,hi-lo});
                }
                begin=i+1;
            }
            else
            {
                if(i>begin && (blocks+size>32 || refs+rows>target || i-begin==32))flush(i);
                blocks+=size;refs+=rows;
            }
        }
        flush(end);
    }
    plan.wideOffsets.pushBack(plan.wideBlocks.size());plan.offsets.pushBack(plan.tiles.size());
    for(PxU32 c=0;c+1<plan.offsets.size();++c)
    {
        const PxU32 first=plan.offsets[c],end=plan.offsets[c+1];
        plan.offsets[c]=plan.waves.size();
        if(first==end)continue;
        // Keep authored tile order. Same-color blocks do not read each other,
        // but reordering them still changes the published floating-point path.
        AvbdColorTasks::Wave wave={first,end-first,0,0,0,0,0};
        for(PxU32 t=first;t<end;++t)
        {
            const auto& tile=plan.tiles[t];wave.blocks+=tile.blocks;
            wave.maxBlocks=PxMax(wave.maxBlocks,tile.blocks);wave.maxRows=PxMax(wave.maxRows,tile.rows);
            if(tile.split!=PX_MAX_U32){wave.splitBlocks+=tile.blocks;++wave.leaves;}
        }
        plan.waves.pushBack(wave);
    }
    plan.offsets.back()=plan.waves.size();
    if(incidence){plan.incidenceOffsets=mColorOffsets;plan.incidenceShape=mIncidenceTaskShape;}
}

void AvbdDynamicsContext::executeColorLeaf(PxU32 tileIndex)
{
    auto& plan=*mColorTasks;const auto tile=plan.tiles[tileIndex];auto& split=plan.splits[tile.split];
    const PxU32 b=split.block;const bool rigid=b<mBodies.size();
    AvbdLocalBlockState state;
    if(rigid)
    {
        state.body=b;state.original=PxTransform(mBodies[b].p,mBodies[b].q);
        sampleLocalBlock(state,tile.begin,tile.end,plan.rigid[tile.partial],false);
    }
    else
    {
        auto& partial=plan.nodal[tile.partial];partial=AvbdNodalAssembly();
        assembleNodalRows(b,tile.begin,tile.end,partial);
    }
    // All leaf reads complete before the final lane solves and publishes.
    // A same-color neighbour cannot read these supports. No worker waits.
    if(PxAtomicDecrement(&split.remaining))return;
    split.remaining=PxI32(split.count);
    if(mLocalDomainFailed)return;
    for(PxU32 stride=1;stride<split.count;stride*=2)
        for(PxU32 i=0;i+stride<split.count;i+=2*stride)
        {
            const PxU32 a=split.first+i,c=a+stride;
            if(rigid)plan.rigid[a].merge(plan.rigid[a],plan.rigid[c],false);
            else plan.nodal[a].add(plan.nodal[c]);
        }
    if(rigid)
    {
        mPrimalSteps[b]=finishRigidAssembly(state,plan.rigid[split.first]);
        if(mBodies[b].p!=state.original.p || !(mBodies[b].q==state.original.q))mPrimalChangedIteration[b]=mStep.iteration+1;
    }
    else finishNodalAssembly(b,plan.nodal[split.first]);
}

bool AvbdDynamicsContext::dispatchRowRanges(PxU32 stage,PxU32& cursor,PxU32 end,PxU32 perTask)
{
    const PxU32 workers=PxClamp(mStep.workers,1u,32u);
    const PxU32 limit=workers==1?1:workers*LocalQueueDepth;
    PxU32 starts[LocalTaskLimit],stops[LocalTaskLimit],count=0;
    while(count<limit && cursor<end)
    {
        starts[count]=cursor;
        cursor+=PxMin(perTask,end-cursor);
        stops[count++]=cursor;
    }
    return submitLocalWave(stage,count,starts,stops);
}

bool AvbdDynamicsContext::dispatchLocalBatch(PxU32 stage)
{
    if(stage==0 && mColorTasks && !mColorTasks->offsets.empty())
    {
        const auto& w=mColorTasks->waves[mStep.localCursor++];
        mStep.localBlocks+=w.blocks;++mStep.bakedColorWaves;
        mStep.splitPrimalBlocks+=w.splitBlocks;mStep.splitRowLeaves+=w.leaves;
        mStep.localPrimalMaxBlocks=PxMax(mStep.localPrimalMaxBlocks,w.maxBlocks);
        mStep.localPrimalMaxRows=PxMax(mStep.localPrimalMaxRows,w.maxRows);
        // One cohort per color, the same shape as a single color-group launch.
        const PxU32 lanes=PxMin(w.count,PxMax(1u,mColorTasks->workers));
        mColorTasks->tileCount=w.count;
        if(lanes<w.count)mColorTasks->nextTile=PxI32(lanes);
        return submitLocalWave(stage,lanes,NULL,NULL,w.first);
    }
    PxU32& cursor=stage==NodalImpactStage?mStep.impactCursor:stage==IncidenceHistoryStage?mStep.historyCursor:stage==NodalLoadStage?mStep.loadCursor:stage==NodalMatchStage?mStep.nominationCursor:stage==3?mLocalWork->cursor:stage==2?mStep.queryCursor:stage==1?mStep.dualCursor:mStep.localCursor;
    const PxU32 end=stage==NodalImpactStage?mNodalImpactRows.size():stage==IncidenceHistoryStage?incidenceHistoryEnd(cursor>=incidenceHistoryEnd(false)):stage==NodalLoadStage?mNodalContacts.size():stage==NodalMatchStage?nominationJobs():stage==3?mLocalWork->jobs():stage==2?mContactRowCount+mNodalContacts.size():stage==1?localDualRows():localColorStart(mStep.nextColor+1);
    if(stage==3)++mStep.tiledWaves;
    const PxU32 workers=mStep.continuation?PxMax(1u,PxMin(32u,mStep.workers)):1u;
    // Every phase queues bounded independent jobs, with one fan-in before
    // changing the color, dual epoch or high-degree reduction level.
    const PxU32 taskLimit=workers>1?workers*LocalQueueDepth:1;
    const PxU32 itemLimit=stage==3?1u:stage==1?PxU32(LocalDualBatchLimit):32u;
    if(stage==0 && cursor==mColorOffsets[mStep.nextColor])
    {
        // One linear cost scan per color, with no persistent topology cache.
        mStep.localColorWork=0;
        for(PxU32 i=cursor;i<end;++i)
        {
            const PxU32 b=mColorBlocks[i],size=localItemBlocks(b);
            const PxU32 work=localItemRows(b,size);
            if(work<=AvbdLocalWork::RowLimit)mStep.localColorWork+=work;
        }
    }
    // Tiny 3x3 stencils cost less than a CPU dispatch. Coalesce independent
    // work within the same color; sufficiently wide colors still issue two
    // jobs per lane. Device work items and both hard batch bounds are intact.
    const PxU32 lanes=mStep.hasParticleBlocks && mStep.localColorWork<=128*workers?workers:2*workers;
    const PxU32 rowTarget=stage==0 && workers>1?
        PxU32(PxMax(PxU64(mStep.hasParticleBlocks?16:1),PxMin(PxU64(256),(mStep.localColorWork+lanes-1)/lanes))):256;
    PxU32 starts[LocalTaskLimit],stops[LocalTaskLimit],count=0;
    while(count<taskLimit && cursor<end)
    {
        if(stage==1)
        {
            const PxU32 dual=mRowProgram.dualRows.rows;
            const auto& ranges=cursor<dual?mRowProgram.dualRows:mRowProgram.historyRows;
            PxU32 first;
            const PxU32 size=PxMin(PxU32(LocalDualBatchLimit),ranges.locate(cursor<dual?cursor:cursor-dual,first));
            starts[count]=first;stops[count]=first+size;cursor+=size;++count;
            mStep.localDualMaxRows=PxMax(mStep.localDualMaxRows,size);mStep.localRows+=size;
            continue;
        }
        if(stage==3)
        {
            const auto& work=*mLocalWork;
            const PxU32 state=work.phase==AvbdLocalWork::Direction || work.phase==AvbdLocalWork::Accept?cursor:
                work.nodes[(work.phase==AvbdLocalWork::Assemble || work.phase==AvbdLocalWork::Trial)?
                    work.leaves[cursor]:work.levels[work.level][cursor]].state;
            // The leaf kernel already returns for a completed local search.
            // Avoid submitting its no-op jobs in later bounded search rounds.
            if(work.states[state].done){++cursor;continue;}
        }
        starts[count]=cursor;PxU32 references=0,physicalBlocks=0;
        while(cursor<end && cursor-starts[count]<itemLimit)
        {
            const PxU32 block=stage==0?mColorBlocks[cursor]:0;
            const PxU32 size=stage==0?localItemBlocks(block):1;
            PxU32 work=stage==NodalMatchStage?nominationRows(cursor):stage==IncidenceHistoryStage?incidenceHistoryRows(cursor):1u;
            if(stage==0)
            {
                work=localItemRows(block,size);
                if(work>AvbdLocalWork::RowLimit)
                {
                    if(size!=1){PxAtomicExchange(&mLocalDomainFailed,1);return true;}
                    if(references)break;
                    // The color's high-degree singletons join the same update
                    // through bounded row leaves after these inline tasks.
                    ++cursor;starts[count]=cursor;continue;
                }
            }
            if(stage==0 && references && references+work>rowTarget)break;
            if(references+work>256 || (stage==0 && physicalBlocks+size>32))break;
            if(stage==0)mStep.localColorWork-=work;
            references+=work;physicalBlocks+=size;++cursor;
        }
        if(cursor==starts[count])break;
        stops[count]=cursor;
        if(stage==0)
        {mStep.localPrimalMaxBlocks=PxMax(mStep.localPrimalMaxBlocks,physicalBlocks);
            mStep.localPrimalMaxRows=PxMax(mStep.localPrimalMaxRows,references);}
        else if(stage==1)mStep.localDualMaxRows=PxMax(mStep.localDualMaxRows,cursor-starts[count]);
        else if(stage==2)mStep.localQueryMaxRows=PxMax(mStep.localQueryMaxRows,cursor-starts[count]);
        else if(stage==NodalMatchStage)mStep.nominationMaxRows=PxMax(mStep.nominationMaxRows,references);
        if(stage==1)mStep.localRows+=cursor-starts[count];
        else if(stage==3)mStep.tiledJobs+=cursor-starts[count];
        else if(stage==0)mStep.localBlocks+=physicalBlocks;
        ++count;
    }
    if(!count)return true;
    return submitLocalWave(stage,count,starts,stops);
}

void AvbdDynamicsContext::runColorLane(PxU32 slot)
{
    const PxU32 lanes=mColorTasks->lanes;
    // Flat submit. A 3-ary fan leaves later workers asleep while the first
    // lane is already inside a wide vertex.
    if(slot==0)
    {
        for(PxU32 child=1;child<lanes;++child)
        {
            if(mStep.localTimeline)mStep.localWaveTaskReady[child]=PxTime::getCurrentCounterValue();
            mColorLanes[child-1]->submit(*mColorContinuation);
        }
    }
    const PxU64 laneStart=mStep.profile?PxTime::getCurrentCounterValue():0;
    StepState::HotItem hottest={};
    if(mWideArmed)drainOneWide();
    const auto runTile=[&](PxU32 tile)
    {
        solveLocalBatch(tile,tile+1,0,slot);
        if(mStep.profile && mStep.hotItems[slot].ticks>=hottest.ticks)hottest=mStep.hotItems[slot];
    };
    if(slot<mColorTasks->tileCount)
        runTile(mColorTasks->firstTile+slot);
    if(mColorTasks->tileCount>lanes)
    {
        for(;;)
        {
            const PxU32 index=PxU32(PxAtomicIncrement(&mColorTasks->nextTile)-1);
            if(index>=mColorTasks->tileCount)break;
            runTile(mColorTasks->firstTile+index);
        }
    }
    while(mWideArmed && drainOneWide()){}
    if(mStep.profile)
    {
        mStep.hotItems[slot]=hottest;
        mStep.localWaveTaskStarts[slot]=laneStart;
        mStep.localWaveTaskTicks[slot]=PxTime::getCurrentCounterValue()-laneStart;
    }
}

bool AvbdDynamicsContext::submitLocalWave(PxU32 stage,PxU32 count,const PxU32* starts,const PxU32* stops,PxU32 firstTile)
{
    const PxU32 workers=mStep.continuation?PxClamp(mStep.workers,1u,32u):1u;
    mStep.localQueuePeak=PxMax(mStep.localQueuePeak,count);
    if(mStep.profile)
    {mStep.localWaveStart=PxTime::getCurrentCounterValue();mStep.localWaveStage=stage;
        mStep.localWaveJobs=PxMin(count,workers);mStep.localWaveTaskCount=count;}
    if(count>1)
    {
        if(!mColorContinuation)mColorContinuation=PX_NEW(AvbdAdvanceTask)(*this,mContextID);
        auto* next=mColorContinuation;auto& tm=*mStep.continuation->getTaskManager();
        if(!starts)
        {
            while(mColorLanes.size()+1<count)
                mColorLanes.pushBack(PX_NEW(AvbdColorLane)(*this,mContextID,mColorLanes.size()+1));
            mColorTasks->firstTile=firstTile;mColorTasks->lanes=count;
            next->reserveWave(tm,count);mStep.submittedTasks+=count-1;
            if(mStep.localTimeline)mStep.localWaveTaskReady[0]=PxTime::getCurrentCounterValue();
            runColorLane(0);
            const bool complete=next->completeInline();if(complete)finishLocalProfileWave();return complete;
        }
        next->setContinuation(tm,NULL);
        while(mBlockTasks.size()+1<count)mBlockTasks.pushBack(PX_NEW(AvbdBlockSolveTask)(*this,mContextID));
        for(PxU32 i=0;i+1<count;++i)
        {
            auto* task=mBlockTasks[i];task->local=true;task->bounded=false;task->localStage=stage;
            task->begin=starts[i];task->end=stops[i];task->slot=i;
            ++mStep.submittedTasks;
            if(mStep.localTimeline)mStep.localWaveTaskReady[i]=PxTime::getCurrentCounterValue();
            task->setContinuation(next);task->removeReference();
        }
        if(mStep.localTimeline)mStep.localWaveTaskReady[count-1]=PxTime::getCurrentCounterValue();
        solveLocalBatch(starts[count-1],stops[count-1],stage,count-1);
        const bool complete=next->completeInline();if(complete)finishLocalProfileWave();return complete;
    }
    if(!starts)
    {
        mColorTasks->firstTile=firstTile;mColorTasks->lanes=1;
        if(mStep.localTimeline)mStep.localWaveTaskReady[0]=PxTime::getCurrentCounterValue();
        runColorLane(0);finishLocalProfileWave();return true;
    }
    if(mStep.localTimeline)mStep.localWaveTaskReady[0]=PxTime::getCurrentCounterValue();
    solveLocalBatch(starts[0],stops[0],stage,0);finishLocalProfileWave();return true;
}

void AvbdDynamicsContext::armWideColor()
{
    if(mWidePrepared==mStep.nextColor)return;
    mWidePrepared=mStep.nextColor;
    mWideActive=0;mWideArmed=false;mWideCursor=0;mWideFusedDrain=false;
    const bool baked=mColorTasks && !mColorTasks->offsets.empty();
    const PxArray<PxU32>& offsets=baked?mColorTasks->wideOffsets:mColorOffsets;
    const PxArray<PxU32>& blocks=baked?mColorTasks->wideBlocks:mColorBlocks;
    if(mStep.nextColor+1>=offsets.size())return;
    PxArray<PxU32> chosen;
    for(PxU32 k=offsets[mStep.nextColor];k<offsets[mStep.nextColor+1];++k)
    {
        const PxU32 b=blocks[k];if(b>=mBlocks.size())continue;
        const PxU32 begin=mRowProgram.blockOffsets[b],end=mRowProgram.blockOffsets[b+1];
        if(end>begin && end-begin>AvbdLocalWork::RowLimit)chosen.pushBack(b);
    }
    if(chosen.empty())return;
    std::sort(chosen.begin(),chosen.end(),[&](PxU32 a,PxU32 b)
    {
        const PxU32 ra=mRowProgram.blockOffsets[a+1]-mRowProgram.blockOffsets[a];
        const PxU32 rb=mRowProgram.blockOffsets[b+1]-mRowProgram.blockOffsets[b];
        return ra!=rb?ra>rb:a<b;
    });
    PxU64 scratch=0;
    for(PxU32 n=0;n<chosen.size();++n)
    {
        const PxU32 b=chosen[n];
        const PxU32 begin=mRowProgram.blockOffsets[b],end=mRowProgram.blockOffsets[b+1];
        while(mWidePool.size()<=n)mWidePool.pushBack(PX_NEW(AvbdLocalWork)());
        AvbdLocalWork& work=*mWidePool[n];
        work.reset();work.add(b,begin,end);
        const bool node=b>=mBodies.size();
        if(node)
        {
            work.nodal.resize(work.nodes.size());work.nodalEnergy.resize(work.nodes.size());
            work.nodalTrials.resize(work.states.size());
        }
        else
        {
            work.states.back().original=PxTransform(mBodies[b].p,mBodies[b].q);
            work.partials.resize(work.nodes.size());
        }
        ++mStep.tiledBlocks;++mStep.localBlocks;
        mStep.tiledLeaves+=work.leaves.size();
        mStep.tiledMaxRows=PxMax(mStep.tiledMaxRows,end-begin);
        scratch+=PxU64(work.nodes.size())*sizeof(AvbdLocalWork::Node)+PxU64(work.partials.size())*sizeof(AvbdLocalPartial)+
            PxU64(work.states.size())*sizeof(AvbdLocalBlockState)+PxU64(work.nodal.size())*sizeof(AvbdNodalAssembly)+
            PxU64(work.nodalEnergy.size())*sizeof(AvbdNodalEnergy)+PxU64(work.nodalTrials.size())*sizeof(AvbdNodalTrial);
    }
    mStep.tiledScratchBytes=PxMax(mStep.tiledScratchBytes,scratch);
    mWideActive=chosen.size();mWideArmed=true;mWideCursor=0;++mStep.tiledWaves;
}

void AvbdDynamicsContext::retireWide()
{
    PxU64 jobs=0;
    for(PxU32 i=0;i<mWideActive;++i)jobs+=mWidePool[i]->executed;
    mStep.tiledJobs+=jobs;
    mWideActive=0;mWideArmed=false;mWideFusedDrain=false;mWideCursor=0;
}

bool AvbdDynamicsContext::drainOneWide()
{
    if(!mWideArmed || mLocalDomainFailed)return false;
    const PxU32 i=PxU32(PxAtomicIncrement(&mWideCursor)-1);
    if(i>=mWideActive)return false;
    runWideBlock(*mWidePool[i]);
    return true;
}

void AvbdDynamicsContext::runWideBlock(AvbdLocalWork& work)
{
    // The binary reduction and the bounded line search are unchanged. They
    // used to be a dispatcher wave per phase; the leaves stay within RowLimit.
    while(!mLocalDomainFailed)
    {
        const PxU32 n=work.jobs();
        for(PxU32 job=0;job<n;++job)
        {
            const bool dir=work.phase==AvbdLocalWork::Direction || work.phase==AvbdLocalWork::Accept;
            PxU32 state;
            if(dir)state=job;
            else
            {
                const bool leaf=work.phase==AvbdLocalWork::Assemble || work.phase==AvbdLocalWork::Trial;
                const PxU32 id=leaf?work.leaves[job]:work.levels[work.level][job];
                state=work.nodes[id].state;
            }
            if(work.states[state].done)continue;
            solveLocalWorkJob(work,job);
            ++work.executed;
        }
        switch(work.phase)
        {
        case AvbdLocalWork::Assemble:
            work.phase=work.levels.empty()?AvbdLocalWork::Direction:AvbdLocalWork::ReduceAssembly;work.level=0;break;
        case AvbdLocalWork::ReduceAssembly:
            if(++work.level==work.levels.size())work.phase=AvbdLocalWork::Direction;break;
        case AvbdLocalWork::Direction:work.phase=AvbdLocalWork::Trial;break;
        case AvbdLocalWork::Trial:
            work.phase=work.levels.empty()?AvbdLocalWork::Accept:AvbdLocalWork::ReduceTrial;work.level=0;break;
        case AvbdLocalWork::ReduceTrial:
            if(++work.level==work.levels.size())work.phase=AvbdLocalWork::Accept;break;
        case AvbdLocalWork::Accept:
        {
            bool done=true;for(const auto& state:work.states)done &= state.done;
            if(done)return;
            if(++work.search>=AvbdLocalWork::SearchLimit){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            work.phase=AvbdLocalWork::Trial;break;
        }
        }
    }
}

bool AvbdDynamicsContext::dispatchFusedWide()
{
    if(!mWideActive){mWideArmed=false;return true;}
    const PxU32 workers=mStep.continuation?PxClamp(mStep.workers,1u,32u):1u;
    const PxU32 lanes=PxMin(mWideActive,workers);
    mWideCursor=0;mWideFusedDrain=true;
    if(workers<=1)
    {
        while(drainOneWide()){}
        retireWide();return true;
    }
    if(lanes==1)
    {
        if(!mColorContinuation)mColorContinuation=PX_NEW(AvbdAdvanceTask)(*this,mContextID);
        auto* next=mColorContinuation;auto& tm=*mStep.continuation->getTaskManager();
        next->setContinuation(tm,NULL);
        if(mBlockTasks.empty())mBlockTasks.pushBack(PX_NEW(AvbdBlockSolveTask)(*this,mContextID));
        auto* task=mBlockTasks[0];
        task->local=true;task->bounded=false;task->localStage=3;task->begin=0;task->end=1;task->slot=0;
        ++mStep.submittedTasks;
        mStep.localQueuePeak=PxMax(mStep.localQueuePeak,1u);
        if(mStep.profile)
        {mStep.localWaveStart=PxTime::getCurrentCounterValue();mStep.localWaveStage=3;
            mStep.localWaveJobs=1;mStep.localWaveTaskCount=1;}
        if(mStep.localTimeline)mStep.localWaveTaskReady[0]=PxTime::getCurrentCounterValue();
        task->setContinuation(next);task->removeReference();
        const bool complete=next->completeInline();
        if(complete){finishLocalProfileWave();retireWide();}
        return complete;
    }
    PxU32 starts[32],stops[32];
    for(PxU32 i=0;i<lanes;++i){starts[i]=0;stops[i]=1;}
    const bool done=submitLocalWave(3,lanes,starts,stops);
    if(done)retireWide();
    return done;
}

void AvbdDynamicsContext::compileNodalSurfaceRows()
{
    // Native materials and NP contacts share the same block CSR and colors.
    PxU32 rowCount=0,supportCount=0;
    for(auto& cache:mIslandSurfaceCache)if(cache.liveStamp==mIslandSurfaceLiveStamp && cache.surface)
    {
        const auto& core=cache.surface->getCore();
        if(core.cpuElasticityDamping!=0 || core.cpuBendingDamping!=0)
        {
            std::fprintf(stderr,"[AVBD_PARTICLE3] status=UNSUPPORTED reason=host_np_or_damping_uncompiled\n");
            PxAtomicExchange(&mLocalDomainFailed,1);return;
        }
        const AvbdNodalReal E=core.cpuYoungs,nu=core.cpuPoissons,t=core.cpuThickness;
        if(!(E>0 && nu>-1 && nu<.5f && t>0) || cache.nbTriangles!=cache.restMesh->getNbTriangles())
        {PxAtomicExchange(&mLocalDomainFailed,1);return;}
        const AvbdNodalReal mu=E*t/(2*(1+nu)),lambda=E*t*nu/(1-nu*nu),bending=core.cpuBendingStiffness;
        const bool flatten=core.surfaceFlags.isSet(PxDeformableSurfaceFlag::eENABLE_FLATTENING);
        const bool bakeBending=!cache.nodalRest || cache.bakedFlattening!=flatten;
        if(!cache.nodalRest)
        {
            cache.nodalRest=static_cast<AvbdNodalShellRest*>(PX_ALLOC(
                sizeof(AvbdNodalShellRest)*(cache.nbTriangles+cache.nbBends),"AvbdNodalShellRest"));
            if(!cache.nodalRest){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            PxF64 areaSum=0;
            for(PxU32 i=0;i<cache.nbTriangles;++i)
            {
                const auto& tri=cache.triangles[i];auto& rest=cache.nodalRest[i];
                rest=AvbdNodalShellRest();rest.area=tri.restArea;
                areaSum+=tri.restArea;
                rest.shape[1]=AvbdNodalVec3(tri.dmInv00,tri.dmInv01,0);
                rest.shape[2]=AvbdNodalVec3(tri.dmInv10,tri.dmInv11,0);
                rest.shape[0]=-rest.shape[1]-rest.shape[2];
            }
            cache.meanRestArea=PxReal(areaSum/cache.nbParticles);
            for(PxU32 i=0;i<cache.nbBends;++i)
            {
                auto& rest=cache.nodalRest[cache.nbTriangles+i];rest=AvbdNodalShellRest();
                rest.kind=rest.Hinge;rest.length=cache.bends[i].restLength;
            }
        }
        cache.longitudinalAreaModulus=PxReal((lambda+2*mu)*cache.meanRestArea);
        if(bakeBending)
        {
            for(PxU32 i=0;i<cache.nbBends;++i)
            {
                auto& rest=cache.nodalRest[cache.nbTriangles+i];
                const PxReal theta=flatten?0:cache.bends[i].restShapeAngle;
                rest.restCos=PxCos(theta);rest.restSin=PxSin(theta);
            }
            cache.bakedFlattening=flatten;
        }
        if(!cache.membrane && bending==0)cache.membrane=PX_NEW(AvbdMembraneCache)();
        bool warm=false;
        if(cache.membrane)
        {
            auto& history=*cache.membrane;
            history.response.resize(3*cache.nbTriangles,PxVec3(0));
            warm=history.published && history.published+1==cache.liveStamp && history.dt==mDt &&
                history.mu==mu && history.lambda==lambda;
            history.dt=mDt;history.mu=mu;history.lambda=lambda;
            history.firstRow=mRows.size()+rowCount;history.published=0;
        }
        const PxU32 bends=bending>0?cache.nbBends:0;
        if(PxU64(mRows.size())+rowCount+cache.nbTriangles+bends>=PX_MAX_U32)
        {PxAtomicExchange(&mLocalDomainFailed,1);return;}
        const PxU32 count=cache.nbTriangles+bends;
        const PxU32 cacheIndex=PxU32(&cache-mIslandSurfaceCache.begin());
        const bool continuous=cache.packedStamp && cache.packedStamp+1==cache.liveStamp && cache.packedRows==count;
        const bool reuseSupports=continuous && cache.packedSupport==supportCount && cache.packedBlock==cache.blockStart;
        bool reuseMaterials=continuous && cache.packedMaterial==rowCount && rowCount+count<=mNodalShellMaterials.size();
        if(reuseMaterials)
        {
            const auto& first=mNodalShellMaterials[rowCount];
            reuseMaterials=first.rest==cache.nodalRest && first.mu==mu && first.lambda==lambda &&
                first.bending==bending && first.warm==warm &&
                first.previous==(cache.membrane?cache.membrane->response.begin():NULL);
        }
        mSurfaceBatches.pushBack({cacheIndex,mRows.size()+rowCount,supportCount,
            rowCount,count,mu,lambda,bending,warm,reuseSupports,reuseMaterials});
        if(!mNpContactRowCount)mSurfacePrefix.pushBack({cache.topologyRevision,cache.blockStart,count});
        rowCount+=count;
        supportCount+=3*cache.nbTriangles+4*bends;
    }
    if(!mNodalShellMaterials.resizeUninitialized(rowCount) || !mRows.appendMaterialSlots(rowCount))
    {PxAtomicExchange(&mLocalDomainFailed,1);return;}
    mNativeMaterialEnd=mRows.size();
}

void AvbdDynamicsContext::packSurfaceRows(PxU32 first,PxU32 end)
{
    // No append or allocation here. A lane owns these rows, their typed
    // payloads and their support slices until the common assembly barrier.
    PxU32 lo=0,hi=mSurfaceBatches.size();
    while(hi-lo>1)
    {
        const PxU32 mid=lo+(hi-lo)/2;
        if(mSurfaceBatches[mid].firstMaterial<=first)lo=mid;else hi=mid;
    }
    for(;first<end;++lo)
    {
        const SurfaceBatch& batch=mSurfaceBatches[lo];
        const auto& cache=mIslandSurfaceCache[batch.cacheIndex];
        const PxU32 stop=PxMin(end,batch.firstMaterial+batch.count);
        for(;first<stop;++first)
        {
            const PxU32 i=first-batch.firstMaterial;
            AvbdRow row={};
            row.kernelId=eAVBD_ROW_KERNEL_NODAL_SHELL;row.bodyA=row.bodyB=PX_MAX_U32;
            row.geometryIndex=first+1;
            const bool triangle=i<cache.nbTriangles;
            row.supportCount=triangle?3:4;
            row.supportStart=batch.firstSupport+(triangle?3*i:3*cache.nbTriangles+4*(i-cache.nbTriangles));
            if(!batch.reuseSupports)
            {
                PxU32 ids[4];
                if(triangle)
                {
                    const auto& tri=cache.triangles[i];
                    ids[0]=tri.p0;ids[1]=tri.p1;ids[2]=tri.p2;
                }
                else
                {
                    const auto& bend=cache.bends[i-cache.nbTriangles];
                    ids[0]=bend.opp0;ids[1]=bend.opp1;ids[2]=bend.edgeStart;ids[3]=bend.edgeEnd;
                }
                for(PxU32 s=0;s<row.supportCount;++s)mRowSupports[row.supportStart+s]={cache.blockStart+ids[s],1.f,PxVec3(0)};
            }
            mRows[batch.firstRow+i].header()=row;
            if(!batch.reuseMaterials)
                mNodalShellMaterials[first]={cache.nodalRest+i,batch.mu,batch.lambda,batch.bending,
                    cache.membrane && triangle?cache.membrane->response.begin()+3*i:NULL,batch.warm};
        }
    }
}

void AvbdDynamicsContext::compileNodalVolumeRows()
{
    AvbdBodyIndex rigidIndex;
    bool indexed = false;
    for(auto& cache:mIslandSoftCache)if(cache.coupled && cache.liveStamp==mIslandSoftLiveStamp && cache.volume)
    {
        auto& volume=*cache.volume;const auto& core=volume.getCore();
        if(!(core.cpuElasticityDamping>=0) ||
            core.cpuMaterialModel!=PxDeformableVolumeMaterialModel::eNEO_HOOKEAN)
        {
            std::fprintf(stderr,"[AVBD_PARTICLE3] status=UNSUPPORTED reason=volume_np_or_material_uncompiled\n");
            PxAtomicExchange(&mLocalDomainFailed,1);return;
        }
        const auto* mesh=volume.getSimulationMesh();
        const auto* auxiliary=static_cast<const Gu::DeformableVolumeAuxData*>(volume.getAuxData());
        const AvbdNodalReal E=core.cpuYoungs,nu=core.cpuPoissons;
        if(!mesh || !auxiliary || !auxiliary->mGridModelTetraRestPoses ||
            cache.nbTets!=mesh->getNbTetrahedrons() || !(E>0 && nu>-1 && nu<.5f))
        {PxAtomicExchange(&mLocalDomainFailed,1);return;}
        if(cache.nodalRest && (cache.nodalRestMesh!=mesh || cache.nodalRestAux!=auxiliary ||
            (core.dirtyFlags&PxDeformableVolumeDataFlag::eREST_POSITION_INVMASS)))PX_FREE(cache.nodalRest);
        if(!cache.nodalRest)
        {
            cache.nodalRest=static_cast<AvbdNodalVolumeElement*>(PX_ALLOC(sizeof(AvbdNodalVolumeElement)*cache.nbTets,"AvbdNodalVolumeRest"));
            if(!cache.nodalRest){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            for(PxU32 t=0;t<cache.nbTets;++t)
            {
                const PxMat33& inverse=auxiliary->mGridModelTetraRestPoses[t];
                auto& rest=cache.nodalRest[t];
                for(PxU32 j=0;j<3;++j)rest.shape[j+1]=AvbdNodalVec3(inverse[0][j],inverse[1][j],inverse[2][j]);
                rest.shape[0]=-rest.shape[1]-rest.shape[2]-rest.shape[3];
                const AvbdNodalReal det=rest.shape[1].dot(rest.shape[2].cross(rest.shape[3]));
                if(!(det!=0 && PxIsFinite(det))){PxAtomicExchange(&mLocalDomainFailed,1);return;}
                rest.inverseDet=det;
                rest.volume=AvbdNodalReal(1)/(AvbdNodalReal(6)*PxAbs(det));
                if(!avbdReadTetIndices(*mesh,t,rest.nodes)){PxAtomicExchange(&mLocalDomainFailed,1);return;}
                for(PxU32 node:rest.nodes)if(node>=cache.nbParticles){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            }
            cache.nodalRestMesh=mesh;cache.nodalRestAux=auxiliary;
        }
        const AvbdNodalReal mu=E/(2*(1+nu)),lambda=E*nu/((1+nu)*(1-2*nu));
        for(PxU32 t=0;t<cache.nbTets;++t)
        {
            const PxU32* nodes=cache.nodalRest[t].nodes;
            AvbdRow row={};row.bodyA=row.bodyB=PX_MAX_U32;row.kernelId=eAVBD_ROW_KERNEL_NODAL_VOLUME;
            row.supportStart=mRowSupports.size();row.supportCount=4;
            for(PxU32 j=0;j<4;++j)
            {
                mRowSupports.pushBack({cache.blockStart+nodes[j],1.f,PxVec3(0)});
            }
            AvbdNodalVolumeMaterial material={cache.nodalRest+t,mu,lambda};
            const auto& shape=material.rest->shape;
            const auto startEdge=[&](PxU32 j)
            {
                const auto e=cache.particles[nodes[j]].pStart-cache.particles[nodes[0]].pStart;
                return AvbdNodalVec3(e.x,e.y,e.z);
            };
            const AvbdNodalVec3 edges[3]={startEdge(1),startEdge(2),startEdge(3)};
            material.minimumJ=avbdNodalVolumeLowerBound(edges[0].dot(edges[1].cross(edges[2]))*material.rest->inverseDet);
            if(core.cpuElasticityDamping>0)
            {
                AvbdNodalMat33 F(PxZero);
                for(PxU32 j=1;j<4;++j)
                    F+=avbdNodalOuter(edges[j-1],shape[j]);
                material.metricIndex=mNodalVolumeStartMetrics.size();
                material.viscosity=AvbdNodalReal(core.cpuElasticityDamping)/mDt;
                mNodalVolumeStartMetrics.pushBack(F.getTranspose()*F);
            }
            mNodalVolumeMaterials.pushBack(material);
            row.geometryIndex=mNodalVolumeMaterials.size();mRows.pushBack(row);
        }
        for(auto& point : volume.rigidAttachments)
        {
            if(!indexed) { rigidIndex.build(mBodies); indexed=true; }
            const PxU32 rigid=rigidIndex.find(point.body);
            if(rigid>=mBodies.size() || point.element >= (point.tetrahedron?cache.nbTets:cache.nbParticles))
            { PxAtomicExchange(&mLocalDomainFailed,1); return; }
            const auto& body=mBodies[rigid];
            const PxVec3 arm=point.body->getBody2Actor().transformInv(point.actorPoint);
            const PxU32 first=mRowSupports.size(), count=point.tetrahedron?4u:1u;
            PxReal particleMobility=0;
            for(PxU32 i=0;i<count;++i)
            {
                const PxReal w=point.weights[i];
                if(w==0) continue;
                const PxU32 id=point.tetrahedron?cache.nodalRest[point.element].nodes[i]:point.element;
                mRowSupports.pushBack({cache.blockStart+id,w,PxVec3(0)});
                particleMobility+=w*w*cache.particles[id].invMass;
            }
            mRowSupports.pushBack({rigid,-1,arm});
            mAttachmentRows.pushBack(mRows.size());
            for(PxU32 axis=0;axis<3;++axis)
            {
                AvbdRowBuilder row={};
                row.bodyA=rigid; row.bodyB=PX_MAX_U32; row.kernelId=eAVBD_ROW_KERNEL_ATTACHMENT;
                row.supportStart=first; row.supportCount=PxU16(mRowSupports.size()-first);
                auto& law=row.scalar();
                law.n=PxVec3(0); law.n[axis]=1;
                const PxReal mobility=particleMobility+avbdEndpointAxisResponse(body,law.n,arm);
                if(!(mobility>0)) { PxAtomicExchange(&mLocalDomainFailed,1); return; }
                law.rho=1/(mDt*mDt*mobility);
                law.lambda=-point.force[axis];
                law.reactionOutput=&point.force[axis];
                law.minLambda=-PX_MAX_REAL; law.maxLambda=PX_MAX_REAL;
                mRows.pushBack(row);
            }
        }
    }
    mNativeMaterialEnd=mRows.size();
}

namespace
{
PX_FORCE_INLINE PxU32 gatherVolumePositions(const AvbdIslandSoftCache& cache,PxU32 block,
    const AvbdNodalVolumeElement& tet,AvbdNodalVec3 (&x)[4])
{
    const PxU32 local=block-cache.blockStart;
    PxU32 corner=PX_MAX_U32;
    for(PxU32 s=0;s<4;++s)
    {
        const PxU32 node=tet.nodes[s];
        if(node==local)corner=s;
        const PxVec3& p=cache.particles[node].p;
        x[s]=AvbdNodalVec3(p.x,p.y,p.z);
    }
    PX_ASSERT(corner<4);
    return corner;
}
}

// The assembled polynomial already accounts for all volume work. During a
// single-node line search the remaining question is the signed tetrahedron volume.
PX_INLINE bool AvbdDynamicsContext::sampleNodalCandidate(PxU32 ri,PxU32 block,const PxVec3& candidate,
    AvbdNodalReal& energy,PxU32 peer,const PxVec3* peerCandidate,AvbdNodalTrialVertices* vertices)
{
    const auto& row=mRows[ri].header();
    if(row.kernelId==eAVBD_ROW_KERNEL_NODAL_VOLUME)
    {
        const auto& material=mNodalVolumeMaterials[row.geometryIndex-1];
        const auto& cache=mIslandSoftCache[mBlocks[block].objectIndex];
        AvbdNodalVec3 x[4];
        const PxU32 corner=gatherVolumePositions(cache,block,*material.rest,x);
        x[corner]=AvbdNodalVec3(candidate.x,candidate.y,candidate.z);
        const auto J=(x[1]-x[0]).dot((x[2]-x[0]).cross(x[3]-x[0]))*material.rest->inverseDet;
        return PxIsFinite(J) && J>0 && J>=material.minimumJ;
    }
    const auto sample=sampleNodalRow(ri,block,&candidate,peer,peerCandidate,NULL,NULL,vertices);
    energy+=sample.energy;
    return sample.valid;
}

AvbdNodalShellSampleT<AvbdNodalReal> AvbdDynamicsContext::sampleNodalRow(
    PxU32 ri,PxU32 block,const PxVec3* candidate,PxU32 peer,const PxVec3* peerCandidate,AvbdNodalWorkT<AvbdNodalReal>* work,
    AvbdNodalPlaneRange* plane,AvbdNodalTrialVertices* vertices)
{
    const auto& row=mRows[ri].header();
    if(row.kernelId==eAVBD_ROW_KERNEL_ATTACHMENT)
    {
        const auto view=mRows[ri];
        const auto& law=view.scalar();
        PxReal c=avbdHeterogeneousRowC(view,mRowSupports,mBlocks,mBodies,contactGeometry());
        PxReal weight=0;
        for(PxU32 k=row.supportStart;k<row.supportStart+row.supportCount;++k)
        {
            const auto& s=mRowSupports[k];
            if(s.block==block) weight+=s.weight;
            if(peerCandidate && s.block==peer) c+=s.weight*law.n.dot(*peerCandidate-mBlocks[peer].particle->p);
        }
        if(candidate) c+=weight*law.n.dot(*candidate-mBlocks[block].particle->p);
        const AvbdNodalVec3 j=AvbdNodalVec3(law.n.x,law.n.y,law.n.z)*weight;
        AvbdNodalShellSampleT<AvbdNodalReal> result;
        result.valid=PxIsFinite(c);
        result.energy=c*(law.lambda+.5f*law.rho*c);
        result.gradient=j*(law.lambda+law.rho*c);
        result.metric=avbdNodalOuter(j,j)*law.rho;
        return result;
    }
    if(row.kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT)
    {
        const auto contact=mRows[ri];AvbdNodalShellSampleT<AvbdNodalReal> result;
        const auto node=[&](PxU32 b){return b==block && candidate?*candidate:
            (b==peer && peerCandidate?*peerCandidate:mBlocks[b].particle->p);};
        PxU32 visits=0;
        auto& domain=mNodalContacts[row.geometryIndex-1];
        auto* tetTrial=vertices && domain.separatedTets?&domain.localTet:NULL;
        if(tetTrial)
        {
            const PxU32 frozen=1-avbdNodalPairSide(domain,block);
            visits=candidate?4u:8u;
            result.valid=avbdTetsSeparated(domain,[&](PxU32 s,PxU32 v){return vertices->read(domain,s,v,node);},
                frozen,&tetTrial->oppositeMinimum,candidate!=NULL);
        }
        else result.valid=avbdNodalContactPath(domain,node,mBodies,mStep.countNodalSolves?&visits:NULL);
        if(!result.valid && !candidate && mStep.countNodalSolves)
        {
            const auto& d=mNodalContacts[row.geometryIndex-1];const auto& f=*d.feature;
            std::fprintf(stderr,"[AVBD_NODAL_DOMAIN] row=%u sides=%u,%u self=%u features=%u,%u vertexMask=%u separation=%.9g offset=%.9g axis=%.7g,%.7g,%.7g\n",
                ri,d.first[0],d.first[1],unsigned(d.geometry->deformable[0]==d.geometry->deformable[1]),f.feature[0],f.feature[1],
                unsigned(f.vertexMask),double(f.separation),double(d.restDistance),double(f.normal.x),double(f.normal.y),double(f.normal.z));
        }
        if(mStep.countNodalSolves){PxAtomicIncrement(&mStep.localPrimalQueries);PxAtomicAdd(&mStep.localPrimalPointVisits,PxI64(visits));}
        if(!result.valid)return result;
        PxReal weight=0,peerWeight=0;
        if(tetTrial && candidate)weight=tetTrial->weight;
        else for(PxU32 k=row.supportStart;k<row.supportStart+row.supportCount;++k)
        {
            const auto& support=mRowSupports[k];
            if(support.block==block)weight+=support.weight;
            if(peerCandidate && support.block==peer)peerWeight+=support.weight;
        }
        if(tetTrial && !candidate)tetTrial->weight=weight;
        if(plane && weight!=0)
        {
            const auto& c=mNodalContacts[row.geometryIndex-1];
            const PxU32 s=c.geometry->deformable[0]?0u:1u;
            const auto& shape=c.geometry->shapes[1-s];
            if((shape.kind==PxsContactShapeGeometry::ePLANE || shape.kind==PxsContactShapeGeometry::eBOX) &&
                (c.feature->vertexMask&(1u<<s)))
            {
                const auto x=avbdNodalCollisionPoint(c,s,c.feature->feature[s],node);
                const auto clearance=c.restDistance+shape.radius+c.geometry->shapes[s].radius;
                if(shape.kind==PxsContactShapeGeometry::ePLANE)
                {
                    const auto n=s?-c.feature->normal:c.feature->normal;
                    plane->include(AvbdNodalVec3(n.x,n.y,n.z),s?-weight:weight,n.dot(x-shape.shapeToWorld.p)-clearance);
                }
                else if(shape.kind==PxsContactShapeGeometry::eBOX)
                {
                    const auto box=c.first[1-s];
                    const auto pose=box==PX_MAX_U32?shape.shapeToWorld:PxTransform(mBodies[box].p,mBodies[box].q)*shape.shapeToBody;
                    const auto local=pose.transformInv(x);
                    const PxU32 face=c.feature->feature[1-s],axis=face/2,u=(axis+1)%3,v=(axis+2)%3;
                    if(axis<3 && PxAbs(local[u])<=shape.halfExtents[u] && PxAbs(local[v])<=shape.halfExtents[v])
                    {
                        PxVec3 faceAxis(0);faceAxis[axis]=(face&1)?1.f:-1.f;
                        const auto n=pose.q.rotate(faceAxis);
                        plane->include(AvbdNodalVec3(n.x,n.y,n.z),s?-weight:weight,
                            local[axis]*faceAxis[axis]-shape.halfExtents[axis]-clearance);
                    }
                }
            }
        }
        // A zero-weight corner still participates in the finite geometry
        // certificate. Its constant potential cancels from this local trial.
        if(weight==0 && peerWeight==0 && mNodalContacts[row.geometryIndex-1].separatedTets)return result;
        const auto frame=mNodalContacts[row.geometryIndex-1].rigidFrame;
        const bool carried=frame!=PX_MAX_U32;
        const auto relative=carried?avbdNodalContactSlip(contact,mRowSupports,mBlocks,mBodies,contactGeometry()):PxVec3(0);
        const auto currentSlip=[&](){return carried?relative:avbdNodalWitnessTravel(contact,mRowSupports,mBlocks,mBodies);};
        const PxReal initialGap=tetTrial && candidate?tetTrial->initialGap:carried?contact.scalar().C0+contact.scalar().n.dot(relative):
            avbdHeterogeneousRowC(contact,mRowSupports,mBlocks,mBodies,contactGeometry());
        if(tetTrial && !candidate)tetTrial->initialGap=initialGap;
        const PxVec3 delta=candidate?*candidate-mBlocks[block].particle->p:PxVec3(0);
        PxVec3 travel=delta*weight;
        if(peerCandidate)travel+=peerWeight*(*peerCandidate-mBlocks[peer].particle->p);
        const auto toWorld=[&](const PxVec3& axis){return carried?
            avbdNodalWorldAxis(mBodies[frame].qStart,mBodies[frame].q,axis):axis;};
        const auto normal=toWorld(contact.scalar().n);
        const PxReal gap=initialGap+normal.dot(travel);
        const auto potential=avbdNpContactPotential(contact,gap);result.energy=potential.energy;
        if(candidate && contact.scalar().mu>0)
        {
            // Freeze the normal load for this tangent trial. Integral of the
            // Coulomb disk projection: quadratic while sticking, linear on slip.
            const auto& law=contact.scalar();
            const PxReal limit=-law.mu*avbdNpContactPotential(contact,initialGap).gradient;
            if(limit>0 && law.rhoT0>0)
            {
                const PxVec3 slip=currentSlip();
                const PxVec3 f(law.lambdaT0+law.rhoT0*(law.t0.dot(slip)+(1-AVBD_ALPHA)*law.tangentC0),
                    law.lambdaT1+law.rhoT1*(law.t1.dot(slip)+(1-AVBD_ALPHA)*law.tangentC1),0);
                const PxVec3 ds(toWorld(law.t0).dot(travel),toWorld(law.t1).dot(travel),0);
                const PxReal a=f.magnitude(),b=(f+ds*law.rhoT0).magnitude();
                const PxReal lo=PxMin(a,b),hi=PxMax(a,b);
                if(hi<=limit)
                    result.energy+=f.dot(ds)+.5f*law.rhoT0*ds.magnitudeSquared();
                else
                {
                    const PxReal lengthChange=(2*f.dot(ds)+law.rhoT0*ds.magnitudeSquared())/(a+b);
                    const PxReal mean=lo>=limit?limit:limit-.5f*(limit-lo)*(limit-lo)/(hi-lo);
                    result.energy+=mean*lengthChange;
                }
            }
        }
        if(!candidate)
        {
            const auto n=normal*weight;const AvbdNodalVec3 j(n.x,n.y,n.z);
            result.gradient=j*potential.gradient;
            result.metric=AvbdNodalMat33(j*j.x,j*j.y,j*j.z)*potential.curvature;
            if(contact.scalar().mu>0 && potential.gradient<0)
            {
                const auto slip=currentSlip();
                const auto& law=contact.scalar();
                PxReal f0=law.lambdaT0+law.rhoT0*(law.t0.dot(slip)+(1-AVBD_ALPHA)*law.tangentC0);
                PxReal f1=law.lambdaT1+law.rhoT1*(law.t1.dot(slip)+(1-AVBD_ALPHA)*law.tangentC1);
                const PxReal scale=avbdProjectCoulomb(potential.gradient,contact.scalar().mu,f0,f1);
                const auto t0=toWorld(contact.scalar().t0)*weight,t1=toWorld(contact.scalar().t1)*weight;
                const AvbdNodalVec3 j0(t0.x,t0.y,t0.z),j1(t1.x,t1.y,t1.z);
                result.gradient+=j0*f0+j1*f1;
                if(mBlocks[block].particleKernel==eAVBD_PARTICLE_KERNEL_TET && scale<1.f && scale>0.f)
                {
                    const auto limit=AvbdNodalReal(-potential.gradient*contact.scalar().mu);
                    result.metric+=avbdNodalSlidingMetric(j0,j1,AvbdNodalReal(f0),AvbdNodalReal(f1),
                        limit,AvbdNodalReal(scale),AvbdNodalReal(contact.scalar().rhoT0));
                }
                else
                    result.metric+=avbdNodalOuter(j0,j0)*(scale*contact.scalar().rhoT0)+
                        avbdNodalOuter(j1,j1)*(scale*contact.scalar().rhoT1);
            }
        }
        return result;
    }
    AvbdNodalShellSampleT<AvbdNodalReal> result;
    if(row.kernelId==eAVBD_ROW_KERNEL_NODAL_VOLUME)
    {
        const auto& material=mNodalVolumeMaterials[row.geometryIndex-1];
        AvbdNodalVec3 x[4];
        const PxU32 corner=gatherVolumePositions(mIslandSoftCache[mBlocks[block].objectIndex],block,*material.rest,x);
        if(work)work->polynomial=true;
        AvbdNodalVolumeArgs args={material.rest,x,material.mu,material.lambda,
            corner,&result};
        if(material.metricIndex!=PX_MAX_U32)
        {
            args.startMetric=&mNodalVolumeStartMetrics[material.metricIndex];
            args.viscosity=material.viscosity;args.work=work;
        }
        mKernels.launch(AVBD_NODAL_VOLUME,&args);
        // Determinant is affine along this one-node path. The current state
        // was admitted; a strictly positive endpoint proves the entire segment.
        return result;
    }
    AvbdNodalVec3 x[4];PxU32 corner=PX_MAX_U32;
    for(PxU32 s=0;s<row.supportCount;++s)
    {
        const PxU32 b=mRowSupports[row.supportStart+s].block;
        if(b==block)corner=s;
        const PxVec3 p=(b==block && candidate)?*candidate:mBlocks[b].particle->p;
        x[s]=AvbdNodalVec3(p.x,p.y,p.z);
    }
    const auto& material=mNodalShellMaterials[row.geometryIndex-1];
    if(candidate && corner<3 && material.rest->kind==AvbdNodalShellRest::Membrane)
    {
        const auto endNormal=(x[1]-x[0]).cross(x[2]-x[0]);
        const PxVec3 old=mBlocks[block].particle->p;x[corner]=AvbdNodalVec3(old.x,old.y,old.z);
        result.valid=avbdNodalTrianglePath((x[1]-x[0]).cross(x[2]-x[0]),endNormal);
        return result;
    }
    AvbdNodalShellArgs args={material.rest,x,material.mu,material.lambda,material.bending,
        candidate?PX_MAX_U32:corner,&result,NULL,work};mKernels.launch(AVBD_NODAL_SHELL,&args);
    return result;
}

PX_INLINE bool AvbdDynamicsContext::isNodalHinge(PxU32 ri) const
{
    const auto& row=mRows[ri].header();
    return row.kernelId==eAVBD_ROW_KERNEL_NODAL_SHELL &&
        mNodalShellMaterials[row.geometryIndex-1].rest->kind==AvbdNodalShellRest::Hinge;
}

PxU32 AvbdDynamicsContext::gatherNodalMembranes(PxU32 block,PxU32 begin,PxU32 end,AvbdMembraneBatchArgs& args)
{
    args.count=0;
    for(PxU32 j=begin;j<end && args.count<AvbdMembraneBatchArgs::Width;++j)
    {
        const auto& row=mRows[mRowProgram.blockRows[j]].header();
        if(row.kernelId!=eAVBD_ROW_KERNEL_NODAL_SHELL || isNodalHinge(mRowProgram.blockRows[j]))break;
        const auto& material=mNodalShellMaterials[row.geometryIndex-1];
        const PxU32 lane=args.count++;
        args.rest[lane]=material.rest;args.mu[lane]=material.mu;args.lambda[lane]=material.lambda;
        args.corner[lane]=PX_MAX_U32;
        for(PxU32 s=0;s<3;++s)
        {
            const PxU32 b=mRowSupports[row.supportStart+s].block;
            const PxVec3& p=mBlocks[b].particle->p;
            args.positions[lane][s]=AvbdNodalVec3(p.x,p.y,p.z);
            if(b==block)args.corner[lane]=s;
        }
    }
    return args.count;
}

PxU32 AvbdDynamicsContext::gatherNodalHinges(PxU32 block,PxU32 begin,PxU32 end,
    const PxVec3* candidate,AvbdHingeBatchArgs& args)
{
    args.count=0;
    for(PxU32 j=begin;j<end && args.count<AvbdHingeBatchArgs::Width;++j)
    {
        const PxU32 ri=mRowProgram.blockRows[j];
        if(!isNodalHinge(ri))break;
        const auto& row=mRows[ri].header();
        const auto& material=mNodalShellMaterials[row.geometryIndex-1];
        const PxU32 lane=args.count++;
        args.rest[lane]=material.rest;args.bending[lane]=material.bending;
        args.corner[lane]=PX_MAX_U32;
        for(PxU32 s=0;s<4;++s)
        {
            const PxU32 b=mRowSupports[row.supportStart+s].block;
            const PxVec3& p=b==block && candidate?*candidate:mBlocks[b].particle->p;
            args.positions[lane][s]=AvbdNodalVec3(p.x,p.y,p.z);
            if(b==block && !candidate)args.corner[lane]=s;
        }
    }
    return args.count;
}

bool AvbdDynamicsContext::sampleNodalCandidates(PxU32 block,PxU32 begin,PxU32 end,
    const PxVec3& candidate,AvbdNodalReal& energy)
{
    AvbdNodalTrialVertices vertices;
    for(PxU32 j=begin;j<end;)
    {
        const PxU32 ri=mRowProgram.blockRows[j];
        if(isNodalHinge(ri))
        {
            AvbdHingeBatchArgs args;
            gatherNodalHinges(block,j,end,&candidate,args);
            mKernels.launch(AVBD_NODAL_HINGE_BATCH,&args);
            if(!args.output.valid)return false;
            energy+=args.output.energy;
            j+=args.count;
        }
        else
        {
            if(!sampleNodalCandidate(ri,block,candidate,energy,PX_MAX_U32,NULL,&vertices))return false;
            ++j;
        }
    }
    return true;
}

void AvbdDynamicsContext::assembleNodalRows(PxU32 block,PxU32 begin,PxU32 end,AvbdNodalAssembly& out)
{
    AvbdNodalTrialVertices vertices;
    // Seed the positive mass matrix and its residual once, including on an
    // empty star. Every other leaf contains only its own row contributions.
    if(begin==mRowProgram.blockOffsets[block])
    {
        const auto& p=*mBlocks[block].particle;const PxVec3 dx=p.p-p.pHat;
        const auto inertia=AvbdNodalReal(p.mass)/(AvbdNodalReal(mDt)*mDt);
        out.gradient=out.polynomialGradient=AvbdNodalVec3(dx.x,dx.y,dx.z)*inertia;
        out.metric=out.polynomialMetric=AvbdNodalMat33(PxIdentity)*inertia;
    }
    // A previously stalled node can still recover along a supporting face.
    const bool stalled=mPrimalChangedIteration[block]<mStep.iteration;
    for(PxU32 j=begin;j<end;++j)
    {
        const PxU32 ri=mRowProgram.blockRows[j];
        if(mRows[ri].header().kernelId==eAVBD_ROW_KERNEL_NODAL_SHELL)
        {
            if(isNodalHinge(ri))
            {
                AvbdHingeBatchArgs args;
                gatherNodalHinges(block,j,end,NULL,args);
                mKernels.launch(AVBD_NODAL_HINGE_BATCH,&args);
                const auto& sample=args.output;
                if(!sample.valid){PxAtomicExchange(&mLocalDomainFailed,1);return;}
                out.gradient+=sample.gradient;out.metric+=sample.metric;out.nonlinearEnergy+=sample.energy;
                j+=args.count-1;
                continue;
            }
            AvbdMembraneBatchArgs args;
            gatherNodalMembranes(block,j,end,args);
            mKernels.launch(AVBD_MEMBRANE_BATCH,&args);
            const auto& sample=args.output;
            if(!sample.valid){PxAtomicExchange(&mLocalDomainFailed,1);return;}
            out.gradient+=sample.gradient;out.metric+=sample.metric;
            out.polynomialGradient+=sample.gradient;out.polynomialMetric+=sample.polynomialMetric;
            out.cubic+=sample.cubic;out.quartic+=sample.quartic;
            j+=args.count-1;
            continue;
        }
        if(!avbdIsNodalMaterial(mRows[ri].header().kernelId) && mRows[ri].header().kernelId!=eAVBD_ROW_KERNEL_NODAL_CONTACT &&
           mRows[ri].header().kernelId!=eAVBD_ROW_KERNEL_ATTACHMENT)
        {PxAtomicExchange(&mLocalDomainFailed,1);return;}
        AvbdNodalWorkT<AvbdNodalReal> work;
        const auto sample=sampleNodalRow(ri,block,NULL,PX_MAX_U32,NULL,&work,stalled?&out.plane:NULL,&vertices);
        if(!sample.valid){PxAtomicExchange(&mLocalDomainFailed,1);return;}
        out.gradient+=sample.gradient;out.metric+=sample.metric;
        if(work.polynomial)
        {
            out.polynomialGradient+=sample.gradient;
            out.polynomialMetric+=sample.metric+AvbdNodalMat33(PxIdentity)*work.geometric;
            out.metric+=AvbdNodalMat33(PxIdentity)*PxMax(AvbdNodalReal(0),work.geometric);
            out.cubic+=work.cubic;out.quartic+=work.quartic;
        }
        else out.nonlinearEnergy+=sample.energy;
    }
}

void AvbdDynamicsContext::solveLocalParticleBlock(PxU32 block)
{
    if(!(mBlocks[block].particle->invMass>0))return;
    AvbdNodalAssembly rows;
    assembleNodalRows(block,mRowProgram.blockOffsets[block],mRowProgram.blockOffsets[block+1],rows);
    if(!mLocalDomainFailed)finishNodalAssembly(block,rows);
}

void AvbdDynamicsContext::finishNodalAssembly(PxU32 block,const AvbdNodalAssembly& rows)
{
    auto& particle=*mBlocks[block].particle;
    const PxVec3 initial=particle.p;
    AvbdNodalVec3 step(0);
    AvbdNodalSolveArgs args={&rows.metric,&rows.gradient,&step};
    args.range=&rows.plane;
    mKernels.launch(AVBD_NODAL_SOLVE3,&args);
    if(mStep.countNodalSolves)PxAtomicIncrement(&mStep.nodalSolves);
    if(!args.solved){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    if(!(rows.gradient.dot(step)<0))return;
    const PxU32 begin=mRowProgram.blockOffsets[block],end=mRowProgram.blockOffsets[block+1];
    // All terms fit in a single tile. Keep the direction and line search in
    // registers. Only oversized stars need resumable candidate storage.
    AvbdNodalReal fraction=1;
    for(PxU32 search=0;search<AvbdLocalWork::SearchLimit;++search,fraction*=AvbdNodalReal(.5))
    {
        const PxVec3 candidate=initial+PxVec3(PxReal(step.x*fraction),PxReal(step.y*fraction),PxReal(step.z*fraction));
        if(candidate==initial)return;
        const auto delta=candidate-initial;
        AvbdNodalReal energy=0;
        const bool valid=sampleNodalCandidates(block,begin,end,candidate,energy);
        if(valid && rows.decreases(AvbdNodalVec3(delta.x,delta.y,delta.z),energy))
        {
            particle.p=candidate;mPrimalSteps[block]=delta.magnitude();mPrimalChangedIteration[block]=mStep.iteration+1;
            return;
        }
    }
}

void AvbdDynamicsContext::beginNodalTrial(PxU32 block,const AvbdNodalAssembly& rows,AvbdNodalTrial& trial)
{
    trial.original=mBlocks[block].particle->p;
    AvbdNodalSolveArgs args={&rows.metric,&rows.gradient,&trial.direction};
    args.range=&rows.plane;
    mKernels.launch(AVBD_NODAL_SOLVE3,&args);
    if(mStep.countNodalSolves)PxAtomicIncrement(&mStep.nodalSolves);
    if(!args.solved){PxAtomicExchange(&mLocalDomainFailed,1);trial.done=true;return;}
    trial.done=!(rows.gradient.dot(trial.direction)<0);
    if(!trial.done)makeNodalCandidate(trial);
}

void AvbdDynamicsContext::makeNodalCandidate(AvbdNodalTrial& trial)
{
    const auto& step=trial.direction;const auto fraction=trial.fraction;
    trial.candidate=trial.original+PxVec3(PxReal(step.x*fraction),PxReal(step.y*fraction),PxReal(step.z*fraction));
    trial.done=trial.candidate==trial.original;
}

void AvbdDynamicsContext::sampleNodalTrial(PxU32 block,PxU32 begin,PxU32 end,const AvbdNodalTrial& trial,AvbdNodalEnergy& out)
{
    AvbdNodalReal energy=0;
    const bool valid=sampleNodalCandidates(block,begin,end,trial.candidate,energy);
    out.valid=valid;out.energy=energy;
}

void AvbdDynamicsContext::acceptNodalTrial(PxU32 block,const AvbdNodalAssembly& rows,AvbdNodalTrial& trial,const AvbdNodalEnergy& sample)
{
    const PxVec3 delta=trial.candidate-trial.original;
    const AvbdNodalVec3 accepted(delta.x,delta.y,delta.z);
    if(sample.valid && rows.decreases(accepted,sample.energy))
    {
        mBlocks[block].particle->p=trial.candidate;mPrimalSteps[block]=delta.magnitude();
        mPrimalChangedIteration[block]=mStep.iteration+1;trial.done=true;
    }
    else {trial.fraction*=AvbdNodalReal(.5);makeNodalCandidate(trial);}
}

void AvbdDynamicsContext::solveLocalParticlePair(PxU32 first,PxU32 second)
{
    using V=AvbdNodalVec3;using M=AvbdNodalMat33;using R=AvbdNodalReal;
    const auto v=[](const PxVec3& x){return V(x.x,x.y,x.z);};
    const PxU32 ids[2]={first,second};
    AvbdNodalPlaneRange planes[2];
    PxVec3 start[2];V g[2],qg[2],step[2];M h[2],qh[2],cross(PxZero);
    for(PxU32 i=0;i<2;++i)
    {
        const auto& p=*mBlocks[ids[i]].particle;start[i]=p.p;
        const R mass=p.mass/(R(mDt)*mDt);
        g[i]=qg[i]=v(p.p-p.pHat)*mass;h[i]=qh[i]=M(PxIdentity)*mass;
    }
    // Merge the two immutable sorted incidence lists. Every boundary term
    // occurs once, including rows whose force witness has a zero weight.
    const auto visit=[&](const auto& fn)
    {
        PxU32 a=mRowProgram.blockOffsets[first],b=mRowProgram.blockOffsets[second];
        const PxU32 ea=mRowProgram.blockOffsets[first+1],eb=mRowProgram.blockOffsets[second+1];
        while(a<ea || b<eb)
        {
            const PxU32 ra=a<ea?mRowProgram.blockRows[a]:PX_MAX_U32;
            const PxU32 rb=b<eb?mRowProgram.blockRows[b]:PX_MAX_U32;
            const PxU32 ri=PxMin(ra,rb);
            if(!fn(ri,ri==ra?0u:1u,ra==rb))return false;
            a+=ri==ra;b+=ri==rb;
        }
        return true;
    };
    R base=0,quartic[2]={};V cubic[2]={V(0),V(0)};
    const bool valid=visit([&](PxU32 ri,PxU32 side,bool shared)
    {
        const auto row=mRows[ri];const auto& head=row.header();
        R weights[2]={};
        if(head.kernelId==eAVBD_ROW_KERNEL_NODAL_CONTACT || head.kernelId==eAVBD_ROW_KERNEL_ATTACHMENT)
        {
            if(shared)
            {
                for(PxU32 k=head.supportStart;k<head.supportStart+head.supportCount;++k)
                    for(PxU32 j=0;j<2;++j)if(mRowSupports[k].block==ids[j])weights[j]+=mRowSupports[k].weight;
                side=PxAbs(weights[0])>=PxAbs(weights[1])?0u:1u;
            }
        }
        else if(shared || !avbdIsNodalMaterial(head.kernelId))return false;
        AvbdNodalWorkT<R> work;
        // A shared embedded witness with two nonzero weights constrains
        // their combined displacement. Only a one-node bound fits this chart.
        const bool oneNodeBound=weights[1-side]==0 && mPrimalChangedIteration[ids[side]]<mStep.iteration;
        const auto sample=sampleNodalRow(ri,ids[side],NULL,PX_MAX_U32,NULL,&work,oneNodeBound?&planes[side]:NULL);
        if(!sample.valid)return false;
        g[side]+=sample.gradient;h[side]+=sample.metric;
        if(work.polynomial)
        {
            qg[side]+=sample.gradient;qh[side]+=sample.metric+M(PxIdentity)*work.geometric;
            h[side]+=M(PxIdentity)*PxMax(R(0),work.geometric);cubic[side]+=work.cubic;quartic[side]+=work.quartic;
        }
        else base+=sample.energy;
        if(shared && weights[side]!=0)
        {
            const R ratio=weights[1-side]/weights[side];
            g[1-side]+=sample.gradient*ratio;h[1-side]+=sample.metric*(ratio*ratio);
            cross+=sample.metric*ratio;
        }
        return true;
    });
    if(!valid){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    AvbdNodalSolveArgs args={&h[0],&g[0],&step[0],false,&cross,&h[1],&g[1],&step[1]};
    args.range=planes;
    mKernels.launch(AVBD_NODAL_SOLVE3,&args);
    if(mStep.countNodalSolves)
    {PxAtomicAdd(&mStep.nodalSolves,2);PxAtomicIncrement(&mStep.nodalPairUpdates);}
    if(!args.solved || !step[0].isFinite()){PxAtomicExchange(&mLocalDomainFailed,1);return;}
    if(!(g[0].dot(step[0])+g[1].dot(step[1])<0))return;
    R fraction=1;
    for(PxU32 trial=0;trial<16;++trial,fraction*=R(.5))
    {
        PxVec3 candidate[2];V dx[2];R slope=0,work=0;
        for(PxU32 i=0;i<2;++i)
        {
            candidate[i]=start[i]+PxVec3(PxReal(step[i].x*fraction),PxReal(step[i].y*fraction),PxReal(step[i].z*fraction));
            dx[i]=v(candidate[i]-start[i]);slope+=g[i].dot(dx[i]);
            work+=dx[i].dot(qg[i]+qh[i]*dx[i]*R(.5));
            const R length2=dx[i].magnitudeSquared();work+=length2*(cubic[i].dot(dx[i])+quartic[i]*length2);
        }
        if(candidate[0]==start[0] && candidate[1]==start[1])return;
        if(!(slope<0))continue;
        R next=0;
        if(!visit([&](PxU32 ri,PxU32 side,bool shared)
        {
            return sampleNodalCandidate(ri,ids[side],candidate[side],next,ids[1-side],shared?&candidate[1-side]:NULL);
        }))continue;
        const R change=work+(next-base);
        if(change>R(1.e-4)*slope || !PxIsFinite(change))continue;
        for(PxU32 i=0;i<2;++i)
        {
            mBlocks[ids[i]].particle->p=candidate[i];mPrimalSteps[ids[i]]=(candidate[i]-start[i]).magnitude();
            mPrimalChangedIteration[ids[i]]=mStep.iteration+1;
        }
        return;
    }
}

void AvbdDynamicsContext::solvePrimalBlock(PxU32 blockIndex)
{
    if(blockIndex>=mBlocks.size()){solveIncidencePrimal(blockIndex-mBlocks.size());return;}
	const PxReal invH2 = mDt > 0.0f ? 1.0f / (mDt * mDt) : 3600.0f;
	const AvbdBlock &block = mBlocks[blockIndex];
	if (block.type == eAVBD_BLOCK_RIGID6)
	{
        if(block.objectIndex<mRigidGroups.size() && mRigidGroups[block.objectIndex].count>1)
        {
            if(mRigidGroups[block.objectIndex].leafCount){solveLoopGroup(block.objectIndex,invH2);return;}
            if(mStep.localBudget){solveLocalRigidGroup(block.objectIndex,invH2);return;}
            const AvbdRigidGroup& group=mRigidGroups[block.objectIndex];
            if(solvePrimalRigidGroup(block.objectIndex,invH2)) return;
            for(PxU32 s=0;s<group.count;++s)
            {
                const PxU32 b=group.blocks[s];const PxVec3 before=mBodies[b].p;const PxQuat rotation=mBodies[b].q;
                mPrimalSteps[b]=solvePrimalRigidBlock(b,invH2);
                if(mBodies[b].p!=before || !(mBodies[b].q==rotation)) mPrimalChangedIteration[b]=mStep.iteration+1;
            }
            return;
        }
		if (block.objectIndex < mBodies.size())
		{
			const PxVec3 before=mBodies[block.objectIndex].p;
			const PxQuat rotation=mBodies[block.objectIndex].q;
			const PxReal step = solvePrimalRigidBlock(block.objectIndex, invH2);
			mPrimalSteps[blockIndex] = step;
			if(mBodies[block.objectIndex].p!=before || !(mBodies[block.objectIndex].q==rotation))
				mPrimalChangedIteration[blockIndex]=mStep.iteration+1;
		}
		return;
	}
	if (block.type != eAVBD_BLOCK_PARTICLE3)
		return;
	if(nodalPartner(blockIndex) != PX_MAX_U32)
		solveLocalParticlePair(blockIndex, mNodalPartners[blockIndex]);
	else
		solveLocalParticleBlock(blockIndex);
}

PxReal AvbdDynamicsContext::finishPrimalSweep(bool *coupledDomainStationary)
{
	PxReal step = 0.0f;
	for(PxU32 i = 0; i < mPrimalSteps.size(); ++i)
		step = PxMax(step, mPrimalSteps[i]);
	*coupledDomainStationary = true;
	return step;
}

void AvbdDynamicsContext::prepareDeformableVelocities(PxReal dt)
{
	for(PxU32 i = 0; i < mIslandSoftCache.size(); ++i)
	{
		AvbdIslandSoftCache& cache = mIslandSoftCache[i];
		if(!cache.coupled || cache.liveStamp != mIslandSoftLiveStamp ||
			!cache.volume || !cache.particles)
			continue;
		avbdSoftWriteback(cache.particles, cache.nbParticles, dt);
		const DeformableVolumeCore& core = cache.volume->getCore();
		const PxReal damping = PxMax(
			core.linearDamping, 0.0f);
		const PxReal dampingScale = 1.0f / (1.0f + damping * dt);
		for(PxU32 particleIndex = 0; particleIndex < cache.nbParticles;
			++particleIndex)
			cache.particles[particleIndex].vel *= dampingScale;
	}
	for(PxU32 i = 0; i < mIslandSurfaceCache.size(); ++i)
	{
		AvbdIslandSurfaceCache& cache = mIslandSurfaceCache[i];
		if(cache.liveStamp != mIslandSurfaceLiveStamp || !cache.surface ||
			!cache.particles)
			continue;
		avbdSoftWriteback(cache.particles, cache.nbParticles, dt);
		const PxReal damping = PxMax(cache.surface->getCore().linearDamping, 0.0f);
		const PxReal dampingScale = 1.0f / (1.0f + damping * dt);
		for(PxU32 particleIndex = 0; particleIndex < cache.nbParticles;
			++particleIndex)
			cache.particles[particleIndex].vel *= dampingScale;
	}
}

void AvbdDynamicsContext::finalizeDeformableBlocks(PxReal dt)
{
    for(PxU32 i = 0; i < mIslandSoftCache.size(); ++i)
    {
        AvbdIslandSoftCache& cache = mIslandSoftCache[i];
        if(cache.coupled && cache.liveStamp == mIslandSoftLiveStamp)
        {
            publishIslandSoftCache(cache);
            cache.publishedDt=dt;cache.publishedStamp=cache.liveStamp;
        }
	}
	for(PxU32 cacheIndex = 0; cacheIndex < mIslandSurfaceCache.size(); ++cacheIndex)
	{
		AvbdIslandSurfaceCache& cache = mIslandSurfaceCache[cacheIndex];
		if(cache.liveStamp != mIslandSurfaceLiveStamp || !cache.surface ||
			!cache.particles)
			continue;
		cache.publishedDt=dt;cache.publishedStamp=cache.liveStamp;
		if(cache.membrane && mStep.incidencePrimal)cache.membrane->published=cache.liveStamp;
		DeformableSurfaceCore& core = cache.surface->getCore();
		if(!core.positionInvMass || !core.velocity)
			continue;
		for(PxU32 particleIndex = 0; particleIndex < cache.nbParticles;
			++particleIndex)
		{
			const AvbdSoftParticle& particle = cache.particles[particleIndex];
			core.positionInvMass[particleIndex] = PxVec4(particle.p,
				particle.invMass);
			core.velocity[particleIndex] = PxVec4(particle.vel, particle.invMass);
		}
		core.dirtyFlags = PxDeformableSurfaceDataFlags(
			PxDeformableSurfaceDataFlag::ePOSITION_INVMASS);
		core.dirtyFlags |= PxDeformableSurfaceDataFlag::eVELOCITY;
		core.cpuAvbdWakeRequested = false;
	}
	pruneIslandSoftCache();
	pruneIslandSurfaceCache();
}

void AvbdDynamicsContext::prepareIslandDeformableVolumes(PxReal dt, const PxVec3& gravity)
{
	const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();
	const PxU32 nbNodes = islandSim.getNbActiveNodes(IG::Node::eDEFORMABLE_VOLUME_TYPE);
	const PxNodeIndex* nodeIds = islandSim.getActiveNodes(IG::Node::eDEFORMABLE_VOLUME_TYPE);
	++mIslandSoftLiveStamp;
	if(mIslandSoftLiveStamp == 0)
		++mIslandSoftLiveStamp;

	for(PxU32 n = 0; n < nbNodes; ++n)
	{
		DeformableVolume* vol = static_cast<DeformableVolume*>(islandSim.getNode(nodeIds[n]).mObject);
		if(!vol)
			continue;
		PxTetrahedronMesh* simMesh = vol->getSimulationMesh();
		DeformableVolumeCore& core = vol->getCore();
		if(!simMesh || !core.simPositionInvMass || !core.simVelocity)
			continue;

		const PxU32 nbParticles = simMesh->getNbVertices();
		const PxU32 nbTetsIn = simMesh->getNbTetrahedrons();
		if(nbParticles == 0 || nbTetsIn == 0)
			continue;

		AvbdIslandSoftCache* cache = getIslandSoftCache(vol);
		if(cache && (cache->nbParticles != nbParticles || cache->nbTets != nbTetsIn))
        {
            avbdFreeIslandSoftBlocks(*cache);PxMemZero(cache,sizeof(*cache));
            cache->volume=vol;
        }
		if(!cache || !cache->particles)
        {
            AvbdIslandSoftCache fresh;PxMemZero(&fresh,sizeof(fresh));
            fresh.volume=vol;fresh.nbParticles=nbParticles;fresh.nbTets=nbTetsIn;
            fresh.particles=static_cast<AvbdSoftParticle*>(PX_ALLOC(sizeof(AvbdSoftParticle)*nbParticles,"AvbdNodalParticles"));
            if(!fresh.particles){PxAtomicExchange(&mLocalDomainFailed,1);continue;}
            PxMemZero(fresh.particles,sizeof(AvbdSoftParticle)*nbParticles);
            if(cache)*cache=fresh;
            else {mIslandSoftCache.pushBack(fresh);cache=&mIslandSoftCache.back();}
        }
        const bool continued=mStep.localBudget && cache->publishedStamp && cache->publishedStamp+1==mIslandSoftLiveStamp &&
            cache->publishedDt==dt;
        const PxVec3 gravityTravel=(core.actorFlags & PxActorFlag::eDISABLE_GRAVITY)?PxVec3(0):gravity*(dt*dt);
        const PxReal gravitySquared=gravityTravel.magnitudeSquared();
        cache->previousSupportNormal=continued && cache->planarSupport ? cache->supportNormal : PxVec3(0);
        cache->supportNormal=PxVec3(0);
        cache->supportFraction=1;
        cache->supportFriction=PX_MAX_F32;
        cache->maximumSpeedSquared=0;
        cache->planarSupport=mStep.localBudget;
		cache->liveStamp = mIslandSoftLiveStamp;

		for(PxU32 i = 0; i < nbParticles; ++i)
		{
			const PxVec4 pv = core.simPositionInvMass[i];
			const PxVec4 vv = core.simVelocity[i];
			AvbdSoftParticle& p = cache->particles[i];
            if(mStep.localBudget && pv.w>0)
                cache->maximumSpeedSquared=PxMax(cache->maximumSpeedSquared,vv.getXYZ().magnitudeSquared());
            p.supportedGravityTravel=PxVec3(0);
            if(continued && gravitySquared>0 && p.p==pv.getXYZ() && p.vel==vv.getXYZ() && p.invMass==pv.w)
            {
                const PxVec3 response=p.pHat-p.p;
                const PxReal norm=response.magnitudeSquared();
                if(norm>0)p.supportedGravityTravel=response*PxClamp(response.dot(gravityTravel)/norm,0.f,1.f);
            }
			p.p = pv.getXYZ();
			p.pHat = p.p;
			p.pStart = p.p;
			p.vel = vv.getXYZ();
			const PxReal invMass = pv.w;
			p.invMass = invMass;
			p.mass = (invMass > 1.0e-8f) ? (1.0f / invMass) : 0.0f;
		}

		cache->coupled = 1;

		PxVec3 g = gravity;
		if(core.actorFlags & PxActorFlag::eDISABLE_GRAVITY)
			g = PxVec3(0.0f);

		avbdSoftPredict(cache->particles, nbParticles, g, dt);
	}

	if(!anyCoupledDeformableVolume())
		pruneIslandSoftCache();
}

namespace
{
bool avbdBuildSurfaceEdgeTopology(AvbdIslandSurfaceCache& cache)
{
	cache.edges = reinterpret_cast<AvbdSoftBoundaryEdgeTopology*>(PX_ALLOC(
		sizeof(AvbdSoftBoundaryEdgeTopology) * 3 * cache.nbTriangles,
		"AvbdIslandSurfaceCache::edges"));
	if(!cache.edges) return false;
	cache.nbEdges = 0;
	for(PxU32 triangleIndex = 0; triangleIndex < cache.nbTriangles; ++triangleIndex)
	{
		const AvbdSoftTri& triangle = cache.triangles[triangleIndex];
		const PxU32 vertices[3] = {triangle.p0, triangle.p1, triangle.p2};
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			PxU32 v0 = vertices[corner], v1 = vertices[(corner + 1) % 3];
			if(v0 > v1) PxSwap(v0, v1);
			PxU32 edgeIndex = PX_MAX_U32;
			for(PxU32 existing = 0; existing < cache.nbEdges; ++existing)
				if(cache.edges[existing].v0 == v0 && cache.edges[existing].v1 == v1)
				{ edgeIndex = existing; break; }
			if(edgeIndex == PX_MAX_U32)
			{
				AvbdSoftBoundaryEdgeTopology& edge = cache.edges[cache.nbEdges++];
				edge.v0 = v0; edge.v1 = v1;
				edge.face0 = triangleIndex; edge.face1 = PX_MAX_U32;
			}
			else if(cache.edges[edgeIndex].face1 == PX_MAX_U32)
				cache.edges[edgeIndex].face1 = triangleIndex;
		}
	}
	cache.bends = reinterpret_cast<AvbdSoftBending*>(PX_ALLOC(
		sizeof(AvbdSoftBending) * cache.nbEdges,
		"AvbdIslandSurfaceCache::bends"));
	if(!cache.bends) return false;
	cache.nbBends = 0;
	for(PxU32 edgeIndex = 0; edgeIndex < cache.nbEdges; ++edgeIndex)
	{
		const AvbdSoftBoundaryEdgeTopology& edge = cache.edges[edgeIndex];
		if(edge.face1 == PX_MAX_U32) continue;
		const AvbdSoftTri& face0 = cache.triangles[edge.face0];
		const AvbdSoftTri& face1 = cache.triangles[edge.face1];
		const PxU32 face0Vertices[3] = {face0.p0, face0.p1, face0.p2};
		const PxU32 face1Vertices[3] = {face1.p0, face1.p1, face1.p2};
		PxU32 opposite0 = PX_MAX_U32, opposite1 = PX_MAX_U32;
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			if(face0Vertices[corner] != edge.v0 && face0Vertices[corner] != edge.v1)
				opposite0 = face0Vertices[corner];
			if(face1Vertices[corner] != edge.v0 && face1Vertices[corner] != edge.v1)
				opposite1 = face1Vertices[corner];
		}
		if(opposite0 == PX_MAX_U32 || opposite1 == PX_MAX_U32) continue;
		AvbdSoftBending& bend = cache.bends[cache.nbBends++];
		bend.opp0 = opposite0; bend.opp1 = opposite1;
		bend.edgeStart = edge.v0; bend.edgeEnd = edge.v1;
		bend.restShapeAngle = avbdSoftDihedralAngle(
			cache.particles[opposite0].p, cache.particles[opposite1].p,
			cache.particles[edge.v0].p, cache.particles[edge.v1].p);
		bend.restLength = (cache.particles[edge.v1].p -
			cache.particles[edge.v0].p).magnitude();
	}
	return cache.nbEdges > 0;
}

} // namespace

void AvbdDynamicsContext::prepareIslandDeformableSurfaces(PxReal dt, const PxVec3& gravity)
{
	const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();
	const PxU32 nbNodes = islandSim.getNbActiveNodes(IG::Node::eDEFORMABLE_SURFACE_TYPE);
	const PxNodeIndex* nodeIds = islandSim.getActiveNodes(IG::Node::eDEFORMABLE_SURFACE_TYPE);
	++mIslandSurfaceLiveStamp;
	if(mIslandSurfaceLiveStamp == 0)
		++mIslandSurfaceLiveStamp;
	for(PxU32 n = 0; n < nbNodes; ++n)
	{
		DeformableSurface* surface = static_cast<DeformableSurface*>(
			islandSim.getNode(nodeIds[n]).mObject);
		if(!surface)
			continue;
		PxTriangleMesh* mesh = surface->getTriangleMesh();
		DeformableSurfaceCore& core = surface->getCore();
		if(!mesh || !core.positionInvMass || !core.velocity || !core.restPosition)
			continue;
		const PxU32 nbParticles = mesh->getNbVertices();
		const PxU32 nbTrianglesIn = mesh->getNbTriangles();
		if(nbParticles == 0 || nbTrianglesIn == 0)
			continue;

		AvbdIslandSurfaceCache* cache = getIslandSurfaceCache(surface);
		if(cache && (cache->nbParticles != nbParticles || cache->nbTriangles > nbTrianglesIn || cache->restMesh!=mesh ||
            (core.dirtyFlags & PxDeformableSurfaceDataFlag::eREST_POSITION)))
		{
			cache->liveStamp = 0;
			pruneIslandSurfaceCache();
			cache = NULL;
		}
		if(!cache)
		{
			AvbdIslandSurfaceCache fresh;
			PxMemZero(&fresh, sizeof(fresh));
			fresh.surface = surface;
            fresh.topologyRevision=++mSurfaceRevision;
			fresh.restMesh = mesh;
			fresh.nbParticles = nbParticles;
			fresh.triangles = reinterpret_cast<AvbdSoftTri*>(PX_ALLOC(
				sizeof(AvbdSoftTri) * nbTrianglesIn, "AvbdIslandSurfaceCache::triangles"));
			fresh.particles = reinterpret_cast<AvbdSoftParticle*>(PX_ALLOC(
				sizeof(AvbdSoftParticle) * nbParticles, "AvbdIslandSurfaceCache::particles"));
			if(!fresh.triangles || !fresh.particles)
			{
				PX_FREE(fresh.triangles);
				PX_FREE(fresh.particles);
				PxAtomicExchange(&mLocalDomainFailed, 1);
				continue;
			}
			for(PxU32 i = 0; i < nbParticles; ++i)
			{
				AvbdSoftParticle& p = fresh.particles[i];
				PxMemZero(&p, sizeof(p));
				p.p = core.restPosition[i].getXYZ();
				p.pHat = p.pStart = p.p;
			}
			for(PxU32 t = 0; t < nbTrianglesIn; ++t)
			{
				PxU32 idx[3];
				if(!avbdReadTriIndices(*mesh, t, idx) || idx[0] >= nbParticles ||
					idx[1] >= nbParticles || idx[2] >= nbParticles)
					continue;
				if(avbdSoftBuildTri(fresh.triangles[fresh.nbTriangles], idx[0],
					idx[1], idx[2], fresh.particles))
					++fresh.nbTriangles;
			}
			if(fresh.nbTriangles == 0)
			{
				PX_FREE(fresh.triangles);
				PX_FREE(fresh.particles);
				PxAtomicExchange(&mLocalDomainFailed, 1);
				continue;
			}
			if(!avbdBuildSurfaceEdgeTopology(fresh))
			{
				PX_FREE(fresh.triangles);
				PX_FREE(fresh.particles);
				PX_FREE(fresh.edges);
				PX_FREE(fresh.bends);
				PxAtomicExchange(&mLocalDomainFailed, 1);
				continue;
			}
			mIslandSurfaceCache.pushBack(fresh);
			cache = &mIslandSurfaceCache.back();
		}
        const bool continued=mStep.localBudget && cache->publishedStamp && cache->publishedStamp+1==mIslandSurfaceLiveStamp &&
            cache->publishedDt==dt;
        const PxVec3 gravityTravel=(core.actorFlags & PxActorFlag::eDISABLE_GRAVITY)?PxVec3(0):gravity*(dt*dt);
        const PxReal gravitySquared=gravityTravel.magnitudeSquared();
		cache->liveStamp = mIslandSurfaceLiveStamp;

		for(PxU32 i = 0; i < nbParticles; ++i)
		{
			const PxVec4 pv = core.positionInvMass[i];
			AvbdSoftParticle& p = cache->particles[i];
            p.supportedGravityTravel=continued && gravitySquared>0 && p.p==pv.getXYZ() &&
                p.vel==core.velocity[i].getXYZ() && p.invMass==pv.w ?
                gravityTravel*PxClamp((p.pHat-p.p).dot(gravityTravel)/gravitySquared,0.f,1.f) : PxVec3(0);
            if(cache->membrane && cache->membrane->published &&
                (p.p!=pv.getXYZ() || p.vel!=core.velocity[i].getXYZ() || p.invMass!=pv.w))
                cache->membrane->published=0;
			p.p = p.pStart = p.pHat = pv.getXYZ();
			p.vel = core.velocity[i].getXYZ();
			p.invMass = pv.w;
			p.mass = pv.w > 1.0e-8f ? 1.0f / pv.w : 0.0f;
		}
		PxVec3 g = (core.actorFlags & PxActorFlag::eDISABLE_GRAVITY) ? PxVec3(0.0f) : gravity;
		avbdSoftPredict(cache->particles, nbParticles, g, dt);
	}
	if(!anyLiveDeformableSurface())
		pruneIslandSurfaceCache();
}

namespace
{
AvbdDynamicsContext* asAvbd(Context* ctx)
{
	if(!ctx || ctx->getSolverType() != PxSolverType::eAVBD)
		return NULL;
	return static_cast<AvbdDynamicsContext*>(ctx);
}
} // namespace

bool Dy::avbdKernelTableInfo(Context* ctx, PxU32& deviceKind, bool& ldlt6Bound, bool& ldlt3Bound)
{
	AvbdDynamicsContext* avbd = asAvbd(ctx);
	if(!avbd)
		return false;
	const KernelTable& table = avbd->getKernelTable();
	deviceKind = PxU32(table.device);
	ldlt6Bound = table.isBound(AVBD_LDLT6);
	ldlt3Bound = table.isBound(AVBD_LDLT3);
	return true;
}

Context* Dy::createAVBDDynamicsContext(PxcNpMemBlockPool* memBlockPool,
	Cm::FlushPool& taskPool, PxvSimStats& simStats,
	Cm::VirtualAllocatorCallback& allocator, PxsMaterialManager* materialManager,
	IG::SimpleIslandManager& islandManager, PxU64 contextID,
	PxReal maxBiasCoefficient, PxReal lengthScale, PxSceneFlags sceneFlags)
{
	return PX_NEW(AvbdDynamicsContext)(memBlockPool, taskPool, simStats, allocator,
		materialManager, islandManager, contextID, maxBiasCoefficient, lengthScale,
		sceneFlags);
}

void AvbdDynamicsContext::mergeResults()
{
}

PxU32 AvbdDynamicsContext::findBody(const PxsBodyCore* core) const
{
	if(!core)
		return PX_MAX_U32;
	for(PxU32 i = 0; i < mBodies.size(); ++i)
	{
		if(mBodies[i].core == core)
			return i;
	}
	return PX_MAX_U32;
}

PxTransform AvbdDynamicsContext::bodyPose(PxU32 index, const PxsBodyCore* fallback) const
{
	if(index < mBodies.size())
		return PxTransform(mBodies[index].p, mBodies[index].q);
	if(fallback)
		return fallback->body2World;
	return PxTransform(PxIdentity);
}

PxU32 AvbdDynamicsContext::solverBodyOf(PxU8 indexType, PxU32 solverBody) const
{
	if(indexType != PxsIndexedInteraction::eBODY)
		return PX_MAX_U32;
	if(solverBody >= mBodies.size())
		return PX_MAX_U32;
	return solverBody;
}

PxU32 AvbdDynamicsContext::contactBody(const PxsIndexedContactManager& cm, PxU32 side,const AvbdBodyIndex& indices) const
{
	if(cm.contactManager)
	{
		const PxcNpWorkUnit& unit = cm.contactManager->getWorkUnit();
		const PxsRigidCore* rigidCore = side ? unit.mRigidCore1 : unit.mRigidCore0;
		const PxU32 body = indices.find(static_cast<const PxsBodyCore*>(rigidCore));
		if(body != PX_MAX_U32)
			return body;
	}
	const PxU8 type = side ? cm.indexType1 : cm.indexType0;
	if(type == PxsIndexedInteraction::eBODY)
		return solverBodyOf(type, side ? PxU32(cm.solverBody1) : PxU32(cm.solverBody0));
	if(type == PxsIndexedInteraction::eARTICULATION && cm.contactManager)
	{
		const PxcNpWorkUnit& unit = cm.contactManager->getWorkUnit();
		const PxsRigidCore* rc = side ? unit.mRigidCore1 : unit.mRigidCore0;
		return indices.find(static_cast<const PxsBodyCore*>(rc));
	}
	return PX_MAX_U32;
}

void AvbdDynamicsContext::initSolverBody(AvbdSolverBody& b, PxsBodyCore& core,
	PxsRigidBody* rigid, const PxVec3& gravity, PxReal dt)
{
	b = AvbdSolverBody();
	b.core = &core;
	b.rigid = rigid;
	b.p = core.body2World.p;
	b.q = core.body2World.q;
	b.pStart = b.p;
	b.qStart = b.q;
	b.pLin = b.p;
	b.qLin = b.q;
	b.linVel = core.linearVelocity;
	b.angVel = core.angularVelocity;
	b.outputLinVel = b.linVel;
	b.outputAngVel = b.angVel;
	b.invMass = core.inverseMass;
	b.invInertia = core.inverseInertia;
	b.disableGravity = core.disableGravity;
	b.isKinematic = core.isKinematic() ? 1 : 0;
	b.hasStaticTouch = 0;
	b.hasContact = 0;
	b.artic = NULL;
	b.linkIndex = 0;
	if(core.fixedBaseLink)
	{
		b.invMass = 0.0f;
		b.isKinematic = 1;
		b.pHat = b.p;
		b.qHat = b.q;
		return;
	}
	if(b.isKinematic)
	{
		// Host supplies the prescribed motion. Keep the DCD pose for anchors
		// and C0; the unified primal consumes its end pose after row prep.
		b.invMass = 0.0f;
		b.invInertia = PxVec3(0.0f);
		b.pHat = b.p + b.linVel * dt;
		const PxReal angle = b.angVel.magnitude() * dt;
		b.qHat = angle > 0.0f ? PxQuat(angle, b.angVel.getNormalized()) * b.q : b.q;
		b.qHat.normalize();
		return;
	}
	if(b.invMass > 0.0f && dt > 0.0f)
	{
		const PxReal invH2 = 1.0f / (dt * dt);
		b.massInvH2 = invH2 / b.invMass;
		PxMat33 localInertia(PxZero);
		localInertia[0][0] = b.invInertia.x > 0.0f ?
			1.0f / b.invInertia.x : 0.0f;
		localInertia[1][1] = b.invInertia.y > 0.0f ?
			1.0f / b.invInertia.y : 0.0f;
		localInertia[2][2] = b.invInertia.z > 0.0f ?
			1.0f / b.invInertia.z : 0.0f;
		const PxMat33 startRotation(b.qStart);
		b.inertiaWorldInvH2 =
			(startRotation * localInertia * startRotation.getTranspose()) * invH2;
	}
	PxVec3 v = b.linVel;
	PxVec3 w = b.angVel;
	const PxReal accelScale = rigid ? rigid->mAccelScale : 1.0f;
	// The inertial target is the undamped BDF1 objective.  Authored damping
	// belongs to the final velocity boundary after pose reconstruction; putting
	// it here as well changes both the objective and the published velocity and
	// therefore damps every constrained mode twice.  Velocity caps likewise
	// apply to the final velocity, not to the position objective.
	bodyCoreComputeUnconstrainedVelocity(gravity, dt, 0.0f, 0.0f,
		accelScale, PX_MAX_F32, PX_MAX_F32, v, w, b.disableGravity != 0);
	b.pHat = b.p + v * dt;
	const PxVec3 halfW = w * (0.5f * dt);
	PxQuat qHat = b.q + PxQuat(halfW.x, halfW.y, halfW.z, 0.0f) * b.q;
	qHat.normalize();
	b.qHat = qHat;
	// Material reconstruction shares the full external-acceleration epoch.
	// Authored damping and velocity caps apply once at final publication.
	b.linVel = v;
	b.angVel = w;
}

void AvbdDynamicsContext::collectArticulationLinks(PxReal dt, const PxVec3& gravity)
{
	const PxU32 nbArtics = mArticulationArray.size();
	PxU32 extra = 0;
	for(PxU32 a = 0; a < nbArtics; ++a)
	{
		if(mArticulationArray[a])
			extra += mArticulationArray[a]->getArticulationData().getLinkCount();
	}
	mBodies.reserve(mBodies.size() + extra);
	for(PxU32 a = 0; a < nbArtics; ++a)
	{
		FeatherstoneArticulation* artic = mArticulationArray[a];
		if(!artic)
			continue;
		ArticulationData& artData = artic->getArticulationData();
		const PxU32 linkCount = artData.getLinkCount();
		ArticulationLink* links = artData.getLinks();
		for(PxU32 i = 0; i < linkCount; ++i)
		{
			if(!links[i].bodyCore)
				continue;
			AvbdSolverBody b;
			initSolverBody(b, *links[i].bodyCore, NULL, gravity, dt);
			b.artic = artic;
			b.linkIndex = i;
			mBodies.pushBack(b);
		}
	}
	buildArticJointSources();
}

void AvbdDynamicsContext::update(Cm::FlushPool& /*flushPool*/,
	PxBaseTask* continuation, PxBaseTask* /*postPartitioningTask*/,
	PxBaseTask* /*lostTouchTask*/, PxvNphaseImplementationContext* nphase,
	PxU32 /*maxPatchesPerCM*/, PxU32 maxArticulationLinks, PxReal dt,
	const PxVec3& gravity, Cm::PinnableBitMap& /*changedHandleMap*/)
{
	mStep=StepState();
    mSurfaceBatches.clear();mSurfacePrefix.clear();
    mFixedReactions.clear();
    mReleasedRigidBodies.clear();
    mAttachmentRows.clear();
    PX_DELETE(mWideNomination);mWideNomination=NULL;
    mNodalPartners.clear();
    mIncidenceDisplacement.clear();
    mIncidenceRotation.clear();
    mNativeMaterialEnd=0;mNodalVolumeMaterials.clear();mNodalContacts.clear();mNodalImpactRows.clear();
    mNodalVolumeStartMetrics.clear();
    mLocalTranslationClosed.clear();
    const IG::IslandSim& activeIsland = mIslandManager.getAccurateIslandSim();
    const bool nativeParticles = activeIsland.getNbActiveNodes(IG::Node::eDEFORMABLE_VOLUME_TYPE) ||
        activeIsland.getNbActiveNodes(IG::Node::eDEFORMABLE_SURFACE_TYPE);
    const char* program = std::getenv("PHYSX_AVBD_COMMON_STEP");
    mStep.commonProgram = nativeParticles || program != NULL;
    mStep.boundedCommon = !nativeParticles && program && std::strcmp(program, "bounded") == 0;
    if(nativeParticles || (program && std::strcmp(program, "budget") == 0))
    {
        mStep.localBudget = nativeParticles ? 4 : 32;
        if(const char* value=std::getenv("PHYSX_AVBD_LOCAL_ITERATIONS"))
        {
            char* end=NULL;const unsigned long n=std::strtoul(value,&end,10);
            if(value[0]<'0' || value[0]>'9' || end==value || *end || !n || n>128)
            {
                std::fprintf(stderr,"[AVBD_LOCAL_BUDGET] status=UNSUPPORTED reason=invalid_iteration_budget\n");
                PxGetFoundation().error(PxErrorCode::eINVALID_PARAMETER,PX_FL,
                    "AVBD local iteration budget must be an integer in [1,128].");return;
            }
            mStep.localBudget=PxU32(n);
        }
    }
    mStep.profile=(mStep.boundedCommon && std::getenv("PHYSX_AVBD_BOUNDED_PROFILE")!=NULL) ||
        (mStep.localBudget && std::getenv("PHYSX_AVBD_LOCAL_PROFILE")!=NULL);
    mStep.localTimeline=mStep.profile && std::getenv("PHYSX_AVBD_LOCAL_TIMELINE")!=NULL;
    mStep.countNodalSolves=mStep.profile || std::getenv("PHYSX_AVBD_COMMON_STATS")!=NULL;
    if(mStep.profile)mStep.backendStart=PxTime::getCurrentCounterValue();
    PxU64 prepareStart=mStep.backendStart;
    const auto markPrepare=[&](PxU32 part)
    {if(mStep.profile){const PxU64 now=PxTime::getCurrentCounterValue();mStep.prepareTicks[part]+=now-prepareStart;prepareStart=now;}};
	mActivePrimalTasks=0;mPeakPrimalTasks=0;
	const bool hasIslandWork = updateShared(nphase, dt, gravity);

	PX_PROFILE_ZONE("AVBD.update", mContextID);

	mLocalDomainFailed = false;
	mGravity = gravity;

	// All deformables enter through their Host actors and active IG nodes. Their
	// Particle3 blocks are compiled below beside Rigid6; no second integrator is
	// permitted before or after this backend program.
	prepareIslandDeformableVolumes(dt, gravity);
	prepareIslandDeformableSurfaces(dt, gravity);
    const bool hasSurface=anyLiveDeformableSurface();
    if(!hasSurface)mNodalShellMaterials.clear();
	mStep.hasParticleBlocks = anyCoupledDeformableVolume() || hasSurface;

	if(!hasIslandWork && !mStep.hasParticleBlocks)
		return;

	const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();
	mKinematicCount = islandSim.getNbActiveKinematics();
	const PxU32 islandCount = islandSim.getNbActiveIslands();
	const IG::IslandId* islandIds = islandSim.getActiveIslands();
	const PxU32 bodyCount = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
	const PxU32 articCount = islandSim.getNbActiveNodes(IG::Node::eARTICULATION_TYPE);
	const PxU32 contactCount = islandSim.getNbActiveEdges(IG::Edge::eCONTACT_MANAGER);

	PxU32 rigidPositionBudget = 1;
	if(bodyCount == 0 && articCount == 0 && !mStep.hasParticleBlocks)
		return;

	if(bodyCount > 0 || articCount > 0)
	{
		PxsIslandIndices counts;
		counts.bodies = bodyCount;
		counts.articulations = articCount;
		counts.contactManagers = contactCount;
		counts.constraints = islandSim.getNbActiveEdges(IG::Edge::eCONSTRAINT);

		reserveSharedSolverConstraintsArrays(islandSim, maxArticulationLinks);
		mSolverBodyRemapTable.forceSize_Unsafe(0);
		mSolverBodyRemapTable.resize(bodyCount);

		iterateIslandsNodes(counts, islandCount, islandIds, mBodyCoreArray.begin(),
			mRigidBodyArray.begin(), mArticulationArray.begin(),
			mSolverBodyRemapTable.begin(), mNodeIndexArray.begin());
		const PxU32 nbCms = iterateIslandsContactEdges(counts, islandCount, islandIds,
			mContactList.begin(), mSolverBodyRemapTable.begin());
		mContactList.forceSize_Unsafe(nbCms);

		mBodies.clear();
		mBodies.reserve(bodyCount + 4);
		mBodies.resize(bodyCount);

		const PxReal h = dt;
		for(PxU32 i = 0; i < bodyCount; ++i)
		{
			mBodies[i] = AvbdSolverBody();
			if(!mBodyCoreArray[i] || !mRigidBodyArray[i])
				continue;
			initSolverBody(mBodies[i], *mBodyCoreArray[i], mRigidBodyArray[i], gravity, h);
			if(avbdBodyResponds(mBodies[i]))
				rigidPositionBudget = PxMax(rigidPositionBudget, PxU32(mBodyCoreArray[i]->solverIterationCounts & 0xff));
		}
		// Active kinematics are not dynamic island nodes. Retain them as
		// read-only endpoints so NP and solverPrep do not freeze world anchors.
		const PxNodeIndex* kinematics = islandSim.getActiveKinematics();
		for(PxU32 i = 0; i < mKinematicCount; ++i)
		{
			PxsRigidBody* rigid = getRigidBodyFromIG(islandSim, kinematics[i]);
			if(!rigid || findBody(&rigid->getCore()) != PX_MAX_U32) continue;
			AvbdSolverBody boundary;
			initSolverBody(boundary, rigid->getCore(), rigid, gravity, h);
			mBodies.pushBack(boundary);
		}

		collectArticulationLinks(h, gravity);

	}
	else
	{
		mBodies.clear();
		mContactList.forceSize_Unsafe(0);
	}

    markPrepare(0);
	// Compile the current island's non-contact program before contact rows.
	// Contact tangent participation is an objective-ownership decision: the
	// AVBD_2 joint path assigns Position-AL the geometry normal only, whereas a
	// contact-only island also admits the Coulomb position coordinates.
	buildJointSources(islandCount, islandIds);
    markPrepare(1);
	const PxU64 contactStart=mStep.profile?PxTime::getCurrentCounterValue():0;
	buildContactRows();
    if(mStep.profile)mStep.contactPackTicks+=PxTime::getCurrentCounterValue()-contactStart;
    markPrepare(2);
	mNpContactRowCount = mRows.size();
    mTetrahedronMaterials.clear();
    mElasticFrameMaterials.beginStep(mDt);


	compileHeterogeneousBlocks();
    // Keep immutable shell supports at the start of the shared table. Host
    // contacts, bilateral attachments and joints append only to its suffix.
    if(hasSurface)
    {
        PxU64 surfaceSupports=0;
        for(const auto& cache:mIslandSurfaceCache)if(cache.liveStamp==mIslandSurfaceLiveStamp && cache.surface)
            surfaceSupports+=3ull*cache.nbTriangles+(cache.surface->getCore().cpuBendingStiffness>0?4ull*cache.nbBends:0);
        if(surfaceSupports>PX_MAX_U32 || !mRowSupports.resize(PxU32(surfaceSupports),{0,1.f,PxVec3(0)}))
        {mLocalDomainFailed=1;finishStep();return;}
    }
    else mRowSupports.clear();
	// Point anchors persist; spring damping freezes a new material coordinate
	// each step. Both must be compiled before the common inertial predictor.
	if(!mJoints.empty())
		refreshJointRows(true);
    markPrepare(3);
	mStep.rigidPositionBudget = rigidPositionBudget;
	mIterations = AVBD_ITERS_CONTACT;
	// A pure rigid program shares a sweep across all its responsive bodies.
	if(!mStep.commonProgram && !mStep.hasParticleBlocks && mJoints.empty() && !articCount)
		mIterations = rigidPositionBudget;
	if(mJoints.size())
		mIterations = AVBD_ITERS_JOINT;
	if(mArticJoints.size())
		mIterations = AVBD_ITERS_ARTIC_MAX;
	if(mStep.hasParticleBlocks)
	{
		PxU32 authoredSoftIterations = 1;
		for(PxU32 cacheIndex = 0; cacheIndex < mIslandSoftCache.size(); ++cacheIndex)
		{
			const AvbdIslandSoftCache& cache = mIslandSoftCache[cacheIndex];
			if(cache.coupled && cache.liveStamp == mIslandSoftLiveStamp && cache.volume)
				authoredSoftIterations = PxMax(authoredSoftIterations,
					PxMax(PxU32(AVBD_DEFORMABLE_ITERS),
						PxU32(cache.volume->getCore().solverIterationCounts & 0xff)));
		}
		for(PxU32 cacheIndex = 0; cacheIndex < mIslandSurfaceCache.size();
			++cacheIndex)
		{
			const AvbdIslandSurfaceCache& cache = mIslandSurfaceCache[cacheIndex];
			if(cache.liveStamp == mIslandSurfaceLiveStamp && cache.surface)
				authoredSoftIterations = PxMax(authoredSoftIterations,
					PxU32(cache.surface->getCore().solverIterationCounts & 0xff));
		}
		mIterations = PxMax(mIterations, authoredSoftIterations);
	}

	mStep.rowProgramValid = false;
	if(mStep.hasParticleBlocks)
	{
		// The early preparation froze joint/material rest data at the DCD pose.
		// Those rows are not part of the contact prefix: retaining them here
		// makes subsequent refreshJointRows append a second physical copy.
		mRows.truncate(mNpContactRowCount);
		mContactRowCount=mNpContactRowCount;
		if(hasSurface)compileNodalSurfaceRows();
        // Consume Host NP and solverPrep handles in update(), before the
        // first yield. Only backend-owned material slots cross the barrier.
        if(!mLocalDomainFailed)
        {
            compileNodalVolumeRows();
            compileNodalContactRows();
            refreshJointRows();
        }
        if(mLocalDomainFailed){finishStep();return;}
    }
    mStep.prepareCheckpoint=prepareStart;
    mStep.workers=continuation?continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount():0;
    if(continuation){mStep.continuation=continuation;continuation->addReference();}
    mStep.packingRows=true;
    advancePrimal();
}

bool AvbdDynamicsContext::finishPrimalPreparation()
{
    PxU64 prepareStart=mStep.prepareCheckpoint;
    const auto markPrepare=[&](PxU32 part)
    {if(mStep.profile){const PxU64 now=PxTime::getCurrentCounterValue();mStep.prepareTicks[part]+=now-prepareStart;prepareStart=now;}};
    warmstartBlocks();
    mWarmFailureMasks.clear();
    if(mStep.hasGroundedJoint) seedAnchoredJointReactions();
    if(mStep.profile)mStep.warmTicks+=PxTime::getCurrentCounterValue()-mStep.warmStartTicks;
    markPrepare(5);

	// Condition contact rows in the same pose epoch consumed by the local
	// primal.  The spatial response contains the live world-space lever arm and
	// inertia, so evaluating it before the nonlinear warm pose gives the solver
	// a Hessian for a different configuration.  OGC stiffness remains geometry
	// admission data; this is the dimensionless Position-AL conditioning floor.
	for(PxU32 ri = 0; ri < mContactRowCount; ++ri)
	{
		avbdFloorContactRho(mRows[ri], mBodies, mDt);
		if(mRows[ri].scalar().lambda < 0.0f)
			avbdFloorLoadedContactTangentRho(mRows[ri], mBodies, mDt);
	}

	mStep.contactJointClosure = mJoints.size() && mContactRowCount;
	mStep.minimumJointIterations = mIterations;
	if(mStep.contactJointClosure)
		mIterations = PxMax(mIterations, AVBD_ITERS_CONTACT_JOINT_MAX);
	mStep.convergedContactJointSweeps = 0;
	if(!mStep.hasParticleBlocks)
	{
		refreshJointRows();
		compileRowProgram();
		if(mLocalDomainFailed)return false;
		mStep.rowProgramValid = true;
	}
	// A point-lock graph has one smooth fixed-point map until contact changes
    markPrepare(6);
	// its active Hessian.  AVBD_2 accelerates this no-contact map with the
	// Chebyshev semi-iteration below.  It is deliberately disabled for mixed
	// row programs: those require their own spectral contract, and contact/joint
	// islands already terminate on complementarity closure.
	mStep.useJointChebyshev = !mStep.localBudget && mJoints.size() && !mContactRowCount &&
		!mArticJoints.size() && !mStep.hasParticleBlocks;
	for(PxU32 ri = mContactRowCount; ri < mRows.size() && mStep.useJointChebyshev; ++ri)
	{
		const AvbdRowConstRef& row = mRows[ri];
		if((row.header().flags & eAVBD_ROW_INACTIVE) == 0 &&
			(row.header().flags & eAVBD_ROW_POINT_LOCK) == 0)
			mStep.useJointChebyshev = false;
	}
	mStep.chebyOmega = 1.0f;
	if(mStep.useJointChebyshev)
	{
		mChebyPrevPosScratch.resize(mBodies.size());
		mChebyPrevPrevPosScratch.resize(mBodies.size());
		mChebyPrevRotScratch.resize(mBodies.size());
		mChebyPrevPrevRotScratch.resize(mBodies.size());
		for(PxU32 i = 0; i < mBodies.size(); ++i)
		{
			mChebyPrevPosScratch[i] = mChebyPrevPrevPosScratch[i] = mBodies[i].p;
			mChebyPrevRotScratch[i] = mChebyPrevPrevRotScratch[i] = mBodies[i].q;
		}
	}
	mStep.minimumIterations = mIterations;
	mStep.requiresDomainClosure = false;
	for(PxU32 blockIndex = 0; blockIndex < mBlocks.size(); ++blockIndex)
	{
		const AvbdBlock& block = mBlocks[blockIndex];
		// Fixed-budget publication checks the material/contact domain, not the
		// legacy iterate-to-stationarity certificate. Physical quality remains
		// independently measured at the authored four-sweep budget.
		mStep.requiresDomainClosure |= !mStep.localBudget && block.type == eAVBD_BLOCK_PARTICLE3 &&
			block.particleKernel == eAVBD_PARTICLE_KERNEL_TET;
	}
	mStep.iterationLimit = mStep.requiresDomainClosure ?
		PxMax(mStep.minimumIterations, AVBD_DEFORMABLE_ITERS_MAX) : mStep.minimumIterations;
	mStep.convergedHeterogeneousSweeps = 0;
	mStep.heterogeneousConverged=!mStep.requiresDomainClosure;
	mStep.contactJointConverged=!mStep.contactJointClosure;
	mStep.lastCoupledDomainStationary=false;
    if(mStep.localBudget)
    {
        mStep.iterationLimit=mStep.localBudget;
        if(!localCapability()){mLocalDomainFailed=1;return false;}
    }

 if(mStep.matchNodal)
 {
     mNodalPartners.resize(mBlocks.size(),PX_MAX_U32);
     mNodalIncidentLoad.resize(mBlocks.size());
     mStep.nominationCursor=mBodies.size();
 }
 if(mStep.hasWideNodes)
     {
         mWideNomination=PX_NEW(AvbdWideNomination)();
         for(PxU32 b=mBodies.size();b<mBlocks.size();++b)
         {
             const PxU32 begin=mRowProgram.blockOffsets[b],end=mRowProgram.blockOffsets[b+1];
             if(end-begin>AvbdLocalWork::RowLimit)mWideNomination->tree.add(b,begin,end);
         }
         mWideNomination->prepare();
 }
 if(mStep.profile)mStep.packTicks=PxTime::getCurrentCounterValue()-mStep.backendStart;
 markPrepare(7);
 return !mLocalDomainFailed;
}

void AvbdDynamicsContext::advancePrimal()
{
    finishLocalProfileWave();
    if(mStep.materialPending){finishStep();return;}
    if(mStep.packingRows)
    {
        while(mStep.surfaceCursor<mNodalShellMaterials.size() && !mLocalDomainFailed)
            if(!dispatchRowRanges(SurfacePackStage,mStep.surfaceCursor,mNodalShellMaterials.size(),AvbdLocalWork::RowLimit))return;
        if(mLocalDomainFailed){finishStep();return;}
        for(const auto& batch:mSurfaceBatches)
        {
            auto& cache=mIslandSurfaceCache[batch.cacheIndex];
            cache.packedStamp=cache.liveStamp;cache.packedBlock=cache.blockStart;
            cache.packedMaterial=batch.firstMaterial;cache.packedSupport=batch.firstSupport;cache.packedRows=batch.count;
        }
        if(mStep.hasParticleBlocks)
        {
            compileRowProgram();
            if(mLocalDomainFailed) { finishStep(); return; }
            mStep.rowProgramValid = true;
        }
        for(PxU32 i = 0; i < mBodies.size(); ++i)
        {
            AvbdSolverBody& boundary = mBodies[i];
            if(boundary.isKinematic)
            {
                boundary.p = boundary.pHat;
                boundary.q = boundary.qHat;
            }
        }
        if(mStep.profile)
        {
            const PxU64 now=PxTime::getCurrentCounterValue();
            mStep.prepareTicks[4]+=now-mStep.prepareCheckpoint;
            mStep.prepareCheckpoint=now;
        }
        mStep.packingRows=false;
        mStep.preparingInertia=true;
    }
    if(mStep.preparingInertia)
    {
        while(mStep.impactCursor<mNodalImpactRows.size() && !mLocalDomainFailed)
            if(!dispatchLocalBatch(NodalImpactStage))return;
        if(!mStep.warmStarted && !mLocalDomainFailed)
        {
            mStep.warmStarted=true;
            if(mStep.profile)mStep.warmStartTicks=PxTime::getCurrentCounterValue();
            prepareWarmChecks();
        }
        while(mStep.warmMaskCursor<mWarmFailureMasks.size() && !mLocalDomainFailed)
            if(!dispatchRowRanges(WarmCheckStage,mStep.warmMaskCursor,mWarmFailureMasks.size(),AvbdLocalWork::RowLimit/32))return;
        mStep.preparingInertia=false;
        if(mLocalDomainFailed || !finishPrimalPreparation()){finishStep();return;}
    }

    while(mStep.matchNodal && mStep.nominationCursor<nominationJobs() && !mLocalDomainFailed)
        if(!dispatchLocalBatch(NodalMatchStage))return;
    while(mStep.matchNodal && mStep.loadCursor<mNodalContacts.size() && !mLocalDomainFailed)
        if(!dispatchLocalBatch(NodalLoadStage))return;
	while ((mStep.iteration < mStep.iterationLimit ||
        (mStep.localBudget && mStep.queryCursor<mContactRowCount+mNodalContacts.size())) && !mLocalDomainFailed)
	{
        if(mStep.localBudget && mStep.iteration==mStep.iterationLimit)
        {if(!dispatchLocalBatch(2))return;continue;}
		if (!mStep.sweepPrepared)
		{
			if (mStep.useJointChebyshev)
			{
				for (PxU32 i = 0; i < mBodies.size(); ++i)
				{
					mChebyPrevPrevPosScratch[i] = mChebyPrevPosScratch[i];
					mChebyPrevPrevRotScratch[i] = mChebyPrevRotScratch[i];
					mChebyPrevPosScratch[i] = mBodies[i].p;
					mChebyPrevRotScratch[i] = mBodies[i].q;
				}
			}
			// The step-start manifold is the DCD epoch.  Recompiling hetero rows
			// mid-step at a crushed pose is a FRESH OGC slam onto both cores, and
			// that iterate is the published pancake.  Certificates may still
			// refresh; new features wait for the next step's DCD.
			bool primalTopologyChanged = false;
			if (mStep.hasParticleBlocks)
				primalTopologyChanged = refreshJointRows();
			// A new DCD epoch initially contains only contact rows. Persisted
			// solverPrep row counts can stay equal even though joints have just
			// been appended to that epoch for the first time.
			primalTopologyChanged |= mRowProgram.rowOffsets.size() != mRows.size() + 1;
			// DCD supports remain fixed until the next step. Recompile the
			// execution graph only when its actual row incidence can have changed.
			const bool colorsChanged =
			    mStep.iteration == 0 || primalTopologyChanged || !mStep.rowProgramValid ||
                (mStep.localBudget && !mLocalTranslationClosed.empty());
			if (!mStep.rowProgramValid || primalTopologyChanged)
			{
                if(mStep.iteration>0 && primalTopologyChanged)mLocalTranslationClosed.clear();
				compileRowProgram();
				if(mLocalDomainFailed) { finishStep(); return; }
				mStep.rowProgramValid = true;
			}
			if (colorsChanged)
            {
				compileBlockColors();
                bakeLocalColorTasks();
                if(mStep.localBudget && !localCapability())
                {mLocalDomainFailed=1;break;}
            }
			if(mStep.iteration==0)captureCommonProgram();
            preparePrimalSweep();
            if(mStep.boundedCommon)
            {
                if(!beginBoundedCommon()){mLocalDomainFailed=1;break;}
                mStep.sweepPrepared=true;
            }
            else if(mStep.commonProgram && !mStep.localBudget)
            {
                bool stationary=false;
                if(!solveCommonCandidate(stationary)){mLocalDomainFailed=1;break;}
                mStep.lastCoupledDomainStationary=stationary;
                ++mStep.iteration;if(stationary)break;
                if(refreshJointRows())
                {mStep.rowProgramValid=false;for(auto& idle:mPrimalIdleIteration)idle=0;}
                continue;
            }
			mStep.nextColor = 0;
            mWidePrepared=PX_MAX_U32;
            // Alternate the traversal of the same independent color batches
            // to avoid keeping one side of a loaded interface last.
            mStep.reverseColors=mStep.localBudget && mStep.hasParticleBlocks &&
                ((mStep.iteration+mIslandSoftLiveStamp)&1);
            if(mStep.reverseColors && mColorOffsets.size()>1)mStep.nextColor=mColorOffsets.size()-2;
            mStep.localCursor=mColorOffsets.empty()?0:localColorStart(mStep.nextColor);mStep.dualCursor=0;
			mStep.sweepPrepared = true;
		}
        while (mStep.pointInterfaceCursor < mStep.pointInterfaceEnd)
            if (!dispatchRowRanges(PointInterfaceStage, mStep.pointInterfaceCursor, mStep.pointInterfaceEnd, 8))
                return;
        if (mLocalDomainFailed)
            break;
        if(mStep.boundedCommon)
        {
            // A pending wave is joined before any program/candidate state changes.
            if(mBoundedResponse && mBoundedResponse->status==AvbdBoundedResponse::Running)
            {
                if(!dispatchBoundedWave())return;
                continue;
            }
            bool done=false,stationary=false;
            if(!completeBoundedCommon(done,stationary)){mLocalDomainFailed=1;break;}
            if(!done)continue;
            mStep.lastCoupledDomainStationary=stationary;mStep.sweepPrepared=false;
            ++mStep.iteration;if(stationary)break;
            if(refreshJointRows())
            {mStep.rowProgramValid=false;for(auto& idle:mPrimalIdleIteration)idle=0;}
            continue;
        }
        if(mStep.localBudget)
        {
            while(mStep.incidencePrimal && mStep.historyCursor<incidenceHistoryEnd(true))
                if(!dispatchLocalBatch(IncidenceHistoryStage))return;
            while(mStep.nextColor+1<mColorOffsets.size() && !mLocalDomainFailed)
            {
                if(mWideInFlight){mWideInFlight=false;retireWide();}
                if(mStep.localCursor==localColorStart(mStep.nextColor))
                    armWideColor();
                if(mStep.localCursor<localColorStart(mStep.nextColor+1))
                {
                    const bool laneDrain=mWideArmed && mColorTasks && !mColorTasks->offsets.empty();
                    if(!dispatchLocalBatch(false)){if(laneDrain)mWideInFlight=true;return;}
                    if(laneDrain)retireWide();
                    continue;
                }
                if(mWideArmed)
                {if(!dispatchFusedWide()){mWideInFlight=true;return;}}
                if(mLocalDomainFailed)break;
                if(mStep.reverseColors)
                    mStep.nextColor=mStep.nextColor?mStep.nextColor-1:mColorOffsets.size()-1;
                else ++mStep.nextColor;
                mStep.localCursor=localColorStart(mStep.nextColor);
            }
            if(mLocalDomainFailed)break;
            if(mStep.dualCursor<localDualRows())
            {if(!dispatchLocalBatch(true))return;continue;}
            traceLocalMomentum("primal");
            writeJointDualState();
            ++mStep.iteration;mStep.sweepPrepared=false;
            if(refreshJointRows())mStep.rowProgramValid=false;
            continue;
        }
		while (mStep.nextColor + 1 < mColorOffsets.size())
		{
			const PxU32 begin = mColorOffsets[mStep.nextColor], end = mColorOffsets[mStep.nextColor + 1];
            const auto color=mStep.nextColor;
            if(color+1<mLoopColors.size() && mLoopColors[color]<mLoopColors[color+1])
            {
                const auto start=mLoopColors[color],stop=mLoopColors[color+1];
                while(start+mLoopCursor<stop)
                {
                    const auto& w=mLoopWaves[start+mLoopCursor++];
                    if(!dispatchReadyWave(w))return;
                }
                if(mLocalDomainFailed)break;
                for (PxU32 i = mLoopOwnerOffsets[color]; i < mLoopOwnerOffsets[color + 1]; ++i)
                {
                    const PxU32 root = mLoopOwners[i];
                    if (!mLoopScratch[mRigidGroups[root].loopSlot]->wholeLoopJob)
                        solvePrimalBlock(root);
                }
                mLoopCursor=0;++mStep.nextColor;continue;
            }
			++mStep.nextColor;
			// Batch independent blocks to amortize dispatcher overhead. Workers claim
			// blocks dynamically because equal block counts need not mean equal work.
			// The advancing worker also solves; none waits for a queued child.
			const PxU32 minimumBlocksPerTask = 2;
			const PxU32 count = PxMin(mStep.workers, (end - begin) / minimumBlocksPerTask);
			if (mStep.continuation && count > 1)
			{
				if (!mColorContinuation)
					mColorContinuation = PX_NEW(AvbdAdvanceTask)(*this, mContextID);
				AvbdAdvanceTask *next = mColorContinuation;
				next->setContinuation(*mStep.continuation->getTaskManager(), NULL);
				mColorCursor = PxI32(begin);
				mColorEnd = end;
				while (mBlockTasks.size() + 1 < count)
					mBlockTasks.pushBack(PX_NEW(AvbdBlockSolveTask)(*this, mContextID));
				for (PxU32 i = 0; i + 1 < count; ++i)
				{
					AvbdBlockSolveTask *task = mBlockTasks[i];
                    task->local=false;task->bounded=false;
					task->setContinuation(next);
					++mStep.submittedTasks;
					task->removeReference();
				}
				solvePrimalBatch();
				if(next->completeInline())
					continue;
				return;
			}
			for (PxU32 i = begin; i < end; ++i)
				solvePrimalBlock(mColorBlocks[i]);
		}
		bool coupledDomainStationary = !mStep.hasParticleBlocks;
		const PxReal primalNormalizedStep = finishPrimalSweep(&coupledDomainStationary);

		mStep.lastCoupledDomainStationary = coupledDomainStationary;
		const bool jointTopologyChanged = refreshJointRows();
		applyDual();
		++mStep.rigidDualSweeps;
		if (mStep.useJointChebyshev && mStep.iteration >= 2)
		{
			const PxReal rhoSq = AVBD_JOINT_CHEBYSHEV_RHO * AVBD_JOINT_CHEBYSHEV_RHO;
			mStep.chebyOmega = mStep.iteration == 2 ? 2.0f / (2.0f - rhoSq)
			                                        : 1.0f / (1.0f - rhoSq * mStep.chebyOmega * 0.25f);
			mStep.chebyOmega = PxClamp(mStep.chebyOmega, 1.0f, 2.0f);
			for (PxU32 i = 0; i < mBodies.size(); ++i)
			{
				AvbdSolverBody &b = mBodies[i];
				if (b.invMass <= 0.0f || b.isKinematic)
					continue;
				b.p = mChebyPrevPrevPosScratch[i] + (b.p - mChebyPrevPrevPosScratch[i]) * mStep.chebyOmega;
				PxQuat qCurrent = b.q;
				const PxQuat &qPrevious = mChebyPrevPrevRotScratch[i];
				if (qPrevious.dot(qCurrent) < 0.0f)
					qCurrent = -qCurrent;
				b.q = PxQuat(qPrevious.x + mStep.chebyOmega * (qCurrent.x - qPrevious.x),
				             qPrevious.y + mStep.chebyOmega * (qCurrent.y - qPrevious.y),
				             qPrevious.z + mStep.chebyOmega * (qCurrent.z - qPrevious.z),
				             qPrevious.w + mStep.chebyOmega * (qCurrent.w - qPrevious.w))
				          .getNormalized();
				if (b.core && b.core->lockFlags)
				{
					PxVec3 dp = b.p - b.pStart;
					PxVec3 dth = avbdQuatLogWorld(b.qStart, b.q);
					avbdProjectLockedDelta(b, dp, dth);
					b.p = b.pStart;
					b.q = b.qStart;
					avbdApplyDelta(b, dp, dth);
				}
			}
		}
		if (jointTopologyChanged)
			mStep.rowProgramValid = false;
		if (mStep.contactJointClosure && mStep.iteration + 1 >= mStep.minimumJointIterations)
		{
			// NP normals are unilateral AL rows. Their stopping certificate is
			// complementarity plus closing displacement, not a zero raw gap or
			// a small pose step (the joint graph may still have a bending mode).
			bool finiteNpState = true;
			bool hasNpRow = false;
			bool hasSpringObjective = false;
			for(PxU32 ri = mContactRowCount; ri < mRows.size(); ++ri)
				if((mRows[ri].header().flags & (eAVBD_ROW_DRIVE | eAVBD_ROW_INACTIVE)) == eAVBD_ROW_DRIVE)
					hasSpringObjective = true;
			PxReal residual = 0.0f, closing = 0.0f;
			const PxReal feasibilityTolerance =
			    AVBD_NP_JOINT_CLOSURE_REL_TOL * PxMax(getLengthScale(), 1.0e-6f);
			for (PxU32 ri = 0; ri < mNpContactRowCount; ++ri)
			{
				const AvbdRowConstRef&row = mRows[ri];
				if (row.header().flags & eAVBD_ROW_INACTIVE)
					continue;
				hasNpRow = true;
				const PxReal C = avbdPrimalC(row, mBodies, getLengthScale());
				const PxReal loadDistance = PxMax(0.0f, -row.scalar().lambda) / PxMax(row.scalar().rho, 1.0e-12f);
				const PxReal relativeDisplacement = row.scalar().n.dot(avbdContactRelDisp(row, mBodies));
				if (!PxIsFinite(C) || !PxIsFinite(loadDistance) || !PxIsFinite(relativeDisplacement))
					finiteNpState = false;
				residual = PxMax(residual, PxAbs(PxMin(C, loadDistance)));
				if (C <= 0.0f || loadDistance > 0.0f)
					closing = PxMax(closing, -relativeDisplacement);
			}
			// Contact complementarity cannot certify a finite spring optimum.
			// Its local primal map must also close within the existing budget.
			if (finiteNpState && (!hasSpringObjective || primalNormalizedStep <= 1.0f) &&
			    (hasNpRow ? (residual <= feasibilityTolerance && closing <= feasibilityTolerance)
			              : primalNormalizedStep <= 1.0f))
				++mStep.convergedContactJointSweeps;
			else
				mStep.convergedContactJointSweeps = 0;
			if (mStep.convergedContactJointSweeps >= 2)
			{
				mStep.contactJointConverged = true;
				if (!mStep.requiresDomainClosure)
					break;
			}
		}
		// WebPhysics uses a fixed primal budget, not a pose-map early-exit.
		// Articulation islands already take AVBD_ITERS_ARTIC_MAX sweeps; stopping
		// on two small GS steps leaves gravity's common mode in ker(J) of the
		// stiff D6 locks.  A DCD refresh only replaces the local admissible
		// region -- it is not another predictor or writeback pass.
		if (mStep.requiresDomainClosure && mStep.iteration + 1 >= mStep.minimumIterations)
		{
			// The authored count is a minimum nonlinear budget.  Publication needs
			// both the GS block map and the shared-domain Schur/KKT direction to be
			// stationary while the compiled topology remains unchanged.
			if (!jointTopologyChanged && coupledDomainStationary &&
			    primalNormalizedStep <= 1.0f)
				++mStep.convergedHeterogeneousSweeps;
			else
				mStep.convergedHeterogeneousSweeps = 0;
			if (mStep.convergedHeterogeneousSweeps >= 2)
			{
				mStep.heterogeneousConverged = true;
				if (mStep.contactJointConverged)
					break;
			}
		}
		++mStep.iteration;
		mStep.sweepPrepared = false;
	}
	finishStep();
}

void AvbdDynamicsContext::finishStep()
{
    if(!mStep.materialPending)
    {
	    clearCommonPublication();
	    if(!mLocalDomainFailed && mStep.requiresDomainClosure && !mStep.lastCoupledDomainStationary)
	    {
	        mLocalDomainFailed=1;
	        std::fprintf(stderr,"[AVBD_COMMON_STEP] stage=publish reason=required_domain_closure status=FAIL\n");
	    }
		// The last solverPrep may change row count at the iteration boundary too.
		// Velocity ownership must consume the final support program, even when
		// there is no following primal sweep to rebuild it.
		if(!mLocalDomainFailed && !mStep.rowProgramValid)
			compileRowProgram();
		if (mLocalDomainFailed)
		{
			if(mStep.commonProgram)
			{
				// Rejected backend iterates must not seed a later physical epoch.
				mContactWarm.clear();mNodalFriction.clear();
				for(auto row:mRows)if(row.hasScalar())row.scalar().lambda=row.scalar().lambdaT0=row.scalar().lambdaT1=0;
				writeJointDualState();
				if(std::getenv("PHYSX_AVBD_COMMON_STATS"))
					std::fprintf(stderr,"[AVBD_COMMON_FRAME] status=REJECTED poseForcePublications=0 bdf=0\n");
			}
			PxGetFoundation().error(
			    PxErrorCode::eINVALID_OPERATION, PX_FL,
			    "AVBD could not verify the local contact/material domain; step not published.");
			goto complete;
		}
		writeJointConstraintWriteback();
		if(mStep.commonProgram)
		{
            for(PxU32 i=mNpContactRowCount;i<mRows.size();++i)
            {
                const auto row=mRows[i];
                if(row.hasScalar() && row.scalar().reactionOutput)
                    *row.scalar().reactionOutput=-row.scalar().lambda;
            }
		}
		if (mStep.hasParticleBlocks)
			prepareDeformableVelocities(mDt);
		prepareRigidVelocities(mDt);
    }
    if(mStep.materialPending)
    {
        if(!advanceContactMaterial()) return;
        publishContactMaterial();
        mStep.materialPending=false;
    }
	applyHeterogeneousContactMaterial();
    publishNpContactImpulses();
	traceLocalMomentum("velocity",true);
	finalizeRigidBodies(mDt);
	if (mStep.hasParticleBlocks)
		finalizeDeformableBlocks(mDt);
	if(mStep.commonProgram && std::getenv("PHYSX_AVBD_COMMON_STATS"))
		std::fprintf(stderr,"[AVBD_COMMON_FRAME] status=ACCEPTED poseForcePublications=1 bdf=1 stationary=%u iterations=%u\n",
			unsigned(mStep.lastCoupledDomainStationary),mStep.iteration);
complete:
    PX_DELETE(mLocalWork);mLocalWork=NULL;
    const PxU64 profileEnd=mStep.profile?PxTime::getCurrentCounterValue():0;
    if(mStep.localBudget && std::getenv("PHYSX_AVBD_COMMON_STATS"))
    {
        PxU32 particles=0;for(const auto& block:mBlocks)particles+=block.type==eAVBD_BLOCK_PARTICLE3;
        std::fprintf(stderr,"[AVBD_BLOCK_DOF] rigid6=%u particle3=%u physicalDof=%u nodalSolves=%d nodalScalarBytes=%u nodalMaterialRows=%u mixedContactRows=%u pairedUpdates=%d nominationMaxRows=%u membranePrimal=%u contactPrimal=%u\n",
            mBodies.size(),particles,6*mBodies.size()+3*particles,int(mStep.nodalSolves),unsigned(sizeof(AvbdNodalReal)),mNodalShellMaterials.size()+mNodalVolumeMaterials.size(),mNodalContacts.size(),int(mStep.nodalPairUpdates),mStep.nominationMaxRows,unsigned(mStep.membranePrimalSolves),unsigned(mStep.contactPrimalSolves));
        PxU64 groups=0,attempts=mStep.retiredGroupStats.attempts,factors=mStep.retiredGroupStats.products,
            accepted=mStep.retiredGroupStats.accepted,evaluations=mStep.retiredGroupStats.materialEvaluations,
            uses=mStep.retiredGroupStats.materialEndpointUses,rhs=mStep.retiredGroupStats.rightHandSides;
        for(const auto& group:mRigidGroups)if(group.count>1)
        {++groups;attempts+=group.attempts;factors+=group.products;accepted+=group.accepted;
            evaluations+=group.materialEvaluations;uses+=group.materialEndpointUses;rhs+=group.rightHandSides;}
        std::fprintf(stderr,"[AVBD_LOCAL_GROUPS] groups=%llu primalGroups=%llu factorizationAttempts=%llu acceptedGroups=%llu materialEvaluations=%llu materialEndpointUses=%llu groupBodyLimit=16 physicalBlocksPerTask=32 rowWorkPerTask=256 triangularSolves=%llu colorBuilds=%u\n",
            static_cast<unsigned long long>(groups),static_cast<unsigned long long>(attempts),static_cast<unsigned long long>(factors),
            static_cast<unsigned long long>(accepted),static_cast<unsigned long long>(evaluations),static_cast<unsigned long long>(uses),static_cast<unsigned long long>(rhs),mStep.localColorBuilds);
        std::fprintf(stderr,"[AVBD_LOCAL_BUDGET] budget=%u iterations=%u primalBlocks=%llu dualRows=%llu tasks=%u peakTasks=%d workers=%u colors=%u batchLimit=32 rowWorkLimit=256 queryRows=%u geometryQueries=%d geometryPointVisits=%llu transportedProofs=%d reselectedProofs=%d warmQueries=%llu strictStationarity=NOT_REQUIRED status=%s\n",
            mStep.localBudget,mStep.iteration,static_cast<unsigned long long>(mStep.localBlocks),
            static_cast<unsigned long long>(mStep.localRows),mStep.submittedTasks,int(mPeakPrimalTasks),mStep.workers,
            mColorOffsets.empty()?0:mColorOffsets.size()-1,mStep.queryCursor,int(mStep.localQueries),static_cast<unsigned long long>(mStep.localPointVisits),
            int(mStep.localTransportedProofs),int(mStep.localReselectedProofs),static_cast<unsigned long long>(mStep.warmDomainTests),
            mLocalDomainFailed?"REJECTED":"ACCEPTED");
        std::fprintf(stderr,"[AVBD_LOCAL_ADMISSION] primalQueries=%d primalPointVisits=%llu searchTrials=%d exhaustedSearches=%d searchLimit=16 axisLimit=15 proofLimit=4 pointsPerSideLimit=256\n",
            int(mStep.localPrimalQueries),static_cast<unsigned long long>(mStep.localPrimalPointVisits),
            int(mStep.localSearchTrials),int(mStep.localSearchExhausted));
        std::fprintf(stderr,"[AVBD_COLOR_TASKS] builds=%u reusedPlans=%u waves=%u splitBlocks=%u leaves=%u leafRows=%u packetFaces=%u packetColors=%u parallelFaces=%u\n",
            mStep.bakedColorPlans,mStep.reusedColorPlans,mStep.bakedColorWaves,mStep.splitPrimalBlocks,
            mStep.splitRowLeaves,unsigned(AvbdColorTasks::LeafRows),mStep.membranePacketFaces,mStep.membranePacketColors,mStep.membraneParallelFaces);
        std::fprintf(stderr,"[AVBD_LOCAL_DISPATCH] queuePeak=%u queueDepth=%u primalMaxBlocks=%u primalMaxRows=%u dualMaxRows=%u queryMaxRows=%u dualRowLimit=%u\n",
            mStep.localQueuePeak,unsigned(LocalQueueDepth),mStep.localPrimalMaxBlocks,mStep.localPrimalMaxRows,
            mStep.localDualMaxRows,mStep.localQueryMaxRows,unsigned(LocalDualBatchLimit));
        std::fprintf(stderr,"[AVBD_LOCAL_TILES] blocks=%llu leaves=%llu jobs=%llu waves=%llu maxIncidentRows=%u scratchBytes=%llu rowLimit=256 reductionInputs=2 searchLimit=16\n",
            static_cast<unsigned long long>(mStep.tiledBlocks),static_cast<unsigned long long>(mStep.tiledLeaves),
            static_cast<unsigned long long>(mStep.tiledJobs),static_cast<unsigned long long>(mStep.tiledWaves),mStep.tiledMaxRows,
            static_cast<unsigned long long>(mStep.tiledScratchBytes));
    }
    if(mStep.boundedCommon && std::getenv("PHYSX_AVBD_COMMON_STATS"))
        std::fprintf(stderr,"[AVBD_BOUNDED_SCHEDULE] waves=%llu batches=%llu sweeps=%llu tasks=%u peakTasks=%d workers=%u batchLimit=%u jobs=%llu groups=%llu tilesPerTask=%u\n",
            static_cast<unsigned long long>(mStep.boundedWaves),static_cast<unsigned long long>(mStep.boundedBatches),
            static_cast<unsigned long long>(mStep.boundedSweeps),mStep.submittedTasks,int(mPeakPrimalTasks),mStep.workers,AvbdBoundedResponse::BatchWidth,
            static_cast<unsigned long long>(mStep.boundedJobs),static_cast<unsigned long long>(mStep.boundedGroups),AvbdCpuBatch::TilesPerTask);
    if(mStep.profile)
    {
        const auto& frequency=PxTime::getBootCounterFrequency();
        const double ms=double(frequency.mNumerator)/double(frequency.mDenominator)*1.e-5;
        if(mStep.localBudget)
        {
            if(mStep.localTimeline)
            {
                std::fprintf(stderr,"[AVBD_LOCAL_TIMELINE] stage=%u jobs=%u lanes=%u spanMs=%.6f lptMs=%.6f selection=worstGapPerStep\n",
                    mStep.localTraceStage,mStep.localTraceJobs,mStep.localTraceLanes,mStep.localTraceSpan*ms,mStep.localTraceLpt*ms);
                for(PxU32 i=0;i<mStep.localTraceJobs;++i)
                    std::fprintf(stderr,"[AVBD_LOCAL_TASK] slot=%u thread=%llu readyMs=%.6f startMs=%.6f endMs=%.6f\n",i,
                        static_cast<unsigned long long>(mStep.localTrace[i][3]),mStep.localTrace[i][0]*ms,
                        mStep.localTrace[i][1]*ms,mStep.localTrace[i][2]*ms);
                for(PxU32 stage=0;stage<LocalStageCount;++stage)
                    std::fprintf(stderr,"[AVBD_LOCAL_IDLE] stage=%u readyLaneMs=%.6f widthLaneMs=%.6f\n",
                        stage,mStep.localReadyIdleTicks[stage]*ms,mStep.localWidthIdleTicks[stage]*ms);
            }
            PxU64 assembly=mStep.retiredGroupStats.assemblyTicks,factor=mStep.retiredGroupStats.factorTicks,search=mStep.retiredGroupStats.searchTicks;
            PxU64 geometry=mStep.retiredGroupStats.geometryTicks,model=mStep.retiredGroupStats.modelTicks;
            for(const auto& g:mRigidGroups)
            {assembly+=g.assemblyTicks;factor+=g.factorTicks;search+=g.searchTicks;geometry+=g.geometryTicks;model+=g.modelTicks;}
            std::fprintf(stderr,"[AVBD_LOCAL_PROFILE] groupAssemblyTaskMs=%.6f groupSolveTaskMs=%.6f groupSearchTaskMs=%.6f groupGeometryTaskMs=%.6f groupModelTaskMs=%.6f colorMs=%.6f warmMs=%.6f backendMs=%.6f\n",
                assembly*ms,factor*ms,search*ms,geometry*ms,model*ms,mStep.colorTicks*ms,mStep.warmTicks*ms,(profileEnd-mStep.backendStart)*ms);
            std::fprintf(stderr,"[AVBD_WARM_CHECKS] checkedRows=%u failedRows=%u masks=%u\n",
                mStep.warmCheckedRows,mStep.warmFailedRows,mStep.warmMaskCursor);
            PxU32 keptSupports=0,keptMaterials=0;
            for(const auto& batch:mSurfaceBatches)
            {
                if(batch.reuseSupports)keptSupports+=batch.count;
                if(batch.reuseMaterials)keptMaterials+=batch.count;
            }
            std::fprintf(stderr,"[AVBD_MATERIAL_PACK] rows=%u reusedTopologyRows=%u topologyBytes=%llu retainedSupportRows=%u retainedMaterials=%u\n",
                mStep.surfaceCursor,mRowProgram.reusedRows(),static_cast<unsigned long long>(mRowProgram.prefixBytes()),keptSupports,keptMaterials);
            for(PxU32 part=0;part<4;++part)
                if( mStep.criticalItemCount[part])std::fprintf(stderr,
                    "[AVBD_CRITICAL_ITEM] kind=%u waves=%llu rows=%llu itemMs=%.6f\n",part,
                    static_cast<unsigned long long>(mStep.criticalItemCount[part]),
                    static_cast<unsigned long long>(mStep.criticalItemRows[part]),mStep.criticalItemTicks[part]*ms);
            for(PxU32 part=0;part<8;++part)
                std::fprintf(stderr,"[AVBD_LOCAL_PREPARE] part=%u ms=%.6f\n",part,mStep.prepareTicks[part]*ms);
            PxU64 shapes[17][3];PxMemCopy(shapes,mStep.localShape,sizeof(shapes));
            for(const auto& g:mRigidGroups)if(g.count<=16)
            {shapes[g.count][0]+=g.attempts;shapes[g.count][1]+=g.products;shapes[g.count][2]+=g.rightHandSides;}
            for(PxU32 n=2;n<=16;++n)if(shapes[n][0])
                std::fprintf(stderr,"[AVBD_LOCAL_SHAPE] bodies=%u solves=%llu factorsIncludingSchur=%llu rhs=%llu\n",n,
                    static_cast<unsigned long long>(shapes[n][0]),static_cast<unsigned long long>(shapes[n][1]),static_cast<unsigned long long>(shapes[n][2]));
            for(PxU32 stage=0;stage<4;++stage)
            {
                PxU64 ticks=mStep.retiredGroupStats.assemblyPartTicks[stage],rows=mStep.retiredGroupStats.assemblyPartRows[stage];
                for(const auto& g:mRigidGroups){ticks+=g.assemblyPartTicks[stage];rows+=g.assemblyPartRows[stage];}
                std::fprintf(stderr,"[AVBD_LOCAL_ASSEMBLY] part=%u taskMs=%.6f rows=%llu\n",stage,ticks*ms,static_cast<unsigned long long>(rows));
            }
            for(PxU32 stage=0;stage<LocalStageCount;++stage)
            {
                std::fprintf(stderr,"[AVBD_LOCAL_STAGE] stage=%u taskMs=%.6f spanMs=%.6f capacityMs=%.6f waves=%llu\n",
                    stage,mStep.localTaskTicks[stage]*ms,mStep.localWaveTicks[stage]*ms,mStep.localCapacityTicks[stage]*ms,
                    static_cast<unsigned long long>(mStep.localWaves[stage]));
                std::fprintf(stderr,"[AVBD_LOCAL_WORKSPAN] stage=%u lowerMs=%.6f longestMs=%.6f lptMs=%.6f startupMs=%.6f drainMs=%.6f\n",
                    stage,mStep.localWorkBoundTicks[stage]*ms,mStep.localLongestTicks[stage]*ms,
                    mStep.localLptTicks[stage]*ms,mStep.localStartupTicks[stage]*ms,mStep.localDrainTicks[stage]*ms);
            }
        }
        std::fprintf(stderr,"[AVBD_BOUNDED_PROFILE] compileMs=%.6f setupMs=%.6f responseMs=%.6f admissionMs=%.6f groupSpanMs=%.6f jobSumMs=%.6f jobMaxMs=%.6f callerTailMs=%.6f builds=%llu resets=%llu backendMs=%.6f packMs=%.6f contactPackMs=%.6f warmMs=%.6f warmDomainTests=%llu warmDomainReuses=%llu\n",
            mStep.compileTicks*ms,mStep.setupTicks*ms,mStep.responseTicks*ms,mStep.admissionTicks*ms,
            mStep.groupTicks*ms,mStep.jobTotal*ms,mStep.jobMaximum*ms,mStep.tailTicks*ms,
            static_cast<unsigned long long>(mStep.responseBuilds),static_cast<unsigned long long>(mStep.responseResets),
            (profileEnd-mStep.backendStart)*ms,mStep.packTicks*ms,mStep.contactPackTicks*ms,mStep.warmTicks*ms,
            static_cast<unsigned long long>(mStep.warmDomainTests),static_cast<unsigned long long>(mStep.warmDomainReuses));
    }
	if (std::getenv("PHYSX_AVBD_SCHEDULE_STATS"))
	{
		PxU32 width = 0;
		PxU32 idleSkips=0;
		for(PxU32 i=0;i<mPrimalIdleSkips.size();++i) idleSkips+=mPrimalIdleSkips[i];
		for (PxU32 c = 0; c + 1 < mColorOffsets.size(); ++c)
			width = PxMax(width, mColorOffsets[c + 1] - mColorOffsets[c]);
		std::fprintf(stderr,
		             "[AVBD_SCHEDULE] blocks=%u colors=%u maxWidth=%u tasks=%u peakTasks=%d workers=%u idleSkips=%u positionBudget=%u rigidDualSweeps=%u\n",
		             mColorBlocks.size(), mColorOffsets.empty() ? 0 : mColorOffsets.size() - 1, width,
		             mStep.submittedTasks, int(mPeakPrimalTasks), mStep.workers,idleSkips,mStep.iterationLimit,mStep.rigidDualSweeps);
		std::fprintf(stderr,"[AVBD_STORAGE] bodies=%u rows=%u joints=%u bodyBytes=%llu rowBytes=%llu jointBytes=%llu rowStride=%llu supports=%u triangleDomains=%u trianglePoints=%u volumeCaches=%u surfaceCaches=%u softPairContacts=%u\n",
			mBodies.size(),mRows.size(),mJoints.size(),static_cast<unsigned long long>(PxU64(mBodies.size())*sizeof(AvbdSolverBody)),
			static_cast<unsigned long long>(mRows.bytes()),static_cast<unsigned long long>(PxU64(mJoints.size())*sizeof(AvbdJointSrc)+PxU64(mJointStates.size())*sizeof(AvbdJointState)),
			static_cast<unsigned long long>(sizeof(AvbdRow)),mRowSupports.size(),0u,0u,mIslandSoftCache.size(),mIslandSurfaceCache.size(),0u);
        std::fprintf(stderr,"[AVBD_ROW_STORAGE] headers=%u scalarPayloads=%u headerStride=%llu scalarStride=%llu bytes=%llu capacityBytes=%llu\n",
            mRows.size(),mRows.scalarCount(),static_cast<unsigned long long>(sizeof(AvbdRow)),
            static_cast<unsigned long long>(sizeof(AvbdScalarRowPayload)),static_cast<unsigned long long>(mRows.bytes()),
            static_cast<unsigned long long>(mRows.capacityBytes()));
		std::fprintf(stderr,"[AVBD_ROW_PROGRAM] rows=%u uniqueSupports=%u blockRefs=%u bytes=%llu\n",
			mRows.size(),mRowProgram.rowBlocks.size(),mRowProgram.blockRows.size(),
			static_cast<unsigned long long>(mRowProgram.bytes()));
        std::fprintf(stderr,"[AVBD_FRAME_STORAGE] instances=%u geometries=%u parameters=%u instanceStride=%llu numericBytes=%llu numericCapacityBytes=%llu internCapacityEstimate=%llu\n",
            mElasticFrameMaterials.size(),mElasticFrameMaterials.geometryCount(),mElasticFrameMaterials.parameterCount(),
            static_cast<unsigned long long>(sizeof(AvbdElasticFrameInstance)),
            static_cast<unsigned long long>(mElasticFrameMaterials.bytes()),
            static_cast<unsigned long long>(mElasticFrameMaterials.capacityBytes()),
            static_cast<unsigned long long>(mElasticFrameMaterials.internCapacityEstimate()));
        PxU64 groupAttempts=0,groupUnsupported=0,groupSolved=0,groupAccepted=0,groupProducts=0;
        PxU64 frictionAttempts=0,frictionAccepted=0,sharedContactAttempts=0,sharedContactAccepted=0;
        PxU64 materialEvaluations=0,materialEndpointUses=0;
        for(PxU32 i=0;i<mRigidGroups.size();++i)if(mRigidGroups[i].count>1)
        {
            const AvbdRigidGroup& g=mRigidGroups[i];groupAttempts+=g.attempts;
            groupUnsupported+=g.unsupported;groupSolved+=g.kernelSolved;groupAccepted+=g.accepted;groupProducts+=g.products;
            frictionAttempts+=g.frictionAttempts;frictionAccepted+=g.frictionAccepted;
            sharedContactAttempts+=g.sharedContactAttempts;sharedContactAccepted+=g.sharedContactAccepted;
            materialEvaluations+=g.materialEvaluations;materialEndpointUses+=g.materialEndpointUses;
        }
        groupAttempts+=mStep.retiredGroupStats.attempts;
        groupUnsupported+=mStep.retiredGroupStats.unsupported;
        groupSolved+=mStep.retiredGroupStats.kernelSolved;
        groupAccepted+=mStep.retiredGroupStats.accepted;
        groupProducts+=mStep.retiredGroupStats.products;
        frictionAttempts+=mStep.retiredGroupStats.frictionAttempts;
        frictionAccepted+=mStep.retiredGroupStats.frictionAccepted;
        sharedContactAttempts+=mStep.retiredGroupStats.sharedContactAttempts;
        sharedContactAccepted+=mStep.retiredGroupStats.sharedContactAccepted;
        materialEvaluations+=mStep.retiredGroupStats.materialEvaluations;
        materialEndpointUses+=mStep.retiredGroupStats.materialEndpointUses;
        std::fprintf(stderr,"[AVBD_GROUP_KERNEL] implementation=shared-linear-response attempts=%llu unsupported=%llu solved=%llu accepted=%llu products=%llu frictionAttempts=%llu frictionAccepted=%llu sharedContactAttempts=%llu sharedContactAccepted=%llu\n",
            static_cast<unsigned long long>(groupAttempts),static_cast<unsigned long long>(groupUnsupported),
            static_cast<unsigned long long>(groupSolved),static_cast<unsigned long long>(groupAccepted),static_cast<unsigned long long>(groupProducts),
            static_cast<unsigned long long>(frictionAttempts),static_cast<unsigned long long>(frictionAccepted),
            static_cast<unsigned long long>(sharedContactAttempts),static_cast<unsigned long long>(sharedContactAccepted));
        std::fprintf(stderr,"[AVBD_MATERIAL_ASSEMBLY] evaluations=%llu endpointUses=%llu\n",
            static_cast<unsigned long long>(materialEvaluations),static_cast<unsigned long long>(materialEndpointUses));
		// A witness AL row is not a complete convex-feature certificate.
		// Expose coverage separately from the existence of OGC helper code.
		PxU32 npEndViolations=0;double npEndViolation=0;
		for(PxU32 r=0;r<mNpContactRowCount;++r)
		{
			const AvbdRowConstRef& row=mRows[r];
			if(row.header().kernelId!=eAVBD_ROW_KERNEL_NP_CONTACT) continue;
			const AvbdNpDomain& domain=mNpDomains[row.header().geometryIndex-1];
			const PxTransform poses[2]={row.header().bodyA<mBodies.size()?PxTransform(mBodies[row.header().bodyA].p,mBodies[row.header().bodyA].q):domain.fixedPose[0],
				row.header().bodyB<mBodies.size()?PxTransform(mBodies[row.header().bodyB].p,mBodies[row.header().bodyB].q):domain.fixedPose[1]};
			if(!avbdNpDomainPath(domain,poses,poses)) ++npEndViolations;
			npEndViolation=PxMax(npEndViolation,-avbdNpGap(domain,poses));
		}
		std::fprintf(stderr,"[AVBD_CONTACT_COVERAGE] npRows=%u rigidCertificateBodies=%u rigidCertificatePlanes=%u triangleDomains=%u npCompleteSupportDomains=%u npUnsupported=%u npPolicyExcluded=%u npUnseparated=%u npDomainMode=%u npCertificateEndViolations=%u npCertificateEndMaxViolation=%.9g npAxisRecovered=%u completeRigidNpOgc=partial_np_pairs\n",
			mNpContactRowCount,0u,0u,0u,
			mNpDomains.size(),mNpDomainUnsupported,mNpDomainPolicyExcluded,mNpDomainUnseparated,mNpDomainMode,npEndViolations,npEndViolation,mNpDomainAxisRecovered);
	}
	PxBaseTask *continuation = mStep.continuation;
	mStep.continuation = NULL;
	if (continuation)
		continuation->removeReference();
}

void AvbdDynamicsContext::applyHeterogeneousContactMaterial()
{
    // Budget contacts already own Coulomb in the shared primal/dual.
    // The legacy velocity material pass would apply it a second time.
    if(mStep.localBudget || mRows.size()==mNpContactRowCount)return;
	if(mBlocks.empty())
		return;

	// Position OGC supplies the physical contact potential.  Its accepted
	// separating velocity is therefore retained; the material solve only adds
	// unilateral impulses when a row is still closing.  Projecting every active
	// row to an equality target turns a resting manifold into an adhesive weld.
	mHeterogeneousMaterialImpulseScratch.resize(mRows.size());
	for(PxU32 rowIndex = 0; rowIndex < mRows.size(); ++rowIndex)
		mHeterogeneousMaterialImpulseScratch[rowIndex] = PxVec3(0.0f);

	auto rowAxisKinematics = [this](const AvbdRowConstRef& row, const PxVec3& axis,
		PxReal& velocity, PxReal& response)
	{
		velocity = 0.0f;
		response = 0.0f;
		if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
		{
			const auto value=avbdContactFrameRow(row,mRowSupports,mBlocks,mBodies,contactGeometry(),PX_MAX_U32,NULL,&axis);
			if(!value.valid) { PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION,PX_FL,"AVBD contact frame invalid during material evaluation");return; }
			avbdForEachContactFrameBlock(row,mRowSupports,[&](PxU32 block)
			{
				const AvbdSoftParticle& particle=*mBlocks[block].particle;
				const PxVec3 j=avbdContactFrameGradient(row,mRowSupports,contactGeometry(),value,block);
				velocity+=j.dot(particle.vel);response+=particle.invMass*j.magnitudeSquared();
			});
			return;
		}
		for(PxU32 supportIndex = row.header().supportStart;
			supportIndex < row.header().supportStart + row.header().supportCount; ++supportIndex)
		{
			if(supportIndex >= mRowSupports.size()) break;
			const AvbdRowSupport& support = mRowSupports[supportIndex];
			if(support.block >= mBlocks.size()) continue;
			const AvbdBlock& block = mBlocks[support.block];
			const PxVec3 linearJacobian = axis * support.weight;
			if(block.type == eAVBD_BLOCK_PARTICLE3)
			{
				if(!block.particle) continue;
				const AvbdSoftParticle& particle = *block.particle;
				velocity += linearJacobian.dot(particle.vel);
				response += particle.invMass * linearJacobian.magnitudeSquared();
			}
			else if(block.type == eAVBD_BLOCK_RIGID6 &&
				block.objectIndex < mBodies.size())
			{
				const AvbdSolverBody& body = mBodies[block.objectIndex];
				if(!body.core || body.invMass <= 0.0f || body.isKinematic) continue;
				const PxVec3 arm = row.header().geometryIndex ?
                    avbdHeterogeneousPlaneOppositePoint(row,mRowSupports,mBlocks,mBodies)-body.p :
                    body.q.rotate(support.localPoint);
				const PxVec3 angularJacobian = arm.cross(linearJacobian);
				velocity += linearJacobian.dot(body.outputLinVel) +
					angularJacobian.dot(body.outputAngVel);
				response += body.invMass * linearJacobian.magnitudeSquared() +
					angularJacobian.dot(avbdWorldInvInertiaMul(body,
						angularJacobian));
			}
		}
	};
	auto rowCrossResponse = [this](const AvbdRowConstRef& row, const PxVec3& axis0,
		const PxVec3& axis1)
	{
		PxReal response = 0.0f;
		if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
		{
			const auto a=avbdContactFrameRow(row,mRowSupports,mBlocks,mBodies,contactGeometry(),PX_MAX_U32,NULL,&axis0);
			const auto b=avbdContactFrameRow(row,mRowSupports,mBlocks,mBodies,contactGeometry(),PX_MAX_U32,NULL,&axis1);
			if(!a.valid || !b.valid) { PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION,PX_FL,"AVBD contact frame invalid during material response");return response; }
			avbdForEachContactFrameBlock(row,mRowSupports,[&](PxU32 block)
			{
				const PxVec3 j0=avbdContactFrameGradient(row,mRowSupports,contactGeometry(),a,block);
				const PxVec3 j1=avbdContactFrameGradient(row,mRowSupports,contactGeometry(),b,block);
				response+=mBlocks[block].particle->invMass*j0.dot(j1);
			});
			return response;
		}
		for(PxU32 supportIndex = row.header().supportStart;
			supportIndex < row.header().supportStart + row.header().supportCount; ++supportIndex)
		{
			if(supportIndex >= mRowSupports.size()) break;
			const AvbdRowSupport& support = mRowSupports[supportIndex];
			if(support.block >= mBlocks.size()) continue;
			const AvbdBlock& block = mBlocks[support.block];
			const PxVec3 linear0 = axis0 * support.weight;
			const PxVec3 linear1 = axis1 * support.weight;
			if(block.type == eAVBD_BLOCK_PARTICLE3)
			{
				if(!block.particle) continue;
				response += block.particle->invMass *
					linear0.dot(linear1);
			}
			else if(block.type == eAVBD_BLOCK_RIGID6 &&
				block.objectIndex < mBodies.size())
			{
				const AvbdSolverBody& body = mBodies[block.objectIndex];
				if(!body.core || body.invMass <= 0.0f || body.isKinematic) continue;
				const PxVec3 arm = row.header().geometryIndex ?
                    avbdHeterogeneousPlaneOppositePoint(row,mRowSupports,mBlocks,mBodies)-body.p :
                    body.q.rotate(support.localPoint);
				const PxVec3 angular0 = arm.cross(linear0);
				const PxVec3 angular1 = arm.cross(linear1);
				response += body.invMass * linear0.dot(linear1) +
					angular0.dot(avbdWorldInvInertiaMul(body, angular1));
			}
		}
		return response;
	};
	auto applyRowAxisImpulse = [this](const AvbdRowConstRef& row, const PxVec3& axis,
		PxReal impulse)
	{
		if(!PxIsFinite(impulse) || PxAbs(impulse) <= 1.0e-12f) return;
		if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
		{
			const auto value=avbdContactFrameRow(row,mRowSupports,mBlocks,mBodies,contactGeometry(),PX_MAX_U32,NULL,&axis);
			if(!value.valid) { PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION,PX_FL,"AVBD contact frame invalid during material impulse");return; }
			avbdForEachContactFrameBlock(row,mRowSupports,[&](PxU32 block)
			{
				AvbdSoftParticle& particle=*mBlocks[block].particle;
				particle.vel+=avbdContactFrameGradient(row,mRowSupports,contactGeometry(),value,block)*(particle.invMass*impulse);
			});
			return;
		}
		for(PxU32 supportIndex = row.header().supportStart;
			supportIndex < row.header().supportStart + row.header().supportCount; ++supportIndex)
		{
			if(supportIndex >= mRowSupports.size()) break;
			const AvbdRowSupport& support = mRowSupports[supportIndex];
			if(support.block >= mBlocks.size()) continue;
			const AvbdBlock& block = mBlocks[support.block];
			const PxVec3 linearJacobian = axis * support.weight;
			if(block.type == eAVBD_BLOCK_PARTICLE3)
			{
				if(!block.particle) continue;
				block.particle->vel += linearJacobian *
					(block.particle->invMass * impulse);
			}
			else if(block.type == eAVBD_BLOCK_RIGID6 &&
				block.objectIndex < mBodies.size())
			{
				AvbdSolverBody& body = mBodies[block.objectIndex];
				if(!body.core || body.invMass <= 0.0f || body.isKinematic) continue;
				const PxVec3 arm = row.header().geometryIndex ?
                    avbdHeterogeneousPlaneOppositePoint(row,mRowSupports,mBlocks,mBodies)-body.p :
                    body.q.rotate(support.localPoint);
				body.outputLinVel += linearJacobian *
					(body.invMass * impulse);
				body.outputAngVel += avbdWorldInvInertiaMul(body,
					arm.cross(linearJacobian)) * impulse;
			}
		}
	};
	// Position OGC owns the heterogeneous normal.  A velocity Signorini on
	// the same rows is the shell/core weld: multi-point vn=0 dumps elastic
	// recovery on a compressed manifold.  Coulomb still consumes the dual
	// load on the offset shell.
	for(PxU32 sweep = 0; sweep < AVBD_MATERIAL_MAX_ITERS; ++sweep)
	{
		PxReal maximumImpulseDelta = 0.0f;
		bool appliedImpulse = false;
		for(PxU32 rowIndex = mNpContactRowCount; rowIndex < mRows.size(); ++rowIndex)
		{
			const AvbdRowConstRef& row = mRows[rowIndex];
			if((row.header().flags & (eAVBD_ROW_CONTACT | eAVBD_ROW_SOFT_PAIR |
				eAVBD_ROW_INACTIVE)) !=
				(eAVBD_ROW_CONTACT | eAVBD_ROW_SOFT_PAIR) || !row.header().supportCount ||
				row.scalar().mu <= 0.0f)
				continue;
			PxVec3 tangent0=row.scalar().t0, tangent1=row.scalar().t1;
			if(row.header().kernelId==eAVBD_ROW_KERNEL_CONTACT_FRAME)
				// Frame kernels consume local coordinate axes, including their
				// frame derivatives, rather than world-axis witness impulses.
				avbdOrthonormalBasis(mContactFrameGeometry[row.header().geometryIndex-1].normalLocal,tangent0,tangent1);
            else if(row.header().geometryIndex)
                avbdOrthonormalBasis(avbdHeterogeneousNormal(row,mBodies,contactGeometry(),mRowSupports,mBlocks),tangent0,tangent1);
			// Coulomb rides the published OGC load, which is nonzero inside
			// the shell at rest.  It does not get a second vn=0 plane.
			if(row.scalar().lambda >= 0.0f)
				continue;
			PxVec3& accumulated = mHeterogeneousMaterialImpulseScratch[rowIndex];
			const PxReal normalCapacity = PxMax(accumulated.x,
				PxAbs(row.scalar().lambda) * mDt);
			if(normalCapacity <= 0.0f)
				continue;
			PxReal velocity0 = 0.0f, response00 = 0.0f;
			PxReal velocity1 = 0.0f, response11 = 0.0f;
			rowAxisKinematics(row, tangent0, velocity0, response00);
			rowAxisKinematics(row, tangent1, velocity1, response11);
			const PxReal response01 = rowCrossResponse(row, tangent0, tangent1);
			const PxReal determinant = response00 * response11 -
				response01 * response01;
			if(!PxIsFinite(determinant) || determinant <= 1.0e-12f)
				continue;
			const PxReal old0 = accumulated.y;
			const PxReal old1 = accumulated.z;
			PxReal next0 = old0 + (-response11 * velocity0 +
				response01 * velocity1) / determinant;
			PxReal next1 = old1 + (response01 * velocity0 -
				response00 * velocity1) / determinant;
			// Coulomb capacity is the dual-published OGC load.  The offset
			// shell is not converted into an extra vn=0 plane.
			avbdProjectCoulomb(normalCapacity, row.scalar().mu, next0, next1);
			const PxReal delta0 = next0 - old0;
			const PxReal delta1 = next1 - old1;
			accumulated.y = next0;
			accumulated.z = next1;
			applyRowAxisImpulse(row, tangent0, delta0);
			applyRowAxisImpulse(row, tangent1, delta1);
			const PxReal deltaMagnitude = PxSqrt(delta0 * delta0 + delta1 * delta1);
			maximumImpulseDelta = PxMax(maximumImpulseDelta, deltaMagnitude);
			appliedImpulse |= deltaMagnitude > 1.0e-8f;
		}
		if(!appliedImpulse || maximumImpulseDelta <= 1.0e-8f)
			break;
	}

}

void AvbdDynamicsContext::clearIslandSurfaceCache()
{
	for(PxU32 i = 0; i < mIslandSurfaceCache.size(); ++i)
	{
		PX_FREE(mIslandSurfaceCache[i].triangles);
		PX_FREE(mIslandSurfaceCache[i].nodalRest);
        PX_DELETE(mIslandSurfaceCache[i].membrane);
		PX_FREE(mIslandSurfaceCache[i].bends);
		PX_FREE(mIslandSurfaceCache[i].particles);


		PX_FREE(mIslandSurfaceCache[i].edges);
	}
	mIslandSurfaceCache.forceSize_Unsafe(0);
}

AvbdIslandSurfaceCache* AvbdDynamicsContext::getIslandSurfaceCache(DeformableSurface* surface)
{
	for(PxU32 i = 0; i < mIslandSurfaceCache.size(); ++i)
	{
		if(mIslandSurfaceCache[i].surface == surface)
			return &mIslandSurfaceCache[i];
	}
	return NULL;
}

void AvbdDynamicsContext::pruneIslandSurfaceCache()
{
	PxU32 w = 0;
	for(PxU32 i = 0; i < mIslandSurfaceCache.size(); ++i)
	{
		if(mIslandSurfaceCache[i].liveStamp == mIslandSurfaceLiveStamp)
		{
			if(w != i)
				mIslandSurfaceCache[w] = mIslandSurfaceCache[i];
			++w;
		}
		else
		{
			PX_FREE(mIslandSurfaceCache[i].triangles);
			PX_FREE(mIslandSurfaceCache[i].nodalRest);
        PX_DELETE(mIslandSurfaceCache[i].membrane);
			PX_FREE(mIslandSurfaceCache[i].bends);
			PX_FREE(mIslandSurfaceCache[i].particles);


			PX_FREE(mIslandSurfaceCache[i].edges);
		}
	}
	mIslandSurfaceCache.forceSize_Unsafe(w);
}
