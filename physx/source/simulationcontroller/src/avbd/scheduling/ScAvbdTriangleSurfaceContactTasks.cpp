// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	RigidTriangleSurfaceContactTask(
		PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mRange(), mForwardOwnerQueryStamp(0),
	  mForwardOwnerResultCacheStamp(0)
{
}

PxU64 AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	reserveBvhCandidateScratch(PxU32 triangleCapacity, PxU32 edgeCapacity,
		PxU32 vertexCapacity, PxU32 forwardOwnerQueryStampCapacity,
		PxU32 forwardOwnerResultCacheCapacity,
		PxU32 forwardOwnerResultCacheSurfaceSlotCapacity)
{
	const PxU32 oldTriangleCapacity =
		mQueryScratch.triangleBvhQueryCandidates.capacity();
	const PxU32 oldEdgeCapacity =
		mQueryScratch.edgeBvhQueryCandidates.capacity();
	const PxU32 oldVertexCapacity =
		mQueryScratch.vertexBvhQueryCandidates.capacity();
	const PxU32 oldEdgeStampCapacity =
		mQueryScratch.edgeBvhCandidateStamps.capacity();
	const PxU32 oldVertexStampCapacity =
		mQueryScratch.vertexBvhCandidateStamps.capacity();
	const PxU32 oldForwardOwnerStampCapacity =
		mForwardOwnerQueryStamps.capacity();
	const PxU32 oldForwardOwnerResultCacheEntryCapacity =
		mForwardOwnerResultCacheEntries.capacity();
	const PxU32 oldForwardOwnerResultCacheSurfaceSlotCapacity =
		mForwardOwnerResultCacheSurfaceSlots.capacity();
	mQueryScratch.reserve(triangleCapacity, edgeCapacity, vertexCapacity);
	if(forwardOwnerQueryStampCapacity)
		mForwardOwnerQueryStamps.reserve(forwardOwnerQueryStampCapacity);
	if(forwardOwnerResultCacheCapacity)
		mForwardOwnerResultCacheEntries.reserve(
			forwardOwnerResultCacheCapacity);
	if(forwardOwnerResultCacheSurfaceSlotCapacity)
		mForwardOwnerResultCacheSurfaceSlots.reserve(
			forwardOwnerResultCacheSurfaceSlotCapacity);
	const PxU64 queryScratchWordGrowth = PxU64(
		mQueryScratch.triangleBvhQueryCandidates.capacity() -
			oldTriangleCapacity) +
		PxU64(mQueryScratch.edgeBvhQueryCandidates.capacity() -
			oldEdgeCapacity) +
		PxU64(mQueryScratch.vertexBvhQueryCandidates.capacity() -
			oldVertexCapacity) +
		PxU64(mQueryScratch.edgeBvhCandidateStamps.capacity() -
			oldEdgeStampCapacity) +
		PxU64(mQueryScratch.vertexBvhCandidateStamps.capacity() -
			oldVertexStampCapacity) +
		PxU64(mForwardOwnerQueryStamps.capacity() -
			oldForwardOwnerStampCapacity) +
		PxU64(mForwardOwnerResultCacheEntries.capacity() -
			oldForwardOwnerResultCacheEntryCapacity) +
		PxU64(mForwardOwnerResultCacheSurfaceSlots.capacity() -
			oldForwardOwnerResultCacheSurfaceSlotCapacity);
	return queryScratchWordGrowth * sizeof(PxU32);
}

PxU64 AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	getBvhCandidateScratchResidentPayloadBytes() const
{
	const PxU64 queryScratchWords =
		PxU64(mQueryScratch.triangleBvhQueryCandidates.capacity()) +
		PxU64(mQueryScratch.edgeBvhQueryCandidates.capacity()) +
		PxU64(mQueryScratch.vertexBvhQueryCandidates.capacity()) +
		PxU64(mQueryScratch.edgeBvhCandidateStamps.capacity()) +
		PxU64(mQueryScratch.vertexBvhCandidateStamps.capacity()) +
		PxU64(mForwardOwnerQueryStamps.capacity()) +
		PxU64(mForwardOwnerResultCacheEntries.capacity()) +
		PxU64(mForwardOwnerResultCacheSurfaceSlots.capacity());
	return queryScratchWords * sizeof(PxU32);
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts,
	PxArray<Dy::AvbdSoftContact>& sweptContacts,
	const Dy::AvbdRigidTriangleSurfaceFeaturePlan& featurePlan,
	PxU32 featurePlanBegin, PxU32 featurePlanEnd,
	PxArray<Dy::AvbdSoftContact>& featureContacts, PxReal margin,
	Dy::AvbdSoftCollisionStats* collisionStats)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_ASSERT(surfaces && numSurfaces > 0 && bodies && numBodies > 0);
	mRange.particles = particles;
	mRange.numParticles = numParticles;
	mRange.particleBegin = particleBegin;
	mRange.particleEnd = particleEnd;
	mRange.surfaces = surfaces;
	mRange.numSurfaces = numSurfaces;
	mRange.bodies = bodies;
	mRange.numBodies = numBodies;
	mRange.contacts = &contacts;
	mRange.sweptContacts = &sweptContacts;
	mRange.featurePlan = &featurePlan;
	mRange.featurePlanBegin = featurePlanBegin;
	mRange.featurePlanEnd = featurePlanEnd;
	mRange.featureContacts = &featureContacts;
	mRange.featurePlanOutputs = NULL;
	mRange.featurePlanRowPrivateOutputs = false;
	mRange.featurePlanRoundRobin = false;
	mRange.featurePlanTaskIndex = 0;
	mRange.featurePlanTaskCount = 0;
	mRange.queryScratch = &mQueryScratch;
	mRange.forwardOwnerQueryStats = NULL;
	mRange.forwardOwnerResultCache = NULL;
	mRange.discreteQueryStatsEnabled = false;
	mRange.discreteBodyLocalBoundsCullEnabled = false;
	mRange.margin = margin;
	mRange.collisionStats = collisionStats;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	configureFeaturePlanRoundRobin(
		PxArray<PxArray<Dy::AvbdSoftContact> >& featurePlanOutputs,
		PxU32 taskIndex, PxU32 taskCount)
{
	PX_ASSERT(taskCount > 0 && taskIndex < taskCount);
	mRange.featurePlanOutputs = &featurePlanOutputs;
	mRange.featurePlanRowPrivateOutputs = true;
	mRange.featurePlanRoundRobin = true;
	mRange.featurePlanTaskIndex = taskIndex;
	mRange.featurePlanTaskCount = taskCount;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	configureFeaturePlanRowPrivateOutputs(
		PxArray<PxArray<Dy::AvbdSoftContact> >& featurePlanOutputs)
{
	mRange.featurePlanOutputs = &featurePlanOutputs;
	mRange.featurePlanRowPrivateOutputs = true;
	mRange.featurePlanRoundRobin = false;
	mRange.featurePlanTaskIndex = 0;
	mRange.featurePlanTaskCount = 0;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	configureForwardOwnerQueryStats()
{
	const PxU64 requiredCapacity64 =
		PxU64(mRange.numParticles) * mRange.numSurfaces;
	PX_ASSERT(requiredCapacity64 <= PX_MAX_U32 && requiredCapacity64 > 0);
	const PxU32 requiredCapacity = PxU32(requiredCapacity64);
	if(mForwardOwnerQueryStamps.size() != requiredCapacity)
	{
		mForwardOwnerQueryStamps.resize(requiredCapacity);
		for(PxU32 index = 0; index < requiredCapacity; ++index)
			mForwardOwnerQueryStamps[index] = 0;
	}
	++mForwardOwnerQueryStamp;
	if(mForwardOwnerQueryStamp == 0)
	{
		mForwardOwnerQueryStamp = 1;
		for(PxU32 index = 0; index < requiredCapacity; ++index)
			mForwardOwnerQueryStamps[index] = 0;
	}
	mForwardOwnerQueryStats.configure(mForwardOwnerQueryStamps,
		mRange.numParticles, mRange.numSurfaces, mForwardOwnerQueryStamp);
	mRange.forwardOwnerQueryStats = &mForwardOwnerQueryStats;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	configureDiscreteQueryStats()
{
	mRange.discreteQueryStatsEnabled = true;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	configureDiscreteBodyLocalBoundsCull()
{
	mRange.discreteBodyLocalBoundsCullEnabled = true;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::
	configureForwardOwnerResultCache()
{
	PX_ASSERT(mRange.featurePlan && mRange.numSurfaces > 0);
	if(mForwardOwnerResultCacheSurfaceSlots.size() != mRange.numSurfaces)
		mForwardOwnerResultCacheSurfaceSlots.resize(mRange.numSurfaces);
	for(PxU32 surfaceIndex = 0;
		surfaceIndex < mRange.numSurfaces; ++surfaceIndex)
		mForwardOwnerResultCacheSurfaceSlots[surfaceIndex] = PX_MAX_U32;
	const PxU32 planBegin = mRange.featurePlanRoundRobin ?
		mRange.featurePlanTaskIndex : mRange.featurePlanBegin;
	const PxU32 planEnd = mRange.featurePlanRoundRobin ?
		mRange.featurePlan->items.size() : mRange.featurePlanEnd;
	const PxU32 planStride = mRange.featurePlanRoundRobin ?
		mRange.featurePlanTaskCount : 1u;
	PX_ASSERT(planStride > 0);
	PxU32 cachedSurfaceCount = 0;
	for(PxU32 planIndex = planBegin;
		planIndex < planEnd; planIndex += planStride)
	{
		const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& item =
			mRange.featurePlan->items[planIndex];
		if(item.phase !=
			Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT ||
			item.surfaceIndex >= mRange.numSurfaces)
			continue;
		PxU32& slot =
			mForwardOwnerResultCacheSurfaceSlots[item.surfaceIndex];
		if(slot == PX_MAX_U32)
			slot = cachedSurfaceCount++;
	}
	if(cachedSurfaceCount == 0)
	{
		mRange.forwardOwnerResultCache = NULL;
		return;
	}
	const PxU64 requiredCapacity64 =
		PxU64(mRange.numParticles) * cachedSurfaceCount;
	PX_ASSERT(requiredCapacity64 <= PX_MAX_U32);
	const PxU32 requiredCapacity = PxU32(requiredCapacity64);
	if(mForwardOwnerResultCacheEntries.size() != requiredCapacity)
	{
		mForwardOwnerResultCacheEntries.resize(requiredCapacity);
		for(PxU32 index = 0; index < requiredCapacity; ++index)
			mForwardOwnerResultCacheEntries[index] = 0;
	}
	++mForwardOwnerResultCacheStamp;
	if(mForwardOwnerResultCacheStamp > (PX_MAX_U32 >> 1))
	{
		mForwardOwnerResultCacheStamp = 1;
		for(PxU32 index = 0; index < requiredCapacity; ++index)
			mForwardOwnerResultCacheEntries[index] = 0;
	}
	mForwardOwnerResultCache.configure(mForwardOwnerResultCacheEntries,
		mForwardOwnerResultCacheSurfaceSlots, mRange.numParticles,
		mRange.numSurfaces,
		cachedSurfaceCount, mForwardOwnerResultCacheStamp);
	mRange.forwardOwnerResultCache = &mForwardOwnerResultCache;
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::runInternal()
{
	PX_ASSERT(mRange.particles && mRange.surfaces && mRange.bodies &&
		mRange.contacts && mRange.sweptContacts && mRange.featurePlan &&
		mRange.featureContacts && mRange.queryScratch);
	mScene.mStandaloneTaskGraphTelemetry.
		beginRigidTriangleSurfaceContactTask();
	executeAvbdRigidTriangleSurfaceContactRange(
		mRange, NULL, NULL, NULL, NULL);

	mScene.mStandaloneTaskGraphTelemetry.
		endRigidTriangleSurfaceContactTask();
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidTriangleSurfaceContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char*
AvbdCpuSoftScene::RigidTriangleSurfaceContactTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidTriangleSurfaceContact";
}

AvbdCpuSoftScene::RigidTriangleSurfaceContactFinishTask::
	RigidTriangleSurfaceContactFinishTask(
		PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner,
		PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.
		recordRigidTriangleSurfaceContactFanIn();
	mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::RigidTriangleSurfaceContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidTriangleSurfaceContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char*
AvbdCpuSoftScene::RigidTriangleSurfaceContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidTriangleSurfaceContactFinish";
}

} // namespace Sc
} // namespace physx
