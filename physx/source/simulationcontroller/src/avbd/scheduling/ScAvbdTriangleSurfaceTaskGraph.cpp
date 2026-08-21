// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

bool AvbdCpuSoftScene::ensureRigidTriangleSurfaceContactTaskPool(PxU32 requiredChildTasks,
	Scene& owner, Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
	mRigidTriangleSurfaceContactTasks.reserve(requiredSlots);
	mRigidTriangleSurfaceContactFinishTasks.reserve(requiredSlots);
	mFreeRigidTriangleSurfaceContactTaskIndices.reserve(requiredSlots);
	mFreeRigidTriangleSurfaceContactFinishTaskIndices.reserve(requiredSlots);
	mRigidTriangleSurfaceContactTaskOutputs.reserve(requiredChildTasks);
	mRigidTriangleSurfaceSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
	mRigidTriangleSurfaceFeatureContactTaskOutputs.reserve(requiredChildTasks);
	mRigidTriangleSurfaceContactTaskStats.reserve(requiredChildTasks);
	while(mRigidTriangleSurfaceContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidTriangleSurfaceContactTasks.size();
		mRigidTriangleSurfaceContactTasks.pushBack(
			PX_NEW(RigidTriangleSurfaceContactTask)(mContextId, *this, index));
		mFreeRigidTriangleSurfaceContactTaskIndices.pushBack(index);
	}
	while(mRigidTriangleSurfaceContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidTriangleSurfaceContactFinishTasks.size();
		mRigidTriangleSurfaceContactFinishTasks.pushBack(
			PX_NEW(RigidTriangleSurfaceContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeRigidTriangleSurfaceContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::RigidTriangleSurfaceContactTask* AvbdCpuSoftScene::acquireRigidTriangleSurfaceContactTask()
{
	PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
	if(mFreeRigidTriangleSurfaceContactTaskIndices.empty()) return NULL;
	const PxU32 index = mFreeRigidTriangleSurfaceContactTaskIndices.back();
	mFreeRigidTriangleSurfaceContactTaskIndices.popBack();
	return mRigidTriangleSurfaceContactTasks[index];
}

AvbdCpuSoftScene::RigidTriangleSurfaceContactFinishTask*
	AvbdCpuSoftScene::acquireRigidTriangleSurfaceContactFinishTask()
{
	PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
	if(mFreeRigidTriangleSurfaceContactFinishTaskIndices.empty()) return NULL;
	const PxU32 index =
		mFreeRigidTriangleSurfaceContactFinishTaskIndices.back();
	mFreeRigidTriangleSurfaceContactFinishTaskIndices.popBack();
	return mRigidTriangleSurfaceContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleRigidTriangleSurfaceContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
	PX_ASSERT(index < mRigidTriangleSurfaceContactTasks.size());
	mFreeRigidTriangleSurfaceContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleRigidTriangleSurfaceContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
	PX_ASSERT(index < mRigidTriangleSurfaceContactFinishTasks.size());
	mFreeRigidTriangleSurfaceContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneRigidTriangleSurfaceContactTask(
	PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& taskGraphContext)
{
	const PxU32 taskCount = getRigidTriangleSurfaceContactTaskCount(
		dispatcherWorkers);
	if(!mRigidTriangleSurfaceContactTransactionPending || !continuation ||
		taskCount == 0 || !ensureRigidTriangleSurfaceContactTaskPool(
			taskCount, owner, taskGraphContext)) return false;
	{
		PxMutex::ScopedLock lock(mRigidTriangleSurfaceContactTaskPoolMutex);
		if(mFreeRigidTriangleSurfaceContactTaskIndices.size() < taskCount ||
			mFreeRigidTriangleSurfaceContactFinishTaskIndices.empty()) return false;
	}
	const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
		mRigidTriangleSurfaces.size();
	if(maxContactCount64 > PX_MAX_U32 / 2u) return false;
	PxU32 maxTriangleCandidateCount = 0;
	PxU32 maxEdgeCandidateCount = 0;
	PxU32 maxVertexCandidateCount = 0;
	for(PxU32 index = 0; index < mRigidTriangleSurfaces.size(); ++index)
	{
		maxTriangleCandidateCount = PxMax(maxTriangleCandidateCount,
			mRigidTriangleSurfaces[index].triangleBvhTriangleIndices.size());
		maxEdgeCandidateCount = PxMax(maxEdgeCandidateCount,
			mRigidTriangleSurfaces[index].edges.size());
		maxVertexCandidateCount = PxMax(maxVertexCandidateCount,
			mRigidTriangleSurfaces[index].vertices.size());
	}
	mContacts.reserve(PxU32(maxContactCount64 * 2u));
	const PxU32 featurePlanCount =
		mRigidTriangleSurfaceFeaturePlan.items.size();
	const bool useFeaturePlanRoundRobin = featurePlanCount > 0 &&
		Dy::avbdUseRigidTriangleSurfaceFeatureRoundRobinTaskPlan();
	const bool useFeaturePlanRowPrivateOutputs = featurePlanCount > 0 &&
		(useFeaturePlanRoundRobin ||
			Dy::avbdUseRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan());
	// P5.41 promotes P5.39's conservative cull only after this already
	// opt-in Scene task route has been admitted. The legacy disable is
	// for controlled A/B and must not broaden the serial/global policy.
	const bool useFeatureDiscreteBodyLocalBoundsCull =
		featurePlanCount > 0 &&
		!Dy::avbdDisableRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull();
	// P5.35's accepted triangle task caches the P5.31-proven exact
	// forward-owner result by default. The overall Scene task route is
	// still opt-in; the disable switch exists only for controlled legacy
	// comparisons and must not be treated as a default policy branch.
	const bool useFeatureForwardOwnerResultCache = featurePlanCount > 0 &&
		!Dy::avbdDisableRigidTriangleSurfaceFeatureForwardOwnerResultCache();
	mRigidTriangleSurfaceContactTaskOutputs.resize(taskCount);
	mRigidTriangleSurfaceSweptSdfContactTaskOutputs.resize(taskCount);
	mRigidTriangleSurfaceFeatureContactTaskOutputs.resize(taskCount);
	if(useFeaturePlanRowPrivateOutputs)
		mRigidTriangleSurfaceFeatureContactPlanOutputs.resize(featurePlanCount);
	else
		mRigidTriangleSurfaceFeatureContactPlanOutputs.clear();
	mRigidTriangleSurfaceContactTaskStats.resize(taskCount);
	mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan =
		useFeaturePlanRowPrivateOutputs;
	mRigidTriangleSurfaceFeatureRoundRobinTaskPlan = useFeaturePlanRoundRobin;
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	const PxU32 featureRowsPerTask = featurePlanCount == 0 ? 0 :
		(featurePlanCount + taskCount - 1) / taskCount;
	for(PxU32 index = 0; index < taskCount; ++index)
	{
		const PxU32 begin = index * particlesPerTask;
		const PxU32 end = PxMin(begin + particlesPerTask, mParticles.size());
		mRigidTriangleSurfaceContactTaskOutputs[index].clear();
		mRigidTriangleSurfaceContactTaskOutputs[index].reserve(
			(end - begin) * mRigidTriangleSurfaces.size());
		mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].clear();
		mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index].reserve(
			(end - begin) * mRigidTriangleSurfaces.size());
		const PxU32 featurePlanBegin = useFeaturePlanRoundRobin ? index :
			PxMin(index * featureRowsPerTask, featurePlanCount);
		const PxU32 featurePlanEnd = useFeaturePlanRoundRobin ?
			featurePlanCount : PxMin(
				featurePlanBegin + featureRowsPerTask, featurePlanCount);
		const PxU32 featurePlanStride = useFeaturePlanRoundRobin ?
			taskCount : 1u;
		PxU64 featureContactCapacity = 0;
		for(PxU32 planIndex = featurePlanBegin;
			planIndex < featurePlanEnd; planIndex += featurePlanStride)
		{
			const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& workItem =
				mRigidTriangleSurfaceFeaturePlan.items[planIndex];
			PX_ASSERT(workItem.bodyIndex < mBodies.size() &&
				workItem.surfaceIndex < mRigidTriangleSurfaces.size());
			const Dy::AvbdRigidTriangleSurface& surface =
				mRigidTriangleSurfaces[workItem.surfaceIndex];
			const PxU64 primitivePairWorkItems = workItem.family ==
				Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE
				? PxU64(workItem.primitiveEnd - workItem.primitiveBegin) *
					surface.edges.size()
				: PxU64(workItem.primitiveEnd - workItem.primitiveBegin) *
					surface.vertices.size();
			if(primitivePairWorkItems > PX_MAX_U32) return false;
			featureContactCapacity += primitivePairWorkItems;
			if(featureContactCapacity > PX_MAX_U32) return false;
			if(useFeaturePlanRowPrivateOutputs)
			{
				PxArray<Dy::AvbdSoftContact>& featureOutput =
					mRigidTriangleSurfaceFeatureContactPlanOutputs[planIndex];
				featureOutput.clear();
				featureOutput.reserve(PxU32(primitivePairWorkItems));
			}
		}
		if(!useFeaturePlanRowPrivateOutputs)
		{
			mRigidTriangleSurfaceFeatureContactTaskOutputs[index].clear();
			mRigidTriangleSurfaceFeatureContactTaskOutputs[index].reserve(
				PxU32(featureContactCapacity));
		}
		mRigidTriangleSurfaceContactTaskStats[index] =
			Dy::AvbdSoftCollisionStats();
	}
	RigidTriangleSurfaceContactFinishTask* const finishTask =
		acquireRigidTriangleSurfaceContactFinishTask();
	if(!finishTask) return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordRigidTriangleSurfaceContactTasksSubmitted(taskCount);
	for(PxU32 index = 0; index < taskCount; ++index)
	{
		const PxU32 begin = index * particlesPerTask;
		const PxU32 end = PxMin(begin + particlesPerTask, mParticles.size());
		const PxU32 featurePlanBegin = useFeaturePlanRoundRobin ? index :
			PxMin(index * featureRowsPerTask, featurePlanCount);
		const PxU32 featurePlanEnd = useFeaturePlanRoundRobin ?
			featurePlanCount : PxMin(
				featurePlanBegin + featureRowsPerTask, featurePlanCount);
		const PxU32 featurePlanStride = useFeaturePlanRoundRobin ?
			taskCount : 1u;
		PxU32 forwardOwnerResultCacheSurfaceCount = 0;
		if(useFeatureForwardOwnerResultCache)
		{
			for(PxU32 planIndex = featurePlanBegin;
				planIndex < featurePlanEnd; planIndex += featurePlanStride)
			{
				const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& item =
					mRigidTriangleSurfaceFeaturePlan.items[planIndex];
				if(item.phase !=
					Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT)
					continue;
				bool alreadyCounted = false;
				for(PxU32 previousPlanIndex = featurePlanBegin;
					previousPlanIndex < planIndex;
					previousPlanIndex += featurePlanStride)
				{
					const Dy::AvbdRigidTriangleSurfaceFeatureWorkItem& previous =
						mRigidTriangleSurfaceFeaturePlan.items[
							previousPlanIndex];
					if(previous.phase ==
						Dy::AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT &&
						previous.surfaceIndex == item.surfaceIndex)
					{
						alreadyCounted = true;
						break;
					}
				}
				if(!alreadyCounted)
					++forwardOwnerResultCacheSurfaceCount;
			}
		}
		const PxU64 forwardOwnerResultCacheTaskCapacity64 =
			PxU64(mParticles.size()) *
			forwardOwnerResultCacheSurfaceCount;
		if(forwardOwnerResultCacheTaskCapacity64 > PX_MAX_U32)
			return false;
		RigidTriangleSurfaceContactTask* const task =
			acquireRigidTriangleSurfaceContactTask();
		if(!task)
		{
			recycleRigidTriangleSurfaceContactFinishTask(
				finishTask->getPoolIndex());
			return false;
		}
		task->reserveBvhCandidateScratch(maxTriangleCandidateCount,
				maxEdgeCandidateCount, maxVertexCandidateCount,
				0,
				PxU32(forwardOwnerResultCacheTaskCapacity64),
				forwardOwnerResultCacheSurfaceCount);
		task->configure(mParticles.begin(), mParticles.size(), begin, end,
			mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
			mBodies.begin(), mBodies.size(),
			mRigidTriangleSurfaceContactTaskOutputs[index],
			mRigidTriangleSurfaceSweptSdfContactTaskOutputs[index],
			mRigidTriangleSurfaceFeaturePlan,
			featurePlanBegin, featurePlanEnd,
			mRigidTriangleSurfaceFeatureContactTaskOutputs[index],
			mContactParams.contactRadius,
			mCollisionStatsEnabled ?
				&mRigidTriangleSurfaceContactTaskStats[index] : NULL);
		if(useFeatureDiscreteBodyLocalBoundsCull)
			task->configureDiscreteBodyLocalBoundsCull();
		if(useFeatureForwardOwnerResultCache &&
			forwardOwnerResultCacheSurfaceCount > 0)
			task->configureForwardOwnerResultCache();
		if(useFeaturePlanRoundRobin)
			task->configureFeaturePlanRoundRobin(
				mRigidTriangleSurfaceFeatureContactPlanOutputs,
				index, taskCount);
		else if(useFeaturePlanRowPrivateOutputs)
			task->configureFeaturePlanRowPrivateOutputs(
				mRigidTriangleSurfaceFeatureContactPlanOutputs);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}

bool AvbdCpuSoftScene::canUseRigidTriangleSurfaceContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseRigidTriangleSurfaceContactTaskFanIn() ||
		mBodies.size() != 1 || mRigidTriangleSurfaces.empty() ||
		!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
		!mRigidSpheres.empty() || !mRigidCapsules.empty() ||
		!mRigidConvexes.empty() ||
		mSelfCollisionEnabled.size() != mBodies.size())
		return false;
	for(PxU32 index = 0; index < mRigidTriangleSurfaces.size(); ++index)
		if(mRigidTriangleSurfaces[index].targetKind !=
			Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
			return false;
	for(PxU32 index = 0; index < mSelfCollisionEnabled.size(); ++index)
		if(mSelfCollisionEnabled[index])
			return false;
	return true;
}

PxU32 AvbdCpuSoftScene::getRigidTriangleSurfaceContactTaskCount(PxU32 dispatcherWorkers) const
{
	static const PxU32 eMIN_PARTICLES_PER_RIGID_TRIANGLE_TASK = 128;
	if(!canUseRigidTriangleSurfaceContactTaskTransaction() ||
		dispatcherWorkers < 2 || mParticles.size() <
		eMIN_PARTICLES_PER_RIGID_TRIANGLE_TASK)
		return 0;
	const PxU32 maxTasksByParticles = (mParticles.size() +
		eMIN_PARTICLES_PER_RIGID_TRIANGLE_TASK - 1) /
		eMIN_PARTICLES_PER_RIGID_TRIANGLE_TASK;
	return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
		mParticles.size());
}

bool AvbdCpuSoftScene::beginRigidTriangleSurfaceContactTaskTransaction()
{
	if(!canUseRigidTriangleSurfaceContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending ||
		mRigidBoxSdfContactTransactionPending ||
		mRigidSphereSdfContactTransactionPending ||
		mRigidCapsuleSdfContactTransactionPending ||
		mRigidConvexSdfContactTransactionPending ||
		mRigidTriangleSurfaceContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, 0, false, 0, 0, 0, 0,
		mRigidTriangleSurfaces.size(), mBodies.size(),
		mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
	// P5.17d builds feature identity once in the parent. The child leaf
	// receives only immutable row intervals, never a mutable broadphase.
	Dy::avbdBuildRigidTriangleSurfaceOGCFeaturePlan(
		mBodies.begin(), mBodies.size(),
		mRigidTriangleSurfaces.size(),
		mRigidTriangleSurfaceFeaturePlan);
	mRigidTriangleSurfaceContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeRigidTriangleSurfaceContactTaskTransaction()
{
	PX_ASSERT(mRigidTriangleSurfaceContactTransactionPending);
	const PxU32 rigidStart = mContacts.size();
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidTriangleSurfaceContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidTriangleSurfaceContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0; contactIndex < source.size();
			++contactIndex)
			mContacts.pushBack(source[contactIndex]);
		if(mCollisionStatsEnabled && taskIndex <
			mRigidTriangleSurfaceContactTaskStats.size())
			mLastCollisionStats.accumulate(
				mRigidTriangleSurfaceContactTaskStats[taskIndex]);
	}
	// P5.16b's only triangle-SDF family merge point: every current-SDF
	// range precedes every swept-SDF range. Child tasks may evaluate both
	// leaves back-to-back, but contacts must never be interleaved by task.
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidTriangleSurfaceSweptSdfContactTaskOutputs.size();
		++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidTriangleSurfaceSweptSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0; contactIndex < source.size();
			++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.rigidParticleTriangleSurfaceTests +=
			PxU64(mParticles.size()) * mRigidTriangleSurfaces.size();
	// P5.17d's only feature-family merge point. P5.27's opt-in
	// round-robin candidate gives each immutable plan row a private output,
	// then restores the same canonical row order here. The accepted route
	// retains its contiguous task-range merge.
	const PxArray<PxArray<Dy::AvbdSoftContact> >& featureOutputs =
		mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan ?
			mRigidTriangleSurfaceFeatureContactPlanOutputs :
			mRigidTriangleSurfaceFeatureContactTaskOutputs;
	for(PxU32 taskIndex = 0; taskIndex < featureOutputs.size();
		++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			featureOutputs[taskIndex];
		for(PxU32 contactIndex = 0; contactIndex < source.size();
			++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedRigidContacts +=
			mContacts.size() - rigidStart;
	completeComponentContactRedetection();
	mRigidTriangleSurfaceContactTransactionPending = false;
	mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan = false;
	mRigidTriangleSurfaceFeatureRoundRobinTaskPlan = false;
}

bool AvbdCpuSoftScene::completeStandaloneRigidTriangleSurfaceContactTask(
	PxReal dt, Dy::AvbdDynamicsContext& /*taskGraphContext*/,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	nextLayerReady = false;
	nextWorldPlaneContactTaskReady = false;
	nextRigidBoxSdfContactTaskReady = false;
	nextRigidSphereSdfContactTaskReady = false;
	if(!mStandaloneComponentSolvePrepared ||
		!mRigidTriangleSurfaceContactTransactionPending) return false;
	completeRigidTriangleSurfaceContactTaskTransaction();
	if(!mStandaloneComponentStepState.completePendingRedetection()) return false;
	const Dy::AvbdSoftBodyStepAdvanceResult result =
		advanceStandaloneComponentStateWithSceneRedetection(
			true, &nextWorldPlaneContactTaskReady,
			true, &nextRigidBoxSdfContactTaskReady,
			true, &nextRigidSphereSdfContactTaskReady);
	if(nextWorldPlaneContactTaskReady || nextRigidBoxSdfContactTaskReady ||
		nextRigidSphereSdfContactTaskReady)
		return true;
	if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
	{
		nextLayerReady = true;
		return true;
	}
	if(result != Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
		return false;
	finishStandaloneComponentSolve(dt);
	return true;
}

bool AvbdCpuSoftScene::finishStandaloneRigidTriangleSurfaceContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mRigidTriangleSurfaceContactTransactionPending) return false;
	mStandaloneTaskGraphTelemetry.
		recordSerialRigidTriangleSurfaceContactFallback();
	mRigidTriangleSurfaceContactTaskOutputs.clear();
	mRigidTriangleSurfaceSweptSdfContactTaskOutputs.clear();
	mRigidTriangleSurfaceFeatureContactTaskOutputs.clear();
	mRigidTriangleSurfaceFeatureContactPlanOutputs.clear();
	mRigidTriangleSurfaceContactTaskStats.clear();
	mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan = false;
	mRigidTriangleSurfaceFeatureRoundRobinTaskPlan = false;
	Dy::avbdDetectSoftRigidTriangleSurface(
		mParticles.begin(), mParticles.size(),
		mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
		mContacts, mContactParams.contactRadius,
		mBodies.begin(), mBodies.size(),
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
	// Submission failure keeps the serial wrapper as the authority. The
	// transaction completion below now only merges child streams, so the
	// fallback explicitly retains the old current -> swept -> feature
	// suffix sequence before parent-only mutable completion.
	Dy::avbdDetectSoftRigidTriangleSurfaceSwept(
		mParticles.begin(), mParticles.size(),
		mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
		mContacts, mContactParams.contactRadius,
		mBodies.begin(), mBodies.size(),
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
	Dy::avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius,
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL, NULL,
		&mWorkspace.contact.rigidTriangleSurfaceForwardOwnerScratch);
	Dy::avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidTriangleSurfaces.begin(), mRigidTriangleSurfaces.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius,
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
	return completeStandaloneRigidTriangleSurfaceContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

} // namespace Sc
} // namespace physx
