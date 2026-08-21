// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

bool AvbdCpuSoftScene::ensureWorldPlaneContactTaskPool(
	PxU32 requiredChildTasks, Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
	mWorldPlaneContactTasks.reserve(requiredSlots);
	mWorldPlaneContactFinishTasks.reserve(requiredSlots);
	mFreeWorldPlaneContactTaskIndices.reserve(requiredSlots);
	mFreeWorldPlaneContactFinishTaskIndices.reserve(requiredSlots);
	mWorldPlaneContactTaskOutputs.reserve(requiredChildTasks);
	while(mWorldPlaneContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mWorldPlaneContactTasks.size();
		mWorldPlaneContactTasks.pushBack(PX_NEW(WorldPlaneContactTask)(
			mContextId, *this, index));
		mFreeWorldPlaneContactTaskIndices.pushBack(index);
	}
	while(mWorldPlaneContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mWorldPlaneContactFinishTasks.size();
		mWorldPlaneContactFinishTasks.pushBack(
			PX_NEW(WorldPlaneContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeWorldPlaneContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::WorldPlaneContactTask*
AvbdCpuSoftScene::acquireWorldPlaneContactTask()
{
	PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
	if(mFreeWorldPlaneContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeWorldPlaneContactTaskIndices.back();
	mFreeWorldPlaneContactTaskIndices.popBack();
	return mWorldPlaneContactTasks[index];
}

AvbdCpuSoftScene::WorldPlaneContactFinishTask*
AvbdCpuSoftScene::acquireWorldPlaneContactFinishTask()
{
	PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
	if(mFreeWorldPlaneContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeWorldPlaneContactFinishTaskIndices.back();
	mFreeWorldPlaneContactFinishTaskIndices.popBack();
	return mWorldPlaneContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleWorldPlaneContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
	PX_ASSERT(index < mWorldPlaneContactTasks.size());
	mFreeWorldPlaneContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleWorldPlaneContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
	PX_ASSERT(index < mWorldPlaneContactFinishTasks.size());
	mFreeWorldPlaneContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneWorldPlaneContactTask(
	PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 taskCount = getWorldPlaneContactTaskCount(dispatcherWorkers);
	if(!mWorldPlaneContactTransactionPending || !continuation ||
		taskCount == 0 ||
		!ensureWorldPlaneContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(mWorldPlaneContactTaskPoolMutex);
		if(mFreeWorldPlaneContactTaskIndices.size() < taskCount ||
			mFreeWorldPlaneContactFinishTaskIndices.empty())
			return false;
	}
	const PxU64 maxContactCount64 =
		PxU64(mParticles.size()) * mWorldPlanes.size();
	if(maxContactCount64 > PX_MAX_U32)
		return false;
	mContacts.reserve(PxU32(maxContactCount64));
	mWorldPlaneContactTaskOutputs.resize(taskCount);
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		PxArray<Dy::AvbdSoftContact>& output =
			mWorldPlaneContactTaskOutputs[taskIndex];
		output.clear();
		output.reserve(
			(particleEnd - particleBegin) * mWorldPlanes.size());
	}
	WorldPlaneContactFinishTask* const finishTask =
		acquireWorldPlaneContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordWorldPlaneContactTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		WorldPlaneContactTask* const task = acquireWorldPlaneContactTask();
		PX_ASSERT(task && particleBegin < particleEnd);
		if(!task)
		{
			recycleWorldPlaneContactFinishTask(finishTask->getPoolIndex());
			return false;
		}
		task->configure(
			mParticles.begin(), mParticles.size(), particleBegin, particleEnd,
			mWorldPlanes.begin(), mWorldPlanes.size(),
			mBodies.begin(), mBodies.size(),
			mWorldPlaneContactTaskOutputs[taskIndex],
			mContactParams.contactRadius);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}

bool AvbdCpuSoftScene::canUseWorldPlaneContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseWorldPlaneContactTaskFanIn() ||
		mBodies.empty() || mWorldPlanes.empty() ||
		!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
		!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
		!mRigidTriangleSurfaces.empty() ||
		mSelfCollisionEnabled.size() != mBodies.size())
		return false;
	// The leaf partition owns disjoint particle ranges and receives the full
	// immutable particle-to-body map.  Multiple deformable bodies therefore do
	// not introduce a write conflict; retaining the historical single-body gate
	// would unnecessarily serialize the contact stage of an otherwise complete
	// prediction/primal/write-back graph.
	for(PxU32 bodyIndex = 0;
		bodyIndex < mSelfCollisionEnabled.size(); ++bodyIndex)
	{
		if(mSelfCollisionEnabled[bodyIndex])
			return false;
	}
	return true;
}

PxU32 AvbdCpuSoftScene::getWorldPlaneContactTaskCount(
	PxU32 dispatcherWorkers) const
{
	static const PxU32 eMIN_PARTICLES_PER_WORLD_PLANE_TASK = 128;
	if(!canUseWorldPlaneContactTaskTransaction() ||
		dispatcherWorkers < 2 || mParticles.size() <
			eMIN_PARTICLES_PER_WORLD_PLANE_TASK)
		return 0;
	const PxU32 maxTasksByParticles = (mParticles.size() +
		eMIN_PARTICLES_PER_WORLD_PLANE_TASK - 1) /
		eMIN_PARTICLES_PER_WORLD_PLANE_TASK;
	return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
		mParticles.size());
}

bool AvbdCpuSoftScene::beginWorldPlaneContactTaskTransaction()
{
	if(!canUseWorldPlaneContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, mWorldPlanes.size(), false,
		0, 0, 0, 0, 0, mBodies.size(),
		mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(),
		mSelfCollisionEnabled.begin());
	mWorldPlaneContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeWorldPlaneContactTaskTransaction()
{
	PX_ASSERT(mWorldPlaneContactTransactionPending);
	for(PxU32 taskIndex = 0;
		taskIndex < mWorldPlaneContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mWorldPlaneContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedGroundContacts += mContacts.size();
	completeComponentContactRedetection();
	mWorldPlaneContactTransactionPending = false;
}

bool AvbdCpuSoftScene::completeStandaloneWorldPlaneContactTask(
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
		!mWorldPlaneContactTransactionPending)
		return false;
	completeWorldPlaneContactTaskTransaction();
	if(!mStandaloneComponentStepState.completePendingRedetection())
		return false;
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

bool AvbdCpuSoftScene::finishStandaloneWorldPlaneContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mWorldPlaneContactTransactionPending)
		return false;
	mStandaloneTaskGraphTelemetry.recordSerialWorldPlaneContactFallback();
	mWorldPlaneContactTaskOutputs.clear();
	Dy::avbdDetectSoftWorldPlaneContacts(
		mParticles.begin(), mParticles.size(),
		mWorldPlanes.begin(), mWorldPlanes.size(), mContacts,
		mContactParams.contactRadius, mBodies.begin(), mBodies.size());
	return completeStandaloneWorldPlaneContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

} // namespace Sc
} // namespace physx
