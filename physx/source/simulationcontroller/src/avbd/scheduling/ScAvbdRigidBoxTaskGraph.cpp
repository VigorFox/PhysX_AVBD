// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

bool AvbdCpuSoftScene::ensureRigidBoxSdfContactTaskPool(
	PxU32 requiredChildTasks, Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
	mRigidBoxSdfContactTasks.reserve(requiredSlots);
	mRigidBoxSdfContactFinishTasks.reserve(requiredSlots);
	mFreeRigidBoxSdfContactTaskIndices.reserve(requiredSlots);
	mFreeRigidBoxSdfContactFinishTaskIndices.reserve(requiredSlots);
	mRigidBoxSdfContactTaskOutputs.reserve(requiredChildTasks);
	mRigidBoxSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
	while(mRigidBoxSdfContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidBoxSdfContactTasks.size();
		mRigidBoxSdfContactTasks.pushBack(PX_NEW(RigidBoxSdfContactTask)(
			mContextId, *this, index));
		mFreeRigidBoxSdfContactTaskIndices.pushBack(index);
	}
	while(mRigidBoxSdfContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidBoxSdfContactFinishTasks.size();
		mRigidBoxSdfContactFinishTasks.pushBack(
			PX_NEW(RigidBoxSdfContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeRigidBoxSdfContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::RigidBoxSdfContactTask*
AvbdCpuSoftScene::acquireRigidBoxSdfContactTask()
{
	PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
	if(mFreeRigidBoxSdfContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeRigidBoxSdfContactTaskIndices.back();
	mFreeRigidBoxSdfContactTaskIndices.popBack();
	return mRigidBoxSdfContactTasks[index];
}

AvbdCpuSoftScene::RigidBoxSdfContactFinishTask*
AvbdCpuSoftScene::acquireRigidBoxSdfContactFinishTask()
{
	PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
	if(mFreeRigidBoxSdfContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeRigidBoxSdfContactFinishTaskIndices.back();
	mFreeRigidBoxSdfContactFinishTaskIndices.popBack();
	return mRigidBoxSdfContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleRigidBoxSdfContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidBoxSdfContactTasks.size());
	mFreeRigidBoxSdfContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleRigidBoxSdfContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidBoxSdfContactFinishTasks.size());
	mFreeRigidBoxSdfContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneRigidBoxSdfContactTask(
	PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 taskCount = getRigidBoxSdfContactTaskCount(dispatcherWorkers);
	if(!mRigidBoxSdfContactTransactionPending || !continuation ||
		taskCount == 0 ||
		!ensureRigidBoxSdfContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(mRigidBoxSdfContactTaskPoolMutex);
		if(mFreeRigidBoxSdfContactTaskIndices.size() < taskCount ||
			mFreeRigidBoxSdfContactFinishTaskIndices.empty())
			return false;
	}
	const PxU64 maxContactCount64 =
		PxU64(mParticles.size()) * mRigidBoxes.size();
	if(maxContactCount64 > PX_MAX_U32)
		return false;
	mContacts.reserve(PxU32(maxContactCount64));
	mRigidBoxSdfContactTaskOutputs.resize(taskCount);
	mRigidBoxSweptSdfContactTaskOutputs.resize(taskCount);
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		PxArray<Dy::AvbdSoftContact>& output =
			mRigidBoxSdfContactTaskOutputs[taskIndex];
		output.clear();
		output.reserve(
			(particleEnd - particleBegin) * mRigidBoxes.size());
		PxArray<Dy::AvbdSoftContact>& sweptOutput =
			mRigidBoxSweptSdfContactTaskOutputs[taskIndex];
		sweptOutput.clear();
		sweptOutput.reserve(
			(particleEnd - particleBegin) * mRigidBoxes.size());
	}
	RigidBoxSdfContactFinishTask* const finishTask =
		acquireRigidBoxSdfContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordRigidBoxSdfContactTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		RigidBoxSdfContactTask* const task = acquireRigidBoxSdfContactTask();
		PX_ASSERT(task && particleBegin < particleEnd);
		if(!task)
		{
			recycleRigidBoxSdfContactFinishTask(finishTask->getPoolIndex());
			return false;
		}
		task->configure(
			mParticles.begin(), mParticles.size(), particleBegin, particleEnd,
			mRigidBoxes.begin(), mRigidBoxes.size(),
			mWorkspace.contact.epoch.previousContacts.begin(),
			mWorkspace.contact.epoch.previousContacts.size(),
			mBodies.begin(), mBodies.size(),
			mRigidBoxSdfContactTaskOutputs[taskIndex],
			mRigidBoxSweptSdfContactTaskOutputs[taskIndex],
			mContactParams.contactRadius);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}

bool AvbdCpuSoftScene::canUseRigidBoxSdfContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseRigidBoxSdfContactTaskFanIn() ||
		mBodies.size() != 1 || mRigidBoxes.empty() ||
		!mWorldPlanes.empty() || !mRigidSpheres.empty() ||
		!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
		!mRigidTriangleSurfaces.empty() ||
		mSelfCollisionEnabled.size() != mBodies.size())
		return false;
	for(PxU32 boxIndex = 0; boxIndex < mRigidBoxes.size(); ++boxIndex)
	{
		if(mRigidBoxes[boxIndex].targetKind !=
			Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
			return false;
	}
	for(PxU32 bodyIndex = 0;
		bodyIndex < mSelfCollisionEnabled.size(); ++bodyIndex)
	{
		if(mSelfCollisionEnabled[bodyIndex])
			return false;
	}
	return true;
}

PxU32 AvbdCpuSoftScene::getRigidBoxSdfContactTaskCount(
	PxU32 dispatcherWorkers) const
{
	static const PxU32 eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK = 128;
	if(!canUseRigidBoxSdfContactTaskTransaction() ||
		dispatcherWorkers < 2 || mParticles.size() <
			eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK)
		return 0;
	const PxU32 maxTasksByParticles = (mParticles.size() +
		eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK - 1) /
		eMIN_PARTICLES_PER_RIGID_BOX_SDF_TASK;
	return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
		mParticles.size());
}

bool AvbdCpuSoftScene::beginRigidBoxSdfContactTaskTransaction()
{
	if(!canUseRigidBoxSdfContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending ||
		mRigidBoxSdfContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, 0, false,
		mRigidBoxes.size(), 0, 0, 0, 0, mBodies.size(),
		mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(),
		mSelfCollisionEnabled.begin());
	mRigidBoxSdfContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeRigidBoxSdfContactTaskTransaction()
{
	PX_ASSERT(mRigidBoxSdfContactTransactionPending);
	const PxU32 rigidStart = mContacts.size();
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidBoxSdfContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidBoxSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidBoxSweptSdfContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidBoxSweptSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.rigidParticleBoxTests +=
			PxU64(mParticles.size()) * mRigidBoxes.size();
	Dy::avbdDetectSoftRigidOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidBoxes.begin(), mRigidBoxes.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius,
		&mWorkspace.componentOgcGeometrySidecar);
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedRigidContacts +=
			mContacts.size() - rigidStart;
	completeComponentContactRedetection();
	mRigidBoxSdfContactTransactionPending = false;
}

bool AvbdCpuSoftScene::completeStandaloneRigidBoxSdfContactTask(
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
		!mRigidBoxSdfContactTransactionPending)
		return false;
	completeRigidBoxSdfContactTaskTransaction();
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

bool AvbdCpuSoftScene::finishStandaloneRigidBoxSdfContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mRigidBoxSdfContactTransactionPending)
		return false;
	mStandaloneTaskGraphTelemetry.recordSerialRigidBoxSdfContactFallback();
	mRigidBoxSdfContactTaskOutputs.clear();
	mRigidBoxSweptSdfContactTaskOutputs.clear();
	Dy::avbdDetectSoftRigidSDF(
		mParticles.begin(), mParticles.size(),
		mRigidBoxes.begin(), mRigidBoxes.size(), mContacts,
		mContactParams.contactRadius,
		mWorkspace.contact.epoch.previousContacts.begin(),
		mWorkspace.contact.epoch.previousContacts.size(),
		mBodies.begin(), mBodies.size());
	Dy::avbdDetectSoftRigidSweptSDF(
		mParticles.begin(), mParticles.size(),
		mRigidBoxes.begin(), mRigidBoxes.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	return completeStandaloneRigidBoxSdfContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

} // namespace Sc
} // namespace physx
