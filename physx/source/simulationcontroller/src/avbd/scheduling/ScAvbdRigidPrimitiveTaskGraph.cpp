// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

bool AvbdCpuSoftScene::canUseRigidSphereSdfContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseRigidSphereSdfContactTaskFanIn() ||
		mBodies.size() != 1 || mRigidSpheres.empty() ||
		!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
		!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
		!mRigidTriangleSurfaces.empty() ||
		mSelfCollisionEnabled.size() != mBodies.size())
		return false;
	// Dynamic/kinematic spheres have relative-motion contact ownership.
	// This first sphere leaf is strictly world-static.
	for(PxU32 sphereIndex = 0;
		sphereIndex < mRigidSpheres.size(); ++sphereIndex)
	{
		if(mRigidSpheres[sphereIndex].targetKind !=
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

PxU32 AvbdCpuSoftScene::getRigidSphereSdfContactTaskCount(
	PxU32 dispatcherWorkers) const
{
	static const PxU32 eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK = 128;
	if(!canUseRigidSphereSdfContactTaskTransaction() ||
		dispatcherWorkers < 2 || mParticles.size() <
			eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK)
		return 0;
	const PxU32 maxTasksByParticles = (mParticles.size() +
			eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK - 1) /
			eMIN_PARTICLES_PER_RIGID_SPHERE_SDF_TASK;
	return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
		mParticles.size());
}

bool AvbdCpuSoftScene::beginRigidSphereSdfContactTaskTransaction()
{
	if(!canUseRigidSphereSdfContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending ||
		mRigidBoxSdfContactTransactionPending ||
		mRigidSphereSdfContactTransactionPending ||
		mRigidCapsuleSdfContactTransactionPending ||
		mRigidConvexSdfContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, 0, false,
		0, mRigidSpheres.size(), 0, 0, 0, mBodies.size(),
		mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(),
		mSelfCollisionEnabled.begin());
	mRigidSphereSdfContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeRigidSphereSdfContactTaskTransaction()
{
	PX_ASSERT(mRigidSphereSdfContactTransactionPending);
	const PxU32 rigidStart = mContacts.size();
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidSphereSdfContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidSphereSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	// P5.13b's only SDF-family merge point: every current-SDF range
	// precedes every swept-SDF range. Do not interleave both streams per
	// child even though a child evaluates its two leaves back-to-back.
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidSphereSweptSdfContactTaskOutputs.size();
		++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidSphereSweptSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.rigidParticleSphereTests +=
			PxU64(mParticles.size()) * mRigidSpheres.size();
	// Both feature suffixes remain parent-owned after current/swept fan-in.
	Dy::avbdDetectSoftRigidSphereSweptOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidSpheres.begin(), mRigidSpheres.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius,
		&mWorkspace.contact.rigidConvexForwardOwnerScratch);
	Dy::avbdDetectSoftRigidSphereOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidSpheres.begin(), mRigidSpheres.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius);
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedRigidContacts +=
			mContacts.size() - rigidStart;
	completeComponentContactRedetection();
	mRigidSphereSdfContactTransactionPending = false;
}

bool AvbdCpuSoftScene::completeStandaloneRigidSphereSdfContactTask(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(mStaticWorldSelfOgcContactTransactionPending)
		return completeStandaloneStaticWorldSelfOgcContactTask(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mSelfBvhContactTransactionPending)
		return completeStandaloneSelfBvhContactTask(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mRigidCapsuleSdfContactTransactionPending)
		return completeStandaloneRigidCapsuleSdfContactTask(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mRigidConvexSdfContactTransactionPending)
		return completeStandaloneRigidConvexSdfContactTask(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mRigidTriangleSurfaceContactTransactionPending)
		return completeStandaloneRigidTriangleSurfaceContactTask(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	nextLayerReady = false;
	nextWorldPlaneContactTaskReady = false;
	nextRigidBoxSdfContactTaskReady = false;
	nextRigidSphereSdfContactTaskReady = false;
	if(!mStandaloneComponentSolvePrepared ||
		!mRigidSphereSdfContactTransactionPending)
		return false;
	completeRigidSphereSdfContactTaskTransaction();
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

bool AvbdCpuSoftScene::finishStandaloneRigidSphereSdfContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(mStaticWorldSelfOgcContactTransactionPending)
		return finishStandaloneStaticWorldSelfOgcContactSerialFallback(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mSelfBvhContactTransactionPending)
		return finishStandaloneSelfBvhContactSerialFallback(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mRigidCapsuleSdfContactTransactionPending)
		return finishStandaloneRigidCapsuleSdfContactSerialFallback(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mRigidConvexSdfContactTransactionPending)
		return finishStandaloneRigidConvexSdfContactSerialFallback(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(mRigidTriangleSurfaceContactTransactionPending)
		return finishStandaloneRigidTriangleSurfaceContactSerialFallback(
			dt, taskGraphContext, nextLayerReady,
			nextWorldPlaneContactTaskReady,
			nextRigidBoxSdfContactTaskReady,
			nextRigidSphereSdfContactTaskReady);
	if(!mRigidSphereSdfContactTransactionPending)
		return false;
	mStandaloneTaskGraphTelemetry.
		recordSerialRigidSphereSdfContactFallback();
	mRigidSphereSdfContactTaskOutputs.clear();
	mRigidSphereSweptSdfContactTaskOutputs.clear();
	Dy::avbdDetectSoftRigidSphereSDF(
		mParticles.begin(), mParticles.size(),
		mRigidSpheres.begin(), mRigidSpheres.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	// Match P5.13b's parent fan-in family order before the two feature
	// suffixes in completeRigidSphereSdfContactTaskTransaction().
	Dy::avbdDetectSoftRigidSphereSweptSDF(
		mParticles.begin(), mParticles.size(),
		mRigidSpheres.begin(), mRigidSpheres.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	return completeStandaloneRigidSphereSdfContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

bool AvbdCpuSoftScene::ensureRigidSphereSdfContactTaskPool(PxU32 requiredChildTasks,
	Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
	mRigidSphereSdfContactTasks.reserve(requiredSlots);
	mRigidSphereSdfContactFinishTasks.reserve(requiredSlots);
	mFreeRigidSphereSdfContactTaskIndices.reserve(requiredSlots);
	mFreeRigidSphereSdfContactFinishTaskIndices.reserve(requiredSlots);
	mRigidSphereSdfContactTaskOutputs.reserve(requiredChildTasks);
	mRigidSphereSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
	while(mRigidSphereSdfContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidSphereSdfContactTasks.size();
		mRigidSphereSdfContactTasks.pushBack(
			PX_NEW(RigidSphereSdfContactTask)(mContextId, *this, index));
		mFreeRigidSphereSdfContactTaskIndices.pushBack(index);
	}
	while(mRigidSphereSdfContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidSphereSdfContactFinishTasks.size();
		mRigidSphereSdfContactFinishTasks.pushBack(
			PX_NEW(RigidSphereSdfContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeRigidSphereSdfContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::RigidSphereSdfContactTask* AvbdCpuSoftScene::acquireRigidSphereSdfContactTask()
{
	PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
	if(mFreeRigidSphereSdfContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeRigidSphereSdfContactTaskIndices.back();
	mFreeRigidSphereSdfContactTaskIndices.popBack();
	return mRigidSphereSdfContactTasks[index];
}

AvbdCpuSoftScene::RigidSphereSdfContactFinishTask*
	AvbdCpuSoftScene::acquireRigidSphereSdfContactFinishTask()
{
	PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
	if(mFreeRigidSphereSdfContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index =
		mFreeRigidSphereSdfContactFinishTaskIndices.back();
	mFreeRigidSphereSdfContactFinishTaskIndices.popBack();
	return mRigidSphereSdfContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleRigidSphereSdfContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidSphereSdfContactTasks.size());
	mFreeRigidSphereSdfContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleRigidSphereSdfContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidSphereSdfContactFinishTasks.size());
	mFreeRigidSphereSdfContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneRigidSphereSdfContactTask(
	PxU32 dispatcherWorkers, Scene& owner,
	PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& taskGraphContext)
{
	if(mStaticWorldSelfOgcContactTransactionPending)
		return submitStandaloneStaticWorldSelfOgcContactTask(
			dispatcherWorkers, owner, continuation, taskGraphContext);
	if(mSelfBvhContactTransactionPending)
		return submitStandaloneSelfBvhContactTask(
			dispatcherWorkers, owner, continuation, taskGraphContext);
	if(mRigidCapsuleSdfContactTransactionPending)
		return submitStandaloneRigidCapsuleSdfContactTask(
			dispatcherWorkers, owner, continuation, taskGraphContext);
	if(mRigidConvexSdfContactTransactionPending)
		return submitStandaloneRigidConvexSdfContactTask(
			dispatcherWorkers, owner, continuation, taskGraphContext);
	if(mRigidTriangleSurfaceContactTransactionPending)
		return submitStandaloneRigidTriangleSurfaceContactTask(
			dispatcherWorkers, owner, continuation, taskGraphContext);
	const PxU32 taskCount = getRigidSphereSdfContactTaskCount(
		dispatcherWorkers);
	if(!mRigidSphereSdfContactTransactionPending || !continuation ||
		taskCount == 0 ||
		!ensureRigidSphereSdfContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(mRigidSphereSdfContactTaskPoolMutex);
		if(mFreeRigidSphereSdfContactTaskIndices.size() < taskCount ||
			mFreeRigidSphereSdfContactFinishTaskIndices.empty())
			return false;
	}
	const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
		mRigidSpheres.size();
	if(maxContactCount64 > PX_MAX_U32)
		return false;
	mContacts.reserve(PxU32(maxContactCount64));
	mRigidSphereSdfContactTaskOutputs.resize(taskCount);
	mRigidSphereSweptSdfContactTaskOutputs.resize(taskCount);
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		PxArray<Dy::AvbdSoftContact>& output =
			mRigidSphereSdfContactTaskOutputs[taskIndex];
		output.clear();
		output.reserve((particleEnd - particleBegin) *
			mRigidSpheres.size());
		PxArray<Dy::AvbdSoftContact>& sweptOutput =
			mRigidSphereSweptSdfContactTaskOutputs[taskIndex];
		sweptOutput.clear();
		sweptOutput.reserve((particleEnd - particleBegin) *
			mRigidSpheres.size());
	}
	RigidSphereSdfContactFinishTask* const finishTask =
		acquireRigidSphereSdfContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordRigidSphereSdfContactTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		RigidSphereSdfContactTask* const task =
			acquireRigidSphereSdfContactTask();
		PX_ASSERT(task && particleBegin < particleEnd);
		if(!task)
		{
			recycleRigidSphereSdfContactFinishTask(
				finishTask->getPoolIndex());
			return false;
		}
		task->configure(
			mParticles.begin(), mParticles.size(),
			particleBegin, particleEnd,
		mRigidSpheres.begin(), mRigidSpheres.size(),
		mBodies.begin(), mBodies.size(),
		mRigidSphereSdfContactTaskOutputs[taskIndex],
		mRigidSphereSweptSdfContactTaskOutputs[taskIndex],
		mContactParams.contactRadius);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}

bool AvbdCpuSoftScene::canUseRigidCapsuleSdfContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseRigidCapsuleSdfContactTaskFanIn() ||
		mBodies.size() != 1 || mRigidCapsules.empty() ||
		!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
		!mRigidSpheres.empty() || !mRigidConvexes.empty() ||
		!mRigidTriangleSurfaces.empty() ||
		mSelfCollisionEnabled.size() != mBodies.size())
		return false;
	// Dynamic/kinematic capsules carry relative-motion ownership. The
	// P5.6b leaf deliberately accepts world-static capsules only.
	for(PxU32 capsuleIndex = 0;
		capsuleIndex < mRigidCapsules.size(); ++capsuleIndex)
	{
		if(mRigidCapsules[capsuleIndex].targetKind !=
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

PxU32 AvbdCpuSoftScene::getRigidCapsuleSdfContactTaskCount(
	PxU32 dispatcherWorkers) const
{
	static const PxU32 eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK = 128;
	if(!canUseRigidCapsuleSdfContactTaskTransaction() ||
		dispatcherWorkers < 2 || mParticles.size() <
			eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK)
		return 0;
	const PxU32 maxTasksByParticles = (mParticles.size() +
			eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK - 1) /
			eMIN_PARTICLES_PER_RIGID_CAPSULE_SDF_TASK;
	return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
		mParticles.size());
}

bool AvbdCpuSoftScene::beginRigidCapsuleSdfContactTaskTransaction()
{
	if(!canUseRigidCapsuleSdfContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending ||
		mRigidBoxSdfContactTransactionPending ||
		mRigidSphereSdfContactTransactionPending ||
		mRigidCapsuleSdfContactTransactionPending ||
		mRigidConvexSdfContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, 0, false,
		0, 0, mRigidCapsules.size(), 0, 0, mBodies.size(),
		mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(),
		mSelfCollisionEnabled.begin());
	mRigidCapsuleSdfContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeRigidCapsuleSdfContactTaskTransaction()
{
	PX_ASSERT(mRigidCapsuleSdfContactTransactionPending);
	const PxU32 rigidStart = mContacts.size();
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidCapsuleSdfContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidCapsuleSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	// P5.14b's only SDF-family merge point: every current-SDF range
	// precedes every swept-SDF range. Do not interleave both streams per
	// child even though a child evaluates its two leaves back-to-back.
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidCapsuleSweptSdfContactTaskOutputs.size();
		++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidCapsuleSweptSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0;
			contactIndex < source.size(); ++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.rigidParticleCapsuleTests +=
			PxU64(mParticles.size()) * mRigidCapsules.size();
	// Both feature suffixes remain parent-owned after current/swept fan-in.
	Dy::avbdDetectSoftRigidCapsuleSweptOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidCapsules.begin(), mRigidCapsules.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius,
		&mWorkspace.contact.rigidConvexForwardOwnerScratch);
	Dy::avbdDetectSoftRigidCapsuleOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidCapsules.begin(), mRigidCapsules.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius);
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedRigidContacts +=
			mContacts.size() - rigidStart;
	completeComponentContactRedetection();
	mRigidCapsuleSdfContactTransactionPending = false;
}

bool AvbdCpuSoftScene::completeStandaloneRigidCapsuleSdfContactTask(
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
		!mRigidCapsuleSdfContactTransactionPending)
		return false;
	completeRigidCapsuleSdfContactTaskTransaction();
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

bool AvbdCpuSoftScene::finishStandaloneRigidCapsuleSdfContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mRigidCapsuleSdfContactTransactionPending)
		return false;
	mStandaloneTaskGraphTelemetry.
		recordSerialRigidCapsuleSdfContactFallback();
	mRigidCapsuleSdfContactTaskOutputs.clear();
	mRigidCapsuleSweptSdfContactTaskOutputs.clear();
	Dy::avbdDetectSoftRigidCapsuleSDF(
		mParticles.begin(), mParticles.size(),
		mRigidCapsules.begin(), mRigidCapsules.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	// Match P5.14b's parent fan-in family order before the two feature
	// suffixes in completeRigidCapsuleSdfContactTaskTransaction().
	Dy::avbdDetectSoftRigidCapsuleSweptSDF(
		mParticles.begin(), mParticles.size(),
		mRigidCapsules.begin(), mRigidCapsules.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	return completeStandaloneRigidCapsuleSdfContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

bool AvbdCpuSoftScene::ensureRigidCapsuleSdfContactTaskPool(PxU32 requiredChildTasks,
	Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
	mRigidCapsuleSdfContactTasks.reserve(requiredSlots);
	mRigidCapsuleSdfContactFinishTasks.reserve(requiredSlots);
	mFreeRigidCapsuleSdfContactTaskIndices.reserve(requiredSlots);
	mFreeRigidCapsuleSdfContactFinishTaskIndices.reserve(requiredSlots);
	mRigidCapsuleSdfContactTaskOutputs.reserve(requiredChildTasks);
	mRigidCapsuleSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
	while(mRigidCapsuleSdfContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidCapsuleSdfContactTasks.size();
		mRigidCapsuleSdfContactTasks.pushBack(
			PX_NEW(RigidCapsuleSdfContactTask)(
				mContextId, *this, index));
		mFreeRigidCapsuleSdfContactTaskIndices.pushBack(index);
	}
	while(mRigidCapsuleSdfContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidCapsuleSdfContactFinishTasks.size();
		mRigidCapsuleSdfContactFinishTasks.pushBack(
			PX_NEW(RigidCapsuleSdfContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeRigidCapsuleSdfContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::RigidCapsuleSdfContactTask* AvbdCpuSoftScene::acquireRigidCapsuleSdfContactTask()
{
	PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
	if(mFreeRigidCapsuleSdfContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeRigidCapsuleSdfContactTaskIndices.back();
	mFreeRigidCapsuleSdfContactTaskIndices.popBack();
	return mRigidCapsuleSdfContactTasks[index];
}

AvbdCpuSoftScene::RigidCapsuleSdfContactFinishTask*
	AvbdCpuSoftScene::acquireRigidCapsuleSdfContactFinishTask()
{
	PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
	if(mFreeRigidCapsuleSdfContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index =
		mFreeRigidCapsuleSdfContactFinishTaskIndices.back();
	mFreeRigidCapsuleSdfContactFinishTaskIndices.popBack();
	return mRigidCapsuleSdfContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleRigidCapsuleSdfContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidCapsuleSdfContactTasks.size());
	mFreeRigidCapsuleSdfContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleRigidCapsuleSdfContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidCapsuleSdfContactFinishTasks.size());
	mFreeRigidCapsuleSdfContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneRigidCapsuleSdfContactTask(
	PxU32 dispatcherWorkers, Scene& owner,
	PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 taskCount = getRigidCapsuleSdfContactTaskCount(
		dispatcherWorkers);
	if(!mRigidCapsuleSdfContactTransactionPending || !continuation ||
		taskCount == 0 ||
		!ensureRigidCapsuleSdfContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(mRigidCapsuleSdfContactTaskPoolMutex);
		if(mFreeRigidCapsuleSdfContactTaskIndices.size() < taskCount ||
			mFreeRigidCapsuleSdfContactFinishTaskIndices.empty())
			return false;
	}
	const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
		mRigidCapsules.size();
	if(maxContactCount64 > PX_MAX_U32)
		return false;
	mContacts.reserve(PxU32(maxContactCount64));
	mRigidCapsuleSdfContactTaskOutputs.resize(taskCount);
	mRigidCapsuleSweptSdfContactTaskOutputs.resize(taskCount);
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		PxArray<Dy::AvbdSoftContact>& output =
			mRigidCapsuleSdfContactTaskOutputs[taskIndex];
		output.clear();
		output.reserve((particleEnd - particleBegin) *
			mRigidCapsules.size());
		PxArray<Dy::AvbdSoftContact>& sweptOutput =
			mRigidCapsuleSweptSdfContactTaskOutputs[taskIndex];
		sweptOutput.clear();
		sweptOutput.reserve((particleEnd - particleBegin) *
			mRigidCapsules.size());
	}
	RigidCapsuleSdfContactFinishTask* const finishTask =
		acquireRigidCapsuleSdfContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordRigidCapsuleSdfContactTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		RigidCapsuleSdfContactTask* const task =
			acquireRigidCapsuleSdfContactTask();
		PX_ASSERT(task && particleBegin < particleEnd);
		if(!task)
		{
			recycleRigidCapsuleSdfContactFinishTask(
				finishTask->getPoolIndex());
			return false;
		}
		task->configure(
			mParticles.begin(), mParticles.size(),
			particleBegin, particleEnd,
			mRigidCapsules.begin(), mRigidCapsules.size(),
		mBodies.begin(), mBodies.size(),
		mRigidCapsuleSdfContactTaskOutputs[taskIndex],
		mRigidCapsuleSweptSdfContactTaskOutputs[taskIndex],
		mContactParams.contactRadius);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}


bool AvbdCpuSoftScene::ensureRigidConvexSdfContactTaskPool(PxU32 requiredChildTasks,
	Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
	mRigidConvexSdfContactTasks.reserve(requiredSlots);
	mRigidConvexSdfContactFinishTasks.reserve(requiredSlots);
	mFreeRigidConvexSdfContactTaskIndices.reserve(requiredSlots);
	mFreeRigidConvexSdfContactFinishTaskIndices.reserve(requiredSlots);
	mRigidConvexSdfContactTaskOutputs.reserve(requiredChildTasks);
	mRigidConvexSweptSdfContactTaskOutputs.reserve(requiredChildTasks);
	while(mRigidConvexSdfContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidConvexSdfContactTasks.size();
		mRigidConvexSdfContactTasks.pushBack(
			PX_NEW(RigidConvexSdfContactTask)(mContextId, *this, index));
		mFreeRigidConvexSdfContactTaskIndices.pushBack(index);
	}
	while(mRigidConvexSdfContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mRigidConvexSdfContactFinishTasks.size();
		mRigidConvexSdfContactFinishTasks.pushBack(
			PX_NEW(RigidConvexSdfContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeRigidConvexSdfContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::RigidConvexSdfContactTask* AvbdCpuSoftScene::acquireRigidConvexSdfContactTask()
{
	PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
	if(mFreeRigidConvexSdfContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeRigidConvexSdfContactTaskIndices.back();
	mFreeRigidConvexSdfContactTaskIndices.popBack();
	return mRigidConvexSdfContactTasks[index];
}

AvbdCpuSoftScene::RigidConvexSdfContactFinishTask*
	AvbdCpuSoftScene::acquireRigidConvexSdfContactFinishTask()
{
	PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
	if(mFreeRigidConvexSdfContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index =
		mFreeRigidConvexSdfContactFinishTaskIndices.back();
	mFreeRigidConvexSdfContactFinishTaskIndices.popBack();
	return mRigidConvexSdfContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleRigidConvexSdfContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidConvexSdfContactTasks.size());
	mFreeRigidConvexSdfContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleRigidConvexSdfContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
	PX_ASSERT(index < mRigidConvexSdfContactFinishTasks.size());
	mFreeRigidConvexSdfContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneRigidConvexSdfContactTask(
	PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 taskCount = getRigidConvexSdfContactTaskCount(
		dispatcherWorkers);
	if(!mRigidConvexSdfContactTransactionPending || !continuation ||
		taskCount == 0 ||
		!ensureRigidConvexSdfContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(mRigidConvexSdfContactTaskPoolMutex);
		if(mFreeRigidConvexSdfContactTaskIndices.size() < taskCount ||
			mFreeRigidConvexSdfContactFinishTaskIndices.empty())
			return false;
	}
	const PxU64 maxContactCount64 = PxU64(mParticles.size()) *
		mRigidConvexes.size();
	if(maxContactCount64 > PX_MAX_U32)
		return false;
	mContacts.reserve(PxU32(maxContactCount64));
	mRigidConvexSdfContactTaskOutputs.resize(taskCount);
	mRigidConvexSweptSdfContactTaskOutputs.resize(taskCount);
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	for(PxU32 index = 0; index < taskCount; ++index)
	{
		const PxU32 begin = index * particlesPerTask;
		const PxU32 end = PxMin(begin + particlesPerTask,
			mParticles.size());
		PxArray<Dy::AvbdSoftContact>& output =
			mRigidConvexSdfContactTaskOutputs[index];
		output.clear();
		output.reserve((end - begin) * mRigidConvexes.size());
		PxArray<Dy::AvbdSoftContact>& sweptOutput =
			mRigidConvexSweptSdfContactTaskOutputs[index];
		sweptOutput.clear();
		sweptOutput.reserve((end - begin) * mRigidConvexes.size());
	}
	RigidConvexSdfContactFinishTask* const finishTask =
		acquireRigidConvexSdfContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordRigidConvexSdfContactTasksSubmitted(taskCount);
	for(PxU32 index = 0; index < taskCount; ++index)
	{
		const PxU32 begin = index * particlesPerTask;
		const PxU32 end = PxMin(begin + particlesPerTask,
			mParticles.size());
		RigidConvexSdfContactTask* const task =
			acquireRigidConvexSdfContactTask();
		if(!task)
		{
			recycleRigidConvexSdfContactFinishTask(
				finishTask->getPoolIndex());
			return false;
		}
		task->configure(mParticles.begin(), mParticles.size(), begin, end,
		mRigidConvexes.begin(), mRigidConvexes.size(),
		mBodies.begin(), mBodies.size(),
		mRigidConvexSdfContactTaskOutputs[index],
		mRigidConvexSweptSdfContactTaskOutputs[index],
		mContactParams.contactRadius);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}


bool AvbdCpuSoftScene::canUseRigidConvexSdfContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseRigidConvexSdfContactTaskFanIn() ||
		mBodies.size() != 1 || mRigidConvexes.empty() ||
		!mWorldPlanes.empty() || !mRigidBoxes.empty() ||
		!mRigidSpheres.empty() || !mRigidCapsules.empty() ||
		!mRigidTriangleSurfaces.empty() ||
		mSelfCollisionEnabled.size() != mBodies.size())
		return false;
	for(PxU32 index = 0; index < mRigidConvexes.size(); ++index)
	{
		if(mRigidConvexes[index].targetKind !=
			Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
			return false;
	}
	for(PxU32 index = 0; index < mSelfCollisionEnabled.size(); ++index)
	{
		if(mSelfCollisionEnabled[index])
			return false;
	}
	return true;
}

PxU32 AvbdCpuSoftScene::getRigidConvexSdfContactTaskCount(PxU32 dispatcherWorkers) const
{
	static const PxU32 eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK = 128;
	if(!canUseRigidConvexSdfContactTaskTransaction() ||
		dispatcherWorkers < 2 || mParticles.size() <
			eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK)
		return 0;
	const PxU32 maxTasksByParticles = (mParticles.size() +
			eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK - 1) /
			eMIN_PARTICLES_PER_RIGID_CONVEX_SDF_TASK;
	return PxMin(PxMin(dispatcherWorkers, maxTasksByParticles),
		mParticles.size());
}

bool AvbdCpuSoftScene::beginRigidConvexSdfContactTaskTransaction()
{
	if(!canUseRigidConvexSdfContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending ||
		mRigidBoxSdfContactTransactionPending ||
		mRigidSphereSdfContactTransactionPending ||
		mRigidCapsuleSdfContactTransactionPending ||
		mRigidConvexSdfContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, 0, false,
		0, 0, 0, mRigidConvexes.size(), 0, mBodies.size(),
		mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(),
		mSelfCollisionEnabled.begin());
	mRigidConvexSdfContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeRigidConvexSdfContactTaskTransaction()
{
	PX_ASSERT(mRigidConvexSdfContactTransactionPending);
	const PxU32 rigidStart = mContacts.size();
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidConvexSdfContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidConvexSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0; contactIndex < source.size();
			++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	// P5.15b's only SDF-family merge point: every current-SDF range
	// precedes every swept-SDF range. Do not interleave both streams per
	// child even though a child evaluates its two leaves back-to-back.
	for(PxU32 taskIndex = 0;
		taskIndex < mRigidConvexSweptSdfContactTaskOutputs.size();
		++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mRigidConvexSweptSdfContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0; contactIndex < source.size();
			++contactIndex)
			mContacts.pushBack(source[contactIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.rigidParticleConvexTests +=
			PxU64(mParticles.size()) * mRigidConvexes.size();
	// Both feature suffixes remain parent-owned after current/swept fan-in.
	Dy::avbdDetectSoftRigidConvexSweptOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidConvexes.begin(), mRigidConvexes.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius,
		&mWorkspace.contact.rigidConvexForwardOwnerScratch);
	Dy::avbdDetectSoftRigidConvexOGCFeatures(
		mParticles.begin(), mParticles.size(),
		mRigidConvexes.begin(), mRigidConvexes.size(),
		mBodies.begin(), mBodies.size(), mContacts,
		mContactParams.contactRadius);
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedRigidContacts +=
			mContacts.size() - rigidStart;
	completeComponentContactRedetection();
	mRigidConvexSdfContactTransactionPending = false;
}

bool AvbdCpuSoftScene::completeStandaloneRigidConvexSdfContactTask(
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
		!mRigidConvexSdfContactTransactionPending)
		return false;
	completeRigidConvexSdfContactTaskTransaction();
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

bool AvbdCpuSoftScene::finishStandaloneRigidConvexSdfContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mRigidConvexSdfContactTransactionPending)
		return false;
	mStandaloneTaskGraphTelemetry.
		recordSerialRigidConvexSdfContactFallback();
	mRigidConvexSdfContactTaskOutputs.clear();
	mRigidConvexSweptSdfContactTaskOutputs.clear();
	Dy::avbdDetectSoftRigidConvexSDF(
		mParticles.begin(), mParticles.size(),
		mRigidConvexes.begin(), mRigidConvexes.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	// Match P5.15b's parent fan-in family order before the two feature
	// suffixes in completeRigidConvexSdfContactTaskTransaction().
	Dy::avbdDetectSoftRigidConvexSweptSDF(
		mParticles.begin(), mParticles.size(),
		mRigidConvexes.begin(), mRigidConvexes.size(), mContacts,
		mContactParams.contactRadius,
		mBodies.begin(), mBodies.size());
	return completeStandaloneRigidConvexSdfContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

} // namespace Sc
} // namespace physx
