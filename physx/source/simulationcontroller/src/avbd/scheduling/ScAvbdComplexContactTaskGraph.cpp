// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

bool AvbdCpuSoftScene::ensureStaticWorldSelfOgcContactTaskPool(PxU32 requiredChildTasks,
	Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(
		mStaticWorldSelfOgcContactFinishTaskPoolMutex);
	mStaticWorldSelfOgcContactTasks.reserve(requiredSlots);
	mStaticWorldSelfOgcContactFinishTasks.reserve(requiredSlots);
	mFreeStaticWorldSelfOgcContactTaskIndices.reserve(requiredSlots);
	mFreeStaticWorldSelfOgcContactFinishTaskIndices.reserve(requiredSlots);
	mStaticWorldSelfOgcWorldTaskOutputs.reserve(requiredChildTasks);
	mStaticWorldSelfOgcBoxTaskOutputs.reserve(requiredChildTasks);
	mStaticWorldSelfOgcBoxSweptTaskOutputs.reserve(requiredChildTasks);
	mStaticWorldSelfOgcSelfVertexTaskOutputs.reserve(requiredChildTasks);
	mStaticWorldSelfOgcSelfEdgeTaskOutputs.reserve(requiredChildTasks);
	mStaticWorldSelfOgcTaskStats.reserve(requiredChildTasks);
	while(mStaticWorldSelfOgcContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mStaticWorldSelfOgcContactTasks.size();
		mStaticWorldSelfOgcContactTasks.pushBack(
			PX_NEW(StaticWorldSelfOgcContactTask)(
				mContextId, *this, index));
		mFreeStaticWorldSelfOgcContactTaskIndices.pushBack(index);
	}
	while(mStaticWorldSelfOgcContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index =
			mStaticWorldSelfOgcContactFinishTasks.size();
		mStaticWorldSelfOgcContactFinishTasks.pushBack(
			PX_NEW(StaticWorldSelfOgcContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeStaticWorldSelfOgcContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::StaticWorldSelfOgcContactTask* AvbdCpuSoftScene::acquireStaticWorldSelfOgcContactTask()
{
	PxMutex::ScopedLock lock(
		mStaticWorldSelfOgcContactFinishTaskPoolMutex);
	if(mFreeStaticWorldSelfOgcContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeStaticWorldSelfOgcContactTaskIndices.back();
	mFreeStaticWorldSelfOgcContactTaskIndices.popBack();
	return mStaticWorldSelfOgcContactTasks[index];
}

AvbdCpuSoftScene::StaticWorldSelfOgcContactFinishTask*
	AvbdCpuSoftScene::acquireStaticWorldSelfOgcContactFinishTask()
{
	PxMutex::ScopedLock lock(
		mStaticWorldSelfOgcContactFinishTaskPoolMutex);
	if(mFreeStaticWorldSelfOgcContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index =
		mFreeStaticWorldSelfOgcContactFinishTaskIndices.back();
	mFreeStaticWorldSelfOgcContactFinishTaskIndices.popBack();
	return mStaticWorldSelfOgcContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleStaticWorldSelfOgcContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(
		mStaticWorldSelfOgcContactFinishTaskPoolMutex);
	PX_ASSERT(index < mStaticWorldSelfOgcContactTasks.size());
	mFreeStaticWorldSelfOgcContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleStaticWorldSelfOgcContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(
		mStaticWorldSelfOgcContactFinishTaskPoolMutex);
	PX_ASSERT(index < mStaticWorldSelfOgcContactFinishTasks.size());
	mFreeStaticWorldSelfOgcContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneStaticWorldSelfOgcContactTask(
	PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 taskCount = getStaticWorldSelfOgcContactTaskCount(
		dispatcherWorkers);
	if(!mStaticWorldSelfOgcContactTransactionPending || !continuation ||
		taskCount == 0 ||
		!ensureStaticWorldSelfOgcContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(
			mStaticWorldSelfOgcContactFinishTaskPoolMutex);
		if(mFreeStaticWorldSelfOgcContactTaskIndices.size() < taskCount ||
			mFreeStaticWorldSelfOgcContactFinishTaskIndices.empty())
			return false;
	}

	const Dy::AvbdSoftBody& body = mBodies[0];
	mStaticWorldSelfOgcWorldTaskOutputs.resize(taskCount);
	mStaticWorldSelfOgcBoxTaskOutputs.resize(taskCount);
	mStaticWorldSelfOgcBoxSweptTaskOutputs.resize(taskCount);
	mStaticWorldSelfOgcSelfVertexTaskOutputs.resize(taskCount);
	mStaticWorldSelfOgcSelfEdgeTaskOutputs.resize(taskCount);
	mStaticWorldSelfOgcTaskStats.resize(taskCount);
	const PxU32 particlesPerTask =
		(mParticles.size() + taskCount - 1) / taskCount;
	const PxU32 verticesPerTask =
		(body.compiled.surfaceVertices.size() + taskCount - 1) /
		taskCount;
	const PxU32 edgesPerTask =
		(body.compiled.surfaceEdges.size() + taskCount - 1) / taskCount;
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		PxArray<Dy::AvbdSoftContact>& worldOutput =
			mStaticWorldSelfOgcWorldTaskOutputs[taskIndex];
		worldOutput.clear();
		worldOutput.reserve((particleEnd - particleBegin) *
			mWorldPlanes.size());
		PxArray<Dy::AvbdSoftContact>& boxOutput =
			mStaticWorldSelfOgcBoxTaskOutputs[taskIndex];
		boxOutput.clear();
		boxOutput.reserve((particleEnd - particleBegin) *
			mRigidBoxes.size());
		PxArray<Dy::AvbdSoftContact>& boxSweptOutput =
			mStaticWorldSelfOgcBoxSweptTaskOutputs[taskIndex];
		boxSweptOutput.clear();
		boxSweptOutput.reserve((particleEnd - particleBegin) *
			mRigidBoxes.size());
		mStaticWorldSelfOgcSelfVertexTaskOutputs[taskIndex].clear();
		mStaticWorldSelfOgcSelfEdgeTaskOutputs[taskIndex].clear();
		mStaticWorldSelfOgcTaskStats[taskIndex] =
			Dy::AvbdSoftCollisionStats();
	}

	StaticWorldSelfOgcContactFinishTask* const finishTask =
		acquireStaticWorldSelfOgcContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	// Attribute the aggregate to each physical OGC source.  The child
	// ownership is shared, but source telemetry must not report a zero
	// collision stage merely because the canonical parent is unified.
	mStandaloneTaskGraphTelemetry.
		recordWorldPlaneContactTasksSubmitted(taskCount);
	mStandaloneTaskGraphTelemetry.
		recordRigidBoxSdfContactTasksSubmitted(taskCount);
	mStandaloneTaskGraphTelemetry.
		recordSelfBvhContactTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 particleBegin = taskIndex * particlesPerTask;
		const PxU32 particleEnd = PxMin(
			particleBegin + particlesPerTask, mParticles.size());
		const PxU32 vertexBegin = taskIndex * verticesPerTask;
		const PxU32 vertexEnd = PxMin(
			vertexBegin + verticesPerTask,
			body.compiled.surfaceVertices.size());
		const PxU32 edgeBegin = taskIndex * edgesPerTask;
		const PxU32 edgeEnd = PxMin(edgeBegin + edgesPerTask,
			body.compiled.surfaceEdges.size());
		StaticWorldSelfOgcContactTask* const task =
			acquireStaticWorldSelfOgcContactTask();
		if(!task)
		{
			recycleStaticWorldSelfOgcContactFinishTask(
				finishTask->getPoolIndex());
			return false;
		}
		task->reserveQueryScratch(body);
		task->configure(mParticles.begin(), mParticles.size(),
			particleBegin, particleEnd, mWorldPlanes.begin(),
			mWorldPlanes.size(), mRigidBoxes.begin(), mRigidBoxes.size(),
		mWorkspace.contact.epoch.previousContacts.begin(),
		mWorkspace.contact.epoch.previousContacts.size(), body,
			mSelfCollisionAdjacencies[0], mWorkspace.contact,
			vertexBegin, vertexEnd, edgeBegin, edgeEnd,
			mStaticWorldSelfOgcWorldTaskOutputs[taskIndex],
			mStaticWorldSelfOgcBoxTaskOutputs[taskIndex],
			mStaticWorldSelfOgcBoxSweptTaskOutputs[taskIndex],
			mStaticWorldSelfOgcSelfVertexTaskOutputs[taskIndex],
			mStaticWorldSelfOgcSelfEdgeTaskOutputs[taskIndex],
			mContactParams, mCollisionStatsEnabled ?
				&mStaticWorldSelfOgcTaskStats[taskIndex] : NULL,
			mContactParams.contactRadius);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}
bool AvbdCpuSoftScene::ensureSelfBvhContactTaskPool(PxU32 requiredChildTasks,
	Scene& owner)
{
	const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
	PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
	mSelfBvhContactTasks.reserve(requiredSlots);
	mSelfBvhContactFinishTasks.reserve(requiredSlots);
	mFreeSelfBvhContactTaskIndices.reserve(requiredSlots);
	mFreeSelfBvhContactFinishTaskIndices.reserve(requiredSlots);
	mSelfBvhContactTaskOutputs.reserve(requiredChildTasks);
	mSelfBvhContactTaskStats.reserve(requiredChildTasks);
	while(mSelfBvhContactTasks.size() < requiredSlots)
	{
		const PxU32 index = mSelfBvhContactTasks.size();
		mSelfBvhContactTasks.pushBack(PX_NEW(SelfBvhContactTask)(
			mContextId, *this, index));
		mFreeSelfBvhContactTaskIndices.pushBack(index);
	}
	while(mSelfBvhContactFinishTasks.size() < requiredSlots)
	{
		const PxU32 index = mSelfBvhContactFinishTasks.size();
		mSelfBvhContactFinishTasks.pushBack(
			PX_NEW(SelfBvhContactFinishTask)(
				mContextId, *this, owner, index));
		mFreeSelfBvhContactFinishTaskIndices.pushBack(index);
	}
	return true;
}

AvbdCpuSoftScene::SelfBvhContactTask* AvbdCpuSoftScene::acquireSelfBvhContactTask()
{
	PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
	if(mFreeSelfBvhContactTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeSelfBvhContactTaskIndices.back();
	mFreeSelfBvhContactTaskIndices.popBack();
	return mSelfBvhContactTasks[index];
}

AvbdCpuSoftScene::SelfBvhContactFinishTask* AvbdCpuSoftScene::acquireSelfBvhContactFinishTask()
{
	PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
	if(mFreeSelfBvhContactFinishTaskIndices.empty())
		return NULL;
	const PxU32 index = mFreeSelfBvhContactFinishTaskIndices.back();
	mFreeSelfBvhContactFinishTaskIndices.popBack();
	return mSelfBvhContactFinishTasks[index];
}

void AvbdCpuSoftScene::recycleSelfBvhContactTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
	PX_ASSERT(index < mSelfBvhContactTasks.size());
	mFreeSelfBvhContactTaskIndices.pushBack(index);
}

void AvbdCpuSoftScene::recycleSelfBvhContactFinishTask(PxU32 index)
{
	PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
	PX_ASSERT(index < mSelfBvhContactFinishTasks.size());
	mFreeSelfBvhContactFinishTaskIndices.pushBack(index);
}

bool AvbdCpuSoftScene::submitStandaloneSelfBvhContactTask(
	PxU32 dispatcherWorkers, Scene& owner, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	const PxU32 taskCount = getSelfBvhContactTaskCount(dispatcherWorkers);
	if(!mSelfBvhContactTransactionPending || !continuation ||
		taskCount == 0 || !ensureSelfBvhContactTaskPool(taskCount, owner))
		return false;
	{
		PxMutex::ScopedLock lock(mSelfBvhContactTaskPoolMutex);
		if(mFreeSelfBvhContactTaskIndices.size() < taskCount ||
			mFreeSelfBvhContactFinishTaskIndices.empty())
			return false;
	}
	const Dy::AvbdSoftBody& body = mBodies[mSelfBvhContactBodyIndex];
	const PxU32 vertexTaskCount = body.compiled.surfaceVertices.empty() ? 0 :
		PxMin(dispatcherWorkers, body.compiled.surfaceVertices.size());
	const PxU32 edgeTaskCount = body.compiled.surfaceEdges.empty() ? 0 :
		PxMin(dispatcherWorkers, body.compiled.surfaceEdges.size());
	mSelfBvhContactTaskOutputs.resize(taskCount);
	mSelfBvhContactTaskStats.resize(taskCount);
	PxU64 totalContactCapacity = 0;
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const bool isVertexPhase = taskIndex < vertexTaskCount;
		const PxU32 phaseTaskIndex = isVertexPhase ? taskIndex :
			taskIndex - vertexTaskCount;
		const PxU32 phaseTaskCount = isVertexPhase ? vertexTaskCount :
			edgeTaskCount;
		const PxU32 itemCount = isVertexPhase ?
			body.compiled.surfaceVertices.size() :
			body.compiled.surfaceEdges.size();
		const PxU32 itemsPerTask =
			(itemCount + phaseTaskCount - 1) / phaseTaskCount;
		const PxU32 itemBegin = phaseTaskIndex * itemsPerTask;
		const PxU32 itemEnd = PxMin(itemBegin + itemsPerTask, itemCount);
		const PxU64 outputCapacity = isVertexPhase
			? PxU64(itemEnd - itemBegin) *
				PxMax(body.compiled.surfaceTriangles.size() / 3, 1u)
			: PxU64(itemEnd - itemBegin) *
				PxMax(body.compiled.surfaceEdges.size(), 1u);
		if(outputCapacity > PX_MAX_U32 ||
			totalContactCapacity > PX_MAX_U32 - outputCapacity)
			return false;
		totalContactCapacity += outputCapacity;
		mSelfBvhContactTaskOutputs[taskIndex].clear();
		mSelfBvhContactTaskOutputs[taskIndex].reserve(PxU32(outputCapacity));
		mSelfBvhContactTaskStats[taskIndex] =
			Dy::AvbdSoftCollisionStats();
	}
	mContacts.reserve(PxU32(totalContactCapacity));
	SelfBvhContactFinishTask* const finishTask =
		acquireSelfBvhContactFinishTask();
	if(!finishTask)
		return false;
	finishTask->setContinuation(continuation);
	mStandaloneTaskGraphTelemetry.
		recordSelfBvhContactTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const bool isVertexPhase = taskIndex < vertexTaskCount;
		const PxU32 phaseTaskIndex = isVertexPhase ? taskIndex :
			taskIndex - vertexTaskCount;
		const PxU32 phaseTaskCount = isVertexPhase ? vertexTaskCount :
			edgeTaskCount;
		const PxU32 itemCount = isVertexPhase ?
			body.compiled.surfaceVertices.size() :
			body.compiled.surfaceEdges.size();
		const PxU32 itemsPerTask =
			(itemCount + phaseTaskCount - 1) / phaseTaskCount;
		const PxU32 itemBegin = phaseTaskIndex * itemsPerTask;
		const PxU32 itemEnd = PxMin(itemBegin + itemsPerTask, itemCount);
		SelfBvhContactTask* const task = acquireSelfBvhContactTask();
		if(!task)
		{
			recycleSelfBvhContactFinishTask(finishTask->getPoolIndex());
			return false;
		}
		task->reserveQueryScratch(body);
		task->configure(mParticles.begin(), body, mSelfBvhContactBodyIndex,
			mSelfCollisionAdjacencies[mSelfBvhContactBodyIndex],
			mWorkspace.contact,
			isVertexPhase ? itemBegin : 0,
			isVertexPhase ? itemEnd : 0,
			isVertexPhase ? 0 : itemBegin,
			isVertexPhase ? 0 : itemEnd,
			mSelfBvhContactTaskOutputs[taskIndex], mContactParams,
			mCollisionStatsEnabled ? &mSelfBvhContactTaskStats[taskIndex] : NULL);
		task->setContinuation(finishTask);
		task->removeReference();
	}
	finishTask->removeReference();
	return true;
}

bool AvbdCpuSoftScene::canUseSelfBvhContactTaskTransaction() const
{
	if(!Dy::avbdUseSceneRedetectionBridge() ||
		!Dy::avbdUseSelfBvhContactTaskFanIn() ||
		mBodies.size() != 1 || mSelfCollisionEnabled.size() != 1 ||
		mSelfCollisionAdjacencies.size() != 1 ||
		!mSelfCollisionEnabled[0] || !mWorldPlanes.empty() ||
		!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
		!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
		!mRigidTriangleSurfaces.empty())
		return false;
	return Dy::avbdCanUseSelfCollisionOGCBvhRanges(mBodies[0]);
}

PxU32 AvbdCpuSoftScene::getSelfBvhContactTaskCount(PxU32 dispatcherWorkers) const
{
	if(!canUseSelfBvhContactTaskTransaction() || dispatcherWorkers < 2)
		return 0;
	const Dy::AvbdSoftBody& body = mBodies[0];
	const PxU32 vertexTasks = body.compiled.surfaceVertices.empty() ? 0 :
		PxMin(dispatcherWorkers, body.compiled.surfaceVertices.size());
	const PxU32 edgeTasks = body.compiled.surfaceEdges.empty() ? 0 :
		PxMin(dispatcherWorkers, body.compiled.surfaceEdges.size());
	const PxU32 taskCount = vertexTasks + edgeTasks;
	if(taskCount < 2 && !Dy::avbdForceSelfBvhContactTaskFanIn())
		return 0;
	return taskCount;
}

bool AvbdCpuSoftScene::beginSelfBvhContactTaskTransaction()
{
	if(!canUseSelfBvhContactTaskTransaction() ||
		mWorldPlaneContactTransactionPending ||
		mRigidBoxSdfContactTransactionPending ||
		mRigidSphereSdfContactTransactionPending ||
		mRigidCapsuleSdfContactTransactionPending ||
		mRigidConvexSdfContactTransactionPending ||
		mRigidTriangleSurfaceContactTransactionPending ||
		mSelfBvhContactTransactionPending)
		return false;
	beginComponentContactRedetection();
	Dy::avbdBuildSoftContactRedetectionPhasePlan(
		mWorkspace.contact, 0, false, 0, 0, 0, 0, 0,
		mBodies.size(), mSelfCollisionAdjacencies.begin(),
		mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
	mSelfBvhContactBodyIndex = 0;
	const bool prepared = Dy::avbdPrepareSelfCollisionOGCBvhRanges(
		mParticles.begin(), mBodies[0], mSelfBvhContactBodyIndex,
		mSelfCollisionAdjacencies[0], mContactParams,
		mWorkspace.contact,
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
	PX_ASSERT(prepared);
	if(!prepared)
		return false;
	mSelfBvhContactTransactionPending = true;
	return true;
}

void AvbdCpuSoftScene::completeSelfBvhContactTaskTransaction()
{
	PX_ASSERT(mSelfBvhContactTransactionPending);
	const PxU32 selfStart = mContacts.size();
	for(PxU32 taskIndex = 0;
		taskIndex < mSelfBvhContactTaskOutputs.size(); ++taskIndex)
	{
		const PxArray<Dy::AvbdSoftContact>& source =
			mSelfBvhContactTaskOutputs[taskIndex];
		for(PxU32 contactIndex = 0; contactIndex < source.size();
			++contactIndex)
			mContacts.pushBack(source[contactIndex]);
		if(mCollisionStatsEnabled && taskIndex <
			mSelfBvhContactTaskStats.size())
			mLastCollisionStats.accumulate(
				mSelfBvhContactTaskStats[taskIndex]);
	}
	if(mCollisionStatsEnabled)
		mLastCollisionStats.generatedSelfContacts +=
			mContacts.size() - selfStart;
	completeComponentContactRedetection();
	mSelfBvhContactTransactionPending = false;
	mSelfBvhContactBodyIndex = PX_MAX_U32;
}

bool AvbdCpuSoftScene::completeStandaloneSelfBvhContactTask(
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
		!mSelfBvhContactTransactionPending)
		return false;
	completeSelfBvhContactTaskTransaction();
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

bool AvbdCpuSoftScene::finishStandaloneSelfBvhContactSerialFallback(
	PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
	bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
	bool& nextRigidBoxSdfContactTaskReady,
	bool& nextRigidSphereSdfContactTaskReady)
{
	if(!mSelfBvhContactTransactionPending)
		return false;
	mStandaloneTaskGraphTelemetry.
		recordSerialSelfBvhContactFallback();
	mSelfBvhContactTaskOutputs.clear();
	mSelfBvhContactTaskStats.clear();
	const Dy::AvbdSoftBody& body = mBodies[mSelfBvhContactBodyIndex];
	Dy::avbdDetectSelfCollisionOGCBvhRange(
		mParticles.begin(), body, mSelfBvhContactBodyIndex,
		mSelfCollisionAdjacencies[mSelfBvhContactBodyIndex],
		mWorkspace.contact, mSelfBvhSerialRangeWorkspace,
		0, body.compiled.surfaceVertices.size(),
		0, body.compiled.surfaceEdges.size(), mContacts, mContactParams,
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
	return completeStandaloneSelfBvhContactTask(
		dt, taskGraphContext, nextLayerReady,
		nextWorldPlaneContactTaskReady,
		nextRigidBoxSdfContactTaskReady,
		nextRigidSphereSdfContactTaskReady);
}

bool AvbdCpuSoftScene::canUseStaticWorldSelfOgcContactTaskTransaction() const {
			// This transaction changes the Scene redetection owner, so it must
			// only be admitted when it can actually submit at least two disjoint
			// workers.  Otherwise the ordinary serial callback remains both
			// cheaper and the authoritative fallback.
			if(!useAvbdStaticWorldSelfOgcTaskFanIn() ||
				mStandaloneTaskGraphEnhancedDeterminism ||
				mStandaloneTaskGraphDispatcherWorkers < 2 ||
				mParticles.size() <
					2u * eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK ||
				mBodies.size() != 1 ||
				mWorldPlanes.empty() || mRigidBoxes.empty() ||
				!mRigidSpheres.empty() || !mRigidCapsules.empty() ||
				!mRigidConvexes.empty() || !mRigidTriangleSurfaces.empty() ||
				mSelfCollisionEnabled.size() != 1 ||
				mSelfCollisionAdjacencies.size() != 1 ||
				!mSelfCollisionEnabled[0])
				return false;
			for(PxU32 boxIndex = 0; boxIndex < mRigidBoxes.size(); ++boxIndex)
			{
				if(mRigidBoxes[boxIndex].targetKind !=
					Dy::AvbdSoftContactTargetKind::eWORLD_STATIC)
					return false;
			}
			const Dy::AvbdSoftBody& body = mBodies[0];
			return body.compiled.surfaceVertices.size() >=
					2u * eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK &&
				body.compiled.surfaceEdges.size() >=
					2u * eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK &&
				Dy::avbdCanUseSelfCollisionOGCBvhRanges(body);
		}

PxU32 AvbdCpuSoftScene::getStaticWorldSelfOgcContactTaskCount(
			PxU32 dispatcherWorkers) const {
			if(!canUseStaticWorldSelfOgcContactTaskTransaction() ||
				dispatcherWorkers < 2)
				return 0;
			const Dy::AvbdSoftBody& body = mBodies[0];
			// Every aggregate child owns five private contact streams and two
			// self-BVH ranges.  Cap fan-out by useful work rather than blindly
			// mirroring a high-core dispatcher; range order and the canonical
			// merge stay unchanged.
			const PxU32 maximumTasks = PxMin(
				mParticles.size() /
					eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK,
				PxMin(body.compiled.surfaceVertices.size() /
						eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK,
					body.compiled.surfaceEdges.size() /
						eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK));
			return maximumTasks < 2 ? 0 :
				PxMin(dispatcherWorkers, maximumTasks);
		}

bool AvbdCpuSoftScene::beginStaticWorldSelfOgcContactTaskTransaction() {
			if(!canUseStaticWorldSelfOgcContactTaskTransaction() ||
				mWorldPlaneContactTransactionPending ||
				mRigidBoxSdfContactTransactionPending ||
				mRigidSphereSdfContactTransactionPending ||
				mRigidCapsuleSdfContactTransactionPending ||
				mRigidConvexSdfContactTransactionPending ||
				mRigidTriangleSurfaceContactTransactionPending ||
				mSelfBvhContactTransactionPending ||
				mStaticWorldSelfOgcContactTransactionPending)
				return false;

	beginComponentContactRedetection();
			Dy::avbdBuildSoftContactRedetectionPhasePlan(
				mWorkspace.contact, mWorldPlanes.size(), false,
				mRigidBoxes.size(), 0, 0, 0, 0, mBodies.size(),
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(), mSelfCollisionEnabled.begin());
			const bool prepared = Dy::avbdPrepareSelfCollisionOGCBvhRanges(
				mParticles.begin(), mBodies[0], 0, mSelfCollisionAdjacencies[0],
				mContactParams, mWorkspace.contact,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
			PX_ASSERT(prepared);
			if(!prepared)
			{
				mContacts.assign(mWorkspace.contact.epoch.previousContacts.begin(),
					mWorkspace.contact.epoch.previousContacts.end());
				return false;
			}
			mStaticWorldSelfOgcContactTransactionPending = true;
			return true;
		}

void AvbdCpuSoftScene::completeStaticWorldSelfOgcContactTaskTransaction() {
			PX_ASSERT(mStaticWorldSelfOgcContactTransactionPending);
			auto appendOutputs = [this](
				const PxArray<PxArray<Dy::AvbdSoftContact> >& outputs)
			{
				for(PxU32 taskIndex = 0; taskIndex < outputs.size(); ++taskIndex)
				{
					const PxArray<Dy::AvbdSoftContact>& source =
						outputs[taskIndex];
					for(PxU32 contactIndex = 0;
						contactIndex < source.size(); ++contactIndex)
						mContacts.pushBack(source[contactIndex]);
				}
			};

			const PxU32 groundStart = mContacts.size();
			appendOutputs(mStaticWorldSelfOgcWorldTaskOutputs);
			if(mCollisionStatsEnabled)
				mLastCollisionStats.generatedGroundContacts +=
					mContacts.size() - groundStart;

			const PxU32 rigidStart = mContacts.size();
			appendOutputs(mStaticWorldSelfOgcBoxTaskOutputs);
			appendOutputs(mStaticWorldSelfOgcBoxSweptTaskOutputs);
			// Feature OGC is intentionally parent-serial: the source stream now
			// has exactly the legacy all-current then all-swept prefix.
			Dy::avbdDetectSoftRigidOGCFeatures(
				mParticles.begin(), mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mBodies.begin(), mBodies.size(), mContacts,
				mContactParams.contactRadius,
				&mWorkspace.componentOgcGeometrySidecar);
			if(mCollisionStatsEnabled)
			{
				mLastCollisionStats.rigidParticleBoxTests +=
					PxU64(mParticles.size()) * mRigidBoxes.size();
				mLastCollisionStats.generatedRigidContacts +=
					mContacts.size() - rigidStart;
			}

			const PxU32 selfStart = mContacts.size();
			appendOutputs(mStaticWorldSelfOgcSelfVertexTaskOutputs);
			appendOutputs(mStaticWorldSelfOgcSelfEdgeTaskOutputs);
			if(mCollisionStatsEnabled)
			{
				for(PxU32 taskIndex = 0;
					taskIndex < mStaticWorldSelfOgcTaskStats.size(); ++taskIndex)
					mLastCollisionStats.accumulate(
						mStaticWorldSelfOgcTaskStats[taskIndex]);
				mLastCollisionStats.generatedSelfContacts +=
					mContacts.size() - selfStart;
			}

	completeComponentContactRedetection();
			mStaticWorldSelfOgcContactTransactionPending = false;
		}

bool AvbdCpuSoftScene::completeStandaloneStaticWorldSelfOgcContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& /*taskGraphContext*/,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady) {
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				!mStaticWorldSelfOgcContactTransactionPending)
				return false;
			completeStaticWorldSelfOgcContactTaskTransaction();
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

bool AvbdCpuSoftScene::finishStandaloneStaticWorldSelfOgcContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady) {
			if(!mStaticWorldSelfOgcContactTransactionPending)
				return false;
			mStandaloneTaskGraphTelemetry.
				recordSerialWorldPlaneContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialRigidBoxSdfContactFallback();
			mStandaloneTaskGraphTelemetry.
				recordSerialSelfBvhContactFallback();
			const Dy::AvbdSoftBody& body = mBodies[0];
			mStaticWorldSelfOgcWorldTaskOutputs.resize(1);
			mStaticWorldSelfOgcBoxTaskOutputs.resize(1);
			mStaticWorldSelfOgcBoxSweptTaskOutputs.resize(1);
			mStaticWorldSelfOgcSelfVertexTaskOutputs.resize(1);
			mStaticWorldSelfOgcSelfEdgeTaskOutputs.resize(1);
			mStaticWorldSelfOgcTaskStats.resize(1);
			mStaticWorldSelfOgcWorldTaskOutputs[0].clear();
			mStaticWorldSelfOgcBoxTaskOutputs[0].clear();
			mStaticWorldSelfOgcBoxSweptTaskOutputs[0].clear();
			mStaticWorldSelfOgcSelfVertexTaskOutputs[0].clear();
			mStaticWorldSelfOgcSelfEdgeTaskOutputs[0].clear();
			mStaticWorldSelfOgcTaskStats[0] = Dy::AvbdSoftCollisionStats();
			Dy::avbdDetectSoftWorldPlaneContactsRange(
				mParticles.begin(), mParticles.size(), 0, mParticles.size(),
				mWorldPlanes.begin(), mWorldPlanes.size(),
				mStaticWorldSelfOgcWorldTaskOutputs[0],
				mContactParams.contactRadius, &body, 1);
			Dy::avbdDetectSoftRigidSDFRange(
				mParticles.begin(), mParticles.size(), 0, mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mStaticWorldSelfOgcBoxTaskOutputs[0],
				mContactParams.contactRadius,
				mWorkspace.contact.epoch.previousContacts.begin(),
				mWorkspace.contact.epoch.previousContacts.size(), &body, 1);
			Dy::avbdDetectSoftRigidSweptSDFRange(
				mParticles.begin(), mParticles.size(), 0, mParticles.size(),
				mRigidBoxes.begin(), mRigidBoxes.size(),
				mStaticWorldSelfOgcBoxSweptTaskOutputs[0],
				mContactParams.contactRadius, &body, 1);
			Dy::avbdDetectSelfCollisionOGCBvhRange(
				mParticles.begin(), body, 0, mSelfCollisionAdjacencies[0],
				mWorkspace.contact, mSelfBvhSerialRangeWorkspace, 0,
				body.compiled.surfaceVertices.size(), 0, 0,
				mStaticWorldSelfOgcSelfVertexTaskOutputs[0], mContactParams,
				mCollisionStatsEnabled ? &mStaticWorldSelfOgcTaskStats[0] : NULL);
			Dy::avbdDetectSelfCollisionOGCBvhRange(
				mParticles.begin(), body, 0, mSelfCollisionAdjacencies[0],
				mWorkspace.contact, mSelfBvhSerialRangeWorkspace, 0, 0, 0,
				body.compiled.surfaceEdges.size(),
				mStaticWorldSelfOgcSelfEdgeTaskOutputs[0], mContactParams,
				mCollisionStatsEnabled ? &mStaticWorldSelfOgcTaskStats[0] : NULL);
			return completeStandaloneStaticWorldSelfOgcContactTask(
				dt, taskGraphContext, nextLayerReady,
				nextWorldPlaneContactTaskReady,
				nextRigidBoxSdfContactTaskReady,
				nextRigidSphereSdfContactTaskReady);
		}

} // namespace Sc
} // namespace physx
