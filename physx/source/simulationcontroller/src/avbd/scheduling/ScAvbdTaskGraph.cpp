// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

void AvbdCpuSoftScene::submitStandalonePredictionTasks(
	PxU32 taskCount, PxReal dt, const PxVec3& gravity,
	PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	PX_ASSERT(continuation);
	PX_ASSERT(taskCount == getStandalonePredictionTaskCount(
		PxMax(taskCount, 2u)));
	PX_ASSERT(taskCount > 0 && taskCount <= mEntries.size());
	while(mPredictionTasks.size() < taskCount)
		mPredictionTasks.pushBack(PX_NEW(PredictionTask)(
			mContextId, *this));
	const PxU32 entriesPerTask =
		(mEntries.size() + taskCount - 1) / taskCount;
	mStandaloneTaskGraphTelemetry.recordPredictionTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 entryBegin = taskIndex * entriesPerTask;
		const PxU32 entryEnd = PxMin(
			entryBegin + entriesPerTask, mEntries.size());
		PX_ASSERT(entryBegin < entryEnd);
		PredictionTask& task = *mPredictionTasks[taskIndex];
		task.configure(entryBegin, entryEnd, dt, gravity);
		task.setContinuation(continuation);
	task.removeReference();
	}
}

		bool AvbdCpuSoftScene::ensureCausalLayerTaskPool(
			PxU32 requiredChildTasks, Scene& owner,
			Dy::AvbdDynamicsContext& /*taskGraphContext*/){
			// One finishing task can be running while it publishes the next
			// layer's finish task, hence two parent-task slots in addition to the
			// currently submitted children.  Children are capped by dispatcher
			// worker count at the parent partition policy.
			const PxU32 requiredSlots = PxMax(requiredChildTasks, 1u) + 2u;
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			if(mCausalLayerTasks.size() >= requiredSlots &&
				mCausalLayerFinishTasks.size() >= requiredSlots &&
				mCausalLayerRangeObservations.capacity() >=
					requiredChildTasks)
				return true;
			// This function is called only by a parent continuation before it
			// submits a layer. Reserve free-list capacity first so dispatcher
			// release never allocates while recycling a task.
			mCausalLayerTasks.reserve(requiredSlots);
			mCausalLayerFinishTasks.reserve(requiredSlots);
			mFreeCausalLayerTaskIndices.reserve(requiredSlots);
			mFreeCausalLayerFinishTaskIndices.reserve(requiredSlots);
			// The parent clears/resizes one observation per child before making
			// any task runnable. Reserve here so neither a child release nor a
			// later same-size layer causes an observation-array allocation.
			mCausalLayerRangeObservations.reserve(requiredChildTasks);
			while(mCausalLayerTasks.size() < requiredSlots)
			{
				const PxU32 index = mCausalLayerTasks.size();
				mCausalLayerTasks.pushBack(PX_NEW(CausalLayerTask)(
					mContextId, *this, index));
				mFreeCausalLayerTaskIndices.pushBack(index);
			}
			while(mCausalLayerFinishTasks.size() < requiredSlots)
			{
				const PxU32 index = mCausalLayerFinishTasks.size();
				mCausalLayerFinishTasks.pushBack(PX_NEW(CausalLayerFinishTask)(
					mContextId, *this, owner, index));
				mFreeCausalLayerFinishTaskIndices.pushBack(index);
			}
			return true;
		}


		AvbdCpuSoftScene::CausalLayerTask* AvbdCpuSoftScene::acquireCausalLayerTask(){
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			if(mFreeCausalLayerTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeCausalLayerTaskIndices.back();
			mFreeCausalLayerTaskIndices.popBack();
			PX_ASSERT(index < mCausalLayerTasks.size());
			return mCausalLayerTasks[index];
		}


		AvbdCpuSoftScene::CausalLayerFinishTask* AvbdCpuSoftScene::acquireCausalLayerFinishTask(){
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			if(mFreeCausalLayerFinishTaskIndices.empty())
				return NULL;
			const PxU32 index = mFreeCausalLayerFinishTaskIndices.back();
			mFreeCausalLayerFinishTaskIndices.popBack();
			PX_ASSERT(index < mCausalLayerFinishTasks.size());
			return mCausalLayerFinishTasks[index];
		}


		void AvbdCpuSoftScene::recycleCausalLayerTask(PxU32 index){
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			PX_ASSERT(index < mCausalLayerTasks.size());
			PX_ASSERT(mFreeCausalLayerTaskIndices.size() <
				mCausalLayerTasks.size());
			mFreeCausalLayerTaskIndices.pushBack(index);
		}


		void AvbdCpuSoftScene::recycleCausalLayerFinishTask(PxU32 index){
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			PX_ASSERT(index < mCausalLayerFinishTasks.size());
			PX_ASSERT(mFreeCausalLayerFinishTaskIndices.size() <
				mCausalLayerFinishTasks.size());
			mFreeCausalLayerFinishTaskIndices.pushBack(index);
		}


		bool AvbdCpuSoftScene::hasCausalLayerTaskSlots(PxU32 taskCount){
			PxMutex::ScopedLock lock(mCausalLayerTaskPoolMutex);
			return mFreeCausalLayerTaskIndices.size() >= taskCount &&
				!mFreeCausalLayerFinishTaskIndices.empty();
		}

		PxU32 AvbdCpuSoftScene::getCausalLayerTaskCount(
			PxU32 dispatcherWorkers, PxU32 layerOccupancy) const
		{
			// The ordered P4 path stays a one-range reference unless its explicit
			// validation switch is selected.  Relaxed colors are the production
			// throughput path: split a conflict-free color once it amortizes the
			// dispatch/fan-in boundary, while preserving the same owner proof.
			static const PxU32 eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK = 16;
			const bool forceSmallLayerPartition =
				Dy::avbdForceCausalLayerTaskPartition();
			const bool relaxedColorFastPath =
				mStandaloneParticlePrimalSchedule ==
					Dy::AvbdParticlePrimalSchedule::eRELAXED_COLOR;
			if((!Dy::avbdUseCausalLayerTaskPartition() &&
				!relaxedColorFastPath) ||
				dispatcherWorkers < 2 ||
				(!forceSmallLayerPartition && layerOccupancy <
					eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK))
				return 1;
			const PxU32 maxTasksByOccupancy =
				forceSmallLayerPartition ? layerOccupancy :
					(layerOccupancy +
						eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK - 1) /
						eMIN_PARTICLES_PER_CAUSAL_LAYER_TASK;
			return PxMin(PxMin(dispatcherWorkers, maxTasksByOccupancy),
				layerOccupancy);
		}

void AvbdCpuSoftScene::submitStandaloneWriteBackTasks(
	PxU32 taskCount, PxBaseTask* continuation,
	Dy::AvbdDynamicsContext& /*taskGraphContext*/)
{
	PX_ASSERT(continuation);
	PX_ASSERT(taskCount == getStandaloneWriteBackTaskCount(
		PxMax(taskCount, 2u)));
	PX_ASSERT(taskCount > 0 && taskCount <= mEntries.size());
	while(mWriteBackTasks.size() < taskCount)
		mWriteBackTasks.pushBack(PX_NEW(WriteBackTask)(mContextId, *this));

	// Ranges are stable and aligned to whole entries. Each task writes only
	// its entries' particle/output buffers; a later fan-in owns the shared
	// sleeping/island continuation.
	const PxU32 entriesPerTask =
		(mEntries.size() + taskCount - 1) / taskCount;
	mStandaloneTaskGraphTelemetry.recordWriteBackTasksSubmitted(taskCount);
	for(PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex)
	{
		const PxU32 entryBegin = taskIndex * entriesPerTask;
		const PxU32 entryEnd = PxMin(
			entryBegin + entriesPerTask, mEntries.size());
		PX_ASSERT(entryBegin < entryEnd);
		WriteBackTask& task = *mWriteBackTasks[taskIndex];
		task.configure(entryBegin, entryEnd);
		task.setContinuation(continuation);
		task.removeReference();
	}
}

Dy::AvbdSoftBodyStepAdvanceResult
AvbdCpuSoftScene::advanceStandaloneComponentStateWithSceneRedetection(
		bool allowWorldPlaneTask,
		bool* worldPlaneContactTaskReady,
		bool allowRigidBoxSdfTask,
		bool* rigidBoxSdfContactTaskReady,
		bool allowRigidSphereSdfTask,
		bool* rigidSphereSdfContactTaskReady)
{
	if(worldPlaneContactTaskReady)
		*worldPlaneContactTaskReady = false;
	if(rigidBoxSdfContactTaskReady)
		*rigidBoxSdfContactTaskReady = false;
	if(rigidSphereSdfContactTaskReady)
		*rigidSphereSdfContactTaskReady = false;
	for(;;)
	{
		const Dy::AvbdSoftBodyStepAdvanceResult result =
			mStandaloneComponentStepState.advance();
		if(result !=
			Dy::AvbdSoftBodyStepAdvanceResult::eREDETECTION_READY)
			return result;
		// The aggregate must claim this epoch before the source-specific
		// leaves. It owns one Begin/Complete pair and reconstructs their
		// serial source order after all private children have joined.
		if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
			beginStaticWorldSelfOgcContactTaskTransaction())
		{
			*rigidSphereSdfContactTaskReady = true;
			return result;
		}
		if(allowWorldPlaneTask && worldPlaneContactTaskReady &&
			beginWorldPlaneContactTaskTransaction())
		{
			*worldPlaneContactTaskReady = true;
			return result;
		}
		if(allowRigidBoxSdfTask && rigidBoxSdfContactTaskReady &&
			beginRigidBoxSdfContactTaskTransaction())
		{
			*rigidBoxSdfContactTaskReady = true;
			return result;
		}
		if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
			beginRigidSphereSdfContactTaskTransaction())
		{
			*rigidSphereSdfContactTaskReady = true;
			return result;
		}
		// Static sphere and capsule leaves are intentionally mutually
		// exclusive. Reuse the existing smooth-SDF continuation readiness
		// bit, while keeping the transaction/pool/telemetry below separate.
		if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
			beginRigidCapsuleSdfContactTaskTransaction())
		{
			*rigidSphereSdfContactTaskReady = true;
			return result;
		}
		if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
			beginRigidConvexSdfContactTaskTransaction())
		{
			*rigidSphereSdfContactTaskReady = true;
			return result;
		}
		if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
			beginRigidTriangleSurfaceContactTaskTransaction())
		{
			*rigidSphereSdfContactTaskReady = true;
			return result;
		}
		if(allowRigidSphereSdfTask && rigidSphereSdfContactTaskReady &&
			beginSelfBvhContactTaskTransaction())
		{
			*rigidSphereSdfContactTaskReady = true;
			return result;
		}
		// The state has published its only redetection boundary.  The
		// Scene parent owns callback execution, mutable contact storage,
		// filtering, state transfer, trace and the final post-detection
		// index rebuild; a future P5 child graph may replace only this
		// synchronous body with a candidate fan-in.
		redetectContacts(
			mParticles.begin(), mParticles.size(),
			mBodies.begin(), mBodies.size(), mContacts, this);
		if(!mStandaloneComponentStepState.
			completePendingRedetection())
			return Dy::AvbdSoftBodyStepAdvanceResult::eINVALID;
	}
}

bool AvbdCpuSoftScene::runStandaloneComponentStateWithSceneRedetection()
{
	for(;;)
	{
		const Dy::AvbdSoftBodyStepAdvanceResult result =
			advanceStandaloneComponentStateWithSceneRedetection();
		if(result == Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
			return true;
		if(result !=
			Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			return false;
		PxU32 layerIndex = 0;
		PxU32 packedBegin = 0;
		PxU32 packedEnd = 0;
		const Dy::AvbdParticlePrimalSolveContext* solveContext = NULL;
		const Dy::AvbdSoftBody* bodies = NULL;
		PxU32 bodyCount = 0;
		const PxU32* particleBodyIndices = NULL;
		const PxU32* packedParticleIndices = NULL;
		if(!mStandaloneComponentStepState.getPublishedCausalLayer(
			layerIndex, packedBegin, packedEnd, solveContext, bodies,
			bodyCount, particleBodyIndices, packedParticleIndices))
			return false;
		PX_UNUSED(layerIndex);
		Dy::AvbdParticlePrimalRangeObservation observation;
		Dy::avbdSolveParticlePrimalPackedRange(
			*solveContext, bodies, bodyCount, particleBodyIndices,
			mParticles.size(), packedParticleIndices,
			packedBegin, packedEnd, observation);
		if(!mStandaloneComponentStepState.
			completePublishedCausalLayer(&observation, 1))
			return false;
	}
}

void AvbdCpuSoftScene::finishStandaloneTaskGraphNoOp()
{
	// A submitted root can discover that every component went to sleep
	// before its prepare stage.  It still completed as a Scene task, but
	// has no component post-solve to close the boundary telemetry.
	mStandaloneTaskGraphTelemetry.endSolveTask();
}
void AvbdCpuSoftScene::writeBackStandaloneComponentRange(
			PxU32 entryBegin, PxU32 entryEnd) {
			PX_ASSERT(mStandaloneComponentPostSolvePending);
			PX_ASSERT(entryBegin <= entryEnd && entryEnd <= mEntries.size());
			for(PxU32 entryIndex = entryBegin; entryIndex < entryEnd;
				entryIndex++)
				writeBack(mEntries[entryIndex]);
		}

void AvbdCpuSoftScene::writeBackStandaloneComponent() {
			writeBackStandaloneComponentRange(0, mEntries.size());
		}

void AvbdCpuSoftScene::finishStandaloneComponentStep(
			PxReal dt, bool sleepingEnabled) {
			PX_ASSERT(mStandaloneComponentPostSolvePending);
			if(!mStandaloneComponentPostSolvePending)
				return;
			updateSleepStates(dt, sleepingEnabled);
			mWorkspacePreflightPending = false;
			mDynamicsSelectedEntryCount = 0;
			mStandaloneComponentPostSolvePending = false;
		}

} // namespace Sc
} // namespace physx
