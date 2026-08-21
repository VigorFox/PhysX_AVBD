// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		// P5 collision-leaf tasks consume the direct simulation topology, while
		// the ordinary component lifecycle expands a cooked collision mesh back
		// to simulation-space Jacobians synchronously.  Keep that narrower P5
		// capability separate from P2/P3/P4/P6 scheduling: a distinct collision
		// mesh must not demote prediction, body-local primal work, or write-back
		// to the scalar reference path.
		bool AvbdCpuSoftScene::hasDirectSimulationCollisionDomain() const
		{
			for(PxU32 entryIndex = 0; entryIndex < mEntries.size(); ++entryIndex)
			{
				const Entry& entry = mEntries[entryIndex];
				if(entry.kind == eVOLUME &&
					entry.collisionMesh != entry.simulationMesh)
					return false;
			}
			return true;
		}

		bool AvbdCpuSoftScene::shouldScheduleStandaloneTaskGraph(
			PxU32 dispatcherWorkers) const
		{
			// A complete native rigid/soft selection already owns its step in
			// Dy::AvbdDynamicsContext.  Only a component-only pure-soft step may
			// be delegated here; never overlap the two ownership paths.
			const bool forceP45CausalLayerTaskFanIn =
				Dy::avbdUseCausalLayerTaskFanIn() &&
				Dy::avbdForceCausalLayerTaskFanIn();
			const bool forceP45CausalLayerTaskGraphReference =
				Dy::avbdForceCausalLayerTaskGraphReference();
			const bool forceP45CausalLayerTaskGraph =
				forceP45CausalLayerTaskFanIn ||
				forceP45CausalLayerTaskGraphReference;
			if(mDynamicsOwnsStep || mStandaloneTaskGraphEnhancedDeterminism ||
				dispatcherWorkers == 0 ||
				(!forceP45CausalLayerTaskGraph &&
					mParticles.size() < 128) || mBodies.empty())
				return false;
			if(forceP45CausalLayerTaskGraph)
				return true;

			// P2 deliberately creates one complete stage task, not a parallel
			// particle solve.  The estimate still accounts for the work that the
			// later P3/P4 graph will divide: material elements, current contact
			// set, collision sources and the fixed dispatch/barrier cost.  It is
			// independent of rigid body count, which is zero for this path.
			PxU64 materialElements = 0;
			for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); bodyIndex++)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					mBodies[bodyIndex].compiled;
				materialElements += compiled.triElements.size();
				materialElements += compiled.tetElements.size();
				materialElements += compiled.bendElements.size();
			}
			const PxU64 collisionSources =
				PxU64(mWorldPlanes.size()) + PxU64(mRigidBoxes.size()) +
				PxU64(mRigidSpheres.size()) + PxU64(mRigidCapsules.size()) +
				PxU64(mRigidConvexes.size()) +
				PxU64(mRigidTriangleSurfaces.size()) +
				(mBodies.size() > 1 ? PxU64(mBodies.size() - 1) : 0);
			const PxU64 estimatedWork =
				PxU64(mParticles.size()) + materialElements * 2 +
				PxU64(mContacts.size()) * 16 + collisionSources * 32;
			const PxU64 dispatchAndBarrierCost = 512;
			return estimatedWork >= dispatchAndBarrierCost * 2;
		}

		// Resolve the component particle schedule at the Scene boundary, where
		// worker availability and the public determinism promise are known.  The
		// default production route is relaxed coloring on a useful parallel
		// workload; an explicit process policy remains available for diagnostics
		// and forced fallback.  Enhanced determinism is a contract, not a tuning
		// hint, so it always retains the ordered scalar authority.
		Dy::AvbdParticlePrimalSchedule AvbdCpuSoftScene::getParticlePrimalSchedule() const
		{
			const Dy::AvbdParticlePrimalSchedule configured =
				Dy::avbdGetConfiguredParticlePrimalSchedule();
			if(mStandaloneTaskGraphEnhancedDeterminism)
				return Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR;
			if(configured != Dy::AvbdParticlePrimalSchedule::eDEFAULT)
				return configured;
			return mStandaloneTaskGraphDispatcherWorkers >= 2 &&
				mParticles.size() >= 128
				? Dy::AvbdParticlePrimalSchedule::eRELAXED_COLOR
				: Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR;
		}

		// A dense persistent soft/soft manifold has many mutually dependent
		// vertices and therefore produces many short color layers. Publishing
		// every one as a dispatcher task turns the layer barriers into the
		// dominant cost (especially with two workers). Keep the identical relaxed
		// color plan, but execute its ordered layers inline in the component
		// continuation once the previous epoch proves that this is a contact-dense
		// manifold. This follows the same batching principle used by CPU XPBD
		// solvers: parallelize useful independent batches, not synchronization
		// points between tiny constraint colors.
		bool AvbdCpuSoftScene::shouldInlineDenseSoftPairColoredPrimal() const
		{
			if(mStandaloneParticlePrimalSchedule !=
				Dy::AvbdParticlePrimalSchedule::eRELAXED_COLOR ||
				mStandaloneTaskGraphDispatcherWorkers < 2)
				return false;

			static const PxU32 eMIN_SOFT_PAIR_ROWS_FOR_INLINE_BATCH = 16;
			PxU32 softPairRows = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < mContacts.size(); ++contactIndex)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					mContacts[contactIndex].geometry;
				if(geometry.source.type ==
						Dy::AvbdSoftContactSource::eSOFT_SURFACE &&
					geometry.targetKind ==
						Dy::AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
					++softPairRows >= eMIN_SOFT_PAIR_ROWS_FOR_INLINE_BATCH)
					return true;
			}
			return false;
		}

		PxU32 AvbdCpuSoftScene::getStandaloneTaskGraphParticleCount() const
		{
			return mParticles.size();
		}

		void AvbdCpuSoftScene::setStandaloneTaskGraphExecutionPolicy(
			PxU32 workerCount, bool enhancedDeterminism)
		{
			// A new Scene simulation cannot overlap the prior continuation.  Reset
			// the boundary-only counters here, before any child can be submitted.
			mStandaloneTaskGraphTelemetry.reset(workerCount);
			mStandaloneTaskGraphDispatcherWorkers = workerCount;
			mStandaloneTaskGraphEnhancedDeterminism = enhancedDeterminism;
			mStandaloneParticlePrimalSchedule =
				getParticlePrimalSchedule();
			// A scene may acquire its dispatcher after bodies were added. Reserve
			// the optional graph storage at that Scene boundary (never from a
			// worker) so changing from one worker to a relaxed production policy
			// cannot silently publish an incomplete plan for the whole frame.
			if(Dy::avbdUsesColoredParticlePrimalSchedule(
				mStandaloneParticlePrimalSchedule))
				mWorkspace.reserve(
					mParticles.size(), estimateInitialComponentContactCapacity(),
					mStandaloneParticlePrimalSchedule);
		}

		bool AvbdCpuSoftScene::canUseIndependentBodySweepTaskFanIn() const
		{
			if(mStandaloneTaskGraphDispatcherWorkers < 2 ||
				Dy::avbdDisableIndependentBodySweepTaskFanIn() ||
				mBodies.size() < 2 || !mContacts.empty() ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() || !mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				!mRigidAttachments.empty() ||
				!mArticulationAttachments.empty() ||
				!mSoftPairAttachments.empty() ||
				!mPrescribedAttachments.empty() ||
				mSelfCollisionEnabled.size() != mBodies.size())
				return false;
			for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); bodyIndex++)
			{
				// A P6 task owns a whole soft body.  A Scene-owned world-plane
				// redetection bridge is compatible with that ownership: the bridge
				// completes before beginInnerSweep(), which publishes this route only
				// while the freshly detected contact epoch is empty.  The first plane
				// contact therefore falls back to the colored contact owner in the
				// same frame rather than permanently disabling body-level batching.
				// With an empty contact epoch,
				// self-collision configuration and kinematic pins remain strictly
				// body-local: their primal writes are confined to this task's
				// particle range and their AL dual update stays at the parent
				// barrier.  Do not reject a production fast path merely because it
				// cannot reproduce the legacy global traversal.  Attachments still
				// couple another generalized owner and must remain serial here.
				if(!mBodies[bodyIndex].runtime.attachments.empty())
					return false;
			}
			return true;
		}

		// P3 prediction setup: all mutable Scene/OGC state up to the exact
		// low-level prediction boundary is complete when this returns. The
		// resulting particle write set is disjoint by whole Entry.
		bool AvbdCpuSoftScene::prepareStandaloneComponentSolve(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
			PX_UNUSED(gravity);
			mStandaloneComponentPostSolvePending = false;
			mStandaloneComponentSolvePrepared = false;
			mLastStepStats.reset();
			if(mCollisionStatsEnabled)
				mLastCollisionStats = Dy::AvbdSoftCollisionStats();
			mLastComponentFallbackSteps = 0;
			mLastNativeIslandSteps = 0;
			if(dt <= 0.0f || mBodies.empty())
				return false;

			PxU32 awakeEntryCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(!mEntries[i].sleeping)
					awakeEntryCount++;
			if(awakeEntryCount == 0 && sleepingEnabled)
			{
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return false;
			}

			// A native rigid/soft island owns both its solve results and its
			// selection restore.  P3 must never split that path.
			PX_ASSERT(!mDynamicsOwnsStep);
			if(mDynamicsOwnsStep)
				return false;

			// This entry reaches the asynchronous component task graph instead of
			// `step()`.  Preserve the same single-owner rule as the serial
			// fallback before it starts mutating canonical particle/contact state.
			invalidateNativeIslandSelectionCaches();
			prepareComponentFallback(materialManager, rigidMaterialManager);
			mStandaloneStepStats.reset();
			Dy::avbdStepSoftBodies(
				mParticles.begin(), mParticles.size(),
				mBodies.begin(), mBodies.size(),
				mContacts.begin(), mContacts.size(),
				dt, gravity,
				mComponentFallbackPlan.outerIterations,
				mComponentFallbackPlan.innerIterations,
				1000.0f,
				redetectContacts, &mContacts, this,
				0.92f, &mStandaloneStepStats, &mWorkspace,
				mComponentFallbackPlan.totalPositionIterations,
				mSelfCollisionAdjacencies.begin(),
				mSelfCollisionAdjacencies.size(),
				mSelfCollisionEnabled.begin(),
				&mContactParams,
				Dy::AvbdSoftBodyStepExecutionMode::ePREPARE);
			mStandaloneComponentSolvePrepared = true;
			return true;
		}

		void AvbdCpuSoftScene::predictStandaloneComponentRange(
			PxU32 entryBegin, PxU32 entryEnd,
			PxReal dt, const PxVec3& gravity)
		{
			PX_ASSERT(mStandaloneComponentSolvePrepared);
			PX_ASSERT(entryBegin <= entryEnd && entryEnd <= mEntries.size());
			// Soft-soft swept admission selects its path if either endpoint opts
			// into speculative CCD.  Keep the adaptive initial guess component
			// wide so one P3 entry cannot shorten only its side of that sweep.
			const bool useRigidInitialGuess =
				Dy::avbdCanUseSoftRigidPrimalInitialization(
					mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size());
			const bool useAdaptiveInitialGuess = !useRigidInitialGuess &&
				Dy::avbdCanUseSoftAdaptivePrimalInitialization(
					mParticles.begin(), mParticles.size(),
					mBodies.begin(), mBodies.size());
			for(PxU32 entryIndex = entryBegin; entryIndex < entryEnd;
				entryIndex++)
			{
				const Entry& entry = mEntries[entryIndex];
				PX_ASSERT(entry.bodyIndex < mBodies.size());
				PX_ASSERT(entry.bodyIndex <
					mWorkspace.contact.softBodyBounds.size());
				PX_ASSERT(entry.bodyIndex <
					mWorkspace.contact.softBodyBoundsReady.size());
				Dy::avbdPredictSoftBodyParticles(
					mParticles.begin() + getParticleStart(entry),
					getParticleCount(entry), dt, gravity,
					useAdaptiveInitialGuess);
				if(useRigidInitialGuess)
					Dy::avbdApplySoftBodyRigidPrimalInitialGuess(
						mParticles.begin(), mParticles.size(),
						mBodies[entry.bodyIndex]);
				// P3 Slice 3: this entry exclusively owns its body and the
				// corresponding pre-sized bounds slot. The later continuation
				// alone marks the whole cache valid after every child completes.
				Dy::avbdComputeSoftBodyBounds(
					mParticles.begin(), mBodies[entry.bodyIndex],
					mWorkspace.contact.softBodyBounds[entry.bodyIndex]);
				mWorkspace.contact.softBodyBoundsReady[entry.bodyIndex] = 1;
			}
		}

		void AvbdCpuSoftScene::predictStandaloneComponent(
			PxReal dt, const PxVec3& gravity)
		{
			predictStandaloneComponentRange(0, mEntries.size(), dt, gravity);
		}

		PxU32 AvbdCpuSoftScene::getStandalonePredictionTaskCount(
			PxU32 dispatcherWorkers) const
		{
			if(!mStandaloneComponentSolvePrepared || mDynamicsOwnsStep ||
				dispatcherWorkers < 2)
				return 0;
			PxU32 awakeEntryCount = 0;
			PxU64 awakeParticleCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				const Entry& entry = mEntries[i];
				if(entry.sleeping)
					continue;
				awakeEntryCount++;
				awakeParticleCount += getParticleCount(entry);
			}
			// Prediction is streaming work, but its child/fan-in cost is paid
			// before any nonlinear work can start. Keep the initial threshold
			// conservative and identical to P3 write-back until timing data can
			// tune it separately.
			if(awakeEntryCount < 2 || awakeParticleCount < 1024)
				return 0;
			return PxMin(dispatcherWorkers, awakeEntryCount);
		}

		bool AvbdCpuSoftScene::submitStandaloneCausalLayerTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext)
		{
			if(!mStandaloneComponentSolvePrepared || !continuation)
				return false;
			const Dy::AvbdParticlePrimalSolveContext* bodySolveContext = NULL;
			const Dy::AvbdSoftBody* bodyRangeBodies = NULL;
			PxU32 bodyRangeCount = 0;
			const bool independentBodySweep =
				mStandaloneComponentStepState.
					getPublishedIndependentBodySweep(
						bodySolveContext, bodyRangeBodies, bodyRangeCount);
			PxU32 layerIndex = 0;
			PxU32 packedBegin = 0;
			PxU32 packedEnd = 0;
			const Dy::AvbdParticlePrimalSolveContext* solveContext = NULL;
			const Dy::AvbdSoftBody* bodies = NULL;
			PxU32 bodyCount = 0;
			const PxU32* particleBodyIndices = NULL;
			const PxU32* packedParticleIndices = NULL;
			if(independentBodySweep)
			{
				solveContext = bodySolveContext;
				bodies = bodyRangeBodies;
				bodyCount = bodyRangeCount;
				if(!solveContext || !bodies || bodyCount < 2)
					return false;
			}
			else if(!mStandaloneComponentStepState.getPublishedCausalLayer(
					layerIndex, packedBegin, packedEnd, solveContext, bodies,
					bodyCount, particleBodyIndices, packedParticleIndices) ||
					packedBegin >= packedEnd || !solveContext || !bodies ||
					!particleBodyIndices || !packedParticleIndices)
				return false;
			PX_UNUSED(layerIndex);
			PxU32 layerOccupancy = independentBodySweep ? 0u :
				packedEnd - packedBegin;
			if(independentBodySweep)
			{
				for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
					layerOccupancy +=
						bodies[bodyIndex].compiled.particleCount;
			}
			PxU64 independentBodyTotalWork = 0;
			PxU64 independentBodyMaximumWork = 0;
			if(independentBodySweep)
			{
				for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
				{
					const PxU64 bodyWork =
						getIndependentBodySweepWorkEstimate(bodies[bodyIndex]);
					independentBodyTotalWork += bodyWork;
					independentBodyMaximumWork =
						PxMax(independentBodyMaximumWork, bodyWork);
				}
			}
			PxU32 independentBodyTaskCount = 0;
			if(independentBodySweep)
			{
				PX_ASSERT(independentBodyMaximumWork > 0);
				const PxU64 wholeTasks =
					independentBodyTotalWork / independentBodyMaximumWork;
				const PxU64 remainder =
					independentBodyTotalWork % independentBodyMaximumWork;
				// A body is indivisible. Do not add a child for a tiny residual
				// range unless it represents at least half of the largest body's
				// primal work; that child cannot lower the dominant makespan enough
				// to repay another dispatch and fan-in transaction.
				const PxU32 usefulTaskCount = PxU32(
					wholeTasks +
					(remainder >= independentBodyMaximumWork -
						independentBodyMaximumWork / 2 ? 1u : 0u));
				independentBodyTaskCount = PxMin(
					dispatcherWorkers,
					PxMin(bodyCount, PxMax(2u, usefulTaskCount)));
			}
			const PxU32 taskCount = independentBodySweep ?
				independentBodyTaskCount :
				getCausalLayerTaskCount(dispatcherWorkers, layerOccupancy);
			if(taskCount == 0)
				return false;
			if(!ensureCausalLayerTaskPool(taskCount, owner, taskGraphContext) ||
				!hasCausalLayerTaskSlots(taskCount))
				return false;
			// Every range is a stable contiguous subinterval of the published
			// packed layer. Layer construction has already rejected every
			// structural/dynamic same-layer read/write conflict, so task ownership
			// is the range alone. The parent reduction remains ascending range
			// order regardless of dispatcher completion order.
			mCausalLayerRangeObservations.resize(taskCount);
			PX_ASSERT(mCausalLayerRangeObservations.capacity() >= taskCount);
			for(PxU32 taskIndex = 0; taskIndex < taskCount; taskIndex++)
				mCausalLayerRangeObservations[taskIndex] =
					Dy::AvbdParticlePrimalRangeObservation();
			CausalLayerFinishTask* const finishTask =
				acquireCausalLayerFinishTask();
			if(!finishTask)
			{
				return false;
			}
			finishTask->setContinuation(continuation);
			mStandaloneTaskGraphTelemetry.recordCausalLayerTasksSubmitted(
				taskCount, layerOccupancy);
			const PxU32 particlesPerTask = independentBodySweep ? 0u :
				(layerOccupancy + taskCount - 1) / taskCount;
			PxU32 nextTaskBodyBegin = 0;
			PxU64 nextTaskBodyPrefixWork = 0;
			for(PxU32 taskIndex = 0; taskIndex < taskCount; taskIndex++)
			{
				PxU32 taskBodyBegin = 0;
				PxU32 taskBodyEnd = 0;
				if(independentBodySweep)
				{
					taskBodyBegin = nextTaskBodyBegin;
					if(taskIndex + 1 == taskCount)
					{
						taskBodyEnd = bodyCount;
						nextTaskBodyPrefixWork = independentBodyTotalWork;
					}
					else
					{
						// Keep every task non-empty and choose the closest cumulative
						// work boundary. Ranges remain contiguous whole-body intervals,
						// so dispatcher completion order cannot change solve ownership.
						const PxU32 remainingTasks = taskCount - taskIndex - 1;
						const PxU32 maximumBodyEnd = bodyCount - remainingTasks;
						const PxU64 targetWork = getIndependentBodySweepTarget(
							independentBodyTotalWork, taskIndex + 1, taskCount);
						taskBodyEnd = taskBodyBegin + 1;
						nextTaskBodyPrefixWork +=
							getIndependentBodySweepWorkEstimate(
								bodies[taskBodyBegin]);
						while(taskBodyEnd < maximumBodyEnd)
						{
							const PxU64 extendedPrefixWork =
								nextTaskBodyPrefixWork +
								getIndependentBodySweepWorkEstimate(
									bodies[taskBodyEnd]);
							if(getIndependentBodySweepDistance(
									extendedPrefixWork, targetWork) >
								getIndependentBodySweepDistance(
									nextTaskBodyPrefixWork, targetWork))
								break;
							nextTaskBodyPrefixWork = extendedPrefixWork;
							taskBodyEnd++;
						}
					}
					nextTaskBodyBegin = taskBodyEnd;
				}
				const PxU32 taskPackedBegin =
					independentBodySweep ? 0u :
						packedBegin + taskIndex * particlesPerTask;
				const PxU32 taskPackedEnd = independentBodySweep ? 0u :
					PxMin(taskPackedBegin + particlesPerTask, packedEnd);
				PX_ASSERT(independentBodySweep ?
					taskBodyBegin < taskBodyEnd :
					taskPackedBegin < taskPackedEnd);
				CausalLayerTask* const task = acquireCausalLayerTask();
				PX_ASSERT(task);
				if(!task)
				{
					// Slot availability was checked before the finish task was
					// acquired. This is defensive only; no child has been
					// submitted before this point in a valid Scene lifecycle.
					recycleCausalLayerFinishTask(finishTask->getPoolIndex());
					return false;
				}
				if(independentBodySweep)
					task->configureIndependentBodyRange(
						*solveContext, bodies, bodyCount,
						taskBodyBegin, taskBodyEnd,
						mCausalLayerRangeObservations[taskIndex]);
				else
					task->configure(
						*solveContext, bodies, bodyCount, particleBodyIndices,
						mParticles.size(), packedParticleIndices,
						taskPackedBegin, taskPackedEnd,
						mCausalLayerRangeObservations[taskIndex]);
				task->setContinuation(finishTask);
				task->removeReference();
			}
			PX_ASSERT(!independentBodySweep ||
				(nextTaskBodyBegin == bodyCount &&
				 nextTaskBodyPrefixWork == independentBodyTotalWork));
			finishTask->removeReference();
			return true;
		}

		// Keep the ownership decision stable for every outer epoch of the
		// resumable solve.  The aggregate is a Scene-owned redetection bridge in
		// exactly the same sense as the legacy P5 leaves; testing only the raw
		// environment bridge here would run its first epoch in parallel and then
		// silently fall back to the synchronous callback.
		bool AvbdCpuSoftScene::usesStandaloneSceneRedetectionBridge() const
		{
			return hasDirectSimulationCollisionDomain() &&
				(canUseWorldPlaneContactTaskTransaction() ||
					canUseRigidBoxSdfContactTaskTransaction() ||
					canUseRigidSphereSdfContactTaskTransaction() ||
					canUseRigidCapsuleSdfContactTaskTransaction() ||
					canUseRigidConvexSdfContactTaskTransaction() ||
					canUseRigidTriangleSurfaceContactTaskTransaction() ||
					canUseStaticWorldSelfOgcContactTaskTransaction());
		}

}
}
