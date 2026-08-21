// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		bool AvbdCpuSoftScene::completeStandaloneCausalLayerTask(
			PxReal dt, Dy::AvbdDynamicsContext& /*taskGraphContext*/,
			bool& nextLayerReady,
			bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady)
		{
			nextLayerReady = false;
			nextWorldPlaneContactTaskReady = false;
			nextRigidBoxSdfContactTaskReady = false;
			nextRigidSphereSdfContactTaskReady = false;
			if(!mStandaloneComponentSolvePrepared ||
				mCausalLayerRangeObservations.empty())
				return false;
			const bool completedIndependentBodySweep =
				mStandaloneComponentStepState.
					completePublishedIndependentBodySweep(
						mCausalLayerRangeObservations.begin(),
						mCausalLayerRangeObservations.size());
			if(!completedIndependentBodySweep &&
				!mStandaloneComponentStepState.completePublishedCausalLayer(
					mCausalLayerRangeObservations.begin(),
					mCausalLayerRangeObservations.size()))
				return false;
			mStandaloneTaskGraphTelemetry.recordCausalLayerFanIn();
			const Dy::AvbdSoftBodyStepAdvanceResult result =
				usesStandaloneSceneRedetectionBridge()
					? advanceStandaloneComponentStateWithSceneRedetection(
						true, &nextWorldPlaneContactTaskReady,
						true, &nextRigidBoxSdfContactTaskReady,
						true, &nextRigidSphereSdfContactTaskReady)
					: mStandaloneComponentStepState.advance();
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

		bool AvbdCpuSoftScene::finishStandaloneCausalLayerSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& /*taskGraphContext*/)
		{
			PxU32 layerIndex = 0;
			PxU32 packedBegin = 0;
			PxU32 packedEnd = 0;
			const Dy::AvbdParticlePrimalSolveContext* solveContext = NULL;
			const Dy::AvbdSoftBody* bodies = NULL;
			PxU32 bodyCount = 0;
			const PxU32* particleBodyIndices = NULL;
			const PxU32* packedParticleIndices = NULL;
			if(mStandaloneComponentStepState.getPublishedCausalLayer(
				layerIndex, packedBegin, packedEnd, solveContext, bodies,
				bodyCount, particleBodyIndices, packedParticleIndices) &&
				packedEnd > packedBegin)
			{
				mStandaloneTaskGraphTelemetry.
					recordSerialCausalLayerFallback();
			}
			else if(mStandaloneComponentStepState.
				getPublishedIndependentBodySweep(
					solveContext, bodies, bodyCount))
			{
				mStandaloneTaskGraphTelemetry.
					recordSerialCausalLayerFallback();
			}
			if(usesStandaloneSceneRedetectionBridge())
			{
				if(!runStandaloneComponentStateWithSceneRedetection())
					return false;
			}
			else
				mStandaloneComponentStepState.runToCompletionSerial();
			if(!mStandaloneComponentStepState.isComplete())
				return false;
			finishStandaloneComponentSolve(dt);
			return true;
		}

		void AvbdCpuSoftScene::finishStandaloneComponentSolve(PxReal dt)
		{
			mStandaloneTaskGraphTelemetry.endSolveTask();
			mStandaloneStepStats.contactWorkspaceGrowthEvents +=
				mComponentFallbackPlan.initialContactWorkspaceGrowthEvents;
			mStandaloneStepStats.contactWorkspaceGrowthBytes +=
				mComponentFallbackPlan.initialContactWorkspaceGrowthBytes;
			mStandaloneStepStats.contactSweepScratchGrowthEvents +=
				mComponentFallbackPlan.initialContactSweepScratchGrowthEvents;
			mStandaloneStepStats.contactSweepScratchGrowthBytes +=
				mComponentFallbackPlan.initialContactSweepScratchGrowthBytes;
			mStandaloneStepStats.contactOutputGrowthEvents +=
				mComponentFallbackPlan.initialContactOutputGrowthEvents;
			mStandaloneStepStats.contactOutputGrowthBytes +=
				mComponentFallbackPlan.initialContactOutputGrowthBytes;
			mLastStepStats = mStandaloneStepStats;
			++mLastComponentFallbackSteps;
			mComponentFallbackPlanPrepared = false;
			mStandaloneComponentSolvePrepared = false;
			finalizeDeformableMotionControls(dt);
			mStandaloneComponentPostSolvePending = true;
		}

		bool AvbdCpuSoftScene::resumeStandaloneComponentSolve(
			PxReal dt, const PxVec3& gravity,
			bool* causalLayerTaskReady,
			bool* worldPlaneContactTaskReady,
			bool* rigidBoxSdfContactTaskReady,
			bool* rigidSphereSdfContactTaskReady)
		{
			PX_ASSERT(mStandaloneComponentSolvePrepared);
			if(!mStandaloneComponentSolvePrepared)
				return false;
			if(causalLayerTaskReady)
				*causalLayerTaskReady = false;
			if(worldPlaneContactTaskReady)
				*worldPlaneContactTaskReady = false;
			if(rigidBoxSdfContactTaskReady)
				*rigidBoxSdfContactTaskReady = false;
			if(rigidSphereSdfContactTaskReady)
				*rigidSphereSdfContactTaskReady = false;
			// All prediction children have joined before this continuation. A
			// defensive completeness check preserves the serial broadphase if a
			// future ownership path omits a body.
			mWorkspace.contact.markSoftBodyBoundsReady();
			// A relaxed color schedule is itself an explicit production fast-path
			// request.  It must enter the resumable task state even when the old
			// P4 validation switch is off; the state still falls back to scalar
			// authority if a complete conflict plan cannot be published.
			mStandaloneParticlePrimalSchedule =
				getParticlePrimalSchedule();
			const bool useCausalLayerTaskFanIn = causalLayerTaskReady &&
				(Dy::avbdUseCausalLayerTaskFanIn() ||
				 Dy::avbdUsesColoredParticlePrimalSchedule(
					mStandaloneParticlePrimalSchedule));
			const bool useIndependentBodySweepTaskFanIn =
				causalLayerTaskReady &&
				canUseIndependentBodySweepTaskFanIn();
			// P5 task leaves have not yet learned the cooked-collision embedding.
			// The relaxed component taskgraph remains safe for that public setup:
			// it keeps redetection on the existing synchronous, authoritative
			// callback while prediction/primal/write-back may still fan out.
			const bool useStaticWorldSelfOgcTaskFanIn =
				canUseStaticWorldSelfOgcContactTaskTransaction();
			const bool useSceneRedetectionBridge =
				usesStandaloneSceneRedetectionBridge();
			const bool useWorldPlaneContactTaskFanIn =
				causalLayerTaskReady && worldPlaneContactTaskReady &&
				useSceneRedetectionBridge &&
				Dy::avbdUseWorldPlaneContactTaskFanIn();
			const bool useRigidBoxSdfContactTaskFanIn =
				causalLayerTaskReady && rigidBoxSdfContactTaskReady &&
				useSceneRedetectionBridge &&
				Dy::avbdUseRigidBoxSdfContactTaskFanIn();
			const bool useRigidSphereSdfContactTaskFanIn =
				causalLayerTaskReady && rigidSphereSdfContactTaskReady &&
				useSceneRedetectionBridge &&
				(Dy::avbdUseRigidSphereSdfContactTaskFanIn() ||
					 Dy::avbdUseRigidCapsuleSdfContactTaskFanIn() ||
						 Dy::avbdUseRigidConvexSdfContactTaskFanIn() ||
						 Dy::avbdUseRigidTriangleSurfaceContactTaskFanIn() ||
						 Dy::avbdUseSelfBvhContactTaskFanIn() ||
					 useStaticWorldSelfOgcTaskFanIn);
			// Both ordered and relaxed colored paths use the persistent state so
			// Scene owns the inter-color barriers.  The ordered schedule remains a
			// reference oracle; relaxed colors intentionally need not reproduce
			// its per-particle traversal.
			const bool useColoredPrimalState =
				Dy::avbdUsesColoredParticlePrimalSchedule(
					mStandaloneParticlePrimalSchedule);
			if(Dy::avbdUsePersistentStepStateSerial() ||
				useCausalLayerTaskFanIn ||
				useIndependentBodySweepTaskFanIn ||
				useColoredPrimalState ||
				useSceneRedetectionBridge || useWorldPlaneContactTaskFanIn ||
				useRigidBoxSdfContactTaskFanIn ||
				useRigidSphereSdfContactTaskFanIn)
			{
				// The Scene-owned state spans all parent transitions. In the
				// P4.5.2c validation route it is consumed synchronously; P4.5.3
				// instead publishes precisely its first causal layer to Scene.
				const bool begun =
					mStandaloneComponentStepState.beginAfterPrediction(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size(),
						mContacts.begin(), mContacts.size(),
						dt, mComponentFallbackPlan.outerIterations,
						mComponentFallbackPlan.innerIterations,
						mComponentFallbackPlan.totalPositionIterations,
						1000.0f, redetectContacts, &mContacts, this,
						0.92f, &mStandaloneStepStats, mWorkspace,
					mSelfCollisionAdjacencies.begin(),
					mSelfCollisionAdjacencies.size(),
					mSelfCollisionEnabled.begin(), &mContactParams,
					mStandaloneParticlePrimalSchedule,
					useSceneRedetectionBridge,
					useIndependentBodySweepTaskFanIn);
				PX_ASSERT(begun);
				if(!begun)
					return false;
				if(useCausalLayerTaskFanIn ||
					useIndependentBodySweepTaskFanIn ||
					useWorldPlaneContactTaskFanIn ||
					useRigidBoxSdfContactTaskFanIn ||
					useRigidSphereSdfContactTaskFanIn)
				{
					const Dy::AvbdSoftBodyStepAdvanceResult result =
						useSceneRedetectionBridge
							? advanceStandaloneComponentStateWithSceneRedetection(
								useWorldPlaneContactTaskFanIn,
								worldPlaneContactTaskReady,
								useRigidBoxSdfContactTaskFanIn,
								rigidBoxSdfContactTaskReady,
								useRigidSphereSdfContactTaskFanIn,
								rigidSphereSdfContactTaskReady)
							: mStandaloneComponentStepState.advance();
					if(useWorldPlaneContactTaskFanIn &&
						*worldPlaneContactTaskReady)
						return false;
					if(useRigidBoxSdfContactTaskFanIn &&
						*rigidBoxSdfContactTaskReady)
						return false;
					if(useRigidSphereSdfContactTaskFanIn &&
						*rigidSphereSdfContactTaskReady)
						return false;
					if(result ==
						Dy::AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
					{
						// Initial redetection has now published the authoritative
						// contact epoch. Only at this point can a dense soft/soft
						// manifold be identified reliably; keep its relaxed color
						// plan but finish the short layers inline instead of creating
						// a task/fan-in pair for every color.
						if(shouldInlineDenseSoftPairColoredPrimal())
						{
							mStandaloneComponentStepState.
								runToCompletionSerial();
							if(!mStandaloneComponentStepState.isComplete())
								return false;
						}
						else
						{
							*causalLayerTaskReady = true;
							return false;
						}
					}
					else if(result !=
						Dy::AvbdSoftBodyStepAdvanceResult::eCOMPLETE)
						return false;
				}
				else if(useSceneRedetectionBridge)
				{
					if(!runStandaloneComponentStateWithSceneRedetection())
						return false;
				}
				else
					mStandaloneComponentStepState.runToCompletionSerial();
				PX_ASSERT(mStandaloneComponentStepState.isComplete());
				if(!mStandaloneComponentStepState.isComplete())
					return false;
			}
			else
			{
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
					Dy::AvbdSoftBodyStepExecutionMode::eRESUME);
			}
			finishStandaloneComponentSolve(dt);
			return true;
		}

		// Serial authority for the split component route. This preserves the
		// exact same ePREPARE -> prediction -> eRESUME ordering as the task
		// graph, while running all three stages on the caller thread.
		bool AvbdCpuSoftScene::stepStandaloneComponentSolveOnly(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
			if(!prepareStandaloneComponentSolve(
				dt, gravity, materialManager, rigidMaterialManager,
				sleepingEnabled))
				return false;
			predictStandaloneComponentRange(
				0, mEntries.size(), dt, gravity);
			return resumeStandaloneComponentSolve(dt, gravity);
		}

		// Returns the number of fixed entry ranges that can safely be submitted
		// after stepStandaloneComponentSolveOnly().  P3 intentionally avoids
		// per-particle partitioning: independent entries are the first proven
		// no-conflict boundary, and a small component remains serial.
		PxU32 AvbdCpuSoftScene::getStandaloneWriteBackTaskCount(
			PxU32 dispatcherWorkers) const
		{
			if(!mStandaloneComponentPostSolvePending ||
				mDynamicsOwnsStep || dispatcherWorkers < 2)
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
			// The task and fan-in overhead is not useful for small copies.  Keep
			// this threshold independent from the P2 solve eligibility: it only
			// describes output bandwidth and leaves the solve semantics untouched.
			if(awakeEntryCount < 2 || awakeParticleCount < 1024)
				return 0;
			return PxMin(dispatcherWorkers, awakeEntryCount);
		}

		void AvbdCpuSoftScene::step(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled)
		{
			mStandaloneComponentPostSolvePending = false;
			// These values are deliberately step-local.  The public scene
			// statistics report a completed simulation step, while the snippet
			// accumulates only its profile window and can therefore assert that
			// warm-up reached a zero-growth steady state.
			mLastStepStats.reset();
			if(mCollisionStatsEnabled)
				mLastCollisionStats = Dy::AvbdSoftCollisionStats();
			mLastComponentFallbackSteps = 0;
			mLastNativeIslandSteps = 0;
			if(dt <= 0.0f)
				return;

			if(mBodies.empty())
				return;

			PxU32 awakeEntryCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(!mEntries[i].sleeping)
					awakeEntryCount++;
			if(awakeEntryCount == 0 && sleepingEnabled)
			{
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return;
			}

			if(mDynamicsOwnsStep)
			{
				mLastNativeIslandSteps = 1;
				// Selection is all-or-nothing at the preparation boundary.  Do
				// not turn an unexpected wake/sleep transition between prep and
				// post-solve into a full component fallback: that would advance
				// the already-native particles a second time.  The changed
				// membership is safely reconsidered on the next prepare boundary.
				PX_ASSERT(mDynamicsSelectedEntryCount == awakeEntryCount);
				for(PxU32 storageIndex = 0;
					storageIndex < mIslandSelectionStorages.size();
					storageIndex++)
				{
					IslandSelectionStorage& storage =
						*mIslandSelectionStorages[storageIndex];
					if(storage.touched)
						copyIslandSelectionResults(storage);
				}
				finalizeDeformableMotionControls(dt);
				for(PxU32 i = 0; i < mEntries.size(); i++)
					writeBack(mEntries[i]);
				updateSleepStates(dt, sleepingEnabled);
				mWorkspacePreflightPending = false;
				mDynamicsOwnsStep = false;
				mDynamicsSelectedEntryCount = 0;
				return;
			}

			// The component route is now the sole owner for this frame.  A native
			// selection contact stream is a separate persistent AL/friction cache;
			// retaining it across this fallback would revive a state that did not
			// participate in the immediately preceding solve.
			invalidateNativeIslandSelectionCaches();
			stepComponentFallback(
				dt, gravity, materialManager,
				rigidMaterialManager);

			finalizeDeformableMotionControls(dt);
			for(PxU32 i = 0; i < mEntries.size(); i++)
				writeBack(mEntries[i]);
			updateSleepStates(dt, sleepingEnabled);
			mWorkspacePreflightPending = false;
			mDynamicsSelectedEntryCount = 0;
		}

} // namespace Sc
} // namespace physx
