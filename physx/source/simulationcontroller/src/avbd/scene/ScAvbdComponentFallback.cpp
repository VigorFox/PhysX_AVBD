// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		PxU32 AvbdCpuSoftScene::estimateInitialComponentContactCapacity() const
		{
			// Rigid primitives and soft-body peers have a source-aware linear
			// initial reserve. Self collision is deliberately excluded: its
			// density has no useful linear topology bound and must remain
			// telemetry-visible until P1 has a separately accepted policy.
			const PxU64 sourceCount =
				PxU64(mWorldPlanes.size()) +
				PxU64(mRigidBoxes.size()) +
				PxU64(mRigidSpheres.size()) +
				PxU64(mRigidCapsules.size()) +
				PxU64(mRigidConvexes.size()) +
				PxU64(mRigidTriangleSurfaces.size()) +
				(mBodies.size() > 1 ? PxU64(mBodies.size() - 1) : 0);
			if(sourceCount == 0 || mParticles.empty())
				return 0;

			// The measured ground and two-surface corpus peaks at four contact
			// slots per particle. Keep a hard 2 MiB budget for all associated
			// output/state/incidence storage; an underestimated or dynamic case
			// falls back to normal, visible growth rather than over-reserving.
			const PxU64 bytesPerContact =
				PxU64(sizeof(Dy::AvbdSoftContact)) * 2 +
				PxU64(sizeof(Dy::AvbdSoftContactParticleRef)) *
					Dy::AVBD_CONTACT_MAX_PARTICLES +
				PxU64(sizeof(Dy::AvbdCompiledSoftVelocityObjective)) +
				sizeof(PxU8);
			const PxU64 capacityBudget =
				(2ull * 1024ull * 1024ull) / bytesPerContact;
			const PxU64 requestedCapacity =
				PxU64(mParticles.size()) * 4 * sourceCount;
			return PxU32(PxMin(requestedCapacity, capacityBudget));
		}

		void AvbdCpuSoftScene::reserveLifecycleContactCapacity()
		{
			const PxU32 contactCapacity =
				estimateInitialComponentContactCapacity();
			if(contactCapacity)
				mContacts.reserve(contactCapacity);
			mWorkspace.reserve(
				mParticles.size(), contactCapacity,
				getParticlePrimalSchedule());
		}

		void AvbdCpuSoftScene::reserveLifecycleCollisionScratch()
		{
			// One contact workspace is shared serially by all component bodies.
			// Reserve the largest self-query topology and the two largest soft-pair
			// edge sets at actor-add time. Refitting still writes current particle
			// bounds every detection; this only removes capacity allocation from
			// that hot path.
			PxU32 maxTetCount = 0;
			PxU32 maxTriangleCount = 0;
			PxU32 maxSurfaceVertexCount = 0;
			PxU32 maxEdgeCountA = 0;
			PxU32 maxEdgeCountB = 0;
			for(PxU32 bodyIndex = 0;
				bodyIndex < mBodies.size(); bodyIndex++)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					mBodies[bodyIndex].compiled;
				maxTetCount = PxMax(
					maxTetCount, compiled.tetElements.size());
				maxTriangleCount = PxMax(
					maxTriangleCount,
					compiled.surfaceTriangles.size() / 3);
				maxSurfaceVertexCount = PxMax(
					maxSurfaceVertexCount,
					compiled.surfaceVertices.size());
				const PxU32 edgeCount = compiled.surfaceEdges.size();
				if(edgeCount > maxEdgeCountA)
				{
					maxEdgeCountB = maxEdgeCountA;
					maxEdgeCountA = edgeCount;
				}
				else if(edgeCount > maxEdgeCountB)
					maxEdgeCountB = edgeCount;
			}
			mWorkspace.contact.reserveSelfCollisionSweep(
				maxTetCount, maxTriangleCount,
				maxSurfaceVertexCount, maxEdgeCountA);
			mWorkspace.contact.reserveSoftPairSweep(
				maxEdgeCountA, maxEdgeCountB,
				maxTriangleCount);
		}

		void AvbdCpuSoftScene::prepareComponentFallback(
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager)
		{
			PX_ASSERT(!mComponentFallbackPlanPrepared);
			mComponentFallbackPlan = ComponentFallbackPlan();
			ComponentFallbackPlan& plan = mComponentFallbackPlan;
			PxU32 requestedPositionIterations = 1;
			PxU32 requestedCollisionPairUpdates = 0;
			PxU32 requestedCollisionSubsteps = 1;
			bool hasExplicitCollisionPairUpdates = false;
			bool allEntriesAreVolumes = !mEntries.empty();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				syncHostInputs(mEntries[i], materialManager);
				allEntriesAreVolumes = allEntriesAreVolumes &&
					mEntries[i].kind == eVOLUME;
				requestedPositionIterations = PxMax<PxU32>(
					requestedPositionIterations,
					mEntries[i].getSolverIterationCounts() & 0xff);
				if(mEntries[i].kind == eSURFACE)
				{
					const PxU32 pairUpdates =
						mEntries[i].surfaceCore->
							getNbCollisionPairUpdatesPerTimestep();
					if(pairUpdates > 0)
					{
						hasExplicitCollisionPairUpdates = true;
						requestedCollisionPairUpdates =
							PxMax(
								requestedCollisionPairUpdates,
								pairUpdates);
					}
					requestedCollisionSubsteps = PxMax(
						requestedCollisionSubsteps,
						PxMax<PxU32>(
							mEntries[i].surfaceCore->
								getNbCollisionSubsteps(),
							1u));
				}
			}

			compileWorldStatics(rigidMaterialManager);
			if(mWorkspacePreflightPending)
			{
				// This is a one-time scene lifecycle preflight, before the
				// first component solve. Later soft actor adds use the same
				// helper at their actor-add boundary; other dynamic mutation
				// paths retain normal, telemetry-visible growth.
				reserveLifecycleContactCapacity();
			}
			// avbdStepSoftBodies() resets its own counters before solver work.
			// Reset and snapshot the initial OGC pass separately so capacity
			// growth in that pass is not lost from the scene-level telemetry.
			mWorkspace.contact.beginStep();
			detectContacts(
				mParticles.begin(), mParticles.size(),
				mBodies.begin(), mBodies.size(), mContacts);
			plan.initialContactWorkspaceGrowthEvents =
				mWorkspace.contact.growthEvents;
			plan.initialContactWorkspaceGrowthBytes =
				mWorkspace.contact.growthBytes;
			plan.initialContactSweepScratchGrowthEvents =
				mWorkspace.contact.sweepScratchGrowthEvents;
			plan.initialContactSweepScratchGrowthBytes =
				mWorkspace.contact.sweepScratchGrowthBytes;
			plan.initialContactOutputGrowthEvents =
				mWorkspace.contact.outputGrowthEvents;
			plan.initialContactOutputGrowthBytes =
				mWorkspace.contact.outputGrowthBytes;
			const bool needsContactRedetection =
				!mContacts.empty() || mBodies.size() > 1 ||
				!mRigidBoxes.empty() || !mRigidSpheres.empty() ||
				!mRigidCapsules.empty() ||
				!mRigidConvexes.empty() ||
				!mRigidTriangleSurfaces.empty() ||
				!mWorldPlanes.empty();
			// OGC may retain one same-time manifold across several material
			// sweeps only when every row has a conservative component-owned
			// safety bound. Dynamic rigid targets use the native shared
			// OgcPairState path instead, so do not pack this fallback schedule
			// merely because a scene happens to contain a contact.
			const bool canPackComponentOgcEpochs =
				allEntriesAreVolumes && !mContacts.empty() &&
				Dy::avbdCanReuseComponentOgcEpoch(
					mContacts.begin(), mContacts.size(),
					mBodies.begin(), mBodies.size(),
					mParticles.begin());
			// PxDeformableBody::setSolverIterationCounts() specifies the
			// minimum position-iteration budget for the complete timestep.
			// A non-zero Surface pair-update count explicitly selects the
			// number of OGC redetection stages; zero retains adaptive
			// redetection. Collision substeps set the minimum number of
			// contact-bearing sweeps in each stage without changing dt or
			// integrating elasticity more than the resulting solver budget.
			const PxU32 requestedRedetectionStages =
				needsContactRedetection
					? (hasExplicitCollisionPairUpdates
						? requestedCollisionPairUpdates
						: 8u)
					: 1u;
			const PxU32 minimumContactIterations =
				needsContactRedetection
					? PxMax<PxU32>(
						8u,
						requestedRedetectionStages *
							requestedCollisionSubsteps)
					: 1u;
			plan.totalPositionIterations = PxMax<PxU32>(
				requestedPositionIterations,
				minimumContactIterations);
			plan.outerIterations =
				needsContactRedetection
					? PxMin<PxU32>(
						requestedRedetectionStages,
						plan.totalPositionIterations)
					: 1u;
			plan.innerIterations =
				(plan.totalPositionIterations + plan.outerIterations - 1) /
				plan.outerIterations;
			// Retain the position-iteration budget, but batch the safe portion
			// into three OGC epochs.  If a candidate reaches its pair bound the
			// solver exits the epoch immediately and rebuilds the current-pose
			// manifold, so this is neither a CCD path nor a hidden substep.
			if(canPackComponentOgcEpochs &&
				needsContactRedetection &&
				!hasExplicitCollisionPairUpdates &&
				requestedCollisionSubsteps == 1u &&
				plan.totalPositionIterations >= 8u)
			{
				plan.outerIterations = PxMin<PxU32>(
					3u, plan.totalPositionIterations);
				plan.innerIterations =
					(plan.totalPositionIterations + plan.outerIterations - 1) /
					plan.outerIterations;
			}
			// This diagnostic route intentionally remains explicit: unlike the
			// safety-admitted scheduler above, it may exercise a cadence even when
			// the contact stream is not eligible for manifold reuse.
			else if(useAvbdVolumeTest3x3Cadence() && allEntriesAreVolumes &&
				needsContactRedetection &&
				!hasExplicitCollisionPairUpdates &&
				requestedRedetectionStages == 8u &&
				requestedCollisionSubsteps == 1u &&
				plan.totalPositionIterations == 8u &&
				plan.outerIterations == 8u)
			{
				plan.outerIterations = 3u;
				plan.innerIterations = 3u;
			}
			mComponentFallbackPlanPrepared = true;
		}

		void AvbdCpuSoftScene::resumeComponentFallback(
			PxReal dt, const PxVec3& gravity)
		{
			PX_ASSERT(mComponentFallbackPlanPrepared);
			if(!mComponentFallbackPlanPrepared)
				return;
			const ComponentFallbackPlan& plan = mComponentFallbackPlan;
			// This is the synchronous fallback authority.  Do not inherit a stale
			// worker count from a prior asynchronous frame merely to choose a
			// different nonlinear trajectory; only an explicit process override
			// may request colored execution here.
			const Dy::AvbdParticlePrimalSchedule configuredSchedule =
				mStandaloneTaskGraphEnhancedDeterminism
					? Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR
					: Dy::avbdGetConfiguredParticlePrimalSchedule();
			const Dy::AvbdParticlePrimalSchedule particlePrimalSchedule =
				configuredSchedule ==
					Dy::AvbdParticlePrimalSchedule::eDEFAULT
					? Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR
					: configuredSchedule;

			const auto executeSoftBodyStep =
				[&](Dy::AvbdSoftBodyStepExecutionMode executionMode)
			{
					Dy::avbdStepSoftBodies(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size(),
						mContacts.begin(), mContacts.size(),
						dt, gravity,
						plan.outerIterations, plan.innerIterations,
						1000.0f,
						redetectContacts, &mContacts, this,
						0.92f, &mLastStepStats, &mWorkspace,
						plan.totalPositionIterations,
						mSelfCollisionAdjacencies.begin(),
						mSelfCollisionAdjacencies.size(),
						mSelfCollisionEnabled.begin(),
						&mContactParams, executionMode,
						particlePrimalSchedule);
				};
			if(mP3ForceSplitPrediction)
			{
				executeSoftBodyStep(
					Dy::AvbdSoftBodyStepExecutionMode::ePREPARE);
				const bool useRigidInitialGuess =
					Dy::avbdCanUseSoftRigidPrimalInitialization(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size());
				const bool useAdaptiveInitialGuess = !useRigidInitialGuess &&
					Dy::avbdCanUseSoftAdaptivePrimalInitialization(
						mParticles.begin(), mParticles.size(),
						mBodies.begin(), mBodies.size());
				Dy::avbdPredictSoftBodyParticles(
					mParticles.begin(), mParticles.size(), dt, gravity,
					useAdaptiveInitialGuess);
				if(useRigidInitialGuess)
				{
					for(PxU32 bodyIndex = 0;
						bodyIndex < mBodies.size(); bodyIndex++)
						Dy::avbdApplySoftBodyRigidPrimalInitialGuess(
							mParticles.begin(), mParticles.size(),
							mBodies[bodyIndex]);
				}
				executeSoftBodyStep(
					Dy::AvbdSoftBodyStepExecutionMode::eRESUME);
			}
			else
			{
				executeSoftBodyStep(
					Dy::AvbdSoftBodyStepExecutionMode::eFULL);
			}
			mLastStepStats.contactWorkspaceGrowthEvents +=
				plan.initialContactWorkspaceGrowthEvents;
			mLastStepStats.contactWorkspaceGrowthBytes +=
				plan.initialContactWorkspaceGrowthBytes;
			mLastStepStats.contactSweepScratchGrowthEvents +=
				plan.initialContactSweepScratchGrowthEvents;
			mLastStepStats.contactSweepScratchGrowthBytes +=
				plan.initialContactSweepScratchGrowthBytes;
			mLastStepStats.contactOutputGrowthEvents +=
				plan.initialContactOutputGrowthEvents;
			mLastStepStats.contactOutputGrowthBytes +=
				plan.initialContactOutputGrowthBytes;
			++mLastComponentFallbackSteps;
			mComponentFallbackPlanPrepared = false;
		}

		void AvbdCpuSoftScene::stepComponentFallback(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager)
		{
			prepareComponentFallback(materialManager, rigidMaterialManager);
			resumeComponentFallback(dt, gravity);
		}

} // namespace Sc
} // namespace physx
