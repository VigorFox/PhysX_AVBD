// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

// Emit the authoritative scalar component step once. Scene task-graph,
// diagnostic and snippet translation units see only its declaration.
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

void avbdStepSoftBodies(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, const PxVec3& gravity,
	PxU32 outerIterations, PxU32 innerIterations,
	PxReal avbdBeta,
	AvbdContactRedetectFn redetectFn,
	PxArray<AvbdSoftContact>* contactsArray,
	void* redetectUserData,
	PxReal chebyshevRho,
	AvbdSoftBodyStepStats* stepStats,
	AvbdSoftBodyWorkspace* persistentWorkspace,
	PxU32 totalInnerIterationBudget,
	const AvbdSelfCollisionAdjacency* selfCollisionAdjacencies,
	PxU32 numSelfCollisionAdjacencies,
	const PxU8* selfCollisionEnabled,
	const AvbdOGCParams* ogcParams,
	AvbdSoftBodyStepExecutionMode executionMode,
	AvbdParticlePrimalSchedule inputParticlePrimalSchedule)
{
	if (numParticles == 0 || numSoftBodies == 0) return;
	if(executionMode != AvbdSoftBodyStepExecutionMode::eFULL &&
		!persistentWorkspace)
	{
		PX_ASSERT(false);
		return;
	}
	// A total budget lets callers retain the outer contact-redetection
	// schedule without rounding every stage up to a full inner batch.
	const PxU32 requestedInnerIterationBudget =
		totalInnerIterationBudget > 0
			? PxMax(totalInnerIterationBudget, outerIterations)
			: outerIterations * innerIterations;
	AvbdSoftBodyWorkspace localWorkspace;
	AvbdSoftBodyWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	PxArray<AvbdCompiledSoftVelocityObjective>&
		compiledVelocityObjectives =
			workspace.compiledVelocityObjectives;
	PxArray<AvbdSoftComponentFinalizeMode>&
		componentFinalizeModes = workspace.componentFinalizeModes;
	if(executionMode != AvbdSoftBodyStepExecutionMode::eRESUME)
	{
		for(PxU32 contactIdx = 0; contactIdx < numContacts; contactIdx++)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[contactIdx].geometry;
			const AvbdSoftContactTargetKind targetKind =
				geometry.targetKind;
			if(targetKind !=
					AvbdSoftContactTargetKind::eWORLD_STATIC &&
				targetKind !=
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				targetKind !=
					AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE)
			{
				// This Scene-external component has no rigid 6x6 block. Accepting
				// a rigid target here would silently turn a two-sided objective
				// into a one-way particle correction.
				PX_ASSERT(false);
				return;
			}
			const bool positionOwned =
				(targetKind ==
						AvbdSoftContactTargetKind::eWORLD_STATIC ||
				 targetKind ==
						AvbdSoftContactTargetKind::
							eDEFORMABLE_SURFACE) &&
				geometry.velocityOwner ==
					AvbdVelocityObjectiveOwner::PositionAL;
			const bool componentOwned =
				targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				geometry.velocityOwner ==
					AvbdVelocityObjectiveOwner::
						ComponentFinalize;
			if(!positionOwned && !componentOwned)
			{
				// Prep must assign exactly one compatible owner.  No solve stage
				// is allowed to reinterpret target kind or flags later.
				PX_ASSERT(false);
				return;
			}
		}
		for (PxU32 si = 0; si < numSoftBodies; si++)
		{
			PX_ASSERT(
				softBodies[si].runtime.isObjectiveProgramCurrent(
					softBodies[si].compiled.particleStart,
					softBodies[si].compiled.particleCount));
		}
		if(stepStats)
		{
			stepStats->reset();
			stepStats->requestedOuterIterations = outerIterations;
			stepStats->requestedInnerIterations =
				requestedInnerIterationBudget;
		}
		workspace.beginStep();
		workspace.contact.prepareSoftBodyBounds(numSoftBodies);
		compiledVelocityObjectives.clear();
		workspace.resize(componentFinalizeModes, numSoftBodies);
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
		{
			componentFinalizeModes[bodyIndex] =
				softBodies[bodyIndex].runtime.compiledObjectives.empty()
					? AvbdSoftComponentFinalizeMode::eMOMENTUM
					: AvbdSoftComponentFinalizeMode::ePOSITION_OWNED;
			const PxU32 particleStart =
				softBodies[bodyIndex].compiled.particleStart;
			const PxU32 particleCount =
				softBodies[bodyIndex].compiled.particleCount;
			if(particleStart > numParticles ||
				particleCount > numParticles - particleStart)
			{
				componentFinalizeModes[bodyIndex] =
					AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
				continue;
			}
			for(PxU32 localIndex = 0;
				localIndex < particleCount; localIndex++)
			{
				if(particles[particleStart + localIndex].invMass <= 0.0f)
				{
					componentFinalizeModes[bodyIndex] =
						AvbdSoftComponentFinalizeMode::ePOSITION_OWNED;
					break;
				}
			}
		}
	}
	PxTime stageTimer;
	if(executionMode != AvbdSoftBodyStepExecutionMode::eRESUME)
	{
		avbdCompileSoftVelocityObjectives(
			compiledVelocityObjectives, componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
		// A persistent contact carries AL/friction state across frames, but its
		// finite depenetration bias is a one-frame target.
		avbdResetSoftContactDepenetrationLimits(
			contacts, numContacts);
	}
	if(executionMode == AvbdSoftBodyStepExecutionMode::ePREPARE)
		return;

	PxReal invDt = dt > 0.0f ? 1.0f / dt : 0.0f;
	PxReal invDtSq = invDt * invDt;

	// Stage 1: prediction. eRESUME is entered only after an owner has written
	// these disjoint particle fields through the P3 continuation boundary.
	if(executionMode == AvbdSoftBodyStepExecutionMode::eFULL)
	{
		const bool useRigidInitialGuess =
			avbdCanUseSoftRigidPrimalInitialization(
				particles, numParticles, softBodies, numSoftBodies);
		const bool useAdaptiveInitialGuess = !useRigidInitialGuess &&
			avbdCanUseSoftAdaptivePrimalInitialization(
				particles, numParticles, softBodies, numSoftBodies);
		avbdPredictSoftBodyParticles(
			particles, numParticles, dt, gravity, useAdaptiveInitialGuess);
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		{
			if(useRigidInitialGuess)
				avbdApplySoftBodyRigidPrimalInitialGuess(
					particles, numParticles, softBodies[bodyIndex]);
			avbdComputeSoftBodyBounds(
				particles, softBodies[bodyIndex],
				workspace.contact.softBodyBounds[bodyIndex]);
			workspace.contact.softBodyBoundsReady[bodyIndex] = 1;
		}
		workspace.contact.markSoftBodyBoundsReady();
	}
	avbdPublishTetMaterialPacketIrStats(
		softBodies, numSoftBodies, stepStats);
	// Contact prep before prediction cannot see a first-impact candidate.
	// Refresh once after predictedPosition is current so speculative plane
	// and swept rigid-SDF contacts can constrain the same timestep instead of
	// recovering from an already intersecting state on the next frame.
	if(redetectFn && contactsArray)
	{
		redetectFn(
			particles, numParticles,
			softBodies, numSoftBodies,
			*contactsArray, redetectUserData);
		contacts = contactsArray->begin();
		numContacts = contactsArray->size();
		avbdCompileSoftVelocityObjectives(
			compiledVelocityObjectives, componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
	}
	avbdInitializeSoftContactDepenetrationLimits(
		contacts, numContacts, particles,
		softBodies, numSoftBodies, dt);
	PxArray<AvbdSoftComponentMomentumTarget>&
		componentMomentumTargets =
			workspace.componentMomentumTargets;
	workspace.resize(componentMomentumTargets, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		AvbdSoftComponentMomentumTarget& target =
			componentMomentumTargets[bodyIndex];
		target = AvbdSoftComponentMomentumTarget();
		if(componentFinalizeModes[bodyIndex] ==
				AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		PxVec3 centroid(0.0f);
		PxMat33 inertia(PxZero);
		target.valid = avbdComputeSoftComponentMomentum(
			particles, numParticles, softBodies[bodyIndex],
			true, invDt, centroid, target.linearMomentum,
			target.angularMomentum, inertia, target.mass);
		target.centroid = centroid;
		avbdApplySoftComponentDampingToMomentumTarget(
			target, softBodies[bodyIndex], dt);
		PX_UNUSED(inertia);
	}
	if(stepStats)
		stepStats->predictionMs += stageTimer.getElapsedSeconds() * 1000.0;

	// P4 policy is selected once at the step boundary.  Contact-index rebuilds
	// are part of every OGC epoch and must not perform runtime configuration
	// discovery themselves.
	const AvbdParticlePrimalSchedule particlePrimalSchedule =
		inputParticlePrimalSchedule ==
			AvbdParticlePrimalSchedule::eDEFAULT
			? avbdGetParticlePrimalSchedule() : inputParticlePrimalSchedule;
	const bool validateParticlePrimalAccessPlan =
		avbdValidateParticlePrimalAccessPlan();

	// Build per-particle contact index to avoid O(particles*contacts) scan.
	// contactStart[pi] = first index into contactIdx for particle pi.
	// contactIdx stores contact indices grouped by particle.
	avbdBuildSoftParticleContactIndex(
		workspace, softBodies, numSoftBodies,
		contacts, numContacts, numParticles, stepStats,
		particlePrimalSchedule, validateParticlePrimalAccessPlan, particles);
	PxArray<AvbdSoftContactParticleRef>& contactIdxBuf =
		workspace.contactIndices;
	const PxArray<PxU32>& contactStart = workspace.contactStarts;

	if(stepStats)
		stepStats->contactIndexMs +=
			stageTimer.getElapsedSeconds() * 1000.0;

	// Chebyshev semi-iterative acceleration state.
	// If chebyshevRho > 0, we use adaptive spectral-radius estimation:
	// measure the actual GS convergence rate from inner iterations 0-1,
	// then use min(measured, user-provided) as the Chebyshev parameter.
	// This prevents over-relaxation on meshes whose spectral radius
	// differs from the user's estimate (e.g., non-uniform voxel meshes).
	const bool useChebyshev = (chebyshevRho > 0.0f && chebyshevRho < 1.0f);
	PxReal chebyOmega = 1.0f;
	PxReal adaptiveRho = chebyshevRho;
	PxArray<PxVec3>& chebyPrevPos = workspace.chebyPrevPos;
	PxArray<PxVec3>& chebyPrevPrevPos = workspace.chebyPrevPrevPos;
	PxArray<PxReal>& selfCollisionSafetyBounds =
		workspace.selfCollisionSafetyBounds;
	PxArray<PxReal>& bodySelfCollisionSafetyBounds =
		workspace.bodySelfCollisionSafetyBounds;
	workspace.resize(selfCollisionSafetyBounds, numParticles);
	if (useChebyshev)
	{
		workspace.resize(chebyPrevPos, numParticles);
		workspace.resize(chebyPrevPrevPos, numParticles);
		for (PxU32 i = 0; i < numParticles; i++)
		{
			chebyPrevPos[i] = particles[i].position;
			chebyPrevPrevPos[i] = particles[i].position;
		}
	}
	// Main iteration loop
	PxU32 remainingInnerIterationBudget =
		requestedInnerIterationBudget;
	bool reuseComponentOgcSafetyEpoch = false;
	bool componentOgcSafetyEpochActive = false;
	for (PxU32 outerIt = 0; outerIt < outerIterations; outerIt++)
	{
		if(stepStats)
			stepStats->executedOuterIterations++;
		const PxU64 particleSweepsBeforeOuter = stepStats
			? stepStats->particleSweeps : 0;
		const PxU32 remainingOuterIterations =
			outerIterations - outerIt;
		const PxU32 currentInnerIterations =
			(remainingInnerIterationBudget +
				remainingOuterIterations - 1) /
			remainingOuterIterations;
		remainingInnerIterationBudget -= currentInnerIterations;
		if(!reuseComponentOgcSafetyEpoch)
		{
			// Snapshot positions as proximal anchor for the AVBD elastic term.
			avbdSnapshotOuterPositionsScalar(
				particles, numParticles, selfCollisionSafetyBounds.begin());

			// OGC Eq. 21-27: each fresh DCD epoch records a known
			// penetration-free anchor and a conservative displacement radius.
			const AvbdOGCParams defaultOgcParams;
			const AvbdOGCParams& activeOgcParams =
				ogcParams ? *ogcParams : defaultOgcParams;
			if(selfCollisionAdjacencies)
			{
				for(PxU32 bodyIndex = 0;
					bodyIndex < numSoftBodies &&
					bodyIndex < numSelfCollisionAdjacencies;
					bodyIndex++)
				{
					if(selfCollisionEnabled &&
						!selfCollisionEnabled[bodyIndex])
						continue;
					const AvbdSoftBody& body =
						softBodies[bodyIndex];
					avbdComputeSafetyBounds(
						body, particles,
						selfCollisionAdjacencies[bodyIndex],
						activeOgcParams.contactRadius,
						activeOgcParams.safetyRelax,
						bodySelfCollisionSafetyBounds,
						workspace.contact);
					for(PxU32 localIndex = 0;
						localIndex < body.compiled.particleCount;
						localIndex++)
					{
						const PxU32 particleIndex =
							body.compiled.particleStart +
							localIndex;
						if(particleIndex < numParticles)
							selfCollisionSafetyBounds[
								particleIndex] =
								bodySelfCollisionSafetyBounds[
									localIndex];
					}
				}
			}
			componentOgcSafetyEpochActive =
				avbdApplyComponentOgcEpochSafetyBounds(
					contacts, numContacts, softBodies, numSoftBodies, particles,
					activeOgcParams.contactRadius,
					activeOgcParams.safetyRelax,
					selfCollisionSafetyBounds.begin(), numParticles,
					workspace);
		}
		reuseComponentOgcSafetyEpoch = false;
		bool componentOgcSafetyEpochLimited = false;

		// Reset Chebyshev state each outer iteration: the system changes
		// (contacts re-detected, elasticK updated) so prior omega/positions
		// are invalid.
		if (useChebyshev)
		{
			chebyOmega = 1.0f;
			for (PxU32 i = 0; i < numParticles; i++)
			{
				chebyPrevPos[i] = particles[i].position;
				chebyPrevPrevPos[i] = particles[i].position;
			}
		}

		PxReal prevMaxDxSq = 0.0f;
		PxU32 shadowResidual1e5ConsecutiveSweeps = 0;
		PxU32 shadowResidual1e4ConsecutiveSweeps = 0;
		bool shadowResidual1e5Recorded = false;
		bool shadowResidual1e4Recorded = false;
		bool legacyAppliedConvergenceRecorded = false;
		AvbdSoftResidualConvergenceTracker residualConvergence(
			1e-8f, 2);

		for (PxU32 innerIt = 0;
			innerIt < currentInnerIterations; innerIt++)
		{
			if(stepStats)
			{
				stepStats->executedInnerIterations++;
				stepStats->particleSweeps++;
			}
			PxReal maxDxSq = 0.0f;
			AvbdParticlePrimalRangeObservation particlePrimalObservation;
			AvbdSoftSweepConvergenceObservation& sweepObservation =
				particlePrimalObservation.sweepObservation;

			// Canonical scalar reference traversal.  Causal-layer scheduling is
			// owned by the Scene continuation boundary; it must not remain as a
			// default-off branch in this scalar primal kernel.
			const AvbdParticlePrimalSolveContext particlePrimalSolveContext =
			{
				particles,
				contacts,
				contactStart.begin(),
				contactIdxBuf.begin(),
				selfCollisionSafetyBounds.begin(),
				invDt,
				invDtSq,
				avbdSelectTetMaterialPacketKernels(
					softBodies, numSoftBodies)
			};

			if(particlePrimalSolveContext.tetMaterialPacketKernels.hasAny())
				avbdSolveParticlePrimalTetMaterialPacketBodyRange(
					particlePrimalSolveContext, softBodies, numSoftBodies,
					particlePrimalObservation);
			else
			{
				for(PxU32 bodyIndex = 0;
					bodyIndex < numSoftBodies; bodyIndex++)
				{
					const AvbdSoftBody& body = softBodies[bodyIndex];
					for(PxU32 localIndex = 0;
						localIndex < body.compiled.particleCount; localIndex++)
						particlePrimalSolveContext.solve(
							body, localIndex, particlePrimalObservation);
				}
			}
			maxDxSq =
				sweepObservation.maxAppliedDisplacementSq;
			if(stepStats)
			{
				stepStats->tetLinearizationCacheFallbackParticleSteps +=
					particlePrimalObservation.
						tetLinearizationCacheFallbackParticleSteps;
				stepStats->trustRegionLimitedParticleSteps +=
					sweepObservation.trustRegionLimitedSteps;
				stepStats->positiveJLimitedParticleSteps +=
					sweepObservation.positiveJLimitedSteps;
				stepStats->positiveJRejectedParticleSteps +=
					sweepObservation.positiveJRejectedSteps;
				stepStats->nonFiniteRejectedParticleSteps +=
					sweepObservation.nonFiniteRejectedSteps;
				stepStats->finalMaxLocalSolveDisplacement =
					PxSqrt(
						sweepObservation.
							maxLocalSolveDisplacementSq);
				stepStats->finalMaxAppliedDisplacement =
					PxSqrt(maxDxSq);
				// Compatibility alias for the original schema.
				stepStats->finalMaxDisplacement =
					stepStats->finalMaxAppliedDisplacement;
			}
			if(sweepObservation.trustRegionLimitedSteps > 0)
				componentOgcSafetyEpochLimited = true;
			if(componentOgcSafetyEpochActive &&
				componentOgcSafetyEpochLimited)
			{
				innerIt = currentInnerIterations;
				break;
			}

			// A small applied displacement is not enough to terminate: a
			// trust-region or positive-J rejection can produce a zero step
			// while the local H^-1 f stationarity residual is still active.
			// Keep the legacy candidate count as diagnostics, but only the
			// pre-limiter residual below 1e-4 for two consecutive feasible
			// sweeps owns early termination.
			const bool appliedDisplacementConverged =
				sweepObservation.isAppliedDisplacementConverged(
					1e-12f);
			const bool strictResidualCandidateConverged =
				sweepObservation.isResidualConverged(1e-12f);
			const bool residualPolicyConverged =
				residualConvergence.observe(sweepObservation);
			const bool shadowResidual1e5Converged =
				sweepObservation.isResidualConverged(1e-10f);
			const bool shadowResidual1e4Converged =
				sweepObservation.isResidualConverged(1e-8f);
			shadowResidual1e5ConsecutiveSweeps =
				shadowResidual1e5Converged
					? shadowResidual1e5ConsecutiveSweeps + 1
					: 0;
			shadowResidual1e4ConsecutiveSweeps =
				shadowResidual1e4Converged
					? shadowResidual1e4ConsecutiveSweeps + 1
					: 0;
			if(!shadowResidual1e5Recorded &&
				shadowResidual1e5ConsecutiveSweeps >= 2)
			{
				shadowResidual1e5Recorded = true;
				if(stepStats)
				{
					stepStats->
						shadowResidual1e5ConvergedOuterIterations++;
					stepStats->
						shadowResidual1e5SavedInnerIterations +=
						currentInnerIterations - (innerIt + 1);
				}
			}
			if(!shadowResidual1e4Recorded &&
				shadowResidual1e4ConsecutiveSweeps >= 2)
			{
				shadowResidual1e4Recorded = true;
				if(stepStats)
				{
					stepStats->
						shadowResidual1e4ConvergedOuterIterations++;
					stepStats->
						shadowResidual1e4SavedInnerIterations +=
						currentInnerIterations - (innerIt + 1);
				}
			}
			if(appliedDisplacementConverged &&
				!legacyAppliedConvergenceRecorded)
			{
				legacyAppliedConvergenceRecorded = true;
				if(stepStats)
				{
					stepStats->
						legacyAppliedConvergedOuterIterations++;
					if(!strictResidualCandidateConverged)
						stepStats->
							unsafeAppliedConvergenceCandidates++;
				}
			}
			if(residualPolicyConverged)
			{
				if(stepStats)
					stepStats->residualConvergedOuterIterations++;
				break;
			}
			if(stepStats &&
				innerIt + 1 == currentInnerIterations)
				stepStats->budgetExhaustedOuterIterations++;

			// Adaptive spectral-radius estimation.
			// Iterations 0-1 are pure GS (Chebyshev starts at iteration 2).
			// Measure the GS convergence ratio from these iterations and use
			// min(measured, user-provided) as the Chebyshev rho.  This makes
			// the solver adapt to any mesh density / quality automatically.
			if (innerIt == 0)
			{
				prevMaxDxSq = maxDxSq;
			}
			else if (innerIt == 1 && useChebyshev)
			{
				if (prevMaxDxSq > 1e-20f)
				{
					PxReal measuredRho = PxSqrt(maxDxSq / prevMaxDxSq);
					// Use the more conservative of measured vs user-provided,
					// and never exceed 0.95 (safety ceiling).
					adaptiveRho = PxMin(measuredRho, chebyshevRho);
					adaptiveRho = PxMin(adaptiveRho, 0.95f);
				}
				prevMaxDxSq = maxDxSq;
			}

			// Chebyshev semi-iterative position relaxation
			// x_acc = x_{k-2} + omega_k * (x_GS - x_{k-2})
			if (useChebyshev && innerIt >= 2)
			{
				PxReal rhoSq = adaptiveRho * adaptiveRho;
				if (innerIt == 2)
					chebyOmega = 2.0f / (2.0f - rhoSq);
				else
					chebyOmega = 1.0f / (1.0f - rhoSq * chebyOmega * 0.25f);
				chebyOmega = PxMax(1.0f, PxMin(chebyOmega, 2.0f));

				// Divergence guard: if displacement grew since last iteration,
				// the rho is still too high.  Disable Chebyshev for the
				// remainder of this outer iteration.
				if (prevMaxDxSq > 1e-20f && maxDxSq > prevMaxDxSq * 1.1f)
				{
					chebyOmega = 1.0f;   // effectively no acceleration
					adaptiveRho = 0.0f;  // stays disabled for remaining inner its
				}

				if (chebyOmega > 1.0f)
				{
					for (PxU32 i = 0; i < numParticles; i++)
					{
						if (particles[i].isStatic()) continue;
						// Skip Chebyshev for particles with active contacts
						// (over-relaxation can push them through surfaces)
						if (contactStart[i + 1] > contactStart[i]) continue;
						particles[i].position = chebyPrevPrevPos[i] +
							(particles[i].position - chebyPrevPrevPos[i]) * chebyOmega;
						avbdTruncateDisplacement(
							particles[i],
							particles[i].outerPosition,
							selfCollisionSafetyBounds[i]);
					}
				}
				prevMaxDxSq = maxDxSq;
			}
			if (useChebyshev)
			{
				for (PxU32 i = 0; i < numParticles; i++)
				{
					chebyPrevPrevPos[i] = chebyPrevPos[i];
					chebyPrevPos[i] = particles[i].position;
				}
			}
		}
		if(stepStats && avbdUseParticlePrimalWorkCensus())
		{
			avbdAccumulateParticlePrimalWorkCensusForOuterEpoch(
				*stepStats, particles, softBodies, numSoftBodies,
				contactStart.begin(),
				stepStats->particleSweeps - particleSweepsBeforeOuter);
		}
		if(stepStats)
			stepStats->particleSolveMs +=
				stageTimer.getElapsedSeconds() * 1000.0;
		// Dual update (contacts, pins, elastic proximal)
		for (PxU32 ci = 0; ci < numContacts; ci++)
		{
			AvbdSoftContact& contact = contacts[ci];
			avbdUpdateSoftContactDual(
				contact.geometry, contact.state,
				particles, avbdBeta);
		}

		for (PxU32 si = 0; si < numSoftBodies; si++)
		{
			AvbdSoftBody& sb = softBodies[si];
			for (PxU32 oi = 0;
				oi < sb.runtime.compiledObjectives.size(); oi++)
			{
				const AvbdCompiledSoftObjective& objective =
					sb.runtime.compiledObjectives[oi];
				if (avbdIsPinPositionOwner(objective.owner))
				{
					avbdUpdatePinDual(
						sb.runtime.pins[objective.runtimeStateIndex],
						objective.point, particles, avbdBeta);
				}
				else
				{
					PX_ASSERT(
						avbdIsPinPositionOwner(
							objective.owner));
				}
			}
		}

		// AVBD elastic proximal dual update: increase proximal weight
		// proportional to displacement from the outer-iteration anchor.
		// The A/B-off route clears this state during prediction and never
		// regrows it, while leaving every other dual update unchanged.
		if(avbdUseSoftElasticProximal())
		{
			for (PxU32 i = 0; i < numParticles; i++)
			{
				AvbdSoftParticle& sp = particles[i];
				if (sp.isStatic()) continue;
				PxReal disp = (sp.position - sp.outerPosition).magnitude();
				sp.elasticK = PxMin(
					sp.elasticK + avbdBeta * disp, sp.elasticKMax);
			}
		}
		if(stepStats)
			stepStats->dualMs +=
				stageTimer.getElapsedSeconds() * 1000.0;

		// Re-detect only after the conservative soft/soft envelope was spent.
		// Otherwise this is the same DCD epoch and the manifold/AL state remains
		// current by construction; rebuilding it per outer iteration is the
		// sustained-contact cost this OGC scheduler is designed to remove.
		const bool mayReusePureSoftPairEpoch =
			redetectFn && contactsArray && outerIt + 1 < outerIterations &&
			componentOgcSafetyEpochActive &&
			!componentOgcSafetyEpochLimited;
		if(mayReusePureSoftPairEpoch)
		{
			reuseComponentOgcSafetyEpoch = true;
		}
		else if (redetectFn && contactsArray &&
			outerIt + 1 < outerIterations)
		{
			redetectFn(particles, numParticles, softBodies, numSoftBodies,
					   *contactsArray, redetectUserData);
			contacts = contactsArray->begin();
			numContacts = contactsArray->size();
			avbdCompileSoftVelocityObjectives(
				compiledVelocityObjectives, componentFinalizeModes,
				softBodies, numSoftBodies, contacts, numContacts);
			// Matching rows retain the original frame anchor through state
			// transfer; only contacts born at this redetection are initialized.
			avbdInitializeSoftContactDepenetrationLimits(
				contacts, numContacts, particles,
				softBodies, numSoftBodies, dt);
			// Rebuild this epoch's per-particle contact index and causal plan.
			avbdBuildSoftParticleContactIndex(
				workspace, softBodies, numSoftBodies,
				contacts, numContacts, numParticles, stepStats,
				particlePrimalSchedule,
				validateParticlePrimalAccessPlan, particles);
			componentOgcSafetyEpochActive = false;
		}
		if(stepStats)
			stepStats->redetectMs +=
				stageTimer.getElapsedSeconds() * 1000.0;
	}

	// Stage 3: terminal same-time DCD, recovery, then velocity update.  This
	// is a contact-epoch refresh, not a time substep or a swept CCD pass.
	avbdRefreshComponentTerminalOgcEpoch(
		particles, numParticles, softBodies, numSoftBodies,
		redetectFn, contactsArray, redetectUserData,
		contacts, numContacts, workspace);
	avbdApplyWorldStaticComponentEndpointDcdRecovery(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, workspace);
	for (PxU32 i = 0; i < numParticles; i++)
		particles[i].updateVelocityFromPosition(invDt);
	avbdApplyBendingDamping(
		particles, softBodies, numSoftBodies, dt);
	avbdFinalizeSoftComponentVelocities(
		particles, numParticles,
		softBodies, numSoftBodies,
		componentMomentumTargets.begin(),
		componentFinalizeModes.begin(),
		contacts, numContacts,
		compiledVelocityObjectives.begin(),
		compiledVelocityObjectives.size(), invDt);
	const AvbdOgcGeometryEpochView componentGeometryEpoch =
		workspace.getComponentOgcGeometryEpochView();
	avbdProjectSoftContactVelocityTangents(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, dt, stepStats, &componentGeometryEpoch);
	avbdClampWorldStaticComponentEndpointDcdVelocities(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, workspace);
	avbdProjectWorldFixedPins(
		particles, numParticles, softBodies, numSoftBodies);
	// A completed solve has advanced positions again; never expose the
	// one-redetection prediction cache to an unrelated later query.
	workspace.contact.invalidateSoftBodyBounds();
	if(stepStats)
		stepStats->velocityMs += stageTimer.getElapsedSeconds() * 1000.0;
	if(stepStats)
	{
		stepStats->workspaceGrowthEvents = workspace.growthEvents;
		stepStats->workspaceGrowthBytes = workspace.growthBytes;
		stepStats->contactWorkspaceGrowthEvents =
			workspace.contact.growthEvents;
		stepStats->contactWorkspaceGrowthBytes =
			workspace.contact.growthBytes;
		stepStats->contactSweepScratchGrowthEvents =
			workspace.contact.sweepScratchGrowthEvents;
		stepStats->contactSweepScratchGrowthBytes =
			workspace.contact.sweepScratchGrowthBytes;
		stepStats->contactOutputGrowthEvents =
			workspace.contact.outputGrowthEvents;
		stepStats->contactOutputGrowthBytes =
			workspace.contact.outputGrowthBytes;
		stepStats->peakContactOutputCount =
			workspace.contact.peakOutputContactCount;
		stepStats->peakContactOutputCapacity =
			workspace.contact.peakOutputContactCapacity;
		stepStats->peakContactIncidenceCount =
			workspace.peakContactIncidenceCount;
		stepStats->peakContactIncidenceCapacity =
			workspace.peakContactIncidenceCapacity;
		stepStats->peakStateTransferContactCount =
			workspace.contact.peakPreviousContactCount;
		stepStats->peakStateTransferContactCapacity =
			workspace.contact.peakPreviousContactCapacity;
		stepStats->peakStateTransferUsedCapacity =
			workspace.contact.peakPreviousUsedCapacity;
	}

}

} // namespace Dy
} // namespace physx
