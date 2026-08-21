// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

// Keep the resumable component continuation implementation out of every
// translation unit that consumes the scalar component API. The declaration
// and minimal shared particle-primal contract remain in the component header;
// this TU is the sole owner of state-machine method definitions.
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

AvbdSoftBodyStepState::AvbdSoftBodyStepState()
	: particles(NULL), numParticles(0), softBodies(NULL), numSoftBodies(0),
	  contacts(NULL), numContacts(0), dt(0.0f), invDt(0.0f),
	  invDtSq(0.0f), outerIterations(0),
	  requestedInnerIterationBudget(0), remainingInnerIterationBudget(0),
	  outerIt(0), currentInnerIterations(0), innerIt(0), avbdBeta(0.0f),
	  redetectFn(NULL), contactsArray(NULL), redetectUserData(NULL),
	  chebyshevRho(0.0f), useChebyshev(false), chebyOmega(1.0f),
	  adaptiveRho(0.0f), prevMaxDxSq(0.0f),
	  shadowResidual1e5ConsecutiveSweeps(0),
	  shadowResidual1e4ConsecutiveSweeps(0),
	  shadowResidual1e5Recorded(false), shadowResidual1e4Recorded(false),
	  legacyAppliedConvergenceRecorded(false), residualConsecutiveSweeps(0),
	  stepStats(NULL), workspace(NULL), selfCollisionAdjacencies(NULL),
	  numSelfCollisionAdjacencies(0), selfCollisionEnabled(NULL),
	  ogcParams(NULL), deferRedetectionToParent(false),
	  pendingInitialRedetection(false),
	  reuseComponentOgcSafetyEpoch(false),
	  componentOgcSafetyEpochActive(false),
	  componentOgcSafetyEpochLimited(false),
	  publishIndependentBodySweeps(false),
	  independentBodySweepPublished(false),
	  particlePrimalSchedule(AvbdParticlePrimalSchedule::eSERIAL_LINEAR),
	  validateParticlePrimalAccessPlan(false),
	  phase(Phase::eIDLE)
{
}

bool AvbdSoftBodyStepState::beginAfterPrediction(
	AvbdSoftParticle* inputParticles, PxU32 inputNumParticles,
	AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
	AvbdSoftContact* inputContacts, PxU32 inputNumContacts,
	PxReal inputDt, PxU32 inputOuterIterations,
	PxU32 inputInnerIterations, PxU32 inputRequestedInnerBudget,
	PxReal inputAvbdBeta, AvbdContactRedetectFn inputRedetectFn,
	PxArray<AvbdSoftContact>* inputContactsArray,
	void* inputRedetectUserData, PxReal inputChebyshevRho,
	AvbdSoftBodyStepStats* inputStepStats,
	AvbdSoftBodyWorkspace& inputWorkspace,
	const AvbdSelfCollisionAdjacency* inputSelfCollisionAdjacencies,
	PxU32 inputNumSelfCollisionAdjacencies,
	const PxU8* inputSelfCollisionEnabled,
	const AvbdOGCParams* inputOgcParams,
	AvbdParticlePrimalSchedule inputParticlePrimalSchedule,
	bool inputDeferRedetectionToParent,
	bool inputPublishIndependentBodySweeps)
{
	PX_UNUSED(inputInnerIterations);
	if(!inputParticles || inputNumParticles == 0 ||
		!inputSoftBodies || inputNumSoftBodies == 0 ||
		(phase != Phase::eIDLE && phase != Phase::eCOMPLETE))
	{
		phase = Phase::eINVALID;
		return false;
	}
	particles = inputParticles;
	numParticles = inputNumParticles;
	softBodies = inputSoftBodies;
	numSoftBodies = inputNumSoftBodies;
	contacts = inputContacts;
	numContacts = inputNumContacts;
	dt = inputDt;
	invDt = dt > 0.0f ? 1.0f / dt : 0.0f;
	invDtSq = invDt * invDt;
	outerIterations = inputOuterIterations;
	requestedInnerIterationBudget = inputRequestedInnerBudget;
	remainingInnerIterationBudget = requestedInnerIterationBudget;
	avbdBeta = inputAvbdBeta;
	redetectFn = inputRedetectFn;
	contactsArray = inputContactsArray;
	redetectUserData = inputRedetectUserData;
	chebyshevRho = inputChebyshevRho;
	useChebyshev = chebyshevRho > 0.0f && chebyshevRho < 1.0f;
	chebyOmega = 1.0f;
	adaptiveRho = chebyshevRho;
	stepStats = inputStepStats;
	avbdPublishTetMaterialPacketIrStats(
		softBodies, numSoftBodies, stepStats);
	workspace = &inputWorkspace;
	selfCollisionAdjacencies = inputSelfCollisionAdjacencies;
	numSelfCollisionAdjacencies = inputNumSelfCollisionAdjacencies;
	selfCollisionEnabled = inputSelfCollisionEnabled;
	ogcParams = inputOgcParams;
	deferRedetectionToParent = inputDeferRedetectionToParent;
	pendingInitialRedetection = false;
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochActive = false;
	componentOgcSafetyEpochLimited = false;
	publishIndependentBodySweeps = inputPublishIndependentBodySweeps;
	independentBodySweepPublished = false;
	particlePrimalSchedule = inputParticlePrimalSchedule ==
		AvbdParticlePrimalSchedule::eDEFAULT
		? avbdGetParticlePrimalSchedule() : inputParticlePrimalSchedule;
	validateParticlePrimalAccessPlan =
		avbdValidateParticlePrimalAccessPlan();
	stageTimer = PxTime();

	// This is the original predicted-position contact epoch barrier.  It is
	// parent-owned and completes before the first outer/layer publication.
	if(redetectFn && contactsArray)
	{
		if(deferRedetectionToParent)
		{
			pendingInitialRedetection = true;
			phase = Phase::eREDETECTION;
			return true;
		}
		redetectFn(particles, numParticles, softBodies, numSoftBodies,
			*contactsArray, redetectUserData);
		contacts = contactsArray->begin();
		numContacts = contactsArray->size();
		avbdCompileSoftVelocityObjectives(
			workspace->compiledVelocityObjectives,
			workspace->componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
	}
	finishInitialRedetection();
	return true;
}

void AvbdSoftBodyStepState::finishInitialRedetection()
{
	avbdInitializeSoftContactDepenetrationLimits(
		contacts, numContacts, particles,
		softBodies, numSoftBodies, dt);
	PxArray<AvbdSoftComponentMomentumTarget>& momentumTargets =
		workspace->componentMomentumTargets;
	workspace->resize(momentumTargets, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		AvbdSoftComponentMomentumTarget& target = momentumTargets[bodyIndex];
		target = AvbdSoftComponentMomentumTarget();
		if(workspace->componentFinalizeModes[bodyIndex] ==
				AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		PxVec3 centroid(0.0f);
		PxMat33 inertia(PxZero);
		target.valid = avbdComputeSoftComponentMomentum(
			particles, numParticles, softBodies[bodyIndex], true, invDt,
			centroid, target.linearMomentum, target.angularMomentum,
			inertia, target.mass);
		target.centroid = centroid;
		avbdApplySoftComponentDampingToMomentumTarget(
			target, softBodies[bodyIndex], dt);
		PX_UNUSED(inertia);
	}
	if(stepStats)
		stepStats->predictionMs += stageTimer.getElapsedSeconds() * 1000.0;
	rebuildParticleContactIndex();
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochActive = false;
	componentOgcSafetyEpochLimited = false;
	if(stepStats)
		stepStats->contactIndexMs += stageTimer.getElapsedSeconds() * 1000.0;
	workspace->resize(workspace->selfCollisionSafetyBounds, numParticles);
	if(useChebyshev)
	{
		workspace->resize(workspace->chebyPrevPos, numParticles);
		workspace->resize(workspace->chebyPrevPrevPos, numParticles);
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			workspace->chebyPrevPos[particleIndex] =
				particles[particleIndex].position;
			workspace->chebyPrevPrevPos[particleIndex] =
				particles[particleIndex].position;
		}
	}
	outerIt = 0;
	phase = Phase::eOUTER_PREPARE;
}

bool AvbdSoftBodyStepState::completePendingRedetection()
{
	if(phase != Phase::eREDETECTION || !contactsArray)
		return false;
	contacts = contactsArray->begin();
	numContacts = contactsArray->size();
	avbdCompileSoftVelocityObjectives(
		workspace->compiledVelocityObjectives,
		workspace->componentFinalizeModes,
		softBodies, numSoftBodies, contacts, numContacts);
	if(pendingInitialRedetection)
	{
		pendingInitialRedetection = false;
		finishInitialRedetection();
		return true;
	}
	avbdInitializeSoftContactDepenetrationLimits(
		contacts, numContacts, particles,
		softBodies, numSoftBodies, dt);
	rebuildParticleContactIndex();
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochActive = false;
	componentOgcSafetyEpochLimited = false;
	if(stepStats)
		stepStats->redetectMs += stageTimer.getElapsedSeconds() * 1000.0;
	outerIt++;
	phase = Phase::eOUTER_PREPARE;
	return true;
}

void AvbdSoftBodyStepState::rebuildParticleContactIndex()
{
	const bool independentBodyEpoch =
		publishIndependentBodySweeps && numSoftBodies > 1 && numContacts == 0;
	if(independentBodyEpoch)
	{
		// A complete-body fan-out has no cross-child write dependency.  Its
		// contact epoch still needs the empty CSR consumed by the particle
		// kernel, but rebuilding the topology color plan would create an
		// O(particles + structural conflicts) plan that this epoch cannot use.
		workspace->particlePrimalDynamicConflictValid = false;
		workspace->particlePrimalColorPlanValid = false;
		workspace->particlePrimalColorCount = 0;
		workspace->particlePrimalDynamicAccessGroups.resize(0);
	}
	avbdBuildSoftParticleContactIndex(
		*workspace, softBodies, numSoftBodies,
		contacts, numContacts, numParticles, stepStats,
		independentBodyEpoch
			? AvbdParticlePrimalSchedule::eSERIAL_LINEAR
			: particlePrimalSchedule,
		independentBodyEpoch ? false : validateParticlePrimalAccessPlan,
		particles);
}

void AvbdSoftBodyStepState::prepareOuterIteration()
{
	PX_ASSERT(outerIt < outerIterations);
	if(stepStats)
		stepStats->executedOuterIterations++;
	const PxU32 remainingOuterIterations = outerIterations - outerIt;
	currentInnerIterations =
		(remainingInnerIterationBudget + remainingOuterIterations - 1) /
		remainingOuterIterations;
	remainingInnerIterationBudget -= currentInnerIterations;
	if(!reuseComponentOgcSafetyEpoch)
	{
		avbdSnapshotOuterPositionsScalar(
			particles, numParticles,
			workspace->selfCollisionSafetyBounds.begin());
		const AvbdOGCParams defaultOgcParams;
		const AvbdOGCParams& activeOgcParams =
			ogcParams ? *ogcParams : defaultOgcParams;
		if(selfCollisionAdjacencies)
		{
			for(PxU32 bodyIndex = 0;
				bodyIndex < numSoftBodies &&
				bodyIndex < numSelfCollisionAdjacencies; bodyIndex++)
			{
				if(selfCollisionEnabled && !selfCollisionEnabled[bodyIndex])
					continue;
				const AvbdSoftBody& body = softBodies[bodyIndex];
				avbdComputeSafetyBounds(
					body, particles, selfCollisionAdjacencies[bodyIndex],
					activeOgcParams.contactRadius, activeOgcParams.safetyRelax,
					workspace->bodySelfCollisionSafetyBounds,
					workspace->contact);
				for(PxU32 localIndex = 0;
					localIndex < body.compiled.particleCount; localIndex++)
				{
					const PxU32 particleIndex =
						body.compiled.particleStart + localIndex;
					if(particleIndex < numParticles)
						workspace->selfCollisionSafetyBounds[particleIndex] =
							workspace->bodySelfCollisionSafetyBounds[localIndex];
				}
			}
		}
		componentOgcSafetyEpochActive =
			avbdApplyComponentOgcEpochSafetyBounds(
				contacts, numContacts, softBodies, numSoftBodies, particles,
				activeOgcParams.contactRadius, activeOgcParams.safetyRelax,
				workspace->selfCollisionSafetyBounds.begin(), numParticles,
				*workspace);
	}
	// A reused epoch intentionally retains its original proximal anchor and
	// bound.  The next outer boundary will either retain it again or publish a
	// fresh DCD epoch after the limiter reports that it was spent.
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochLimited = false;
	if(useChebyshev)
	{
		chebyOmega = 1.0f;
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			workspace->chebyPrevPos[particleIndex] =
				particles[particleIndex].position;
			workspace->chebyPrevPrevPos[particleIndex] =
				particles[particleIndex].position;
		}
	}
	innerIt = 0;
	prevMaxDxSq = 0.0f;
	shadowResidual1e5ConsecutiveSweeps = 0;
	shadowResidual1e4ConsecutiveSweeps = 0;
	shadowResidual1e5Recorded = false;
	shadowResidual1e4Recorded = false;
	legacyAppliedConvergenceRecorded = false;
	residualConsecutiveSweeps = 0;
	phase = Phase::eINNER_BEGIN;
}

bool AvbdSoftBodyStepState::beginInnerSweep()
{
	if(innerIt >= currentInnerIterations)
	{
		phase = Phase::eDUAL;
		return false;
	}
	if(stepStats)
	{
		stepStats->executedInnerIterations++;
		stepStats->particleSweeps++;
	}
	particlePrimalObservation = AvbdParticlePrimalRangeObservation();
	particlePrimalSolveContext =
	{
		particles,
		contacts,
		workspace->contactStarts.begin(),
		workspace->contactIndices.begin(),
		workspace->selfCollisionSafetyBounds.begin(),
		invDt,
		invDtSq,
		avbdSelectTetMaterialPacketKernels(
			softBodies, numSoftBodies)
	};
	// A complete-body layer has no Gauss-Seidel dependency between children.
	// Scene freezes the structural eligibility for the whole step; the contact
	// epoch is checked here because redetection can introduce a soft pair at an
	// outer boundary.  Such an epoch immediately returns to scalar authority.
	if(publishIndependentBodySweeps && numSoftBodies > 1 && numContacts == 0)
	{
		independentBodySweepPublished = true;
		phase = Phase::eCAUSAL_LAYER;
		return true;
	}
	const bool useColoredPrimal =
		avbdUsesColoredParticlePrimalSchedule(
			particlePrimalSchedule) &&
		workspace->particlePrimalColorPlanValid;
	if(stepStats && avbdUsesColoredParticlePrimalSchedule(
		particlePrimalSchedule))
	{
		if(useColoredPrimal)
			stepStats->particlePrimalColoredSerialSweeps++;
		else
			stepStats->particlePrimalColoredSerialFallbackSweeps++;
	}
	if(useColoredPrimal)
	{
		const bool published = causalLayerState.begin(
			particlePrimalSolveContext, softBodies, numSoftBodies,
			workspace->particlePrimalBodyIndices.begin(), numParticles,
			workspace->particlePrimalColorParticles.begin(),
			workspace->particlePrimalColorOffsets.begin(),
			workspace->particlePrimalColorCount);
		if(!published)
		{
			phase = Phase::eINVALID;
			return false;
		}
		phase = Phase::eCAUSAL_LAYER;
		return true;
	}
	if(particlePrimalSolveContext.tetMaterialPacketKernels.hasAny())
		avbdSolveParticlePrimalTetMaterialPacketBodyRange(
			particlePrimalSolveContext, softBodies, numSoftBodies,
			particlePrimalObservation);
	else
	{
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
		{
			const AvbdSoftBody& body = softBodies[bodyIndex];
			for(PxU32 localIndex = 0;
				localIndex < body.compiled.particleCount; localIndex++)
				particlePrimalSolveContext.solve(
					body, localIndex, particlePrimalObservation);
		}
	}
	finishParticlePrimalSweep();
	return false;
}

void AvbdSoftBodyStepState::finishParticlePrimalSweep()
{
	const AvbdSoftSweepConvergenceObservation& sweepObservation =
		particlePrimalObservation.sweepObservation;
	const PxReal maxDxSq = sweepObservation.maxAppliedDisplacementSq;
	if(stepStats)
	{
		if(avbdUseParticlePrimalWorkCensus())
		{
			AvbdParticlePrimalWorkCensus workCensus;
			avbdRecordParticlePrimalWorkCensusForSweep(
				particles, softBodies, numSoftBodies,
				particlePrimalSolveContext.contactStarts, workCensus);
			avbdAccumulateParticlePrimalWorkCensus(*stepStats, workCensus);
		}
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
		stepStats->finalMaxLocalSolveDisplacement = PxSqrt(
			sweepObservation.maxLocalSolveDisplacementSq);
		stepStats->finalMaxAppliedDisplacement = PxSqrt(maxDxSq);
		stepStats->finalMaxDisplacement =
			stepStats->finalMaxAppliedDisplacement;
	}
	if(sweepObservation.trustRegionLimitedSteps > 0)
		componentOgcSafetyEpochLimited = true;
	// The OGC trust region is an epoch boundary, not a signal to spend the
	// remaining material sweeps at a clamped pose.  End this same-time epoch
	// immediately so updateDualAndRedetect() can publish a fresh manifold.
	if(componentOgcSafetyEpochActive && componentOgcSafetyEpochLimited)
	{
		innerIt = currentInnerIterations;
		phase = Phase::eDUAL;
		return;
	}

	const bool appliedDisplacementConverged =
		sweepObservation.isAppliedDisplacementConverged(1e-12f);
	const bool strictResidualCandidateConverged =
		sweepObservation.isResidualConverged(1e-12f);
	const bool residualPolicyConverged =
		sweepObservation.isResidualConverged(1e-8f)
			? ++residualConsecutiveSweeps >= 2
			: (residualConsecutiveSweeps = 0, false);
	const bool shadowResidual1e5Converged =
		sweepObservation.isResidualConverged(1e-10f);
	const bool shadowResidual1e4Converged =
		sweepObservation.isResidualConverged(1e-8f);
	shadowResidual1e5ConsecutiveSweeps = shadowResidual1e5Converged
		? shadowResidual1e5ConsecutiveSweeps + 1 : 0;
	shadowResidual1e4ConsecutiveSweeps = shadowResidual1e4Converged
		? shadowResidual1e4ConsecutiveSweeps + 1 : 0;
	if(!shadowResidual1e5Recorded &&
		shadowResidual1e5ConsecutiveSweeps >= 2)
	{
		shadowResidual1e5Recorded = true;
		if(stepStats)
		{
			stepStats->shadowResidual1e5ConvergedOuterIterations++;
			stepStats->shadowResidual1e5SavedInnerIterations +=
				currentInnerIterations - (innerIt + 1);
		}
	}
	if(!shadowResidual1e4Recorded &&
		shadowResidual1e4ConsecutiveSweeps >= 2)
	{
		shadowResidual1e4Recorded = true;
		if(stepStats)
		{
			stepStats->shadowResidual1e4ConvergedOuterIterations++;
			stepStats->shadowResidual1e4SavedInnerIterations +=
				currentInnerIterations - (innerIt + 1);
		}
	}
	if(appliedDisplacementConverged && !legacyAppliedConvergenceRecorded)
	{
		legacyAppliedConvergenceRecorded = true;
		if(stepStats)
		{
			stepStats->legacyAppliedConvergedOuterIterations++;
			if(!strictResidualCandidateConverged)
				stepStats->unsafeAppliedConvergenceCandidates++;
		}
	}
	if(residualPolicyConverged)
	{
		if(stepStats)
			stepStats->residualConvergedOuterIterations++;
		innerIt = currentInnerIterations;
		phase = Phase::eDUAL;
		return;
	}
	if(stepStats && innerIt + 1 == currentInnerIterations)
		stepStats->budgetExhaustedOuterIterations++;
	if(innerIt == 0)
		prevMaxDxSq = maxDxSq;
	else if(innerIt == 1 && useChebyshev)
	{
		if(prevMaxDxSq > 1e-20f)
		{
			PxReal measuredRho = PxSqrt(maxDxSq / prevMaxDxSq);
			adaptiveRho = PxMin(measuredRho, chebyshevRho);
			adaptiveRho = PxMin(adaptiveRho, 0.95f);
		}
		prevMaxDxSq = maxDxSq;
	}
	if(useChebyshev && innerIt >= 2)
	{
		const PxReal rhoSq = adaptiveRho * adaptiveRho;
		if(innerIt == 2)
			chebyOmega = 2.0f / (2.0f - rhoSq);
		else
			chebyOmega = 1.0f /
				(1.0f - rhoSq * chebyOmega * 0.25f);
		chebyOmega = PxMax(1.0f, PxMin(chebyOmega, 2.0f));
		if(prevMaxDxSq > 1e-20f && maxDxSq > prevMaxDxSq * 1.1f)
		{
			chebyOmega = 1.0f;
			adaptiveRho = 0.0f;
		}
		if(chebyOmega > 1.0f)
		{
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
			{
				if(particles[particleIndex].isStatic() ||
					workspace->contactStarts[particleIndex + 1] >
						workspace->contactStarts[particleIndex])
					continue;
				particles[particleIndex].position =
					workspace->chebyPrevPrevPos[particleIndex] +
					(particles[particleIndex].position -
						workspace->chebyPrevPrevPos[particleIndex]) *
						chebyOmega;
				avbdTruncateDisplacement(
					particles[particleIndex],
					particles[particleIndex].outerPosition,
					workspace->selfCollisionSafetyBounds[particleIndex]);
			}
		}
		prevMaxDxSq = maxDxSq;
	}
	if(useChebyshev)
	{
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			workspace->chebyPrevPrevPos[particleIndex] =
				workspace->chebyPrevPos[particleIndex];
			workspace->chebyPrevPos[particleIndex] =
				particles[particleIndex].position;
		}
	}
	innerIt++;
	phase = innerIt < currentInnerIterations
		? Phase::eINNER_BEGIN : Phase::eDUAL;
}

void AvbdSoftBodyStepState::updateDualAndRedetect()
{
	if(stepStats)
		stepStats->particleSolveMs +=
			stageTimer.getElapsedSeconds() * 1000.0;
	for(PxU32 contactIndex = 0; contactIndex < numContacts; contactIndex++)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		avbdUpdateSoftContactDual(
			contact.geometry, contact.state, particles, avbdBeta);
	}
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 objectiveIndex = 0;
			objectiveIndex < body.runtime.compiledObjectives.size();
			objectiveIndex++)
		{
			const AvbdCompiledSoftObjective& objective =
				body.runtime.compiledObjectives[objectiveIndex];
			PX_ASSERT(avbdIsPinPositionOwner(objective.owner));
			if(avbdIsPinPositionOwner(objective.owner))
				avbdUpdatePinDual(
					body.runtime.pins[objective.runtimeStateIndex],
					objective.point, particles, avbdBeta);
		}
	}
	if(avbdUseSoftElasticProximal())
	{
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			AvbdSoftParticle& particle = particles[particleIndex];
			if(particle.isStatic())
				continue;
			const PxReal displacement =
				(particle.position - particle.outerPosition).magnitude();
			particle.elasticK = PxMin(
				particle.elasticK + avbdBeta * displacement,
				particle.elasticKMax);
		}
	}
	if(stepStats)
		stepStats->dualMs += stageTimer.getElapsedSeconds() * 1000.0;
	const bool mayReusePureSoftPairEpoch =
		redetectFn && contactsArray && outerIt + 1 < outerIterations &&
		componentOgcSafetyEpochActive &&
		!componentOgcSafetyEpochLimited;
	if(mayReusePureSoftPairEpoch)
	{
		// The complete DCD manifold remains valid inside the conservative
		// inter-body envelope installed in prepareOuterIteration().  Reuse the
		// contact index and AL state exactly as Jolt-style XPBD iterations reuse
		// a substep manifold; the moment a particle reaches that envelope the
		// regular redetection route below resumes.
		reuseComponentOgcSafetyEpoch = true;
	}
	else if(redetectFn && contactsArray && outerIt + 1 < outerIterations)
	{
		if(deferRedetectionToParent)
		{
			reuseComponentOgcSafetyEpoch = false;
			componentOgcSafetyEpochActive = false;
			componentOgcSafetyEpochLimited = false;
			pendingInitialRedetection = false;
			phase = Phase::eREDETECTION;
			return;
		}
		redetectFn(particles, numParticles, softBodies, numSoftBodies,
			*contactsArray, redetectUserData);
		contacts = contactsArray->begin();
		numContacts = contactsArray->size();
		avbdCompileSoftVelocityObjectives(
			workspace->compiledVelocityObjectives,
			workspace->componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
		avbdInitializeSoftContactDepenetrationLimits(
			contacts, numContacts, particles,
			softBodies, numSoftBodies, dt);
		rebuildParticleContactIndex();
		reuseComponentOgcSafetyEpoch = false;
		componentOgcSafetyEpochActive = false;
		componentOgcSafetyEpochLimited = false;
	}
	if(stepStats)
		stepStats->redetectMs += stageTimer.getElapsedSeconds() * 1000.0;
	outerIt++;
	phase = Phase::eOUTER_PREPARE;
}

void AvbdSoftBodyStepState::finalizeStep()
{
	// The final material sweep can consume the remaining OGC safety envelope.
	// Publish a same-time DCD epoch before terminal recovery rather than
	// applying a cached normal to the final pose.  Scene-owned deferred
	// redetection is intentionally left to its parent continuation: calling
	// the callback here would race its shared collision workspace.
	if(!deferRedetectionToParent)
		avbdRefreshComponentTerminalOgcEpoch(
			particles, numParticles, softBodies, numSoftBodies,
			redetectFn, contactsArray, redetectUserData,
			contacts, numContacts, *workspace);

	// Component fallback has no native post-AL static recovery.  Perform the
	// narrow true-gap endpoint translation before reconstructing velocity, so
	// its geometric correction cannot become a spurious separating impulse.
	avbdApplyWorldStaticComponentEndpointDcdRecovery(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, *workspace);
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; particleIndex++)
		particles[particleIndex].updateVelocityFromPosition(invDt);
	avbdApplyBendingDamping(
		particles, softBodies, numSoftBodies, dt);
	avbdFinalizeSoftComponentVelocities(
		particles, numParticles, softBodies, numSoftBodies,
		workspace->componentMomentumTargets.begin(),
		workspace->componentFinalizeModes.begin(),
		contacts, numContacts,
		workspace->compiledVelocityObjectives.begin(),
		workspace->compiledVelocityObjectives.size(), invDt);
	const AvbdOgcGeometryEpochView componentGeometryEpoch =
		workspace->getComponentOgcGeometryEpochView();
	avbdProjectSoftContactVelocityTangents(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, dt, stepStats, &componentGeometryEpoch);
	avbdClampWorldStaticComponentEndpointDcdVelocities(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, *workspace);
	avbdProjectWorldFixedPins(
		particles, numParticles, softBodies, numSoftBodies);
	workspace->contact.invalidateSoftBodyBounds();
	if(stepStats)
	{
		stepStats->velocityMs += stageTimer.getElapsedSeconds() * 1000.0;
		stepStats->workspaceGrowthEvents = workspace->growthEvents;
		stepStats->workspaceGrowthBytes = workspace->growthBytes;
		stepStats->contactWorkspaceGrowthEvents =
			workspace->contact.growthEvents;
		stepStats->contactWorkspaceGrowthBytes =
			workspace->contact.growthBytes;
		stepStats->contactSweepScratchGrowthEvents =
			workspace->contact.sweepScratchGrowthEvents;
		stepStats->contactSweepScratchGrowthBytes =
			workspace->contact.sweepScratchGrowthBytes;
		stepStats->contactOutputGrowthEvents =
			workspace->contact.outputGrowthEvents;
		stepStats->contactOutputGrowthBytes =
			workspace->contact.outputGrowthBytes;
		stepStats->peakContactOutputCount =
			workspace->contact.peakOutputContactCount;
		stepStats->peakContactOutputCapacity =
			workspace->contact.peakOutputContactCapacity;
		stepStats->peakContactIncidenceCount =
			workspace->peakContactIncidenceCount;
		stepStats->peakContactIncidenceCapacity =
			workspace->peakContactIncidenceCapacity;
		stepStats->peakStateTransferContactCount =
			workspace->contact.peakPreviousContactCount;
		stepStats->peakStateTransferContactCapacity =
			workspace->contact.peakPreviousContactCapacity;
		stepStats->peakStateTransferUsedCapacity =
			workspace->contact.peakPreviousUsedCapacity;
	}
}

AvbdSoftBodyStepAdvanceResult AvbdSoftBodyStepState::advance()
{
	for(;;)
	{
		switch(phase)
		{
		case Phase::eOUTER_PREPARE:
			if(outerIt >= outerIterations)
			{
				finalizeStep();
				phase = Phase::eCOMPLETE;
				return AvbdSoftBodyStepAdvanceResult::eCOMPLETE;
			}
			prepareOuterIteration();
			break;
		case Phase::eINNER_BEGIN:
			if(beginInnerSweep())
				return AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY;
			break;
	case Phase::eCAUSAL_LAYER:
		return AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY;
	case Phase::eDUAL:
		updateDualAndRedetect();
		break;
	case Phase::eREDETECTION:
		return AvbdSoftBodyStepAdvanceResult::eREDETECTION_READY;
		case Phase::eCOMPLETE:
			return AvbdSoftBodyStepAdvanceResult::eCOMPLETE;
		case Phase::eINVALID:
		case Phase::eIDLE:
			return AvbdSoftBodyStepAdvanceResult::eINVALID;
		}
	}
}

bool AvbdSoftBodyStepState::getPublishedCausalLayer(
	PxU32& layerIndex, PxU32& packedBegin, PxU32& packedEnd,
	const AvbdParticlePrimalSolveContext*& solveContext,
	const AvbdSoftBody*& bodies, PxU32& bodyCount,
	const PxU32*& particleBodyIndices,
	const PxU32*& packedParticleIndices) const
{
	if(independentBodySweepPublished ||
		phase != Phase::eCAUSAL_LAYER ||
		!causalLayerState.hasPublishedLayer())
		return false;
	layerIndex = causalLayerState.getPublishedLayerIndex();
	causalLayerState.getPublishedPackedRange(packedBegin, packedEnd);
	solveContext = &particlePrimalSolveContext;
	bodies = softBodies;
	bodyCount = numSoftBodies;
	particleBodyIndices = workspace->particlePrimalBodyIndices.begin();
	packedParticleIndices = workspace->particlePrimalColorParticles.begin();
	return true;
}

bool AvbdSoftBodyStepState::getPublishedIndependentBodySweep(
	const AvbdParticlePrimalSolveContext*& solveContext,
	const AvbdSoftBody*& bodies, PxU32& bodyCount) const
{
	if(phase != Phase::eCAUSAL_LAYER ||
		!independentBodySweepPublished || numSoftBodies < 2)
		return false;
	solveContext = &particlePrimalSolveContext;
	bodies = softBodies;
	bodyCount = numSoftBodies;
	return true;
}

bool AvbdSoftBodyStepState::completePublishedIndependentBodySweep(
	const AvbdParticlePrimalRangeObservation* observations,
	PxU32 observationCount)
{
	if(phase != Phase::eCAUSAL_LAYER ||
		!independentBodySweepPublished || !observations ||
		observationCount == 0)
		return false;
	particlePrimalObservation = AvbdParticlePrimalRangeObservation();
	for(PxU32 observationIndex = 0;
		observationIndex < observationCount; observationIndex++)
		particlePrimalObservation.merge(observations[observationIndex]);
	independentBodySweepPublished = false;
	finishParticlePrimalSweep();
	return true;
}

bool AvbdSoftBodyStepState::completePublishedCausalLayer(
	const AvbdParticlePrimalRangeObservation* observations,
	PxU32 observationCount)
{
	if(phase != Phase::eCAUSAL_LAYER ||
		!causalLayerState.completePublishedLayer(
			observations, observationCount))
		return false;
	if(causalLayerState.hasPublishedLayer())
		return true;
	particlePrimalObservation = causalLayerState.getSweepObservation();
	finishParticlePrimalSweep();
	return true;
}

void AvbdSoftBodyStepState::runToCompletionSerial()
{
	for(;;)
	{
		const AvbdSoftBodyStepAdvanceResult result = advance();
		if(result == AvbdSoftBodyStepAdvanceResult::eREDETECTION_READY)
		{
			if(!redetectFn || !contactsArray)
			{
				phase = Phase::eINVALID;
				return;
			}
			redetectFn(particles, numParticles, softBodies, numSoftBodies,
				*contactsArray, redetectUserData);
			if(!completePendingRedetection())
			{
				phase = Phase::eINVALID;
				return;
			}
			continue;
		}
		if(result != AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			return;
		PxU32 layerIndex = 0;
		PxU32 packedBegin = 0;
		PxU32 packedEnd = 0;
		const AvbdParticlePrimalSolveContext* solveContext = NULL;
		const AvbdSoftBody* bodies = NULL;
		PxU32 bodyCount = 0;
		const PxU32* particleBodyIndices = NULL;
		const PxU32* packedParticleIndices = NULL;
		const bool published = getPublishedCausalLayer(
			layerIndex, packedBegin, packedEnd, solveContext,
			bodies, bodyCount, particleBodyIndices, packedParticleIndices);
		PX_UNUSED(layerIndex);
		if(!published)
		{
			const AvbdParticlePrimalSolveContext* bodySolveContext = NULL;
			const AvbdSoftBody* bodyRangeBodies = NULL;
			PxU32 bodyRangeCount = 0;
			if(!getPublishedIndependentBodySweep(
				bodySolveContext, bodyRangeBodies, bodyRangeCount))
			{
				phase = Phase::eINVALID;
				return;
			}
			AvbdParticlePrimalRangeObservation bodyObservation;
			avbdSolveParticlePrimalIndependentBodyRange(
				*bodySolveContext, bodyRangeBodies, bodyRangeCount,
				0, bodyRangeCount, bodyObservation);
			if(!completePublishedIndependentBodySweep(
				&bodyObservation, 1))
			{
				phase = Phase::eINVALID;
				return;
			}
			continue;
		}
		AvbdParticlePrimalRangeObservation observation;
		avbdSolveParticlePrimalPackedRange(
			*solveContext, bodies, bodyCount, particleBodyIndices,
			numParticles, packedParticleIndices, packedBegin, packedEnd,
			observation);
		if(!completePublishedCausalLayer(&observation, 1))
		{
			phase = Phase::eINVALID;
			return;
		}
	}
}

} // namespace Dy
} // namespace physx
