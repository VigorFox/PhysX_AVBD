// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Hot template kernel included after mechanics and contact evaluators.

template<
	bool enableTetMaterialPackets,
	bool tetMaterialPacketEligibilityProven>
PX_FORCE_INLINE void AvbdParticlePrimalSolveContext::solve(
	const AvbdSoftBody& sb, PxU32 localParticleIndex,
	AvbdParticlePrimalRangeObservation& observation) const
{
	const PxU32 particleIndex =
		sb.compiled.particleStart + localParticleIndex;
	AvbdSoftParticle& particle = particles[particleIndex];
	if(particle.isStatic())
		return;

	// Inertial term
	const PxReal massDtSq = particle.mass * invDtSq;
	PxMat33 H = PxMat33::createDiagonal(PxVec3(massDtSq));
	PxVec3 f = (particle.predictedPosition - particle.position) *
		massDtSq;

	const AvbdParticleElementAdjacency& elementAdjacency =
		sb.compiled.elementAdjacency[localParticleIndex];
	const AvbdParticleObjectiveAdjacency& objectiveAdjacency =
		sb.runtime.objectiveAdjacency[localParticleIndex];
	static const PxU32 eMAX_CACHED_TET_INCIDENCE = 64;
	AvbdTetVertexLinearization tetLinearizations[
		eMAX_CACHED_TET_INCIDENCE];
	const PxU32 tetIncidenceCount = elementAdjacency.tetRefs.size();
	const bool cacheTetLinearizations = tetIncidenceCount <=
		eMAX_CACHED_TET_INCIDENCE;
	if(!cacheTetLinearizations)
		observation.tetLinearizationCacheFallbackParticleSteps++;

	// Triangle (StVK) contributions
	for(PxU32 triangleRefIndex = 0;
		triangleRefIndex < elementAdjacency.triRefs.size();
		triangleRefIndex++)
	{
		const AvbdParticleElementRef& ref =
			elementAdjacency.triRefs[triangleRefIndex];
		PxVec3 elementForce;
		PxMat33 elementHessian;
		avbdEvaluateStVKForceHessian(
			sb.compiled.triElements[ref.index], int(ref.vOrder),
			sb.material.mu, sb.material.lambda, particles,
			elementForce, elementHessian);
		f = f + elementForce;
		H = H + elementHessian;
	}

	// Tetrahedral material-model contributions.  The scalar loop remains
	// the authority and the default route.  P8.3's candidate is admitted
	// only for a valid canonical program with at least one full packet.
	const bool useTetMaterialPackets =
		enableTetMaterialPackets &&
		(tetMaterialPacketEligibilityProven ||
		 canUseTetMaterialPackets(sb, localParticleIndex));
	if(useTetMaterialPackets)
		avbdAccumulateTetMaterialPacketContributions(
			sb, localParticleIndex, particles,
			tetMaterialPacketKernels, cacheTetLinearizations,
			tetLinearizations, f, H);
	else
	{
		for(PxU32 tetRefIndex = 0;
			tetRefIndex < elementAdjacency.tetRefs.size(); tetRefIndex++)
		{
			const AvbdParticleElementRef& ref =
				elementAdjacency.tetRefs[tetRefIndex];
			PxVec3 elementForce;
			PxMat33 elementHessian;
			if(sb.material.coRotationalVolumeModel)
				avbdEvaluateCorotationalForceHessianPrepared(
					sb.compiled.tetElements[ref.index], int(ref.vOrder),
					sb.material.mu, sb.material.lambda, particles,
					elementForce, elementHessian,
					cacheTetLinearizations
						? &tetLinearizations[tetRefIndex] : NULL);
			else
				avbdEvaluateNeoHookeanForceHessianPrepared(
					sb.compiled.tetElements[ref.index], int(ref.vOrder),
					sb.material.mu, sb.material.lambda,
					sb.material.neoHookeanAlpha, particles,
					elementForce, elementHessian,
					cacheTetLinearizations
						? &tetLinearizations[tetRefIndex] : NULL);
			f = f + elementForce;
			H = H + elementHessian;
		}
	}

	// Bending contributions
	for(PxU32 bendRefIndex = 0;
		bendRefIndex < elementAdjacency.bendRefs.size(); bendRefIndex++)
	{
		const AvbdParticleElementRef& ref =
			elementAdjacency.bendRefs[bendRefIndex];
		PxVec3 elementForce;
		PxMat33 elementHessian;
		avbdEvaluateBendingForceHessian(
			sb.compiled.bendElements[ref.index], int(ref.vOrder),
			sb.material.bendingStiffness, particles,
			elementForce, elementHessian);
		f = f + elementForce;
		H = H + elementHessian;
	}

	// Contact contributions (indexed lookup)
	for(PxU32 contactRefIndex = contactStarts[particleIndex];
		contactRefIndex < contactStarts[particleIndex + 1];
		contactRefIndex++)
	{
		PxVec3 contactForce;
		PxMat33 contactHessian;
		const AvbdSoftContactParticleRef& contactRef =
			contactIndices[contactRefIndex];
		const AvbdSoftContact& contact = contacts[contactRef.contactIndex];
		avbdEvaluateContactParticleBlock(
			contact.geometry, contact.state, particles,
			contactRef.jacobianScale, contactForce, contactHessian);
		f = f + contactForce;
		H = H + contactHessian;
	}

	// Scene-external component supports only compiled one-way pin owners.
	// Rigid attachments require the low-level rigid-body block and must
	// never be consumed as a one-way particle-only objective here.
	for(PxU32 objectiveRefIndex = 0;
		objectiveRefIndex < objectiveAdjacency.objectiveIndices.size();
		objectiveRefIndex++)
	{
		const PxU32 objectiveIndex =
			objectiveAdjacency.objectiveIndices[objectiveRefIndex];
		const AvbdCompiledSoftObjective& objective =
			sb.runtime.compiledObjectives[objectiveIndex];
		if(!avbdIsPinPositionOwner(objective.owner))
		{
			PX_ASSERT(avbdIsPinPositionOwner(objective.owner));
			continue;
		}
		PxVec3 pinForce;
		PxMat33 pinHessian;
		avbdEvaluatePinForceHessian(
			objective.point,
			sb.runtime.pins[objective.runtimeStateIndex], particles,
			particleIndex, pinForce, pinHessian);
		f = f + pinForce;
		H = H + pinHessian;
	}

	// Stiffness-proportional Rayleigh damping (Newton VBD style):
	// Per-axis damping is proportional to elastic stiffness and clamped
	// so no axis receives less damping than the mass-proportional floor.
	if(particle.damping > 0.0f)
	{
		const PxReal dampingCoefficient =
			particle.damping * particle.mass * invDt;
		const PxReal elasticHxx = PxMax(H.column0.x - massDtSq, 0.0f);
		const PxReal elasticHyy = PxMax(H.column1.y - massDtSq, 0.0f);
		const PxReal elasticHzz = PxMax(H.column2.z - massDtSq, 0.0f);
		const PxReal traceElasticH =
			elasticHxx + elasticHyy + elasticHzz;
		PxReal dampingX;
		PxReal dampingY;
		PxReal dampingZ;
		if(traceElasticH > 1e-10f)
		{
			const PxReal scale = dampingCoefficient * 3.0f /
				traceElasticH;
			dampingX = PxMax(elasticHxx * scale, dampingCoefficient);
			dampingY = PxMax(elasticHyy * scale, dampingCoefficient);
			dampingZ = PxMax(elasticHzz * scale, dampingCoefficient);
		}
		else
			dampingX = dampingY = dampingZ = dampingCoefficient;
		const PxVec3 dampingDisplacement =
			particle.position - particle.initialPosition;
		f.x -= dampingX * dampingDisplacement.x;
		f.y -= dampingY * dampingDisplacement.y;
		f.z -= dampingZ * dampingDisplacement.z;
		H.column0.x += dampingX;
		H.column1.y += dampingY;
		H.column2.z += dampingZ;
	}

	// AVBD elastic proximal term: pulls toward the outer-iteration anchor.
	if(particle.elasticK > 0.0f)
	{
		H.column0.x += particle.elasticK;
		H.column1.y += particle.elasticK;
		H.column2.z += particle.elasticK;
		f = f + (particle.outerPosition - particle.position) *
			particle.elasticK;
	}

	const PxVec3 localSolveDisplacement = avbdSolveSymmetric33(H, f);
	const PxReal localSolveDisplacementSq =
		localSolveDisplacement.magnitudeSquared();
	PxVec3 proposedDisplacement = localSolveDisplacement;
	bool trustRegionLimited = false;
	const PxReal maxDisplacement = 1.0f;
	AvbdSoftTetDisplacementLimitResult limitResult;
	if(localSolveDisplacement.isFinite() &&
		PxIsFinite(localSolveDisplacementSq))
	{
		if(localSolveDisplacementSq >
			maxDisplacement * maxDisplacement)
		{
			proposedDisplacement *= maxDisplacement /
				PxSqrt(localSolveDisplacementSq);
			trustRegionLimited = true;
		}
		limitResult = cacheTetLinearizations
			? avbdLimitTetDisplacementFromLinearizations(
				proposedDisplacement, tetLinearizations, tetIncidenceCount)
			: avbdLimitTetDisplacementObserved(
				sb, particleIndex, particles, proposedDisplacement);
	}
	else
	{
		limitResult = AvbdSoftTetDisplacementLimitResult(
			PxVec3(0.0f), 0.0f,
			AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
	}
	const PxVec3 positionBeforeStep = particle.position;
	if(limitResult.appliedDisplacement.isFinite())
	{
		particle.position += limitResult.appliedDisplacement;
		const PxVec3 positionBeforeOgc = particle.position;
		avbdTruncateDisplacement(
			particle, particle.outerPosition,
			selfCollisionSafetyBounds[particleIndex]);
		if((particle.position - positionBeforeOgc).magnitudeSquared() >
			1.0e-20f)
			trustRegionLimited = true;
		limitResult.appliedDisplacement =
			particle.position - positionBeforeStep;
	}
	observation.sweepObservation.observe(
		localSolveDisplacement, trustRegionLimited, limitResult);
}
