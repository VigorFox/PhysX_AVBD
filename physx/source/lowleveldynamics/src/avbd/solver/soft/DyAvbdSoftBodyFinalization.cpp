// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

// Component finalization and velocity-objective boundary.  Position AL and
// contact detection publish inputs here; this unit computes momentum targets
// and the final velocity-owned response without owning Scene scheduling.

bool avbdComputeSoftComponentMomentum(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody& body, bool usePrediction, PxReal invDt,
	PxVec3& centroid, PxVec3& linearMomentum,
	PxVec3& angularMomentum, PxMat33& inertia, PxReal& mass)
{
	centroid = PxVec3(0.0f);
	linearMomentum = PxVec3(0.0f);
	angularMomentum = PxVec3(0.0f);
	inertia = PxMat33::createDiagonal(PxVec3(0.0f));
	mass = 0.0f;
	const PxU32 particleStart = body.compiled.particleStart;
	const PxU32 particleCount = body.compiled.particleCount;
	if(particleStart > numParticles ||
		particleCount > numParticles - particleStart)
		return false;
	for(PxU32 localIndex = 0; localIndex < particleCount; localIndex++)
	{
		const AvbdSoftParticle& particle =
			particles[particleStart + localIndex];
		if(particle.invMass <= 0.0f || particle.mass <= 0.0f)
			continue;
		const PxVec3 position =
			usePrediction ? particle.initialPosition : particle.position;
		const PxVec3 velocity = usePrediction
			? (particle.predictedPosition -
			   particle.initialPosition) * invDt
			: particle.velocity;
		if(!position.isFinite() || !velocity.isFinite())
			return false;
		centroid += position * particle.mass;
		linearMomentum += velocity * particle.mass;
		mass += particle.mass;
	}
	if(mass <= 0.0f)
		return false;
	centroid *= 1.0f / mass;
	for(PxU32 localIndex = 0; localIndex < particleCount; localIndex++)
	{
		const AvbdSoftParticle& particle =
			particles[particleStart + localIndex];
		if(particle.invMass <= 0.0f || particle.mass <= 0.0f)
			continue;
		const PxVec3 position =
			usePrediction ? particle.initialPosition : particle.position;
		const PxVec3 velocity = usePrediction
			? (particle.predictedPosition -
			   particle.initialPosition) * invDt
			: particle.velocity;
		const PxVec3 offset = position - centroid;
		inertia = inertia +
			(PxMat33::createDiagonal(
				PxVec3(offset.magnitudeSquared())) -
			 avbdOuter(offset, offset)) * particle.mass;
		angularMomentum +=
			offset.cross(velocity) * particle.mass;
	}
	return linearMomentum.isFinite() &&
		angularMomentum.isFinite() &&
		PxIsFinite(inertia.getDeterminant());
}

void avbdApplySoftComponentDampingToMomentumTarget(
	AvbdSoftComponentMomentumTarget& target,
	const AvbdSoftBody& body, PxReal dt)
{
	if(!target.valid)
		return;
	// The particle solve damps deformation modes, but the component finalizer
	// subsequently restores the predicted rigid linear/angular momentum. Apply
	// the same timestep damping to that authoritative target so contact
	// ownership cannot resurrect an undamped rigid mode every frame (most
	// visibly as runaway rolling).
	const PxReal dampingScale = 1.0f /
		(1.0f + PxMax(body.material.damping, 0.0f) * dt);
	target.linearMomentum *= dampingScale;
	target.angularMomentum *= dampingScale;
}

namespace
{

struct AvbdWorldPinSupport
{
	PxU32 particleIndex;
	PxReal weight;
};

static PxU32 collectWorldPinSupport(
	const AvbdKinematicPin& pin, PxU32 numParticles,
	AvbdWorldPinSupport (&support)[4])
{
	PxU32 supportCount = 0;
	for(PxU32 endpoint = 0; endpoint < pin.point.particleCount; endpoint++)
	{
		const PxU32 particleIndex = pin.point.particleIndices[endpoint];
		const PxReal weight = pin.point.weights[endpoint];
		if(particleIndex >= numParticles || !PxIsFinite(weight))
			return 0;
		PxU32 existing = 0;
		for(; existing < supportCount; existing++)
		{
			if(support[existing].particleIndex == particleIndex)
			{
				support[existing].weight += weight;
				break;
			}
		}
		if(existing == supportCount)
		{
			if(supportCount == 4)
				return 0;
			support[supportCount].particleIndex = particleIndex;
			support[supportCount].weight = weight;
			supportCount++;
		}
	}
	return supportCount;
}

static bool projectWorldFixedPinPosition(
	const AvbdKinematicPin& pin,
	AvbdSoftParticle* particles, PxU32 numParticles)
{
	AvbdWorldPinSupport support[4];
	const PxU32 supportCount = collectWorldPinSupport(
		pin, numParticles, support);
	if(supportCount == 0 || !pin.worldTarget.isFinite())
		return false;

	PxVec3 point(0.0f);
	PxReal response = 0.0f;
	for(PxU32 index = 0; index < supportCount; index++)
	{
		const AvbdSoftParticle& particle =
			particles[support[index].particleIndex];
		if(!particle.position.isFinite() || !PxIsFinite(particle.invMass) ||
			particle.invMass < 0.0f || !PxIsFinite(support[index].weight))
			return false;
		point += particle.position * support[index].weight;
		response += particle.invMass * support[index].weight *
			support[index].weight;
	}
	if(!point.isFinite() || !PxIsFinite(response) || response <= 1.0e-20f)
		return false;

	const PxVec3 multiplier = (pin.worldTarget - point) / response;
	PxVec3 candidatePositions[4];
	for(PxU32 index = 0; index < supportCount; index++)
	{
		AvbdSoftParticle& particle =
			particles[support[index].particleIndex];
		candidatePositions[index] = particle.position + multiplier *
			(particle.invMass * support[index].weight);
		if(!candidatePositions[index].isFinite())
			return false;
	}
	for(PxU32 index = 0; index < supportCount; index++)
		particles[support[index].particleIndex].position =
			candidatePositions[index];
	return true;
}

static bool projectWorldFixedPinVelocity(
	const AvbdKinematicPin& pin,
	AvbdSoftParticle* particles, PxU32 numParticles)
{
	AvbdWorldPinSupport support[4];
	const PxU32 supportCount = collectWorldPinSupport(
		pin, numParticles, support);
	if(supportCount == 0)
		return false;

	PxVec3 pointVelocity(0.0f);
	PxReal response = 0.0f;
	for(PxU32 index = 0; index < supportCount; index++)
	{
		const AvbdSoftParticle& particle =
			particles[support[index].particleIndex];
		if(!particle.velocity.isFinite() || !PxIsFinite(particle.invMass) ||
			particle.invMass < 0.0f || !PxIsFinite(support[index].weight))
			return false;
		pointVelocity += particle.velocity * support[index].weight;
		response += particle.invMass * support[index].weight *
			support[index].weight;
	}
	if(!pointVelocity.isFinite() || !PxIsFinite(response) ||
		response <= 1.0e-20f)
		return false;

	const PxVec3 multiplier = -pointVelocity / response;
	PxVec3 candidateVelocities[4];
	for(PxU32 index = 0; index < supportCount; index++)
	{
		AvbdSoftParticle& particle =
			particles[support[index].particleIndex];
		candidateVelocities[index] = particle.velocity + multiplier *
			(particle.invMass * support[index].weight);
		if(!candidateVelocities[index].isFinite())
			return false;
	}
	for(PxU32 index = 0; index < supportCount; index++)
		particles[support[index].particleIndex].velocity =
			candidateVelocities[index];
	return true;
}

} // namespace

PX_NOINLINE void avbdProjectWorldFixedPins(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	if(!particles || !softBodies)
		return;
	// Shared-support pins form a very small constraint graph. A fixed number of
	// deterministic PGS sweeps makes each hard point exact without allocating a
	// dense system in this terminal path.
	for(PxU32 sweep = 0; sweep < 8; sweep++)
	{
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
		{
			const PxArray<AvbdKinematicPin>& pins =
				softBodies[bodyIndex].runtime.pins;
			for(PxU32 pinIndex = 0; pinIndex < pins.size(); pinIndex++)
			{
				if(pins[pinIndex].targetKind ==
					AvbdSoftPinTargetKind::eWORLD_FIXED)
					projectWorldFixedPinPosition(
						pins[pinIndex], particles, numParticles);
			}
		}
	}
	for(PxU32 sweep = 0; sweep < 8; sweep++)
	{
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
		{
			const PxArray<AvbdKinematicPin>& pins =
				softBodies[bodyIndex].runtime.pins;
			for(PxU32 pinIndex = 0; pinIndex < pins.size(); pinIndex++)
			{
				if(pins[pinIndex].targetKind ==
					AvbdSoftPinTargetKind::eWORLD_FIXED)
					projectWorldFixedPinVelocity(
						pins[pinIndex], particles, numParticles);
			}
		}
	}
}

// This is a once-per-step component finalization stage with several distinct
// contact-owner policies.  Keeping it out of the particle primal solve's code
// body limits instruction-cache and stack pressure without adding a call to
// the per-particle/per-sweep hot loop.
PX_NOINLINE void avbdFinalizeSoftComponentVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftComponentMomentumTarget* momentumTargets,
	const AvbdSoftComponentFinalizeMode* finalizeModes,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdCompiledSoftVelocityObjective* velocityObjectives,
	PxU32 numVelocityObjectives, PxReal invDt)
{
	if(!particles || !softBodies || !momentumTargets ||
		!finalizeModes || invDt <= 0.0f)
		return;
	bool hasSpeculativeCcdBody = false;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		hasSpeculativeCcdBody = hasSpeculativeCcdBody ||
			softBodies[bodyIndex].compiled.speculativeCCDEnabled;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftComponentMomentumTarget& target =
			momentumTargets[bodyIndex];
		const AvbdSoftComponentFinalizeMode mode =
			finalizeModes[bodyIndex];
		// Position-AL contacts already own local non-penetration in the particle
		// solve. Recasting their multipliers as one rigid component impulse
		// distributes a local impact across the whole deformable and injects
		// spurious translation/rotation at first contact.
		if(!target.valid ||
			mode == AvbdSoftComponentFinalizeMode::ePOSITION_OWNED ||
			mode == AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 centroid(0.0f);
		PxVec3 actualLinearMomentum(0.0f);
		PxVec3 actualAngularMomentum(0.0f);
		PxMat33 inertia(PxZero);
		PxReal mass = 0.0f;
		if(!avbdComputeSoftComponentMomentum(
				particles, numParticles, body, false, invDt,
				centroid, actualLinearMomentum,
				actualAngularMomentum, inertia, mass))
			continue;
		if(PxAbs(mass - target.mass) >
			PxMax(1.0e-5f, target.mass * 1.0e-5f))
			continue;

		const PxReal inertiaDeterminant = inertia.getDeterminant();
		const bool hasAngularResponse =
			PxIsFinite(inertiaDeterminant) &&
			PxAbs(inertiaDeterminant) > 1.0e-12f;
		const PxMat33 inverseInertia = hasAngularResponse
			? inertia.getInverse()
			: PxMat33::createDiagonal(PxVec3(0.0f));
		// Swept AL multipliers and contact-owned depenetration caps are not
		// discrete end-step impulses. Preserve their stable position-derived
		// component momentum; an opted-in swept body receives only the bounded
		// uniform velocity boundary below. Ordinary discrete collision batches
		// use AL's external force to restore their missing global angular impulse.
		const bool preservePositionDerivedMomentum =
			mode == AvbdSoftComponentFinalizeMode::ePOSITION_OWNED &&
			(hasSpeculativeCcdBody ||
			 body.compiled.maxDepenetrationVelocity < 1.0e20f);
		PxVec3 targetLinearMomentum = preservePositionDerivedMomentum
			? actualLinearMomentum : target.linearMomentum;
		PxVec3 targetAngularMomentum = preservePositionDerivedMomentum
			? actualAngularMomentum : target.angularMomentum;

		if(mode == AvbdSoftComponentFinalizeMode::ePOSITION_OWNED &&
			!preservePositionDerivedMomentum)
		{
			PxVec3 targetAbsoluteAngularMomentum =
				target.angularMomentum +
					target.centroid.cross(target.linearMomentum);
			const PxReal dt = 1.0f / invDt;
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleEnd =
				particleStart + body.compiled.particleCount;
			for(PxU32 contactIndex = 0;
				contactIndex < numContacts; ++contactIndex)
			{
				const AvbdSoftContact& contact = contacts[contactIndex];
				const AvbdSoftContactGeometry& geometry = contact.geometry;
				if(geometry.velocityOwner !=
					AvbdVelocityObjectiveOwner::PositionAL ||
					contact.state.alLambda >= 0.0f ||
					geometry.hasWorldStaticTarget())
					continue;
				const PxVec3 contactForce =
					geometry.normal * (-contact.state.alLambda) -
					geometry.tangent1 *
						contact.state.alLambdaTangent[0] -
					geometry.tangent2 *
						contact.state.alLambdaTangent[1];
				if(!contactForce.isFinite())
					continue;
				PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleCount =
					avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				for(PxU32 supportIndex = 0;
					supportIndex < particleCount; ++supportIndex)
				{
					const PxU32 particleIndex =
						particleIndices[supportIndex];
					if(particleIndex < particleStart ||
						particleIndex >= particleEnd ||
						particleIndex >= numParticles)
						continue;
					const PxReal jacobianScale =
						avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex);
					if(PxAbs(jacobianScale) <= 1.0e-12f)
						continue;
					const PxVec3 impulse =
						contactForce * (jacobianScale * dt);
					targetLinearMomentum += impulse;
					targetAbsoluteAngularMomentum +=
						particles[particleIndex].position.cross(impulse);
				}
			}
			targetAngularMomentum =
				targetAbsoluteAngularMomentum -
				centroid.cross(targetLinearMomentum);

			// Position AL owns geometric non-penetration, but its accumulated
			// multiplier is not a discrete material impulse. Replaying that
			// multiplier against a world-static surface can overshoot the
			// inelastic velocity boundary every frame and pump both translation
			// and rotation. Rebuild the static response from the damped inertial
			// component momentum instead: each sequential row applies exactly the
			// non-negative impulse needed to remove inward point velocity, followed
			// by a Coulomb-bounded tangent impulse. This retains impact torque while
			// preventing a resting contact from creating separating kinetic energy.
			for(PxU32 contactIndex = 0;
				contactIndex < numContacts; ++contactIndex)
			{
				const AvbdSoftContact& contact = contacts[contactIndex];
				const AvbdSoftContactGeometry& geometry = contact.geometry;
				if(geometry.velocityOwner !=
						AvbdVelocityObjectiveOwner::PositionAL ||
					contact.state.alLambda >= 0.0f ||
					!geometry.hasWorldStaticTarget())
					continue;

				PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleCount =
					avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				bool belongsToBody = false;
				for(PxU32 supportIndex = 0;
					supportIndex < particleCount; ++supportIndex)
				{
					const PxU32 particleIndex =
						particleIndices[supportIndex];
					if(particleIndex >= particleStart &&
						particleIndex < particleEnd &&
						particleIndex < numParticles &&
						PxAbs(avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex)) > 1.0e-12f)
					{
						belongsToBody = true;
						break;
					}
				}
				if(!belongsToBody)
					continue;

				const PxVec3 normal = geometry.normal;
				const PxVec3 queryPoint =
					avbdGetSoftContactQueryPoint(geometry, particles);
				if(!normal.isFinite() || !queryPoint.isFinite())
					continue;
				const PxVec3 offset = queryPoint - centroid;
				const PxVec3 angularMomentum =
					targetAbsoluteAngularMomentum -
					centroid.cross(targetLinearMomentum);
				const PxVec3 angularVelocity = hasAngularResponse
					? inverseInertia * angularMomentum
					: PxVec3(0.0f);
				const PxVec3 linearVelocity =
					targetLinearMomentum * (1.0f / mass);
				const PxVec3 normalAngularJacobian =
					offset.cross(normal);
				const PxReal normalResponse =
					1.0f / mass +
					(hasAngularResponse
						? normalAngularJacobian.dot(
							inverseInertia * normalAngularJacobian)
						: 0.0f);
				const PxReal relativeNormalVelocity =
					(linearVelocity + angularVelocity.cross(offset)).
						dot(normal);
				if(normalResponse <= 1.0e-12f ||
					!PxIsFinite(relativeNormalVelocity) ||
					relativeNormalVelocity >= 0.0f)
					continue;

				const PxReal normalImpulseMagnitude =
					-relativeNormalVelocity / normalResponse;
				const PxVec3 normalImpulse =
					normal * normalImpulseMagnitude;
				targetLinearMomentum += normalImpulse;
				targetAbsoluteAngularMomentum +=
					queryPoint.cross(normalImpulse);

				const PxReal frictionLimit =
					PxMax(geometry.friction, 0.0f) *
						normalImpulseMagnitude;
				if(frictionLimit <= 0.0f)
					continue;
				const PxVec3 postNormalAngularMomentum =
					targetAbsoluteAngularMomentum -
					centroid.cross(targetLinearMomentum);
				const PxVec3 postNormalAngularVelocity =
					hasAngularResponse
						? inverseInertia * postNormalAngularMomentum
						: PxVec3(0.0f);
				const PxVec3 postNormalLinearVelocity =
					targetLinearMomentum * (1.0f / mass);
				const PxVec3 pointVelocity =
					postNormalLinearVelocity +
					postNormalAngularVelocity.cross(offset);
				const PxVec3 tangentAngularJacobian0 =
					offset.cross(geometry.tangent1);
				const PxVec3 tangentAngularJacobian1 =
					offset.cross(geometry.tangent2);
				const PxReal response00 =
					1.0f / mass +
					(hasAngularResponse
						? tangentAngularJacobian0.dot(
							inverseInertia * tangentAngularJacobian0)
						: 0.0f);
				const PxReal response11 =
					1.0f / mass +
					(hasAngularResponse
						? tangentAngularJacobian1.dot(
							inverseInertia * tangentAngularJacobian1)
						: 0.0f);
				const PxReal response01 = hasAngularResponse
					? tangentAngularJacobian0.dot(
						inverseInertia * tangentAngularJacobian1)
					: 0.0f;
				const PxReal determinant =
					response00 * response11 - response01 * response01;
				if(!PxIsFinite(determinant) ||
					PxAbs(determinant) <= 1.0e-12f)
					continue;
				const PxReal rhs0 =
					-pointVelocity.dot(geometry.tangent1);
				const PxReal rhs1 =
					-pointVelocity.dot(geometry.tangent2);
				PxReal tangentImpulse0 =
					(response11 * rhs0 - response01 * rhs1) /
						determinant;
				PxReal tangentImpulse1 =
					(response00 * rhs1 - response01 * rhs0) /
						determinant;
				const PxReal tangentMagnitude = PxSqrt(
					tangentImpulse0 * tangentImpulse0 +
					tangentImpulse1 * tangentImpulse1);
				if(tangentMagnitude > frictionLimit &&
					tangentMagnitude > 1.0e-12f)
				{
					const PxReal scale =
						frictionLimit / tangentMagnitude;
					tangentImpulse0 *= scale;
					tangentImpulse1 *= scale;
				}
				const PxVec3 tangentImpulse =
					geometry.tangent1 * tangentImpulse0 +
					geometry.tangent2 * tangentImpulse1;
				if(!tangentImpulse.isFinite())
					continue;
				targetLinearMomentum += tangentImpulse;
				targetAbsoluteAngularMomentum +=
					queryPoint.cross(tangentImpulse);
			}
			targetAngularMomentum =
				targetAbsoluteAngularMomentum -
				centroid.cross(targetLinearMomentum);
		}
		else if(preservePositionDerivedMomentum &&
			body.compiled.speculativeCCDEnabled)
		{
			// A uniform component correction preserves all relative particle
			// velocities and therefore cannot distort volume. It only removes
			// inward normal speed at active static/kinematic swept contacts.
			PxVec3 linearVelocityCorrection(0.0f);
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleEnd =
				particleStart + body.compiled.particleCount;
			for(PxU32 contactIndex = 0;
				contactIndex < numContacts; ++contactIndex)
			{
				const AvbdSoftContact& contact = contacts[contactIndex];
				const AvbdSoftContactGeometry& geometry = contact.geometry;
				if(geometry.velocityOwner !=
						AvbdVelocityObjectiveOwner::PositionAL ||
					contact.state.alLambda >= 0.0f ||
					(!geometry.hasWorldStaticTarget() &&
					 !geometry.hasKinematicRigidTarget()))
					continue;
				PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleCount =
					avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				PxVec3 queryVelocity(0.0f);
				bool belongsToBody = false;
				for(PxU32 supportIndex = 0;
					supportIndex < particleCount; ++supportIndex)
				{
					const PxU32 particleIndex =
						particleIndices[supportIndex];
					if(particleIndex < particleStart ||
						particleIndex >= particleEnd ||
						particleIndex >= numParticles)
						continue;
					const PxReal jacobianScale =
						avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex);
					if(PxAbs(jacobianScale) <= 1.0e-12f)
						continue;
					belongsToBody = true;
					queryVelocity +=
						particles[particleIndex].velocity *
							jacobianScale;
				}
				if(!belongsToBody)
					continue;
				const PxVec3 surfaceVelocity =
					geometry.hasKinematicRigidTarget()
						? (geometry.surfacePoint -
						   geometry.kinematicSurfacePointPrevious) *
							invDt
						: PxVec3(0.0f);
				const PxReal relativeNormalVelocity =
					(queryVelocity + linearVelocityCorrection -
					 surfaceVelocity).dot(geometry.normal);
				if(relativeNormalVelocity < 0.0f &&
					PxIsFinite(relativeNormalVelocity))
					linearVelocityCorrection +=
						geometry.normal * (-relativeNormalVelocity);
			}
			if(linearVelocityCorrection.isFinite())
				targetLinearMomentum +=
					linearVelocityCorrection * mass;
		}

		if(mode ==
			AvbdSoftComponentFinalizeMode::eKINEMATIC_CONTACT)
		{
			for(PxU32 objectiveIndex = 0;
				objectiveIndex < numVelocityObjectives;
				objectiveIndex++)
			{
				const AvbdCompiledSoftVelocityObjective& objective =
					velocityObjectives[objectiveIndex];
				if(objective.owner !=
						AvbdVelocityObjectiveOwner::
							ComponentFinalize ||
					objective.bodyIndex != bodyIndex ||
					objective.particleIndex <
						body.compiled.particleStart ||
					objective.particleIndex >=
						body.compiled.particleStart +
							body.compiled.particleCount)
					continue;
				const PxVec3 normal = objective.normal;
				PxVec3 queryPoint =
					particles[objective.particleIndex].position;
				if(objective.queryPoint.count != 0)
				{
					queryPoint = PxVec3(0.0f);
					for(PxU32 queryVertex = 0;
						queryVertex < objective.queryPoint.count;
						queryVertex++)
					{
						const PxU32 queryParticle =
							objective.queryPoint.particleIndices[
								queryVertex];
						queryPoint +=
							particles[queryParticle].position *
							objective.queryPoint.weights[queryVertex];
					}
				}
				const PxVec3 offset = queryPoint - centroid;
				const PxVec3 targetLinearVelocity =
					targetLinearMomentum * (1.0f / mass);
				const PxVec3 targetAngularVelocity =
					hasAngularResponse
						? inverseInertia * targetAngularMomentum
						: PxVec3(0.0f);
				const PxVec3 surfaceVelocity =
					(objective.surfacePoint -
					 objective.previousSurfacePoint) *
						invDt;
				const PxReal relativeNormalVelocity =
					(targetLinearVelocity +
					 targetAngularVelocity.cross(offset) -
					 surfaceVelocity).dot(normal);
				const PxVec3 angularJacobian =
					offset.cross(normal);
				const PxReal response =
					1.0f / mass +
					(hasAngularResponse
						? angularJacobian.dot(
							inverseInertia * angularJacobian)
						: 0.0f);
				if(response <= 1.0e-12f ||
					!PxIsFinite(relativeNormalVelocity))
					continue;
				// Position AL already owns penetration.  This typed
				// component owner supplies only the e=0 velocity boundary
				// at the prescribed surface; it does not iterate impulses.
				const PxReal correction =
					-relativeNormalVelocity / response;
				const PxVec3 momentumDelta = normal * correction;
				targetLinearMomentum += momentumDelta;
				targetAngularMomentum +=
					offset.cross(momentumDelta);
			}
		}

		const PxVec3 actualLinearVelocity =
			actualLinearMomentum * (1.0f / mass);
		const PxVec3 targetLinearVelocity =
			targetLinearMomentum * (1.0f / mass);
		const PxVec3 actualAngularVelocity =
			hasAngularResponse
				? inverseInertia * actualAngularMomentum
				: PxVec3(0.0f);
		const PxVec3 targetAngularVelocity =
			hasAngularResponse
				? inverseInertia * targetAngularMomentum
				: actualAngularVelocity;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount;
			localIndex++)
		{
			AvbdSoftParticle& particle =
				particles[body.compiled.particleStart + localIndex];
			if(particle.invMass <= 0.0f)
				continue;
			const PxVec3 offset = particle.position - centroid;
			particle.velocity +=
				(targetLinearVelocity +
				 targetAngularVelocity.cross(offset)) -
				(actualLinearVelocity +
				 actualAngularVelocity.cross(offset));
			if(!particle.velocity.isFinite())
			{
				PX_ASSERT(false);
					particle.velocity = particle.prevVelocity;
			}
		}
	}
	}

void avbdUpdateSoftContactDual(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	avbdUpdateSoftContactDualAtSurfacePoint(
		geometry, state, particles,
		avbdGetSoftContactSurfacePoint(geometry, particles), beta);
}

} // namespace Dy
} // namespace physx
