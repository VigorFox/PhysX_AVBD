// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactEpoch.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"

namespace physx
{
namespace Dy
{

// AVBD contact epoch/state-transfer and all-contact orchestration helpers.
//
// This unit owns query-point reconstruction, warm-start state transfer, and
// component OGC safety-epoch checks. Geometry-specific detectors remain in
// their dedicated units; the complete detector wrapper remains separate.
// =============================================================================

// =============================================================================
// Convenience: detect all OGC contacts (ground + soft-rigid + soft-soft + self)
// =============================================================================

void avbdResetSoftContactDepenetrationLimits(
	AvbdSoftContact* contacts, PxU32 numContacts)
{
	for(PxU32 contactIndex = 0;
		contactIndex < numContacts; contactIndex++)
	{
		contacts[contactIndex].state.
			depenetrationConstraintOffset = 0.0f;
		contacts[contactIndex].state.
			depenetrationLimitInitialized = false;
	}
}

void avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
	AvbdSoftContact& contact,
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxVec3& initialSurfacePoint, PxReal dt)
{
	AvbdSoftContactAugmentedState& state = contact.state;
	if(state.depenetrationLimitInitialized)
		return;

	const AvbdSoftContactGeometry& geometry = contact.geometry;
	const PxU32 queryRepresentative = geometry.hasWeightedQueryPoint()
		? geometry.queryPoint.particleIndices[0]
		: geometry.particleIdx;
	const AvbdSoftBody* queryBody = geometry.queryBodyIndex < numSoftBodies
		? &softBodies[geometry.queryBodyIndex]
		: avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, queryRepresentative);
	PxReal maxDepenetrationVelocity = queryBody
		? queryBody->compiled.maxDepenetrationVelocity
		: PX_MAX_F32;
	if(geometry.hasDeformableSurfaceTarget())
	{
		const PxU32 targetRepresentative =
			geometry.hasWeightedTargetPoint()
				? geometry.targetPoint.particleIndices[0]
				: geometry.surfaceParticleIndices[0];
		const AvbdSoftBody* targetBody =
			geometry.source.targetBodyIndex < numSoftBodies
				? &softBodies[geometry.source.targetBodyIndex]
				: avbdFindSoftBodyForParticle(
					softBodies, numSoftBodies, targetRepresentative);
		if(targetBody)
			maxDepenetrationVelocity = PxMin(
				maxDepenetrationVelocity,
				targetBody->compiled.maxDepenetrationVelocity);
	}
	maxDepenetrationVelocity =
		PxMax(maxDepenetrationVelocity, 0.0f);
	state.depenetrationConstraintOffset = 0.0f;
	state.depenetrationLimitInitialized = true;
	if(dt <= 0.0f ||
		maxDepenetrationVelocity >= 1.0e20f)
		return;

	const PxVec3 initialQueryPoint =
		avbdGetSoftContactInitialQueryPoint(
			geometry, particles);
	const PxReal initialConstraint =
		avbdEvaluateSoftContactNormalConstraint(
			geometry, initialQueryPoint, initialSurfacePoint);
	const PxReal maxRecoveryDistance =
		maxDepenetrationVelocity * dt;
	state.depenetrationConstraintOffset =
		PxMin(0.0f, initialConstraint + maxRecoveryDistance);
	if(state.depenetrationConstraintOffset < 0.0f)
	{
		// A carried normal multiplier can otherwise spend the new frame's
		// finite bias budget before the shifted row has converged.
		state.alLambda = 0.0f;
	}
}

void avbdInitializeSoftContactDepenetrationLimits(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxReal dt)
{
	for(PxU32 contactIndex = 0;
		contactIndex < numContacts; contactIndex++)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
			contact, particles, softBodies, numSoftBodies,
			avbdGetSoftContactInitialSurfacePoint(
				contact.geometry, particles),
			dt);
	}
}

static bool avbdGetSoftContactDetectionQueryPoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* detectionParticles,
	PxVec3& point)
{
	point = PxVec3(0.0f);
	if(geometry.queryParticleIndices[0] != PX_MAX_U32)
	{
		for(PxU32 i = 0; i < 3; ++i)
		{
			const PxU32 particleIndex = geometry.queryParticleIndices[i];
			if(particleIndex == PX_MAX_U32)
				break;
			point += detectionParticles[particleIndex].position *
				geometry.queryWeights[i];
		}
		return point.isFinite();
	}
	const PxU32 particleIndex =
		geometry.collisionFeatureParticleIdx != PX_MAX_U32
			? geometry.collisionFeatureParticleIdx
			: geometry.particleIdx;
	if(particleIndex == PX_MAX_U32)
		return false;
	point = detectionParticles[particleIndex].position;
	return point.isFinite();
}

static bool avbdGetSoftContactDetectionSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* detectionParticles,
	PxVec3& point)
{
	point = PxVec3(0.0f);
	if(!geometry.hasDeformableSurfaceTarget())
		return false;
	for(PxU32 i = 0; i < 3; ++i)
	{
		const PxU32 particleIndex = geometry.surfaceParticleIndices[i];
		if(particleIndex == PX_MAX_U32)
			break;
		point += detectionParticles[particleIndex].position *
			geometry.surfaceWeights[i];
	}
	return point.isFinite();
}

static bool avbdCanTransferSoftContactFrictionAnchors(
	const AvbdSoftContactGeometry& previousGeometry,
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* detectionParticles)
{
	if(!previousGeometry.hasDeformableSurfaceTarget() ||
		!geometry.hasDeformableSurfaceTarget())
		return true;

	PxVec3 previousQueryPoint, queryPoint;
	PxVec3 previousSurfacePoint, surfacePoint;
	if(!avbdGetSoftContactDetectionQueryPoint(
			previousGeometry, detectionParticles, previousQueryPoint) ||
		!avbdGetSoftContactDetectionQueryPoint(
			geometry, detectionParticles, queryPoint) ||
		!avbdGetSoftContactDetectionSurfacePoint(
			previousGeometry, detectionParticles, previousSurfacePoint) ||
		!avbdGetSoftContactDetectionSurfacePoint(
			geometry, detectionParticles, surfacePoint))
		return false;

	// A collision feature can remain identical while the closest point moves
	// along a long edge or across a face. The normal multiplier still belongs
	// to that feature, but carrying its static-friction anchor to the new
	// material points creates an artificial tangential tether. One contact-shell
	// radius is the largest migration for which the old friction patch remains
	// local. Detection-domain supports are used deliberately: public Volume
	// contacts have already-expanded simulation points in queryPoint/targetPoint,
	// while these legacy support arrays retain the authoritative proxy feature.
	const PxReal anchorRadius = PxMax(
		PxMax(previousGeometry.margin, geometry.margin), 1.0e-4f);
	const PxReal anchorRadiusSq = anchorRadius * anchorRadius;
	return (previousQueryPoint - queryPoint).magnitudeSquared() <=
			anchorRadiusSq &&
		(previousSurfacePoint - surfacePoint).magnitudeSquared() <=
			anchorRadiusSq;
}

static bool avbdCanTransferSoftContactNormalState(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
	const PxVec3 queryPoint =
		avbdGetSoftContactQueryPoint(geometry, particles);
	const PxVec3 surfacePoint =
		avbdGetSoftContactSurfacePoint(geometry, particles);
	if(!queryPoint.isFinite() || !surfacePoint.isFinite() ||
		!geometry.normal.isFinite())
		return false;

	const PxReal constraint = avbdEvaluateSoftContactNormalConstraint(
		geometry, queryPoint, surfacePoint);
	if(!PxIsFinite(constraint))
		return false;

	// Contact detection deliberately retains rows in the proximity shell.
	// A unilateral row that has separated inside that shell is no longer the
	// same active normal objective, even when its feature identity is stable.
	// Carrying its negative multiplier and elevated AL penalty across the
	// flight phase preloads the next impact and can freeze or launch the body.
	// Keep a tiny tolerance at the boundary to preserve useful warm-starting
	// for a genuinely active resting contact.
	const PxReal activeTolerance = PxMax(
		1.0e-5f, 1.0e-4f * PxMax(geometry.margin, 0.0f));
	return constraint <= activeTolerance;
}

void avbdTransferSoftContactState(
	const AvbdSoftContact* previousContacts, PxU32 numPreviousContacts,
	const AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace* persistentWorkspace)
{
	const PxReal normalMatch = 0.8f;
	const PxReal pointMatchSq = 0.05f * 0.05f;
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	workspace.resizePreviousUsed(numPreviousContacts);
	PxArray<PxU8>& previousUsed = workspace.epoch.previousUsed;
	for(PxU32 oldIdx = 0; oldIdx < numPreviousContacts; ++oldIdx)
		previousUsed[oldIdx] = 0;
	for(PxU32 contactIdx = 0; contactIdx < contacts.size(); ++contactIdx)
	{
		AvbdSoftContact& contact = contacts[contactIdx];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		AvbdSoftContactAugmentedState& state = contact.state;
		avbdInitializeSoftContactAnchors(
			geometry, state, particles);

		const AvbdSoftContact* best = NULL;
		PxU32 bestIdx = PX_MAX_U32;
		PxReal bestDistanceSq = PX_MAX_F32;
		for(PxU32 oldIdx = 0; oldIdx < numPreviousContacts; ++oldIdx)
		{
			if(previousUsed[oldIdx])
				continue;
			const AvbdSoftContact& old = previousContacts[oldIdx];
			const AvbdSoftContactGeometry& oldGeometry = old.geometry;
			const PxU32 oldFeatureParticle =
				oldGeometry.collisionFeatureParticleIdx != PX_MAX_U32
					? oldGeometry.collisionFeatureParticleIdx
					: oldGeometry.particleIdx;
			const PxU32 newFeatureParticle =
				geometry.collisionFeatureParticleIdx != PX_MAX_U32
					? geometry.collisionFeatureParticleIdx
					: geometry.particleIdx;
			if(oldFeatureParticle != newFeatureParticle ||
				oldGeometry.normal.dot(geometry.normal) < normalMatch)
				continue;

			const PxReal distanceSq =
				(oldGeometry.surfacePoint -
				 geometry.surfacePoint).magnitudeSquared();
			if(geometry.source.isValid() || oldGeometry.source.isValid())
			{
				if(!geometry.source.isValid() ||
					!oldGeometry.source.isValid() ||
					!(geometry.source == oldGeometry.source))
					continue;
			}
			else
			{
				// Compatibility path for manually authored legacy contacts.
				if(oldGeometry.targetKind != geometry.targetKind ||
					oldGeometry.targetIndex != geometry.targetIndex)
					continue;
				if(!geometry.hasWorldStaticTarget() &&
					distanceSq > pointMatchSq)
					continue;
			}
			if(distanceSq < bestDistanceSq)
			{
				best = &old;
				bestIdx = oldIdx;
				bestDistanceSq = distanceSq;
			}
		}
		if(!best)
			continue;
		if(!avbdCanTransferSoftContactNormalState(geometry, particles))
			continue;
		previousUsed[bestIdx] = 1;

		const PxReal dualDecay = 0.99f;
		const PxReal penaltyDecay = 0.999f;
		const AvbdSoftContactGeometry& bestGeometry = best->geometry;
		const AvbdSoftContactAugmentedState& bestState = best->state;
		state.alLambda = bestState.alLambda * dualDecay;
		state.k = PxClamp(
			bestState.k * penaltyDecay, state.k, state.ke);
		state.depenetrationConstraintOffset =
			bestState.depenetrationConstraintOffset;
		state.depenetrationLimitInitialized =
			bestState.depenetrationLimitInitialized;
		if(bestState.frictionStick &&
			!avbdCanTransferSoftContactFrictionAnchors(
				bestGeometry, geometry, particles))
			continue;

		state.penTangent[0] = PxClamp(
			bestState.penTangent[0] * penaltyDecay,
			1000.0f, state.ke);
		state.penTangent[1] = PxClamp(
			bestState.penTangent[1] * penaltyDecay,
			1000.0f, state.ke);

		const PxVec3 oldTangentForce =
			bestGeometry.tangent1 * bestState.alLambdaTangent[0] +
			bestGeometry.tangent2 * bestState.alLambdaTangent[1];
		state.alLambdaTangent[0] =
			oldTangentForce.dot(geometry.tangent1) * dualDecay;
		state.alLambdaTangent[1] =
			oldTangentForce.dot(geometry.tangent2) * dualDecay;
		state.frictionStick = bestState.frictionStick;
		if(bestState.frictionStick)
		{
			state.particlePointPrev = bestState.particlePointPrev;
			state.surfacePointPrev = bestState.surfacePointPrev;
		}
	}
}

void avbdBuildSoftContactRedetectionPhasePlan(
	AvbdSoftContactWorkspace& workspace,
	PxU32 numWorldPlanes, bool includeLegacyGround,
	PxU32 numRigidBoxes, PxU32 numRigidSpheres,
	PxU32 numRigidCapsules, PxU32 numRigidConvexes,
	PxU32 numRigidTriangleSurfaces, PxU32 numSoftBodies,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	const PxU8* selfCollisionEnabled)
{
	workspace.beginRedetectionPhasePlan();
	if(numWorldPlanes > 0)
	{
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eWORLD_PLANES,
			0, numWorldPlanes);
	}
	else if(includeLegacyGround)
	{
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eLEGACY_GROUND, 0, 1);
	}
	if(numRigidBoxes > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_BOXES,
			0, numRigidBoxes);
	if(numRigidSpheres > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_SPHERES,
			0, numRigidSpheres);
	if(numRigidCapsules > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_CAPSULES,
			0, numRigidCapsules);
	if(numRigidConvexes > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_CONVEXES,
			0, numRigidConvexes);
	if(numRigidTriangleSurfaces > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_TRIANGLE_SURFACES,
			0, numRigidTriangleSurfaces);
	if(numSoftBodies > 1)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eSOFT_SOFT,
			0, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		if(bodyIndex < numAdj && perBodyAdj &&
			(!selfCollisionEnabled || selfCollisionEnabled[bodyIndex]))
		{
			workspace.appendRedetectionPhasePlan(
				AvbdSoftContactRedetectionPhase::eSELF_BODY,
				bodyIndex, bodyIndex + 1);
		}
	}
}

// Parent-owned boundaries for one complete contact-redetection transaction.
// Detection leaves may populate private streams between these calls, but only
// this parent is allowed to snapshot prior state, mutate the canonical stream
// and transfer persistent contact state. This is deliberately independent of
// any particular detection source so a future Scene task fan-in can preserve
// the serial stream without exposing workspace mutation to workers.
void avbdBeginSoftContactRedetection(
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace& workspace,
	AvbdSoftCollisionStats* stats)
{
	workspace.copyPreviousContacts(contacts);
	contacts.clear();
	workspace.redetectionOutputCapacityBefore = contacts.capacity();
	if(stats)
		stats->detectionCalls++;
}

void avbdCompleteSoftContactRedetection(
	AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace& workspace)
{
	workspace.recordOutputCapacityGrowth(
		workspace.redetectionOutputCapacityBefore, contacts.capacity());
	workspace.recordOutputWatermark(contacts.size(), contacts.capacity());
	avbdTransferSoftContactState(
		workspace.epoch.previousContacts.begin(), workspace.epoch.previousContacts.size(),
		particles, contacts, &workspace);
	// A later outer iteration observes mutated primal positions, so it must not
	// reuse the post-prediction bounds prepared for this single redetection.
	workspace.invalidateSoftBodyBounds();
}

} // namespace Dy
} // namespace physx
