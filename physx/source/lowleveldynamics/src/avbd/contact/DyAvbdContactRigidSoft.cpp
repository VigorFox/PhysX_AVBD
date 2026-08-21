// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactMaterial.h"
#include "avbd/contact/DyAvbdContactRigidSoft.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/contact/DyAvbdContactRigidBoxGeometry.h"
#include "avbd/contact/DyAvbdContactMeshQueries.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"

namespace physx
{
namespace Dy
{

// =============================================================================
// OGC (Offset Geometric Contact) -- 4-Path Collision Detection
//
// Reference: "Offset Geometric Contact", SIGGRAPH 2025
//            Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel
//
// Path 1: Rigid-Rigid -> PhysX native broadphase/narrowphase
// Path 2: Rigid-Soft -> Analytical box SDF query
// Path 3: Soft-Soft -> OGC simplified (Sec 3.9: outward offset, pure quadratic)
// Path 4: Self-collision -> OGC full (safety bubble + two-stage C2 activation)
// =============================================================================
// =============================================================================
// PATH 2 (OGC): Analytical SDF Rigid-Soft Contact
// =============================================================================

PX_FORCE_INLINE void avbdUpdateClosestSegmentTriangle(
	AvbdClosestSegmentTriangleResult& result,
	const PxVec3& segmentPoint, const PxVec3& trianglePoint,
	const PxVec3& barycentric, PxReal segmentWeight1,
	AvbdClosestFeature feature, PxU32 featureIndex)
{
	const PxReal distance =
		(segmentPoint - trianglePoint).magnitude();
	if(PxIsFinite(distance) && distance < result.distance)
	{
		result.segmentPoint = segmentPoint;
		result.trianglePoint = trianglePoint;
		result.barycentric = barycentric;
		result.segmentWeight1 = segmentWeight1;
		result.distance = distance;
		result.feature = feature;
		result.featureIndex = featureIndex;
	}
}

// Complete segment/triangle closest query used by the capsule reverse OGC
// path. Endpoint/triangle candidates own cap features, segment/edge candidates
// own side features, and the explicit plane crossing covers a medial segment
// passing through a triangle interior.
PX_FORCE_INLINE AvbdClosestSegmentTriangleResult
avbdClosestSegmentTriangleOGC(
	const PxVec3& segment0, const PxVec3& segment1,
	const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	AvbdClosestSegmentTriangleResult result;
	result.segmentPoint = segment0;
	result.trianglePoint = a;
	result.barycentric = PxVec3(1.0f, 0.0f, 0.0f);
	result.segmentWeight1 = 0.0f;
	result.distance = PX_MAX_F32;
	result.feature = AVBD_FEATURE_UNKNOWN;
	result.featureIndex = 0;

	const PxVec3 segmentDirection = segment1 - segment0;
	const PxVec3 triangleNormal = (b - a).cross(c - a);
	const PxReal normalMagnitudeSq =
		triangleNormal.magnitudeSquared();
	const PxReal planeDenominator =
		triangleNormal.dot(segmentDirection);
	if(normalMagnitudeSq > 1.0e-16f &&
		PxAbs(planeDenominator) > 1.0e-12f)
	{
		const PxReal segmentWeight =
			triangleNormal.dot(a - segment0) / planeDenominator;
		if(segmentWeight >= 0.0f && segmentWeight <= 1.0f)
		{
			const PxVec3 planePoint =
				segment0 + segmentDirection * segmentWeight;
			const AvbdClosestPointResult planeClosest =
				avbdClosestPointOnTriangleOGC(
					planePoint, a, b, c);
			if(planeClosest.distance <= 1.0e-6f)
				avbdUpdateClosestSegmentTriangle(
					result, planePoint, planeClosest.point,
					planeClosest.barycentric, segmentWeight,
					planeClosest.feature,
					planeClosest.featureIndex);
		}
	}

	const PxVec3 endpoints[2] = {segment0, segment1};
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				endpoints[endpoint], a, b, c);
		avbdUpdateClosestSegmentTriangle(
			result, endpoints[endpoint], closest.point,
			closest.barycentric, PxReal(endpoint),
			closest.feature, closest.featureIndex);
	}

	const PxVec3 edge0[3] = {a, a, b};
	const PxVec3 edge1[3] = {b, c, c};
	for(PxU32 edge = 0; edge < 3; ++edge)
	{
		PxReal segmentWeight = 0.0f;
		PxReal edgeWeight = 0.0f;
		PxVec3 segmentClosest;
		PxVec3 edgeClosest;
		avbdClosestPointsOnSegments(
			segment0, segment1, edge0[edge], edge1[edge],
			segmentWeight, edgeWeight,
			segmentClosest, edgeClosest);
		PxVec3 barycentric(0.0f);
		if(edge == 0)
			barycentric = PxVec3(
				1.0f - edgeWeight, edgeWeight, 0.0f);
		else if(edge == 1)
			barycentric = PxVec3(
				1.0f - edgeWeight, 0.0f, edgeWeight);
		else
			barycentric = PxVec3(
				0.0f, 1.0f - edgeWeight, edgeWeight);
		AvbdClosestFeature feature = AVBD_FEATURE_EDGE;
		PxU32 featureIndex = edge;
		if(edgeWeight <= 1.0e-5f ||
			edgeWeight >= 1.0f - 1.0e-5f)
		{
			feature = AVBD_FEATURE_VERTEX;
			featureIndex = edgeWeight <= 1.0e-5f
				? (edge == 2 ? 1u : 0u)
				: (edge == 0 ? 1u : 2u);
		}
		avbdUpdateClosestSegmentTriangle(
			result, segmentClosest, edgeClosest,
			barycentric, segmentWeight,
			feature, featureIndex);
	}
	return result;
}

PX_FORCE_INLINE bool avbdFindPreviousRigidBoxFace(
	const AvbdSoftContact* previousContacts,
	PxU32 numPreviousContacts,
	PxU32 particleIndex,
	const AvbdRigidBox& box,
	PxVec3& localFaceNormal)
{
	for(PxU32 contactIndex = 0;
		contactIndex < numPreviousContacts; contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			previousContacts[contactIndex].geometry;
		if(geometry.particleIdx != particleIndex ||
			geometry.targetKind != box.targetKind ||
			geometry.source.type !=
				AvbdSoftContactSource::eRIGID_SDF ||
			geometry.source.primitiveKey != box.primitiveKey ||
			geometry.source.featureKey < 1u ||
			geometry.source.featureKey > 6u)
			continue;
		if(box.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY &&
			geometry.targetIndex != box.targetIndex)
			continue;
		const PxVec3 candidate =
			box.rotation.getConjugate().rotate(geometry.normal);
		if(!candidate.isFinite() ||
			candidate.magnitudeSquared() < 0.25f)
			continue;
		localFaceNormal = avbdGetRigidBoxFaceNormal(candidate);
		return true;
	}
	return false;
}

// P5.4a candidate leaf: a worker owns one canonical particle interval and
// only appends the current-pose box SDF contacts to its private stream. It
// reads the immutable previous-contact snapshot for inside-face continuity;
// swept SDF and OGC feature passes retain their existing parent order.
void avbdDetectSoftRigidSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftContact* previousContacts,
	PxU32 numPreviousContacts,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for (PxU32 pi = particleBegin; pi < particleEnd; pi++)
	{
		if (particles[pi].invMass <= 0.0f) continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, pi);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(*sourceBody, pi))
			continue;
		const PxVec3& pp = particles[pi].position;

		for (PxU32 bi = 0; bi < numBoxes; bi++)
		{
			const AvbdRigidBox& box = boxes[bi];
			PxVec3 he = box.halfExtent;
			if (he.x <= 0.0f && he.y <= 0.0f && he.z <= 0.0f) continue;

			// Broadphase AABB
			PxReal maxExt = PxSqrt(he.x*he.x + he.y*he.y + he.z*he.z) + margin;
			PxVec3 bMin = box.center - PxVec3(maxExt);
			PxVec3 bMax = box.center + PxVec3(maxExt);
			if (pp.x < bMin.x || pp.x > bMax.x ||
				pp.y < bMin.y || pp.y > bMax.y ||
				pp.z < bMin.z || pp.z > bMax.z) continue;

			PxVec3 localP = box.rotation.getConjugate().rotate(pp - box.center);

			// Analytical box SDF
			PxVec3 q(PxAbs(localP.x) - he.x,
			         PxAbs(localP.y) - he.y,
			         PxAbs(localP.z) - he.z);

			bool inside = (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f);
			PxReal sdf;
			PxVec3 localNormal;
			PxU64 featureKey = 0;

			if (inside) {
				if(avbdFindPreviousRigidBoxFace(
					previousContacts, numPreviousContacts,
					pi, box, localNormal))
				{
					const PxVec3 signedPosition(
						localNormal.x * localP.x,
						localNormal.y * localP.y,
						localNormal.z * localP.z);
					const PxVec3 selectedExtent(
						PxAbs(localNormal.x) * he.x,
						PxAbs(localNormal.y) * he.y,
						PxAbs(localNormal.z) * he.z);
					sdf =
						signedPosition.x + signedPosition.y +
						signedPosition.z -
						selectedExtent.x - selectedExtent.y -
						selectedExtent.z;
				}
				else
				{
					sdf = PxMax(q.x, PxMax(q.y, q.z));
					if (q.x > q.y && q.x > q.z)
						localNormal = PxVec3(
							localP.x > 0 ? 1.0f : -1.0f, 0, 0);
					else if (q.y > q.z)
						localNormal = PxVec3(
							0, localP.y > 0 ? 1.0f : -1.0f, 0);
					else
						localNormal = PxVec3(
							0, 0, localP.z > 0 ? 1.0f : -1.0f);
				}
				featureKey =
					avbdGetRigidBoxFaceFeatureKey(localNormal);
			} else {
				PxVec3 clamped(PxMax(q.x, 0.0f), PxMax(q.y, 0.0f), PxMax(q.z, 0.0f));
				sdf = clamped.magnitude();
				if (sdf > 1e-10f)
				{
					localNormal = PxVec3(
						(localP.x >= 0.0f ? 1.0f : -1.0f) * clamped.x,
						(localP.y >= 0.0f ? 1.0f : -1.0f) * clamped.y,
						(localP.z >= 0.0f ? 1.0f : -1.0f) * clamped.z) * (1.0f / sdf);
				}
				else
					localNormal = PxVec3(0, 1, 0);
				featureKey = avbdGetRigidBoxFaceFeatureKey(
					avbdGetRigidBoxFaceNormal(localNormal));
			}

			if (sdf >= margin) continue;

			PxReal depth = inside ? -sdf : PxMax(0.0f, margin - sdf);
			PxVec3 worldNormal = box.rotation.rotate(localNormal).getNormalized();

			// Surface point on box
			PxVec3 surfaceLocal = localP;
			if (inside)
				surfaceLocal = localP - localNormal * sdf;
			else
			{
				surfaceLocal.x = PxClamp(localP.x, -he.x, he.x);
				surfaceLocal.y = PxClamp(localP.y, -he.y, he.y);
				surfaceLocal.z = PxClamp(localP.z, -he.z, he.z);
			}
			PxVec3 worldSurf = box.center + box.rotation.rotate(surfaceLocal);

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF, PX_MAX_U32,
				box.primitiveKey, featureKey);
			geometry.particleIdx  = pi;
			geometry.targetKind = box.targetKind;
			geometry.velocityOwner =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? AvbdVelocityObjectiveOwner::
						ComponentFinalize
					: box.targetKind ==
						AvbdSoftContactTargetKind::eRIGID_BODY
						? AvbdVelocityObjectiveOwner::
							ManifoldFinalize
						: AvbdVelocityObjectiveOwner::
							PositionAL;
			geometry.targetIndex =
				box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
				? box.targetIndex : bi;
			geometry.normal       = worldNormal;
			geometry.projNormal   = worldNormal;
			geometry.depth        = depth;
			geometry.margin       = margin;
			geometry.surfacePoint = worldSurf;
			geometry.kinematicSurfacePointPrevious =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
				? box.previousCenter +
					box.previousRotation.rotate(surfaceLocal)
				: worldSurf;
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					box.friction, box.frictionCombineMode)
				: PxMax(box.friction, 0.0f);
			if(box.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY)
			{
				geometry.rigidLocalPoint =
					box.shapeToRigidBody.transform(surfaceLocal);
			}
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f, particles, contacts);
		}
	}
}

void avbdDetectSoftRigidSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftContact* previousContacts,
	PxU32 numPreviousContacts,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	avbdDetectSoftRigidSDFRange(
		particles, numParticles, 0, numParticles, boxes, numBoxes,
		contacts, margin, previousContacts, numPreviousContacts,
		softBodies, numSoftBodies);
}

// P5.12a candidate leaf: swept box SDF is particle-major and has no mutable
// query state. A child owns one particle interval and appends only to its
// private stream. The caller must complete the current-SDF family before it
// stable-merges any swept-SDF family ranges.
void avbdDetectSoftRigidSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; particleIndex++)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.predictedPosition.isFinite())
			continue;
		const PxVec3 displacement =
			particle.predictedPosition - particle.position;
		if(displacement.magnitudeSquared() <= 1e-12f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!sourceBody->compiled.speculativeCCDEnabled)
			continue;
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 boxIndex = 0; boxIndex < numBoxes; boxIndex++)
		{
			const AvbdRigidBox& box = boxes[boxIndex];
			const PxQuat inverseRotation =
				box.rotation.getConjugate();
			const PxVec3 startLocal = inverseRotation.rotate(
				particle.position - box.center);
			const PxVec3 predictedLocal = inverseRotation.rotate(
				particle.predictedPosition - box.center);

			const PxVec3 currentQ(
				PxAbs(startLocal.x) - box.halfExtent.x,
				PxAbs(startLocal.y) - box.halfExtent.y,
				PxAbs(startLocal.z) - box.halfExtent.z);
			const bool currentInside =
				currentQ.x <= 0.0f &&
				currentQ.y <= 0.0f &&
				currentQ.z <= 0.0f;
			const PxReal currentSdf = currentInside
				? PxMax(currentQ.x, PxMax(currentQ.y, currentQ.z))
				: PxVec3(
					PxMax(currentQ.x, 0.0f),
					PxMax(currentQ.y, 0.0f),
					PxMax(currentQ.z, 0.0f)).magnitude();
			if(currentSdf < margin)
				continue;

			PxReal entryTime = 0.0f;
			PxVec3 entryNormalLocal(0.0f);
			if(!avbdSegmentEnterExpandedBox(
					startLocal, predictedLocal,
					box.halfExtent + PxVec3(margin),
					entryTime, entryNormalLocal))
				continue;

			const PxVec3 expandedEntryLocal =
				startLocal +
				(predictedLocal - startLocal) * entryTime;
			PxVec3 surfaceLocal =
				expandedEntryLocal - entryNormalLocal * margin;
			surfaceLocal.x = PxClamp(
				surfaceLocal.x,
				-box.halfExtent.x, box.halfExtent.x);
			surfaceLocal.y = PxClamp(
				surfaceLocal.y,
				-box.halfExtent.y, box.halfExtent.y);
			surfaceLocal.z = PxClamp(
				surfaceLocal.z,
				-box.halfExtent.z, box.halfExtent.z);

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, box.primitiveKey,
				avbdGetRigidBoxFaceFeatureKey(entryNormalLocal));
			geometry.particleIdx = particleIndex;
			geometry.targetKind = box.targetKind;
			geometry.velocityOwner =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? AvbdVelocityObjectiveOwner::ComponentFinalize
					: box.targetKind ==
						AvbdSoftContactTargetKind::eRIGID_BODY
						? AvbdVelocityObjectiveOwner::ManifoldFinalize
						: AvbdVelocityObjectiveOwner::PositionAL;
			geometry.targetIndex =
				box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
					? box.targetIndex : boxIndex;
			geometry.normal =
				box.rotation.rotate(entryNormalLocal).getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			geometry.surfacePoint =
				box.center + box.rotation.rotate(surfaceLocal);
			geometry.kinematicSurfacePointPrevious =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? box.previousCenter +
						box.previousRotation.rotate(surfaceLocal)
					: geometry.surfacePoint;
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					box.friction, box.frictionCombineMode)
				: PxMax(box.friction, 0.0f);
			if(box.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY)
				geometry.rigidLocalPoint =
					box.shapeToRigidBody.transform(surfaceLocal);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

void avbdDetectSoftRigidSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	avbdDetectSoftRigidSweptSDFRange(
		particles, numParticles, 0, numParticles, boxes, numBoxes, contacts,
		margin, softBodies, numSoftBodies);
}

PX_FORCE_INLINE PxVec3 avbdGetRigidSphereNormal(
	const PxVec3& offset, const AvbdSoftParticle& particle)
{
	const PxReal offsetMagnitudeSq = offset.magnitudeSquared();
	if(offsetMagnitudeSq > 1e-12f && PxIsFinite(offsetMagnitudeSq))
		return offset * PxRecipSqrt(offsetMagnitudeSq);
	const PxVec3 initialOffset =
		particle.initialPosition - particle.position;
	const PxReal initialMagnitudeSq =
		initialOffset.magnitudeSquared();
	if(initialMagnitudeSq > 1e-12f &&
		PxIsFinite(initialMagnitudeSq))
		return initialOffset * PxRecipSqrt(initialMagnitudeSq);
	return PxVec3(0.0f, 1.0f, 0.0f);
}

PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidSphere& sphere, PxU32 sphereIndex,
	const PxVec3& surfaceLocal)
{
	geometry.targetKind = sphere.targetKind;
	geometry.velocityOwner =
		sphere.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: sphere.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY
				? AvbdVelocityObjectiveOwner::ManifoldFinalize
				: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex =
		sphere.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY
			? sphere.targetIndex : sphereIndex;
	geometry.surfacePoint =
		sphere.center + sphere.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		sphere.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? sphere.previousCenter +
				sphere.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
	if(sphere.targetKind ==
		AvbdSoftContactTargetKind::eRIGID_BODY)
		geometry.rigidLocalPoint =
			sphere.shapeToRigidBody.transform(surfaceLocal);
}

// P5.5a candidate leaf: one worker owns a canonical particle interval and
// appends only the current-pose sphere SDF contacts to its private stream.
// Swept SDF and feature passes retain their parent order, exactly as the
// static-box transaction does.
void avbdDetectSoftRigidSphereSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite())
				continue;
			const PxVec3 offset =
				particle.position - sphere.center;
			const PxReal distanceSq = offset.magnitudeSquared();
			const PxReal queryRadius = sphere.radius + margin;
			if(!PxIsFinite(distanceSq) ||
				distanceSq >= queryRadius * queryRadius)
				continue;
			const PxReal distance = PxSqrt(PxMax(distanceSq, 0.0f));
			const PxVec3 normal =
				avbdGetRigidSphereNormal(offset, particle);
			const PxReal sdf = distance - sphere.radius;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, sphere.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = sdf < 0.0f
				? -sdf : PxMax(0.0f, margin - sdf);
			geometry.margin = margin;
			const PxVec3 surfaceLocal =
				sphere.rotation.getConjugate().rotate(
					normal * sphere.radius);
			avbdConfigureRigidSphereTarget(
				geometry, sphere, sphereIndex, surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					sphere.friction,
					sphere.frictionCombineMode)
				: PxMax(sphere.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

// P5.13a candidate leaf: swept sphere SDF is particle-major and retains no
// mutable query state. A caller that partitions it must complete the entire
// current-SDF family before stable-merging any swept-family private ranges.
void avbdDetectSoftRigidSphereSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;
		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			const bool dynamicTarget =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite() ||
				!sphere.rotation.isFinite() ||
				(sphere.targetKind !=
						AvbdSoftContactTargetKind::eWORLD_STATIC &&
				 sphere.targetKind !=
						AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				 !dynamicTarget) ||
				(dynamicTarget &&
				 (!sphere.predictedPoseValid ||
				  !sphere.predictedCenter.isFinite() ||
				  !sphere.predictedRotation.isFinite())))
				continue;
			const PxVec3 sphereCenterStart =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? sphere.previousCenter : sphere.center;
			const PxVec3 sphereCenterEnd =
				dynamicTarget
					? sphere.predictedCenter : sphere.center;
			if(!sphereCenterStart.isFinite() ||
				(sphere.targetKind ==
						AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				 !sphere.previousRotation.isFinite()))
				continue;
			// A moving sphere is swept in relative coordinates.  This keeps
			// a prescribed target that crosses a stationary soft vertex from
			// being discarded merely because the soft displacement is zero.
			const PxVec3 relativeStart =
				particle.position - sphereCenterStart;
			const PxVec3 relativeEnd =
				particle.predictedPosition - sphereCenterEnd;
			const PxVec3 relativeDisplacement =
				relativeEnd - relativeStart;
			if(relativeDisplacement.magnitudeSquared() <= 1e-12f)
				continue;
			const PxReal currentSdf =
				relativeStart.magnitude() - sphere.radius;
			if(!PxIsFinite(currentSdf) || currentSdf < margin)
				continue;

			PxReal entryTime = 0.0f;
			PxVec3 entryNormal(0.0f);
			if(!avbdSegmentEnterExpandedSphere(
					relativeStart,
					relativeEnd,
					PxVec3(0.0f),
					sphere.radius + margin,
					entryTime, entryNormal))
				continue;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, sphere.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = entryNormal;
			geometry.projNormal = entryNormal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			const PxVec3 surfaceLocal =
				sphere.rotation.getConjugate().rotate(
					entryNormal * sphere.radius);
			avbdConfigureRigidSphereTarget(
				geometry, sphere, sphereIndex, surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					sphere.friction,
					sphere.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e6f, 1e6f,
				particles, contacts);
		}
	}
}

PX_FORCE_INLINE bool avbdRigidSphereForwardVertexOwnsSweptFeature(
	const AvbdSoftParticle& particle,
	const AvbdRigidSphere& sphere,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	PxReal margin)
{
	if(particle.invMass <= 0.0f)
		return false;
	const PxVec3 vertexRelativeStart =
		particle.initialPosition - centerStart;
	const PxVec3 vertexRelativeEnd =
		particle.predictedPosition - centerEnd;
	const PxReal currentSdf =
		vertexRelativeStart.magnitude() - sphere.radius;
	if(!PxIsFinite(currentSdf))
		return false;
	if(currentSdf < margin)
		return true;
	PxReal vertexEntryTime = 0.0f;
	PxVec3 vertexEntryNormal(0.0f);
	return avbdSegmentEnterExpandedSphere(
		vertexRelativeStart, vertexRelativeEnd,
		PxVec3(0.0f), sphere.radius + margin,
		vertexEntryTime, vertexEntryNormal);
}

void avbdDetectSoftRigidSphereSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	PxArray<PxU8>* persistentForwardOwnerScratch)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8> localForwardOwnerScratch;
	PxArray<PxU8>& forwardOwnerScratch =
		persistentForwardOwnerScratch
			? *persistentForwardOwnerScratch
			: localForwardOwnerScratch;
	if(forwardOwnerScratch.capacity() < numParticles)
		forwardOwnerScratch.reserve(numParticles);
	forwardOwnerScratch.resize(numParticles);
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			const bool kinematicTarget =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
			const bool dynamicTarget =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite() ||
				!sphere.rotation.isFinite() ||
				(sphere.targetKind !=
						AvbdSoftContactTargetKind::eWORLD_STATIC &&
				 !kinematicTarget && !dynamicTarget) ||
				(kinematicTarget &&
				 (!sphere.previousCenter.isFinite() ||
				  !sphere.previousRotation.isFinite())) ||
				(dynamicTarget &&
				 (!sphere.predictedPoseValid ||
				  !sphere.predictedCenter.isFinite() ||
				  !sphere.predictedRotation.isFinite())))
				continue;
			const PxVec3 centerStart =
				kinematicTarget
					? sphere.previousCenter : sphere.center;
			const PxVec3 centerEnd =
				dynamicTarget
					? sphere.predictedCenter : sphere.center;
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleCount = body.compiled.particleCount;
			if(particleStart <= numParticles)
			{
				const PxU32 boundedParticleCount = PxMin(
					particleCount, numParticles - particleStart);
				for(PxU32 localParticle = 0;
					localParticle < boundedParticleCount; ++localParticle)
					forwardOwnerScratch[
						particleStart + localParticle] = 0;
			}
			for(PxU32 surfaceVertexIndex = 0;
				surfaceVertexIndex < body.compiled.surfaceVertices.size();
				++surfaceVertexIndex)
			{
				const PxU32 vertexIndex =
					body.compiled.surfaceVertices[surfaceVertexIndex];
				if(vertexIndex >= numParticles)
					continue;
				forwardOwnerScratch[vertexIndex] = PxU8(
					avbdRigidSphereForwardVertexOwnsSweptFeature(
						particles[vertexIndex], sphere,
						centerStart, centerEnd, margin));
			}
			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				// Outer contact redetection must reconstruct the same
				// frame-level sweep.  Using the iterated position here makes
				// a first-impact row disappear as soon as the first primal
				// sweep advances through the obstacle.
				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;
				const PxReal expandedRadius =
					sphere.radius + margin;
				const bool forwardVertexOwns =
					forwardOwnerScratch[v0] != 0 ||
					forwardOwnerScratch[v1] != 0 ||
					forwardOwnerScratch[v2] != 0;
				if(forwardVertexOwns)
					continue;
				const PxVec3 relativeEnd =
					centerEnd - displacement0;
				AvbdSweptTriangleEntry entry;
				const bool entered =
					softTriangleTranslationOnly
						? avbdSegmentEnterExpandedTriangleNonVertex(
							centerStart, relativeEnd,
							p0, p1, p2,
							expandedRadius, entry)
						: avbdLinearPointEnterExpandedDeformingTriangleNonVertex(
							centerStart, relativeEnd,
							p0, p1, p2, p0,
							p1 + relativeDisplacement1,
							p2 + relativeDisplacement2,
							expandedRadius, entry);
				if(!entered)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						entry.feature, entry.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x53505357u);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, sphere.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] = entry.barycentric.x;
				geometry.queryWeights[1] = entry.barycentric.y;
				geometry.queryWeights[2] = entry.barycentric.z;
				geometry.normal = entry.normal;
				geometry.projNormal = entry.normal;
				geometry.depth = 0.0f;
				geometry.margin = margin;
				const PxVec3 surfaceLocal =
					sphere.rotation.getConjugate().rotate(
						entry.normal * sphere.radius);
				avbdConfigureRigidSphereTarget(
					geometry, sphere, sphereIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						sphere.friction,
						sphere.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1.0e6f, 1.0e6f,
					particles, contacts);
			}
		}
	}
}

void avbdDetectSoftRigidSphereOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin)
{
	const PxReal normalEpsilon = 1.0e-12f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			const PxVec3& position =
				particles[particleIndex].position;
			bodyMinimum = bodyMinimum.minimum(position);
			bodyMaximum = bodyMaximum.maximum(position);
		}

		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite() ||
				!sphere.rotation.isFinite())
				continue;
			const PxReal queryRadius =
				sphere.radius + margin;
			if(bodyMinimum.x > sphere.center.x + queryRadius ||
				bodyMaximum.x < sphere.center.x - queryRadius ||
				bodyMinimum.y > sphere.center.y + queryRadius ||
				bodyMaximum.y < sphere.center.y - queryRadius ||
				bodyMinimum.z > sphere.center.z + queryRadius ||
				bodyMaximum.z < sphere.center.z - queryRadius)
				continue;

			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2) -
						PxVec3(queryRadius);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2) +
						PxVec3(queryRadius);
				if(sphere.center.x < triangleMinimum.x ||
					sphere.center.x > triangleMaximum.x ||
					sphere.center.y < triangleMinimum.y ||
					sphere.center.y > triangleMaximum.y ||
					sphere.center.z < triangleMinimum.z ||
					sphere.center.z > triangleMaximum.z)
					continue;

				const AvbdClosestPointResult closest =
					avbdClosestPointOnTriangleOGC(
						sphere.center, p0, p1, p2);
				// Vertex ownership already belongs to the forward
				// soft-vertex/sphere SDF. Reverse smooth ownership adds only
				// edge and face features, preventing duplicate AL states.
				if(closest.feature == AVBD_FEATURE_VERTEX ||
					closest.feature == AVBD_FEATURE_UNKNOWN ||
					!PxIsFinite(closest.distance) ||
					closest.distance >= queryRadius)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						closest.feature, closest.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x53504852u);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				PxVec3 normal = -closest.normal;
				const PxReal normalMagnitudeSq =
					normal.magnitudeSquared();
				if(!normal.isFinite() ||
					normalMagnitudeSq <= normalEpsilon)
					continue;
				normal *= PxRecipSqrt(normalMagnitudeSq);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, sphere.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] =
					closest.barycentric.x;
				geometry.queryWeights[1] =
					closest.barycentric.y;
				geometry.queryWeights[2] =
					closest.barycentric.z;
				geometry.normal = normal;
				geometry.projNormal = normal;
				geometry.depth =
					queryRadius - closest.distance;
				geometry.margin = margin;
				const PxVec3 surfaceLocal =
					sphere.rotation.getConjugate().rotate(
						normal * sphere.radius);
				avbdConfigureRigidSphereTarget(
					geometry, sphere, sphereIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						sphere.friction,
						sphere.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1e5f, 1e6f,
					particles, contacts);
			}
		}
	}
}

PX_FORCE_INLINE void avbdConfigureRigidCapsuleTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidCapsule& capsule, PxU32 capsuleIndex,
	const PxVec3& surfaceLocal)
{
	geometry.targetKind = capsule.targetKind;
	geometry.velocityOwner =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: capsule.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY
				? AvbdVelocityObjectiveOwner::ManifoldFinalize
				: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY
			? capsule.targetIndex : capsuleIndex;
	geometry.surfacePoint =
		capsule.center + capsule.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? capsule.previousCenter +
				capsule.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
	if(capsule.targetKind ==
		AvbdSoftContactTargetKind::eRIGID_BODY)
		geometry.rigidLocalPoint =
			capsule.shapeToRigidBody.transform(surfaceLocal);
}

// P5.6a candidate leaf: current-pose capsule SDF is particle-major and uses
// only immutable primitive/body inputs plus a caller-owned output stream.
// Swept and feature suffixes remain parent-owned until separately proven.
void avbdDetectSoftRigidCapsuleSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particles && particleBegin <= particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			if(capsule.radius <= 0.0f ||
				capsule.halfHeight < 0.0f ||
				!PxIsFinite(capsule.radius) ||
				!PxIsFinite(capsule.halfHeight) ||
				!capsule.center.isFinite() ||
				!capsule.rotation.isFinite())
				continue;
			const PxReal broadphaseRadius =
				capsule.radius + capsule.halfHeight + margin;
			const PxVec3 worldOffset =
				particle.position - capsule.center;
			if(worldOffset.magnitudeSquared() >=
				broadphaseRadius * broadphaseRadius)
				continue;
			const PxQuat inverseRotation =
				capsule.rotation.getConjugate();
			const PxVec3 particleLocal =
				inverseRotation.rotate(worldOffset);
			const PxVec3 axisLocal(
				PxClamp(particleLocal.x,
					-capsule.halfHeight,
					capsule.halfHeight),
				0.0f, 0.0f);
			const PxVec3 radialLocal =
				particleLocal - axisLocal;
			const PxReal distanceSq =
				radialLocal.magnitudeSquared();
			const PxReal queryRadius =
				capsule.radius + margin;
			if(!PxIsFinite(distanceSq) ||
				distanceSq >= queryRadius * queryRadius)
				continue;
			const PxReal distance =
				PxSqrt(PxMax(distanceSq, 0.0f));
			PxVec3 normalLocal(0.0f, 1.0f, 0.0f);
			if(distance > 1.0e-6f)
				normalLocal = radialLocal * (1.0f / distance);
			else
			{
				const PxVec3 initialLocal =
					inverseRotation.rotate(
						particle.initialPosition -
							capsule.center);
				const PxVec3 initialAxis(
					PxClamp(initialLocal.x,
						-capsule.halfHeight,
						capsule.halfHeight),
					0.0f, 0.0f);
				const PxVec3 initialRadial =
					initialLocal - initialAxis;
				const PxReal initialMagnitudeSq =
					initialRadial.magnitudeSquared();
				if(initialMagnitudeSq > 1.0e-12f &&
					PxIsFinite(initialMagnitudeSq))
					normalLocal = initialRadial *
						PxRecipSqrt(initialMagnitudeSq);
			}
			const PxVec3 normal =
				capsule.rotation.rotate(normalLocal).
					getNormalized();
			const PxReal sdf = distance - capsule.radius;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, capsule.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = sdf < 0.0f
				? -sdf : PxMax(0.0f, margin - sdf);
			geometry.margin = margin;
			const PxVec3 surfaceLocal =
				axisLocal + normalLocal * capsule.radius;
			avbdConfigureRigidCapsuleTarget(
				geometry, capsule, capsuleIndex,
				surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					capsule.friction,
					capsule.frictionCombineMode)
				: PxMax(capsule.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

PX_FORCE_INLINE bool avbdGetRigidCapsuleSweepPose(
	const AvbdRigidCapsule& capsule,
	PxVec3& centerStart, PxVec3& centerEnd,
	PxQuat& rotationStart, PxQuat& rotationEnd,
	bool& rotationsEquivalent)
{
	const bool kinematicTarget =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	const bool dynamicTarget =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY;
	if(capsule.radius <= 0.0f ||
		capsule.halfHeight < 0.0f ||
		!PxIsFinite(capsule.radius) ||
		!PxIsFinite(capsule.halfHeight) ||
		!capsule.center.isFinite() ||
		!capsule.rotation.isFinite() ||
		(capsule.targetKind !=
				AvbdSoftContactTargetKind::eWORLD_STATIC &&
		 !kinematicTarget && !dynamicTarget) ||
		(kinematicTarget &&
		 (!capsule.previousCenter.isFinite() ||
		  !capsule.previousRotation.isFinite())) ||
		(dynamicTarget &&
		 (!capsule.predictedPoseValid ||
		  !capsule.predictedCenter.isFinite() ||
		  !capsule.predictedRotation.isFinite())))
		return false;

	centerStart =
		kinematicTarget ? capsule.previousCenter : capsule.center;
	centerEnd =
		dynamicTarget ? capsule.predictedCenter : capsule.center;
	rotationStart =
		kinematicTarget ? capsule.previousRotation : capsule.rotation;
	rotationEnd =
		dynamicTarget ? capsule.predictedRotation : capsule.rotation;
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite())
		return false;
	rotationsEquivalent = avbdAreSweepRotationsEquivalent(
		rotationStart, rotationEnd);
	return true;
}

// Conservative point entry against a capsule whose center translates
// linearly and whose orientation follows shortest-path quaternion slerp.
// Point/center relative translation plus halfHeight*angularDistance bounds
// the Hausdorff speed of the moving medial segment, so gap/speed cannot step
// across first contact. The returned shape-local point is the material point
// at entry and remains valid for the prescribed end pose.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedRotatingCapsule(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal halfHeight, PxReal capsuleRadius, PxReal margin,
	AvbdSweptRotatingCapsulePointEntry& result)
{
	if(!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		halfHeight < 0.0f || capsuleRadius <= 0.0f ||
		margin <= 0.0f || !PxIsFinite(halfHeight) ||
		!PxIsFinite(capsuleRadius) || !PxIsFinite(margin))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance) ||
		angularDistance <= 0.0f)
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 relativeTranslation =
		(pointEnd - pointStart) - (centerEnd - centerStart);
	const PxReal speed =
		relativeTranslation.magnitude() +
		halfHeight * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;

	const PxReal expandedRadius = capsuleRadius + margin;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);
	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + (pointEnd - pointStart) * time;
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!point.isFinite() || !center.isFinite() ||
			!rotation.isFinite())
			return false;
		const PxVec3 axis = rotation.getBasisVector0();
		const PxReal axisCoordinate = PxClamp(
			(point - center).dot(axis),
			-halfHeight, halfHeight);
		const PxVec3 medialPoint =
			center + axis * axisCoordinate;
		const PxVec3 delta = point - medialPoint;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		if(iteration == 0 && distance < expandedRadius)
			return false;
		const PxReal gap = distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			const PxReal normalMagnitudeSq =
				delta.magnitudeSquared();
			if(normalMagnitudeSq <= 1.0e-12f ||
				!PxIsFinite(normalMagnitudeSq))
				return false;
			result.entryTime = time;
			result.normal =
				delta * PxRecipSqrt(normalMagnitudeSq);
			const PxVec3 normalLocal =
				rotation.getConjugate().rotate(result.normal);
			result.surfaceLocal =
				PxVec3(axisCoordinate, 0.0f, 0.0f) +
				normalLocal * capsuleRadius;
			return result.normal.isFinite() &&
				result.surfaceLocal.isFinite();
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Exact segment entry against a static capsule whose medial segment is the
// shape-local X interval [-halfHeight, halfHeight]. The query segment is
// already expressed in the common capsule frame; callers must fail closed
// when the capsule rotates during the timestep.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedCapsule(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	PxReal halfHeight, PxReal expandedRadius,
	PxReal& entryTime, PxVec3& entryNormalLocal,
	PxVec3& medialPointLocal)
{
	if(!segmentStart.isFinite() || !segmentEnd.isFinite() ||
		halfHeight < 0.0f || expandedRadius <= 0.0f ||
		!PxIsFinite(halfHeight) || !PxIsFinite(expandedRadius))
		return false;

	const PxVec3 direction = segmentEnd - segmentStart;
	const PxReal directionMagnitudeSq =
		direction.magnitudeSquared();
	if(directionMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(directionMagnitudeSq))
		return false;

	PxReal bestTime = PX_MAX_F32;
	PxVec3 bestNormal(0.0f);
	PxVec3 bestMedial(0.0f);

	// Infinite-cylinder entry, restricted to the finite medial interval.
	const PxReal cylinderA =
		direction.y * direction.y +
		direction.z * direction.z;
	const PxReal cylinderHalfB =
		segmentStart.y * direction.y +
		segmentStart.z * direction.z;
	const PxReal cylinderC =
		segmentStart.y * segmentStart.y +
		segmentStart.z * segmentStart.z -
			expandedRadius * expandedRadius;
	if(cylinderA > 1.0e-12f && cylinderC >= 0.0f)
	{
		const PxReal discriminant =
			cylinderHalfB * cylinderHalfB -
				cylinderA * cylinderC;
		if(discriminant >= 0.0f &&
			PxIsFinite(discriminant))
		{
			const PxReal candidateTime =
				(-cylinderHalfB - PxSqrt(discriminant)) /
					cylinderA;
			if(candidateTime >= 0.0f &&
				candidateTime <= 1.0f)
			{
				const PxVec3 candidate =
					segmentStart + direction * candidateTime;
				if(candidate.x >= -halfHeight &&
					candidate.x <= halfHeight)
				{
					const PxVec3 radial(
						0.0f, candidate.y, candidate.z);
					const PxReal radialMagnitudeSq =
						radial.magnitudeSquared();
					if(radialMagnitudeSq > 1.0e-12f &&
						PxIsFinite(radialMagnitudeSq))
					{
						bestTime = candidateTime;
						bestNormal = radial *
							PxRecipSqrt(radialMagnitudeSq);
						bestMedial =
							PxVec3(candidate.x, 0.0f, 0.0f);
					}
				}
			}
		}
	}

	// Full endpoint spheres plus the finite cylinder form the capsule union.
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		const PxVec3 capCenter(
			endpoint == 0 ? -halfHeight : halfHeight,
			0.0f, 0.0f);
		PxReal candidateTime = 0.0f;
		PxVec3 candidateNormal(0.0f);
		if(avbdSegmentEnterExpandedSphere(
				segmentStart, segmentEnd, capCenter,
				expandedRadius, candidateTime,
				candidateNormal) &&
			candidateTime < bestTime)
		{
			bestTime = candidateTime;
			bestNormal = candidateNormal;
			bestMedial = capCenter;
		}
	}

	if(bestTime == PX_MAX_F32 ||
		!bestNormal.isFinite() ||
		bestNormal.magnitudeSquared() <= 1.0e-12f)
		return false;
	entryTime = bestTime;
	entryNormalLocal = bestNormal.getNormalized();
	medialPointLocal = bestMedial;
	return true;
}

// P5.14a candidate leaf: swept capsule SDF is particle-major and carries all
// conservative-advancement state on the stack. A caller that partitions it
// must merge the full current-SDF family before its swept-family ranges.
void avbdDetectSoftRigidCapsuleSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle =
			particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite() ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			const bool kinematicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
			const bool dynamicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidCapsuleSweepPose(
					capsule, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent) ||
				(!rotationsEquivalent &&
				 !kinematicTarget && !dynamicTarget))
				continue;

			PxVec3 contactNormal(0.0f);
			PxVec3 surfaceLocal(0.0f);
			if(rotationsEquivalent)
			{
				// With a fixed orientation, both moving endpoints share one
				// exact capsule-local frame. Translation of either object is
				// represented by the relative point segment.
				const PxQuat inverseRotation =
					rotationEnd.getConjugate();
				const PxVec3 relativeStart =
					inverseRotation.rotate(
						particle.position - centerStart);
				const PxVec3 relativeEnd =
					inverseRotation.rotate(
						particle.predictedPosition - centerEnd);
				const PxVec3 currentAxis(
					PxClamp(
						relativeStart.x,
						-capsule.halfHeight,
						capsule.halfHeight),
					0.0f, 0.0f);
				const PxReal currentSdf =
					(relativeStart - currentAxis).magnitude() -
						capsule.radius;
				if(!PxIsFinite(currentSdf) || currentSdf < margin)
					continue;

				PxReal entryTime = 0.0f;
				PxVec3 entryNormalLocal(0.0f);
				PxVec3 medialPointLocal(0.0f);
				if(!avbdSegmentEnterExpandedCapsule(
						relativeStart, relativeEnd,
						capsule.halfHeight,
						capsule.radius + margin,
						entryTime, entryNormalLocal,
						medialPointLocal))
					continue;
				contactNormal =
					rotationEnd.rotate(entryNormalLocal).
						getNormalized();
				surfaceLocal =
					medialPointLocal +
						entryNormalLocal * capsule.radius;
			}
			else
			{
				AvbdSweptRotatingCapsulePointEntry entry;
				if(!avbdSegmentEnterExpandedRotatingCapsule(
						particle.position,
						particle.predictedPosition,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						capsule.halfHeight, capsule.radius,
						margin, entry))
					continue;
				contactNormal = entry.normal;
				surfaceLocal = entry.surfaceLocal;
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, capsule.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = contactNormal;
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			avbdConfigureRigidCapsuleTarget(
				geometry, capsule, capsuleIndex,
				surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					capsule.friction,
					capsule.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e6f, 1e6f,
				particles, contacts);
		}
	}
}

// Continuous entry of a translated finite segment into a triangle expanded
// by a radius. Exact segment/triangle distance queries drive conservative
// advancement, so every step is bounded by the relative translation speed.
// Soft-triangle vertices are deliberately excluded: the forward
// soft-vertex/capsule swept SDF is their unique owner.
PX_FORCE_INLINE bool
avbdTranslatedSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& segment0, const PxVec3& segment1,
	const PxVec3& translation,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius,
	AvbdSweptCapsuleTriangleEntry& result)
{
	if(!segment0.isFinite() || !segment1.isFinite() ||
		!translation.isFinite() || !a.isFinite() ||
		!b.isFinite() || !c.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxReal speedSq = translation.magnitudeSquared();
	if(speedSq <= 1.0e-12f || !PxIsFinite(speedSq))
		return false;
	const PxReal speed = PxSqrt(speedSq);
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	const AvbdClosestSegmentTriangleResult currentClosest =
		avbdClosestSegmentTriangleOGC(
			segment0, segment1, a, b, c);
	if(!PxIsFinite(currentClosest.distance) ||
		currentClosest.distance < expandedRadius)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		const PxVec3 offset = translation * time;
		const AvbdClosestSegmentTriangleResult closest =
			avbdClosestSegmentTriangleOGC(
				segment0 + offset, segment1 + offset,
				a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			PxVec3 normal =
				closest.trianglePoint - closest.segmentPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.segmentWeight1 =
				PxClamp(closest.segmentWeight1, 0.0f, 1.0f);
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}

		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f)
			return false;
		if(nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous entry of a rotating/translating finite segment into a triangle
// expanded by a radius. The triangle is expressed in a frame with its common
// translation removed. Relative center translation plus
// halfHeight*angularDistance bounds the Hausdorff speed of the medial
// segment. Soft-triangle vertices remain owned by the forward capsule SDF.
PX_FORCE_INLINE bool
avbdRotatingSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal halfHeight,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius,
	AvbdSweptCapsuleTriangleEntry& result)
{
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		halfHeight < 0.0f || !PxIsFinite(halfHeight) ||
		!a.isFinite() || !b.isFinite() || !c.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		halfHeight * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!center.isFinite() || !rotation.isFinite())
			return false;
		const PxVec3 axisOffset =
			rotation.getBasisVector0() * halfHeight;
		const AvbdClosestSegmentTriangleResult closest =
			avbdClosestSegmentTriangleOGC(
				center - axisOffset, center + axisOffset,
				a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 && closest.distance < expandedRadius)
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			PxVec3 normal =
				closest.trianglePoint - closest.segmentPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.segmentWeight1 =
				PxClamp(closest.segmentWeight1, 0.0f, 1.0f);
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}

		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous rotating/translating finite segment entry into a linearly
// deforming triangle. A common soft displacement may be removed by the
// caller. Center translation plus endpoint angular speed plus the maximum
// residual soft-vertex speed is a conservative Hausdorff speed bound.
// Triangle vertex caps remain owned by forward rigid-SDF sweeps.
PX_FORCE_INLINE bool
avbdRotatingSegmentEnterExpandedDeformingTriangleNonVertex(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal expandedRadius,
	AvbdSweptCapsuleTriangleEntry& result)
{
	if(!rigidLocal0.isFinite() || !rigidLocal1.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!aStart.isFinite() || !bStart.isFinite() ||
		!cStart.isFinite() || !aEnd.isFinite() ||
		!bEnd.isFinite() || !cEnd.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 displacementA = aEnd - aStart;
	const PxVec3 displacementB = bEnd - bStart;
	const PxVec3 displacementC = cEnd - cStart;
	const PxReal triangleSpeed = PxMax(
		displacementA.magnitude(),
		PxMax(displacementB.magnitude(),
			displacementC.magnitude()));
	const PxReal segmentRadius = PxMax(
		rigidLocal0.magnitude(), rigidLocal1.magnitude());
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		segmentRadius * angularDistance + triangleSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		const PxVec3 a = aStart + displacementA * time;
		const PxVec3 b = bStart + displacementB * time;
		const PxVec3 c = cStart + displacementC * time;
		const PxVec3 triangleNormal = (b - a).cross(c - a);
		if(!center.isFinite() || !rotation.isFinite() ||
			!a.isFinite() || !b.isFinite() || !c.isFinite() ||
			!triangleNormal.isFinite() ||
			triangleNormal.magnitudeSquared() <= 1.0e-16f)
			return false;
		const PxVec3 rigid0 =
			center + rotation.rotate(rigidLocal0);
		const PxVec3 rigid1 =
			center + rotation.rotate(rigidLocal1);
		const AvbdClosestSegmentTriangleResult closest =
			avbdClosestSegmentTriangleOGC(
				rigid0, rigid1, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 &&
			closest.distance < expandedRadius)
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			const PxVec3 normal =
				closest.trianglePoint - closest.segmentPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.segmentWeight1 =
				PxClamp(closest.segmentWeight1, 0.0f, 1.0f);
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

PX_NOINLINE inline bool avbdRigidCapsuleForwardVertexOwnsSweptFeature(
	const AvbdSoftParticle& particle,
	const AvbdRigidCapsule& capsule,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxQuat& inverseRotationEnd,
	bool rotationsEquivalent, PxReal margin)
{
	if(particle.invMass <= 0.0f)
		return false;
	const PxVec3 pointStart = particle.initialPosition;
	const PxVec3 pointEnd = particle.predictedPosition;
	if(rotationsEquivalent)
	{
		const PxVec3 relativeStart =
			inverseRotationEnd.rotate(pointStart - centerStart);
		const PxVec3 relativeEnd =
			inverseRotationEnd.rotate(pointEnd - centerEnd);
		const PxVec3 currentAxis(
			PxClamp(relativeStart.x,
				-capsule.halfHeight, capsule.halfHeight),
			0.0f, 0.0f);
		const PxReal currentSdf =
			(relativeStart - currentAxis).magnitude() -
				capsule.radius;
		if(!PxIsFinite(currentSdf))
			return false;
		if(currentSdf < margin)
			return true;
		PxReal vertexEntryTime = 0.0f;
		PxVec3 vertexEntryNormal(0.0f);
		PxVec3 vertexMedialPoint(0.0f);
		return avbdSegmentEnterExpandedCapsule(
			relativeStart, relativeEnd,
			capsule.halfHeight, capsule.radius + margin,
			vertexEntryTime, vertexEntryNormal,
			vertexMedialPoint);
	}

	const PxVec3 startAxis = rotationStart.getBasisVector0();
	const PxReal axisCoordinate = PxClamp(
		(pointStart - centerStart).dot(startAxis),
		-capsule.halfHeight, capsule.halfHeight);
	const PxReal currentSdf =
		(pointStart - (centerStart + startAxis * axisCoordinate)).
			magnitude() - capsule.radius;
	if(!PxIsFinite(currentSdf))
		return false;
	if(currentSdf < margin)
		return true;
	AvbdSweptRotatingCapsulePointEntry vertexEntry;
	return avbdSegmentEnterExpandedRotatingCapsule(
		pointStart, pointEnd,
		centerStart, centerEnd,
		rotationStart, rotationEnd,
		capsule.halfHeight, capsule.radius, margin,
		vertexEntry);
}

void avbdDetectSoftRigidCapsuleSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	PxArray<PxU8>* persistentForwardOwnerScratch)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8> localForwardOwnerScratch;
	PxArray<PxU8>& forwardOwnerScratch =
		persistentForwardOwnerScratch
			? *persistentForwardOwnerScratch
			: localForwardOwnerScratch;
	if(forwardOwnerScratch.capacity() < numParticles)
		forwardOwnerScratch.reserve(numParticles);
	forwardOwnerScratch.resize(numParticles);
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			const bool kinematicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
			const bool dynamicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			if(capsule.radius <= 0.0f ||
				capsule.halfHeight < 0.0f ||
				!PxIsFinite(capsule.radius) ||
				!PxIsFinite(capsule.halfHeight) ||
				!capsule.center.isFinite() ||
				!capsule.rotation.isFinite() ||
				(capsule.targetKind !=
						AvbdSoftContactTargetKind::eWORLD_STATIC &&
				 !kinematicTarget && !dynamicTarget) ||
				(kinematicTarget &&
				 (!capsule.previousCenter.isFinite() ||
				  !capsule.previousRotation.isFinite())) ||
				(dynamicTarget &&
				 (!capsule.predictedPoseValid ||
				  !capsule.predictedCenter.isFinite() ||
				  !capsule.predictedRotation.isFinite())))
				continue;

			const PxVec3 centerStart =
				kinematicTarget
					? capsule.previousCenter : capsule.center;
			const PxVec3 centerEnd =
				dynamicTarget
					? capsule.predictedCenter : capsule.center;
			const PxQuat rotationStart =
				kinematicTarget
					? capsule.previousRotation : capsule.rotation;
			const PxQuat rotationEnd =
				dynamicTarget
					? capsule.predictedRotation : capsule.rotation;
			if(!centerStart.isFinite() || !centerEnd.isFinite() ||
				!rotationStart.isFinite() || !rotationEnd.isFinite())
				continue;

			const bool rotationsEquivalent =
				avbdAreSweepRotationsEquivalent(
					rotationStart, rotationEnd);
			const PxQuat inverseRotation =
				rotationEnd.getConjugate();
			const PxVec3 axisOffset =
				rotationEnd.getBasisVector0() *
					capsule.halfHeight;
			const PxVec3 segment0 = centerStart - axisOffset;
			const PxVec3 segment1 = centerStart + axisOffset;
			const PxReal expandedRadius =
				capsule.radius + margin;
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleCount = body.compiled.particleCount;
			if(particleStart <= numParticles)
			{
				const PxU32 boundedParticleCount = PxMin(
					particleCount, numParticles - particleStart);
				for(PxU32 localParticle = 0;
					localParticle < boundedParticleCount; ++localParticle)
					forwardOwnerScratch[
						particleStart + localParticle] = 0;
			}
			for(PxU32 surfaceVertexIndex = 0;
				surfaceVertexIndex < body.compiled.surfaceVertices.size();
				++surfaceVertexIndex)
			{
				const PxU32 vertexIndex =
					body.compiled.surfaceVertices[surfaceVertexIndex];
				if(vertexIndex >= numParticles)
					continue;
				forwardOwnerScratch[vertexIndex] = PxU8(
					avbdRigidCapsuleForwardVertexOwnsSweptFeature(
						particles[vertexIndex], capsule,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						inverseRotation, rotationsEquivalent,
						margin));
			}
			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;

				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;

				const bool forwardVertexOwns =
					forwardOwnerScratch[v0] != 0 ||
					forwardOwnerScratch[v1] != 0 ||
					forwardOwnerScratch[v2] != 0;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				PxVec3 sweptMinimum(0.0f);
				PxVec3 sweptMaximum(0.0f);
				if(rotationsEquivalent)
				{
					const PxVec3 segment0End =
						segment0 + relativeTranslation;
					const PxVec3 segment1End =
						segment1 + relativeTranslation;
					sweptMinimum =
						segment0.minimum(segment1).
							minimum(segment0End).minimum(segment1End) -
								PxVec3(expandedRadius);
					sweptMaximum =
						segment0.maximum(segment1).
							maximum(segment0End).maximum(segment1End) +
								PxVec3(expandedRadius);
				}
				else
				{
					const PxReal rotationExtent =
						capsule.halfHeight + expandedRadius;
					sweptMinimum =
						centerStart.minimum(relativeCenterEnd) -
							PxVec3(rotationExtent);
					sweptMaximum =
						centerStart.maximum(relativeCenterEnd) +
							PxVec3(rotationExtent);
				}
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2).
						minimum(
							p1 + relativeDisplacement1).
						minimum(
							p2 + relativeDisplacement2);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2).
						maximum(
							p1 + relativeDisplacement1).
						maximum(
							p2 + relativeDisplacement2);
				if(sweptMinimum.x > triangleMaximum.x ||
					sweptMaximum.x < triangleMinimum.x ||
					sweptMinimum.y > triangleMaximum.y ||
					sweptMaximum.y < triangleMinimum.y ||
					sweptMinimum.z > triangleMaximum.z ||
					sweptMaximum.z < triangleMinimum.z)
					continue;

				AvbdSweptCapsuleTriangleEntry entry;
				const bool entered =
					softTriangleTranslationOnly &&
						rotationsEquivalent
						? avbdTranslatedSegmentEnterExpandedTriangleNonVertex(
							segment0, segment1, relativeTranslation,
							p0, p1, p2, expandedRadius, entry)
						: softTriangleTranslationOnly
						? avbdRotatingSegmentEnterExpandedTriangleNonVertex(
							centerStart, relativeCenterEnd,
							rotationStart, rotationEnd,
							capsule.halfHeight,
							p0, p1, p2, expandedRadius, entry)
						: avbdRotatingSegmentEnterExpandedDeformingTriangleNonVertex(
							PxVec3(-capsule.halfHeight, 0.0f, 0.0f),
							PxVec3(capsule.halfHeight, 0.0f, 0.0f),
							centerStart, relativeCenterEnd,
							rotationStart, rotationEnd,
							p0, p1, p2, p0,
							p1 + relativeDisplacement1,
							p2 + relativeDisplacement2,
							expandedRadius, entry);
				if(!entered)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						entry.feature, entry.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x43505257u);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, capsule.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] = entry.barycentric.x;
				geometry.queryWeights[1] = entry.barycentric.y;
				geometry.queryWeights[2] = entry.barycentric.z;
				geometry.normal = entry.normal;
				geometry.projNormal = entry.normal;
				geometry.depth = 0.0f;
				geometry.margin = margin;
				const PxVec3 medialPointLocal(
					-capsule.halfHeight +
						2.0f * capsule.halfHeight *
							entry.segmentWeight1,
					0.0f, 0.0f);
				const PxQuat entryRotation =
					rotationsEquivalent
						? rotationEnd.getNormalized()
						: PxSlerp(
							entry.entryTime,
							rotationStart.getNormalized(),
							rotationEnd.getNormalized()).
								getNormalized();
				const PxVec3 surfaceLocal =
					medialPointLocal +
						entryRotation.getConjugate().
							rotate(entry.normal) *
							capsule.radius;
				avbdConfigureRigidCapsuleTarget(
					geometry, capsule, capsuleIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						capsule.friction,
						capsule.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1.0e6f, 1.0e6f,
					particles, contacts);
			}
		}
	}
}

void avbdDetectSoftRigidCapsuleOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin)
{
	const PxReal normalEpsilon = 1.0e-12f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			bodyMinimum = bodyMinimum.minimum(
				particles[particleIndex].position);
			bodyMaximum = bodyMaximum.maximum(
				particles[particleIndex].position);
		}

		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			if(capsule.radius <= 0.0f ||
				capsule.halfHeight < 0.0f ||
				!PxIsFinite(capsule.radius) ||
				!PxIsFinite(capsule.halfHeight) ||
				!capsule.center.isFinite() ||
				!capsule.rotation.isFinite())
				continue;
			const PxVec3 axisOffset =
				capsule.rotation.getBasisVector0() *
					capsule.halfHeight;
			const PxVec3 segment0 =
				capsule.center - axisOffset;
			const PxVec3 segment1 =
				capsule.center + axisOffset;
			const PxReal queryRadius =
				capsule.radius + margin;
			const PxVec3 capsuleMinimum =
				segment0.minimum(segment1) -
					PxVec3(queryRadius);
			const PxVec3 capsuleMaximum =
				segment0.maximum(segment1) +
					PxVec3(queryRadius);
			if(bodyMinimum.x > capsuleMaximum.x ||
				bodyMaximum.x < capsuleMinimum.x ||
				bodyMinimum.y > capsuleMaximum.y ||
				bodyMaximum.y < capsuleMinimum.y ||
				bodyMinimum.z > capsuleMaximum.z ||
				bodyMaximum.z < capsuleMinimum.z)
				continue;

			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2) -
						PxVec3(queryRadius);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2) +
						PxVec3(queryRadius);
				if(segment0.minimum(segment1).x >
						triangleMaximum.x ||
					segment0.maximum(segment1).x <
						triangleMinimum.x ||
					segment0.minimum(segment1).y >
						triangleMaximum.y ||
					segment0.maximum(segment1).y <
						triangleMinimum.y ||
					segment0.minimum(segment1).z >
						triangleMaximum.z ||
					segment0.maximum(segment1).z <
						triangleMinimum.z)
					continue;

				AvbdClosestSegmentTriangleResult candidates[2];
				candidates[0] = avbdClosestSegmentTriangleOGC(
					segment0, segment1, p0, p1, p2);
				PxU32 candidateCount = 1;

				// A capsule axis parallel to a coarse triangle face has a
				// line manifold. The generic closest query deliberately keeps
				// one strict minimum, so an exact tie otherwise collapses that
				// manifold to segment0. Preserve both independent face support
				// points only when both endpoint projections are owned by this
				// face; edge and vertex ownership remains unchanged.
				const PxVec3 segmentDirection = segment1 - segment0;
				const PxVec3 triangleNormal = (p1 - p0).cross(p2 - p0);
				const PxReal segmentMagnitudeSq =
					segmentDirection.magnitudeSquared();
				const PxReal triangleNormalMagnitudeSq =
					triangleNormal.magnitudeSquared();
				const PxReal axisNormalDot =
					segmentDirection.dot(triangleNormal);
				if(segmentMagnitudeSq > normalEpsilon &&
					triangleNormalMagnitudeSq > normalEpsilon &&
					axisNormalDot * axisNormalDot <=
						1.0e-10f * segmentMagnitudeSq *
							triangleNormalMagnitudeSq)
				{
					const AvbdClosestPointResult endpoint0 =
						avbdClosestPointOnTriangleOGC(
							segment0, p0, p1, p2);
					const AvbdClosestPointResult endpoint1 =
						avbdClosestPointOnTriangleOGC(
							segment1, p0, p1, p2);
					if(endpoint0.feature == AVBD_FEATURE_FACE &&
						endpoint1.feature == AVBD_FEATURE_FACE &&
						PxIsFinite(endpoint0.distance) &&
						PxIsFinite(endpoint1.distance) &&
						endpoint0.distance < queryRadius &&
						endpoint1.distance < queryRadius)
					{
						candidates[0].segmentPoint = segment0;
						candidates[0].trianglePoint = endpoint0.point;
						candidates[0].barycentric = endpoint0.barycentric;
						candidates[0].segmentWeight1 = 0.0f;
						candidates[0].distance = endpoint0.distance;
						candidates[0].feature = endpoint0.feature;
						candidates[0].featureIndex = endpoint0.featureIndex;
						candidates[1].segmentPoint = segment1;
						candidates[1].trianglePoint = endpoint1.point;
						candidates[1].barycentric = endpoint1.barycentric;
						candidates[1].segmentWeight1 = 1.0f;
						candidates[1].distance = endpoint1.distance;
						candidates[1].feature = endpoint1.feature;
						candidates[1].featureIndex = endpoint1.featureIndex;
						candidateCount = 2;
					}
				}

				for(PxU32 candidateIndex = 0;
					candidateIndex < candidateCount; ++candidateIndex)
				{
					const AvbdClosestSegmentTriangleResult& closest =
						candidates[candidateIndex];
					// Soft vertices are exclusively owned by the forward
					// vertex/capsule SDF. Reverse ownership fills only edge/face
					// gaps, including a medial segment under a coarse face.
					if(closest.feature == AVBD_FEATURE_VERTEX ||
						closest.feature == AVBD_FEATURE_UNKNOWN ||
						!PxIsFinite(closest.distance) ||
						closest.distance >= queryRadius)
						continue;

					const PxU64 softFeatureKey =
						avbdSoftTriangleFeatureKey(
							v0, v1, v2,
							closest.feature,
							closest.featureIndex);
					PxU64 featureKey = 1469598103934665603ull;
					featureKey = avbdSoftContactHashValue(
						featureKey, 0x4350534cu);
					featureKey = avbdSoftContactHashValue(
						featureKey, PxU32(softFeatureKey));
					featureKey = avbdSoftContactHashValue(
						featureKey, PxU32(softFeatureKey >> 32));
					if(candidateIndex > 0)
					{
						featureKey = avbdSoftContactHashValue(
							featureKey, 0x4d414e49u);
						featureKey = avbdSoftContactHashValue(
							featureKey, candidateIndex);
					}
					bool duplicate = false;
					for(PxU32 emittedIndex = 0;
						emittedIndex < emittedFeatureKeys.size();
						++emittedIndex)
					{
						if(emittedFeatureKeys[emittedIndex] ==
							featureKey)
						{
							duplicate = true;
							break;
						}
					}
					if(duplicate)
						continue;
					emittedFeatureKeys.pushBack(featureKey);

					PxVec3 normal =
						closest.trianglePoint -
							closest.segmentPoint;
					PxReal normalMagnitudeSq =
						normal.magnitudeSquared();
					if(normalMagnitudeSq <= normalEpsilon ||
						!PxIsFinite(normalMagnitudeSq))
					{
						normal = triangleNormal;
						normalMagnitudeSq =
							normal.magnitudeSquared();
						if(normalMagnitudeSq <= normalEpsilon ||
							!PxIsFinite(normalMagnitudeSq))
							continue;
						normal *= PxRecipSqrt(normalMagnitudeSq);
						const PxVec3 triangleCentroid =
							(p0 + p1 + p2) * (1.0f / 3.0f);
						if(normal.dot(
							triangleCentroid - capsule.center) < 0.0f)
							normal = -normal;
					}
					else
						normal *= PxRecipSqrt(normalMagnitudeSq);

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, capsule.primitiveKey,
						featureKey);
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						closest.barycentric.x;
					geometry.queryWeights[1] =
						closest.barycentric.y;
					geometry.queryWeights[2] =
						closest.barycentric.z;
					geometry.normal = normal;
					geometry.projNormal = normal;
					geometry.depth =
						queryRadius - closest.distance;
					geometry.margin = margin;
					const PxVec3 surfaceWorld =
						closest.segmentPoint +
							normal * capsule.radius;
					const PxVec3 surfaceLocal =
						capsule.rotation.getConjugate().rotate(
							surfaceWorld - capsule.center);
					avbdConfigureRigidCapsuleTarget(
						geometry, capsule, capsuleIndex,
						surfaceLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							capsule.friction,
							capsule.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}
		}
	}
}

PX_FORCE_INLINE bool avbdIsRigidConvexValid(
	const AvbdRigidConvex& convex)
{
	return convex.center.isFinite() &&
		convex.rotation.isFinite() &&
		PxIsFinite(convex.localRadius) &&
		convex.localRadius > 0.0f &&
		convex.vertices.size() >= 4 &&
		!convex.faces.empty() &&
		!convex.triangles.empty();
}

PX_FORCE_INLINE void avbdConfigureRigidConvexTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidConvex& convex, PxU32 convexIndex,
	const PxVec3& surfaceLocal)
{
	geometry.targetKind = convex.targetKind;
	geometry.velocityOwner =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: convex.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY
				? AvbdVelocityObjectiveOwner::ManifoldFinalize
				: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY
			? convex.targetIndex : convexIndex;
	geometry.surfacePoint =
		convex.center + convex.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? convex.previousCenter +
				convex.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
	if(convex.targetKind ==
		AvbdSoftContactTargetKind::eRIGID_BODY)
		geometry.rigidLocalPoint =
			convex.shapeToRigidBody.transform(surfaceLocal);
}

PX_FORCE_INLINE PxU64 avbdRigidConvexFeatureKey(
	PxU32 tag, PxU32 triangleOrFaceIndex,
	AvbdClosestFeature feature, PxU32 featureIndex)
{
	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, tag);
	hash = avbdSoftContactHashValue(hash, triangleOrFaceIndex);
	hash = avbdSoftContactHashValue(hash, PxU32(feature));
	return avbdSoftContactHashValue(hash, featureIndex);
}

// Exact closed-hull point query shared by discrete and swept vertex owners.
// Negative distance denotes a point inside the convex; outside distance and
// material point come from the closest baked hull triangle.
PX_FORCE_INLINE bool avbdQueryRigidConvexLocal(
	const AvbdRigidConvex& convex, const PxVec3& localPoint,
	AvbdRigidConvexPointQuery& result)
{
	if(!avbdIsRigidConvexValid(convex) || !localPoint.isFinite())
		return false;

	PxReal maximumPlaneDistance = -PX_MAX_F32;
	PxU32 maximumFace = PX_MAX_U32;
	for(PxU32 faceIndex = 0;
		faceIndex < convex.faces.size(); ++faceIndex)
	{
		const AvbdRigidConvexFace& face =
			convex.faces[faceIndex];
		const PxReal planeDistance =
			face.normal.dot(localPoint) - face.offset;
		if(planeDistance > maximumPlaneDistance)
		{
			maximumPlaneDistance = planeDistance;
			maximumFace = faceIndex;
		}
	}
	if(maximumFace == PX_MAX_U32 ||
		!PxIsFinite(maximumPlaneDistance))
		return false;

	if(maximumPlaneDistance <= 0.0f)
	{
		result.signedDistance = maximumPlaneDistance;
		result.normalLocal = convex.faces[maximumFace].normal;
		result.surfaceLocal =
			localPoint -
				result.normalLocal * result.signedDistance;
		result.featureKey = avbdRigidConvexFeatureKey(
			0x43465846u, maximumFace,
			AVBD_FEATURE_FACE, 0u);
	}
	else
	{
		AvbdClosestPointResult bestClosest = {};
		PxReal bestDistance = PX_MAX_F32;
		PxU32 bestTriangle = PX_MAX_U32;
		for(PxU32 triangleIndex = 0;
			triangleIndex < convex.triangles.size();
			++triangleIndex)
		{
			const AvbdRigidConvexTriangle& triangle =
				convex.triangles[triangleIndex];
			if(triangle.p0 >= convex.vertices.size() ||
				triangle.p1 >= convex.vertices.size() ||
				triangle.p2 >= convex.vertices.size())
				continue;
			const AvbdClosestPointResult closest =
				avbdClosestPointOnTriangleOGC(
					localPoint,
					convex.vertices[triangle.p0],
					convex.vertices[triangle.p1],
					convex.vertices[triangle.p2]);
			if(closest.distance < bestDistance)
			{
				bestDistance = closest.distance;
				bestClosest = closest;
				bestTriangle = triangleIndex;
			}
		}
		if(bestTriangle == PX_MAX_U32 ||
			!PxIsFinite(bestDistance))
			return false;
		result.signedDistance = bestDistance;
		result.surfaceLocal = bestClosest.point;
		result.normalLocal = bestClosest.normal;
		const PxReal normalMagnitudeSq =
			result.normalLocal.magnitudeSquared();
		if(!result.normalLocal.isFinite() ||
			normalMagnitudeSq <= 1.0e-12f)
		{
			const PxU32 faceIndex =
				convex.triangles[bestTriangle].faceIndex;
			if(faceIndex >= convex.faces.size())
				return false;
			result.normalLocal =
				convex.faces[faceIndex].normal;
		}
		result.featureKey = avbdRigidConvexFeatureKey(
			0x43465854u, bestTriangle,
			bestClosest.feature,
			bestClosest.featureIndex);
	}

	const PxReal normalMagnitudeSq =
		result.normalLocal.magnitudeSquared();
	if(!result.surfaceLocal.isFinite() ||
		!result.normalLocal.isFinite() ||
		normalMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(result.signedDistance))
		return false;
	result.normalLocal *= PxRecipSqrt(normalMagnitudeSq);
	return true;
}

// P5.7a candidate leaf: current-pose convex SDF is particle-major, reads
// immutable baked hull topology, and appends only to caller-owned output.
void avbdDetectSoftRigidConvexSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particles && particleBegin <= particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			if(!avbdIsRigidConvexValid(convex))
				continue;
			const PxVec3 worldOffset =
				particle.position - convex.center;
			const PxReal queryRadius =
				convex.localRadius + margin;
			if(worldOffset.magnitudeSquared() >
				queryRadius * queryRadius)
				continue;
			const PxVec3 localPoint =
				convex.rotation.getConjugate().rotate(
					worldOffset);
			AvbdRigidConvexPointQuery query;
			if(!avbdQueryRigidConvexLocal(
					convex, localPoint, query) ||
				query.signedDistance >= margin)
				continue;
			const PxVec3 normal =
				convex.rotation.rotate(query.normalLocal).
					getNormalized();

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, convex.primitiveKey,
				query.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = query.signedDistance < 0.0f
				? -query.signedDistance
				: PxMax(
					0.0f, margin - query.signedDistance);
			geometry.margin = margin;
			avbdConfigureRigidConvexTarget(
				geometry, convex, convexIndex,
				query.surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					convex.friction,
					convex.frictionCombineMode)
				: PxMax(convex.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

PX_FORCE_INLINE bool avbdGetRigidConvexSweepPose(
	const AvbdRigidConvex& convex,
	PxVec3& centerStart, PxVec3& centerEnd,
	PxQuat& rotationStart, PxQuat& rotationEnd,
	bool& rotationsEquivalent)
{
	if(!avbdIsRigidConvexValid(convex))
		return false;
	const bool kinematicTarget =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	const bool dynamicTarget =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY;
	if(convex.targetKind !=
			AvbdSoftContactTargetKind::eWORLD_STATIC &&
		!kinematicTarget && !dynamicTarget)
		return false;
	if(kinematicTarget &&
		(!convex.previousCenter.isFinite() ||
		 !convex.previousRotation.isFinite()))
		return false;
	if(dynamicTarget &&
		(!convex.predictedPoseValid ||
		 !convex.predictedCenter.isFinite() ||
		 !convex.predictedRotation.isFinite()))
		return false;

	centerStart =
		kinematicTarget ? convex.previousCenter : convex.center;
	centerEnd =
		dynamicTarget ? convex.predictedCenter : convex.center;
	rotationStart =
		kinematicTarget ? convex.previousRotation : convex.rotation;
	rotationEnd =
		dynamicTarget ? convex.predictedRotation : convex.rotation;
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite())
		return false;
	rotationsEquivalent = avbdAreSweepRotationsEquivalent(
		rotationStart, rotationEnd);
	return true;
}

// The convex remains inside localRadius about its swept center at every pose.
// Reject only when the complete relative point segment misses that sphere's
// axis-aligned outer bound, leaving exact SDF/TOI ownership unchanged.
PX_FORCE_INLINE bool avbdSweptPointMayReachRigidConvexBound(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	PxReal expandedRadius)
{
	if(!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite())
		return false;
	// An invalid bound cannot authorize rejection; the exact query keeps its
	// existing input validation and remains the conservative fallback.
	if(expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return true;
	const PxVec3 relativeStart = pointStart - centerStart;
	const PxVec3 relativeEnd = pointEnd - centerEnd;
	const PxVec3 relativeMinimum =
		relativeStart.minimum(relativeEnd);
	const PxVec3 relativeMaximum =
		relativeStart.maximum(relativeEnd);
	return relativeMinimum.x <= expandedRadius &&
		relativeMaximum.x >= -expandedRadius &&
		relativeMinimum.y <= expandedRadius &&
		relativeMaximum.y >= -expandedRadius &&
		relativeMinimum.z <= expandedRadius &&
		relativeMaximum.z >= -expandedRadius;
}

// Continuous point entry into a fixed-orientation convex expanded by margin.
// The exact closed-hull point distance is 1-Lipschitz, so gap/speed is a
// conservative advancement step and cannot cross first contact.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedConvex(
	const AvbdRigidConvex& convex,
	const PxVec3& segmentStartLocal,
	const PxVec3& segmentEndLocal,
	PxReal margin, AvbdSweptConvexPointEntry& result,
	const AvbdRigidConvexPointQuery* initialQuery = NULL)
{
	if(!segmentStartLocal.isFinite() ||
		!segmentEndLocal.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	const PxVec3 direction =
		segmentEndLocal - segmentStartLocal;
	const PxReal speedSq = direction.magnitudeSquared();
	if(speedSq <= 1.0e-12f || !PxIsFinite(speedSq))
		return false;
	const PxReal speed = PxSqrt(speedSq);
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	AvbdRigidConvexPointQuery currentQuery;
	if(initialQuery)
		currentQuery = *initialQuery;
	else if(!avbdQueryRigidConvexLocal(
			convex, segmentStartLocal, currentQuery))
		return false;
	if(currentQuery.signedDistance < margin)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		AvbdRigidConvexPointQuery query;
		if(iteration == 0)
			query = currentQuery;
		else if(!avbdQueryRigidConvexLocal(
				convex,
				segmentStartLocal + direction * time,
				query))
			return false;
		const PxReal gap = query.signedDistance - margin;
		if(gap <= distanceTolerance)
		{
			result.entryTime = time;
			result.normalLocal = query.normalLocal;
			result.surfaceLocal = query.surfaceLocal;
			result.featureKey = query.featureKey;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous point entry against a translating/rotating convex. The exact
// local closed-hull point distance is sampled at the shortest-path slerped
// pose. Relative point/center translation plus localRadius*angularDistance
// bounds the Hausdorff speed of the hull, so gap/speed cannot step across
// first contact.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedRotatingConvex(
	const AvbdRigidConvex& convex,
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal margin, AvbdSweptConvexPointEntry& result,
	const AvbdRigidConvexPointQuery* initialQuery = NULL)
{
	if(!avbdIsRigidConvexValid(convex) ||
		!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance) ||
		angularDistance <= 0.0f)
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 relativeTranslation =
		(pointEnd - pointStart) - (centerEnd - centerStart);
	const PxReal speed =
		relativeTranslation.magnitude() +
		convex.localRadius * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + (pointEnd - pointStart) * time;
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!point.isFinite() || !center.isFinite() ||
			!rotation.isFinite())
			return false;
		const PxVec3 localPoint =
			rotation.getConjugate().rotate(point - center);
		AvbdRigidConvexPointQuery query;
		if(iteration == 0 && initialQuery)
			query = *initialQuery;
		else if(!avbdQueryRigidConvexLocal(
				convex, localPoint, query))
			return false;
		if(iteration == 0 && query.signedDistance < margin)
			return false;
		const PxReal gap = query.signedDistance - margin;
		if(gap <= distanceTolerance)
		{
			result.entryTime = time;
			result.normalLocal = query.normalLocal;
			result.surfaceLocal = query.surfaceLocal;
			result.featureKey = query.featureKey;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// P5.15a candidate leaf: swept convex SDF is particle-major. Convex topology
// is immutable here and every conservative-advancement/query object is local
// to one particle/convex evaluation. A partitioned caller must merge the full
// current-SDF family before stable-merging any swept-family range.
void avbdDetectSoftRigidConvexSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle =
			particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite() ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidConvexSweepPose(
					convex, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;
			if(!avbdSweptPointMayReachRigidConvexBound(
					particle.position,
					particle.predictedPosition,
					centerStart, centerEnd,
					convex.localRadius + margin))
				continue;
			AvbdSweptConvexPointEntry entry;
			PxQuat entryRotation(PxIdentity);
			if(rotationsEquivalent)
			{
				const PxQuat inverseRotation =
					rotationEnd.getConjugate();
				const PxVec3 relativeStart =
					inverseRotation.rotate(
						particle.position - centerStart);
				const PxVec3 relativeEnd =
					inverseRotation.rotate(
						particle.predictedPosition - centerEnd);
				if(!avbdSegmentEnterExpandedConvex(
						convex, relativeStart, relativeEnd,
						margin, entry))
					continue;
				entryRotation = rotationEnd.getNormalized();
			}
			else
			{
				if(!avbdSegmentEnterExpandedRotatingConvex(
						convex,
						particle.position,
						particle.predictedPosition,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						margin, entry))
					continue;
				entryRotation = PxSlerp(
					entry.entryTime,
					rotationStart.getNormalized(),
					rotationEnd.getNormalized()).getNormalized();
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, convex.primitiveKey,
				entry.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal =
				entryRotation.rotate(entry.normalLocal).
					getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			avbdConfigureRigidConvexTarget(
				geometry, convex, convexIndex,
				entry.surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					convex.friction,
					convex.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1.0e6f, 1.0e6f,
				particles, contacts);
		}
	}
}

PX_FORCE_INLINE bool avbdRigidConvexForwardVertexOwnsSweptFeature(
	const AvbdSoftParticle& particle,
	const AvbdRigidConvex& convex,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxQuat& inverseRotationEnd,
	bool rotationsEquivalent, PxReal margin)
{
	if(particle.invMass <= 0.0f)
		return false;
	AvbdRigidConvexPointQuery currentQuery;
	AvbdSweptConvexPointEntry vertexEntry;
	const PxVec3 pointStart = particle.initialPosition;
	const PxVec3 pointEnd = particle.predictedPosition;
	if(!avbdSweptPointMayReachRigidConvexBound(
			pointStart, pointEnd, centerStart, centerEnd,
			convex.localRadius + margin))
		return false;
	const PxVec3 relativeStart =
		rotationStart.getConjugate().rotate(
			pointStart - centerStart);
	if(!avbdQueryRigidConvexLocal(
			convex, relativeStart, currentQuery))
		return false;
	if(currentQuery.signedDistance < margin)
		return true;
	if(rotationsEquivalent)
	{
		const PxVec3 relativeEnd =
			inverseRotationEnd.rotate(pointEnd - centerEnd);
		return avbdSegmentEnterExpandedConvex(
			convex, relativeStart, relativeEnd,
			margin, vertexEntry, &currentQuery);
	}
	return avbdSegmentEnterExpandedRotatingConvex(
		convex, pointStart, pointEnd,
		centerStart, centerEnd,
		rotationStart, rotationEnd,
		margin, vertexEntry, &currentQuery);
}

void avbdDetectSoftRigidConvexSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	PxArray<PxU8>* persistentForwardOwnerScratch)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8> localForwardOwnerScratch;
	PxArray<PxU8>& forwardOwnerScratch =
		persistentForwardOwnerScratch
			? *persistentForwardOwnerScratch
			: localForwardOwnerScratch;
	if(forwardOwnerScratch.capacity() < numParticles)
		forwardOwnerScratch.reserve(numParticles);
	forwardOwnerScratch.resize(numParticles);
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			PxArray<AvbdRigidConvexEdgeBounds>& edgeBoundsScratch =
				convex.edgeBoundsScratch;
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidConvexSweepPose(
					convex, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;
			const PxQuat inverseRotation =
				rotationEnd.getConjugate();
			// Broadphase rejection is invariant under a common frame change. Cache
			// each rigid edge's world-space swept AABB once; the exact kernels
			// below retain their original relative-motion coordinates and TOI.
			const PxVec3 rigidTranslation =
				centerEnd - centerStart;
			const PxReal sweptRotationExtent =
				convex.localRadius + margin;
			const PxVec3 rotatingMinimum =
				centerStart.minimum(centerEnd) -
					PxVec3(sweptRotationExtent);
			const PxVec3 rotatingMaximum =
				centerStart.maximum(centerEnd) +
					PxVec3(sweptRotationExtent);
			if(edgeBoundsScratch.capacity() < convex.edges.size())
				edgeBoundsScratch.reserve(convex.edges.size());
			edgeBoundsScratch.resize(convex.edges.size());
			for(PxU32 rigidEdgeIndex = 0;
				rigidEdgeIndex < convex.edges.size(); ++rigidEdgeIndex)
			{
				const AvbdRigidConvexEdge& rigidEdge =
					convex.edges[rigidEdgeIndex];
				if(rigidEdge.p0 >= convex.vertices.size() ||
					rigidEdge.p1 >= convex.vertices.size())
					continue;
				AvbdRigidConvexEdgeBounds& edgeBounds =
					edgeBoundsScratch[rigidEdgeIndex];
				edgeBounds.point0 =
					centerStart + rotationStart.rotate(
						convex.vertices[rigidEdge.p0]);
				edgeBounds.point1 =
					centerStart + rotationStart.rotate(
						convex.vertices[rigidEdge.p1]);
				const PxVec3 edgeMinimum =
					edgeBounds.point0.minimum(edgeBounds.point1);
				const PxVec3 edgeMaximum =
					edgeBounds.point0.maximum(edgeBounds.point1);
				edgeBounds.minimum = rotationsEquivalent
					? edgeMinimum.minimum(
						edgeMinimum + rigidTranslation) -
							PxVec3(margin)
					: rotatingMinimum;
				edgeBounds.maximum = rotationsEquivalent
					? edgeMaximum.maximum(
						edgeMaximum + rigidTranslation) +
							PxVec3(margin)
					: rotatingMaximum;
			}
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleCount = body.compiled.particleCount;
			if(particleStart <= numParticles)
			{
				const PxU32 boundedParticleCount = PxMin(
					particleCount, numParticles - particleStart);
				for(PxU32 localParticle = 0;
					localParticle < boundedParticleCount; ++localParticle)
					forwardOwnerScratch[
						particleStart + localParticle] = 0;
			}
			for(PxU32 surfaceVertexIndex = 0;
				surfaceVertexIndex < body.compiled.surfaceVertices.size();
				++surfaceVertexIndex)
			{
				const PxU32 vertexIndex =
					body.compiled.surfaceVertices[surfaceVertexIndex];
				if(vertexIndex >= numParticles)
					continue;
				forwardOwnerScratch[vertexIndex] = PxU8(
					avbdRigidConvexForwardVertexOwnsSweptFeature(
						particles[vertexIndex], convex,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						inverseRotation, rotationsEquivalent,
						margin));
			}

			// Convex edge versus a linearly deforming soft edge. The
			// translation-only kernel remains the zero-relative-motion
			// fast path.
			for(PxU32 softEdgeIndex = 0;
				softEdgeIndex <
					body.compiled.surfaceEdges.size();
				++softEdgeIndex)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0 =
					particles[softEdge.p0].initialPosition;
				const PxVec3 soft1 =
					particles[softEdge.p1].initialPosition;
				const PxVec3 predicted0 =
					particles[softEdge.p0].predictedPosition;
				const PxVec3 predicted1 =
					particles[softEdge.p1].predictedPosition;
				const PxVec3 displacement0 =
					predicted0 - soft0;
				const PxVec3 displacement1 =
					predicted1 - soft1;
				if(!soft0.isFinite() || !soft1.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite())
					continue;
				const PxVec3 relativeSoftDisplacement1 =
					displacement1 - displacement0;
				const bool softEdgeTranslationOnly =
					relativeSoftDisplacement1.
						magnitudeSquared() <=
							translationToleranceSq;

				const bool forwardVertexOwns =
					forwardOwnerScratch[softEdge.p0] != 0 ||
					forwardOwnerScratch[softEdge.p1] != 0;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				if(softEdgeTranslationOnly &&
					rotationsEquivalent &&
					relativeTranslation.magnitudeSquared() <=
						translationToleranceSq)
					continue;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				const PxVec3 soft0End = soft0;
				const PxVec3 soft1End =
					soft1 + relativeSoftDisplacement1;
				const PxVec3 sweptSoftMinimum =
					soft0.minimum(soft1).
						minimum(predicted0).
						minimum(predicted1);
				const PxVec3 sweptSoftMaximum =
					soft0.maximum(soft1).
						maximum(predicted0).
						maximum(predicted1);
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < convex.edges.size();
					++rigidEdgeIndex)
				{
					const AvbdRigidConvexEdge& rigidEdge =
						convex.edges[rigidEdgeIndex];
					if(rigidEdge.p0 >= convex.vertices.size() ||
						rigidEdge.p1 >= convex.vertices.size())
						continue;
					const AvbdRigidConvexEdgeBounds& edgeBounds =
						edgeBoundsScratch[rigidEdgeIndex];
					const PxVec3& rigid0 = edgeBounds.point0;
					const PxVec3& rigid1 = edgeBounds.point1;
					const PxVec3& rigidMinimum = edgeBounds.minimum;
					const PxVec3& rigidMaximum = edgeBounds.maximum;
					if(rigidMinimum.x > sweptSoftMaximum.x ||
						rigidMaximum.x < sweptSoftMinimum.x ||
						rigidMinimum.y > sweptSoftMaximum.y ||
						rigidMaximum.y < sweptSoftMinimum.y ||
						rigidMinimum.z > sweptSoftMaximum.z ||
						rigidMaximum.z < sweptSoftMinimum.z)
						continue;
					AvbdSweptConvexEdgeEntry entry;
					const bool entered =
						softEdgeTranslationOnly &&
							rotationsEquivalent
							? avbdTranslatedSegmentEnterExpandedSegmentInteriors(
								rigid0, rigid1,
								relativeTranslation,
								soft0, soft1, margin, entry)
							: softEdgeTranslationOnly
							? avbdRotatingSegmentEnterExpandedSegmentInteriors(
								convex.vertices[rigidEdge.p0],
								convex.vertices[rigidEdge.p1],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1, margin, entry)
							: avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
								convex.vertices[rigidEdge.p0],
								convex.vertices[rigidEdge.p1],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1,
								soft0End, soft1End,
								margin, entry);
					if(!entered)
						continue;
					PxVec3 normal = entry.normal;
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(rigidEdge.outward);
					if(normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43564545u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] =
						softEdge.p0;
					geometry.queryParticleIndices[1] =
						softEdge.p1;
					geometry.queryWeights[0] =
						1.0f - entry.softWeight1;
					geometry.queryWeights[1] =
						entry.softWeight1;
					geometry.normal = normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					const PxVec3 surfaceLocal =
						convex.vertices[rigidEdge.p0] *
							(1.0f - entry.rigidWeight1) +
						convex.vertices[rigidEdge.p1] *
							entry.rigidWeight1;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						surfaceLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
			}

			// Convex vertex versus a linearly deforming soft face. The
			// translation-only kernel remains the zero-relative-motion fast
			// path. Any forward soft-vertex owner suppresses the complete
			// triangle candidate.
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[
						triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;

				const bool forwardVertexOwns =
					forwardOwnerScratch[v0] != 0 ||
					forwardOwnerScratch[v1] != 0 ||
					forwardOwnerScratch[v2] != 0;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				if(softTriangleTranslationOnly &&
					rotationsEquivalent &&
					relativeTranslation.magnitudeSquared() <=
						translationToleranceSq)
					continue;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2).
						minimum(p1 + relativeDisplacement1).
						minimum(p2 + relativeDisplacement2) -
						PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2).
						maximum(p1 + relativeDisplacement1).
						maximum(p2 + relativeDisplacement2) +
						PxVec3(margin);
				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < convex.vertices.size();
					++rigidVertexIndex)
				{
					const PxVec3 rigidVertexStart =
						centerStart + rotationStart.rotate(
							convex.vertices[
								rigidVertexIndex]);
					const PxVec3 rigidVertexEnd =
						rotationsEquivalent
							? rigidVertexStart + relativeTranslation
							: relativeCenterEnd +
								rotationEnd.rotate(
									convex.vertices[
										rigidVertexIndex]);
					PxVec3 sweptMinimum(0.0f);
					PxVec3 sweptMaximum(0.0f);
					if(rotationsEquivalent)
					{
						sweptMinimum =
							rigidVertexStart.minimum(
								rigidVertexEnd);
						sweptMaximum =
							rigidVertexStart.maximum(
								rigidVertexEnd);
					}
					else
					{
						const PxReal rotationExtent =
							convex.localRadius;
						sweptMinimum =
							centerStart.minimum(relativeCenterEnd) -
								PxVec3(rotationExtent);
						sweptMaximum =
							centerStart.maximum(relativeCenterEnd) +
								PxVec3(rotationExtent);
					}
					if(sweptMinimum.x > triangleMaximum.x ||
						sweptMaximum.x < triangleMinimum.x ||
						sweptMinimum.y > triangleMaximum.y ||
						sweptMaximum.y < triangleMinimum.y ||
						sweptMinimum.z > triangleMaximum.z ||
						sweptMaximum.z < triangleMinimum.z)
						continue;
					AvbdSweptTriangleEntry entry;
					const bool entered =
						softTriangleTranslationOnly &&
							rotationsEquivalent
							? avbdSegmentEnterExpandedTriangleNonVertex(
								rigidVertexStart, rigidVertexEnd,
								p0, p1, p2, margin, entry)
							: softTriangleTranslationOnly
							? avbdRotatingPointEnterExpandedTriangleFace(
								convex.vertices[rigidVertexIndex],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, margin, entry)
							: avbdRotatingPointEnterExpandedDeformingTriangleFace(
								convex.vertices[rigidVertexIndex],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, p0,
								p1 + relativeDisplacement1,
								p2 + relativeDisplacement2,
								margin, entry);
					if(!entered ||
						entry.feature != AVBD_FEATURE_FACE)
						continue;
					PxVec3 outwardLocal =
						rigidVertexIndex <
								convex.vertexNormals.size()
							? convex.vertexNormals[
								rigidVertexIndex]
							: convex.vertices[
								rigidVertexIndex];
					if(!outwardLocal.isFinite() ||
						outwardLocal.magnitudeSquared() <=
							1.0e-12f)
						outwardLocal =
							PxVec3(0.0f, 1.0f, 0.0f);
					outwardLocal.normalize();
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(outwardLocal);
					PxVec3 normal = entry.normal;
					if(normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43565646u,
							v0, v1, v2,
							rigidVertexIndex));
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						entry.barycentric.x;
					geometry.queryWeights[1] =
						entry.barycentric.y;
					geometry.queryWeights[2] =
						entry.barycentric.z;
					geometry.normal = normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						convex.vertices[rigidVertexIndex]);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
			}
		}
	}
}

void avbdDetectSoftRigidConvexOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin)
{
	const PxReal featureEpsilon = 1.0e-4f;
	const PxReal distanceEpsilon = 1.0e-8f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			bodyMinimum = bodyMinimum.minimum(
				particles[particleIndex].position);
			bodyMaximum = bodyMaximum.maximum(
				particles[particleIndex].position);
		}

		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			PxArray<AvbdRigidConvexEdgeBounds>& edgeBoundsScratch =
				convex.edgeBoundsScratch;
			if(!avbdIsRigidConvexValid(convex))
				continue;
			const PxReal broadphaseRadius =
				convex.localRadius + margin;
			if(bodyMinimum.x >
					convex.center.x + broadphaseRadius ||
				bodyMaximum.x <
					convex.center.x - broadphaseRadius ||
				bodyMinimum.y >
					convex.center.y + broadphaseRadius ||
				bodyMaximum.y <
					convex.center.y - broadphaseRadius ||
				bodyMinimum.z >
					convex.center.z + broadphaseRadius ||
				bodyMaximum.z <
					convex.center.z - broadphaseRadius)
				continue;
			const PxQuat inverseRotation =
				convex.rotation.getConjugate();
			if(edgeBoundsScratch.capacity() < convex.edges.size())
				edgeBoundsScratch.reserve(convex.edges.size());
			edgeBoundsScratch.resize(convex.edges.size());
			for(PxU32 rigidEdgeIndex = 0;
				rigidEdgeIndex < convex.edges.size(); ++rigidEdgeIndex)
			{
				const AvbdRigidConvexEdge& rigidEdge =
					convex.edges[rigidEdgeIndex];
				if(rigidEdge.p0 >= convex.vertices.size() ||
					rigidEdge.p1 >= convex.vertices.size())
					continue;
				AvbdRigidConvexEdgeBounds& edgeBounds =
					edgeBoundsScratch[rigidEdgeIndex];
				edgeBounds.point0 = convex.vertices[rigidEdge.p0];
				edgeBounds.point1 = convex.vertices[rigidEdge.p1];
				edgeBounds.minimum =
					edgeBounds.point0.minimum(edgeBounds.point1) -
						PxVec3(margin);
				edgeBounds.maximum =
					edgeBounds.point0.maximum(edgeBounds.point1) +
						PxVec3(margin);
			}

			// Convex boundary edge versus soft boundary edge. Endpoint cases
			// remain owned by forward vertex-SDF or reverse vertex-face.
			for(PxU32 softEdgeIndex = 0;
				softEdgeIndex <
					body.compiled.surfaceEdges.size();
				++softEdgeIndex)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0Local =
					inverseRotation.rotate(
						particles[softEdge.p0].position -
							convex.center);
				const PxVec3 soft1Local =
					inverseRotation.rotate(
						particles[softEdge.p1].position -
							convex.center);
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < convex.edges.size();
					++rigidEdgeIndex)
				{
					const AvbdRigidConvexEdge& rigidEdge =
						convex.edges[rigidEdgeIndex];
					if(rigidEdge.p0 >= convex.vertices.size() ||
						rigidEdge.p1 >= convex.vertices.size())
						continue;
					const AvbdRigidConvexEdgeBounds& edgeBounds =
						edgeBoundsScratch[rigidEdgeIndex];
					const PxVec3& rigid0Local = edgeBounds.point0;
					const PxVec3& rigid1Local = edgeBounds.point1;
					const PxVec3 softMinimum =
						soft0Local.minimum(soft1Local);
					const PxVec3 softMaximum =
						soft0Local.maximum(soft1Local);
					const PxVec3& rigidMinimum = edgeBounds.minimum;
					const PxVec3& rigidMaximum = edgeBounds.maximum;
					if(softMinimum.x > rigidMaximum.x ||
						softMaximum.x < rigidMinimum.x ||
						softMinimum.y > rigidMaximum.y ||
						softMaximum.y < rigidMinimum.y ||
						softMinimum.z > rigidMaximum.z ||
						softMaximum.z < rigidMinimum.z)
						continue;
					PxReal softWeight1 = 0.0f;
					PxReal rigidWeight1 = 0.0f;
					PxVec3 softClosestLocal;
					PxVec3 rigidClosestLocal;
					avbdClosestPointsOnSegments(
						soft0Local, soft1Local,
						rigid0Local, rigid1Local,
						softWeight1, rigidWeight1,
						softClosestLocal, rigidClosestLocal);
					if(softWeight1 <= featureEpsilon ||
						softWeight1 >=
							1.0f - featureEpsilon ||
						rigidWeight1 <= featureEpsilon ||
						rigidWeight1 >=
							1.0f - featureEpsilon)
						continue;
					PxVec3 deltaLocal =
						softClosestLocal - rigidClosestLocal;
					const PxReal distance =
						deltaLocal.magnitude();
					if(!PxIsFinite(distance) ||
						distance >= margin)
						continue;
					PxVec3 normalLocal =
						distance > distanceEpsilon
							? deltaLocal * (1.0f / distance)
							: rigidEdge.outward;
					if(normalLocal.dot(rigidEdge.outward) < 0.0f)
						normalLocal = -normalLocal;
					if(!normalLocal.isFinite() ||
						normalLocal.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43564545u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] =
						softEdge.p0;
					geometry.queryParticleIndices[1] =
						softEdge.p1;
					geometry.queryWeights[0] =
						1.0f - softWeight1;
					geometry.queryWeights[1] =
						softWeight1;
					geometry.normal =
						convex.rotation.rotate(normalLocal).
							getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - distance;
					geometry.margin = margin;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						rigidClosestLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}

			// Convex vertex versus soft face. Soft vertex/edge closest
			// features are excluded so the forward/edge-edge paths retain
			// unique physical feature ownership.
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[
						triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 p0Local =
					inverseRotation.rotate(p0 - convex.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - convex.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - convex.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local) -
						PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local) +
						PxVec3(margin);

				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < convex.vertices.size();
					++rigidVertexIndex)
				{
					const PxVec3& rigidVertexLocal =
						convex.vertices[rigidVertexIndex];
					if(rigidVertexLocal.x < triangleMinimum.x ||
						rigidVertexLocal.x > triangleMaximum.x ||
						rigidVertexLocal.y < triangleMinimum.y ||
						rigidVertexLocal.y > triangleMaximum.y ||
						rigidVertexLocal.z < triangleMinimum.z ||
						rigidVertexLocal.z > triangleMaximum.z)
						continue;
					const PxVec3 rigidVertexWorld =
						convex.center +
						convex.rotation.rotate(
							rigidVertexLocal);
					const AvbdClosestPointResult closest =
						avbdClosestPointOnTriangleOGC(
							rigidVertexWorld, p0, p1, p2);
					if(closest.feature != AVBD_FEATURE_FACE ||
						!PxIsFinite(closest.distance) ||
						closest.distance >= margin)
						continue;
					PxVec3 outwardLocal =
						rigidVertexIndex <
								convex.vertexNormals.size()
							? convex.vertexNormals[
								rigidVertexIndex]
							: rigidVertexLocal;
					if(!outwardLocal.isFinite() ||
						outwardLocal.magnitudeSquared() <=
							1.0e-12f)
						outwardLocal =
							PxVec3(0.0f, 1.0f, 0.0f);
					outwardLocal.normalize();
					const PxVec3 outwardWorld =
						convex.rotation.rotate(outwardLocal);
					PxVec3 normalWorld =
						closest.distance > distanceEpsilon
							? (closest.point -
								rigidVertexWorld) *
								(1.0f / closest.distance)
							: outwardWorld;
					if(normalWorld.dot(outwardWorld) < 0.0f)
						normalWorld = -normalWorld;
					if(!normalWorld.isFinite() ||
						normalWorld.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43565646u,
							v0, v1, v2,
							rigidVertexIndex));
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						closest.barycentric.x;
					geometry.queryWeights[1] =
						closest.barycentric.y;
					geometry.queryWeights[2] =
						closest.barycentric.z;
					geometry.normal =
						normalWorld.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth =
						margin - closest.distance;
					geometry.margin = margin;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						rigidVertexLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}
		}
	}
}

// P5.17a permits a caller-owned override for every query write in both
// triangle OGC feature suffixes. It is intentionally only a serial-equivalence
// contract: a future range leaf must still preserve body/surface/edge/face
// family order rather than treating the two feature loops as one flat stream.
// Reverse OGC completion for translating/rotating triangle surfaces. Active
// rigid edges sweep against linearly deforming soft edge interiors and active
// rigid vertices sweep against linearly deforming soft face interiors. The
// translation-only kernels remain zero-relative-motion fast paths. A current
// or swept forward soft-vertex owner suppresses each candidate.
void avbdDetectSoftRigidOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	AvbdOgcGeometryEpochSidecar* geometrySidecar)
{
	if(geometrySidecar)
		geometrySidecar->resizeContactMapping(contacts.size());
	const PxReal featureEpsilon = 1e-4f;
	const PxReal distanceEpsilon = 1e-8f;

	auto configureRigidTarget = [](
		AvbdSoftContactGeometry& geometry,
		const AvbdRigidBox& box, PxU32 boxIndex,
		const PxVec3& surfaceLocal)
	{
		geometry.targetKind = box.targetKind;
		geometry.velocityOwner =
			box.targetKind == AvbdSoftContactTargetKind::eKINEMATIC_RIGID
				? AvbdVelocityObjectiveOwner::ComponentFinalize
				: box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
					? AvbdVelocityObjectiveOwner::ManifoldFinalize
					: AvbdVelocityObjectiveOwner::PositionAL;
		geometry.targetIndex =
			box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY
				? box.targetIndex : boxIndex;
		geometry.surfacePoint =
			box.center + box.rotation.rotate(surfaceLocal);
		geometry.kinematicSurfacePointPrevious =
			box.targetKind == AvbdSoftContactTargetKind::eKINEMATIC_RIGID
				? box.previousCenter +
					box.previousRotation.rotate(surfaceLocal)
				: geometry.surfacePoint;
		if(box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY)
			geometry.rigidLocalPoint =
				box.shapeToRigidBody.transform(surfaceLocal);
	};

	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			localParticle++)
		{
			const PxVec3& position =
				particles[
					body.compiled.particleStart + localParticle].
					position;
			bodyMinimum = bodyMinimum.minimum(position);
			bodyMaximum = bodyMaximum.maximum(position);
		}
		for(PxU32 boxIndex = 0; boxIndex < numBoxes; boxIndex++)
		{
			const AvbdRigidBox& box = boxes[boxIndex];
			if(box.halfExtent.x <= 0.0f &&
				box.halfExtent.y <= 0.0f &&
				box.halfExtent.z <= 0.0f)
				continue;
			const PxReal boxRadius =
				box.halfExtent.magnitude() + margin;
			const PxVec3 broadphaseExtent(boxRadius);
			if(bodyMinimum.x > box.center.x + broadphaseExtent.x ||
				bodyMaximum.x < box.center.x - broadphaseExtent.x ||
				bodyMinimum.y > box.center.y + broadphaseExtent.y ||
				bodyMaximum.y < box.center.y - broadphaseExtent.y ||
				bodyMinimum.z > box.center.z + broadphaseExtent.z ||
				bodyMaximum.z < box.center.z - broadphaseExtent.z)
				continue;
			const PxQuat inverseRotation = box.rotation.getConjugate();

			// OGC edge-edge blocks: closest points must lie in the interiors
			// of both edges. Endpoint cases are owned by the adjacent
			// vertex/face blocks and are intentionally excluded.
			for(PxU32 softEdgeIndex = 0;
				softEdgeIndex < body.compiled.surfaceEdges.size();
				softEdgeIndex++)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0Local = inverseRotation.rotate(
					particles[softEdge.p0].position - box.center);
				const PxVec3 soft1Local = inverseRotation.rotate(
					particles[softEdge.p1].position - box.center);
				const PxVec3 softMinimum =
					soft0Local.minimum(soft1Local);
				const PxVec3 softMaximum =
					soft0Local.maximum(soft1Local);
				const PxVec3 expandedHalfExtent =
					box.halfExtent + PxVec3(margin);
				if(softMinimum.x > expandedHalfExtent.x ||
					softMaximum.x < -expandedHalfExtent.x ||
					softMinimum.y > expandedHalfExtent.y ||
					softMaximum.y < -expandedHalfExtent.y ||
					softMinimum.z > expandedHalfExtent.z ||
					softMaximum.z < -expandedHalfExtent.z)
					continue;

				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < 12; rigidEdgeIndex++)
				{
					PxVec3 rigid0Local, rigid1Local, outwardLocal;
					avbdGetRigidBoxEdgeLocal(
						box.halfExtent, rigidEdgeIndex,
						rigid0Local, rigid1Local, outwardLocal);
					const PxVec3 rigidMinimum =
						rigid0Local.minimum(rigid1Local) -
						PxVec3(margin);
					const PxVec3 rigidMaximum =
						rigid0Local.maximum(rigid1Local) +
						PxVec3(margin);
					if(softMinimum.x > rigidMaximum.x ||
						softMaximum.x < rigidMinimum.x ||
						softMinimum.y > rigidMaximum.y ||
						softMaximum.y < rigidMinimum.y ||
						softMinimum.z > rigidMaximum.z ||
						softMaximum.z < rigidMinimum.z)
						continue;
					PxReal softWeight1 = 0.0f;
					PxReal rigidWeight1 = 0.0f;
					PxVec3 softClosestLocal, rigidClosestLocal;
					avbdClosestPointsOnSegments(
						soft0Local, soft1Local,
						rigid0Local, rigid1Local,
						softWeight1, rigidWeight1,
						softClosestLocal, rigidClosestLocal);
					if(softWeight1 <= featureEpsilon ||
						softWeight1 >= 1.0f - featureEpsilon ||
						rigidWeight1 <= featureEpsilon ||
						rigidWeight1 >= 1.0f - featureEpsilon)
						continue;

					PxVec3 deltaLocal =
						softClosestLocal - rigidClosestLocal;
					const PxReal distance = deltaLocal.magnitude();
					if(distance >= margin)
						continue;
					PxVec3 normalLocal = distance > distanceEpsilon
						? deltaLocal * (1.0f / distance)
						: outwardLocal;
					if(normalLocal.dot(outwardLocal) < 0.0f)
						normalLocal = -normalLocal;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, box.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x45444745u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					// particleIdx remains the representative used by the
					// rigid/soft routing code.  Prefer a movable endpoint so
					// an edge incident to a pinned cloth vertex is not
					// mistaken for a wholly kinematic shell contact.
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] = softEdge.p0;
					geometry.queryParticleIndices[1] = softEdge.p1;
					geometry.queryWeights[0] = 1.0f - softWeight1;
					geometry.queryWeights[1] = softWeight1;
					geometry.normal =
						box.rotation.rotate(normalLocal).getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - distance;
					geometry.margin = margin;
					configureRigidTarget(
						geometry, box, boxIndex, rigidClosestLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							box.friction, box.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}

			// Reverse vertex-facet blocks: a rigid box vertex can approach or
			// cross the interior of a cloth triangle while every cloth vertex
			// remains outside the box SDF. Store the closest cloth point as a
			// barycentric query so its response is distributed to the face.
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 p0Local =
					inverseRotation.rotate(p0 - box.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - box.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - box.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local) -
					PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local) +
					PxVec3(margin);

				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < 8; rigidVertexIndex++)
				{
					const PxVec3 rigidVertexLocal =
						avbdGetRigidBoxVertexLocal(
							box.halfExtent, rigidVertexIndex);
					if(rigidVertexLocal.x < triangleMinimum.x ||
						rigidVertexLocal.x > triangleMaximum.x ||
						rigidVertexLocal.y < triangleMinimum.y ||
						rigidVertexLocal.y > triangleMaximum.y ||
						rigidVertexLocal.z < triangleMinimum.z ||
						rigidVertexLocal.z > triangleMaximum.z)
						continue;
					const PxVec3 rigidVertexWorld =
						box.center +
						box.rotation.rotate(rigidVertexLocal);
					const AvbdClosestPointResult closest =
						avbdClosestPointOnTriangleOGC(
							rigidVertexWorld, p0, p1, p2);
					if(closest.feature != AVBD_FEATURE_FACE ||
						closest.distance >= margin)
						continue;

					const PxVec3 outwardLocal =
						rigidVertexLocal.getNormalized();
					PxVec3 normalWorld;
					if(closest.distance > distanceEpsilon)
						normalWorld =
							(closest.point - rigidVertexWorld) *
							(1.0f / closest.distance);
					else
						normalWorld =
							box.rotation.rotate(outwardLocal);
					const PxVec3 outwardWorld =
						box.rotation.rotate(outwardLocal);
					if(normalWorld.dot(outwardWorld) < 0.0f)
						normalWorld = -normalWorld;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, box.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x56464143u, v0, v1, v2,
							rigidVertexIndex));
					// As above, keep the representative on a movable query
					// vertex whenever the triangle is not fully prescribed.
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f ? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						closest.barycentric.x;
					geometry.queryWeights[1] =
						closest.barycentric.y;
					geometry.queryWeights[2] =
						closest.barycentric.z;
					geometry.normal = normalWorld.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - closest.distance;
					geometry.margin = margin;
					configureRigidTarget(
						geometry, box, boxIndex,
						rigidVertexLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							box.friction, box.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}

				// Triangle/box core block.  The vertex-SDF, edge-edge and
				// reverse vertex-face blocks above deliberately own their normal
				// cases.  They still leave one discrete blind spot: a soft
				// collision triangle can pass through an OBB while all three soft
				// vertices and all eight OBB vertices remain outside the opposing
				// primitive.  Clip the triangle against the current box and emit
				// exactly one barycentric face row for a genuinely interior patch.
				// This is current-pose OGC/DCD, not a swept test.
				// Do not limit the clipper to the historical all-vertices-outside
				// blind spot.  A triangle with one vertex inside is still a complete
				// triangle/OBB core overlap, and a single vertex SDF row cannot prove
				// that the rest of its face exits the box in this same DCD step.
				{
					// The terminal verifier accepts up to 1 mm of numerical overlap.
					// Build the core witness against the same eroded OBB so a triangle
					// which merely touches the exterior does not create a repair row,
					// while every visible core penetration has a robust interior
					// barycentric witness rather than a boundary-only centroid.
					const PxReal coreInset = PxMin(1.0e-3f,
						0.25f * PxMin(box.halfExtent.x,
							PxMin(box.halfExtent.y, box.halfExtent.z)));
					const PxVec3 coreHalfExtent = box.halfExtent -
						PxVec3(PxMax(coreInset, 0.0f));
					if(coreHalfExtent.x <= 0.0f || coreHalfExtent.y <= 0.0f ||
						coreHalfExtent.z <= 0.0f)
						continue;
					struct TriangleBoxClipVertex
					{
						PxVec3 point;
						PxVec3 barycentric;
					};
					TriangleBoxClipVertex input[16] =
					{
						{p0Local, PxVec3(1.0f, 0.0f, 0.0f)},
						{p1Local, PxVec3(0.0f, 1.0f, 0.0f)},
						{p2Local, PxVec3(0.0f, 0.0f, 1.0f)}
					};
					TriangleBoxClipVertex output[16];
					PxU32 inputCount = 3;
					bool validClip = true;
					for(PxU32 axis = 0; axis < 3 && validClip;
						++axis)
					{
						for(PxU32 side = 0; side < 2 && validClip;
							++side)
						{
							const PxReal bound =
								side == 0 ? coreHalfExtent[axis] :
									-coreHalfExtent[axis];
							const bool upperBound = side == 0;
							PxU32 outputCount = 0;
							TriangleBoxClipVertex previous =
								input[inputCount - 1];
							bool previousInside = upperBound ?
								previous.point[axis] <= bound :
								previous.point[axis] >= bound;
							for(PxU32 vertex = 0; vertex < inputCount; ++vertex)
							{
								const TriangleBoxClipVertex current = input[vertex];
								const bool currentInside = upperBound ?
									current.point[axis] <= bound :
									current.point[axis] >= bound;
								if(currentInside != previousInside)
								{
									const PxReal denominator =
										current.point[axis] - previous.point[axis];
									if(PxAbs(denominator) <= 1.0e-12f ||
										outputCount >= 16)
									{
										validClip = false;
										break;
									}
									const PxReal t = PxClamp(
										(bound - previous.point[axis]) / denominator,
										0.0f, 1.0f);
									output[outputCount].point =
										previous.point +
										(current.point - previous.point) * t;
									output[outputCount].barycentric =
										previous.barycentric +
										(current.barycentric - previous.barycentric) * t;
									++outputCount;
								}
								if(currentInside)
								{
									if(outputCount >= 16)
									{
										validClip = false;
										break;
									}
									output[outputCount++] = current;
								}
								previous = current;
								previousInside = currentInside;
							}
							if(outputCount == 0)
							{
								validClip = false;
								break;
							}
							inputCount = outputCount;
							for(PxU32 vertex = 0; vertex < inputCount; ++vertex)
								input[vertex] = output[vertex];
						}
					}
					if(validClip && inputCount >= 3)
					{
						PxVec3 clippedPoint(0.0f);
						PxVec3 clippedBarycentric(0.0f);
						for(PxU32 vertex = 0; vertex < inputCount; ++vertex)
						{
							clippedPoint += input[vertex].point;
							clippedBarycentric += input[vertex].barycentric;
						}
						const PxReal reciprocalCount = 1.0f / inputCount;
						clippedPoint *= reciprocalCount;
						clippedBarycentric *= reciprocalCount;
						const PxReal barycentricSum =
							clippedBarycentric.x + clippedBarycentric.y +
							clippedBarycentric.z;
						if(PxIsFinite(barycentricSum) &&
							PxAbs(barycentricSum) > 1.0e-6f)
							clippedBarycentric *= 1.0f / barycentricSum;

						const PxVec3 coreQ(
							PxAbs(clippedPoint.x) - box.halfExtent.x,
							PxAbs(clippedPoint.y) - box.halfExtent.y,
							PxAbs(clippedPoint.z) - box.halfExtent.z);
						const PxReal coreSdf =
							PxMax(coreQ.x, PxMax(coreQ.y, coreQ.z));
						if(clippedPoint.isFinite() &&
							clippedBarycentric.isFinite() &&
							PxIsFinite(coreSdf) && coreSdf < -1.0e-5f)
						{
							// The centroid is the right compact AL query, but it is
							// not a sufficient escape witness for a triangle that
							// crosses the box.  Pick the shortest whole-triangle
							// translation through one OBB face. Once all three source
							// vertices lie beyond that face, their convex hull (the
							// complete triangle) is rigorously separated from the box.
							// Keep this certificate separate from the centroid normal so
							// ordinary Position-AL rows preserve their existing semantics.
							const PxVec3 triangleMin =
								p0Local.minimum(p1Local).minimum(p2Local);
							const PxVec3 triangleMax =
								p0Local.maximum(p1Local).maximum(p2Local);
							const PxReal exitDistances[6] =
							{
								box.halfExtent.x - triangleMin.x,
								triangleMax.x + box.halfExtent.x,
								box.halfExtent.y - triangleMin.y,
								triangleMax.y + box.halfExtent.y,
								box.halfExtent.z - triangleMin.z,
								triangleMax.z + box.halfExtent.z
							};
							const PxVec3 exitNormals[6] =
							{
								PxVec3(1.0f, 0.0f, 0.0f),
								PxVec3(-1.0f, 0.0f, 0.0f),
								PxVec3(0.0f, 1.0f, 0.0f),
								PxVec3(0.0f, -1.0f, 0.0f),
								PxVec3(0.0f, 0.0f, 1.0f),
								PxVec3(0.0f, 0.0f, -1.0f)
							};
							PxReal coreExitDistance = PX_MAX_F32;
							PxVec3 coreExitNormalLocal(0.0f);
							for(PxU32 exitIndex = 0; exitIndex < 6;
								++exitIndex)
							{
								const PxReal candidate = exitDistances[exitIndex];
								if(PxIsFinite(candidate) && candidate >= 0.0f &&
									candidate < coreExitDistance)
								{
									coreExitDistance = candidate;
									coreExitNormalLocal = exitNormals[exitIndex];
								}
							}
							if(!PxIsFinite(coreExitDistance) ||
								!coreExitNormalLocal.isFinite())
								continue;
							PxVec3 normalLocal;
							if(coreQ.x > coreQ.y && coreQ.x > coreQ.z)
								normalLocal = PxVec3(
									clippedPoint.x >= 0.0f ? 1.0f : -1.0f,
									0.0f, 0.0f);
							else if(coreQ.y > coreQ.z)
								normalLocal = PxVec3(0.0f,
									clippedPoint.y >= 0.0f ? 1.0f : -1.0f,
									0.0f);
							else
								normalLocal = PxVec3(0.0f, 0.0f,
									clippedPoint.z >= 0.0f ? 1.0f : -1.0f);
							const PxVec3 surfaceLocal =
								clippedPoint - normalLocal * coreSdf;
							AvbdSoftContactGeometry geometry;
							geometry.source = AvbdSoftContactSource(
								AvbdSoftContactSource::eRIGID_SDF,
								PX_MAX_U32, box.primitiveKey,
								avbdGetRigidSoftFeatureKey(
									0x54424958u, v0, v1, v2, 0u));
							geometry.particleIdx =
								particles[v0].invMass > 0.0f ? v0 :
								(particles[v1].invMass > 0.0f ? v1 : v2);
							geometry.queryParticleIndices[0] = v0;
							geometry.queryParticleIndices[1] = v1;
							geometry.queryParticleIndices[2] = v2;
							geometry.queryWeights[0] = clippedBarycentric.x;
							geometry.queryWeights[1] = clippedBarycentric.y;
							geometry.queryWeights[2] = clippedBarycentric.z;
							geometry.normal =
								box.rotation.rotate(normalLocal).getNormalized();
							geometry.projNormal = geometry.normal;
							geometry.depth = -coreSdf;
							geometry.margin = margin;
							configureRigidTarget(
								geometry, box, boxIndex, surfaceLocal);
							geometry.friction = avbdCombineDeformableRigidFriction(
								body.material.dynamicFriction, box.friction,
								box.frictionCombineMode);
							avbdBuildSoftContactTangents(geometry);
							const PxU32 contactIndex = contacts.size();
							avbdAppendPreparedSoftContact(
								geometry, 1e5f, 1e6f, particles, contacts);
							if(geometrySidecar &&
								coreExitNormalLocal.isFinite())
							{
								AvbdOgcTriangleCoreCertificate certificate;
								certificate.points[0].setVertex(v0);
								certificate.points[1].setVertex(v1);
								certificate.points[2].setVertex(v2);
								if(!geometrySidecar->publishTriangleCore(
									contactIndex, certificate))
									contacts.popBack();
							}
						}
					}
				}
		}
	}
	if(geometrySidecar)
		geometrySidecar->resizeContactMapping(contacts.size());
}
}

// =============================================================================

void avbdDetectSoftRigidSphereSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	avbdDetectSoftRigidSphereSDFRange(particles, numParticles, 0,
		numParticles, spheres, numSpheres, contacts, margin,
		softBodies, numSoftBodies);
}

void avbdDetectSoftRigidSphereSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	avbdDetectSoftRigidSphereSweptSDFRange(particles, numParticles, 0,
		numParticles, spheres, numSpheres, contacts, margin,
		softBodies, numSoftBodies);
}

void avbdDetectSoftRigidCapsuleSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	avbdDetectSoftRigidCapsuleSDFRange(particles, numParticles, 0,
		numParticles, capsules, numCapsules, contacts, margin,
		softBodies, numSoftBodies);
}

void avbdDetectSoftRigidCapsuleSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	avbdDetectSoftRigidCapsuleSweptSDFRange(particles, numParticles, 0,
		numParticles, capsules, numCapsules, contacts, margin,
		softBodies, numSoftBodies);
}

void avbdDetectSoftRigidConvexSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	avbdDetectSoftRigidConvexSDFRange(particles, numParticles, 0,
		numParticles, convexes, numConvexes, contacts, margin,
		softBodies, numSoftBodies);
}

void avbdDetectSoftRigidConvexSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	avbdDetectSoftRigidConvexSweptSDFRange(particles, numParticles, 0,
		numParticles, convexes, numConvexes, contacts, margin,
		softBodies, numSoftBodies);
}

} // namespace Dy
} // namespace physx
