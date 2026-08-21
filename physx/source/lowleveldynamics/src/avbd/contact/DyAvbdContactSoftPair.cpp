// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactSoftPair.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"

namespace physx
{
namespace Dy
{

// AVBD soft-soft OGC geometry and detection-plan core.
//
// This unit owns pair planning, optional surface-BVH refit, particle/surface
// and edge/edge feature generation. The public serial wrapper remains in
// DyAvbdContactSoftPair.inl so orchestration stays separate from geometry.
//
// The code intentionally remains in the original namespace and preserves
// the legacy traversal order and contact ownership rules.
// =============================================================================

// PATH 3 (OGC): Simplified Soft-Soft Contact (Sec 3.9)
//
// Outward-only offset, pure quadratic energy, DCD for penetration.
// =============================================================================

// P5.2b makes the existing serial body-pair broadphase an explicit immutable
// input to later refit/query work.  The plan's lexicographic pair order and
// current-versus-swept choice are the canonical merge order for any future
// private candidate tasks.
void avbdBuildSoftSoftOGCDetectionPlan(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace& workspace)
{
	const PxReal r = params.contactRadius;
	const bool hasPreparedBodyBounds =
		workspace.softBodyBoundsValid &&
		workspace.softBodyBounds.size() == numSoftBodies;

	// P3 Slice 5 planning boundary. This is intentionally serial: it produces
	// the same lexicographic overlapping-pair stream the legacy loop would
	// visit, while making pair-specific swept-mode ownership explicit.
	workspace.beginSoftPairDetectionPlan();
	for(PxU32 sA = 0; sA < numSoftBodies; sA++)
	{
		for(PxU32 sB = sA + 1; sB < numSoftBodies; sB++)
		{
			if(stats)
				stats->bodyPairs++;
			const AvbdSoftBody& bodyA = softBodies[sA];
			const AvbdSoftBody& bodyB = softBodies[sB];
			const bool pairSpeculative =
				bodyA.compiled.speculativeCCDEnabled ||
				bodyB.compiled.speculativeCCDEnabled;

			// AABB broadphase per body pair. A P3 prediction fan-in may provide
			// one current/swept result per body. The direct path retains the
			// identical legacy traversal, and no cache survives this detection.
			PxVec3 minA(PX_MAX_F32), maxA(-PX_MAX_F32);
			PxVec3 minB(PX_MAX_F32), maxB(-PX_MAX_F32);
			if(hasPreparedBodyBounds)
			{
				const AvbdSoftBodyBounds& boundsA =
					workspace.softBodyBounds[sA];
				const AvbdSoftBodyBounds& boundsB =
					workspace.softBodyBounds[sB];
				minA = pairSpeculative ? boundsA.sweptMinimum :
					boundsA.currentMinimum;
				maxA = pairSpeculative ? boundsA.sweptMaximum :
					boundsA.currentMaximum;
				minB = pairSpeculative ? boundsB.sweptMinimum :
					boundsB.currentMinimum;
				maxB = pairSpeculative ? boundsB.sweptMaximum :
					boundsB.currentMaximum;
			}
			else
			{
				for (PxU32 i = 0; i < bodyA.compiled.particleCount; i++) {
					const AvbdSoftParticle& particle =
						particles[bodyA.compiled.particleStart + i];
					const PxVec3& p = particle.position;
					minA = minA.minimum(p); maxA = maxA.maximum(p);
					if(pairSpeculative)
					{
						minA = minA.minimum(particle.initialPosition);
						maxA = maxA.maximum(particle.initialPosition);
					}
				}
				for (PxU32 i = 0; i < bodyB.compiled.particleCount; i++) {
					const AvbdSoftParticle& particle =
						particles[bodyB.compiled.particleStart + i];
					const PxVec3& p = particle.position;
					minB = minB.minimum(p); maxB = maxB.maximum(p);
					if(pairSpeculative)
					{
						minB = minB.minimum(particle.initialPosition);
						maxB = maxB.maximum(particle.initialPosition);
					}
				}
			}
			if (minA.x > maxB.x + r || maxA.x < minB.x - r ||
				minA.y > maxB.y + r || maxA.y < minB.y - r ||
				minA.z > maxB.z + r || maxA.z < minB.z - r)
				continue;
			if(stats)
				stats->overlappingBodyPairs++;
			AvbdSoftPairDetectionPlan plan;
			plan.bodyA = sA;
			plan.bodyB = sB;
			plan.swept = pairSpeculative;
			plan.minimumA = minA;
			plan.maximumA = maxA;
			plan.minimumB = minB;
			plan.maximumB = maxB;
			workspace.appendSoftPairDetectionPlan(plan);
		}
	}
}

// P5.2c owns the shared body/mode refit barrier independently from the
// pair-query loop. The workspace epoch spans are still parent-owned; this
// seam merely makes every required write/read dependency explicit.
bool avbdRefitSoftSoftOGCDetectionPlan(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace& workspace)
{
	const bool useSurfaceTriangleBvh = avbdUseSurfaceTriangleBvh();
	if(!useSurfaceTriangleBvh)
		return false;
	workspace.beginSoftPairTriangleBvhEpoch(numSoftBodies);
	for(PxU32 planIndex = 0;
		planIndex < workspace.softPairDetectionPlan.size(); ++planIndex)
	{
		const AvbdSoftPairDetectionPlan& plan =
			workspace.softPairDetectionPlan[planIndex];
		workspace.requireSoftPairTriangleBvhBounds(
			plan.bodyA, plan.swept,
			softBodies[plan.bodyA].compiled.
				surfaceTriangleBvhNodes.size());
		workspace.requireSoftPairTriangleBvhBounds(
			plan.bodyB, plan.swept,
			softBodies[plan.bodyB].compiled.
				surfaceTriangleBvhNodes.size());
	}
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 mode = 0; mode < 2; ++mode)
		{
			const bool swept = mode != 0;
			if(!workspace.isSoftPairTriangleBvhBoundsRequired(
					bodyIndex, swept))
				continue;
			body.compiled.refitSurfaceTriangleBvh(
				particles, swept,
				workspace.getSoftPairTriangleBvhBoundsForRefit(
					bodyIndex, swept));
			workspace.markSoftPairTriangleBvhBoundsRefit(
				bodyIndex, swept);
			if(stats)
				stats->surfaceTriangleBvhRefitNodes +=
					body.compiled.surfaceTriangleBvhNodes.size();
		}
	}
	return true;
}

// P5.9c's post-refit work unit. The parent has already frozen the canonical
// pair plan and refitted every required body/mode BVH span. This leaf reads
// only that immutable epoch and writes only its output stream, statistics and
// caller-owned pair-query scratch.
void avbdDetectSoftSoftOGCPlanRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContactWorkspace& refitWorkspace,
	AvbdSoftContactWorkspace* serialScratchWorkspace,
	AvbdSoftSoftPairQueryScratch& queryScratch,
	bool useSurfaceTriangleBvh,
	PxU32 planBegin, PxU32 planEnd,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats)
{
	PX_UNUSED(numParticles);
	PX_UNUSED(numSoftBodies);
	auto reserveSoftPairQueryScratch = [&](
		PxU32 edgeCountA, PxU32 edgeCountB,
		PxU32 triangleCandidateCapacity)
	{
		if(serialScratchWorkspace)
			serialScratchWorkspace->reserveSoftPairSweep(
				edgeCountA, edgeCountB, triangleCandidateCapacity);
		else
			queryScratch.reserve(
				edgeCountA, edgeCountB, triangleCandidateCapacity);
	};
	const PxReal r = params.contactRadius;
	const PxU32 clampedPlanEnd = PxMin(
		planEnd, refitWorkspace.softPairDetectionPlan.size());
	PX_ASSERT(planBegin <= clampedPlanEnd);
	PxArray<AvbdSurfaceBvhNodeBounds> emptySoftPairTriangleBvhBounds;

	for(PxU32 planIndex = planBegin;
		planIndex < clampedPlanEnd; ++planIndex)
	{
		const AvbdSoftPairDetectionPlan& plan =
			refitWorkspace.softPairDetectionPlan[planIndex];
		PX_ASSERT(plan.bodyA < numSoftBodies &&
			plan.bodyB < numSoftBodies && plan.bodyA < plan.bodyB);
		const PxU32 sA = plan.bodyA;
		const PxU32 sB = plan.bodyB;
		const AvbdSoftBody& bodyA = softBodies[sA];
		const AvbdSoftBody& bodyB = softBodies[sB];
		const bool pairSpeculative = plan.swept;
		const PxVec3& minA = plan.minimumA;
		const PxVec3& maxA = plan.maximumA;
		const PxVec3& minB = plan.minimumB;
		const PxVec3& maxB = plan.maximumB;
		const PxReal pairFriction = 0.5f * (
			PxMax(bodyA.material.dynamicFriction, 0.0f) +
			PxMax(bodyB.material.dynamicFriction, 0.0f));
		const PxArray<AvbdSurfaceBvhNodeBounds>& triangleBvhBoundsA =
			useSurfaceTriangleBvh
				? refitWorkspace.getSoftPairTriangleBvhBounds(
					sA, pairSpeculative)
				: emptySoftPairTriangleBvhBounds;
		const PxArray<AvbdSurfaceBvhNodeBounds>& triangleBvhBoundsB =
			useSurfaceTriangleBvh
				? refitWorkspace.getSoftPairTriangleBvhBounds(
					sB, pairSpeculative)
				: emptySoftPairTriangleBvhBounds;
			reserveSoftPairQueryScratch(
				bodyA.compiled.surfaceEdges.size(),
				bodyB.compiled.surfaceEdges.size(),
				PxMax(bodyA.compiled.surfaceTriangles.size() / 3,
					bodyB.compiled.surfaceTriangles.size() / 3));

			// Lambda: test particles of testBody against surface of surfBody
			auto testParticlesVsSurface = [&](
				const AvbdSoftBody& testBody, const AvbdSoftBody& surfBody,
				PxU32 surfBodyIdx,
				const PxVec3& aabbLo, const PxVec3& aabbHi,
				const PxArray<AvbdSurfaceBvhNodeBounds>&
					surfaceBvhBounds)
			{
				const bool targetIsShell =
					surfBody.compiled.tetrahedra.empty();
				for(PxU32 queryVertexIndex = 0;
					queryVertexIndex <
						testBody.compiled.surfaceVertices.size();
					queryVertexIndex++)
				{
					const PxU32 pi =
						testBody.compiled.surfaceVertices[
							queryVertexIndex];
					if(pi < testBody.compiled.particleStart ||
						pi - testBody.compiled.particleStart >=
							testBody.compiled.particleCount)
						continue;
					if (particles[pi].invMass <= 0.0f) continue;
					const PxVec3& pp = particles[pi].position;

					// Per-particle AABB cull
					const PxVec3 queryMinimum = pairSpeculative
						? pp.minimum(particles[pi].initialPosition) : pp;
					const PxVec3 queryMaximum = pairSpeculative
						? pp.maximum(particles[pi].initialPosition) : pp;
					if (queryMaximum.x < aabbLo.x - r ||
						queryMinimum.x > aabbHi.x + r ||
						queryMaximum.y < aabbLo.y - r ||
						queryMinimum.y > aabbHi.y + r ||
						queryMaximum.z < aabbLo.z - r ||
						queryMinimum.z > aabbHi.z + r)
						continue;
					if(stats)
						stats->particleSurfaceCandidates++;
					const bool useSurfaceTriangleBvhForBody =
						useSurfaceTriangleBvh &&
						!surfBody.compiled.surfaceTriangleBvhNodes.empty();
					PxArray<PxU32>& triangleCandidates =
						queryScratch.triangleCandidates;
					if(useSurfaceTriangleBvhForBody)
					{
						surfBody.compiled.collectSurfaceTriangleBvhCandidates(
							queryMinimum, queryMaximum, r,
							surfaceBvhBounds,
							triangleCandidates);
						if(stats)
							stats->surfaceTriangleBvhCandidateTriangles +=
								triangleCandidates.size();
					}
					const PxU32 candidateTriangleCount =
						useSurfaceTriangleBvhForBody
							? triangleCandidates.size()
							: surfBody.compiled.surfaceTriangles.size() / 3;

					// The discrete owner below cannot see a vertex that
					// crossed the complete target face during this step and
					// finished outside the contact shell.  Select the first
					// face entry over the same initial->predicted interval
					// used by rigid-soft speculative OGC.  Selecting only the
					// earliest face gives redetection a stable, unique owner.
					if(pairSpeculative)
					{
						PxReal bestEntryTime = PX_MAX_F32;
						PxU32 bestTriangleOffset = PX_MAX_U32;
						AvbdSweptTriangleEntry bestEntry;
						for(PxU32 candidateIndex = 0;
							candidateIndex < candidateTriangleCount;
							candidateIndex++)
						{
							const PxU32 ti = useSurfaceTriangleBvhForBody
								? triangleCandidates[candidateIndex] * 3
								: candidateIndex * 3;
							const PxU32 source0 =
								surfBody.compiled.surfaceTriangles[ti];
							const PxU32 source1 =
								surfBody.compiled.surfaceTriangles[ti + 1];
							const PxU32 source2 =
								surfBody.compiled.surfaceTriangles[ti + 2];
							const PxVec3 targetMinimum =
								particles[source0].initialPosition.minimum(
									particles[source1].initialPosition).
								minimum(
									particles[source2].initialPosition).
								minimum(particles[source0].position).
								minimum(particles[source1].position).
								minimum(particles[source2].position);
							const PxVec3 targetMaximum =
								particles[source0].initialPosition.maximum(
									particles[source1].initialPosition).
								maximum(
									particles[source2].initialPosition).
								maximum(particles[source0].position).
								maximum(particles[source1].position).
								maximum(particles[source2].position);
							if(queryMaximum.x < targetMinimum.x - r ||
								queryMinimum.x > targetMaximum.x + r ||
								queryMaximum.y < targetMinimum.y - r ||
								queryMinimum.y > targetMaximum.y + r ||
								queryMaximum.z < targetMinimum.z - r ||
								queryMinimum.z > targetMaximum.z + r)
								continue;
							if(stats)
								stats->closestTriangleTests++;
							AvbdSweptTriangleEntry entry;
							if(avbdRotatingPointEnterExpandedDeformingTriangleFace(
									PxVec3(0.0f),
									particles[pi].initialPosition, pp,
									PxQuat(PxIdentity), PxQuat(PxIdentity),
									particles[source0].initialPosition,
									particles[source1].initialPosition,
									particles[source2].initialPosition,
									particles[source0].position,
									particles[source1].position,
									particles[source2].position,
									r, entry) &&
								entry.entryTime < bestEntryTime)
							{
								bestEntryTime = entry.entryTime;
								bestTriangleOffset = ti;
								bestEntry = entry;
							}
						}
						if(bestTriangleOffset != PX_MAX_U32)
						{
							const PxU32 source0 =
								surfBody.compiled.surfaceTriangles[
									bestTriangleOffset];
							const PxU32 source1 =
								surfBody.compiled.surfaceTriangles[
									bestTriangleOffset + 1];
							const PxU32 source2 =
								surfBody.compiled.surfaceTriangles[
									bestTriangleOffset + 2];
							PxVec3 contactNormal = -bestEntry.normal;
							const AvbdClosestPointResult initialClosest =
								avbdClosestPointOnTriangleOGC(
									particles[pi].initialPosition,
									particles[source0].initialPosition,
									particles[source1].initialPosition,
									particles[source2].initialPosition);
							if(contactNormal.dot(initialClosest.normal) < 0.0f)
								contactNormal = -contactNormal;
							const PxVec3 entry0 =
								particles[source0].initialPosition +
								(particles[source0].position -
								 particles[source0].initialPosition) *
									bestEntry.entryTime;
							const PxVec3 entry1 =
								particles[source1].initialPosition +
								(particles[source1].position -
								 particles[source1].initialPosition) *
									bestEntry.entryTime;
							const PxVec3 entry2 =
								particles[source2].initialPosition +
								(particles[source2].position -
								 particles[source2].initialPosition) *
									bestEntry.entryTime;
							AvbdSoftContactGeometry geometry;
							geometry.source = AvbdSoftContactSource(
								AvbdSoftContactSource::eSOFT_SURFACE,
								surfBodyIdx,
								avbdSoftTrianglePrimitiveKey(
									source0, source1, source2),
								avbdSoftTriangleFeatureKey(
									source0, source1, source2,
									AVBD_FEATURE_FACE, 0));
							geometry.particleIdx = pi;
							geometry.targetKind =
								AvbdSoftContactTargetKind::
									eDEFORMABLE_SURFACE;
							geometry.velocityOwner =
								AvbdVelocityObjectiveOwner::PositionAL;
							geometry.targetIndex = surfBodyIdx;
							const PxU32 triangleIndex =
								bestTriangleOffset / 3;
							geometry.targetSourceElementIndex =
								triangleIndex <
									surfBody.compiled.
										surfaceTriangleElementIndices.size()
								? surfBody.compiled.
									surfaceTriangleElementIndices[
										triangleIndex]
								: PX_MAX_U32;
							geometry.normal = contactNormal;
							geometry.projNormal = contactNormal;
							geometry.depth = 0.0f;
							geometry.margin = r;
							geometry.surfacePoint =
								entry0 * bestEntry.barycentric.x +
								entry1 * bestEntry.barycentric.y +
								entry2 * bestEntry.barycentric.z;
							geometry.surfaceParticleIndices[0] = source0;
							geometry.surfaceParticleIndices[1] = source1;
							geometry.surfaceParticleIndices[2] = source2;
							geometry.surfaceWeights[0] =
								bestEntry.barycentric.x;
							geometry.surfaceWeights[1] =
								bestEntry.barycentric.y;
							geometry.surfaceWeights[2] =
								bestEntry.barycentric.z;
							geometry.friction = pairFriction;
							avbdBuildSoftContactTangents(geometry);
							avbdAppendPreparedSoftContact(
								geometry,
								params.contactStiffness,
								params.contactStiffness * 10.0f,
								particles, contacts);
							continue;
						}
					}

					// DCD: check if particle is inside the other body
					const bool isInside = !targetIsShell &&
						avbdIsPointInsideTetMesh(
							pp, surfBody.compiled.surfaceTriangles,
							particles, stats);
					if (isInside)
					{
						// Find closest surface triangle for direction
						PxReal minDist = PX_MAX_F32;
						PxVec3 bestNormal(0.0f, 1.0f, 0.0f);
						PxVec3 bestClosest(0.0f);
						PxVec3 bestBarycentric(1.0f, 0.0f, 0.0f);
						PxU32 bestTriangle = PX_MAX_U32;
						AvbdClosestFeature bestFeature = AVBD_FEATURE_UNKNOWN;
						PxU32 bestFeatureIndex = 0;
						for (PxU32 ti = 0; ti + 2 < surfBody.compiled.surfaceTriangles.size(); ti += 3)
						{
							if(stats)
								stats->closestTriangleTests++;
							const PxVec3& va = particles[surfBody.compiled.surfaceTriangles[ti]].position;
							const PxVec3& vb = particles[surfBody.compiled.surfaceTriangles[ti+1]].position;
							const PxVec3& vc = particles[surfBody.compiled.surfaceTriangles[ti+2]].position;
							AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
							if (cp.distance < minDist) {
								minDist = cp.distance;
								bestClosest = cp.point;
								bestBarycentric = cp.barycentric;
								bestTriangle = ti / 3;
								bestFeature = cp.feature;
								bestFeatureIndex = cp.featureIndex;
								PxVec3 faceN = (vb - va).cross(vc - va);
								PxReal fLen = faceN.magnitude();
								bestNormal = fLen > 1e-10f ? faceN * (1.0f / fLen) : cp.normal;
							}
						}

						PxReal depth = minDist + r;
						AvbdSoftContactGeometry geometry;
						const PxU32 source0 =
							surfBody.compiled.surfaceTriangles[bestTriangle * 3];
						const PxU32 source1 =
							surfBody.compiled.surfaceTriangles[bestTriangle * 3 + 1];
						const PxU32 source2 =
							surfBody.compiled.surfaceTriangles[bestTriangle * 3 + 2];
						geometry.source = AvbdSoftContactSource(
							AvbdSoftContactSource::eSOFT_SURFACE,
							surfBodyIdx,
							avbdSoftTrianglePrimitiveKey(
								source0, source1, source2),
							avbdSoftTriangleFeatureKey(
								source0, source1, source2,
								bestFeature, bestFeatureIndex));
						geometry.particleIdx  = pi;
						geometry.targetKind =
							AvbdSoftContactTargetKind::
								eDEFORMABLE_SURFACE;
						geometry.velocityOwner =
							AvbdVelocityObjectiveOwner::
								PositionAL;
						geometry.targetIndex = surfBodyIdx;
						geometry.targetSourceElementIndex =
							bestTriangle <
								surfBody.compiled.
									surfaceTriangleElementIndices.size()
							? surfBody.compiled.
								surfaceTriangleElementIndices[
									bestTriangle]
							: PX_MAX_U32;
						geometry.normal       = bestNormal;
						geometry.projNormal   = bestNormal;
						geometry.depth        = depth;
						geometry.margin       = r;
						geometry.surfacePoint = bestClosest;
						geometry.surfaceParticleIndices[0] = source0;
						geometry.surfaceParticleIndices[1] = source1;
						geometry.surfaceParticleIndices[2] = source2;
						geometry.surfaceWeights[0] = bestBarycentric.x;
						geometry.surfaceWeights[1] = bestBarycentric.y;
						geometry.surfaceWeights[2] = bestBarycentric.z;
						geometry.friction = pairFriction;
						avbdBuildSoftContactTangents(geometry);
						avbdAppendPreparedSoftContact(
							geometry,
							params.contactStiffness,
							params.contactStiffness * 10.0f,
							particles, contacts);
						continue;
					}

					auto appendOutwardContact = [&](
						PxU32 ti,
						const AvbdClosestPointResult& cp,
						const PxVec3& contactNormal)
					{
						const PxReal depth = r - cp.distance;
						AvbdSoftContactGeometry geometry;
						const PxU32 source0 =
							surfBody.compiled.surfaceTriangles[ti];
						const PxU32 source1 =
							surfBody.compiled.surfaceTriangles[ti + 1];
						const PxU32 source2 =
							surfBody.compiled.surfaceTriangles[ti + 2];
						geometry.source = AvbdSoftContactSource(
							AvbdSoftContactSource::eSOFT_SURFACE,
							surfBodyIdx,
							avbdSoftTrianglePrimitiveKey(
								source0, source1, source2),
							avbdSoftTriangleFeatureKey(
								source0, source1, source2,
								cp.feature, cp.featureIndex));
						geometry.particleIdx  = pi;
						geometry.targetKind =
							AvbdSoftContactTargetKind::
								eDEFORMABLE_SURFACE;
						geometry.velocityOwner =
							AvbdVelocityObjectiveOwner::
								PositionAL;
						geometry.targetIndex = surfBodyIdx;
						geometry.targetSourceElementIndex =
							ti / 3 <
								surfBody.compiled.
									surfaceTriangleElementIndices.size()
							? surfBody.compiled.
								surfaceTriangleElementIndices[
									ti / 3]
							: PX_MAX_U32;
						geometry.normal       = contactNormal;
						geometry.projNormal   = contactNormal;
						geometry.depth        = depth;
						geometry.margin       = r;
						geometry.surfacePoint = cp.point;
						geometry.surfaceParticleIndices[0] = source0;
						geometry.surfaceParticleIndices[1] = source1;
						geometry.surfaceParticleIndices[2] = source2;
						geometry.surfaceWeights[0] =
							cp.barycentric.x;
						geometry.surfaceWeights[1] =
							cp.barycentric.y;
						geometry.surfaceWeights[2] =
							cp.barycentric.z;
						geometry.friction = pairFriction;
						avbdBuildSoftContactTangents(geometry);
						avbdAppendPreparedSoftContact(
							geometry,
							params.contactStiffness,
							params.contactStiffness * 10.0f,
							particles, contacts);
					};

					// A shared vertex or edge can be represented by several
					// surface triangles. Compiling all of them duplicates
					// one physical signed-distance objective and injects
					// energy. Keep one deterministic closest feature per
					// particle/body pair, matching the penetration branch.
					PxReal bestDistance = r;
					PxU32 bestTriangle = PX_MAX_U32;
					AvbdClosestPointResult bestClosest = {};
					PxVec3 bestContactNormal(0.0f);

					// Not inside: OGC outward offset blocks on surface. Iterate
					// candidates in ascending compiled triangle order so a rare
					// exact closest-distance tie keeps the old traversal owner.
					for(PxU32 candidateIndex = 0;
						candidateIndex < candidateTriangleCount;
						candidateIndex++)
					{
						const PxU32 ti = useSurfaceTriangleBvhForBody
							? triangleCandidates[candidateIndex] * 3
							: candidateIndex * 3;
						if(stats)
							stats->closestTriangleTests++;
						const PxVec3& va = particles[surfBody.compiled.surfaceTriangles[ti]].position;
						const PxVec3& vb = particles[surfBody.compiled.surfaceTriangles[ti+1]].position;
						const PxVec3& vc = particles[surfBody.compiled.surfaceTriangles[ti+2]].position;

						AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
						if (cp.distance >= r) continue;

						// Face normal for outward check
						PxVec3 faceN = (vb - va).cross(vc - va);
						PxReal fLen = faceN.magnitude();
						if (fLen < 1e-10f) continue;
						faceN = faceN * (1.0f / fLen);

						// Sec 3.9: outward-only offset
						PxVec3 toPoint = pp - cp.point;
						if (toPoint.dot(faceN) < 0.0f) continue;

						// OGC contact normal per feature type
						PxVec3 contactNormal = (cp.feature == AVBD_FEATURE_FACE) ? faceN : cp.normal;

						if(cp.distance < bestDistance)
						{
							bestDistance = cp.distance;
							bestTriangle = ti;
							bestClosest = cp;
							bestContactNormal = contactNormal;
						}
					}
					if(bestTriangle != PX_MAX_U32)
						appendOutwardContact(
							bestTriangle,
							bestClosest,
							bestContactNormal);
				}
			};

			// Test A particles vs B surface, then B particles vs A surface
			testParticlesVsSurface(
				bodyA, bodyB, sB, minB, maxB,
				triangleBvhBoundsB);
			testParticlesVsSurface(
				bodyB, bodyA, sA, minA, maxA,
				triangleBvhBoundsA);

			// Vertex-face features alone do not own a crossing between two
			// edge interiors.  Compile one canonical A-edge/B-edge row for
			// that missing OGC feature, with the swept owner taking
			// precedence over the end-of-step discrete owner.
			auto buildEdgeBounds =
				[&](const AvbdSoftBody& body,
					PxArray<AvbdSoftPairEdgeBounds>& bounds)
			{
				bounds.clear();
				for(PxU32 edgeIndex = 0;
					edgeIndex < body.compiled.surfaceEdges.size();
					edgeIndex++)
				{
					const AvbdEdgeInfo& edge =
						body.compiled.surfaceEdges[edgeIndex];
					if(!edge.collisionFeature)
						continue;
					if(edge.p0 >= numParticles ||
						edge.p1 >= numParticles)
						continue;
					AvbdSoftPairEdgeBounds edgeBounds;
					edgeBounds.edgeIndex = edgeIndex;
					edgeBounds.adjacentNormal0 = PxVec3(0.0f);
					edgeBounds.adjacentNormal1 = PxVec3(0.0f);
					edgeBounds.hasExteriorNormalCone = false;
					if(edge.adjacentSurfaceFace0 != PX_MAX_U32 &&
						edge.adjacentSurfaceFace1 != PX_MAX_U32)
					{
						const PxU32 face0 =
							edge.adjacentSurfaceFace0 * 3;
						const PxU32 face1 =
							edge.adjacentSurfaceFace1 * 3;
						if(face0 + 2 <
								body.compiled.surfaceTriangles.size() &&
							face1 + 2 <
								body.compiled.surfaceTriangles.size())
						{
							const PxU32 f00 =
								body.compiled.surfaceTriangles[face0];
							const PxU32 f01 =
								body.compiled.surfaceTriangles[face0 + 1];
							const PxU32 f02 =
								body.compiled.surfaceTriangles[face0 + 2];
							const PxU32 f10 =
								body.compiled.surfaceTriangles[face1];
							const PxU32 f11 =
								body.compiled.surfaceTriangles[face1 + 1];
							const PxU32 f12 =
								body.compiled.surfaceTriangles[face1 + 2];
							if(f00 < numParticles && f01 < numParticles &&
								f02 < numParticles && f10 < numParticles &&
								f11 < numParticles && f12 < numParticles)
							{
								edgeBounds.adjacentNormal0 =
									(particles[f01].position -
									 particles[f00].position).cross(
										particles[f02].position -
										particles[f00].position);
								edgeBounds.adjacentNormal1 =
									(particles[f11].position -
									 particles[f10].position).cross(
										particles[f12].position -
										particles[f10].position);
								edgeBounds.hasExteriorNormalCone =
									edgeBounds.adjacentNormal0.
										magnitudeSquared() > 1.0e-12f &&
									edgeBounds.adjacentNormal1.
										magnitudeSquared() > 1.0e-12f;
							}
						}
					}
					edgeBounds.minimum =
						particles[edge.p0].position.minimum(
							particles[edge.p1].position);
					edgeBounds.maximum =
						particles[edge.p0].position.maximum(
							particles[edge.p1].position);
					if(pairSpeculative)
					{
						edgeBounds.minimum =
							edgeBounds.minimum.minimum(
								particles[edge.p0].initialPosition).
							minimum(
								particles[edge.p1].initialPosition);
						edgeBounds.maximum =
							edgeBounds.maximum.maximum(
								particles[edge.p0].initialPosition).
							maximum(
								particles[edge.p1].initialPosition);
					}
					bounds.pushBack(edgeBounds);
				}
				PxSort(
					bounds.begin(), bounds.size(),
					[](const AvbdSoftPairEdgeBounds& a,
					   const AvbdSoftPairEdgeBounds& b)
					{
						return a.minimum.x < b.minimum.x;
					});
			};
			reserveSoftPairQueryScratch(
				bodyA.compiled.surfaceEdges.size(),
				bodyB.compiled.surfaceEdges.size(),
				PxMax(bodyA.compiled.surfaceTriangles.size() / 3,
					bodyB.compiled.surfaceTriangles.size() / 3));
			PxArray<AvbdSoftPairEdgeBounds>& edgeBoundsA =
				queryScratch.edgeBoundsA;
			PxArray<AvbdSoftPairEdgeBounds>& edgeBoundsB =
				queryScratch.edgeBoundsB;
			buildEdgeBounds(bodyA, edgeBoundsA);
			buildEdgeBounds(bodyB, edgeBoundsB);
			const PxReal edgeFeatureEpsilon = 1.0e-4f;
			const PxReal edgeDistanceEpsilon = 1.0e-8f;
			auto ownsEdgeContactDirection = [](
				const AvbdSoftPairEdgeBounds& edgeBounds,
				const PxVec3& outwardDirection) -> bool
			{
				return !edgeBounds.hasExteriorNormalCone ||
					avbdIsDirectionInSurfaceEdgeNormalCone(
						outwardDirection,
						edgeBounds.adjacentNormal0,
						edgeBounds.adjacentNormal1);
			};
			auto ownsSweptEdgeContactDirection = [&particles, numParticles](
				const AvbdSoftBody& body,
				const AvbdEdgeInfo& edge,
				PxReal time,
				const PxVec3& outwardDirection) -> bool
			{
				if(edge.adjacentSurfaceFace0 == PX_MAX_U32 ||
					edge.adjacentSurfaceFace1 == PX_MAX_U32)
					return true;
				const PxU32 face0 = edge.adjacentSurfaceFace0 * 3;
				const PxU32 face1 = edge.adjacentSurfaceFace1 * 3;
				if(face0 + 2 >= body.compiled.surfaceTriangles.size() ||
					face1 + 2 >= body.compiled.surfaceTriangles.size())
					return true;
				const PxU32 f00 = body.compiled.surfaceTriangles[face0];
				const PxU32 f01 = body.compiled.surfaceTriangles[face0 + 1];
				const PxU32 f02 = body.compiled.surfaceTriangles[face0 + 2];
				const PxU32 f10 = body.compiled.surfaceTriangles[face1];
				const PxU32 f11 = body.compiled.surfaceTriangles[face1 + 1];
				const PxU32 f12 = body.compiled.surfaceTriangles[face1 + 2];
				if(f00 >= numParticles || f01 >= numParticles ||
					f02 >= numParticles || f10 >= numParticles ||
					f11 >= numParticles || f12 >= numParticles)
					return true;
				auto positionAtTime = [&particles, time](PxU32 index)
				{
					return particles[index].initialPosition +
						(particles[index].position -
						 particles[index].initialPosition) * time;
				};
				const PxVec3 p00 = positionAtTime(f00);
				const PxVec3 p10 = positionAtTime(f10);
				const PxVec3 normal0 =
					(positionAtTime(f01) - p00).cross(
						positionAtTime(f02) - p00);
				const PxVec3 normal1 =
					(positionAtTime(f11) - p10).cross(
						positionAtTime(f12) - p10);
				return avbdIsDirectionInSurfaceEdgeNormalCone(
					outwardDirection, normal0, normal1);
			};

			auto findTargetEdgeElement =
				[&](const AvbdSoftBody& target,
					PxU32 edge0, PxU32 edge1) -> PxU32
			{
				for(PxU32 triangleOffset = 0;
					triangleOffset + 2 <
						target.compiled.surfaceTriangles.size();
					triangleOffset += 3)
				{
					const PxU32 v0 =
						target.compiled.surfaceTriangles[
							triangleOffset];
					const PxU32 v1 =
						target.compiled.surfaceTriangles[
							triangleOffset + 1];
					const PxU32 v2 =
						target.compiled.surfaceTriangles[
							triangleOffset + 2];
					const bool has0 =
						v0 == edge0 || v1 == edge0 || v2 == edge0;
					const bool has1 =
						v0 == edge1 || v1 == edge1 || v2 == edge1;
					if(has0 && has1)
					{
						const PxU32 triangleIndex =
							triangleOffset / 3;
						return triangleIndex <
								target.compiled.
									surfaceTriangleElementIndices.size()
							? target.compiled.
								surfaceTriangleElementIndices[
									triangleIndex]
							: PX_MAX_U32;
					}
				}
				return PX_MAX_U32;
			};

			for(PxU32 sortedEdgeA = 0;
				sortedEdgeA < edgeBoundsA.size();
				sortedEdgeA++)
			{
				const AvbdSoftPairEdgeBounds& boundsA =
					edgeBoundsA[sortedEdgeA];
				for(PxU32 sortedEdgeB = 0;
					sortedEdgeB < edgeBoundsB.size();
					sortedEdgeB++)
				{
					const AvbdSoftPairEdgeBounds& boundsB =
						edgeBoundsB[sortedEdgeB];
					if(boundsB.minimum.x > boundsA.maximum.x + r)
						break;
					if(boundsB.maximum.x < boundsA.minimum.x - r ||
						boundsA.minimum.y > boundsB.maximum.y + r ||
						boundsA.maximum.y < boundsB.minimum.y - r ||
						boundsA.minimum.z > boundsB.maximum.z + r ||
						boundsA.maximum.z < boundsB.minimum.z - r)
						continue;
					const AvbdEdgeInfo& queryEdge =
						bodyA.compiled.surfaceEdges[
							boundsA.edgeIndex];
					const AvbdEdgeInfo& targetEdge =
						bodyB.compiled.surfaceEdges[
							boundsB.edgeIndex];
					const PxU32 q0 = queryEdge.p0;
					const PxU32 q1 = queryEdge.p1;
					const PxU32 t0 = targetEdge.p0;
					const PxU32 t1 = targetEdge.p1;
					if(particles[q0].invMass <= 0.0f &&
						particles[q1].invMass <= 0.0f &&
						particles[t0].invMass <= 0.0f &&
						particles[t1].invMass <= 0.0f)
						continue;

					PxReal previousQueryWeight1 = 0.0f;
					PxReal previousTargetWeight1 = 0.0f;
					PxVec3 previousQueryClosest;
					PxVec3 previousTargetClosest;
					avbdClosestPointsOnSegments(
						particles[q0].initialPosition,
						particles[q1].initialPosition,
						particles[t0].initialPosition,
						particles[t1].initialPosition,
						previousQueryWeight1,
						previousTargetWeight1,
						previousQueryClosest,
						previousTargetClosest);
					const PxVec3 previousDelta =
						previousQueryClosest -
						previousTargetClosest;
					auto stabilizeNormal =
						[&](PxVec3 normal) -> PxVec3
					{
						if(previousDelta.magnitudeSquared() >
							edgeDistanceEpsilon *
								edgeDistanceEpsilon)
						{
							if(normal.dot(previousDelta) < 0.0f)
								normal = -normal;
						}
						else
						{
							const PxVec3 previousCross =
								(particles[q1].initialPosition -
								 particles[q0].initialPosition).cross(
									particles[t1].initialPosition -
									particles[t0].initialPosition);
							if(previousCross.magnitudeSquared() >
									edgeDistanceEpsilon *
										edgeDistanceEpsilon &&
								normal.dot(previousCross) < 0.0f)
								normal = -normal;
						}
						return normal;
					};
					auto appendEdgeContact =
						[&](PxReal queryWeight1,
							PxReal targetWeight1,
							const PxVec3& normal,
							PxReal depth,
							const PxVec3& surfacePoint)
					{
						AvbdSoftContactGeometry geometry;
						geometry.source = AvbdSoftContactSource(
							AvbdSoftContactSource::eSOFT_SURFACE,
							sB,
							avbdGetRigidSoftFeatureKey(
								0x53504530u, t0, t1, 0u, 0u),
							avbdGetRigidSoftFeatureKey(
								0x53504531u, q0, q1, t0, t1));
						geometry.particleIdx =
							particles[q0].invMass > 0.0f ? q0 :
							(particles[q1].invMass > 0.0f ? q1 :
							(particles[t0].invMass > 0.0f ? t0 : t1));
						geometry.queryParticleIndices[0] = q0;
						geometry.queryParticleIndices[1] = q1;
						geometry.queryWeights[0] =
							1.0f - queryWeight1;
						geometry.queryWeights[1] = queryWeight1;
						geometry.targetKind =
							AvbdSoftContactTargetKind::
								eDEFORMABLE_SURFACE;
						geometry.velocityOwner =
							AvbdVelocityObjectiveOwner::PositionAL;
						geometry.targetIndex = sB;
						geometry.targetSourceElementIndex =
							findTargetEdgeElement(bodyB, t0, t1);
						geometry.surfaceParticleIndices[0] = t0;
						geometry.surfaceParticleIndices[1] = t1;
						geometry.surfaceWeights[0] =
							1.0f - targetWeight1;
						geometry.surfaceWeights[1] = targetWeight1;
						geometry.normal = normal;
						geometry.projNormal = normal;
						geometry.depth = depth;
						geometry.margin = r;
						geometry.surfacePoint = surfacePoint;
						geometry.friction = pairFriction;
						avbdBuildSoftContactTangents(geometry);
						avbdAppendPreparedSoftContact(
							geometry,
							params.contactStiffness,
							params.contactStiffness * 10.0f,
							particles, contacts);
					};

					if(pairSpeculative)
					{
						AvbdSweptConvexEdgeEntry entry;
						if(avbdDeformingSegmentsEnterExpandedInteriors(
								particles[q0].initialPosition,
								particles[q1].initialPosition,
								particles[q0].position,
								particles[q1].position,
								particles[t0].initialPosition,
								particles[t1].initialPosition,
								particles[t0].position,
								particles[t1].position,
								r, entry))
						{
							const PxVec3 sweptNormal =
								stabilizeNormal(entry.normal);
							// A swept edge row is valid only when the entry
							// direction is owned by both exterior edge cones.
							// Evaluate those cones at the exact entry time so a
							// rotating crease cannot be rejected by its end pose.
							if(ownsSweptEdgeContactDirection(
									bodyA, queryEdge, entry.entryTime,
									-sweptNormal) &&
								ownsSweptEdgeContactDirection(
									bodyB, targetEdge, entry.entryTime,
									sweptNormal))
							{
								const PxVec3 target0AtEntry =
									particles[t0].initialPosition +
									(particles[t0].position -
									 particles[t0].initialPosition) *
										entry.entryTime;
								const PxVec3 target1AtEntry =
									particles[t1].initialPosition +
									(particles[t1].position -
									 particles[t1].initialPosition) *
										entry.entryTime;
								appendEdgeContact(
									entry.softWeight1,
									entry.rigidWeight1,
									sweptNormal,
									0.0f,
									target0AtEntry *
										(1.0f - entry.rigidWeight1) +
									target1AtEntry *
										entry.rigidWeight1);
								continue;
							}
						}
					}

					PxReal queryWeight1 = 0.0f;
					PxReal targetWeight1 = 0.0f;
					PxVec3 queryClosest;
					PxVec3 targetClosest;
					avbdClosestPointsOnSegments(
						particles[q0].position,
						particles[q1].position,
						particles[t0].position,
						particles[t1].position,
						queryWeight1, targetWeight1,
						queryClosest, targetClosest);
					if(queryWeight1 <= edgeFeatureEpsilon ||
						queryWeight1 >=
							1.0f - edgeFeatureEpsilon ||
						targetWeight1 <= edgeFeatureEpsilon ||
						targetWeight1 >=
							1.0f - edgeFeatureEpsilon)
						continue;
					const PxVec3 delta =
						queryClosest - targetClosest;
					const PxReal distance = delta.magnitude();
					if(distance >= r)
						continue;
					PxVec3 normal;
					if(distance > edgeDistanceEpsilon)
						normal = delta * (1.0f / distance);
					else
					{
						normal =
							(particles[q1].position -
							 particles[q0].position).cross(
								particles[t1].position -
								particles[t0].position);
						if(normal.magnitudeSquared() <=
							edgeDistanceEpsilon *
								edgeDistanceEpsilon)
							continue;
						normal.normalize();
					}
					normal = stabilizeNormal(normal);
					if(!ownsEdgeContactDirection(boundsA, -normal) ||
						!ownsEdgeContactDirection(boundsB, normal))
						continue;
					appendEdgeContact(
						queryWeight1, targetWeight1,
						normal,
						r - distance, targetClosest);
				}
		}
	}
}

// Legacy serial entry. It owns the mutable plan/refit epoch, then consumes the
// entire canonical stream through the same P5.9c range leaf used by a future
// task transaction.
void avbdDetectSoftSoftOGC(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace* persistentWorkspace,
	AvbdSoftSoftPairQueryScratch* queryScratchOverride)
{
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	avbdBuildSoftSoftOGCDetectionPlan(
		particles, softBodies, numSoftBodies, params, stats, workspace);
	const bool useSurfaceTriangleBvh =
		avbdRefitSoftSoftOGCDetectionPlan(
			particles, softBodies, numSoftBodies, stats, workspace);
	AvbdSoftSoftPairQueryScratch& queryScratch =
		queryScratchOverride ? *queryScratchOverride :
		workspace.softPairQueryScratch;
	avbdDetectSoftSoftOGCPlanRange(
		particles, numParticles, softBodies, numSoftBodies, workspace,
		queryScratchOverride ? NULL : &workspace, queryScratch,
		useSurfaceTriangleBvh, 0, workspace.softPairDetectionPlan.size(),
		contacts, params, stats);
}

} // namespace Dy
} // namespace physx
