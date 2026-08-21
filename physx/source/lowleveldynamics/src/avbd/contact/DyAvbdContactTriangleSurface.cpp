// CPU AVBD triangle-surface feature planning. The planner only materializes
// deterministic body/surface/family work identities; narrow-phase predicates
// remain in the triangle-surface detector leaves.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactFeatureGeometry.h"
#include "avbd/contact/DyAvbdContactMaterial.h"
#include "avbd/contact/DyAvbdContactRigidSoft.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"

namespace physx
{
namespace Dy
{

void avbdBuildRigidTriangleSurfaceOGCFeaturePlan(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 numSurfaces, AvbdRigidTriangleSurfaceFeaturePlan& plan,
	bool includeSwept, bool includeDiscrete)
{
	PX_ASSERT(softBodies || numSoftBodies == 0);
	plan.clear();
	if(numSurfaces == 0 || numSoftBodies == 0)
		return;

	auto appendPhase = [&] (
		AvbdRigidTriangleSurfaceFeatureWorkItem::Phase phase,
		bool speculativeOnly)
	{
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		{
			const AvbdSoftBody& body = softBodies[bodyIndex];
			if(speculativeOnly && !body.compiled.speculativeCCDEnabled)
				continue;
			const PxU32 edgeCount = body.compiled.surfaceEdges.size();
			const PxU32 triangleCount =
				body.compiled.surfaceTriangles.size() / 3;
			for(PxU32 surfaceIndex = 0;
				surfaceIndex < numSurfaces; ++surfaceIndex)
			{
				if(edgeCount > 0)
					plan.items.pushBack(
						AvbdRigidTriangleSurfaceFeatureWorkItem(
							phase,
							AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE,
							bodyIndex, surfaceIndex, 0, edgeCount));
				if(triangleCount > 0)
					plan.items.pushBack(
						AvbdRigidTriangleSurfaceFeatureWorkItem(
							phase,
							AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_TRIANGLE,
							bodyIndex, surfaceIndex, 0, triangleCount));
			}
		}
	};

	if(includeSwept)
		appendPhase(AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT, true);
	if(includeDiscrete)
		appendPhase(AvbdRigidTriangleSurfaceFeatureWorkItem::eDISCRETE, false);
}

PX_FORCE_INLINE bool avbdIsRigidTriangleSurfaceValid(
	const AvbdRigidTriangleSurface& surface)
{
	return surface.center.isFinite() &&
		surface.rotation.isFinite() &&
		!surface.localBounds.isEmpty() &&
		PxIsFinite(surface.localRadius) &&
		surface.localRadius > 0.0f &&
		surface.vertices.size() >= 3 &&
		!surface.triangles.empty();
}

PX_FORCE_INLINE void avbdConfigureRigidTriangleSurfaceTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidTriangleSurface& surface,
	PxU32 surfaceIndex, const PxVec3& surfaceLocal)
{
	geometry.targetKind = surface.targetKind;
	geometry.velocityOwner =
		surface.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex = surfaceIndex;
	geometry.surfacePoint =
		surface.center + surface.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		surface.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? surface.previousCenter +
				surface.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
}

PX_FORCE_INLINE PxU64 avbdRigidTriangleSurfaceFeatureKey(
	PxU32 tag, PxU32 featureIndex)
{
	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, tag);
	return avbdSoftContactHashValue(hash, featureIndex);
}

PX_FORCE_INLINE bool avbdRigidTriangleSurfaceBvhIntersects(
	const AvbdRigidTriangleSurfaceBvhNode& node,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin)
{
	return node.minimum.x <= queryMaximum.x + margin &&
		node.maximum.x >= queryMinimum.x - margin &&
		node.minimum.y <= queryMaximum.y + margin &&
		node.maximum.y >= queryMinimum.y - margin &&
		node.minimum.z <= queryMaximum.z + margin &&
		node.maximum.z >= queryMinimum.z - margin;
}

inline void avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
	const AvbdRigidTriangleSurface& surface, PxU32 nodeIndex,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates)
{
	const AvbdRigidTriangleSurfaceBvhNode& node =
		surface.triangleBvhNodes[nodeIndex];
	if(!avbdRigidTriangleSurfaceBvhIntersects(
			node, queryMinimum, queryMaximum, margin))
		return;
	if(!node.isLeaf())
	{
		avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
			surface, node.leftChild, queryMinimum, queryMaximum,
			margin, candidates);
		avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
			surface, node.rightChild, queryMinimum, queryMaximum,
			margin, candidates);
		return;
	}
	for(PxU32 entry = node.firstPrimitive;
		entry < node.firstPrimitive + node.primitiveCount; ++entry)
		candidates.pushBack(surface.triangleBvhTriangleIndices[entry]);
}

// Candidate ids are restored to original triangle order before the retained
// exact OGC test. This keeps the hierarchy an acceleration only: tie owner,
// contact source order and feature keys stay byte-identical to the reference.
PX_FORCE_INLINE bool avbdCollectRigidTriangleSurfaceBvhCandidates(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates)
{
	if(!avbdUseRigidTriangleSurfaceBvh() ||
		surface.triangleBvhNodes.empty())
		return false;
	candidates.clear();
	avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
		surface, 0, queryMinimum, queryMaximum, margin, candidates);
	PxSort(candidates.begin(), candidates.size());
	return true;
}

PX_FORCE_INLINE bool avbdBeginRigidTriangleSurfaceFeatureCandidates(
	const AvbdRigidTriangleSurface& surface)
{
	if(++surface.featureBvhCandidateStamp == 0)
	{
		surface.featureBvhCandidateStamp = 1;
		for(PxU32 index = 0;
			index < surface.edgeBvhCandidateStamps.size(); ++index)
			surface.edgeBvhCandidateStamps[index] = 0;
		for(PxU32 index = 0;
			index < surface.vertexBvhCandidateStamps.size(); ++index)
			surface.vertexBvhCandidateStamps[index] = 0;
	}
	return surface.edgeBvhCandidateStamps.size() == surface.edges.size() &&
		surface.vertexBvhCandidateStamps.size() == surface.vertices.size();
}

PX_FORCE_INLINE bool avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	PxArray<PxU32>& triangleCandidates = queryScratch
		? queryScratch->triangleBvhQueryCandidates
		: surface.triangleBvhQueryCandidates;
	if(!avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			triangleCandidates) ||
		(queryScratch
			? !queryScratch->beginFeatureCandidates(
				surface.edges.size(), surface.vertices.size())
			: !avbdBeginRigidTriangleSurfaceFeatureCandidates(surface)))
		return false;
	candidates.clear();
	const PxU32 stamp = queryScratch
		? queryScratch->featureBvhCandidateStamp
		: surface.featureBvhCandidateStamp;
	PxArray<PxU32>& edgeStamps = queryScratch
		? queryScratch->edgeBvhCandidateStamps
		: surface.edgeBvhCandidateStamps;
	for(PxU32 entry = 0; entry < triangleCandidates.size(); ++entry)
	{
		const PxU32 triangleIndex = triangleCandidates[entry];
		if(triangleIndex >= surface.triangles.size())
			continue;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		const PxU32 triangleEdges[3] =
			{triangle.edge0, triangle.edge1, triangle.edge2};
		for(PxU32 localEdge = 0; localEdge < 3; ++localEdge)
		{
			const PxU32 edgeIndex = triangleEdges[localEdge];
			if(edgeIndex >= surface.edges.size() ||
				!surface.edges[edgeIndex].active ||
				edgeStamps[edgeIndex] == stamp)
				continue;
			edgeStamps[edgeIndex] = stamp;
			candidates.pushBack(edgeIndex);
		}
	}
	PxSort(candidates.begin(), candidates.size());
	return true;
}

PX_FORCE_INLINE bool avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	PxArray<PxU32>& triangleCandidates = queryScratch
		? queryScratch->triangleBvhQueryCandidates
		: surface.triangleBvhQueryCandidates;
	if(!avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			triangleCandidates) ||
		(queryScratch
			? !queryScratch->beginFeatureCandidates(
				surface.edges.size(), surface.vertices.size())
			: !avbdBeginRigidTriangleSurfaceFeatureCandidates(surface)))
		return false;
	candidates.clear();
	const PxU32 stamp = queryScratch
		? queryScratch->featureBvhCandidateStamp
		: surface.featureBvhCandidateStamp;
	PxArray<PxU32>& vertexStamps = queryScratch
		? queryScratch->vertexBvhCandidateStamps
		: surface.vertexBvhCandidateStamps;
	for(PxU32 entry = 0; entry < triangleCandidates.size(); ++entry)
	{
		const PxU32 triangleIndex = triangleCandidates[entry];
		if(triangleIndex >= surface.triangles.size())
			continue;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		const PxU32 triangleVertices[3] =
			{triangle.p0, triangle.p1, triangle.p2};
		for(PxU32 localVertex = 0; localVertex < 3; ++localVertex)
		{
			const PxU32 vertexIndex = triangleVertices[localVertex];
			if(vertexIndex >= surface.vertices.size() ||
				!surface.vertices[vertexIndex].active ||
				vertexStamps[vertexIndex] == stamp)
				continue;
			vertexStamps[vertexIndex] = stamp;
			candidates.pushBack(vertexIndex);
		}
	}
	PxSort(candidates.begin(), candidates.size());
	return true;
}

// Canonical one-sided point query shared by discrete and continuous triangle
// surface owners. Inactive tessellation seams only own an orthogonal
// projection from an adjacent face; rounded seam features remain excluded.
PX_FORCE_INLINE bool avbdQueryRigidTriangleSurfaceLocal(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& localPoint, PxReal maximumDistance,
	AvbdRigidTriangleSurfacePointQuery& result,
	AvbdSoftCollisionStats* stats = NULL,
	PxArray<PxU32>* triangleBvhQueryCandidatesOverride = NULL)
{
	if(!avbdIsRigidTriangleSurfaceValid(surface) ||
		!localPoint.isFinite() || maximumDistance <= 0.0f ||
		!PxIsFinite(maximumDistance))
		return false;
	const PxBounds3 expandedBounds(
		surface.localBounds.minimum - PxVec3(maximumDistance),
		surface.localBounds.maximum + PxVec3(maximumDistance));
	if(!expandedBounds.contains(localPoint))
		return false;

	const PxReal normalEpsilon = 1.0e-12f;
	const PxReal featureProjectionTolerance = 1.0e-5f;
	PxArray<PxU32>& triangleCandidates =
		triangleBvhQueryCandidatesOverride
			? *triangleBvhQueryCandidatesOverride
			: surface.triangleBvhQueryCandidates;
	const bool useTriangleBvh =
		avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, localPoint, localPoint, maximumDistance,
			triangleCandidates);
	const PxU32 triangleCount = useTriangleBvh
		? triangleCandidates.size() : surface.triangles.size();
	if(stats)
	{
		stats->rigidTriangleSurfaceFaceCandidates += triangleCount;
		stats->rigidTriangleSurfaceFaceTests += triangleCount;
	}
	bool found = false;
	for(PxU32 triangleEntry = 0; triangleEntry < triangleCount;
		++triangleEntry)
	{
		const PxU32 triangleIndex = useTriangleBvh
			? triangleCandidates[triangleEntry] : triangleEntry;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		if(triangle.p0 >= surface.vertices.size() ||
			triangle.p1 >= surface.vertices.size() ||
			triangle.p2 >= surface.vertices.size())
			continue;
		const PxVec3& p0 = surface.vertices[triangle.p0].point;
		const PxVec3& p1 = surface.vertices[triangle.p1].point;
		const PxVec3& p2 = surface.vertices[triangle.p2].point;
		const PxReal signedPlaneDistance =
			triangle.normal.dot(localPoint - p0);
		if(!PxIsFinite(signedPlaneDistance) || signedPlaneDistance < 0.0f)
			continue;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(localPoint, p0, p1, p2);
		if(!PxIsFinite(closest.distance) ||
			closest.distance >= maximumDistance ||
			closest.distance >= result.distance)
			continue;

		PxVec3 featureOutward = triangle.normal;
		PxU64 featureKey = avbdRigidTriangleSurfaceFeatureKey(
			0x54534641u, triangleIndex);
		if(closest.feature == AVBD_FEATURE_EDGE)
		{
			const PxU32 edgeIndex = closest.featureIndex == 0
				? triangle.edge0
				: closest.featureIndex == 1 ? triangle.edge1 : triangle.edge2;
			if(edgeIndex >= surface.edges.size())
				continue;
			const AvbdRigidTriangleSurfaceEdge& edge =
				surface.edges[edgeIndex];
			if(!edge.active && closest.distance >
				signedPlaneDistance + featureProjectionTolerance)
				continue;
			featureOutward = edge.outward;
			featureKey = avbdRigidTriangleSurfaceFeatureKey(
				0x54534544u, edgeIndex);
		}
		else if(closest.feature == AVBD_FEATURE_VERTEX)
		{
			const PxU32 vertexIndex = closest.featureIndex == 0
				? triangle.p0
				: closest.featureIndex == 1 ? triangle.p1 : triangle.p2;
			if(vertexIndex >= surface.vertices.size())
				continue;
			const AvbdRigidTriangleSurfaceVertex& vertex =
				surface.vertices[vertexIndex];
			if(!vertex.active && closest.distance >
				signedPlaneDistance + featureProjectionTolerance)
				continue;
			featureOutward = vertex.outward;
			featureKey = avbdRigidTriangleSurfaceFeatureKey(
				0x54535654u, vertexIndex);
		}
		if(!featureOutward.isFinite() ||
			featureOutward.magnitudeSquared() <= normalEpsilon)
			continue;
		featureOutward.normalize();
		PxVec3 normalLocal = closest.distance > 1.0e-8f
			? closest.normal : featureOutward;
		if(normalLocal.dot(featureOutward) < -1.0e-5f)
			continue;
		if(closest.feature == AVBD_FEATURE_FACE)
			normalLocal = triangle.normal;
		if(!normalLocal.isFinite() ||
			normalLocal.magnitudeSquared() <= normalEpsilon)
			continue;

		result.distance = closest.distance;
		result.surfaceLocal = closest.point;
		result.normalLocal = normalLocal.getNormalized();
		result.friction = triangle.friction;
		result.frictionCombineMode = triangle.frictionCombineMode;
		result.featureKey = featureKey;
		found = true;
	}
	return found;
}

// Soft boundary vertex versus an open rigid triangle surface. One contact is
// selected per particle/surface using canonical face/edge/vertex ownership.
void avbdDetectSoftRigidTriangleSurface(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts, PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats)
{
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f || !particle.position.isFinite())
			continue;
		const AvbdSoftBody* sourceBody = avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, particleIndex);
		if(sourceBody && !avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;
		for(PxU32 surfaceIndex = 0; surfaceIndex < numSurfaces;
			++surfaceIndex)
		{
			const AvbdRigidTriangleSurface& surface = surfaces[surfaceIndex];
			if(!avbdIsRigidTriangleSurfaceValid(surface))
				continue;
			const PxVec3 worldOffset = particle.position - surface.center;
			const PxReal broadphaseRadius = surface.localRadius + margin;
			if(worldOffset.magnitudeSquared() >
				broadphaseRadius * broadphaseRadius)
				continue;
			const PxVec3 localPoint = surface.rotation.getConjugate().rotate(
				worldOffset);
			AvbdRigidTriangleSurfacePointQuery query;
			if(!avbdQueryRigidTriangleSurfaceLocal(
					surface, localPoint, margin, query, stats))
				continue;
			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF, PX_MAX_U32,
				surface.primitiveKey, query.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal = surface.rotation.rotate(query.normalLocal).
				getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = margin - query.distance;
			geometry.margin = margin;
			avbdConfigureRigidTriangleSurfaceTarget(
				geometry, surface, surfaceIndex, query.surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction, query.friction,
					query.frictionCombineMode)
				: PxMax(query.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f, particles, contacts);
		}
	}
}
// Particle-major range leaf. A parallel caller supplies private BVH query
// storage; the baked surface topology remains read-only.
void avbdDetectSoftRigidTriangleSurfaceRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts,
	PxArray<PxU32>& triangleBvhQueryCandidates,
	PxReal margin,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats)
{
	PX_ASSERT(particleBegin <= particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f || !particle.position.isFinite())
			continue;
		const AvbdSoftBody* sourceBody = avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, particleIndex);
		if(sourceBody && !avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;
		for(PxU32 surfaceIndex = 0; surfaceIndex < numSurfaces;
			++surfaceIndex)
		{
			const AvbdRigidTriangleSurface& surface = surfaces[surfaceIndex];
			if(!avbdIsRigidTriangleSurfaceValid(surface))
				continue;
			const PxVec3 worldOffset = particle.position - surface.center;
			const PxReal broadphaseRadius = surface.localRadius + margin;
			if(worldOffset.magnitudeSquared() >
				broadphaseRadius * broadphaseRadius)
				continue;
			const PxVec3 localPoint = surface.rotation.getConjugate().rotate(
				worldOffset);
			AvbdRigidTriangleSurfacePointQuery query;
			if(!avbdQueryRigidTriangleSurfaceLocal(
					surface, localPoint, margin, query, stats,
					&triangleBvhQueryCandidates))
				continue;
			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF, PX_MAX_U32,
				surface.primitiveKey, query.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal = surface.rotation.rotate(query.normalLocal).
				getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = margin - query.distance;
			geometry.margin = margin;
			avbdConfigureRigidTriangleSurfaceTarget(
				geometry, surface, surfaceIndex, query.surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction, query.friction,
					query.frictionCombineMode)
				: PxMax(query.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f, particles, contacts);
		}
	}
}
PX_FORCE_INLINE bool avbdGetRigidTriangleSurfaceSweepPose(
	const AvbdRigidTriangleSurface& surface,
	PxVec3& centerStart, PxVec3& centerEnd,
	PxQuat& rotationStart, PxQuat& rotationEnd,
	bool& rotationsEquivalent)
{
	if(!avbdIsRigidTriangleSurfaceValid(surface))
		return false;
	const bool kinematicTarget = surface.targetKind ==
		AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	if(surface.targetKind != AvbdSoftContactTargetKind::eWORLD_STATIC &&
		!kinematicTarget)
		return false;
	if(kinematicTarget && (!surface.previousCenter.isFinite() ||
		!surface.previousRotation.isFinite()))
		return false;
	centerStart = kinematicTarget ? surface.previousCenter : surface.center;
	centerEnd = surface.center;
	rotationStart = kinematicTarget
		? surface.previousRotation : surface.rotation;
	rotationEnd = surface.rotation;
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite())
		return false;
	rotationsEquivalent = avbdAreSweepRotationsEquivalent(
		rotationStart, rotationEnd);
	return true;
}

PX_FORCE_INLINE void avbdUpdateSweptTriangleSurfacePointEntry(
	AvbdSweptTriangleSurfacePointEntry& result,
	PxReal entryTime, const PxVec3& normalLocal,
	const PxVec3& surfaceLocal, PxReal friction,
	PxU8 frictionCombineMode, PxU64 featureKey)
{
	if(entryTime < 0.0f || entryTime > 1.0f ||
		entryTime >= result.entryTime || !normalLocal.isFinite() ||
		normalLocal.magnitudeSquared() <= 1.0e-12f ||
		!surfaceLocal.isFinite())
		return;
	result.entryTime = entryTime;
	result.normalLocal = normalLocal.getNormalized();
	result.surfaceLocal = surfaceLocal;
	result.friction = friction;
	result.frictionCombineMode = frictionCombineMode;
	result.featureKey = featureKey;
}

// Exact moving-point entry into the cylindrical interior of an expanded
// segment. Rounded endpoint caps remain vertex-owned.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedSegmentInterior(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& edge0, const PxVec3& edge1,
	PxReal expandedRadius, PxReal& entryTime,
	PxReal& edgeWeight1, PxVec3& entryNormal)
{
	if(!segmentStart.isFinite() || !segmentEnd.isFinite() ||
		!edge0.isFinite() || !edge1.isFinite() || expandedRadius <= 0.0f ||
		!PxIsFinite(expandedRadius))
		return false;
	const PxVec3 direction = segmentEnd - segmentStart;
	const PxVec3 edge = edge1 - edge0;
	const PxReal edgeLengthSq = edge.magnitudeSquared();
	if(direction.magnitudeSquared() <= 1.0e-12f ||
		edgeLengthSq <= 1.0e-12f || !PxIsFinite(edgeLengthSq))
		return false;
	const PxVec3 startOffset = segmentStart - edge0;
	const PxReal startWeight = startOffset.dot(edge) / edgeLengthSq;
	const PxReal weightDirection = direction.dot(edge) / edgeLengthSq;
	const PxVec3 radialStart = startOffset - edge * startWeight;
	const PxVec3 radialDirection = direction - edge * weightDirection;
	const PxReal quadraticA = radialDirection.magnitudeSquared();
	const PxReal halfB = radialStart.dot(radialDirection);
	const PxReal quadraticC = radialStart.magnitudeSquared() -
		expandedRadius * expandedRadius;
	if(quadraticA <= 1.0e-12f || !PxIsFinite(quadraticA) ||
		quadraticC < 0.0f)
		return false;
	const PxReal discriminant = halfB * halfB - quadraticA * quadraticC;
	if(discriminant < 0.0f || !PxIsFinite(discriminant))
		return false;
	entryTime = (-halfB - PxSqrt(discriminant)) / quadraticA;
	if(entryTime < 0.0f || entryTime > 1.0f)
		return false;
	edgeWeight1 = startWeight + weightDirection * entryTime;
	const PxReal featureEpsilon = 1.0e-4f;
	if(edgeWeight1 <= featureEpsilon ||
		edgeWeight1 >= 1.0f - featureEpsilon)
		return false;
	const PxVec3 radial = radialStart + radialDirection * entryTime;
	if(!radial.isFinite() || radial.magnitudeSquared() <= 1.0e-12f)
		return false;
	entryNormal = radial.getNormalized();
	return true;
}

// Exact translation-only entry of a point into the one-sided triangle
// surface offset. Face slabs, active finite edge cylinders, and active vertex
// caps are reduced to the earliest canonical owner.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedTriangleSurface(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& segmentStartLocal, const PxVec3& segmentEndLocal,
	PxReal margin, AvbdSweptTriangleSurfacePointEntry& result,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	if(!segmentStartLocal.isFinite() || !segmentEndLocal.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	const PxVec3 direction = segmentEndLocal - segmentStartLocal;
	if(direction.magnitudeSquared() <= 1.0e-12f)
		return false;
	AvbdRigidTriangleSurfacePointQuery currentQuery;
	if(avbdQueryRigidTriangleSurfaceLocal(
			surface, segmentStartLocal, margin, currentQuery, stats,
			queryScratch ? &queryScratch->triangleBvhQueryCandidates : NULL))
		return false;
	const PxVec3 queryMinimum = segmentStartLocal.minimum(segmentEndLocal);
	const PxVec3 queryMaximum = segmentStartLocal.maximum(segmentEndLocal);
	PxArray<PxU32>& triangleCandidates = queryScratch
		? queryScratch->triangleBvhQueryCandidates
		: surface.triangleBvhQueryCandidates;
	const bool useTriangleBvh = avbdCollectRigidTriangleSurfaceBvhCandidates(
		surface, queryMinimum, queryMaximum, margin, triangleCandidates);
	const PxU32 triangleCount = useTriangleBvh
		? triangleCandidates.size() : surface.triangles.size();
	if(stats)
	{
		stats->rigidTriangleSurfaceFaceCandidates += triangleCount;
		stats->rigidTriangleSurfaceFaceTests += triangleCount;
	}

	const PxReal projectionTolerance = 1.0e-5f;
	for(PxU32 triangleEntry = 0; triangleEntry < triangleCount;
		++triangleEntry)
	{
		const PxU32 triangleIndex = useTriangleBvh
			? triangleCandidates[triangleEntry] : triangleEntry;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		if(triangle.p0 >= surface.vertices.size() ||
			triangle.p1 >= surface.vertices.size() ||
			triangle.p2 >= surface.vertices.size() ||
			!triangle.normal.isFinite())
			continue;
		const PxVec3& p0 = surface.vertices[triangle.p0].point;
		const PxVec3& p1 = surface.vertices[triangle.p1].point;
		const PxVec3& p2 = surface.vertices[triangle.p2].point;
		const PxReal startPlaneDistance =
			triangle.normal.dot(segmentStartLocal - p0);
		const PxReal planeDirection = triangle.normal.dot(direction);
		if(!PxIsFinite(startPlaneDistance) || startPlaneDistance < margin ||
			planeDirection >= -1.0e-12f)
			continue;
		const PxReal entryTime =
			(margin - startPlaneDistance) / planeDirection;
		if(entryTime < 0.0f || entryTime > 1.0f ||
			entryTime >= result.entryTime)
			continue;
		const PxVec3 centerAtEntry =
			segmentStartLocal + direction * entryTime;
		const PxVec3 projected = centerAtEntry - triangle.normal * margin;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(projected, p0, p1, p2);
		if(!PxIsFinite(closest.distance) ||
			closest.distance > projectionTolerance)
			continue;

		PxVec3 featureOutward = triangle.normal;
		PxU64 featureKey = avbdRigidTriangleSurfaceFeatureKey(
			0x54534641u, triangleIndex);
		if(closest.feature == AVBD_FEATURE_EDGE)
		{
			const PxU32 edgeIndex = closest.featureIndex == 0
				? triangle.edge0
				: closest.featureIndex == 1 ? triangle.edge1 : triangle.edge2;
			if(edgeIndex >= surface.edges.size())
				continue;
			featureOutward = surface.edges[edgeIndex].outward;
			featureKey = avbdRigidTriangleSurfaceFeatureKey(
				0x54534544u, edgeIndex);
		}
		else if(closest.feature == AVBD_FEATURE_VERTEX)
		{
			const PxU32 vertexIndex = closest.featureIndex == 0
				? triangle.p0
				: closest.featureIndex == 1 ? triangle.p1 : triangle.p2;
			if(vertexIndex >= surface.vertices.size())
				continue;
			featureOutward = surface.vertices[vertexIndex].outward;
			featureKey = avbdRigidTriangleSurfaceFeatureKey(
				0x54535654u, vertexIndex);
		}
		if(!featureOutward.isFinite() ||
			triangle.normal.dot(featureOutward) < -1.0e-5f)
			continue;
		avbdUpdateSweptTriangleSurfacePointEntry(
			result, entryTime, triangle.normal, closest.point,
			triangle.friction, triangle.frictionCombineMode, featureKey);
	}

	PxArray<PxU32>& edgeCandidates = queryScratch
		? queryScratch->edgeBvhQueryCandidates
		: surface.edgeBvhQueryCandidates;
	const bool useEdgeBvh = avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
		surface, queryMinimum, queryMaximum, margin, edgeCandidates,
		queryScratch);
	const PxU32 edgeCount = useEdgeBvh
		? edgeCandidates.size() : surface.edges.size();
	for(PxU32 edgeEntry = 0; edgeEntry < edgeCount; ++edgeEntry)
	{
		const PxU32 edgeIndex = useEdgeBvh
			? edgeCandidates[edgeEntry] : edgeEntry;
		const AvbdRigidTriangleSurfaceEdge& edge = surface.edges[edgeIndex];
		if(!edge.active || edge.p0 >= surface.vertices.size() ||
			edge.p1 >= surface.vertices.size())
			continue;
		if(stats)
		{
			stats->rigidTriangleSurfaceEdgeCandidates++;
			stats->rigidTriangleSurfaceEdgeTests++;
		}
		const PxVec3& edge0 = surface.vertices[edge.p0].point;
		const PxVec3& edge1 = surface.vertices[edge.p1].point;
		PxReal entryTime = 0.0f;
		PxReal edgeWeight1 = 0.0f;
		PxVec3 entryNormal(0.0f);
		if(!avbdSegmentEnterExpandedSegmentInterior(
				segmentStartLocal, segmentEndLocal, edge0, edge1, margin,
				entryTime, edgeWeight1, entryNormal) ||
			entryTime >= result.entryTime ||
			entryNormal.dot(edge.outward) < -1.0e-5f)
			continue;
		avbdUpdateSweptTriangleSurfacePointEntry(
			result, entryTime,
			entryNormal,
			edge0 * (1.0f - edgeWeight1) + edge1 * edgeWeight1,
			edge.friction, edge.frictionCombineMode,
			avbdRigidTriangleSurfaceFeatureKey(0x54534544u, edgeIndex));
	}

	PxArray<PxU32>& vertexCandidates = queryScratch
		? queryScratch->vertexBvhQueryCandidates
		: surface.vertexBvhQueryCandidates;
	const bool useVertexBvh = avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
		surface, queryMinimum, queryMaximum, margin, vertexCandidates,
		queryScratch);
	const PxU32 vertexCount = useVertexBvh
		? vertexCandidates.size() : surface.vertices.size();
	for(PxU32 vertexEntry = 0; vertexEntry < vertexCount; ++vertexEntry)
	{
		const PxU32 vertexIndex = useVertexBvh
			? vertexCandidates[vertexEntry] : vertexEntry;
		const AvbdRigidTriangleSurfaceVertex& vertex =
			surface.vertices[vertexIndex];
		if(!vertex.active)
			continue;
		if(stats)
		{
			stats->rigidTriangleSurfaceVertexCandidates++;
			stats->rigidTriangleSurfaceVertexTests++;
		}
		PxReal entryTime = 0.0f;
		PxVec3 entryNormal(0.0f);
		if(!avbdSegmentEnterExpandedSphere(
				segmentStartLocal, segmentEndLocal, vertex.point, margin,
				entryTime, entryNormal) ||
			entryTime >= result.entryTime ||
			entryNormal.dot(vertex.outward) < -1.0e-5f)
			continue;
		avbdUpdateSweptTriangleSurfacePointEntry(
			result, entryTime, entryNormal, vertex.point, vertex.friction,
			vertex.frictionCombineMode,
			avbdRigidTriangleSurfaceFeatureKey(0x54535654u, vertexIndex));
	}
	return result.entryTime <= 1.0f;
}

// Continuous point entry against a translating/rotating one-sided triangle
// surface. Each shortest-path slerped pose uses the canonical exact local
// face/active-edge/active-vertex query. Relative point/center translation
// plus localRadius*angularDistance bounds the surface speed, so the
// conservative step cannot cross first contact.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedRotatingTriangleSurface(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal margin, AvbdSweptTriangleSurfacePointEntry& result,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	if(!avbdIsRigidTriangleSurfaceValid(surface) ||
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
		surface.localRadius * angularDistance;
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
			PxSlerp(time, normalizedStart, normalizedEnd).getNormalized();
		if(!point.isFinite() || !center.isFinite() ||
			!rotation.isFinite())
			return false;
		const PxVec3 localPoint =
			rotation.getConjugate().rotate(point - center);
		const PxReal maximumDistance =
			localPoint.magnitude() + surface.localRadius +
			margin + 1.0f;
		if(!PxIsFinite(maximumDistance) || maximumDistance <= margin)
			return false;
		AvbdRigidTriangleSurfacePointQuery query;
		if(!avbdQueryRigidTriangleSurfaceLocal(
				surface, localPoint, maximumDistance, query, stats,
				queryScratch
					? &queryScratch->triangleBvhQueryCandidates : NULL))
			return false;
		if(iteration == 0 && query.distance < margin)
			return false;
		const PxReal gap = query.distance - margin;
		if(gap <= distanceTolerance)
		{
			result.entryTime = time;
			result.normalLocal = query.normalLocal;
			result.surfaceLocal = query.surfaceLocal;
			result.friction = query.friction;
			result.frictionCombineMode = query.frictionCombineMode;
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

inline void avbdDetectSoftRigidTriangleSurfaceSweptImpl(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite() ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(*sourceBody, particleIndex))
			continue;

		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidTriangleSurfaceSweepPose(
					surface, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;

			AvbdSweptTriangleSurfacePointEntry entry;
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
				const PxBounds3 sweptBounds(
					relativeStart.minimum(relativeEnd),
					relativeStart.maximum(relativeEnd));
				const PxBounds3 expandedSurfaceBounds(
					surface.localBounds.minimum - PxVec3(margin),
					surface.localBounds.maximum + PxVec3(margin));
				if(!sweptBounds.intersects(expandedSurfaceBounds) ||
					!avbdSegmentEnterExpandedTriangleSurface(
						surface, relativeStart, relativeEnd,
						margin, entry, stats, queryScratch))
					continue;
				entryRotation = rotationEnd.getNormalized();
			}
			else
			{
				const PxReal rotationExtent =
					surface.localRadius + margin;
				const PxVec3 pointMinimum =
					particle.position.minimum(particle.predictedPosition);
				const PxVec3 pointMaximum =
					particle.position.maximum(particle.predictedPosition);
				const PxVec3 centerMinimum =
					centerStart.minimum(centerEnd) - PxVec3(rotationExtent);
				const PxVec3 centerMaximum =
					centerStart.maximum(centerEnd) + PxVec3(rotationExtent);
				if(pointMinimum.x > centerMaximum.x ||
					pointMaximum.x < centerMinimum.x ||
					pointMinimum.y > centerMaximum.y ||
					pointMaximum.y < centerMinimum.y ||
					pointMinimum.z > centerMaximum.z ||
					pointMaximum.z < centerMinimum.z ||
					!avbdSegmentEnterExpandedRotatingTriangleSurface(
						surface, particle.position,
						particle.predictedPosition,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						margin, entry, stats, queryScratch))
					continue;
				entryRotation = PxSlerp(
					entry.entryTime,
					rotationStart.getNormalized(),
					rotationEnd.getNormalized()).getNormalized();
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, surface.primitiveKey,
				entry.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal = entryRotation.rotate(entry.normalLocal).
				getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			avbdConfigureRigidTriangleSurfaceTarget(
				geometry, surface, surfaceIndex, entry.surfaceLocal);
			geometry.friction = avbdCombineDeformableRigidFriction(
				sourceBody->material.dynamicFriction,
				entry.friction, entry.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1.0e6f, 1.0e6f,
				particles, contacts);
		}
	}
}

void avbdDetectSoftRigidTriangleSurfaceSwept(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats)
{
	avbdDetectSoftRigidTriangleSurfaceSweptImpl(
		particles, numParticles, 0, numParticles,
		surfaces, numSurfaces, contacts, margin,
		softBodies, numSoftBodies, stats, NULL);
}

// P5.16a candidate leaf: every swept forward-SDF query write is supplied by
// caller-owned scratch. The parent merges current-SDF ranges first and
// retains both current and swept OGC feature suffixes.
void avbdDetectSoftRigidTriangleSurfaceSweptRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	AvbdRigidTriangleSurfaceQueryScratch& queryScratch,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats)
{
	avbdDetectSoftRigidTriangleSurfaceSweptImpl(
		particles, numParticles, particleBegin, particleEnd,
		surfaces, numSurfaces, contacts, margin,
		softBodies, numSoftBodies, stats, &queryScratch);
}

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweep(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent,
	const AvbdSoftParticle& particle, PxReal margin,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	const PxQuat inverseStart = rotationStart.getConjugate();
	const PxVec3 relativeStart = inverseStart.rotate(
		particle.initialPosition - centerStart);
	const PxReal maximumDistance = relativeStart.magnitude() +
		surface.localRadius + margin + 1.0f;
	if(!PxIsFinite(maximumDistance))
		return false;
	AvbdRigidTriangleSurfacePointQuery currentQuery;
	if(avbdQueryRigidTriangleSurfaceLocal(
			surface, relativeStart, maximumDistance, currentQuery, stats,
			queryScratch ? &queryScratch->triangleBvhQueryCandidates : NULL) &&
		currentQuery.distance < margin)
		return true;
	AvbdSweptTriangleSurfacePointEntry entry;
	if(!rotationsEquivalent)
		return avbdSegmentEnterExpandedRotatingTriangleSurface(
			surface, particle.initialPosition, particle.predictedPosition,
			centerStart, centerEnd, rotationStart, rotationEnd,
			margin, entry, stats, queryScratch);
	const PxVec3 relativeEnd = rotationEnd.getConjugate().rotate(
		particle.predictedPosition - centerEnd);
	return avbdSegmentEnterExpandedTriangleSurface(
		surface, relativeStart, relativeEnd, margin,
		entry, stats, queryScratch);
}

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweepCached(
	const AvbdRigidTriangleSurface& surface, PxU32 surfaceSlot,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent, const AvbdSoftParticle& particle,
	PxU32 particleIndex, PxReal margin, AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache& resultCache)
{
	bool cachedResult = false;
	if(resultCache.lookup(surfaceSlot, particleIndex, cachedResult))
		return cachedResult;
	const bool result = avbdTriangleSurfaceForwardVertexOwnsSweep(
		surface, centerStart, centerEnd, rotationStart, rotationEnd,
		rotationsEquivalent, particle, margin, stats, queryScratch);
	resultCache.store(surfaceSlot, particleIndex, result);
	return result;
}

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweepParentCached(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent, const AvbdSoftParticle& particle,
	PxU32 particleIndex, PxReal margin, AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch,
	PxArray<PxU8>& resultCache)
{
	PX_ASSERT(particleIndex < resultCache.size());
	PxU8& state = resultCache[particleIndex];
	if(state != 0)
		return state == 2;
	const bool result = avbdTriangleSurfaceForwardVertexOwnsSweep(
		surface, centerStart, centerEnd, rotationStart, rotationEnd,
		rotationsEquivalent, particle, margin, stats, queryScratch);
	state = result ? PxU8(2) : PxU8(1);
	return result;
}
PX_FORCE_INLINE void avbdUpdateSweptTriangleEntry(
	AvbdSweptTriangleEntry& result,
	PxReal entryTime, const PxVec3& normal,
	const PxVec3& barycentric,
	AvbdClosestFeature feature, PxU32 featureIndex)
{
	if(entryTime < 0.0f || entryTime > 1.0f ||
		entryTime >= result.entryTime ||
		!normal.isFinite() ||
		normal.magnitudeSquared() <= 1.0e-12f ||
		!barycentric.isFinite())
		return;
	result.entryTime = entryTime;
	result.normal = normal.getNormalized();
	result.barycentric = barycentric;
	result.feature = feature;
	result.featureIndex = featureIndex;
}
// Continuous rotating convex-vertex entry into the face-owned portion of a
// translation-only soft triangle. Soft edges/vertices and convex-edge cases
// retain their separate owners.
bool avbdRotatingPointEnterExpandedTriangleFace(
	const PxVec3& rigidLocalPoint,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal margin, AvbdSweptTriangleEntry& result)
{
	if(!rigidLocalPoint.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!a.isFinite() || !b.isFinite() || !c.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		rigidLocalPoint.magnitude() * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

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
		const PxVec3 rigidPoint =
			center + rotation.rotate(rigidLocalPoint);
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				rigidPoint, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 && closest.distance < margin)
			return false;
		const PxReal gap = closest.distance - margin;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE)
				return false;
			const PxVec3 normal = closest.point - rigidPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.feature = AVBD_FEATURE_FACE;
			result.featureIndex = 0;
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

// Continuous rotating/translating rigid vertex entry into the face-owned
// portion of a linearly deforming soft triangle. The maximum residual
// soft-vertex speed augments the rigid point speed bound. Soft edges and
// vertices retain their existing unique owners.
bool
avbdRotatingPointEnterExpandedDeformingTriangleFace(
	const PxVec3& rigidLocalPoint,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal margin, AvbdSweptTriangleEntry& result)
{
	if(!rigidLocalPoint.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!aStart.isFinite() || !bStart.isFinite() ||
		!cStart.isFinite() || !aEnd.isFinite() ||
		!bEnd.isFinite() || !cEnd.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
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
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		rigidLocalPoint.magnitude() * angularDistance +
		triangleSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		const PxVec3 rigidPoint =
			center + rotation.rotate(rigidLocalPoint);
		const PxVec3 a = aStart + displacementA * time;
		const PxVec3 b = bStart + displacementB * time;
		const PxVec3 c = cStart + displacementC * time;
		const PxVec3 triangleNormal = (b - a).cross(c - a);
		if(!center.isFinite() || !rotation.isFinite() ||
			!rigidPoint.isFinite() || !a.isFinite() ||
			!b.isFinite() || !c.isFinite() ||
			!triangleNormal.isFinite() ||
			triangleNormal.magnitudeSquared() <= 1.0e-16f)
			return false;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				rigidPoint, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 && closest.distance < margin)
			return false;
		const PxReal gap = closest.distance - margin;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE)
				return false;
			const PxVec3 normal = closest.point - rigidPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.feature = AVBD_FEATURE_FACE;
			result.featureIndex = 0;
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


// Continuous translated rigid-edge/soft-edge entry. Endpoint ownership is
// excluded on both segments so soft vertices remain forward-SDF owned and
// convex vertices remain reverse vertex/face owned.
bool
avbdTranslatedSegmentEnterExpandedSegmentInteriors(
	const PxVec3& rigid0, const PxVec3& rigid1,
	const PxVec3& rigidTranslation,
	const PxVec3& soft0, const PxVec3& soft1,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!rigid0.isFinite() || !rigid1.isFinite() ||
		!rigidTranslation.isFinite() || !soft0.isFinite() ||
		!soft1.isFinite() || margin <= 0.0f ||
		!PxIsFinite(margin))
		return false;
	const PxReal speedSq =
		rigidTranslation.magnitudeSquared();
	if(speedSq <= 1.0e-12f || !PxIsFinite(speedSq))
		return false;
	const PxReal speed = PxSqrt(speedSq);
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;
	PxReal currentSoftWeight1 = 0.0f;
	PxReal currentRigidWeight1 = 0.0f;
	PxVec3 currentSoftClosest(0.0f);
	PxVec3 currentRigidClosest(0.0f);
	avbdClosestPointsOnSegments(
		soft0, soft1, rigid0, rigid1,
		currentSoftWeight1, currentRigidWeight1,
		currentSoftClosest, currentRigidClosest);
	const PxReal currentDistance =
		(currentSoftClosest - currentRigidClosest).magnitude();
	if(!PxIsFinite(currentDistance) ||
		currentDistance < margin)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		const PxVec3 offset = rigidTranslation * time;
		PxReal softWeight1 = 0.0f;
		PxReal rigidWeight1 = 0.0f;
		PxVec3 softClosest(0.0f);
		PxVec3 rigidClosest(0.0f);
		avbdClosestPointsOnSegments(
			soft0, soft1,
			rigid0 + offset, rigid1 + offset,
			softWeight1, rigidWeight1,
			softClosest, rigidClosest);
		const PxVec3 delta = softClosest - rigidClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(softWeight1 <= featureEpsilon ||
				softWeight1 >= 1.0f - featureEpsilon ||
				rigidWeight1 <= featureEpsilon ||
				rigidWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = softWeight1;
			result.rigidWeight1 = rigidWeight1;
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

// Continuous rotating/translating rigid-edge versus a translation-only soft
// edge. The common soft displacement is removed by the caller. Endpoint
// ownership stays excluded on both segments.
bool
avbdRotatingSegmentEnterExpandedSegmentInteriors(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& soft0, const PxVec3& soft1,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!rigidLocal0.isFinite() || !rigidLocal1.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!soft0.isFinite() || !soft1.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxReal edgeRadius = PxMax(
		rigidLocal0.magnitude(), rigidLocal1.magnitude());
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		edgeRadius * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;

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
		const PxVec3 rigid0 =
			center + rotation.rotate(rigidLocal0);
		const PxVec3 rigid1 =
			center + rotation.rotate(rigidLocal1);
		PxReal softWeight1 = 0.0f;
		PxReal rigidWeight1 = 0.0f;
		PxVec3 softClosest(0.0f);
		PxVec3 rigidClosest(0.0f);
		avbdClosestPointsOnSegments(
			soft0, soft1, rigid0, rigid1,
			softWeight1, rigidWeight1,
			softClosest, rigidClosest);
		const PxVec3 delta = softClosest - rigidClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		if(iteration == 0 && distance < margin)
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(softWeight1 <= featureEpsilon ||
				softWeight1 >= 1.0f - featureEpsilon ||
				rigidWeight1 <= featureEpsilon ||
				rigidWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = softWeight1;
			result.rigidWeight1 = rigidWeight1;
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

// Continuous rotating/translating rigid edge versus a linearly deforming
// soft edge. The maximum endpoint speeds bound each segment's Hausdorff
// motion. Endpoint-owned pairs remain excluded on both features.
bool
avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& soft0Start, const PxVec3& soft1Start,
	const PxVec3& soft0End, const PxVec3& soft1End,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!rigidLocal0.isFinite() || !rigidLocal1.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!soft0Start.isFinite() || !soft1Start.isFinite() ||
		!soft0End.isFinite() || !soft1End.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 softDisplacement0 = soft0End - soft0Start;
	const PxVec3 softDisplacement1 = soft1End - soft1Start;
	const PxReal softSpeed = PxMax(
		softDisplacement0.magnitude(),
		softDisplacement1.magnitude());
	const PxReal rigidRadius = PxMax(
		rigidLocal0.magnitude(), rigidLocal1.magnitude());
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		rigidRadius * angularDistance + softSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		const PxVec3 rigid0 =
			center + rotation.rotate(rigidLocal0);
		const PxVec3 rigid1 =
			center + rotation.rotate(rigidLocal1);
		const PxVec3 soft0 =
			soft0Start + softDisplacement0 * time;
		const PxVec3 soft1 =
			soft1Start + softDisplacement1 * time;
		if(!center.isFinite() || !rotation.isFinite() ||
			!rigid0.isFinite() || !rigid1.isFinite() ||
			!soft0.isFinite() || !soft1.isFinite() ||
			(rigid1 - rigid0).magnitudeSquared() <= 1.0e-16f ||
			(soft1 - soft0).magnitudeSquared() <= 1.0e-16f)
			return false;
		PxReal softWeight1 = 0.0f;
		PxReal rigidWeight1 = 0.0f;
		PxVec3 softClosest(0.0f);
		PxVec3 rigidClosest(0.0f);
		avbdClosestPointsOnSegments(
			soft0, soft1, rigid0, rigid1,
			softWeight1, rigidWeight1,
			softClosest, rigidClosest);
		const PxVec3 delta = softClosest - rigidClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		if(iteration == 0 && distance < margin)
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(softWeight1 <= featureEpsilon ||
				softWeight1 >= 1.0f - featureEpsilon ||
				rigidWeight1 <= featureEpsilon ||
				rigidWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = softWeight1;
			result.rigidWeight1 = rigidWeight1;
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

// Continuous linearly deforming edge versus linearly deforming edge.  The sum
// of the two maximum endpoint displacements is a conservative relative-speed
// bound for the segment distance.  Only the two edge interiors are owned here;
// vertex-edge and vertex-vertex contacts retain their existing owners.
bool
avbdDeformingSegmentsEnterExpandedInteriors(
	const PxVec3& query0Start, const PxVec3& query1Start,
	const PxVec3& query0End, const PxVec3& query1End,
	const PxVec3& target0Start, const PxVec3& target1Start,
	const PxVec3& target0End, const PxVec3& target1End,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!query0Start.isFinite() || !query1Start.isFinite() ||
		!query0End.isFinite() || !query1End.isFinite() ||
		!target0Start.isFinite() || !target1Start.isFinite() ||
		!target0End.isFinite() || !target1End.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;

	const PxVec3 queryDisplacement0 = query0End - query0Start;
	const PxVec3 queryDisplacement1 = query1End - query1Start;
	const PxVec3 targetDisplacement0 = target0End - target0Start;
	const PxVec3 targetDisplacement1 = target1End - target1Start;
	const PxReal querySpeed = PxMax(
		queryDisplacement0.magnitude(),
		queryDisplacement1.magnitude());
	const PxReal targetSpeed = PxMax(
		targetDisplacement0.magnitude(),
		targetDisplacement1.magnitude());
	const PxReal speed = querySpeed + targetSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;

	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;
	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 query0 =
			query0Start + queryDisplacement0 * time;
		const PxVec3 query1 =
			query1Start + queryDisplacement1 * time;
		const PxVec3 target0 =
			target0Start + targetDisplacement0 * time;
		const PxVec3 target1 =
			target1Start + targetDisplacement1 * time;
		if(!query0.isFinite() || !query1.isFinite() ||
			!target0.isFinite() || !target1.isFinite() ||
			(query1 - query0).magnitudeSquared() <= 1.0e-16f ||
			(target1 - target0).magnitudeSquared() <= 1.0e-16f)
			return false;

		PxReal queryWeight1 = 0.0f;
		PxReal targetWeight1 = 0.0f;
		PxVec3 queryClosest(0.0f);
		PxVec3 targetClosest(0.0f);
		avbdClosestPointsOnSegments(
			query0, query1, target0, target1,
			queryWeight1, targetWeight1,
			queryClosest, targetClosest);
		const PxVec3 delta = queryClosest - targetClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		// A contact already active at the beginning of the step is owned by
		// the discrete path.  This also prevents a second swept owner.
		if(iteration == 0 && distance < margin)
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(queryWeight1 <= featureEpsilon ||
				queryWeight1 >= 1.0f - featureEpsilon ||
				targetWeight1 <= featureEpsilon ||
				targetWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = queryWeight1;
			result.rigidWeight1 = targetWeight1;
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





// Exact entry of a moving point into the face/edge-owned part of a static
// triangle expanded by a radius. The rounded vertex caps are intentionally
// excluded: soft-vertex/sphere swept SDF is their unique owner.
bool avbdSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius, AvbdSweptTriangleEntry& result)
{
	if(!segmentStart.isFinite() || !segmentEnd.isFinite() ||
		!a.isFinite() || !b.isFinite() || !c.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxVec3 direction = segmentEnd - segmentStart;
	const PxReal directionMagnitudeSq = direction.magnitudeSquared();
	if(directionMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(directionMagnitudeSq))
		return false;
	const AvbdClosestPointResult currentClosest =
		avbdClosestPointOnTriangleOGC(segmentStart, a, b, c);
	if(!PxIsFinite(currentClosest.distance) ||
		currentClosest.distance < expandedRadius)
		return false;

	const PxVec3 unnormalizedNormal = (b - a).cross(c - a);
	const PxReal normalMagnitudeSq =
		unnormalizedNormal.magnitudeSquared();
	if(normalMagnitudeSq <= 1.0e-16f ||
		!PxIsFinite(normalMagnitudeSq))
		return false;
	const PxVec3 triangleNormal =
		unnormalizedNormal * PxRecipSqrt(normalMagnitudeSq);
	const PxReal startPlaneDistance =
		triangleNormal.dot(segmentStart - a);
	const PxReal planeDirection =
		triangleNormal.dot(direction);

	if(PxAbs(planeDirection) > 1.0e-12f)
	{
		PxReal side = 0.0f;
		if(startPlaneDistance >= expandedRadius)
			side = 1.0f;
		else if(startPlaneDistance <= -expandedRadius)
			side = -1.0f;
		if(side != 0.0f)
		{
			const PxReal entryTime =
				(side * expandedRadius - startPlaneDistance) /
					planeDirection;
			if(entryTime >= 0.0f && entryTime <= 1.0f)
			{
				const PxVec3 centerAtEntry =
					segmentStart + direction * entryTime;
				const PxVec3 trianglePoint =
					centerAtEntry -
						triangleNormal *
							(side * expandedRadius);
				const AvbdClosestPointResult faceClosest =
					avbdClosestPointOnTriangleOGC(
						trianglePoint, a, b, c);
				if(faceClosest.feature == AVBD_FEATURE_FACE &&
					faceClosest.distance <= 1.0e-5f)
					avbdUpdateSweptTriangleEntry(
						result, entryTime,
						-triangleNormal * side,
						faceClosest.barycentric,
						AVBD_FEATURE_FACE, 0);
			}
		}
	}

	const PxVec3 edgeStart[3] = {a, a, b};
	const PxVec3 edgeEnd[3] = {b, c, c};
	for(PxU32 edgeIndex = 0; edgeIndex < 3; ++edgeIndex)
	{
		const PxVec3 edge = edgeEnd[edgeIndex] - edgeStart[edgeIndex];
		const PxReal edgeLengthSq = edge.magnitudeSquared();
		if(edgeLengthSq <= 1.0e-16f ||
			!PxIsFinite(edgeLengthSq))
			continue;
		const PxReal edgeLength = PxSqrt(edgeLengthSq);
		const PxVec3 edgeDirection = edge / edgeLength;
		const PxVec3 startOffset =
			segmentStart - edgeStart[edgeIndex];
		const PxReal startAxial =
			startOffset.dot(edgeDirection);
		const PxReal directionAxial =
			direction.dot(edgeDirection);
		const PxVec3 startRadial =
			startOffset - edgeDirection * startAxial;
		const PxVec3 directionRadial =
			direction - edgeDirection * directionAxial;
		const PxReal quadraticA =
			directionRadial.magnitudeSquared();
		const PxReal quadraticHalfB =
			startRadial.dot(directionRadial);
		const PxReal quadraticC =
			startRadial.magnitudeSquared() -
				expandedRadius * expandedRadius;
		if(quadraticA <= 1.0e-12f || quadraticC < 0.0f)
			continue;
		const PxReal discriminant =
			quadraticHalfB * quadraticHalfB -
				quadraticA * quadraticC;
		if(discriminant < 0.0f || !PxIsFinite(discriminant))
			continue;
		const PxReal entryTime =
			(-quadraticHalfB - PxSqrt(discriminant)) /
				quadraticA;
		if(entryTime < 0.0f || entryTime > 1.0f)
			continue;
		const PxReal axial =
			startAxial + directionAxial * entryTime;
		const PxReal endpointEpsilon =
			PxMax(1.0e-5f, edgeLength * 1.0e-5f);
		if(axial <= endpointEpsilon ||
			axial >= edgeLength - endpointEpsilon)
			continue;
		const PxVec3 centerAtEntry =
			segmentStart + direction * entryTime;
		const PxVec3 edgePoint =
			edgeStart[edgeIndex] + edgeDirection * axial;
		const PxVec3 centerToEdge = edgePoint - centerAtEntry;
		PxVec3 barycentric(0.0f);
		const PxReal edgeWeight = axial / edgeLength;
		if(edgeIndex == 0)
			barycentric =
				PxVec3(1.0f - edgeWeight, edgeWeight, 0.0f);
		else if(edgeIndex == 1)
			barycentric =
				PxVec3(1.0f - edgeWeight, 0.0f, edgeWeight);
		else
			barycentric =
				PxVec3(0.0f, 1.0f - edgeWeight, edgeWeight);
		avbdUpdateSweptTriangleEntry(
			result, entryTime, centerToEdge, barycentric,
			AVBD_FEATURE_EDGE, edgeIndex);
	}
	return result.entryTime < PX_MAX_F32 &&
		(result.feature == AVBD_FEATURE_FACE ||
		 result.feature == AVBD_FEATURE_EDGE);
}

// Continuous entry of a linearly moving point into the face/edge-owned
// portion of a linearly deforming triangle expanded by a radius. The
// point speed plus the maximum triangle-vertex speed bounds the Hausdorff
// speed of the two features, so conservative advancement cannot step over
// first contact. Rounded triangle vertices remain forward-SDF owned.
bool
avbdLinearPointEnterExpandedDeformingTriangleNonVertex(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal expandedRadius, AvbdSweptTriangleEntry& result)
{
	if(!pointStart.isFinite() || !pointEnd.isFinite() ||
		!aStart.isFinite() || !bStart.isFinite() ||
		!cStart.isFinite() || !aEnd.isFinite() ||
		!bEnd.isFinite() || !cEnd.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxVec3 pointDisplacement = pointEnd - pointStart;
	const PxVec3 displacementA = aEnd - aStart;
	const PxVec3 displacementB = bEnd - bStart;
	const PxVec3 displacementC = cEnd - cStart;
	const PxReal triangleSpeed = PxMax(
		displacementA.magnitude(),
		PxMax(displacementB.magnitude(),
			displacementC.magnitude()));
	const PxReal speed =
		pointDisplacement.magnitude() + triangleSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + pointDisplacement * time;
		const PxVec3 a = aStart + displacementA * time;
		const PxVec3 b = bStart + displacementB * time;
		const PxVec3 c = cStart + displacementC * time;
		const PxVec3 triangleNormal = (b - a).cross(c - a);
		if(!point.isFinite() || !a.isFinite() ||
			!b.isFinite() || !c.isFinite() ||
			!triangleNormal.isFinite() ||
			triangleNormal.magnitudeSquared() <= 1.0e-16f)
			return false;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(point, a, b, c);
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
			const PxVec3 normal = closest.point - point;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
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




// The local bounds of the eight transformed corners conservatively contain
// every transformed point of the world-axis-aligned body box. A miss against
// the expanded immutable triangle-surface bounds therefore proves that none
// of this row's soft edges/triangles can recover a rigid feature candidate.
PX_FORCE_INLINE bool avbdRigidTriangleSurfaceBodyMayReachLocalBounds(
	const PxVec3& bodyMinimum, const PxVec3& bodyMaximum,
	const AvbdRigidTriangleSurface& surface, const PxQuat& inverseRotation,
	PxReal margin)
{
	if(!bodyMinimum.isFinite() || !bodyMaximum.isFinite() ||
		bodyMinimum.x > bodyMaximum.x || bodyMinimum.y > bodyMaximum.y ||
		bodyMinimum.z > bodyMaximum.z || margin < 0.0f || !PxIsFinite(margin))
		return true;
	PxVec3 localMinimum(PX_MAX_F32);
	PxVec3 localMaximum(-PX_MAX_F32);
	for(PxU32 cornerIndex = 0; cornerIndex < 8; ++cornerIndex)
	{
		const PxVec3 worldPoint(
			(cornerIndex & 1u) ? bodyMaximum.x : bodyMinimum.x,
			(cornerIndex & 2u) ? bodyMaximum.y : bodyMinimum.y,
			(cornerIndex & 4u) ? bodyMaximum.z : bodyMinimum.z);
		const PxVec3 localPoint = inverseRotation.rotate(
			worldPoint - surface.center);
		if(!localPoint.isFinite())
			return true;
		localMinimum = localMinimum.minimum(localPoint);
		localMaximum = localMaximum.maximum(localPoint);
	}
	const PxVec3 expandedMinimum =
		surface.localBounds.minimum - PxVec3(margin);
	const PxVec3 expandedMaximum =
		surface.localBounds.maximum + PxVec3(margin);
	return !(localMinimum.x > expandedMaximum.x ||
		localMaximum.x < expandedMinimum.x ||
		localMinimum.y > expandedMaximum.y ||
		localMaximum.y < expandedMinimum.y ||
		localMinimum.z > expandedMaximum.z ||
		localMaximum.z < expandedMinimum.z);
}
void avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch,
	PxArray<PxU8>* persistentForwardOwnerScratch,
	const AvbdRigidTriangleSurfaceFeatureWorkItem* workItem,
	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		sweptSubstageTiming,
	AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
		forwardOwnerQueryStats,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache*
		forwardOwnerResultCache)
{
	typedef AvbdRigidTriangleSurfaceFeatureWorkItem WorkItem;
	if(workItem && workItem->phase != WorkItem::eSWEPT)
		return;
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8>* parentForwardOwnerScratch = workItem ?
		NULL : persistentForwardOwnerScratch;
	if(parentForwardOwnerScratch)
	{
		if(parentForwardOwnerScratch->capacity() < numParticles)
			parentForwardOwnerScratch->reserve(numParticles);
		parentForwardOwnerScratch->resize(numParticles);
	}
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		if(workItem && bodyIndex != workItem->bodyIndex)
			continue;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			if(workItem && surfaceIndex != workItem->surfaceIndex)
				continue;
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			const PxU32 forwardOwnerResultCacheSurfaceSlot =
				forwardOwnerResultCache ?
					forwardOwnerResultCache->getSurfaceSlot(surfaceIndex) :
					PX_MAX_U32;
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidTriangleSurfaceSweepPose(
					surface, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;
			if(parentForwardOwnerScratch)
			{
				const PxU32 particleStart = body.compiled.particleStart;
				if(particleStart <= numParticles)
				{
					const PxU32 particleCount = PxMin(
						body.compiled.particleCount,
						numParticles - particleStart);
					for(PxU32 localParticle = 0;
						localParticle < particleCount; ++localParticle)
						(*parentForwardOwnerScratch)[
							particleStart + localParticle] = 0;
				}
			}

			PxU32 softEdgeBegin = 0;
			PxU32 softEdgeEnd = 0;
			if(!workItem)
				softEdgeEnd = body.compiled.surfaceEdges.size();
			else if(workItem->family == WorkItem::eSOFT_EDGE)
			{
				softEdgeBegin = PxMin(workItem->primitiveBegin,
					body.compiled.surfaceEdges.size());
				softEdgeEnd = PxMin(
					PxMax(workItem->primitiveEnd, softEdgeBegin),
					body.compiled.surfaceEdges.size());
			}
			for(PxU32 softEdgeIndex = softEdgeBegin;
				softEdgeIndex < softEdgeEnd;
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
				const PxVec3 displacement0 =
					particles[softEdge.p0].predictedPosition -
						soft0;
				const PxVec3 displacement1 =
					particles[softEdge.p1].predictedPosition -
						soft1;
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

				const PxU64 forwardOwnerStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				const PxU32 softVertices[2] =
					{softEdge.p0, softEdge.p1};
				bool forwardVertexOwns = false;
				for(PxU32 endpoint = 0;
					endpoint < 2 && !forwardVertexOwns;
					++endpoint)
				{
					const PxU32 vertexIndex =
						softVertices[endpoint];
					if(particles[vertexIndex].invMass <= 0.0f)
						continue;
					if(forwardOwnerQueryStats)
						forwardOwnerQueryStats->record(surfaceIndex, vertexIndex);
					forwardVertexOwns = parentForwardOwnerScratch ?
						avbdTriangleSurfaceForwardVertexOwnsSweepParentCached(
							surface, centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[vertexIndex], vertexIndex, margin, stats,
							queryScratch, *parentForwardOwnerScratch) :
						forwardOwnerResultCacheSurfaceSlot !=
						PX_MAX_U32 ?
						avbdTriangleSurfaceForwardVertexOwnsSweepCached(
							surface, forwardOwnerResultCacheSurfaceSlot,
							centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[vertexIndex], vertexIndex, margin, stats,
							queryScratch, *forwardOwnerResultCache) :
						avbdTriangleSurfaceForwardVertexOwnsSweep(
							surface, centerStart, centerEnd, rotationStart, rotationEnd,
							rotationsEquivalent, particles[vertexIndex], margin,
							stats, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptEdgeForwardOwnerNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						forwardOwnerStartNanos;
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
				PxArray<PxU32>& rigidEdgeCandidates = queryScratch
					? queryScratch->edgeBvhQueryCandidates
					: surface.edgeBvhQueryCandidates;
				const PxU64 bvhRecoveryStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				bool useRigidEdgeBvh = false;
				if(softEdgeTranslationOnly && rotationsEquivalent)
				{
					const PxQuat inverseRotation =
						rotationStart.getConjugate();
					const PxVec3 localSoft0 = inverseRotation.rotate(
						soft0 - centerStart);
					const PxVec3 localSoft1 = inverseRotation.rotate(
						soft1 - centerStart);
					const PxVec3 localRelativeTranslation =
						inverseRotation.rotate(relativeTranslation);
					const PxVec3 localSoftMinimum =
						localSoft0.minimum(localSoft1);
					const PxVec3 localSoftMaximum =
						localSoft0.maximum(localSoft1);
					// A rigid local edge E can enter the static local soft edge
					// only when E overlaps the soft edge swept backwards by the
					// relative translation. This conservative local AABB can
					// therefore recover owning triangle leaves.
					useRigidEdgeBvh =
						avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
							surface,
							localSoftMinimum.minimum(
								localSoftMinimum -
									localRelativeTranslation),
							localSoftMaximum.maximum(
								localSoftMaximum -
									localRelativeTranslation),
							margin, rigidEdgeCandidates, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptEdgeBvhRecoveryNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						bvhRecoveryStartNanos;
				// Rotation or a deforming soft edge has no independently
				// validated local translation envelope. Keep the legacy full
				// traversal as the conservative authority in those branches.
				const PxU32 rigidEdgeCount = useRigidEdgeBvh
					? rigidEdgeCandidates.size() : surface.edges.size();
				const PxU64 narrowPhaseStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				for(PxU32 rigidEdgeEntry = 0;
					rigidEdgeEntry < rigidEdgeCount;
					++rigidEdgeEntry)
				{
					const PxU32 rigidEdgeIndex = useRigidEdgeBvh
						? rigidEdgeCandidates[rigidEdgeEntry]
						: rigidEdgeEntry;
					const AvbdRigidTriangleSurfaceEdge& rigidEdge =
						surface.edges[rigidEdgeIndex];
					if(!rigidEdge.active ||
						rigidEdge.p0 >= surface.vertices.size() ||
						rigidEdge.p1 >= surface.vertices.size())
						continue;
					const PxVec3 rigid0 =
						centerStart + rotationStart.rotate(
							surface.vertices[
								rigidEdge.p0].point);
					const PxVec3 rigid1 =
						centerStart + rotationStart.rotate(
							surface.vertices[
								rigidEdge.p1].point);
					PxVec3 rigidMinimum(0.0f);
					PxVec3 rigidMaximum(0.0f);
					if(rotationsEquivalent)
					{
						rigidMinimum =
							rigid0.minimum(rigid1).
								minimum(
									rigid0 + relativeTranslation).
								minimum(
									rigid1 + relativeTranslation) -
									PxVec3(margin);
						rigidMaximum =
							rigid0.maximum(rigid1).
								maximum(
									rigid0 + relativeTranslation).
								maximum(
									rigid1 + relativeTranslation) +
									PxVec3(margin);
					}
					else
					{
						const PxReal rotationExtent =
							surface.localRadius + margin;
						rigidMinimum =
							centerStart.minimum(
								relativeCenterEnd) -
								PxVec3(rotationExtent);
						rigidMaximum =
							centerStart.maximum(
								relativeCenterEnd) +
								PxVec3(rotationExtent);
					}
					const PxVec3 softMinimum =
						soft0.minimum(soft1).
							minimum(soft0End).
							minimum(soft1End);
					const PxVec3 softMaximum =
						soft0.maximum(soft1).
							maximum(soft0End).
							maximum(soft1End);
					if(rigidMinimum.x > softMaximum.x ||
						rigidMaximum.x < softMinimum.x ||
						rigidMinimum.y > softMaximum.y ||
						rigidMaximum.y < softMinimum.y ||
						rigidMinimum.z > softMaximum.z ||
						rigidMaximum.z < softMinimum.z)
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceEdgeCandidates++;
						stats->rigidTriangleSurfaceEdgeTests++;
					}

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
								surface.vertices[
									rigidEdge.p0].point,
								surface.vertices[
									rigidEdge.p1].point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1, margin, entry)
							: avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
								surface.vertices[
									rigidEdge.p0].point,
								surface.vertices[
									rigidEdge.p1].point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1,
								soft0End, soft1End,
								margin, entry);
					if(!entered)
						continue;
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
					if(entry.normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54534553u,
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
					geometry.normal =
						entry.normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					const PxVec3 surfaceLocal =
						surface.vertices[rigidEdge.p0].point *
							(1.0f - entry.rigidWeight1) +
						surface.vertices[rigidEdge.p1].point *
							entry.rigidWeight1;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						surfaceLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							rigidEdge.friction,
							rigidEdge.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptEdgeNarrowPhaseNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						narrowPhaseStartNanos;
			}

			PxU32 softTriangleBegin = 0;
			PxU32 softTriangleEnd = 0;
			const PxU32 softTriangleCount =
				body.compiled.surfaceTriangles.size() / 3;
			if(!workItem)
				softTriangleEnd = softTriangleCount;
			else if(workItem->family == WorkItem::eSOFT_TRIANGLE)
			{
				softTriangleBegin = PxMin(workItem->primitiveBegin,
					softTriangleCount);
				softTriangleEnd = PxMin(
					PxMax(workItem->primitiveEnd, softTriangleBegin),
					softTriangleCount);
			}
			for(PxU32 softTriangleIndex = softTriangleBegin;
				softTriangleIndex < softTriangleEnd;
				++softTriangleIndex)
			{
				const PxU32 triangleOffset = softTriangleIndex * 3;
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

				const PxU64 forwardOwnerStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				const PxU32 triangleVertices[3] =
					{v0, v1, v2};
				bool forwardVertexOwns = false;
				for(PxU32 vertexIndex = 0;
					vertexIndex < 3 && !forwardVertexOwns;
					++vertexIndex)
				{
					const PxU32 particleIndex =
						triangleVertices[vertexIndex];
					if(particles[particleIndex].invMass <= 0.0f)
						continue;
					if(forwardOwnerQueryStats)
						forwardOwnerQueryStats->record(surfaceIndex, particleIndex);
					forwardVertexOwns = parentForwardOwnerScratch ?
						avbdTriangleSurfaceForwardVertexOwnsSweepParentCached(
							surface, centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[particleIndex], particleIndex, margin, stats,
							queryScratch, *parentForwardOwnerScratch) :
						forwardOwnerResultCacheSurfaceSlot !=
						PX_MAX_U32 ?
						avbdTriangleSurfaceForwardVertexOwnsSweepCached(
							surface, forwardOwnerResultCacheSurfaceSlot,
							centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[particleIndex], particleIndex, margin, stats,
							queryScratch, *forwardOwnerResultCache) :
						avbdTriangleSurfaceForwardVertexOwnsSweep(
							surface, centerStart, centerEnd, rotationStart, rotationEnd,
							rotationsEquivalent, particles[particleIndex], margin,
							stats, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptTriangleForwardOwnerNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						forwardOwnerStartNanos;
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
				PxArray<PxU32>& rigidVertexCandidates = queryScratch
					? queryScratch->vertexBvhQueryCandidates
					: surface.vertexBvhQueryCandidates;
				const PxU64 bvhRecoveryStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				bool useRigidVertexBvh = false;
				if(softTriangleTranslationOnly && rotationsEquivalent)
				{
					const PxQuat inverseRotation =
						rotationStart.getConjugate();
					const PxVec3 localP0 = inverseRotation.rotate(
						p0 - centerStart);
					const PxVec3 localP1 = inverseRotation.rotate(
						p1 - centerStart);
					const PxVec3 localP2 = inverseRotation.rotate(
						p2 - centerStart);
					const PxVec3 localRelativeTranslation =
						inverseRotation.rotate(relativeTranslation);
					const PxVec3 localTriangleMinimum =
						localP0.minimum(localP1).minimum(localP2);
					const PxVec3 localTriangleMaximum =
						localP0.maximum(localP1).maximum(localP2);
					// Transform the relative translation to the stationary
					// surface-local frame before triangle-leaf traversal.
					useRigidVertexBvh =
						avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
							surface,
							localTriangleMinimum.minimum(
								localTriangleMinimum -
									localRelativeTranslation),
							localTriangleMaximum.maximum(
								localTriangleMaximum -
									localRelativeTranslation),
							margin, rigidVertexCandidates, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptTriangleBvhRecoveryNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						bvhRecoveryStartNanos;
				// Rotation or soft-face deformation retains the exact legacy
				// scan until a separately proven swept local envelope exists.
				const PxU32 rigidVertexCount = useRigidVertexBvh
					? rigidVertexCandidates.size() : surface.vertices.size();
				const PxU64 narrowPhaseStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				for(PxU32 rigidVertexEntry = 0;
					rigidVertexEntry < rigidVertexCount;
					++rigidVertexEntry)
				{
					const PxU32 rigidVertexIndex = useRigidVertexBvh
						? rigidVertexCandidates[rigidVertexEntry]
						: rigidVertexEntry;
					const AvbdRigidTriangleSurfaceVertex& vertex =
						surface.vertices[rigidVertexIndex];
					if(!vertex.active)
						continue;
					const PxVec3 rigidVertexStart =
						centerStart +
							rotationStart.rotate(vertex.point);
					const PxVec3 rigidVertexEnd =
						rotationsEquivalent
							? rigidVertexStart +
								relativeTranslation
							: relativeCenterEnd +
								rotationEnd.rotate(vertex.point);
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
							surface.localRadius;
						sweptMinimum =
							centerStart.minimum(
								relativeCenterEnd) -
								PxVec3(rotationExtent);
						sweptMaximum =
							centerStart.maximum(
								relativeCenterEnd) +
								PxVec3(rotationExtent);
					}
					if(sweptMinimum.x > triangleMaximum.x ||
						sweptMaximum.x < triangleMinimum.x ||
						sweptMinimum.y > triangleMaximum.y ||
						sweptMaximum.y < triangleMinimum.y ||
						sweptMinimum.z > triangleMaximum.z ||
						sweptMaximum.z < triangleMinimum.z)
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceVertexCandidates++;
						stats->rigidTriangleSurfaceVertexTests++;
					}

					AvbdSweptTriangleEntry entry;
					const bool entered =
						softTriangleTranslationOnly &&
							rotationsEquivalent
							? avbdSegmentEnterExpandedTriangleNonVertex(
								rigidVertexStart, rigidVertexEnd,
								p0, p1, p2, margin, entry)
							: softTriangleTranslationOnly
							? avbdRotatingPointEnterExpandedTriangleFace(
								vertex.point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, margin, entry)
							: avbdRotatingPointEnterExpandedDeformingTriangleFace(
								vertex.point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, p0,
								p1 + relativeDisplacement1,
								p2 + relativeDisplacement2,
								margin, entry);
					if(!entered ||
						entry.feature != AVBD_FEATURE_FACE)
						continue;
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(vertex.outward);
					if(entry.normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54535653u,
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
					geometry.normal =
						entry.normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						vertex.point);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							vertex.friction,
							vertex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptTriangleNarrowPhaseNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						narrowPhaseStartNanos;
			}
		}
	}
}
void avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch,
	const AvbdRigidTriangleSurfaceFeatureWorkItem* workItem,
	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats*
		discreteQueryStats,
	bool useBodyLocalBoundsCull)
{
	typedef AvbdRigidTriangleSurfaceFeatureWorkItem WorkItem;
	if(workItem && workItem->phase != WorkItem::eDISCRETE)
		return;
	const PxReal featureEpsilon = 1.0e-4f;
	const PxReal distanceEpsilon = 1.0e-8f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		if(workItem && bodyIndex != workItem->bodyIndex)
			continue;
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

		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			if(workItem && surfaceIndex != workItem->surfaceIndex)
				continue;
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			if(!avbdIsRigidTriangleSurfaceValid(surface))
				continue;
			const PxReal broadphaseRadius =
				surface.localRadius + margin;
			if(bodyMinimum.x >
					surface.center.x + broadphaseRadius ||
				bodyMaximum.x <
					surface.center.x - broadphaseRadius ||
				bodyMinimum.y >
					surface.center.y + broadphaseRadius ||
				bodyMaximum.y <
					surface.center.y - broadphaseRadius ||
				bodyMinimum.z >
					surface.center.z + broadphaseRadius ||
				bodyMaximum.z <
					surface.center.z - broadphaseRadius)
				continue;
			const PxQuat inverseRotation =
				surface.rotation.getConjugate();
			if(useBodyLocalBoundsCull &&
				!avbdRigidTriangleSurfaceBodyMayReachLocalBounds(
					bodyMinimum, bodyMaximum, surface, inverseRotation, margin))
				continue;

			PxU32 softEdgeBegin = 0;
			PxU32 softEdgeEnd = 0;
			if(!workItem)
				softEdgeEnd = body.compiled.surfaceEdges.size();
			else if(workItem->family == WorkItem::eSOFT_EDGE)
			{
				softEdgeBegin = PxMin(workItem->primitiveBegin,
					body.compiled.surfaceEdges.size());
				softEdgeEnd = PxMin(
					PxMax(workItem->primitiveEnd, softEdgeBegin),
					body.compiled.surfaceEdges.size());
			}
			for(PxU32 softEdgeIndex = softEdgeBegin;
				softEdgeIndex < softEdgeEnd;
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
							surface.center);
				const PxVec3 soft1Local =
					inverseRotation.rotate(
						particles[softEdge.p1].position -
							surface.center);
				const PxVec3 softMinimum =
					soft0Local.minimum(soft1Local);
				const PxVec3 softMaximum =
					soft0Local.maximum(soft1Local);
				PxArray<PxU32>& rigidEdgeCandidates = queryScratch
					? queryScratch->edgeBvhQueryCandidates
					: surface.edgeBvhQueryCandidates;
				const bool useEdgeBvh =
					avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
						surface, softMinimum, softMaximum, margin,
						rigidEdgeCandidates, queryScratch);
				const PxU32 rigidEdgeCount = useEdgeBvh
					? rigidEdgeCandidates.size() : surface.edges.size();
				if(discreteQueryStats)
				{
					const PxU32 triangleCandidateCount = useEdgeBvh
						? (queryScratch
							? queryScratch->triangleBvhQueryCandidates.size()
							: surface.triangleBvhQueryCandidates.size()) : 0;
					discreteQueryStats->recordEdgeQuery(
						useEdgeBvh, triangleCandidateCount, rigidEdgeCount);
				}
				for(PxU32 rigidEdgeEntry = 0;
					rigidEdgeEntry < rigidEdgeCount;
					++rigidEdgeEntry)
				{
					const PxU32 rigidEdgeIndex = useEdgeBvh
						? rigidEdgeCandidates[rigidEdgeEntry] :
							rigidEdgeEntry;
					const AvbdRigidTriangleSurfaceEdge& rigidEdge =
						surface.edges[rigidEdgeIndex];
					if(!rigidEdge.active ||
						rigidEdge.p0 >= surface.vertices.size() ||
						rigidEdge.p1 >= surface.vertices.size())
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceEdgeCandidates++;
						stats->rigidTriangleSurfaceEdgeTests++;
					}
					const PxVec3& rigid0Local =
						surface.vertices[rigidEdge.p0].point;
					const PxVec3& rigid1Local =
						surface.vertices[rigidEdge.p1].point;
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
					PxVec3 softClosestLocal;
					PxVec3 rigidClosestLocal;
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
					const PxVec3 deltaLocal =
						softClosestLocal - rigidClosestLocal;
					const PxReal distance =
						deltaLocal.magnitude();
					if(!PxIsFinite(distance) ||
						distance >= margin ||
						deltaLocal.dot(rigidEdge.outward) <
							-1.0e-5f)
						continue;
					PxVec3 normalLocal =
						distance > distanceEpsilon
							? deltaLocal * (1.0f / distance)
							: rigidEdge.outward;
					if(!normalLocal.isFinite() ||
						normalLocal.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54534545u,
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
						surface.rotation.rotate(normalLocal).
							getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - distance;
					geometry.margin = margin;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						rigidClosestLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							rigidEdge.friction,
							rigidEdge.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}

			PxU32 softTriangleBegin = 0;
			PxU32 softTriangleEnd = 0;
			const PxU32 softTriangleCount =
				body.compiled.surfaceTriangles.size() / 3;
			if(!workItem)
				softTriangleEnd = softTriangleCount;
			else if(workItem->family == WorkItem::eSOFT_TRIANGLE)
			{
				softTriangleBegin = PxMin(workItem->primitiveBegin,
					softTriangleCount);
				softTriangleEnd = PxMin(
					PxMax(workItem->primitiveEnd, softTriangleBegin),
					softTriangleCount);
			}
			for(PxU32 softTriangleIndex = softTriangleBegin;
				softTriangleIndex < softTriangleEnd;
				++softTriangleIndex)
			{
				const PxU32 triangleOffset = softTriangleIndex * 3;
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
					inverseRotation.rotate(p0 - surface.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - surface.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - surface.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local);
				PxArray<PxU32>& rigidVertexCandidates = queryScratch
					? queryScratch->vertexBvhQueryCandidates
					: surface.vertexBvhQueryCandidates;
				const bool useVertexBvh =
					avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
						surface, triangleMinimum, triangleMaximum, margin,
						rigidVertexCandidates, queryScratch);
				const PxU32 rigidVertexCount = useVertexBvh
					? rigidVertexCandidates.size() :
						surface.vertices.size();
				if(discreteQueryStats)
				{
					const PxU32 triangleCandidateCount = useVertexBvh
						? (queryScratch
							? queryScratch->triangleBvhQueryCandidates.size()
							: surface.triangleBvhQueryCandidates.size()) : 0;
					discreteQueryStats->recordTriangleQuery(
						useVertexBvh, triangleCandidateCount, rigidVertexCount);
				}

				for(PxU32 rigidVertexEntry = 0;
					rigidVertexEntry < rigidVertexCount;
					++rigidVertexEntry)
				{
					const PxU32 rigidVertexIndex = useVertexBvh
						? rigidVertexCandidates[rigidVertexEntry] :
							rigidVertexEntry;
					const AvbdRigidTriangleSurfaceVertex& vertex =
						surface.vertices[rigidVertexIndex];
					if(!vertex.active ||
						vertex.point.x < triangleMinimum.x - margin ||
						vertex.point.x > triangleMaximum.x + margin ||
						vertex.point.y < triangleMinimum.y - margin ||
						vertex.point.y > triangleMaximum.y + margin ||
						vertex.point.z < triangleMinimum.z - margin ||
						vertex.point.z > triangleMaximum.z + margin)
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceVertexCandidates++;
						stats->rigidTriangleSurfaceVertexTests++;
					}
					const PxVec3 rigidVertexWorld =
						surface.center +
							surface.rotation.rotate(vertex.point);
					const AvbdClosestPointResult closest =
						avbdClosestPointOnTriangleOGC(
							rigidVertexWorld, p0, p1, p2);
					if(closest.feature != AVBD_FEATURE_FACE ||
						!PxIsFinite(closest.distance) ||
						closest.distance >= margin)
						continue;
					const PxVec3 outwardWorld =
						surface.rotation.rotate(vertex.outward);
					const PxVec3 deltaWorld =
						closest.point - rigidVertexWorld;
					if(deltaWorld.dot(outwardWorld) < -1.0e-5f)
						continue;
					PxVec3 normalWorld =
						closest.distance > distanceEpsilon
							? deltaWorld *
								(1.0f / closest.distance)
							: outwardWorld;
					if(!normalWorld.isFinite() ||
						normalWorld.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54535646u,
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
					geometry.normal = normalWorld.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth =
						margin - closest.distance;
					geometry.margin = margin;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						vertex.point);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							vertex.friction,
							vertex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}
		}
	}
}

// P5.17c candidate leaf: consume a contiguous interval of the canonical
// feature plan. Each call owns its contact output and complete query scratch;
// parent code must stable-merge outputs by plan index. The serial suffixes use
// the same row filter internally, so the leaf cannot introduce a second
// feature predicate, traversal order or BVH ownership model.
void avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidTriangleSurfaceFeaturePlan& plan,
	PxU32 planBegin, PxU32 planEnd,
	PxArray<AvbdSoftContact>& contacts,
	AvbdRigidTriangleSurfaceQueryScratch& queryScratch,
	PxReal margin,
	AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceFeaturePlanRangeTiming* timing,
	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		sweptSubstageTiming,
	AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
		forwardOwnerQueryStats,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache*
		forwardOwnerResultCache,
	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats*
		discreteQueryStats,
	bool useDiscreteBodyLocalBoundsCull)
{
	PX_ASSERT(planBegin <= planEnd && planEnd <= plan.items.size());
	const PxU32 clampedBegin = PxMin(planBegin, plan.items.size());
	const PxU32 clampedEnd = PxMin(PxMax(planEnd, clampedBegin),
		plan.items.size());
	for(PxU32 planIndex = clampedBegin;
		planIndex < clampedEnd; ++planIndex)
	{
		const AvbdRigidTriangleSurfaceFeatureWorkItem& workItem =
			plan.items[planIndex];
		PX_ASSERT(workItem.bodyIndex < numSoftBodies);
		PX_ASSERT(workItem.surfaceIndex < numSurfaces);
		if(workItem.bodyIndex >= numSoftBodies ||
			workItem.surfaceIndex >= numSurfaces)
			continue;
		const PxU64 workItemStartNanos = timing ?
			PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
		if(workItem.phase ==
			AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT)
		{
			avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
				particles, numParticles, surfaces, numSurfaces,
				softBodies, numSoftBodies, contacts, margin, stats,
				&queryScratch, NULL, &workItem, sweptSubstageTiming,
				forwardOwnerQueryStats, forwardOwnerResultCache);
		}
		else
		{
			avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
				particles, numParticles, surfaces, numSurfaces,
				softBodies, numSoftBodies, contacts, margin, stats,
				&queryScratch, &workItem, discreteQueryStats,
				useDiscreteBodyLocalBoundsCull);
		}
		if(timing)
			timing->record(workItem,
				PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
				workItemStartNanos);
	}
}

} // namespace Dy
} // namespace physx
