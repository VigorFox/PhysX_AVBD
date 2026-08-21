// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactSelf.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"

namespace physx
{
namespace Dy
{

// AVBD self-collision OGC geometry and topology core.
//
// This unit owns self-collision filtering and the canonical vertex/edge
// feature detector. Topology construction lives in
// DyAvbdSelfCollisionTopology.cpp; range wrappers remain in
// DyAvbdContactSelf.inl so preparation and fan-out stay separate.
//
// The original namespace, feature ownership, and traversal order are kept
// unchanged for a behavior-preserving extraction.
// =============================================================================

// PATH 4 (OGC): Full Self-Collision Detection
//
// Two-stage C2 activation, topological adjacency filtering, safety bubble.
// =============================================================================

PX_FORCE_INLINE bool avbdIsAdjacentSelfCollision(
	PxU32 localA, PxU32 localB,
	const PxArray<PxArray<PxU32> >& adj)
{
	if (localA >= adj.size()) return false;
	const PxArray<PxU32>& a = adj[localA];
	// Binary search in sorted array
	PxU32 lo = 0, hi = a.size();
	while (lo < hi) {
		PxU32 mid = (lo + hi) / 2;
		if (a[mid] < localB) lo = mid + 1;
		else if (a[mid] > localB) hi = mid;
		else return true;
	}
	return false;
}

// Per-vertex conservative displacement bound (Eq. 21)
void avbdComputeSafetyBounds(
	const AvbdSoftBody& sb,
	const AvbdSoftParticle* particles,
	const PxArray<PxArray<PxU32> >& adj,
	PxReal queryRadius,
	PxReal gammaP,
	PxArray<PxReal>& bounds,
	AvbdSoftContactWorkspace& workspace)
{
	PX_UNUSED(adj);
	const PxU32 particleCount = sb.compiled.particleCount;
	const PxU32 particleStart = sb.compiled.particleStart;
	const PxReal rq = PxMax(queryRadius, 1.0e-6f);
	const PxReal gamma = PxClamp(gammaP, 1.0e-4f, 0.499f);
	const PxReal filterDistance =
		PxMax(sb.compiled.selfCollisionFilterDistance, 0.0f);
	const bool hasRestFilter =
		filterDistance > 0.0f &&
		sb.compiled.selfCollisionRestPositions.size() == particleCount;

	bounds.resize(particleCount);
	const PxU32 triangleCount = sb.compiled.surfaceTriangles.size() / 3;
	const PxU32 edgeCount = sb.compiled.surfaceEdges.size();
	// Detection and safety execute in separate outer-epoch phases.  Reuse the
	// same caller-owned sweep buffers here, but preserve the serial traversal,
	// sort predicates and floating-point reduction order below.
	workspace.reserveSweepScratch(
		workspace.selfSafetyTriangleMinimums, triangleCount);
	workspace.reserveSweepScratch(
		workspace.selfSafetyEdgeMinimums, edgeCount);
	workspace.reserveSweepScratch(
		workspace.selfTriangleBounds, triangleCount);
	workspace.reserveSweepScratch(
		workspace.selfSortedVertices,
		sb.compiled.surfaceVertices.size());
	workspace.reserveSweepScratch(
		workspace.selfActiveTriangles, triangleCount);
	workspace.reserveSweepScratch(workspace.selfEdgeBounds, edgeCount);
	PxArray<PxReal>& triangleMinimums =
		workspace.selfSafetyTriangleMinimums;
	PxArray<PxReal>& edgeMinimums =
		workspace.selfSafetyEdgeMinimums;
	triangleMinimums.resize(triangleCount);
	edgeMinimums.resize(edgeCount);
	for(PxU32 vertexIndex = 0;
		vertexIndex < particleCount; vertexIndex++)
		bounds[vertexIndex] = rq;
	for(PxU32 triangleIndex = 0;
		triangleIndex < triangleMinimums.size(); triangleIndex++)
		triangleMinimums[triangleIndex] = rq;
	for(PxU32 edgeIndex = 0;
		edgeIndex < edgeMinimums.size(); edgeIndex++)
		edgeMinimums[edgeIndex] = rq;

	PxArray<AvbdSelfCollisionTriangleBounds>& triangleBounds =
		workspace.selfTriangleBounds;
	triangleBounds.clear();
	for(PxU32 triangleOffset = 0;
		triangleOffset + 2 <
			sb.compiled.surfaceTriangles.size();
		triangleOffset += 3)
	{
		const PxU32 vertex0 =
			sb.compiled.surfaceTriangles[triangleOffset];
		const PxU32 vertex1 =
			sb.compiled.surfaceTriangles[triangleOffset + 1];
		const PxU32 vertex2 =
			sb.compiled.surfaceTriangles[triangleOffset + 2];
		if(vertex0 < particleStart ||
			vertex1 < particleStart ||
			vertex2 < particleStart ||
			vertex0 - particleStart >= particleCount ||
			vertex1 - particleStart >= particleCount ||
			vertex2 - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionTriangleBounds triangle;
		triangle.triangleOffset = triangleOffset;
		triangle.minimum =
			particles[vertex0].position.minimum(
				particles[vertex1].position).minimum(
				particles[vertex2].position);
		triangle.maximum =
			particles[vertex0].position.maximum(
				particles[vertex1].position).maximum(
				particles[vertex2].position);
		triangleBounds.pushBack(triangle);
	}
	PxSort(
		triangleBounds.begin(), triangleBounds.size(),
		[](const AvbdSelfCollisionTriangleBounds& a,
		   const AvbdSelfCollisionTriangleBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});
	PxArray<AvbdSelfCollisionVertexSweepEntry>& sortedVertices =
		workspace.selfSortedVertices;
	sortedVertices.clear();
	for(PxU32 surfaceVertexIndex = 0;
		surfaceVertexIndex <
			sb.compiled.surfaceVertices.size();
		surfaceVertexIndex++)
	{
		const PxU32 globalIndex =
			sb.compiled.surfaceVertices[surfaceVertexIndex];
		if(globalIndex < particleStart ||
			globalIndex - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionVertexSweepEntry vertex;
		vertex.localIndex = globalIndex - particleStart;
		vertex.minimumX =
			particles[globalIndex].position.x;
		vertex.maximumX = vertex.minimumX;
		sortedVertices.pushBack(vertex);
	}
	PxSort(
		sortedVertices.begin(), sortedVertices.size(),
		[](const AvbdSelfCollisionVertexSweepEntry& a,
		   const AvbdSelfCollisionVertexSweepEntry& b)
		{
			return a.minimumX < b.minimumX;
		});

	// OGC Eq. 22 and Eq. 26.  The sweep-and-prune list is the CPU
	// equivalent of the paper's facet-BVH radius query.  Values are
	// initialized to rq, so pairs outside the query shell cannot reduce the
	// conservative bound.
	PxArray<PxU32>& activeTriangles = workspace.selfActiveTriangles;
	activeTriangles.clear();
	PxU32 triangleCursor = 0;
	for(PxU32 sortedVertexIndex = 0;
		sortedVertexIndex < sortedVertices.size();
		sortedVertexIndex++)
	{
		const PxU32 localIndex =
			sortedVertices[sortedVertexIndex].localIndex;
		const PxU32 globalIndex = particleStart + localIndex;
		const PxVec3& point = particles[globalIndex].position;
		while(triangleCursor < triangleBounds.size() &&
			triangleBounds[triangleCursor].minimum.x <=
				point.x + rq)
			activeTriangles.pushBack(triangleCursor++);

		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size();)
		{
			const AvbdSelfCollisionTriangleBounds& triangle =
				triangleBounds[activeTriangles[activeIndex]];
			if(triangle.maximum.x < point.x - rq)
			{
				activeTriangles[activeIndex] =
					activeTriangles.back();
				activeTriangles.popBack();
				continue;
			}
			activeIndex++;
			if(triangle.minimum.y > point.y + rq ||
				triangle.maximum.y < point.y - rq ||
				triangle.minimum.z > point.z + rq ||
				triangle.maximum.z < point.z - rq)
				continue;

			const PxU32 triangleOffset =
				triangle.triangleOffset;
			const PxU32 vertex0 =
				sb.compiled.surfaceTriangles[triangleOffset];
			const PxU32 vertex1 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 1];
			const PxU32 vertex2 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 2];
			if(globalIndex == vertex0 ||
				globalIndex == vertex1 ||
				globalIndex == vertex2)
				continue;
			if(hasRestFilter)
			{
				if(sb.compiled.selfCollisionRestFilterCacheValid)
				{
					if(avbdIsSelfRestVertexTriangleFiltered(
						sb, localIndex, triangleOffset / 3))
						continue;
				}
				else
				{
					const PxArray<PxVec3>& restPositions =
						sb.compiled.selfCollisionRestPositions;
					const AvbdClosestPointResult restClosest =
						avbdClosestPointOnTriangleOGC(
							restPositions[localIndex],
							restPositions[vertex0 - particleStart],
							restPositions[vertex1 - particleStart],
							restPositions[vertex2 - particleStart]);
					if(restClosest.distance <= filterDistance)
						continue;
				}
			}
			const AvbdClosestPointResult closest =
				avbdClosestPointOnTriangleOGC(
					point,
					particles[vertex0].position,
					particles[vertex1].position,
					particles[vertex2].position);
			if(closest.distance >= rq)
				continue;
			bounds[localIndex] = PxMin(
				bounds[localIndex],
				closest.distance);
			const PxU32 triangleIndex = triangleOffset / 3;
			triangleMinimums[triangleIndex] = PxMin(
				triangleMinimums[triangleIndex],
				closest.distance);
		}
	}

	PxArray<AvbdSelfCollisionEdgeBounds>& edgeBounds =
		workspace.selfEdgeBounds;
	edgeBounds.clear();
	for(PxU32 edgeIndex = 0;
		edgeIndex < sb.compiled.surfaceEdges.size(); edgeIndex++)
	{
		const AvbdEdgeInfo& edge =
			sb.compiled.surfaceEdges[edgeIndex];
		if(edge.p0 < particleStart ||
			edge.p1 < particleStart ||
			edge.p0 - particleStart >= particleCount ||
			edge.p1 - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionEdgeBounds edgeBound;
		edgeBound.edgeIndex = edgeIndex;
		edgeBound.minimum =
			particles[edge.p0].position.minimum(
				particles[edge.p1].position);
		edgeBound.maximum =
			particles[edge.p0].position.maximum(
				particles[edge.p1].position);
		edgeBounds.pushBack(edgeBound);
	}
	PxSort(
		edgeBounds.begin(), edgeBounds.size(),
		[](const AvbdSelfCollisionEdgeBounds& a,
		   const AvbdSelfCollisionEdgeBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});

	// OGC Eq. 24.  Every unordered non-incident edge pair contributes its
	// distance to both edge minima.
	for(PxU32 sortedEdge0 = 0;
		sortedEdge0 < edgeBounds.size(); sortedEdge0++)
	{
		const AvbdSelfCollisionEdgeBounds& bounds0 =
			edgeBounds[sortedEdge0];
		for(PxU32 sortedEdge1 = sortedEdge0 + 1;
			sortedEdge1 < edgeBounds.size(); sortedEdge1++)
		{
			const AvbdSelfCollisionEdgeBounds& bounds1 =
				edgeBounds[sortedEdge1];
			if(bounds1.minimum.x > bounds0.maximum.x + rq)
				break;
			if(bounds0.minimum.y > bounds1.maximum.y + rq ||
				bounds0.maximum.y < bounds1.minimum.y - rq ||
				bounds0.minimum.z > bounds1.maximum.z + rq ||
				bounds0.maximum.z < bounds1.minimum.z - rq)
				continue;
			const AvbdEdgeInfo& edge0 =
				sb.compiled.surfaceEdges[bounds0.edgeIndex];
			const AvbdEdgeInfo& edge1 =
				sb.compiled.surfaceEdges[bounds1.edgeIndex];
			if(edge0.p0 == edge1.p0 ||
				edge0.p0 == edge1.p1 ||
				edge0.p1 == edge1.p0 ||
				edge0.p1 == edge1.p1)
				continue;
			if(hasRestFilter)
			{
				const PxArray<PxVec3>& restPositions =
					sb.compiled.selfCollisionRestPositions;
				PxReal restWeight0 = 0.0f;
				PxReal restWeight1 = 0.0f;
				PxVec3 restClosest0, restClosest1;
				avbdClosestPointsOnSegments(
					restPositions[edge0.p0 - particleStart],
					restPositions[edge0.p1 - particleStart],
					restPositions[edge1.p0 - particleStart],
					restPositions[edge1.p1 - particleStart],
					restWeight0, restWeight1,
					restClosest0, restClosest1);
				if((restClosest0 - restClosest1).
						magnitude() <= filterDistance)
					continue;
			}
			PxReal weight0 = 0.0f;
			PxReal weight1 = 0.0f;
			PxVec3 closest0, closest1;
			avbdClosestPointsOnSegments(
				particles[edge0.p0].position,
				particles[edge0.p1].position,
				particles[edge1.p0].position,
				particles[edge1.p1].position,
				weight0, weight1, closest0, closest1);
			const PxReal distance =
				(closest0 - closest1).magnitude();
			if(distance >= rq)
				continue;
			edgeMinimums[bounds0.edgeIndex] = PxMin(
				edgeMinimums[bounds0.edgeIndex], distance);
			edgeMinimums[bounds1.edgeIndex] = PxMin(
				edgeMinimums[bounds1.edgeIndex], distance);
		}
	}

	// OGC Eq. 21, 23, and 25: gather the incident edge and triangle
	// minima onto each vertex, then apply gamma_p.
	for(PxU32 triangleOffset = 0;
		triangleOffset + 2 <
			sb.compiled.surfaceTriangles.size();
		triangleOffset += 3)
	{
		const PxReal triangleMinimum =
			triangleMinimums[triangleOffset / 3];
		for(PxU32 corner = 0; corner < 3; corner++)
		{
			const PxU32 globalIndex =
				sb.compiled.surfaceTriangles[
					triangleOffset + corner];
			if(globalIndex >= particleStart &&
				globalIndex - particleStart < particleCount)
				bounds[globalIndex - particleStart] = PxMin(
					bounds[globalIndex - particleStart],
					triangleMinimum);
		}
	}
	for(PxU32 edgeIndex = 0;
		edgeIndex < sb.compiled.surfaceEdges.size(); edgeIndex++)
	{
		const AvbdEdgeInfo& edge =
			sb.compiled.surfaceEdges[edgeIndex];
		const PxReal edgeMinimum = edgeMinimums[edgeIndex];
		if(edge.p0 >= particleStart &&
			edge.p0 - particleStart < particleCount)
			bounds[edge.p0 - particleStart] = PxMin(
				bounds[edge.p0 - particleStart], edgeMinimum);
		if(edge.p1 >= particleStart &&
			edge.p1 - particleStart < particleCount)
			bounds[edge.p1 - particleStart] = PxMin(
				bounds[edge.p1 - particleStart], edgeMinimum);
	}
	for(PxU32 localIndex = 0;
		localIndex < particleCount; localIndex++)
		bounds[localIndex] =
			gamma * PxMax(bounds[localIndex], 1.0e-6f);
}

// Detect self-collision contacts within a single soft body
void avbdDetectSelfCollisionOGC(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& sb,
	PxU32 softBodyIdx,
	const PxArray<PxArray<PxU32> >& adj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace* persistentWorkspace,
	// When supplied, the immutable stress coefficients and both BVH refits
	// belong to a parent transaction.  The caller must provide a distinct
	// persistentWorkspace for this range leaf: candidates, emitted-feature
	// keys and all output remain private to that leaf.
	const AvbdSoftContactWorkspace* preparedBvhWorkspace,
	PxU32 vertexLoopBegin,
	PxU32 vertexLoopEnd,
	PxU32 edgeLoopBegin,
	PxU32 edgeLoopEnd)
{
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	const bool usePreparedBvhWorkspace = preparedBvhWorkspace != NULL;
	workspace.reserveSelfCollisionSweep(
		sb.compiled.tetElements.size(),
		sb.compiled.surfaceTriangles.size() / 3,
		sb.compiled.surfaceVertices.size(),
		sb.compiled.surfaceEdges.size());
	if(!usePreparedBvhWorkspace)
		workspace.prepareSelfBvhBounds(
			sb.compiled.surfaceTriangleBvhNodes.size(),
			sb.compiled.surfaceEdgeBvhNodes.size());
	PxReal r   = params.contactRadius;
	PxReal tau = params.getTau();
	PX_UNUSED(adj);
	const bool sweepEnabled =
		sb.compiled.speculativeCCDEnabled;
	const PxReal filterDistance =
		PxMax(sb.compiled.selfCollisionFilterDistance, 0.0f);
	const bool hasRestFilter =
		filterDistance > 0.0f &&
		sb.compiled.selfCollisionRestPositions.size() ==
			sb.compiled.particleCount;
	PX_ASSERT(
		filterDistance == 0.0f ||
		sb.compiled.selfCollisionRestPositions.size() ==
			sb.compiled.particleCount);
	PxArray<PxReal>& localTetStressCoefficients =
		workspace.selfTetStressCoefficients;
	const PxArray<PxReal>* tetStressCoefficients =
		usePreparedBvhWorkspace
			? &preparedBvhWorkspace->selfTetStressCoefficients
			: &localTetStressCoefficients;
	if(!usePreparedBvhWorkspace)
	{
		localTetStressCoefficients.clear();
		if(!sb.compiled.tetElements.empty())
		{
			localTetStressCoefficients.resize(
				sb.compiled.tetElements.size());
			for(PxU32 tetIndex = 0;
				tetIndex < sb.compiled.tetElements.size();
				tetIndex++)
			{
				localTetStressCoefficients[tetIndex] =
					avbdComputeTetStressCoefficient(
						sb.compiled.tetElements[tetIndex],
						particles);
			}
		}
	}
	auto targetStressAllowsTriangle =
		[&](PxU32 triangleOffset) -> bool
	{
		if(tetStressCoefficients->empty())
			return true;
		const PxU32 triangleIndex = triangleOffset / 3;
		if(triangleIndex >=
			sb.compiled.surfaceTriangleTetElementIndices.size())
			return true;
		const PxU32 tetElementIndex =
			sb.compiled.surfaceTriangleTetElementIndices[
				triangleIndex];
		if(tetElementIndex < tetStressCoefficients->size())
			return (*tetStressCoefficients)[tetElementIndex] <=
				sb.compiled.selfCollisionStressTolerance;
		// Preserve the previous behavior when a source tetrahedron was skipped
		// during element compilation: no known compiled stress owner means the
		// boundary triangle remains eligible for self collision.
		return true;
	};
	// Keep the topology-stable hierarchy refitted from the authoritative
	// current/swept particle positions.  The legacy x-sweep remains the exact
	// fallback for bodies with no compiled hierarchy and for the benchmark
	// switch above; no contact-owner policy changes with this choice.
	const bool useSurfaceTriangleBvh = usePreparedBvhWorkspace ||
		(avbdUseSurfaceTriangleBvh() &&
		 !sb.compiled.surfaceTriangleBvhNodes.empty());
	PX_ASSERT(!usePreparedBvhWorkspace ||
		!sb.compiled.surfaceTriangleBvhNodes.empty());
	if(useSurfaceTriangleBvh && !usePreparedBvhWorkspace)
	{
		sb.compiled.refitSurfaceTriangleBvh(
			particles, sweepEnabled, workspace.selfTriangleBvhBounds);
		if(stats)
			stats->surfaceTriangleBvhRefitNodes +=
				sb.compiled.surfaceTriangleBvhNodes.size();
	}
	const PxArray<AvbdSurfaceBvhNodeBounds>& triangleBvhBounds =
		usePreparedBvhWorkspace
			? preparedBvhWorkspace->selfTriangleBvhBounds
			: workspace.selfTriangleBvhBounds;
	PX_ASSERT(!usePreparedBvhWorkspace ||
		triangleBvhBounds.size() ==
			sb.compiled.surfaceTriangleBvhNodes.size());

	PxArray<AvbdSelfCollisionTriangleBounds>& triangleBounds =
		workspace.selfTriangleBounds;
	triangleBounds.clear();
	if(!useSurfaceTriangleBvh)
	{
		for(PxU32 triangleOffset = 0;
			triangleOffset + 2 <
				sb.compiled.surfaceTriangles.size();
			triangleOffset += 3)
		{
			const PxU32 source0 =
				sb.compiled.surfaceTriangles[triangleOffset];
			const PxU32 source1 =
				sb.compiled.surfaceTriangles[triangleOffset + 1];
			const PxU32 source2 =
				sb.compiled.surfaceTriangles[triangleOffset + 2];
			AvbdSelfCollisionTriangleBounds triangle;
			triangle.triangleOffset = triangleOffset;
			triangle.minimum =
				particles[source0].position.minimum(
					particles[source1].position).minimum(
					particles[source2].position);
			triangle.maximum =
				particles[source0].position.maximum(
					particles[source1].position).maximum(
					particles[source2].position);
			if(sweepEnabled)
			{
				triangle.minimum = triangle.minimum.minimum(
					particles[source0].initialPosition).minimum(
					particles[source1].initialPosition).minimum(
					particles[source2].initialPosition);
				triangle.maximum = triangle.maximum.maximum(
					particles[source0].initialPosition).maximum(
					particles[source1].initialPosition).maximum(
					particles[source2].initialPosition);
			}
			triangleBounds.pushBack(triangle);
			if(stats)
				stats->selfTriangleBoundsBuilt++;
		}
	}
	if(!useSurfaceTriangleBvh)
		PxSort(
			triangleBounds.begin(), triangleBounds.size(),
			[](const AvbdSelfCollisionTriangleBounds& a,
			   const AvbdSelfCollisionTriangleBounds& b)
			{
				return a.minimum.x < b.minimum.x;
			});
	PxArray<AvbdSelfCollisionVertexSweepEntry>& sortedVertices =
		workspace.selfSortedVertices;
	sortedVertices.clear();
	if(!useSurfaceTriangleBvh)
	{
		for(PxU32 surfaceVertexIndex = 0;
			surfaceVertexIndex <
				sb.compiled.surfaceVertices.size();
			surfaceVertexIndex++)
		{
			const PxU32 globalIndex =
				sb.compiled.surfaceVertices[surfaceVertexIndex];
			if(globalIndex < sb.compiled.particleStart ||
				globalIndex - sb.compiled.particleStart >=
				sb.compiled.particleCount)
				continue;
			AvbdSelfCollisionVertexSweepEntry vertex;
			vertex.localIndex =
				globalIndex - sb.compiled.particleStart;
			vertex.minimumX = particles[globalIndex].position.x;
			vertex.maximumX = particles[globalIndex].position.x;
			if(sweepEnabled)
			{
				vertex.minimumX = PxMin(
					vertex.minimumX,
					particles[globalIndex].initialPosition.x);
				vertex.maximumX = PxMax(
					vertex.maximumX,
					particles[globalIndex].initialPosition.x);
			}
			sortedVertices.pushBack(vertex);
			if(stats)
				stats->selfVertexSweepEntriesBuilt++;
		}
	}
	if(!useSurfaceTriangleBvh)
		PxSort(
			sortedVertices.begin(), sortedVertices.size(),
			[](const AvbdSelfCollisionVertexSweepEntry& a,
			   const AvbdSelfCollisionVertexSweepEntry& b)
			{
				return a.minimumX < b.minimumX;
			});

	// Radius-query broadphase.  The previous all-vertices by all-triangles
	// traversal made each OGC redetection O(V*T).
	PxArray<PxU32>& activeTriangles = workspace.selfActiveTriangles;
	activeTriangles.clear();
	PxArray<PxU32>& triangleCandidates =
		workspace.selfTriangleCandidates;
	triangleCandidates.clear();
	PxArray<PxU64>& emittedFeatureKeys = workspace.selfEmittedFeatureKeys;
	emittedFeatureKeys.clear();
	auto triangleOverlapsQuery =
		[&](PxU32 triangleOffset, const PxVec3& queryMinimum,
			const PxVec3& queryMaximum) -> bool
	{
		const PxU32 source0 =
			sb.compiled.surfaceTriangles[triangleOffset];
		const PxU32 source1 =
			sb.compiled.surfaceTriangles[triangleOffset + 1];
		const PxU32 source2 =
			sb.compiled.surfaceTriangles[triangleOffset + 2];
		PxVec3 minimum = particles[source0].position.minimum(
			particles[source1].position).minimum(
			particles[source2].position);
		PxVec3 maximum = particles[source0].position.maximum(
			particles[source1].position).maximum(
			particles[source2].position);
		if(sweepEnabled)
		{
			minimum = minimum.minimum(
				particles[source0].initialPosition).minimum(
				particles[source1].initialPosition).minimum(
				particles[source2].initialPosition);
			maximum = maximum.maximum(
				particles[source0].initialPosition).maximum(
				particles[source1].initialPosition).maximum(
				particles[source2].initialPosition);
		}
		return !(minimum.x > queryMaximum.x + r ||
			maximum.x < queryMinimum.x - r ||
			minimum.y > queryMaximum.y + r ||
			maximum.y < queryMinimum.y - r ||
			minimum.z > queryMaximum.z + r ||
			maximum.z < queryMinimum.z - r);
	};
	PxU32 triangleCursor = 0;
	const PxU32 vertexLoopCount = useSurfaceTriangleBvh
		? sb.compiled.surfaceVertices.size() : sortedVertices.size();
	const PxU32 clampedVertexLoopBegin =
		PxMin(vertexLoopBegin, vertexLoopCount);
	const PxU32 clampedVertexLoopEnd =
		PxMin(PxMax(vertexLoopEnd, clampedVertexLoopBegin),
			vertexLoopCount);
	for(PxU32 vertexLoopIndex = clampedVertexLoopBegin;
		vertexLoopIndex < clampedVertexLoopEnd;
		vertexLoopIndex++)
	{
		const PxU32 gi = useSurfaceTriangleBvh
			? sb.compiled.surfaceVertices[vertexLoopIndex]
			: sb.compiled.particleStart +
				sortedVertices[vertexLoopIndex].localIndex;
		if(gi < sb.compiled.particleStart ||
			gi - sb.compiled.particleStart >=
				sb.compiled.particleCount)
			continue;
		const PxU32 li = gi - sb.compiled.particleStart;
		const PxVec3& pp = particles[gi].position;
		const PxVec3 vertexMinimum = sweepEnabled
			? particles[gi].initialPosition.minimum(pp) : pp;
		const PxVec3 vertexMaximum = sweepEnabled
			? particles[gi].initialPosition.maximum(pp) : pp;
		if(useSurfaceTriangleBvh)
		{
			sb.compiled.collectSurfaceTriangleBvhCandidates(
				vertexMinimum, vertexMaximum, r,
				triangleBvhBounds, triangleCandidates);
			if(stats)
				stats->surfaceTriangleBvhCandidateTriangles +=
					triangleCandidates.size();
		}
		if(!useSurfaceTriangleBvh)
		{
		const PxReal vertexMinimumX =
			sortedVertices[vertexLoopIndex].minimumX;
		const PxReal vertexMaximumX =
			sortedVertices[vertexLoopIndex].maximumX;
		while(triangleCursor < triangleBounds.size() &&
			triangleBounds[triangleCursor].minimum.x <=
			vertexMaximumX + r)
			activeTriangles.pushBack(triangleCursor++);
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size();)
		{
			const AvbdSelfCollisionTriangleBounds& triangle =
				triangleBounds[activeTriangles[activeIndex]];
			if(triangle.maximum.x < vertexMinimumX - r)
			{
				activeTriangles[activeIndex] =
					activeTriangles.back();
				activeTriangles.popBack();
				continue;
			}
			activeIndex++;
		}
		}
		const PxU32 activeTriangleCount = useSurfaceTriangleBvh
			? triangleCandidates.size() : activeTriangles.size();

		// Select a single first face crossing for this vertex.  This prevents
		// adjacent triangles from compiling several speculative rows for one
		// physical crossing and keeps every outer redetection deterministic.
		if(sweepEnabled)
		{
			PxReal bestEntryTime = PX_MAX_F32;
			PxU32 bestTriangleOffset = PX_MAX_U32;
			AvbdSweptTriangleEntry bestEntry;
			for(PxU32 activeIndex = 0;
				activeIndex < activeTriangleCount;
				activeIndex++)
			{
				const PxU32 ti = useSurfaceTriangleBvh
					? triangleCandidates[activeIndex] * 3
					: triangleBounds[
						activeTriangles[activeIndex]].triangleOffset;
				if(useSurfaceTriangleBvh)
				{
					if(!triangleOverlapsQuery(
						ti, vertexMinimum, vertexMaximum))
						continue;
				}
				else
				{
					const AvbdSelfCollisionTriangleBounds& triangle =
						triangleBounds[activeTriangles[activeIndex]];
					if(triangle.minimum.y > vertexMaximum.y + r ||
						triangle.maximum.y < vertexMinimum.y - r ||
						triangle.minimum.z > vertexMaximum.z + r ||
						triangle.maximum.z < vertexMinimum.z - r)
						continue;
				}
				if(stats)
					stats->selfTriangleTests++;
				const PxU32 source0 =
					sb.compiled.surfaceTriangles[ti];
				const PxU32 source1 =
					sb.compiled.surfaceTriangles[ti + 1];
				const PxU32 source2 =
					sb.compiled.surfaceTriangles[ti + 2];
				const PxU32 lv0 =
					source0 - sb.compiled.particleStart;
				const PxU32 lv1 =
					source1 - sb.compiled.particleStart;
				const PxU32 lv2 =
					source2 - sb.compiled.particleStart;
				if(lv0 == li || lv1 == li || lv2 == li)
					continue;
				if(particles[gi].invMass <= 0.0f &&
					particles[source0].invMass <= 0.0f &&
					particles[source1].invMass <= 0.0f &&
					particles[source2].invMass <= 0.0f)
					continue;
				if(!targetStressAllowsTriangle(ti))
					continue;
				if(hasRestFilter)
				{
					if(sb.compiled.selfCollisionRestFilterCacheValid)
					{
						if(avbdIsSelfRestVertexTriangleFiltered(
							sb, li, ti / 3))
							continue;
					}
					else
					{
						const PxArray<PxVec3>& restPositions =
							sb.compiled.selfCollisionRestPositions;
						const AvbdClosestPointResult restClosest =
							avbdClosestPointOnTriangleOGC(
								restPositions[li],
								restPositions[lv0],
								restPositions[lv1],
								restPositions[lv2]);
						if(restClosest.distance <= filterDistance)
							continue;
					}
				}
				AvbdSweptTriangleEntry entry;
				if(avbdRotatingPointEnterExpandedDeformingTriangleFace(
						PxVec3(0.0f),
						particles[gi].initialPosition, pp,
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
					sb.compiled.surfaceTriangles[
						bestTriangleOffset];
				const PxU32 source1 =
					sb.compiled.surfaceTriangles[
						bestTriangleOffset + 1];
				const PxU32 source2 =
					sb.compiled.surfaceTriangles[
						bestTriangleOffset + 2];
				PxVec3 contactNormal = -bestEntry.normal;
				const AvbdClosestPointResult initialClosest =
					avbdClosestPointOnTriangleOGC(
						particles[gi].initialPosition,
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
					AvbdSoftContactSource::eSELF_SURFACE,
					softBodyIdx,
					avbdSoftTrianglePrimitiveKey(
						source0, source1, source2),
					avbdSoftTriangleFeatureKey(
						source0, source1, source2,
						AVBD_FEATURE_FACE, 0));
				geometry.particleIdx = gi;
				geometry.targetKind =
					AvbdSoftContactTargetKind::
						eDEFORMABLE_SURFACE;
				geometry.velocityOwner =
					AvbdVelocityObjectiveOwner::PositionAL;
				geometry.targetIndex = softBodyIdx;
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
				geometry.friction =
					PxMax(sb.material.dynamicFriction, 0.0f);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry,
					params.contactStiffness,
					params.contactStiffness * 10.0f,
					particles, contacts);
				continue;
			}
		}
		emittedFeatureKeys.clear();
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangleCount;
			activeIndex++)
		{
			const PxU32 ti = useSurfaceTriangleBvh
				? triangleCandidates[activeIndex] * 3
				: triangleBounds[
					activeTriangles[activeIndex]].triangleOffset;
			if(useSurfaceTriangleBvh)
			{
				if(!triangleOverlapsQuery(ti, pp, pp))
					continue;
			}
			else
			{
				const AvbdSelfCollisionTriangleBounds& triangle =
					triangleBounds[activeTriangles[activeIndex]];
				if(triangle.minimum.y > pp.y + r ||
					triangle.maximum.y < pp.y - r ||
					triangle.minimum.z > pp.z + r ||
					triangle.maximum.z < pp.z - r)
					continue;
			}
			if(stats)
				stats->selfTriangleTests++;

			const PxU32 source0 =
				sb.compiled.surfaceTriangles[ti];
			const PxU32 source1 =
				sb.compiled.surfaceTriangles[ti + 1];
			const PxU32 source2 =
				sb.compiled.surfaceTriangles[ti + 2];
			const PxU32 lv0 =
				source0 - sb.compiled.particleStart;
			const PxU32 lv1 =
				source1 - sb.compiled.particleStart;
			const PxU32 lv2 =
				source2 - sb.compiled.particleStart;

			// The OGC conservative proof excludes incident facets.  A
			// caller-requested rest-distance filter handles any wider
			// topological exclusion explicitly.
			if(lv0 == li || lv1 == li || lv2 == li)
				continue;
			if(particles[gi].invMass <= 0.0f &&
				particles[source0].invMass <= 0.0f &&
				particles[source1].invMass <= 0.0f &&
				particles[source2].invMass <= 0.0f)
				continue;
			if(!targetStressAllowsTriangle(ti))
				continue;
			if(hasRestFilter)
			{
				if(sb.compiled.selfCollisionRestFilterCacheValid)
				{
					if(avbdIsSelfRestVertexTriangleFiltered(
						sb, li, ti / 3))
						continue;
				}
				else
				{
					const PxArray<PxVec3>& restPositions =
						sb.compiled.selfCollisionRestPositions;
					const AvbdClosestPointResult restClosest =
						avbdClosestPointOnTriangleOGC(
							restPositions[li],
							restPositions[lv0],
							restPositions[lv1],
							restPositions[lv2]);
					if(restClosest.distance <= filterDistance)
						continue;
				}
			}

			const PxVec3& va = particles[source0].position;
			const PxVec3& vb = particles[source1].position;
			const PxVec3& vc = particles[source2].position;
			const AvbdClosestPointResult cp =
				avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
			if(cp.distance >= r)
				continue;

			PxVec3 faceNormal = (vb - va).cross(vc - va);
			const PxReal faceNormalLength =
				faceNormal.magnitude();
			if(faceNormalLength < 1.0e-10f)
				continue;
			faceNormal *= 1.0f / faceNormalLength;
			PxVec3 contactNormal =
				cp.feature == AVBD_FEATURE_FACE
					? (cp.normal.dot(faceNormal) >= 0.0f
						? faceNormal : -faceNormal)
					: cp.normal;

			// Keep the normal on the penetration-free side recorded at the
			// beginning of this time step.  Choosing the current nearest side
			// after a crossing makes the normal flip every redetection and is
			// the source of the observed self-twitching.
			const AvbdClosestPointResult previousClosest =
				avbdClosestPointOnTriangleOGC(
					particles[gi].initialPosition,
					particles[source0].initialPosition,
					particles[source1].initialPosition,
					particles[source2].initialPosition);
			PxVec3 previousNormal = previousClosest.normal;
			if(previousNormal.magnitudeSquared() <= 1.0e-16f)
			{
				previousNormal =
					(particles[source1].initialPosition -
					 particles[source0].initialPosition).cross(
						particles[source2].initialPosition -
						particles[source0].initialPosition);
				if(previousNormal.magnitudeSquared() <= 1.0e-16f)
					continue;
				previousNormal.normalize();
			}
			if(contactNormal.dot(previousNormal) < 0.0f)
				contactNormal = -contactNormal;

			const AvbdActivationResult activation =
				avbdOGCActivationFull(
					cp.distance, r,
					params.contactStiffness, tau);
			if(activation.force <= 0.0f)
				continue;
			const PxU64 featureKey =
				avbdSoftTriangleFeatureKey(
					source0, source1, source2,
					cp.feature, cp.featureIndex);
			bool duplicateFeature = false;
			for(PxU32 emittedIndex = 0;
				emittedIndex < emittedFeatureKeys.size();
				emittedIndex++)
			{
				if(emittedFeatureKeys[emittedIndex] ==
					featureKey)
				{
					duplicateFeature = true;
					break;
				}
			}
			if(duplicateFeature)
				continue;
			emittedFeatureKeys.pushBack(featureKey);

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eSELF_SURFACE,
				softBodyIdx,
				cp.feature == AVBD_FEATURE_FACE
					? avbdSoftTrianglePrimitiveKey(
						source0, source1, source2)
					: featureKey,
				featureKey);
			geometry.particleIdx = gi;
			geometry.targetKind =
				AvbdSoftContactTargetKind::
					eDEFORMABLE_SURFACE;
			geometry.velocityOwner =
				AvbdVelocityObjectiveOwner::PositionAL;
			geometry.targetIndex = softBodyIdx;
			geometry.normal = contactNormal;
			geometry.projNormal = contactNormal;
			geometry.depth = r - cp.distance;
			geometry.margin = r;
			geometry.surfacePoint = cp.point;
			geometry.surfaceParticleIndices[0] = source0;
			geometry.surfaceParticleIndices[1] = source1;
			geometry.surfaceParticleIndices[2] = source2;
			geometry.surfaceWeights[0] = cp.barycentric.x;
			geometry.surfaceWeights[1] = cp.barycentric.y;
			geometry.surfaceWeights[2] = cp.barycentric.z;
			geometry.friction =
				PxMax(sb.material.dynamicFriction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry,
				params.contactStiffness,
				params.contactStiffness * 10.0f,
				particles, contacts);
		}
	}

	// Vertex-face alone does not preserve the topology of two crossing
	// triangle interiors: both endpoints of a cloth edge can remain outside
	// the opposing triangles while the edges pass through one another.
	// Complete the self-collision feature set with one barycentric edge-edge
	// objective for every non-adjacent edge pair in the contact shell.
	const bool useSurfaceEdgeBvh = usePreparedBvhWorkspace ||
		(avbdUseSurfaceEdgeBvh() &&
		 !sb.compiled.surfaceEdgeBvhNodes.empty());
	PX_ASSERT(!usePreparedBvhWorkspace ||
		(sb.compiled.surfaceEdges.empty() ||
		 !sb.compiled.surfaceEdgeBvhNodes.empty()));
	if(useSurfaceEdgeBvh && !usePreparedBvhWorkspace)
	{
		sb.compiled.refitSurfaceEdgeBvh(
			particles, sweepEnabled, workspace.selfEdgeBvhBounds);
		if(stats)
			stats->surfaceEdgeBvhRefitNodes +=
				sb.compiled.surfaceEdgeBvhNodes.size();
	}
	const PxArray<AvbdSurfaceBvhNodeBounds>& edgeBvhBounds =
		usePreparedBvhWorkspace
			? preparedBvhWorkspace->selfEdgeBvhBounds
			: workspace.selfEdgeBvhBounds;
	PX_ASSERT(!usePreparedBvhWorkspace ||
		edgeBvhBounds.size() == sb.compiled.surfaceEdgeBvhNodes.size());
	auto getEdgeBounds =
		[&](PxU32 edgeIndex, PxVec3& minimum, PxVec3& maximum)
	{
		const AvbdEdgeInfo& edge = sb.compiled.surfaceEdges[edgeIndex];
		minimum = particles[edge.p0].position.minimum(
			particles[edge.p1].position);
		maximum = particles[edge.p0].position.maximum(
			particles[edge.p1].position);
		if(sweepEnabled)
		{
			minimum = minimum.minimum(particles[edge.p0].initialPosition).
				minimum(particles[edge.p1].initialPosition);
			maximum = maximum.maximum(particles[edge.p0].initialPosition).
				maximum(particles[edge.p1].initialPosition);
		}
	};
	PxArray<AvbdSelfCollisionEdgeBounds>& edgeBounds =
		workspace.selfEdgeBounds;
	edgeBounds.clear();
	if(!useSurfaceEdgeBvh)
	{
		for(PxU32 edgeIndex = 0;
			edgeIndex < sb.compiled.surfaceEdges.size(); edgeIndex++)
		{
			const AvbdEdgeInfo& edge =
				sb.compiled.surfaceEdges[edgeIndex];
			if(edge.p0 >= sb.compiled.particleStart +
					sb.compiled.particleCount ||
				edge.p1 >= sb.compiled.particleStart +
					sb.compiled.particleCount)
				continue;
			AvbdSelfCollisionEdgeBounds bounds;
			bounds.edgeIndex = edgeIndex;
			getEdgeBounds(edgeIndex, bounds.minimum, bounds.maximum);
			edgeBounds.pushBack(bounds);
			if(stats)
				stats->selfEdgeBoundsBuilt++;
		}
	}
	if(!useSurfaceEdgeBvh)
		PxSort(
			edgeBounds.begin(), edgeBounds.size(),
			[](const AvbdSelfCollisionEdgeBounds& a,
			   const AvbdSelfCollisionEdgeBounds& b)
			{
				return a.minimum.x < b.minimum.x;
			});

	const PxReal edgeFeatureEpsilon = 1.0e-4f;
	const PxReal edgeDistanceEpsilon = 1.0e-8f;
	auto targetStressAllowsEdge =
		[&](PxU32 edge0, PxU32 edge1) -> bool
	{
		if(tetStressCoefficients->empty())
			return true;
		for(PxU32 triangleOffset = 0;
			triangleOffset + 2 <
				sb.compiled.surfaceTriangles.size();
			triangleOffset += 3)
		{
			const PxU32 v0 =
				sb.compiled.surfaceTriangles[triangleOffset];
			const PxU32 v1 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 1];
			const PxU32 v2 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 2];
			const bool has0 =
				v0 == edge0 || v1 == edge0 || v2 == edge0;
			const bool has1 =
				v0 == edge1 || v1 == edge1 || v2 == edge1;
			if(has0 && has1 &&
				!targetStressAllowsTriangle(triangleOffset))
				return false;
		}
		return true;
	};
	PxArray<PxU32>& edgeCandidates = workspace.selfEdgeCandidates;
	edgeCandidates.clear();
	const PxU32 outerEdgeCount = useSurfaceEdgeBvh
		? sb.compiled.surfaceEdges.size() : edgeBounds.size();
	const PxU32 clampedEdgeLoopBegin =
		PxMin(edgeLoopBegin, outerEdgeCount);
	const PxU32 clampedEdgeLoopEnd =
		PxMin(PxMax(edgeLoopEnd, clampedEdgeLoopBegin), outerEdgeCount);
	for(PxU32 outerEdgeIndex = clampedEdgeLoopBegin;
		outerEdgeIndex < clampedEdgeLoopEnd; outerEdgeIndex++)
	{
		const PxU32 sourceEdgeIndex = useSurfaceEdgeBvh
			? outerEdgeIndex : edgeBounds[outerEdgeIndex].edgeIndex;
		if(sourceEdgeIndex >= sb.compiled.surfaceEdges.size())
			continue;
		PxVec3 sourceMinimum, sourceMaximum;
		if(useSurfaceEdgeBvh)
		{
			getEdgeBounds(sourceEdgeIndex, sourceMinimum, sourceMaximum);
			sb.compiled.collectSurfaceEdgeBvhCandidates(
				sourceMinimum, sourceMaximum, r,
				edgeBvhBounds, edgeCandidates);
			if(stats)
				stats->surfaceEdgeBvhCandidateEdges += edgeCandidates.size();
		}
		else
		{
			sourceMinimum = edgeBounds[outerEdgeIndex].minimum;
			sourceMaximum = edgeBounds[outerEdgeIndex].maximum;
		}
		const PxU32 innerFirst = useSurfaceEdgeBvh
			? 0u : outerEdgeIndex + 1;
		const PxU32 innerEdgeCount = useSurfaceEdgeBvh
			? edgeCandidates.size() : edgeBounds.size();
		for(PxU32 innerEdgeIndex = innerFirst;
			innerEdgeIndex < innerEdgeCount; innerEdgeIndex++)
		{
			const PxU32 candidateEdgeIndex = useSurfaceEdgeBvh
				? edgeCandidates[innerEdgeIndex]
				: edgeBounds[innerEdgeIndex].edgeIndex;
			if(candidateEdgeIndex >= sb.compiled.surfaceEdges.size())
				continue;
			PxVec3 targetMinimum, targetMaximum;
			if(useSurfaceEdgeBvh)
			{
				if(candidateEdgeIndex <= sourceEdgeIndex)
					continue;
				getEdgeBounds(
					candidateEdgeIndex, targetMinimum, targetMaximum);
			}
			else
			{
				targetMinimum = edgeBounds[innerEdgeIndex].minimum;
				targetMaximum = edgeBounds[innerEdgeIndex].maximum;
				if(targetMinimum.x > sourceMaximum.x + r)
					break;
			}
			if(sourceMinimum.y > targetMaximum.y + r ||
				sourceMaximum.y < targetMinimum.y - r ||
				sourceMinimum.z > targetMaximum.z + r ||
				sourceMaximum.z < targetMinimum.z - r ||
				(sourceMinimum.x > targetMaximum.x + r ||
				 sourceMaximum.x < targetMinimum.x - r))
				continue;

			const PxU32 queryEdgeIndex =
				PxMin(sourceEdgeIndex, candidateEdgeIndex);
			const PxU32 targetEdgeIndex =
				PxMax(sourceEdgeIndex, candidateEdgeIndex);
			const AvbdEdgeInfo& queryEdge =
				sb.compiled.surfaceEdges[queryEdgeIndex];
			const AvbdEdgeInfo& targetEdge =
				sb.compiled.surfaceEdges[targetEdgeIndex];
			const PxU32 q0 = queryEdge.p0;
			const PxU32 q1 = queryEdge.p1;
			const PxU32 t0 = targetEdge.p0;
			const PxU32 t1 = targetEdge.p1;
			if(q0 == t0 || q0 == t1 ||
				q1 == t0 || q1 == t1)
				continue;

			const PxU32 lq0 = q0 - sb.compiled.particleStart;
			const PxU32 lq1 = q1 - sb.compiled.particleStart;
			const PxU32 lt0 = t0 - sb.compiled.particleStart;
			const PxU32 lt1 = t1 - sb.compiled.particleStart;
			if(particles[q0].invMass <= 0.0f &&
				particles[q1].invMass <= 0.0f &&
				particles[t0].invMass <= 0.0f &&
				particles[t1].invMass <= 0.0f)
				continue;
			if(!targetStressAllowsEdge(t0, t1))
				continue;

			if(hasRestFilter)
			{
				const PxArray<PxVec3>& restPositions =
					sb.compiled.selfCollisionRestPositions;
				PxReal restQueryWeight = 0.0f;
				PxReal restTargetWeight = 0.0f;
				PxVec3 restQueryClosest, restTargetClosest;
				avbdClosestPointsOnSegments(
					restPositions[lq0], restPositions[lq1],
					restPositions[lt0], restPositions[lt1],
					restQueryWeight, restTargetWeight,
					restQueryClosest, restTargetClosest);
				if((restQueryClosest - restTargetClosest).
						magnitude() <= filterDistance)
					continue;
			}

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
				previousQueryClosest - previousTargetClosest;

			auto stabilizeEdgeNormal =
				[&](PxVec3 contactNormal) -> PxVec3
			{
				if(previousDelta.magnitudeSquared() >
					edgeDistanceEpsilon *
						edgeDistanceEpsilon)
				{
					if(contactNormal.dot(previousDelta) < 0.0f)
						contactNormal = -contactNormal;
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
						contactNormal.dot(previousCross) < 0.0f)
						contactNormal = -contactNormal;
				}
				return contactNormal;
			};

			auto appendEdgeContact =
				[&](PxReal queryWeight1,
					PxReal targetWeight1,
					const PxVec3& contactNormal,
					PxReal depth,
					const PxVec3& targetClosest)
			{
				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eSELF_SURFACE,
					softBodyIdx,
					avbdGetRigidSoftFeatureKey(
						0x53454530u, q0, q1, 0u, 0u),
					avbdGetRigidSoftFeatureKey(
						0x53454531u, q0, q1, t0, t1));
				geometry.particleIdx =
					particles[q0].invMass > 0.0f ? q0 :
					(particles[q1].invMass > 0.0f ? q1 :
					(particles[t0].invMass > 0.0f ? t0 : t1));
				geometry.queryParticleIndices[0] = q0;
				geometry.queryParticleIndices[1] = q1;
				geometry.queryWeights[0] = 1.0f - queryWeight1;
				geometry.queryWeights[1] = queryWeight1;
				geometry.targetKind =
					AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE;
				geometry.velocityOwner =
					AvbdVelocityObjectiveOwner::PositionAL;
				geometry.targetIndex = softBodyIdx;
				geometry.surfaceParticleIndices[0] = t0;
				geometry.surfaceParticleIndices[1] = t1;
				geometry.surfaceWeights[0] = 1.0f - targetWeight1;
				geometry.surfaceWeights[1] = targetWeight1;
				geometry.normal = contactNormal;
				geometry.projNormal = contactNormal;
				geometry.depth = depth;
				geometry.margin = r;
				geometry.surfacePoint = targetClosest;
				geometry.friction =
					PxMax(sb.material.dynamicFriction, 0.0f);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry,
					params.contactStiffness,
					params.contactStiffness * 10.0f,
					particles, contacts);
			};

			if(sweepEnabled)
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
					const PxVec3 contactNormal =
						stabilizeEdgeNormal(entry.normal);
					appendEdgeContact(
						entry.softWeight1,
						entry.rigidWeight1,
						contactNormal,
						0.0f,
						target0AtEntry *
							(1.0f - entry.rigidWeight1) +
						target1AtEntry * entry.rigidWeight1);
					continue;
				}
			}

			PxReal queryWeight1 = 0.0f;
			PxReal targetWeight1 = 0.0f;
			PxVec3 queryClosest, targetClosest;
			avbdClosestPointsOnSegments(
				particles[q0].position,
				particles[q1].position,
				particles[t0].position,
				particles[t1].position,
				queryWeight1, targetWeight1,
				queryClosest, targetClosest);
			if(queryWeight1 <= edgeFeatureEpsilon ||
				queryWeight1 >= 1.0f - edgeFeatureEpsilon ||
				targetWeight1 <= edgeFeatureEpsilon ||
				targetWeight1 >= 1.0f - edgeFeatureEpsilon)
				continue;
			const PxVec3 delta = queryClosest - targetClosest;
			const PxReal distance = delta.magnitude();
			if(distance >= r)
				continue;

			PxVec3 contactNormal;
			if(distance > edgeDistanceEpsilon)
				contactNormal = delta * (1.0f / distance);
			else
			{
				contactNormal =
					(particles[q1].position -
					 particles[q0].position).cross(
						particles[t1].position -
						particles[t0].position);
				if(contactNormal.magnitudeSquared() <=
					edgeDistanceEpsilon *
						edgeDistanceEpsilon)
					continue;
				contactNormal.normalize();
			}
			contactNormal = stabilizeEdgeNormal(contactNormal);
			const AvbdActivationResult activation =
				avbdOGCActivationFull(
					distance, r,
					params.contactStiffness, tau);
			if(activation.force <= 0.0f)
				continue;
			appendEdgeContact(
				queryWeight1, targetWeight1,
				contactNormal, r - distance,
				targetClosest);
		}
	}
}

bool avbdCanUseSelfCollisionOGCBvhRanges(
	const AvbdSoftBody& softBody)
{
	return avbdUseSurfaceTriangleBvh() &&
		!softBody.compiled.surfaceTriangleBvhNodes.empty() &&
		(softBody.compiled.surfaceEdges.empty() ||
			(avbdUseSurfaceEdgeBvh() &&
				!softBody.compiled.surfaceEdgeBvhNodes.empty()));
}

bool avbdPrepareSelfCollisionOGCBvhRanges(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& softBody,
	PxU32 softBodyIndex,
	const PxArray<PxArray<PxU32> >& adjacency,
	const AvbdOGCParams& params,
	AvbdSoftContactWorkspace& parentWorkspace,
	AvbdSoftCollisionStats* stats)
{
	if(!avbdCanUseSelfCollisionOGCBvhRanges(softBody))
		return false;
	PxArray<AvbdSoftContact> noContacts;
	avbdDetectSelfCollisionOGC(
		particles, softBody, softBodyIndex, adjacency, noContacts,
		params, stats, &parentWorkspace, NULL, 0, 0, 0, 0);
	return true;
}

void avbdDetectSelfCollisionOGCBvhRange(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& softBody,
	PxU32 softBodyIndex,
	const PxArray<PxArray<PxU32> >& adjacency,
	const AvbdSoftContactWorkspace& parentWorkspace,
	AvbdSoftContactWorkspace& rangeWorkspace,
	PxU32 vertexLoopBegin,
	PxU32 vertexLoopEnd,
	PxU32 edgeLoopBegin,
	PxU32 edgeLoopEnd,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats)
{
	PX_ASSERT(&parentWorkspace != &rangeWorkspace);
	PX_ASSERT(avbdCanUseSelfCollisionOGCBvhRanges(softBody));
	avbdDetectSelfCollisionOGC(
		particles, softBody, softBodyIndex, adjacency, contacts,
		params, stats, &rangeWorkspace, &parentWorkspace,
		vertexLoopBegin, vertexLoopEnd, edgeLoopBegin, edgeLoopEnd);
}

} // namespace Dy
} // namespace physx
