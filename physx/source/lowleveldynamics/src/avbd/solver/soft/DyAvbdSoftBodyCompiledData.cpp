/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of NVIDIA CORPORATION nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

AvbdSoftBodyCompiledData::AvbdSoftBodyCompiledData()
	: particleStart(0), particleCount(0),
	  tetIncidenceFullPacketCount(0),
	  tetIncidencePacketProgramValid(false),
	  particlePrimalStructuralConflictValid(false),
	  flatteningEnabled(false),
	  selfCollisionFilterDistance(0.0f),
	  selfCollisionRestFilterCacheDistance(-PX_MAX_F32),
	  selfCollisionRestFilterCacheValid(false),
	  selfCollisionRestFilterCacheFallback(false),
	  maxDepenetrationVelocity(PX_MAX_F32),
	  selfCollisionStressTolerance(0.9f),
	  speculativeCCDEnabled(false)
{
}

void AvbdSoftBodyCompiledData::compileBendingRestAngles(bool enabled)
{
	if(flatteningEnabled == enabled)
		return;
	flatteningEnabled = enabled;
	for(PxU32 i = 0; i < bendElements.size(); i++)
		bendElements[i].restAngle =
			enabled ? 0.0f : bendElements[i].restShapeAngle;
}

PxReal AvbdSoftBodyCompiledData::computeDihedralAngle(const PxVec3& x0, const PxVec3& x1,
                                   const PxVec3& x2, const PxVec3& x3)
{
	const PxReal eps = 1e-8f;
	PxVec3 e = x3 - x2;
	PxVec3 n1 = (x2 - x0).cross(x3 - x0);
	PxVec3 n2 = (x3 - x1).cross(x2 - x1);
	PxReal n1Norm = n1.magnitude();
	PxReal n2Norm = n2.magnitude();
	PxReal eNorm = e.magnitude();
	if (n1Norm < eps || n2Norm < eps || eNorm < eps)
		return 0.0f;
	PxVec3 n1Hat = n1 * (1.0f / n1Norm);
	PxVec3 n2Hat = n2 * (1.0f / n2Norm);
	PxVec3 eHat = e * (1.0f / eNorm);
	PxReal sinTheta = n1Hat.cross(n2Hat).dot(eHat);
	PxReal cosTheta = PxClamp(n1Hat.dot(n2Hat), -1.0f, 1.0f);
	return PxAtan2(sinTheta, cosTheta);
}

void AvbdSoftBodyCompiledData::buildTriElements(const PxArray<AvbdSoftParticle>& particles)
{
	triElements.clear();
	for (PxU32 i = 0; i + 2 < triangles.size(); i += 3)
	{
		PxU32 i0 = triangles[i] + particleStart;
		PxU32 i1 = triangles[i + 1] + particleStart;
		PxU32 i2 = triangles[i + 2] + particleStart;

		PxVec3 x0 = particles[i0].position;
		PxVec3 e01 = particles[i1].position - x0;
		PxVec3 e02 = particles[i2].position - x0;

		PxVec3 t1 = e01.getNormalized();
		PxVec3 n = e01.cross(e02);
		PxReal area = n.magnitude() * 0.5f;
		if (area < 1e-12f) continue;
		PxVec3 t2 = n.cross(t1).getNormalized();

		PxReal d00 = e01.dot(t1), d10 = e01.dot(t2);
		PxReal d01 = e02.dot(t1), d11 = e02.dot(t2);
		PxReal det = d00 * d11 - d01 * d10;
		if (PxAbs(det) < 1e-12f) continue;
		PxReal invDet = 1.0f / det;

		AvbdTriElement tri;
		tri.p0 = i0; tri.p1 = i1; tri.p2 = i2;
		tri.sourceElementIndex = i / 3;
		tri.DmInv00 =  d11 * invDet;
		tri.DmInv01 = -d01 * invDet;
		tri.DmInv10 = -d10 * invDet;
		tri.DmInv11 =  d00 * invDet;
		tri.restArea = area;
		triElements.pushBack(tri);
	}
}

void AvbdSoftBodyCompiledData::buildTetElements(const PxArray<AvbdSoftParticle>& particles)
{
	tetElements.clear();
	for (PxU32 i = 0; i + 3 < tetrahedra.size(); i += 4)
	{
		PxU32 i0 = tetrahedra[i] + particleStart;
		PxU32 i1 = tetrahedra[i + 1] + particleStart;
		PxU32 i2 = tetrahedra[i + 2] + particleStart;
		PxU32 i3 = tetrahedra[i + 3] + particleStart;

		PxVec3 x0 = particles[i0].position;
		PxVec3 e1 = particles[i1].position - x0;
		PxVec3 e2 = particles[i2].position - x0;
		PxVec3 e3 = particles[i3].position - x0;

		PxMat33 Dm(e1, e2, e3);
		PxReal det = Dm.getDeterminant();
		PxReal vol = PxAbs(det) / 6.0f;
		if (vol < 1e-15f) continue;

		AvbdTetElement tet;
		tet.p0 = i0; tet.p1 = i1; tet.p2 = i2; tet.p3 = i3;
		tet.sourceElementIndex = i / 4;
		tet.DmInv = Dm.getInverse();
		tet.restVolume = vol;
		const PxMat33& inverseRestShape = tet.DmInv;
		const PxVec3 shapeGradients[4] =
		{
			PxVec3(
				-avbdColSum(inverseRestShape.column0),
				-avbdColSum(inverseRestShape.column1),
				-avbdColSum(inverseRestShape.column2)),
			avbdMatRow(inverseRestShape, 0),
			avbdMatRow(inverseRestShape, 1),
			avbdMatRow(inverseRestShape, 2)
		};
		for(PxU32 vertexOrder = 0; vertexOrder < 4; vertexOrder++)
		{
			tet.shapeGradients[vertexOrder] =
				shapeGradients[vertexOrder];
			tet.deformationGradientWeights[vertexOrder] =
				inverseRestShape * shapeGradients[vertexOrder];
			tet.shapeGradientNormSq[vertexOrder] =
				shapeGradients[vertexOrder].magnitudeSquared();
		}
		tet.inverseRestDeterminant =
			inverseRestShape.getDeterminant();
		tetElements.pushBack(tet);
	}
}

void AvbdSoftBodyCompiledData::buildBendingElements(const PxArray<AvbdSoftParticle>& particles)
{
	bendElements.clear();
	if (triangles.empty()) return;

	// Sort-and-scan approach: collect all half-edge records, sort by
	// canonical edge key, then scan for adjacent pairs sharing the same key.
	// This is O(e log e) with contiguous memory access -- friendly to SIMD
	// sort and parallel scan in future optimisations.

	struct HalfEdge
	{
		PxU64 key;       // canonical (lo << 32 | hi)
		PxU32 edgeV0;    // original (unsorted) first vertex
		PxU32 edgeV1;    // original (unsorted) second vertex
		PxU32 oppVertex;
	};

	const PxU32 numTris = triangles.size() / 3;
	PxArray<HalfEdge> halfEdges;
	halfEdges.reserve(numTris * 3);

	for (PxU32 ti = 0; ti < numTris; ti++)
	{
		PxU32 v[3] = {
			triangles[ti * 3]     + particleStart,
			triangles[ti * 3 + 1] + particleStart,
			triangles[ti * 3 + 2] + particleStart
		};
		for (int e = 0; e < 3; e++)
		{
			PxU32 ea = v[e], eb = v[(e + 1) % 3], opp = v[(e + 2) % 3];
			HalfEdge he;
			he.key = (PxU64(ea < eb ? ea : eb) << 32) | PxU64(ea < eb ? eb : ea);
			he.edgeV0 = ea;
			he.edgeV1 = eb;
			he.oppVertex = opp;
			halfEdges.pushBack(he);
		}
	}

	// Sort by key (PxSort for now, replaceable with parallel radix sort later)
	PxSort(halfEdges.begin(), halfEdges.size(),
		[](const HalfEdge& a, const HalfEdge& b) { return a.key < b.key; });

	// Scan: match consecutive pairs with same key (manifold edge = exactly 2)
	for (PxU32 i = 0; i + 1 < halfEdges.size(); i++)
	{
		if (halfEdges[i].key != halfEdges[i + 1].key)
			continue;

		// Count how many share this key (skip non-manifold > 2)
		PxU32 run = 1;
		while (i + run < halfEdges.size() && halfEdges[i + run].key == halfEdges[i].key)
			run++;
		if (run == 2)
		{
			PxU32 edgeA = PxU32(halfEdges[i].key >> 32);
			PxU32 edgeB = PxU32(halfEdges[i].key & 0xFFFFFFFF);

			AvbdBendingElement be;
			be.opp0 = halfEdges[i].oppVertex;
			be.opp1 = halfEdges[i + 1].oppVertex;
			be.edgeStart = edgeA;
			be.edgeEnd = edgeB;
			be.restShapeAngle = computeDihedralAngle(
				particles[be.opp0].position, particles[be.opp1].position,
				particles[edgeA].position, particles[edgeB].position);
			be.restAngle = be.restShapeAngle;
			be.restLength = (particles[edgeB].position - particles[edgeA].position).magnitude();
			bendElements.pushBack(be);
		}
		i += run - 1; // advance past run
	}
}

void AvbdSoftBodyCompiledData::buildEdges(const PxArray<AvbdSoftParticle>& particles)
{
	edges.clear();

	// Collect all candidate edges as canonical (lo, hi) keys into a flat
	// array, sort, then deduplicate with a linear scan.

	PxArray<PxU64> keys;
	const PxU32 numTris = triangles.size() / 3;
	const PxU32 numTets = tetrahedra.size() / 4;
	keys.reserve(numTris * 3 + numTets * 6);

	auto pushEdge = [&](PxU32 a, PxU32 b) {
		PxU32 lo = a < b ? a : b;
		PxU32 hi = a < b ? b : a;
		keys.pushBack((PxU64(lo) << 32) | PxU64(hi));
	};

	for (PxU32 i = 0; i < numTris; i++)
	{
		PxU32 v[3] = {
			triangles[i * 3]     + particleStart,
			triangles[i * 3 + 1] + particleStart,
			triangles[i * 3 + 2] + particleStart
		};
		pushEdge(v[0], v[1]); pushEdge(v[1], v[2]); pushEdge(v[2], v[0]);
	}
	for (PxU32 i = 0; i < numTets; i++)
	{
		PxU32 v[4] = {
			tetrahedra[i * 4]     + particleStart,
			tetrahedra[i * 4 + 1] + particleStart,
			tetrahedra[i * 4 + 2] + particleStart,
			tetrahedra[i * 4 + 3] + particleStart
		};
		pushEdge(v[0], v[1]); pushEdge(v[0], v[2]); pushEdge(v[0], v[3]);
		pushEdge(v[1], v[2]); pushEdge(v[1], v[3]); pushEdge(v[2], v[3]);
	}

	PxSort(keys.begin(), keys.size());

	// Linear unique scan
	for (PxU32 i = 0; i < keys.size(); )
	{
		PxU64 k = keys[i];
		PxU32 a = PxU32(k >> 32);
		PxU32 b = PxU32(k & 0xFFFFFFFF);
		AvbdEdgeInfo ei;
		ei.p0 = a; ei.p1 = b;
		ei.restLength = (particles[a].position - particles[b].position).magnitude();
		edges.pushBack(ei);
		// Skip duplicates
		while (i < keys.size() && keys[i] == k) i++;
	}
}

void AvbdSoftBodyCompiledData::invalidateTetIncidencePacketProgram()
{
	tetIncidencePackets.clear();
	tetIncidencePacketRanges.clear();
	tetIncidenceFullPacketCount = 0;
	tetIncidencePacketProgramValid = false;
}

bool AvbdSoftBodyCompiledData::validateTetIncidencePacketProgram() const
{
	if(tetIncidencePacketRanges.size() != particleCount)
		return false;
	PxU32 expectedPacketStart = 0;
	for(PxU32 localParticleIndex = 0;
		localParticleIndex < particleCount; localParticleIndex++)
	{
		const PxArray<AvbdParticleElementRef>& refs =
			elementAdjacency[localParticleIndex].tetRefs;
		const AvbdTetIncidencePacketRange& range =
			tetIncidencePacketRanges[localParticleIndex];
		const PxU32 expectedPacketCount =
			(refs.size() + eAVBD_TET_INCIDENCE_PACKET_WIDTH - 1) /
			eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		if(range.packetStart != expectedPacketStart ||
			range.packetCount != expectedPacketCount ||
			range.packetStart + range.packetCount >
				tetIncidencePackets.size())
			return false;
		for(PxU32 ordinal = 0; ordinal < refs.size(); ordinal++)
		{
			const PxU32 packetIndex = range.packetStart +
				ordinal / eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			const PxU32 lane = ordinal %
				eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			const AvbdTetIncidencePacket8& packet =
				tetIncidencePackets[packetIndex];
			if((packet.validMask & PxU8(1u << lane)) == 0u ||
				packet.tetIndices[lane] != refs[ordinal].index ||
				packet.vertexOrders[lane] != refs[ordinal].vOrder)
				return false;
		}
		if(expectedPacketCount)
		{
			const PxU32 tailLanes = refs.size() %
				eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			if(tailLanes)
			{
				const PxU8 expectedMask = PxU8(
					(1u << tailLanes) - 1u);
				if(tetIncidencePackets[
					range.packetStart + range.packetCount - 1].validMask !=
					expectedMask)
					return false;
			}
		}
		expectedPacketStart += expectedPacketCount;
	}
	return expectedPacketStart == tetIncidencePackets.size();
}

void AvbdSoftBodyCompiledData::buildTetIncidencePacketProgram()
{
	invalidateTetIncidencePacketProgram();
	if(!avbdUseTetMaterialPacketIr() || tetElements.empty())
		return;
	tetIncidencePacketRanges.resize(particleCount);
	for(PxU32 localParticleIndex = 0;
		localParticleIndex < particleCount; localParticleIndex++)
	{
		const PxArray<AvbdParticleElementRef>& refs =
			elementAdjacency[localParticleIndex].tetRefs;
		tetIncidenceFullPacketCount +=
			refs.size() / eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		AvbdTetIncidencePacketRange& range =
			tetIncidencePacketRanges[localParticleIndex];
		range.packetStart = tetIncidencePackets.size();
		range.packetCount = (refs.size() +
			eAVBD_TET_INCIDENCE_PACKET_WIDTH - 1) /
			eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		for(PxU32 packetOrdinal = 0;
			packetOrdinal < range.packetCount; packetOrdinal++)
		{
			AvbdTetIncidencePacket8 packet;
			for(PxU32 lane = 0;
				lane < eAVBD_TET_INCIDENCE_PACKET_WIDTH; lane++)
			{
				const PxU32 ordinal = packetOrdinal *
					eAVBD_TET_INCIDENCE_PACKET_WIDTH + lane;
				if(ordinal >= refs.size())
					break;
				packet.tetIndices[lane] = refs[ordinal].index;
				packet.vertexOrders[lane] = refs[ordinal].vOrder;
				packet.validMask |= PxU8(1u << lane);
			}
			tetIncidencePackets.pushBack(packet);
		}
	}
	tetIncidencePacketProgramValid = validateTetIncidencePacketProgram();
	PX_ASSERT(tetIncidencePacketProgramValid);
	if(!tetIncidencePacketProgramValid)
		invalidateTetIncidencePacketProgram();
}

void AvbdSoftBodyCompiledData::buildAdjacency(AvbdSoftBodyRuntimeState& runtime)
{
	elementAdjacency.resize(particleCount);
	for (PxU32 i = 0; i < particleCount; i++)
	{
		elementAdjacency[i].triRefs.clear();
		elementAdjacency[i].tetRefs.clear();
		elementAdjacency[i].bendRefs.clear();
	}

	for (PxU32 ei = 0; ei < triElements.size(); ei++)
	{
		const AvbdTriElement& tri = triElements[ei];
		PxU32 verts[3] = { tri.p0, tri.p1, tri.p2 };
		for (PxU8 v = 0; v < 3; v++)
		{
			PxU32 localIdx = verts[v] - particleStart;
			if (localIdx < particleCount)
			{
				AvbdParticleElementRef ref;
				ref.index = ei; ref.vOrder = v;
				ref.padding[0] = ref.padding[1] = ref.padding[2] = 0;
				elementAdjacency[localIdx].triRefs.pushBack(ref);
			}
		}
	}

	for (PxU32 ei = 0; ei < tetElements.size(); ei++)
	{
		const AvbdTetElement& tet = tetElements[ei];
		PxU32 verts[4] = { tet.p0, tet.p1, tet.p2, tet.p3 };
		for (PxU8 v = 0; v < 4; v++)
		{
			PxU32 localIdx = verts[v] - particleStart;
			if (localIdx < particleCount)
			{
				AvbdParticleElementRef ref;
				ref.index = ei; ref.vOrder = v;
				ref.padding[0] = ref.padding[1] = ref.padding[2] = 0;
				elementAdjacency[localIdx].tetRefs.pushBack(ref);
			}
		}
	}

	for (PxU32 ei = 0; ei < bendElements.size(); ei++)
	{
		const AvbdBendingElement& be = bendElements[ei];
		PxU32 verts[4] = { be.opp0, be.opp1, be.edgeStart, be.edgeEnd };
		for (PxU8 v = 0; v < 4; v++)
		{
			PxU32 localIdx = verts[v] - particleStart;
			if (localIdx < particleCount)
			{
				AvbdParticleElementRef ref;
				ref.index = ei; ref.vOrder = v;
				ref.padding[0] = ref.padding[1] = ref.padding[2] = 0;
				elementAdjacency[localIdx].bendRefs.pushBack(ref);
			}
		}
	}

	runtime.compileObjectiveProgram(particleStart, particleCount);
}

void AvbdSoftBodyCompiledData::buildParticlePrimalStructuralAccessDescriptor()
{
	particlePrimalStructuralConflictOffsets.clear();
	particlePrimalStructuralConflictIndices.clear();
	particlePrimalStructuralConflictValid = false;
	particlePrimalStructuralConflictOffsets.resize(particleCount + 1);
	if(!particleCount)
	{
		particlePrimalStructuralConflictValid = true;
		return;
	}

	// This temporary adjacency exists only at topology-compile time.  The
	// persistent form below is compact CSR, so a P4 color plan can walk it
	// without nested PxArray access or allocation in a solve step.
	PxArray<PxArray<PxU32> > localConflicts;
	localConflicts.resize(particleCount);
	auto appendClique = [this, &localConflicts](
		const PxU32* vertices, PxU32 vertexCount)
	{
		for(PxU32 sourceOrder = 0; sourceOrder < vertexCount;
			sourceOrder++)
		{
			const PxU32 source = vertices[sourceOrder];
			if(source < particleStart ||
				source - particleStart >= particleCount)
				continue;
			PxArray<PxU32>& conflicts =
				localConflicts[source - particleStart];
			for(PxU32 targetOrder = 0; targetOrder < vertexCount;
				targetOrder++)
			{
				if(targetOrder == sourceOrder)
					continue;
				const PxU32 target = vertices[targetOrder];
				if(target < particleStart ||
					target - particleStart >= particleCount)
					continue;
				conflicts.pushBack(target - particleStart);
			}
		}
	};

	for(PxU32 elementIndex = 0;
		elementIndex < triElements.size(); elementIndex++)
	{
		const AvbdTriElement& element = triElements[elementIndex];
		const PxU32 vertices[3] =
			{element.p0, element.p1, element.p2};
		appendClique(vertices, 3);
	}
	for(PxU32 elementIndex = 0;
		elementIndex < tetElements.size(); elementIndex++)
	{
		const AvbdTetElement& element = tetElements[elementIndex];
		const PxU32 vertices[4] =
			{element.p0, element.p1, element.p2, element.p3};
		appendClique(vertices, 4);
	}
	for(PxU32 elementIndex = 0;
		elementIndex < bendElements.size(); elementIndex++)
	{
		const AvbdBendingElement& element = bendElements[elementIndex];
		const PxU32 vertices[4] =
			{element.opp0, element.opp1,
				element.edgeStart, element.edgeEnd};
		appendClique(vertices, 4);
	}

	for(PxU32 localIndex = 0; localIndex < particleCount;
		localIndex++)
	{
		particlePrimalStructuralConflictOffsets[localIndex] =
			particlePrimalStructuralConflictIndices.size();
		PxArray<PxU32>& conflicts = localConflicts[localIndex];
		PxSort(conflicts.begin(), conflicts.size());
		PxU32 uniqueCount = 0;
		for(PxU32 conflictIndex = 0;
			conflictIndex < conflicts.size(); conflictIndex++)
		{
			const PxU32 conflict = conflicts[conflictIndex];
			if(conflict == localIndex ||
				(uniqueCount &&
					conflicts[uniqueCount - 1] == conflict))
				continue;
			conflicts[uniqueCount++] = conflict;
			particlePrimalStructuralConflictIndices.pushBack(conflict);
		}
		conflicts.resize(uniqueCount);
	}
	particlePrimalStructuralConflictOffsets[particleCount] =
		particlePrimalStructuralConflictIndices.size();
	particlePrimalStructuralConflictValid = true;
}

bool AvbdSoftBodyCompiledData::validateParticlePrimalStructuralAccessDescriptor() const
{
	if(!particlePrimalStructuralConflictValid ||
		particlePrimalStructuralConflictOffsets.size() !=
			particleCount + 1 ||
		particlePrimalStructuralConflictOffsets[0] != 0 ||
		particlePrimalStructuralConflictOffsets[particleCount] !=
			particlePrimalStructuralConflictIndices.size())
		return false;
	for(PxU32 localIndex = 0; localIndex < particleCount;
		localIndex++)
	{
		const PxU32 begin =
			particlePrimalStructuralConflictOffsets[localIndex];
		const PxU32 end =
			particlePrimalStructuralConflictOffsets[localIndex + 1];
		if(begin > end ||
			end > particlePrimalStructuralConflictIndices.size())
			return false;
		for(PxU32 conflictIndex = begin; conflictIndex < end;
			conflictIndex++)
		{
			const PxU32 conflict =
				particlePrimalStructuralConflictIndices[conflictIndex];
			if(conflict >= particleCount || conflict == localIndex ||
				(conflictIndex > begin &&
					particlePrimalStructuralConflictIndices[
						conflictIndex - 1] >= conflict))
				return false;
			const PxU32 reverseBegin =
				particlePrimalStructuralConflictOffsets[conflict];
			const PxU32 reverseEnd =
				particlePrimalStructuralConflictOffsets[conflict + 1];
			bool reverseFound = false;
			for(PxU32 reverseIndex = reverseBegin;
				reverseIndex < reverseEnd; reverseIndex++)
			{
				if(particlePrimalStructuralConflictIndices[
					reverseIndex] == localIndex)
				{
					reverseFound = true;
					break;
				}
			}
			if(!reverseFound)
				return false;
		}
	}
	return true;
}

void AvbdSoftBodyCompiledData::buildSurfaceTriangles(
	const PxArray<AvbdSoftParticle>& particles)
{
	surfaceTriangles.clear();
	surfaceVertices.clear();
	surfaceEdges.clear();
	surfaceTriangleElementIndices.clear();
	surfaceTriangleTetElementIndices.clear();

	if (!tetrahedra.empty())
	{
		// Newton-style sort + unique-count: collect all tet faces with a
		// canonical (sorted) key alongside the original winding, sort by
		// key, then emit faces that appear exactly once (boundary).

		struct FaceRecord
		{
			PxU64 keyHi;   // upper 40 bits of canonical key (smallest vertex)
			PxU32 keyLo;   // lower 20 bits packed into PxU32 for simplicity
			PxU32 v0, v1, v2;  // original winding
		};

		// More compact: pack full key into single PxU64
		// key = (sorted_a << 40) | (sorted_b << 20) | sorted_c
		// This fits since vertex indices < 2^20 = 1M for practical meshes

		const PxU32 numTets = tetrahedra.size() / 4;
		PxArray<PxU32> faceIndices; // flat: 3 * (numTets * 4) original winding indices
		PxArray<PxU64> faceKeys;    // canonical sorted key per face
		PxArray<PxU32> faceElementIndices;
		faceIndices.reserve(numTets * 4 * 3);
		faceKeys.reserve(numTets * 4);
		faceElementIndices.reserve(numTets * 4);

		// 4 faces per tet, following Newton's winding convention
		static const int faceLUT[4][3] = {
			{0,2,1}, {1,2,3}, {0,1,3}, {0,3,2}
		};

		for (PxU32 ti = 0; ti < numTets; ti++)
		{
			PxU32 tv[4] = {
				tetrahedra[ti * 4]     + particleStart,
				tetrahedra[ti * 4 + 1] + particleStart,
				tetrahedra[ti * 4 + 2] + particleStart,
				tetrahedra[ti * 4 + 3] + particleStart
			};
			for (int f = 0; f < 4; f++)
			{
				PxU32 a = tv[faceLUT[f][0]];
				PxU32 b = tv[faceLUT[f][1]];
				PxU32 c = tv[faceLUT[f][2]];
				faceIndices.pushBack(a);
				faceIndices.pushBack(b);
				faceIndices.pushBack(c);
				faceElementIndices.pushBack(ti);

				// Canonical key: sort the 3 indices
				PxU32 sa = a, sb = b, sc = c;
				if (sa > sb) { PxU32 t = sa; sa = sb; sb = t; }
				if (sb > sc) { PxU32 t = sb; sb = sc; sc = t; }
				if (sa > sb) { PxU32 t = sa; sa = sb; sb = t; }
				faceKeys.pushBack((PxU64(sa) << 40) | (PxU64(sb) << 20) | PxU64(sc));
			}
		}

		// Build an index array and sort it by key (indirect sort preserves
		// the original winding in faceIndices)
		const PxU32 numFaces = faceKeys.size();
		PxArray<PxU32> order;
		order.resize(numFaces);
		for (PxU32 i = 0; i < numFaces; i++) order[i] = i;

		const PxU64* keyPtr = faceKeys.begin();
		PxSort(order.begin(), order.size(),
			[keyPtr](PxU32 a, PxU32 b) { return keyPtr[a] < keyPtr[b]; });

		// Linear scan: emit faces whose key appears exactly once
		for (PxU32 i = 0; i < numFaces; )
		{
			PxU32 run = 1;
			while (i + run < numFaces && faceKeys[order[i + run]] == faceKeys[order[i]])
				run++;
			if (run == 1)
			{
				PxU32 base = order[i] * 3;
				surfaceTriangles.pushBack(faceIndices[base]);
				surfaceTriangles.pushBack(faceIndices[base + 1]);
				surfaceTriangles.pushBack(faceIndices[base + 2]);
				surfaceTriangleElementIndices.pushBack(
					faceElementIndices[order[i]]);
			}
			i += run;
		}
	}
	else if (!triangles.empty())
	{
		// For triangle mesh: all triangles are surface
		for (PxU32 i = 0; i + 2 < triangles.size(); i += 3)
		{
			surfaceTriangles.pushBack(triangles[i] + particleStart);
			surfaceTriangles.pushBack(triangles[i+1] + particleStart);
			surfaceTriangles.pushBack(triangles[i+2] + particleStart);
			surfaceTriangleElementIndices.pushBack(i / 3);
		}
	}

	surfaceVertices = surfaceTriangles;
	PxSort(surfaceVertices.begin(), surfaceVertices.size());
	if(!surfaceVertices.empty())
	{
		PxU32 writeIndex = 1;
		for(PxU32 i = 1; i < surfaceVertices.size(); ++i)
		{
			if(surfaceVertices[i] !=
				surfaceVertices[writeIndex - 1])
				surfaceVertices[writeIndex++] =
					surfaceVertices[i];
		}
		surfaceVertices.resize(writeIndex);
	}

	struct SurfaceHalfEdge
	{
		PxU64 key;
		PxU32 faceIndex;
	};
	PxArray<SurfaceHalfEdge> halfEdges;
	halfEdges.reserve(surfaceTriangles.size());
	for(PxU32 i = 0; i + 2 < surfaceTriangles.size(); i += 3)
	{
		const PxU32 vertices[3] =
		{
			surfaceTriangles[i],
			surfaceTriangles[i + 1],
			surfaceTriangles[i + 2]
		};
		for(PxU32 edgeIndex = 0; edgeIndex < 3; edgeIndex++)
		{
			const PxU32 a = vertices[edgeIndex];
			const PxU32 b = vertices[(edgeIndex + 1) % 3];
			const PxU32 lo = PxMin(a, b);
			const PxU32 hi = PxMax(a, b);
			SurfaceHalfEdge halfEdge;
			halfEdge.key = (PxU64(lo) << 32) | PxU64(hi);
			halfEdge.faceIndex = i / 3;
			halfEdges.pushBack(halfEdge);
		}
	}
	PxSort(
		halfEdges.begin(), halfEdges.size(),
		[](const SurfaceHalfEdge& a, const SurfaceHalfEdge& b)
		{
			return a.key < b.key;
		});
	for(PxU32 i = 0; i < halfEdges.size();)
	{
		const PxU64 key = halfEdges[i].key;
		PxU32 run = 1;
		while(i + run < halfEdges.size() &&
			halfEdges[i + run].key == key)
			run++;
		AvbdEdgeInfo edge;
		edge.p0 = PxU32(key >> 32);
		edge.p1 = PxU32(key & 0xffffffffu);
		edge.restLength =
			(particles[edge.p1].position -
			 particles[edge.p0].position).magnitude();
		edge.adjacentSurfaceFace0 = halfEdges[i].faceIndex;
		if(run == 2)
		{
			edge.adjacentSurfaceFace1 =
				halfEdges[i + 1].faceIndex;
			const PxU32 face0 = edge.adjacentSurfaceFace0 * 3;
			const PxU32 face1 = edge.adjacentSurfaceFace1 * 3;
			const PxVec3 normal0 =
				(particles[surfaceTriangles[face0 + 1]].position -
				 particles[surfaceTriangles[face0]].position).cross(
					particles[surfaceTriangles[face0 + 2]].position -
					particles[surfaceTriangles[face0]].position);
			const PxVec3 normal1 =
				(particles[surfaceTriangles[face1 + 1]].position -
				 particles[surfaceTriangles[face1]].position).cross(
					particles[surfaceTriangles[face1 + 2]].position -
					particles[surfaceTriangles[face1]].position);
			const PxReal normalLengthProduct =
				normal0.magnitude() * normal1.magnitude();
			// Stable rest-topology ownership prevents a planar seam from
			// flickering into and out of the edge manifold as the FEM surface
			// deforms. A genuine crease, curved tessellation edge, open edge or
			// non-manifold edge remains active.
			if(normalLengthProduct > 1.0e-12f &&
				normal0.dot(normal1) >=
					normalLengthProduct * 0.9999f)
				edge.collisionFeature = false;
		}
		surfaceEdges.pushBack(edge);
		i += run;
	}
}

void AvbdSoftBodyCompiledData::buildSurfaceTriangleTetElementIndices()
{
	// The mapping is a compiled-topology property: current and predicted
	// particle positions do not affect it.  Resolving it here removes the
	// former O(surfaceTriangles * tetElements) source-element scan from
	// every volume self-collision redetection.
	surfaceTriangleTetElementIndices.resize(
		surfaceTriangleElementIndices.size());
	for(PxU32 triangleIndex = 0;
		triangleIndex <
			surfaceTriangleTetElementIndices.size();
		triangleIndex++)
	{
		surfaceTriangleTetElementIndices[triangleIndex] =
			PX_MAX_U32;
	}
	if(tetElements.empty() || tetrahedra.empty())
		return;

	PxArray<PxU32> sourceToTetElement;
	sourceToTetElement.resize(tetrahedra.size() / 4);
	for(PxU32 sourceIndex = 0;
		sourceIndex < sourceToTetElement.size(); sourceIndex++)
		sourceToTetElement[sourceIndex] = PX_MAX_U32;
	for(PxU32 tetElementIndex = 0;
		tetElementIndex < tetElements.size(); tetElementIndex++)
	{
		const PxU32 sourceElementIndex =
			tetElements[tetElementIndex].sourceElementIndex;
		if(sourceElementIndex < sourceToTetElement.size())
			sourceToTetElement[sourceElementIndex] =
				tetElementIndex;
	}
	for(PxU32 triangleIndex = 0;
		triangleIndex <
			surfaceTriangleElementIndices.size(); triangleIndex++)
	{
		const PxU32 sourceElementIndex =
			surfaceTriangleElementIndices[triangleIndex];
		if(sourceElementIndex < sourceToTetElement.size())
			surfaceTriangleTetElementIndices[triangleIndex] =
				sourceToTetElement[sourceElementIndex];
	}
}

PxU32 AvbdSoftBodyCompiledData::buildSurfaceTriangleBvhNode(PxU32 first, PxU32 count)
{
	const PxU32 nodeIndex = surfaceTriangleBvhNodes.size();
	AvbdSurfaceTriangleBvhNode node;
	node.minimum = PxVec3(PX_MAX_F32);
	node.maximum = PxVec3(-PX_MAX_F32);
	node.leftChild = PX_MAX_U32;
	node.rightChild = PX_MAX_U32;
	node.firstTriangle = first;
	node.triangleCount = count;
	for(PxU32 entry = first; entry < first + count; entry++)
	{
		const PxU32 triangleIndex =
			surfaceTriangleBvhTriangleIndices[entry];
		const PxU32 triangleOffset = triangleIndex * 3;
		const PxU32 vertex0 = surfaceTriangles[triangleOffset];
		const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
		const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
		const PxVec3& point0 =
			selfCollisionRestPositions[vertex0 - particleStart];
		const PxVec3& point1 =
			selfCollisionRestPositions[vertex1 - particleStart];
		const PxVec3& point2 =
			selfCollisionRestPositions[vertex2 - particleStart];
		node.minimum = node.minimum.minimum(point0).
			minimum(point1).minimum(point2);
		node.maximum = node.maximum.maximum(point0).
			maximum(point1).maximum(point2);
	}
	surfaceTriangleBvhNodes.pushBack(node);
	if(count <= 4)
		return nodeIndex;

	const PxVec3 extent = node.maximum - node.minimum;
	const PxU32 axis = extent.y > extent.x && extent.y >= extent.z
		? 1u : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
	PxSort(
		surfaceTriangleBvhTriangleIndices.begin() + first, count,
		[this, axis](PxU32 lhs, PxU32 rhs)
		{
			const PxU32 lhsOffset = lhs * 3;
			const PxU32 rhsOffset = rhs * 3;
			const PxU32 lhsVertices[3] =
			{
				surfaceTriangles[lhsOffset],
				surfaceTriangles[lhsOffset + 1],
				surfaceTriangles[lhsOffset + 2]
			};
			const PxU32 rhsVertices[3] =
			{
				surfaceTriangles[rhsOffset],
				surfaceTriangles[rhsOffset + 1],
				surfaceTriangles[rhsOffset + 2]
			};
			const PxVec3 lhsCenter =
				(selfCollisionRestPositions[
					lhsVertices[0] - particleStart] +
				 selfCollisionRestPositions[
					lhsVertices[1] - particleStart] +
				 selfCollisionRestPositions[
					lhsVertices[2] - particleStart]) *
				(1.0f / 3.0f);
			const PxVec3 rhsCenter =
				(selfCollisionRestPositions[
					rhsVertices[0] - particleStart] +
				 selfCollisionRestPositions[
					rhsVertices[1] - particleStart] +
				 selfCollisionRestPositions[
					rhsVertices[2] - particleStart]) *
				(1.0f / 3.0f);
			const PxReal lhsValue = axis == 0 ? lhsCenter.x :
				axis == 1 ? lhsCenter.y : lhsCenter.z;
			const PxReal rhsValue = axis == 0 ? rhsCenter.x :
				axis == 1 ? rhsCenter.y : rhsCenter.z;
			return lhsValue == rhsValue ? lhs < rhs : lhsValue < rhsValue;
		});
	const PxU32 leftCount = count / 2;
	const PxU32 leftChild = buildSurfaceTriangleBvhNode(
		first, leftCount);
	const PxU32 rightChild = buildSurfaceTriangleBvhNode(
		first + leftCount, count - leftCount);
	surfaceTriangleBvhNodes[nodeIndex].leftChild = leftChild;
	surfaceTriangleBvhNodes[nodeIndex].rightChild = rightChild;
	return nodeIndex;
}

void AvbdSoftBodyCompiledData::buildSurfaceTriangleBvh()
{
	surfaceTriangleBvhTriangleIndices.clear();
	surfaceTriangleBvhNodes.clear();
	const PxU32 triangleCount = surfaceTriangles.size() / 3;
	if(triangleCount == 0 ||
		selfCollisionRestPositions.size() != particleCount)
		return;
	surfaceTriangleBvhTriangleIndices.resize(triangleCount);
	for(PxU32 triangleIndex = 0;
		triangleIndex < triangleCount; triangleIndex++)
		surfaceTriangleBvhTriangleIndices[triangleIndex] = triangleIndex;
	buildSurfaceTriangleBvhNode(0, triangleCount);
}

void AvbdSoftBodyCompiledData::refitSurfaceTriangleBvh(
	const AvbdSoftParticle* particles, bool swept,
	PxArray<AvbdSurfaceBvhNodeBounds>& bounds) const
{
	PX_ASSERT(bounds.size() == surfaceTriangleBvhNodes.size());
	for(PxU32 index = surfaceTriangleBvhNodes.size(); index > 0;)
	{
		const AvbdSurfaceTriangleBvhNode& node =
			surfaceTriangleBvhNodes[--index];
		AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[index];
		if(!node.isLeaf())
		{
			nodeBounds.minimum = bounds[node.leftChild].minimum.minimum(
				bounds[node.rightChild].minimum);
			nodeBounds.maximum = bounds[node.leftChild].maximum.maximum(
				bounds[node.rightChild].maximum);
			continue;
		}
		nodeBounds.minimum = PxVec3(PX_MAX_F32);
		nodeBounds.maximum = PxVec3(-PX_MAX_F32);
		for(PxU32 entry = node.firstTriangle;
			entry < node.firstTriangle + node.triangleCount; entry++)
		{
			const PxU32 triangleOffset =
				surfaceTriangleBvhTriangleIndices[entry] * 3;
			const PxU32 vertex0 = surfaceTriangles[triangleOffset];
			const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
			const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
			const AvbdSoftParticle& particle0 = particles[vertex0];
			const AvbdSoftParticle& particle1 = particles[vertex1];
			const AvbdSoftParticle& particle2 = particles[vertex2];
			nodeBounds.minimum = nodeBounds.minimum.minimum(particle0.position).
				minimum(particle1.position).minimum(particle2.position);
			nodeBounds.maximum = nodeBounds.maximum.maximum(particle0.position).
				maximum(particle1.position).maximum(particle2.position);
			if(swept)
			{
				nodeBounds.minimum = nodeBounds.minimum.minimum(
					particle0.initialPosition).minimum(
					particle1.initialPosition).minimum(
					particle2.initialPosition);
				nodeBounds.maximum = nodeBounds.maximum.maximum(
					particle0.initialPosition).maximum(
					particle1.initialPosition).maximum(
					particle2.initialPosition);
			}
		}
	}
}

void AvbdSoftBodyCompiledData::collectSurfaceTriangleBvhNodeCandidates(
	PxU32 nodeIndex,
	const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
	const PxVec3& queryMinimum,
	const PxVec3& queryMaximum, PxReal margin,
	PxArray<PxU32>& candidates) const
{
	const AvbdSurfaceTriangleBvhNode& node =
		surfaceTriangleBvhNodes[nodeIndex];
	const AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[nodeIndex];
	if(nodeBounds.minimum.x > queryMaximum.x + margin ||
		nodeBounds.maximum.x < queryMinimum.x - margin ||
		nodeBounds.minimum.y > queryMaximum.y + margin ||
		nodeBounds.maximum.y < queryMinimum.y - margin ||
		nodeBounds.minimum.z > queryMaximum.z + margin ||
		nodeBounds.maximum.z < queryMinimum.z - margin)
		return;
	if(!node.isLeaf())
	{
		collectSurfaceTriangleBvhNodeCandidates(
			node.leftChild, bounds, queryMinimum, queryMaximum,
			margin, candidates);
		collectSurfaceTriangleBvhNodeCandidates(
			node.rightChild, bounds, queryMinimum, queryMaximum,
			margin, candidates);
		return;
	}
	for(PxU32 entry = node.firstTriangle;
		entry < node.firstTriangle + node.triangleCount; entry++)
		candidates.pushBack(surfaceTriangleBvhTriangleIndices[entry]);
}

void AvbdSoftBodyCompiledData::collectSurfaceTriangleBvhCandidates(
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin,
	const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
	PxArray<PxU32>& candidates) const
{
	candidates.clear();
	if(surfaceTriangleBvhNodes.empty())
		return;
	PX_ASSERT(bounds.size() == surfaceTriangleBvhNodes.size());
	collectSurfaceTriangleBvhNodeCandidates(
		0, bounds, queryMinimum, queryMaximum, margin, candidates);
	PxSort(candidates.begin(), candidates.size());
}

PxU32 AvbdSoftBodyCompiledData::buildSurfaceEdgeBvhNode(PxU32 first, PxU32 count)
{
	const PxU32 nodeIndex = surfaceEdgeBvhNodes.size();
	AvbdSurfaceEdgeBvhNode node;
	node.minimum = PxVec3(PX_MAX_F32);
	node.maximum = PxVec3(-PX_MAX_F32);
	node.leftChild = PX_MAX_U32;
	node.rightChild = PX_MAX_U32;
	node.firstEdge = first;
	node.edgeCount = count;
	for(PxU32 entry = first; entry < first + count; entry++)
	{
		const AvbdEdgeInfo& edge = surfaceEdges[
			surfaceEdgeBvhEdgeIndices[entry]];
		const PxVec3& point0 = selfCollisionRestPositions[
			edge.p0 - particleStart];
		const PxVec3& point1 = selfCollisionRestPositions[
			edge.p1 - particleStart];
		node.minimum = node.minimum.minimum(point0).minimum(point1);
		node.maximum = node.maximum.maximum(point0).maximum(point1);
	}
	surfaceEdgeBvhNodes.pushBack(node);
	if(count <= 4)
		return nodeIndex;

	const PxVec3 extent = node.maximum - node.minimum;
	const PxU32 axis = extent.y > extent.x && extent.y >= extent.z
		? 1u : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
	PxSort(
		surfaceEdgeBvhEdgeIndices.begin() + first, count,
		[this, axis](PxU32 lhs, PxU32 rhs)
		{
			const AvbdEdgeInfo& lhsEdge = surfaceEdges[lhs];
			const AvbdEdgeInfo& rhsEdge = surfaceEdges[rhs];
			const PxVec3 lhsCenter = (
				selfCollisionRestPositions[
					lhsEdge.p0 - particleStart] +
				selfCollisionRestPositions[
					lhsEdge.p1 - particleStart]) * 0.5f;
			const PxVec3 rhsCenter = (
				selfCollisionRestPositions[
					rhsEdge.p0 - particleStart] +
				selfCollisionRestPositions[
					rhsEdge.p1 - particleStart]) * 0.5f;
			const PxReal lhsValue = axis == 0 ? lhsCenter.x :
				axis == 1 ? lhsCenter.y : lhsCenter.z;
			const PxReal rhsValue = axis == 0 ? rhsCenter.x :
				axis == 1 ? rhsCenter.y : rhsCenter.z;
			return lhsValue == rhsValue ? lhs < rhs : lhsValue < rhsValue;
		});
	const PxU32 leftCount = count / 2;
	const PxU32 leftChild = buildSurfaceEdgeBvhNode(first, leftCount);
	const PxU32 rightChild = buildSurfaceEdgeBvhNode(
		first + leftCount, count - leftCount);
	surfaceEdgeBvhNodes[nodeIndex].leftChild = leftChild;
	surfaceEdgeBvhNodes[nodeIndex].rightChild = rightChild;
	return nodeIndex;
}

void AvbdSoftBodyCompiledData::buildSurfaceEdgeBvh()
{
	surfaceEdgeBvhEdgeIndices.clear();
	surfaceEdgeBvhNodes.clear();
	if(surfaceEdges.empty() ||
		selfCollisionRestPositions.size() != particleCount)
		return;
	surfaceEdgeBvhEdgeIndices.resize(surfaceEdges.size());
	for(PxU32 edgeIndex = 0; edgeIndex < surfaceEdges.size(); edgeIndex++)
		surfaceEdgeBvhEdgeIndices[edgeIndex] = edgeIndex;
	buildSurfaceEdgeBvhNode(0, surfaceEdges.size());
}

void AvbdSoftBodyCompiledData::refitSurfaceEdgeBvh(
	const AvbdSoftParticle* particles, bool swept,
	PxArray<AvbdSurfaceBvhNodeBounds>& bounds) const
{
	PX_ASSERT(bounds.size() == surfaceEdgeBvhNodes.size());
	for(PxU32 index = surfaceEdgeBvhNodes.size(); index > 0;)
	{
		const AvbdSurfaceEdgeBvhNode& node =
			surfaceEdgeBvhNodes[--index];
		AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[index];
		if(!node.isLeaf())
		{
			nodeBounds.minimum = bounds[node.leftChild].minimum.
				minimum(bounds[node.rightChild].minimum);
			nodeBounds.maximum = bounds[node.leftChild].maximum.
				maximum(bounds[node.rightChild].maximum);
			continue;
		}
		nodeBounds.minimum = PxVec3(PX_MAX_F32);
		nodeBounds.maximum = PxVec3(-PX_MAX_F32);
		for(PxU32 entry = node.firstEdge;
			entry < node.firstEdge + node.edgeCount; entry++)
		{
			const AvbdEdgeInfo& edge = surfaceEdges[
				surfaceEdgeBvhEdgeIndices[entry]];
			const AvbdSoftParticle& particle0 = particles[edge.p0];
			const AvbdSoftParticle& particle1 = particles[edge.p1];
			nodeBounds.minimum = nodeBounds.minimum.minimum(particle0.position).
				minimum(particle1.position);
			nodeBounds.maximum = nodeBounds.maximum.maximum(particle0.position).
				maximum(particle1.position);
			if(swept)
			{
				nodeBounds.minimum = nodeBounds.minimum.minimum(
					particle0.initialPosition).minimum(
					particle1.initialPosition);
				nodeBounds.maximum = nodeBounds.maximum.maximum(
					particle0.initialPosition).maximum(
					particle1.initialPosition);
			}
		}
	}
}

void AvbdSoftBodyCompiledData::collectSurfaceEdgeBvhNodeCandidates(
	PxU32 nodeIndex,
	const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
	const PxVec3& queryMinimum,
	const PxVec3& queryMaximum, PxReal margin,
	PxArray<PxU32>& candidates) const
{
	const AvbdSurfaceEdgeBvhNode& node = surfaceEdgeBvhNodes[nodeIndex];
	const AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[nodeIndex];
	if(nodeBounds.minimum.x > queryMaximum.x + margin ||
		nodeBounds.maximum.x < queryMinimum.x - margin ||
		nodeBounds.minimum.y > queryMaximum.y + margin ||
		nodeBounds.maximum.y < queryMinimum.y - margin ||
		nodeBounds.minimum.z > queryMaximum.z + margin ||
		nodeBounds.maximum.z < queryMinimum.z - margin)
		return;
	if(!node.isLeaf())
	{
		collectSurfaceEdgeBvhNodeCandidates(
			node.leftChild, bounds, queryMinimum, queryMaximum, margin,
			candidates);
		collectSurfaceEdgeBvhNodeCandidates(
			node.rightChild, bounds, queryMinimum, queryMaximum, margin,
			candidates);
		return;
	}
	for(PxU32 entry = node.firstEdge;
		entry < node.firstEdge + node.edgeCount; entry++)
		candidates.pushBack(surfaceEdgeBvhEdgeIndices[entry]);
}

void AvbdSoftBodyCompiledData::collectSurfaceEdgeBvhCandidates(
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin,
	const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
	PxArray<PxU32>& candidates) const
{
	candidates.clear();
	if(surfaceEdgeBvhNodes.empty())
		return;
	PX_ASSERT(bounds.size() == surfaceEdgeBvhNodes.size());
	collectSurfaceEdgeBvhNodeCandidates(
		0, bounds, queryMinimum, queryMaximum, margin, candidates);
	// Self EE evaluates every eligible unordered pair exactly once using
	// min/max edge identity; unlike soft-soft VF it does not choose a
	// first triangle or a tie owner from candidate order.  The fixed tree
	// traversal is deterministic, and avoiding a per-query candidate sort
	// is essential when a dense cloth issues one query per boundary edge.
}

void AvbdSoftBodyCompiledData::buildSelfCollisionRestVertexTriangleFilter()
{
	const PxReal filterDistance =
		PxMax(selfCollisionFilterDistance, 0.0f);
	selfCollisionRestFilteredTriangles.clear();
	selfCollisionRestFilterCacheDistance = filterDistance;
	selfCollisionRestFilterCacheValid = false;
	selfCollisionRestFilterCacheFallback = false;
	if(filterDistance <= 0.0f ||
		selfCollisionRestPositions.size() != particleCount)
		return;

	selfCollisionRestFilteredTriangles.resize(particleCount);
	const PxU64 pairBudget =
		(2ull * 1024ull * 1024ull) / sizeof(PxU32);
	PxU64 pairCount = 0;
	PxArray<AvbdSelfCollisionTriangleBounds> triangleBounds;
	triangleBounds.reserve(surfaceTriangles.size() / 3);
	for(PxU32 triangleOffset = 0;
		triangleOffset + 2 < surfaceTriangles.size();
		triangleOffset += 3)
	{
		const PxU32 vertex0 = surfaceTriangles[triangleOffset];
		const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
		const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
		if(vertex0 < particleStart || vertex1 < particleStart ||
			vertex2 < particleStart ||
			vertex0 - particleStart >= particleCount ||
			vertex1 - particleStart >= particleCount ||
			vertex2 - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionTriangleBounds bounds;
		bounds.triangleOffset = triangleOffset;
		bounds.minimum =
			selfCollisionRestPositions[vertex0 - particleStart].minimum(
				selfCollisionRestPositions[vertex1 - particleStart]).minimum(
				selfCollisionRestPositions[vertex2 - particleStart]);
		bounds.maximum =
			selfCollisionRestPositions[vertex0 - particleStart].maximum(
				selfCollisionRestPositions[vertex1 - particleStart]).maximum(
				selfCollisionRestPositions[vertex2 - particleStart]);
		triangleBounds.pushBack(bounds);
	}
	PxSort(
		triangleBounds.begin(), triangleBounds.size(),
		[](const AvbdSelfCollisionTriangleBounds& a,
		   const AvbdSelfCollisionTriangleBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});

	PxArray<AvbdSelfCollisionVertexSweepEntry> sortedVertices;
	sortedVertices.resize(particleCount);
	for(PxU32 localIndex = 0; localIndex < particleCount;
		localIndex++)
	{
		sortedVertices[localIndex].localIndex = localIndex;
		sortedVertices[localIndex].minimumX =
			selfCollisionRestPositions[localIndex].x;
		sortedVertices[localIndex].maximumX =
			selfCollisionRestPositions[localIndex].x;
	}
	PxSort(
		sortedVertices.begin(), sortedVertices.size(),
		[](const AvbdSelfCollisionVertexSweepEntry& a,
		   const AvbdSelfCollisionVertexSweepEntry& b)
		{
			return a.minimumX < b.minimumX;
		});

	PxArray<PxU32> activeTriangles;
	activeTriangles.reserve(triangleBounds.size());
	PxU32 triangleCursor = 0;
	for(PxU32 sortedVertexIndex = 0;
		sortedVertexIndex < sortedVertices.size(); sortedVertexIndex++)
	{
		const PxU32 localIndex =
			sortedVertices[sortedVertexIndex].localIndex;
		const PxVec3& point = selfCollisionRestPositions[localIndex];
		while(triangleCursor < triangleBounds.size() &&
			triangleBounds[triangleCursor].minimum.x <=
				point.x + filterDistance)
			activeTriangles.pushBack(triangleCursor++);
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size();)
		{
			const AvbdSelfCollisionTriangleBounds& bounds =
				triangleBounds[activeTriangles[activeIndex]];
			if(bounds.maximum.x < point.x - filterDistance)
			{
				activeTriangles[activeIndex] = activeTriangles.back();
				activeTriangles.popBack();
				continue;
			}
			activeIndex++;
		}
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size(); activeIndex++)
		{
			const AvbdSelfCollisionTriangleBounds& bounds =
				triangleBounds[activeTriangles[activeIndex]];
			if(bounds.minimum.y > point.y + filterDistance ||
				bounds.maximum.y < point.y - filterDistance ||
				bounds.minimum.z > point.z + filterDistance ||
				bounds.maximum.z < point.z - filterDistance)
				continue;
			const PxU32 triangleOffset = bounds.triangleOffset;
			const PxU32 vertex0 = surfaceTriangles[triangleOffset];
			const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
			const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
			const PxU32 localVertex0 = vertex0 - particleStart;
			const PxU32 localVertex1 = vertex1 - particleStart;
			const PxU32 localVertex2 = vertex2 - particleStart;
			if(localIndex == localVertex0 || localIndex == localVertex1 ||
				localIndex == localVertex2)
				continue;
			const PxReal restDistance =
				avbdGetRestPointTriangleDistance(
					point,
					selfCollisionRestPositions[localVertex0],
					selfCollisionRestPositions[localVertex1],
					selfCollisionRestPositions[localVertex2]);
			if(restDistance > filterDistance)
				continue;
			if(pairCount >= pairBudget)
			{
				selfCollisionRestFilteredTriangles.clear();
				selfCollisionRestFilterCacheFallback = true;
				return;
			}
			selfCollisionRestFilteredTriangles[localIndex].pushBack(
				triangleOffset / 3);
			++pairCount;
		}
	}
	for(PxU32 localIndex = 0;
		localIndex < selfCollisionRestFilteredTriangles.size();
		localIndex++)
	{
		PxArray<PxU32>& filteredForVertex =
			selfCollisionRestFilteredTriangles[localIndex];
		PxSort(filteredForVertex.begin(), filteredForVertex.size());
	}
	selfCollisionRestFilterCacheValid = true;
}

void AvbdSoftBodyCompiledData::ensureSelfCollisionRestVertexTriangleFilter()
{
	const PxReal filterDistance =
		PxMax(selfCollisionFilterDistance, 0.0f);
	if(selfCollisionRestFilterCacheDistance == filterDistance &&
		(selfCollisionRestFilterCacheValid ||
			selfCollisionRestFilterCacheFallback))
		return;
	buildSelfCollisionRestVertexTriangleFilter();
}

void AvbdSoftBodyCompiledData::buildElements(
	const PxArray<AvbdSoftParticle>& particles,
	AvbdSoftBodyMaterialData& material,
	AvbdSoftBodyRuntimeState& runtime)
{
	material.computeLameParameters();
	if (!triangles.empty())
		buildTriElements(particles);
	if (!tetrahedra.empty())
		buildTetElements(particles);
	if (!triangles.empty())
		buildBendingElements(particles);
	buildEdges(particles);
	buildAdjacency(runtime);
	buildTetIncidencePacketProgram();
	buildParticlePrimalStructuralAccessDescriptor();
	PX_ASSERT(validateParticlePrimalStructuralAccessDescriptor());
	buildSurfaceTriangles(particles);
	buildSurfaceTriangleTetElementIndices();
	buildSurfaceTriangleBvh();
	buildSurfaceEdgeBvh();
	buildSelfCollisionRestVertexTriangleFilter();
}

} // namespace Dy
} // namespace physx
