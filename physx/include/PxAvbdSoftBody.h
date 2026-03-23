// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef PX_AVBD_SOFT_BODY_H
#define PX_AVBD_SOFT_BODY_H

// =============================================================================
// Public AVBD Soft Body / Cloth -- unified AVBD energy-based deformable system
//
// All energy terms (elastic, contact, pin) use the AVBD framework: adaptive
// proximal penalty with dual variable update ensures convergence independent
// of update order (Jacobi-safe).  Pure VBD (no proximal term) is available
// as a lightweight fallback for simple elastic-only scenarios.
//
// Elastic forces: StVK (triangles), Neo-Hookean (tetrahedra), dihedral bending
// Constraints: contact (ground/soft-soft/soft-rigid), kinematic pins
//
// This header contains only public-API types (PxVec3, PxMat33, PxArray, etc.)
// and can be included by snippets and user code.
//
// Reference: AVBD (SIGGRAPH 2024)
// =============================================================================

#include "foundation/PxAllocator.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "foundation/PxMat33.h"
#include "foundation/PxQuat.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxSort.h"
#include "foundation/PxVec3.h"


namespace physx
{
namespace Dy
{

// =============================================================================
// PxMat33 helper utilities (column-major <-> element access)
// =============================================================================

PX_FORCE_INLINE PxMat33 avbdOuter(const PxVec3& a, const PxVec3& b)
{
	return PxMat33(a * b.x, a * b.y, a * b.z);
}

PX_FORCE_INLINE PxMat33 avbdSkew(const PxVec3& v)
{
	return PxMat33(
		PxVec3(0.0f,  v.z, -v.y),
		PxVec3(-v.z,  0.0f,  v.x),
		PxVec3( v.y, -v.x,  0.0f));
}

PX_FORCE_INLINE PxVec3 avbdMatRow(const PxMat33& m, int row)
{
	return PxVec3(m.column0[row], m.column1[row], m.column2[row]);
}

PX_FORCE_INLINE PxReal avbdColSum(const PxVec3& col)
{
	return col.x + col.y + col.z;
}

// =============================================================================
// AvbdSoftParticle -- 3-DOF mass point (no rotation)
// =============================================================================

struct PX_ALIGN_PREFIX(16) AvbdSoftParticle
{
	PxVec3 position;
	PxReal invMass;

	PxVec3 velocity;
	PxReal mass;

	PxVec3 prevVelocity;
	PxReal damping;

	PxVec3 initialPosition;
	PxReal padding0;

	PxVec3 predictedPosition;
	PxReal elasticK;         // AVBD elastic proximal weight (adaptive)

	PxVec3 outerPosition;    // position snapshot at start of outer iteration (proximal anchor)
	PxReal elasticKMax;      // AVBD elastic proximal upper bound

	AvbdSoftParticle()
		: position(0.0f), invMass(1.0f), velocity(0.0f), mass(1.0f),
		  prevVelocity(0.0f), damping(0.0f), initialPosition(0.0f), padding0(0.0f),
		  predictedPosition(0.0f), elasticK(0.0f), outerPosition(0.0f), elasticKMax(1e6f) {}

	PX_FORCE_INLINE bool isStatic() const { return invMass <= 0.0f; }

	PX_FORCE_INLINE void computePrediction(PxReal dt, const PxVec3& gravity)
	{
		if (invMass <= 0.0f) return;
		predictedPosition = position + velocity * dt + gravity * (dt * dt);
		initialPosition = position;
	}

	PX_FORCE_INLINE void updateVelocityFromPosition(PxReal invDt)
	{
		if (invMass <= 0.0f) return;
		prevVelocity = velocity;
		PxVec3 v = (position - initialPosition) * invDt;
		// NaN/inf guard: reset to zero on degenerate values
		if (v.x != v.x || PxAbs(v.x) > 1e6f ||
		    v.y != v.y || PxAbs(v.y) > 1e6f ||
		    v.z != v.z || PxAbs(v.z) > 1e6f)
		{
			velocity = PxVec3(0.0f);
			position = initialPosition;
		}
		else
			velocity = v;
	}
} PX_ALIGN_SUFFIX(16);

// =============================================================================
// VBD Element types -- precomputed rest-state data
// =============================================================================

struct AvbdTriElement
{
	PxU32 p0, p1, p2;
	PxReal DmInv00, DmInv01;
	PxReal DmInv10, DmInv11;
	PxReal restArea;
};

struct AvbdTetElement
{
	PxU32 p0, p1, p2, p3;
	PxMat33 DmInv;
	PxReal restVolume;
};

struct AvbdBendingElement
{
	PxU32 opp0, opp1;
	PxU32 edgeStart, edgeEnd;
	PxReal restAngle;
	PxReal restLength;
};

struct AvbdEdgeInfo
{
	PxU32 p0, p1;
	PxReal restLength;
};

// =============================================================================
// AVBD constraint types -- adaptive penalty only
// =============================================================================

struct AvbdSoftAttachment
{
	PxU32 particleIdx;
	PxU32 rigidBodyIdx;
	PxVec3 localOffset;
	PxReal k;
	PxReal kMax;

	AvbdSoftAttachment()
		: particleIdx(0), rigidBodyIdx(0), localOffset(0.0f),
		  k(1e3f), kMax(1e5f) {}
};

struct AvbdKinematicPin
{
	PxU32 particleIdx;
	PxVec3 worldTarget;
	PxReal k;
	PxReal kMax;

	AvbdKinematicPin()
		: particleIdx(0), worldTarget(0.0f), k(1e4f), kMax(1e6f) {}
};

struct AvbdSoftContact
{
	PxU32 particleIdx;
	PxU32 rigidBodyIdx;     // PX_MAX_U32 = ground, any other value = soft/rigid body
	PxVec3 normal;          // penalty direction (closest-point, VBD-stable)
	PxVec3 projNormal;      // projection direction (face-normal corrected, always outward)
	PxReal depth;
	PxReal margin;          // contact shell thickness (used for proximity contacts)
	PxReal friction;
	PxVec3 tangent1, tangent2;
	PxVec3 surfacePoint;    // reference point on the other body's surface (world space)

	PxReal k;
	PxReal ke;

	AvbdSoftContact()
		: particleIdx(0), rigidBodyIdx(PX_MAX_U32), normal(0.0f, 1.0f, 0.0f),
		  projNormal(0.0f, 1.0f, 0.0f),
		  depth(0.0f), margin(0.0f), friction(0.5f), tangent1(1.0f, 0.0f, 0.0f),
		  tangent2(0.0f, 0.0f, 1.0f), surfacePoint(0.0f), k(1e4f), ke(1e6f) {}
};

// =============================================================================
// Per-particle element adjacency
// =============================================================================

struct AvbdParticleElementRef
{
	PxU32 index;
	PxU8 vOrder;
	PxU8 padding[3];
};

struct AvbdParticleAdjacency
{
	PxArray<AvbdParticleElementRef> triRefs;
	PxArray<AvbdParticleElementRef> tetRefs;
	PxArray<AvbdParticleElementRef> bendRefs;
	PxArray<PxU32>                  attachmentIndices;
	PxArray<PxU32>                  pinIndices;
};

// =============================================================================
// AvbdSoftBody -- aggregate of mesh topology, materials, elements
// =============================================================================

struct AvbdSoftBody
{
	PxU32 particleStart;
	PxU32 particleCount;

	PxArray<PxU32> tetrahedra;
	PxArray<PxU32> triangles;

	PxReal youngsModulus;
	PxReal poissonsRatio;
	PxReal density;
	PxReal damping;
	PxReal bendingStiffness;
	PxReal thickness;

	PxReal mu;
	PxReal lambda;

	PxArray<AvbdTriElement> triElements;
	PxArray<AvbdTetElement> tetElements;
	PxArray<AvbdBendingElement> bendElements;
	PxArray<AvbdEdgeInfo> edges;

	PxArray<AvbdSoftAttachment> attachments;
	PxArray<AvbdKinematicPin> pins;

	PxArray<AvbdParticleAdjacency> adjacency;

	PxArray<PxU32> surfaceTriangles;  // boundary face indices (3 per tri, global particle indices)

	AvbdSoftBody()
		: particleStart(0), particleCount(0),
		  youngsModulus(1e5f), poissonsRatio(0.3f),
		  density(1000.0f), damping(0.0f),
		  bendingStiffness(0.0f), thickness(0.01f),
		  mu(0.0f), lambda(0.0f) {}

	void computeLameParameters()
	{
		mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
		lambda = youngsModulus * poissonsRatio /
		         ((1.0f + poissonsRatio) * (1.0f - 2.0f * poissonsRatio));
	}

	static PxReal computeDihedralAngle(const PxVec3& x0, const PxVec3& x1,
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

	void buildTriElements(const PxArray<AvbdSoftParticle>& particles)
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
			tri.DmInv00 =  d11 * invDet;
			tri.DmInv01 = -d01 * invDet;
			tri.DmInv10 = -d10 * invDet;
			tri.DmInv11 =  d00 * invDet;
			tri.restArea = area;
			triElements.pushBack(tri);
		}
	}

	void buildTetElements(const PxArray<AvbdSoftParticle>& particles)
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
			tet.DmInv = Dm.getInverse();
			tet.restVolume = vol;
			tetElements.pushBack(tet);
		}
	}

	void buildBendingElements(const PxArray<AvbdSoftParticle>& particles)
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
				be.restAngle = computeDihedralAngle(
					particles[be.opp0].position, particles[be.opp1].position,
					particles[edgeA].position, particles[edgeB].position);
				be.restLength = (particles[edgeB].position - particles[edgeA].position).magnitude();
				bendElements.pushBack(be);
			}
			i += run - 1; // advance past run
		}
	}

	void buildEdges(const PxArray<AvbdSoftParticle>& particles)
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

	void buildAdjacency()
	{
		adjacency.resize(particleCount);
		for (PxU32 i = 0; i < particleCount; i++)
		{
			adjacency[i].triRefs.clear();
			adjacency[i].tetRefs.clear();
			adjacency[i].bendRefs.clear();
			adjacency[i].attachmentIndices.clear();
			adjacency[i].pinIndices.clear();
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
					adjacency[localIdx].triRefs.pushBack(ref);
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
					adjacency[localIdx].tetRefs.pushBack(ref);
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
					adjacency[localIdx].bendRefs.pushBack(ref);
				}
			}
		}

		for (PxU32 ai = 0; ai < attachments.size(); ai++)
		{
			PxU32 localIdx = attachments[ai].particleIdx - particleStart;
			if (localIdx < particleCount)
				adjacency[localIdx].attachmentIndices.pushBack(ai);
		}

		for (PxU32 pi = 0; pi < pins.size(); pi++)
		{
			PxU32 localIdx = pins[pi].particleIdx - particleStart;
			if (localIdx < particleCount)
				adjacency[localIdx].pinIndices.pushBack(pi);
		}
	}

	void buildSurfaceTriangles()
	{
		surfaceTriangles.clear();

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
			faceIndices.reserve(numTets * 4 * 3);
			faceKeys.reserve(numTets * 4);

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
			}
		}
	}

	void buildElements(const PxArray<AvbdSoftParticle>& particles)
	{
		computeLameParameters();
		if (!triangles.empty())
			buildTriElements(particles);
		if (!tetrahedra.empty())
			buildTetElements(particles);
		if (!triangles.empty())
			buildBendingElements(particles);
		buildEdges(particles);
		buildAdjacency();
		buildSurfaceTriangles();
	}
};

// =============================================================================
// VBD Force/Hessian evaluators
// =============================================================================

PX_FORCE_INLINE void avbdEvaluateStVKForceHessian(
	const AvbdTriElement& tri, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	PxVec3 x0 = particles[tri.p0].position;
	PxVec3 x01 = particles[tri.p1].position - x0;
	PxVec3 x02 = particles[tri.p2].position - x0;

	PxReal D00 = tri.DmInv00, D01 = tri.DmInv01;
	PxReal D10 = tri.DmInv10, D11 = tri.DmInv11;

	PxVec3 f0 = x01 * D00 + x02 * D10;
	PxVec3 f1 = x01 * D01 + x02 * D11;

	PxReal f0f0 = f0.dot(f0);
	PxReal f1f1 = f1.dot(f1);
	PxReal f0f1 = f0.dot(f1);

	PxReal G00 = 0.5f * (f0f0 - 1.0f);
	PxReal G11 = 0.5f * (f1f1 - 1.0f);
	PxReal G01 = 0.5f * f0f1;

	PxReal Gfro2 = G00 * G00 + G11 * G11 + 2.0f * G01 * G01;
	if (Gfro2 < 1e-20f)
	{
		outForce = PxVec3(0.0f);
		outHessian = PxMat33(PxZero);
		return;
	}

	PxReal trG = G00 + G11;
	PxReal ltrG = lam * trG;
	PxReal twoMu = 2.0f * mu;
	PxVec3 PK1_0 = f0 * (twoMu * G00 + ltrG) + f1 * (twoMu * G01);
	PxVec3 PK1_1 = f0 * (twoMu * G01) + f1 * (twoMu * G11 + ltrG);

	PxReal df0, df1;
	if (vOrder == 0)      { df0 = -D00 - D10; df1 = -D01 - D11; }
	else if (vOrder == 1) { df0 = D00; df1 = D01; }
	else                  { df0 = D10; df1 = D11; }

	outForce = (PK1_0 * df0 + PK1_1 * df1) * (-tri.restArea);

	PxReal df0sq = df0 * df0;
	PxReal df1sq = df1 * df1;
	PxReal df0df1 = df0 * df1;

	PxReal Ic = f0f0 + f1f1;
	PxReal two_dpsi_dIc = -mu + (0.5f * Ic - 1.0f) * lam;
	PxMat33 I33 = PxMat33(PxIdentity);

	PxMat33 f0f0m = avbdOuter(f0, f0);
	PxMat33 f1f1m = avbdOuter(f1, f1);
	PxMat33 f0f1m = avbdOuter(f0, f1);
	PxMat33 f1f0m = avbdOuter(f1, f0);

	PxMat33 H00 = f0f0m * lam + I33 * two_dpsi_dIc
	            + (I33 * f0f0 + f0f0m * 2.0f + f1f1m) * mu;
	PxMat33 H01 = f0f1m * lam + (I33 * f0f1 + f1f0m) * mu;
	PxMat33 H11 = f1f1m * lam + I33 * two_dpsi_dIc
	            + (I33 * f1f1 + f1f1m * 2.0f + f0f0m) * mu;

	PxReal area = tri.restArea;
	outHessian = H00 * (df0sq * area) + H11 * (df1sq * area)
	           + (H01 + H01.getTranspose()) * (df0df1 * area);
}

PX_FORCE_INLINE void avbdEvaluateNeoHookeanForceHessian(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	PxVec3 p0 = particles[tet.p0].position;
	PxVec3 e1 = particles[tet.p1].position - p0;
	PxVec3 e2 = particles[tet.p2].position - p0;
	PxVec3 e3 = particles[tet.p3].position - p0;

	PxMat33 Ds(e1, e2, e3);
	PxMat33 F = Ds * tet.DmInv;
	PxReal J = F.getDeterminant();

	PxReal lam_safe = (PxAbs(lam) < 1e-6f) ? 1e-6f : lam;
	PxReal alpha = 1.0f + mu / lam_safe;

	PxMat33 cof;
	cof.column0 = PxVec3(
		F.column1.y * F.column2.z - F.column2.y * F.column1.z,
		F.column2.x * F.column1.z - F.column1.x * F.column2.z,
		F.column1.x * F.column2.y - F.column2.x * F.column1.y);
	cof.column1 = PxVec3(
		F.column2.y * F.column0.z - F.column0.y * F.column2.z,
		F.column0.x * F.column2.z - F.column2.x * F.column0.z,
		F.column2.x * F.column0.y - F.column0.x * F.column2.y);
	cof.column2 = PxVec3(
		F.column0.y * F.column1.z - F.column1.y * F.column0.z,
		F.column1.x * F.column0.z - F.column0.x * F.column1.z,
		F.column0.x * F.column1.y - F.column1.x * F.column0.y);

	const PxMat33& DI = tet.DmInv;
	PxVec3 m;
	if (vOrder == 0)
	{
		m = PxVec3(
			-avbdColSum(DI.column0),
			-avbdColSum(DI.column1),
			-avbdColSum(DI.column2));
	}
	else if (vOrder == 1) { m = avbdMatRow(DI, 0); }
	else if (vOrder == 2) { m = avbdMatRow(DI, 1); }
	else                  { m = avbdMatRow(DI, 2); }

	PxVec3 Fm = F * m;
	PxVec3 cofm = cof * m;

	PxReal V0 = tet.restVolume;

	// Inversion protection: clamp J to a small positive value so that
	// fully inverted tets produce bounded restoration forces instead of
	// catastrophic blowup.  The force direction remains correct (cofactor
	// still points toward un-inverting the tet).
	const PxReal Jmin = 0.05f;
	PxReal Jsafe = PxMax(J, Jmin);

	outForce = (Fm * mu + cofm * (lam * (Jsafe - alpha))) * (-V0);

	PxReal m2 = m.dot(m);
	outHessian = PxMat33::createDiagonal(PxVec3(mu * m2 * V0))
	           + avbdOuter(cofm, cofm) * (lam * V0);

	// Extra diagonal regularization for severely compressed / inverted tets
	// to keep the Hessian well-conditioned.
	if (J < 0.5f)
	{
		PxReal reg = (0.5f - J) * lam * V0 * m2;
		outHessian = outHessian + PxMat33::createDiagonal(PxVec3(reg));
	}
}

PX_FORCE_INLINE void avbdEvaluateBendingForceHessian(
	const AvbdBendingElement& be, int vOrder,
	PxReal stiffness,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal eps = 1e-6f;

	PxVec3 x0 = particles[be.opp0].position;
	PxVec3 x1 = particles[be.opp1].position;
	PxVec3 x2 = particles[be.edgeStart].position;
	PxVec3 x3 = particles[be.edgeEnd].position;

	PxVec3 e = x3 - x2;
	PxVec3 x02 = x2 - x0, x03 = x3 - x0;
	PxVec3 x13 = x3 - x1, x12 = x2 - x1;

	PxVec3 n1 = x02.cross(x03);
	PxVec3 n2 = x13.cross(x12);

	PxReal n1Norm = n1.magnitude();
	PxReal n2Norm = n2.magnitude();
	PxReal eNorm = e.magnitude();

	if (n1Norm < eps || n2Norm < eps || eNorm < eps)
	{
		outForce = PxVec3(0.0f);
		outHessian = PxMat33(PxZero);
		return;
	}

	PxVec3 n1Hat = n1 * (1.0f / n1Norm);
	PxVec3 n2Hat = n2 * (1.0f / n2Norm);
	PxVec3 eHat = e * (1.0f / eNorm);

	PxReal sinTheta = n1Hat.cross(n2Hat).dot(eHat);
	PxReal cosTheta = PxClamp(n1Hat.dot(n2Hat), -1.0f, 1.0f);
	PxReal theta = PxAtan2(sinTheta, cosTheta);

	PxReal k = stiffness * be.restLength;
	PxReal dE_dtheta = k * (theta - be.restAngle);

	auto normalizedDerivative = [](PxReal unnormLen, const PxVec3& nHat,
	                                const PxMat33& dNdx) -> PxMat33 {
		PxMat33 P = PxMat33(PxIdentity) - avbdOuter(nHat, nHat);
		return (P * dNdx) * (1.0f / unnormLen);
	};

	auto angleDerivative = [](const PxVec3& n1h, const PxVec3& n2h, const PxVec3& eh,
	                          const PxMat33& dn1dx, const PxMat33& dn2dx,
	                          PxReal sinT, PxReal cosT,
	                          const PxMat33& skN1, const PxMat33& skN2) -> PxVec3 {
		PxMat33 dSinMat = skN1 * dn2dx - skN2 * dn1dx;
		PxVec3 dSin = dSinMat.getTranspose() * eh;
		PxVec3 dCos = dn1dx.getTranspose() * n2h + dn2dx.getTranspose() * n1h;
		return dSin * cosT - dCos * sinT;
	};

	PxMat33 skE = avbdSkew(e);
	PxMat33 skX03 = avbdSkew(x03);
	PxMat33 skX02 = avbdSkew(x02);
	PxMat33 skX13 = avbdSkew(x13);
	PxMat33 skX12 = avbdSkew(x12);
	PxMat33 skN1 = avbdSkew(n1Hat);
	PxMat33 skN2 = avbdSkew(n2Hat);

	PxMat33 dn1hat_dx0 = normalizedDerivative(n1Norm, n1Hat, skE);
	PxMat33 dn1hat_dx1(PxZero);
	PxMat33 dn1hat_dx2 = normalizedDerivative(n1Norm, n1Hat, skX03 * (-1.0f));
	PxMat33 dn1hat_dx3 = normalizedDerivative(n1Norm, n1Hat, skX02);

	PxMat33 dn2hat_dx0(PxZero);
	PxMat33 dn2hat_dx1 = normalizedDerivative(n2Norm, n2Hat, skE * (-1.0f));
	PxMat33 dn2hat_dx2 = normalizedDerivative(n2Norm, n2Hat, skX13);
	PxMat33 dn2hat_dx3 = normalizedDerivative(n2Norm, n2Hat, skX12 * (-1.0f));

	PxVec3 dtheta_dx0 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx0, dn2hat_dx0,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx1 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx1, dn2hat_dx1,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx2 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx2, dn2hat_dx2,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx3 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx3, dn2hat_dx3,
	                                     sinTheta, cosTheta, skN1, skN2);

	PxVec3 dtheta_dx;
	switch (vOrder)
	{
		case 0: dtheta_dx = dtheta_dx0; break;
		case 1: dtheta_dx = dtheta_dx1; break;
		case 2: dtheta_dx = dtheta_dx2; break;
		case 3: dtheta_dx = dtheta_dx3; break;
		default:
			outForce = PxVec3(0.0f);
			outHessian = PxMat33(PxZero);
			return;
	}

	outForce = dtheta_dx * (-dE_dtheta);
	outHessian = avbdOuter(dtheta_dx, dtheta_dx) * k;
}

// =============================================================================
// AVBD contact/pin evaluators (penalty only)
// =============================================================================

PX_FORCE_INLINE void avbdEvaluateContactForceHessian(
	const AvbdSoftContact& sc,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	outForce = PxVec3(0.0f);
	outHessian = PxMat33(PxZero);

	const AvbdSoftParticle& sp = particles[sc.particleIdx];
	PxVec3 n = sc.normal;

	PxReal penetration;
	if (sc.rigidBodyIdx == PX_MAX_U32)
		penetration = -(sp.position.dot(n));         // ground: analytical
	else
		penetration = sc.margin - (sp.position - sc.surfacePoint).dot(n);  // proximity shell

	if (penetration <= 0.0f) return;

	PxReal fn = sc.k * penetration;
	outForce = n * fn;
	outHessian = avbdOuter(n, n) * sc.k;

	// NOTE: friction is applied as explicit velocity/position correction
	// after the VBD inner loop (see avbdApplyExplicitFriction), NOT here.
	// Including it in the Hessian creates enormous tangential stiffness
	// near zero slip that locks particles and prevents macroscopic rotation.
}

PX_FORCE_INLINE void avbdEvaluatePinForceHessian(
	const AvbdKinematicPin& kp,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	PxVec3 C = particles[kp.particleIdx].position - kp.worldTarget;
	outForce = C * (-kp.k);
	outHessian = PxMat33::createDiagonal(PxVec3(kp.k));
}

// =============================================================================
// AVBD Dual updates
// =============================================================================

PX_FORCE_INLINE void avbdUpdatePinDual(
	AvbdKinematicPin& kp,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	PxReal C_lin = (particles[kp.particleIdx].position - kp.worldTarget).magnitude();
	kp.k = PxMin(kp.k + beta * C_lin, kp.kMax);
}

PX_FORCE_INLINE void avbdUpdateSoftContactDual(
	AvbdSoftContact& sc,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	PxVec3 n = sc.normal;
	PxReal penetration;
	if (sc.rigidBodyIdx == PX_MAX_U32)
		penetration = -(particles[sc.particleIdx].position.dot(n));
	else
		penetration = sc.margin - (particles[sc.particleIdx].position - sc.surfacePoint).dot(n);
	penetration = PxMax(0.0f, penetration);
	sc.k = PxMin(sc.k + beta * penetration, sc.ke);
}

// =============================================================================
// Ground contact detection
// =============================================================================

inline void avbdDetectSoftGroundContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxArray<AvbdSoftContact>& contacts,
	PxReal groundY = 0.0f, PxReal margin = 0.02f, PxReal friction = 0.5f)
{
	contacts.clear();
	PxVec3 normal(0.0f, 1.0f, 0.0f);
	PxVec3 t1(1.0f, 0.0f, 0.0f), t2(0.0f, 0.0f, 1.0f);

	for (PxU32 i = 0; i < numParticles; i++)
	{
		if (particles[i].invMass <= 0.0f) continue;
		PxReal dist = particles[i].position.y - groundY;
		if (dist < margin)
		{
			AvbdSoftContact sc;
			sc.particleIdx = i;
			sc.rigidBodyIdx = PX_MAX_U32;
			sc.normal = normal;
			sc.projNormal = normal;
			sc.depth = PxMax(0.0f, -dist);
			sc.friction = friction;
			sc.tangent1 = t1;
			sc.tangent2 = t2;
			contacts.pushBack(sc);
		}
	}
}

// =============================================================================
// Closest point on triangle (Barycentric projection)
// =============================================================================

inline PxVec3 avbdClosestPointOnTriangle(
	const PxVec3& p, const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	PxVec3 ab = b - a, ac = c - a, ap = p - a;
	PxReal d1 = ab.dot(ap), d2 = ac.dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a;

	PxVec3 bp = p - b;
	PxReal d3 = ab.dot(bp), d4 = ac.dot(bp);
	if (d3 >= 0.0f && d4 <= d3) return b;

	PxReal vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		PxReal v = d1 / (d1 - d3);
		return a + ab * v;
	}

	PxVec3 cp = p - c;
	PxReal d5 = ab.dot(cp), d6 = ac.dot(cp);
	if (d6 >= 0.0f && d5 <= d6) return c;

	PxReal vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		PxReal w = d2 / (d2 - d6);
		return a + ac * w;
	}

	PxReal va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
	{
		PxReal w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + (c - b) * w;
	}

	PxReal denom = 1.0f / (va + vb + vc);
	PxReal v = vb * denom, w = vc * denom;
	return a + ab * v + ac * w;
}

// =============================================================================
// Rigid box descriptor for soft-rigid collision
// =============================================================================

struct AvbdRigidBox
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 halfExtent;
	PxReal friction;

	AvbdRigidBox()
		: center(0.0f), rotation(PxIdentity), halfExtent(0.0f), friction(0.5f) {}
};

// =============================================================================
// Soft-soft collision detection (vertex-to-surface proximity)
// =============================================================================

inline void avbdDetectSoftSoftContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.2f, PxReal friction = 0.5f,
	PxReal k0 = 1e5f, PxReal ke0 = 1e7f,
	PxReal forceMargin = -1.0f)
{
	PX_UNUSED(numParticles);
	// If forceMargin not specified, default to margin (backward-compatible).
	// forceMargin controls the equilibrium gap; margin controls detection range.
	if (forceMargin < 0.0f) forceMargin = margin;

	for (PxU32 sA = 0; sA < numSoftBodies; sA++)
	{
		for (PxU32 sB = sA + 1; sB < numSoftBodies; sB++)
		{
			const AvbdSoftBody& bodyA = softBodies[sA];
			const AvbdSoftBody& bodyB = softBodies[sB];

			// AABB broadphase per body pair
			PxVec3 minA(PX_MAX_F32), maxA(-PX_MAX_F32);
			for (PxU32 i = 0; i < bodyA.particleCount; i++)
			{
				const PxVec3& p = particles[bodyA.particleStart + i].position;
				minA = minA.minimum(p);
				maxA = maxA.maximum(p);
			}
			PxVec3 minB(PX_MAX_F32), maxB(-PX_MAX_F32);
			for (PxU32 i = 0; i < bodyB.particleCount; i++)
			{
				const PxVec3& p = particles[bodyB.particleStart + i].position;
				minB = minB.minimum(p);
				maxB = maxB.maximum(p);
			}
			if (minA.x > maxB.x + margin || maxA.x < minB.x - margin ||
				minA.y > maxB.y + margin || maxA.y < minB.y - margin ||
				minA.z > maxB.z + margin || maxA.z < minB.z - margin)
				continue;

			// Helper lambda: test particles of one body against surface of another
			auto testParticlesVsSurface = [&](
				const AvbdSoftBody& testBody, const AvbdSoftBody& surfBody,
				PxU32 surfBodyIdx)
			{
				// Pre-compute per-triangle AABBs for early rejection
				PxU32 numTris = surfBody.surfaceTriangles.size() / 3;
				PxArray<PxVec3> triMin(numTris), triMax(numTris);
				for (PxU32 t = 0; t < numTris; t++)
				{
					const PxVec3& va = particles[surfBody.surfaceTriangles[t*3]].position;
					const PxVec3& vb = particles[surfBody.surfaceTriangles[t*3+1]].position;
					const PxVec3& vc = particles[surfBody.surfaceTriangles[t*3+2]].position;
					triMin[t] = va.minimum(vb).minimum(vc) - PxVec3(margin);
					triMax[t] = va.maximum(vb).maximum(vc) + PxVec3(margin);
				}

				for (PxU32 li = 0; li < testBody.particleCount; li++)
				{
					PxU32 pi = testBody.particleStart + li;
					if (particles[pi].invMass <= 0.0f) continue;
					const PxVec3& pp = particles[pi].position;

					PxReal bestDist2 = PX_MAX_F32;
					PxVec3 bestClosest(0.0f);
					PxVec3 bestTriNorm(0.0f, 1.0f, 0.0f);

					for (PxU32 t = 0; t < numTris; t++)
					{
						// AABB early rejection
						if (pp.x < triMin[t].x || pp.x > triMax[t].x ||
							pp.y < triMin[t].y || pp.y > triMax[t].y ||
							pp.z < triMin[t].z || pp.z > triMax[t].z)
							continue;

						const PxVec3& va = particles[surfBody.surfaceTriangles[t*3]].position;
						const PxVec3& vb = particles[surfBody.surfaceTriangles[t*3+1]].position;
						const PxVec3& vc = particles[surfBody.surfaceTriangles[t*3+2]].position;

						PxVec3 closest = avbdClosestPointOnTriangle(pp, va, vb, vc);
						PxReal d2 = (closest - pp).magnitudeSquared();
						if (d2 < bestDist2)
						{
							bestDist2 = d2;
							bestClosest = closest;
							bestTriNorm = (vb - va).cross(vc - va);
						}
					}

					PxReal bestDist = PxSqrt(bestDist2);
					if (bestDist > margin) continue;

					// Two normals per contact:
					// - penalty normal (closest-point direction): stable in
					//   the VBD Newton solve but flips inward when the
					//   particle is behind the surface.
					// - projection normal (face-normal corrected): always
					//   points outward from the surface so projection pushes
					//   penetrated particles in the right direction.
					// When the particle is inside, disable penalty (k=0) to
					// prevent enormous inward forces; projection handles it.
					PxVec3 penN;   // penalty normal
					PxVec3 projN;  // projection normal
					bool isInside = false;
					PxReal tnLen = bestTriNorm.magnitude();
					if (bestDist > 1e-6f)
					{
						penN = (pp - bestClosest) * (1.0f / bestDist);
						projN = penN;
						// If particle is behind the closest triangle face
						// (inside the mesh), correct projN to face outward.
						if (tnLen > 1e-10f && penN.dot(bestTriNorm) < 0.0f)
						{
							projN = bestTriNorm * (1.0f / tnLen);
							isInside = true;
						}
					}
					else
					{
						if (tnLen < 1e-10f) continue;
						penN = bestTriNorm * (1.0f / tnLen);
						projN = penN;
					}

					AvbdSoftContact sc;
					sc.particleIdx = pi;
					sc.rigidBodyIdx = surfBodyIdx;
					sc.normal = penN;
					sc.projNormal = projN;
					sc.depth = PxMax(0.0f, forceMargin - bestDist);
					sc.margin = forceMargin;
					sc.surfacePoint = bestClosest;
					sc.friction = friction;
					sc.k = isInside ? 0.0f : k0;
					sc.ke = isInside ? 0.0f : ke0;
					if (PxAbs(penN.x) < 0.9f)
						sc.tangent1 = penN.cross(PxVec3(1.0f, 0.0f, 0.0f)).getNormalized();
					else
						sc.tangent1 = penN.cross(PxVec3(0.0f, 1.0f, 0.0f)).getNormalized();
					sc.tangent2 = penN.cross(sc.tangent1);
					contacts.pushBack(sc);
				}
			};

			// Test particles of A against surface of B
			testParticlesVsSurface(bodyA, bodyB, sB);
			// Test particles of B against surface of A (symmetric)
			testParticlesVsSurface(bodyB, bodyA, sA);
		}
	}
}

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

struct AvbdOGCParams
{
	PxReal contactRadius;     // r: offset radius
	PxReal contactStiffness;  // k_c: contact stiffness
	PxReal friction;          // mu_c
	PxReal safetyRelax;       // gamma_p: safety bound relaxation (0 < gamma_p < 0.5)
	PxReal redetectRatio;     // gamma_e: redetection trigger ratio
	PxReal tau;               // activation threshold; -1 means r/2 (auto)

	AvbdOGCParams()
		: contactRadius(0.05f)
		, contactStiffness(1e5f)
		, friction(0.3f)
		, safetyRelax(0.45f)
		, redetectRatio(0.01f)
		, tau(-1.0f) {}

	PxReal getTau() const { return (tau < 0.0f) ? contactRadius * 0.5f : tau; }
};

// Closest-point feature type for OGC block classification
enum AvbdClosestFeature { AVBD_FEATURE_FACE, AVBD_FEATURE_EDGE, AVBD_FEATURE_VERTEX };

struct AvbdClosestPointResult
{
	PxVec3              point;     // closest point on triangle
	PxVec3              normal;    // direction from closest point to query point
	PxReal              distance;  // unsigned distance
	AvbdClosestFeature  feature;
};

// Enhanced closest-point-on-triangle with feature classification
inline AvbdClosestPointResult avbdClosestPointOnTriangleOGC(
	const PxVec3& p, const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	AvbdClosestPointResult result;
	PxVec3 ab = b - a, ac = c - a, ap = p - a;
	PxReal d1 = ab.dot(ap), d2 = ac.dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		result.point = a; result.feature = AVBD_FEATURE_VERTEX;
		PxVec3 diff = p - a; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxVec3 bp = p - b;
	PxReal d3 = ab.dot(bp), d4 = ac.dot(bp);
	if (d3 >= 0.0f && d4 <= d3) {
		result.point = b; result.feature = AVBD_FEATURE_VERTEX;
		PxVec3 diff = p - b; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxReal vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		PxReal v = d1 / (d1 - d3);
		result.point = a + ab * v; result.feature = AVBD_FEATURE_EDGE;
		PxVec3 diff = p - result.point; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxVec3 cp = p - c;
	PxReal d5 = ab.dot(cp), d6 = ac.dot(cp);
	if (d6 >= 0.0f && d5 <= d6) {
		result.point = c; result.feature = AVBD_FEATURE_VERTEX;
		PxVec3 diff = p - c; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxReal vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		PxReal w = d2 / (d2 - d6);
		result.point = a + ac * w; result.feature = AVBD_FEATURE_EDGE;
		PxVec3 diff = p - result.point; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxReal va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		PxReal w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		result.point = b + (c - b) * w; result.feature = AVBD_FEATURE_EDGE;
		PxVec3 diff = p - result.point; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	// Inside triangle
	PxReal denom = 1.0f / (va + vb + vc);
	PxReal v = vb * denom;
	PxReal w = vc * denom;
	result.point = a + ab * v + ac * w;
	result.feature = AVBD_FEATURE_FACE;
	PxVec3 diff = p - result.point;
	result.distance = diff.magnitude();
	if (result.distance > 1e-10f)
		result.normal = diff * (1.0f / result.distance);
	else {
		PxVec3 faceN = ab.cross(ac);
		PxReal fLen = faceN.magnitude();
		result.normal = fLen > 1e-10f ? faceN * (1.0f / fLen) : PxVec3(0,1,0);
	}
	return result;
}

// =============================================================================
// Two-stage C2 activation function (OGC Eq. 18-20)
// =============================================================================

struct AvbdActivationResult
{
	PxReal energy;
	PxReal force;    // -dg/dd (positive = repulsive)
	PxReal hessian;  // d2g/dd2
};

PX_FORCE_INLINE AvbdActivationResult avbdOGCActivationQuadratic(PxReal d, PxReal r, PxReal kc)
{
	PxReal pen = r - d;
	AvbdActivationResult res;
	res.energy  = 0.5f * kc * pen * pen;
	res.force   = kc * pen;
	res.hessian = kc;
	return res;
}

PX_FORCE_INLINE AvbdActivationResult avbdOGCActivationFull(PxReal d, PxReal r, PxReal kc, PxReal tau)
{
	AvbdActivationResult res;
	if (d >= r) {
		res.energy = 0.0f; res.force = 0.0f; res.hessian = 0.0f;
	} else if (d >= tau) {
		PxReal pen = r - d;
		res.energy  = 0.5f * kc * pen * pen;
		res.force   = kc * pen;
		res.hessian = kc;
	} else if (d > 1e-10f) {
		PxReal rmt = r - tau;
		PxReal kc_prime = tau * kc * rmt * rmt;
		PxReal b = 0.5f * kc * rmt * rmt + kc_prime * PxLog(tau);
		res.energy  = -kc_prime * PxLog(d) + b;
		res.force   = kc_prime / d;
		res.hessian = kc_prime / (d * d);
	} else {
		PxReal rmt = r - tau;
		PxReal kc_prime = tau * kc * rmt * rmt;
		PxReal d_clamp = 1e-10f;
		res.energy  = kc_prime * 10.0f;
		res.force   = kc_prime / d_clamp;
		res.hessian = kc_prime / (d_clamp * d_clamp);
	}
	return res;
}

// =============================================================================
// Helper: point-inside-tet-mesh via Moller-Trumbore ray casting (parity)
// =============================================================================

inline bool avbdIsPointInsideTetMesh(
	const PxVec3& point,
	const PxArray<PxU32>& surfaceTriangles,
	const AvbdSoftParticle* particles)
{
	int crossings = 0;
	PxVec3 rayDir(0.0f, 1.0f, 0.0f);
	for (PxU32 ti = 0; ti + 2 < surfaceTriangles.size(); ti += 3)
	{
		const PxVec3& a = particles[surfaceTriangles[ti]].position;
		const PxVec3& b = particles[surfaceTriangles[ti+1]].position;
		const PxVec3& c = particles[surfaceTriangles[ti+2]].position;
		PxVec3 e1 = b - a, e2 = c - a;
		PxVec3 h = rayDir.cross(e2);
		PxReal det = e1.dot(h);
		if (PxAbs(det) < 1e-10f) continue;
		PxReal invDet = 1.0f / det;
		PxVec3 s = point - a;
		PxReal u = invDet * s.dot(h);
		if (u < 0.0f || u > 1.0f) continue;
		PxVec3 q = s.cross(e1);
		PxReal v = invDet * rayDir.dot(q);
		if (v < 0.0f || u + v > 1.0f) continue;
		PxReal t = invDet * e2.dot(q);
		if (t > 1e-6f) crossings++;
	}
	return (crossings & 1) != 0;
}

// =============================================================================
// PATH 2 (OGC): Analytical SDF Rigid-Soft Contact
// =============================================================================

inline void avbdDetectSoftRigidSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	for (PxU32 pi = 0; pi < numParticles; pi++)
	{
		if (particles[pi].invMass <= 0.0f) continue;
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

			if (inside) {
				sdf = PxMax(q.x, PxMax(q.y, q.z));
				if (q.x > q.y && q.x > q.z)
					localNormal = PxVec3(localP.x > 0 ? 1.0f : -1.0f, 0, 0);
				else if (q.y > q.z)
					localNormal = PxVec3(0, localP.y > 0 ? 1.0f : -1.0f, 0);
				else
					localNormal = PxVec3(0, 0, localP.z > 0 ? 1.0f : -1.0f);
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

			AvbdSoftContact sc;
			sc.particleIdx  = pi;
			sc.rigidBodyIdx = PX_MAX_U32 - 1 - bi;
			sc.normal       = worldNormal;
			sc.projNormal   = worldNormal;
			sc.depth        = depth;
			sc.margin       = margin;
			sc.surfacePoint = worldSurf;
			sc.friction     = box.friction;
			sc.k            = 1e5f;
			sc.ke           = 1e6f;
			if (PxAbs(worldNormal.x) < 0.9f)
				sc.tangent1 = worldNormal.cross(PxVec3(1.0f, 0.0f, 0.0f)).getNormalized();
			else
				sc.tangent1 = worldNormal.cross(PxVec3(0.0f, 1.0f, 0.0f)).getNormalized();
			sc.tangent2 = worldNormal.cross(sc.tangent1);
			contacts.pushBack(sc);
		}
	}
}

// =============================================================================
// PATH 3 (OGC): Simplified Soft-Soft Contact (Sec 3.9)
//
// Outward-only offset, pure quadratic energy, DCD for penetration.
// =============================================================================

inline void avbdDetectSoftSoftOGC(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams())
{
	PX_UNUSED(numParticles);
	PxReal r = params.contactRadius;

	for (PxU32 sA = 0; sA < numSoftBodies; sA++)
	{
		for (PxU32 sB = sA + 1; sB < numSoftBodies; sB++)
		{
			const AvbdSoftBody& bodyA = softBodies[sA];
			const AvbdSoftBody& bodyB = softBodies[sB];

			// AABB broadphase per body pair
			PxVec3 minA(PX_MAX_F32), maxA(-PX_MAX_F32);
			for (PxU32 i = 0; i < bodyA.particleCount; i++) {
				const PxVec3& p = particles[bodyA.particleStart + i].position;
				minA = minA.minimum(p); maxA = maxA.maximum(p);
			}
			PxVec3 minB(PX_MAX_F32), maxB(-PX_MAX_F32);
			for (PxU32 i = 0; i < bodyB.particleCount; i++) {
				const PxVec3& p = particles[bodyB.particleStart + i].position;
				minB = minB.minimum(p); maxB = maxB.maximum(p);
			}
			if (minA.x > maxB.x + r || maxA.x < minB.x - r ||
				minA.y > maxB.y + r || maxA.y < minB.y - r ||
				minA.z > maxB.z + r || maxA.z < minB.z - r)
				continue;

			// Lambda: test particles of testBody against surface of surfBody
			auto testParticlesVsSurface = [&](
				const AvbdSoftBody& testBody, const AvbdSoftBody& surfBody,
				PxU32 surfBodyIdx,
				const PxVec3& aabbLo, const PxVec3& aabbHi)
			{
				for (PxU32 li = 0; li < testBody.particleCount; li++)
				{
					PxU32 pi = testBody.particleStart + li;
					if (particles[pi].invMass <= 0.0f) continue;
					const PxVec3& pp = particles[pi].position;

					// Per-particle AABB cull
					if (pp.x < aabbLo.x - r || pp.x > aabbHi.x + r ||
						pp.y < aabbLo.y - r || pp.y > aabbHi.y + r ||
						pp.z < aabbLo.z - r || pp.z > aabbHi.z + r)
						continue;

					// DCD: check if particle is inside the other body
					bool isInside = avbdIsPointInsideTetMesh(pp, surfBody.surfaceTriangles, particles);
					if (isInside)
					{
						// Find closest surface triangle for direction
						PxReal minDist = PX_MAX_F32;
						PxVec3 bestNormal(0.0f, 1.0f, 0.0f);
						PxVec3 bestClosest(0.0f);
						for (PxU32 ti = 0; ti + 2 < surfBody.surfaceTriangles.size(); ti += 3)
						{
							const PxVec3& va = particles[surfBody.surfaceTriangles[ti]].position;
							const PxVec3& vb = particles[surfBody.surfaceTriangles[ti+1]].position;
							const PxVec3& vc = particles[surfBody.surfaceTriangles[ti+2]].position;
							AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
							if (cp.distance < minDist) {
								minDist = cp.distance;
								bestClosest = cp.point;
								PxVec3 faceN = (vb - va).cross(vc - va);
								PxReal fLen = faceN.magnitude();
								bestNormal = fLen > 1e-10f ? faceN * (1.0f / fLen) : cp.normal;
							}
						}

						PxReal depth = minDist + r;
						AvbdSoftContact sc;
						sc.particleIdx  = pi;
						sc.rigidBodyIdx = surfBodyIdx;
						sc.normal       = bestNormal;
						sc.projNormal   = bestNormal;
						sc.depth        = depth;
						sc.margin       = r;
						sc.surfacePoint = bestClosest;
						sc.friction     = params.friction;
						sc.k  = params.contactStiffness;
						sc.ke = params.contactStiffness * 10.0f;
						if (PxAbs(bestNormal.x) < 0.9f)
							sc.tangent1 = bestNormal.cross(PxVec3(1.0f,0.0f,0.0f)).getNormalized();
						else
							sc.tangent1 = bestNormal.cross(PxVec3(0.0f,1.0f,0.0f)).getNormalized();
						sc.tangent2 = bestNormal.cross(sc.tangent1);
						contacts.pushBack(sc);
						continue;
					}

					// Not inside: OGC outward offset blocks on surface
					for (PxU32 ti = 0; ti + 2 < surfBody.surfaceTriangles.size(); ti += 3)
					{
						const PxVec3& va = particles[surfBody.surfaceTriangles[ti]].position;
						const PxVec3& vb = particles[surfBody.surfaceTriangles[ti+1]].position;
						const PxVec3& vc = particles[surfBody.surfaceTriangles[ti+2]].position;

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

						PxReal depth = r - cp.distance;
						AvbdSoftContact sc;
						sc.particleIdx  = pi;
						sc.rigidBodyIdx = surfBodyIdx;
						sc.normal       = contactNormal;
						sc.projNormal   = contactNormal;
						sc.depth        = depth;
						sc.margin       = r;
						sc.surfacePoint = cp.point;
						sc.friction     = params.friction;
						sc.k  = params.contactStiffness;
						sc.ke = params.contactStiffness * 10.0f;
						if (PxAbs(contactNormal.x) < 0.9f)
							sc.tangent1 = contactNormal.cross(PxVec3(1.0f,0.0f,0.0f)).getNormalized();
						else
							sc.tangent1 = contactNormal.cross(PxVec3(0.0f,1.0f,0.0f)).getNormalized();
						sc.tangent2 = contactNormal.cross(sc.tangent1);
						contacts.pushBack(sc);
					}
				}
			};

			// Test A particles vs B surface, then B particles vs A surface
			testParticlesVsSurface(bodyA, bodyB, sB, minB, maxB);
			testParticlesVsSurface(bodyB, bodyA, sA, minA, maxA);
		}
	}
}

// =============================================================================
// PATH 4 (OGC): Full Self-Collision Detection
//
// Two-stage C2 activation, topological adjacency filtering, safety bubble.
// =============================================================================

// Build topological adjacency for self-collision filtering.
// Returns per-particle sorted list of connected local particle indices.
inline void avbdBuildSelfCollisionAdjacency(
	const AvbdSoftBody& sb,
	PxArray<PxArray<PxU32> >& adj)
{
	adj.resize(sb.particleCount);
	for (PxU32 i = 0; i < sb.particleCount; i++)
		adj[i].clear();

	auto addAdj = [&](PxU32 la, PxU32 lb) {
		adj[la].pushBack(lb);
		adj[lb].pushBack(la);
	};

	for (PxU32 i = 0; i + 3 < sb.tetrahedra.size(); i += 4) {
		PxU32 v[4];
		for (int j = 0; j < 4; j++) v[j] = sb.tetrahedra[i + PxU32(j)];
		for (int a = 0; a < 4; a++)
			for (int b = a + 1; b < 4; b++)
				addAdj(v[a], v[b]);
	}
	for (PxU32 i = 0; i + 2 < sb.triangles.size(); i += 3) {
		PxU32 v[3];
		for (int j = 0; j < 3; j++) v[j] = sb.triangles[i + PxU32(j)];
		for (int a = 0; a < 3; a++)
			for (int b = a + 1; b < 3; b++)
				addAdj(v[a], v[b]);
	}

	// Sort and deduplicate
	for (PxU32 i = 0; i < sb.particleCount; i++) {
		PxArray<PxU32>& a = adj[i];
		if (a.size() > 1) {
			PxSort(a.begin(), a.size());
			PxU32 writeIdx = 1;
			for (PxU32 k = 1; k < a.size(); k++)
				if (a[k] != a[k-1])
					a[writeIdx++] = a[k];
			a.resize(writeIdx);
		}
	}
}

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
inline void avbdComputeSafetyBounds(
	const AvbdSoftBody& sb,
	const AvbdSoftParticle* particles,
	const PxArray<PxArray<PxU32> >& adj,
	PxReal gammaP,
	PxArray<PxReal>& bounds)
{
	bounds.resize(sb.particleCount);
	for (PxU32 i = 0; i < sb.particleCount; i++)
		bounds[i] = PX_MAX_F32;

	for (PxU32 li = 0; li < sb.particleCount; li++)
	{
		PxU32 gi = sb.particleStart + li;
		const PxVec3& pi = particles[gi].position;
		PxReal dMin = PX_MAX_F32;

		for (PxU32 ti = 0; ti + 2 < sb.surfaceTriangles.size(); ti += 3)
		{
			PxU32 lv0 = sb.surfaceTriangles[ti]   - sb.particleStart;
			PxU32 lv1 = sb.surfaceTriangles[ti+1] - sb.particleStart;
			PxU32 lv2 = sb.surfaceTriangles[ti+2] - sb.particleStart;
			if (lv0 == li || lv1 == li || lv2 == li) continue;
			if (avbdIsAdjacentSelfCollision(li, lv0, adj) &&
				avbdIsAdjacentSelfCollision(li, lv1, adj) &&
				avbdIsAdjacentSelfCollision(li, lv2, adj))
				continue;

			AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(
				pi,
				particles[sb.surfaceTriangles[ti]].position,
				particles[sb.surfaceTriangles[ti+1]].position,
				particles[sb.surfaceTriangles[ti+2]].position);
			dMin = PxMin(dMin, cp.distance);
		}

		bounds[li] = gammaP * PxMax(dMin, 1e-6f);
	}
}

// Truncate displacement to safety bound
PX_FORCE_INLINE void avbdTruncateDisplacement(
	AvbdSoftParticle& sp,
	const PxVec3& prevPosition,
	PxReal bound)
{
	PxVec3 disp = sp.position - prevPosition;
	PxReal dispMag = disp.magnitude();
	if (dispMag > bound && dispMag > 1e-10f)
		sp.position = prevPosition + disp * (bound / dispMag);
}

// Detect self-collision contacts within a single soft body
inline void avbdDetectSelfCollisionOGC(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& sb,
	const PxArray<PxArray<PxU32> >& adj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams())
{
	PxReal r   = params.contactRadius;
	PxReal tau = params.getTau();

	for (PxU32 li = 0; li < sb.particleCount; li++)
	{
		PxU32 gi = sb.particleStart + li;
		if (particles[gi].invMass <= 0.0f) continue;
		const PxVec3& pp = particles[gi].position;

		for (PxU32 ti = 0; ti + 2 < sb.surfaceTriangles.size(); ti += 3)
		{
			PxU32 lv0 = sb.surfaceTriangles[ti]   - sb.particleStart;
			PxU32 lv1 = sb.surfaceTriangles[ti+1] - sb.particleStart;
			PxU32 lv2 = sb.surfaceTriangles[ti+2] - sb.particleStart;

			// Skip vertex-on-triangle and topologically adjacent
			if (lv0 == li || lv1 == li || lv2 == li) continue;
			if (avbdIsAdjacentSelfCollision(li, lv0, adj) ||
				avbdIsAdjacentSelfCollision(li, lv1, adj) ||
				avbdIsAdjacentSelfCollision(li, lv2, adj))
				continue;

			const PxVec3& va = particles[sb.surfaceTriangles[ti]].position;
			const PxVec3& vb = particles[sb.surfaceTriangles[ti+1]].position;
			const PxVec3& vc = particles[sb.surfaceTriangles[ti+2]].position;

			AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
			if (cp.distance >= r) continue;

			PxVec3 faceN = (vb - va).cross(vc - va);
			PxReal fLen = faceN.magnitude();
			if (fLen < 1e-10f) continue;
			faceN = faceN * (1.0f / fLen);

			PxVec3 contactNormal;
			if (cp.feature == AVBD_FEATURE_FACE)
				contactNormal = (cp.normal.dot(faceN) >= 0.0f) ? faceN : faceN * (-1.0f);
			else
				contactNormal = cp.normal;

			// Two-stage activation: use force > 0 as the activation gate,
			// but rely on the existing AVBD penalty evaluator for force computation.
			AvbdActivationResult act = avbdOGCActivationFull(cp.distance, r, params.contactStiffness, tau);
			if (act.force <= 0.0f) continue;

			PxReal depth = r - cp.distance;
			AvbdSoftContact sc;
			sc.particleIdx  = gi;
			sc.rigidBodyIdx = PX_MAX_U32;
			sc.normal       = contactNormal;
			sc.projNormal   = contactNormal;
			sc.depth        = depth;
			sc.margin       = r;
			sc.surfacePoint = cp.point;
			sc.friction     = params.friction;
			sc.k  = params.contactStiffness;
			sc.ke = params.contactStiffness * 10.0f;
			if (PxAbs(contactNormal.x) < 0.9f)
				sc.tangent1 = contactNormal.cross(PxVec3(1.0f,0.0f,0.0f)).getNormalized();
			else
				sc.tangent1 = contactNormal.cross(PxVec3(0.0f,1.0f,0.0f)).getNormalized();
			sc.tangent2 = contactNormal.cross(sc.tangent1);
			contacts.pushBack(sc);
		}
	}
}

// =============================================================================
// Convenience: detect all OGC contacts (ground + soft-rigid + soft-soft + self)
// =============================================================================

// Per-body self-collision adjacency array type
typedef PxArray<PxArray<PxU32> > AvbdSelfCollisionAdjacency;

inline void avbdDetectAllOGCContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidBox* rigidBoxes, PxU32 numRigidBoxes,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	PxReal groundY = 0.0f)
{
	contacts.clear();

	// Ground
	avbdDetectSoftGroundContacts(particles, numParticles,
	                             contacts, groundY, params.contactRadius, params.friction);

	// Path 2: Rigid-soft SDF
	if (numRigidBoxes > 0)
		avbdDetectSoftRigidSDF(particles, numParticles,
		                       rigidBoxes, numRigidBoxes,
		                       contacts, params.contactRadius);

	// Path 3: Soft-soft OGC simplified
	if (numSoftBodies > 1)
		avbdDetectSoftSoftOGC(particles, numParticles,
		                      softBodies, numSoftBodies,
		                      contacts, params);

	// Path 4: Self-collision OGC full
	for (PxU32 si = 0; si < numSoftBodies; si++)
	{
		if (si < numAdj && perBodyAdj)
			avbdDetectSelfCollisionOGC(particles, softBodies[si],
			                           perBodyAdj[si], contacts, params);
	}
}

// Build all per-body self-collision adjacencies
inline void avbdBuildAllSelfCollisionAdjacencies(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSelfCollisionAdjacency>& outAdj)
{
	outAdj.resize(numSoftBodies);
	for (PxU32 si = 0; si < numSoftBodies; si++)
		avbdBuildSelfCollisionAdjacency(softBodies[si], outAdj[si]);
}

// =============================================================================
// Mesh generators
// =============================================================================

inline void avbdGenerateCubeTets(
	PxVec3 center, PxReal halfSize,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	PxReal h = halfSize;
	outVerts.clear();
	outVerts.pushBack(center + PxVec3(-h, -h, -h));
	outVerts.pushBack(center + PxVec3( h, -h, -h));
	outVerts.pushBack(center + PxVec3( h,  h, -h));
	outVerts.pushBack(center + PxVec3(-h,  h, -h));
	outVerts.pushBack(center + PxVec3(-h, -h,  h));
	outVerts.pushBack(center + PxVec3( h, -h,  h));
	outVerts.pushBack(center + PxVec3( h,  h,  h));
	outVerts.pushBack(center + PxVec3(-h,  h,  h));

	outTets.clear();
	PxU32 tets[] = { 0,1,3,4, 1,2,3,6, 3,4,6,7, 1,4,5,6, 1,3,4,6 };
	for (PxU32 i = 0; i < 20; i++)
		outTets.pushBack(tets[i]);
}

inline void avbdGenerateSubdividedCubeTets(
	PxVec3 center, PxReal halfSize, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	outVerts.clear();
	outTets.clear();
	PxReal cellSize = 2.0f * halfSize / PxReal(N);
	PxVec3 origin = center - PxVec3(halfSize, halfSize, halfSize);

	for (int iz = 0; iz <= N; iz++)
		for (int iy = 0; iy <= N; iy++)
			for (int ix = 0; ix <= N; ix++)
				outVerts.pushBack(origin + PxVec3(PxReal(ix) * cellSize,
				                                  PxReal(iy) * cellSize,
				                                  PxReal(iz) * cellSize));

	for (int iz = 0; iz < N; iz++)
		for (int iy = 0; iy < N; iy++)
			for (int ix = 0; ix < N; ix++)
			{
				PxU32 v[8];
				v[0] = PxU32(iz * (N+1) * (N+1) + iy * (N+1) + ix);
				v[1] = v[0] + 1;
				v[2] = v[0] + PxU32(N+1) + 1;
				v[3] = v[0] + PxU32(N+1);
				v[4] = v[0] + PxU32((N+1) * (N+1));
				v[5] = v[4] + 1;
				v[6] = v[4] + PxU32(N+1) + 1;
				v[7] = v[4] + PxU32(N+1);

				PxU32 t[] = {
					v[0],v[1],v[3],v[4], v[1],v[2],v[3],v[6],
					v[3],v[4],v[6],v[7], v[1],v[4],v[5],v[6],
					v[1],v[3],v[4],v[6]
				};
				for (PxU32 i = 0; i < 20; i++)
					outTets.pushBack(t[i]);
			}
}

inline void avbdGenerateClothGrid(
	PxVec3 center, PxReal sizeX, PxReal sizeZ,
	int M, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTris)
{
	outVerts.clear();
	outTris.clear();
	PxReal dx = sizeX / PxReal(M - 1);
	PxReal dz = sizeZ / PxReal(N - 1);
	PxVec3 origin = center - PxVec3(sizeX * 0.5f, 0.0f, sizeZ * 0.5f);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < M; i++)
			outVerts.pushBack(origin + PxVec3(PxReal(i) * dx, 0.0f, PxReal(j) * dz));

	for (int j = 0; j < N - 1; j++)
		for (int i = 0; i < M - 1; i++)
		{
			PxU32 v00 = PxU32(j * M + i);
			PxU32 v10 = v00 + 1;
			PxU32 v01 = v00 + PxU32(M);
			PxU32 v11 = v01 + 1;
			outTris.pushBack(v00); outTris.pushBack(v10); outTris.pushBack(v01);
			outTris.pushBack(v10); outTris.pushBack(v11); outTris.pushBack(v01);
		}
}

inline void avbdGenerateSubdividedSphereTets(
	PxVec3 center, PxReal radius, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	// Generate a subdivided cube, then map vertices proportionally onto a sphere.
	// Each vertex keeps its fractional distance from center to cube surface,
	// but the direction is spherically normalized.  This avoids collapsing
	// multiple interior vertices onto the same surface point (which would
	// create degenerate zero-volume tetrahedra).
	avbdGenerateSubdividedCubeTets(center, radius, N, outVerts, outTets);

	for (PxU32 i = 0; i < outVerts.size(); i++)
	{
		PxVec3 d = outVerts[i] - center;
		PxReal len = d.magnitude();
		if (len > 1e-8f)
		{
			// Distance from center to cube surface in direction d:
			//   cubeSurfR = halfSize * len / max(|dx|,|dy|,|dz|)
			// Fraction of the way from center to cube surface:
			//   frac = len / cubeSurfR = max(|dx|,|dy|,|dz|) / halfSize
			// Map to sphere: new distance = frac * radius
			PxReal maxAbs = PxMax(PxAbs(d.x), PxMax(PxAbs(d.y), PxAbs(d.z)));
			PxReal frac = maxAbs / radius;  // 0 at center, 1 at cube face
			outVerts[i] = center + d * (1.0f / len) * (frac * radius);
		}
	}

	// Fix tet orientation after the non-linear mapping (some tets may invert)
	for (PxU32 t = 0; t + 3 < outTets.size(); t += 4)
	{
		PxVec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
		PxVec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
		PxVec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
		if (e1.dot(e2.cross(e3)) < 0.0f)
		{
			PxU32 tmp = outTets[t+1]; outTets[t+1] = outTets[t+2]; outTets[t+2] = tmp;
		}
	}
}

// Generate a cone-shaped tet mesh directly from layered rings + apex.
// Base center at `center`, base radius `radius`, height along +Y.
inline void avbdGenerateConeTets(
	PxVec3 center, PxReal radius, PxReal height, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	outVerts.clear();
	outTets.clear();

	const int nLayers = PxMax(N, 2);
	const int nRing   = PxMax(4 * N, 8);
	const PxReal pi2  = 2.0f * 3.14159265358979f;

	// --- vertices ---
	// Each layer i (0..nLayers-1): 1 center + nRing ring vertices
	// Final vertex: apex
	for (int i = 0; i < nLayers; i++)
	{
		PxReal t = PxReal(i) / PxReal(nLayers); // 0 = base, approaches 1 near tip
		PxReal h = t * height;
		PxReal r = radius * (1.0f - t);

		// Center of this layer
		outVerts.pushBack(center + PxVec3(0.0f, h, 0.0f));

		// Ring vertices
		for (int j = 0; j < nRing; j++)
		{
			PxReal angle = pi2 * PxReal(j) / PxReal(nRing);
			outVerts.pushBack(center + PxVec3(r * PxCos(angle), h, r * PxSin(angle)));
		}
	}

	PxU32 apexIdx = outVerts.size();
	outVerts.pushBack(center + PxVec3(0.0f, height, 0.0f));

	// --- helper lambdas ---
	const int stride = 1 + nRing;
	// center vertex of layer i
	auto ci = [stride](int layer) -> PxU32 { return PxU32(layer * stride); };
	// ring vertex j of layer i (j wraps around)
	auto ri = [stride, nRing](int layer, int j) -> PxU32
	{
		return PxU32(layer * stride + 1 + ((j % nRing + nRing) % nRing));
	};

	// --- tets between adjacent layers (prism decomposition) ---
	for (int i = 0; i + 1 < nLayers; i++)
	{
		for (int j = 0; j < nRing; j++)
		{
			// 3 tets per wedge (triangular prism between two ring segments)
			outTets.pushBack(ci(i));   outTets.pushBack(ri(i, j+1)); outTets.pushBack(ri(i, j));   outTets.pushBack(ci(i+1));
			outTets.pushBack(ri(i,j)); outTets.pushBack(ri(i,j+1)); outTets.pushBack(ci(i+1));     outTets.pushBack(ri(i+1,j));
			outTets.pushBack(ri(i,j+1)); outTets.pushBack(ci(i+1)); outTets.pushBack(ri(i+1,j));   outTets.pushBack(ri(i+1,j+1));
		}
	}

	// --- apex cap (connect top layer to apex) ---
	{
		int top = nLayers - 1;
		for (int j = 0; j < nRing; j++)
		{
			outTets.pushBack(ci(top)); outTets.pushBack(ri(top, j+1)); outTets.pushBack(ri(top, j)); outTets.pushBack(apexIdx);
		}
	}

	// --- fix orientation: ensure positive tet volume ---
	for (PxU32 t = 0; t + 3 < outTets.size(); t += 4)
	{
		PxVec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
		PxVec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
		PxVec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
		if (e1.dot(e2.cross(e3)) < 0.0f)
		{
			PxU32 tmp = outTets[t+1]; outTets[t+1] = outTets[t+2]; outTets[t+2] = tmp;
		}
	}
}

// =============================================================================
// NOTE: For production tet mesh generation from arbitrary shapes, use the
// PhysX TetMaker API (PxTetMaker::createConformingTetrahedronMesh +
// PxTetMaker::createVoxelTetrahedronMesh) which provides BVH-based surface
// projection, boundary cell subdivision, and iterative relaxation.
// See extensions/PxTetMakerExt.h.
// =============================================================================

// =============================================================================
// Soft body creation helper
// =============================================================================

inline PxU32 avbdCreateSoftBody(
	const PxVec3* vertices, PxU32 numVertices,
	const PxU32* tets, PxU32 numTetIndices,
	const PxU32* tris, PxU32 numTriIndices,
	PxReal youngsModulus, PxReal poissonsRatio,
	PxReal density, PxReal damping,
	PxReal bendingStiffness, PxReal thickness,
	PxArray<AvbdSoftParticle>& outParticles,
	PxArray<AvbdSoftBody>& outSoftBodies)
{
	PxU32 particleStart = outParticles.size();

	PxArray<PxReal> vertexMass;
	vertexMass.resize(numVertices, 0.0f);

	if (numTetIndices > 0)
	{
		for (PxU32 i = 0; i + 3 < numTetIndices; i += 4)
		{
			PxVec3 e1 = vertices[tets[i+1]] - vertices[tets[i]];
			PxVec3 e2 = vertices[tets[i+2]] - vertices[tets[i]];
			PxVec3 e3 = vertices[tets[i+3]] - vertices[tets[i]];
			PxReal vol = PxAbs(e1.dot(e2.cross(e3)) / 6.0f);
			PxReal tetMass = vol * density;
			PxReal perVertex = tetMass / 4.0f;
			vertexMass[tets[i]]   += perVertex;
			vertexMass[tets[i+1]] += perVertex;
			vertexMass[tets[i+2]] += perVertex;
			vertexMass[tets[i+3]] += perVertex;
		}
	}
	else if (numTriIndices > 0)
	{
		for (PxU32 i = 0; i + 2 < numTriIndices; i += 3)
		{
			PxVec3 e1 = vertices[tris[i+1]] - vertices[tris[i]];
			PxVec3 e2 = vertices[tris[i+2]] - vertices[tris[i]];
			PxReal area = e1.cross(e2).magnitude() * 0.5f;
			PxReal triMass = area * thickness * density;
			PxReal perVertex = triMass / 3.0f;
			vertexMass[tris[i]]   += perVertex;
			vertexMass[tris[i+1]] += perVertex;
			vertexMass[tris[i+2]] += perVertex;
		}
	}

	PxReal minMass = 1e-4f;
	for (PxU32 i = 0; i < numVertices; i++)
		vertexMass[i] = PxMax(vertexMass[i], minMass);

	// Mass uniformization (matches PhysX GPU's maxInvMassRatio=50):
	// Clamp minimum mass so that max/min mass ratio <= 50.
	// Prevents tiny tets from creating particles with extreme inv-mass
	// that blow up under elastic forces.
	{
		PxReal maxMass = 0.0f;
		for (PxU32 i = 0; i < numVertices; i++)
			maxMass = PxMax(maxMass, vertexMass[i]);
		const PxReal maxInvMassRatio = 50.0f;
		PxReal massFloor = maxMass / maxInvMassRatio;
		for (PxU32 i = 0; i < numVertices; i++)
			vertexMass[i] = PxMax(vertexMass[i], massFloor);
	}

	for (PxU32 i = 0; i < numVertices; i++)
	{
		AvbdSoftParticle sp;
		sp.position = vertices[i];
		sp.velocity = PxVec3(0.0f);
		sp.prevVelocity = PxVec3(0.0f);
		sp.initialPosition = vertices[i];
		sp.predictedPosition = vertices[i];
		sp.mass = vertexMass[i];
		sp.invMass = 1.0f / sp.mass;
		sp.damping = damping;
		outParticles.pushBack(sp);
	}

	AvbdSoftBody sb;
	sb.particleStart = particleStart;
	sb.particleCount = numVertices;

	if (numTetIndices > 0)
		for (PxU32 i = 0; i < numTetIndices; i++)
			sb.tetrahedra.pushBack(tets[i]);

	if (numTriIndices > 0)
		for (PxU32 i = 0; i < numTriIndices; i++)
			sb.triangles.pushBack(tris[i]);

	sb.youngsModulus = youngsModulus;
	sb.poissonsRatio = poissonsRatio;
	sb.density = density;
	sb.damping = damping;
	sb.bendingStiffness = bendingStiffness;
	sb.thickness = thickness;

	sb.buildElements(outParticles);

	outSoftBodies.pushBack(sb);
	return particleStart;
}

// =============================================================================
// AVBD unified solver step -- self-contained soft body simulation step
// =============================================================================

// Callback type for contact re-detection between outer iterations.
// Called with current particles/bodies + the contacts array to refill.
typedef void (*AvbdContactRedetectFn)(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, void* userData);

inline void avbdStepSoftBodies(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, const PxVec3& gravity,
	PxU32 outerIterations = 1, PxU32 innerIterations = 10,
	PxReal avbdBeta = 1000.0f,
	AvbdContactRedetectFn redetectFn = NULL,
	PxArray<AvbdSoftContact>* contactsArray = NULL,
	void* redetectUserData = NULL,
	PxReal chebyshevRho = 0.92f)
{
	if (numParticles == 0 || numSoftBodies == 0) return;

	PxReal invDt = dt > 0.0f ? 1.0f / dt : 0.0f;
	PxReal invDtSq = invDt * invDt;

	// Stage 1: prediction
	for (PxU32 i = 0; i < numParticles; i++)
	{
		particles[i].computePrediction(dt, gravity);
		// Reset elastic proximal weight for new timestep
		// (warmstart: retain a fraction from prior timestep for stability)
		particles[i].elasticK = particles[i].elasticK * 0.5f;
	}

	// Stage 2: warmstart contact penalties
	for (PxU32 ci = 0; ci < numContacts; ci++)
		contacts[ci].k = PxMin(contacts[ci].k * 2.0f, contacts[ci].ke);

	// Build per-particle contact index to avoid O(particles*contacts) scan.
	// contactStart[pi] = first index into contactIdx for particle pi.
	// contactIdx stores contact indices grouped by particle.
	PxArray<PxU32> contactIdxBuf(numContacts);
	PxArray<PxU32> contactStart(numParticles + 1);
	PxArray<PxU32> contactCount(numParticles);
	auto buildContactIndex = [&]()
	{
		for (PxU32 i = 0; i < numParticles; i++) contactCount[i] = 0;
		for (PxU32 ci = 0; ci < numContacts; ci++)
			contactCount[contacts[ci].particleIdx]++;
		contactStart[0] = 0;
		for (PxU32 i = 0; i < numParticles; i++)
			contactStart[i + 1] = contactStart[i] + contactCount[i];
		for (PxU32 i = 0; i < numParticles; i++) contactCount[i] = 0;
		for (PxU32 ci = 0; ci < numContacts; ci++)
		{
			PxU32 pi = contacts[ci].particleIdx;
			contactIdxBuf[contactStart[pi] + contactCount[pi]] = ci;
			contactCount[pi]++;
		}
	};
	buildContactIndex();

	// Pre-compute body-level inertial targets for Newton-style body solve
	PxArray<PxVec3> bodyComPred(numSoftBodies);
	PxArray<PxVec3> bodyThetaPred(numSoftBodies);
	PxArray<PxVec3> bodyAccumTheta(numSoftBodies);
	for (PxU32 si = 0; si < numSoftBodies; si++)
	{
		const AvbdSoftBody& sb = softBodies[si];
		PxVec3 com(0.0f), comPred(0.0f);
		PxReal totalMass = 0.0f;
		PxVec3 angMom(0.0f);
		for (PxU32 li = 0; li < sb.particleCount; li++)
		{
			PxU32 pi = sb.particleStart + li;
			if (particles[pi].isStatic()) continue;
			PxReal m = particles[pi].mass;
			com = com + particles[pi].position * m;
			comPred = comPred + particles[pi].predictedPosition * m;
			totalMass += m;
		}
		if (totalMass > 0.0f)
		{
			PxReal invM = 1.0f / totalMass;
			com = com * invM;
			comPred = comPred * invM;
		}
		bodyComPred[si] = comPred;
		PxMat33 bodyI = PxMat33::createDiagonal(PxVec3(0.0f));
		for (PxU32 li = 0; li < sb.particleCount; li++)
		{
			PxU32 pi = sb.particleStart + li;
			if (particles[pi].isStatic()) continue;
			PxReal m = particles[pi].mass;
			PxVec3 r = particles[pi].position - com;
			PxReal r2 = r.dot(r);
			bodyI = bodyI + (PxMat33::createDiagonal(PxVec3(r2)) - avbdOuter(r, r)) * m;
			angMom = angMom + r.cross(particles[pi].velocity) * m;
		}
		PxVec3 omega = bodyI.getInverse() * angMom;
		if (omega.x != omega.x) omega = PxVec3(0.0f);
		bodyThetaPred[si] = omega * dt;
		bodyAccumTheta[si] = PxVec3(0.0f);
	}

	// Chebyshev semi-iterative acceleration state.
	// If chebyshevRho > 0, we use adaptive spectral-radius estimation:
	// measure the actual GS convergence rate from inner iterations 0-1,
	// then use min(measured, user-provided) as the Chebyshev parameter.
	// This prevents over-relaxation on meshes whose spectral radius
	// differs from the user's estimate (e.g., non-uniform voxel meshes).
	const bool useChebyshev = (chebyshevRho > 0.0f && chebyshevRho < 1.0f);
	PxReal chebyOmega = 1.0f;
	PxReal adaptiveRho = chebyshevRho;
	PxArray<PxVec3> chebyPrevPos(numParticles);
	PxArray<PxVec3> chebyPrevPrevPos(numParticles);
	if (useChebyshev)
	{
		for (PxU32 i = 0; i < numParticles; i++)
		{
			chebyPrevPos[i] = particles[i].position;
			chebyPrevPrevPos[i] = particles[i].position;
		}
	}

	// Main iteration loop
	for (PxU32 outerIt = 0; outerIt < outerIterations; outerIt++)
	{
		// Body-level 6x6 solve: Newton-style rigid-body update for soft bodies.
		// Solves for body-level translation and rotation from contact forces,
		// then applies as a rigid-body motion to all particles.
		// Neo-Hookean elastic energy is rotation-invariant, so the subsequent
		// per-particle VBD solve won't fight this body-level correction.
		// Ref: DyAvbdSolver::solveLocalSystem, Newton VBD rigid_vbd_kernels.py
		for (PxU32 si = 0; si < numSoftBodies; si++)
		{
			const AvbdSoftBody& sb = softBodies[si];
			PxVec3 com(0.0f);
			PxReal bodyMass = 0.0f;
			for (PxU32 li = 0; li < sb.particleCount; li++)
			{
				PxU32 pi = sb.particleStart + li;
				if (particles[pi].isStatic()) continue;
				com = com + particles[pi].position * particles[pi].mass;
				bodyMass += particles[pi].mass;
			}
			if (bodyMass <= 0.0f) continue;
			com = com * (1.0f / bodyMass);

			PxU32 bodyContactCount = 0;
			for (PxU32 li = 0; li < sb.particleCount; li++)
			{
				PxU32 pi = sb.particleStart + li;
				bodyContactCount += contactStart[pi + 1] - contactStart[pi];
			}
			if (bodyContactCount == 0) continue;

			PxMat33 bodyInertia = PxMat33::createDiagonal(PxVec3(0.0f));
			for (PxU32 li = 0; li < sb.particleCount; li++)
			{
				PxU32 pi = sb.particleStart + li;
				if (particles[pi].isStatic()) continue;
				PxVec3 r = particles[pi].position - com;
				PxReal r2 = r.dot(r);
				bodyInertia = bodyInertia +
					(PxMat33::createDiagonal(PxVec3(r2)) - avbdOuter(r, r)) * particles[pi].mass;
			}

			PxReal bodyMassDtSq = bodyMass * invDtSq;
			PxMat33 A_ll = PxMat33::createDiagonal(PxVec3(bodyMassDtSq));
			PxMat33 A_la = PxMat33::createDiagonal(PxVec3(0.0f));
			PxMat33 A_al = PxMat33::createDiagonal(PxVec3(0.0f));
			PxMat33 A_aa = bodyInertia * invDtSq;
			A_aa = A_aa + PxMat33::createDiagonal(PxVec3(1e-4f * bodyMassDtSq));

			PxVec3 g_l = (com - bodyComPred[si]) * bodyMassDtSq;
			PxVec3 g_a = (bodyInertia * invDtSq) * (bodyAccumTheta[si] - bodyThetaPred[si]);

			for (PxU32 li = 0; li < sb.particleCount; li++)
			{
				PxU32 pi = sb.particleStart + li;
				PxVec3 r = particles[pi].position - com;
				for (PxU32 k = contactStart[pi]; k < contactStart[pi + 1]; k++)
				{
					const AvbdSoftContact& sc = contacts[contactIdxBuf[k]];
					PxVec3 n = sc.normal;
					PxReal violation;
					if (sc.rigidBodyIdx == PX_MAX_U32)
						violation = particles[pi].position.dot(n);
					else
						violation = (particles[pi].position - sc.surfacePoint).dot(n) - sc.margin;

					PxReal pen = sc.k;
					PxVec3 rCrossN = r.cross(n);
					A_ll = A_ll + avbdOuter(n, n) * pen;
					A_la = A_la + avbdOuter(n, rCrossN) * pen;
					A_al = A_al + avbdOuter(rCrossN, n) * pen;
					A_aa = A_aa + avbdOuter(rCrossN, rCrossN) * pen;

					PxReal f = PxMin(0.0f, pen * violation);
					if (f < 0.0f)
					{
						g_l = g_l + n * f;
						g_a = g_a + rCrossN * f;
					}
				}
			}

			PxMat33 A_ll_inv = A_ll.getInverse();
			PxMat33 S = A_aa - A_al * A_ll_inv * A_la;
			PxVec3 deltaTheta = S.getInverse() * (g_a - A_al * (A_ll_inv * g_l));
			PxVec3 deltaPos = A_ll_inv * (g_l - A_la * deltaTheta);

			if (deltaPos.x != deltaPos.x || deltaTheta.x != deltaTheta.x) continue;

			PxReal thetaMag = deltaTheta.magnitude();
			if (thetaMag > 0.5f) deltaTheta = deltaTheta * (0.5f / thetaMag);

			for (PxU32 li = 0; li < sb.particleCount; li++)
			{
				PxU32 pi = sb.particleStart + li;
				if (particles[pi].isStatic()) continue;
				PxVec3 r = particles[pi].position - com;
				particles[pi].position = particles[pi].position - deltaPos - deltaTheta.cross(r);
			}
			bodyAccumTheta[si] = bodyAccumTheta[si] - deltaTheta;
		}

		// Snapshot positions as proximal anchor for AVBD elastic term
		for (PxU32 i = 0; i < numParticles; i++)
			particles[i].outerPosition = particles[i].position;

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

		for (PxU32 innerIt = 0; innerIt < innerIterations; innerIt++)
		{
			PxReal maxDxSq = 0.0f;

			// Soft particle primal
			for (PxU32 si = 0; si < numSoftBodies; si++)
			{
				const AvbdSoftBody& sb = softBodies[si];
				for (PxU32 li = 0; li < sb.particleCount; li++)
				{
					PxU32 pi = sb.particleStart + li;
					AvbdSoftParticle& sp = particles[pi];
					if (sp.isStatic()) continue;

					// Inertial term
					PxReal m_dtSq = sp.mass * invDtSq;
					PxMat33 H = PxMat33::createDiagonal(PxVec3(m_dtSq));
					PxVec3 f = (sp.predictedPosition - sp.position) * m_dtSq;

					const AvbdParticleAdjacency& adj = sb.adjacency[li];

					// Triangle (StVK) contributions
					for (PxU32 ti = 0; ti < adj.triRefs.size(); ti++)
					{
						const AvbdParticleElementRef& ref = adj.triRefs[ti];
						PxVec3 ef; PxMat33 eH;
						avbdEvaluateStVKForceHessian(
							sb.triElements[ref.index], int(ref.vOrder),
							sb.mu, sb.lambda, particles, ef, eH);
						f = f + ef;
						H = H + eH;
					}

					// Tet (Neo-Hookean) contributions
					for (PxU32 ti = 0; ti < adj.tetRefs.size(); ti++)
					{
						const AvbdParticleElementRef& ref = adj.tetRefs[ti];
						PxVec3 ef; PxMat33 eH;
						avbdEvaluateNeoHookeanForceHessian(
							sb.tetElements[ref.index], int(ref.vOrder),
							sb.mu, sb.lambda, particles, ef, eH);
						f = f + ef;
						H = H + eH;
					}

					// Bending contributions
					for (PxU32 bi = 0; bi < adj.bendRefs.size(); bi++)
					{
						const AvbdParticleElementRef& ref = adj.bendRefs[bi];
						PxVec3 ef; PxMat33 eH;
						avbdEvaluateBendingForceHessian(
							sb.bendElements[ref.index], int(ref.vOrder),
							sb.bendingStiffness, particles, ef, eH);
						f = f + ef;
						H = H + eH;
					}

					// Contact contributions (indexed lookup)
					for (PxU32 k = contactStart[pi]; k < contactStart[pi + 1]; k++)
					{
						PxVec3 cf; PxMat33 cH;
						avbdEvaluateContactForceHessian(contacts[contactIdxBuf[k]], particles, cf, cH);
						f = f + cf;
						H = H + cH;
					}

					// Pin contributions
					for (PxU32 ki = 0; ki < adj.pinIndices.size(); ki++)
					{
						PxVec3 pf; PxMat33 pH;
						avbdEvaluatePinForceHessian(sb.pins[adj.pinIndices[ki]], particles, pf, pH);
						f = f + pf;
						H = H + pH;
					}

					// Stiffness-proportional Rayleigh damping (Newton VBD style):
					// Per-axis damping proportional to elastic stiffness, clamped so no
					// axis gets less damping than mass-proportional (baseline stability).
					if (sp.damping > 0.0f)
					{
						PxReal dampCoeff = sp.damping * sp.mass * invDt;
						PxMat33 H_elastic = H - PxMat33::createDiagonal(PxVec3(m_dtSq));
						PxReal he_xx = PxMax(H_elastic[0][0], 0.0f);
						PxReal he_yy = PxMax(H_elastic[1][1], 0.0f);
						PxReal he_zz = PxMax(H_elastic[2][2], 0.0f);
						PxReal trHe = he_xx + he_yy + he_zz;
						PxReal dx, dy, dz;
						if (trHe > 1e-10f)
						{
							PxReal s = dampCoeff * 3.0f / trHe;
							dx = PxMax(he_xx * s, dampCoeff);
							dy = PxMax(he_yy * s, dampCoeff);
							dz = PxMax(he_zz * s, dampCoeff);
						}
						else
						{
							dx = dy = dz = dampCoeff;
						}
						PxMat33 H_damp = PxMat33::createDiagonal(PxVec3(dx, dy, dz));
						f = f - H_damp * (sp.position - sp.initialPosition);
						H = H + H_damp;
					}

					// AVBD elastic proximal term: pulls toward outer-iteration anchor
					// to ensure convergence independent of update order (Jacobi-safe)
					if (sp.elasticK > 0.0f)
					{
						H = H + PxMat33::createDiagonal(PxVec3(sp.elasticK));
						f = f + (sp.outerPosition - sp.position) * sp.elasticK;
					}

					// Solve H * dx = f  (with displacement clamping)
					PxVec3 dx = H.getInverse() * f;
					PxReal dxLenSq = dx.magnitudeSquared();
					const PxReal maxDx = 1.0f;
					if (dxLenSq > maxDx * maxDx)
						dx = dx * (maxDx / PxSqrt(dxLenSq));
					// NaN guard: skip update if degenerate
					if (dx.x == dx.x)
					{
						sp.position = sp.position + dx;
						if (dxLenSq > maxDxSq) maxDxSq = dxLenSq;
					}
				}
			}

			// Early termination: converged if max displacement < 1e-6
			if (maxDxSq < 1e-12f) break;

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

		// Collision projection (Jolt-style hard constraint).
		// Uses projNormal (face-normal corrected, always outward)
		// so inside particles are pushed out correctly.
		for (PxU32 ci = 0; ci < numContacts; ci++)
		{
			AvbdSoftParticle& sp = particles[contacts[ci].particleIdx];
			if (sp.isStatic()) continue;
			const AvbdSoftContact& sc = contacts[ci];
			PxVec3 n = sc.projNormal;
			PxReal projPen;
			if (sc.rigidBodyIdx == PX_MAX_U32)
				projPen = -(sp.position.dot(n));          // ground plane
			else
				projPen = -(sp.position - sc.surfacePoint).dot(n);  // body surface
			if (projPen > 0.0f)
			{
				// Clamp soft-soft projection per iteration to avoid
				// destabilizing elastic forces.
				if (sc.rigidBodyIdx != PX_MAX_U32 && sc.rigidBodyIdx < numSoftBodies)
					projPen = PxMin(projPen, 0.05f);
				sp.position += n * projPen;
			}
		}

		// Dual update (contacts, pins, elastic proximal)
		for (PxU32 ci = 0; ci < numContacts; ci++)
			avbdUpdateSoftContactDual(contacts[ci], particles, avbdBeta);

		for (PxU32 si = 0; si < numSoftBodies; si++)
		{
			AvbdSoftBody& sb = softBodies[si];
			for (PxU32 ki = 0; ki < sb.pins.size(); ki++)
				avbdUpdatePinDual(sb.pins[ki], particles, avbdBeta);
		}

		// AVBD elastic proximal dual update: increase proximal weight
		// proportional to displacement from the outer-iteration anchor
		for (PxU32 i = 0; i < numParticles; i++)
		{
			AvbdSoftParticle& sp = particles[i];
			if (sp.isStatic()) continue;
			PxReal disp = (sp.position - sp.outerPosition).magnitude();
			sp.elasticK = PxMin(sp.elasticK + avbdBeta * disp, sp.elasticKMax);
		}

		// Re-detect contacts between outer iterations so surface anchors
		// track the deforming geometry instead of going stale.
		if (redetectFn && contactsArray && outerIt + 1 < outerIterations)
		{
			redetectFn(particles, numParticles, softBodies, numSoftBodies,
					   *contactsArray, redetectUserData);
			contacts = contactsArray->begin();
			numContacts = contactsArray->size();
			// Warmstart the fresh contacts
			for (PxU32 ci = 0; ci < numContacts; ci++)
				contacts[ci].k = PxMin(contacts[ci].k * 2.0f, contacts[ci].ke);
			// Rebuild per-particle contact index
			contactIdxBuf.resize(numContacts);
			buildContactIndex();
		}
	}

	// Stage 3: velocity update
	for (PxU32 i = 0; i < numParticles; i++)
		particles[i].updateVelocityFromPosition(invDt);

	// Stage 4: explicit contact friction on updated velocities.
	// The contact Hessian intentionally omits tangential terms, so without
	// an explicit pass bodies can keep sliding forever after contact.
	for (PxU32 i = 0; i < numParticles; i++)
	{
		AvbdSoftParticle& sp = particles[i];
		if (sp.isStatic()) continue;

		PxVec3 accumulatedNormal(0.0f);
		PxReal maxFriction = 0.0f;
		PxReal maxSupportSpeed = 0.0f;

		for (PxU32 k = contactStart[i]; k < contactStart[i + 1]; k++)
		{
			const AvbdSoftContact& sc = contacts[contactIdxBuf[k]];
			PxReal penetration;
			if (sc.rigidBodyIdx == PX_MAX_U32)
				penetration = -(sp.position.dot(sc.projNormal));
			else
				penetration = -(sp.position - sc.surfacePoint).dot(sc.projNormal);
			if (penetration <= 0.0f) continue;

			PxVec3 n = sc.projNormal.getNormalized();
			accumulatedNormal += n * penetration;
			maxFriction = PxMax(maxFriction, sc.friction);
			maxSupportSpeed = PxMax(maxSupportSpeed, penetration * invDt);
		}

		PxReal nLen = accumulatedNormal.magnitude();
		if (nLen <= 1e-8f || maxFriction <= 0.0f) continue;

		PxVec3 n = accumulatedNormal * (1.0f / nLen);
		PxReal vn = sp.velocity.dot(n);
		PxVec3 vt = sp.velocity - n * vn;
		PxReal vtMag = vt.magnitude();
		if (vtMag <= 1e-8f) continue;

		// Coulomb-like explicit tangential damping. The support speed term gives
		// static-friction-like stopping even once the normal velocity is near zero.
		PxReal frictionBudget = maxFriction * (PxMax(0.0f, -vn) + maxSupportSpeed + 0.25f);
		PxReal newVtMag = PxMax(0.0f, vtMag - frictionBudget);
		sp.velocity = n * PxMax(0.0f, vn) + vt * (newVtMag / vtMag);
	}

}

} // namespace Dy
} // namespace physx

#endif // PX_AVBD_SOFT_BODY_H
