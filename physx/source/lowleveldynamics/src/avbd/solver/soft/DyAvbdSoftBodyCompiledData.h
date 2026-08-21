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

#ifndef DY_AVBD_SOFT_BODY_COMPILED_DATA_H
#define DY_AVBD_SOFT_BODY_COMPILED_DATA_H

#include "foundation/PxArray.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/solver/soft/DyAvbdSoftBodyData.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopology.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_COMPILED_DATA_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_COMPILED_DATA_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_COMPILED_DATA_API
#endif

struct AvbdSoftBodyCompiledData
{
	PxU32 particleStart;
	PxU32 particleCount;

	PxArray<PxU32> tetrahedra;
	PxArray<PxU32> triangles;

	PxArray<AvbdTriElement> triElements;
	PxArray<AvbdTetElement> tetElements;
	PxArray<AvbdBendingElement> bendElements;
	PxArray<AvbdEdgeInfo> edges;
	PxArray<AvbdParticleElementAdjacency> elementAdjacency;
	// P8.2 keeps a canonical, topology-versioned projection of tetRefs. A
	// particle's packet lanes are consecutive original adjacency ordinals, so
	// future packet arithmetic can retain scalar-order reduction and stable
	// positive-J linearization ownership.
	PxArray<AvbdTetIncidencePacket8> tetIncidencePackets;
	PxArray<AvbdTetIncidencePacketRange> tetIncidencePacketRanges;
	PxU32 tetIncidenceFullPacketCount;
	bool tetIncidencePacketProgramValid;
	// P4.1: immutable, body-local structural conflict CSR for particle-primal
	// blocks.  It contains the union of every tri/tet/bending hyperedge.  It is
	// intentionally not a complete color plan: contacts and point objectives
	// add dynamic supports after redetection and must be overlaid before a
	// colored schedule is allowed to consume it.
	PxArray<PxU32> particlePrimalStructuralConflictOffsets;
	PxArray<PxU32> particlePrimalStructuralConflictIndices;
	bool particlePrimalStructuralConflictValid;
	bool flatteningEnabled;

	// Immutable local rest-space input used to compile self-collision
	// candidates. The public filter distance is copied beside it during
	// prep so contact detection does not reinterpret actor flags or use the
	// global current-space contact radius as a rest-neighborhood policy.
	PxArray<PxVec3> selfCollisionRestPositions;
	PxReal selfCollisionFilterDistance;
	// Immutable rest-space exclusion cache for the self vertex/triangle
	// filter. Each local query vertex stores boundary-triangle indices whose
	// rest-space closest distance is within the actor filter distance. This is
	// deliberately separate from current-position bounds: it survives refits
	// but must be rebuilt when rest topology/positions or filter distance
	// changes. A bounded fallback preserves the direct query for unusually
	// dense rest-space neighborhoods.
	PxArray<PxArray<PxU32> > selfCollisionRestFilteredTriangles;
	PxReal selfCollisionRestFilterCacheDistance;
	bool selfCollisionRestFilterCacheValid;
	bool selfCollisionRestFilterCacheFallback;
	// Public per-actor collision-bias speed. This is consumed only by the
	// prepared contact owner; generic motion finalization must not clamp it.
	PxReal maxDepenetrationVelocity;
	// Volume self-collision contacts whose owning tetrahedron exceeds this
	// dimensionless co-rotated strain threshold are omitted, matching the
	// public GPU deformable-volume policy.
	PxReal selfCollisionStressTolerance;
	// Public deformable speculative-CCD policy. Swept candidates are
	// generated only for bodies that explicitly opt in.
	bool speculativeCCDEnabled;

	PxArray<PxU32> surfaceTriangles;  // boundary face indices (3 per tri, global particle indices)
	PxArray<PxU32> surfaceVertices;   // unique sorted boundary vertices, global particle indices
	PxArray<AvbdEdgeInfo> surfaceEdges; // unique boundary edges, global indices
	// Source element owning each boundary face. Surface entries are public
	// triangle indices; Volume entries are simulation tetrahedron indices.
	PxArray<PxU32> surfaceTriangleElementIndices;
	// For volume boundary faces, resolve the source tetrahedron to its compiled
	// tet-element slot once at topology build time. A non-existent slot is kept
	// as PX_MAX_U32 so the self-collision stress filter preserves its existing
	// permissive fallback for an invalid/omitted compiled source element.
	PxArray<PxU32> surfaceTriangleTetElementIndices;
	// The hierarchy's topology/permutation is compiled once. Bounds are mutable
	// detection scratch stored with the owning immutable topology, so a refit
	// cannot change feature identity or leaf membership.
	PxArray<PxU32> surfaceTriangleBvhTriangleIndices;
	PxArray<AvbdSurfaceTriangleBvhNode> surfaceTriangleBvhNodes;
	PxArray<PxU32> surfaceEdgeBvhEdgeIndices;
	PxArray<AvbdSurfaceEdgeBvhNode> surfaceEdgeBvhNodes;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API AvbdSoftBodyCompiledData();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void compileBendingRestAngles(bool enabled);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API static PxReal computeDihedralAngle(const PxVec3& x0, const PxVec3& x1,
	                                   const PxVec3& x2, const PxVec3& x3);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildTriElements(const PxArray<AvbdSoftParticle>& particles);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildTetElements(const PxArray<AvbdSoftParticle>& particles);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildBendingElements(const PxArray<AvbdSoftParticle>& particles);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildEdges(const PxArray<AvbdSoftParticle>& particles);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void invalidateTetIncidencePacketProgram();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API bool validateTetIncidencePacketProgram() const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildTetIncidencePacketProgram();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildAdjacency(AvbdSoftBodyRuntimeState& runtime);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildParticlePrimalStructuralAccessDescriptor();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API bool validateParticlePrimalStructuralAccessDescriptor() const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildSurfaceTriangles(
		const PxArray<AvbdSoftParticle>& particles);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildSurfaceTriangleTetElementIndices();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API PxU32 buildSurfaceTriangleBvhNode(PxU32 first, PxU32 count);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildSurfaceTriangleBvh();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void refitSurfaceTriangleBvh(
		const AvbdSoftParticle* particles, bool swept,
		PxArray<AvbdSurfaceBvhNodeBounds>& bounds) const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void collectSurfaceTriangleBvhNodeCandidates(
		PxU32 nodeIndex,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		const PxVec3& queryMinimum,
		const PxVec3& queryMaximum, PxReal margin,
		PxArray<PxU32>& candidates) const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void collectSurfaceTriangleBvhCandidates(
		const PxVec3& queryMinimum, const PxVec3& queryMaximum,
		PxReal margin,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		PxArray<PxU32>& candidates) const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API PxU32 buildSurfaceEdgeBvhNode(PxU32 first, PxU32 count);

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildSurfaceEdgeBvh();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void refitSurfaceEdgeBvh(
		const AvbdSoftParticle* particles, bool swept,
		PxArray<AvbdSurfaceBvhNodeBounds>& bounds) const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void collectSurfaceEdgeBvhNodeCandidates(
		PxU32 nodeIndex,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		const PxVec3& queryMinimum,
		const PxVec3& queryMaximum, PxReal margin,
		PxArray<PxU32>& candidates) const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void collectSurfaceEdgeBvhCandidates(
		const PxVec3& queryMinimum, const PxVec3& queryMaximum,
		PxReal margin,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		PxArray<PxU32>& candidates) const;

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildSelfCollisionRestVertexTriangleFilter();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void ensureSelfCollisionRestVertexTriangleFilter();

	DY_AVBD_SOFT_BODY_COMPILED_DATA_API void buildElements(
		const PxArray<AvbdSoftParticle>& particles,
		AvbdSoftBodyMaterialData& material,
		AvbdSoftBodyRuntimeState& runtime);
};

#undef DY_AVBD_SOFT_BODY_COMPILED_DATA_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_COMPILED_DATA_H
