// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_TRIANGLE_SURFACE_TYPES_H
#define DY_AVBD_CONTACT_TRIANGLE_SURFACE_TYPES_H

#include "avbd/contact/DyAvbdContactRigidPrimitives.h"

namespace physx
{
namespace Dy
{

// CPU AVBD rigid triangle-surface descriptor.
//
// The descriptor owns only immutable topology/cache identity plus the pose
// supplied by the caller. Query and feature detection remain in
// DyAvbdContactTriangleSurface.inl.

struct AvbdRigidTriangleSurface
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxBounds3 localBounds;
	PxReal localRadius;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxTransform shapeToRigidBody;
	// Scene-owned immutable-topology cache identity. Pose fields above are
	// refreshed every detection; these fields invalidate only when mesh, scale,
	// or heightfield geometry changes.
	const void* topologySource;
	PxU8 topologyGeometryType;
	PxVec3 topologyScale;
	PxQuat topologyScaleRotation;
	PxReal topologyHeightScale;
	PxReal topologyRowScale;
	PxReal topologyColumnScale;
	PxU32 topologyContentTimestamp;
	PxU32 sceneCompileStamp;
	PxU32 sceneCompileOrder;
	PxArray<AvbdRigidTriangleSurfaceVertex> vertices;
	PxArray<AvbdRigidTriangleSurfaceEdge> edges;
	PxArray<AvbdRigidTriangleSurfaceTriangle> triangles;
	PxArray<PxU32> triangleBvhTriangleIndices;
	PxArray<AvbdRigidTriangleSurfaceBvhNode> triangleBvhNodes;
	// Detection is serial through P1. These pre-reserved query candidates
	// remain mutable only within the current serial reference/BVH comparison.
	mutable PxArray<PxU32> triangleBvhQueryCandidates;
	// Active reverse features are recovered from immutable triangle leaves.
	// Stamps deduplicate shared features without a per-query clear.
	mutable PxArray<PxU32> edgeBvhQueryCandidates;
	mutable PxArray<PxU32> vertexBvhQueryCandidates;
	mutable PxArray<PxU32> edgeBvhCandidateStamps;
	mutable PxArray<PxU32> vertexBvhCandidateStamps;
	mutable PxU32 featureBvhCandidateStamp;

	AvbdRigidTriangleSurface()
		: center(0.0f), rotation(PxIdentity),
		  previousCenter(0.0f), previousRotation(PxIdentity),
		  localBounds(PxBounds3::empty()), localRadius(0.0f),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), shapeToRigidBody(PxIdentity),
		  topologySource(NULL), topologyGeometryType(PX_MAX_U8),
		  topologyScale(1.0f), topologyScaleRotation(PxIdentity),
		  topologyHeightScale(0.0f), topologyRowScale(0.0f),
		  topologyColumnScale(0.0f), topologyContentTimestamp(0),
		  sceneCompileStamp(0),
		  sceneCompileOrder(0), featureBvhCandidateStamp(0)
	{
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_TRIANGLE_SURFACE_TYPES_H
