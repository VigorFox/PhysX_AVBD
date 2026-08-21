// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions are met.

#ifndef DY_AVBD_CONTACT_RIGID_PRIMITIVES_H
#define DY_AVBD_CONTACT_RIGID_PRIMITIVES_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdContactBounds.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "PxMaterial.h"
#include "foundation/PxBounds3.h"

namespace physx
{
namespace Dy
{

// Scene-owned rigid primitive descriptors.  They contain only immutable
// geometry/material identity plus the pose snapshots needed by the current
// contact epoch; detector and solver code do not retain public ShapeCore
// pointers.
struct AvbdRigidBox
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 halfExtent;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxTransform shapeToRigidBody;

	AvbdRigidBox()
		: center(0.0f), rotation(PxIdentity), halfExtent(0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), previousCenter(0.0f),
		  previousRotation(PxIdentity), shapeToRigidBody(PxIdentity)
	{
	}
};

struct AvbdRigidSphere
{
	PxVec3 center;
	PxQuat rotation;
	PxReal radius;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxVec3 predictedCenter;
	PxQuat predictedRotation;
	bool predictedPoseValid;
	PxTransform shapeToRigidBody;

	AvbdRigidSphere()
		: center(0.0f), rotation(PxIdentity), radius(0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), previousCenter(0.0f),
		  previousRotation(PxIdentity), predictedCenter(0.0f),
		  predictedRotation(PxIdentity), predictedPoseValid(false),
		  shapeToRigidBody(PxIdentity)
	{
	}
};

struct AvbdRigidCapsule
{
	PxVec3 center;
	PxQuat rotation;
	PxReal radius;
	PxReal halfHeight;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxVec3 predictedCenter;
	PxQuat predictedRotation;
	bool predictedPoseValid;
	PxTransform shapeToRigidBody;

	AvbdRigidCapsule()
		: center(0.0f), rotation(PxIdentity), radius(0.0f),
		  halfHeight(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), previousCenter(0.0f),
		  previousRotation(PxIdentity), predictedCenter(0.0f),
		  predictedRotation(PxIdentity), predictedPoseValid(false),
		  shapeToRigidBody(PxIdentity)
	{
	}
};

struct AvbdRigidConvexFace
{
	PxVec3 normal;
	PxReal offset;

	AvbdRigidConvexFace()
		: normal(0.0f, 1.0f, 0.0f), offset(0.0f)
	{
	}
};

struct AvbdRigidConvexEdge
{
	PxU32 p0;
	PxU32 p1;
	PxVec3 outward;

	AvbdRigidConvexEdge()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  outward(0.0f, 1.0f, 0.0f)
	{
	}
};

struct AvbdRigidConvexTriangle
{
	PxU32 p0;
	PxU32 p1;
	PxU32 p2;
	PxU32 faceIndex;

	AvbdRigidConvexTriangle()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  p2(PX_MAX_U32), faceIndex(PX_MAX_U32)
	{
	}
};

// Scene-owned convex topology. Shape-local geometry and all pose snapshots
// are kept in the detector IR so the narrow phase does not depend on public
// ShapeCore or geomutils object lifetimes.
struct AvbdRigidConvex
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxReal localRadius;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 predictedCenter;
	PxQuat predictedRotation;
	bool predictedPoseValid;
	PxTransform shapeToRigidBody;
	PxArray<PxVec3> vertices;
	PxArray<PxVec3> vertexNormals;
	PxArray<AvbdRigidConvexFace> faces;
	PxArray<AvbdRigidConvexEdge> edges;
	PxArray<AvbdRigidConvexTriangle> triangles;
	mutable PxArray<AvbdRigidConvexEdgeBounds> edgeBoundsScratch;

	AvbdRigidConvex()
		: center(0.0f), rotation(PxIdentity),
		  previousCenter(0.0f), previousRotation(PxIdentity),
		  localRadius(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), predictedCenter(0.0f),
		  predictedRotation(PxIdentity), predictedPoseValid(false),
		  shapeToRigidBody(PxIdentity)
	{
	}
};

// Caller-owned query state for a particle-range triangle-surface leaf. The
// descriptor stays immutable while this object owns all BVH candidate and
// feature-dedup writes for one serial query.
struct AvbdRigidTriangleSurfaceQueryScratch
{
	PxArray<PxU32> triangleBvhQueryCandidates;
	PxArray<PxU32> edgeBvhQueryCandidates;
	PxArray<PxU32> vertexBvhQueryCandidates;
	PxArray<PxU32> edgeBvhCandidateStamps;
	PxArray<PxU32> vertexBvhCandidateStamps;
	PxU32 featureBvhCandidateStamp;

	AvbdRigidTriangleSurfaceQueryScratch()
		: featureBvhCandidateStamp(0)
	{
	}

	void reserve(PxU32 triangleCount, PxU32 edgeCount, PxU32 vertexCount)
	{
		triangleBvhQueryCandidates.reserve(triangleCount);
		edgeBvhQueryCandidates.reserve(edgeCount);
		vertexBvhQueryCandidates.reserve(vertexCount);
		if(edgeBvhCandidateStamps.size() < edgeCount)
		{
			const PxU32 previousCount = edgeBvhCandidateStamps.size();
			edgeBvhCandidateStamps.resize(edgeCount);
			for(PxU32 index = previousCount; index < edgeCount; ++index)
				edgeBvhCandidateStamps[index] = 0;
		}
		if(vertexBvhCandidateStamps.size() < vertexCount)
		{
			const PxU32 previousCount = vertexBvhCandidateStamps.size();
			vertexBvhCandidateStamps.resize(vertexCount);
			for(PxU32 index = previousCount; index < vertexCount; ++index)
				vertexBvhCandidateStamps[index] = 0;
		}
	}

	bool beginFeatureCandidates(PxU32 edgeCount, PxU32 vertexCount)
	{
		if(edgeBvhCandidateStamps.size() != edgeCount)
		{
			edgeBvhCandidateStamps.resize(edgeCount);
			for(PxU32 index = 0; index < edgeCount; ++index)
				edgeBvhCandidateStamps[index] = 0;
		}
		if(vertexBvhCandidateStamps.size() != vertexCount)
		{
			vertexBvhCandidateStamps.resize(vertexCount);
			for(PxU32 index = 0; index < vertexCount; ++index)
				vertexBvhCandidateStamps[index] = 0;
		}
		if(++featureBvhCandidateStamp == 0)
		{
			featureBvhCandidateStamp = 1;
			for(PxU32 index = 0;
				index < edgeBvhCandidateStamps.size(); ++index)
				edgeBvhCandidateStamps[index] = 0;
			for(PxU32 index = 0;
				index < vertexBvhCandidateStamps.size(); ++index)
				vertexBvhCandidateStamps[index] = 0;
		}
		return true;
	}
};

struct AvbdRigidTriangleSurfaceFeatureWorkItem
{
	enum Phase : PxU8
	{
		eSWEPT,
		eDISCRETE
	};
	enum Family : PxU8
	{
		eSOFT_EDGE,
		eSOFT_TRIANGLE
	};

	Phase phase;
	Family family;
	PxU32 bodyIndex;
	PxU32 surfaceIndex;
	PxU32 primitiveBegin;
	PxU32 primitiveEnd;

	AvbdRigidTriangleSurfaceFeatureWorkItem(
		Phase inputPhase, Family inputFamily,
		PxU32 inputBodyIndex, PxU32 inputSurfaceIndex,
		PxU32 inputPrimitiveBegin, PxU32 inputPrimitiveEnd)
		: phase(inputPhase), family(inputFamily),
		  bodyIndex(inputBodyIndex), surfaceIndex(inputSurfaceIndex),
		  primitiveBegin(inputPrimitiveBegin), primitiveEnd(inputPrimitiveEnd)
	{
	}
};

struct AvbdRigidTriangleSurfaceFeaturePlan
{
	PxArray<AvbdRigidTriangleSurfaceFeatureWorkItem> items;

	void clear()
	{
		items.clear();
	}
};

struct AvbdRigidTriangleSurfacePointQuery
{
	PxReal distance;
	PxVec3 surfaceLocal;
	PxVec3 normalLocal;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 featureKey;

	AvbdRigidTriangleSurfacePointQuery()
		: distance(PX_MAX_F32), surfaceLocal(0.0f),
		  normalLocal(0.0f, 1.0f, 0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  featureKey(0)
	{
	}
};

struct AvbdRigidTriangleSurfaceVertex
{
	PxVec3 point;
	PxVec3 outward;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU32 sourceTriangleIndex;
	bool active;

	AvbdRigidTriangleSurfaceVertex()
		: point(0.0f), outward(0.0f, 1.0f, 0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  sourceTriangleIndex(PX_MAX_U32), active(false)
	{
	}
};

struct AvbdRigidTriangleSurfaceEdge
{
	PxU32 p0;
	PxU32 p1;
	PxU32 triangle0;
	PxU32 triangle1;
	PxU32 adjacentCount;
	PxVec3 outward;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU32 sourceTriangleIndex;
	bool active;

	AvbdRigidTriangleSurfaceEdge()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  triangle0(PX_MAX_U32), triangle1(PX_MAX_U32),
		  adjacentCount(0), outward(0.0f, 1.0f, 0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  sourceTriangleIndex(PX_MAX_U32), active(false)
	{
	}
};

struct AvbdRigidTriangleSurfaceTriangle
{
	PxU32 p0;
	PxU32 p1;
	PxU32 p2;
	PxU32 edge0;
	PxU32 edge1;
	PxU32 edge2;
	PxU32 sourceTriangleIndex;
	PxVec3 normal;
	PxReal friction;
	PxU8 frictionCombineMode;

	AvbdRigidTriangleSurfaceTriangle()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  p2(PX_MAX_U32), edge0(PX_MAX_U32),
		  edge1(PX_MAX_U32), edge2(PX_MAX_U32),
		  sourceTriangleIndex(PX_MAX_U32),
		  normal(0.0f, 1.0f, 0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE))
	{
	}
};

struct AvbdRigidTriangleSurfaceBvhNode
{
	PxVec3 minimum;
	PxVec3 maximum;
	PxU32 leftChild;
	PxU32 rightChild;
	PxU32 firstPrimitive;
	PxU32 primitiveCount;

	PX_FORCE_INLINE bool isLeaf() const
	{
		return leftChild == PX_MAX_U32;
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_RIGID_PRIMITIVES_H
