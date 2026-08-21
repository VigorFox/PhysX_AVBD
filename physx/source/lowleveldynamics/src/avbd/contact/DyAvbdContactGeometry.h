// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions are met.

#ifndef DY_AVBD_CONTACT_GEOMETRY_H
#define DY_AVBD_CONTACT_GEOMETRY_H

#include "avbd/contact/DyAvbdContactFeature.h"
#include "PxMaterial.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

struct AvbdClosestPointResult
{
	PxVec3 point;
	PxVec3 barycentric;
	PxVec3 normal;
	PxReal distance;
	AvbdClosestFeature feature;
	PxU32 featureIndex;
};

struct AvbdClosestSegmentTriangleResult
{
	PxVec3 segmentPoint;
	PxVec3 trianglePoint;
	PxVec3 barycentric;
	PxReal segmentWeight1;
	PxReal distance;
	AvbdClosestFeature feature;
	PxU32 featureIndex;
};

struct AvbdSweptTriangleEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxVec3 barycentric;
	AvbdClosestFeature feature;
	PxU32 featureIndex;

	AvbdSweptTriangleEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  barycentric(1.0f, 0.0f, 0.0f),
		  feature(AVBD_FEATURE_UNKNOWN), featureIndex(0)
	{
	}
};

struct AvbdSweptRotatingCapsulePointEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxVec3 surfaceLocal;

	AvbdSweptRotatingCapsulePointEntry()
		: entryTime(PX_MAX_F32), normal(0.0f), surfaceLocal(0.0f)
	{
	}
};

struct AvbdSweptCapsuleTriangleEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxVec3 barycentric;
	PxReal segmentWeight1;
	AvbdClosestFeature feature;
	PxU32 featureIndex;

	AvbdSweptCapsuleTriangleEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  barycentric(1.0f, 0.0f, 0.0f), segmentWeight1(0.0f),
		  feature(AVBD_FEATURE_UNKNOWN), featureIndex(0)
	{
	}
};

struct AvbdRigidConvexPointQuery
{
	PxReal signedDistance;
	PxVec3 surfaceLocal;
	PxVec3 normalLocal;
	PxU64 featureKey;

	AvbdRigidConvexPointQuery()
		: signedDistance(PX_MAX_F32), surfaceLocal(0.0f),
		  normalLocal(0.0f, 1.0f, 0.0f), featureKey(0)
	{
	}
};

struct AvbdSweptConvexPointEntry
{
	PxReal entryTime;
	PxVec3 normalLocal;
	PxVec3 surfaceLocal;
	PxU64 featureKey;

	AvbdSweptConvexPointEntry()
		: entryTime(PX_MAX_F32), normalLocal(0.0f, 1.0f, 0.0f),
		  surfaceLocal(0.0f), featureKey(0)
	{
	}
};

struct AvbdSweptConvexEdgeEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxReal softWeight1;
	PxReal rigidWeight1;

	AvbdSweptConvexEdgeEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  softWeight1(0.0f), rigidWeight1(0.0f)
	{
	}
};

struct AvbdSweptTriangleSurfacePointEntry
{
	PxReal entryTime;
	PxVec3 normalLocal;
	PxVec3 surfaceLocal;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 featureKey;

	AvbdSweptTriangleSurfacePointEntry()
		: entryTime(PX_MAX_F32), normalLocal(0.0f, 1.0f, 0.0f),
		  surfaceLocal(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)), featureKey(0)
	{
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_GEOMETRY_H
