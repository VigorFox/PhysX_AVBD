// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// CPU AVBD rigid-box OGC geometry primitives.
//
// These helpers are intentionally stateless and shared by the box SDF and
// rigid-feature paths.  Detection orchestration remains in the rigid-soft
// adapter so this file carries no contact ownership or solver policy.

#ifndef DY_AVBD_CONTACT_RIGID_BOX_GEOMETRY_H
#define DY_AVBD_CONTACT_RIGID_BOX_GEOMETRY_H

#include "foundation/PxMath.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

PX_FORCE_INLINE PxVec3 avbdGetRigidBoxFaceNormal(
	const PxVec3& localNormal)
{
	const PxVec3 absNormal(
		PxAbs(localNormal.x),
		PxAbs(localNormal.y),
		PxAbs(localNormal.z));
	if(absNormal.x >= absNormal.y && absNormal.x >= absNormal.z)
		return PxVec3(
			localNormal.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
	if(absNormal.y >= absNormal.z)
		return PxVec3(
			0.0f, localNormal.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
	return PxVec3(
		0.0f, 0.0f, localNormal.z >= 0.0f ? 1.0f : -1.0f);
}

PX_FORCE_INLINE PxU64 avbdGetRigidBoxFaceFeatureKey(
	const PxVec3& faceNormal)
{
	if(PxAbs(faceNormal.x) > 0.5f)
		return faceNormal.x > 0.0f ? 1u : 2u;
	if(PxAbs(faceNormal.y) > 0.5f)
		return faceNormal.y > 0.0f ? 3u : 4u;
	return faceNormal.z > 0.0f ? 5u : 6u;
}

PX_FORCE_INLINE PxVec3 avbdGetRigidBoxVertexLocal(
	const PxVec3& halfExtent, PxU32 vertexIndex)
{
	return PxVec3(
		(vertexIndex & 1) ? halfExtent.x : -halfExtent.x,
		(vertexIndex & 2) ? halfExtent.y : -halfExtent.y,
		(vertexIndex & 4) ? halfExtent.z : -halfExtent.z);
}

PX_FORCE_INLINE void avbdGetRigidBoxEdgeLocal(
	const PxVec3& halfExtent, PxU32 edgeIndex,
	PxVec3& endpoint0, PxVec3& endpoint1,
	PxVec3& outwardNormal)
{
	const PxU32 axis = edgeIndex / 4;
	const PxU32 variant = edgeIndex & 3;
	const PxReal sign0 = (variant & 1) ? 1.0f : -1.0f;
	const PxReal sign1 = (variant & 2) ? 1.0f : -1.0f;
	endpoint0 = endpoint1 = PxVec3(0.0f);
	outwardNormal = PxVec3(0.0f);

	if(axis == 0)
	{
		endpoint0 = PxVec3(
			-halfExtent.x, sign0 * halfExtent.y,
			sign1 * halfExtent.z);
		endpoint1 = PxVec3(
			halfExtent.x, sign0 * halfExtent.y,
			sign1 * halfExtent.z);
		outwardNormal = PxVec3(0.0f, sign0, sign1);
	}
	else if(axis == 1)
	{
		endpoint0 = PxVec3(
			sign0 * halfExtent.x, -halfExtent.y,
			sign1 * halfExtent.z);
		endpoint1 = PxVec3(
			sign0 * halfExtent.x, halfExtent.y,
			sign1 * halfExtent.z);
		outwardNormal = PxVec3(sign0, 0.0f, sign1);
	}
	else
	{
		endpoint0 = PxVec3(
			sign0 * halfExtent.x, sign1 * halfExtent.y,
			-halfExtent.z);
		endpoint1 = PxVec3(
			sign0 * halfExtent.x, sign1 * halfExtent.y,
			halfExtent.z);
		outwardNormal = PxVec3(sign0, sign1, 0.0f);
	}
	outwardNormal.normalize();
}

PX_FORCE_INLINE bool avbdSegmentEnterExpandedBox(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& expandedHalfExtent,
	PxReal& entryTime, PxVec3& entryNormal)
{
	const PxVec3 direction = segmentEnd - segmentStart;
	entryTime = 0.0f;
	PxReal exitTime = 1.0f;
	entryNormal = PxVec3(0.0f);
	for(PxU32 axis = 0; axis < 3; axis++)
	{
		if(PxAbs(direction[axis]) <= 1e-12f)
		{
			if(segmentStart[axis] < -expandedHalfExtent[axis] ||
				segmentStart[axis] > expandedHalfExtent[axis])
				return false;
			continue;
		}
		const PxReal inverseDirection = 1.0f / direction[axis];
		PxReal nearTime =
			(-expandedHalfExtent[axis] - segmentStart[axis]) *
			inverseDirection;
		PxReal farTime =
			(expandedHalfExtent[axis] - segmentStart[axis]) *
			inverseDirection;
		PxReal nearNormalSign = -1.0f;
		if(nearTime > farTime)
		{
			const PxReal swapTime = nearTime;
			nearTime = farTime;
			farTime = swapTime;
			nearNormalSign = 1.0f;
		}
		if(nearTime > entryTime)
		{
			entryTime = nearTime;
			entryNormal = PxVec3(0.0f);
			entryNormal[axis] = nearNormalSign;
		}
		exitTime = PxMin(exitTime, farTime);
		if(entryTime > exitTime)
			return false;
	}
	return entryTime >= 0.0f && entryTime <= 1.0f &&
		entryNormal.magnitudeSquared() > 0.5f;
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_RIGID_BOX_GEOMETRY_H
