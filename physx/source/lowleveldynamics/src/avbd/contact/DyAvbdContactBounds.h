// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

#ifndef DY_AVBD_CONTACT_BOUNDS_H
#define DY_AVBD_CONTACT_BOUNDS_H

#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftBody;
struct AvbdSoftParticle;

struct AvbdSelfCollisionTriangleBounds
{
	PxU32 triangleOffset;
	PxVec3 minimum;
	PxVec3 maximum;
};

struct AvbdSelfCollisionVertexSweepEntry
{
	PxU32 localIndex;
	PxReal minimumX;
	PxReal maximumX;
};

struct AvbdSelfCollisionEdgeBounds
{
	PxU32 edgeIndex;
	PxVec3 minimum;
	PxVec3 maximum;
};

struct AvbdSoftPairEdgeBounds
{
	PxU32 edgeIndex;
	PxVec3 minimum;
	PxVec3 maximum;
	PxVec3 adjacentNormal0;
	PxVec3 adjacentNormal1;
	bool hasExteriorNormalCone;
};

PX_FORCE_INLINE bool avbdIsDirectionInSurfaceEdgeNormalCone(
	const PxVec3& direction,
	const PxVec3& adjacentNormal0,
	const PxVec3& adjacentNormal1)
{
	const PxReal directionLengthSq = direction.magnitudeSquared();
	const PxReal normalLengthSq0 = adjacentNormal0.magnitudeSquared();
	const PxReal normalLengthSq1 = adjacentNormal1.magnitudeSquared();
	if(directionLengthSq <= 1.0e-12f ||
		normalLengthSq0 <= 1.0e-12f || normalLengthSq1 <= 1.0e-12f)
		return true;

	const PxVec3 unitDirection =
		direction * PxRecipSqrt(directionLengthSq);
	const PxVec3 normal0 =
		adjacentNormal0 * PxRecipSqrt(normalLengthSq0);
	const PxVec3 normal1 =
		adjacentNormal1 * PxRecipSqrt(normalLengthSq1);
	const PxReal normalDot = PxClamp(normal0.dot(normal1), -1.0f, 1.0f);
	const PxReal determinant = 1.0f - normalDot * normalDot;
	if(determinant <= 1.0e-6f)
		return true;

	const PxReal directionDot0 = unitDirection.dot(normal0);
	const PxReal directionDot1 = unitDirection.dot(normal1);
	const PxReal coefficient0 =
		(directionDot0 - normalDot * directionDot1) / determinant;
	const PxReal coefficient1 =
		(directionDot1 - normalDot * directionDot0) / determinant;
	const PxReal coefficientTolerance = 1.0e-3f;
	if(coefficient0 < -coefficientTolerance ||
		coefficient1 < -coefficientTolerance)
		return false;

	const PxVec3 reconstructed =
		normal0 * coefficient0 + normal1 * coefficient1;
	return (unitDirection - reconstructed).magnitudeSquared() <= 1.0e-3f;
}

struct AvbdSoftBodyBounds
{
	PxVec3 currentMinimum;
	PxVec3 currentMaximum;
	PxVec3 sweptMinimum;
	PxVec3 sweptMaximum;

	AvbdSoftBodyBounds()
		: currentMinimum(PX_MAX_F32), currentMaximum(-PX_MAX_F32),
		  sweptMinimum(PX_MAX_F32), sweptMaximum(-PX_MAX_F32)
	{
	}
};

void avbdComputeSoftBodyBounds(
	const AvbdSoftParticle* particles, const AvbdSoftBody& body,
	AvbdSoftBodyBounds& bounds);

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_BOUNDS_H
