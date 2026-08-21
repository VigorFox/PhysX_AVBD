// CPU AVBD stateless contact geometry helpers.
//
// This file intentionally contains no contact ownership, workspace, or solver
// state.  The closest-point result is shared by the triangle-surface detector
// and the remaining rigid/soft narrow-phase leaves.

#ifndef DY_AVBD_CONTACT_GEOMETRY_QUERIES_H
#define DY_AVBD_CONTACT_GEOMETRY_QUERIES_H

#include "avbd/contact/DyAvbdContactGeometry.h"
#include "foundation/PxMath.h"
#include "foundation/PxQuat.h"

namespace physx
{
namespace Dy
{

PX_FORCE_INLINE bool avbdAreSweepRotationsEquivalent(
	const PxQuat& startRotation, const PxQuat& endRotation,
	PxReal tolerance = 0.0f)
{
	if(!startRotation.isFinite() || !endRotation.isFinite())
		return false;
	const PxReal alignment = PxAbs(startRotation.dot(endRotation));
	return PxIsFinite(alignment) && alignment >= 1.0f - tolerance;
}

PX_FORCE_INLINE bool avbdGetSweepAngularDistance(
	const PxQuat& startRotation, const PxQuat& endRotation,
	PxReal& angularDistance)
{
	if(!startRotation.isFinite() || !endRotation.isFinite())
		return false;
	const PxReal startMagnitudeSq = startRotation.magnitudeSquared();
	const PxReal endMagnitudeSq = endRotation.magnitudeSquared();
	if(startMagnitudeSq <= 1.0e-12f || endMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(startMagnitudeSq) || !PxIsFinite(endMagnitudeSq))
		return false;
	const PxQuat normalizedStart = startRotation.getNormalized();
	const PxQuat normalizedEnd = endRotation.getNormalized();
	const PxReal alignment = PxClamp(
		PxAbs(normalizedStart.dot(normalizedEnd)), 0.0f, 1.0f);
	angularDistance = 2.0f * PxAcos(alignment);
	return PxIsFinite(angularDistance);
}

PX_FORCE_INLINE bool avbdSegmentEnterExpandedSphere(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& sphereCenter, PxReal expandedRadius,
	PxReal& entryTime, PxVec3& entryNormal)
{
	const PxVec3 direction = segmentEnd - segmentStart;
	const PxReal directionMagnitudeSq = direction.magnitudeSquared();
	if(directionMagnitudeSq <= 1e-12f || !PxIsFinite(directionMagnitudeSq) ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxVec3 startOffset = segmentStart - sphereCenter;
	const PxReal halfB = startOffset.dot(direction);
	const PxReal c = startOffset.magnitudeSquared() -
		expandedRadius * expandedRadius;
	if(c < 0.0f)
		return false;
	const PxReal discriminant =
		halfB * halfB - directionMagnitudeSq * c;
	if(discriminant < 0.0f || !PxIsFinite(discriminant))
		return false;
	entryTime = (-halfB - PxSqrt(discriminant)) /
		directionMagnitudeSq;
	if(entryTime < 0.0f || entryTime > 1.0f)
		return false;
	const PxVec3 entryOffset = startOffset + direction * entryTime;
	const PxReal entryMagnitudeSq = entryOffset.magnitudeSquared();
	if(entryMagnitudeSq <= 1e-12f || !PxIsFinite(entryMagnitudeSq))
		return false;
	entryNormal = entryOffset * PxRecipSqrt(entryMagnitudeSq);
	return true;
}

// Enhanced closest-point-on-triangle with feature classification.
inline AvbdClosestPointResult avbdClosestPointOnTriangleOGC(
	const PxVec3& p, const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	AvbdClosestPointResult result;
	PxVec3 ab = b - a, ac = c - a, ap = p - a;
	PxReal d1 = ab.dot(ap), d2 = ac.dot(ap);
	if(d1 <= 0.0f && d2 <= 0.0f)
	{
		result.point = a;
		result.feature = AVBD_FEATURE_VERTEX;
		result.featureIndex = 0;
		result.barycentric = PxVec3(1.0f, 0.0f, 0.0f);
		PxVec3 diff = p - a;
		result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f
			? diff * (1.0f / result.distance) : PxVec3(0, 1, 0);
		return result;
	}
	PxVec3 bp = p - b;
	PxReal d3 = ab.dot(bp), d4 = ac.dot(bp);
	if(d3 >= 0.0f && d4 <= d3)
	{
		result.point = b;
		result.feature = AVBD_FEATURE_VERTEX;
		result.featureIndex = 1;
		result.barycentric = PxVec3(0.0f, 1.0f, 0.0f);
		PxVec3 diff = p - b;
		result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f
			? diff * (1.0f / result.distance) : PxVec3(0, 1, 0);
		return result;
	}
	PxReal vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		PxReal v = d1 / (d1 - d3);
		result.point = a + ab * v;
		result.feature = AVBD_FEATURE_EDGE;
		result.featureIndex = 0;
		result.barycentric = PxVec3(1.0f - v, v, 0.0f);
		PxVec3 diff = p - result.point;
		result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f
			? diff * (1.0f / result.distance) : PxVec3(0, 1, 0);
		return result;
	}
	PxVec3 cp = p - c;
	PxReal d5 = ab.dot(cp), d6 = ac.dot(cp);
	if(d6 >= 0.0f && d5 <= d6)
	{
		result.point = c;
		result.feature = AVBD_FEATURE_VERTEX;
		result.featureIndex = 2;
		result.barycentric = PxVec3(0.0f, 0.0f, 1.0f);
		PxVec3 diff = p - c;
		result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f
			? diff * (1.0f / result.distance) : PxVec3(0, 1, 0);
		return result;
	}
	PxReal vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		PxReal w = d2 / (d2 - d6);
		result.point = a + ac * w;
		result.feature = AVBD_FEATURE_EDGE;
		result.featureIndex = 1;
		result.barycentric = PxVec3(1.0f - w, 0.0f, w);
		PxVec3 diff = p - result.point;
		result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f
			? diff * (1.0f / result.distance) : PxVec3(0, 1, 0);
		return result;
	}
	PxReal va = d3 * d6 - d5 * d4;
	if(va <= 0.0f && (d4 - d3) >= 0.0f &&
		(d5 - d6) >= 0.0f)
	{
		PxReal w = (d4 - d3) /
			((d4 - d3) + (d5 - d6));
		result.point = b + (c - b) * w;
		result.feature = AVBD_FEATURE_EDGE;
		result.featureIndex = 2;
		result.barycentric = PxVec3(0.0f, 1.0f - w, w);
		PxVec3 diff = p - result.point;
		result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f
			? diff * (1.0f / result.distance) : PxVec3(0, 1, 0);
		return result;
	}
	// Inside triangle.
	PxReal denom = 1.0f / (va + vb + vc);
	PxReal v = vb * denom;
	PxReal w = vc * denom;
	result.point = a + ab * v + ac * w;
	result.barycentric = PxVec3(1.0f - v - w, v, w);
	result.feature = AVBD_FEATURE_FACE;
	result.featureIndex = 0;
	PxVec3 diff = p - result.point;
	result.distance = diff.magnitude();
	if(result.distance > 1e-10f)
		result.normal = diff * (1.0f / result.distance);
	else
	{
		PxVec3 faceN = ab.cross(ac);
		PxReal fLen = faceN.magnitude();
		result.normal = fLen > 1e-10f
			? faceN * (1.0f / fLen) : PxVec3(0, 1, 0);
	}
	return result;
}

PX_FORCE_INLINE PxReal avbdGetRestPointTriangleDistance(
	const PxVec3& point, const PxVec3& vertex0,
	const PxVec3& vertex1, const PxVec3& vertex2)
{
	return avbdClosestPointOnTriangleOGC(
		point, vertex0, vertex1, vertex2).distance;
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_GEOMETRY_QUERIES_H
