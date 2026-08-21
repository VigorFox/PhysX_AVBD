// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_FEATURE_GEOMETRY_H
#define DY_AVBD_CONTACT_FEATURE_GEOMETRY_H

#include "avbd/contact/DyAvbdContactGeometry.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_FEATURE_GEOMETRY_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_FEATURE_GEOMETRY_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_FEATURE_GEOMETRY_API
#endif

PX_FORCE_INLINE PxU64 avbdGetRigidSoftFeatureKey(
	PxU32 tag, PxU32 vertex0, PxU32 vertex1,
	PxU32 vertex2, PxU32 rigidFeatureIndex)
{
	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, tag);
	hash = avbdSoftContactHashValue(hash, vertex0);
	hash = avbdSoftContactHashValue(hash, vertex1);
	hash = avbdSoftContactHashValue(hash, vertex2);
	return avbdSoftContactHashValue(hash, rigidFeatureIndex);
}

PX_FORCE_INLINE void avbdClosestPointsOnSegments(
	const PxVec3& p0, const PxVec3& p1,
	const PxVec3& q0, const PxVec3& q1,
	PxReal& pWeight1, PxReal& qWeight1,
	PxVec3& pClosest, PxVec3& qClosest)
{
	const PxVec3 dP = p1 - p0;
	const PxVec3 dQ = q1 - q0;
	const PxVec3 r = p0 - q0;
	const PxReal a = dP.dot(dP);
	const PxReal e = dQ.dot(dQ);
	const PxReal epsilon = 1e-12f;

	if(a <= epsilon && e <= epsilon)
	{
		pWeight1 = qWeight1 = 0.0f;
		pClosest = p0;
		qClosest = q0;
		return;
	}
	if(a <= epsilon)
	{
		pWeight1 = 0.0f;
		qWeight1 = PxClamp(dQ.dot(-r) / e, 0.0f, 1.0f);
	}
	else
	{
		const PxReal c = dP.dot(r);
		if(e <= epsilon)
		{
			qWeight1 = 0.0f;
			pWeight1 = PxClamp(-c / a, 0.0f, 1.0f);
		}
		else
		{
			const PxReal b = dP.dot(dQ);
			const PxReal f = dQ.dot(r);
			const PxReal denominator = a * e - b * b;
			pWeight1 = denominator > epsilon
				? PxClamp((b * f - c * e) / denominator, 0.0f, 1.0f)
				: 0.0f;
			qWeight1 = (b * pWeight1 + f) / e;
			if(qWeight1 < 0.0f)
			{
				qWeight1 = 0.0f;
				pWeight1 = PxClamp(-c / a, 0.0f, 1.0f);
			}
			else if(qWeight1 > 1.0f)
			{
				qWeight1 = 1.0f;
				pWeight1 = PxClamp((b - c) / a, 0.0f, 1.0f);
			}
		}
	}
	pClosest = p0 + dP * pWeight1;
	qClosest = q0 + dQ * qWeight1;
}

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdRotatingPointEnterExpandedDeformingTriangleFace(
	const PxVec3& rigidLocalPoint,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal margin, AvbdSweptTriangleEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdRotatingPointEnterExpandedTriangleFace(
	const PxVec3& rigidLocalPoint,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal margin, AvbdSweptTriangleEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdTranslatedSegmentEnterExpandedSegmentInteriors(
	const PxVec3& rigid0, const PxVec3& rigid1,
	const PxVec3& rigidTranslation,
	const PxVec3& soft0, const PxVec3& soft1,
	PxReal margin, AvbdSweptConvexEdgeEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdRotatingSegmentEnterExpandedSegmentInteriors(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& soft0, const PxVec3& soft1,
	PxReal margin, AvbdSweptConvexEdgeEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& soft0Start, const PxVec3& soft1Start,
	const PxVec3& soft0End, const PxVec3& soft1End,
	PxReal margin, AvbdSweptConvexEdgeEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdDeformingSegmentsEnterExpandedInteriors(
	const PxVec3& query0Start, const PxVec3& query1Start,
	const PxVec3& query0End, const PxVec3& query1End,
	const PxVec3& target0Start, const PxVec3& target1Start,
	const PxVec3& target0End, const PxVec3& target1End,
	PxReal margin, AvbdSweptConvexEdgeEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius, AvbdSweptTriangleEntry& result);

DY_AVBD_CONTACT_FEATURE_GEOMETRY_API bool
avbdLinearPointEnterExpandedDeformingTriangleNonVertex(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal expandedRadius, AvbdSweptTriangleEntry& result);

#undef DY_AVBD_CONTACT_FEATURE_GEOMETRY_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_FEATURE_GEOMETRY_H
