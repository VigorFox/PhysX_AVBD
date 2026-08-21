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

#ifndef DY_AVBD_CONTACT_FEATURE_H
#define DY_AVBD_CONTACT_FEATURE_H

#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

// Closest-point feature type used by OGC contact classification. Keeping the
// identity/hash helpers beside the contact IR prevents detector code from
// depending on the monolithic soft-body component header.
enum AvbdClosestFeature
{
	AVBD_FEATURE_FACE,
	AVBD_FEATURE_EDGE,
	AVBD_FEATURE_VERTEX,
	AVBD_FEATURE_UNKNOWN
};

PX_FORCE_INLINE PxU64 avbdSoftContactHashValue(PxU64 hash, PxU32 value)
{
	hash ^= PxU64(value);
	return hash * 1099511628211ull;
}

PX_FORCE_INLINE PxU64 avbdSoftTrianglePrimitiveKey(
	PxU32 vertex0, PxU32 vertex1, PxU32 vertex2)
{
	if(vertex1 < vertex0)
	{
		const PxU32 tmp = vertex0;
		vertex0 = vertex1;
		vertex1 = tmp;
	}
	if(vertex2 < vertex1)
	{
		const PxU32 tmp = vertex1;
		vertex1 = vertex2;
		vertex2 = tmp;
	}
	if(vertex1 < vertex0)
	{
		const PxU32 tmp = vertex0;
		vertex0 = vertex1;
		vertex1 = tmp;
	}

	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, vertex0);
	hash = avbdSoftContactHashValue(hash, vertex1);
	return avbdSoftContactHashValue(hash, vertex2);
}

PX_FORCE_INLINE PxU64 avbdSoftTriangleFeatureKey(
	PxU32 vertex0, PxU32 vertex1, PxU32 vertex2,
	AvbdClosestFeature feature, PxU32 localFeatureIndex)
{
	PxU32 featureVertex0 = vertex0;
	PxU32 featureVertex1 = vertex1;
	PxU32 featureVertex2 = vertex2;
	PxU32 featureVertexCount = 3;

	if(feature == AVBD_FEATURE_VERTEX)
	{
		const PxU32 vertices[3] = {vertex0, vertex1, vertex2};
		featureVertex0 = vertices[PxMin(localFeatureIndex, 2u)];
		featureVertexCount = 1;
	}
	else if(feature == AVBD_FEATURE_EDGE)
	{
		if(localFeatureIndex == 0)
		{
			featureVertex0 = vertex0;
			featureVertex1 = vertex1;
		}
		else if(localFeatureIndex == 1)
		{
			featureVertex0 = vertex0;
			featureVertex1 = vertex2;
		}
		else
		{
			featureVertex0 = vertex1;
			featureVertex1 = vertex2;
		}
		if(featureVertex1 < featureVertex0)
		{
			const PxU32 tmp = featureVertex0;
			featureVertex0 = featureVertex1;
			featureVertex1 = tmp;
		}
		featureVertexCount = 2;
	}
	else
	{
		if(featureVertex1 < featureVertex0)
		{
			const PxU32 tmp = featureVertex0;
			featureVertex0 = featureVertex1;
			featureVertex1 = tmp;
		}
		if(featureVertex2 < featureVertex1)
		{
			const PxU32 tmp = featureVertex1;
			featureVertex1 = featureVertex2;
			featureVertex2 = tmp;
		}
		if(featureVertex1 < featureVertex0)
		{
			const PxU32 tmp = featureVertex0;
			featureVertex0 = featureVertex1;
			featureVertex1 = tmp;
		}
	}

	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, PxU32(feature));
	hash = avbdSoftContactHashValue(hash, featureVertex0);
	if(featureVertexCount > 1)
		hash = avbdSoftContactHashValue(hash, featureVertex1);
	if(featureVertexCount > 2)
		hash = avbdSoftContactHashValue(hash, featureVertex2);
	return hash;
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_FEATURE_H
