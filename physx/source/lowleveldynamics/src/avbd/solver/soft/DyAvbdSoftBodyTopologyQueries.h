// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_TOPOLOGY_QUERIES_H
#define DY_AVBD_SOFT_BODY_TOPOLOGY_QUERIES_H

#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"

namespace physx
{
namespace Dy
{

PX_FORCE_INLINE const AvbdSoftBody* avbdFindSoftBodyForParticle(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 particleIndex)
{
	for(PxU32 i = 0; i < numSoftBodies; i++)
	{
		const AvbdSoftBody& body = softBodies[i];
		const PxU32 start = body.compiled.particleStart;
		if(particleIndex >= start &&
			particleIndex - start < body.compiled.particleCount)
			return &body;
	}
	return NULL;
}

PX_FORCE_INLINE bool avbdIsSoftBodySurfaceVertex(
	const AvbdSoftBody& body,
	PxU32 particleIndex)
{
	const PxArray<PxU32>& surfaceVertices =
		body.compiled.surfaceVertices;
	PxU32 lower = 0;
	PxU32 upper = surfaceVertices.size();
	while(lower < upper)
	{
		const PxU32 middle = lower + (upper - lower) / 2;
		if(surfaceVertices[middle] < particleIndex)
			lower = middle + 1;
		else
			upper = middle;
	}
	return lower < surfaceVertices.size() &&
		surfaceVertices[lower] == particleIndex;
}

PX_FORCE_INLINE bool avbdIsSelfRestVertexTriangleFiltered(
	const AvbdSoftBody& body, PxU32 localVertexIndex,
	PxU32 surfaceTriangleIndex)
{
	const PxArray<PxArray<PxU32> >& filteredTriangles =
		body.compiled.selfCollisionRestFilteredTriangles;
	if(!body.compiled.selfCollisionRestFilterCacheValid ||
		localVertexIndex >= filteredTriangles.size())
		return false;
	const PxArray<PxU32>& filteredForVertex =
		filteredTriangles[localVertexIndex];
	PxU32 lower = 0;
	PxU32 upper = filteredForVertex.size();
	while(lower < upper)
	{
		const PxU32 middle = lower + (upper - lower) / 2;
		if(filteredForVertex[middle] < surfaceTriangleIndex)
			lower = middle + 1;
		else
			upper = middle;
	}
	return lower < filteredForVertex.size() &&
		filteredForVertex[lower] == surfaceTriangleIndex;
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_TOPOLOGY_QUERIES_H
