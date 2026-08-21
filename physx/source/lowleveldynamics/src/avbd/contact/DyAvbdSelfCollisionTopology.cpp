// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/contact/DyAvbdSelfCollisionTopology.h"

#include "foundation/PxSort.h"

namespace physx
{
namespace Dy
{

void avbdBuildSelfCollisionAdjacency(
	const AvbdSoftBody& softBody, AvbdSelfCollisionAdjacency& adjacency)
{
	adjacency.resize(softBody.compiled.particleCount);
	for(PxU32 i = 0; i < softBody.compiled.particleCount; ++i)
		adjacency[i].clear();

	const auto addAdjacentPair = [&](PxU32 localA, PxU32 localB)
	{
		adjacency[localA].pushBack(localB);
		adjacency[localB].pushBack(localA);
	};

	for(PxU32 i = 0; i + 3 < softBody.compiled.tetrahedra.size(); i += 4)
	{
		PxU32 vertices[4];
		for(PxU32 j = 0; j < 4; ++j)
			vertices[j] = softBody.compiled.tetrahedra[i + j];
		for(PxU32 a = 0; a < 4; ++a)
			for(PxU32 b = a + 1; b < 4; ++b)
				addAdjacentPair(vertices[a], vertices[b]);
	}
	for(PxU32 i = 0; i + 2 < softBody.compiled.triangles.size(); i += 3)
	{
		PxU32 vertices[3];
		for(PxU32 j = 0; j < 3; ++j)
			vertices[j] = softBody.compiled.triangles[i + j];
		for(PxU32 a = 0; a < 3; ++a)
			for(PxU32 b = a + 1; b < 3; ++b)
				addAdjacentPair(vertices[a], vertices[b]);
	}

	for(PxU32 i = 0; i < softBody.compiled.particleCount; ++i)
	{
		PxArray<PxU32>& neighbours = adjacency[i];
		if(neighbours.size() <= 1)
			continue;
		PxSort(neighbours.begin(), neighbours.size());
		PxU32 writeIndex = 1;
		for(PxU32 readIndex = 1; readIndex < neighbours.size(); ++readIndex)
		{
			if(neighbours[readIndex] != neighbours[readIndex - 1])
				neighbours[writeIndex++] = neighbours[readIndex];
		}
		neighbours.resize(writeIndex);
	}
}

void avbdBuildAllSelfCollisionAdjacencies(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSelfCollisionAdjacency>& adjacencies)
{
	adjacencies.resize(numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		avbdBuildSelfCollisionAdjacency(
			softBodies[bodyIndex], adjacencies[bodyIndex]);
}

} // namespace Dy
} // namespace physx
