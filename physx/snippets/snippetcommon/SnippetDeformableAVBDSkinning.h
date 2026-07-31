// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef PHYSX_SNIPPET_DEFORMABLE_AVBD_SKINNING_H
#define PHYSX_SNIPPET_DEFORMABLE_AVBD_SKINNING_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"

namespace Snippets
{

struct AvbdTriangleSkinningBinding
{
	physx::PxU32 indices[3];
	physx::PxVec3 weights;
};

struct AvbdTetrahedronSkinningBinding
{
	physx::PxU32 indices[4];
	physx::PxVec4 weights;
};

inline void appendTriangleSkinningPatch(
	const physx::PxU32 triangle[3], physx::PxU32 subdivisions,
	physx::PxArray<AvbdTriangleSkinningBinding>& bindings,
	physx::PxArray<physx::PxU32>& outputTriangles)
{
	using namespace physx;
	subdivisions = PxMax<PxU32>(subdivisions, 1);
	PxArray<PxU32> rowStarts;
	rowStarts.reserve(subdivisions + 1);
	for(PxU32 row = 0; row <= subdivisions; ++row)
	{
		rowStarts.pushBack(bindings.size());
		for(PxU32 column = 0;
			column <= subdivisions - row; ++column)
		{
			const PxReal weight1 =
				PxReal(row) / PxReal(subdivisions);
			const PxReal weight2 =
				PxReal(column) / PxReal(subdivisions);
			AvbdTriangleSkinningBinding binding;
			binding.indices[0] = triangle[0];
			binding.indices[1] = triangle[1];
			binding.indices[2] = triangle[2];
			binding.weights =
				PxVec3(1.0f - weight1 - weight2,
					weight1, weight2);
			bindings.pushBack(binding);
		}
	}

	for(PxU32 row = 0; row < subdivisions; ++row)
	{
		const PxU32 columns = subdivisions - row;
		for(PxU32 column = 0; column < columns; ++column)
		{
			const PxU32 a = rowStarts[row] + column;
			const PxU32 b = rowStarts[row + 1] + column;
			const PxU32 c = rowStarts[row] + column + 1;
			outputTriangles.pushBack(a);
			outputTriangles.pushBack(b);
			outputTriangles.pushBack(c);
			if(column + 1 < columns)
			{
				const PxU32 d =
					rowStarts[row + 1] + column + 1;
				outputTriangles.pushBack(b);
				outputTriangles.pushBack(d);
				outputTriangles.pushBack(c);
			}
		}
	}
}

inline void appendTetrahedronSurfaceSkinning(
	const physx::PxU32 tetrahedron[4], physx::PxU32 subdivisions,
	physx::PxArray<AvbdTetrahedronSkinningBinding>& bindings,
	physx::PxArray<physx::PxU32>& outputTriangles)
{
	using namespace physx;
	// The face order is outward for a positively wound tetrahedron. Rendering
	// correctness does not depend on winding because normals are accumulated
	// from the generated triangles, but a stable order keeps the demo
	// deterministic.
	static const PxU32 faces[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	subdivisions = PxMax<PxU32>(subdivisions, 1);
	for(PxU32 face = 0; face < 4; ++face)
	{
		PxArray<PxU32> rowStarts;
		rowStarts.reserve(subdivisions + 1);
		for(PxU32 row = 0; row <= subdivisions; ++row)
		{
			rowStarts.pushBack(bindings.size());
			for(PxU32 column = 0;
				column <= subdivisions - row; ++column)
			{
				const PxReal weight1 =
					PxReal(row) / PxReal(subdivisions);
				const PxReal weight2 =
					PxReal(column) / PxReal(subdivisions);
				AvbdTetrahedronSkinningBinding binding;
				binding.indices[0] = tetrahedron[0];
				binding.indices[1] = tetrahedron[1];
				binding.indices[2] = tetrahedron[2];
				binding.indices[3] = tetrahedron[3];
				binding.weights = PxVec4(0.0f);
				binding.weights[faces[face][0]] =
					1.0f - weight1 - weight2;
				binding.weights[faces[face][1]] = weight1;
				binding.weights[faces[face][2]] = weight2;
				bindings.pushBack(binding);
			}
		}

		for(PxU32 row = 0; row < subdivisions; ++row)
		{
			const PxU32 columns = subdivisions - row;
			for(PxU32 column = 0; column < columns; ++column)
			{
				const PxU32 a = rowStarts[row] + column;
				const PxU32 b = rowStarts[row + 1] + column;
				const PxU32 c = rowStarts[row] + column + 1;
				outputTriangles.pushBack(a);
				outputTriangles.pushBack(b);
				outputTriangles.pushBack(c);
				if(column + 1 < columns)
				{
					const PxU32 d =
						rowStarts[row + 1] + column + 1;
					outputTriangles.pushBack(b);
					outputTriangles.pushBack(d);
					outputTriangles.pushBack(c);
				}
			}
		}
	}
}

inline bool updateSkinningNormals(
	const physx::PxArray<physx::PxVec3>& positions,
	const physx::PxArray<physx::PxU32>& triangles,
	physx::PxArray<physx::PxVec3>& normals)
{
	using namespace physx;
	normals.resize(positions.size());
	for(PxU32 i = 0; i < normals.size(); ++i)
		normals[i] = PxVec3(0.0f);
	for(PxU32 i = 0; i + 2 < triangles.size(); i += 3)
	{
		const PxU32 i0 = triangles[i + 0];
		const PxU32 i1 = triangles[i + 1];
		const PxU32 i2 = triangles[i + 2];
		if(i0 >= positions.size() || i1 >= positions.size() ||
			i2 >= positions.size())
			return false;
		const PxVec3 normal =
			(positions[i1] - positions[i0]).cross(
				positions[i2] - positions[i0]);
		if(!normal.isFinite())
			return false;
		normals[i0] += normal;
		normals[i1] += normal;
		normals[i2] += normal;
	}
	for(PxU32 i = 0; i < normals.size(); ++i)
	{
		const PxReal magnitudeSquared = normals[i].magnitudeSquared();
		if(!PxIsFinite(magnitudeSquared))
			return false;
		normals[i] = magnitudeSquared > 1.0e-20f
			? normals[i] * PxRecipSqrt(magnitudeSquared)
			: PxVec3(0.0f, 1.0f, 0.0f);
	}
	return true;
}

inline bool evaluateTriangleSkinning(
	const physx::PxVec4* simulationPositions,
	physx::PxU32 simulationVertexCount,
	const physx::PxArray<AvbdTriangleSkinningBinding>& bindings,
	const physx::PxArray<physx::PxU32>& triangles,
	physx::PxArray<physx::PxVec3>& outputPositions,
	physx::PxArray<physx::PxVec3>& outputNormals)
{
	using namespace physx;
	if(!simulationPositions)
		return false;
	outputPositions.resize(bindings.size());
	for(PxU32 i = 0; i < bindings.size(); ++i)
	{
		const AvbdTriangleSkinningBinding& binding = bindings[i];
		if(binding.indices[0] >= simulationVertexCount ||
			binding.indices[1] >= simulationVertexCount ||
			binding.indices[2] >= simulationVertexCount)
			return false;
		const PxVec3 position =
			simulationPositions[binding.indices[0]].getXYZ() *
				binding.weights.x +
			simulationPositions[binding.indices[1]].getXYZ() *
				binding.weights.y +
			simulationPositions[binding.indices[2]].getXYZ() *
				binding.weights.z;
		if(!position.isFinite())
			return false;
		outputPositions[i] = position;
	}
	return updateSkinningNormals(
		outputPositions, triangles, outputNormals);
}

inline bool evaluateTetrahedronSkinning(
	const physx::PxVec4* simulationPositions,
	physx::PxU32 simulationVertexCount,
	const physx::PxArray<AvbdTetrahedronSkinningBinding>& bindings,
	const physx::PxArray<physx::PxU32>& triangles,
	physx::PxArray<physx::PxVec3>& outputPositions,
	physx::PxArray<physx::PxVec3>& outputNormals)
{
	using namespace physx;
	if(!simulationPositions)
		return false;
	outputPositions.resize(bindings.size());
	for(PxU32 i = 0; i < bindings.size(); ++i)
	{
		const AvbdTetrahedronSkinningBinding& binding = bindings[i];
		PxVec3 position(0.0f);
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			if(binding.indices[endpoint] >= simulationVertexCount)
				return false;
			position +=
				simulationPositions[binding.indices[endpoint]].
					getXYZ() *
				binding.weights[endpoint];
		}
		if(!position.isFinite())
			return false;
		outputPositions[i] = position;
	}
	return updateSkinningNormals(
		outputPositions, triangles, outputNormals);
}

} // namespace Snippets

#endif // PHYSX_SNIPPET_DEFORMABLE_AVBD_SKINNING_H
