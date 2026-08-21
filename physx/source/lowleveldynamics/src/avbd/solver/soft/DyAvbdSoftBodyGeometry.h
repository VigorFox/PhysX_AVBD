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
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef DY_AVBD_SOFT_BODY_GEOMETRY_H
#define DY_AVBD_SOFT_BODY_GEOMETRY_H

#include "foundation/PxArray.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

// =============================================================================
// Mesh generators
// =============================================================================

inline void avbdGenerateCubeTets(
	PxVec3 center, PxReal halfSize,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	PxReal h = halfSize;
	outVerts.clear();
	outVerts.pushBack(center + PxVec3(-h, -h, -h));
	outVerts.pushBack(center + PxVec3( h, -h, -h));
	outVerts.pushBack(center + PxVec3( h,  h, -h));
	outVerts.pushBack(center + PxVec3(-h,  h, -h));
	outVerts.pushBack(center + PxVec3(-h, -h,  h));
	outVerts.pushBack(center + PxVec3( h, -h,  h));
	outVerts.pushBack(center + PxVec3( h,  h,  h));
	outVerts.pushBack(center + PxVec3(-h,  h,  h));

	outTets.clear();
	PxU32 tets[] = { 0,1,3,4, 1,2,3,6, 3,4,6,7, 1,4,5,6, 1,3,4,6 };
	for (PxU32 i = 0; i < 20; i++)
		outTets.pushBack(tets[i]);
}

inline void avbdGenerateSubdividedCubeTets(
	PxVec3 center, PxReal halfSize, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	outVerts.clear();
	outTets.clear();
	PxReal cellSize = 2.0f * halfSize / PxReal(N);
	PxVec3 origin = center - PxVec3(halfSize, halfSize, halfSize);

	for (int iz = 0; iz <= N; iz++)
		for (int iy = 0; iy <= N; iy++)
			for (int ix = 0; ix <= N; ix++)
				outVerts.pushBack(origin + PxVec3(PxReal(ix) * cellSize,
				                                  PxReal(iy) * cellSize,
				                                  PxReal(iz) * cellSize));

	for (int iz = 0; iz < N; iz++)
		for (int iy = 0; iy < N; iy++)
			for (int ix = 0; ix < N; ix++)
			{
				PxU32 v[8];
				v[0] = PxU32(iz * (N+1) * (N+1) + iy * (N+1) + ix);
				v[1] = v[0] + 1;
				v[2] = v[0] + PxU32(N+1) + 1;
				v[3] = v[0] + PxU32(N+1);
				v[4] = v[0] + PxU32((N+1) * (N+1));
				v[5] = v[4] + 1;
				v[6] = v[4] + PxU32(N+1) + 1;
				v[7] = v[4] + PxU32(N+1);

				PxU32 t[] = {
					v[0],v[1],v[3],v[4], v[1],v[2],v[3],v[6],
					v[3],v[4],v[6],v[7], v[1],v[4],v[5],v[6],
					v[1],v[3],v[4],v[6]
				};
				for (PxU32 i = 0; i < 20; i++)
					outTets.pushBack(t[i]);
			}
}

inline void avbdGenerateClothGrid(
	PxVec3 center, PxReal sizeX, PxReal sizeZ,
	int M, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTris)
{
	outVerts.clear();
	outTris.clear();
	PxReal dx = sizeX / PxReal(M - 1);
	PxReal dz = sizeZ / PxReal(N - 1);
	PxVec3 origin = center - PxVec3(sizeX * 0.5f, 0.0f, sizeZ * 0.5f);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < M; i++)
			outVerts.pushBack(origin + PxVec3(PxReal(i) * dx, 0.0f, PxReal(j) * dz));

	for (int j = 0; j < N - 1; j++)
		for (int i = 0; i < M - 1; i++)
		{
			PxU32 v00 = PxU32(j * M + i);
			PxU32 v10 = v00 + 1;
			PxU32 v01 = v00 + PxU32(M);
			PxU32 v11 = v01 + 1;
			outTris.pushBack(v00); outTris.pushBack(v10); outTris.pushBack(v01);
			outTris.pushBack(v10); outTris.pushBack(v11); outTris.pushBack(v01);
		}
}

inline void avbdGenerateSubdividedSphereTets(
	PxVec3 center, PxReal radius, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	// Generate a subdivided cube, then map vertices proportionally onto a sphere.
	// Each vertex keeps its fractional distance from center to cube surface,
	// but the direction is spherically normalized.  This avoids collapsing
	// multiple interior vertices onto the same surface point (which would
	// create degenerate zero-volume tetrahedra).
	avbdGenerateSubdividedCubeTets(center, radius, N, outVerts, outTets);

	for (PxU32 i = 0; i < outVerts.size(); i++)
	{
		PxVec3 d = outVerts[i] - center;
		PxReal len = d.magnitude();
		if (len > 1e-8f)
		{
			// Distance from center to cube surface in direction d:
			//   cubeSurfR = halfSize * len / max(|dx|,|dy|,|dz|)
			// Fraction of the way from center to cube surface:
			//   frac = len / cubeSurfR = max(|dx|,|dy|,|dz|) / halfSize
			// Map to sphere: new distance = frac * radius
			PxReal maxAbs = PxMax(PxAbs(d.x), PxMax(PxAbs(d.y), PxAbs(d.z)));
			PxReal frac = maxAbs / radius;  // 0 at center, 1 at cube face
			outVerts[i] = center + d * (1.0f / len) * (frac * radius);
		}
	}

	// Fix tet orientation after the non-linear mapping (some tets may invert)
	for (PxU32 t = 0; t + 3 < outTets.size(); t += 4)
	{
		PxVec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
		PxVec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
		PxVec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
		if (e1.dot(e2.cross(e3)) < 0.0f)
		{
			PxU32 tmp = outTets[t+1]; outTets[t+1] = outTets[t+2]; outTets[t+2] = tmp;
		}
	}
}

// Generate a cone-shaped tet mesh directly from layered rings + apex.
// Base center at `center`, base radius `radius`, height along +Y.
inline void avbdGenerateConeTets(
	PxVec3 center, PxReal radius, PxReal height, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	outVerts.clear();
	outTets.clear();

	const int nLayers = PxMax(N, 2);
	const int nRing   = PxMax(4 * N, 8);
	const PxReal pi2  = 2.0f * 3.14159265358979f;

	// --- vertices ---
	// Each layer i (0..nLayers-1): 1 center + nRing ring vertices
	// Final vertex: apex
	for (int i = 0; i < nLayers; i++)
	{
		PxReal t = PxReal(i) / PxReal(nLayers); // 0 = base, approaches 1 near tip
		PxReal h = t * height;
		PxReal r = radius * (1.0f - t);

		// Center of this layer
		outVerts.pushBack(center + PxVec3(0.0f, h, 0.0f));

		// Ring vertices
		for (int j = 0; j < nRing; j++)
		{
			PxReal angle = pi2 * PxReal(j) / PxReal(nRing);
			outVerts.pushBack(center + PxVec3(r * PxCos(angle), h, r * PxSin(angle)));
		}
	}

	PxU32 apexIdx = outVerts.size();
	outVerts.pushBack(center + PxVec3(0.0f, height, 0.0f));

	// --- helper lambdas ---
	const int stride = 1 + nRing;
	// center vertex of layer i
	auto ci = [stride](int layer) -> PxU32 { return PxU32(layer * stride); };
	// ring vertex j of layer i (j wraps around)
	auto ri = [stride, nRing](int layer, int j) -> PxU32
	{
		return PxU32(layer * stride + 1 + ((j % nRing + nRing) % nRing));
	};

	// --- tets between adjacent layers (prism decomposition) ---
	for (int i = 0; i + 1 < nLayers; i++)
	{
		for (int j = 0; j < nRing; j++)
		{
			// 3 tets per wedge (triangular prism between two ring segments)
			outTets.pushBack(ci(i));   outTets.pushBack(ri(i, j+1)); outTets.pushBack(ri(i, j));   outTets.pushBack(ci(i+1));
			outTets.pushBack(ri(i,j)); outTets.pushBack(ri(i,j+1)); outTets.pushBack(ci(i+1));     outTets.pushBack(ri(i+1,j));
			outTets.pushBack(ri(i,j+1)); outTets.pushBack(ci(i+1)); outTets.pushBack(ri(i+1,j));   outTets.pushBack(ri(i+1,j+1));
		}
	}

	// --- apex cap (connect top layer to apex) ---
	{
		int top = nLayers - 1;
		for (int j = 0; j < nRing; j++)
		{
			outTets.pushBack(ci(top)); outTets.pushBack(ri(top, j+1)); outTets.pushBack(ri(top, j)); outTets.pushBack(apexIdx);
		}
	}

	// --- fix orientation: ensure positive tet volume ---
	for (PxU32 t = 0; t + 3 < outTets.size(); t += 4)
	{
		PxVec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
		PxVec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
		PxVec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
		if (e1.dot(e2.cross(e3)) < 0.0f)
		{
			PxU32 tmp = outTets[t+1]; outTets[t+1] = outTets[t+2]; outTets[t+2] = tmp;
		}
	}
}

// =============================================================================
// NOTE: For production tet mesh generation from arbitrary shapes, use the
// PhysX TetMaker API (PxTetMaker::createConformingTetrahedronMesh +
// PxTetMaker::createVoxelTetrahedronMesh) which provides BVH-based surface
// projection, boundary cell subdivision, and iterative relaxation.
// See extensions/PxTetMakerExt.h.
// =============================================================================


}
}

#endif // DY_AVBD_SOFT_BODY_GEOMETRY_H
