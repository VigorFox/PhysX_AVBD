// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef PHYSX_SNIPPET_DEFORMABLE_SURFACE_AVBD_H
#define PHYSX_SNIPPET_DEFORMABLE_SURFACE_AVBD_H

#include "PxPhysicsAPI.h"

struct SurfaceAvbdRenderData
{
	physx::PxVec4* positionsInvMass;
	physx::PxTriangleMesh* triangleMesh;
	const physx::PxVec3* skinnedPositions;
	const physx::PxVec3* skinnedNormals;
	const physx::PxU32* skinnedTriangles;
	physx::PxU32 skinnedVertexCount;
	physx::PxU32 skinnedTriangleCount;

	SurfaceAvbdRenderData()
		: positionsInvMass(NULL), triangleMesh(NULL),
		  skinnedPositions(NULL), skinnedNormals(NULL),
		  skinnedTriangles(NULL), skinnedVertexCount(0),
		  skinnedTriangleCount(0)
	{
	}
};

extern physx::PxScene* gSurfaceAvbdScene;
extern SurfaceAvbdRenderData gSurfaceAvbdRenderData;

bool initVisualPhysics();
void stepVisualPhysics();
void cleanupVisualPhysics();
void keyPress(unsigned char key, const physx::PxTransform& camera);

#ifdef RENDER_SNIPPET
// Emits post-fetch CPU AVBD scheduler telemetry for a bounded visual smoke
// run.  This is diagnostic only and never participates in simulation.
void printVisualParallelismTelemetry(physx::PxU32 expectedFrames);
#endif

#endif // PHYSX_SNIPPET_DEFORMABLE_SURFACE_AVBD_H
