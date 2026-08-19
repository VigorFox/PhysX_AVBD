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

#ifdef RENDER_SNIPPET

#include "SnippetDeformableSurfaceAVBD.h"

#include "../snippetrender/SnippetCamera.h"
#include "../snippetrender/SnippetRender.h"
#include "foundation/PxTime.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

using namespace physx;

#ifndef AVBD_SURFACE_RENDER_TITLE
#define AVBD_SURFACE_RENDER_TITLE \
	"PhysX Snippet Deformable Surface AVBD"
#endif

#ifndef AVBD_SURFACE_SNIPPET_NAME
#define AVBD_SURFACE_SNIPPET_NAME "SnippetDeformableSurfaceAVBD"
#endif

namespace
{

Snippets::Camera* sCamera = NULL;
PxU32 sFrameCount = 0;
PxU32 sMaxFrames = 0;
PxF64 sSimulationMs = 0.0;
PxF64 sRenderingMs = 0.0;

static PxU32 getVisualFrameLimit()
{
	const char* value = std::getenv("PHYSX_AVBD_VISUAL_MAX_FRAMES");
	if(!value || !value[0])
		return 0;

	errno = 0;
	char* end = NULL;
	const unsigned long parsed = std::strtoul(value, &end, 10);
	if(errno || end == value || *end != '\0' || parsed > PX_MAX_U32)
	{
		std::printf(
			"[AVBD_VISUAL_CONFIG_WARNING] invalid "
			"PHYSX_AVBD_VISUAL_MAX_FRAMES=%s; frame limit disabled\n",
			value);
		return 0;
	}
	return PxU32(parsed);
}

static void renderCallback()
{
	if(sMaxFrames && sFrameCount >= sMaxFrames)
	{
		printVisualParallelismTelemetry(sFrameCount);
		std::printf(
			"[AVBD_VISUAL_SMOKE_PASS] frames=%u simulationMs=%.3f "
			"simulationMsPerFrame=%.3f renderingMs=%.3f "
			"renderingMsPerFrame=%.3f\n",
			sFrameCount, sSimulationMs,
			sSimulationMs / PxF64(sFrameCount),
			sRenderingMs, sRenderingMs / PxF64(sFrameCount));
		glutLeaveMainLoop();
		return;
	}

	PxTime stageTimer;
	stepVisualPhysics();
	sSimulationMs +=
		stageTimer.getElapsedSeconds() * 1000.0;
	++sFrameCount;

	stageTimer = PxTime();
	Snippets::startRender(sCamera);

	if(gSurfaceAvbdScene)
	{
		const PxActorTypeFlags rigidTypes =
			PxActorTypeFlag::eRIGID_DYNAMIC |
			PxActorTypeFlag::eRIGID_STATIC;
		const PxU32 actorCount =
			gSurfaceAvbdScene->getNbActors(rigidTypes);
		if(actorCount)
		{
			PxArray<PxRigidActor*> actors(actorCount);
			gSurfaceAvbdScene->getActors(
				rigidTypes,
				reinterpret_cast<PxActor**>(actors.begin()),
				actorCount);
			Snippets::renderActors(
				actors.begin(), actorCount, true,
				PxVec3(0.65f, 0.72f, 0.82f));
		}
	}

	if(gSurfaceAvbdRenderData.skinnedPositions &&
		gSurfaceAvbdRenderData.skinnedTriangles)
	{
		Snippets::renderMesh(
			gSurfaceAvbdRenderData.skinnedVertexCount,
			gSurfaceAvbdRenderData.skinnedPositions,
			gSurfaceAvbdRenderData.skinnedTriangleCount,
			gSurfaceAvbdRenderData.skinnedTriangles,
			PxVec3(0.95f, 0.34f, 0.16f),
			gSurfaceAvbdRenderData.skinnedNormals);
	}
	else
	{
		PxTriangleMesh* mesh =
			gSurfaceAvbdRenderData.triangleMesh;
		const PxVec4* positions =
			gSurfaceAvbdRenderData.positionsInvMass;
		if(mesh && positions)
		{
			const bool has16BitIndices =
				mesh->getTriangleMeshFlags().isSet(
					PxTriangleMeshFlag::e16_BIT_INDICES);
			const PxVec3 clothColor(0.95f, 0.34f, 0.16f);
			Snippets::renderMesh(
				mesh->getNbVertices(), positions,
				mesh->getNbTriangles(), mesh->getTriangles(),
				has16BitIndices, clothColor, NULL, false, false);
		}
	}

	Snippets::showFPS();
	Snippets::finishRender();
	sRenderingMs +=
		stageTimer.getElapsedSeconds() * 1000.0;
}

static void releaseRenderResources()
{
	delete sCamera;
	sCamera = NULL;
	cleanupVisualPhysics();
}

static void exitCallback()
{
	releaseRenderResources();
}

} // namespace

void renderLoop()
{
	if(!initVisualPhysics())
	{
		std::printf(
			"%s: failed to initialize the visual scene.\n",
			AVBD_SURFACE_SNIPPET_NAME);
		cleanupVisualPhysics();
		return;
	}

	sFrameCount = 0;
	sMaxFrames = getVisualFrameLimit();
	sSimulationMs = 0.0;
	sRenderingMs = 0.0;
	sCamera = new Snippets::Camera(
		PxVec3(7.5f, 5.5f, 8.0f),
		PxVec3(-0.62f, -0.28f, -0.73f));
	Snippets::setupDefault(
		AVBD_SURFACE_RENDER_TITLE,
		sCamera, keyPress, renderCallback, exitCallback);
	Snippets::initFPS();
	glutMainLoop();
	releaseRenderResources();
}

#endif // RENDER_SNIPPET
