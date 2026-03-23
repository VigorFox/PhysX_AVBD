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
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifdef RENDER_SNIPPET

#include "PxPhysicsAPI.h"
#include "PxAvbdSoftBody.h"

#include "../snippetrender/SnippetRender.h"
#include "../snippetrender/SnippetCamera.h"

#include "SnippetDeformableVolumeAVBD.h"

using namespace physx;
using namespace physx::Dy;

extern void initPhysics(bool interactive);
extern void stepPhysics(bool interactive);
extern void cleanupPhysics(bool interactive);
extern void keyPress(unsigned char key, const PxTransform& camera);

namespace
{
Snippets::Camera* sCamera;

static void buildSurfaceMesh(
	const AvbdSoftParticle* particles,
	const PxU32* surfTriIndices, PxU32 numSurfTris,
	PxArray<PxVec3>& outVerts, PxArray<PxU32>& outTris,
	PxArray<PxVec3>& outNormals)
{
	outVerts.clear();
	outTris.clear();
	outNormals.clear();

	for (PxU32 t = 0; t < numSurfTris; t++)
	{
		const PxVec3& p0 = particles[surfTriIndices[t * 3 + 0]].position;
		const PxVec3& p1 = particles[surfTriIndices[t * 3 + 1]].position;
		const PxVec3& p2 = particles[surfTriIndices[t * 3 + 2]].position;

		PxVec3 n = (p1 - p0).cross(p2 - p0);
		PxReal len = n.magnitude();
		if (len > 1e-12f) n *= (1.0f / len);

		PxU32 base = outVerts.size();
		outVerts.pushBack(p0);
		outVerts.pushBack(p1);
		outVerts.pushBack(p2);

		outNormals.pushBack(n);
		outNormals.pushBack(n);
		outNormals.pushBack(n);

		outTris.pushBack(base);
		outTris.pushBack(base + 1);
		outTris.pushBack(base + 2);
	}
}

void renderCallback()
{
	stepPhysics(true);

	Snippets::startRender(sCamera);

	// Render rigid actors (ground plane)
	PxU32 nbActors = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC);
	if (nbActors)
	{
		PxArray<PxRigidActor*> actors(nbActors);
		gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC,
			reinterpret_cast<PxActor**>(&actors[0]), nbActors);
		Snippets::renderActors(&actors[0], static_cast<PxU32>(actors.size()), true);
	}

	// Render soft bodies
	const PxVec3 colors[] = {
		PxVec3(1.0f, 0.5f, 0.25f),  // orange  (cube, offset)
		PxVec3(0.45f, 0.6f, 0.75f), // blue    (sphere)
		PxVec3(0.6f, 0.9f, 0.4f),   // green   (cone, offset)
		PxVec3(0.9f, 0.85f, 0.2f),  // yellow  (tilted cube)
		PxVec3(0.8f, 0.3f, 0.7f)    // magenta (falling cube)
	};

	PxArray<PxVec3> triVerts;
	PxArray<PxU32>  triIndices;
	PxArray<PxVec3> triNormals;

	for (PxU32 i = 0; i < gSoftBodyRenderData.size(); i++)
	{
		const SoftBodyRenderData& rd = gSoftBodyRenderData[i];
		if (!rd.surfaceTriIndices || rd.numSurfaceTris == 0) continue;

		buildSurfaceMesh(
			gParticles.begin(), rd.surfaceTriIndices, rd.numSurfaceTris,
			triVerts, triIndices, triNormals);

		PxVec3 color = colors[i % 5];

		Snippets::renderMesh(
			static_cast<PxU32>(triVerts.size()), triVerts.begin(),
			static_cast<PxU32>(triIndices.size() / 3), triIndices.begin(),
			color, triNormals.begin());
	}

	Snippets::finishRender();
}

void exitCallback()
{
	delete sCamera;
	cleanupPhysics(true);
}
}

void renderLoop()
{
	sCamera = new Snippets::Camera(PxVec3(10.0f, 10.0f, 10.0f), PxVec3(-0.6f, -0.2f, -0.7f));

	Snippets::setupDefault("PhysX Snippet Deformable Volume AVBD", sCamera, keyPress, renderCallback, exitCallback);

	initPhysics(true);
	glutMainLoop();
}
#endif
