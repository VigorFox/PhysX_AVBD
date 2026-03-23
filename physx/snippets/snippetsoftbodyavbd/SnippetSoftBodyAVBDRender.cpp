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

#include "SnippetSoftBodyAVBD.h"

using namespace physx;
using namespace physx::Dy;

extern void stepPhysics(bool interactive);
extern void keyPress(unsigned char key, const PxTransform& camera);

namespace
{
Snippets::Camera* sCamera;

// Extract the 4 surface triangles from each tet for rendering.
// Each tet (v0,v1,v2,v3) produces 4 faces: (0,2,1),(0,1,3),(0,3,2),(1,2,3).
static void buildTetSurfaceTriangles(
	const PxVec3* positions, PxU32 /*numParticles*/,
	const PxU32* tetIndices, PxU32 numTets,
	PxArray<PxVec3>& outVerts, PxArray<PxU32>& outTris,
	PxArray<PxVec3>& outNormals)
{
	outVerts.clear();
	outTris.clear();
	outNormals.clear();

	for (PxU32 t = 0; t < numTets; t++)
	{
		PxU32 i0 = tetIndices[t * 4 + 0];
		PxU32 i1 = tetIndices[t * 4 + 1];
		PxU32 i2 = tetIndices[t * 4 + 2];
		PxU32 i3 = tetIndices[t * 4 + 3];

		// Note: positions are interleaved in AvbdSoftParticle (stride = sizeof(AvbdSoftParticle))
		// We need to index through the real particle array.
		const PxVec3& p0 = *(const PxVec3*)((const char*)positions + i0 * sizeof(AvbdSoftParticle));
		const PxVec3& p1 = *(const PxVec3*)((const char*)positions + i1 * sizeof(AvbdSoftParticle));
		const PxVec3& p2 = *(const PxVec3*)((const char*)positions + i2 * sizeof(AvbdSoftParticle));
		const PxVec3& p3 = *(const PxVec3*)((const char*)positions + i3 * sizeof(AvbdSoftParticle));

		// 4 faces per tet, winding consistent with outward normals
		PxU32 faces[4][3] = { {0,2,1}, {0,1,3}, {0,3,2}, {1,2,3} };
		const PxVec3 pts[4] = { p0, p1, p2, p3 };

		for (int f = 0; f < 4; f++)
		{
			PxU32 a = faces[f][0], b = faces[f][1], c = faces[f][2];
			PxVec3 e1 = pts[b] - pts[a];
			PxVec3 e2 = pts[c] - pts[a];
			PxVec3 n = e1.cross(e2);
			PxReal len = n.magnitude();
			if (len > 1e-12f) n *= (1.0f / len);

			PxU32 base = outVerts.size();
			outVerts.pushBack(pts[a]);
			outVerts.pushBack(pts[b]);
			outVerts.pushBack(pts[c]);

			outNormals.pushBack(n);
			outNormals.pushBack(n);
			outNormals.pushBack(n);

			outTris.pushBack(base);
			outTris.pushBack(base + 1);
			outTris.pushBack(base + 2);
		}
	}
}

void renderCallback()
{
	// Check frame limit -- exit the current test when done
	if (gVisMaxFrames > 0 && gVisFrameCount >= gVisMaxFrames)
	{
		glutLeaveMainLoop();
		return;
	}

	// Update window title with progress
	if (gVisTestName && gVisFrameCount % 10 == 0)
	{
		char title[256];
		snprintf(title, sizeof(title), "AVBD - %s [%u/%u]",
		         gVisTestName, gVisFrameCount, gVisMaxFrames);
		glutSetWindowTitle(title);
	}

	stepPhysics(true);

	Snippets::startRender(sCamera);

	// Render rigid actors from the PhysX scene (ground plane etc.)
	PxU32 nbActors = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC);
	if (nbActors)
	{
		PxArray<PxRigidActor*> actors(nbActors);
		gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC,
			reinterpret_cast<PxActor**>(&actors[0]), nbActors);
		Snippets::renderActors(&actors[0], static_cast<PxU32>(actors.size()), true);
	}

	// Render soft bodies
	PxArray<PxVec3> triVerts;
	PxArray<PxU32>  triIndices;
	PxArray<PxVec3> triNormals;

	for (PxU32 i = 0; i < gSoftBodyRenderData.size(); i++)
	{
		const SoftBodyRenderData& rd = gSoftBodyRenderData[i];

		if (rd.tetIndices && rd.numTets > 0)
		{
			buildTetSurfaceTriangles(
				rd.positions, rd.numParticles,
				rd.tetIndices, rd.numTets,
				triVerts, triIndices, triNormals);

			Snippets::renderMesh(
				static_cast<PxU32>(triVerts.size()), triVerts.begin(),
				static_cast<PxU32>(triIndices.size() / 3), triIndices.begin(),
				PxVec3(0.2f, 0.6f, 0.9f),
				triNormals.begin());
		}
		else if (rd.triIndices && rd.numTris > 0)
		{
			// Cloth: build per-face normals from triangle connectivity
			triVerts.clear();
			triIndices.clear();
			triNormals.clear();

			for (PxU32 t = 0; t < rd.numTris; t++)
			{
				PxU32 a = rd.triIndices[t * 3 + 0];
				PxU32 b = rd.triIndices[t * 3 + 1];
				PxU32 c = rd.triIndices[t * 3 + 2];

				const PxVec3& pa = *(const PxVec3*)((const char*)rd.positions + a * sizeof(AvbdSoftParticle));
				const PxVec3& pb = *(const PxVec3*)((const char*)rd.positions + b * sizeof(AvbdSoftParticle));
				const PxVec3& pc = *(const PxVec3*)((const char*)rd.positions + c * sizeof(AvbdSoftParticle));

				PxVec3 n = (pb - pa).cross(pc - pa);
				PxReal len = n.magnitude();
				if (len > 1e-12f) n *= (1.0f / len);

				PxU32 base = triVerts.size();
				triVerts.pushBack(pa);
				triVerts.pushBack(pb);
				triVerts.pushBack(pc);
				triNormals.pushBack(n);
				triNormals.pushBack(n);
				triNormals.pushBack(n);
				triIndices.pushBack(base);
				triIndices.pushBack(base + 1);
				triIndices.pushBack(base + 2);
			}

			Snippets::renderMesh(
				static_cast<PxU32>(triVerts.size()), triVerts.begin(),
				static_cast<PxU32>(triIndices.size() / 3), triIndices.begin(),
				PxVec3(0.2f, 0.6f, 0.9f),
				triNormals.begin());
		}
	}

	Snippets::finishRender();
}

void exitCallback()
{
	delete sCamera;
	sCamera = NULL;
}
}

// ---------------------------------------------------------------------------
// Visual-mode render entry points (called from snippetMain)
// ---------------------------------------------------------------------------

void renderInit()
{
	sCamera = new Snippets::Camera(PxVec3(3.0f, 4.0f, 6.0f), PxVec3(-0.3f, -0.3f, -0.9f));
	Snippets::setupDefault("PhysX Snippet Soft Body AVBD", sCamera, keyPress, renderCallback, exitCallback);
}

void renderRun()
{
	glutMainLoop();
}
#endif
