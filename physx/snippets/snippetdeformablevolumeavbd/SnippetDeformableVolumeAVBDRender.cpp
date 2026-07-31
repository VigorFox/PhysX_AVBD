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
#include "DyAvbdSoftBodyComponent.h"

#include "../snippetrender/SnippetRender.h"
#include "../snippetrender/SnippetCamera.h"

#include "SnippetDeformableVolumeAVBD.h"

using namespace physx;
using namespace physx::Dy;

#ifndef AVBD_VOLUME_RENDER_TITLE
#define AVBD_VOLUME_RENDER_TITLE "PhysX Snippet Deformable Volume AVBD"
#endif

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

static void buildPublicVolumeSurfaceMesh(
	PxDeformableVolume& volume,
	PxArray<PxVec3>& outVerts, PxArray<PxU32>& outTris,
	PxArray<PxVec3>& outNormals)
{
	outVerts.clear();
	outTris.clear();
	outNormals.clear();
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferH();
	if(!mesh || !positions)
		return;
	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* tets16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* tets32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	static const PxU32 faces[4][3] =
	{
		{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}
	};
	for(PxU32 tet = 0; tet < mesh->getNbTetrahedrons(); ++tet)
	{
		PxU32 indices[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			indices[endpoint] = has16BitIndices
				? PxU32(tets16[4 * tet + endpoint])
				: tets32[4 * tet + endpoint];
		for(PxU32 face = 0; face < 4; ++face)
		{
			const PxVec3 p0 =
				positions[indices[faces[face][0]]].getXYZ();
			const PxVec3 p1 =
				positions[indices[faces[face][1]]].getXYZ();
			const PxVec3 p2 =
				positions[indices[faces[face][2]]].getXYZ();
			PxVec3 normal = (p1 - p0).cross(p2 - p0);
			const PxReal magnitude = normal.magnitude();
			if(magnitude > 1.0e-12f)
				normal *= 1.0f / magnitude;
			const PxU32 base = outVerts.size();
			outVerts.pushBack(p0);
			outVerts.pushBack(p1);
			outVerts.pushBack(p2);
			outNormals.pushBack(normal);
			outNormals.pushBack(normal);
			outNormals.pushBack(normal);
			outTris.pushBack(base);
			outTris.pushBack(base + 1);
			outTris.pushBack(base + 2);
		}
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

	if(gVolumeAvbdSkinningRenderData.positions &&
		gVolumeAvbdSkinningRenderData.triangles)
	{
		Snippets::renderMesh(
			gVolumeAvbdSkinningRenderData.numVertices,
			gVolumeAvbdSkinningRenderData.positions,
			gVolumeAvbdSkinningRenderData.numTriangles,
			gVolumeAvbdSkinningRenderData.triangles,
			PxVec3(0.95f, 0.34f, 0.16f),
			gVolumeAvbdSkinningRenderData.normals);
	}
	else
	{
		PxDeformableVolume* publicVolume =
			getPrimaryCpuAvbdVolume();
		if(publicVolume)
		{
			buildPublicVolumeSurfaceMesh(
				*publicVolume, triVerts, triIndices, triNormals);
			if(!triVerts.empty())
			{
				Snippets::renderMesh(
					triVerts.size(), triVerts.begin(),
					triIndices.size() / 3, triIndices.begin(),
					PxVec3(0.95f, 0.34f, 0.16f),
					triNormals.begin());
			}
		}
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

	Snippets::setupDefault(
		AVBD_VOLUME_RENDER_TITLE, sCamera, keyPress,
		renderCallback, exitCallback);

	initPhysics(true);
	glutMainLoop();
}
#endif
