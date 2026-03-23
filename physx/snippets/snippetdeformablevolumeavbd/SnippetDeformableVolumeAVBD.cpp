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

// ****************************************************************************
// SnippetDeformableVolumeAVBD
//
// CPU-only AVBD equivalent of SnippetDeformableVolume (GPU FEM).
// Demonstrates multiple VBD soft bodies -- a cube, a sphere, and a tall
// cube (cone substitute) -- dropping onto a rigid ground plane.  All elastic
// forces use Neo-Hookean energy via VBD; contacts (ground, soft-soft,
// soft-rigid) are enforced through AVBD adaptive penalty.
//
// Scene layout:
//   Body 0 : cuboid at (-1.8, 8.0, 0.0) -- tilted, falls onto sphere edge and spins
//   Body 1 : sphere at (-3.8, 2.0, 0.0) -- restored visual anchor for soft-soft collision
//   Body 2 : cone   at (-0.8,11.0, 1.2) -- glancing hit into the left stack
//   Body 3 : cuboid at ( 7.0, 4.2, 0.0) -- tilted on a narrow rigid box edge
//   Body 4 : cube   at ( 5.4, 8.8, 0.3) -- off-center follower amplifies body 3 rotation
//   Rigid  : box    at ( 7.6, 0.55,0.0) -- narrow step, uses SDF contact path
//
// No GPU or CUDA dependency -- runs entirely on the CPU.
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include "PxPhysicsAPI.h"
#include "PxAvbdSoftBody.h"
#include "extensions/PxTetMakerExt.h"

#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

#include "SnippetDeformableVolumeAVBD.h"

using namespace physx;
using namespace physx::Dy;

// ---------------------------------------------------------------------------
// Generate cone surface triangles, then use PxTetMaker conforming->voxel
// pipeline to produce a uniform voxel tet mesh.
// ---------------------------------------------------------------------------
static void rotateVerticesAroundZ(
	PxArray<PxVec3>& verts,
	const PxVec3& center,
	PxReal angle)
{
	const PxReal cs = PxCos(angle);
	const PxReal sn = PxSin(angle);
	for (PxU32 i = 0; i < verts.size(); i++)
	{
		const PxVec3 r = verts[i] - center;
		verts[i].x = center.x + r.x * cs - r.y * sn;
		verts[i].y = center.y + r.x * sn + r.y * cs;
	}
}

static void scaleVerticesAboutCenter(
	PxArray<PxVec3>& verts,
	const PxVec3& center,
	const PxVec3& scale)
{
	for (PxU32 i = 0; i < verts.size(); i++)
	{
		const PxVec3 r = verts[i] - center;
		verts[i] = center + PxVec3(r.x * scale.x, r.y * scale.y, r.z * scale.z);
	}
}

static void generateConeTetsViaTetMaker(
	const PxVec3& center, PxReal radius, PxReal height,
	PxU32 numVoxels,
	PxArray<PxVec3>& outVerts, PxArray<PxU32>& outTets)
{
	// Build a cone surface mesh (triangle fan base + lateral)
	const PxU32 N = 16; // ring segments
	PxArray<PxVec3> surfVerts;
	PxArray<PxU32>  surfTris;

	// vertex 0 = apex
	surfVerts.pushBack(center + PxVec3(0, height, 0));
	// vertices 1..N = base ring
	for (PxU32 i = 0; i < N; i++)
	{
		PxReal a = 2.0f * 3.14159265f * i / N;
		surfVerts.pushBack(center + PxVec3(radius * cosf(a), 0, radius * sinf(a)));
	}
	// vertex N+1 = base center
	surfVerts.pushBack(center);

	// Lateral triangles (apex -> ring[i] -> ring[i+1])
	for (PxU32 i = 0; i < N; i++)
	{
		surfTris.pushBack(0);
		surfTris.pushBack(1 + i);
		surfTris.pushBack(1 + (i + 1) % N);
	}
	// Base triangles (center -> ring[i+1] -> ring[i])
	for (PxU32 i = 0; i < N; i++)
	{
		surfTris.pushBack(N + 1);
		surfTris.pushBack(1 + (i + 1) % N);
		surfTris.pushBack(1 + i);
	}

	// Step 1: conforming tet mesh from surface
	PxArray<PxVec3> confVerts;
	PxArray<PxU32>  confTets;
	{
		PxSimpleTriangleMesh surfMesh;
		surfMesh.points.count  = surfVerts.size();
		surfMesh.points.data   = surfVerts.begin();
		surfMesh.points.stride = sizeof(PxVec3);
		surfMesh.triangles.count  = surfTris.size() / 3;
		surfMesh.triangles.data   = surfTris.begin();
		surfMesh.triangles.stride = sizeof(PxU32) * 3;

		if (!PxTetMaker::createConformingTetrahedronMesh(surfMesh, confVerts, confTets))
		{
			printf("TetMaker: conforming mesh failed, falling back to hand-made cone\n");
			avbdGenerateConeTets(center, radius, height, 4, outVerts, outTets);
			return;
		}
	}

	// Step 2: voxel tet mesh from the conforming mesh
	{
		PxTetrahedronMeshDesc meshDesc;
		meshDesc.points.count  = confVerts.size();
		meshDesc.points.data   = confVerts.begin();
		meshDesc.points.stride = sizeof(PxVec3);
		meshDesc.tetrahedrons.count  = confTets.size() / 4;
		meshDesc.tetrahedrons.data   = confTets.begin();
		meshDesc.tetrahedrons.stride = sizeof(PxU32) * 4;

		if (!PxTetMaker::createVoxelTetrahedronMesh(meshDesc, numVoxels,
				outVerts, outTets))
		{
			printf("TetMaker: voxel mesh failed, falling back to hand-made cone\n");
			avbdGenerateConeTets(center, radius, height, 4, outVerts, outTets);
			return;
		}
	}

	printf("TetMaker voxel cone: %u verts, %u tets\n",
		outVerts.size(), outTets.size() / 4);
}

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static PxDefaultAllocator      gAllocator;
static PxDefaultErrorCallback  gErrorCallback;
static PxFoundation*           gFoundation  = NULL;
static PxPhysics*              gPhysics     = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*             gMaterial    = NULL;
static PxPvd*                  gPvd         = NULL;

PxScene*                       gScene       = NULL;

PxArray<AvbdSoftParticle>      gParticles;
PxArray<AvbdSoftBody>          gSoftBodies;
PxArray<SoftBodyRenderData>    gSoftBodyRenderData;

static PxArray<AvbdSoftContact> gContacts;
static PxArray<AvbdRigidBox>     gRigidBoxes;

// ---------------------------------------------------------------------------
// Push AVBD soft-body surface triangles to PVD as debug geometry.
// ---------------------------------------------------------------------------
static void sendSoftBodiesToPvd()
{
	PxPvdSceneClient* pvdClient = gScene ? gScene->getScenePvdClient() : NULL;
	if (!pvdClient)
		return;

	static const PxU32 bodyColors[] = { 0xFFFF8000, 0xFF0080FF, 0xFF00FF80 };

	PxArray<PxDebugTriangle> tris;
	for (PxU32 b = 0; b < gSoftBodies.size(); b++)
	{
		const AvbdSoftBody& sb = gSoftBodies[b];
		const PxU32* idx = sb.surfaceTriangles.begin();
		const PxU32 numTris = sb.surfaceTriangles.size() / 3;
		const PxU32 color = bodyColors[b % (sizeof(bodyColors) / sizeof(bodyColors[0]))];

		for (PxU32 t = 0; t < numTris; t++)
		{
			const PxVec3& p0 = gParticles[idx[t * 3 + 0]].position;
			const PxVec3& p1 = gParticles[idx[t * 3 + 1]].position;
			const PxVec3& p2 = gParticles[idx[t * 3 + 2]].position;
			tris.pushBack(PxDebugTriangle(p0, p1, p2, color));
		}
	}

	if (tris.size())
		pvdClient->drawTriangles(tris.begin(), tris.size());
}

// ---------------------------------------------------------------------------
static AvbdOGCParams gOGCParams;

static void initOGCParams()
{
	gOGCParams.contactRadius    = 0.20f;
	gOGCParams.contactStiffness = 3e5f;
	gOGCParams.friction         = 0.35f;
}

// ---------------------------------------------------------------------------
static void updateRenderData()
{
	gSoftBodyRenderData.clear();
	for (PxU32 i = 0; i < gSoftBodies.size(); i++)
	{
		SoftBodyRenderData rd;
		rd.surfaceTriIndices = gSoftBodies[i].surfaceTriangles.begin();
		rd.numSurfaceTris    = gSoftBodies[i].surfaceTriangles.size() / 3;
		gSoftBodyRenderData.pushBack(rd);
	}
}

// ---------------------------------------------------------------------------
void initPhysics(bool /*interactive*/)
{
	initOGCParams();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

	gPhysics    = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	PxInitExtensions(*gPhysics, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity       = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher             = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader  = PxDefaultSimulationFilterShader;
	sceneDesc.solverType    = PxSolverType::eAVBD;
	gScene = gPhysics->createScene(sceneDesc);

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);

	// Ground plane
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*ground);

	// ------------------------------------------------------------------
	// Body 0: Tilted cuboid for visible soft-soft tumbling
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		const PxVec3 center(-1.8f, 8.0f, 0.0f);
		avbdGenerateSubdividedCubeTets(center, 1.0f, 4, verts, tets);
		scaleVerticesAboutCenter(verts, center, PxVec3(1.8f, 0.65f, 0.9f));
		rotateVerticesAroundZ(verts, center, -0.55f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 160.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 1: Sphere restored as the soft-soft support body
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		avbdGenerateSubdividedSphereTets(PxVec3(-3.8f, 2.0f, 0.0f), 1.8f, 4, verts, tets);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 130.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 2: Cone glancing into the left stack
	//   Uses PxTetMaker conforming->voxel pipeline for uniform voxel tets.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		generateConeTetsViaTetMaker(PxVec3(-0.8f, 11.0f, 1.2f), 0.8f, 3.0f, 14, verts, tets);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 100.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 3: Tilted cuboid (rigid-soft toppling rotation)
	//   Pre-rotated and offset on a narrow edge so rigid-soft torque is obvious.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		PxVec3 center(7.0f, 4.2f, 0.0f);
		avbdGenerateSubdividedCubeTets(center, 1.0f, 3, verts, tets);
		scaleVerticesAboutCenter(verts, center, PxVec3(1.7f, 0.7f, 0.9f));
		rotateVerticesAroundZ(verts, center, 0.95f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 160.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Body 4: Off-center follower that keeps Body 3 rotating after impact.
	// ------------------------------------------------------------------
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		const PxVec3 center(5.4f, 8.8f, 0.3f);
		avbdGenerateSubdividedCubeTets(center, 0.85f, 3, verts, tets);
		rotateVerticesAroundZ(verts, center, -0.28f);

		avbdCreateSoftBody(
			verts.begin(), verts.size(),
			tets.begin(), tets.size(),
			NULL, 0,
			2e5f, 0.3f, 120.0f, 0.015f, 0.0f, 0.01f,
			gParticles, gSoftBodies);
	}

	// ------------------------------------------------------------------
	// Rigid box obstacle (narrow support edge for Body 3)
	// ------------------------------------------------------------------
	{
		AvbdRigidBox rb;
		rb.center     = PxVec3(7.6f, 0.55f, 0.0f);
		rb.halfExtent = PxVec3(0.7f, 0.55f, 2.2f);
		rb.friction   = 0.5f;
		gRigidBoxes.pushBack(rb);

		// Also add as PxRigidStatic for rendering
		PxRigidStatic* rigidBox = gPhysics->createRigidStatic(
			PxTransform(PxVec3(7.6f, 0.55f, 0.0f)));
		PxRigidActorExt::createExclusiveShape(
			*rigidBox, PxBoxGeometry(0.7f, 0.55f, 2.2f), *gMaterial);
		gScene->addActor(*rigidBox);
	}

	updateRenderData();

	printf("SnippetDeformableVolumeAVBD: %u particles, %u soft bodies, %u rigid boxes\n",
		gParticles.size(), gSoftBodies.size(), gRigidBoxes.size());
}

// ---------------------------------------------------------------------------
// Contact re-detection callback for use inside avbdStepSoftBodies outer loop.
// Re-creates all ground + soft-soft contacts with fresh surface positions.
// ---------------------------------------------------------------------------
static void redetectContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, void* /*userData*/)
{
	avbdDetectAllOGCContacts(
		particles, numParticles,
		softBodies, numSoftBodies,
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		contacts, gOGCParams, 0.0f);
}

void stepPhysics(bool /*interactive*/)
{
	PxReal dt = 1.0f / 60.0f;

	// Initial contact detection: ground + soft-soft OGC + rigid-soft SDF.
	avbdDetectAllOGCContacts(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		gContacts, gOGCParams, 0.0f);

	// 8 outer iterations with contact re-detection between each.
	// Contacts are re-detected via callback so surface-point anchors
	// track the deforming geometry instead of going stale.
	avbdStepSoftBodies(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gContacts.begin(), gContacts.size(),
		dt, PxVec3(0.0f, -9.81f, 0.0f), 8, 20, 1000.0f,
		redetectContacts, &gContacts, NULL);

	gScene->simulate(dt);
	gScene->fetchResults(true);

	sendSoftBodiesToPvd();
}

void cleanupPhysics(bool /*interactive*/)
{
	gSoftBodyRenderData.reset();
	gContacts.reset();
	gRigidBoxes.reset();
	gSoftBodies.reset();
	gParticles.reset();

	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PxCloseExtensions();
	PX_RELEASE(gFoundation);

	printf("SnippetDeformableVolumeAVBD done.\n");
}

void keyPress(unsigned char /*key*/, const PxTransform& /*camera*/)
{
}

int snippetMain(int, const char*const*)
{
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	printf("SnippetDeformableVolumeAVBD: No render snippet, nothing to do.\n");
#endif

	return 0;
}
