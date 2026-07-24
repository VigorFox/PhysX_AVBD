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

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

#include "SnippetDeformableVolumeAVBD.h"

#include <cfloat>
#include <cstring>
#include <string>

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
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*           gFoundation  = NULL;
static PxPhysics*              gPhysics     = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*             gMaterial    = NULL;
static PxPvd*                  gPvd         = NULL;
static bool                    gExtensionsInitialized = false;
static Snippets::HeadlessOptions gHeadlessOptions;

PxScene*                       gScene       = NULL;

PxArray<AvbdSoftParticle>      gParticles;
PxArray<AvbdSoftBody>          gSoftBodies;
PxArray<SoftBodyRenderData>    gSoftBodyRenderData;

static PxArray<AvbdSoftContact> gContacts;
static PxArray<AvbdRigidBox>     gRigidBoxes;

struct DeformableVolumeMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFiniteParticleSamples;
	PxU32 invertedElementSamples;
	PxU32 firstInversionFrame;
	PxU32 firstInversionBody;
	PxU32 firstInversionElement;
	PxU32 invertedBodiesMask;
	PxU32 particles;
	PxU32 softBodies;
	PxU32 tetElements;
	PxU32 surfaceTriangles;
	PxU32 rigidBoxes;
	PxU32 sceneStatics;
	PxU32 sceneDynamics;
	PxU32 sceneDeformableVolumes;
	PxU32 groundContactFrames;
	PxU32 rigidContactFrames;
	PxU32 softContactFrames;
	PxU32 maxGroundContacts;
	PxU32 maxRigidContacts;
	PxU32 maxSoftContacts;
	PxU32 finalInsideParticles;
	PxU32 cleanupComplete;
	PxReal minDetF;
	PxReal maxDetF;
	PxReal minBodyVolumeRatio;
	PxReal maxBodyVolumeRatio;
	PxReal minY;
	PxReal maxY;
	PxReal maxParticleSpeed;
	PxReal finalMinY;
	PxReal finalMaxY;
	PxReal finalMaxParticleSpeed;
	PxReal maxCentroidDrop;
	bool solverReadbackMatched;

	DeformableVolumeMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0),
	  nonFiniteParticleSamples(0), invertedElementSamples(0),
	  firstInversionFrame(PX_MAX_U32), firstInversionBody(PX_MAX_U32),
	  firstInversionElement(PX_MAX_U32), invertedBodiesMask(0),
	  particles(0), softBodies(0), tetElements(0), surfaceTriangles(0),
	  rigidBoxes(0),
	  sceneStatics(0), sceneDynamics(0), sceneDeformableVolumes(0),
	  groundContactFrames(0), rigidContactFrames(0), softContactFrames(0),
	  maxGroundContacts(0), maxRigidContacts(0), maxSoftContacts(0),
	  finalInsideParticles(0), cleanupComplete(0), minDetF(FLT_MAX),
	  maxDetF(-FLT_MAX), minBodyVolumeRatio(FLT_MAX),
	  maxBodyVolumeRatio(-FLT_MAX), minY(FLT_MAX), maxY(-FLT_MAX),
	  maxParticleSpeed(0.0f), finalMinY(FLT_MAX), finalMaxY(-FLT_MAX),
	  finalMaxParticleSpeed(0.0f),
	  maxCentroidDrop(0.0f), solverReadbackMatched(false)
	{
	}
};

static DeformableVolumeMetrics gMetrics;
static PxArray<PxVec3> gInitialCentroids;

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

static PxVec3 getSoftBodyCentroid(const AvbdSoftBody& body)
{
	PxVec3 centroid(0.0f);
	PxReal totalMass = 0.0f;
	for(PxU32 localId = 0; localId < body.particleCount; ++localId)
	{
		const AvbdSoftParticle& particle =
			gParticles[body.particleStart + localId];
		centroid += particle.position * particle.mass;
		totalMass += particle.mass;
	}
	return totalMass > 0.0f ?
		centroid * (1.0f / totalMass) : PxVec3(0.0f);
}

static void addCubeSoftBody(
	const PxVec3& center, PxReal halfExtent, PxU32 subdivisions,
	PxReal youngsModulus = 2e5f, PxReal density = 500.0f,
	PxReal damping = 0.015f)
{
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(
		center, halfExtent, int(subdivisions), verts, tets);
	avbdCreateSoftBody(
		verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		youngsModulus, 0.3f, density, damping, 0.0f, 0.01f,
		gParticles, gSoftBodies);
}

static void addConeSoftBody(const PxVec3& baseCenter)
{
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	generateConeTetsViaTetMaker(
		baseCenter, 0.8f, 3.0f, 14, verts, tets);
	avbdCreateSoftBody(
		verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		2e5f, 0.3f, 100.0f, 0.015f, 0.0f, 0.01f,
		gParticles, gSoftBodies);
}

static bool addRigidBox(
	const PxVec3& center, const PxVec3& halfExtent)
{
	AvbdRigidBox rigidBox;
	rigidBox.center = center;
	rigidBox.halfExtent = halfExtent;
	rigidBox.friction = 0.5f;
	gRigidBoxes.pushBack(rigidBox);

	PxRigidStatic* actor =
		gPhysics->createRigidStatic(PxTransform(center));
	if(!actor)
		return false;
	if(!PxRigidActorExt::createExclusiveShape(
		*actor, PxBoxGeometry(halfExtent), *gMaterial))
	{
		actor->release();
		return false;
	}
	gScene->addActor(*actor);
	return true;
}

// ---------------------------------------------------------------------------
static bool initPhysicsInternal(
	bool interactive, const std::string& caseName)
{
	gMetrics = DeformableVolumeMetrics();
	gErrorCallback.reset();
	gParticles.clear();
	gSoftBodies.clear();
	gContacts.clear();
	gRigidBoxes.clear();
	gSoftBodyRenderData.clear();
	gInitialCentroids.clear();
	initOGCParams();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			PxPvdTransport* transport =
				PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(transport)
				gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
		}
	}

	gPhysics    = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity       = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.solverType = interactive ?
		PxSolverType::eAVBD : gHeadlessOptions.solverType;
	const PxU32 workerCount =
		interactive ? 2u : gHeadlessOptions.dispatcherThreads;
	gDispatcher = PxDefaultCpuDispatcherCreate(workerCount);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader  = PxDefaultSimulationFilterShader;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	gMetrics.solverReadbackMatched =
		gScene->getSolverType() == sceneDesc.solverType;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (interactive && pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	if(!gMaterial)
		return false;

	// Ground plane
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	if(!ground)
		return false;
	gScene->addActor(*ground);

	if(caseName == "volume-ground")
	{
		addCubeSoftBody(PxVec3(0.0f, 3.0f, 0.0f), 0.5f, 3);
	}
	else if(caseName == "volume-static-box")
	{
		addCubeSoftBody(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3);
		if(!addRigidBox(
			PxVec3(0.0f, 0.5f, 0.0f),
			PxVec3(2.0f, 0.5f, 2.0f)))
		{
			return false;
		}
	}
	else if(caseName == "soft-soft")
	{
		addCubeSoftBody(PxVec3(0.0f, 1.0f, 0.0f), 0.5f, 3);
		addCubeSoftBody(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3);
	}
	else if(caseName == "cone-ground")
	{
		addConeSoftBody(PxVec3(-0.8f, 11.0f, 1.2f));
	}
	else
	{

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
	addConeSoftBody(PxVec3(-0.8f, 11.0f, 1.2f));

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
	if(!addRigidBox(
		PxVec3(7.6f, 0.55f, 0.0f),
		PxVec3(0.7f, 0.55f, 2.2f)))
	{
		return false;
	}
	}

	if(gSoftBodies.empty() || gParticles.empty())
		return false;

	updateRenderData();
	gInitialCentroids.reserve(gSoftBodies.size());
	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
		gInitialCentroids.pushBack(getSoftBodyCentroid(gSoftBodies[bodyId]));

	gMetrics.initialized = 1;
	gMetrics.particles = gParticles.size();
	gMetrics.softBodies = gSoftBodies.size();
	gMetrics.rigidBoxes = gRigidBoxes.size();
	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
	{
		gMetrics.tetElements += gSoftBodies[bodyId].tetElements.size();
		gMetrics.surfaceTriangles +=
			gSoftBodies[bodyId].surfaceTriangles.size() / 3;
	}
	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.sceneDeformableVolumes = gScene->getNbDeformableVolumes();

	printf("SnippetDeformableVolumeAVBD: %u particles, %u soft bodies, %u rigid boxes\n",
		gParticles.size(), gSoftBodies.size(), gRigidBoxes.size());
	printf(
		"[AVBD_COMPONENT_TOPOLOGY] particles=%u softBodies=%u "
		"tetElements=%u surfaceTriangles=%u rigidBoxes=%u "
		"sceneStatics=%u sceneDynamics=%u sceneDeformableVolumes=%u\n",
		gMetrics.particles, gMetrics.softBodies, gMetrics.tetElements,
		gMetrics.surfaceTriangles, gMetrics.rigidBoxes,
		gMetrics.sceneStatics, gMetrics.sceneDynamics,
		gMetrics.sceneDeformableVolumes);
	return true;
}

void initPhysics(bool interactive)
{
	if(!initPhysicsInternal(interactive, "current-all"))
		printf("SnippetDeformableVolumeAVBD initialization failed.\n");
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

static void recordContactMetrics()
{
	PxU32 groundContacts = 0;
	PxU32 rigidContacts = 0;
	PxU32 softContacts = 0;
	for(PxU32 contactId = 0; contactId < gContacts.size(); ++contactId)
	{
		const PxU32 rigidBodyId = gContacts[contactId].rigidBodyIdx;
		if(rigidBodyId == PX_MAX_U32)
			groundContacts++;
		else if(rigidBodyId < gSoftBodies.size())
			softContacts++;
		else
			rigidContacts++;
	}
	if(groundContacts)
		gMetrics.groundContactFrames++;
	if(rigidContacts)
		gMetrics.rigidContactFrames++;
	if(softContacts)
		gMetrics.softContactFrames++;
	gMetrics.maxGroundContacts =
		PxMax(gMetrics.maxGroundContacts, groundContacts);
	gMetrics.maxRigidContacts =
		PxMax(gMetrics.maxRigidContacts, rigidContacts);
	gMetrics.maxSoftContacts =
		PxMax(gMetrics.maxSoftContacts, softContacts);
}

static void recordStateMetrics()
{
	for(PxU32 particleId = 0; particleId < gParticles.size(); ++particleId)
	{
		const AvbdSoftParticle& particle = gParticles[particleId];
		if(!particle.position.isFinite() || !particle.velocity.isFinite())
		{
			gMetrics.nonFiniteParticleSamples++;
			continue;
		}
		gMetrics.minY = PxMin(gMetrics.minY, particle.position.y);
		gMetrics.maxY = PxMax(gMetrics.maxY, particle.position.y);
		gMetrics.maxParticleSpeed =
			PxMax(gMetrics.maxParticleSpeed, particle.velocity.magnitude());
	}

	for(PxU32 bodyId = 0; bodyId < gSoftBodies.size(); ++bodyId)
	{
		const AvbdSoftBody& body = gSoftBodies[bodyId];
		PxReal restVolume = 0.0f;
		PxReal currentVolume = 0.0f;
		for(PxU32 elementId = 0;
			elementId < body.tetElements.size(); ++elementId)
		{
			const AvbdTetElement& tet = body.tetElements[elementId];
			const PxVec3& x0 = gParticles[tet.p0].position;
			const PxMat33 ds(
				gParticles[tet.p1].position - x0,
				gParticles[tet.p2].position - x0,
				gParticles[tet.p3].position - x0);
			const PxReal detF = (ds * tet.DmInv).getDeterminant();
			restVolume += tet.restVolume;
			if(!PxIsFinite(detF))
			{
				gMetrics.invertedElementSamples++;
				if(gMetrics.firstInversionFrame == PX_MAX_U32)
				{
					gMetrics.firstInversionFrame = gMetrics.completedFrames;
					gMetrics.firstInversionBody = bodyId;
					gMetrics.firstInversionElement = elementId;
				}
				if(bodyId < 32)
					gMetrics.invertedBodiesMask |= 1u << bodyId;
				continue;
			}
			gMetrics.minDetF = PxMin(gMetrics.minDetF, detF);
			gMetrics.maxDetF = PxMax(gMetrics.maxDetF, detF);
			if(detF <= 0.0f)
			{
				gMetrics.invertedElementSamples++;
				if(gMetrics.firstInversionFrame == PX_MAX_U32)
				{
					gMetrics.firstInversionFrame = gMetrics.completedFrames;
					gMetrics.firstInversionBody = bodyId;
					gMetrics.firstInversionElement = elementId;
				}
				if(bodyId < 32)
					gMetrics.invertedBodiesMask |= 1u << bodyId;
			}
			currentVolume += detF * tet.restVolume;
		}
		if(restVolume > 0.0f)
		{
			const PxReal ratio = currentVolume / restVolume;
			if(PxIsFinite(ratio))
			{
				gMetrics.minBodyVolumeRatio =
					PxMin(gMetrics.minBodyVolumeRatio, ratio);
				gMetrics.maxBodyVolumeRatio =
					PxMax(gMetrics.maxBodyVolumeRatio, ratio);
			}
			else
				gMetrics.nonFiniteParticleSamples++;
		}
		const PxVec3 centroid = getSoftBodyCentroid(body);
		if(centroid.isFinite() && bodyId < gInitialCentroids.size())
		{
			gMetrics.maxCentroidDrop = PxMax(
				gMetrics.maxCentroidDrop,
				gInitialCentroids[bodyId].y - centroid.y);
		}
	}
}

static bool stepPhysicsInternal(PxReal dt)
{

	// Initial contact detection: ground + soft-soft OGC + rigid-soft SDF.
	avbdDetectAllOGCContacts(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gRigidBoxes.begin(), gRigidBoxes.size(),
		NULL, 0,
		gContacts, gOGCParams, 0.0f);
	recordContactMetrics();

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
	if(!gScene->fetchResults(true))
	{
		gMetrics.fetchFailures++;
		return false;
	}

	gMetrics.completedFrames++;
	recordStateMetrics();
	sendSoftBodiesToPvd();
	return true;
}

void stepPhysics(bool /*interactive*/)
{
	stepPhysicsInternal(1.0f / 60.0f);
}

static void finalizeMetrics()
{
	gMetrics.finalInsideParticles = 0;
	gMetrics.finalMinY = FLT_MAX;
	gMetrics.finalMaxY = -FLT_MAX;
	gMetrics.finalMaxParticleSpeed = 0.0f;
	for(PxU32 particleId = 0; particleId < gParticles.size(); ++particleId)
	{
		const AvbdSoftParticle& particle = gParticles[particleId];
		if(!particle.position.isFinite() || !particle.velocity.isFinite())
			continue;
		gMetrics.finalMinY =
			PxMin(gMetrics.finalMinY, particle.position.y);
		gMetrics.finalMaxY =
			PxMax(gMetrics.finalMaxY, particle.position.y);
		gMetrics.finalMaxParticleSpeed = PxMax(
			gMetrics.finalMaxParticleSpeed, particle.velocity.magnitude());
	}

	for(PxU32 bodyA = 0; bodyA < gSoftBodies.size(); ++bodyA)
	{
		const AvbdSoftBody& source = gSoftBodies[bodyA];
		for(PxU32 bodyB = 0; bodyB < gSoftBodies.size(); ++bodyB)
		{
			if(bodyA == bodyB)
				continue;
			const AvbdSoftBody& target = gSoftBodies[bodyB];
			for(PxU32 localId = 0;
				localId < source.particleCount; ++localId)
			{
				const PxU32 particleId = source.particleStart + localId;
				if(avbdIsPointInsideTetMesh(
					gParticles[particleId].position,
					target.surfaceTriangles, gParticles.begin()))
				{
					gMetrics.finalInsideParticles++;
				}
			}
		}
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	gSoftBodyRenderData.reset();
	gContacts.reset();
	gRigidBoxes.reset();
	gSoftBodies.reset();
	gParticles.reset();
	gInitialCentroids.reset();

	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gDispatcher && !gMaterial && !gPhysics &&
		!gPvd && !gFoundation ? 1u : 0u;

	printf("SnippetDeformableVolumeAVBD done.\n");
}

void keyPress(unsigned char /*key*/, const PxTransform& /*camera*/)
{
}

static bool isKnownCase(const std::string& caseName)
{
	return caseName == "volume-ground" ||
		caseName == "volume-static-box" ||
		caseName == "soft-soft" ||
		caseName == "cone-ground" ||
		caseName == "current-all";
}

static bool validateHeadlessResult(const std::string& caseName)
{
	bool passed =
		gMetrics.initialized == 1 &&
		gMetrics.completedFrames == gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 &&
		gMetrics.nonFiniteParticleSamples == 0 &&
		gMetrics.invertedElementSamples == 0 &&
		gMetrics.particles > 0 &&
		gMetrics.softBodies > 0 &&
		gMetrics.tetElements > 0 &&
		gMetrics.surfaceTriangles > 0 &&
		gMetrics.sceneStatics == gMetrics.rigidBoxes + 1 &&
		gMetrics.sceneDynamics == 0 &&
		gMetrics.sceneDeformableVolumes == 0 &&
		gMetrics.solverReadbackMatched &&
		gMetrics.cleanupComplete == 1 &&
		PxIsFinite(gMetrics.minDetF) && gMetrics.minDetF > 0.0f &&
		PxIsFinite(gMetrics.maxDetF) && gMetrics.maxDetF < 20.0f &&
		PxIsFinite(gMetrics.minBodyVolumeRatio) &&
		gMetrics.minBodyVolumeRatio > 0.01f &&
		PxIsFinite(gMetrics.maxBodyVolumeRatio) &&
		gMetrics.maxBodyVolumeRatio < 20.0f &&
		PxIsFinite(gMetrics.minY) && gMetrics.minY > -0.25f &&
		PxIsFinite(gMetrics.maxY) && gMetrics.maxY < 100.0f &&
		PxIsFinite(gMetrics.maxParticleSpeed) &&
		gMetrics.maxParticleSpeed < 250.0f &&
		gErrorCallback.getFatalCount() == 0;

	if(caseName == "volume-ground")
	{
		passed = passed &&
			gMetrics.softBodies == 1 &&
			gMetrics.rigidBoxes == 0 &&
			gMetrics.groundContactFrames > 0 &&
			gMetrics.maxGroundContacts > 0 &&
			gMetrics.rigidContactFrames == 0 &&
			gMetrics.softContactFrames == 0 &&
			gMetrics.maxCentroidDrop > 1.0f;
	}
	else if(caseName == "volume-static-box")
	{
		passed = passed &&
			gMetrics.softBodies == 1 &&
			gMetrics.rigidBoxes == 1 &&
			gMetrics.rigidContactFrames > 0 &&
			gMetrics.maxRigidContacts > 0 &&
			gMetrics.softContactFrames == 0 &&
			gMetrics.maxCentroidDrop > 1.0f &&
			gMetrics.finalMinY > 0.70f;
	}
	else if(caseName == "soft-soft")
	{
		passed = passed &&
			gMetrics.softBodies == 2 &&
			gMetrics.rigidBoxes == 0 &&
			gMetrics.softContactFrames > 0 &&
			gMetrics.maxSoftContacts > 0 &&
			gMetrics.finalInsideParticles == 0 &&
			gMetrics.maxCentroidDrop > 1.0f;
	}
	else if(caseName == "cone-ground")
	{
		passed = passed &&
			gMetrics.softBodies == 1 &&
			gMetrics.rigidBoxes == 0 &&
			gMetrics.groundContactFrames > 0 &&
			gMetrics.maxGroundContacts > 0 &&
			gMetrics.rigidContactFrames == 0 &&
			gMetrics.softContactFrames == 0 &&
			gMetrics.maxCentroidDrop > 5.0f;
	}
	else
	{
		passed = passed &&
			gMetrics.softBodies == 5 &&
			gMetrics.rigidBoxes == 1 &&
			gMetrics.groundContactFrames > 0 &&
			gMetrics.rigidContactFrames > 0 &&
			gMetrics.softContactFrames > 0 &&
			gMetrics.maxCentroidDrop > 1.0f;
	}
	return passed;
}

static void printHeadlessResult(bool passed)
{
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetDeformableVolumeAVBD "
		"case=%s solver=%s validation=COMPONENT_GATED "
		"sceneSoftIntegration=0 status=%s initialized=%u "
		"frames=%u fetchFailures=%u particles=%u softBodies=%u "
		"tetElements=%u surfaceTriangles=%u rigidBoxes=%u "
		"sceneStatics=%u sceneDynamics=%u sceneDeformableVolumes=%u "
		"groundContactFrames=%u rigidContactFrames=%u "
		"softContactFrames=%u maxGroundContacts=%u "
		"maxRigidContacts=%u maxSoftContacts=%u "
		"finalInsideParticles=%u nonFiniteParticleSamples=%u "
		"invertedElementSamples=%u firstInversionFrame=%u "
		"firstInversionBody=%u firstInversionElement=%u "
		"invertedBodiesMask=%u minDetF=%.9g maxDetF=%.9g "
		"minBodyVolumeRatio=%.9g maxBodyVolumeRatio=%.9g "
		"minY=%.9g maxY=%.9g finalMinY=%.9g finalMaxY=%.9g "
		"maxParticleSpeed=%.9g finalMaxParticleSpeed=%.9g "
		"maxCentroidDrop=%.9g solverReadbackMatched=%u "
		"fatalErrors=%u warningErrors=%u cleanupComplete=%u\n",
		gHeadlessOptions.caseName.c_str(),
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		passed ? "PASS" : "FAIL", gMetrics.initialized,
		gMetrics.completedFrames, gMetrics.fetchFailures,
		gMetrics.particles, gMetrics.softBodies, gMetrics.tetElements,
		gMetrics.surfaceTriangles, gMetrics.rigidBoxes,
		gMetrics.sceneStatics, gMetrics.sceneDynamics,
		gMetrics.sceneDeformableVolumes, gMetrics.groundContactFrames,
		gMetrics.rigidContactFrames, gMetrics.softContactFrames,
		gMetrics.maxGroundContacts, gMetrics.maxRigidContacts,
		gMetrics.maxSoftContacts, gMetrics.finalInsideParticles,
		gMetrics.nonFiniteParticleSamples, gMetrics.invertedElementSamples,
		gMetrics.firstInversionFrame, gMetrics.firstInversionBody,
		gMetrics.firstInversionElement, gMetrics.invertedBodiesMask,
		double(gMetrics.minDetF), double(gMetrics.maxDetF),
		double(gMetrics.minBodyVolumeRatio),
		double(gMetrics.maxBodyVolumeRatio), double(gMetrics.minY),
		double(gMetrics.maxY), double(gMetrics.finalMinY),
		double(gMetrics.finalMaxY), double(gMetrics.maxParticleSpeed),
		double(gMetrics.finalMaxParticleSpeed),
		double(gMetrics.maxCentroidDrop),
		gMetrics.solverReadbackMatched ? 1u : 0u,
		gErrorCallback.getFatalCount(), gErrorCallback.getWarningCount(),
		gMetrics.cleanupComplete);
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.caseName = "current-all";
	defaults.frames = 600;
	defaults.dispatcherThreads = 2;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, parseError))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] %s\n", parseError.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int argId = 1; argId < argc; ++argId)
	{
		if(!Snippets::isCommonHeadlessOption(argv[argId]))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown option: %s\n",
				argv[argId]);
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
	}
	if(gHeadlessOptions.headless)
	{
		if(gHeadlessOptions.solverType != PxSolverType::eAVBD)
		{
			printf(
				"[AVBD_GATE_UNSUPPORTED] reason=component-is-avbd-only\n");
			return Snippets::eHEADLESS_UNSUPPORTED;
		}
		if(!isKnownCase(gHeadlessOptions.caseName))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown case: %s\n",
				gHeadlessOptions.caseName.c_str());
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		{
			printf(
				"[AVBD_GATE_CONFIG_ERROR] "
				"failed to apply execution environment\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		Snippets::printHeadlessConfig(
			"SnippetDeformableVolumeAVBD", gHeadlessOptions);
		bool initialized = initPhysicsInternal(
			false, gHeadlessOptions.caseName);
		if(initialized)
		{
			for(PxU32 frame = 0;
				frame < gHeadlessOptions.frames; ++frame)
			{
				if(!stepPhysicsInternal(gHeadlessOptions.dt))
					break;
			}
			finalizeMetrics();
		}
		cleanupPhysics(false);
		const bool passed =
			initialized &&
			validateHeadlessResult(gHeadlessOptions.caseName);
		printHeadlessResult(passed);
		return passed ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	printf("SnippetDeformableVolumeAVBD: No render snippet, nothing to do.\n");
#endif

	return 0;
}
