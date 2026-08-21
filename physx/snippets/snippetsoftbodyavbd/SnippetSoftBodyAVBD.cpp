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
// SnippetSoftBodyAVBD
//
// Demonstrates a CPU-side VBD/AVBD soft body simulation rendered alongside a
// PhysX AVBD rigid-body scene.  A subdivided tetrahedral cube drops onto a
// ground plane; elastic forces use Neo-Hookean energy via VBD, while ground
// contacts are enforced through AVBD adaptive penalty.
//
// Headless unit coverage is owned by SnippetSoftBodyAVBDTests.cpp; this file
// owns only the visual fixture and executable lifecycle.
//
// No GPU or CUDA dependency -- runs entirely on the CPU.
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "PxPhysicsAPI.h"
#include "avbd/solver/soft/DyAvbdSoftBody.h"
#include "avbd/contact/DyAvbdContactDetection.h"

#include "SnippetSoftBodyAVBD.h"
#include "SnippetSoftBodyAVBDDiagnostics.h"
#include "SnippetSoftBodyAVBDTests.h"

using namespace physx;
using namespace physx::Dy;
using namespace SnippetSoftBodyAVBDDiagnostics;
using namespace SnippetSoftBodyAVBDTests;

// ---------------------------------------------------------------------------
// Globals (shared with render file for visual mode)
// ---------------------------------------------------------------------------

static PxDefaultAllocator      gAllocator;
static PxDefaultErrorCallback  gErrorCallback;
static PxFoundation*           gFoundation  = NULL;
static PxPhysics*              gPhysics     = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*             gMaterial    = NULL;

PxScene*                       gScene       = NULL;

// VBD soft body data (managed outside PhysX scene)
PxArray<AvbdSoftParticle>        gParticles;
PxArray<AvbdSoftBody>            gSoftBodies;
static PxArray<AvbdSoftContact>  gContacts;

// Render data shared with the render file
PxArray<SoftBodyRenderData>      gSoftBodyRenderData;

// ===========================================================================
// Visual-mode infrastructure
// ===========================================================================

PxU32       gVisFrameCount = 0;
PxU32       gVisMaxFrames  = 0;
const char* gVisTestName   = NULL;

enum VisStepMode {
	VS_GROUND,
	VS_NO_GROUND,
	VS_FULL,
	VS_OGC,
	VS_OGC_RIGID,
	VS_OGC_SELF,
	VS_OGC_FULL,
};

static VisStepMode                          gVisStepMode  = VS_GROUND;
static PxArray<AvbdRigidBox>                gVisRigidBoxes;
static AvbdOGCParams                        gVisOGC;
static PxArray<AvbdSelfCollisionAdjacency>  gVisSelfAdj;
static PxArray<PxArray<PxVec3> >            gVisBodyRefs;
static PxVec3                               gVisGravity   = PxVec3(0.0f, -9.81f, 0.0f);
static PxReal                               gVisGroundY   = 0.0f;
static PxU32                                gVisOuterIter = 1;
static PxU32                                gVisInnerIter = 10;

static void addVisCube(PxVec3 center, PxReal half, PxU32 sub,
                       PxReal E, PxReal nu, PxReal density, PxReal damping,
                       PxReal compliance = 0.0f, PxReal lsTol = 0.01f)
{
	PxArray<PxVec3> v;
	PxArray<PxU32>  t;
	avbdGenerateSubdividedCubeTets(center, half, sub, v, t);
	avbdCreateSoftBody(v.begin(), v.size(),
	                   t.begin(), t.size(), NULL, 0,
	                   E, nu, density, damping, compliance, lsTol,
	                   gParticles, gSoftBodies);
}

static void updateVisRenderData()
{
	gSoftBodyRenderData.clear();
	for (PxU32 i = 0; i < gSoftBodies.size(); i++)
	{
		SoftBodyRenderData rd;
		rd.positions    = &gParticles[gSoftBodies[i].compiled.particleStart].position;
		rd.numParticles = gSoftBodies[i].compiled.particleCount;
		rd.tetIndices   = gSoftBodies[i].compiled.tetrahedra.begin();
		rd.numTets      = gSoftBodies[i].compiled.tetrahedra.size() / 4;
		rd.triIndices   = gSoftBodies[i].compiled.triangles.size() > 0 ?
			gSoftBodies[i].compiled.triangles.begin() : NULL;
		rd.numTris      = gSoftBodies[i].compiled.triangles.size() / 3;
		gSoftBodyRenderData.pushBack(rd);
	}
}

static void resetVisState()
{
	gParticles.clear();
	gSoftBodies.clear();
	gContacts.clear();
	gSoftBodyRenderData.clear();
	gVisRigidBoxes.clear();
	gVisSelfAdj.clear();
	gVisBodyRefs.clear();
	gVisFrameCount = 0;
	gVisMaxFrames  = 300;
	gVisGravity    = PxVec3(0.0f, -9.81f, 0.0f);
	gVisGroundY    = 0.0f;
	gVisOuterIter  = 1;
	gVisInnerIter  = 10;
	gVisStepMode   = VS_GROUND;
	gVisOGC        = AvbdOGCParams();
}

static void setupVisualTest(int testId)
{
	resetVisState();

	switch (testId)
	{
	case 1:
		addVisCube(PxVec3(0,5,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 0.01f);
		gVisMaxFrames = 90;
		gVisStepMode  = VS_NO_GROUND;
		gVisTestName  = "Test 1: Gravity Free-Fall";
		break;

	case 2:
		addVisCube(PxVec3(0,1,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 2: Ground Contact";
		break;

	case 3:
		addVisCube(PxVec3(0,2,0), 0.3f, 3, 5e5f, 0.4f, 1000.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 3: Volume Preservation";
		break;

	case 4:
	{
		addVisCube(PxVec3(0,3,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		PxReal topY = -PX_MAX_F32;
		for (PxU32 i = 0; i < gParticles.size(); i++)
			if (gParticles[i].position.y > topY) topY = gParticles[i].position.y;
		for (PxU32 i = 0; i < gParticles.size(); i++)
		{
			if (PxAbs(gParticles[i].position.y - topY) < 0.01f)
			{
				AvbdKinematicPin pin;
				pin.point.setVertex(i);
				pin.worldTarget = gParticles[i].position;
				pin.k = 1e5f;
				pin.kMax = 1e7f;
				gSoftBodies[0].runtime.pins.pushBack(pin);
			}
		}
		gSoftBodies[0].runtime.compileObjectiveProgram(
			gSoftBodies[0].compiled.particleStart,
			gSoftBodies[0].compiled.particleCount);
		gVisMaxFrames = 180;
		gVisStepMode  = VS_NO_GROUND;
		gVisTestName  = "Test 4: Kinematic Pin";
		break;
	}

	case 5:
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tris;
		avbdGenerateClothGrid(PxVec3(0,2,0), 2.0f, 2.0f, 10, 10, verts, tris);
		avbdCreateSoftBody(verts.begin(), verts.size(),
		                   NULL, 0, tris.begin(), tris.size(),
		                   1e4f, 0.3f, 500.0f, 5.0f, 1.0f, 0.005f,
		                   gParticles, gSoftBodies);
		gVisMaxFrames = 180;
		gVisTestName  = "Test 5: Cloth Drape";
		break;
	}

	case 6:
		addVisCube(PxVec3(0,1.5f,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 50.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 6: Energy Dissipation";
		break;

	case 7:
		addVisCube(PxVec3(0,2,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 5.0f);
		gVisMaxFrames = 120;
		gVisStepMode  = VS_NO_GROUND;
		gVisGravity   = PxVec3(0.0f);
		gVisTestName  = "Test 7: Static Equilibrium";
		break;

	case 8:
		addVisCube(PxVec3(-2,3,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		addVisCube(PxVec3( 2,5,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 8: Multiple Soft Bodies";
		break;

	case 9:
		addVisCube(PxVec3(0,1,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		addVisCube(PxVec3(0,4,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisStepMode  = VS_FULL;
		gVisOGC.contactRadius = 0.3f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction = 0.5f;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 9: Soft-Soft Collision";
		break;

	case 10:
	{
		addVisCube(PxVec3(0,5,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox box;
		box.center     = PxVec3(0,1.5f,0);
		box.halfExtent = PxVec3(2,1.5f,2);
		box.friction   = 0.5f;
		gVisRigidBoxes.pushBack(box);
		gVisMaxFrames = 300;
		gVisStepMode  = VS_FULL;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 10: Soft-Rigid Collision";
		break;
	}

	case 11:
	{
		addVisCube(PxVec3(0,5,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox ramp;
		ramp.center     = PxVec3(0,2,0);
		ramp.rotation   = PxQuat(3.14159265f / 6.0f, PxVec3(0,0,1));
		ramp.halfExtent = PxVec3(4,0.5f,4);
		ramp.friction   = 0.3f;
		gVisRigidBoxes.pushBack(ramp);
		gVisMaxFrames = 180;
		gVisStepMode  = VS_FULL;
		gVisOuterIter = 3; gVisInnerIter = 10;
		gVisTestName  = "Test 11: Slope Rotation";
		break;
	}

	case 12:
	{
		addVisCube(PxVec3(0,0.6f,0), 0.5f, 3, 2e5f, 0.3f, 1000.0f, 0.01f);
		PxArray<PxVec3> cv; PxArray<PxU32> ct;
		avbdGenerateConeTets(PxVec3(0,2,0), 0.5f, 1.5f, 3, cv, ct);
		avbdCreateSoftBody(cv.begin(), cv.size(), ct.begin(), ct.size(), NULL, 0,
		                   2e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
		                   gParticles, gSoftBodies);
		gVisOGC.contactRadius    = 0.15f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 12: Cone-Cube Penetration";
		break;
	}

	case 13:
		addVisCube(PxVec3(0,1,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		addVisCube(PxVec3(0,4,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		gVisOGC.contactRadius    = 0.15f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 13: OGC Soft-Soft";
		break;

	case 14:
	{
		addVisCube(PxVec3(0,3,0), 0.5f, 3, 1e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox rb;
		rb.center     = PxVec3(0,0.5f,0);
		rb.halfExtent = PxVec3(2,0.5f,2);
		rb.friction   = 0.5f;
		gVisRigidBoxes.pushBack(rb);
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC_RIGID;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 14: OGC Rigid-Soft SDF";
		break;
	}

	case 15:
	{
		addVisCube(PxVec3(0,3,0), 1.0f, 3, 5e4f, 0.3f, 500.0f, 10.0f);
		avbdBuildAllSelfCollisionAdjacencies(gSoftBodies.begin(), gSoftBodies.size(), gVisSelfAdj);
		gVisOGC.contactRadius    = 0.08f;
		gVisOGC.contactStiffness = 5e4f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 180;
		gVisStepMode  = VS_OGC_SELF;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 15: OGC Self-Collision";
		break;
	}

	case 16:
	{
		addVisCube(PxVec3(0,1.5f,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		addVisCube(PxVec3(0,4,0),    0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox rb;
		rb.center     = PxVec3(3,0.5f,0);
		rb.halfExtent = PxVec3(1,0.5f,1);
		rb.friction   = 0.5f;
		gVisRigidBoxes.pushBack(rb);
		avbdBuildAllSelfCollisionAdjacencies(gSoftBodies.begin(), gSoftBodies.size(), gVisSelfAdj);
		gVisOGC.contactRadius    = 0.15f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC_FULL;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 16: OGC Full Pipeline";
		break;
	}

	case 17:
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32>  tets;
		PxVec3 center(0,1,0);
		avbdGenerateSubdividedCubeTets(center, 0.5f, 2, verts, tets);
		const PxReal angle = 0.52f;
		const PxReal cs = PxCos(angle), sn = PxSin(angle);
		for (PxU32 i = 0; i < verts.size(); i++)
		{
			PxVec3 r = verts[i] - center;
			verts[i].x = center.x + r.x * cs - r.y * sn;
			verts[i].y = center.y + r.x * sn + r.y * cs;
		}
		avbdCreateSoftBody(verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		                   1e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
		                   gParticles, gSoftBodies);
		gVisMaxFrames = 180;
		gVisOuterIter = 8; gVisInnerIter = 20;
		gVisTestName  = "Test 17: Asymmetric Toppling";
		break;
	}

	case 18:
		addVisCube(PxVec3(-1,2,0), 0.3f, 2, 1e3f, 0.3f, 1000.0f, 0.01f);
		addVisCube(PxVec3( 1,2,0), 0.3f, 2, 1e6f, 0.3f, 1000.0f, 0.01f);
		gVisMaxFrames = 120;
		gVisOuterIter = 8; gVisInnerIter = 20;
		gVisTestName  = "Test 18: Material Stiffness";
		break;

	case 19:
		addVisCube(PxVec3(0,1,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 0.01f);
		gVisMaxFrames = 600;
		gVisOuterIter = 8; gVisInnerIter = 20;
		gVisTestName  = "Test 19: Long-Term Stability";
		break;
	}

	captureBodyReferenceLocals(gParticles, gSoftBodies, gVisBodyRefs);
	updateVisRenderData();
	printf("=== Visual: %s (%u frames) ===\n", gVisTestName ? gVisTestName : "?", gVisMaxFrames);
}

// ---------------------------------------------------------------------------
// Visual-mode scene setup (PhysX scene for ground rendering only)
// ---------------------------------------------------------------------------

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPhysics    = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity       = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher             = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader  = PxDefaultSimulationFilterShader;
	sceneDesc.solverType    = PxSolverType::eAVBD;
	gScene = gPhysics->createScene(sceneDesc);

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.1f);

	// Ground plane (rendered via PhysX scene; collision handled by AVBD)
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*ground);
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = 1.0f / 60.0f;

	gContacts.clear();

	// Ground contacts (unless disabled)
	if (gVisStepMode != VS_NO_GROUND)
		avbdDetectSoftGroundContacts(gParticles.begin(), gParticles.size(),
		                             gContacts, gVisGroundY, 0.02f, 0.5f);

	// Mode-specific additional contacts
	switch (gVisStepMode)
	{
	case VS_FULL:
		avbdDetectSoftSoftOGC(gParticles.begin(), gParticles.size(),
		                      gSoftBodies.begin(), gSoftBodies.size(),
		                      gContacts, gVisOGC);
		if (gVisRigidBoxes.size())
			avbdDetectSoftRigidSDF(gParticles.begin(), gParticles.size(),
			                       gVisRigidBoxes.begin(), gVisRigidBoxes.size(),
			                       gContacts, 0.1f);
		break;
	case VS_OGC:
		avbdDetectSoftSoftOGC(gParticles.begin(), gParticles.size(),
		                       gSoftBodies.begin(), gSoftBodies.size(),
		                       gContacts, gVisOGC);
		break;
	case VS_OGC_RIGID:
		avbdDetectSoftRigidSDF(gParticles.begin(), gParticles.size(),
		                        gVisRigidBoxes.begin(), gVisRigidBoxes.size(),
		                        gContacts, 0.05f);
		break;
	case VS_OGC_SELF:
		if (gVisSelfAdj.size())
			avbdDetectSelfCollisionOGC(
				gParticles.begin(), gSoftBodies[0], 0,
				gVisSelfAdj[0], gContacts, gVisOGC);
		break;
	case VS_OGC_FULL:
		avbdDetectAllOGCContacts(
			gParticles.begin(), gParticles.size(),
			gSoftBodies.begin(), gSoftBodies.size(),
			gVisRigidBoxes.begin(), gVisRigidBoxes.size(),
			gVisSelfAdj.begin(), gVisSelfAdj.size(),
			gContacts, gVisOGC, gVisGroundY);
		break;
	default:
		break;
	}

	avbdStepSoftBodies(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gContacts.begin(), gContacts.size(),
		dt, gVisGravity, gVisOuterIter, gVisInnerIter, 1000.0f);

	if (isRotationTraceEnabled())
	{
		const PxU32 traceInterval = getRotationTraceInterval();
		if ((gVisFrameCount % traceInterval) == 0)
			printBodyRotationTrace(gVisTestName ? gVisTestName : "visual", gVisFrameCount, gParticles, gSoftBodies, gVisBodyRefs);
	}

	// Update render data pointers
	for (PxU32 i = 0; i < gSoftBodyRenderData.size(); i++)
		gSoftBodyRenderData[i].positions =
			&gParticles[gSoftBodies[i].compiled.particleStart].position;

	gVisFrameCount++;
}

void cleanupPhysics(bool /*interactive*/)
{
	gSoftBodyRenderData.reset();
	gContacts.reset();
	gSoftBodies.reset();
	gParticles.reset();
	gVisRigidBoxes.reset();
	gVisSelfAdj.reset();

	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	PX_RELEASE(gFoundation);

	printf("SnippetSoftBodyAVBD done.\n");
}

void keyPress(unsigned char /*key*/, const PxTransform& /*camera*/)
{
}

static bool isVisualMode()
{
	const char* v = std::getenv("PHYSX_AVBD_SOFTBODY_VISUAL");
	return v && v[0] && v[0] != '0';
}

static bool isExplicitHeadlessMode(int argc, const char*const* argv)
{
	const char* environment = std::getenv("PHYSX_SNIPPET_HEADLESS");
	if(environment && environment[0] && environment[0] != '0')
		return true;

	for(int argId = 1; argId < argc; ++argId)
	{
		if(argv[argId] && std::strcmp(argv[argId], "--headless") == 0)
			return true;
	}
	return false;
}

int snippetMain(int argc, const char*const* argv)
{
	const int selectedId = getSelectedTestId();
	const bool headless = isExplicitHeadlessMode(argc, argv);

#ifdef RENDER_SNIPPET
	if (isVisualMode() && !headless)
	{
		// Create PhysX scene (ground plane for rendering)
		initPhysics(true);

		extern void renderInit();
		renderInit();

		printf("=== AVBD Soft Body Visual Tests ===\n");
		if (selectedId > 0)
			printf("=== Running only test %d ===\n", selectedId);

		for (int testId = 1; testId <= 19; testId++)
		{
			if (!shouldRunTest(selectedId, testId)) continue;
			setupVisualTest(testId);
			extern void renderRun();
			renderRun();
		}

		printf("=== All visual tests completed ===\n");
		cleanupPhysics(true);
		return 0;
	}
#endif

	if(headless)
		printf("[AVBD_HEADLESS_CONFIG] mode=unit visual=disabled\n");

	// PxArray uses the foundation allocator -- must create PxFoundation first
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	const TestRunResult result = runSelectedTests(*gFoundation, selectedId);

	PX_RELEASE(gFoundation);
	return result.failed > 0 ? 1 : 0;
}
