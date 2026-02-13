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
// SnippetChainmail -- 20x20 spherical-joint mesh catching a heavy falling ball
//
// Demonstrates the AVBD solver's 3-mechanism joint AL algorithm:
//   (A) Auto-boosted primal penalty:  effectiveRho = max(rho, M/h^2)
//   (B) ADMM-safe dual step:          rhoDual = min(Mh2, rho^2/(rho+Mh2))
//   (C) Lambda decay (leaky integrator): lambda = 0.99*lambda + rhoDual*C
//
// Scene layout:
//   - 20x20 grid of nodes connected by spherical joints
//   - Each grid edge has a capsule collision shape ("strut") so the net
//     is a continuous collision surface that can catch the ball
//   - Four edges are anchored (kinematic) to form a hammock/net
//   - A heavy ball (~1000x mass ratio) drops onto the mesh center
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

using namespace physx;

static PxDefaultAllocator		gAllocator;
static PxDefaultErrorCallback	gErrorCallback;
static PxFoundation*			gFoundation		= NULL;
static PxPhysics*				gPhysics		= NULL;
static PxDefaultCpuDispatcher*	gDispatcher		= NULL;
static PxScene*					gScene			= NULL;
static PxMaterial*				gMaterial		= NULL;
static PxPvd*					gPvd			= NULL;

// ---------------------------------------------------------------------------
// Grid parameters
// ---------------------------------------------------------------------------
static const PxU32 GRID_W = 30;
static const PxU32 GRID_H = 30;
static const PxReal SPACING = 0.65f;			// tighter grid (~19m footprint)
static const PxReal NODE_RADIUS = 0.12f;
static const PxReal LINK_RADIUS = 0.06f;		// thin struts to allow bending
static const PxReal NODE_DENSITY = 3.0f;		// light nodes (~0.22 kg each, net ~196 kg)
static const PxReal BALL_RADIUS = 2.0f;
static const PxReal BALL_DENSITY = 300.0f;		// ball ~10000 kg, net ~196 kg (51:1)
static const PxReal MESH_HEIGHT = 35.0f;
static const PxReal BALL_DROP_HEIGHT = 70.0f;	// 35m drop above the mesh

static PxRigidDynamic* gGridBodies[GRID_W * GRID_H] = {}; // 30x30 = 900
static PxRigidDynamic* gBall = NULL;

// ---------------------------------------------------------------------------
// Collision filter: suppress net-internal collision (word0==1 vs word0==1),
// allow everything else (net vs ball, net vs ground, ball vs ground).
// ---------------------------------------------------------------------------
static PxFilterFlags chainmailFilterShader(
	PxFilterObjectAttributes /*a0*/, PxFilterData fd0,
	PxFilterObjectAttributes /*a1*/, PxFilterData fd1,
	PxPairFlags& pairFlags, const void* /*constantBlock*/, PxU32 /*cbSize*/)
{
	if (fd0.word0 == 1 && fd1.word0 == 1)
		return PxFilterFlag::eSUPPRESS;

	pairFlags = PxPairFlag::eCONTACT_DEFAULT;
	return PxFilterFlags();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Tag all shapes on an actor with NET collision group (word0 = 1)
static void setNetFilterData(PxRigidActor* actor)
{
	const PxU32 maxS = 8;
	PxShape* shapes[maxS];
	PxU32 n = actor->getShapes(shapes, maxS);
	PxFilterData fd;
	fd.word0 = 1;
	for (PxU32 i = 0; i < n; i++)
		shapes[i]->setSimulationFilterData(fd);
}

static PxRigidDynamic* createGridNode(const PxVec3& pos)
{
	PxRigidDynamic* body = PxCreateDynamic(
		*gPhysics, PxTransform(pos),
		PxSphereGeometry(NODE_RADIUS), *gMaterial, NODE_DENSITY);
	body->setAngularDamping(0.5f);  // low damping for soft net
	gScene->addActor(*body);
	return body;
}

// Attach a capsule strut from this body towards a neighbour along `dir`.
// `dist` is the actual distance to the neighbour (SPACING for axis-aligned,
// SPACING*sqrt(2) for diagonal).
static void addLinkCapsule(PxRigidDynamic* body, const PxVec3& dir, PxReal dist)
{
	PxReal halfHeight = dist * 0.5f - LINK_RADIUS;
	if (halfHeight < 0.01f) halfHeight = 0.01f;

	PxShape* shape = gPhysics->createShape(
		PxCapsuleGeometry(LINK_RADIUS, halfHeight),
		*gMaterial, true);

	// Capsule default axis = local +X.
	// We need to rotate +X to `dir`.
	PxVec3 centre = dir * (dist * 0.5f);
	PxVec3 dirN = dir.getNormalized();
	PxVec3 ax = PxVec3(1.0f, 0.0f, 0.0f);
	PxQuat rot(PxIdentity);
	PxReal d = ax.dot(dirN);
	if (d < -0.999f)
	{
		rot = PxQuat(PxPi, PxVec3(0.0f, 1.0f, 0.0f));
	}
	else if (d < 0.999f)
	{
		PxVec3 cross = ax.cross(dirN);
		rot = PxQuat(cross.x, cross.y, cross.z, 1.0f + d).getNormalized();
	}

	shape->setLocalPose(PxTransform(centre, rot));

	PxFilterData fd;
	fd.word0 = 1;
	shape->setSimulationFilterData(fd);

	body->attachShape(*shape);
	shape->release();
}

static void createSphericalJoint(PxRigidActor* a0, const PxVec3& anchor0,
								 PxRigidActor* a1, const PxVec3& anchor1)
{
	PxSphericalJointCreate(
		*gPhysics,
		a0, PxTransform(anchor0),
		a1, PxTransform(anchor1));
	// No cone limit -- pure free ball joint for maximum softness
}

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------
void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	PxInitExtensions(*gPhysics, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(4);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = chainmailFilterShader;   // custom filter!
	sceneDesc.solverType = PxSolverType::eAVBD;
	gScene = gPhysics->createScene(sceneDesc);

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.2f);

	// =====================================================================
	// Ground plane
	// =====================================================================
	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*groundPlane);

	// =====================================================================
	// Create 20x20 mesh grid with capsule struts
	//
	//   Each node is a small sphere.  For every rightward edge the node owns
	//   a capsule shape extending +X; for every downward edge it owns a
	//   capsule extending +Z.  Together these form a solid collision net.
	//
	//   Four borders (top/bottom rows, left/right columns) are kinematic.
	// =====================================================================
	PxReal offsetX = -SPACING * (GRID_W - 1) * 0.5f;
	PxReal offsetZ = -SPACING * (GRID_H - 1) * 0.5f;

	for (PxU32 row = 0; row < GRID_H; ++row)
	{
		for (PxU32 col = 0; col < GRID_W; ++col)
		{
			PxVec3 pos(offsetX + col * SPACING, MESH_HEIGHT,
					   offsetZ + row * SPACING);
			PxU32 idx = row * GRID_W + col;

			gGridBodies[idx] = createGridNode(pos);

			// Capsule struts: right link and down link
			if (col + 1 < GRID_W)
				addLinkCapsule(gGridBodies[idx], PxVec3(1.0f, 0.0f, 0.0f), SPACING);
			if (row + 1 < GRID_H)
				addLinkCapsule(gGridBodies[idx], PxVec3(0.0f, 0.0f, 1.0f), SPACING);

			// Mark all shapes as NET group
			setNetFilterData(gGridBodies[idx]);

			// Anchor four corners only
			bool isCorner = (row == 0 || row == GRID_H - 1) &&
							(col == 0 || col == GRID_W - 1);
			if (isCorner)
			{
				gGridBodies[idx]->setRigidBodyFlag(
					PxRigidBodyFlag::eKINEMATIC, true);
				gGridBodies[idx]->setKinematicTarget(PxTransform(pos));
			}
		}
	}

	// =====================================================================
	// Spherical joints between adjacent nodes
	// =====================================================================
	PxVec3 halfX(SPACING * 0.5f, 0.0f, 0.0f);
	PxVec3 halfZ(0.0f, 0.0f, SPACING * 0.5f);

	for (PxU32 row = 0; row < GRID_H; ++row)
	{
		for (PxU32 col = 0; col < GRID_W; ++col)
		{
			PxU32 idx = row * GRID_W + col;

			if (col + 1 < GRID_W)
			{
				PxU32 right = row * GRID_W + (col + 1);
				createSphericalJoint(
					gGridBodies[idx], halfX,
					gGridBodies[right], -halfX);
			}
			if (row + 1 < GRID_H)
			{
				PxU32 below = (row + 1) * GRID_W + col;
				createSphericalJoint(
					gGridBodies[idx], halfZ,
					gGridBodies[below], -halfZ);
			}
		}
	}

	// =====================================================================
	// Heavy ball dropping from above
	// =====================================================================
	{
		PxVec3 ballPos(0.0f, BALL_DROP_HEIGHT, 0.0f);
		gBall = PxCreateDynamic(
			*gPhysics, PxTransform(ballPos),
			PxSphereGeometry(BALL_RADIUS), *gMaterial, BALL_DENSITY);
		gBall->setAngularDamping(0.5f);
		gScene->addActor(*gBall);
	}

	printf("SnippetChainmail: %ux%u mesh (%u struts), ball mass=%.0f\n",
		   GRID_W, GRID_H,
		   (GRID_W - 1) * GRID_H + GRID_W * (GRID_H - 1),
		   gBall->getMass());
}

void stepPhysics(bool /*interactive*/)
{
	gScene->simulate(1.0f / 60.0f);
	gScene->fetchResults(true);
}

void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PxCloseExtensions();
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);

	printf("SnippetChainmail done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch (toupper(key))
	{
	case ' ':
		{
			PxRigidDynamic* ball = PxCreateDynamic(
				*gPhysics, camera,
				PxSphereGeometry(1.0f), *gMaterial, 100.0f);
			ball->setLinearVelocity(camera.rotate(PxVec3(0, 0, -1)) * 50.0f);
			gScene->addActor(*ball);
		}
		break;
	case 'R':
		cleanupPhysics(false);
		initPhysics(false);
		break;
	}
}

int snippetMain(int, const char*const*)
{
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	static const PxU32 frameCount = 600;
	initPhysics(false);
	for (PxU32 i = 0; i < frameCount; i++)
	{
		stepPhysics(false);
		if (i % 60 == 0)
		{
			PxVec3 ballPos = gBall->getGlobalPose().p;
			printf("  frame %u: ball at (%.2f, %.2f, %.2f)\n",
				   i, ballPos.x, ballPos.y, ballPos.z);
		}
	}
	cleanupPhysics(false);
#endif
	return 0;
}
