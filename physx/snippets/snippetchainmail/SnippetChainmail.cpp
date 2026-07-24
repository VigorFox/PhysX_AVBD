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
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include <cstdio>
#include <string>

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation		= NULL;
static PxPhysics*				gPhysics		= NULL;
static PxDefaultCpuDispatcher*	gDispatcher		= NULL;
static PxScene*					gScene			= NULL;
static PxMaterial*				gMaterial		= NULL;
static PxPvd*					gPvd			= NULL;
static bool						gExtensionsInitialized = false;
static Snippets::HeadlessOptions gHeadlessOptions;

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
static const PxU32 MAX_JOINTS =
	(GRID_W - 1) * GRID_H + GRID_W * (GRID_H - 1);
static PxSphericalJoint* gJoints[MAX_JOINTS] = {};
static PxU32 gJointCount = 0;

struct ChainmailMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 callbackCount;
	PxU32 pairCount;
	PxU32 pointCount;
	PxU32 ballNetPairs;
	PxU32 ballNetPoints;
	PxU32 movingNetBodies;
	PxU32 sleepingNetBodies;
	PxReal initialBallY;
	PxReal finalBallY;
	PxReal minBallY;
	PxReal maxBallSpeed;
	PxReal finalCenterY;
	PxReal minNetY;
	PxReal maxNetSpeed;
	PxReal maxAnchorError;
	PxReal maxCornerDrift;
	PxU32 cleanupComplete;

	ChainmailMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0),
	  callbackCount(0), pairCount(0), pointCount(0), ballNetPairs(0),
	  ballNetPoints(0), movingNetBodies(0), sleepingNetBodies(0),
	  initialBallY(0.0f), finalBallY(0.0f), minBallY(PX_MAX_F32),
	  maxBallSpeed(0.0f), finalCenterY(0.0f), minNetY(PX_MAX_F32),
	  maxNetSpeed(0.0f), maxAnchorError(0.0f),
	  maxCornerDrift(0.0f), cleanupComplete(0)
	{
	}
};

static ChainmailMetrics gMetrics;

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 600;
	defaults.caseName = "impact";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") +
				(argv[i] ? argv[i] : "<null>");
			return false;
		}
	if(!Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), "impact"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames < 360)
	{
		error = "--frames must be at least 360";
		return false;
	}
	return true;
}

class ChainmailContactCallback : public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count)
	{
		PX_UNUSED(constraints);
		PX_UNUSED(count);
	}
	void onWake(PxActor** actors, PxU32 count)
	{
		PX_UNUSED(actors);
		PX_UNUSED(count);
	}
	void onSleep(PxActor** actors, PxU32 count)
	{
		PX_UNUSED(actors);
		PX_UNUSED(count);
	}
	void onTrigger(PxTriggerPair* pairs, PxU32 count)
	{
		PX_UNUSED(pairs);
		PX_UNUSED(count);
	}
	void onAdvance(
		const PxRigidBody* const* bodies,
		const PxTransform* poses, const PxU32 count)
	{
		PX_UNUSED(bodies);
		PX_UNUSED(poses);
		PX_UNUSED(count);
	}
	void onContact(
		const PxContactPairHeader& pairHeader,
		const PxContactPair* pairs, PxU32 nbPairs)
	{
		++gMetrics.callbackCount;
		gMetrics.pairCount += nbPairs;
		const bool ballNet =
			(pairHeader.actors[0] && pairHeader.actors[1]) &&
			((pairHeader.actors[0]->userData == gBall &&
			  pairHeader.actors[1]->userData == gGridBodies) ||
			 (pairHeader.actors[1]->userData == gBall &&
			  pairHeader.actors[0]->userData == gGridBodies));
		if(ballNet)
			gMetrics.ballNetPairs += nbPairs;
		for(PxU32 i = 0; i < nbPairs; ++i)
		{
			gMetrics.pointCount += pairs[i].contactCount;
			if(ballNet)
				gMetrics.ballNetPoints += pairs[i].contactCount;
		}
	}
};

static ChainmailContactCallback gContactCallback;

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
	if(gHeadlessOptions.headless)
		pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
			PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
			PxPairFlag::eNOTIFY_CONTACT_POINTS;
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
	PxSphericalJoint* joint = PxSphericalJointCreate(
		*gPhysics,
		a0, PxTransform(anchor0),
		a1, PxTransform(anchor1));
	if(joint && gJointCount < MAX_JOINTS)
		gJoints[gJointCount++] = joint;
	// No cone limit -- pure free ball joint for maximum softness
}

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------
void initPhysics(bool /*interactive*/)
{
	gJointCount = 0;
	for(PxU32 i = 0; i < MAX_JOINTS; ++i)
		gJoints[i] = NULL;
	for(PxU32 i = 0; i < GRID_W * GRID_H; ++i)
		gGridBodies[i] = NULL;
	gBall = NULL;
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless ?
			gHeadlessOptions.dispatcherThreads : 4);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = chainmailFilterShader;   // custom filter!
	sceneDesc.solverType = gHeadlessOptions.headless ?
		gHeadlessOptions.solverType : PxSolverType::eAVBD;
	sceneDesc.simulationEventCallback =
		gHeadlessOptions.headless ? &gContactCallback : NULL;
	gScene = gPhysics->createScene(sceneDesc);

	if(!gHeadlessOptions.headless)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if (pvdClient)
		{
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
		}
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
			gGridBodies[idx]->userData = gGridBodies;

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
		gBall->userData = gBall;
		gScene->addActor(*gBall);
	}
	if(gHeadlessOptions.headless)
	{
		gMetrics.initialBallY = gBall->getGlobalPose().p.y;
		gMetrics.finalBallY = gMetrics.initialBallY;
	}

	printf("SnippetChainmail: %ux%u mesh (%u struts), ball mass=%.0f\n",
		   GRID_W, GRID_H,
		   (GRID_W - 1) * GRID_H + GRID_W * (GRID_H - 1),
		   gBall->getMass());
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f / 60.0f;
	if(!gScene->simulate(dt) && gHeadlessOptions.headless)
		++gMetrics.fetchFailures;
	const bool fetched = gScene->fetchResults(true);
	if(!gHeadlessOptions.headless)
		return;
	if(!fetched)
		++gMetrics.fetchFailures;
	++gMetrics.completedFrames;

	gMetrics.movingNetBodies = 0;
	gMetrics.sleepingNetBodies = 0;
	const PxReal offsetX = -SPACING * (GRID_W - 1) * 0.5f;
	const PxReal offsetZ = -SPACING * (GRID_H - 1) * 0.5f;
	for(PxU32 row = 0; row < GRID_H; ++row)
	{
		for(PxU32 col = 0; col < GRID_W; ++col)
		{
			const PxU32 idx = row * GRID_W + col;
			const PxRigidDynamic* body = gGridBodies[idx];
			const PxTransform pose = body->getGlobalPose();
			const PxVec3 linearVelocity = body->getLinearVelocity();
			const PxVec3 angularVelocity = body->getAngularVelocity();
			if(!pose.isFinite() || !linearVelocity.isFinite() ||
				!angularVelocity.isFinite())
				++gMetrics.nonFinite;
			const PxReal speed = linearVelocity.magnitude();
			gMetrics.minNetY = PxMin(gMetrics.minNetY, pose.p.y);
			gMetrics.maxNetSpeed = PxMax(gMetrics.maxNetSpeed, speed);
			if(speed > 1e-3f || angularVelocity.magnitude() > 1e-3f)
				++gMetrics.movingNetBodies;
			if(body->isSleeping())
				++gMetrics.sleepingNetBodies;
			const bool corner = (row == 0 || row == GRID_H - 1) &&
				(col == 0 || col == GRID_W - 1);
			if(corner)
			{
				const PxVec3 initial(
					offsetX + col * SPACING, MESH_HEIGHT,
					offsetZ + row * SPACING);
				gMetrics.maxCornerDrift = PxMax(
					gMetrics.maxCornerDrift,
					(pose.p - initial).magnitude());
			}
		}
	}
	gMetrics.finalCenterY =
		gGridBodies[(GRID_H / 2) * GRID_W + GRID_W / 2]->
			getGlobalPose().p.y;

	for(PxU32 i = 0; i < gJointCount; ++i)
	{
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		gJoints[i]->getActors(actor0, actor1);
		if(!actor0 || !actor1)
		{
			++gMetrics.nonFinite;
			continue;
		}
		const PxVec3 world0 =
			actor0->getGlobalPose().transform(
				gJoints[i]->getLocalPose(PxJointActorIndex::eACTOR0)).p;
		const PxVec3 world1 =
			actor1->getGlobalPose().transform(
				gJoints[i]->getLocalPose(PxJointActorIndex::eACTOR1)).p;
		const PxReal error = (world0 - world1).magnitude();
		if(!world0.isFinite() || !world1.isFinite() || !PxIsFinite(error))
			++gMetrics.nonFinite;
		gMetrics.maxAnchorError =
			PxMax(gMetrics.maxAnchorError, error);
	}

	const PxTransform ballPose = gBall->getGlobalPose();
	const PxVec3 ballVelocity = gBall->getLinearVelocity();
	if(!ballPose.isFinite() || !ballVelocity.isFinite() ||
		!gBall->getAngularVelocity().isFinite())
		++gMetrics.nonFinite;
	gMetrics.finalBallY = ballPose.p.y;
	gMetrics.minBallY = PxMin(gMetrics.minBallY, ballPose.p.y);
	gMetrics.maxBallSpeed =
		PxMax(gMetrics.maxBallSpeed, ballVelocity.magnitude());
}

void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	for(PxU32 i = 0; i < MAX_JOINTS; ++i)
		gJoints[i] = NULL;
	for(PxU32 i = 0; i < GRID_W * GRID_H; ++i)
		gGridBodies[i] = NULL;
	gJointCount = 0;
	gBall = NULL;
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
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
		!gScene && !gMaterial && !gDispatcher && !gPhysics &&
		!gPvd && !gFoundation && !gExtensionsInitialized &&
		!gBall && gJointCount == 0 ? 1u : 0u;

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

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig("SnippetChainmail", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized &&
		gDispatcher && gScene && gMaterial && gBall &&
		gJointCount == MAX_JOINTS;
	if(initialized)
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);

	const char* reason = "none";
	bool passed = true;
	if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(gMetrics.completedFrames != gHeadlessOptions.frames ||
		gMetrics.fetchFailures != 0)
	{
		passed = false;
		reason = "incomplete_simulation";
	}
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.ballNetPairs == 0 || gMetrics.ballNetPoints == 0 ||
		gMetrics.movingNetBodies == 0 ||
		gMetrics.minBallY >= gMetrics.initialBallY - 5.0f ||
		gMetrics.maxCornerDrift > 1e-4f)
	{
		passed = false;
		reason = "missing_large_island_activity";
	}
	else if(gHeadlessOptions.solverType == PxSolverType::eTGS &&
		(gMetrics.finalBallY > 3.0f || gMetrics.minBallY > 3.0f))
	{
		passed = false;
		reason = "tgs_negative_control_not_distinct";
	}
	else if(gHeadlessOptions.solverType == PxSolverType::eAVBD &&
		(gMetrics.finalBallY < 20.0f || gMetrics.minBallY < 20.0f ||
		 gMetrics.finalCenterY < 20.0f || gMetrics.minNetY < 20.0f ||
		 gMetrics.maxNetSpeed > 100.0f ||
		 gMetrics.maxAnchorError > 1.0f ||
		 gMetrics.ballNetPoints < 100))
	{
		passed = false;
		reason = "avbd_large_island_not_retained";
	}

	const PxU32 jointCount = gJointCount;
	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetChainmail solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED gridBodies=%u jointCount=%u "
		"callbackCount=%u pairCount=%u pointCount=%u "
		"ballNetPairs=%u ballNetPoints=%u movingNetBodies=%u "
		"sleepingNetBodies=%u initialBallY=%.9g finalBallY=%.9g "
		"minBallY=%.9g maxBallSpeed=%.9g finalCenterY=%.9g "
		"minNetY=%.9g maxNetSpeed=%.9g maxAnchorError=%.9g "
		"maxCornerDrift=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, GRID_W * GRID_H,
		jointCount, gMetrics.callbackCount, gMetrics.pairCount,
		gMetrics.pointCount, gMetrics.ballNetPairs,
		gMetrics.ballNetPoints, gMetrics.movingNetBodies,
		gMetrics.sleepingNetBodies, double(gMetrics.initialBallY),
		double(gMetrics.finalBallY), double(gMetrics.minBallY),
		double(gMetrics.maxBallSpeed), double(gMetrics.finalCenterY),
		double(gMetrics.minNetY), double(gMetrics.maxNetSpeed),
		double(gMetrics.maxAnchorError),
		double(gMetrics.maxCornerDrift), gMetrics.nonFinite,
		gMetrics.fetchFailures, gErrorCallback.getFatalCount(),
		gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetChainmail "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetChainmail "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

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
