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
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  


// *******************************************************************************************************
// In addition to the simulate() function, which performs both collision detection and dynamics update,
// the PhysX SDK provides an api for separate execution of the collision detection and dynamics update steps.
// We shall refer to this feature as "split sim". This snippet demonstrates two ways to use the split sim feature 
// so that application work can be performed concurrently with the collision detection step.

// The snippet creates a list of kinematic box actors along with a number of dynamic actors that
// interact with the kinematic actors. 

//The defines OVERLAP_COLLISION_AND_RENDER_WITH_NO_LAG and OVERLAP_COLLISION_AND_RENDER_WITH_ONE_FRAME_LAG 
//demonstrate two distinct modes of split sim operation:

// (1)Enabling OVERLAP_COLLISION_AND_RENDER_WITH_NO_LAG allows the collision detection step to run in parallel 
//    with the renderer and with the update of the kinematic target poses without introducing any lag between 
//    application time and physics time.  This is equivalent to calling simulate() and fetchResults() with the key 
//    difference being that the application can schedule work to run concurrently with the collision detection.  
//    A consequence of this approach is that the first frame is more expensive than subsequent frames because it has to 
//    perform blocking collision detection and dynamics update calls.

// (2)OVERLAP_COLLISION_AND_RENDER_WITH_ONE_FRAME_LAG also allows the collision to run in parallel with 
//    the renderer and the update of the kinematic target poses but this time with a lag between physics time and 
//    application time; that is, the physics is always a single timestep behind the application because the first
//    frame merely starts the collision detection for the subsequent frame.  A consequence of this approach is that 
//    the first frame is cheaper than subsequent frames.
// ********************************************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include <cstdio>
#include <string>

//This will allow the split sim to overlap collision and render and game logic.
#define OVERLAP_COLLISION_AND_RENDER_WITH_NO_LAG  1
#define OVERLAP_COLLISION_AND_RENDER_WITH_ONE_FRAME_LAG 0

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;

#define NB_KINE_X	16
#define NB_KINE_Y	16
#define KINE_SCALE	3.1f

static bool isFirstFrame = true;
static PxU32 gActiveKineX = NB_KINE_X;
static PxU32 gActiveKineY = NB_KINE_Y;
static PxU32 gDynamicX = 8;
static PxU32 gDynamicY = 8;
static PxU32 gDynamicLayers = 3;
static PxReal gSimulationTime = 0.0f;
static Snippets::HeadlessOptions gHeadlessOptions;
static PxArray<PxRigidDynamic*> gDynamics;

PxRigidDynamic* gKinematics[NB_KINE_Y][NB_KINE_X];
PxTransform gKinematicTargets[NB_KINE_Y][NB_KINE_X];

struct SplitSimMetrics
{
	PxU32 completedFrames;
	PxU32 simulateCalls;
	PxU32 collideCalls;
	PxU32 fetchCollisionCalls;
	PxU32 advanceCalls;
	PxU32 fetchResultsCalls;
	PxU32 fetchFailures;
	PxU32 callbackCount;
	PxU32 pairCount;
	PxU32 pointCount;
	PxU32 nonFinite;
	PxU32 movingBodies;
	PxU32 sleepingBodies;
	PxReal maxCollisionPhasePoseDelta;
	PxReal maxTargetPositionError;
	PxReal sumX;
	PxReal sumY;
	PxReal sumZ;
	PxReal minY;
	PxReal maxY;
	PxReal sumSpeed;
	PxReal maxSpeed;
	PxReal maxObservedY;
	PxReal maxObservedSpeed;
	PxU32 firstUnsafeFrame;
	PxU32 cleanupComplete;

	SplitSimMetrics()
	: completedFrames(0), simulateCalls(0), collideCalls(0),
	  fetchCollisionCalls(0), advanceCalls(0), fetchResultsCalls(0),
	  fetchFailures(0), callbackCount(0), pairCount(0), pointCount(0),
	  nonFinite(0), movingBodies(0), sleepingBodies(0),
	  maxCollisionPhasePoseDelta(0.0f), maxTargetPositionError(0.0f),
	  sumX(0.0f), sumY(0.0f), sumZ(0.0f), minY(PX_MAX_F32),
	  maxY(-PX_MAX_F32), sumSpeed(0.0f), maxSpeed(0.0f),
	  maxObservedY(-PX_MAX_F32), maxObservedSpeed(0.0f),
	  firstUnsafeFrame(PX_MAX_U32), cleanupComplete(0)
	{
	}
};

static SplitSimMetrics gMetrics;

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), name);
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.caseName = "split";
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
	if(!isHeadlessCase("simulate") && !isHeadlessCase("split"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames < 120)
	{
		error = "--frames must be at least 120";
		return false;
	}
	return true;
}

static PxFilterFlags contactReportFilterShader(
	PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock,
	PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	pairFlags = PxPairFlag::eCONTACT_DEFAULT |
		PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
		PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

class SplitSimContactCallback : public PxSimulationEventCallback
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
		const PxRigidBody* const* bodyBuffer,
		const PxTransform* poseBuffer, const PxU32 count)
	{
		PX_UNUSED(bodyBuffer);
		PX_UNUSED(poseBuffer);
		PX_UNUSED(count);
	}
	void onContact(
		const PxContactPairHeader& pairHeader,
		const PxContactPair* pairs, PxU32 nbPairs)
	{
		PX_UNUSED(pairHeader);
		++gMetrics.callbackCount;
		gMetrics.pairCount += nbPairs;
		for(PxU32 i = 0; i < nbPairs; ++i)
			gMetrics.pointCount += pairs[i].contactCount;
	}
};

static SplitSimContactCallback gContactCallback;

void createDynamics()
{
	const PxU32 NbX = gDynamicX;
	const PxU32 NbY = gDynamicY;

	const PxVec3 dims(0.2f, 0.1f, 0.2f);
	const PxReal sphereRadius = 0.2f;
	const PxReal capsuleRadius = 0.2f;
	const PxReal halfHeight = 0.5f;

	const PxU32 NbLayers = gDynamicLayers;
	const float YScale = 0.4f;
	const float YStart = 6.0f;
	PxShape* boxShape = gPhysics->createShape(PxBoxGeometry(dims), *gMaterial);
	PxShape* sphereShape = gPhysics->createShape(PxSphereGeometry(sphereRadius), *gMaterial);
	PxShape* capsuleShape = gPhysics->createShape(PxCapsuleGeometry(capsuleRadius, halfHeight), *gMaterial);
	PX_UNUSED(boxShape);
	PX_UNUSED(sphereShape);
	PX_UNUSED(capsuleShape);
	for(PxU32 j=0;j<NbLayers;j++)
	{
		const float angle = float(j)*0.08f;
		const PxQuat rot = PxGetRotYQuat(angle);

		const float ScaleX = 4.0f;
		const float ScaleY = 4.0f;

		for(PxU32 y=0;y<NbY;y++)
		{
			for(PxU32 x=0;x<NbX;x++)
			{
				const float xf = (float(x)-float(NbX)*0.5f)*ScaleX;
				const float yf = (float(y)-float(NbY)*0.5f)*ScaleY;

				PxRigidDynamic* dynamic = NULL;

				PxU32 v = j&3;
				PxVec3 pos = PxVec3(xf, YStart + float(j)*YScale, yf);

				switch(v)
				{
					case 0:
						{
							PxTransform pose(pos, rot);
							dynamic = gPhysics->createRigidDynamic(pose);
							dynamic->attachShape(*boxShape);
							break;
						}
					case 1:
						{
							PxTransform pose(pos, PxQuat(PxIdentity));
							dynamic = gPhysics->createRigidDynamic(pose);
							dynamic->attachShape(*sphereShape);
							break;
						}
					default:
						{
							PxTransform pose(pos, rot);
							dynamic = gPhysics->createRigidDynamic(pose);
							dynamic->attachShape(*capsuleShape);
							break;
						}
				};

				PxRigidBodyExt::updateMassAndInertia(*dynamic, 10.f);

				gScene->addActor(*dynamic);
				gDynamics.pushBack(dynamic);
			}
		}
	}
	boxShape->release();
	sphereShape->release();
	capsuleShape->release();
}

void createGroudPlane()
{
	PxTransform pose = PxTransform(PxVec3(0.0f, 0.0f, 0.0f),PxQuat(PxHalfPi, PxVec3(0.0f, 0.0f, 1.0f)));
	PxRigidStatic* actor = gPhysics->createRigidStatic(pose);
	PxShape* shape = PxRigidActorExt::createExclusiveShape(*actor, PxPlaneGeometry(), *gMaterial);
	PX_UNUSED(shape);
	gScene->addActor(*actor);
}

void createKinematics()
{
	const PxU32 NbX = gActiveKineX;
	const PxU32 NbY = gActiveKineY;

	const PxVec3 dims(1.5f, 0.2f, 1.5f);
	const PxQuat rot = PxQuat(PxIdentity);

	const float YScale = 0.4f;
	
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(dims), *gMaterial);

	
	const float ScaleX = KINE_SCALE;
	const float ScaleY = KINE_SCALE;
	for(PxU32 y=0;y<NbY;y++)
	{
		for(PxU32 x=0;x<NbX;x++)
		{
			const float xf = (float(x)-float(NbX)*0.5f)*ScaleX;
			const float yf = (float(y)-float(NbY)*0.5f)*ScaleY;
			PxTransform pose(PxVec3(xf, 0.2f + YScale, yf), rot);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(pose);
			body->attachShape(*shape);
			gScene->addActor(*body);
			body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);

			gKinematics[y][x] = body;
		}
	}
	shape->release();
}

void updateKinematicTargets(PxReal timeStep)
{
	const float YScale = 0.4f;
	
	PxTransform motion;
	motion.q = PxQuat(PxIdentity);

	gSimulationTime += timeStep;

	const PxU32 NbX = gActiveKineX;
	const PxU32 NbY = gActiveKineY;

	const float Coeff = 0.2f;

	const float ScaleX = KINE_SCALE;
	const float ScaleY = KINE_SCALE;
	for(PxU32 y=0;y<NbY;y++)
	{
		for(PxU32 x=0;x<NbX;x++)
		{
			const float xf = (float(x)-float(NbX)*0.5f)*ScaleX;
			const float yf = (float(y)-float(NbY)*0.5f)*ScaleY;

			const float h =
				sinf(gSimulationTime*2.0f + float(x)*Coeff +
					float(y)*Coeff)*2.0f;
			motion.p = PxVec3(xf, h + 2.0f + YScale, yf);

			gKinematicTargets[y][x] = motion;
		}
	}
}

void applyKinematicTargets()
{
	const PxU32 NbX = gActiveKineX;
	const PxU32 NbY = gActiveKineY;

	for(PxU32 y=0;y<NbY;y++)
	{
		for(PxU32 x=0;x<NbX;x++)
		{
			PxRigidDynamic* kine = gKinematics[y][x];
			const PxTransform& target = gKinematicTargets[y][x];
			kine->setKinematicTarget(target);
			if(gHeadlessOptions.headless)
			{
				PxTransform readback(PxIdentity);
				if(!kine->getKinematicTarget(readback))
					++gMetrics.nonFinite;
				else
				{
					gMetrics.maxTargetPositionError = PxMax(
						gMetrics.maxTargetPositionError,
						(readback.p - target.p).magnitude());
					if(!readback.isFinite())
						++gMetrics.nonFinite;
				}
			}
		}
	}
}

void initPhysics(bool /*interactive*/)
{
	isFirstFrame = true;
	gSimulationTime = 0.0f;
	gDynamics.clear();
	if(gHeadlessOptions.headless)
	{
		gActiveKineX = 4;
		gActiveKineY = 4;
		gDynamicX = 4;
		gDynamicY = 4;
		gDynamicLayers = 2;
	}
	else
	{
		gActiveKineX = NB_KINE_X;
		gActiveKineY = NB_KINE_Y;
		gDynamicX = 8;
		gDynamicY = 8;
		gDynamicLayers = 3;
	}
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless ?
			gHeadlessOptions.dispatcherThreads : 2);
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader = gHeadlessOptions.headless ?
		contactReportFilterShader : PxDefaultSimulationFilterShader;
	sceneDesc.simulationEventCallback = gHeadlessOptions.headless ?
		&gContactCallback : NULL;
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gHeadlessOptions.headless)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
		{
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
		}
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	createKinematics();
	createDynamics();

}

static void recordDynamicState()
{
	gMetrics.movingBodies = 0;
	gMetrics.sleepingBodies = 0;
	gMetrics.sumX = 0.0f;
	gMetrics.sumY = 0.0f;
	gMetrics.sumZ = 0.0f;
	gMetrics.minY = PX_MAX_F32;
	gMetrics.maxY = -PX_MAX_F32;
	gMetrics.sumSpeed = 0.0f;
	gMetrics.maxSpeed = 0.0f;
	for(PxU32 i = 0; i < gDynamics.size(); ++i)
	{
		const PxRigidDynamic* body = gDynamics[i];
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 linearVelocity = body->getLinearVelocity();
		const PxVec3 angularVelocity = body->getAngularVelocity();
		const PxReal speed = linearVelocity.magnitude();
		if(!pose.isFinite() || !linearVelocity.isFinite() ||
			!angularVelocity.isFinite() || !PxIsFinite(speed))
			++gMetrics.nonFinite;
		if(speed > 1e-3f || angularVelocity.magnitude() > 1e-3f)
			++gMetrics.movingBodies;
		if(body->isSleeping())
			++gMetrics.sleepingBodies;
		gMetrics.sumX += pose.p.x;
		gMetrics.sumY += pose.p.y;
		gMetrics.sumZ += pose.p.z;
		gMetrics.minY = PxMin(gMetrics.minY, pose.p.y);
		gMetrics.maxY = PxMax(gMetrics.maxY, pose.p.y);
		gMetrics.sumSpeed += speed;
		gMetrics.maxSpeed = PxMax(gMetrics.maxSpeed, speed);
		gMetrics.maxObservedY =
			PxMax(gMetrics.maxObservedY, pose.p.y);
		gMetrics.maxObservedSpeed =
			PxMax(gMetrics.maxObservedSpeed, speed);
		if(gMetrics.firstUnsafeFrame == PX_MAX_U32 &&
			(pose.p.y > 50.0f || speed > 50.0f))
			gMetrics.firstUnsafeFrame = gMetrics.completedFrames;
	}
}

static void stepHeadless()
{
	const PxReal timeStep = gHeadlessOptions.dt;
	if(isHeadlessCase("simulate"))
	{
		updateKinematicTargets(timeStep);
		applyKinematicTargets();
		++gMetrics.simulateCalls;
		if(!gScene->simulate(timeStep))
			++gMetrics.fetchFailures;
		++gMetrics.fetchResultsCalls;
		if(!gScene->fetchResults(true))
			++gMetrics.fetchFailures;
	}
	else
	{
		PxArray<PxTransform> posesBeforeCollision;
		posesBeforeCollision.reserve(gDynamics.size());
		for(PxU32 i = 0; i < gDynamics.size(); ++i)
			posesBeforeCollision.pushBack(
				gDynamics[i]->getGlobalPose());

		++gMetrics.collideCalls;
		if(!gScene->collide(timeStep))
			++gMetrics.fetchFailures;
		updateKinematicTargets(timeStep);
		++gMetrics.fetchCollisionCalls;
		if(!gScene->fetchCollision(true))
			++gMetrics.fetchFailures;

		for(PxU32 i = 0; i < gDynamics.size(); ++i)
		{
			const PxTransform poseAfterCollision =
				gDynamics[i]->getGlobalPose();
			const PxTransform& poseBeforeCollision =
				posesBeforeCollision[i];
			const PxReal positionDelta =
				(poseAfterCollision.p - poseBeforeCollision.p).magnitude();
			const PxReal quaternionDelta =
				PxAbs(poseAfterCollision.q.x - poseBeforeCollision.q.x) +
				PxAbs(poseAfterCollision.q.y - poseBeforeCollision.q.y) +
				PxAbs(poseAfterCollision.q.z - poseBeforeCollision.q.z) +
				PxAbs(poseAfterCollision.q.w - poseBeforeCollision.q.w);
			gMetrics.maxCollisionPhasePoseDelta = PxMax(
				gMetrics.maxCollisionPhasePoseDelta,
				PxMax(positionDelta, quaternionDelta));
		}

		applyKinematicTargets();
		++gMetrics.advanceCalls;
		if(!gScene->advance())
			++gMetrics.fetchFailures;
		++gMetrics.fetchResultsCalls;
		if(!gScene->fetchResults(true))
			++gMetrics.fetchFailures;
	}
	++gMetrics.completedFrames;
	recordDynamicState();
}

#if OVERLAP_COLLISION_AND_RENDER_WITH_NO_LAG
void stepPhysics(bool /*interactive*/)
{
	if(gHeadlessOptions.headless)
	{
		stepHeadless();
		return;
	}
	const PxReal timeStep = 1.0f/60.0f;

	if(isFirstFrame)
	{
		//Run the first frame's collision detection
		gScene->collide(timeStep);
		isFirstFrame = false;
	}
	//update the kinematice target pose in parallel with collision running
	updateKinematicTargets(timeStep);
	gScene->fetchCollision(true);
	//apply the computed and buffered kinematic target poses
	applyKinematicTargets();
	gScene->advance();
	gScene->fetchResults(true); 
	
	//Run the deferred collision detection for the next frame. This will run in parallel with render.
	gScene->collide(timeStep);
}
#elif OVERLAP_COLLISION_AND_RENDER_WITH_ONE_FRAME_LAG

void stepPhysics(bool /*interactive*/)
{
	if(gHeadlessOptions.headless)
	{
		stepHeadless();
		return;
	}
	PxReal timeStep = 1.0/60.0f;

	//update the kinematice target pose in parallel with collision running
	updateKinematicTargets(timeStep);
	if(!isFirstFrame)
	{
		gScene->fetchCollision(true);
		//apply the computed and buffered kinematic target poses
		applyKinematicTargets();
		gScene->advance();
		gScene->fetchResults(true); 
	}
	else
		applyKinematicTargets();

	isFirstFrame = false;
	//Run the deferred collision detection for the next frame. This will run in parallel with render.
	gScene->collide(timeStep);
}

#else

void stepPhysics(bool /*interactive*/)
{
	if(gHeadlessOptions.headless)
	{
		stepHeadless();
		return;
	}
	PxReal timeStep = 1.0/60.0f;
	//update the kinematice target pose in parallel with collision running
	gScene->collide(timeStep);
	updateKinematicTargets(timeStep);
	gScene->fetchCollision(true);
	//apply the computed and buffered kinematic target poses
	applyKinematicTargets();
	gScene->advance();
	gScene->fetchResults(true); 
}
#endif

void cleanupPhysics(bool /*interactive*/)
{
#if OVERLAP_COLLISION_AND_RENDER_WITH_NO_LAG || OVERLAP_COLLISION_AND_RENDER_WITH_ONE_FRAME_LAG
	//Close out remainder of previously running scene. If we don't do this, it will be implicitly done
	//in gScene->release() but a warning will be issued.
	if(!gHeadlessOptions.headless)
	{
		gScene->fetchCollision(true);
		gScene->advance();
		gScene->fetchResults(true);
	}
#endif

	PX_RELEASE(gScene);
	gDynamics.reset();
	for(PxU32 y = 0; y < NB_KINE_Y; ++y)
		for(PxU32 x = 0; x < NB_KINE_X; ++x)
			gKinematics[y][x] = NULL;
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
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
		!gPvd && !gFoundation && gDynamics.empty() ? 1u : 0u;
	
	printf("SnippetSplitSim done.\n");
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig("SnippetSplitSim", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gScene && gMaterial &&
		gDynamics.size() == gDynamicX * gDynamicY * gDynamicLayers;
	if(initialized)
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);

	const bool splitCase = isHeadlessCase("split");
	const PxU32 expectedSplitCalls =
		splitCase ? gHeadlessOptions.frames : 0;
	const PxU32 expectedSimulateCalls =
		splitCase ? 0 : gHeadlessOptions.frames;
	const char* reason = "none";
	bool passed = true;
	if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(gMetrics.completedFrames != gHeadlessOptions.frames ||
		gMetrics.simulateCalls != expectedSimulateCalls ||
		gMetrics.collideCalls != expectedSplitCalls ||
		gMetrics.fetchCollisionCalls != expectedSplitCalls ||
		gMetrics.advanceCalls != expectedSplitCalls ||
		gMetrics.fetchResultsCalls != gHeadlessOptions.frames ||
		gMetrics.fetchFailures != 0)
	{
		passed = false;
		reason = "api_sequence_incomplete";
	}
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0 ||
		gMetrics.maxCollisionPhasePoseDelta > 1e-6f ||
		gMetrics.maxTargetPositionError > 1e-6f)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.callbackCount == 0 || gMetrics.pairCount == 0 ||
		gMetrics.pointCount == 0 || gMetrics.movingBodies == 0 ||
		gMetrics.minY < -10.0f || gMetrics.maxSpeed > 100.0f)
	{
		passed = false;
		reason = "missing_scene_activity";
	}

	const PxU32 dynamicBodyCount = gDynamics.size();
	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetSplitSim solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED dynamicBodies=%u "
		"simulateCalls=%u collideCalls=%u fetchCollisionCalls=%u "
		"advanceCalls=%u fetchResultsCalls=%u fetchFailures=%u "
		"callbackCount=%u pairCount=%u pointCount=%u movingBodies=%u "
		"sleepingBodies=%u nonFinite=%u "
		"maxCollisionPhasePoseDelta=%.9g maxTargetPositionError=%.9g "
		"sumX=%.9g sumY=%.9g sumZ=%.9g minY=%.9g maxY=%.9g "
		"sumSpeed=%.9g maxSpeed=%.9g maxObservedY=%.9g "
		"maxObservedSpeed=%.9g firstUnsafeFrame=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, dynamicBodyCount,
		gMetrics.simulateCalls, gMetrics.collideCalls,
		gMetrics.fetchCollisionCalls, gMetrics.advanceCalls,
		gMetrics.fetchResultsCalls, gMetrics.fetchFailures,
		gMetrics.callbackCount, gMetrics.pairCount, gMetrics.pointCount,
		gMetrics.movingBodies, gMetrics.sleepingBodies,
		gMetrics.nonFinite,
		double(gMetrics.maxCollisionPhasePoseDelta),
		double(gMetrics.maxTargetPositionError),
		double(gMetrics.sumX), double(gMetrics.sumY),
		double(gMetrics.sumZ), double(gMetrics.minY),
		double(gMetrics.maxY), double(gMetrics.sumSpeed),
		double(gMetrics.maxSpeed), double(gMetrics.maxObservedY),
		double(gMetrics.maxObservedSpeed), gMetrics.firstUnsafeFrame,
		gErrorCallback.getFatalCount(),
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
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetSplitSim "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetSplitSim "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	static const PxU32 frameCount = 100;
	initPhysics(false);
	for(PxU32 i=0; i<frameCount; i++)
		stepPhysics(false);
	cleanupPhysics(false);
#endif

	return 0;
}
