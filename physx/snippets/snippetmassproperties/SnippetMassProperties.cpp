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

// ****************************************************************************
// This snippet illustrates different ways of setting mass for rigid bodies.
//
// It creates 5 snowmen with different mass properties:
// - massless with a weight at the bottom
// - only the mass of the lowest snowball
// - the mass of all the snowballs
// - the whole mass but with a low center of gravity
// - manual setup of masses
//
// The different mass properties can be visually inspected by firing a rigid 
// ball towards each snowman using the space key.
// 
// For more details, please consult the "Rigid Body Dynamics" section of the 
// user guide.
// 
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include <cfloat>
#include <cstdio>
#include <string>

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd = NULL;
static Snippets::HeadlessOptions gHeadlessOptions;
static PxRigidDynamic*			gSnowmen[6] = { NULL };
static PxU32					gSnowmanCount = 0;

struct MassResponse
{
	PxReal mass;
	PxVec3 expectedLinearVelocity;
	PxVec3 expectedAngularVelocity;
	PxVec3 observedLinearVelocity;
	PxVec3 observedAngularVelocity;
};

struct MassPropertiesMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 cleanupComplete;
	PxReal massMin;
	PxReal massMax;
	PxReal linearScaleMin;
	PxReal linearScaleMax;
	PxReal angularScaleMin;
	PxReal angularScaleMax;
	PxReal maxLinearResidual;
	PxReal maxAngularResidual;
	PxReal rotatedOmegaX;
	PxReal rotatedOmegaZ;
	MassResponse responses[6];

	MassPropertiesMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0),
	  cleanupComplete(0), massMin(FLT_MAX), massMax(0.0f),
	  linearScaleMin(FLT_MAX), linearScaleMax(-FLT_MAX),
	  angularScaleMin(FLT_MAX), angularScaleMax(-FLT_MAX),
	  maxLinearResidual(0.0f), maxAngularResidual(0.0f),
	  rotatedOmegaX(0.0f), rotatedOmegaZ(0.0f)
	{
	}
};

static MassPropertiesMetrics gMetrics;


// create a dynamic ball to throw at the snowmen.
static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0))
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static PxRigidDynamic* createSnowMan(const PxTransform& pos, PxU32 mode)
{
	PxRigidDynamic* snowmanActor = gPhysics->createRigidDynamic(PxTransform(pos));
	if(!snowmanActor)
	{
		printf("create snowman actor failed");
		return NULL;
	}

	PxShape* armL = NULL; PxShape* armR = NULL;

	switch(mode%5)
	{
	case 0: // with a weight at the bottom
		{
			PxShape* shape = NULL;
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.2), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,-.29,0)));
			
			PxRigidBodyExt::updateMassAndInertia(*snowmanActor,10);

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.5), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.4), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,.6,0)));
			
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.3), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,1.1,0)));
			
			armL = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armL)
				printf("creating snowman shape failed");
			armL->setLocalPose(PxTransform(PxVec3(-.4,.7,0)));

			armR = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armR)
				printf("creating snowman shape failed");			
			armR->setLocalPose(PxTransform(PxVec3( .4,.7,0)));
		}
		break;
	case 1: // only considering lowest shape mass
		{
			PxShape* shape = NULL;
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.5), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			
			PxRigidBodyExt::updateMassAndInertia(*snowmanActor,1);

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.4), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,.6,0)));

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.3), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,1.1,0)));

			armL = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armL)
				printf("creating snowman shape failed");
			armL->setLocalPose(PxTransform(PxVec3(-.4,.7,0)));

			armR = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armR)
				printf("creating snowman shape failed");			
			armR->setLocalPose(PxTransform(PxVec3( .4,.7,0)));

			snowmanActor->setCMassLocalPose(PxTransform(PxVec3(0,-.5,0)));
		}
		break;
	case 2: // considering whole mass
		{
			PxShape* shape = NULL;
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.5), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.4), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,.6,0)));

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.3), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,1.1,0)));
			
			armL = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armL)
				printf("creating snowman shape failed");
			armL->setLocalPose(PxTransform(PxVec3(-.4,.7,0)));

			armR = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armR)
				printf("creating snowman shape failed");			
			armR->setLocalPose(PxTransform(PxVec3( .4,.7,0)));

			PxRigidBodyExt::updateMassAndInertia(*snowmanActor,1);
			snowmanActor->setCMassLocalPose(PxTransform(PxVec3(0,-.5,0)));
		}
		break;
	case 3: // considering whole mass with low COM
		{
			PxShape* shape = NULL;
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.5), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.4), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,.6,0)));

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.3), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,1.1,0)));

			armL = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armL)
				printf("creating snowman shape failed");
			armL->setLocalPose(PxTransform(PxVec3(-.4,.7,0)));

			armR = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armR)
				printf("creating snowman shape failed");			
			armR->setLocalPose(PxTransform(PxVec3( .4,.7,0)));

			const PxVec3 localPos = PxVec3(0,-.5,0);
			PxRigidBodyExt::updateMassAndInertia(*snowmanActor,1,&localPos);
		}
		break;
	case 4: // setting up mass properties manually
		{
			PxShape* shape = NULL;
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.5), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			
			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.4), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,.6,0)));

			shape = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxSphereGeometry(.3), *gMaterial);
			if(!shape)
				printf("creating snowman shape failed");
			shape->setLocalPose(PxTransform(PxVec3(0,1.1,0)));

			armL = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armL)
				printf("creating snowman shape failed");
			armL->setLocalPose(PxTransform(PxVec3(-.4,.7,0)));

			armR = PxRigidActorExt::createExclusiveShape(*snowmanActor, PxCapsuleGeometry(.1,.1), *gMaterial);
			if(!armR)
				printf("creating snowman shape failed");			
			armR->setLocalPose(PxTransform(PxVec3( .4,.7,0)));

			snowmanActor->setMass(1);
			snowmanActor->setCMassLocalPose(PxTransform(PxVec3(0,-.5,0)));
			snowmanActor->setMassSpaceInertiaTensor(PxVec3(.05,100,100));
		}
		break;
	default:
		break;
	}
	
	gScene->addActor(*snowmanActor);

	return snowmanActor;
}

static void createSnowMen()
{
	PxU32 numSnowmen = 5;
	gSnowmanCount = 0;
	for(PxU32 i=0; i<numSnowmen; i++)
	{	
		PxVec3 pos(i * 2.5f,1,-8);
		gSnowmen[gSnowmanCount++] =
			createSnowMan(PxTransform(pos), i);
	}
	if(gHeadlessOptions.headless)
	{
		PxRigidDynamic* rotated = createSnowMan(
			PxTransform(PxVec3(12.5f, 1.0f, -8.0f)), 4);
		rotated->setCMassLocalPose(PxTransform(
			PxVec3(0.0f, -0.5f, 0.0f),
			PxQuat(0.45f, PxVec3(0.0f, 1.0f, 0.0f))));
		rotated->setMassSpaceInertiaTensor(
			PxVec3(0.05f, 0.2f, 0.8f));
		gSnowmen[gSnowmanCount++] = rotated;
	}
}

static bool initPhysicsInternal(bool interactive)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && transport)
			gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = gHeadlessOptions.headless ?
		PxVec3(0.0f) : PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless ?
		gHeadlessOptions.dispatcherThreads : 2u);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
		return false;

	if(interactive)
	{
		PxRigidStatic* groundPlane =
			PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
		gScene->addActor(*groundPlane);
	}

    createSnowMen();
	return gSnowmanCount == (gHeadlessOptions.headless ? 6u : 5u);
}

static bool stepPhysicsInternal(bool interactive)
{
	gScene->simulate(gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f);
	if(!gScene->fetchResults(true))
	{
		if(gHeadlessOptions.headless)
			gMetrics.fetchFailures++;
		return false;
	}
	PX_UNUSED(interactive);
	if(gHeadlessOptions.headless)
		gMetrics.completedFrames++;
	return true;
}

void initPhysics(bool interactive)
{
	initPhysicsInternal(interactive);
}

void stepPhysics(bool interactive)
{
	stepPhysicsInternal(interactive);
}
	
void cleanupPhysics(bool interactive)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gPhysics);

	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}

	PX_RELEASE(gFoundation);
	for(PxU32 i = 0; i < 6; ++i)
		gSnowmen[i] = NULL;
	gSnowmanCount = 0;
	gMetrics.cleanupComplete = 1;
	
	if(interactive)
		printf("SnippetMassProperties done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case ' ':	createDynamic(camera, PxSphereGeometry(0.1f), camera.rotate(PxVec3(0,0,-1))*20);	break;
	}
}

static bool prepareHeadlessImpulse()
{
	const PxVec3 impulse(3.0f, 0.0f, 0.0f);
	for(PxU32 i = 0; i < gSnowmanCount; ++i)
	{
		PxRigidDynamic* body = gSnowmen[i];
		if(!body)
			return false;
		body->setLinearDamping(0.0f);
		body->setAngularDamping(0.0f);
		body->setMaxLinearVelocity(1e6f);
		body->setMaxAngularVelocity(1e6f);
		body->setLinearVelocity(PxVec3(0.0f));
		body->setAngularVelocity(PxVec3(0.0f));
		const PxReal mass = body->getMass();
		const PxTransform bodyPose = body->getGlobalPose();
		const PxTransform cMassPose = body->getCMassLocalPose();
		const PxVec3 worldCom = bodyPose.transform(cMassPose.p);
		const PxVec3 applicationPoint = bodyPose.p;
		const PxVec3 angularImpulse =
			(applicationPoint - worldCom).cross(impulse);
		const PxQuat massOrientation =
			(bodyPose.q * cMassPose.q).getNormalized();
		const PxVec3 inertia = body->getMassSpaceInertiaTensor();
		if(!PxIsFinite(mass) || mass <= 0.0f ||
			!inertia.isFinite() || inertia.x <= 0.0f ||
			inertia.y <= 0.0f || inertia.z <= 0.0f)
			return false;
		const PxVec3 angularImpulseMass =
			massOrientation.rotateInv(angularImpulse);
		const PxVec3 angularVelocityMass(
			angularImpulseMass.x / inertia.x,
			angularImpulseMass.y / inertia.y,
			angularImpulseMass.z / inertia.z);
		MassResponse& response = gMetrics.responses[i];
		response.mass = mass;
		response.expectedLinearVelocity = impulse / mass;
		response.expectedAngularVelocity =
			massOrientation.rotate(angularVelocityMass);
		gMetrics.massMin = PxMin(gMetrics.massMin, mass);
		gMetrics.massMax = PxMax(gMetrics.massMax, mass);
		PxRigidBodyExt::addForceAtPos(
			*body, impulse, applicationPoint, PxForceMode::eIMPULSE);
	}
	return true;
}

static bool evaluateHeadlessResponse()
{
	bool finite = true;
	for(PxU32 i = 0; i < gSnowmanCount; ++i)
	{
		MassResponse& response = gMetrics.responses[i];
		response.observedLinearVelocity =
			gSnowmen[i]->getLinearVelocity();
		response.observedAngularVelocity =
			gSnowmen[i]->getAngularVelocity();
		if(!response.observedLinearVelocity.isFinite() ||
			!response.observedAngularVelocity.isFinite())
		{
			gMetrics.nonFinite++;
			finite = false;
			continue;
		}
		const PxReal expectedLinearMagnitudeSquared =
			response.expectedLinearVelocity.magnitudeSquared();
		const PxReal expectedAngularMagnitudeSquared =
			response.expectedAngularVelocity.magnitudeSquared();
		if(expectedLinearMagnitudeSquared <= 1e-12f ||
			expectedAngularMagnitudeSquared <= 1e-12f)
		{
			gMetrics.nonFinite++;
			finite = false;
			continue;
		}
		const PxReal linearScale =
			response.observedLinearVelocity.dot(
				response.expectedLinearVelocity) /
			expectedLinearMagnitudeSquared;
		const PxReal angularScale =
			response.observedAngularVelocity.dot(
				response.expectedAngularVelocity) /
			expectedAngularMagnitudeSquared;
		const PxReal linearResidual =
			(response.observedLinearVelocity -
				response.expectedLinearVelocity * linearScale).magnitude() /
			PxSqrt(expectedLinearMagnitudeSquared);
		const PxReal angularResidual =
			(response.observedAngularVelocity -
				response.expectedAngularVelocity * angularScale).magnitude() /
			PxSqrt(expectedAngularMagnitudeSquared);
		gMetrics.linearScaleMin =
			PxMin(gMetrics.linearScaleMin, linearScale);
		gMetrics.linearScaleMax =
			PxMax(gMetrics.linearScaleMax, linearScale);
		gMetrics.angularScaleMin =
			PxMin(gMetrics.angularScaleMin, angularScale);
		gMetrics.angularScaleMax =
			PxMax(gMetrics.angularScaleMax, angularScale);
		gMetrics.maxLinearResidual =
			PxMax(gMetrics.maxLinearResidual, linearResidual);
		gMetrics.maxAngularResidual =
			PxMax(gMetrics.maxAngularResidual, angularResidual);
	}
	if(gSnowmanCount == 6)
	{
		gMetrics.rotatedOmegaX =
			gMetrics.responses[5].observedAngularVelocity.x;
		gMetrics.rotatedOmegaZ =
			gMetrics.responses[5].observedAngularVelocity.z;
	}
	return finite;
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 1;
	defaults.caseName = "impulse-response";
	defaults.solverType = PxSolverType::eAVBD;
	defaults.dispatcherThreads = 4;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") + argv[i];
			return false;
		}
	if(gHeadlessOptions.headless &&
		!Snippets::equalsIgnoreCase(
			gHeadlessOptions.caseName.c_str(), "impulse-response"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.headless && gHeadlessOptions.frames != 1)
	{
		error = "impulse-response requires --frames=1";
		return false;
	}
	if(gHeadlessOptions.headless &&
		gHeadlessOptions.solverType != PxSolverType::eTGS &&
		gHeadlessOptions.solverType != PxSolverType::eAVBD)
	{
		error = "headless gate supports only tgs or avbd";
		return false;
	}
	return true;
}

static bool headlessPassed()
{
	return
		gMetrics.completedFrames == 1 &&
		gMetrics.fetchFailures == 0 &&
		gMetrics.nonFinite == 0 &&
		gMetrics.cleanupComplete == 1 &&
		gMetrics.massMin > 0.0f &&
		gMetrics.massMax / gMetrics.massMin > 2.0f &&
		gMetrics.linearScaleMin > 0.95f &&
		gMetrics.linearScaleMax < 1.01f &&
		gMetrics.linearScaleMax - gMetrics.linearScaleMin < 0.005f &&
		gMetrics.angularScaleMin > 0.95f &&
		gMetrics.angularScaleMax < 1.01f &&
		gMetrics.angularScaleMax - gMetrics.angularScaleMin < 0.01f &&
		gMetrics.maxLinearResidual < 1e-4f &&
		gMetrics.maxAngularResidual < 1e-3f &&
		PxAbs(gMetrics.rotatedOmegaX) > 0.1f &&
		PxAbs(gMetrics.rotatedOmegaZ) > 0.1f &&
		gErrorCallback.getFatalCount() == 0;
}

static const char* failureReason()
{
	if(gErrorCallback.getFatalCount())
		return "fatal_error";
	if(gMetrics.cleanupComplete != 1)
		return "cleanup_incomplete";
	if(gMetrics.fetchFailures)
		return "fetch_failed";
	if(gMetrics.nonFinite)
		return "non_finite";
	if(gMetrics.linearScaleMin <= 0.95f ||
		gMetrics.linearScaleMax >= 1.01f)
		return "mass_response_scale";
	if(gMetrics.angularScaleMin <= 0.95f ||
		gMetrics.angularScaleMax >= 1.01f)
		return "inertia_response_scale";
	if(gMetrics.maxLinearResidual >= 1e-4f)
		return "mass_response_direction";
	if(gMetrics.maxAngularResidual >= 1e-3f)
		return "com_inertia_response_direction";
	if(PxAbs(gMetrics.rotatedOmegaX) <= 0.1f ||
		PxAbs(gMetrics.rotatedOmegaZ) <= 0.1f)
		return "rotated_inertia_not_consumed";
	return "response_mismatch";
}

static void printHeadlessResult()
{
	const bool pass = headlessPassed();
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetMassProperties solver=%s "
		"case=%s execution=%s frames=1 completedFrames=%u bodies=6 "
		"massMin=%.9g massMax=%.9g linearScaleMin=%.9g "
		"linearScaleMax=%.9g angularScaleMin=%.9g angularScaleMax=%.9g "
		"maxLinearResidual=%.9g maxAngularResidual=%.9g "
		"rotatedOmegaX=%.9g rotatedOmegaZ=%.9g "
		"speed0=%.9g speed1=%.9g speed2=%.9g speed3=%.9g "
		"speed4=%.9g speed5=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0 status=%s reason=%s "
		"validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gMetrics.completedFrames, double(gMetrics.massMin),
		double(gMetrics.massMax), double(gMetrics.linearScaleMin),
		double(gMetrics.linearScaleMax), double(gMetrics.angularScaleMin),
		double(gMetrics.angularScaleMax),
		double(gMetrics.maxLinearResidual),
		double(gMetrics.maxAngularResidual),
		double(gMetrics.rotatedOmegaX), double(gMetrics.rotatedOmegaZ),
		double(gMetrics.responses[0].observedLinearVelocity.magnitude()),
		double(gMetrics.responses[1].observedLinearVelocity.magnitude()),
		double(gMetrics.responses[2].observedLinearVelocity.magnitude()),
		double(gMetrics.responses[3].observedLinearVelocity.magnitude()),
		double(gMetrics.responses[4].observedLinearVelocity.magnitude()),
		double(gMetrics.responses[5].observedLinearVelocity.magnitude()),
		gMetrics.nonFinite, gMetrics.fetchFailures,
		gErrorCallback.getFatalCount(), gMetrics.cleanupComplete,
		pass ? "PASS" : "FAIL", pass ? "none" : failureReason());
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] %s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
	{
		if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		{
			std::fprintf(stderr,
				"[AVBD_GATE_CONFIG_ERROR] failed to set execution mode\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		Snippets::printHeadlessConfig(
			"SnippetMassProperties", gHeadlessOptions);
		gErrorCallback.reset();
		bool runOk = initPhysicsInternal(false);
		runOk = runOk && prepareHeadlessImpulse();
		runOk = runOk && stepPhysicsInternal(false);
		runOk = runOk && evaluateHeadlessResponse();
		cleanupPhysics(false);
		printHeadlessResult();
		return runOk && headlessPassed() ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}

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
