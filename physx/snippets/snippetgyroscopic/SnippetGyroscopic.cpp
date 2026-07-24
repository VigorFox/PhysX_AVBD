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
// This snippet illustrates how to enable gyroscopic forces. The behavior of
// the object is known as the Dzhanibekov effect.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#ifdef RENDER_SNIPPET
	#include "../snippetrender/SnippetRender.h"
#endif
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
static PxPvd*					gPvd        = NULL;
static PxRigidDynamic*			gActor		= NULL;
static bool						gExtensionsInitialized = false;
#if PX_SUPPORT_GPU_PHYSX
static PxCudaContextManager*	gCudaContextManager	= NULL;
#endif

static bool			gPause		= false;
static bool			gOneFrame	= false;
static bool			gGyro		= true;
static Snippets::HeadlessOptions gHeadlessOptions;

struct GyroscopicMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 sleepingFrames;
	PxReal initialEnergy;
	PxReal finalEnergy;
	PxReal maxEnergyDrift;
	PxReal initialMomentumMagnitude;
	PxReal finalMomentumMagnitude;
	PxReal maxMomentumVectorDrift;
	PxQuat initialOrientation;
	PxQuat finalOrientation;
	PxVec3 initialAngularVelocity;
	PxVec3 finalAngularVelocity;
	PxVec3 initialMomentum;
	PxVec3 finalMomentum;
	PxU32 cleanupComplete;

	GyroscopicMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0), sleepingFrames(0),
	  initialEnergy(0.0f), finalEnergy(0.0f), maxEnergyDrift(0.0f),
	  initialMomentumMagnitude(0.0f), finalMomentumMagnitude(0.0f),
	  maxMomentumVectorDrift(0.0f), initialOrientation(PxIdentity),
	  finalOrientation(PxIdentity), initialAngularVelocity(0.0f),
	  finalAngularVelocity(0.0f), initialMomentum(0.0f),
	  finalMomentum(0.0f), cleanupComplete(0)
	{
	}
};

static GyroscopicMetrics gMetrics;

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), name);
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 600;
	defaults.caseName = "gyro-on";
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
	if(!isHeadlessCase("gyro-off") && !isHeadlessCase("gyro-on"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames < 300)
	{
		error = "--frames must be at least 300";
		return false;
	}
	return true;
}

static PxVec3 computeWorldMomentum(
	const PxRigidDynamic& actor, const PxQuat& orientation,
	const PxVec3& angularVelocity)
{
	const PxVec3 bodyAngularVelocity =
		orientation.rotateInv(angularVelocity);
	const PxVec3 bodyMomentum =
		actor.getMassSpaceInertiaTensor().multiply(bodyAngularVelocity);
	return orientation.rotate(bodyMomentum);
}

static PxReal computeRotationalEnergy(
	const PxVec3& angularVelocity, const PxVec3& worldMomentum)
{
	return 0.5f * angularVelocity.dot(worldMomentum);
}
#if PX_SUPPORT_GPU_PHYSX
static bool			gGpu		= false;
#endif

static void initScene()
{
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.solverType = gHeadlessOptions.solverType;

#if PX_SUPPORT_GPU_PHYSX
	if(gGpu)
	{
		sceneDesc.cudaContextManager = gCudaContextManager;
		sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
		sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
	}
#endif

	gScene = gPhysics->createScene(sceneDesc);

	const PxTransform pose(PxVec3(0.0f, 1.0f, 0.0f));

	PxRigidDynamic* actor = gPhysics->createRigidDynamic(pose);
	gActor = actor;

	PxShape* shape0 = gPhysics->createShape(PxBoxGeometry(PxVec3(0.05f, 0.5f, 0.05f)), *gMaterial, true);
	actor->attachShape(*shape0);
	shape0->release();

	PxShape* shape1 = gPhysics->createShape(PxBoxGeometry(PxVec3(0.1f, 0.05f, 0.05f)), *gMaterial, true);
	shape1->setLocalPose(PxTransform(PxVec3(0.1f, 0.0f, 0.0f)));
	actor->attachShape(*shape1);
	shape1->release();

	PxRigidBodyExt::updateMassAndInertia(*actor, 1.0f);

	actor->setAngularVelocity(PxVec3(30.f*0.25f, 20.1f*0.25f, 0.0f));
	actor->setAngularDamping(0.0f);
	if(gHeadlessOptions.headless)
	{
		actor->setSleepThreshold(0.0f);
		actor->setStabilizationThreshold(0.0f);
	}

	actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);

	if(gGyro)
		actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);

	gScene->addActor(*actor);

	if(gHeadlessOptions.headless)
	{
		gMetrics.initialOrientation = actor->getGlobalPose().q;
		gMetrics.finalOrientation = gMetrics.initialOrientation;
		gMetrics.initialAngularVelocity = actor->getAngularVelocity();
		gMetrics.finalAngularVelocity = gMetrics.initialAngularVelocity;
		gMetrics.initialMomentum = computeWorldMomentum(
			*actor, gMetrics.initialOrientation,
			gMetrics.initialAngularVelocity);
		gMetrics.finalMomentum = gMetrics.initialMomentum;
		gMetrics.initialMomentumMagnitude =
			gMetrics.initialMomentum.magnitude();
		gMetrics.finalMomentumMagnitude =
			gMetrics.initialMomentumMagnitude;
		gMetrics.initialEnergy = computeRotationalEnergy(
			gMetrics.initialAngularVelocity, gMetrics.initialMomentum);
		gMetrics.finalEnergy = gMetrics.initialEnergy;
	}
	else
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
}

void renderText()
{
#ifdef RENDER_SNIPPET
	Snippets::print("Press F1 to toggle gyroscopic forces.");
	#if PX_SUPPORT_GPU_PHYSX
	Snippets::print("Press F2 to toggle GPU simulation.");
	#endif
	if(gGyro)
		Snippets::print("Gyroscopic forces: ON");
	else
		Snippets::print("Gyroscopic forces: OFF");
	#if PX_SUPPORT_GPU_PHYSX
	if(gGpu)
		Snippets::print("GPU: ON");
	else
		Snippets::print("GPU: OFF");
	#endif
#endif
}

void initPhysics(bool /*interactive*/)
{
	if(!gHeadlessOptions.headless)
	{
		printf("Gyroscopic snippet. Use these keys:\n");
		printf(" P  - enable/disable pause\n");
		printf(" O  - step simulation for one frame\n");
		printf(" R  - reset scene\n");
		printf(" F1 - enable/disable gyroscopic forces\n");
#if PX_SUPPORT_GPU_PHYSX
		printf(" F2 - enable/disable GPU simulation\n");
#endif
		printf("\n");
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
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);

#if PX_SUPPORT_GPU_PHYSX
	PxCudaContextManagerDesc cudaContextManagerDesc;
	gCudaContextManager = PxCreateCudaContextManager(*gFoundation, cudaContextManagerDesc, PxGetProfilerCallback());
	if( gCudaContextManager )
	{
		if( !gCudaContextManager->contextIsValid() )
			PX_RELEASE(gCudaContextManager);
	}	
#endif

	if(gHeadlessOptions.headless)
		gDispatcher = PxDefaultCpuDispatcherCreate(
			gHeadlessOptions.dispatcherThreads);
	else
	{
		const PxU32 numCores = SnippetUtils::getNbPhysicalCores();
		gDispatcher = PxDefaultCpuDispatcherCreate(
			numCores == 0 ? 0 : numCores - 1);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.25f);

	gGyro = !gHeadlessOptions.headless || isHeadlessCase("gyro-on");
	initScene();
}

void stepPhysics(bool /*interactive*/)
{
	if (gPause && !gOneFrame)
		return;
	gOneFrame = false;

	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f / 60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gActor)
		{
			if(gActor->isSleeping())
				++gMetrics.sleepingFrames;
			const PxTransform pose = gActor->getGlobalPose();
			const PxVec3 angularVelocity = gActor->getAngularVelocity();
			const PxVec3 momentum = computeWorldMomentum(
				*gActor, pose.q, angularVelocity);
			const PxReal energy =
				computeRotationalEnergy(angularVelocity, momentum);
			gMetrics.finalOrientation = pose.q;
			gMetrics.finalAngularVelocity = angularVelocity;
			gMetrics.finalMomentum = momentum;
			gMetrics.finalMomentumMagnitude = momentum.magnitude();
			gMetrics.finalEnergy = energy;
			const PxReal energyDenominator =
				PxMax(PxAbs(gMetrics.initialEnergy), 1e-6f);
			const PxReal momentumDenominator =
				PxMax(gMetrics.initialMomentumMagnitude, 1e-6f);
			gMetrics.maxEnergyDrift = PxMax(
				gMetrics.maxEnergyDrift,
				PxAbs(energy - gMetrics.initialEnergy) /
					energyDenominator);
			gMetrics.maxMomentumVectorDrift = PxMax(
				gMetrics.maxMomentumVectorDrift,
				(momentum - gMetrics.initialMomentum).magnitude() /
					momentumDenominator);
			if(!pose.isFinite() || !angularVelocity.isFinite() ||
				!momentum.isFinite() || !PxIsFinite(energy))
				++gMetrics.nonFinite;
		}
		++gMetrics.completedFrames;
	}
}
	
static void releaseScene()
{
	PX_RELEASE(gScene);
	gActor = NULL;
}

void cleanupPhysics(bool /*interactive*/)
{
	releaseScene();
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
#if PX_SUPPORT_GPU_PHYSX
	PX_RELEASE(gCudaContextManager);
#endif
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gActor && !gMaterial && !gDispatcher && !gPhysics &&
		!gPvd && !gFoundation && !gExtensionsInitialized ? 1u : 0u;
	
	printf("SnippetGyroscopic done.\n");
}

void keyPress(unsigned char key, const PxTransform& /*camera*/)
{
	if(key == 'p' || key == 'P')
		gPause = !gPause;

	if(key == 'o' || key == 'O')
	{
		gPause = true;
		gOneFrame = true;
	}

	if(gScene)
	{
		if(key == 1)
		{
			gGyro = !gGyro;
			releaseScene();
			initScene();
		}

#if PX_SUPPORT_GPU_PHYSX
		if(key == 2)
		{
			gGpu = !gGpu;
			releaseScene();
			initScene();
		}
#endif

		if(key == 'r' || key == 'R')
		{
			releaseScene();
			initScene();
		}
	}
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig("SnippetGyroscopic", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gActor;
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
		gMetrics.sleepingFrames != 0 ||
		gErrorCallback.getFatalCount() != 0 ||
		!gMetrics.finalOrientation.isUnit())
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.initialEnergy <= 0.0f ||
		gMetrics.initialMomentumMagnitude <= 0.0f ||
		gMetrics.finalAngularVelocity.magnitude() <= 1e-3f)
	{
		passed = false;
		reason = "missing_rotational_state";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetGyroscopic solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED gyroEnabled=%u "
		"initialQx=%.9g initialQy=%.9g initialQz=%.9g initialQw=%.9g "
		"finalQx=%.9g finalQy=%.9g finalQz=%.9g finalQw=%.9g "
		"initialWx=%.9g initialWy=%.9g initialWz=%.9g "
		"finalWx=%.9g finalWy=%.9g finalWz=%.9g "
		"initialLx=%.9g initialLy=%.9g initialLz=%.9g "
		"finalLx=%.9g finalLy=%.9g finalLz=%.9g "
		"initialEnergy=%.9g finalEnergy=%.9g maxEnergyDrift=%.9g "
		"initialMomentumMagnitude=%.9g finalMomentumMagnitude=%.9g "
		"maxMomentumVectorDrift=%.9g sleepingFrames=%u nonFinite=%u "
		"fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, gGyro ? 1u : 0u,
		double(gMetrics.initialOrientation.x),
		double(gMetrics.initialOrientation.y),
		double(gMetrics.initialOrientation.z),
		double(gMetrics.initialOrientation.w),
		double(gMetrics.finalOrientation.x),
		double(gMetrics.finalOrientation.y),
		double(gMetrics.finalOrientation.z),
		double(gMetrics.finalOrientation.w),
		double(gMetrics.initialAngularVelocity.x),
		double(gMetrics.initialAngularVelocity.y),
		double(gMetrics.initialAngularVelocity.z),
		double(gMetrics.finalAngularVelocity.x),
		double(gMetrics.finalAngularVelocity.y),
		double(gMetrics.finalAngularVelocity.z),
		double(gMetrics.initialMomentum.x),
		double(gMetrics.initialMomentum.y),
		double(gMetrics.initialMomentum.z),
		double(gMetrics.finalMomentum.x),
		double(gMetrics.finalMomentum.y),
		double(gMetrics.finalMomentum.z),
		double(gMetrics.initialEnergy), double(gMetrics.finalEnergy),
		double(gMetrics.maxEnergyDrift),
		double(gMetrics.initialMomentumMagnitude),
		double(gMetrics.finalMomentumMagnitude),
		double(gMetrics.maxMomentumVectorDrift), gMetrics.sleepingFrames,
		gMetrics.nonFinite,
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
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetGyroscopic "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetGyroscopic "
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
