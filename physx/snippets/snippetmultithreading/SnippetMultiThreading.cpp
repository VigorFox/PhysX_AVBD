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
// This snippet shows how to coordinate threads performing asynchronous
// work during the scene simulation. After simulate() is called, user threads 
// are started that perform ray-casts against the scene. The call to 
// fetchResults() is delayed until all ray-casts have completed.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd = NULL;

struct RaycastThread
{
	SnippetUtils::Sync*		mWorkReadySyncHandle;
	SnippetUtils::Thread*	mThreadHandle;
};
const PxU32				 gMaxQueryThreads = 8;
RaycastThread			 gThreads[gMaxQueryThreads];
PxU32					 gNumQueryThreads = 1;

SnippetUtils::Sync*		 gWorkDoneSyncHandle;

const PxI32				 gRayCount = 1024;
volatile PxI32			 gRaysAvailable;
volatile PxI32			 gRaysCompleted;
volatile PxI32			 gRayHits;
PxU32					 gBatchIndex;
Snippets::HeadlessOptions gHeadlessOptions;
PxU32					 gLifecycleCycles = 1;

struct MultiThreadingMetrics
{
	PxU32 completedFrames;
	PxU32 completedCycles;
	PxU32 raycastBatches;
	PxU64 raysCompleted;
	PxU64 rayHits;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 createdDynamicBodies;
	PxU32 cleanupComplete;
	PxReal minBodyY;
	PxReal maxBodyY;
	PxReal maxBodySpeed;

	MultiThreadingMetrics()
	: completedFrames(0), completedCycles(0), raycastBatches(0),
	  raysCompleted(0), rayHits(0), fetchFailures(0), nonFinite(0),
	  createdDynamicBodies(0), cleanupComplete(0), minBodyY(FLT_MAX),
	  maxBodyY(-FLT_MAX), maxBodySpeed(0.0f)
	{
	}
};

static MultiThreadingMetrics gMetrics;

static PxU32 hashU32(PxU32 value)
{
	value ^= value >> 16;
	value *= 0x7feb352du;
	value ^= value >> 15;
	value *= 0x846ca68bu;
	return value ^ (value >> 16);
}

static PxReal hashSignedUnit(PxU32 value)
{
	return (PxReal(hashU32(value) & 0x00ffffffu) /
		PxReal(0x007fffffu)) - 1.0f;
}

static PxVec3 rayDirection(PxU32 rayIndex)
{
	const PxU32 key = rayIndex + gBatchIndex * 4099u +
		gHeadlessOptions.seed * 131071u;
	PxVec3 direction(
		hashSignedUnit(key),
		hashSignedUnit(key + 0x9e3779b9u),
		hashSignedUnit(key + 0x3c6ef372u));
	if(direction.magnitudeSquared() < 1e-8f)
		direction = PxVec3(0.0f, 1.0f, 0.0f);
	return direction.getNormalized();
}

static void threadExecute(void* data)
{
	RaycastThread* raycastThread = static_cast<RaycastThread*>(data);

	// Perform random raycasts against the scene until stop.
	for(;;)
	{
		// Wait here for the sync to be set then reset the sync
		// to ensure that we only perform raycast work after the 
		// sync has been set again.
		SnippetUtils::syncWait(raycastThread->mWorkReadySyncHandle);
		SnippetUtils::syncReset(raycastThread->mWorkReadySyncHandle);

		// If the thread has been signaled to quit then exit this function.
		if (SnippetUtils::threadQuitIsSignalled(raycastThread->mThreadHandle))
			break;

		// Perform a fixed number of random raycasts against the scene
		// and share the work between multiple threads.
		PxI32 rayIndex;
		while ((rayIndex =
			SnippetUtils::atomicDecrement(&gRaysAvailable)) >= 0)
		{
			const PxVec3 dir = rayDirection(PxU32(rayIndex));

			PxRaycastBuffer buf;
			if(gScene->raycast(
				PxVec3(0.0f), dir, 1000.0f, buf, PxHitFlag::eDEFAULT))
				SnippetUtils::atomicIncrement(&gRayHits);

			// If this is the last raycast then signal this to the main thread.
			if (SnippetUtils::atomicIncrement(&gRaysCompleted) == gRayCount)
			{
				SnippetUtils::syncSet(gWorkDoneSyncHandle);
			}
		}
	}

	// Quit the current thread.
	SnippetUtils::threadQuit(raycastThread->mThreadHandle);
}

void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for(PxU32 i=0; i<size;i++)
	{
		for(PxU32 j=0;j<size-i;j++)
		{
			PxTransform localTm(PxVec3(PxReal(j*2) - PxReal(size-i), PxReal(i*2+1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			gScene->addActor(*body);
			gMetrics.createdDynamicBodies++;
		}
	}
	shape->release();
}

bool createPhysicsAndScene()
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	
	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && transport)
			gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	
	const PxU32 physicalCores = SnippetUtils::getNbPhysicalCores();
	PxU32 dispatcherThreads = gHeadlessOptions.headless ?
		gHeadlessOptions.dispatcherThreads :
		(physicalCores > 1 ? physicalCores - 1 : 1);
	gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	
	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	for(PxU32 i=0;i<5;i++)
		createStack(PxTransform(PxVec3(0,0,i*10.0f)), 10, 2.0f);
	return true;
}

bool createRaycastThreads()
{
	// Create and start threads that will perform raycasts.
	// Create a sync for each thread so that a signal may be sent
	// from the main thread to the raycast thread that it can start 
	// performing raycasts.
	gWorkDoneSyncHandle = SnippetUtils::syncCreate();
	if(!gWorkDoneSyncHandle)
		return false;

	for (PxU32 i=0; i < gNumQueryThreads; ++i)
	{
		//Create a sync.
		gThreads[i].mWorkReadySyncHandle = SnippetUtils::syncCreate();
		if(!gThreads[i].mWorkReadySyncHandle)
			return false;

		//Create and start a thread.
		gThreads[i].mThreadHandle =  SnippetUtils::threadCreate(threadExecute, &gThreads[i]);
		if(!gThreads[i].mThreadHandle)
			return false;
	}
	return true;
}

bool initPhysics()
{
	return createPhysicsAndScene() && createRaycastThreads();
}

bool sampleActors()
{
	const PxU32 actorCount =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	std::vector<PxActor*> actors(actorCount);
	if(actorCount)
		gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
			actors.data(), actorCount);
	bool finite = true;
	for(PxU32 i = 0; i < actorCount; ++i)
	{
		PxRigidDynamic* body = actors[i]->is<PxRigidDynamic>();
		if(!body)
			continue;
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 linearVelocity = body->getLinearVelocity();
		const PxVec3 angularVelocity = body->getAngularVelocity();
		if(!pose.isValid() || !linearVelocity.isFinite() ||
			!angularVelocity.isFinite())
		{
			gMetrics.nonFinite++;
			finite = false;
			continue;
		}
		gMetrics.minBodyY = PxMin(gMetrics.minBodyY, pose.p.y);
		gMetrics.maxBodyY = PxMax(gMetrics.maxBodyY, pose.p.y);
		gMetrics.maxBodySpeed =
			PxMax(gMetrics.maxBodySpeed, linearVelocity.magnitude());
	}
	return finite;
}

bool stepPhysics()
{
	// Start simulation
	gScene->simulate(gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f);

	// Start ray-cast threads
	gRaysAvailable = gRayCount;
	gRaysCompleted = 0;
	gRayHits = 0;

	// Signal to each raycast thread that they can start performing raycasts.
	for (PxU32 i=0; i < gNumQueryThreads; ++i)
	{
		SnippetUtils::syncSet(gThreads[i].mWorkReadySyncHandle);
	}

	// Wait for raycast threads to finish.
	SnippetUtils::syncWait(gWorkDoneSyncHandle);
	SnippetUtils::syncReset(gWorkDoneSyncHandle);
	gMetrics.raycastBatches++;
	gMetrics.raysCompleted += PxU64(gRaysCompleted);
	gMetrics.rayHits += PxU64(gRayHits);

	// Fetch simulation results
	if(!gScene->fetchResults(true))
	{
		gMetrics.fetchFailures++;
		return false;
	}
	gMetrics.completedFrames++;
	gBatchIndex++;
	return sampleActors();
}
	
void cleanupPhysics()
{
	// Signal threads to quit.
	for (PxU32 i=0; i < gNumQueryThreads; ++i)
	{
		if(gThreads[i].mThreadHandle)
			SnippetUtils::threadSignalQuit(gThreads[i].mThreadHandle);
		if(gThreads[i].mWorkReadySyncHandle)
			SnippetUtils::syncSet(gThreads[i].mWorkReadySyncHandle);
	}

	// Clean up raycast threads and syncs.
	for (PxU32 i=0; i < gNumQueryThreads; ++i)
	{
		if(gThreads[i].mThreadHandle)
		{
			SnippetUtils::threadWaitForQuit(gThreads[i].mThreadHandle);
			SnippetUtils::threadRelease(gThreads[i].mThreadHandle);
		}
		if(gThreads[i].mWorkReadySyncHandle)
			SnippetUtils::syncRelease(gThreads[i].mWorkReadySyncHandle);
		gThreads[i].mThreadHandle = NULL;
		gThreads[i].mWorkReadySyncHandle = NULL;
	}

	// Clean up the sync for the main thread.
	if(gWorkDoneSyncHandle)
		SnippetUtils::syncRelease(gWorkDoneSyncHandle);
	gWorkDoneSyncHandle = NULL;

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
	gMetrics.cleanupComplete++;
	
	if(!gHeadlessOptions.headless)
		printf("SnippetMultiThreading done.\n");
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 120;
	defaults.caseName = "concurrent-query";
	defaults.solverType = PxSolverType::eAVBD;
	defaults.dispatcherThreads = 4;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;

	gLifecycleCycles = gHeadlessOptions.headless ? 2u : 1u;
	gNumQueryThreads = gHeadlessOptions.headless ? 4u : 1u;
	bool cyclesSeen = false;
	bool queryThreadsSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		if(Snippets::hasOptionPrefix(arg, "--cycles="))
		{
			if(cyclesSeen || !Snippets::parseU32(
				arg + std::strlen("--cycles="), 1, 16, gLifecycleCycles))
			{
				error = "invalid or duplicate --cycles";
				return false;
			}
			cyclesSeen = true;
		}
		else if(Snippets::hasOptionPrefix(arg, "--query-threads="))
		{
			if(queryThreadsSeen || !Snippets::parseU32(
				arg + std::strlen("--query-threads="), 1,
				gMaxQueryThreads, gNumQueryThreads))
			{
				error = "invalid or duplicate --query-threads";
				return false;
			}
			queryThreadsSeen = true;
		}
		else
		{
			error = std::string("unknown option: ") + arg;
			return false;
		}
	}
	if(gHeadlessOptions.headless &&
		!Snippets::equalsIgnoreCase(
			gHeadlessOptions.caseName.c_str(), "concurrent-query"))
	{
		error = "unsupported --case";
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

static void printHeadlessResult()
{
	const PxU32 expectedFrames =
		gHeadlessOptions.frames * gLifecycleCycles;
	const PxU64 expectedRays =
		PxU64(expectedFrames) * PxU64(gRayCount);
	const PxU32 fatalErrors = gErrorCallback.getFatalCount();
	const bool pass =
		gMetrics.completedFrames == expectedFrames &&
		gMetrics.completedCycles == gLifecycleCycles &&
		gMetrics.raycastBatches == expectedFrames &&
		gMetrics.raysCompleted == expectedRays &&
		gMetrics.rayHits > 0 &&
		gMetrics.fetchFailures == 0 &&
		gMetrics.nonFinite == 0 &&
		gMetrics.createdDynamicBodies == 275u * gLifecycleCycles &&
		gMetrics.cleanupComplete == gLifecycleCycles &&
		fatalErrors == 0;
	const char* reason = "none";
	if(!pass)
	{
		if(fatalErrors)
			reason = "fatal_error";
		else if(gMetrics.cleanupComplete != gLifecycleCycles)
			reason = "cleanup_incomplete";
		else if(gMetrics.fetchFailures)
			reason = "fetch_failed";
		else if(gMetrics.nonFinite)
			reason = "non_finite";
		else if(gMetrics.raysCompleted != expectedRays)
			reason = "query_work_incomplete";
		else if(!gMetrics.rayHits)
			reason = "query_no_hits";
		else
			reason = "lifecycle_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetMultiThreading solver=%s "
		"case=%s execution=%s frames=%u cycles=%u queryThreads=%u "
		"dispatcherThreads=%u completedFrames=%u completedCycles=%u "
		"raycastBatches=%u raysExpected=%llu raysCompleted=%llu "
		"rayHits=%llu dynamicBodies=%u minBodyY=%.9g maxBodyY=%.9g "
		"maxBodySpeed=%.9g nonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gLifecycleCycles, gNumQueryThreads,
		gHeadlessOptions.dispatcherThreads, gMetrics.completedFrames,
		gMetrics.completedCycles, gMetrics.raycastBatches,
		static_cast<unsigned long long>(expectedRays),
		static_cast<unsigned long long>(gMetrics.raysCompleted),
		static_cast<unsigned long long>(gMetrics.rayHits),
		gMetrics.createdDynamicBodies, double(gMetrics.minBodyY),
		double(gMetrics.maxBodyY), double(gMetrics.maxBodySpeed),
		gMetrics.nonFinite, gMetrics.fetchFailures, fatalErrors,
		gMetrics.cleanupComplete, pass ? "PASS" : "FAIL", reason);
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
			"SnippetMultiThreading", gHeadlessOptions);
		gErrorCallback.reset();
	}

	const PxU32 frames =
		gHeadlessOptions.headless ? gHeadlessOptions.frames : 100u;
	bool runOk = true;
	for(PxU32 cycle = 0; cycle < gLifecycleCycles; ++cycle)
	{
		if(!initPhysics())
		{
			runOk = false;
			cleanupPhysics();
			break;
		}
		for(PxU32 frame = 0; frame < frames; ++frame)
		{
			if(!stepPhysics())
			{
				runOk = false;
				break;
			}
		}
		if(runOk)
			gMetrics.completedCycles++;
		cleanupPhysics();
		if(!runOk)
			break;
	}

	if(gHeadlessOptions.headless)
	{
		printHeadlessResult();
		const PxU32 expectedFrames = frames * gLifecycleCycles;
		const PxU64 expectedRays =
			PxU64(expectedFrames) * PxU64(gRayCount);
		return runOk &&
			gMetrics.completedFrames == expectedFrames &&
			gMetrics.raysCompleted == expectedRays &&
			gMetrics.rayHits > 0 &&
			gMetrics.nonFinite == 0 &&
			gMetrics.fetchFailures == 0 &&
			gMetrics.cleanupComplete == gLifecycleCycles &&
			gErrorCallback.getFatalCount() == 0
			? Snippets::eHEADLESS_PASS
			: Snippets::eHEADLESS_GATE_FAILED;
	}

	return 0;
}
