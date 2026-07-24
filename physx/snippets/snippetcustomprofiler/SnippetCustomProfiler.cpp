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
// This snippet illustrates how to set up a custom profiler, and potentially
// reroute it to PVD's profiling functions.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

#include <atomic>
#include <cstdio>
#include <string>

using namespace physx;

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static PxPvd* gPvd = NULL;
static PxRigidDynamic* gProjectile = NULL;

static PxReal stackZ = 10.0f;
static Snippets::HeadlessOptions gHeadlessOptions;

struct CustomProfilerMetrics
{
	PxU32 completedFrames;
	PxU32 profiledFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 initialStaticActors;
	PxU32 initialDynamicActors;
	PxU32 callbackInstalled;
	PxU32 callbackCleared;
	PxU32 cleanupComplete;
	PxReal maxBodySpeed;
	PxReal minBodyY;
	PxVec3 initialProjectilePosition;
	PxVec3 finalProjectilePosition;
	PxVec3 finalProjectileVelocity;

	CustomProfilerMetrics()
	: completedFrames(0), profiledFrames(0), fetchFailures(0),
	  nonFinite(0), initialStaticActors(0), initialDynamicActors(0),
	  callbackInstalled(0), callbackCleared(0), cleanupComplete(0),
	  maxBodySpeed(0.0f), minBodyY(PX_MAX_F32),
	  initialProjectilePosition(0.0f), finalProjectilePosition(0.0f),
	  finalProjectileVelocity(0.0f)
	{
	}
};

static CustomProfilerMetrics gMetrics;
static const bool gCallPVDProfilingFunctions = false;

class CustomProfilerCallback : public PxProfilerCallback
{
public:
	CustomProfilerCallback()
	: mZoneStarts(0), mZoneEnds(0), mDetachedStarts(0),
	  mDetachedEnds(0), mIntegerData(0), mFloatData(0), mFrames(0),
	  mInvalidNames(0)
	{
	}

	virtual ~CustomProfilerCallback() {}

	virtual void* zoneStart(
		const char* eventName, bool detached, uint64_t contextId)
	{
		mZoneStarts.fetch_add(1, std::memory_order_relaxed);
		if(detached)
			mDetachedStarts.fetch_add(1, std::memory_order_relaxed);
		if(!eventName || !eventName[0])
			mInvalidNames.fetch_add(1, std::memory_order_relaxed);

		if(!gHeadlessOptions.headless)
			std::printf("start: %s\n", eventName ? eventName : "<null>");

		return gCallPVDProfilingFunctions
			? gPvd->zoneStart(eventName, detached, contextId) : NULL;
	}

	virtual void zoneEnd(void* profilerData, const char* eventName,
		bool detached, uint64_t contextId)
	{
		if(gCallPVDProfilingFunctions)
			gPvd->zoneEnd(
				profilerData, eventName, detached, contextId);

		mZoneEnds.fetch_add(1, std::memory_order_relaxed);
		if(detached)
			mDetachedEnds.fetch_add(1, std::memory_order_relaxed);
		if(!eventName || !eventName[0])
			mInvalidNames.fetch_add(1, std::memory_order_relaxed);

		if(!gHeadlessOptions.headless)
			std::printf("end: %s\n", eventName ? eventName : "<null>");
	}

	virtual void recordData(
		int32_t value, const char* valueName, uint64_t contextId)
	{
		PX_UNUSED(value);
		mIntegerData.fetch_add(1, std::memory_order_relaxed);
		if(!valueName || !valueName[0])
			mInvalidNames.fetch_add(1, std::memory_order_relaxed);
		if(!gHeadlessOptions.headless)
		{
			std::printf(
				"data: %s (context ID %llu) = %d\n",
				valueName ? valueName : "<null>",
				static_cast<unsigned long long>(contextId), value);
		}
	}

	virtual void recordData(
		float value, const char* valueName, uint64_t contextId)
	{
		mFloatData.fetch_add(1, std::memory_order_relaxed);
		if(!valueName || !valueName[0])
			mInvalidNames.fetch_add(1, std::memory_order_relaxed);
		if(!gHeadlessOptions.headless)
		{
			std::printf(
				"data: %s (context ID %llu) = %f\n",
				valueName ? valueName : "<null>",
				static_cast<unsigned long long>(contextId),
				static_cast<double>(value));
		}
	}

	virtual void recordFrame(const char* name, uint64_t contextId)
	{
		mFrames.fetch_add(1, std::memory_order_relaxed);
		if(!name || !name[0])
			mInvalidNames.fetch_add(1, std::memory_order_relaxed);
		if(!gHeadlessOptions.headless)
		{
			std::printf(
				"frame: %s (context ID %llu)\n",
				name ? name : "<null>",
				static_cast<unsigned long long>(contextId));
		}
	}

	uint64_t zoneStarts() const
	{
		return mZoneStarts.load(std::memory_order_relaxed);
	}

	uint64_t zoneEnds() const
	{
		return mZoneEnds.load(std::memory_order_relaxed);
	}

	uint64_t detachedStarts() const
	{
		return mDetachedStarts.load(std::memory_order_relaxed);
	}

	uint64_t detachedEnds() const
	{
		return mDetachedEnds.load(std::memory_order_relaxed);
	}

	uint64_t integerData() const
	{
		return mIntegerData.load(std::memory_order_relaxed);
	}

	uint64_t floatData() const
	{
		return mFloatData.load(std::memory_order_relaxed);
	}

	uint64_t frames() const
	{
		return mFrames.load(std::memory_order_relaxed);
	}

	uint64_t invalidNames() const
	{
		return mInvalidNames.load(std::memory_order_relaxed);
	}

private:
	std::atomic<uint64_t> mZoneStarts;
	std::atomic<uint64_t> mZoneEnds;
	std::atomic<uint64_t> mDetachedStarts;
	std::atomic<uint64_t> mDetachedEnds;
	std::atomic<uint64_t> mIntegerData;
	std::atomic<uint64_t> mFloatData;
	std::atomic<uint64_t> mFrames;
	std::atomic<uint64_t> mInvalidNames;
};

static CustomProfilerCallback gCustomProfilerCallback;

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), name);
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 120;
	defaults.caseName = "stack-workload";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
	{
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") +
				(argv[i] ? argv[i] : "<null>");
			return false;
		}
	}
	if(!isHeadlessCase("stack-workload"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames < 60)
	{
		error = "--frames must be at least 60";
		return false;
	}
	return true;
}

static PxRigidDynamic* createDynamic(
	const PxTransform& transform, const PxGeometry& geometry,
	const PxVec3& velocity = PxVec3(0))
{
	PxRigidDynamic* dynamic = PxCreateDynamic(
		*gPhysics, transform, geometry, *gMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void createStack(
	const PxTransform& transform, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(
		PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for(PxU32 i = 0; i < size; ++i)
	{
		for(PxU32 j = 0; j < size - i; ++j)
		{
			const PxTransform localTransform(
				PxVec3(
					PxReal(j * 2) - PxReal(size - i),
					PxReal(i * 2 + 1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(
				transform.transform(localTransform));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			gScene->addActor(*body);
		}
	}
	shape->release();
}

void initPhysics(bool interactive)
{
	gFoundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
	}

	// Register after optional PVD connection because PVD profile connection
	// can replace the global callback.
	PxSetProfilerCallback(&gCustomProfilerCallback);
	if(gHeadlessOptions.headless)
	{
		gMetrics.callbackInstalled =
			PxGetProfilerCallback() == &gCustomProfilerCallback ? 1u : 0u;
	}

	gPhysics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless
			? gHeadlessOptions.dispatcherThreads : 2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
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
	PxRigidStatic* groundPlane = PxCreatePlane(
		*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*groundPlane);

	for(PxU32 i = 0; i < 5; ++i)
	{
		stackZ -= 10.0f;
		createStack(PxTransform(PxVec3(0, 0, stackZ)), 10, 2.0f);
	}

	if(!interactive)
	{
		const PxTransform projectilePose(PxVec3(0, 40, 100));
		gProjectile = createDynamic(
			projectilePose, PxSphereGeometry(10),
			PxVec3(0, -50, -100));
		gMetrics.initialProjectilePosition = projectilePose.p;
		gMetrics.finalProjectilePosition = projectilePose.p;
		gMetrics.initialStaticActors =
			gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
		gMetrics.initialDynamicActors =
			gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	}
}

static void recordSceneState()
{
	const PxU32 actorCount =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	PxArray<PxActor*> actors(actorCount);
	if(actorCount)
	{
		gScene->getActors(
			PxActorTypeFlag::eRIGID_DYNAMIC, actors.begin(), actorCount);
	}
	for(PxU32 i = 0; i < actorCount; ++i)
	{
		PxRigidDynamic* body = actors[i]->is<PxRigidDynamic>();
		if(!body)
		{
			++gMetrics.nonFinite;
			continue;
		}
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 linearVelocity = body->getLinearVelocity();
		const PxVec3 angularVelocity = body->getAngularVelocity();
		if(!pose.isFinite() || !linearVelocity.isFinite() ||
			!angularVelocity.isFinite())
			++gMetrics.nonFinite;
		gMetrics.minBodyY = PxMin(gMetrics.minBodyY, pose.p.y);
		gMetrics.maxBodySpeed = PxMax(
			gMetrics.maxBodySpeed, linearVelocity.magnitude());
	}

	if(gProjectile)
	{
		const PxTransform pose = gProjectile->getGlobalPose();
		gMetrics.finalProjectilePosition = pose.p;
		gMetrics.finalProjectileVelocity =
			gProjectile->getLinearVelocity();
	}
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = gHeadlessOptions.headless
		? gHeadlessOptions.dt : 1.0f / 60.0f;
	const uint64_t startsBefore = gCustomProfilerCallback.zoneStarts();
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gCustomProfilerCallback.zoneStarts() > startsBefore)
			++gMetrics.profiledFrames;
		recordSceneState();
		++gMetrics.completedFrames;
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	gProjectile = NULL;
	PX_RELEASE(gScene);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);

	PxSetProfilerCallback(NULL);
	if(gHeadlessOptions.headless)
	{
		gMetrics.callbackCleared =
			PxGetProfilerCallback() == NULL ? 1u : 0u;
	}

	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);

	gMetrics.cleanupComplete =
		!gScene && !gMaterial && !gDispatcher && !gPhysics &&
		!gPvd && !gFoundation && !gProjectile &&
		PxGetProfilerCallback() == NULL ? 1u : 0u;

#if (PX_DEBUG || PX_CHECKED || PX_PROFILE)
	std::printf("SnippetCustomProfiler done.\n");
#else
	std::printf(
		"Warning: SnippetCustomProfiler does not capture profiler "
		"timings in release build.\n");
#endif
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case 'B':
		stackZ -= 10.0f;
		createStack(PxTransform(PxVec3(0, 0, stackZ)), 10, 2.0f);
		break;
	case ' ':
		createDynamic(
			camera, PxSphereGeometry(3.0f),
			camera.rotate(PxVec3(0, 0, -1)) * 200);
		break;
	}
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig(
		"SnippetCustomProfiler", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gScene && gMaterial &&
		gProjectile && gMetrics.callbackInstalled == 1 &&
		gMetrics.initialStaticActors == 1 &&
		gMetrics.initialDynamicActors == 276;
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
	}

	cleanupPhysics(false);

	const uint64_t zoneStarts = gCustomProfilerCallback.zoneStarts();
	const uint64_t zoneEnds = gCustomProfilerCallback.zoneEnds();
	const uint64_t detachedStarts =
		gCustomProfilerCallback.detachedStarts();
	const uint64_t detachedEnds = gCustomProfilerCallback.detachedEnds();
	const PxReal projectileDisplacement =
		(gMetrics.finalProjectilePosition -
			gMetrics.initialProjectilePosition).magnitude();

	bool passed = true;
	const char* reason = "none";
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
	else if(gMetrics.profiledFrames != gHeadlessOptions.frames ||
		zoneStarts == 0)
	{
		passed = false;
		reason = "missing_profiler_activity";
	}
	else if(zoneStarts != zoneEnds ||
		detachedStarts != detachedEnds)
	{
		passed = false;
		reason = "unbalanced_profiler_zones";
	}
	else if(gCustomProfilerCallback.invalidNames() != 0)
	{
		passed = false;
		reason = "invalid_profiler_event";
	}
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(!gMetrics.finalProjectilePosition.isFinite() ||
		!gMetrics.finalProjectileVelocity.isFinite() ||
		!PxIsFinite(gMetrics.maxBodySpeed) ||
		projectileDisplacement <= 1.0f)
	{
		passed = false;
		reason = "simulation_workload_inactive";
	}
	else if(!gMetrics.callbackCleared ||
		!gMetrics.cleanupComplete)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetCustomProfiler solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED callbackInstalled=%u "
		"callbackCleared=%u profiledFrames=%u zoneStarts=%llu "
		"zoneEnds=%llu detachedStarts=%llu detachedEnds=%llu "
		"integerData=%llu floatData=%llu recordFrames=%llu "
		"invalidNames=%llu initialStaticActors=%u "
		"initialDynamicActors=%u minBodyY=%.9g maxBodySpeed=%.9g "
		"projectileDisplacement=%.9g finalProjectileX=%.9g "
		"finalProjectileY=%.9g finalProjectileZ=%.9g "
		"nonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, gMetrics.callbackInstalled,
		gMetrics.callbackCleared, gMetrics.profiledFrames,
		static_cast<unsigned long long>(zoneStarts),
		static_cast<unsigned long long>(zoneEnds),
		static_cast<unsigned long long>(detachedStarts),
		static_cast<unsigned long long>(detachedEnds),
		static_cast<unsigned long long>(
			gCustomProfilerCallback.integerData()),
		static_cast<unsigned long long>(
			gCustomProfilerCallback.floatData()),
		static_cast<unsigned long long>(
			gCustomProfilerCallback.frames()),
		static_cast<unsigned long long>(
			gCustomProfilerCallback.invalidNames()),
		gMetrics.initialStaticActors, gMetrics.initialDynamicActors,
		double(gMetrics.minBodyY), double(gMetrics.maxBodySpeed),
		double(projectileDisplacement),
		double(gMetrics.finalProjectilePosition.x),
		double(gMetrics.finalProjectilePosition.y),
		double(gMetrics.finalProjectilePosition.z),
		gMetrics.nonFinite, gMetrics.fetchFailures,
		gErrorCallback.getFatalCount(), gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char* const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(
			stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomProfiler "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(
			stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomProfiler "
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
	for(PxU32 i = 0; i < frameCount; ++i)
		stepPhysics(false);
	cleanupPhysics(false);
#endif

	return 0;
}
