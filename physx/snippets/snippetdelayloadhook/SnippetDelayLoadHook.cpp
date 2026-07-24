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
// This snippet illustrates the use of the dll delay load hooks in physx.
//
// The hooks are needed if the application executable either doesn't reside 
// in the same directory as the PhysX dlls, or if the PhysX dlls have been renamed.
// Some PhysX dlls delay load the PhysXFoundation, PhysXCommon or PhysXGpu dlls and
// the non-standard names or loactions of these dlls need to be communicated so the 
// delay loading can succeed.
//
// This snippet shows how this can be done using the delay load hooks.
//
// In order to show functionality with the renamed dlls some basic physics 
// simulation is performed.
// ****************************************************************************

#include <ctype.h>
#include <wtypes.h>
#include "PxPhysicsAPI.h"
// Include the delay load hook headers
#include "common/windows/PxWindowsDelayLoadHook.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

// This snippet uses the default PhysX distro dlls, making the example here somewhat artificial, 
// as default locations and default naming makes implementing delay load hooks unnecessary.
#define APP_BIN_DIR "..\\"
#if PX_WIN64
	#define DLL_NAME_BITS "64" 
#else
	#define DLL_NAME_BITS "32" 
#endif
#if PX_DEBUG
	#define DLL_DIR "debug\\"
#elif PX_CHECKED
	#define DLL_DIR "checked\\" 
#elif PX_PROFILE
	#define DLL_DIR "profile\\" 
#else
	#define DLL_DIR "release\\" 
#endif

const char* foundationLibraryPath = APP_BIN_DIR DLL_DIR "PhysXFoundation_" DLL_NAME_BITS ".dll";
const char* commonLibraryPath = APP_BIN_DIR DLL_DIR "PhysXCommon_" DLL_NAME_BITS ".dll";
const char* physxLibraryPath = APP_BIN_DIR DLL_DIR "PhysX_" DLL_NAME_BITS ".dll";
const char* gpuLibraryPath = APP_BIN_DIR DLL_DIR "PhysXGpu_" DLL_NAME_BITS ".dll";

HMODULE foundationLibrary = NULL;
HMODULE commonLibrary = NULL;
HMODULE physxLibrary = NULL;

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
PxScene*						gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;
static PxRigidDynamic*			gProjectile = NULL;
static bool						gExtensionsInitialized = false;
static bool						gInitializationFailed = false;
static Snippets::HeadlessOptions gHeadlessOptions;

struct DelayLoadMetrics
{
	PxU32 foundationLoaded;
	PxU32 commonLoaded;
	PxU32 physxLoaded;
	PxU32 foundationPathMatched;
	PxU32 commonPathMatched;
	PxU32 physxPathMatched;
	PxU32 exportsResolved;
	PxU32 hooksRegistered;
	PxU32 initialized;
	PxU32 solverReadbackMatched;
	PxU32 sceneStatics;
	PxU32 sceneDynamics;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFiniteActorSamples;
	PxU32 unloadCompleted;
	PxU32 cleanupComplete;
	PxReal projectileDisplacement;
	PxReal maxProjectileSpeed;
	PxVec3 projectileInitialPosition;

	DelayLoadMetrics()
	: foundationLoaded(0), commonLoaded(0), physxLoaded(0),
	  foundationPathMatched(0), commonPathMatched(0), physxPathMatched(0),
	  exportsResolved(0), hooksRegistered(0), initialized(0),
	  solverReadbackMatched(0), sceneStatics(0), sceneDynamics(0),
	  completedFrames(0), fetchFailures(0), nonFiniteActorSamples(0),
	  unloadCompleted(0), cleanupComplete(0), projectileDisplacement(0.0f),
	  maxProjectileSpeed(0.0f), projectileInitialPosition(0.0f)
	{
	}
};

static DelayLoadMetrics gMetrics;

// typedef the PhysX entry points
typedef PxFoundation*(PxCreateFoundation_FUNC)(PxU32, PxAllocatorCallback&, PxErrorCallback&);
typedef PxPhysics* (PxCreatePhysics_FUNC)(PxU32,PxFoundation&,const PxTolerancesScale& scale,bool,PxPvd*);
typedef void (PxSetPhysXDelayLoadHook_FUNC)(const PxDelayLoadHook* hook);
typedef void (PxSetPhysXCommonDelayLoadHook_FUNC)(const PxDelayLoadHook* hook);
#if PX_SUPPORT_GPU_PHYSX
typedef void (PxSetPhysXGpuLoadHook_FUNC)(const PxGpuLoadHook* hook);
typedef int (PxGetSuggestedCudaDeviceOrdinal_FUNC)(PxErrorCallback& errc);
typedef PxCudaContextManager* (PxCreateCudaContextManager_FUNC)(PxFoundation& foundation, const PxCudaContextManagerDesc& desc, physx::PxProfilerCallback* profilerCallback);
#endif

// set the function pointers to NULL
PxCreateFoundation_FUNC* s_PxCreateFoundation_Func = NULL;
PxCreatePhysics_FUNC* s_PxCreatePhysics_Func = NULL;
PxSetPhysXDelayLoadHook_FUNC* s_PxSetPhysXDelayLoadHook_Func = NULL;
PxSetPhysXCommonDelayLoadHook_FUNC* s_PxSetPhysXCommonDelayLoadHook_Func = NULL;
#if PX_SUPPORT_GPU_PHYSX
PxSetPhysXGpuLoadHook_FUNC* s_PxSetPhysXGpuLoadHook_Func = NULL;
PxGetSuggestedCudaDeviceOrdinal_FUNC* s_PxGetSuggestedCudaDeviceOrdinal_Func = NULL;
PxCreateCudaContextManager_FUNC* s_PxCreateCudaContextManager_Func = NULL;
#endif

static bool loadedModuleMatchesPath(
	HMODULE module, const char* configuredPath)
{
	char modulePath[MAX_PATH] = {};
	char expectedPath[MAX_PATH] = {};
	const DWORD moduleLength =
		GetModuleFileNameA(module, modulePath, MAX_PATH);
	const DWORD expectedLength =
		GetFullPathNameA(configuredPath, MAX_PATH, expectedPath, NULL);
	return moduleLength > 0 && moduleLength < MAX_PATH &&
		expectedLength > 0 && expectedLength < MAX_PATH &&
		_stricmp(modulePath, expectedPath) == 0;
}

bool loadPhysicsExplicitely()
{
	// load the dlls
	foundationLibrary = LoadLibraryA(foundationLibraryPath);	
	if(!foundationLibrary)
		return false;
	gMetrics.foundationLoaded = 1;
	gMetrics.foundationPathMatched =
		loadedModuleMatchesPath(foundationLibrary, foundationLibraryPath) ? 1u : 0u;

	commonLibrary = LoadLibraryA(commonLibraryPath);	
	if(!commonLibrary)
	{
		FreeLibrary(foundationLibrary);
		foundationLibrary = NULL;
		return false;
	}
	gMetrics.commonLoaded = 1;
	gMetrics.commonPathMatched =
		loadedModuleMatchesPath(commonLibrary, commonLibraryPath) ? 1u : 0u;

	physxLibrary = LoadLibraryA(physxLibraryPath);	
	if(!physxLibrary)
	{
		FreeLibrary(foundationLibrary);
		FreeLibrary(commonLibrary);
		foundationLibrary = NULL;
		commonLibrary = NULL;
		return false;
	}
	gMetrics.physxLoaded = 1;
	gMetrics.physxPathMatched =
		loadedModuleMatchesPath(physxLibrary, physxLibraryPath) ? 1u : 0u;

	// get the function pointers
	s_PxCreateFoundation_Func = (PxCreateFoundation_FUNC*)GetProcAddress(foundationLibrary, "PxCreateFoundation");
	s_PxCreatePhysics_Func = (PxCreatePhysics_FUNC*)GetProcAddress(physxLibrary, "PxCreatePhysics");
	s_PxSetPhysXDelayLoadHook_Func = (PxSetPhysXDelayLoadHook_FUNC*)GetProcAddress(physxLibrary, "PxSetPhysXDelayLoadHook");
	s_PxSetPhysXCommonDelayLoadHook_Func = (PxSetPhysXCommonDelayLoadHook_FUNC*)GetProcAddress(commonLibrary, "PxSetPhysXCommonDelayLoadHook");

#if PX_SUPPORT_GPU_PHYSX
	s_PxSetPhysXGpuLoadHook_Func = (PxSetPhysXGpuLoadHook_FUNC*)GetProcAddress(physxLibrary, "PxSetPhysXGpuLoadHook");
	s_PxGetSuggestedCudaDeviceOrdinal_Func = (PxGetSuggestedCudaDeviceOrdinal_FUNC*)GetProcAddress(physxLibrary, "PxGetSuggestedCudaDeviceOrdinal");
	s_PxCreateCudaContextManager_Func = (PxCreateCudaContextManager_FUNC*)GetProcAddress(physxLibrary, "PxCreateCudaContextManager");
#endif

	// check if we have all required function pointers
	if(s_PxCreateFoundation_Func == NULL || s_PxCreatePhysics_Func == NULL || s_PxSetPhysXDelayLoadHook_Func == NULL || s_PxSetPhysXCommonDelayLoadHook_Func == NULL)
		return false;
	gMetrics.exportsResolved = 4;

#if PX_SUPPORT_GPU_PHYSX
	if(s_PxSetPhysXGpuLoadHook_Func == NULL || s_PxGetSuggestedCudaDeviceOrdinal_Func == NULL || s_PxCreateCudaContextManager_Func == NULL)
		return false;
#endif
	return true;
}

// unload the dlls
void unloadPhysicsExplicitely()
{
	bool unloaded = true;
	if(physxLibrary)
		unloaded = FreeLibrary(physxLibrary) != 0 && unloaded;
	if(commonLibrary)
		unloaded = FreeLibrary(commonLibrary) != 0 && unloaded;
	if(foundationLibrary)
		unloaded = FreeLibrary(foundationLibrary) != 0 && unloaded;
	physxLibrary = NULL;
	commonLibrary = NULL;
	foundationLibrary = NULL;
	s_PxCreateFoundation_Func = NULL;
	s_PxCreatePhysics_Func = NULL;
	s_PxSetPhysXDelayLoadHook_Func = NULL;
	s_PxSetPhysXCommonDelayLoadHook_Func = NULL;
	gMetrics.unloadCompleted = unloaded ? 1u : 0u;
}

// Overriding the PxDelayLoadHook allows the load of a custom name dll inside PhysX, PhysXCommon and PhysXCooking dlls
struct SnippetDelayLoadHook : public PxDelayLoadHook
{
	virtual const char* getPhysXFoundationDllName() const 
	{
		return foundationLibraryPath;
	}

	virtual const char* getPhysXCommonDllName() const 
	{
		return commonLibraryPath;
	}
};

static SnippetDelayLoadHook gDelayLoadHook;

#if PX_SUPPORT_GPU_PHYSX
// Overriding the PxGpuLoadHook allows the load of a custom GPU name dll
struct SnippetGpuLoadHook : public PxGpuLoadHook
{
	virtual const char* getPhysXGpuDllName() const
	{
		return gpuLibraryPath;
	}
};

static SnippetGpuLoadHook gGpuLoadHook;
#endif

PxReal stackZ = 10.0f;

PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0))
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
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
		}
	}
	shape->release();
}

void initPhysics(bool interactive)
{	
	gInitializationFailed = false;
	gMetrics = DelayLoadMetrics();
	gErrorCallback.reset();

	// load the explictely named dlls
	const bool isLoaded = loadPhysicsExplicitely();
	if (!isLoaded)
	{
		gInitializationFailed = true;
		return;
	}
	gFoundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
	{
		gInitializationFailed = true;
		return;
	}
	// set PhysX and PhysXCommon delay load hook, this must be done before the create physics is called, before
	// the PhysXFoundation, PhysXCommon delay load happens.
	s_PxSetPhysXDelayLoadHook_Func(&gDelayLoadHook);
	s_PxSetPhysXCommonDelayLoadHook_Func(&gDelayLoadHook);
	gMetrics.hooksRegistered = 2;

#if PX_SUPPORT_GPU_PHYSX
	// set PhysXGpu load hook
	s_PxSetPhysXGpuLoadHook_Func(&gGpuLoadHook);
#endif

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			PxPvdTransport* transport =
				PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(transport)
				gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
		}
	}

	gPhysics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
	{
		gInitializationFailed = true;
		return;
	}
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
	{
		gInitializationFailed = true;
		return;
	}
	// We setup the delay load hooks first

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	if(!interactive)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gDispatcher = PxDefaultCpuDispatcherCreate(
		interactive ? 2u : gHeadlessOptions.dispatcherThreads);
	if(!gDispatcher)
	{
		gInitializationFailed = true;
		return;
	}
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
	{
		gInitializationFailed = true;
		return;
	}
	gMetrics.solverReadbackMatched =
		gScene->getSolverType() == sceneDesc.solverType ? 1u : 0u;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(interactive && pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
	{
		gInitializationFailed = true;
		return;
	}

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	if(!groundPlane)
	{
		gInitializationFailed = true;
		return;
	}
	gScene->addActor(*groundPlane);

	for(PxU32 i=0;i<5;i++)
		createStack(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 10, 2.0f);

	if(!interactive)
	{
		gProjectile = createDynamic(
			PxTransform(PxVec3(0,40,100)), PxSphereGeometry(10),
			PxVec3(0,-50,-100));
		gMetrics.projectileInitialPosition =
			gProjectile->getGlobalPose().p;
	}

	gMetrics.sceneStatics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
	gMetrics.sceneDynamics =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	gMetrics.initialized = 1;
}

void stepPhysics(bool interactive)
{
	if (gScene)
	{
		gScene->simulate(interactive ? 1.0f/60.0f : gHeadlessOptions.dt);
		if(!gScene->fetchResults(true))
			gMetrics.fetchFailures++;
		else if(!interactive)
			gMetrics.completedFrames++;

		if(!interactive)
		{
			const PxU32 actorCount =
				gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
			PxArray<PxRigidDynamic*> actors(actorCount);
			if(actorCount)
			{
				gScene->getActors(
					PxActorTypeFlag::eRIGID_DYNAMIC,
					reinterpret_cast<PxActor**>(actors.begin()), actorCount);
			}
			for(PxU32 actorId = 0; actorId < actorCount; ++actorId)
			{
				const PxTransform pose = actors[actorId]->getGlobalPose();
				const PxVec3 linearVelocity =
					actors[actorId]->getLinearVelocity();
				const PxVec3 angularVelocity =
					actors[actorId]->getAngularVelocity();
				if(!pose.isFinite() || !linearVelocity.isFinite() ||
					!angularVelocity.isFinite())
				{
					gMetrics.nonFiniteActorSamples++;
				}
			}
			if(gProjectile)
			{
				gMetrics.projectileDisplacement = PxMax(
					gMetrics.projectileDisplacement,
					(gProjectile->getGlobalPose().p -
						gMetrics.projectileInitialPosition).magnitude());
				gMetrics.maxProjectileSpeed = PxMax(
					gMetrics.maxProjectileSpeed,
					gProjectile->getLinearVelocity().magnitude());
			}
		}
	}
}
	
void cleanupPhysics(bool /*interactive*/)
{
	gProjectile = NULL;
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
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
	
	PX_RELEASE(gFoundation);

	unloadPhysicsExplicitely();
	gMetrics.cleanupComplete =
		!gScene && !gDispatcher && !gMaterial && !gPhysics &&
		!gPvd && !gFoundation && gMetrics.unloadCompleted ? 1u : 0u;
	
	printf("SnippetDelayLoadHook done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case 'B':	createStack(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 10, 2.0f);						break;
	case ' ':	createDynamic(camera, PxSphereGeometry(3.0f), camera.rotate(PxVec3(0,0,-1))*200);	break;
	}
}

static int reportConfigurationError(
	const Snippets::HeadlessOptions& options, const char* reason)
{
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetDelayLoadHook solver=%s "
		"case=%s execution=%s status=CONFIG_ERROR reason=%s "
		"validation=GATED\n",
		Snippets::getSolverTypeName(options.solverType),
		options.caseName.c_str(),
		Snippets::getExecutionName(options.execution), reason);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

static bool evaluateHeadlessGate(const char*& reason)
{
	reason = "none";
	if(gInitializationFailed || !gMetrics.initialized)
		reason = "initialization";
	else if(gMetrics.foundationLoaded != 1 ||
		gMetrics.commonLoaded != 1 || gMetrics.physxLoaded != 1)
		reason = "explicit_load";
	else if(gMetrics.foundationPathMatched != 1 ||
		gMetrics.commonPathMatched != 1 || gMetrics.physxPathMatched != 1)
		reason = "configured_path";
	else if(gMetrics.exportsResolved != 4 || gMetrics.hooksRegistered != 2)
		reason = "hook_exports";
	else if(!gMetrics.solverReadbackMatched)
		reason = "solver_readback";
	else if(gMetrics.sceneStatics != 1 || gMetrics.sceneDynamics != 276)
		reason = "scene_topology";
	else if(gMetrics.completedFrames != gHeadlessOptions.frames ||
		gMetrics.fetchFailures)
		reason = "simulation";
	else if(gMetrics.nonFiniteActorSamples)
		reason = "non_finite";
	else if(gMetrics.projectileDisplacement <= 1.0f ||
		gMetrics.maxProjectileSpeed <= 1.0f)
		reason = "rigid_response";
	else if(gErrorCallback.getFatalCount())
		reason = "physx_error";
	else if(!gMetrics.cleanupComplete)
		reason = "cleanup";
	return strcmp(reason, "none") == 0;
}

static void printHeadlessGate(bool passed, const char* reason)
{
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetDelayLoadHook solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u "
		"foundationLoaded=%u commonLoaded=%u physxLoaded=%u "
		"foundationPathMatched=%u commonPathMatched=%u "
		"physxPathMatched=%u exportsResolved=%u hooksRegistered=%u "
		"initialized=%u solverReadbackMatched=%u sceneStatics=%u "
		"sceneDynamics=%u projectileDisplacement=%.9g "
		"maxProjectileSpeed=%.9g nonFiniteActorSamples=%u "
		"fetchFailures=%u fatalErrors=%u warnings=%u "
		"unloadCompleted=%u cleanupComplete=%u pvd=0 status=%s "
		"reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		gMetrics.foundationLoaded, gMetrics.commonLoaded,
		gMetrics.physxLoaded, gMetrics.foundationPathMatched,
		gMetrics.commonPathMatched, gMetrics.physxPathMatched,
		gMetrics.exportsResolved, gMetrics.hooksRegistered,
		gMetrics.initialized, gMetrics.solverReadbackMatched,
		gMetrics.sceneStatics, gMetrics.sceneDynamics,
		double(gMetrics.projectileDisplacement),
		double(gMetrics.maxProjectileSpeed),
		gMetrics.nonFiniteActorSamples, gMetrics.fetchFailures,
		gErrorCallback.getFatalCount(), gErrorCallback.getWarningCount(),
		gMetrics.unloadCompleted, gMetrics.cleanupComplete,
		passed ? "PASS" : "FAIL", reason);
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.frames = 180;
	defaults.caseName = "delay-load-scene";
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
	{
		return reportConfigurationError(gHeadlessOptions, "invalid_arguments");
	}
	if(gHeadlessOptions.headless &&
		!Snippets::equalsIgnoreCase(
			gHeadlessOptions.caseName.c_str(), "delay-load-scene"))
	{
		return reportConfigurationError(gHeadlessOptions, "invalid_case");
	}
	if(gHeadlessOptions.headless &&
		gHeadlessOptions.execution == Snippets::eHEADLESS_SEQUENTIAL &&
		gHeadlessOptions.solverType != PxSolverType::eAVBD)
	{
		return reportConfigurationError(
			gHeadlessOptions, "sequential_requires_avbd");
	}
	if(gHeadlessOptions.headless &&
		!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		return reportConfigurationError(
			gHeadlessOptions, "execution_environment");
	}

#ifdef RENDER_SNIPPET
	if(gHeadlessOptions.headless)
	{
		setvbuf(stdout, NULL, _IONBF, 0);
		Snippets::printHeadlessConfig(
			"SnippetDelayLoadHook", gHeadlessOptions);
		initPhysics(false);
		if(!gInitializationFailed)
		{
			for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			{
				stepPhysics(false);
				if(gMetrics.fetchFailures ||
					gMetrics.nonFiniteActorSamples)
					break;
			}
		}
		cleanupPhysics(false);
		const char* reason = "none";
		const bool passed = evaluateHeadlessGate(reason);
		printHeadlessGate(passed, reason);
		return passed ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}
	extern void renderLoop();
	renderLoop();
#else
	gHeadlessOptions.headless = true;
	Snippets::printHeadlessConfig(
		"SnippetDelayLoadHook", gHeadlessOptions);
	initPhysics(false);
	if(!gInitializationFailed)
	{
		for(PxU32 i=0; i<gHeadlessOptions.frames; i++)
			stepPhysics(false);
	}
	cleanupPhysics(false);
	const char* reason = "none";
	const bool passed = evaluateHeadlessGate(reason);
	printHeadlessGate(passed, reason);
	return passed ?
		Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
#endif

	return 0;
}
