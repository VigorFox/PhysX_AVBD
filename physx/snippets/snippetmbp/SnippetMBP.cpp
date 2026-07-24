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
// This snippet demonstrates the use of broad phase regions (MBP).
//
// It shows the setup of MBP and its regions. In this example 4 regions are setup
// and set for the MBP. Created stacks are then simulated in multiple regions.
// Note that current regions setup is not optimal, some objects get out of regions bounds.
// In this case a warning is reported. It is possible to add PxBroadPhaseCallback
// to scene to handle such cases. 
//
// ****************************************************************************

#include <ctype.h>
#include <cfloat>
#include <vector>
#include "PxPhysicsAPI.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;

static PxReal stackZ = 10.0f;

PxU32 gRegionHandles[4];
static Snippets::HeadlessOptions gHeadlessOptions;
static bool gSolverReadbackMatched = false;
static PxU32 gCleanupComplete = 0;

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0))
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
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

class SnippetMBPBroadPhaseCallback : public physx::PxBroadPhaseCallback
{
	PxArray<PxActor*> outOfBoundsActors;
	PxU32 outOfBoundsCount;
	PxU32 purgedCount;
public:
	SnippetMBPBroadPhaseCallback()
	: outOfBoundsCount(0), purgedCount(0)
	{
	}

	virtual void onObjectOutOfBounds(PxShape& /*shape*/, PxActor& actor)
	{
		outOfBoundsCount++;
		PxU32 i = 0;
		for(; i < outOfBoundsActors.size(); ++i)
		{
			if(outOfBoundsActors[i] == &actor)
				break;
		}
		if(i == outOfBoundsActors.size())
		{
			outOfBoundsActors.pushBack(&actor);
		}
	}

	virtual void onObjectOutOfBounds(PxAggregate& /*aggregate*/)
	{
		//This test does not use aggregates so no need to do anything here
	}

	void purgeOutOfBoundsObjects()
	{
		for(PxU32 i = 0; i < outOfBoundsActors.size(); ++i)
		{
			outOfBoundsActors[i]->release();
			purgedCount++;
		}
		outOfBoundsActors.clear();
	}
    
    void release()
    {
        outOfBoundsActors.reset();
    }

	void resetMetrics()
	{
		outOfBoundsActors.clear();
		outOfBoundsCount = 0;
		purgedCount = 0;
	}

	PxU32 getOutOfBoundsCount() const { return outOfBoundsCount; }
	PxU32 getPurgedCount() const { return purgedCount; }
} gBroadPhaseCallback;


static bool initPhysicsInternal(bool interactive, bool headless)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	if(!headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && transport)
			gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
		
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	if(headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	
	PxU32 numCores = SnippetUtils::getNbPhysicalCores();
	gDispatcher = PxDefaultCpuDispatcherCreate(headless ?
		gHeadlessOptions.dispatcherThreads :
		(numCores == 0 ? 0 : numCores - 1));
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;

	sceneDesc.broadPhaseType = PxBroadPhaseType::eMBP;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	gSolverReadbackMatched =
		!headless || gScene->getSolverType() == sceneDesc.solverType;
	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	PxBroadPhaseRegion regions[4] =
	{
		{	PxBounds3(PxVec3(-100, -100, -100),  PxVec3(  0, 100,   0)), reinterpret_cast<void*>(1) },
		{	PxBounds3(PxVec3(-100, -100,    0),  PxVec3(  0, 100, 100)), reinterpret_cast<void*>(2) }, 
		{	PxBounds3(PxVec3(   0, -100, -100),  PxVec3(100, 100,   0)), reinterpret_cast<void*>(3) },
		{	PxBounds3(PxVec3(   0, -100,    0),  PxVec3(100, 100, 100)), reinterpret_cast<void*>(4) }
	};

	for(PxU32 i=0;i<4;i++)
		gRegionHandles[i] = gScene->addBroadPhaseRegion(regions[i]);

	gScene->setBroadPhaseCallback(&gBroadPhaseCallback);

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	for(PxU32 i=0;i<5;i++)
		createStack(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 10, 2.0f);

	if(!interactive)
		createDynamic(PxTransform(PxVec3(0,40,100)), PxSphereGeometry(10), PxVec3(0,-50,-100));
	return true;
}

void initPhysics(bool interactive)
{
	initPhysicsInternal(interactive, false);
}

static bool stepPhysicsInternal()
{
	gScene->simulate(gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f);
	if(!gScene->fetchResults(true))
		return false;
	gBroadPhaseCallback.purgeOutOfBoundsObjects();
	return true;
}

void stepPhysics(bool /*interactive*/)
{
	stepPhysicsInternal();
}
	
void cleanupPhysics(bool /*interactive*/)
{
    gBroadPhaseCallback.release();
    
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
	gCleanupComplete = 1;

	printf("SnippetMBP done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case 'B':	createStack(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 10, 2.0f);						break;
	case ' ':	createDynamic(camera, PxSphereGeometry(3.0f), camera.rotate(PxVec3(0,0,-1))*200);	break;
	}
}

static PxU32 queryDynamicStacks()
{
	PxU32 hits = 0;
	for(PxU32 i=0; i<5; ++i)
	{
		const PxReal z = -10.0f * PxReal(i);
		PxRaycastBuffer hit;
		if(gScene->raycast(PxVec3(0.0f, 80.0f, z),
			PxVec3(0.0f, -1.0f, 0.0f), 160.0f, hit) &&
		   hit.hasBlock && hit.block.actor &&
		   hit.block.actor->is<PxRigidDynamic>())
			hits++;
	}
	return hits;
}

static int runHeadless()
{
	gErrorCallback.reset();
	gBroadPhaseCallback.resetMetrics();
	gCleanupComplete = 0;
	stackZ = 10.0f;
	if(!initPhysicsInternal(false, true))
	{
		cleanupPhysics(false);
		return Snippets::eHEADLESS_GATE_FAILED;
	}
	PxU32 validRegions = 0;
	for(PxU32 i=0; i<4; ++i)
		validRegions += gRegionHandles[i] != PX_INVALID_U32 ? 1u : 0u;
	const PxU32 initialDynamicActors =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	const PxU32 queryHitsBefore = queryDynamicStacks();
	PxU32 completedFrames = 0;
	PxU32 fetchFailures = 0;
	PxU32 nonFinite = 0;
	PxReal minDynamicY = FLT_MAX;
	PxReal maxDynamicSpeed = 0.0f;
	for(PxU32 frame=0; frame<gHeadlessOptions.frames; ++frame)
	{
		if(!stepPhysicsInternal())
		{
			fetchFailures++;
			break;
		}
		completedFrames++;
		const PxU32 count =
			gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
		std::vector<PxActor*> actors(count);
		if(count)
			gScene->getActors(
				PxActorTypeFlag::eRIGID_DYNAMIC, actors.data(), count);
		for(PxU32 i=0; i<count; ++i)
		{
			PxRigidDynamic* dynamic = actors[i]->is<PxRigidDynamic>();
			if(!dynamic || !dynamic->getGlobalPose().isFinite() ||
			   !dynamic->getLinearVelocity().isFinite() ||
			   !dynamic->getAngularVelocity().isFinite())
			{
				nonFinite++;
				continue;
			}
			minDynamicY = PxMin(
				minDynamicY, dynamic->getGlobalPose().p.y);
			maxDynamicSpeed = PxMax(
				maxDynamicSpeed,
				dynamic->getLinearVelocity().magnitude());
		}
		if(nonFinite)
			break;
	}
	const PxU32 finalDynamicActors =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	const PxU32 queryHitsAfter = queryDynamicStacks();
	const PxU32 outOfBounds = gBroadPhaseCallback.getOutOfBoundsCount();
	const PxU32 purged = gBroadPhaseCallback.getPurgedCount();
	const bool passed =
		gScene->getBroadPhaseType() == PxBroadPhaseType::eMBP &&
		validRegions == 4 && initialDynamicActors == 276 &&
		finalDynamicActors + purged == initialDynamicActors &&
		queryHitsBefore == 5 && queryHitsAfter == 5 &&
		outOfBounds > 0 && purged > 0 && outOfBounds >= purged &&
		completedFrames == gHeadlessOptions.frames &&
		minDynamicY > 1.0f && nonFinite == 0 && fetchFailures == 0 &&
		gSolverReadbackMatched && gErrorCallback.getFatalCount() == 0;
	const PxU32 fatalErrors = gErrorCallback.getFatalCount();
	cleanupPhysics(false);
	printf("[AVBD_GATE] schema=1 snippet=SnippetMBP solver=%s "
		"case=mbp-regions execution=%s frames=%u regions=%u "
		"initialDynamicActors=%u finalDynamicActors=%u "
		"queryHitsBefore=%u queryHitsAfter=%u outOfBounds=%u purged=%u "
		"completedFrames=%u minDynamicY=%.9g maxDynamicSpeed=%.9g "
		"nonFinite=%u fetchFailures=%u fatalErrors=%u cleanupComplete=%u "
		"pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, validRegions, initialDynamicActors,
		finalDynamicActors, queryHitsBefore, queryHitsAfter, outOfBounds,
		purged, completedFrames, minDynamicY, maxDynamicSpeed, nonFinite,
		fetchFailures, fatalErrors, gCleanupComplete,
		passed ? "PASS" : "FAIL",
		passed ? "none" : "mbp_region_or_dynamics");
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.caseName = "mbp-regions";
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] snippet=SnippetMBP reason=%s\n",
			error.c_str());
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
