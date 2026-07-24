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
// This snippet illustrates the usage of PxPruningStructure.
//
// It creates a box stack, then prepares a pruning structure. This structure
// together with the actors is serialized into a collection. When the collection
// is added to the scene, the actor's scene query shape AABBs are directly merged
// into the current scene query AABB tree through the precomputed pruning structure.
// This may unbalance the AABB tree but should provide significant speedup in 
// case of large world scenarios where parts get streamed in on the fly.
// ****************************************************************************

#include <ctype.h>
#include <cfloat>
#include <vector>
#include "PxPhysicsAPI.h"
#include "extensions/PxCollectionExt.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;
static PxCollection*			gLoadedCollection = NULL;

#define MAX_MEMBLOCKS 10
PxU8*					gMemBlocks[MAX_MEMBLOCKS];
PxU32					gMemBlockCount = 0;

PxReal stackZ = 10.0f;
static Snippets::HeadlessOptions gHeadlessOptions;
static bool gSolverReadbackMatched = false;
static bool gSerializationSucceeded = false;
static PxU64 gSerializedBytes = 0;
static PxU32 gPruningStructuresCreated = 0;
static PxU32 gCleanupComplete = 0;

/**
Allocates 128 byte aligned memory block for binary serialized data
Stores pointer to memory in gMemBlocks for later deallocation
*/
void* createAlignedBlock(PxU64 size)
{
	PX_ASSERT(gMemBlockCount < MAX_MEMBLOCKS);
	PxU8* baseAddr = static_cast<PxU8*>(malloc(size + PX_SERIAL_FILE_ALIGN - 1));
	gMemBlocks[gMemBlockCount++] = baseAddr;
	void* alignedBlock = reinterpret_cast<void*>((size_t(baseAddr) + PX_SERIAL_FILE_ALIGN - 1)&~(PX_SERIAL_FILE_ALIGN - 1));
	return alignedBlock;
}

// Create a regular stack, with actors added directly into a scene.
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

// Create a stack where pruning structure is build in runtime and used to merge
// the query shapes into the AABB tree.
void createStackWithRuntimePrunerStructure(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxArray<PxRigidActor*> actors;
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for (PxU32 i = 0; i < size; i++)
	{
		for (PxU32 j = 0; j < size - i; j++)
		{
			PxTransform localTm(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			// store the actors, will be added later
			actors.pushBack(body);
		}
	}
	shape->release();

	// Create pruning structure from given actors.
	PxPruningStructure* ps = gPhysics->createPruningStructure(&actors[0], PxU32(actors.size()));
	if(ps)
		gPruningStructuresCreated++;
	// Add actors into a scene together with the precomputed pruning structure.	
	gScene->addActors(*ps);
	ps->release();
}

// Create a stack where pruning structure is build in runtime and then stored into a collection.
// The collection is stored into a stream and loaded into another stream. The loaded collection
// is added to a scene. While the collection is added to the scene the pruning structure is used.
void createStackWithSerializedPrunerStructure(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxCollection* collection = PxCreateCollection();		// collection for all the objects
	PxSerializationRegistry* sr = PxSerialization::createSerializationRegistry(*gPhysics);

	PxArray<PxRigidActor*> actors;
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for (PxU32 i = 0; i < size; i++)
	{
		for (PxU32 j = 0; j < size - i; j++)
		{
			PxTransform localTm(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			// store the actors, will be added later
			actors.pushBack(body);
		}
	}
	collection->add(*shape);	

	// Create pruner structure from given actors.
	PxPruningStructure* ps = gPhysics->createPruningStructure(&actors[0], PxU32(actors.size()));
	if(ps)
		gPruningStructuresCreated++;
	// Add the pruning structure into the collection. Adding the pruning structure will automatically
	// add the actors from which the collection was build.
	collection->add(*ps);
	PxSerialization::complete(*collection, *sr);

	// Store the collection into a stream.
	PxDefaultMemoryOutputStream outStream;
	const bool serialized =
		PxSerialization::serializeCollectionToBinary(outStream, *collection, *sr);
	gSerializedBytes = outStream.getSize();
	collection->release();

	// Release the used items added to the collection.
	ps->release();
	for (PxU32 i = 0; i < actors.size(); i++)
	{
		actors[i]->release();
	}	
	shape->release();

	// Load collection from the stream into and input stream.
	PxDefaultMemoryInputData inputStream(outStream.getData(), outStream.getSize());
	void* alignedBlock = createAlignedBlock(inputStream.getLength());
	inputStream.read(alignedBlock, inputStream.getLength());
	PxCollection* collection1 = PxSerialization::createCollectionFromBinary(alignedBlock, *sr);
	gSerializationSucceeded =
		serialized && outStream.getSize() > 0 && collection1 != NULL;

	// Add collection to the scene.
	if(collection1)
		gScene->addCollection(*collection1);

	// Keep the deserialized objects alive while the scene uses them. They are
	// released together, in pruning-structure-first order, during cleanup.
	gLoadedCollection = collection1;
	sr->release();
}

static bool initPhysicsInternal(bool headless)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	if(!headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && transport)
			gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(),true, gPvd);
	if(!gPhysics)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	if(headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gDispatcher = PxDefaultCpuDispatcherCreate(
		headless ? gHeadlessOptions.dispatcherThreads : 2);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
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
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	// Create a regular stack.
	createStack(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 3, 2.0f);
	// Create a stack using the runtime pruner structure usage.
	createStackWithRuntimePrunerStructure(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 3, 2.0f);
	// Create a stack using the serialized pruner structure usage.
	createStackWithSerializedPrunerStructure(PxTransform(PxVec3(0,0,stackZ-=10.0f)), 3, 2.0f);
	return true;
}

void initPhysics(bool)
{
	initPhysicsInternal(false);
}

static bool stepPhysicsInternal()
{
	gScene->simulate(gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f);
	return gScene->fetchResults(true);
}

void stepPhysics(bool /*interactive*/)
{
	stepPhysicsInternal();
}
	
void cleanupPhysics(bool /*interactive*/)
{
	if(gLoadedCollection)
	{
		PxCollectionExt::releaseObjects(*gLoadedCollection);
		PX_RELEASE(gLoadedCollection);
	}
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
	
	// Now that the objects have been released, it's safe to release the space they occupy.
	for (PxU32 i = 0; i < gMemBlockCount; i++)
		free(gMemBlocks[i]);

	gMemBlockCount = 0;

	PX_RELEASE(gFoundation);
	gCleanupComplete = 1;
	
	printf("SnippetPrunerSerialization done.\n");
}

static PxU32 queryDynamicStacks()
{
	PxU32 hits = 0;
	const PxReal stackPositions[3] = {0.0f, -10.0f, -20.0f};
	for(PxU32 i=0; i<3; ++i)
	{
		PxRaycastBuffer hit;
		if(gScene->raycast(PxVec3(0.0f, 50.0f, stackPositions[i]),
			PxVec3(0.0f, -1.0f, 0.0f), 100.0f, hit) &&
		   hit.hasBlock && hit.block.actor &&
		   hit.block.actor->is<PxRigidDynamic>())
			hits++;
	}
	return hits;
}

static int runHeadless()
{
	gErrorCallback.reset();
	gCleanupComplete = 0;
	gSerializationSucceeded = false;
	gSerializedBytes = 0;
	gPruningStructuresCreated = 0;
	gLoadedCollection = NULL;
	stackZ = 10.0f;
	if(!initPhysicsInternal(true))
	{
		cleanupPhysics(false);
		return Snippets::eHEADLESS_GATE_FAILED;
	}

	const PxActorTypeFlags actorTypes =
		PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC;
	const PxU32 actorCount = gScene->getNbActors(actorTypes);
	const PxU32 dynamicCount =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	std::vector<PxActor*> actors(actorCount);
	if(actorCount)
		gScene->getActors(actorTypes, actors.data(), actorCount);
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
		for(PxU32 i=0; i<actorCount; ++i)
		{
			PxRigidActor* rigidActor = actors[i]->is<PxRigidActor>();
			if(!rigidActor || !rigidActor->getGlobalPose().isFinite())
			{
				nonFinite++;
				continue;
			}
			PxRigidDynamic* dynamic = actors[i]->is<PxRigidDynamic>();
			if(dynamic)
			{
				if(!dynamic->getLinearVelocity().isFinite() ||
				   !dynamic->getAngularVelocity().isFinite())
					nonFinite++;
				minDynamicY = PxMin(
					minDynamicY, dynamic->getGlobalPose().p.y);
				maxDynamicSpeed = PxMax(
					maxDynamicSpeed,
					dynamic->getLinearVelocity().magnitude());
			}
		}
		if(nonFinite)
			break;
	}
	const PxU32 queryHitsAfter = queryDynamicStacks();
	const bool passed =
		actorCount == 19 && dynamicCount == 18 &&
		gPruningStructuresCreated == 2 &&
		gSerializationSucceeded && gSerializedBytes > 0 &&
		queryHitsBefore == 3 && queryHitsAfter == 3 &&
		completedFrames == gHeadlessOptions.frames &&
		minDynamicY > 1.0f && nonFinite == 0 && fetchFailures == 0 &&
		gSolverReadbackMatched && gErrorCallback.getFatalCount() == 0;
	const PxU32 fatalErrors = gErrorCallback.getFatalCount();
	cleanupPhysics(false);
	printf("[AVBD_GATE] schema=1 snippet=SnippetPrunerSerialization "
		"solver=%s case=pruner-roundtrip execution=%s frames=%u "
		"actors=%u dynamicActors=%u pruningStructures=%u "
		"serializationSucceeded=%u serializedBytes=%llu "
		"queryHitsBefore=%u queryHitsAfter=%u completedFrames=%u "
		"minDynamicY=%.9g maxDynamicSpeed=%.9g nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0 "
		"status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, actorCount, dynamicCount,
		gPruningStructuresCreated, gSerializationSucceeded ? 1u : 0u,
		static_cast<unsigned long long>(gSerializedBytes),
		queryHitsBefore, queryHitsAfter,
		completedFrames, minDynamicY, maxDynamicSpeed, nonFinite,
		fetchFailures, fatalErrors, gCleanupComplete,
		passed ? "PASS" : "FAIL",
		passed ? "none" : "pruner_roundtrip_or_dynamics");
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.caseName = "pruner-roundtrip";
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] "
			"snippet=SnippetPrunerSerialization reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();
	static const PxU32 frameCount = 100;
	initPhysics(false);
	for(PxU32 i=0; i<frameCount; i++)
		stepPhysics(false);
	cleanupPhysics(false);
	return 0;
}
