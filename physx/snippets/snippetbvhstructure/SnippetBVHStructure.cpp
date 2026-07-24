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
// This snippet illustrates the usage of PxBVH for PxScene's addActor function.
//
// It creates a large number of small sphere shapes forming a large sphere. Large sphere
// represents an actor and the actor is inserted into the scene with a BVH
// that is precomputed from all the small spheres. When an actor is inserted this
// way the scene queries against this object behave actor centric rather than shape
// centric.
// Each actor that is added with a BVH does not update any of its shape bounds
// within a pruning structure. It does update just the actor bounds and the query then
// goes into actors bounds pruner, then a local query is done against the shapes in the
// actor.
// For a dynamic actor consisting of a large amound of shapes there can be a significant
// performance benefits. During fetch results, there is no need to synchronize all
// shape bounds into scene query system. Also when a new AABB tree is build inside
// scene query system these actors shapes are not contained there.
// ****************************************************************************

#include <ctype.h>
#include <cfloat>
#include "PxPhysicsAPI.h"
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
static Snippets::HeadlessOptions gHeadlessOptions;
static PxRigidDynamic*			gLargeSpheres[10];
static PxU32					gLargeSphereCount = 0;
static PxU32					gBvhCreated = 0;
static PxU32					gTotalShapeBounds = 0;
static bool						gSolverReadbackMatched = false;
static PxU32					gCleanupComplete = 0;

static bool createLargeSphere(const PxTransform& t, PxU32 density, PxReal largeRadius, PxReal radius, bool useAggregate)
{
	PxRigidDynamic* body = gPhysics->createRigidDynamic(t);
	if(!body)
		return false;

	// generate the sphere shapes
	const float gStep = PxPi/float(density);
	const float tStep = 2.0f*PxPi/float(density);
	for(PxU32 i=0; i<density;i++)
	{
		for(PxU32 j=0;j<density;j++)
		{
			const float sinG = PxSin(gStep * i);
			const float cosG = PxCos(gStep * i);
			const float sinT = PxSin(tStep * j);
			const float cosT = PxCos(tStep * j);

			PxTransform localTm(PxVec3(largeRadius*sinG*cosT, largeRadius*sinG*sinT, largeRadius*cosG));
			PxShape* shape = gPhysics->createShape(PxSphereGeometry(radius), *gMaterial);
			shape->setLocalPose(localTm);
			body->attachShape(*shape);
			shape->release();
		}
	}
	PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);

	// get the bounds from the actor, this can be done through a helper function in PhysX extensions
	PxU32 numBounds = 0;
	PxBounds3* bounds = PxRigidActorExt::getRigidActorShapeLocalBoundsList(*body, numBounds);
	gTotalShapeBounds += numBounds;

	printf("Creating BVH structure for large compound actor...\n");

	// setup the PxBVHDesc, it does contain only the PxBounds3 data
	PxBVHDesc bvhDesc;
	bvhDesc.bounds.count = numBounds;
	bvhDesc.bounds.data = bounds;
	bvhDesc.bounds.stride = sizeof(PxBounds3);

	// cook the bvh
	PxBVH* bvh = PxCreateBVH(bvhDesc, gPhysics->getPhysicsInsertionCallback());

	// release the memory allocated within extensions, the bounds are not required anymore
	gAllocator.deallocate(bounds);
	if(!bvh)
	{
		body->release();
		return false;
	}
	gBvhCreated++;

	if(useAggregate)
		printf("Adding actor + BVH structure to aggregate...\n");
	else
		printf("Adding actor + BVH structure to scene...\n");

	// add the actor to the scene and provide the bvh structure (regular path without aggregate usage)
	if(!useAggregate)
		gScene->addActor(*body, bvh);

	// Note that when objects with large amound of shapes are created it is also
	// recommended to create an aggregate from them, see the code below that would replace
	// the gScene->addActor(*body, bvh)
	if(useAggregate)
	{
		PxAggregate* aggregate = gPhysics->createAggregate(1, body->getNbShapes(), false);
		aggregate->addActor(*body, bvh);
		gScene->addAggregate(*aggregate);
	}

	// bvh can be released at this point, the precomputed BVH structure was copied to the SDK pruners.
	bvh->release();
	if(gLargeSphereCount < 10)
		gLargeSpheres[gLargeSphereCount++] = body;
	return true;
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
			gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
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

	for(PxU32 i = 0; i < 10; i++)
	{
		if(!createLargeSphere(
			PxTransform(PxVec3(200.0f*i, .0f, 100.0f)),
			50, 30.0f, 1.0f, false))
			return false;
	}
	return true;
}

void initPhysics(bool /*interactive*/)
{
	initPhysicsInternal(false);
}

static bool stepPhysicsInternal()
{
	if(!gHeadlessOptions.headless)
		printf("Simulating...\n");
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

	printf("SnippetBVH done.\n");
}

void keyPress(unsigned char , const PxTransform& )
{
}

static PxU32 queryLargeSpheres()
{
	PxU32 hits = 0;
	for(PxU32 i=0; i<gLargeSphereCount; ++i)
	{
		const PxTransform pose = gLargeSpheres[i]->getGlobalPose();
		const PxVec3 direction = pose.rotate(PxVec3(0.0f, 0.0f, 1.0f));
		const PxVec3 origin = pose.p - direction * 45.0f;
		PxRaycastBuffer hit;
		if(gScene->raycast(origin, direction, 90.0f, hit) &&
		   hit.hasBlock && hit.block.actor == gLargeSpheres[i])
			hits++;
	}
	return hits;
}

static int runHeadless()
{
	gErrorCallback.reset();
	gCleanupComplete = 0;
	gLargeSphereCount = 0;
	gBvhCreated = 0;
	gTotalShapeBounds = 0;
	for(PxU32 i=0; i<10; ++i)
		gLargeSpheres[i] = NULL;
	if(!initPhysicsInternal(true))
	{
		cleanupPhysics(false);
		return Snippets::eHEADLESS_GATE_FAILED;
	}
	const PxU32 initialDynamicActors =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	PxU32 totalShapes = 0;
	for(PxU32 i=0; i<gLargeSphereCount; ++i)
		totalShapes += gLargeSpheres[i]->getNbShapes();
	const PxU32 queryHitsBefore = queryLargeSpheres();
	PxU32 completedFrames = 0;
	PxU32 nonFinite = 0;
	PxU32 fetchFailures = 0;
	PxReal maxSpeed = 0.0f;
	PxReal maxAngularSpeed = 0.0f;
	PxReal maxDisplacement = 0.0f;
	for(PxU32 frame=0; frame<gHeadlessOptions.frames; ++frame)
	{
		if(!stepPhysicsInternal())
		{
			fetchFailures++;
			break;
		}
		completedFrames++;
		for(PxU32 i=0; i<gLargeSphereCount; ++i)
		{
			PxRigidDynamic* body = gLargeSpheres[i];
			if(!body->getGlobalPose().isFinite() ||
			   !body->getLinearVelocity().isFinite() ||
			   !body->getAngularVelocity().isFinite())
			{
				nonFinite++;
				continue;
			}
			maxSpeed = PxMax(maxSpeed, body->getLinearVelocity().magnitude());
			maxAngularSpeed = PxMax(
				maxAngularSpeed, body->getAngularVelocity().magnitude());
			const PxVec3 initialPosition(200.0f * PxReal(i), 0.0f, 100.0f);
			maxDisplacement = PxMax(
				maxDisplacement,
				(body->getGlobalPose().p - initialPosition).magnitude());
		}
		if(nonFinite)
			break;
	}
	const PxU32 queryHitsAfter = queryLargeSpheres();
	const bool passed =
		gLargeSphereCount == 10 && gBvhCreated == 10 &&
		gTotalShapeBounds == 25000 && totalShapes == 25000 &&
		initialDynamicActors == 10 &&
		queryHitsBefore == 10 && queryHitsAfter == 10 &&
		completedFrames == gHeadlessOptions.frames &&
		maxSpeed > 0.0f && maxSpeed < 50.0f &&
		maxAngularSpeed >= 0.0f && maxAngularSpeed < 50.0f &&
		maxDisplacement > 0.0f && maxDisplacement < 50.0f &&
		nonFinite == 0 && fetchFailures == 0 &&
		gSolverReadbackMatched && gErrorCallback.getFatalCount() == 0;
	const PxU32 fatalErrors = gErrorCallback.getFatalCount();
	cleanupPhysics(false);
	printf("[AVBD_GATE] schema=1 snippet=SnippetBVHStructure solver=%s "
		"case=bvh-compounds execution=%s frames=%u compounds=%u "
		"bvhCreated=%u shapeBounds=%u totalShapes=%u dynamicActors=%u "
		"queryHitsBefore=%u queryHitsAfter=%u completedFrames=%u "
		"maxSpeed=%.9g maxAngularSpeed=%.9g maxDisplacement=%.9g nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0 "
		"status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gLargeSphereCount, gBvhCreated,
		gTotalShapeBounds, totalShapes, initialDynamicActors,
		queryHitsBefore, queryHitsAfter, completedFrames, maxSpeed,
		maxAngularSpeed, maxDisplacement, nonFinite, fetchFailures, fatalErrors,
		gCleanupComplete, passed ? "PASS" : "FAIL",
		passed ? "none" : "bvh_query_or_dynamics");
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 60;
	defaults.caseName = "bvh-compounds";
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] snippet=SnippetBVHStructure "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();
	static const PxU32 frameCount = 50;
	initPhysics(false);
	for(PxU32 i=0; i<frameCount; i++)
		stepPhysics(false);
	cleanupPhysics(false);
	return 0;
}
