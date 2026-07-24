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
// This snippet shows how to use GJK queries to create custom convex geometry.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "geometry/PxGjkQuery.h"
#include "geomutils/PxContactBuffer.h"
#include "extensions/PxCustomGeometryExt.h"

#ifdef RENDER_SNIPPET

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetrender/SnippetRender.h"
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
static PxRigidDynamic* gHeadlessActor = NULL;
static PxRigidStatic* gGroundActor = NULL;
static bool gExtensionsInitialized = false;
static PxArray<PxCustomGeometryExt::BaseConvexCallbacks*> gConvexes;
static PxArray<PxRigidActor*> gActors;
struct RenderMesh;
static PxArray<RenderMesh*> gMeshes;
static Snippets::HeadlessOptions gHeadlessOptions;

struct CustomConvexMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 generateCalls;
	PxU32 generatedContacts;
	PxU32 nonFiniteGeneratedContacts;
	PxU32 callbackCount;
	PxU32 pairCount;
	PxU32 reportedPoints;
	PxU32 nonzeroImpulseCount;
	PxU32 identityErrors;
	PxU32 nonFinite;
	PxReal impulseSum;
	PxReal maxImpulse;
	PxReal minBodyY;
	PxVec3 initialPosition;
	PxVec3 finalPosition;
	PxVec3 finalVelocity;
	PxU32 cleanupComplete;

	CustomConvexMetrics()
	: completedFrames(0), fetchFailures(0), generateCalls(0),
	  generatedContacts(0), nonFiniteGeneratedContacts(0),
	  callbackCount(0), pairCount(0), reportedPoints(0),
	  nonzeroImpulseCount(0), identityErrors(0), nonFinite(0),
	  impulseSum(0.0f), maxImpulse(0.0f), minBodyY(PX_MAX_F32),
	  initialPosition(0.0f), finalPosition(0.0f),
	  finalVelocity(0.0f), cleanupComplete(0)
	{
	}
};

static CustomConvexMetrics gMetrics;

static void recordGeneratedContacts(
	const PxContactBuffer& buffer, PxU32 begin)
{
	for(PxU32 i = begin; i < buffer.count; ++i)
	{
		++gMetrics.generatedContacts;
		const PxContactPoint& point = buffer.contacts[i];
		if(!point.point.isFinite() || !point.normal.isFinite() ||
			!PxIsFinite(point.separation))
			++gMetrics.nonFiniteGeneratedContacts;
	}
}

struct InstrumentedCylinderCallbacks
	: PxCustomGeometryExt::CylinderCallbacks
{
	InstrumentedCylinderCallbacks(
		float height, float radius, int axis, float margin)
	: PxCustomGeometryExt::CylinderCallbacks(
		height, radius, axis, margin)
	{
	}

	virtual bool generateContacts(
		const PxGeometry& geom0, const PxGeometry& geom1,
		const PxTransform& pose0, const PxTransform& pose1,
		const PxReal contactDistance, const PxReal meshContactMargin,
		const PxReal toleranceLength, PxContactBuffer& contactBuffer) const
	{
		++gMetrics.generateCalls;
		const PxU32 begin = contactBuffer.count;
		const bool result =
			PxCustomGeometryExt::BaseConvexCallbacks::generateContacts(
				geom0, geom1, pose0, pose1, contactDistance,
				meshContactMargin, toleranceLength, contactBuffer);
		recordGeneratedContacts(contactBuffer, begin);
		return result;
	}
};

struct InstrumentedConeCallbacks : PxCustomGeometryExt::ConeCallbacks
{
	InstrumentedConeCallbacks(
		float height, float radius, int axis, float margin)
	: PxCustomGeometryExt::ConeCallbacks(height, radius, axis, margin)
	{
	}

	virtual bool generateContacts(
		const PxGeometry& geom0, const PxGeometry& geom1,
		const PxTransform& pose0, const PxTransform& pose1,
		const PxReal contactDistance, const PxReal meshContactMargin,
		const PxReal toleranceLength, PxContactBuffer& contactBuffer) const
	{
		++gMetrics.generateCalls;
		const PxU32 begin = contactBuffer.count;
		const bool result =
			PxCustomGeometryExt::BaseConvexCallbacks::generateContacts(
				geom0, geom1, pose0, pose1, contactDistance,
				meshContactMargin, toleranceLength, contactBuffer);
		recordGeneratedContacts(contactBuffer, begin);
		return result;
	}
};

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), name);
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 180;
	defaults.caseName = "cylinder-drop";
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
	if(!isHeadlessCase("cylinder-drop") &&
		!isHeadlessCase("cone-impact"))
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
	PxFilterObjectAttributes, PxFilterData,
	PxFilterObjectAttributes, PxFilterData,
	PxPairFlags& pairFlags, const void*, PxU32)
{
	pairFlags = PxPairFlag::eCONTACT_DEFAULT |
		PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
		PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

class ContactReportCallback : public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo*, PxU32) {}
	void onWake(PxActor**, PxU32) {}
	void onSleep(PxActor**, PxU32) {}
	void onTrigger(PxTriggerPair*, PxU32) {}
	void onAdvance(
		const PxRigidBody*const*, const PxTransform*, const PxU32) {}
	void onContact(const PxContactPairHeader& pairHeader,
		const PxContactPair* pairs, PxU32 nbPairs)
	{
		if(!gHeadlessOptions.headless)
			return;
		++gMetrics.callbackCount;
		gMetrics.pairCount += nbPairs;
		const bool identityValid =
			(pairHeader.actors[0] == gHeadlessActor &&
			 pairHeader.actors[1] == gGroundActor) ||
			(pairHeader.actors[1] == gHeadlessActor &&
			 pairHeader.actors[0] == gGroundActor);
		if(!identityValid)
			++gMetrics.identityErrors;
		PxArray<PxContactPairPoint> points;
		for(PxU32 i = 0; i < nbPairs; ++i)
		{
			const PxU32 count = pairs[i].contactCount;
			if(!count)
				continue;
			points.resize(count);
			const PxU32 extracted =
				pairs[i].extractContacts(points.begin(), count);
			for(PxU32 j = 0; j < extracted; ++j)
			{
				const PxContactPairPoint& point = points[j];
				const PxReal impulse = point.impulse.magnitude();
				++gMetrics.reportedPoints;
				gMetrics.impulseSum += impulse;
				gMetrics.maxImpulse = PxMax(gMetrics.maxImpulse, impulse);
				if(impulse > 1e-5f)
					++gMetrics.nonzeroImpulseCount;
				if(!point.position.isFinite() || !point.normal.isFinite() ||
					!point.impulse.isFinite() ||
					!PxIsFinite(point.separation))
					++gMetrics.nonFinite;
			}
		}
	}
};

static ContactReportCallback gContactReportCallback;

RenderMesh* createRenderCylinder(float radius, float height, float margin);
RenderMesh* createRenderCone(float height, float radius, float margin);
void destroyRenderMesh(RenderMesh* mesh);
void renderMesh(const RenderMesh& mesh, const PxTransform& pose, bool sleeping);
void renderRaycast(const PxVec3& origin, const PxVec3& unitDir, float maxDist, const PxRaycastHit* hit);
void renderSweepBox(const PxVec3& origin, const PxVec3& unitDir, float maxDist, const PxVec3& halfExtents, const PxSweepHit* hit);
void renderOverlapBox(const PxVec3& origin, const PxVec3& halfExtents, bool hit);

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity = PxVec3(0), PxReal density = 1.0f)
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, density);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void createCylinderActor(float height, float radius, float margin, const PxTransform& pose)
{
	PxCustomGeometryExt::CylinderCallbacks* cylinder = new PxCustomGeometryExt::CylinderCallbacks(height, radius, 0, margin);
	gConvexes.pushBack(cylinder);

	PxRigidDynamic* actor = gPhysics->createRigidDynamic(pose);
	actor->setActorFlag(PxActorFlag::eVISUALIZATION, true);

	PxShape* shape = PxRigidActorExt::createExclusiveShape(*actor, PxCustomGeometry(*cylinder), *gMaterial);
	shape->setFlag(PxShapeFlag::eVISUALIZATION, true);
	PxRigidBodyExt::updateMassAndInertia(*actor, 100);
	gScene->addActor(*actor);
	gActors.pushBack(actor);

	RenderMesh* mesh = createRenderCylinder(height, radius, margin);
	gMeshes.pushBack(mesh);
}

static void createConeActor(float height, float radius, float margin, const PxTransform& pose)
{
	PxCustomGeometryExt::ConeCallbacks* cone = new PxCustomGeometryExt::ConeCallbacks(height, radius, 0, margin);
	gConvexes.pushBack(cone);

	PxRigidDynamic* actor = gPhysics->createRigidDynamic(pose);
	actor->setActorFlag(PxActorFlag::eVISUALIZATION, true);

	PxShape* shape = PxRigidActorExt::createExclusiveShape(*actor, PxCustomGeometry(*cone), *gMaterial);
	shape->setFlag(PxShapeFlag::eVISUALIZATION, true);
	PxRigidBodyExt::updateMassAndInertia(*actor, 100);
	gScene->addActor(*actor);
	gActors.pushBack(actor);

	RenderMesh* mesh = createRenderCone(height, radius, margin);
	gMeshes.pushBack(mesh);
}

void initPhysics(bool /*interactive*/)
{
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
		gHeadlessOptions.dispatcherThreads : 2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = gHeadlessOptions.headless ?
		contactReportFilterShader : PxDefaultSimulationFilterShader;
	sceneDesc.simulationEventCallback = gHeadlessOptions.headless ?
		&gContactReportCallback : NULL;
	sceneDesc.solverType = gHeadlessOptions.solverType;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gHeadlessOptions.headless)
	{
		gScene->setVisualizationParameter(
			PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
		gScene->setVisualizationParameter(
			PxVisualizationParameter::eSCALE, 1.0f);
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

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	if(gHeadlessOptions.headless)
	{
		PxCustomGeometryExt::BaseConvexCallbacks* callbacks = NULL;
		PxVec3 position(0.0f, 6.0f, 0.0f);
		PxVec3 velocity(0.0f);
		if(isHeadlessCase("cylinder-drop"))
			callbacks = new InstrumentedCylinderCallbacks(
				2.0f, 1.0f, 1, 0.05f);
		else
		{
			callbacks = new InstrumentedConeCallbacks(
				2.0f, 1.0f, 1, 0.05f);
			position.y = 8.0f;
			velocity.y = -25.0f;
		}
		gConvexes.pushBack(callbacks);
		gHeadlessActor = gPhysics->createRigidDynamic(
			PxTransform(position));
		PxRigidActorExt::createExclusiveShape(
			*gHeadlessActor, PxCustomGeometry(*callbacks), *gMaterial);
		PxRigidBodyExt::updateMassAndInertia(*gHeadlessActor, 10.0f);
		gHeadlessActor->setLinearVelocity(velocity);
		gHeadlessActor->setLinearDamping(0.0f);
		gHeadlessActor->setAngularDamping(0.0f);
		gHeadlessActor->setSleepThreshold(0.0f);
		gScene->addActor(*gHeadlessActor);
		gActors.pushBack(gHeadlessActor);
		gMetrics.initialPosition = position;
		gMetrics.finalPosition = position;
		gMetrics.finalVelocity = velocity;
	}
	else
	{
		// Some custom convexes
		float heights[] = { 1.0f, 1.25f, 1.5f, 1.75f };
		float radiuss[] = { 0.3f, 0.35f, 0.4f, 0.45f };
		float margins[] = { 0.0f, 0.05f, 0.1f, 0.15f };
		for (int i = 0; i < 50; ++i)
		{
			float height = heights[rand() % (sizeof(heights) / sizeof(heights[0]))];
			float radius = radiuss[rand() % (sizeof(radiuss) / sizeof(radiuss[0]))];
			float margin = margins[rand() % (sizeof(margins) / sizeof(margins[0]))];
			createCylinderActor(height, radius, margin,
				PxTransform(PxVec3(-2.0f, 2.0f + i * 2, 2.0f),
					PxQuat(PX_PIDIV2, PxVec3(0.0f, 0.0f, 1.0f))));
		}
		for (int i = 0; i < 50; ++i)
		{
			float height = heights[rand() % (sizeof(heights) / sizeof(heights[0]))];
			float radius = radiuss[rand() % (sizeof(radiuss) / sizeof(radiuss[0]))];
			float margin = margins[rand() % (sizeof(margins) / sizeof(margins[0]))];
			createConeActor(height, radius, margin,
				PxTransform(PxVec3(2.0f, 2.0f + i * 2, -2.0f),
					PxQuat(PX_PIDIV2, PxVec3(0, 0, 1))));
		}
	}

	// Ground plane
	gGroundActor = gPhysics->createRigidStatic(
		PxTransform(PxQuat(PX_PIDIV2, PxVec3(0.0f, 0.0f, 1.0f))));
	PxRigidActorExt::createExclusiveShape(
		*gGroundActor, PxPlaneGeometry(), *gMaterial);
	gScene->addActor(*gGroundActor);
}

void debugRender()
{
	for (int i = 0; i < int(gConvexes.size()); ++i)
	{
		PxRigidActor* actor = gActors[i];
		RenderMesh* mesh = gMeshes[i];
		renderMesh(*mesh, actor->getGlobalPose(), !actor->is<PxRigidDynamic>() || actor->is<PxRigidDynamic>()->isSleeping());
	}

	int count = 20;
	for (int i = 0; i < count; ++i)
	{
		float x = -count / 2.0f;
		PxVec3 origin(x + i, 0.5f, x);
		PxVec3 unitDir(0, 0, 1);
		float maxDist = (float)count;
		PxRaycastBuffer buffer;
		gScene->raycast(origin, unitDir, maxDist, buffer);
		renderRaycast(origin, unitDir, maxDist, buffer.hasBlock ? &buffer.block : nullptr);
	}
	for (int i = 0; i < count; ++i)
	{
		float x = -count / 2.0f;
		PxVec3 origin(x, 0.5f, x + i);
		PxVec3 unitDir(1, 0, 0);
		float maxDist = (float)count;
		PxVec3 halfExtents(0.2f, 0.1f, 0.4f);
		PxSweepBuffer buffer;
		gScene->sweep(PxBoxGeometry(halfExtents), PxTransform(origin), unitDir, maxDist, buffer);
		renderSweepBox(origin, unitDir, maxDist, halfExtents, buffer.hasBlock ? &buffer.block : nullptr);
	}
	for (int i = 0; i < count; ++i)
	{
		float x = -count / 2.0f;
		for (int j = 0; j < count; ++j)
		{
			PxVec3 origin(x + i, 0.0f, x + j);
			PxVec3 halfExtents(0.4f, 0.1f, 0.4f);
			PxOverlapBuffer buffer;
			gScene->overlap(PxBoxGeometry(halfExtents), PxTransform(origin), buffer, PxQueryFilterData(PxQueryFlag::eANY_HIT | PxQueryFlag::eDYNAMIC));
			renderOverlapBox(origin, halfExtents, buffer.hasAnyHits());
		}
	}
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f / 60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gHeadlessActor)
		{
			const PxTransform pose = gHeadlessActor->getGlobalPose();
			const PxVec3 velocity = gHeadlessActor->getLinearVelocity();
			gMetrics.finalPosition = pose.p;
			gMetrics.finalVelocity = velocity;
			gMetrics.minBodyY = PxMin(gMetrics.minBodyY, pose.p.y);
			if(!pose.isFinite() || !velocity.isFinite())
				++gMetrics.nonFinite;
		}
		++gMetrics.completedFrames;
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	while (!gActors.empty())
	{
		PX_RELEASE(gActors.back());
		gActors.popBack();
	}
	gActors.reset();
	gHeadlessActor = NULL;
	PX_RELEASE(gGroundActor);

	while (!gConvexes.empty())
	{
		delete gConvexes.back();
		gConvexes.popBack();
	}
	gConvexes.reset();

	while (!gMeshes.empty())
	{
		destroyRenderMesh(gMeshes.back());
		gMeshes.popBack();
	}
	gMeshes.reset();

	PX_RELEASE(gScene);
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
		!gScene && !gMaterial && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation && !gHeadlessActor && !gGroundActor &&
		!gExtensionsInitialized && gActors.empty() && gConvexes.empty() ?
		1u : 0u;

	printf("SnippetCustomConvex done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch (toupper(key))
	{
	case ' ':	createDynamic(camera, PxSphereGeometry(1.0f), camera.rotate(PxVec3(0, 0, -1)) * 100, 3.0f);	break;
	}
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig("SnippetCustomConvex", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gHeadlessActor && gGroundActor &&
		gConvexes.size() == 1;
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
		gMetrics.nonFiniteGeneratedContacts != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.generateCalls == 0 ||
		gMetrics.generatedContacts == 0)
	{
		passed = false;
		reason = "missing_generated_contacts";
	}
	else if(gMetrics.callbackCount == 0 || gMetrics.pairCount == 0 ||
		gMetrics.reportedPoints == 0)
	{
		passed = false;
		reason = "missing_contact_report";
	}
	else if(gMetrics.identityErrors != 0)
	{
		passed = false;
		reason = "contact_identity_mismatch";
	}
	else if(gMetrics.nonzeroImpulseCount == 0 ||
		gMetrics.maxImpulse <= 1e-5f)
	{
		passed = false;
		reason = "generated_contact_not_consumed";
	}
	else if(gMetrics.minBodyY < -0.2f ||
		gMetrics.finalPosition.y < 0.4f)
	{
		passed = false;
		reason = "custom_convex_fell_through";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetCustomConvex solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED generateCalls=%u generatedContacts=%u "
		"callbackCount=%u pairCount=%u reportedPoints=%u "
		"nonzeroImpulseCount=%u identityErrors=%u impulseSum=%.9g "
		"maxImpulse=%.9g minBodyY=%.9g initialX=%.9g initialY=%.9g "
		"initialZ=%.9g finalX=%.9g finalY=%.9g finalZ=%.9g "
		"finalVx=%.9g finalVy=%.9g finalVz=%.9g nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, gMetrics.generateCalls,
		gMetrics.generatedContacts, gMetrics.callbackCount,
		gMetrics.pairCount, gMetrics.reportedPoints,
		gMetrics.nonzeroImpulseCount, gMetrics.identityErrors,
		double(gMetrics.impulseSum), double(gMetrics.maxImpulse),
		double(gMetrics.minBodyY), double(gMetrics.initialPosition.x),
		double(gMetrics.initialPosition.y),
		double(gMetrics.initialPosition.z),
		double(gMetrics.finalPosition.x),
		double(gMetrics.finalPosition.y),
		double(gMetrics.finalPosition.z),
		double(gMetrics.finalVelocity.x),
		double(gMetrics.finalVelocity.y),
		double(gMetrics.finalVelocity.z), gMetrics.nonFinite,
		gMetrics.fetchFailures, gErrorCallback.getFatalCount(),
		gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char* const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomConvex "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomConvex "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

	extern void renderLoop();
	renderLoop();

	return 0;
}

#else
int snippetMain(int, const char* const*)
{
	return 0;
}

#endif
