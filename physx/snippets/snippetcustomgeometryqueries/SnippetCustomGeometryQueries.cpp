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
// This snippet shows how to implement custom geometry query callbacks, using
// PhysX geometry queries.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"

// This is a render-built snippet. Headless execution is routed before the
// render loop so the query coverage remains available without a GL window.
#ifdef RENDER_SNIPPET

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetrender/SnippetRender.h"

#include <cstdio>
#include <string>

using namespace physx;

void renderRaycast(const PxVec3& origin, const PxVec3& unitDir,
	float maxDist, const PxRaycastHit* hit);
void renderSweepBox(const PxVec3& origin, const PxVec3& unitDir,
	float maxDist, const PxVec3& halfExtents, const PxSweepHit* hit);
void renderOverlapBox(const PxVec3& origin, const PxVec3& halfExtents,
	bool hit);

struct CustomGeometryQueryMetrics
{
	PxU32 raycastCallbackCalls;
	PxU32 raycastCallbackHits;
	PxU32 sweepCallbackCalls;
	PxU32 sweepCallbackHits;
	PxU32 overlapCallbackCalls;
	PxU32 overlapCallbackHits;
	PxU32 raycastHitQueries;
	PxU32 raycastMissQueries;
	PxU32 sweepHitQueries;
	PxU32 sweepMissQueries;
	PxU32 overlapHitQueries;
	PxU32 overlapMissQueries;
	PxU32 negativeControlFailures;
	PxU32 queryIdentityErrors;
	PxU32 queryValueErrors;
	PxU32 solverQueryHits;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 cleanupComplete;
	PxReal minSolverY;
	PxReal maxSolverSpeed;
	PxVec3 initialSolverPosition;
	PxVec3 finalSolverPosition;
	PxVec3 finalSolverVelocity;

	CustomGeometryQueryMetrics()
	: raycastCallbackCalls(0), raycastCallbackHits(0),
	  sweepCallbackCalls(0), sweepCallbackHits(0),
	  overlapCallbackCalls(0), overlapCallbackHits(0),
	  raycastHitQueries(0), raycastMissQueries(0),
	  sweepHitQueries(0), sweepMissQueries(0),
	  overlapHitQueries(0), overlapMissQueries(0),
	  negativeControlFailures(0), queryIdentityErrors(0),
	  queryValueErrors(0), solverQueryHits(0), completedFrames(0),
	  fetchFailures(0), nonFinite(0), cleanupComplete(0),
	  minSolverY(PX_MAX_F32), maxSolverSpeed(0.0f),
	  initialSolverPosition(0.0f), finalSolverPosition(0.0f),
	  finalSolverVelocity(0.0f)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static CustomGeometryQueryMetrics gMetrics;

/*
	Two crossed bars.
*/
struct BarCrosss : PxCustomGeometry::Callbacks
{
	PxVec3 barExtents;

	DECLARE_CUSTOM_GEOMETRY_TYPE

	BarCrosss() : barExtents(27, 9, 3) {}

	virtual PxBounds3 getLocalBounds(const PxGeometry&) const
	{
		return PxBounds3(
			-PxVec3(barExtents.x * 0.5f, barExtents.y * 0.5f,
				barExtents.x * 0.5f),
			PxVec3(barExtents.x * 0.5f, barExtents.y * 0.5f,
				barExtents.x * 0.5f));
	}

	virtual bool generateContacts(const PxGeometry&, const PxGeometry&,
		const PxTransform&, const PxTransform&, const PxReal, const PxReal,
		const PxReal, PxContactBuffer&) const
	{
		return false;
	}

	virtual PxU32 raycast(const PxVec3& origin, const PxVec3& unitDir,
		const PxGeometry&, const PxTransform& pose, PxReal maxDist,
		PxHitFlags hitFlags, PxU32 maxHits, PxGeomRaycastHit* rayHits,
		PxU32, PxRaycastThreadContext*) const
	{
		++gMetrics.raycastCallbackCalls;
		if(!maxHits || !rayHits)
			return 0;

		const PxBoxGeometry barGeom(barExtents * 0.5f);
		PxGeomRaycastHit hits[2];
		PxTransform barPose = pose;
		const PxU32 count0 = PxGeometryQuery::raycast(
			origin, unitDir, barGeom, barPose, maxDist, hitFlags, 1,
			hits + 0);
		barPose = pose.transform(
			PxTransform(PxQuat(PX_PIDIV2, PxVec3(0, 1, 0))));
		const PxU32 count1 = PxGeometryQuery::raycast(
			origin, unitDir, barGeom, barPose, maxDist, hitFlags, 1,
			hits + 1);
		if(!count0 && !count1)
			return 0;

		rayHits[0] = !count1 || (count0 && hits[0].distance < hits[1].distance)
			? hits[0] : hits[1];
		++gMetrics.raycastCallbackHits;
		return 1;
	}

	virtual bool overlap(const PxGeometry&, const PxTransform& pose0,
		const PxGeometry& geom1, const PxTransform& pose1,
		PxOverlapThreadContext*) const
	{
		++gMetrics.overlapCallbackCalls;
		const PxBoxGeometry barGeom(barExtents * 0.5f);
		PxTransform barPose = pose0;
		bool hit = PxGeometryQuery::overlap(
			barGeom, barPose, geom1, pose1, PxGeometryQueryFlags(0));
		if(!hit)
		{
			barPose = pose0.transform(
				PxTransform(PxQuat(PX_PIDIV2, PxVec3(0, 1, 0))));
			hit = PxGeometryQuery::overlap(
				barGeom, barPose, geom1, pose1,
				PxGeometryQueryFlags(0));
		}
		if(hit)
			++gMetrics.overlapCallbackHits;
		return hit;
	}

	virtual bool sweep(const PxVec3& unitDir, const PxReal maxDist,
		const PxGeometry&, const PxTransform& pose0,
		const PxGeometry& geom1, const PxTransform& pose1,
		PxGeomSweepHit& sweepHit, PxHitFlags hitFlags,
		const PxReal inflation, PxSweepThreadContext*) const
	{
		++gMetrics.sweepCallbackCalls;
		const PxBoxGeometry barGeom(barExtents * 0.5f);
		PxGeomSweepHit hits[2];
		PxTransform barPose = pose0;
		const bool hit0 = PxGeometryQuery::sweep(
			unitDir, maxDist, geom1, pose1, barGeom, barPose, hits[0],
			hitFlags, inflation);
		barPose = pose0.transform(
			PxTransform(PxQuat(PX_PIDIV2, PxVec3(0, 1, 0))));
		const bool hit1 = PxGeometryQuery::sweep(
			unitDir, maxDist, geom1, pose1, barGeom, barPose, hits[1],
			hitFlags, inflation);
		if(!hit0 && !hit1)
			return false;

		sweepHit = !hit1 || (hit0 && hits[0].distance < hits[1].distance)
			? hits[0] : hits[1];
		++gMetrics.sweepCallbackHits;
		return true;
	}

	virtual void visualize(const PxGeometry&, PxRenderOutput&,
		const PxTransform&, const PxBounds3&) const {}
	virtual void computeMassProperties(const PxGeometry&,
		PxMassProperties&) const {}
	virtual bool usePersistentContactManifold(const PxGeometry&,
		PxReal&) const { return false; }
};

IMPLEMENT_CUSTOM_GEOMETRY_TYPE(BarCrosss)

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static PxPvd* gPvd = NULL;
static PxRigidDynamic* gActor = NULL;
static PxShape* gCustomShape = NULL;
static PxRigidDynamic* gSolverActor = NULL;
static PxRigidStatic* gSolverSupport = NULL;

static BarCrosss gBarCrosss;
static PxReal gTime = 0;

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
	defaults.caseName = "all-queries";
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
	if(!isHeadlessCase("all-queries"))
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

static PxRigidDynamic* createDynamic(const PxTransform& transform,
	const PxGeometry& geometry, const PxVec3& velocity = PxVec3(0),
	PxReal density = 1.0f)
{
	PxRigidDynamic* dynamic = PxCreateDynamic(
		*gPhysics, transform, geometry, *gMaterial, density);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

void initPhysics(bool /*interactive*/)
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

	gPhysics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f * 3, 0.0f);
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

	// Create the official moving bar-cross actor.
	gActor = gPhysics->createRigidDynamic(PxTransform(
		PxVec3(0, gBarCrosss.barExtents.y * 0.5f, 0)));
	gActor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	gCustomShape = PxRigidActorExt::createExclusiveShape(
		*gActor, PxCustomGeometry(gBarCrosss), *gMaterial);
	gScene->addActor(*gActor);

	if(gHeadlessOptions.headless)
	{
		// Keep the solver witness far from the custom-query region. The
		// custom actor has no contact generation, so a primitive pair proves
		// the selected solver is actually consuming gravity and contact.
		const PxVec3 witnessOrigin(100.0f, 0.0f, 100.0f);
		gSolverSupport = gPhysics->createRigidStatic(
			PxTransform(witnessOrigin));
		PxRigidActorExt::createExclusiveShape(
			*gSolverSupport, PxBoxGeometry(3.0f, 0.5f, 3.0f),
			*gMaterial);
		gScene->addActor(*gSolverSupport);

		const PxVec3 solverPosition =
			witnessOrigin + PxVec3(0.0f, 5.0f, 0.0f);
		gSolverActor = createDynamic(
			PxTransform(solverPosition), PxSphereGeometry(0.5f),
			PxVec3(0.0f), 1.0f);
		gSolverActor->setLinearDamping(0.0f);
		gSolverActor->setAngularDamping(0.0f);
		gSolverActor->setSleepThreshold(0.0f);
		gMetrics.initialSolverPosition = solverPosition;
		gMetrics.finalSolverPosition = solverPosition;
	}
}

static bool isExpectedCustomHit(const PxActorShape& hit)
{
	return hit.actor == gActor && hit.shape == gCustomShape;
}

static void recordLocationHit(const PxLocationHit& hit, PxReal maxDist)
{
	if(!hit.position.isFinite() || !hit.normal.isFinite() ||
		!PxIsFinite(hit.distance) || hit.distance < 0.0f ||
		hit.distance > maxDist)
		++gMetrics.queryValueErrors;
}

static void runCustomQuerySuite()
{
	const PxTransform pose = gActor->getGlobalPose();
	const PxReal maxDist = gBarCrosss.barExtents.x * 2.0f;
	const PxReal missHeight = gBarCrosss.barExtents.y;
	const PxQueryFilterData dynamicOnly(PxQueryFlag::eDYNAMIC);

	// Raycast through the first bar in the actor's moving local frame.
	{
		const PxVec3 origin = pose.transform(
			PxVec3(gBarCrosss.barExtents.x, 0.0f, 0.0f));
		const PxVec3 unitDir = pose.rotate(PxVec3(-1.0f, 0.0f, 0.0f));
		PxRaycastBuffer buffer;
		if(gScene->raycast(
			origin, unitDir, maxDist, buffer, PxHitFlag::eDEFAULT,
			dynamicOnly) && buffer.hasBlock)
		{
			++gMetrics.raycastHitQueries;
			if(!isExpectedCustomHit(buffer.block))
				++gMetrics.queryIdentityErrors;
			recordLocationHit(buffer.block, maxDist);
		}
	}

	// Same trajectory above both bars is the raycast negative control.
	{
		const PxVec3 origin = pose.transform(PxVec3(
			gBarCrosss.barExtents.x, missHeight, 0.0f));
		const PxVec3 unitDir = pose.rotate(PxVec3(-1.0f, 0.0f, 0.0f));
		PxRaycastBuffer buffer;
		if(gScene->raycast(
			origin, unitDir, maxDist, buffer, PxHitFlag::eDEFAULT,
			dynamicOnly))
			++gMetrics.negativeControlFailures;
		else
			++gMetrics.raycastMissQueries;
	}

	const PxVec3 sweepHalfExtents(0.75f, 0.5f, 0.75f);
	const PxVec3 sweepDir = pose.rotate(PxVec3(0.0f, 0.0f, -1.0f));

	// Sweep a primitive through the second bar.
	{
		const PxVec3 origin = pose.transform(
			PxVec3(0.0f, 0.0f, gBarCrosss.barExtents.x));
		PxSweepBuffer buffer;
		if(gScene->sweep(
			PxBoxGeometry(sweepHalfExtents), PxTransform(origin, pose.q),
			sweepDir, maxDist, buffer, PxHitFlag::eDEFAULT, dynamicOnly) &&
			buffer.hasBlock)
		{
			++gMetrics.sweepHitQueries;
			if(!isExpectedCustomHit(buffer.block))
				++gMetrics.queryIdentityErrors;
			recordLocationHit(buffer.block, maxDist);
		}
	}

	// The matching sweep above the bars must miss.
	{
		const PxVec3 origin = pose.transform(PxVec3(
			0.0f, missHeight, gBarCrosss.barExtents.x));
		PxSweepBuffer buffer;
		if(gScene->sweep(
			PxBoxGeometry(sweepHalfExtents), PxTransform(origin, pose.q),
			sweepDir, maxDist, buffer, PxHitFlag::eDEFAULT, dynamicOnly))
			++gMetrics.negativeControlFailures;
		else
			++gMetrics.sweepMissQueries;
	}

	const PxBoxGeometry overlapGeometry(PxVec3(0.75f));

	// A small box at the cross center must overlap the custom geometry.
	{
		PxOverlapBuffer buffer;
		if(gScene->overlap(
			overlapGeometry, pose, buffer,
			PxQueryFilterData(
				PxQueryFlag::eANY_HIT | PxQueryFlag::eDYNAMIC)) &&
			buffer.hasAnyHits())
		{
			++gMetrics.overlapHitQueries;
			if(!isExpectedCustomHit(buffer.getAnyHit(0)))
				++gMetrics.queryIdentityErrors;
		}
	}

	// A box above the bars is the overlap negative control.
	{
		PxOverlapBuffer buffer;
		const PxTransform missPose(
			pose.transform(PxVec3(0.0f, missHeight, 0.0f)), pose.q);
		if(gScene->overlap(
			overlapGeometry, missPose, buffer,
			PxQueryFilterData(
				PxQueryFlag::eANY_HIT | PxQueryFlag::eDYNAMIC)))
			++gMetrics.negativeControlFailures;
		else
			++gMetrics.overlapMissQueries;
	}
}

static void recordSolverWitness()
{
	const PxTransform pose = gSolverActor->getGlobalPose();
	const PxVec3 velocity = gSolverActor->getLinearVelocity();
	gMetrics.finalSolverPosition = pose.p;
	gMetrics.finalSolverVelocity = velocity;
	gMetrics.minSolverY = PxMin(gMetrics.minSolverY, pose.p.y);
	gMetrics.maxSolverSpeed = PxMax(
		gMetrics.maxSolverSpeed, velocity.magnitude());
	if(!pose.isFinite() || !velocity.isFinite())
		++gMetrics.nonFinite;

	PxOverlapBuffer buffer;
	if(gScene->overlap(
		PxSphereGeometry(0.1f), pose, buffer,
		PxQueryFilterData(
			PxQueryFlag::eANY_HIT | PxQueryFlag::eDYNAMIC)) &&
		buffer.hasAnyHits() && buffer.getAnyHit(0).actor == gSolverActor)
		++gMetrics.solverQueryHits;
	else
		++gMetrics.queryIdentityErrors;
}

void debugRender()
{
	PxGeometryHolder geom;
	geom.storeAny(PxBoxGeometry(gBarCrosss.barExtents * 0.5f));
	PxTransform pose = gActor->getGlobalPose();
	Snippets::renderGeoms(1, &geom, &pose, false, PxVec3(0.7f));
	pose = pose.transform(
		PxTransform(PxQuat(PX_PIDIV2, PxVec3(0, 1, 0))));
	Snippets::renderGeoms(1, &geom, &pose, false, PxVec3(0.7f));

	// Raycast
	{
		const PxVec3 origin(
			(gBarCrosss.barExtents.x + 10) * 0.5f, 0, 0);
		const PxVec3 unitDir(-1, 0, 0);
		const float maxDist = gBarCrosss.barExtents.x + 20;
		PxRaycastBuffer buffer;
		gScene->raycast(origin, unitDir, maxDist, buffer);
		renderRaycast(origin, unitDir, maxDist,
			buffer.hasBlock ? &buffer.block : nullptr);
	}

	// Sweep
	{
		const PxVec3 origin(
			0, 0, (gBarCrosss.barExtents.x + 10) * 0.5f);
		const PxVec3 unitDir(0, 0, -1);
		const float maxDist = gBarCrosss.barExtents.x + 20;
		const PxVec3 halfExtents(1.5f, 0.5f, 1.0f);
		PxSweepBuffer buffer;
		gScene->sweep(
			PxBoxGeometry(halfExtents), PxTransform(origin), unitDir,
			maxDist, buffer);
		renderSweepBox(origin, unitDir, maxDist, halfExtents,
			buffer.hasBlock ? &buffer.block : nullptr);
	}

	// Overlap
	{
		const PxVec3 origin(
			gBarCrosss.barExtents.x * -0.4f, 0,
			gBarCrosss.barExtents.x * -0.4f);
		const PxVec3 halfExtents(
			gBarCrosss.barExtents.z * 1.5f,
			gBarCrosss.barExtents.y * 1.1f,
			gBarCrosss.barExtents.z * 1.5f);
		PxOverlapBuffer buffer;
		gScene->overlap(
			PxBoxGeometry(halfExtents), PxTransform(origin), buffer,
			PxQueryFilterData(
				PxQueryFlag::eANY_HIT | PxQueryFlag::eDYNAMIC));
		renderOverlapBox(origin, halfExtents, buffer.hasAnyHits());
	}
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = gHeadlessOptions.headless
		? gHeadlessOptions.dt : 1.0f / 60.0f;
	gTime += dt;
	gActor->setKinematicTarget(PxTransform(
		PxQuat(gTime * 0.3f, PxVec3(0, 1, 0))));

	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		runCustomQuerySuite();
		recordSolverWitness();
		++gMetrics.completedFrames;
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	gCustomShape = NULL;
	PX_RELEASE(gSolverActor);
	PX_RELEASE(gSolverSupport);
	PX_RELEASE(gActor);
	PX_RELEASE(gScene);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);

	gMetrics.cleanupComplete =
		!gFoundation && !gPhysics && !gDispatcher && !gScene &&
		!gMaterial && !gPvd && !gActor && !gCustomShape &&
		!gSolverActor && !gSolverSupport ? 1u : 0u;

	std::printf("SnippetCustomGeometryQueries done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case ' ':
		createDynamic(
			camera, PxSphereGeometry(3.0f),
			camera.rotate(PxVec3(0, 0, -1)) * 200, 3.0f);
		break;
	}
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig(
		"SnippetCustomGeometryQueries", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gScene && gMaterial &&
		gActor && gCustomShape && gSolverActor && gSolverSupport;
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
	}

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
	else if(gMetrics.raycastCallbackCalls == 0 ||
		gMetrics.raycastCallbackHits == 0 ||
		gMetrics.sweepCallbackCalls == 0 ||
		gMetrics.sweepCallbackHits == 0 ||
		gMetrics.overlapCallbackCalls == 0 ||
		gMetrics.overlapCallbackHits == 0)
	{
		passed = false;
		reason = "missing_custom_query_callback";
	}
	else if(gMetrics.raycastHitQueries != gHeadlessOptions.frames ||
		gMetrics.raycastMissQueries != gHeadlessOptions.frames ||
		gMetrics.sweepHitQueries != gHeadlessOptions.frames ||
		gMetrics.sweepMissQueries != gHeadlessOptions.frames ||
		gMetrics.overlapHitQueries != gHeadlessOptions.frames ||
		gMetrics.overlapMissQueries != gHeadlessOptions.frames)
	{
		passed = false;
		reason = "incomplete_query_matrix";
	}
	else if(gMetrics.negativeControlFailures != 0)
	{
		passed = false;
		reason = "query_negative_control_failed";
	}
	else if(gMetrics.queryIdentityErrors != 0 ||
		gMetrics.queryValueErrors != 0)
	{
		passed = false;
		reason = "query_result_invalid";
	}
	else if(gMetrics.solverQueryHits != gHeadlessOptions.frames)
	{
		passed = false;
		reason = "solver_actor_query_missing";
	}
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.minSolverY < 0.4f ||
		gMetrics.finalSolverPosition.y < 0.7f ||
		gMetrics.finalSolverPosition.y > 1.5f ||
		gMetrics.maxSolverSpeed <= 0.1f ||
		gMetrics.maxSolverSpeed > 50.0f)
	{
		passed = false;
		reason = "solver_witness_invalid";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetCustomGeometryQueries "
		"solver=%s case=%s execution=%s frames=%u completedFrames=%u "
		"status=%s reason=%s validation=GATED "
		"raycastCallbackCalls=%u raycastCallbackHits=%u "
		"sweepCallbackCalls=%u sweepCallbackHits=%u "
		"overlapCallbackCalls=%u overlapCallbackHits=%u "
		"raycastHitQueries=%u raycastMissQueries=%u "
		"sweepHitQueries=%u sweepMissQueries=%u "
		"overlapHitQueries=%u overlapMissQueries=%u "
		"negativeControlFailures=%u queryIdentityErrors=%u "
		"queryValueErrors=%u solverQueryHits=%u minSolverY=%.9g "
		"maxSolverSpeed=%.9g initialSolverY=%.9g finalSolverY=%.9g "
		"finalSolverVy=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		gMetrics.raycastCallbackCalls, gMetrics.raycastCallbackHits,
		gMetrics.sweepCallbackCalls, gMetrics.sweepCallbackHits,
		gMetrics.overlapCallbackCalls, gMetrics.overlapCallbackHits,
		gMetrics.raycastHitQueries, gMetrics.raycastMissQueries,
		gMetrics.sweepHitQueries, gMetrics.sweepMissQueries,
		gMetrics.overlapHitQueries, gMetrics.overlapMissQueries,
		gMetrics.negativeControlFailures, gMetrics.queryIdentityErrors,
		gMetrics.queryValueErrors, gMetrics.solverQueryHits,
		double(gMetrics.minSolverY), double(gMetrics.maxSolverSpeed),
		double(gMetrics.initialSolverPosition.y),
		double(gMetrics.finalSolverPosition.y),
		double(gMetrics.finalSolverVelocity.y), gMetrics.nonFinite,
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
		std::fprintf(
			stderr,
			"[AVBD_GATE_CONFIG_ERROR] "
			"snippet=SnippetCustomGeometryQueries reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(
			stderr,
			"[AVBD_GATE_CONFIG_ERROR] "
			"snippet=SnippetCustomGeometryQueries "
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
