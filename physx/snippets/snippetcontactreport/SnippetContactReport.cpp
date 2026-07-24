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
// This snippet illustrates the use of simple contact reports.
//
// It defines a filter shader function that requests touch reports for 
// all pairs, and a contact callback function that saves the contact points.  
// It configures the scene to use this filter and callback, and prints the 
// number of contact reports each frame. If rendering, it renders each 
// contact as a line whose length and direction are defined by the contact 
// impulse.
// 
// ****************************************************************************

#include "PxPhysicsAPI.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
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
static PxRigidDynamic*			gHeadlessBody = NULL;
static PxRigidStatic*			gHeadlessGround = NULL;
static bool						gExtensionsInitialized = false;

struct ContactReportMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 callbackCount;
	PxU32 pairCount;
	PxU32 pointCount;
	PxU32 nonzeroImpulseCount;
	PxU32 identityErrors;
	PxU32 nonFinite;
	PxReal impulseSum;
	PxReal maxImpulse;
	PxReal minBodyY;
	PxU32 cleanupComplete;

	ContactReportMetrics()
	: completedFrames(0), fetchFailures(0), callbackCount(0), pairCount(0),
	  pointCount(0), nonzeroImpulseCount(0), identityErrors(0),
	  nonFinite(0), impulseSum(0.0f), maxImpulse(0.0f),
	  minBodyY(PX_MAX_F32), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static ContactReportMetrics gMetrics;

PxArray<PxVec3> gContactPositions;
PxArray<PxVec3> gContactImpulses;

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.caseName = "drop";
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
	if(!Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(), "drop"))
	{
		error = "unsupported --case (expected drop)";
		return false;
	}
	if(gHeadlessOptions.frames < 180)
	{
		error = "--frames must be at least 180";
		return false;
	}
	return true;
}

static PxFilterFlags contactReportFilterShader(	PxFilterObjectAttributes attributes0, PxFilterData filterData0, 
												PxFilterObjectAttributes attributes1, PxFilterData filterData1,
												PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlockSize);
	PX_UNUSED(constantBlock);

	// all initial and persisting reports for everything, with per-point data
	pairFlags = PxPairFlag::eSOLVE_CONTACT | PxPairFlag::eDETECT_DISCRETE_CONTACT
			  |	PxPairFlag::eNOTIFY_TOUCH_FOUND 
			  | PxPairFlag::eNOTIFY_TOUCH_PERSISTS
			  | PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

class ContactReportCallback: public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count)	{ PX_UNUSED(constraints); PX_UNUSED(count); }
	void onWake(PxActor** actors, PxU32 count)							{ PX_UNUSED(actors); PX_UNUSED(count); }
	void onSleep(PxActor** actors, PxU32 count)							{ PX_UNUSED(actors); PX_UNUSED(count); }
	void onTrigger(PxTriggerPair* pairs, PxU32 count)					{ PX_UNUSED(pairs); PX_UNUSED(count); }
	void onAdvance(const PxRigidBody*const*, const PxTransform*, const PxU32) {}
	void onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs, PxU32 nbPairs) 
	{
		if(gHeadlessOptions.headless)
		{
			++gMetrics.callbackCount;
			gMetrics.pairCount += nbPairs;
			const bool identityValid =
				(pairHeader.actors[0] == gHeadlessBody &&
				 pairHeader.actors[1] == gHeadlessGround) ||
				(pairHeader.actors[1] == gHeadlessBody &&
				 pairHeader.actors[0] == gHeadlessGround);
			if(!identityValid)
				++gMetrics.identityErrors;
		}
		PxArray<PxContactPairPoint> contactPoints;
		
		for(PxU32 i=0;i<nbPairs;i++)
		{
			PxU32 contactCount = pairs[i].contactCount;
			if(contactCount)
			{
				contactPoints.resize(contactCount);
				pairs[i].extractContacts(&contactPoints[0], contactCount);

				for(PxU32 j=0;j<contactCount;j++)
				{
					gContactPositions.pushBack(contactPoints[j].position);
					gContactImpulses.pushBack(contactPoints[j].impulse);
					if(gHeadlessOptions.headless)
					{
						++gMetrics.pointCount;
						const PxReal impulse =
							contactPoints[j].impulse.magnitude();
						if(!contactPoints[j].position.isFinite() ||
							!contactPoints[j].normal.isFinite() ||
							!contactPoints[j].impulse.isFinite() ||
							!PxIsFinite(contactPoints[j].separation))
							++gMetrics.nonFinite;
						gMetrics.impulseSum += impulse;
						gMetrics.maxImpulse =
							PxMax(gMetrics.maxImpulse, impulse);
						if(impulse > 1e-5f)
							++gMetrics.nonzeroImpulseCount;
					}
				}
			}
		}
	}
};

ContactReportCallback gContactReportCallback;

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

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	gExtensionsInitialized = PxInitExtensions(*gPhysics,gPvd);
	if(gHeadlessOptions.headless)
		gDispatcher = PxDefaultCpuDispatcherCreate(
			gHeadlessOptions.dispatcherThreads);
	else
	{
		PxU32 numCores = SnippetUtils::getNbPhysicalCores();
		gDispatcher = PxDefaultCpuDispatcherCreate(
			numCores == 0 ? 0 : numCores - 1);
	}
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.gravity = PxVec3(0, -9.81f, 0);
	sceneDesc.filterShader	= contactReportFilterShader;			
	sceneDesc.simulationEventCallback = &gContactReportCallback;	
	sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);

	if(!gHeadlessOptions.headless)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
	}
	gMaterial = gHeadlessOptions.headless ?
		gPhysics->createMaterial(0.0f, 0.0f, 0.0f) :
		gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	gHeadlessGround =
		PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*gHeadlessGround);
	if(gHeadlessOptions.headless)
	{
		gHeadlessBody = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f, 5.0f, 0.0f)),
			PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
		gHeadlessBody->setLinearDamping(0.0f);
		gHeadlessBody->setAngularDamping(0.0f);
		gHeadlessBody->setSleepThreshold(0.0f);
		gScene->addActor(*gHeadlessBody);
	}
	else
		createStack(PxTransform(PxVec3(0,3.0f,10.0f)), 5, 2.0f);
}

void stepPhysics(bool /*interactive*/)
{
	gContactPositions.clear();
	gContactImpulses.clear();

	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gHeadlessBody)
		{
			const PxTransform pose = gHeadlessBody->getGlobalPose();
			const PxVec3 velocity = gHeadlessBody->getLinearVelocity();
			if(!pose.isFinite() || !velocity.isFinite())
				++gMetrics.nonFinite;
			gMetrics.minBodyY = PxMin(gMetrics.minBodyY, pose.p.y);
		}
		++gMetrics.completedFrames;
	}
	else
		printf("%u contact reports\n", PxU32(gContactPositions.size()));
}
	
void cleanupPhysics(bool /*interactive*/)
{
	gContactPositions.reset();
	gContactImpulses.reset();
    
	PX_RELEASE(gScene);
	gHeadlessBody = NULL;
	gHeadlessGround = NULL;
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
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gMaterial && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation && !gExtensionsInitialized ? 1u : 0u;
	
	printf("SnippetContactReport done.\n");
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	std::printf(
		"[AVBD_GATE_CONFIG] schema=1 snippet=SnippetContactReport solver=%s "
		"case=drop execution=%s frames=%u dt=%.9g dispatcherThreads=%u "
		"seed=%u\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, double(gHeadlessOptions.dt),
		gHeadlessOptions.dispatcherThreads, gHeadlessOptions.seed);

	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gHeadlessBody && gHeadlessGround;
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
	}

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
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.callbackCount == 0 || gMetrics.pairCount == 0)
	{
		passed = false;
		reason = "missing_contact_callback";
	}
	else if(gMetrics.identityErrors != 0)
	{
		passed = false;
		reason = "contact_identity_mismatch";
	}
	else if(gMetrics.pointCount == 0)
	{
		passed = false;
		reason = "missing_contact_points";
	}
	else if(gMetrics.nonzeroImpulseCount == 0 ||
		gMetrics.maxImpulse <= 1e-5f)
	{
		passed = false;
		reason = "missing_contact_impulse";
	}
	else if(gMetrics.minBodyY < 0.3f)
	{
		passed = false;
		reason = "body_fell_through";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetContactReport solver=%s "
		"case=drop execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED callbackCount=%u pairCount=%u "
		"pointCount=%u nonzeroImpulseCount=%u identityErrors=%u "
		"impulseSum=%.9g maxImpulse=%.9g minBodyY=%.9g nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, gMetrics.callbackCount,
		gMetrics.pairCount, gMetrics.pointCount,
		gMetrics.nonzeroImpulseCount, gMetrics.identityErrors,
		double(gMetrics.impulseSum), double(gMetrics.maxImpulse),
		double(gMetrics.minBodyY), gMetrics.nonFinite,
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
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetContactReport reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetContactReport "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysics(false);
	for(PxU32 i=0; i<250; i++)
		stepPhysics(false);
	cleanupPhysics(false);
#endif

	return 0;
}
