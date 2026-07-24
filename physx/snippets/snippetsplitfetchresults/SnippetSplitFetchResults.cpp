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
// This snippet illustrates the use of split fetchResults() calls to improve
// the performace contact report processing.
//
// It defines a filter shader function that requests touch reports for 
// all pairs, and a contact callback function that saves the contact points.  
// It configures the scene to use this filter and callback, and prints the 
// number of contact reports each frame. If rendering, it renders each 
// contact as a line whose length and direction are defined by the contact 
// impulse. The callback can be processed earlier than usual by using the
// split fetchResults() sequence of fetchResultsStart(), processCallbacks(), 
// fetchResultsFinish().
// 
// ****************************************************************************

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "foundation/PxAtomic.h"
#include "task/PxTask.h"
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

#define PARALLEL_CALLBACKS 1

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics = NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene = NULL;
static PxMaterial*				gMaterial = NULL;
static PxPvd*					gPvd = NULL;

const PxI32 maxCount = 10000;

PxI32 gSharedIndex = 0;
volatile PxI32 gCallbackInvocations = 0;
volatile PxI32 gCallbackPairs = 0;
volatile PxI32 gNonzeroImpulses = 0;
volatile PxI32 gNonFiniteContactPoints = 0;
volatile PxI32 gPhaseViolations = 0;
volatile PxI32 gContinuationReleases = 0;
volatile PxI32 gCallbackPhase = 0;

PxVec3* gContactPositions;
PxVec3* gContactImpulses;
PxVec3* gContactVertices;
Snippets::HeadlessOptions gHeadlessOptions;
bool gExtensionsInitialized = false;

struct SplitFetchMetrics
{
	PxU32 completedFrames;
	PxU32 fetchStartCalls;
	PxU32 processCallbackCalls;
	PxU32 fetchFinishCalls;
	PxU64 pairHeaders;
	PxU64 callbackInvocations;
	PxU64 callbackPairs;
	PxU64 contactPoints;
	PxU64 storedContactPoints;
	PxU64 overflowContactPoints;
	PxU64 nonzeroImpulses;
	PxU32 continuationReleases;
	PxU32 phaseViolations;
	PxU32 nonFiniteContacts;
	PxU32 nonFiniteActors;
	PxU32 fetchFailures;
	PxU32 dynamicBodies;
	PxReal minBodyY;
	PxReal maxBodyY;
	PxReal maxBodySpeed;
	PxU32 cleanupComplete;

	SplitFetchMetrics()
	: completedFrames(0), fetchStartCalls(0), processCallbackCalls(0),
	  fetchFinishCalls(0), pairHeaders(0), callbackInvocations(0),
	  callbackPairs(0), contactPoints(0), storedContactPoints(0),
	  overflowContactPoints(0), nonzeroImpulses(0),
	  continuationReleases(0), phaseViolations(0),
	  nonFiniteContacts(0), nonFiniteActors(0), fetchFailures(0),
	  dynamicBodies(0), minBodyY(FLT_MAX), maxBodyY(-FLT_MAX),
	  maxBodySpeed(0.0f), cleanupComplete(0)
	{
	}
};

static SplitFetchMetrics gMetrics;

class CallbackFinishTask : public PxLightCpuTask
{
	SnippetUtils::Sync* mSync;
public:
	CallbackFinishTask(){ mSync = SnippetUtils::syncCreate(); }
	~CallbackFinishTask(){ SnippetUtils::syncRelease(mSync); }

	virtual void release()
	{
		PxLightCpuTask::release();
		SnippetUtils::atomicIncrement(&gContinuationReleases);
		SnippetUtils::syncSet(mSync);
	}

	void reset() { SnippetUtils::syncReset(mSync); }

	void wait() { SnippetUtils::syncWait(mSync); }

	virtual void run() { /*Do nothing - release the sync in the release method for thread-safety*/}

	virtual const char* getName() const { return "CallbackFinishTask"; }
} 
callbackFinishTask;


PxFilterFlags contactReportFilterShader(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
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
		| PxPairFlag::eNOTIFY_TOUCH_FOUND
		| PxPairFlag::eNOTIFY_TOUCH_PERSISTS
		| PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

class ContactReportCallback : public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo* constraints, PxU32 count)	{ PX_UNUSED(constraints); PX_UNUSED(count); }
	void onWake(PxActor** actors, PxU32 count)							{ PX_UNUSED(actors); PX_UNUSED(count); }
	void onSleep(PxActor** actors, PxU32 count)							{ PX_UNUSED(actors); PX_UNUSED(count); }
	void onTrigger(PxTriggerPair* pairs, PxU32 count)					{ PX_UNUSED(pairs); PX_UNUSED(count); }
	void onAdvance(const PxRigidBody*const*, const PxTransform*, const PxU32) {}
	void onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs, PxU32 nbPairs)
	{
		PX_UNUSED((pairHeader));
		SnippetUtils::atomicIncrement(&gCallbackInvocations);
		physx::PxAtomicAdd(&gCallbackPairs, PxI32(nbPairs));
		if(gCallbackPhase != 3)
			SnippetUtils::atomicIncrement(&gPhaseViolations);
		//Maximum of 64 vertices can be produced by contact gen
		PxContactPairPoint contactPoints[64];

		for (PxU32 i = 0; i<nbPairs; i++)
		{
			PxU32 contactCount = pairs[i].contactCount;
			if (contactCount)
			{
				const PxU32 extracted =
					pairs[i].extractContacts(&contactPoints[0], contactCount);

				PxI32 startIdx = physx::PxAtomicAdd(
					&gSharedIndex, int32_t(extracted));
				for (PxU32 j = 0; j<extracted; j++)
				{
					const PxContactPairPoint& point = contactPoints[j];
					if(!point.position.isFinite() || !point.impulse.isFinite())
						SnippetUtils::atomicIncrement(
							&gNonFiniteContactPoints);
					if(point.impulse.magnitudeSquared() > 1e-12f)
						SnippetUtils::atomicIncrement(&gNonzeroImpulses);
					const PxI32 index = startIdx + PxI32(j);
					if(index < maxCount)
					{
						gContactPositions[index] = point.position;
						gContactImpulses[index] = point.impulse;
						gContactVertices[2*index] = point.position;
						gContactVertices[2*index + 1] =
							point.position + point.impulse * 0.1f;
					}
				}
			}
		}
	}
};

ContactReportCallback gContactReportCallback;

void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for (PxU32 i = 0; i<size; i++)
	{
		for (PxU32 j = 0; j<size - i; j++)
		{
			PxTransform localTm(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			gScene->addActor(*body);
		}
	}
	shape->release();
}

static bool initPhysicsInternal(bool interactive)
{
	gContactPositions = new PxVec3[maxCount];
	gContactImpulses = new PxVec3[maxCount];
	gContactVertices = new PxVec3[2*maxCount];

	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	if(interactive)
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
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
		return false;
	const PxU32 numCores = SnippetUtils::getNbPhysicalCores();
	const PxU32 dispatcherThreads = gHeadlessOptions.headless ?
		gHeadlessOptions.dispatcherThreads :
		(numCores > 1 ? numCores - 1 : 1);
	gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
	if(!gDispatcher)
		return false;
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.gravity = PxVec3(0, -9.81f, 0);
	sceneDesc.filterShader = contactReportFilterShader;
	sceneDesc.simulationEventCallback = &gContactReportCallback;
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
		return false;

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*groundPlane);

	const PxU32 nbStacks = 50;

	for (PxU32 i = 0; i < nbStacks; ++i)
	{
		createStack(PxTransform(PxVec3(0, 3.0f, 10.f - 5.f*i)), 5, 2.0f);
	}
	gMetrics.dynamicBodies =
		gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	return true;
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
			gMetrics.nonFiniteActors++;
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

static bool stepPhysicsInternal(bool interactive)
{
	gSharedIndex = 0;
	gCallbackInvocations = 0;
	gCallbackPairs = 0;
	gNonzeroImpulses = 0;
	gNonFiniteContactPoints = 0;
	gPhaseViolations = 0;
	gContinuationReleases = 0;
	gCallbackPhase = 1;

	gScene->simulate(gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f / 60.0f);

#if !PARALLEL_CALLBACKS
	gScene->fetchResults(true);
#else
	//Call fetchResultsStart. Get the set of pair headers
	const PxContactPairHeader* pairHeader;
	PxU32 nbContactPairs;
	if(!gScene->fetchResultsStart(pairHeader, nbContactPairs, true))
	{
		gMetrics.fetchFailures++;
		return false;
	}
	gMetrics.fetchStartCalls++;
	gMetrics.pairHeaders += nbContactPairs;
	gCallbackPhase = 2;

	//Set up continuation task to be run after callbacks have been processed in parallel
	callbackFinishTask.setContinuation(*gScene->getTaskManager(), NULL);
	callbackFinishTask.reset();

	//process the callbacks
	gCallbackPhase = 3;
	gScene->processCallbacks(&callbackFinishTask);
	gMetrics.processCallbackCalls++;

	callbackFinishTask.removeReference();

	callbackFinishTask.wait();
	gCallbackPhase = 4;
	
	gScene->fetchResultsFinish();
	gMetrics.fetchFinishCalls++;
	gCallbackPhase = 5;
#endif

	const PxU32 observedPoints = PxU32(PxMax(gSharedIndex, PxI32(0)));
	const PxU32 storedPoints =
		PxMin(observedPoints, PxU32(maxCount));
	gMetrics.callbackInvocations += PxU32(gCallbackInvocations);
	gMetrics.callbackPairs += PxU32(gCallbackPairs);
	gMetrics.contactPoints += observedPoints;
	gMetrics.storedContactPoints += storedPoints;
	gMetrics.overflowContactPoints += observedPoints - storedPoints;
	gMetrics.nonzeroImpulses += PxU32(gNonzeroImpulses);
	gMetrics.nonFiniteContacts += PxU32(gNonFiniteContactPoints);
	gMetrics.phaseViolations += PxU32(gPhaseViolations);
	gMetrics.continuationReleases += PxU32(gContinuationReleases);
	gMetrics.completedFrames++;
	gSharedIndex = PxI32(storedPoints);
	if(interactive)
		printf("%u contact reports\n", storedPoints);
	return sampleActors();
}

void initPhysics(bool interactive)
{
	initPhysicsInternal(interactive);
}

void stepPhysics(bool interactive)
{
	stepPhysicsInternal(interactive);
}

void cleanupPhysics(bool interactive)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
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

	delete[] gContactPositions;
	delete[] gContactImpulses;
	delete[] gContactVertices;
	gContactPositions = NULL;
	gContactImpulses = NULL;
	gContactVertices = NULL;
	gMetrics.cleanupComplete = 1;

	if(interactive)
		printf("SnippetSplitFetchResults done.\n");
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 180;
	defaults.caseName = "split-callbacks";
	defaults.solverType = PxSolverType::eAVBD;
	defaults.dispatcherThreads = 4;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") + argv[i];
			return false;
		}
	if(gHeadlessOptions.headless &&
		!Snippets::equalsIgnoreCase(
			gHeadlessOptions.caseName.c_str(), "split-callbacks"))
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

static bool headlessPassed()
{
	return
		gMetrics.completedFrames == gHeadlessOptions.frames &&
		gMetrics.fetchStartCalls == gHeadlessOptions.frames &&
		gMetrics.processCallbackCalls == gHeadlessOptions.frames &&
		gMetrics.fetchFinishCalls == gHeadlessOptions.frames &&
		gMetrics.pairHeaders > 0 &&
		gMetrics.callbackInvocations > 0 &&
		gMetrics.callbackPairs > 0 &&
		gMetrics.contactPoints > 0 &&
		gMetrics.storedContactPoints > 0 &&
		gMetrics.overflowContactPoints == 0 &&
		gMetrics.nonzeroImpulses > 0 &&
		gMetrics.continuationReleases == gHeadlessOptions.frames &&
		gMetrics.phaseViolations == 0 &&
		gMetrics.nonFiniteContacts == 0 &&
		gMetrics.nonFiniteActors == 0 &&
		gMetrics.fetchFailures == 0 &&
		gMetrics.dynamicBodies == 750 &&
		gMetrics.cleanupComplete == 1 &&
		gErrorCallback.getFatalCount() == 0;
}

static const char* failureReason()
{
	if(gErrorCallback.getFatalCount())
		return "fatal_error";
	if(gMetrics.cleanupComplete != 1)
		return "cleanup_incomplete";
	if(gMetrics.fetchFailures)
		return "fetch_failed";
	if(gMetrics.phaseViolations)
		return "callback_phase_violation";
	if(gMetrics.continuationReleases != gHeadlessOptions.frames)
		return "continuation_incomplete";
	if(gMetrics.nonFiniteContacts || gMetrics.nonFiniteActors)
		return "non_finite";
	if(gMetrics.overflowContactPoints)
		return "contact_buffer_overflow";
	if(!gMetrics.callbackInvocations || !gMetrics.contactPoints)
		return "callbacks_missing";
	if(!gMetrics.nonzeroImpulses)
		return "impulse_writeback_missing";
	return "lifecycle_incomplete";
}

static void printHeadlessResult()
{
	const bool pass = headlessPassed();
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetSplitFetchResults solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u "
		"fetchStartCalls=%u processCallbackCalls=%u fetchFinishCalls=%u "
		"pairHeaders=%llu callbackInvocations=%llu callbackPairs=%llu "
		"contactPoints=%llu storedContactPoints=%llu "
		"overflowContactPoints=%llu nonzeroImpulses=%llu "
		"continuationReleases=%u phaseViolations=%u "
		"nonFiniteContacts=%u nonFiniteActors=%u fetchFailures=%u "
		"dynamicBodies=%u minBodyY=%.9g maxBodyY=%.9g "
		"maxBodySpeed=%.9g fatalErrors=%u cleanupComplete=%u pvd=0 "
		"status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		gMetrics.fetchStartCalls, gMetrics.processCallbackCalls,
		gMetrics.fetchFinishCalls,
		static_cast<unsigned long long>(gMetrics.pairHeaders),
		static_cast<unsigned long long>(gMetrics.callbackInvocations),
		static_cast<unsigned long long>(gMetrics.callbackPairs),
		static_cast<unsigned long long>(gMetrics.contactPoints),
		static_cast<unsigned long long>(gMetrics.storedContactPoints),
		static_cast<unsigned long long>(gMetrics.overflowContactPoints),
		static_cast<unsigned long long>(gMetrics.nonzeroImpulses),
		gMetrics.continuationReleases, gMetrics.phaseViolations,
		gMetrics.nonFiniteContacts, gMetrics.nonFiniteActors,
		gMetrics.fetchFailures, gMetrics.dynamicBodies,
		double(gMetrics.minBodyY), double(gMetrics.maxBodyY),
		double(gMetrics.maxBodySpeed), gErrorCallback.getFatalCount(),
		gMetrics.cleanupComplete, pass ? "PASS" : "FAIL",
		pass ? "none" : failureReason());
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
			"SnippetSplitFetchResults", gHeadlessOptions);
		gErrorCallback.reset();
		bool runOk = initPhysicsInternal(false);
		if(runOk)
			for(PxU32 i = 0; i < gHeadlessOptions.frames; ++i)
				if(!stepPhysicsInternal(false))
				{
					runOk = false;
					break;
				}
		cleanupPhysics(false);
		printHeadlessResult();
		return runOk && headlessPassed() ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysics(true);
	for (PxU32 i = 0; i<250; i++)
		stepPhysics(true);
	cleanupPhysics(true);
#endif

	return 0;
}
