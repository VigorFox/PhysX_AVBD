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
static PxRigidDynamic*			gHeadlessDynamicTarget = NULL;
static PxRigidStatic*			gHeadlessGround = NULL;
static PxRigidActor*			gHeadlessContactTarget = NULL;
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
	PxU32 thresholdFoundCount;
	PxU32 thresholdPersistsCount;
	PxU32 thresholdLostCount;
	PxU32 unexpectedTouchEventCount;
	PxU32 thresholdReadbackErrors;
	PxU32 frictionAnchorCount;
	PxU32 nonzeroFrictionAnchorImpulseCount;
	PxU32 nonFinite;
	PxReal impulseSum;
	PxReal maxImpulse;
	PxReal maxFrictionAnchorImpulse;
	PxReal minBodyY;
	PxReal finalAbsVelocityX;
	PxReal displacementX;
	PxU32 cleanupComplete;

	ContactReportMetrics()
	: completedFrames(0), fetchFailures(0), callbackCount(0), pairCount(0),
	  pointCount(0), nonzeroImpulseCount(0), identityErrors(0),
	  thresholdFoundCount(0), thresholdPersistsCount(0),
	  thresholdLostCount(0), unexpectedTouchEventCount(0),
	  thresholdReadbackErrors(0), frictionAnchorCount(0),
	  nonzeroFrictionAnchorImpulseCount(0), nonFinite(0),
	  impulseSum(0.0f), maxImpulse(0.0f),
	  maxFrictionAnchorImpulse(0.0f),
	  minBodyY(PX_MAX_F32), finalAbsVelocityX(0.0f),
	  displacementX(0.0f), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static ContactReportMetrics gMetrics;
static PxReal gHeadlessInitialX = 0.0f;

PxArray<PxVec3> gContactPositions;
PxArray<PxVec3> gContactImpulses;

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(), name);
}

static bool isFrictionAnchorCase()
{
	return isHeadlessCase("friction-anchor") ||
		isHeadlessCase("dynamic-friction-anchor");
}

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
	if(!isHeadlessCase("drop") &&
		!isHeadlessCase("force-threshold") &&
		!isHeadlessCase("friction-anchor") &&
		!isHeadlessCase("dynamic-friction-anchor"))
	{
		error = "unsupported --case (expected drop, force-threshold, "
			"friction-anchor, or dynamic-friction-anchor)";
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

	pairFlags = PxPairFlag::eSOLVE_CONTACT |
		PxPairFlag::eDETECT_DISCRETE_CONTACT |
		PxPairFlag::eNOTIFY_CONTACT_POINTS;
	if(isHeadlessCase("force-threshold"))
	{
		pairFlags |= PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND |
			PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS |
			PxPairFlag::eNOTIFY_THRESHOLD_FORCE_LOST;
	}
	else
	{
		pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
			PxPairFlag::eNOTIFY_TOUCH_PERSISTS;
	}
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
			const bool identityValid =
				(pairHeader.actors[0] == gHeadlessBody &&
				 pairHeader.actors[1] == gHeadlessContactTarget) ||
				(pairHeader.actors[1] == gHeadlessBody &&
				 pairHeader.actors[0] == gHeadlessContactTarget);
			const bool involvesHeadlessBody =
				pairHeader.actors[0] == gHeadlessBody ||
				pairHeader.actors[1] == gHeadlessBody;
			if(!identityValid)
			{
				if(involvesHeadlessBody)
					++gMetrics.identityErrors;
				return;
			}
			++gMetrics.callbackCount;
			gMetrics.pairCount += nbPairs;
		}
		PxArray<PxContactPairPoint> contactPoints;
		
		for(PxU32 i=0;i<nbPairs;i++)
		{
			if(gHeadlessOptions.headless)
			{
				if(pairs[i].events &
					PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND)
					++gMetrics.thresholdFoundCount;
				if(pairs[i].events &
					PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS)
					++gMetrics.thresholdPersistsCount;
				if(pairs[i].events &
					PxPairFlag::eNOTIFY_THRESHOLD_FORCE_LOST)
					++gMetrics.thresholdLostCount;
				if(isHeadlessCase("force-threshold") &&
					(pairs[i].events &
					 (PxPairFlag::eNOTIFY_TOUCH_FOUND |
					  PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
					  PxPairFlag::eNOTIFY_TOUCH_LOST)))
					++gMetrics.unexpectedTouchEventCount;
			}
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
			if(gHeadlessOptions.headless && pairs[i].frictionPatches &&
				pairs[i].patchCount)
			{
				PxArray<PxContactPairFrictionAnchor> frictionAnchors;
				frictionAnchors.resize(PxU32(pairs[i].patchCount) * 2u);
				const PxU32 anchorCount = pairs[i].extractFrictionAnchors(
					frictionAnchors.begin(), frictionAnchors.size());
				gMetrics.frictionAnchorCount += anchorCount;
				for(PxU32 j = 0; j < anchorCount; ++j)
				{
					const PxContactPairFrictionAnchor& anchor =
						frictionAnchors[j];
					if(!anchor.position.isFinite() ||
						!anchor.impulse.isFinite())
					{
						++gMetrics.nonFinite;
						continue;
					}
					const PxReal impulse = anchor.impulse.magnitude();
					gMetrics.maxFrictionAnchorImpulse = PxMax(
						gMetrics.maxFrictionAnchorImpulse, impulse);
					if(impulse > 1.0e-5f)
						++gMetrics.nonzeroFrictionAnchorImpulseCount;
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
		gPhysics->createMaterial(
			isFrictionAnchorCase() ? 1.0f : 0.0f,
			isFrictionAnchorCase() ? 1.0f : 0.0f, 0.0f) :
		gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	if(gHeadlessOptions.headless)
	{
		gHeadlessGround =
			PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
		if(gHeadlessGround)
			gScene->addActor(*gHeadlessGround);

		if(isHeadlessCase("dynamic-friction-anchor"))
		{
			gHeadlessDynamicTarget = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(0.0f, 0.5f, 0.0f)),
				PxBoxGeometry(50.0f, 0.5f, 50.0f),
				*gMaterial, 0.1f);
			if(gHeadlessDynamicTarget)
			{
				gHeadlessDynamicTarget->setLinearDamping(0.0f);
				gHeadlessDynamicTarget->setAngularDamping(0.0f);
				gHeadlessDynamicTarget->setSleepThreshold(0.0f);
				gScene->addActor(*gHeadlessDynamicTarget);
				gHeadlessContactTarget = gHeadlessDynamicTarget;
			}
		}
		else
		{
			gHeadlessContactTarget = gHeadlessGround;
		}

		const PxReal initialY =
			isHeadlessCase("dynamic-friction-anchor") ? 1.5f :
			(isHeadlessCase("friction-anchor") ? 0.5f : 5.0f);
		gHeadlessBody = PxCreateDynamic(
			*gPhysics,
			PxTransform(PxVec3(0.0f, initialY, 0.0f)),
			PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
		if(gHeadlessBody)
		{
			gHeadlessBody->setLinearDamping(0.0f);
			gHeadlessBody->setAngularDamping(0.0f);
			gHeadlessBody->setSleepThreshold(0.0f);
			if(isFrictionAnchorCase())
				gHeadlessBody->setLinearVelocity(
					PxVec3(5.0f, 0.0f, 0.0f));
			if(isHeadlessCase("force-threshold"))
			{
				const PxReal threshold = 20.0f;
				gHeadlessBody->setContactReportThreshold(threshold);
				if(PxAbs(gHeadlessBody->getContactReportThreshold() -
					threshold) > 1.0e-6f)
					++gMetrics.thresholdReadbackErrors;
			}
			gHeadlessInitialX = gHeadlessBody->getGlobalPose().p.x;
			gScene->addActor(*gHeadlessBody);
		}
	}
	else
	{
		gHeadlessGround =
			PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
		gScene->addActor(*gHeadlessGround);
		createStack(PxTransform(PxVec3(0,3.0f,10.0f)), 5, 2.0f);
	}
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
			gMetrics.finalAbsVelocityX = PxAbs(velocity.x);
			gMetrics.displacementX =
				PxAbs(pose.p.x - gHeadlessInitialX);
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
	gHeadlessDynamicTarget = NULL;
	gHeadlessGround = NULL;
	gHeadlessContactTarget = NULL;
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
		!gScene && !gHeadlessBody && !gHeadlessDynamicTarget &&
		!gHeadlessGround && !gHeadlessContactTarget && !gMaterial &&
		!gDispatcher && !gPhysics && !gPvd && !gFoundation &&
		!gExtensionsInitialized ? 1u : 0u;
	
	printf("SnippetContactReport done.\n");
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	std::printf(
		"[AVBD_GATE_CONFIG] schema=2 snippet=SnippetContactReport solver=%s "
		"case=%s execution=%s frames=%u dt=%.9g dispatcherThreads=%u "
		"seed=%u\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, double(gHeadlessOptions.dt),
		gHeadlessOptions.dispatcherThreads, gHeadlessOptions.seed);

	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gHeadlessBody &&
		gHeadlessContactTarget;
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
	else if(gMetrics.thresholdReadbackErrors != 0 ||
		gMetrics.unexpectedTouchEventCount != 0)
	{
		passed = false;
		reason = "event_payload_error";
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
	else if(isHeadlessCase("force-threshold") &&
		(gMetrics.thresholdFoundCount == 0 ||
		 gMetrics.thresholdLostCount == 0))
	{
		passed = false;
		reason = "missing_force_threshold_transition";
	}
	else if(isFrictionAnchorCase() &&
		(gMetrics.frictionAnchorCount == 0 ||
		 gMetrics.nonzeroFrictionAnchorImpulseCount == 0 ||
		 gMetrics.maxFrictionAnchorImpulse <= 1.0e-3f ||
		 gMetrics.finalAbsVelocityX >= 4.0f ||
		 gMetrics.displacementX <= 0.01f))
	{
		passed = false;
		reason = "missing_friction_anchor_impulse";
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
		"[AVBD_GATE] schema=2 snippet=SnippetContactReport solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED callbackCount=%u pairCount=%u "
		"pointCount=%u nonzeroImpulseCount=%u identityErrors=%u "
		"thresholdFoundCount=%u thresholdPersistsCount=%u "
		"thresholdLostCount=%u unexpectedTouchEventCount=%u "
		"thresholdReadbackErrors=%u frictionAnchorCount=%u "
		"nonzeroFrictionAnchorImpulseCount=%u impulseSum=%.9g "
		"maxImpulse=%.9g maxFrictionAnchorImpulse=%.9g minBodyY=%.9g "
		"finalAbsVelocityX=%.9g displacementX=%.9g nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, gMetrics.callbackCount,
		gMetrics.pairCount, gMetrics.pointCount,
		gMetrics.nonzeroImpulseCount, gMetrics.identityErrors,
		gMetrics.thresholdFoundCount, gMetrics.thresholdPersistsCount,
		gMetrics.thresholdLostCount, gMetrics.unexpectedTouchEventCount,
		gMetrics.thresholdReadbackErrors,
		gMetrics.frictionAnchorCount,
		gMetrics.nonzeroFrictionAnchorImpulseCount,
		double(gMetrics.impulseSum), double(gMetrics.maxImpulse),
		double(gMetrics.maxFrictionAnchorImpulse),
		double(gMetrics.minBodyY),
		double(gMetrics.finalAbsVelocityX),
		double(gMetrics.displacementX), gMetrics.nonFinite,
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
