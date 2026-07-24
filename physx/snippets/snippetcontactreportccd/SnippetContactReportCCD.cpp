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
// This snippet illustrates the use of simple contact reports in combination
// with continuous collision detection (CCD). Furthermore, extra contact report
// data will be requested.
//
// The snippet defines a filter shader function that enables CCD and requests
// touch reports for all pairs, and a contact callback function that saves the 
// contact points and the actor positions at time of impact.
// It configures the scene to use this filter and callback, enables CCD and 
// prints the number of contact points found. If rendering, it renders each 
// contact as a line whose length and direction are defined by the contact 
// impulse (the line points in the opposite direction of the impulse). In
// addition, the path of the fast moving dynamic object is drawn with lines.
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
static PxFoundation*			gFoundation			= NULL;
static PxPhysics*				gPhysics			= NULL;
static PxDefaultCpuDispatcher*	gDispatcher			= NULL;
static PxScene*					gScene				= NULL;
static PxMaterial*				gMaterial			= NULL;
static PxTriangleMesh*			gTriangleMesh		= NULL;
static PxRigidStatic*			gTriangleMeshActor	= NULL;
static PxRigidDynamic*			gSphereActor		= NULL;
static PxPvd*					gPvd                = NULL;
static PxU32					gSimStepCount		= 0;
static bool						gExtensionsInitialized = false;

struct ContactReportCcdMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 callbackCount;
	PxU32 pairCount;
	PxU32 foundEventCount;
	PxU32 ccdEventCount;
	PxU32 contactPointCount;
	PxU32 nonzeroImpulseCount;
	PxU32 eventPoseCount;
	PxU32 identityErrors;
	PxU32 nonFinite;
	PxReal impulseSum;
	PxReal maxImpulse;
	PxVec3 initialPosition;
	PxVec3 finalPosition;
	PxU32 cleanupComplete;

	ContactReportCcdMetrics()
	: completedFrames(0), fetchFailures(0), callbackCount(0), pairCount(0),
	  foundEventCount(0), ccdEventCount(0), contactPointCount(0),
	  nonzeroImpulseCount(0), eventPoseCount(0), identityErrors(0),
	  nonFinite(0), impulseSum(0.0f), maxImpulse(0.0f),
	  initialPosition(0.0f), finalPosition(0.0f), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static ContactReportCcdMetrics gMetrics;

PxArray<PxVec3> gContactPositions;
PxArray<PxVec3> gContactImpulses;
PxArray<PxVec3> gContactSphereActorPositions;

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(), name);
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 1;
	defaults.caseName = "ccd-report";
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
	if(!isHeadlessCase("ccd-report") &&
		!isHeadlessCase("no-ccd-control"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames != 1)
	{
		error = "--frames must be 1";
		return false;
	}
	if(isHeadlessCase("no-ccd-control") &&
		gHeadlessOptions.solverType != PxSolverType::eTGS)
	{
		error = "no-ccd-control requires TGS";
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

	//
	// Enable CCD for the pair, request contact reports for initial and CCD contacts.
	// Additionally, provide information per contact point and provide the actor
	// pose at the time of contact.
	//

	pairFlags = PxPairFlag::eCONTACT_DEFAULT
			  |	PxPairFlag::eNOTIFY_TOUCH_FOUND
			  | PxPairFlag::eNOTIFY_CONTACT_POINTS
			  | PxPairFlag::eCONTACT_EVENT_POSE;
	if(!gHeadlessOptions.headless || isHeadlessCase("ccd-report"))
		pairFlags |= PxPairFlag::eDETECT_CCD_CONTACT
				  | PxPairFlag::eNOTIFY_TOUCH_CCD;
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
			const bool expectedIdentity =
				(pairHeader.actors[0] == gSphereActor &&
				 pairHeader.actors[1] == gTriangleMeshActor) ||
				(pairHeader.actors[1] == gSphereActor &&
				 pairHeader.actors[0] == gTriangleMeshActor);
			if(!expectedIdentity)
				++gMetrics.identityErrors;
		}

		PxArray<PxContactPairPoint> contactPoints;

		PxTransform spherePose(PxIdentity);
		PxU32 nextPairIndex = 0xffffffff;

		PxContactPairExtraDataIterator iter(pairHeader.extraDataStream, pairHeader.extraDataStreamSize);
		bool hasItemSet = iter.nextItemSet();
		if (hasItemSet)
			nextPairIndex = iter.contactPairIndex;

		for(PxU32 i=0; i < nbPairs; i++)
		{
			//
			// Get the pose of the dynamic object at time of impact.
			//
			if (nextPairIndex == i)
			{
				if(iter.eventPose)
				{
					if (pairHeader.actors[0]->is<PxRigidDynamic>())
						spherePose = iter.eventPose->globalPose[0];
					else
						spherePose = iter.eventPose->globalPose[1];

					gContactSphereActorPositions.pushBack(spherePose.p);
					if(gHeadlessOptions.headless)
					{
						++gMetrics.eventPoseCount;
						if(!spherePose.isFinite())
							++gMetrics.nonFinite;
					}
				}

				hasItemSet = iter.nextItemSet();
				if (hasItemSet)
					nextPairIndex = iter.contactPairIndex;
			}

			//
			// Get the contact points for the pair.
			//
			const PxContactPair& cPair = pairs[i];
			if(gHeadlessOptions.headless)
			{
				if(cPair.events & PxPairFlag::eNOTIFY_TOUCH_FOUND)
					++gMetrics.foundEventCount;
				if(cPair.events & PxPairFlag::eNOTIFY_TOUCH_CCD)
					++gMetrics.ccdEventCount;
			}
			if (cPair.events & (PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_CCD))
			{
				PxU32 contactCount = cPair.contactCount;
				contactPoints.resize(contactCount);
				const PxU32 extracted = contactCount ?
					cPair.extractContacts(contactPoints.begin(), contactCount) :
					0;
				if(gHeadlessOptions.headless)
					gMetrics.contactPointCount += extracted;

				for(PxU32 j=0; j < extracted; j++)
				{
					gContactPositions.pushBack(contactPoints[j].position);
					gContactImpulses.pushBack(contactPoints[j].impulse);
					if(gHeadlessOptions.headless)
					{
						const PxReal impulse =
							contactPoints[j].impulse.magnitude();
						gMetrics.impulseSum += impulse;
						gMetrics.maxImpulse =
							PxMax(gMetrics.maxImpulse, impulse);
						if(impulse > 1e-6f)
							++gMetrics.nonzeroImpulseCount;
						if(!contactPoints[j].position.isFinite() ||
							!contactPoints[j].normal.isFinite() ||
							!contactPoints[j].impulse.isFinite() ||
							!PxIsFinite(contactPoints[j].separation))
							++gMetrics.nonFinite;
					}
				}
			}
		}
	}
};

ContactReportCallback gContactReportCallback;

static void initScene()
{
	//
	// Create a static triangle mesh
	//

	PxVec3 vertices[] = {	PxVec3(-8.0f, 0.0f, -3.0f),
							PxVec3(-8.0f, 0.0f, 3.0f),
							PxVec3(0.0f, 0.0f, 3.0f),
							PxVec3(0.0f, 0.0f, -3.0f),
							PxVec3(-8.0f, 10.0f, -3.0f),
							PxVec3(-8.0f, 10.0f, 3.0f),
							PxVec3(0.0f, 10.0f, 3.0f),
							PxVec3(0.0f, 10.0f, -3.0f),
						};

	PxU32 vertexCount = sizeof(vertices) / sizeof(vertices[0]);

	PxU32 triangleIndices[] = {	0, 1, 2,
								0, 2, 3,
								0, 5, 1,
								0, 4, 5,
								4, 6, 5,
								4, 7, 6
							};
	PxU32 triangleCount = (sizeof(triangleIndices) / sizeof(triangleIndices[0])) / 3;

	PxTriangleMeshDesc triangleMeshDesc;
	triangleMeshDesc.points.count = vertexCount;
	triangleMeshDesc.points.data = vertices;
	triangleMeshDesc.points.stride = sizeof(PxVec3);
	triangleMeshDesc.triangles.count = triangleCount;
	triangleMeshDesc.triangles.data = triangleIndices;
	triangleMeshDesc.triangles.stride = 3 * sizeof(PxU32);

	PxTolerancesScale tolerances;
	const PxCookingParams params(tolerances);
	gTriangleMesh = PxCreateTriangleMesh(params, triangleMeshDesc, gPhysics->getPhysicsInsertionCallback());

	if (!gTriangleMesh)
		return;

	gTriangleMeshActor = gPhysics->createRigidStatic(PxTransform(PxVec3(0.0f, 1.0f, 0.0f), PxQuat(PxHalfPi / 60.0f, PxVec3(0.0f, 1.0f, 0.0f))));

	if (!gTriangleMeshActor)
		return;

	PxTriangleMeshGeometry triGeom(gTriangleMesh);
	PxShape* triangleMeshShape = PxRigidActorExt::createExclusiveShape(*gTriangleMeshActor, triGeom, *gMaterial);

	if (!triangleMeshShape)
		return;

	gScene->addActor(*gTriangleMeshActor);

	
	//
	// Create a fast moving sphere that will hit and bounce off the static triangle mesh 3 times
	// in one simulation step.
	//

	PxTransform spherePose(PxVec3(0.0f, 5.0f, 1.0f));
	gContactSphereActorPositions.pushBack(spherePose.p);
	gMetrics.initialPosition = spherePose.p;
	gSphereActor = gPhysics->createRigidDynamic(spherePose);

	if (!gSphereActor)
		return;
	if(!gHeadlessOptions.headless || isHeadlessCase("ccd-report"))
		gSphereActor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);

	PxSphereGeometry sphereGeom(1.0f);
	PxShape* sphereShape = PxRigidActorExt::createExclusiveShape(*gSphereActor, sphereGeom, *gMaterial);

	if (!sphereShape)
		return;

	PxRigidBodyExt::updateMassAndInertia(*gSphereActor, 1.0f);

	PxReal velMagn = 900.0f;
	PxVec3 vel = PxVec3(-1.0f, -1.0f, 0.0f);
	vel.normalize();
	vel *= velMagn;
	gSphereActor->setLinearVelocity(vel);

	gScene->addActor(*gSphereActor);
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
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);

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
	sceneDesc.gravity = PxVec3(0, 0, 0);
	sceneDesc.filterShader	= contactReportFilterShader;			
	sceneDesc.simulationEventCallback = &gContactReportCallback;
	sceneDesc.solverType = gHeadlessOptions.solverType;
	if(!gHeadlessOptions.headless || isHeadlessCase("ccd-report"))
		sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;
	sceneDesc.ccdMaxPasses = 4;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gHeadlessOptions.headless)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 1.0f);

	initScene();
}

void stepPhysics(bool /*interactive*/)
{
	if (!gSimStepCount)
	{
		const PxReal dt = gHeadlessOptions.headless ?
			gHeadlessOptions.dt : 1.0f/60.0f;
		gScene->simulate(dt);
		const bool fetched = gScene->fetchResults(true);
		if(gHeadlessOptions.headless)
		{
			if(!fetched)
				++gMetrics.fetchFailures;
			++gMetrics.completedFrames;
		}
		else
			printf("%u contact points\n", PxU32(gContactPositions.size()));

		if (gSphereActor)
		{
			gContactSphereActorPositions.pushBack(gSphereActor->getGlobalPose().p);
			if(gHeadlessOptions.headless)
			{
				const PxTransform pose = gSphereActor->getGlobalPose();
				const PxVec3 velocity = gSphereActor->getLinearVelocity();
				gMetrics.finalPosition = pose.p;
				if(!pose.isFinite() || !velocity.isFinite())
					++gMetrics.nonFinite;
			}
		}

		gSimStepCount = 1;
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	gContactPositions.reset();
	gContactImpulses.reset();
	gContactSphereActorPositions.reset();

	PX_RELEASE(gSphereActor);
	PX_RELEASE(gTriangleMeshActor);
	PX_RELEASE(gTriangleMesh);

	PX_RELEASE(gScene);
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
		!gSphereActor && !gTriangleMeshActor && !gTriangleMesh &&
		!gScene && !gMaterial && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation && !gExtensionsInitialized ? 1u : 0u;

	printf("SnippetContactReportCCD done.\n");
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig(
		"SnippetContactReportCCD", gHeadlessOptions);

	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gTriangleMesh && gTriangleMeshActor &&
		gSphereActor;
	if(initialized)
		stepPhysics(false);

	const bool controlCase = isHeadlessCase("no-ccd-control");
	const bool tunneled =
		gMetrics.finalPosition.x < -8.0f &&
		gMetrics.finalPosition.y < 0.0f;
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
	else if(controlCase &&
		(gMetrics.ccdEventCount != 0 ||
		 gMetrics.contactPointCount != 0 ||
		 gMetrics.eventPoseCount != 0 ||
		 !tunneled))
	{
		passed = false;
		reason = "negative_control_failed";
	}
	else if(!controlCase &&
		(gMetrics.callbackCount == 0 ||
		 gMetrics.pairCount < 3 ||
		 gMetrics.foundEventCount == 0 ||
		 gMetrics.ccdEventCount < 3 ||
		 gMetrics.contactPointCount < 3 ||
		 gMetrics.eventPoseCount < 3))
	{
		passed = false;
		reason = "missing_ccd_report";
	}
	else if(!controlCase &&
		gMetrics.nonzeroImpulseCount < 3)
	{
		passed = false;
		reason = "missing_ccd_impulse";
	}
	else if(!controlCase && tunneled)
	{
		passed = false;
		reason = "complete_tunneling";
	}
	else if(gMetrics.identityErrors != 0)
	{
		passed = false;
		reason = "actor_identity_error";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetContactReportCCD solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED callbackCount=%u pairCount=%u "
		"foundEventCount=%u ccdEventCount=%u contactPointCount=%u "
		"nonzeroImpulseCount=%u eventPoseCount=%u identityErrors=%u "
		"impulseSum=%.9g maxImpulse=%.9g initialX=%.9g initialY=%.9g "
		"finalX=%.9g finalY=%.9g tunneled=%u nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		gMetrics.callbackCount, gMetrics.pairCount,
		gMetrics.foundEventCount, gMetrics.ccdEventCount,
		gMetrics.contactPointCount, gMetrics.nonzeroImpulseCount,
		gMetrics.eventPoseCount, gMetrics.identityErrors,
		double(gMetrics.impulseSum), double(gMetrics.maxImpulse),
		double(gMetrics.initialPosition.x), double(gMetrics.initialPosition.y),
		double(gMetrics.finalPosition.x), double(gMetrics.finalPosition.y),
		tunneled ? 1u : 0u, gMetrics.nonFinite,
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
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetContactReportCCD "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetContactReportCCD "
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

	stepPhysics(false);

	cleanupPhysics(false);
#endif

	return 0;
}
