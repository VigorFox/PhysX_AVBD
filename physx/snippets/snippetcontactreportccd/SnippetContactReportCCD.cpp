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
#include "extensions/PxRaycastCCD.h"
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
static PxRigidDynamic*			gDynamicTargetActor	= NULL;
static PxRigidStatic*			gStaticTargetActor	= NULL;
static PxRigidActor*			gReportTargetActor	= NULL;
static RaycastCCDManager*		gRaycastCCD			= NULL;
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
	PxU32 sceneLinearCCD;
	PxU32 sourceLinearCCD;
	PxU32 sourceSpeculativeCCD;
	PxU32 targetLinearCCD;
	PxU32 targetSpeculativeCCD;
	PxU32 targetDynamic;
	PxU32 raycastRegistrations;
	PxU32 raycastCorrections;
	PxReal impulseSum;
	PxReal maxImpulse;
	PxReal targetDisplacement;
	PxVec3 initialPosition;
	PxVec3 finalPosition;
	PxVec3 targetInitialPosition;
	PxU32 cleanupComplete;

	ContactReportCcdMetrics()
	: completedFrames(0), fetchFailures(0), callbackCount(0), pairCount(0),
	  foundEventCount(0), ccdEventCount(0), contactPointCount(0),
	  nonzeroImpulseCount(0), eventPoseCount(0), identityErrors(0),
	  nonFinite(0), sceneLinearCCD(0), sourceLinearCCD(0),
	  sourceSpeculativeCCD(0), targetLinearCCD(0),
	  targetSpeculativeCCD(0), targetDynamic(0),
	  raycastRegistrations(0), raycastCorrections(0),
	  impulseSum(0.0f), maxImpulse(0.0f), targetDisplacement(0.0f),
	  initialPosition(0.0f), finalPosition(0.0f),
	  targetInitialPosition(0.0f), cleanupComplete(0)
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

static bool isAngularReportCase()
{
	return isHeadlessCase("angular-report") ||
		isHeadlessCase("full-report");
}

static bool isDynamicReportCase()
{
	return isHeadlessCase("dynamic-report");
}

static bool isRaycastReportCase()
{
	return isHeadlessCase("raycast-report");
}

static bool isExtendedReportCase()
{
	return isAngularReportCase() ||
		isDynamicReportCase() ||
		isRaycastReportCase();
}

static bool usesLinearCCD()
{
	return !gHeadlessOptions.headless ||
		isHeadlessCase("ccd-report") ||
		isHeadlessCase("full-report") ||
		isDynamicReportCase();
}

static bool usesSpeculativeCCD()
{
	return isHeadlessCase("angular-report") ||
		isHeadlessCase("full-report");
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
		!isHeadlessCase("no-ccd-control") &&
		!isHeadlessCase("angular-report") &&
		!isHeadlessCase("full-report") &&
		!isHeadlessCase("raycast-report") &&
		!isHeadlessCase("dynamic-report"))
	{
		error = "unsupported --case (expected ccd-report, "
			"no-ccd-control, angular-report, full-report, "
			"raycast-report, or dynamic-report)";
		return false;
	}
	const PxU32 expectedFrames = isAngularReportCase() ? 24u :
		(isRaycastReportCase() || isDynamicReportCase() ? 2u : 1u);
	if(gHeadlessOptions.frames != expectedFrames)
	{
		error = isAngularReportCase() ?
			"--frames must be 24 for angular/full report cases" :
			(isRaycastReportCase() || isDynamicReportCase() ?
				"--frames must be 2 for raycast/dynamic report cases" :
				"--frames must be 1 for this report case");
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
	if(usesLinearCCD())
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
				 pairHeader.actors[1] == gReportTargetActor) ||
				(pairHeader.actors[1] == gSphereActor &&
				 pairHeader.actors[0] == gReportTargetActor);
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
					const PxU32 sourceIndex =
						pairHeader.actors[0] == gSphereActor ? 0u : 1u;
					spherePose =
						iter.eventPose->globalPose[sourceIndex];

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

static void recordHeadlessCcdFlags()
{
	if(!gScene || !gSphereActor)
		return;

	gMetrics.sceneLinearCCD =
		(gScene->getFlags() & PxSceneFlag::eENABLE_CCD) ? 1u : 0u;
	const PxRigidBodyFlags sourceFlags =
		gSphereActor->getRigidBodyFlags();
	gMetrics.sourceLinearCCD =
		(sourceFlags & PxRigidBodyFlag::eENABLE_CCD) ? 1u : 0u;
	gMetrics.sourceSpeculativeCCD =
		(sourceFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD) ?
			1u : 0u;
	if(gDynamicTargetActor)
	{
		const PxRigidBodyFlags targetFlags =
			gDynamicTargetActor->getRigidBodyFlags();
		gMetrics.targetLinearCCD =
			(targetFlags & PxRigidBodyFlag::eENABLE_CCD) ? 1u : 0u;
		gMetrics.targetSpeculativeCCD =
			(targetFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD) ?
				1u : 0u;
		gMetrics.targetDynamic = 1;
	}
}

static bool registerForRaycastCcd(PxRigidDynamic* actor)
{
	if(!actor || !gRaycastCCD)
		return false;

	PxShape* shape = NULL;
	if(actor->getShapes(&shape, 1) != 1 || !shape)
		return false;
	return gRaycastCCD->registerRaycastCCDObject(actor, shape);
}

static void initExtendedHeadlessScene()
{
	if(isAngularReportCase())
	{
		gMaterial->setRestitution(0.0f);
		gSphereActor = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f)),
			PxBoxGeometry(10.0f, 1.0f, 0.1f), *gMaterial, 10.0f);
		gDynamicTargetActor = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f, 0.0f, 10.0f)),
			PxBoxGeometry(0.1f, 1.0f, 1.0f), *gMaterial, 10.0f);
		gReportTargetActor = gDynamicTargetActor;
		if(!gSphereActor || !gDynamicTargetActor)
			return;

		gSphereActor->setLinearDamping(0.0f);
		gSphereActor->setAngularDamping(0.0f);
		gSphereActor->setAngularVelocity(PxVec3(0.0f, 10.0f, 0.0f));
		gSphereActor->setSleepThreshold(0.0f);
		gSphereActor->setRigidDynamicLockFlags(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
		gDynamicTargetActor->setLinearDamping(0.0f);
		gDynamicTargetActor->setAngularDamping(0.0f);
		gDynamicTargetActor->setSleepThreshold(0.0f);

		if(usesLinearCCD())
		{
			gSphereActor->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_CCD, true);
			gDynamicTargetActor->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_CCD, true);
		}
		if(usesSpeculativeCCD())
		{
			gSphereActor->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
			gDynamicTargetActor->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		}
	}
	else
	{
		if(isDynamicReportCase())
		{
			gDynamicTargetActor = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(0.0f)),
				PxBoxGeometry(4.0f, 4.0f, 0.1f),
				*gMaterial, 1000.0f);
			gReportTargetActor = gDynamicTargetActor;
			if(gDynamicTargetActor)
			{
				gDynamicTargetActor->setLinearDamping(0.0f);
				gDynamicTargetActor->setAngularDamping(0.0f);
				gDynamicTargetActor->setSleepThreshold(0.0f);
				gDynamicTargetActor->setRigidDynamicLockFlags(
					PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
					PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
					PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
					PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
					PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
					PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
			}
		}
		else
		{
			gStaticTargetActor = PxCreateStatic(
				*gPhysics, PxTransform(PxVec3(0.0f)),
				PxBoxGeometry(4.0f, 4.0f, 0.1f), *gMaterial);
			gReportTargetActor = gStaticTargetActor;
		}

		gSphereActor = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f, 0.0f, 20.0f)),
			PxSphereGeometry(0.5f), *gMaterial, 10.0f);
		if(!gReportTargetActor || !gSphereActor)
			return;
		gSphereActor->setLinearDamping(0.0f);
		gSphereActor->setAngularDamping(0.0f);
		gSphereActor->setLinearVelocity(PxVec3(0.0f, 0.0f, -1000.0f));
		gSphereActor->setSleepThreshold(0.0f);
		if(usesLinearCCD())
			gSphereActor->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_CCD, true);

		if(isRaycastReportCase())
		{
			gRaycastCCD = new RaycastCCDManager(gScene);
			if(registerForRaycastCcd(gSphereActor))
				++gMetrics.raycastRegistrations;
		}
	}

	if(!gSphereActor || !gReportTargetActor)
		return;
	gMetrics.initialPosition = gSphereActor->getGlobalPose().p;
	gMetrics.targetInitialPosition =
		gReportTargetActor->getGlobalPose().p;
	gContactSphereActorPositions.pushBack(gMetrics.initialPosition);
	gScene->addActor(*gReportTargetActor);
	gScene->addActor(*gSphereActor);
	recordHeadlessCcdFlags();
}

static void initScene()
{
	if(gHeadlessOptions.headless && isExtendedReportCase())
	{
		initExtendedHeadlessScene();
		return;
	}

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
	gReportTargetActor = gTriangleMeshActor;

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
	recordHeadlessCcdFlags();
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
	if(usesLinearCCD())
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
	if (gSimStepCount < gHeadlessOptions.frames)
	{
		const PxReal dt = gHeadlessOptions.headless ?
			gHeadlessOptions.dt : 1.0f/60.0f;
		gScene->simulate(dt);
		const bool fetched = gScene->fetchResults(true);
		PxTransform raycastPoseBefore(PxIdentity);
		if(gHeadlessOptions.headless && gRaycastCCD && gSphereActor)
			raycastPoseBefore = gSphereActor->getGlobalPose();
		if(gRaycastCCD)
			gRaycastCCD->doRaycastCCD(true);
		if(gHeadlessOptions.headless)
		{
			if(!fetched)
				++gMetrics.fetchFailures;
			++gMetrics.completedFrames;
			if(gRaycastCCD && gSphereActor &&
				(gSphereActor->getGlobalPose().p -
				 raycastPoseBefore.p).magnitudeSquared() > 1.0e-12f)
				++gMetrics.raycastCorrections;
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
		if(gHeadlessOptions.headless && gReportTargetActor)
		{
			const PxTransform targetPose =
				gReportTargetActor->getGlobalPose();
			gMetrics.targetDisplacement =
				(targetPose.p -
				 gMetrics.targetInitialPosition).magnitude();
			if(!targetPose.isFinite() ||
				!PxIsFinite(gMetrics.targetDisplacement))
				++gMetrics.nonFinite;
		}

		++gSimStepCount;
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	gContactPositions.reset();
	gContactImpulses.reset();
	gContactSphereActorPositions.reset();

	PX_DELETE(gRaycastCCD);
	PX_RELEASE(gSphereActor);
	PX_RELEASE(gDynamicTargetActor);
	PX_RELEASE(gStaticTargetActor);
	PX_RELEASE(gTriangleMeshActor);
	gReportTargetActor = NULL;
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
		!gSphereActor && !gDynamicTargetActor && !gStaticTargetActor &&
		!gTriangleMeshActor && !gReportTargetActor && !gTriangleMesh &&
		!gRaycastCCD && !gScene && !gMaterial && !gDispatcher &&
		!gPhysics && !gPvd && !gFoundation &&
		!gExtensionsInitialized ? 1u : 0u;

	printf("SnippetContactReportCCD done.\n");
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig(
		"SnippetContactReportCCD", gHeadlessOptions);

	initPhysics(false);
	const bool baseInitialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gSphereActor;
	const bool initialized = baseInitialized &&
		(isExtendedReportCase() ?
			(gReportTargetActor &&
			 (!isRaycastReportCase() || gRaycastCCD)) :
			(gTriangleMesh && gTriangleMeshActor));
	if(initialized)
	{
		while(gMetrics.completedFrames < gHeadlessOptions.frames)
			stepPhysics(false);
	}

	const bool controlCase = isHeadlessCase("no-ccd-control");
	const bool legacyReportCase = isHeadlessCase("ccd-report");
	const bool angularReportCase = isAngularReportCase();
	const bool raycastReportCase = isRaycastReportCase();
	const bool dynamicReportCase = isDynamicReportCase();
	const bool tunneled = isExtendedReportCase() ?
		gMetrics.finalPosition.z < -0.6f :
		(gMetrics.finalPosition.x < -8.0f &&
		 gMetrics.finalPosition.y < 0.0f);
	const PxU32 expectedSceneLinear = usesLinearCCD() ? 1u : 0u;
	const PxU32 expectedSourceLinear = usesLinearCCD() ? 1u : 0u;
	const PxU32 expectedSourceSpeculative =
		usesSpeculativeCCD() ? 1u : 0u;
	const PxU32 expectedTargetLinear =
		angularReportCase && usesLinearCCD() ? 1u : 0u;
	const PxU32 expectedTargetSpeculative =
		angularReportCase && usesSpeculativeCCD() ? 1u : 0u;
	const PxU32 expectedTargetDynamic =
		(angularReportCase || dynamicReportCase) ? 1u : 0u;
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
	else if(gMetrics.sceneLinearCCD != expectedSceneLinear ||
		gMetrics.sourceLinearCCD != expectedSourceLinear ||
		gMetrics.sourceSpeculativeCCD != expectedSourceSpeculative ||
		gMetrics.targetLinearCCD != expectedTargetLinear ||
		gMetrics.targetSpeculativeCCD !=
			expectedTargetSpeculative ||
		gMetrics.targetDynamic != expectedTargetDynamic)
	{
		passed = false;
		reason = "ccd_flag_readback_mismatch";
	}
	else if(controlCase &&
		(gMetrics.callbackCount != 0 ||
		 gMetrics.ccdEventCount != 0 ||
		 gMetrics.contactPointCount != 0 ||
		 gMetrics.eventPoseCount != 0 ||
		 !tunneled))
	{
		passed = false;
		reason = "negative_control_failed";
	}
	else if(legacyReportCase &&
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
	else if(legacyReportCase &&
		gMetrics.nonzeroImpulseCount < 3)
	{
		passed = false;
		reason = "missing_ccd_impulse";
	}
	else if(legacyReportCase && tunneled)
	{
		passed = false;
		reason = "complete_tunneling";
	}
	else if(angularReportCase &&
		(gMetrics.callbackCount == 0 ||
		 gMetrics.pairCount == 0 ||
		 gMetrics.foundEventCount == 0 ||
		 gMetrics.contactPointCount == 0 ||
		 gMetrics.nonzeroImpulseCount == 0 ||
		 gMetrics.eventPoseCount == 0 ||
		 gMetrics.targetDisplacement <= 0.05f))
	{
		passed = false;
		reason = "missing_speculative_report";
	}
	else if(isHeadlessCase("angular-report") &&
		gMetrics.ccdEventCount != 0)
	{
		passed = false;
		reason = "unexpected_sweep_ccd_event";
	}
	else if(isHeadlessCase("full-report") &&
		gMetrics.ccdEventCount == 0)
	{
		passed = false;
		reason = "missing_full_sweep_ccd_event";
	}
	else if(dynamicReportCase &&
		(gMetrics.callbackCount == 0 ||
		 gMetrics.pairCount == 0 ||
		 gMetrics.ccdEventCount == 0 ||
		 gMetrics.contactPointCount == 0 ||
		 gMetrics.nonzeroImpulseCount == 0 ||
		 gMetrics.eventPoseCount == 0 ||
		 tunneled))
	{
		passed = false;
		reason = "missing_dynamic_ccd_report";
	}
	else if(raycastReportCase &&
		(gMetrics.raycastRegistrations != 1 ||
		 gMetrics.raycastCorrections == 0 ||
		 gMetrics.callbackCount != 0 ||
		 gMetrics.pairCount != 0 ||
		 gMetrics.foundEventCount != 0 ||
		 gMetrics.ccdEventCount != 0 ||
		 gMetrics.contactPointCount != 0 ||
		 gMetrics.eventPoseCount != 0 ||
		 tunneled))
	{
		passed = false;
		reason = "raycast_report_boundary_failed";
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
		"[AVBD_GATE] schema=2 snippet=SnippetContactReportCCD solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED callbackCount=%u pairCount=%u "
		"foundEventCount=%u ccdEventCount=%u contactPointCount=%u "
		"nonzeroImpulseCount=%u eventPoseCount=%u identityErrors=%u "
		"impulseSum=%.9g maxImpulse=%.9g targetDisplacement=%.9g "
		"initialX=%.9g initialY=%.9g initialZ=%.9g "
		"finalX=%.9g finalY=%.9g finalZ=%.9g tunneled=%u "
		"sceneLinearCCD=%u sourceLinearCCD=%u "
		"sourceSpeculativeCCD=%u targetLinearCCD=%u "
		"targetSpeculativeCCD=%u targetDynamic=%u "
		"raycastRegistrations=%u raycastCorrections=%u nonFinite=%u "
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
		double(gMetrics.targetDisplacement),
		double(gMetrics.initialPosition.x), double(gMetrics.initialPosition.y),
		double(gMetrics.initialPosition.z),
		double(gMetrics.finalPosition.x), double(gMetrics.finalPosition.y),
		double(gMetrics.finalPosition.z), tunneled ? 1u : 0u,
		gMetrics.sceneLinearCCD, gMetrics.sourceLinearCCD,
		gMetrics.sourceSpeculativeCCD, gMetrics.targetLinearCCD,
		gMetrics.targetSpeculativeCCD, gMetrics.targetDynamic,
		gMetrics.raycastRegistrations, gMetrics.raycastCorrections,
		gMetrics.nonFinite,
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
