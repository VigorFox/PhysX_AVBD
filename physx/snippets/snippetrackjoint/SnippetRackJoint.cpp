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
// This snippet illustrates simple use of rack & pinion joints
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "extensions/PxCollectionExt.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;
static PxPvdTransport*			gPvdTransport = NULL;
static PxRigidDynamic*			gPinionActor = NULL;
static PxRigidDynamic*			gRackActor = NULL;
static PxRevoluteJoint*			gHinge0 = NULL;
static PxPrismaticJoint*		gPrismatic = NULL;
static PxRackAndPinionJoint*	gRackJoint = NULL;
static PxSerializationRegistry*	gSerializationRegistry = NULL;
static PxCollection*			gLoadedCollection = NULL;
static PxU8*						gBinaryBlockRaw = NULL;
static void*					gBinaryBlockAligned = NULL;
static PxU64					gBinaryBlockSize = 0;

enum RackHeadlessCase
{
	ePINION_IMPULSE,
	eRACK_IMPULSE
};

enum RackSerializationMode
{
	eRACK_RUNTIME,
	eRACK_BINARY
};

struct RackHeadlessConfig
{
	PxReal	ratio;
	PxU32	impulseFrame;
	RackSerializationMode serializationMode;

	RackHeadlessConfig()
	: ratio(2.0f), impulseFrame(30), serializationMode(eRACK_RUNTIME)
	{
	}
};

struct RackMetrics
{
	PxU32	simulateCalls;
	PxU32	fetchCalls;
	PxU32	completedFrames;
	PxU32	fetchFailures;
	PxU32	nonFinite;
	PxU32	impulseEvents;
	PxU32	responseSamples;
	PxU32	directionViolations;
	PxReal	ratioReadback;
	PxReal	coefficient;
	PxReal	peakPinionSpeed;
	PxReal	peakRackSpeed;
	PxReal	maxVelocityResidual;
	PxReal	velocityResidualSquaredSum;
	PxReal	maxPositionError;
	PxReal	expectedPinionSpeed;
	PxReal	expectedRackSpeed;
	PxReal	actualPinionSpeed;
	PxReal	actualRackSpeed;
	PxReal	impulseProjectionError;
	PxU32	projectionSamples;
	PxReal	finalPinionAngle;
	PxReal	finalRackPosition;
	PxU32	serializationRequested;
	PxU32	registryCreated;
	PxU32	collectionCompleted;
	PxU32	serializable;
	PxU32	serializeSuccess;
	PxU64	serializedBytes;
	PxU32	binaryBlockAllocated;
	PxU32	binaryAligned;
	PxU32	deserializeSuccess;
	PxU32	loadedObjects;
	PxU32	loadedActors;
	PxU32	loadedConstraints;
	PxU32	loadedRevolute;
	PxU32	loadedPrismatic;
	PxU32	loadedRack;
	PxU32	dependencyIdentity;
	PxU32	actorIdentity;
	PxU32	sceneActors;
	PxU32	sceneConstraints;
	PxU32	authoringReleased;
	PxU32	loadedCollectionReleased;
	PxU32	binaryBlockFreed;
	PxU32	cleanupComplete;

	RackMetrics()
	: simulateCalls(0), fetchCalls(0), completedFrames(0), fetchFailures(0),
	  nonFinite(0), impulseEvents(0), responseSamples(0),
	  directionViolations(0), ratioReadback(0.0f), coefficient(0.0f),
	  peakPinionSpeed(0.0f), peakRackSpeed(0.0f),
	  maxVelocityResidual(0.0f), velocityResidualSquaredSum(0.0f),
	  maxPositionError(0.0f), expectedPinionSpeed(0.0f),
	  expectedRackSpeed(0.0f), actualPinionSpeed(0.0f),
	  actualRackSpeed(0.0f), impulseProjectionError(PX_MAX_F32),
	  projectionSamples(0), finalPinionAngle(0.0f),
	  finalRackPosition(0.0f), serializationRequested(0),
	  registryCreated(0), collectionCompleted(0), serializable(0),
	  serializeSuccess(0), serializedBytes(0), binaryBlockAllocated(0),
	  binaryAligned(0), deserializeSuccess(0), loadedObjects(0),
	  loadedActors(0), loadedConstraints(0), loadedRevolute(0),
	  loadedPrismatic(0), loadedRack(0), dependencyIdentity(0),
	  actorIdentity(0), sceneActors(0), sceneConstraints(0),
	  authoringReleased(0), loadedCollectionReleased(0),
	  binaryBlockFreed(0), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions	gHeadlessOptions;
static RackHeadlessConfig			gHeadlessConfig;
static RackHeadlessCase				gHeadlessCase = ePINION_IMPULSE;
static RackMetrics					gMetrics;
static PxVec3						gPinionAxis(0.0f, 0.0f, 1.0f);
static PxVec3						gRackAxis(1.0f, 0.0f, 0.0f);
static PxReal						gPersistentPinionAngle = 0.0f;
static PxReal						gVirtualPinionAngle = 0.0f;
static bool							gPinionAngleInitialized = false;
static const char*					gSerializationFailureReason = "none";

static const PxU32 gResponseDelayFrames = 2;
static const PxU32 gResponseWindowFrames = 60;
static const PxReal gImpulseMagnitude = 1.0f;
static const PxReal gMinimumEndpointSpeed = 0.05f;
static const PxReal gImpulseProjectionErrorCap = 0.02f;
static const PxReal gVelocityResidualCap = 0.08f;
static const PxReal gPositionErrorCap = 0.05f;
static const PxSerialObjectId gPinionActorId = PxSerialObjectId(0x1001);
static const PxSerialObjectId gRackActorId = PxSerialObjectId(0x1002);
static const PxSerialObjectId gHingeJointId = PxSerialObjectId(0x2001);
static const PxSerialObjectId gPrismaticJointId = PxSerialObjectId(0x2002);
static const PxSerialObjectId gRackJointId = PxSerialObjectId(0x2003);

static bool isFinite(const PxVec3& value)
{
	return value.isFinite();
}

static PxReal wrappedAngleTravel(PxReal current, PxReal previous)
{
	const PxReal diff =
		fmodf(previous - current + PxPi, PxTwoPi) - PxPi;
	return diff < -PxPi ? diff + PxTwoPi : diff;
}

static const char* getRackCaseName()
{
	return gHeadlessCase == eRACK_IMPULSE ? "rack-impulse" : "pinion-impulse";
}

static const char* getSerializationModeName()
{
	return gHeadlessConfig.serializationMode == eRACK_BINARY
		? "binary"
		: "runtime";
}

static bool parseSerializationMode(const char* value,
								   RackSerializationMode& result)
{
	if(Snippets::equalsIgnoreCase(value, "runtime"))
	{
		result = eRACK_RUNTIME;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "binary"))
	{
		result = eRACK_BINARY;
		return true;
	}
	return false;
}

static bool parseRackCase(const char* value, RackHeadlessCase& result)
{
	if(Snippets::equalsIgnoreCase(value, "pinion-impulse"))
	{
		result = ePINION_IMPULSE;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "rack-impulse"))
	{
		result = eRACK_IMPULSE;
		return true;
	}
	return false;
}

static int reportConfigurationError(const char* reason)
{
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetRackJoint solver=%s "
		"case=config-error execution=%s frames=%u completedFrames=0 "
		"status=ERROR reason=%s serialization=%s ratio=%.9g "
		"impulseFrame=%u\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, reason, getSerializationModeName(),
		double(gHeadlessConfig.ratio),
		gHeadlessConfig.impulseFrame);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

static bool parseHeadlessOptions(int argc, const char*const* argv,
								std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.dt = 1.0f / 60.0f;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.caseName = "pinion-impulse";
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults,
											gHeadlessOptions, error))
		return false;

	bool ratioSeen = false;
	bool impulseFrameSeen = false;
	bool serializationSeen = false;
	for(int i=1; i<argc; i++)
	{
		const char* arg = argv[i];
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		if(Snippets::hasOptionPrefix(arg, "--ratio="))
		{
			if(ratioSeen)
			{
				error = "duplicate_--ratio";
				return false;
			}
			ratioSeen = true;
			if(!Snippets::parseReal(arg + std::strlen("--ratio="), -4.0f, 4.0f,
								   gHeadlessConfig.ratio) ||
			   PxAbs(gHeadlessConfig.ratio) < 0.25f)
			{
				error = "invalid_--ratio_value";
				return false;
			}
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--impulse-frame="))
		{
			if(impulseFrameSeen)
			{
				error = "duplicate_--impulse-frame";
				return false;
			}
			impulseFrameSeen = true;
			if(!Snippets::parseU32(arg + std::strlen("--impulse-frame="), 1,
								  100000000u, gHeadlessConfig.impulseFrame))
			{
				error = "invalid_--impulse-frame_value";
				return false;
			}
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--serialization="))
		{
			if(serializationSeen)
			{
				error = "duplicate_--serialization";
				return false;
			}
			serializationSeen = true;
			if(!parseSerializationMode(
					arg + std::strlen("--serialization="),
					gHeadlessConfig.serializationMode))
			{
				error = "invalid_--serialization_value";
				return false;
			}
			continue;
		}
		error = "unknown_argument";
		return false;
	}

	if(!parseRackCase(gHeadlessOptions.caseName.c_str(), gHeadlessCase))
	{
		error = "invalid_--case_value";
		return false;
	}
	if(gHeadlessOptions.solverType == PxSolverType::ePGS)
	{
		error = "headless_solver_requires_avbd_or_tgs";
		return false;
	}
	if(gHeadlessOptions.execution == Snippets::eHEADLESS_SEQUENTIAL &&
	   gHeadlessOptions.solverType != PxSolverType::eAVBD)
	{
		error = "sequential_execution_requires_avbd";
		return false;
	}
	if(PxAbs(gHeadlessOptions.dt - 1.0f/60.0f) > 1e-6f)
	{
		error = "dt_requires_60hz_calibration";
		return false;
	}
	if(gHeadlessConfig.impulseFrame + gResponseDelayFrames +
		   gResponseWindowFrames >= gHeadlessOptions.frames)
	{
		error = "frames_require_complete_response_window";
		return false;
	}
	return true;
}

static PxRigidDynamic* createGearWithBoxes(PxPhysics& sdk, const PxBoxGeometry& boxGeom, const PxTransform& transform, PxMaterial& material, int nbShapes)
{
	PxRigidDynamic* actor = sdk.createRigidDynamic(transform);

	PxMat33 m(PxIdentity);

	for(int i=0;i<nbShapes;i++)
	{
		const float coeff = float(i)/float(nbShapes);
		const float angle = PxPi * 0.5f * coeff;

		PxShape* shape = sdk.createShape(boxGeom, material, true);

		const PxReal cos = cosf(angle);
		const PxReal sin = sinf(angle);

		m[0][0] = m[1][1] = cos;
		m[0][1] = sin;
		m[1][0] = -sin;

		PxTransform localPose;
		localPose.p = PxVec3(0.0f);
		localPose.q = PxQuat(m);

		shape->setLocalPose(localPose);

		actor->attachShape(*shape);
	}
	PxRigidBodyExt::updateMassAndInertia(*actor, 1.0f);

	return actor;
}

static PxRigidDynamic* createRackWithBoxes(PxPhysics& sdk, const PxTransform& transform, PxMaterial& material, int nbTeeth, float rackLength)
{
	PxRigidDynamic* actor = sdk.createRigidDynamic(transform);

	{
		const PxBoxGeometry boxGeom(rackLength*0.5f, 0.25f, 0.25f);
		PxShape* shape = sdk.createShape(boxGeom, material, true);
		actor->attachShape(*shape);
	}

	PxMat33 m(PxIdentity);
	const float angle = PxPi * 0.25f;
	const PxReal cos = cosf(angle);
	const PxReal sin = sinf(angle);
	m[0][0] = m[1][1] = cos;
	m[0][1] = sin;
	m[1][0] = -sin;

	PxTransform localPose;
	localPose.p = PxVec3(0.0f);
	localPose.q = PxQuat(m);

	const float offset = rackLength / float(nbTeeth);
	localPose.p.x = (offset - rackLength)*0.5f;

	for(int i=0;i<nbTeeth;i++)
	{
		const PxBoxGeometry boxGeom(0.75f, 0.75f, 0.25f);
		PxShape* shape = sdk.createShape(boxGeom, material, true);
		shape->setLocalPose(localPose);

		actor->attachShape(*shape);

		localPose.p.x += offset;
	}

	PxRigidBodyExt::updateMassAndInertia(*actor, 1.0f);

	return actor;
}

static void setSerializationFailure(const char* reason)
{
	if(std::strcmp(gSerializationFailureReason, "none") == 0)
		gSerializationFailureReason = reason;
}

static void releaseAuthoringFixture()
{
	PX_RELEASE(gRackJoint);
	PX_RELEASE(gPrismatic);
	PX_RELEASE(gHinge0);
	PX_RELEASE(gPinionActor);
	PX_RELEASE(gRackActor);
	gMetrics.authoringReleased =
		!gRackJoint && !gPrismatic && !gHinge0 &&
		!gPinionActor && !gRackActor;
}

template<typename T>
static T* findSerializedObject(PxSerialObjectId id)
{
	if(!gLoadedCollection)
		return NULL;
	PxBase* object = gLoadedCollection->find(id);
	return object ? object->is<T>() : NULL;
}

static bool replaceFixtureWithBinaryRoundTrip()
{
	gMetrics.serializationRequested = 1;
	gSerializationRegistry =
		PxSerialization::createSerializationRegistry(*gPhysics);
	if(!gSerializationRegistry)
	{
		setSerializationFailure("serialization_registry");
		return false;
	}
	gMetrics.registryCreated = 1;

	PxCollection* authoringCollection = PxCreateCollection();
	if(!authoringCollection)
	{
		setSerializationFailure("serialization_collection");
		return false;
	}

	authoringCollection->add(*gPinionActor, gPinionActorId);
	authoringCollection->add(*gRackActor, gRackActorId);
	authoringCollection->add(*gHinge0, gHingeJointId);
	authoringCollection->add(*gPrismatic, gPrismaticJointId);
	authoringCollection->add(*gRackJoint, gRackJointId);
	PxSerialization::complete(*authoringCollection,
							 *gSerializationRegistry);
	gMetrics.collectionCompleted = 1;
	gMetrics.serializable = PxSerialization::isSerializable(
		*authoringCollection, *gSerializationRegistry) ? 1u : 0u;
	if(!gMetrics.serializable)
	{
		authoringCollection->release();
		setSerializationFailure("serialization_not_serializable");
		return false;
	}

	PxDefaultMemoryOutputStream output;
	gMetrics.serializeSuccess =
		PxSerialization::serializeCollectionToBinary(
			output, *authoringCollection, *gSerializationRegistry)
			? 1u
			: 0u;
	gMetrics.serializedBytes = output.getSize();
	authoringCollection->release();
	if(!gMetrics.serializeSuccess || gMetrics.serializedBytes == 0)
	{
		setSerializationFailure("serialization_write");
		return false;
	}

	const PxU64 allocationSize =
		gMetrics.serializedBytes + PX_SERIAL_FILE_ALIGN - 1;
	if(allocationSize < gMetrics.serializedBytes ||
	   allocationSize >
		   static_cast<PxU64>(std::numeric_limits<size_t>::max()))
	{
		setSerializationFailure("serialization_allocation_size");
		return false;
	}
	gBinaryBlockRaw = static_cast<PxU8*>(
		std::malloc(static_cast<size_t>(allocationSize)));
	if(!gBinaryBlockRaw)
	{
		setSerializationFailure("serialization_allocation");
		return false;
	}
	gMetrics.binaryBlockAllocated = 1;
	const size_t rawAddress =
		reinterpret_cast<size_t>(gBinaryBlockRaw);
	const size_t alignedAddress =
		(rawAddress + PX_SERIAL_FILE_ALIGN - 1) &
		~(size_t(PX_SERIAL_FILE_ALIGN) - 1);
	gBinaryBlockAligned = reinterpret_cast<void*>(alignedAddress);
	gBinaryBlockSize = gMetrics.serializedBytes;
	gMetrics.binaryAligned =
		(alignedAddress & (PX_SERIAL_FILE_ALIGN - 1)) == 0 ? 1u : 0u;
	if(!gMetrics.binaryAligned)
	{
		setSerializationFailure("serialization_alignment");
		return false;
	}
	std::memcpy(gBinaryBlockAligned, output.getData(),
				static_cast<size_t>(gMetrics.serializedBytes));

	// Prove that the deserialized graph is self-contained. No authoring actor
	// or joint remains alive when pointer fixup and scene consumption occur.
	releaseAuthoringFixture();
	gLoadedCollection = PxSerialization::createCollectionFromBinary(
		gBinaryBlockAligned, *gSerializationRegistry);
	if(!gLoadedCollection)
	{
		setSerializationFailure("deserialization_failed");
		return false;
	}
	gMetrics.deserializeSuccess = 1;

	gPinionActor =
		findSerializedObject<PxRigidDynamic>(gPinionActorId);
	gRackActor =
		findSerializedObject<PxRigidDynamic>(gRackActorId);
	gHinge0 =
		findSerializedObject<PxRevoluteJoint>(gHingeJointId);
	gPrismatic =
		findSerializedObject<PxPrismaticJoint>(gPrismaticJointId);
	gRackJoint =
		findSerializedObject<PxRackAndPinionJoint>(gRackJointId);

	gMetrics.loadedObjects = gLoadedCollection->getNbObjects();
	for(PxU32 i = 0; i < gMetrics.loadedObjects; ++i)
	{
		PxBase& object = gLoadedCollection->getObject(i);
		gMetrics.loadedActors += object.is<PxRigidDynamic>() ? 1u : 0u;
		gMetrics.loadedConstraints += object.is<PxConstraint>() ? 1u : 0u;
		gMetrics.loadedRevolute +=
			object.is<PxRevoluteJoint>() ? 1u : 0u;
		gMetrics.loadedPrismatic +=
			object.is<PxPrismaticJoint>() ? 1u : 0u;
		gMetrics.loadedRack +=
			object.is<PxRackAndPinionJoint>() ? 1u : 0u;
	}
	if(!gPinionActor || !gRackActor || !gHinge0 ||
	   !gPrismatic || !gRackJoint ||
	   gMetrics.loadedActors != 2 ||
	   gMetrics.loadedConstraints != 3 ||
	   gMetrics.loadedRevolute != 1 ||
	   gMetrics.loadedPrismatic != 1 ||
	   gMetrics.loadedRack != 1)
	{
		setSerializationFailure("deserialized_types");
		return false;
	}
	return true;
}

static void validateFixtureIdentity()
{
	if(!gPinionActor || !gRackActor || !gHinge0 ||
	   !gPrismatic || !gRackJoint)
		return;

	const PxBase* hinge = NULL;
	const PxBase* prismatic = NULL;
	gRackJoint->getJoints(hinge, prismatic);
	gMetrics.dependencyIdentity =
		hinge == gHinge0 && prismatic == gPrismatic ? 1u : 0u;

	PxRigidActor* actor0 = NULL;
	PxRigidActor* actor1 = NULL;
	gHinge0->getActors(actor0, actor1);
	const bool hingeIdentity =
		actor0 == NULL && actor1 == gPinionActor;
	gPrismatic->getActors(actor0, actor1);
	const bool prismaticIdentity =
		actor0 == NULL && actor1 == gRackActor;
	gRackJoint->getActors(actor0, actor1);
	const bool rackIdentity =
		actor0 == gPinionActor && actor1 == gRackActor;
	gMetrics.actorIdentity =
		hingeIdentity && prismaticIdentity && rackIdentity ? 1u : 0u;
}

void initPhysics(bool interactive)
{
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return;

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		gPvdTransport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && gPvdTransport)
			gPvd->connect(*gPvdTransport, PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(),true, gPvd);
	if(!gPhysics)
		return;
	PxInitExtensions(*gPhysics, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = interactive ? PxVec3(0.0f, -9.81f, 0.0f) : PxVec3(0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		interactive ? 2u : gHeadlessOptions.dispatcherThreads);
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	sceneDesc.solverType = interactive ? PxSolverType::eAVBD
									  : gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return;

	if(interactive)
	{
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if(pvdClient)
		{
			pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
			pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
			pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
		}
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
		return;

	const float radius = 3.0f;
	const float rackLength = 3.0f*10.0f;
	const int nbPinionTeeth = int(radius)*4;	// 'radius' teeth for PI/2
	const int nbRackTeeth = 5*3;
	const PxVec3 boxPos0 = interactive ? PxVec3(0.0f, 10.0f, 0.0f)
									 : PxVec3(0.0f, 0.0f, 0.0f);
	const PxVec3 boxPos1 = interactive
		? PxVec3(0.0f, 10.0f+radius+1.5f, 0.0f)
		: PxVec3(0.0f, 2.0f, 0.0f);

	if(interactive)
	{
		const PxBoxGeometry boxGeom0(radius, radius, 0.5f);
		gPinionActor = createGearWithBoxes(*gPhysics, boxGeom0,
										  PxTransform(boxPos0), *gMaterial,
										  int(radius));
		gRackActor = createRackWithBoxes(*gPhysics, PxTransform(boxPos1),
										*gMaterial, nbRackTeeth, rackLength);
	}
	else
	{
		const PxBoxGeometry isolatedBox(0.5f, 0.5f, 0.5f);
		gPinionActor = PxCreateDynamic(*gPhysics, PxTransform(boxPos0),
									 isolatedBox, *gMaterial, 1.0f);
		gRackActor = PxCreateDynamic(*gPhysics, PxTransform(boxPos1),
									isolatedBox, *gMaterial, 1.0f);
		if(gPinionActor)
		{
			gPinionActor->setAngularDamping(0.0f);
			gPinionActor->setLinearDamping(0.0f);
			gPinionActor->setSleepThreshold(0.0f);
			gPinionActor->setSolverIterationCounts(16, 4);
		}
		if(gRackActor)
		{
			gRackActor->setAngularDamping(0.0f);
			gRackActor->setLinearDamping(0.0f);
			gRackActor->setSleepThreshold(0.0f);
			gRackActor->setSolverIterationCounts(16, 4);
		}
	}
	if(!gPinionActor || !gRackActor)
		return;

	const PxQuat x2z = PxShortestRotation(PxVec3(1.0f, 0.0f, 0.0f), PxVec3(0.0f, 0.0f, 1.0f));

	gHinge0 = PxRevoluteJointCreate(*gPhysics, NULL,
		PxTransform(boxPos0, x2z), gPinionActor,
		PxTransform(PxVec3(0.0f), x2z));
	gPrismatic = PxPrismaticJointCreate(*gPhysics, NULL,
		PxTransform(boxPos1), gRackActor, PxTransform(PxVec3(0.0f)));
	gRackJoint = PxRackAndPinionJointCreate(*gPhysics, gPinionActor,
		PxTransform(PxVec3(0.0f), x2z), gRackActor,
		PxTransform(PxVec3(0.0f)));
	if(!gHinge0 || !gPrismatic || !gRackJoint)
		return;

	if(interactive)
	{
		gHinge0->setDriveVelocity(0.5f);
		gHinge0->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED, true);
	}

	if(!gRackJoint->setJoints(gHinge0, gPrismatic))
		return;
	if(interactive)
		gRackJoint->setData(nbRackTeeth, nbPinionTeeth, rackLength);
	else
		gRackJoint->setRatio(gHeadlessConfig.ratio);

	if(!interactive &&
	   gHeadlessConfig.serializationMode == eRACK_BINARY)
	{
		if(!replaceFixtureWithBinaryRoundTrip())
			return;
		if(!gScene->addCollection(*gLoadedCollection))
		{
			setSerializationFailure("deserialized_scene_add");
			return;
		}
	}
	else
	{
		gScene->addActor(*gPinionActor);
		gScene->addActor(*gRackActor);
	}

	if(!interactive)
	{
		validateFixtureIdentity();
		gMetrics.sceneActors = gScene->getNbActors(
			PxActorTypeFlag::eRIGID_DYNAMIC);
		gMetrics.sceneConstraints = gScene->getNbConstraints();
		if(gHeadlessConfig.serializationMode == eRACK_BINARY &&
		   (gMetrics.dependencyIdentity != 1 ||
			gMetrics.actorIdentity != 1))
			setSerializationFailure("deserialized_identity");
		if(gHeadlessConfig.serializationMode == eRACK_BINARY &&
		   (gMetrics.sceneActors != 2 ||
			gMetrics.sceneConstraints != 3))
			setSerializationFailure("deserialized_scene");

		const PxTransform cA2w =
			gPinionActor->getGlobalPose() * PxTransform(PxVec3(0.0f), x2z);
		const PxTransform cB2w = gRackActor->getGlobalPose();
		gPinionAxis = cA2w.q.getBasisVector0().getNormalized();
		gRackAxis = cB2w.q.getBasisVector0().getNormalized();
		const PxVec3 delta = cB2w.p - cA2w.p;
		const PxVec3 tangent = gPinionAxis.cross(delta);
		const PxReal projection = tangent.dot(gRackAxis);
		gMetrics.coefficient =
			PxAbs(projection) > 0.001f ? PxSign(projection) : 1.0f;
		gMetrics.ratioReadback = gRackJoint->getRatio();
	}
}

static void applyHeadlessExcitation()
{
	if(gMetrics.completedFrames != gHeadlessConfig.impulseFrame)
		return;

	if(gHeadlessCase == eRACK_IMPULSE)
		gRackActor->addForce(gRackAxis * gImpulseMagnitude,
							 PxForceMode::eIMPULSE);
	else
		gPinionActor->addTorque(gPinionAxis * gImpulseMagnitude,
							   PxForceMode::eIMPULSE);

	PxReal pinionSpeed =
		gPinionActor->getAngularVelocity().dot(gPinionAxis);
	PxReal rackSpeed =
		gRackActor->getLinearVelocity().dot(gRackAxis);
	const PxVec3 localPinionAxis =
		gPinionActor->getGlobalPose().q.rotateInv(gPinionAxis);
	const PxVec3 inertia = gPinionActor->getMassSpaceInertiaTensor();
	const PxReal pinionInverseInertia =
		localPinionAxis.x * localPinionAxis.x / inertia.x +
		localPinionAxis.y * localPinionAxis.y / inertia.y +
		localPinionAxis.z * localPinionAxis.z / inertia.z;
	const PxReal rackInverseMass = 1.0f / gRackActor->getMass();
	if(gHeadlessCase == eRACK_IMPULSE)
		rackSpeed += rackInverseMass * gImpulseMagnitude;
	else
		pinionSpeed += pinionInverseInertia * gImpulseMagnitude;
	const PxReal rackJacobian =
		-gMetrics.coefficient * gHeadlessConfig.ratio;
	const PxReal denominator =
		pinionInverseInertia +
		rackInverseMass * rackJacobian * rackJacobian;
	if(denominator > 1e-12f)
	{
		const PxReal impulse =
			-(pinionSpeed + rackJacobian * rackSpeed) / denominator;
		gMetrics.expectedPinionSpeed =
			pinionSpeed + pinionInverseInertia * impulse;
		gMetrics.expectedRackSpeed =
			rackSpeed + rackInverseMass * rackJacobian * impulse;
	}
	++gMetrics.impulseEvents;
}

static void sampleHeadlessState()
{
	if(!gPinionActor || !gRackActor || !gHinge0 || !gPrismatic)
		return;

	const PxTransform pinionPose = gPinionActor->getGlobalPose();
	const PxTransform rackPose = gRackActor->getGlobalPose();
	const PxVec3 pinionAngularVelocity = gPinionActor->getAngularVelocity();
	const PxVec3 rackLinearVelocity = gRackActor->getLinearVelocity();
	const PxReal pinionAngle = gHinge0->getAngle();
	const PxReal rackPosition = gPrismatic->getPosition();
	if(!pinionPose.isFinite() || !rackPose.isFinite() ||
	   !isFinite(pinionAngularVelocity) || !isFinite(rackLinearVelocity) ||
	   !PxIsFinite(pinionAngle) || !PxIsFinite(rackPosition))
	{
		++gMetrics.nonFinite;
		return;
	}

	if(!gPinionAngleInitialized)
	{
		gPinionAngleInitialized = true;
		gPersistentPinionAngle = pinionAngle;
	}
	gVirtualPinionAngle +=
		wrappedAngleTravel(pinionAngle, gPersistentPinionAngle);
	gPersistentPinionAngle = pinionAngle;

	gMetrics.finalPinionAngle = gVirtualPinionAngle;
	gMetrics.finalRackPosition = rackPosition;
	const PxReal pinionSpeed = pinionAngularVelocity.dot(gPinionAxis);
	const PxReal rackSpeed = rackLinearVelocity.dot(gRackAxis);
	if(gMetrics.completedFrames == gHeadlessConfig.impulseFrame + 1)
	{
		gMetrics.actualPinionSpeed = pinionSpeed;
		gMetrics.actualRackSpeed = rackSpeed;
		const PxReal projectionScale =
			PxMax(0.1f, PxAbs(gMetrics.expectedPinionSpeed) +
						  PxAbs(gMetrics.expectedRackSpeed));
		gMetrics.impulseProjectionError =
			(PxAbs(pinionSpeed - gMetrics.expectedPinionSpeed) +
			 PxAbs(rackSpeed - gMetrics.expectedRackSpeed)) /
			projectionScale;
		++gMetrics.projectionSamples;
	}

	const PxU32 responseStart = gHeadlessConfig.impulseFrame +
								gResponseDelayFrames + 1;
	const PxU32 responseEnd = responseStart + gResponseWindowFrames;
	if(gMetrics.completedFrames < responseStart ||
	   gMetrics.completedFrames >= responseEnd)
		return;

	gMetrics.peakPinionSpeed =
		PxMax(gMetrics.peakPinionSpeed, PxAbs(pinionSpeed));
	gMetrics.peakRackSpeed =
		PxMax(gMetrics.peakRackSpeed, PxAbs(rackSpeed));

	// Px1DConstraint publishes J={linear0, angular0, -linear1, -angular1}.
	// ExtRackAndPinion emits angular0=pinionAxis and
	// linear1=coefficient*ratio*rackAxis.
	const PxReal residual =
		pinionSpeed - gMetrics.coefficient * gHeadlessConfig.ratio * rackSpeed;
	const PxReal denominator =
		PxMax(0.1f, PxAbs(pinionSpeed) +
					  PxAbs(gHeadlessConfig.ratio * rackSpeed));
	const PxReal normalizedResidual = PxAbs(residual) / denominator;
	gMetrics.maxVelocityResidual =
		PxMax(gMetrics.maxVelocityResidual, normalizedResidual);
	gMetrics.velocityResidualSquaredSum +=
		normalizedResidual * normalizedResidual;

	const PxReal positionError =
		-gMetrics.coefficient * gHeadlessConfig.ratio * rackPosition -
		gVirtualPinionAngle;
	gMetrics.maxPositionError =
		PxMax(gMetrics.maxPositionError, PxAbs(positionError));

	if(PxAbs(pinionSpeed) > gMinimumEndpointSpeed &&
	   PxAbs(rackSpeed) > gMinimumEndpointSpeed &&
	   pinionSpeed * rackSpeed * gHeadlessConfig.ratio *
			   gMetrics.coefficient <= 0.0f)
		++gMetrics.directionViolations;
	++gMetrics.responseSamples;
}

void stepPhysics(bool interactive)
{
	if(interactive && gHinge0)
	{
		static float globalTime = 0.0f;
		globalTime += 1.0f/60.0f;
		gHinge0->setDriveVelocity(cosf(globalTime)*3.0f);
	}
	else if(!interactive)
		applyHeadlessExcitation();

	const PxReal dt = interactive ? 1.0f/60.0f : gHeadlessOptions.dt;
	++gMetrics.simulateCalls;
	gScene->simulate(dt);
	++gMetrics.fetchCalls;
	if(!gScene->fetchResults(true))
	{
		++gMetrics.fetchFailures;
		return;
	}
	++gMetrics.completedFrames;
	if(!interactive)
		sampleHeadlessState();
}

static void releaseLoadedFixture()
{
	if(gLoadedCollection)
	{
		std::vector<PxShape*> exclusiveShapes;
		for(PxU32 i = 0; i < gLoadedCollection->getNbObjects(); ++i)
		{
			PxShape* shape =
				gLoadedCollection->getObject(i).is<PxShape>();
			if(shape && shape->isExclusive())
			{
				shape->acquireReference();
				exclusiveShapes.push_back(shape);
			}
		}

		// Exclusive shapes normally have only their actor-held reference.
		// Keep them alive across actor teardown, skip the unsafe second release
		// in releaseObjects, then consume every remaining reference explicitly.
		PxCollectionExt::releaseObjects(*gLoadedCollection, false);
		for(PxU32 i = 0; i < exclusiveShapes.size(); ++i)
		{
			PxShape* shape = exclusiveShapes[i];
			while(shape->getReferenceCount() > 1)
				shape->release();
			shape->release();
		}
		gLoadedCollection->release();
		gLoadedCollection = NULL;
		gMetrics.loadedCollectionReleased = 1;
		gPinionActor = NULL;
		gRackActor = NULL;
		gHinge0 = NULL;
		gPrismatic = NULL;
		gRackJoint = NULL;
	}

	if(gBinaryBlockRaw)
	{
		std::free(gBinaryBlockRaw);
		gBinaryBlockRaw = NULL;
		gBinaryBlockAligned = NULL;
		gBinaryBlockSize = 0;
		gMetrics.binaryBlockFreed = 1;
	}
	PX_RELEASE(gSerializationRegistry);
}

void cleanupPhysics(bool interactive)
{
	if(gLoadedCollection)
		releaseLoadedFixture();
	else
	{
		PX_RELEASE(gRackJoint);
		PX_RELEASE(gPrismatic);
		PX_RELEASE(gHinge0);
		PX_RELEASE(gPinionActor);
		PX_RELEASE(gRackActor);
		if(gBinaryBlockRaw || gSerializationRegistry)
			releaseLoadedFixture();
	}
	PX_RELEASE(gMaterial);
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	if(gPhysics)
		PxCloseExtensions();
	PX_RELEASE(gPhysics);
	if(gPvd)
		PX_RELEASE(gPvd);
	PX_RELEASE(gPvdTransport);
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gRackJoint && !gPrismatic && !gHinge0 && !gPinionActor &&
		!gRackActor && !gMaterial && !gScene && !gDispatcher && !gPhysics &&
		!gPvd && !gPvdTransport && !gFoundation &&
		!gSerializationRegistry && !gLoadedCollection &&
		!gBinaryBlockRaw && !gBinaryBlockAligned &&
		gBinaryBlockSize == 0;
	if(interactive)
		printf("SnippetRackJoint done.\n");
}

static int runHeadless()
{
	Snippets::printHeadlessConfig("SnippetRackJoint", gHeadlessOptions);
	initPhysics(false);

	const bool initialized =
		gFoundation && gPhysics && gScene && gMaterial && gPinionActor &&
		gRackActor && gHinge0 && gPrismatic && gRackJoint;
	if(initialized)
	{
		for(PxU32 frame=0; frame<gHeadlessOptions.frames; frame++)
			stepPhysics(false);
	}

	const PxReal residualRms =
		gMetrics.responseSamples
			? PxSqrt(gMetrics.velocityResidualSquaredSum /
					 PxReal(gMetrics.responseSamples))
			: PX_MAX_F32;
	bool passed = true;
	const char* reason = "none";
	if(gHeadlessConfig.serializationMode == eRACK_BINARY &&
	   std::strcmp(gSerializationFailureReason, "none") != 0)
	{
		passed = false;
		reason = gSerializationFailureReason;
	}
	else if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(gMetrics.dependencyIdentity != 1 ||
			gMetrics.actorIdentity != 1)
	{
		passed = false;
		reason = "joint_identity";
	}
	else if(gMetrics.sceneActors != 2 ||
			gMetrics.sceneConstraints != 3)
	{
		passed = false;
		reason = "scene_identity";
	}
	else if(gHeadlessConfig.serializationMode == eRACK_BINARY &&
			(gMetrics.serializationRequested != 1 ||
			 gMetrics.registryCreated != 1 ||
			 gMetrics.collectionCompleted != 1 ||
			 gMetrics.serializable != 1 ||
			 gMetrics.serializeSuccess != 1 ||
			 gMetrics.serializedBytes == 0 ||
			 gMetrics.binaryBlockAllocated != 1 ||
			 gMetrics.binaryAligned != 1 ||
			 gMetrics.deserializeSuccess != 1 ||
			 gMetrics.loadedActors != 2 ||
			 gMetrics.loadedConstraints != 3 ||
			 gMetrics.loadedRevolute != 1 ||
			 gMetrics.loadedPrismatic != 1 ||
			 gMetrics.loadedRack != 1 ||
			 gMetrics.authoringReleased != 1))
	{
		passed = false;
		reason = "serialization_metrics";
	}
	else if(gMetrics.simulateCalls != gHeadlessOptions.frames ||
			gMetrics.fetchCalls != gHeadlessOptions.frames ||
			gMetrics.completedFrames != gHeadlessOptions.frames ||
			gMetrics.fetchFailures != 0 || gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_failed";
	}
	else if(gMetrics.nonFinite != 0)
	{
		passed = false;
		reason = "non_finite";
	}
	else if(PxAbs(gMetrics.ratioReadback - gHeadlessConfig.ratio) > 1e-6f)
	{
		passed = false;
		reason = "ratio_readback";
	}
	else if(PxAbs(gMetrics.coefficient) < 0.5f)
	{
		passed = false;
		reason = "invalid_geometry_coefficient";
	}
	else if(gMetrics.impulseEvents != 1 ||
			gMetrics.projectionSamples != 1)
	{
		passed = false;
		reason = "impulse_projection_missing";
	}
	else if(gMetrics.impulseProjectionError > gImpulseProjectionErrorCap)
	{
		passed = false;
		reason = "impulse_projection_error";
	}
	else if(
			gMetrics.responseSamples != gResponseWindowFrames ||
			gMetrics.peakPinionSpeed < gMinimumEndpointSpeed ||
			gMetrics.peakRackSpeed < gMinimumEndpointSpeed)
	{
		passed = false;
		reason = "missing_coupled_response";
	}
	else if(gMetrics.directionViolations != 0)
	{
		passed = false;
		reason = "wrong_coupling_direction";
	}
	else if(gMetrics.maxVelocityResidual > gVelocityResidualCap)
	{
		passed = false;
		reason = "velocity_ratio_residual";
	}
	else if(gMetrics.maxPositionError > gPositionErrorCap)
	{
		passed = false;
		reason = "position_ratio_error";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetRackJoint solver=%s case=%s "
		"execution=%s frames=%u completedFrames=%u status=%s reason=%s "
		"validation=GATED serialization=%s ratio=%.9g "
		"ratioReadback=%.9g coefficient=%.9g "
		"impulseFrame=%u impulseEvents=%u responseSamples=%u "
		"peakPinionSpeed=%.9g peakRackSpeed=%.9g directionViolations=%u "
		"velocityResidualMax=%.9g velocityResidualRms=%.9g "
		"positionErrorMax=%.9g finalPinionAngle=%.9g "
		"finalRackPosition=%.9g expectedPinionSpeed=%.9g "
		"expectedRackSpeed=%.9g actualPinionSpeed=%.9g "
		"actualRackSpeed=%.9g impulseProjectionError=%.9g "
		"projectionSamples=%u nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u serializationRequested=%u registryCreated=%u "
		"collectionCompleted=%u serializable=%u serializeSuccess=%u "
		"serializedBytes=%llu binaryBlockAllocated=%u binaryAligned=%u "
		"deserializeSuccess=%u loadedObjects=%u loadedActors=%u "
		"loadedConstraints=%u loadedRevolute=%u loadedPrismatic=%u "
		"loadedRack=%u dependencyIdentity=%u actorIdentity=%u "
		"sceneActors=%u sceneConstraints=%u authoringReleased=%u "
		"loadedCollectionReleased=%u binaryBlockFreed=%u "
		"cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		getRackCaseName(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, getSerializationModeName(),
		double(gHeadlessConfig.ratio),
		double(gMetrics.ratioReadback), double(gMetrics.coefficient),
		gHeadlessConfig.impulseFrame, gMetrics.impulseEvents,
		gMetrics.responseSamples, double(gMetrics.peakPinionSpeed),
		double(gMetrics.peakRackSpeed), gMetrics.directionViolations,
		double(gMetrics.maxVelocityResidual), double(residualRms),
		double(gMetrics.maxPositionError), double(gMetrics.finalPinionAngle),
		double(gMetrics.finalRackPosition),
		double(gMetrics.expectedPinionSpeed),
		double(gMetrics.expectedRackSpeed),
		double(gMetrics.actualPinionSpeed), double(gMetrics.actualRackSpeed),
		double(gMetrics.impulseProjectionError), gMetrics.projectionSamples,
		gMetrics.nonFinite,
		gMetrics.fetchFailures, gErrorCallback.getFatalCount(),
		gMetrics.serializationRequested, gMetrics.registryCreated,
		gMetrics.collectionCompleted, gMetrics.serializable,
		gMetrics.serializeSuccess,
		static_cast<unsigned long long>(gMetrics.serializedBytes),
		gMetrics.binaryBlockAllocated, gMetrics.binaryAligned,
		gMetrics.deserializeSuccess, gMetrics.loadedObjects,
		gMetrics.loadedActors, gMetrics.loadedConstraints,
		gMetrics.loadedRevolute, gMetrics.loadedPrismatic,
		gMetrics.loadedRack, gMetrics.dependencyIdentity,
		gMetrics.actorIdentity, gMetrics.sceneActors,
		gMetrics.sceneConstraints, gMetrics.authoringReleased,
		gMetrics.loadedCollectionReleased, gMetrics.binaryBlockFreed,
		gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS
				  : Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		if(error.empty())
			error = "configuration_error";
		return reportConfigurationError(error.c_str());
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		return reportConfigurationError("execution_environment_failed");
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
