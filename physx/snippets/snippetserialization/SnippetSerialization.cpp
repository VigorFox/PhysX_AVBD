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
// This snippet illustrates the use of binary and xml serialization
//
// Note: RepX/Xml serialization has been DEPRECATED.
//
// It creates a chain of boxes and serializes them as two collections: 
// a collection with shared objects and a collection with actors and joints
// which can be instantiated multiple times.
//
// Then physics is setup based on the serialized data. The collection with the 
// actors and the joints is instantiated multiple times with different 
// transforms.
//
// Finally phyics is teared down again, including deallocation of memory
// occupied by deserialized objects (in the case of binary serialization).
//
// ****************************************************************************

#include "PxPhysicsAPI.h"
#include "foundation/PxMemory.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "extensions/PxCollectionExt.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace physx;

static bool						gUseBinarySerialization = false;

static PxDefaultAllocator		gAllocator;
static PxDefaultErrorCallback	gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxPvd*					gPvd        = NULL;

#define MAX_MEMBLOCKS 10
static PxU8*					gMemBlocks[MAX_MEMBLOCKS];
static PxU32					gMemBlockCount = 0;

/**
Creates two example collections: 
- collection with actors and joints that can be instantiated multiple times in the scene
- collection with shared objects
*/
void createCollections(PxCollection*& sharedCollection, PxCollection*& actorCollection, PxSerializationRegistry& sr)
{
	PxMaterial* material = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	PxReal halfLength = 2.0f, height = 25.0f;
	PxVec3 offset(halfLength, 0, 0);
	PxRigidActor* prevActor = PxCreateStatic(*gPhysics, PxTransform(PxVec3(0,height,0)), PxSphereGeometry(halfLength), *material, PxTransform(offset));

	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfLength, 1.0f, 1.0f), *material);
	for(PxU32 i=1; i<8;i++)
	{
		PxTransform tm(PxVec3(PxReal(i*2)* halfLength, height, 0));	
		PxRigidDynamic* dynamic = gPhysics->createRigidDynamic(tm);
		dynamic->attachShape(*shape);
		PxRigidBodyExt::updateMassAndInertia(*dynamic, 10.0f);

		PxSphericalJointCreate(*gPhysics, prevActor, PxTransform(offset), dynamic, PxTransform(-offset));
		prevActor = dynamic;
	}
		
	sharedCollection = PxCreateCollection();		// collection for all the shared objects
	actorCollection = PxCreateCollection();			// collection for all the nonshared objects

	sharedCollection->add(*shape);
	PxSerialization::complete(*sharedCollection, sr);									// chases the pointer from shape to material, and adds it
	PxSerialization::createSerialObjectIds(*sharedCollection, PxSerialObjectId(77));	// arbitrary choice of base for references to shared objects

	actorCollection->add(*prevActor);
	PxSerialization::complete(*actorCollection, sr, sharedCollection, true);			// chases all pointers and recursively adds actors and joints
}

/**
Allocates 128 byte aligned memory block for binary serialized data 
Stores pointer to memory in gMemBlocks for later deallocation
*/
void* createAlignedBlock(PxU64 size)
{
	PX_ASSERT(gMemBlockCount < MAX_MEMBLOCKS);
	PxU8* baseAddr = static_cast<PxU8*>(malloc(size+PX_SERIAL_FILE_ALIGN-1));
	gMemBlocks[gMemBlockCount++] = baseAddr;
	void* alignedBlock = reinterpret_cast<void*>((size_t(baseAddr)+PX_SERIAL_FILE_ALIGN-1)&~(PX_SERIAL_FILE_ALIGN-1));
	return alignedBlock;
}

/**
Create objects, add them to collections and serialize the collections to the steams gSharedStream and gActorStream
This function doesn't setup the gPhysics global as the corresponding physics object is only used locally 
*/
void serializeObjects(PxOutputStream& sharedStream, PxOutputStream& actorStream)
{
	PxSerializationRegistry* sr = PxSerialization::createSerializationRegistry(*gPhysics);

	PxCollection* sharedCollection = NULL;
	PxCollection* actorCollection = NULL;
	createCollections(sharedCollection, actorCollection, *sr);

	// Alternatively to using PxDefaultMemoryOutputStream it would be possible to serialize to files using 
	// PxDefaultFileOutputStream or a similar implementation of PxOutputStream.
	if (gUseBinarySerialization)
	{
		PxSerialization::serializeCollectionToBinary(sharedStream, *sharedCollection, *sr);
		PxSerialization::serializeCollectionToBinary(actorStream, *actorCollection, *sr, sharedCollection);	
	}
	else
	{
		PxSerialization::serializeCollectionToXml(sharedStream, *sharedCollection, *sr);
		PxSerialization::serializeCollectionToXml(actorStream, *actorCollection, *sr, NULL, sharedCollection);	
	}

	actorCollection->release();
	sharedCollection->release();

	sr->release();
}

/**
Deserialize shared data and use resulting collection to deserialize and instance actor collections
*/
void deserializeObjects(PxInputData& sharedData, PxInputData& actorData, const PxCookingParams& params)
{
	PxSerializationRegistry* sr = PxSerialization::createSerializationRegistry(*gPhysics);

	PxCollection* sharedCollection = NULL;
	{
		if (gUseBinarySerialization)
		{
			void* alignedBlock = createAlignedBlock(sharedData.getLength());
			sharedData.read(alignedBlock, sharedData.getLength());
			sharedCollection = PxSerialization::createCollectionFromBinary(alignedBlock, *sr);
		}
		else
		{
			sharedCollection = PxSerialization::createCollectionFromXml(sharedData, params, *sr);
		}
	}

	// Deserialize collection and instantiate objects twice, each time with a different transform
	PxTransform transforms[2] = { PxTransform(PxVec3(-5.0f, 0.0f, 0.0f)), PxTransform(PxVec3(5.0f, 0.0f, 0.0f)) };
	
	for (PxU32 i = 0; i < 2; i++)
	{
		PxCollection* collection = NULL;

		// If the PxInputData actorData would refer to a file, it would be better to avoid reading from it twice.
		// This could be achieved by reading the file once to memory, and then working with copies.
		// This is particulary practical when using binary serialization, where the data can be directly 
		// converted to physics objects.
		actorData.seek(0);		

		if (gUseBinarySerialization)
		{
			void* alignedBlock = createAlignedBlock(actorData.getLength());
			actorData.read(alignedBlock, actorData.getLength());
			collection = PxSerialization::createCollectionFromBinary(alignedBlock, *sr, sharedCollection);
		}
		else
		{
			collection = PxSerialization::createCollectionFromXml(actorData, params, *sr, sharedCollection);
		}

		for (PxU32 o = 0; o < collection->getNbObjects(); o++)
		{
			PxRigidActor* rigidActor = collection->getObject(o).is<PxRigidActor>();
			if (rigidActor)
			{
				PxTransform globalPose = rigidActor->getGlobalPose();
				globalPose = globalPose.transform(transforms[i]);
				rigidActor->setGlobalPose(globalPose);
			}
		}

		gScene->addCollection(*collection);
		collection->release();
	}
	sharedCollection->release();

	PxMaterial* material;
	gPhysics->getMaterials(&material,1);
	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *material);
	gScene->addActor(*groundPlane);
	sr->release();
}

/**
Initializes physics and creates a scene
*/
void initPhysics()
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	PxInitExtensions(*gPhysics, gPvd);
	
	PxU32 numCores = SnippetUtils::getNbPhysicalCores();
	gDispatcher = PxDefaultCpuDispatcherCreate(numCores == 0 ? 0 : numCores - 1);
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0, -9.81f, 0);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	sceneDesc.solverType = PxSolverType::eAVBD;
	gScene = gPhysics->createScene(sceneDesc);

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
}

void stepPhysics()
{
	gScene->simulate(1.0f/60.0f);
	gScene->fetchResults(true);
}

/**
Releases all physics objects, including memory blocks containing deserialized data
*/
void cleanupPhysics()
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PxCloseExtensions();

	PX_RELEASE(gPhysics);	// releases all objects	
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}

	// Now that the objects have been released, it's safe to release the space they occupy
	for (PxU32 i = 0; i < gMemBlockCount; i++)
		free(gMemBlocks[i]);

	gMemBlockCount = 0;

	PX_RELEASE(gFoundation);
}

namespace
{

enum SerializationFormat
{
	eSERIALIZATION_FORMAT_UNSET,
	eSERIALIZATION_FORMAT_BINARY,
	eSERIALIZATION_FORMAT_XML
};

static const PxU32 gExpectedActorCount = 8;
static const PxU32 gExpectedDynamicCount = 7;
static const PxU32 gExpectedJointCount = 7;
static const PxSerialObjectId gSharedMaterialId = PxSerialObjectId(1001);
static const PxSerialObjectId gSharedShapeId = PxSerialObjectId(1002);
static const PxSerialObjectId gActorIdBase = PxSerialObjectId(2000);
static const PxSerialObjectId gJointIdBase = PxSerialObjectId(3000);
static const PxReal gHalfLength = 2.0f;
static const PxReal gChainHeight = 25.0f;
static const PxReal gLaneOffset = 20.0f;
static const PxReal gMotionMinimum = 0.01f;
static const PxReal gAnchorErrorLimit = 0.5f;
static const PxReal gClonePositionErrorLimit = 0.02f;
static const PxReal gCloneVelocityErrorLimit = 0.05f;
static const PxReal gCloneAngleErrorLimit = 0.01f;
static const PxReal gRunawayPositionLimit = 1000.0f;
static const PxReal gRunawayVelocityLimit = 1000.0f;

struct SerializationHeadlessConfig
{
	SerializationFormat format;
	PxU32 cycles;
	bool cyclesExplicit;
	PxU64 requestedFrames;

	SerializationHeadlessConfig()
	: format(eSERIALIZATION_FORMAT_UNSET), cycles(1), cyclesExplicit(false), requestedFrames(240)
	{
	}
};

struct SerializationMetrics
{
	bool infrastructureError;
	std::string infrastructureReason;
	bool serializeShared;
	bool serializeActors;
	bool immutableBytes;
	bool authoringCleanup;
	PxU64 sharedBytes;
	PxU64 actorBytes;
	PxU64 sharedByteHash;
	PxU64 actorByteHash;
	PxU64 deserializeAttempts;
	PxU64 deserializeSuccesses;
	PxU64 expectedActors;
	PxU64 loadedActors;
	PxU64 expectedStatics;
	PxU64 loadedStatics;
	PxU64 expectedDynamics;
	PxU64 loadedDynamics;
	PxU64 expectedJoints;
	PxU64 loadedJoints;
	PxU64 expectedConstraints;
	PxU64 loadedConstraints;
	PxU64 sharedIdMismatches;
	PxU64 actorIdMismatches;
	PxU64 jointIdMismatches;
	PxU64 constraintNulls;
	PxU64 constraintDuplicates;
	PxU64 endpointMismatches;
	PxU64 crossReferenceMismatches;
	PxU64 graphMismatches;
	PxU64 externalReferenceMismatches;
	PxU64 exclusiveShapeMismatches;
	PxU64 parameterMismatches;
	PxU64 scenePopulationMismatches;
	PxU64 nonFinite;
	PxU64 runawaySamples;
	PxU64 quaternionMismatches;
	PxU64 brokenConstraints;
	PxU64 anchorSamples;
	PxU64 cloneSamples;
	PxU64 expectedMotionWitnesses;
	PxU64 motionWitnesses;
	PxReal maxTerminalMotion;
	PxReal maxAnchorError;
	PxReal maxClonePositionError;
	PxReal maxCloneVelocityError;
	PxReal maxCloneAngleError;
	PxU64 simulateCalls;
	PxU64 fetchCalls;
	PxU64 fetchFailures;
	PxU64 requestedFrames;
	PxU64 completedFrames;
	PxU32 requestedCycles;
	PxU32 completedCycles;
	PxU64 collectionsCreated;
	PxU64 collectionsReleased;
	PxU64 binaryBlocksAllocated;
	PxU64 binaryBlocksFreed;
	PxU64 binaryBytesAllocated;
	PxU64 binaryBytesFreed;
	PxU64 alignmentFailures;
	PxU64 shortReads;
	PxU64 sceneCleanupMismatches;
	PxU64 finalObjectMismatches;
	PxU32 remainingConstraints;
	PxU32 remainingShapes;
	PxU32 remainingMaterials;
	PxU32 pvdCreated;

	SerializationMetrics()
	: infrastructureError(false), serializeShared(false), serializeActors(false),
	  immutableBytes(false), authoringCleanup(false), sharedBytes(0), actorBytes(0),
	  sharedByteHash(0), actorByteHash(0),
	  deserializeAttempts(0), deserializeSuccesses(0), expectedActors(0), loadedActors(0),
	  expectedStatics(0), loadedStatics(0), expectedDynamics(0), loadedDynamics(0),
	  expectedJoints(0), loadedJoints(0), expectedConstraints(0), loadedConstraints(0),
	  sharedIdMismatches(0), actorIdMismatches(0), jointIdMismatches(0),
	  constraintNulls(0), constraintDuplicates(0), endpointMismatches(0),
	  crossReferenceMismatches(0), graphMismatches(0), externalReferenceMismatches(0),
	  exclusiveShapeMismatches(0), parameterMismatches(0), scenePopulationMismatches(0),
	  nonFinite(0), runawaySamples(0), quaternionMismatches(0), brokenConstraints(0), anchorSamples(0),
	  cloneSamples(0), expectedMotionWitnesses(0), motionWitnesses(0),
	  maxTerminalMotion(0.0f), maxAnchorError(0.0f), maxClonePositionError(0.0f),
	  maxCloneVelocityError(0.0f), maxCloneAngleError(0.0f), simulateCalls(0),
	  fetchCalls(0), fetchFailures(0), requestedFrames(0), completedFrames(0),
	  requestedCycles(0), completedCycles(0), collectionsCreated(0),
	  collectionsReleased(0), binaryBlocksAllocated(0), binaryBlocksFreed(0),
	  binaryBytesAllocated(0), binaryBytesFreed(0), alignmentFailures(0), shortReads(0),
	  sceneCleanupMismatches(0), finalObjectMismatches(0), remainingConstraints(0),
	  remainingShapes(0), remainingMaterials(0), pvdCreated(0)
	{
	}
};

struct BinaryBlock
{
	PxU8* raw;
	void* aligned;
	PxU64 size;

	BinaryBlock() : raw(NULL), aligned(NULL), size(0) {}
};

struct LoadedShared
{
	PxCollection* collection;
	PxMaterial* material;
	PxShape* shape;
	BinaryBlock block;

	LoadedShared() : collection(NULL), material(NULL), shape(NULL) {}
};

struct LoadedClone
{
	PxCollection* collection;
	PxRigidActor* actors[gExpectedActorCount];
	PxSphericalJoint* joints[gExpectedJointCount];
	PxConstraint* constraints[gExpectedJointCount];
	PxShape* exclusiveShape;
	PxRigidDynamic* terminal;
	PxVec3 initialTerminalPosition;
	PxReal laneOffset;
	PxReal maxMotion;
	BinaryBlock block;
	bool canSimulate;

	LoadedClone()
	: collection(NULL), exclusiveShape(NULL), terminal(NULL), initialTerminalPosition(PxZero),
	  laneOffset(0.0f), maxMotion(0.0f), canSimulate(true)
	{
		for(PxU32 i = 0; i < gExpectedActorCount; ++i)
			actors[i] = NULL;
		for(PxU32 i = 0; i < gExpectedJointCount; ++i)
		{
			joints[i] = NULL;
			constraints[i] = NULL;
		}
	}
};

static Snippets::TrackingErrorCallback gHeadlessErrorCallback;

static const char* getFormatName(SerializationFormat format)
{
	switch(format)
	{
	case eSERIALIZATION_FORMAT_BINARY:
		return "binary";
	case eSERIALIZATION_FORMAT_XML:
		return "xml";
	default:
		return "unset";
	}
}

static PxSerialObjectId getActorId(PxU32 index)
{
	return gActorIdBase + PxSerialObjectId(index);
}

static PxSerialObjectId getJointId(PxU32 index)
{
	return gJointIdBase + PxSerialObjectId(index);
}

static PxReal getExpectedMass(PxU32 actorIndex)
{
	return 5.0f + PxReal(actorIndex);
}

static PxVec3 getExpectedInertia(PxU32 actorIndex)
{
	const PxReal offset = 0.2f * PxReal(actorIndex);
	return PxVec3(2.0f + offset, 2.5f + offset, 3.0f + offset);
}

static PxTransform getExpectedCMassPose(PxU32 actorIndex)
{
	return PxTransform(PxVec3(0.01f * PxReal(actorIndex), -0.005f * PxReal(actorIndex),
	                         0.003f * PxReal(actorIndex)),
	                   PxQuat(0.02f * PxReal(actorIndex), PxVec3(0.0f, 0.0f, 1.0f)));
}

static PxTransform getExpectedJointFrame(PxU32 jointIndex, bool actorOne)
{
	const PxQuat orientation(0.025f * PxReal(jointIndex + 1), PxVec3(1.0f, 0.0f, 0.0f));
	return PxTransform(PxVec3(actorOne ? -gHalfLength : gHalfLength, 0.0f, 0.0f), orientation);
}

static PxJointLimitCone getExpectedCone(PxU32 jointIndex)
{
	PxJointLimitCone cone(0.75f + 0.01f * PxReal(jointIndex),
	                      0.65f + 0.01f * PxReal(jointIndex));
	cone.restitution = 0.1f;
	cone.bounceThreshold = 0.25f;
	return cone;
}

static PxConstraintFlags getExpectedConstraintFlags()
{
	PxConstraintFlags flags(PxConstraintFlag::eVISUALIZATION);
	flags |= PxConstraintFlag::eALWAYS_UPDATE;
	return flags;
}

static bool nearReal(PxReal lhs, PxReal rhs, PxReal tolerance = 1.0e-5f)
{
	return PxAbs(lhs - rhs) <= tolerance;
}

static bool nearVec(const PxVec3& lhs, const PxVec3& rhs, PxReal tolerance = 1.0e-5f)
{
	return (lhs - rhs).magnitude() <= tolerance;
}

static bool nearQuat(const PxQuat& lhs, const PxQuat& rhs, PxReal tolerance = 1.0e-5f)
{
	return PxAbs(PxAbs(lhs.dot(rhs)) - 1.0f) <= tolerance;
}

static bool nearTransform(const PxTransform& lhs, const PxTransform& rhs,
	                      PxReal tolerance = 1.0e-5f)
{
	return nearVec(lhs.p, rhs.p, tolerance) && nearQuat(lhs.q, rhs.q, tolerance);
}

static std::string makeToken(const std::string& value)
{
	std::string token;
	token.reserve(value.size());
	for(size_t i = 0; i < value.size(); ++i)
	{
		const unsigned char c = static_cast<unsigned char>(value[i]);
		if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		   (c >= '0' && c <= '9') || c == '_' || c == '-')
			token.push_back(static_cast<char>(c));
		else
			token.push_back('_');
	}
	return token.empty() ? std::string("configuration_error") : token;
}

static void setInfrastructureError(SerializationMetrics& metrics, const char* reason)
{
	if(!metrics.infrastructureError)
	{
		metrics.infrastructureError = true;
		metrics.infrastructureReason = reason ? reason : "infrastructure_error";
	}
}

static PxU64 hashBytes(const std::vector<PxU8>& bytes)
{
	PxU64 hash = PxU64(14695981039346656037ULL);
	for(size_t i = 0; i < bytes.size(); ++i)
	{
		hash ^= PxU64(bytes[i]);
		hash *= PxU64(1099511628211ULL);
	}
	return hash;
}

static int reportConfigurationError(const Snippets::HeadlessOptions& options,
	                                const SerializationHeadlessConfig& config,
	                                const std::string& message)
{
	const std::string reason = makeToken(message);
	PxU32 reportedFrames = options.frames;
	PxU32 reportedCycles = config.cycles;
	if(Snippets::equalsIgnoreCase(options.caseName.c_str(), "repeat-create-release"))
	{
		if(!options.framesExplicit)
			reportedFrames = 30;
		if(!config.cyclesExplicit)
			reportedCycles = 20;
	}
	else if(Snippets::equalsIgnoreCase(options.caseName.c_str(), "spherical-chain"))
	{
		if(!options.framesExplicit)
			reportedFrames = 240;
		if(!config.cyclesExplicit)
			reportedCycles = 1;
	}
	const PxU64 requestedFrames = PxU64(reportedCycles) * PxU64(reportedFrames);
	printf("[AVBD_GATE_ERROR] snippet=SnippetSerialization message=%s\n", message.c_str());
	printf("[AVBD_GATE] schema=1 snippet=SnippetSerialization case=config-error "
	       "solver=%s execution=%s requestedFrames=%llu completedFrames=0 dt=%.9g "
	       "seed=%u dispatcherThreads=%u capability=SUPPORTED validation=GATED "
	       "status=ERROR reason=%s format=%s cycles=%u completedCycles=0 "
	       "framesPerCycle=%u nonFinite=0 physicsErrors=0 physicsWarnings=0\n",
	       Snippets::getSolverTypeName(options.solverType),
	       Snippets::getExecutionName(options.execution),
	       static_cast<unsigned long long>(requestedFrames), double(options.dt), options.seed,
	       options.dispatcherThreads, reason.c_str(), getFormatName(config.format), reportedCycles,
	       reportedFrames);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

static bool isHeadlessInvocation(int argc, const char* const* argv)
{
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))
		return true;
	return argc > 1 && argv != NULL;
}

static bool parseHeadlessConfiguration(int argc, const char* const* argv,
	                                  Snippets::HeadlessOptions& options,
	                                  SerializationHeadlessConfig& config,
	                                  std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.caseName = "spherical-chain";
	defaults.frames = 240;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.dispatcherThreads = 2;
	defaults.dt = 1.0f / 60.0f;

	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, error))
		return false;

	bool formatSeen = false;
	bool cyclesSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(!arg)
			continue;
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		if(Snippets::hasOptionPrefix(arg, "--format="))
		{
			if(formatSeen)
			{
				error = "duplicate --format";
				return false;
			}
			formatSeen = true;
			const char* value = arg + std::strlen("--format=");
			if(Snippets::equalsIgnoreCase(value, "binary"))
				config.format = eSERIALIZATION_FORMAT_BINARY;
			else if(Snippets::equalsIgnoreCase(value, "xml"))
				config.format = eSERIALIZATION_FORMAT_XML;
			else
			{
				error = "invalid --format value";
				return false;
			}
		}
		else if(Snippets::hasOptionPrefix(arg, "--cycles="))
		{
			if(cyclesSeen)
			{
				error = "duplicate --cycles";
				return false;
			}
			cyclesSeen = true;
			config.cyclesExplicit = true;
			if(!Snippets::parseU32(arg + std::strlen("--cycles="), 1, 100000u, config.cycles))
			{
				error = "invalid --cycles value";
				return false;
			}
		}
		else
		{
			error = std::string("unknown argument ") + arg;
			return false;
		}
	}

	if(!options.headless)
	{
		error = "gate option requires --headless";
		return false;
	}
	if(config.format == eSERIALIZATION_FORMAT_UNSET)
	{
		error = "--format is required";
		return false;
	}

	if(Snippets::equalsIgnoreCase(options.caseName.c_str(), "spherical-chain"))
	{
		options.caseName = "spherical-chain";
		if(!options.framesExplicit)
			options.frames = 240;
		if(!config.cyclesExplicit)
			config.cycles = 1;
		if(config.cycles != 1)
		{
			error = "spherical-chain requires cycles=1";
			return false;
		}
	}
	else if(Snippets::equalsIgnoreCase(options.caseName.c_str(), "repeat-create-release"))
	{
		options.caseName = "repeat-create-release";
		if(!options.framesExplicit)
			options.frames = 30;
		if(!config.cyclesExplicit)
			config.cycles = 20;
		if(config.cycles < 2)
		{
			error = "repeat-create-release requires at least 2 cycles";
			return false;
		}
	}
	else
	{
		error = "invalid --case value";
		return false;
	}

	if(options.solverType == PxSolverType::ePGS)
	{
		error = "headless solver requires avbd or tgs";
		return false;
	}
	if(options.solverType == PxSolverType::eTGS &&
	   options.execution == Snippets::eHEADLESS_SEQUENTIAL)
	{
		error = "sequential execution requires avbd";
		return false;
	}
	if(PxAbs(options.dt - 1.0f / 60.0f) > 1.0e-8f)
	{
		error = "dt requires 60hz calibration";
		return false;
	}

	config.requestedFrames = PxU64(config.cycles) * PxU64(options.frames);
	if(config.requestedFrames == 0 || config.requestedFrames > PxU64(1000000000000ULL))
	{
		error = "total requested frames exceeds limit";
		return false;
	}
	return true;
}

static bool allocateBinaryBlock(const std::vector<PxU8>& bytes, BinaryBlock& block,
	                            SerializationMetrics& metrics)
{
	if(bytes.empty() || bytes.size() > std::numeric_limits<PxU32>::max())
	{
		setInfrastructureError(metrics, "binary_source_size");
		return false;
	}

	const PxU64 size = PxU64(bytes.size());
	if(size > std::numeric_limits<size_t>::max() - (PX_SERIAL_FILE_ALIGN - 1))
	{
		setInfrastructureError(metrics, "binary_allocation_overflow");
		return false;
	}
	block.raw = static_cast<PxU8*>(std::malloc(static_cast<size_t>(size) + PX_SERIAL_FILE_ALIGN - 1));
	if(!block.raw)
	{
		setInfrastructureError(metrics, "binary_allocation_failed");
		return false;
	}

	const size_t rawAddress = reinterpret_cast<size_t>(block.raw);
	const size_t alignedAddress =
		(rawAddress + PX_SERIAL_FILE_ALIGN - 1) & ~(size_t(PX_SERIAL_FILE_ALIGN) - 1);
	block.aligned = reinterpret_cast<void*>(alignedAddress);
	block.size = size;
	metrics.binaryBlocksAllocated++;
	metrics.binaryBytesAllocated += size;
	if((reinterpret_cast<size_t>(block.aligned) & (PX_SERIAL_FILE_ALIGN - 1)) != 0)
	{
		metrics.alignmentFailures++;
		setInfrastructureError(metrics, "binary_alignment");
		return false;
	}

	PxDefaultMemoryInputData input(&bytes[0], static_cast<PxU32>(bytes.size()));
	const PxU64 copied = input.read(block.aligned, PxU64(bytes.size()));
	if(copied != PxU64(bytes.size()))
	{
		metrics.shortReads++;
		setInfrastructureError(metrics, "binary_short_read");
		return false;
	}
	return true;
}

static void freeBinaryBlock(BinaryBlock& block, SerializationMetrics& metrics)
{
	if(block.raw)
	{
		std::free(block.raw);
		metrics.binaryBlocksFreed++;
		metrics.binaryBytesFreed += block.size;
	}
	block.raw = NULL;
	block.aligned = NULL;
	block.size = 0;
}

static PxCollection* deserializeCollection(const std::vector<PxU8>& bytes,
	                                       SerializationFormat format,
	                                       PxSerializationRegistry& registry,
	                                       const PxCollection* externalReferences,
	                                       BinaryBlock& block,
	                                       const char* failureReason,
	                                       SerializationMetrics& metrics)
{
	metrics.deserializeAttempts++;
	PxCollection* collection = NULL;
	if(format == eSERIALIZATION_FORMAT_BINARY)
	{
		if(!allocateBinaryBlock(bytes, block, metrics))
			return NULL;
		collection = PxSerialization::createCollectionFromBinary(block.aligned, registry,
		                                                       externalReferences);
	}
	else
	{
		if(bytes.empty() || bytes.size() > std::numeric_limits<PxU32>::max())
		{
			setInfrastructureError(metrics, "xml_source_size");
			return NULL;
		}
		PxDefaultMemoryInputData input(&bytes[0], static_cast<PxU32>(bytes.size()));
		const PxCookingParams cookingParams(gPhysics->getTolerancesScale());
		collection = PxSerialization::createCollectionFromXml(input, cookingParams, registry,
		                                                    externalReferences);
	}
	if(!collection)
	{
		setInfrastructureError(metrics, failureReason);
		return NULL;
	}
	metrics.deserializeSuccesses++;
	metrics.collectionsCreated++;
	return collection;
}

static bool initHeadlessPhysics(const Snippets::HeadlessOptions& options,
	                           SerializationMetrics& metrics)
{
	gHeadlessErrorCallback.reset();
	gPvd = NULL;
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gHeadlessErrorCallback);
	if(!gFoundation)
	{
		setInfrastructureError(metrics, "foundation_create");
		return false;
	}
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, NULL);
	if(!gPhysics)
	{
		setInfrastructureError(metrics, "physics_create");
		return false;
	}
	if(!PxInitExtensions(*gPhysics, NULL))
	{
		setInfrastructureError(metrics, "extensions_init");
		return false;
	}

	gDispatcher = PxDefaultCpuDispatcherCreate(options.dispatcherThreads);
	if(!gDispatcher)
	{
		setInfrastructureError(metrics, "dispatcher_create");
		return false;
	}
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(PxZero);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.solverType = options.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
	{
		setInfrastructureError(metrics, "scene_create");
		return false;
	}
	return true;
}

static void cleanupHeadlessPhysics(SerializationMetrics& metrics)
{
	if(gScene)
	{
		if(gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC |
		                       PxActorTypeFlag::eRIGID_DYNAMIC) != 0 ||
		   gScene->getNbConstraints() != 0)
			metrics.finalObjectMismatches++;
		PX_RELEASE(gScene);
	}
	PX_RELEASE(gDispatcher);
	if(gPhysics)
	{
		metrics.remainingConstraints = PxMax(metrics.remainingConstraints,
		                                    gPhysics->getNbConstraints());
		metrics.remainingShapes = PxMax(metrics.remainingShapes, gPhysics->getNbShapes());
		metrics.remainingMaterials = PxMax(metrics.remainingMaterials,
		                                  gPhysics->getNbMaterials());
		if(gPhysics->getNbConstraints() != 0 || gPhysics->getNbShapes() != 0 ||
		   gPhysics->getNbMaterials() != 0)
			metrics.finalObjectMismatches++;
		PxCloseExtensions();
		PX_RELEASE(gPhysics);
	}
	PX_RELEASE(gFoundation);
}

static void releaseAuthoringObjects(PxCollection*& sharedCollection,
	                                PxCollection*& actorCollection,
	                                PxMaterial*& material, PxShape*& shape,
	                                PxRigidActor* actors[gExpectedActorCount],
	                                PxSphericalJoint* joints[gExpectedJointCount],
	                                SerializationMetrics& metrics)
{
	if(actorCollection)
	{
		actorCollection->release();
		actorCollection = NULL;
		metrics.collectionsReleased++;
	}
	if(sharedCollection)
	{
		sharedCollection->release();
		sharedCollection = NULL;
		metrics.collectionsReleased++;
	}
	for(PxU32 i = 0; i < gExpectedJointCount; ++i)
		PX_RELEASE(joints[i]);
	for(PxU32 i = 0; i < gExpectedActorCount; ++i)
		PX_RELEASE(actors[i]);
	PX_RELEASE(shape);
	PX_RELEASE(material);
}

static bool serializeAuthoringCollections(SerializationFormat format,
	                                      PxSerializationRegistry& registry,
	                                      std::vector<PxU8>& sharedBytes,
	                                      std::vector<PxU8>& actorBytes,
	                                      SerializationMetrics& metrics)
{
	PxMaterial* material = NULL;
	PxShape* shape = NULL;
	PxRigidActor* actors[gExpectedActorCount] = {};
	PxSphericalJoint* joints[gExpectedJointCount] = {};
	PxCollection* sharedCollection = NULL;
	PxCollection* actorCollection = NULL;
	bool created = true;

	material = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	shape = material ? gPhysics->createShape(PxBoxGeometry(gHalfLength, 1.0f, 1.0f),
	                                         *material, false) : NULL;
	if(!material || !shape)
		created = false;

	if(created)
	{
		actors[0] = PxCreateStatic(*gPhysics, PxTransform(PxVec3(0.0f, gChainHeight, 0.0f)),
		                           PxSphereGeometry(gHalfLength), *material,
		                           PxTransform(PxVec3(gHalfLength, 0.0f, 0.0f)));
		created = actors[0] != NULL;
	}

	PxRigidActor* previous = actors[0];
	for(PxU32 i = 1; created && i < gExpectedActorCount; ++i)
	{
		PxRigidDynamic* dynamic = gPhysics->createRigidDynamic(
			PxTransform(PxVec3(PxReal(i * 2) * gHalfLength, gChainHeight, 0.0f)));
		actors[i] = dynamic;
		if(!dynamic || !dynamic->attachShape(*shape))
		{
			created = false;
			break;
		}
		dynamic->setMass(getExpectedMass(i));
		dynamic->setMassSpaceInertiaTensor(getExpectedInertia(i));
		dynamic->setCMassLocalPose(getExpectedCMassPose(i));
		dynamic->setLinearDamping(0.04f);
		dynamic->setAngularDamping(0.07f);
		dynamic->setSolverIterationCounts(8, 2);

		PxSphericalJoint* joint = PxSphericalJointCreate(*gPhysics, previous,
			getExpectedJointFrame(i - 1, false), dynamic,
			getExpectedJointFrame(i - 1, true));
		joints[i - 1] = joint;
		if(!joint)
		{
			created = false;
			break;
		}
		joint->setConstraintFlags(getExpectedConstraintFlags());
		joint->setBreakForce(100000.0f + 1000.0f * PxReal(i - 1),
		                     80000.0f + 1000.0f * PxReal(i - 1));
		joint->setInvMassScale0(0.9f);
		joint->setInvInertiaScale0(0.85f);
		joint->setInvMassScale1(1.1f);
		joint->setInvInertiaScale1(1.15f);
		joint->setLimitCone(getExpectedCone(i - 1));
		joint->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
		previous = dynamic;
	}

	if(created)
	{
		sharedCollection = PxCreateCollection();
		actorCollection = PxCreateCollection();
		if(sharedCollection)
			metrics.collectionsCreated++;
		if(actorCollection)
			metrics.collectionsCreated++;
		created = sharedCollection != NULL && actorCollection != NULL;
	}

	if(created)
	{
		sharedCollection->add(*material, gSharedMaterialId);
		sharedCollection->add(*shape, gSharedShapeId);
		PxSerialization::complete(*sharedCollection, registry);
		for(PxU32 i = 0; i < gExpectedActorCount; ++i)
			actorCollection->add(*actors[i], getActorId(i));
		for(PxU32 i = 0; i < gExpectedJointCount; ++i)
			actorCollection->add(*joints[i], getJointId(i));
		PxSerialization::complete(*actorCollection, registry, sharedCollection, true);
		created = PxSerialization::isSerializable(*sharedCollection, registry) &&
		          PxSerialization::isSerializable(*actorCollection, registry, sharedCollection);
	}

	PxDefaultMemoryOutputStream sharedOutput;
	PxDefaultMemoryOutputStream actorOutput;
	if(created)
	{
		if(format == eSERIALIZATION_FORMAT_BINARY)
		{
			metrics.serializeShared = PxSerialization::serializeCollectionToBinary(
				sharedOutput, *sharedCollection, registry);
			metrics.serializeActors = PxSerialization::serializeCollectionToBinary(
				actorOutput, *actorCollection, registry, sharedCollection);
		}
		else
		{
			metrics.serializeShared = PxSerialization::serializeCollectionToXml(
				sharedOutput, *sharedCollection, registry);
			metrics.serializeActors = PxSerialization::serializeCollectionToXml(
				actorOutput, *actorCollection, registry, NULL, sharedCollection);
		}
	}

	if(metrics.serializeShared && metrics.serializeActors &&
	   sharedOutput.getSize() != 0 && actorOutput.getSize() != 0)
	{
		sharedBytes.assign(sharedOutput.getData(), sharedOutput.getData() + sharedOutput.getSize());
		actorBytes.assign(actorOutput.getData(), actorOutput.getData() + actorOutput.getSize());
		metrics.sharedBytes = sharedBytes.size();
		metrics.actorBytes = actorBytes.size();
		metrics.sharedByteHash = hashBytes(sharedBytes);
		metrics.actorByteHash = hashBytes(actorBytes);
		metrics.immutableBytes = !sharedBytes.empty() && !actorBytes.empty();
	}
	else
	{
		setInfrastructureError(metrics, created ? "serialization_failed" : "authoring_failed");
	}

	releaseAuthoringObjects(sharedCollection, actorCollection, material, shape, actors, joints,
	                       metrics);
	metrics.authoringCleanup = gPhysics->getNbConstraints() == 0 &&
	                           gPhysics->getNbShapes() == 0 &&
	                           gPhysics->getNbMaterials() == 0;
	if(!metrics.authoringCleanup)
		setInfrastructureError(metrics, "authoring_cleanup");
	return !metrics.infrastructureError;
}

static bool shapeUsesMaterial(PxShape& shape, PxMaterial* expectedMaterial)
{
	if(shape.getNbMaterials() != 1)
		return false;
	PxMaterial* material = NULL;
	return shape.getMaterials(&material, 1) == 1 && material == expectedMaterial;
}

static bool validateSharedCollection(LoadedShared& shared, SerializationMetrics& metrics)
{
	if(!shared.collection)
		return false;
	PxBase* materialBase = shared.collection->find(gSharedMaterialId);
	PxBase* shapeBase = shared.collection->find(gSharedShapeId);
	shared.material = materialBase ? materialBase->is<PxMaterial>() : NULL;
	shared.shape = shapeBase ? shapeBase->is<PxShape>() : NULL;
	if(!shared.material || !shared.shape ||
	   (shared.material && shared.collection->getId(*shared.material) != gSharedMaterialId) ||
	   (shared.shape && shared.collection->getId(*shared.shape) != gSharedShapeId))
		metrics.sharedIdMismatches++;

	if(!shared.material || !nearReal(shared.material->getStaticFriction(), 0.5f) ||
	   !nearReal(shared.material->getDynamicFriction(), 0.5f) ||
	   !nearReal(shared.material->getRestitution(), 0.6f))
		metrics.parameterMismatches++;

	if(!shared.shape)
	{
		metrics.externalReferenceMismatches++;
		return false;
	}
	if(shared.shape->isExclusive() ||
	   shared.shape->getGeometry().getType() != PxGeometryType::eBOX)
	{
		metrics.externalReferenceMismatches++;
	}
	else
	{
		const PxBoxGeometry& box = static_cast<const PxBoxGeometry&>(shared.shape->getGeometry());
		if(!nearVec(box.halfExtents, PxVec3(gHalfLength, 1.0f, 1.0f)))
			metrics.parameterMismatches++;
	}
	if(!shapeUsesMaterial(*shared.shape, shared.material))
		metrics.externalReferenceMismatches++;
	return shared.material != NULL && shared.shape != NULL;
}

static PxI32 findActorIndex(const LoadedClone& clone, const PxRigidActor* actor)
{
	for(PxU32 i = 0; i < gExpectedActorCount; ++i)
	{
		if(clone.actors[i] == actor)
			return static_cast<PxI32>(i);
	}
	return -1;
}

static PxI32 findSetRoot(PxI32 parents[gExpectedActorCount], PxI32 node)
{
	while(parents[node] != node)
	{
		parents[node] = parents[parents[node]];
		node = parents[node];
	}
	return node;
}

static bool validateActorParameters(PxRigidDynamic& dynamic, PxU32 actorIndex)
{
	PxU32 positionIterations = 0;
	PxU32 velocityIterations = 0;
	dynamic.getSolverIterationCounts(positionIterations, velocityIterations);
	return nearReal(dynamic.getMass(), getExpectedMass(actorIndex)) &&
	       nearVec(dynamic.getMassSpaceInertiaTensor(), getExpectedInertia(actorIndex)) &&
	       nearTransform(dynamic.getCMassLocalPose(), getExpectedCMassPose(actorIndex)) &&
	       nearReal(dynamic.getLinearDamping(), 0.04f) &&
	       nearReal(dynamic.getAngularDamping(), 0.07f) &&
	       positionIterations == 8 && velocityIterations == 2;
}

static bool validateJointParameters(PxSphericalJoint& joint, PxU32 jointIndex)
{
	PxReal breakForce = 0.0f;
	PxReal breakTorque = 0.0f;
	joint.getBreakForce(breakForce, breakTorque);
	const PxJointLimitCone actualCone = joint.getLimitCone();
	const PxJointLimitCone expectedCone = getExpectedCone(jointIndex);
	return nearTransform(joint.getLocalPose(PxJointActorIndex::eACTOR0),
	                     getExpectedJointFrame(jointIndex, false)) &&
	       nearTransform(joint.getLocalPose(PxJointActorIndex::eACTOR1),
	                     getExpectedJointFrame(jointIndex, true)) &&
	       joint.getConstraintFlags() == getExpectedConstraintFlags() &&
	       nearReal(breakForce, 100000.0f + 1000.0f * PxReal(jointIndex)) &&
	       nearReal(breakTorque, 80000.0f + 1000.0f * PxReal(jointIndex)) &&
	       nearReal(joint.getInvMassScale0(), 0.9f) &&
	       nearReal(joint.getInvInertiaScale0(), 0.85f) &&
	       nearReal(joint.getInvMassScale1(), 1.1f) &&
	       nearReal(joint.getInvInertiaScale1(), 1.15f) &&
	       joint.getSphericalJointFlags() ==
	           PxSphericalJointFlags(PxSphericalJointFlag::eLIMIT_ENABLED) &&
	       nearReal(actualCone.yAngle, expectedCone.yAngle) &&
	       nearReal(actualCone.zAngle, expectedCone.zAngle) &&
	       nearReal(actualCone.restitution, expectedCone.restitution) &&
	       nearReal(actualCone.bounceThreshold, expectedCone.bounceThreshold) &&
	       nearReal(actualCone.stiffness, expectedCone.stiffness) &&
	       nearReal(actualCone.damping, expectedCone.damping);
}

static bool resolveAndValidateClone(LoadedClone& clone, LoadedShared& shared,
	                                PxReal laneOffset,
	                                SerializationMetrics& metrics)
{
	clone.laneOffset = laneOffset;
	if(!clone.collection)
	{
		clone.canSimulate = false;
		return false;
	}
	PxU32 enumeratedActors = 0;
	PxU32 enumeratedSphericalJoints = 0;
	PxU32 enumeratedOtherJoints = 0;
	for(PxU32 i = 0; i < clone.collection->getNbObjects(); ++i)
	{
		PxBase& object = clone.collection->getObject(i);
		if(object.is<PxRigidActor>())
			enumeratedActors++;
		if(object.is<PxSphericalJoint>())
			enumeratedSphericalJoints++;
		else if(object.is<PxJoint>())
			enumeratedOtherJoints++;
	}
	if(enumeratedActors != gExpectedActorCount)
		metrics.actorIdMismatches++;
	if(enumeratedSphericalJoints != gExpectedJointCount || enumeratedOtherJoints != 0)
		metrics.jointIdMismatches++;

	for(PxU32 i = 0; i < gExpectedActorCount; ++i)
	{
		PxBase* base = clone.collection->find(getActorId(i));
		clone.actors[i] = base ? base->is<PxRigidActor>() : NULL;
		if(!clone.actors[i] || clone.collection->getId(*clone.actors[i]) != getActorId(i))
		{
			metrics.actorIdMismatches++;
			clone.canSimulate = false;
			continue;
		}
		metrics.loadedActors++;
		for(PxU32 prior = 0; prior < i; ++prior)
		{
			if(clone.actors[prior] == clone.actors[i])
				metrics.actorIdMismatches++;
		}
		PxTransform pose = clone.actors[i]->getGlobalPose();
		pose.p.z += laneOffset;
		clone.actors[i]->setGlobalPose(pose);
		const PxVec3 expectedPosition(PxReal(i * 2) * gHalfLength, gChainHeight, laneOffset);
		if(!nearVec(clone.actors[i]->getGlobalPose().p, expectedPosition))
			metrics.parameterMismatches++;

		if(i == 0)
		{
			PxRigidStatic* rigidStatic = clone.actors[i]->is<PxRigidStatic>();
			if(!rigidStatic)
			{
				metrics.parameterMismatches++;
				clone.canSimulate = false;
				continue;
			}
			metrics.loadedStatics++;
			if(rigidStatic->getNbShapes() != 1)
			{
				metrics.exclusiveShapeMismatches++;
				continue;
			}
			PxShape* staticShape = NULL;
			if(rigidStatic->getShapes(&staticShape, 1) != 1 || !staticShape)
			{
				metrics.exclusiveShapeMismatches++;
				continue;
			}
			clone.exclusiveShape = staticShape;
			if(!staticShape->isExclusive() ||
			   staticShape->getGeometry().getType() != PxGeometryType::eSPHERE ||
			   !nearTransform(staticShape->getLocalPose(),
			                  PxTransform(PxVec3(gHalfLength, 0.0f, 0.0f))))
				metrics.exclusiveShapeMismatches++;
			else
			{
				const PxSphereGeometry& sphere =
					static_cast<const PxSphereGeometry&>(staticShape->getGeometry());
				if(!nearReal(sphere.radius, gHalfLength))
					metrics.parameterMismatches++;
			}
			if(!shapeUsesMaterial(*staticShape, shared.material))
				metrics.externalReferenceMismatches++;
		}
		else
		{
			PxRigidDynamic* dynamic = clone.actors[i]->is<PxRigidDynamic>();
			if(!dynamic)
			{
				metrics.parameterMismatches++;
				clone.canSimulate = false;
				continue;
			}
			metrics.loadedDynamics++;
			if(!validateActorParameters(*dynamic, i))
				metrics.parameterMismatches++;
			if(dynamic->getNbShapes() != 1)
			{
				metrics.externalReferenceMismatches++;
				continue;
			}
			PxShape* dynamicShape = NULL;
			if(dynamic->getShapes(&dynamicShape, 1) != 1 || dynamicShape != shared.shape ||
			   (dynamicShape && dynamicShape->isExclusive()) ||
			   (dynamicShape && !shapeUsesMaterial(*dynamicShape, shared.material)))
				metrics.externalReferenceMismatches++;
			if(i == gExpectedActorCount - 1)
				clone.terminal = dynamic;
		}
	}

	for(PxU32 i = 0; i < gExpectedJointCount; ++i)
	{
		PxBase* base = clone.collection->find(getJointId(i));
		clone.joints[i] = base ? base->is<PxSphericalJoint>() : NULL;
		if(!clone.joints[i] || clone.collection->getId(*clone.joints[i]) != getJointId(i))
		{
			metrics.jointIdMismatches++;
			clone.canSimulate = false;
			continue;
		}
		metrics.loadedJoints++;
		for(PxU32 prior = 0; prior < i; ++prior)
		{
			if(clone.joints[prior] == clone.joints[i])
				metrics.jointIdMismatches++;
		}
		if(!validateJointParameters(*clone.joints[i], i))
			metrics.parameterMismatches++;
		clone.constraints[i] = clone.joints[i]->getConstraint();
		if(!clone.constraints[i])
		{
			metrics.constraintNulls++;
			clone.canSimulate = false;
			continue;
		}
		metrics.loadedConstraints++;
		for(PxU32 prior = 0; prior < i; ++prior)
		{
			if(clone.constraints[prior] == clone.constraints[i])
				metrics.constraintDuplicates++;
		}
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		clone.joints[i]->getActors(actor0, actor1);
		if(actor0 != clone.actors[i] || actor1 != clone.actors[i + 1])
			metrics.endpointMismatches++;
		PxRigidActor* constraintActor0 = NULL;
		PxRigidActor* constraintActor1 = NULL;
		clone.constraints[i]->getActors(constraintActor0, constraintActor1);
		if(constraintActor0 != actor0 || constraintActor1 != actor1)
			metrics.endpointMismatches++;
	}

	PxI32 parents[gExpectedActorCount];
	for(PxU32 i = 0; i < gExpectedActorCount; ++i)
		parents[i] = static_cast<PxI32>(i);
	bool graphValid = true;
	PxU32 validEdges = 0;
	for(PxU32 i = 0; i < gExpectedJointCount; ++i)
	{
		if(!clone.joints[i])
		{
			graphValid = false;
			continue;
		}
		PxRigidActor* actor0 = NULL;
		PxRigidActor* actor1 = NULL;
		clone.joints[i]->getActors(actor0, actor1);
		const PxI32 index0 = findActorIndex(clone, actor0);
		const PxI32 index1 = findActorIndex(clone, actor1);
		if(index0 < 0 || index1 < 0)
		{
			graphValid = false;
			continue;
		}
		const PxI32 root0 = findSetRoot(parents, index0);
		const PxI32 root1 = findSetRoot(parents, index1);
		if(root0 == root1)
			graphValid = false;
		else
			parents[root1] = root0;
		validEdges++;
	}
	if(validEdges != gExpectedJointCount)
		graphValid = false;
	if(graphValid)
	{
		const PxI32 root = findSetRoot(parents, 0);
		for(PxU32 i = 1; i < gExpectedActorCount; ++i)
			graphValid = graphValid && findSetRoot(parents, static_cast<PxI32>(i)) == root;
	}
	if(!graphValid)
		metrics.graphMismatches++;

	if(!clone.terminal)
		clone.canSimulate = false;
	else
		clone.initialTerminalPosition = clone.terminal->getGlobalPose().p;
	return clone.canSimulate;
}

static void validateCloneIsolation(LoadedClone& clone0, LoadedClone& clone1,
	                               LoadedShared& shared,
	                               SerializationMetrics& metrics)
{
	for(PxU32 i = 0; i < gExpectedActorCount; ++i)
	{
		for(PxU32 j = 0; j < gExpectedActorCount; ++j)
		{
			if(clone0.actors[i] && clone0.actors[i] == clone1.actors[j])
				metrics.crossReferenceMismatches++;
		}
	}
	for(PxU32 i = 0; i < gExpectedJointCount; ++i)
	{
		for(PxU32 j = 0; j < gExpectedJointCount; ++j)
		{
			if(clone0.joints[i] && clone0.joints[i] == clone1.joints[j])
				metrics.crossReferenceMismatches++;
			if(clone0.constraints[i] && clone0.constraints[i] == clone1.constraints[j])
				metrics.crossReferenceMismatches++;
		}
		if(clone0.joints[i])
		{
			PxRigidActor* actor0 = NULL;
			PxRigidActor* actor1 = NULL;
			clone0.joints[i]->getActors(actor0, actor1);
			if(findActorIndex(clone0, actor0) < 0 || findActorIndex(clone0, actor1) < 0)
				metrics.crossReferenceMismatches++;
		}
		if(clone1.joints[i])
		{
			PxRigidActor* actor0 = NULL;
			PxRigidActor* actor1 = NULL;
			clone1.joints[i]->getActors(actor0, actor1);
			if(findActorIndex(clone1, actor0) < 0 || findActorIndex(clone1, actor1) < 0)
				metrics.crossReferenceMismatches++;
		}
	}
	if(clone0.exclusiveShape && clone0.exclusiveShape == clone1.exclusiveShape)
		metrics.exclusiveShapeMismatches++;
	for(PxU32 i = 1; i < gExpectedActorCount; ++i)
	{
		PxShape* shape0 = NULL;
		PxShape* shape1 = NULL;
		if(clone0.actors[i])
			clone0.actors[i]->getShapes(&shape0, 1);
		if(clone1.actors[i])
			clone1.actors[i]->getShapes(&shape1, 1);
		if(shape0 != shared.shape || shape1 != shared.shape || shape0 != shape1)
			metrics.externalReferenceMismatches++;
	}
}

static PxReal quaternionAngle(const PxQuat& lhs, const PxQuat& rhs)
{
	const PxReal dot = PxClamp(PxAbs(lhs.dot(rhs)), 0.0f, 1.0f);
	return 2.0f * static_cast<PxReal>(std::acos(static_cast<double>(dot)));
}

static PxTransform getJointWorldFrame(PxSphericalJoint& joint,
	                                  PxJointActorIndex::Enum actorIndex,
	                                  PxRigidActor* actor)
{
	const PxTransform localPose = joint.getLocalPose(actorIndex);
	return actor ? actor->getGlobalPose().transform(localPose) : localPose;
}

static void samplePhysics(LoadedClone clones[2], SerializationMetrics& metrics)
{
	for(PxU32 lane = 0; lane < 2; ++lane)
	{
		LoadedClone& clone = clones[lane];
		for(PxU32 i = 0; i < gExpectedActorCount; ++i)
		{
			PxRigidActor* actor = clone.actors[i];
			if(!actor)
				continue;
			const PxTransform pose = actor->getGlobalPose();
			if(!pose.isFinite())
			{
				metrics.nonFinite++;
				continue;
			}
			const PxReal quaternionNormError = PxAbs(pose.q.magnitudeSquared() - 1.0f);
			if(quaternionNormError > 1.0e-3f)
				metrics.quaternionMismatches++;
			const PxVec3 normalizedPosition = pose.p - PxVec3(0.0f, 0.0f, clone.laneOffset);
			if(normalizedPosition.magnitude() > gRunawayPositionLimit)
				metrics.runawaySamples++;
			PxRigidDynamic* dynamic = actor->is<PxRigidDynamic>();
			if(dynamic)
			{
				const PxVec3 linearVelocity = dynamic->getLinearVelocity();
				const PxVec3 angularVelocity = dynamic->getAngularVelocity();
				if(!linearVelocity.isFinite() || !angularVelocity.isFinite())
					metrics.nonFinite++;
				else if(linearVelocity.magnitude() > gRunawayVelocityLimit ||
				        angularVelocity.magnitude() > gRunawayVelocityLimit)
					metrics.runawaySamples++;
			}
		}

		if(clone.terminal)
		{
			const PxReal motion =
				(clone.terminal->getGlobalPose().p - clone.initialTerminalPosition).magnitude();
			clone.maxMotion = PxMax(clone.maxMotion, motion);
			metrics.maxTerminalMotion = PxMax(metrics.maxTerminalMotion, motion);
		}

		for(PxU32 i = 0; i < gExpectedJointCount; ++i)
		{
			PxSphericalJoint* joint = clone.joints[i];
			if(!joint)
				continue;
			if(joint->getConstraintFlags() & PxConstraintFlag::eBROKEN)
				metrics.brokenConstraints++;
			PxRigidActor* actor0 = NULL;
			PxRigidActor* actor1 = NULL;
			joint->getActors(actor0, actor1);
			if(!actor0 || !actor1)
			{
				metrics.endpointMismatches++;
				continue;
			}
			const PxTransform frame0 = getJointWorldFrame(*joint, PxJointActorIndex::eACTOR0, actor0);
			const PxTransform frame1 = getJointWorldFrame(*joint, PxJointActorIndex::eACTOR1, actor1);
			if(!frame0.isFinite() || !frame1.isFinite())
				metrics.nonFinite++;
			else
				metrics.maxAnchorError = PxMax(metrics.maxAnchorError,
				                               (frame0.p - frame1.p).magnitude());
			metrics.anchorSamples++;
		}
	}

	for(PxU32 i = 0; i < gExpectedActorCount; ++i)
	{
		PxRigidActor* actor0 = clones[0].actors[i];
		PxRigidActor* actor1 = clones[1].actors[i];
		if(!actor0 || !actor1)
			continue;
		const PxTransform pose0 = actor0->getGlobalPose();
		const PxTransform pose1 = actor1->getGlobalPose();
		if(!pose0.isFinite() || !pose1.isFinite())
		{
			metrics.nonFinite++;
			continue;
		}
		const PxVec3 normalized0 = pose0.p - PxVec3(0.0f, 0.0f, clones[0].laneOffset);
		const PxVec3 normalized1 = pose1.p - PxVec3(0.0f, 0.0f, clones[1].laneOffset);
		metrics.maxClonePositionError = PxMax(metrics.maxClonePositionError,
		                                      (normalized0 - normalized1).magnitude());
		metrics.maxCloneAngleError = PxMax(metrics.maxCloneAngleError,
		                                   quaternionAngle(pose0.q, pose1.q));
		PxRigidDynamic* dynamic0 = actor0->is<PxRigidDynamic>();
		PxRigidDynamic* dynamic1 = actor1->is<PxRigidDynamic>();
		if(dynamic0 && dynamic1)
		{
			const PxReal linearError =
				(dynamic0->getLinearVelocity() - dynamic1->getLinearVelocity()).magnitude();
			const PxReal angularError =
				(dynamic0->getAngularVelocity() - dynamic1->getAngularVelocity()).magnitude();
			metrics.maxCloneVelocityError = PxMax(metrics.maxCloneVelocityError,
			                                      PxMax(linearError, angularError));
		}
		metrics.cloneSamples++;
	}
}

static void releaseLoadedClone(LoadedClone& clone, SerializationMetrics& metrics)
{
	if(clone.collection)
	{
		// The static actor owns its exclusive sphere shape. Releasing that shape
		// while the actor is live would decrement an actor-held reference twice.
		// Hold one protective reference across actor teardown, then consume every
		// remaining loader/application reference after releaseObjects has removed
		// all collection entries. At that point no actor can reference the shape.
		PxShape* exclusiveShape = clone.exclusiveShape;
		if(exclusiveShape)
			exclusiveShape->acquireReference();
		PxCollectionExt::releaseObjects(*clone.collection, false);
		if(exclusiveShape)
		{
			while(exclusiveShape->getReferenceCount() > 1)
				exclusiveShape->release();
			exclusiveShape->release();
		}
		clone.collection->release();
		clone.collection = NULL;
		metrics.collectionsReleased++;
	}
	freeBinaryBlock(clone.block, metrics);
}

static void releaseLoadedShared(LoadedShared& shared, SerializationMetrics& metrics)
{
	if(shared.collection)
	{
		PxCollectionExt::releaseObjects(*shared.collection, true);
		shared.collection->release();
		shared.collection = NULL;
		metrics.collectionsReleased++;
	}
	freeBinaryBlock(shared.block, metrics);
}

static void cleanupCycle(LoadedShared& shared, LoadedClone clones[2],
	                     SerializationMetrics& metrics)
{
	// Actor objects are placement-deserialized into their own binary blocks.
	// Both actor lanes must be gone before the shared external objects, and a
	// backing block must outlive every object placed in it.
	releaseLoadedClone(clones[0], metrics);
	releaseLoadedClone(clones[1], metrics);
	releaseLoadedShared(shared, metrics);
	if(gScene &&
	   (gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC |
	                        PxActorTypeFlag::eRIGID_DYNAMIC) != 0 ||
	    gScene->getNbConstraints() != 0))
		metrics.sceneCleanupMismatches++;
	if(gPhysics)
	{
		metrics.remainingConstraints = PxMax(metrics.remainingConstraints,
		                                    gPhysics->getNbConstraints());
		metrics.remainingShapes = PxMax(metrics.remainingShapes, gPhysics->getNbShapes());
		metrics.remainingMaterials = PxMax(metrics.remainingMaterials,
		                                  gPhysics->getNbMaterials());
		if(gPhysics->getNbConstraints() != 0 || gPhysics->getNbShapes() != 0 ||
		   gPhysics->getNbMaterials() != 0)
			metrics.sceneCleanupMismatches++;
	}
	if(metrics.sceneCleanupMismatches != 0)
		setInfrastructureError(metrics, "cycle_cleanup");
}

static bool executeCycle(const std::vector<PxU8>& sharedBytes,
	                    const std::vector<PxU8>& actorBytes,
	                    SerializationFormat format,
	                    PxSerializationRegistry& registry,
	                    const Snippets::HeadlessOptions& options,
	                    SerializationMetrics& metrics)
{
	LoadedShared shared;
	LoadedClone clones[2];
	shared.collection = deserializeCollection(sharedBytes, format, registry, NULL,
	                                          shared.block, "deserialize_shared", metrics);
	bool sharedValid = validateSharedCollection(shared, metrics);
	if(shared.collection)
	{
		clones[0].collection = deserializeCollection(actorBytes, format, registry,
		                                               shared.collection, clones[0].block,
		                                               "deserialize_actor", metrics);
		clones[1].collection = deserializeCollection(actorBytes, format, registry,
		                                               shared.collection, clones[1].block,
		                                               "deserialize_actor", metrics);
	}
	bool clone0Valid = resolveAndValidateClone(clones[0], shared, -gLaneOffset, metrics);
	bool clone1Valid = resolveAndValidateClone(clones[1], shared, gLaneOffset, metrics);
	if(clones[0].collection && clones[1].collection)
		validateCloneIsolation(clones[0], clones[1], shared, metrics);

	for(PxU32 lane = 0; lane < 2; ++lane)
	{
		if(clones[lane].collection)
			gScene->addCollection(*clones[lane].collection);
	}
	if(gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC |
	                       PxActorTypeFlag::eRIGID_DYNAMIC) != 2 * gExpectedActorCount ||
	   gScene->getNbConstraints() != 2 * gExpectedJointCount)
		metrics.scenePopulationMismatches++;

	bool cycleComplete = false;
	if(!metrics.infrastructureError && sharedValid && clone0Valid && clone1Valid)
	{
		clones[0].terminal->addForce(PxVec3(0.0f, 20.0f, 0.0f), PxForceMode::eIMPULSE);
		clones[1].terminal->addForce(PxVec3(0.0f, 20.0f, 0.0f), PxForceMode::eIMPULSE);
		PxU32 completedInCycle = 0;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			gScene->simulate(options.dt);
			metrics.simulateCalls++;
			PxU32 errorState = 0;
			const bool fetched = gScene->fetchResults(true, &errorState);
			metrics.fetchCalls++;
			if(!fetched || errorState != 0)
			{
				metrics.fetchFailures++;
				setInfrastructureError(metrics, "fetch_results");
				break;
			}
			metrics.completedFrames++;
			completedInCycle++;
			samplePhysics(clones, metrics);
		}
		cycleComplete = completedInCycle == options.frames;
		for(PxU32 lane = 0; lane < 2; ++lane)
		{
			if(clones[lane].maxMotion >= gMotionMinimum)
				metrics.motionWitnesses++;
		}
	}

	cleanupCycle(shared, clones, metrics);
	if(cycleComplete && metrics.sceneCleanupMismatches == 0)
		metrics.completedCycles++;
	return cycleComplete && !metrics.infrastructureError;
}

struct GateEvaluation
{
	bool serializationGate;
	bool structureGate;
	bool physicsGate;
	bool lifecycleGate;
	const char* status;
	std::string reason;
	int exitCode;

	GateEvaluation()
	: serializationGate(false), structureGate(false), physicsGate(false), lifecycleGate(false),
	  status("ERROR"), reason("infrastructure_error"),
	  exitCode(Snippets::eHEADLESS_CONFIG_ERROR)
	{
	}
};

static GateEvaluation evaluateGates(const SerializationMetrics& metrics,
	                                SerializationFormat format)
{
	GateEvaluation evaluation;
	const PxU64 expectedDeserializations = PxU64(metrics.requestedCycles) * 3;
	evaluation.serializationGate = metrics.serializeShared && metrics.serializeActors &&
	                               metrics.immutableBytes && metrics.authoringCleanup &&
	                               metrics.sharedBytes != 0 && metrics.actorBytes != 0 &&
	                               metrics.deserializeAttempts == expectedDeserializations &&
	                               metrics.deserializeSuccesses == expectedDeserializations;
	evaluation.structureGate =
		metrics.loadedActors == metrics.expectedActors &&
		metrics.loadedStatics == metrics.expectedStatics &&
		metrics.loadedDynamics == metrics.expectedDynamics &&
		metrics.loadedJoints == metrics.expectedJoints &&
		metrics.loadedConstraints == metrics.expectedConstraints &&
		metrics.sharedIdMismatches == 0 && metrics.actorIdMismatches == 0 &&
		metrics.jointIdMismatches == 0 && metrics.constraintNulls == 0 &&
		metrics.constraintDuplicates == 0 && metrics.endpointMismatches == 0 &&
		metrics.crossReferenceMismatches == 0 && metrics.graphMismatches == 0 &&
		metrics.externalReferenceMismatches == 0 &&
		metrics.exclusiveShapeMismatches == 0 && metrics.parameterMismatches == 0 &&
		metrics.scenePopulationMismatches == 0;
	evaluation.physicsGate = metrics.completedFrames == metrics.requestedFrames &&
	                         metrics.completedCycles == metrics.requestedCycles &&
	                         metrics.simulateCalls == metrics.requestedFrames &&
	                         metrics.fetchCalls == metrics.requestedFrames &&
	                         metrics.fetchFailures == 0 && metrics.nonFinite == 0 &&
	                         metrics.runawaySamples == 0 && metrics.quaternionMismatches == 0 &&
	                         metrics.brokenConstraints == 0 &&
	                         metrics.anchorSamples == metrics.requestedFrames * 2 * gExpectedJointCount &&
	                         metrics.cloneSamples == metrics.requestedFrames * gExpectedActorCount &&
	                         metrics.motionWitnesses == metrics.expectedMotionWitnesses &&
	                         metrics.maxAnchorError <= gAnchorErrorLimit &&
	                         metrics.maxClonePositionError <= gClonePositionErrorLimit &&
	                         metrics.maxCloneVelocityError <= gCloneVelocityErrorLimit &&
	                         metrics.maxCloneAngleError <= gCloneAngleErrorLimit;
	const PxU64 expectedBinaryBlocks = PxU64(metrics.requestedCycles) * 3;
	const PxU64 expectedBinaryBytes = PxU64(metrics.requestedCycles) *
	                                    (metrics.sharedBytes + 2 * metrics.actorBytes);
	const bool binaryAccounting =
		format == eSERIALIZATION_FORMAT_BINARY
			? metrics.binaryBlocksAllocated == expectedBinaryBlocks &&
			  metrics.binaryBlocksFreed == expectedBinaryBlocks &&
			  metrics.binaryBytesAllocated == expectedBinaryBytes &&
			  metrics.binaryBytesFreed == expectedBinaryBytes &&
			  metrics.alignmentFailures == 0 && metrics.shortReads == 0
			: metrics.binaryBlocksAllocated == 0 && metrics.binaryBlocksFreed == 0 &&
			  metrics.binaryBytesAllocated == 0 && metrics.binaryBytesFreed == 0 &&
			  metrics.alignmentFailures == 0 && metrics.shortReads == 0;
	evaluation.lifecycleGate =
	                           metrics.collectionsCreated == 2 + expectedDeserializations &&
	                           metrics.collectionsCreated == metrics.collectionsReleased &&
	                           binaryAccounting && metrics.sceneCleanupMismatches == 0 &&
	                           metrics.finalObjectMismatches == 0 && metrics.pvdCreated == 0;

	if(metrics.infrastructureError)
	{
		evaluation.status = "ERROR";
		evaluation.reason = makeToken(metrics.infrastructureReason);
		evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
	}
	else if(!evaluation.serializationGate)
	{
		evaluation.status = "ERROR";
		evaluation.reason = "serialization_gate";
		evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
	}
	else if(!evaluation.lifecycleGate)
	{
		evaluation.status = "ERROR";
		evaluation.reason = "lifecycle_gate";
		evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
	}
	else if(!evaluation.structureGate)
	{
		evaluation.status = "FAIL";
		evaluation.reason = "semantic_structure";
		evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
	}
	else if(!evaluation.physicsGate)
	{
		evaluation.status = "FAIL";
		evaluation.reason = metrics.nonFinite ? "non_finite" : "physical_oracle";
		evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
	}
	else
	{
		evaluation.status = "PASS";
		evaluation.reason = "none";
		evaluation.exitCode = Snippets::eHEADLESS_PASS;
	}
	return evaluation;
}

static int runHeadless(int argc, const char* const* argv)
{
	Snippets::HeadlessOptions options;
	SerializationHeadlessConfig config;
	std::string parseError;
	if(!parseHeadlessConfiguration(argc, argv, options, config, parseError))
		return reportConfigurationError(options, config, parseError);
	if(!Snippets::applyExecutionEnvironment(options))
		return reportConfigurationError(options, config, "execution environment update failed");

	Snippets::printHeadlessConfig("SnippetSerialization", options);
	SerializationMetrics metrics;
	metrics.requestedFrames = config.requestedFrames;
	metrics.requestedCycles = config.cycles;
	metrics.expectedActors = PxU64(config.cycles) * 2 * gExpectedActorCount;
	metrics.expectedStatics = PxU64(config.cycles) * 2;
	metrics.expectedDynamics = PxU64(config.cycles) * 2 * gExpectedDynamicCount;
	metrics.expectedJoints = PxU64(config.cycles) * 2 * gExpectedJointCount;
	metrics.expectedConstraints = metrics.expectedJoints;
	metrics.expectedMotionWitnesses = PxU64(config.cycles) * 2;

	PxSerializationRegistry* registry = NULL;
	std::vector<PxU8> sharedBytes;
	std::vector<PxU8> actorBytes;
	if(initHeadlessPhysics(options, metrics))
	{
		registry = PxSerialization::createSerializationRegistry(*gPhysics);
		if(!registry)
			setInfrastructureError(metrics, "registry_create");
	}
	if(registry && !metrics.infrastructureError)
		serializeAuthoringCollections(config.format, *registry, sharedBytes, actorBytes, metrics);
	if(registry && !metrics.infrastructureError)
	{
		for(PxU32 cycle = 0; cycle < config.cycles; ++cycle)
		{
			executeCycle(sharedBytes, actorBytes, config.format, *registry, options, metrics);
			// A semantic/topology failure is a FAIL/1 and must not truncate the
			// requested repeat lifecycle. Only an operational error makes later
			// deserialization or cleanup unsafe.
			if(metrics.infrastructureError)
				break;
		}
	}
	metrics.immutableBytes = metrics.immutableBytes &&
	                         metrics.sharedByteHash == hashBytes(sharedBytes) &&
	                         metrics.actorByteHash == hashBytes(actorBytes);
	PX_RELEASE(registry);
	cleanupHeadlessPhysics(metrics);
	const PxU32 physicsErrors = gHeadlessErrorCallback.getFatalCount();
	const PxU32 physicsWarnings = gHeadlessErrorCallback.getWarningCount();
	if(physicsErrors != 0)
		setInfrastructureError(metrics, "physics_error");

	const GateEvaluation evaluation = evaluateGates(metrics, config.format);
	printf("[AVBD_GATE] schema=1 snippet=SnippetSerialization case=%s solver=%s "
	       "execution=%s requestedFrames=%llu completedFrames=%llu dt=%.9g seed=%u "
	       "dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s reason=%s "
	       "format=%s cycles=%u completedCycles=%u framesPerCycle=%u "
	       "serializeShared=%u serializeActors=%u sharedBytes=%llu actorBytes=%llu "
	       "sharedByteHash=%llu actorByteHash=%llu "
	       "immutableBytes=%u authoringCleanup=%u deserializeAttempts=%llu "
	       "deserializeSuccesses=%llu expectedActors=%llu loadedActors=%llu "
	       "expectedStatics=%llu loadedStatics=%llu expectedDynamics=%llu loadedDynamics=%llu "
	       "expectedJoints=%llu loadedJoints=%llu expectedConstraints=%llu loadedConstraints=%llu "
	       "sharedIdMismatches=%llu actorIdMismatches=%llu jointIdMismatches=%llu "
	       "constraintNulls=%llu constraintDuplicates=%llu endpointMismatches=%llu "
	       "crossReferenceMismatches=%llu graphMismatches=%llu "
	       "externalReferenceMismatches=%llu exclusiveShapeMismatches=%llu "
	       "parameterMismatches=%llu scenePopulationMismatches=%llu projectionParams=NOT_APPLICABLE "
	       "motionWitnesses=%llu expectedMotionWitnesses=%llu maxTerminalMotion=%.9g "
	       "anchorSamples=%llu maxAnchorError=%.9g cloneSamples=%llu "
	       "maxClonePositionError=%.9g maxCloneVelocityError=%.9g maxCloneAngleError=%.9g "
	       "runawaySamples=%llu quaternionMismatches=%llu brokenConstraints=%llu "
	       "simulateCalls=%llu fetchCalls=%llu "
	       "fetchFailures=%llu collectionsCreated=%llu collectionsReleased=%llu "
	       "binaryBlocksAllocated=%llu binaryBlocksFreed=%llu binaryBytesAllocated=%llu "
	       "binaryBytesFreed=%llu alignmentFailures=%llu shortReads=%llu "
	       "sceneCleanupMismatches=%llu finalObjectMismatches=%llu remainingConstraints=%u "
	       "remainingShapes=%u remainingMaterials=%u pvdCreated=%u "
	       "serializationGate=%s structureGate=%s physicsGate=%s lifecycleGate=%s "
	       "nonFinite=%llu physicsErrors=%u physicsWarnings=%u\n",
	       options.caseName.c_str(), Snippets::getSolverTypeName(options.solverType),
	       Snippets::getExecutionName(options.execution),
	       static_cast<unsigned long long>(metrics.requestedFrames),
	       static_cast<unsigned long long>(metrics.completedFrames), double(options.dt), options.seed,
	       options.dispatcherThreads, evaluation.status, evaluation.reason.c_str(),
	       getFormatName(config.format), config.cycles, metrics.completedCycles, options.frames,
	       metrics.serializeShared ? 1u : 0u, metrics.serializeActors ? 1u : 0u,
	       static_cast<unsigned long long>(metrics.sharedBytes),
	       static_cast<unsigned long long>(metrics.actorBytes),
	       static_cast<unsigned long long>(metrics.sharedByteHash),
	       static_cast<unsigned long long>(metrics.actorByteHash), metrics.immutableBytes ? 1u : 0u,
	       metrics.authoringCleanup ? 1u : 0u,
	       static_cast<unsigned long long>(metrics.deserializeAttempts),
	       static_cast<unsigned long long>(metrics.deserializeSuccesses),
	       static_cast<unsigned long long>(metrics.expectedActors),
	       static_cast<unsigned long long>(metrics.loadedActors),
	       static_cast<unsigned long long>(metrics.expectedStatics),
	       static_cast<unsigned long long>(metrics.loadedStatics),
	       static_cast<unsigned long long>(metrics.expectedDynamics),
	       static_cast<unsigned long long>(metrics.loadedDynamics),
	       static_cast<unsigned long long>(metrics.expectedJoints),
	       static_cast<unsigned long long>(metrics.loadedJoints),
	       static_cast<unsigned long long>(metrics.expectedConstraints),
	       static_cast<unsigned long long>(metrics.loadedConstraints),
	       static_cast<unsigned long long>(metrics.sharedIdMismatches),
	       static_cast<unsigned long long>(metrics.actorIdMismatches),
	       static_cast<unsigned long long>(metrics.jointIdMismatches),
	       static_cast<unsigned long long>(metrics.constraintNulls),
	       static_cast<unsigned long long>(metrics.constraintDuplicates),
	       static_cast<unsigned long long>(metrics.endpointMismatches),
	       static_cast<unsigned long long>(metrics.crossReferenceMismatches),
	       static_cast<unsigned long long>(metrics.graphMismatches),
	       static_cast<unsigned long long>(metrics.externalReferenceMismatches),
	       static_cast<unsigned long long>(metrics.exclusiveShapeMismatches),
	       static_cast<unsigned long long>(metrics.parameterMismatches),
	       static_cast<unsigned long long>(metrics.scenePopulationMismatches),
	       static_cast<unsigned long long>(metrics.motionWitnesses),
	       static_cast<unsigned long long>(metrics.expectedMotionWitnesses),
	       double(metrics.maxTerminalMotion),
	       static_cast<unsigned long long>(metrics.anchorSamples), double(metrics.maxAnchorError),
	       static_cast<unsigned long long>(metrics.cloneSamples),
	       double(metrics.maxClonePositionError), double(metrics.maxCloneVelocityError),
	       double(metrics.maxCloneAngleError),
	       static_cast<unsigned long long>(metrics.runawaySamples),
	       static_cast<unsigned long long>(metrics.quaternionMismatches),
	       static_cast<unsigned long long>(metrics.brokenConstraints),
	       static_cast<unsigned long long>(metrics.simulateCalls),
	       static_cast<unsigned long long>(metrics.fetchCalls),
	       static_cast<unsigned long long>(metrics.fetchFailures),
	       static_cast<unsigned long long>(metrics.collectionsCreated),
	       static_cast<unsigned long long>(metrics.collectionsReleased),
	       static_cast<unsigned long long>(metrics.binaryBlocksAllocated),
	       static_cast<unsigned long long>(metrics.binaryBlocksFreed),
	       static_cast<unsigned long long>(metrics.binaryBytesAllocated),
	       static_cast<unsigned long long>(metrics.binaryBytesFreed),
	       static_cast<unsigned long long>(metrics.alignmentFailures),
	       static_cast<unsigned long long>(metrics.shortReads),
	       static_cast<unsigned long long>(metrics.sceneCleanupMismatches),
	       static_cast<unsigned long long>(metrics.finalObjectMismatches),
	       metrics.remainingConstraints, metrics.remainingShapes, metrics.remainingMaterials,
	       metrics.pvdCreated,
	       evaluation.serializationGate ? "PASS" : "FAIL",
	       evaluation.structureGate ? "PASS" : "FAIL",
	       evaluation.physicsGate ? "PASS" : "FAIL",
	       evaluation.lifecycleGate ? "PASS" : "FAIL",
	       static_cast<unsigned long long>(metrics.nonFinite), physicsErrors, physicsWarnings);
	return evaluation.exitCode;
}

} // anonymous namespace




int snippetMain(int argc, const char*const* argv)
{
	if(isHeadlessInvocation(argc, argv))
		return runHeadless(argc, argv);

	initPhysics();
	// Alternatively PxDefaultFileOutputStream could be used 
	PxDefaultMemoryOutputStream sharedOutputStream;
	PxDefaultMemoryOutputStream actorOutputStream;
	serializeObjects(sharedOutputStream, actorOutputStream);
	cleanupPhysics();

	initPhysics();
	// Alternatively PxDefaultFileInputData could be used 
	PxDefaultMemoryInputData sharedInputStream(sharedOutputStream.getData(), sharedOutputStream.getSize());
	PxDefaultMemoryInputData actorInputStream(actorOutputStream.getData(), actorOutputStream.getSize());

	PxTolerancesScale scale;
	const PxCookingParams params(scale);

	deserializeObjects(sharedInputStream, actorInputStream, params);
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	static const PxU32 frameCount = 250;
	for(PxU32 i=0; i<frameCount; i++)
		stepPhysics();
	cleanupPhysics();
	printf("SnippetSerialization done.\n");
#endif

	return 0;
}
