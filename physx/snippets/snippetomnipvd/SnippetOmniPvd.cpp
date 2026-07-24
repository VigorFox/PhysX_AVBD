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
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetutils/SnippetUtils.h"
#include "omnipvd/PxOmniPvd.h"

#if PX_SUPPORT_OMNI_PVD
#include "../pvdruntime/include/OmniPvdCommands.h"
#include "../pvdruntime/include/OmniPvdFileReadStream.h"
#include "../pvdruntime/include/OmniPvdFileWriteStream.h"
#include "../pvdruntime/include/OmniPvdReader.h"
#include "../pvdruntime/include/OmniPvdWriter.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>

extern "C"
{
	OmniPvdReader* OMNI_PVD_CALL createOmniPvdReader();
	void OMNI_PVD_CALL destroyOmniPvdReader(OmniPvdReader& reader);
	OmniPvdFileReadStream* OMNI_PVD_CALL createOmniPvdFileReadStream();
	void OMNI_PVD_CALL destroyOmniPvdFileReadStream(
		OmniPvdFileReadStream& stream);
}

using namespace physx;

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static PxOmniPvd* gOmniPvd = NULL;
static PxRigidDynamic* gProjectile = NULL;

static Snippets::HeadlessOptions gHeadlessOptions;
static std::string gOmniPvdPathStorage;
const char* gOmniPvdPath = NULL;

struct OmniPvdGateMetrics
{
	PxU32 samplingStarted;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 initialStaticActors;
	PxU32 initialDynamicActors;
	PxReal minProjectileY;
	PxReal maxProjectileSpeed;
	PxVec3 initialProjectilePosition;
	PxVec3 finalProjectilePosition;
	PxVec3 finalProjectileVelocity;
	PxU64 fileBytes;
	PxU32 readerStarted;
	PxU32 versionMajor;
	PxU32 versionMinor;
	PxU32 versionPatch;
	PxU64 commandCount;
	PxU64 classRegistrations;
	PxU64 attributeRegistrations;
	PxU64 setAttributes;
	PxU64 createObjects;
	PxU64 destroyObjects;
	PxU64 startFrames;
	PxU64 stopFrames;
	PxU32 requiredClasses;
	PxU32 physicsCreates;
	PxU32 sceneCreates;
	PxU32 materialCreates;
	PxU32 rigidStaticCreates;
	PxU32 rigidDynamicCreates;
	PxU32 shapeCreates;
	PxU32 physicsDestroys;
	PxU32 sceneDestroys;
	PxU32 materialDestroys;
	PxU32 rigidStaticDestroys;
	PxU32 rigidDynamicDestroys;
	PxU32 shapeDestroys;
	PxU32 solverTypeSamples;
	PxU32 solverTypeMatches;
	PxU32 solverTypeMismatches;
	PxU32 cleanupComplete;

	OmniPvdGateMetrics()
	: samplingStarted(0), completedFrames(0), fetchFailures(0),
	  nonFinite(0), initialStaticActors(0), initialDynamicActors(0),
	  minProjectileY(PX_MAX_F32), maxProjectileSpeed(0.0f),
	  initialProjectilePosition(0.0f), finalProjectilePosition(0.0f),
	  finalProjectileVelocity(0.0f), fileBytes(0), readerStarted(0),
	  versionMajor(0), versionMinor(0), versionPatch(0),
	  commandCount(0), classRegistrations(0),
	  attributeRegistrations(0), setAttributes(0), createObjects(0),
	  destroyObjects(0), startFrames(0), stopFrames(0),
	  requiredClasses(0), physicsCreates(0), sceneCreates(0),
	  materialCreates(0), rigidStaticCreates(0),
	  rigidDynamicCreates(0), shapeCreates(0), physicsDestroys(0),
	  sceneDestroys(0), materialDestroys(0), rigidStaticDestroys(0),
	  rigidDynamicDestroys(0), shapeDestroys(0), solverTypeSamples(0),
	  solverTypeMatches(0), solverTypeMismatches(0),
	  cleanupComplete(0)
	{
	}
};

static OmniPvdGateMetrics gMetrics;

struct RequiredClassHandles
{
	OmniPvdClassHandle physics;
	OmniPvdClassHandle scene;
	OmniPvdClassHandle material;
	OmniPvdClassHandle rigidStatic;
	OmniPvdClassHandle rigidDynamic;
	OmniPvdClassHandle shape;
	OmniPvdAttributeHandle solverType;

	RequiredClassHandles()
	: physics(OMNI_PVD_INVALID_HANDLE),
	  scene(OMNI_PVD_INVALID_HANDLE),
	  material(OMNI_PVD_INVALID_HANDLE),
	  rigidStatic(OMNI_PVD_INVALID_HANDLE),
	  rigidDynamic(OMNI_PVD_INVALID_HANDLE),
	  shape(OMNI_PVD_INVALID_HANDLE),
	  solverType(OMNI_PVD_INVALID_HANDLE)
	{
	}
};

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), name);
}

static bool parseOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 120;
	defaults.caseName = "record-scene";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;

	bool outputSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		static const char prefix[] = "--omnipvdfile=";
		if(arg && std::strncmp(arg, prefix, sizeof(prefix) - 1) == 0)
		{
			if(outputSeen)
			{
				error = "duplicate --omnipvdfile";
				return false;
			}
			outputSeen = true;
			gOmniPvdPathStorage = arg + sizeof(prefix) - 1;
			if(gOmniPvdPathStorage.empty())
			{
				error = "empty --omnipvdfile";
				return false;
			}
			gOmniPvdPath = gOmniPvdPathStorage.c_str();
		}
		else
		{
			error = std::string("unknown option: ") +
				(arg ? arg : "<null>");
			return false;
		}
	}
	if(!outputSeen)
	{
		error = "missing --omnipvdfile";
		return false;
	}
	if(!isHeadlessCase("record-scene"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.headless && gHeadlessOptions.frames < 60)
	{
		error = "--frames must be at least 60";
		return false;
	}
	return true;
}

static PxRigidDynamic* createDynamic(
	const PxTransform& transform, const PxGeometry& geometry,
	const PxVec3& velocity = PxVec3(0))
{
	PxRigidDynamic* dynamic = PxCreateDynamic(
		*gPhysics, transform, geometry, *gMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void initPhysXScene()
{
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless
			? gHeadlessOptions.dispatcherThreads : 2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	PxRigidStatic* groundPlane = PxCreatePlane(
		*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*groundPlane);
	const PxTransform projectilePose(PxVec3(0, 40, 100));
	gProjectile = createDynamic(
		projectilePose, PxSphereGeometry(10),
		PxVec3(0, -50, -100));
	if(gHeadlessOptions.headless)
	{
		gMetrics.initialProjectilePosition = projectilePose.p;
		gMetrics.finalProjectilePosition = projectilePose.p;
		gMetrics.initialStaticActors =
			gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
		gMetrics.initialDynamicActors =
			gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
	}
}

void initPhysicsWithOmniPvd()
{
	gFoundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return;

	gOmniPvd = PxCreateOmniPvd(*gFoundation);
	if(!gOmniPvd)
		return;
	OmniPvdWriter* omniWriter = gOmniPvd->getWriter();
	OmniPvdFileWriteStream* fileStream =
		gOmniPvd->getFileWriteStream();
	if(!omniWriter || !fileStream)
		return;
	fileStream->setFileName(gOmniPvdPath);
	omniWriter->setWriteStream(
		static_cast<OmniPvdWriteStream&>(*fileStream));

	gPhysics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true,
		NULL, gOmniPvd);
	if(!gPhysics || !gPhysics->getOmniPvd())
		return;

	gMetrics.samplingStarted =
		gPhysics->getOmniPvd()->startSampling() ? 1u : 0u;
	if(!gMetrics.samplingStarted)
		return;

	initPhysXScene();
}

void stepPhysics()
{
	const PxReal dt = gHeadlessOptions.headless
		? gHeadlessOptions.dt : 1.0f / 60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gProjectile)
		{
			const PxTransform pose = gProjectile->getGlobalPose();
			const PxVec3 linearVelocity =
				gProjectile->getLinearVelocity();
			const PxVec3 angularVelocity =
				gProjectile->getAngularVelocity();
			gMetrics.finalProjectilePosition = pose.p;
			gMetrics.finalProjectileVelocity = linearVelocity;
			gMetrics.minProjectileY = PxMin(
				gMetrics.minProjectileY, pose.p.y);
			gMetrics.maxProjectileSpeed = PxMax(
				gMetrics.maxProjectileSpeed,
				linearVelocity.magnitude());
			if(!pose.isFinite() || !linearVelocity.isFinite() ||
				!angularVelocity.isFinite())
				++gMetrics.nonFinite;
		}
		++gMetrics.completedFrames;
	}
}

static PxU64 getFileSize(const char* path)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if(!file)
		return 0;
	const std::streampos size = file.tellg();
	return size > 0 ? static_cast<PxU64>(size) : 0;
}

static void recordRequiredClass(
	const char* name, OmniPvdClassHandle handle,
	RequiredClassHandles& handles)
{
	if(std::strcmp(name, "PxPhysics") == 0)
		handles.physics = handle;
	else if(std::strcmp(name, "PxScene") == 0)
		handles.scene = handle;
	else if(std::strcmp(name, "PxMaterial") == 0)
		handles.material = handle;
	else if(std::strcmp(name, "PxRigidStatic") == 0)
		handles.rigidStatic = handle;
	else if(std::strcmp(name, "PxRigidDynamic") == 0)
		handles.rigidDynamic = handle;
	else if(std::strcmp(name, "PxShape") == 0)
		handles.shape = handle;
}

static void recordCreate(
	OmniPvdClassHandle classHandle, const RequiredClassHandles& handles)
{
	if(classHandle == handles.physics)
		++gMetrics.physicsCreates;
	else if(classHandle == handles.scene)
		++gMetrics.sceneCreates;
	else if(classHandle == handles.material)
		++gMetrics.materialCreates;
	else if(classHandle == handles.rigidStatic)
		++gMetrics.rigidStaticCreates;
	else if(classHandle == handles.rigidDynamic)
		++gMetrics.rigidDynamicCreates;
	else if(classHandle == handles.shape)
		++gMetrics.shapeCreates;
}

static void recordDestroy(
	OmniPvdClassHandle classHandle, const RequiredClassHandles& handles)
{
	if(classHandle == handles.physics)
		++gMetrics.physicsDestroys;
	else if(classHandle == handles.scene)
		++gMetrics.sceneDestroys;
	else if(classHandle == handles.material)
		++gMetrics.materialDestroys;
	else if(classHandle == handles.rigidStatic)
		++gMetrics.rigidStaticDestroys;
	else if(classHandle == handles.rigidDynamic)
		++gMetrics.rigidDynamicDestroys;
	else if(classHandle == handles.shape)
		++gMetrics.shapeDestroys;
}

static void parseOmniPvdOutput()
{
	gMetrics.fileBytes = getFileSize(gOmniPvdPath);
	OmniPvdReader* reader = createOmniPvdReader();
	OmniPvdFileReadStream* stream = createOmniPvdFileReadStream();
	if(!reader || !stream)
	{
		if(reader)
			destroyOmniPvdReader(*reader);
		if(stream)
			destroyOmniPvdFileReadStream(*stream);
		return;
	}

	stream->setFileName(gOmniPvdPath);
	reader->setReadStream(*stream);
	OmniPvdVersionType major = 0;
	OmniPvdVersionType minor = 0;
	OmniPvdVersionType patch = 0;
	if(!reader->startReading(major, minor, patch))
	{
		destroyOmniPvdReader(*reader);
		destroyOmniPvdFileReadStream(*stream);
		return;
	}
	gMetrics.readerStarted = 1;
	gMetrics.versionMajor = major;
	gMetrics.versionMinor = minor;
	gMetrics.versionPatch = patch;

	RequiredClassHandles handles;
	std::unordered_map<OmniPvdObjectHandle, OmniPvdClassHandle>
		objectClasses;
	for(;;)
	{
		const OmniPvdCommand::Enum command = reader->getNextCommand();
		if(command == OmniPvdCommand::eINVALID)
			break;
		++gMetrics.commandCount;
		switch(command)
		{
		case OmniPvdCommand::eREGISTER_CLASS:
			++gMetrics.classRegistrations;
			recordRequiredClass(
				reader->getClassName(), reader->getClassHandle(), handles);
			break;
		case OmniPvdCommand::eREGISTER_ATTRIBUTE:
		case OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE:
		case OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE:
			++gMetrics.attributeRegistrations;
			if(reader->getClassHandle() == handles.scene &&
				std::strcmp(
					reader->getAttributeName(), "solverType") == 0)
				handles.solverType = reader->getAttributeHandle();
			break;
		case OmniPvdCommand::eSET_ATTRIBUTE:
		{
			++gMetrics.setAttributes;
			const OmniPvdObjectHandle objectHandle =
				reader->getObjectHandle();
			const std::unordered_map<
				OmniPvdObjectHandle,
				OmniPvdClassHandle>::const_iterator found =
				objectClasses.find(objectHandle);
			if(found != objectClasses.end() &&
				found->second == handles.scene &&
				reader->getAttributeHandle() == handles.solverType)
			{
				++gMetrics.solverTypeSamples;
				PxSolverType::Enum recorded =
					PxSolverType::ePGS;
				if(reader->getAttributeDataLength() >=
					sizeof(recorded) &&
					reader->getAttributeDataPointer())
				{
					std::memcpy(
						&recorded, reader->getAttributeDataPointer(),
						sizeof(recorded));
					if(recorded == gHeadlessOptions.solverType)
						++gMetrics.solverTypeMatches;
					else
						++gMetrics.solverTypeMismatches;
				}
				else
					++gMetrics.solverTypeMismatches;
			}
			break;
		}
		case OmniPvdCommand::eCREATE_OBJECT:
		{
			++gMetrics.createObjects;
			const OmniPvdClassHandle classHandle =
				reader->getClassHandle();
			objectClasses[reader->getObjectHandle()] = classHandle;
			recordCreate(classHandle, handles);
			break;
		}
		case OmniPvdCommand::eDESTROY_OBJECT:
		{
			++gMetrics.destroyObjects;
			const std::unordered_map<
				OmniPvdObjectHandle,
				OmniPvdClassHandle>::const_iterator found =
				objectClasses.find(reader->getObjectHandle());
			if(found != objectClasses.end())
			{
				recordDestroy(found->second, handles);
				objectClasses.erase(reader->getObjectHandle());
			}
			break;
		}
		case OmniPvdCommand::eSTART_FRAME:
			++gMetrics.startFrames;
			break;
		case OmniPvdCommand::eSTOP_FRAME:
			++gMetrics.stopFrames;
			break;
		default:
			break;
		}
	}

	gMetrics.requiredClasses =
		handles.physics != OMNI_PVD_INVALID_HANDLE &&
		handles.scene != OMNI_PVD_INVALID_HANDLE &&
		handles.material != OMNI_PVD_INVALID_HANDLE &&
		handles.rigidStatic != OMNI_PVD_INVALID_HANDLE &&
		handles.rigidDynamic != OMNI_PVD_INVALID_HANDLE &&
		handles.shape != OMNI_PVD_INVALID_HANDLE ? 6u : 0u;

	stream->closeFile();
	destroyOmniPvdReader(*reader);
	destroyOmniPvdFileReadStream(*stream);
}

void cleanupPhysics()
{
	gProjectile = NULL;
	PX_RELEASE(gScene);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	PX_RELEASE(gOmniPvd);

	if(gHeadlessOptions.headless)
		parseOmniPvdOutput();

	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gMaterial && !gDispatcher && !gPhysics &&
		!gOmniPvd && !gFoundation && !gProjectile ? 1u : 0u;
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case ' ':
		createDynamic(
			camera, PxSphereGeometry(3.0f),
			camera.rotate(PxVec3(0, 0, -1)) * 200);
		break;
	}
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig("SnippetOmniPvd", gHeadlessOptions);
	initPhysicsWithOmniPvd();
	const bool initialized =
		gFoundation && gOmniPvd && gPhysics && gDispatcher && gScene &&
		gMaterial && gProjectile && gMetrics.samplingStarted &&
		gMetrics.initialStaticActors == 1 &&
		gMetrics.initialDynamicActors == 1;
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics();
	}

	cleanupPhysics();
	const PxReal projectileDisplacement =
		(gMetrics.finalProjectilePosition -
			gMetrics.initialProjectilePosition).magnitude();

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
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0 ||
		!gMetrics.finalProjectilePosition.isFinite() ||
		!gMetrics.finalProjectileVelocity.isFinite() ||
		projectileDisplacement <= 1.0f)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.fileBytes < 1024 ||
		!gMetrics.readerStarted || gMetrics.commandCount == 0)
	{
		passed = false;
		reason = "invalid_omnipvd_file";
	}
	else if(gMetrics.requiredClasses != 6 ||
		gMetrics.physicsCreates == 0 || gMetrics.sceneCreates == 0 ||
		gMetrics.materialCreates == 0 ||
		gMetrics.rigidStaticCreates == 0 ||
		gMetrics.rigidDynamicCreates == 0 ||
		gMetrics.shapeCreates < 2)
	{
		passed = false;
		reason = "missing_omnipvd_metadata";
	}
	else if(gMetrics.physicsDestroys == 0 ||
		gMetrics.sceneDestroys == 0 ||
		gMetrics.materialDestroys == 0 ||
		gMetrics.rigidStaticDestroys == 0 ||
		gMetrics.rigidDynamicDestroys == 0 ||
		gMetrics.shapeDestroys < 2)
	{
		passed = false;
		reason = "missing_omnipvd_teardown";
	}
	else if(gMetrics.startFrames !=
			gHeadlessOptions.frames * 2u + 1u ||
		gMetrics.stopFrames != gMetrics.startFrames)
	{
		passed = false;
		reason = "omnipvd_frame_mismatch";
	}
	else if(gMetrics.solverTypeSamples == 0 ||
		gMetrics.solverTypeMatches != gMetrics.solverTypeSamples ||
		gMetrics.solverTypeMismatches != 0)
	{
		passed = false;
		reason = "omnipvd_solver_metadata_mismatch";
	}
	else if(!gMetrics.cleanupComplete)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetOmniPvd solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED samplingStarted=%u "
		"fileBytes=%llu readerStarted=%u version=%u.%u.%u "
		"commands=%llu classRegistrations=%llu "
		"attributeRegistrations=%llu setAttributes=%llu "
		"createObjects=%llu destroyObjects=%llu startFrames=%llu "
		"stopFrames=%llu requiredClasses=%u physicsCreates=%u "
		"sceneCreates=%u materialCreates=%u rigidStaticCreates=%u "
		"rigidDynamicCreates=%u shapeCreates=%u physicsDestroys=%u "
		"sceneDestroys=%u materialDestroys=%u rigidStaticDestroys=%u "
		"rigidDynamicDestroys=%u shapeDestroys=%u "
		"solverTypeSamples=%u solverTypeMatches=%u "
		"solverTypeMismatches=%u initialStaticActors=%u "
		"initialDynamicActors=%u minProjectileY=%.9g "
		"maxProjectileSpeed=%.9g projectileDisplacement=%.9g "
		"nonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, gMetrics.samplingStarted,
		static_cast<unsigned long long>(gMetrics.fileBytes),
		gMetrics.readerStarted, gMetrics.versionMajor,
		gMetrics.versionMinor, gMetrics.versionPatch,
		static_cast<unsigned long long>(gMetrics.commandCount),
		static_cast<unsigned long long>(gMetrics.classRegistrations),
		static_cast<unsigned long long>(gMetrics.attributeRegistrations),
		static_cast<unsigned long long>(gMetrics.setAttributes),
		static_cast<unsigned long long>(gMetrics.createObjects),
		static_cast<unsigned long long>(gMetrics.destroyObjects),
		static_cast<unsigned long long>(gMetrics.startFrames),
		static_cast<unsigned long long>(gMetrics.stopFrames),
		gMetrics.requiredClasses, gMetrics.physicsCreates,
		gMetrics.sceneCreates, gMetrics.materialCreates,
		gMetrics.rigidStaticCreates, gMetrics.rigidDynamicCreates,
		gMetrics.shapeCreates, gMetrics.physicsDestroys,
		gMetrics.sceneDestroys, gMetrics.materialDestroys,
		gMetrics.rigidStaticDestroys, gMetrics.rigidDynamicDestroys,
		gMetrics.shapeDestroys, gMetrics.solverTypeSamples,
		gMetrics.solverTypeMatches, gMetrics.solverTypeMismatches,
		gMetrics.initialStaticActors, gMetrics.initialDynamicActors,
		double(gMetrics.minProjectileY),
		double(gMetrics.maxProjectileSpeed),
		double(projectileDisplacement), gMetrics.nonFinite,
		gMetrics.fetchFailures, gErrorCallback.getFatalCount(),
		gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

#endif // PX_SUPPORT_OMNI_PVD

int snippetMain(int argc, const char* const* argv)
{
#if PX_SUPPORT_OMNI_PVD
	std::string error;
	if(!parseOptions(argc, argv, error))
	{
		std::fprintf(
			stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetOmniPvd "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(
			stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetOmniPvd "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysicsWithOmniPvd();
	static const PxU32 frameCount = 100;
	for(PxU32 i = 0; i < frameCount; ++i)
		stepPhysics();
	cleanupPhysics();
#endif
#else
	PX_UNUSED(argc);
	PX_UNUSED(argv);
	std::printf(
		"OmniPVD is not supported in release build configuration. "
		"Use debug, checked, or profile.\n");
	return Snippets::eHEADLESS_UNSUPPORTED;
#endif

	return 0;
}
