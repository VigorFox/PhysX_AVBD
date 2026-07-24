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
// This snippet illustrates loading xml or binary serialized collections and instantiating the objects in a scene.
//
// It only compiles and runs on authoring platforms (windows, osx and linux). 
// The snippet supports connecting to PVD in order to display the scene.
//
// It is a simple command-line tool supporting the following options::
// SnippetLoadCollection [--pvdhost=<ip address> ] [--pvdport=<ip port> ] [--pvdtimeout=<time ms> ] [--generateExampleFiles] <filename>...
//
// --pvdhost=<ip address>              Defines ip address of PVD, default is 127.0.0.1
// --pvdport=<ip port>                 Defines ip port of PVD, default is 5425
// --pvdtimeout=<time ms>              Defines time out of PVD, default is 10
// --generateExampleFiles              Generates a set of example files
//   <filename>...	                   Input files containing serialized collections (either xml or binary)
//
// Multiple collection files can be specified. The snippet is currently restricted to load a list of collections which obey
// the following rule: The first collection needs to be complete. All following collections - if any - may only maintain
// dependencies to objects in the first collection.
//
// The set of example files that can be generated consists of
// collection.xml|collection.bin: Can be used individually.
// collectionA.xml|collectionA.bin: Can also be used individually but only contain materials and shapes without actors.
// collectionB.xml|collectionB.bin: Need to be used together with collectionA files. The actors contained in collectionB
// maintain references to objects in the collectionA.
//
// ****************************************************************************

#include "PxPhysicsAPI.h"
#include <iostream>
#include <cfloat>
#include <vector>
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"

#define MAX_INPUT_FILES   16

using namespace physx;

static PxDefaultAllocator		    gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			    gFoundation = NULL;
static PxPhysics*				    gPhysics	= NULL;
static PxSerializationRegistry*		gSerializationRegistry = NULL;
static PxDefaultCpuDispatcher*		gDispatcher = NULL;
static PxScene*						gScene		= NULL;
static PxPvd*						gPvd        = NULL;

static PxU8*						gMemBlocks[MAX_INPUT_FILES];
static PxU32						gNbMemBlocks = 0;
static Snippets::HeadlessOptions	gHeadlessOptions;
static bool						gSolverReadbackMatched = false;
static PxU32						gCleanupComplete = 0;

struct CmdLineParameters
{
	const char*		pvdhost;		
	PxU32		    pvdport;			
	PxU32		    pvdtimeout;	
	const char*     inputFiles[MAX_INPUT_FILES];
	PxU32           nbFiles;
	bool			generateExampleFiles;

	CmdLineParameters()	: 
	      pvdhost(PVD_HOST)
		, pvdport(5425)
		, pvdtimeout(10)
		, nbFiles(0)
		, generateExampleFiles(false)
	{}
} gParameters;

static bool match(const char* opt, const char* ref)
{
	std::string s1(opt);
	std::string s2(ref);
	return !s1.compare(0, s2.length(), s2);
}

static void printHelpMsg()
{
	printf("SnippetLoadCollection usage:\n"
		"SnippetLoadCollection "
		"[--pvdhost=<ip address> ] "
		"[--pvdport=<ip port> ]"
		"[--pvdtimeout=<time ms> ] "
		"[--generateExampleFiles]"		
		"<filename>...\n\n"		   
		"Load binary or xml serialized collections and instatiate the objects in a PhysX scene.\n");

	printf("--pvdhost=<ip address> \n");
	printf("  Defines ip address of PVD, default is 127.0.0.1 \n");

	printf("--pvdport=<ip port> \n");
	printf("  Defines ip port of PVD, default is 5425\n");

	printf("--pvdtimeout=<time ms> \n");
	printf("  Defines timeout of PVD, default is 10\n");

	printf("--generateExampleFiles\n");
	printf("  Generates a set of example files\n");	

	printf("<filename>...\n");
	printf("  Input files (xml or binary), if a collection contains shared objects, it needs to be provided with the first file. \n\n");

}

static bool parseCommandLine(CmdLineParameters& result, int argc, const char *const*argv)
{
	if( argc <= 1 )
	{
		printHelpMsg();	
		return false;
	}

	for(int i = 1; i < argc; ++i)
	{
		if(Snippets::isCommonHeadlessOption(argv[i]))
			continue;
		if(argv[i][0] != '-' || argv[i][1] != '-')
		{
			if (result.nbFiles < MAX_INPUT_FILES)
			{
				result.inputFiles[result.nbFiles++] = argv[i];
			}
			else
				printf( "[WARNING] more input files are specified than supported (maximum %d). Ignoring the file  %s\n", MAX_INPUT_FILES, argv[i] );				
		}
		else if(match(argv[i], "--pvdhost="))
		{
			const char* hostStr = argv[i] + strlen("--pvdhost=");
			if(hostStr)
				result.pvdhost = hostStr;
		}
		else if(match(argv[i], "--pvdport="))
		{			
			const char* portStr = argv[i] + strlen("--pvdport=");		
			if (portStr)
				result.pvdport = PxU32(atoi(portStr));	
		}
		else if(match(argv[i], "--pvdtimeout="))
		{			
			const char* timeoutStr = argv[i] + strlen("--pvdtimeout=");		
			if (timeoutStr)
				result.pvdtimeout = PxU32(atoi(timeoutStr));	
		}
		else if(match(argv[i], "--generateExampleFiles"))
		{
			result.generateExampleFiles = true;
		}
		else
		{
			printf( "[ERROR] Unknown command line parameter \"%s\"\n", argv[i] );
			printHelpMsg();
			return false;
		}
	}
	
	if(result.nbFiles == 0 &&  !result.generateExampleFiles)
	{
		printf( "[ERROR] parameter missing.\n" );
		printHelpMsg();
		return false;				   
	}

	return true;
}

static bool checkFile(bool& isBinary, const char* filename)
{
	PxDefaultFileInputData fileStream(filename);
	if (fileStream.getLength() == 0)
	{
		printf( "[ERROR] input file %s can't be opened!\n", filename);
		return false;
	}

	char testString[17];
	fileStream.read(testString, 16);
	testString[16] = 0;

	if (strcmp("SEBD", testString) == 0)
	{
		isBinary = true;
		return true;
	}

	if (strcmp("<PhysXCollection", testString) == 0)
	{
		isBinary = false;
		return true;
	}

	printf( "[ERROR] input file %s seems neither an xml nor a binary serialized collection file!\n", filename);
	return false;
}

static PxCollection* deserializeCollection(PxInputData& inputData, bool isBinary, PxCollection* sharedCollection, PxSerializationRegistry& sr)
{
	PxCollection* collection = NULL;
	if(isBinary)
	{
		PxU32 length = (PxU32)inputData.getLength();
		PxU8* memBlock = static_cast<PxU8*>(malloc(length+PX_SERIAL_FILE_ALIGN-1));
		gMemBlocks[gNbMemBlocks++] = memBlock;
		void* alignedBlock = reinterpret_cast<void*>((size_t(memBlock)+PX_SERIAL_FILE_ALIGN-1)&~(PX_SERIAL_FILE_ALIGN-1));
		inputData.read(alignedBlock, length);
		collection = PxSerialization::createCollectionFromBinary(alignedBlock, sr, sharedCollection);
	}
	else
	{
		PxTolerancesScale scale;
		PxCookingParams params(scale);

		collection = PxSerialization::createCollectionFromXml(inputData, params, sr, sharedCollection);		
	}

	return collection;
}

static bool initPhysics(bool headless)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	if(!headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && transport)
			gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}
	
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
	PxInitExtensions(*gPhysics, gPvd);
	
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0, -9.81f, 0);	
	if(headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gDispatcher = PxDefaultCpuDispatcherCreate(
		headless ? gHeadlessOptions.dispatcherThreads : 1);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	gSolverReadbackMatched =
		!headless || gScene->getSolverType() == sceneDesc.solverType;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gSerializationRegistry = PxSerialization::createSerializationRegistry(*gPhysics);
	return gSerializationRegistry != NULL;
}

void cleanupPhysics()
{	
	PX_RELEASE(gSerializationRegistry);
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PxCloseExtensions();
		
	PX_RELEASE(gPhysics);	// releases of all objects	

	for(PxU32 i=0; i<gNbMemBlocks; i++)
		free(gMemBlocks[i]); // now that the objects have been released, it's safe to release the space they occupy
	gNbMemBlocks = 0;

	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}

	PX_RELEASE(gFoundation);
	gCleanupComplete = 1;

	printf("SnippetLoadCollection done.\n");
}

static bool serializeCollection(PxCollection& collection, PxCollection* externalRefs, const char* filename, bool toBinary)
{
	PxDefaultFileOutputStream outputStream(filename);
	if (!outputStream.isValid())
	{
		printf( "[ERROR] Could not open file %s!\n", filename);
		return false;
	}

	bool bret;
	if (toBinary)
	{
		bret = PxSerialization::serializeCollectionToBinary(outputStream, collection, *gSerializationRegistry, externalRefs);
	}
	else
	{
		bret = PxSerialization::serializeCollectionToXml(outputStream, collection, *gSerializationRegistry, NULL, externalRefs);
	}

	if(bret)
		printf( "Generated: \"%s\"\n", filename);
	else
		printf( "[ERROR] Failure when generating %s!\n", filename);
	return bret;
}

static bool generateExampleFiles()
{
	PxCollection* collection = PxCreateCollection();
	PxCollection* collectionA = PxCreateCollection();
	PxCollection* collectionB = PxCreateCollection();
	PX_ASSERT( (collection != NULL) && (collectionA != NULL) && (collectionB != NULL) );
	
	PxMaterial *material = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	PX_ASSERT( material );
	PxShape* planeShape = gPhysics->createShape(PxPlaneGeometry(), *material);
	PxShape* boxShape = gPhysics->createShape(PxBoxGeometry(2.f, 2.f, 2.f), *material);
	PxRigidStatic* rigidStatic = PxCreateStatic(*gPhysics, PxTransform(PxVec3(0.f, 0.f, 0.f), PxQuat(PxHalfPi, PxVec3(0.f, 0.f, 1.f))), *planeShape);
	PxRigidDynamic* rigidDynamic = PxCreateDynamic(*gPhysics, PxTransform(PxVec3(0.f, 2.f, 0.f)), *boxShape, 1.f); 
	
	collection->add(*material);
	collection->add(*planeShape);
	collection->add(*boxShape);
	collection->add(*rigidStatic);
	collection->add(*rigidDynamic);
	PxSerialization::complete(*collection, *gSerializationRegistry);
	PX_ASSERT(PxSerialization::isSerializable(*collection, *gSerializationRegistry));

	collectionA->add(*material);
	collectionA->add(*planeShape);
	collectionA->add(*boxShape);
	PxSerialization::complete(*collectionA, *gSerializationRegistry);
	PxSerialization::createSerialObjectIds(*collectionA, PxSerialObjectId(1));
	PX_ASSERT(PxSerialization::isSerializable(*collectionA, *gSerializationRegistry));

	collectionB->add(*rigidStatic);
	collectionB->add(*rigidDynamic);
	PxSerialization::complete(*collectionB, *gSerializationRegistry, collectionA);
	PX_ASSERT(PxSerialization::isSerializable(*collectionB, *gSerializationRegistry, collectionA));
	
	bool generated = true;
	generated &= serializeCollection(*collection, NULL, "collection.xml", false);
	generated &= serializeCollection(*collectionA, NULL, "collectionA.xml", false);
	generated &= serializeCollection(*collectionB, collectionA, "collectionB.xml", false);
	generated &= serializeCollection(*collection, NULL, "collection.bin", true);
	generated &= serializeCollection(*collectionA, NULL, "collectionA.bin", true);
	generated &= serializeCollection(*collectionB, collectionA, "collectionB.bin", true);

	collection->release();
	collectionA->release();
	collectionB->release();
	return generated;
}

static bool isFiniteActor(const PxActor& actor)
{
	const PxRigidActor* rigidActor = actor.is<PxRigidActor>();
	if(!rigidActor || !rigidActor->getGlobalPose().isFinite())
		return false;
	const PxRigidDynamic* dynamic = actor.is<PxRigidDynamic>();
	return !dynamic ||
		(dynamic->getLinearVelocity().isFinite() &&
		 dynamic->getAngularVelocity().isFinite());
}

static int runHeadlessLoad()
{
	PxU32 filesLoaded = 0;
	PxU32 binaryFiles = 0;
	PxU32 xmlFiles = 0;
	PxU32 collectionsAdded = 0;
	PxU32 staticActors = 0;
	PxU32 dynamicActors = 0;
	PxU32 actorShapeRefs = 0;
	PxU32 externalShapeRefs = 0;
	PxU32 nonFinite = 0;
	PxU32 fetchFailures = 0;
	PxU32 completedFrames = 0;
	bool loadSuccess = true;
	bool materialIdentity = true;
	PxReal initialDynamicY = 0.0f;
	PxReal finalDynamicY = 0.0f;
	PxReal minDynamicY = FLT_MAX;
	PxReal maxDynamicSpeed = 0.0f;

	PxCollection* firstCollection = NULL;
	std::vector<PxShape*> firstCollectionShapes;
	for(PxU32 i=0; i<gParameters.nbFiles; ++i)
	{
		const char* filename = gParameters.inputFiles[i];
		bool isBinary = false;
		if(!checkFile(isBinary, filename))
		{
			loadSuccess = false;
			break;
		}
		PxDefaultFileInputData inputStream(filename);
		PxCollection* collection = deserializeCollection(
			inputStream, isBinary, firstCollection, *gSerializationRegistry);
		if(!collection)
		{
			printf("[ERROR] deserialization failure! filename: %s\n", filename);
			loadSuccess = false;
			break;
		}
		printf("Loaded: \"%s\"\n", filename);
		filesLoaded++;
		binaryFiles += isBinary ? 1u : 0u;
		xmlFiles += isBinary ? 0u : 1u;

		gScene->addCollection(*collection);
		collectionsAdded++;
		if(i == 0)
		{
			firstCollection = collection;
			const PxU32 objectCount = collection->getNbObjects();
			std::vector<PxBase*> objects(objectCount);
			if(objectCount)
				collection->getObjects(objects.data(), objectCount);
			for(PxU32 objectIndex=0; objectIndex<objectCount; ++objectIndex)
			{
				PxShape* shape = objects[objectIndex]->is<PxShape>();
				if(shape)
					firstCollectionShapes.push_back(shape);
			}
		}
		else
			collection->release();
	}

	if(firstCollection)
		firstCollection->release();

	const PxActorTypeFlags actorTypes =
		PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC;
	const PxU32 actorCount = gScene->getNbActors(actorTypes);
	std::vector<PxActor*> actors(actorCount);
	if(actorCount)
		gScene->getActors(actorTypes, actors.data(), actorCount);
	PxMaterial* commonMaterial = NULL;
	PxRigidDynamic* loadedDynamic = NULL;
	for(PxU32 i=0; i<actorCount; ++i)
	{
		PxRigidStatic* rigidStatic = actors[i]->is<PxRigidStatic>();
		PxRigidDynamic* rigidDynamic = actors[i]->is<PxRigidDynamic>();
		staticActors += rigidStatic ? 1u : 0u;
		dynamicActors += rigidDynamic ? 1u : 0u;
		if(rigidDynamic)
			loadedDynamic = rigidDynamic;

		PxRigidActor* rigidActor = actors[i]->is<PxRigidActor>();
		if(!rigidActor)
			continue;
		const PxU32 shapeCount = rigidActor->getNbShapes();
		std::vector<PxShape*> shapes(shapeCount);
		if(shapeCount)
			rigidActor->getShapes(shapes.data(), shapeCount);
		for(PxU32 shapeIndex=0; shapeIndex<shapeCount; ++shapeIndex)
		{
			PxShape* shape = shapes[shapeIndex];
			actorShapeRefs++;
			for(PxU32 firstIndex=0;
				firstIndex<firstCollectionShapes.size(); ++firstIndex)
			{
				if(shape == firstCollectionShapes[firstIndex])
				{
					externalShapeRefs++;
					break;
				}
			}
			PxMaterial* material = NULL;
			if(shape->getNbMaterials() != 1 ||
			   shape->getMaterials(&material, 1) != 1 || !material)
				materialIdentity = false;
			else if(!commonMaterial)
				commonMaterial = material;
			else if(commonMaterial != material)
				materialIdentity = false;
		}
	}

	if(loadedDynamic)
	{
		PxTransform pose = loadedDynamic->getGlobalPose();
		pose.p.y = 8.0f;
		loadedDynamic->setGlobalPose(pose);
		loadedDynamic->setLinearVelocity(PxVec3(0.0f));
		loadedDynamic->setAngularVelocity(PxVec3(0.0f));
		loadedDynamic->wakeUp();
		initialDynamicY = pose.p.y;
	}

	for(PxU32 frame=0; frame<gHeadlessOptions.frames; ++frame)
	{
		gScene->simulate(gHeadlessOptions.dt);
		if(!gScene->fetchResults(true))
		{
			fetchFailures++;
			break;
		}
		completedFrames++;
		for(PxU32 i=0; i<actorCount; ++i)
		{
			if(!isFiniteActor(*actors[i]))
				nonFinite++;
		}
		if(loadedDynamic)
		{
			const PxReal y = loadedDynamic->getGlobalPose().p.y;
			const PxReal speed = loadedDynamic->getLinearVelocity().magnitude();
			minDynamicY = PxMin(minDynamicY, y);
			maxDynamicSpeed = PxMax(maxDynamicSpeed, speed);
		}
		if(nonFinite)
			break;
	}
	if(loadedDynamic)
		finalDynamicY = loadedDynamic->getGlobalPose().p.y;

	const PxU32 constraintCount = gScene->getNbConstraints();
	const bool splitReferences =
		gParameters.nbFiles == 1 ||
		(actorShapeRefs == 2 && externalShapeRefs == actorShapeRefs);
	const bool passed =
		loadSuccess && filesLoaded == gParameters.nbFiles &&
		collectionsAdded == gParameters.nbFiles &&
		staticActors == 1 && dynamicActors == 1 &&
		actorShapeRefs == 2 && materialIdentity && splitReferences &&
		constraintCount == 0 && loadedDynamic &&
		initialDynamicY == 8.0f && finalDynamicY < 7.5f &&
		minDynamicY > 1.0f &&
		completedFrames == gHeadlessOptions.frames &&
		nonFinite == 0 && fetchFailures == 0 &&
		gSolverReadbackMatched && gErrorCallback.getFatalCount() == 0;
	const PxU32 fatalErrors = gErrorCallback.getFatalCount();
	cleanupPhysics();
	printf("[AVBD_GATE] schema=1 snippet=SnippetLoadCollection solver=%s "
		"case=%s execution=%s frames=%u filesRequested=%u filesLoaded=%u "
		"binaryFiles=%u xmlFiles=%u collectionsAdded=%u staticActors=%u "
		"dynamicActors=%u constraints=%u actorShapeRefs=%u "
		"externalShapeRefs=%u materialIdentity=%u initialDynamicY=%.9g "
		"finalDynamicY=%.9g minDynamicY=%.9g maxDynamicSpeed=%.9g "
		"completedFrames=%u nonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gParameters.nbFiles, filesLoaded,
		binaryFiles, xmlFiles, collectionsAdded, staticActors,
		dynamicActors, constraintCount, actorShapeRefs, externalShapeRefs,
		materialIdentity ? 1u : 0u, initialDynamicY, finalDynamicY,
		minDynamicY, maxDynamicSpeed, completedFrames, nonFinite,
		fetchFailures, fatalErrors, gCleanupComplete,
		passed ? "PASS" : "FAIL",
		passed ? "none" : "collection_identity_or_dynamics");
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char *const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.caseName = "load-collection";
	std::vector<const char*> headlessArgs;
	headlessArgs.push_back(argv[0]);
	for(int i=1; i<argc; ++i)
	{
		if(Snippets::isCommonHeadlessOption(argv[i]))
			headlessArgs.push_back(argv[i]);
	}
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(
		static_cast<int>(headlessArgs.size()), headlessArgs.data(),
		defaults, gHeadlessOptions, error))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] snippet=SnippetLoadCollection "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!parseCommandLine(gParameters, argc, argv))
		return Snippets::eHEADLESS_CONFIG_ERROR;

	gErrorCallback.reset();
	gCleanupComplete = 0;
	if(!initPhysics(gHeadlessOptions.headless))
	{
		cleanupPhysics();
		return Snippets::eHEADLESS_GATE_FAILED;
	}

	bool generated = true;
	if(gParameters.generateExampleFiles)
		generated = generateExampleFiles();

	if(gHeadlessOptions.headless && gParameters.generateExampleFiles &&
	   gParameters.nbFiles == 0)
	{
		const PxU32 fatalErrors = gErrorCallback.getFatalCount();
		const bool passed = generated && gSolverReadbackMatched &&
			fatalErrors == 0;
		cleanupPhysics();
		printf("[AVBD_GATE] schema=1 snippet=SnippetLoadCollection solver=%s "
			"case=generate execution=%s filesGenerated=%u fatalErrors=%u "
			"cleanupComplete=%u pvd=0 status=%s reason=%s "
			"validation=GATED\n",
			Snippets::getSolverTypeName(gHeadlessOptions.solverType),
			Snippets::getExecutionName(gHeadlessOptions.execution),
			generated ? 6u : 0u, fatalErrors, gCleanupComplete,
			passed ? "PASS" : "FAIL",
			passed ? "none" : "generation_failed");
		return passed ? Snippets::eHEADLESS_PASS :
			Snippets::eHEADLESS_GATE_FAILED;
	}

	if(gHeadlessOptions.headless)
		return runHeadlessLoad();
	
	// collection that may have shared objects
	PxCollection* firstCollection = NULL;

	for(PxU32 i=0; i<gParameters.nbFiles; i++)
	{
		const char*	filename = gParameters.inputFiles[i];

		bool isBinary;
		bool validFile = checkFile(isBinary, filename);

		if (!validFile)
			break;

		PxDefaultFileInputData inputStream(filename);
		PxCollection* collection = deserializeCollection(inputStream, isBinary, firstCollection, *gSerializationRegistry);
		if (!collection)
		{
			printf( "[ERROR] deserialization failure! filename: %s\n", filename);
			break;
		}
		else
		{
			printf( "Loaded: \"%s\"\n", filename);
		}

		gScene->addCollection(*collection);

		if (i == 0)
		{
			firstCollection = collection;
		}
		else
		{
			collection->release();
		}
	}

	if (firstCollection)
		firstCollection->release();

	for (unsigned i = 0; i < 20; i++)
	{
	gScene->simulate(1.0f/60.0f);
	gScene->fetchResults(true);
	}

	cleanupPhysics();	

	return 0;
}
