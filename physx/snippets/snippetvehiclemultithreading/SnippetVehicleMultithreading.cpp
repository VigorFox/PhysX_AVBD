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
// This snippet illustrates simple use of the physx vehicle sdk and demonstrates
// how to simulate a vehicle with direct drive using parameters, states and 
// components maintained by the PhysX Vehicle SDK. Particlar attention is paid
// to the simulation of a PhysX vehicle in a multi-threaded environment.

// Vehicles are made of parameters, states and components.

// Parameters describe the configuration of a vehicle.  Examples are vehicle mass, wheel radius 
// and suspension stiffness.

// States describe the instantaneous dynamic state of a vehicle.  Examples are engine revs, wheel 
// yaw angle and tire slip angles.

// Components forward integrate the dynamic state of the vehicle, given the previous vehicle state 
// and the vehicle's parameterisation.
// Components update dynamic state by invoking reusable functions in a particular sequence. 
// An example component is a rigid body component that updates the linear and angular velocity of 
// the vehicle's rigid body given the instantaneous forces and torques of the suspension and tire 
// states.

// The pipeline of vehicle computation is a sequence of components that run in order.  For example, 
// one component might compute the plane under the wheel by performing a scene query against the 
// world geometry. The next component in the sequence might compute the suspension compression required 
// to place the wheel on the surface of the hit plane. Following this, another component might compute 
// the suspension force that arises from that compression.  The rigid body component, as discussed earlier, 
// can then forward integrate the rigid body's linear velocity using the suspension force.

// Custom combinations of parameter, state and component allow different behaviours to be simulated with 
// different simulation fidelities.  For example, a suspension component that implements a linear force 
// response with respect to its compression state could be replaced with one that imlements a non-linear
// response.  The replacement component would consume the same suspension compression state data and 
// would output the same suspension force data structure.  In this example, the change has been localised 
// to the  component that converts suspension compression to force and to the parameterisation that governs 
// that conversion.
// Another combination example could be the replacement of the tire component from a low fidelity model to 
// a high fidelty model such as Pacejka. The low and high fidelity components consume the same state data 
// (tire slip, load, friction) and  output the same state data  for the tire forces. Again, the 
// change has been localised to the component that converts slip angle to tire force and the 
// parameterisation that governs the conversion.

//The PhysX Vehicle SDK presents a maintained set of parameters, states and components.  The maintained 
//set of parameters, states and components may be combined on their own or combined with custom parameters, 
//states and components.

//This snippet breaks the vehicle into into three distinct models:
//1) a base vehicle model that describes the mechanical configuration of suspensions, tires, wheels and an 
//   associated rigid body.
//2) a direct drive drivetrain model that forwards input controls to wheel torques and angles.
//3) a physx integration model that provides a representation of the vehicle in an associated physx scene.

// It is a good idea to record and playback with pvd (PhysX Visual Debugger).

//This snippet 
// ****************************************************************************

#include <ctype.h>

#include "PxPhysicsAPI.h"

#include "../snippetvehiclecommon/directdrivetrain/DirectDrivetrain.h"
#include "../snippetvehiclecommon/serialization/BaseSerialization.h"
#include "../snippetvehiclecommon/serialization/DirectDrivetrainSerialization.h"
#include "../snippetvehiclecommon/SnippetVehicleHelpers.h"

#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"

#include "common/PxProfileZone.h"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace physx;
using namespace snippetvehicle;

//PhysX management class instances.

PxDefaultAllocator		gAllocator;
Snippets::TrackingErrorCallback gErrorCallback;
PxFoundation*			gFoundation = NULL;
PxPhysics*				gPhysics	= NULL;
PxDefaultCpuDispatcher*	gDispatcher = NULL;
PxScene*				gScene		= NULL;
PxMaterial*				gMaterial	= NULL;
PxPvd*                  gPvd        = NULL;
PxTaskManager*			gTaskManager = NULL;
Snippets::HeadlessOptions gHeadlessOptions;
bool					gVehicleExtensionInitialized = false;
PxU32					gInitializedVehicleCount = 0;

//The path to the vehicle json files to be loaded.
const char* gVehicleDataPath = NULL;

//The vehicles with direct drivetrain
#define NUM_VEHICLES 1024
DirectDriveVehicle gVehicles[NUM_VEHICLES];
PxVehiclePhysXActorBeginComponent* gPhysXBeginComponents[NUM_VEHICLES];
PxVehiclePhysXActorEndComponent* gPhysXEndComponents[NUM_VEHICLES];

#define NUM_WORKER_THREADS 4
#define UPDATE_BATCH_SIZE 1
#define NB_SUBSTEPS 1

std::atomic<PxU64> gTaskRuns[NUM_WORKER_THREADS];
std::atomic<PxU64> gTaskVehicleUpdates[NUM_WORKER_THREADS];
std::atomic<PxU64> gOffMainTaskRuns(0);
std::atomic<PxU64> gWaitTaskRuns(0);
std::atomic<PxU64> gWaitTaskReleases(0);
PxU32 gMainThreadId = 0;

//Vehicle simulation needs a simulation context
//to store global parameters of the simulation such as 
//gravitational acceleration.
PxVehiclePhysXSimulationContext gVehicleSimulationContext;

//Gravitational acceleration
const PxVec3 gGravity(0.0f, -9.81f, 0.0f);

//The mapping between PxMaterial and friction.
PxVehiclePhysXMaterialFriction gPhysXMaterialFrictions[16];
PxU32 gNbPhysXMaterialFrictions = 0;
PxReal gPhysXDefaultMaterialFriction = 1.0f;

//Give the vehicles a name so they can be identified in PVD.
const char gVehicleName[] = "directDrive";

//A ground plane to drive on.
PxRigidStatic*	gGroundPlane = NULL;

//Track the number of simulation steps.
PxU32 gNbSimulateSteps = 0;

//Commands are issued to the vehicle in a pre-choreographed sequence.
struct Command
{
	PxF32 brake;
	PxF32 throttle;
	PxF32 steer;
	PxF32 duration;
};
Command gCommands[] =
{
	{0.0f, 0.5f, 0.0f, 4.26f},		//throttle for 256 update steps at 60Hz
};
const PxU32 gNbCommands = sizeof(gCommands) / sizeof(Command);
PxReal gCommandTime = 0.0f;			//Time spent on current command
PxU32 gCommandProgress = 0;			//The id of the current command.

struct VehicleMultithreadingMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 vehicleCount;
	PxU32 wheelCount;
	PxU32 constraintCount;
	PxU32 beginComponentCalls;
	PxU32 endComponentCalls;
	PxU32 allTaskPartitionsCompleteFrames;
	PxU32 continuationCompleteFrames;
	PxU32 roadHitVehicleFrames;
	PxU64 roadHitSamples;
	PxU32 driveVehicleFrames;
	PxU32 tireForceVehicleFrames;
	PxU64 nonZeroLongForceSamples;
	PxU64 nonZeroLatForceSamples;
	PxU32 tireForceNonFinite;
	PxU32 activeConstraintVehicleFrames;
	PxU64 activeConstraintRows;
	PxU32 cleanupComplete;
	PxReal maxTireLongForce;
	PxReal maxTireLatForce;
	PxReal minFinalForwardDisplacement;
	PxReal maxFinalForwardDisplacement;
	PxReal maxLateralDrift;
	PxReal minHeight;
	PxReal maxHeight;
	PxReal maxLinearSpeed;
	PxReal maxAngularSpeed;
	bool solverReadbackMatched;

	VehicleMultithreadingMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0), nonFinite(0),
	  vehicleCount(0), wheelCount(0), constraintCount(0),
	  beginComponentCalls(0), endComponentCalls(0),
	  allTaskPartitionsCompleteFrames(0), continuationCompleteFrames(0),
	  roadHitVehicleFrames(0), roadHitSamples(0), driveVehicleFrames(0),
	  tireForceVehicleFrames(0), nonZeroLongForceSamples(0),
	  nonZeroLatForceSamples(0), tireForceNonFinite(0),
	  activeConstraintVehicleFrames(0), activeConstraintRows(0),
	  cleanupComplete(0), maxTireLongForce(0.0f),
	  maxTireLatForce(0.0f), minFinalForwardDisplacement(FLT_MAX),
	  maxFinalForwardDisplacement(-FLT_MAX), maxLateralDrift(0.0f),
	  minHeight(FLT_MAX), maxHeight(-FLT_MAX), maxLinearSpeed(0.0f),
	  maxAngularSpeed(0.0f), solverReadbackMatched(false)
	{
	}
};

VehicleMultithreadingMetrics gMetrics;


//Profile the different phases of a simulate step.

struct UpdatePhases
{
	enum Enum
	{
		eVEHICLE_PHYSX_BEGIN_COMPONENTS,
		eVEHICLE_UPDATE_COMPONENTS,
		eVEHICLE_PHYSX_END_COMPONENTS,
		ePHYSX_SCENE_SIMULATE,
		eMAX_NUM_UPDATE_STAGES
	};
};
static const char* gUpdatePhaseNames[UpdatePhases::eMAX_NUM_UPDATE_STAGES] =
{
	"vehiclePhysXBeginComponents",
	"vehicleUpdateComponents",
	"vehiclePhysXEndComponents",
	"physXSceneSimulate"
};

struct ProfileZones
{
	PxU64 times[UpdatePhases::eMAX_NUM_UPDATE_STAGES];

	ProfileZones()
	{
		for (int i = 0; i < UpdatePhases::eMAX_NUM_UPDATE_STAGES; ++i)
			times[i] = 0;
	}

	void print()
	{
		for (int i = 0; i < UpdatePhases::eMAX_NUM_UPDATE_STAGES; ++i)
		{
			float ms = SnippetUtils::getElapsedTimeInMilliseconds(times[i]);
			printf("%s: %f ms\n", gUpdatePhaseNames[i], PxF64(ms));
		}
	}

	void zoneStart(UpdatePhases::Enum zoneId)
	{
		PxU64 time = SnippetUtils::getCurrentTimeCounterValue();
		times[zoneId] -= time;
	}

	void zoneEnd(UpdatePhases::Enum zoneId)
	{
		PxU64 time = SnippetUtils::getCurrentTimeCounterValue();
		times[zoneId] += time;
	}
};
ProfileZones gProfileZones;

class ScopedProfileZone
{
private:
	ScopedProfileZone(const ScopedProfileZone&);
	ScopedProfileZone& operator=(const ScopedProfileZone&);

public:
	ScopedProfileZone(ProfileZones& zones, UpdatePhases::Enum zoneId)
		: mZones(zones)
		, mZoneId(zoneId)
	{
		zones.zoneStart(zoneId);
	}

	~ScopedProfileZone()
	{
		mZones.zoneEnd(mZoneId);
	}


private:
	ProfileZones& mZones;
	UpdatePhases::Enum mZoneId;
};

#define SNIPPET_PROFILE_ZONE(zoneId) ScopedProfileZone PX_CONCAT(_scoped, __LINE__)(gProfileZones, zoneId)

//TaskVehicleUpdates allows vehicle updates to be performed concurrently across
//multiple threads.
class TaskVehicleUpdates : public PxLightCpuTask
{
public:

	TaskVehicleUpdates()
		: PxLightCpuTask(),
		mTimestep(0),
		mGravity(PxVec3(0, 0, 0)),
		mThreadId(0xffffffff),
		mCommandProgress(0)
	{
	}

	void setThreadId(const PxU32 threadId)
	{
		mThreadId = threadId;
	}

	void setTimestep(const PxF32 timestep)
	{
		mTimestep = timestep;
	}

	void setGravity(const PxVec3& gravity)
	{
		mGravity = gravity;
	}

	void setCommandProgress(const PxU32 commandProgress)
	{
		mCommandProgress = commandProgress;
	}

	virtual void run()
	{
		if(mThreadId >= NUM_WORKER_THREADS)
			return;
		gTaskRuns[mThreadId].fetch_add(1, std::memory_order_relaxed);
		if(SnippetUtils::getThreadId() != gMainThreadId)
			gOffMainTaskRuns.fetch_add(1, std::memory_order_relaxed);
		PxU64 updatedVehicles = 0;
		PxU32 vehicleId = mThreadId * UPDATE_BATCH_SIZE;
		while (vehicleId < NUM_VEHICLES)
		{
			const PxU32 numToUpdate = PxMin(NUM_VEHICLES - vehicleId, static_cast<PxU32>(UPDATE_BATCH_SIZE));
			for (PxU32 i = 0; i < numToUpdate; i++)
			{
				gVehicles[vehicleId + i].mCommandState.brakes[0] = gCommands[mCommandProgress].brake;
				gVehicles[vehicleId + i].mCommandState.nbBrakes = 1;
				gVehicles[vehicleId + i].mCommandState.throttle = gCommands[mCommandProgress].throttle;
				gVehicles[vehicleId + i].mCommandState.steer = gCommands[mCommandProgress].steer;
				gVehicles[vehicleId + i].mTransmissionCommandState.gear = PxVehicleDirectDriveTransmissionCommandState::eFORWARD;
				gVehicles[vehicleId + i].step(mTimestep, gVehicleSimulationContext);
				updatedVehicles++;
			}
			vehicleId += NUM_WORKER_THREADS * UPDATE_BATCH_SIZE;
		}
		gTaskVehicleUpdates[mThreadId].fetch_add(
			updatedVehicles, std::memory_order_relaxed);
	}

	virtual const char* getName() const { return "TaskVehicleUpdates"; }

private:

	PxF32 mTimestep;
	PxVec3 mGravity;

	PxU32 mThreadId;

	PxU32 mCommandProgress;
};

//TaskWait runs after all concurrent updates have completed.
class TaskWait : public PxLightCpuTask
{
public:

	TaskWait(SnippetUtils::Sync* syncHandle)
		: PxLightCpuTask(),
		mSyncHandle(syncHandle)
	{
	}

	virtual void run()
	{
		gWaitTaskRuns.fetch_add(1, std::memory_order_relaxed);
	}

	PX_INLINE void release()
	{
		PxLightCpuTask::release();
		gWaitTaskReleases.fetch_add(1, std::memory_order_relaxed);
		SnippetUtils::syncSet(mSyncHandle);
	}

	virtual const char* getName() const { return "TaskWait"; }

private:

	SnippetUtils::Sync* mSyncHandle;
};

bool initPhysX(bool headless)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	if(!headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			PxPvdTransport* transport =
				PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(transport)
				gPvd->connect(*transport, PxPvdInstrumentationFlag::ePROFILE);
		}
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
		
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = gGravity;
	if(headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	
	const PxU32 workerCount =
		headless ? gHeadlessOptions.dispatcherThreads : NUM_WORKER_THREADS;
	gDispatcher = PxDefaultCpuDispatcherCreate(workerCount);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= VehicleFilterShader;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	gMetrics.solverReadbackMatched =
		gScene->getSolverType() == sceneDesc.solverType;
	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(!headless && pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, false);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, false);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, false);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);	
	if(!gMaterial)
		return false;

	/////////////////////////////////////////////
	//Create a task manager that will be used to 
	//update the vehicles concurrently across 
	//multiple threads.
	/////////////////////////////////////////////

	gTaskManager = PxTaskManager::createTaskManager(gFoundation->getErrorCallback(), gDispatcher);
	if(!gTaskManager)
		return false;

	gVehicleExtensionInitialized = PxInitVehicleExtension(*gFoundation);
	return gVehicleExtensionInitialized;
}

void cleanupPhysX()
{
	if(gVehicleExtensionInitialized)
	{
		PxCloseVehicleExtension();
		gVehicleExtensionInitialized = false;
	}

	PX_RELEASE(gTaskManager);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
}

bool initGroundPlane()
{
	gGroundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	if(!gGroundPlane)
		return false;
	for (PxU32 i = 0; i < gGroundPlane->getNbShapes(); i++)
	{
		PxShape* shape = NULL;
		gGroundPlane->getShapes(&shape, 1, i);
		shape->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, true);
		shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
		shape->setFlag(PxShapeFlag::eTRIGGER_SHAPE, false);
	}
	gScene->addActor(*gGroundPlane);
	return true;
}

void cleanupGroundPlane()
{
	PX_RELEASE(gGroundPlane);
}

void initMaterialFrictionTable()
{
	//Each physx material can be mapped to a tire friction value on a per tire basis.
	//If a material is encountered that is not mapped to a friction value, the friction value used is the specified default value.
	//In this snippet there is only a single material so there can only be a single mapping between material and friction.
	//In this snippet the same mapping is used by all tires.
	gPhysXMaterialFrictions[0].friction = 1.0f;
	gPhysXMaterialFrictions[0].material = gMaterial;
	gPhysXDefaultMaterialFriction = 1.0f;
	gNbPhysXMaterialFrictions = 1;
}

bool initVehicles()
{
	//Load the params from json 

	BaseVehicleParams baseParams;
	if(!readBaseParamsFromJsonFile(gVehicleDataPath, "Base.json", baseParams))
		return false;

	PhysXIntegrationParams physxParams;
	setPhysXIntegrationParams(baseParams.axleDescription,
		gPhysXMaterialFrictions, gNbPhysXMaterialFrictions, gPhysXDefaultMaterialFriction,
		physxParams);

	DirectDrivetrainParams directDrivetrainParams;
	if(!readDirectDrivetrainParamsFromJsonFile(
		gVehicleDataPath, "DirectDrive.json", baseParams.axleDescription,
		directDrivetrainParams))
	{
		return false;
	}

	//Create the params, states and component sequences for direct drive vehicles.
	//Take care not to add PxVehiclePhysXActorBeginComponent or PxVehiclePhysXActorEndComponent
	//to the sequences because are executed in a separate step.
	for (PxU32 i = 0; i < NUM_VEHICLES; i++)
	{
		//Set the vehicle params.
		//Every vehicle is identical.
		gVehicles[i].mBaseParams = baseParams;
		gVehicles[i].mPhysXParams = physxParams;
		gVehicles[i].mDirectDriveParams = directDrivetrainParams;

		//Set the states to default and create the component sequence.
		//Take care not to add PxVehiclePhysXActorBeginComponent and PxVehiclePhysXActorEndComponent 
		//to the sequence because these are handled separately to take advantage of multi-threading.
		const bool addPhysXBeginAndEndComponentsToSequence = false;
		if (!gVehicles[i].initialize(*gPhysics, PxCookingParams(PxTolerancesScale()), *gMaterial, 
			addPhysXBeginAndEndComponentsToSequence))
		{
			return false;
		}
		gInitializedVehicleCount++;

		//Force a known substep count per simulation step so that we have a perfect understanding of 
		//the amount of computational effort involved in running the snippet.
		gVehicles[i].mComponentSequence.setSubsteps(gVehicles[i].mComponentSequenceSubstepGroupHandle, NB_SUBSTEPS);

		//Apply a start pose to the physx actor and add it to the physx scene.
		PxTransform pose(PxVec3(5.0f*(PxI32(i) - NUM_VEHICLES/2), 0.0f, 0.0f), PxQuat(PxIdentity));
		gVehicles[i].setUpActor(*gScene, pose, gVehicleName);
	}

	//PhysX reads/writes require read/write locks that serialize executions.
	//Perform all physx reads/writes serially in a separate step to avoid serializing code that can take 
	//advantage of multithreading.
	for (PxU32 i = 0; i < NUM_VEHICLES; i++)
	{
		gPhysXBeginComponents[i] = (static_cast<PxVehiclePhysXActorBeginComponent*>(gVehicles + i));
		gPhysXEndComponents[i] = (static_cast<PxVehiclePhysXActorEndComponent*>(gVehicles + i));
	}
	
	//Set up the simulation context.
	//The snippet is set up with
	//a) z as the longitudinal axis
	//b) x as the lateral axis
	//c) y as the vertical axis.
	//d) metres  as the lengthscale.
	gVehicleSimulationContext.setToDefault();
	gVehicleSimulationContext.frame.lngAxis = PxVehicleAxes::ePosZ;
	gVehicleSimulationContext.frame.latAxis = PxVehicleAxes::ePosX;
	gVehicleSimulationContext.frame.vrtAxis = PxVehicleAxes::ePosY;
	gVehicleSimulationContext.scale.scale = 1.0f;
	gVehicleSimulationContext.gravity = gGravity;
	gVehicleSimulationContext.physxScene = gScene;
	gVehicleSimulationContext.physxActorUpdateMode = PxVehiclePhysXActorUpdateMode::eAPPLY_ACCELERATION;
	return true;
}

void cleanupVehicles()
{
	for (PxU32 i = 0; i < gInitializedVehicleCount; i++)
	{
		gVehicles[i].destroy();
		gPhysXBeginComponents[i] = NULL;
		gPhysXEndComponents[i] = NULL;
	}
	gInitializedVehicleCount = 0;
}

static bool initPhysicsInternal(bool headless)
{
	gMetrics = VehicleMultithreadingMetrics();
	gCommandTime = 0.0f;
	gCommandProgress = 0;
	gNbSimulateSteps = 0;
	gInitializedVehicleCount = 0;
	gMainThreadId = SnippetUtils::getThreadId();
	gOffMainTaskRuns.store(0, std::memory_order_relaxed);
	gWaitTaskRuns.store(0, std::memory_order_relaxed);
	gWaitTaskReleases.store(0, std::memory_order_relaxed);
	for(PxU32 i = 0; i < NUM_WORKER_THREADS; ++i)
	{
		gTaskRuns[i].store(0, std::memory_order_relaxed);
		gTaskVehicleUpdates[i].store(0, std::memory_order_relaxed);
	}
	if(!initPhysX(headless))
		return false;
	if(!initGroundPlane())
		return false;
	initMaterialFrictionTable();
	if (!initVehicles())
		return false;
	gMetrics.initialized = 1;
	gMetrics.vehicleCount = gInitializedVehicleCount;
	for(PxU32 vehicleId = 0; vehicleId < gInitializedVehicleCount; ++vehicleId)
	{
		gMetrics.wheelCount +=
			gVehicles[vehicleId].mBaseParams.axleDescription.nbWheels;
		for(PxU32 constraintId = 0;
			constraintId <
				PxVehiclePhysXConstraintLimits::eNB_CONSTRAINTS_PER_VEHICLE;
			++constraintId)
		{
			if(gVehicles[vehicleId].mPhysXState.physxConstraints
				.constraints[constraintId])
			{
				gMetrics.constraintCount++;
			}
		}
	}
	return true;
}

bool initPhysics()
{
	return initPhysicsInternal(false);
}

void cleanupPhysics()
{
	cleanupVehicles();
	cleanupGroundPlane();
	cleanupPhysX();
	gMetrics.cleanupComplete =
		!gInitializedVehicleCount && !gGroundPlane && !gMaterial &&
		!gTaskManager && !gScene && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation && !gVehicleExtensionInitialized ? 1u : 0u;
	printf("SnippetVehicleMultithreading done.\n");
}

static bool concurrentVehicleUpdates(
	const PxReal timestep, const bool collectMetrics)
{
	SnippetUtils::Sync* vehicleUpdatesComplete = SnippetUtils::syncCreate();
	if(!vehicleUpdatesComplete)
		return false;
	SnippetUtils::syncReset(vehicleUpdatesComplete);
	PxU64 taskRunsBefore[NUM_WORKER_THREADS];
	PxU64 taskUpdatesBefore[NUM_WORKER_THREADS];
	for(PxU32 i = 0; i < NUM_WORKER_THREADS; ++i)
	{
		taskRunsBefore[i] =
			gTaskRuns[i].load(std::memory_order_relaxed);
		taskUpdatesBefore[i] =
			gTaskVehicleUpdates[i].load(std::memory_order_relaxed);
	}
	const PxU64 waitRunsBefore =
		gWaitTaskRuns.load(std::memory_order_relaxed);
	const PxU64 waitReleasesBefore =
		gWaitTaskReleases.load(std::memory_order_relaxed);

	//Create tasks that will update the vehicles concurrently then wait until all vehicles 
	//have completed their update.
	TaskWait taskWait(vehicleUpdatesComplete);
	TaskVehicleUpdates taskVehicleUpdates[NUM_WORKER_THREADS];
	for (PxU32 i = 0; i < NUM_WORKER_THREADS; i++)
	{
		taskVehicleUpdates[i].setThreadId(i);
		taskVehicleUpdates[i].setTimestep(timestep);
		taskVehicleUpdates[i].setGravity(gScene->getGravity());
		taskVehicleUpdates[i].setCommandProgress(gCommandProgress);
	}

	//Start the task manager.
	gTaskManager->resetDependencies();
	gTaskManager->startSimulation();

	//Perform a vehicle simulation step and profile each phase of the simulation.
	{
		//PhysX reads/writes require read/write locks that serialize executions.
		//Perform all physx reads/writes serially in a separate step to avoid serializing code that can take 
		//advantage of multithreading.
		{
			SNIPPET_PROFILE_ZONE(UpdatePhases::eVEHICLE_PHYSX_BEGIN_COMPONENTS);
			for (PxU32 i = 0; i < NUM_VEHICLES; i++)
			{
				gPhysXBeginComponents[i]->update(timestep, gVehicleSimulationContext);
			}
			if(collectMetrics)
				gMetrics.beginComponentCalls += NUM_VEHICLES;
		}

		//Multi-threaded update of direct drive vehicles.
		{
			SNIPPET_PROFILE_ZONE(UpdatePhases::eVEHICLE_UPDATE_COMPONENTS);

			//Update the vehicles concurrently then wait until all vehicles 
			//have completed their update.
			taskWait.setContinuation(*gTaskManager, NULL);
			for (PxU32 i = 0; i < NUM_WORKER_THREADS; i++)
			{
				taskVehicleUpdates[i].setContinuation(&taskWait);
			}
			taskWait.removeReference();
			for (PxU32 i = 0; i < NUM_WORKER_THREADS; i++)
			{
				taskVehicleUpdates[i].removeReference();
			}

			//Wait for the signal that the work has been completed.
			SnippetUtils::syncWait(vehicleUpdatesComplete);

			//Release the sync handle
			SnippetUtils::syncRelease(vehicleUpdatesComplete);

			if(collectMetrics)
			{
				bool allPartitionsComplete = true;
				const PxU64 expectedUpdatesPerPartition =
					(NUM_VEHICLES + NUM_WORKER_THREADS - 1) /
					NUM_WORKER_THREADS;
				for(PxU32 i = 0; i < NUM_WORKER_THREADS; ++i)
				{
					const PxU64 taskRunDelta =
						gTaskRuns[i].load(std::memory_order_relaxed) -
						taskRunsBefore[i];
					const PxU64 updateDelta =
						gTaskVehicleUpdates[i].load(
							std::memory_order_relaxed) -
						taskUpdatesBefore[i];
					if(taskRunDelta != 1 ||
						updateDelta != expectedUpdatesPerPartition)
					{
						allPartitionsComplete = false;
					}
				}
				if(allPartitionsComplete)
					gMetrics.allTaskPartitionsCompleteFrames++;
				const PxU64 waitRunDelta =
					gWaitTaskRuns.load(std::memory_order_relaxed) -
					waitRunsBefore;
				const PxU64 waitReleaseDelta =
					gWaitTaskReleases.load(std::memory_order_relaxed) -
					waitReleasesBefore;
				if(waitRunDelta == 1 && waitReleaseDelta == 1)
					gMetrics.continuationCompleteFrames++;
			}
		}

		//PhysX reads/writes require read/write locks that serialize executions.
		//Perform all physx reads/writes serially in a separate step to avoid serializing code that can take 
		//advantage of multithreading.
		{
			SNIPPET_PROFILE_ZONE(UpdatePhases::eVEHICLE_PHYSX_END_COMPONENTS);
			for (PxU32 i = 0; i < NUM_VEHICLES; i++)
			{
				gPhysXEndComponents[i]->update(timestep, gVehicleSimulationContext);
			}
			if(collectMetrics)
				gMetrics.endComponentCalls += NUM_VEHICLES;
		}
	}
	return true;
}

static void collectVehicleMetrics()
{
	PxReal minForwardDisplacement = FLT_MAX;
	PxReal maxForwardDisplacement = -FLT_MAX;
	for(PxU32 vehicleId = 0; vehicleId < NUM_VEHICLES; ++vehicleId)
	{
		DirectDriveVehicle& vehicle = gVehicles[vehicleId];
		const PxVehicleAxleDescription& axle =
			vehicle.mBaseParams.axleDescription;
		PxU32 roadHits = 0;
		bool driveApplied = false;
		bool tireForceApplied = false;
		PxU32 activeRows = 0;
		for(PxU32 i = 0; i < axle.nbWheels; ++i)
		{
			const PxU32 wheelId = axle.wheelIdsInAxleOrder[i];
			const PxVehicleRoadGeometryState& road =
				vehicle.mBaseState.roadGeomStates[wheelId];
			const PxVehicleWheelActuationState& actuation =
				vehicle.mBaseState.actuationStates[wheelId];
			const PxVehiclePhysXConstraintState& constraint =
				vehicle.mPhysXState.physxConstraints
					.constraintStates[wheelId];
			if(road.hitState)
				roadHits++;
			driveApplied = driveApplied || actuation.isDriveApplied;
			if(constraint.suspActiveStatus)
				activeRows++;
			for(PxU32 direction = 0;
				direction <
					PxVehicleTireDirectionModes::eMAX_NB_PLANAR_DIRECTIONS;
				++direction)
			{
				if(constraint.tireActiveStatus[direction])
					activeRows++;
			}
			const PxVehicleTireForce& force =
				vehicle.mBaseState.tireForces[wheelId];
			const PxReal longMagnitude =
				force.forces[PxVehicleTireDirectionModes::eLONGITUDINAL]
					.magnitude();
			const PxReal latMagnitude =
				force.forces[PxVehicleTireDirectionModes::eLATERAL]
					.magnitude();
			if(!PxIsFinite(longMagnitude) || !PxIsFinite(latMagnitude) ||
				!PxIsFinite(force.wheelTorque) ||
				!PxIsFinite(force.aligningMoment))
			{
				gMetrics.tireForceNonFinite++;
			}
			else
			{
				gMetrics.maxTireLongForce =
					PxMax(gMetrics.maxTireLongForce, longMagnitude);
				gMetrics.maxTireLatForce =
					PxMax(gMetrics.maxTireLatForce, latMagnitude);
				if(longMagnitude > 1e-4f)
				{
					gMetrics.nonZeroLongForceSamples++;
					tireForceApplied = true;
				}
				if(latMagnitude > 1e-4f)
				{
					gMetrics.nonZeroLatForceSamples++;
					tireForceApplied = true;
				}
			}
		}
		gMetrics.roadHitSamples += roadHits;
		if(roadHits == axle.nbWheels)
			gMetrics.roadHitVehicleFrames++;
		if(driveApplied)
			gMetrics.driveVehicleFrames++;
		if(tireForceApplied)
			gMetrics.tireForceVehicleFrames++;
		if(activeRows)
			gMetrics.activeConstraintVehicleFrames++;
		gMetrics.activeConstraintRows += activeRows;

		PxRigidBody* rigidBody =
			vehicle.mPhysXState.physxActor.rigidBody;
		const PxTransform pose = rigidBody->getGlobalPose();
		const PxVec3 linearVelocity = rigidBody->getLinearVelocity();
		const PxVec3 angularVelocity = rigidBody->getAngularVelocity();
		if(!pose.isFinite() || !linearVelocity.isFinite() ||
			!angularVelocity.isFinite())
		{
			gMetrics.nonFinite++;
			continue;
		}
		const PxReal initialX =
			5.0f * (PxI32(vehicleId) - NUM_VEHICLES / 2);
		minForwardDisplacement =
			PxMin(minForwardDisplacement, pose.p.z);
		maxForwardDisplacement =
			PxMax(maxForwardDisplacement, pose.p.z);
		gMetrics.maxLateralDrift = PxMax(
			gMetrics.maxLateralDrift, PxAbs(pose.p.x - initialX));
		gMetrics.minHeight = PxMin(gMetrics.minHeight, pose.p.y);
		gMetrics.maxHeight = PxMax(gMetrics.maxHeight, pose.p.y);
		gMetrics.maxLinearSpeed = PxMax(
			gMetrics.maxLinearSpeed, linearVelocity.magnitude());
		gMetrics.maxAngularSpeed = PxMax(
			gMetrics.maxAngularSpeed, angularVelocity.magnitude());
	}
	if(minForwardDisplacement != FLT_MAX)
	{
		gMetrics.minFinalForwardDisplacement = minForwardDisplacement;
		gMetrics.maxFinalForwardDisplacement = maxForwardDisplacement;
	}
}

static bool stepPhysicsInternal(
	const PxReal timestep, const bool collectMetrics)
{
	if(gNbCommands == gCommandProgress)
		return true;

	//Multithreaded update of all vehicles.
	if(!concurrentVehicleUpdates(timestep, collectMetrics))
		return false;

	//Forward integrate the phsyx scene by a single timestep.
	SNIPPET_PROFILE_ZONE(UpdatePhases::ePHYSX_SCENE_SIMULATE);
	gScene->simulate(timestep);
	if(!gScene->fetchResults(true))
	{
		if(collectMetrics)
			gMetrics.fetchFailures++;
		return false;
	}
	if(collectMetrics)
	{
		gMetrics.completedFrames++;
		collectVehicleMetrics();
	}

	//Increment the time spent on the current command.
	//Move to the next command in the list if enough time has lapsed.
	gCommandTime += timestep;
	if(gCommandTime > gCommands[gCommandProgress].duration)
	{
		gCommandProgress++;
		gCommandTime = 0.0f;
	}

	gNbSimulateSteps++;
	return true;
}

void stepPhysics()
{
	stepPhysicsInternal(0.0166667f, false);
}

static bool parseHeadlessOptions(
	int argc, const char*const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.frames = 300;
	defaults.dispatcherThreads = NUM_WORKER_THREADS;
	defaults.dt = 0.0166667f;
	defaults.caseName = "task-graph";
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	if(!gHeadlessOptions.headless)
		return true;
	if(gHeadlessOptions.caseName != "task-graph")
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.dispatcherThreads != NUM_WORKER_THREADS)
	{
		error = "--dispatcher-threads must preserve the official 4-worker task graph";
		return false;
	}

	PxU32 vehiclePathCount = 0;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		if(Snippets::hasOptionPrefix(arg, "--vehicleDataPath="))
		{
			vehiclePathCount++;
			gVehicleDataPath =
				arg + std::strlen("--vehicleDataPath=");
			if(!gVehicleDataPath[0])
			{
				error = "empty --vehicleDataPath";
				return false;
			}
			continue;
		}
		error = "unknown headless option";
		return false;
	}
	if(vehiclePathCount != 1)
	{
		error = vehiclePathCount == 0 ?
			"missing --vehicleDataPath" :
			"duplicate --vehicleDataPath";
		return false;
	}
	return true;
}

static bool taskCountersPassed()
{
	const PxU64 completedFrames = gMetrics.completedFrames;
	const PxU64 expectedUpdatesPerPartition =
		(NUM_VEHICLES + NUM_WORKER_THREADS - 1) / NUM_WORKER_THREADS;
	for(PxU32 i = 0; i < NUM_WORKER_THREADS; ++i)
	{
		if(gTaskRuns[i].load(std::memory_order_relaxed) != completedFrames ||
			gTaskVehicleUpdates[i].load(std::memory_order_relaxed) !=
				completedFrames * expectedUpdatesPerPartition)
		{
			return false;
		}
	}
	return gWaitTaskRuns.load(std::memory_order_relaxed) ==
			completedFrames &&
		gWaitTaskReleases.load(std::memory_order_relaxed) ==
			completedFrames;
}

static bool headlessPassed()
{
	const PxU64 vehicleFrames =
		PxU64(gMetrics.completedFrames) * NUM_VEHICLES;
	return gMetrics.initialized == 1 &&
		gMetrics.solverReadbackMatched &&
		gCommandProgress == gNbCommands &&
		gMetrics.completedFrames <= gHeadlessOptions.frames &&
		gMetrics.completedFrames >= 250 &&
		gMetrics.fetchFailures == 0 && gMetrics.nonFinite == 0 &&
		gMetrics.vehicleCount == NUM_VEHICLES &&
		gMetrics.wheelCount == NUM_VEHICLES * 4 &&
		gMetrics.constraintCount == NUM_VEHICLES &&
		gMetrics.beginComponentCalls ==
			gMetrics.completedFrames * NUM_VEHICLES &&
		gMetrics.endComponentCalls ==
			gMetrics.completedFrames * NUM_VEHICLES &&
		gMetrics.allTaskPartitionsCompleteFrames ==
			gMetrics.completedFrames &&
		gMetrics.continuationCompleteFrames ==
			gMetrics.completedFrames &&
		taskCountersPassed() &&
		gOffMainTaskRuns.load(std::memory_order_relaxed) > 0 &&
		gMetrics.roadHitVehicleFrames > vehicleFrames / 2 &&
		gMetrics.driveVehicleFrames > vehicleFrames / 2 &&
		gMetrics.tireForceVehicleFrames > vehicleFrames / 2 &&
		gMetrics.nonZeroLongForceSamples > 0 &&
		gMetrics.tireForceNonFinite == 0 &&
		gMetrics.maxTireLongForce > 1.0f &&
		gMetrics.minFinalForwardDisplacement > 1.0f &&
		gMetrics.maxLinearSpeed > 1.0f &&
		gMetrics.maxLateralDrift < 5.0f &&
		gMetrics.minHeight > -1.0f && gMetrics.maxHeight < 5.0f &&
		gErrorCallback.getFatalCount() == 0 &&
		gMetrics.cleanupComplete == 1;
}

static const char* failureReason()
{
	const PxU64 vehicleFrames =
		PxU64(gMetrics.completedFrames) * NUM_VEHICLES;
	if(gMetrics.initialized != 1)
		return "initialization";
	if(!gMetrics.solverReadbackMatched)
		return "solver_readback";
	if(gCommandProgress != gNbCommands ||
		gMetrics.completedFrames < 250 ||
		gMetrics.completedFrames > gHeadlessOptions.frames)
	{
		return "command_cycle_incomplete";
	}
	if(gMetrics.fetchFailures)
		return "fetch_failure";
	if(gMetrics.nonFinite)
		return "non_finite";
	if(gMetrics.vehicleCount != NUM_VEHICLES ||
		gMetrics.wheelCount != NUM_VEHICLES * 4 ||
		gMetrics.constraintCount != NUM_VEHICLES)
	{
		return "vehicle_topology";
	}
	if(gMetrics.beginComponentCalls !=
			gMetrics.completedFrames * NUM_VEHICLES ||
		gMetrics.endComponentCalls !=
			gMetrics.completedFrames * NUM_VEHICLES)
	{
		return "physx_component_phases";
	}
	if(gMetrics.allTaskPartitionsCompleteFrames !=
			gMetrics.completedFrames ||
		!taskCountersPassed())
	{
		return "worker_task_completion";
	}
	if(gMetrics.continuationCompleteFrames !=
			gMetrics.completedFrames)
	{
		return "continuation_completion";
	}
	if(!gOffMainTaskRuns.load(std::memory_order_relaxed))
		return "worker_dispatch";
	if(gMetrics.roadHitVehicleFrames <= vehicleFrames / 2)
		return "road_query_coverage";
	if(gMetrics.driveVehicleFrames <= vehicleFrames / 2)
		return "drive_actuation";
	if(gMetrics.tireForceNonFinite)
		return "tire_force_non_finite";
	if(gMetrics.tireForceVehicleFrames <= vehicleFrames / 2 ||
		!gMetrics.nonZeroLongForceSamples ||
		gMetrics.maxTireLongForce <= 1.0f)
	{
		return "tire_force_activity";
	}
	if(gMetrics.minFinalForwardDisplacement <= 1.0f ||
		gMetrics.maxLinearSpeed <= 1.0f)
	{
		return "rigid_response";
	}
	if(gMetrics.maxLateralDrift >= 5.0f ||
		gMetrics.minHeight <= -1.0f || gMetrics.maxHeight >= 5.0f)
	{
		return "motion_bounds";
	}
	if(gErrorCallback.getFatalCount())
		return "physx_error";
	if(gMetrics.cleanupComplete != 1)
		return "cleanup";
	return "none";
}

static void printHeadlessResult()
{
	const bool pass = headlessPassed();
	PxU64 totalTaskRuns = 0;
	PxU64 totalTaskVehicleUpdates = 0;
	PxU64 taskRuns[NUM_WORKER_THREADS];
	PxU64 taskVehicleUpdates[NUM_WORKER_THREADS];
	for(PxU32 i = 0; i < NUM_WORKER_THREADS; ++i)
	{
		taskRuns[i] = gTaskRuns[i].load(std::memory_order_relaxed);
		taskVehicleUpdates[i] =
			gTaskVehicleUpdates[i].load(std::memory_order_relaxed);
		totalTaskRuns += taskRuns[i];
		totalTaskVehicleUpdates += taskVehicleUpdates[i];
	}
	const PxReal minimumForward =
		gMetrics.completedFrames ?
			gMetrics.minFinalForwardDisplacement : 0.0f;
	const PxReal maximumForward =
		gMetrics.completedFrames ?
			gMetrics.maxFinalForwardDisplacement : 0.0f;
	const PxReal minimumHeight =
		gMetrics.completedFrames ? gMetrics.minHeight : 0.0f;
	const PxReal maximumHeight =
		gMetrics.completedFrames ? gMetrics.maxHeight : 0.0f;
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetVehicleMultithreading "
		"solver=%s case=task-graph execution=%s frames=%u "
		"completedFrames=%u commands=%u vehicles=%u wheels=%u "
		"constraints=%u workerPartitions=%u updateBatchSize=%u substeps=%u "
		"beginComponentCalls=%u endComponentCalls=%u "
		"allTaskPartitionsCompleteFrames=%u "
		"continuationCompleteFrames=%u "
		"taskRuns=%llu taskRuns0=%llu taskRuns1=%llu "
		"taskRuns2=%llu taskRuns3=%llu "
		"taskVehicleUpdates=%llu taskVehicleUpdates0=%llu "
		"taskVehicleUpdates1=%llu taskVehicleUpdates2=%llu "
		"taskVehicleUpdates3=%llu offMainTaskRuns=%llu "
		"waitTaskRuns=%llu waitTaskReleases=%llu "
		"roadHitVehicleFrames=%u roadHitSamples=%llu "
		"driveVehicleFrames=%u tireForceVehicleFrames=%u "
		"nonZeroLongForceSamples=%llu nonZeroLatForceSamples=%llu "
		"tireForceNonFinite=%u maxTireLongForce=%.9g "
		"maxTireLatForce=%.9g activeConstraintVehicleFrames=%u "
		"activeConstraintRows=%llu minFinalForwardDisplacement=%.9g "
		"maxFinalForwardDisplacement=%.9g maxLateralDrift=%.9g "
		"minHeight=%.9g maxHeight=%.9g maxLinearSpeed=%.9g "
		"maxAngularSpeed=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0 "
		"status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		unsigned(gHeadlessOptions.frames), unsigned(gMetrics.completedFrames),
		unsigned(gCommandProgress), unsigned(gMetrics.vehicleCount),
		unsigned(gMetrics.wheelCount), unsigned(gMetrics.constraintCount),
		unsigned(NUM_WORKER_THREADS), unsigned(UPDATE_BATCH_SIZE),
		unsigned(NB_SUBSTEPS), unsigned(gMetrics.beginComponentCalls),
		unsigned(gMetrics.endComponentCalls),
		unsigned(gMetrics.allTaskPartitionsCompleteFrames),
		unsigned(gMetrics.continuationCompleteFrames),
		static_cast<unsigned long long>(totalTaskRuns),
		static_cast<unsigned long long>(taskRuns[0]),
		static_cast<unsigned long long>(taskRuns[1]),
		static_cast<unsigned long long>(taskRuns[2]),
		static_cast<unsigned long long>(taskRuns[3]),
		static_cast<unsigned long long>(totalTaskVehicleUpdates),
		static_cast<unsigned long long>(taskVehicleUpdates[0]),
		static_cast<unsigned long long>(taskVehicleUpdates[1]),
		static_cast<unsigned long long>(taskVehicleUpdates[2]),
		static_cast<unsigned long long>(taskVehicleUpdates[3]),
		static_cast<unsigned long long>(
			gOffMainTaskRuns.load(std::memory_order_relaxed)),
		static_cast<unsigned long long>(
			gWaitTaskRuns.load(std::memory_order_relaxed)),
		static_cast<unsigned long long>(
			gWaitTaskReleases.load(std::memory_order_relaxed)),
		unsigned(gMetrics.roadHitVehicleFrames),
		static_cast<unsigned long long>(gMetrics.roadHitSamples),
		unsigned(gMetrics.driveVehicleFrames),
		unsigned(gMetrics.tireForceVehicleFrames),
		static_cast<unsigned long long>(
			gMetrics.nonZeroLongForceSamples),
		static_cast<unsigned long long>(
			gMetrics.nonZeroLatForceSamples),
		unsigned(gMetrics.tireForceNonFinite),
		double(gMetrics.maxTireLongForce),
		double(gMetrics.maxTireLatForce),
		unsigned(gMetrics.activeConstraintVehicleFrames),
		static_cast<unsigned long long>(gMetrics.activeConstraintRows),
		double(minimumForward), double(maximumForward),
		double(gMetrics.maxLateralDrift), double(minimumHeight),
		double(maximumHeight), double(gMetrics.maxLinearSpeed),
		double(gMetrics.maxAngularSpeed), unsigned(gMetrics.nonFinite),
		unsigned(gMetrics.fetchFailures),
		unsigned(gErrorCallback.getFatalCount()),
		unsigned(gMetrics.cleanupComplete), pass ? "PASS" : "FAIL",
		pass ? "none" : failureReason());
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(
			stderr, "[AVBD_GATE_CONFIG_ERROR] %s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!gHeadlessOptions.headless &&
		!parseVehicleDataPath(
			argc, argv, "SnippetVehicleMultithreading", gVehicleDataPath))
		return 1;

	//Check that we can read from the json file before continuing.
	BaseVehicleParams baseParams;
	if (!readBaseParamsFromJsonFile(gVehicleDataPath, "Base.json", baseParams))
		return 1;

	//Check that we can read from the json file before continuing.
	DirectDrivetrainParams directDrivetrainParams;
	if (!readDirectDrivetrainParamsFromJsonFile(gVehicleDataPath, "DirectDrive.json",
		baseParams.axleDescription, directDrivetrainParams))
		return 1;

	if(gHeadlessOptions.headless)
	{
		if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		{
			std::fprintf(
				stderr,
				"[AVBD_GATE_CONFIG_ERROR] failed to set execution mode\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		Snippets::printHeadlessConfig(
			"SnippetVehicleMultithreading", gHeadlessOptions);
		gErrorCallback.reset();
		bool runOk = initPhysicsInternal(true);
		while(runOk && gCommandProgress != gNbCommands &&
			gMetrics.completedFrames < gHeadlessOptions.frames)
		{
			runOk = stepPhysicsInternal(gHeadlessOptions.dt, true);
		}
		cleanupPhysics();
		printHeadlessResult();
		return runOk && headlessPassed() ?
			Snippets::eHEADLESS_PASS :
			Snippets::eHEADLESS_GATE_FAILED;
	}

	printf("Initialising ... \n");
	if(initPhysics())
	{
		printf("Simulating %d vehicles with %d threads \n", NUM_VEHICLES, NUM_WORKER_THREADS);
		while(gCommandProgress != gNbCommands)
		{
			stepPhysics();
		}
		printf("Completed %d simulate steps with %d substeps per simulate step \n", gNbSimulateSteps, NB_SUBSTEPS);
		gProfileZones.print();
		cleanupPhysics();
	}
	return 0;
}
