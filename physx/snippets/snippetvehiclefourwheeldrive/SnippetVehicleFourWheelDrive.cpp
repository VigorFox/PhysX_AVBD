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
// how to simulate a vehicle with a fully featured drivetrain comprising engine,
// clutch, differential and gears.  The snippet uses only parameters, states and 
// components maintained by the PhysX Vehicle SDK.

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
//2) a drivetrain model that forwards input controls to wheel torques via a drivetrain model
//   that includes engine, clutch, differential and gears.
//3) a physx integration model that provides a representation of the vehicle in an associated physx scene.

// It is a good idea to record and playback with pvd (PhysX Visual Debugger).
// ****************************************************************************

#include <ctype.h>

#include "PxPhysicsAPI.h"
#include "../snippetvehiclecommon/enginedrivetrain/EngineDrivetrain.h"
#include "../snippetvehiclecommon/serialization/BaseSerialization.h"
#include "../snippetvehiclecommon/serialization/EngineDrivetrainSerialization.h"
#include "../snippetvehiclecommon/SnippetVehicleHelpers.h"

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"

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
Snippets::HeadlessOptions gHeadlessOptions;
bool					gVehicleExtensionInitialized = false;
bool					gVehicleInitialized = false;

//The path to the vehicle json files to be loaded.
const char* gVehicleDataPath = NULL;

//The vehicle with engine drivetrain
EngineDriveVehicle gVehicle;

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

//Give the vehicle a name so it can be identified in PVD.
const char gVehicleName[] = "engineDrive";

//Commands are issued to the vehicle in a pre-choreographed sequence.
struct Command
{
	PxF32 brake;
	PxF32 throttle;
	PxF32 steer;
	PxU32 gear;
	PxF32 duration;
};
const PxU32 gTargetGearCommand = PxVehicleEngineDriveTransmissionCommandState::eAUTOMATIC_GEAR;
Command gCommands[] =
{
	{0.5f, 0.0f, 0.0f, gTargetGearCommand, 2.0f},	//brake on and come to rest for 2 seconds
	{0.0f, 0.65f, 0.0f, gTargetGearCommand, 5.0f},	//throttle for 5 seconds
	{0.5f, 0.0f, 0.0f, gTargetGearCommand, 5.0f},	//brake for 5 seconds
	{0.0f, 0.75f, 0.0f, gTargetGearCommand, 5.0f},	//throttle for 5 seconds
	{0.0f, 0.25f, 0.5f, gTargetGearCommand, 5.0f}	//light throttle and steer for 5 seconds.
};
const PxU32 gNbCommands = sizeof(gCommands) / sizeof(Command);
PxReal gCommandTime = 0.0f;			//Time spent on current command
PxU32 gCommandProgress = 0;			//The id of the current command.

//A ground plane to drive on.
PxRigidStatic*	gGroundPlane = NULL;

struct VehicleMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 drivetrainNonFinite;
	PxU32 wheelCount;
	PxU32 constraintCount;
	PxU32 roadHitFrames;
	PxU32 roadHitSamples;
	PxU32 driveFrames;
	PxU32 brakeFrames;
	PxU32 activeConstraintFrames;
	PxU32 activeConstraintRows;
	PxU32 fourWheelDifferentialFrames;
	PxU32 clutchEngagedFrames;
	PxU32 gearChanges;
	PxU32 minimumGear;
	PxU32 maximumGear;
	PxU32 cleanupComplete;
	PxReal initialBrakeMaxSpeed;
	PxReal maxThrottleSpeed;
	PxReal brakeStartSpeed;
	PxReal minBrakeSpeed;
	PxReal maxForwardDisplacement;
	PxReal steerLateralDisplacement;
	PxReal steerHeadingChange;
	PxReal minHeight;
	PxReal maxHeight;
	PxReal maxEngineSpeed;
	PxReal maxClutchSlip;
	PxReal maxDifferentialRatioError;
	PxReal firstUpshiftThreshold;
	PxTransform initialPose;
	PxTransform steerStartPose;
	PxReal steerStartHeading;
	PxU32 previousGear;
	bool brakePhaseSeen;
	bool steerPhaseSeen;
	bool previousGearValid;
	bool solverReadbackMatched;

	VehicleMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0), nonFinite(0),
	  drivetrainNonFinite(0), wheelCount(0), constraintCount(0),
	  roadHitFrames(0), roadHitSamples(0), driveFrames(0), brakeFrames(0),
	  activeConstraintFrames(0), activeConstraintRows(0),
	  fourWheelDifferentialFrames(0), clutchEngagedFrames(0),
	  gearChanges(0), minimumGear(0xffffffffu), maximumGear(0),
	  cleanupComplete(0), initialBrakeMaxSpeed(0.0f),
	  maxThrottleSpeed(0.0f), brakeStartSpeed(0.0f),
	  minBrakeSpeed(FLT_MAX), maxForwardDisplacement(0.0f),
	  steerLateralDisplacement(0.0f), steerHeadingChange(0.0f),
	  minHeight(FLT_MAX), maxHeight(-FLT_MAX), maxEngineSpeed(0.0f),
	  maxClutchSlip(0.0f), maxDifferentialRatioError(0.0f),
	  firstUpshiftThreshold(0.0f),
	  initialPose(PxIdentity), steerStartPose(PxIdentity),
	  steerStartHeading(0.0f), previousGear(0),
	  brakePhaseSeen(false), steerPhaseSeen(false),
	  previousGearValid(false), solverReadbackMatched(false)
	{
	}
};

VehicleMetrics gMetrics;

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
				gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
		}
	}
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
		
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = gGravity;
	if(headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	
	const PxU32 numWorkers =
		headless ? gHeadlessOptions.dispatcherThreads : 1u;
	gDispatcher = PxDefaultCpuDispatcherCreate(numWorkers);
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
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
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
	//Load the params from json or set directly.
	readBaseParamsFromJsonFile(gVehicleDataPath, "Base.json", gVehicle.mBaseParams);
	setPhysXIntegrationParams(gVehicle.mBaseParams.axleDescription,
		gPhysXMaterialFrictions, gNbPhysXMaterialFrictions, gPhysXDefaultMaterialFriction,
		gVehicle.mPhysXParams);
	readEngineDrivetrainParamsFromJsonFile(gVehicleDataPath, "EngineDrive.json", 
		gVehicle.mEngineDriveParams);

	//Set the states to default.
	if (!gVehicle.initialize(*gPhysics, PxCookingParams(PxTolerancesScale()), *gMaterial, EngineDriveVehicle::eDIFFTYPE_FOURWHEELDRIVE))
	{
		return false;
	}
	gVehicleInitialized = true;

	//Apply a start pose to the physx actor and add it to the physx scene.
	PxTransform pose(PxVec3(0.000000000f, -0.0500000119f, -1.59399998f), PxQuat(PxIdentity));
	gVehicle.setUpActor(*gScene, pose, gVehicleName);

	//Set the vehicle in 1st gear.
	gVehicle.mEngineDriveState.gearboxState.currentGear = gVehicle.mEngineDriveParams.gearBoxParams.neutralGear + 1;
	gVehicle.mEngineDriveState.gearboxState.targetGear = gVehicle.mEngineDriveParams.gearBoxParams.neutralGear + 1;

	//Set the vehicle to use the automatic gearbox.
	gVehicle.mTransmissionCommandState.targetGear = PxVehicleEngineDriveTransmissionCommandState::eAUTOMATIC_GEAR;

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
	if(gVehicleInitialized)
	{
		gVehicle.destroy();
		gVehicleInitialized = false;
	}
}

static bool initPhysicsInternal(bool headless)
{
	gMetrics = VehicleMetrics();
	gCommandTime = 0.0f;
	gCommandProgress = 0;
	if(!initPhysX(headless))
		return false;
	if(!initGroundPlane())
		return false;
	initMaterialFrictionTable();
	if (!initVehicles())
		return false;
	gMetrics.initialized = 1;
	gMetrics.wheelCount = gVehicle.mBaseParams.axleDescription.nbWheels;
	for(PxU32 i = 0;
		i < PxVehiclePhysXConstraintLimits::eNB_CONSTRAINTS_PER_VEHICLE; ++i)
	{
		if(gVehicle.mPhysXState.physxConstraints.constraints[i])
			gMetrics.constraintCount++;
	}
	gMetrics.initialPose =
		gVehicle.mPhysXState.physxActor.rigidBody->getGlobalPose();
	const PxU32 firstForwardGear =
		gVehicle.mEngineDriveParams.gearBoxParams.neutralGear + 1;
	gMetrics.firstUpshiftThreshold =
		gVehicle.mEngineDriveParams.autoboxParams.upRatios[firstForwardGear] *
		gVehicle.mEngineDriveParams.engineParams.maxOmega;
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
		!gVehicleInitialized && !gGroundPlane && !gMaterial && !gScene &&
		!gDispatcher && !gPhysics && !gPvd && !gFoundation ? 1u : 0u;
	printf("SnippetVehicleFourWheelDrive done.\n");
}

static PxReal getHeading(const PxTransform& pose)
{
	const PxVec3 forward = pose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f));
	return PxReal(std::atan2(double(forward.x), double(forward.z)));
}

static PxReal getHeadingDifference(PxReal heading, PxReal reference)
{
	const PxReal difference = heading - reference;
	return PxAbs(PxReal(std::atan2(
		std::sin(double(difference)), std::cos(double(difference)))));
}

static void collectVehicleMetrics(PxU32 commandIndex)
{
	const PxVehicleAxleDescription& axle =
		gVehicle.mBaseParams.axleDescription;
	PxU32 roadHits = 0;
	bool driveApplied = false;
	bool brakeApplied = false;
	PxU32 activeRows = 0;
	for(PxU32 i = 0; i < axle.nbWheels; ++i)
	{
		const PxU32 wheelId = axle.wheelIdsInAxleOrder[i];
		const PxVehicleRoadGeometryState& road =
			gVehicle.mBaseState.roadGeomStates[wheelId];
		const PxVehicleWheelActuationState& actuation =
			gVehicle.mBaseState.actuationStates[wheelId];
		const PxVehiclePhysXConstraintState& constraint =
			gVehicle.mPhysXState.physxConstraints.constraintStates[wheelId];
		if(road.hitState)
			roadHits++;
		driveApplied = driveApplied || actuation.isDriveApplied;
		brakeApplied = brakeApplied || actuation.isBrakeApplied;
		if(constraint.suspActiveStatus)
			activeRows++;
		for(PxU32 direction = 0;
			direction < PxVehicleTireDirectionModes::eMAX_NB_PLANAR_DIRECTIONS;
			++direction)
		{
			if(constraint.tireActiveStatus[direction])
				activeRows++;
		}
	}
	gMetrics.roadHitSamples += roadHits;
	if(roadHits == axle.nbWheels)
		gMetrics.roadHitFrames++;
	if(driveApplied)
		gMetrics.driveFrames++;
	if(brakeApplied)
		gMetrics.brakeFrames++;
	if(activeRows)
		gMetrics.activeConstraintFrames++;
	gMetrics.activeConstraintRows += activeRows;

	const EngineDrivetrainState& drivetrain = gVehicle.mEngineDriveState;
	const PxReal engineSpeed = drivetrain.engineState.rotationSpeed;
	const PxReal clutchSlip = drivetrain.clutchState.clutchSlip;
	const PxReal clutchResponse =
		drivetrain.clutchCommandResponseState.commandResponse;
	PxReal ratioMagnitudeSum = 0.0f;
	for(PxU32 i = 0; i < axle.nbWheels; ++i)
	{
		const PxU32 wheelId = axle.wheelIdsInAxleOrder[i];
		ratioMagnitudeSum += PxAbs(
			drivetrain.differentialState.torqueRatiosAllWheels[wheelId]);
	}
	const PxReal ratioError = PxAbs(ratioMagnitudeSum - 1.0f);
	if(!PxIsFinite(engineSpeed) || !PxIsFinite(clutchSlip) ||
		!PxIsFinite(clutchResponse) || !PxIsFinite(ratioError))
	{
		gMetrics.drivetrainNonFinite++;
	}
	else
	{
		gMetrics.maxEngineSpeed =
			PxMax(gMetrics.maxEngineSpeed, PxAbs(engineSpeed));
		gMetrics.maxClutchSlip =
			PxMax(gMetrics.maxClutchSlip, PxAbs(clutchSlip));
		gMetrics.maxDifferentialRatioError =
			PxMax(gMetrics.maxDifferentialRatioError, ratioError);
		if(drivetrain.differentialState.nbConnectedWheels == 4 &&
			ratioError < 1e-3f)
		{
			gMetrics.fourWheelDifferentialFrames++;
		}
		if(PxAbs(clutchResponse) > 1e-4f)
			gMetrics.clutchEngagedFrames++;
	}
	const PxU32 currentGear = drivetrain.gearboxState.currentGear;
	gMetrics.minimumGear = PxMin(gMetrics.minimumGear, currentGear);
	gMetrics.maximumGear = PxMax(gMetrics.maximumGear, currentGear);
	if(gMetrics.previousGearValid && currentGear != gMetrics.previousGear)
		gMetrics.gearChanges++;
	gMetrics.previousGear = currentGear;
	gMetrics.previousGearValid = true;

	PxRigidBody* rigidBody = gVehicle.mPhysXState.physxActor.rigidBody;
	const PxTransform pose = rigidBody->getGlobalPose();
	const PxVec3 linearVelocity = rigidBody->getLinearVelocity();
	const PxVec3 angularVelocity = rigidBody->getAngularVelocity();
	if(!pose.isFinite() || !linearVelocity.isFinite() ||
		!angularVelocity.isFinite())
	{
		gMetrics.nonFinite++;
		return;
	}

	const PxVec3 forward = pose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f));
	const PxReal longitudinalSpeed =
		PxAbs(linearVelocity.dot(forward));
	gMetrics.minHeight = PxMin(gMetrics.minHeight, pose.p.y);
	gMetrics.maxHeight = PxMax(gMetrics.maxHeight, pose.p.y);
	gMetrics.maxForwardDisplacement = PxMax(
		gMetrics.maxForwardDisplacement,
		pose.p.z - gMetrics.initialPose.p.z);
	if(commandIndex == 0)
	{
		gMetrics.initialBrakeMaxSpeed = PxMax(
			gMetrics.initialBrakeMaxSpeed, longitudinalSpeed);
	}
	if(commandIndex == 1 || commandIndex == 3 || commandIndex == 4)
	{
		gMetrics.maxThrottleSpeed = PxMax(
			gMetrics.maxThrottleSpeed, longitudinalSpeed);
	}
	if(commandIndex == 2)
	{
		if(!gMetrics.brakePhaseSeen)
		{
			gMetrics.brakePhaseSeen = true;
			gMetrics.brakeStartSpeed = longitudinalSpeed;
		}
		gMetrics.minBrakeSpeed = PxMin(
			gMetrics.minBrakeSpeed, longitudinalSpeed);
	}
	if(commandIndex == 4)
	{
		if(!gMetrics.steerPhaseSeen)
		{
			gMetrics.steerPhaseSeen = true;
			gMetrics.steerStartPose = pose;
			gMetrics.steerStartHeading = getHeading(pose);
		}
		gMetrics.steerLateralDisplacement =
			pose.p.x - gMetrics.steerStartPose.p.x;
		gMetrics.steerHeadingChange = getHeadingDifference(
			getHeading(pose), gMetrics.steerStartHeading);
	}
}

static bool stepPhysicsInternal(PxReal timestep, bool collectMetrics)
{
	if(gNbCommands == gCommandProgress)
		return true;

	//Apply the brake, throttle and steer to the command state of the vehicle.
	const PxU32 commandIndex = gCommandProgress;
	const Command& command = gCommands[commandIndex];
	gVehicle.mCommandState.brakes[0] = command.brake;
	gVehicle.mCommandState.nbBrakes = 1;
	gVehicle.mCommandState.throttle = command.throttle;
	gVehicle.mCommandState.steer = command.steer;
	gVehicle.mTransmissionCommandState.targetGear = command.gear;

	//Forward integrate the vehicle by a single timestep.
	//Apply substepping at low forward speed to improve simulation fidelity.
	const PxVec3 linVel = gVehicle.mPhysXState.physxActor.rigidBody->getLinearVelocity();
	const PxVec3 forwardDir = gVehicle.mPhysXState.physxActor.rigidBody->getGlobalPose().q.getBasisVector2();
	const PxReal forwardSpeed = linVel.dot(forwardDir);
	const PxU8 nbSubsteps = (forwardSpeed < 5.0f ? 3 : 1);
	gVehicle.mComponentSequence.setSubsteps(gVehicle.mComponentSequenceSubstepGroupHandle, nbSubsteps);
	gVehicle.step(timestep, gVehicleSimulationContext);

	//Forward integrate the phsyx scene by a single timestep.
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
		collectVehicleMetrics(commandIndex);
	}

	//Increment the time spent on the current command.
	//Move to the next command in the list if enough time has lapsed.
	gCommandTime += timestep;
	if (gCommandTime > gCommands[gCommandProgress].duration)
	{
		gCommandProgress++;
		gCommandTime = 0.0f;
	}
	return true;
}

void stepPhysics()
{
	stepPhysicsInternal(1.0f/60.0f, false);
}

static bool parseHeadlessOptions(
	int argc, const char*const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.frames = 1400;
	defaults.dispatcherThreads = 4;
	defaults.dt = 0.0166667f;
	defaults.caseName = "command-cycle";
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	if(!gHeadlessOptions.headless)
		return true;
	if(gHeadlessOptions.caseName != "command-cycle")
	{
		error = "unsupported --case";
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

static bool headlessPassed()
{
	const PxReal brakeSpeedDrop =
		gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	const bool automaticShiftExpected =
		gMetrics.maxEngineSpeed > gMetrics.firstUpshiftThreshold;
	const bool automaticShiftConsistent =
		automaticShiftExpected ?
			(gMetrics.gearChanges > 0 &&
				gMetrics.maximumGear > gMetrics.minimumGear) :
			(gMetrics.gearChanges == 0 &&
				gMetrics.maximumGear == gMetrics.minimumGear);
	return gMetrics.initialized == 1 &&
		gMetrics.solverReadbackMatched &&
		gCommandProgress == gNbCommands &&
		gMetrics.completedFrames <= gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 && gMetrics.nonFinite == 0 &&
		gMetrics.drivetrainNonFinite == 0 &&
		gMetrics.wheelCount == 4 && gMetrics.constraintCount == 1 &&
		gMetrics.roadHitFrames > gMetrics.completedFrames / 2 &&
		gMetrics.driveFrames > 0 && gMetrics.brakeFrames > 0 &&
		gMetrics.activeConstraintFrames > 0 &&
		gMetrics.fourWheelDifferentialFrames >
			gMetrics.completedFrames / 2 &&
		gMetrics.clutchEngagedFrames > 0 &&
		automaticShiftConsistent &&
		gMetrics.maxEngineSpeed > 1.0f &&
		gMetrics.maxDifferentialRatioError < 1e-3f &&
		gMetrics.initialBrakeMaxSpeed < 0.5f &&
		gMetrics.maxThrottleSpeed > 1.0f &&
		brakeSpeedDrop > 0.5f &&
		gMetrics.maxForwardDisplacement > 1.0f &&
		PxAbs(gMetrics.steerLateralDisplacement) > 0.1f &&
		gMetrics.steerHeadingChange > 0.02f &&
		gMetrics.minHeight > -1.0f && gMetrics.maxHeight < 5.0f &&
		gErrorCallback.getFatalCount() == 0 &&
		gMetrics.cleanupComplete == 1;
}

static const char* failureReason()
{
	const PxReal brakeSpeedDrop =
		gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	const bool automaticShiftExpected =
		gMetrics.maxEngineSpeed > gMetrics.firstUpshiftThreshold;
	if(gMetrics.initialized != 1)
		return "initialization";
	if(!gMetrics.solverReadbackMatched)
		return "solver_readback";
	if(gCommandProgress != gNbCommands)
		return "command_cycle_incomplete";
	if(gMetrics.fetchFailures)
		return "fetch_failure";
	if(gMetrics.nonFinite)
		return "non_finite";
	if(gMetrics.drivetrainNonFinite)
		return "drivetrain_non_finite";
	if(gMetrics.wheelCount != 4 || gMetrics.constraintCount != 1)
		return "vehicle_topology";
	if(gMetrics.roadHitFrames <= gMetrics.completedFrames / 2)
		return "road_query_coverage";
	if(!gMetrics.driveFrames || !gMetrics.brakeFrames)
		return "actuation_coverage";
	if(!gMetrics.activeConstraintFrames)
		return "constraint_activity";
	if(gMetrics.fourWheelDifferentialFrames <=
		gMetrics.completedFrames / 2 ||
		gMetrics.maxDifferentialRatioError >= 1e-3f)
	{
		return "four_wheel_differential";
	}
	if(!gMetrics.clutchEngagedFrames || gMetrics.maxEngineSpeed <= 1.0f)
		return "engine_clutch_activity";
	if((automaticShiftExpected &&
			(!gMetrics.gearChanges ||
				gMetrics.maximumGear <= gMetrics.minimumGear)) ||
		(!automaticShiftExpected &&
			(gMetrics.gearChanges ||
				gMetrics.maximumGear != gMetrics.minimumGear)))
	{
		return "automatic_gearbox_activity";
	}
	if(gMetrics.initialBrakeMaxSpeed >= 0.5f)
		return "initial_brake_drift";
	if(gMetrics.maxThrottleSpeed <= 1.0f)
		return "throttle_response";
	if(brakeSpeedDrop <= 0.5f)
		return "brake_response";
	if(gMetrics.maxForwardDisplacement <= 1.0f)
		return "forward_motion";
	if(PxAbs(gMetrics.steerLateralDisplacement) <= 0.1f ||
		gMetrics.steerHeadingChange <= 0.02f)
	{
		return "steer_response";
	}
	if(gMetrics.minHeight <= -1.0f || gMetrics.maxHeight >= 5.0f)
		return "vertical_bounds";
	if(gErrorCallback.getFatalCount())
		return "physx_error";
	if(gMetrics.cleanupComplete != 1)
		return "cleanup";
	return "none";
}

static void printHeadlessResult()
{
	const bool pass = headlessPassed();
	const PxReal minimumBrakeSpeed =
		gMetrics.brakePhaseSeen ? gMetrics.minBrakeSpeed : 0.0f;
	const PxReal brakeSpeedDrop =
		gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	const PxReal minimumHeight =
		gMetrics.completedFrames ? gMetrics.minHeight : 0.0f;
	const PxReal maximumHeight =
		gMetrics.completedFrames ? gMetrics.maxHeight : 0.0f;
	const PxU32 minimumGear =
		gMetrics.completedFrames ? gMetrics.minimumGear : 0u;
	const PxU32 automaticShiftExpected =
		gMetrics.maxEngineSpeed > gMetrics.firstUpshiftThreshold ? 1u : 0u;
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetVehicleFourWheelDrive "
		"solver=%s case=command-cycle execution=%s frames=%u "
		"completedFrames=%u commands=%u wheels=%u constraints=%u "
		"roadHitFrames=%u roadHitSamples=%u driveFrames=%u brakeFrames=%u "
		"activeConstraintFrames=%u activeConstraintRows=%u "
		"fourWheelDifferentialFrames=%u clutchEngagedFrames=%u "
		"gearChanges=%u minimumGear=%u maximumGear=%u "
		"maxEngineSpeed=%.9g maxClutchSlip=%.9g "
		"maxDifferentialRatioError=%.9g firstUpshiftThreshold=%.9g "
		"automaticShiftExpected=%u "
		"initialBrakeMaxSpeed=%.9g maxThrottleSpeed=%.9g "
		"brakeStartSpeed=%.9g minBrakeSpeed=%.9g brakeSpeedDrop=%.9g "
		"maxForwardDisplacement=%.9g steerLateralDisplacement=%.9g "
		"steerHeadingChange=%.9g minHeight=%.9g maxHeight=%.9g "
		"nonFinite=%u drivetrainNonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		unsigned(gHeadlessOptions.frames), unsigned(gMetrics.completedFrames),
		unsigned(gCommandProgress), unsigned(gMetrics.wheelCount),
		unsigned(gMetrics.constraintCount), unsigned(gMetrics.roadHitFrames),
		unsigned(gMetrics.roadHitSamples), unsigned(gMetrics.driveFrames),
		unsigned(gMetrics.brakeFrames),
		unsigned(gMetrics.activeConstraintFrames),
		unsigned(gMetrics.activeConstraintRows),
		unsigned(gMetrics.fourWheelDifferentialFrames),
		unsigned(gMetrics.clutchEngagedFrames),
		unsigned(gMetrics.gearChanges), unsigned(minimumGear),
		unsigned(gMetrics.maximumGear), double(gMetrics.maxEngineSpeed),
		double(gMetrics.maxClutchSlip),
		double(gMetrics.maxDifferentialRatioError),
		double(gMetrics.firstUpshiftThreshold),
		unsigned(automaticShiftExpected),
		double(gMetrics.initialBrakeMaxSpeed),
		double(gMetrics.maxThrottleSpeed), double(gMetrics.brakeStartSpeed),
		double(minimumBrakeSpeed), double(brakeSpeedDrop),
		double(gMetrics.maxForwardDisplacement),
		double(gMetrics.steerLateralDisplacement),
		double(gMetrics.steerHeadingChange), double(minimumHeight),
		double(maximumHeight), unsigned(gMetrics.nonFinite),
		unsigned(gMetrics.drivetrainNonFinite),
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
			argc, argv, "SnippetVehicleFourWheelDrive", gVehicleDataPath))
		return 1;

	//Check that we can read from the json file before continuing.
	BaseVehicleParams baseParams;
	if (!readBaseParamsFromJsonFile(gVehicleDataPath, "Base.json", baseParams))
		return 1;

	//Check that we can read from the json file before continuing.
	EngineDrivetrainParams engineDrivetrainParams;
	if (!readEngineDrivetrainParamsFromJsonFile(gVehicleDataPath, "EngineDrive.json",
		engineDrivetrainParams))
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
			"SnippetVehicleFourWheelDrive", gHeadlessOptions);
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

#ifdef RENDER_SNIPPET
	extern void renderLoop(const char*);
	renderLoop("PhysX Snippet Vehicle Four-Wheel Drive");
#else
	if (initPhysics())
	{
		while (gCommandProgress != gNbCommands)
		{
			stepPhysics();
		}
		cleanupPhysics();
	}
#endif

	return 0;
}
