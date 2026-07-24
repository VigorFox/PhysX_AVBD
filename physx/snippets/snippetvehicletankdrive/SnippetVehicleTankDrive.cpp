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
// how to simulate a tank with a fully featured drivetrain comprising engine,
// clutch, tank differential and gears.  The snippet uses only parameters, states and 
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
	PxF32 brake0;			//Tanks have two brake controllers:
	PxF32 brake1;			//  one brake controller for the left track and one for the right track.
	PxF32 thrust0;			//Tanks have two thrust controllers that divert engine torque to the left and right tracks:
	PxF32 thrust1;			//  one thrust controller for the left track and one for the right track.
	PxF32 throttle;			//Tanks are driven by an engine that requires a throttle to generate engine drive torque.
	PxU32 gear;				//Tanks are geared and may use automatic gearing.
	PxF32 duration;			
};
const PxU32 gTargetGearCommand = 2;
Command gCommands[] =
{
	{0.5f, 0.5f, 0.0f, 0.0f, 1.0f, gTargetGearCommand, 2.0f},	//brake on and come to rest for 2 seconds
	{0.0f, 0.0f, 0.5f, 0.5f, 1.0f, gTargetGearCommand, 5.0f},	//drive forwards: symmetric forward thrust for 5 seconds
	{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, gTargetGearCommand, 5.0f},	//sharp turn: brake on track 0, forward thrust on track 1 for 5 seconds
	{0.0f, 0.0f, 1.0f, -1.0f,1.0f, gTargetGearCommand, 5.0f},	//turn on spot: forward thrust on track 0, reverse thrust on track track 1 for 5 seconds
	{0.0f, 0.0f, 1.0f, 0.25f,1.0f, gTargetGearCommand, 5.0f},	//gentle steer: asymmetric forward thrust for 5 seconds
	{0.0f, 0.0f, -1.0f,-1.0f, 1.0f, gTargetGearCommand, 5.0f}	//drive backwards: symmetric negative thrust for 5 seconds
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
	PxU32 trackCount;
	PxU32 trackWheelCount;
	PxU32 roadHitFrames;
	PxU32 roadHitSamples;
	PxU32 driveFrames;
	PxU32 brakeFrames;
	PxU32 activeConstraintFrames;
	PxU32 activeConstraintRows;
	PxU32 tankDifferentialFrames;
	PxU32 tankConstraintFrames;
	PxU32 clutchEngagedFrames;
	PxU32 commandFrames[6];
	PxU32 cleanupComplete;
	PxReal maxDifferentialRatioError;
	PxReal maxTrackSpeedMismatch;
	PxReal maxEngineSpeed;
	PxReal maxClutchSlip;
	PxReal initialBrakeMaxSpeed;
	PxReal forwardDisplacement;
	PxReal sharpTurnHeading;
	PxReal spotTurnHeading;
	PxReal spotTurnTranslation;
	PxReal gentleTurnHeading;
	PxReal reverseDisplacement;
	PxReal minHeight;
	PxReal maxHeight;
	PxTransform commandStartPose;
	PxU32 previousCommand;
	bool commandStartValid;
	bool solverReadbackMatched;

	VehicleMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0), nonFinite(0),
	  drivetrainNonFinite(0), wheelCount(0), constraintCount(0),
	  trackCount(0), trackWheelCount(0), roadHitFrames(0),
	  roadHitSamples(0), driveFrames(0), brakeFrames(0),
	  activeConstraintFrames(0), activeConstraintRows(0),
	  tankDifferentialFrames(0), tankConstraintFrames(0),
	  clutchEngagedFrames(0), cleanupComplete(0),
	  maxDifferentialRatioError(0.0f), maxTrackSpeedMismatch(0.0f),
	  maxEngineSpeed(0.0f), maxClutchSlip(0.0f),
	  initialBrakeMaxSpeed(0.0f), forwardDisplacement(0.0f),
	  sharpTurnHeading(0.0f), spotTurnHeading(0.0f),
	  spotTurnTranslation(0.0f), gentleTurnHeading(0.0f),
	  reverseDisplacement(0.0f), minHeight(FLT_MAX),
	  maxHeight(-FLT_MAX), commandStartPose(PxIdentity),
	  previousCommand(0), commandStartValid(false),
	  solverReadbackMatched(false)
	{
		PxMemZero(commandFrames, sizeof(commandFrames));
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
	if (!gVehicle.initialize(*gPhysics, PxCookingParams(PxTolerancesScale()), *gMaterial, 
		EngineDriveVehicle::eDIFFTYPE_TANKDRIVE))
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

	//Set the vehicle to use automatic gears.
	gVehicle.mTankDriveTransmissionCommandState.targetGear = PxVehicleEngineDriveTransmissionCommandState::eAUTOMATIC_GEAR;

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
	const PxVehicleTankDriveDifferentialParams& params =
		gVehicle.mEngineDriveParams.tankDifferentialParams;
	gMetrics.trackCount = params.getNbTracks();
	for(PxU32 i = 0; i < params.getNbTracks(); ++i)
		gMetrics.trackWheelCount += params.getNbWheelsInTrack(i);
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
	printf("SnippetVehicleTankDrive done.\n");
}

static PxReal getHeading(const PxTransform& pose)
{
	const PxVec3 forward = pose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f));
	return PxReal(std::atan2(double(forward.x), double(forward.z)));
}

static PxReal getHeadingDifference(
	const PxTransform& pose, const PxTransform& reference)
{
	const PxReal difference = getHeading(pose) - getHeading(reference);
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
	if(commandIndex < gNbCommands)
		gMetrics.commandFrames[commandIndex]++;

	const EngineDrivetrainState& drivetrain = gVehicle.mEngineDriveState;
	const PxVehicleTankDriveDifferentialParams& params =
		gVehicle.mEngineDriveParams.tankDifferentialParams;
	PxReal ratioError = 0.0f;
	PxReal trackSpeedMismatch = 0.0f;
	for(PxU32 track = 0; track < params.getNbTracks(); ++track)
	{
		const PxU32 thrustId = params.getThrustControllerIndex(track);
		const PxReal thrust =
			gVehicle.mTankDriveTransmissionCommandState.thrusts[thrustId];
		const PxU32 wheelCount = params.getNbWheelsInTrack(track);
		PxReal referenceTrackSpeed = 0.0f;
		for(PxU32 i = 0; i < wheelCount; ++i)
		{
			const PxU32 wheelId = params.getWheelInTrack(i, track);
			const PxReal expectedRatio =
				params.torqueRatios[wheelId] * thrust;
			ratioError = PxMax(ratioError, PxAbs(
				drivetrain.differentialState
					.torqueRatiosAllWheels[wheelId] - expectedRatio));
			const PxReal trackSpeed =
				gVehicle.mBaseState.wheelRigidBody1dStates[wheelId]
					.rotationSpeed *
				gVehicle.mBaseParams.wheelParams[wheelId].radius;
			if(i == 0)
				referenceTrackSpeed = trackSpeed;
			else
			{
				trackSpeedMismatch = PxMax(
					trackSpeedMismatch,
					PxAbs(trackSpeed - referenceTrackSpeed));
			}
		}
	}
	const PxVehicleWheelConstraintGroupState& groups =
		drivetrain.wheelConstraintGroupState;
	bool groupsValid =
		groups.getNbConstraintGroups() == params.getNbTracks() &&
		groups.nbWheelsInGroups == gMetrics.trackWheelCount;
	for(PxU32 i = 0; i < groups.getNbConstraintGroups(); ++i)
	{
		groupsValid = groupsValid &&
			groups.getNbWheelsInConstraintGroup(i) ==
				params.getNbWheelsInTrack(i);
	}
	const PxReal engineSpeed = drivetrain.engineState.rotationSpeed;
	const PxReal clutchSlip = drivetrain.clutchState.clutchSlip;
	const PxReal clutchResponse =
		drivetrain.clutchCommandResponseState.commandResponse;
	if(!PxIsFinite(ratioError) || !PxIsFinite(trackSpeedMismatch) ||
		!PxIsFinite(engineSpeed) || !PxIsFinite(clutchSlip) ||
		!PxIsFinite(clutchResponse))
	{
		gMetrics.drivetrainNonFinite++;
	}
	else
	{
		gMetrics.maxDifferentialRatioError =
			PxMax(gMetrics.maxDifferentialRatioError, ratioError);
		gMetrics.maxTrackSpeedMismatch =
			PxMax(gMetrics.maxTrackSpeedMismatch, trackSpeedMismatch);
		gMetrics.maxEngineSpeed =
			PxMax(gMetrics.maxEngineSpeed, PxAbs(engineSpeed));
		gMetrics.maxClutchSlip =
			PxMax(gMetrics.maxClutchSlip, PxAbs(clutchSlip));
		if(drivetrain.differentialState.nbConnectedWheels ==
			axle.nbWheels && ratioError < 1e-5f)
		{
			gMetrics.tankDifferentialFrames++;
		}
		if(groupsValid)
			gMetrics.tankConstraintFrames++;
		if(PxAbs(clutchResponse) > 1e-4f)
			gMetrics.clutchEngagedFrames++;
	}

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
	gMetrics.minHeight = PxMin(gMetrics.minHeight, pose.p.y);
	gMetrics.maxHeight = PxMax(gMetrics.maxHeight, pose.p.y);
	if(commandIndex == 0)
		gMetrics.initialBrakeMaxSpeed = PxMax(
			gMetrics.initialBrakeMaxSpeed, linearVelocity.magnitude());
	if(!gMetrics.commandStartValid ||
		commandIndex != gMetrics.previousCommand)
	{
		gMetrics.commandStartPose = pose;
		gMetrics.previousCommand = commandIndex;
		gMetrics.commandStartValid = true;
	}
	const PxVec3 displacement = pose.p - gMetrics.commandStartPose.p;
	const PxVec3 startForward =
		gMetrics.commandStartPose.q.rotate(PxVec3(0.0f, 0.0f, 1.0f));
	const PxReal longitudinal = displacement.dot(startForward);
	const PxReal heading =
		getHeadingDifference(pose, gMetrics.commandStartPose);
	if(commandIndex == 1)
		gMetrics.forwardDisplacement =
			PxMax(gMetrics.forwardDisplacement, longitudinal);
	else if(commandIndex == 2)
		gMetrics.sharpTurnHeading =
			PxMax(gMetrics.sharpTurnHeading, heading);
	else if(commandIndex == 3)
	{
		gMetrics.spotTurnHeading =
			PxMax(gMetrics.spotTurnHeading, heading);
		gMetrics.spotTurnTranslation = PxMax(
			gMetrics.spotTurnTranslation,
			PxSqrt(displacement.x * displacement.x +
				displacement.z * displacement.z));
	}
	else if(commandIndex == 4)
		gMetrics.gentleTurnHeading =
			PxMax(gMetrics.gentleTurnHeading, heading);
	else if(commandIndex == 5)
		gMetrics.reverseDisplacement =
			PxMax(gMetrics.reverseDisplacement, -longitudinal);
}

static bool stepPhysicsInternal(PxReal timestep, bool collectMetrics)
{
	if (gNbCommands == gCommandProgress)
		return true;

	//Apply the brake, throttle and thrusts to the command state of the tank.
	const PxU32 commandIndex = gCommandProgress;
	const Command& command = gCommands[commandIndex];
	gVehicle.mCommandState.brakes[0] = command.brake0;
	gVehicle.mCommandState.brakes[1] = command.brake1;
	gVehicle.mCommandState.nbBrakes = 2;
	gVehicle.mCommandState.throttle = command.throttle;
	gVehicle.mCommandState.steer = 0.0f;
	gVehicle.mTankDriveTransmissionCommandState.thrusts[0] = command.thrust0;
	gVehicle.mTankDriveTransmissionCommandState.thrusts[1] = command.thrust1;
	gVehicle.mTankDriveTransmissionCommandState.targetGear = command.gear;

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
	defaults.frames = 1700;
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
			gVehicleDataPath = arg + std::strlen("--vehicleDataPath=");
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
	bool commandsCovered = true;
	for(PxU32 i = 0; i < gNbCommands; ++i)
		commandsCovered = commandsCovered && gMetrics.commandFrames[i] > 0;
	return gMetrics.initialized == 1 &&
		gMetrics.solverReadbackMatched &&
		gCommandProgress == gNbCommands &&
		gMetrics.completedFrames <= gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 && gMetrics.nonFinite == 0 &&
		gMetrics.drivetrainNonFinite == 0 &&
		gMetrics.wheelCount == 4 && gMetrics.constraintCount == 1 &&
		gMetrics.trackCount == 2 && gMetrics.trackWheelCount == 4 &&
		commandsCovered &&
		gMetrics.roadHitFrames > gMetrics.completedFrames / 2 &&
		gMetrics.driveFrames > 0 && gMetrics.brakeFrames > 0 &&
		gMetrics.activeConstraintFrames > 0 &&
		gMetrics.tankDifferentialFrames == gMetrics.completedFrames &&
		gMetrics.tankConstraintFrames == gMetrics.completedFrames &&
		gMetrics.clutchEngagedFrames > 0 &&
		gMetrics.maxDifferentialRatioError < 1e-5f &&
		gMetrics.maxEngineSpeed > 1.0f &&
		gMetrics.initialBrakeMaxSpeed < 0.5f &&
		gMetrics.forwardDisplacement > 1.0f &&
		gMetrics.sharpTurnHeading > 0.1f &&
		gMetrics.spotTurnHeading > 0.2f &&
		gMetrics.gentleTurnHeading > 0.05f &&
		gMetrics.reverseDisplacement > 0.5f &&
		gMetrics.minHeight > -1.0f && gMetrics.maxHeight < 5.0f &&
		gErrorCallback.getFatalCount() == 0 &&
		gMetrics.cleanupComplete == 1;
}

static const char* failureReason()
{
	bool commandsCovered = true;
	for(PxU32 i = 0; i < gNbCommands; ++i)
		commandsCovered = commandsCovered && gMetrics.commandFrames[i] > 0;
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
	if(gMetrics.wheelCount != 4 || gMetrics.constraintCount != 1 ||
		gMetrics.trackCount != 2 || gMetrics.trackWheelCount != 4)
		return "tank_topology";
	if(!commandsCovered)
		return "command_coverage";
	if(gMetrics.roadHitFrames <= gMetrics.completedFrames / 2)
		return "road_query_coverage";
	if(!gMetrics.driveFrames || !gMetrics.brakeFrames)
		return "actuation_coverage";
	if(!gMetrics.activeConstraintFrames)
		return "constraint_activity";
	if(gMetrics.tankDifferentialFrames != gMetrics.completedFrames ||
		gMetrics.maxDifferentialRatioError >= 1e-5f)
		return "tank_differential";
	if(gMetrics.tankConstraintFrames != gMetrics.completedFrames)
		return "tank_track_constraint";
	if(!gMetrics.clutchEngagedFrames || gMetrics.maxEngineSpeed <= 1.0f)
		return "engine_clutch_activity";
	if(gMetrics.initialBrakeMaxSpeed >= 0.5f)
		return "initial_brake_drift";
	if(gMetrics.forwardDisplacement <= 1.0f)
		return "forward_response";
	if(gMetrics.sharpTurnHeading <= 0.1f)
		return "sharp_turn_response";
	if(gMetrics.spotTurnHeading <= 0.2f)
		return "spot_turn_response";
	if(gMetrics.gentleTurnHeading <= 0.05f)
		return "gentle_turn_response";
	if(gMetrics.reverseDisplacement <= 0.5f)
		return "reverse_response";
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
	const PxReal minimumHeight =
		gMetrics.completedFrames ? gMetrics.minHeight : 0.0f;
	const PxReal maximumHeight =
		gMetrics.completedFrames ? gMetrics.maxHeight : 0.0f;
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetVehicleTankDrive "
		"solver=%s case=command-cycle execution=%s frames=%u "
		"completedFrames=%u commands=%u wheels=%u constraints=%u "
		"tracks=%u trackWheels=%u roadHitFrames=%u roadHitSamples=%u "
		"driveFrames=%u brakeFrames=%u activeConstraintFrames=%u "
		"activeConstraintRows=%u tankDifferentialFrames=%u "
		"tankConstraintFrames=%u clutchEngagedFrames=%u "
		"command0Frames=%u command1Frames=%u command2Frames=%u "
		"command3Frames=%u command4Frames=%u command5Frames=%u "
		"maxDifferentialRatioError=%.9g maxTrackSpeedMismatch=%.9g "
		"maxEngineSpeed=%.9g maxClutchSlip=%.9g "
		"initialBrakeMaxSpeed=%.9g forwardDisplacement=%.9g "
		"sharpTurnHeading=%.9g spotTurnHeading=%.9g "
		"spotTurnTranslation=%.9g gentleTurnHeading=%.9g "
		"reverseDisplacement=%.9g minHeight=%.9g maxHeight=%.9g "
		"nonFinite=%u drivetrainNonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		unsigned(gHeadlessOptions.frames), unsigned(gMetrics.completedFrames),
		unsigned(gCommandProgress), unsigned(gMetrics.wheelCount),
		unsigned(gMetrics.constraintCount), unsigned(gMetrics.trackCount),
		unsigned(gMetrics.trackWheelCount), unsigned(gMetrics.roadHitFrames),
		unsigned(gMetrics.roadHitSamples), unsigned(gMetrics.driveFrames),
		unsigned(gMetrics.brakeFrames),
		unsigned(gMetrics.activeConstraintFrames),
		unsigned(gMetrics.activeConstraintRows),
		unsigned(gMetrics.tankDifferentialFrames),
		unsigned(gMetrics.tankConstraintFrames),
		unsigned(gMetrics.clutchEngagedFrames),
		unsigned(gMetrics.commandFrames[0]),
		unsigned(gMetrics.commandFrames[1]),
		unsigned(gMetrics.commandFrames[2]),
		unsigned(gMetrics.commandFrames[3]),
		unsigned(gMetrics.commandFrames[4]),
		unsigned(gMetrics.commandFrames[5]),
		double(gMetrics.maxDifferentialRatioError),
		double(gMetrics.maxTrackSpeedMismatch),
		double(gMetrics.maxEngineSpeed), double(gMetrics.maxClutchSlip),
		double(gMetrics.initialBrakeMaxSpeed),
		double(gMetrics.forwardDisplacement),
		double(gMetrics.sharpTurnHeading),
		double(gMetrics.spotTurnHeading),
		double(gMetrics.spotTurnTranslation),
		double(gMetrics.gentleTurnHeading),
		double(gMetrics.reverseDisplacement), double(minimumHeight),
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
			argc, argv, "SnippetVehicleTankDrive", gVehicleDataPath))
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
			"SnippetVehicleTankDrive", gHeadlessOptions);
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
	renderLoop("PhysX Snippet Vehicle Tank Drive");
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
