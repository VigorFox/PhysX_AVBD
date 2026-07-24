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
// how to simulate multiple vehicles jointed together.  The snippet introduces 
// the simple example of a tractor pulling a trailer.  The wheels of the tractor
// respond to brake, throttle and steer.  The trailer, on the other hand, has no 
// engine or steering column and is only able to apply brake torques to the wheels.
// The snippet uses only parameters, states and components maintained by the PhysX Vehicle SDK.

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
#include "../snippetvehiclecommon/serialization/DirectDrivetrainSerialization.h"
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
bool					gExtensionsInitialized = false;
bool					gVehicleExtensionInitialized = false;
bool					gTractorInitialized = false;
bool					gTrailerInitialized = false;

//The path to the vehicle json files to be loaded.
const char* gVehicleDataPath = NULL;

//The tractor with engine drivetrain.
//The trailer with direct drivetrain
//A joint connecting the two vehicles together.
EngineDriveVehicle gTractor;
DirectDriveVehicle gTrailer;
PxD6Joint* gJoint = NULL;
PxTransform gAnchorTractorFrame(PxIdentity);
PxTransform gAnchorTrailerFrame(PxIdentity);

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

//Give the vehicles names so they can be identified in PVD.
const char gTractorName[] = "tractor";
const char gTrailerName[] = "trailer";

//Commands are issued to the vehicles in a pre-choreographed sequence.
//Note: 
//   the tractor responds to brake, throttle and steer commands.
//   the trailer responds only to brake commands.
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
	{0.0f, 0.75f, 0.0f, gTargetGearCommand, 5.0f}	//throttle for 5 seconds
};
const PxU32 gNbCommands = sizeof(gCommands) / sizeof(Command);
PxReal gCommandTime = 0.0f;			//Time spent on current command
PxU32 gCommandProgress = 0;			//The id of the current command.

//A ground plane to drive on.
PxRigidStatic*	gGroundPlane = NULL;

struct TruckMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 drivetrainNonFinite;
	PxU32 tractorWheels;
	PxU32 trailerWheels;
	PxU32 tractorConstraints;
	PxU32 trailerConstraints;
	PxU32 tractorRoadHitFrames;
	PxU32 trailerRoadHitFrames;
	PxU32 tractorRoadHitSamples;
	PxU32 trailerRoadHitSamples;
	PxU32 tractorDriveFrames;
	PxU32 tractorBrakeFrames;
	PxU32 trailerDriveFrames;
	PxU32 trailerBrakeFrames;
	PxU32 tractorActiveRows;
	PxU32 trailerActiveRows;
	PxU32 accelerationIntentFrames;
	PxU32 clutchEngagedFrames;
	PxU32 commandFrames[4];
	PxU32 jointConfigured;
	PxU32 jointPresentDuringRun;
	PxU32 jointForceFrames;
	PxU32 cleanupComplete;
	PxReal maxJointAnchorError;
	PxReal finalJointAnchorError;
	PxReal maxJointLinearForce;
	PxReal maxJointAngularForce;
	PxReal maxEngineSpeed;
	PxReal maxClutchSlip;
	PxReal initialBrakeMaxSpeed;
	PxReal throttleMaxSpeed;
	PxReal brakeStartSpeed;
	PxReal minBrakeSpeed;
	PxReal tractorForwardDisplacement;
	PxReal trailerForwardDisplacement;
	PxReal finalActorSeparation;
	PxReal minTractorHeight;
	PxReal maxTractorHeight;
	PxReal minTrailerHeight;
	PxReal maxTrailerHeight;
	PxTransform initialTractorPose;
	PxTransform initialTrailerPose;
	bool brakePhaseSeen;
	bool solverReadbackMatched;

	TruckMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0), nonFinite(0),
	  drivetrainNonFinite(0), tractorWheels(0), trailerWheels(0),
	  tractorConstraints(0), trailerConstraints(0),
	  tractorRoadHitFrames(0), trailerRoadHitFrames(0),
	  tractorRoadHitSamples(0), trailerRoadHitSamples(0),
	  tractorDriveFrames(0), tractorBrakeFrames(0),
	  trailerDriveFrames(0), trailerBrakeFrames(0),
	  tractorActiveRows(0), trailerActiveRows(0),
	  accelerationIntentFrames(0), clutchEngagedFrames(0),
	  jointConfigured(0), jointPresentDuringRun(0), jointForceFrames(0),
	  cleanupComplete(0),
	  maxJointAnchorError(0.0f), finalJointAnchorError(0.0f),
	  maxJointLinearForce(0.0f), maxJointAngularForce(0.0f),
	  maxEngineSpeed(0.0f), maxClutchSlip(0.0f),
	  initialBrakeMaxSpeed(0.0f), throttleMaxSpeed(0.0f),
	  brakeStartSpeed(0.0f), minBrakeSpeed(FLT_MAX),
	  tractorForwardDisplacement(0.0f), trailerForwardDisplacement(0.0f),
	  finalActorSeparation(0.0f), minTractorHeight(FLT_MAX),
	  maxTractorHeight(-FLT_MAX), minTrailerHeight(FLT_MAX),
	  maxTrailerHeight(-FLT_MAX), initialTractorPose(PxIdentity),
	  initialTrailerPose(PxIdentity), brakePhaseSeen(false),
	  solverReadbackMatched(false)
	{
		PxMemZero(commandFrames, sizeof(commandFrames));
	}
};

TruckMetrics gMetrics;

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
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, false);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);	
	if(!gMaterial)
		return false;

	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if(!gExtensionsInitialized)
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
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
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
	//Load the tractor params from json or set directly.
	readBaseParamsFromJsonFile(gVehicleDataPath, "Base.json", gTractor.mBaseParams);
	setPhysXIntegrationParams(gTractor.mBaseParams.axleDescription,
		gPhysXMaterialFrictions, gNbPhysXMaterialFrictions, gPhysXDefaultMaterialFriction,
		gTractor.mPhysXParams);
	readEngineDrivetrainParamsFromJsonFile(gVehicleDataPath, "EngineDrive.json", 
		gTractor.mEngineDriveParams);

	//Load the trailer params from json or set directly.
	readBaseParamsFromJsonFile(gVehicleDataPath, "Base.json", gTrailer.mBaseParams);
	setPhysXIntegrationParams(gTrailer.mBaseParams.axleDescription,
		gPhysXMaterialFrictions, gNbPhysXMaterialFrictions, gPhysXDefaultMaterialFriction,
		gTrailer.mPhysXParams);
	readDirectDrivetrainParamsFromJsonFile(gVehicleDataPath, "DirectDrive.json", gTrailer.mBaseParams.axleDescription,
		gTrailer.mDirectDriveParams);

	//Set the states to default.
	if (!gTractor.initialize(*gPhysics, PxCookingParams(PxTolerancesScale()), *gMaterial, 
		EngineDriveVehicle::eDIFFTYPE_FOURWHEELDRIVE))
	{
		return false;
	}
	gTractorInitialized = true;
	if (!gTrailer.initialize(*gPhysics, PxCookingParams(PxTolerancesScale()), *gMaterial))
	{
		return false;
	}
	gTrailerInitialized = true;

	//Create a PhysX joint to connect tractor and trailer.
	//Create a joint anchor that is 1.5m behind the rear wheels of the tractor and 1.5m in front of the front wheels of the trailer.
	gAnchorTractorFrame = PxTransform(PxIdentity);
	{
		//Rear wheels are 2 and 3.
		PxRigidBody* rigidActor = gTractor.mPhysXState.physxActor.rigidBody;
		const PxTransform cMassLocalPoseActorFrame = rigidActor->getCMassLocalPose();
		const PxVec3 frontAxlePosCMassFrame = (gTractor.mBaseParams.suspensionParams[2].suspensionAttachment.p + gTractor.mBaseParams.suspensionParams[3].suspensionAttachment.p)*0.5f;
		const PxQuat frontAxleQuatCMassFrame = gTractor.mBaseParams.suspensionParams[2].suspensionAttachment.q;
		const PxTransform anchorCMassFrame(frontAxlePosCMassFrame - PxVec3(0, 0, 1.5f), frontAxleQuatCMassFrame);
		gAnchorTractorFrame = cMassLocalPoseActorFrame * anchorCMassFrame;
	}
	gAnchorTrailerFrame = PxTransform(PxIdentity);
	{
		//Front wheels are 0 and 1.
		PxRigidBody* rigidActor = gTrailer.mPhysXState.physxActor.rigidBody;
		const PxTransform cMassLocalPoseActorFrame = rigidActor->getCMassLocalPose();
		const PxVec3 rearAxlePosCMassFrame = (gTrailer.mBaseParams.suspensionParams[0].suspensionAttachment.p + gTractor.mBaseParams.suspensionParams[1].suspensionAttachment.p)*0.5f;
		const PxQuat rearAxleQuatCMassFrame = gTrailer.mBaseParams.suspensionParams[0].suspensionAttachment.q;
		const PxTransform anchorCMassFrame(rearAxlePosCMassFrame + PxVec3(0, 0, 1.5f), rearAxleQuatCMassFrame);
		gAnchorTrailerFrame = cMassLocalPoseActorFrame * anchorCMassFrame;
	}

	//Apply a start pose to the physx actor of tractor and trailer and add them to the physx scene.
	const PxTransform tractorPose(PxVec3(0.000000000f, -0.0500000119f, -1.59399998f), PxQuat(PxIdentity));
	gTractor.setUpActor(*gScene, tractorPose, gTractorName);
	const PxTransform trailerPose =
		tractorPose * gAnchorTractorFrame *
		gAnchorTrailerFrame.getInverse();
	gTrailer.setUpActor(*gScene, trailerPose, gTrailerName);

	//Create a joint between tractor and trailer.
	{
		gJoint = PxD6JointCreate(
			*gPhysics,
			gTractor.mPhysXState.physxActor.rigidBody,
			gAnchorTractorFrame,
			gTrailer.mPhysXState.physxActor.rigidBody,
			gAnchorTrailerFrame);
		if(!gJoint)
			return false;
		gJoint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
		gJoint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
		gJoint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
		gJoint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
		gJoint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
		gJoint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
		gJoint->getConstraint()->setFlags(gJoint->getConstraint()->getFlags() | PxConstraintFlag::eCOLLISION_ENABLED);
	}

	//Set the tractor in 1st gear and to use the autobox
	gTractor.mEngineDriveState.gearboxState.currentGear = gTractor.mEngineDriveParams.gearBoxParams.neutralGear + 1;
	gTractor.mEngineDriveState.gearboxState.targetGear = gTractor.mEngineDriveParams.gearBoxParams.neutralGear + 1;
	gTractor.mTransmissionCommandState.targetGear = PxVehicleEngineDriveTransmissionCommandState::eAUTOMATIC_GEAR;

	//Set the trailer in neutral gear to prevent any drive torques being applied to the trailer.
	gTrailer.mTransmissionCommandState.gear = PxVehicleDirectDriveTransmissionCommandState::eNEUTRAL;

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
	PX_RELEASE(gJoint);
	if(gTractorInitialized)
	{
		gTractor.destroy();
		gTractorInitialized = false;
	}
	if(gTrailerInitialized)
	{
		gTrailer.destroy();
		gTrailerInitialized = false;
	}
}

static bool initPhysicsInternal(bool headless)
{
	gMetrics = TruckMetrics();
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
	gMetrics.tractorWheels =
		gTractor.mBaseParams.axleDescription.nbWheels;
	gMetrics.trailerWheels =
		gTrailer.mBaseParams.axleDescription.nbWheels;
	for(PxU32 i = 0;
		i < PxVehiclePhysXConstraintLimits::eNB_CONSTRAINTS_PER_VEHICLE; ++i)
	{
		if(gTractor.mPhysXState.physxConstraints.constraints[i])
			gMetrics.tractorConstraints++;
		if(gTrailer.mPhysXState.physxConstraints.constraints[i])
			gMetrics.trailerConstraints++;
	}
	gMetrics.initialTractorPose =
		gTractor.mPhysXState.physxActor.rigidBody->getGlobalPose();
	gMetrics.initialTrailerPose =
		gTrailer.mPhysXState.physxActor.rigidBody->getGlobalPose();
	if(gJoint)
	{
		const PxConstraintFlags flags =
			gJoint->getConstraint()->getFlags();
		gMetrics.jointConfigured =
			gJoint->getMotion(PxD6Axis::eX) == PxD6Motion::eLOCKED &&
			gJoint->getMotion(PxD6Axis::eY) == PxD6Motion::eLOCKED &&
			gJoint->getMotion(PxD6Axis::eZ) == PxD6Motion::eLOCKED &&
			gJoint->getMotion(PxD6Axis::eTWIST) == PxD6Motion::eLOCKED &&
			gJoint->getMotion(PxD6Axis::eSWING1) == PxD6Motion::eFREE &&
			gJoint->getMotion(PxD6Axis::eSWING2) == PxD6Motion::eFREE &&
			flags.isSet(PxConstraintFlag::eCOLLISION_ENABLED) ? 1u : 0u;
	}
	if(headless && gHeadlessOptions.caseName == "uncoupled")
		PX_RELEASE(gJoint);
	gMetrics.jointPresentDuringRun = gJoint ? 1u : 0u;
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
		!gJoint && !gTractorInitialized && !gTrailerInitialized &&
		!gGroundPlane && !gMaterial && !gScene && !gDispatcher &&
		!gPhysics && !gPvd && !gFoundation ? 1u : 0u;
	printf("SnippetVehicleTruck done.\n");
}

static void collectWheelMetrics(
	const PhysXActorVehicle& vehicle,
	PxU32& roadHitFrames,
	PxU32& roadHitSamples,
	PxU32& driveFrames,
	PxU32& brakeFrames,
	PxU32& activeRows)
{
	const PxVehicleAxleDescription& axle =
		vehicle.mBaseParams.axleDescription;
	PxU32 roadHits = 0;
	bool driveApplied = false;
	bool brakeApplied = false;
	PxU32 frameRows = 0;
	for(PxU32 i = 0; i < axle.nbWheels; ++i)
	{
		const PxU32 wheelId = axle.wheelIdsInAxleOrder[i];
		if(vehicle.mBaseState.roadGeomStates[wheelId].hitState)
			roadHits++;
		const PxVehicleWheelActuationState& actuation =
			vehicle.mBaseState.actuationStates[wheelId];
		driveApplied = driveApplied || actuation.isDriveApplied;
		brakeApplied = brakeApplied || actuation.isBrakeApplied;
		const PxVehiclePhysXConstraintState& constraint =
			vehicle.mPhysXState.physxConstraints.constraintStates[wheelId];
		if(constraint.suspActiveStatus)
			frameRows++;
		for(PxU32 direction = 0;
			direction < PxVehicleTireDirectionModes::eMAX_NB_PLANAR_DIRECTIONS;
			++direction)
		{
			if(constraint.tireActiveStatus[direction])
				frameRows++;
		}
	}
	roadHitSamples += roadHits;
	if(roadHits == axle.nbWheels)
		roadHitFrames++;
	if(driveApplied)
		driveFrames++;
	if(brakeApplied)
		brakeFrames++;
	activeRows += frameRows;
}

static void collectTruckMetrics(PxU32 commandIndex)
{
	collectWheelMetrics(
		gTractor, gMetrics.tractorRoadHitFrames,
		gMetrics.tractorRoadHitSamples, gMetrics.tractorDriveFrames,
		gMetrics.tractorBrakeFrames, gMetrics.tractorActiveRows);
	collectWheelMetrics(
		gTrailer, gMetrics.trailerRoadHitFrames,
		gMetrics.trailerRoadHitSamples, gMetrics.trailerDriveFrames,
		gMetrics.trailerBrakeFrames, gMetrics.trailerActiveRows);
	if(commandIndex < gNbCommands)
		gMetrics.commandFrames[commandIndex]++;

	const EngineDrivetrainState& drivetrain = gTractor.mEngineDriveState;
	const PxReal engineSpeed = drivetrain.engineState.rotationSpeed;
	const PxReal clutchSlip = drivetrain.clutchState.clutchSlip;
	const PxReal clutchResponse =
		drivetrain.clutchCommandResponseState.commandResponse;
	if(!PxIsFinite(engineSpeed) || !PxIsFinite(clutchSlip) ||
		!PxIsFinite(clutchResponse))
	{
		gMetrics.drivetrainNonFinite++;
	}
	else
	{
		gMetrics.maxEngineSpeed =
			PxMax(gMetrics.maxEngineSpeed, PxAbs(engineSpeed));
		gMetrics.maxClutchSlip =
			PxMax(gMetrics.maxClutchSlip, PxAbs(clutchSlip));
		if(PxAbs(clutchResponse) > 1e-4f)
			gMetrics.clutchEngagedFrames++;
	}

	PxRigidBody* tractorBody =
		gTractor.mPhysXState.physxActor.rigidBody;
	PxRigidBody* trailerBody =
		gTrailer.mPhysXState.physxActor.rigidBody;
	const PxTransform tractorPose = tractorBody->getGlobalPose();
	const PxTransform trailerPose = trailerBody->getGlobalPose();
	const PxVec3 tractorVelocity = tractorBody->getLinearVelocity();
	const PxVec3 trailerVelocity = trailerBody->getLinearVelocity();
	const PxVec3 tractorAngular = tractorBody->getAngularVelocity();
	const PxVec3 trailerAngular = trailerBody->getAngularVelocity();
	if(!tractorPose.isFinite() || !trailerPose.isFinite() ||
		!tractorVelocity.isFinite() || !trailerVelocity.isFinite() ||
		!tractorAngular.isFinite() || !trailerAngular.isFinite())
	{
		gMetrics.nonFinite++;
		return;
	}
	gMetrics.minTractorHeight =
		PxMin(gMetrics.minTractorHeight, tractorPose.p.y);
	gMetrics.maxTractorHeight =
		PxMax(gMetrics.maxTractorHeight, tractorPose.p.y);
	gMetrics.minTrailerHeight =
		PxMin(gMetrics.minTrailerHeight, trailerPose.p.y);
	gMetrics.maxTrailerHeight =
		PxMax(gMetrics.maxTrailerHeight, trailerPose.p.y);
	gMetrics.tractorForwardDisplacement = PxMax(
		gMetrics.tractorForwardDisplacement,
		tractorPose.p.z - gMetrics.initialTractorPose.p.z);
	gMetrics.trailerForwardDisplacement = PxMax(
		gMetrics.trailerForwardDisplacement,
		trailerPose.p.z - gMetrics.initialTrailerPose.p.z);
	gMetrics.finalActorSeparation =
		(tractorPose.p - trailerPose.p).magnitude();

	const PxReal tractorSpeed = tractorVelocity.magnitude();
	if(commandIndex == 0)
		gMetrics.initialBrakeMaxSpeed =
			PxMax(gMetrics.initialBrakeMaxSpeed, tractorSpeed);
	if(commandIndex == 1 || commandIndex == 3)
		gMetrics.throttleMaxSpeed =
			PxMax(gMetrics.throttleMaxSpeed, tractorSpeed);
	if(commandIndex == 2)
	{
		if(!gMetrics.brakePhaseSeen)
		{
			gMetrics.brakePhaseSeen = true;
			gMetrics.brakeStartSpeed = tractorSpeed;
		}
		gMetrics.minBrakeSpeed =
			PxMin(gMetrics.minBrakeSpeed, tractorSpeed);
	}

	const PxTransform tractorAnchor =
		tractorPose * gAnchorTractorFrame;
	const PxTransform trailerAnchor =
		trailerPose * gAnchorTrailerFrame;
	gMetrics.finalJointAnchorError =
		(tractorAnchor.p - trailerAnchor.p).magnitude();
	gMetrics.maxJointAnchorError = PxMax(
		gMetrics.maxJointAnchorError,
		gMetrics.finalJointAnchorError);
	if(gJoint)
	{
		PxVec3 linearForce(0.0f);
		PxVec3 angularForce(0.0f);
		gJoint->getConstraint()->getForce(linearForce, angularForce);
		if(!linearForce.isFinite() || !angularForce.isFinite())
			gMetrics.nonFinite++;
		else
		{
			const PxReal linearMagnitude = linearForce.magnitude();
			const PxReal angularMagnitude = angularForce.magnitude();
			gMetrics.maxJointLinearForce = PxMax(
				gMetrics.maxJointLinearForce, linearMagnitude);
			gMetrics.maxJointAngularForce = PxMax(
				gMetrics.maxJointAngularForce, angularMagnitude);
			if(linearMagnitude > 1e-4f || angularMagnitude > 1e-4f)
				gMetrics.jointForceFrames++;
		}
	}
}

static bool stepPhysicsInternal(PxReal timestep, bool collectMetrics)
{
	if (gNbCommands == gCommandProgress)
		return true;

	//Apply the brake, throttle and steer to the command state of the tractor.
	const PxU32 commandIndex = gCommandProgress;
	const Command& command = gCommands[commandIndex];
	gTractor.mCommandState.brakes[0] = command.brake;
	gTractor.mCommandState.nbBrakes = 1;
	gTractor.mCommandState.throttle = command.throttle;
	gTractor.mCommandState.steer = command.steer;
	gTractor.mTransmissionCommandState.targetGear = command.gear;

	//Apply the brake to the command state of the trailer.
	gTrailer.mCommandState.brakes[0] = command.brake;
	gTrailer.mCommandState.nbBrakes = 1;

	//Apply substepping at low forward speed to improve simulation fidelity.
	//Tractor and trailer will have approximately the same forward speed so we can apply
	//the same substepping rules to the tractor and trailer.
	const PxVec3 linVel = gTractor.mPhysXState.physxActor.rigidBody->getLinearVelocity();
	const PxVec3 forwardDir = gTractor.mPhysXState.physxActor.rigidBody->getGlobalPose().q.getBasisVector2();
	const PxReal forwardSpeed = linVel.dot(forwardDir);
	const PxU8 nbSubsteps = (forwardSpeed < 5.0f ? 3 : 1);
	gTractor.mComponentSequence.setSubsteps(gTractor.mComponentSequenceSubstepGroupHandle, nbSubsteps);
	gTrailer.mComponentSequence.setSubsteps(gTrailer.mComponentSequenceSubstepGroupHandle, nbSubsteps);

	//Reset the sticky states on the trailer using the actuation state of the truck.
	//Vehicles are brought to rest with sticky constraints that apply velocity constraints to the tires of the vehicle.
	//A drive torque applied to any wheel of the tractor signals an intent to accelerate.
	//An intent to accelerate will release any sticky constraints applied to the tires of the tractor.
	//It is important to apply the intent to accelerate to the wheels of the trailer as well to prevent the trailer being
	//held at rest with its own sticky constraints. 
	//It is not possible to determine an intent to accelerate from the trailer alone because the wheels of the trailer
	//do not respond to the throttle commands and therefore receive zero drive torque.
	//The function PxVehicleTireStickyStateReset() will reset the sticky states of the trailer using an intention to 
	//accelerate derived from the state of the tractor.
	const PxVehicleArrayData<const PxVehicleWheelActuationState> tractorActuationStates(gTractor.mBaseState.actuationStates);
	PxVehicleArrayData<PxVehicleTireStickyState> trailerTireStickyStates(gTrailer.mBaseState.tireStickyStates);
	const bool intentionToAccelerate = PxVehicleAccelerationIntentCompute(gTractor.mBaseParams.axleDescription, tractorActuationStates);
	PxVehicleTireStickyStateReset(intentionToAccelerate, gTrailer.mBaseParams.axleDescription, trailerTireStickyStates);
	if(collectMetrics && intentionToAccelerate)
		gMetrics.accelerationIntentFrames++;

	//Forward integrate the vehicles by a single timestep.
	gTractor.step(timestep, gVehicleSimulationContext);
	gTrailer.step(timestep, gVehicleSimulationContext);
	
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
		collectTruckMetrics(commandIndex);
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
	defaults.frames = 1100;
	defaults.dispatcherThreads = 4;
	defaults.dt = 0.0166667f;
	defaults.caseName = "coupled";
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	if(!gHeadlessOptions.headless)
		return true;
	if(gHeadlessOptions.caseName != "coupled" &&
		gHeadlessOptions.caseName != "uncoupled")
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
	const bool coupled = gHeadlessOptions.caseName == "coupled";
	const PxReal brakeDrop = gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	bool commandsCovered = true;
	for(PxU32 i = 0; i < gNbCommands; ++i)
		commandsCovered = commandsCovered && gMetrics.commandFrames[i] > 0;
	const bool couplingValid = coupled ?
		(gMetrics.jointPresentDuringRun == 1 &&
			gMetrics.maxJointAnchorError < 0.5f &&
			gMetrics.trailerForwardDisplacement > 1.0f) :
		(gMetrics.jointPresentDuringRun == 0 &&
			gMetrics.trailerForwardDisplacement < 1.0f);
	return gMetrics.initialized == 1 &&
		gMetrics.solverReadbackMatched &&
		gCommandProgress == gNbCommands &&
		gMetrics.completedFrames <= gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 && gMetrics.nonFinite == 0 &&
		gMetrics.drivetrainNonFinite == 0 &&
		gMetrics.tractorWheels == 4 && gMetrics.trailerWheels == 4 &&
		gMetrics.tractorConstraints == 1 &&
		gMetrics.trailerConstraints == 1 &&
		gMetrics.jointConfigured == 1 && couplingValid &&
		commandsCovered &&
		gMetrics.tractorRoadHitFrames > gMetrics.completedFrames / 2 &&
		gMetrics.trailerRoadHitFrames > gMetrics.completedFrames / 2 &&
		gMetrics.tractorDriveFrames > 0 &&
		gMetrics.tractorBrakeFrames > 0 &&
		gMetrics.trailerDriveFrames == 0 &&
		gMetrics.trailerBrakeFrames > 0 &&
		gMetrics.tractorActiveRows > 0 &&
		gMetrics.trailerActiveRows > 0 &&
		gMetrics.accelerationIntentFrames > 0 &&
		gMetrics.clutchEngagedFrames > 0 &&
		gMetrics.maxEngineSpeed > 1.0f &&
		gMetrics.initialBrakeMaxSpeed < 1.0f &&
		gMetrics.throttleMaxSpeed > 1.0f &&
		brakeDrop > 0.5f &&
		gMetrics.tractorForwardDisplacement > 1.0f &&
		gMetrics.minTractorHeight > -1.0f &&
		gMetrics.maxTractorHeight < 5.0f &&
		gMetrics.minTrailerHeight > -1.0f &&
		gMetrics.maxTrailerHeight < 5.0f &&
		gErrorCallback.getFatalCount() == 0 &&
		gMetrics.cleanupComplete == 1;
}

static const char* failureReason()
{
	const bool coupled = gHeadlessOptions.caseName == "coupled";
	const PxReal brakeDrop = gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
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
	if(gMetrics.tractorWheels != 4 || gMetrics.trailerWheels != 4 ||
		gMetrics.tractorConstraints != 1 ||
		gMetrics.trailerConstraints != 1)
		return "vehicle_topology";
	if(gMetrics.jointConfigured != 1)
		return "joint_configuration";
	if((coupled && (gMetrics.jointPresentDuringRun != 1 ||
			gMetrics.maxJointAnchorError >= 0.5f ||
			gMetrics.trailerForwardDisplacement <= 1.0f)) ||
		(!coupled && (gMetrics.jointPresentDuringRun != 0 ||
			gMetrics.trailerForwardDisplacement >= 1.0f)))
		return "coupling_response";
	if(!commandsCovered)
		return "command_coverage";
	if(gMetrics.tractorRoadHitFrames <= gMetrics.completedFrames / 2 ||
		gMetrics.trailerRoadHitFrames <= gMetrics.completedFrames / 2)
		return "road_query_coverage";
	if(!gMetrics.tractorDriveFrames || !gMetrics.tractorBrakeFrames ||
		gMetrics.trailerDriveFrames || !gMetrics.trailerBrakeFrames)
		return "actuation_coverage";
	if(!gMetrics.tractorActiveRows || !gMetrics.trailerActiveRows)
		return "constraint_activity";
	if(!gMetrics.accelerationIntentFrames)
		return "sticky_reset_intent";
	if(!gMetrics.clutchEngagedFrames || gMetrics.maxEngineSpeed <= 1.0f)
		return "engine_clutch_activity";
	if(gMetrics.initialBrakeMaxSpeed >= 1.0f)
		return "initial_brake_drift";
	if(gMetrics.throttleMaxSpeed <= 1.0f)
		return "throttle_response";
	if(brakeDrop <= 0.5f)
		return "brake_response";
	if(gMetrics.tractorForwardDisplacement <= 1.0f)
		return "tractor_motion";
	if(gMetrics.minTractorHeight <= -1.0f ||
		gMetrics.maxTractorHeight >= 5.0f ||
		gMetrics.minTrailerHeight <= -1.0f ||
		gMetrics.maxTrailerHeight >= 5.0f)
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
	const PxReal brakeMinimum =
		gMetrics.brakePhaseSeen ? gMetrics.minBrakeSpeed : 0.0f;
	const PxReal brakeDrop = gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetVehicleTruck "
		"solver=%s case=%s execution=%s frames=%u completedFrames=%u "
		"commands=%u tractorWheels=%u trailerWheels=%u "
		"tractorConstraints=%u trailerConstraints=%u "
		"jointConfigured=%u jointPresent=%u "
		"tractorRoadHitFrames=%u trailerRoadHitFrames=%u "
		"tractorRoadHitSamples=%u trailerRoadHitSamples=%u "
		"tractorDriveFrames=%u tractorBrakeFrames=%u "
		"trailerDriveFrames=%u trailerBrakeFrames=%u "
		"tractorActiveRows=%u trailerActiveRows=%u "
		"accelerationIntentFrames=%u clutchEngagedFrames=%u "
		"command0Frames=%u command1Frames=%u command2Frames=%u "
		"command3Frames=%u maxJointAnchorError=%.9g "
		"finalJointAnchorError=%.9g jointForceFrames=%u "
		"maxJointLinearForce=%.9g maxJointAngularForce=%.9g "
		"maxEngineSpeed=%.9g maxClutchSlip=%.9g "
		"initialBrakeMaxSpeed=%.9g throttleMaxSpeed=%.9g "
		"brakeStartSpeed=%.9g minBrakeSpeed=%.9g brakeDrop=%.9g "
		"tractorForwardDisplacement=%.9g trailerForwardDisplacement=%.9g "
		"finalActorSeparation=%.9g minTractorHeight=%.9g "
		"maxTractorHeight=%.9g minTrailerHeight=%.9g "
		"maxTrailerHeight=%.9g nonFinite=%u drivetrainNonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0 "
		"status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		unsigned(gHeadlessOptions.frames), unsigned(gMetrics.completedFrames),
		unsigned(gCommandProgress), unsigned(gMetrics.tractorWheels),
		unsigned(gMetrics.trailerWheels),
		unsigned(gMetrics.tractorConstraints),
		unsigned(gMetrics.trailerConstraints),
		unsigned(gMetrics.jointConfigured),
		unsigned(gMetrics.jointPresentDuringRun),
		unsigned(gMetrics.tractorRoadHitFrames),
		unsigned(gMetrics.trailerRoadHitFrames),
		unsigned(gMetrics.tractorRoadHitSamples),
		unsigned(gMetrics.trailerRoadHitSamples),
		unsigned(gMetrics.tractorDriveFrames),
		unsigned(gMetrics.tractorBrakeFrames),
		unsigned(gMetrics.trailerDriveFrames),
		unsigned(gMetrics.trailerBrakeFrames),
		unsigned(gMetrics.tractorActiveRows),
		unsigned(gMetrics.trailerActiveRows),
		unsigned(gMetrics.accelerationIntentFrames),
		unsigned(gMetrics.clutchEngagedFrames),
		unsigned(gMetrics.commandFrames[0]),
		unsigned(gMetrics.commandFrames[1]),
		unsigned(gMetrics.commandFrames[2]),
		unsigned(gMetrics.commandFrames[3]),
		double(gMetrics.maxJointAnchorError),
		double(gMetrics.finalJointAnchorError),
		unsigned(gMetrics.jointForceFrames),
		double(gMetrics.maxJointLinearForce),
		double(gMetrics.maxJointAngularForce),
		double(gMetrics.maxEngineSpeed), double(gMetrics.maxClutchSlip),
		double(gMetrics.initialBrakeMaxSpeed),
		double(gMetrics.throttleMaxSpeed),
		double(gMetrics.brakeStartSpeed), double(brakeMinimum),
		double(brakeDrop), double(gMetrics.tractorForwardDisplacement),
		double(gMetrics.trailerForwardDisplacement),
		double(gMetrics.finalActorSeparation),
		double(gMetrics.minTractorHeight),
		double(gMetrics.maxTractorHeight),
		double(gMetrics.minTrailerHeight),
		double(gMetrics.maxTrailerHeight),
		unsigned(gMetrics.nonFinite),
		unsigned(gMetrics.drivetrainNonFinite),
		unsigned(gMetrics.fetchFailures),
		unsigned(gErrorCallback.getFatalCount()),
		unsigned(gMetrics.cleanupComplete),
		pass ? "PASS" : "FAIL",
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
			argc, argv, "SnippetVehicleTruck", gVehicleDataPath))
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
	DirectDrivetrainParams directDrivetrainParams;
	if(!readDirectDrivetrainParamsFromJsonFile(
		gVehicleDataPath, "DirectDrive.json",
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
			"SnippetVehicleTruck", gHeadlessOptions);
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
	renderLoop("PhysX Snippet Vehicle Truck");
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
