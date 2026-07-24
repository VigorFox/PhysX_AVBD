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
// This snippet illustrates how to change the tire model using custom vehicle components.
//

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

// Custom combinations of parameter, state and component allow different behaviours to be simulated with 
// different simulation fidelities. In this example, the change has been localised to the tire component 
// to replace the low fidelity model with the high fidelty .
// The low and high fidelity components consume the same state data (tire slip, load, friction) and
// output the same state data for the tire forces. Again, the change has been localised to the component
// that converts slip angle to tire force and the parameterisation that governs the conversion.

// This snippet demonstrates how to modify the vehicle component pipeline to include a custom tire model. 
// The vehicle is then a mixture of custom components and components maintained by the PhysX Vehicle SDK.
// In this instance, the custom component implements a version of the Magic Formula Tire Model (also 
// known as Pacejka) to compute the tire forces. The Magic Formula Tire Model uses measurements on
// real tires to infer the parameters of predefined formulas such that the resulting graphs will fit closely
// to the measurement points.

//This snippet organises the components into four distinct groups.
//1) a base vehicle model that describes the mechanical configuration of suspensions, tires, wheels and an 
//   associated rigid body.
//2) a direct drive drivetrain model that forwards input controls to wheel torques and steer angles.
//3) a physx integration model that provides a representation of the vehicle in an associated physx scene.
//4) a custom tire model

// It is a good idea to record and playback with pvd (PhysX Visual Debugger).
// ****************************************************************************

#include <ctype.h>

#include "PxPhysicsAPI.h"

#include "../snippetvehiclecommon/serialization/BaseSerialization.h"
#include "../snippetvehiclecommon/serialization/DirectDrivetrainSerialization.h"
#include "../snippetvehiclecommon/SnippetVehicleHelpers.h"

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"

#include "CustomTireVehicle.h"

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
PxPhysics*				gPhysics = NULL;
PxDefaultCpuDispatcher*	gDispatcher = NULL;
PxScene*				gScene = NULL;
PxMaterial*				gMaterial = NULL;
PxPvd*                  gPvd = NULL;
Snippets::HeadlessOptions gHeadlessOptions;
bool					gVehicleExtensionInitialized = false;
bool					gVehicleInitialized = false;

//The path to the vehicle json files to be loaded.
const char* gVehicleDataPath = NULL;

//The vehicle with the custom tire component
CustomTireVehicle gVehicle;

//Vehicle simulation needs a simulation context
//to store global parameters of the simulation such as 
//gravitational acceleration.
PxVehiclePhysXSimulationContext gVehicleSimulationContext;

//If sweeps are used for wheel vs. ground collision detection, then
//a cylinder mesh (with unit size) needs to be provided to do the sweep.
PxConvexMesh* gUnitCylinderSweepMesh = NULL;

//Gravitational acceleration
const PxVec3 gGravity(0.0f, -9.81f, 0.0f);

//The timestep of the simulation
const PxReal gTimestep = 1.0f / 60.0f;

//The timestep for the tire model simulation. For the Magic Formula
//Tire Model a high update rate is recommended, like 1 kHz.
const PxReal gTireModelTimestep = 1.0f / 1000.0f;

//The mapping between PxMaterial and friction.
PxVehiclePhysXMaterialFriction gPhysXMaterialFrictions[16];
PxU32 gNbPhysXMaterialFrictions = 0;
PxReal gPhysXDefaultMaterialFriction = 1.0f;

//Give the vehicle a name so it can be identified in PVD.
const char gVehicleName[] = "CustomTireVehicle";

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
	{0.0f, 0.0f, 0.0f, 2.0f},		//fall on ground and come to rest for 2 seconds
    {0.0f, 0.5f, 0.0f, 5.0f},		//throttle for 5 seconds
    {0.5f, 0.0f, 0.0f, 5.0f},		//brake for 5 seconds
	{0.0f, 0.5f, 0.0f, 5.0f},		//throttle for 5 seconds
    {0.0f, 0.1f, 0.5f, 5.0f}		//light throttle and steer for 5 seconds.
};
const PxU32 gNbCommands = sizeof(gCommands) / sizeof(Command);
PxReal gCommandTime = 0.0f;			//Time spent on current command
PxU32 gCommandProgress = 0;			//The id of the current command.

//A ground plane to drive on.
PxRigidStatic* gGroundPlane = NULL;

struct VehicleMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 tireForceNonFinite;
	PxU32 wheelCount;
	PxU32 constraintCount;
	PxU32 roadHitFrames;
	PxU32 roadHitSamples;
	PxU32 driveFrames;
	PxU32 brakeFrames;
	PxU32 activeConstraintFrames;
	PxU32 activeConstraintRows;
	PxU32 tireForceFrames;
	PxU32 nonZeroLongForceSamples;
	PxU32 nonZeroLatForceSamples;
	PxU32 commandFrames[5];
	PxU32 cleanupComplete;
	PxReal maxTireLongForce;
	PxReal maxTireLatForce;
	PxReal settledSpeed;
	PxReal throttleMaxSpeed;
	PxReal brakeStartSpeed;
	PxReal minBrakeSpeed;
	PxReal forwardDisplacement;
	PxReal steerLateralDisplacement;
	PxReal steerHeadingChange;
	PxReal minHeight;
	PxReal maxHeight;
	PxReal maxLinearSpeed;
	PxReal maxAngularSpeed;
	PxTransform initialPose;
	PxTransform steerStartPose;
	PxReal steerStartHeading;
	bool brakePhaseSeen;
	bool steerPhaseSeen;
	bool solverReadbackMatched;

	VehicleMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0), nonFinite(0),
	  tireForceNonFinite(0), wheelCount(0), constraintCount(0),
	  roadHitFrames(0), roadHitSamples(0), driveFrames(0), brakeFrames(0),
	  activeConstraintFrames(0), activeConstraintRows(0),
	  tireForceFrames(0), nonZeroLongForceSamples(0),
	  nonZeroLatForceSamples(0), cleanupComplete(0),
	  maxTireLongForce(0.0f), maxTireLatForce(0.0f),
	  settledSpeed(0.0f), throttleMaxSpeed(0.0f),
	  brakeStartSpeed(0.0f), minBrakeSpeed(FLT_MAX),
	  forwardDisplacement(0.0f), steerLateralDisplacement(0.0f),
	  steerHeadingChange(0.0f), minHeight(FLT_MAX),
	  maxHeight(-FLT_MAX), maxLinearSpeed(0.0f), maxAngularSpeed(0.0f),
	  initialPose(PxIdentity), steerStartPose(PxIdentity),
	  steerStartHeading(0.0f), brakePhaseSeen(false),
	  steerPhaseSeen(false), solverReadbackMatched(false)
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
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = VehicleFilterShader;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	gMetrics.solverReadbackMatched =
		gScene->getSolverType() == sceneDesc.solverType;
	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (!headless && pvdClient)
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
	readDirectDrivetrainParamsFromJsonFile(gVehicleDataPath, "DirectDrive.json", 
		gVehicle.mBaseParams.axleDescription, gVehicle.mDirectDriveParams);

	if(gHeadlessOptions.headless)
		gVehicle.mUseCustomTire =
			gHeadlessOptions.caseName == "magic-formula";
	else
		gVehicle.mUseCustomTire = true;
	resetCustomTireDiagnostics();

	//Set the states to default.
	PxCookingParams cookingParams(gPhysics->getTolerancesScale());
	if (!gVehicle.initialize(*gPhysics, cookingParams, *gMaterial))
	{
		return false;
	}
	gVehicleInitialized = true;

	gVehicle.mCommandState.nbBrakes = 1;
	gVehicle.mTransmissionCommandState.gear = PxVehicleDirectDriveTransmissionCommandState::eFORWARD;

	//Apply a start pose to the physx actor and add it to the physx scene.
	PxTransform pose(PxVec3(-5.0f, 0.5f, 0.0f), PxQuat(PxIdentity));
	gVehicle.setUpActor(*gScene, pose, gVehicleName);

	//Disabling sleeping in this snippet to show stability at rest is reached independent of
	//sleeping.
	gVehicle.mPhysXState.physxActor.rigidBody->is<PxRigidDynamic>()->setSleepThreshold(0.0f);
	gVehicle.mPhysXState.physxActor.rigidBody->is<PxRigidDynamic>()->setStabilizationThreshold(0.0f);

	//Set the substep count to match the targeted tire model simulation timestep
	PxU8 substepCount = static_cast<PxU8>(gTimestep / gTireModelTimestep);
	gVehicle.mComponentSequence.setSubsteps(gVehicle.mComponentSequenceSubstepGroupHandle, substepCount);

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

	gUnitCylinderSweepMesh = PxVehicleUnitCylinderSweepMeshCreate(gVehicleSimulationContext.frame,
		*gPhysics, cookingParams);
	if (!gUnitCylinderSweepMesh)
	{
		return false;
	}
	gVehicleSimulationContext.physxUnitCylinderSweepMesh = gUnitCylinderSweepMesh;

	//Larger lateral damping factor than default to avoid drift when nearly rest
	gVehicleSimulationContext.tireStickyParams.stickyParams[PxVehicleTireDirectionModes::eLATERAL].damping = 1.0f;

	return true;
}

void cleanupVehicles()
{
	if (gUnitCylinderSweepMesh)
	{
		PxVehicleUnitCylinderSweepMeshDestroy(gUnitCylinderSweepMesh);
		gUnitCylinderSweepMesh = NULL;
	}

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
		!gVehicleInitialized && !gUnitCylinderSweepMesh && !gGroundPlane &&
		!gMaterial && !gScene && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation ? 1u : 0u;
	printf("SnippetVehicleCustomTire done.\n");
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
	bool tireForceApplied = false;
	PxU32 activeRows = 0;
	for(PxU32 i = 0; i < axle.nbWheels; ++i)
	{
		const PxU32 wheelId = axle.wheelIdsInAxleOrder[i];
		if(gVehicle.mBaseState.roadGeomStates[wheelId].hitState)
			roadHits++;
		const PxVehicleWheelActuationState& actuation =
			gVehicle.mBaseState.actuationStates[wheelId];
		driveApplied = driveApplied || actuation.isDriveApplied;
		brakeApplied = brakeApplied || actuation.isBrakeApplied;
		const PxVehiclePhysXConstraintState& constraint =
			gVehicle.mPhysXState.physxConstraints.constraintStates[wheelId];
		if(constraint.suspActiveStatus)
			activeRows++;
		for(PxU32 direction = 0;
			direction < PxVehicleTireDirectionModes::eMAX_NB_PLANAR_DIRECTIONS;
			++direction)
		{
			if(constraint.tireActiveStatus[direction])
				activeRows++;
		}
		const PxVehicleTireForce& force =
			gVehicle.mBaseState.tireForces[wheelId];
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
		gMetrics.roadHitFrames++;
	if(driveApplied)
		gMetrics.driveFrames++;
	if(brakeApplied)
		gMetrics.brakeFrames++;
	if(activeRows)
		gMetrics.activeConstraintFrames++;
	gMetrics.activeConstraintRows += activeRows;
	if(tireForceApplied)
		gMetrics.tireForceFrames++;
	if(commandIndex < gNbCommands)
		gMetrics.commandFrames[commandIndex]++;

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
	const PxReal linearSpeed = linearVelocity.magnitude();
	gMetrics.minHeight = PxMin(gMetrics.minHeight, pose.p.y);
	gMetrics.maxHeight = PxMax(gMetrics.maxHeight, pose.p.y);
	gMetrics.maxLinearSpeed =
		PxMax(gMetrics.maxLinearSpeed, linearSpeed);
	gMetrics.maxAngularSpeed =
		PxMax(gMetrics.maxAngularSpeed, angularVelocity.magnitude());
	gMetrics.forwardDisplacement = PxMax(
		gMetrics.forwardDisplacement,
		pose.p.z - gMetrics.initialPose.p.z);
	if(commandIndex == 0)
		gMetrics.settledSpeed = linearSpeed;
	if(commandIndex == 1 || commandIndex == 3 || commandIndex == 4)
		gMetrics.throttleMaxSpeed =
			PxMax(gMetrics.throttleMaxSpeed, linearSpeed);
	if(commandIndex == 2)
	{
		if(!gMetrics.brakePhaseSeen)
		{
			gMetrics.brakePhaseSeen = true;
			gMetrics.brakeStartSpeed = linearSpeed;
		}
		gMetrics.minBrakeSpeed =
			PxMin(gMetrics.minBrakeSpeed, linearSpeed);
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
	if (gNbCommands == gCommandProgress)
		return true;

	//Apply the brake, throttle and steer to the command state of the direct drive vehicle.
	const PxU32 commandIndex = gCommandProgress;
	const Command& command = gCommands[commandIndex];
	gVehicle.mCommandState.brakes[0] = command.brake;
	gVehicle.mCommandState.throttle = command.throttle;
	gVehicle.mCommandState.steer = command.steer;

	//Forward integrate the vehicle by a single timestep.
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
	stepPhysicsInternal(gTimestep, false);
}

static bool parseHeadlessOptions(
	int argc, const char*const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.frames = 1400;
	defaults.dispatcherThreads = 4;
	defaults.dt = gTimestep;
	defaults.caseName = "magic-formula";
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	if(!gHeadlessOptions.headless)
		return true;
	if(gHeadlessOptions.caseName != "magic-formula" &&
		gHeadlessOptions.caseName != "standard-tire")
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

static bool customDiagnosticsValid()
{
	const CustomTireDiagnostics& diagnostics =
		getCustomTireDiagnostics();
	const bool custom =
		gHeadlessOptions.caseName == "magic-formula";
	const PxU32 expectedCalls =
		gMetrics.completedFrames * gMetrics.wheelCount * 16u;
	if(custom)
	{
		return diagnostics.gripCalls == expectedCalls &&
			diagnostics.slipCalls == expectedCalls &&
			diagnostics.forceCalls == expectedCalls &&
			diagnostics.onGroundCalls > 0 &&
			diagnostics.nonZeroLongForceCalls > 0 &&
			diagnostics.nonZeroLatForceCalls > 0 &&
			diagnostics.maxLongForce > 1.0f &&
			diagnostics.maxLatForce > 1.0f &&
			diagnostics.nonFiniteCount == 0;
	}
	return diagnostics.gripCalls == 0 &&
		diagnostics.slipCalls == 0 &&
		diagnostics.forceCalls == 0 &&
		diagnostics.nonFiniteCount == 0;
}

static bool headlessPassed()
{
	const PxReal brakeDrop = gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	bool commandsCovered = true;
	for(PxU32 i = 0; i < gNbCommands; ++i)
		commandsCovered = commandsCovered && gMetrics.commandFrames[i] > 0;
	return gMetrics.initialized == 1 &&
		gMetrics.solverReadbackMatched &&
		gCommandProgress == gNbCommands &&
		gMetrics.completedFrames <= gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 && gMetrics.nonFinite == 0 &&
		gMetrics.tireForceNonFinite == 0 &&
		gMetrics.wheelCount == 4 && gMetrics.constraintCount == 1 &&
		commandsCovered && customDiagnosticsValid() &&
		gMetrics.roadHitFrames > gMetrics.completedFrames / 2 &&
		gMetrics.driveFrames > 0 && gMetrics.brakeFrames > 0 &&
		gMetrics.activeConstraintFrames > 0 &&
		gMetrics.tireForceFrames > 0 &&
		gMetrics.nonZeroLongForceSamples > 0 &&
		gMetrics.nonZeroLatForceSamples > 0 &&
		gMetrics.maxTireLongForce > 1.0f &&
		gMetrics.maxTireLatForce > 1.0f &&
		gMetrics.settledSpeed < 1.0f &&
		gMetrics.throttleMaxSpeed > 1.0f &&
		brakeDrop > 0.5f &&
		gMetrics.forwardDisplacement > 1.0f &&
		PxAbs(gMetrics.steerLateralDisplacement) > 0.1f &&
		gMetrics.steerHeadingChange > 0.02f &&
		gMetrics.minHeight > -1.0f && gMetrics.maxHeight < 5.0f &&
		gMetrics.maxLinearSpeed < 100.0f &&
		gMetrics.maxAngularSpeed < 100.0f &&
		gErrorCallback.getFatalCount() == 0 &&
		gMetrics.cleanupComplete == 1;
}

static const char* failureReason()
{
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
	if(gMetrics.nonFinite || gMetrics.tireForceNonFinite)
		return "non_finite";
	if(gMetrics.wheelCount != 4 || gMetrics.constraintCount != 1)
		return "vehicle_topology";
	if(!commandsCovered)
		return "command_coverage";
	if(!customDiagnosticsValid())
		return "tire_component_activity";
	if(gMetrics.roadHitFrames <= gMetrics.completedFrames / 2)
		return "road_query_coverage";
	if(!gMetrics.driveFrames || !gMetrics.brakeFrames)
		return "actuation_coverage";
	if(!gMetrics.activeConstraintFrames)
		return "constraint_activity";
	if(!gMetrics.tireForceFrames ||
		!gMetrics.nonZeroLongForceSamples ||
		!gMetrics.nonZeroLatForceSamples ||
		gMetrics.maxTireLongForce <= 1.0f ||
		gMetrics.maxTireLatForce <= 1.0f)
		return "tire_force_activity";
	if(gMetrics.settledSpeed >= 1.0f)
		return "initial_settle";
	if(gMetrics.throttleMaxSpeed <= 1.0f)
		return "throttle_response";
	if(brakeDrop <= 0.5f)
		return "brake_response";
	if(gMetrics.forwardDisplacement <= 1.0f)
		return "forward_motion";
	if(PxAbs(gMetrics.steerLateralDisplacement) <= 0.1f ||
		gMetrics.steerHeadingChange <= 0.02f)
		return "steer_response";
	if(gMetrics.minHeight <= -1.0f || gMetrics.maxHeight >= 5.0f)
		return "vertical_bounds";
	if(gMetrics.maxLinearSpeed >= 100.0f ||
		gMetrics.maxAngularSpeed >= 100.0f)
		return "velocity_bounds";
	if(gErrorCallback.getFatalCount())
		return "physx_error";
	if(gMetrics.cleanupComplete != 1)
		return "cleanup";
	return "none";
}

static void printHeadlessResult()
{
	const CustomTireDiagnostics& diagnostics =
		getCustomTireDiagnostics();
	const bool pass = headlessPassed();
	const PxReal brakeMinimum =
		gMetrics.brakePhaseSeen ? gMetrics.minBrakeSpeed : 0.0f;
	const PxReal brakeDrop = gMetrics.brakePhaseSeen ?
		gMetrics.brakeStartSpeed - gMetrics.minBrakeSpeed : 0.0f;
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetVehicleCustomTire "
		"solver=%s case=%s execution=%s frames=%u completedFrames=%u "
		"commands=%u wheels=%u constraints=%u "
		"roadHitFrames=%u roadHitSamples=%u driveFrames=%u brakeFrames=%u "
		"activeConstraintFrames=%u activeConstraintRows=%u "
		"tireForceFrames=%u nonZeroLongForceSamples=%u "
		"nonZeroLatForceSamples=%u gripCalls=%u onGroundCalls=%u "
		"slipCalls=%u forceCalls=%u customNonZeroLongCalls=%u "
		"customNonZeroLatCalls=%u customNonFinite=%u "
		"customMaxLongForce=%.9g customMaxLatForce=%.9g "
		"customMaxWheelTorque=%.9g maxTireLongForce=%.9g "
		"maxTireLatForce=%.9g command0Frames=%u command1Frames=%u "
		"command2Frames=%u command3Frames=%u command4Frames=%u "
		"settledSpeed=%.9g throttleMaxSpeed=%.9g "
		"brakeStartSpeed=%.9g minBrakeSpeed=%.9g brakeDrop=%.9g "
		"forwardDisplacement=%.9g steerLateralDisplacement=%.9g "
		"steerHeadingChange=%.9g minHeight=%.9g maxHeight=%.9g "
		"maxLinearSpeed=%.9g maxAngularSpeed=%.9g nonFinite=%u "
		"tireForceNonFinite=%u fetchFailures=%u fatalErrors=%u "
		"cleanupComplete=%u pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		unsigned(gHeadlessOptions.frames), unsigned(gMetrics.completedFrames),
		unsigned(gCommandProgress), unsigned(gMetrics.wheelCount),
		unsigned(gMetrics.constraintCount), unsigned(gMetrics.roadHitFrames),
		unsigned(gMetrics.roadHitSamples), unsigned(gMetrics.driveFrames),
		unsigned(gMetrics.brakeFrames),
		unsigned(gMetrics.activeConstraintFrames),
		unsigned(gMetrics.activeConstraintRows),
		unsigned(gMetrics.tireForceFrames),
		unsigned(gMetrics.nonZeroLongForceSamples),
		unsigned(gMetrics.nonZeroLatForceSamples),
		unsigned(diagnostics.gripCalls),
		unsigned(diagnostics.onGroundCalls),
		unsigned(diagnostics.slipCalls),
		unsigned(diagnostics.forceCalls),
		unsigned(diagnostics.nonZeroLongForceCalls),
		unsigned(diagnostics.nonZeroLatForceCalls),
		unsigned(diagnostics.nonFiniteCount),
		double(diagnostics.maxLongForce),
		double(diagnostics.maxLatForce),
		double(diagnostics.maxWheelTorque),
		double(gMetrics.maxTireLongForce),
		double(gMetrics.maxTireLatForce),
		unsigned(gMetrics.commandFrames[0]),
		unsigned(gMetrics.commandFrames[1]),
		unsigned(gMetrics.commandFrames[2]),
		unsigned(gMetrics.commandFrames[3]),
		unsigned(gMetrics.commandFrames[4]),
		double(gMetrics.settledSpeed),
		double(gMetrics.throttleMaxSpeed),
		double(gMetrics.brakeStartSpeed), double(brakeMinimum),
		double(brakeDrop), double(gMetrics.forwardDisplacement),
		double(gMetrics.steerLateralDisplacement),
		double(gMetrics.steerHeadingChange), double(gMetrics.minHeight),
		double(gMetrics.maxHeight), double(gMetrics.maxLinearSpeed),
		double(gMetrics.maxAngularSpeed), unsigned(gMetrics.nonFinite),
		unsigned(gMetrics.tireForceNonFinite),
		unsigned(gMetrics.fetchFailures),
		unsigned(gErrorCallback.getFatalCount()),
		unsigned(gMetrics.cleanupComplete),
		pass ? "PASS" : "FAIL",
		pass ? "none" : failureReason());
}

int snippetMain(int argc, const char *const* argv)
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
			argc, argv, "SnippetVehicleCustomTire", gVehicleDataPath))
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
			"SnippetVehicleCustomTire", gHeadlessOptions);
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
	renderLoop("PhysX Snippet Vehicle Custom Tire");
#else
	if(initPhysics())
	{
		while(gCommandProgress != gNbCommands)
		{
			stepPhysics();
		}
		cleanupPhysics();
	}
#endif

	return 0;
}
