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
// This snippet illustrates how to implement and apply custom vehicle components.
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

// This snippet demonstrates how to modify the vehicle component pipeline to include a custom suspension model. 
// The vehicle is then a mixture of custom components and components maintained by the PhysX Vehicle SDK.
// In this instance, the custom component computes an additional sinusoidal suspension force that is applied 
// to the vehicle and complements the suspension force of a linear spring model. The combination of sinusoidal
// suspension forces entices the vehicle to perform a kind of mechanical dance.

//This snippet organises the components into four distinct groups.
//1) a base vehicle model that describes the mechanical configuration of suspensions, tires, wheels and an 
//   associated rigid body.
//2) a direct drive drivetrain model that forwards input controls to wheel torques and steer angles.
//3) a physx integration model that provides a representation of the vehicle in an associated physx scene.
//4) a custom suspension model

// It is a good idea to record and playback with pvd (PhysX Visual Debugger).
// ****************************************************************************

#include <ctype.h>

#include "PxPhysicsAPI.h"

#include "../snippetvehiclecommon/serialization/BaseSerialization.h"
#include "../snippetvehiclecommon/serialization/DirectDrivetrainSerialization.h"
#include "../snippetvehiclecommon/SnippetVehicleHelpers.h"

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"

#include "CustomSuspension.h"

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

//The vehicle with the custom suspension component
CustomSuspensionVehicle gVehicle;

//Vehicle simulation needs a simulation context
//to store global parameters of the simulation such as 
//gravitational acceleration.
PxVehiclePhysXSimulationContext gVehicleSimulationContext;

//Gravitational acceleration
const PxVec3 gGravity(0.0f, -9.81f, 0.0f);

//The timestep of the simulation
const PxReal gTimestep = 0.016667f;

//The mapping between PxMaterial and friction.
PxVehiclePhysXMaterialFriction gPhysXMaterialFrictions[16];
PxU32 gNbPhysXMaterialFrictions = 0;
PxReal gPhysXDefaultMaterialFriction = 1.0f;

//Give the vehicle a name so it can be identified in PVD.
const char gVehicleName[] = "customsuspension";

//A ground plane to drive on.
PxRigidStatic*	gGroundPlane = NULL;

struct CustomSuspensionMetrics
{
	PxU32 initialized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 suspensionNonFinite;
	PxU32 wheelCount;
	PxU32 constraintCount;
	PxU32 roadHitFrames;
	PxU32 roadHitSamples;
	PxU32 activeConstraintFrames;
	PxU32 activeConstraintRows;
	PxU32 sleepingFrames;
	PxU32 cleanupComplete;
	PxReal minHeight;
	PxReal maxHeight;
	PxReal maxLinearSpeed;
	PxReal maxAngularSpeed;
	PxReal minUpY;
	PxReal minJounce;
	PxReal maxJounce;
	PxReal maxJounceSpeed;
	PxReal maxSuspensionForce;
	PxReal maxTheta;
	bool solverReadbackMatched;

	CustomSuspensionMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0), nonFinite(0),
	  suspensionNonFinite(0), wheelCount(0), constraintCount(0),
	  roadHitFrames(0), roadHitSamples(0), activeConstraintFrames(0),
	  activeConstraintRows(0), sleepingFrames(0), cleanupComplete(0),
	  minHeight(FLT_MAX),
	  maxHeight(-FLT_MAX), maxLinearSpeed(0.0f), maxAngularSpeed(0.0f),
	  minUpY(1.0f), minJounce(FLT_MAX), maxJounce(-FLT_MAX),
	  maxJounceSpeed(0.0f), maxSuspensionForce(0.0f), maxTheta(0.0f),
	  solverReadbackMatched(false)
	{
	}
};

CustomSuspensionMetrics gMetrics;

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
	readDirectDrivetrainParamsFromJsonFile(gVehicleDataPath, "DirectDrive.json", 
		gVehicle.mBaseParams.axleDescription, gVehicle.mDirectDriveParams);

	//Set the states to default.
	if (!gVehicle.initialize(*gPhysics, PxCookingParams(PxTolerancesScale()), *gMaterial))
	{
		return false;
	}
	gVehicleInitialized = true;

	gVehicle.mTransmissionCommandState.gear = PxVehicleDirectDriveTransmissionCommandState::eNEUTRAL;

	//Apply a start pose to the physx actor and add it to the physx scene.
	PxTransform pose(PxVec3(-5.0f, 0.5f, 0.0f), PxQuat(PxIdentity));
	gVehicle.setUpActor(*gScene, pose, gVehicleName);

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
	gMetrics = CustomSuspensionMetrics();
	if(!initPhysX(headless))
		return false;
	if(!initGroundPlane())
		return false;
	initMaterialFrictionTable();
	if (!initVehicles())
		return false;
	if(headless && gHeadlessOptions.caseName == "zero-amplitude")
	{
		for(PxU32 i = 0; i < 4; ++i)
			gVehicle.mCustomSuspensionParams[i].amplitude = 0.0f;
	}
	resetCustomSuspensionDiagnostics();
	gMetrics.initialized = 1;
	gMetrics.wheelCount = gVehicle.mBaseParams.axleDescription.nbWheels;
	for(PxU32 i = 0;
		i < PxVehiclePhysXConstraintLimits::eNB_CONSTRAINTS_PER_VEHICLE; ++i)
	{
		if(gVehicle.mPhysXState.physxConstraints.constraints[i])
			gMetrics.constraintCount++;
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
		!gVehicleInitialized && !gGroundPlane && !gMaterial && !gScene &&
		!gDispatcher && !gPhysics && !gPvd && !gFoundation ? 1u : 0u;
	printf("SnippetVehicleCustomSuspension done.\n");
}

static void collectCustomSuspensionMetrics()
{
	const PxVehicleAxleDescription& axle =
		gVehicle.mBaseParams.axleDescription;
	PxU32 roadHits = 0;
	PxU32 activeRows = 0;
	for(PxU32 i = 0; i < axle.nbWheels; ++i)
	{
		const PxU32 wheelId = axle.wheelIdsInAxleOrder[i];
		const PxVehicleRoadGeometryState& road =
			gVehicle.mBaseState.roadGeomStates[wheelId];
		const PxVehicleSuspensionState& suspension =
			gVehicle.mBaseState.suspensionStates[wheelId];
		const PxVehicleSuspensionForce& force =
			gVehicle.mBaseState.suspensionForces[wheelId];
		const PxVehiclePhysXConstraintState& constraint =
			gVehicle.mPhysXState.physxConstraints.constraintStates[wheelId];
		const PxReal theta = gVehicle.mCustomSuspensionStates[wheelId].theta;
		if(road.hitState)
			roadHits++;
		if(constraint.suspActiveStatus)
			activeRows++;
		for(PxU32 direction = 0;
			direction < PxVehicleTireDirectionModes::eMAX_NB_PLANAR_DIRECTIONS;
			++direction)
		{
			if(constraint.tireActiveStatus[direction])
				activeRows++;
		}
		if(!PxIsFinite(suspension.jounce) ||
			!PxIsFinite(suspension.jounceSpeed) ||
			!force.force.isFinite() || !force.torque.isFinite() ||
			!PxIsFinite(theta))
		{
			gMetrics.suspensionNonFinite++;
			continue;
		}
		gMetrics.minJounce = PxMin(
			gMetrics.minJounce, suspension.jounce);
		gMetrics.maxJounce = PxMax(
			gMetrics.maxJounce, suspension.jounce);
		gMetrics.maxJounceSpeed = PxMax(
			gMetrics.maxJounceSpeed, PxAbs(suspension.jounceSpeed));
		gMetrics.maxSuspensionForce = PxMax(
			gMetrics.maxSuspensionForce, force.force.magnitude());
		gMetrics.maxTheta = PxMax(gMetrics.maxTheta, PxAbs(theta));
	}
	gMetrics.roadHitSamples += roadHits;
	if(roadHits == axle.nbWheels)
		gMetrics.roadHitFrames++;
	if(activeRows)
		gMetrics.activeConstraintFrames++;
	gMetrics.activeConstraintRows += activeRows;

	PxRigidBody* rigidBody =
		gVehicle.mPhysXState.physxActor.rigidBody;
	PxRigidDynamic* rigidDynamic =
		static_cast<PxRigidDynamic*>(rigidBody);
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
	gMetrics.maxLinearSpeed = PxMax(
		gMetrics.maxLinearSpeed, linearVelocity.magnitude());
	gMetrics.maxAngularSpeed = PxMax(
		gMetrics.maxAngularSpeed, angularVelocity.magnitude());
	gMetrics.minUpY = PxMin(
		gMetrics.minUpY, pose.q.getBasisVector1().y);
	if(rigidDynamic->isSleeping())
		gMetrics.sleepingFrames++;
}

static bool stepPhysicsInternal(PxReal timestep, bool collectMetrics)
{
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
		collectCustomSuspensionMetrics();
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
	defaults.frames = 900;
	defaults.dispatcherThreads = 4;
	defaults.dt = gTimestep;
	defaults.caseName = "custom-dance";
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	if(!gHeadlessOptions.headless)
		return true;
	if(gHeadlessOptions.caseName != "custom-dance" &&
		gHeadlessOptions.caseName != "zero-amplitude")
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
	const CustomSuspensionDiagnostics& diagnostics =
		getCustomSuspensionDiagnostics();
	const PxU32 awakeFrames =
		gMetrics.completedFrames - gMetrics.sleepingFrames;
	const PxU32 callsPerFrame = gMetrics.wheelCount * 3u;
	const PxU32 minExpectedCalls = awakeFrames * callsPerFrame;
	// The component runs during simulate(), while sleeping is sampled after
	// fetchResults().  The transition frame can therefore be counted as asleep
	// after still executing one final component update.
	const PxU32 maxExpectedCalls =
		minExpectedCalls + (gMetrics.sleepingFrames ? callsPerFrame : 0u);
	const bool callCountValid =
		callsPerFrame &&
		diagnostics.callCount >= minExpectedCalls &&
		diagnostics.callCount <= maxExpectedCalls &&
		(diagnostics.callCount % callsPerFrame) == 0;
	const bool customCase =
		gHeadlessOptions.caseName == "custom-dance";
	const bool forceActivityValid = customCase ?
		(diagnostics.nonZeroForceCount > 0 &&
			diagnostics.maxMagnitude > 1000.0f) :
		(diagnostics.nonZeroForceCount == 0 &&
			diagnostics.maxMagnitude < 1e-5f &&
			diagnostics.accumulatedMagnitude < 1e-5f);
	const bool sleepStateValid = customCase ?
		gMetrics.sleepingFrames == 0 :
		gMetrics.sleepingFrames > 0;
	return gMetrics.initialized == 1 &&
		gMetrics.solverReadbackMatched &&
		gMetrics.completedFrames == gHeadlessOptions.frames &&
		gMetrics.fetchFailures == 0 && gMetrics.nonFinite == 0 &&
		gMetrics.suspensionNonFinite == 0 &&
		gMetrics.wheelCount == 4 && gMetrics.constraintCount == 1 &&
		callCountValid &&
		diagnostics.onGroundCallCount > 0 &&
		diagnostics.nonFiniteCount == 0 && forceActivityValid &&
		sleepStateValid &&
		gMetrics.roadHitSamples > gMetrics.completedFrames &&
		gMetrics.maxSuspensionForce > 1.0f &&
		gMetrics.maxTheta > 1.0f &&
		gMetrics.minHeight > -1.0f && gMetrics.maxHeight < 10.0f &&
		gMetrics.maxLinearSpeed < 100.0f &&
		gMetrics.maxAngularSpeed < 100.0f &&
		gMetrics.minUpY >= -1.0f && gMetrics.minUpY <= 1.0f &&
		gErrorCallback.getFatalCount() == 0 &&
		gMetrics.cleanupComplete == 1;
}

static const char* failureReason()
{
	const CustomSuspensionDiagnostics& diagnostics =
		getCustomSuspensionDiagnostics();
	const PxU32 awakeFrames =
		gMetrics.completedFrames - gMetrics.sleepingFrames;
	const PxU32 callsPerFrame = gMetrics.wheelCount * 3u;
	const PxU32 minExpectedCalls = awakeFrames * callsPerFrame;
	const PxU32 maxExpectedCalls =
		minExpectedCalls + (gMetrics.sleepingFrames ? callsPerFrame : 0u);
	const bool customCase =
		gHeadlessOptions.caseName == "custom-dance";
	if(gMetrics.initialized != 1)
		return "initialization";
	if(!gMetrics.solverReadbackMatched)
		return "solver_readback";
	if(gMetrics.completedFrames != gHeadlessOptions.frames)
		return "frame_count";
	if(gMetrics.fetchFailures)
		return "fetch_failure";
	if(gMetrics.nonFinite)
		return "non_finite";
	if(gMetrics.suspensionNonFinite || diagnostics.nonFiniteCount)
		return "custom_suspension_non_finite";
	if(gMetrics.wheelCount != 4 || gMetrics.constraintCount != 1)
		return "vehicle_topology";
	if(!callsPerFrame ||
		diagnostics.callCount < minExpectedCalls ||
		diagnostics.callCount > maxExpectedCalls ||
		(diagnostics.callCount % callsPerFrame) != 0)
		return "custom_component_call_count";
	if(!diagnostics.onGroundCallCount)
		return "custom_component_ground_input";
	if(customCase &&
		(!diagnostics.nonZeroForceCount ||
			diagnostics.maxMagnitude <= 1000.0f))
	{
		return "custom_force_activity";
	}
	if(!customCase &&
		(diagnostics.nonZeroForceCount ||
			diagnostics.maxMagnitude >= 1e-5f ||
			diagnostics.accumulatedMagnitude >= 1e-5f))
	{
		return "zero_amplitude_control";
	}
	if((customCase && gMetrics.sleepingFrames) ||
		(!customCase && !gMetrics.sleepingFrames))
	{
		return "sleep_state_contrast";
	}
	if(gMetrics.roadHitSamples <= gMetrics.completedFrames)
		return "road_query_coverage";
	if(gMetrics.maxSuspensionForce <= 1.0f ||
		gMetrics.maxTheta <= 1.0f)
	{
		return "suspension_state_activity";
	}
	if(gMetrics.minHeight <= -1.0f || gMetrics.maxHeight >= 10.0f)
		return "vertical_bounds";
	if(gMetrics.maxLinearSpeed >= 100.0f ||
		gMetrics.maxAngularSpeed >= 100.0f)
	{
		return "velocity_bounds";
	}
	if(gMetrics.minUpY < -1.0f || gMetrics.minUpY > 1.0f)
		return "orientation_bounds";
	if(gErrorCallback.getFatalCount())
		return "physx_error";
	if(gMetrics.cleanupComplete != 1)
		return "cleanup";
	return "none";
}

static void printHeadlessResult()
{
	const bool pass = headlessPassed();
	const CustomSuspensionDiagnostics& diagnostics =
		getCustomSuspensionDiagnostics();
	const PxReal minimumHeight =
		gMetrics.completedFrames ? gMetrics.minHeight : 0.0f;
	const PxReal maximumHeight =
		gMetrics.completedFrames ? gMetrics.maxHeight : 0.0f;
	const PxReal minimumJounce =
		gMetrics.completedFrames ? gMetrics.minJounce : 0.0f;
	const PxReal maximumJounce =
		gMetrics.completedFrames ? gMetrics.maxJounce : 0.0f;
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetVehicleCustomSuspension "
		"solver=%s case=%s execution=%s frames=%u completedFrames=%u "
		"wheels=%u constraints=%u roadHitFrames=%u roadHitSamples=%u "
		"activeConstraintFrames=%u activeConstraintRows=%u sleepingFrames=%u "
		"customCalls=%u customOnGroundCalls=%u "
		"customNonZeroForceCalls=%u customMaxMagnitude=%.9g "
		"customAccumulatedMagnitude=%.9g customNonFinite=%u "
		"minJounce=%.9g maxJounce=%.9g maxJounceSpeed=%.9g "
		"maxSuspensionForce=%.9g maxTheta=%.9g "
		"minHeight=%.9g maxHeight=%.9g heightSpan=%.9g "
		"maxLinearSpeed=%.9g maxAngularSpeed=%.9g minUpY=%.9g "
		"nonFinite=%u suspensionNonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0 "
		"status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		unsigned(gHeadlessOptions.frames), unsigned(gMetrics.completedFrames),
		unsigned(gMetrics.wheelCount), unsigned(gMetrics.constraintCount),
		unsigned(gMetrics.roadHitFrames),
		unsigned(gMetrics.roadHitSamples),
		unsigned(gMetrics.activeConstraintFrames),
		unsigned(gMetrics.activeConstraintRows),
		unsigned(gMetrics.sleepingFrames),
		unsigned(diagnostics.callCount),
		unsigned(diagnostics.onGroundCallCount),
		unsigned(diagnostics.nonZeroForceCount),
		double(diagnostics.maxMagnitude),
		double(diagnostics.accumulatedMagnitude),
		unsigned(diagnostics.nonFiniteCount), double(minimumJounce),
		double(maximumJounce), double(gMetrics.maxJounceSpeed),
		double(gMetrics.maxSuspensionForce), double(gMetrics.maxTheta),
		double(minimumHeight), double(maximumHeight),
		double(maximumHeight - minimumHeight),
		double(gMetrics.maxLinearSpeed), double(gMetrics.maxAngularSpeed),
		double(gMetrics.minUpY), unsigned(gMetrics.nonFinite),
		unsigned(gMetrics.suspensionNonFinite),
		unsigned(gMetrics.fetchFailures),
		unsigned(gErrorCallback.getFatalCount()),
		unsigned(gMetrics.cleanupComplete), pass ? "PASS" : "FAIL",
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
			argc, argv, "SnippetVehicleCustomSuspension", gVehicleDataPath))
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
			"SnippetVehicleCustomSuspension", gHeadlessOptions);
		gErrorCallback.reset();
		bool runOk = initPhysicsInternal(true);
		while(runOk &&
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
	renderLoop("PhysX Snippet Vehicle Custom Suspension");
#else
	if (initPhysics())
	{
		PxReal accumulatedTime = 0.0f;
		while (accumulatedTime < 15.0f)
		{
			stepPhysics();
			accumulatedTime += gTimestep;
		}
		cleanupPhysics();
	}
#endif

	return 0;
}
