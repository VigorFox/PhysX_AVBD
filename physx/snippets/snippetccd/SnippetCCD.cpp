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
// This snippet illustrates how to use different types of CCD methods, 
// including linear, raycast and speculative CCD.
//
// The scene has two parts:
// - a simple box stack and a fast moving sphere. Without (linear) CCD the
// sphere goes through the box stack. With CCD the sphere hits the stack and
// the behavior is more convincing.
// - a simple rotating plank (fixed in space except for one rotation axis)
// that collides with a falling box. Wihout (angular) CCD the collision is missed.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "extensions/PxRaycastCCD.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include <cstdio>
#include <string>
#ifdef RENDER_SNIPPET
	#include "../snippetrender/SnippetRender.h"
#endif

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;
static RaycastCCDManager*		gRaycastCCD	= NULL;
static PxReal					stackZ = 10.0f;

enum HeadlessCCDCase
{
	HEADLESS_LINEAR_CCD,
	HEADLESS_SPECULATIVE_CCD,
	HEADLESS_NO_CCD,
	HEADLESS_ANGULAR_NO_CCD,
	HEADLESS_ANGULAR_SPECULATIVE_CCD,
	HEADLESS_FULL_CCD,
	HEADLESS_RAYCAST_CCD,
	HEADLESS_DYNAMIC_NO_CCD,
	HEADLESS_DYNAMIC_LINEAR_CCD,
	HEADLESS_RAYCAST_DYNAMIC_CCD
};

struct HeadlessCCDMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 crossedWall;
	PxU32 responseSamples;
	PxU32 ccdPairs;
	PxReal minMoverZ;
	PxReal finalMoverZ;
	PxReal finalMoverVelocityZ;
	PxReal moverMaxAngularSpeed;
	PxReal targetMaxDisplacement;
	PxReal targetMaxLinearSpeed;
	PxReal targetMaxAngularSpeed;
	PxU32 sceneLinearCCD;
	PxU32 sourceLinearCCD;
	PxU32 sourceSpeculativeCCD;
	PxU32 targetLinearCCD;
	PxU32 targetSpeculativeCCD;
	PxU32 targetDynamic;
	PxU32 raycastRegistrations;
	PxU32 raycastCorrections;
	PxU32 cleanupComplete;

	HeadlessCCDMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0), crossedWall(0),
	  responseSamples(0), ccdPairs(0), minMoverZ(PX_MAX_F32),
	  finalMoverZ(0.0f), finalMoverVelocityZ(0.0f),
	  moverMaxAngularSpeed(0.0f), targetMaxDisplacement(0.0f),
	  targetMaxLinearSpeed(0.0f), targetMaxAngularSpeed(0.0f),
	  sceneLinearCCD(0), sourceLinearCCD(0), sourceSpeculativeCCD(0),
	  targetLinearCCD(0), targetSpeculativeCCD(0), targetDynamic(0),
	  raycastRegistrations(0), raycastCorrections(0), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static HeadlessCCDCase gHeadlessCase = HEADLESS_LINEAR_CCD;
static HeadlessCCDMetrics gHeadlessMetrics;
static PxRigidDynamic* gHeadlessMover = NULL;
static PxRigidActor* gHeadlessObstacle = NULL;
static PxRigidDynamic* gHeadlessResponseTarget = NULL;
static PxVec3 gHeadlessTargetInitialPosition(0.0f);

enum CCDAlgorithm
{
	// Uses linear CCD algorithm 
	LINEAR_CCD,

	// Uses speculative/angular CCD algorithm
	SPECULATIVE_CCD,

	// Uses linear & angular CCD at the same time
	FULL_CCD,

	// Uses raycast CCD algorithm 
	RAYCAST_CCD,

	// Switch to NO_CCD to see the sphere go through the box stack without CCD.
	NO_CCD,

	// Number of CCD algorithms used in this snippet
	CCD_COUNT
};

static bool			gPause			= false;
static bool			gOneFrame		= false;
static const PxU32	gScenarioCount	= CCD_COUNT;
static PxU32		gScenario		= 0;

static PX_FORCE_INLINE CCDAlgorithm getCCDAlgorithm()
{
	return CCDAlgorithm(gScenario);
}

static const char* getHeadlessCaseName()
{
	switch(gHeadlessCase)
	{
	case HEADLESS_LINEAR_CCD:
		return "linear";
	case HEADLESS_SPECULATIVE_CCD:
		return "speculative";
	case HEADLESS_NO_CCD:
		return "no-ccd";
	case HEADLESS_ANGULAR_NO_CCD:
		return "angular-no-ccd";
	case HEADLESS_ANGULAR_SPECULATIVE_CCD:
		return "angular-speculative";
	case HEADLESS_FULL_CCD:
		return "full";
	case HEADLESS_RAYCAST_CCD:
		return "raycast";
	case HEADLESS_DYNAMIC_NO_CCD:
		return "dynamic-no-ccd";
	case HEADLESS_DYNAMIC_LINEAR_CCD:
		return "dynamic-linear";
	case HEADLESS_RAYCAST_DYNAMIC_CCD:
		return "raycast-dynamic";
	default:
		return "unknown";
	}
}

static bool isHeadlessAngularCase()
{
	return gHeadlessCase == HEADLESS_ANGULAR_NO_CCD ||
		gHeadlessCase == HEADLESS_ANGULAR_SPECULATIVE_CCD ||
		gHeadlessCase == HEADLESS_FULL_CCD;
}

static bool isHeadlessRaycastCase()
{
	return gHeadlessCase == HEADLESS_RAYCAST_CCD ||
		gHeadlessCase == HEADLESS_RAYCAST_DYNAMIC_CCD;
}

static bool isHeadlessDynamicTargetCase()
{
	return isHeadlessAngularCase() ||
		gHeadlessCase == HEADLESS_DYNAMIC_NO_CCD ||
		gHeadlessCase == HEADLESS_DYNAMIC_LINEAR_CCD ||
		gHeadlessCase == HEADLESS_RAYCAST_DYNAMIC_CCD;
}

static bool usesHeadlessLinearCCD()
{
	return gHeadlessCase == HEADLESS_LINEAR_CCD ||
		gHeadlessCase == HEADLESS_FULL_CCD ||
		gHeadlessCase == HEADLESS_DYNAMIC_LINEAR_CCD;
}

static bool usesHeadlessSpeculativeCCD()
{
	return gHeadlessCase == HEADLESS_SPECULATIVE_CCD ||
		gHeadlessCase == HEADLESS_ANGULAR_SPECULATIVE_CCD ||
		gHeadlessCase == HEADLESS_FULL_CCD;
}

static bool parseHeadlessCase(const char* value, HeadlessCCDCase& result)
{
	if(Snippets::equalsIgnoreCase(value, "linear"))
	{
		result = HEADLESS_LINEAR_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "speculative"))
	{
		result = HEADLESS_SPECULATIVE_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "no-ccd"))
	{
		result = HEADLESS_NO_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "angular-no-ccd"))
	{
		result = HEADLESS_ANGULAR_NO_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "angular-speculative"))
	{
		result = HEADLESS_ANGULAR_SPECULATIVE_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "full"))
	{
		result = HEADLESS_FULL_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "raycast"))
	{
		result = HEADLESS_RAYCAST_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "dynamic-no-ccd"))
	{
		result = HEADLESS_DYNAMIC_NO_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "dynamic-linear"))
	{
		result = HEADLESS_DYNAMIC_LINEAR_CCD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "raycast-dynamic"))
	{
		result = HEADLESS_RAYCAST_DYNAMIC_CCD;
		return true;
	}
	return false;
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 12;
	defaults.caseName = "linear";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
	{
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") +
				(argv[i] ? argv[i] : "<null>");
			return false;
		}
	}
	if(!parseHeadlessCase(gHeadlessOptions.caseName.c_str(), gHeadlessCase))
	{
		error = "unsupported --case (expected linear, speculative, no-ccd, "
			"angular-no-ccd, angular-speculative, full, raycast, "
			"dynamic-no-ccd, dynamic-linear, or raycast-dynamic)";
		return false;
	}
	if(gHeadlessOptions.frames < 4 || gHeadlessOptions.frames > 120)
	{
		error = "--frames must be in [4, 120]";
		return false;
	}
	return true;
}

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0), bool enableLinearCCD = false, bool enableSpeculativeCCD = false)
{
	PX_ASSERT(gScene);
	PxRigidDynamic* dynamic = NULL;
	if(gScene)
	{
		dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
		dynamic->setAngularDamping(0.5f);
		dynamic->setLinearVelocity(velocity);
		gScene->addActor(*dynamic);
		if(enableLinearCCD)
			dynamic->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
		if(enableSpeculativeCCD)
			dynamic->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
	}

	return dynamic;
}

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent, bool enableLinearCCD = false, bool enableSpeculativeCCD = false)
{
	PX_ASSERT(gScene);
	if(!gScene)
		return;

	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);	
	PX_ASSERT(shape);
	if(!shape)
		return;

	for(PxU32 i=0; i<size;i++)
	{
		for(PxU32 j=0;j<size-i;j++)
		{
			const PxTransform localTm(PxVec3(PxReal(j*2) - PxReal(size-i), PxReal(i*2+1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			if(enableLinearCCD)
				body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
			if(enableSpeculativeCCD)
				body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);

			gScene->addActor(*body);
		}
	}
	shape->release();
}

static PxFilterFlags ccdFilterShader(
	PxFilterObjectAttributes attributes0,
	PxFilterData filterData0,
	PxFilterObjectAttributes attributes1,
	PxFilterData filterData1,
	PxPairFlags& pairFlags,
	const void* constantBlock,
	PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(filterData0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);

	pairFlags = PxPairFlag::eSOLVE_CONTACT | PxPairFlag::eDETECT_DISCRETE_CONTACT | PxPairFlag::eDETECT_CCD_CONTACT;

	return PxFilterFlags();
}

static bool registerForRaycastCCD(PxRigidDynamic* actor)
{
	if(!actor || !gRaycastCCD)
		return false;

	PxShape* shape = NULL;
	if(actor->getShapes(&shape, 1) != 1 || !shape)
		return false;

	// Register each object for which raycast CCD should be enabled.
	return gRaycastCCD->registerRaycastCCDObject(actor, shape);
}

static void recordHeadlessCCDFlags()
{
	if(!gScene || !gHeadlessMover)
		return;

	gHeadlessMetrics.sceneLinearCCD =
		(gScene->getFlags() & PxSceneFlag::eENABLE_CCD) ? 1u : 0u;
	const PxRigidBodyFlags sourceFlags = gHeadlessMover->getRigidBodyFlags();
	gHeadlessMetrics.sourceLinearCCD =
		(sourceFlags & PxRigidBodyFlag::eENABLE_CCD) ? 1u : 0u;
	gHeadlessMetrics.sourceSpeculativeCCD =
		(sourceFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD) ? 1u : 0u;
	if(gHeadlessResponseTarget)
	{
		const PxRigidBodyFlags targetFlags =
			gHeadlessResponseTarget->getRigidBodyFlags();
		gHeadlessMetrics.targetLinearCCD =
			(targetFlags & PxRigidBodyFlag::eENABLE_CCD) ? 1u : 0u;
		gHeadlessMetrics.targetSpeculativeCCD =
			(targetFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD) ?
				1u : 0u;
	}
	gHeadlessMetrics.targetDynamic =
		gHeadlessObstacle &&
		gHeadlessObstacle->getConcreteType() == PxConcreteType::eRIGID_DYNAMIC ?
			1u : 0u;
}

static void initHeadlessScene()
{
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.solverType = gHeadlessOptions.solverType;
	if(usesHeadlessLinearCCD())
	{
		sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;
		sceneDesc.filterShader = ccdFilterShader;
	}
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return;

	if(isHeadlessRaycastCase())
		gRaycastCCD = new RaycastCCDManager(gScene);

	if(isHeadlessAngularCase())
	{
		gHeadlessMover = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f)),
			PxBoxGeometry(10.0f, 1.0f, 0.1f), *gMaterial, 10.0f);
		gHeadlessResponseTarget = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f, 0.0f, 10.0f)),
			PxBoxGeometry(0.1f, 1.0f, 1.0f), *gMaterial, 10.0f);
		gHeadlessObstacle = gHeadlessResponseTarget;
		if(!gHeadlessMover || !gHeadlessResponseTarget)
			return;

		gHeadlessMover->setLinearDamping(0.0f);
		gHeadlessMover->setAngularDamping(0.0f);
		gHeadlessMover->setAngularVelocity(PxVec3(0.0f, 10.0f, 0.0f));
		gHeadlessMover->setSleepThreshold(0.0f);
		gHeadlessMover->setRigidDynamicLockFlag(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_X, true);
		gHeadlessMover->setRigidDynamicLockFlag(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Y, true);
		gHeadlessMover->setRigidDynamicLockFlag(
			PxRigidDynamicLockFlag::eLOCK_LINEAR_Z, true);
		gHeadlessMover->setRigidDynamicLockFlag(
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_X, true);
		gHeadlessMover->setRigidDynamicLockFlag(
			PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z, true);

		gHeadlessResponseTarget->setLinearDamping(0.0f);
		gHeadlessResponseTarget->setAngularDamping(0.0f);
		gHeadlessResponseTarget->setSleepThreshold(0.0f);
		if(usesHeadlessLinearCCD())
		{
			gHeadlessMover->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_CCD, true);
			gHeadlessResponseTarget->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_CCD, true);
		}
		if(usesHeadlessSpeculativeCCD())
		{
			gHeadlessMover->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
			gHeadlessResponseTarget->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		}

		gHeadlessTargetInitialPosition =
			gHeadlessResponseTarget->getGlobalPose().p;
		gScene->addActor(*gHeadlessMover);
		gScene->addActor(*gHeadlessResponseTarget);
	}
	else
	{
		const bool dynamicTarget = isHeadlessDynamicTargetCase();
		if(dynamicTarget)
		{
			PxRigidDynamic* target = PxCreateDynamic(
				*gPhysics, PxTransform(PxVec3(0.0f)),
				PxBoxGeometry(4.0f, 4.0f, 0.1f), *gMaterial, 1000.0f);
			gHeadlessObstacle = target;
			gHeadlessResponseTarget = target;
			if(target)
			{
				target->setLinearDamping(0.0f);
				target->setAngularDamping(0.0f);
				target->setSleepThreshold(0.0f);
				target->setRigidDynamicLockFlags(
					PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
					PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
					PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
					PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
					PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
					PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
			}
		}
		else
		{
			gHeadlessObstacle = PxCreateStatic(
				*gPhysics, PxTransform(PxVec3(0.0f)),
				PxBoxGeometry(4.0f, 4.0f, 0.1f), *gMaterial);
		}
		gHeadlessMover = PxCreateDynamic(
			*gPhysics, PxTransform(PxVec3(0.0f, 0.0f, 20.0f)),
			PxSphereGeometry(0.5f), *gMaterial, 10.0f);
		if(!gHeadlessObstacle || !gHeadlessMover)
			return;

		gHeadlessMover->setLinearDamping(0.0f);
		gHeadlessMover->setAngularDamping(0.0f);
		gHeadlessMover->setLinearVelocity(PxVec3(0.0f, 0.0f, -1000.0f));
		gHeadlessMover->setSleepThreshold(0.0f);
		if(usesHeadlessLinearCCD())
			gHeadlessMover->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_CCD, true);
		else if(usesHeadlessSpeculativeCCD())
			gHeadlessMover->setRigidBodyFlag(
				PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);

		gHeadlessTargetInitialPosition =
			gHeadlessObstacle->getGlobalPose().p;
		gScene->addActor(*gHeadlessObstacle);
		gScene->addActor(*gHeadlessMover);
		if(isHeadlessRaycastCase() &&
			registerForRaycastCCD(gHeadlessMover))
			++gHeadlessMetrics.raycastRegistrations;
	}

	recordHeadlessCCDFlags();
}

static void initScene()
{
	if(gHeadlessOptions.headless)
	{
		initHeadlessScene();
		return;
	}

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;

	bool enableLinearCCD = false;
	bool enableSpeculativeCCD = false;

	const CCDAlgorithm ccd = getCCDAlgorithm();
	if(ccd == LINEAR_CCD)
	{
		enableLinearCCD = true;
		sceneDesc.filterShader = ccdFilterShader;
		sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;
		gScene = gPhysics->createScene(sceneDesc);

		printf("- Using linear CCD.\n");
	}
	else if(ccd == SPECULATIVE_CCD)
	{
		enableSpeculativeCCD = true;
		gScene = gPhysics->createScene(sceneDesc);

		printf("- Using speculative/angular CCD.\n");
	}
	else if(ccd == FULL_CCD)
	{
		enableLinearCCD = true;
		enableSpeculativeCCD = true;
		sceneDesc.filterShader = ccdFilterShader;
		sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;

		gScene = gPhysics->createScene(sceneDesc);

		printf("- Using full CCD.\n");
	}
	else if(ccd == RAYCAST_CCD)
	{
		gScene = gPhysics->createScene(sceneDesc);
		gRaycastCCD = new RaycastCCDManager(gScene);

		printf("- Using raycast CCD.\n");
	}
	else if(ccd == NO_CCD)
	{
		gScene = gPhysics->createScene(sceneDesc);

		printf("- Using no CCD.\n");
	}

	// Create a scenario that requires angular CCD: a rotating plank colliding with a falling box.
	{
		PxRigidDynamic* actor = createDynamic(PxTransform(PxVec3(40.0f, 20.0f, 0.0f)), PxBoxGeometry(10.0f, 1.0f, 0.1f), PxVec3(0.0f), enableLinearCCD, enableSpeculativeCCD);
		actor->setAngularVelocity(PxVec3(0.0f, 10.0f, 0.0f));
		actor->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
		actor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_LINEAR_X, true);
		actor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_LINEAR_Y, true);
		actor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_LINEAR_Z, true);
		actor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X, true);
		actor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z, true);

		if(gRaycastCCD)
			registerForRaycastCCD(actor);

		PxRigidDynamic* actor2 = createDynamic(PxTransform(PxVec3(40.0f, 20.0f, 10.0f)), PxBoxGeometry(0.1f, 1.0f, 1.0f), PxVec3(0.0f), enableLinearCCD, enableSpeculativeCCD);

		if(gRaycastCCD)
			registerForRaycastCCD(actor2);
	}

	// Create a scenario that requires linear CCD: a fast moving sphere moving towards a box stack.
	{
		PxRigidDynamic* actor = createDynamic(PxTransform(PxVec3(0.0f, 18.0f, 100.0f)), PxSphereGeometry(2.0f), PxVec3(0.0f, 0.0f, -1000.0f), enableLinearCCD, enableSpeculativeCCD);
	
		if(gRaycastCCD)
			registerForRaycastCCD(actor);

		createStack(PxTransform(PxVec3(0, 0, stackZ)), 10, 2.0f, enableLinearCCD, enableSpeculativeCCD);

		PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
		gScene->addActor(*groundPlane);
	}

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
}

void renderText()
{
#ifdef RENDER_SNIPPET
	Snippets::print("Press F1 to F4 to select a scenario.");
	switch(PxU32(gScenario))
	{
		case 0:	{ Snippets::print("Current scenario: linear CCD");				}break;
		case 1:	{ Snippets::print("Current scenario: angular CCD");				}break;
		case 2:	{ Snippets::print("Current scenario: linear + angular CCD");	}break;
		case 3:	{ Snippets::print("Current scenario: raycast CCD");				}break;
		case 4:	{ Snippets::print("Current scenario: no CCD");					}break;
	}
#endif
}

void initPhysics(bool /*interactive*/)
{
	if(!gHeadlessOptions.headless)
	{
		printf("CCD snippet. Use these keys:\n");
		printf(" P        - enable/disable pause\n");
		printf(" O        - step simulation for one frame\n");
		printf(" R        - reset scene\n");
		printf(" F1 to F4 - select scenes with different CCD algorithms\n");
		printf("    F1	  - Using linear CCD\n");
		printf("    F2    - Using speculative/angular CCD\n");
		printf("    F3    - Using full CCD (linear+angular)\n");
		printf("    F4	  - Using raycast CCD\n");
		printf("    F5	  - Using no CCD\n");
		printf("\n");
	}

	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(
			PVD_HOST, 5425, 10);
		gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);

	if(gHeadlessOptions.headless)
		gDispatcher = PxDefaultCpuDispatcherCreate(
			gHeadlessOptions.dispatcherThreads);
	else
	{
		const PxU32 numCores = SnippetUtils::getNbPhysicalCores();
		gDispatcher = PxDefaultCpuDispatcherCreate(
			numCores == 0 ? 0 : numCores - 1);
	}

	gMaterial = gHeadlessOptions.headless ?
		gPhysics->createMaterial(0.0f, 0.0f, 0.0f) :
		gPhysics->createMaterial(0.5f, 0.5f, 0.25f);

	initScene();
}

void stepPhysics(bool /*interactive*/)
{
	if (gPause && !gOneFrame)
		return;
	gOneFrame = false;

	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);

	PxTransform raycastPoseBefore(PxIdentity);
	if(gHeadlessOptions.headless && gRaycastCCD && gHeadlessMover)
		raycastPoseBefore = gHeadlessMover->getGlobalPose();

	// Raycast CCD is an Extensions post-process and must run after fetchResults
	// but before the headless witness samples the corrected pose.
	if(gRaycastCCD)
		gRaycastCCD->doRaycastCCD(true);

	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gHeadlessMetrics.fetchFailures;
		if(gRaycastCCD && gHeadlessMover)
		{
			const PxTransform poseAfter = gHeadlessMover->getGlobalPose();
			if((poseAfter.p - raycastPoseBefore.p).magnitudeSquared() >
				1.0e-12f)
				++gHeadlessMetrics.raycastCorrections;
		}
		if(gHeadlessMover)
		{
			const PxTransform pose = gHeadlessMover->getGlobalPose();
			const PxVec3 velocity = gHeadlessMover->getLinearVelocity();
			const PxVec3 angularVelocity =
				gHeadlessMover->getAngularVelocity();
			if(!pose.isFinite() || !velocity.isFinite() ||
				!angularVelocity.isFinite())
				++gHeadlessMetrics.nonFinite;
			gHeadlessMetrics.moverMaxAngularSpeed = PxMax(
				gHeadlessMetrics.moverMaxAngularSpeed,
				angularVelocity.magnitude());
			gHeadlessMetrics.minMoverZ =
				PxMin(gHeadlessMetrics.minMoverZ, pose.p.z);
			gHeadlessMetrics.finalMoverZ = pose.p.z;
			gHeadlessMetrics.finalMoverVelocityZ = velocity.z;
			if(!isHeadlessAngularCase())
			{
				if(pose.p.z < -0.6f)
					++gHeadlessMetrics.crossedWall;
				if(velocity.z > -100.0f)
					++gHeadlessMetrics.responseSamples;
			}
		}
		if(gHeadlessResponseTarget)
		{
			const PxTransform targetPose =
				gHeadlessResponseTarget->getGlobalPose();
			const PxVec3 targetLinearVelocity =
				gHeadlessResponseTarget->getLinearVelocity();
			const PxVec3 targetAngularVelocity =
				gHeadlessResponseTarget->getAngularVelocity();
			if(!targetPose.isFinite() || !targetLinearVelocity.isFinite() ||
				!targetAngularVelocity.isFinite())
				++gHeadlessMetrics.nonFinite;
			gHeadlessMetrics.targetMaxDisplacement = PxMax(
				gHeadlessMetrics.targetMaxDisplacement,
				(targetPose.p - gHeadlessTargetInitialPosition).magnitude());
			gHeadlessMetrics.targetMaxLinearSpeed = PxMax(
				gHeadlessMetrics.targetMaxLinearSpeed,
				targetLinearVelocity.magnitude());
			gHeadlessMetrics.targetMaxAngularSpeed = PxMax(
				gHeadlessMetrics.targetMaxAngularSpeed,
				targetAngularVelocity.magnitude());
		}
		PxSimulationStatistics stats;
		gScene->getSimulationStatistics(stats);
		gHeadlessMetrics.ccdPairs +=
			stats.nbCCDPairs[PxGeometryType::eSPHERE][PxGeometryType::eBOX];
		++gHeadlessMetrics.completedFrames;
	}
}
	
static void releaseScene()
{
	PX_RELEASE(gScene);
	PX_DELETE(gRaycastCCD);
	gHeadlessMover = NULL;
	gHeadlessObstacle = NULL;
	gHeadlessResponseTarget = NULL;
}

void cleanupPhysics(bool /*interactive*/)
{
	releaseScene();
		
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gHeadlessMetrics.cleanupComplete =
		!gScene && !gRaycastCCD && !gDispatcher && !gPhysics &&
		!gPvd && !gFoundation ? 1u : 0u;
	
	printf("SnippetCCD done.\n");
}

void keyPress(unsigned char key, const PxTransform& /*camera*/)
{
	if(key == 'p' || key == 'P')
		gPause = !gPause;

	if(key == 'o' || key == 'O')
	{
		gPause = true;
		gOneFrame = true;
	}

	if(gScene)
	{
		if(key >= 1 && key <= gScenarioCount)
		{
			gScenario = key - 1;
			releaseScene();
			initScene();
		}

		if(key == 'r' || key == 'R')
		{
			releaseScene();
			initScene();
		}
	}
}

static int runHeadless()
{
	std::printf(
		"[AVBD_GATE_CONFIG] schema=2 snippet=SnippetCCD solver=%s case=%s "
		"execution=%s frames=%u dt=%.9g dispatcherThreads=%u seed=%u\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		getHeadlessCaseName(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, double(gHeadlessOptions.dt),
		gHeadlessOptions.dispatcherThreads, gHeadlessOptions.seed);

	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gMaterial && gScene &&
		gHeadlessMover && gHeadlessObstacle &&
		(!isHeadlessRaycastCase() || gRaycastCCD);
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
	}

	const char* reason = "none";
	bool passed = true;
	const PxU32 expectedLinearCCD = usesHeadlessLinearCCD() ? 1u : 0u;
	const PxU32 expectedSpeculativeCCD =
		usesHeadlessSpeculativeCCD() ? 1u : 0u;
	const PxU32 expectedTargetLinearCCD =
		isHeadlessAngularCase() && usesHeadlessLinearCCD() ? 1u : 0u;
	const PxU32 expectedTargetSpeculativeCCD =
		isHeadlessAngularCase() && usesHeadlessSpeculativeCCD() ? 1u : 0u;
	if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(gHeadlessMetrics.completedFrames != gHeadlessOptions.frames ||
		gHeadlessMetrics.fetchFailures != 0)
	{
		passed = false;
		reason = "incomplete_simulation";
	}
	else if(gHeadlessMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gHeadlessMetrics.sceneLinearCCD != expectedLinearCCD ||
		gHeadlessMetrics.sourceLinearCCD != expectedLinearCCD ||
		gHeadlessMetrics.sourceSpeculativeCCD != expectedSpeculativeCCD ||
		gHeadlessMetrics.targetLinearCCD != expectedTargetLinearCCD ||
		gHeadlessMetrics.targetSpeculativeCCD !=
			expectedTargetSpeculativeCCD)
	{
		passed = false;
		reason = "ccd_flag_readback_mismatch";
	}
	else if(gHeadlessMetrics.targetDynamic !=
		(isHeadlessDynamicTargetCase() ? 1u : 0u))
	{
		passed = false;
		reason = "target_type_mismatch";
	}
	else if(isHeadlessRaycastCase() &&
		gHeadlessMetrics.raycastRegistrations != 1)
	{
		passed = false;
		reason = "raycast_registration_failed";
	}
	else if(gHeadlessCase == HEADLESS_ANGULAR_NO_CCD)
	{
		if(gHeadlessMetrics.moverMaxAngularSpeed < 5.0f)
		{
			passed = false;
			reason = "angular_negative_control_not_exercised";
		}
		else if(gHeadlessMetrics.targetMaxDisplacement > 0.02f ||
			gHeadlessMetrics.targetMaxLinearSpeed > 0.05f ||
			gHeadlessMetrics.targetMaxAngularSpeed > 0.05f)
		{
			passed = false;
			reason = "angular_negative_control_detected_contact";
		}
	}
	else if(isHeadlessAngularCase())
	{
		if(gHeadlessMetrics.moverMaxAngularSpeed < 1.0f)
		{
			passed = false;
			reason = "angular_source_not_exercised";
		}
		else if(gHeadlessMetrics.targetMaxDisplacement < 0.05f ||
			(gHeadlessMetrics.targetMaxLinearSpeed < 0.1f &&
			 gHeadlessMetrics.targetMaxAngularSpeed < 0.1f))
		{
			passed = false;
			reason = "missing_angular_ccd_response";
		}
	}
	else if(gHeadlessCase == HEADLESS_NO_CCD ||
		gHeadlessCase == HEADLESS_DYNAMIC_NO_CCD)
	{
		if(gHeadlessMetrics.crossedWall == 0 ||
			gHeadlessMetrics.minMoverZ > -5.0f)
		{
			passed = false;
			reason = "negative_control_did_not_tunnel";
		}
	}
	else if(isHeadlessRaycastCase())
	{
		if(gHeadlessMetrics.raycastCorrections == 0 ||
			gHeadlessMetrics.crossedWall != 0 ||
			gHeadlessMetrics.minMoverZ < -0.6f)
		{
			passed = false;
			reason = "raycast_ccd_did_not_correct_pose";
		}
		else if(gHeadlessCase == HEADLESS_RAYCAST_DYNAMIC_CCD &&
			(gHeadlessMetrics.targetMaxDisplacement > 1.0e-5f ||
			 gHeadlessMetrics.targetMaxLinearSpeed > 1.0e-3f ||
			 gHeadlessMetrics.targetMaxAngularSpeed > 1.0e-3f))
		{
			passed = false;
			reason = "locked_dynamic_target_moved";
		}
	}
	else if(gHeadlessMetrics.crossedWall != 0 ||
		gHeadlessMetrics.minMoverZ < -0.6f)
	{
		passed = false;
		reason = "complete_tunneling";
	}
	else if(gHeadlessMetrics.responseSamples == 0)
	{
		passed = false;
		reason = "missing_impact_response";
	}
	else if((gHeadlessCase == HEADLESS_LINEAR_CCD ||
			 gHeadlessCase == HEADLESS_DYNAMIC_LINEAR_CCD) &&
		gHeadlessMetrics.ccdPairs == 0)
	{
		passed = false;
		reason = "missing_ccd_pair";
	}
	else if(gHeadlessCase == HEADLESS_DYNAMIC_LINEAR_CCD &&
		(gHeadlessMetrics.targetMaxDisplacement > 1.0e-5f ||
		 gHeadlessMetrics.targetMaxLinearSpeed > 1.0e-3f ||
		 gHeadlessMetrics.targetMaxAngularSpeed > 1.0e-3f))
	{
		passed = false;
		reason = "locked_dynamic_target_moved";
	}

	cleanupPhysics(false);
	if(!gHeadlessMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=2 snippet=SnippetCCD solver=%s case=%s "
		"execution=%s frames=%u completedFrames=%u status=%s reason=%s "
		"validation=GATED fixture=%s sourceGeometry=%s "
		"targetGeometry=box sphereRadius=0.5 plankHalfLength=10 "
		"wallHalfThickness=0.1 sourceInitialZ=%.9g "
		"sourceInitialVelocityZ=%.9g sourceInitialAngularVelocityY=%.9g "
		"crossedWall=%u "
		"responseSamples=%u ccdPairs=%u minMoverZ=%.9g "
		"finalMoverZ=%.9g finalMoverVelocityZ=%.9g "
		"moverMaxAngularSpeed=%.9g targetMaxDisplacement=%.9g "
		"targetMaxLinearSpeed=%.9g targetMaxAngularSpeed=%.9g "
		"sceneLinearCCD=%u sourceLinearCCD=%u sourceSpeculativeCCD=%u "
		"targetLinearCCD=%u targetSpeculativeCCD=%u targetDynamic=%u "
		"raycastRegistrations=%u raycastCorrections=%u nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		getHeadlessCaseName(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gHeadlessMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		isHeadlessAngularCase() ? "angular" : "linear",
		isHeadlessAngularCase() ? "box" : "sphere",
		isHeadlessAngularCase() ? 0.0 : 20.0,
		isHeadlessAngularCase() ? 0.0 : -1000.0,
		isHeadlessAngularCase() ? 10.0 : 0.0,
		gHeadlessMetrics.crossedWall, gHeadlessMetrics.responseSamples,
		gHeadlessMetrics.ccdPairs, double(gHeadlessMetrics.minMoverZ),
		double(gHeadlessMetrics.finalMoverZ),
		double(gHeadlessMetrics.finalMoverVelocityZ),
		double(gHeadlessMetrics.moverMaxAngularSpeed),
		double(gHeadlessMetrics.targetMaxDisplacement),
		double(gHeadlessMetrics.targetMaxLinearSpeed),
		double(gHeadlessMetrics.targetMaxAngularSpeed),
		gHeadlessMetrics.sceneLinearCCD,
		gHeadlessMetrics.sourceLinearCCD,
		gHeadlessMetrics.sourceSpeculativeCCD,
		gHeadlessMetrics.targetLinearCCD,
		gHeadlessMetrics.targetSpeculativeCCD,
		gHeadlessMetrics.targetDynamic,
		gHeadlessMetrics.raycastRegistrations,
		gHeadlessMetrics.raycastCorrections,
		gHeadlessMetrics.nonFinite, gHeadlessMetrics.fetchFailures,
		gErrorCallback.getFatalCount(), gHeadlessMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCCD reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCCD "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
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
