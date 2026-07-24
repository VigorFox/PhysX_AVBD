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
// This snippet illustrates the use of built-in triggers, and how to emulate
// them with regular shapes if you need CCD or trigger-trigger notifications.
// 
// ****************************************************************************

#include "PxPhysicsAPI.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"

#include <cfloat>
#include <cstdio>

using namespace physx;

enum TriggerImpl
{
	// Uses built-in triggers (PxShapeFlag::eTRIGGER_SHAPE).
	REAL_TRIGGERS,

	// Emulates triggers using a filter shader. Needs one reserved value in PxFilterData.
	FILTER_SHADER,

	// Emulates triggers using a filter callback. Doesn't use PxFilterData but needs user-defined way to mark a shape as a trigger.
	FILTER_CALLBACK,
};

struct ScenarioData
{
	TriggerImpl	mImpl;
	bool		mCCD;
	bool		mTriggerTrigger;
};

#define SCENARIO_COUNT	9

static ScenarioData gData[SCENARIO_COUNT] = {
	{REAL_TRIGGERS,		false,	false},
	{FILTER_SHADER,		false,	false},
	{FILTER_CALLBACK,	false,	false},
	{REAL_TRIGGERS,		true,	false},
	{FILTER_SHADER,		true,	false},
	{FILTER_CALLBACK,	true,	false},
	{REAL_TRIGGERS,		false,	true},
	{FILTER_SHADER,		false,	true},
	{FILTER_CALLBACK,	false,	true},
};

static PxU32 gScenario = 0;

static PX_FORCE_INLINE	TriggerImpl	getImpl()				{ return gData[gScenario].mImpl;			}
static PX_FORCE_INLINE	bool		usesCCD()				{ return gData[gScenario].mCCD;				}
static PX_FORCE_INLINE	bool		usesTriggerTrigger()	{ return gData[gScenario].mTriggerTrigger;	}

static	PxDefaultAllocator		gAllocator;
static	Snippets::TrackingErrorCallback gErrorCallback;

static	PxFoundation*			gFoundation = NULL;
static	PxPhysics*				gPhysics	= NULL;

static	PxDefaultCpuDispatcher*	gDispatcher = NULL;
static	PxScene*				gScene		= NULL;
static	PxMaterial*				gMaterial	= NULL;
static	PxPvd*                  gPvd        = NULL;

static bool	gPause		= false;
static bool	gOneFrame	= false;
static Snippets::HeadlessOptions gHeadlessOptions;
static PxRigidDynamic* gRemovalActor = NULL;
static PxRigidDynamic* gTriggerActors[2] = {NULL, NULL};

struct ScenarioMetrics
{
	PxU32 found;
	PxU32 lost;
	PxU32 ccd;
	PxU32 triggerTrigger;
	PxU32 filterFound;
	PxU32 filterLost;
	PxU32 objectRemoved;
	PxU32 completedFrames;
	PxU32 nonFinite;
	PxU32 fetchFailures;
	bool removalIssued;
	bool overlapActuationIssued;
	PxReal minActorDistance;

	ScenarioMetrics()
	: found(0), lost(0), ccd(0), triggerTrigger(0), filterFound(0),
	  filterLost(0), objectRemoved(0), completedFrames(0), nonFinite(0),
	  fetchFailures(0), removalIssued(false), overlapActuationIssued(false),
	  minActorDistance(FLT_MAX)
	{
	}
};

static ScenarioMetrics gScenarioMetrics[SCENARIO_COUNT];
static bool gSolverReadbackMatched = false;
static PxU32 gCleanupComplete = 0;

// Detects a trigger using the shape's simulation filter data. See createTriggerShape() function.
bool isTrigger(const PxFilterData& data)
{
	if(data.word0!=0xffffffff)
		return false;
	if(data.word1!=0xffffffff)
		return false;
	if(data.word2!=0xffffffff)
		return false;
	if(data.word3!=0xffffffff)
		return false;
	return true;
}

bool isTriggerShape(PxShape* shape)
{
	const TriggerImpl impl = getImpl();

	// Detects native built-in triggers.
	if(impl==REAL_TRIGGERS && (shape->getFlags() & PxShapeFlag::eTRIGGER_SHAPE))
		return true;

	// Detects our emulated triggers using the simulation filter data. See createTriggerShape() function.
	if(impl==FILTER_SHADER && ::isTrigger(shape->getSimulationFilterData()))
		return true;

	// Detects our emulated triggers using the simulation filter callback. See createTriggerShape() function.
	if(impl==FILTER_CALLBACK && shape->userData)
		return true;

	return false;
}

static	PxFilterFlags triggersUsingFilterShader(PxFilterObjectAttributes /*attributes0*/, PxFilterData filterData0, 
												PxFilterObjectAttributes /*attributes1*/, PxFilterData filterData1,
												PxPairFlags& pairFlags, const void* /*constantBlock*/, PxU32 /*constantBlockSize*/)
{
//	printf("contactReportFilterShader\n");

	PX_ASSERT(getImpl()==FILTER_SHADER);

	// We need to detect whether one of the shapes is a trigger.
	const bool isTriggerPair = isTrigger(filterData0) || isTrigger(filterData1);

	// If we have a trigger, replicate the trigger codepath from PxDefaultSimulationFilterShader
	if(isTriggerPair)
	{
		pairFlags = PxPairFlag::eTRIGGER_DEFAULT;

		if(usesCCD())
			pairFlags |= PxPairFlag::eDETECT_CCD_CONTACT;

		return PxFilterFlag::eDEFAULT;
	}
	else
	{
		// Otherwise use the default flags for regular pairs
		pairFlags = PxPairFlag::eCONTACT_DEFAULT;
		return PxFilterFlag::eDEFAULT;
	}
}

static	PxFilterFlags triggersUsingFilterCallback(PxFilterObjectAttributes /*attributes0*/, PxFilterData /*filterData0*/, 
												PxFilterObjectAttributes /*attributes1*/, PxFilterData /*filterData1*/,
												PxPairFlags& pairFlags, const void* /*constantBlock*/, PxU32 /*constantBlockSize*/)
{
//	printf("contactReportFilterShader\n");

	PX_ASSERT(getImpl()==FILTER_CALLBACK);

	pairFlags = PxPairFlag::eCONTACT_DEFAULT;

	if(usesCCD())
		pairFlags |= PxPairFlag::eDETECT_CCD_CONTACT|PxPairFlag::eNOTIFY_TOUCH_CCD;

	return PxFilterFlag::eCALLBACK;
}

class TriggersFilterCallback : public PxSimulationFilterCallback
{
	virtual		PxFilterFlags	pairFound(	PxU64 /*pairID*/,
											PxFilterObjectAttributes /*attributes0*/, PxFilterData /*filterData0*/, const PxActor* /*a0*/, const PxShape* s0,
											PxFilterObjectAttributes /*attributes1*/, PxFilterData /*filterData1*/, const PxActor* /*a1*/, const PxShape* s1,
											PxPairFlags& pairFlags)	PX_OVERRIDE
	{
//		printf("pairFound\n");
		gScenarioMetrics[gScenario].filterFound++;

		if(s0->userData || s1->userData)	// See createTriggerShape() function
		{
			pairFlags = PxPairFlag::eTRIGGER_DEFAULT;

			if(usesCCD())
				pairFlags |= PxPairFlag::eDETECT_CCD_CONTACT|PxPairFlag::eNOTIFY_TOUCH_CCD;
		}
		else
			pairFlags = PxPairFlag::eCONTACT_DEFAULT;

		return PxFilterFlags();
	}

	virtual		void	pairLost(	PxU64 /*pairID*/,
									PxFilterObjectAttributes /*attributes0*/, PxFilterData /*filterData0*/,
									PxFilterObjectAttributes /*attributes1*/, PxFilterData /*filterData1*/,
									bool objectRemoved)	PX_OVERRIDE
	{
//		printf("pairLost\n");
		ScenarioMetrics& metrics = gScenarioMetrics[gScenario];
		metrics.filterLost++;
		if(objectRemoved)
			metrics.objectRemoved++;
	}

	virtual		bool	statusChange(PxU64& /*pairID*/, PxPairFlags& /*pairFlags*/, PxFilterFlags& /*filterFlags*/)	PX_OVERRIDE
	{
//		printf("statusChange\n");
		return false;
	}
}gTriggersFilterCallback;

class ContactReportCallback: public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo* /*constraints*/, PxU32 /*count*/)	PX_OVERRIDE
	{
		printf("onConstraintBreak\n");
	}

	void onWake(PxActor** /*actors*/, PxU32 /*count*/)	PX_OVERRIDE
	{
		printf("onWake\n");
	}

	void onSleep(PxActor** /*actors*/, PxU32 /*count*/)	PX_OVERRIDE
	{
		printf("onSleep\n");
	}

	void onTrigger(PxTriggerPair* pairs, PxU32 count)	PX_OVERRIDE
	{
//		printf("onTrigger: %d trigger pairs\n", count);
		while(count--)
		{
			const PxTriggerPair& current = *pairs++;
			if(current.status & PxPairFlag::eNOTIFY_TOUCH_FOUND)
			{
				gScenarioMetrics[gScenario].found++;
				printf("Shape is entering trigger volume\n");
			}
			if(current.status & PxPairFlag::eNOTIFY_TOUCH_LOST)
			{
				gScenarioMetrics[gScenario].lost++;
				printf("Shape is leaving trigger volume\n");
			}
			if(current.flags &
			   (PxTriggerPairFlag::eREMOVED_SHAPE_TRIGGER |
			    PxTriggerPairFlag::eREMOVED_SHAPE_OTHER))
				gScenarioMetrics[gScenario].objectRemoved++;
		}
	}

	void onAdvance(const PxRigidBody*const*, const PxTransform*, const PxU32)	PX_OVERRIDE
	{
		printf("onAdvance\n");
	}

	void onContact(const PxContactPairHeader& /*pairHeader*/, const PxContactPair* pairs, PxU32 count)	PX_OVERRIDE
	{
//		printf("onContact: %d pairs\n", count);

		while(count--)
		{
			const PxContactPair& current = *pairs++;

			// The reported pairs can be trigger pairs or not. We only enabled contact reports for
			// trigger pairs in the filter shader, so we don't need to do further checks here. In a
			// real-world scenario you would probably need a way to tell whether one of the shapes
			// is a trigger or not. You could e.g. reuse the PxFilterData like we did in the filter
			// shader, or maybe use the shape's userData to identify triggers, or maybe put triggers
			// in a hash-set and test the reported shape pointers against it. Many options here.

			if(current.events & (PxPairFlag::eNOTIFY_TOUCH_FOUND|PxPairFlag::eNOTIFY_TOUCH_CCD))
			{
				gScenarioMetrics[gScenario].found++;
				if(current.events & PxPairFlag::eNOTIFY_TOUCH_CCD)
					gScenarioMetrics[gScenario].ccd++;
				printf("Shape is entering trigger volume\n");
			}
			if(current.events & PxPairFlag::eNOTIFY_TOUCH_LOST)
			{
				gScenarioMetrics[gScenario].lost++;
				printf("Shape is leaving trigger volume\n");
			}

			const bool removedShape =
				!!(current.flags &
				   (PxContactPairFlag::eREMOVED_SHAPE_0 |
				    PxContactPairFlag::eREMOVED_SHAPE_1));
			if(removedShape)
				gScenarioMetrics[gScenario].objectRemoved++;
			if(!removedShape && isTriggerShape(current.shapes[0]) &&
			   isTriggerShape(current.shapes[1]))
			{
				gScenarioMetrics[gScenario].triggerTrigger++;
				printf("Trigger-trigger overlap detected\n");
			}
		}
	}
};

static ContactReportCallback gContactReportCallback;

static PxShape* createTriggerShape(const PxGeometry& geom, bool isExclusive)
{
	const TriggerImpl impl = getImpl();

	PxShape* shape = nullptr;
	if(impl==REAL_TRIGGERS)
	{
		const PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eTRIGGER_SHAPE;
		shape = gPhysics->createShape(geom, *gMaterial, isExclusive, shapeFlags);
	}
	else if(impl==FILTER_SHADER)
	{
		PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSIMULATION_SHAPE;
		shape = gPhysics->createShape(geom, *gMaterial, isExclusive, shapeFlags);

		// For this method to work, you need a way to mark shapes as triggers without using PxShapeFlag::eTRIGGER_SHAPE
		// (so that trigger-trigger pairs are reported), and without calling a PxShape function (so that the data is
		// available in a filter shader).
		//
		// One way is to reserve a special PxFilterData value/mask for triggers. It may not always be possible depending
		// on how you otherwise use the filter data).
		const PxFilterData triggerFilterData(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
		shape->setSimulationFilterData(triggerFilterData);
	}
	else if(impl==FILTER_CALLBACK)
	{
		// We will have access to shape pointers in the filter callback so we just mark triggers in an arbitrary way here,
		// for example using the shape's userData.
		shape = gPhysics->createShape(geom, *gMaterial, isExclusive);
		shape->userData = shape;	// Arbitrary rule: it's a trigger if non null
	}
	return shape;
}

static void createDefaultScene()
{
	const bool ccd = usesCCD();

	// Create trigger shape
	{
		const PxVec3 halfExtent(10.0f, ccd ? 0.01f : 1.0f, 10.0f);
		PxShape* shape = createTriggerShape(PxBoxGeometry(halfExtent), false);

		if(shape)
		{
			PxRigidStatic* body = gPhysics->createRigidStatic(PxTransform(0.0f, 10.0f, 0.0f));
			body->attachShape(*shape);
			gScene->addActor(*body);
			shape->release();
		}
	}

	// Create falling rigid body
	{
		const PxVec3 halfExtent(ccd ? 0.1f : 1.0f);

		PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent), *gMaterial);

		PxRigidDynamic* body = gPhysics->createRigidDynamic(PxTransform(0.0f, ccd ? 30.0f : 20.0f, 0.0f));
		body->attachShape(*shape);

		PxRigidBodyExt::updateMassAndInertia(*body, 1.0f);
		gScene->addActor(*body);
		shape->release();

		if(ccd)
		{
			body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
			body->setLinearVelocity(PxVec3(0.0f, -140.0f, 0.0f));
		}
	}
}

static void createTriggerTriggerScene()
{
	struct Local
	{
		static void createSphereActor(const PxVec3& pos, const PxVec3& linVel)
		{
			PxShape* sphereShape = gPhysics->createShape(PxSphereGeometry(1.0f), *gMaterial, false);

			PxRigidDynamic* body = gPhysics->createRigidDynamic(PxTransform(pos));
			body->attachShape(*sphereShape);

			PxShape* triggerShape = createTriggerShape(PxSphereGeometry(4.0f), true);
			body->attachShape(*triggerShape);

			const bool isTriggershape = triggerShape->getFlags() & PxShapeFlag::eTRIGGER_SHAPE;
			if(!isTriggershape)
				triggerShape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
			PxRigidBodyExt::updateMassAndInertia(*body, 1.0f);
			if(!isTriggershape)
				triggerShape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, true);
			gScene->addActor(*body);
			sphereShape->release();
			triggerShape->release();

			body->setLinearVelocity(linVel);
			if(!gTriggerActors[0])
				gTriggerActors[0] = body;
			else if(!gTriggerActors[1])
				gTriggerActors[1] = body;
			if(!gRemovalActor)
				gRemovalActor = body;
		}
	};

	Local::createSphereActor(PxVec3(-5.0f, 1.0f, 0.0f), PxVec3( 1.0f, 0.0f, 0.0f));
	Local::createSphereActor(PxVec3( 5.0f, 1.0f, 0.0f), PxVec3(-1.0f, 0.0f, 0.0f));
}

static void initScene()
{
	gRemovalActor = NULL;
	gTriggerActors[0] = gTriggerActors[1] = NULL;
	const TriggerImpl impl = getImpl();

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
//	sceneDesc.flags &= ~PxSceneFlag::eENABLE_PCM;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.gravity = PxVec3(0, -9.81f, 0);
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	sceneDesc.simulationEventCallback = &gContactReportCallback;
	if(impl==REAL_TRIGGERS)
	{
		sceneDesc.filterShader		= PxDefaultSimulationFilterShader;
		printf("- Using built-in triggers.\n");
	}
	else if(impl==FILTER_SHADER)
	{
		sceneDesc.filterShader		= triggersUsingFilterShader;
		printf("- Using regular shapes emulating triggers with a filter shader.\n");
	}
	else if(impl==FILTER_CALLBACK)
	{
		sceneDesc.filterShader		= triggersUsingFilterCallback;
		sceneDesc.filterCallback	= &gTriggersFilterCallback;
		printf("- Using regular shapes emulating triggers with a filter callback.\n");
	}

	if(usesCCD())
	{
		sceneDesc.flags |= PxSceneFlag::eENABLE_CCD;
		printf("- Using CCD.\n");
	}
	else
	{
		printf("- Using no CCD.\n");
	}

	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return;
	if(gHeadlessOptions.headless)
		gSolverReadbackMatched &=
			gScene->getSolverType() == sceneDesc.solverType;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	if(usesTriggerTrigger())
		createTriggerTriggerScene();
	else
		createDefaultScene();
}

static void releaseScene()
{
	PX_RELEASE(gScene);
	gRemovalActor = NULL;
	gTriggerActors[0] = gTriggerActors[1] = NULL;
}

static bool stepPhysicsInternal()
{
	if(gPause && !gOneFrame)
		return true;
	gOneFrame = false;

	if(gScene)
	{
//		printf("Update...\n");
		gScene->simulate(gHeadlessOptions.headless ? gHeadlessOptions.dt :
			1.0f/60.0f);
		if(!gScene->fetchResults(true))
			return false;
	}
	return true;
}

void stepPhysics(bool /*interactive*/)
{
	stepPhysicsInternal();
}

static bool initPhysicsInternal(bool headless)
{
	printf("Press keys F1 to F9 to select a scenario.\n");

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
	PxInitExtensions(*gPhysics,gPvd);
	const PxU32 numCores = SnippetUtils::getNbPhysicalCores();
	const PxU32 workerCount = headless ? gHeadlessOptions.dispatcherThreads :
		(numCores == 0 ? 0 : numCores - 1);
	gDispatcher = PxDefaultCpuDispatcherCreate(workerCount);
	if(!gDispatcher)
		return false;
	gMaterial = gPhysics->createMaterial(1.0f, 1.0f, 0.0f);
	if(!gMaterial)
		return false;

	gSolverReadbackMatched = true;
	initScene();
	return gScene != NULL;
}

void initPhysics(bool /*interactive*/)
{
	initPhysicsInternal(false);
}
	
void cleanupPhysics(bool /*interactive*/)
{
	releaseScene();

	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	PxCloseExtensions();
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gCleanupComplete = 1;
	
	printf("SnippetTriggers done.\n");
}

void keyPress(unsigned char key, const PxTransform& /*camera*/)
{
	if(key=='p' || key=='P')
		gPause = !gPause;

	if(key=='o' || key=='O')
	{
		gPause = true;
		gOneFrame = true;
	}

	if(gScene)
	{
		if(key>=1 && key<=SCENARIO_COUNT)
		{
			gScenario = key-1;
			releaseScene();
			initScene();
		}

		if(key=='r' || key=='R')
		{
			releaseScene();
			initScene();
		}
	}
}

static bool validateFiniteActors()
{
	PxActor* actors[16];
	const PxU32 count = gScene->getActors(
		PxActorTypeFlag::eRIGID_DYNAMIC, actors, 16);
	for(PxU32 i=0; i<count; ++i)
	{
		PxRigidDynamic* body = static_cast<PxRigidDynamic*>(actors[i]);
		if(!body->getGlobalPose().isFinite() ||
		   !body->getLinearVelocity().isFinite() ||
		   !body->getAngularVelocity().isFinite())
			return false;
	}
	return true;
}

static int runHeadless()
{
	gErrorCallback.reset();
	gCleanupComplete = 0;
	for(PxU32 i=0; i<SCENARIO_COUNT; ++i)
		gScenarioMetrics[i] = ScenarioMetrics();
	gScenario = 0;
	if(!initPhysicsInternal(true))
	{
		cleanupPhysics(false);
		return Snippets::eHEADLESS_GATE_FAILED;
	}

	for(PxU32 scenario=0; scenario<SCENARIO_COUNT; ++scenario)
	{
		if(scenario)
		{
			releaseScene();
			gScenario = scenario;
			initScene();
			if(!gScene)
				break;
		}
		ScenarioMetrics& metrics = gScenarioMetrics[scenario];
		for(PxU32 frame=0; frame<gHeadlessOptions.frames; ++frame)
		{
			if(usesTriggerTrigger() && frame == 120 &&
			   gTriggerActors[0] && gTriggerActors[1])
			{
				gTriggerActors[0]->setGlobalPose(
					PxTransform(PxVec3(-3.5f, 1.0f, 0.0f)));
				gTriggerActors[1]->setGlobalPose(
					PxTransform(PxVec3(3.5f, 1.0f, 0.0f)));
				metrics.overlapActuationIssued = true;
			}
			if(gTriggerActors[0] && gTriggerActors[1])
			{
				const PxReal distance =
					(gTriggerActors[0]->getGlobalPose().p -
					 gTriggerActors[1]->getGlobalPose().p).magnitude();
				metrics.minActorDistance =
					PxMin(metrics.minActorDistance, distance);
			}
			if(usesTriggerTrigger() && frame == 150 && gRemovalActor)
			{
				PxRigidDynamic* removedActor = gRemovalActor;
				gScene->removeActor(*removedActor);
				if(gTriggerActors[0] == removedActor)
					gTriggerActors[0] = NULL;
				if(gTriggerActors[1] == removedActor)
					gTriggerActors[1] = NULL;
				removedActor->release();
				gRemovalActor = NULL;
				metrics.removalIssued = true;
			}
			if(!stepPhysicsInternal())
			{
				metrics.fetchFailures++;
				break;
			}
			metrics.completedFrames++;
			if(!validateFiniteActors())
			{
				metrics.nonFinite++;
				break;
			}
		}
	}

	PxU32 passedScenarios = 0;
	PxU32 foundScenarios = 0;
	PxU32 lostScenarios = 0;
	PxU32 negativeControlPasses = 0;
	PxU32 removalScenarios = 0;
	PxU32 triggerTriggerScenarios = 0;
	PxU32 overlapActuationScenarios = 0;
	PxU32 totalFound = 0;
	PxU32 totalLost = 0;
	PxU32 totalRemoved = 0;
	PxU32 totalNonFinite = 0;
	PxU32 totalFetchFailures = 0;
	PxReal maxPositiveTriggerMinDistance = 0.0f;
	for(PxU32 i=0; i<SCENARIO_COUNT; ++i)
	{
		const ScenarioMetrics& metrics = gScenarioMetrics[i];
		const bool ccdNegativeControl = i == 3;
		const bool triggerTriggerNegativeControl = i == 6;
		const bool found = metrics.found > 0;
		const bool lost = metrics.lost > 0 || metrics.filterLost > 0;
		const bool complete =
			metrics.completedFrames == gHeadlessOptions.frames;
		const bool finite =
			metrics.nonFinite == 0 && metrics.fetchFailures == 0;
		const bool expectedEvents =
			ccdNegativeControl ? !found : (found && lost);
		const bool triggerTriggerSemantics =
			i < 6 ||
			(triggerTriggerNegativeControl ?
				metrics.triggerTrigger == 0 :
				metrics.triggerTrigger > 0);
		const bool removalOk =
			i < 6 || (metrics.removalIssued &&
				lost);
		const bool actuationOk =
			i < 6 || metrics.overlapActuationIssued;
		const bool passed = complete && finite && expectedEvents &&
			triggerTriggerSemantics && removalOk && actuationOk;
		passedScenarios += passed ? 1u : 0u;
		foundScenarios += found ? 1u : 0u;
		lostScenarios += lost ? 1u : 0u;
		negativeControlPasses +=
			((ccdNegativeControl && !found) ||
			 (triggerTriggerNegativeControl &&
			  metrics.triggerTrigger == 0)) ? 1u : 0u;
		removalScenarios +=
			i >= 6 && metrics.removalIssued && lost ? 1u : 0u;
		triggerTriggerScenarios +=
			metrics.triggerTrigger > 0 ? 1u : 0u;
		overlapActuationScenarios +=
			metrics.overlapActuationIssued ? 1u : 0u;
		totalFound += metrics.found;
		totalLost += metrics.lost + metrics.filterLost;
		totalRemoved += metrics.objectRemoved;
		totalNonFinite += metrics.nonFinite;
		totalFetchFailures += metrics.fetchFailures;
		if(i == 7 || i == 8)
			maxPositiveTriggerMinDistance = PxMax(
				maxPositiveTriggerMinDistance, metrics.minActorDistance);
	}

	const bool passed =
		passedScenarios == SCENARIO_COUNT &&
		negativeControlPasses == 2 &&
		removalScenarios >= 2 &&
		triggerTriggerScenarios == 2 &&
		overlapActuationScenarios == 3 &&
		maxPositiveTriggerMinDistance < 8.0f &&
		totalRemoved >= 2 &&
		gSolverReadbackMatched && totalNonFinite == 0 &&
		totalFetchFailures == 0 && gErrorCallback.getFatalCount() == 0;
	const PxU32 fatalErrors = gErrorCallback.getFatalCount();
	cleanupPhysics(false);
	printf("[AVBD_GATE] schema=1 snippet=SnippetTriggers solver=%s "
		"case=trigger-matrix execution=%s frames=%u scenarios=9 "
		"passedScenarios=%u foundScenarios=%u lostScenarios=%u "
		"negativeControlPasses=%u removalScenarios=%u "
		"triggerTriggerScenarios=%u overlapActuationScenarios=%u "
		"totalFound=%u "
		"totalLost=%u objectRemoved=%u maxPositiveTriggerMinDistance=%.9g "
		"nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0 status=%s reason=%s "
		"validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, passedScenarios, foundScenarios,
		lostScenarios, negativeControlPasses, removalScenarios,
		triggerTriggerScenarios, overlapActuationScenarios, totalFound,
		totalLost, totalRemoved, maxPositiveTriggerMinDistance,
		totalNonFinite, totalFetchFailures,
		fatalErrors, gCleanupComplete, passed ? "PASS" : "FAIL",
		passed ? "none" : "trigger_semantics");
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 240;
	defaults.caseName = "trigger-matrix";
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
	{
		printf("[AVBD_GATE_CONFIG_ERROR] snippet=SnippetTriggers reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysics(false);
	for(PxU32 i=0; i<250; i++)
		stepPhysics(false);
	cleanupPhysics(false);
#endif

	return 0;
}
