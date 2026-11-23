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

// Distilled HelloWorld headless gates. Scene contract from PhysX_AVBD_2;
// not a copy of that 4k-line demo. Visual path stays in SnippetHelloWorld.cpp.

#include "SnippetHelloWorldHeadless.h"

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace physx;

namespace
{

enum HelloWorldHeadlessCase
{
	eHELLO_CASE_STACK_SETTLE,
	eHELLO_CASE_BALL_SHOT,
	eHELLO_CASE_SLEEP_IDLE,
	eHELLO_CASE_SLEEP_WAKE,
	eHELLO_CASE_SLEEP_DISABLED,
	eHELLO_CASE_LOCK_FLAGS,
	eHELLO_CASE_TRIPLE_BALL_SLIDE,
	eHELLO_CASE_SUB_BOUNCE_STACK,
	eHELLO_CASE_RIGID_STRESS
};

enum
{
	eHELLO_FILTER_BALL = 1,
	eHELLO_FILTER_TARGET_BOX = 2,
	eHELLO_FILTER_SUB_BOUNCE_GROUND = 4,
	eHELLO_FILTER_SUB_BOUNCE_BOX_0 = 5,
	eHELLO_FILTER_SUB_BOUNCE_BOX_1 = 6,
	eHELLO_FILTER_SUB_BOUNCE_BOX_2 = 7
};

enum
{
	eHELLO_SUB_BOUNCE_BOX_COUNT = 3,
	eHELLO_SUB_BOUNCE_INTERFACE_COUNT = 3
};

enum
{
	eHELLO_SLEEP_FREE = 0,
	eHELLO_SLEEP_STATIC_TOUCH = 1,
	eHELLO_SLEEP_WITNESS_COUNT = 2
};

enum
{
	eHELLO_LOCK_LINEAR_X = 0,
	eHELLO_LOCK_LINEAR_Y,
	eHELLO_LOCK_LINEAR_Z,
	eHELLO_LOCK_ANGULAR_X,
	eHELLO_LOCK_ANGULAR_Y,
	eHELLO_LOCK_ANGULAR_Z,
	eHELLO_LOCK_WITNESS_COUNT
};

static const PxU32 gExpectedBoxCount = 275;
static const PxU32 gExpectedTargetBoxCount = 55;
static const PxU32 gHelloWorldStackCount = 5;
static const PxU32 gHelloWorldBoxesPerStack = 55;
static const PxReal gBoxHalfExtent = 2.0f;
static const PxReal gBallShotRadius = 3.0f;
static const PxVec3 gBallShotPos(0.0f, 22.0f, 70.0f);
static const PxVec3 gBallShotVel(0.0f, -20.0f, -220.0f);
static const PxU32 gBallShotFrame = 30;
static const PxU32 gBallResponseWindowFrames = 3;
static const PxU32 gBoxResponseWindowFrames = 12;
static const PxReal gMinBallResponseFraction = 0.05f;
static const PxReal gMinTargetResponseMomentumRatio = 0.01f;
static const PxReal gMinTargetResponseDisplacement = 0.5f;
static const PxReal gSettleSpeedCap = 12.0f;
static const PxReal gMechanicalEnergyGainCap = 0.05f;
static const PxReal gTailEnergyGrowthCap = 0.05f;
static const PxReal gMaxAbsPositionCap = 5000.0f;
static const PxReal gMaxLinearSpeedCap = 300.0f;
static const PxReal gMaxAngularSpeedCap = 120.0f;
static const PxReal gSleepBoxHalfExtent = 0.5f;
static const PxU32 gSleepWakeFrame = 60;
static const PxReal gSleepWakeTargetDeltaVelocity = 2.0f;
static const PxReal gLockLinearSpeed = 3.0f;
static const PxReal gLockAngularSpeed = 2.0f;
static const PxU32 gLockImpulseFrame = 30;
static const PxReal gLockMotionTolerance = 1e-4f;
static const PxReal gLockSpeedTolerance = 1e-4f;
static const PxReal gLockControlMotionMinimum = 1.0f;
static const PxReal gLockControlSpeedMinimum = 1.0f;
// AVBD_2 HelloWorld sub-bounce-stack: 3 unit boxes, vertical support gap.
// Caps are stop-ship (no visible embed / no settle twitch), not TGS parity.
static const PxReal gSubBounceBoxHalfExtent = 0.5f;
static const PxU32 gSubBounceTailFrames = 60;
static const PxReal gSubBouncePenetrationCap = 0.02f;
static const PxReal gSubBounceTailLinearCap = 0.25f;
static const PxReal gSubBounceTailAngularCap = 0.25f;

struct BoxRecord
{
	PxRigidDynamic* body;
	bool target;
	bool contacted;
	bool responded;
	PxU32 contactFrame;
	PxVec3 prevVel;
	PxVec3 prevPos;
	PxQuat prevQ;
	PxVec3 contactVelBaseline;
	PxVec3 contactPosBaseline;
};

struct SubBounceMetrics
{
	PxU32 actorCount;
	PxU32 contactFrames[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxU32 contactPoints[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxU32 firstContactFrame[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxU32 lastContactFrame[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxReal maxGap[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxReal maxPenetration[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxReal minReportedSeparation[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxReal maxReportedSeparation[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	PxReal maxTiltRadians;
	PxReal tailLinearSpeedSum;
	PxReal tailAngularSpeedSum;
	PxReal tailPeakLinearSpeed;
	PxReal tailPeakAngularSpeed;
	PxReal maxAbsPosition;
	PxU32 tailSamples;
	PxU32 firstAllSleepFrame;
	PxU32 finalSleepingCount;
	bool contactFinite[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
	bool finite;

	void reset()
	{
		actorCount = 0;
		maxTiltRadians = 0.0f;
		tailLinearSpeedSum = 0.0f;
		tailAngularSpeedSum = 0.0f;
		tailPeakLinearSpeed = 0.0f;
		tailPeakAngularSpeed = 0.0f;
		maxAbsPosition = 0.0f;
		tailSamples = 0;
		firstAllSleepFrame = PX_MAX_U32;
		finalSleepingCount = 0;
		finite = true;
		for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
		{
			contactFrames[i] = 0;
			contactPoints[i] = 0;
			firstContactFrame[i] = PX_MAX_U32;
			lastContactFrame[i] = PX_MAX_U32;
			maxGap[i] = 0.0f;
			maxPenetration[i] = 0.0f;
			minReportedSeparation[i] = PX_MAX_F32;
			maxReportedSeparation[i] = -PX_MAX_F32;
			contactFinite[i] = true;
		}
	}
};

struct SleepWitnessMetrics
{
	PxVec3 initialPosition;
	PxVec3 wakePosition;
	PxVec3 finalPosition;
	PxReal initialWakeCounter;
	PxReal finalWakeCounter;
	PxReal minWakeCounter;
	PxReal maxWakeCounter;
	PxReal maxLinearSpeed;
	PxReal maxAngularSpeed;
	PxReal maxAbsPosition;
	PxReal maxWakeVelocityX;
	PxReal maxWakeDisplacementX;
	PxReal minTopY;
	PxReal finalLinearSpeed;
	PxReal finalAngularSpeed;
	PxU32 firstSleepFrame;
	PxU32 firstSleepAfterWakeFrame;
	PxU32 firstAwakeAfterWakeFrame;
	PxU32 firstSleepNotificationFrame;
	PxU32 firstWakeNotificationFrame;
	PxU32 sleepSamples;
	PxU32 awakeSamples;
	PxU32 sleepNotifications;
	PxU32 wakeNotifications;
	PxU32 invalidWakeCounterSamples;
	bool finite;
	bool initialSleeping;
	bool finalSleeping;

	SleepWitnessMetrics() { reset(); }

	void reset()
	{
		initialPosition = PxVec3(0.0f);
		wakePosition = PxVec3(0.0f);
		finalPosition = PxVec3(0.0f);
		initialWakeCounter = 0.0f;
		finalWakeCounter = 0.0f;
		minWakeCounter = PX_MAX_F32;
		maxWakeCounter = 0.0f;
		maxLinearSpeed = 0.0f;
		maxAngularSpeed = 0.0f;
		maxAbsPosition = 0.0f;
		maxWakeVelocityX = 0.0f;
		maxWakeDisplacementX = 0.0f;
		minTopY = PX_MAX_F32;
		finalLinearSpeed = 0.0f;
		finalAngularSpeed = 0.0f;
		firstSleepFrame = PX_MAX_U32;
		firstSleepAfterWakeFrame = PX_MAX_U32;
		firstAwakeAfterWakeFrame = PX_MAX_U32;
		firstSleepNotificationFrame = PX_MAX_U32;
		firstWakeNotificationFrame = PX_MAX_U32;
		sleepSamples = 0;
		awakeSamples = 0;
		sleepNotifications = 0;
		wakeNotifications = 0;
		invalidWakeCounterSamples = 0;
		finite = true;
		initialSleeping = false;
		finalSleeping = false;
	}
};

struct GateEvaluation
{
	Snippets::HeadlessExitCode exitCode;
	const char* status;
	const char* reason;
	PxU32 boxCount;
	PxU32 targetBoxCount;
	PxReal maxEnergyRatio;
	PxReal tailEnergyW1;
	PxReal tailEnergyW2;
	PxReal tailEnergyW3;
	PxReal tailEnergyW4;
	PxReal tailEnergyGrowth;
	PxReal lateAwakeRatio;
	PxReal targetResponseMomentumRatio;
	PxReal ballResponseFraction;
	PxU32 levitatingBodies;

	GateEvaluation()
		: exitCode(Snippets::eHEADLESS_PASS)
		, status("PASS")
		, reason("none")
		, boxCount(0)
		, targetBoxCount(0)
		, maxEnergyRatio(0.0f)
		, tailEnergyW1(0.0f)
		, tailEnergyW2(0.0f)
		, tailEnergyW3(0.0f)
		, tailEnergyW4(0.0f)
		, tailEnergyGrowth(0.0f)
		, lateAwakeRatio(0.0f)
		, targetResponseMomentumRatio(0.0f)
		, ballResponseFraction(0.0f)
		, levitatingBodies(0)
	{
	}
};

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;

static Snippets::HeadlessOptions gOptions;
static HelloWorldHeadlessCase gCase = eHELLO_CASE_STACK_SETTLE;
static PxU32 gSimFrame = 0;

static std::vector<BoxRecord> gBoxes;
static std::vector<PxReal> gEnergyHistory;
static PxRigidDynamic* gShotBall = NULL;
static PxRigidDynamic* gSleepBodies[eHELLO_SLEEP_WITNESS_COUNT] = {NULL, NULL};
static PxRigidDynamic* gLockedBodies[eHELLO_LOCK_WITNESS_COUNT] = {};
static PxRigidDynamic* gLockControlBodies[eHELLO_LOCK_WITNESS_COUNT] = {};
static PxTransform gLockInitialPoses[eHELLO_LOCK_WITNESS_COUNT * 2];
static SleepWitnessMetrics gSleepWitnesses[eHELLO_SLEEP_WITNESS_COUNT];
static PxRigidStatic* gSubBounceGround = NULL;
static PxRigidDynamic* gSubBounceBodies[eHELLO_SUB_BOUNCE_BOX_COUNT] = {};
static SubBounceMetrics gSubBounce;

static PxU32 gCompletedFrames = 0;
static PxU32 gFetchFailures = 0;
static PxU32 gFetchErrorState = 0;
static PxU32 gLaunchFailures = 0;
static PxU32 gFinalAwakeBoxes = 0;
static PxU32 gLateAwakeSamples = 0;
static PxU32 gLateBoxSamples = 0;
static PxU32 gMaxSunkBoxes = 0;
static PxU32 gBallBoxContactEvents = 0;
static PxU32 gBallBoxContactPoints = 0;
static PxU32 gFirstBallBoxContactFrame = PX_MAX_U32;
static PxU32 gContactedBoxes = 0;
static PxU32 gRespondedBoxes = 0;
static PxU32 gSleepActorCount = 0;
static PxU32 gUnexpectedSleepNotifications = 0;
static PxU32 gFreeAwakeBeforeWakeSamples = 0;
static PxU32 gLockActorCount = 0;
static PxU32 gLockFlagsReadback = 0;
static PxU32 gLockRuntimeExcitations = 0;
static PxU32 gLockFiniteSamples = 0;
static PxU32 gLockNonFiniteSamples = 0;
static bool gNanDetected = false;
static bool gMechanicalReferenceSet = false;
static bool gWakeImpulseApplied = false;
static bool gSleepingBeforeWake = false;
static bool gAwakeImmediatelyAfterWake = false;
static bool gSceneSleepingDisabled = false;
static PxReal gMechanicalEnergyReference = 0.0f;
static PxReal gMaxMechanicalEnergyAfterReference = 0.0f;
static PxReal gMaxBoxSpeedSettle = 0.0f;
static PxReal gTailAvgBoxSpeed = 0.0f;
static PxU32 gTailSpeedSamples = 0;
static PxReal gMinBoxCenterY = 1e9f;
static PxReal gMaxBoxCenterY = -1e9f;
static PxReal gMinBallCenterY = PX_MAX_F32;
static PxReal gMaxAbsPosition = 0.0f;
static PxReal gMaxSpeedAll = 0.0f;
static PxReal gMaxAngularSpeed = 0.0f;
static PxReal gMaxQuaternionNormError = 0.0f;
static PxReal gBallLaunchMomentum = 0.0f;
static PxReal gMaxContactImpulse = 0.0f;
static PxReal gMaxTargetImpactAxisVelocityDelta = 0.0f;
static PxReal gMaxTargetResponseMomentum = 0.0f;
static PxReal gMaxTargetResponseDisplacement = 0.0f;
static PxReal gMaxBallImpactAxisVelocityDelta = 0.0f;
static PxReal gWakeCounterResetValue = 0.0f;
static PxReal gWakeCounterAfterImpulse = 0.0f;
static PxReal gLockedAxisMotion[eHELLO_LOCK_WITNESS_COUNT] = {};
static PxReal gLockedAxisSpeed[eHELLO_LOCK_WITNESS_COUNT] = {};
static PxReal gControlAxisMotion[eHELLO_LOCK_WITNESS_COUNT] = {};
static PxReal gControlAxisSpeed[eHELLO_LOCK_WITNESS_COUNT] = {};
static std::string gPoseLogPath;
static FILE* gPoseLog = NULL;
static PxReal gPoseMaxPen = 0.0f;
static PxU32 gPoseMaxPenFrame = 0;
static PxU32 gPoseMaxPenA = 0;
static PxU32 gPoseMaxPenB = 0;
static PxReal gPoseMaxJitter = 0.0f;
static PxU32 gPoseMaxJitterFrame = 0;
static PxU32 gPoseMaxJitterBox = 0;
static PxU32 gPoseFirstAllSleep = PX_MAX_U32;
static PxReal gPosePostSleepMaxDp = 0.0f;

static PxI32 findBoxIndex(const PxActor* actor)
{
	for(PxU32 i = 0; i < gBoxes.size(); ++i)
	{
		if(gBoxes[i].body == actor)
			return static_cast<PxI32>(i);
	}
	return -1;
}

static PxI32 findSleepWitnessIndex(const PxActor* actor)
{
	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		if(gSleepBodies[i] == actor)
			return static_cast<PxI32>(i);
	}
	return -1;
}

static PxI32 findSubBounceInterface(const PxActor* actor0, const PxActor* actor1)
{
	const PxActor* lowerActors[eHELLO_SUB_BOUNCE_INTERFACE_COUNT] = {
		gSubBounceGround, gSubBounceBodies[0], gSubBounceBodies[1]};
	const PxActor* upperActors[eHELLO_SUB_BOUNCE_INTERFACE_COUNT] = {
		gSubBounceBodies[0], gSubBounceBodies[1], gSubBounceBodies[2]};
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
	{
		if((actor0 == lowerActors[i] && actor1 == upperActors[i]) ||
			(actor1 == lowerActors[i] && actor0 == upperActors[i]))
			return static_cast<PxI32>(i);
	}
	return -1;
}

static PxU32 getSubBounceInterfaceMask()
{
	PxU32 mask = 0;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
		mask |= gSubBounce.contactPoints[i] ? (1u << i) : 0u;
	return mask;
}

static void recordSubBounceContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs,
	PxU32 pairCount)
{
	const PxI32 interfaceIndex = findSubBounceInterface(pairHeader.actors[0], pairHeader.actors[1]);
	if(interfaceIndex < 0)
		return;

	const PxU32 index = static_cast<PxU32>(interfaceIndex);
	PxU32 extractedForInterface = 0;
	for(PxU32 pairIndex = 0; pairIndex < pairCount; ++pairIndex)
	{
		if(pairs[pairIndex].flags &
			(PxContactPairFlag::eREMOVED_SHAPE_0 | PxContactPairFlag::eREMOVED_SHAPE_1))
			continue;
		PxContactPairPoint points[16];
		const PxU32 capacity = PxMin(PxU32(pairs[pairIndex].contactCount), PxU32(16));
		const PxU32 extracted = pairs[pairIndex].extractContacts(points, capacity);
		extractedForInterface += extracted;
		for(PxU32 pointIndex = 0; pointIndex < extracted; ++pointIndex)
		{
			const PxReal separation = points[pointIndex].separation;
			if(!PxIsFinite(separation))
			{
				gSubBounce.contactFinite[index] = false;
				gSubBounce.finite = false;
				continue;
			}
			gSubBounce.minReportedSeparation[index] =
				PxMin(gSubBounce.minReportedSeparation[index], separation);
			gSubBounce.maxReportedSeparation[index] =
				PxMax(gSubBounce.maxReportedSeparation[index], separation);
		}
	}
	if(!extractedForInterface)
		return;

	const PxU32 completedFrame = gSimFrame + 1u;
	gSubBounce.contactPoints[index] += extractedForInterface;
	if(gSubBounce.firstContactFrame[index] == PX_MAX_U32)
		gSubBounce.firstContactFrame[index] = completedFrame;
	if(gSubBounce.lastContactFrame[index] != completedFrame)
	{
		gSubBounce.lastContactFrame[index] = completedFrame;
		gSubBounce.contactFrames[index]++;
	}
}

class HelloWorldHeadlessCallback : public PxSimulationEventCallback
{
public:
	virtual void onConstraintBreak(PxConstraintInfo*, PxU32) PX_OVERRIDE {}
	virtual void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
	virtual void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) PX_OVERRIDE {}

	virtual void onWake(PxActor** actors, PxU32 count) PX_OVERRIDE
	{
		if(gCase != eHELLO_CASE_SLEEP_IDLE && gCase != eHELLO_CASE_SLEEP_WAKE &&
			gCase != eHELLO_CASE_SLEEP_DISABLED)
			return;
		for(PxU32 i = 0; i < count; ++i)
		{
			const PxI32 witness = findSleepWitnessIndex(actors[i]);
			if(witness >= 0)
			{
				gSleepWitnesses[witness].wakeNotifications++;
				if(gSleepWitnesses[witness].firstWakeNotificationFrame == PX_MAX_U32)
					gSleepWitnesses[witness].firstWakeNotificationFrame = gSimFrame + 1;
			}
			else
				gUnexpectedSleepNotifications++;
		}
	}

	virtual void onSleep(PxActor** actors, PxU32 count) PX_OVERRIDE
	{
		if(gCase != eHELLO_CASE_SLEEP_IDLE && gCase != eHELLO_CASE_SLEEP_WAKE &&
			gCase != eHELLO_CASE_SLEEP_DISABLED)
			return;
		for(PxU32 i = 0; i < count; ++i)
		{
			const PxI32 witness = findSleepWitnessIndex(actors[i]);
			if(witness >= 0)
			{
				gSleepWitnesses[witness].sleepNotifications++;
				if(gSleepWitnesses[witness].firstSleepNotificationFrame == PX_MAX_U32)
					gSleepWitnesses[witness].firstSleepNotificationFrame = gSimFrame + 1;
			}
			else
				gUnexpectedSleepNotifications++;
		}
	}

	virtual void onContact(const PxContactPairHeader& pairHeader, const PxContactPair* pairs,
		PxU32 pairCount) PX_OVERRIDE
	{
		if(pairHeader.flags &
			(PxContactPairHeaderFlag::eREMOVED_ACTOR_0 | PxContactPairHeaderFlag::eREMOVED_ACTOR_1))
			return;
		if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
		{
			recordSubBounceContact(pairHeader, pairs, pairCount);
			return;
		}
		if(gCase != eHELLO_CASE_BALL_SHOT)
			return;

		const PxActor* boxActor = NULL;
		if(pairHeader.actors[0] == gShotBall)
			boxActor = pairHeader.actors[1];
		else if(pairHeader.actors[1] == gShotBall)
			boxActor = pairHeader.actors[0];
		else
			return;

		const PxI32 boxIndex = findBoxIndex(boxActor);
		if(boxIndex < 0 || !gBoxes[static_cast<PxU32>(boxIndex)].target)
			return;

		gBallBoxContactEvents++;
		if(gFirstBallBoxContactFrame == PX_MAX_U32)
			gFirstBallBoxContactFrame = gSimFrame + 1;
		BoxRecord& rec = gBoxes[static_cast<PxU32>(boxIndex)];
		if(!rec.contacted)
		{
			rec.contacted = true;
			rec.contactVelBaseline = rec.prevVel;
			rec.contactPosBaseline = rec.prevPos;
			rec.contactFrame = gSimFrame + 1;
			gContactedBoxes++;
		}

		for(PxU32 pairIndex = 0; pairIndex < pairCount; ++pairIndex)
		{
			if(pairs[pairIndex].flags &
				(PxContactPairFlag::eREMOVED_SHAPE_0 | PxContactPairFlag::eREMOVED_SHAPE_1))
				continue;
			const PxU32 capacity = PxMin(PxU32(pairs[pairIndex].contactCount), PxU32(16));
			PxContactPairPoint points[16];
			const PxU32 extracted = pairs[pairIndex].extractContacts(points, capacity);
			gBallBoxContactPoints += extracted;
			for(PxU32 pointIndex = 0; pointIndex < extracted; ++pointIndex)
			{
				if(!points[pointIndex].impulse.isFinite())
				{
					gNanDetected = true;
					continue;
				}
				const PxReal impulse = points[pointIndex].impulse.magnitude();
				if(PxIsFinite(impulse) && impulse < PX_MAX_F32)
					gMaxContactImpulse = PxMax(gMaxContactImpulse, impulse);
				else
					gNanDetected = true;
			}
		}
	}
};

static HelloWorldHeadlessCallback gCallback;

static PxFilterFlags helloWorldHeadlessFilterShader(PxFilterObjectAttributes attributes0,
	PxFilterData filterData0, PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void*, PxU32)
{
	if(PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1))
	{
		pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
		return PxFilterFlag::eDEFAULT;
	}
	pairFlags = PxPairFlag::eCONTACT_DEFAULT;
	const bool ballAndBox =
		(filterData0.word0 == eHELLO_FILTER_BALL &&
			filterData1.word0 == eHELLO_FILTER_TARGET_BOX) ||
		(filterData1.word0 == eHELLO_FILTER_BALL &&
			filterData0.word0 == eHELLO_FILTER_TARGET_BOX);
	if(ballAndBox)
		pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_CONTACT_POINTS;
	const PxU32 tag0 = filterData0.word0;
	const PxU32 tag1 = filterData1.word0;
	const bool subBounceInterface =
		tag0 >= eHELLO_FILTER_SUB_BOUNCE_GROUND && tag0 <= eHELLO_FILTER_SUB_BOUNCE_BOX_2 &&
		tag1 >= eHELLO_FILTER_SUB_BOUNCE_GROUND && tag1 <= eHELLO_FILTER_SUB_BOUNCE_BOX_2 &&
		(tag0 + 1u == tag1 || tag1 + 1u == tag0);
	if(subBounceInterface)
	{
		pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
			PxPairFlag::eNOTIFY_CONTACT_POINTS;
	}
	return PxFilterFlag::eDEFAULT;
}

static bool tryParseHeadlessCase(const char* value, HelloWorldHeadlessCase& headlessCase)
{
	if(Snippets::equalsIgnoreCase(value, "stack-settle") ||
		Snippets::equalsIgnoreCase(value, "default"))
	{
		headlessCase = eHELLO_CASE_STACK_SETTLE;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "ball-shot"))
	{
		headlessCase = eHELLO_CASE_BALL_SHOT;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "sleep-idle"))
	{
		headlessCase = eHELLO_CASE_SLEEP_IDLE;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "sleep-wake"))
	{
		headlessCase = eHELLO_CASE_SLEEP_WAKE;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "sleep-disabled"))
	{
		headlessCase = eHELLO_CASE_SLEEP_DISABLED;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "lock-flags"))
	{
		headlessCase = eHELLO_CASE_LOCK_FLAGS;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "triple-ball-slide"))
	{
		headlessCase = eHELLO_CASE_TRIPLE_BALL_SLIDE;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "sub-bounce-stack"))
	{
		headlessCase = eHELLO_CASE_SUB_BOUNCE_STACK;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "rigid-stress") ||
		Snippets::equalsIgnoreCase(value, "hellogrb-cpu"))
	{
		headlessCase = eHELLO_CASE_RIGID_STRESS;
		return true;
	}
	return false;
}

static const char* getHeadlessCaseName(HelloWorldHeadlessCase headlessCase)
{
	switch(headlessCase)
	{
	case eHELLO_CASE_BALL_SHOT: return "ball-shot";
	case eHELLO_CASE_SLEEP_IDLE: return "sleep-idle";
	case eHELLO_CASE_SLEEP_WAKE: return "sleep-wake";
	case eHELLO_CASE_SLEEP_DISABLED: return "sleep-disabled";
	case eHELLO_CASE_LOCK_FLAGS: return "lock-flags";
	case eHELLO_CASE_TRIPLE_BALL_SLIDE: return "triple-ball-slide";
	case eHELLO_CASE_SUB_BOUNCE_STACK: return "sub-bounce-stack";
	case eHELLO_CASE_RIGID_STRESS: return "rigid-stress";
	case eHELLO_CASE_STACK_SETTLE:
	default: return "stack-settle";
	}
}

static bool isSleepProbeCase(HelloWorldHeadlessCase headlessCase)
{
	return headlessCase == eHELLO_CASE_SLEEP_IDLE ||
		headlessCase == eHELLO_CASE_SLEEP_WAKE ||
		headlessCase == eHELLO_CASE_SLEEP_DISABLED;
}

static bool isUnsupportedCase(HelloWorldHeadlessCase headlessCase)
{
	return headlessCase == eHELLO_CASE_TRIPLE_BALL_SLIDE ||
		headlessCase == eHELLO_CASE_RIGID_STRESS;
}

static const char* unsupportedReason(HelloWorldHeadlessCase headlessCase)
{
	if(headlessCase == eHELLO_CASE_RIGID_STRESS)
		return "not_ported_sceneDesc_avbd_fields";
	return "not_ported";
}

static void setFilterWord(PxRigidActor* actor, PxU32 word0)
{
	const PxU32 nb = actor->getNbShapes();
	std::vector<PxShape*> shapes(nb);
	if(!nb)
		return;
	actor->getShapes(shapes.data(), nb);
	PxFilterData fd;
	fd.word0 = word0;
	for(PxU32 i = 0; i < nb; ++i)
		shapes[i]->setSimulationFilterData(fd);
}

static void setActorContactOffset(PxRigidActor* actor, PxReal offset)
{
	const PxU32 nb = actor->getNbShapes();
	if(!nb)
		return;
	std::vector<PxShape*> shapes(nb);
	actor->getShapes(shapes.data(), nb);
	for(PxU32 i = 0; i < nb; ++i)
		shapes[i]->setContactOffset(offset);
}

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry,
	const PxVec3& velocity = PxVec3(0), PxU32 filterWord0 = 0)
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
	if(!dynamic)
		return NULL;
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	if(filterWord0)
		setFilterWord(dynamic, filterWord0);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent, bool targetStack)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	if(targetStack)
	{
		PxFilterData fd;
		fd.word0 = eHELLO_FILTER_TARGET_BOX;
		shape->setSimulationFilterData(fd);
	}
	for(PxU32 i = 0; i < size; i++)
	{
		for(PxU32 j = 0; j < size - i; j++)
		{
			PxTransform localTm(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			if(!body)
			{
				gInitializationFailed = true;
				continue;
			}
			body->attachShape(*shape);
			if(!PxRigidBodyExt::updateMassAndInertia(*body, 10.0f))
			{
				body->release();
				gInitializationFailed = true;
				continue;
			}
			gScene->addActor(*body);
			BoxRecord rec;
			rec.body = body;
			rec.target = targetStack;
			rec.contacted = false;
			rec.responded = false;
			rec.contactFrame = PX_MAX_U32;
			rec.prevVel = PxVec3(0.0f);
			rec.prevPos = body->getGlobalPose().p;
			rec.prevQ = body->getGlobalPose().q;
			rec.contactVelBaseline = PxVec3(0.0f);
			rec.contactPosBaseline = rec.prevPos;
			gBoxes.push_back(rec);
		}
	}
	shape->release();
}

static void createSubBounceStackFixture()
{
	const PxBoxGeometry geometry(gSubBounceBoxHalfExtent, gSubBounceBoxHalfExtent,
		gSubBounceBoxHalfExtent);
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_BOX_COUNT; ++i)
	{
		gSubBounceBodies[i] = createDynamic(
			PxTransform(PxVec3(0.0f, gSubBounceBoxHalfExtent + PxReal(i), 0.0f)), geometry,
			PxVec3(0.0f), eHELLO_FILTER_SUB_BOUNCE_BOX_0 + i);
		if(!gSubBounceBodies[i])
		{
			gInitializationFailed = true;
			continue;
		}
		gSubBounce.actorCount++;
	}
}

static void createSleepProbeFixture()
{
	const PxBoxGeometry geometry(gSleepBoxHalfExtent, gSleepBoxHalfExtent, gSleepBoxHalfExtent);
	if(gCase == eHELLO_CASE_SLEEP_WAKE)
	{
		gSleepBodies[eHELLO_SLEEP_FREE] =
			createDynamic(PxTransform(PxVec3(-0.49f, 4.0f, 0.0f)), geometry);
		gSleepBodies[eHELLO_SLEEP_STATIC_TOUCH] =
			createDynamic(PxTransform(PxVec3(0.49f, 4.0f, 0.0f)), geometry);
	}
	else
	{
		gSleepBodies[eHELLO_SLEEP_FREE] =
			createDynamic(PxTransform(PxVec3(-2.0f, 4.0f, 0.0f)), geometry);
		gSleepBodies[eHELLO_SLEEP_STATIC_TOUCH] = createDynamic(
			PxTransform(PxVec3(2.0f, gSleepBoxHalfExtent + 0.04f, 0.0f)), geometry);
	}

	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		PxRigidDynamic* body = gSleepBodies[i];
		if(!body)
		{
			gInitializationFailed = true;
			continue;
		}
		body->setActorFlag(PxActorFlag::eSEND_SLEEP_NOTIFIES, true);
		body->setLinearDamping(2.0f);
		body->setAngularDamping(gCase == eHELLO_CASE_SLEEP_WAKE ? 0.5f : 2.0f);
		setActorContactOffset(body, 0.05f);
		gSleepActorCount++;
	}

	if(gCase == eHELLO_CASE_SLEEP_WAKE)
	{
		for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
		{
			if(gSleepBodies[i])
				gSleepBodies[i]->putToSleep();
		}
	}

	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		PxRigidDynamic* body = gSleepBodies[i];
		if(!body)
			continue;
		SleepWitnessMetrics& metrics = gSleepWitnesses[i];
		metrics.reset();
		const PxTransform pose = body->getGlobalPose();
		metrics.initialPosition = pose.p;
		metrics.initialSleeping = body->isSleeping();
		metrics.initialWakeCounter = body->getWakeCounter();
		metrics.minWakeCounter = metrics.initialWakeCounter;
		metrics.maxWakeCounter = metrics.initialWakeCounter;
		if(metrics.initialSleeping)
			metrics.firstSleepFrame = 0;
		if(!pose.isValid() || !PxIsFinite(metrics.initialWakeCounter))
			metrics.finite = false;
	}
}

static PxVec3 getLockAxis(PxU32 witness)
{
	const PxU32 axis = witness % 3u;
	return axis == 0u ? PxVec3(1.0f, 0.0f, 0.0f)
		: axis == 1u ? PxVec3(0.0f, 1.0f, 0.0f)
		: PxVec3(0.0f, 0.0f, 1.0f);
}

static PxRigidDynamicLockFlag::Enum getLockFlag(PxU32 witness)
{
	static const PxRigidDynamicLockFlag::Enum flags[eHELLO_LOCK_WITNESS_COUNT] = {
		PxRigidDynamicLockFlag::eLOCK_LINEAR_X,
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Y,
		PxRigidDynamicLockFlag::eLOCK_LINEAR_Z,
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_X,
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y,
		PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z};
	return flags[witness];
}

static PxReal getOrientationDelta(const PxQuat& initial, const PxQuat& current)
{
	const PxReal cosine = PxMin(1.0f, PxMax(0.0f, PxAbs(initial.dot(current))));
	return 2.0f * PxAcos(cosine);
}

static void createLockProbeFixture()
{
	const PxSphereGeometry geometry(0.25f);
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		const PxReal x = -30.0f + PxReal(i) * 12.0f;
		PxRigidDynamic* locked = createDynamic(PxTransform(PxVec3(x, 20.0f, 0.0f)), geometry);
		PxRigidDynamic* control = createDynamic(PxTransform(PxVec3(x, 32.0f, 0.0f)), geometry);
		gLockedBodies[i] = locked;
		gLockControlBodies[i] = control;
		if(!locked || !control)
		{
			gInitializationFailed = true;
			continue;
		}

		const PxVec3 axis = getLockAxis(i);
		const PxRigidDynamicLockFlag::Enum flag = getLockFlag(i);
		PxRigidDynamic* bodies[2] = {locked, control};
		for(PxU32 bodyIndex = 0; bodyIndex < 2; ++bodyIndex)
		{
			PxRigidDynamic* body = bodies[bodyIndex];
			body->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
			body->setLinearDamping(0.0f);
			body->setAngularDamping(0.0f);
			body->setSleepThreshold(0.0f);
		}

		locked->setRigidDynamicLockFlag(flag, true);
		if(i < eHELLO_LOCK_ANGULAR_X)
		{
			locked->setLinearVelocity(axis * gLockLinearSpeed);
			control->setLinearVelocity(axis * gLockLinearSpeed);
		}
		else
		{
			locked->setAngularVelocity(axis * gLockAngularSpeed);
			control->setAngularVelocity(axis * gLockAngularSpeed);
		}

		gLockInitialPoses[i] = locked->getGlobalPose();
		gLockInitialPoses[i + eHELLO_LOCK_WITNESS_COUNT] = control->getGlobalPose();
		if(locked->getRigidDynamicLockFlags() == PxRigidDynamicLockFlags(flag) &&
			control->getRigidDynamicLockFlags() == PxRigidDynamicLockFlags())
			gLockFlagsReadback++;
		gLockActorCount += 2u;
	}
}

static void applyLockProbeRuntimeExcitation()
{
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		const PxVec3 axis = getLockAxis(i);
		PxRigidDynamic* bodies[2] = {gLockedBodies[i], gLockControlBodies[i]};
		for(PxU32 bodyIndex = 0; bodyIndex < 2; ++bodyIndex)
		{
			PxRigidDynamic* body = bodies[bodyIndex];
			if(!body)
				continue;
			if(i < eHELLO_LOCK_ANGULAR_X)
			{
				const PxReal impulse = body->getMass() * gLockLinearSpeed;
				body->addForce(axis * impulse, PxForceMode::eIMPULSE, true);
			}
			else
			{
				const PxReal angularImpulse =
					body->getMassSpaceInertiaTensor().dot(axis) * gLockAngularSpeed;
				body->addTorque(axis * angularImpulse, PxForceMode::eIMPULSE, true);
			}
			gLockRuntimeExcitations++;
		}
	}
}

static void spawnBallShot()
{
	if(gShotBall)
		return;
	gShotBall = createDynamic(PxTransform(gBallShotPos), PxSphereGeometry(gBallShotRadius),
		gBallShotVel, eHELLO_FILTER_BALL);
	if(!gShotBall)
	{
		gLaunchFailures++;
		return;
	}
	gBallLaunchMomentum = gShotBall->getMass() * PxAbs(gBallShotVel.z);
	printf("[HelloWorldBallShot] spawn pos=(%.1f,%.1f,%.1f) vel=(%.1f,%.1f,%.1f) radius=%.2f\n",
		gBallShotPos.x, gBallShotPos.y, gBallShotPos.z, gBallShotVel.x, gBallShotVel.y,
		gBallShotVel.z, gBallShotRadius);
}

static void applySleepWakeImpulse()
{
	PxRigidDynamic* body = gSleepBodies[eHELLO_SLEEP_FREE];
	if(!body || gWakeImpulseApplied)
		return;
	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		if(gSleepBodies[i])
			gSleepWitnesses[i].wakePosition = gSleepBodies[i]->getGlobalPose().p;
	}
	gSleepingBeforeWake = body->isSleeping();
	const PxReal impulse = body->getMass() * gSleepWakeTargetDeltaVelocity;
	body->addForce(PxVec3(impulse, 0.0f, 0.0f), PxForceMode::eIMPULSE, true);
	gWakeImpulseApplied = true;
	gAwakeImmediatelyAfterWake = !body->isSleeping();
	gWakeCounterAfterImpulse = body->getWakeCounter();
}

static PxReal saturateMetric(double value)
{
	if(!std::isfinite(value) || std::abs(value) >= double(PX_MAX_F32))
		return PX_MAX_F32;
	return PxReal(value);
}

static bool sampleBodyState(PxRigidDynamic* body, PxVec3& position, PxQuat& orientation,
	PxVec3& linearVelocity, PxReal& speed, PxReal& mechanicalEnergy)
{
	mechanicalEnergy = 0.0f;
	if(!body)
		return false;
	const PxTransform pose = body->getGlobalPose();
	position = pose.p;
	orientation = pose.q;
	linearVelocity = body->getLinearVelocity();
	const PxVec3 angularVelocity = body->getAngularVelocity();
	if(!pose.p.isFinite() || !pose.q.isFinite() || !linearVelocity.isFinite() ||
		!angularVelocity.isFinite())
		return false;

	const double quaternionNormSquared =
		double(pose.q.x) * double(pose.q.x) + double(pose.q.y) * double(pose.q.y) +
		double(pose.q.z) * double(pose.q.z) + double(pose.q.w) * double(pose.q.w);
	gMaxQuaternionNormError =
		PxMax(gMaxQuaternionNormError, saturateMetric(std::abs(quaternionNormSquared - 1.0)));
	gMaxAbsPosition = PxMax(gMaxAbsPosition,
		PxMax(PxAbs(position.x), PxMax(PxAbs(position.y), PxAbs(position.z))));
	speed = linearVelocity.magnitude();
	const PxReal angularSpeed = angularVelocity.magnitude();
	gMaxAngularSpeed = PxMax(gMaxAngularSpeed, angularSpeed);
	if(!PxIsFinite(speed) || !PxIsFinite(angularSpeed) || speed >= PX_MAX_F32 ||
		angularSpeed >= PX_MAX_F32)
		return false;

	const PxReal mass = body->getMass();
	const PxVec3 inertia = body->getMassSpaceInertiaTensor();
	const PxTransform comPose = pose * body->getCMassLocalPose();
	const PxQuat massFrame = pose.q * body->getCMassLocalPose().q;
	const PxVec3 massAngularVelocity = massFrame.rotateInv(angularVelocity);
	if(!PxIsFinite(mass) || mass <= 0.0f || !inertia.isFinite() || inertia.x <= 0.0f ||
		inertia.y <= 0.0f || inertia.z <= 0.0f || !comPose.p.isFinite() ||
		!massAngularVelocity.isFinite())
		return false;

	const double linearEnergy = 0.5 * double(mass) *
		(double(linearVelocity.x) * double(linearVelocity.x) +
			double(linearVelocity.y) * double(linearVelocity.y) +
			double(linearVelocity.z) * double(linearVelocity.z));
	const double angularEnergy =
		0.5 * (double(inertia.x) * double(massAngularVelocity.x) * double(massAngularVelocity.x) +
			double(inertia.y) * double(massAngularVelocity.y) * double(massAngularVelocity.y) +
			double(inertia.z) * double(massAngularVelocity.z) * double(massAngularVelocity.z));
	const double potentialEnergy = double(mass) * 9.81 * double(comPose.p.y);
	const double totalEnergy = linearEnergy + angularEnergy + potentialEnergy;
	if(!std::isfinite(totalEnergy) || std::abs(totalEnergy) >= double(PX_MAX_F32))
		return false;
	mechanicalEnergy = PxReal(totalEnergy);
	return true;
}

static void sampleSleepProbeAfterFetch(PxU32 frameIndex)
{
	const PxU32 completedFrame = frameIndex + 1;
	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		PxRigidDynamic* body = gSleepBodies[i];
		if(!body)
			continue;
		SleepWitnessMetrics& metrics = gSleepWitnesses[i];
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 linearVelocity = body->getLinearVelocity();
		const PxVec3 angularVelocity = body->getAngularVelocity();
		const PxReal wakeCounter = body->getWakeCounter();
		if(!pose.p.isFinite() || !pose.q.isFinite() || !linearVelocity.isFinite() ||
			!angularVelocity.isFinite() || !PxIsFinite(wakeCounter))
		{
			metrics.finite = false;
			gNanDetected = true;
			continue;
		}

		const PxReal linearSpeed = linearVelocity.magnitude();
		const PxReal angularSpeed = angularVelocity.magnitude();
		if(!PxIsFinite(linearSpeed) || !PxIsFinite(angularSpeed) || linearSpeed >= PX_MAX_F32 ||
			angularSpeed >= PX_MAX_F32)
		{
			metrics.finite = false;
			gNanDetected = true;
			continue;
		}

		metrics.maxLinearSpeed = PxMax(metrics.maxLinearSpeed, linearSpeed);
		metrics.maxAngularSpeed = PxMax(metrics.maxAngularSpeed, angularSpeed);
		metrics.maxAbsPosition = PxMax(metrics.maxAbsPosition,
			PxMax(PxAbs(pose.p.x), PxMax(PxAbs(pose.p.y), PxAbs(pose.p.z))));
		const PxMat33 rotation(pose.q);
		const PxReal supportY = gSleepBoxHalfExtent *
			(PxAbs(rotation.column0.y) + PxAbs(rotation.column1.y) + PxAbs(rotation.column2.y));
		metrics.minTopY = PxMin(metrics.minTopY, pose.p.y + supportY);
		metrics.minWakeCounter = PxMin(metrics.minWakeCounter, wakeCounter);
		metrics.maxWakeCounter = PxMax(metrics.maxWakeCounter, wakeCounter);
		if(wakeCounter < 0.0f)
			metrics.invalidWakeCounterSamples++;

		const bool sleeping = body->isSleeping();
		if(gCase == eHELLO_CASE_SLEEP_WAKE && !gWakeImpulseApplied && i == eHELLO_SLEEP_FREE &&
			!sleeping)
			gFreeAwakeBeforeWakeSamples++;
		if(sleeping)
		{
			metrics.sleepSamples++;
			if(metrics.firstSleepFrame == PX_MAX_U32)
				metrics.firstSleepFrame = completedFrame;
			if(gWakeImpulseApplied && metrics.firstSleepAfterWakeFrame == PX_MAX_U32)
				metrics.firstSleepAfterWakeFrame = completedFrame;
		}
		else
		{
			metrics.awakeSamples++;
			if(gWakeImpulseApplied && metrics.firstAwakeAfterWakeFrame == PX_MAX_U32)
				metrics.firstAwakeAfterWakeFrame = completedFrame;
		}
		metrics.finalSleeping = sleeping;
		metrics.finalPosition = pose.p;
		metrics.finalLinearSpeed = linearSpeed;
		metrics.finalAngularSpeed = angularSpeed;
		metrics.finalWakeCounter = wakeCounter;
		if(gWakeImpulseApplied)
		{
			metrics.maxWakeVelocityX = PxMax(metrics.maxWakeVelocityX, linearVelocity.x);
			metrics.maxWakeDisplacementX =
				PxMax(metrics.maxWakeDisplacementX, pose.p.x - metrics.wakePosition.x);
		}
	}
	gCompletedFrames = completedFrame;
}

static PxReal getBoxVerticalSupport(const PxQuat& orientation, PxReal halfExtent)
{
	const PxMat33 rotation(orientation);
	return halfExtent *
		(PxAbs(rotation.column0.y) + PxAbs(rotation.column1.y) + PxAbs(rotation.column2.y));
}

static PxU32 countLevitatingHelloWorldBoxes()
{
	const PxU32 n = static_cast<PxU32>(gBoxes.size());
	if(!n)
		return 0;
	const PxReal groundTol = 0.05f;
	const PxReal stackGapTol = 0.15f;
	std::vector<PxReal> bottom(n), top(n), minX(n), maxX(n), minZ(n), maxZ(n);
	std::vector<PxU8> valid(n, 0);
	for(PxU32 i = 0; i < n; ++i)
	{
		PxRigidDynamic* body = gBoxes[i].body;
		if(!body)
			continue;
		const PxTransform pose = body->getGlobalPose();
		if(!pose.p.isFinite() || !pose.q.isFinite())
			continue;
		const PxMat33 r(pose.q);
		const PxReal he = gBoxHalfExtent;
		const PxReal supportY = he * (PxAbs(r.column0.y) + PxAbs(r.column1.y) + PxAbs(r.column2.y));
		bottom[i] = pose.p.y - supportY;
		top[i] = pose.p.y + supportY;
		const PxReal extX = he * (PxAbs(r.column0.x) + PxAbs(r.column1.x) + PxAbs(r.column2.x));
		const PxReal extZ = he * (PxAbs(r.column0.z) + PxAbs(r.column1.z) + PxAbs(r.column2.z));
		minX[i] = pose.p.x - extX;
		maxX[i] = pose.p.x + extX;
		minZ[i] = pose.p.z - extZ;
		maxZ[i] = pose.p.z + extZ;
		valid[i] = 1;
	}
	PxU32 levitating = 0;
	for(PxU32 i = 0; i < n; ++i)
	{
		if(!valid[i] || bottom[i] <= groundTol)
			continue;
		bool supported = false;
		for(PxU32 j = 0; j < n && !supported; ++j)
		{
			if(i == j || !valid[j])
				continue;
			if(maxX[i] < minX[j] || maxX[j] < minX[i] || maxZ[i] < minZ[j] || maxZ[j] < minZ[i])
				continue;
			const PxReal gap = bottom[i] - top[j];
			if(gap >= -groundTol && gap <= stackGapTol)
				supported = true;
		}
		if(!supported)
			levitating++;
	}
	return levitating;
}

static void sampleSubBounceStackAfterFetch(PxU32 frameIndex)
{
	PxTransform poses[eHELLO_SUB_BOUNCE_BOX_COUNT] = {
		PxTransform(PxIdentity), PxTransform(PxIdentity), PxTransform(PxIdentity)};
	PxReal supports[eHELLO_SUB_BOUNCE_BOX_COUNT] = {};
	PxReal frameMaxLinearSpeed = 0.0f;
	PxReal frameMaxAngularSpeed = 0.0f;
	PxU32 sleepingCount = 0;
	bool frameFinite = true;

	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_BOX_COUNT; ++i)
	{
		PxRigidDynamic* body = gSubBounceBodies[i];
		if(!body)
		{
			frameFinite = false;
			continue;
		}
		poses[i] = body->getGlobalPose();
		const PxVec3 linearVelocity = body->getLinearVelocity();
		const PxVec3 angularVelocity = body->getAngularVelocity();
		if(!poses[i].isValid() || !linearVelocity.isFinite() || !angularVelocity.isFinite())
		{
			frameFinite = false;
			continue;
		}

		const PxReal linearSpeed = linearVelocity.magnitude();
		const PxReal angularSpeed = angularVelocity.magnitude();
		if(!PxIsFinite(linearSpeed) || !PxIsFinite(angularSpeed) || linearSpeed >= PX_MAX_F32 ||
			angularSpeed >= PX_MAX_F32)
		{
			frameFinite = false;
			continue;
		}
		frameMaxLinearSpeed = PxMax(frameMaxLinearSpeed, linearSpeed);
		frameMaxAngularSpeed = PxMax(frameMaxAngularSpeed, angularSpeed);
		gSubBounce.maxAbsPosition = PxMax(gSubBounce.maxAbsPosition,
			PxMax(PxAbs(poses[i].p.x), PxMax(PxAbs(poses[i].p.y), PxAbs(poses[i].p.z))));
		supports[i] = getBoxVerticalSupport(poses[i].q, gSubBounceBoxHalfExtent);
		const PxVec3 worldUp = poses[i].q.rotate(PxVec3(0.0f, 1.0f, 0.0f));
		const PxReal upDot = PxClamp(worldUp.y, -1.0f, 1.0f);
		gSubBounce.maxTiltRadians =
			PxMax(gSubBounce.maxTiltRadians, PxReal(std::acos(double(upDot))));
		sleepingCount += body->isSleeping() ? 1u : 0u;
	}

	if(frameFinite)
	{
		PxReal gaps[eHELLO_SUB_BOUNCE_INTERFACE_COUNT];
		gaps[0] = poses[0].p.y - supports[0];
		gaps[1] = (poses[1].p.y - supports[1]) - (poses[0].p.y + supports[0]);
		gaps[2] = (poses[2].p.y - supports[2]) - (poses[1].p.y + supports[1]);
		for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
		{
			if(!PxIsFinite(gaps[i]))
			{
				frameFinite = false;
				break;
			}
			gSubBounce.maxGap[i] = PxMax(gSubBounce.maxGap[i], PxMax(0.0f, gaps[i]));
			gSubBounce.maxPenetration[i] =
				PxMax(gSubBounce.maxPenetration[i], PxMax(0.0f, -gaps[i]));
		}
	}

	if(frameIndex + gSubBounceTailFrames >= gOptions.frames)
	{
		gSubBounce.tailLinearSpeedSum =
			saturateMetric(double(gSubBounce.tailLinearSpeedSum) + double(frameMaxLinearSpeed));
		gSubBounce.tailAngularSpeedSum =
			saturateMetric(double(gSubBounce.tailAngularSpeedSum) + double(frameMaxAngularSpeed));
		gSubBounce.tailPeakLinearSpeed =
			PxMax(gSubBounce.tailPeakLinearSpeed, frameMaxLinearSpeed);
		gSubBounce.tailPeakAngularSpeed =
			PxMax(gSubBounce.tailPeakAngularSpeed, frameMaxAngularSpeed);
		gSubBounce.tailSamples++;
	}

	if(sleepingCount == eHELLO_SUB_BOUNCE_BOX_COUNT &&
		gSubBounce.firstAllSleepFrame == PX_MAX_U32)
		gSubBounce.firstAllSleepFrame = frameIndex + 1u;
	gSubBounce.finalSleepingCount = sleepingCount;
	gSubBounce.finite = gSubBounce.finite && frameFinite;
	if(!frameFinite)
		gNanDetected = true;
	gCompletedFrames = frameIndex + 1u;
}

static void sampleLockProbeAfterFetch(PxU32 frameIndex)
{
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		PxRigidDynamic* bodies[2] = {gLockedBodies[i], gLockControlBodies[i]};
		for(PxU32 bodyIndex = 0; bodyIndex < 2; ++bodyIndex)
		{
			PxRigidDynamic* body = bodies[bodyIndex];
			if(!body)
			{
				gLockNonFiniteSamples++;
				continue;
			}
			const PxTransform pose = body->getGlobalPose();
			const PxVec3 linearVelocity = body->getLinearVelocity();
			const PxVec3 angularVelocity = body->getAngularVelocity();
			if(!pose.isValid() || !linearVelocity.isFinite() || !angularVelocity.isFinite())
			{
				gLockNonFiniteSamples++;
				continue;
			}
			gLockFiniteSamples++;
			const PxVec3 axis = getLockAxis(i);
			const PxTransform& initial =
				gLockInitialPoses[i + bodyIndex * eHELLO_LOCK_WITNESS_COUNT];
			const bool angular = i >= eHELLO_LOCK_ANGULAR_X;
			const PxReal motion = angular ? getOrientationDelta(initial.q, pose.q)
										  : PxAbs((pose.p - initial.p).dot(axis));
			const PxReal speed = angular ? PxAbs(angularVelocity.dot(axis))
										 : PxAbs(linearVelocity.dot(axis));
			PxReal* motions = bodyIndex == 0u ? gLockedAxisMotion : gControlAxisMotion;
			PxReal* speeds = bodyIndex == 0u ? gLockedAxisSpeed : gControlAxisSpeed;
			motions[i] = PxMax(motions[i], motion);
			speeds[i] = PxMax(speeds[i], speed);
		}
	}
	gCompletedFrames = frameIndex + 1u;
}

static void closePoseLog()
{
	if(!gPoseLog)
		return;
	printf("[POSE_SUMMARY] maxPen=%.6f frame=%u pair=%u,%u maxJitter=%.6f "
		"jitterFrame=%u jitterBox=%u firstAllSleep=%u postSleepMaxDp=%.6f\n",
		double(gPoseMaxPen), gPoseMaxPenFrame, gPoseMaxPenA, gPoseMaxPenB,
		double(gPoseMaxJitter), gPoseMaxJitterFrame, gPoseMaxJitterBox,
		gPoseFirstAllSleep, double(gPosePostSleepMaxDp));
	fclose(gPoseLog);
	gPoseLog = NULL;
}

static bool openPoseLog()
{
	if(gPoseLogPath.empty())
	{
		const char* env = std::getenv("AVBD_POSE_LOG");
		if(env && env[0])
			gPoseLogPath = env;
	}
	if(gPoseLogPath.empty())
		return true;
	gPoseLog = fopen(gPoseLogPath.c_str(), "w");
	if(!gPoseLog)
	{
		printf("[POSE] failed to open %s\n", gPoseLogPath.c_str());
		return false;
	}
	fprintf(gPoseLog,
		"frame,id,sleep,x,y,z,qx,qy,qz,qw,lx,ly,lz,ax,ay,az,dp,dang,pen\n");
	printf("[POSE] writing %s\n", gPoseLogPath.c_str());
	gPoseMaxPen = 0.0f;
	gPoseMaxPenFrame = 0;
	gPoseMaxPenA = 0;
	gPoseMaxPenB = 0;
	gPoseMaxJitter = 0.0f;
	gPoseMaxJitterFrame = 0;
	gPoseMaxJitterBox = 0;
	gPoseFirstAllSleep = PX_MAX_U32;
	gPosePostSleepMaxDp = 0.0f;
	return true;
}

static PxReal aabbOverlapPen(PxReal minA, PxReal maxA, PxReal minB, PxReal maxB)
{
	return PxMin(maxA, maxB) - PxMax(minA, minB);
}

static void logBoxPoses(PxU32 frameIndex)
{
	if(!gPoseLog || gBoxes.empty())
		return;
	const PxU32 n = static_cast<PxU32>(gBoxes.size());
	const PxReal he = gBoxHalfExtent;
	std::vector<PxVec3> p(n), lin(n), ang(n);
	std::vector<PxQuat> q(n);
	std::vector<PxReal> minX(n), maxX(n), minY(n), maxY(n), minZ(n), maxZ(n);
	std::vector<PxU8> sleep(n, 1);
	std::vector<PxReal> dp(n, 0.0f), dang(n, 0.0f);
	PxReal maxDp = 0.0f;
	PxU32 maxDpBox = 0;
	PxReal maxAng = 0.0f;
	for(PxU32 i = 0; i < n; ++i)
	{
		PxRigidDynamic* body = gBoxes[i].body;
		if(!body)
			continue;
		const PxTransform pose = body->getGlobalPose();
		p[i] = pose.p;
		q[i] = pose.q;
		lin[i] = body->getLinearVelocity();
		ang[i] = body->getAngularVelocity();
		sleep[i] = body->isSleeping() ? 1 : 0;
		const PxMat33 r(pose.q);
		const PxReal ex = he * (PxAbs(r.column0.x) + PxAbs(r.column1.x) + PxAbs(r.column2.x));
		const PxReal ey = he * (PxAbs(r.column0.y) + PxAbs(r.column1.y) + PxAbs(r.column2.y));
		const PxReal ez = he * (PxAbs(r.column0.z) + PxAbs(r.column1.z) + PxAbs(r.column2.z));
		minX[i] = pose.p.x - ex;
		maxX[i] = pose.p.x + ex;
		minY[i] = pose.p.y - ey;
		maxY[i] = pose.p.y + ey;
		minZ[i] = pose.p.z - ez;
		maxZ[i] = pose.p.z + ez;
		if(frameIndex > 0)
		{
			dp[i] = (pose.p - gBoxes[i].prevPos).magnitude();
			PxQuat dq = pose.q * gBoxes[i].prevQ.getConjugate();
			if(dq.w < 0.0f)
				dq = -dq;
			const PxReal w = PxClamp(dq.w, -1.0f, 1.0f);
			dang[i] = 2.0f * PxAcos(w);
		}
		if(dp[i] > maxDp)
		{
			maxDp = dp[i];
			maxDpBox = i;
		}
		maxAng = PxMax(maxAng, ang[i].magnitude());
		if(dp[i] > gPoseMaxJitter)
		{
			gPoseMaxJitter = dp[i];
			gPoseMaxJitterFrame = frameIndex;
			gPoseMaxJitterBox = i;
		}
	}

	PxU32 overlapPairs = 0;
	PxReal frameMaxPen = 0.0f;
	PxU32 penA = 0;
	PxU32 penB = 0;
	std::vector<PxReal> boxPen(n, 0.0f);
	const PxReal contactSlop = 0.02f;
	for(PxU32 i = 0; i < n; ++i)
	{
		if(!gBoxes[i].body)
			continue;
		for(PxU32 j = i + 1; j < n; ++j)
		{
			if(!gBoxes[j].body)
				continue;
			const PxReal ox = aabbOverlapPen(minX[i], maxX[i], minX[j], maxX[j]);
			const PxReal oy = aabbOverlapPen(minY[i], maxY[i], minY[j], maxY[j]);
			const PxReal oz = aabbOverlapPen(minZ[i], maxZ[i], minZ[j], maxZ[j]);
			if(ox <= contactSlop || oy <= contactSlop || oz <= contactSlop)
				continue;
			const PxReal pen = PxMin(ox, PxMin(oy, oz));
			++overlapPairs;
			boxPen[i] = PxMax(boxPen[i], pen);
			boxPen[j] = PxMax(boxPen[j], pen);
			if(pen > frameMaxPen)
			{
				frameMaxPen = pen;
				penA = i;
				penB = j;
			}
		}
	}
	if(frameMaxPen > gPoseMaxPen)
	{
		gPoseMaxPen = frameMaxPen;
		gPoseMaxPenFrame = frameIndex;
		gPoseMaxPenA = penA;
		gPoseMaxPenB = penB;
	}
	PxU32 awakeNow = 0;
	for(PxU32 i = 0; i < n; ++i)
	{
		if(gBoxes[i].body && !sleep[i])
			++awakeNow;
	}
	if(awakeNow == 0 && gPoseFirstAllSleep == PX_MAX_U32)
		gPoseFirstAllSleep = frameIndex;
	if(gPoseFirstAllSleep != PX_MAX_U32 && frameIndex > gPoseFirstAllSleep)
		gPosePostSleepMaxDp = PxMax(gPosePostSleepMaxDp, maxDp);

	PxReal yMin = 1e9f;
	PxReal yMax = -1e9f;
	PxReal maxLin = 0.0f;
	for(PxU32 i = 0; i < n; ++i)
	{
		if(!gBoxes[i].body)
			continue;
		yMin = PxMin(yMin, p[i].y);
		yMax = PxMax(yMax, p[i].y);
		maxLin = PxMax(maxLin, lin[i].magnitude());
		fprintf(gPoseLog,
			"%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
			frameIndex, i, unsigned(sleep[i]),
			double(p[i].x), double(p[i].y), double(p[i].z),
			double(q[i].x), double(q[i].y), double(q[i].z), double(q[i].w),
			double(lin[i].x), double(lin[i].y), double(lin[i].z),
			double(ang[i].x), double(ang[i].y), double(ang[i].z),
			double(dp[i]), double(dang[i]), double(boxPen[i]));
	}

	if(frameIndex < 20u || (frameIndex % 10u) == 0u || frameMaxPen > 0.05f || maxDp > 0.02f)
	{
		printf("[POSE] f=%u awake=%u maxSpeed=%.4f maxDp=%.4f box=%u maxW=%.4f "
			"overlap=%u maxPen=%.4f pair=%u,%u y=[%.3f,%.3f]\n",
			frameIndex, awakeNow, double(maxLin),
			double(maxDp), maxDpBox, double(maxAng), overlapPairs, double(frameMaxPen),
			penA, penB, double(yMin), double(yMax));
	}
}

static void sampleDynamics(PxU32 frameIndex, PxU32 settleTailFrames)
{
	PxReal maxBoxSpeed = 0.0f;
	PxU32 awakeBoxes = 0;
	PxU32 sunkBoxes = 0;
	double totalMechanicalEnergy = 0.0;
	const PxU32 currentFrame = frameIndex + 1;
	const bool inSettleWindow = frameIndex + settleTailFrames >= gOptions.frames;

	if(gPoseLog)
		logBoxPoses(frameIndex);

	for(PxU32 boxIndex = 0; boxIndex < gBoxes.size(); ++boxIndex)
	{
		BoxRecord& rec = gBoxes[boxIndex];
		PxVec3 position(0.0f), velocity(0.0f);
		PxQuat orientation(PxIdentity);
		PxReal speed = 0.0f, bodyEnergy = 0.0f;
		if(!sampleBodyState(rec.body, position, orientation, velocity, speed, bodyEnergy))
		{
			gNanDetected = true;
			continue;
		}
		totalMechanicalEnergy += double(bodyEnergy);
		gMaxSpeedAll = PxMax(gMaxSpeedAll, speed);
		gMaxBoxCenterY = PxMax(gMaxBoxCenterY, position.y);
		gMinBoxCenterY = PxMin(gMinBoxCenterY, position.y);
		maxBoxSpeed = PxMax(maxBoxSpeed, speed);
		if(!rec.body->isSleeping())
			awakeBoxes++;
		const PxMat33 rotation(orientation);
		const PxReal supportY = gBoxHalfExtent *
			(PxAbs(rotation.column0.y) + PxAbs(rotation.column1.y) + PxAbs(rotation.column2.y));
		if(position.y + supportY < -0.05f)
			sunkBoxes++;

		if(rec.contacted && currentFrame >= rec.contactFrame &&
			currentFrame < rec.contactFrame + gBoxResponseWindowFrames)
		{
			const PxReal responseAxisDelta =
				PxMax(0.0f, saturateMetric(double(rec.contactVelBaseline.z) - double(velocity.z)));
			const PxReal responseDisplacement =
				PxMax(0.0f, saturateMetric(double(rec.contactPosBaseline.z) - double(position.z)));
			const PxReal responseMomentum =
				saturateMetric(double(rec.body->getMass()) * double(responseAxisDelta));
			gMaxTargetImpactAxisVelocityDelta =
				PxMax(gMaxTargetImpactAxisVelocityDelta, responseAxisDelta);
			gMaxTargetResponseMomentum = PxMax(gMaxTargetResponseMomentum, responseMomentum);
			gMaxTargetResponseDisplacement =
				PxMax(gMaxTargetResponseDisplacement, responseDisplacement);
			const PxReal responseMomentumRatio =
				gBallLaunchMomentum > 0.0f ? responseMomentum / gBallLaunchMomentum : 0.0f;
			if(!rec.responded && responseMomentumRatio >= gMinTargetResponseMomentumRatio &&
				responseDisplacement >= gMinTargetResponseDisplacement)
			{
				rec.responded = true;
				gRespondedBoxes++;
			}
		}
		rec.prevVel = velocity;
		rec.prevPos = position;
		rec.prevQ = orientation;
	}
	gFinalAwakeBoxes = awakeBoxes;
	gMaxSunkBoxes = PxMax(gMaxSunkBoxes, sunkBoxes);

	if(gShotBall)
	{
		PxVec3 position(0.0f), velocity(0.0f);
		PxQuat orientation(PxIdentity);
		PxReal speed = 0.0f, ballEnergy = 0.0f;
		if(!sampleBodyState(gShotBall, position, orientation, velocity, speed, ballEnergy))
			gNanDetected = true;
		else
		{
			totalMechanicalEnergy += double(ballEnergy);
			gMaxSpeedAll = PxMax(gMaxSpeedAll, speed);
			gMinBallCenterY = PxMin(gMinBallCenterY, position.y);
			if(gFirstBallBoxContactFrame != PX_MAX_U32 &&
				currentFrame >= gFirstBallBoxContactFrame &&
				currentFrame < gFirstBallBoxContactFrame + gBallResponseWindowFrames)
			{
				const PxReal impactAxisDelta =
					saturateMetric(double(velocity.z) - double(gBallShotVel.z));
				gMaxBallImpactAxisVelocityDelta =
					PxMax(gMaxBallImpactAxisVelocityDelta, PxMax(0.0f, impactAxisDelta));
			}
		}
	}

	PxReal currentMechanicalEnergy = 0.0f;
	if(!std::isfinite(totalMechanicalEnergy) || std::abs(totalMechanicalEnergy) >= double(PX_MAX_F32))
	{
		gNanDetected = true;
		currentMechanicalEnergy = PX_MAX_F32;
	}
	else
		currentMechanicalEnergy = PxReal(totalMechanicalEnergy);

	const bool establishEnergyReference = !gMechanicalReferenceSet &&
		((gCase != eHELLO_CASE_BALL_SHOT && frameIndex == 0) ||
			(gCase == eHELLO_CASE_BALL_SHOT && gShotBall && frameIndex >= gBallShotFrame));
	if(establishEnergyReference)
	{
		gMechanicalReferenceSet = true;
		gMechanicalEnergyReference = currentMechanicalEnergy;
		gMaxMechanicalEnergyAfterReference = currentMechanicalEnergy;
	}
	if(gMechanicalReferenceSet)
		gMaxMechanicalEnergyAfterReference =
			PxMax(gMaxMechanicalEnergyAfterReference, currentMechanicalEnergy);
	gEnergyHistory.push_back(currentMechanicalEnergy);

	if(inSettleWindow && !gBoxes.empty())
	{
		gMaxBoxSpeedSettle = PxMax(gMaxBoxSpeedSettle, maxBoxSpeed);
		gTailAvgBoxSpeed += maxBoxSpeed;
		gTailSpeedSamples++;
		gLateBoxSamples += static_cast<PxU32>(gBoxes.size());
		gLateAwakeSamples += awakeBoxes;
	}
	gCompletedFrames = currentFrame;
}

static void resetRuntimeState()
{
	gBoxes.clear();
	gEnergyHistory.clear();
	gShotBall = NULL;
	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		gSleepBodies[i] = NULL;
		gSleepWitnesses[i].reset();
	}
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		gLockedBodies[i] = NULL;
		gLockControlBodies[i] = NULL;
		gLockedAxisMotion[i] = 0.0f;
		gLockedAxisSpeed[i] = 0.0f;
		gControlAxisMotion[i] = 0.0f;
		gControlAxisSpeed[i] = 0.0f;
	}
	gSubBounceGround = NULL;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_BOX_COUNT; ++i)
		gSubBounceBodies[i] = NULL;
	gSubBounce.reset();
	gSimFrame = 0;
	gCompletedFrames = 0;
	gFetchFailures = 0;
	gFetchErrorState = 0;
	gLaunchFailures = 0;
	gFinalAwakeBoxes = 0;
	gLateAwakeSamples = 0;
	gLateBoxSamples = 0;
	gMaxSunkBoxes = 0;
	gBallBoxContactEvents = 0;
	gBallBoxContactPoints = 0;
	gFirstBallBoxContactFrame = PX_MAX_U32;
	gContactedBoxes = 0;
	gRespondedBoxes = 0;
	gSleepActorCount = 0;
	gUnexpectedSleepNotifications = 0;
	gFreeAwakeBeforeWakeSamples = 0;
	gLockActorCount = 0;
	gLockFlagsReadback = 0;
	gLockRuntimeExcitations = 0;
	gLockFiniteSamples = 0;
	gLockNonFiniteSamples = 0;
	gNanDetected = false;
	gMechanicalReferenceSet = false;
	gWakeImpulseApplied = false;
	gSleepingBeforeWake = false;
	gAwakeImmediatelyAfterWake = false;
	gSceneSleepingDisabled = false;
	gMechanicalEnergyReference = 0.0f;
	gMaxMechanicalEnergyAfterReference = 0.0f;
	gMaxBoxSpeedSettle = 0.0f;
	gTailAvgBoxSpeed = 0.0f;
	gTailSpeedSamples = 0;
	gMinBoxCenterY = 1e9f;
	gMaxBoxCenterY = -1e9f;
	gMinBallCenterY = PX_MAX_F32;
	gMaxAbsPosition = 0.0f;
	gMaxSpeedAll = 0.0f;
	gMaxAngularSpeed = 0.0f;
	gMaxQuaternionNormError = 0.0f;
	gBallLaunchMomentum = 0.0f;
	gMaxContactImpulse = 0.0f;
	gMaxTargetImpactAxisVelocityDelta = 0.0f;
	gMaxTargetResponseMomentum = 0.0f;
	gMaxTargetResponseDisplacement = 0.0f;
	gMaxBallImpactAxisVelocityDelta = 0.0f;
	gWakeCounterResetValue = 0.0f;
	gWakeCounterAfterImpulse = 0.0f;
	gInitializationFailed = false;
	gExtensionsInitialized = false;
}

static bool initHeadlessPhysics()
{
	resetRuntimeState();
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, NULL);
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, NULL);
	if(!gExtensionsInitialized)
		return false;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = isSleepProbeCase(gCase) || gCase == eHELLO_CASE_LOCK_FLAGS
		? PxVec3(0.0f)
		: PxVec3(0.0f, -9.81f, 0.0f);
	if(gCase == eHELLO_CASE_SLEEP_DISABLED)
		sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
	if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
		sceneDesc.bounceThresholdVelocity = 2.0f;
	gDispatcher = PxDefaultCpuDispatcherCreate(gOptions.dispatcherThreads);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = helloWorldHeadlessFilterShader;
	sceneDesc.simulationEventCallback = &gCallback;
	sceneDesc.solverType = gOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;
	gSceneSleepingDisabled = gScene->getFlags() & PxSceneFlag::eDISABLE_SLEEPING;
	gWakeCounterResetValue = gScene->getWakeCounterResetValue();

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, isSleepProbeCase(gCase) ? 0.0f : 0.6f);
	if(!gMaterial)
		return false;
	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	if(!groundPlane)
		return false;
	gScene->addActor(*groundPlane);
	if(isSleepProbeCase(gCase))
		setActorContactOffset(groundPlane, 0.05f);
	if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
	{
		gSubBounceGround = groundPlane;
		setFilterWord(groundPlane, eHELLO_FILTER_SUB_BOUNCE_GROUND);
	}

	if(isSleepProbeCase(gCase))
		createSleepProbeFixture();
	else if(gCase == eHELLO_CASE_LOCK_FLAGS)
		createLockProbeFixture();
	else if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
	{
		createSubBounceStackFixture();
		printf("[HelloWorld] init solver=%s subBounceStack=1 boxes=%u "
			"boxSize=1 restitution=0.6 bounceThresholdVelocity=2 "
			"gravity=-9.81 ground=plane\n",
			Snippets::getSolverTypeName(gOptions.solverType), gSubBounce.actorCount);
	}
	else
	{
		PxReal stackZ = 10.0f;
		for(PxU32 i = 0; i < gHelloWorldStackCount; i++)
			createStack(PxTransform(PxVec3(0, 0, stackZ -= 10.0f)), 10, gBoxHalfExtent, i == 0);
	}
	return !gInitializationFailed;
}

static void stepHeadlessPhysics()
{
	if(!gScene)
		return;
	if(gCase == eHELLO_CASE_BALL_SHOT && gSimFrame == gBallShotFrame)
		spawnBallShot();
	if(gCase == eHELLO_CASE_SLEEP_WAKE && gSimFrame == gSleepWakeFrame)
		applySleepWakeImpulse();
	if(gCase == eHELLO_CASE_LOCK_FLAGS && gSimFrame == gLockImpulseFrame)
		applyLockProbeRuntimeExcitation();

	PxU32 errorState = 0;
	gScene->simulate(gOptions.dt);
	const bool fetched = gScene->fetchResults(true, &errorState);
	if(!fetched)
	{
		gFetchFailures++;
		gFetchErrorState |= errorState;
		return;
	}
	if(errorState)
		gFetchErrorState |= errorState;
	if(isSleepProbeCase(gCase))
		sampleSleepProbeAfterFetch(gSimFrame);
	else if(gCase == eHELLO_CASE_LOCK_FLAGS)
		sampleLockProbeAfterFetch(gSimFrame);
	else if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
		sampleSubBounceStackAfterFetch(gSimFrame);
	else
		sampleDynamics(gSimFrame, 120);
	++gSimFrame;
}

static void cleanupHeadlessPhysics()
{
	PX_RELEASE(gScene);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	PX_RELEASE(gFoundation);
	gShotBall = NULL;
	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
		gSleepBodies[i] = NULL;
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		gLockedBodies[i] = NULL;
		gLockControlBodies[i] = NULL;
	}
	gSubBounceGround = NULL;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_BOX_COUNT; ++i)
		gSubBounceBodies[i] = NULL;
	gBoxes.clear();
	closePoseLog();
}

static void setGateError(GateEvaluation& evaluation, const char* reason)
{
	if(evaluation.exitCode != Snippets::eHEADLESS_PASS)
		return;
	evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
	evaluation.status = "ERROR";
	evaluation.reason = reason;
}

static void setGateFailure(GateEvaluation& evaluation, const char* reason)
{
	if(evaluation.exitCode != Snippets::eHEADLESS_PASS)
		return;
	evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
	evaluation.status = "FAIL";
	evaluation.reason = reason;
}

static PxReal getSubBounceMaxPenetration()
{
	PxReal maxPenetration = 0.0f;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
		maxPenetration = PxMax(maxPenetration, gSubBounce.maxPenetration[i]);
	return maxPenetration;
}

static GateEvaluation evaluateSubBounceStackGate()
{
	GateEvaluation evaluation;
	evaluation.boxCount = gSubBounce.actorCount;
	if(gCompletedFrames != gOptions.frames || gFetchFailures)
		setGateError(evaluation, "incomplete_simulation");
	if(gSubBounce.actorCount != eHELLO_SUB_BOUNCE_BOX_COUNT)
		setGateError(evaluation, "actor_registry");
	if(gErrorCallback.getFatalCount() || gFetchErrorState)
		setGateFailure(evaluation, "physx_error");

	bool contactsFinite = true;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
		contactsFinite = contactsFinite && gSubBounce.contactFinite[i];
	if(!gSubBounce.finite || !contactsFinite || gNanDetected)
		setGateFailure(evaluation, "non_finite");
	if(gSubBounce.maxAbsPosition > 100.0f)
		setGateFailure(evaluation, "runaway");
	if(getSubBounceInterfaceMask() != ((1u << eHELLO_SUB_BOUNCE_INTERFACE_COUNT) - 1u))
		setGateFailure(evaluation, "missing_contact");
	if(getSubBounceMaxPenetration() > gSubBouncePenetrationCap)
		setGateFailure(evaluation, "persistent_penetration");
	if(gSubBounce.tailPeakLinearSpeed > gSubBounceTailLinearCap ||
		gSubBounce.tailPeakAngularSpeed > gSubBounceTailAngularCap)
		setGateFailure(evaluation, "contact_twitch");
	return evaluation;
}

static PxReal getTailWindowMean(const std::vector<PxReal>& values, PxU32 windowFromEnd)
{
	const PxU32 windowSize = 60;
	const PxU32 required = (windowFromEnd + 1) * windowSize;
	if(values.size() < required)
		return 0.0f;
	const PxU32 end = static_cast<PxU32>(values.size()) - windowFromEnd * windowSize;
	const PxU32 begin = end - windowSize;
	double sum = 0.0;
	for(PxU32 i = begin; i < end; ++i)
		sum += double(values[i]);
	return saturateMetric(sum / double(windowSize));
}

static GateEvaluation evaluateSleepProbeGate()
{
	GateEvaluation evaluation;
	evaluation.boxCount = gSleepActorCount;
	if(gCompletedFrames != gOptions.frames || gFetchFailures)
		setGateError(evaluation, "incomplete_simulation");
	if(gSleepActorCount != eHELLO_SLEEP_WITNESS_COUNT)
		setGateError(evaluation, "actor_registry");
	if(gErrorCallback.getFatalCount() || gFetchErrorState)
		setGateFailure(evaluation, "physx_error");
	if(gNanDetected)
		setGateFailure(evaluation, "non_finite");
	if(gUnexpectedSleepNotifications)
		setGateFailure(evaluation, "unexpected_sleep_notification_actor");

	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		const SleepWitnessMetrics& metrics = gSleepWitnesses[i];
		if(!metrics.finite)
			setGateFailure(evaluation, "non_finite");
		if(metrics.maxAbsPosition > 100.0f || metrics.maxLinearSpeed > 50.0f ||
			metrics.maxAngularSpeed > 50.0f)
			setGateFailure(evaluation, "runaway");
	}
	if(gSleepWitnesses[eHELLO_SLEEP_STATIC_TOUCH].minTopY < -0.05f)
		setGateFailure(evaluation, "ground_escape");

	const bool shouldDisableSleeping = gCase == eHELLO_CASE_SLEEP_DISABLED;
	if(gSceneSleepingDisabled != shouldDisableSleeping)
		setGateFailure(evaluation, "sleep_scene_flag");

	const SleepWitnessMetrics& freeWitness = gSleepWitnesses[eHELLO_SLEEP_FREE];
	const SleepWitnessMetrics& staticWitness = gSleepWitnesses[eHELLO_SLEEP_STATIC_TOUCH];
	if(!shouldDisableSleeping)
	{
		if(freeWitness.invalidWakeCounterSamples || staticWitness.invalidWakeCounterSamples)
			setGateFailure(evaluation, "wake_counter_range");
		if(!PxIsFinite(gWakeCounterResetValue) || gWakeCounterResetValue <= 0.0f)
			setGateFailure(evaluation, "wake_counter_reset");
	}
	if(gCase == eHELLO_CASE_SLEEP_IDLE)
	{
		if(freeWitness.initialSleeping || staticWitness.initialSleeping)
			setGateFailure(evaluation, "idle_initial_state");
		if(!freeWitness.finalSleeping || freeWitness.firstSleepFrame == PX_MAX_U32 ||
			freeWitness.firstSleepFrame > 60u || !freeWitness.sleepSamples ||
			!freeWitness.sleepNotifications)
			setGateFailure(evaluation, "free_auto_sleep");
		if(!staticWitness.finalSleeping || staticWitness.firstSleepFrame == PX_MAX_U32 ||
			staticWitness.firstSleepFrame > 60u || !staticWitness.sleepSamples ||
			!staticWitness.sleepNotifications)
			setGateFailure(evaluation, "static_touch_auto_sleep");
	}
	else if(gCase == eHELLO_CASE_SLEEP_WAKE)
	{
		if(!freeWitness.initialSleeping || !staticWitness.initialSleeping ||
			gFreeAwakeBeforeWakeSamples)
			setGateFailure(evaluation, "pre_wake_sleep_state");
		if(!gWakeImpulseApplied || !gSleepingBeforeWake || !gAwakeImmediatelyAfterWake ||
			gWakeCounterAfterImpulse <= 0.0f)
			setGateFailure(evaluation, "impulse_autowake");
		if(freeWitness.firstAwakeAfterWakeFrame == PX_MAX_U32 ||
			freeWitness.firstAwakeAfterWakeFrame > gSleepWakeFrame + 2u ||
			!freeWitness.wakeNotifications ||
			freeWitness.firstWakeNotificationFrame > gSleepWakeFrame + 2u)
			setGateFailure(evaluation, "wake_observation");
		if(staticWitness.firstAwakeAfterWakeFrame == PX_MAX_U32 ||
			staticWitness.firstAwakeAfterWakeFrame > gSleepWakeFrame + 2u ||
			!staticWitness.wakeNotifications ||
			staticWitness.firstWakeNotificationFrame > gSleepWakeFrame + 2u)
			setGateFailure(evaluation, "wake_propagation");
		if(freeWitness.maxWakeVelocityX < 0.5f || freeWitness.maxWakeDisplacementX < 0.25f)
			setGateFailure(evaluation, "wake_response");
		if(staticWitness.maxWakeVelocityX < 0.05f || staticWitness.maxWakeDisplacementX < 0.05f)
			setGateFailure(evaluation, "propagated_wake_response");
		if(!freeWitness.finalSleeping || freeWitness.firstSleepAfterWakeFrame == PX_MAX_U32 ||
			freeWitness.firstSleepAfterWakeFrame <= gSleepWakeFrame ||
			freeWitness.firstSleepAfterWakeFrame > gOptions.frames ||
			freeWitness.sleepNotifications < 2u)
			setGateFailure(evaluation, "post_wake_resleep");
		if(!staticWitness.finalSleeping || staticWitness.firstSleepAfterWakeFrame == PX_MAX_U32 ||
			staticWitness.firstSleepAfterWakeFrame <= gSleepWakeFrame ||
			staticWitness.firstSleepAfterWakeFrame > gOptions.frames ||
			staticWitness.sleepNotifications < 2u)
			setGateFailure(evaluation, "propagated_post_wake_resleep");
	}
	else
	{
		for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
		{
			const SleepWitnessMetrics& metrics = gSleepWitnesses[i];
			if(metrics.initialSleeping || metrics.finalSleeping || metrics.sleepSamples ||
				metrics.sleepNotifications)
				setGateFailure(evaluation, "sleep_disabled_control");
		}
	}
	return evaluation;
}

static GateEvaluation evaluateLockProbeGate()
{
	GateEvaluation evaluation;
	evaluation.boxCount = gLockActorCount;
	PxReal maxLockedAxisMotion = 0.0f;
	PxReal maxLockedAxisSpeed = 0.0f;
	PxReal minControlAxisMotion = PX_MAX_F32;
	PxReal minControlAxisSpeed = PX_MAX_F32;
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		maxLockedAxisMotion = PxMax(maxLockedAxisMotion, gLockedAxisMotion[i]);
		maxLockedAxisSpeed = PxMax(maxLockedAxisSpeed, gLockedAxisSpeed[i]);
		minControlAxisMotion = PxMin(minControlAxisMotion, gControlAxisMotion[i]);
		minControlAxisSpeed = PxMin(minControlAxisSpeed, gControlAxisSpeed[i]);
	}

	if(gCompletedFrames != gOptions.frames || gFetchFailures)
		setGateError(evaluation, "incomplete_simulation");
	if(gLockActorCount != eHELLO_LOCK_WITNESS_COUNT * 2u ||
		gLockFlagsReadback != eHELLO_LOCK_WITNESS_COUNT ||
		gLockRuntimeExcitations != eHELLO_LOCK_WITNESS_COUNT * 2u)
		setGateError(evaluation, "actor_or_flag_registry");
	if(gLockFiniteSamples != gOptions.frames * eHELLO_LOCK_WITNESS_COUNT * 2u)
		setGateError(evaluation, "sample_registry");
	if(gErrorCallback.getFatalCount() || gFetchErrorState)
		setGateFailure(evaluation, "physx_error");
	if(gLockNonFiniteSamples)
		setGateFailure(evaluation, "non_finite");
	if(minControlAxisMotion < gLockControlMotionMinimum ||
		minControlAxisSpeed < gLockControlSpeedMinimum)
		setGateFailure(evaluation, "control_response");
	if(maxLockedAxisMotion > gLockMotionTolerance || maxLockedAxisSpeed > gLockSpeedTolerance)
		setGateFailure(evaluation, "locked_axis_motion");
	return evaluation;
}

static GateEvaluation evaluateStackOrBallGate()
{
	GateEvaluation evaluation;
	evaluation.boxCount = static_cast<PxU32>(gBoxes.size());
	for(PxU32 i = 0; i < gBoxes.size(); ++i)
		evaluation.targetBoxCount += gBoxes[i].target ? 1u : 0u;
	evaluation.tailEnergyW1 = getTailWindowMean(gEnergyHistory, 3);
	evaluation.tailEnergyW2 = getTailWindowMean(gEnergyHistory, 2);
	evaluation.tailEnergyW3 = getTailWindowMean(gEnergyHistory, 1);
	evaluation.tailEnergyW4 = getTailWindowMean(gEnergyHistory, 0);
	if(gMechanicalEnergyReference > 0.0f)
		evaluation.maxEnergyRatio = gMaxMechanicalEnergyAfterReference / gMechanicalEnergyReference;
	if(PxAbs(evaluation.tailEnergyW1) > 1e-6f)
		evaluation.tailEnergyGrowth =
			(evaluation.tailEnergyW4 - evaluation.tailEnergyW1) / PxAbs(evaluation.tailEnergyW1);
	if(gLateBoxSamples)
		evaluation.lateAwakeRatio = PxReal(gLateAwakeSamples) / PxReal(gLateBoxSamples);
	if(gBallLaunchMomentum > 0.0f)
		evaluation.targetResponseMomentumRatio = gMaxTargetResponseMomentum / gBallLaunchMomentum;
	evaluation.ballResponseFraction =
		gMaxBallImpactAxisVelocityDelta / PxAbs(gBallShotVel.z);

	if(gCompletedFrames != gOptions.frames || gFetchFailures)
		setGateError(evaluation, "incomplete_simulation");
	if(evaluation.boxCount != gExpectedBoxCount ||
		evaluation.targetBoxCount != gExpectedTargetBoxCount)
		setGateError(evaluation, "actor_registry");
	if(gLaunchFailures)
		setGateError(evaluation, "projectile_launch");
	if(!gMechanicalReferenceSet || gEnergyHistory.size() < 240)
		setGateError(evaluation, "energy_window");
	if(gErrorCallback.getFatalCount() || gFetchErrorState)
		setGateFailure(evaluation, "physx_error");
	if(gNanDetected)
		setGateFailure(evaluation, "non_finite");
	if(gMaxQuaternionNormError > 1e-3f)
		setGateFailure(evaluation, "quaternion_norm");
	if(gMaxAbsPosition > gMaxAbsPositionCap || gMaxSpeedAll > gMaxLinearSpeedCap ||
		gMaxAngularSpeed > gMaxAngularSpeedCap)
		setGateFailure(evaluation, "runaway");
	if(gMaxSunkBoxes)
		setGateFailure(evaluation, "box_ground_escape");
	if(gCase == eHELLO_CASE_BALL_SHOT && gMinBallCenterY < -gBallShotRadius)
		setGateFailure(evaluation, "ball_ground_escape");
	if(gMechanicalEnergyReference <= 0.0f ||
		evaluation.maxEnergyRatio > 1.0f + gMechanicalEnergyGainCap)
		setGateFailure(evaluation, "mechanical_energy_gain");
	if(evaluation.tailEnergyGrowth > gTailEnergyGrowthCap)
		setGateFailure(evaluation, "tail_energy_growth");

	if(gCase == eHELLO_CASE_STACK_SETTLE)
	{
		if(gMaxBoxSpeedSettle > gSettleSpeedCap)
			setGateFailure(evaluation, "settle_speed");
		evaluation.levitatingBodies = countLevitatingHelloWorldBoxes();
		if(evaluation.levitatingBodies)
			setGateFailure(evaluation, "levitating_bodies");
	}
	else
	{
		if(!gShotBall || !gBallLaunchMomentum)
			setGateError(evaluation, "projectile_missing");
		if(!gBallBoxContactEvents || !gBallBoxContactPoints || !gContactedBoxes)
			setGateFailure(evaluation, "missing_target_contact");
		if(gFirstBallBoxContactFrame != PX_MAX_U32 &&
			(gFirstBallBoxContactFrame < gBallShotFrame + 12u ||
				gFirstBallBoxContactFrame > gBallShotFrame + 60u))
			setGateFailure(evaluation, "target_contact_window");
		if(gFirstBallBoxContactFrame != PX_MAX_U32 &&
			gFirstBallBoxContactFrame + gBoxResponseWindowFrames > gCompletedFrames)
			setGateFailure(evaluation, "incomplete_response_window");
		if(!gRespondedBoxes ||
			evaluation.targetResponseMomentumRatio < gMinTargetResponseMomentumRatio ||
			gMaxTargetResponseDisplacement < gMinTargetResponseDisplacement)
			setGateFailure(evaluation, "missing_target_response");
		if(evaluation.ballResponseFraction < gMinBallResponseFraction)
			setGateFailure(evaluation, "missing_projectile_response");
	}
	return evaluation;
}

static const char* getSleepProbeFinding()
{
	const SleepWitnessMetrics& freeWitness = gSleepWitnesses[eHELLO_SLEEP_FREE];
	const SleepWitnessMetrics& staticWitness = gSleepWitnesses[eHELLO_SLEEP_STATIC_TOUCH];
	if(gCase == eHELLO_CASE_SLEEP_IDLE)
	{
		if(freeWitness.finalSleeping && staticWitness.finalSleeping)
			return "auto-sleep-observed";
		if(!freeWitness.finalSleeping && !staticWitness.finalSleeping)
			return "auto-sleep-missing";
		return "free-sleep-static-touch-awake";
	}
	if(gCase == eHELLO_CASE_SLEEP_WAKE)
	{
		return freeWitness.firstSleepAfterWakeFrame == PX_MAX_U32 ||
					staticWitness.firstSleepAfterWakeFrame == PX_MAX_U32
			? "post-wake-resleep-missing"
			: "wake-propagation-resleep-observed";
	}
	return "sleep-disabled-control-ok";
}

static const char* getSleepWakeCounterFinding()
{
	if(gCase == eHELLO_CASE_SLEEP_DISABLED)
		return "not-gated-scene-disabled";
	for(PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
	{
		if(gSleepWitnesses[i].invalidWakeCounterSamples)
			return "negative-counter-observed";
	}
	return "counter-range-valid";
}

static void printSubBounceGate(const GateEvaluation& evaluation)
{
	const PxReal tailMeanLinear = gSubBounce.tailSamples
		? gSubBounce.tailLinearSpeedSum / PxReal(gSubBounce.tailSamples)
		: 0.0f;
	const PxReal tailMeanAngular = gSubBounce.tailSamples
		? gSubBounce.tailAngularSpeedSum / PxReal(gSubBounce.tailSamples)
		: 0.0f;
	PxReal minReportedSeparation = PX_MAX_F32;
	PxReal maxReportedSeparation = -PX_MAX_F32;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
	{
		minReportedSeparation = PxMin(minReportedSeparation, gSubBounce.minReportedSeparation[i]);
		maxReportedSeparation = PxMax(maxReportedSeparation, gSubBounce.maxReportedSeparation[i]);
	}
	if(minReportedSeparation == PX_MAX_F32)
		minReportedSeparation = 0.0f;
	if(maxReportedSeparation == -PX_MAX_F32)
		maxReportedSeparation = 0.0f;
	const PxU32 interfaceMask = getSubBounceInterfaceMask();
	bool contactsFinite = true;
	for(PxU32 i = 0; i < eHELLO_SUB_BOUNCE_INTERFACE_COUNT; ++i)
		contactsFinite = contactsFinite && gSubBounce.contactFinite[i];

	printf(
		"[SnippetHelloWorldSubBounce] interfaceMask=0x%x allInterfacesObserved=%u "
		"contactFrames=[%u,%u,%u] contactPoints=[%u,%u,%u] firstContactFrames=[%u,%u,%u] "
		"lastContactFrames=[%u,%u,%u] maxGap=[%.9g,%.9g,%.9g] "
		"maxPenetration=[%.9g,%.9g,%.9g] reportedSeparation=[%.9g,%.9g] "
		"maxTiltDegrees=%.9g tailSamples=%u tailMeanMaxLinearSpeed=%.9g "
		"tailMeanMaxAngularSpeed=%.9g tailPeakLinearSpeed=%.9g "
		"tailPeakAngularSpeed=%.9g firstAllSleepFrame=%u finalSleeping=%u "
		"maxAbsPosition=%.9g penetrationCap=%.9g tailLinearCap=%.9g tailAngularCap=%.9g\n",
		interfaceMask, interfaceMask == ((1u << eHELLO_SUB_BOUNCE_INTERFACE_COUNT) - 1u) ? 1u : 0u,
		gSubBounce.contactFrames[0], gSubBounce.contactFrames[1], gSubBounce.contactFrames[2],
		gSubBounce.contactPoints[0], gSubBounce.contactPoints[1], gSubBounce.contactPoints[2],
		gSubBounce.firstContactFrame[0], gSubBounce.firstContactFrame[1],
		gSubBounce.firstContactFrame[2], gSubBounce.lastContactFrame[0],
		gSubBounce.lastContactFrame[1], gSubBounce.lastContactFrame[2],
		double(gSubBounce.maxGap[0]), double(gSubBounce.maxGap[1]), double(gSubBounce.maxGap[2]),
		double(gSubBounce.maxPenetration[0]), double(gSubBounce.maxPenetration[1]),
		double(gSubBounce.maxPenetration[2]), double(minReportedSeparation),
		double(maxReportedSeparation), double(gSubBounce.maxTiltRadians * 180.0f / PxPi),
		gSubBounce.tailSamples, double(tailMeanLinear), double(tailMeanAngular),
		double(gSubBounce.tailPeakLinearSpeed), double(gSubBounce.tailPeakAngularSpeed),
		gSubBounce.firstAllSleepFrame, gSubBounce.finalSleepingCount,
		double(gSubBounce.maxAbsPosition), double(gSubBouncePenetrationCap),
		double(gSubBounceTailLinearCap), double(gSubBounceTailAngularCap));
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=sub-bounce-stack solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
		"fetchErrorState=%u boxCount=%u restitution=0.6 bounceThresholdVelocity=2 "
		"interfaceMask=0x%x allInterfacesObserved=%u maxPenetration=%.9g "
		"maxPenetration0=%.9g maxPenetration1=%.9g maxPenetration2=%.9g "
		"maxGap0=%.9g maxGap1=%.9g maxGap2=%.9g reportedSeparationMin=%.9g "
		"reportedSeparationMax=%.9g tailPeakLinearSpeed=%.9g tailPeakAngularSpeed=%.9g "
		"tailMeanMaxLinearSpeed=%.9g tailMeanMaxAngularSpeed=%.9g maxTiltDegrees=%.9g "
		"firstAllSleepFrame=%u finalSleeping=%u sleepGate=DIAGNOSTIC "
		"penetrationCap=%.9g tailLinearCap=%.9g tailAngularCap=%.9g\n",
		Snippets::getSolverTypeName(gOptions.solverType),
		Snippets::getExecutionName(gOptions.execution), gOptions.frames, gCompletedFrames,
		double(gOptions.dt), gOptions.seed, gOptions.dispatcherThreads, evaluation.status,
		evaluation.reason,
		(!gSubBounce.finite || !contactsFinite || gNanDetected) ? 1u : 0u,
		gErrorCallback.getFatalCount(), gErrorCallback.getWarningCount(), gFetchFailures,
		gFetchErrorState, evaluation.boxCount, interfaceMask,
		interfaceMask == ((1u << eHELLO_SUB_BOUNCE_INTERFACE_COUNT) - 1u) ? 1u : 0u,
		double(getSubBounceMaxPenetration()), double(gSubBounce.maxPenetration[0]),
		double(gSubBounce.maxPenetration[1]), double(gSubBounce.maxPenetration[2]),
		double(gSubBounce.maxGap[0]), double(gSubBounce.maxGap[1]), double(gSubBounce.maxGap[2]),
		double(minReportedSeparation), double(maxReportedSeparation),
		double(gSubBounce.tailPeakLinearSpeed), double(gSubBounce.tailPeakAngularSpeed),
		double(tailMeanLinear), double(tailMeanAngular),
		double(gSubBounce.maxTiltRadians * 180.0f / PxPi), gSubBounce.firstAllSleepFrame,
		gSubBounce.finalSleepingCount, double(gSubBouncePenetrationCap),
		double(gSubBounceTailLinearCap), double(gSubBounceTailAngularCap));
}

static void printSleepProbeGate(const GateEvaluation& evaluation)
{
	const SleepWitnessMetrics& freeWitness = gSleepWitnesses[eHELLO_SLEEP_FREE];
	const SleepWitnessMetrics& staticWitness = gSleepWitnesses[eHELLO_SLEEP_STATIC_TOUCH];
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
		"fetchFailures=%u fetchErrorState=%u actorCount=%u "
		"sceneSleepingDisabled=%u probeFinding=%s wakeCounterFinding=%s "
		"freeSleepLifecycleGate=HARD staticTouchSleepGate=HARD "
		"wakeCounterReset=%.9g wakeFrame=%u wakeImpulseApplied=%u "
		"sleepingBeforeWake=%u awakeImmediatelyAfterWake=%u "
		"wakeCounterAfterImpulse=%.9g "
		"freeInitialSleeping=%u freeFinalSleeping=%u freeFirstSleepFrame=%u "
		"freeFirstSleepAfterWakeFrame=%u freeSleepSamples=%u freeAwakeSamples=%u "
		"freeSleepNotify=%u freeWakeNotify=%u freeFirstWakeNotifyFrame=%u "
		"freeInitialWakeCounter=%.9g freeFinalWakeCounter=%.9g "
		"freeMinWakeCounter=%.9g freeMaxWakeCounter=%.9g "
		"freeInvalidWakeCounterSamples=%u freeMaxLinearSpeed=%.9g "
		"freeMaxAngularSpeed=%.9g freeMaxWakeVelocityX=%.9g "
		"freeMaxWakeDisplacementX=%.9g staticInitialSleeping=%u "
		"staticFinalSleeping=%u staticFirstSleepFrame=%u "
		"staticFirstSleepAfterWakeFrame=%u staticSleepSamples=%u "
		"staticAwakeSamples=%u staticSleepNotify=%u staticWakeNotify=%u "
		"staticFirstWakeNotifyFrame=%u staticInitialWakeCounter=%.9g "
		"staticFinalWakeCounter=%.9g staticMinWakeCounter=%.9g "
		"staticMaxWakeCounter=%.9g staticInvalidWakeCounterSamples=%u "
		"staticMaxLinearSpeed=%.9g staticMaxAngularSpeed=%.9g "
		"staticMaxWakeVelocityX=%.9g staticMaxWakeDisplacementX=%.9g "
		"staticMinTopY=%.9g unexpectedNotifications=%u "
		"freeAwakeBeforeWakeSamples=%u\n",
		getHeadlessCaseName(gCase), Snippets::getSolverTypeName(gOptions.solverType),
		Snippets::getExecutionName(gOptions.execution), gOptions.frames, gCompletedFrames,
		double(gOptions.dt), gOptions.seed, gOptions.dispatcherThreads, evaluation.status,
		evaluation.reason, gNanDetected ? 1u : 0u, gErrorCallback.getFatalCount(),
		gErrorCallback.getWarningCount(), gFetchFailures, gFetchErrorState, gSleepActorCount,
		gSceneSleepingDisabled ? 1u : 0u, getSleepProbeFinding(), getSleepWakeCounterFinding(),
		double(gWakeCounterResetValue), gSleepWakeFrame, gWakeImpulseApplied ? 1u : 0u,
		gSleepingBeforeWake ? 1u : 0u, gAwakeImmediatelyAfterWake ? 1u : 0u,
		double(gWakeCounterAfterImpulse), freeWitness.initialSleeping ? 1u : 0u,
		freeWitness.finalSleeping ? 1u : 0u, freeWitness.firstSleepFrame,
		freeWitness.firstSleepAfterWakeFrame, freeWitness.sleepSamples, freeWitness.awakeSamples,
		freeWitness.sleepNotifications, freeWitness.wakeNotifications,
		freeWitness.firstWakeNotificationFrame, double(freeWitness.initialWakeCounter),
		double(freeWitness.finalWakeCounter), double(freeWitness.minWakeCounter),
		double(freeWitness.maxWakeCounter), freeWitness.invalidWakeCounterSamples,
		double(freeWitness.maxLinearSpeed), double(freeWitness.maxAngularSpeed),
		double(freeWitness.maxWakeVelocityX), double(freeWitness.maxWakeDisplacementX),
		staticWitness.initialSleeping ? 1u : 0u, staticWitness.finalSleeping ? 1u : 0u,
		staticWitness.firstSleepFrame, staticWitness.firstSleepAfterWakeFrame,
		staticWitness.sleepSamples, staticWitness.awakeSamples, staticWitness.sleepNotifications,
		staticWitness.wakeNotifications, staticWitness.firstWakeNotificationFrame,
		double(staticWitness.initialWakeCounter), double(staticWitness.finalWakeCounter),
		double(staticWitness.minWakeCounter), double(staticWitness.maxWakeCounter),
		staticWitness.invalidWakeCounterSamples, double(staticWitness.maxLinearSpeed),
		double(staticWitness.maxAngularSpeed), double(staticWitness.maxWakeVelocityX),
		double(staticWitness.maxWakeDisplacementX), double(staticWitness.minTopY),
		gUnexpectedSleepNotifications, gFreeAwakeBeforeWakeSamples);
}

static void printLockProbeGate(const GateEvaluation& evaluation)
{
	PxReal maxLockedAxisMotion = 0.0f;
	PxReal maxLockedAxisSpeed = 0.0f;
	PxReal minControlAxisMotion = PX_MAX_F32;
	PxReal minControlAxisSpeed = PX_MAX_F32;
	PxReal maxControlAxisMotion = 0.0f;
	PxReal maxControlAxisSpeed = 0.0f;
	static const char* names[eHELLO_LOCK_WITNESS_COUNT] = {
		"linear-x", "linear-y", "linear-z", "angular-x", "angular-y", "angular-z"};
	for(PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i)
	{
		maxLockedAxisMotion = PxMax(maxLockedAxisMotion, gLockedAxisMotion[i]);
		maxLockedAxisSpeed = PxMax(maxLockedAxisSpeed, gLockedAxisSpeed[i]);
		minControlAxisMotion = PxMin(minControlAxisMotion, gControlAxisMotion[i]);
		minControlAxisSpeed = PxMin(minControlAxisSpeed, gControlAxisSpeed[i]);
		maxControlAxisMotion = PxMax(maxControlAxisMotion, gControlAxisMotion[i]);
		maxControlAxisSpeed = PxMax(maxControlAxisSpeed, gControlAxisSpeed[i]);
		printf("[SnippetHelloWorldLock] witness=%s lockedMotion=%.9g lockedSpeed=%.9g "
			"controlMotion=%.9g controlSpeed=%.9g\n",
			names[i], double(gLockedAxisMotion[i]), double(gLockedAxisSpeed[i]),
			double(gControlAxisMotion[i]), double(gControlAxisSpeed[i]));
	}
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
		"fetchFailures=%u fetchErrorState=%u actorCount=%u lockWitnessCount=%u "
		"lockFlagsReadback=%u runtimeImpulseFrame=%u runtimeExcitations=%u "
		"finiteSamples=%u maxLockedAxisMotion=%.9g maxLockedAxisSpeed=%.9g "
		"minControlAxisMotion=%.9g minControlAxisSpeed=%.9g "
		"maxControlAxisMotion=%.9g maxControlAxisSpeed=%.9g "
		"lockMotionTolerance=%.9g lockSpeedTolerance=%.9g "
		"controlMotionMinimum=%.9g controlSpeedMinimum=%.9g\n",
		getHeadlessCaseName(gCase), Snippets::getSolverTypeName(gOptions.solverType),
		Snippets::getExecutionName(gOptions.execution), gOptions.frames, gCompletedFrames,
		double(gOptions.dt), gOptions.seed, gOptions.dispatcherThreads, evaluation.status,
		evaluation.reason, gLockNonFiniteSamples ? 1u : 0u, gErrorCallback.getFatalCount(),
		gErrorCallback.getWarningCount(), gFetchFailures, gFetchErrorState, gLockActorCount,
		eHELLO_LOCK_WITNESS_COUNT, gLockFlagsReadback, gLockImpulseFrame, gLockRuntimeExcitations,
		gLockFiniteSamples, double(maxLockedAxisMotion), double(maxLockedAxisSpeed),
		double(minControlAxisMotion), double(minControlAxisSpeed), double(maxControlAxisMotion),
		double(maxControlAxisSpeed), double(gLockMotionTolerance), double(gLockSpeedTolerance),
		double(gLockControlMotionMinimum), double(gLockControlSpeedMinimum));
}

static void printStackOrBallGate(const GateEvaluation& evaluation)
{
	const PxReal tailAverageMaxBoxSpeed =
		gTailSpeedSamples ? gTailAvgBoxSpeed / PxReal(gTailSpeedSamples) : 0.0f;
	const PxReal minBallCenterY = gMinBallCenterY == PX_MAX_F32 ? 0.0f : gMinBallCenterY;
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
		"reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
		"fetchErrorState=%u launchFailures=%u boxCount=%u targetBoxCount=%u "
		"hitCount=%u hitPoints=%u firstHitFrame=%u contactedBoxes=%u "
		"responseBodies=%u responseMomentumRatio=%.9g responseDisplacement=%.9g "
		"ballResponseFraction=%.9g maxContactImpulseDiagnostic=%.9g "
		"maxEnergyRatio=%.9g tailEnergyW1=%.9g tailEnergyW2=%.9g "
		"tailEnergyW3=%.9g tailEnergyW4=%.9g tailEnergyGrowth=%.9g "
		"maxBoxSpeedSettle=%.9g tailAverageMaxBoxSpeed=%.9g "
		"finalAwake=%u lateAwakeRatio=%.9g sleepGate=DIAGNOSTIC "
		"avbdRigidSleepCheckPresent=%u groundGate=FULL_FALL_THROUGH "
		"minBoxCenterY=%.9g maxBoxCenterY=%.9g minBallCenterY=%.9g "
		"sunkBodies=%u levitatingBodies=%u maxQuaternionNormError=%.9g maxAbsPosition=%.9g "
		"maxLinearSpeed=%.9g maxAngularSpeed=%.9g energyGainCap=%.9g "
		"tailGrowthCap=%.9g settleSpeedCap=%.9g minResponseMomentumCap=%.9g "
		"minResponseDisplacementCap=%.9g minBallResponseCap=%.9g "
		"responseWindowFrames=%u\n",
		getHeadlessCaseName(gCase), Snippets::getSolverTypeName(gOptions.solverType),
		Snippets::getExecutionName(gOptions.execution), gOptions.frames, gCompletedFrames,
		double(gOptions.dt), gOptions.seed, gOptions.dispatcherThreads, evaluation.status,
		evaluation.reason, gNanDetected ? 1u : 0u, gErrorCallback.getFatalCount(),
		gErrorCallback.getWarningCount(), gFetchFailures, gFetchErrorState, gLaunchFailures,
		evaluation.boxCount, evaluation.targetBoxCount, gBallBoxContactEvents,
		gBallBoxContactPoints, gFirstBallBoxContactFrame, gContactedBoxes, gRespondedBoxes,
		double(evaluation.targetResponseMomentumRatio), double(gMaxTargetResponseDisplacement),
		double(evaluation.ballResponseFraction), double(gMaxContactImpulse),
		double(evaluation.maxEnergyRatio), double(evaluation.tailEnergyW1),
		double(evaluation.tailEnergyW2), double(evaluation.tailEnergyW3),
		double(evaluation.tailEnergyW4), double(evaluation.tailEnergyGrowth),
		double(gMaxBoxSpeedSettle), double(tailAverageMaxBoxSpeed), gFinalAwakeBoxes,
		double(evaluation.lateAwakeRatio), 1u, double(gMinBoxCenterY), double(gMaxBoxCenterY),
		double(minBallCenterY), gMaxSunkBoxes, evaluation.levitatingBodies, double(gMaxQuaternionNormError), double(gMaxAbsPosition),
		double(gMaxSpeedAll), double(gMaxAngularSpeed), double(gMechanicalEnergyGainCap),
		double(gTailEnergyGrowthCap), double(gSettleSpeedCap),
		double(gMinTargetResponseMomentumRatio), double(gMinTargetResponseDisplacement),
		double(gMinBallResponseFraction), gBoxResponseWindowFrames);
}

static int reportConfigurationError(const Snippets::HeadlessOptions& options, const char* message)
{
	printf("[AVBD_GATE_ERROR] snippet=SnippetHelloWorld message=%s\n", message);
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=config-error "
		"solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
		"seed=%u dispatcherThreads=%u capability=SUPPORTED validation=GATED "
		"status=ERROR reason=config nonFinite=0 physicsErrors=0 physicsWarnings=0\n",
		Snippets::getSolverTypeName(options.solverType),
		Snippets::getExecutionName(options.execution), options.frames, double(options.dt),
		options.seed, options.dispatcherThreads);
	return Snippets::eHEADLESS_CONFIG_ERROR;
}

static int reportUnsupported(const Snippets::HeadlessOptions& options, const char* reason)
{
	printf(
		"[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
		"execution=%s requestedFrames=%u completedFrames=0 dt=%.9g seed=%u "
		"dispatcherThreads=%u capability=UNSUPPORTED validation=GATED status=ERROR "
		"reason=%s nonFinite=0 physicsErrors=0 physicsWarnings=0 "
		"fetchFailures=0 fetchErrorState=0 launchFailures=0 sunkBodies=0\n",
		options.caseName.c_str(), Snippets::getSolverTypeName(options.solverType),
		Snippets::getExecutionName(options.execution), options.frames, double(options.dt),
		options.seed, options.dispatcherThreads, reason);
	return Snippets::eHEADLESS_UNSUPPORTED;
}

} // namespace

bool snippetHelloWorldHeadlessShouldRun(int argc, const char* const* argv)
{
#ifndef RENDER_SNIPPET
	PX_UNUSED(argc);
	PX_UNUSED(argv);
	return true;
#else
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))
		return true;
	for(int i = 1; i < argc; ++i)
	{
		if(argv[i] && std::strcmp(argv[i], "--headless") == 0)
			return true;
	}
	return false;
#endif
}

int snippetHelloWorldHeadlessMain(int argc, const char* const* argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);

	Snippets::HeadlessOptions defaults;
	defaults.caseName = "stack-settle";
	defaults.frames = 600;
	defaults.seed = 1;
	defaults.dispatcherThreads = 2;
	defaults.dt = 1.0f / 60.0f;

	Snippets::HeadlessOptions options;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, parseError))
		return reportConfigurationError(options, parseError.c_str());

	bool legacyBallShotSeen = false;
	bool caseSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(!arg)
			continue;
		if(Snippets::isCommonHeadlessOption(arg))
		{
			if(Snippets::hasOptionPrefix(arg, "--case=") ||
				Snippets::hasOptionPrefix(arg, "--scenario="))
				caseSeen = true;
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--pose-log="))
		{
			gPoseLogPath = arg + std::strlen("--pose-log=");
			continue;
		}
		if(std::strcmp(arg, "--headless-ball-shot") == 0)
		{
			if(legacyBallShotSeen)
				return reportConfigurationError(options, "duplicate_--headless-ball-shot");
			legacyBallShotSeen = true;
			options.headless = true;
			continue;
		}
		return reportConfigurationError(options, "unknown_argument");
	}

#ifndef RENDER_SNIPPET
	options.headless = true;
#endif

	if(legacyBallShotSeen)
	{
		if(caseSeen && !Snippets::equalsIgnoreCase(options.caseName.c_str(), "ball-shot"))
			return reportConfigurationError(options, "legacy_ball_shot_case_conflict");
		options.caseName = "ball-shot";
	}

	HelloWorldHeadlessCase headlessCase = eHELLO_CASE_STACK_SETTLE;
	if(!tryParseHeadlessCase(options.caseName.c_str(), headlessCase))
		return reportConfigurationError(options, "invalid_--case_value");
	options.caseName = getHeadlessCaseName(headlessCase);

	if(isSleepProbeCase(headlessCase) && !options.framesExplicit)
		options.frames = headlessCase == eHELLO_CASE_SLEEP_WAKE ? 360u : 180u;
	if(headlessCase == eHELLO_CASE_LOCK_FLAGS && !options.framesExplicit)
		options.frames = 120u;
	if(headlessCase == eHELLO_CASE_SUB_BOUNCE_STACK && !options.framesExplicit)
		options.frames = 360u;

	if(!Snippets::applyExecutionEnvironment(options))
		return reportConfigurationError(options, "execution_environment");

	Snippets::printHeadlessConfig("SnippetHelloWorld", options);
	if(isUnsupportedCase(headlessCase))
		return reportUnsupported(options, unsupportedReason(headlessCase));

	gOptions = options;
	gCase = headlessCase;
	if(!initHeadlessPhysics())
	{
		cleanupHeadlessPhysics();
		return reportConfigurationError(options, "initialization_failed");
	}
	if(!openPoseLog())
	{
		cleanupHeadlessPhysics();
		return reportConfigurationError(options, "pose_log_open_failed");
	}

	for(PxU32 i = 0; i < gOptions.frames; ++i)
		stepHeadlessPhysics();

	GateEvaluation evaluation;
	if(isSleepProbeCase(gCase))
		evaluation = evaluateSleepProbeGate();
	else if(gCase == eHELLO_CASE_LOCK_FLAGS)
		evaluation = evaluateLockProbeGate();
	else if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
		evaluation = evaluateSubBounceStackGate();
	else
		evaluation = evaluateStackOrBallGate();

	if(isSleepProbeCase(gCase))
		printSleepProbeGate(evaluation);
	else if(gCase == eHELLO_CASE_LOCK_FLAGS)
		printLockProbeGate(evaluation);
	else if(gCase == eHELLO_CASE_SUB_BOUNCE_STACK)
		printSubBounceGate(evaluation);
	else
		printStackOrBallGate(evaluation);

	cleanupHeadlessPhysics();
	return evaluation.exitCode;
}
