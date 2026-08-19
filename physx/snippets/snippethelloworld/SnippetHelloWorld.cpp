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

// Box stacks on a plane; optional headless ball shot knocks stacks down.

#include <ctype.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "PxPhysicsAPI.h"
#include "foundation/PxTime.h"
#include "../snippetcommon/SnippetHeadless.h"

#ifdef RENDER_SNIPPET
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#endif

using namespace physx;

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation *gFoundation = NULL;
static PxPhysics *gPhysics = NULL;
static PxDefaultCpuDispatcher *gDispatcher = NULL;
static PxScene *gScene = NULL;
static PxMaterial *gMaterial = NULL;
static PxPvd *gPvd = NULL;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;

static PxReal stackZ = 10.0f;
static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;
static Snippets::HeadlessOptions gHeadlessOptions;
static bool gHeadlessMode = false;
static bool gHeadlessBallShot = false;
static bool gHeadlessSleepProbe = false;
static bool gHeadlessLockProbe = false;
static bool gHeadlessRigidStress = false;
static bool gRigidStressWorkAttribution = false;
static bool gRigidStressAvbdIterationsExplicit = false;
static bool gRigidStressAvbdJointIterationOverrideExplicit = false;
static bool gRigidStressAvbdEarlyStopExplicit = false;
static bool gRigidStressEnhancedDeterminismRequested = false;
static bool gRigidStressEnhancedDeterminismObserved = false;
enum RigidStressLayout {
  eRIGID_STRESS_INDEPENDENT_ISLANDS = 0,
  eRIGID_STRESS_CONNECTED_ISLAND
};
static RigidStressLayout gRigidStressLayout =
    eRIGID_STRESS_INDEPENDENT_ISLANDS;
static PxU32 gRigidStressSceneAvbdIterations = 0;
static PxU32 gRigidStressSceneAvbdJointIterationOverride = 0;
static bool gRigidStressSceneAvbdEnableEarlyStop = false;
static PxU32 gRigidStressRequestedAvbdIterations = 0;
static PxU32 gRigidStressRequestedAvbdJointIterationOverride = 0;
static bool gRigidStressRequestedAvbdEnableEarlyStop = false;
static PxU32 gHeadlessFrameCount = 600;
static PxU32 gBallShotFrame = 30;
static PxU32 gRigidStressWarmupFrames = 10;
static PxU32 gSimFrame = 0;
static PxRigidDynamic *gShotBall = NULL;
enum HelloWorldSleepWitness {
  eHELLO_SLEEP_FREE = 0,
  eHELLO_SLEEP_STATIC_TOUCH = 1,
  eHELLO_SLEEP_WITNESS_COUNT = 2
};
static PxRigidDynamic *gSleepBodies[eHELLO_SLEEP_WITNESS_COUNT] = {NULL,
                                                                   NULL};
enum HelloWorldLockWitness {
  eHELLO_LOCK_LINEAR_X = 0,
  eHELLO_LOCK_LINEAR_Y,
  eHELLO_LOCK_LINEAR_Z,
  eHELLO_LOCK_ANGULAR_X,
  eHELLO_LOCK_ANGULAR_Y,
  eHELLO_LOCK_ANGULAR_Z,
  eHELLO_LOCK_WITNESS_COUNT
};
static PxRigidDynamic *gLockedBodies[eHELLO_LOCK_WITNESS_COUNT] = {
    NULL, NULL, NULL, NULL, NULL, NULL};
static PxRigidDynamic *gLockControlBodies[eHELLO_LOCK_WITNESS_COUNT] = {
    NULL, NULL, NULL, NULL, NULL, NULL};
static PxTransform gLockInitialPoses[eHELLO_LOCK_WITNESS_COUNT * 2];
static std::vector<PxRigidDynamic *> gBoxes;
static std::vector<PxVec3> gPreviousBoxVelocities;
static std::vector<PxVec3> gPreviousBoxPositions;
static std::vector<PxVec3> gBoxContactBaselines;
static std::vector<PxVec3> gBoxContactPositionBaselines;
static std::vector<PxU8> gBoxContacted;
static std::vector<PxU8> gBoxResponseObserved;
static std::vector<PxU8> gBoxIsTarget;
static std::vector<PxU32> gBoxContactFrames;
static std::vector<PxReal> gMechanicalEnergyHistory;

static const PxU32 gRigidStressStackCount = 40;
static const PxU32 gRigidStressStackSize = 20;
static const PxReal gRigidStressBoxHalfExtent = 1.0f;
static const PxU32 gRigidStressExpectedBoxCount =
    gRigidStressStackCount * gRigidStressStackSize *
    (gRigidStressStackSize + 1u) / 2u;
static const PxReal gRigidStressBallRadius = 5.0f;
static const PxReal gRigidStressBallDensity = 1000.0f;
static const PxVec3 gRigidStressBallPosition(0.0f, 20.0f, 100.0f);
static const PxVec3 gRigidStressBallVelocity(0.0f, -25.0f, -100.0f);
static const PxU32 gRigidStressPositionIterations = 4;
static const PxU32 gRigidStressVelocityIterations = 1;

struct RigidStressWorkMetrics {
  PxU64 avbdIslandSolves = 0;
  PxU64 avbdInnerSweeps = 0;
  PxU64 avbdBlockDescentZones = 0;
  PxU64 avbdBodyColorPlans = 0;
  PxU64 avbdBodyColorPasses = 0;
  PxU64 avbdDualPasses = 0;
  PxU64 avbdDualRanges = 0;
  PxU64 avbdBodyRanges = 0;
  PxU64 awakeDynamicBodies = 0;
  PxU64 sceneStatsActiveDynamicBodies = 0;
  PxU64 discreteContactPairs = 0;
  PxU64 contactPairsWithContacts = 0;
  PxU64 axisSolverConstraints = 0;
  PxU64 solverPartitions = 0;
  PxU32 peakAwakeDynamicBodies = 0;
  PxU32 peakSceneStatsActiveDynamicBodies = 0;
  PxU32 peakDiscreteContactPairs = 0;
  PxU32 peakContactPairsWithContacts = 0;
  PxU32 peakAxisSolverConstraints = 0;
  PxU32 peakSolverPartitions = 0;
  PxU32 sampledFrames = 0;
  bool observed = false;
};

struct RigidStressProfilerCounts {
  PxU64 islandSolves;
  PxU64 innerSweeps;
  PxU64 blockDescentZones;
  PxU64 bodyColorPlans;
  PxU64 bodyColorPasses;
  PxU64 dualPasses;
  PxU64 dualRanges;
  PxU64 bodyRanges;

  RigidStressProfilerCounts()
      : islandSolves(0), innerSweeps(0), blockDescentZones(0),
        bodyColorPlans(0), bodyColorPasses(0), dualPasses(0), dualRanges(0),
        bodyRanges(0) {}
};

#if PX_DEBUG || PX_CHECKED || PX_PROFILE
class RigidStressProfilerCallback : public PxProfilerCallback {
public:
  RigidStressProfilerCallback()
      : mIslandSolves(0), mInnerSweeps(0), mBlockDescentZones(0),
        mBodyColorPlans(0), mBodyColorPasses(0), mDualPasses(0),
        mDualRanges(0), mBodyRanges(0) {}
  virtual ~RigidStressProfilerCallback() {}

  void reset() {
    mIslandSolves.store(0, std::memory_order_relaxed);
    mInnerSweeps.store(0, std::memory_order_relaxed);
    mBlockDescentZones.store(0, std::memory_order_relaxed);
    mBodyColorPlans.store(0, std::memory_order_relaxed);
    mBodyColorPasses.store(0, std::memory_order_relaxed);
    mDualPasses.store(0, std::memory_order_relaxed);
    mDualRanges.store(0, std::memory_order_relaxed);
    mBodyRanges.store(0, std::memory_order_relaxed);
  }

  RigidStressProfilerCounts snapshot() const {
    RigidStressProfilerCounts result;
    result.islandSolves = mIslandSolves.load(std::memory_order_relaxed);
    result.innerSweeps = mInnerSweeps.load(std::memory_order_relaxed);
    result.blockDescentZones =
        mBlockDescentZones.load(std::memory_order_relaxed);
    result.bodyColorPlans = mBodyColorPlans.load(std::memory_order_relaxed);
    result.bodyColorPasses = mBodyColorPasses.load(std::memory_order_relaxed);
    result.dualPasses = mDualPasses.load(std::memory_order_relaxed);
    result.dualRanges = mDualRanges.load(std::memory_order_relaxed);
    result.bodyRanges = mBodyRanges.load(std::memory_order_relaxed);
    return result;
  }

  virtual void *zoneStart(const char *eventName, bool,
                          uint64_t) PX_OVERRIDE {
    if (!eventName)
      return NULL;
    if (strcmp(eventName, "AVBD.solveIsland") == 0)
      mIslandSolves.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.updateLambda") == 0)
      mInnerSweeps.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.blockDescent") == 0)
      mBlockDescentZones.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.buildRigidBodyColorPlan") == 0)
      mBodyColorPlans.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.submitRigidColor") == 0)
      mBodyColorPasses.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.submitRigidDual") == 0)
      mDualPasses.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.updateLambdaRange") == 0)
      mDualRanges.fetch_add(1, std::memory_order_relaxed);
    else if (strcmp(eventName, "AVBD.solveRigidBodyRange") == 0)
      mBodyRanges.fetch_add(1, std::memory_order_relaxed);
    return NULL;
  }

  virtual void zoneEnd(void *, const char *, bool, uint64_t) PX_OVERRIDE {}

private:
  std::atomic<PxU64> mIslandSolves;
  std::atomic<PxU64> mInnerSweeps;
  std::atomic<PxU64> mBlockDescentZones;
  std::atomic<PxU64> mBodyColorPlans;
  std::atomic<PxU64> mBodyColorPasses;
  std::atomic<PxU64> mDualPasses;
  std::atomic<PxU64> mDualRanges;
  std::atomic<PxU64> mBodyRanges;
};

static RigidStressProfilerCallback gRigidStressProfiler;
#endif

static RigidStressWorkMetrics gRigidStressWorkMetrics;
static RigidStressProfilerCounts gRigidStressProfilerBaseline;
static bool gRigidStressProfilerBaselineCaptured = false;

struct RigidStressMetrics {
  std::vector<PxReal> stepSamplesMs;
  PxReal avgStepMs = 0.0f;
  PxReal p50StepMs = 0.0f;
  PxReal p95StepMs = 0.0f;
  PxReal maxStepMs = 0.0f;
  PxReal minBoxCenterY = PX_MAX_F32;
  PxReal maxAbsPosition = 0.0f;
  PxVec3 finalBallPosition = PxVec3(0.0f);
  PxU32 finiteBoxes = 0;
  PxU32 awakeBoxes = 0;
  PxU32 movedBoxes = 0;
  PxU64 stateDigestLow = 0;
  PxU64 stateDigestHigh = 0;
  PxU32 stateDigestActorCount = 0;
  bool finalBallFinite = false;
};

static RigidStressMetrics gRigidStressMetrics;

static const PxReal gBallShotRadius = 3.0f;
static const PxVec3 gBallShotPos(0.0f, 22.0f, 70.0f);
static const PxVec3 gBallShotVel(0.0f, -20.0f, -220.0f);
static const PxU32 gBallResponseWindowFrames = 3;
static const PxU32 gBoxResponseWindowFrames = 12;
static const PxU32 gExpectedBoxCount = 275;
static const PxU32 gExpectedTargetBoxCount = 55;
static const PxReal gMinBallResponseFraction = 0.05f;
static const PxReal gMinTargetResponseMomentumRatio = 0.01f;
static const PxReal gMinTargetResponseDisplacement = 0.5f;
static const PxReal gBoxHalfExtent = 2.0f;
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
static const PxU32 gLockImpulseFrame = 30u;
static const PxReal gLockMotionTolerance = 1e-4f;
static const PxReal gLockSpeedTolerance = 1e-4f;
static const PxReal gLockControlMotionMinimum = 1.0f;
static const PxReal gLockControlSpeedMinimum = 1.0f;

struct HelloWorldMetrics {
  PxReal maxSpeedAll = 0.0f;
  PxReal maxBoxSpeedSettle = 0.0f;
  PxReal maxBoxCenterY = -1e9f;
  PxReal minBoxCenterY = 1e9f;
  PxReal maxTargetImpactAxisVelocityDelta = 0.0f;
  PxReal maxTargetResponseMomentum = 0.0f;
  PxReal maxTargetResponseDisplacement = 0.0f;
  PxReal maxBallImpactAxisVelocityDelta = 0.0f;
  PxReal ballLaunchMomentum = 0.0f;
  PxReal maxContactImpulse = 0.0f;
  PxReal mechanicalEnergyReference = 0.0f;
  PxReal maxMechanicalEnergyAfterReference = 0.0f;
  PxReal finalMechanicalEnergy = 0.0f;
  PxReal maxQuaternionNormError = 0.0f;
  PxReal maxAbsPosition = 0.0f;
  PxReal maxAngularSpeed = 0.0f;
  // Ball ground-bounce diagnostics (restitution e=0.6 material).
  PxReal maxBallUpVy = 0.0f;     // max +vy after shot (rebound signature)
  PxReal maxBallSpeed = 0.0f;
  PxReal minBallCenterY = PX_MAX_F32;
  PxReal ballVyAtFirstGround = 0.0f; // vy when first near plane (y < r+0.5)
  bool ballSawGround = false;
  PxU32 awakeBoxesSettle = 0;
  PxU32 finalAwakeBoxes = 0;
  PxU32 lateAwakeSamples = 0;
  PxU32 lateBoxSamples = 0;
  PxU32 maxSunkBoxes = 0;
  PxU32 launchFailures = 0;
  PxU32 ballBoxContactEvents = 0;
  PxU32 ballBoxContactPoints = 0;
  PxU32 contactedBoxes = 0;
  PxU32 respondedBoxes = 0;
  PxU32 firstBallBoxContactFrame = PX_MAX_U32;
  PxU32 completedFrames = 0;
  PxU32 fetchFailures = 0;
  PxU32 fetchErrorState = 0;
  PxReal tailAvgBoxSpeed = 0.0f;
  PxU32 tailSpeedSamples = 0;
  bool nanDetected = false;
  bool mechanicalReferenceSet = false;
};

static HelloWorldMetrics gMetrics;

struct HelloWorldSleepWitnessMetrics {
  PxVec3 initialPosition = PxVec3(0.0f);
  PxVec3 wakePosition = PxVec3(0.0f);
  PxVec3 finalPosition = PxVec3(0.0f);
  PxU32 firstSleepFrame = PX_MAX_U32;
  PxU32 firstSleepAfterWakeFrame = PX_MAX_U32;
  PxU32 firstAwakeAfterWakeFrame = PX_MAX_U32;
  PxU32 sleepSamples = 0;
  PxU32 awakeSamples = 0;
  PxU32 sleepNotifications = 0;
  PxU32 wakeNotifications = 0;
  PxU32 invalidWakeCounterSamples = 0;
  PxU32 firstSleepNotificationFrame = PX_MAX_U32;
  PxU32 firstWakeNotificationFrame = PX_MAX_U32;
  PxReal maxLinearSpeed = 0.0f;
  PxReal maxAngularSpeed = 0.0f;
  PxReal maxAbsPosition = 0.0f;
  PxReal maxWakeDisplacementX = 0.0f;
  PxReal maxWakeVelocityX = 0.0f;
  PxReal finalLinearSpeed = 0.0f;
  PxReal finalAngularSpeed = 0.0f;
  PxReal minTopY = PX_MAX_F32;
  PxReal initialWakeCounter = 0.0f;
  PxReal finalWakeCounter = 0.0f;
  PxReal minWakeCounter = PX_MAX_F32;
  PxReal maxWakeCounter = -PX_MAX_F32;
  bool initialSleeping = false;
  bool finalSleeping = false;
  bool previousSleeping = false;
  bool previousSleepingValid = false;
  bool finite = true;
};

struct HelloWorldSleepProbeMetrics {
  HelloWorldSleepWitnessMetrics witnesses[eHELLO_SLEEP_WITNESS_COUNT];
  PxU32 actorCount = 0;
  PxU32 unexpectedNotifications = 0;
  PxU32 freeAwakeBeforeWakeSamples = 0;
  bool sceneSleepingDisabled = false;
  bool wakeImpulseApplied = false;
  bool sleepingBeforeWake = false;
  bool awakeImmediatelyAfterWake = false;
  PxReal wakeCounterResetValue = 0.0f;
  PxReal wakeCounterAfterImpulse = 0.0f;
};

static HelloWorldSleepProbeMetrics gSleepMetrics;

struct HelloWorldLockProbeMetrics {
  PxReal lockedAxisMotion[eHELLO_LOCK_WITNESS_COUNT] = {};
  PxReal lockedAxisSpeed[eHELLO_LOCK_WITNESS_COUNT] = {};
  PxReal controlAxisMotion[eHELLO_LOCK_WITNESS_COUNT] = {};
  PxReal controlAxisSpeed[eHELLO_LOCK_WITNESS_COUNT] = {};
  PxReal maxLockedAxisMotion = 0.0f;
  PxReal maxLockedAxisSpeed = 0.0f;
  PxReal minControlAxisMotion = PX_MAX_F32;
  PxReal minControlAxisSpeed = PX_MAX_F32;
  PxReal maxControlAxisMotion = 0.0f;
  PxReal maxControlAxisSpeed = 0.0f;
  PxU32 actorCount = 0;
  PxU32 lockFlagsReadback = 0;
  PxU32 runtimeExcitations = 0;
  PxU32 finiteSamples = 0;
  PxU32 nonFiniteSamples = 0;
};

static HelloWorldLockProbeMetrics gLockMetrics;

enum HelloWorldFilterKind {
  eHELLO_FILTER_UNTAGGED = 0,
  eHELLO_FILTER_BOX = 1,
  eHELLO_FILTER_TARGET_BOX = 2,
  eHELLO_FILTER_BALL = 3
};

static PxReal saturateMetric(double value) {
  if (!std::isfinite(value) || value >= double(PX_MAX_F32))
    return PX_MAX_F32;
  if (value <= -double(PX_MAX_F32))
    return -PX_MAX_F32;
  return PxReal(value);
}

static PxReal getSafeMagnitude(const PxVec3 &value) {
  if (!value.isFinite())
    return PX_MAX_F32;
  const double x = double(value.x);
  const double y = double(value.y);
  const double z = double(value.z);
  return saturateMetric(std::sqrt(x * x + y * y + z * z));
}

static PxI32 findBoxIndex(const PxActor *actor) {
  for (PxU32 i = 0; i < gBoxes.size(); ++i) {
    if (gBoxes[i] == actor)
      return static_cast<PxI32>(i);
  }
  return -1;
}

static PxI32 findSleepWitnessIndex(const PxActor *actor) {
  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    if (gSleepBodies[i] == actor)
      return static_cast<PxI32>(i);
  }
  return -1;
}

class HelloWorldSimulationCallback : public PxSimulationEventCallback {
public:
  virtual void onConstraintBreak(PxConstraintInfo *, PxU32) PX_OVERRIDE {}
  virtual void onWake(PxActor **actors, PxU32 count) PX_OVERRIDE {
    if (!gHeadlessSleepProbe)
      return;
    for (PxU32 i = 0; i < count; ++i) {
      const PxI32 witness = findSleepWitnessIndex(actors[i]);
      if (witness >= 0) {
        gSleepMetrics.witnesses[witness].wakeNotifications++;
        if (gSleepMetrics.witnesses[witness].firstWakeNotificationFrame ==
            PX_MAX_U32)
          gSleepMetrics.witnesses[witness].firstWakeNotificationFrame =
              gSimFrame + 1;
      } else
        gSleepMetrics.unexpectedNotifications++;
    }
  }
  virtual void onSleep(PxActor **actors, PxU32 count) PX_OVERRIDE {
    if (!gHeadlessSleepProbe)
      return;
    for (PxU32 i = 0; i < count; ++i) {
      const PxI32 witness = findSleepWitnessIndex(actors[i]);
      if (witness >= 0) {
        gSleepMetrics.witnesses[witness].sleepNotifications++;
        if (gSleepMetrics.witnesses[witness].firstSleepNotificationFrame ==
            PX_MAX_U32)
          gSleepMetrics.witnesses[witness].firstSleepNotificationFrame =
              gSimFrame + 1;
      } else
        gSleepMetrics.unexpectedNotifications++;
    }
  }
  virtual void onTrigger(PxTriggerPair *, PxU32) PX_OVERRIDE {}
  virtual void onAdvance(const PxRigidBody *const *, const PxTransform *,
                         const PxU32) PX_OVERRIDE {}

  virtual void onContact(const PxContactPairHeader &pairHeader,
                         const PxContactPair *pairs,
                         PxU32 pairCount) PX_OVERRIDE {
    if (pairHeader.flags &
        (PxContactPairHeaderFlag::eREMOVED_ACTOR_0 |
         PxContactPairHeaderFlag::eREMOVED_ACTOR_1))
      return;
    const PxActor *boxActor = NULL;
    if (pairHeader.actors[0] == gShotBall)
      boxActor = pairHeader.actors[1];
    else if (pairHeader.actors[1] == gShotBall)
      boxActor = pairHeader.actors[0];
    else
      return;

    const PxI32 boxIndex = findBoxIndex(boxActor);
    if (boxIndex < 0 || !gBoxIsTarget[static_cast<PxU32>(boxIndex)])
      return;

    gMetrics.ballBoxContactEvents++;
    if (gMetrics.firstBallBoxContactFrame == PX_MAX_U32)
      gMetrics.firstBallBoxContactFrame = gSimFrame + 1;
    const PxU32 index = static_cast<PxU32>(boxIndex);
    if (!gBoxContacted[index]) {
      gBoxContacted[index] = 1;
      gBoxContactBaselines[index] = gPreviousBoxVelocities[index];
      gBoxContactPositionBaselines[index] = gPreviousBoxPositions[index];
      gBoxContactFrames[index] = gSimFrame + 1;
      gMetrics.contactedBoxes++;
    }

    for (PxU32 pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
      if (pairs[pairIndex].flags &
          (PxContactPairFlag::eREMOVED_SHAPE_0 |
           PxContactPairFlag::eREMOVED_SHAPE_1))
        continue;
      std::vector<PxContactPairPoint> points(pairs[pairIndex].contactCount);
      const PxU32 extracted = pairs[pairIndex].extractContacts(
          points.data(), pairs[pairIndex].contactCount);
      gMetrics.ballBoxContactPoints += extracted;
      for (PxU32 pointIndex = 0; pointIndex < extracted; ++pointIndex) {
        if (!points[pointIndex].impulse.isFinite()) {
          gMetrics.nanDetected = true;
          continue;
        }
        const PxReal impulse = getSafeMagnitude(points[pointIndex].impulse);
        if (impulse < PX_MAX_F32)
          gMetrics.maxContactImpulse =
              PxMax(gMetrics.maxContactImpulse, impulse);
        else
          gMetrics.nanDetected = true;
      }
    }
  }
};

static HelloWorldSimulationCallback gSimulationCallback;

static PxFilterFlags helloWorldFilterShader(
    PxFilterObjectAttributes attributes0, PxFilterData filterData0,
    PxFilterObjectAttributes attributes1, PxFilterData filterData1,
    PxPairFlags &pairFlags, const void *, PxU32) {
  if (PxFilterObjectIsTrigger(attributes0) ||
      PxFilterObjectIsTrigger(attributes1)) {
    pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
    return PxFilterFlag::eDEFAULT;
  }
  pairFlags = PxPairFlag::eCONTACT_DEFAULT;
  const bool ballAndBox =
      (filterData0.word0 == eHELLO_FILTER_BALL &&
       filterData1.word0 == eHELLO_FILTER_TARGET_BOX) ||
      (filterData1.word0 == eHELLO_FILTER_BALL &&
       filterData0.word0 == eHELLO_FILTER_TARGET_BOX);
  if (ballAndBox)
    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
  return PxFilterFlag::eDEFAULT;
}

enum HelloWorldHeadlessCase {
  eHELLO_CASE_STACK_SETTLE,
  eHELLO_CASE_BALL_SHOT,
  eHELLO_CASE_RIGID_STRESS,
  eHELLO_CASE_SLEEP_IDLE,
  eHELLO_CASE_SLEEP_WAKE,
  eHELLO_CASE_SLEEP_DISABLED,
  eHELLO_CASE_LOCK_FLAGS
};

static HelloWorldHeadlessCase gHeadlessCase = eHELLO_CASE_STACK_SETTLE;

static bool tryParseHeadlessCase(const char *value,
                                 HelloWorldHeadlessCase &headlessCase) {
  if (Snippets::equalsIgnoreCase(value, "stack-settle") ||
      Snippets::equalsIgnoreCase(value, "default")) {
    headlessCase = eHELLO_CASE_STACK_SETTLE;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "ball-shot")) {
    headlessCase = eHELLO_CASE_BALL_SHOT;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "rigid-stress") ||
      Snippets::equalsIgnoreCase(value, "hellogrb-cpu")) {
    headlessCase = eHELLO_CASE_RIGID_STRESS;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "sleep-idle")) {
    headlessCase = eHELLO_CASE_SLEEP_IDLE;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "sleep-wake")) {
    headlessCase = eHELLO_CASE_SLEEP_WAKE;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "sleep-disabled")) {
    headlessCase = eHELLO_CASE_SLEEP_DISABLED;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "lock-flags")) {
    headlessCase = eHELLO_CASE_LOCK_FLAGS;
    return true;
  }
  return false;
}

static const char *getHeadlessCaseName(HelloWorldHeadlessCase headlessCase) {
  switch (headlessCase) {
  case eHELLO_CASE_BALL_SHOT:
    return "ball-shot";
  case eHELLO_CASE_RIGID_STRESS:
    return "rigid-stress";
  case eHELLO_CASE_SLEEP_IDLE:
    return "sleep-idle";
  case eHELLO_CASE_SLEEP_WAKE:
    return "sleep-wake";
  case eHELLO_CASE_SLEEP_DISABLED:
    return "sleep-disabled";
  case eHELLO_CASE_LOCK_FLAGS:
    return "lock-flags";
  default:
    return "stack-settle";
  }
}

static bool isSleepProbeCase(HelloWorldHeadlessCase headlessCase) {
  return headlessCase == eHELLO_CASE_SLEEP_IDLE ||
         headlessCase == eHELLO_CASE_SLEEP_WAKE ||
         headlessCase == eHELLO_CASE_SLEEP_DISABLED;
}

static bool isLockProbeCase(HelloWorldHeadlessCase headlessCase) {
  return headlessCase == eHELLO_CASE_LOCK_FLAGS;
}

static const char *getRigidStressLayoutName() {
  return gRigidStressLayout == eRIGID_STRESS_CONNECTED_ISLAND
             ? "connected"
             : "independent";
}

static void resetRuntimeState() {
  stackZ = 10.0f;
  gSimFrame = 0;
  gShotBall = NULL;
  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
    gSleepBodies[i] = NULL;
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    gLockedBodies[i] = NULL;
    gLockControlBodies[i] = NULL;
    gLockInitialPoses[i] = PxTransform(PxIdentity);
    gLockInitialPoses[i + eHELLO_LOCK_WITNESS_COUNT] =
        PxTransform(PxIdentity);
  }
  gMetrics = HelloWorldMetrics();
  gSleepMetrics = HelloWorldSleepProbeMetrics();
  gLockMetrics = HelloWorldLockProbeMetrics();
  gRigidStressMetrics = RigidStressMetrics();
  gRigidStressWorkMetrics = RigidStressWorkMetrics();
  gRigidStressProfilerBaseline = RigidStressProfilerCounts();
  gRigidStressProfilerBaselineCaptured = false;
  gRigidStressEnhancedDeterminismObserved = false;
#if PX_DEBUG || PX_CHECKED || PX_PROFILE
  gRigidStressProfiler.reset();
#endif
  gBoxes.clear();
  gPreviousBoxVelocities.clear();
  gPreviousBoxPositions.clear();
  gBoxContactBaselines.clear();
  gBoxContactPositionBaselines.clear();
  gBoxContacted.clear();
  gBoxResponseObserved.clear();
  gBoxIsTarget.clear();
  gBoxContactFrames.clear();
  gMechanicalEnergyHistory.clear();
}

static PxRigidDynamic *createDynamic(const PxTransform &t,
                                     const PxGeometry &geometry,
                                     const PxVec3 &velocity = PxVec3(0),
                                     HelloWorldFilterKind filterKind =
                                         eHELLO_FILTER_UNTAGGED) {
  PxRigidDynamic *dynamic =
      PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
  if (!dynamic)
    return NULL;
  dynamic->setAngularDamping(0.5f);
  dynamic->setLinearVelocity(velocity);
  if (gHeadlessRigidStress)
    dynamic->setSolverIterationCounts(gRigidStressPositionIterations,
                                      gRigidStressVelocityIterations);
  PxShape *shape = NULL;
  if (dynamic->getShapes(&shape, 1) == 1 && shape) {
    PxFilterData filterData;
    filterData.word0 = filterKind;
    shape->setSimulationFilterData(filterData);
  }
  gScene->addActor(*dynamic);
  return dynamic;
}

static void createSleepProbeFixture() {
  const PxBoxGeometry geometry(gSleepBoxHalfExtent, gSleepBoxHalfExtent,
                               gSleepBoxHalfExtent);
  if (gHeadlessCase == eHELLO_CASE_SLEEP_WAKE) {
    gSleepBodies[eHELLO_SLEEP_FREE] =
        createDynamic(PxTransform(PxVec3(-0.49f, 4.0f, 0.0f)), geometry);
    gSleepBodies[eHELLO_SLEEP_STATIC_TOUCH] =
        createDynamic(PxTransform(PxVec3(0.49f, 4.0f, 0.0f)), geometry);
  } else {
    gSleepBodies[eHELLO_SLEEP_FREE] =
        createDynamic(PxTransform(PxVec3(-2.0f, 4.0f, 0.0f)), geometry);
    gSleepBodies[eHELLO_SLEEP_STATIC_TOUCH] = createDynamic(
        PxTransform(PxVec3(2.0f, gSleepBoxHalfExtent - 0.01f, 0.0f)),
        geometry);
  }

  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    PxRigidDynamic *body = gSleepBodies[i];
    if (!body) {
      gInitializationFailed = true;
      continue;
    }
    body->setActorFlag(PxActorFlag::eSEND_SLEEP_NOTIFIES, true);
    if (i == eHELLO_SLEEP_FREE ||
        gHeadlessCase == eHELLO_CASE_SLEEP_WAKE) {
      body->setLinearDamping(2.0f);
    }
    gSleepMetrics.actorCount++;
  }

  if (gHeadlessCase == eHELLO_CASE_SLEEP_WAKE) {
    for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
      if (gSleepBodies[i])
        gSleepBodies[i]->putToSleep();
    }
  }

  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    PxRigidDynamic *body = gSleepBodies[i];
    if (!body)
      continue;
    HelloWorldSleepWitnessMetrics &metrics = gSleepMetrics.witnesses[i];
    const PxTransform pose = body->getGlobalPose();
    metrics.initialPosition = pose.p;
    metrics.initialSleeping = body->isSleeping();
    metrics.previousSleeping = metrics.initialSleeping;
    metrics.previousSleepingValid = true;
    metrics.initialWakeCounter = body->getWakeCounter();
    metrics.minWakeCounter = metrics.initialWakeCounter;
    metrics.maxWakeCounter = metrics.initialWakeCounter;
    if (metrics.initialSleeping)
      metrics.firstSleepFrame = 0;
    if (!pose.isValid() || !PxIsFinite(metrics.initialWakeCounter))
      metrics.finite = false;
  }
}

static PxVec3 getLockAxis(PxU32 witness) {
  const PxU32 axis = witness % 3u;
  return axis == 0u ? PxVec3(1.0f, 0.0f, 0.0f)
                    : axis == 1u ? PxVec3(0.0f, 1.0f, 0.0f)
                                 : PxVec3(0.0f, 0.0f, 1.0f);
}

static PxRigidDynamicLockFlag::Enum getLockFlag(PxU32 witness) {
  static const PxRigidDynamicLockFlag::Enum flags[eHELLO_LOCK_WITNESS_COUNT] = {
      PxRigidDynamicLockFlag::eLOCK_LINEAR_X,
      PxRigidDynamicLockFlag::eLOCK_LINEAR_Y,
      PxRigidDynamicLockFlag::eLOCK_LINEAR_Z,
      PxRigidDynamicLockFlag::eLOCK_ANGULAR_X,
      PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y,
      PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z};
  return flags[witness];
}

static PxReal getOrientationDelta(const PxQuat &initial,
                                  const PxQuat &current) {
  const PxReal cosine =
      PxMin(1.0f, PxMax(0.0f, PxAbs(initial.dot(current))));
  return 2.0f * PxAcos(cosine);
}

static void createLockProbeFixture() {
  const PxSphereGeometry geometry(0.25f);
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    const PxReal x = -30.0f + PxReal(i) * 12.0f;
    PxRigidDynamic *locked =
        createDynamic(PxTransform(PxVec3(x, 20.0f, 0.0f)), geometry);
    PxRigidDynamic *control =
        createDynamic(PxTransform(PxVec3(x, 32.0f, 0.0f)), geometry);
    gLockedBodies[i] = locked;
    gLockControlBodies[i] = control;
    if (!locked || !control) {
      gInitializationFailed = true;
      continue;
    }

    const PxVec3 axis = getLockAxis(i);
    const PxRigidDynamicLockFlag::Enum flag = getLockFlag(i);
    PxRigidDynamic *bodies[2] = {locked, control};
    for (PxU32 bodyIndex = 0; bodyIndex < 2; ++bodyIndex) {
      PxRigidDynamic *body = bodies[bodyIndex];
      body->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
      body->setLinearDamping(0.0f);
      body->setAngularDamping(0.0f);
      body->setSleepThreshold(0.0f);
    }

    locked->setRigidDynamicLockFlag(flag, true);
    if (i < eHELLO_LOCK_ANGULAR_X) {
      locked->setLinearVelocity(axis * gLockLinearSpeed);
      control->setLinearVelocity(axis * gLockLinearSpeed);
    } else {
      locked->setAngularVelocity(axis * gLockAngularSpeed);
      control->setAngularVelocity(axis * gLockAngularSpeed);
    }

    gLockInitialPoses[i] = locked->getGlobalPose();
    gLockInitialPoses[i + eHELLO_LOCK_WITNESS_COUNT] =
        control->getGlobalPose();
    if (locked->getRigidDynamicLockFlags() ==
            PxRigidDynamicLockFlags(flag) &&
        control->getRigidDynamicLockFlags() == PxRigidDynamicLockFlags())
      gLockMetrics.lockFlagsReadback++;
    gLockMetrics.actorCount += 2u;
  }
}

static void applyLockProbeRuntimeExcitation() {
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    const PxVec3 axis = getLockAxis(i);
    PxRigidDynamic *bodies[2] = {gLockedBodies[i], gLockControlBodies[i]};
    for (PxU32 bodyIndex = 0; bodyIndex < 2; ++bodyIndex) {
      PxRigidDynamic *body = bodies[bodyIndex];
      if (!body)
        continue;
      if (i < eHELLO_LOCK_ANGULAR_X) {
        const PxReal impulse = body->getMass() * gLockLinearSpeed;
        body->addForce(axis * impulse, PxForceMode::eIMPULSE, true);
      } else {
        const PxReal angularImpulse =
            body->getMassSpaceInertiaTensor().dot(axis) * gLockAngularSpeed;
        body->addTorque(axis * angularImpulse, PxForceMode::eIMPULSE, true);
      }
      gLockMetrics.runtimeExcitations++;
    }
  }
}

static void createStack(const PxTransform &t, PxU32 size, PxReal halfExtent,
                        bool targetStack = false) {
  PxShape *shape = gPhysics->createShape(
      PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
  if (!shape) {
    gInitializationFailed = true;
    return;
  }
  PxFilterData boxFilterData;
  boxFilterData.word0 =
      targetStack ? eHELLO_FILTER_TARGET_BOX : eHELLO_FILTER_BOX;
  shape->setSimulationFilterData(boxFilterData);
  for (PxU32 i = 0; i < size; i++) {
    for (PxU32 j = 0; j < size - i; j++) {
      PxTransform localTm(
          PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) *
          halfExtent);
      PxRigidDynamic *body = gPhysics->createRigidDynamic(t.transform(localTm));
      if (!body) {
        gInitializationFailed = true;
        continue;
      }
      body->attachShape(*shape);
      if (gHeadlessRigidStress)
        body->setSolverIterationCounts(gRigidStressPositionIterations,
                                       gRigidStressVelocityIterations);
      if (!PxRigidBodyExt::updateMassAndInertia(*body, 10.0f)) {
        body->release();
        gInitializationFailed = true;
        continue;
      }
      gScene->addActor(*body);
      gBoxes.push_back(body);
      gPreviousBoxVelocities.push_back(PxVec3(0.0f));
      gPreviousBoxPositions.push_back(body->getGlobalPose().p);
      gBoxContactBaselines.push_back(PxVec3(0.0f));
      gBoxContactPositionBaselines.push_back(body->getGlobalPose().p);
      gBoxContacted.push_back(0);
      gBoxResponseObserved.push_back(0);
      gBoxIsTarget.push_back(targetStack ? 1 : 0);
      gBoxContactFrames.push_back(PX_MAX_U32);
    }
  }
  shape->release();
}

static void spawnBallShot() {
  if (gShotBall)
    return;
  gShotBall = createDynamic(PxTransform(gBallShotPos),
                            PxSphereGeometry(gBallShotRadius), gBallShotVel,
                            eHELLO_FILTER_BALL);
  if (!gShotBall) {
    gMetrics.launchFailures++;
    return;
  }
  gMetrics.ballLaunchMomentum =
      saturateMetric(double(gShotBall->getMass()) * PxAbs(gBallShotVel.z));
  printf("[HelloWorldBallShot] spawn pos=(%.1f,%.1f,%.1f) vel=(%.1f,%.1f,%.1f) "
         "radius=%.2f\n",
         gBallShotPos.x, gBallShotPos.y, gBallShotPos.z, gBallShotVel.x,
         gBallShotVel.y, gBallShotVel.z, gBallShotRadius);
}

static void createRigidStressProjectile() {
  gShotBall = createDynamic(PxTransform(gRigidStressBallPosition),
                            PxSphereGeometry(gRigidStressBallRadius),
                            gRigidStressBallVelocity);
  if (!gShotBall ||
      !PxRigidBodyExt::updateMassAndInertia(*gShotBall,
                                           gRigidStressBallDensity)) {
    gMetrics.launchFailures++;
    gInitializationFailed = true;
    return;
  }
}

static PxReal addMetric(PxReal lhs, PxReal rhs) {
  return saturateMetric(double(lhs) + double(rhs));
}

static bool sampleBodyState(PxRigidDynamic *body, PxVec3 &position,
                            PxQuat &orientation, PxVec3 &linearVelocity,
                            PxReal &speed, PxReal &mechanicalEnergy) {
  mechanicalEnergy = 0.0f;
  if (!body)
    return false;
  const PxTransform pose = body->getGlobalPose();
  position = pose.p;
  orientation = pose.q;
  linearVelocity = body->getLinearVelocity();
  const PxVec3 angularVelocity = body->getAngularVelocity();
  if (!pose.p.isFinite() || !pose.q.isFinite() ||
      !linearVelocity.isFinite() || !angularVelocity.isFinite())
    return false;

  const double quaternionNormSquared =
      double(pose.q.x) * double(pose.q.x) +
      double(pose.q.y) * double(pose.q.y) +
      double(pose.q.z) * double(pose.q.z) +
      double(pose.q.w) * double(pose.q.w);
  gMetrics.maxQuaternionNormError =
      PxMax(gMetrics.maxQuaternionNormError,
            saturateMetric(std::abs(quaternionNormSquared - 1.0)));
  gMetrics.maxAbsPosition =
      PxMax(gMetrics.maxAbsPosition,
            PxMax(PxAbs(position.x),
                  PxMax(PxAbs(position.y), PxAbs(position.z))));
  speed = getSafeMagnitude(linearVelocity);
  const PxReal angularSpeed = getSafeMagnitude(angularVelocity);
  gMetrics.maxAngularSpeed =
      PxMax(gMetrics.maxAngularSpeed, angularSpeed);
  if (speed >= PX_MAX_F32 || angularSpeed >= PX_MAX_F32)
    return false;

  const PxReal mass = body->getMass();
  const PxVec3 inertia = body->getMassSpaceInertiaTensor();
  const PxTransform comPose = pose * body->getCMassLocalPose();
  const PxQuat massFrame = pose.q * body->getCMassLocalPose().q;
  const PxVec3 massAngularVelocity = massFrame.rotateInv(angularVelocity);
  if (!PxIsFinite(mass) || mass <= 0.0f || !inertia.isFinite() ||
      inertia.x <= 0.0f || inertia.y <= 0.0f || inertia.z <= 0.0f ||
      !comPose.p.isFinite() || !massAngularVelocity.isFinite())
    return false;

  const double linearEnergy =
      0.5 * double(mass) *
      (double(linearVelocity.x) * double(linearVelocity.x) +
       double(linearVelocity.y) * double(linearVelocity.y) +
       double(linearVelocity.z) * double(linearVelocity.z));
  const double angularEnergy =
      0.5 * (double(inertia.x) * double(massAngularVelocity.x) *
                 double(massAngularVelocity.x) +
             double(inertia.y) * double(massAngularVelocity.y) *
                 double(massAngularVelocity.y) +
             double(inertia.z) * double(massAngularVelocity.z) *
                 double(massAngularVelocity.z));
  const double potentialEnergy = double(mass) * 9.81 * double(comPose.p.y);
  const double totalEnergy = linearEnergy + angularEnergy + potentialEnergy;
  if (!std::isfinite(totalEnergy) ||
      std::abs(totalEnergy) >= double(PX_MAX_F32))
    return false;
  mechanicalEnergy = PxReal(totalEnergy);
  return true;
}

static void applySleepWakeImpulse() {
  PxRigidDynamic *body = gSleepBodies[eHELLO_SLEEP_FREE];
  if (!body || gSleepMetrics.wakeImpulseApplied)
    return;

  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    if (gSleepBodies[i])
      gSleepMetrics.witnesses[i].wakePosition =
          gSleepBodies[i]->getGlobalPose().p;
  }
  gSleepMetrics.sleepingBeforeWake = body->isSleeping();
  const PxReal impulse = body->getMass() * gSleepWakeTargetDeltaVelocity;
  body->addForce(PxVec3(impulse, 0.0f, 0.0f), PxForceMode::eIMPULSE, true);
  gSleepMetrics.wakeImpulseApplied = true;
  gSleepMetrics.awakeImmediatelyAfterWake = !body->isSleeping();
  gSleepMetrics.wakeCounterAfterImpulse = body->getWakeCounter();
}

static void sampleSleepProbeAfterFetch(PxU32 frameIndex) {
  const PxU32 completedFrame = frameIndex + 1;
  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    PxRigidDynamic *body = gSleepBodies[i];
    if (!body)
      continue;

    HelloWorldSleepWitnessMetrics &metrics = gSleepMetrics.witnesses[i];
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 linearVelocity = body->getLinearVelocity();
    const PxVec3 angularVelocity = body->getAngularVelocity();
    const PxReal wakeCounter = body->getWakeCounter();
    if (!pose.p.isFinite() || !pose.q.isFinite() ||
        !linearVelocity.isFinite() || !angularVelocity.isFinite() ||
        !PxIsFinite(wakeCounter)) {
      metrics.finite = false;
      gMetrics.nanDetected = true;
      continue;
    }

    const PxReal linearSpeed = getSafeMagnitude(linearVelocity);
    const PxReal angularSpeed = getSafeMagnitude(angularVelocity);
    if (linearSpeed >= PX_MAX_F32 || angularSpeed >= PX_MAX_F32) {
      metrics.finite = false;
      gMetrics.nanDetected = true;
      continue;
    }

    metrics.maxLinearSpeed = PxMax(metrics.maxLinearSpeed, linearSpeed);
    metrics.maxAngularSpeed = PxMax(metrics.maxAngularSpeed, angularSpeed);
    metrics.maxAbsPosition =
        PxMax(metrics.maxAbsPosition,
              PxMax(PxAbs(pose.p.x),
                    PxMax(PxAbs(pose.p.y), PxAbs(pose.p.z))));
    const PxMat33 rotation(pose.q);
    const PxReal supportY =
        gSleepBoxHalfExtent *
        (PxAbs(rotation.column0.y) + PxAbs(rotation.column1.y) +
         PxAbs(rotation.column2.y));
    metrics.minTopY = PxMin(metrics.minTopY, pose.p.y + supportY);
    metrics.minWakeCounter = PxMin(metrics.minWakeCounter, wakeCounter);
    metrics.maxWakeCounter = PxMax(metrics.maxWakeCounter, wakeCounter);
    if (wakeCounter < 0.0f)
      metrics.invalidWakeCounterSamples++;

    const bool sleeping = body->isSleeping();
    if (sleeping) {
      metrics.sleepSamples++;
      if (metrics.firstSleepFrame == PX_MAX_U32)
        metrics.firstSleepFrame = completedFrame;
      if (gSleepMetrics.wakeImpulseApplied &&
          metrics.firstSleepAfterWakeFrame == PX_MAX_U32)
        metrics.firstSleepAfterWakeFrame = completedFrame;
    } else {
      metrics.awakeSamples++;
      if (gSleepMetrics.wakeImpulseApplied &&
          metrics.firstAwakeAfterWakeFrame == PX_MAX_U32)
        metrics.firstAwakeAfterWakeFrame = completedFrame;
    }
    metrics.previousSleeping = sleeping;
    metrics.previousSleepingValid = true;
    metrics.finalSleeping = sleeping;
    metrics.finalPosition = pose.p;
    metrics.finalLinearSpeed = linearSpeed;
    metrics.finalAngularSpeed = angularSpeed;
    metrics.finalWakeCounter = wakeCounter;

    if (gSleepMetrics.wakeImpulseApplied) {
      metrics.maxWakeDisplacementX =
          PxMax(metrics.maxWakeDisplacementX,
                PxAbs(pose.p.x - metrics.wakePosition.x));
      metrics.maxWakeVelocityX =
          PxMax(metrics.maxWakeVelocityX, PxAbs(linearVelocity.x));
    }
  }

  if (gHeadlessCase == eHELLO_CASE_SLEEP_WAKE &&
      !gSleepMetrics.wakeImpulseApplied &&
      gSleepBodies[eHELLO_SLEEP_FREE] &&
      !gSleepBodies[eHELLO_SLEEP_FREE]->isSleeping())
    gSleepMetrics.freeAwakeBeforeWakeSamples++;
  gMetrics.completedFrames = completedFrame;
}

static void sampleLockProbeAfterFetch(PxU32 frameIndex) {
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    PxRigidDynamic *bodies[2] = {gLockedBodies[i], gLockControlBodies[i]};
    for (PxU32 bodyIndex = 0; bodyIndex < 2; ++bodyIndex) {
      PxRigidDynamic *body = bodies[bodyIndex];
      if (!body) {
        gLockMetrics.nonFiniteSamples++;
        continue;
      }

      const PxTransform pose = body->getGlobalPose();
      const PxVec3 linearVelocity = body->getLinearVelocity();
      const PxVec3 angularVelocity = body->getAngularVelocity();
      if (!pose.isValid() || !linearVelocity.isFinite() ||
          !angularVelocity.isFinite()) {
        gLockMetrics.nonFiniteSamples++;
        continue;
      }
      gLockMetrics.finiteSamples++;

      const PxVec3 axis = getLockAxis(i);
      const PxTransform &initial =
          gLockInitialPoses[i + bodyIndex * eHELLO_LOCK_WITNESS_COUNT];
      const bool angular = i >= eHELLO_LOCK_ANGULAR_X;
      const PxReal motion =
          angular ? getOrientationDelta(initial.q, pose.q)
                  : PxAbs((pose.p - initial.p).dot(axis));
      const PxReal speed =
          angular ? PxAbs(angularVelocity.dot(axis))
                  : PxAbs(linearVelocity.dot(axis));

      PxReal *motions = bodyIndex == 0u
                            ? gLockMetrics.lockedAxisMotion
                            : gLockMetrics.controlAxisMotion;
      PxReal *speeds = bodyIndex == 0u
                           ? gLockMetrics.lockedAxisSpeed
                           : gLockMetrics.controlAxisSpeed;
      motions[i] = PxMax(motions[i], motion);
      speeds[i] = PxMax(speeds[i], speed);
    }
  }
  gMetrics.completedFrames = frameIndex + 1u;
}

static void sampleDynamics(PxU32 frameIndex, PxU32 settleTailFrames) {
  PxReal maxBoxSpeed = 0.0f;
  PxReal sumBoxSpeed = 0.0f;
  PxU32 countAbove5 = 0;
  PxU32 countAbove15 = 0;
  PxU32 awakeBoxes = 0;
  PxU32 sunkBoxes = 0;
  double totalMechanicalEnergy = 0.0;
  PxReal maxSpeedBoxY = 0.0f;
  const PxU32 currentFrame = frameIndex + 1;

  for (PxU32 boxIndex = 0; boxIndex < gBoxes.size(); ++boxIndex) {
    PxRigidDynamic *body = gBoxes[boxIndex];
    PxVec3 position(0.0f), velocity(0.0f);
    PxQuat orientation(PxIdentity);
    PxReal speed = 0.0f, bodyEnergy = 0.0f;
    if (!sampleBodyState(body, position, orientation, velocity, speed,
                         bodyEnergy)) {
      gMetrics.nanDetected = true;
      continue;
    }
    totalMechanicalEnergy += double(bodyEnergy);
    gMetrics.maxSpeedAll = PxMax(gMetrics.maxSpeedAll, speed);
    gMetrics.maxBoxCenterY = PxMax(gMetrics.maxBoxCenterY, position.y);
    gMetrics.minBoxCenterY = PxMin(gMetrics.minBoxCenterY, position.y);
    if (speed > maxBoxSpeed) {
      maxBoxSpeed = speed;
      maxSpeedBoxY = position.y;
    }
    sumBoxSpeed += speed;
    if (speed > 5.0f)
      countAbove5++;
    if (speed > 15.0f)
      countAbove15++;
    if (!body->isSleeping())
      awakeBoxes++;
    const PxMat33 rotation(orientation);
    const PxReal supportY =
        gBoxHalfExtent *
        (PxAbs(rotation.column0.y) + PxAbs(rotation.column1.y) +
         PxAbs(rotation.column2.y));
    if (position.y + supportY < -0.05f)
      sunkBoxes++;

    if (gBoxContacted[boxIndex] &&
        currentFrame >= gBoxContactFrames[boxIndex] &&
        currentFrame <
            gBoxContactFrames[boxIndex] + gBoxResponseWindowFrames) {
      const PxReal responseAxisDelta = PxMax(
          0.0f, saturateMetric(double(gBoxContactBaselines[boxIndex].z) -
                               double(velocity.z)));
      const PxReal responseDisplacement = PxMax(
          0.0f,
          saturateMetric(double(gBoxContactPositionBaselines[boxIndex].z) -
                         double(position.z)));
      const PxReal responseMomentum = saturateMetric(
          double(body->getMass()) * double(responseAxisDelta));
      gMetrics.maxTargetImpactAxisVelocityDelta =
          PxMax(gMetrics.maxTargetImpactAxisVelocityDelta,
                responseAxisDelta);
      gMetrics.maxTargetResponseMomentum =
          PxMax(gMetrics.maxTargetResponseMomentum, responseMomentum);
      gMetrics.maxTargetResponseDisplacement =
          PxMax(gMetrics.maxTargetResponseDisplacement,
                responseDisplacement);
      const PxReal responseMomentumRatio =
          gMetrics.ballLaunchMomentum > 0.0f
              ? responseMomentum / gMetrics.ballLaunchMomentum
              : 0.0f;
      if (!gBoxResponseObserved[boxIndex] &&
          responseMomentumRatio >= gMinTargetResponseMomentumRatio &&
          responseDisplacement >= gMinTargetResponseDisplacement) {
        gBoxResponseObserved[boxIndex] = 1;
        gMetrics.respondedBoxes++;
      }
    }
    gPreviousBoxVelocities[boxIndex] = velocity;
    gPreviousBoxPositions[boxIndex] = position;
  }
  gMetrics.finalAwakeBoxes = awakeBoxes;
  gMetrics.maxSunkBoxes = PxMax(gMetrics.maxSunkBoxes, sunkBoxes);

  if (gShotBall) {
    PxVec3 position(0.0f), velocity(0.0f);
    PxQuat orientation(PxIdentity);
    PxReal speed = 0.0f, ballEnergy = 0.0f;
    if (!sampleBodyState(gShotBall, position, orientation, velocity, speed,
                         ballEnergy)) {
      gMetrics.nanDetected = true;
    } else {
      totalMechanicalEnergy += double(ballEnergy);
      gMetrics.maxSpeedAll = PxMax(gMetrics.maxSpeedAll, speed);
      gMetrics.maxBallSpeed = PxMax(gMetrics.maxBallSpeed, speed);
      gMetrics.minBallCenterY =
          PxMin(gMetrics.minBallCenterY, position.y);
      gMetrics.maxBallUpVy = PxMax(gMetrics.maxBallUpVy, velocity.y);
      if (!gMetrics.ballSawGround &&
          position.y < (gBallShotRadius + 0.5f)) {
        gMetrics.ballSawGround = true;
        gMetrics.ballVyAtFirstGround = velocity.y;
      }
      if (gMetrics.firstBallBoxContactFrame != PX_MAX_U32 &&
          currentFrame >= gMetrics.firstBallBoxContactFrame &&
          currentFrame < gMetrics.firstBallBoxContactFrame +
                             gBallResponseWindowFrames) {
        const PxReal impactAxisDelta = saturateMetric(
            double(velocity.z) - double(gBallShotVel.z));
        gMetrics.maxBallImpactAxisVelocityDelta =
            PxMax(gMetrics.maxBallImpactAxisVelocityDelta,
                  PxMax(0.0f, impactAxisDelta));
      }
    }
  }

  PxReal currentMechanicalEnergy = 0.0f;
  if (!std::isfinite(totalMechanicalEnergy) ||
      std::abs(totalMechanicalEnergy) >= double(PX_MAX_F32)) {
    gMetrics.nanDetected = true;
    currentMechanicalEnergy = PX_MAX_F32;
  } else {
    currentMechanicalEnergy = PxReal(totalMechanicalEnergy);
  }

  const bool establishEnergyReference =
      !gMetrics.mechanicalReferenceSet &&
      ((!gHeadlessBallShot && frameIndex == 0) ||
       (gHeadlessBallShot && gShotBall && frameIndex >= gBallShotFrame));
  if (establishEnergyReference) {
    gMetrics.mechanicalReferenceSet = true;
    gMetrics.mechanicalEnergyReference = currentMechanicalEnergy;
    gMetrics.maxMechanicalEnergyAfterReference = currentMechanicalEnergy;
  }
  if (gMetrics.mechanicalReferenceSet) {
    gMetrics.maxMechanicalEnergyAfterReference =
        PxMax(gMetrics.maxMechanicalEnergyAfterReference,
              currentMechanicalEnergy);
    gMetrics.finalMechanicalEnergy = currentMechanicalEnergy;
  }
  gMechanicalEnergyHistory.push_back(currentMechanicalEnergy);

  const PxU32 boxCount = static_cast<PxU32>(gBoxes.size());
  const bool inSettleWindow =
      frameIndex + settleTailFrames >= gHeadlessFrameCount;
  if (inSettleWindow && boxCount > 0) {
    gMetrics.maxBoxSpeedSettle =
        PxMax(gMetrics.maxBoxSpeedSettle, maxBoxSpeed);
    gMetrics.awakeBoxesSettle =
        PxMax(gMetrics.awakeBoxesSettle, awakeBoxes);
    gMetrics.lateAwakeSamples += awakeBoxes;
    gMetrics.lateBoxSamples += boxCount;
    gMetrics.tailAvgBoxSpeed =
        addMetric(gMetrics.tailAvgBoxSpeed, maxBoxSpeed);
    gMetrics.tailSpeedSamples++;
  }

  const char *traceEnv = std::getenv("AVBD_HELLOWORLD_TRACE");
  if (traceEnv && traceEnv[0] && traceEnv[0] != '0' &&
      gHeadlessBallShot && frameIndex >= gBallShotFrame &&
      (frameIndex - gBallShotFrame) % 10 == 0) {
    const PxReal avgSpeed =
        boxCount ? sumBoxSpeed / PxReal(boxCount) : 0.0f;
    printf("[HelloWorldTrace] frame=%u maxBoxSpeed=%.3f avgBoxSpeed=%.3f "
           "above5=%u above15=%u awakeBoxes=%u maxSpeedBoxY=%.2f\n",
           frameIndex, maxBoxSpeed, avgSpeed, countAbove5, countAbove15,
           awakeBoxes, maxSpeedBoxY);
  }
  gMetrics.completedFrames = currentFrame;
}

static void finalizeRigidStressMetrics() {
  if (gRigidStressMetrics.stepSamplesMs.empty())
    return;
  PxF64 sumMs = 0.0;
  for (PxU32 i = 0;
       i < static_cast<PxU32>(gRigidStressMetrics.stepSamplesMs.size()); ++i)
    sumMs += gRigidStressMetrics.stepSamplesMs[i];
  std::sort(gRigidStressMetrics.stepSamplesMs.begin(),
            gRigidStressMetrics.stepSamplesMs.end());
  const PxU32 last =
      static_cast<PxU32>(gRigidStressMetrics.stepSamplesMs.size()) - 1u;
  gRigidStressMetrics.avgStepMs = PxReal(
      sumMs / PxF64(gRigidStressMetrics.stepSamplesMs.size()));
  gRigidStressMetrics.p50StepMs =
      gRigidStressMetrics.stepSamplesMs[
          PxU32(PxCeil(0.50f * PxReal(last)))];
  gRigidStressMetrics.p95StepMs =
      gRigidStressMetrics.stepSamplesMs[
          PxU32(PxCeil(0.95f * PxReal(last)))];
  gRigidStressMetrics.maxStepMs =
      gRigidStressMetrics.stepSamplesMs[last];
}

struct RigidStressStateHasher {
  PxU64 low;
  PxU64 high;

  RigidStressStateHasher()
      : low(PxU64(14695981039346656037ULL)),
        high(PxU64(7809847782465536322ULL)) {}

  void appendByte(PxU8 value) {
    low ^= PxU64(value);
    low *= PxU64(1099511628211ULL);
    high ^= PxU64(value ^ 0xa5u);
    high *= PxU64(14029467366897019727ULL);
    high ^= high >> 29;
  }

  void appendU32(PxU32 value) {
    for (PxU32 byteIndex = 0; byteIndex < 4; ++byteIndex)
      appendByte(PxU8((value >> (byteIndex * 8u)) & 0xffu));
  }

  void appendReal(PxReal value) {
    PX_COMPILE_TIME_ASSERT(sizeof(PxReal) == sizeof(PxU32));
    PxU32 bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    appendU32(bits);
  }

  void appendVec3(const PxVec3 &value) {
    appendReal(value.x);
    appendReal(value.y);
    appendReal(value.z);
  }

  void appendQuat(const PxQuat &value) {
    appendReal(value.x);
    appendReal(value.y);
    appendReal(value.z);
    appendReal(value.w);
  }
};

static void appendRigidStressActorState(RigidStressStateHasher &hasher,
                                        PxU32 actorKind, PxU32 actorIndex,
                                        const PxRigidDynamic *body,
                                        PxU32 &actorCount) {
  hasher.appendU32(actorKind);
  hasher.appendU32(actorIndex);
  hasher.appendByte(body ? 1u : 0u);
  if (!body)
    return;
  const PxTransform pose = body->getGlobalPose();
  hasher.appendVec3(pose.p);
  hasher.appendQuat(pose.q);
  hasher.appendVec3(body->getLinearVelocity());
  hasher.appendVec3(body->getAngularVelocity());
  hasher.appendByte(body->isSleeping() ? 1u : 0u);
  hasher.appendReal(body->getWakeCounter());
  ++actorCount;
}

static void finalizeRigidStressWorkMetrics() {
  if (!gRigidStressWorkAttribution)
    return;
#if PX_DEBUG || PX_CHECKED || PX_PROFILE
  const RigidStressProfilerCounts finalCounts =
      gRigidStressProfiler.snapshot();
  gRigidStressWorkMetrics.avbdIslandSolves =
      finalCounts.islandSolves - gRigidStressProfilerBaseline.islandSolves;
  gRigidStressWorkMetrics.avbdInnerSweeps =
      finalCounts.innerSweeps - gRigidStressProfilerBaseline.innerSweeps;
  gRigidStressWorkMetrics.avbdBlockDescentZones =
      finalCounts.blockDescentZones -
      gRigidStressProfilerBaseline.blockDescentZones;
  gRigidStressWorkMetrics.avbdBodyColorPlans =
      finalCounts.bodyColorPlans -
      gRigidStressProfilerBaseline.bodyColorPlans;
  gRigidStressWorkMetrics.avbdBodyColorPasses =
      finalCounts.bodyColorPasses -
      gRigidStressProfilerBaseline.bodyColorPasses;
  gRigidStressWorkMetrics.avbdDualPasses =
      finalCounts.dualPasses - gRigidStressProfilerBaseline.dualPasses;
  gRigidStressWorkMetrics.avbdDualRanges =
      finalCounts.dualRanges - gRigidStressProfilerBaseline.dualRanges;
  gRigidStressWorkMetrics.avbdBodyRanges =
      finalCounts.bodyRanges - gRigidStressProfilerBaseline.bodyRanges;
  gRigidStressWorkMetrics.observed =
      gRigidStressProfilerBaselineCaptured &&
      gRigidStressWorkMetrics.sampledFrames ==
          gHeadlessOptions.frames - gRigidStressWarmupFrames;
#endif
}

static void sampleRigidStressFinalState() {
  gRigidStressMetrics.finiteBoxes = 0;
  gRigidStressMetrics.awakeBoxes = 0;
  gRigidStressMetrics.movedBoxes = 0;
  gRigidStressMetrics.minBoxCenterY = PX_MAX_F32;
  gRigidStressMetrics.maxAbsPosition = 0.0f;
  RigidStressStateHasher stateHasher;
  PxU32 stateActorCount = 0;
  for (PxU32 i = 0; i < static_cast<PxU32>(gBoxes.size()); ++i) {
    const PxRigidDynamic *body = gBoxes[i];
    appendRigidStressActorState(stateHasher, 1u, i, body, stateActorCount);
    if (!body)
      continue;
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 linearVelocity = body->getLinearVelocity();
    const PxVec3 angularVelocity = body->getAngularVelocity();
    if (!pose.isValid() || !linearVelocity.isFinite() ||
        !angularVelocity.isFinite())
      continue;
    gRigidStressMetrics.finiteBoxes++;
    gRigidStressMetrics.awakeBoxes += body->isSleeping() ? 0u : 1u;
    gRigidStressMetrics.minBoxCenterY =
        PxMin(gRigidStressMetrics.minBoxCenterY, pose.p.y);
    gRigidStressMetrics.maxAbsPosition =
        PxMax(gRigidStressMetrics.maxAbsPosition,
              PxMax(PxAbs(pose.p.x),
                    PxMax(PxAbs(pose.p.y), PxAbs(pose.p.z))));
    if (i < static_cast<PxU32>(gPreviousBoxPositions.size()) &&
        (pose.p - gPreviousBoxPositions[i]).magnitudeSquared() > 0.0025f)
      gRigidStressMetrics.movedBoxes++;
  }
  appendRigidStressActorState(stateHasher, 2u, 0u, gShotBall,
                              stateActorCount);
  if (gShotBall) {
    const PxTransform pose = gShotBall->getGlobalPose();
    const PxVec3 linearVelocity = gShotBall->getLinearVelocity();
    const PxVec3 angularVelocity = gShotBall->getAngularVelocity();
    gRigidStressMetrics.finalBallFinite =
        pose.isValid() && linearVelocity.isFinite() &&
        angularVelocity.isFinite();
    if (gRigidStressMetrics.finalBallFinite) {
      gRigidStressMetrics.finalBallPosition = pose.p;
      gRigidStressMetrics.maxAbsPosition =
          PxMax(gRigidStressMetrics.maxAbsPosition,
                PxMax(PxAbs(pose.p.x),
                      PxMax(PxAbs(pose.p.y), PxAbs(pose.p.z))));
    }
  }
  gRigidStressMetrics.stateDigestLow = stateHasher.low;
  gRigidStressMetrics.stateDigestHigh = stateHasher.high;
  gRigidStressMetrics.stateDigestActorCount = stateActorCount;
  finalizeRigidStressWorkMetrics();
  finalizeRigidStressMetrics();
}

struct HelloWorldGateEvaluation {
  PxU32 exitCode;
  const char *status;
  const char *reason;
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

  HelloWorldGateEvaluation()
      : exitCode(Snippets::eHEADLESS_PASS), status("PASS"), reason("none"),
        boxCount(0), targetBoxCount(0), maxEnergyRatio(0.0f),
        tailEnergyW1(0.0f), tailEnergyW2(0.0f), tailEnergyW3(0.0f),
        tailEnergyW4(0.0f), tailEnergyGrowth(0.0f), lateAwakeRatio(0.0f),
        targetResponseMomentumRatio(0.0f), ballResponseFraction(0.0f) {}
};

static void setGateError(HelloWorldGateEvaluation &evaluation,
                         const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static void setGateFailure(HelloWorldGateEvaluation &evaluation,
                           const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
  evaluation.status = "FAIL";
  evaluation.reason = reason;
}

static HelloWorldGateEvaluation evaluateRigidStressGate() {
  HelloWorldGateEvaluation evaluation;
  evaluation.boxCount = static_cast<PxU32>(gBoxes.size());
  const PxU32 expectedProfileFrames =
      gHeadlessOptions.frames - gRigidStressWarmupFrames;
  if (gMetrics.completedFrames != gHeadlessOptions.frames ||
      gMetrics.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (evaluation.boxCount != gRigidStressExpectedBoxCount || !gShotBall ||
      gMetrics.launchFailures)
    setGateError(evaluation, "actor_registry");
  if (gRigidStressMetrics.stateDigestActorCount !=
          gRigidStressExpectedBoxCount + 1u ||
      (!gRigidStressMetrics.stateDigestLow &&
       !gRigidStressMetrics.stateDigestHigh))
    setGateError(evaluation, "state_fingerprint");
  if (gRigidStressMetrics.stepSamplesMs.size() != expectedProfileFrames)
    setGateError(evaluation, "profile_window");
  if (gErrorCallback.getFatalCount() || gMetrics.fetchErrorState)
    setGateFailure(evaluation, "physx_error");
  if (gRigidStressMetrics.finiteBoxes != gRigidStressExpectedBoxCount ||
      !gRigidStressMetrics.finalBallFinite)
    setGateFailure(evaluation, "non_finite");
  if (gRigidStressMetrics.maxAbsPosition > gMaxAbsPositionCap)
    setGateFailure(evaluation, "runaway");
  if (!gRigidStressMetrics.movedBoxes ||
      gRigidStressMetrics.finalBallPosition.z >=
          gRigidStressBallPosition.z - 10.0f)
    setGateFailure(evaluation, "inactive_workload");
  return evaluation;
}

static const char *getSleepProbeFinding() {
  const HelloWorldSleepWitnessMetrics &freeWitness =
      gSleepMetrics.witnesses[eHELLO_SLEEP_FREE];
  const HelloWorldSleepWitnessMetrics &staticWitness =
      gSleepMetrics.witnesses[eHELLO_SLEEP_STATIC_TOUCH];
  if (gHeadlessCase == eHELLO_CASE_SLEEP_IDLE) {
    if (freeWitness.finalSleeping && staticWitness.finalSleeping)
      return "auto-sleep-observed";
    if (!freeWitness.finalSleeping && !staticWitness.finalSleeping)
      return "auto-sleep-missing";
    return "free-sleep-static-touch-awake";
  }
  if (gHeadlessCase == eHELLO_CASE_SLEEP_WAKE) {
    return freeWitness.firstSleepAfterWakeFrame == PX_MAX_U32 ||
                   staticWitness.firstSleepAfterWakeFrame == PX_MAX_U32
               ? "post-wake-resleep-missing"
               : "wake-propagation-resleep-observed";
  }
  return "sleep-disabled-control-ok";
}

static const char *getSleepWakeCounterFinding() {
  if (gHeadlessCase == eHELLO_CASE_SLEEP_DISABLED)
    return "not-gated-scene-disabled";
  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    if (gSleepMetrics.witnesses[i].invalidWakeCounterSamples)
      return "negative-counter-observed";
  }
  return "counter-range-valid";
}

static const char *getSleepCapability() {
  return "SUPPORTED";
}

static HelloWorldGateEvaluation evaluateSleepProbeGate() {
  HelloWorldGateEvaluation evaluation;
  evaluation.boxCount = gSleepMetrics.actorCount;

  if (gMetrics.completedFrames != gHeadlessOptions.frames ||
      gMetrics.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (gSleepMetrics.actorCount != eHELLO_SLEEP_WITNESS_COUNT)
    setGateError(evaluation, "actor_registry");

  if (gErrorCallback.getFatalCount() || gMetrics.fetchErrorState)
    setGateFailure(evaluation, "physx_error");
  if (gMetrics.nanDetected)
    setGateFailure(evaluation, "non_finite");
  if (gSleepMetrics.unexpectedNotifications)
    setGateFailure(evaluation, "unexpected_sleep_notification_actor");

  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    const HelloWorldSleepWitnessMetrics &metrics = gSleepMetrics.witnesses[i];
    if (!metrics.finite)
      setGateFailure(evaluation, "non_finite");
    if (metrics.maxAbsPosition > 100.0f || metrics.maxLinearSpeed > 50.0f ||
        metrics.maxAngularSpeed > 50.0f)
      setGateFailure(evaluation, "runaway");
  }
  if (gSleepMetrics.witnesses[eHELLO_SLEEP_STATIC_TOUCH].minTopY < -0.05f)
    setGateFailure(evaluation, "ground_escape");

  const bool shouldDisableSleeping =
      gHeadlessCase == eHELLO_CASE_SLEEP_DISABLED;
  if (gSleepMetrics.sceneSleepingDisabled != shouldDisableSleeping)
    setGateFailure(evaluation, "sleep_scene_flag");

  const HelloWorldSleepWitnessMetrics &freeWitness =
      gSleepMetrics.witnesses[eHELLO_SLEEP_FREE];
  const HelloWorldSleepWitnessMetrics &staticWitness =
      gSleepMetrics.witnesses[eHELLO_SLEEP_STATIC_TOUCH];
  if (!shouldDisableSleeping) {
    if (freeWitness.invalidWakeCounterSamples ||
        staticWitness.invalidWakeCounterSamples)
      setGateFailure(evaluation, "wake_counter_range");
    if (!PxIsFinite(gSleepMetrics.wakeCounterResetValue) ||
        gSleepMetrics.wakeCounterResetValue <= 0.0f)
      setGateFailure(evaluation, "wake_counter_reset");
  }
  if (gHeadlessCase == eHELLO_CASE_SLEEP_IDLE) {
    if (freeWitness.initialSleeping || staticWitness.initialSleeping)
      setGateFailure(evaluation, "idle_initial_state");
    if (!freeWitness.finalSleeping ||
        freeWitness.firstSleepFrame == PX_MAX_U32 ||
        freeWitness.firstSleepFrame > 60u || !freeWitness.sleepSamples ||
        !freeWitness.sleepNotifications)
      setGateFailure(evaluation, "free_auto_sleep");
    if (!staticWitness.finalSleeping ||
        staticWitness.firstSleepFrame == PX_MAX_U32 ||
        staticWitness.firstSleepFrame > 60u ||
        !staticWitness.sleepSamples || !staticWitness.sleepNotifications)
      setGateFailure(evaluation, "static_touch_auto_sleep");
  } else if (gHeadlessCase == eHELLO_CASE_SLEEP_WAKE) {
    if (!freeWitness.initialSleeping || !staticWitness.initialSleeping ||
        gSleepMetrics.freeAwakeBeforeWakeSamples)
      setGateFailure(evaluation, "pre_wake_sleep_state");
    if (!gSleepMetrics.wakeImpulseApplied ||
        !gSleepMetrics.sleepingBeforeWake ||
        !gSleepMetrics.awakeImmediatelyAfterWake ||
        gSleepMetrics.wakeCounterAfterImpulse <= 0.0f)
      setGateFailure(evaluation, "impulse_autowake");
    if (freeWitness.firstAwakeAfterWakeFrame == PX_MAX_U32 ||
        freeWitness.firstAwakeAfterWakeFrame > gSleepWakeFrame + 2u ||
        !freeWitness.wakeNotifications ||
        freeWitness.firstWakeNotificationFrame > gSleepWakeFrame + 2u)
      setGateFailure(evaluation, "wake_observation");
    if (staticWitness.firstAwakeAfterWakeFrame == PX_MAX_U32 ||
        staticWitness.firstAwakeAfterWakeFrame > gSleepWakeFrame + 2u ||
        !staticWitness.wakeNotifications ||
        staticWitness.firstWakeNotificationFrame > gSleepWakeFrame + 2u)
      setGateFailure(evaluation, "wake_propagation");
    if (freeWitness.maxWakeVelocityX < 0.5f ||
        freeWitness.maxWakeDisplacementX < 0.25f)
      setGateFailure(evaluation, "wake_response");
    if (staticWitness.maxWakeVelocityX < 0.05f ||
        staticWitness.maxWakeDisplacementX < 0.05f)
      setGateFailure(evaluation, "propagated_wake_response");
    if (!freeWitness.finalSleeping ||
        freeWitness.firstSleepAfterWakeFrame == PX_MAX_U32 ||
        freeWitness.firstSleepAfterWakeFrame <= gSleepWakeFrame ||
        freeWitness.firstSleepAfterWakeFrame > gHeadlessOptions.frames ||
        freeWitness.sleepNotifications < 2u)
      setGateFailure(evaluation, "post_wake_resleep");
    if (!staticWitness.finalSleeping ||
        staticWitness.firstSleepAfterWakeFrame == PX_MAX_U32 ||
        staticWitness.firstSleepAfterWakeFrame <= gSleepWakeFrame ||
        staticWitness.firstSleepAfterWakeFrame > gHeadlessOptions.frames ||
        staticWitness.sleepNotifications < 2u)
      setGateFailure(evaluation, "propagated_post_wake_resleep");
  } else {
    for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
      const HelloWorldSleepWitnessMetrics &metrics =
          gSleepMetrics.witnesses[i];
      if (metrics.initialSleeping || metrics.finalSleeping ||
          metrics.sleepSamples || metrics.sleepNotifications)
        setGateFailure(evaluation, "sleep_disabled_control");
    }
  }
  return evaluation;
}

static HelloWorldGateEvaluation evaluateLockProbeGate() {
  HelloWorldGateEvaluation evaluation;
  evaluation.boxCount = gLockMetrics.actorCount;
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    gLockMetrics.maxLockedAxisMotion =
        PxMax(gLockMetrics.maxLockedAxisMotion,
              gLockMetrics.lockedAxisMotion[i]);
    gLockMetrics.maxLockedAxisSpeed =
        PxMax(gLockMetrics.maxLockedAxisSpeed,
              gLockMetrics.lockedAxisSpeed[i]);
    gLockMetrics.minControlAxisMotion =
        PxMin(gLockMetrics.minControlAxisMotion,
              gLockMetrics.controlAxisMotion[i]);
    gLockMetrics.minControlAxisSpeed =
        PxMin(gLockMetrics.minControlAxisSpeed,
              gLockMetrics.controlAxisSpeed[i]);
    gLockMetrics.maxControlAxisMotion =
        PxMax(gLockMetrics.maxControlAxisMotion,
              gLockMetrics.controlAxisMotion[i]);
    gLockMetrics.maxControlAxisSpeed =
        PxMax(gLockMetrics.maxControlAxisSpeed,
              gLockMetrics.controlAxisSpeed[i]);
  }

  if (gMetrics.completedFrames != gHeadlessOptions.frames ||
      gMetrics.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (gLockMetrics.actorCount != eHELLO_LOCK_WITNESS_COUNT * 2u ||
      gLockMetrics.lockFlagsReadback != eHELLO_LOCK_WITNESS_COUNT ||
      gLockMetrics.runtimeExcitations != eHELLO_LOCK_WITNESS_COUNT * 2u)
    setGateError(evaluation, "actor_or_flag_registry");
  if (gLockMetrics.finiteSamples !=
      gHeadlessOptions.frames * eHELLO_LOCK_WITNESS_COUNT * 2u)
    setGateError(evaluation, "sample_registry");

  if (gErrorCallback.getFatalCount() || gMetrics.fetchErrorState)
    setGateFailure(evaluation, "physx_error");
  if (gLockMetrics.nonFiniteSamples)
    setGateFailure(evaluation, "non_finite");
  if (gLockMetrics.minControlAxisMotion < gLockControlMotionMinimum ||
      gLockMetrics.minControlAxisSpeed < gLockControlSpeedMinimum)
    setGateFailure(evaluation, "control_response");
  if (gLockMetrics.maxLockedAxisMotion > gLockMotionTolerance ||
      gLockMetrics.maxLockedAxisSpeed > gLockSpeedTolerance)
    setGateFailure(evaluation, "locked_axis_motion");
  return evaluation;
}

static PxReal getTailWindowMean(const std::vector<PxReal> &values,
                                PxU32 windowFromEnd) {
  const PxU32 windowSize = 60;
  const PxU32 required = (windowFromEnd + 1) * windowSize;
  if (values.size() < required)
    return 0.0f;
  const PxU32 end = static_cast<PxU32>(values.size()) -
                    windowFromEnd * windowSize;
  const PxU32 begin = end - windowSize;
  double sum = 0.0;
  for (PxU32 i = begin; i < end; ++i)
    sum += double(values[i]);
  return saturateMetric(sum / double(windowSize));
}

static HelloWorldGateEvaluation evaluateGate() {
  if (gHeadlessSleepProbe)
    return evaluateSleepProbeGate();
  if (gHeadlessLockProbe)
    return evaluateLockProbeGate();
  if (gHeadlessRigidStress)
    return evaluateRigidStressGate();

  HelloWorldGateEvaluation evaluation;
  evaluation.boxCount = static_cast<PxU32>(gBoxes.size());
  for (PxU32 i = 0; i < gBoxIsTarget.size(); ++i)
    evaluation.targetBoxCount += gBoxIsTarget[i] ? 1u : 0u;
  evaluation.tailEnergyW1 = getTailWindowMean(gMechanicalEnergyHistory, 3);
  evaluation.tailEnergyW2 = getTailWindowMean(gMechanicalEnergyHistory, 2);
  evaluation.tailEnergyW3 = getTailWindowMean(gMechanicalEnergyHistory, 1);
  evaluation.tailEnergyW4 = getTailWindowMean(gMechanicalEnergyHistory, 0);
  if (gMetrics.mechanicalEnergyReference > 0.0f) {
    evaluation.maxEnergyRatio =
        gMetrics.maxMechanicalEnergyAfterReference /
        gMetrics.mechanicalEnergyReference;
  }
  if (PxAbs(evaluation.tailEnergyW1) > 1e-6f) {
    evaluation.tailEnergyGrowth =
        (evaluation.tailEnergyW4 - evaluation.tailEnergyW1) /
        PxAbs(evaluation.tailEnergyW1);
  }
  if (gMetrics.lateBoxSamples) {
    evaluation.lateAwakeRatio =
        PxReal(gMetrics.lateAwakeSamples) /
        PxReal(gMetrics.lateBoxSamples);
  }
  if (gMetrics.ballLaunchMomentum > 0.0f) {
    evaluation.targetResponseMomentumRatio =
        gMetrics.maxTargetResponseMomentum / gMetrics.ballLaunchMomentum;
  }
  evaluation.ballResponseFraction =
      gMetrics.maxBallImpactAxisVelocityDelta / PxAbs(gBallShotVel.z);

  if (gMetrics.completedFrames != gHeadlessOptions.frames ||
      gMetrics.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (evaluation.boxCount != gExpectedBoxCount ||
      evaluation.targetBoxCount != gExpectedTargetBoxCount)
    setGateError(evaluation, "actor_registry");
  if (gMetrics.launchFailures)
    setGateError(evaluation, "projectile_launch");
  if (!gMetrics.mechanicalReferenceSet ||
      gMechanicalEnergyHistory.size() < 240)
    setGateError(evaluation, "energy_window");

  if (gErrorCallback.getFatalCount() || gMetrics.fetchErrorState)
    setGateFailure(evaluation, "physx_error");
  if (gMetrics.nanDetected)
    setGateFailure(evaluation, "non_finite");
  if (gMetrics.maxQuaternionNormError > 1e-3f)
    setGateFailure(evaluation, "quaternion_norm");
  if (gMetrics.maxAbsPosition > gMaxAbsPositionCap ||
      gMetrics.maxSpeedAll > gMaxLinearSpeedCap ||
      gMetrics.maxAngularSpeed > gMaxAngularSpeedCap)
    setGateFailure(evaluation, "runaway");
  if (gMetrics.maxSunkBoxes)
    setGateFailure(evaluation, "box_ground_escape");
  if (gHeadlessBallShot &&
      gMetrics.minBallCenterY < -gBallShotRadius)
    setGateFailure(evaluation, "ball_ground_escape");
  if (gMetrics.mechanicalEnergyReference <= 0.0f ||
      evaluation.maxEnergyRatio > 1.0f + gMechanicalEnergyGainCap)
    setGateFailure(evaluation, "mechanical_energy_gain");

  if (evaluation.tailEnergyGrowth > gTailEnergyGrowthCap)
    setGateFailure(evaluation, "tail_energy_growth");

  if (gHeadlessCase == eHELLO_CASE_STACK_SETTLE) {
    if (gMetrics.maxBoxSpeedSettle > gSettleSpeedCap)
      setGateFailure(evaluation, "settle_speed");
  } else {
    if (!gShotBall || !gMetrics.ballLaunchMomentum)
      setGateError(evaluation, "projectile_missing");
    if (!gMetrics.ballBoxContactEvents || !gMetrics.ballBoxContactPoints ||
        !gMetrics.contactedBoxes)
      setGateFailure(evaluation, "missing_target_contact");
    if (gMetrics.firstBallBoxContactFrame != PX_MAX_U32 &&
        (gMetrics.firstBallBoxContactFrame < gBallShotFrame + 12u ||
         gMetrics.firstBallBoxContactFrame > gBallShotFrame + 60u))
      setGateFailure(evaluation, "target_contact_window");
    if (gMetrics.firstBallBoxContactFrame != PX_MAX_U32 &&
        gMetrics.firstBallBoxContactFrame + gBoxResponseWindowFrames >
            gMetrics.completedFrames)
      setGateFailure(evaluation, "incomplete_response_window");
    if (!gMetrics.respondedBoxes ||
        evaluation.targetResponseMomentumRatio <
            gMinTargetResponseMomentumRatio ||
        gMetrics.maxTargetResponseDisplacement <
            gMinTargetResponseDisplacement)
      setGateFailure(evaluation, "missing_target_response");
    if (evaluation.ballResponseFraction < gMinBallResponseFraction)
      setGateFailure(evaluation, "missing_projectile_response");
  }
  return evaluation;
}

void initPhysics(bool interactive) {
  resetRuntimeState();
  gErrorCallback.reset();
  gInitializationFailed = false;
  gExtensionsInitialized = false;
  gFoundation =
      PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
  if (!gFoundation) {
    gInitializationFailed = true;
    return;
  }
#if PX_DEBUG || PX_CHECKED || PX_PROFILE
  if (!interactive && gHeadlessRigidStress && gRigidStressWorkAttribution)
    PxSetProfilerCallback(&gRigidStressProfiler);
#endif

  if (interactive) {
    gPvd = PxCreatePvd(*gFoundation);
    if (gPvd) {
      PxPvdTransport *transport =
          PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
      if (transport)
        gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
    }
  }

  gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
                             PxTolerancesScale(), true, gPvd);
  if (!gPhysics) {
    gInitializationFailed = true;
    return;
  }
  gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
  if (!gExtensionsInitialized) {
    gInitializationFailed = true;
    return;
  }

  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  if (gHeadlessRigidStress) {
    if (gRigidStressAvbdIterationsExplicit)
      sceneDesc.avbdIterations = gRigidStressRequestedAvbdIterations;
    if (gRigidStressAvbdJointIterationOverrideExplicit)
      sceneDesc.avbdJointIterationOverride =
          gRigidStressRequestedAvbdJointIterationOverride;
    if (gRigidStressAvbdEarlyStopExplicit)
      sceneDesc.avbdEnableEarlyStop =
          gRigidStressRequestedAvbdEnableEarlyStop;
    gRigidStressSceneAvbdIterations = sceneDesc.avbdIterations;
    gRigidStressSceneAvbdJointIterationOverride =
        sceneDesc.avbdJointIterationOverride;
    gRigidStressSceneAvbdEnableEarlyStop = sceneDesc.avbdEnableEarlyStop;
    if (gRigidStressEnhancedDeterminismRequested)
      sceneDesc.flags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
  }
  sceneDesc.gravity = (gHeadlessSleepProbe || gHeadlessLockProbe)
                          ? PxVec3(0.0f)
                          : PxVec3(0.0f, -9.81f, 0.0f);
  if (gHeadlessCase == eHELLO_CASE_SLEEP_DISABLED)
    sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
  gDispatcher = PxDefaultCpuDispatcherCreate(
      interactive ? 2 : gHeadlessOptions.dispatcherThreads);
  if (!gDispatcher) {
    gInitializationFailed = true;
    return;
  }
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader =
      (interactive || gHeadlessRigidStress)
          ? PxDefaultSimulationFilterShader
          : helloWorldFilterShader;
  sceneDesc.simulationEventCallback =
      (interactive || gHeadlessRigidStress) ? NULL : &gSimulationCallback;
  sceneDesc.solverType = gSolverType;
  gScene = gPhysics->createScene(sceneDesc);
  if (!gScene) {
    gInitializationFailed = true;
    return;
  }
  if (gHeadlessRigidStress)
    gRigidStressEnhancedDeterminismObserved =
        gScene->getFlags().isSet(
            PxSceneFlag::eENABLE_ENHANCED_DETERMINISM);
  gSleepMetrics.sceneSleepingDisabled =
      gScene->getFlags().isSet(PxSceneFlag::eDISABLE_SLEEPING);
  gSleepMetrics.wakeCounterResetValue = gScene->getWakeCounterResetValue();

  PxPvdSceneClient *pvdClient = gScene->getScenePvdClient();
  if (pvdClient) {
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
  }
  gMaterial = gPhysics->createMaterial(
      0.5f, 0.5f,
      (gHeadlessSleepProbe || gHeadlessLockProbe) ? 0.0f : 0.6f);
  if (!gMaterial) {
    gInitializationFailed = true;
    return;
  }

  PxRigidStatic *groundPlane =
      PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  if (!groundPlane) {
    gInitializationFailed = true;
    return;
  }
  gScene->addActor(*groundPlane);

  if (gHeadlessSleepProbe) {
    createSleepProbeFixture();
    printf("[HelloWorld] init solver=%s sleepProbe=%s actors=%u "
           "gravity=zero ground=plane sleepingDisabled=%u headless=%s\n",
           Snippets::getSolverTypeName(gSolverType),
            getHeadlessCaseName(gHeadlessCase), gSleepMetrics.actorCount,
            gSleepMetrics.sceneSleepingDisabled ? 1u : 0u,
            gHeadlessMode ? "yes" : "no");
  } else if (gHeadlessLockProbe) {
    createLockProbeFixture();
    printf("[HelloWorld] init solver=%s lockProbe=%s actors=%u "
           "gravity=zero ground=plane headless=%s\n",
           Snippets::getSolverTypeName(gSolverType),
           getHeadlessCaseName(gHeadlessCase), gLockMetrics.actorCount,
           gHeadlessMode ? "yes" : "no");
  } else if (gHeadlessRigidStress) {
    gBoxes.reserve(gRigidStressExpectedBoxCount);
    gPreviousBoxVelocities.reserve(gRigidStressExpectedBoxCount);
    gPreviousBoxPositions.reserve(gRigidStressExpectedBoxCount);
    gBoxContactBaselines.reserve(gRigidStressExpectedBoxCount);
    gBoxContactPositionBaselines.reserve(gRigidStressExpectedBoxCount);
    gBoxContacted.reserve(gRigidStressExpectedBoxCount);
    gBoxResponseObserved.reserve(gRigidStressExpectedBoxCount);
    gBoxIsTarget.reserve(gRigidStressExpectedBoxCount);
    gBoxContactFrames.reserve(gRigidStressExpectedBoxCount);
    const PxReal stackSpacing =
        gRigidStressLayout == eRIGID_STRESS_CONNECTED_ISLAND
            ? 2.0f * gRigidStressBoxHalfExtent
            : 10.0f;
    for (PxU32 i = 0; i < gRigidStressStackCount; ++i)
      createStack(PxTransform(PxVec3(0, 0, stackZ -= stackSpacing)),
                  gRigidStressStackSize, gRigidStressBoxHalfExtent);
    createRigidStressProjectile();
    if (gBoxes.size() != gRigidStressExpectedBoxCount)
      gInitializationFailed = true;
    printf("[HelloWorld] init solver=%s workload=rigid-stress layout=%s stacks=%u "
           "boxes=%u projectile=1 ground=plane gpuDynamics=0 "
           "broadphase=cpu headless=%s\n",
           Snippets::getSolverTypeName(gSolverType), getRigidStressLayoutName(),
           gRigidStressStackCount,
           static_cast<PxU32>(gBoxes.size()),
           gHeadlessMode ? "yes" : "no");
  } else {
    for (PxU32 i = 0; i < 5; i++)
      createStack(PxTransform(PxVec3(0, 0, stackZ -= 10.0f)), 10, 2.0f,
                  i == 0);
    if (gBoxes.size() != gExpectedBoxCount)
      gInitializationFailed = true;

    printf("[HelloWorld] init solver=%s stacks=5 boxes=%u targetBoxes=%u "
           "ground=plane headless=%s\n",
           Snippets::getSolverTypeName(gSolverType),
           static_cast<PxU32>(gBoxes.size()), gExpectedTargetBoxCount,
           gHeadlessMode ? "yes" : "no");
  }
}

static void beginRigidStressAttributionFrame() {
  if (!gRigidStressWorkAttribution ||
      gRigidStressProfilerBaselineCaptured ||
      gSimFrame != gRigidStressWarmupFrames)
    return;
#if PX_DEBUG || PX_CHECKED || PX_PROFILE
  gRigidStressProfilerBaseline = gRigidStressProfiler.snapshot();
  gRigidStressProfilerBaselineCaptured = true;
#endif
}

static void sampleRigidStressWorkAfterFetch() {
  if (!gRigidStressWorkAttribution ||
      gSimFrame < gRigidStressWarmupFrames)
    return;
  PxSimulationStatistics stats;
  gScene->getSimulationStatistics(stats);
  PxU32 awakeDynamicBodies = 0;
  for (PxU32 i = 0; i < static_cast<PxU32>(gBoxes.size()); ++i)
    awakeDynamicBodies +=
        gBoxes[i] && !gBoxes[i]->isSleeping() ? 1u : 0u;
  awakeDynamicBodies +=
      gShotBall && !gShotBall->isSleeping() ? 1u : 0u;
  gRigidStressWorkMetrics.awakeDynamicBodies += awakeDynamicBodies;
  gRigidStressWorkMetrics.sceneStatsActiveDynamicBodies +=
      stats.nbActiveDynamicBodies;
  gRigidStressWorkMetrics.discreteContactPairs +=
      stats.nbDiscreteContactPairsTotal;
  gRigidStressWorkMetrics.contactPairsWithContacts +=
      stats.nbDiscreteContactPairsWithContacts;
  gRigidStressWorkMetrics.axisSolverConstraints +=
      stats.nbAxisSolverConstraints;
  gRigidStressWorkMetrics.solverPartitions += stats.nbPartitions;
  gRigidStressWorkMetrics.peakAwakeDynamicBodies =
      PxMax(gRigidStressWorkMetrics.peakAwakeDynamicBodies,
            awakeDynamicBodies);
  gRigidStressWorkMetrics.peakSceneStatsActiveDynamicBodies =
      PxMax(gRigidStressWorkMetrics.peakSceneStatsActiveDynamicBodies,
            stats.nbActiveDynamicBodies);
  gRigidStressWorkMetrics.peakDiscreteContactPairs =
      PxMax(gRigidStressWorkMetrics.peakDiscreteContactPairs,
            stats.nbDiscreteContactPairsTotal);
  gRigidStressWorkMetrics.peakContactPairsWithContacts =
      PxMax(gRigidStressWorkMetrics.peakContactPairsWithContacts,
            stats.nbDiscreteContactPairsWithContacts);
  gRigidStressWorkMetrics.peakAxisSolverConstraints =
      PxMax(gRigidStressWorkMetrics.peakAxisSolverConstraints,
            stats.nbAxisSolverConstraints);
  gRigidStressWorkMetrics.peakSolverPartitions =
      PxMax(gRigidStressWorkMetrics.peakSolverPartitions,
            stats.nbPartitions);
  ++gRigidStressWorkMetrics.sampledFrames;
}

void stepPhysics(bool interactive) {
  if (!gScene)
    return;
  if (!interactive && gHeadlessBallShot && gSimFrame == gBallShotFrame)
    spawnBallShot();
  if (!interactive && gHeadlessCase == eHELLO_CASE_SLEEP_WAKE &&
      gSimFrame == gSleepWakeFrame)
    applySleepWakeImpulse();
  if (!interactive && gHeadlessLockProbe &&
      gSimFrame == gLockImpulseFrame)
    applyLockProbeRuntimeExcitation();
  PxU32 errorState = 0;
  bool fetched = false;
  if (!interactive && gHeadlessRigidStress) {
    beginRigidStressAttributionFrame();
    const bool profileFrame = gSimFrame >= gRigidStressWarmupFrames;
    PxTime stepTimer;
    gScene->simulate(gHeadlessOptions.dt);
    fetched = gScene->fetchResults(true, &errorState);
    if (fetched && profileFrame) {
      gRigidStressMetrics.stepSamplesMs.push_back(
          PxReal(stepTimer.getElapsedSeconds() * 1000.0));
    }
    if (fetched)
      sampleRigidStressWorkAfterFetch();
  } else {
    gScene->simulate(interactive ? (1.0f / 60.0f) : gHeadlessOptions.dt);
    fetched = gScene->fetchResults(true, &errorState);
  }
  if (!fetched) {
    if (!interactive) {
      gMetrics.fetchFailures++;
      gMetrics.fetchErrorState |= errorState;
    }
    return;
  }
  if (!interactive && errorState)
    gMetrics.fetchErrorState |= errorState;
  if (!interactive) {
    if (gHeadlessSleepProbe)
      sampleSleepProbeAfterFetch(gSimFrame);
    else if (gHeadlessLockProbe)
      sampleLockProbeAfterFetch(gSimFrame);
    else if (gHeadlessRigidStress)
      gMetrics.completedFrames = gSimFrame + 1u;
    else
      sampleDynamics(gSimFrame, 120);
  }
  ++gSimFrame;
}

void cleanupPhysics(bool interactive) {
  PX_RELEASE(gScene);
  PX_RELEASE(gMaterial);
  PX_RELEASE(gDispatcher);
  if (gExtensionsInitialized) {
    PxCloseExtensions();
    gExtensionsInitialized = false;
  }
  PX_RELEASE(gPhysics);
  if (gPvd) {
    PxPvdTransport *transport = gPvd->getTransport();
    PX_RELEASE(gPvd);
    PX_RELEASE(transport);
  }
#if PX_DEBUG || PX_CHECKED || PX_PROFILE
  if (PxGetProfilerCallback() == &gRigidStressProfiler)
    PxSetProfilerCallback(NULL);
#endif
  PX_RELEASE(gFoundation);
  gShotBall = NULL;
  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i)
    gSleepBodies[i] = NULL;
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    gLockedBodies[i] = NULL;
    gLockControlBodies[i] = NULL;
  }
  gBoxes.clear();
  gPreviousBoxVelocities.clear();
  gPreviousBoxPositions.clear();
  gBoxContactBaselines.clear();
  gBoxContactPositionBaselines.clear();
  gBoxContacted.clear();
  gBoxResponseObserved.clear();
  gBoxIsTarget.clear();
  gBoxContactFrames.clear();
  if (interactive)
    printf("SnippetHelloWorld done.\n");
}

#ifdef RENDER_SNIPPET
void keyPress(unsigned char key, const PxTransform &camera) {
  switch (toupper(key)) {
  case 'B':
    createStack(PxTransform(PxVec3(0, 0, stackZ -= 10.0f)), 10, 2.0f);
    break;
  case ' ':
    createDynamic(camera, PxSphereGeometry(3.0f),
                  camera.rotate(PxVec3(0, 0, -1)) * 200);
    break;
  }
}
#endif

static const char *getSleepWitnessName(PxU32 witness) {
  if (gHeadlessCase == eHELLO_CASE_SLEEP_WAKE)
    return witness == eHELLO_SLEEP_FREE ? "wake-source" : "wake-peer";
  return witness == eHELLO_SLEEP_FREE ? "free" : "static-touch";
}

static void printSleepProbeDetails() {
  for (PxU32 i = 0; i < eHELLO_SLEEP_WITNESS_COUNT; ++i) {
    const HelloWorldSleepWitnessMetrics &metrics = gSleepMetrics.witnesses[i];
    printf(
        "[SnippetHelloWorldSleep] witness=%s initialSleeping=%u "
        "finalSleeping=%u firstSleepFrame=%u firstSleepAfterWakeFrame=%u "
        "sleepSamples=%u awakeSamples=%u sleepNotify=%u wakeNotify=%u "
        "firstSleepNotifyFrame=%u firstWakeNotifyFrame=%u "
        "initialWakeCounter=%.9g finalWakeCounter=%.9g "
        "wakeCounterRange=[%.9g,%.9g] invalidWakeCounterSamples=%u "
        "maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
        "maxWakeVelocityX=%.9g maxWakeDisplacementX=%.9g "
        "finalPosition=(%.9g,%.9g,%.9g) finite=%u\n",
        getSleepWitnessName(i), metrics.initialSleeping ? 1u : 0u,
        metrics.finalSleeping ? 1u : 0u, metrics.firstSleepFrame,
        metrics.firstSleepAfterWakeFrame, metrics.sleepSamples,
        metrics.awakeSamples, metrics.sleepNotifications,
        metrics.wakeNotifications, metrics.firstSleepNotificationFrame,
        metrics.firstWakeNotificationFrame, double(metrics.initialWakeCounter),
        double(metrics.finalWakeCounter), double(metrics.minWakeCounter),
        double(metrics.maxWakeCounter), metrics.invalidWakeCounterSamples,
        double(metrics.maxLinearSpeed),
        double(metrics.maxAngularSpeed), double(metrics.maxWakeVelocityX),
        double(metrics.maxWakeDisplacementX), double(metrics.finalPosition.x),
        double(metrics.finalPosition.y), double(metrics.finalPosition.z),
        metrics.finite ? 1u : 0u);
  }
  printf("[SnippetHelloWorldSleepFinding] case=%s finding=%s "
         "wakeCounterFinding=%s "
         "sceneSleepingDisabled=%u wakeCounterReset=%.9g "
         "wakeImpulseApplied=%u sleepingBeforeWake=%u "
         "awakeImmediatelyAfterWake=%u wakeCounterAfterImpulse=%.9g\n",
         getHeadlessCaseName(gHeadlessCase), getSleepProbeFinding(),
         getSleepWakeCounterFinding(),
         gSleepMetrics.sceneSleepingDisabled ? 1u : 0u,
         double(gSleepMetrics.wakeCounterResetValue),
         gSleepMetrics.wakeImpulseApplied ? 1u : 0u,
         gSleepMetrics.sleepingBeforeWake ? 1u : 0u,
         gSleepMetrics.awakeImmediatelyAfterWake ? 1u : 0u,
         double(gSleepMetrics.wakeCounterAfterImpulse));
}

static void printSleepProbeGateResult(
    const HelloWorldGateEvaluation &evaluation, PxU32 physicsErrors,
    PxU32 physicsWarnings) {
  const HelloWorldSleepWitnessMetrics &freeWitness =
      gSleepMetrics.witnesses[eHELLO_SLEEP_FREE];
  const HelloWorldSleepWitnessMetrics &staticWitness =
      gSleepMetrics.witnesses[eHELLO_SLEEP_STATIC_TOUCH];
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
      "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
      "dispatcherThreads=%u capability=%s validation=GATED status=%s "
      "reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
      "fetchFailures=%u fetchErrorState=%u actorCount=%u "
      "sceneSleepingDisabled=%u probeFinding=%s wakeCounterFinding=%s "
      "freeSleepLifecycleGate=HARD staticTouchSleepGate=HARD "
      "wakeCounterReset=%.9g "
      "wakeFrame=%u wakeImpulseApplied=%u sleepingBeforeWake=%u "
      "awakeImmediatelyAfterWake=%u wakeCounterAfterImpulse=%.9g "
      "freeInitialSleeping=%u freeFinalSleeping=%u freeFirstSleepFrame=%u "
      "freeFirstSleepAfterWakeFrame=%u freeSleepSamples=%u "
      "freeAwakeSamples=%u freeSleepNotify=%u freeWakeNotify=%u "
      "freeFirstWakeNotifyFrame=%u freeInitialWakeCounter=%.9g "
      "freeFinalWakeCounter=%.9g freeMinWakeCounter=%.9g "
      "freeMaxWakeCounter=%.9g freeInvalidWakeCounterSamples=%u "
      "freeMaxLinearSpeed=%.9g "
      "freeMaxAngularSpeed=%.9g freeMaxWakeVelocityX=%.9g "
      "freeMaxWakeDisplacementX=%.9g staticInitialSleeping=%u "
      "staticFinalSleeping=%u staticFirstSleepFrame=%u "
      "staticFirstSleepAfterWakeFrame=%u "
      "staticSleepSamples=%u staticAwakeSamples=%u staticSleepNotify=%u "
      "staticWakeNotify=%u staticFirstWakeNotifyFrame=%u "
      "staticInitialWakeCounter=%.9g "
      "staticFinalWakeCounter=%.9g staticMinWakeCounter=%.9g "
      "staticMaxWakeCounter=%.9g staticInvalidWakeCounterSamples=%u "
      "staticMaxLinearSpeed=%.9g "
      "staticMaxAngularSpeed=%.9g staticMaxWakeVelocityX=%.9g "
      "staticMaxWakeDisplacementX=%.9g staticMinTopY=%.9g "
      "unexpectedNotifications=%u freeAwakeBeforeWakeSamples=%u\n",
      getHeadlessCaseName(gHeadlessCase),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, getSleepCapability(),
      evaluation.status, evaluation.reason,
      gMetrics.nanDetected ? 1u : 0u, physicsErrors, physicsWarnings,
      gMetrics.fetchFailures, gMetrics.fetchErrorState,
      gSleepMetrics.actorCount,
      gSleepMetrics.sceneSleepingDisabled ? 1u : 0u,
      getSleepProbeFinding(), getSleepWakeCounterFinding(),
      double(gSleepMetrics.wakeCounterResetValue),
      gSleepWakeFrame, gSleepMetrics.wakeImpulseApplied ? 1u : 0u,
      gSleepMetrics.sleepingBeforeWake ? 1u : 0u,
      gSleepMetrics.awakeImmediatelyAfterWake ? 1u : 0u,
      double(gSleepMetrics.wakeCounterAfterImpulse),
      freeWitness.initialSleeping ? 1u : 0u,
      freeWitness.finalSleeping ? 1u : 0u, freeWitness.firstSleepFrame,
      freeWitness.firstSleepAfterWakeFrame, freeWitness.sleepSamples,
      freeWitness.awakeSamples, freeWitness.sleepNotifications,
      freeWitness.wakeNotifications, freeWitness.firstWakeNotificationFrame,
      double(freeWitness.initialWakeCounter),
      double(freeWitness.finalWakeCounter),
      double(freeWitness.minWakeCounter),
      double(freeWitness.maxWakeCounter),
      freeWitness.invalidWakeCounterSamples,
      double(freeWitness.maxLinearSpeed),
      double(freeWitness.maxAngularSpeed),
      double(freeWitness.maxWakeVelocityX),
      double(freeWitness.maxWakeDisplacementX),
      staticWitness.initialSleeping ? 1u : 0u,
      staticWitness.finalSleeping ? 1u : 0u, staticWitness.firstSleepFrame,
      staticWitness.firstSleepAfterWakeFrame,
      staticWitness.sleepSamples, staticWitness.awakeSamples,
      staticWitness.sleepNotifications, staticWitness.wakeNotifications,
      staticWitness.firstWakeNotificationFrame,
      double(staticWitness.initialWakeCounter),
      double(staticWitness.finalWakeCounter),
      double(staticWitness.minWakeCounter),
      double(staticWitness.maxWakeCounter),
      staticWitness.invalidWakeCounterSamples,
      double(staticWitness.maxLinearSpeed),
      double(staticWitness.maxAngularSpeed),
      double(staticWitness.maxWakeVelocityX),
      double(staticWitness.maxWakeDisplacementX),
      double(staticWitness.minTopY),
      gSleepMetrics.unexpectedNotifications,
      gSleepMetrics.freeAwakeBeforeWakeSamples);
}

static const char *getLockWitnessName(PxU32 witness) {
  static const char *names[eHELLO_LOCK_WITNESS_COUNT] = {
      "linear-x",  "linear-y",  "linear-z",
      "angular-x", "angular-y", "angular-z"};
  return names[witness];
}

static void printLockProbeDetails() {
  for (PxU32 i = 0; i < eHELLO_LOCK_WITNESS_COUNT; ++i) {
    printf("[SnippetHelloWorldLock] witness=%s lockedMotion=%.9g "
           "lockedSpeed=%.9g controlMotion=%.9g controlSpeed=%.9g\n",
           getLockWitnessName(i), double(gLockMetrics.lockedAxisMotion[i]),
           double(gLockMetrics.lockedAxisSpeed[i]),
           double(gLockMetrics.controlAxisMotion[i]),
           double(gLockMetrics.controlAxisSpeed[i]));
  }
}

static void printLockProbeGateResult(
    const HelloWorldGateEvaluation &evaluation, PxU32 physicsErrors,
    PxU32 physicsWarnings) {
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
      "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
      "dispatcherThreads=%u capability=SUPPORTED validation=GATED status=%s "
      "reason=%s nonFinite=%u physicsErrors=%u physicsWarnings=%u "
      "fetchFailures=%u fetchErrorState=%u actorCount=%u lockWitnessCount=%u "
      "lockFlagsReadback=%u runtimeImpulseFrame=%u runtimeExcitations=%u "
      "finiteSamples=%u maxLockedAxisMotion=%.9g "
      "maxLockedAxisSpeed=%.9g minControlAxisMotion=%.9g "
      "minControlAxisSpeed=%.9g maxControlAxisMotion=%.9g "
      "maxControlAxisSpeed=%.9g lockMotionTolerance=%.9g "
      "lockSpeedTolerance=%.9g controlMotionMinimum=%.9g "
      "controlSpeedMinimum=%.9g\n",
      getHeadlessCaseName(gHeadlessCase),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, evaluation.status, evaluation.reason,
      gLockMetrics.nonFiniteSamples ? 1u : 0u, physicsErrors, physicsWarnings,
      gMetrics.fetchFailures, gMetrics.fetchErrorState, gLockMetrics.actorCount,
      eHELLO_LOCK_WITNESS_COUNT, gLockMetrics.lockFlagsReadback,
      gLockImpulseFrame, gLockMetrics.runtimeExcitations,
      gLockMetrics.finiteSamples, double(gLockMetrics.maxLockedAxisMotion),
      double(gLockMetrics.maxLockedAxisSpeed),
      double(gLockMetrics.minControlAxisMotion),
      double(gLockMetrics.minControlAxisSpeed),
      double(gLockMetrics.maxControlAxisMotion),
      double(gLockMetrics.maxControlAxisSpeed), double(gLockMotionTolerance),
      double(gLockSpeedTolerance), double(gLockControlMotionMinimum),
      double(gLockControlSpeedMinimum));
}

static void printRigidStressDetails() {
  printf("[SnippetHelloWorldRigidStress] layout=%s stacks=%u stackSize=%u boxes=%u "
         "projectileRadius=%.9g projectileDensity=%.9g "
         "projectileInitialPosition=(%.9g,%.9g,%.9g) "
         "projectileInitialVelocity=(%.9g,%.9g,%.9g) "
         "projectileFinalPosition=(%.9g,%.9g,%.9g) finiteBoxes=%u "
         "awakeBoxes=%u movedBoxes=%u minBoxCenterY=%.9g "
         "maxAbsPosition=%.9g\n",
         getRigidStressLayoutName(), gRigidStressStackCount,
         gRigidStressStackSize,
         static_cast<PxU32>(gBoxes.size()), double(gRigidStressBallRadius),
         double(gRigidStressBallDensity), double(gRigidStressBallPosition.x),
         double(gRigidStressBallPosition.y), double(gRigidStressBallPosition.z),
         double(gRigidStressBallVelocity.x),
         double(gRigidStressBallVelocity.y),
         double(gRigidStressBallVelocity.z),
         double(gRigidStressMetrics.finalBallPosition.x),
         double(gRigidStressMetrics.finalBallPosition.y),
         double(gRigidStressMetrics.finalBallPosition.z),
         gRigidStressMetrics.finiteBoxes, gRigidStressMetrics.awakeBoxes,
         gRigidStressMetrics.movedBoxes,
         double(gRigidStressMetrics.minBoxCenterY),
         double(gRigidStressMetrics.maxAbsPosition));
}

static PxU32 getRigidStressAvbdEarlyStopActive() {
  const bool avbdSolver =
      gHeadlessOptions.solverType == PxSolverType::eAVBD;
  const PxU32 earlyStopFloor =
      PxMin(gRigidStressSceneAvbdIterations, PxU32(4));
  return avbdSolver && gRigidStressSceneAvbdEnableEarlyStop &&
                 gRigidStressSceneAvbdIterations - earlyStopFloor > 1
             ? 1u
             : 0u;
}

static void printRigidStressGateResult(
    const HelloWorldGateEvaluation &evaluation, PxU32 physicsErrors,
    PxU32 physicsWarnings) {
  printf(
      "[AVBD_GATE] schema=4 snippet=SnippetHelloWorld case=rigid-stress "
      "solver=%s execution=%s requestedFrames=%u completedFrames=%u "
      "layout=%s enhancedDeterminismRequested=%u "
      "enhancedDeterminismObserved=%u avbdBackendPolicy=%s "
      "dt=%.9g seed=%u dispatcherThreads=%u "
      "avbdIterationPolicy=scene-desc avbdIterationSource=%s "
      "avbdIterations=%u avbdIterationActive=%u "
      "avbdIterationSemantics=budgeted-complete-primal-dual-stiffness "
      "avbdJointIterationOverrideSource=%s "
      "avbdJointIterationOverride=%u avbdJointIterationOverrideActive=0 "
      "avbdEarlyStopSource=%s avbdEarlyStopEnabled=%u "
      "avbdEarlyStopActive=%u "
      "capability=SUPPORTED "
      "validation=GATED status=%s reason=%s physicsErrors=%u "
      "physicsWarnings=%u fetchFailures=%u fetchErrorState=%u "
      "stacks=%u stackSize=%u boxCount=%u projectileCount=%u "
      "finiteBoxes=%u finalBallFinite=%u awakeBoxes=%u movedBoxes=%u "
      "minBoxCenterY=%.9g maxAbsPosition=%.9g finalBallX=%.9g "
      "finalBallY=%.9g finalBallZ=%.9g "
      "stateDigestAlgorithm=fnv1a64x2-v1 stateDigestActorCount=%u "
      "stateDigest=%016llx%016llx\n",
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      getRigidStressLayoutName(),
      gRigidStressEnhancedDeterminismRequested ? 1u : 0u,
      gRigidStressEnhancedDeterminismObserved ? 1u : 0u,
      gRigidStressEnhancedDeterminismObserved ? "ordered" : "fast",
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads,
      gRigidStressAvbdIterationsExplicit ? "explicit" : "default",
      gRigidStressSceneAvbdIterations,
      gHeadlessOptions.solverType == PxSolverType::eAVBD ? 1u : 0u,
      gRigidStressAvbdJointIterationOverrideExplicit ? "explicit" : "default",
      gRigidStressSceneAvbdJointIterationOverride,
      gRigidStressAvbdEarlyStopExplicit ? "explicit" : "default",
      gRigidStressSceneAvbdEnableEarlyStop ? 1u : 0u,
      getRigidStressAvbdEarlyStopActive(),
      evaluation.status,
      evaluation.reason, physicsErrors, physicsWarnings,
      gMetrics.fetchFailures, gMetrics.fetchErrorState,
      gRigidStressStackCount, gRigidStressStackSize,
      evaluation.boxCount,
      gRigidStressMetrics.finalBallFinite ? 1u : 0u,
      gRigidStressMetrics.finiteBoxes,
      gRigidStressMetrics.finalBallFinite ? 1u : 0u,
      gRigidStressMetrics.awakeBoxes, gRigidStressMetrics.movedBoxes,
      double(gRigidStressMetrics.minBoxCenterY),
      double(gRigidStressMetrics.maxAbsPosition),
      double(gRigidStressMetrics.finalBallPosition.x),
      double(gRigidStressMetrics.finalBallPosition.y),
      double(gRigidStressMetrics.finalBallPosition.z),
      gRigidStressMetrics.stateDigestActorCount,
      static_cast<unsigned long long>(gRigidStressMetrics.stateDigestHigh),
      static_cast<unsigned long long>(gRigidStressMetrics.stateDigestLow));
}

static void printRigidStressPerformanceResult(
    const HelloWorldGateEvaluation &evaluation) {
#if PX_DEBUG
  const char *buildProfile = "debug";
#elif PX_CHECKED
  const char *buildProfile = "checked";
#elif PX_PROFILE
  const char *buildProfile = "profile";
#else
  const char *buildProfile = "release";
#endif
  printf(
      "[AVBD_RIGID_PERF] schema=4 snippet=SnippetHelloWorld "
      "case=rigid-stress buildProfile=%s solver=%s sceneExecution=%s "
      "layout=%s enhancedDeterminismRequested=%u "
      "enhancedDeterminismObserved=%u avbdBackendPolicy=%s "
      "dispatcherThreads=%u cpuOnly=1 gpuDynamics=0 broadphase=cpu pvd=0 "
      "stacks=%u stackSize=%u rigidBoxes=%u projectileCount=1 "
      "projectileRadius=%.9g projectileDensity=%.9g "
      "actorPositionIterations=%u actorVelocityIterations=%u "
      "avbdIterationPolicy=scene-desc avbdIterationSource=%s "
      "avbdIterations=%u avbdIterationActive=%u "
      "avbdIterationSemantics=budgeted-complete-primal-dual-stiffness "
      "avbdJointIterationOverrideSource=%s "
      "avbdJointIterationOverride=%u avbdJointIterationOverrideActive=0 "
      "avbdEarlyStopSource=%s avbdEarlyStopEnabled=%u "
      "avbdEarlyStopActive=%u "
      "warmupFrames=%u "
      "profileFrames=%u avgStepMs=%.9g p50StepMs=%.9g p95StepMs=%.9g "
      "maxStepMs=%.9g measurement=simulate-fetch instrumentation=%s "
      "status=%s\n",
      buildProfile,
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      getRigidStressLayoutName(),
      gRigidStressEnhancedDeterminismRequested ? 1u : 0u,
      gRigidStressEnhancedDeterminismObserved ? 1u : 0u,
      gRigidStressEnhancedDeterminismObserved ? "ordered" : "fast",
      gHeadlessOptions.dispatcherThreads, gRigidStressStackCount,
      gRigidStressStackSize, gRigidStressExpectedBoxCount,
      double(gRigidStressBallRadius), double(gRigidStressBallDensity),
      gRigidStressPositionIterations, gRigidStressVelocityIterations,
      gRigidStressAvbdIterationsExplicit ? "explicit" : "default",
      gRigidStressSceneAvbdIterations,
      gHeadlessOptions.solverType == PxSolverType::eAVBD ? 1u : 0u,
      gRigidStressAvbdJointIterationOverrideExplicit ? "explicit" : "default",
      gRigidStressSceneAvbdJointIterationOverride,
      gRigidStressAvbdEarlyStopExplicit ? "explicit" : "default",
      gRigidStressSceneAvbdEnableEarlyStop ? 1u : 0u,
      getRigidStressAvbdEarlyStopActive(),
      gRigidStressWarmupFrames,
      static_cast<PxU32>(gRigidStressMetrics.stepSamplesMs.size()),
      double(gRigidStressMetrics.avgStepMs),
      double(gRigidStressMetrics.p50StepMs),
      double(gRigidStressMetrics.p95StepMs),
      double(gRigidStressMetrics.maxStepMs),
      gRigidStressWorkAttribution ? "work-attribution" : "none",
      evaluation.status);
}

static void printRigidStressWorkResult() {
#if PX_DEBUG || PX_CHECKED || PX_PROFILE
  const PxU32 profilerZoneBuild = 1u;
#else
  const PxU32 profilerZoneBuild = 0u;
#endif
  const bool avbdSolver =
      gHeadlessOptions.solverType == PxSolverType::eAVBD;
  const char *iterationSource =
      gRigidStressAvbdIterationsExplicit ? "explicit" : "default";
  const char *jointIterationOverrideSource =
      gRigidStressAvbdJointIterationOverrideExplicit ? "explicit" : "default";
  const char *earlyStopSource =
      gRigidStressAvbdEarlyStopExplicit ? "explicit" : "default";
  const PxU32 earlyStopActive = getRigidStressAvbdEarlyStopActive();
  if (!gRigidStressWorkMetrics.observed) {
    printf(
        "[AVBD_RIGID_WORK] schema=4 snippet=SnippetHelloWorld "
        "case=rigid-stress solver=%s workTelemetry=UNAVAILABLE "
        "attributionMode=none profilerZoneBuild=%u profileFrames=%u "
        "actorPositionIterations=%u actorVelocityIterations=%u "
        "avbdIterationPolicy=scene-desc avbdIterationSource=%s "
        "avbdIterations=%u avbdIterationActive=%u "
        "avbdIterationSemantics=budgeted-complete-primal-dual-stiffness "
        "avbdJointIterationOverrideSource=%s "
        "avbdJointIterationOverride=%u avbdJointIterationOverrideActive=0 "
        "avbdEarlyStopSource=%s avbdEarlyStopEnabled=%u "
        "avbdEarlyStopActive=%u "
        "avbdIslandSolves=UNAVAILABLE "
        "avbdInnerSweeps=UNAVAILABLE avbdBlockDescentZones=UNAVAILABLE "
        "avbdBodyColorPlans=UNAVAILABLE avbdBodyColorPasses=UNAVAILABLE "
        "avbdDualPasses=UNAVAILABLE avbdDualRanges=UNAVAILABLE "
        "avbdBodyRanges=UNAVAILABLE "
        "avbdLocalSolveCount=UNAVAILABLE localSolveTelemetry=UNAVAILABLE "
        "awakeDynamicBodies=UNAVAILABLE "
        "sceneStatsActiveDynamicBodies=UNAVAILABLE "
        "discreteContactPairs=UNAVAILABLE "
        "contactPairsWithContacts=UNAVAILABLE "
        "axisSolverConstraints=UNAVAILABLE solverPartitions=UNAVAILABLE\n",
        Snippets::getSolverTypeName(gHeadlessOptions.solverType),
        profilerZoneBuild,
        static_cast<PxU32>(gRigidStressMetrics.stepSamplesMs.size()),
        gRigidStressPositionIterations, gRigidStressVelocityIterations,
        iterationSource, gRigidStressSceneAvbdIterations,
        avbdSolver ? 1u : 0u, jointIterationOverrideSource,
        gRigidStressSceneAvbdJointIterationOverride, earlyStopSource,
        gRigidStressSceneAvbdEnableEarlyStop ? 1u : 0u, earlyStopActive);
    return;
  }
  printf(
      "[AVBD_RIGID_WORK] schema=4 snippet=SnippetHelloWorld "
      "case=rigid-stress solver=%s workTelemetry=OBSERVED "
      "attributionMode=profiler-zones profilerZoneBuild=%u profileFrames=%u "
      "actorPositionIterations=%u actorVelocityIterations=%u "
      "avbdIterationPolicy=scene-desc avbdIterationSource=%s "
      "avbdIterations=%u avbdIterationActive=%u "
      "avbdIterationSemantics=budgeted-complete-primal-dual-stiffness "
      "avbdJointIterationOverrideSource=%s "
      "avbdJointIterationOverride=%u avbdJointIterationOverrideActive=0 "
      "avbdEarlyStopSource=%s avbdEarlyStopEnabled=%u "
      "avbdEarlyStopActive=%u "
      "avbdIslandSolves=%llu "
      "avbdInnerSweeps=%llu avbdBlockDescentZones=%llu "
      "avbdBodyColorPlans=%llu avbdBodyColorPasses=%llu "
      "avbdDualPasses=%llu avbdDualRanges=%llu "
      "avbdBodyRanges=%llu "
      "avbdLocalSolveCount=UNAVAILABLE localSolveTelemetry=UNAVAILABLE "
      "awakeDynamicBodies=%llu peakAwakeDynamicBodies=%u "
      "sceneStatsActiveDynamicBodies=%llu "
      "peakSceneStatsActiveDynamicBodies=%u "
      "discreteContactPairs=%llu peakDiscreteContactPairs=%u "
      "contactPairsWithContacts=%llu peakContactPairsWithContacts=%u "
      "axisSolverConstraints=%llu peakAxisSolverConstraints=%u "
      "solverPartitions=%llu peakSolverPartitions=%u\n",
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      profilerZoneBuild, gRigidStressWorkMetrics.sampledFrames,
      gRigidStressPositionIterations, gRigidStressVelocityIterations,
      iterationSource, gRigidStressSceneAvbdIterations,
      avbdSolver ? 1u : 0u, jointIterationOverrideSource,
      gRigidStressSceneAvbdJointIterationOverride, earlyStopSource,
      gRigidStressSceneAvbdEnableEarlyStop ? 1u : 0u, earlyStopActive,
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdIslandSolves),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdInnerSweeps),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdBlockDescentZones),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdBodyColorPlans),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdBodyColorPasses),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdDualPasses),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdDualRanges),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.avbdBodyRanges),
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.awakeDynamicBodies),
      gRigidStressWorkMetrics.peakAwakeDynamicBodies,
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.sceneStatsActiveDynamicBodies),
      gRigidStressWorkMetrics.peakSceneStatsActiveDynamicBodies,
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.discreteContactPairs),
      gRigidStressWorkMetrics.peakDiscreteContactPairs,
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.contactPairsWithContacts),
      gRigidStressWorkMetrics.peakContactPairsWithContacts,
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.axisSolverConstraints),
      gRigidStressWorkMetrics.peakAxisSolverConstraints,
      static_cast<unsigned long long>(
          gRigidStressWorkMetrics.solverPartitions),
      gRigidStressWorkMetrics.peakSolverPartitions);
}

static void printGateDetails(const HelloWorldGateEvaluation &evaluation) {
  if (gHeadlessSleepProbe) {
    printSleepProbeDetails();
    return;
  }
  if (gHeadlessLockProbe) {
    printLockProbeDetails();
    return;
  }
  if (gHeadlessRigidStress) {
    printRigidStressDetails();
    return;
  }
  const PxReal tailAverageMaxBoxSpeed =
      gMetrics.tailSpeedSamples
          ? gMetrics.tailAvgBoxSpeed / PxReal(gMetrics.tailSpeedSamples)
          : 0.0f;
  printf(
      "[SnippetHelloWorldDiag] boxCenterY=[%.9g,%.9g] maxSettleSpeed=%.9g "
      "tailAverageMaxBoxSpeed=%.9g finalAwake=%u lateAwakeRatio=%.9g "
      "energyRatio=%.9g tailEnergy=[%.9g,%.9g,%.9g,%.9g] "
      "tailGrowth=%.9g sleepGate=DIAGNOSTIC\n",
      double(gMetrics.minBoxCenterY), double(gMetrics.maxBoxCenterY),
      double(gMetrics.maxBoxSpeedSettle), double(tailAverageMaxBoxSpeed),
      gMetrics.finalAwakeBoxes, double(evaluation.lateAwakeRatio),
      double(evaluation.maxEnergyRatio), double(evaluation.tailEnergyW1),
      double(evaluation.tailEnergyW2), double(evaluation.tailEnergyW3),
      double(evaluation.tailEnergyW4), double(evaluation.tailEnergyGrowth));
  if (gHeadlessBallShot) {
    printf(
        "[SnippetHelloWorldImpact] events=%u points=%u firstHitFrame=%u "
        "contactedBoxes=%u respondedBoxes=%u targetAxisDelta=%.9g "
        "targetMomentumRatio=%.9g targetDisplacement=%.9g "
        "ballResponseFraction=%.9g contactImpulseDiagnostic=%.9g\n",
        gMetrics.ballBoxContactEvents, gMetrics.ballBoxContactPoints,
        gMetrics.firstBallBoxContactFrame, gMetrics.contactedBoxes,
        gMetrics.respondedBoxes,
        double(gMetrics.maxTargetImpactAxisVelocityDelta),
        double(evaluation.targetResponseMomentumRatio),
        double(gMetrics.maxTargetResponseDisplacement),
        double(evaluation.ballResponseFraction),
        double(gMetrics.maxContactImpulse));
  }
}

static void printGateResult(const HelloWorldGateEvaluation &evaluation,
                            PxU32 physicsErrors, PxU32 physicsWarnings) {
  if (gHeadlessSleepProbe) {
    printSleepProbeGateResult(evaluation, physicsErrors, physicsWarnings);
    return;
  }
  if (gHeadlessLockProbe) {
    printLockProbeGateResult(evaluation, physicsErrors, physicsWarnings);
    return;
  }
  if (gHeadlessRigidStress) {
    printRigidStressGateResult(evaluation, physicsErrors, physicsWarnings);
    printRigidStressPerformanceResult(evaluation);
    printRigidStressWorkResult();
    return;
  }
  const PxReal tailAverageMaxBoxSpeed =
      gMetrics.tailSpeedSamples
          ? gMetrics.tailAvgBoxSpeed / PxReal(gMetrics.tailSpeedSamples)
          : 0.0f;
  const PxReal minBallCenterY =
      gMetrics.minBallCenterY == PX_MAX_F32 ? 0.0f
                                            : gMetrics.minBallCenterY;
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=%s solver=%s "
      "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
      "dispatcherThreads=%u "
      "capability=SUPPORTED validation=GATED status=%s reason=%s "
      "nonFinite=%u physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
      "fetchErrorState=%u launchFailures=%u boxCount=%u targetBoxCount=%u "
      "hitCount=%u hitPoints=%u firstHitFrame=%u contactedBoxes=%u "
      "responseBodies=%u responseMomentumRatio=%.9g responseDisplacement=%.9g "
      "ballResponseFraction=%.9g maxContactImpulseDiagnostic=%.9g "
      "maxEnergyRatio=%.9g tailEnergyW1=%.9g tailEnergyW2=%.9g "
      "tailEnergyW3=%.9g tailEnergyW4=%.9g tailEnergyGrowth=%.9g "
      "maxBoxSpeedSettle=%.9g tailAverageMaxBoxSpeed=%.9g "
      "finalAwake=%u lateAwakeRatio=%.9g sleepGate=DIAGNOSTIC "
      "avbdRigidSleepCheckPresent=%u groundGate=FULL_FALL_THROUGH "
      "minBoxCenterY=%.9g minBallCenterY=%.9g "
      "sunkBodies=%u maxQuaternionNormError=%.9g maxAbsPosition=%.9g "
      "maxLinearSpeed=%.9g maxAngularSpeed=%.9g energyGainCap=%.9g "
      "tailGrowthCap=%.9g settleSpeedCap=%.9g minResponseMomentumCap=%.9g "
      "minResponseDisplacementCap=%.9g minBallResponseCap=%.9g "
      "responseWindowFrames=%u\n",
      getHeadlessCaseName(gHeadlessCase),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, evaluation.status, evaluation.reason,
      gMetrics.nanDetected ? 1u : 0u, physicsErrors,
      physicsWarnings, gMetrics.fetchFailures, gMetrics.fetchErrorState,
      gMetrics.launchFailures, evaluation.boxCount, evaluation.targetBoxCount,
      gMetrics.ballBoxContactEvents, gMetrics.ballBoxContactPoints,
      gMetrics.firstBallBoxContactFrame, gMetrics.contactedBoxes,
      gMetrics.respondedBoxes,
      double(evaluation.targetResponseMomentumRatio),
      double(gMetrics.maxTargetResponseDisplacement),
      double(evaluation.ballResponseFraction),
      double(gMetrics.maxContactImpulse), double(evaluation.maxEnergyRatio),
      double(evaluation.tailEnergyW1), double(evaluation.tailEnergyW2),
      double(evaluation.tailEnergyW3), double(evaluation.tailEnergyW4),
      double(evaluation.tailEnergyGrowth),
      double(gMetrics.maxBoxSpeedSettle), double(tailAverageMaxBoxSpeed),
      gMetrics.finalAwakeBoxes, double(evaluation.lateAwakeRatio),
      1u,
      double(gMetrics.minBoxCenterY), double(minBallCenterY),
      gMetrics.maxSunkBoxes, double(gMetrics.maxQuaternionNormError),
      double(gMetrics.maxAbsPosition), double(gMetrics.maxSpeedAll),
      double(gMetrics.maxAngularSpeed), double(gMechanicalEnergyGainCap),
      double(gTailEnergyGrowthCap), double(gSettleSpeedCap),
      double(gMinTargetResponseMomentumRatio),
      double(gMinTargetResponseDisplacement),
      double(gMinBallResponseFraction), gBoxResponseWindowFrames);
}

static int reportConfigurationError(const Snippets::HeadlessOptions &options,
                                    const char *message) {
  printf("[AVBD_GATE_ERROR] snippet=SnippetHelloWorld message=%s\n", message);
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetHelloWorld case=config-error "
      "solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
      "seed=%u dispatcherThreads=%u capability=SUPPORTED validation=GATED "
      "status=ERROR reason=config nonFinite=0 physicsErrors=0 "
      "physicsWarnings=0\n",
      Snippets::getSolverTypeName(options.solverType),
      Snippets::getExecutionName(options.execution), options.frames,
      double(options.dt), options.seed, options.dispatcherThreads);
  return Snippets::eHEADLESS_CONFIG_ERROR;
}

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  Snippets::HeadlessOptions defaults;
  defaults.caseName = "stack-settle";
  defaults.frames = 600;
  defaults.seed = 1;
  defaults.dispatcherThreads = 2;
  defaults.dt = 1.0f / 60.0f;

  Snippets::HeadlessOptions options;
  std::string parseError;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
                                            parseError))
    return reportConfigurationError(options, parseError.c_str());

  bool legacyBallShotSeen = false;
  bool ballShotFrameSeen = false;
  bool warmupFramesSeen = false;
  bool workAttributionSeen = false;
  bool avbdIterationsSeen = false;
  bool avbdJointIterationOverrideSeen = false;
  bool avbdEarlyStopSeen = false;
  bool enhancedDeterminismSeen = false;
  bool rigidStressLayoutSeen = false;
  bool caseSeen = false;
  bool headlessOnlyOptionSeen = false;
  PxU32 ballShotFrame = 30;
  PxU32 rigidStressWarmupFrames = 10;
  PxU32 rigidStressAvbdIterations = 0;
  PxU32 rigidStressAvbdJointIterationOverride = 0;
  bool rigidStressAvbdEnableEarlyStop = false;
  bool rigidStressEnhancedDeterminism = false;
  RigidStressLayout rigidStressLayout =
      eRIGID_STRESS_INDEPENDENT_ISLANDS;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!arg)
      continue;
    if (strcmp(arg, "--work-attribution") == 0) {
      if (workAttributionSeen)
        return reportConfigurationError(options,
                                        "duplicate_--work-attribution");
      workAttributionSeen = true;
      headlessOnlyOptionSeen = true;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--warmup-frames=")) {
      if (warmupFramesSeen)
        return reportConfigurationError(options,
                                        "duplicate_--warmup-frames");
      warmupFramesSeen = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseU32(arg + strlen("--warmup-frames="), 0,
                              100000000u, rigidStressWarmupFrames))
        return reportConfigurationError(options,
                                        "invalid_--warmup-frames_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--avbd-iterations=")) {
      if (avbdIterationsSeen)
        return reportConfigurationError(options,
                                        "duplicate_--avbd-iterations");
      avbdIterationsSeen = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseU32(arg + strlen("--avbd-iterations="), 1u, 255u,
                              rigidStressAvbdIterations))
        return reportConfigurationError(options,
                                        "invalid_--avbd-iterations_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(
            arg, "--avbd-joint-iteration-override=")) {
      if (avbdJointIterationOverrideSeen)
        return reportConfigurationError(
            options, "duplicate_--avbd-joint-iteration-override");
      avbdJointIterationOverrideSeen = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseU32(
              arg + strlen("--avbd-joint-iteration-override="), 0u, 255u,
              rigidStressAvbdJointIterationOverride))
        return reportConfigurationError(
            options, "invalid_--avbd-joint-iteration-override_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--avbd-early-stop=")) {
      if (avbdEarlyStopSeen)
        return reportConfigurationError(options,
                                        "duplicate_--avbd-early-stop");
      avbdEarlyStopSeen = true;
      headlessOnlyOptionSeen = true;
      const char *value = arg + strlen("--avbd-early-stop=");
      if (Snippets::equalsIgnoreCase(value, "on"))
        rigidStressAvbdEnableEarlyStop = true;
      else if (Snippets::equalsIgnoreCase(value, "off"))
        rigidStressAvbdEnableEarlyStop = false;
      else
        return reportConfigurationError(options,
                                        "invalid_--avbd-early-stop_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--enhanced-determinism=")) {
      if (enhancedDeterminismSeen)
        return reportConfigurationError(
            options, "duplicate_--enhanced-determinism");
      enhancedDeterminismSeen = true;
      headlessOnlyOptionSeen = true;
      const char *value = arg + strlen("--enhanced-determinism=");
      if (Snippets::equalsIgnoreCase(value, "on"))
        rigidStressEnhancedDeterminism = true;
      else if (Snippets::equalsIgnoreCase(value, "off"))
        rigidStressEnhancedDeterminism = false;
      else
        return reportConfigurationError(
            options, "invalid_--enhanced-determinism_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--rigid-stress-layout=")) {
      if (rigidStressLayoutSeen)
        return reportConfigurationError(
            options, "duplicate_--rigid-stress-layout");
      rigidStressLayoutSeen = true;
      headlessOnlyOptionSeen = true;
      const char *value = arg + strlen("--rigid-stress-layout=");
      if (Snippets::equalsIgnoreCase(value, "independent"))
        rigidStressLayout = eRIGID_STRESS_INDEPENDENT_ISLANDS;
      else if (Snippets::equalsIgnoreCase(value, "connected"))
        rigidStressLayout = eRIGID_STRESS_CONNECTED_ISLAND;
      else
        return reportConfigurationError(
            options, "invalid_--rigid-stress-layout_value");
      continue;
    }
    if (Snippets::isCommonHeadlessOption(arg)) {
      if (Snippets::hasOptionPrefix(arg, "--case=") ||
          Snippets::hasOptionPrefix(arg, "--scenario="))
        caseSeen = true;
      if (strcmp(arg, "--headless") != 0 &&
          !Snippets::hasOptionPrefix(arg, "--solver="))
        headlessOnlyOptionSeen = true;
      continue;
    }
    if (strcmp(arg, "--headless-ball-shot") == 0) {
      if (legacyBallShotSeen)
        return reportConfigurationError(options,
                                        "duplicate_--headless-ball-shot");
      legacyBallShotSeen = true;
      headlessOnlyOptionSeen = true;
      options.headless = true;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--ball-shot-frame=")) {
      if (ballShotFrameSeen)
        return reportConfigurationError(options,
                                        "duplicate_--ball-shot-frame");
      ballShotFrameSeen = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseU32(arg + strlen("--ball-shot-frame="), 0,
                              100000000u, ballShotFrame))
        return reportConfigurationError(options,
                                        "invalid_--ball-shot-frame_value");
      continue;
    }
    return reportConfigurationError(options, "unknown_argument");
  }

#ifndef RENDER_SNIPPET
  options.headless = true;
#endif

  if (legacyBallShotSeen) {
    if (caseSeen &&
        !Snippets::equalsIgnoreCase(options.caseName.c_str(), "ball-shot"))
      return reportConfigurationError(options,
                                      "legacy_ball_shot_case_conflict");
    options.caseName = "ball-shot";
  }

  HelloWorldHeadlessCase headlessCase = eHELLO_CASE_STACK_SETTLE;
  if (!tryParseHeadlessCase(options.caseName.c_str(), headlessCase))
    return reportConfigurationError(options, "invalid_--case_value");
  options.caseName = getHeadlessCaseName(headlessCase);
  const bool sleepProbeCase = isSleepProbeCase(headlessCase);
  const bool lockProbeCase = isLockProbeCase(headlessCase);
  const bool rigidStressCase = headlessCase == eHELLO_CASE_RIGID_STRESS;
  if (sleepProbeCase && !options.framesExplicit)
    options.frames = headlessCase == eHELLO_CASE_SLEEP_WAKE ? 360u : 180u;
  if (lockProbeCase && !options.framesExplicit)
    options.frames = 120u;
  if (rigidStressCase && !options.framesExplicit)
    options.frames = 120u;

  if (ballShotFrameSeen && headlessCase != eHELLO_CASE_BALL_SHOT)
    return reportConfigurationError(options,
                                    "--ball-shot-frame_requires_ball-shot");
  if (warmupFramesSeen && !rigidStressCase)
    return reportConfigurationError(options,
                                    "--warmup-frames_requires_rigid-stress");
  if (workAttributionSeen && !rigidStressCase)
    return reportConfigurationError(
        options, "--work-attribution_requires_rigid-stress");
  if (avbdIterationsSeen && !rigidStressCase)
    return reportConfigurationError(
        options, "--avbd-iterations_requires_rigid-stress");
  if (avbdJointIterationOverrideSeen && !rigidStressCase)
    return reportConfigurationError(
        options,
        "--avbd-joint-iteration-override_requires_rigid-stress");
  if (avbdEarlyStopSeen && !rigidStressCase)
    return reportConfigurationError(
        options, "--avbd-early-stop_requires_rigid-stress");
  if (enhancedDeterminismSeen && !rigidStressCase)
    return reportConfigurationError(
        options, "--enhanced-determinism_requires_rigid-stress");
  if (rigidStressLayoutSeen && !rigidStressCase)
    return reportConfigurationError(
        options, "--rigid-stress-layout_requires_rigid-stress");
#if !(PX_DEBUG || PX_CHECKED || PX_PROFILE)
  if (workAttributionSeen)
    return reportConfigurationError(
        options, "--work-attribution_requires_instrumented_build");
#endif
  if (!options.headless && headlessOnlyOptionSeen)
    return reportConfigurationError(options,
                                    "gate_option_requires_--headless");
  if ((!sleepProbeCase && !lockProbeCase && !rigidStressCase &&
       options.frames < 360) ||
      (lockProbeCase && options.frames < 120) ||
      (sleepProbeCase && headlessCase != eHELLO_CASE_SLEEP_WAKE &&
       options.frames < 180) ||
      (headlessCase == eHELLO_CASE_SLEEP_WAKE && options.frames < 300))
    return reportConfigurationError(options,
                                    headlessCase == eHELLO_CASE_SLEEP_WAKE
                                        ? "sleep-wake_frames_must_be_at_least_300"
                                    : lockProbeCase
                                        ? "lock_frames_must_be_at_least_120"
                                    : sleepProbeCase
                                        ? "sleep_frames_must_be_at_least_180"
                                        : "frames_must_be_at_least_360");
  if (rigidStressCase &&
      (options.frames < 2u || rigidStressWarmupFrames >= options.frames))
    return reportConfigurationError(
        options, "rigid-stress_requires_profile_frame_after_warmup");
  if (headlessCase == eHELLO_CASE_BALL_SHOT &&
      options.frames < ballShotFrame + 120u)
    return reportConfigurationError(options,
                                    "ball-shot_response_window_incomplete");
  if (options.execution == Snippets::eHEADLESS_SEQUENTIAL &&
      options.solverType != PxSolverType::eAVBD)
    return reportConfigurationError(options, "sequential_requires_avbd");
  if (PxAbs(options.dt - (1.0f / 60.0f)) > 1e-7f)
    return reportConfigurationError(options, "dt_requires_60hz_calibration");
  if (!Snippets::applyExecutionEnvironment(options))
    return reportConfigurationError(options, "execution_environment_failed");

  gHeadlessOptions = options;
  gHeadlessCase = headlessCase;
  gSolverType = options.solverType;
  gHeadlessMode = options.headless;
  gHeadlessBallShot = headlessCase == eHELLO_CASE_BALL_SHOT;
  gHeadlessSleepProbe = sleepProbeCase;
  gHeadlessLockProbe = lockProbeCase;
  gHeadlessRigidStress = rigidStressCase;
  gRigidStressWorkAttribution = workAttributionSeen;
  gRigidStressAvbdIterationsExplicit = avbdIterationsSeen;
  gRigidStressAvbdJointIterationOverrideExplicit =
      avbdJointIterationOverrideSeen;
  gRigidStressAvbdEarlyStopExplicit = avbdEarlyStopSeen;
  gRigidStressRequestedAvbdIterations = rigidStressAvbdIterations;
  gRigidStressRequestedAvbdJointIterationOverride =
      rigidStressAvbdJointIterationOverride;
  gRigidStressRequestedAvbdEnableEarlyStop =
      rigidStressAvbdEnableEarlyStop;
  gRigidStressEnhancedDeterminismRequested =
      rigidStressEnhancedDeterminism;
  gRigidStressLayout = rigidStressLayout;
  gHeadlessFrameCount = options.frames;
  gBallShotFrame = ballShotFrame;
  gRigidStressWarmupFrames = rigidStressWarmupFrames;

#ifdef RENDER_SNIPPET
  if (!options.headless) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  Snippets::printHeadlessConfig("SnippetHelloWorld", gHeadlessOptions);
  if (gHeadlessRigidStress) {
    printf("[SnippetHelloWorldConfig] workload=rigid-stress source=HelloGRB "
           "layout=%s enhancedDeterminismRequested=%u "
           "stacks=%u stackSize=%u boxes=%u projectileRadius=%.9g "
           "projectileDensity=%.9g warmupFrames=%u profileFrames=%u "
           "gpuDynamics=0 broadphase=cpu pvd=0 validation=GATED\n",
           getRigidStressLayoutName(),
           gRigidStressEnhancedDeterminismRequested ? 1u : 0u,
           gRigidStressStackCount, gRigidStressStackSize,
           gRigidStressExpectedBoxCount, double(gRigidStressBallRadius),
           double(gRigidStressBallDensity), gRigidStressWarmupFrames,
           gHeadlessOptions.frames - gRigidStressWarmupFrames);
  } else if (gHeadlessBallShot) {
    printf("[SnippetHelloWorldConfig] shotFrame=%u targetStackZ=0 "
           "direction=negative-z responseWindowFrames=%u\n",
           gBallShotFrame, gBoxResponseWindowFrames);
  } else if (gHeadlessSleepProbe) {
    printf("[SnippetHelloWorldConfig] sleepProbe=1 witnessCount=%u "
           "gravity=zero wakeFrame=%u wakeDeltaVelocity=%.9g "
           "validation=GATED\n",
           eHELLO_SLEEP_WITNESS_COUNT, gSleepWakeFrame,
           double(gSleepWakeTargetDeltaVelocity));
  } else if (gHeadlessLockProbe) {
    printf("[SnippetHelloWorldConfig] lockProbe=1 witnessCount=%u "
           "actorCount=%u linearSpeed=%.9g angularSpeed=%.9g "
           "runtimeImpulseFrame=%u "
           "validation=GATED\n",
           eHELLO_LOCK_WITNESS_COUNT, eHELLO_LOCK_WITNESS_COUNT * 2u,
           double(gLockLinearSpeed), double(gLockAngularSpeed),
           gLockImpulseFrame);
  }

  initPhysics(false);
  if (gInitializationFailed) {
    HelloWorldGateEvaluation evaluation;
    evaluation.boxCount = static_cast<PxU32>(gBoxes.size());
    for (PxU32 i = 0; i < gBoxIsTarget.size(); ++i)
      evaluation.targetBoxCount += gBoxIsTarget[i] ? 1u : 0u;
    setGateError(evaluation, "initialization");
    cleanupPhysics(false);
    printGateResult(evaluation, gErrorCallback.getFatalCount(),
                    gErrorCallback.getWarningCount());
    return static_cast<int>(evaluation.exitCode);
  }

  for (PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame) {
    PX_UNUSED(frame);
    stepPhysics(false);
    if (gMetrics.fetchFailures)
      break;
  }
  if (gHeadlessRigidStress)
    sampleRigidStressFinalState();
  HelloWorldGateEvaluation evaluation = evaluateGate();
  printGateDetails(evaluation);
  cleanupPhysics(false);

  const PxU32 physicsErrors = gErrorCallback.getFatalCount();
  if (physicsErrors && evaluation.exitCode == Snippets::eHEADLESS_PASS)
    setGateFailure(evaluation, "physx_error");
  printGateResult(evaluation, physicsErrors, gErrorCallback.getWarningCount());
  return static_cast<int>(evaluation.exitCode);
}
