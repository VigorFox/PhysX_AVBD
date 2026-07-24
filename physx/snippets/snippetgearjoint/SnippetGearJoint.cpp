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
// This snippet illustrates simple use of gear joints
// ****************************************************************************

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "PxPhysicsAPI.h"
#include "extensions/PxCollectionExt.h"
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace physx;

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation *gFoundation = NULL;
static PxPhysics *gPhysics = NULL;
static PxDefaultCpuDispatcher *gDispatcher = NULL;
static PxScene *gScene = NULL;
static PxMaterial *gMaterial = NULL;
static PxPvd *gPvd = NULL;
static PxPvdTransport *gPvdTransport = NULL;
static PxRigidDynamic *gGearActor0 = NULL;
static PxRigidDynamic *gGearActor1 = NULL;
static PxRevoluteJoint *gHinge0 = NULL;
static PxRevoluteJoint *gHinge1 = NULL;
static PxGearJoint *gGearJoint = NULL;
static PxSerializationRegistry *gSerializationRegistry = NULL;
static PxCollection *gLoadedCollection = NULL;
static PxU8 *gBinaryBlockRaw = NULL;
static void *gBinaryBlockAligned = NULL;
static PxU64 gBinaryBlockSize = 0;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;
static bool gOracleInitialized = false;
static const char *gSerializationFailureReason = "none";

enum GearHeadlessCase {
  eCASE_STEADY,
  eCASE_UNIT_RATIO,
  eCASE_PHASE_OFFSET,
  eCASE_REVERSE,
  eCASE_SINUSOIDAL,
  eCASE_EXTERNAL_IMPULSE
};

enum GearSerializationMode {
  eSERIALIZATION_RUNTIME,
  eSERIALIZATION_BINARY
};

struct GearHeadlessConfig {
  PxReal ratio;
  PxU32 reverseFrame;
  PxU32 impulseFrame;
  GearSerializationMode serializationMode;

  GearHeadlessConfig()
      : ratio(2.5f), reverseFrame(300), impulseFrame(300),
        serializationMode(eSERIALIZATION_RUNTIME) {}
};

struct GearOptionSeen {
  bool ratio;
  bool reverseFrame;
  bool impulseFrame;
  bool serialization;

  GearOptionSeen()
      : ratio(false), reverseFrame(false), impulseFrame(false),
        serialization(false) {}
};

struct GearGateMetrics {
  PxU32 completedFrames;
  PxU32 simulateCalls;
  PxU32 fetchCalls;
  PxU32 fetchFailures;
  PxU32 fetchErrorState;
  PxU32 nonFinite;
  PxU32 unwrapAliasRisk;
  PxU32 sampleCount;
  PxU32 tailSamples;
  PxU32 ratioSamples;
  PxU32 measuredRatioSamples;
  PxU32 directionSamples;
  PxU32 directionViolations;
  PxU32 preReverseAlignedSamples;
  PxU32 postReverseAlignedSamples;
  PxU32 positiveMotionSamples;
  PxU32 negativeMotionSamples;
  PxU32 targetSignChanges;
  PxU32 reverseEvents;
  PxU32 impulseEvents;
  PxU32 driveTrackingSamples;
  PxU32 driveAlignedSamples;
  PxU32 driveOpposedSamples;
  PxU32 impulseResponseSamples;
  PxU32 impulseRecoverySamples;
  PxU32 driveEnabledReadback;
  PxU32 shapeRefsCreated;
  PxU32 shapeRefsReleased;
  PxU32 actorsCreated;
  PxU32 actorsReleased;
  PxU32 jointsCreated;
  PxU32 jointsReleased;
  PxU32 topologyDynamicActors;
  PxU32 topologyStaticActors;
  PxU32 topologyConstraints;
  PxU32 topologyShapes0;
  PxU32 topologyShapes1;
  PxU32 topologyOk;
  PxU32 serializationRequested;
  PxU32 registryCreated;
  PxU32 collectionCompleted;
  PxU32 serializable;
  PxU32 serializeSuccess;
  PxU64 serializedBytes;
  PxU32 binaryBlockAllocated;
  PxU32 binaryAligned;
  PxU32 deserializeSuccess;
  PxU32 loadedObjects;
  PxU32 loadedActors;
  PxU32 loadedConstraints;
  PxU32 loadedRevolute;
  PxU32 loadedGear;
  PxU32 dependencyIdentity;
  PxU32 actorIdentity;
  PxU32 authoringReleased;
  PxU32 loadedCollectionReleased;
  PxU32 binaryBlockFreed;
  PxU32 cleanupComplete;
  PxReal ratioReadback;
  PxReal endpointSign0;
  PxReal endpointSign1;
  PxReal previousRawAngle0;
  PxReal previousRawAngle1;
  PxReal unwrappedAngle0;
  PxReal unwrappedAngle1;
  PxReal initialAngle0;
  PxReal initialAngle1;
  PxReal baselinePhase;
  PxReal totalAbsTravel0;
  PxReal totalAbsTravel1;
  PxReal signedTravel0;
  PxReal signedTravel1;
  PxReal maxVelocityResidualNorm;
  PxReal velocityResidualSquaredSum;
  PxReal measuredRatioSum;
  PxReal maxPhaseError;
  PxReal maxTailPhaseError;
  PxReal tailPhaseSquaredSum;
  PxReal maxLockedAxisSpeed;
  PxReal maxAnchorError;
  PxReal minAxisDot;
  PxReal maxCenterDrift;
  PxReal maxQuaternionNormError;
  PxReal maxAbsPosition;
  PxReal maxLinearSpeed;
  PxReal maxAngularSpeed;
  PxReal impulseDelta0;
  PxReal impulseDelta1;
  PxReal impulseResponseMagnitude;
  PxReal impulseResponseResidualNorm;
  PxReal impulseProjectionErrorNorm;
  PxReal impulseExpectedDelta0;
  PxReal impulseExpectedDelta1;
  PxReal impulseActualWorldDelta0;
  PxReal impulseActualWorldDelta1;
  PxReal impulseProjectionDenominator;
  PxReal impulseRecoveryResidualMax;
  PxReal impulsePreSpeedMax;
  PxReal driveTrackingErrorSquaredSum;
  PxReal driveTrackingErrorMax;
  PxReal initialPhaseOffsetWitness;
  double phaseTimeSum;
  double phaseValueSum;
  double phaseTimeSquaredSum;
  double phaseTimeValueSum;

  GearGateMetrics()
      : completedFrames(0), simulateCalls(0), fetchCalls(0),
        fetchFailures(0), fetchErrorState(0), nonFinite(0),
        unwrapAliasRisk(0), sampleCount(0), tailSamples(0), ratioSamples(0),
        measuredRatioSamples(0), directionSamples(0), directionViolations(0),
        preReverseAlignedSamples(0), postReverseAlignedSamples(0),
        positiveMotionSamples(0), negativeMotionSamples(0),
        targetSignChanges(0), reverseEvents(0), impulseEvents(0),
        driveTrackingSamples(0), driveAlignedSamples(0),
        driveOpposedSamples(0), impulseResponseSamples(0),
        impulseRecoverySamples(0), driveEnabledReadback(0),
        shapeRefsCreated(0), shapeRefsReleased(0), actorsCreated(0),
        actorsReleased(0), jointsCreated(0), jointsReleased(0),
        topologyDynamicActors(0), topologyStaticActors(0),
        topologyConstraints(0), topologyShapes0(0), topologyShapes1(0),
        topologyOk(0), serializationRequested(0), registryCreated(0),
        collectionCompleted(0), serializable(0), serializeSuccess(0),
        serializedBytes(0), binaryBlockAllocated(0), binaryAligned(0),
        deserializeSuccess(0), loadedObjects(0), loadedActors(0),
        loadedConstraints(0), loadedRevolute(0), loadedGear(0),
        dependencyIdentity(0), actorIdentity(0), authoringReleased(0),
        loadedCollectionReleased(0), binaryBlockFreed(0), cleanupComplete(0),
        ratioReadback(0.0f),
        endpointSign0(0.0f), endpointSign1(0.0f), previousRawAngle0(0.0f),
        previousRawAngle1(0.0f), unwrappedAngle0(0.0f),
        unwrappedAngle1(0.0f), initialAngle0(0.0f), initialAngle1(0.0f),
        baselinePhase(0.0f), totalAbsTravel0(0.0f), totalAbsTravel1(0.0f),
        signedTravel0(0.0f), signedTravel1(0.0f),
        maxVelocityResidualNorm(0.0f), velocityResidualSquaredSum(0.0f),
        measuredRatioSum(0.0f), maxPhaseError(0.0f),
        maxTailPhaseError(0.0f), tailPhaseSquaredSum(0.0f),
        maxLockedAxisSpeed(0.0f), maxAnchorError(0.0f), minAxisDot(1.0f),
        maxCenterDrift(0.0f), maxQuaternionNormError(0.0f),
        maxAbsPosition(0.0f), maxLinearSpeed(0.0f),
        maxAngularSpeed(0.0f), impulseDelta0(0.0f), impulseDelta1(0.0f),
        impulseResponseMagnitude(0.0f), impulseResponseResidualNorm(0.0f),
        impulseProjectionErrorNorm(0.0f), impulseExpectedDelta0(0.0f),
        impulseExpectedDelta1(0.0f), impulseActualWorldDelta0(0.0f),
        impulseActualWorldDelta1(0.0f),
        impulseProjectionDenominator(0.0f),
        impulseRecoveryResidualMax(0.0f), impulsePreSpeedMax(0.0f),
        driveTrackingErrorSquaredSum(0.0f), driveTrackingErrorMax(0.0f),
        initialPhaseOffsetWitness(0.0f), phaseTimeSum(0.0),
        phaseValueSum(0.0), phaseTimeSquaredSum(0.0),
        phaseTimeValueSum(0.0) {}
};

struct GearGateEvaluation {
  Snippets::HeadlessExitCode exitCode;
  const char *status;
  const char *reason;
  PxReal velocityResidualRms;
  PxReal measuredRatioMean;
  PxReal tailPhaseRms;
  PxReal phaseDriftSlope;
  PxReal driveTrackingErrorRms;

  GearGateEvaluation()
      : exitCode(Snippets::eHEADLESS_PASS), status("PASS"), reason("none"),
        velocityResidualRms(0.0f), measuredRatioMean(0.0f),
        tailPhaseRms(0.0f), phaseDriftSlope(0.0f),
        driveTrackingErrorRms(0.0f) {}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static GearHeadlessConfig gHeadlessConfig;
static GearHeadlessCase gHeadlessCase = eCASE_STEADY;
static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;
static GearGateMetrics gMetrics;
static PxVec3 gInitialPosition0(0.0f);
static PxVec3 gInitialPosition1(0.0f);
static PxReal gCurrentDriveTarget = 0.5f;
static PxReal gImpulsePreVelocity0 = 0.0f;
static PxReal gImpulsePreVelocity1 = 0.0f;
static PxVec3 gImpulsePreWorldAngular0(0.0f);
static PxVec3 gImpulsePreWorldAngular1(0.0f);
static PxVec3 gImpulseExpectedWorldAngular0(0.0f);
static PxVec3 gImpulseExpectedWorldAngular1(0.0f);
static PxVec3 gImpulseGearAxis0(0.0f);
static PxVec3 gImpulseGearAxis1(0.0f);
static int gLastSinusoidalTargetSign = 0;

static const PxReal gDriveVelocity = 0.5f;
static const PxReal gSinusoidalDriveAmplitude = 3.0f;
static const PxReal gPhaseOffsetAngle = 0.4f;
static const PxReal gAngularImpulse = 50.0f;
static const PxReal gVelocityFloor = 0.05f;
static const PxReal gSinusoidalTargetAnalysisFloor = 0.25f;
static const PxReal gVelocityResidualCap = 0.1f;
static const PxReal gPhaseErrorCap = 0.2f;
static const PxReal gPhaseDriftSlopeCap = 0.02f;
static const PxReal gMinimumTravel = 0.25f;
static const PxReal gAnchorErrorCap = 0.25f;
static const PxReal gMinimumAxisDot = 0.98f;
static const PxReal gLockedAxisSpeedCap = 0.5f;
static const PxReal gPositionRunawayCap = 1000.0f;
static const PxReal gLinearSpeedRunawayCap = 200.0f;
static const PxReal gAngularSpeedRunawayCap = 200.0f;
static const PxReal gQuaternionNormErrorCap = 1e-3f;
static const PxReal gDriveTrackingErrorCap = 0.25f;
static const PxReal gImpulseBaselineSpeedCap = 0.01f;
static const PxReal gImpulseMinimumResponse = 0.01f;
static const PxReal gImpulseResidualCap = 0.1f;
static const PxReal gImpulseProjectionErrorCap = 0.05f;
static const PxReal gImpulseRecoveryVelocityFloor = 0.001f;
static const PxU32 gWarmupFrames = 120;
static const PxU32 gEventRecoveryFrames = 120;
static const PxU32 gImpulseResponseWindow = 4;
static const PxU32 gImpulseRecoveryWindow = 120;
static const PxSerialObjectId gGearActor0Id = PxSerialObjectId(0x1101);
static const PxSerialObjectId gGearActor1Id = PxSerialObjectId(0x1102);
static const PxSerialObjectId gHinge0Id = PxSerialObjectId(0x2101);
static const PxSerialObjectId gHinge1Id = PxSerialObjectId(0x2102);
static const PxSerialObjectId gGearJointId = PxSerialObjectId(0x2103);

static const char *getCaseName(GearHeadlessCase testCase) {
  switch (testCase) {
  case eCASE_STEADY:
    return "steady";
  case eCASE_UNIT_RATIO:
    return "unit-ratio";
  case eCASE_PHASE_OFFSET:
    return "phase-offset";
  case eCASE_REVERSE:
    return "reverse";
  case eCASE_SINUSOIDAL:
    return "sinusoidal";
  case eCASE_EXTERNAL_IMPULSE:
    return "external-impulse";
  }
  return "unknown";
}

static bool tryParseCase(const char *value, GearHeadlessCase &testCase) {
  if (Snippets::equalsIgnoreCase(value, "steady"))
    testCase = eCASE_STEADY;
  else if (Snippets::equalsIgnoreCase(value, "unit-ratio"))
    testCase = eCASE_UNIT_RATIO;
  else if (Snippets::equalsIgnoreCase(value, "phase-offset"))
    testCase = eCASE_PHASE_OFFSET;
  else if (Snippets::equalsIgnoreCase(value, "reverse"))
    testCase = eCASE_REVERSE;
  else if (Snippets::equalsIgnoreCase(value, "sinusoidal"))
    testCase = eCASE_SINUSOIDAL;
  else if (Snippets::equalsIgnoreCase(value, "external-impulse"))
    testCase = eCASE_EXTERNAL_IMPULSE;
  else
    return false;
  return true;
}

static const char *getSerializationModeName(GearSerializationMode mode) {
  return mode == eSERIALIZATION_BINARY ? "binary" : "runtime";
}

static bool tryParseSerializationMode(const char *value,
                                      GearSerializationMode &mode) {
  if (Snippets::equalsIgnoreCase(value, "runtime"))
    mode = eSERIALIZATION_RUNTIME;
  else if (Snippets::equalsIgnoreCase(value, "binary"))
    mode = eSERIALIZATION_BINARY;
  else
    return false;
  return true;
}

static std::string makeAuthorityToken(const char *value) {
  std::string token = value && value[0] ? value : "unknown";
  for (size_t i = 0; i < token.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(token[i]);
    if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.'))
      token[i] = '_';
  }
  return token;
}

static PxReal shortestAngleDelta(PxReal current, PxReal previous) {
  PxReal delta = PxReal(std::fmod(double(current - previous + PxPi),
                                 double(PxTwoPi)));
  if (delta < 0.0f)
    delta += PxTwoPi;
  return delta - PxPi;
}

static PxReal maxAbsElement(const PxVec3 &value) {
  return PxMax(PxAbs(value.x), PxMax(PxAbs(value.y), PxAbs(value.z)));
}

static PxVec3 applyInverseInertiaWorld(const PxRigidDynamic &actor,
                                       const PxVec3 &worldVector) {
  const PxVec3 inertia = actor.getMassSpaceInertiaTensor();
  const PxQuat worldRotation = actor.getGlobalPose().q;
  const PxVec3 localVector = worldRotation.rotateInv(worldVector);
  const PxVec3 localResponse(
      inertia.x > 1e-12f ? localVector.x / inertia.x : 0.0f,
      inertia.y > 1e-12f ? localVector.y / inertia.y : 0.0f,
      inertia.z > 1e-12f ? localVector.z / inertia.z : 0.0f);
  return worldRotation.rotate(localResponse);
}

static PxTransform getJointFrameWorld(const PxJoint &joint,
                                      PxJointActorIndex::Enum index) {
  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  joint.getActors(actor0, actor1);
  PxRigidActor *actor =
      index == PxJointActorIndex::eACTOR0 ? actor0 : actor1;
  const PxTransform localPose = joint.getLocalPose(index);
  return actor ? actor->getGlobalPose() * localPose : localPose;
}

static PxReal getEndpointSign(const PxJoint &hinge,
                              const PxRigidActor *gearActor) {
  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  hinge.getActors(actor0, actor1);
  if (gearActor == actor0)
    return -1.0f;
  if (gearActor == actor1)
    return 1.0f;
  return 0.0f;
}

static void resetRuntimeState() {
  gMetrics = GearGateMetrics();
  gInitializationFailed = false;
  gOracleInitialized = false;
  gSerializationFailureReason = "none";
  gCurrentDriveTarget = gDriveVelocity;
  gImpulsePreVelocity0 = 0.0f;
  gImpulsePreVelocity1 = 0.0f;
  gImpulsePreWorldAngular0 = PxVec3(0.0f);
  gImpulsePreWorldAngular1 = PxVec3(0.0f);
  gImpulseExpectedWorldAngular0 = PxVec3(0.0f);
  gImpulseExpectedWorldAngular1 = PxVec3(0.0f);
  gImpulseGearAxis0 = PxVec3(0.0f);
  gImpulseGearAxis1 = PxVec3(0.0f);
  gLastSinusoidalTargetSign = 0;
  gErrorCallback.reset();
}

static PxRigidDynamic *createGearWithBoxes(PxPhysics &sdk,
                                           const PxBoxGeometry &boxGeometry,
                                           const PxTransform &transform,
                                           PxMaterial &material,
                                           PxU32 shapeCount) {
  PxRigidDynamic *actor = sdk.createRigidDynamic(transform);
  if (!actor)
    return NULL;
  gMetrics.actorsCreated++;

  PxMat33 rotation(PxIdentity);
  for (PxU32 i = 0; i < shapeCount; ++i) {
    const PxReal coefficient = PxReal(i) / PxReal(shapeCount);
    const PxReal angle = PxPi * 0.5f * coefficient;
    PxShape *shape = sdk.createShape(boxGeometry, material, true);
    if (!shape) {
      actor->release();
      gMetrics.actorsReleased++;
      return NULL;
    }
    gMetrics.shapeRefsCreated++;

    const PxReal cosine = PxCos(angle);
    const PxReal sine = PxSin(angle);
    rotation[0][0] = rotation[1][1] = cosine;
    rotation[0][1] = sine;
    rotation[1][0] = -sine;

    shape->setLocalPose(PxTransform(PxVec3(0.0f), PxQuat(rotation)));
    const bool attached = actor->attachShape(*shape);
    shape->release();
    gMetrics.shapeRefsReleased++;
    if (!attached) {
      actor->release();
      gMetrics.actorsReleased++;
      return NULL;
    }
  }

  if (!PxRigidBodyExt::updateMassAndInertia(*actor, 1.0f)) {
    actor->release();
    gMetrics.actorsReleased++;
    return NULL;
  }
  return actor;
}

static void markJointCreated(const PxJoint *joint) {
  if (joint)
    gMetrics.jointsCreated++;
}

static void setSerializationFailure(const char *reason) {
  if (std::strcmp(gSerializationFailureReason, "none") == 0)
    gSerializationFailureReason = reason;
}

static void releaseAuthoringFixture() {
  if (gGearJoint) {
    gGearJoint->release();
    gGearJoint = NULL;
    gMetrics.jointsReleased++;
  }
  if (gHinge0) {
    gHinge0->release();
    gHinge0 = NULL;
    gMetrics.jointsReleased++;
  }
  if (gHinge1) {
    gHinge1->release();
    gHinge1 = NULL;
    gMetrics.jointsReleased++;
  }
  if (gGearActor0) {
    gGearActor0->release();
    gGearActor0 = NULL;
    gMetrics.actorsReleased++;
  }
  if (gGearActor1) {
    gGearActor1->release();
    gGearActor1 = NULL;
    gMetrics.actorsReleased++;
  }
  gMetrics.authoringReleased =
      !gGearJoint && !gHinge0 && !gHinge1 && !gGearActor0 && !gGearActor1
          ? 1u
          : 0u;
}

template <typename T>
static T *findSerializedObject(PxSerialObjectId id) {
  if (!gLoadedCollection)
    return NULL;
  PxBase *object = gLoadedCollection->find(id);
  return object ? object->is<T>() : NULL;
}

static void validateSerializedIdentity() {
  if (!gGearActor0 || !gGearActor1 || !gHinge0 || !gHinge1 || !gGearJoint)
    return;

  const PxBase *hinge0 = NULL;
  const PxBase *hinge1 = NULL;
  gGearJoint->getHinges(hinge0, hinge1);
  gMetrics.dependencyIdentity =
      hinge0 == gHinge0 && hinge1 == gHinge1 ? 1u : 0u;

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gHinge0->getActors(actor0, actor1);
  const bool hinge0Identity = actor0 == NULL && actor1 == gGearActor0;
  gHinge1->getActors(actor0, actor1);
  const bool hinge1Identity = actor0 == NULL && actor1 == gGearActor1;
  gGearJoint->getActors(actor0, actor1);
  const bool gearIdentity =
      actor0 == gGearActor0 && actor1 == gGearActor1;
  gMetrics.actorIdentity =
      hinge0Identity && hinge1Identity && gearIdentity ? 1u : 0u;
}

static bool replaceFixtureWithBinaryRoundTrip() {
  gMetrics.serializationRequested = 1;
  gSerializationRegistry =
      PxSerialization::createSerializationRegistry(*gPhysics);
  if (!gSerializationRegistry) {
    setSerializationFailure("serialization_registry");
    return false;
  }
  gMetrics.registryCreated = 1;

  PxCollection *authoringCollection = PxCreateCollection();
  if (!authoringCollection) {
    setSerializationFailure("serialization_collection");
    return false;
  }

  authoringCollection->add(*gGearActor0, gGearActor0Id);
  authoringCollection->add(*gGearActor1, gGearActor1Id);
  authoringCollection->add(*gHinge0, gHinge0Id);
  authoringCollection->add(*gHinge1, gHinge1Id);
  authoringCollection->add(*gGearJoint, gGearJointId);
  PxSerialization::complete(*authoringCollection, *gSerializationRegistry);
  gMetrics.collectionCompleted = 1;
  gMetrics.serializable =
      PxSerialization::isSerializable(*authoringCollection,
                                      *gSerializationRegistry)
          ? 1u
          : 0u;
  if (!gMetrics.serializable) {
    authoringCollection->release();
    setSerializationFailure("serialization_not_serializable");
    return false;
  }

  PxDefaultMemoryOutputStream output;
  gMetrics.serializeSuccess =
      PxSerialization::serializeCollectionToBinary(
          output, *authoringCollection, *gSerializationRegistry)
          ? 1u
          : 0u;
  gMetrics.serializedBytes = output.getSize();
  authoringCollection->release();
  if (!gMetrics.serializeSuccess || gMetrics.serializedBytes == 0) {
    setSerializationFailure("serialization_write");
    return false;
  }

  const PxU64 allocationSize =
      gMetrics.serializedBytes + PX_SERIAL_FILE_ALIGN - 1;
  if (allocationSize < gMetrics.serializedBytes ||
      allocationSize >
          static_cast<PxU64>(std::numeric_limits<size_t>::max())) {
    setSerializationFailure("serialization_allocation_size");
    return false;
  }
  gBinaryBlockRaw =
      static_cast<PxU8 *>(std::malloc(static_cast<size_t>(allocationSize)));
  if (!gBinaryBlockRaw) {
    setSerializationFailure("serialization_allocation");
    return false;
  }
  gMetrics.binaryBlockAllocated = 1;
  const size_t rawAddress = reinterpret_cast<size_t>(gBinaryBlockRaw);
  const size_t alignedAddress =
      (rawAddress + PX_SERIAL_FILE_ALIGN - 1) &
      ~(size_t(PX_SERIAL_FILE_ALIGN) - 1);
  gBinaryBlockAligned = reinterpret_cast<void *>(alignedAddress);
  gBinaryBlockSize = gMetrics.serializedBytes;
  gMetrics.binaryAligned =
      (alignedAddress & (PX_SERIAL_FILE_ALIGN - 1)) == 0 ? 1u : 0u;
  if (!gMetrics.binaryAligned) {
    setSerializationFailure("serialization_alignment");
    return false;
  }
  std::memcpy(gBinaryBlockAligned, output.getData(),
              static_cast<size_t>(gMetrics.serializedBytes));

  releaseAuthoringFixture();
  gLoadedCollection = PxSerialization::createCollectionFromBinary(
      gBinaryBlockAligned, *gSerializationRegistry);
  if (!gLoadedCollection) {
    setSerializationFailure("deserialization_failed");
    return false;
  }
  gMetrics.deserializeSuccess = 1;

  gGearActor0 = findSerializedObject<PxRigidDynamic>(gGearActor0Id);
  gGearActor1 = findSerializedObject<PxRigidDynamic>(gGearActor1Id);
  gHinge0 = findSerializedObject<PxRevoluteJoint>(gHinge0Id);
  gHinge1 = findSerializedObject<PxRevoluteJoint>(gHinge1Id);
  gGearJoint = findSerializedObject<PxGearJoint>(gGearJointId);

  gMetrics.loadedObjects = gLoadedCollection->getNbObjects();
  for (PxU32 i = 0; i < gMetrics.loadedObjects; ++i) {
    PxBase &object = gLoadedCollection->getObject(i);
    gMetrics.loadedActors += object.is<PxRigidDynamic>() ? 1u : 0u;
    gMetrics.loadedConstraints += object.is<PxConstraint>() ? 1u : 0u;
    gMetrics.loadedRevolute += object.is<PxRevoluteJoint>() ? 1u : 0u;
    gMetrics.loadedGear += object.is<PxGearJoint>() ? 1u : 0u;
  }
  validateSerializedIdentity();
  if (!gGearActor0 || !gGearActor1 || !gHinge0 || !gHinge1 ||
      !gGearJoint || gMetrics.loadedActors != 2 ||
      gMetrics.loadedConstraints != 3 || gMetrics.loadedRevolute != 2 ||
      gMetrics.loadedGear != 1) {
    setSerializationFailure("deserialized_types");
    return false;
  }
  if (gMetrics.dependencyIdentity != 1 || gMetrics.actorIdentity != 1) {
    setSerializationFailure("deserialized_identity");
    return false;
  }
  return true;
}

static void recordJointGeometry(const PxJoint &joint, bool checkAnchor) {
  const PxTransform frame0 =
      getJointFrameWorld(joint, PxJointActorIndex::eACTOR0);
  const PxTransform frame1 =
      getJointFrameWorld(joint, PxJointActorIndex::eACTOR1);
  if (!frame0.isFinite() || !frame1.isFinite()) {
    gMetrics.nonFinite++;
    return;
  }
  if (checkAnchor)
    gMetrics.maxAnchorError =
        PxMax(gMetrics.maxAnchorError, (frame1.p - frame0.p).magnitude());
  const PxReal axisDot = frame0.q.getBasisVector0().dot(
      frame1.q.getBasisVector0());
  gMetrics.minAxisDot = PxMin(gMetrics.minAxisDot, axisDot);
}

static void recordActorState(const PxRigidDynamic &actor,
                             const PxVec3 &initialPosition) {
  const PxTransform pose = actor.getGlobalPose();
  const PxVec3 linearVelocity = actor.getLinearVelocity();
  const PxVec3 angularVelocity = actor.getAngularVelocity();
  if (!pose.isFinite() || !linearVelocity.isFinite() ||
      !angularVelocity.isFinite()) {
    gMetrics.nonFinite++;
    return;
  }
  gMetrics.maxCenterDrift = PxMax(
      gMetrics.maxCenterDrift, (pose.p - initialPosition).magnitude());
  gMetrics.maxQuaternionNormError =
      PxMax(gMetrics.maxQuaternionNormError,
            PxAbs(pose.q.magnitudeSquared() - 1.0f));
  gMetrics.maxAbsPosition =
      PxMax(gMetrics.maxAbsPosition, maxAbsElement(pose.p));
  gMetrics.maxLinearSpeed =
      PxMax(gMetrics.maxLinearSpeed, linearVelocity.magnitude());
  gMetrics.maxAngularSpeed =
      PxMax(gMetrics.maxAngularSpeed, angularVelocity.magnitude());
}

static bool initializeOracle() {
  if (!gScene || !gGearActor0 || !gGearActor1 || !gHinge0 || !gHinge1 ||
      !gGearJoint)
    return false;

  gMetrics.endpointSign0 = getEndpointSign(*gHinge0, gGearActor0);
  gMetrics.endpointSign1 = getEndpointSign(*gHinge1, gGearActor1);
  gMetrics.ratioReadback = gGearJoint->getGearRatio();
  gMetrics.initialAngle0 = gHinge0->getAngle();
  gMetrics.initialAngle1 = gHinge1->getAngle();
  gMetrics.previousRawAngle0 = gMetrics.initialAngle0;
  gMetrics.previousRawAngle1 = gMetrics.initialAngle1;
  gMetrics.unwrappedAngle0 = gMetrics.initialAngle0;
  gMetrics.unwrappedAngle1 = gMetrics.initialAngle1;
  gMetrics.baselinePhase =
      gMetrics.endpointSign0 * gHeadlessConfig.ratio *
          gMetrics.unwrappedAngle0 +
      gMetrics.endpointSign1 * gMetrics.unwrappedAngle1;
  gMetrics.initialPhaseOffsetWitness = gMetrics.initialAngle1;
  gMetrics.driveEnabledReadback =
      gHinge0->getRevoluteJointFlags() & PxRevoluteJointFlag::eDRIVE_ENABLED
          ? 1u
          : 0u;

  const PxBase *readbackHinge0 = NULL;
  const PxBase *readbackHinge1 = NULL;
  gGearJoint->getHinges(readbackHinge0, readbackHinge1);
  PxRigidActor *gearActor0 = NULL;
  PxRigidActor *gearActor1 = NULL;
  gGearJoint->getActors(gearActor0, gearActor1);
  PxRigidActor *hinge0Actor0 = NULL;
  PxRigidActor *hinge0Actor1 = NULL;
  PxRigidActor *hinge1Actor0 = NULL;
  PxRigidActor *hinge1Actor1 = NULL;
  gHinge0->getActors(hinge0Actor0, hinge0Actor1);
  gHinge1->getActors(hinge1Actor0, hinge1Actor1);

  gMetrics.topologyDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gMetrics.topologyStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gMetrics.topologyConstraints = gScene->getNbConstraints();
  gMetrics.topologyShapes0 = gGearActor0->getNbShapes();
  gMetrics.topologyShapes1 = gGearActor1->getNbShapes();
  const bool topologyOk =
      gMetrics.topologyDynamicActors == 2 &&
      gMetrics.topologyStaticActors == 0 &&
      gMetrics.topologyConstraints == 3 && gMetrics.topologyShapes0 == 5 &&
      gMetrics.topologyShapes1 == 2 && readbackHinge0 == gHinge0 &&
      readbackHinge1 == gHinge1 && gearActor0 == gGearActor0 &&
      gearActor1 == gGearActor1 && hinge0Actor0 == NULL &&
      hinge0Actor1 == gGearActor0 && hinge1Actor0 == NULL &&
      hinge1Actor1 == gGearActor1 &&
      gGearJoint->getConcreteType() == PxJointConcreteType::eGEAR &&
      std::strcmp(gGearJoint->getConcreteTypeName(), "PxGearJoint") == 0 &&
      PxAbs(gMetrics.ratioReadback - gHeadlessConfig.ratio) <= 1e-6f &&
      gMetrics.endpointSign0 == 1.0f && gMetrics.endpointSign1 == 1.0f;
  gMetrics.topologyOk = topologyOk ? 1u : 0u;

  if (!PxIsFinite(gMetrics.initialAngle0) ||
      !PxIsFinite(gMetrics.initialAngle1) ||
      !PxIsFinite(gMetrics.baselinePhase))
    return false;

  recordJointGeometry(*gHinge0, true);
  recordJointGeometry(*gHinge1, true);
  recordJointGeometry(*gGearJoint, false);
  gOracleInitialized = true;
  return true;
}

static PxU32 getTailStartFrame() {
  PxU32 start = PxMax(gWarmupFrames, (gHeadlessOptions.frames * 3u) / 4u);
  if (gHeadlessCase == eCASE_REVERSE)
    start = PxMax(start,
                  gHeadlessConfig.reverseFrame + gEventRecoveryFrames);
  else if (gHeadlessCase == eCASE_EXTERNAL_IMPULSE)
    start = PxMax(start,
                  gHeadlessConfig.impulseFrame + gEventRecoveryFrames);
  return start;
}

static void recordTailSample(PxU32 frame, PxReal spin0, PxReal spin1,
                             PxReal phaseResidual) {
  gMetrics.tailSamples++;
  gMetrics.maxTailPhaseError =
      PxMax(gMetrics.maxTailPhaseError, PxAbs(phaseResidual));
  gMetrics.tailPhaseSquaredSum += phaseResidual * phaseResidual;

  const double time =
      double(frame - getTailStartFrame()) * double(gHeadlessOptions.dt);
  gMetrics.phaseTimeSum += time;
  gMetrics.phaseValueSum += double(phaseResidual);
  gMetrics.phaseTimeSquaredSum += time * time;
  gMetrics.phaseTimeValueSum += time * double(phaseResidual);

  if (gHeadlessCase != eCASE_EXTERNAL_IMPULSE) {
    const PxReal targetMagnitude = PxAbs(gCurrentDriveTarget);
    const PxReal targetFloor =
        gHeadlessCase == eCASE_SINUSOIDAL
            ? gSinusoidalTargetAnalysisFloor
            : gVelocityFloor;
    if (targetMagnitude >= targetFloor) {
      const PxReal normalizedTrackingError =
          PxAbs(spin0 - gCurrentDriveTarget) / targetMagnitude;
      gMetrics.driveTrackingSamples++;
      gMetrics.driveTrackingErrorSquaredSum +=
          normalizedTrackingError * normalizedTrackingError;
      gMetrics.driveTrackingErrorMax =
          PxMax(gMetrics.driveTrackingErrorMax, normalizedTrackingError);
      if (spin0 * gCurrentDriveTarget > 0.0f &&
          PxAbs(spin0) >= gVelocityFloor)
        gMetrics.driveAlignedSamples++;
      else
        gMetrics.driveOpposedSamples++;
    }
  }

  if (gHeadlessCase == eCASE_SINUSOIDAL &&
      PxAbs(gCurrentDriveTarget) < gSinusoidalTargetAnalysisFloor)
    return;

  const PxReal scaledVelocity0 = gHeadlessConfig.ratio * spin0;
  const PxReal denominator =
      PxMax(gVelocityFloor, PxMax(PxAbs(scaledVelocity0), PxAbs(spin1)));
  if (denominator <= gVelocityFloor && PxAbs(spin0) < gVelocityFloor &&
      PxAbs(spin1) < gVelocityFloor)
    return;

  const PxReal residual = scaledVelocity0 + spin1;
  const PxReal normalizedResidual = PxAbs(residual) / denominator;
  gMetrics.maxVelocityResidualNorm =
      PxMax(gMetrics.maxVelocityResidualNorm, normalizedResidual);
  gMetrics.velocityResidualSquaredSum +=
      normalizedResidual * normalizedResidual;
  gMetrics.ratioSamples++;

  if (PxAbs(spin0) >= gVelocityFloor) {
    gMetrics.measuredRatioSum += -spin1 / spin0;
    gMetrics.measuredRatioSamples++;
    if (PxAbs(spin1) >= gVelocityFloor) {
      const bool oppositeDirections = spin0 * spin1 < 0.0f;
      const bool expectedOpposite = gHeadlessConfig.ratio > 0.0f;
      gMetrics.directionSamples++;
      if (oppositeDirections != expectedOpposite)
        gMetrics.directionViolations++;
    }
  }
}

static void sampleHeadlessState() {
  if (!gOracleInitialized || !gGearActor0 || !gGearActor1 || !gHinge0 ||
      !gHinge1 || !gGearJoint || !gMetrics.completedFrames)
    return;

  const PxU32 frame = gMetrics.completedFrames - 1u;
  const PxReal rawAngle0 = gHinge0->getAngle();
  const PxReal rawAngle1 = gHinge1->getAngle();
  const PxVec3 relativeVelocity0 = gHinge0->getRelativeAngularVelocity();
  const PxVec3 relativeVelocity1 = gHinge1->getRelativeAngularVelocity();
  if (!PxIsFinite(rawAngle0) || !PxIsFinite(rawAngle1) ||
      !relativeVelocity0.isFinite() || !relativeVelocity1.isFinite()) {
    gMetrics.nonFinite++;
    return;
  }

  const PxReal delta0 =
      shortestAngleDelta(rawAngle0, gMetrics.previousRawAngle0);
  const PxReal delta1 =
      shortestAngleDelta(rawAngle1, gMetrics.previousRawAngle1);
  gMetrics.previousRawAngle0 = rawAngle0;
  gMetrics.previousRawAngle1 = rawAngle1;
  gMetrics.unwrappedAngle0 += delta0;
  gMetrics.unwrappedAngle1 += delta1;
  gMetrics.totalAbsTravel0 += PxAbs(delta0);
  gMetrics.totalAbsTravel1 += PxAbs(delta1);
  gMetrics.signedTravel0 += delta0;
  gMetrics.signedTravel1 += delta1;
  if (PxAbs(delta0) > PxPi * 0.75f || PxAbs(delta1) > PxPi * 0.75f ||
      PxAbs(relativeVelocity0.x * gHeadlessOptions.dt) > PxPi * 0.75f ||
      PxAbs(relativeVelocity1.x * gHeadlessOptions.dt) > PxPi * 0.75f)
    gMetrics.unwrapAliasRisk++;

  const PxReal spin0 = gMetrics.endpointSign0 * relativeVelocity0.x;
  const PxReal spin1 = gMetrics.endpointSign1 * relativeVelocity1.x;
  const PxReal phaseResidual =
      gMetrics.endpointSign0 * gHeadlessConfig.ratio *
          gMetrics.unwrappedAngle0 +
      gMetrics.endpointSign1 * gMetrics.unwrappedAngle1 -
      gMetrics.baselinePhase;
  if (!PxIsFinite(spin0) || !PxIsFinite(spin1) ||
      !PxIsFinite(phaseResidual)) {
    gMetrics.nonFinite++;
    return;
  }

  gMetrics.sampleCount++;
  if (frame >= gWarmupFrames)
    gMetrics.maxPhaseError =
        PxMax(gMetrics.maxPhaseError, PxAbs(phaseResidual));
  if (frame >= getTailStartFrame())
    recordTailSample(frame, spin0, spin1, phaseResidual);

  const PxReal lockedSpeed0 =
      PxSqrt(relativeVelocity0.y * relativeVelocity0.y +
             relativeVelocity0.z * relativeVelocity0.z);
  const PxReal lockedSpeed1 =
      PxSqrt(relativeVelocity1.y * relativeVelocity1.y +
             relativeVelocity1.z * relativeVelocity1.z);
  gMetrics.maxLockedAxisSpeed =
      PxMax(gMetrics.maxLockedAxisSpeed, PxMax(lockedSpeed0, lockedSpeed1));

  if (frame >= gWarmupFrames) {
    if (spin0 >= gVelocityFloor)
      gMetrics.positiveMotionSamples++;
    else if (spin0 <= -gVelocityFloor)
      gMetrics.negativeMotionSamples++;
  }

  if (gHeadlessCase == eCASE_REVERSE) {
    if (frame >= gWarmupFrames && frame < gHeadlessConfig.reverseFrame &&
        spin0 >= gVelocityFloor)
      gMetrics.preReverseAlignedSamples++;
    if (frame >= gHeadlessConfig.reverseFrame + gEventRecoveryFrames &&
        spin0 <= -gVelocityFloor)
      gMetrics.postReverseAlignedSamples++;
  }

  if (gHeadlessCase == eCASE_EXTERNAL_IMPULSE &&
      frame >= gHeadlessConfig.impulseFrame &&
      frame < gHeadlessConfig.impulseFrame + gImpulseResponseWindow) {
    const PxReal deltaSpin0 = spin0 - gImpulsePreVelocity0;
    const PxReal deltaSpin1 = spin1 - gImpulsePreVelocity1;
    const PxReal scaledDelta0 = gHeadlessConfig.ratio * deltaSpin0;
    const PxReal responseMagnitude =
        PxMax(PxAbs(scaledDelta0), PxAbs(deltaSpin1));
    gMetrics.impulseResponseSamples++;
    if (responseMagnitude > gMetrics.impulseResponseMagnitude) {
      gMetrics.impulseResponseMagnitude = responseMagnitude;
      gMetrics.impulseDelta0 = deltaSpin0;
      gMetrics.impulseDelta1 = deltaSpin1;
      gMetrics.impulseResponseResidualNorm =
          PxAbs(scaledDelta0 + deltaSpin1) /
          PxMax(gVelocityFloor, responseMagnitude);
      const PxVec3 actualWorldAngular0 = gGearActor0->getAngularVelocity();
      const PxVec3 actualWorldAngular1 = gGearActor1->getAngularVelocity();
      const PxVec3 actualWorldDelta0 =
          actualWorldAngular0 - gImpulsePreWorldAngular0;
      const PxVec3 actualWorldDelta1 =
          actualWorldAngular1 - gImpulsePreWorldAngular1;
      const PxVec3 expectedWorldDelta0 =
          gImpulseExpectedWorldAngular0 - gImpulsePreWorldAngular0;
      const PxVec3 expectedWorldDelta1 =
          gImpulseExpectedWorldAngular1 - gImpulsePreWorldAngular1;
      gMetrics.impulseExpectedDelta0 =
          expectedWorldDelta0.dot(gImpulseGearAxis0);
      gMetrics.impulseExpectedDelta1 =
          expectedWorldDelta1.dot(gImpulseGearAxis1);
      gMetrics.impulseActualWorldDelta0 =
          actualWorldDelta0.dot(gImpulseGearAxis0);
      gMetrics.impulseActualWorldDelta1 =
          actualWorldDelta1.dot(gImpulseGearAxis1);
      const PxReal expectedMagnitude =
          PxMax(gImpulseMinimumResponse,
                PxMax(expectedWorldDelta0.magnitude(),
                      expectedWorldDelta1.magnitude()));
      gMetrics.impulseProjectionErrorNorm =
          PxMax((actualWorldDelta0 - expectedWorldDelta0).magnitude(),
                (actualWorldDelta1 - expectedWorldDelta1).magnitude()) /
          expectedMagnitude;
    }
  }
  if (gHeadlessCase == eCASE_EXTERNAL_IMPULSE &&
      frame >= gHeadlessConfig.impulseFrame + gEventRecoveryFrames &&
      frame < gHeadlessConfig.impulseFrame + gEventRecoveryFrames +
                  gImpulseRecoveryWindow) {
    const PxReal scaledSpin0 = gHeadlessConfig.ratio * spin0;
    const PxReal denominator = PxMax(PxAbs(scaledSpin0), PxAbs(spin1));
    if (denominator >= gImpulseRecoveryVelocityFloor) {
      gMetrics.impulseRecoverySamples++;
      gMetrics.impulseRecoveryResidualMax =
          PxMax(gMetrics.impulseRecoveryResidualMax,
                PxAbs(scaledSpin0 + spin1) / denominator);
    }
  }

  recordActorState(*gGearActor0, gInitialPosition0);
  recordActorState(*gGearActor1, gInitialPosition1);
  recordJointGeometry(*gHinge0, true);
  recordJointGeometry(*gHinge1, true);
  recordJointGeometry(*gGearJoint, false);
}

static void updateDriveTarget(PxReal target) {
  gCurrentDriveTarget = target;
  if (gHinge0)
    gHinge0->setDriveVelocity(target);
}

static void applyHeadlessExcitation(PxU32 frame) {
  if (gHeadlessCase == eCASE_REVERSE &&
      frame == gHeadlessConfig.reverseFrame) {
    updateDriveTarget(-gDriveVelocity);
    gMetrics.reverseEvents++;
  } else if (gHeadlessCase == eCASE_SINUSOIDAL) {
    const PxReal time = PxReal(frame) * gHeadlessOptions.dt;
    const PxReal target = gSinusoidalDriveAmplitude * PxSin(time);
    updateDriveTarget(target);
    int targetSign = 0;
    if (target > 0.1f)
      targetSign = 1;
    else if (target < -0.1f)
      targetSign = -1;
    if (targetSign && gLastSinusoidalTargetSign &&
        targetSign != gLastSinusoidalTargetSign)
      gMetrics.targetSignChanges++;
    if (targetSign)
      gLastSinusoidalTargetSign = targetSign;
  }

  if (gHeadlessCase == eCASE_EXTERNAL_IMPULSE &&
      frame == gHeadlessConfig.impulseFrame && gGearActor1 && gHinge0 &&
      gHinge1) {
    gImpulsePreVelocity0 =
        gMetrics.endpointSign0 * gHinge0->getRelativeAngularVelocity().x;
    gImpulsePreVelocity1 =
        gMetrics.endpointSign1 * gHinge1->getRelativeAngularVelocity().x;
    gImpulsePreWorldAngular0 = gGearActor0->getAngularVelocity();
    gImpulsePreWorldAngular1 = gGearActor1->getAngularVelocity();
    gImpulseGearAxis0 =
        getJointFrameWorld(*gGearJoint, PxJointActorIndex::eACTOR0)
            .q.getBasisVector0()
            .getNormalized();
    gImpulseGearAxis1 =
        getJointFrameWorld(*gGearJoint, PxJointActorIndex::eACTOR1)
            .q.getBasisVector0()
            .getNormalized();
    const PxVec3 jacobian0 =
        gImpulseGearAxis0 * gHeadlessConfig.ratio;
    // ExtGearJoint emits angular1=-solverAxis1.  For this fixture the
    // solver-prep body-B axis is opposite the readable hinge/world axis, so
    // the scalar row observed by the public hinge velocities is
    // ratio*spin0+spin1=0.
    const PxVec3 jacobian1 = gImpulseGearAxis1;
    const PxVec3 rawWorldAngular0 = gImpulsePreWorldAngular0;
    const PxVec3 rawWorldAngular1 =
        gImpulsePreWorldAngular1 +
        applyInverseInertiaWorld(*gGearActor1,
                                 PxVec3(0.0f, 0.0f, gAngularImpulse));
    const PxVec3 response0 =
        applyInverseInertiaWorld(*gGearActor0, jacobian0);
    const PxVec3 response1 =
        applyInverseInertiaWorld(*gGearActor1, jacobian1);
    gMetrics.impulseProjectionDenominator =
        jacobian0.dot(response0) + jacobian1.dot(response1);
    if (gMetrics.impulseProjectionDenominator > 1e-12f) {
      const PxReal lambda =
          -(jacobian0.dot(rawWorldAngular0) +
            jacobian1.dot(rawWorldAngular1)) /
          gMetrics.impulseProjectionDenominator;
      gImpulseExpectedWorldAngular0 =
          rawWorldAngular0 + response0 * lambda;
      gImpulseExpectedWorldAngular1 =
          rawWorldAngular1 + response1 * lambda;
    } else {
      gImpulseExpectedWorldAngular0 = rawWorldAngular0;
      gImpulseExpectedWorldAngular1 = rawWorldAngular1;
    }
    gMetrics.impulsePreSpeedMax =
        PxMax(PxAbs(gImpulsePreVelocity0), PxAbs(gImpulsePreVelocity1));
    gGearActor1->addTorque(PxVec3(0.0f, 0.0f, gAngularImpulse),
                           PxForceMode::eIMPULSE, true);
    gMetrics.impulseEvents++;
  }
}

void initPhysics(bool interactive) {
  resetRuntimeState();
  gFoundation =
      PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
  if (!gFoundation) {
    gInitializationFailed = true;
    return;
  }

  if (interactive) {
    gPvd = PxCreatePvd(*gFoundation);
    if (gPvd) {
      gPvdTransport =
          PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
      if (gPvdTransport)
        gPvd->connect(*gPvdTransport, PxPvdInstrumentationFlag::eALL);
    }
  }

  gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
                             PxTolerancesScale(), true, gPvd);
  if (!gPhysics) {
    gInitializationFailed = true;
    return;
  }
  if (!PxInitExtensions(*gPhysics, gPvd)) {
    gInitializationFailed = true;
    return;
  }
  gExtensionsInitialized = true;

  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
  const PxU32 dispatcherThreads =
      interactive ? 2u : gHeadlessOptions.dispatcherThreads;
  gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
  if (!gDispatcher) {
    gInitializationFailed = true;
    return;
  }
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = PxDefaultSimulationFilterShader;
  sceneDesc.solverType = gSolverType;
  gScene = gPhysics->createScene(sceneDesc);
  if (!gScene) {
    gInitializationFailed = true;
    return;
  }

  if (interactive) {
    PxPvdSceneClient *pvdClient = gScene->getScenePvdClient();
    if (pvdClient) {
      pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS,
                                 true);
      pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
      pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES,
                                 true);
    }
  }

  gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
  if (!gMaterial) {
    gInitializationFailed = true;
    return;
  }

  const PxReal radius0 = 5.0f;
  const PxReal radius1 = 2.0f;
  const PxReal extent0 = radius0 * PxSqrt(2.0f);
  const PxReal extent1 = radius1 * PxSqrt(2.0f);
  const PxReal teethLength0 = extent0 - radius0;
  const PxReal teethLength1 = extent1 - radius1;
  const PxReal extra = (teethLength0 + teethLength1) * 0.75f;
  const PxBoxGeometry boxGeometry0(radius0, radius0, 0.5f);
  const PxBoxGeometry boxGeometry1(radius1, radius1, 0.25f);
  gInitialPosition0 = PxVec3(0.0f, 10.0f, 0.0f);
  gInitialPosition1 = PxVec3(radius0 + radius1 + extra, 10.0f, 0.0f);

  gGearActor0 = createGearWithBoxes(*gPhysics, boxGeometry0,
                                    PxTransform(gInitialPosition0),
                                    *gMaterial, 5u);
  if (!gGearActor0) {
    gInitializationFailed = true;
    return;
  }

  const PxQuat phaseOffset =
      !interactive && gHeadlessCase == eCASE_PHASE_OFFSET
          ? PxQuat(gPhaseOffsetAngle, PxVec3(0.0f, 0.0f, 1.0f))
          : PxQuat(PxIdentity);
  gGearActor1 = createGearWithBoxes(
      *gPhysics, boxGeometry1, PxTransform(gInitialPosition1, phaseOffset),
      *gMaterial, 2u);
  if (!gGearActor1) {
    gInitializationFailed = true;
    return;
  }

  const PxQuat x2z = PxShortestRotation(PxVec3(1.0f, 0.0f, 0.0f),
                                        PxVec3(0.0f, 0.0f, 1.0f));
  gHinge0 = PxRevoluteJointCreate(
      *gPhysics, NULL, PxTransform(gInitialPosition0, x2z), gGearActor0,
      PxTransform(PxVec3(0.0f), x2z));
  markJointCreated(gHinge0);
  gHinge1 = PxRevoluteJointCreate(
      *gPhysics, NULL, PxTransform(gInitialPosition1, x2z), gGearActor1,
      PxTransform(PxVec3(0.0f), x2z));
  markJointCreated(gHinge1);
  if (!gHinge0 || !gHinge1) {
    gInitializationFailed = true;
    return;
  }

  const bool passiveImpulseCase =
      !interactive && gHeadlessCase == eCASE_EXTERNAL_IMPULSE;
  gCurrentDriveTarget =
      !interactive && (gHeadlessCase == eCASE_SINUSOIDAL ||
                       gHeadlessCase == eCASE_EXTERNAL_IMPULSE)
          ? 0.0f
          : gDriveVelocity;
  gHinge0->setDriveVelocity(gCurrentDriveTarget);
  gHinge0->setRevoluteJointFlag(PxRevoluteJointFlag::eDRIVE_ENABLED,
                                !passiveImpulseCase);

  gGearJoint = PxGearJointCreate(
      *gPhysics, gGearActor0, PxTransform(PxVec3(0.0f), x2z), gGearActor1,
      PxTransform(PxVec3(0.0f), x2z));
  markJointCreated(gGearJoint);
  if (!gGearJoint || !gGearJoint->setHinges(gHinge0, gHinge1)) {
    gInitializationFailed = true;
    return;
  }
  gGearJoint->setGearRatio(gHeadlessConfig.ratio);

  if (!interactive &&
      gHeadlessConfig.serializationMode == eSERIALIZATION_BINARY) {
    if (!replaceFixtureWithBinaryRoundTrip()) {
      gInitializationFailed = true;
      return;
    }
    if (!gScene->addCollection(*gLoadedCollection)) {
      setSerializationFailure("deserialized_scene_add");
      gInitializationFailed = true;
      return;
    }
  } else {
    gScene->addActor(*gGearActor0);
    gScene->addActor(*gGearActor1);
    validateSerializedIdentity();
  }

  if (!interactive && !initializeOracle())
    gInitializationFailed = true;
}

void stepPhysics(bool interactive) {
  if (!gScene || gInitializationFailed)
    return;

  if (!interactive)
    applyHeadlessExcitation(gMetrics.completedFrames);

  const PxReal timeStep = interactive ? (1.0f / 60.0f) : gHeadlessOptions.dt;
  gScene->simulate(timeStep);
  if (!interactive)
    gMetrics.simulateCalls++;
  PxU32 errorState = 0;
  if (!interactive)
    gMetrics.fetchCalls++;
  if (!gScene->fetchResults(true, &errorState)) {
    if (!interactive) {
      gMetrics.fetchFailures++;
      gMetrics.fetchErrorState |= errorState;
    }
    return;
  }

  if (!interactive) {
    gMetrics.fetchErrorState |= errorState;
    gMetrics.completedFrames++;
    sampleHeadlessState();
  }
}

static void releaseLoadedFixture() {
  if (gLoadedCollection) {
    std::vector<PxShape *> exclusiveShapes;
    for (PxU32 i = 0; i < gLoadedCollection->getNbObjects(); ++i) {
      PxShape *shape = gLoadedCollection->getObject(i).is<PxShape>();
      if (shape && shape->isExclusive()) {
        shape->acquireReference();
        exclusiveShapes.push_back(shape);
      }
    }

    PxCollectionExt::releaseObjects(*gLoadedCollection, false);
    for (size_t i = 0; i < exclusiveShapes.size(); ++i) {
      PxShape *shape = exclusiveShapes[i];
      while (shape->getReferenceCount() > 1)
        shape->release();
      shape->release();
    }
    gLoadedCollection->release();
    gLoadedCollection = NULL;
    gMetrics.loadedCollectionReleased = 1;
    gGearJoint = NULL;
    gHinge0 = NULL;
    gHinge1 = NULL;
    gGearActor0 = NULL;
    gGearActor1 = NULL;
  }

  if (gBinaryBlockRaw) {
    std::free(gBinaryBlockRaw);
    gBinaryBlockRaw = NULL;
    gBinaryBlockAligned = NULL;
    gBinaryBlockSize = 0;
    gMetrics.binaryBlockFreed = 1;
  }
  PX_RELEASE(gSerializationRegistry);
}

void cleanupPhysics(bool interactive) {
  if (gLoadedCollection) {
    releaseLoadedFixture();
  } else {
    releaseAuthoringFixture();
    if (gBinaryBlockRaw || gSerializationRegistry)
      releaseLoadedFixture();
  }
  PX_RELEASE(gMaterial);
  PX_RELEASE(gScene);
  PX_RELEASE(gDispatcher);
  if (gExtensionsInitialized) {
    PxCloseExtensions();
    gExtensionsInitialized = false;
  }
  PX_RELEASE(gPhysics);
  PX_RELEASE(gPvd);
  PX_RELEASE(gPvdTransport);
  PX_RELEASE(gFoundation);

  gOracleInitialized = false;
  gMetrics.cleanupComplete =
      !gGearJoint && !gHinge0 && !gHinge1 && !gGearActor0 && !gGearActor1 &&
              !gMaterial && !gScene && !gDispatcher && !gPhysics && !gPvd &&
               !gPvdTransport && !gFoundation && !gExtensionsInitialized &&
               !gSerializationRegistry && !gLoadedCollection &&
               !gBinaryBlockRaw && !gBinaryBlockAligned &&
               gBinaryBlockSize == 0 &&
               gMetrics.shapeRefsReleased == gMetrics.shapeRefsCreated &&
              gMetrics.actorsReleased == gMetrics.actorsCreated &&
              gMetrics.jointsReleased == gMetrics.jointsCreated
          ? 1u
          : 0u;

  if (interactive)
    std::printf("SnippetGearJoint done.\n");
}

static void setGateError(GearGateEvaluation &evaluation, const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static void setGateFailure(GearGateEvaluation &evaluation,
                           const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
  evaluation.status = "FAIL";
  evaluation.reason = reason;
}

static GearGateEvaluation evaluateGate(PxU32 physicsErrors) {
  GearGateEvaluation evaluation;
  if (gMetrics.ratioSamples) {
    evaluation.velocityResidualRms =
        PxSqrt(gMetrics.velocityResidualSquaredSum /
               PxReal(gMetrics.ratioSamples));
  }
  if (gMetrics.measuredRatioSamples) {
    evaluation.measuredRatioMean =
        gMetrics.measuredRatioSum / PxReal(gMetrics.measuredRatioSamples);
  }
  if (gMetrics.tailSamples)
    evaluation.tailPhaseRms =
        PxSqrt(gMetrics.tailPhaseSquaredSum / PxReal(gMetrics.tailSamples));
  if (gMetrics.tailSamples >= 2) {
    const double count = double(gMetrics.tailSamples);
    const double denominator =
        count * gMetrics.phaseTimeSquaredSum -
        gMetrics.phaseTimeSum * gMetrics.phaseTimeSum;
    if (std::fabs(denominator) > 1e-12) {
      evaluation.phaseDriftSlope = PxReal(
          (count * gMetrics.phaseTimeValueSum -
           gMetrics.phaseTimeSum * gMetrics.phaseValueSum) /
          denominator);
    }
  }
  if (gMetrics.driveTrackingSamples) {
    evaluation.driveTrackingErrorRms = PxSqrt(
        gMetrics.driveTrackingErrorSquaredSum /
        PxReal(gMetrics.driveTrackingSamples));
  }

  const PxU32 expectedTailSamples =
      gHeadlessOptions.frames - getTailStartFrame();

  if (gHeadlessConfig.serializationMode == eSERIALIZATION_BINARY &&
      std::strcmp(gSerializationFailureReason, "none") != 0)
    setGateFailure(evaluation, gSerializationFailureReason);
  if (gHeadlessConfig.serializationMode == eSERIALIZATION_BINARY &&
      (gMetrics.serializationRequested != 1 ||
       gMetrics.registryCreated != 1 ||
       gMetrics.collectionCompleted != 1 || gMetrics.serializable != 1 ||
       gMetrics.serializeSuccess != 1 || gMetrics.serializedBytes == 0 ||
       gMetrics.binaryBlockAllocated != 1 || gMetrics.binaryAligned != 1 ||
       gMetrics.deserializeSuccess != 1 || gMetrics.loadedActors != 2 ||
       gMetrics.loadedConstraints != 3 || gMetrics.loadedRevolute != 2 ||
       gMetrics.loadedGear != 1 || gMetrics.dependencyIdentity != 1 ||
       gMetrics.actorIdentity != 1 || gMetrics.authoringReleased != 1 ||
       gMetrics.loadedCollectionReleased != 1 ||
       gMetrics.binaryBlockFreed != 1))
    setGateFailure(evaluation, "serialization_metrics");
  if (gInitializationFailed)
    setGateError(evaluation, "initialization");
  if (gMetrics.fetchFailures || gMetrics.fetchErrorState)
    setGateError(evaluation, "fetch_failure");
  if (gMetrics.simulateCalls != gHeadlessOptions.frames ||
      gMetrics.fetchCalls != gMetrics.simulateCalls ||
      gMetrics.completedFrames != gHeadlessOptions.frames)
    setGateError(evaluation, "execution_incomplete");
  if (!gMetrics.nonFinite &&
      (gMetrics.sampleCount != gMetrics.completedFrames ||
       gMetrics.tailSamples != expectedTailSamples))
    setGateError(evaluation, "sampling_incomplete");
  if (!gMetrics.cleanupComplete)
    setGateError(evaluation, "cleanup_incomplete");
  if (!gMetrics.topologyOk)
    setGateFailure(evaluation, "topology");
  if (physicsErrors)
    setGateFailure(evaluation, "physx_error");
  if (gMetrics.nonFinite)
    setGateFailure(evaluation, "non_finite");
  if (!PxIsFinite(evaluation.velocityResidualRms) ||
      !PxIsFinite(evaluation.measuredRatioMean) ||
      !PxIsFinite(evaluation.tailPhaseRms) ||
      !PxIsFinite(evaluation.phaseDriftSlope) ||
      !PxIsFinite(evaluation.driveTrackingErrorRms))
    setGateFailure(evaluation, "derived_non_finite");
  if (gMetrics.unwrapAliasRisk)
    setGateFailure(evaluation, "unwrap_alias");
  if (gMetrics.maxAbsPosition > gPositionRunawayCap ||
      gMetrics.maxLinearSpeed > gLinearSpeedRunawayCap ||
      gMetrics.maxAngularSpeed > gAngularSpeedRunawayCap ||
      gMetrics.maxQuaternionNormError > gQuaternionNormErrorCap)
    setGateFailure(evaluation, "runaway");
  if (gMetrics.maxAnchorError > gAnchorErrorCap)
    setGateFailure(evaluation, "anchor");
  if (gMetrics.minAxisDot < gMinimumAxisDot)
    setGateFailure(evaluation, "axis_alignment");
  if (gMetrics.maxLockedAxisSpeed > gLockedAxisSpeedCap)
    setGateFailure(evaluation, "locked_axis");
  if (gHeadlessCase != eCASE_EXTERNAL_IMPULSE &&
      (gMetrics.totalAbsTravel0 < gMinimumTravel ||
       gMetrics.totalAbsTravel1 < gMinimumTravel))
    setGateFailure(evaluation, "motion");
  if (gMetrics.tailSamples < 30)
    setGateFailure(evaluation, "ratio_samples");
  if (gHeadlessCase != eCASE_EXTERNAL_IMPULSE &&
      gMetrics.ratioSamples < 10)
    setGateFailure(evaluation, "ratio_samples");
  if (gHeadlessCase != eCASE_EXTERNAL_IMPULSE &&
      gMetrics.maxVelocityResidualNorm > gVelocityResidualCap)
    setGateFailure(evaluation, "ratio_residual");
  if (gHeadlessCase != eCASE_EXTERNAL_IMPULSE &&
      (gMetrics.directionSamples < 10 ||
       gMetrics.directionViolations * 10u > gMetrics.directionSamples))
    setGateFailure(evaluation, "direction");
  if (gHeadlessCase != eCASE_EXTERNAL_IMPULSE &&
      (gMetrics.driveTrackingSamples < 10 ||
       gMetrics.driveOpposedSamples * 10u >
           gMetrics.driveTrackingSamples ||
       evaluation.driveTrackingErrorRms > gDriveTrackingErrorCap))
    setGateFailure(evaluation, "drive_tracking");
  if (gMetrics.maxTailPhaseError > gPhaseErrorCap)
    setGateFailure(evaluation, "phase");
  if (PxAbs(evaluation.phaseDriftSlope) > gPhaseDriftSlopeCap)
    setGateFailure(evaluation, "phase_drift");

  if (gHeadlessCase == eCASE_PHASE_OFFSET &&
      PxAbs(gMetrics.initialPhaseOffsetWitness - gPhaseOffsetAngle) > 1e-3f)
    setGateFailure(evaluation, "phase_offset_witness");
  if (gHeadlessCase == eCASE_REVERSE &&
      (gMetrics.reverseEvents != 1 ||
       gMetrics.preReverseAlignedSamples < 5 ||
       gMetrics.postReverseAlignedSamples < 5))
    setGateFailure(evaluation, "reverse_event");
  if (gHeadlessCase == eCASE_SINUSOIDAL &&
      (gMetrics.targetSignChanges < 4 ||
       gMetrics.positiveMotionSamples < 5 ||
       gMetrics.negativeMotionSamples < 5))
    setGateFailure(evaluation, "sinusoidal_event");
  if (gHeadlessCase == eCASE_EXTERNAL_IMPULSE &&
      (gMetrics.impulseEvents != 1 || gMetrics.driveEnabledReadback != 0 ||
       gMetrics.impulsePreSpeedMax > gImpulseBaselineSpeedCap ||
       gMetrics.impulseResponseSamples != gImpulseResponseWindow ||
       gMetrics.impulseDelta1 * gAngularImpulse <= 0.0f ||
       gHeadlessConfig.ratio * gMetrics.impulseDelta0 *
               gMetrics.impulseDelta1 >=
           0.0f ||
       PxAbs(gHeadlessConfig.ratio * gMetrics.impulseDelta0) <
           gImpulseMinimumResponse ||
       PxAbs(gMetrics.impulseDelta1) < gImpulseMinimumResponse ||
       gMetrics.impulseResponseResidualNorm > gImpulseResidualCap ||
       gMetrics.impulseProjectionDenominator <= 1e-12f ||
       gMetrics.impulseProjectionErrorNorm > gImpulseProjectionErrorCap ||
       gMetrics.impulseRecoverySamples < 10 ||
       gMetrics.impulseRecoveryResidualMax > gImpulseResidualCap))
    setGateFailure(evaluation, "external_impulse_event");
  return evaluation;
}

static void printGateResult(const GearGateEvaluation &evaluation,
                            PxU32 physicsErrors, PxU32 physicsWarnings) {
  std::printf(
      "[AVBD_GATE] schema=1 snippet=SnippetGearJoint case=%s solver=%s "
      "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
      "dispatcherThreads=%u reverseFrame=%u impulseFrame=%u "
      "serialization=%s capability=PARTIAL validation=PROBE status=%s "
      "reason=%s thresholdPolicy=DIAGNOSTIC nonFinite=%u physicsErrors=%u "
      "physicsWarnings=%u simulateCalls=%u fetchCalls=%u fetchFailures=%u "
      "fetchErrorState=%u cleanupComplete=%u topologyOk=%u dynamicActors=%u "
      "staticActors=%u constraints=%u shapes0=%u shapes1=%u "
      "shapeRefsCreated=%u shapeRefsReleased=%u actorsCreated=%u "
      "actorsReleased=%u jointsCreated=%u jointsReleased=%u ratio=%.9g "
      "ratioReadback=%.9g endpointSign0=%.9g endpointSign1=%.9g "
      "initialAngle0=%.9g initialAngle1=%.9g baselinePhase=%.9g "
      "initialPhaseOffsetWitness=%.9g sampleCount=%u "
      "travel0=%.9g travel1=%.9g signedTravel0=%.9g signedTravel1=%.9g "
      "ratioSamples=%u measuredRatioSamples=%u measuredRatioMean=%.9g "
      "velocityResidualRms=%.9g "
      "velocityResidualMax=%.9g directionSamples=%u directionViolations=%u "
      "tailSamples=%u phaseErrorMax=%.9g tailPhaseErrorMax=%.9g "
      "tailPhaseRms=%.9g phaseDriftSlope=%.9g maxLockedAxisSpeed=%.9g "
      "maxAnchorError=%.9g minAxisDot=%.9g maxCenterDrift=%.9g "
      "maxQuaternionNormError=%.9g maxAbsPosition=%.9g "
      "maxLinearSpeed=%.9g maxAngularSpeed=%.9g unwrapAliasRisk=%u "
      "reverseEvents=%u preReverseAligned=%u postReverseAligned=%u "
      "targetSignChanges=%u positiveMotionSamples=%u negativeMotionSamples=%u "
      "driveEnabledReadback=%u driveTrackingSamples=%u driveAlignedSamples=%u "
      "driveOpposedSamples=%u driveTrackingErrorRms=%.9g "
      "driveTrackingErrorMax=%.9g impulseEvents=%u impulseResponseSamples=%u "
      "impulseDelta0=%.9g impulseDelta1=%.9g impulseResponseMagnitude=%.9g "
      "impulseResponseResidual=%.9g impulseProjectionError=%.9g "
      "impulseExpectedDelta0=%.9g impulseExpectedDelta1=%.9g "
      "impulseActualWorldDelta0=%.9g impulseActualWorldDelta1=%.9g "
      "impulseProjectionDenominator=%.9g impulseRecoverySamples=%u "
      "impulseRecoveryResidualMax=%.9g impulsePreSpeedMax=%.9g "
      "currentDriveTarget=%.9g velocityResidualCap=%.9g phaseErrorCap=%.9g "
      "phaseDriftSlopeCap=%.9g minimumTravel=%.9g anchorCap=%.9g "
      "minimumAxisDot=%.9g lockedAxisSpeedCap=%.9g "
      "driveTrackingErrorCap=%.9g impulseMinimumResponse=%.9g "
      "impulseResidualCap=%.9g impulseBaselineSpeedCap=%.9g "
      "impulseProjectionErrorCap=%.9g "
      "impulseRecoveryVelocityFloor=%.9g impulseResponseWindow=%u "
      "impulseRecoveryWindow=%u serializationRequested=%u "
      "registryCreated=%u collectionCompleted=%u serializable=%u "
      "serializeSuccess=%u serializedBytes=%llu binaryBlockAllocated=%u "
      "binaryAligned=%u deserializeSuccess=%u loadedObjects=%u "
      "loadedActors=%u loadedConstraints=%u loadedRevolute=%u "
      "loadedGear=%u dependencyIdentity=%u actorIdentity=%u "
      "authoringReleased=%u loadedCollectionReleased=%u "
      "binaryBlockFreed=%u pvd=0 "
      "outputForceGate=NOT_COVERED\n",
      getCaseName(gHeadlessCase),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, gHeadlessConfig.reverseFrame,
      gHeadlessConfig.impulseFrame,
      getSerializationModeName(gHeadlessConfig.serializationMode),
      evaluation.status, evaluation.reason,
      gMetrics.nonFinite, physicsErrors, physicsWarnings,
      gMetrics.simulateCalls, gMetrics.fetchCalls, gMetrics.fetchFailures,
      gMetrics.fetchErrorState, gMetrics.cleanupComplete, gMetrics.topologyOk,
      gMetrics.topologyDynamicActors, gMetrics.topologyStaticActors,
      gMetrics.topologyConstraints, gMetrics.topologyShapes0,
      gMetrics.topologyShapes1, gMetrics.shapeRefsCreated,
      gMetrics.shapeRefsReleased, gMetrics.actorsCreated,
      gMetrics.actorsReleased, gMetrics.jointsCreated,
      gMetrics.jointsReleased, double(gHeadlessConfig.ratio),
      double(gMetrics.ratioReadback), double(gMetrics.endpointSign0),
      double(gMetrics.endpointSign1), double(gMetrics.initialAngle0),
      double(gMetrics.initialAngle1), double(gMetrics.baselinePhase),
      double(gMetrics.initialPhaseOffsetWitness), gMetrics.sampleCount,
      double(gMetrics.totalAbsTravel0), double(gMetrics.totalAbsTravel1),
      double(gMetrics.signedTravel0), double(gMetrics.signedTravel1),
      gMetrics.ratioSamples, gMetrics.measuredRatioSamples,
      double(evaluation.measuredRatioMean),
      double(evaluation.velocityResidualRms),
      double(gMetrics.maxVelocityResidualNorm), gMetrics.directionSamples,
      gMetrics.directionViolations, gMetrics.tailSamples,
      double(gMetrics.maxPhaseError), double(gMetrics.maxTailPhaseError),
      double(evaluation.tailPhaseRms), double(evaluation.phaseDriftSlope),
      double(gMetrics.maxLockedAxisSpeed), double(gMetrics.maxAnchorError),
      double(gMetrics.minAxisDot), double(gMetrics.maxCenterDrift),
      double(gMetrics.maxQuaternionNormError),
      double(gMetrics.maxAbsPosition), double(gMetrics.maxLinearSpeed),
      double(gMetrics.maxAngularSpeed), gMetrics.unwrapAliasRisk,
      gMetrics.reverseEvents, gMetrics.preReverseAlignedSamples,
      gMetrics.postReverseAlignedSamples, gMetrics.targetSignChanges,
      gMetrics.positiveMotionSamples, gMetrics.negativeMotionSamples,
      gMetrics.driveEnabledReadback, gMetrics.driveTrackingSamples,
      gMetrics.driveAlignedSamples, gMetrics.driveOpposedSamples,
      double(evaluation.driveTrackingErrorRms),
      double(gMetrics.driveTrackingErrorMax), gMetrics.impulseEvents,
      gMetrics.impulseResponseSamples, double(gMetrics.impulseDelta0),
      double(gMetrics.impulseDelta1),
      double(gMetrics.impulseResponseMagnitude),
      double(gMetrics.impulseResponseResidualNorm),
      double(gMetrics.impulseProjectionErrorNorm),
      double(gMetrics.impulseExpectedDelta0),
      double(gMetrics.impulseExpectedDelta1),
      double(gMetrics.impulseActualWorldDelta0),
      double(gMetrics.impulseActualWorldDelta1),
      double(gMetrics.impulseProjectionDenominator),
      gMetrics.impulseRecoverySamples,
      double(gMetrics.impulseRecoveryResidualMax),
      double(gMetrics.impulsePreSpeedMax), double(gCurrentDriveTarget),
      double(gVelocityResidualCap), double(gPhaseErrorCap),
      double(gPhaseDriftSlopeCap), double(gMinimumTravel),
      double(gAnchorErrorCap), double(gMinimumAxisDot),
      double(gLockedAxisSpeedCap), double(gDriveTrackingErrorCap),
      double(gImpulseMinimumResponse), double(gImpulseResidualCap),
      double(gImpulseBaselineSpeedCap),
      double(gImpulseProjectionErrorCap),
      double(gImpulseRecoveryVelocityFloor), gImpulseResponseWindow,
      gImpulseRecoveryWindow, gMetrics.serializationRequested,
      gMetrics.registryCreated, gMetrics.collectionCompleted,
      gMetrics.serializable, gMetrics.serializeSuccess,
      static_cast<unsigned long long>(gMetrics.serializedBytes),
      gMetrics.binaryBlockAllocated, gMetrics.binaryAligned,
      gMetrics.deserializeSuccess, gMetrics.loadedObjects,
      gMetrics.loadedActors, gMetrics.loadedConstraints,
      gMetrics.loadedRevolute, gMetrics.loadedGear,
      gMetrics.dependencyIdentity, gMetrics.actorIdentity,
      gMetrics.authoringReleased, gMetrics.loadedCollectionReleased,
      gMetrics.binaryBlockFreed);
}

static int reportConfigurationError(const Snippets::HeadlessOptions &options,
                                    const GearHeadlessConfig &config,
                                    const char *reason) {
  const std::string reasonToken = makeAuthorityToken(reason);
  const std::string requestedCaseToken =
      makeAuthorityToken(options.caseName.c_str());
  std::printf(
      "[AVBD_GATE] schema=1 snippet=SnippetGearJoint case=config-error "
      "requestedCase=%s "
      "solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
      "seed=%u dispatcherThreads=%u capability=PARTIAL validation=PROBE "
      "status=ERROR reason=%s thresholdPolicy=DIAGNOSTIC ratio=%.9g "
      "reverseFrame=%u impulseFrame=%u serialization=%s "
      "nonFinite=0 physicsErrors=0 physicsWarnings=0 simulateCalls=0 "
      "fetchCalls=0 fetchFailures=0 cleanupComplete=1 "
      "outputForceGate=NOT_COVERED\n",
      requestedCaseToken.c_str(),
      Snippets::getSolverTypeName(options.solverType),
      Snippets::getExecutionName(options.execution), options.frames,
      double(options.dt), options.seed, options.dispatcherThreads,
      reasonToken.c_str(), double(config.ratio), config.reverseFrame,
      config.impulseFrame, getSerializationModeName(config.serializationMode));
  return Snippets::eHEADLESS_CONFIG_ERROR;
}

static void recoverCaseHintForConfigurationError(
    int argc, const char *const *argv, Snippets::HeadlessOptions &options) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    const bool caseOption = Snippets::hasOptionPrefix(arg, "--case=");
    const bool scenarioOption =
        Snippets::hasOptionPrefix(arg, "--scenario=");
    if (!caseOption && !scenarioOption)
      continue;
    options.caseName =
        arg + std::strlen(caseOption ? "--case=" : "--scenario=");
    return;
  }
}

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  Snippets::HeadlessOptions defaults;
  defaults.caseName = "steady";
  defaults.frames = 1200;
  defaults.seed = 1;
  defaults.dispatcherThreads = 2;
  defaults.dt = 1.0f / 60.0f;

  Snippets::HeadlessOptions options;
  GearHeadlessConfig config;
  std::string parseError;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
                                            parseError)) {
    recoverCaseHintForConfigurationError(argc, argv, options);
    return reportConfigurationError(options, config, parseError.c_str());
  }

  GearOptionSeen seen;
  bool headlessOnlyOptionSeen = false;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!arg)
      continue;
    if (Snippets::isCommonHeadlessOption(arg)) {
      if (std::strcmp(arg, "--headless") != 0 &&
          !Snippets::hasOptionPrefix(arg, "--solver="))
        headlessOnlyOptionSeen = true;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--ratio=")) {
      if (seen.ratio)
        return reportConfigurationError(options, config, "duplicate_--ratio");
      seen.ratio = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseReal(arg + std::strlen("--ratio="), -2.5f, 2.5f,
                               config.ratio) ||
          PxAbs(PxAbs(config.ratio) - 2.5f) > 1e-6f)
        return reportConfigurationError(options, config,
                                        "ratio_requires_plus_or_minus_2.5");
      config.ratio = config.ratio < 0.0f ? -2.5f : 2.5f;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--reverse-frame=")) {
      if (seen.reverseFrame)
        return reportConfigurationError(options, config,
                                        "duplicate_--reverse-frame");
      seen.reverseFrame = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseU32(arg + std::strlen("--reverse-frame="), 1,
                              100000000u, config.reverseFrame))
        return reportConfigurationError(options, config,
                                        "invalid_--reverse-frame_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--impulse-frame=")) {
      if (seen.impulseFrame)
        return reportConfigurationError(options, config,
                                        "duplicate_--impulse-frame");
      seen.impulseFrame = true;
      headlessOnlyOptionSeen = true;
      if (!Snippets::parseU32(arg + std::strlen("--impulse-frame="), 1,
                              100000000u, config.impulseFrame))
        return reportConfigurationError(options, config,
                                        "invalid_--impulse-frame_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--serialization=")) {
      if (seen.serialization)
        return reportConfigurationError(options, config,
                                        "duplicate_--serialization");
      seen.serialization = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseSerializationMode(
              arg + std::strlen("--serialization="),
              config.serializationMode))
        return reportConfigurationError(options, config,
                                        "invalid_--serialization_value");
      continue;
    }
    return reportConfigurationError(options, config, "unknown_argument");
  }

#ifndef RENDER_SNIPPET
  options.headless = true;
#endif

  GearHeadlessCase testCase = eCASE_STEADY;
  if (!tryParseCase(options.caseName.c_str(), testCase))
    return reportConfigurationError(options, config, "invalid_--case_value");
  options.caseName = getCaseName(testCase);
  if (!options.headless && headlessOnlyOptionSeen)
    return reportConfigurationError(options, config,
                                    "gate_option_requires_--headless");
  if (options.headless && options.solverType == PxSolverType::ePGS)
    return reportConfigurationError(options, config,
                                    "headless_solver_requires_avbd_or_tgs");
  if (options.frames < 600)
    return reportConfigurationError(options, config,
                                     "frames_must_be_at_least_600");
  if (testCase == eCASE_SINUSOIDAL && options.frames < 1200)
    return reportConfigurationError(
        options, config, "sinusoidal_frames_must_be_at_least_1200");
  if (PxAbs(options.dt - (1.0f / 60.0f)) > 1e-7f)
    return reportConfigurationError(options, config,
                                    "dt_requires_60hz_calibration");
  if (options.execution == Snippets::eHEADLESS_SEQUENTIAL &&
      options.solverType != PxSolverType::eAVBD)
    return reportConfigurationError(options, config,
                                    "sequential_requires_avbd");

  if (testCase == eCASE_STEADY) {
    if (seen.reverseFrame || seen.impulseFrame)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  } else {
    if (seen.ratio)
      return reportConfigurationError(options, config,
                                      "ratio_only_valid_for_steady");
    config.ratio = testCase == eCASE_UNIT_RATIO ? 1.0f : 2.5f;
  }
  if (seen.reverseFrame && testCase != eCASE_REVERSE)
    return reportConfigurationError(options, config,
                                    "reverse_frame_requires_reverse_case");
  if (seen.impulseFrame && testCase != eCASE_EXTERNAL_IMPULSE)
    return reportConfigurationError(
        options, config, "impulse_frame_requires_external_impulse_case");
  if (testCase == eCASE_REVERSE &&
      (config.reverseFrame < gWarmupFrames + gEventRecoveryFrames ||
       config.reverseFrame > options.frames - 240u))
    return reportConfigurationError(options, config,
                                    "reverse_frame_requires_recovery_window");
  if (testCase == eCASE_EXTERNAL_IMPULSE &&
      (config.impulseFrame < gWarmupFrames + gEventRecoveryFrames ||
       config.impulseFrame > options.frames - 240u))
    return reportConfigurationError(options, config,
                                    "impulse_frame_requires_recovery_window");
  if (!Snippets::applyExecutionEnvironment(options))
    return reportConfigurationError(options, config,
                                    "execution_environment_failed");

  gHeadlessOptions = options;
  gHeadlessConfig = config;
  gHeadlessCase = testCase;
  gSolverType = options.solverType;

#ifdef RENDER_SNIPPET
  if (!options.headless) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  Snippets::printHeadlessConfig("SnippetGearJoint", gHeadlessOptions);
  initPhysics(false);
  for (PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame) {
    PX_UNUSED(frame);
    if (gInitializationFailed || gMetrics.fetchFailures ||
        gMetrics.fetchErrorState)
      break;
    stepPhysics(false);
  }

  cleanupPhysics(false);
  const PxU32 physicsErrors = gErrorCallback.getFatalCount();
  const PxU32 physicsWarnings = gErrorCallback.getWarningCount();
  const GearGateEvaluation evaluation = evaluateGate(physicsErrors);
  printGateResult(evaluation, physicsErrors, physicsWarnings);
  return static_cast<int>(evaluation.exitCode);
}
