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
// This snippet illustrates simple use of joint drives in PhysX.
// ****************************************************************************

#include <cfloat>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetPrint.h"
#ifdef RENDER_SNIPPET
#include "../snippetrender/SnippetRender.h"
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
#if PX_SUPPORT_GPU_PHYSX
static PxCudaContextManager *gCudaContextManager = NULL;
#endif

static bool gPause = false;
static bool gOneFrame = false;
static bool gChangeObjectAType = false; // false=static, true=kinematic
static bool gChangeObjectBRotation = false;
static bool gChangeJointFrameARotation = false;
static bool gChangeJointFrameBRotation = false;
#if PX_SUPPORT_GPU_PHYSX
static bool gUseGPU = false;
#endif
static PxU32 gSceneIndex = 0;
static const PxU32 gMaxSceneIndex = 4;

enum JointDriveKind {
  eDRIVE_LINEAR_X,
  eDRIVE_LINEAR_Y,
  eDRIVE_LINEAR_Z,
  eDRIVE_TWIST,
  eDRIVE_SWING1,
  eDRIVE_SWING2,
  eDRIVE_SLERP
};

enum JointDriveCase {
  eCASE_VELOCITY,
  eCASE_VELOCITY_ORDERING,
  eCASE_ANGULAR_ORDERING,
  eCASE_ANGULAR_OUTPUT_FORCE,
  eCASE_POSITION,
  eCASE_ANGULAR_POSITION,
  eCASE_OUTPUT_FORCE,
  eCASE_MASS_SCALING,
  eCASE_ACCELERATION_MODE,
  eCASE_FORCE_LIMIT,
  eCASE_LEGACY_ANGULAR_LIMIT_CONE_OUTSIDE,
  eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE
};

enum JointDriveMode { eDRIVE_MODE_FORCE, eDRIVE_MODE_ACCELERATION };

enum JointDriveEndpoint { eENDPOINT_FORWARD, eENDPOINT_REVERSE };

enum JointDriveBreakMode {
  eBREAK_UNBREAKABLE,
  eBREAK_BELOW_DRIVE_LIMIT,
  eBREAK_ABOVE_DRIVE_LIMIT,
  eBREAK_BELOW_OFFSET_MOMENT,
  eBREAK_ABOVE_OFFSET_MOMENT
};

enum JointDriveTopology {
  eTOPOLOGY_STATIC_DYNAMIC,
  eTOPOLOGY_DYNAMIC_DYNAMIC,
  eTOPOLOGY_CONTACT_DYNAMIC_DYNAMIC
};

enum JointFrameOrientation {
  eFRAME_IDENTITY,
  eFRAME_ROTZ_NEG45,
  eFRAME_ROTX_NEG45
};

enum JointKinematicMotion {
  eKINEMATIC_STATIONARY,
  eKINEMATIC_SPIN_WORLD_Y
};

struct JointDriveHeadlessConfig {
  JointDriveKind drive;
  JointDriveMode driveMode;
  bool actorAKinematic;
  JointFrameOrientation frameAOrientation;
  JointFrameOrientation frameBOrientation;
  bool bodyBRotated;
  bool initialRelativeOffset;
  bool offsetAnchor;
  PxReal comparisonMass;
  bool lowForceLimit;
  bool outputForceEnabled;
  JointDriveBreakMode breakMode;
  JointDriveEndpoint endpoint;
  JointDriveTopology topology;
  JointKinematicMotion kinematicMotion;

  JointDriveHeadlessConfig()
      : drive(eDRIVE_LINEAR_X), driveMode(eDRIVE_MODE_FORCE),
        actorAKinematic(false), frameAOrientation(eFRAME_IDENTITY),
        frameBOrientation(eFRAME_IDENTITY),
        bodyBRotated(false), initialRelativeOffset(false), offsetAnchor(false),
        comparisonMass(10.0f), lowForceLimit(true), outputForceEnabled(false),
        breakMode(eBREAK_UNBREAKABLE),
        endpoint(eENDPOINT_FORWARD), topology(eTOPOLOGY_STATIC_DYNAMIC),
        kinematicMotion(eKINEMATIC_STATIONARY) {}
};

struct JointDriveOptionSeen {
  bool drive;
  bool driveMode;
  bool actorA;
  bool frameA;
  bool frameB;
  bool bodyB;
  bool initialRelative;
  bool anchor;
  bool mass;
  bool limit;
  bool outputForce;
  bool breakMode;
  bool endpoint;
  bool topology;
  bool kinematicMotion;

  JointDriveOptionSeen()
      : drive(false), driveMode(false), actorA(false), frameA(false),
        frameB(false), bodyB(false), initialRelative(false), anchor(false), mass(false),
        limit(false), outputForce(false), breakMode(false), endpoint(false),
        topology(false), kinematicMotion(false) {}
};

struct DrivePairRuntime {
  PxD6Joint *joint;
  PxRigidActor *actorA;
  PxRigidDynamic *dynamicActorA;
  PxRigidDynamic *actorB;
  PxTransform jointFrameA;
  PxTransform jointFrameB;
  PxTransform initialPoseA;
  PxTransform initialPoseB;
  PxTransform expectedPoseA;
  PxVec3 signedWorldAxis;

  DrivePairRuntime()
      : joint(NULL), actorA(NULL), dynamicActorA(NULL), actorB(NULL),
        jointFrameA(PxIdentity), jointFrameB(PxIdentity),
        initialPoseA(PxIdentity), initialPoseB(PxIdentity),
        expectedPoseA(PxIdentity),
        signedWorldAxis(1.0f, 0.0f, 0.0f) {}
};

struct OrderingDriveRuntime {
  PxD6Joint *joint;
  PxRigidDynamic *dynamicActor;
  PxTransform localFrameA;
  PxTransform localFrameB;
  PxTransform initialDynamicPose;
  PxTransform previousDynamicPose;
  PxQuat actorAWorldFrameRotation;
  PxVec3 expectedAxis;
  PxVec3 expectedDynamicAxis;
  PxVec3 expectedFrameAAxis;
  PxVec3 expectedFrameBAxis;
  PxVec3 expectedDynamicFrameAxis;
  PxVec3 expectedDynamicLocalAxis;
  PxVec3 expectedActor0Torque;
  bool angular;

  OrderingDriveRuntime()
      : joint(NULL), dynamicActor(NULL), localFrameA(PxIdentity),
        localFrameB(PxIdentity), initialDynamicPose(PxIdentity),
        previousDynamicPose(PxIdentity), actorAWorldFrameRotation(PxIdentity),
        expectedAxis(1.0f, 0.0f, 0.0f),
        expectedDynamicAxis(1.0f, 0.0f, 0.0f),
        expectedFrameAAxis(1.0f, 0.0f, 0.0f),
        expectedFrameBAxis(1.0f, 0.0f, 0.0f),
        expectedDynamicFrameAxis(1.0f, 0.0f, 0.0f),
        expectedDynamicLocalAxis(1.0f, 0.0f, 0.0f),
        expectedActor0Torque(0.0f), angular(false) {}
};

struct OrderingDriveMetrics {
  PxTransform expectedWorldFrame;
  PxTransform expectedDynamicLocalFrame;
  PxVec3 expectedAxis;
  PxVec3 expectedDynamicAxis;
  PxVec3 expectedFrameAAxis;
  PxVec3 expectedFrameBAxis;
  PxVec3 expectedDynamicFrameAxis;
  PxVec3 actor0Axis;
  PxVec3 actor1Axis;
  PxVec3 dynamicLocalAxis;
  PxVec3 dynamicWorldAxis;
  PxVec3 gravity;
  PxVec3 inertiaReadback;
  PxReal massReadback;
  PxReal linearDampingReadback;
  PxReal angularDampingReadback;
  PxReal worldFramePositionError;
  PxReal dynamicLocalPositionError;
  PxReal worldFrameRotationDot;
  PxReal dynamicLocalRotationDot;
  PxReal actor0AxisDot;
  PxReal actor1AxisDot;
  PxReal dynamicLocalAxisDot;
  PxReal dynamicWorldAxisDot;
  PxReal frameAxisSeparationDot;
  PxReal expectedFrameAxisSeparationDot;
  PxReal bodyRotationDot;
  PxReal driveStiffnessReadback;
  PxReal driveDampingReadback;
  PxReal driveForceLimitReadback;
  PxReal driveLinearTargetError;
  PxReal driveAngularTargetError;
  PxReal firstRelativeProjection;
  PxReal firstDynamicProjection;
  PxReal firstDynamicAcceleration;
  PxReal previousDynamicProjection;
  PxReal firstDrivenDynamicAcceleration;
  PxReal maximumInitialDynamicAcceleration;
  PxVec3 firstPublicForce;
  PxVec3 firstPublicTorque;
  PxVec3 firstActor0FrameTorque;
  PxVec3 expectedActor0Torque;
  PxReal firstSignedPublicTorque;
  PxReal maximumPublicForce;
  PxReal maximumPublicTorque;
  PxReal linearBreakForceReadback;
  PxReal angularBreakForceReadback;
  PxReal relativeLateProjectionSum;
  PxReal relativeLateTargetErrorSquaredSum;
  PxReal relativeLateOrthogonalSquaredSum;
  PxReal dynamicLateProjectionSum;
  PxReal dynamicLateTargetErrorSquaredSum;
  PxReal dynamicLateOrthogonalSquaredSum;
  PxReal minLateRelativeProjection;
  PxReal minLateRelativeDirectionDot;
  PxReal maxLateRelativeOrthogonal;
  PxReal minLateDynamicProjection;
  PxReal minLateDynamicDirectionDot;
  PxReal maxLateDynamicOrthogonal;
  PxReal finalRelativeProjection;
  PxReal finalRelativeDirectionDot;
  PxReal finalRelativeOrthogonal;
  PxReal finalDynamicProjection;
  PxReal finalDynamicDirectionDot;
  PxReal finalDynamicOrthogonal;
  PxReal finalRelativeDisplacement;
  PxReal finalDynamicDisplacement;
  PxReal maxDynamicRotationError;
  PxReal maxDynamicAngularSpeed;
  PxReal maxDynamicPositionError;
  PxReal maxQuaternionNormError;
  PxReal maxAbsPosition;
  PxReal maxLinearSpeed;
  PxU32 shapeCount;
  PxU32 freeMotionCount;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxU32 completedFrames;
  PxU32 fetchFailures;
  PxU32 fetchErrorState;
  PxU32 sampleAttempts;
  PxU32 sampleCount;
  PxU32 nonFiniteSamples;
  PxU32 sampleErrors;
  PxU32 lateSampleAttempts;
  PxU32 lateSampleCount;
  PxU32 lateNonFiniteSamples;
  PxU32 publicForceSampleAttempts;
  PxU32 publicForceSamples;
  PxU32 nonFinitePublicForceSamples;
  PxU32 brokenSamples;
  PxU32 firstBrokenFrame;
  bool actorOrderValid;
  bool angular;
  bool frameWitnessValid;
  bool driveReadbackValid;
  bool angularDriveConfigValid;
  bool driveLimitsAreForcesReadback;
  bool outputForceFlagReadback;
  bool fixtureWitnessValid;
  bool cleanupComplete;
  bool nonFinite;

  OrderingDriveMetrics()
      : expectedWorldFrame(PxIdentity), expectedDynamicLocalFrame(PxIdentity),
        expectedAxis(1.0f, 0.0f, 0.0f),
        expectedDynamicAxis(1.0f, 0.0f, 0.0f),
        expectedFrameAAxis(1.0f, 0.0f, 0.0f),
        expectedFrameBAxis(1.0f, 0.0f, 0.0f),
        expectedDynamicFrameAxis(1.0f, 0.0f, 0.0f), actor0Axis(0.0f),
        actor1Axis(0.0f), dynamicLocalAxis(0.0f), dynamicWorldAxis(0.0f),
        gravity(0.0f), inertiaReadback(0.0f), massReadback(0.0f),
        linearDampingReadback(0.0f), angularDampingReadback(0.0f),
        worldFramePositionError(PX_MAX_F32),
        dynamicLocalPositionError(PX_MAX_F32), worldFrameRotationDot(0.0f),
        dynamicLocalRotationDot(0.0f), actor0AxisDot(-1.0f),
        actor1AxisDot(-1.0f), dynamicLocalAxisDot(-1.0f),
        dynamicWorldAxisDot(-1.0f), frameAxisSeparationDot(1.0f),
        expectedFrameAxisSeparationDot(1.0f), bodyRotationDot(0.0f),
        driveStiffnessReadback(PX_MAX_F32),
        driveDampingReadback(PX_MAX_F32),
        driveForceLimitReadback(PX_MAX_F32),
        driveLinearTargetError(PX_MAX_F32),
        driveAngularTargetError(PX_MAX_F32),
        firstRelativeProjection(0.0f), firstDynamicProjection(0.0f),
        firstDynamicAcceleration(0.0f), previousDynamicProjection(0.0f),
        firstDrivenDynamicAcceleration(0.0f),
        maximumInitialDynamicAcceleration(0.0f),
        firstPublicForce(0.0f), firstPublicTorque(0.0f),
        firstActor0FrameTorque(0.0f), expectedActor0Torque(0.0f),
        firstSignedPublicTorque(0.0f), maximumPublicForce(0.0f),
        maximumPublicTorque(0.0f), linearBreakForceReadback(PX_MAX_F32),
        angularBreakForceReadback(PX_MAX_F32),
        relativeLateProjectionSum(0.0f),
        relativeLateTargetErrorSquaredSum(0.0f),
        relativeLateOrthogonalSquaredSum(0.0f),
        dynamicLateProjectionSum(0.0f),
        dynamicLateTargetErrorSquaredSum(0.0f),
        dynamicLateOrthogonalSquaredSum(0.0f),
        minLateRelativeProjection(PX_MAX_F32),
        minLateRelativeDirectionDot(1.0f),
        maxLateRelativeOrthogonal(0.0f),
        minLateDynamicProjection(PX_MAX_F32),
        minLateDynamicDirectionDot(1.0f),
        maxLateDynamicOrthogonal(0.0f), finalRelativeProjection(0.0f),
        finalRelativeDirectionDot(-1.0f), finalRelativeOrthogonal(0.0f),
        finalDynamicProjection(0.0f), finalDynamicDirectionDot(-1.0f),
        finalDynamicOrthogonal(0.0f), finalRelativeDisplacement(0.0f),
        finalDynamicDisplacement(0.0f), maxDynamicRotationError(0.0f),
        maxDynamicAngularSpeed(0.0f), maxDynamicPositionError(0.0f),
        maxQuaternionNormError(0.0f),
        maxAbsPosition(0.0f), maxLinearSpeed(0.0f), shapeCount(PX_MAX_U32),
        freeMotionCount(0), initialDynamicActors(PX_MAX_U32),
        initialStaticActors(PX_MAX_U32), initialConstraints(PX_MAX_U32),
        finalDynamicActors(PX_MAX_U32), finalStaticActors(PX_MAX_U32),
        finalConstraints(PX_MAX_U32), cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32), cleanupConstraints(PX_MAX_U32),
        completedFrames(0), fetchFailures(0), fetchErrorState(0),
        sampleAttempts(0), sampleCount(0), nonFiniteSamples(0),
        sampleErrors(0), lateSampleAttempts(0), lateSampleCount(0),
        lateNonFiniteSamples(0), publicForceSampleAttempts(0),
        publicForceSamples(0), nonFinitePublicForceSamples(0),
        brokenSamples(0), firstBrokenFrame(PX_MAX_U32),
        actorOrderValid(false), angular(false),
        frameWitnessValid(false), driveReadbackValid(false),
        angularDriveConfigValid(false),
        driveLimitsAreForcesReadback(false), outputForceFlagReadback(false),
        fixtureWitnessValid(false), cleanupComplete(false), nonFinite(false) {}
};

struct JointDriveMetrics {
  PxTransform initialPoseA;
  PxTransform initialPoseB;
  PxTransform previousPoseA;
  PxTransform previousPoseB;
  PxVec3 signedWorldAxis;
  PxReal targetMagnitude;
  PxReal maxQuaternionNormError;
  PxReal maxAbsPosition;
  PxReal maxLinearSpeed;
  PxReal maxAngularSpeed;
  PxReal maxActorAPositionError;
  PxReal maxActorAAngleError;
  PxReal signedTravel;
  PxReal finalSignedDisplacement;
  PxReal finalSignedProjection;
  PxReal finalAxisDot;
  PxReal finalOrthogonalSpeed;
  PxReal lateProjectionSum;
  PxReal lateProjectionErrorSquaredSum;
  PxReal lateOrthogonalSquaredSum;
  PxReal minLateAxisDot;
  PxU32 sampleCount;
  PxU32 lateSampleCount;
  PxU32 completedFrames;
  PxU32 fetchFailures;
  PxU32 fetchErrorState;
  PxU32 pairCountWitness;
  bool nonFinite;

  JointDriveMetrics()
      : initialPoseA(PxIdentity), initialPoseB(PxIdentity),
        previousPoseA(PxIdentity), previousPoseB(PxIdentity),
        signedWorldAxis(1.0f, 0.0f, 0.0f), targetMagnitude(1.0f),
        maxQuaternionNormError(0.0f), maxAbsPosition(0.0f),
        maxLinearSpeed(0.0f), maxAngularSpeed(0.0f),
        maxActorAPositionError(0.0f), maxActorAAngleError(0.0f),
        signedTravel(0.0f), finalSignedDisplacement(0.0f),
        finalSignedProjection(0.0f), finalAxisDot(-1.0f),
        finalOrthogonalSpeed(0.0f), lateProjectionSum(0.0f),
        lateProjectionErrorSquaredSum(0.0f),
        lateOrthogonalSquaredSum(0.0f), minLateAxisDot(1.0f),
        sampleCount(0), lateSampleCount(0), completedFrames(0),
        fetchFailures(0), fetchErrorState(0), pairCountWitness(0),
        nonFinite(false) {}
};

struct JointDriveGateEvaluation {
  Snippets::HeadlessExitCode exitCode;
  const char *status;
  const char *reason;
  PxReal lateTargetMean;
  PxReal lateTargetRms;
  PxReal lateOrthogonalRms;
  PxReal motionWitness;
  PxReal positionLateErrorRms;
  PxReal positionLateErrorRatio;
  PxReal positionLateSpeedRms;
  PxReal positionMotionRatio;
  PxReal testToReferenceResponseRatio;
  PxReal referenceToTestRateRatio;
  PxReal expectedTestDeltaVelocity;
  PxReal normalizedImpulse;
  PxReal meanTestAcceleration;

  JointDriveGateEvaluation()
      : exitCode(Snippets::eHEADLESS_PASS), status("PASS"), reason("none"),
        lateTargetMean(0.0f), lateTargetRms(0.0f),
        lateOrthogonalRms(0.0f), motionWitness(0.0f),
        positionLateErrorRms(0.0f), positionLateErrorRatio(PX_MAX_F32),
        positionLateSpeedRms(0.0f), positionMotionRatio(0.0f),
        testToReferenceResponseRatio(PX_MAX_F32),
        referenceToTestRateRatio(PX_MAX_F32),
        expectedTestDeltaVelocity(0.0f), normalizedImpulse(0.0f),
        meanTestAcceleration(0.0f) {}
};

struct JointDrivePositionMetrics {
  PxTransform initialRelativePose;
  PxTransform targetRelativePose;
  PxVec3 previousRelativeVelocity;
  PxVec3 initialCenterOfMass;
  bool actorOrderValid;
  bool angularFrameWitnessValid;
  bool driveLimitsAreForcesReadback;
  bool outputForceFlagReadback;
  PxReal actorAMassReadback;
  PxVec3 actorAInertiaReadback;
  PxReal massReadback;
  PxVec3 inertiaReadback;
  PxReal stiffnessReadback;
  PxReal dampingReadback;
  PxReal forceLimitReadback;
  PxReal worldFrameAxisDot;
  PxReal wrongRawFrameAxisDot;
  PxReal targetReadbackError;
  PxReal initialTargetError;
  PxReal initialRelativeMagnitude;
  PxReal initialRelativeSetupError;
  PxReal targetRelativeMagnitude;
  PxReal finalTargetError;
  PxReal finalErrorRatio;
  PxReal lateErrorSquaredSum;
  PxReal lateSpeedSquaredSum;
  PxReal maximumSignedProgress;
  PxReal minimumSignedProgress;
  PxReal maximumOrthogonalError;
  PxReal maximumOvershoot;
  PxReal firstRelativeAcceleration;
  PxReal maximumRelativeAcceleration;
  PxReal expectedFirstRelativeAcceleration;
  PxReal expectedSignedAngularAccelerationA;
  PxReal expectedSignedAngularAccelerationB;
  PxReal firstSignedAngularAccelerationA;
  PxReal firstSignedAngularAccelerationB;
  PxReal maximumCenterOfMassDrift;
  PxReal maximumLinearMomentum;
  PxReal maximumAngularMomentum;
  PxVec3 firstPublicForce;
  PxVec3 firstPublicTorque;
  PxVec3 actor0WorldArm;
  PxVec3 dynamicWorldArm;
  PxVec3 expectedNormalizedPublicTorque;
  PxVec3 firstNormalizedPublicTorque;
  PxReal firstSignedPublicForce;
  PxReal maximumPublicForce;
  PxReal maximumPublicTorque;
  PxU32 publicForceSampleAttempts;
  PxU32 publicForceSamples;
  PxU32 nonFinitePublicForceSamples;
  PxReal linearBreakForceReadback;
  PxReal angularBreakForceReadback;
  PxU32 brokenSamples;
  PxU32 firstBrokenFrame;
  PxU32 lateSampleCount;
  PxU32 awakeSamples;
  PxU32 kinematicTargetFrames;
  PxU32 kinematicMotionFrames;
  PxReal finalKinematicTravel;
  PxReal maximumKinematicAngularSpeedError;

  JointDrivePositionMetrics()
      : initialRelativePose(PxIdentity), targetRelativePose(PxIdentity),
        previousRelativeVelocity(0.0f), initialCenterOfMass(0.0f),
        actorOrderValid(false),
        angularFrameWitnessValid(false),
        driveLimitsAreForcesReadback(false), outputForceFlagReadback(false),
        actorAMassReadback(0.0f), actorAInertiaReadback(0.0f),
        massReadback(0.0f),
        inertiaReadback(0.0f), stiffnessReadback(0.0f),
        dampingReadback(0.0f), forceLimitReadback(0.0f),
        worldFrameAxisDot(0.0f), wrongRawFrameAxisDot(1.0f),
        targetReadbackError(0.0f),
        initialTargetError(0.0f),
        initialRelativeMagnitude(0.0f), initialRelativeSetupError(0.0f),
        targetRelativeMagnitude(0.0f), finalTargetError(0.0f),
        finalErrorRatio(PX_MAX_F32),
        lateErrorSquaredSum(0.0f), lateSpeedSquaredSum(0.0f),
        maximumSignedProgress(0.0f), minimumSignedProgress(0.0f),
        maximumOrthogonalError(0.0f), maximumOvershoot(0.0f),
        firstRelativeAcceleration(0.0f), maximumRelativeAcceleration(0.0f),
        expectedFirstRelativeAcceleration(0.0f),
        expectedSignedAngularAccelerationA(0.0f),
        expectedSignedAngularAccelerationB(0.0f),
        firstSignedAngularAccelerationA(0.0f),
        firstSignedAngularAccelerationB(0.0f),
        maximumCenterOfMassDrift(0.0f), maximumLinearMomentum(0.0f),
        maximumAngularMomentum(0.0f),
        firstPublicForce(0.0f), firstPublicTorque(0.0f),
        actor0WorldArm(0.0f), dynamicWorldArm(0.0f),
        expectedNormalizedPublicTorque(0.0f),
        firstNormalizedPublicTorque(0.0f),
        firstSignedPublicForce(0.0f), maximumPublicForce(0.0f),
        maximumPublicTorque(0.0f), publicForceSampleAttempts(0),
        publicForceSamples(0), nonFinitePublicForceSamples(0),
        linearBreakForceReadback(PX_MAX_F32),
        angularBreakForceReadback(PX_MAX_F32), brokenSamples(0),
        firstBrokenFrame(PX_MAX_U32),
        lateSampleCount(0), awakeSamples(0), kinematicTargetFrames(0),
        kinematicMotionFrames(0), finalKinematicTravel(0.0f),
        maximumKinematicAngularSpeedError(0.0f) {}
};

struct JointDriveComparisonLaneMetrics {
  PxVec3 finalRelativeVelocity;
  PxVec3 maximumOrthogonalVelocity;
  PxReal previousProjection;
  PxReal firstProjection;
  PxReal frameFourProjection;
  PxReal transientProjection;
  PxReal response;
  PxReal decayRate;
  PxReal maximumOrthogonalSpeed;
  PxReal minimumAxisDot;
  PxReal peakAcceleration;
  PxReal maximumMonotonicDrop;
  PxU32 transientSamples;
  PxU32 monotonicViolations;
  PxU32 overshootCount;

  JointDriveComparisonLaneMetrics()
      : finalRelativeVelocity(0.0f), maximumOrthogonalVelocity(0.0f),
        previousProjection(0.0f), firstProjection(0.0f),
        frameFourProjection(0.0f),
        transientProjection(0.0f), response(0.0f), decayRate(0.0f),
        maximumOrthogonalSpeed(0.0f), minimumAxisDot(1.0f),
        peakAcceleration(0.0f), maximumMonotonicDrop(0.0f),
        transientSamples(0),
        monotonicViolations(0), overshootCount(0) {}
};

struct JointDriveComparisonMetrics {
  JointDriveComparisonLaneMetrics lanes[2];
  PxReal actorAMassReadback[2];
  PxVec3 actorAInertiaReadback[2];
  PxReal massReadback[2];
  PxVec3 inertiaReadback[2];
  PxReal dampingReadback[2];
  PxReal forceLimitReadback[2];
  bool accelerationFlagReadback[2];
  bool driveLimitsAreForcesReadback[2];
  bool finiteReadback[2];
  PxVec3 initialCenterOfMass[2];
  PxReal maximumCenterOfMassDrift[2];
  PxReal maximumMomentumMagnitude[2];
  PxReal minimumBottom[2];
  PxReal maximumAbsVerticalSpeed[2];
  PxU32 bodyContactFrames[2][2];
  PxU32 bothBodyContactFrames[2];
  PxU32 contactPointCount[2];
  PxU32 currentContactMask[2];

  JointDriveComparisonMetrics() {
    for (PxU32 i = 0; i < 2; ++i) {
      actorAMassReadback[i] = 0.0f;
      actorAInertiaReadback[i] = PxVec3(0.0f);
      massReadback[i] = 0.0f;
      inertiaReadback[i] = PxVec3(0.0f);
      dampingReadback[i] = 0.0f;
      forceLimitReadback[i] = 0.0f;
      accelerationFlagReadback[i] = false;
      driveLimitsAreForcesReadback[i] = false;
      finiteReadback[i] = false;
      initialCenterOfMass[i] = PxVec3(0.0f);
      maximumCenterOfMassDrift[i] = 0.0f;
      maximumMomentumMagnitude[i] = 0.0f;
      minimumBottom[i] = PX_MAX_F32;
      maximumAbsVerticalSpeed[i] = 0.0f;
      bodyContactFrames[i][0] = 0;
      bodyContactFrames[i][1] = 0;
      bothBodyContactFrames[i] = 0;
      contactPointCount[i] = 0;
      currentContactMask[i] = 0;
    }
  }
};

struct JointDriveAngularLimitMetrics {
  PxReal limitYReadback;
  PxReal limitZReadback;
  PxD6Motion::Enum twistMotionReadback;
  PxD6Motion::Enum swing1MotionReadback;
  PxD6Motion::Enum swing2MotionReadback;
  PxReal initialConeAngle;
  PxReal finalConeAngle;
  PxReal minimumConeAngle;
  PxReal maximumConeAngle;
  PxReal maximumLateConeAngle;
  PxReal maximumInsideDeviation;
  PxReal initialEllipseRadius;
  PxReal finalEllipseRadius;
  PxReal maximumLateEllipseRadius;
  PxReal maximumInsideEllipseDeviation;
  PxU32 lateSampleCount;

  JointDriveAngularLimitMetrics()
      : limitYReadback(PX_MAX_F32), limitZReadback(PX_MAX_F32),
        twistMotionReadback(PxD6Motion::eFREE),
        swing1MotionReadback(PxD6Motion::eFREE),
        swing2MotionReadback(PxD6Motion::eFREE),
        initialConeAngle(PX_MAX_F32), finalConeAngle(PX_MAX_F32),
        minimumConeAngle(PX_MAX_F32), maximumConeAngle(0.0f),
        maximumLateConeAngle(0.0f), maximumInsideDeviation(0.0f),
        initialEllipseRadius(PX_MAX_F32),
        finalEllipseRadius(PX_MAX_F32),
        maximumLateEllipseRadius(0.0f),
        maximumInsideEllipseDeviation(0.0f),
        lateSampleCount(0) {}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static JointDriveHeadlessConfig gHeadlessConfig;
static JointDriveCase gHeadlessCase = eCASE_VELOCITY;
static JointDriveMetrics gMetrics;
static JointDrivePositionMetrics gPositionMetrics;
static JointDriveComparisonMetrics gComparisonMetrics;
static JointDriveAngularLimitMetrics gAngularLimitMetrics;
static OrderingDriveRuntime gOrderingRuntime;
static OrderingDriveMetrics gOrderingMetrics;
static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;
static const PxU32 gMaxDrivePairs = 2;
static DrivePairRuntime gPairs[gMaxDrivePairs];
static PxU32 gPairCount = 0;
static bool gHeadlessMode = false;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;

static const PxReal gTargetVelocity = 1.0f;
static const PxReal gDriveDamping = 1000.0f;
static const PxReal gPositionTargetMagnitude = 0.5f;
static const PxReal gPositionInitialRelativeMagnitude = 0.2f;
static const PxReal gPositionDriveStiffness = 100.0f;
static const PxReal gPositionDriveDamping = 20.0f;
static const PxReal gPositionLowForceLimit = 5.0f;
static const PxReal gPositionOffsetAnchorMagnitude = 0.25f;
static const PxReal gPositionDuration = 3.0f;
static const PxReal gPositionLateWindowDuration = 1.0f;
static const PxReal gKinematicMotionStartTime = 0.5f;
static const PxReal gKinematicAngularSpeed = 0.25f;
static const PxU32 gLateWindowFrames = 60;
static const PxReal gMinimumMotionWitness = 0.5f;
static const PxReal gMinimumLateProjection = 0.75f;
static const PxReal gMaximumLateProjection = 1.25f;
static const PxReal gMaximumLateTargetRms = 0.35f;
static const PxReal gMinimumAxisDot = 0.98f;
static const PxReal gMaximumOrthogonalRms = 0.10f;
static const PxReal gMaximumActorAPositionError = 1e-4f;
static const PxReal gMaximumActorAAngleError = 1e-4f;
static const PxReal gMaximumKinematicAngularSpeedError = 1e-4f;
static const PxReal gLegacyConeLimitY = PxPi / 9.0f;
static const PxReal gLegacyConeLimitZ = 7.0f * PxPi / 36.0f;
static const PxReal gLegacyConeOutsideY = PxPi / 10.0f;
static const PxReal gLegacyConeOutsideZ = PxPi / 6.0f;
static const PxReal gLegacyConeInsideY = PxPi / 18.0f;
static const PxReal gLegacyConeInsideZ = PxPi / 12.0f;
static const PxReal gLegacyConeFinalRadiusTolerance = 0.01f;
static const PxReal gLegacyConeLateRadiusTolerance = 0.02f;
static const PxReal gLegacyConeMinimumRadiusCorrection = 0.10f;
static const PxReal gLegacyConeInsideRadiusDeviationTolerance = 0.01f;
// A moving kinematic endpoint is part of the authored relative drive
// objective.  TGS stays below 0.007 for both accepted initial states at
// 30/60/120 Hz; 0.02 leaves authority margin while still rejecting a
// frame-rate-dependent steady lag.
static const PxReal gMaximumMovingKinematicFinalErrorRatio = 0.02f;
static const PxReal gMaximumMovingKinematicLateErrorRatio = 0.02f;
static const PxReal gMaximumQuaternionNormError = 1e-3f;
static const PxReal gMaximumAbsPosition = 1000.0f;
static const PxReal gMaximumLinearSpeed = 20.0f;
static const PxReal gMaximumAngularSpeed = 20.0f;
static const PxReal gMaximumPositionErrorRatio = 0.25f;
static const PxReal gMaximumLatePositionErrorRatio = 0.30f;
static const PxReal gMinimumPositionMotionRatio = 0.50f;
static const PxReal gMaximumPositionOrthogonalError = 0.05f;
// Official SLERP spring rows follow the quaternion Jacobian rather than a
// fixed authored axis.  PGS/TGS therefore take a small curved orientation path
// (reference maximum 0.052782 across 30/60/120 Hz) while converging exactly.
static const PxReal gMaximumSlerpPositionOrthogonalError = 0.06f;
static const PxReal gMaximumPositionOvershootRatio = 0.50f;
static const PxReal gMaximumPositionReverseMotionRatio = 0.10f;
static const PxReal gMaximumLatePositionSpeed = 0.25f;
static const PxReal gOutputForceZeroTolerance = 1e-4f;
static const PxReal gOutputForceDirectionTolerance = 0.01f;
static const PxReal gOutputTorqueTolerance = 1e-3f;
static const PxReal gOutputForceBreakBelow = 4.0f;
static const PxReal gOutputForceBreakAbove = 6.0f;
static const PxReal gOutputMomentBreakBelow = 1.0f;
static const PxReal gOutputMomentBreakAbove = 1.5f;
static const PxReal gAngularOutputTorqueLimit = 5.0f;
static const PxReal gAngularOutputForceTolerance = 1e-3f;
static const PxReal gAngularOutputTorqueZeroTolerance = 1e-4f;
static const PxU32 gAngularOutputTransientFrames = 12;
static const PxReal gMassProbeReferenceMass = 1.0f;
static const PxReal gMassProbeDamping = 6.0f;
static const PxReal gMassProbeTargetVelocity = 1.0f;
static const PxU32 gMassProbeTransientFrames = 20;
static const PxReal gAccelerationLimitProbeLowLimit = 1.0f;
static const PxReal gForceLimitProbeDamping = 600.0f;
static const PxReal gForceLimitProbeTargetVelocity = 4.0f;
static const PxReal gForceLimitProbeHighLimit = 6000.0f;
static const PxReal gForceLimitProbeLowLimit = 6.0f;
static const PxU32 gForceLimitProbeTransientFrames = 12;
static const PxReal gComparisonMinimumAxisDot = 0.99f;
static const PxReal gComparisonMaximumOrthogonalScale = 0.05f;
static const PxReal gComparisonMaximumMonotonicDropScale = 0.05f;
static const PxReal gDynamicComparisonMaximumComDrift = 1e-3f;
static const PxReal gDynamicComparisonMaximumMomentum = 1e-3f;
static const PxReal gDynamicAngularPositionMaximumComDrift = 1e-4f;
static const PxReal gDynamicAngularPositionMaximumLinearMomentum = 1e-4f;
static const PxReal gDynamicAngularPositionMaximumAngularMomentum = 1e-3f;
static const PxReal gContactComparisonMinimumBottom = -0.05f;
static const PxReal gContactComparisonMaximumVerticalSpeed = 1.0f;
static const PxReal gOrderingMaximumRotationError = 1e-3f;
static const PxReal gOrderingMaximumAngularSpeed = 1e-3f;

static PxU32 getPositionLateWindowFrames() {
  return PxU32(PxFloor(gPositionLateWindowDuration / gHeadlessOptions.dt +
                       0.5f));
}

static PxReal getPositionForceLimit() {
  return gHeadlessConfig.lowForceLimit ? gPositionLowForceLimit : FLT_MAX;
}

static bool isDynamicComparisonTopology(JointDriveTopology topology) {
  return topology == eTOPOLOGY_DYNAMIC_DYNAMIC ||
         topology == eTOPOLOGY_CONTACT_DYNAMIC_DYNAMIC;
}

static bool isContactComparisonTopology(JointDriveTopology topology) {
  return topology == eTOPOLOGY_CONTACT_DYNAMIC_DYNAMIC;
}

static bool passesComparisonConservationGate() {
  if (!isDynamicComparisonTopology(gHeadlessConfig.topology))
    return true;
  const PxReal maximumComDrift =
      isContactComparisonTopology(gHeadlessConfig.topology)
          ? 2e-3f
          : gDynamicComparisonMaximumComDrift;
  for (PxU32 i = 0; i < 2; ++i) {
    if (gComparisonMetrics.maximumCenterOfMassDrift[i] >
            maximumComDrift ||
        gComparisonMetrics.maximumMomentumMagnitude[i] >
            gDynamicComparisonMaximumMomentum)
      return false;
  }
  return true;
}

static bool passesContactCoverageGate() {
  if (!isContactComparisonTopology(gHeadlessConfig.topology))
    return true;
  for (PxU32 i = 0; i < gPairCount; ++i) {
    if (gComparisonMetrics.bothBodyContactFrames[i] !=
            gHeadlessOptions.frames ||
        !gComparisonMetrics.contactPointCount[i])
      return false;
  }
  return true;
}

static bool passesContactSupportGate() {
  if (!isContactComparisonTopology(gHeadlessConfig.topology))
    return true;
  for (PxU32 i = 0; i < gPairCount; ++i) {
    if (gComparisonMetrics.minimumBottom[i] <
            gContactComparisonMinimumBottom ||
        gComparisonMetrics.maximumAbsVerticalSpeed[i] >
            gContactComparisonMaximumVerticalSpeed)
      return false;
  }
  return true;
}

static const char *getDriveName(JointDriveKind drive) {
  switch (drive) {
  case eDRIVE_LINEAR_Y:
    return "y";
  case eDRIVE_LINEAR_Z:
    return "z";
  case eDRIVE_TWIST:
    return "twist";
  case eDRIVE_SWING1:
    return "swing1";
  case eDRIVE_SWING2:
    return "swing2";
  case eDRIVE_SLERP:
    return "slerp";
  default:
    return "x";
  }
}

static const char *getCaseName(JointDriveCase testCase) {
  switch (testCase) {
  case eCASE_VELOCITY_ORDERING:
    return "velocity-ordering";
  case eCASE_ANGULAR_ORDERING:
    return "angular-ordering";
  case eCASE_ANGULAR_OUTPUT_FORCE:
    return "angular-output-force";
  case eCASE_POSITION:
    return "position";
  case eCASE_ANGULAR_POSITION:
    return "angular-position";
  case eCASE_OUTPUT_FORCE:
    return "output-force";
  case eCASE_MASS_SCALING:
    return "mass-scaling";
  case eCASE_ACCELERATION_MODE:
    return "acceleration-mode";
  case eCASE_FORCE_LIMIT:
    return "force-limit";
  case eCASE_LEGACY_ANGULAR_LIMIT_CONE_OUTSIDE:
    return "legacy-angular-limit-cone-outside";
  case eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE:
    return "legacy-angular-limit-cone-inside";
  default:
    return "velocity";
  }
}

static bool tryParseCase(const char *value, JointDriveCase &testCase) {
  if (Snippets::equalsIgnoreCase(value, "velocity"))
    testCase = eCASE_VELOCITY;
  else if (Snippets::equalsIgnoreCase(value, "velocity-ordering"))
    testCase = eCASE_VELOCITY_ORDERING;
  else if (Snippets::equalsIgnoreCase(value, "angular-ordering"))
    testCase = eCASE_ANGULAR_ORDERING;
  else if (Snippets::equalsIgnoreCase(value, "angular-output-force"))
    testCase = eCASE_ANGULAR_OUTPUT_FORCE;
  else if (Snippets::equalsIgnoreCase(value, "position"))
    testCase = eCASE_POSITION;
  else if (Snippets::equalsIgnoreCase(value, "angular-position"))
    testCase = eCASE_ANGULAR_POSITION;
  else if (Snippets::equalsIgnoreCase(value, "output-force"))
    testCase = eCASE_OUTPUT_FORCE;
  else if (Snippets::equalsIgnoreCase(value, "mass-scaling"))
    testCase = eCASE_MASS_SCALING;
  else if (Snippets::equalsIgnoreCase(value, "acceleration-mode"))
    testCase = eCASE_ACCELERATION_MODE;
  else if (Snippets::equalsIgnoreCase(value, "force-limit"))
    testCase = eCASE_FORCE_LIMIT;
  else if (Snippets::equalsIgnoreCase(
               value, "legacy-angular-limit-cone-outside"))
    testCase = eCASE_LEGACY_ANGULAR_LIMIT_CONE_OUTSIDE;
  else if (Snippets::equalsIgnoreCase(
               value, "legacy-angular-limit-cone-inside"))
    testCase = eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE;
  else
    return false;
  return true;
}

static const char *getValidationName(JointDriveCase testCase) {
  return testCase == eCASE_VELOCITY ||
                 testCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_OUTSIDE ||
                 testCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE
             ? "GATED"
             : "PROBE";
}

static bool isPositionLikeCase(JointDriveCase testCase) {
  return testCase == eCASE_POSITION ||
         testCase == eCASE_ANGULAR_POSITION ||
         testCase == eCASE_OUTPUT_FORCE;
}

static bool isLegacyAngularLimitCase(JointDriveCase testCase) {
  return testCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_OUTSIDE ||
         testCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE;
}

static const char *getEndpointName(JointDriveEndpoint endpoint) {
  return endpoint == eENDPOINT_REVERSE ? "reverse" : "forward";
}

static const char *getOrderingActor0Name() {
  return gHeadlessConfig.endpoint == eENDPOINT_REVERSE ? "dynamic" : "world";
}

static const char *getOrderingActor1Name() {
  return gHeadlessConfig.endpoint == eENDPOINT_REVERSE ? "world" : "dynamic";
}

static bool tryParseEndpoint(const char *value, JointDriveEndpoint &endpoint) {
  if (Snippets::equalsIgnoreCase(value, "forward"))
    endpoint = eENDPOINT_FORWARD;
  else if (Snippets::equalsIgnoreCase(value, "reverse"))
    endpoint = eENDPOINT_REVERSE;
  else
    return false;
  return true;
}

static const char *getBreakModeName(JointDriveBreakMode mode) {
  if (mode == eBREAK_BELOW_DRIVE_LIMIT)
    return "below";
  if (mode == eBREAK_ABOVE_DRIVE_LIMIT)
    return "above";
  if (mode == eBREAK_BELOW_OFFSET_MOMENT)
    return "moment-below";
  if (mode == eBREAK_ABOVE_OFFSET_MOMENT)
    return "moment-above";
  return "none";
}

static bool tryParseBreakMode(const char *value,
                              JointDriveBreakMode &mode) {
  if (Snippets::equalsIgnoreCase(value, "none"))
    mode = eBREAK_UNBREAKABLE;
  else if (Snippets::equalsIgnoreCase(value, "below"))
    mode = eBREAK_BELOW_DRIVE_LIMIT;
  else if (Snippets::equalsIgnoreCase(value, "above"))
    mode = eBREAK_ABOVE_DRIVE_LIMIT;
  else if (Snippets::equalsIgnoreCase(value, "moment-below"))
    mode = eBREAK_BELOW_OFFSET_MOMENT;
  else if (Snippets::equalsIgnoreCase(value, "moment-above"))
    mode = eBREAK_ABOVE_OFFSET_MOMENT;
  else
    return false;
  return true;
}

static const char *getTopologyName(JointDriveTopology topology) {
  if (topology == eTOPOLOGY_DYNAMIC_DYNAMIC)
    return "dynamic-dynamic";
  if (topology == eTOPOLOGY_CONTACT_DYNAMIC_DYNAMIC)
    return "contact-dynamic-dynamic";
  return "static-dynamic";
}

static bool tryParseTopology(const char *value,
                             JointDriveTopology &topology) {
  if (Snippets::equalsIgnoreCase(value, "static-dynamic"))
    topology = eTOPOLOGY_STATIC_DYNAMIC;
  else if (Snippets::equalsIgnoreCase(value, "dynamic-dynamic"))
    topology = eTOPOLOGY_DYNAMIC_DYNAMIC;
  else if (Snippets::equalsIgnoreCase(value,
                                      "contact-dynamic-dynamic"))
    topology = eTOPOLOGY_CONTACT_DYNAMIC_DYNAMIC;
  else
    return false;
  return true;
}

static const char *
getKinematicMotionName(JointKinematicMotion motion) {
  return motion == eKINEMATIC_SPIN_WORLD_Y ? "spin-world-y"
                                           : "stationary";
}

static bool tryParseKinematicMotion(const char *value,
                                    JointKinematicMotion &motion) {
  if (Snippets::equalsIgnoreCase(value, "stationary"))
    motion = eKINEMATIC_STATIONARY;
  else if (Snippets::equalsIgnoreCase(value, "spin-world-y"))
    motion = eKINEMATIC_SPIN_WORLD_Y;
  else
    return false;
  return true;
}

static const char *getDriveModeName(JointDriveMode driveMode) {
  return driveMode == eDRIVE_MODE_ACCELERATION ? "acceleration" : "force";
}

static bool tryParseDriveMode(const char *value, JointDriveMode &driveMode) {
  if (Snippets::equalsIgnoreCase(value, "force"))
    driveMode = eDRIVE_MODE_FORCE;
  else if (Snippets::equalsIgnoreCase(value, "acceleration"))
    driveMode = eDRIVE_MODE_ACCELERATION;
  else
    return false;
  return true;
}

static bool tryParseDrive(const char *value, JointDriveKind &drive) {
  if (Snippets::equalsIgnoreCase(value, "x"))
    drive = eDRIVE_LINEAR_X;
  else if (Snippets::equalsIgnoreCase(value, "y"))
    drive = eDRIVE_LINEAR_Y;
  else if (Snippets::equalsIgnoreCase(value, "z"))
    drive = eDRIVE_LINEAR_Z;
  else if (Snippets::equalsIgnoreCase(value, "twist"))
    drive = eDRIVE_TWIST;
  else if (Snippets::equalsIgnoreCase(value, "swing1"))
    drive = eDRIVE_SWING1;
  else if (Snippets::equalsIgnoreCase(value, "swing2"))
    drive = eDRIVE_SWING2;
  else if (Snippets::equalsIgnoreCase(value, "slerp"))
    drive = eDRIVE_SLERP;
  else
    return false;
  return true;
}

static bool isLinearDrive(JointDriveKind drive) {
  return drive == eDRIVE_LINEAR_X || drive == eDRIVE_LINEAR_Y ||
         drive == eDRIVE_LINEAR_Z;
}

static bool isAngularDrive(JointDriveKind drive) {
  return !isLinearDrive(drive);
}

static bool isMovingKinematicPositionFixture() {
  return gHeadlessMode &&
         gHeadlessCase == eCASE_ANGULAR_POSITION &&
         gHeadlessConfig.drive == eDRIVE_SLERP &&
         gHeadlessConfig.actorAKinematic &&
         gHeadlessConfig.kinematicMotion ==
             eKINEMATIC_SPIN_WORLD_Y;
}

static PxVec3 getLocalDriveAxis(JointDriveKind drive) {
  switch (drive) {
  case eDRIVE_SLERP:
    // A non-axis-aligned target is required to prove that SLERP consumes all
    // three angular rows.  A single Y component can be mistaken for SWING1 or
    // for the shared SWING2/SLERP parameter slot in external prep.
    return PxVec3(1.0f, -1.0f, 1.0f).getNormalized();
  case eDRIVE_LINEAR_Y:
  case eDRIVE_SWING1:
    return PxVec3(0.0f, 1.0f, 0.0f);
  case eDRIVE_LINEAR_Z:
  case eDRIVE_SWING2:
    return PxVec3(0.0f, 0.0f, 1.0f);
  default:
    return PxVec3(1.0f, 0.0f, 0.0f);
  }
}

static PxVec3 getExpectedActor0WorldTorque(JointDriveKind drive,
                                           const PxQuat &actorAFrame) {
  if (drive == eDRIVE_SLERP) {
    // With zero stiffness, PhysX emits three fixed world-axis SLERP rows.  The
    // actor-A frame transforms the authored target first, then the shared
    // scalar limit clamps each world row independently.
    const PxVec3 target = actorAFrame.rotate(getLocalDriveAxis(drive));
    return PxVec3(target.x >= 0.0f ? -gAngularOutputTorqueLimit
                                  : gAngularOutputTorqueLimit,
                  target.y >= 0.0f ? -gAngularOutputTorqueLimit
                                  : gAngularOutputTorqueLimit,
                  target.z >= 0.0f ? -gAngularOutputTorqueLimit
                                  : gAngularOutputTorqueLimit);
  }
  return actorAFrame.rotate(getLocalDriveAxis(drive)) *
         gAngularOutputTorqueLimit;
}

static PxReal getAngularOutputBreakScale(JointDriveKind drive) {
  return drive == eDRIVE_SLERP ? PxSqrt(3.0f) : 1.0f;
}

static PxReal getAngularOutputBreakThreshold(JointDriveKind drive,
                                             JointDriveBreakMode breakMode) {
  const PxReal scale = getAngularOutputBreakScale(drive);
  if (breakMode == eBREAK_BELOW_DRIVE_LIMIT)
    return gOutputForceBreakBelow * scale;
  if (breakMode == eBREAK_ABOVE_DRIVE_LIMIT)
    return gOutputForceBreakAbove * scale;
  return PX_MAX_F32;
}

static PxReal getRelativeTargetSign(JointDriveKind drive) {
  // PhysX D6 convention: axis twist/swing rows target (wA-wB), while the
  // SLERP row and linear rows target (vB-vA).
  return drive == eDRIVE_TWIST || drive == eDRIVE_SWING1 ||
                 drive == eDRIVE_SWING2
             ? -1.0f
             : 1.0f;
}

static PxD6Drive::Enum getDriveSlot(JointDriveKind drive) {
  switch (drive) {
  case eDRIVE_LINEAR_Y:
    return PxD6Drive::eY;
  case eDRIVE_LINEAR_Z:
    return PxD6Drive::eZ;
  case eDRIVE_TWIST:
    return PxD6Drive::eTWIST;
  case eDRIVE_SWING1:
    return PxD6Drive::eSWING1;
  case eDRIVE_SWING2:
    return PxD6Drive::eSWING2;
  case eDRIVE_SLERP:
    return PxD6Drive::eSLERP;
  default:
    return PxD6Drive::eX;
  }
}

static const char *getOrientationName(bool rotated) {
  return rotated ? "rotz-neg45" : "identity";
}

static const char *getOrientationName(JointFrameOrientation orientation) {
  switch (orientation) {
  case eFRAME_ROTZ_NEG45:
    return "rotz-neg45";
  case eFRAME_ROTX_NEG45:
    return "rotx-neg45";
  default:
    return "identity";
  }
}

static const char *getInitialRelativeName(bool offset) {
  return offset ? "driven-pos20" : "identity";
}

static const char *getAnchorName(bool offset) {
  return offset ? "symmetric-y25" : "centered";
}

static bool tryParseAnchor(const char *value, bool &offset) {
  if (Snippets::equalsIgnoreCase(value, "centered")) {
    offset = false;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "symmetric-y25")) {
    offset = true;
    return true;
  }
  return false;
}

static bool tryParseInitialRelative(const char *value, bool &offset) {
  if (Snippets::equalsIgnoreCase(value, "identity")) {
    offset = false;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "driven-pos20")) {
    offset = true;
    return true;
  }
  return false;
}

static bool tryParseOrientation(const char *value, bool &rotated) {
  if (Snippets::equalsIgnoreCase(value, "identity")) {
    rotated = false;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "rotz-neg45")) {
    rotated = true;
    return true;
  }
  return false;
}

static bool tryParseOrientation(const char *value,
                                JointFrameOrientation &orientation) {
  if (Snippets::equalsIgnoreCase(value, "identity")) {
    orientation = eFRAME_IDENTITY;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "rotz-neg45")) {
    orientation = eFRAME_ROTZ_NEG45;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "rotx-neg45")) {
    orientation = eFRAME_ROTX_NEG45;
    return true;
  }
  return false;
}

static PxReal safeMagnitude(const PxVec3 &value) {
  if (!value.isFinite())
    return PX_MAX_F32;
  const double x = double(value.x);
  const double y = double(value.y);
  const double z = double(value.z);
  const double magnitude = std::sqrt(x * x + y * y + z * z);
  return std::isfinite(magnitude) && magnitude < double(PX_MAX_F32)
             ? PxReal(magnitude)
             : PX_MAX_F32;
}

static PxReal maxAbsComponent(const PxVec3 &value) {
  if (!value.isFinite())
    return PX_MAX_F32;
  return PxMax(PxAbs(value.x), PxMax(PxAbs(value.y), PxAbs(value.z)));
}

static PxReal quaternionAngle(const PxQuat &a, const PxQuat &b) {
  if (!a.isFinite() || !b.isFinite())
    return PX_MAX_F32;
  const PxReal dot = PxClamp(PxAbs(a.dot(b)), 0.0f, 1.0f);
  return 2.0f * PxAcos(dot);
}

static PxVec3 quaternionDeltaVector(const PxQuat &current,
                                    const PxQuat &previous) {
  PxQuat delta = current * previous.getConjugate();
  if (delta.w < 0.0f)
    delta = -delta;
  const PxReal clampedW = PxClamp(delta.w, -1.0f, 1.0f);
  const PxReal angle = 2.0f * PxAcos(clampedW);
  const PxReal sinHalfSquared = PxMax(0.0f, 1.0f - clampedW * clampedW);
  if (sinHalfSquared > 1e-12f)
    return PxVec3(delta.x, delta.y, delta.z) *
           (angle / PxSqrt(sinHalfSquared));
  return PxVec3(delta.x, delta.y, delta.z) * 2.0f;
}

static PxReal computeJointFrameConeAngle(const PxTransform &poseA,
                                         const PxTransform &poseB,
                                         const DrivePairRuntime &pair) {
  const PxVec3 axisA =
      (poseA.q * pair.jointFrameA.q)
          .rotate(PxVec3(1.0f, 0.0f, 0.0f));
  const PxVec3 axisB =
      (poseB.q * pair.jointFrameB.q)
          .rotate(PxVec3(1.0f, 0.0f, 0.0f));
  return PxAcos(PxClamp(axisA.dot(axisB), -1.0f, 1.0f));
}

static PxQuat makeLegacyConeSwing(PxReal swingY, PxReal swingZ) {
  const PxVec3 tanQuarter(0.0f, PxTan(swingY * 0.25f),
                         PxTan(swingZ * 0.25f));
  const PxReal magnitudeSquared = tanQuarter.magnitudeSquared();
  const PxReal inverseDenominator = 1.0f / (1.0f + magnitudeSquared);
  return PxQuat(0.0f, 2.0f * tanQuarter.y * inverseDenominator,
                2.0f * tanQuarter.z * inverseDenominator,
                (1.0f - magnitudeSquared) * inverseDenominator)
      .getNormalized();
}

static PxReal computeLegacyConeEllipseRadius(
    const PxTransform &poseA, const PxTransform &poseB,
    const DrivePairRuntime &pair, PxReal limitY, PxReal limitZ) {
  PxQuat worldFrameA = poseA.q * pair.jointFrameA.q;
  PxQuat worldFrameB = poseB.q * pair.jointFrameB.q;
  worldFrameA.normalize();
  worldFrameB.normalize();
  PxQuat relative = worldFrameA.getConjugate() * worldFrameB;
  relative.normalize();
  if (relative.w < 0.0f)
    relative = -relative;
  PxQuat swing, twist;
  PxSeparateSwingTwist(relative, swing, twist);
  if (swing.w < 0.0f)
    swing = -swing;
  const PxReal denominator = 1.0f + swing.w;
  if (denominator <= 1e-6f || limitY <= 0.0f || limitZ <= 0.0f)
    return PX_MAX_F32;
  const PxReal swingY = 4.0f * PxAtan2(swing.y, denominator);
  const PxReal swingZ = 4.0f * PxAtan2(swing.z, denominator);
  const PxReal normalizedY = swingY / limitY;
  const PxReal normalizedZ = swingZ / limitZ;
  return PxSqrt(normalizedY * normalizedY +
                normalizedZ * normalizedZ);
}

static void computeSlerpJacobianAxes(PxVec3 rows[3], const PxQuat &qa,
                                     const PxQuat &qb) {
  const PxReal wa = qa.w;
  const PxReal wb = qb.w;
  const PxVec3 va(qa.x, qa.y, qa.z);
  const PxVec3 vb(qb.x, qb.y, qb.z);
  const PxVec3 c = vb * wa + va * wb;
  const PxReal d0 = wa * wb;
  const PxReal d1 = va.dot(vb);
  const PxReal d = d0 - d1;
  rows[0] =
      (va * vb.x + vb * va.x + PxVec3(d, c.z, -c.y)) * 0.5f;
  rows[1] =
      (va * vb.y + vb * va.y + PxVec3(-c.z, d, c.x)) * 0.5f;
  rows[2] =
      (va * vb.z + vb * va.z + PxVec3(c.y, -c.x, d)) * 0.5f;
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

static void resetRuntimeState() {
  gMetrics = JointDriveMetrics();
  gPositionMetrics = JointDrivePositionMetrics();
  gComparisonMetrics = JointDriveComparisonMetrics();
  gAngularLimitMetrics = JointDriveAngularLimitMetrics();
  gOrderingRuntime = OrderingDriveRuntime();
  gOrderingMetrics = OrderingDriveMetrics();
  for (PxU32 i = 0; i < gMaxDrivePairs; ++i)
    gPairs[i] = DrivePairRuntime();
  gPairCount = 0;
  gInitializationFailed = false;
  gErrorCallback.reset();
}

static void releaseSceneState() {
  for (PxU32 i = 0; i < gPairCount; ++i)
    PX_RELEASE(gPairs[i].joint);
  PX_RELEASE(gScene);
  PX_RELEASE(gDispatcher);
  for (PxU32 i = 0; i < gMaxDrivePairs; ++i)
    gPairs[i] = DrivePairRuntime();
  gPairCount = 0;
}

static bool setupActor(PxRigidActor *actor) {
  if (!actor || !gScene)
    return false;
  actor->setActorFlag(PxActorFlag::eVISUALIZATION, true);
  gScene->addActor(*actor);
  return true;
}

struct JointDriveContactActorTag {
  PxU32 lane;
  PxU32 body;
  bool ground;

  JointDriveContactActorTag()
      : lane(PX_MAX_U32), body(PX_MAX_U32), ground(false) {}
};

static JointDriveContactActorTag gContactGroundTag;
static JointDriveContactActorTag gContactBodyTags[2][2];

static PxFilterFlags contactComparisonFilterShader(
    PxFilterObjectAttributes attributes0, PxFilterData filterData0,
    PxFilterObjectAttributes attributes1, PxFilterData filterData1,
    PxPairFlags &pairFlags, const void *constantBlock,
    PxU32 constantBlockSize) {
  PX_UNUSED(attributes0);
  PX_UNUSED(attributes1);
  PX_UNUSED(constantBlock);
  PX_UNUSED(constantBlockSize);
  if (filterData0.word0 == 1u && filterData1.word0 == 1u)
    return PxFilterFlag::eSUPPRESS;
  pairFlags = PxPairFlag::eSOLVE_CONTACT |
              PxPairFlag::eDETECT_DISCRETE_CONTACT |
              PxPairFlag::eNOTIFY_TOUCH_FOUND |
              PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
              PxPairFlag::eNOTIFY_CONTACT_POINTS;
  return PxFilterFlag::eDEFAULT;
}

class JointDriveContactCallback : public PxSimulationEventCallback {
public:
  virtual void onConstraintBreak(PxConstraintInfo *, PxU32) PX_OVERRIDE {}
  virtual void onWake(PxActor **, PxU32) PX_OVERRIDE {}
  virtual void onSleep(PxActor **, PxU32) PX_OVERRIDE {}
  virtual void onTrigger(PxTriggerPair *, PxU32) PX_OVERRIDE {}
  virtual void onAdvance(const PxRigidBody *const *, const PxTransform *,
                         const PxU32) PX_OVERRIDE {}

  virtual void onContact(const PxContactPairHeader &pairHeader,
                         const PxContactPair *pairs,
                         PxU32 pairCount) PX_OVERRIDE {
    const JointDriveContactActorTag *tag0 =
        static_cast<const JointDriveContactActorTag *>(
            pairHeader.actors[0] ? pairHeader.actors[0]->userData : NULL);
    const JointDriveContactActorTag *tag1 =
        static_cast<const JointDriveContactActorTag *>(
            pairHeader.actors[1] ? pairHeader.actors[1]->userData : NULL);
    if (!tag0 || !tag1 || tag0->ground == tag1->ground)
      return;
    const JointDriveContactActorTag *bodyTag = tag0->ground ? tag1 : tag0;
    if (bodyTag->lane >= 2 || bodyTag->body >= 2)
      return;

    gComparisonMetrics.currentContactMask[bodyTag->lane] |=
        1u << bodyTag->body;
    for (PxU32 i = 0; i < pairCount; ++i)
      gComparisonMetrics.contactPointCount[bodyTag->lane] +=
          pairs[i].contactCount;
  }
};

static JointDriveContactCallback gContactCallback;

static void beginContactComparisonFrame() {
  if (!isContactComparisonTopology(gHeadlessConfig.topology))
    return;
  for (PxU32 i = 0; i < 2; ++i)
    gComparisonMetrics.currentContactMask[i] = 0;
}

static void commitContactComparisonFrame() {
  if (!isContactComparisonTopology(gHeadlessConfig.topology))
    return;
  for (PxU32 i = 0; i < 2; ++i) {
    const PxU32 mask = gComparisonMetrics.currentContactMask[i];
    if (mask & 1u)
      gComparisonMetrics.bodyContactFrames[i][0]++;
    if (mask & 2u)
      gComparisonMetrics.bodyContactFrames[i][1]++;
    if ((mask & 3u) == 3u)
      gComparisonMetrics.bothBodyContactFrames[i]++;
  }
}

static bool configureContactBody(
    PxRigidDynamic &actor, PxU32 lane, PxU32 body) {
  if (lane >= 2 || body >= 2)
    return false;
  gContactBodyTags[lane][body].lane = lane;
  gContactBodyTags[lane][body].body = body;
  gContactBodyTags[lane][body].ground = false;
  actor.userData = &gContactBodyTags[lane][body];
  PxShape *shape = NULL;
  if (actor.getShapes(&shape, 1) != 1 || !shape)
    return false;
  PxFilterData bodyFilterData;
  bodyFilterData.word0 = 1u;
  shape->setSimulationFilterData(bodyFilterData);
  return true;
}

static JointDriveKind getInteractiveDrive() {
  switch (gSceneIndex) {
  case 1:
    return eDRIVE_TWIST;
  case 2:
    return eDRIVE_SWING1;
  case 3:
    return eDRIVE_SLERP;
  default:
    return eDRIVE_LINEAR_X;
  }
}

static bool configureDrive(PxD6Joint &joint, JointDriveKind drive,
                           bool accelerationDrive,
                           PxReal forceLimit = FLT_MAX,
                           bool outputForce = false) {
  PxD6JointDrive parameters(0.0f, gDriveDamping, forceLimit,
                            accelerationDrive);
  if (outputForce)
    parameters.flags |= PxD6JointDriveFlag::eOUTPUT_FORCE;
  PxVec3 linearVelocity(0.0f);
  PxVec3 angularVelocity(0.0f);

  switch (drive) {
  case eDRIVE_LINEAR_X:
    joint.setDrive(PxD6Drive::eX, parameters);
    linearVelocity.x = gTargetVelocity;
    break;
  case eDRIVE_LINEAR_Y:
    joint.setDrive(PxD6Drive::eY, parameters);
    linearVelocity.y = gTargetVelocity;
    break;
  case eDRIVE_LINEAR_Z:
    joint.setDrive(PxD6Drive::eZ, parameters);
    linearVelocity.z = gTargetVelocity;
    break;
  case eDRIVE_TWIST:
    joint.setDrive(PxD6Drive::eTWIST, parameters);
    angularVelocity.x = gTargetVelocity;
    break;
  case eDRIVE_SWING1:
    joint.setDrive(PxD6Drive::eSWING1, parameters);
    angularVelocity.y = gTargetVelocity;
    break;
  case eDRIVE_SWING2:
    joint.setDrive(PxD6Drive::eSWING2, parameters);
    angularVelocity.z = gTargetVelocity;
    break;
  case eDRIVE_SLERP:
    joint.setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
    joint.setDrive(PxD6Drive::eSLERP, parameters);
    angularVelocity = getLocalDriveAxis(eDRIVE_SLERP) * gTargetVelocity;
    break;
  }

  joint.setDriveVelocity(linearVelocity, angularVelocity, true);
  return true;
}

static PxTransform getOrderingWorldFrame(PxRigidActor *actor,
                                         const PxTransform &localFrame) {
  return actor ? actor->getGlobalPose() * localFrame : localFrame;
}

static bool createOrderingDrive() {
  const bool angular = gHeadlessCase == eCASE_ANGULAR_ORDERING ||
                       gHeadlessCase == eCASE_ANGULAR_OUTPUT_FORCE;
  const bool angularOutputForce =
      gHeadlessCase == eCASE_ANGULAR_OUTPUT_FORCE;
  const bool reverse = gHeadlessConfig.endpoint == eENDPOINT_REVERSE;
  const PxQuat bodyRotation = PxGetRotZQuat(-PxPi / 4.0f);
  const PxTransform dynamicPose(PxVec3(0.0f, 10.0f, 0.0f), bodyRotation);
  const PxTransform worldFrameA(
      dynamicPose.p, angular ? PxQuat(PxIdentity) : bodyRotation);
  const PxQuat cyclicFrameRotation(
      2.0f * PxPi / 3.0f, PxVec3(1.0f, 1.0f, 1.0f).getNormalized());
  const PxTransform worldFrameB(
      dynamicPose.p, angular ? cyclicFrameRotation : bodyRotation);
  const PxTransform dynamicLocalFrameA =
      dynamicPose.getInverse() * worldFrameA;
  const PxTransform dynamicLocalFrameB =
      dynamicPose.getInverse() * worldFrameB;
  const PxVec3 localAxis =
      angular ? getLocalDriveAxis(gHeadlessConfig.drive)
              : PxVec3(1.0f, 0.0f, 0.0f);
  const PxReal relativeTargetSign =
      angular ? getRelativeTargetSign(gHeadlessConfig.drive) : 1.0f;

  gOrderingRuntime.angular = angular;
  gOrderingMetrics.angular = angular;
  gOrderingRuntime.localFrameA =
      reverse ? dynamicLocalFrameA : worldFrameA;
  gOrderingRuntime.localFrameB =
      reverse ? worldFrameB : dynamicLocalFrameB;
  gOrderingRuntime.initialDynamicPose = dynamicPose;
  gOrderingRuntime.previousDynamicPose = dynamicPose;
  gOrderingRuntime.actorAWorldFrameRotation = worldFrameA.q;
  gOrderingRuntime.expectedFrameAAxis =
      worldFrameA.q.rotate(localAxis).getNormalized();
  gOrderingRuntime.expectedFrameBAxis =
      worldFrameB.q.rotate(localAxis).getNormalized();
  gOrderingRuntime.expectedAxis =
      gOrderingRuntime.expectedFrameAAxis * relativeTargetSign;
  gOrderingRuntime.expectedDynamicAxis =
      reverse ? -gOrderingRuntime.expectedAxis : gOrderingRuntime.expectedAxis;
  gOrderingRuntime.expectedDynamicFrameAxis =
      reverse ? gOrderingRuntime.expectedFrameAAxis
              : gOrderingRuntime.expectedFrameBAxis;
  const PxTransform expectedDynamicLocalFrame =
      reverse ? dynamicLocalFrameA : dynamicLocalFrameB;
  gOrderingRuntime.expectedDynamicLocalAxis =
      expectedDynamicLocalFrame.q.rotate(localAxis).getNormalized();
  gOrderingRuntime.expectedActor0Torque = getExpectedActor0WorldTorque(
      gHeadlessConfig.drive, worldFrameA.q);

  gOrderingMetrics.expectedWorldFrame =
      reverse ? worldFrameB : worldFrameA;
  gOrderingMetrics.expectedDynamicLocalFrame = expectedDynamicLocalFrame;
  gOrderingMetrics.expectedAxis = gOrderingRuntime.expectedAxis;
  gOrderingMetrics.expectedDynamicAxis =
      gOrderingRuntime.expectedDynamicAxis;
  gOrderingMetrics.expectedFrameAAxis =
      gOrderingRuntime.expectedFrameAAxis;
  gOrderingMetrics.expectedFrameBAxis =
      gOrderingRuntime.expectedFrameBAxis;
  gOrderingMetrics.expectedDynamicFrameAxis =
      gOrderingRuntime.expectedDynamicFrameAxis;
  gOrderingMetrics.expectedActor0Torque =
      gOrderingRuntime.expectedActor0Torque;

  gOrderingRuntime.dynamicActor = gPhysics->createRigidDynamic(dynamicPose);
  if (!gOrderingRuntime.dynamicActor)
    return false;
  gOrderingRuntime.dynamicActor->setMass(1.0f);
  gOrderingRuntime.dynamicActor->setMassSpaceInertiaTensor(PxVec3(1.0f));
  gOrderingRuntime.dynamicActor->setLinearDamping(0.0f);
  gOrderingRuntime.dynamicActor->setAngularDamping(0.0f);
  gOrderingRuntime.dynamicActor->setSolverIterationCounts(4, 1);
  if (!setupActor(gOrderingRuntime.dynamicActor))
    return false;

  PxRigidActor *actorA = reverse ? gOrderingRuntime.dynamicActor : NULL;
  PxRigidActor *actorB = reverse ? NULL : gOrderingRuntime.dynamicActor;
  gOrderingRuntime.joint =
      PxD6JointCreate(*gPhysics, actorA, gOrderingRuntime.localFrameA, actorB,
                      gOrderingRuntime.localFrameB);
  if (!gOrderingRuntime.joint)
    return false;
  gOrderingRuntime.joint->setAngularDriveConfig(
      PxD6AngularDriveConfig::eSWING_TWIST);
  gOrderingRuntime.joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
  gOrderingRuntime.joint->setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
  gOrderingRuntime.joint->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
  gOrderingRuntime.joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  gOrderingRuntime.joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  gOrderingRuntime.joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
  configureDrive(*gOrderingRuntime.joint,
                 angular ? gHeadlessConfig.drive : eDRIVE_LINEAR_X, false,
                 angularOutputForce ? gAngularOutputTorqueLimit : FLT_MAX,
                 angularOutputForce &&
                     gHeadlessConfig.outputForceEnabled);
  if (angularOutputForce) {
    gOrderingRuntime.joint->setConstraintFlag(
        PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
    if (gHeadlessConfig.breakMode != eBREAK_UNBREAKABLE)
      gOrderingRuntime.joint->setBreakForce(
          PX_MAX_F32,
          getAngularOutputBreakThreshold(gHeadlessConfig.drive,
                                         gHeadlessConfig.breakMode));
  }

  PxRigidActor *readbackA = NULL;
  PxRigidActor *readbackB = NULL;
  gOrderingRuntime.joint->getActors(readbackA, readbackB);
  gOrderingMetrics.actorOrderValid =
      reverse ? readbackA == gOrderingRuntime.dynamicActor && !readbackB
              : !readbackA && readbackB == gOrderingRuntime.dynamicActor;

  const PxTransform actualFrameA =
      gOrderingRuntime.joint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform actualFrameB =
      gOrderingRuntime.joint->getLocalPose(PxJointActorIndex::eACTOR1);
  const PxTransform actualWorldEndpointFrame =
      reverse ? actualFrameB : actualFrameA;
  const PxTransform actualDynamicLocalFrame =
      reverse ? actualFrameA : actualFrameB;
  const PxTransform actor0WorldFrame =
      getOrderingWorldFrame(readbackA, actualFrameA);
  const PxTransform actor1WorldFrame =
      getOrderingWorldFrame(readbackB, actualFrameB);
  gOrderingMetrics.worldFramePositionError = safeMagnitude(
      actualWorldEndpointFrame.p - gOrderingMetrics.expectedWorldFrame.p);
  gOrderingMetrics.dynamicLocalPositionError =
      safeMagnitude(actualDynamicLocalFrame.p -
                    gOrderingMetrics.expectedDynamicLocalFrame.p);
  gOrderingMetrics.worldFrameRotationDot =
      PxAbs(actualWorldEndpointFrame.q.dot(
          gOrderingMetrics.expectedWorldFrame.q));
  gOrderingMetrics.dynamicLocalRotationDot =
      PxAbs(actualDynamicLocalFrame.q.dot(
          gOrderingMetrics.expectedDynamicLocalFrame.q));
  gOrderingMetrics.actor0Axis =
      actor0WorldFrame.q.rotate(localAxis).getNormalized();
  gOrderingMetrics.actor1Axis =
      actor1WorldFrame.q.rotate(localAxis).getNormalized();
  gOrderingMetrics.dynamicLocalAxis =
      actualDynamicLocalFrame.q.rotate(localAxis).getNormalized();
  gOrderingMetrics.dynamicWorldAxis =
      (gOrderingRuntime.dynamicActor->getGlobalPose().q *
       actualDynamicLocalFrame.q)
          .rotate(localAxis)
          .getNormalized();
  gOrderingMetrics.actor0AxisDot =
      gOrderingMetrics.actor0Axis.dot(gOrderingMetrics.expectedFrameAAxis);
  gOrderingMetrics.actor1AxisDot =
      gOrderingMetrics.actor1Axis.dot(gOrderingMetrics.expectedFrameBAxis);
  gOrderingMetrics.dynamicLocalAxisDot = angular
      ? gOrderingMetrics.dynamicLocalAxis.dot(
            gOrderingRuntime.expectedDynamicLocalAxis)
      : gOrderingMetrics.dynamicLocalAxis.dot(gOrderingMetrics.expectedAxis);
  gOrderingMetrics.dynamicWorldAxisDot =
      gOrderingMetrics.dynamicWorldAxis.dot(
          gOrderingMetrics.expectedDynamicFrameAxis);
  gOrderingMetrics.frameAxisSeparationDot =
      gOrderingMetrics.actor0Axis.dot(gOrderingMetrics.actor1Axis);
  gOrderingMetrics.expectedFrameAxisSeparationDot =
      gOrderingMetrics.expectedFrameAAxis.dot(
          gOrderingMetrics.expectedFrameBAxis);
  gOrderingMetrics.bodyRotationDot = PxAbs(
      gOrderingRuntime.dynamicActor->getGlobalPose().q.dot(bodyRotation));

  gOrderingMetrics.freeMotionCount = 0;
  for (PxU32 axis = 0; axis < PxD6Axis::eCOUNT; ++axis) {
    if (gOrderingRuntime.joint->getMotion(PxD6Axis::Enum(axis)) ==
        PxD6Motion::eFREE)
      gOrderingMetrics.freeMotionCount++;
  }
  const PxD6Drive::Enum activeDrive =
      getDriveSlot(angular ? gHeadlessConfig.drive : eDRIVE_LINEAR_X);
  const PxD6JointDrive driveReadback =
      gOrderingRuntime.joint->getDrive(activeDrive);
  gOrderingMetrics.driveStiffnessReadback = driveReadback.stiffness;
  gOrderingMetrics.driveDampingReadback = driveReadback.damping;
  gOrderingMetrics.driveForceLimitReadback = driveReadback.forceLimit;
  gOrderingMetrics.outputForceFlagReadback =
      driveReadback.flags.isSet(PxD6JointDriveFlag::eOUTPUT_FORCE);
  gOrderingMetrics.driveLimitsAreForcesReadback =
      gOrderingRuntime.joint->getConstraintFlags().isSet(
          PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES);
  gOrderingRuntime.joint->getBreakForce(
      gOrderingMetrics.linearBreakForceReadback,
      gOrderingMetrics.angularBreakForceReadback);
  PxVec3 linearTarget(0.0f);
  PxVec3 angularTarget(0.0f);
  gOrderingRuntime.joint->getDriveVelocity(linearTarget, angularTarget);
  const PxVec3 expectedLinearTarget =
      angular ? PxVec3(0.0f) : PxVec3(gTargetVelocity, 0.0f, 0.0f);
  const PxVec3 expectedAngularTarget =
      angular ? localAxis * gTargetVelocity : PxVec3(0.0f);
  gOrderingMetrics.driveLinearTargetError =
      safeMagnitude(linearTarget - expectedLinearTarget);
  gOrderingMetrics.driveAngularTargetError =
      safeMagnitude(angularTarget - expectedAngularTarget);
  const PxD6AngularDriveConfig::Enum expectedAngularDriveConfig =
      angular && gHeadlessConfig.drive == eDRIVE_SLERP
          ? PxD6AngularDriveConfig::eSLERP
          : PxD6AngularDriveConfig::eSWING_TWIST;
  gOrderingMetrics.angularDriveConfigValid =
      gOrderingRuntime.joint->getAngularDriveConfig() ==
      expectedAngularDriveConfig;
  bool inactiveDrivesValid = true;
  const PxD6Drive::Enum inactiveDrives[] = {
      PxD6Drive::eX, PxD6Drive::eY, PxD6Drive::eZ, PxD6Drive::eTWIST,
      PxD6Drive::eSWING1, PxD6Drive::eSWING2};
  for (PxU32 drive = 0; drive < PX_ARRAY_SIZE(inactiveDrives); ++drive) {
    if (inactiveDrives[drive] == activeDrive)
      continue;
    if (angular && gHeadlessConfig.drive == eDRIVE_SLERP &&
        inactiveDrives[drive] >= PxD6Drive::eTWIST)
      continue;
    const PxD6JointDrive inactive =
        gOrderingRuntime.joint->getDrive(inactiveDrives[drive]);
    inactiveDrivesValid =
        inactiveDrivesValid && PxAbs(inactive.stiffness) <= 1e-6f &&
        PxAbs(inactive.damping) <= 1e-6f &&
        !inactive.flags.isSet(PxD6JointDriveFlag::eACCELERATION) &&
        !inactive.flags.isSet(PxD6JointDriveFlag::eOUTPUT_FORCE);
  }
  gOrderingMetrics.driveReadbackValid =
      driveReadback.isValid() &&
      PxAbs(driveReadback.stiffness) <= 1e-6f &&
      PxAbs(driveReadback.damping - gDriveDamping) <= 1e-4f &&
      PxAbs(driveReadback.forceLimit -
            (angularOutputForce ? gAngularOutputTorqueLimit : PX_MAX_F32)) <=
          (angularOutputForce ? 1e-5f : 0.0f) &&
      !driveReadback.flags.isSet(PxD6JointDriveFlag::eACCELERATION) &&
      gOrderingMetrics.outputForceFlagReadback ==
          (angularOutputForce && gHeadlessConfig.outputForceEnabled) &&
      gOrderingMetrics.driveLimitsAreForcesReadback == angularOutputForce &&
      gOrderingMetrics.driveLinearTargetError <= 1e-6f &&
      gOrderingMetrics.driveAngularTargetError <= 1e-6f &&
      gOrderingMetrics.angularDriveConfigValid && inactiveDrivesValid;

  gOrderingMetrics.gravity = gScene->getGravity();
  gOrderingMetrics.shapeCount = gOrderingRuntime.dynamicActor->getNbShapes();
  gOrderingMetrics.massReadback = gOrderingRuntime.dynamicActor->getMass();
  gOrderingMetrics.inertiaReadback =
      gOrderingRuntime.dynamicActor->getMassSpaceInertiaTensor();
  gOrderingMetrics.linearDampingReadback =
      gOrderingRuntime.dynamicActor->getLinearDamping();
  gOrderingMetrics.angularDampingReadback =
      gOrderingRuntime.dynamicActor->getAngularDamping();
  gOrderingMetrics.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gOrderingMetrics.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gOrderingMetrics.initialConstraints = gScene->getNbConstraints();

  const PxReal expectedLocalAxisDot = angular ? 1.0f : PxSqrt(0.5f);
  const bool authoredFrameAxesDistinct =
      angular && gHeadlessConfig.drive == eDRIVE_SLERP
          ? PxAbs(gOrderingMetrics.expectedFrameAxisSeparationDot) < 0.9f
          : PxAbs(gOrderingMetrics.expectedFrameAxisSeparationDot) <= 1e-5f;
  const bool frameSourceSeparationValid =
      !angular ||
      (authoredFrameAxesDistinct &&
       PxAbs(gOrderingMetrics.frameAxisSeparationDot -
             gOrderingMetrics.expectedFrameAxisSeparationDot) <= 1e-5f);
  gOrderingMetrics.frameWitnessValid =
      gOrderingMetrics.expectedAxis.isFinite() &&
      gOrderingMetrics.expectedDynamicAxis.isFinite() &&
      gOrderingMetrics.expectedFrameAAxis.isFinite() &&
      gOrderingMetrics.expectedFrameBAxis.isFinite() &&
      gOrderingMetrics.expectedDynamicFrameAxis.isFinite() &&
      gOrderingMetrics.actor0Axis.isFinite() &&
      gOrderingMetrics.actor1Axis.isFinite() &&
      gOrderingMetrics.dynamicLocalAxis.isFinite() &&
      gOrderingMetrics.dynamicWorldAxis.isFinite() &&
      gOrderingMetrics.worldFramePositionError <= 1e-6f &&
      gOrderingMetrics.dynamicLocalPositionError <= 1e-6f &&
      gOrderingMetrics.worldFrameRotationDot >= 0.99999f &&
      gOrderingMetrics.dynamicLocalRotationDot >= 0.99999f &&
      gOrderingMetrics.actor0AxisDot >= 0.99999f &&
      gOrderingMetrics.actor1AxisDot >= 0.99999f &&
      PxAbs(gOrderingMetrics.dynamicLocalAxisDot - expectedLocalAxisDot) <=
          1e-5f &&
      gOrderingMetrics.dynamicWorldAxisDot >= 0.99999f &&
      frameSourceSeparationValid &&
      gOrderingMetrics.bodyRotationDot >= 0.99999f;
  gOrderingMetrics.fixtureWitnessValid =
      gOrderingMetrics.actorOrderValid &&
      gOrderingMetrics.frameWitnessValid &&
      gOrderingMetrics.driveReadbackValid &&
      gOrderingMetrics.freeMotionCount == PxD6Axis::eCOUNT &&
      gOrderingMetrics.shapeCount == 0 &&
      gOrderingMetrics.gravity.isFinite() &&
      gOrderingMetrics.gravity.magnitudeSquared() <= 1e-12f &&
      PxAbs(gOrderingMetrics.massReadback - 1.0f) <= 1e-6f &&
      (gOrderingMetrics.inertiaReadback - PxVec3(1.0f)).magnitude() <=
          1e-6f &&
      PxAbs(gOrderingMetrics.linearDampingReadback) <= 1e-6f &&
      PxAbs(gOrderingMetrics.angularDampingReadback) <= 1e-6f;
  return true;
}

static PxTransform getJointRelativePose(const DrivePairRuntime &pair) {
  if (!pair.actorA || !pair.actorB)
    return PxTransform(PxIdentity);
  const PxTransform frameA = pair.actorA->getGlobalPose() * pair.jointFrameA;
  const PxTransform frameB = pair.actorB->getGlobalPose() * pair.jointFrameB;
  return frameA.transformInv(frameB);
}

static PxVec3 getPositionTargetAxis(JointDriveKind drive) {
  if (drive == eDRIVE_SLERP)
    return PxVec3(1.0f, 2.0f, 3.0f).getNormalized();
  return getLocalDriveAxis(drive);
}

static void configurePositionMotions(PxD6Joint &joint,
                                     JointDriveKind drive) {
  joint.setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
  joint.setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
  joint.setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
  joint.setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
  joint.setMotion(PxD6Axis::eSWING1, PxD6Motion::eLOCKED);
  joint.setMotion(PxD6Axis::eSWING2, PxD6Motion::eLOCKED);

  switch (drive) {
  case eDRIVE_LINEAR_X:
    joint.setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
    break;
  case eDRIVE_LINEAR_Y:
    joint.setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
    break;
  case eDRIVE_LINEAR_Z:
    joint.setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
    break;
  case eDRIVE_TWIST:
    joint.setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    break;
  case eDRIVE_SWING1:
    joint.setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    break;
  case eDRIVE_SWING2:
    joint.setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
    break;
  case eDRIVE_SLERP:
    joint.setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    joint.setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    joint.setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
    break;
  }
}

static bool configurePositionDrive(DrivePairRuntime &pair,
                                   JointDriveKind drive) {
  if (!pair.joint || !pair.actorA || !pair.actorB)
    return false;

  configurePositionMotions(*pair.joint, drive);
  pair.joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSWING_TWIST);
  const PxReal forceLimit = getPositionForceLimit();
  PxD6JointDrive parameters(gPositionDriveStiffness,
                            gPositionDriveDamping, forceLimit, false);
  if (gHeadlessCase == eCASE_OUTPUT_FORCE &&
      gHeadlessConfig.outputForceEnabled)
    parameters.flags |= PxD6JointDriveFlag::eOUTPUT_FORCE;
  PxD6Drive::Enum activeDrive = PxD6Drive::eX;
  switch (drive) {
  case eDRIVE_LINEAR_X:
    pair.joint->setDrive(PxD6Drive::eX, parameters);
    break;
  case eDRIVE_LINEAR_Y:
    activeDrive = PxD6Drive::eY;
    pair.joint->setDrive(PxD6Drive::eY, parameters);
    break;
  case eDRIVE_LINEAR_Z:
    activeDrive = PxD6Drive::eZ;
    pair.joint->setDrive(PxD6Drive::eZ, parameters);
    break;
  case eDRIVE_TWIST:
    activeDrive = PxD6Drive::eTWIST;
    pair.joint->setDrive(PxD6Drive::eTWIST, parameters);
    break;
  case eDRIVE_SWING1:
    activeDrive = PxD6Drive::eSWING1;
    pair.joint->setDrive(PxD6Drive::eSWING1, parameters);
    break;
  case eDRIVE_SWING2:
    activeDrive = PxD6Drive::eSWING2;
    pair.joint->setDrive(PxD6Drive::eSWING2, parameters);
    break;
  case eDRIVE_SLERP:
    activeDrive = PxD6Drive::eSLERP;
    pair.joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
    pair.joint->setDrive(PxD6Drive::eSLERP, parameters);
    break;
  }
  pair.joint->setDriveVelocity(PxVec3(0.0f), PxVec3(0.0f), true);
  pair.joint->setConstraintFlag(
      PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);

  const PxD6JointDrive driveReadback = pair.joint->getDrive(activeDrive);
  gPositionMetrics.stiffnessReadback = driveReadback.stiffness;
  gPositionMetrics.dampingReadback = driveReadback.damping;
  gPositionMetrics.forceLimitReadback = driveReadback.forceLimit;
  gPositionMetrics.outputForceFlagReadback =
      driveReadback.flags.isSet(PxD6JointDriveFlag::eOUTPUT_FORCE);
  gPositionMetrics.driveLimitsAreForcesReadback =
      pair.joint->getConstraintFlags().isSet(
          PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES);

  gPositionMetrics.initialRelativePose = getJointRelativePose(pair);
  gPositionMetrics.initialRelativeMagnitude =
      isLinearDrive(drive)
          ? safeMagnitude(gPositionMetrics.initialRelativePose.p)
          : quaternionAngle(PxQuat(PxIdentity),
                            gPositionMetrics.initialRelativePose.q);
  PxTransform expectedInitialRelative(PxIdentity);
  if (gHeadlessConfig.initialRelativeOffset) {
    const PxVec3 initialAxis = getPositionTargetAxis(drive);
    if (isLinearDrive(drive))
      expectedInitialRelative.p =
          initialAxis * gPositionInitialRelativeMagnitude;
    else
      expectedInitialRelative.q =
          PxQuat(gPositionInitialRelativeMagnitude, initialAxis);
  }
  gPositionMetrics.initialRelativeSetupError =
      safeMagnitude(gPositionMetrics.initialRelativePose.p -
                    expectedInitialRelative.p) +
      quaternionAngle(gPositionMetrics.initialRelativePose.q,
                      expectedInitialRelative.q);
  gPositionMetrics.targetRelativePose = gPositionMetrics.initialRelativePose;
  const PxVec3 targetAxis = getPositionTargetAxis(drive);
  if (isLinearDrive(drive)) {
    gPositionMetrics.targetRelativePose.p +=
        targetAxis * gPositionTargetMagnitude;
  } else {
    gPositionMetrics.targetRelativePose.q =
        (PxQuat(gPositionTargetMagnitude, targetAxis) *
         gPositionMetrics.initialRelativePose.q)
            .getNormalized();
  }
  gPositionMetrics.targetRelativeMagnitude =
      isLinearDrive(drive)
          ? safeMagnitude(gPositionMetrics.targetRelativePose.p)
          : quaternionAngle(PxQuat(PxIdentity),
                            gPositionMetrics.targetRelativePose.q);
  const bool reverse = gHeadlessConfig.endpoint == eENDPOINT_REVERSE;
  const PxTransform jointTarget =
      reverse ? gPositionMetrics.targetRelativePose.getInverse()
              : gPositionMetrics.targetRelativePose;
  pair.joint->setDrivePosition(jointTarget, true);
  const PxTransform readback = pair.joint->getDrivePosition();
  if (!readback.isValid()) {
    gPositionMetrics.targetReadbackError = PX_MAX_F32;
    return false;
  }
  gPositionMetrics.targetReadbackError =
      safeMagnitude(readback.p - jointTarget.p) +
      quaternionAngle(readback.q, jointTarget.q);
  gPositionMetrics.initialTargetError =
      isLinearDrive(drive)
          ? safeMagnitude(gPositionMetrics.targetRelativePose.p -
                          gPositionMetrics.initialRelativePose.p)
          : quaternionAngle(gPositionMetrics.initialRelativePose.q,
                            gPositionMetrics.targetRelativePose.q);
  gPositionMetrics.finalTargetError =
      gPositionMetrics.initialTargetError;
  return gPositionMetrics.targetReadbackError <= 1e-5f &&
         PxIsFinite(gPositionMetrics.initialTargetError) &&
         gPositionMetrics.initialTargetError > 1e-6f;
}

static bool isComparisonCase(JointDriveCase testCase) {
  return testCase == eCASE_MASS_SCALING ||
         testCase == eCASE_ACCELERATION_MODE ||
         testCase == eCASE_FORCE_LIMIT;
}

static bool isOrderingCase(JointDriveCase testCase) {
  return testCase == eCASE_VELOCITY_ORDERING ||
         testCase == eCASE_ANGULAR_ORDERING ||
         testCase == eCASE_ANGULAR_OUTPUT_FORCE;
}

static bool isAngularOrderingCase(JointDriveCase testCase) {
  return testCase == eCASE_ANGULAR_ORDERING ||
         testCase == eCASE_ANGULAR_OUTPUT_FORCE;
}

static bool isAngularOutputForceCase(JointDriveCase testCase) {
  return testCase == eCASE_ANGULAR_OUTPUT_FORCE;
}

static PxU32 getComparisonTransientFrames(JointDriveCase testCase) {
  const PxU32 calibratedFrames =
      testCase == eCASE_FORCE_LIMIT ? gForceLimitProbeTransientFrames
                                    : gMassProbeTransientFrames;
  const bool fixedDurationAccelerationLimit =
      testCase == eCASE_ACCELERATION_MODE &&
      gHeadlessConfig.lowForceLimit;
  if (!isContactComparisonTopology(gHeadlessConfig.topology) &&
      !fixedDurationAccelerationLimit)
    return calibratedFrames;
  const PxReal duration = PxReal(calibratedFrames) / 60.0f;
  return PxU32(PxFloor(duration / gHeadlessOptions.dt + 0.5f));
}

static bool createComparisonPair(PxU32 pairIndex, const PxVec3 &origin,
                                 PxReal mass, PxReal damping,
                                 PxReal targetVelocity, PxReal forceLimit,
                                 bool accelerationDrive, bool dynamicPair,
                                 bool contactPair, bool reverse) {
  if (pairIndex >= gMaxDrivePairs)
    return false;

  DrivePairRuntime &pair = gPairs[pairIndex];
  gPairCount = PxMax(gPairCount, pairIndex + 1u);
  const PxTransform pose(origin);
  if (dynamicPair) {
    const PxQuat rotationA = contactPair
                                 ? PxGetRotYQuat(-PxPi / 4.0f)
                                 : PxGetRotZQuat(-PxPi / 4.0f);
    const PxQuat rotationB =
        contactPair ? PxGetRotYQuat(PxPi / 4.0f) : PxQuat(PxIdentity);
    const PxVec3 positionA = origin;
    const PxVec3 positionB = origin;
    PxRigidDynamic *physicalA =
        contactPair
            ? PxCreateDynamic(*gPhysics, PxTransform(positionA, rotationA),
                              PxSphereGeometry(0.5f), *gMaterial,
                              1.0f)
            : gPhysics->createRigidDynamic(
                  PxTransform(positionA, rotationA));
    PxRigidDynamic *physicalB =
        contactPair
            ? PxCreateDynamic(*gPhysics, PxTransform(positionB, rotationB),
                              PxSphereGeometry(0.5f), *gMaterial,
                              1.0f)
            : gPhysics->createRigidDynamic(
                  PxTransform(positionB, rotationB));
    if (!physicalA || !physicalB) {
      PX_RELEASE(physicalA);
      PX_RELEASE(physicalB);
      return false;
    }
    physicalA->setMass(mass);
    physicalA->setMassSpaceInertiaTensor(PxVec3(mass));
    physicalB->setMass(mass);
    physicalB->setMassSpaceInertiaTensor(PxVec3(mass));
    physicalA->setLinearDamping(0.0f);
    physicalA->setAngularDamping(0.0f);
    physicalB->setLinearDamping(0.0f);
    physicalB->setAngularDamping(0.0f);
    physicalA->setSolverIterationCounts(4, 1);
    physicalB->setSolverIterationCounts(4, 1);
    if (contactPair) {
      gContactBodyTags[pairIndex][0].lane = pairIndex;
      gContactBodyTags[pairIndex][0].body = 0;
      gContactBodyTags[pairIndex][0].ground = false;
      gContactBodyTags[pairIndex][1].lane = pairIndex;
      gContactBodyTags[pairIndex][1].body = 1;
      gContactBodyTags[pairIndex][1].ground = false;
      physicalA->userData = &gContactBodyTags[pairIndex][0];
      physicalB->userData = &gContactBodyTags[pairIndex][1];
      PxShape *shapeA = NULL;
      PxShape *shapeB = NULL;
      if (physicalA->getShapes(&shapeA, 1) != 1 ||
          physicalB->getShapes(&shapeB, 1) != 1 || !shapeA || !shapeB) {
        PX_RELEASE(physicalA);
        PX_RELEASE(physicalB);
        return false;
      }
      PxFilterData bodyFilterData;
      bodyFilterData.word0 = 1u;
      shapeA->setSimulationFilterData(bodyFilterData);
      shapeB->setSimulationFilterData(bodyFilterData);
    }
    if (!setupActor(physicalA) || !setupActor(physicalB)) {
      PX_RELEASE(physicalA);
      PX_RELEASE(physicalB);
      return false;
    }

    const PxTransform localA(
        contactPair ? PxGetRotYQuat(PxPi / 4.0f) : PxQuat(PxIdentity));
    const PxTransform localB(
        contactPair
            ? PxGetRotYQuat(-PxPi / 4.0f)
            : PxQuat(2.0f * PxPi / 3.0f,
                     PxVec3(1.0f, 1.0f, 1.0f).getNormalized()));
    pair.actorA = reverse ? static_cast<PxRigidActor *>(physicalB)
                          : static_cast<PxRigidActor *>(physicalA);
    pair.dynamicActorA = reverse ? physicalB : physicalA;
    pair.actorB = reverse ? physicalA : physicalB;
    pair.jointFrameA = reverse ? localB : localA;
    pair.jointFrameB = reverse ? localA : localB;
  } else {
    pair.actorA = gPhysics->createRigidStatic(pose);
    if (!pair.actorA)
      return false;
    if (!setupActor(pair.actorA)) {
      PX_RELEASE(pair.actorA);
      return false;
    }
    pair.actorB = gPhysics->createRigidDynamic(pose);
    if (!pair.actorB) {
      PX_RELEASE(pair.actorA);
      return false;
    }
    pair.actorB->setMass(mass);
    pair.actorB->setMassSpaceInertiaTensor(PxVec3(mass));
    pair.actorB->setLinearDamping(0.0f);
    pair.actorB->setAngularDamping(0.0f);
    pair.actorB->setSolverIterationCounts(4, 1);
    if (!setupActor(pair.actorB)) {
      PX_RELEASE(pair.actorB);
      PX_RELEASE(pair.actorA);
      return false;
    }
  }

  pair.joint = PxD6JointCreate(*gPhysics, pair.actorA, pair.jointFrameA,
                               pair.actorB, pair.jointFrameB);
  if (!pair.joint)
    return false;
  pair.joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
  pair.joint->setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
  pair.joint->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
  pair.joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  pair.joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  pair.joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
  pair.joint->setDrive(
      PxD6Drive::eX,
      PxD6JointDrive(0.0f, damping, forceLimit, accelerationDrive));
  pair.joint->setDriveVelocity(PxVec3(targetVelocity, 0.0f, 0.0f),
                               PxVec3(0.0f), true);
  pair.joint->setConstraintFlag(
      PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);

  pair.initialPoseA = pair.actorA->getGlobalPose();
  pair.initialPoseB = pair.actorB->getGlobalPose();
  pair.signedWorldAxis =
      (pair.initialPoseA.q * pair.jointFrameA.q)
          .rotate(PxVec3(1.0f, 0.0f, 0.0f))
          .getNormalized();

  const PxD6JointDrive readback = pair.joint->getDrive(PxD6Drive::eX);
  if (pair.dynamicActorA) {
    gComparisonMetrics.actorAMassReadback[pairIndex] =
        pair.dynamicActorA->getMass();
    gComparisonMetrics.actorAInertiaReadback[pairIndex] =
        pair.dynamicActorA->getMassSpaceInertiaTensor();
  }
  gComparisonMetrics.massReadback[pairIndex] = pair.actorB->getMass();
  gComparisonMetrics.inertiaReadback[pairIndex] =
      pair.actorB->getMassSpaceInertiaTensor();
  gComparisonMetrics.dampingReadback[pairIndex] = readback.damping;
  gComparisonMetrics.forceLimitReadback[pairIndex] = readback.forceLimit;
  gComparisonMetrics.accelerationFlagReadback[pairIndex] =
      readback.flags.isSet(PxD6JointDriveFlag::eACCELERATION);
  gComparisonMetrics.driveLimitsAreForcesReadback[pairIndex] =
      pair.joint->getConstraintFlags().isSet(
          PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES);
  gComparisonMetrics.finiteReadback[pairIndex] =
      PxIsFinite(gComparisonMetrics.actorAMassReadback[pairIndex]) &&
      gComparisonMetrics.actorAInertiaReadback[pairIndex].isFinite() &&
      PxIsFinite(gComparisonMetrics.massReadback[pairIndex]) &&
      gComparisonMetrics.inertiaReadback[pairIndex].isFinite() &&
      readback.isValid();
  if (dynamicPair) {
    gComparisonMetrics.initialCenterOfMass[pairIndex] =
        (pair.initialPoseA.p + pair.initialPoseB.p) * 0.5f;
  }
  return true;
}

static bool createComparisonPairs() {
  const bool massProbe = gHeadlessCase == eCASE_MASS_SCALING ||
                         gHeadlessCase == eCASE_ACCELERATION_MODE;
  const bool accelerationModeProbe =
      gHeadlessCase == eCASE_ACCELERATION_MODE;
  const bool accelerationLimitProbe =
      accelerationModeProbe && gHeadlessConfig.lowForceLimit;
  const PxReal damping =
      massProbe ? gMassProbeDamping : gForceLimitProbeDamping;
  const PxReal targetVelocity =
      massProbe ? gMassProbeTargetVelocity : gForceLimitProbeTargetVelocity;
  const PxReal referenceMass = gMassProbeReferenceMass;
  const PxReal testMass = massProbe ? gHeadlessConfig.comparisonMass
                                    : gMassProbeReferenceMass;
  const PxReal testLimit =
      accelerationLimitProbe
          ? gAccelerationLimitProbeLowLimit
          : massProbe
          ? FLT_MAX
          : (gHeadlessConfig.lowForceLimit ? gForceLimitProbeLowLimit
                                           : gForceLimitProbeHighLimit);
  const bool dynamicPair =
      isDynamicComparisonTopology(gHeadlessConfig.topology);
  const bool contactPair =
      isContactComparisonTopology(gHeadlessConfig.topology);
  const bool reverse =
      gHeadlessConfig.endpoint == eENDPOINT_REVERSE;

  const PxVec3 referenceOrigin =
      contactPair ? PxVec3(-50.0f, 0.5f, 0.0f) : PxVec3(0.0f);
  const PxVec3 testOrigin =
      contactPair ? PxVec3(50.0f, 0.5f, 0.0f)
                  : PxVec3(0.0f, 4.0f, 0.0f);
  if (!createComparisonPair(0, referenceOrigin,
                            referenceMass, damping, targetVelocity,
                            massProbe ? FLT_MAX : gForceLimitProbeHighLimit,
                            accelerationModeProbe, dynamicPair, contactPair,
                            reverse) ||
      !createComparisonPair(1, testOrigin, testMass, damping,
                            targetVelocity, testLimit,
                            accelerationModeProbe, dynamicPair, contactPair,
                            reverse))
    return false;

  gMetrics.initialPoseA = gPairs[0].initialPoseA;
  gMetrics.initialPoseB = gPairs[0].initialPoseB;
  gMetrics.previousPoseA = gPairs[0].initialPoseA;
  gMetrics.previousPoseB = gPairs[0].initialPoseB;
  gMetrics.signedWorldAxis = gPairs[0].signedWorldAxis;
  gMetrics.targetMagnitude = targetVelocity;
  gMetrics.pairCountWitness = gPairCount;
  return true;
}

static bool createDrivePair(bool interactive) {
  const JointDriveKind drive = interactive ? getInteractiveDrive()
                                            : gHeadlessConfig.drive;
  const bool actorAKinematic =
      interactive ? gChangeObjectAType : gHeadlessConfig.actorAKinematic;
  const JointFrameOrientation frameAOrientation =
      interactive ? (gChangeJointFrameARotation ? eFRAME_ROTZ_NEG45
                                                : eFRAME_IDENTITY)
                  : gHeadlessConfig.frameAOrientation;
  const JointFrameOrientation frameBOrientation =
      interactive ? (gChangeJointFrameBRotation ? eFRAME_ROTZ_NEG45
                                                : eFRAME_IDENTITY)
                  : gHeadlessConfig.frameBOrientation;
  const bool bodyBRotated = interactive ? gChangeObjectBRotation
                                        : gHeadlessConfig.bodyBRotated;
  const bool positionProbe =
      !interactive && isPositionLikeCase(gHeadlessCase);
  const bool contactPositionPair =
      positionProbe &&
      isContactComparisonTopology(gHeadlessConfig.topology);
  const bool dynamicPositionPair =
      positionProbe &&
      isDynamicComparisonTopology(gHeadlessConfig.topology);
  const bool reversePosition =
      positionProbe && gHeadlessConfig.endpoint == eENDPOINT_REVERSE;
  const PxQuat rotZ = PxGetRotZQuat(-PxPi / 4.0f);
  const PxQuat rotX(-PxPi / 4.0f, PxVec3(1.0f, 0.0f, 0.0f));
  const PxBoxGeometry boxGeometry(0.5f, 0.5f, 0.5f);
  PxTransform poseA(
      interactive
          ? PxVec3(0.0f, 2.0f, -20.0f)
          : (contactPositionPair ? PxVec3(0.0f, 0.5f, 0.0f)
                                 : PxVec3(0.0f)));
  DrivePairRuntime &pair = gPairs[0];
  gPairCount = 1;

  if (dynamicPositionPair) {
    PxRigidDynamic *actor =
        contactPositionPair
            ? PxCreateDynamic(
                  *gPhysics, poseA, PxSphereGeometry(0.5f),
                  *gMaterial, 1.0f)
            : gPhysics->createRigidDynamic(poseA);
    if (!actor)
      return false;
    actor->setMass(1.0f);
    actor->setMassSpaceInertiaTensor(PxVec3(1.0f));
    actor->setLinearDamping(0.0f);
    actor->setAngularDamping(0.0f);
    actor->setSolverIterationCounts(4, 1);
    pair.actorA = actor;
    pair.dynamicActorA = actor;
    if (contactPositionPair &&
        !configureContactBody(*actor, 0, 0))
      return false;
    gPositionMetrics.actorAMassReadback = actor->getMass();
    gPositionMetrics.actorAInertiaReadback =
        actor->getMassSpaceInertiaTensor();
  } else if (actorAKinematic) {
    PxRigidDynamic *actor = positionProbe
                                ? gPhysics->createRigidDynamic(poseA)
                                : PxCreateDynamic(*gPhysics, poseA, boxGeometry,
                                                  *gMaterial, 1.0f);
    if (!actor)
      return false;
    actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
    pair.actorA = actor;
    pair.dynamicActorA = actor;
  } else {
    pair.actorA = positionProbe
                      ? gPhysics->createRigidStatic(poseA)
                      : PxCreateStatic(*gPhysics, poseA, boxGeometry, *gMaterial);
  }
  if (!setupActor(pair.actorA))
    return false;

  if (frameAOrientation == eFRAME_ROTZ_NEG45)
    pair.jointFrameA.q = rotZ;
  else if (frameAOrientation == eFRAME_ROTX_NEG45)
    pair.jointFrameA.q = rotX;
  if (frameBOrientation == eFRAME_ROTZ_NEG45)
    pair.jointFrameB.q = rotZ;
  else if (frameBOrientation == eFRAME_ROTX_NEG45)
    pair.jointFrameB.q = rotX;
  if (positionProbe && gHeadlessConfig.offsetAnchor) {
    // Give both endpoints the same joint-Y lever arm.  The finite drive force
    // therefore needs a locked-angular reaction torque.  Since drive rows
    // report about bodyAWorldOffset, toggling eOUTPUT_FORCE must not invent a
    // second COM moment on top of that physical reaction.
    pair.jointFrameA.p = pair.jointFrameA.q.rotate(
        PxVec3(0.0f, gPositionOffsetAnchorMagnitude, 0.0f));
    pair.jointFrameB.p = pair.jointFrameA.p;
  }

  PxTransform poseB = poseA;
  if (positionProbe && (gHeadlessConfig.initialRelativeOffset ||
                        gHeadlessConfig.offsetAnchor ||
                        gHeadlessCase == eCASE_ANGULAR_POSITION)) {
    PxTransform initialRelative(PxIdentity);
    const PxVec3 targetAxis = getPositionTargetAxis(drive);
    if (gHeadlessConfig.initialRelativeOffset && isLinearDrive(drive))
      initialRelative.p = targetAxis * gPositionInitialRelativeMagnitude;
    else if (gHeadlessConfig.initialRelativeOffset)
      initialRelative.q =
          PxQuat(gPositionInitialRelativeMagnitude, targetAxis);
    poseB = poseA * pair.jointFrameA * initialRelative *
            pair.jointFrameB.getInverse();
  } else if (!positionProbe) {
    poseB.p.x += boxGeometry.halfExtents.x * 2.0f;
    if (bodyBRotated)
      poseB.q = rotZ;
  }
  pair.actorB =
      contactPositionPair
          ? PxCreateDynamic(
                *gPhysics, poseB, PxSphereGeometry(0.5f),
                *gMaterial, 1.0f)
          : PxCreateDynamic(
                *gPhysics, poseB, boxGeometry, *gMaterial, 1.0f);
  if (positionProbe && pair.actorB) {
    pair.actorB->setMass(gHeadlessConfig.comparisonMass);
    pair.actorB->setMassSpaceInertiaTensor(
        PxVec3(gHeadlessConfig.comparisonMass));
    pair.actorB->setLinearDamping(0.0f);
    pair.actorB->setAngularDamping(0.0f);
    pair.actorB->setSolverIterationCounts(4, 1);
    gPositionMetrics.massReadback = pair.actorB->getMass();
    gPositionMetrics.inertiaReadback =
        pair.actorB->getMassSpaceInertiaTensor();
  }
  if (contactPositionPair && pair.actorB &&
      !configureContactBody(*pair.actorB, 0, 1))
    return false;
  if (!setupActor(pair.actorB))
    return false;

  pair.joint = reversePosition
                   ? PxD6JointCreate(*gPhysics, pair.actorB, pair.jointFrameB,
                                     pair.actorA, pair.jointFrameA)
                   : PxD6JointCreate(*gPhysics, pair.actorA, pair.jointFrameA,
                                     pair.actorB, pair.jointFrameB);
  if (!pair.joint)
    return false;
  if (positionProbe) {
    PxRigidActor *jointActor0 = NULL;
    PxRigidActor *jointActor1 = NULL;
    pair.joint->getActors(jointActor0, jointActor1);
    gPositionMetrics.actorOrderValid =
        reversePosition
            ? jointActor0 == pair.actorB && jointActor1 == pair.actorA
            : jointActor0 == pair.actorA && jointActor1 == pair.actorB;
  }
  pair.joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSWING_TWIST);
  pair.joint->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);

  if (!positionProbe) {
    pair.joint->setMotion(PxD6Axis::eX, PxD6Motion::eFREE);
    pair.joint->setMotion(PxD6Axis::eY, PxD6Motion::eFREE);
    pair.joint->setMotion(PxD6Axis::eZ, PxD6Motion::eFREE);
    pair.joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    pair.joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
    pair.joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  }

  // Preserve the upstream interactive configuration. The headless gate is
  // deliberately force-mode velocity only; acceleration semantics are a
  // separate PARTIAL/PROBE because external AVBD D6 prep currently drops the
  // acceleration flag.
  if (positionProbe) {
    if (!configurePositionDrive(pair, drive))
      return false;
    if (gHeadlessCase == eCASE_OUTPUT_FORCE) {
      if (gHeadlessConfig.breakMode == eBREAK_BELOW_DRIVE_LIMIT)
        pair.joint->setBreakForce(gOutputForceBreakBelow, PX_MAX_F32);
      else if (gHeadlessConfig.breakMode == eBREAK_ABOVE_DRIVE_LIMIT)
        pair.joint->setBreakForce(gOutputForceBreakAbove, PX_MAX_F32);
      else if (gHeadlessConfig.breakMode == eBREAK_BELOW_OFFSET_MOMENT)
        pair.joint->setBreakForce(PX_MAX_F32, gOutputMomentBreakBelow);
      else if (gHeadlessConfig.breakMode == eBREAK_ABOVE_OFFSET_MOMENT)
        pair.joint->setBreakForce(PX_MAX_F32, gOutputMomentBreakAbove);
      pair.joint->getBreakForce(
          gPositionMetrics.linearBreakForceReadback,
          gPositionMetrics.angularBreakForceReadback);
    }
  } else {
    configureDrive(*pair.joint, drive, interactive);
  }

  if (!interactive) {
    pair.initialPoseA = pair.actorA->getGlobalPose();
    pair.initialPoseB = pair.actorB->getGlobalPose();
    pair.expectedPoseA = pair.initialPoseA;
    gMetrics.initialPoseA = pair.actorA->getGlobalPose();
    gMetrics.initialPoseB = pair.actorB->getGlobalPose();
    gMetrics.previousPoseA = gMetrics.initialPoseA;
    gMetrics.previousPoseB = gMetrics.initialPoseB;
    gMetrics.targetMagnitude =
        positionProbe ? gPositionTargetMagnitude : gTargetVelocity;
    PxVec3 worldAxis =
        (gMetrics.initialPoseA.q * pair.jointFrameA.q)
            .rotate(positionProbe ? getPositionTargetAxis(drive)
                                  : getLocalDriveAxis(drive));
    const PxReal axisMagnitude = safeMagnitude(worldAxis);
    if (axisMagnitude <= 1e-6f || axisMagnitude >= PX_MAX_F32)
      return false;
    worldAxis *= 1.0f / axisMagnitude;
    if (gHeadlessCase == eCASE_ANGULAR_POSITION) {
      PxVec3 worldFrameBAxis =
          (gMetrics.initialPoseB.q * pair.jointFrameB.q)
              .rotate(getPositionTargetAxis(drive));
      const PxReal worldFrameBMagnitude = safeMagnitude(worldFrameBAxis);
      PxVec3 wrongRawFrameAxis =
          pair.jointFrameB.q.rotate(getPositionTargetAxis(drive));
      const PxReal wrongRawFrameMagnitude = safeMagnitude(wrongRawFrameAxis);
      if (worldFrameBMagnitude <= 1e-6f ||
          worldFrameBMagnitude >= PX_MAX_F32 ||
          wrongRawFrameMagnitude <= 1e-6f ||
          wrongRawFrameMagnitude >= PX_MAX_F32)
        return false;
      worldFrameBAxis *= 1.0f / worldFrameBMagnitude;
      wrongRawFrameAxis *= 1.0f / wrongRawFrameMagnitude;
      gPositionMetrics.worldFrameAxisDot =
          worldAxis.dot(worldFrameBAxis);
      gPositionMetrics.wrongRawFrameAxisDot =
          worldAxis.dot(wrongRawFrameAxis);
      const bool rawFrameSeparated =
          drive == eDRIVE_SLERP
              ? (gPositionMetrics.wrongRawFrameAxisDot > 0.72f &&
                 gPositionMetrics.wrongRawFrameAxisDot < 0.74f)
              : (gPositionMetrics.wrongRawFrameAxisDot > 0.70f &&
                 gPositionMetrics.wrongRawFrameAxisDot < 0.72f);
      gPositionMetrics.angularFrameWitnessValid =
          gPositionMetrics.worldFrameAxisDot >= 0.99999f &&
          rawFrameSeparated;
    }
    gMetrics.signedWorldAxis =
        positionProbe ? worldAxis : worldAxis * getRelativeTargetSign(drive);
    pair.signedWorldAxis = gMetrics.signedWorldAxis;
    if (positionProbe) {
      PxRigidActor *actor0 = NULL;
      PxRigidActor *actor1 = NULL;
      pair.joint->getActors(actor0, actor1);
      PX_UNUSED(actor1);
      const PxTransform &actor0Frame =
          reversePosition ? pair.jointFrameB : pair.jointFrameA;
      gPositionMetrics.actor0WorldArm =
          actor0->getGlobalPose().q.rotate(actor0Frame.p);
      gPositionMetrics.dynamicWorldArm =
          pair.actorB->getGlobalPose().q.rotate(pair.jointFrameB.p);
      // The linear drive acts at the dynamic anchor.  With angular DOFs
      // locked, the joint supplies the opposite angular reaction.  Public
      // torque is normalized across actor ordering with actor0Sign, so the
      // expected reaction is -r_dynamic x F_dynamic for both endpoints.
      gPositionMetrics.expectedNormalizedPublicTorque =
          -gPositionMetrics.dynamicWorldArm.cross(
              gMetrics.signedWorldAxis * gPositionLowForceLimit);
      if (dynamicPositionPair) {
        const PxReal totalMass =
            gPositionMetrics.actorAMassReadback +
            gPositionMetrics.massReadback;
        if (!(totalMass > 0.0f) || !PxIsFinite(totalMass))
          return false;
        gPositionMetrics.initialCenterOfMass =
            (gMetrics.initialPoseA.p *
                 gPositionMetrics.actorAMassReadback +
             gMetrics.initialPoseB.p * gPositionMetrics.massReadback) /
            totalMass;
        if (gHeadlessConfig.lowForceLimit &&
            gHeadlessCase == eCASE_ANGULAR_POSITION) {
          if (drive == eDRIVE_SLERP) {
            PxRigidActor *slerpActor0 = NULL;
            PxRigidActor *slerpActor1 = NULL;
            pair.joint->getActors(slerpActor0, slerpActor1);
            if (!slerpActor0 || !slerpActor1)
              return false;
            const PxTransform &localFrame0 =
                reversePosition ? pair.jointFrameB : pair.jointFrameA;
            const PxTransform &localFrame1 =
                reversePosition ? pair.jointFrameA : pair.jointFrameB;
            PxQuat frame0 =
                slerpActor0->getGlobalPose().q * localFrame0.q;
            PxQuat frame1 =
                slerpActor1->getGlobalPose().q * localFrame1.q;
            frame0.normalize();
            frame1.normalize();
            PxQuat currentRelative = frame0.getConjugate() * frame1;
            currentRelative.normalize();
            PxQuat targetRelative = pair.joint->getDrivePosition().q;
            if (currentRelative.dot(targetRelative) < 0.0f)
              targetRelative = -targetRelative;
            const PxQuat delta =
                targetRelative.getConjugate() * currentRelative;
            PxVec3 rows[3];
            computeSlerpJacobianAxes(rows, frame0 * targetRelative,
                                     frame1);
            PxVec3 actor0Torque(0.0f);
            for (PxU32 rowIndex = 0; rowIndex < 3; ++rowIndex) {
              const PxReal rawTorque =
                  gPositionDriveStiffness * (&delta.x)[rowIndex];
              actor0Torque +=
                  rows[rowIndex] *
                  PxClamp(rawTorque, -gPositionLowForceLimit,
                          gPositionLowForceLimit);
            }
            const PxReal inertia0 =
                reversePosition ? gPositionMetrics.inertiaReadback.x
                                : gPositionMetrics.actorAInertiaReadback.x;
            const PxReal inertia1 =
                reversePosition
                    ? gPositionMetrics.actorAInertiaReadback.x
                    : gPositionMetrics.inertiaReadback.x;
            const PxVec3 acceleration0 = actor0Torque / inertia0;
            const PxVec3 acceleration1 = -actor0Torque / inertia1;
            const PxVec3 accelerationA =
                reversePosition ? acceleration1 : acceleration0;
            const PxVec3 accelerationB =
                reversePosition ? acceleration0 : acceleration1;
            gPositionMetrics.expectedSignedAngularAccelerationA =
                accelerationA.dot(gMetrics.signedWorldAxis);
            gPositionMetrics.expectedSignedAngularAccelerationB =
                accelerationB.dot(gMetrics.signedWorldAxis);
            gPositionMetrics.expectedFirstRelativeAcceleration =
                safeMagnitude(accelerationB - accelerationA);
          } else {
            gPositionMetrics.expectedSignedAngularAccelerationA =
                -gPositionLowForceLimit /
                gPositionMetrics.actorAInertiaReadback.x;
            gPositionMetrics.expectedSignedAngularAccelerationB =
                gPositionLowForceLimit /
                gPositionMetrics.inertiaReadback.x;
            gPositionMetrics.expectedFirstRelativeAcceleration =
                gPositionLowForceLimit *
                (1.0f / gPositionMetrics.actorAInertiaReadback.x +
                 1.0f / gPositionMetrics.inertiaReadback.x);
          }
        }
      }
    }
    gMetrics.pairCountWitness = gPairCount;
  }
  return true;
}

static bool createLegacyAngularLimitConeFixture() {
  const bool inside =
      gHeadlessCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE;
  const PxReal initialSwingY =
      inside ? gLegacyConeInsideY : gLegacyConeOutsideY;
  const PxReal initialSwingZ =
      inside ? gLegacyConeInsideZ : gLegacyConeOutsideZ;

  DrivePairRuntime &pair = gPairs[0];
  pair.initialPoseA = PxTransform(PxIdentity);
  pair.initialPoseB =
      PxTransform(PxVec3(0.0f),
                  makeLegacyConeSwing(initialSwingY, initialSwingZ));
  pair.expectedPoseA = pair.initialPoseA;
  pair.jointFrameA = PxTransform(PxIdentity);
  pair.jointFrameB = PxTransform(PxIdentity);

  PxRigidStatic *actorA =
      gPhysics->createRigidStatic(pair.initialPoseA);
  PxRigidDynamic *actorB = PxCreateDynamic(
      *gPhysics, pair.initialPoseB,
      PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
  if (!actorA || !actorB) {
    PX_RELEASE(actorA);
    PX_RELEASE(actorB);
    return false;
  }
  actorB->setMass(1.0f);
  actorB->setMassSpaceInertiaTensor(PxVec3(1.0f));
  actorB->setLinearDamping(0.0f);
  actorB->setAngularDamping(0.0f);
  actorB->setMaxAngularVelocity(100.0f);
  if (!setupActor(actorA) || !setupActor(actorB))
    return false;

  PxD6Joint *joint =
      PxD6JointCreate(*gPhysics, actorA, pair.jointFrameA,
                      actorB, pair.jointFrameB);
  if (!joint)
    return false;
  pair.joint = joint;
  pair.actorA = actorA;
  pair.actorB = actorB;
  pair.dynamicActorA = NULL;
  gPairCount = 1;

  joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
  joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLIMITED);
  joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLIMITED);
  joint->setSwingLimit(
      PxJointLimitCone(gLegacyConeLimitY, gLegacyConeLimitZ));

  const PxJointLimitCone limitReadback = joint->getSwingLimit();
  gAngularLimitMetrics.limitYReadback = limitReadback.yAngle;
  gAngularLimitMetrics.limitZReadback = limitReadback.zAngle;
  gAngularLimitMetrics.twistMotionReadback =
      joint->getMotion(PxD6Axis::eTWIST);
  gAngularLimitMetrics.swing1MotionReadback =
      joint->getMotion(PxD6Axis::eSWING1);
  gAngularLimitMetrics.swing2MotionReadback =
      joint->getMotion(PxD6Axis::eSWING2);
  gAngularLimitMetrics.initialConeAngle =
      computeJointFrameConeAngle(pair.initialPoseA, pair.initialPoseB, pair);
  gAngularLimitMetrics.initialEllipseRadius =
      computeLegacyConeEllipseRadius(
          pair.initialPoseA, pair.initialPoseB, pair,
          gLegacyConeLimitY, gLegacyConeLimitZ);
  gMetrics.targetMagnitude = 1.0f;
  gMetrics.pairCountWitness = gPairCount;
  return true;
}

static void createScene() {
  releaseSceneState();

  if (!gPhysics || !gMaterial) {
    gInitializationFailed = true;
    return;
  }

  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  const bool contactComparison =
      gHeadlessMode &&
      isContactComparisonTopology(gHeadlessConfig.topology);
  sceneDesc.gravity =
      contactComparison ? PxVec3(0.0f, -9.81f, 0.0f) : PxVec3(0.0f);
  if (gHeadlessMode && gHeadlessCase != eCASE_VELOCITY)
    sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
  gDispatcher = PxDefaultCpuDispatcherCreate(
      gHeadlessMode ? gHeadlessOptions.dispatcherThreads : 2u);
  if (!gDispatcher) {
    gInitializationFailed = true;
    return;
  }
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = contactComparison
                               ? contactComparisonFilterShader
                               : PxDefaultSimulationFilterShader;
  sceneDesc.simulationEventCallback =
      contactComparison ? &gContactCallback : NULL;
  sceneDesc.solverType = gSolverType;
#if PX_SUPPORT_GPU_PHYSX
  if (!gHeadlessMode && gUseGPU) {
    sceneDesc.cudaContextManager = gCudaContextManager;
    sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    sceneDesc.gpuMaxNumPartitions = 8;
  }
#endif
  gScene = gPhysics->createScene(sceneDesc);
  if (!gScene) {
    gInitializationFailed = true;
    return;
  }

  if (!gHeadlessMode) {
    gScene->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0f);
    gScene->setVisualizationParameter(
        PxVisualizationParameter::eJOINT_LOCAL_FRAMES, 1.0f);
    PxPvdSceneClient *pvdClient = gScene->getScenePvdClient();
    if (pvdClient) {
      pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS,
                                 true);
      pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
      pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES,
                                 true);
    }
  }

  if (!gHeadlessMode || contactComparison) {
    PxRigidStatic *groundPlane =
        PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
    if (!groundPlane) {
      gInitializationFailed = true;
      return;
    }
    if (contactComparison) {
      gContactGroundTag.lane = PX_MAX_U32;
      gContactGroundTag.body = PX_MAX_U32;
      gContactGroundTag.ground = true;
      groundPlane->userData = &gContactGroundTag;
    }
    gScene->addActor(*groundPlane);
  }

  const bool created =
      gHeadlessMode && isLegacyAngularLimitCase(gHeadlessCase)
          ? createLegacyAngularLimitConeFixture()
          : gHeadlessMode && isOrderingCase(gHeadlessCase)
          ? createOrderingDrive()
          : (gHeadlessMode && isComparisonCase(gHeadlessCase)
                 ? createComparisonPairs()
                 : createDrivePair(!gHeadlessMode));
  if (!created)
    gInitializationFailed = true;
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

#if PX_SUPPORT_GPU_PHYSX
  if (interactive) {
    PxCudaContextManagerDesc cudaContextManagerDesc;
    gCudaContextManager = PxCreateCudaContextManager(
        *gFoundation, cudaContextManagerDesc, PxGetProfilerCallback());
    if (gCudaContextManager && !gCudaContextManager->contextIsValid())
      PX_RELEASE(gCudaContextManager);
  }
#endif

  const bool contactComparison =
      !interactive &&
      isContactComparisonTopology(gHeadlessConfig.topology);
  gMaterial = contactComparison
                  ? gPhysics->createMaterial(0.0f, 0.0f, 0.0f)
                  : gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
  if (!gMaterial) {
    gInitializationFailed = true;
    return;
  }
  createScene();
}

struct PrimaryPairSample {
  PxTransform poseA;
  PxTransform poseB;
  PxVec3 linearA;
  PxVec3 angularA;
  PxVec3 linearB;
  PxVec3 angularB;

  PrimaryPairSample()
      : poseA(PxIdentity), poseB(PxIdentity), linearA(0.0f), angularA(0.0f),
        linearB(0.0f), angularB(0.0f) {}
};

static bool updateHeadlessKinematicTarget() {
  if (!isMovingKinematicPositionFixture())
    return true;
  DrivePairRuntime &pair = gPairs[0];
  if (!pair.dynamicActorA)
    return false;

  const PxReal nextTime =
      PxReal(gMetrics.completedFrames + 1u) * gHeadlessOptions.dt;
  const PxReal motionTime =
      PxMax(0.0f, nextTime - gKinematicMotionStartTime);
  pair.expectedPoseA = pair.initialPoseA;
  pair.expectedPoseA.q =
      (PxQuat(gKinematicAngularSpeed * motionTime,
              PxVec3(0.0f, 1.0f, 0.0f)) *
       pair.initialPoseA.q)
          .getNormalized();
  if (!pair.expectedPoseA.isValid())
    return false;
  pair.dynamicActorA->setKinematicTarget(pair.expectedPoseA);
  gPositionMetrics.kinematicTargetFrames++;
  if (motionTime > 0.0f)
    gPositionMetrics.kinematicMotionFrames++;
  return true;
}

static bool gatherPairSample(PxU32 pairIndex, PrimaryPairSample &sample) {
  if (pairIndex >= gPairCount) {
    gMetrics.nonFinite = true;
    return false;
  }
  const DrivePairRuntime &pair = gPairs[pairIndex];
  if (!pair.actorA || !pair.actorB) {
    gMetrics.nonFinite = true;
    return false;
  }

  sample.poseA = pair.actorA->getGlobalPose();
  sample.poseB = pair.actorB->getGlobalPose();
  sample.linearA = pair.dynamicActorA ? pair.dynamicActorA->getLinearVelocity()
                                      : PxVec3(0.0f);
  sample.angularA = pair.dynamicActorA ? pair.dynamicActorA->getAngularVelocity()
                                       : PxVec3(0.0f);
  sample.linearB = pair.actorB->getLinearVelocity();
  sample.angularB = pair.actorB->getAngularVelocity();

  if (!sample.poseA.isValid() || !sample.poseB.isValid() ||
      !sample.linearA.isFinite() || !sample.angularA.isFinite() ||
      !sample.linearB.isFinite() || !sample.angularB.isFinite()) {
    gMetrics.nonFinite = true;
    return false;
  }

  gMetrics.maxQuaternionNormError =
      PxMax(gMetrics.maxQuaternionNormError,
            PxMax(PxAbs(sample.poseA.q.magnitudeSquared() - 1.0f),
                  PxAbs(sample.poseB.q.magnitudeSquared() - 1.0f)));
  gMetrics.maxAbsPosition =
      PxMax(gMetrics.maxAbsPosition,
            PxMax(maxAbsComponent(sample.poseA.p),
                  maxAbsComponent(sample.poseB.p)));
  gMetrics.maxLinearSpeed =
      PxMax(gMetrics.maxLinearSpeed,
            PxMax(safeMagnitude(sample.linearA),
                  safeMagnitude(sample.linearB)));
  gMetrics.maxAngularSpeed =
      PxMax(gMetrics.maxAngularSpeed,
            PxMax(safeMagnitude(sample.angularA),
                  safeMagnitude(sample.angularB)));
  const PxTransform expectedPoseA =
      pairIndex == 0 && isMovingKinematicPositionFixture()
          ? pair.expectedPoseA
          : pair.initialPoseA;
  gMetrics.maxActorAPositionError =
      PxMax(gMetrics.maxActorAPositionError,
            safeMagnitude(sample.poseA.p - expectedPoseA.p));
  gMetrics.maxActorAAngleError =
      PxMax(gMetrics.maxActorAAngleError,
            quaternionAngle(sample.poseA.q, expectedPoseA.q));
  if (pairIndex == 0 && isMovingKinematicPositionFixture()) {
    const bool moving =
        PxReal(gMetrics.completedFrames) * gHeadlessOptions.dt >
        gKinematicMotionStartTime;
    const PxVec3 expectedAngularVelocity =
        moving ? PxVec3(0.0f, gKinematicAngularSpeed, 0.0f)
               : PxVec3(0.0f);
    gPositionMetrics.maximumKinematicAngularSpeedError =
        PxMax(gPositionMetrics.maximumKinematicAngularSpeedError,
              safeMagnitude(sample.angularA -
                            expectedAngularVelocity));
    gPositionMetrics.finalKinematicTravel =
        quaternionAngle(pair.initialPoseA.q, sample.poseA.q);
  }
  return true;
}

static void sampleLegacyAngularLimitState() {
  PrimaryPairSample sample;
  if (!gatherPairSample(0, sample))
    return;
  const PxReal coneAngle =
      computeJointFrameConeAngle(sample.poseA, sample.poseB, gPairs[0]);
  const PxReal ellipseRadius =
      computeLegacyConeEllipseRadius(
          sample.poseA, sample.poseB, gPairs[0],
          gLegacyConeLimitY, gLegacyConeLimitZ);
  if (!PxIsFinite(coneAngle) || !PxIsFinite(ellipseRadius) ||
      ellipseRadius >= PX_MAX_F32) {
    gMetrics.nonFinite = true;
    return;
  }

  gAngularLimitMetrics.finalConeAngle = coneAngle;
  gAngularLimitMetrics.minimumConeAngle =
      PxMin(gAngularLimitMetrics.minimumConeAngle, coneAngle);
  gAngularLimitMetrics.maximumConeAngle =
      PxMax(gAngularLimitMetrics.maximumConeAngle, coneAngle);
  gAngularLimitMetrics.maximumInsideDeviation =
      PxMax(gAngularLimitMetrics.maximumInsideDeviation,
            PxAbs(coneAngle - gAngularLimitMetrics.initialConeAngle));
  gAngularLimitMetrics.finalEllipseRadius = ellipseRadius;
  gAngularLimitMetrics.maximumInsideEllipseDeviation =
      PxMax(gAngularLimitMetrics.maximumInsideEllipseDeviation,
            PxAbs(ellipseRadius -
                  gAngularLimitMetrics.initialEllipseRadius));
  if (gMetrics.completedFrames >
      gHeadlessOptions.frames - gLateWindowFrames) {
    gAngularLimitMetrics.maximumLateConeAngle =
        PxMax(gAngularLimitMetrics.maximumLateConeAngle, coneAngle);
    gAngularLimitMetrics.maximumLateEllipseRadius =
        PxMax(gAngularLimitMetrics.maximumLateEllipseRadius,
              ellipseRadius);
    gAngularLimitMetrics.lateSampleCount++;
  }
  gMetrics.finalSignedDisplacement =
      gAngularLimitMetrics.initialEllipseRadius - ellipseRadius;
  gMetrics.sampleCount++;
}

static void sampleVelocityState(const PrimaryPairSample &sample) {
  const PxVec3 relativeVelocity =
      isLinearDrive(gHeadlessConfig.drive)
          ? sample.linearB - sample.linearA
          : sample.angularB - sample.angularA;

  const PxReal relativeMagnitude = safeMagnitude(relativeVelocity);
  const PxReal projection = relativeVelocity.dot(gMetrics.signedWorldAxis);
  const PxVec3 orthogonal =
      relativeVelocity - gMetrics.signedWorldAxis * projection;
  const PxReal orthogonalMagnitude = safeMagnitude(orthogonal);
  if (!PxIsFinite(projection) || relativeMagnitude >= PX_MAX_F32 ||
      orthogonalMagnitude >= PX_MAX_F32) {
    gMetrics.nonFinite = true;
    return;
  }

  const PxReal axisDot =
      relativeMagnitude > 1e-6f ? projection / relativeMagnitude : -1.0f;
  gMetrics.signedTravel += projection * gHeadlessOptions.dt;
  gMetrics.finalSignedProjection = projection;
  gMetrics.finalAxisDot = axisDot;
  gMetrics.finalOrthogonalSpeed = orthogonalMagnitude;
  if (isLinearDrive(gHeadlessConfig.drive)) {
    const PxVec3 relativeDisplacement =
        (sample.poseB.p - gMetrics.initialPoseB.p) -
        (sample.poseA.p - gMetrics.initialPoseA.p);
    gMetrics.finalSignedDisplacement =
        relativeDisplacement.dot(gMetrics.signedWorldAxis);
  } else {
    const PxVec3 angularStepA =
        quaternionDeltaVector(sample.poseA.q, gMetrics.previousPoseA.q);
    const PxVec3 angularStepB =
        quaternionDeltaVector(sample.poseB.q, gMetrics.previousPoseB.q);
    gMetrics.finalSignedDisplacement +=
        (angularStepB - angularStepA).dot(gMetrics.signedWorldAxis);
  }
  gMetrics.previousPoseA = sample.poseA;
  gMetrics.previousPoseB = sample.poseB;

  const PxU32 lateStart =
      gHeadlessOptions.frames > gLateWindowFrames
          ? gHeadlessOptions.frames - gLateWindowFrames
          : 0u;
  if (gMetrics.completedFrames > lateStart) {
    const PxReal targetError = projection - gMetrics.targetMagnitude;
    gMetrics.lateProjectionSum += projection;
    gMetrics.lateProjectionErrorSquaredSum += targetError * targetError;
    gMetrics.lateOrthogonalSquaredSum +=
        orthogonalMagnitude * orthogonalMagnitude;
    gMetrics.minLateAxisDot = PxMin(gMetrics.minLateAxisDot, axisDot);
    gMetrics.lateSampleCount++;
  }
  gMetrics.sampleCount++;
}

static void samplePositionState(const PrimaryPairSample &sample) {
  const DrivePairRuntime &pair = gPairs[0];
  if (gHeadlessCase == eCASE_OUTPUT_FORCE) {
    gPositionMetrics.publicForceSampleAttempts++;
    PxVec3 publicForce(0.0f);
    PxVec3 publicTorque(0.0f);
    PxConstraint *constraint = pair.joint ? pair.joint->getConstraint() : NULL;
    if (!constraint) {
      gPositionMetrics.nonFinitePublicForceSamples++;
      gMetrics.nonFinite = true;
    } else {
      if (constraint->getFlags().isSet(PxConstraintFlag::eBROKEN)) {
        if (gPositionMetrics.brokenSamples == 0)
          gPositionMetrics.firstBrokenFrame = gMetrics.completedFrames;
        gPositionMetrics.brokenSamples++;
      }
      constraint->getForce(publicForce, publicTorque);
      if (!publicForce.isFinite() || !publicTorque.isFinite()) {
        gPositionMetrics.nonFinitePublicForceSamples++;
        gMetrics.nonFinite = true;
      } else {
        const PxReal forceMagnitude = safeMagnitude(publicForce);
        const PxReal torqueMagnitude = safeMagnitude(publicTorque);
        const PxReal actor0Sign =
            gHeadlessConfig.endpoint == eENDPOINT_REVERSE ? 1.0f : -1.0f;
        if (gPositionMetrics.publicForceSamples == 0) {
          gPositionMetrics.firstPublicForce = publicForce;
          gPositionMetrics.firstPublicTorque = publicTorque;
          gPositionMetrics.firstNormalizedPublicTorque =
              publicTorque * actor0Sign;
          gPositionMetrics.firstSignedPublicForce =
              actor0Sign * publicForce.dot(gMetrics.signedWorldAxis);
        }
        gPositionMetrics.maximumPublicForce =
            PxMax(gPositionMetrics.maximumPublicForce, forceMagnitude);
        gPositionMetrics.maximumPublicTorque =
            PxMax(gPositionMetrics.maximumPublicTorque, torqueMagnitude);
        gPositionMetrics.publicForceSamples++;
      }
    }
  }
  const PxTransform frameA = sample.poseA * pair.jointFrameA;
  const PxTransform frameB = sample.poseB * pair.jointFrameB;
  const PxTransform relativePose = frameA.transformInv(frameB);
  const PxVec3 localAxis = getPositionTargetAxis(gHeadlessConfig.drive);
  const PxVec3 relativeVelocity =
      isLinearDrive(gHeadlessConfig.drive)
          ? sample.linearB - sample.linearA
          : sample.angularB - sample.angularA;
  const PxReal relativeSpeed = safeMagnitude(relativeVelocity);
  const PxReal relativeAcceleration =
      safeMagnitude(relativeVelocity -
                    gPositionMetrics.previousRelativeVelocity) /
      gHeadlessOptions.dt;
  if (isContactComparisonTopology(gHeadlessConfig.topology)) {
    const PxReal minimumBottom =
        PxMin(sample.poseA.p.y, sample.poseB.p.y) - 0.5f;
    const PxReal maximumAbsVerticalSpeed =
        PxMax(PxAbs(sample.linearA.y), PxAbs(sample.linearB.y));
    gComparisonMetrics.minimumBottom[0] =
        PxMin(gComparisonMetrics.minimumBottom[0], minimumBottom);
    gComparisonMetrics.maximumAbsVerticalSpeed[0] =
        PxMax(gComparisonMetrics.maximumAbsVerticalSpeed[0],
              maximumAbsVerticalSpeed);
  }
  if (pair.dynamicActorA &&
      isDynamicComparisonTopology(gHeadlessConfig.topology)) {
    const PxReal massA = gPositionMetrics.actorAMassReadback;
    const PxReal massB = gPositionMetrics.massReadback;
    const PxReal totalMass = massA + massB;
    if (!(totalMass > 0.0f) || !PxIsFinite(totalMass)) {
      gMetrics.nonFinite = true;
      return;
    }
    const PxVec3 centerOfMass =
        (sample.poseA.p * massA + sample.poseB.p * massB) / totalMass;
    const PxVec3 linearMomentum =
        sample.linearA * massA + sample.linearB * massB;
    // The fixture authors isotropic principal inertias, so I*w is already a
    // world-space angular-momentum vector and needs no frame transform.
    const PxVec3 angularMomentum =
        sample.angularA * gPositionMetrics.actorAInertiaReadback.x +
        sample.angularB * gPositionMetrics.inertiaReadback.x;
    PxVec3 centerOfMassDelta =
        centerOfMass - gPositionMetrics.initialCenterOfMass;
    PxVec3 measuredLinearMomentum = linearMomentum;
    if (isContactComparisonTopology(gHeadlessConfig.topology)) {
      centerOfMassDelta.y = 0.0f;
      measuredLinearMomentum.y = 0.0f;
    }
    const PxReal centerOfMassDrift =
        safeMagnitude(centerOfMassDelta);
    const PxReal linearMomentumMagnitude =
        safeMagnitude(measuredLinearMomentum);
    const PxReal angularMomentumMagnitude = safeMagnitude(angularMomentum);
    if (centerOfMassDrift >= PX_MAX_F32 ||
        linearMomentumMagnitude >= PX_MAX_F32 ||
        angularMomentumMagnitude >= PX_MAX_F32) {
      gMetrics.nonFinite = true;
      return;
    }
    gPositionMetrics.maximumCenterOfMassDrift =
        PxMax(gPositionMetrics.maximumCenterOfMassDrift,
              centerOfMassDrift);
    gPositionMetrics.maximumLinearMomentum =
        PxMax(gPositionMetrics.maximumLinearMomentum,
              linearMomentumMagnitude);
    gPositionMetrics.maximumAngularMomentum =
        PxMax(gPositionMetrics.maximumAngularMomentum,
              angularMomentumMagnitude);
    if (gMetrics.sampleCount == 0) {
      gPositionMetrics.firstSignedAngularAccelerationA =
          sample.angularA.dot(gMetrics.signedWorldAxis) /
          gHeadlessOptions.dt;
      gPositionMetrics.firstSignedAngularAccelerationB =
          sample.angularB.dot(gMetrics.signedWorldAxis) /
          gHeadlessOptions.dt;
    }
  }
  PxReal targetError = 0.0f;
  PxReal signedProgress = 0.0f;
  PxReal orthogonalError = 0.0f;

  if (isLinearDrive(gHeadlessConfig.drive)) {
    const PxVec3 targetErrorVector =
        gPositionMetrics.targetRelativePose.p - relativePose.p;
    const PxReal targetAxisError = targetErrorVector.dot(localAxis);
    targetError = PxAbs(targetAxisError);
    orthogonalError =
        safeMagnitude(targetErrorVector - localAxis * targetAxisError);
    signedProgress =
        (relativePose.p - gPositionMetrics.initialRelativePose.p)
            .dot(localAxis);
  } else {
    targetError = quaternionAngle(relativePose.q,
                                  gPositionMetrics.targetRelativePose.q);
    const PxVec3 progressVector = quaternionDeltaVector(
        relativePose.q, gPositionMetrics.initialRelativePose.q);
    signedProgress = progressVector.dot(localAxis);
    const PxReal targetAxisProgress = progressVector.dot(localAxis);
    orthogonalError =
        safeMagnitude(progressVector - localAxis * targetAxisProgress);
  }

  if (!PxIsFinite(targetError) || !PxIsFinite(signedProgress) ||
      !PxIsFinite(orthogonalError) || relativeSpeed >= PX_MAX_F32 ||
      relativeAcceleration >= PX_MAX_F32) {
    gMetrics.nonFinite = true;
    return;
  }

  gPositionMetrics.finalTargetError = targetError;
  gPositionMetrics.finalErrorRatio =
      gPositionMetrics.initialTargetError > 1e-6f
          ? targetError / gPositionMetrics.initialTargetError
          : PX_MAX_F32;
  gPositionMetrics.maximumSignedProgress =
      PxMax(gPositionMetrics.maximumSignedProgress, signedProgress);
  gPositionMetrics.minimumSignedProgress =
      PxMin(gPositionMetrics.minimumSignedProgress, signedProgress);
  gPositionMetrics.maximumOrthogonalError =
      PxMax(gPositionMetrics.maximumOrthogonalError, orthogonalError);
  gPositionMetrics.maximumOvershoot =
      PxMax(gPositionMetrics.maximumOvershoot,
             PxMax(0.0f, signedProgress - gMetrics.targetMagnitude));
  gPositionMetrics.maximumRelativeAcceleration =
      PxMax(gPositionMetrics.maximumRelativeAcceleration,
            relativeAcceleration);
  if (gMetrics.sampleCount == 0)
    gPositionMetrics.firstRelativeAcceleration = relativeAcceleration;
  gPositionMetrics.previousRelativeVelocity = relativeVelocity;
  gMetrics.finalSignedDisplacement = signedProgress;
  gMetrics.finalSignedProjection =
      relativeVelocity.dot(gMetrics.signedWorldAxis);
  if (!pair.actorB->isSleeping())
    gPositionMetrics.awakeSamples++;

  const PxU32 lateWindowFrames = getPositionLateWindowFrames();
  const PxU32 lateStart =
      gHeadlessOptions.frames > lateWindowFrames
          ? gHeadlessOptions.frames - lateWindowFrames
          : 0u;
  if (gMetrics.completedFrames > lateStart) {
    gPositionMetrics.lateErrorSquaredSum += targetError * targetError;
    gPositionMetrics.lateSpeedSquaredSum += relativeSpeed * relativeSpeed;
    gPositionMetrics.lateSampleCount++;
  }
  gMetrics.previousPoseA = sample.poseA;
  gMetrics.previousPoseB = sample.poseB;
  gMetrics.sampleCount++;
}

static bool sampleComparisonLane(PxU32 pairIndex) {
  PrimaryPairSample sample;
  if (!gatherPairSample(pairIndex, sample))
    return false;

  JointDriveComparisonLaneMetrics &lane =
      gComparisonMetrics.lanes[pairIndex];
  const DrivePairRuntime &pair = gPairs[pairIndex];
  const PxVec3 relativeVelocity = sample.linearB - sample.linearA;
  const PxReal relativeMagnitude = safeMagnitude(relativeVelocity);
  const PxReal projection = relativeVelocity.dot(pair.signedWorldAxis);
  const PxVec3 orthogonal =
      relativeVelocity - pair.signedWorldAxis * projection;
  const PxReal orthogonalMagnitude = safeMagnitude(orthogonal);
  if (!PxIsFinite(projection) || relativeMagnitude >= PX_MAX_F32 ||
      orthogonalMagnitude >= PX_MAX_F32) {
    gMetrics.nonFinite = true;
    return false;
  }

  if (isDynamicComparisonTopology(gHeadlessConfig.topology)) {
    const PxReal massA =
        gComparisonMetrics.actorAMassReadback[pairIndex];
    const PxReal massB = gComparisonMetrics.massReadback[pairIndex];
    const PxReal totalMass = massA + massB;
    if (totalMass <= 0.0f || !PxIsFinite(totalMass)) {
      gMetrics.nonFinite = true;
      return false;
    }
    const PxVec3 centerOfMass =
        (sample.poseA.p * massA + sample.poseB.p * massB) / totalMass;
    const PxVec3 momentum =
        sample.linearA * massA + sample.linearB * massB;
    PxVec3 centerOfMassDelta =
        centerOfMass - gComparisonMetrics.initialCenterOfMass[pairIndex];
    PxVec3 measuredMomentum = momentum;
    if (isContactComparisonTopology(gHeadlessConfig.topology)) {
      centerOfMassDelta.y = 0.0f;
      measuredMomentum.y = 0.0f;
      const PxReal minimumBottom =
          PxMin(sample.poseA.p.y, sample.poseB.p.y) - 0.5f;
      const PxReal maximumAbsVerticalSpeed =
          PxMax(PxAbs(sample.linearA.y), PxAbs(sample.linearB.y));
      gComparisonMetrics.minimumBottom[pairIndex] =
          PxMin(gComparisonMetrics.minimumBottom[pairIndex], minimumBottom);
      gComparisonMetrics.maximumAbsVerticalSpeed[pairIndex] =
          PxMax(gComparisonMetrics.maximumAbsVerticalSpeed[pairIndex],
                maximumAbsVerticalSpeed);
    }
    const PxReal centerOfMassDrift = safeMagnitude(centerOfMassDelta);
    const PxReal momentumMagnitude = safeMagnitude(measuredMomentum);
    if (centerOfMassDrift >= PX_MAX_F32 ||
        momentumMagnitude >= PX_MAX_F32) {
      gMetrics.nonFinite = true;
      return false;
    }
    gComparisonMetrics.maximumCenterOfMassDrift[pairIndex] = PxMax(
        gComparisonMetrics.maximumCenterOfMassDrift[pairIndex],
        centerOfMassDrift);
    gComparisonMetrics.maximumMomentumMagnitude[pairIndex] = PxMax(
        gComparisonMetrics.maximumMomentumMagnitude[pairIndex],
        momentumMagnitude);
  }

  lane.finalRelativeVelocity = relativeVelocity;
  if (orthogonalMagnitude > lane.maximumOrthogonalSpeed) {
    lane.maximumOrthogonalSpeed = orthogonalMagnitude;
    lane.maximumOrthogonalVelocity = orthogonal;
  }
  if (relativeMagnitude > 1e-6f) {
    lane.minimumAxisDot =
        PxMin(lane.minimumAxisDot, projection / relativeMagnitude);
  }

  const PxU32 transientFrames = getComparisonTransientFrames(gHeadlessCase);
  if (gMetrics.completedFrames <= transientFrames) {
    const PxReal monotonicDrop = lane.previousProjection - projection;
    if (monotonicDrop > 1e-4f)
      lane.monotonicViolations++;
    lane.maximumMonotonicDrop =
        PxMax(lane.maximumMonotonicDrop, monotonicDrop);
    if (projection < -1e-4f ||
        projection > 1.02f * gMetrics.targetMagnitude)
      lane.overshootCount++;
    const PxReal acceleration =
        PxAbs(projection - lane.previousProjection) / gHeadlessOptions.dt;
    lane.peakAcceleration = PxMax(lane.peakAcceleration, acceleration);
    lane.previousProjection = projection;
    lane.transientSamples++;
    if (gMetrics.completedFrames == 1)
      lane.firstProjection = projection;
    if (gMetrics.completedFrames == 4)
      lane.frameFourProjection = projection;
    if (gMetrics.completedFrames == transientFrames) {
      lane.transientProjection = projection;
      lane.response = projection / gMetrics.targetMagnitude;
      const PxReal remaining = gMetrics.targetMagnitude - projection;
      if (remaining > 1e-6f &&
          remaining < gMetrics.targetMagnitude + 1e-6f) {
        const PxReal duration =
            PxReal(transientFrames) * gHeadlessOptions.dt;
        lane.decayRate =
            -PxReal(std::log(double(remaining / gMetrics.targetMagnitude))) /
            duration;
      } else {
        lane.decayRate = PX_MAX_F32;
      }
    }
  }

  if (pairIndex == 1) {
    gMetrics.finalSignedProjection = projection;
    gMetrics.finalOrthogonalSpeed = orthogonalMagnitude;
  }
  return true;
}

static void sampleOrderingState() {
  const PxU32 lateStart = gHeadlessOptions.frames - gLateWindowFrames;
  const bool lateSample = gOrderingMetrics.completedFrames > lateStart;
  gOrderingMetrics.sampleAttempts++;
  if (lateSample)
    gOrderingMetrics.lateSampleAttempts++;

  if (isAngularOutputForceCase(gHeadlessCase)) {
    gOrderingMetrics.publicForceSampleAttempts++;
    PxConstraint *constraint = gOrderingRuntime.joint
                                   ? gOrderingRuntime.joint->getConstraint()
                                   : NULL;
    if (!constraint) {
      gOrderingMetrics.nonFinitePublicForceSamples++;
      gOrderingMetrics.nonFinite = true;
    } else {
      if (constraint->getFlags().isSet(PxConstraintFlag::eBROKEN)) {
        if (gOrderingMetrics.brokenSamples == 0)
          gOrderingMetrics.firstBrokenFrame =
              gOrderingMetrics.completedFrames;
        gOrderingMetrics.brokenSamples++;
      }
      PxVec3 publicForce(0.0f);
      PxVec3 publicTorque(0.0f);
      constraint->getForce(publicForce, publicTorque);
      if (!publicForce.isFinite() || !publicTorque.isFinite()) {
        gOrderingMetrics.nonFinitePublicForceSamples++;
        gOrderingMetrics.nonFinite = true;
      } else {
        if (gOrderingMetrics.publicForceSamples == 0) {
          gOrderingMetrics.firstPublicForce = publicForce;
          gOrderingMetrics.firstPublicTorque = publicTorque;
          gOrderingMetrics.firstActor0FrameTorque =
              gOrderingRuntime.actorAWorldFrameRotation.rotateInv(
                  publicTorque);
          gOrderingMetrics.firstSignedPublicTorque =
              publicTorque.dot(gOrderingMetrics.expectedFrameAAxis);
        }
        gOrderingMetrics.maximumPublicForce =
            PxMax(gOrderingMetrics.maximumPublicForce,
                  safeMagnitude(publicForce));
        gOrderingMetrics.maximumPublicTorque =
            PxMax(gOrderingMetrics.maximumPublicTorque,
                  safeMagnitude(publicTorque));
        gOrderingMetrics.publicForceSamples++;
      }
    }
  }

  if (!gOrderingRuntime.dynamicActor) {
    gOrderingMetrics.sampleErrors++;
    return;
  }

  const PxTransform pose = gOrderingRuntime.dynamicActor->getGlobalPose();
  const PxVec3 linearVelocity =
      gOrderingRuntime.dynamicActor->getLinearVelocity();
  const PxVec3 angularVelocity =
      gOrderingRuntime.dynamicActor->getAngularVelocity();
  if (!pose.isValid() || !linearVelocity.isFinite() ||
      !angularVelocity.isFinite()) {
    gOrderingMetrics.nonFinite = true;
    gOrderingMetrics.nonFiniteSamples++;
    if (lateSample)
      gOrderingMetrics.lateNonFiniteSamples++;
    return;
  }
  gOrderingMetrics.sampleCount++;
  if (lateSample)
  gOrderingMetrics.lateSampleCount++;

  const bool reverse = gHeadlessConfig.endpoint == eENDPOINT_REVERSE;
  const PxVec3 dynamicVelocity =
      gOrderingRuntime.angular ? angularVelocity : linearVelocity;
  const PxVec3 relativeVelocity =
      reverse ? -dynamicVelocity : dynamicVelocity;
  const PxReal relativeMagnitude = safeMagnitude(relativeVelocity);
  const PxReal dynamicMagnitude = safeMagnitude(dynamicVelocity);
  const PxReal relativeProjection =
      relativeVelocity.dot(gOrderingRuntime.expectedAxis);
  const PxReal dynamicProjection =
      dynamicVelocity.dot(gOrderingRuntime.expectedDynamicAxis);
  const PxVec3 relativeOrthogonal =
      relativeVelocity -
      gOrderingRuntime.expectedAxis * relativeProjection;
  const PxVec3 dynamicOrthogonal =
      dynamicVelocity -
      gOrderingRuntime.expectedDynamicAxis * dynamicProjection;
  const PxReal relativeOrthogonalMagnitude =
      safeMagnitude(relativeOrthogonal);
  const PxReal dynamicOrthogonalMagnitude = safeMagnitude(dynamicOrthogonal);
  const PxReal relativeDirectionDot =
      relativeMagnitude > 1e-6f ? relativeProjection / relativeMagnitude
                                : -1.0f;
  const PxReal dynamicDirectionDot =
      dynamicMagnitude > 1e-6f ? dynamicProjection / dynamicMagnitude : -1.0f;
  if (!PxIsFinite(relativeProjection) || !PxIsFinite(dynamicProjection) ||
      relativeMagnitude >= PX_MAX_F32 || dynamicMagnitude >= PX_MAX_F32 ||
      relativeOrthogonalMagnitude >= PX_MAX_F32 ||
      dynamicOrthogonalMagnitude >= PX_MAX_F32) {
    gOrderingMetrics.nonFinite = true;
    gOrderingMetrics.nonFiniteSamples++;
    gOrderingMetrics.sampleCount--;
    if (lateSample) {
      gOrderingMetrics.lateNonFiniteSamples++;
      gOrderingMetrics.lateSampleCount--;
    }
    return;
  }

  if (gOrderingMetrics.sampleCount == 0) {
    gOrderingMetrics.firstRelativeProjection = relativeProjection;
    gOrderingMetrics.firstDynamicProjection = dynamicProjection;
    gOrderingMetrics.firstDynamicAcceleration =
        dynamicProjection / gHeadlessOptions.dt;
  }
  const PxReal dynamicAcceleration =
      PxAbs(dynamicProjection -
            gOrderingMetrics.previousDynamicProjection) /
      gHeadlessOptions.dt;
  if (gOrderingMetrics.firstDrivenDynamicAcceleration == 0.0f &&
      dynamicAcceleration > 1e-4f)
    gOrderingMetrics.firstDrivenDynamicAcceleration = dynamicAcceleration;
  if (gOrderingMetrics.sampleCount < gAngularOutputTransientFrames)
    gOrderingMetrics.maximumInitialDynamicAcceleration =
        PxMax(gOrderingMetrics.maximumInitialDynamicAcceleration,
              dynamicAcceleration);
  gOrderingMetrics.previousDynamicProjection = dynamicProjection;

  gOrderingMetrics.finalRelativeProjection = relativeProjection;
  gOrderingMetrics.finalRelativeDirectionDot = relativeDirectionDot;
  gOrderingMetrics.finalRelativeOrthogonal = relativeOrthogonalMagnitude;
  gOrderingMetrics.finalDynamicProjection = dynamicProjection;
  gOrderingMetrics.finalDynamicDirectionDot = dynamicDirectionDot;
  gOrderingMetrics.finalDynamicOrthogonal = dynamicOrthogonalMagnitude;
  const PxVec3 displacement = pose.p - gOrderingRuntime.initialDynamicPose.p;
  if (gOrderingRuntime.angular) {
    const PxVec3 angularStep = quaternionDeltaVector(
        pose.q, gOrderingRuntime.previousDynamicPose.q);
    gOrderingMetrics.finalDynamicDisplacement +=
        angularStep.dot(gOrderingRuntime.expectedDynamicAxis);
    gOrderingMetrics.finalRelativeDisplacement +=
        (reverse ? -angularStep : angularStep)
            .dot(gOrderingRuntime.expectedAxis);
  } else {
    gOrderingMetrics.finalDynamicDisplacement =
        displacement.dot(gOrderingRuntime.expectedDynamicAxis);
    gOrderingMetrics.finalRelativeDisplacement =
        (reverse ? -displacement : displacement)
            .dot(gOrderingRuntime.expectedAxis);
  }
  gOrderingRuntime.previousDynamicPose = pose;
  gOrderingMetrics.maxDynamicRotationError =
      PxMax(gOrderingMetrics.maxDynamicRotationError,
            quaternionAngle(pose.q,
                            gOrderingRuntime.initialDynamicPose.q));
  gOrderingMetrics.maxDynamicAngularSpeed =
      PxMax(gOrderingMetrics.maxDynamicAngularSpeed,
            safeMagnitude(angularVelocity));
  gOrderingMetrics.maxDynamicPositionError =
      PxMax(gOrderingMetrics.maxDynamicPositionError,
            safeMagnitude(displacement));
  gOrderingMetrics.maxQuaternionNormError =
      PxMax(gOrderingMetrics.maxQuaternionNormError,
            PxAbs(pose.q.magnitudeSquared() - 1.0f));
  gOrderingMetrics.maxAbsPosition =
      PxMax(gOrderingMetrics.maxAbsPosition, maxAbsComponent(pose.p));
  gOrderingMetrics.maxLinearSpeed =
      PxMax(gOrderingMetrics.maxLinearSpeed, safeMagnitude(linearVelocity));

  if (lateSample) {
    const PxReal relativeTargetError =
        relativeProjection - gTargetVelocity;
    const PxReal dynamicTargetError = dynamicProjection - gTargetVelocity;
    gOrderingMetrics.relativeLateProjectionSum += relativeProjection;
    gOrderingMetrics.relativeLateTargetErrorSquaredSum +=
        relativeTargetError * relativeTargetError;
    gOrderingMetrics.relativeLateOrthogonalSquaredSum +=
        relativeOrthogonalMagnitude * relativeOrthogonalMagnitude;
    gOrderingMetrics.dynamicLateProjectionSum += dynamicProjection;
    gOrderingMetrics.dynamicLateTargetErrorSquaredSum +=
        dynamicTargetError * dynamicTargetError;
    gOrderingMetrics.dynamicLateOrthogonalSquaredSum +=
        dynamicOrthogonalMagnitude * dynamicOrthogonalMagnitude;
    gOrderingMetrics.minLateRelativeProjection =
        PxMin(gOrderingMetrics.minLateRelativeProjection,
              relativeProjection);
    gOrderingMetrics.minLateRelativeDirectionDot =
        PxMin(gOrderingMetrics.minLateRelativeDirectionDot,
              relativeDirectionDot);
    gOrderingMetrics.maxLateRelativeOrthogonal =
        PxMax(gOrderingMetrics.maxLateRelativeOrthogonal,
              relativeOrthogonalMagnitude);
    gOrderingMetrics.minLateDynamicProjection =
        PxMin(gOrderingMetrics.minLateDynamicProjection,
              dynamicProjection);
    gOrderingMetrics.minLateDynamicDirectionDot =
        PxMin(gOrderingMetrics.minLateDynamicDirectionDot,
              dynamicDirectionDot);
    gOrderingMetrics.maxLateDynamicOrthogonal =
        PxMax(gOrderingMetrics.maxLateDynamicOrthogonal,
              dynamicOrthogonalMagnitude);
  }
}

static void sampleHeadlessState() {
  if (isLegacyAngularLimitCase(gHeadlessCase)) {
    sampleLegacyAngularLimitState();
    return;
  }
  if (isOrderingCase(gHeadlessCase)) {
    sampleOrderingState();
    return;
  }
  if (isComparisonCase(gHeadlessCase)) {
    if (sampleComparisonLane(0) && sampleComparisonLane(1))
      gMetrics.sampleCount++;
    return;
  }
  PrimaryPairSample sample;
  if (!gatherPairSample(0, sample))
    return;
  if (isPositionLikeCase(gHeadlessCase))
    samplePositionState(sample);
  else
    sampleVelocityState(sample);
}

void stepPhysics(bool interactive) {
  if (!gScene)
    return;
  if (interactive && gPause && !gOneFrame)
    return;
  gOneFrame = false;

  if (!interactive) {
    beginContactComparisonFrame();
    if (!updateHeadlessKinematicTarget()) {
      gInitializationFailed = true;
      return;
    }
  }
  gScene->simulate(interactive ? (1.0f / 60.0f) : gHeadlessOptions.dt);
  PxU32 errorState = 0;
  if (!gScene->fetchResults(true, &errorState)) {
    if (!interactive) {
      if (isOrderingCase(gHeadlessCase)) {
        gOrderingMetrics.fetchFailures++;
        gOrderingMetrics.fetchErrorState |= errorState;
      } else {
        gMetrics.fetchFailures++;
        gMetrics.fetchErrorState |= errorState;
      }
    }
    return;
  }

  if (!interactive) {
    if (isOrderingCase(gHeadlessCase)) {
      gOrderingMetrics.fetchErrorState |= errorState;
      gOrderingMetrics.completedFrames++;
    } else {
      gMetrics.fetchErrorState |= errorState;
      gMetrics.completedFrames++;
    }
    commitContactComparisonFrame();
    sampleHeadlessState();
  }
}

static void releaseOrderingSceneState() {
  PX_RELEASE(gOrderingRuntime.joint);
  if (gOrderingRuntime.dynamicActor) {
    if (gScene && gOrderingRuntime.dynamicActor->getScene() == gScene)
      gScene->removeActor(*gOrderingRuntime.dynamicActor);
    PX_RELEASE(gOrderingRuntime.dynamicActor);
  }
  if (gScene) {
    gOrderingMetrics.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gOrderingMetrics.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gOrderingMetrics.cleanupConstraints = gScene->getNbConstraints();
  } else {
    gOrderingMetrics.cleanupDynamicActors = 0;
    gOrderingMetrics.cleanupStaticActors = 0;
    gOrderingMetrics.cleanupConstraints = 0;
  }
  gOrderingRuntime = OrderingDriveRuntime();
}

void cleanupPhysics(bool interactive) {
  if (!interactive && isOrderingCase(gHeadlessCase))
    releaseOrderingSceneState();
  releaseSceneState();
  PX_RELEASE(gMaterial);
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
#if PX_SUPPORT_GPU_PHYSX
  PX_RELEASE(gCudaContextManager);
#endif
  PX_RELEASE(gFoundation);

  if (!interactive && isOrderingCase(gHeadlessCase)) {
    gOrderingMetrics.cleanupComplete =
        !gScene && !gDispatcher && !gMaterial && !gPhysics && !gFoundation &&
        !gPvd && !gOrderingRuntime.joint &&
        !gOrderingRuntime.dynamicActor &&
        gOrderingMetrics.cleanupDynamicActors == 0 &&
        gOrderingMetrics.cleanupStaticActors == 0 &&
        gOrderingMetrics.cleanupConstraints == 0;
  }

  if (interactive)
    printf("SnippetJointDrive done.\n");
}

void renderText() {
#ifdef RENDER_SNIPPET
  Snippets::print("Press F1 to change body0's joint frame orientation");
  Snippets::print("Press F2 to change body0's type (static/kinematic)");
  Snippets::print("Press F3 to change body1's joint frame orientation");
  Snippets::print("Press F4 to change body1's orientation");
#if PX_SUPPORT_GPU_PHYSX
  Snippets::print("Press F5 to use CPU or GPU");
#endif
  Snippets::print("Press F6 to select the next drive");
  switch (gSceneIndex) {
  case 0:
    Snippets::print("Current drive: linear X");
    break;
  case 1:
    Snippets::print("Current drive: angular twist (around X)");
    break;
  case 2:
    Snippets::print("Current drive: angular swing (around Y)");
    break;
  case 3:
    Snippets::print("Current drive: angular slerp (around Y)");
    break;
  }
  if (gChangeObjectAType)
    Snippets::print("Body0 type: KINEMATIC (dynamic)");
  else
    Snippets::print("Body0 type: STATIC");
#if PX_SUPPORT_GPU_PHYSX
  if (gUseGPU)
    Snippets::print("Current mode: GPU");
  else
    Snippets::print("Current mode: CPU");
#endif
  Snippets::print("body1's drive axis should only depend on body0's joint axes.");
#endif
}

void keyPress(unsigned char key, const PxTransform &) {
  if (key == 'p' || key == 'P')
    gPause = !gPause;
  if (key == 'o' || key == 'O') {
    gPause = true;
    gOneFrame = true;
  }

  if (key == 1) {
    gChangeJointFrameARotation = !gChangeJointFrameARotation;
    createScene();
  } else if (key == 2) {
    gChangeObjectAType = !gChangeObjectAType;
    createScene();
  } else if (key == 3) {
    gChangeJointFrameBRotation = !gChangeJointFrameBRotation;
    createScene();
  } else if (key == 4) {
    gChangeObjectBRotation = !gChangeObjectBRotation;
    createScene();
  }
#if PX_SUPPORT_GPU_PHYSX
  else if (key == 5) {
    gUseGPU = !gUseGPU;
    createScene();
  }
#endif
  else if (key == 6) {
    gSceneIndex = (gSceneIndex + 1) % gMaxSceneIndex;
    createScene();
  }
}

static void setGateFailure(JointDriveGateEvaluation &evaluation,
                           const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
  evaluation.status = "FAIL";
  evaluation.reason = reason;
}

static void setGateError(JointDriveGateEvaluation &evaluation,
                         const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static void setGateErrorOverFailure(JointDriveGateEvaluation &evaluation,
                                    const char *reason) {
  if (evaluation.exitCode == Snippets::eHEADLESS_CONFIG_ERROR)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static PxReal getOrderingLateDynamicMean() {
  return gOrderingMetrics.lateSampleCount
             ? gOrderingMetrics.dynamicLateProjectionSum /
                   PxReal(gOrderingMetrics.lateSampleCount)
             : 0.0f;
}

static PxReal getOrderingLateDynamicTargetRms() {
  return gOrderingMetrics.lateSampleCount
             ? PxSqrt(gOrderingMetrics.dynamicLateTargetErrorSquaredSum /
                      PxReal(gOrderingMetrics.lateSampleCount))
             : 0.0f;
}

static PxReal getOrderingLateDynamicOrthogonalRms() {
  return gOrderingMetrics.lateSampleCount
             ? PxSqrt(gOrderingMetrics.dynamicLateOrthogonalSquaredSum /
                      PxReal(gOrderingMetrics.lateSampleCount))
             : 0.0f;
}

static JointDriveGateEvaluation evaluateOrderingGate() {
  JointDriveGateEvaluation evaluation;
  const bool angularOutputForce =
      isAngularOutputForceCase(gHeadlessCase);
  const bool componentClampedSlerp =
      angularOutputForce && gHeadlessConfig.drive == eDRIVE_SLERP;
  const bool expectsOutputBreak =
      angularOutputForce && gHeadlessConfig.outputForceEnabled &&
      gHeadlessConfig.breakMode == eBREAK_BELOW_DRIVE_LIMIT;
  if (gOrderingMetrics.lateSampleCount) {
    const PxReal invSamples =
        1.0f / PxReal(gOrderingMetrics.lateSampleCount);
    evaluation.lateTargetMean =
        gOrderingMetrics.relativeLateProjectionSum * invSamples;
    evaluation.lateTargetRms =
        PxSqrt(gOrderingMetrics.relativeLateTargetErrorSquaredSum *
               invSamples);
    evaluation.lateOrthogonalRms =
        PxSqrt(gOrderingMetrics.relativeLateOrthogonalSquaredSum *
               invSamples);
  }
  evaluation.motionWitness =
      PxMin(gOrderingMetrics.finalRelativeDisplacement,
            gOrderingMetrics.finalDynamicDisplacement);

  if (gOrderingMetrics.nonFinite)
    setGateFailure(evaluation, "ordering_non_finite");
  if (gOrderingMetrics.maxQuaternionNormError >
      gMaximumQuaternionNormError)
    setGateFailure(evaluation, "ordering_quaternion_norm");
  if (gOrderingMetrics.maxAbsPosition > gMaximumAbsPosition ||
      (!gOrderingMetrics.angular &&
       gOrderingMetrics.maxLinearSpeed > gMaximumLinearSpeed) ||
      (gOrderingMetrics.angular &&
       gOrderingMetrics.maxDynamicAngularSpeed > gMaximumAngularSpeed))
    setGateFailure(evaluation, "ordering_runaway");
  if (!expectsOutputBreak &&
      (gOrderingMetrics.minLateRelativeProjection <= 0.0f ||
       gOrderingMetrics.minLateDynamicProjection <= 0.0f ||
       (!componentClampedSlerp &&
        (gOrderingMetrics.minLateRelativeDirectionDot < gMinimumAxisDot ||
         gOrderingMetrics.minLateDynamicDirectionDot < gMinimumAxisDot ||
         gOrderingMetrics.maxLateRelativeOrthogonal >
             gMaximumOrthogonalRms ||
         gOrderingMetrics.maxLateDynamicOrthogonal >
             gMaximumOrthogonalRms ||
         evaluation.lateOrthogonalRms > gMaximumOrthogonalRms ||
         getOrderingLateDynamicOrthogonalRms() >
             gMaximumOrthogonalRms))))
    setGateFailure(evaluation, "ordering_axis_or_sign");
  if (!expectsOutputBreak &&
      (evaluation.lateTargetMean < gMinimumLateProjection ||
      evaluation.lateTargetMean > gMaximumLateProjection ||
      evaluation.lateTargetRms > gMaximumLateTargetRms ||
      getOrderingLateDynamicMean() < gMinimumLateProjection ||
      getOrderingLateDynamicMean() > gMaximumLateProjection ||
      getOrderingLateDynamicTargetRms() > gMaximumLateTargetRms))
    setGateFailure(evaluation, "ordering_target_tracking");
  if (!expectsOutputBreak &&
      (gOrderingMetrics.finalRelativeDisplacement < gMinimumMotionWitness ||
       gOrderingMetrics.finalDynamicDisplacement < gMinimumMotionWitness))
    setGateFailure(evaluation, "ordering_missing_motion");
  if (gOrderingMetrics.angular) {
    if (gOrderingMetrics.maxDynamicPositionError > 1e-3f ||
        gOrderingMetrics.maxLinearSpeed > 1e-3f)
      setGateFailure(evaluation, "ordering_linear_leakage");
  } else if (gOrderingMetrics.maxDynamicRotationError >
                 gOrderingMaximumRotationError ||
             gOrderingMetrics.maxDynamicAngularSpeed >
                 gOrderingMaximumAngularSpeed) {
    setGateFailure(evaluation, "ordering_angular_leakage");
  }

  if (gScene) {
    gOrderingMetrics.finalDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gOrderingMetrics.finalStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gOrderingMetrics.finalConstraints = gScene->getNbConstraints();
  }
  if (gInitializationFailed || !gScene || !gOrderingRuntime.dynamicActor ||
      !gOrderingRuntime.joint)
    setGateErrorOverFailure(evaluation, "ordering_initialization");
  if (!gOrderingMetrics.actorOrderValid ||
      !gOrderingMetrics.frameWitnessValid ||
      !gOrderingMetrics.driveReadbackValid ||
      !gOrderingMetrics.fixtureWitnessValid)
    setGateErrorOverFailure(evaluation, "ordering_fixture");
  if (angularOutputForce) {
    const PxReal expectedBreakTorque = getAngularOutputBreakThreshold(
        gHeadlessConfig.drive, gHeadlessConfig.breakMode);
    if (gOrderingMetrics.linearBreakForceReadback != PX_MAX_F32 ||
        PxAbs(gOrderingMetrics.angularBreakForceReadback -
              expectedBreakTorque) >
            (expectedBreakTorque == PX_MAX_F32 ? 0.0f : 1e-5f))
      setGateErrorOverFailure(evaluation,
                              "angular_output_break_readback");
    if (gOrderingMetrics.publicForceSampleAttempts !=
            gHeadlessOptions.frames ||
        gOrderingMetrics.publicForceSamples != gHeadlessOptions.frames ||
        gOrderingMetrics.nonFinitePublicForceSamples != 0)
      setGateErrorOverFailure(evaluation,
                              "angular_output_sample_accounting");

    const PxReal expectedTorqueMagnitude =
        safeMagnitude(gOrderingMetrics.expectedActor0Torque);
    const PxReal firstTorqueMagnitude =
        safeMagnitude(gOrderingMetrics.firstPublicTorque);
    const PxReal firstTorqueDirectionDot =
        firstTorqueMagnitude > gAngularOutputTorqueZeroTolerance &&
                expectedTorqueMagnitude > gAngularOutputTorqueZeroTolerance
            ? gOrderingMetrics.firstPublicTorque.dot(
                  gOrderingMetrics.expectedActor0Torque) /
                  (firstTorqueMagnitude * expectedTorqueMagnitude)
            : -1.0f;
    const PxReal expectedDynamicAcceleration = PxAbs(
        gOrderingMetrics.expectedActor0Torque.dot(
            gOrderingMetrics.expectedFrameAAxis));
    if (gHeadlessConfig.outputForceEnabled) {
      if (safeMagnitude(gOrderingMetrics.firstPublicForce) >
              gAngularOutputForceTolerance ||
          gOrderingMetrics.maximumPublicForce >
              gAngularOutputForceTolerance)
        setGateFailure(evaluation, "angular_output_linear_force_nonzero");
      if (gOrderingMetrics.maximumPublicTorque <=
          gAngularOutputTorqueZeroTolerance)
        setGateFailure(evaluation, "angular_output_torque_missing");
      if (firstTorqueDirectionDot <= -0.99f)
        setGateFailure(evaluation, "angular_output_torque_sign");
      if (firstTorqueMagnitude < 0.80f * expectedTorqueMagnitude ||
          firstTorqueMagnitude > 1.20f * expectedTorqueMagnitude)
        setGateFailure(evaluation, "angular_output_torque_units");
      if (firstTorqueDirectionDot < 1.0f - gOutputForceDirectionTolerance)
        setGateFailure(evaluation, "angular_output_torque_frame");
      if (gOrderingMetrics.maximumPublicTorque >
          1.20f * expectedTorqueMagnitude)
        setGateFailure(evaluation, "angular_output_torque_limit");
      if (gHeadlessConfig.drive == eDRIVE_SLERP &&
          safeMagnitude(gOrderingMetrics.firstPublicTorque -
                        gOrderingMetrics.expectedActor0Torque) >
              gAngularOutputForceTolerance)
        setGateFailure(evaluation, "angular_output_slerp_components");
    } else if (gOrderingMetrics.maximumPublicForce >
                   gAngularOutputForceTolerance ||
               gOrderingMetrics.maximumPublicTorque >
                   gAngularOutputTorqueZeroTolerance) {
      setGateFailure(evaluation, "angular_output_force_disabled_nonzero");
    }

    // eOUTPUT_FORCE controls public writeback and break contribution, not the
    // physical finite drive limit.  Both flag states must therefore exhibit
    // the same limited first response.
    if (gOrderingMetrics.firstDrivenDynamicAcceleration <
            0.80f * expectedDynamicAcceleration ||
        gOrderingMetrics.firstDrivenDynamicAcceleration >
            1.20f * expectedDynamicAcceleration)
      setGateFailure(evaluation, "angular_output_torque_limit_motion");

    if (expectsOutputBreak) {
      if (gOrderingMetrics.brokenSamples == 0 ||
          gOrderingMetrics.firstBrokenFrame > 2)
        setGateFailure(evaluation, "angular_output_break_missing");
    } else if (gOrderingMetrics.brokenSamples != 0) {
      setGateFailure(evaluation, "angular_output_break_unexpected");
    }
  }
  if (gOrderingMetrics.initialDynamicActors != 1 ||
      gOrderingMetrics.initialStaticActors != 0 ||
      gOrderingMetrics.initialConstraints != 1 ||
      gOrderingMetrics.finalDynamicActors != 1 ||
      gOrderingMetrics.finalStaticActors != 0 ||
      gOrderingMetrics.finalConstraints != 1)
    setGateErrorOverFailure(evaluation, "ordering_topology");
  if (gOrderingMetrics.completedFrames != gHeadlessOptions.frames ||
      gOrderingMetrics.fetchFailures || gOrderingMetrics.fetchErrorState)
    setGateErrorOverFailure(evaluation, "ordering_simulation");
  if (gOrderingMetrics.sampleAttempts != gHeadlessOptions.frames ||
      gOrderingMetrics.sampleCount + gOrderingMetrics.nonFiniteSamples +
              gOrderingMetrics.sampleErrors !=
          gOrderingMetrics.sampleAttempts ||
      gOrderingMetrics.lateSampleAttempts != gLateWindowFrames ||
      gOrderingMetrics.lateSampleCount +
              gOrderingMetrics.lateNonFiniteSamples !=
          gOrderingMetrics.lateSampleAttempts ||
      gOrderingMetrics.sampleErrors)
    setGateErrorOverFailure(evaluation, "ordering_sample_accounting");
  if (gErrorCallback.getFatalCount())
    setGateErrorOverFailure(evaluation, "ordering_physx_error");
  return evaluation;
}

static JointDriveGateEvaluation evaluateGate() {
  JointDriveGateEvaluation evaluation;
  if (gHeadlessCase == eCASE_VELOCITY && gMetrics.lateSampleCount) {
    const PxReal invSamples = 1.0f / PxReal(gMetrics.lateSampleCount);
    evaluation.lateTargetMean = gMetrics.lateProjectionSum * invSamples;
    evaluation.lateTargetRms =
        PxSqrt(gMetrics.lateProjectionErrorSquaredSum * invSamples);
    evaluation.lateOrthogonalRms =
        PxSqrt(gMetrics.lateOrthogonalSquaredSum * invSamples);
  }
  if (isPositionLikeCase(gHeadlessCase) &&
      gPositionMetrics.lateSampleCount) {
    const PxReal invSamples =
        1.0f / PxReal(gPositionMetrics.lateSampleCount);
    evaluation.positionLateErrorRms =
        PxSqrt(gPositionMetrics.lateErrorSquaredSum * invSamples);
    evaluation.positionLateSpeedRms =
        PxSqrt(gPositionMetrics.lateSpeedSquaredSum * invSamples);
    if (gPositionMetrics.initialTargetError > 1e-6f) {
      evaluation.positionLateErrorRatio =
          evaluation.positionLateErrorRms /
          gPositionMetrics.initialTargetError;
      evaluation.positionMotionRatio =
          gPositionMetrics.maximumSignedProgress /
          gPositionMetrics.initialTargetError;
    }
  }
  if (isComparisonCase(gHeadlessCase)) {
    const JointDriveComparisonLaneMetrics &reference =
        gComparisonMetrics.lanes[0];
    const JointDriveComparisonLaneMetrics &test =
        gComparisonMetrics.lanes[1];
    if (reference.response > 1e-6f)
      evaluation.testToReferenceResponseRatio =
          test.response / reference.response;
    if (test.decayRate > 1e-6f && test.decayRate < PX_MAX_F32 &&
        reference.decayRate < PX_MAX_F32)
      evaluation.referenceToTestRateRatio =
          reference.decayRate / test.decayRate;
    const bool accelerationLimitProbe =
        gHeadlessCase == eCASE_ACCELERATION_MODE &&
        gHeadlessConfig.lowForceLimit;
    if ((gHeadlessCase == eCASE_FORCE_LIMIT &&
         gHeadlessConfig.lowForceLimit) ||
        accelerationLimitProbe) {
      const PxReal lowLimit = accelerationLimitProbe
                                  ? gAccelerationLimitProbeLowLimit
                                  : gForceLimitProbeLowLimit;
      const PxReal inverseMassSum =
          isDynamicComparisonTopology(gHeadlessConfig.topology)
              ? (1.0f / gComparisonMetrics.actorAMassReadback[1] +
                 1.0f / gComparisonMetrics.massReadback[1])
              : (1.0f / gComparisonMetrics.massReadback[1]);
      const PxU32 transientFrames =
          getComparisonTransientFrames(gHeadlessCase);
      evaluation.expectedTestDeltaVelocity =
          lowLimit *
          (PxReal(transientFrames) * gHeadlessOptions.dt) *
          inverseMassSum;
      if (evaluation.expectedTestDeltaVelocity > 1e-6f)
        evaluation.normalizedImpulse =
            test.transientProjection /
            evaluation.expectedTestDeltaVelocity;
      evaluation.meanTestAcceleration =
          test.transientProjection /
          (PxReal(transientFrames) * gHeadlessOptions.dt);
    }
  }
  evaluation.motionWitness =
      isPositionLikeCase(gHeadlessCase)
          ? gPositionMetrics.maximumSignedProgress
          : (isLegacyAngularLimitCase(gHeadlessCase)
                 ? gAngularLimitMetrics.initialEllipseRadius -
                       gAngularLimitMetrics.finalEllipseRadius
                 : gMetrics.finalSignedDisplacement);

  const PxU32 expectedPairCount =
      isComparisonCase(gHeadlessCase) ? 2u : 1u;
  bool pairStateValid = gPairCount == expectedPairCount;
  for (PxU32 i = 0; i < gPairCount && i < gMaxDrivePairs; ++i) {
    pairStateValid = pairStateValid && gPairs[i].actorA &&
                     gPairs[i].actorB && gPairs[i].joint;
  }
  if (gInitializationFailed || !gScene || !pairStateValid)
    setGateError(evaluation, "initialization");
  if (gMetrics.nonFinite)
    setGateFailure(evaluation, "non_finite");
  if (gMetrics.completedFrames != gHeadlessOptions.frames ||
      gMetrics.sampleCount != gHeadlessOptions.frames ||
      gMetrics.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (gHeadlessCase == eCASE_VELOCITY) {
    if (gMetrics.lateSampleCount != gLateWindowFrames)
      setGateError(evaluation, "late_window_accounting");
  } else if (isPositionLikeCase(gHeadlessCase) &&
             gPositionMetrics.lateSampleCount !=
                 getPositionLateWindowFrames()) {
    setGateError(evaluation, "late_window_accounting");
  } else if (isLegacyAngularLimitCase(gHeadlessCase) &&
             gAngularLimitMetrics.lateSampleCount != gLateWindowFrames) {
    setGateError(evaluation, "angular_limit_late_window_accounting");
  } else if (isComparisonCase(gHeadlessCase)) {
    const PxU32 transientFrames =
        getComparisonTransientFrames(gHeadlessCase);
    if (gComparisonMetrics.lanes[0].transientSamples != transientFrames ||
        gComparisonMetrics.lanes[1].transientSamples != transientFrames)
      setGateError(evaluation, "transient_window_accounting");
  }
  if (gMetrics.fetchErrorState || gErrorCallback.getFatalCount())
    setGateFailure(evaluation, "physx_error");
  if (!PxIsFinite(gMetrics.targetMagnitude) ||
      gMetrics.targetMagnitude <= 0.0f)
    setGateError(evaluation, "missing_stimulus");

  if (gMetrics.maxQuaternionNormError > gMaximumQuaternionNormError)
    setGateFailure(evaluation, "quaternion_norm");
  if (gMetrics.maxAbsPosition > gMaximumAbsPosition ||
      gMetrics.maxLinearSpeed > gMaximumLinearSpeed ||
      gMetrics.maxAngularSpeed > gMaximumAngularSpeed)
    setGateFailure(evaluation, "runaway");
  if (gMetrics.maxActorAPositionError > gMaximumActorAPositionError ||
      gMetrics.maxActorAAngleError > gMaximumActorAAngleError) {
    if (!isDynamicComparisonTopology(gHeadlessConfig.topology) &&
        !isMovingKinematicPositionFixture())
      setGateFailure(evaluation, "actor_a_moved");
  }

  if (gHeadlessCase == eCASE_VELOCITY) {
    if (gMetrics.minLateAxisDot < gMinimumAxisDot)
      setGateFailure(evaluation, "drive_axis_or_sign");
    if (evaluation.lateOrthogonalRms > gMaximumOrthogonalRms)
      setGateFailure(evaluation, "orthogonal_leakage");
    if (evaluation.lateTargetMean < gMinimumLateProjection ||
        evaluation.lateTargetMean > gMaximumLateProjection ||
        evaluation.lateTargetRms > gMaximumLateTargetRms)
      setGateFailure(evaluation, "target_tracking");
    if (evaluation.motionWitness < gMinimumMotionWitness)
      setGateFailure(evaluation, "missing_motion");
  } else if (isPositionLikeCase(gHeadlessCase)) {
    const bool dynamicPositionPair =
        gHeadlessCase == eCASE_ANGULAR_POSITION &&
        isDynamicComparisonTopology(gHeadlessConfig.topology);
    const bool expectsOutputBreak =
        gHeadlessCase == eCASE_OUTPUT_FORCE &&
        ((gHeadlessConfig.outputForceEnabled &&
          gHeadlessConfig.breakMode == eBREAK_BELOW_DRIVE_LIMIT) ||
         (gHeadlessConfig.offsetAnchor &&
          gHeadlessConfig.breakMode == eBREAK_BELOW_OFFSET_MOMENT));
    if (!gPositionMetrics.actorOrderValid)
      setGateError(evaluation, "position_actor_order");
    if (gHeadlessCase == eCASE_ANGULAR_POSITION &&
        !gPositionMetrics.angularFrameWitnessValid)
      setGateError(evaluation, "angular_position_frame_fixture");
    if (isMovingKinematicPositionFixture()) {
      const PxU32 stationaryFrames = PxU32(
          PxFloor(gKinematicMotionStartTime / gHeadlessOptions.dt + 0.5f));
      const PxU32 expectedMotionFrames =
          gHeadlessOptions.frames - stationaryFrames;
      const PxReal expectedTravel =
          gKinematicAngularSpeed *
          (gPositionDuration - gKinematicMotionStartTime);
      if (gPositionMetrics.kinematicTargetFrames !=
              gHeadlessOptions.frames ||
          gPositionMetrics.kinematicMotionFrames !=
              expectedMotionFrames)
        setGateError(evaluation, "kinematic_target_accounting");
      if (PxAbs(gPositionMetrics.finalKinematicTravel -
                expectedTravel) > 1e-4f)
        setGateFailure(evaluation, "kinematic_motion_witness");
      if (gMetrics.maxActorAPositionError >
              gMaximumActorAPositionError ||
          gMetrics.maxActorAAngleError >
              gMaximumActorAAngleError ||
          gPositionMetrics.maximumKinematicAngularSpeedError >
              gMaximumKinematicAngularSpeedError)
        setGateFailure(evaluation, "kinematic_target_tracking");
      if (gPositionMetrics.finalErrorRatio >
              gMaximumMovingKinematicFinalErrorRatio ||
          evaluation.positionLateErrorRatio >
              gMaximumMovingKinematicLateErrorRatio)
        setGateFailure(evaluation,
                       "kinematic_relative_target_not_tracked");
    }
    const PxReal expectedForceLimit = getPositionForceLimit();
    if ((dynamicPositionPair &&
         (PxAbs(gPositionMetrics.actorAMassReadback - 1.0f) > 1e-5f ||
          PxAbs(gPositionMetrics.actorAInertiaReadback.x - 1.0f) > 1e-5f ||
          PxAbs(gPositionMetrics.actorAInertiaReadback.y - 1.0f) > 1e-5f ||
          PxAbs(gPositionMetrics.actorAInertiaReadback.z - 1.0f) > 1e-5f)) ||
        PxAbs(gPositionMetrics.massReadback -
              gHeadlessConfig.comparisonMass) > 1e-5f ||
        PxAbs(gPositionMetrics.inertiaReadback.x -
              gHeadlessConfig.comparisonMass) > 1e-5f ||
        PxAbs(gPositionMetrics.inertiaReadback.y -
              gHeadlessConfig.comparisonMass) > 1e-5f ||
        PxAbs(gPositionMetrics.inertiaReadback.z -
              gHeadlessConfig.comparisonMass) > 1e-5f ||
        PxAbs(gPositionMetrics.stiffnessReadback -
              gPositionDriveStiffness) > 1e-5f ||
        PxAbs(gPositionMetrics.dampingReadback -
              gPositionDriveDamping) > 1e-5f ||
        PxAbs(gPositionMetrics.forceLimitReadback - expectedForceLimit) >
            1e-5f ||
        !gPositionMetrics.driveLimitsAreForcesReadback)
      setGateError(evaluation, "position_fixture_readback");
    if (gHeadlessCase == eCASE_OUTPUT_FORCE &&
        gPositionMetrics.outputForceFlagReadback !=
            gHeadlessConfig.outputForceEnabled)
      setGateError(evaluation, "output_force_flag_readback");
    if (gPositionMetrics.targetReadbackError > 1e-5f)
      setGateError(evaluation, "position_target_readback");
    if (gPositionMetrics.initialTargetError <= 1e-6f)
      setGateError(evaluation, "missing_position_stimulus");
    const PxReal expectedInitialRelativeMagnitude =
        gHeadlessConfig.initialRelativeOffset
            ? gPositionInitialRelativeMagnitude
            : 0.0f;
    if (!PxIsFinite(gPositionMetrics.initialRelativeMagnitude) ||
        !PxIsFinite(gPositionMetrics.initialRelativeSetupError) ||
        PxAbs(gPositionMetrics.initialRelativeMagnitude -
              expectedInitialRelativeMagnitude) > 1e-4f ||
        gPositionMetrics.initialRelativeSetupError > 1e-4f)
      setGateError(evaluation, "initial_relative_setup");
    if (gPositionMetrics.minimumSignedProgress <
        -gMaximumPositionReverseMotionRatio *
            gPositionMetrics.initialTargetError)
      setGateFailure(evaluation, "position_wrong_direction");
    if ((!gHeadlessConfig.lowForceLimit ||
         gHeadlessCase == eCASE_ANGULAR_POSITION) &&
        (gPositionMetrics.finalErrorRatio > gMaximumPositionErrorRatio ||
         evaluation.positionLateErrorRatio >
             gMaximumLatePositionErrorRatio))
      setGateFailure(evaluation, "position_target_not_tracked");
    if (gHeadlessConfig.lowForceLimit) {
      const PxReal expectedAcceleration =
          dynamicPositionPair
              ? gPositionMetrics.expectedFirstRelativeAcceleration
              : gPositionLowForceLimit /
                    gPositionMetrics.inertiaReadback.x;
      if (gPositionMetrics.firstRelativeAcceleration <
              0.80f * expectedAcceleration ||
          gPositionMetrics.firstRelativeAcceleration >
              1.20f * expectedAcceleration)
        setGateFailure(evaluation, "position_force_limit_semantics");
      if (dynamicPositionPair) {
        const PxReal expectedAccelerationA =
            gPositionMetrics.expectedSignedAngularAccelerationA;
        const PxReal expectedAccelerationB =
            gPositionMetrics.expectedSignedAngularAccelerationB;
        if (PxAbs(gPositionMetrics.firstSignedAngularAccelerationA -
                  expectedAccelerationA) >
                0.20f * PxAbs(expectedAccelerationA) ||
            PxAbs(gPositionMetrics.firstSignedAngularAccelerationB -
                  expectedAccelerationB) >
                0.20f * PxAbs(expectedAccelerationB))
          setGateFailure(evaluation,
                         "angular_position_endpoint_response");
      }
    }
    if (dynamicPositionPair &&
        (gPositionMetrics.maximumCenterOfMassDrift >
             gDynamicAngularPositionMaximumComDrift ||
         gPositionMetrics.maximumLinearMomentum >
             gDynamicAngularPositionMaximumLinearMomentum))
      setGateFailure(evaluation,
                     "angular_position_linear_conservation");
    if (dynamicPositionPair &&
        gPositionMetrics.maximumAngularMomentum >
            gDynamicAngularPositionMaximumAngularMomentum)
      setGateFailure(evaluation,
                     "angular_position_angular_momentum");
    if (!passesContactCoverageGate())
      setGateFailure(evaluation, "contact_not_sustained");
    if (!passesContactSupportGate())
      setGateFailure(evaluation, "contact_support");
    if (!expectsOutputBreak &&
        evaluation.positionMotionRatio < gMinimumPositionMotionRatio)
      setGateFailure(evaluation, "missing_position_motion");
    const PxReal maximumOrthogonalError =
        gHeadlessCase == eCASE_ANGULAR_POSITION &&
                gHeadlessConfig.drive == eDRIVE_SLERP
            ? gMaximumSlerpPositionOrthogonalError
            : gMaximumPositionOrthogonalError;
    if (gPositionMetrics.maximumOrthogonalError > maximumOrthogonalError)
      setGateFailure(evaluation, "position_orthogonal_leakage");
    const PxReal maximumOvershootRatio =
        gHeadlessConfig.lowForceLimit ? 0.60f
                                      : gMaximumPositionOvershootRatio;
    if (!expectsOutputBreak &&
        gPositionMetrics.maximumOvershoot >
        maximumOvershootRatio * gPositionMetrics.initialTargetError)
      setGateFailure(evaluation, "position_overshoot");
    if (!expectsOutputBreak &&
        evaluation.positionLateSpeedRms > gMaximumLatePositionSpeed)
      setGateFailure(evaluation, "position_late_speed");
    if (gHeadlessCase == eCASE_OUTPUT_FORCE) {
      const PxReal expectedBreakForce =
          gHeadlessConfig.breakMode == eBREAK_BELOW_DRIVE_LIMIT
              ? gOutputForceBreakBelow
              : (gHeadlessConfig.breakMode == eBREAK_ABOVE_DRIVE_LIMIT
                     ? gOutputForceBreakAbove
                     : PX_MAX_F32);
      const PxReal expectedBreakTorque =
          gHeadlessConfig.breakMode == eBREAK_BELOW_OFFSET_MOMENT
              ? gOutputMomentBreakBelow
              : (gHeadlessConfig.breakMode == eBREAK_ABOVE_OFFSET_MOMENT
                     ? gOutputMomentBreakAbove
                     : PX_MAX_F32);
      if (PxAbs(gPositionMetrics.linearBreakForceReadback -
                expectedBreakForce) >
              (expectedBreakForce == PX_MAX_F32 ? 0.0f : 1e-5f) ||
          PxAbs(gPositionMetrics.angularBreakForceReadback -
                expectedBreakTorque) >
              (expectedBreakTorque == PX_MAX_F32 ? 0.0f : 1e-5f))
        setGateError(evaluation, "output_force_break_readback");
      if (gPositionMetrics.publicForceSampleAttempts !=
              gHeadlessOptions.frames ||
          gPositionMetrics.publicForceSamples != gHeadlessOptions.frames ||
          gPositionMetrics.nonFinitePublicForceSamples != 0)
        setGateError(evaluation, "output_force_sample_accounting");
      const PxReal expectedTorqueMagnitude = safeMagnitude(
          gPositionMetrics.expectedNormalizedPublicTorque);
      const PxReal firstTorqueMagnitude = safeMagnitude(
          gPositionMetrics.firstNormalizedPublicTorque);
      const PxReal firstTorqueError = safeMagnitude(
          gPositionMetrics.firstNormalizedPublicTorque -
          gPositionMetrics.expectedNormalizedPublicTorque);
      if (gHeadlessConfig.outputForceEnabled) {
        const PxReal firstMagnitude =
            safeMagnitude(gPositionMetrics.firstPublicForce);
        const PxReal firstAxisProjection =
            gPositionMetrics.firstPublicForce.dot(gMetrics.signedWorldAxis);
        const PxReal firstOrthogonal = safeMagnitude(
            gPositionMetrics.firstPublicForce -
            gMetrics.signedWorldAxis * firstAxisProjection);
        if (firstMagnitude < 0.80f * gPositionLowForceLimit)
          setGateFailure(evaluation, "output_force_missing");
        if (gPositionMetrics.firstSignedPublicForce <
                0.80f * gPositionLowForceLimit ||
            gPositionMetrics.firstSignedPublicForce >
                1.20f * gPositionLowForceLimit ||
            firstOrthogonal >
                gOutputForceDirectionTolerance * gPositionLowForceLimit ||
            gPositionMetrics.maximumPublicForce >
                1.20f * gPositionLowForceLimit)
          setGateFailure(evaluation, "output_force_semantics");
      } else if (PxAbs(gPositionMetrics.firstSignedPublicForce) >
                 gOutputForceZeroTolerance) {
        setGateFailure(evaluation, "output_force_disabled_nonzero");
      }
      // Hard angular rows are always output rows.  Their reaction moment must
      // therefore be invariant under the linear drive's eOUTPUT_FORCE flag.
      if (expectedTorqueMagnitude > gOutputTorqueTolerance) {
        const PxReal torqueDirectionDot =
            firstTorqueMagnitude > gOutputForceZeroTolerance
                ? gPositionMetrics.firstNormalizedPublicTorque.dot(
                      gPositionMetrics.expectedNormalizedPublicTorque) /
                      (firstTorqueMagnitude * expectedTorqueMagnitude)
                : -1.0f;
        if (firstTorqueMagnitude < 0.80f * expectedTorqueMagnitude)
          setGateFailure(evaluation, "output_moment_missing");
        if (torqueDirectionDot <= -0.99f)
          setGateFailure(evaluation, "output_moment_sign");
        if (firstTorqueMagnitude < 0.80f * expectedTorqueMagnitude ||
            firstTorqueMagnitude > 1.20f * expectedTorqueMagnitude)
          setGateFailure(evaluation, "output_moment_units");
        if (firstTorqueError >
            PxMax(gOutputTorqueTolerance, 0.10f * expectedTorqueMagnitude))
          setGateFailure(evaluation, "output_moment_arm");
      } else if (firstTorqueMagnitude > gOutputTorqueTolerance) {
        setGateFailure(evaluation, "output_moment_unexpected");
      }
      if (expectsOutputBreak) {
        if (gPositionMetrics.brokenSamples == 0 ||
            gPositionMetrics.firstBrokenFrame > 2)
          setGateFailure(
              evaluation,
              gHeadlessConfig.breakMode == eBREAK_BELOW_OFFSET_MOMENT
                  ? "output_moment_break_missing"
                  : "output_force_break_missing");
      } else if (gPositionMetrics.brokenSamples != 0) {
        setGateFailure(
            evaluation,
            gHeadlessConfig.breakMode == eBREAK_ABOVE_OFFSET_MOMENT
                ? "output_moment_break_unexpected"
                : "output_force_break_unexpected");
      }
    }
  } else if (isLegacyAngularLimitCase(gHeadlessCase)) {
    const bool inside =
        gHeadlessCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE;
    const PxReal expectedInitialY =
        inside ? gLegacyConeInsideY : gLegacyConeOutsideY;
    const PxReal expectedInitialZ =
        inside ? gLegacyConeInsideZ : gLegacyConeOutsideZ;
    const PxReal expectedInitialRadius =
        PxSqrt(PxSqr(expectedInitialY / gLegacyConeLimitY) +
               PxSqr(expectedInitialZ / gLegacyConeLimitZ));
    if (gAngularLimitMetrics.twistMotionReadback !=
            PxD6Motion::eLOCKED ||
        gAngularLimitMetrics.swing1MotionReadback !=
            PxD6Motion::eLIMITED ||
        gAngularLimitMetrics.swing2MotionReadback !=
            PxD6Motion::eLIMITED ||
        PxAbs(gAngularLimitMetrics.limitYReadback -
              gLegacyConeLimitY) > 1e-6f ||
        PxAbs(gAngularLimitMetrics.limitZReadback -
              gLegacyConeLimitZ) > 1e-6f)
      setGateError(evaluation, "legacy_cone_limit_readback");
    if (PxAbs(gAngularLimitMetrics.initialEllipseRadius -
              expectedInitialRadius) > 1e-4f)
      setGateError(evaluation, "legacy_cone_initial_fixture");
    if (inside) {
      if (gAngularLimitMetrics.initialEllipseRadius >= 1.0f)
        setGateError(evaluation, "legacy_cone_inside_fixture");
      if (gAngularLimitMetrics.maximumInsideEllipseDeviation >
          gLegacyConeInsideRadiusDeviationTolerance)
        setGateFailure(evaluation,
                       "legacy_cone_inside_state_disturbed");
    } else {
      if (gAngularLimitMetrics.initialEllipseRadius <= 1.0f)
        setGateError(evaluation, "legacy_cone_outside_fixture");
      if (gAngularLimitMetrics.finalEllipseRadius >
              1.0f + gLegacyConeFinalRadiusTolerance ||
          gAngularLimitMetrics.maximumLateEllipseRadius >
              1.0f + gLegacyConeLateRadiusTolerance ||
          evaluation.motionWitness <
              gLegacyConeMinimumRadiusCorrection)
        setGateFailure(evaluation,
                       "legacy_cone_limit_not_enforced");
    }
  } else {
    const JointDriveComparisonLaneMetrics &reference =
        gComparisonMetrics.lanes[0];
    const JointDriveComparisonLaneMetrics &test =
        gComparisonMetrics.lanes[1];
    const bool dynamicTopology =
        isDynamicComparisonTopology(gHeadlessConfig.topology);
    const PxReal orthogonalLimit =
        gComparisonMaximumOrthogonalScale * gMetrics.targetMagnitude + 1e-4f;
    if (reference.minimumAxisDot < gComparisonMinimumAxisDot ||
        test.minimumAxisDot < gComparisonMinimumAxisDot)
      setGateFailure(evaluation, "drive_axis_or_sign");
    if (reference.maximumOrthogonalSpeed > orthogonalLimit ||
        test.maximumOrthogonalSpeed > orthogonalLimit)
      setGateFailure(evaluation, "orthogonal_leakage");
    if (reference.maximumMonotonicDrop >
            gComparisonMaximumMonotonicDropScale *
                gMetrics.targetMagnitude ||
        test.maximumMonotonicDrop >
            gComparisonMaximumMonotonicDropScale *
                gMetrics.targetMagnitude ||
        reference.overshootCount || test.overshootCount)
      setGateFailure(evaluation, "invalid_transient_shape");
    const bool accelerationModeProbe =
        gHeadlessCase == eCASE_ACCELERATION_MODE;
    const bool accelerationLimitProbe =
        accelerationModeProbe && gHeadlessConfig.lowForceLimit;
    if (accelerationModeProbe) {
      if (reference.response < 0.50f || test.response < 0.50f)
        setGateFailure(evaluation, "missing_acceleration_response");
    } else if (reference.response < 0.50f) {
      setGateFailure(evaluation, "missing_reference_response");
    }
    for (PxU32 i = 0; i < 2; ++i) {
      if (!gComparisonMetrics.finiteReadback[i])
        setGateError(evaluation, "non_finite_readback");
      const PxReal expectedMass =
          gHeadlessCase == eCASE_FORCE_LIMIT || i == 0
              ? gMassProbeReferenceMass
              : gHeadlessConfig.comparisonMass;
      const PxVec3 inertia = gComparisonMetrics.inertiaReadback[i];
      if (PxAbs(gComparisonMetrics.massReadback[i] - expectedMass) > 1e-5f ||
          PxAbs(inertia.x - expectedMass) > 1e-5f ||
          PxAbs(inertia.y - expectedMass) > 1e-5f ||
          PxAbs(inertia.z - expectedMass) > 1e-5f)
        setGateError(evaluation, "mass_inertia_readback");
      if (isDynamicComparisonTopology(gHeadlessConfig.topology)) {
        const PxVec3 inertiaA =
            gComparisonMetrics.actorAInertiaReadback[i];
        if (PxAbs(gComparisonMetrics.actorAMassReadback[i] - expectedMass) >
                1e-5f ||
            PxAbs(inertiaA.x - expectedMass) > 1e-5f ||
            PxAbs(inertiaA.y - expectedMass) > 1e-5f ||
            PxAbs(inertiaA.z - expectedMass) > 1e-5f)
          setGateError(evaluation, "actor_a_mass_inertia_readback");
      }
      if (!gComparisonMetrics.driveLimitsAreForcesReadback[i])
        setGateError(evaluation, "drive_limit_flag_readback");
    }

    if (gHeadlessCase == eCASE_MASS_SCALING || accelerationModeProbe) {
      for (PxU32 i = 0; i < 2; ++i) {
        if (PxAbs(gComparisonMetrics.dampingReadback[i] -
                  gMassProbeDamping) > 1e-5f ||
            gComparisonMetrics.accelerationFlagReadback[i] !=
                accelerationModeProbe)
          setGateError(evaluation, "drive_mode_readback");
      }
      if (accelerationLimitProbe) {
        if (PxAbs(gComparisonMetrics.forceLimitReadback[0] - FLT_MAX) >
                0.0f ||
            PxAbs(gComparisonMetrics.forceLimitReadback[1] -
                  gAccelerationLimitProbeLowLimit) >
                1e-5f)
          setGateError(evaluation, "acceleration_limit_readback");
        const PxReal expectedRelativeAcceleration =
            gAccelerationLimitProbeLowLimit *
            (1.0f / gComparisonMetrics.actorAMassReadback[1] +
             1.0f / gComparisonMetrics.massReadback[1]);
        const PxReal firstTestAcceleration =
            test.firstProjection / gHeadlessOptions.dt;
        if (firstTestAcceleration <
                0.95f * expectedRelativeAcceleration ||
            firstTestAcceleration >
                1.05f * expectedRelativeAcceleration ||
            reference.firstProjection <
                1.5f * test.firstProjection ||
            test.peakAcceleration >
                1.15f * expectedRelativeAcceleration)
          setGateFailure(evaluation, "acceleration_force_limit");
      } else if (accelerationModeProbe) {
        if (evaluation.testToReferenceResponseRatio < 0.98f ||
            evaluation.testToReferenceResponseRatio > 1.02f ||
            evaluation.referenceToTestRateRatio < 0.98f ||
            evaluation.referenceToTestRateRatio > 1.02f)
          setGateFailure(evaluation, "acceleration_mass_scaling");
      } else if (gHeadlessConfig.comparisonMass == 1.0f) {
        if (evaluation.testToReferenceResponseRatio < 0.90f ||
            evaluation.testToReferenceResponseRatio > 1.10f ||
            evaluation.referenceToTestRateRatio < 0.90f ||
            evaluation.referenceToTestRateRatio > 1.10f)
          setGateFailure(evaluation, "mass_lane_control");
      } else if (evaluation.testToReferenceResponseRatio < 0.10f ||
                 evaluation.testToReferenceResponseRatio > 0.40f ||
                 evaluation.referenceToTestRateRatio < 6.0f ||
                 evaluation.referenceToTestRateRatio > 14.0f) {
        setGateFailure(evaluation, "force_mass_scaling");
      }
    } else {
      const PxReal expectedTestLimit =
          gHeadlessConfig.lowForceLimit ? gForceLimitProbeLowLimit
                                        : gForceLimitProbeHighLimit;
      if (PxAbs(gComparisonMetrics.dampingReadback[0] -
                gForceLimitProbeDamping) > 1e-5f ||
          PxAbs(gComparisonMetrics.dampingReadback[1] -
                gForceLimitProbeDamping) > 1e-5f ||
          PxAbs(gComparisonMetrics.forceLimitReadback[0] -
                gForceLimitProbeHighLimit) > 1e-3f ||
          PxAbs(gComparisonMetrics.forceLimitReadback[1] -
                expectedTestLimit) > 1e-3f ||
          gComparisonMetrics.accelerationFlagReadback[0] ||
          gComparisonMetrics.accelerationFlagReadback[1])
        setGateError(evaluation, "force_limit_readback");
      if (reference.frameFourProjection <
          0.85f * gForceLimitProbeTargetVelocity)
        setGateFailure(evaluation, "missing_high_limit_response");
      if (!gHeadlessConfig.lowForceLimit) {
        if (evaluation.testToReferenceResponseRatio < 0.95f ||
            evaluation.testToReferenceResponseRatio > 1.05f)
          setGateFailure(evaluation, "force_limit_lane_control");
      } else if (test.transientProjection <
                     (dynamicTopology ? 0.50f : 0.20f) *
                         gForceLimitProbeTargetVelocity ||
                 evaluation.testToReferenceResponseRatio <
                     (dynamicTopology ? 0.50f : 0.20f) ||
                 evaluation.testToReferenceResponseRatio >
                     (dynamicTopology ? 0.70f : 0.40f) ||
                 evaluation.normalizedImpulse < 0.85f ||
                 evaluation.normalizedImpulse > 1.15f ||
                 evaluation.meanTestAcceleration <
                     0.85f * gForceLimitProbeLowLimit *
                         (dynamicTopology ? 2.0f : 1.0f) ||
                 evaluation.meanTestAcceleration >
                     1.15f * gForceLimitProbeLowLimit *
                         (dynamicTopology ? 2.0f : 1.0f) ||
                 test.peakAcceleration >
                     1.15f * gForceLimitProbeLowLimit *
                         (dynamicTopology ? 2.0f : 1.0f)) {
        setGateFailure(evaluation, "force_limit_ignored");
      }
    }
    if (!passesContactCoverageGate())
      setGateFailure(evaluation, "contact_not_sustained");
    if (!passesContactSupportGate())
      setGateFailure(evaluation, "contact_support");
    if (!passesComparisonConservationGate())
      setGateFailure(
          evaluation,
          isContactComparisonTopology(gHeadlessConfig.topology)
              ? "contact_horizontal_conservation"
              : "internal_drive_conservation");
  }
  return evaluation;
}

static void printGateDetails(const JointDriveGateEvaluation &evaluation) {
  if (isLegacyAngularLimitCase(gHeadlessCase)) {
    printf(
        "[SnippetJointDriveAngularLimitDiag] case=%s "
        "limitY=%.9g limitZ=%.9g initialConeAngle=%.9g "
        "finalConeAngle=%.9g minimumConeAngle=%.9g "
        "maximumConeAngle=%.9g maximumLateConeAngle=%.9g "
        "maximumInsideDeviation=%.9g initialEllipseRadius=%.9g "
        "finalEllipseRadius=%.9g maximumLateEllipseRadius=%.9g "
        "maximumInsideEllipseDeviation=%.9g correction=%.9g "
        "lateSampleCount=%u maxAngularSpeed=%.9g\n",
        getCaseName(gHeadlessCase),
        double(gAngularLimitMetrics.limitYReadback),
        double(gAngularLimitMetrics.limitZReadback),
        double(gAngularLimitMetrics.initialConeAngle),
        double(gAngularLimitMetrics.finalConeAngle),
        double(gAngularLimitMetrics.minimumConeAngle),
        double(gAngularLimitMetrics.maximumConeAngle),
        double(gAngularLimitMetrics.maximumLateConeAngle),
        double(gAngularLimitMetrics.maximumInsideDeviation),
        double(gAngularLimitMetrics.initialEllipseRadius),
        double(gAngularLimitMetrics.finalEllipseRadius),
        double(gAngularLimitMetrics.maximumLateEllipseRadius),
        double(gAngularLimitMetrics.maximumInsideEllipseDeviation),
        double(evaluation.motionWitness),
        gAngularLimitMetrics.lateSampleCount,
        double(gMetrics.maxAngularSpeed));
    return;
  }
  if (isOrderingCase(gHeadlessCase)) {
    printf(
        "[SnippetJointDriveOrderingDiag] case=%s drive=%s quantity=%s "
        "endpoint=%s actor0=%s actor1=%s "
        "expectedAxis=(%.9g,%.9g,%.9g) "
        "expectedDynamicAxis=(%.9g,%.9g,%.9g) "
        "frameAxisSeparationDot=%.9g "
        "actorOrderValid=%u frameWitnessValid=%u driveReadbackValid=%u "
        "fixtureWitnessValid=%u relativeLateMean=%.9g "
        "relativeLateTargetRms=%.9g relativeLateOrthogonalRms=%.9g "
        "dynamicLateMean=%.9g dynamicLateTargetRms=%.9g "
        "dynamicLateOrthogonalRms=%.9g relativeMotion=%.9g "
        "dynamicMotion=%.9g maxRotationAngle=%.9g "
        "maxPositionError=%.9g maxLinearSpeed=%.9g maxAngularSpeed=%.9g\n",
        getCaseName(gHeadlessCase), getDriveName(gHeadlessConfig.drive),
        gOrderingMetrics.angular ? "angular" : "linear",
        getEndpointName(gHeadlessConfig.endpoint), getOrderingActor0Name(),
        getOrderingActor1Name(), double(gOrderingMetrics.expectedAxis.x),
        double(gOrderingMetrics.expectedAxis.y),
        double(gOrderingMetrics.expectedAxis.z),
        double(gOrderingMetrics.expectedDynamicAxis.x),
        double(gOrderingMetrics.expectedDynamicAxis.y),
        double(gOrderingMetrics.expectedDynamicAxis.z),
        double(gOrderingMetrics.frameAxisSeparationDot),
        gOrderingMetrics.actorOrderValid ? 1u : 0u,
        gOrderingMetrics.frameWitnessValid ? 1u : 0u,
        gOrderingMetrics.driveReadbackValid ? 1u : 0u,
        gOrderingMetrics.fixtureWitnessValid ? 1u : 0u,
        double(evaluation.lateTargetMean),
        double(evaluation.lateTargetRms),
        double(evaluation.lateOrthogonalRms),
        double(getOrderingLateDynamicMean()),
        double(getOrderingLateDynamicTargetRms()),
        double(getOrderingLateDynamicOrthogonalRms()),
        double(gOrderingMetrics.finalRelativeDisplacement),
        double(gOrderingMetrics.finalDynamicDisplacement),
        double(gOrderingMetrics.maxDynamicRotationError),
        double(gOrderingMetrics.maxDynamicPositionError),
        double(gOrderingMetrics.maxLinearSpeed),
        double(gOrderingMetrics.maxDynamicAngularSpeed));
    if (isAngularOutputForceCase(gHeadlessCase)) {
      printf(
          "[SnippetJointDriveAngularOutputDiag] endpoint=%s outputForce=%s "
          "outputForceFlagReadback=%u driveLimitsAreForces=%u "
          "firstPublicForce=(%.9g,%.9g,%.9g) "
          "firstPublicTorque=(%.9g,%.9g,%.9g) "
          "firstActor0FrameTorque=(%.9g,%.9g,%.9g) "
          "expectedActor0Torque=(%.9g,%.9g,%.9g) "
          "firstSignedPublicTorque=%.9g maximumPublicForce=%.9g "
          "maximumPublicTorque=%.9g firstRelativeProjection=%.9g "
          "firstDynamicProjection=%.9g firstDynamicAcceleration=%.9g "
          "firstDrivenDynamicAcceleration=%.9g "
          "maximumInitialDynamicAcceleration=%.9g "
          "publicForceSamples=%u/%u "
          "nonFinitePublicForceSamples=%u breakMode=%s "
          "linearBreakForceReadback=%.9g angularBreakForceReadback=%.9g "
          "brokenSamples=%u firstBrokenFrame=%u\n",
          getEndpointName(gHeadlessConfig.endpoint),
          gHeadlessConfig.outputForceEnabled ? "on" : "off",
          gOrderingMetrics.outputForceFlagReadback ? 1u : 0u,
          gOrderingMetrics.driveLimitsAreForcesReadback ? 1u : 0u,
          double(gOrderingMetrics.firstPublicForce.x),
          double(gOrderingMetrics.firstPublicForce.y),
          double(gOrderingMetrics.firstPublicForce.z),
          double(gOrderingMetrics.firstPublicTorque.x),
          double(gOrderingMetrics.firstPublicTorque.y),
          double(gOrderingMetrics.firstPublicTorque.z),
          double(gOrderingMetrics.firstActor0FrameTorque.x),
          double(gOrderingMetrics.firstActor0FrameTorque.y),
          double(gOrderingMetrics.firstActor0FrameTorque.z),
          double(gOrderingMetrics.expectedActor0Torque.x),
          double(gOrderingMetrics.expectedActor0Torque.y),
          double(gOrderingMetrics.expectedActor0Torque.z),
          double(gOrderingMetrics.firstSignedPublicTorque),
          double(gOrderingMetrics.maximumPublicForce),
          double(gOrderingMetrics.maximumPublicTorque),
          double(gOrderingMetrics.firstRelativeProjection),
          double(gOrderingMetrics.firstDynamicProjection),
          double(gOrderingMetrics.firstDynamicAcceleration),
          double(gOrderingMetrics.firstDrivenDynamicAcceleration),
          double(gOrderingMetrics.maximumInitialDynamicAcceleration),
          gOrderingMetrics.publicForceSamples,
          gOrderingMetrics.publicForceSampleAttempts,
          gOrderingMetrics.nonFinitePublicForceSamples,
          getBreakModeName(gHeadlessConfig.breakMode),
          double(gOrderingMetrics.linearBreakForceReadback),
          double(gOrderingMetrics.angularBreakForceReadback),
          gOrderingMetrics.brokenSamples,
          gOrderingMetrics.firstBrokenFrame);
    }
    return;
  }
  if (isComparisonCase(gHeadlessCase)) {
    const JointDriveComparisonLaneMetrics &reference =
        gComparisonMetrics.lanes[0];
    const JointDriveComparisonLaneMetrics &test =
        gComparisonMetrics.lanes[1];
    printf(
        "[SnippetJointDriveDiag] case=%s pairCount=%u drive=x driveMode=%s "
        "topology=%s endpoint=%s "
        "targetMagnitude=%.9g transientFrames=%u referenceProjection=%.9g "
        "testProjection=%.9g referenceResponse=%.9g testResponse=%.9g "
        "testToReferenceResponseRatio=%.9g referenceDecayRate=%.9g "
        "testDecayRate=%.9g referenceToTestRateRatio=%.9g "
        "referenceFirstProjection=%.9g testFirstProjection=%.9g "
        "testFirstAcceleration=%.9g "
        "referenceFrameFourProjection=%.9g testFrameFourProjection=%.9g "
        "expectedTestDeltaVelocity=%.9g normalizedImpulse=%.9g "
        "meanTestAcceleration=%.9g peakTestAcceleration=%.9g "
        "referenceMonotonicViolations=%u testMonotonicViolations=%u "
        "referenceMaximumMonotonicDrop=%.9g "
        "testMaximumMonotonicDrop=%.9g referenceOvershootCount=%u "
        "testOvershootCount=%u\n",
        getCaseName(gHeadlessCase), gMetrics.pairCountWitness,
        getDriveModeName(gHeadlessConfig.driveMode),
        getTopologyName(gHeadlessConfig.topology),
        getEndpointName(gHeadlessConfig.endpoint),
        double(gMetrics.targetMagnitude),
        getComparisonTransientFrames(gHeadlessCase),
        double(reference.transientProjection),
        double(test.transientProjection), double(reference.response),
        double(test.response),
        double(evaluation.testToReferenceResponseRatio),
        double(reference.decayRate), double(test.decayRate),
        double(evaluation.referenceToTestRateRatio),
        double(reference.firstProjection), double(test.firstProjection),
        double(test.firstProjection / gHeadlessOptions.dt),
        double(reference.frameFourProjection),
        double(test.frameFourProjection),
        double(evaluation.expectedTestDeltaVelocity),
        double(evaluation.normalizedImpulse),
        double(evaluation.meanTestAcceleration),
        double(test.peakAcceleration), reference.monotonicViolations,
        test.monotonicViolations, double(reference.maximumMonotonicDrop),
        double(test.maximumMonotonicDrop), reference.overshootCount,
        test.overshootCount);
    return;
  }

  if (isPositionLikeCase(gHeadlessCase)) {
    const bool dynamicPositionPair =
        gHeadlessCase == eCASE_ANGULAR_POSITION &&
        isDynamicComparisonTopology(gHeadlessConfig.topology);
    printf(
        "[SnippetJointDriveDiag] case=%s drive=%s actorA=%s frameA=%s "
        "frameB=%s bodyBRotation=%s initialRelative=%s anchor=%s driveMode=%s "
        "endpoint=%s kinematicMotion=%s actorOrderValid=%u "
        "angularFrameWitnessValid=%u "
        "worldFrameAxisDot=%.9g wrongRawFrameAxisDot=%.9g "
        "massReadback=%.9g inertiaReadback=(%.9g,%.9g,%.9g) "
        "stiffnessReadback=%.9g dampingReadback=%.9g "
        "forceLimitReadback=%.9g driveLimitsAreForces=%u "
        "targetDeltaMagnitude=%.9g targetRelativeMagnitude=%.9g "
        "initialRelativeMagnitude=%.9g "
        "initialRelativeSetupError=%.9g "
        "targetReadbackError=%.9g initialTargetError=%.9g "
        "finalTargetError=%.9g finalErrorRatio=%.9g lateErrorRms=%.9g "
        "lateErrorRatio=%.9g motionWitness=%.9g motionRatio=%.9g "
        "minimumSignedProgress=%.9g maximumOrthogonalError=%.9g "
        "maximumOvershoot=%.9g "
        "firstRelativeAcceleration=%.9g maximumRelativeAcceleration=%.9g "
        "lateSpeedRms=%.9g "
        "awakeSamples=%u actorAPositionError=%.9g "
        "actorAAngleError=%.9g kinematicTargetFrames=%u "
        "kinematicMotionFrames=%u finalKinematicTravel=%.9g "
        "maximumKinematicAngularSpeedError=%.9g outputForce=%s "
        "outputForceFlagReadback=%u firstPublicForce=(%.9g,%.9g,%.9g) "
        "firstPublicTorque=(%.9g,%.9g,%.9g) "
        "actor0WorldArm=(%.9g,%.9g,%.9g) "
        "dynamicWorldArm=(%.9g,%.9g,%.9g) "
        "expectedNormalizedPublicTorque=(%.9g,%.9g,%.9g) "
        "firstNormalizedPublicTorque=(%.9g,%.9g,%.9g) "
        "firstSignedPublicForce=%.9g maximumPublicForce=%.9g "
        "maximumPublicTorque=%.9g publicForceSamples=%u/%u "
        "nonFinitePublicForceSamples=%u breakMode=%s "
        "linearBreakForceReadback=%.9g angularBreakForceReadback=%.9g "
        "brokenSamples=%u firstBrokenFrame=%u\n",
        getCaseName(gHeadlessCase),
        getDriveName(gHeadlessConfig.drive),
        dynamicPositionPair
            ? "dynamic"
            : (gHeadlessConfig.actorAKinematic ? "kinematic" : "static"),
        getOrientationName(gHeadlessConfig.frameAOrientation),
        getOrientationName(gHeadlessConfig.frameBOrientation),
        getOrientationName(gHeadlessConfig.bodyBRotated),
        getInitialRelativeName(gHeadlessConfig.initialRelativeOffset),
        getAnchorName(gHeadlessConfig.offsetAnchor),
        getDriveModeName(gHeadlessConfig.driveMode),
        getEndpointName(gHeadlessConfig.endpoint),
        getKinematicMotionName(gHeadlessConfig.kinematicMotion),
        gPositionMetrics.actorOrderValid ? 1u : 0u,
        gPositionMetrics.angularFrameWitnessValid ? 1u : 0u,
        double(gPositionMetrics.worldFrameAxisDot),
        double(gPositionMetrics.wrongRawFrameAxisDot),
        double(gPositionMetrics.massReadback),
        double(gPositionMetrics.inertiaReadback.x),
        double(gPositionMetrics.inertiaReadback.y),
        double(gPositionMetrics.inertiaReadback.z),
        double(gPositionMetrics.stiffnessReadback),
        double(gPositionMetrics.dampingReadback),
        double(gPositionMetrics.forceLimitReadback),
        gPositionMetrics.driveLimitsAreForcesReadback ? 1u : 0u,
        double(gMetrics.targetMagnitude),
        double(gPositionMetrics.targetRelativeMagnitude),
        double(gPositionMetrics.initialRelativeMagnitude),
        double(gPositionMetrics.initialRelativeSetupError),
        double(gPositionMetrics.targetReadbackError),
        double(gPositionMetrics.initialTargetError),
        double(gPositionMetrics.finalTargetError),
        double(gPositionMetrics.finalErrorRatio),
        double(evaluation.positionLateErrorRms),
        double(evaluation.positionLateErrorRatio),
        double(evaluation.motionWitness),
        double(evaluation.positionMotionRatio),
        double(gPositionMetrics.minimumSignedProgress),
        double(gPositionMetrics.maximumOrthogonalError),
        double(gPositionMetrics.maximumOvershoot),
        double(gPositionMetrics.firstRelativeAcceleration),
        double(gPositionMetrics.maximumRelativeAcceleration),
        double(evaluation.positionLateSpeedRms),
        gPositionMetrics.awakeSamples,
        double(gMetrics.maxActorAPositionError),
        double(gMetrics.maxActorAAngleError),
        gPositionMetrics.kinematicTargetFrames,
        gPositionMetrics.kinematicMotionFrames,
        double(gPositionMetrics.finalKinematicTravel),
        double(gPositionMetrics.maximumKinematicAngularSpeedError),
        gHeadlessConfig.outputForceEnabled ? "on" : "off",
        gPositionMetrics.outputForceFlagReadback ? 1u : 0u,
        double(gPositionMetrics.firstPublicForce.x),
        double(gPositionMetrics.firstPublicForce.y),
        double(gPositionMetrics.firstPublicForce.z),
        double(gPositionMetrics.firstPublicTorque.x),
        double(gPositionMetrics.firstPublicTorque.y),
        double(gPositionMetrics.firstPublicTorque.z),
        double(gPositionMetrics.actor0WorldArm.x),
        double(gPositionMetrics.actor0WorldArm.y),
        double(gPositionMetrics.actor0WorldArm.z),
        double(gPositionMetrics.dynamicWorldArm.x),
        double(gPositionMetrics.dynamicWorldArm.y),
        double(gPositionMetrics.dynamicWorldArm.z),
        double(gPositionMetrics.expectedNormalizedPublicTorque.x),
        double(gPositionMetrics.expectedNormalizedPublicTorque.y),
        double(gPositionMetrics.expectedNormalizedPublicTorque.z),
        double(gPositionMetrics.firstNormalizedPublicTorque.x),
        double(gPositionMetrics.firstNormalizedPublicTorque.y),
        double(gPositionMetrics.firstNormalizedPublicTorque.z),
        double(gPositionMetrics.firstSignedPublicForce),
        double(gPositionMetrics.maximumPublicForce),
        double(gPositionMetrics.maximumPublicTorque),
        gPositionMetrics.publicForceSamples,
        gPositionMetrics.publicForceSampleAttempts,
        gPositionMetrics.nonFinitePublicForceSamples,
        getBreakModeName(gHeadlessConfig.breakMode),
        double(gPositionMetrics.linearBreakForceReadback),
        double(gPositionMetrics.angularBreakForceReadback),
        gPositionMetrics.brokenSamples, gPositionMetrics.firstBrokenFrame);
    return;
  }

  printf(
      "[SnippetJointDriveDiag] drive=%s actorA=%s frameA=%s frameB=%s "
      "bodyBRotation=%s driveMode=%s targetMagnitude=%.9g "
      "targetSign=%.9g signedWorldAxis=(%.9g,%.9g,%.9g) "
      "motionWitness=%.9g finalProjection=%.9g finalAxisDot=%.9g "
      "velocityIntegratedTravel=%.9g finalOrthogonalSpeed=%.9g "
      "lateTargetMean=%.9g lateTargetRms=%.9g "
      "lateOrthogonalRms=%.9g minLateAxisDot=%.9g "
      "actorAPositionError=%.9g actorAAngleError=%.9g\n",
      getDriveName(gHeadlessConfig.drive),
      gHeadlessConfig.actorAKinematic ? "kinematic" : "static",
      getOrientationName(gHeadlessConfig.frameAOrientation),
      getOrientationName(gHeadlessConfig.frameBOrientation),
      getOrientationName(gHeadlessConfig.bodyBRotated),
      getDriveModeName(gHeadlessConfig.driveMode),
      double(gMetrics.targetMagnitude),
      double(getRelativeTargetSign(gHeadlessConfig.drive)),
      double(gMetrics.signedWorldAxis.x), double(gMetrics.signedWorldAxis.y),
      double(gMetrics.signedWorldAxis.z), double(evaluation.motionWitness),
      double(gMetrics.finalSignedProjection), double(gMetrics.finalAxisDot),
      double(gMetrics.signedTravel), double(gMetrics.finalOrthogonalSpeed),
      double(evaluation.lateTargetMean), double(evaluation.lateTargetRms),
      double(evaluation.lateOrthogonalRms), double(gMetrics.minLateAxisDot),
      double(gMetrics.maxActorAPositionError),
      double(gMetrics.maxActorAAngleError));
}

static void printGateResult(const JointDriveGateEvaluation &evaluation,
                            PxU32 physicsErrors, PxU32 physicsWarnings) {
  if (isLegacyAngularLimitCase(gHeadlessCase)) {
    const bool inside =
        gHeadlessCase == eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE;
    printf(
        "[AVBD_GATE] schema=1 snippet=SnippetJointDrive case=%s "
        "solver=%s execution=%s requestedFrames=%u completedFrames=%u "
        "dt=%.9g seed=%u dispatcherThreads=%u capability=PARTIAL "
        "validation=GATED status=%s reason=%s nonFinite=%u "
        "physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
        "fetchErrorState=%u pairCount=%u limitKind=legacy-cone "
        "fixture=%s stimulusWitness=%u "
        "twistMotionReadback=%u swing1MotionReadback=%u "
        "swing2MotionReadback=%u limitY=%.9g limitZ=%.9g "
        "initialConeAngle=%.9g finalConeAngle=%.9g "
        "minimumConeAngle=%.9g maximumConeAngle=%.9g "
        "maximumLateConeAngle=%.9g maximumInsideDeviation=%.9g "
        "initialEllipseRadius=%.9g finalEllipseRadius=%.9g "
        "maximumLateEllipseRadius=%.9g "
        "maximumInsideEllipseDeviation=%.9g "
        "correction=%.9g sampleCount=%u lateSampleCount=%u "
        "maxQuaternionNormError=%.9g maxAbsPosition=%.9g "
        "maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
        "actorAPositionError=%.9g actorAAngleError=%.9g "
        "legacyAngularLimitGate=%s insideControlGate=%s "
        "positionDriveGate=NOT_COVERED "
        "forceLimitGate=NOT_COVERED outputForceGate=NOT_COVERED\n",
        getCaseName(gHeadlessCase),
        Snippets::getSolverTypeName(gHeadlessOptions.solverType),
        Snippets::getExecutionName(gHeadlessOptions.execution),
        gHeadlessOptions.frames, gMetrics.completedFrames,
        double(gHeadlessOptions.dt), gHeadlessOptions.seed,
        gHeadlessOptions.dispatcherThreads, evaluation.status,
        evaluation.reason, gMetrics.nonFinite ? 1u : 0u,
        physicsErrors, physicsWarnings, gMetrics.fetchFailures,
        gMetrics.fetchErrorState, gMetrics.pairCountWitness,
        inside ? "inside" : "outside",
        gAngularLimitMetrics.initialConeAngle > 0.0f ? 1u : 0u,
        PxU32(gAngularLimitMetrics.twistMotionReadback),
        PxU32(gAngularLimitMetrics.swing1MotionReadback),
        PxU32(gAngularLimitMetrics.swing2MotionReadback),
        double(gAngularLimitMetrics.limitYReadback),
        double(gAngularLimitMetrics.limitZReadback),
        double(gAngularLimitMetrics.initialConeAngle),
        double(gAngularLimitMetrics.finalConeAngle),
        double(gAngularLimitMetrics.minimumConeAngle),
        double(gAngularLimitMetrics.maximumConeAngle),
        double(gAngularLimitMetrics.maximumLateConeAngle),
        double(gAngularLimitMetrics.maximumInsideDeviation),
        double(gAngularLimitMetrics.initialEllipseRadius),
        double(gAngularLimitMetrics.finalEllipseRadius),
        double(gAngularLimitMetrics.maximumLateEllipseRadius),
        double(gAngularLimitMetrics.maximumInsideEllipseDeviation),
        double(evaluation.motionWitness), gMetrics.sampleCount,
        gAngularLimitMetrics.lateSampleCount,
        double(gMetrics.maxQuaternionNormError),
        double(gMetrics.maxAbsPosition), double(gMetrics.maxLinearSpeed),
        double(gMetrics.maxAngularSpeed),
        double(gMetrics.maxActorAPositionError),
        double(gMetrics.maxActorAAngleError), evaluation.status,
        inside ? evaluation.status : "NOT_APPLICABLE");
    return;
  }
  if (isOrderingCase(gHeadlessCase)) {
    printf(
        "[AVBD_GATE] schema=1 snippet=SnippetJointDrive "
        "case=%s solver=%s execution=%s requestedFrames=%u "
        "completedFrames=%u dt=%.9g seed=%u dispatcherThreads=%u "
        "capability=PARTIAL validation=PROBE status=%s reason=%s "
        "nonFinite=%u physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
        "fetchErrorState=%u endpoint=%s endpointActor0=%s endpointActor1=%s "
        "endpointProbe=1 actorOrderValid=%u frameWitnessValid=%u "
        "fixtureWitnessValid=%u driveReadbackValid=%u "
        "angularDriveConfigValid=%u drive=%s quantity=%s "
        "motionWitnessQuantity=%s driveMode=force targetConvention=%s "
        "relativeTargetSign=%.9g "
        "targetMagnitude=%.9g stimulusWitness=%u "
        "worldFramePositionError=%.9g dynamicLocalPositionError=%.9g "
        "worldFrameRotationDot=%.9g dynamicLocalRotationDot=%.9g "
        "actor0AxisDot=%.9g actor1AxisDot=%.9g "
        "dynamicLocalAxisDot=%.9g dynamicWorldAxisDot=%.9g "
        "frameAxisSeparationDot=%.9g "
        "expectedFrameAxisSeparationDot=%.9g "
        "bodyRotationDot=%.9g expectedAxisX=%.9g expectedAxisY=%.9g "
        "expectedAxisZ=%.9g expectedDynamicAxisX=%.9g "
        "expectedDynamicAxisY=%.9g expectedDynamicAxisZ=%.9g "
        "expectedFrameAAxisX=%.9g expectedFrameAAxisY=%.9g "
        "expectedFrameAAxisZ=%.9g expectedFrameBAxisX=%.9g "
        "expectedFrameBAxisY=%.9g expectedFrameBAxisZ=%.9g "
        "gravityMagnitude=%.9g shapeCount=%u freeMotionCount=%u "
        "massReadback=%.9g inertiaReadbackX=%.9g inertiaReadbackY=%.9g "
        "inertiaReadbackZ=%.9g linearDampingReadback=%.9g "
        "angularDampingReadback=%.9g driveStiffnessReadback=%.9g "
        "driveDampingReadback=%.9g driveForceLimitReadback=%.9g "
        "driveLinearTargetError=%.9g driveAngularTargetError=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u finalStaticActors=%u "
        "finalConstraints=%u cleanupDynamicActors=%u "
        "cleanupStaticActors=%u cleanupConstraints=%u cleanupComplete=%u "
        "sampleAttempts=%u sampleCount=%u nonFiniteSamples=%u "
        "sampleErrors=%u lateSampleAttempts=%u lateSampleCount=%u "
        "lateNonFiniteSamples=%u relativeLateMean=%.9g "
        "relativeLateTargetRms=%.9g relativeLateOrthogonalRms=%.9g "
        "minLateRelativeProjection=%.9g minLateRelativeDirectionDot=%.9g "
        "maxLateRelativeOrthogonal=%.9g dynamicLateMean=%.9g "
        "dynamicLateTargetRms=%.9g dynamicLateOrthogonalRms=%.9g "
        "minLateDynamicProjection=%.9g minLateDynamicDirectionDot=%.9g "
        "maxLateDynamicOrthogonal=%.9g finalRelativeProjection=%.9g "
        "finalRelativeDirectionDot=%.9g finalRelativeOrthogonal=%.9g "
        "finalDynamicProjection=%.9g finalDynamicDirectionDot=%.9g "
        "finalDynamicOrthogonal=%.9g relativeDisplacement=%.9g "
        "dynamicDisplacement=%.9g motionWitness=%.9g "
        "relativeMotion=%.9g dynamicMotion=%.9g "
        "maxDynamicRotationError=%.9g maxDynamicAngularSpeed=%.9g "
        "maxDynamicPositionError=%.9g "
        "maxQuaternionNormError=%.9g maxAbsPosition=%.9g "
        "maxLinearSpeed=%.9g driveLimitsAreForces=%u outputForce=%s "
        "outputForceFlagReadback=%u firstPublicForceX=%.9g "
        "firstPublicForceY=%.9g firstPublicForceZ=%.9g "
        "firstPublicTorqueX=%.9g firstPublicTorqueY=%.9g "
        "firstPublicTorqueZ=%.9g firstActor0FrameTorqueX=%.9g "
        "firstActor0FrameTorqueY=%.9g firstActor0FrameTorqueZ=%.9g "
        "expectedActor0TorqueX=%.9g expectedActor0TorqueY=%.9g "
        "expectedActor0TorqueZ=%.9g firstSignedPublicTorque=%.9g "
        "maximumPublicForce=%.9g maximumPublicTorque=%.9g "
        "firstRelativeProjection=%.9g firstDynamicProjection=%.9g "
        "firstDynamicAcceleration=%.9g "
        "firstDrivenDynamicAcceleration=%.9g "
        "maximumInitialDynamicAcceleration=%.9g "
        "publicForceSampleAttempts=%u publicForceSamples=%u "
        "nonFinitePublicForceSamples=%u breakMode=%s "
        "linearBreakForceReadback=%.9g angularBreakForceReadback=%.9g "
        "brokenSamples=%u firstBrokenFrame=%u "
        "positionDriveGate=NOT_COVERED "
        "accelerationDriveGate=NOT_COVERED massScalingGate=NOT_COVERED "
        "forceLimitGate=NOT_COVERED legacyImpulseLimitGate=NOT_COVERED "
        "angularOutputTorqueGate=%s outputForceGate=%s\n",
        getCaseName(gHeadlessCase),
        Snippets::getSolverTypeName(gHeadlessOptions.solverType),
        Snippets::getExecutionName(gHeadlessOptions.execution),
        gHeadlessOptions.frames, gOrderingMetrics.completedFrames,
        double(gHeadlessOptions.dt), gHeadlessOptions.seed,
        gHeadlessOptions.dispatcherThreads, evaluation.status,
        evaluation.reason, gOrderingMetrics.nonFinite ? 1u : 0u,
        physicsErrors, physicsWarnings, gOrderingMetrics.fetchFailures,
        gOrderingMetrics.fetchErrorState,
        getEndpointName(gHeadlessConfig.endpoint), getOrderingActor0Name(),
        getOrderingActor1Name(),
        gOrderingMetrics.actorOrderValid ? 1u : 0u,
        gOrderingMetrics.frameWitnessValid ? 1u : 0u,
        gOrderingMetrics.fixtureWitnessValid ? 1u : 0u,
        gOrderingMetrics.driveReadbackValid ? 1u : 0u,
        gOrderingMetrics.angularDriveConfigValid ? 1u : 0u,
        getDriveName(gHeadlessConfig.drive),
        gOrderingMetrics.angular ? "angular" : "linear",
        gOrderingMetrics.angular ? "angle" : "distance",
        gOrderingMetrics.angular &&
                gHeadlessConfig.drive != eDRIVE_SLERP
            ? "relative-a-minus-b"
            : "relative-b-minus-a",
        double(gOrderingMetrics.angular
                   ? getRelativeTargetSign(gHeadlessConfig.drive)
                   : 1.0f),
        double(gTargetVelocity),
        gOrderingMetrics.driveReadbackValid ? 1u : 0u,
        double(gOrderingMetrics.worldFramePositionError),
        double(gOrderingMetrics.dynamicLocalPositionError),
        double(gOrderingMetrics.worldFrameRotationDot),
        double(gOrderingMetrics.dynamicLocalRotationDot),
        double(gOrderingMetrics.actor0AxisDot),
        double(gOrderingMetrics.actor1AxisDot),
        double(gOrderingMetrics.dynamicLocalAxisDot),
        double(gOrderingMetrics.dynamicWorldAxisDot),
        double(gOrderingMetrics.frameAxisSeparationDot),
        double(gOrderingMetrics.expectedFrameAxisSeparationDot),
        double(gOrderingMetrics.bodyRotationDot),
        double(gOrderingMetrics.expectedAxis.x),
        double(gOrderingMetrics.expectedAxis.y),
        double(gOrderingMetrics.expectedAxis.z),
        double(gOrderingMetrics.expectedDynamicAxis.x),
        double(gOrderingMetrics.expectedDynamicAxis.y),
        double(gOrderingMetrics.expectedDynamicAxis.z),
        double(gOrderingMetrics.expectedFrameAAxis.x),
        double(gOrderingMetrics.expectedFrameAAxis.y),
        double(gOrderingMetrics.expectedFrameAAxis.z),
        double(gOrderingMetrics.expectedFrameBAxis.x),
        double(gOrderingMetrics.expectedFrameBAxis.y),
        double(gOrderingMetrics.expectedFrameBAxis.z),
        double(safeMagnitude(gOrderingMetrics.gravity)),
        gOrderingMetrics.shapeCount, gOrderingMetrics.freeMotionCount,
        double(gOrderingMetrics.massReadback),
        double(gOrderingMetrics.inertiaReadback.x),
        double(gOrderingMetrics.inertiaReadback.y),
        double(gOrderingMetrics.inertiaReadback.z),
        double(gOrderingMetrics.linearDampingReadback),
        double(gOrderingMetrics.angularDampingReadback),
        double(gOrderingMetrics.driveStiffnessReadback),
        double(gOrderingMetrics.driveDampingReadback),
        double(gOrderingMetrics.driveForceLimitReadback),
        double(gOrderingMetrics.driveLinearTargetError),
        double(gOrderingMetrics.driveAngularTargetError),
        gOrderingMetrics.initialDynamicActors,
        gOrderingMetrics.initialStaticActors,
        gOrderingMetrics.initialConstraints,
        gOrderingMetrics.finalDynamicActors,
        gOrderingMetrics.finalStaticActors, gOrderingMetrics.finalConstraints,
        gOrderingMetrics.cleanupDynamicActors,
        gOrderingMetrics.cleanupStaticActors,
        gOrderingMetrics.cleanupConstraints,
        gOrderingMetrics.cleanupComplete ? 1u : 0u,
        gOrderingMetrics.sampleAttempts, gOrderingMetrics.sampleCount,
        gOrderingMetrics.nonFiniteSamples, gOrderingMetrics.sampleErrors,
        gOrderingMetrics.lateSampleAttempts,
        gOrderingMetrics.lateSampleCount,
        gOrderingMetrics.lateNonFiniteSamples,
        double(evaluation.lateTargetMean),
        double(evaluation.lateTargetRms),
        double(evaluation.lateOrthogonalRms),
        double(gOrderingMetrics.minLateRelativeProjection),
        double(gOrderingMetrics.minLateRelativeDirectionDot),
        double(gOrderingMetrics.maxLateRelativeOrthogonal),
        double(getOrderingLateDynamicMean()),
        double(getOrderingLateDynamicTargetRms()),
        double(getOrderingLateDynamicOrthogonalRms()),
        double(gOrderingMetrics.minLateDynamicProjection),
        double(gOrderingMetrics.minLateDynamicDirectionDot),
        double(gOrderingMetrics.maxLateDynamicOrthogonal),
        double(gOrderingMetrics.finalRelativeProjection),
        double(gOrderingMetrics.finalRelativeDirectionDot),
        double(gOrderingMetrics.finalRelativeOrthogonal),
        double(gOrderingMetrics.finalDynamicProjection),
        double(gOrderingMetrics.finalDynamicDirectionDot),
        double(gOrderingMetrics.finalDynamicOrthogonal),
        double(gOrderingMetrics.finalRelativeDisplacement),
        double(gOrderingMetrics.finalDynamicDisplacement),
        double(evaluation.motionWitness),
        double(gOrderingMetrics.finalRelativeDisplacement),
        double(gOrderingMetrics.finalDynamicDisplacement),
        double(gOrderingMetrics.maxDynamicRotationError),
        double(gOrderingMetrics.maxDynamicAngularSpeed),
        double(gOrderingMetrics.maxDynamicPositionError),
        double(gOrderingMetrics.maxQuaternionNormError),
        double(gOrderingMetrics.maxAbsPosition),
        double(gOrderingMetrics.maxLinearSpeed),
        gOrderingMetrics.driveLimitsAreForcesReadback ? 1u : 0u,
        gHeadlessConfig.outputForceEnabled ? "on" : "off",
        gOrderingMetrics.outputForceFlagReadback ? 1u : 0u,
        double(gOrderingMetrics.firstPublicForce.x),
        double(gOrderingMetrics.firstPublicForce.y),
        double(gOrderingMetrics.firstPublicForce.z),
        double(gOrderingMetrics.firstPublicTorque.x),
        double(gOrderingMetrics.firstPublicTorque.y),
        double(gOrderingMetrics.firstPublicTorque.z),
        double(gOrderingMetrics.firstActor0FrameTorque.x),
        double(gOrderingMetrics.firstActor0FrameTorque.y),
        double(gOrderingMetrics.firstActor0FrameTorque.z),
        double(gOrderingMetrics.expectedActor0Torque.x),
        double(gOrderingMetrics.expectedActor0Torque.y),
        double(gOrderingMetrics.expectedActor0Torque.z),
        double(gOrderingMetrics.firstSignedPublicTorque),
        double(gOrderingMetrics.maximumPublicForce),
        double(gOrderingMetrics.maximumPublicTorque),
        double(gOrderingMetrics.firstRelativeProjection),
        double(gOrderingMetrics.firstDynamicProjection),
        double(gOrderingMetrics.firstDynamicAcceleration),
        double(gOrderingMetrics.firstDrivenDynamicAcceleration),
        double(gOrderingMetrics.maximumInitialDynamicAcceleration),
        gOrderingMetrics.publicForceSampleAttempts,
        gOrderingMetrics.publicForceSamples,
        gOrderingMetrics.nonFinitePublicForceSamples,
        getBreakModeName(gHeadlessConfig.breakMode),
        double(gOrderingMetrics.linearBreakForceReadback),
        double(gOrderingMetrics.angularBreakForceReadback),
        gOrderingMetrics.brokenSamples,
        gOrderingMetrics.firstBrokenFrame,
        isAngularOutputForceCase(gHeadlessCase) ? evaluation.status
                                                : "NOT_COVERED",
        isAngularOutputForceCase(gHeadlessCase) ? evaluation.status
                                                : "NOT_COVERED");
    return;
  }
  if (isComparisonCase(gHeadlessCase)) {
    const JointDriveComparisonLaneMetrics &reference =
        gComparisonMetrics.lanes[0];
    const JointDriveComparisonLaneMetrics &test =
        gComparisonMetrics.lanes[1];
    const PxReal referenceMass = gComparisonMetrics.massReadback[0];
    const PxReal testMass = gComparisonMetrics.massReadback[1];
    const PxReal referenceLimitImpulse =
        gComparisonMetrics.forceLimitReadback[0] * gHeadlessOptions.dt;
    const PxReal testLimitImpulse =
        gComparisonMetrics.forceLimitReadback[1] * gHeadlessOptions.dt;
    const bool laneControl =
        (gHeadlessCase == eCASE_MASS_SCALING &&
         gHeadlessConfig.comparisonMass == 1.0f) ||
        (gHeadlessCase == eCASE_ACCELERATION_MODE &&
         gHeadlessConfig.comparisonMass == 1.0f &&
         !gHeadlessConfig.lowForceLimit) ||
        (gHeadlessCase == eCASE_FORCE_LIMIT &&
         !gHeadlessConfig.lowForceLimit);
    const char *gateOutcome =
        laneControl && evaluation.exitCode == Snippets::eHEADLESS_PASS
            ? "CONTROL_PASS"
            : evaluation.status;
    const char *conservationGate =
        isDynamicComparisonTopology(gHeadlessConfig.topology)
            ? (passesComparisonConservationGate() ? "PASS" : "FAIL")
            : "NOT_COVERED";
    const char *contactCoverageGate =
        isContactComparisonTopology(gHeadlessConfig.topology)
            ? (passesContactCoverageGate() ? "PASS" : "FAIL")
            : "NOT_COVERED";
    const char *contactSupportGate =
        isContactComparisonTopology(gHeadlessConfig.topology)
            ? (passesContactSupportGate() ? "PASS" : "FAIL")
            : "NOT_COVERED";
    printf(
        "[AVBD_GATE] schema=1 snippet=SnippetJointDrive case=%s solver=%s "
        "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g "
        "seed=%u dispatcherThreads=%u capability=PARTIAL validation=PROBE "
        "status=%s reason=%s nonFinite=%u physicsErrors=%u "
        "physicsWarnings=%u fetchFailures=%u fetchErrorState=%u "
        "pairCount=%u comparisonKind=%s capabilityWitness=%u drive=x "
        "driveMode=%s referenceDriveMode=%s testDriveMode=%s "
        "stimulusWitness=%u "
        "targetMagnitude=%.9g transientFrames=%u transientDuration=%.9g "
        "referenceMass=%.9g testMass=%.9g "
        "referenceInertia=(%.9g,%.9g,%.9g) "
        "testInertia=(%.9g,%.9g,%.9g) damping=%.9g "
        "referenceProjection=%.9g testProjection=%.9g "
        "referenceResponse=%.9g testResponse=%.9g "
        "testToReferenceResponseRatio=%.9g referenceDecayRate=%.9g "
        "testDecayRate=%.9g referenceToTestRateRatio=%.9g "
        "referenceFirstProjection=%.9g testFirstProjection=%.9g "
        "testFirstAcceleration=%.9g "
        "referenceFrameFourProjection=%.9g testFrameFourProjection=%.9g "
        "referenceMonotonicViolations=%u testMonotonicViolations=%u "
        "referenceMaximumMonotonicDrop=%.9g "
        "testMaximumMonotonicDrop=%.9g referenceOvershootCount=%u "
        "testOvershootCount=%u "
        "referenceAxisDot=%.9g testAxisDot=%.9g "
        "referenceOrthogonalMax=%.9g testOrthogonalMax=%.9g "
        "referenceFinalRelativeVelocity=(%.9g,%.9g,%.9g) "
        "testFinalRelativeVelocity=(%.9g,%.9g,%.9g) "
        "referenceMaximumOrthogonalVelocity=(%.9g,%.9g,%.9g) "
        "testMaximumOrthogonalVelocity=(%.9g,%.9g,%.9g) "
        "referenceAccelerationFlagReadback=%u "
        "testAccelerationFlagReadback=%u driveLimitsAreForces=%u "
        "referenceLimit=%.9g testLimit=%.9g referenceLimitImpulse=%.9g "
        "testLimitImpulse=%.9g testDeltaV=%.9g expectedTestDeltaV=%.9g "
        "normalizedImpulse=%.9g meanTestAcceleration=%.9g "
        "peakTestAcceleration=%.9g sampleCount=%u "
        "referenceTransientSamples=%u testTransientSamples=%u "
        "maxQuaternionNormError=%.9g maxAbsPosition=%.9g "
        "maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
        "actorAPositionError=%.9g actorAAngleError=%.9g "
        "topology=%s endpoint=%s referenceActorAMass=%.9g "
        "testActorAMass=%.9g referenceComDriftMax=%.9g "
        "testComDriftMax=%.9g referenceMomentumMax=%.9g "
        "testMomentumMax=%.9g "
        "referenceBody0ContactFrames=%u referenceBody1ContactFrames=%u "
        "testBody0ContactFrames=%u testBody1ContactFrames=%u "
        "referenceBothContactFrames=%u testBothContactFrames=%u "
        "referenceContactPoints=%u testContactPoints=%u "
        "referenceMinimumBottom=%.9g testMinimumBottom=%.9g "
        "referenceMaximumVerticalSpeed=%.9g "
        "testMaximumVerticalSpeed=%.9g contactCoverageGate=%s "
        "contactSupportGate=%s conservationGate=%s "
        "positionDriveGate=NOT_COVERED accelerationDriveGate=%s "
        "massScalingGate=%s forceLimitGate=%s "
        "legacyImpulseLimitGate=NOT_COVERED outputForceGate=NOT_COVERED\n",
        getCaseName(gHeadlessCase),
        Snippets::getSolverTypeName(gHeadlessOptions.solverType),
        Snippets::getExecutionName(gHeadlessOptions.execution),
        gHeadlessOptions.frames, gMetrics.completedFrames,
        double(gHeadlessOptions.dt), gHeadlessOptions.seed,
        gHeadlessOptions.dispatcherThreads, evaluation.status,
        evaluation.reason, gMetrics.nonFinite ? 1u : 0u, physicsErrors,
        physicsWarnings, gMetrics.fetchFailures, gMetrics.fetchErrorState,
        gMetrics.pairCountWitness, laneControl ? "lane-control" : "capability",
        laneControl ? 0u : 1u,
        getDriveModeName(gHeadlessConfig.driveMode),
        gHeadlessCase == eCASE_ACCELERATION_MODE ? "acceleration" : "force",
        gHeadlessCase == eCASE_ACCELERATION_MODE ? "acceleration" : "force",
        reference.transientSamples && test.transientSamples ? 1u : 0u,
        double(gMetrics.targetMagnitude),
        getComparisonTransientFrames(gHeadlessCase),
        double(PxReal(getComparisonTransientFrames(gHeadlessCase)) *
               gHeadlessOptions.dt),
        double(referenceMass), double(testMass),
        double(gComparisonMetrics.inertiaReadback[0].x),
        double(gComparisonMetrics.inertiaReadback[0].y),
        double(gComparisonMetrics.inertiaReadback[0].z),
        double(gComparisonMetrics.inertiaReadback[1].x),
        double(gComparisonMetrics.inertiaReadback[1].y),
        double(gComparisonMetrics.inertiaReadback[1].z),
        double(gComparisonMetrics.dampingReadback[0]),
        double(reference.transientProjection),
        double(test.transientProjection), double(reference.response),
        double(test.response),
        double(evaluation.testToReferenceResponseRatio),
        double(reference.decayRate), double(test.decayRate),
        double(evaluation.referenceToTestRateRatio),
        double(reference.firstProjection), double(test.firstProjection),
        double(test.firstProjection / gHeadlessOptions.dt),
        double(reference.frameFourProjection),
        double(test.frameFourProjection), reference.monotonicViolations,
        test.monotonicViolations, double(reference.maximumMonotonicDrop),
        double(test.maximumMonotonicDrop), reference.overshootCount,
        test.overshootCount, double(reference.minimumAxisDot),
        double(test.minimumAxisDot),
        double(reference.maximumOrthogonalSpeed),
        double(test.maximumOrthogonalSpeed),
        double(reference.finalRelativeVelocity.x),
        double(reference.finalRelativeVelocity.y),
        double(reference.finalRelativeVelocity.z),
        double(test.finalRelativeVelocity.x),
        double(test.finalRelativeVelocity.y),
        double(test.finalRelativeVelocity.z),
        double(reference.maximumOrthogonalVelocity.x),
        double(reference.maximumOrthogonalVelocity.y),
        double(reference.maximumOrthogonalVelocity.z),
        double(test.maximumOrthogonalVelocity.x),
        double(test.maximumOrthogonalVelocity.y),
        double(test.maximumOrthogonalVelocity.z),
        gComparisonMetrics.accelerationFlagReadback[0] ? 1u : 0u,
        gComparisonMetrics.accelerationFlagReadback[1] ? 1u : 0u,
        gComparisonMetrics.driveLimitsAreForcesReadback[0] &&
                gComparisonMetrics.driveLimitsAreForcesReadback[1]
            ? 1u
            : 0u,
        double(gComparisonMetrics.forceLimitReadback[0]),
        double(gComparisonMetrics.forceLimitReadback[1]),
        double(referenceLimitImpulse), double(testLimitImpulse),
        double(test.transientProjection),
        double(evaluation.expectedTestDeltaVelocity),
        double(evaluation.normalizedImpulse),
        double(evaluation.meanTestAcceleration),
        double(test.peakAcceleration), gMetrics.sampleCount,
        reference.transientSamples, test.transientSamples,
        double(gMetrics.maxQuaternionNormError),
        double(gMetrics.maxAbsPosition), double(gMetrics.maxLinearSpeed),
        double(gMetrics.maxAngularSpeed),
        double(gMetrics.maxActorAPositionError),
        double(gMetrics.maxActorAAngleError),
        getTopologyName(gHeadlessConfig.topology),
        getEndpointName(gHeadlessConfig.endpoint),
        double(gComparisonMetrics.actorAMassReadback[0]),
        double(gComparisonMetrics.actorAMassReadback[1]),
        double(gComparisonMetrics.maximumCenterOfMassDrift[0]),
        double(gComparisonMetrics.maximumCenterOfMassDrift[1]),
        double(gComparisonMetrics.maximumMomentumMagnitude[0]),
        double(gComparisonMetrics.maximumMomentumMagnitude[1]),
        gComparisonMetrics.bodyContactFrames[0][0],
        gComparisonMetrics.bodyContactFrames[0][1],
        gComparisonMetrics.bodyContactFrames[1][0],
        gComparisonMetrics.bodyContactFrames[1][1],
        gComparisonMetrics.bothBodyContactFrames[0],
        gComparisonMetrics.bothBodyContactFrames[1],
        gComparisonMetrics.contactPointCount[0],
        gComparisonMetrics.contactPointCount[1],
        double(gComparisonMetrics.minimumBottom[0]),
        double(gComparisonMetrics.minimumBottom[1]),
        double(gComparisonMetrics.maximumAbsVerticalSpeed[0]),
        double(gComparisonMetrics.maximumAbsVerticalSpeed[1]),
        contactCoverageGate, contactSupportGate,
        conservationGate,
        gHeadlessCase == eCASE_ACCELERATION_MODE ? gateOutcome
                                                 : "NOT_COVERED",
        gHeadlessCase == eCASE_MASS_SCALING ? gateOutcome : "NOT_COVERED",
        gHeadlessCase == eCASE_FORCE_LIMIT ? gateOutcome : "NOT_COVERED");
    return;
  }

  if (isPositionLikeCase(gHeadlessCase)) {
    const bool dynamicPositionPair =
        gHeadlessCase == eCASE_ANGULAR_POSITION &&
        isDynamicComparisonTopology(gHeadlessConfig.topology);
    const bool outputOnlyFailure =
        gHeadlessCase == eCASE_OUTPUT_FORCE &&
        strncmp(evaluation.reason, "output_force_", 13) == 0;
    const bool expectedBreak =
        gHeadlessCase == eCASE_OUTPUT_FORCE &&
        gHeadlessConfig.outputForceEnabled &&
        gHeadlessConfig.breakMode == eBREAK_BELOW_DRIVE_LIMIT;
    const char *positionGate =
        expectedBreak ? "NOT_APPLICABLE"
                      : (outputOnlyFailure ? "PASS" : evaluation.status);
    const char *finiteLimitGate =
        outputOnlyFailure ? "PASS" : evaluation.status;
    const char *outputForceGate =
        gHeadlessCase == eCASE_OUTPUT_FORCE ? evaluation.status
                                            : "NOT_COVERED";
    const char *kinematicMotionGate =
        isMovingKinematicPositionFixture() ? evaluation.status
                                           : "NOT_COVERED";
    const char *validation =
        (gHeadlessConfig.topology == eTOPOLOGY_CONTACT_DYNAMIC_DYNAMIC ||
         isMovingKinematicPositionFixture())
            ? "GATED"
            : "PROBE";
    const bool conservationPass =
        gPositionMetrics.maximumCenterOfMassDrift <=
            gDynamicAngularPositionMaximumComDrift &&
        gPositionMetrics.maximumLinearMomentum <=
            gDynamicAngularPositionMaximumLinearMomentum &&
        gPositionMetrics.maximumAngularMomentum <=
            gDynamicAngularPositionMaximumAngularMomentum;
    const char *conservationGate =
        dynamicPositionPair
            ? (conservationPass ? "PASS" : "FAIL")
            : "NOT_COVERED";
    printf(
        "[AVBD_GATE] schema=1 snippet=SnippetJointDrive case=%s "
        "solver=%s execution=%s requestedFrames=%u completedFrames=%u "
        "dt=%.9g seed=%u dispatcherThreads=%u capability=PARTIAL "
        "validation=%s status=%s reason=%s nonFinite=%u "
        "physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
        "fetchErrorState=%u pairCount=%u drive=%s actorA=%s topology=%s "
        "frameA=%s "
        "frameB=%s bodyBRotation=%s initialRelative=%s anchor=%s driveMode=%s "
        "endpoint=%s kinematicMotion=%s actorOrderValid=%u "
        "angularFrameWitnessValid=%u "
        "worldFrameAxisDot=%.9g wrongRawFrameAxisDot=%.9g "
        "actorAMassReadback=%.9g actorAInertiaReadbackX=%.9g "
        "actorAInertiaReadbackY=%.9g actorAInertiaReadbackZ=%.9g "
        "massReadback=%.9g inertiaReadbackX=%.9g inertiaReadbackY=%.9g "
        "inertiaReadbackZ=%.9g stiffnessReadback=%.9g "
        "dampingReadback=%.9g forceLimitReadback=%.9g "
        "driveLimitsAreForces=%u lateWindowFrames=%u "
        "targetDeltaMagnitude=%.9g targetRelativeMagnitude=%.9g "
        "initialRelativeMagnitude=%.9g "
        "initialRelativeSetupError=%.9g "
        "stimulusWitness=%u targetReadbackError=%.9g "
        "initialTargetError=%.9g finalTargetError=%.9g "
        "finalErrorRatio=%.9g lateErrorRms=%.9g lateErrorRatio=%.9g "
        "motionWitness=%.9g motionRatio=%.9g minimumSignedProgress=%.9g "
        "maximumOrthogonalError=%.9g maximumOvershoot=%.9g "
        "firstRelativeAcceleration=%.9g maximumRelativeAcceleration=%.9g "
        "expectedFirstRelativeAcceleration=%.9g "
        "expectedSignedAngularAccelerationA=%.9g "
        "expectedSignedAngularAccelerationB=%.9g "
        "firstSignedAngularAccelerationA=%.9g "
        "firstSignedAngularAccelerationB=%.9g "
        "maximumCenterOfMassDrift=%.9g maximumLinearMomentum=%.9g "
        "maximumAngularMomentum=%.9g conservationGate=%s "
        "bodyContactFramesA=%u bodyContactFramesB=%u "
        "bothBodyContactFrames=%u contactPointCount=%u "
        "minimumBottom=%.9g maximumAbsVerticalSpeed=%.9g "
        "contactCoverageGate=%s contactSupportGate=%s "
        "lateSpeedRms=%.9g awakeSamples=%u "
        "sampleCount=%u lateSampleCount=%u maxQuaternionNormError=%.9g "
        "maxAbsPosition=%.9g maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
        "actorAPositionError=%.9g actorAAngleError=%.9g "
        "kinematicTargetFrames=%u kinematicMotionFrames=%u "
        "finalKinematicTravel=%.9g "
        "maximumKinematicAngularSpeedError=%.9g "
        "outputForce=%s outputForceFlagReadback=%u "
        "firstPublicForceX=%.9g firstPublicForceY=%.9g "
        "firstPublicForceZ=%.9g firstPublicTorqueX=%.9g "
        "firstPublicTorqueY=%.9g firstPublicTorqueZ=%.9g "
        "actor0WorldArmX=%.9g actor0WorldArmY=%.9g actor0WorldArmZ=%.9g "
        "dynamicWorldArmX=%.9g dynamicWorldArmY=%.9g dynamicWorldArmZ=%.9g "
        "expectedNormalizedPublicTorqueX=%.9g "
        "expectedNormalizedPublicTorqueY=%.9g "
        "expectedNormalizedPublicTorqueZ=%.9g "
        "firstNormalizedPublicTorqueX=%.9g "
        "firstNormalizedPublicTorqueY=%.9g "
        "firstNormalizedPublicTorqueZ=%.9g "
        "firstSignedPublicForce=%.9g maximumPublicForce=%.9g "
        "maximumPublicTorque=%.9g publicForceSampleAttempts=%u "
        "publicForceSamples=%u nonFinitePublicForceSamples=%u "
        "breakMode=%s linearBreakForceReadback=%.9g "
        "angularBreakForceReadback=%.9g brokenSamples=%u "
        "firstBrokenFrame=%u "
        "positionDriveGate=%s positionFiniteLimitGate=%s "
        "kinematicMotionGate=%s "
        "accelerationDriveGate=NOT_COVERED "
        "massScalingGate=NOT_COVERED forceLimitGate=NOT_COVERED "
        "outputForceGate=%s\n",
        getCaseName(gHeadlessCase),
        Snippets::getSolverTypeName(gHeadlessOptions.solverType),
        Snippets::getExecutionName(gHeadlessOptions.execution),
        gHeadlessOptions.frames, gMetrics.completedFrames,
        double(gHeadlessOptions.dt), gHeadlessOptions.seed,
        gHeadlessOptions.dispatcherThreads, validation,
        evaluation.status, evaluation.reason,
        gMetrics.nonFinite ? 1u : 0u, physicsErrors, physicsWarnings,
        gMetrics.fetchFailures, gMetrics.fetchErrorState,
        gMetrics.pairCountWitness, getDriveName(gHeadlessConfig.drive),
        dynamicPositionPair
            ? "dynamic"
            : (gHeadlessConfig.actorAKinematic ? "kinematic" : "static"),
        getTopologyName(gHeadlessConfig.topology),
        getOrientationName(gHeadlessConfig.frameAOrientation),
        getOrientationName(gHeadlessConfig.frameBOrientation),
        getOrientationName(gHeadlessConfig.bodyBRotated),
        getInitialRelativeName(gHeadlessConfig.initialRelativeOffset),
        getAnchorName(gHeadlessConfig.offsetAnchor),
        getDriveModeName(gHeadlessConfig.driveMode),
        getEndpointName(gHeadlessConfig.endpoint),
        getKinematicMotionName(gHeadlessConfig.kinematicMotion),
        gPositionMetrics.actorOrderValid ? 1u : 0u,
        gPositionMetrics.angularFrameWitnessValid ? 1u : 0u,
        double(gPositionMetrics.worldFrameAxisDot),
        double(gPositionMetrics.wrongRawFrameAxisDot),
        double(gPositionMetrics.actorAMassReadback),
        double(gPositionMetrics.actorAInertiaReadback.x),
        double(gPositionMetrics.actorAInertiaReadback.y),
        double(gPositionMetrics.actorAInertiaReadback.z),
        double(gPositionMetrics.massReadback),
        double(gPositionMetrics.inertiaReadback.x),
        double(gPositionMetrics.inertiaReadback.y),
        double(gPositionMetrics.inertiaReadback.z),
        double(gPositionMetrics.stiffnessReadback),
        double(gPositionMetrics.dampingReadback),
        double(gPositionMetrics.forceLimitReadback),
        gPositionMetrics.driveLimitsAreForcesReadback ? 1u : 0u,
        getPositionLateWindowFrames(),
        double(gMetrics.targetMagnitude),
        double(gPositionMetrics.targetRelativeMagnitude),
        double(gPositionMetrics.initialRelativeMagnitude),
        double(gPositionMetrics.initialRelativeSetupError),
        gPositionMetrics.initialTargetError > 0.0f ? 1u : 0u,
        double(gPositionMetrics.targetReadbackError),
        double(gPositionMetrics.initialTargetError),
        double(gPositionMetrics.finalTargetError),
        double(gPositionMetrics.finalErrorRatio),
        double(evaluation.positionLateErrorRms),
        double(evaluation.positionLateErrorRatio),
        double(evaluation.motionWitness),
        double(evaluation.positionMotionRatio),
        double(gPositionMetrics.minimumSignedProgress),
        double(gPositionMetrics.maximumOrthogonalError),
        double(gPositionMetrics.maximumOvershoot),
        double(gPositionMetrics.firstRelativeAcceleration),
        double(gPositionMetrics.maximumRelativeAcceleration),
        double(gPositionMetrics.expectedFirstRelativeAcceleration),
        double(gPositionMetrics.expectedSignedAngularAccelerationA),
        double(gPositionMetrics.expectedSignedAngularAccelerationB),
        double(gPositionMetrics.firstSignedAngularAccelerationA),
        double(gPositionMetrics.firstSignedAngularAccelerationB),
        double(gPositionMetrics.maximumCenterOfMassDrift),
        double(gPositionMetrics.maximumLinearMomentum),
        double(gPositionMetrics.maximumAngularMomentum),
        conservationGate,
        gComparisonMetrics.bodyContactFrames[0][0],
        gComparisonMetrics.bodyContactFrames[0][1],
        gComparisonMetrics.bothBodyContactFrames[0],
        gComparisonMetrics.contactPointCount[0],
        double(gComparisonMetrics.minimumBottom[0]),
        double(gComparisonMetrics.maximumAbsVerticalSpeed[0]),
        isContactComparisonTopology(gHeadlessConfig.topology)
            ? (passesContactCoverageGate() ? "PASS" : "FAIL")
            : "NOT_COVERED",
        isContactComparisonTopology(gHeadlessConfig.topology)
            ? (passesContactSupportGate() ? "PASS" : "FAIL")
            : "NOT_COVERED",
        double(evaluation.positionLateSpeedRms), gPositionMetrics.awakeSamples,
        gMetrics.sampleCount, gPositionMetrics.lateSampleCount,
        double(gMetrics.maxQuaternionNormError),
        double(gMetrics.maxAbsPosition), double(gMetrics.maxLinearSpeed),
        double(gMetrics.maxAngularSpeed),
        double(gMetrics.maxActorAPositionError),
        double(gMetrics.maxActorAAngleError),
        gPositionMetrics.kinematicTargetFrames,
        gPositionMetrics.kinematicMotionFrames,
        double(gPositionMetrics.finalKinematicTravel),
        double(gPositionMetrics.maximumKinematicAngularSpeedError),
        gHeadlessConfig.outputForceEnabled ? "on" : "off",
        gPositionMetrics.outputForceFlagReadback ? 1u : 0u,
        double(gPositionMetrics.firstPublicForce.x),
        double(gPositionMetrics.firstPublicForce.y),
        double(gPositionMetrics.firstPublicForce.z),
        double(gPositionMetrics.firstPublicTorque.x),
        double(gPositionMetrics.firstPublicTorque.y),
        double(gPositionMetrics.firstPublicTorque.z),
        double(gPositionMetrics.actor0WorldArm.x),
        double(gPositionMetrics.actor0WorldArm.y),
        double(gPositionMetrics.actor0WorldArm.z),
        double(gPositionMetrics.dynamicWorldArm.x),
        double(gPositionMetrics.dynamicWorldArm.y),
        double(gPositionMetrics.dynamicWorldArm.z),
        double(gPositionMetrics.expectedNormalizedPublicTorque.x),
        double(gPositionMetrics.expectedNormalizedPublicTorque.y),
        double(gPositionMetrics.expectedNormalizedPublicTorque.z),
        double(gPositionMetrics.firstNormalizedPublicTorque.x),
        double(gPositionMetrics.firstNormalizedPublicTorque.y),
        double(gPositionMetrics.firstNormalizedPublicTorque.z),
        double(gPositionMetrics.firstSignedPublicForce),
        double(gPositionMetrics.maximumPublicForce),
        double(gPositionMetrics.maximumPublicTorque),
        gPositionMetrics.publicForceSampleAttempts,
        gPositionMetrics.publicForceSamples,
        gPositionMetrics.nonFinitePublicForceSamples,
        getBreakModeName(gHeadlessConfig.breakMode),
        double(gPositionMetrics.linearBreakForceReadback),
        double(gPositionMetrics.angularBreakForceReadback),
        gPositionMetrics.brokenSamples, gPositionMetrics.firstBrokenFrame,
        positionGate,
        gHeadlessConfig.lowForceLimit ? finiteLimitGate : "NOT_COVERED",
        kinematicMotionGate,
        outputForceGate);
    return;
  }

  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetJointDrive case=velocity "
      "solver=%s execution=%s requestedFrames=%u completedFrames=%u "
      "dt=%.9g seed=%u dispatcherThreads=%u capability=PARTIAL "
      "validation=GATED status=%s reason=%s nonFinite=%u "
      "physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
      "fetchErrorState=%u drive=%s actorA=%s frameA=%s frameB=%s "
      "bodyBRotation=%s driveMode=%s targetMagnitude=%.9g "
      "stimulusWitness=%u motionWitness=%.9g signedAxisProjection=%.9g "
      "velocityIntegratedTravel=%.9g axisDot=%.9g orthogonalRms=%.9g "
      "lateTargetMean=%.9g "
      "lateTargetRms=%.9g minLateAxisDot=%.9g sampleCount=%u "
      "lateSampleCount=%u maxQuaternionNormError=%.9g "
      "maxAbsPosition=%.9g maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
      "actorAPositionError=%.9g actorAAngleError=%.9g "
      "positionDriveGate=NOT_COVERED accelerationDriveGate=NOT_COVERED "
      "massScalingGate=NOT_COVERED forceLimitGate=NOT_COVERED "
      "outputForceGate=NOT_COVERED\n",
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, evaluation.status,
      evaluation.reason, gMetrics.nonFinite ? 1u : 0u, physicsErrors,
      physicsWarnings, gMetrics.fetchFailures, gMetrics.fetchErrorState,
      getDriveName(gHeadlessConfig.drive),
      gHeadlessConfig.actorAKinematic ? "kinematic" : "static",
      getOrientationName(gHeadlessConfig.frameAOrientation),
      getOrientationName(gHeadlessConfig.frameBOrientation),
      getOrientationName(gHeadlessConfig.bodyBRotated),
      getDriveModeName(gHeadlessConfig.driveMode),
      double(gMetrics.targetMagnitude),
      gMetrics.targetMagnitude > 0.0f ? 1u : 0u,
      double(evaluation.motionWitness), double(gMetrics.finalSignedProjection),
      double(gMetrics.signedTravel), double(gMetrics.finalAxisDot),
      double(evaluation.lateOrthogonalRms),
      double(evaluation.lateTargetMean), double(evaluation.lateTargetRms),
      double(gMetrics.minLateAxisDot), gMetrics.sampleCount,
      gMetrics.lateSampleCount, double(gMetrics.maxQuaternionNormError),
      double(gMetrics.maxAbsPosition), double(gMetrics.maxLinearSpeed),
      double(gMetrics.maxAngularSpeed),
      double(gMetrics.maxActorAPositionError),
      double(gMetrics.maxActorAAngleError));
}

static int reportConfigurationError(const Snippets::HeadlessOptions &options,
                                    const JointDriveHeadlessConfig &config,
                                    const char *reason) {
  const std::string reasonToken = makeAuthorityToken(reason);
  JointDriveCase requestedCase = eCASE_VELOCITY;
  const bool knownCase = tryParseCase(options.caseName.c_str(), requestedCase);
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetJointDrive case=config-error "
      "solver=%s "
      "execution=%s requestedFrames=%u completedFrames=0 dt=%.9g seed=%u "
      "dispatcherThreads=%u capability=PARTIAL validation=%s "
      "status=ERROR reason=%s nonFinite=0 physicsErrors=0 "
      "physicsWarnings=0 drive=%s actorA=%s frameA=%s frameB=%s "
      "bodyBRotation=%s initialRelative=%s anchor=%s driveMode=%s topology=%s "
      "endpoint=%s kinematicMotion=%s\n",
      Snippets::getSolverTypeName(options.solverType),
      Snippets::getExecutionName(options.execution), options.frames,
      double(options.dt), options.seed, options.dispatcherThreads,
      knownCase ? getValidationName(requestedCase) : "GATED",
      reasonToken.c_str(), getDriveName(config.drive),
      config.actorAKinematic ? "kinematic" : "static",
      getOrientationName(config.frameAOrientation),
      getOrientationName(config.frameBOrientation),
      getOrientationName(config.bodyBRotated),
      getInitialRelativeName(config.initialRelativeOffset),
      getAnchorName(config.offsetAnchor),
      getDriveModeName(config.driveMode), getTopologyName(config.topology),
      getEndpointName(config.endpoint),
      getKinematicMotionName(config.kinematicMotion));
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
    const char *value =
        arg + strlen(caseOption ? "--case=" : "--scenario=");
    // Match parseCommonHeadlessOptions: the first case/scenario argument is
    // authoritative even if a later duplicate names a known case.
    options.caseName = value;
    return;
  }
}

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  Snippets::HeadlessOptions defaults;
  defaults.caseName = "velocity";
  defaults.frames = 180;
  defaults.seed = 1;
  defaults.dispatcherThreads = 2;
  defaults.dt = 1.0f / 60.0f;

  Snippets::HeadlessOptions options;
  JointDriveHeadlessConfig config;
  std::string parseError;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
                                            parseError)) {
    recoverCaseHintForConfigurationError(argc, argv, options);
    return reportConfigurationError(options, config, parseError.c_str());
  }

  JointDriveOptionSeen seen;
  bool headlessOnlyOptionSeen = false;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!arg)
      continue;
    if (Snippets::isCommonHeadlessOption(arg)) {
      if (strcmp(arg, "--headless") != 0 &&
          !Snippets::hasOptionPrefix(arg, "--solver="))
        headlessOnlyOptionSeen = true;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--drive=")) {
      if (seen.drive)
        return reportConfigurationError(options, config,
                                        "duplicate_--drive");
      seen.drive = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseDrive(arg + strlen("--drive="), config.drive))
        return reportConfigurationError(options, config,
                                        "invalid_--drive_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--drive-mode=")) {
      if (seen.driveMode)
        return reportConfigurationError(options, config,
                                        "duplicate_--drive-mode");
      seen.driveMode = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseDriveMode(arg + strlen("--drive-mode="),
                             config.driveMode))
        return reportConfigurationError(options, config,
                                        "invalid_--drive-mode_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--actor-a=")) {
      if (seen.actorA)
        return reportConfigurationError(options, config,
                                        "duplicate_--actor-a");
      seen.actorA = true;
      headlessOnlyOptionSeen = true;
      const char *value = arg + strlen("--actor-a=");
      if (Snippets::equalsIgnoreCase(value, "static"))
        config.actorAKinematic = false;
      else if (Snippets::equalsIgnoreCase(value, "kinematic"))
        config.actorAKinematic = true;
      else
        return reportConfigurationError(options, config,
                                        "invalid_--actor-a_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--frame-a=")) {
      if (seen.frameA)
        return reportConfigurationError(options, config,
                                        "duplicate_--frame-a");
      seen.frameA = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseOrientation(arg + strlen("--frame-a="),
                               config.frameAOrientation))
        return reportConfigurationError(options, config,
                                        "invalid_--frame-a_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--frame-b=")) {
      if (seen.frameB)
        return reportConfigurationError(options, config,
                                        "duplicate_--frame-b");
      seen.frameB = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseOrientation(arg + strlen("--frame-b="),
                               config.frameBOrientation))
        return reportConfigurationError(options, config,
                                        "invalid_--frame-b_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--body-b-rotation=")) {
      if (seen.bodyB)
        return reportConfigurationError(options, config,
                                        "duplicate_--body-b-rotation");
      seen.bodyB = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseOrientation(arg + strlen("--body-b-rotation="),
                               config.bodyBRotated))
        return reportConfigurationError(
            options, config, "invalid_--body-b-rotation_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--initial-relative=")) {
      if (seen.initialRelative)
        return reportConfigurationError(options, config,
                                        "duplicate_--initial-relative");
      seen.initialRelative = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseInitialRelative(arg + strlen("--initial-relative="),
                                   config.initialRelativeOffset))
        return reportConfigurationError(
            options, config, "invalid_--initial-relative_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--anchor=")) {
      if (seen.anchor)
        return reportConfigurationError(options, config,
                                        "duplicate_--anchor");
      seen.anchor = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseAnchor(arg + strlen("--anchor="), config.offsetAnchor))
        return reportConfigurationError(options, config,
                                        "invalid_--anchor_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--mass=")) {
      if (seen.mass)
        return reportConfigurationError(options, config,
                                        "duplicate_--mass");
      seen.mass = true;
      headlessOnlyOptionSeen = true;
      PxReal parsedMass = 0.0f;
      if (!Snippets::parseReal(arg + strlen("--mass="), 1.0f, 10.0f,
                               parsedMass) ||
          (PxAbs(parsedMass - 1.0f) > 1e-6f &&
           PxAbs(parsedMass - 10.0f) > 1e-6f))
        return reportConfigurationError(options, config,
                                        "invalid_--mass_value");
      config.comparisonMass =
          PxAbs(parsedMass - 1.0f) <= 1e-6f ? 1.0f : 10.0f;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--limit=")) {
      if (seen.limit)
        return reportConfigurationError(options, config,
                                        "duplicate_--limit");
      seen.limit = true;
      headlessOnlyOptionSeen = true;
      const char *value = arg + strlen("--limit=");
      if (Snippets::equalsIgnoreCase(value, "low"))
        config.lowForceLimit = true;
      else if (Snippets::equalsIgnoreCase(value, "high"))
        config.lowForceLimit = false;
      else
        return reportConfigurationError(options, config,
                                        "invalid_--limit_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--output-force=")) {
      if (seen.outputForce)
        return reportConfigurationError(options, config,
                                        "duplicate_--output-force");
      seen.outputForce = true;
      headlessOnlyOptionSeen = true;
      const char *value = arg + strlen("--output-force=");
      if (Snippets::equalsIgnoreCase(value, "on"))
        config.outputForceEnabled = true;
      else if (Snippets::equalsIgnoreCase(value, "off"))
        config.outputForceEnabled = false;
      else
        return reportConfigurationError(options, config,
                                        "invalid_--output-force_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--break=")) {
      if (seen.breakMode)
        return reportConfigurationError(options, config,
                                        "duplicate_--break");
      seen.breakMode = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseBreakMode(arg + strlen("--break="), config.breakMode))
        return reportConfigurationError(options, config,
                                        "invalid_--break_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--endpoint=")) {
      if (seen.endpoint)
        return reportConfigurationError(options, config,
                                        "duplicate_--endpoint");
      seen.endpoint = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseEndpoint(arg + strlen("--endpoint="), config.endpoint))
        return reportConfigurationError(options, config,
                                        "invalid_--endpoint_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--topology=")) {
      if (seen.topology)
        return reportConfigurationError(options, config,
                                        "duplicate_--topology");
      seen.topology = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseTopology(arg + strlen("--topology="), config.topology))
        return reportConfigurationError(options, config,
                                        "invalid_--topology_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--kinematic-motion=")) {
      if (seen.kinematicMotion)
        return reportConfigurationError(
            options, config, "duplicate_--kinematic-motion");
      seen.kinematicMotion = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseKinematicMotion(
              arg + strlen("--kinematic-motion="),
              config.kinematicMotion))
        return reportConfigurationError(
            options, config, "invalid_--kinematic-motion_value");
      continue;
    }
    return reportConfigurationError(options, config, "unknown_argument");
  }

#ifndef RENDER_SNIPPET
  options.headless = true;
#endif

  JointDriveCase testCase = eCASE_VELOCITY;
  if (!tryParseCase(options.caseName.c_str(), testCase))
    return reportConfigurationError(options, config, "invalid_--case_value");
  options.caseName = getCaseName(testCase);
  if (!options.headless && headlessOnlyOptionSeen)
    return reportConfigurationError(options, config,
                                    "gate_option_requires_--headless");
  if (!isPositionLikeCase(testCase) &&
      !isAngularOutputForceCase(testCase) && options.frames < 120)
    return reportConfigurationError(options, config,
                                    "frames_must_be_at_least_120");
  const bool contactComparison =
      isContactComparisonTopology(config.topology);
  const bool accelerationLimitComparison =
      testCase == eCASE_ACCELERATION_MODE && seen.limit;
  if (contactComparison || accelerationLimitComparison) {
    const bool supportedDt =
        PxAbs(options.dt - (1.0f / 30.0f)) <= 1e-7f ||
        PxAbs(options.dt - (1.0f / 60.0f)) <= 1e-7f ||
        PxAbs(options.dt - (1.0f / 120.0f)) <= 1e-7f;
    if (!supportedDt)
      return reportConfigurationError(
          options, config,
          contactComparison ? "contact_topology_requires_30_60_120hz"
                            : "acceleration_limit_requires_30_60_120hz");
    const PxU32 expectedFrames =
        PxU32(PxFloor(10.0f / options.dt + 0.5f));
    if (options.frames != expectedFrames)
      return reportConfigurationError(
          options, config,
          contactComparison ? "contact_topology_requires_10_seconds"
                            : "acceleration_limit_requires_10_seconds");
  } else if (isPositionLikeCase(testCase) ||
             isAngularOutputForceCase(testCase)) {
    const bool supportedDt =
        PxAbs(options.dt - (1.0f / 30.0f)) <= 1e-7f ||
        PxAbs(options.dt - (1.0f / 60.0f)) <= 1e-7f ||
        PxAbs(options.dt - (1.0f / 120.0f)) <= 1e-7f;
    if (!supportedDt)
      return reportConfigurationError(
          options, config,
          isAngularOutputForceCase(testCase)
              ? "angular_output_force_requires_30_60_120hz"
              : "position_requires_30_60_120hz");
    const PxU32 expectedFrames =
        PxU32(PxFloor(gPositionDuration / options.dt + 0.5f));
    if (options.frames != expectedFrames)
      return reportConfigurationError(
          options, config,
          isAngularOutputForceCase(testCase)
              ? "angular_output_force_requires_3_seconds"
              : "position_requires_3_seconds");
  } else if (PxAbs(options.dt - (1.0f / 60.0f)) > 1e-7f) {
    return reportConfigurationError(options, config,
                                    "dt_requires_60hz_calibration");
  }
  if (options.execution == Snippets::eHEADLESS_SEQUENTIAL &&
      options.solverType != PxSolverType::eAVBD)
    return reportConfigurationError(options, config,
                                    "sequential_requires_avbd");

  if (isLegacyAngularLimitCase(testCase)) {
    if (options.frames != 180)
      return reportConfigurationError(
          options, config,
          "legacy_angular_limit_requires_180_frames");
    if (seen.drive || seen.driveMode || seen.actorA || seen.frameA ||
        seen.frameB || seen.bodyB || seen.initialRelative || seen.anchor ||
        seen.mass || seen.limit || seen.outputForce || seen.breakMode ||
        seen.endpoint || seen.topology || seen.kinematicMotion)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  } else if (testCase == eCASE_VELOCITY_ORDERING) {
    if (!seen.drive || config.drive != eDRIVE_LINEAR_X)
      return reportConfigurationError(
          options, config, "velocity_ordering_requires_drive_x");
    if (!seen.endpoint)
      return reportConfigurationError(
          options, config, "velocity_ordering_requires_endpoint");
    if (options.frames != 180)
      return reportConfigurationError(
          options, config, "velocity_ordering_requires_180_frames");
    if (seen.driveMode || seen.actorA || seen.frameA || seen.frameB ||
        seen.bodyB || seen.initialRelative || seen.mass || seen.limit ||
        seen.outputForce)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  } else if (testCase == eCASE_ANGULAR_OUTPUT_FORCE) {
    if (!seen.drive ||
        (config.drive != eDRIVE_TWIST && config.drive != eDRIVE_SWING1 &&
         config.drive != eDRIVE_SWING2 && config.drive != eDRIVE_SLERP))
      return reportConfigurationError(
          options, config,
          "angular_output_force_requires_angular_drive");
    if (!seen.endpoint)
      return reportConfigurationError(
          options, config, "angular_output_force_requires_endpoint");
    if (!seen.outputForce)
      return reportConfigurationError(
          options, config, "angular_output_force_requires_selector");
    if (seen.driveMode || seen.actorA || seen.frameA || seen.frameB ||
        seen.bodyB || seen.initialRelative || seen.mass || seen.limit ||
        seen.topology)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  } else if (testCase == eCASE_ANGULAR_ORDERING) {
    if (!seen.drive || !isAngularDrive(config.drive))
      return reportConfigurationError(
          options, config, "angular_ordering_requires_angular_drive");
    if (!seen.endpoint)
      return reportConfigurationError(
          options, config, "angular_ordering_requires_endpoint");
    if (options.frames != 180)
      return reportConfigurationError(
          options, config, "angular_ordering_requires_180_frames");
    if (seen.driveMode || seen.actorA || seen.frameA || seen.frameB ||
        seen.bodyB || seen.initialRelative || seen.mass || seen.limit ||
        seen.outputForce)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  } else if (seen.endpoint &&
             !isPositionLikeCase(testCase) &&
             !(isComparisonCase(testCase) &&
               isDynamicComparisonTopology(config.topology))) {
    return reportConfigurationError(options, config,
                                    "endpoint_requires_velocity_ordering");
  }

  if (seen.topology && !isComparisonCase(testCase) &&
      !(testCase == eCASE_ANGULAR_POSITION &&
        isDynamicComparisonTopology(config.topology)))
    return reportConfigurationError(options, config,
                                    "topology_requires_comparison_case");
  if (seen.anchor && testCase != eCASE_OUTPUT_FORCE)
    return reportConfigurationError(options, config,
                                    "anchor_requires_output_force_case");
  if (seen.breakMode && testCase != eCASE_OUTPUT_FORCE &&
      testCase != eCASE_ANGULAR_OUTPUT_FORCE)
    return reportConfigurationError(options, config,
                                    "break_requires_output_force_case");
  const bool momentBreakMode =
      config.breakMode == eBREAK_BELOW_OFFSET_MOMENT ||
      config.breakMode == eBREAK_ABOVE_OFFSET_MOMENT;
  if (momentBreakMode &&
      (testCase != eCASE_OUTPUT_FORCE || !config.offsetAnchor))
    return reportConfigurationError(options, config,
                                    "moment_break_requires_offset_anchor");
  if (isComparisonCase(testCase) &&
      isDynamicComparisonTopology(config.topology) &&
      (!seen.topology || !seen.endpoint))
    return reportConfigurationError(
        options, config, "dynamic_topology_requires_endpoint");
  if (testCase == eCASE_VELOCITY) {
    if (seen.mass || seen.limit || seen.outputForce)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  }
  if (testCase == eCASE_VELOCITY || isPositionLikeCase(testCase) ||
      isAngularOutputForceCase(testCase)) {
    if (config.driveMode != eDRIVE_MODE_FORCE)
      return reportConfigurationError(options, config,
                                      "drive_mode_incompatible_with_case");
  }
  if (testCase == eCASE_VELOCITY && seen.initialRelative)
    return reportConfigurationError(options, config,
                                    "option_incompatible_with_case");
  if (isPositionLikeCase(testCase)) {
    if (testCase == eCASE_OUTPUT_FORCE) {
      config.comparisonMass = 1.0f;
      config.lowForceLimit = true;
    } else if (testCase == eCASE_ANGULAR_POSITION) {
      if (!seen.mass)
        config.comparisonMass = 1.0f;
      if (!seen.limit)
        config.lowForceLimit = true;
    } else {
      if (!seen.mass)
        config.comparisonMass = 1.0f;
      if (!seen.limit)
        config.lowForceLimit = false;
    }
    if (!seen.drive)
      return reportConfigurationError(options, config,
                                      "position_requires_explicit_drive");
    if (config.bodyBRotated)
      return reportConfigurationError(
          options, config, "position_requires_identity_body_b_rotation");
    if (testCase != eCASE_ANGULAR_POSITION &&
        config.frameAOrientation != config.frameBOrientation)
      return reportConfigurationError(
          options, config, "position_requires_aligned_joint_frames");
    if (seen.endpoint && config.drive != eDRIVE_LINEAR_X &&
        testCase != eCASE_ANGULAR_POSITION)
      return reportConfigurationError(
          options, config, "position_endpoint_requires_drive_x");
    if (config.endpoint == eENDPOINT_REVERSE && config.actorAKinematic)
      return reportConfigurationError(
          options, config, "position_reverse_requires_static_actor_a");
    if (testCase != eCASE_ANGULAR_POSITION &&
        (seen.mass || seen.limit ||
         PxAbs(options.dt - (1.0f / 60.0f)) > 1e-7f) &&
        config.drive != eDRIVE_LINEAR_X)
      return reportConfigurationError(
          options, config, "position_extended_gate_requires_drive_x");
    if (testCase == eCASE_ANGULAR_POSITION) {
      if (config.drive != eDRIVE_TWIST &&
          config.drive != eDRIVE_SWING1 &&
          config.drive != eDRIVE_SWING2 &&
          config.drive != eDRIVE_SLERP)
        return reportConfigurationError(
            options, config,
            "angular_position_requires_twist_swing1_swing2_or_slerp");
      if (!seen.endpoint)
        return reportConfigurationError(
            options, config, "angular_position_requires_endpoint");
      if (!seen.initialRelative)
        return reportConfigurationError(
            options, config, "angular_position_requires_initial_relative");
      if (!seen.frameA || !seen.frameB ||
          config.frameAOrientation == eFRAME_IDENTITY ||
          config.frameBOrientation != eFRAME_IDENTITY)
        return reportConfigurationError(
            options, config, "angular_position_requires_separated_frames");
      const bool dynamicPositionPair =
          isDynamicComparisonTopology(config.topology);
      if (dynamicPositionPair && (!seen.topology || !seen.mass || !seen.limit))
        return reportConfigurationError(
            options, config,
            "dynamic_angular_position_requires_topology_mass_limit");
      if (!dynamicPositionPair && (seen.mass || seen.limit || seen.topology))
        return reportConfigurationError(
            options, config,
            "angular_position_extended_options_require_dynamic_topology");
      const bool movingKinematicSlerp =
          config.drive == eDRIVE_SLERP && seen.actorA &&
          config.actorAKinematic && seen.kinematicMotion &&
          config.kinematicMotion == eKINEMATIC_SPIN_WORLD_Y &&
          config.endpoint == eENDPOINT_FORWARD &&
          !dynamicPositionPair;
      if ((seen.actorA && !movingKinematicSlerp) ||
          seen.bodyB || seen.outputForce ||
          seen.breakMode || seen.anchor ||
          (seen.kinematicMotion && !movingKinematicSlerp))
        return reportConfigurationError(options, config,
                                        "option_incompatible_with_case");
    } else if (testCase == eCASE_OUTPUT_FORCE) {
      if (!seen.endpoint)
        return reportConfigurationError(
            options, config, "output_force_requires_endpoint");
      if (!seen.outputForce)
        return reportConfigurationError(
            options, config, "output_force_requires_selector");
      if (config.drive != eDRIVE_LINEAR_X)
        return reportConfigurationError(
            options, config, "output_force_requires_drive_x");
      if (seen.mass || seen.limit || seen.actorA || seen.bodyB)
        return reportConfigurationError(options, config,
                                        "option_incompatible_with_case");
    } else if (seen.outputForce) {
      return reportConfigurationError(options, config,
                                      "output_force_requires_case");
    }
  }
  if (seen.kinematicMotion &&
      !(testCase == eCASE_ANGULAR_POSITION &&
        config.drive == eDRIVE_SLERP &&
        config.actorAKinematic &&
        config.kinematicMotion == eKINEMATIC_SPIN_WORLD_Y))
    return reportConfigurationError(
        options, config, "kinematic_motion_requires_angular_slerp_position");
  if (testCase == eCASE_MASS_SCALING ||
      testCase == eCASE_ACCELERATION_MODE) {
    if (testCase == eCASE_ACCELERATION_MODE && !seen.limit)
      config.lowForceLimit = false;
    if (!seen.drive || config.drive != eDRIVE_LINEAR_X)
      return reportConfigurationError(options, config,
                                      "comparison_requires_drive_x");
    if (!seen.driveMode)
      return reportConfigurationError(
          options, config, "comparison_requires_explicit_drive_mode");
    if (!seen.mass)
      return reportConfigurationError(options, config,
                                      "comparison_requires_mass");
    if (seen.actorA || seen.frameA || seen.frameB || seen.bodyB ||
        seen.initialRelative ||
        (seen.limit && testCase != eCASE_ACCELERATION_MODE) ||
        seen.outputForce)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
    if (testCase == eCASE_ACCELERATION_MODE && seen.limit &&
        (!config.lowForceLimit || config.comparisonMass != 1.0f))
      return reportConfigurationError(
          options, config, "acceleration_limit_requires_low_mass1");
    const JointDriveMode requiredMode =
        testCase == eCASE_ACCELERATION_MODE ? eDRIVE_MODE_ACCELERATION
                                            : eDRIVE_MODE_FORCE;
    if (config.driveMode != requiredMode)
      return reportConfigurationError(options, config,
                                      "drive_mode_incompatible_with_case");
  }
  if (testCase == eCASE_FORCE_LIMIT) {
    if (!seen.drive || config.drive != eDRIVE_LINEAR_X)
      return reportConfigurationError(options, config,
                                      "force_limit_requires_drive_x");
    if (!seen.limit)
      return reportConfigurationError(options, config,
                                      "force_limit_requires_limit");
    if (config.driveMode != eDRIVE_MODE_FORCE)
      return reportConfigurationError(options, config,
                                      "force_limit_requires_force_mode");
    if (seen.actorA || seen.frameA || seen.frameB || seen.bodyB ||
        seen.initialRelative || seen.mass || seen.outputForce)
      return reportConfigurationError(options, config,
                                      "option_incompatible_with_case");
  }
  if (!Snippets::applyExecutionEnvironment(options))
    return reportConfigurationError(options, config,
                                    "execution_environment_failed");

  gHeadlessOptions = options;
  gHeadlessConfig = config;
  gHeadlessCase = testCase;
  gSolverType = options.solverType;
  gHeadlessMode = options.headless;

#ifdef RENDER_SNIPPET
  if (!options.headless) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  Snippets::printHeadlessConfig("SnippetJointDrive", gHeadlessOptions);
  const PxReal configuredTargetMagnitude =
      isPositionLikeCase(gHeadlessCase)
          ? gPositionTargetMagnitude
          : (gHeadlessCase == eCASE_FORCE_LIMIT
                 ? gForceLimitProbeTargetVelocity
                 : gTargetVelocity);
  if (isLegacyAngularLimitCase(gHeadlessCase)) {
    printf(
        "[SnippetJointDriveAngularLimitConfig] case=%s "
        "limitKind=legacy-cone limitY=%.9g limitZ=%.9g "
        "initialSwingY=%.9g initialSwingZ=%.9g "
        "twist=locked swing1=limited swing2=limited gravity=zero "
        "ground=none damping=zero frames=%u\n",
        getCaseName(gHeadlessCase), double(gLegacyConeLimitY),
        double(gLegacyConeLimitZ),
        double(gHeadlessCase ==
                       eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE
                   ? gLegacyConeInsideY
                   : gLegacyConeOutsideY),
        double(gHeadlessCase ==
                       eCASE_LEGACY_ANGULAR_LIMIT_CONE_INSIDE
                   ? gLegacyConeInsideZ
                   : gLegacyConeOutsideZ),
        gHeadlessOptions.frames);
  } else if (isOrderingCase(gHeadlessCase)) {
    printf(
        "[SnippetJointDriveOrderingConfig] case=%s drive=%s quantity=%s "
        "endpoint=%s actor0=%s actor1=%s driveMode=force "
        "targetMagnitude=%.9g targetConvention=%s relativeTargetSign=%.9g "
        "bodyRotation=rotz-neg45 frameASource=actor-a "
        "frameBSource=actor-b frameSourceSeparated=%u "
        "gravity=zero ground=none pvd=off shapeCount=0 frames=%u "
        "outputForce=%s breakMode=%s torqueLimit=%.9g\n",
        getCaseName(gHeadlessCase), getDriveName(gHeadlessConfig.drive),
        isAngularOrderingCase(gHeadlessCase) ? "angular" : "linear",
        getEndpointName(gHeadlessConfig.endpoint), getOrderingActor0Name(),
        getOrderingActor1Name(), double(configuredTargetMagnitude),
        isAngularOrderingCase(gHeadlessCase) &&
                gHeadlessConfig.drive != eDRIVE_SLERP
            ? "relative-a-minus-b"
            : "relative-b-minus-a",
        double(isAngularOrderingCase(gHeadlessCase)
                   ? getRelativeTargetSign(gHeadlessConfig.drive)
                   : 1.0f),
        isAngularOrderingCase(gHeadlessCase) ? 1u : 0u,
        gHeadlessOptions.frames,
        gHeadlessConfig.outputForceEnabled ? "on" : "off",
        getBreakModeName(gHeadlessConfig.breakMode),
        double(isAngularOutputForceCase(gHeadlessCase)
                   ? gAngularOutputTorqueLimit
                   : PX_MAX_F32));
  } else {
    printf(
      "[SnippetJointDriveConfig] case=%s drive=%s actorA=%s frameA=%s "
      "frameB=%s bodyBRotation=%s initialRelative=%s anchor=%s driveMode=%s "
      "targetMagnitude=%.9g "
      "comparisonMass=%.9g limit=%s outputForce=%s breakMode=%s "
      "topology=%s endpoint=%s kinematicMotion=%s "
      "contactDriveGate=%s gravity=%s ground=%s friction=%s "
      "positionDriveGate=%s "
      "accelerationDriveGate=%s massScalingGate=%s forceLimitGate=%s "
      "outputForceGate=%s\n",
      getCaseName(gHeadlessCase),
      getDriveName(gHeadlessConfig.drive),
      gHeadlessConfig.actorAKinematic ? "kinematic" : "static",
      getOrientationName(gHeadlessConfig.frameAOrientation),
      getOrientationName(gHeadlessConfig.frameBOrientation),
      getOrientationName(gHeadlessConfig.bodyBRotated),
      getInitialRelativeName(gHeadlessConfig.initialRelativeOffset),
      getAnchorName(gHeadlessConfig.offsetAnchor),
      getDriveModeName(gHeadlessConfig.driveMode),
      double(configuredTargetMagnitude),
      double(gHeadlessConfig.comparisonMass),
      gHeadlessConfig.lowForceLimit ? "low" : "high",
      gHeadlessConfig.outputForceEnabled ? "on" : "off",
      getBreakModeName(gHeadlessConfig.breakMode),
      getTopologyName(gHeadlessConfig.topology),
      getEndpointName(gHeadlessConfig.endpoint),
      getKinematicMotionName(gHeadlessConfig.kinematicMotion),
      isContactComparisonTopology(gHeadlessConfig.topology) ? "ACTIVE"
                                                            : "NOT_COVERED",
      isContactComparisonTopology(gHeadlessConfig.topology) ? "earth"
                                                            : "zero",
      isContactComparisonTopology(gHeadlessConfig.topology) ? "plane"
                                                            : "none",
      isContactComparisonTopology(gHeadlessConfig.topology) ? "zero"
                                                            : "not-applicable",
      isPositionLikeCase(gHeadlessCase) ? "ACTIVE" : "NOT_COVERED",
      gHeadlessCase == eCASE_ACCELERATION_MODE ? "ACTIVE" : "NOT_COVERED",
      gHeadlessCase == eCASE_MASS_SCALING ? "ACTIVE" : "NOT_COVERED",
      gHeadlessCase == eCASE_FORCE_LIMIT ? "ACTIVE" : "NOT_COVERED",
      (gHeadlessCase == eCASE_OUTPUT_FORCE ||
       isAngularOutputForceCase(gHeadlessCase))
          ? "ACTIVE"
          : "NOT_COVERED");
  }

  initPhysics(false);
  for (PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame) {
    PX_UNUSED(frame);
    const bool fetchFailed = isOrderingCase(gHeadlessCase)
                                 ? gOrderingMetrics.fetchFailures != 0
                                 : gMetrics.fetchFailures != 0;
    if (gInitializationFailed || fetchFailed)
      break;
    stepPhysics(false);
  }

  JointDriveGateEvaluation evaluation =
      isOrderingCase(gHeadlessCase) ? evaluateOrderingGate() : evaluateGate();
  printGateDetails(evaluation);
  cleanupPhysics(false);
  const PxU32 physicsErrors = gErrorCallback.getFatalCount();
  const PxU32 physicsWarnings = gErrorCallback.getWarningCount();
  if (isOrderingCase(gHeadlessCase) && !gOrderingMetrics.cleanupComplete)
    setGateErrorOverFailure(evaluation, "ordering_cleanup");
  if (physicsErrors && isOrderingCase(gHeadlessCase))
    setGateErrorOverFailure(evaluation, "ordering_physx_error");
  else if (physicsErrors)
    setGateFailure(evaluation, "physx_error");
  printGateResult(evaluation, physicsErrors, physicsWarnings);
  return static_cast<int>(evaluation.exitCode);
}
