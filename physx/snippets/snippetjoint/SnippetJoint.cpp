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
// This snippet illustrates simple use of joints in physx
//
// It creates a chain of objects joined by limited spherical joints, a chain
// joined by fixed joints which is breakable, and a chain of damped D6 joints
// ****************************************************************************

#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetutils/SnippetUtils.h"
#include "PxPhysicsAPI.h"
#include <algorithm>
#include <ctype.h>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
static bool gExtensionsInitialized = false;
static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;
static Snippets::HeadlessOptions gHeadlessOptions;
static bool gInitializationFailed = false;
static PxReal gFixedLinearBreakForce = 1000.0f;
static PxReal gFixedAngularBreakForce = 100000.0f;

enum JointKind {
  eJOINT_SPHERICAL,
  eJOINT_FIXED,
  eJOINT_D6,
  eJOINT_PRISMATIC,
  eJOINT_REVOLUTE
};

enum JointHeadlessCase {
  eCASE_PASSIVE,
  eCASE_IMPACT_ALL,
  eCASE_WIDE_JOINT_STRESS,
  eCASE_IMPACT_SINGLE,
  eCASE_FIXED_NO_BREAK,
  eCASE_FIXED_BREAK,
  eCASE_FORCE_STATIC,
  eCASE_FORCE_OFFSET,
  eCASE_FORCE_PAIR,
  eCASE_FORCE_PAIR_DISABLED,
  eCASE_SPHERICAL_CONE_INSIDE,
  eCASE_SPHERICAL_CONE_OUTSIDE,
  eCASE_NATIVE_REACTION,
  eCASE_NATIVE_NO_BREAK,
  eCASE_NATIVE_BREAK,
  eCASE_REVOLUTE_MOTOR,
  eCASE_REVOLUTE_MOTOR_LIMIT,
  eCASE_REVOLUTE_MOTOR_DYNAMIC_LIMIT,
  eCASE_REVOLUTE_MOTOR_FREESPIN,
  eCASE_REVOLUTE_MOTOR_DYNAMIC_FREESPIN,
  eCASE_REVOLUTE_MOTOR_RATIO,
  eCASE_REVOLUTE_MOTOR_CONTACT,
  eCASE_REVOLUTE_MOTOR_KINEMATIC,
  eCASE_REVOLUTE_MOTOR_OFF_PRINCIPAL,
  eCASE_REVOLUTE_MOTOR_OFF_CENTER,
  eCASE_REVOLUTE_MOTOR_SPATIAL,
  eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_PRINCIPAL,
  eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_CENTER,
  eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL
};

enum EndpointKind {
  eENDPOINT_IMPLICIT,
  eENDPOINT_FORWARD,
  eENDPOINT_REVERSE
};

enum ForceActorOrder {
  eFORCE_ACTOR_ORDER_NORMAL,
  eFORCE_ACTOR_ORDER_SWAPPED
};

static ForceActorOrder gForceActorOrder = eFORCE_ACTOR_ORDER_NORMAL;

enum SphericalConeTopology {
  eSPHERICAL_CONE_STATIC_DYNAMIC,
  eSPHERICAL_CONE_DYNAMIC_DYNAMIC
};

static SphericalConeTopology gSphericalConeTopology =
    eSPHERICAL_CONE_STATIC_DYNAMIC;

enum TargetResponseKind {
  eTARGET_RESPONSE_NONE,
  eTARGET_RESPONSE_LINEAR_Z
};

struct ChainRecord {
  JointKind kind;
  const char *name;
  std::vector<PxRigidDynamic *> bodies;
  std::vector<PxJoint *> joints;
  PxRigidDynamic *targetBody;
  PxReal maxAnchorError;
  PxReal anchorErrorSquaredSum;
  PxU32 anchorSamples;
  PxReal maxAngularLimitViolation;
  PxReal maxLinearLimitViolation;
  PxReal maxConstrainedAngularError;
  PxReal peakKineticEnergy;
  PxU32 lateAwakeSamples;
  PxU32 lateBodySamples;
  PxReal lateAngularSpeedSum;
  PxReal lateAngularSpeedMax;
  PxU32 lateAngularSpeedSamples;
  std::vector<PxReal> kineticEnergy;

  ChainRecord(JointKind jointKind = eJOINT_SPHERICAL,
              const char *jointName = "spherical")
      : kind(jointKind), name(jointName), targetBody(NULL),
        maxAnchorError(0.0f), anchorErrorSquaredSum(0.0f), anchorSamples(0),
        maxAngularLimitViolation(0.0f), maxLinearLimitViolation(0.0f),
        maxConstrainedAngularError(0.0f), peakKineticEnergy(0.0f),
        lateAwakeSamples(0), lateBodySamples(0),
        lateAngularSpeedSum(0.0f), lateAngularSpeedMax(0.0f),
        lateAngularSpeedSamples(0) {}
};

struct ProjectileRecord {
  PxRigidDynamic *actor;
  PxU32 targetChain;
  PxU32 contactCount;
  PxU32 contactPointCount;
  PxU32 firstChainContact;
  PxU32 firstChainBody;
  PxU32 crossChainContacts;
  PxU32 firstContactFrame;
  PxU32 firstWrongContactFrame;
  PxU32 wrongBodyContacts;
  PxReal expectedMomentum;
  PxReal totalContactImpulse;
  PxReal peakContactImpulse;
  PxVec3 launchVelocity;
  PxReal maxVelocityDelta;
  PxReal maxImpactAxisVelocityDelta;
  PxReal targetMass;
  PxReal preContactTargetVelocityZ;
  PxReal preContactChainMomentumZ;
  PxReal targetDeltaVelocityZ;
  PxReal chainDeltaMomentumZ;
  PxReal maxOppositeTargetDeltaVelocityZ;
  PxReal maxOppositeChainDeltaMomentumZ;
  PxU32 targetResponseSamples;
  bool targetResponseBaselineValid;

  ProjectileRecord(PxRigidDynamic *projectile = NULL,
                   PxU32 chainIndex = PX_MAX_U32)
      : actor(projectile), targetChain(chainIndex), contactCount(0),
        contactPointCount(0), firstChainContact(PX_MAX_U32),
        firstChainBody(PX_MAX_U32), crossChainContacts(0),
        firstContactFrame(PX_MAX_U32), firstWrongContactFrame(PX_MAX_U32),
        wrongBodyContacts(0), expectedMomentum(0.0f),
        totalContactImpulse(0.0f), peakContactImpulse(0.0f),
        launchVelocity(0.0f), maxVelocityDelta(0.0f),
        maxImpactAxisVelocityDelta(0.0f), targetMass(0.0f),
        preContactTargetVelocityZ(0.0f), preContactChainMomentumZ(0.0f),
        targetDeltaVelocityZ(0.0f), chainDeltaMomentumZ(0.0f),
        maxOppositeTargetDeltaVelocityZ(0.0f),
        maxOppositeChainDeltaMomentumZ(0.0f), targetResponseSamples(0),
        targetResponseBaselineValid(false) {}
};

struct JointGateStats {
  PxU32 completedFrames;
  PxU32 fetchFailures;
  PxU32 fetchErrorState;
  PxU32 launchFailures;
  PxU32 nonFinite;
  PxU32 breakCallbackCount;
  PxU32 breakCallbackIdentityMatches;
  PxU32 breakCallbackConstraintMismatches;
  PxU32 breakCallbackExternalReferenceMismatches;
  PxU32 breakCallbackTypeMismatches;
  PxU32 breakCallbackBrokenFlagMismatches;
  PxU32 breakCallbackDuplicateMismatches;
  PxU32 breakCallbackPollMismatches;
  PxReal maxQuaternionNormError;
  PxReal maxAbsPosition;
  PxReal maxLinearSpeed;
  PxReal maxAngularSpeed;
  PxReal maxAnchorError;
  PxReal anchorErrorSquaredSum;
  PxU32 anchorSamples;
  PxReal maxRevoluteLimitViolation;
  PxReal maxPrismaticLimitViolation;
  PxReal maxSphericalLimitViolation;
  PxReal maxD6LockedLinearError;
  PxReal maxConstrainedAngularError;
  PxReal peakKineticEnergy;
  std::vector<PxReal> kineticEnergy;

  JointGateStats()
      : completedFrames(0), fetchFailures(0), fetchErrorState(0),
        launchFailures(0), nonFinite(0),
        breakCallbackCount(0), breakCallbackIdentityMatches(0),
        breakCallbackConstraintMismatches(0),
        breakCallbackExternalReferenceMismatches(0),
        breakCallbackTypeMismatches(0),
        breakCallbackBrokenFlagMismatches(0),
        breakCallbackDuplicateMismatches(0),
        breakCallbackPollMismatches(0), maxQuaternionNormError(0.0f),
        maxAbsPosition(0.0f), maxLinearSpeed(0.0f),
        maxAngularSpeed(0.0f), maxAnchorError(0.0f),
        anchorErrorSquaredSum(0.0f), anchorSamples(0),
        maxRevoluteLimitViolation(0.0f),
        maxPrismaticLimitViolation(0.0f),
        maxSphericalLimitViolation(0.0f), maxD6LockedLinearError(0.0f),
        maxConstrainedAngularError(0.0f), peakKineticEnergy(0.0f) {}
};

static JointHeadlessCase gHeadlessCase = eCASE_PASSIVE;
static JointKind gImpactJointKind = eJOINT_SPHERICAL;
static EndpointKind gEndpointKind = eENDPOINT_IMPLICIT;
static bool gEndpointOptionMentioned = false;
static std::vector<ChainRecord> gChains;
static std::vector<ProjectileRecord> gProjectiles;
static JointGateStats gGateStats;

static const PxU32 gImpactLaunchFrame = 120;
static const PxU32 gProjectileObservationFrames = 120;
static const PxReal gImpactRadius = 0.75f;
static const PxReal gImpactHeight = 6.0f;
static const PxReal gImpactSpeed = 20.0f;
static const PxReal gImpactTransverseOffset = 0.25f;
static const PxReal gAnchorErrorCap = 0.25f;
static const PxReal gAngularLimitViolationCap = 0.15f;
static const PxReal gLinearLimitViolationCap = 0.1f;
static const PxReal gConstrainedAngularErrorCap = 0.15f;
static const PxReal gImpactLinearSpeedCap = gImpactSpeed * 10.0f;
static const PxReal gImpactAngularSpeedCap = gImpactSpeed * 10.0f;
static const PxReal gMinProjectileResponseFraction = 0.05f;
static const PxReal gMinTargetResponseFraction =
    gMinProjectileResponseFraction;
static const PxU32 gProjectileResponseWindowFrames = 3;

struct EndpointProbeStats {
  PxU32 stateSampleAttempts;
  PxU32 stateSamples;
  PxU32 responseSampleAttempts;
  PxU32 responseSamples;
  PxU32 responseBaselineSamples;
  PxU32 nonFiniteStateSamples;
  PxU32 nonFiniteResponseSamples;
  PxU32 actualLaunchFrame;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxU32 shapeCount;
  PxVec3 expectedAxis;
  PxVec3 actor0Axis;
  PxVec3 actor1Axis;
  PxVec3 dynamicLocalAxis;
  PxVec3 launchDirection;
  PxVec3 gravity;
  PxVec3 initialTargetPosition;
  PxVec3 initialTargetVelocity;
  PxVec3 responseBaselinePosition;
  PxVec3 responseBaselineVelocity;
  PxVec3 responsePositionDeltaSum;
  PxVec3 responseVelocityDeltaSum;
  double positionOrthogonalSquaredSum;
  double velocityOrthogonalSquaredSum;
  PxReal bodyRotationDot;
  PxReal expectedAxisDot;
  PxReal dynamicLocalAxisDot;
  PxReal maxPositionOrthogonalDelta;
  PxReal maxVelocityOrthogonalDelta;
  PxReal maxPrecontactPositionDrift;
  PxReal maxPrecontactSpeed;
  PxReal maxTransverseAnchor;
  PxReal worldFramePositionError;
  PxReal dynamicLocalPositionError;
  PxReal worldFrameRotationDot;
  PxReal dynamicLocalRotationDot;
  PxReal shapeLocalPositionError;
  PxReal shapeLocalRotationDot;
  PxReal shapeRadius;
  bool limitEnabled;
  bool actorOrderValid;
  bool frameWitnessValid;
  bool fixtureWitnessValid;
  bool cleanupComplete;

  EndpointProbeStats()
      : stateSampleAttempts(0), stateSamples(0), responseSampleAttempts(0),
        responseSamples(0), responseBaselineSamples(0),
        nonFiniteStateSamples(0), nonFiniteResponseSamples(0),
        actualLaunchFrame(PX_MAX_U32), initialDynamicActors(PX_MAX_U32),
        initialStaticActors(PX_MAX_U32), initialConstraints(PX_MAX_U32),
        finalDynamicActors(PX_MAX_U32), finalStaticActors(PX_MAX_U32),
        finalConstraints(PX_MAX_U32), cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32), cleanupConstraints(PX_MAX_U32),
        shapeCount(0),
        expectedAxis(0.0f, 0.0f, -1.0f), actor0Axis(0.0f),
        actor1Axis(0.0f), dynamicLocalAxis(0.0f), launchDirection(0.0f),
        gravity(0.0f),
        initialTargetPosition(0.0f), initialTargetVelocity(0.0f),
        responseBaselinePosition(0.0f), responseBaselineVelocity(0.0f),
        responsePositionDeltaSum(0.0f), responseVelocityDeltaSum(0.0f),
        positionOrthogonalSquaredSum(0.0),
        velocityOrthogonalSquaredSum(0.0),
        bodyRotationDot(0.0f), expectedAxisDot(0.0f),
        dynamicLocalAxisDot(0.0f),
        maxPositionOrthogonalDelta(0.0f),
        maxVelocityOrthogonalDelta(0.0f),
        maxPrecontactPositionDrift(0.0f), maxPrecontactSpeed(0.0f),
        maxTransverseAnchor(0.0f), worldFramePositionError(PX_MAX_F32),
        dynamicLocalPositionError(PX_MAX_F32), worldFrameRotationDot(0.0f),
        dynamicLocalRotationDot(0.0f),
        shapeLocalPositionError(PX_MAX_F32), shapeLocalRotationDot(0.0f),
        shapeRadius(0.0f), limitEnabled(false), actorOrderValid(false),
        frameWitnessValid(false), fixtureWitnessValid(false),
        cleanupComplete(false) {}
};

struct EndpointAngularProbeStats {
  PxU32 launchAttempts;
  PxU32 launchSuccesses;
  PxU32 tailSamples;
  PxU32 apiSamples;
  PxU32 nonFiniteApiSamples;
  PxVec3 perpendicularAxis;
  PxVec3 requestedAngularVelocity;
  PxVec3 actualLaunchAngularVelocity;
  PxQuat initialTargetOrientation;
  PxQuat responseBaselineOrientation;
  PxVec3 initialTargetAngularVelocity;
  PxVec3 responseBaselineAngularVelocity;
  PxVec3 responseRotationVectorSum;
  PxVec3 responseAngularVelocityDeltaSum;
  double rotationOrthogonalSquaredSum;
  double angularVelocityOrthogonalSquaredSum;
  double rawJointAngleDeltaSum;
  double semanticJointAngleDeltaSum;
  double apiVelocityMagnitudeSum;
  PxReal bodyWorldAxisDot;
  PxReal launchVelocityError;
  PxReal minSignedRotation;
  PxReal minSignedAngularVelocity;
  PxReal maxRotationOrthogonalDelta;
  PxReal maxAngularVelocityOrthogonalDelta;
  PxReal maxPrelaunchOrientationDrift;
  PxReal maxPrelaunchAngularSpeed;
  PxReal maxAnchorError;
  PxReal maxAxisMisalignment;
  PxReal baselineRawJointAngle;
  PxReal lastRawJointAngleDelta;
  PxReal lastSemanticJointAngleDelta;
  PxReal maxJointAnglePoseMismatch;
  PxReal maxApiVelocityMagnitudeMismatch;
  bool driveEnabled;
  bool nonIdentityWitnessValid;
  bool launchWakeValid;

  EndpointAngularProbeStats()
      : launchAttempts(0), launchSuccesses(0), tailSamples(0), apiSamples(0),
        nonFiniteApiSamples(0), perpendicularAxis(0.0f),
        requestedAngularVelocity(0.0f), actualLaunchAngularVelocity(0.0f),
        initialTargetOrientation(PxIdentity),
        responseBaselineOrientation(PxIdentity),
        initialTargetAngularVelocity(0.0f),
        responseBaselineAngularVelocity(0.0f),
        responseRotationVectorSum(0.0f),
        responseAngularVelocityDeltaSum(0.0f),
        rotationOrthogonalSquaredSum(0.0),
        angularVelocityOrthogonalSquaredSum(0.0),
        rawJointAngleDeltaSum(0.0), semanticJointAngleDeltaSum(0.0),
        apiVelocityMagnitudeSum(0.0), bodyWorldAxisDot(0.0f),
        launchVelocityError(PX_MAX_F32),
        minSignedRotation(PX_MAX_F32),
        minSignedAngularVelocity(PX_MAX_F32),
        maxRotationOrthogonalDelta(0.0f),
        maxAngularVelocityOrthogonalDelta(0.0f),
        maxPrelaunchOrientationDrift(0.0f),
        maxPrelaunchAngularSpeed(0.0f), maxAnchorError(0.0f),
        maxAxisMisalignment(0.0f), baselineRawJointAngle(0.0f),
        lastRawJointAngleDelta(0.0f),
        lastSemanticJointAngleDelta(0.0f),
        maxJointAnglePoseMismatch(0.0f),
        maxApiVelocityMagnitudeMismatch(0.0f), driveEnabled(false),
        nonIdentityWitnessValid(false), launchWakeValid(false) {}
};

static EndpointProbeStats gEndpointStats;
static EndpointAngularProbeStats gEndpointAngularStats;
static PxRigidDynamic *gEndpointTarget = NULL;
static PxPrismaticJoint *gEndpointJoint = NULL;
static PxRevoluteJoint *gEndpointRevoluteJoint = NULL;
static const PxU32 gEndpointResponseWindowFrames = 3;
static const PxReal gEndpointPositionResponseMinimum = 0.05f;
static const PxReal gEndpointVelocityResponseMinimum =
    gImpactSpeed * gMinProjectileResponseFraction;
static const PxReal gEndpointDirectionDotMinimum = 0.99f;
static const PxReal gEndpointOrthogonalRatioMaximum = 0.05f;
static const PxReal gEndpointPositionOrthogonalAbsoluteEpsilon = 1e-4f;
static const PxReal gEndpointVelocityOrthogonalAbsoluteEpsilon = 1e-4f;
static const PxReal gEndpointPrecontactPositionDriftMaximum = 1e-4f;
static const PxReal gEndpointPrecontactSpeedMaximum = 1e-4f;
static const PxReal gEndpointTransverseAnchorMaximum = 0.05f;
static const PxU32 gEndpointAngularResponseWindowFrames = 6;
static const PxU32 gEndpointAngularSettleFrames = 2;
static const PxReal gEndpointAngularAxialLaunchSpeed = 8.0f;
static const PxReal gEndpointAngularTransverseLaunchSpeed = 2.0f;
static const PxReal gEndpointAngularPoseResponseMinimum = 0.05f;
static const PxReal gEndpointAngularVelocityResponseMinimum = 1.0f;
static const PxReal gEndpointAngularDirectionDotMinimum = 0.99f;
static const PxReal gEndpointAngularOrthogonalRatioMaximum = 0.05f;
static const PxReal gEndpointAngularOrthogonalAbsoluteEpsilon = 1e-4f;
static const PxReal gEndpointAngularPrelaunchOrientationDriftMaximum = 1e-4f;
static const PxReal gEndpointAngularPrelaunchSpeedMaximum = 1e-4f;
static const PxReal gEndpointAngularAnchorMaximum = 0.05f;
static const PxReal gEndpointAngularAxisMisalignmentMaximum = 0.05f;
static const PxReal gEndpointAngularJointAngleMismatchMaximum = 0.05f;

static std::vector<PxRigidDynamic *> gRevoluteChainBodies;
static std::vector<PxRevoluteJoint *> gRevoluteChainJoints;
static std::vector<PxVec3> gRevoluteRestPositions;
static std::vector<PxFixedJoint *> gFixedChainJoints;
static std::vector<PxJoint *> gBreakableJoints;
static std::vector<PxU8> gBreakPollReported;
static std::vector<PxU8> gBreakCallbackReported;
static std::vector<PxRigidDynamic *> gPrismaticChainBodies;
static std::vector<PxPrismaticJoint *> gPrismaticChainJoints;
static std::vector<PxVec3> gPrismaticRestPositions;

static void getJointWorldAxes(PxRevoluteJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1);
static void getJointWorldAxes(PxPrismaticJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1);

struct RevoluteJitterStats {
  PxU32 frame = 0;
  PxReal maxTailLateral = 0.0f;
  PxReal sumW4Early = 0.0f, sumW5Early = 0.0f;
  PxReal sumW4Late = 0.0f, sumW5Late = 0.0f;
  PxReal sumW4PerpEarly = 0.0f, sumW5PerpEarly = 0.0f;
  PxReal sumW4PerpLate = 0.0f, sumW5PerpLate = 0.0f;
  PxReal maxW4PerpLate = 0.0f, maxW5PerpLate = 0.0f;
  PxU32 awake4Late = 0, awake5Late = 0;
  PxU32 cntEarly = 0, cntLate = 0;
  PxReal prevAngle3 = 0.0f, prevAngle4 = 0.0f;
  PxReal prevD3 = 0.0f, prevD4 = 0.0f;
  PxU32 flip3 = 0, flip4 = 0;
  PxReal maxAbsAngle3 = 0.0f, maxAbsAngle4 = 0.0f;
  PxReal maxAxisMisalign3Deg = 0.0f, maxAxisMisalign4Deg = 0.0f;
};

static RevoluteJitterStats gRevoluteStats;

struct FixedChainStats {
  PxReal maxLinearForce = 0.0f;
  PxReal maxAngularForce = 0.0f;
  PxU32 firstBrokenFrame = PX_MAX_U32;
  PxU32 brokenCount = 0;
};

static FixedChainStats gFixedStats;

struct ForceStaticStats {
  PxU32 forceReads;
  PxU32 stateSamples;
  PxU32 steadyBeginFrame;
  PxU32 expectedSteadySamples;
  PxU32 steadySampleAttempts;
  PxU32 steadySamples;
  PxU32 nonFiniteForceSamples;
  PxU32 nonFiniteSteadyForceSamples;
  PxU32 nonFiniteStateSamples;
  PxVec3 linearForceSum;
  PxVec3 angularForceSum;
  PxReal linearMagnitudeSum;
  PxReal angularMagnitudeSum;
  double orthogonalMagnitudeSquaredSum;
  double angularMagnitudeSquaredSum;
  PxReal maxAngularMagnitude;
  PxReal maxPositionError;
  PxReal maxRotationError;
  PxReal maxLinearSpeed;
  PxReal maxAngularSpeed;
  PxU32 maxPositionErrorFrame;
  PxU32 maxLinearSpeedFrame;
  PxReal steadyMaxLinearSpeed;
  PxU32 steadyMaxLinearSpeedFrame;
  PxReal finalPositionError;
  PxReal finalLinearSpeed;
  PxVec3 anchorOffset;
  PxVec3 expectedLinearForce;
  PxVec3 expectedTorque;
  PxVec3 appliedForceActor0;
  PxVec3 appliedForceActor1;
  PxReal actor0FramePositionError;
  PxReal actor1FramePositionError;
  double angularOrthogonalMagnitudeSquaredSum;
  PxU32 topologyDynamicActors;
  PxU32 topologyStaticActors;
  PxU32 topologyConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  bool cleanupComplete;
  PxReal actualMass;
  PxVec3 gravity;
  PxReal expectedWeight;
  PxReal pairInitialSeparation;
  PxVec3 pairInitialCenterOfMass;
  bool pairActorOrderValid;
  PxReal pairMaxSeparationError;
  PxReal pairFinalSeparation;
  PxReal pairMaxRelativeSpeed;
  PxReal pairFinalRelativeSpeed;
  PxReal pairMaxCenterOfMassError;
  PxReal pairFinalCenterOfMassError;
  PxReal pairMaxTotalMomentum;
  PxReal pairFinalTotalMomentum;
  PxVec3 pairFinalTotalMomentumVector;
  PxVec3 pairActor0FinalPosition;
  PxVec3 pairActor1FinalPosition;
  PxVec3 pairActor0FinalVelocity;
  PxVec3 pairActor1FinalVelocity;

  ForceStaticStats()
      : forceReads(0), stateSamples(0), steadyBeginFrame(0),
        expectedSteadySamples(0), steadySampleAttempts(0), steadySamples(0),
        nonFiniteForceSamples(0), nonFiniteSteadyForceSamples(0),
        nonFiniteStateSamples(0),
        linearForceSum(0.0f), angularForceSum(0.0f),
        linearMagnitudeSum(0.0f), angularMagnitudeSum(0.0f),
        orthogonalMagnitudeSquaredSum(0.0),
        angularMagnitudeSquaredSum(0.0),
        maxAngularMagnitude(0.0f), maxPositionError(0.0f),
        maxRotationError(0.0f), maxLinearSpeed(0.0f),
        maxAngularSpeed(0.0f), maxPositionErrorFrame(0),
        maxLinearSpeedFrame(0), steadyMaxLinearSpeed(0.0f),
        steadyMaxLinearSpeedFrame(0), finalPositionError(0.0f),
        finalLinearSpeed(0.0f), anchorOffset(0.0f),
        expectedLinearForce(0.0f), expectedTorque(0.0f),
        appliedForceActor0(0.0f), appliedForceActor1(0.0f),
        actor0FramePositionError(PX_MAX_F32),
        actor1FramePositionError(PX_MAX_F32),
        angularOrthogonalMagnitudeSquaredSum(0.0), topologyDynamicActors(0),
        topologyStaticActors(0), topologyConstraints(0),
        finalDynamicActors(PX_MAX_U32), finalStaticActors(PX_MAX_U32),
        finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32), cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), cleanupComplete(false),
        actualMass(0.0f), gravity(0.0f), expectedWeight(0.0f),
        pairInitialSeparation(0.0f), pairInitialCenterOfMass(0.0f),
        pairActorOrderValid(false), pairMaxSeparationError(0.0f),
        pairFinalSeparation(0.0f), pairMaxRelativeSpeed(0.0f),
        pairFinalRelativeSpeed(0.0f), pairMaxCenterOfMassError(0.0f),
        pairFinalCenterOfMassError(0.0f), pairMaxTotalMomentum(0.0f),
        pairFinalTotalMomentum(0.0f), pairFinalTotalMomentumVector(0.0f),
        pairActor0FinalPosition(0.0f), pairActor1FinalPosition(0.0f),
        pairActor0FinalVelocity(0.0f), pairActor1FinalVelocity(0.0f) {}
};

static ForceStaticStats gForceStaticStats;
static PxRigidDynamic *gForceStaticBody = NULL;
static PxRigidDynamic *gForcePairBody1 = NULL;
static PxFixedJoint *gForceStaticJoint = NULL;
static PxTransform gForceStaticInitialPose(PxIdentity);
static PxTransform gForcePairBody1InitialPose(PxIdentity);
static const PxReal gForceStaticMass = 4.0f;
static const PxReal gGravityMagnitude = 9.81f;
static const PxReal gForcePairAppliedMagnitude = 39.24f;
static const PxReal gForcePairSeparation = 2.0f;
static const PxVec3 gForceOffsetAnchor(1.0f, 0.0f, 0.0f);
static const PxU32 gForceStaticWarmupSeconds = 2;
static const PxU32 gForceStaticDurationSeconds = 10;
static const PxReal gForceRatioMinimum = 0.9f;
static const PxReal gForceRatioMaximum = 1.1f;
static const PxReal gForceDirectionDotMinimum = 0.99f;
static const PxReal gForceOrthogonalRatioMaximum = 0.01f;
static const PxReal gForceTorqueMaximum = 0.01f;
static const PxReal gForceTorqueRatioMinimum = 0.9f;
static const PxReal gForceTorqueRatioMaximum = 1.1f;
static const PxReal gForceTorqueDirectionDotMinimum = 0.99f;
static const PxReal gForceTorqueOrthogonalRatioMaximum = 0.01f;
static const PxReal gForcePositionErrorMaximum = 1e-3f;
static const PxReal gForceRotationErrorMaximum = 1e-3f;
static const PxReal gForceLinearSpeedMaximum = 1e-3f;
static const PxReal gForceAngularSpeedMaximum = 1e-3f;
static const PxReal gForcePairTotalMomentumMaximum = 1e-3f;
static const PxReal gForcePairSeparationErrorMaximum = 1e-3f;
static const PxReal gForcePairRelativeSpeedMaximum = 1e-3f;
static const PxReal gForcePairCenterOfMassErrorMaximum = 1e-3f;
static const PxReal gDisabledPairSeparationErrorMinimum = 100.0f;
static const PxReal gDisabledPairRelativeSpeedMinimum = 50.0f;
static const PxReal gDisabledPairReactionMaximum = 1e-4f;
static const PxReal gDisabledPairCenterOfMassErrorMaximum = 1e-2f;
static const PxReal gDisabledPairTotalMomentumMaximum = 5e-2f;

struct RevoluteMotorStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal finalRelativeVelocity;
  PxReal finalRelativeError;
  PxReal maximumLateRelativeError;
  PxReal maximumAngularMomentumDrift;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorStats()
      : stateSamples(0), nonFiniteSamples(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), finalRelativeVelocity(0.0f),
        finalRelativeError(PX_MAX_F32),
        maximumLateRelativeError(0.0f),
        maximumAngularMomentumDrift(0.0f), maximumAnchorError(0.0f),
        maximumAxisMisalignment(0.0f), driveEnabledReadback(false),
        actorOrderValid(false), cleanupComplete(false) {}
};

static RevoluteMotorStats gRevoluteMotorStats;
static PxRigidDynamic *gRevoluteMotorBodyA = NULL;
static PxRigidDynamic *gRevoluteMotorBodyB = NULL;
static PxRevoluteJoint *gRevoluteMotorJoint = NULL;
static const PxReal gRevoluteMotorTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorForceLimit = 1000.0f;
static const PxReal gRevoluteMotorInertiaA = 1.0f;
static const PxReal gRevoluteMotorInertiaB = 3.0f;
static const PxU32 gRevoluteMotorLateBeginFrame = 30;
static const PxReal gRevoluteMotorRelativeErrorMaximum = 0.05f;
static const PxReal gRevoluteMotorMomentumDriftMaximum = 1e-3f;
static const PxReal gRevoluteMotorAnchorErrorMaximum = 1e-3f;
static const PxReal gRevoluteMotorAxisMisalignmentMaximum = 1e-3f;

struct RevoluteMotorLimitStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 reverseEvents;
  PxU32 upperBoundSamples;
  PxU32 lowerBoundSamples;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal finalTargetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal lowerLimitReadback;
  PxReal upperLimitReadback;
  PxReal initialAngle;
  PxReal finalAngle;
  PxReal minimumAngle;
  PxReal maximumAngle;
  PxReal maximumUpperViolation;
  PxReal maximumLowerViolation;
  PxReal maximumLateOutwardVelocity;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool limitEnabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorLimitStats()
      : stateSamples(0), nonFiniteSamples(0), reverseEvents(0),
        upperBoundSamples(0), lowerBoundSamples(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        finalTargetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), lowerLimitReadback(0.0f),
        upperLimitReadback(0.0f), initialAngle(0.0f), finalAngle(0.0f),
        minimumAngle(PX_MAX_F32), maximumAngle(-PX_MAX_F32),
        maximumUpperViolation(0.0f), maximumLowerViolation(0.0f),
        maximumLateOutwardVelocity(0.0f), maximumAnchorError(0.0f),
        maximumAxisMisalignment(0.0f), driveEnabledReadback(false),
        limitEnabledReadback(false), actorOrderValid(false),
        cleanupComplete(false) {}
};

static RevoluteMotorLimitStats gRevoluteMotorLimitStats;
static PxRigidDynamic *gRevoluteMotorLimitBodyA = NULL;
static PxRigidDynamic *gRevoluteMotorLimitBody = NULL;
static PxRevoluteJoint *gRevoluteMotorLimitJoint = NULL;
static const PxReal gRevoluteMotorLimitTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorLimitForceLimit = 1000.0f;
static const PxReal gRevoluteMotorLimitLower = -0.5f;
static const PxReal gRevoluteMotorLimitUpper = 0.5f;
static const PxU32 gRevoluteMotorLimitReverseFrame = 180;
static const PxReal gRevoluteMotorLimitTravelMinimum = 0.4f;
static const PxReal gRevoluteMotorLimitRangeMinimum = 0.9f;
static const PxU32 gRevoluteMotorLimitBoundarySettleSamples = 8;
static const PxReal gRevoluteMotorLimitFinalTolerance = 0.05f;
static const PxReal gRevoluteMotorLimitViolationMaximum = 0.02f;
static const PxReal gRevoluteMotorLimitOutwardVelocityMaximum = 0.05f;
static const PxReal gRevoluteMotorLimitAnchorErrorMaximum = 1e-3f;
static const PxReal gRevoluteMotorLimitAxisMisalignmentMaximum = 1e-3f;

struct RevoluteMotorFreeSpinStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 boostEvents;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal boostVelocityReadback;
  PxReal preBoostFinalVelocity;
  PxReal maximumLatePreBoostError;
  PxReal finalVelocity;
  PxReal minimumPostBoostVelocity;
  PxReal maximumPostBoostVelocityDrop;
  PxReal maximumAngularMomentumDrift;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool freeSpinEnabledReadback;
  bool limitDisabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorFreeSpinStats()
      : stateSamples(0), nonFiniteSamples(0), boostEvents(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), boostVelocityReadback(0.0f),
        preBoostFinalVelocity(0.0f),
        maximumLatePreBoostError(0.0f), finalVelocity(0.0f),
        minimumPostBoostVelocity(PX_MAX_F32),
        maximumPostBoostVelocityDrop(0.0f),
        maximumAngularMomentumDrift(0.0f), maximumAnchorError(0.0f),
        maximumAxisMisalignment(0.0f), driveEnabledReadback(false),
        freeSpinEnabledReadback(false), limitDisabledReadback(false),
        actorOrderValid(false), cleanupComplete(false) {}
};

static RevoluteMotorFreeSpinStats gRevoluteMotorFreeSpinStats;
static PxRigidDynamic *gRevoluteMotorFreeSpinBodyA = NULL;
static PxRigidDynamic *gRevoluteMotorFreeSpinBody = NULL;
static PxRevoluteJoint *gRevoluteMotorFreeSpinJoint = NULL;
static const PxReal gRevoluteMotorFreeSpinTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorFreeSpinForceLimit = 1000.0f;
static const PxReal gRevoluteMotorFreeSpinBoostVelocity = 5.0f;
static const PxU32 gRevoluteMotorFreeSpinLateBeginFrame = 30;
static const PxU32 gRevoluteMotorFreeSpinBoostFrame = 120;
static const PxReal gRevoluteMotorFreeSpinTargetErrorMaximum = 0.05f;
static const PxReal gRevoluteMotorFreeSpinMinimumCoastVelocity = 4.9f;
static const PxReal gRevoluteMotorFreeSpinVelocityDropMaximum = 0.1f;
static const PxReal gRevoluteMotorFreeSpinAnchorErrorMaximum = 1e-3f;
static const PxReal gRevoluteMotorFreeSpinAxisMisalignmentMaximum = 1e-3f;

struct RevoluteMotorRatioStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal driveGearRatioReadback;
  PxReal finalVelocityA;
  PxReal finalVelocityB;
  PxReal finalWeightedVelocity;
  PxReal finalWeightedVelocityError;
  PxReal maximumLateWeightedVelocityError;
  PxReal initialOffPrincipalResponseA;
  PxReal initialOffPrincipalResponseB;
  PxReal maximumLateRelativeSwingVelocity;
  PxReal maximumInitialGeneralizedMomentumDrift;
  PxReal maximumGeneralizedMomentumDrift;
  PxReal initialPerpendicularLeverArmA;
  PxReal initialPerpendicularLeverArmB;
  PxReal finalRelativeAnchorPointSpeed;
  PxReal maximumLateRelativeAnchorPointSpeed;
  PxReal maximumTotalLinearMomentum;
  PxReal maximumInitialTotalAngularMomentum;
  PxReal maximumLinearSpeed;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool freeSpinDisabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorRatioStats()
      : stateSamples(0), nonFiniteSamples(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), driveGearRatioReadback(0.0f),
        finalVelocityA(0.0f), finalVelocityB(0.0f),
        finalWeightedVelocity(0.0f),
        finalWeightedVelocityError(PX_MAX_F32),
        maximumLateWeightedVelocityError(0.0f),
        initialOffPrincipalResponseA(0.0f),
        initialOffPrincipalResponseB(0.0f),
        maximumLateRelativeSwingVelocity(0.0f),
        maximumInitialGeneralizedMomentumDrift(0.0f),
        maximumGeneralizedMomentumDrift(0.0f),
        initialPerpendicularLeverArmA(0.0f),
        initialPerpendicularLeverArmB(0.0f),
        finalRelativeAnchorPointSpeed(PX_MAX_F32),
        maximumLateRelativeAnchorPointSpeed(0.0f),
        maximumTotalLinearMomentum(0.0f),
        maximumInitialTotalAngularMomentum(0.0f),
        maximumLinearSpeed(0.0f),
        maximumAnchorError(0.0f), maximumAxisMisalignment(0.0f),
        driveEnabledReadback(false), freeSpinDisabledReadback(false),
        actorOrderValid(false), cleanupComplete(false) {}
};

static RevoluteMotorRatioStats gRevoluteMotorRatioStats;
static PxRigidDynamic *gRevoluteMotorRatioBodyA = NULL;
static PxRigidDynamic *gRevoluteMotorRatioBodyB = NULL;
static PxRevoluteJoint *gRevoluteMotorRatioJoint = NULL;
static PxVec3 gRevoluteMotorRatioConfiguredAnchorA(0.0f);
static PxVec3 gRevoluteMotorRatioConfiguredAnchorB(0.0f);
static const PxReal gRevoluteMotorRatioTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorRatioForceLimit = 1000.0f;
static const PxReal gRevoluteMotorRatioDriveGearRatio = 2.5f;
static const PxReal gRevoluteMotorRatioInertiaA = 1.0f;
static const PxReal gRevoluteMotorRatioInertiaB = 3.0f;
static const PxVec3 gRevoluteMotorDynamicOffPrincipalInertiaA(
    1.0f, 4.0f, 7.0f);
static const PxVec3 gRevoluteMotorDynamicOffPrincipalInertiaB(
    2.0f, 5.0f, 8.0f);
static const PxU32 gRevoluteMotorRatioLateBeginFrame = 30;
static const PxU32
    gRevoluteMotorDynamicOffPrincipalMomentumWindowFrames = 12;
static const PxReal gRevoluteMotorRatioVelocityErrorMaximum = 0.05f;
static const PxReal
    gRevoluteMotorDynamicOffPrincipalResponseMinimum = 0.05f;
static const PxReal
    gRevoluteMotorDynamicOffPrincipalSwingVelocityMaximum = 0.05f;
static const PxReal gRevoluteMotorRatioMomentumDriftMaximum = 1e-3f;
static const PxReal
    gRevoluteMotorDynamicOffPrincipalInitialMomentumDriftMaximum =
        0.25f;
static const PxReal
    gRevoluteMotorDynamicOffCenterLeverArmMinimum = 0.5f;
static const PxReal
    gRevoluteMotorDynamicOffCenterAnchorSpeedMaximum = 0.05f;
static const PxReal
    gRevoluteMotorDynamicOffCenterLinearMomentumMaximum = 1e-3f;
static const PxReal
    gRevoluteMotorDynamicOffCenterInitialAngularMomentumMaximum =
        0.25f;
static const PxReal
    gRevoluteMotorDynamicOffCenterLinearSpeedMinimum = 0.5f;
static const PxReal gRevoluteMotorRatioAnchorErrorMaximum = 1e-3f;
static const PxReal gRevoluteMotorRatioAxisMisalignmentMaximum = 1e-3f;

struct RevoluteMotorContactStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 contactEvents;
  PxU32 contactPointCount;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal finalVelocityA;
  PxReal finalVelocityB;
  PxReal finalRelativeVelocity;
  PxReal finalRelativeError;
  PxReal maximumLateRelativeError;
  PxU32 lateDriveReactionSamples;
  PxReal lateDriveReactionSum;
  PxReal maximumLateDriveReaction;
  PxReal totalNormalImpulse;
  PxReal totalTangentialImpulse;
  PxReal maximumTangentialImpulse;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  PxReal maximumCenterHeightError;
  bool driveEnabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorContactStats()
      : stateSamples(0), nonFiniteSamples(0), contactEvents(0),
        contactPointCount(0), initialDynamicActors(PX_MAX_U32),
        initialStaticActors(PX_MAX_U32), initialConstraints(PX_MAX_U32),
        finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), finalVelocityA(0.0f),
        finalVelocityB(0.0f), finalRelativeVelocity(0.0f),
        finalRelativeError(PX_MAX_F32),
        maximumLateRelativeError(0.0f), lateDriveReactionSamples(0),
        lateDriveReactionSum(0.0f), maximumLateDriveReaction(0.0f),
        totalNormalImpulse(0.0f),
        totalTangentialImpulse(0.0f), maximumTangentialImpulse(0.0f),
        maximumAnchorError(0.0f), maximumAxisMisalignment(0.0f),
        maximumCenterHeightError(0.0f), driveEnabledReadback(false),
        actorOrderValid(false), cleanupComplete(false) {}
};

static RevoluteMotorContactStats gRevoluteMotorContactStats;
static PxRigidDynamic *gRevoluteMotorContactBodyA = NULL;
static PxRigidDynamic *gRevoluteMotorContactBodyB = NULL;
static PxRigidStatic *gRevoluteMotorContactGround = NULL;
static PxRevoluteJoint *gRevoluteMotorContactJoint = NULL;
static const PxReal gRevoluteMotorContactTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorContactForceLimit = 1000.0f;
static const PxReal gRevoluteMotorContactRadius = 0.5f;
static const PxReal gRevoluteMotorContactHalfHeight = 0.5f;
static const PxReal gRevoluteMotorContactCenterHeight = 0.5f;
static const PxU32 gRevoluteMotorContactLateBeginFrame = 60;
static const PxReal gRevoluteMotorContactVelocityErrorMaximum = 0.05f;
static const PxReal gRevoluteMotorContactDriveReactionMinimum = 1e-3f;
static const PxReal gRevoluteMotorContactAnchorErrorMaximum = 1e-3f;
static const PxReal gRevoluteMotorContactAxisMisalignmentMaximum = 1e-3f;
static const PxReal gRevoluteMotorContactCenterHeightErrorMaximum = 0.02f;

struct RevoluteMotorKinematicStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 targetUpdates;
  PxU32 targetUpdateFailures;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal finalKinematicVelocity;
  PxReal finalDynamicVelocity;
  PxReal finalRelativeVelocity;
  PxReal finalRelativeError;
  PxReal maximumLateRelativeError;
  PxReal maximumLateKinematicVelocityError;
  PxReal maximumLateDynamicVelocityError;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool actorOrderValid;
  bool kinematicFlagReadback;
  bool cleanupComplete;

  RevoluteMotorKinematicStats()
      : stateSamples(0), nonFiniteSamples(0), targetUpdates(0),
        targetUpdateFailures(0), initialDynamicActors(PX_MAX_U32),
        initialStaticActors(PX_MAX_U32), initialConstraints(PX_MAX_U32),
        finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), finalKinematicVelocity(0.0f),
        finalDynamicVelocity(0.0f), finalRelativeVelocity(0.0f),
        finalRelativeError(PX_MAX_F32),
        maximumLateRelativeError(0.0f),
        maximumLateKinematicVelocityError(0.0f),
        maximumLateDynamicVelocityError(0.0f),
        maximumAnchorError(0.0f), maximumAxisMisalignment(0.0f),
        driveEnabledReadback(false), actorOrderValid(false),
        kinematicFlagReadback(false), cleanupComplete(false) {}
};

static RevoluteMotorKinematicStats gRevoluteMotorKinematicStats;
static PxRigidDynamic *gRevoluteMotorKinematicBody = NULL;
static PxRigidDynamic *gRevoluteMotorKinematicDynamicBody = NULL;
static PxRevoluteJoint *gRevoluteMotorKinematicJoint = NULL;
static const PxTransform gRevoluteMotorKinematicInitialPose(
    PxVec3(0.0f, 10.0f, 0.0f));
static const PxReal gRevoluteMotorKinematicEndpointVelocity = 1.0f;
static const PxReal gRevoluteMotorKinematicTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorKinematicExpectedDynamicVelocity = 3.0f;
static const PxReal gRevoluteMotorKinematicForceLimit = 1000.0f;
static const PxU32 gRevoluteMotorKinematicLateBeginFrame = 60;
static const PxReal gRevoluteMotorKinematicVelocityErrorMaximum = 0.05f;
static const PxReal gRevoluteMotorKinematicAnchorErrorMaximum = 1e-3f;
static const PxReal gRevoluteMotorKinematicAxisMisalignmentMaximum = 1e-3f;

struct RevoluteMotorOffPrincipalStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal initialOffPrincipalResponse;
  PxReal finalHingeVelocity;
  PxReal finalHingeVelocityError;
  PxReal maximumLateHingeVelocityError;
  PxReal maximumLateSwingVelocity;
  PxReal maximumSwingReaction;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorOffPrincipalStats()
      : stateSamples(0), nonFiniteSamples(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f), initialOffPrincipalResponse(0.0f),
        finalHingeVelocity(0.0f),
        finalHingeVelocityError(PX_MAX_F32),
        maximumLateHingeVelocityError(0.0f),
        maximumLateSwingVelocity(0.0f),
        maximumSwingReaction(0.0f), maximumAnchorError(0.0f),
        maximumAxisMisalignment(0.0f), driveEnabledReadback(false),
        actorOrderValid(false), cleanupComplete(false) {}
};

static RevoluteMotorOffPrincipalStats
    gRevoluteMotorOffPrincipalStats;
static PxRigidDynamic *gRevoluteMotorOffPrincipalBody = NULL;
static PxRevoluteJoint *gRevoluteMotorOffPrincipalJoint = NULL;
static const PxVec3 gRevoluteMotorOffPrincipalInertia(1.0f, 4.0f, 7.0f);
static const PxReal gRevoluteMotorOffPrincipalAngle = PxPi / 6.0f;
static const PxReal gRevoluteMotorOffPrincipalTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorOffPrincipalForceLimit = 1000.0f;
static const PxU32 gRevoluteMotorOffPrincipalLateBeginFrame = 60;
static const PxReal gRevoluteMotorOffPrincipalResponseMinimum = 0.05f;
static const PxReal gRevoluteMotorOffPrincipalVelocityErrorMaximum = 0.05f;
static const PxReal gRevoluteMotorOffPrincipalSwingVelocityMaximum = 0.05f;
static const PxReal gRevoluteMotorOffPrincipalAnchorErrorMaximum = 1e-3f;
static const PxReal
    gRevoluteMotorOffPrincipalAxisMisalignmentMaximum = 1e-3f;

struct RevoluteMotorOffCenterStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal targetVelocityReadback;
  PxReal forceLimitReadback;
  PxReal initialPerpendicularLeverArm;
  PxReal initialOffPrincipalResponse;
  PxReal finalHingeVelocity;
  PxReal finalHingeVelocityError;
  PxReal maximumLateHingeVelocityError;
  PxReal maximumLateSwingVelocity;
  PxReal finalAnchorPointSpeed;
  PxReal maximumLateAnchorPointSpeed;
  PxReal maximumLinearSpeed;
  PxReal maximumLinearReaction;
  PxReal maximumAnchorError;
  PxReal maximumAxisMisalignment;
  bool driveEnabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  RevoluteMotorOffCenterStats()
      : stateSamples(0), nonFiniteSamples(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), targetVelocityReadback(0.0f),
        forceLimitReadback(0.0f),
        initialPerpendicularLeverArm(0.0f),
        initialOffPrincipalResponse(0.0f),
        finalHingeVelocity(0.0f),
        finalHingeVelocityError(PX_MAX_F32),
        maximumLateHingeVelocityError(0.0f),
        maximumLateSwingVelocity(0.0f),
        finalAnchorPointSpeed(PX_MAX_F32),
        maximumLateAnchorPointSpeed(0.0f),
        maximumLinearSpeed(0.0f), maximumLinearReaction(0.0f),
        maximumAnchorError(0.0f), maximumAxisMisalignment(0.0f),
        driveEnabledReadback(false), actorOrderValid(false),
        cleanupComplete(false) {}
};

static RevoluteMotorOffCenterStats gRevoluteMotorOffCenterStats;
static PxRigidDynamic *gRevoluteMotorOffCenterBody = NULL;
static PxRevoluteJoint *gRevoluteMotorOffCenterJoint = NULL;
static const PxTransform gRevoluteMotorOffCenterInitialPose(
    PxVec3(0.0f, 11.0f, 0.0f));
static const PxVec3 gRevoluteMotorOffCenterLocalAnchor(0.0f, -1.0f, 0.0f);
static PxVec3 gRevoluteMotorOffCenterConfiguredLocalAnchor(
    gRevoluteMotorOffCenterLocalAnchor);
static const PxReal gRevoluteMotorOffCenterTargetVelocity = 2.0f;
static const PxReal gRevoluteMotorOffCenterForceLimit = 1000.0f;
static const PxU32 gRevoluteMotorOffCenterLateBeginFrame = 60;
static const PxReal gRevoluteMotorOffCenterLeverArmMinimum = 0.5f;
static const PxReal gRevoluteMotorOffCenterVelocityErrorMaximum = 0.05f;
static const PxReal gRevoluteMotorOffCenterAnchorSpeedMaximum = 0.05f;
static const PxReal gRevoluteMotorOffCenterAnchorErrorMaximum = 2e-3f;
static const PxReal gRevoluteMotorOffCenterAxisMisalignmentMaximum = 1e-3f;

struct NativeBreakReactionStats {
  PxU32 stateSamples;
  PxU32 forceReads;
  PxU32 reactionSamples;
  PxU32 nonFiniteSamples;
  PxU32 firstBrokenFrame;
  PxU32 brokenPollCount;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxVec3 linearForceSum;
  PxVec3 angularForceSum;
  PxReal linearMagnitudeSum;
  PxReal angularMagnitudeSum;
  PxReal maximumPositionError;
  PxReal maximumRotationError;
  PxReal maximumLinearSpeed;
  PxReal maximumAngularSpeed;
  PxReal steadyMaximumPositionError;
  PxReal steadyMaximumRotationError;
  PxReal steadyMaximumLinearSpeed;
  PxReal steadyMaximumAngularSpeed;
  PxReal breakForceReadback;
  PxReal breakTorqueReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  NativeBreakReactionStats()
      : stateSamples(0), forceReads(0), reactionSamples(0),
        nonFiniteSamples(0), firstBrokenFrame(PX_MAX_U32),
        brokenPollCount(0), initialDynamicActors(PX_MAX_U32),
        initialStaticActors(PX_MAX_U32), initialConstraints(PX_MAX_U32),
        finalDynamicActors(PX_MAX_U32), finalStaticActors(PX_MAX_U32),
        finalConstraints(PX_MAX_U32), cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32), cleanupConstraints(PX_MAX_U32),
        linearForceSum(0.0f), angularForceSum(0.0f),
        linearMagnitudeSum(0.0f), angularMagnitudeSum(0.0f),
        maximumPositionError(0.0f), maximumRotationError(0.0f),
        maximumLinearSpeed(0.0f), maximumAngularSpeed(0.0f),
        steadyMaximumPositionError(0.0f),
        steadyMaximumRotationError(0.0f),
        steadyMaximumLinearSpeed(0.0f),
        steadyMaximumAngularSpeed(0.0f),
        breakForceReadback(PX_MAX_F32), breakTorqueReadback(PX_MAX_F32),
        actorOrderValid(false), cleanupComplete(false) {}
};

static NativeBreakReactionStats gNativeBreakReactionStats;
static PxRigidDynamic *gNativeBreakReactionBody = NULL;
static PxJoint *gNativeBreakReactionJoint = NULL;
static const PxVec3 gNativeLinearLoad(0.0f, 100.0f, 0.0f);
static const PxVec3 gNativeAngularLoad(0.0f, 0.0f, 100.0f);
static const PxReal gNativeLowBreakThreshold = 50.0f;
static const PxReal gNativeHighBreakThreshold = 200.0f;
static const PxU32 gNativeReactionWarmupFrames = 120;
static const PxReal gNativeReactionRatioMinimum = 0.9f;
static const PxReal gNativeReactionRatioMaximum = 1.1f;
static const PxReal gNativeReactionDirectionMinimum = 0.99f;
static const PxReal gNativeReactionOrthogonalRatioMaximum = 0.01f;
static const PxReal gNativeConstrainedErrorMaximum = 1e-3f;
static const PxReal gNativeConstrainedSpeedMaximum = 1e-3f;
static const PxReal gNativePostBreakLinearSpeedMinimum = 1.0f;
static const PxReal gNativePostBreakAngularSpeedMinimum = 1.0f;

struct SphericalConeStats {
  PxU32 stateSamples;
  PxU32 nonFiniteSamples;
  PxU32 initialDynamicActors;
  PxU32 initialStaticActors;
  PxU32 initialConstraints;
  PxU32 finalDynamicActors;
  PxU32 finalStaticActors;
  PxU32 finalConstraints;
  PxU32 cleanupDynamicActors;
  PxU32 cleanupStaticActors;
  PxU32 cleanupConstraints;
  PxReal limitYReadback;
  PxReal limitZReadback;
  PxReal initialEllipseRadius;
  PxReal finalEllipseRadius;
  PxReal minimumEllipseRadius;
  PxReal maximumEllipseRadius;
  PxReal maximumLateEllipseRadius;
  PxReal minimumLateEllipseRadius;
  PxReal maximumInsideDeviation;
  PxReal maximumTotalAngularMomentum;
  PxReal maximumAnchorSeparation;
  bool limitEnabledReadback;
  bool actorOrderValid;
  bool cleanupComplete;

  SphericalConeStats()
      : stateSamples(0), nonFiniteSamples(0),
        initialDynamicActors(PX_MAX_U32), initialStaticActors(PX_MAX_U32),
        initialConstraints(PX_MAX_U32), finalDynamicActors(PX_MAX_U32),
        finalStaticActors(PX_MAX_U32), finalConstraints(PX_MAX_U32),
        cleanupDynamicActors(PX_MAX_U32),
        cleanupStaticActors(PX_MAX_U32),
        cleanupConstraints(PX_MAX_U32), limitYReadback(PX_MAX_F32),
        limitZReadback(PX_MAX_F32), initialEllipseRadius(PX_MAX_F32),
        finalEllipseRadius(PX_MAX_F32), minimumEllipseRadius(PX_MAX_F32),
        maximumEllipseRadius(0.0f), maximumLateEllipseRadius(0.0f),
        minimumLateEllipseRadius(PX_MAX_F32),
        maximumInsideDeviation(0.0f),
        maximumTotalAngularMomentum(0.0f),
        maximumAnchorSeparation(0.0f), limitEnabledReadback(false),
        actorOrderValid(false), cleanupComplete(false) {}
};

static SphericalConeStats gSphericalConeStats;
static PxRigidActor *gSphericalConeActorA = NULL;
static PxRigidDynamic *gSphericalConeDynamicA = NULL;
static PxRigidDynamic *gSphericalConeActorB = NULL;
static PxSphericalJoint *gSphericalConeJoint = NULL;
static const PxReal gSphericalConeLimitY = PxPi / 9.0f;
static const PxReal gSphericalConeLimitZ = 7.0f * PxPi / 36.0f;
static const PxReal gSphericalConeInsideY = PxPi / 36.0f;
static const PxReal gSphericalConeInsideZ = PxPi / 6.0f;
static const PxReal gSphericalConeOutsideY = PxPi / 10.0f;
static const PxReal gSphericalConeOutsideZ = PxPi / 6.0f;
static const PxReal gSphericalConeFinalRadiusTolerance = 0.01f;
static const PxReal gSphericalConeLateRadiusTolerance = 0.02f;
static const PxReal gSphericalConeMinimumRadiusCorrection = 0.10f;
static const PxReal gSphericalConeInsideDeviationTolerance = 0.01f;
static const PxReal gSphericalConeAngularMomentumMaximum = 1e-3f;
static const PxReal gSphericalConeAnchorSeparationMaximum = 1e-3f;
static const PxU32 gSphericalConeLateFrames = 60;

struct PrismaticDriftStats {
  PxReal maxTailTransverse = 0.0f;
  PxReal sumTailEarly = 0.0f;
  PxReal sumTailLate = 0.0f;
  PxReal sumTailAngVelEarly = 0.0f;
  PxReal sumTailAngVelLate = 0.0f;
  PxReal maxJointAxisMisalignDeg = 0.0f;
  PxU32 cntEarly = 0;
  PxU32 cntLate = 0;
};

static PrismaticDriftStats gPrismaticStats;

static const char *getJointKindName(JointKind kind) {
  switch (kind) {
  case eJOINT_SPHERICAL:
    return "spherical";
  case eJOINT_FIXED:
    return "fixed";
  case eJOINT_D6:
    return "d6";
  case eJOINT_PRISMATIC:
    return "prismatic";
  case eJOINT_REVOLUTE:
    return "revolute";
  default:
    return "unknown";
  }
}

static bool tryParseJointKind(const char *value, JointKind &kind) {
  for (PxU32 i = 0; i < 5; ++i) {
    const JointKind candidate = static_cast<JointKind>(i);
    if (Snippets::equalsIgnoreCase(value, getJointKindName(candidate))) {
      kind = candidate;
      return true;
    }
  }
  return false;
}

static const char *getEndpointName(EndpointKind endpoint) {
  switch (endpoint) {
  case eENDPOINT_FORWARD:
    return "forward";
  case eENDPOINT_REVERSE:
    return "reverse";
  case eENDPOINT_IMPLICIT:
  default:
    return "implicit";
  }
}

static const char *getEndpointActor0Name() {
  if (gEndpointKind == eENDPOINT_FORWARD)
    return "world";
  if (gEndpointKind == eENDPOINT_REVERSE)
    return "dynamic";
  return "legacy";
}

static const char *getEndpointActor1Name() {
  if (gEndpointKind == eENDPOINT_FORWARD)
    return "dynamic";
  if (gEndpointKind == eENDPOINT_REVERSE)
    return "world";
  return "legacy";
}

static bool tryParseEndpointKind(const char *value, EndpointKind &endpoint) {
  if (Snippets::equalsIgnoreCase(value, "forward"))
    endpoint = eENDPOINT_FORWARD;
  else if (Snippets::equalsIgnoreCase(value, "reverse"))
    endpoint = eENDPOINT_REVERSE;
  else
    return false;
  return true;
}

static bool tryParseHeadlessCase(const char *value,
                                 JointHeadlessCase &headlessCase) {
  if (Snippets::equalsIgnoreCase(value, "passive"))
    headlessCase = eCASE_PASSIVE;
  else if (Snippets::equalsIgnoreCase(value, "impact-all"))
    headlessCase = eCASE_IMPACT_ALL;
  else if (Snippets::equalsIgnoreCase(value, "wide-joint-stress"))
    headlessCase = eCASE_WIDE_JOINT_STRESS;
  else if (Snippets::equalsIgnoreCase(value, "impact"))
    headlessCase = eCASE_IMPACT_SINGLE;
  else if (Snippets::equalsIgnoreCase(value, "fixed-no-break"))
    headlessCase = eCASE_FIXED_NO_BREAK;
  else if (Snippets::equalsIgnoreCase(value, "fixed-break"))
    headlessCase = eCASE_FIXED_BREAK;
  else if (Snippets::equalsIgnoreCase(value, "force-static"))
    headlessCase = eCASE_FORCE_STATIC;
  else if (Snippets::equalsIgnoreCase(value, "force-offset"))
    headlessCase = eCASE_FORCE_OFFSET;
  else if (Snippets::equalsIgnoreCase(value, "force-pair"))
    headlessCase = eCASE_FORCE_PAIR;
  else if (Snippets::equalsIgnoreCase(value, "force-pair-disabled"))
    headlessCase = eCASE_FORCE_PAIR_DISABLED;
  else if (Snippets::equalsIgnoreCase(value, "spherical-cone-inside"))
    headlessCase = eCASE_SPHERICAL_CONE_INSIDE;
  else if (Snippets::equalsIgnoreCase(value, "spherical-cone-outside"))
    headlessCase = eCASE_SPHERICAL_CONE_OUTSIDE;
  else if (Snippets::equalsIgnoreCase(value, "native-reaction"))
    headlessCase = eCASE_NATIVE_REACTION;
  else if (Snippets::equalsIgnoreCase(value, "native-no-break"))
    headlessCase = eCASE_NATIVE_NO_BREAK;
  else if (Snippets::equalsIgnoreCase(value, "native-break"))
    headlessCase = eCASE_NATIVE_BREAK;
  else if (Snippets::equalsIgnoreCase(value, "revolute-motor"))
    headlessCase = eCASE_REVOLUTE_MOTOR;
  else if (Snippets::equalsIgnoreCase(value, "revolute-motor-limit"))
    headlessCase = eCASE_REVOLUTE_MOTOR_LIMIT;
  else if (Snippets::equalsIgnoreCase(
               value, "revolute-motor-dynamic-limit"))
    headlessCase = eCASE_REVOLUTE_MOTOR_DYNAMIC_LIMIT;
  else if (Snippets::equalsIgnoreCase(value, "revolute-motor-freespin"))
    headlessCase = eCASE_REVOLUTE_MOTOR_FREESPIN;
  else if (Snippets::equalsIgnoreCase(
               value, "revolute-motor-dynamic-freespin"))
    headlessCase = eCASE_REVOLUTE_MOTOR_DYNAMIC_FREESPIN;
  else if (Snippets::equalsIgnoreCase(value, "revolute-motor-ratio"))
    headlessCase = eCASE_REVOLUTE_MOTOR_RATIO;
  else if (Snippets::equalsIgnoreCase(value, "revolute-motor-contact"))
    headlessCase = eCASE_REVOLUTE_MOTOR_CONTACT;
  else if (Snippets::equalsIgnoreCase(value, "revolute-motor-kinematic"))
    headlessCase = eCASE_REVOLUTE_MOTOR_KINEMATIC;
  else if (Snippets::equalsIgnoreCase(value,
                                      "revolute-motor-off-principal"))
    headlessCase = eCASE_REVOLUTE_MOTOR_OFF_PRINCIPAL;
  else if (Snippets::equalsIgnoreCase(value,
                                      "revolute-motor-off-center"))
    headlessCase = eCASE_REVOLUTE_MOTOR_OFF_CENTER;
  else if (Snippets::equalsIgnoreCase(value,
                                      "revolute-motor-spatial"))
    headlessCase = eCASE_REVOLUTE_MOTOR_SPATIAL;
  else if (Snippets::equalsIgnoreCase(
               value, "revolute-motor-dynamic-off-principal"))
    headlessCase =
        eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_PRINCIPAL;
  else if (Snippets::equalsIgnoreCase(
               value, "revolute-motor-dynamic-off-center"))
    headlessCase =
        eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_CENTER;
  else if (Snippets::equalsIgnoreCase(
               value, "revolute-motor-dynamic-spatial"))
    headlessCase =
        eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL;
  else
    return false;
  return true;
}

static const char *getHeadlessCaseName(JointHeadlessCase headlessCase) {
  switch (headlessCase) {
  case eCASE_PASSIVE:
    return "passive";
  case eCASE_IMPACT_ALL:
    return "impact-all";
  case eCASE_WIDE_JOINT_STRESS:
    return "wide-joint-stress";
  case eCASE_IMPACT_SINGLE:
    return "impact";
  case eCASE_FIXED_NO_BREAK:
    return "fixed-no-break";
  case eCASE_FIXED_BREAK:
    return "fixed-break";
  case eCASE_FORCE_STATIC:
    return "force-static";
  case eCASE_FORCE_OFFSET:
    return "force-offset";
  case eCASE_FORCE_PAIR:
    return "force-pair";
  case eCASE_FORCE_PAIR_DISABLED:
    return "force-pair-disabled";
  case eCASE_SPHERICAL_CONE_INSIDE:
    return "spherical-cone-inside";
  case eCASE_SPHERICAL_CONE_OUTSIDE:
    return "spherical-cone-outside";
  case eCASE_NATIVE_REACTION:
    return "native-reaction";
  case eCASE_NATIVE_NO_BREAK:
    return "native-no-break";
  case eCASE_NATIVE_BREAK:
    return "native-break";
  case eCASE_REVOLUTE_MOTOR:
    return "revolute-motor";
  case eCASE_REVOLUTE_MOTOR_LIMIT:
    return "revolute-motor-limit";
  case eCASE_REVOLUTE_MOTOR_DYNAMIC_LIMIT:
    return "revolute-motor-dynamic-limit";
  case eCASE_REVOLUTE_MOTOR_FREESPIN:
    return "revolute-motor-freespin";
  case eCASE_REVOLUTE_MOTOR_DYNAMIC_FREESPIN:
    return "revolute-motor-dynamic-freespin";
  case eCASE_REVOLUTE_MOTOR_RATIO:
    return "revolute-motor-ratio";
  case eCASE_REVOLUTE_MOTOR_CONTACT:
    return "revolute-motor-contact";
  case eCASE_REVOLUTE_MOTOR_KINEMATIC:
    return "revolute-motor-kinematic";
  case eCASE_REVOLUTE_MOTOR_OFF_PRINCIPAL:
    return "revolute-motor-off-principal";
  case eCASE_REVOLUTE_MOTOR_OFF_CENTER:
    return "revolute-motor-off-center";
  case eCASE_REVOLUTE_MOTOR_SPATIAL:
    return "revolute-motor-spatial";
  case eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_PRINCIPAL:
    return "revolute-motor-dynamic-off-principal";
  case eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_CENTER:
    return "revolute-motor-dynamic-off-center";
  case eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL:
    return "revolute-motor-dynamic-spatial";
  default:
    return "unknown";
  }
}

static bool isSphericalConeCase() {
  return gHeadlessCase == eCASE_SPHERICAL_CONE_INSIDE ||
         gHeadlessCase == eCASE_SPHERICAL_CONE_OUTSIDE;
}

static bool isSphericalConeInsideCase() {
  return gHeadlessCase == eCASE_SPHERICAL_CONE_INSIDE;
}

static bool isNativeBreakReactionCase() {
  return gHeadlessCase == eCASE_NATIVE_REACTION ||
         gHeadlessCase == eCASE_NATIVE_NO_BREAK ||
         gHeadlessCase == eCASE_NATIVE_BREAK;
}

static bool isNativeBreakCase() {
  return gHeadlessCase == eCASE_NATIVE_BREAK;
}

static bool isRevoluteMotorCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR;
}

static bool isRevoluteMotorLimitCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_LIMIT ||
         gHeadlessCase == eCASE_REVOLUTE_MOTOR_DYNAMIC_LIMIT;
}

static bool isRevoluteMotorDynamicLimitCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_DYNAMIC_LIMIT;
}

static bool isRevoluteMotorFreeSpinCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_FREESPIN ||
         gHeadlessCase == eCASE_REVOLUTE_MOTOR_DYNAMIC_FREESPIN;
}

static bool isRevoluteMotorDynamicFreeSpinCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_DYNAMIC_FREESPIN;
}

static bool isRevoluteMotorRatioCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_RATIO ||
         gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_PRINCIPAL ||
         gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_CENTER ||
         gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL;
}

static bool isRevoluteMotorDynamicOffPrincipalCase() {
  return gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_PRINCIPAL ||
         gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL;
}

static bool isRevoluteMotorDynamicOffCenterCase() {
  return gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_OFF_CENTER ||
         gHeadlessCase ==
             eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL;
}

static bool isRevoluteMotorDynamicSpatialCase() {
  return gHeadlessCase ==
         eCASE_REVOLUTE_MOTOR_DYNAMIC_SPATIAL;
}

static bool isRevoluteMotorContactCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_CONTACT;
}

static bool isRevoluteMotorKinematicCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_KINEMATIC;
}

static bool isRevoluteMotorOffPrincipalCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_OFF_PRINCIPAL;
}

static bool isRevoluteMotorOffCenterCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_OFF_CENTER ||
         gHeadlessCase == eCASE_REVOLUTE_MOTOR_SPATIAL;
}

static bool isRevoluteMotorSpatialCase() {
  return gHeadlessCase == eCASE_REVOLUTE_MOTOR_SPATIAL;
}

static bool isRevoluteMotorFamilyCase() {
  return isRevoluteMotorCase() || isRevoluteMotorLimitCase() ||
         isRevoluteMotorFreeSpinCase() ||
         isRevoluteMotorRatioCase() ||
         isRevoluteMotorContactCase() ||
         isRevoluteMotorKinematicCase() ||
         isRevoluteMotorOffPrincipalCase() ||
         isRevoluteMotorOffCenterCase();
}

static bool isNativeAngularReactionCase() {
  return isNativeBreakReactionCase() &&
         gImpactJointKind == eJOINT_REVOLUTE;
}

static const char *getSphericalConeTopologyName() {
  return gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC
             ? "dynamic-dynamic"
             : "static-dynamic";
}

static bool tryParseSphericalConeTopology(
    const char *value, SphericalConeTopology &topology) {
  if (Snippets::equalsIgnoreCase(value, "static-dynamic"))
    topology = eSPHERICAL_CONE_STATIC_DYNAMIC;
  else if (Snippets::equalsIgnoreCase(value, "dynamic-dynamic"))
    topology = eSPHERICAL_CONE_DYNAMIC_DYNAMIC;
  else
    return false;
  return true;
}

static bool isImpactCase() {
  return gHeadlessCase == eCASE_IMPACT_ALL ||
         gHeadlessCase == eCASE_IMPACT_SINGLE ||
         gHeadlessCase == eCASE_FIXED_NO_BREAK ||
         gHeadlessCase == eCASE_FIXED_BREAK;
}

static bool isWideJointStressCase() {
  return gHeadlessCase == eCASE_WIDE_JOINT_STRESS;
}

static bool isForceStaticCase() {
  return gHeadlessCase == eCASE_FORCE_STATIC;
}

static bool isForceOffsetCase() {
  return gHeadlessCase == eCASE_FORCE_OFFSET;
}

static bool isForcePairCase() {
  return gHeadlessCase == eCASE_FORCE_PAIR ||
         gHeadlessCase == eCASE_FORCE_PAIR_DISABLED;
}

static bool isForcePairDisabledCase() {
  return gHeadlessCase == eCASE_FORCE_PAIR_DISABLED;
}

static bool isForceReactionCase() {
  return isForceStaticCase() || isForceOffsetCase() || isForcePairCase();
}

static const char *getForceActorOrderName() {
  return gForceActorOrder == eFORCE_ACTOR_ORDER_SWAPPED ? "swapped"
                                                        : "normal";
}

static const char *getForceFixtureName() {
  if (isForcePairCase())
    return isForcePairDisabledCase() ? "dynamic-pair-disabled"
                                     : "dynamic-pair";
  return isForceOffsetCase() ? "offset" : "centered";
}

static bool tryParseForceActorOrder(const char *value,
                                    ForceActorOrder &actorOrder) {
  if (Snippets::equalsIgnoreCase(value, "normal"))
    actorOrder = eFORCE_ACTOR_ORDER_NORMAL;
  else if (Snippets::equalsIgnoreCase(value, "swapped"))
    actorOrder = eFORCE_ACTOR_ORDER_SWAPPED;
  else
    return false;
  return true;
}

static bool isEndpointProbe() {
  return gEndpointKind == eENDPOINT_FORWARD ||
         gEndpointKind == eENDPOINT_REVERSE;
}

static bool isRevoluteEndpointProbe() {
  return isEndpointProbe() && gImpactJointKind == eJOINT_REVOLUTE;
}

static bool usesProjectileExcitation() {
  return isImpactCase() && !isRevoluteEndpointProbe();
}

static const char *getEndpointResponseAuthorityName() {
  if (!isEndpointProbe())
    return "NOT_COVERED";
  return isRevoluteEndpointProbe() ? "PROBE" : "GATED";
}

static const char *getEndpointTargetDeltaQuantityName() {
  if (!isEndpointProbe())
    return "NOT_COVERED";
  return isRevoluteEndpointProbe() ? "rotation_vector" : "position";
}

static const char *getEndpointVelocityDeltaQuantityName() {
  if (!isEndpointProbe())
    return "NOT_COVERED";
  return isRevoluteEndpointProbe() ? "angular_velocity" : "linear_velocity";
}

static const char *getEndpointJointKindName() {
  if (!isEndpointProbe())
    return "NOT_COVERED";
  return isRevoluteEndpointProbe() ? "revolute" : "prismatic";
}

static const char *getEndpointExcitationName() {
  if (!isEndpointProbe())
    return "none";
  return isRevoluteEndpointProbe() ? "direct-angular-velocity" : "projectile";
}

static const char *getEndpointPerSampleSignedGateName() {
  return isRevoluteEndpointProbe() ? "positive" : "NOT_COVERED";
}

static bool getForceStaticFrequency(PxReal dt, PxU32 &frequency) {
  const PxU32 candidates[] = {30, 60, 120};
  for (PxU32 i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
    if (PxAbs(dt - 1.0f / PxReal(candidates[i])) <= 1e-7f) {
      frequency = candidates[i];
      return true;
    }
  }
  frequency = 0;
  return false;
}

static TargetResponseKind getTargetResponseKind(PxU32 chainIndex) {
  if (chainIndex >= gChains.size())
    return eTARGET_RESPONSE_NONE;
  // A -Z center-of-mass impulse can bend spherical/D6 chains through their
  // free angular DOFs. Fixed/prismatic chains constrain that response, while
  // the revolute X axis only admits the offset torque and does not provide an
  // unambiguous translational momentum witness. Keep those cases explicit N/A
  // until a validated reaction-force oracle exists.
  switch (gChains[chainIndex].kind) {
  case eJOINT_SPHERICAL:
  case eJOINT_D6:
    return eTARGET_RESPONSE_LINEAR_Z;
  case eJOINT_FIXED:
  case eJOINT_PRISMATIC:
  case eJOINT_REVOLUTE:
  default:
    return eTARGET_RESPONSE_NONE;
  }
}

static const char *getTargetResponseKindName(TargetResponseKind kind) {
  switch (kind) {
  case eTARGET_RESPONSE_LINEAR_Z:
    return "linear-z";
  case eTARGET_RESPONSE_NONE:
  default:
    return "none";
  }
}

static const char *getTargetResponseGateName(TargetResponseKind kind) {
  return kind == eTARGET_RESPONSE_NONE ? "NOT_APPLICABLE" : "GATED";
}

static const char *getAuthorityTargetResponseGateName(
    PxU32 expectedTargetResponses) {
  if (!usesProjectileExcitation())
    return "NOT_COVERED";
  return expectedTargetResponses ? "GATED" : "NOT_APPLICABLE";
}

static const char *getGateJointName() {
  if (isRevoluteMotorFamilyCase())
    return "revolute";
  if (isSphericalConeCase())
    return "spherical";
  if (isNativeBreakReactionCase())
    return getJointKindName(gImpactJointKind);
  if (gHeadlessCase == eCASE_IMPACT_SINGLE)
    return getJointKindName(gImpactJointKind);
  if (gHeadlessCase == eCASE_FIXED_NO_BREAK ||
      gHeadlessCase == eCASE_FIXED_BREAK)
    return "fixed";
  if (isForceReactionCase())
    return "fixed";
  return "all";
}

static void resetRuntimeState() {
  gChains.clear();
  gProjectiles.clear();
  gGateStats = JointGateStats();
  gRevoluteChainBodies.clear();
  gRevoluteChainJoints.clear();
  gRevoluteRestPositions.clear();
  gFixedChainJoints.clear();
  gBreakableJoints.clear();
  gBreakPollReported.clear();
  gBreakCallbackReported.clear();
  gNativeBreakReactionStats = NativeBreakReactionStats();
  gNativeBreakReactionBody = NULL;
  gNativeBreakReactionJoint = NULL;
  gSphericalConeStats = SphericalConeStats();
  gSphericalConeActorA = NULL;
  gSphericalConeDynamicA = NULL;
  gSphericalConeActorB = NULL;
  gSphericalConeJoint = NULL;
  gPrismaticChainBodies.clear();
  gPrismaticChainJoints.clear();
  gPrismaticRestPositions.clear();
  gRevoluteStats = RevoluteJitterStats();
  gFixedStats = FixedChainStats();
  gForceStaticStats = ForceStaticStats();
  gForceStaticBody = NULL;
  gForcePairBody1 = NULL;
  gForceStaticJoint = NULL;
  gForceStaticInitialPose = PxTransform(PxIdentity);
  gForcePairBody1InitialPose = PxTransform(PxIdentity);
  gRevoluteMotorStats = RevoluteMotorStats();
  gRevoluteMotorBodyA = NULL;
  gRevoluteMotorBodyB = NULL;
  gRevoluteMotorJoint = NULL;
  gRevoluteMotorLimitStats = RevoluteMotorLimitStats();
  gRevoluteMotorLimitBodyA = NULL;
  gRevoluteMotorLimitBody = NULL;
  gRevoluteMotorLimitJoint = NULL;
  gRevoluteMotorFreeSpinStats = RevoluteMotorFreeSpinStats();
  gRevoluteMotorFreeSpinBodyA = NULL;
  gRevoluteMotorFreeSpinBody = NULL;
  gRevoluteMotorFreeSpinJoint = NULL;
  gRevoluteMotorRatioStats = RevoluteMotorRatioStats();
  gRevoluteMotorRatioBodyA = NULL;
  gRevoluteMotorRatioBodyB = NULL;
  gRevoluteMotorRatioJoint = NULL;
  gRevoluteMotorRatioConfiguredAnchorA = PxVec3(0.0f);
  gRevoluteMotorRatioConfiguredAnchorB = PxVec3(0.0f);
  gRevoluteMotorContactStats = RevoluteMotorContactStats();
  gRevoluteMotorContactBodyA = NULL;
  gRevoluteMotorContactBodyB = NULL;
  gRevoluteMotorContactGround = NULL;
  gRevoluteMotorContactJoint = NULL;
  gRevoluteMotorKinematicStats =
      RevoluteMotorKinematicStats();
  gRevoluteMotorKinematicBody = NULL;
  gRevoluteMotorKinematicDynamicBody = NULL;
  gRevoluteMotorKinematicJoint = NULL;
  gRevoluteMotorOffPrincipalStats =
      RevoluteMotorOffPrincipalStats();
  gRevoluteMotorOffPrincipalBody = NULL;
  gRevoluteMotorOffPrincipalJoint = NULL;
  gRevoluteMotorOffCenterStats =
      RevoluteMotorOffCenterStats();
  gRevoluteMotorOffCenterBody = NULL;
  gRevoluteMotorOffCenterJoint = NULL;
  gRevoluteMotorOffCenterConfiguredLocalAnchor =
      gRevoluteMotorOffCenterLocalAnchor;
  gEndpointStats = EndpointProbeStats();
  gEndpointAngularStats = EndpointAngularProbeStats();
  gEndpointTarget = NULL;
  gEndpointJoint = NULL;
  gEndpointRevoluteJoint = NULL;
  gPrismaticStats = PrismaticDriftStats();
}

enum GateFilterKind {
  eFILTER_UNTAGGED = 0,
  eFILTER_CHAIN_BODY = 1,
  eFILTER_PROJECTILE = 2,
  eFILTER_MOTOR_CONTACT_BODY = 3,
  eFILTER_MOTOR_CONTACT_GROUND = 4
};

static bool findChainBody(const PxActor *actor, PxU32 &chainIndex,
                          PxU32 &bodyIndex) {
  for (PxU32 chain = 0; chain < gChains.size(); ++chain) {
    for (PxU32 body = 0; body < gChains[chain].bodies.size(); ++body) {
      if (gChains[chain].bodies[body] == actor) {
        chainIndex = chain;
        bodyIndex = body;
        return true;
      }
    }
  }
  return false;
}

static PxI32 findProjectile(const PxActor *actor) {
  for (PxU32 i = 0; i < gProjectiles.size(); ++i) {
    if (gProjectiles[i].actor == actor)
      return static_cast<PxI32>(i);
  }
  return -1;
}

static void registerBreakableJoint(PxJoint *joint) {
  if (!joint)
    return;
  gBreakableJoints.push_back(joint);
  gBreakPollReported.push_back(0);
  gBreakCallbackReported.push_back(0);
}

static PxI32 findBreakableJoint(const PxConstraint *constraint) {
  for (PxU32 jointIndex = 0; jointIndex < gBreakableJoints.size();
       ++jointIndex) {
    if (gBreakableJoints[jointIndex] &&
        gBreakableJoints[jointIndex]->getConstraint() == constraint)
      return static_cast<PxI32>(jointIndex);
  }
  return -1;
}

class JointGateSimulationCallback : public PxSimulationEventCallback {
public:
  virtual void onConstraintBreak(PxConstraintInfo *constraints, PxU32 count)
      PX_OVERRIDE {
    for (PxU32 infoIndex = 0; infoIndex < count; ++infoIndex) {
      gGateStats.breakCallbackCount++;
      const PxConstraintInfo &info = constraints[infoIndex];
      const PxI32 breakableIndex = findBreakableJoint(info.constraint);
      const bool constraintMatches = breakableIndex >= 0;
      const bool externalReferenceMatches =
          constraintMatches &&
          info.externalReference == static_cast<PxJoint *>(
                                        gBreakableJoints[
                                            PxU32(breakableIndex)]);
      const bool typeMatches =
          info.type == static_cast<PxU32>(PxConstraintExtIDs::eJOINT);
      const bool brokenFlagMatches =
          info.constraint &&
          info.constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
      const bool duplicate =
          constraintMatches && PxU32(breakableIndex) <
                                   gBreakCallbackReported.size() &&
          gBreakCallbackReported[PxU32(breakableIndex)] != 0;

      if (!constraintMatches)
        gGateStats.breakCallbackConstraintMismatches++;
      if (!externalReferenceMatches)
        gGateStats.breakCallbackExternalReferenceMismatches++;
      if (!typeMatches)
        gGateStats.breakCallbackTypeMismatches++;
      if (!brokenFlagMatches)
        gGateStats.breakCallbackBrokenFlagMismatches++;
      if (duplicate)
        gGateStats.breakCallbackDuplicateMismatches++;

      if (constraintMatches && externalReferenceMatches && typeMatches &&
          brokenFlagMatches && !duplicate) {
        gBreakCallbackReported[PxU32(breakableIndex)] = 1;
        gGateStats.breakCallbackIdentityMatches++;
      }
    }
  }

  virtual void onWake(PxActor **, PxU32) PX_OVERRIDE {}
  virtual void onSleep(PxActor **, PxU32) PX_OVERRIDE {}
  virtual void onTrigger(PxTriggerPair *, PxU32) PX_OVERRIDE {}
  virtual void onAdvance(const PxRigidBody *const *, const PxTransform *,
                         const PxU32) PX_OVERRIDE {}

  virtual void onContact(const PxContactPairHeader &pairHeader,
                         const PxContactPair *pairs,
                         PxU32 pairCount) PX_OVERRIDE {
    const PxActor *actor0 = pairHeader.actors[0];
    const PxActor *actor1 = pairHeader.actors[1];
    const bool actor0IsMotorBody =
        actor0 == gRevoluteMotorContactBodyA ||
        actor0 == gRevoluteMotorContactBodyB;
    const bool actor1IsMotorBody =
        actor1 == gRevoluteMotorContactBodyA ||
        actor1 == gRevoluteMotorContactBodyB;
    const bool motorContactPair =
        isRevoluteMotorContactCase() &&
        ((actor0IsMotorBody &&
          actor1 == gRevoluteMotorContactGround) ||
         (actor1IsMotorBody &&
          actor0 == gRevoluteMotorContactGround));
    if (motorContactPair) {
      RevoluteMotorContactStats &stats =
          gRevoluteMotorContactStats;
      stats.contactEvents += pairCount;
      for (PxU32 i = 0; i < pairCount; ++i) {
        std::vector<PxContactPairPoint> points(pairs[i].contactCount);
        const PxU32 extracted =
            pairs[i].extractContacts(points.data(),
                                     pairs[i].contactCount);
        stats.contactPointCount += extracted;
        for (PxU32 pointIndex = 0; pointIndex < extracted;
             ++pointIndex) {
          const PxContactPairPoint &point = points[pointIndex];
          if (!point.impulse.isFinite() || !point.normal.isFinite()) {
            stats.nonFiniteSamples++;
            continue;
          }
          const PxReal normalImpulse =
              PxAbs(point.impulse.dot(point.normal));
          const PxVec3 tangentImpulse =
              point.impulse - point.normal * point.impulse.dot(point.normal);
          const PxReal tangentialImpulse = tangentImpulse.magnitude();
          stats.totalNormalImpulse += normalImpulse;
          stats.totalTangentialImpulse += tangentialImpulse;
          stats.maximumTangentialImpulse =
              PxMax(stats.maximumTangentialImpulse,
                    tangentialImpulse);
        }
      }
    }

    PxI32 projectileIndex = findProjectile(pairHeader.actors[0]);
    const PxActor *chainActor = pairHeader.actors[1];
    if (projectileIndex < 0) {
      projectileIndex = findProjectile(pairHeader.actors[1]);
      chainActor = pairHeader.actors[0];
    }
    if (projectileIndex < 0)
      return;

    PxU32 chainIndex = PX_MAX_U32;
    PxU32 bodyIndex = PX_MAX_U32;
    if (!findChainBody(chainActor, chainIndex, bodyIndex))
      return;

    ProjectileRecord &projectile = gProjectiles[PxU32(projectileIndex)];
    if (projectile.firstChainContact == PX_MAX_U32) {
      projectile.firstChainContact = chainIndex;
      projectile.firstChainBody = bodyIndex;
    }

    if (chainIndex != projectile.targetChain) {
      projectile.crossChainContacts++;
      if (projectile.firstWrongContactFrame == PX_MAX_U32)
        projectile.firstWrongContactFrame = gGateStats.completedFrames + 1;
      return;
    }

    if (gChains[chainIndex].targetBody != chainActor) {
      projectile.wrongBodyContacts++;
      if (projectile.firstWrongContactFrame == PX_MAX_U32)
        projectile.firstWrongContactFrame = gGateStats.completedFrames + 1;
      return;
    }

    projectile.contactCount++;
    if (projectile.firstContactFrame == PX_MAX_U32)
      projectile.firstContactFrame = gGateStats.completedFrames + 1;
    for (PxU32 i = 0; i < pairCount; ++i) {
      std::vector<PxContactPairPoint> points(pairs[i].contactCount);
      const PxU32 extracted =
          pairs[i].extractContacts(points.data(), pairs[i].contactCount);
      projectile.contactPointCount += extracted;
      for (PxU32 pointIndex = 0; pointIndex < extracted; ++pointIndex) {
        const PxReal impulse = points[pointIndex].impulse.magnitude();
        if (!PxIsFinite(impulse)) {
          gGateStats.nonFinite++;
          continue;
        }
        const double accumulated =
            double(projectile.totalContactImpulse) + double(impulse);
        projectile.totalContactImpulse =
            accumulated >= double(PX_MAX_F32) ? PX_MAX_F32
                                               : PxReal(accumulated);
        projectile.peakContactImpulse =
            PxMax(projectile.peakContactImpulse, impulse);
      }
    }
  }
};

static JointGateSimulationCallback gSimulationCallback;

static PxFilterFlags jointGateFilterShader(
    PxFilterObjectAttributes attributes0, PxFilterData filterData0,
    PxFilterObjectAttributes attributes1, PxFilterData filterData1,
    PxPairFlags &pairFlags, const void *, PxU32) {
  if (PxFilterObjectIsTrigger(attributes0) ||
      PxFilterObjectIsTrigger(attributes1)) {
    pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
    return PxFilterFlag::eDEFAULT;
  }

  pairFlags = PxPairFlag::eCONTACT_DEFAULT;
  const bool projectileAndChain =
      (filterData0.word0 == eFILTER_PROJECTILE &&
       filterData1.word0 == eFILTER_CHAIN_BODY) ||
      (filterData1.word0 == eFILTER_PROJECTILE &&
       filterData0.word0 == eFILTER_CHAIN_BODY);
  if (projectileAndChain)
    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
  const bool motorBodyAndGround =
      (filterData0.word0 == eFILTER_MOTOR_CONTACT_BODY &&
       filterData1.word0 == eFILTER_MOTOR_CONTACT_GROUND) ||
      (filterData1.word0 == eFILTER_MOTOR_CONTACT_BODY &&
       filterData0.word0 == eFILTER_MOTOR_CONTACT_GROUND);
  if (motorBodyAndGround)
    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
  return PxFilterFlag::eDEFAULT;
}

static void getJointWorldAxes(PxRevoluteJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1) {
  axis0 = PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = PxVec3(1.0f, 0.0f, 0.0f);
  if (!joint)
    return;

  PxRigidActor *a0 = nullptr;
  PxRigidActor *a1 = nullptr;
  joint->getActors(a0, a1);

  const PxTransform lp0 = joint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform lp1 = joint->getLocalPose(PxJointActorIndex::eACTOR1);
  const PxVec3 localAxis(1.0f, 0.0f, 0.0f);

  axis0 = a0 ? a0->getGlobalPose().q.rotate(lp0.q.rotate(localAxis))
             : lp0.q.rotate(localAxis);
  axis1 = a1 ? a1->getGlobalPose().q.rotate(lp1.q.rotate(localAxis))
             : lp1.q.rotate(localAxis);

  const PxReal l0 = axis0.magnitudeSquared();
  const PxReal l1 = axis1.magnitudeSquared();
  axis0 = (l0 > 1e-12f) ? axis0 * PxRecipSqrt(l0) : PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = (l1 > 1e-12f) ? axis1 * PxRecipSqrt(l1) : PxVec3(1.0f, 0.0f, 0.0f);
}

static void getJointWorldAxes(PxPrismaticJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1) {
  axis0 = PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = PxVec3(1.0f, 0.0f, 0.0f);
  if (!joint)
    return;

  PxRigidActor *a0 = nullptr;
  PxRigidActor *a1 = nullptr;
  joint->getActors(a0, a1);

  const PxTransform lp0 = joint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform lp1 = joint->getLocalPose(PxJointActorIndex::eACTOR1);
  const PxVec3 localAxis(1.0f, 0.0f, 0.0f);

  axis0 = a0 ? a0->getGlobalPose().q.rotate(lp0.q.rotate(localAxis))
             : lp0.q.rotate(localAxis);
  axis1 = a1 ? a1->getGlobalPose().q.rotate(lp1.q.rotate(localAxis))
             : lp1.q.rotate(localAxis);

  const PxReal l0 = axis0.magnitudeSquared();
  const PxReal l1 = axis1.magnitudeSquared();
  axis0 = (l0 > 1e-12f) ? axis0 * PxRecipSqrt(l0) : PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = (l1 > 1e-12f) ? axis1 * PxRecipSqrt(l1) : PxVec3(1.0f, 0.0f, 0.0f);
}

static PxVec3 getJointWorldAxis(PxRevoluteJoint *joint) {
  if (!joint)
    return PxVec3(1.0f, 0.0f, 0.0f);

  PxVec3 axis0, axis1;
  getJointWorldAxes(joint, axis0, axis1);

  PxVec3 axis = axis0 + axis1;
  const PxReal len2 = axis.magnitudeSquared();
  if (len2 > 1e-12f)
    axis *= PxRecipSqrt(len2);
  else
    axis = (axis0.magnitudeSquared() > 1e-12f) ? axis0.getNormalized()
                                               : PxVec3(1.0f, 0.0f, 0.0f);
  return axis;
}

static PxRigidDynamic *createDynamic(const PxTransform &t,
                                     const PxGeometry &geometry,
                                     const PxVec3 &velocity = PxVec3(0)) {
  PxRigidDynamic *dynamic =
      PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
  if (!dynamic)
    return NULL;
  dynamic->setAngularDamping(0.5f);
  dynamic->setLinearVelocity(velocity);
  gScene->addActor(*dynamic);
  return dynamic;
}

// revolute joint limited to an angle range of ±45 degrees
static PxJoint *createLimitedRevolute(PxRigidActor *a0, const PxTransform &t0,
                                      PxRigidActor *a1,
                                      const PxTransform &t1) {
  PxRevoluteJoint *j = PxRevoluteJointCreate(*gPhysics, a0, t0, a1, t1);
  if (!j)
    return NULL;
  j->setLimit(PxJointAngularLimitPair(-PxPi / 4, PxPi / 4));
  j->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);
  return j;
}

// spherical joint limited to an angle of at most pi/4 radians (45 degrees)
static PxJoint *createLimitedSpherical(PxRigidActor *a0, const PxTransform &t0,
                                       PxRigidActor *a1,
                                       const PxTransform &t1) {
  PxSphericalJoint *j = PxSphericalJointCreate(*gPhysics, a0, t0, a1, t1);
  if (!j)
    return NULL;
  j->setLimitCone(PxJointLimitCone(PxPi / 4, PxPi / 4));
  j->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
  return j;
}

// prismatic joint limited between -2 and 2
static PxJoint *createLimitedPrismatic(PxRigidActor *a0, const PxTransform &t0,
                                       PxRigidActor *a1,
                                       const PxTransform &t1) {
  PxPrismaticJoint *j = PxPrismaticJointCreate(*gPhysics, a0, t0, a1, t1);
  if (!j)
    return NULL;
  j->setLimit(PxJointLinearLimitPair(-2.0f, 2.0f, PxSpring(0, 0)));
  j->setPrismaticJointFlag(PxPrismaticJointFlag::eLIMIT_ENABLED, true);
  return j;
}

// fixed, breakable joint
static PxJoint *createBreakableFixed(PxRigidActor *a0, const PxTransform &t0,
                                     PxRigidActor *a1, const PxTransform &t1) {
  PxFixedJoint *j = PxFixedJointCreate(*gPhysics, a0, t0, a1, t1);
  if (!j)
    return NULL;
  j->setBreakForce(gFixedLinearBreakForce, gFixedAngularBreakForce);
  j->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
  j->setConstraintFlag(PxConstraintFlag::eDISABLE_PREPROCESSING, true);
  return j;
}

// D6 joint with a spring maintaining its position
static PxJoint *createDampedD6(PxRigidActor *a0, const PxTransform &t0,
                               PxRigidActor *a1, const PxTransform &t1) {
  PxD6Joint *j = PxD6JointCreate(*gPhysics, a0, t0, a1, t1);
  if (!j)
    return NULL;
  j->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
  j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  j->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
  j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(0, 1000, FLT_MAX, true));
  return j;
}

typedef PxJoint *(*JointCreateFunction)(PxRigidActor *a0, const PxTransform &t0,
                                        PxRigidActor *a1,
                                        const PxTransform &t1);

// create a chain rooted at the origin and extending along the x-axis, all
// transformed by the argument t.

static void createChain(const PxTransform &t, PxU32 length, const PxGeometry &g,
                        PxReal separation, JointCreateFunction createJoint,
                        JointKind kind, const char *name) {
  const PxU32 chainIndex = static_cast<PxU32>(gChains.size());
  gChains.push_back(ChainRecord(kind, name));
  ChainRecord &chain = gChains.back();
  PxVec3 offset(separation / 2, 0, 0);
  PxTransform localTm(offset);
  PxRigidDynamic *prev = NULL;

  for (PxU32 i = 0; i < length; i++) {
    PxRigidDynamic *current =
        PxCreateDynamic(*gPhysics, t * localTm, g, *gMaterial, 1.0f);
    if (!current) {
      gInitializationFailed = true;
      return;
    }
    PxJoint *joint = (*createJoint)(prev, prev ? PxTransform(offset) : t,
                                    current, PxTransform(-offset));
    if (!joint) {
      current->release();
      gInitializationFailed = true;
      return;
    }

    PxShape *shape = NULL;
    if (current->getShapes(&shape, 1) == 1 && shape) {
      PxFilterData filterData;
      filterData.word0 = eFILTER_CHAIN_BODY;
      filterData.word1 = chainIndex;
      filterData.word2 = i;
      shape->setSimulationFilterData(filterData);
    }

    chain.bodies.push_back(current);
    chain.joints.push_back(joint);
    if (i == length / 2)
      chain.targetBody = current;

    if (kind == eJOINT_REVOLUTE) {
      gRevoluteChainBodies.push_back(current);
      gRevoluteRestPositions.push_back((t * localTm).p);
      if (joint) {
        PxRevoluteJoint *revolute = joint->is<PxRevoluteJoint>();
        if (revolute)
          gRevoluteChainJoints.push_back(revolute);
      }
    } else if (kind == eJOINT_FIXED) {
      if (joint) {
        PxFixedJoint *fixed = joint->is<PxFixedJoint>();
          if (fixed) {
            gFixedChainJoints.push_back(fixed);
            registerBreakableJoint(fixed);
          }
      }
    } else if (kind == eJOINT_PRISMATIC) {
      gPrismaticChainBodies.push_back(current);
      gPrismaticRestPositions.push_back((t * localTm).p);
      if (joint) {
        PxPrismaticJoint *prismatic = joint->is<PxPrismaticJoint>();
        if (prismatic)
          gPrismaticChainJoints.push_back(prismatic);
      }
    }

    gScene->addActor(*current);
    prev = current;
    localTm.p.x += separation;
  }
}

static void createIsolatedFixedTarget() {
  const PxU32 chainIndex = static_cast<PxU32>(gChains.size());
  gChains.push_back(ChainRecord(eJOINT_FIXED, "fixed"));
  ChainRecord &chain = gChains.back();
  const PxTransform targetPose(PxVec3(0.0f, 10.0f, 0.0f));
  PxRigidDynamic *target = PxCreateDynamic(
      *gPhysics, targetPose, PxBoxGeometry(2.0f, 0.5f, 0.5f), *gMaterial,
      1.0f);
  if (!target) {
    gInitializationFailed = true;
    return;
  }

  PxShape *shape = NULL;
  if (target->getShapes(&shape, 1) == 1 && shape) {
    PxFilterData filterData;
    filterData.word0 = eFILTER_CHAIN_BODY;
    filterData.word1 = chainIndex;
    filterData.word2 = 0;
    shape->setSimulationFilterData(filterData);
  }

  PxJoint *created =
      createBreakableFixed(NULL, targetPose, target, PxTransform(PxIdentity));
  PxFixedJoint *joint = created ? created->is<PxFixedJoint>() : NULL;
  if (!joint) {
    target->release();
    gInitializationFailed = true;
    return;
  }

  chain.bodies.push_back(target);
  chain.joints.push_back(joint);
  chain.targetBody = target;
  gFixedChainJoints.push_back(joint);
  registerBreakableJoint(joint);
  gScene->addActor(*target);
}

static void createForceStaticTarget() {
  const PxU32 frequency =
      static_cast<PxU32>(1.0f / gHeadlessOptions.dt + 0.5f);
  gForceStaticStats.steadyBeginFrame =
      frequency * gForceStaticWarmupSeconds;
  gForceStaticStats.expectedSteadySamples =
      gHeadlessOptions.frames - gForceStaticStats.steadyBeginFrame;
  gForceStaticInitialPose = PxTransform(PxVec3(0.0f, 10.0f, 0.0f));
  gForceStaticBody = PxCreateDynamic(
      *gPhysics, gForceStaticInitialPose,
      PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
  if (!gForceStaticBody ||
      !PxRigidBodyExt::setMassAndUpdateInertia(*gForceStaticBody,
                                               gForceStaticMass)) {
    PX_RELEASE(gForceStaticBody);
    gInitializationFailed = true;
    return;
  }
  gForceStaticBody->setLinearDamping(0.0f);
  gForceStaticBody->setAngularDamping(0.0f);
  gScene->addActor(*gForceStaticBody);

  const PxTransform bodyLocalFrame(
      isForceOffsetCase() ? gForceOffsetAnchor : PxVec3(0.0f));
  const PxTransform worldFrame = gForceStaticInitialPose * bodyLocalFrame;
  gForceStaticJoint = PxFixedJointCreate(
      *gPhysics, NULL, worldFrame, gForceStaticBody, bodyLocalFrame);
  if (!gForceStaticJoint) {
    gInitializationFailed = true;
    return;
  }
  gForceStaticJoint->setBreakForce(PX_MAX_F32, PX_MAX_F32);

  gForceStaticStats.actualMass = gForceStaticBody->getMass();
  gForceStaticStats.gravity = gScene->getGravity();
  gForceStaticStats.expectedWeight =
      gForceStaticStats.actualMass * gForceStaticStats.gravity.magnitude();
  const PxTransform actor0Frame =
      gForceStaticJoint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform actor1Frame =
      gForceStaticJoint->getLocalPose(PxJointActorIndex::eACTOR1);
  gForceStaticStats.anchorOffset = actor1Frame.p;
  gForceStaticStats.actor0FramePositionError =
      (actor0Frame.p - worldFrame.p).magnitude();
  gForceStaticStats.actor1FramePositionError =
      (actor1Frame.p - bodyLocalFrame.p).magnitude();
  const PxVec3 actor0ExpectedForce =
      gForceStaticStats.actualMass * gForceStaticStats.gravity;
  gForceStaticStats.expectedLinearForce = actor0ExpectedForce;
  const PxVec3 anchorToCom =
      gForceStaticInitialPose.p - actor0Frame.p;
  gForceStaticStats.expectedTorque = anchorToCom.cross(actor0ExpectedForce);

  gForceStaticStats.topologyDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gForceStaticStats.topologyStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gForceStaticStats.topologyConstraints = gScene->getNbConstraints();
}

static void createForcePairTarget() {
  const PxU32 frequency =
      static_cast<PxU32>(1.0f / gHeadlessOptions.dt + 0.5f);
  gForceStaticStats.steadyBeginFrame =
      frequency * gForceStaticWarmupSeconds;
  gForceStaticStats.expectedSteadySamples =
      gHeadlessOptions.frames - gForceStaticStats.steadyBeginFrame;
  const PxTransform lowerPose(PxVec3(0.0f, 9.0f, 0.0f));
  const PxTransform upperPose(PxVec3(0.0f, 11.0f, 0.0f));
  const PxTransform jointWorldFrame(PxVec3(0.0f, 10.0f, 0.0f));
  const PxTransform lowerLocalFrame = lowerPose.getInverse() * jointWorldFrame;
  const PxTransform upperLocalFrame = upperPose.getInverse() * jointWorldFrame;

  PxRigidDynamic *lower = PxCreateDynamic(
      *gPhysics, lowerPose, PxBoxGeometry(0.25f, 0.25f, 0.25f), *gMaterial,
      1.0f);
  PxRigidDynamic *upper = PxCreateDynamic(
      *gPhysics, upperPose, PxBoxGeometry(0.25f, 0.25f, 0.25f), *gMaterial,
      1.0f);
  if (!lower || !upper ||
      !PxRigidBodyExt::setMassAndUpdateInertia(*lower, gForceStaticMass) ||
      !PxRigidBodyExt::setMassAndUpdateInertia(*upper, gForceStaticMass)) {
    PX_RELEASE(lower);
    PX_RELEASE(upper);
    gInitializationFailed = true;
    return;
  }
  lower->setLinearDamping(0.0f);
  lower->setAngularDamping(0.0f);
  upper->setLinearDamping(0.0f);
  upper->setAngularDamping(0.0f);
  gScene->addActor(*lower);
  gScene->addActor(*upper);

  const bool swapped =
      gForceActorOrder == eFORCE_ACTOR_ORDER_SWAPPED;
  gForceStaticBody = swapped ? upper : lower;
  gForcePairBody1 = swapped ? lower : upper;
  gForceStaticInitialPose = swapped ? upperPose : lowerPose;
  gForcePairBody1InitialPose = swapped ? lowerPose : upperPose;
  const PxTransform actor0LocalFrame =
      swapped ? upperLocalFrame : lowerLocalFrame;
  const PxTransform actor1LocalFrame =
      swapped ? lowerLocalFrame : upperLocalFrame;
  gForceStaticJoint = PxFixedJointCreate(
      *gPhysics, gForceStaticBody, actor0LocalFrame, gForcePairBody1,
      actor1LocalFrame);
  if (!gForceStaticJoint) {
    gInitializationFailed = true;
    return;
  }
  gForceStaticJoint->setBreakForce(PX_MAX_F32, PX_MAX_F32);
  if (isForcePairDisabledCase()) {
    gForceStaticJoint->setConstraintFlag(
        PxConstraintFlag::eDISABLE_CONSTRAINT, true);
    PxConstraint *constraint = gForceStaticJoint->getConstraint();
    if (!constraint ||
        !constraint->getFlags().isSet(
            PxConstraintFlag::eDISABLE_CONSTRAINT)) {
      gInitializationFailed = true;
      return;
    }
  }

  const PxVec3 lowerAppliedForce(
      0.0f,
      isForcePairDisabledCase() ? -gForcePairAppliedMagnitude
                                : gForcePairAppliedMagnitude,
      0.0f);
  const PxVec3 upperAppliedForce = -lowerAppliedForce;
  gForceStaticStats.appliedForceActor0 =
      swapped ? upperAppliedForce : lowerAppliedForce;
  gForceStaticStats.appliedForceActor1 =
      swapped ? lowerAppliedForce : upperAppliedForce;
  gForceStaticStats.expectedLinearForce =
      -gForceStaticStats.appliedForceActor0;
  gForceStaticStats.expectedTorque = PxVec3(0.0f);
  gForceStaticStats.actualMass = gForceStaticBody->getMass();
  gForceStaticStats.gravity = gScene->getGravity();
  gForceStaticStats.expectedWeight =
      gForceStaticStats.expectedLinearForce.magnitude();
  gForceStaticStats.anchorOffset = actor0LocalFrame.p;
  gForceStaticStats.actor0FramePositionError =
      (gForceStaticJoint->getLocalPose(PxJointActorIndex::eACTOR0).p -
       actor0LocalFrame.p)
          .magnitude();
  gForceStaticStats.actor1FramePositionError =
      (gForceStaticJoint->getLocalPose(PxJointActorIndex::eACTOR1).p -
       actor1LocalFrame.p)
          .magnitude();
  gForceStaticStats.pairInitialSeparation =
      (upperPose.p - lowerPose.p).magnitude();
  gForceStaticStats.pairInitialCenterOfMass =
      (gForceStaticInitialPose.p * gForceStaticBody->getMass() +
       gForcePairBody1InitialPose.p * gForcePairBody1->getMass()) /
      (gForceStaticBody->getMass() + gForcePairBody1->getMass());
  PxRigidActor *actualActor0 = NULL;
  PxRigidActor *actualActor1 = NULL;
  gForceStaticJoint->getActors(actualActor0, actualActor1);
  gForceStaticStats.pairActorOrderValid =
      actualActor0 == gForceStaticBody && actualActor1 == gForcePairBody1;
  gForceStaticStats.topologyDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gForceStaticStats.topologyStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gForceStaticStats.topologyConstraints = gScene->getNbConstraints();
}

static void createRevoluteMotorTarget() {
  const PxTransform poseA(PxVec3(-1.0f, 10.0f, 0.0f));
  const PxTransform poseB(PxVec3(1.0f, 10.0f, 0.0f));
  gRevoluteMotorBodyA = PxCreateDynamic(
      *gPhysics, poseA, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  gRevoluteMotorBodyB = PxCreateDynamic(
      *gPhysics, poseB, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  if (!gRevoluteMotorBodyA || !gRevoluteMotorBodyB) {
    PX_RELEASE(gRevoluteMotorBodyA);
    PX_RELEASE(gRevoluteMotorBodyB);
    gInitializationFailed = true;
    return;
  }

  gRevoluteMotorBodyA->setMassSpaceInertiaTensor(
      PxVec3(gRevoluteMotorInertiaA, 2.0f, 3.0f));
  gRevoluteMotorBodyB->setMassSpaceInertiaTensor(
      PxVec3(gRevoluteMotorInertiaB, 4.0f, 5.0f));
  gRevoluteMotorBodyA->setLinearDamping(0.0f);
  gRevoluteMotorBodyA->setAngularDamping(0.0f);
  gRevoluteMotorBodyB->setLinearDamping(0.0f);
  gRevoluteMotorBodyB->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorBodyA);
  gScene->addActor(*gRevoluteMotorBodyB);

  gRevoluteMotorJoint = PxRevoluteJointCreate(
      *gPhysics, gRevoluteMotorBodyA,
      PxTransform(PxVec3(1.0f, 0.0f, 0.0f)),
      gRevoluteMotorBodyB,
      PxTransform(PxVec3(-1.0f, 0.0f, 0.0f)));
  if (!gRevoluteMotorJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorJoint->setDriveVelocity(
      gRevoluteMotorTargetVelocity, false);
  gRevoluteMotorJoint->setDriveForceLimit(
      gRevoluteMotorForceLimit);
  gRevoluteMotorJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorJoint->getActors(actor0, actor1);
  gRevoluteMotorStats.actorOrderValid =
      actor0 == gRevoluteMotorBodyA &&
      actor1 == gRevoluteMotorBodyB;
  gRevoluteMotorStats.targetVelocityReadback =
      gRevoluteMotorJoint->getDriveVelocity();
  gRevoluteMotorStats.forceLimitReadback =
      gRevoluteMotorJoint->getDriveForceLimit();
  gRevoluteMotorStats.driveEnabledReadback =
      gRevoluteMotorJoint->getRevoluteJointFlags().isSet(
          PxRevoluteJointFlag::eDRIVE_ENABLED);
  gRevoluteMotorStats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gRevoluteMotorStats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gRevoluteMotorStats.initialConstraints =
      gScene->getNbConstraints();
}

static void createRevoluteMotorLimitTarget() {
  const PxTransform pose(PxVec3(0.0f, 10.0f, 0.0f));
  if (isRevoluteMotorDynamicLimitCase()) {
    gRevoluteMotorLimitBodyA = PxCreateDynamic(
        *gPhysics, pose, PxBoxGeometry(0.25f, 0.25f, 0.25f),
        *gMaterial, 1.0f);
  }
  gRevoluteMotorLimitBody = PxCreateDynamic(
      *gPhysics, pose, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  if (!gRevoluteMotorLimitBody ||
      (isRevoluteMotorDynamicLimitCase() &&
       !gRevoluteMotorLimitBodyA)) {
    PX_RELEASE(gRevoluteMotorLimitBodyA);
    PX_RELEASE(gRevoluteMotorLimitBody);
    gInitializationFailed = true;
    return;
  }

  if (gRevoluteMotorLimitBodyA) {
    gRevoluteMotorLimitBodyA->setMassSpaceInertiaTensor(
        PxVec3(1.0f, 2.0f, 3.0f));
    gRevoluteMotorLimitBodyA->setLinearDamping(0.0f);
    gRevoluteMotorLimitBodyA->setAngularDamping(0.0f);
    gScene->addActor(*gRevoluteMotorLimitBodyA);
  }
  gRevoluteMotorLimitBody->setMassSpaceInertiaTensor(
      isRevoluteMotorDynamicLimitCase()
          ? PxVec3(3.0f, 4.0f, 5.0f)
          : PxVec3(1.0f, 2.0f, 3.0f));
  gRevoluteMotorLimitBody->setLinearDamping(0.0f);
  gRevoluteMotorLimitBody->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorLimitBody);

  gRevoluteMotorLimitJoint = PxRevoluteJointCreate(
      *gPhysics, gRevoluteMotorLimitBodyA,
      gRevoluteMotorLimitBodyA ? PxTransform(PxIdentity) : pose,
      gRevoluteMotorLimitBody,
      PxTransform(PxIdentity));
  if (!gRevoluteMotorLimitJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorLimitJoint->setLimit(
      PxJointAngularLimitPair(gRevoluteMotorLimitLower,
                              gRevoluteMotorLimitUpper));
  gRevoluteMotorLimitJoint->setDriveVelocity(
      gRevoluteMotorLimitTargetVelocity, false);
  gRevoluteMotorLimitJoint->setDriveForceLimit(
      gRevoluteMotorLimitForceLimit);
  gRevoluteMotorLimitJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eLIMIT_ENABLED, true);
  gRevoluteMotorLimitJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorLimitJoint->getActors(actor0, actor1);
  const PxJointAngularLimitPair limit =
      gRevoluteMotorLimitJoint->getLimit();
  const PxRevoluteJointFlags flags =
      gRevoluteMotorLimitJoint->getRevoluteJointFlags();
  RevoluteMotorLimitStats &stats = gRevoluteMotorLimitStats;
  stats.actorOrderValid = isRevoluteMotorDynamicLimitCase()
                              ? actor0 == gRevoluteMotorLimitBodyA &&
                                    actor1 == gRevoluteMotorLimitBody
                              : actor0 == NULL &&
                                    actor1 == gRevoluteMotorLimitBody;
  stats.targetVelocityReadback =
      gRevoluteMotorLimitJoint->getDriveVelocity();
  stats.finalTargetVelocityReadback = stats.targetVelocityReadback;
  stats.forceLimitReadback =
      gRevoluteMotorLimitJoint->getDriveForceLimit();
  stats.lowerLimitReadback = limit.lower;
  stats.upperLimitReadback = limit.upper;
  stats.initialAngle = gRevoluteMotorLimitJoint->getAngle();
  stats.finalAngle = stats.initialAngle;
  stats.minimumAngle = stats.initialAngle;
  stats.maximumAngle = stats.initialAngle;
  stats.driveEnabledReadback =
      flags.isSet(PxRevoluteJointFlag::eDRIVE_ENABLED);
  stats.limitEnabledReadback =
      flags.isSet(PxRevoluteJointFlag::eLIMIT_ENABLED);
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static void createRevoluteMotorFreeSpinTarget() {
  const PxTransform pose(PxVec3(0.0f, 10.0f, 0.0f));
  if (isRevoluteMotorDynamicFreeSpinCase()) {
    gRevoluteMotorFreeSpinBodyA = PxCreateDynamic(
        *gPhysics, pose, PxBoxGeometry(0.25f, 0.25f, 0.25f),
        *gMaterial, 1.0f);
  }
  gRevoluteMotorFreeSpinBody = PxCreateDynamic(
      *gPhysics, pose, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  if (!gRevoluteMotorFreeSpinBody ||
      (isRevoluteMotorDynamicFreeSpinCase() &&
       !gRevoluteMotorFreeSpinBodyA)) {
    PX_RELEASE(gRevoluteMotorFreeSpinBodyA);
    PX_RELEASE(gRevoluteMotorFreeSpinBody);
    gInitializationFailed = true;
    return;
  }

  if (gRevoluteMotorFreeSpinBodyA) {
    gRevoluteMotorFreeSpinBodyA->setMassSpaceInertiaTensor(
        PxVec3(1.0f, 2.0f, 3.0f));
    gRevoluteMotorFreeSpinBodyA->setLinearDamping(0.0f);
    gRevoluteMotorFreeSpinBodyA->setAngularDamping(0.0f);
    gScene->addActor(*gRevoluteMotorFreeSpinBodyA);
  }
  gRevoluteMotorFreeSpinBody->setMassSpaceInertiaTensor(
      isRevoluteMotorDynamicFreeSpinCase()
          ? PxVec3(3.0f, 4.0f, 5.0f)
          : PxVec3(1.0f, 2.0f, 3.0f));
  gRevoluteMotorFreeSpinBody->setLinearDamping(0.0f);
  gRevoluteMotorFreeSpinBody->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorFreeSpinBody);

  gRevoluteMotorFreeSpinJoint = PxRevoluteJointCreate(
      *gPhysics, gRevoluteMotorFreeSpinBodyA,
      gRevoluteMotorFreeSpinBodyA
          ? PxTransform(PxIdentity)
          : pose,
      gRevoluteMotorFreeSpinBody,
      PxTransform(PxIdentity));
  if (!gRevoluteMotorFreeSpinJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorFreeSpinJoint->setDriveVelocity(
      gRevoluteMotorFreeSpinTargetVelocity, false);
  gRevoluteMotorFreeSpinJoint->setDriveForceLimit(
      gRevoluteMotorFreeSpinForceLimit);
  gRevoluteMotorFreeSpinJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);
  gRevoluteMotorFreeSpinJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_FREESPIN, true);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorFreeSpinJoint->getActors(actor0, actor1);
  const PxRevoluteJointFlags flags =
      gRevoluteMotorFreeSpinJoint->getRevoluteJointFlags();
  RevoluteMotorFreeSpinStats &stats = gRevoluteMotorFreeSpinStats;
  stats.actorOrderValid =
      isRevoluteMotorDynamicFreeSpinCase()
          ? actor0 == gRevoluteMotorFreeSpinBodyA &&
                actor1 == gRevoluteMotorFreeSpinBody
          : actor0 == NULL &&
                actor1 == gRevoluteMotorFreeSpinBody;
  stats.targetVelocityReadback =
      gRevoluteMotorFreeSpinJoint->getDriveVelocity();
  stats.forceLimitReadback =
      gRevoluteMotorFreeSpinJoint->getDriveForceLimit();
  stats.driveEnabledReadback =
      flags.isSet(PxRevoluteJointFlag::eDRIVE_ENABLED);
  stats.freeSpinEnabledReadback =
      flags.isSet(PxRevoluteJointFlag::eDRIVE_FREESPIN);
  stats.limitDisabledReadback =
      !flags.isSet(PxRevoluteJointFlag::eLIMIT_ENABLED);
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static void createRevoluteMotorRatioTarget() {
  const bool dynamicOffPrincipal =
      isRevoluteMotorDynamicOffPrincipalCase();
  const bool dynamicOffCenter =
      isRevoluteMotorDynamicOffCenterCase();
  const bool dynamicSpatial =
      isRevoluteMotorDynamicSpatialCase();
  const PxTransform poseA =
      dynamicSpatial
          ? PxTransform(
                PxVec3(0.0f, 9.0f, 0.0f),
                PxQuat(PxPi / 6.0f, PxVec3(0.0f, 0.0f, 1.0f)))
          : (dynamicOffPrincipal
          ? PxTransform(
                PxVec3(0.0f, 10.0f, 0.0f),
                PxQuat(PxPi / 6.0f, PxVec3(0.0f, 0.0f, 1.0f)))
          : (dynamicOffCenter
                 ? PxTransform(PxVec3(0.0f, 9.0f, 0.0f))
                 : PxTransform(PxVec3(-1.0f, 10.0f, 0.0f))));
  const PxTransform poseB =
      dynamicSpatial
          ? PxTransform(
                PxVec3(0.0f, 11.0f, 0.0f),
                PxQuat(-PxPi / 5.0f, PxVec3(0.0f, 0.0f, 1.0f)))
          : (dynamicOffPrincipal
          ? PxTransform(
                PxVec3(0.0f, 10.0f, 0.0f),
                PxQuat(-PxPi / 5.0f, PxVec3(0.0f, 0.0f, 1.0f)))
          : (dynamicOffCenter
                 ? PxTransform(PxVec3(0.0f, 11.0f, 0.0f))
                 : PxTransform(PxVec3(1.0f, 10.0f, 0.0f))));
  const PxTransform worldFrame(PxVec3(0.0f, 10.0f, 0.0f));
  const PxVec3 inertiaA =
      dynamicOffPrincipal
          ? gRevoluteMotorDynamicOffPrincipalInertiaA
          : PxVec3(gRevoluteMotorRatioInertiaA, 2.0f, 3.0f);
  const PxVec3 inertiaB =
      dynamicOffPrincipal
          ? gRevoluteMotorDynamicOffPrincipalInertiaB
          : PxVec3(gRevoluteMotorRatioInertiaB, 4.0f, 5.0f);
  const PxReal driveGearRatio =
      (dynamicOffPrincipal || dynamicOffCenter)
          ? 1.0f
          : gRevoluteMotorRatioDriveGearRatio;
  gRevoluteMotorRatioConfiguredAnchorA =
      poseA.transformInv(worldFrame).p;
  gRevoluteMotorRatioConfiguredAnchorB =
      poseB.transformInv(worldFrame).p;
  gRevoluteMotorRatioBodyA = PxCreateDynamic(
      *gPhysics, poseA, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  gRevoluteMotorRatioBodyB = PxCreateDynamic(
      *gPhysics, poseB, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  if (!gRevoluteMotorRatioBodyA || !gRevoluteMotorRatioBodyB) {
    PX_RELEASE(gRevoluteMotorRatioBodyA);
    PX_RELEASE(gRevoluteMotorRatioBodyB);
    gInitializationFailed = true;
    return;
  }

  gRevoluteMotorRatioBodyA->setMassSpaceInertiaTensor(
      inertiaA);
  gRevoluteMotorRatioBodyB->setMassSpaceInertiaTensor(
      inertiaB);
  if (dynamicOffCenter) {
    gRevoluteMotorRatioBodyA->setMass(1.0f);
    gRevoluteMotorRatioBodyB->setMass(1.0f);
  }
  gRevoluteMotorRatioBodyA->setLinearDamping(0.0f);
  gRevoluteMotorRatioBodyA->setAngularDamping(0.0f);
  gRevoluteMotorRatioBodyB->setLinearDamping(0.0f);
  gRevoluteMotorRatioBodyB->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorRatioBodyA);
  gScene->addActor(*gRevoluteMotorRatioBodyB);

  gRevoluteMotorRatioJoint = PxRevoluteJointCreate(
      *gPhysics, gRevoluteMotorRatioBodyA,
      poseA.transformInv(worldFrame),
      gRevoluteMotorRatioBodyB,
      poseB.transformInv(worldFrame));
  if (!gRevoluteMotorRatioJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorRatioJoint->setDriveVelocity(
      gRevoluteMotorRatioTargetVelocity, false);
  gRevoluteMotorRatioJoint->setDriveForceLimit(
      gRevoluteMotorRatioForceLimit);
  gRevoluteMotorRatioJoint->setDriveGearRatio(
      driveGearRatio);
  gRevoluteMotorRatioJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorRatioJoint->getActors(actor0, actor1);
  const PxRevoluteJointFlags flags =
      gRevoluteMotorRatioJoint->getRevoluteJointFlags();
  RevoluteMotorRatioStats &stats = gRevoluteMotorRatioStats;
  stats.actorOrderValid =
      actor0 == gRevoluteMotorRatioBodyA &&
      actor1 == gRevoluteMotorRatioBodyB;
  stats.targetVelocityReadback =
      gRevoluteMotorRatioJoint->getDriveVelocity();
  stats.forceLimitReadback =
      gRevoluteMotorRatioJoint->getDriveForceLimit();
  stats.driveGearRatioReadback =
      gRevoluteMotorRatioJoint->getDriveGearRatio();
  stats.driveEnabledReadback =
      flags.isSet(PxRevoluteJointFlag::eDRIVE_ENABLED);
  stats.freeSpinDisabledReadback =
      !flags.isSet(PxRevoluteJointFlag::eDRIVE_FREESPIN);
  if (dynamicOffPrincipal) {
    const PxVec3 worldAxis(1.0f, 0.0f, 0.0f);
    const PxVec3 localAxisA = poseA.q.rotateInv(worldAxis);
    const PxVec3 localAxisB = poseB.q.rotateInv(worldAxis);
    const PxVec3 worldResponseA = poseA.q.rotate(PxVec3(
        localAxisA.x / inertiaA.x, localAxisA.y / inertiaA.y,
        localAxisA.z / inertiaA.z));
    const PxVec3 worldResponseB = poseB.q.rotate(PxVec3(
        localAxisB.x / inertiaB.x, localAxisB.y / inertiaB.y,
        localAxisB.z / inertiaB.z));
    stats.initialOffPrincipalResponseA =
        (worldResponseA -
         worldAxis * worldResponseA.dot(worldAxis))
            .magnitude();
    stats.initialOffPrincipalResponseB =
        (worldResponseB -
         worldAxis * worldResponseB.dot(worldAxis))
            .magnitude();
  }
  if (dynamicOffCenter) {
    const PxVec3 worldAxis(1.0f, 0.0f, 0.0f);
    const PxVec3 leverA =
        poseA.q.rotate(gRevoluteMotorRatioConfiguredAnchorA);
    const PxVec3 leverB =
        poseB.q.rotate(gRevoluteMotorRatioConfiguredAnchorB);
    stats.initialPerpendicularLeverArmA =
        (leverA - worldAxis * leverA.dot(worldAxis)).magnitude();
    stats.initialPerpendicularLeverArmB =
        (leverB - worldAxis * leverB.dot(worldAxis)).magnitude();
  }
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static bool setSingleShapeFilterData(PxRigidActor &actor,
                                     PxU32 tag) {
  PxShape *shape = NULL;
  if (actor.getNbShapes() != 1 ||
      actor.getShapes(&shape, 1) != 1 || !shape)
    return false;
  PxFilterData filterData;
  filterData.word0 = tag;
  shape->setSimulationFilterData(filterData);
  return true;
}

static void createRevoluteMotorContactTarget() {
  const PxTransform poseA(
      PxVec3(-1.0f, gRevoluteMotorContactCenterHeight, 0.0f));
  const PxTransform poseB(
      PxVec3(1.0f, gRevoluteMotorContactCenterHeight, 0.0f));
  const PxCapsuleGeometry geometry(gRevoluteMotorContactRadius,
                                   gRevoluteMotorContactHalfHeight);
  gRevoluteMotorContactBodyA = PxCreateDynamic(
      *gPhysics, poseA, geometry, *gMaterial, 1.0f);
  gRevoluteMotorContactBodyB = PxCreateDynamic(
      *gPhysics, poseB, geometry, *gMaterial, 1.0f);
  gRevoluteMotorContactGround =
      PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0),
                    *gMaterial);
  if (!gRevoluteMotorContactBodyA ||
      !gRevoluteMotorContactBodyB ||
      !gRevoluteMotorContactGround) {
    PX_RELEASE(gRevoluteMotorContactBodyA);
    PX_RELEASE(gRevoluteMotorContactBodyB);
    PX_RELEASE(gRevoluteMotorContactGround);
    gInitializationFailed = true;
    return;
  }

  gRevoluteMotorContactBodyA->setMass(1.0f);
  gRevoluteMotorContactBodyB->setMass(1.0f);
  gRevoluteMotorContactBodyA->setMassSpaceInertiaTensor(
      PxVec3(1.0f, 2.0f, 3.0f));
  gRevoluteMotorContactBodyB->setMassSpaceInertiaTensor(
      PxVec3(3.0f, 4.0f, 5.0f));
  gRevoluteMotorContactBodyA->setLinearDamping(0.0f);
  gRevoluteMotorContactBodyA->setAngularDamping(0.0f);
  gRevoluteMotorContactBodyB->setLinearDamping(0.0f);
  gRevoluteMotorContactBodyB->setAngularDamping(0.0f);
  if (!setSingleShapeFilterData(
          *gRevoluteMotorContactBodyA,
          eFILTER_MOTOR_CONTACT_BODY) ||
      !setSingleShapeFilterData(
          *gRevoluteMotorContactBodyB,
          eFILTER_MOTOR_CONTACT_BODY) ||
      !setSingleShapeFilterData(
          *gRevoluteMotorContactGround,
          eFILTER_MOTOR_CONTACT_GROUND)) {
    gInitializationFailed = true;
    return;
  }
  gScene->addActor(*gRevoluteMotorContactGround);
  gScene->addActor(*gRevoluteMotorContactBodyA);
  gScene->addActor(*gRevoluteMotorContactBodyB);

  gRevoluteMotorContactJoint = PxRevoluteJointCreate(
      *gPhysics, gRevoluteMotorContactBodyA,
      PxTransform(PxVec3(1.0f, 0.0f, 0.0f)),
      gRevoluteMotorContactBodyB,
      PxTransform(PxVec3(-1.0f, 0.0f, 0.0f)));
  if (!gRevoluteMotorContactJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorContactJoint->setDriveVelocity(
      gRevoluteMotorContactTargetVelocity, false);
  gRevoluteMotorContactJoint->setDriveForceLimit(
      gRevoluteMotorContactForceLimit);
  gRevoluteMotorContactJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorContactJoint->getActors(actor0, actor1);
  RevoluteMotorContactStats &stats =
      gRevoluteMotorContactStats;
  stats.actorOrderValid =
      actor0 == gRevoluteMotorContactBodyA &&
      actor1 == gRevoluteMotorContactBodyB;
  stats.targetVelocityReadback =
      gRevoluteMotorContactJoint->getDriveVelocity();
  stats.forceLimitReadback =
      gRevoluteMotorContactJoint->getDriveForceLimit();
  stats.driveEnabledReadback =
      gRevoluteMotorContactJoint->getRevoluteJointFlags().isSet(
          PxRevoluteJointFlag::eDRIVE_ENABLED);
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static void createRevoluteMotorKinematicTarget() {
  gRevoluteMotorKinematicBody = PxCreateDynamic(
      *gPhysics, gRevoluteMotorKinematicInitialPose,
      PxBoxGeometry(0.25f, 0.25f, 0.25f), *gMaterial, 1.0f);
  gRevoluteMotorKinematicDynamicBody = PxCreateDynamic(
      *gPhysics, gRevoluteMotorKinematicInitialPose,
      PxBoxGeometry(0.25f, 0.25f, 0.25f), *gMaterial, 1.0f);
  if (!gRevoluteMotorKinematicBody ||
      !gRevoluteMotorKinematicDynamicBody) {
    PX_RELEASE(gRevoluteMotorKinematicBody);
    PX_RELEASE(gRevoluteMotorKinematicDynamicBody);
    gInitializationFailed = true;
    return;
  }

  gRevoluteMotorKinematicBody->setRigidBodyFlag(
      PxRigidBodyFlag::eKINEMATIC, true);
  gRevoluteMotorKinematicBody->setLinearDamping(0.0f);
  gRevoluteMotorKinematicBody->setAngularDamping(0.0f);
  gRevoluteMotorKinematicDynamicBody->setMass(1.0f);
  gRevoluteMotorKinematicDynamicBody->setMassSpaceInertiaTensor(
      PxVec3(1.0f, 2.0f, 3.0f));
  gRevoluteMotorKinematicDynamicBody->setLinearDamping(0.0f);
  gRevoluteMotorKinematicDynamicBody->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorKinematicBody);
  gScene->addActor(*gRevoluteMotorKinematicDynamicBody);

  gRevoluteMotorKinematicJoint = PxRevoluteJointCreate(
      *gPhysics, gRevoluteMotorKinematicBody,
      PxTransform(PxIdentity),
      gRevoluteMotorKinematicDynamicBody,
      PxTransform(PxIdentity));
  if (!gRevoluteMotorKinematicJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorKinematicJoint->setDriveVelocity(
      gRevoluteMotorKinematicTargetVelocity, false);
  gRevoluteMotorKinematicJoint->setDriveForceLimit(
      gRevoluteMotorKinematicForceLimit);
  gRevoluteMotorKinematicJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorKinematicJoint->getActors(actor0, actor1);
  RevoluteMotorKinematicStats &stats =
      gRevoluteMotorKinematicStats;
  stats.actorOrderValid =
      actor0 == gRevoluteMotorKinematicBody &&
      actor1 == gRevoluteMotorKinematicDynamicBody;
  stats.targetVelocityReadback =
      gRevoluteMotorKinematicJoint->getDriveVelocity();
  stats.forceLimitReadback =
      gRevoluteMotorKinematicJoint->getDriveForceLimit();
  stats.driveEnabledReadback =
      gRevoluteMotorKinematicJoint->getRevoluteJointFlags().isSet(
          PxRevoluteJointFlag::eDRIVE_ENABLED);
  stats.kinematicFlagReadback =
      gRevoluteMotorKinematicBody->getRigidBodyFlags().isSet(
          PxRigidBodyFlag::eKINEMATIC);
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static void createRevoluteMotorOffPrincipalTarget() {
  const PxTransform bodyPose(
      PxVec3(0.0f, 10.0f, 0.0f),
      PxQuat(gRevoluteMotorOffPrincipalAngle, PxVec3(0.0f, 0.0f, 1.0f)));
  const PxTransform worldFrame(bodyPose.p);
  gRevoluteMotorOffPrincipalBody = PxCreateDynamic(
      *gPhysics, bodyPose, PxBoxGeometry(0.25f, 0.25f, 0.25f),
      *gMaterial, 1.0f);
  if (!gRevoluteMotorOffPrincipalBody) {
    gInitializationFailed = true;
    return;
  }

  gRevoluteMotorOffPrincipalBody->setMass(1.0f);
  gRevoluteMotorOffPrincipalBody->setMassSpaceInertiaTensor(
      gRevoluteMotorOffPrincipalInertia);
  gRevoluteMotorOffPrincipalBody->setLinearDamping(0.0f);
  gRevoluteMotorOffPrincipalBody->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorOffPrincipalBody);

  gRevoluteMotorOffPrincipalJoint = PxRevoluteJointCreate(
      *gPhysics, NULL, worldFrame, gRevoluteMotorOffPrincipalBody,
      bodyPose.transformInv(worldFrame));
  if (!gRevoluteMotorOffPrincipalJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorOffPrincipalJoint->setDriveVelocity(
      gRevoluteMotorOffPrincipalTargetVelocity, false);
  gRevoluteMotorOffPrincipalJoint->setDriveForceLimit(
      gRevoluteMotorOffPrincipalForceLimit);
  gRevoluteMotorOffPrincipalJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  RevoluteMotorOffPrincipalStats &stats =
      gRevoluteMotorOffPrincipalStats;
  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorOffPrincipalJoint->getActors(actor0, actor1);
  stats.actorOrderValid =
      actor0 == NULL && actor1 == gRevoluteMotorOffPrincipalBody;
  stats.targetVelocityReadback =
      gRevoluteMotorOffPrincipalJoint->getDriveVelocity();
  stats.forceLimitReadback =
      gRevoluteMotorOffPrincipalJoint->getDriveForceLimit();
  stats.driveEnabledReadback =
      gRevoluteMotorOffPrincipalJoint->getRevoluteJointFlags().isSet(
          PxRevoluteJointFlag::eDRIVE_ENABLED);

  const PxVec3 worldAxis(1.0f, 0.0f, 0.0f);
  const PxVec3 localAxis = bodyPose.q.rotateInv(worldAxis);
  const PxVec3 localResponse(
      localAxis.x / gRevoluteMotorOffPrincipalInertia.x,
      localAxis.y / gRevoluteMotorOffPrincipalInertia.y,
      localAxis.z / gRevoluteMotorOffPrincipalInertia.z);
  const PxVec3 worldResponse = bodyPose.q.rotate(localResponse);
  stats.initialOffPrincipalResponse =
      (worldResponse -
       worldAxis * worldResponse.dot(worldAxis)).magnitude();
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static void createRevoluteMotorOffCenterTarget() {
  const bool spatial = isRevoluteMotorSpatialCase();
  const PxTransform bodyPose =
      spatial
          ? PxTransform(
                gRevoluteMotorOffCenterInitialPose.p,
                PxQuat(gRevoluteMotorOffPrincipalAngle,
                       PxVec3(0.0f, 0.0f, 1.0f)))
          : gRevoluteMotorOffCenterInitialPose;
  const PxTransform worldFrame(PxVec3(0.0f, 10.0f, 0.0f));
  const PxTransform bodyFrame = bodyPose.transformInv(worldFrame);
  const PxVec3 inertia =
      spatial ? gRevoluteMotorOffPrincipalInertia
              : PxVec3(1.0f, 2.0f, 3.0f);
  gRevoluteMotorOffCenterConfiguredLocalAnchor = bodyFrame.p;
  gRevoluteMotorOffCenterBody = PxCreateDynamic(
      *gPhysics, bodyPose,
      PxBoxGeometry(0.25f, 0.25f, 0.25f), *gMaterial, 1.0f);
  if (!gRevoluteMotorOffCenterBody) {
    gInitializationFailed = true;
    return;
  }

  gRevoluteMotorOffCenterBody->setMass(1.0f);
  gRevoluteMotorOffCenterBody->setMassSpaceInertiaTensor(
      inertia);
  gRevoluteMotorOffCenterBody->setLinearDamping(0.0f);
  gRevoluteMotorOffCenterBody->setAngularDamping(0.0f);
  gScene->addActor(*gRevoluteMotorOffCenterBody);

  gRevoluteMotorOffCenterJoint = PxRevoluteJointCreate(
      *gPhysics, NULL, worldFrame, gRevoluteMotorOffCenterBody,
      bodyFrame);
  if (!gRevoluteMotorOffCenterJoint) {
    gInitializationFailed = true;
    return;
  }
  gRevoluteMotorOffCenterJoint->setDriveVelocity(
      gRevoluteMotorOffCenterTargetVelocity, false);
  gRevoluteMotorOffCenterJoint->setDriveForceLimit(
      gRevoluteMotorOffCenterForceLimit);
  gRevoluteMotorOffCenterJoint->setRevoluteJointFlag(
      PxRevoluteJointFlag::eDRIVE_ENABLED, true);

  RevoluteMotorOffCenterStats &stats =
      gRevoluteMotorOffCenterStats;
  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gRevoluteMotorOffCenterJoint->getActors(actor0, actor1);
  stats.actorOrderValid =
      actor0 == NULL && actor1 == gRevoluteMotorOffCenterBody;
  stats.targetVelocityReadback =
      gRevoluteMotorOffCenterJoint->getDriveVelocity();
  stats.forceLimitReadback =
      gRevoluteMotorOffCenterJoint->getDriveForceLimit();
  stats.driveEnabledReadback =
      gRevoluteMotorOffCenterJoint->getRevoluteJointFlags().isSet(
          PxRevoluteJointFlag::eDRIVE_ENABLED);
  const PxVec3 worldAxis(1.0f, 0.0f, 0.0f);
  const PxVec3 worldLeverArm =
      bodyPose.q.rotate(gRevoluteMotorOffCenterConfiguredLocalAnchor);
  stats.initialPerpendicularLeverArm =
      (worldLeverArm -
       worldAxis * worldLeverArm.dot(worldAxis))
          .magnitude();
  const PxVec3 localAxis = bodyPose.q.rotateInv(worldAxis);
  const PxVec3 localResponse(
      localAxis.x / inertia.x,
      localAxis.y / inertia.y,
      localAxis.z / inertia.z);
  const PxVec3 worldResponse = bodyPose.q.rotate(localResponse);
  stats.initialOffPrincipalResponse =
      (worldResponse -
       worldAxis * worldResponse.dot(worldAxis)).magnitude();
  stats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  stats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  stats.initialConstraints = gScene->getNbConstraints();
}

static void createNativeBreakReactionTarget() {
  const PxTransform targetPose(PxVec3(0.0f, 10.0f, 0.0f));
  gNativeBreakReactionBody = PxCreateDynamic(
      *gPhysics, targetPose, PxBoxGeometry(0.5f, 0.5f, 0.5f),
      *gMaterial, 1.0f);
  if (!gNativeBreakReactionBody ||
      !PxRigidBodyExt::setMassAndUpdateInertia(
          *gNativeBreakReactionBody, 1.0f)) {
    PX_RELEASE(gNativeBreakReactionBody);
    gInitializationFailed = true;
    return;
  }
  gNativeBreakReactionBody->setMassSpaceInertiaTensor(PxVec3(1.0f));
  gNativeBreakReactionBody->setLinearDamping(0.0f);
  gNativeBreakReactionBody->setAngularDamping(0.0f);
  gScene->addActor(*gNativeBreakReactionBody);

  if (gImpactJointKind == eJOINT_PRISMATIC) {
    gNativeBreakReactionJoint = PxPrismaticJointCreate(
        *gPhysics, NULL, targetPose, gNativeBreakReactionBody,
        PxTransform(PxIdentity));
  } else if (gImpactJointKind == eJOINT_REVOLUTE) {
    gNativeBreakReactionJoint = PxRevoluteJointCreate(
        *gPhysics, NULL, targetPose, gNativeBreakReactionBody,
        PxTransform(PxIdentity));
  }
  if (!gNativeBreakReactionJoint) {
    gInitializationFailed = true;
    return;
  }

  PxReal linearThreshold = PX_MAX_F32;
  PxReal angularThreshold = PX_MAX_F32;
  if (gHeadlessCase == eCASE_NATIVE_NO_BREAK) {
    if (isNativeAngularReactionCase())
      angularThreshold = gNativeHighBreakThreshold;
    else
      linearThreshold = gNativeHighBreakThreshold;
  } else if (gHeadlessCase == eCASE_NATIVE_BREAK) {
    if (isNativeAngularReactionCase())
      angularThreshold = gNativeLowBreakThreshold;
    else
      linearThreshold = gNativeLowBreakThreshold;
  }
  gNativeBreakReactionJoint->setBreakForce(
      linearThreshold, angularThreshold);
  gNativeBreakReactionJoint->getBreakForce(
      gNativeBreakReactionStats.breakForceReadback,
      gNativeBreakReactionStats.breakTorqueReadback);
  registerBreakableJoint(gNativeBreakReactionJoint);

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gNativeBreakReactionJoint->getActors(actor0, actor1);
  gNativeBreakReactionStats.actorOrderValid =
      actor0 == NULL && actor1 == gNativeBreakReactionBody;
  gNativeBreakReactionStats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gNativeBreakReactionStats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gNativeBreakReactionStats.initialConstraints =
      gScene->getNbConstraints();
}

static PxQuat makeSphericalConeSwing(PxReal swingY, PxReal swingZ) {
  const PxVec3 tanQuarter(0.0f, PxTan(swingY * 0.25f),
                         PxTan(swingZ * 0.25f));
  const PxReal magnitudeSquared = tanQuarter.magnitudeSquared();
  const PxReal inverseDenominator = 1.0f / (1.0f + magnitudeSquared);
  return PxQuat(0.0f, 2.0f * tanQuarter.y * inverseDenominator,
                2.0f * tanQuarter.z * inverseDenominator,
                (1.0f - magnitudeSquared) * inverseDenominator)
      .getNormalized();
}

static PxReal computeSphericalConeEllipseRadius(const PxTransform &poseA,
                                                const PxTransform &poseB) {
  PxQuat relative = poseA.q.getConjugate() * poseB.q;
  relative.normalize();
  if (relative.w < 0.0f)
    relative = -relative;
  PxQuat swing, twist;
  PxSeparateSwingTwist(relative, swing, twist);
  if (swing.w < 0.0f)
    swing = -swing;
  const PxReal denominator = 1.0f + swing.w;
  if (denominator <= 1e-6f)
    return PX_MAX_F32;
  const PxReal swingY = 4.0f * PxAtan2(swing.y, denominator);
  const PxReal swingZ = 4.0f * PxAtan2(swing.z, denominator);
  const PxReal normalizedY = swingY / gSphericalConeLimitY;
  const PxReal normalizedZ = swingZ / gSphericalConeLimitZ;
  return PxSqrt(normalizedY * normalizedY +
                normalizedZ * normalizedZ);
}

static bool configureSphericalConeDynamic(PxRigidDynamic *body) {
  if (!body)
    return false;
  body->setMass(1.0f);
  body->setMassSpaceInertiaTensor(PxVec3(1.0f));
  body->setLinearDamping(0.0f);
  body->setAngularDamping(0.0f);
  body->setMaxAngularVelocity(100.0f);
  return body->getMass() == 1.0f &&
         body->getMassSpaceInertiaTensor() == PxVec3(1.0f);
}

static void createSphericalConeTarget() {
  const PxReal initialSwingY =
      isSphericalConeInsideCase() ? gSphericalConeInsideY
                                  : gSphericalConeOutsideY;
  const PxReal initialSwingZ =
      isSphericalConeInsideCase() ? gSphericalConeInsideZ
                                  : gSphericalConeOutsideZ;
  const PxTransform poseA(PxVec3(0.0f, 10.0f, 0.0f), PxQuat(PxIdentity));
  const PxTransform poseB(PxVec3(0.0f, 10.0f, 0.0f),
                          makeSphericalConeSwing(initialSwingY,
                                                initialSwingZ));
  const PxBoxGeometry geometry(0.5f, 0.5f, 0.5f);

  if (gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC) {
    gSphericalConeDynamicA =
        PxCreateDynamic(*gPhysics, poseA, geometry, *gMaterial, 1.0f);
    gSphericalConeActorA = gSphericalConeDynamicA;
    if (!configureSphericalConeDynamic(gSphericalConeDynamicA)) {
      gInitializationFailed = true;
      return;
    }
  } else {
    gSphericalConeActorA =
        PxCreateStatic(*gPhysics, poseA, geometry, *gMaterial);
    if (!gSphericalConeActorA) {
      gInitializationFailed = true;
      return;
    }
  }

  gSphericalConeActorB =
      PxCreateDynamic(*gPhysics, poseB, geometry, *gMaterial, 1.0f);
  if (!configureSphericalConeDynamic(gSphericalConeActorB)) {
    gInitializationFailed = true;
    return;
  }

  gScene->addActor(*gSphericalConeActorA);
  gScene->addActor(*gSphericalConeActorB);
  gSphericalConeJoint = PxSphericalJointCreate(
      *gPhysics, gSphericalConeActorA, PxTransform(PxIdentity),
      gSphericalConeActorB, PxTransform(PxIdentity));
  if (!gSphericalConeJoint) {
    gInitializationFailed = true;
    return;
  }
  gSphericalConeJoint->setLimitCone(
      PxJointLimitCone(gSphericalConeLimitY, gSphericalConeLimitZ));
  gSphericalConeJoint->setSphericalJointFlag(
      PxSphericalJointFlag::eLIMIT_ENABLED, true);

  const PxJointLimitCone readback = gSphericalConeJoint->getLimitCone();
  gSphericalConeStats.limitYReadback = readback.yAngle;
  gSphericalConeStats.limitZReadback = readback.zAngle;
  gSphericalConeStats.limitEnabledReadback =
      gSphericalConeJoint->getSphericalJointFlags().isSet(
          PxSphericalJointFlag::eLIMIT_ENABLED);
  gSphericalConeStats.initialEllipseRadius =
      computeSphericalConeEllipseRadius(poseA, poseB);
  gSphericalConeStats.minimumEllipseRadius =
      gSphericalConeStats.initialEllipseRadius;
  gSphericalConeStats.maximumEllipseRadius =
      gSphericalConeStats.initialEllipseRadius;

  PxRigidActor *actualActor0 = NULL;
  PxRigidActor *actualActor1 = NULL;
  gSphericalConeJoint->getActors(actualActor0, actualActor1);
  gSphericalConeStats.actorOrderValid =
      actualActor0 == gSphericalConeActorA &&
      actualActor1 == gSphericalConeActorB;
  gSphericalConeStats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gSphericalConeStats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gSphericalConeStats.initialConstraints = gScene->getNbConstraints();
}

static void createEndpointPrismaticTarget() {
  const PxQuat bodyRotation(PxPi / 4.0f, PxVec3(0.0f, 1.0f, 0.0f));
  const PxQuat worldFrameRotation(PxPi / 2.0f,
                                  PxVec3(0.0f, 1.0f, 0.0f));
  const PxTransform targetPose(PxVec3(0.0f, 10.0f, 0.0f), bodyRotation);
  const PxTransform worldFrame(targetPose.p, worldFrameRotation);
  const PxTransform bodyLocalFrame = targetPose.getInverse() * worldFrame;

  gEndpointTarget = PxCreateDynamic(*gPhysics, targetPose,
                                    PxSphereGeometry(1.0f), *gMaterial, 1.0f);
  if (!gEndpointTarget) {
    gInitializationFailed = true;
    return;
  }
  gEndpointTarget->setLinearDamping(0.0f);
  gEndpointTarget->setAngularDamping(0.0f);

  PxShape *shape = NULL;
  if (gEndpointTarget->getShapes(&shape, 1) != 1 || !shape) {
    gInitializationFailed = true;
    return;
  }
  PxFilterData filterData;
  filterData.word0 = eFILTER_CHAIN_BODY;
  filterData.word1 = 0;
  filterData.word2 = 0;
  shape->setSimulationFilterData(filterData);
  gScene->addActor(*gEndpointTarget);

  if (gEndpointKind == eENDPOINT_FORWARD) {
    gEndpointJoint = PxPrismaticJointCreate(
        *gPhysics, NULL, worldFrame, gEndpointTarget, bodyLocalFrame);
  } else {
    gEndpointJoint = PxPrismaticJointCreate(
        *gPhysics, gEndpointTarget, bodyLocalFrame, NULL, worldFrame);
  }
  if (!gEndpointJoint) {
    gInitializationFailed = true;
    return;
  }
  gEndpointStats.limitEnabled =
      gEndpointJoint->getPrismaticJointFlags().isSet(
          PxPrismaticJointFlag::eLIMIT_ENABLED);
  const PxJointActorIndex::Enum worldIndex =
      gEndpointKind == eENDPOINT_FORWARD ? PxJointActorIndex::eACTOR0
                                         : PxJointActorIndex::eACTOR1;
  const PxJointActorIndex::Enum dynamicIndex =
      gEndpointKind == eENDPOINT_FORWARD ? PxJointActorIndex::eACTOR1
                                         : PxJointActorIndex::eACTOR0;
  const PxTransform actualWorldFrame =
      gEndpointJoint->getLocalPose(worldIndex);
  const PxTransform actualDynamicLocalFrame =
      gEndpointJoint->getLocalPose(dynamicIndex);
  const PxTransform shapeLocalPose = shape->getLocalPose();
  const PxGeometry &shapeGeometry = shape->getGeometry();
  const bool sphereGeometryValid =
      shapeGeometry.getType() == PxGeometryType::eSPHERE;
  gEndpointStats.shapeCount = gEndpointTarget->getNbShapes();
  gEndpointStats.worldFramePositionError =
      (actualWorldFrame.p - worldFrame.p).magnitude();
  gEndpointStats.dynamicLocalPositionError =
      (actualDynamicLocalFrame.p - bodyLocalFrame.p).magnitude();
  gEndpointStats.worldFrameRotationDot =
      PxAbs(actualWorldFrame.q.dot(worldFrame.q));
  gEndpointStats.dynamicLocalRotationDot =
      PxAbs(actualDynamicLocalFrame.q.dot(bodyLocalFrame.q));
  gEndpointStats.shapeLocalPositionError = shapeLocalPose.p.magnitude();
  gEndpointStats.shapeLocalRotationDot =
      PxAbs(shapeLocalPose.q.dot(PxQuat(PxIdentity)));
  gEndpointStats.shapeRadius =
      sphereGeometryValid
          ? static_cast<const PxSphereGeometry &>(shapeGeometry).radius
          : 0.0f;
  gEndpointStats.fixtureWitnessValid =
      gEndpointStats.shapeCount == 1 && sphereGeometryValid &&
      PxAbs(gEndpointStats.shapeRadius - 1.0f) <= 1e-6f &&
      gEndpointStats.worldFramePositionError <= 1e-6f &&
      gEndpointStats.dynamicLocalPositionError <= 1e-6f &&
      gEndpointStats.worldFrameRotationDot >= 0.99999f &&
      gEndpointStats.dynamicLocalRotationDot >= 0.99999f &&
      gEndpointStats.shapeLocalPositionError <= 1e-6f &&
      gEndpointStats.shapeLocalRotationDot >= 0.99999f;
  gChains.push_back(ChainRecord(eJOINT_PRISMATIC, "prismatic"));
  ChainRecord &chain = gChains.back();
  chain.bodies.push_back(gEndpointTarget);
  chain.joints.push_back(gEndpointJoint);
  chain.targetBody = gEndpointTarget;

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gEndpointJoint->getActors(actor0, actor1);
  gEndpointStats.actorOrderValid =
      gEndpointKind == eENDPOINT_FORWARD
          ? actor0 == NULL && actor1 == gEndpointTarget
          : actor0 == gEndpointTarget && actor1 == NULL;
  getJointWorldAxes(gEndpointJoint, gEndpointStats.actor0Axis,
                    gEndpointStats.actor1Axis);
  gEndpointStats.expectedAxis =
      worldFrameRotation.rotate(PxVec3(1.0f, 0.0f, 0.0f));
  const PxReal expectedAxisMagnitude = gEndpointStats.expectedAxis.magnitude();
  if (expectedAxisMagnitude > 1e-12f)
    gEndpointStats.expectedAxis /= expectedAxisMagnitude;
  const PxReal actor0AxisDot =
      gEndpointStats.actor0Axis.dot(gEndpointStats.expectedAxis);
  const PxReal actor1AxisDot =
      gEndpointStats.actor1Axis.dot(gEndpointStats.expectedAxis);
  gEndpointStats.bodyRotationDot = PxAbs(
      gEndpointTarget->getGlobalPose().q.dot(bodyRotation));
  gEndpointStats.expectedAxisDot =
      gEndpointStats.expectedAxis.dot(PxVec3(0.0f, 0.0f, -1.0f));
  gEndpointStats.dynamicLocalAxis =
      bodyLocalFrame.q.rotate(PxVec3(1.0f, 0.0f, 0.0f));
  gEndpointStats.dynamicLocalAxisDot =
      gEndpointStats.dynamicLocalAxis.dot(gEndpointStats.expectedAxis);
  gEndpointStats.gravity = gScene->getGravity();
  gEndpointStats.frameWitnessValid =
      gEndpointStats.actor0Axis.isFinite() &&
      gEndpointStats.actor1Axis.isFinite() &&
      gEndpointStats.expectedAxis.isFinite() &&
      gEndpointStats.gravity.isFinite() && actor0AxisDot >= 0.99999f &&
      actor1AxisDot >= 0.99999f &&
      gEndpointStats.bodyRotationDot >= 0.99999f &&
      gEndpointStats.expectedAxisDot >= 0.99999f &&
      gEndpointStats.dynamicLocalAxis.isFinite() &&
      PxAbs(gEndpointStats.dynamicLocalAxisDot - PxSqrt(0.5f)) <= 1e-5f &&
      gEndpointStats.gravity.magnitudeSquared() <= 1e-12f;
  gEndpointStats.initialTargetPosition = targetPose.p;
  gEndpointStats.initialTargetVelocity =
      gEndpointTarget->getLinearVelocity();
  gEndpointStats.responseBaselinePosition =
      gEndpointStats.initialTargetPosition;
  gEndpointStats.responseBaselineVelocity =
      gEndpointStats.initialTargetVelocity;
  gEndpointStats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gEndpointStats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gEndpointStats.initialConstraints = gScene->getNbConstraints();
}

static void createEndpointRevoluteTarget() {
  const PxQuat bodyRotation(PxPi / 4.0f, PxVec3(0.0f, 1.0f, 0.0f));
  const PxQuat worldFrameRotation(PxPi / 2.0f,
                                  PxVec3(0.0f, 1.0f, 0.0f));
  const PxTransform targetPose(PxVec3(0.0f, 10.0f, 0.0f), bodyRotation);
  const PxTransform worldFrame(targetPose.p, worldFrameRotation);
  const PxTransform bodyLocalFrame = targetPose.getInverse() * worldFrame;

  gEndpointTarget = PxCreateDynamic(*gPhysics, targetPose,
                                    PxSphereGeometry(1.0f), *gMaterial, 1.0f);
  if (!gEndpointTarget ||
      !PxRigidBodyExt::setMassAndUpdateInertia(*gEndpointTarget, 1.0f)) {
    gInitializationFailed = true;
    return;
  }
  gEndpointTarget->setLinearDamping(0.0f);
  gEndpointTarget->setAngularDamping(0.0f);

  PxShape *shape = NULL;
  if (gEndpointTarget->getShapes(&shape, 1) != 1 || !shape) {
    gInitializationFailed = true;
    return;
  }
  PxFilterData filterData;
  filterData.word0 = eFILTER_CHAIN_BODY;
  filterData.word1 = 0;
  filterData.word2 = 0;
  shape->setSimulationFilterData(filterData);
  gScene->addActor(*gEndpointTarget);

  if (gEndpointKind == eENDPOINT_FORWARD) {
    gEndpointRevoluteJoint = PxRevoluteJointCreate(
        *gPhysics, NULL, worldFrame, gEndpointTarget, bodyLocalFrame);
  } else {
    gEndpointRevoluteJoint = PxRevoluteJointCreate(
        *gPhysics, gEndpointTarget, bodyLocalFrame, NULL, worldFrame);
  }
  if (!gEndpointRevoluteJoint) {
    gInitializationFailed = true;
    return;
  }

  const PxRevoluteJointFlags flags =
      gEndpointRevoluteJoint->getRevoluteJointFlags();
  gEndpointStats.limitEnabled =
      flags.isSet(PxRevoluteJointFlag::eLIMIT_ENABLED);
  gEndpointAngularStats.driveEnabled =
      flags.isSet(PxRevoluteJointFlag::eDRIVE_ENABLED);
  const PxJointActorIndex::Enum worldIndex =
      gEndpointKind == eENDPOINT_FORWARD ? PxJointActorIndex::eACTOR0
                                         : PxJointActorIndex::eACTOR1;
  const PxJointActorIndex::Enum dynamicIndex =
      gEndpointKind == eENDPOINT_FORWARD ? PxJointActorIndex::eACTOR1
                                         : PxJointActorIndex::eACTOR0;
  const PxTransform actualWorldFrame =
      gEndpointRevoluteJoint->getLocalPose(worldIndex);
  const PxTransform actualDynamicLocalFrame =
      gEndpointRevoluteJoint->getLocalPose(dynamicIndex);
  const PxTransform shapeLocalPose = shape->getLocalPose();
  const PxGeometry &shapeGeometry = shape->getGeometry();
  const bool sphereGeometryValid =
      shapeGeometry.getType() == PxGeometryType::eSPHERE;
  gEndpointStats.shapeCount = gEndpointTarget->getNbShapes();
  gEndpointStats.worldFramePositionError =
      (actualWorldFrame.p - worldFrame.p).magnitude();
  gEndpointStats.dynamicLocalPositionError =
      (actualDynamicLocalFrame.p - bodyLocalFrame.p).magnitude();
  gEndpointStats.worldFrameRotationDot =
      PxAbs(actualWorldFrame.q.dot(worldFrame.q));
  gEndpointStats.dynamicLocalRotationDot =
      PxAbs(actualDynamicLocalFrame.q.dot(bodyLocalFrame.q));
  gEndpointStats.shapeLocalPositionError = shapeLocalPose.p.magnitude();
  gEndpointStats.shapeLocalRotationDot =
      PxAbs(shapeLocalPose.q.dot(PxQuat(PxIdentity)));
  gEndpointStats.shapeRadius =
      sphereGeometryValid
          ? static_cast<const PxSphereGeometry &>(shapeGeometry).radius
          : 0.0f;
  gEndpointStats.fixtureWitnessValid =
      gEndpointStats.shapeCount == 1 && sphereGeometryValid &&
      PxAbs(gEndpointStats.shapeRadius - 1.0f) <= 1e-6f &&
      gEndpointStats.worldFramePositionError <= 1e-6f &&
      gEndpointStats.dynamicLocalPositionError <= 1e-6f &&
      gEndpointStats.worldFrameRotationDot >= 0.99999f &&
      gEndpointStats.dynamicLocalRotationDot >= 0.99999f &&
      gEndpointStats.shapeLocalPositionError <= 1e-6f &&
      gEndpointStats.shapeLocalRotationDot >= 0.99999f;

  gChains.push_back(ChainRecord(eJOINT_REVOLUTE, "revolute"));
  ChainRecord &chain = gChains.back();
  chain.bodies.push_back(gEndpointTarget);
  chain.joints.push_back(gEndpointRevoluteJoint);
  chain.targetBody = gEndpointTarget;

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  gEndpointRevoluteJoint->getActors(actor0, actor1);
  gEndpointStats.actorOrderValid =
      gEndpointKind == eENDPOINT_FORWARD
          ? actor0 == NULL && actor1 == gEndpointTarget
          : actor0 == gEndpointTarget && actor1 == NULL;
  getJointWorldAxes(gEndpointRevoluteJoint, gEndpointStats.actor0Axis,
                    gEndpointStats.actor1Axis);
  gEndpointStats.expectedAxis =
      worldFrameRotation.rotate(PxVec3(1.0f, 0.0f, 0.0f));
  const PxReal expectedAxisMagnitude = gEndpointStats.expectedAxis.magnitude();
  if (expectedAxisMagnitude > 1e-12f)
    gEndpointStats.expectedAxis /= expectedAxisMagnitude;
  gEndpointAngularStats.perpendicularAxis =
      worldFrameRotation.rotate(PxVec3(0.0f, 1.0f, 0.0f));
  const PxReal perpendicularMagnitude =
      gEndpointAngularStats.perpendicularAxis.magnitude();
  if (perpendicularMagnitude > 1e-12f)
    gEndpointAngularStats.perpendicularAxis /= perpendicularMagnitude;
  gEndpointAngularStats.requestedAngularVelocity =
      gEndpointStats.expectedAxis * gEndpointAngularAxialLaunchSpeed +
      gEndpointAngularStats.perpendicularAxis *
          gEndpointAngularTransverseLaunchSpeed;

  const PxReal actor0AxisDot =
      gEndpointStats.actor0Axis.dot(gEndpointStats.expectedAxis);
  const PxReal actor1AxisDot =
      gEndpointStats.actor1Axis.dot(gEndpointStats.expectedAxis);
  gEndpointStats.bodyRotationDot = PxAbs(
      gEndpointTarget->getGlobalPose().q.dot(bodyRotation));
  gEndpointStats.expectedAxisDot =
      gEndpointStats.expectedAxis.dot(PxVec3(0.0f, 0.0f, -1.0f));
  gEndpointStats.dynamicLocalAxis =
      bodyLocalFrame.q.rotate(PxVec3(1.0f, 0.0f, 0.0f));
  gEndpointStats.dynamicLocalAxisDot =
      gEndpointStats.dynamicLocalAxis.dot(gEndpointStats.expectedAxis);
  gEndpointAngularStats.bodyWorldAxisDot =
      bodyRotation.rotate(gEndpointStats.dynamicLocalAxis)
          .dot(gEndpointStats.expectedAxis);
  gEndpointStats.gravity = gScene->getGravity();
  gEndpointAngularStats.nonIdentityWitnessValid =
      PxAbs(bodyRotation.w) < 0.95f &&
      PxAbs(worldFrameRotation.w) < 0.8f &&
      PxAbs(gEndpointStats.dynamicLocalAxisDot - PxSqrt(0.5f)) <= 1e-5f &&
      gEndpointAngularStats.bodyWorldAxisDot >= 0.99999f;
  gEndpointStats.frameWitnessValid =
      gEndpointStats.actor0Axis.isFinite() &&
      gEndpointStats.actor1Axis.isFinite() &&
      gEndpointStats.expectedAxis.isFinite() &&
      gEndpointAngularStats.perpendicularAxis.isFinite() &&
      gEndpointStats.gravity.isFinite() && actor0AxisDot >= 0.99999f &&
      actor1AxisDot >= 0.99999f &&
      gEndpointStats.bodyRotationDot >= 0.99999f &&
      gEndpointStats.expectedAxisDot >= 0.99999f &&
      gEndpointStats.dynamicLocalAxis.isFinite() &&
      PxAbs(gEndpointStats.expectedAxis.dot(
                gEndpointAngularStats.perpendicularAxis)) <= 1e-6f &&
      gEndpointAngularStats.nonIdentityWitnessValid &&
      gEndpointStats.gravity.magnitudeSquared() <= 1e-12f;

  gEndpointStats.initialTargetPosition = targetPose.p;
  gEndpointStats.initialTargetVelocity =
      gEndpointTarget->getLinearVelocity();
  gEndpointStats.responseBaselinePosition =
      gEndpointStats.initialTargetPosition;
  gEndpointStats.responseBaselineVelocity =
      gEndpointStats.initialTargetVelocity;
  gEndpointAngularStats.initialTargetOrientation = targetPose.q;
  gEndpointAngularStats.responseBaselineOrientation = targetPose.q;
  gEndpointAngularStats.initialTargetAngularVelocity =
      gEndpointTarget->getAngularVelocity();
  gEndpointAngularStats.responseBaselineAngularVelocity =
      gEndpointAngularStats.initialTargetAngularVelocity;
  gEndpointAngularStats.baselineRawJointAngle =
      gEndpointRevoluteJoint->getAngle();
  gEndpointStats.initialDynamicActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  gEndpointStats.initialStaticActors =
      gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
  gEndpointStats.initialConstraints = gScene->getNbConstraints();
}

static void createJointChain(JointKind kind, PxReal z) {
  JointCreateFunction createJoint = createLimitedSpherical;
  switch (kind) {
  case eJOINT_SPHERICAL:
    createJoint = createLimitedSpherical;
    break;
  case eJOINT_FIXED:
    createJoint = createBreakableFixed;
    break;
  case eJOINT_D6:
    createJoint = createDampedD6;
    break;
  case eJOINT_PRISMATIC:
    createJoint = createLimitedPrismatic;
    break;
  case eJOINT_REVOLUTE:
    createJoint = createLimitedRevolute;
    break;
  }

  createChain(PxTransform(PxVec3(0.0f, 20.0f, z)), 5,
              PxBoxGeometry(2.0f, 0.5f, 0.5f), 4.0f, createJoint, kind,
              getJointKindName(kind));
}

// Opt-in coverage fixture for the AVBD producer-owned local-system path.
// One connected D6 star deliberately creates a wide joint island while
// remaining outside the ordinary correctness snippets and their baselines.
static void createWideJointStressScene() {
  const PxU32 childCount = 32;
  const PxU32 chainIndex = static_cast<PxU32>(gChains.size());
  gChains.push_back(ChainRecord(eJOINT_D6, "wide-d6-star"));
  ChainRecord &chain = gChains.back();
  const PxTransform rootPose(PxVec3(0.0f, 30.0f, 0.0f));
  const PxBoxGeometry childGeometry(1.0f, 0.5f, 0.5f);
  PxRigidDynamic *root = PxCreateDynamic(
      *gPhysics, rootPose, PxBoxGeometry(1.5f, 0.75f, 0.75f), *gMaterial,
      1.0f);
  if (!root) {
    gInitializationFailed = true;
    return;
  }
  PxShape *rootShape = NULL;
  if (root->getShapes(&rootShape, 1) == 1 && rootShape) {
    PxFilterData filterData;
    filterData.word0 = eFILTER_CHAIN_BODY;
    filterData.word1 = chainIndex;
    filterData.word2 = 0;
    rootShape->setSimulationFilterData(filterData);
  }
  gScene->addActor(*root);
  chain.bodies.push_back(root);

  for (PxU32 childIndex = 0; childIndex < childCount; ++childIndex) {
    const PxU32 column = childIndex & 7u;
    const PxU32 row = childIndex >> 3u;
    const PxVec3 offset((PxReal(column) - 3.5f) * 4.0f,
                        0.0f,
                        (PxReal(row) - 1.5f) * 4.0f);
    PxRigidDynamic *child = PxCreateDynamic(
        *gPhysics, PxTransform(rootPose.p + offset), childGeometry, *gMaterial,
        1.0f);
    if (!child) {
      gInitializationFailed = true;
      return;
    }
    PxJoint *joint = createDampedD6(
        root, PxTransform(offset), child, PxTransform(PxIdentity));
    if (!joint) {
      child->release();
      gInitializationFailed = true;
      return;
    }
    PxShape *shape = NULL;
    if (child->getShapes(&shape, 1) == 1 && shape) {
      PxFilterData filterData;
      filterData.word0 = eFILTER_CHAIN_BODY;
      filterData.word1 = chainIndex;
      filterData.word2 = childIndex + 1;
      shape->setSimulationFilterData(filterData);
    }
    chain.bodies.push_back(child);
    chain.joints.push_back(joint);
    if (childIndex == childCount / 2)
      chain.targetBody = child;
    gScene->addActor(*child);
  }
}

static void createConfiguredJointScene() {
  gChains.reserve(5);
  if (isRevoluteMotorCase()) {
    createRevoluteMotorTarget();
    return;
  }
  if (isRevoluteMotorLimitCase()) {
    createRevoluteMotorLimitTarget();
    return;
  }
  if (isRevoluteMotorFreeSpinCase()) {
    createRevoluteMotorFreeSpinTarget();
    return;
  }
  if (isRevoluteMotorRatioCase()) {
    createRevoluteMotorRatioTarget();
    return;
  }
  if (isRevoluteMotorContactCase()) {
    createRevoluteMotorContactTarget();
    return;
  }
  if (isRevoluteMotorKinematicCase()) {
    createRevoluteMotorKinematicTarget();
    return;
  }
  if (isRevoluteMotorOffPrincipalCase()) {
    createRevoluteMotorOffPrincipalTarget();
    return;
  }
  if (isRevoluteMotorOffCenterCase()) {
    createRevoluteMotorOffCenterTarget();
    return;
  }
  if (isSphericalConeCase()) {
    createSphericalConeTarget();
    return;
  }
  if (isNativeBreakReactionCase()) {
    createNativeBreakReactionTarget();
    return;
  }
  if (isForceReactionCase()) {
    if (isForcePairCase())
      createForcePairTarget();
    else
      createForceStaticTarget();
    return;
  }
  if (isEndpointProbe()) {
    if (isRevoluteEndpointProbe())
      createEndpointRevoluteTarget();
    else
      createEndpointPrismaticTarget();
    return;
  }
  if (gHeadlessCase == eCASE_FIXED_NO_BREAK ||
      gHeadlessCase == eCASE_FIXED_BREAK) {
    createIsolatedFixedTarget();
    return;
  }

  if (gHeadlessCase == eCASE_IMPACT_SINGLE) {
    createJointChain(gImpactJointKind, 0.0f);
    return;
  }

  if (isWideJointStressCase()) {
    createWideJointStressScene();
    return;
  }

  if (gHeadlessCase == eCASE_IMPACT_ALL) {
    createJointChain(eJOINT_SPHERICAL, 0.0f);
    if (gInitializationFailed)
      return;
    createJointChain(eJOINT_FIXED, -100.0f);
    if (gInitializationFailed)
      return;
    createJointChain(eJOINT_D6, -200.0f);
    if (gInitializationFailed)
      return;
    createJointChain(eJOINT_PRISMATIC, -300.0f);
    if (gInitializationFailed)
      return;
    createJointChain(eJOINT_REVOLUTE, -400.0f);
    return;
  }

  createJointChain(eJOINT_SPHERICAL, 0.0f);
  if (gInitializationFailed)
    return;
  createJointChain(eJOINT_FIXED, -10.0f);
  if (gInitializationFailed)
    return;
  createJointChain(eJOINT_D6, -20.0f);
  if (gInitializationFailed)
    return;
  createJointChain(eJOINT_PRISMATIC, -30.0f);
  if (gInitializationFailed)
    return;
  createJointChain(eJOINT_REVOLUTE, -40.0f);
}

static PxReal getJointFrameAxisMisalignment(PxJoint *joint) {
  if (!joint)
    return 0.0f;

  PxRigidActor *actor0 = NULL;
  PxRigidActor *actor1 = NULL;
  joint->getActors(actor0, actor1);
  const PxTransform local0 = joint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform local1 = joint->getLocalPose(PxJointActorIndex::eACTOR1);
  const PxQuat world0 = actor0 ? actor0->getGlobalPose().q * local0.q : local0.q;
  const PxQuat world1 = actor1 ? actor1->getGlobalPose().q * local1.q : local1.q;
  const PxVec3 axis0 = world0.rotate(PxVec3(1.0f, 0.0f, 0.0f));
  const PxVec3 axis1 = world1.rotate(PxVec3(1.0f, 0.0f, 0.0f));
  if (!axis0.isFinite() || !axis1.isFinite())
    return PX_MAX_F32;
  return PxAcos(PxClamp(axis0.dot(axis1), -1.0f, 1.0f));
}

static PxReal saturateMetric(double value) {
  if (!std::isfinite(value) || value >= double(PX_MAX_F32))
    return PX_MAX_F32;
  if (value <= -double(PX_MAX_F32))
    return -PX_MAX_F32;
  return PxReal(value);
}

static PxReal addMetrics(PxReal lhs, PxReal rhs) {
  return saturateMetric(double(lhs) + double(rhs));
}

static PxReal getSafeMagnitude(const PxVec3 &value) {
  const double x = double(value.x);
  const double y = double(value.y);
  const double z = double(value.z);
  return saturateMetric(std::sqrt(x * x + y * y + z * z));
}

static bool getShortestRotationVector(const PxQuat &orientation,
                                      const PxQuat &baseline,
                                      PxVec3 &rotationVector) {
  rotationVector = PxVec3(0.0f);
  if (!orientation.isFinite() || !baseline.isFinite())
    return false;
  PxQuat delta = orientation * baseline.getConjugate();
  const PxReal magnitudeSquared = delta.magnitudeSquared();
  if (!PxIsFinite(magnitudeSquared) || magnitudeSquared <= 1e-12f)
    return false;
  delta *= 1.0f / PxSqrt(magnitudeSquared);
  if (delta.w < 0.0f)
    delta *= -1.0f;
  const PxVec3 imaginary(delta.x, delta.y, delta.z);
  const PxReal imaginaryMagnitude = getSafeMagnitude(imaginary);
  if (!PxIsFinite(imaginaryMagnitude))
    return false;
  if (imaginaryMagnitude <= 1e-8f) {
    rotationVector = imaginary * 2.0f;
    return rotationVector.isFinite();
  }
  const PxReal angle = 2.0f *
      PxReal(std::atan2(double(imaginaryMagnitude),
                        double(PxClamp(delta.w, -1.0f, 1.0f))));
  rotationVector = imaginary * (angle / imaginaryMagnitude);
  return rotationVector.isFinite();
}

static PxReal wrapAngleDelta(PxReal delta) {
  if (!PxIsFinite(delta))
    return delta;
  return PxReal(std::atan2(std::sin(double(delta)),
                           std::cos(double(delta))));
}

static bool getChainMomentumZ(PxU32 chainIndex, PxReal &momentumZ) {
  momentumZ = 0.0f;
  if (chainIndex >= gChains.size())
    return false;

  double accumulatedMomentumZ = 0.0;
  const ChainRecord &chain = gChains[chainIndex];
  for (PxU32 bodyIndex = 0; bodyIndex < chain.bodies.size(); ++bodyIndex) {
    const PxRigidDynamic *body = chain.bodies[bodyIndex];
    if (!body)
      return false;
    const PxReal mass = body->getMass();
    const PxVec3 velocity = body->getLinearVelocity();
    if (!PxIsFinite(mass) || mass <= 0.0f || !velocity.isFinite())
      return false;
    accumulatedMomentumZ += double(mass) * double(velocity.z);
  }
  if (!std::isfinite(accumulatedMomentumZ))
    return false;
  momentumZ = saturateMetric(accumulatedMomentumZ);
  return PxIsFinite(momentumZ);
}

static bool captureTargetResponseBaseline(ProjectileRecord &projectile) {
  if (projectile.targetChain >= gChains.size())
    return false;
  PxRigidDynamic *target = gChains[projectile.targetChain].targetBody;
  if (!target)
    return false;

  const PxVec3 targetVelocity = target->getLinearVelocity();
  PxReal chainMomentumZ = 0.0f;
  const PxReal targetMass = target->getMass();
  if (!targetVelocity.isFinite() || !PxIsFinite(targetMass) ||
      targetMass <= 0.0f ||
      !getChainMomentumZ(projectile.targetChain, chainMomentumZ))
    return false;

  projectile.targetMass = targetMass;
  projectile.preContactTargetVelocityZ = targetVelocity.z;
  projectile.preContactChainMomentumZ = chainMomentumZ;
  projectile.targetResponseBaselineValid = true;
  return true;
}

static PxReal getTargetBodyResponseFraction(const ProjectileRecord &projectile) {
  if (projectile.expectedMomentum <= 0.0f)
    return 0.0f;
  return PxMax(0.0f, -projectile.targetDeltaVelocityZ) *
         projectile.targetMass / projectile.expectedMomentum;
}

static PxReal getTargetChainResponseFraction(const ProjectileRecord &projectile) {
  if (projectile.expectedMomentum <= 0.0f)
    return 0.0f;
  return PxMax(0.0f, -projectile.chainDeltaMomentumZ) /
         projectile.expectedMomentum;
}

static PxReal getTargetResponseFraction(const ProjectileRecord &projectile) {
  const TargetResponseKind responseKind =
      getTargetResponseKind(projectile.targetChain);
  if (responseKind == eTARGET_RESPONSE_LINEAR_Z)
    return PxMax(getTargetBodyResponseFraction(projectile),
                 getTargetChainResponseFraction(projectile));
  return 0.0f;
}

static bool sampleRigidBody(PxRigidDynamic *body, PxReal &kineticEnergy) {
  kineticEnergy = 0.0f;
  if (!body)
    return false;

  const PxTransform pose = body->getGlobalPose();
  const PxVec3 linearVelocity = body->getLinearVelocity();
  const PxVec3 angularVelocity = body->getAngularVelocity();
  if (!pose.p.isFinite() || !pose.q.isFinite() || !linearVelocity.isFinite() ||
      !angularVelocity.isFinite())
    return false;

  const double quaternionNormSquared =
      double(pose.q.x) * double(pose.q.x) +
      double(pose.q.y) * double(pose.q.y) +
      double(pose.q.z) * double(pose.q.z) +
      double(pose.q.w) * double(pose.q.w);
  const PxReal quaternionError =
      saturateMetric(std::abs(quaternionNormSquared - 1.0));
  gGateStats.maxQuaternionNormError =
      PxMax(gGateStats.maxQuaternionNormError, quaternionError);
  gGateStats.maxAbsPosition =
      PxMax(gGateStats.maxAbsPosition,
            PxMax(PxAbs(pose.p.x), PxMax(PxAbs(pose.p.y), PxAbs(pose.p.z))));
  gGateStats.maxLinearSpeed =
      PxMax(gGateStats.maxLinearSpeed, getSafeMagnitude(linearVelocity));
  gGateStats.maxAngularSpeed =
      PxMax(gGateStats.maxAngularSpeed, getSafeMagnitude(angularVelocity));

  const PxReal mass = body->getMass();
  const PxVec3 inertia = body->getMassSpaceInertiaTensor();
  const PxQuat massFrame = pose.q * body->getCMassLocalPose().q;
  const PxVec3 massAngularVelocity = massFrame.rotateInv(angularVelocity);
  if (!PxIsFinite(mass) || mass <= 0.0f || !inertia.isFinite() ||
      inertia.x <= 0.0f || inertia.y <= 0.0f || inertia.z <= 0.0f ||
      !massAngularVelocity.isFinite())
    return false;

  const double linearEnergy =
      0.5 * double(mass) *
      (double(linearVelocity.x) * double(linearVelocity.x) +
       double(linearVelocity.y) * double(linearVelocity.y) +
       double(linearVelocity.z) * double(linearVelocity.z));
  const double angularEnergy =
      0.5 *
      (double(inertia.x) * double(massAngularVelocity.x) *
           double(massAngularVelocity.x) +
       double(inertia.y) * double(massAngularVelocity.y) *
           double(massAngularVelocity.y) +
       double(inertia.z) * double(massAngularVelocity.z) *
           double(massAngularVelocity.z));
  kineticEnergy = saturateMetric(linearEnergy + angularEnergy);
  return true;
}

static PxReal getLimitViolation(PxReal value, PxReal lower, PxReal upper) {
  return PxMax(0.0f, PxMax(lower - value, value - upper));
}

static void sampleJoint(ChainRecord &chain, PxJoint *joint) {
  if (!joint)
    return;
  PxConstraint *constraint = joint->getConstraint();
  if (constraint && constraint->getFlags().isSet(PxConstraintFlag::eBROKEN))
    return;

  const PxTransform relative = joint->getRelativeTransform();
  if (!relative.p.isFinite() || !relative.q.isFinite()) {
    gGateStats.nonFinite++;
    return;
  }

  PxReal anchorError = getSafeMagnitude(relative.p);
  if (chain.kind == eJOINT_PRISMATIC)
    anchorError = getSafeMagnitude(PxVec3(0.0f, relative.p.y, relative.p.z));
  chain.maxAnchorError = PxMax(chain.maxAnchorError, anchorError);
  chain.anchorErrorSquaredSum = addMetrics(
      chain.anchorErrorSquaredSum,
      saturateMetric(double(anchorError) * double(anchorError)));
  chain.anchorSamples++;
  gGateStats.maxAnchorError =
      PxMax(gGateStats.maxAnchorError, anchorError);
  gGateStats.anchorErrorSquaredSum = addMetrics(
      gGateStats.anchorErrorSquaredSum,
      saturateMetric(double(anchorError) * double(anchorError)));
  gGateStats.anchorSamples++;

  if (chain.kind == eJOINT_REVOLUTE) {
    const PxReal axisMisalignment = getJointFrameAxisMisalignment(joint);
    if (!PxIsFinite(axisMisalignment)) {
      gGateStats.nonFinite++;
    } else {
      chain.maxConstrainedAngularError =
          PxMax(chain.maxConstrainedAngularError, axisMisalignment);
      gGateStats.maxConstrainedAngularError =
          PxMax(gGateStats.maxConstrainedAngularError, axisMisalignment);
    }
  } else if (chain.kind == eJOINT_PRISMATIC || chain.kind == eJOINT_FIXED) {
    const PxReal lockedAngularError =
        2.0f * PxAcos(PxClamp(PxAbs(relative.q.w), 0.0f, 1.0f));
    chain.maxConstrainedAngularError =
        PxMax(chain.maxConstrainedAngularError, lockedAngularError);
    gGateStats.maxConstrainedAngularError =
        PxMax(gGateStats.maxConstrainedAngularError, lockedAngularError);
  }

  if (chain.kind == eJOINT_REVOLUTE) {
    PxRevoluteJoint *revolute = joint->is<PxRevoluteJoint>();
    if (revolute &&
        revolute->getRevoluteJointFlags().isSet(
            PxRevoluteJointFlag::eLIMIT_ENABLED)) {
      const PxJointAngularLimitPair limit = revolute->getLimit();
      const PxReal angle = revolute->getAngle();
      if (!PxIsFinite(angle)) {
        gGateStats.nonFinite++;
        return;
      }
      const PxReal violation = getLimitViolation(angle, limit.lower, limit.upper);
      chain.maxAngularLimitViolation =
          PxMax(chain.maxAngularLimitViolation, violation);
      gGateStats.maxRevoluteLimitViolation =
          PxMax(gGateStats.maxRevoluteLimitViolation, violation);
    }
  } else if (chain.kind == eJOINT_PRISMATIC) {
    PxPrismaticJoint *prismatic = joint->is<PxPrismaticJoint>();
    if (prismatic &&
        prismatic->getPrismaticJointFlags().isSet(
            PxPrismaticJointFlag::eLIMIT_ENABLED)) {
      const PxJointLinearLimitPair limit = prismatic->getLimit();
      const PxReal position = prismatic->getPosition();
      if (!PxIsFinite(position)) {
        gGateStats.nonFinite++;
        return;
      }
      const PxReal violation =
          getLimitViolation(position, limit.lower, limit.upper);
      chain.maxLinearLimitViolation =
          PxMax(chain.maxLinearLimitViolation, violation);
      gGateStats.maxPrismaticLimitViolation =
          PxMax(gGateStats.maxPrismaticLimitViolation, violation);
    }
  } else if (chain.kind == eJOINT_SPHERICAL) {
    PxSphericalJoint *spherical = joint->is<PxSphericalJoint>();
    if (spherical &&
        spherical->getSphericalJointFlags().isSet(
            PxSphericalJointFlag::eLIMIT_ENABLED)) {
      const PxJointLimitCone limit = spherical->getLimitCone();
      const PxReal swingY = spherical->getSwingYAngle();
      const PxReal swingZ = spherical->getSwingZAngle();
      if (!PxIsFinite(swingY) || !PxIsFinite(swingZ)) {
        gGateStats.nonFinite++;
        return;
      }
      const double normalizedY = double(swingY) / double(limit.yAngle);
      const double normalizedZ = double(swingZ) / double(limit.zAngle);
      const double normalized =
          std::sqrt(normalizedY * normalizedY + normalizedZ * normalizedZ);
      const PxReal violation = saturateMetric(
          PxMax(0.0, normalized - 1.0) *
          double(PxMin(limit.yAngle, limit.zAngle)));
      chain.maxAngularLimitViolation =
          PxMax(chain.maxAngularLimitViolation, violation);
      gGateStats.maxSphericalLimitViolation =
          PxMax(gGateStats.maxSphericalLimitViolation, violation);
    }
  } else if (chain.kind == eJOINT_D6) {
    gGateStats.maxD6LockedLinearError =
        PxMax(gGateStats.maxD6LockedLinearError, anchorError);
  }
}

static void sampleGateState() {
  PxReal totalKineticEnergy = 0.0f;
  const PxU32 lateWindow = PxMin(gHeadlessOptions.frames, PxU32(180));
  const bool inLateWindow =
      gGateStats.completedFrames + lateWindow >= gHeadlessOptions.frames;

  for (PxU32 chainIndex = 0; chainIndex < gChains.size(); ++chainIndex) {
    ChainRecord &chain = gChains[chainIndex];
    PxReal chainEnergy = 0.0f;
    for (PxU32 bodyIndex = 0; bodyIndex < chain.bodies.size(); ++bodyIndex) {
      PxReal bodyEnergy = 0.0f;
      if (!sampleRigidBody(chain.bodies[bodyIndex], bodyEnergy))
        gGateStats.nonFinite++;
      else
        chainEnergy = addMetrics(chainEnergy, bodyEnergy);
      if (inLateWindow) {
        chain.lateBodySamples++;
        if (!chain.bodies[bodyIndex]->isSleeping())
          chain.lateAwakeSamples++;
        const PxReal angularSpeed =
            getSafeMagnitude(chain.bodies[bodyIndex]->getAngularVelocity());
        if (PxIsFinite(angularSpeed)) {
          chain.lateAngularSpeedSum =
              addMetrics(chain.lateAngularSpeedSum, angularSpeed);
          chain.lateAngularSpeedMax =
              PxMax(chain.lateAngularSpeedMax, angularSpeed);
          chain.lateAngularSpeedSamples++;
        }
      }
    }
    for (PxU32 jointIndex = 0; jointIndex < chain.joints.size(); ++jointIndex)
      sampleJoint(chain, chain.joints[jointIndex]);
    if (!PxIsFinite(chainEnergy)) {
      gGateStats.nonFinite++;
      chainEnergy = 0.0f;
    }
    chain.kineticEnergy.push_back(chainEnergy);
    chain.peakKineticEnergy = PxMax(chain.peakKineticEnergy, chainEnergy);
    totalKineticEnergy = addMetrics(totalKineticEnergy, chainEnergy);
  }

  for (PxU32 i = 0; i < gProjectiles.size(); ++i) {
    PxReal ignoredEnergy = 0.0f;
    ProjectileRecord &projectile = gProjectiles[i];
    if (!projectile.actor)
      continue;
    if (!sampleRigidBody(projectile.actor, ignoredEnergy)) {
      gGateStats.nonFinite++;
      continue;
    }
    const PxU32 currentFrame = gGateStats.completedFrames + 1;
    if (projectile.firstContactFrame == PX_MAX_U32) {
      if (!captureTargetResponseBaseline(projectile))
        gGateStats.nonFinite++;
    } else if (currentFrame >= projectile.firstContactFrame &&
               currentFrame < projectile.firstContactFrame +
                                  gProjectileResponseWindowFrames) {
      PxReal currentChainMomentumZ = 0.0f;
      PxRigidDynamic *target =
          projectile.targetChain < gChains.size()
              ? gChains[projectile.targetChain].targetBody
              : NULL;
      if (!projectile.targetResponseBaselineValid || !target ||
          !target->getLinearVelocity().isFinite() ||
          !getChainMomentumZ(projectile.targetChain, currentChainMomentumZ)) {
        gGateStats.nonFinite++;
      } else {
        const PxReal targetDeltaVelocityZ = saturateMetric(
            double(target->getLinearVelocity().z) -
            double(projectile.preContactTargetVelocityZ));
        const PxReal chainDeltaMomentumZ = saturateMetric(
            double(currentChainMomentumZ) -
            double(projectile.preContactChainMomentumZ));
        if (!PxIsFinite(targetDeltaVelocityZ) ||
            !PxIsFinite(chainDeltaMomentumZ)) {
          gGateStats.nonFinite++;
        } else {
          // Incoming momentum is along -Z. Preserve the raw signed deltas so
          // zero/opposite responses cannot be hidden by a magnitude-only gate.
          projectile.targetDeltaVelocityZ =
              PxMin(projectile.targetDeltaVelocityZ, targetDeltaVelocityZ);
          projectile.chainDeltaMomentumZ =
              PxMin(projectile.chainDeltaMomentumZ, chainDeltaMomentumZ);
          projectile.maxOppositeTargetDeltaVelocityZ =
              PxMax(projectile.maxOppositeTargetDeltaVelocityZ,
                    targetDeltaVelocityZ);
          projectile.maxOppositeChainDeltaMomentumZ =
              PxMax(projectile.maxOppositeChainDeltaMomentumZ,
                    chainDeltaMomentumZ);
          projectile.targetResponseSamples++;
        }
      }
    }
    if (projectile.firstContactFrame != PX_MAX_U32 &&
        currentFrame >= projectile.firstContactFrame &&
        currentFrame <
            projectile.firstContactFrame + gProjectileResponseWindowFrames) {
      const PxVec3 currentVelocity = projectile.actor->getLinearVelocity();
      const PxVec3 velocityDelta(
          saturateMetric(double(currentVelocity.x) -
                         double(projectile.launchVelocity.x)),
          saturateMetric(double(currentVelocity.y) -
                         double(projectile.launchVelocity.y)),
          saturateMetric(double(currentVelocity.z) -
                         double(projectile.launchVelocity.z)));
      const PxReal deltaMagnitude = getSafeMagnitude(velocityDelta);
      if (!PxIsFinite(deltaMagnitude)) {
        gGateStats.nonFinite++;
      } else {
        projectile.maxVelocityDelta =
            PxMax(projectile.maxVelocityDelta, deltaMagnitude);
        // Projectiles launch along -Z. Limit the witness to the first three
        // contact frames so later ground contacts cannot create a false pass.
        projectile.maxImpactAxisVelocityDelta =
            PxMax(projectile.maxImpactAxisVelocityDelta,
                  PxMax(0.0f, velocityDelta.z));
      }
    }
  }

  if (!PxIsFinite(totalKineticEnergy)) {
    gGateStats.nonFinite++;
    totalKineticEnergy = 0.0f;
  }
  gGateStats.kineticEnergy.push_back(totalKineticEnergy);
  gGateStats.peakKineticEnergy =
      PxMax(gGateStats.peakKineticEnergy, totalKineticEnergy);
}

static void sampleSphericalConeState() {
  if (!isSphericalConeCase() || !gSphericalConeActorA ||
      !gSphericalConeActorB)
    return;

  const PxTransform poseA = gSphericalConeActorA->getGlobalPose();
  const PxTransform poseB = gSphericalConeActorB->getGlobalPose();
  PxReal ignoredEnergy = 0.0f;
  const bool bodyAFinite =
      !gSphericalConeDynamicA ||
      sampleRigidBody(gSphericalConeDynamicA, ignoredEnergy);
  const bool bodyBFinite =
      sampleRigidBody(gSphericalConeActorB, ignoredEnergy);
  const PxReal ellipseRadius =
      computeSphericalConeEllipseRadius(poseA, poseB);
  const PxReal anchorSeparation = getSafeMagnitude(poseA.p - poseB.p);
  PxVec3 totalAngularMomentum =
      gSphericalConeActorB->getAngularVelocity();
  if (gSphericalConeDynamicA)
    totalAngularMomentum +=
        gSphericalConeDynamicA->getAngularVelocity();

  if (!bodyAFinite || !bodyBFinite || !poseA.p.isFinite() ||
      !poseA.q.isFinite() || !poseB.p.isFinite() ||
      !poseB.q.isFinite() || !PxIsFinite(ellipseRadius) ||
      ellipseRadius >= PX_MAX_F32 || !PxIsFinite(anchorSeparation) ||
      !totalAngularMomentum.isFinite()) {
    gSphericalConeStats.nonFiniteSamples++;
    gGateStats.nonFinite++;
    return;
  }

  gSphericalConeStats.stateSamples++;
  gSphericalConeStats.finalEllipseRadius = ellipseRadius;
  gSphericalConeStats.minimumEllipseRadius =
      PxMin(gSphericalConeStats.minimumEllipseRadius, ellipseRadius);
  gSphericalConeStats.maximumEllipseRadius =
      PxMax(gSphericalConeStats.maximumEllipseRadius, ellipseRadius);
  gSphericalConeStats.maximumInsideDeviation =
      PxMax(gSphericalConeStats.maximumInsideDeviation,
            PxAbs(ellipseRadius -
                  gSphericalConeStats.initialEllipseRadius));
  gSphericalConeStats.maximumTotalAngularMomentum =
      PxMax(gSphericalConeStats.maximumTotalAngularMomentum,
            getSafeMagnitude(totalAngularMomentum));
  gSphericalConeStats.maximumAnchorSeparation =
      PxMax(gSphericalConeStats.maximumAnchorSeparation,
            anchorSeparation);
  const PxU32 currentFrame = gGateStats.completedFrames + 1;
  if (currentFrame + gSphericalConeLateFrames >
      gHeadlessOptions.frames) {
    gSphericalConeStats.maximumLateEllipseRadius =
        PxMax(gSphericalConeStats.maximumLateEllipseRadius,
              ellipseRadius);
    gSphericalConeStats.minimumLateEllipseRadius =
        PxMin(gSphericalConeStats.minimumLateEllipseRadius,
              ellipseRadius);
  }
}

static void launchGateProjectiles() {
  if (!usesProjectileExcitation() || !gProjectiles.empty())
    return;

  gProjectiles.reserve(gChains.size());
  for (PxU32 chainIndex = 0; chainIndex < gChains.size(); ++chainIndex) {
    ChainRecord &chain = gChains[chainIndex];
    PxRigidDynamic *target = chain.targetBody;
    if (!target) {
      gGateStats.launchFailures++;
      continue;
    }

    const PxReal offsetSign =
        ((chainIndex + gHeadlessOptions.seed) & 1u) ? 1.0f : -1.0f;
    const PxVec3 targetPosition = target->getGlobalPose().p;
    const PxTransform projectilePose(
        isEndpointProbe()
            ? targetPosition - gEndpointStats.expectedAxis * gImpactHeight
            : targetPosition +
                  PxVec3(0.0f, offsetSign * gImpactTransverseOffset,
                         gImpactHeight));
    PxRigidDynamic *projectile = PxCreateDynamic(
        *gPhysics, projectilePose, PxSphereGeometry(gImpactRadius), *gMaterial,
        1.0f);
    if (!projectile ||
        !PxRigidBodyExt::setMassAndUpdateInertia(*projectile,
                                                 target->getMass())) {
      PX_RELEASE(projectile);
      gGateStats.launchFailures++;
      continue;
    }

    projectile->setAngularDamping(0.5f);
    const PxVec3 impactVelocity =
        isEndpointProbe()
            ? gEndpointStats.expectedAxis * gImpactSpeed
            : PxVec3(0.0f, 0.0f, -gImpactSpeed);
    projectile->setLinearVelocity(target->getLinearVelocity() +
                                  impactVelocity);
    PxShape *shape = NULL;
    if (projectile->getShapes(&shape, 1) != 1 || !shape) {
      projectile->release();
      gGateStats.launchFailures++;
      continue;
    }
    PxFilterData filterData;
    filterData.word0 = eFILTER_PROJECTILE;
    filterData.word1 = chainIndex;
    shape->setSimulationFilterData(filterData);
    gScene->addActor(*projectile);
    ProjectileRecord record(projectile, chainIndex);
    record.expectedMomentum = target->getMass() * gImpactSpeed;
    record.launchVelocity = projectile->getLinearVelocity();
    if (!captureTargetResponseBaseline(record))
      gGateStats.nonFinite++;
    gProjectiles.push_back(record);
    if (isEndpointProbe()) {
      gEndpointStats.actualLaunchFrame = gGateStats.completedFrames;
      const PxReal launchMagnitude = impactVelocity.magnitude();
      gEndpointStats.launchDirection =
          launchMagnitude > 1e-12f ? impactVelocity / launchMagnitude
                                   : PxVec3(0.0f);
    }
  }
}

static void launchEndpointAngularVelocity() {
  if (!isRevoluteEndpointProbe())
    return;

  gEndpointAngularStats.launchAttempts++;
  gEndpointStats.actualLaunchFrame = gGateStats.completedFrames;
  if (!gEndpointTarget || !gEndpointRevoluteJoint ||
      !gEndpointAngularStats.requestedAngularVelocity.isFinite()) {
    gGateStats.launchFailures++;
    return;
  }

  const PxTransform baselinePose = gEndpointTarget->getGlobalPose();
  const PxVec3 baselineLinearVelocity =
      gEndpointTarget->getLinearVelocity();
  const PxVec3 baselineAngularVelocity =
      gEndpointTarget->getAngularVelocity();
  const PxReal baselineAngle = gEndpointRevoluteJoint->getAngle();
  if (!baselinePose.p.isFinite() || !baselinePose.q.isFinite() ||
      !baselineLinearVelocity.isFinite() ||
      !baselineAngularVelocity.isFinite() || !PxIsFinite(baselineAngle)) {
    gGateStats.launchFailures++;
    return;
  }

  gEndpointStats.responseBaselinePosition = baselinePose.p;
  gEndpointStats.responseBaselineVelocity = baselineLinearVelocity;
  gEndpointAngularStats.responseBaselineOrientation = baselinePose.q;
  gEndpointAngularStats.responseBaselineAngularVelocity =
      baselineAngularVelocity;
  gEndpointAngularStats.baselineRawJointAngle = baselineAngle;

  gEndpointTarget->wakeUp();
  gEndpointTarget->setAngularVelocity(
      gEndpointAngularStats.requestedAngularVelocity, true);
  gEndpointAngularStats.actualLaunchAngularVelocity =
      gEndpointTarget->getAngularVelocity();
  const PxVec3 launchVelocityError =
      gEndpointAngularStats.actualLaunchAngularVelocity -
      gEndpointAngularStats.requestedAngularVelocity;
  gEndpointAngularStats.launchVelocityError =
      getSafeMagnitude(launchVelocityError);
  const PxReal launchMagnitude =
      getSafeMagnitude(gEndpointAngularStats.actualLaunchAngularVelocity);
  if (!gEndpointAngularStats.actualLaunchAngularVelocity.isFinite() ||
      !PxIsFinite(gEndpointAngularStats.launchVelocityError) ||
      !PxIsFinite(launchMagnitude) || launchMagnitude <= 1e-12f) {
    gGateStats.launchFailures++;
    return;
  }
  gEndpointStats.launchDirection =
      gEndpointAngularStats.actualLaunchAngularVelocity / launchMagnitude;
  gEndpointAngularStats.launchWakeValid = !gEndpointTarget->isSleeping();
  gEndpointAngularStats.launchSuccesses++;
}

static void retireObservedProjectiles() {
  if (isEndpointProbe())
    return;
  for (PxU32 i = 0; i < gProjectiles.size(); ++i) {
    ProjectileRecord &projectile = gProjectiles[i];
    if (!projectile.actor || projectile.firstContactFrame == PX_MAX_U32)
      continue;
    if (gGateStats.completedFrames + 1 <
        projectile.firstContactFrame + gProjectileObservationFrames)
      continue;
    gScene->removeActor(*projectile.actor);
    projectile.actor->release();
    projectile.actor = NULL;
  }
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
  sceneDesc.gravity =
      (isEndpointProbe() || isForcePairCase() || isSphericalConeCase() ||
       isNativeBreakReactionCase() ||
       (isRevoluteMotorFamilyCase() &&
        !isRevoluteMotorContactCase()))
                          ? PxVec3(0.0f)
                          : PxVec3(0.0f, -9.81f, 0.0f);
  if (isSphericalConeCase() || isRevoluteMotorFamilyCase())
    sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
  const PxU32 dispatcherThreads =
      interactive ? 2 : gHeadlessOptions.dispatcherThreads;
  gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
  if (!gDispatcher) {
    gInitializationFailed = true;
    return;
  }
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader =
      interactive ? PxDefaultSimulationFilterShader : jointGateFilterShader;
  sceneDesc.simulationEventCallback =
      interactive ? NULL : &gSimulationCallback;
  sceneDesc.solverType = gSolverType;
  gScene = gPhysics->createScene(sceneDesc);
  if (!gScene) {
    gInitializationFailed = true;
    return;
  }

  PxPvdSceneClient *pvdClient = gScene->getScenePvdClient();
  if (pvdClient) {
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
  }

  gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
  if (!gMaterial) {
    gInitializationFailed = true;
    return;
  }

  if (!isForceReactionCase() && !isEndpointProbe() &&
      !isSphericalConeCase() && !isNativeBreakReactionCase() &&
      !isRevoluteMotorFamilyCase()) {
    PxRigidStatic *groundPlane =
        PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
    if (!groundPlane) {
      gInitializationFailed = true;
      return;
    }
    gScene->addActor(*groundPlane);
  }
  createConfiguredJointScene();
  if (isRevoluteMotorCase()) {
    if (!gRevoluteMotorBodyA || !gRevoluteMotorBodyB ||
        !gRevoluteMotorJoint || !gChains.empty() ||
        !gRevoluteMotorStats.actorOrderValid ||
        !gRevoluteMotorStats.driveEnabledReadback ||
        gRevoluteMotorStats.initialDynamicActors != 2 ||
        gRevoluteMotorStats.initialStaticActors != 0 ||
        gRevoluteMotorStats.initialConstraints != 1 ||
        gRevoluteMotorStats.targetVelocityReadback !=
            gRevoluteMotorTargetVelocity ||
        gRevoluteMotorStats.forceLimitReadback !=
            gRevoluteMotorForceLimit ||
        gRevoluteMotorBodyA->getMassSpaceInertiaTensor().x !=
            gRevoluteMotorInertiaA ||
        gRevoluteMotorBodyB->getMassSpaceInertiaTensor().x !=
            gRevoluteMotorInertiaB)
      gInitializationFailed = true;
  } else if (isRevoluteMotorLimitCase()) {
    const PxU32 expectedDynamicActors =
        isRevoluteMotorDynamicLimitCase() ? 2u : 1u;
    if (!gRevoluteMotorLimitBody || !gRevoluteMotorLimitJoint ||
        (isRevoluteMotorDynamicLimitCase() &&
         !gRevoluteMotorLimitBodyA) ||
        !gChains.empty() ||
        !gRevoluteMotorLimitStats.actorOrderValid ||
        !gRevoluteMotorLimitStats.driveEnabledReadback ||
        !gRevoluteMotorLimitStats.limitEnabledReadback ||
        gRevoluteMotorLimitStats.initialDynamicActors !=
            expectedDynamicActors ||
        gRevoluteMotorLimitStats.initialStaticActors != 0 ||
        gRevoluteMotorLimitStats.initialConstraints != 1 ||
        gRevoluteMotorLimitStats.targetVelocityReadback !=
            gRevoluteMotorLimitTargetVelocity ||
        gRevoluteMotorLimitStats.forceLimitReadback !=
            gRevoluteMotorLimitForceLimit ||
        gRevoluteMotorLimitStats.lowerLimitReadback !=
            gRevoluteMotorLimitLower ||
        gRevoluteMotorLimitStats.upperLimitReadback !=
            gRevoluteMotorLimitUpper)
      gInitializationFailed = true;
  } else if (isRevoluteMotorFreeSpinCase()) {
    RevoluteMotorFreeSpinStats &stats = gRevoluteMotorFreeSpinStats;
    const PxU32 expectedDynamicActors =
        isRevoluteMotorDynamicFreeSpinCase() ? 2u : 1u;
    if (!gRevoluteMotorFreeSpinBody ||
        (isRevoluteMotorDynamicFreeSpinCase() &&
         !gRevoluteMotorFreeSpinBodyA) ||
        !gRevoluteMotorFreeSpinJoint || !gChains.empty() ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.freeSpinEnabledReadback ||
        !stats.limitDisabledReadback ||
        stats.initialDynamicActors != expectedDynamicActors ||
        stats.initialStaticActors != 0 ||
        stats.initialConstraints != 1 ||
        stats.targetVelocityReadback !=
            gRevoluteMotorFreeSpinTargetVelocity ||
        stats.forceLimitReadback !=
            gRevoluteMotorFreeSpinForceLimit)
      gInitializationFailed = true;
  } else if (isRevoluteMotorRatioCase()) {
    const bool dynamicOffPrincipal =
        isRevoluteMotorDynamicOffPrincipalCase();
    const bool dynamicOffCenter =
        isRevoluteMotorDynamicOffCenterCase();
    const PxReal expectedDriveGearRatio =
        (dynamicOffPrincipal || dynamicOffCenter)
            ? 1.0f
            : gRevoluteMotorRatioDriveGearRatio;
    const PxVec3 expectedInertiaA =
        dynamicOffPrincipal
            ? gRevoluteMotorDynamicOffPrincipalInertiaA
            : PxVec3(gRevoluteMotorRatioInertiaA, 2.0f, 3.0f);
    const PxVec3 expectedInertiaB =
        dynamicOffPrincipal
            ? gRevoluteMotorDynamicOffPrincipalInertiaB
            : PxVec3(gRevoluteMotorRatioInertiaB, 4.0f, 5.0f);
    RevoluteMotorRatioStats &stats = gRevoluteMotorRatioStats;
    if (!gRevoluteMotorRatioBodyA || !gRevoluteMotorRatioBodyB ||
        !gRevoluteMotorRatioJoint || !gChains.empty() ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.freeSpinDisabledReadback ||
        stats.initialDynamicActors != 2 ||
        stats.initialStaticActors != 0 ||
        stats.initialConstraints != 1 ||
        stats.targetVelocityReadback !=
            gRevoluteMotorRatioTargetVelocity ||
        stats.forceLimitReadback !=
            gRevoluteMotorRatioForceLimit ||
        stats.driveGearRatioReadback !=
            expectedDriveGearRatio ||
        (gRevoluteMotorRatioBodyA->getMassSpaceInertiaTensor() -
         expectedInertiaA)
                .magnitude() >
            1e-5f ||
        (gRevoluteMotorRatioBodyB->getMassSpaceInertiaTensor() -
         expectedInertiaB)
                .magnitude() >
            1e-5f ||
        (dynamicOffCenter &&
         (PxAbs(gRevoluteMotorRatioBodyA->getMass() - 1.0f) >
              1e-6f ||
          PxAbs(gRevoluteMotorRatioBodyB->getMass() - 1.0f) >
              1e-6f))) {
      gInitializationFailed = true;
    }
  } else if (isRevoluteMotorContactCase()) {
    RevoluteMotorContactStats &stats =
        gRevoluteMotorContactStats;
    if (!gRevoluteMotorContactBodyA ||
        !gRevoluteMotorContactBodyB ||
        !gRevoluteMotorContactGround ||
        !gRevoluteMotorContactJoint || !gChains.empty() ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        stats.initialDynamicActors != 2 ||
        stats.initialStaticActors != 1 ||
        stats.initialConstraints != 1 ||
        stats.targetVelocityReadback !=
            gRevoluteMotorContactTargetVelocity ||
        stats.forceLimitReadback !=
            gRevoluteMotorContactForceLimit ||
        gRevoluteMotorContactBodyA->getMass() != 1.0f ||
        gRevoluteMotorContactBodyB->getMass() != 1.0f)
      gInitializationFailed = true;
  } else if (isRevoluteMotorKinematicCase()) {
    RevoluteMotorKinematicStats &stats =
        gRevoluteMotorKinematicStats;
    if (!gRevoluteMotorKinematicBody ||
        !gRevoluteMotorKinematicDynamicBody ||
        !gRevoluteMotorKinematicJoint || !gChains.empty() ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.kinematicFlagReadback ||
        stats.initialDynamicActors != 2 ||
        stats.initialStaticActors != 0 ||
        stats.initialConstraints != 1 ||
        stats.targetVelocityReadback !=
            gRevoluteMotorKinematicTargetVelocity ||
        stats.forceLimitReadback !=
            gRevoluteMotorKinematicForceLimit)
      gInitializationFailed = true;
  } else if (isRevoluteMotorOffPrincipalCase()) {
    RevoluteMotorOffPrincipalStats &stats =
        gRevoluteMotorOffPrincipalStats;
    if (!gRevoluteMotorOffPrincipalBody ||
        !gRevoluteMotorOffPrincipalJoint || !gChains.empty() ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        stats.initialDynamicActors != 1 ||
        stats.initialStaticActors != 0 ||
        stats.initialConstraints != 1 ||
        stats.targetVelocityReadback !=
            gRevoluteMotorOffPrincipalTargetVelocity ||
        stats.forceLimitReadback !=
            gRevoluteMotorOffPrincipalForceLimit ||
        stats.initialOffPrincipalResponse <
            gRevoluteMotorOffPrincipalResponseMinimum)
      gInitializationFailed = true;
  } else if (isRevoluteMotorOffCenterCase()) {
    RevoluteMotorOffCenterStats &stats =
        gRevoluteMotorOffCenterStats;
    if (!gRevoluteMotorOffCenterBody ||
        !gRevoluteMotorOffCenterJoint || !gChains.empty() ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        stats.initialDynamicActors != 1 ||
        stats.initialStaticActors != 0 ||
        stats.initialConstraints != 1 ||
        stats.targetVelocityReadback !=
            gRevoluteMotorOffCenterTargetVelocity ||
        stats.forceLimitReadback !=
            gRevoluteMotorOffCenterForceLimit ||
        stats.initialPerpendicularLeverArm <
            gRevoluteMotorOffCenterLeverArmMinimum ||
        (isRevoluteMotorSpatialCase() &&
         stats.initialOffPrincipalResponse <
             gRevoluteMotorOffPrincipalResponseMinimum))
      gInitializationFailed = true;
  } else if (isSphericalConeCase()) {
    const PxU32 expectedDynamicActors =
        gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC ? 2u : 1u;
    const PxU32 expectedStaticActors =
        gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC ? 0u : 1u;
    if (!gSphericalConeActorA || !gSphericalConeActorB ||
        !gSphericalConeJoint || !gChains.empty() ||
        !gSphericalConeStats.actorOrderValid ||
        !gSphericalConeStats.limitEnabledReadback ||
        gSphericalConeStats.initialDynamicActors != expectedDynamicActors ||
        gSphericalConeStats.initialStaticActors != expectedStaticActors ||
        gSphericalConeStats.initialConstraints != 1 ||
        !PxIsFinite(gSphericalConeStats.initialEllipseRadius) ||
        gSphericalConeStats.initialEllipseRadius >= PX_MAX_F32)
      gInitializationFailed = true;
  } else if (isNativeBreakReactionCase()) {
    const PxReal expectedLinearThreshold =
        !isNativeAngularReactionCase() &&
                gHeadlessCase == eCASE_NATIVE_NO_BREAK
            ? gNativeHighBreakThreshold
            : (!isNativeAngularReactionCase() &&
                       gHeadlessCase == eCASE_NATIVE_BREAK
                   ? gNativeLowBreakThreshold
                   : PX_MAX_F32);
    const PxReal expectedAngularThreshold =
        isNativeAngularReactionCase() &&
                gHeadlessCase == eCASE_NATIVE_NO_BREAK
            ? gNativeHighBreakThreshold
            : (isNativeAngularReactionCase() &&
                       gHeadlessCase == eCASE_NATIVE_BREAK
                   ? gNativeLowBreakThreshold
                   : PX_MAX_F32);
    if (!gNativeBreakReactionBody || !gNativeBreakReactionJoint ||
        !gChains.empty() ||
        !gNativeBreakReactionStats.actorOrderValid ||
        gNativeBreakReactionStats.initialDynamicActors != 1 ||
        gNativeBreakReactionStats.initialStaticActors != 0 ||
        gNativeBreakReactionStats.initialConstraints != 1 ||
        gNativeBreakReactionBody->getMass() != 1.0f ||
        gNativeBreakReactionBody->getMassSpaceInertiaTensor() !=
            PxVec3(1.0f) ||
        gNativeBreakReactionStats.breakForceReadback !=
            expectedLinearThreshold ||
        gNativeBreakReactionStats.breakTorqueReadback !=
            expectedAngularThreshold ||
        (gImpactJointKind != eJOINT_PRISMATIC &&
         gImpactJointKind != eJOINT_REVOLUTE))
      gInitializationFailed = true;
  } else if (isForceReactionCase()) {
    const PxU32 expectedDynamicActors = isForcePairCase() ? 2u : 1u;
    if (!gForceStaticBody || !gForceStaticJoint ||
        (isForcePairCase() && !gForcePairBody1) || !gChains.empty() ||
        gForceStaticStats.topologyDynamicActors != expectedDynamicActors ||
        gForceStaticStats.topologyStaticActors != 0 ||
        gForceStaticStats.topologyConstraints != 1 ||
        !PxIsFinite(gForceStaticStats.actualMass) ||
        PxAbs(gForceStaticStats.actualMass - gForceStaticMass) > 1e-5f ||
        !gForceStaticStats.gravity.isFinite() ||
        PxAbs(gForceStaticStats.gravity.x) > 1e-6f ||
        PxAbs(gForceStaticStats.gravity.y -
              (isForcePairCase() ? 0.0f : -gGravityMagnitude)) > 1e-6f ||
        PxAbs(gForceStaticStats.gravity.z) > 1e-6f ||
        !PxIsFinite(gForceStaticStats.expectedWeight) ||
        gForceStaticStats.expectedWeight <= 0.0f ||
        !gForceStaticStats.anchorOffset.isFinite() ||
        !gForceStaticStats.expectedTorque.isFinite() ||
        !PxIsFinite(gForceStaticStats.actor0FramePositionError) ||
        !PxIsFinite(gForceStaticStats.actor1FramePositionError) ||
        gForceStaticStats.actor0FramePositionError > 1e-6f ||
        gForceStaticStats.actor1FramePositionError > 1e-6f ||
        (isForceStaticCase() &&
         (getSafeMagnitude(gForceStaticStats.anchorOffset) > 1e-6f ||
          getSafeMagnitude(gForceStaticStats.expectedTorque) > 1e-6f)) ||
        (isForceOffsetCase() &&
         (getSafeMagnitude(gForceStaticStats.anchorOffset -
                           gForceOffsetAnchor) > 1e-6f ||
          getSafeMagnitude(gForceStaticStats.expectedTorque) <= 0.0f)) ||
        (isForcePairCase() &&
         (getSafeMagnitude(gForceStaticStats.appliedForceActor0 +
                           gForceStaticStats.appliedForceActor1) > 1e-6f ||
          getSafeMagnitude(gForceStaticStats.expectedLinearForce +
                           gForceStaticStats.appliedForceActor0) > 1e-6f ||
          !gForceStaticStats.pairActorOrderValid ||
          PxAbs(gForceStaticStats.pairInitialSeparation -
                gForcePairSeparation) > 1e-6f ||
          getSafeMagnitude(gForceStaticStats.expectedTorque) > 1e-6f)))
      gInitializationFailed = true;
  } else {
    if (gChains.empty())
      gInitializationFailed = true;
    for (PxU32 i = 0; i < gChains.size(); ++i) {
      if (!gChains[i].targetBody || gChains[i].bodies.empty() ||
          gChains[i].joints.empty())
        gInitializationFailed = true;
    }
  }
}

static void sampleNativeBreakReactionState() {
  if (!gNativeBreakReactionBody || !gNativeBreakReactionJoint)
    return;

  NativeBreakReactionStats &stats = gNativeBreakReactionStats;
  stats.stateSamples++;
  const PxU32 currentFrame = gGateStats.completedFrames + 1;
  const PxTransform bodyPose =
      gNativeBreakReactionBody->getGlobalPose();
  const PxVec3 linearVelocity =
      gNativeBreakReactionBody->getLinearVelocity();
  const PxVec3 angularVelocity =
      gNativeBreakReactionBody->getAngularVelocity();
  const PxTransform relative =
      gNativeBreakReactionJoint->getRelativeTransform();
  const PxReal positionError =
      gImpactJointKind == eJOINT_PRISMATIC
          ? PxVec3(0.0f, relative.p.y, relative.p.z).magnitude()
          : relative.p.magnitude();
  const PxReal rotationError =
      2.0f * PxAcos(PxClamp(PxAbs(relative.q.w), 0.0f, 1.0f));
  const PxReal linearSpeed = linearVelocity.magnitude();
  const PxReal angularSpeed = angularVelocity.magnitude();
  if (!bodyPose.p.isFinite() || !bodyPose.q.isFinite() ||
      !linearVelocity.isFinite() || !angularVelocity.isFinite() ||
      !relative.p.isFinite() || !relative.q.isFinite() ||
      !PxIsFinite(positionError) || !PxIsFinite(rotationError) ||
      !PxIsFinite(linearSpeed) || !PxIsFinite(angularSpeed)) {
    stats.nonFiniteSamples++;
    gGateStats.nonFinite++;
    return;
  }
  stats.maximumPositionError =
      PxMax(stats.maximumPositionError, positionError);
  stats.maximumRotationError =
      PxMax(stats.maximumRotationError, rotationError);
  stats.maximumLinearSpeed =
      PxMax(stats.maximumLinearSpeed, linearSpeed);
  stats.maximumAngularSpeed =
      PxMax(stats.maximumAngularSpeed, angularSpeed);

  PxConstraint *constraint = gNativeBreakReactionJoint->getConstraint();
  if (!constraint)
    return;
  PxVec3 linearForce(0.0f), angularForce(0.0f);
  constraint->getForce(linearForce, angularForce);
  stats.forceReads++;
  const bool broken =
      constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
  const PxI32 breakableIndex = findBreakableJoint(constraint);
  if (broken && breakableIndex >= 0 &&
      PxU32(breakableIndex) < gBreakPollReported.size() &&
      !gBreakPollReported[PxU32(breakableIndex)]) {
    gBreakPollReported[PxU32(breakableIndex)] = 1;
    stats.firstBrokenFrame = currentFrame;
    stats.brokenPollCount++;
  }
  if (!broken && currentFrame > gNativeReactionWarmupFrames) {
    stats.steadyMaximumPositionError =
        PxMax(stats.steadyMaximumPositionError, positionError);
    stats.steadyMaximumRotationError =
        PxMax(stats.steadyMaximumRotationError, rotationError);
    stats.steadyMaximumLinearSpeed =
        PxMax(stats.steadyMaximumLinearSpeed, linearSpeed);
    stats.steadyMaximumAngularSpeed =
        PxMax(stats.steadyMaximumAngularSpeed, angularSpeed);
  }
  if (broken || currentFrame <= gNativeReactionWarmupFrames)
    return;

  const PxReal linearMagnitude = linearForce.magnitude();
  const PxReal angularMagnitude = angularForce.magnitude();
  if (!linearForce.isFinite() || !angularForce.isFinite() ||
      !PxIsFinite(linearMagnitude) || !PxIsFinite(angularMagnitude)) {
    stats.nonFiniteSamples++;
    gGateStats.nonFinite++;
    return;
  }
  stats.reactionSamples++;
  stats.linearForceSum += linearForce;
  stats.angularForceSum += angularForce;
  stats.linearMagnitudeSum += linearMagnitude;
  stats.angularMagnitudeSum += angularMagnitude;
}

static PxVec3 getNativeMeanReactionVector() {
  if (!gNativeBreakReactionStats.reactionSamples)
    return PxVec3(0.0f);
  return (isNativeAngularReactionCase()
              ? gNativeBreakReactionStats.angularForceSum
              : gNativeBreakReactionStats.linearForceSum) /
         PxReal(gNativeBreakReactionStats.reactionSamples);
}

static PxReal getNativeMeanReactionMagnitude() {
  if (!gNativeBreakReactionStats.reactionSamples)
    return 0.0f;
  return (isNativeAngularReactionCase()
              ? gNativeBreakReactionStats.angularMagnitudeSum
              : gNativeBreakReactionStats.linearMagnitudeSum) /
         PxReal(gNativeBreakReactionStats.reactionSamples);
}

static PxVec3 getNativeExpectedReactionVector() {
  return isNativeAngularReactionCase() ? gNativeAngularLoad
                                       : gNativeLinearLoad;
}

static PxReal getNativeReactionDirectionDot() {
  const PxVec3 meanReaction = getNativeMeanReactionVector();
  const PxVec3 expectedReaction = getNativeExpectedReactionVector();
  const PxReal denominator =
      meanReaction.magnitude() * expectedReaction.magnitude();
  return denominator > 1e-12f
             ? meanReaction.dot(expectedReaction) / denominator
             : 0.0f;
}

static PxReal getNativeReactionOrthogonalRatio() {
  const PxVec3 meanReaction = getNativeMeanReactionVector();
  const PxVec3 expectedReaction = getNativeExpectedReactionVector();
  const PxReal expectedMagnitude = expectedReaction.magnitude();
  if (expectedMagnitude <= 1e-12f)
    return 0.0f;
  const PxVec3 expectedAxis = expectedReaction / expectedMagnitude;
  const PxVec3 orthogonal =
      meanReaction - expectedAxis * meanReaction.dot(expectedAxis);
  return orthogonal.magnitude() / expectedMagnitude;
}

static void sampleForceStaticState() {
  if (!gForceStaticBody || !gForceStaticJoint)
    return;

  gForceStaticStats.stateSamples++;
  const PxU32 currentFrame = gGateStats.completedFrames + 1;
  const auto sampleBodyState = [&](PxRigidDynamic *body,
                                   const PxTransform &initialPose) {
    PxReal ignoredEnergy = 0.0f;
    const bool rigidBodyFinite = sampleRigidBody(body, ignoredEnergy);
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 linearVelocity = body->getLinearVelocity();
    const PxVec3 angularVelocity = body->getAngularVelocity();
    if (!rigidBodyFinite || !pose.p.isFinite() || !pose.q.isFinite() ||
        !linearVelocity.isFinite() || !angularVelocity.isFinite()) {
      gForceStaticStats.nonFiniteStateSamples++;
      gGateStats.nonFinite++;
      return;
    }
    const PxReal positionError =
        getSafeMagnitude(pose.p - initialPose.p);
    const PxReal orientationDot =
        PxClamp(PxAbs(pose.q.dot(initialPose.q)), 0.0f, 1.0f);
    const PxReal rotationError = 2.0f * PxAcos(orientationDot);
    const PxReal linearSpeed = getSafeMagnitude(linearVelocity);
    const PxReal angularSpeed = getSafeMagnitude(angularVelocity);
    if (positionError > gForceStaticStats.maxPositionError) {
      gForceStaticStats.maxPositionError = positionError;
      gForceStaticStats.maxPositionErrorFrame = currentFrame;
    }
    gForceStaticStats.maxRotationError =
        PxMax(gForceStaticStats.maxRotationError, rotationError);
    if (linearSpeed > gForceStaticStats.maxLinearSpeed) {
      gForceStaticStats.maxLinearSpeed = linearSpeed;
      gForceStaticStats.maxLinearSpeedFrame = currentFrame;
    }
    if (currentFrame > gForceStaticStats.steadyBeginFrame &&
        linearSpeed > gForceStaticStats.steadyMaxLinearSpeed) {
      gForceStaticStats.steadyMaxLinearSpeed = linearSpeed;
      gForceStaticStats.steadyMaxLinearSpeedFrame = currentFrame;
    }
    gForceStaticStats.finalPositionError = positionError;
    gForceStaticStats.finalLinearSpeed = linearSpeed;
    gForceStaticStats.maxAngularSpeed =
        PxMax(gForceStaticStats.maxAngularSpeed, angularSpeed);
  };
  sampleBodyState(gForceStaticBody, gForceStaticInitialPose);
  if (isForcePairCase() && gForcePairBody1) {
    sampleBodyState(gForcePairBody1, gForcePairBody1InitialPose);
    const PxTransform actor0Pose = gForceStaticBody->getGlobalPose();
    const PxTransform actor1Pose = gForcePairBody1->getGlobalPose();
    const PxVec3 actor0Velocity = gForceStaticBody->getLinearVelocity();
    const PxVec3 actor1Velocity = gForcePairBody1->getLinearVelocity();
    const PxReal actor0Mass = gForceStaticBody->getMass();
    const PxReal actor1Mass = gForcePairBody1->getMass();
    const PxReal totalMass = actor0Mass + actor1Mass;
    const PxVec3 totalMomentum =
        actor0Velocity * actor0Mass + actor1Velocity * actor1Mass;
    const PxReal totalMomentumMagnitude = getSafeMagnitude(totalMomentum);
    const PxReal separation =
        getSafeMagnitude(actor1Pose.p - actor0Pose.p);
    const PxReal separationError =
        PxAbs(separation - gForceStaticStats.pairInitialSeparation);
    const PxReal relativeSpeed =
        getSafeMagnitude(actor1Velocity - actor0Velocity);
    const PxVec3 centerOfMass =
        totalMass > 0.0f
            ? (actor0Pose.p * actor0Mass + actor1Pose.p * actor1Mass) /
                  totalMass
            : PxVec3(0.0f);
    const PxReal centerOfMassError = getSafeMagnitude(
        centerOfMass - gForceStaticStats.pairInitialCenterOfMass);
    if (!actor0Pose.p.isFinite() || !actor1Pose.p.isFinite() ||
        !actor0Velocity.isFinite() || !actor1Velocity.isFinite() ||
        !totalMomentum.isFinite() || !centerOfMass.isFinite() ||
        !PxIsFinite(totalMomentumMagnitude) || !PxIsFinite(separation) ||
        !PxIsFinite(separationError) || !PxIsFinite(relativeSpeed) ||
        !PxIsFinite(centerOfMassError)) {
      gForceStaticStats.nonFiniteStateSamples++;
      gGateStats.nonFinite++;
    } else {
      gForceStaticStats.pairMaxSeparationError = PxMax(
          gForceStaticStats.pairMaxSeparationError, separationError);
      gForceStaticStats.pairFinalSeparation = separation;
      gForceStaticStats.pairMaxRelativeSpeed = PxMax(
          gForceStaticStats.pairMaxRelativeSpeed, relativeSpeed);
      gForceStaticStats.pairFinalRelativeSpeed = relativeSpeed;
      gForceStaticStats.pairMaxCenterOfMassError = PxMax(
          gForceStaticStats.pairMaxCenterOfMassError, centerOfMassError);
      gForceStaticStats.pairFinalCenterOfMassError = centerOfMassError;
      gForceStaticStats.pairMaxTotalMomentum = PxMax(
          gForceStaticStats.pairMaxTotalMomentum, totalMomentumMagnitude);
      gForceStaticStats.pairFinalTotalMomentum = totalMomentumMagnitude;
      gForceStaticStats.pairFinalTotalMomentumVector = totalMomentum;
      gForceStaticStats.pairActor0FinalPosition = actor0Pose.p;
      gForceStaticStats.pairActor1FinalPosition = actor1Pose.p;
      gForceStaticStats.pairActor0FinalVelocity = actor0Velocity;
      gForceStaticStats.pairActor1FinalVelocity = actor1Velocity;
    }
  }

  PxConstraint *constraint = gForceStaticJoint->getConstraint();
  if (!constraint)
    return;
  PxVec3 linearForce(0.0f), angularForce(0.0f);
  constraint->getForce(linearForce, angularForce);
  gForceStaticStats.forceReads++;
  const bool inSteadyWindow =
      currentFrame > gForceStaticStats.steadyBeginFrame;
  if (inSteadyWindow)
    gForceStaticStats.steadySampleAttempts++;
  const PxReal linearMagnitude = getSafeMagnitude(linearForce);
  const PxReal angularMagnitude = getSafeMagnitude(angularForce);
  if (!linearForce.isFinite() || !angularForce.isFinite() ||
      !PxIsFinite(linearMagnitude) || !PxIsFinite(angularMagnitude)) {
    gForceStaticStats.nonFiniteForceSamples++;
    if (inSteadyWindow)
      gForceStaticStats.nonFiniteSteadyForceSamples++;
    gGateStats.nonFinite++;
    return;
  }

  if (!inSteadyWindow)
    return;
  gForceStaticStats.linearForceSum.x = saturateMetric(
      double(gForceStaticStats.linearForceSum.x) + double(linearForce.x));
  gForceStaticStats.linearForceSum.y = saturateMetric(
      double(gForceStaticStats.linearForceSum.y) + double(linearForce.y));
  gForceStaticStats.linearForceSum.z = saturateMetric(
      double(gForceStaticStats.linearForceSum.z) + double(linearForce.z));
  gForceStaticStats.angularForceSum.x = saturateMetric(
      double(gForceStaticStats.angularForceSum.x) + double(angularForce.x));
  gForceStaticStats.angularForceSum.y = saturateMetric(
      double(gForceStaticStats.angularForceSum.y) + double(angularForce.y));
  gForceStaticStats.angularForceSum.z = saturateMetric(
      double(gForceStaticStats.angularForceSum.z) + double(angularForce.z));
  gForceStaticStats.linearMagnitudeSum =
      addMetrics(gForceStaticStats.linearMagnitudeSum, linearMagnitude);
  gForceStaticStats.angularMagnitudeSum =
      addMetrics(gForceStaticStats.angularMagnitudeSum, angularMagnitude);
  gForceStaticStats.orthogonalMagnitudeSquaredSum +=
      double(linearForce.x) * double(linearForce.x) +
      double(linearForce.z) * double(linearForce.z);
  gForceStaticStats.angularMagnitudeSquaredSum +=
      double(angularMagnitude) * double(angularMagnitude);
  const PxReal expectedTorqueMagnitude =
      getSafeMagnitude(gForceStaticStats.expectedTorque);
  if (expectedTorqueMagnitude > 0.0f) {
    const PxVec3 torqueAxis =
        gForceStaticStats.expectedTorque / expectedTorqueMagnitude;
    const PxVec3 angularOrthogonal =
        angularForce - torqueAxis * angularForce.dot(torqueAxis);
    const PxReal angularOrthogonalMagnitude =
        getSafeMagnitude(angularOrthogonal);
    gForceStaticStats.angularOrthogonalMagnitudeSquaredSum +=
        double(angularOrthogonalMagnitude) *
        double(angularOrthogonalMagnitude);
  }
  gForceStaticStats.maxAngularMagnitude =
      PxMax(gForceStaticStats.maxAngularMagnitude, angularMagnitude);
  gForceStaticStats.steadySamples++;
}

static void sampleEndpointAngularState() {
  gEndpointStats.stateSampleAttempts++;
  if (!gEndpointTarget || !gEndpointRevoluteJoint)
    return;
  gEndpointStats.stateSamples++;

  const PxU32 currentFrame = gGateStats.completedFrames + 1;
  const bool launched =
      gEndpointAngularStats.launchSuccesses == 1 &&
      gEndpointStats.actualLaunchFrame != PX_MAX_U32;
  const PxU32 responseFirstFrame =
      launched ? gEndpointStats.actualLaunchFrame + 1 : PX_MAX_U32;
  const bool inResponseWindow =
      launched && currentFrame >= responseFirstFrame &&
      currentFrame <
          responseFirstFrame + gEndpointAngularResponseWindowFrames;
  const PxU32 responseIndex =
      inResponseWindow ? currentFrame - responseFirstFrame : PX_MAX_U32;
  if (inResponseWindow)
    gEndpointStats.responseSampleAttempts++;

  const PxTransform pose = gEndpointTarget->getGlobalPose();
  const PxVec3 linearVelocity = gEndpointTarget->getLinearVelocity();
  const PxVec3 angularVelocity = gEndpointTarget->getAngularVelocity();
  if (!pose.p.isFinite() || !pose.q.isFinite() ||
      !linearVelocity.isFinite() || !angularVelocity.isFinite()) {
    gEndpointStats.nonFiniteStateSamples++;
    if (inResponseWindow)
      gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }

  const PxTransform relative =
      gEndpointRevoluteJoint->getRelativeTransform();
  if (!relative.p.isFinite() || !relative.q.isFinite()) {
    gEndpointStats.nonFiniteStateSamples++;
    if (inResponseWindow)
      gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }
  const PxReal anchorError = getSafeMagnitude(relative.p);
  if (!PxIsFinite(anchorError)) {
    gEndpointStats.nonFiniteStateSamples++;
    if (inResponseWindow)
      gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }
  gEndpointAngularStats.maxAnchorError =
      PxMax(gEndpointAngularStats.maxAnchorError, anchorError);

  PxVec3 actor0Axis, actor1Axis;
  getJointWorldAxes(gEndpointRevoluteJoint, actor0Axis, actor1Axis);
  if (!actor0Axis.isFinite() || !actor1Axis.isFinite()) {
    gEndpointStats.nonFiniteStateSamples++;
    if (inResponseWindow)
      gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }
  const PxReal axisDot =
      PxClamp(actor0Axis.dot(actor1Axis), -1.0f, 1.0f);
  const PxReal axisMisalignment = PxAcos(axisDot);
  if (!PxIsFinite(axisMisalignment)) {
    gEndpointStats.nonFiniteStateSamples++;
    if (inResponseWindow)
      gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }
  gEndpointAngularStats.maxAxisMisalignment =
      PxMax(gEndpointAngularStats.maxAxisMisalignment, axisMisalignment);

  if (!launched) {
    PxVec3 orientationDriftVector;
    const PxReal positionDrift =
        getSafeMagnitude(pose.p - gEndpointStats.initialTargetPosition);
    const PxReal linearSpeed = getSafeMagnitude(linearVelocity);
    const PxReal angularSpeed = getSafeMagnitude(angularVelocity);
    if (!getShortestRotationVector(
            pose.q, gEndpointAngularStats.initialTargetOrientation,
            orientationDriftVector) ||
        !PxIsFinite(positionDrift) || !PxIsFinite(linearSpeed) ||
        !PxIsFinite(angularSpeed)) {
      gEndpointStats.nonFiniteStateSamples++;
      gGateStats.nonFinite++;
      return;
    }
    const PxReal orientationDrift =
        getSafeMagnitude(orientationDriftVector);
    if (!PxIsFinite(orientationDrift)) {
      gEndpointStats.nonFiniteStateSamples++;
      gGateStats.nonFinite++;
      return;
    }
    gEndpointStats.maxPrecontactPositionDrift = PxMax(
        gEndpointStats.maxPrecontactPositionDrift, positionDrift);
    gEndpointStats.maxPrecontactSpeed =
        PxMax(gEndpointStats.maxPrecontactSpeed, linearSpeed);
    gEndpointAngularStats.maxPrelaunchOrientationDrift = PxMax(
        gEndpointAngularStats.maxPrelaunchOrientationDrift,
        orientationDrift);
    gEndpointAngularStats.maxPrelaunchAngularSpeed = PxMax(
        gEndpointAngularStats.maxPrelaunchAngularSpeed, angularSpeed);
    gEndpointStats.responseBaselinePosition = pose.p;
    gEndpointStats.responseBaselineVelocity = linearVelocity;
    gEndpointAngularStats.responseBaselineOrientation = pose.q;
    gEndpointAngularStats.responseBaselineAngularVelocity = angularVelocity;
    const PxReal baselineAngle = gEndpointRevoluteJoint->getAngle();
    if (!PxIsFinite(baselineAngle)) {
      gEndpointStats.nonFiniteStateSamples++;
      gEndpointAngularStats.nonFiniteApiSamples++;
      gGateStats.nonFinite++;
      return;
    }
    gEndpointAngularStats.baselineRawJointAngle = baselineAngle;
    gEndpointStats.responseBaselineSamples++;
    return;
  }

  if (!inResponseWindow)
    return;

  PxVec3 rotationVector;
  const PxVec3 angularVelocityDelta =
      angularVelocity -
      gEndpointAngularStats.responseBaselineAngularVelocity;
  const PxReal rawJointAngle = gEndpointRevoluteJoint->getAngle();
  const PxReal apiVelocityMagnitude =
      gEndpointRevoluteJoint->getVelocity();
  if (!getShortestRotationVector(
          pose.q, gEndpointAngularStats.responseBaselineOrientation,
          rotationVector) ||
      !angularVelocityDelta.isFinite() || !PxIsFinite(rawJointAngle) ||
      !PxIsFinite(apiVelocityMagnitude)) {
    gEndpointStats.nonFiniteStateSamples++;
    gEndpointStats.nonFiniteResponseSamples++;
    gEndpointAngularStats.nonFiniteApiSamples++;
    gGateStats.nonFinite++;
    return;
  }

  const PxReal signedRotation =
      rotationVector.dot(gEndpointStats.expectedAxis);
  const PxReal signedAngularVelocity =
      angularVelocityDelta.dot(gEndpointStats.expectedAxis);
  const PxVec3 rotationOrthogonal =
      rotationVector - gEndpointStats.expectedAxis * signedRotation;
  const PxVec3 angularVelocityOrthogonal =
      angularVelocityDelta -
      gEndpointStats.expectedAxis * signedAngularVelocity;
  const PxReal rotationOrthogonalMagnitude =
      getSafeMagnitude(rotationOrthogonal);
  const PxReal angularVelocityOrthogonalMagnitude =
      getSafeMagnitude(angularVelocityOrthogonal);
  const PxReal rawJointAngleDelta =
      wrapAngleDelta(rawJointAngle -
                     gEndpointAngularStats.baselineRawJointAngle);
  const PxReal endpointAngleSign =
      gEndpointKind == eENDPOINT_FORWARD ? 1.0f : -1.0f;
  const PxReal semanticJointAngleDelta =
      endpointAngleSign * rawJointAngleDelta;
  const PxReal jointAnglePoseMismatch =
      PxAbs(semanticJointAngleDelta - signedRotation);
  const PxReal bodyAngularSpeed = getSafeMagnitude(angularVelocity);
  const PxReal apiVelocityMagnitudeMismatch =
      PxAbs(apiVelocityMagnitude - bodyAngularSpeed);
  if (!PxIsFinite(signedRotation) ||
      !PxIsFinite(signedAngularVelocity) ||
      !PxIsFinite(rotationOrthogonalMagnitude) ||
      !PxIsFinite(angularVelocityOrthogonalMagnitude) ||
      !PxIsFinite(rawJointAngleDelta) ||
      !PxIsFinite(semanticJointAngleDelta) ||
      !PxIsFinite(jointAnglePoseMismatch) ||
      !PxIsFinite(bodyAngularSpeed) ||
      !PxIsFinite(apiVelocityMagnitudeMismatch)) {
    gEndpointStats.nonFiniteStateSamples++;
    gEndpointStats.nonFiniteResponseSamples++;
    gEndpointAngularStats.nonFiniteApiSamples++;
    gGateStats.nonFinite++;
    return;
  }

  gEndpointStats.responseSamples++;
  gEndpointAngularStats.apiSamples++;
  gEndpointAngularStats.lastRawJointAngleDelta = rawJointAngleDelta;
  gEndpointAngularStats.lastSemanticJointAngleDelta =
      semanticJointAngleDelta;
  if (responseIndex < gEndpointAngularSettleFrames)
    return;

  gEndpointAngularStats.tailSamples++;
  gEndpointAngularStats.minSignedRotation =
      PxMin(gEndpointAngularStats.minSignedRotation, signedRotation);
  gEndpointAngularStats.minSignedAngularVelocity = PxMin(
      gEndpointAngularStats.minSignedAngularVelocity,
      signedAngularVelocity);
  gEndpointAngularStats.responseRotationVectorSum.x = saturateMetric(
      double(gEndpointAngularStats.responseRotationVectorSum.x) +
      double(rotationVector.x));
  gEndpointAngularStats.responseRotationVectorSum.y = saturateMetric(
      double(gEndpointAngularStats.responseRotationVectorSum.y) +
      double(rotationVector.y));
  gEndpointAngularStats.responseRotationVectorSum.z = saturateMetric(
      double(gEndpointAngularStats.responseRotationVectorSum.z) +
      double(rotationVector.z));
  gEndpointAngularStats.responseAngularVelocityDeltaSum.x = saturateMetric(
      double(gEndpointAngularStats.responseAngularVelocityDeltaSum.x) +
      double(angularVelocityDelta.x));
  gEndpointAngularStats.responseAngularVelocityDeltaSum.y = saturateMetric(
      double(gEndpointAngularStats.responseAngularVelocityDeltaSum.y) +
      double(angularVelocityDelta.y));
  gEndpointAngularStats.responseAngularVelocityDeltaSum.z = saturateMetric(
      double(gEndpointAngularStats.responseAngularVelocityDeltaSum.z) +
      double(angularVelocityDelta.z));
  gEndpointAngularStats.rotationOrthogonalSquaredSum +=
      double(rotationOrthogonalMagnitude) *
      double(rotationOrthogonalMagnitude);
  gEndpointAngularStats.angularVelocityOrthogonalSquaredSum +=
      double(angularVelocityOrthogonalMagnitude) *
      double(angularVelocityOrthogonalMagnitude);
  gEndpointAngularStats.rawJointAngleDeltaSum +=
      double(rawJointAngleDelta);
  gEndpointAngularStats.semanticJointAngleDeltaSum +=
      double(semanticJointAngleDelta);
  gEndpointAngularStats.apiVelocityMagnitudeSum +=
      double(apiVelocityMagnitude);
  gEndpointAngularStats.maxRotationOrthogonalDelta = PxMax(
      gEndpointAngularStats.maxRotationOrthogonalDelta,
      rotationOrthogonalMagnitude);
  gEndpointAngularStats.maxAngularVelocityOrthogonalDelta = PxMax(
      gEndpointAngularStats.maxAngularVelocityOrthogonalDelta,
      angularVelocityOrthogonalMagnitude);
  gEndpointAngularStats.maxJointAnglePoseMismatch = PxMax(
      gEndpointAngularStats.maxJointAnglePoseMismatch,
      jointAnglePoseMismatch);
  gEndpointAngularStats.maxApiVelocityMagnitudeMismatch = PxMax(
      gEndpointAngularStats.maxApiVelocityMagnitudeMismatch,
      apiVelocityMagnitudeMismatch);
}

static void sampleRevoluteMotorState() {
  if (!gRevoluteMotorBodyA || !gRevoluteMotorBodyB ||
      !gRevoluteMotorJoint)
    return;

  RevoluteMotorStats &stats = gRevoluteMotorStats;
  stats.stateSamples++;
  const PxTransform poseA = gRevoluteMotorBodyA->getGlobalPose();
  const PxTransform poseB = gRevoluteMotorBodyB->getGlobalPose();
  const PxVec3 angularA = gRevoluteMotorBodyA->getAngularVelocity();
  const PxVec3 angularB = gRevoluteMotorBodyB->getAngularVelocity();
  PxVec3 axisA, axisB;
  getJointWorldAxes(gRevoluteMotorJoint, axisA, axisB);
  const PxTransform relative =
      gRevoluteMotorJoint->getRelativeTransform();
  if (!poseA.isValid() || !poseB.isValid() ||
      !angularA.isFinite() || !angularB.isFinite() ||
      !axisA.isFinite() || !axisB.isFinite() ||
      !relative.isValid()) {
    stats.nonFiniteSamples++;
    return;
  }

  const PxReal relativeVelocity =
      (angularB - angularA).dot(axisA);
  const PxReal relativeError =
      PxAbs(relativeVelocity - gRevoluteMotorTargetVelocity);
  stats.finalRelativeVelocity = relativeVelocity;
  stats.finalRelativeError = relativeError;
  if (stats.stateSamples > gRevoluteMotorLateBeginFrame)
    stats.maximumLateRelativeError =
        PxMax(stats.maximumLateRelativeError, relativeError);

  const PxVec3 localAngularA = poseA.q.rotateInv(angularA);
  const PxVec3 localAngularB = poseB.q.rotateInv(angularB);
  const PxVec3 localMomentumA =
      localAngularA.multiply(
          gRevoluteMotorBodyA->getMassSpaceInertiaTensor());
  const PxVec3 localMomentumB =
      localAngularB.multiply(
          gRevoluteMotorBodyB->getMassSpaceInertiaTensor());
  const PxVec3 totalMomentum =
      poseA.q.rotate(localMomentumA) +
      poseB.q.rotate(localMomentumB);
  stats.maximumAngularMomentumDrift =
      PxMax(stats.maximumAngularMomentumDrift,
            totalMomentum.magnitude());
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axisA.cross(axisB).magnitude());
}

static void sampleRevoluteMotorLimitState() {
  if (!gRevoluteMotorLimitBody || !gRevoluteMotorLimitJoint)
    return;

  RevoluteMotorLimitStats &stats = gRevoluteMotorLimitStats;
  stats.stateSamples++;
  const PxTransform pose = gRevoluteMotorLimitBody->getGlobalPose();
  const PxTransform poseA =
      gRevoluteMotorLimitBodyA
          ? gRevoluteMotorLimitBodyA->getGlobalPose()
          : PxTransform(PxIdentity);
  const PxVec3 angularVelocity =
      gRevoluteMotorLimitBody->getAngularVelocity();
  const PxVec3 angularVelocityA =
      gRevoluteMotorLimitBodyA
          ? gRevoluteMotorLimitBodyA->getAngularVelocity()
          : PxVec3(0.0f);
  PxVec3 axis0, axis1;
  getJointWorldAxes(gRevoluteMotorLimitJoint, axis0, axis1);
  const PxTransform relative =
      gRevoluteMotorLimitJoint->getRelativeTransform();
  const PxReal angle = gRevoluteMotorLimitJoint->getAngle();
  const PxReal jointVelocity =
      gRevoluteMotorLimitJoint->getVelocity();
  if (!pose.isValid() || !poseA.isValid() ||
      !angularVelocity.isFinite() || !angularVelocityA.isFinite() ||
      !axis0.isFinite() || !axis1.isFinite() ||
      !relative.isValid() || !PxIsFinite(angle) ||
      !PxIsFinite(jointVelocity)) {
    stats.nonFiniteSamples++;
    return;
  }

  stats.finalAngle = angle;
  stats.minimumAngle = PxMin(stats.minimumAngle, angle);
  stats.maximumAngle = PxMax(stats.maximumAngle, angle);
  stats.maximumUpperViolation =
      PxMax(stats.maximumUpperViolation,
            PxMax(0.0f, angle - gRevoluteMotorLimitUpper));
  stats.maximumLowerViolation =
      PxMax(stats.maximumLowerViolation,
            PxMax(0.0f, gRevoluteMotorLimitLower - angle));
  stats.finalTargetVelocityReadback =
      gRevoluteMotorLimitJoint->getDriveVelocity();
  if (stats.finalTargetVelocityReadback > 0.0f &&
      angle >=
          gRevoluteMotorLimitUpper -
              gRevoluteMotorLimitFinalTolerance) {
    stats.upperBoundSamples++;
    stats.lowerBoundSamples = 0;
    if (stats.upperBoundSamples >
        gRevoluteMotorLimitBoundarySettleSamples)
      stats.maximumLateOutwardVelocity =
          PxMax(stats.maximumLateOutwardVelocity,
                PxMax(0.0f, jointVelocity));
  } else if (stats.finalTargetVelocityReadback < 0.0f &&
             angle <=
                 gRevoluteMotorLimitLower +
                     gRevoluteMotorLimitFinalTolerance) {
    stats.lowerBoundSamples++;
    stats.upperBoundSamples = 0;
    if (stats.lowerBoundSamples >
        gRevoluteMotorLimitBoundarySettleSamples)
      stats.maximumLateOutwardVelocity =
          PxMax(stats.maximumLateOutwardVelocity,
                PxMax(0.0f, -jointVelocity));
  } else {
    stats.upperBoundSamples = 0;
    stats.lowerBoundSamples = 0;
  }
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axis0.cross(axis1).magnitude());
}

static void sampleRevoluteMotorFreeSpinState() {
  if (!gRevoluteMotorFreeSpinBody || !gRevoluteMotorFreeSpinJoint)
    return;

  RevoluteMotorFreeSpinStats &stats = gRevoluteMotorFreeSpinStats;
  stats.stateSamples++;
  const PxTransform pose =
      gRevoluteMotorFreeSpinBody->getGlobalPose();
  const PxTransform poseA =
      gRevoluteMotorFreeSpinBodyA
          ? gRevoluteMotorFreeSpinBodyA->getGlobalPose()
          : PxTransform(PxIdentity);
  const PxVec3 angularVelocity =
      gRevoluteMotorFreeSpinBody->getAngularVelocity();
  const PxVec3 angularVelocityA =
      gRevoluteMotorFreeSpinBodyA
          ? gRevoluteMotorFreeSpinBodyA->getAngularVelocity()
          : PxVec3(0.0f);
  PxVec3 axis0, axis1;
  getJointWorldAxes(gRevoluteMotorFreeSpinJoint, axis0, axis1);
  const PxTransform relative =
      gRevoluteMotorFreeSpinJoint->getRelativeTransform();
  const PxReal velocity =
      gRevoluteMotorFreeSpinJoint->getVelocity();
  if (!pose.isValid() || !poseA.isValid() ||
      !angularVelocity.isFinite() || !angularVelocityA.isFinite() ||
      !axis0.isFinite() || !axis1.isFinite() ||
      !relative.isValid() || !PxIsFinite(velocity)) {
    stats.nonFiniteSamples++;
    return;
  }

  stats.finalVelocity = velocity;
  if (stats.stateSamples <= gRevoluteMotorFreeSpinBoostFrame) {
    stats.preBoostFinalVelocity = velocity;
    if (stats.stateSamples > gRevoluteMotorFreeSpinLateBeginFrame)
      stats.maximumLatePreBoostError =
          PxMax(stats.maximumLatePreBoostError,
                PxAbs(velocity -
                      gRevoluteMotorFreeSpinTargetVelocity));
  } else {
    stats.minimumPostBoostVelocity =
        PxMin(stats.minimumPostBoostVelocity, velocity);
    stats.maximumPostBoostVelocityDrop =
        PxMax(stats.maximumPostBoostVelocityDrop,
              PxMax(0.0f,
                    stats.boostVelocityReadback - velocity));
  }
  if (gRevoluteMotorFreeSpinBodyA) {
    const PxVec3 localAngularA =
        poseA.q.rotateInv(angularVelocityA);
    const PxVec3 localAngularB =
        pose.q.rotateInv(angularVelocity);
    const PxVec3 localMomentumA =
        localAngularA.multiply(
            gRevoluteMotorFreeSpinBodyA
                ->getMassSpaceInertiaTensor());
    const PxVec3 localMomentumB =
        localAngularB.multiply(
            gRevoluteMotorFreeSpinBody
                ->getMassSpaceInertiaTensor());
    const PxVec3 totalMomentum =
        poseA.q.rotate(localMomentumA) +
        pose.q.rotate(localMomentumB);
    stats.maximumAngularMomentumDrift =
        PxMax(stats.maximumAngularMomentumDrift,
              totalMomentum.magnitude());
  }
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axis0.cross(axis1).magnitude());
}

static void sampleRevoluteMotorRatioState() {
  if (!gRevoluteMotorRatioBodyA || !gRevoluteMotorRatioBodyB ||
      !gRevoluteMotorRatioJoint)
    return;

  RevoluteMotorRatioStats &stats = gRevoluteMotorRatioStats;
  stats.stateSamples++;
  const PxTransform poseA =
      gRevoluteMotorRatioBodyA->getGlobalPose();
  const PxTransform poseB =
      gRevoluteMotorRatioBodyB->getGlobalPose();
  const PxVec3 angularA =
      gRevoluteMotorRatioBodyA->getAngularVelocity();
  const PxVec3 angularB =
      gRevoluteMotorRatioBodyB->getAngularVelocity();
  const PxVec3 linearA =
      gRevoluteMotorRatioBodyA->getLinearVelocity();
  const PxVec3 linearB =
      gRevoluteMotorRatioBodyB->getLinearVelocity();
  PxVec3 axisA, axisB;
  getJointWorldAxes(gRevoluteMotorRatioJoint, axisA, axisB);
  const PxTransform relative =
      gRevoluteMotorRatioJoint->getRelativeTransform();
  if (!poseA.isValid() || !poseB.isValid() ||
      !angularA.isFinite() || !angularB.isFinite() ||
      !linearA.isFinite() || !linearB.isFinite() ||
      !axisA.isFinite() || !axisB.isFinite() ||
      !relative.isValid()) {
    stats.nonFiniteSamples++;
    return;
  }

  const PxReal velocityA = angularA.dot(axisA);
  const PxReal velocityB = angularB.dot(axisA);
  const PxReal driveGearRatio =
      gRevoluteMotorRatioStats.driveGearRatioReadback;
  const PxReal weightedVelocity =
      driveGearRatio * velocityB - velocityA;
  const PxReal weightedVelocityError =
      PxAbs(weightedVelocity -
            gRevoluteMotorRatioTargetVelocity);
  stats.finalVelocityA = velocityA;
  stats.finalVelocityB = velocityB;
  stats.finalWeightedVelocity = weightedVelocity;
  stats.finalWeightedVelocityError = weightedVelocityError;
  if (stats.stateSamples > gRevoluteMotorRatioLateBeginFrame)
    stats.maximumLateWeightedVelocityError =
        PxMax(stats.maximumLateWeightedVelocityError,
              weightedVelocityError);
  const PxVec3 relativeAngular = angularB - angularA;
  const PxVec3 relativeSwing =
      relativeAngular -
      axisA * relativeAngular.dot(axisA);
  if (stats.stateSamples > gRevoluteMotorRatioLateBeginFrame)
    stats.maximumLateRelativeSwingVelocity =
        PxMax(stats.maximumLateRelativeSwingVelocity,
              relativeSwing.magnitude());
  const PxVec3 worldLeverA =
      poseA.q.rotate(gRevoluteMotorRatioConfiguredAnchorA);
  const PxVec3 worldLeverB =
      poseB.q.rotate(gRevoluteMotorRatioConfiguredAnchorB);
  const PxVec3 anchorVelocityA =
      linearA + angularA.cross(worldLeverA);
  const PxVec3 anchorVelocityB =
      linearB + angularB.cross(worldLeverB);
  const PxReal relativeAnchorPointSpeed =
      (anchorVelocityB - anchorVelocityA).magnitude();
  stats.finalRelativeAnchorPointSpeed =
      relativeAnchorPointSpeed;
  if (stats.stateSamples > gRevoluteMotorRatioLateBeginFrame)
    stats.maximumLateRelativeAnchorPointSpeed =
        PxMax(stats.maximumLateRelativeAnchorPointSpeed,
              relativeAnchorPointSpeed);
  stats.maximumLinearSpeed =
      PxMax(stats.maximumLinearSpeed,
            PxMax(linearA.magnitude(), linearB.magnitude()));

  const PxVec3 localAngularA = poseA.q.rotateInv(angularA);
  const PxVec3 localAngularB = poseB.q.rotateInv(angularB);
  const PxVec3 localMomentumA =
      localAngularA.multiply(
          gRevoluteMotorRatioBodyA->getMassSpaceInertiaTensor());
  const PxVec3 localMomentumB =
      localAngularB.multiply(
          gRevoluteMotorRatioBodyB->getMassSpaceInertiaTensor());
  const PxVec3 generalizedMomentum =
      poseA.q.rotate(localMomentumA) *
          driveGearRatio +
      poseB.q.rotate(localMomentumB);
  const PxReal massA = gRevoluteMotorRatioBodyA->getMass();
  const PxReal massB = gRevoluteMotorRatioBodyB->getMass();
  const PxVec3 totalLinearMomentum =
      linearA * massA + linearB * massB;
  const PxVec3 totalAngularMomentum =
      poseA.p.cross(linearA * massA) +
      poseA.q.rotate(localMomentumA) +
      poseB.p.cross(linearB * massB) +
      poseB.q.rotate(localMomentumB);
  stats.maximumTotalLinearMomentum =
      PxMax(stats.maximumTotalLinearMomentum,
            totalLinearMomentum.magnitude());
  if (stats.stateSamples <=
      gRevoluteMotorDynamicOffPrincipalMomentumWindowFrames)
    stats.maximumInitialTotalAngularMomentum =
        PxMax(stats.maximumInitialTotalAngularMomentum,
              totalAngularMomentum.magnitude());
  if (stats.stateSamples <=
      gRevoluteMotorDynamicOffPrincipalMomentumWindowFrames)
    stats.maximumInitialGeneralizedMomentumDrift =
        PxMax(stats.maximumInitialGeneralizedMomentumDrift,
              generalizedMomentum.magnitude());
  stats.maximumGeneralizedMomentumDrift =
      PxMax(stats.maximumGeneralizedMomentumDrift,
            generalizedMomentum.magnitude());
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axisA.cross(axisB).magnitude());
}

static void sampleRevoluteMotorContactState() {
  if (!gRevoluteMotorContactBodyA ||
      !gRevoluteMotorContactBodyB ||
      !gRevoluteMotorContactJoint)
    return;

  RevoluteMotorContactStats &stats =
      gRevoluteMotorContactStats;
  stats.stateSamples++;
  const PxTransform poseA =
      gRevoluteMotorContactBodyA->getGlobalPose();
  const PxTransform poseB =
      gRevoluteMotorContactBodyB->getGlobalPose();
  const PxVec3 angularA =
      gRevoluteMotorContactBodyA->getAngularVelocity();
  const PxVec3 angularB =
      gRevoluteMotorContactBodyB->getAngularVelocity();
  PxVec3 axisA, axisB;
  getJointWorldAxes(gRevoluteMotorContactJoint, axisA, axisB);
  const PxTransform relative =
      gRevoluteMotorContactJoint->getRelativeTransform();
  if (!poseA.isValid() || !poseB.isValid() ||
      !angularA.isFinite() || !angularB.isFinite() ||
      !axisA.isFinite() || !axisB.isFinite() ||
      !relative.isValid()) {
    stats.nonFiniteSamples++;
    return;
  }

  const PxReal relativeVelocity =
      (angularB - angularA).dot(axisA);
  const PxReal relativeError =
      PxAbs(relativeVelocity -
            gRevoluteMotorContactTargetVelocity);
  stats.finalVelocityA = angularA.dot(axisA);
  stats.finalVelocityB = angularB.dot(axisA);
  stats.finalRelativeVelocity = relativeVelocity;
  stats.finalRelativeError = relativeError;
  if (stats.stateSamples > gRevoluteMotorContactLateBeginFrame)
    stats.maximumLateRelativeError =
        PxMax(stats.maximumLateRelativeError, relativeError);
  PxConstraint *constraint =
      gRevoluteMotorContactJoint->getConstraint();
  if (!constraint) {
    stats.nonFiniteSamples++;
    return;
  }
  PxVec3 linearForce(0.0f), angularForce(0.0f);
  constraint->getForce(linearForce, angularForce);
  if (!linearForce.isFinite() || !angularForce.isFinite()) {
    stats.nonFiniteSamples++;
    return;
  }
  if (stats.stateSamples > gRevoluteMotorContactLateBeginFrame) {
    const PxReal driveReaction =
        PxAbs(angularForce.dot(axisA));
    stats.lateDriveReactionSamples++;
    stats.lateDriveReactionSum += driveReaction;
    stats.maximumLateDriveReaction =
        PxMax(stats.maximumLateDriveReaction, driveReaction);
  }
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axisA.cross(axisB).magnitude());
  stats.maximumCenterHeightError =
      PxMax(stats.maximumCenterHeightError,
            PxMax(PxAbs(poseA.p.y -
                        gRevoluteMotorContactCenterHeight),
                  PxAbs(poseB.p.y -
                        gRevoluteMotorContactCenterHeight)));
}

static void sampleRevoluteMotorKinematicState() {
  if (!gRevoluteMotorKinematicBody ||
      !gRevoluteMotorKinematicDynamicBody ||
      !gRevoluteMotorKinematicJoint)
    return;

  RevoluteMotorKinematicStats &stats =
      gRevoluteMotorKinematicStats;
  stats.stateSamples++;
  const PxTransform kinematicPose =
      gRevoluteMotorKinematicBody->getGlobalPose();
  const PxTransform dynamicPose =
      gRevoluteMotorKinematicDynamicBody->getGlobalPose();
  const PxVec3 kinematicAngular =
      gRevoluteMotorKinematicBody->getAngularVelocity();
  const PxVec3 dynamicAngular =
      gRevoluteMotorKinematicDynamicBody->getAngularVelocity();
  PxVec3 axisA, axisB;
  getJointWorldAxes(gRevoluteMotorKinematicJoint, axisA, axisB);
  const PxTransform relative =
      gRevoluteMotorKinematicJoint->getRelativeTransform();
  if (!kinematicPose.isValid() || !dynamicPose.isValid() ||
      !kinematicAngular.isFinite() || !dynamicAngular.isFinite() ||
      !axisA.isFinite() || !axisB.isFinite() ||
      !relative.isValid()) {
    stats.nonFiniteSamples++;
    return;
  }

  const PxReal kinematicVelocity =
      kinematicAngular.dot(axisA);
  const PxReal dynamicVelocity =
      dynamicAngular.dot(axisA);
  const PxReal relativeVelocity =
      dynamicVelocity - kinematicVelocity;
  const PxReal relativeError =
      PxAbs(relativeVelocity -
            gRevoluteMotorKinematicTargetVelocity);
  stats.finalKinematicVelocity = kinematicVelocity;
  stats.finalDynamicVelocity = dynamicVelocity;
  stats.finalRelativeVelocity = relativeVelocity;
  stats.finalRelativeError = relativeError;
  if (stats.stateSamples > gRevoluteMotorKinematicLateBeginFrame) {
    stats.maximumLateRelativeError =
        PxMax(stats.maximumLateRelativeError, relativeError);
    stats.maximumLateKinematicVelocityError =
        PxMax(stats.maximumLateKinematicVelocityError,
              PxAbs(kinematicVelocity -
                    gRevoluteMotorKinematicEndpointVelocity));
    stats.maximumLateDynamicVelocityError =
        PxMax(stats.maximumLateDynamicVelocityError,
              PxAbs(dynamicVelocity -
                    gRevoluteMotorKinematicExpectedDynamicVelocity));
  }
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axisA.cross(axisB).magnitude());
}

static void sampleRevoluteMotorOffPrincipalState() {
  if (!gRevoluteMotorOffPrincipalBody ||
      !gRevoluteMotorOffPrincipalJoint)
    return;

  RevoluteMotorOffPrincipalStats &stats =
      gRevoluteMotorOffPrincipalStats;
  stats.stateSamples++;
  const PxTransform pose =
      gRevoluteMotorOffPrincipalBody->getGlobalPose();
  const PxVec3 angularVelocity =
      gRevoluteMotorOffPrincipalBody->getAngularVelocity();
  PxVec3 axisA, axisB;
  getJointWorldAxes(gRevoluteMotorOffPrincipalJoint, axisA, axisB);
  const PxTransform relative =
      gRevoluteMotorOffPrincipalJoint->getRelativeTransform();
  PxConstraint *constraint =
      gRevoluteMotorOffPrincipalJoint->getConstraint();
  if (!pose.isValid() || !angularVelocity.isFinite() ||
      !axisA.isFinite() || !axisB.isFinite() ||
      !relative.isValid() || !constraint) {
    stats.nonFiniteSamples++;
    return;
  }

  PxVec3 linearForce(0.0f), angularForce(0.0f);
  constraint->getForce(linearForce, angularForce);
  if (!linearForce.isFinite() || !angularForce.isFinite()) {
    stats.nonFiniteSamples++;
    return;
  }

  const PxReal hingeVelocity = angularVelocity.dot(axisA);
  const PxReal hingeVelocityError =
      PxAbs(hingeVelocity -
            gRevoluteMotorOffPrincipalTargetVelocity);
  const PxReal swingVelocity =
      (angularVelocity - axisA * hingeVelocity).magnitude();
  const PxReal axialReaction = angularForce.dot(axisA);
  const PxReal swingReaction =
      (angularForce - axisA * axialReaction).magnitude();
  stats.finalHingeVelocity = hingeVelocity;
  stats.finalHingeVelocityError = hingeVelocityError;
  if (stats.stateSamples >
      gRevoluteMotorOffPrincipalLateBeginFrame) {
    stats.maximumLateHingeVelocityError =
        PxMax(stats.maximumLateHingeVelocityError,
              hingeVelocityError);
    stats.maximumLateSwingVelocity =
        PxMax(stats.maximumLateSwingVelocity, swingVelocity);
  }
  stats.maximumSwingReaction =
      PxMax(stats.maximumSwingReaction, swingReaction);
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axisA.cross(axisB).magnitude());
}

static void sampleRevoluteMotorOffCenterState() {
  if (!gRevoluteMotorOffCenterBody ||
      !gRevoluteMotorOffCenterJoint)
    return;

  RevoluteMotorOffCenterStats &stats =
      gRevoluteMotorOffCenterStats;
  stats.stateSamples++;
  const PxTransform pose =
      gRevoluteMotorOffCenterBody->getGlobalPose();
  const PxVec3 linearVelocity =
      gRevoluteMotorOffCenterBody->getLinearVelocity();
  const PxVec3 angularVelocity =
      gRevoluteMotorOffCenterBody->getAngularVelocity();
  PxVec3 axisA, axisB;
  getJointWorldAxes(gRevoluteMotorOffCenterJoint, axisA, axisB);
  const PxTransform relative =
      gRevoluteMotorOffCenterJoint->getRelativeTransform();
  PxConstraint *constraint =
      gRevoluteMotorOffCenterJoint->getConstraint();
  if (!pose.isValid() || !linearVelocity.isFinite() ||
      !angularVelocity.isFinite() || !axisA.isFinite() ||
      !axisB.isFinite() || !relative.isValid() || !constraint) {
    stats.nonFiniteSamples++;
    return;
  }

  PxVec3 linearForce(0.0f), angularForce(0.0f);
  constraint->getForce(linearForce, angularForce);
  if (!linearForce.isFinite() || !angularForce.isFinite()) {
    stats.nonFiniteSamples++;
    return;
  }

  const PxReal hingeVelocity = angularVelocity.dot(axisA);
  const PxReal hingeVelocityError =
      PxAbs(hingeVelocity -
            gRevoluteMotorOffCenterTargetVelocity);
  const PxReal swingVelocity =
      (angularVelocity - axisA * hingeVelocity).magnitude();
  const PxVec3 worldLeverArm =
      pose.q.rotate(gRevoluteMotorOffCenterConfiguredLocalAnchor);
  const PxVec3 anchorPointVelocity =
      linearVelocity + angularVelocity.cross(worldLeverArm);
  const PxReal anchorPointSpeed =
      anchorPointVelocity.magnitude();
  stats.finalHingeVelocity = hingeVelocity;
  stats.finalHingeVelocityError = hingeVelocityError;
  stats.finalAnchorPointSpeed = anchorPointSpeed;
  if (stats.stateSamples > gRevoluteMotorOffCenterLateBeginFrame) {
    stats.maximumLateHingeVelocityError =
        PxMax(stats.maximumLateHingeVelocityError,
              hingeVelocityError);
    stats.maximumLateSwingVelocity =
        PxMax(stats.maximumLateSwingVelocity, swingVelocity);
    stats.maximumLateAnchorPointSpeed =
        PxMax(stats.maximumLateAnchorPointSpeed,
              anchorPointSpeed);
  }
  stats.maximumLinearSpeed =
      PxMax(stats.maximumLinearSpeed, linearVelocity.magnitude());
  stats.maximumLinearReaction =
      PxMax(stats.maximumLinearReaction, linearForce.magnitude());
  stats.maximumAnchorError =
      PxMax(stats.maximumAnchorError, relative.p.magnitude());
  stats.maximumAxisMisalignment =
      PxMax(stats.maximumAxisMisalignment,
            axisA.cross(axisB).magnitude());
}

static void sampleEndpointState() {
  if (isRevoluteEndpointProbe()) {
    sampleEndpointAngularState();
    return;
  }
  gEndpointStats.stateSampleAttempts++;
  if (!gEndpointTarget)
    return;
  gEndpointStats.stateSamples++;

  const PxU32 currentFrame = gGateStats.completedFrames + 1;
  const bool contactObserved =
      gProjectiles.size() == 1 &&
      gProjectiles[0].firstContactFrame != PX_MAX_U32;
  const PxU32 firstContactFrame =
      contactObserved ? gProjectiles[0].firstContactFrame : PX_MAX_U32;
  const bool inResponseWindow =
      contactObserved && currentFrame >= firstContactFrame &&
      currentFrame < firstContactFrame + gEndpointResponseWindowFrames;
  if (inResponseWindow)
    gEndpointStats.responseSampleAttempts++;

  const PxTransform pose = gEndpointTarget->getGlobalPose();
  const PxVec3 linearVelocity = gEndpointTarget->getLinearVelocity();
  if (!pose.p.isFinite() || !pose.q.isFinite() ||
      !linearVelocity.isFinite()) {
    gEndpointStats.nonFiniteStateSamples++;
    if (inResponseWindow)
      gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }

  if (gEndpointJoint) {
    const PxTransform relative = gEndpointJoint->getRelativeTransform();
    if (!relative.p.isFinite() || !relative.q.isFinite()) {
      gEndpointStats.nonFiniteStateSamples++;
      gGateStats.nonFinite++;
    } else {
      const PxReal transverseAnchor =
          getSafeMagnitude(PxVec3(0.0f, relative.p.y, relative.p.z));
      if (!PxIsFinite(transverseAnchor)) {
        gEndpointStats.nonFiniteStateSamples++;
        gGateStats.nonFinite++;
      } else {
        gEndpointStats.maxTransverseAnchor =
            PxMax(gEndpointStats.maxTransverseAnchor, transverseAnchor);
      }
    }
  }

  if (!contactObserved) {
    const PxReal precontactPositionDrift =
        getSafeMagnitude(pose.p - gEndpointStats.initialTargetPosition);
    const PxReal precontactSpeed = getSafeMagnitude(linearVelocity);
    if (!PxIsFinite(precontactPositionDrift) ||
        !PxIsFinite(precontactSpeed)) {
      gEndpointStats.nonFiniteStateSamples++;
      gGateStats.nonFinite++;
      return;
    }
    gEndpointStats.maxPrecontactPositionDrift = PxMax(
        gEndpointStats.maxPrecontactPositionDrift, precontactPositionDrift);
    gEndpointStats.maxPrecontactSpeed =
        PxMax(gEndpointStats.maxPrecontactSpeed, precontactSpeed);
    gEndpointStats.responseBaselinePosition = pose.p;
    gEndpointStats.responseBaselineVelocity = linearVelocity;
    gEndpointStats.responseBaselineSamples++;
    return;
  }
  if (!inResponseWindow)
    return;

  const PxVec3 positionDelta =
      pose.p - gEndpointStats.responseBaselinePosition;
  const PxVec3 velocityDelta =
      linearVelocity - gEndpointStats.responseBaselineVelocity;
  if (!positionDelta.isFinite() || !velocityDelta.isFinite()) {
    gEndpointStats.nonFiniteStateSamples++;
    gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }
  const PxReal signedPositionDelta =
      positionDelta.dot(gEndpointStats.expectedAxis);
  const PxReal signedVelocityDelta =
      velocityDelta.dot(gEndpointStats.expectedAxis);
  const PxReal positionOrthogonalMagnitude = getSafeMagnitude(
      positionDelta - gEndpointStats.expectedAxis * signedPositionDelta);
  const PxReal velocityOrthogonalMagnitude = getSafeMagnitude(
      velocityDelta - gEndpointStats.expectedAxis * signedVelocityDelta);
  if (!PxIsFinite(signedPositionDelta) ||
      !PxIsFinite(signedVelocityDelta) ||
      !PxIsFinite(positionOrthogonalMagnitude) ||
      !PxIsFinite(velocityOrthogonalMagnitude)) {
    gEndpointStats.nonFiniteStateSamples++;
    gEndpointStats.nonFiniteResponseSamples++;
    gGateStats.nonFinite++;
    return;
  }
  gEndpointStats.responseSamples++;
  gEndpointStats.responsePositionDeltaSum.x = saturateMetric(
      double(gEndpointStats.responsePositionDeltaSum.x) +
      double(positionDelta.x));
  gEndpointStats.responsePositionDeltaSum.y = saturateMetric(
      double(gEndpointStats.responsePositionDeltaSum.y) +
      double(positionDelta.y));
  gEndpointStats.responsePositionDeltaSum.z = saturateMetric(
      double(gEndpointStats.responsePositionDeltaSum.z) +
      double(positionDelta.z));
  gEndpointStats.responseVelocityDeltaSum.x = saturateMetric(
      double(gEndpointStats.responseVelocityDeltaSum.x) +
      double(velocityDelta.x));
  gEndpointStats.responseVelocityDeltaSum.y = saturateMetric(
      double(gEndpointStats.responseVelocityDeltaSum.y) +
      double(velocityDelta.y));
  gEndpointStats.responseVelocityDeltaSum.z = saturateMetric(
      double(gEndpointStats.responseVelocityDeltaSum.z) +
      double(velocityDelta.z));
  gEndpointStats.positionOrthogonalSquaredSum +=
      double(positionOrthogonalMagnitude) *
      double(positionOrthogonalMagnitude);
  gEndpointStats.velocityOrthogonalSquaredSum +=
      double(velocityOrthogonalMagnitude) *
      double(velocityOrthogonalMagnitude);
  gEndpointStats.maxPositionOrthogonalDelta = PxMax(
      gEndpointStats.maxPositionOrthogonalDelta,
      positionOrthogonalMagnitude);
  gEndpointStats.maxVelocityOrthogonalDelta = PxMax(
      gEndpointStats.maxVelocityOrthogonalDelta,
      velocityOrthogonalMagnitude);
}

void stepPhysics(bool interactive) {
  if (!interactive && isRevoluteMotorKinematicCase() &&
      gRevoluteMotorKinematicBody) {
    const PxReal targetTime =
        PxReal(gGateStats.completedFrames + 1) *
        gHeadlessOptions.dt;
    const PxQuat targetRotation(
        gRevoluteMotorKinematicEndpointVelocity * targetTime,
        PxVec3(1.0f, 0.0f, 0.0f));
    gRevoluteMotorKinematicBody->setKinematicTarget(
        PxTransform(gRevoluteMotorKinematicInitialPose.p,
                    targetRotation));
    gRevoluteMotorKinematicStats.targetUpdates++;
  }
  if (!interactive && isRevoluteMotorFreeSpinCase() &&
      gRevoluteMotorFreeSpinBody &&
      gGateStats.completedFrames ==
          gRevoluteMotorFreeSpinBoostFrame) {
    if (isRevoluteMotorDynamicFreeSpinCase() &&
        gRevoluteMotorFreeSpinBodyA) {
      gRevoluteMotorFreeSpinBodyA->setAngularVelocity(
          PxVec3(-3.75f, 0.0f, 0.0f), true);
      gRevoluteMotorFreeSpinBody->setAngularVelocity(
          PxVec3(1.25f, 0.0f, 0.0f), true);
    } else {
      gRevoluteMotorFreeSpinBody->setAngularVelocity(
          PxVec3(gRevoluteMotorFreeSpinBoostVelocity, 0.0f, 0.0f),
          true);
    }
    gRevoluteMotorFreeSpinStats.boostVelocityReadback =
        gRevoluteMotorFreeSpinJoint
            ? gRevoluteMotorFreeSpinJoint->getVelocity()
            : 0.0f;
    gRevoluteMotorFreeSpinStats.boostEvents++;
  }
  if (!interactive && isRevoluteMotorLimitCase() &&
      gRevoluteMotorLimitJoint &&
      gGateStats.completedFrames == gRevoluteMotorLimitReverseFrame) {
    gRevoluteMotorLimitJoint->setDriveVelocity(
        -gRevoluteMotorLimitTargetVelocity, false);
    gRevoluteMotorLimitStats.reverseEvents++;
  }
  if (!interactive && isNativeBreakReactionCase() &&
      gNativeBreakReactionBody) {
    if (isNativeAngularReactionCase())
      gNativeBreakReactionBody->addTorque(
          gNativeAngularLoad, PxForceMode::eFORCE, true);
    else
      gNativeBreakReactionBody->addForce(
          gNativeLinearLoad, PxForceMode::eFORCE, true);
  }
  if (!interactive && isForcePairCase() && gForceStaticBody &&
      gForcePairBody1) {
    gForceStaticBody->addForce(gForceStaticStats.appliedForceActor0,
                               PxForceMode::eFORCE, true);
    gForcePairBody1->addForce(gForceStaticStats.appliedForceActor1,
                              PxForceMode::eFORCE, true);
  }
  if (!interactive && gGateStats.completedFrames == gImpactLaunchFrame) {
    if (isRevoluteEndpointProbe())
      launchEndpointAngularVelocity();
    else
      launchGateProjectiles();
  }

  gScene->simulate(interactive ? (1.0f / 60.0f) : gHeadlessOptions.dt);
  PxU32 errorState = 0;
  if (!gScene->fetchResults(true, &errorState)) {
    if (!interactive)
      gGateStats.fetchFailures++;
    return;
  }

  if (interactive)
    return;

  gGateStats.fetchErrorState |= errorState;
  sampleGateState();
  if (isRevoluteMotorCase())
    sampleRevoluteMotorState();
  if (isRevoluteMotorLimitCase())
    sampleRevoluteMotorLimitState();
  if (isRevoluteMotorFreeSpinCase())
    sampleRevoluteMotorFreeSpinState();
  if (isRevoluteMotorRatioCase())
    sampleRevoluteMotorRatioState();
  if (isRevoluteMotorContactCase())
    sampleRevoluteMotorContactState();
  if (isRevoluteMotorKinematicCase())
    sampleRevoluteMotorKinematicState();
  if (isRevoluteMotorOffPrincipalCase())
    sampleRevoluteMotorOffPrincipalState();
  if (isRevoluteMotorOffCenterCase())
    sampleRevoluteMotorOffCenterState();
  if (isSphericalConeCase())
    sampleSphericalConeState();
  if (isNativeBreakReactionCase())
    sampleNativeBreakReactionState();
  if (isForceReactionCase())
    sampleForceStaticState();
  if (isEndpointProbe())
    sampleEndpointState();

  for (PxU32 j = 0; j < gFixedChainJoints.size(); ++j) {
    PxVec3 linearForce(0.0f), angularForce(0.0f);
    PxConstraint *constraint = gFixedChainJoints[j]->getConstraint();
    if (!constraint)
      continue;
    constraint->getForce(linearForce, angularForce);
    gFixedStats.maxLinearForce =
        PxMax(gFixedStats.maxLinearForce, linearForce.magnitude());
    gFixedStats.maxAngularForce =
        PxMax(gFixedStats.maxAngularForce, angularForce.magnitude());
    const bool broken = constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
    const PxI32 breakableIndex = findBreakableJoint(constraint);
    if (broken && breakableIndex >= 0 &&
        PxU32(breakableIndex) < gBreakPollReported.size() &&
        !gBreakPollReported[PxU32(breakableIndex)]) {
      gBreakPollReported[PxU32(breakableIndex)] = 1;
      if (gFixedStats.firstBrokenFrame == PX_MAX_U32)
        gFixedStats.firstBrokenFrame = gGateStats.completedFrames + 1;
      gFixedStats.brokenCount++;
    }
  }

  const PxU32 earlyBegin = 250, earlyEnd = 550;
  const PxU32 lateBegin = 1000, lateEnd = 1300;

  if (gRevoluteChainBodies.size() >= 5 && gRevoluteChainJoints.size() >= 5) {
    const PxVec3 tailDelta =
        gRevoluteChainBodies[4]->getGlobalPose().p - gRevoluteRestPositions[4];
    const PxReal lateral = PxSqrt(tailDelta.x * tailDelta.x +
                                  tailDelta.z * tailDelta.z);
    gRevoluteStats.maxTailLateral = PxMax(gRevoluteStats.maxTailLateral, lateral);

    const PxReal w4 = gRevoluteChainBodies[3]->getAngularVelocity().magnitude();
    const PxReal w5 = gRevoluteChainBodies[4]->getAngularVelocity().magnitude();

    const PxVec3 axis3 = getJointWorldAxis(gRevoluteChainJoints[3]);
    const PxVec3 axis4 = getJointWorldAxis(gRevoluteChainJoints[4]);
    const PxVec3 wv4 = gRevoluteChainBodies[3]->getAngularVelocity();
    const PxVec3 wv5 = gRevoluteChainBodies[4]->getAngularVelocity();
    const PxVec3 wv4Perp = wv4 - axis3 * wv4.dot(axis3);
    const PxVec3 wv5Perp = wv5 - axis4 * wv5.dot(axis4);
    const PxReal w4Perp = wv4Perp.magnitude();
    const PxReal w5Perp = wv5Perp.magnitude();

    if (gRevoluteStats.frame >= earlyBegin && gRevoluteStats.frame < earlyEnd) {
      gRevoluteStats.sumW4Early += w4;
      gRevoluteStats.sumW5Early += w5;
      gRevoluteStats.sumW4PerpEarly += w4Perp;
      gRevoluteStats.sumW5PerpEarly += w5Perp;
      gRevoluteStats.cntEarly++;
    }
    if (gRevoluteStats.frame >= lateBegin && gRevoluteStats.frame < lateEnd) {
      gRevoluteStats.sumW4Late += w4;
      gRevoluteStats.sumW5Late += w5;
      gRevoluteStats.sumW4PerpLate += w4Perp;
      gRevoluteStats.sumW5PerpLate += w5Perp;
      gRevoluteStats.maxW4PerpLate = PxMax(gRevoluteStats.maxW4PerpLate, w4Perp);
      gRevoluteStats.maxW5PerpLate = PxMax(gRevoluteStats.maxW5PerpLate, w5Perp);
      if (!gRevoluteChainBodies[3]->isSleeping())
        gRevoluteStats.awake4Late++;
      if (!gRevoluteChainBodies[4]->isSleeping())
        gRevoluteStats.awake5Late++;
      gRevoluteStats.cntLate++;
    }

    PxVec3 j3a0, j3a1, j4a0, j4a1;
    getJointWorldAxes(gRevoluteChainJoints[3], j3a0, j3a1);
    getJointWorldAxes(gRevoluteChainJoints[4], j4a0, j4a1);
    const PxReal dot3 = PxClamp(j3a0.dot(j3a1), -1.0f, 1.0f);
    const PxReal dot4 = PxClamp(j4a0.dot(j4a1), -1.0f, 1.0f);
    const PxReal mis3 = PxAcos(dot3) * 180.0f / PxPi;
    const PxReal mis4 = PxAcos(dot4) * 180.0f / PxPi;
    gRevoluteStats.maxAxisMisalign3Deg = PxMax(gRevoluteStats.maxAxisMisalign3Deg, mis3);
    gRevoluteStats.maxAxisMisalign4Deg = PxMax(gRevoluteStats.maxAxisMisalign4Deg, mis4);

    const PxReal a3 = gRevoluteChainJoints[3]->getAngle();
    const PxReal a4 = gRevoluteChainJoints[4]->getAngle();
    gRevoluteStats.maxAbsAngle3 = PxMax(gRevoluteStats.maxAbsAngle3, PxAbs(a3));
    gRevoluteStats.maxAbsAngle4 = PxMax(gRevoluteStats.maxAbsAngle4, PxAbs(a4));
    const PxReal d3 = a3 - gRevoluteStats.prevAngle3;
    const PxReal d4 = a4 - gRevoluteStats.prevAngle4;

    if (PxAbs(d3) > 1e-5f && PxAbs(gRevoluteStats.prevD3) > 1e-5f &&
        d3 * gRevoluteStats.prevD3 < 0.0f)
      gRevoluteStats.flip3++;
    if (PxAbs(d4) > 1e-5f && PxAbs(gRevoluteStats.prevD4) > 1e-5f &&
        d4 * gRevoluteStats.prevD4 < 0.0f)
      gRevoluteStats.flip4++;

    gRevoluteStats.prevD3 = d3;
    gRevoluteStats.prevD4 = d4;
    gRevoluteStats.prevAngle3 = a3;
    gRevoluteStats.prevAngle4 = a4;
  }

  if (gPrismaticChainBodies.size() >= 5 && gPrismaticRestPositions.size() >= 5 &&
      gPrismaticChainJoints.size() >= 5) {
    const PxVec3 tailDelta =
        gPrismaticChainBodies[4]->getGlobalPose().p - gPrismaticRestPositions[4];
    const PxReal tailTransverse =
        PxSqrt(tailDelta.y * tailDelta.y + tailDelta.z * tailDelta.z);
    const PxReal tailAngVel =
        gPrismaticChainBodies[4]->getAngularVelocity().magnitude();
    gPrismaticStats.maxTailTransverse =
        PxMax(gPrismaticStats.maxTailTransverse, tailTransverse);

    if (gRevoluteStats.frame >= earlyBegin && gRevoluteStats.frame < earlyEnd) {
      gPrismaticStats.sumTailEarly += tailTransverse;
      gPrismaticStats.sumTailAngVelEarly += tailAngVel;
      gPrismaticStats.cntEarly++;
    }
    if (gRevoluteStats.frame >= lateBegin && gRevoluteStats.frame < lateEnd) {
      gPrismaticStats.sumTailLate += tailTransverse;
      gPrismaticStats.sumTailAngVelLate += tailAngVel;
      gPrismaticStats.cntLate++;
    }

    for (PxU32 j = 0; j < gPrismaticChainJoints.size(); ++j) {
      PxVec3 a0, a1;
      getJointWorldAxes(gPrismaticChainJoints[j], a0, a1);
      const PxReal dot = PxClamp(a0.dot(a1), -1.0f, 1.0f);
      const PxReal misDeg = PxAcos(dot) * 180.0f / PxPi;
      gPrismaticStats.maxJointAxisMisalignDeg =
          PxMax(gPrismaticStats.maxJointAxisMisalignDeg, misDeg);
    }
  }

  gRevoluteStats.frame++;
  retireObservedProjectiles();
  gGateStats.completedFrames++;
}

struct GateEvaluation {
  PxU32 exitCode;
  const char *status;
  const char *reason;
  PxU32 expectedHits;
  PxU32 hitChains;
  PxU32 wrongFirstContacts;
  PxU32 crossChainContacts;
  PxU32 incompleteObservations;
  PxU32 responseProjectiles;
  PxU32 expectedTargetResponses;
  PxU32 linearTargetResponses;
  PxU32 notApplicableTargetResponses;
  PxU32 respondedTargetChains;
  PxU32 incompleteTargetResponses;
  PxReal minProjectileResponseFraction;
  PxReal minTargetResponseFraction;
  PxReal tailEnergyW1;
  PxReal tailEnergyW2;
  PxReal tailEnergyW3;
  bool jitterReproduced;

  GateEvaluation()
      : exitCode(Snippets::eHEADLESS_PASS), status("PASS"), reason("none"),
        expectedHits(0), hitChains(0), wrongFirstContacts(0),
        crossChainContacts(0), incompleteObservations(0),
        responseProjectiles(0), expectedTargetResponses(0),
        linearTargetResponses(0), notApplicableTargetResponses(0),
        respondedTargetChains(0),
        incompleteTargetResponses(0), minProjectileResponseFraction(0.0f),
        minTargetResponseFraction(0.0f), tailEnergyW1(0.0f),
        tailEnergyW2(0.0f), tailEnergyW3(0.0f), jitterReproduced(false) {}
};

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

static bool isLegacyRevoluteJitterReproduced() {
  if (!gRevoluteStats.cntEarly || !gRevoluteStats.cntLate)
    return false;
  const PxReal avgW4PerpEarly =
      gRevoluteStats.sumW4PerpEarly / gRevoluteStats.cntEarly;
  const PxReal avgW5PerpEarly =
      gRevoluteStats.sumW5PerpEarly / gRevoluteStats.cntEarly;
  const PxReal avgW4PerpLate =
      gRevoluteStats.sumW4PerpLate / gRevoluteStats.cntLate;
  const PxReal avgW5PerpLate =
      gRevoluteStats.sumW5PerpLate / gRevoluteStats.cntLate;
  const PxReal awake4LateRatio =
      PxReal(gRevoluteStats.awake4Late) / PxReal(gRevoluteStats.cntLate);
  const PxReal awake5LateRatio =
      PxReal(gRevoluteStats.awake5Late) / PxReal(gRevoluteStats.cntLate);
  const PxReal growth4Perp =
      avgW4PerpEarly > 1e-6f ? avgW4PerpLate / avgW4PerpEarly : 0.0f;
  const PxReal growth5Perp =
      avgW5PerpEarly > 1e-6f ? avgW5PerpLate / avgW5PerpEarly : 0.0f;
  return (growth4Perp > 1.10f) || (growth5Perp > 1.10f) ||
         (gRevoluteStats.maxW4PerpLate > 2.0f && awake4LateRatio > 0.30f) ||
         (gRevoluteStats.maxW5PerpLate > 2.0f && awake5LateRatio > 0.30f) ||
         (gRevoluteStats.flip4 > 140);
}

static void setGateError(GateEvaluation &evaluation, const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static void setInfrastructureErrorOverFailure(GateEvaluation &evaluation,
                                              const char *reason) {
  if (evaluation.exitCode == Snippets::eHEADLESS_CONFIG_ERROR)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static void setGateFailure(GateEvaluation &evaluation, const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
  evaluation.status = "FAIL";
  evaluation.reason = reason;
}

static PxVec3 getMeanForceStaticLinearForce() {
  return gForceStaticStats.steadySamples
             ? gForceStaticStats.linearForceSum /
                   PxReal(gForceStaticStats.steadySamples)
             : PxVec3(0.0f);
}

static PxVec3 getMeanForceStaticAngularForce() {
  return gForceStaticStats.steadySamples
             ? gForceStaticStats.angularForceSum /
                   PxReal(gForceStaticStats.steadySamples)
             : PxVec3(0.0f);
}

static PxReal getForceStaticExpectedWeight() {
  return gForceStaticStats.expectedWeight > 0.0f
             ? gForceStaticStats.expectedWeight
             : gForceStaticMass * gGravityMagnitude;
}

static PxReal getForceStaticRatio() {
  const PxReal expectedWeight = getForceStaticExpectedWeight();
  return expectedWeight > 0.0f
             ? getSafeMagnitude(getMeanForceStaticLinearForce()) /
                   expectedWeight
             : 0.0f;
}

static PxReal getForceStaticMeanSampleRatio() {
  const PxReal expectedWeight = getForceStaticExpectedWeight();
  return gForceStaticStats.steadySamples && expectedWeight > 0.0f
             ? (gForceStaticStats.linearMagnitudeSum /
                PxReal(gForceStaticStats.steadySamples)) /
                   expectedWeight
             : 0.0f;
}

static PxReal getForceStaticDirectionDot() {
  const PxVec3 meanForce = getMeanForceStaticLinearForce();
  const PxReal forceMagnitude = getSafeMagnitude(meanForce);
  const PxReal expectedForceMagnitude =
      getSafeMagnitude(gForceStaticStats.expectedLinearForce);
  if (forceMagnitude <= 0.0f || expectedForceMagnitude <= 0.0f)
    return 0.0f;
  const double numerator = double(meanForce.x) *
                               double(gForceStaticStats.expectedLinearForce.x) +
                           double(meanForce.y) *
                               double(gForceStaticStats.expectedLinearForce.y) +
                           double(meanForce.z) *
                               double(gForceStaticStats.expectedLinearForce.z);
  const double denominator =
      double(forceMagnitude) * double(expectedForceMagnitude);
  return PxClamp(saturateMetric(numerator / denominator), -1.0f, 1.0f);
}

static PxReal getForceStaticMeanVectorOrthogonalRatio() {
  const PxVec3 meanForce = getMeanForceStaticLinearForce();
  const PxReal expectedWeight = getForceStaticExpectedWeight();
  return expectedWeight > 0.0f
             ? getSafeMagnitude(PxVec3(meanForce.x, 0.0f, meanForce.z)) /
                   expectedWeight
             : 0.0f;
}

static PxReal getForceStaticOrthogonalRmsRatio() {
  const PxReal expectedWeight = getForceStaticExpectedWeight();
  return gForceStaticStats.steadySamples && expectedWeight > 0.0f
             ? saturateMetric(
                   std::sqrt(gForceStaticStats.orthogonalMagnitudeSquaredSum /
                             double(gForceStaticStats.steadySamples)) /
                   double(expectedWeight))
             : 0.0f;
}

static PxReal getForceStaticTorqueRms() {
  return gForceStaticStats.steadySamples
             ? saturateMetric(std::sqrt(
                   gForceStaticStats.angularMagnitudeSquaredSum /
                   double(gForceStaticStats.steadySamples)))
              : 0.0f;
}

static PxReal getForceStaticExpectedTorqueMagnitude() {
  return getSafeMagnitude(gForceStaticStats.expectedTorque);
}

static PxReal getForceStaticTorqueRatio() {
  const PxReal expectedTorqueMagnitude =
      getForceStaticExpectedTorqueMagnitude();
  return expectedTorqueMagnitude > 0.0f
             ? getSafeMagnitude(getMeanForceStaticAngularForce()) /
                   expectedTorqueMagnitude
             : 0.0f;
}

static PxReal getForceStaticMeanSampleTorqueRatio() {
  const PxReal expectedTorqueMagnitude =
      getForceStaticExpectedTorqueMagnitude();
  return gForceStaticStats.steadySamples && expectedTorqueMagnitude > 0.0f
             ? (gForceStaticStats.angularMagnitudeSum /
                PxReal(gForceStaticStats.steadySamples)) /
                   expectedTorqueMagnitude
             : 0.0f;
}

static PxReal getForceStaticTorqueDirectionDot() {
  const PxVec3 meanTorque = getMeanForceStaticAngularForce();
  const PxReal meanTorqueMagnitude = getSafeMagnitude(meanTorque);
  const PxReal expectedTorqueMagnitude =
      getForceStaticExpectedTorqueMagnitude();
  if (meanTorqueMagnitude <= 0.0f || expectedTorqueMagnitude <= 0.0f)
    return 0.0f;
  return PxClamp(meanTorque.dot(gForceStaticStats.expectedTorque) /
                     (meanTorqueMagnitude * expectedTorqueMagnitude),
                 -1.0f, 1.0f);
}

static PxReal getForceStaticTorqueOrthogonalRmsRatio() {
  const PxReal expectedTorqueMagnitude =
      getForceStaticExpectedTorqueMagnitude();
  return gForceStaticStats.steadySamples && expectedTorqueMagnitude > 0.0f
             ? saturateMetric(std::sqrt(
                   gForceStaticStats.angularOrthogonalMagnitudeSquaredSum /
                   double(gForceStaticStats.steadySamples)) /
                              double(expectedTorqueMagnitude))
             : 0.0f;
}

static PxVec3 getEndpointMeanTargetDelta() {
  if (isRevoluteEndpointProbe()) {
    return gEndpointAngularStats.tailSamples
               ? gEndpointAngularStats.responseRotationVectorSum /
                     PxReal(gEndpointAngularStats.tailSamples)
               : PxVec3(0.0f);
  }
  return gEndpointStats.responseSamples
             ? gEndpointStats.responsePositionDeltaSum /
                   PxReal(gEndpointStats.responseSamples)
             : PxVec3(0.0f);
}

static PxVec3 getEndpointMeanTargetVelocityDelta() {
  if (isRevoluteEndpointProbe()) {
    return gEndpointAngularStats.tailSamples
               ? gEndpointAngularStats.responseAngularVelocityDeltaSum /
                     PxReal(gEndpointAngularStats.tailSamples)
               : PxVec3(0.0f);
  }
  return gEndpointStats.responseSamples
             ? gEndpointStats.responseVelocityDeltaSum /
                   PxReal(gEndpointStats.responseSamples)
             : PxVec3(0.0f);
}

static PxReal getEndpointSignedTargetDelta() {
  return getEndpointMeanTargetDelta().dot(gEndpointStats.expectedAxis);
}

static PxReal getEndpointDirectionDot() {
  const PxVec3 delta = getEndpointMeanTargetDelta();
  const PxReal magnitude = getSafeMagnitude(delta);
  return magnitude > 1e-12f
             ? PxClamp(delta.dot(gEndpointStats.expectedAxis) / magnitude,
                       -1.0f, 1.0f)
             : 0.0f;
}

static PxReal getEndpointOrthogonalDelta() {
  const PxVec3 delta = getEndpointMeanTargetDelta();
  const PxReal signedDelta = delta.dot(gEndpointStats.expectedAxis);
  return getSafeMagnitude(delta -
                          gEndpointStats.expectedAxis * signedDelta);
}

static PxReal getEndpointOrthogonalRatio() {
  const PxReal signedDelta = PxAbs(getEndpointSignedTargetDelta());
  return getEndpointOrthogonalDelta() /
         PxMax(signedDelta,
               gEndpointPositionOrthogonalAbsoluteEpsilon);
}

static PxReal getEndpointPositionOrthogonalRms() {
  if (isRevoluteEndpointProbe()) {
    return gEndpointAngularStats.tailSamples
               ? saturateMetric(std::sqrt(
                     gEndpointAngularStats.rotationOrthogonalSquaredSum /
                     double(gEndpointAngularStats.tailSamples)))
               : 0.0f;
  }
  return gEndpointStats.responseSamples
             ? saturateMetric(std::sqrt(
                   gEndpointStats.positionOrthogonalSquaredSum /
                   double(gEndpointStats.responseSamples)))
             : 0.0f;
}

static PxReal getEndpointSignedTargetVelocityDelta() {
  return getEndpointMeanTargetVelocityDelta().dot(
      gEndpointStats.expectedAxis);
}

static PxReal getEndpointVelocityDirectionDot() {
  const PxVec3 delta = getEndpointMeanTargetVelocityDelta();
  const PxReal magnitude = getSafeMagnitude(delta);
  return magnitude > 1e-12f
             ? PxClamp(delta.dot(gEndpointStats.expectedAxis) / magnitude,
                       -1.0f, 1.0f)
             : 0.0f;
}

static PxReal getEndpointVelocityOrthogonalDelta() {
  const PxVec3 delta = getEndpointMeanTargetVelocityDelta();
  const PxReal signedDelta = delta.dot(gEndpointStats.expectedAxis);
  return getSafeMagnitude(delta -
                          gEndpointStats.expectedAxis * signedDelta);
}

static PxReal getEndpointVelocityOrthogonalRatio() {
  const PxReal signedDelta =
      PxAbs(getEndpointSignedTargetVelocityDelta());
  return getEndpointVelocityOrthogonalDelta() /
         PxMax(signedDelta,
               gEndpointVelocityOrthogonalAbsoluteEpsilon);
}

static PxReal getEndpointVelocityOrthogonalRms() {
  if (isRevoluteEndpointProbe()) {
    return gEndpointAngularStats.tailSamples
               ? saturateMetric(std::sqrt(
                     gEndpointAngularStats
                         .angularVelocityOrthogonalSquaredSum /
                     double(gEndpointAngularStats.tailSamples)))
               : 0.0f;
  }
  return gEndpointStats.responseSamples
             ? saturateMetric(std::sqrt(
                   gEndpointStats.velocityOrthogonalSquaredSum /
                   double(gEndpointStats.responseSamples)))
             : 0.0f;
}

static PxReal getEndpointMaxTargetOrthogonalDelta() {
  return isRevoluteEndpointProbe()
             ? gEndpointAngularStats.maxRotationOrthogonalDelta
             : gEndpointStats.maxPositionOrthogonalDelta;
}

static PxReal getEndpointMaxVelocityOrthogonalDelta() {
  return isRevoluteEndpointProbe()
             ? gEndpointAngularStats.maxAngularVelocityOrthogonalDelta
             : gEndpointStats.maxVelocityOrthogonalDelta;
}

static PxReal getEndpointMaxAnchorError() {
  return isRevoluteEndpointProbe()
             ? gEndpointAngularStats.maxAnchorError
             : gEndpointStats.maxTransverseAnchor;
}

static PxReal getEndpointMeanRawJointAngleDelta() {
  return gEndpointAngularStats.tailSamples
             ? saturateMetric(
                   gEndpointAngularStats.rawJointAngleDeltaSum /
                   double(gEndpointAngularStats.tailSamples))
             : 0.0f;
}

static PxReal getEndpointMeanSemanticJointAngleDelta() {
  return gEndpointAngularStats.tailSamples
             ? saturateMetric(
                   gEndpointAngularStats.semanticJointAngleDeltaSum /
                   double(gEndpointAngularStats.tailSamples))
             : 0.0f;
}

static PxReal getEndpointMeanApiVelocityMagnitude() {
  return gEndpointAngularStats.tailSamples
             ? saturateMetric(
                   gEndpointAngularStats.apiVelocityMagnitudeSum /
                   double(gEndpointAngularStats.tailSamples))
             : 0.0f;
}

static GateEvaluation evaluateGate() {
  GateEvaluation evaluation;
  evaluation.tailEnergyW1 = getTailWindowMean(gGateStats.kineticEnergy, 2);
  evaluation.tailEnergyW2 = getTailWindowMean(gGateStats.kineticEnergy, 1);
  evaluation.tailEnergyW3 = getTailWindowMean(gGateStats.kineticEnergy, 0);
  evaluation.jitterReproduced = isLegacyRevoluteJitterReproduced();

  gGateStats.breakCallbackPollMismatches = 0;
  const PxU32 callbackPollCount =
      PxMin(static_cast<PxU32>(gBreakPollReported.size()),
            static_cast<PxU32>(gBreakCallbackReported.size()));
  for (PxU32 i = 0; i < callbackPollCount; ++i) {
    if (gBreakPollReported[i] != gBreakCallbackReported[i])
      gGateStats.breakCallbackPollMismatches++;
  }
  if (gBreakPollReported.size() != gBreakCallbackReported.size()) {
    const size_t larger = PxMax(gBreakPollReported.size(),
                                gBreakCallbackReported.size());
    const size_t smaller = PxMin(gBreakPollReported.size(),
                                 gBreakCallbackReported.size());
    gGateStats.breakCallbackPollMismatches +=
        static_cast<PxU32>(larger - smaller);
  }

  if (usesProjectileExcitation()) {
    evaluation.expectedHits = static_cast<PxU32>(gChains.size());
    evaluation.minProjectileResponseFraction = PX_MAX_F32;
    for (PxU32 i = 0; i < gProjectiles.size(); ++i) {
      const ProjectileRecord &projectile = gProjectiles[i];
      evaluation.crossChainContacts += projectile.crossChainContacts;
      if (projectile.contactCount)
        evaluation.hitChains++;
      if (projectile.contactCount &&
          (projectile.contactPointCount == 0 ||
           projectile.firstContactFrame + gProjectileObservationFrames >
               gGateStats.completedFrames))
        evaluation.incompleteObservations++;

      if (projectile.firstWrongContactFrame != PX_MAX_U32 &&
          (projectile.firstContactFrame == PX_MAX_U32 ||
           projectile.firstWrongContactFrame <= projectile.firstContactFrame))
        evaluation.wrongFirstContacts++;
      const PxReal responseFraction =
          projectile.maxImpactAxisVelocityDelta / gImpactSpeed;
      evaluation.minProjectileResponseFraction =
          PxMin(evaluation.minProjectileResponseFraction, responseFraction);
      if (responseFraction >= gMinProjectileResponseFraction)
        evaluation.responseProjectiles++;
      const TargetResponseKind targetResponseKind =
          getTargetResponseKind(projectile.targetChain);
      if (targetResponseKind != eTARGET_RESPONSE_NONE) {
        if (!evaluation.expectedTargetResponses)
          evaluation.minTargetResponseFraction = PX_MAX_F32;
        evaluation.expectedTargetResponses++;
        const PxReal targetResponseFraction =
            getTargetResponseFraction(projectile);
        evaluation.minTargetResponseFraction = PxMin(
            evaluation.minTargetResponseFraction, targetResponseFraction);
        if (targetResponseFraction >= gMinTargetResponseFraction)
          evaluation.respondedTargetChains++;
        if (projectile.contactCount && !projectile.targetResponseSamples)
          evaluation.incompleteTargetResponses++;
        evaluation.linearTargetResponses++;
      } else
        evaluation.notApplicableTargetResponses++;
    }
    if (gProjectiles.empty()) {
      evaluation.minProjectileResponseFraction = 0.0f;
      evaluation.minTargetResponseFraction = 0.0f;
    }
    if (!evaluation.expectedTargetResponses)
      evaluation.minTargetResponseFraction = 0.0f;
  }

  if (gGateStats.completedFrames != gHeadlessOptions.frames ||
      gGateStats.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (gGateStats.launchFailures)
    setGateError(evaluation, isRevoluteEndpointProbe()
                                 ? "endpoint_launch"
                                 : "projectile_launch");
  if (usesProjectileExcitation() &&
      gProjectiles.size() != gChains.size())
    setGateError(evaluation, "projectile_launch");
  if (usesProjectileExcitation() &&
      evaluation.hitChains != evaluation.expectedHits)
    setGateFailure(evaluation, "missing_target_contact");
  if (usesProjectileExcitation() && evaluation.wrongFirstContacts)
    setGateFailure(evaluation, "wrong_first_contact");
  if (usesProjectileExcitation() && evaluation.incompleteObservations)
    setGateFailure(evaluation, "incomplete_impact_observation");
  if (usesProjectileExcitation() && evaluation.incompleteTargetResponses)
    setGateFailure(evaluation, "incomplete_target_response");
  if (usesProjectileExcitation() &&
      evaluation.responseProjectiles != evaluation.expectedHits)
    setGateFailure(evaluation, "missing_impact_response");
  if (usesProjectileExcitation() &&
      evaluation.respondedTargetChains != evaluation.expectedTargetResponses)
    setGateFailure(evaluation, "missing_target_response");

  if (gErrorCallback.getFatalCount() || gGateStats.fetchErrorState) {
    if (isEndpointProbe() || isForceReactionCase() ||
        isSphericalConeCase() || isNativeBreakReactionCase() ||
        isRevoluteMotorFamilyCase())
      setInfrastructureErrorOverFailure(evaluation, "physx_error");
    else
      setGateFailure(evaluation, "physx_error");
  }
  if (gGateStats.nonFinite)
    setGateFailure(evaluation, "non_finite");
  if (gGateStats.maxQuaternionNormError > 1e-3f)
    setGateFailure(evaluation, "quaternion_norm");
  if (gGateStats.maxAbsPosition > 100000.0f ||
      gGateStats.maxLinearSpeed > 10000.0f ||
      gGateStats.maxAngularSpeed > 10000.0f)
    setGateFailure(evaluation, "runaway");

  if (isRevoluteMotorCase()) {
    if (gScene) {
      gRevoluteMotorStats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      gRevoluteMotorStats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      gRevoluteMotorStats.finalConstraints =
          gScene->getNbConstraints();
    }
    if (!gRevoluteMotorJoint || !gRevoluteMotorBodyA ||
        !gRevoluteMotorBodyB ||
        !gRevoluteMotorStats.actorOrderValid ||
        !gRevoluteMotorStats.driveEnabledReadback ||
        gRevoluteMotorStats.stateSamples != gHeadlessOptions.frames ||
        gRevoluteMotorStats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_fixture");
    if (gRevoluteMotorStats.finalRelativeError >
            gRevoluteMotorRelativeErrorMaximum ||
        gRevoluteMotorStats.maximumLateRelativeError >
            gRevoluteMotorRelativeErrorMaximum)
      setGateFailure(evaluation, "revolute_motor_relative_velocity");
    if (gRevoluteMotorStats.maximumAngularMomentumDrift >
        gRevoluteMotorMomentumDriftMaximum)
      setGateFailure(evaluation, "revolute_motor_momentum");
    if (gRevoluteMotorStats.maximumAnchorError >
            gRevoluteMotorAnchorErrorMaximum ||
        gRevoluteMotorStats.maximumAxisMisalignment >
            gRevoluteMotorAxisMisalignmentMaximum)
      setGateFailure(evaluation, "revolute_motor_joint_error");
  }
  if (isRevoluteMotorLimitCase()) {
    RevoluteMotorLimitStats &stats = gRevoluteMotorLimitStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorLimitJoint || !gRevoluteMotorLimitBody ||
        (isRevoluteMotorDynamicLimitCase() &&
         !gRevoluteMotorLimitBodyA) ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.limitEnabledReadback ||
        stats.reverseEvents != 1 ||
        stats.finalTargetVelocityReadback !=
            -gRevoluteMotorLimitTargetVelocity ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_limit_fixture");

    const PxReal upperTravel =
        stats.maximumAngle - stats.initialAngle;
    const PxReal range = stats.maximumAngle - stats.minimumAngle;
    if (upperTravel < gRevoluteMotorLimitTravelMinimum ||
        range < gRevoluteMotorLimitRangeMinimum ||
        stats.finalAngle >
            gRevoluteMotorLimitLower +
                gRevoluteMotorLimitFinalTolerance)
      setGateFailure(evaluation, "revolute_motor_limit_travel");
    if (stats.maximumUpperViolation >
            gRevoluteMotorLimitViolationMaximum ||
        stats.maximumLowerViolation >
            gRevoluteMotorLimitViolationMaximum)
      setGateFailure(evaluation, "revolute_motor_limit_violation");
    if (stats.maximumLateOutwardVelocity >
        gRevoluteMotorLimitOutwardVelocityMaximum)
      setGateFailure(evaluation,
                     "revolute_motor_limit_outward_velocity");
    if (stats.maximumAnchorError >
            gRevoluteMotorLimitAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorLimitAxisMisalignmentMaximum)
      setGateFailure(evaluation,
                     "revolute_motor_limit_joint_error");
  }
  if (isRevoluteMotorFreeSpinCase()) {
    RevoluteMotorFreeSpinStats &stats =
        gRevoluteMotorFreeSpinStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorFreeSpinJoint ||
        !gRevoluteMotorFreeSpinBody ||
        (isRevoluteMotorDynamicFreeSpinCase() &&
         !gRevoluteMotorFreeSpinBodyA) ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.freeSpinEnabledReadback ||
        !stats.limitDisabledReadback || stats.boostEvents != 1 ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0 ||
        stats.minimumPostBoostVelocity == PX_MAX_F32)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_freespin_fixture");
    if (PxAbs(stats.preBoostFinalVelocity -
              gRevoluteMotorFreeSpinTargetVelocity) >
            gRevoluteMotorFreeSpinTargetErrorMaximum ||
        stats.maximumLatePreBoostError >
            gRevoluteMotorFreeSpinTargetErrorMaximum)
      setGateFailure(
          evaluation, "revolute_motor_freespin_acceleration");
    if (stats.minimumPostBoostVelocity <
            gRevoluteMotorFreeSpinMinimumCoastVelocity ||
        stats.finalVelocity <
            gRevoluteMotorFreeSpinMinimumCoastVelocity ||
        stats.maximumPostBoostVelocityDrop >
            gRevoluteMotorFreeSpinVelocityDropMaximum)
      setGateFailure(
          evaluation, "revolute_motor_freespin_braking");
    if (isRevoluteMotorDynamicFreeSpinCase() &&
        stats.maximumAngularMomentumDrift >
            gRevoluteMotorMomentumDriftMaximum)
      setGateFailure(
          evaluation, "revolute_motor_freespin_momentum");
    if (stats.maximumAnchorError >
            gRevoluteMotorFreeSpinAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorFreeSpinAxisMisalignmentMaximum)
      setGateFailure(
          evaluation, "revolute_motor_freespin_joint_error");
  }
  if (isRevoluteMotorRatioCase()) {
    RevoluteMotorRatioStats &stats = gRevoluteMotorRatioStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorRatioJoint ||
        !gRevoluteMotorRatioBodyA || !gRevoluteMotorRatioBodyB ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.freeSpinDisabledReadback ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_ratio_fixture");
    if (stats.finalWeightedVelocityError >
            gRevoluteMotorRatioVelocityErrorMaximum ||
        stats.maximumLateWeightedVelocityError >
            gRevoluteMotorRatioVelocityErrorMaximum)
      setGateFailure(
          evaluation, "revolute_motor_ratio_weighted_velocity");
    const PxReal gatedMomentumDrift =
        isRevoluteMotorDynamicOffCenterCase()
            ? stats.maximumInitialTotalAngularMomentum
            : (isRevoluteMotorDynamicOffPrincipalCase()
                   ? stats.maximumInitialGeneralizedMomentumDrift
                   : stats.maximumGeneralizedMomentumDrift);
    const PxReal momentumDriftMaximum =
        isRevoluteMotorDynamicOffCenterCase()
            ? gRevoluteMotorDynamicOffCenterInitialAngularMomentumMaximum
            : (isRevoluteMotorDynamicOffPrincipalCase()
                   ? gRevoluteMotorDynamicOffPrincipalInitialMomentumDriftMaximum
                   : gRevoluteMotorRatioMomentumDriftMaximum);
    if (gatedMomentumDrift >
        momentumDriftMaximum)
      setGateFailure(
          evaluation, "revolute_motor_ratio_generalized_momentum");
    if (isRevoluteMotorDynamicOffPrincipalCase() &&
        (stats.initialOffPrincipalResponseA <
             gRevoluteMotorDynamicOffPrincipalResponseMinimum ||
         stats.initialOffPrincipalResponseB <
             gRevoluteMotorDynamicOffPrincipalResponseMinimum))
      setInfrastructureErrorOverFailure(
          evaluation,
          "revolute_motor_dynamic_off_principal_fixture");
    if (isRevoluteMotorDynamicOffPrincipalCase() &&
        stats.maximumLateRelativeSwingVelocity >
            gRevoluteMotorDynamicOffPrincipalSwingVelocityMaximum)
      setGateFailure(
          evaluation,
          "revolute_motor_dynamic_off_principal_swing");
    if (isRevoluteMotorDynamicOffCenterCase() &&
        stats.maximumLateRelativeSwingVelocity >
            gRevoluteMotorDynamicOffPrincipalSwingVelocityMaximum)
      setGateFailure(
          evaluation,
          "revolute_motor_dynamic_off_center_swing");
    if (isRevoluteMotorDynamicOffCenterCase() &&
        (stats.initialPerpendicularLeverArmA <
             gRevoluteMotorDynamicOffCenterLeverArmMinimum ||
         stats.initialPerpendicularLeverArmB <
             gRevoluteMotorDynamicOffCenterLeverArmMinimum))
      setInfrastructureErrorOverFailure(
          evaluation,
          "revolute_motor_dynamic_off_center_fixture");
    if (isRevoluteMotorDynamicOffCenterCase() &&
        (stats.finalRelativeAnchorPointSpeed >
             gRevoluteMotorDynamicOffCenterAnchorSpeedMaximum ||
         stats.maximumLateRelativeAnchorPointSpeed >
             gRevoluteMotorDynamicOffCenterAnchorSpeedMaximum))
      setGateFailure(
          evaluation,
          "revolute_motor_dynamic_off_center_anchor_velocity");
    if (isRevoluteMotorDynamicOffCenterCase() &&
        stats.maximumTotalLinearMomentum >
            gRevoluteMotorDynamicOffCenterLinearMomentumMaximum)
      setGateFailure(
          evaluation,
          "revolute_motor_dynamic_off_center_linear_momentum");
    if (isRevoluteMotorDynamicOffCenterCase() &&
        stats.maximumLinearSpeed <
            gRevoluteMotorDynamicOffCenterLinearSpeedMinimum)
      setGateFailure(
          evaluation,
          "revolute_motor_dynamic_off_center_motion");
    if (stats.maximumAnchorError >
            gRevoluteMotorRatioAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorRatioAxisMisalignmentMaximum)
      setGateFailure(
          evaluation, "revolute_motor_ratio_joint_error");
  }
  if (isRevoluteMotorContactCase()) {
    RevoluteMotorContactStats &stats =
        gRevoluteMotorContactStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorContactJoint ||
        !gRevoluteMotorContactBodyA ||
        !gRevoluteMotorContactBodyB ||
        !gRevoluteMotorContactGround ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0 ||
        stats.contactEvents == 0 || stats.contactPointCount == 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_contact_fixture");
    if (stats.finalRelativeError >
            gRevoluteMotorContactVelocityErrorMaximum ||
        stats.maximumLateRelativeError >
            gRevoluteMotorContactVelocityErrorMaximum)
      setGateFailure(
          evaluation, "revolute_motor_contact_relative_velocity");
    const PxReal meanLateDriveReaction =
        stats.lateDriveReactionSamples
            ? stats.lateDriveReactionSum /
                  PxReal(stats.lateDriveReactionSamples)
            : 0.0f;
    if (stats.lateDriveReactionSamples == 0 ||
        meanLateDriveReaction <
            gRevoluteMotorContactDriveReactionMinimum)
      setGateFailure(
          evaluation, "revolute_motor_contact_drive_reaction");
    if (stats.maximumAnchorError >
            gRevoluteMotorContactAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorContactAxisMisalignmentMaximum ||
        stats.maximumCenterHeightError >
            gRevoluteMotorContactCenterHeightErrorMaximum)
      setGateFailure(
          evaluation, "revolute_motor_contact_joint_error");
  }
  if (isRevoluteMotorKinematicCase()) {
    RevoluteMotorKinematicStats &stats =
        gRevoluteMotorKinematicStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorKinematicJoint ||
        !gRevoluteMotorKinematicBody ||
        !gRevoluteMotorKinematicDynamicBody ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        !stats.kinematicFlagReadback ||
        stats.targetUpdates != gHeadlessOptions.frames ||
        stats.targetUpdateFailures != 0 ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_kinematic_fixture");
    if (stats.finalRelativeError >
            gRevoluteMotorKinematicVelocityErrorMaximum ||
        stats.maximumLateRelativeError >
            gRevoluteMotorKinematicVelocityErrorMaximum ||
        stats.maximumLateKinematicVelocityError >
            gRevoluteMotorKinematicVelocityErrorMaximum ||
        stats.maximumLateDynamicVelocityError >
            gRevoluteMotorKinematicVelocityErrorMaximum)
      setGateFailure(
          evaluation, "revolute_motor_kinematic_velocity");
    if (stats.maximumAnchorError >
            gRevoluteMotorKinematicAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorKinematicAxisMisalignmentMaximum)
      setGateFailure(
          evaluation, "revolute_motor_kinematic_joint_error");
  }
  if (isRevoluteMotorOffPrincipalCase()) {
    RevoluteMotorOffPrincipalStats &stats =
        gRevoluteMotorOffPrincipalStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorOffPrincipalJoint ||
        !gRevoluteMotorOffPrincipalBody ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        stats.initialOffPrincipalResponse <
            gRevoluteMotorOffPrincipalResponseMinimum ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_off_principal_fixture");
    if (stats.finalHingeVelocityError >
            gRevoluteMotorOffPrincipalVelocityErrorMaximum ||
        stats.maximumLateHingeVelocityError >
            gRevoluteMotorOffPrincipalVelocityErrorMaximum)
      setGateFailure(
          evaluation, "revolute_motor_off_principal_velocity");
    if (stats.maximumLateSwingVelocity >
        gRevoluteMotorOffPrincipalSwingVelocityMaximum)
      setGateFailure(
          evaluation, "revolute_motor_off_principal_swing");
    if (stats.maximumAnchorError >
            gRevoluteMotorOffPrincipalAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorOffPrincipalAxisMisalignmentMaximum)
      setGateFailure(
          evaluation, "revolute_motor_off_principal_joint_error");
  }
  if (isRevoluteMotorOffCenterCase()) {
    RevoluteMotorOffCenterStats &stats =
        gRevoluteMotorOffCenterStats;
    if (gScene) {
      stats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      stats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      stats.finalConstraints = gScene->getNbConstraints();
    }
    if (!gRevoluteMotorOffCenterJoint ||
        !gRevoluteMotorOffCenterBody ||
        !stats.actorOrderValid || !stats.driveEnabledReadback ||
        stats.initialPerpendicularLeverArm <
            gRevoluteMotorOffCenterLeverArmMinimum ||
        (isRevoluteMotorSpatialCase() &&
         stats.initialOffPrincipalResponse <
             gRevoluteMotorOffPrincipalResponseMinimum) ||
        stats.stateSamples != gHeadlessOptions.frames ||
        stats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(
          evaluation, "revolute_motor_off_center_fixture");
    if (stats.finalHingeVelocityError >
            gRevoluteMotorOffCenterVelocityErrorMaximum ||
        stats.maximumLateHingeVelocityError >
            gRevoluteMotorOffCenterVelocityErrorMaximum)
      setGateFailure(
          evaluation,
          isRevoluteMotorSpatialCase()
              ? "revolute_motor_spatial_velocity"
              : "revolute_motor_off_center_velocity");
    if (isRevoluteMotorSpatialCase() &&
        stats.maximumLateSwingVelocity >
            gRevoluteMotorOffPrincipalSwingVelocityMaximum)
      setGateFailure(
          evaluation, "revolute_motor_spatial_swing");
    if (stats.finalAnchorPointSpeed >
            gRevoluteMotorOffCenterAnchorSpeedMaximum ||
        stats.maximumLateAnchorPointSpeed >
            gRevoluteMotorOffCenterAnchorSpeedMaximum)
      setGateFailure(
          evaluation,
          isRevoluteMotorSpatialCase()
              ? "revolute_motor_spatial_anchor_velocity"
              : "revolute_motor_off_center_anchor_velocity");
    if (stats.maximumAnchorError >
            gRevoluteMotorOffCenterAnchorErrorMaximum ||
        stats.maximumAxisMisalignment >
            gRevoluteMotorOffCenterAxisMisalignmentMaximum)
      setGateFailure(
          evaluation,
          isRevoluteMotorSpatialCase()
              ? "revolute_motor_spatial_joint_error"
              : "revolute_motor_off_center_joint_error");
  }
  if (isImpactCase() &&
      (gGateStats.maxLinearSpeed > gImpactLinearSpeedCap ||
       gGateStats.maxAngularSpeed > gImpactAngularSpeedCap))
    setGateFailure(evaluation, "impact_runaway");
  if (!isEndpointProbe() && gGateStats.maxAnchorError > gAnchorErrorCap)
    setGateFailure(evaluation, "anchor_error");
  if (gGateStats.maxRevoluteLimitViolation > gAngularLimitViolationCap ||
      gGateStats.maxSphericalLimitViolation > gAngularLimitViolationCap)
    setGateFailure(evaluation, "angular_limit");
  if (!isEndpointProbe() &&
      gGateStats.maxPrismaticLimitViolation > gLinearLimitViolationCap)
    setGateFailure(evaluation, "linear_limit");
  if (!isEndpointProbe() && gGateStats.maxConstrainedAngularError >
      gConstrainedAngularErrorCap)
    setGateFailure(evaluation, "constrained_angular_error");
  if (evaluation.crossChainContacts)
    setGateFailure(evaluation, "cross_chain_contact");
  PxU32 polledBreakCount = 0;
  for (PxU32 i = 0; i < gBreakPollReported.size(); ++i)
    polledBreakCount += gBreakPollReported[i] ? 1u : 0u;
  if (polledBreakCount != gGateStats.breakCallbackCount)
    setGateFailure(evaluation, "break_callback_mismatch");
  if (gGateStats.breakCallbackIdentityMatches !=
          gGateStats.breakCallbackCount ||
      gGateStats.breakCallbackConstraintMismatches ||
      gGateStats.breakCallbackExternalReferenceMismatches ||
      gGateStats.breakCallbackTypeMismatches ||
      gGateStats.breakCallbackBrokenFlagMismatches ||
      gGateStats.breakCallbackDuplicateMismatches ||
      gGateStats.breakCallbackPollMismatches)
    setGateFailure(evaluation, "break_callback_identity");

  if (gHeadlessCase == eCASE_PASSIVE) {
    if (gFixedStats.brokenCount || gGateStats.breakCallbackCount)
      setGateFailure(evaluation, "unexpected_break");
    if (evaluation.jitterReproduced)
      setGateFailure(evaluation, "revolute_jitter");
  } else if (gHeadlessCase == eCASE_FIXED_NO_BREAK) {
    if (gFixedStats.brokenCount || gGateStats.breakCallbackCount)
      setGateFailure(evaluation, "unexpected_break");
  } else if (gHeadlessCase == eCASE_FIXED_BREAK) {
    if (gFixedStats.brokenCount != 1 || gGateStats.breakCallbackCount != 1)
      setGateFailure(evaluation, "break_count");
    if (gGateStats.breakCallbackIdentityMatches != 1)
      setGateFailure(evaluation, "break_callback_identity");
    if (!gProjectiles.empty() &&
        gFixedStats.firstBrokenFrame < gProjectiles[0].firstContactFrame)
      setGateFailure(evaluation, "break_before_contact");
  } else if (isNativeBreakReactionCase()) {
    if (gScene) {
      gNativeBreakReactionStats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      gNativeBreakReactionStats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      gNativeBreakReactionStats.finalConstraints =
          gScene->getNbConstraints();
    }
    if (!gNativeBreakReactionStats.actorOrderValid ||
        gNativeBreakReactionStats.initialDynamicActors != 1 ||
        gNativeBreakReactionStats.initialStaticActors != 0 ||
        gNativeBreakReactionStats.initialConstraints != 1 ||
        gNativeBreakReactionStats.finalDynamicActors != 1 ||
        gNativeBreakReactionStats.finalStaticActors != 0 ||
        gNativeBreakReactionStats.finalConstraints != 1)
      setInfrastructureErrorOverFailure(evaluation, "native_topology");
    if (gNativeBreakReactionStats.stateSamples !=
            gHeadlessOptions.frames ||
        gNativeBreakReactionStats.forceReads !=
            gHeadlessOptions.frames ||
        gNativeBreakReactionStats.nonFiniteSamples)
      setInfrastructureErrorOverFailure(evaluation,
                                        "native_sample_accounting");

    if (isNativeBreakCase()) {
      if (gNativeBreakReactionStats.brokenPollCount != 1 ||
          gGateStats.breakCallbackCount != 1)
        setGateFailure(evaluation, "native_constraint_not_broken");
      if (gGateStats.breakCallbackIdentityMatches != 1)
        setGateFailure(evaluation, "break_callback_identity");
      if (gNativeBreakReactionStats.firstBrokenFrame == PX_MAX_U32 ||
          gNativeBreakReactionStats.firstBrokenFrame >
              gNativeReactionWarmupFrames)
        setGateFailure(evaluation, "native_break_timing");
      const PxReal postBreakSpeed =
          isNativeAngularReactionCase()
              ? gNativeBreakReactionStats.maximumAngularSpeed
              : gNativeBreakReactionStats.maximumLinearSpeed;
      const PxReal requiredSpeed =
          isNativeAngularReactionCase()
              ? gNativePostBreakAngularSpeedMinimum
              : gNativePostBreakLinearSpeedMinimum;
      if (postBreakSpeed < requiredSpeed)
        setGateFailure(evaluation, "native_post_break_motion");
    } else {
      if (gNativeBreakReactionStats.brokenPollCount ||
          gGateStats.breakCallbackCount)
        setGateFailure(evaluation, "native_constraint_unexpected_break");
      const PxU32 expectedReactionSamples =
          gHeadlessOptions.frames - gNativeReactionWarmupFrames;
      if (gNativeBreakReactionStats.reactionSamples !=
          expectedReactionSamples)
        setInfrastructureErrorOverFailure(evaluation,
                                          "native_reaction_samples");
      const PxReal expectedMagnitude =
          getNativeExpectedReactionVector().magnitude();
      const PxReal reactionRatio =
          expectedMagnitude > 0.0f
              ? getNativeMeanReactionMagnitude() / expectedMagnitude
              : 0.0f;
      if (reactionRatio < gNativeReactionRatioMinimum ||
          reactionRatio > gNativeReactionRatioMaximum)
        setGateFailure(evaluation, "native_reaction_magnitude");
      if (getNativeReactionDirectionDot() <
          gNativeReactionDirectionMinimum)
        setGateFailure(evaluation, "native_reaction_direction");
      if (getNativeReactionOrthogonalRatio() >
          gNativeReactionOrthogonalRatioMaximum)
        setGateFailure(evaluation, "native_reaction_orthogonal");
      if (gNativeBreakReactionStats.steadyMaximumPositionError >
              gNativeConstrainedErrorMaximum ||
          gNativeBreakReactionStats.steadyMaximumRotationError >
              gNativeConstrainedErrorMaximum ||
          gNativeBreakReactionStats.steadyMaximumLinearSpeed >
              gNativeConstrainedSpeedMaximum ||
          gNativeBreakReactionStats.steadyMaximumAngularSpeed >
              gNativeConstrainedSpeedMaximum)
        setGateFailure(evaluation, "native_constraint_stability");
    }
  } else if (isSphericalConeCase()) {
    if (gScene) {
      gSphericalConeStats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      gSphericalConeStats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      gSphericalConeStats.finalConstraints = gScene->getNbConstraints();
    }
    const PxU32 expectedDynamicActors =
        gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC ? 2u : 1u;
    const PxU32 expectedStaticActors =
        gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC ? 0u : 1u;
    const PxReal expectedInitialY =
        isSphericalConeInsideCase() ? gSphericalConeInsideY
                                    : gSphericalConeOutsideY;
    const PxReal expectedInitialZ =
        isSphericalConeInsideCase() ? gSphericalConeInsideZ
                                    : gSphericalConeOutsideZ;
    const PxReal expectedInitialRadius =
        PxSqrt(PxSqr(expectedInitialY / gSphericalConeLimitY) +
               PxSqr(expectedInitialZ / gSphericalConeLimitZ));
    if (!gSphericalConeJoint || !gSphericalConeStats.actorOrderValid ||
        !gSphericalConeStats.limitEnabledReadback ||
        PxAbs(gSphericalConeStats.limitYReadback -
              gSphericalConeLimitY) > 1e-6f ||
        PxAbs(gSphericalConeStats.limitZReadback -
              gSphericalConeLimitZ) > 1e-6f ||
        PxAbs(gSphericalConeStats.initialEllipseRadius -
              expectedInitialRadius) > 1e-4f)
      setInfrastructureErrorOverFailure(evaluation,
                                        "spherical_cone_fixture");
    if (gSphericalConeStats.initialDynamicActors !=
            expectedDynamicActors ||
        gSphericalConeStats.initialStaticActors !=
            expectedStaticActors ||
        gSphericalConeStats.initialConstraints != 1 ||
        gSphericalConeStats.finalDynamicActors !=
            expectedDynamicActors ||
        gSphericalConeStats.finalStaticActors !=
            expectedStaticActors ||
        gSphericalConeStats.finalConstraints != 1)
      setInfrastructureErrorOverFailure(evaluation,
                                        "spherical_cone_topology");
    if (gSphericalConeStats.stateSamples != gHeadlessOptions.frames ||
        gSphericalConeStats.nonFiniteSamples != 0)
      setInfrastructureErrorOverFailure(evaluation,
                                        "spherical_cone_samples");
    if (gSphericalConeStats.maximumAnchorSeparation >
        gSphericalConeAnchorSeparationMaximum)
      setGateFailure(evaluation, "spherical_cone_anchor");
    if (gSphericalConeTopology == eSPHERICAL_CONE_DYNAMIC_DYNAMIC &&
        gSphericalConeStats.maximumTotalAngularMomentum >
            gSphericalConeAngularMomentumMaximum)
      setGateFailure(evaluation, "spherical_cone_conservation");

    if (isSphericalConeInsideCase()) {
      if (gSphericalConeStats.initialEllipseRadius >= 1.0f)
        setInfrastructureErrorOverFailure(evaluation,
                                          "spherical_cone_inside_fixture");
      if (gSphericalConeStats.maximumInsideDeviation >
          gSphericalConeInsideDeviationTolerance)
        setGateFailure(evaluation,
                       "spherical_cone_inside_state_disturbed");
    } else {
      const PxReal radiusCorrection =
          gSphericalConeStats.initialEllipseRadius -
          gSphericalConeStats.finalEllipseRadius;
      if (gSphericalConeStats.initialEllipseRadius <= 1.0f)
        setInfrastructureErrorOverFailure(evaluation,
                                          "spherical_cone_outside_fixture");
      if (gSphericalConeStats.finalEllipseRadius <
              1.0f - gSphericalConeFinalRadiusTolerance ||
          gSphericalConeStats.finalEllipseRadius >
              1.0f + gSphericalConeFinalRadiusTolerance ||
          gSphericalConeStats.minimumLateEllipseRadius <
              1.0f - gSphericalConeLateRadiusTolerance ||
          gSphericalConeStats.maximumLateEllipseRadius >
              1.0f + gSphericalConeLateRadiusTolerance ||
          radiusCorrection < gSphericalConeMinimumRadiusCorrection)
        setGateFailure(evaluation,
                       "spherical_cone_limit_not_enforced");
    }
  } else if (isForceReactionCase()) {
    if (gScene) {
      gForceStaticStats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      gForceStaticStats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      gForceStaticStats.finalConstraints = gScene->getNbConstraints();
    }
    if (gForceStaticStats.forceReads != gHeadlessOptions.frames ||
        gForceStaticStats.stateSamples != gHeadlessOptions.frames ||
        gForceStaticStats.steadySampleAttempts !=
            gForceStaticStats.expectedSteadySamples ||
        gForceStaticStats.steadySamples +
                gForceStaticStats.nonFiniteSteadyForceSamples !=
            gForceStaticStats.steadySampleAttempts)
      setInfrastructureErrorOverFailure(evaluation,
                                        "force_sample_accounting");
    const PxU32 expectedDynamicActors = isForcePairCase() ? 2u : 1u;
    if (gForceStaticStats.topologyDynamicActors != expectedDynamicActors ||
        gForceStaticStats.topologyStaticActors != 0 ||
        gForceStaticStats.topologyConstraints != 1 ||
        gForceStaticStats.finalDynamicActors != expectedDynamicActors ||
        gForceStaticStats.finalStaticActors != 0 ||
        gForceStaticStats.finalConstraints != 1)
      setInfrastructureErrorOverFailure(evaluation, "force_topology");
    if (isForcePairDisabledCase()) {
      const PxReal meanSampleMagnitude =
          gForceStaticStats.steadySamples
              ? gForceStaticStats.linearMagnitudeSum /
                    PxReal(gForceStaticStats.steadySamples)
              : 0.0f;
      if (meanSampleMagnitude > gDisabledPairReactionMaximum ||
          getForceStaticTorqueRms() > gDisabledPairReactionMaximum)
        setGateFailure(evaluation, "disabled_constraint_reaction");
      if (gForceStaticStats.pairMaxSeparationError <
              gDisabledPairSeparationErrorMinimum ||
          gForceStaticStats.pairMaxRelativeSpeed <
              gDisabledPairRelativeSpeedMinimum)
        setGateFailure(evaluation, "disabled_constraint_motion");
    } else {
      const PxReal forceRatio = getForceStaticRatio();
      const PxReal meanSampleForceRatio = getForceStaticMeanSampleRatio();
      if (forceRatio < gForceRatioMinimum || forceRatio > gForceRatioMaximum ||
          meanSampleForceRatio < gForceRatioMinimum ||
          meanSampleForceRatio > gForceRatioMaximum)
        setGateFailure(evaluation, "force_unit_dt");
      if (getForceStaticDirectionDot() < gForceDirectionDotMinimum)
        setGateFailure(evaluation, "force_direction");
      if (getForceStaticOrthogonalRmsRatio() >
          gForceOrthogonalRatioMaximum)
        setGateFailure(evaluation, "force_orthogonal");
      if (!isForceOffsetCase()) {
        if (getForceStaticTorqueRms() > gForceTorqueMaximum ||
            gForceStaticStats.maxAngularMagnitude > gForceTorqueMaximum)
          setGateFailure(evaluation, "force_torque");
      } else {
        const PxReal torqueRatio = getForceStaticTorqueRatio();
        const PxReal meanSampleTorqueRatio =
            getForceStaticMeanSampleTorqueRatio();
        if (torqueRatio < gForceTorqueRatioMinimum ||
            torqueRatio > gForceTorqueRatioMaximum ||
            meanSampleTorqueRatio < gForceTorqueRatioMinimum ||
            meanSampleTorqueRatio > gForceTorqueRatioMaximum)
          setGateFailure(evaluation, "force_offset_torque_unit_dt");
        if (getForceStaticTorqueDirectionDot() <
            gForceTorqueDirectionDotMinimum)
          setGateFailure(evaluation, "force_offset_torque_direction");
        if (getForceStaticTorqueOrthogonalRmsRatio() >
            gForceTorqueOrthogonalRatioMaximum)
          setGateFailure(evaluation, "force_offset_torque_orthogonal");
      }
      if (gForceStaticStats.maxPositionError > gForcePositionErrorMaximum ||
          gForceStaticStats.maxRotationError > gForceRotationErrorMaximum ||
          gForceStaticStats.maxLinearSpeed > gForceLinearSpeedMaximum ||
          gForceStaticStats.maxAngularSpeed > gForceAngularSpeedMaximum)
        setGateFailure(evaluation, "force_body_stability");
    }
    if (isForcePairCase()) {
      if (!gForceStaticStats.pairActorOrderValid)
        setInfrastructureErrorOverFailure(evaluation,
                                          "force_pair_actor_order");
      const PxReal totalMomentumMaximum =
          isForcePairDisabledCase()
              ? gDisabledPairTotalMomentumMaximum
              : gForcePairTotalMomentumMaximum;
      if (gForceStaticStats.pairMaxTotalMomentum >
          totalMomentumMaximum)
        setGateFailure(evaluation, "force_pair_momentum");
      const PxReal centerOfMassErrorMaximum =
          isForcePairDisabledCase()
              ? gDisabledPairCenterOfMassErrorMaximum
              : gForcePairCenterOfMassErrorMaximum;
      if (gForceStaticStats.pairMaxCenterOfMassError >
          centerOfMassErrorMaximum)
        setGateFailure(evaluation, "force_pair_center_of_mass");
      if (!isForcePairDisabledCase() &&
          (gForceStaticStats.pairMaxSeparationError >
               gForcePairSeparationErrorMaximum ||
           gForceStaticStats.pairMaxRelativeSpeed >
               gForcePairRelativeSpeedMaximum))
        setGateFailure(evaluation, "force_pair_relative_stability");
    }
  }

  if (isEndpointProbe()) {
    if (gScene) {
      gEndpointStats.finalDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      gEndpointStats.finalStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      gEndpointStats.finalConstraints = gScene->getNbConstraints();
    }
    if (isRevoluteEndpointProbe()) {
      if (!gEndpointRevoluteJoint || gEndpointJoint ||
          !gEndpointStats.actorOrderValid ||
          !gEndpointStats.frameWitnessValid ||
          !gEndpointStats.fixtureWitnessValid ||
          !gEndpointAngularStats.nonIdentityWitnessValid ||
          gEndpointStats.limitEnabled ||
          gEndpointAngularStats.driveEnabled)
        setInfrastructureErrorOverFailure(evaluation, "endpoint_fixture");

      const PxReal actualAxialLaunch =
          gEndpointAngularStats.actualLaunchAngularVelocity.dot(
              gEndpointStats.expectedAxis);
      const PxReal actualTransverseLaunch =
          gEndpointAngularStats.actualLaunchAngularVelocity.dot(
              gEndpointAngularStats.perpendicularAxis);
      const PxVec3 unexplainedLaunch =
          gEndpointAngularStats.actualLaunchAngularVelocity -
          gEndpointStats.expectedAxis * actualAxialLaunch -
          gEndpointAngularStats.perpendicularAxis * actualTransverseLaunch;
      if (gEndpointAngularStats.launchAttempts != 1 ||
          gEndpointAngularStats.launchSuccesses != 1 ||
          gEndpointStats.actualLaunchFrame != gImpactLaunchFrame ||
          !gEndpointAngularStats.launchWakeValid ||
          !gEndpointStats.launchDirection.isFinite() ||
          !gEndpointAngularStats.actualLaunchAngularVelocity.isFinite() ||
          gEndpointAngularStats.launchVelocityError > 1e-6f ||
          PxAbs(actualAxialLaunch -
                gEndpointAngularAxialLaunchSpeed) > 1e-5f ||
          PxAbs(actualTransverseLaunch -
                gEndpointAngularTransverseLaunchSpeed) > 1e-5f ||
          getSafeMagnitude(unexplainedLaunch) > 1e-5f ||
          !gProjectiles.empty())
        setInfrastructureErrorOverFailure(evaluation, "endpoint_launch");

      if (gEndpointStats.initialDynamicActors != 1 ||
          gEndpointStats.initialStaticActors != 0 ||
          gEndpointStats.initialConstraints != 1 ||
          gEndpointStats.finalDynamicActors != 1 ||
          gEndpointStats.finalStaticActors != 0 ||
          gEndpointStats.finalConstraints != 1)
        setInfrastructureErrorOverFailure(evaluation, "endpoint_topology");

      const PxU32 expectedTailSamples =
          gEndpointAngularResponseWindowFrames -
          gEndpointAngularSettleFrames;
      if (gEndpointStats.stateSampleAttempts != gHeadlessOptions.frames ||
          gEndpointStats.stateSamples != gHeadlessOptions.frames ||
          gEndpointStats.nonFiniteStateSamples != 0 ||
          gEndpointStats.responseBaselineSamples != gImpactLaunchFrame ||
          gEndpointStats.responseSampleAttempts !=
              gEndpointAngularResponseWindowFrames ||
          gEndpointStats.responseSamples !=
              gEndpointAngularResponseWindowFrames ||
          gEndpointStats.nonFiniteResponseSamples != 0 ||
          gEndpointAngularStats.tailSamples != expectedTailSamples ||
          gEndpointAngularStats.apiSamples !=
              gEndpointAngularResponseWindowFrames ||
          gEndpointAngularStats.nonFiniteApiSamples != 0)
        setInfrastructureErrorOverFailure(evaluation,
                                          "endpoint_samples");

      if (gEndpointStats.maxPrecontactPositionDrift >
              gEndpointPrecontactPositionDriftMaximum ||
          gEndpointStats.maxPrecontactSpeed >
              gEndpointPrecontactSpeedMaximum ||
          gEndpointAngularStats.maxPrelaunchOrientationDrift >
              gEndpointAngularPrelaunchOrientationDriftMaximum ||
          gEndpointAngularStats.maxPrelaunchAngularSpeed >
              gEndpointAngularPrelaunchSpeedMaximum)
        setGateFailure(evaluation, "endpoint_prelaunch_drift");

      if (gEndpointAngularStats.maxAnchorError >
          gEndpointAngularAnchorMaximum)
        setGateFailure(evaluation, "endpoint_anchor");

      const PxReal signedRotation = getEndpointSignedTargetDelta();
      const PxReal rotationOrthogonalAllowance =
          gEndpointAngularOrthogonalRatioMaximum * PxAbs(signedRotation) +
          gEndpointAngularOrthogonalAbsoluteEpsilon;
      const PxReal signedAngularVelocity =
          getEndpointSignedTargetVelocityDelta();
      const PxReal angularVelocityOrthogonalAllowance =
          gEndpointAngularOrthogonalRatioMaximum *
              PxAbs(signedAngularVelocity) +
          gEndpointAngularOrthogonalAbsoluteEpsilon;
      if (signedRotation < gEndpointAngularPoseResponseMinimum ||
          gEndpointAngularStats.minSignedRotation <= 0.0f ||
          getEndpointDirectionDot() <
              gEndpointAngularDirectionDotMinimum ||
          getEndpointOrthogonalDelta() >
              rotationOrthogonalAllowance ||
          getEndpointPositionOrthogonalRms() >
              rotationOrthogonalAllowance ||
          getEndpointMaxTargetOrthogonalDelta() >
              rotationOrthogonalAllowance ||
          signedAngularVelocity <
              gEndpointAngularVelocityResponseMinimum ||
          gEndpointAngularStats.minSignedAngularVelocity <= 0.0f ||
          getEndpointVelocityDirectionDot() <
              gEndpointAngularDirectionDotMinimum ||
          getEndpointVelocityOrthogonalDelta() >
              angularVelocityOrthogonalAllowance ||
          getEndpointVelocityOrthogonalRms() >
              angularVelocityOrthogonalAllowance ||
          getEndpointMaxVelocityOrthogonalDelta() >
              angularVelocityOrthogonalAllowance ||
          gEndpointAngularStats.maxAxisMisalignment >
              gEndpointAngularAxisMisalignmentMaximum)
        setGateFailure(evaluation, "endpoint_angular_axis_response");

      const PxReal expectedApiSign =
          gEndpointKind == eENDPOINT_FORWARD ? 1.0f : -1.0f;
      const PxReal meanRawAngle = getEndpointMeanRawJointAngleDelta();
      if (expectedApiSign * meanRawAngle <
              gEndpointAngularPoseResponseMinimum ||
          getEndpointMeanSemanticJointAngleDelta() <
              gEndpointAngularPoseResponseMinimum ||
          gEndpointAngularStats.maxJointAnglePoseMismatch >
              gEndpointAngularJointAngleMismatchMaximum)
        setGateFailure(evaluation, "endpoint_joint_angle_semantics");
    } else {
      if (!gEndpointStats.actorOrderValid ||
          !gEndpointStats.frameWitnessValid ||
          !gEndpointStats.fixtureWitnessValid ||
          gEndpointStats.limitEnabled)
        setInfrastructureErrorOverFailure(evaluation, "endpoint_frame");
      if (gEndpointStats.actualLaunchFrame != gImpactLaunchFrame ||
          !gEndpointStats.launchDirection.isFinite() ||
          gEndpointStats.launchDirection.dot(gEndpointStats.expectedAxis) <
              0.99999f)
        setInfrastructureErrorOverFailure(evaluation, "endpoint_launch");
      if (gEndpointStats.initialDynamicActors != 1 ||
          gEndpointStats.initialStaticActors != 0 ||
          gEndpointStats.initialConstraints != 1 ||
          gEndpointStats.finalDynamicActors != 2 ||
          gEndpointStats.finalStaticActors != 0 ||
          gEndpointStats.finalConstraints != 1)
        setInfrastructureErrorOverFailure(evaluation, "endpoint_topology");
      const bool contactObserved =
          gProjectiles.size() == 1 &&
          gProjectiles[0].firstContactFrame != PX_MAX_U32;
      if (gEndpointStats.stateSampleAttempts != gHeadlessOptions.frames ||
          gEndpointStats.stateSamples != gHeadlessOptions.frames ||
          !gEndpointStats.responseBaselineSamples ||
          (contactObserved &&
           (gEndpointStats.responseSampleAttempts !=
                gEndpointResponseWindowFrames ||
            gEndpointStats.responseSamples +
                    gEndpointStats.nonFiniteResponseSamples !=
                gEndpointStats.responseSampleAttempts)))
        setInfrastructureErrorOverFailure(evaluation,
                                          "endpoint_sample_accounting");

      if (gEndpointStats.maxPrecontactPositionDrift >
              gEndpointPrecontactPositionDriftMaximum ||
          gEndpointStats.maxPrecontactSpeed >
              gEndpointPrecontactSpeedMaximum)
        setGateFailure(evaluation, "endpoint_precontact_drift");

      const PxReal signedPositionDelta = getEndpointSignedTargetDelta();
      const PxReal positionOrthogonalAllowance =
          gEndpointOrthogonalRatioMaximum * PxAbs(signedPositionDelta) +
          gEndpointPositionOrthogonalAbsoluteEpsilon;
      const PxReal signedVelocityDelta =
          getEndpointSignedTargetVelocityDelta();
      const PxReal velocityOrthogonalAllowance =
          gEndpointOrthogonalRatioMaximum * PxAbs(signedVelocityDelta) +
          gEndpointVelocityOrthogonalAbsoluteEpsilon;
      if (signedPositionDelta < gEndpointPositionResponseMinimum ||
          getEndpointDirectionDot() < gEndpointDirectionDotMinimum ||
          getEndpointOrthogonalDelta() > positionOrthogonalAllowance ||
          getEndpointPositionOrthogonalRms() >
              positionOrthogonalAllowance ||
          gEndpointStats.maxPositionOrthogonalDelta >
              positionOrthogonalAllowance ||
          signedVelocityDelta < gEndpointVelocityResponseMinimum ||
          getEndpointVelocityDirectionDot() <
              gEndpointDirectionDotMinimum ||
          getEndpointVelocityOrthogonalDelta() >
              velocityOrthogonalAllowance ||
          getEndpointVelocityOrthogonalRms() >
              velocityOrthogonalAllowance ||
          gEndpointStats.maxVelocityOrthogonalDelta >
              velocityOrthogonalAllowance ||
          gEndpointStats.maxTransverseAnchor >
              gEndpointTransverseAnchorMaximum ||
          gGateStats.maxConstrainedAngularError >
              gConstrainedAngularErrorCap)
        setGateFailure(evaluation, "endpoint_axis_response");
    }
  }

  // Kinetic energy alone is not monotonic for gravity-driven pendula. Keep
  // the legacy three-window trend as a passive jitter oracle, but do not use
  // it to reject a directed impact while potential and kinetic energy trade.
  if (!isImpactCase() && !isWideJointStressCase()) {
    const PxReal energyFloor =
        PxMax(1e-4f, gGateStats.peakKineticEnergy * 1e-5f);
    const PxReal energyMargin12 =
        PxMax(energyFloor, PxAbs(evaluation.tailEnergyW1) * 0.05f);
    const PxReal energyMargin23 =
        PxMax(energyFloor, PxAbs(evaluation.tailEnergyW2) * 0.05f);
    if (evaluation.tailEnergyW2 > evaluation.tailEnergyW1 + energyMargin12 &&
        evaluation.tailEnergyW3 > evaluation.tailEnergyW2 + energyMargin23)
      setGateFailure(evaluation, "tail_energy_growth");
    for (PxU32 i = 0; i < gChains.size(); ++i) {
      const ChainRecord &chain = gChains[i];
      const PxReal chainW1 = getTailWindowMean(chain.kineticEnergy, 2);
      const PxReal chainW2 = getTailWindowMean(chain.kineticEnergy, 1);
      const PxReal chainW3 = getTailWindowMean(chain.kineticEnergy, 0);
      const PxReal chainFloor =
          PxMax(1e-4f, chain.peakKineticEnergy * 1e-5f);
      if (chainW2 > chainW1 +
                        PxMax(chainFloor, PxAbs(chainW1) * 0.05f) &&
          chainW3 > chainW2 +
                        PxMax(chainFloor, PxAbs(chainW2) * 0.05f))
        setGateFailure(evaluation, "chain_tail_energy_growth");
    }
  }
  return evaluation;
}

static void printWideJointStressDetails() {
  PxU32 bodyCount = 0;
  PxU32 jointCount = 0;
  PxU32 nonFinite = 0;
  double positionDigest = 0.0;
  double rotationDigest = 0.0;
  for (PxU32 chainIndex = 0; chainIndex < gChains.size(); ++chainIndex) {
    const ChainRecord &chain = gChains[chainIndex];
    jointCount += static_cast<PxU32>(chain.joints.size());
    for (PxU32 bodyIndex = 0; bodyIndex < chain.bodies.size(); ++bodyIndex) {
      const PxRigidDynamic *body = chain.bodies[bodyIndex];
      if (!body) {
        nonFinite++;
        continue;
      }
      const PxTransform pose = body->getGlobalPose();
      if (!pose.p.isFinite() || !pose.q.isFinite()) {
        nonFinite++;
        continue;
      }
      const double weight =
          1.0 + double(chainIndex) * 1000.0 + double(bodyIndex);
      positionDigest += weight *
                        (double(pose.p.x) + 3.0 * double(pose.p.y) +
                         5.0 * double(pose.p.z));
      rotationDigest += weight *
                        (double(pose.q.x) + 3.0 * double(pose.q.y) +
                         5.0 * double(pose.q.z) + 7.0 * double(pose.q.w));
      bodyCount++;
    }
  }
  std::printf(
      "[AVBD_JOINT_STRESS] chains=%u bodies=%u d6Rows=%u nonFinite=%u "
      "positionDigest=%.17g rotationDigest=%.17g\n",
      static_cast<PxU32>(gChains.size()), bodyCount, jointCount, nonFinite,
      positionDigest, rotationDigest);
}

static void printGateDetails() {
  if (isWideJointStressCase())
    printWideJointStressDetails();
  if (isRevoluteMotorOffCenterCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorOffCenter] "
        "case=%s topology=world-dynamic "
        "actorOrderValid=%u driveEnabled=%u "
        "targetVelocity=%.9g forceLimit=%.9g "
        "initialPerpendicularLeverArm=%.9g "
        "initialOffPrincipalResponse=%.9g "
        "inertiaX=%.9g inertiaY=%.9g inertiaZ=%.9g "
        "bodyAngle=%.9g "
        "stateSamples=%u nonFiniteSamples=%u "
        "finalHingeVelocity=%.9g finalHingeVelocityError=%.9g "
        "maximumLateHingeVelocityError=%.9g "
        "maximumLateSwingVelocity=%.9g "
        "finalAnchorPointSpeed=%.9g "
        "maximumLateAnchorPointSpeed=%.9g "
        "maximumLinearSpeed=%.9g maximumLinearReaction=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        gRevoluteMotorOffCenterStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorOffCenterStats.driveEnabledReadback ? 1u : 0u,
        double(gRevoluteMotorOffCenterStats.targetVelocityReadback),
        double(gRevoluteMotorOffCenterStats.forceLimitReadback),
        double(gRevoluteMotorOffCenterStats
                   .initialPerpendicularLeverArm),
        double(gRevoluteMotorOffCenterStats
                   .initialOffPrincipalResponse),
        double(isRevoluteMotorSpatialCase()
                   ? gRevoluteMotorOffPrincipalInertia.x
                   : 1.0f),
        double(isRevoluteMotorSpatialCase()
                   ? gRevoluteMotorOffPrincipalInertia.y
                   : 2.0f),
        double(isRevoluteMotorSpatialCase()
                   ? gRevoluteMotorOffPrincipalInertia.z
                   : 3.0f),
        double(isRevoluteMotorSpatialCase()
                   ? gRevoluteMotorOffPrincipalAngle
                   : 0.0f),
        gRevoluteMotorOffCenterStats.stateSamples,
        gRevoluteMotorOffCenterStats.nonFiniteSamples,
        double(gRevoluteMotorOffCenterStats.finalHingeVelocity),
        double(gRevoluteMotorOffCenterStats.finalHingeVelocityError),
        double(gRevoluteMotorOffCenterStats
                   .maximumLateHingeVelocityError),
        double(gRevoluteMotorOffCenterStats
                   .maximumLateSwingVelocity),
        double(gRevoluteMotorOffCenterStats.finalAnchorPointSpeed),
        double(gRevoluteMotorOffCenterStats
                   .maximumLateAnchorPointSpeed),
        double(gRevoluteMotorOffCenterStats.maximumLinearSpeed),
        double(gRevoluteMotorOffCenterStats.maximumLinearReaction),
        double(gRevoluteMotorOffCenterStats.maximumAnchorError),
        double(gRevoluteMotorOffCenterStats.maximumAxisMisalignment),
        gRevoluteMotorOffCenterStats.initialDynamicActors,
        gRevoluteMotorOffCenterStats.initialStaticActors,
        gRevoluteMotorOffCenterStats.initialConstraints,
        gRevoluteMotorOffCenterStats.finalDynamicActors,
        gRevoluteMotorOffCenterStats.finalStaticActors,
        gRevoluteMotorOffCenterStats.finalConstraints);
  } else if (isRevoluteMotorOffPrincipalCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorOffPrincipal] "
        "case=%s topology=world-dynamic "
        "actorOrderValid=%u driveEnabled=%u "
        "targetVelocity=%.9g forceLimit=%.9g "
        "inertiaX=%.9g inertiaY=%.9g inertiaZ=%.9g "
        "bodyAngle=%.9g initialOffPrincipalResponse=%.9g "
        "stateSamples=%u nonFiniteSamples=%u "
        "finalHingeVelocity=%.9g finalHingeVelocityError=%.9g "
        "maximumLateHingeVelocityError=%.9g "
        "maximumLateSwingVelocity=%.9g "
        "maximumSwingReaction=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        gRevoluteMotorOffPrincipalStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorOffPrincipalStats.driveEnabledReadback ? 1u : 0u,
        double(gRevoluteMotorOffPrincipalStats.targetVelocityReadback),
        double(gRevoluteMotorOffPrincipalStats.forceLimitReadback),
        double(gRevoluteMotorOffPrincipalInertia.x),
        double(gRevoluteMotorOffPrincipalInertia.y),
        double(gRevoluteMotorOffPrincipalInertia.z),
        double(gRevoluteMotorOffPrincipalAngle),
        double(gRevoluteMotorOffPrincipalStats
                   .initialOffPrincipalResponse),
        gRevoluteMotorOffPrincipalStats.stateSamples,
        gRevoluteMotorOffPrincipalStats.nonFiniteSamples,
        double(gRevoluteMotorOffPrincipalStats.finalHingeVelocity),
        double(gRevoluteMotorOffPrincipalStats
                   .finalHingeVelocityError),
        double(gRevoluteMotorOffPrincipalStats
                   .maximumLateHingeVelocityError),
        double(gRevoluteMotorOffPrincipalStats
                   .maximumLateSwingVelocity),
        double(gRevoluteMotorOffPrincipalStats.maximumSwingReaction),
        double(gRevoluteMotorOffPrincipalStats.maximumAnchorError),
        double(gRevoluteMotorOffPrincipalStats.maximumAxisMisalignment),
        gRevoluteMotorOffPrincipalStats.initialDynamicActors,
        gRevoluteMotorOffPrincipalStats.initialStaticActors,
        gRevoluteMotorOffPrincipalStats.initialConstraints,
        gRevoluteMotorOffPrincipalStats.finalDynamicActors,
        gRevoluteMotorOffPrincipalStats.finalStaticActors,
        gRevoluteMotorOffPrincipalStats.finalConstraints);
  } else if (isRevoluteMotorKinematicCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorKinematic] "
        "case=%s topology=kinematic-dynamic "
        "actorOrderValid=%u driveEnabled=%u kinematicFlag=%u "
        "targetVelocity=%.9g forceLimit=%.9g "
        "endpointVelocity=%.9g expectedDynamicVelocity=%.9g "
        "targetUpdates=%u targetUpdateFailures=%u "
        "stateSamples=%u nonFiniteSamples=%u "
        "finalKinematicVelocity=%.9g finalDynamicVelocity=%.9g "
        "finalRelativeVelocity=%.9g finalRelativeError=%.9g "
        "maximumLateRelativeError=%.9g "
        "maximumLateKinematicVelocityError=%.9g "
        "maximumLateDynamicVelocityError=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        gRevoluteMotorKinematicStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorKinematicStats.driveEnabledReadback ? 1u : 0u,
        gRevoluteMotorKinematicStats.kinematicFlagReadback ? 1u : 0u,
        double(gRevoluteMotorKinematicStats.targetVelocityReadback),
        double(gRevoluteMotorKinematicStats.forceLimitReadback),
        double(gRevoluteMotorKinematicEndpointVelocity),
        double(gRevoluteMotorKinematicExpectedDynamicVelocity),
        gRevoluteMotorKinematicStats.targetUpdates,
        gRevoluteMotorKinematicStats.targetUpdateFailures,
        gRevoluteMotorKinematicStats.stateSamples,
        gRevoluteMotorKinematicStats.nonFiniteSamples,
        double(gRevoluteMotorKinematicStats.finalKinematicVelocity),
        double(gRevoluteMotorKinematicStats.finalDynamicVelocity),
        double(gRevoluteMotorKinematicStats.finalRelativeVelocity),
        double(gRevoluteMotorKinematicStats.finalRelativeError),
        double(gRevoluteMotorKinematicStats.maximumLateRelativeError),
        double(gRevoluteMotorKinematicStats
                   .maximumLateKinematicVelocityError),
        double(gRevoluteMotorKinematicStats
                   .maximumLateDynamicVelocityError),
        double(gRevoluteMotorKinematicStats.maximumAnchorError),
        double(gRevoluteMotorKinematicStats.maximumAxisMisalignment),
        gRevoluteMotorKinematicStats.initialDynamicActors,
        gRevoluteMotorKinematicStats.initialStaticActors,
        gRevoluteMotorKinematicStats.initialConstraints,
        gRevoluteMotorKinematicStats.finalDynamicActors,
        gRevoluteMotorKinematicStats.finalStaticActors,
        gRevoluteMotorKinematicStats.finalConstraints);
  } else if (isRevoluteMotorContactCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorContact] "
        "case=%s topology=dynamic-dynamic-ground "
        "actorOrderValid=%u driveEnabled=%u "
        "targetVelocity=%.9g forceLimit=%.9g "
        "radius=%.9g halfHeight=%.9g centerHeight=%.9g "
        "stateSamples=%u nonFiniteSamples=%u "
        "contactEvents=%u contactPointCount=%u "
        "finalVelocityA=%.9g finalVelocityB=%.9g "
        "finalRelativeVelocity=%.9g finalRelativeError=%.9g "
        "maximumLateRelativeError=%.9g "
        "lateDriveReactionSamples=%u meanLateDriveReaction=%.9g "
        "maximumLateDriveReaction=%.9g "
        "totalNormalImpulse=%.9g totalTangentialImpulse=%.9g "
        "maximumTangentialImpulse=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "maximumCenterHeightError=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        gRevoluteMotorContactStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorContactStats.driveEnabledReadback ? 1u : 0u,
        double(gRevoluteMotorContactStats.targetVelocityReadback),
        double(gRevoluteMotorContactStats.forceLimitReadback),
        double(gRevoluteMotorContactRadius),
        double(gRevoluteMotorContactHalfHeight),
        double(gRevoluteMotorContactCenterHeight),
        gRevoluteMotorContactStats.stateSamples,
        gRevoluteMotorContactStats.nonFiniteSamples,
        gRevoluteMotorContactStats.contactEvents,
        gRevoluteMotorContactStats.contactPointCount,
        double(gRevoluteMotorContactStats.finalVelocityA),
        double(gRevoluteMotorContactStats.finalVelocityB),
        double(gRevoluteMotorContactStats.finalRelativeVelocity),
        double(gRevoluteMotorContactStats.finalRelativeError),
        double(gRevoluteMotorContactStats.maximumLateRelativeError),
        gRevoluteMotorContactStats.lateDriveReactionSamples,
        gRevoluteMotorContactStats.lateDriveReactionSamples
            ? double(gRevoluteMotorContactStats.lateDriveReactionSum /
                     PxReal(gRevoluteMotorContactStats
                                .lateDriveReactionSamples))
            : 0.0,
        double(gRevoluteMotorContactStats.maximumLateDriveReaction),
        double(gRevoluteMotorContactStats.totalNormalImpulse),
        double(gRevoluteMotorContactStats.totalTangentialImpulse),
        double(gRevoluteMotorContactStats.maximumTangentialImpulse),
        double(gRevoluteMotorContactStats.maximumAnchorError),
        double(gRevoluteMotorContactStats.maximumAxisMisalignment),
        double(gRevoluteMotorContactStats.maximumCenterHeightError),
        gRevoluteMotorContactStats.initialDynamicActors,
        gRevoluteMotorContactStats.initialStaticActors,
        gRevoluteMotorContactStats.initialConstraints,
        gRevoluteMotorContactStats.finalDynamicActors,
        gRevoluteMotorContactStats.finalStaticActors,
        gRevoluteMotorContactStats.finalConstraints);
  } else if (isRevoluteMotorRatioCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorRatio] "
        "case=%s topology=%s actorOrderValid=%u "
        "driveEnabled=%u freeSpinDisabled=%u "
        "targetVelocity=%.9g forceLimit=%.9g driveGearRatio=%.9g "
        "inertiaA=%.9g inertiaB=%.9g stateSamples=%u "
        "nonFiniteSamples=%u finalVelocityA=%.9g "
        "finalVelocityB=%.9g finalWeightedVelocity=%.9g "
        "finalWeightedVelocityError=%.9g "
        "maximumLateWeightedVelocityError=%.9g "
        "initialOffPrincipalResponseA=%.9g "
        "initialOffPrincipalResponseB=%.9g "
        "maximumLateRelativeSwingVelocity=%.9g "
        "maximumInitialGeneralizedMomentumDrift=%.9g "
        "maximumGeneralizedMomentumDrift=%.9g "
        "initialPerpendicularLeverArmA=%.9g "
        "initialPerpendicularLeverArmB=%.9g "
        "finalRelativeAnchorPointSpeed=%.9g "
        "maximumLateRelativeAnchorPointSpeed=%.9g "
        "maximumTotalLinearMomentum=%.9g "
        "maximumInitialTotalAngularMomentum=%.9g "
        "maximumLinearSpeed=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        isRevoluteMotorDynamicSpatialCase()
            ? "dynamic-dynamic-spatial"
            : (isRevoluteMotorDynamicOffPrincipalCase()
                   ? "dynamic-dynamic-centered"
                   : (isRevoluteMotorDynamicOffCenterCase()
                          ? "dynamic-dynamic-off-center"
                          : "dynamic-dynamic")),
        gRevoluteMotorRatioStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorRatioStats.driveEnabledReadback ? 1u : 0u,
        gRevoluteMotorRatioStats.freeSpinDisabledReadback ? 1u : 0u,
        double(gRevoluteMotorRatioStats.targetVelocityReadback),
        double(gRevoluteMotorRatioStats.forceLimitReadback),
        double(gRevoluteMotorRatioStats.driveGearRatioReadback),
        double(gRevoluteMotorRatioInertiaA),
        double(gRevoluteMotorRatioInertiaB),
        gRevoluteMotorRatioStats.stateSamples,
        gRevoluteMotorRatioStats.nonFiniteSamples,
        double(gRevoluteMotorRatioStats.finalVelocityA),
        double(gRevoluteMotorRatioStats.finalVelocityB),
        double(gRevoluteMotorRatioStats.finalWeightedVelocity),
        double(gRevoluteMotorRatioStats.finalWeightedVelocityError),
        double(gRevoluteMotorRatioStats.maximumLateWeightedVelocityError),
        double(gRevoluteMotorRatioStats.initialOffPrincipalResponseA),
        double(gRevoluteMotorRatioStats.initialOffPrincipalResponseB),
        double(
            gRevoluteMotorRatioStats.maximumLateRelativeSwingVelocity),
        double(gRevoluteMotorRatioStats
                   .maximumInitialGeneralizedMomentumDrift),
        double(gRevoluteMotorRatioStats.maximumGeneralizedMomentumDrift),
        double(gRevoluteMotorRatioStats
                   .initialPerpendicularLeverArmA),
        double(gRevoluteMotorRatioStats
                   .initialPerpendicularLeverArmB),
        double(gRevoluteMotorRatioStats
                   .finalRelativeAnchorPointSpeed),
        double(gRevoluteMotorRatioStats
                   .maximumLateRelativeAnchorPointSpeed),
        double(gRevoluteMotorRatioStats
                   .maximumTotalLinearMomentum),
        double(gRevoluteMotorRatioStats
                   .maximumInitialTotalAngularMomentum),
        double(gRevoluteMotorRatioStats.maximumLinearSpeed),
        double(gRevoluteMotorRatioStats.maximumAnchorError),
        double(gRevoluteMotorRatioStats.maximumAxisMisalignment),
        gRevoluteMotorRatioStats.initialDynamicActors,
        gRevoluteMotorRatioStats.initialStaticActors,
        gRevoluteMotorRatioStats.initialConstraints,
        gRevoluteMotorRatioStats.finalDynamicActors,
        gRevoluteMotorRatioStats.finalStaticActors,
        gRevoluteMotorRatioStats.finalConstraints);
  } else if (isRevoluteMotorFreeSpinCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorFreeSpin] "
        "case=%s topology=%s actorOrderValid=%u "
        "driveEnabled=%u freeSpinEnabled=%u limitDisabled=%u "
        "targetVelocity=%.9g forceLimit=%.9g boostFrame=%u "
        "boostVelocity=%.9g boostVelocityReadback=%.9g "
        "boostEvents=%u stateSamples=%u nonFiniteSamples=%u "
        "preBoostFinalVelocity=%.9g "
        "maximumLatePreBoostError=%.9g finalVelocity=%.9g "
        "minimumPostBoostVelocity=%.9g "
        "maximumPostBoostVelocityDrop=%.9g "
        "maximumAngularMomentumDrift=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        isRevoluteMotorDynamicFreeSpinCase()
            ? "dynamic-dynamic"
            : "world-dynamic",
        gRevoluteMotorFreeSpinStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorFreeSpinStats.driveEnabledReadback ? 1u : 0u,
        gRevoluteMotorFreeSpinStats.freeSpinEnabledReadback ? 1u : 0u,
        gRevoluteMotorFreeSpinStats.limitDisabledReadback ? 1u : 0u,
        double(gRevoluteMotorFreeSpinStats.targetVelocityReadback),
        double(gRevoluteMotorFreeSpinStats.forceLimitReadback),
        gRevoluteMotorFreeSpinBoostFrame,
        double(gRevoluteMotorFreeSpinBoostVelocity),
        double(gRevoluteMotorFreeSpinStats.boostVelocityReadback),
        gRevoluteMotorFreeSpinStats.boostEvents,
        gRevoluteMotorFreeSpinStats.stateSamples,
        gRevoluteMotorFreeSpinStats.nonFiniteSamples,
        double(gRevoluteMotorFreeSpinStats.preBoostFinalVelocity),
        double(gRevoluteMotorFreeSpinStats.maximumLatePreBoostError),
        double(gRevoluteMotorFreeSpinStats.finalVelocity),
        double(gRevoluteMotorFreeSpinStats.minimumPostBoostVelocity),
        double(gRevoluteMotorFreeSpinStats.maximumPostBoostVelocityDrop),
        double(gRevoluteMotorFreeSpinStats.maximumAngularMomentumDrift),
        double(gRevoluteMotorFreeSpinStats.maximumAnchorError),
        double(gRevoluteMotorFreeSpinStats.maximumAxisMisalignment),
        gRevoluteMotorFreeSpinStats.initialDynamicActors,
        gRevoluteMotorFreeSpinStats.initialStaticActors,
        gRevoluteMotorFreeSpinStats.initialConstraints,
        gRevoluteMotorFreeSpinStats.finalDynamicActors,
        gRevoluteMotorFreeSpinStats.finalStaticActors,
        gRevoluteMotorFreeSpinStats.finalConstraints);
  } else if (isRevoluteMotorLimitCase()) {
    const PxReal upperTravel =
        gRevoluteMotorLimitStats.maximumAngle -
        gRevoluteMotorLimitStats.initialAngle;
    const PxReal range =
        gRevoluteMotorLimitStats.maximumAngle -
        gRevoluteMotorLimitStats.minimumAngle;
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorLimit] "
        "case=%s topology=%s actorOrderValid=%u "
        "driveEnabled=%u limitEnabled=%u "
        "targetVelocity=%.9g finalTargetVelocity=%.9g "
        "forceLimit=%.9g reverseFrame=%u reverseEvents=%u "
        "lowerLimit=%.9g upperLimit=%.9g stateSamples=%u "
        "nonFiniteSamples=%u initialAngle=%.9g finalAngle=%.9g "
        "minimumAngle=%.9g maximumAngle=%.9g "
        "upperTravel=%.9g range=%.9g "
        "maximumUpperViolation=%.9g maximumLowerViolation=%.9g "
        "maximumLateOutwardVelocity=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        isRevoluteMotorDynamicLimitCase()
            ? "dynamic-dynamic"
            : "world-dynamic",
        gRevoluteMotorLimitStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorLimitStats.driveEnabledReadback ? 1u : 0u,
        gRevoluteMotorLimitStats.limitEnabledReadback ? 1u : 0u,
        double(gRevoluteMotorLimitStats.targetVelocityReadback),
        double(gRevoluteMotorLimitStats.finalTargetVelocityReadback),
        double(gRevoluteMotorLimitStats.forceLimitReadback),
        gRevoluteMotorLimitReverseFrame,
        gRevoluteMotorLimitStats.reverseEvents,
        double(gRevoluteMotorLimitStats.lowerLimitReadback),
        double(gRevoluteMotorLimitStats.upperLimitReadback),
        gRevoluteMotorLimitStats.stateSamples,
        gRevoluteMotorLimitStats.nonFiniteSamples,
        double(gRevoluteMotorLimitStats.initialAngle),
        double(gRevoluteMotorLimitStats.finalAngle),
        double(gRevoluteMotorLimitStats.minimumAngle),
        double(gRevoluteMotorLimitStats.maximumAngle),
        double(upperTravel),
        double(range),
        double(gRevoluteMotorLimitStats.maximumUpperViolation),
        double(gRevoluteMotorLimitStats.maximumLowerViolation),
        double(gRevoluteMotorLimitStats.maximumLateOutwardVelocity),
        double(gRevoluteMotorLimitStats.maximumAnchorError),
        double(gRevoluteMotorLimitStats.maximumAxisMisalignment),
        gRevoluteMotorLimitStats.initialDynamicActors,
        gRevoluteMotorLimitStats.initialStaticActors,
        gRevoluteMotorLimitStats.initialConstraints,
        gRevoluteMotorLimitStats.finalDynamicActors,
        gRevoluteMotorLimitStats.finalStaticActors,
        gRevoluteMotorLimitStats.finalConstraints);
  } else if (isRevoluteMotorCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotor] "
        "case=%s topology=dynamic-dynamic actorOrderValid=%u "
        "driveEnabled=%u targetVelocity=%.9g forceLimit=%.9g "
        "inertiaA=%.9g inertiaB=%.9g stateSamples=%u "
        "nonFiniteSamples=%u finalRelativeVelocity=%.9g "
        "finalRelativeError=%.9g maximumLateRelativeError=%.9g "
        "maximumAngularMomentumDrift=%.9g "
        "maximumAnchorError=%.9g maximumAxisMisalignment=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        gRevoluteMotorStats.actorOrderValid ? 1u : 0u,
        gRevoluteMotorStats.driveEnabledReadback ? 1u : 0u,
        double(gRevoluteMotorStats.targetVelocityReadback),
        double(gRevoluteMotorStats.forceLimitReadback),
        double(gRevoluteMotorInertiaA),
        double(gRevoluteMotorInertiaB),
        gRevoluteMotorStats.stateSamples,
        gRevoluteMotorStats.nonFiniteSamples,
        double(gRevoluteMotorStats.finalRelativeVelocity),
        double(gRevoluteMotorStats.finalRelativeError),
        double(gRevoluteMotorStats.maximumLateRelativeError),
        double(gRevoluteMotorStats.maximumAngularMomentumDrift),
        double(gRevoluteMotorStats.maximumAnchorError),
        double(gRevoluteMotorStats.maximumAxisMisalignment),
        gRevoluteMotorStats.initialDynamicActors,
        gRevoluteMotorStats.initialStaticActors,
        gRevoluteMotorStats.initialConstraints,
        gRevoluteMotorStats.finalDynamicActors,
        gRevoluteMotorStats.finalStaticActors,
        gRevoluteMotorStats.finalConstraints);
  } else if (isNativeBreakReactionCase()) {
    const PxVec3 meanReaction = getNativeMeanReactionVector();
    const PxVec3 expectedReaction = getNativeExpectedReactionVector();
    const PxReal expectedMagnitude = expectedReaction.magnitude();
    const PxReal reactionRatio =
        expectedMagnitude > 0.0f
            ? getNativeMeanReactionMagnitude() / expectedMagnitude
            : 0.0f;
    std::printf(
        "[PROBE] [SnippetJointNativeBreakReaction] "
        "case=%s joint=%s loadKind=%s actorOrderValid=%u "
        "stateSamples=%u forceReads=%u reactionSamples=%u "
        "nonFiniteSamples=%u "
        "expectedReaction=(%.9g,%.9g,%.9g) "
        "meanReaction=(%.9g,%.9g,%.9g) "
        "meanReactionMagnitude=%.9g reactionRatio=%.9g "
        "reactionDirectionDot=%.9g reactionOrthogonalRatio=%.9g "
        "breakForceReadback=%.9g breakTorqueReadback=%.9g "
        "brokenPollCount=%u breakCallbackCount=%u "
        "breakCallbackIdentityMatches=%u firstBrokenFrame=%u "
        "maximumPositionError=%.9g maximumRotationError=%.9g "
        "maximumLinearSpeed=%.9g maximumAngularSpeed=%.9g "
        "steadyMaximumPositionError=%.9g "
        "steadyMaximumRotationError=%.9g "
        "steadyMaximumLinearSpeed=%.9g "
        "steadyMaximumAngularSpeed=%.9g "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u\n",
        getHeadlessCaseName(gHeadlessCase),
        getJointKindName(gImpactJointKind),
        isNativeAngularReactionCase() ? "angular" : "linear",
        gNativeBreakReactionStats.actorOrderValid ? 1u : 0u,
        gNativeBreakReactionStats.stateSamples,
        gNativeBreakReactionStats.forceReads,
        gNativeBreakReactionStats.reactionSamples,
        gNativeBreakReactionStats.nonFiniteSamples,
        double(expectedReaction.x), double(expectedReaction.y),
        double(expectedReaction.z), double(meanReaction.x),
        double(meanReaction.y), double(meanReaction.z),
        double(getNativeMeanReactionMagnitude()), double(reactionRatio),
        double(getNativeReactionDirectionDot()),
        double(getNativeReactionOrthogonalRatio()),
        double(gNativeBreakReactionStats.breakForceReadback),
        double(gNativeBreakReactionStats.breakTorqueReadback),
        gNativeBreakReactionStats.brokenPollCount,
        gGateStats.breakCallbackCount,
        gGateStats.breakCallbackIdentityMatches,
        gNativeBreakReactionStats.firstBrokenFrame,
        double(gNativeBreakReactionStats.maximumPositionError),
        double(gNativeBreakReactionStats.maximumRotationError),
        double(gNativeBreakReactionStats.maximumLinearSpeed),
        double(gNativeBreakReactionStats.maximumAngularSpeed),
        double(gNativeBreakReactionStats.steadyMaximumPositionError),
        double(gNativeBreakReactionStats.steadyMaximumRotationError),
        double(gNativeBreakReactionStats.steadyMaximumLinearSpeed),
        double(gNativeBreakReactionStats.steadyMaximumAngularSpeed),
        gNativeBreakReactionStats.initialDynamicActors,
        gNativeBreakReactionStats.initialStaticActors,
        gNativeBreakReactionStats.initialConstraints,
        gNativeBreakReactionStats.finalDynamicActors,
        gNativeBreakReactionStats.finalStaticActors,
        gNativeBreakReactionStats.finalConstraints);
  } else if (isSphericalConeCase()) {
    const PxReal radiusCorrection =
        gSphericalConeStats.initialEllipseRadius -
        gSphericalConeStats.finalEllipseRadius;
    std::printf(
        "[PROBE] [SnippetJointSphericalCone] case=%s topology=%s "
        "limitEnabled=%u actorOrderValid=%u limitY=%.9g limitZ=%.9g "
        "initialEllipseRadius=%.9g finalEllipseRadius=%.9g "
        "minimumEllipseRadius=%.9g maximumEllipseRadius=%.9g "
        "minimumLateEllipseRadius=%.9g maximumLateEllipseRadius=%.9g "
        "maximumInsideDeviation=%.9g radiusCorrection=%.9g "
        "stateSamples=%u nonFiniteSamples=%u "
        "initialDynamicActors=%u initialStaticActors=%u "
        "initialConstraints=%u finalDynamicActors=%u "
        "finalStaticActors=%u finalConstraints=%u "
        "maximumTotalAngularMomentum=%.9g "
        "maximumAnchorSeparation=%.9g\n",
        getHeadlessCaseName(gHeadlessCase),
        getSphericalConeTopologyName(),
        gSphericalConeStats.limitEnabledReadback ? 1u : 0u,
        gSphericalConeStats.actorOrderValid ? 1u : 0u,
        double(gSphericalConeStats.limitYReadback),
        double(gSphericalConeStats.limitZReadback),
        double(gSphericalConeStats.initialEllipseRadius),
        double(gSphericalConeStats.finalEllipseRadius),
        double(gSphericalConeStats.minimumEllipseRadius),
        double(gSphericalConeStats.maximumEllipseRadius),
        double(gSphericalConeStats.minimumLateEllipseRadius),
        double(gSphericalConeStats.maximumLateEllipseRadius),
        double(gSphericalConeStats.maximumInsideDeviation),
        double(radiusCorrection), gSphericalConeStats.stateSamples,
        gSphericalConeStats.nonFiniteSamples,
        gSphericalConeStats.initialDynamicActors,
        gSphericalConeStats.initialStaticActors,
        gSphericalConeStats.initialConstraints,
        gSphericalConeStats.finalDynamicActors,
        gSphericalConeStats.finalStaticActors,
        gSphericalConeStats.finalConstraints,
        double(gSphericalConeStats.maximumTotalAngularMomentum),
        double(gSphericalConeStats.maximumAnchorSeparation));
  } else if (isEndpointProbe()) {
    if (isRevoluteEndpointProbe()) {
      const PxVec3 rotationVector = getEndpointMeanTargetDelta();
      const PxVec3 angularVelocity =
          getEndpointMeanTargetVelocityDelta();
      std::printf(
          "[PROBE] [SnippetJointEndpointAngularFixture] endpoint=%s "
          "actor0=%s actor1=%s joint=revolute excitation=direct-angular-velocity "
          "actorOrderValid=%u frameWitnessValid=%u fixtureWitnessValid=%u "
          "nonIdentityWitnessValid=%u limitEnabled=%u driveEnabled=%u "
          "shapeCount=%u shapeRadius=%.9g bodyRotationDot=%.9g "
          "expectedAxis=(%.9g,%.9g,%.9g) "
          "perpendicularAxis=(%.9g,%.9g,%.9g) "
          "actor0Axis=(%.9g,%.9g,%.9g) actor1Axis=(%.9g,%.9g,%.9g) "
          "dynamicLocalAxis=(%.9g,%.9g,%.9g) "
          "dynamicLocalAxisDot=%.9g bodyWorldAxisDot=%.9g "
          "worldFramePositionError=%.9g dynamicLocalPositionError=%.9g "
          "worldFrameRotationDot=%.9g dynamicLocalRotationDot=%.9g "
          "shapeLocalPositionError=%.9g shapeLocalRotationDot=%.9g "
          "gravity=(%.9g,%.9g,%.9g) projectileCount=%u\n",
          getEndpointName(gEndpointKind), getEndpointActor0Name(),
          getEndpointActor1Name(),
          gEndpointStats.actorOrderValid ? 1u : 0u,
          gEndpointStats.frameWitnessValid ? 1u : 0u,
          gEndpointStats.fixtureWitnessValid ? 1u : 0u,
          gEndpointAngularStats.nonIdentityWitnessValid ? 1u : 0u,
          gEndpointStats.limitEnabled ? 1u : 0u,
          gEndpointAngularStats.driveEnabled ? 1u : 0u,
          gEndpointStats.shapeCount, double(gEndpointStats.shapeRadius),
          double(gEndpointStats.bodyRotationDot),
          double(gEndpointStats.expectedAxis.x),
          double(gEndpointStats.expectedAxis.y),
          double(gEndpointStats.expectedAxis.z),
          double(gEndpointAngularStats.perpendicularAxis.x),
          double(gEndpointAngularStats.perpendicularAxis.y),
          double(gEndpointAngularStats.perpendicularAxis.z),
          double(gEndpointStats.actor0Axis.x),
          double(gEndpointStats.actor0Axis.y),
          double(gEndpointStats.actor0Axis.z),
          double(gEndpointStats.actor1Axis.x),
          double(gEndpointStats.actor1Axis.y),
          double(gEndpointStats.actor1Axis.z),
          double(gEndpointStats.dynamicLocalAxis.x),
          double(gEndpointStats.dynamicLocalAxis.y),
          double(gEndpointStats.dynamicLocalAxis.z),
          double(gEndpointStats.dynamicLocalAxisDot),
          double(gEndpointAngularStats.bodyWorldAxisDot),
          double(gEndpointStats.worldFramePositionError),
          double(gEndpointStats.dynamicLocalPositionError),
          double(gEndpointStats.worldFrameRotationDot),
          double(gEndpointStats.dynamicLocalRotationDot),
          double(gEndpointStats.shapeLocalPositionError),
          double(gEndpointStats.shapeLocalRotationDot),
          double(gEndpointStats.gravity.x), double(gEndpointStats.gravity.y),
          double(gEndpointStats.gravity.z),
          static_cast<PxU32>(gProjectiles.size()));
      std::printf(
          "[PROBE] [SnippetJointEndpointAngularResponse] "
          "calibrationRequired=1 launchAttempts=%u launchSuccesses=%u "
          "launchFrame=%u launchWakeValid=%u "
          "requestedAngularVelocity=(%.9g,%.9g,%.9g) "
          "actualLaunchAngularVelocity=(%.9g,%.9g,%.9g) "
          "launchVelocityError=%.9g launchAxisDot=%.9g "
          "stateSampleAttempts=%u stateSamples=%u baselineSamples=%u "
          "responseSampleAttempts=%u responseSamples=%u tailSamples=%u "
          "apiSamples=%u nonFiniteStateSamples=%u "
          "nonFiniteResponseSamples=%u nonFiniteApiSamples=%u "
          "baselineOrientation=(%.9g,%.9g,%.9g,%.9g) "
          "baselineAngularVelocity=(%.9g,%.9g,%.9g) "
          "maxPrelaunchPositionDrift=%.9g maxPrelaunchLinearSpeed=%.9g "
          "maxPrelaunchOrientationDrift=%.9g "
          "maxPrelaunchAngularSpeed=%.9g "
          "rotationVector=(%.9g,%.9g,%.9g) signedRotation=%.9g "
          "minSignedRotation=%.9g "
          "rotationDirectionDot=%.9g rotationOrthogonal=%.9g "
          "rotationOrthogonalRms=%.9g rotationOrthogonalRatio=%.9g "
          "maxRotationOrthogonal=%.9g "
          "angularVelocity=(%.9g,%.9g,%.9g) signedAngularVelocity=%.9g "
          "minSignedAngularVelocity=%.9g "
          "angularVelocityDirectionDot=%.9g "
          "angularVelocityOrthogonal=%.9g "
          "angularVelocityOrthogonalRms=%.9g "
          "angularVelocityOrthogonalRatio=%.9g "
          "maxAngularVelocityOrthogonal=%.9g "
          "meanRawJointAngleDelta=%.9g meanSemanticJointAngleDelta=%.9g "
          "lastRawJointAngleDelta=%.9g lastSemanticJointAngleDelta=%.9g "
          "maxJointAnglePoseMismatch=%.9g meanApiVelocityMagnitude=%.9g "
          "maxApiVelocityMagnitudeMismatch=%.9g maxAnchorError=%.9g "
          "maxAxisMisalignment=%.9g\n",
          gEndpointAngularStats.launchAttempts,
          gEndpointAngularStats.launchSuccesses,
          gEndpointStats.actualLaunchFrame,
          gEndpointAngularStats.launchWakeValid ? 1u : 0u,
          double(gEndpointAngularStats.requestedAngularVelocity.x),
          double(gEndpointAngularStats.requestedAngularVelocity.y),
          double(gEndpointAngularStats.requestedAngularVelocity.z),
          double(gEndpointAngularStats.actualLaunchAngularVelocity.x),
          double(gEndpointAngularStats.actualLaunchAngularVelocity.y),
          double(gEndpointAngularStats.actualLaunchAngularVelocity.z),
          double(gEndpointAngularStats.launchVelocityError),
          double(gEndpointStats.launchDirection.dot(
              gEndpointStats.expectedAxis)),
          gEndpointStats.stateSampleAttempts, gEndpointStats.stateSamples,
          gEndpointStats.responseBaselineSamples,
          gEndpointStats.responseSampleAttempts,
          gEndpointStats.responseSamples,
          gEndpointAngularStats.tailSamples,
          gEndpointAngularStats.apiSamples,
          gEndpointStats.nonFiniteStateSamples,
          gEndpointStats.nonFiniteResponseSamples,
          gEndpointAngularStats.nonFiniteApiSamples,
          double(gEndpointAngularStats.responseBaselineOrientation.x),
          double(gEndpointAngularStats.responseBaselineOrientation.y),
          double(gEndpointAngularStats.responseBaselineOrientation.z),
          double(gEndpointAngularStats.responseBaselineOrientation.w),
          double(gEndpointAngularStats.responseBaselineAngularVelocity.x),
          double(gEndpointAngularStats.responseBaselineAngularVelocity.y),
          double(gEndpointAngularStats.responseBaselineAngularVelocity.z),
          double(gEndpointStats.maxPrecontactPositionDrift),
          double(gEndpointStats.maxPrecontactSpeed),
          double(gEndpointAngularStats.maxPrelaunchOrientationDrift),
          double(gEndpointAngularStats.maxPrelaunchAngularSpeed),
          double(rotationVector.x), double(rotationVector.y),
          double(rotationVector.z),
          double(getEndpointSignedTargetDelta()),
          double(gEndpointAngularStats.minSignedRotation),
          double(getEndpointDirectionDot()),
          double(getEndpointOrthogonalDelta()),
          double(getEndpointPositionOrthogonalRms()),
          double(getEndpointOrthogonalRatio()),
          double(getEndpointMaxTargetOrthogonalDelta()),
          double(angularVelocity.x), double(angularVelocity.y),
          double(angularVelocity.z),
          double(getEndpointSignedTargetVelocityDelta()),
          double(gEndpointAngularStats.minSignedAngularVelocity),
          double(getEndpointVelocityDirectionDot()),
          double(getEndpointVelocityOrthogonalDelta()),
          double(getEndpointVelocityOrthogonalRms()),
          double(getEndpointVelocityOrthogonalRatio()),
          double(getEndpointMaxVelocityOrthogonalDelta()),
          double(getEndpointMeanRawJointAngleDelta()),
          double(getEndpointMeanSemanticJointAngleDelta()),
          double(gEndpointAngularStats.lastRawJointAngleDelta),
          double(gEndpointAngularStats.lastSemanticJointAngleDelta),
          double(gEndpointAngularStats.maxJointAnglePoseMismatch),
          double(getEndpointMeanApiVelocityMagnitude()),
          double(gEndpointAngularStats.maxApiVelocityMagnitudeMismatch),
          double(gEndpointAngularStats.maxAnchorError),
          double(gEndpointAngularStats.maxAxisMisalignment));
    } else {
    const PxVec3 positionDelta = getEndpointMeanTargetDelta();
    const PxVec3 velocityDelta = getEndpointMeanTargetVelocityDelta();
    std::printf(
        "[PROBE] [SnippetJointEndpointFixture] endpoint=%s actor0=%s "
        "actor1=%s actorOrderValid=%u frameWitnessValid=%u "
        "fixtureWitnessValid=%u limitEnabled=%u shapeCount=%u "
        "shapeRadius=%.9g worldFramePositionError=%.9g "
        "dynamicLocalPositionError=%.9g worldFrameRotationDot=%.9g "
        "dynamicLocalRotationDot=%.9g shapeLocalPositionError=%.9g "
        "shapeLocalRotationDot=%.9g "
        "expectedAxis=(%.9g,%.9g,%.9g) actor0Axis=(%.9g,%.9g,%.9g) "
        "actor1Axis=(%.9g,%.9g,%.9g) bodyRotationDot=%.9g "
        "expectedAxisDot=%.9g dynamicLocalAxis=(%.9g,%.9g,%.9g) "
        "dynamicLocalAxisDot=%.9g gravity=(%.9g,%.9g,%.9g) "
        "launchFrame=%u launchAxisDot=%.9g\n",
        getEndpointName(gEndpointKind), getEndpointActor0Name(),
        getEndpointActor1Name(), gEndpointStats.actorOrderValid ? 1u : 0u,
        gEndpointStats.frameWitnessValid ? 1u : 0u,
        gEndpointStats.fixtureWitnessValid ? 1u : 0u,
        gEndpointStats.limitEnabled ? 1u : 0u,
        gEndpointStats.shapeCount, double(gEndpointStats.shapeRadius),
        double(gEndpointStats.worldFramePositionError),
        double(gEndpointStats.dynamicLocalPositionError),
        double(gEndpointStats.worldFrameRotationDot),
        double(gEndpointStats.dynamicLocalRotationDot),
        double(gEndpointStats.shapeLocalPositionError),
        double(gEndpointStats.shapeLocalRotationDot),
        double(gEndpointStats.expectedAxis.x),
        double(gEndpointStats.expectedAxis.y),
        double(gEndpointStats.expectedAxis.z),
        double(gEndpointStats.actor0Axis.x),
        double(gEndpointStats.actor0Axis.y),
        double(gEndpointStats.actor0Axis.z),
        double(gEndpointStats.actor1Axis.x),
        double(gEndpointStats.actor1Axis.y),
        double(gEndpointStats.actor1Axis.z),
        double(gEndpointStats.bodyRotationDot),
        double(gEndpointStats.expectedAxisDot),
        double(gEndpointStats.dynamicLocalAxis.x),
        double(gEndpointStats.dynamicLocalAxis.y),
        double(gEndpointStats.dynamicLocalAxis.z),
        double(gEndpointStats.dynamicLocalAxisDot),
        double(gEndpointStats.gravity.x), double(gEndpointStats.gravity.y),
        double(gEndpointStats.gravity.z), gEndpointStats.actualLaunchFrame,
        double(gEndpointStats.launchDirection.dot(
            gEndpointStats.expectedAxis)));
    std::printf(
        "[PROBE] [SnippetJointEndpointResponse] stateSampleAttempts=%u "
        "stateSamples=%u baselineSamples=%u responseSampleAttempts=%u "
        "responseSamples=%u nonFiniteResponseSamples=%u "
        "baselinePosition=(%.9g,%.9g,%.9g) "
        "baselineVelocity=(%.9g,%.9g,%.9g) "
        "maxPrecontactPositionDrift=%.9g maxPrecontactSpeed=%.9g "
        "positionDelta=(%.9g,%.9g,%.9g) signedPositionDelta=%.9g "
        "positionDirectionDot=%.9g positionOrthogonalDelta=%.9g "
        "positionOrthogonalRms=%.9g maxPositionOrthogonalDelta=%.9g "
        "positionOrthogonalRatio=%.9g "
        "velocityDelta=(%.9g,%.9g,%.9g) signedVelocityDelta=%.9g "
        "velocityDirectionDot=%.9g velocityOrthogonalDelta=%.9g "
        "velocityOrthogonalRms=%.9g maxVelocityOrthogonalDelta=%.9g "
        "velocityOrthogonalRatio=%.9g maxTransverseAnchor=%.9g\n",
        gEndpointStats.stateSampleAttempts, gEndpointStats.stateSamples,
        gEndpointStats.responseBaselineSamples,
        gEndpointStats.responseSampleAttempts,
        gEndpointStats.responseSamples,
        gEndpointStats.nonFiniteResponseSamples,
        double(gEndpointStats.responseBaselinePosition.x),
        double(gEndpointStats.responseBaselinePosition.y),
        double(gEndpointStats.responseBaselinePosition.z),
        double(gEndpointStats.responseBaselineVelocity.x),
        double(gEndpointStats.responseBaselineVelocity.y),
        double(gEndpointStats.responseBaselineVelocity.z),
        double(gEndpointStats.maxPrecontactPositionDrift),
        double(gEndpointStats.maxPrecontactSpeed),
        double(positionDelta.x), double(positionDelta.y),
        double(positionDelta.z),
        double(getEndpointSignedTargetDelta()),
        double(getEndpointDirectionDot()),
        double(getEndpointOrthogonalDelta()),
        double(getEndpointPositionOrthogonalRms()),
        double(gEndpointStats.maxPositionOrthogonalDelta),
        double(getEndpointOrthogonalRatio()),
        double(velocityDelta.x), double(velocityDelta.y),
        double(velocityDelta.z),
        double(getEndpointSignedTargetVelocityDelta()),
        double(getEndpointVelocityDirectionDot()),
        double(getEndpointVelocityOrthogonalDelta()),
        double(getEndpointVelocityOrthogonalRms()),
        double(gEndpointStats.maxVelocityOrthogonalDelta),
        double(getEndpointVelocityOrthogonalRatio()),
        double(gEndpointStats.maxTransverseAnchor));
    }
  }

  if (isForceReactionCase()) {
    if (isForcePairDisabledCase()) {
      PxConstraint *constraint =
          gForceStaticJoint ? gForceStaticJoint->getConstraint() : NULL;
      const bool disabledReadback =
          constraint &&
          constraint->getFlags().isSet(
              PxConstraintFlag::eDISABLE_CONSTRAINT);
      std::printf(
          "[PROBE] [SnippetJointConstraintFlag] "
          "requestedDisabled=1 readbackDisabled=%u "
          "separationErrorMin=%.9g relativeSpeedMin=%.9g "
          "reactionMax=%.9g centerOfMassErrorMax=%.9g "
          "totalMomentumMax=%.9g\n",
          disabledReadback ? 1u : 0u,
          double(gDisabledPairSeparationErrorMinimum),
          double(gDisabledPairRelativeSpeedMinimum),
          double(gDisabledPairReactionMaximum),
          double(gDisabledPairCenterOfMassErrorMaximum),
          double(gDisabledPairTotalMomentumMaximum));
    }
    const PxVec3 meanForce = getMeanForceStaticLinearForce();
    const PxVec3 meanTorque = getMeanForceStaticAngularForce();
    const PxReal meanSampleMagnitude =
        gForceStaticStats.steadySamples
            ? gForceStaticStats.linearMagnitudeSum /
                  PxReal(gForceStaticStats.steadySamples)
            : 0.0f;
    std::printf(
        "[PROBE] [SnippetJointForceStatic] forceReads=%u stateSamples=%u "
        "steadyBeginFrame=%u expectedSteadySamples=%u "
        "steadySampleAttempts=%u steadySamples=%u "
        "meanForce=(%.9g,%.9g,%.9g) meanForceMagnitude=%.9g "
        "meanSampleMagnitude=%.9g meanSampleForceRatio=%.9g "
        "actualMass=%.9g "
        "gravity=(%.9g,%.9g,%.9g) expectedWeight=%.9g forceRatio=%.9g "
        "directionDot=%.9g meanVectorOrthogonalRatio=%.9g "
        "orthogonalRmsRatio=%.9g "
        "fixture=%s anchorOffset=(%.9g,%.9g,%.9g) "
        "actorOrder=%s appliedForceActor0=(%.9g,%.9g,%.9g) "
        "appliedForceActor1=(%.9g,%.9g,%.9g) "
        "expectedLinearForce=(%.9g,%.9g,%.9g) "
        "expectedTorque=(%.9g,%.9g,%.9g) expectedTorqueMagnitude=%.9g "
        "torqueRatio=%.9g meanSampleTorqueRatio=%.9g "
        "torqueDirectionDot=%.9g torqueOrthogonalRmsRatio=%.9g "
        "actor0FramePositionError=%.9g actor1FramePositionError=%.9g "
        "meanTorque=(%.9g,%.9g,%.9g) meanTorqueMagnitude=%.9g "
        "torqueRms=%.9g maxTorqueMagnitude=%.9g maxPositionError=%.9g "
        "maxPositionErrorFrame=%u maxRotationError=%.9g "
        "maxLinearSpeed=%.9g maxLinearSpeedFrame=%u "
        "steadyMaxLinearSpeed=%.9g steadyMaxLinearSpeedFrame=%u "
        "finalPositionError=%.9g finalLinearSpeed=%.9g "
        "maxAngularSpeed=%.9g nonFiniteForceSamples=%u "
        "nonFiniteSteadyForceSamples=%u "
        "nonFiniteStateSamples=%u pairInitialSeparation=%.9g "
        "pairActorOrderValid=%u pairMaxSeparationError=%.9g "
        "pairFinalSeparation=%.9g pairMaxRelativeSpeed=%.9g "
        "pairFinalRelativeSpeed=%.9g pairMaxCenterOfMassError=%.9g "
        "pairFinalCenterOfMassError=%.9g "
        "pairMaxTotalMomentum=%.9g pairFinalTotalMomentum=%.9g "
        "pairFinalTotalMomentumVector=(%.9g,%.9g,%.9g) "
        "pairActor0FinalPosition=(%.9g,%.9g,%.9g) "
        "pairActor1FinalPosition=(%.9g,%.9g,%.9g) "
        "pairActor0FinalVelocity=(%.9g,%.9g,%.9g) "
        "pairActor1FinalVelocity=(%.9g,%.9g,%.9g) "
        "pairTotalMomentumMax=%.9g pairSeparationErrorMax=%.9g "
        "pairRelativeSpeedMax=%.9g pairCenterOfMassErrorMax=%.9g\n",
        gForceStaticStats.forceReads, gForceStaticStats.stateSamples,
        gForceStaticStats.steadyBeginFrame,
        gForceStaticStats.expectedSteadySamples,
        gForceStaticStats.steadySampleAttempts,
        gForceStaticStats.steadySamples, double(meanForce.x),
        double(meanForce.y), double(meanForce.z),
        double(getSafeMagnitude(meanForce)), double(meanSampleMagnitude),
        double(getForceStaticMeanSampleRatio()),
        double(gForceStaticStats.actualMass),
        double(gForceStaticStats.gravity.x),
        double(gForceStaticStats.gravity.y),
        double(gForceStaticStats.gravity.z),
        double(getForceStaticExpectedWeight()),
        double(getForceStaticRatio()), double(getForceStaticDirectionDot()),
        double(getForceStaticMeanVectorOrthogonalRatio()),
        double(getForceStaticOrthogonalRmsRatio()),
        getForceFixtureName(),
        double(gForceStaticStats.anchorOffset.x),
        double(gForceStaticStats.anchorOffset.y),
        double(gForceStaticStats.anchorOffset.z),
        getForceActorOrderName(),
        double(gForceStaticStats.appliedForceActor0.x),
        double(gForceStaticStats.appliedForceActor0.y),
        double(gForceStaticStats.appliedForceActor0.z),
        double(gForceStaticStats.appliedForceActor1.x),
        double(gForceStaticStats.appliedForceActor1.y),
        double(gForceStaticStats.appliedForceActor1.z),
        double(gForceStaticStats.expectedLinearForce.x),
        double(gForceStaticStats.expectedLinearForce.y),
        double(gForceStaticStats.expectedLinearForce.z),
        double(gForceStaticStats.expectedTorque.x),
        double(gForceStaticStats.expectedTorque.y),
        double(gForceStaticStats.expectedTorque.z),
        double(getForceStaticExpectedTorqueMagnitude()),
        double(getForceStaticTorqueRatio()),
        double(getForceStaticMeanSampleTorqueRatio()),
        double(getForceStaticTorqueDirectionDot()),
        double(getForceStaticTorqueOrthogonalRmsRatio()),
        double(gForceStaticStats.actor0FramePositionError),
        double(gForceStaticStats.actor1FramePositionError),
        double(meanTorque.x),
        double(meanTorque.y), double(meanTorque.z),
        double(getSafeMagnitude(meanTorque)),
        double(getForceStaticTorqueRms()),
        double(gForceStaticStats.maxAngularMagnitude),
        double(gForceStaticStats.maxPositionError),
        gForceStaticStats.maxPositionErrorFrame,
        double(gForceStaticStats.maxRotationError),
        double(gForceStaticStats.maxLinearSpeed),
        gForceStaticStats.maxLinearSpeedFrame,
        double(gForceStaticStats.steadyMaxLinearSpeed),
        gForceStaticStats.steadyMaxLinearSpeedFrame,
        double(gForceStaticStats.finalPositionError),
        double(gForceStaticStats.finalLinearSpeed),
        double(gForceStaticStats.maxAngularSpeed),
        gForceStaticStats.nonFiniteForceSamples,
        gForceStaticStats.nonFiniteSteadyForceSamples,
        gForceStaticStats.nonFiniteStateSamples,
        double(gForceStaticStats.pairInitialSeparation),
        gForceStaticStats.pairActorOrderValid ? 1u : 0u,
        double(gForceStaticStats.pairMaxSeparationError),
        double(gForceStaticStats.pairFinalSeparation),
        double(gForceStaticStats.pairMaxRelativeSpeed),
        double(gForceStaticStats.pairFinalRelativeSpeed),
        double(gForceStaticStats.pairMaxCenterOfMassError),
        double(gForceStaticStats.pairFinalCenterOfMassError),
        double(gForceStaticStats.pairMaxTotalMomentum),
        double(gForceStaticStats.pairFinalTotalMomentum),
        double(gForceStaticStats.pairFinalTotalMomentumVector.x),
        double(gForceStaticStats.pairFinalTotalMomentumVector.y),
        double(gForceStaticStats.pairFinalTotalMomentumVector.z),
        double(gForceStaticStats.pairActor0FinalPosition.x),
        double(gForceStaticStats.pairActor0FinalPosition.y),
        double(gForceStaticStats.pairActor0FinalPosition.z),
        double(gForceStaticStats.pairActor1FinalPosition.x),
        double(gForceStaticStats.pairActor1FinalPosition.y),
        double(gForceStaticStats.pairActor1FinalPosition.z),
        double(gForceStaticStats.pairActor0FinalVelocity.x),
        double(gForceStaticStats.pairActor0FinalVelocity.y),
        double(gForceStaticStats.pairActor0FinalVelocity.z),
        double(gForceStaticStats.pairActor1FinalVelocity.x),
        double(gForceStaticStats.pairActor1FinalVelocity.y),
        double(gForceStaticStats.pairActor1FinalVelocity.z),
        double(gForcePairTotalMomentumMaximum),
        double(gForcePairSeparationErrorMaximum),
        double(gForcePairRelativeSpeedMaximum),
        double(gForcePairCenterOfMassErrorMaximum));
  }

  for (PxU32 i = 0; i < gChains.size(); ++i) {
    const ChainRecord &chain = gChains[i];
    const PxReal anchorRms =
        chain.anchorSamples
            ? PxSqrt(chain.anchorErrorSquaredSum / PxReal(chain.anchorSamples))
            : 0.0f;
    const PxReal lateAwakeRatio =
        chain.lateBodySamples
            ? PxReal(chain.lateAwakeSamples) / PxReal(chain.lateBodySamples)
            : 0.0f;
    const PxReal lateAngularSpeedMean =
        chain.lateAngularSpeedSamples
            ? chain.lateAngularSpeedSum / PxReal(chain.lateAngularSpeedSamples)
            : 0.0f;
    std::printf(
        "[SnippetJointChain] index=%u kind=%s maxAnchor=%.9g anchorRms=%.9g "
        "maxAngularLimitViolation=%.9g maxLinearLimitViolation=%.9g "
        "maxConstrainedAngularError=%.9g peakEnergy=%.9g tailEnergyW1=%.9g "
        "tailEnergyW2=%.9g tailEnergyW3=%.9g lateAwakeRatio=%.9g "
        "lateAngularSpeedMean=%.9g lateAngularSpeedMax=%.9g axisGate=%s\n",
        i, chain.name, double(chain.maxAnchorError), double(anchorRms),
        double(chain.maxAngularLimitViolation),
        double(chain.maxLinearLimitViolation),
        double(chain.maxConstrainedAngularError),
        double(chain.peakKineticEnergy),
        double(getTailWindowMean(chain.kineticEnergy, 2)),
        double(getTailWindowMean(chain.kineticEnergy, 1)),
        double(getTailWindowMean(chain.kineticEnergy, 0)),
        double(lateAwakeRatio), double(lateAngularSpeedMean),
        double(chain.lateAngularSpeedMax),
        (chain.kind == eJOINT_SPHERICAL || chain.kind == eJOINT_D6) ? "NA"
                                                                   : "GATED");
  }

  for (PxU32 i = 0; i < gProjectiles.size(); ++i) {
    const ProjectileRecord &projectile = gProjectiles[i];
    const TargetResponseKind targetResponseKind =
        getTargetResponseKind(projectile.targetChain);
    std::printf(
        "[SnippetJointProjectile] index=%u targetChain=%u targetContacts=%u "
        "contactPoints=%u firstChain=%u firstBody=%u crossChainContacts=%u "
        "wrongBodyContacts=%u firstContactFrame=%u firstWrongContactFrame=%u "
        "expectedMomentum=%.9g totalContactImpulse=%.9g "
        "peakContactImpulse=%.9g maxVelocityDelta=%.9g "
        "maxImpactAxisVelocityDelta=%.9g responseFraction=%.9g "
        "targetMass=%.9g preContactTargetVz=%.9g "
        "preContactChainPz=%.9g targetDeltaVz=%.9g chainDeltaPz=%.9g "
        "maxOppositeTargetDeltaVz=%.9g maxOppositeChainDeltaPz=%.9g "
        "targetBodyResponseFraction=%.9g chainResponseFraction=%.9g "
        "targetResponseFraction=%.9g targetResponseSamples=%u "
        "targetResponseKind=%s targetResponseGate=%s\n",
        i, projectile.targetChain, projectile.contactCount,
        projectile.contactPointCount, projectile.firstChainContact,
        projectile.firstChainBody, projectile.crossChainContacts,
        projectile.wrongBodyContacts, projectile.firstContactFrame,
        projectile.firstWrongContactFrame, double(projectile.expectedMomentum),
        double(projectile.totalContactImpulse),
        double(projectile.peakContactImpulse),
        double(projectile.maxVelocityDelta),
        double(projectile.maxImpactAxisVelocityDelta),
        double(projectile.maxImpactAxisVelocityDelta / gImpactSpeed),
        double(projectile.targetMass),
        double(projectile.preContactTargetVelocityZ),
        double(projectile.preContactChainMomentumZ),
        double(projectile.targetDeltaVelocityZ),
        double(projectile.chainDeltaMomentumZ),
        double(projectile.maxOppositeTargetDeltaVelocityZ),
        double(projectile.maxOppositeChainDeltaMomentumZ),
        double(getTargetBodyResponseFraction(projectile)),
        double(getTargetChainResponseFraction(projectile)),
        double(getTargetResponseFraction(projectile)),
        projectile.targetResponseSamples,
        getTargetResponseKindName(targetResponseKind),
        getTargetResponseGateName(targetResponseKind));
  }
}

void cleanupPhysics(bool interactive) {
  if (!interactive && !isForceReactionCase() &&
      !isSphericalConeCase() && !isNativeBreakReactionCase()) {
    const PxReal avgW4Early =
        gRevoluteStats.cntEarly
            ? gRevoluteStats.sumW4Early / gRevoluteStats.cntEarly
            : 0.0f;
    const PxReal avgW5Early =
        gRevoluteStats.cntEarly
            ? gRevoluteStats.sumW5Early / gRevoluteStats.cntEarly
            : 0.0f;
    const PxReal avgW4Late = gRevoluteStats.cntLate ? gRevoluteStats.sumW4Late /
                                                          gRevoluteStats.cntLate
                                                    : 0.0f;
    const PxReal avgW5Late = gRevoluteStats.cntLate ? gRevoluteStats.sumW5Late /
                                                          gRevoluteStats.cntLate
                                                    : 0.0f;
    const PxReal avgW4PerpEarly =
        gRevoluteStats.cntEarly
            ? gRevoluteStats.sumW4PerpEarly / gRevoluteStats.cntEarly
            : 0.0f;
    const PxReal avgW5PerpEarly =
        gRevoluteStats.cntEarly
            ? gRevoluteStats.sumW5PerpEarly / gRevoluteStats.cntEarly
            : 0.0f;
    const PxReal avgW4PerpLate =
        gRevoluteStats.cntLate
            ? gRevoluteStats.sumW4PerpLate / gRevoluteStats.cntLate
            : 0.0f;
    const PxReal avgW5PerpLate =
        gRevoluteStats.cntLate
            ? gRevoluteStats.sumW5PerpLate / gRevoluteStats.cntLate
            : 0.0f;
    const PxReal awake4LateRatio =
        gRevoluteStats.cntLate
            ? PxReal(gRevoluteStats.awake4Late) / PxReal(gRevoluteStats.cntLate)
            : 0.0f;
    const PxReal awake5LateRatio =
        gRevoluteStats.cntLate
            ? PxReal(gRevoluteStats.awake5Late) / PxReal(gRevoluteStats.cntLate)
            : 0.0f;
    const PxReal growth4 =
        (avgW4Early > 1e-6f) ? (avgW4Late / avgW4Early) : 0.0f;
    const PxReal growth5 =
        (avgW5Early > 1e-6f) ? (avgW5Late / avgW5Early) : 0.0f;
    const PxReal growth4Perp =
        (avgW4PerpEarly > 1e-6f) ? (avgW4PerpLate / avgW4PerpEarly) : 0.0f;
    const PxReal growth5Perp =
        (avgW5PerpEarly > 1e-6f) ? (avgW5PerpLate / avgW5PerpEarly) : 0.0f;
    const bool jitterReproduced = isLegacyRevoluteJitterReproduced();

    printf(
        "[RevoluteDiag] tail displacementXZ max=%.5f, avgW early=(%.5f,%.5f), "
        "avgW late=(%.5f,%.5f), growth=(%.3f,%.3f), "
        "avgWPerp early=(%.5f,%.5f), avgWPerp late=(%.5f,%.5f), "
        "growthPerp=(%.3f,%.3f), maxWPerpLate=(%.5f,%.5f), "
        "awakeLate=(%.3f,%.3f), "
        "max|angle|=(%.3f,%.3f), axisMisalignDeg=(%.3f,%.3f), flips=(%u,%u), "
        "jitter_reproduced=%s\n",
        gRevoluteStats.maxTailLateral, avgW4Early, avgW5Early, avgW4Late,
        avgW5Late, growth4, growth5, avgW4PerpEarly, avgW5PerpEarly,
        avgW4PerpLate, avgW5PerpLate, growth4Perp, growth5Perp,
        gRevoluteStats.maxW4PerpLate, gRevoluteStats.maxW5PerpLate,
        awake4LateRatio, awake5LateRatio, gRevoluteStats.maxAbsAngle3,
        gRevoluteStats.maxAbsAngle4, gRevoluteStats.maxAxisMisalign3Deg,
        gRevoluteStats.maxAxisMisalign4Deg, gRevoluteStats.flip3,
        gRevoluteStats.flip4, jitterReproduced ? "true" : "false");

    const PxReal avgPrismaticTailEarly =
        gPrismaticStats.cntEarly
            ? gPrismaticStats.sumTailEarly / gPrismaticStats.cntEarly
            : 0.0f;
    const PxReal avgPrismaticTailLate =
        gPrismaticStats.cntLate
            ? gPrismaticStats.sumTailLate / gPrismaticStats.cntLate
            : 0.0f;
    const PxReal avgPrismaticTailAngVelEarly =
        gPrismaticStats.cntEarly
            ? gPrismaticStats.sumTailAngVelEarly / gPrismaticStats.cntEarly
            : 0.0f;
    const PxReal avgPrismaticTailAngVelLate =
        gPrismaticStats.cntLate
            ? gPrismaticStats.sumTailAngVelLate / gPrismaticStats.cntLate
            : 0.0f;
    const PxReal prismaticTailGrowth =
        (avgPrismaticTailEarly > 1e-6f)
            ? (avgPrismaticTailLate / avgPrismaticTailEarly)
            : 0.0f;
    const PxReal prismaticAngVelGrowth =
        (avgPrismaticTailAngVelEarly > 1e-6f)
            ? (avgPrismaticTailAngVelLate / avgPrismaticTailAngVelEarly)
            : 0.0f;

    printf(
        "[PrismaticDiag] tail transverse max=%.5f, avgTail early=%.5f, "
        "avgTail late=%.5f, tailGrowth=%.3f, avgTailAngVel early=%.5f, "
        "avgTailAngVel late=%.5f, angVelGrowth=%.3f, maxAxisMisalignDeg=%.3f\n",
        gPrismaticStats.maxTailTransverse, avgPrismaticTailEarly,
        avgPrismaticTailLate, prismaticTailGrowth, avgPrismaticTailAngVelEarly,
        avgPrismaticTailAngVelLate, prismaticAngVelGrowth,
        gPrismaticStats.maxJointAxisMisalignDeg);

    printf("[FixedDiag] maxLinForce=%.5f, maxAngForce=%.5f, "
           "firstBrokenFrame=%u, brokenCount=%u\n",
           gFixedStats.maxLinearForce, gFixedStats.maxAngularForce,
           gFixedStats.firstBrokenFrame, gFixedStats.brokenCount);
  }

  if (!interactive && isRevoluteMotorCase() && gScene) {
    PX_RELEASE(gRevoluteMotorJoint);
    if (gRevoluteMotorBodyA) {
      if (gRevoluteMotorBodyA->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorBodyA);
      PX_RELEASE(gRevoluteMotorBodyA);
    }
    if (gRevoluteMotorBodyB) {
      if (gRevoluteMotorBodyB->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorBodyB);
      PX_RELEASE(gRevoluteMotorBodyB);
    }
    gRevoluteMotorStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorLimitCase() && gScene) {
    PX_RELEASE(gRevoluteMotorLimitJoint);
    if (gRevoluteMotorLimitBodyA) {
      if (gRevoluteMotorLimitBodyA->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorLimitBodyA);
      PX_RELEASE(gRevoluteMotorLimitBodyA);
    }
    if (gRevoluteMotorLimitBody) {
      if (gRevoluteMotorLimitBody->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorLimitBody);
      PX_RELEASE(gRevoluteMotorLimitBody);
    }
    gRevoluteMotorLimitStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorLimitStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorLimitStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorFreeSpinCase() && gScene) {
    PX_RELEASE(gRevoluteMotorFreeSpinJoint);
    if (gRevoluteMotorFreeSpinBodyA) {
      if (gRevoluteMotorFreeSpinBodyA->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorFreeSpinBodyA);
      PX_RELEASE(gRevoluteMotorFreeSpinBodyA);
    }
    if (gRevoluteMotorFreeSpinBody) {
      if (gRevoluteMotorFreeSpinBody->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorFreeSpinBody);
      PX_RELEASE(gRevoluteMotorFreeSpinBody);
    }
    gRevoluteMotorFreeSpinStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorFreeSpinStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorFreeSpinStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorRatioCase() && gScene) {
    PX_RELEASE(gRevoluteMotorRatioJoint);
    if (gRevoluteMotorRatioBodyA) {
      if (gRevoluteMotorRatioBodyA->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorRatioBodyA);
      PX_RELEASE(gRevoluteMotorRatioBodyA);
    }
    if (gRevoluteMotorRatioBodyB) {
      if (gRevoluteMotorRatioBodyB->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorRatioBodyB);
      PX_RELEASE(gRevoluteMotorRatioBodyB);
    }
    gRevoluteMotorRatioStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorRatioStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorRatioStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorContactCase() && gScene) {
    PX_RELEASE(gRevoluteMotorContactJoint);
    if (gRevoluteMotorContactBodyA) {
      if (gRevoluteMotorContactBodyA->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorContactBodyA);
      PX_RELEASE(gRevoluteMotorContactBodyA);
    }
    if (gRevoluteMotorContactBodyB) {
      if (gRevoluteMotorContactBodyB->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorContactBodyB);
      PX_RELEASE(gRevoluteMotorContactBodyB);
    }
    if (gRevoluteMotorContactGround) {
      if (gRevoluteMotorContactGround->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorContactGround);
      PX_RELEASE(gRevoluteMotorContactGround);
    }
    gRevoluteMotorContactStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorContactStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorContactStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorKinematicCase() && gScene) {
    PX_RELEASE(gRevoluteMotorKinematicJoint);
    if (gRevoluteMotorKinematicBody) {
      if (gRevoluteMotorKinematicBody->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorKinematicBody);
      PX_RELEASE(gRevoluteMotorKinematicBody);
    }
    if (gRevoluteMotorKinematicDynamicBody) {
      if (gRevoluteMotorKinematicDynamicBody->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorKinematicDynamicBody);
      PX_RELEASE(gRevoluteMotorKinematicDynamicBody);
    }
    gRevoluteMotorKinematicStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorKinematicStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorKinematicStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorOffPrincipalCase() && gScene) {
    PX_RELEASE(gRevoluteMotorOffPrincipalJoint);
    if (gRevoluteMotorOffPrincipalBody) {
      if (gRevoluteMotorOffPrincipalBody->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorOffPrincipalBody);
      PX_RELEASE(gRevoluteMotorOffPrincipalBody);
    }
    gRevoluteMotorOffPrincipalStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorOffPrincipalStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorOffPrincipalStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isRevoluteMotorOffCenterCase() && gScene) {
    PX_RELEASE(gRevoluteMotorOffCenterJoint);
    if (gRevoluteMotorOffCenterBody) {
      if (gRevoluteMotorOffCenterBody->getScene() == gScene)
        gScene->removeActor(*gRevoluteMotorOffCenterBody);
      PX_RELEASE(gRevoluteMotorOffCenterBody);
    }
    gRevoluteMotorOffCenterStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gRevoluteMotorOffCenterStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gRevoluteMotorOffCenterStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isForceReactionCase() && gScene) {
    PX_RELEASE(gForceStaticJoint);
    if (gForceStaticBody) {
      if (gForceStaticBody->getScene() == gScene)
        gScene->removeActor(*gForceStaticBody);
      PX_RELEASE(gForceStaticBody);
    }
    if (gForcePairBody1) {
      if (gForcePairBody1->getScene() == gScene)
        gScene->removeActor(*gForcePairBody1);
      PX_RELEASE(gForcePairBody1);
    }
    gForceStaticStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gForceStaticStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gForceStaticStats.cleanupConstraints = gScene->getNbConstraints();
  }

  if (!interactive && isNativeBreakReactionCase() && gScene) {
    PX_RELEASE(gNativeBreakReactionJoint);
    if (gNativeBreakReactionBody) {
      if (gNativeBreakReactionBody->getScene() == gScene)
        gScene->removeActor(*gNativeBreakReactionBody);
      PX_RELEASE(gNativeBreakReactionBody);
    }
    gNativeBreakReactionStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gNativeBreakReactionStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gNativeBreakReactionStats.cleanupConstraints =
        gScene->getNbConstraints();
  }

  if (!interactive && isSphericalConeCase() && gScene) {
    PX_RELEASE(gSphericalConeJoint);
    if (gSphericalConeActorA) {
      if (gSphericalConeActorA->getScene() == gScene)
        gScene->removeActor(*gSphericalConeActorA);
      PX_RELEASE(gSphericalConeActorA);
      gSphericalConeDynamicA = NULL;
    }
    if (gSphericalConeActorB) {
      if (gSphericalConeActorB->getScene() == gScene)
        gScene->removeActor(*gSphericalConeActorB);
      PX_RELEASE(gSphericalConeActorB);
    }
    gSphericalConeStats.cleanupDynamicActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    gSphericalConeStats.cleanupStaticActors =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
    gSphericalConeStats.cleanupConstraints = gScene->getNbConstraints();
  }

  if (!interactive && isEndpointProbe()) {
    PX_RELEASE(gEndpointJoint);
    PX_RELEASE(gEndpointRevoluteJoint);
    for (PxU32 i = 0; i < gProjectiles.size(); ++i) {
      PxRigidDynamic *&projectile = gProjectiles[i].actor;
      if (projectile) {
        if (gScene && projectile->getScene() == gScene)
          gScene->removeActor(*projectile);
        PX_RELEASE(projectile);
      }
    }
    if (gEndpointTarget) {
      if (gScene && gEndpointTarget->getScene() == gScene)
        gScene->removeActor(*gEndpointTarget);
      PX_RELEASE(gEndpointTarget);
    }
    if (gScene) {
      gEndpointStats.cleanupDynamicActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
      gEndpointStats.cleanupStaticActors =
          gScene->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
      gEndpointStats.cleanupConstraints = gScene->getNbConstraints();
    }
  }

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
  PX_RELEASE(gFoundation);

  if (!interactive && isForceReactionCase()) {
    gForceStaticStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics && !gFoundation &&
        !gPvd && !gForceStaticBody && !gForcePairBody1 &&
        !gForceStaticJoint &&
        gForceStaticStats.cleanupDynamicActors == 0 &&
        gForceStaticStats.cleanupStaticActors == 0 &&
        gForceStaticStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorCase()) {
    gRevoluteMotorStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorBodyA &&
        !gRevoluteMotorBodyB && !gRevoluteMotorJoint &&
        gRevoluteMotorStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorStats.cleanupStaticActors == 0 &&
        gRevoluteMotorStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorLimitCase()) {
    gRevoluteMotorLimitStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorLimitBodyA &&
        !gRevoluteMotorLimitBody &&
        !gRevoluteMotorLimitJoint &&
        gRevoluteMotorLimitStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorLimitStats.cleanupStaticActors == 0 &&
        gRevoluteMotorLimitStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorFreeSpinCase()) {
    gRevoluteMotorFreeSpinStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorFreeSpinBodyA &&
        !gRevoluteMotorFreeSpinBody &&
        !gRevoluteMotorFreeSpinJoint &&
        gRevoluteMotorFreeSpinStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorFreeSpinStats.cleanupStaticActors == 0 &&
        gRevoluteMotorFreeSpinStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorRatioCase()) {
    gRevoluteMotorRatioStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorRatioBodyA &&
        !gRevoluteMotorRatioBodyB && !gRevoluteMotorRatioJoint &&
        gRevoluteMotorRatioStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorRatioStats.cleanupStaticActors == 0 &&
        gRevoluteMotorRatioStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorContactCase()) {
    gRevoluteMotorContactStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorContactBodyA &&
        !gRevoluteMotorContactBodyB &&
        !gRevoluteMotorContactGround &&
        !gRevoluteMotorContactJoint &&
        gRevoluteMotorContactStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorContactStats.cleanupStaticActors == 0 &&
        gRevoluteMotorContactStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorKinematicCase()) {
    gRevoluteMotorKinematicStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorKinematicBody &&
        !gRevoluteMotorKinematicDynamicBody &&
        !gRevoluteMotorKinematicJoint &&
        gRevoluteMotorKinematicStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorKinematicStats.cleanupStaticActors == 0 &&
        gRevoluteMotorKinematicStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorOffPrincipalCase()) {
    gRevoluteMotorOffPrincipalStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd &&
        !gRevoluteMotorOffPrincipalBody &&
        !gRevoluteMotorOffPrincipalJoint &&
        gRevoluteMotorOffPrincipalStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorOffPrincipalStats.cleanupStaticActors == 0 &&
        gRevoluteMotorOffPrincipalStats.cleanupConstraints == 0;
  }
  if (!interactive && isRevoluteMotorOffCenterCase()) {
    gRevoluteMotorOffCenterStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gRevoluteMotorOffCenterBody &&
        !gRevoluteMotorOffCenterJoint &&
        gRevoluteMotorOffCenterStats.cleanupDynamicActors == 0 &&
        gRevoluteMotorOffCenterStats.cleanupStaticActors == 0 &&
        gRevoluteMotorOffCenterStats.cleanupConstraints == 0;
  }
  if (!interactive && isNativeBreakReactionCase()) {
    gNativeBreakReactionStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gNativeBreakReactionBody &&
        !gNativeBreakReactionJoint &&
        gNativeBreakReactionStats.cleanupDynamicActors == 0 &&
        gNativeBreakReactionStats.cleanupStaticActors == 0 &&
        gNativeBreakReactionStats.cleanupConstraints == 0;
  }
  if (!interactive && isSphericalConeCase()) {
    gSphericalConeStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics &&
        !gFoundation && !gPvd && !gSphericalConeActorA &&
        !gSphericalConeDynamicA && !gSphericalConeActorB &&
        !gSphericalConeJoint &&
        gSphericalConeStats.cleanupDynamicActors == 0 &&
        gSphericalConeStats.cleanupStaticActors == 0 &&
        gSphericalConeStats.cleanupConstraints == 0;
  }
  if (!interactive && isEndpointProbe()) {
    bool projectileCleanupComplete = true;
    for (PxU32 i = 0; i < gProjectiles.size(); ++i)
      projectileCleanupComplete =
          projectileCleanupComplete && gProjectiles[i].actor == NULL;
    gEndpointStats.cleanupComplete =
        !gScene && !gMaterial && !gDispatcher && !gPhysics && !gFoundation &&
        !gPvd && !gEndpointTarget && !gEndpointJoint &&
        !gEndpointRevoluteJoint &&
        projectileCleanupComplete &&
        gEndpointStats.cleanupDynamicActors == 0 &&
        gEndpointStats.cleanupStaticActors == 0 &&
        gEndpointStats.cleanupConstraints == 0;
  }

  if (interactive)
    printf("SnippetJoint done.\n");
}

void keyPress(unsigned char key, const PxTransform &camera) {
  switch (toupper(key)) {
  case ' ':
    createDynamic(camera, PxSphereGeometry(3.0f),
                  camera.rotate(PxVec3(0, 0, -1)) * 200);
    break;
  }
}

static void printGateResult(const GateEvaluation &evaluation,
                            PxU32 physicsErrors, PxU32 physicsWarnings) {
  const PxReal anchorRms =
      gGateStats.anchorSamples
          ? PxSqrt(gGateStats.anchorErrorSquaredSum /
                   PxReal(gGateStats.anchorSamples))
          : 0.0f;
  const PxReal maxAngularLimitViolation =
      PxMax(gGateStats.maxRevoluteLimitViolation,
            gGateStats.maxSphericalLimitViolation);
  const PxReal tailToPeak =
      gGateStats.peakKineticEnergy > 1e-9f
          ? evaluation.tailEnergyW3 / gGateStats.peakKineticEnergy
          : 0.0f;
  PxU32 lateAwakeSamples = 0;
  PxU32 lateBodySamples = 0;
  PxReal d6LateAngularSpeedMean = 0.0f;
  PxReal d6LateAngularSpeedMax = 0.0f;
  for (PxU32 i = 0; i < gChains.size(); ++i) {
    lateAwakeSamples += gChains[i].lateAwakeSamples;
    lateBodySamples += gChains[i].lateBodySamples;
    if (gChains[i].kind == eJOINT_D6) {
      if (gChains[i].lateAngularSpeedSamples)
        d6LateAngularSpeedMean =
            gChains[i].lateAngularSpeedSum /
            PxReal(gChains[i].lateAngularSpeedSamples);
      d6LateAngularSpeedMax = gChains[i].lateAngularSpeedMax;
    }
  }
  const PxReal lateAwakeRatio =
      lateBodySamples ? PxReal(lateAwakeSamples) / PxReal(lateBodySamples)
                      : 0.0f;
  const PxVec3 meanForce = getMeanForceStaticLinearForce();
  const PxVec3 meanTorque = getMeanForceStaticAngularForce();
  const PxReal meanForceMagnitude = getSafeMagnitude(meanForce);
  const PxReal meanSampleForceMagnitude =
      gForceStaticStats.steadySamples
          ? gForceStaticStats.linearMagnitudeSum /
                PxReal(gForceStaticStats.steadySamples)
          : 0.0f;
  const PxReal meanTorqueMagnitude = getSafeMagnitude(meanTorque);
  const PxVec3 endpointDelta = getEndpointMeanTargetDelta();
  const PxVec3 endpointVelocityDelta =
      getEndpointMeanTargetVelocityDelta();
  const bool partialProbe =
      isForceReactionCase() || isEndpointProbe() ||
      isSphericalConeCase() || isNativeBreakReactionCase() ||
      isRevoluteMotorFamilyCase();
  const char *capability = partialProbe ? "PARTIAL" : "SUPPORTED";
  const char *validation = partialProbe ? "PROBE" : "GATED";
  std::printf(
      "[AVBD_GATE] schema=1 snippet=SnippetJoint case=%s joint=%s solver=%s "
      "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
      "dispatcherThreads=%u "
      "capability=%s validation=%s status=%s reason=%s "
      "nonFinite=%u physicsErrors=%u physicsWarnings=%u fetchFailures=%u "
      "fetchErrorState=%u launchFailures=%u expectedHits=%u hitChains=%u "
      "responseProjectiles=%u minProjectileResponseFraction=%.9g "
      "expectedTargetResponses=%u respondedTargetChains=%u "
      "linearTargetResponses=%u notApplicableTargetResponses=%u "
      "minTargetResponseFraction=%.9g targetResponseGate=%s "
      "incompleteTargetResponses=%u "
      "wrongFirstContacts=%u crossChainContacts=%u incompleteObservations=%u "
      "maxQuaternionNormError=%.9g "
      "maxAbsPosition=%.9g maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
      "maxAnchorError=%.9g anchorRms=%.9g maxAngularLimitViolation=%.9g "
      "maxLinearLimitViolation=%.9g d6LockedLinearMax=%.9g "
      "d6LimitedAngularDofs=0 d6LateAngularSpeedMean=%.9g "
      "d6LateAngularSpeedMax=%.9g maxConstrainedAngularError=%.9g "
      "lateAwakeRatio=%.9g brokenCount=%u breakCallbacks=%u "
      "breakCallbackIdentityMatches=%u "
      "breakCallbackConstraintMismatches=%u "
      "breakCallbackExternalReferenceMismatches=%u "
      "breakCallbackTypeMismatches=%u "
      "breakCallbackBrokenFlagMismatches=%u "
      "breakCallbackDuplicateMismatches=%u "
      "breakCallbackPollMismatches=%u "
      "firstBrokenFrame=%u peakEnergy=%.9g tailEnergyW1=%.9g "
      "tailEnergyW2=%.9g tailEnergyW3=%.9g tailToPeak=%.9g "
      "jitterReproduced=%u anchorCap=%.9g angularLimitCap=%.9g "
      "linearLimitCap=%.9g constrainedAngularCap=%.9g "
      "minProjectileResponseCap=%.9g minTargetResponseCap=%.9g "
      "impactLinearSpeedCap=%.9g impactAngularSpeedCap=%.9g "
      "responseWindowFrames=%u forceReads=%u forceStateSamples=%u "
      "forceSteadyBeginFrame=%u expectedForceSamples=%u "
      "forceSampleAttempts=%u forceSamples=%u "
      "nonFiniteForceSamples=%u nonFiniteSteadyForceSamples=%u "
      "nonFiniteForceStateSamples=%u "
      "forceX=%.9g forceY=%.9g forceZ=%.9g forceMagnitude=%.9g "
      "meanSampleForceMagnitude=%.9g meanSampleForceRatio=%.9g "
      "actualMass=%.9g gravityX=%.9g "
      "gravityY=%.9g gravityZ=%.9g expectedWeight=%.9g forceRatio=%.9g "
      "forceDirectionDot=%.9g meanVectorOrthogonalRatio=%.9g "
      "forceOrthogonalRmsRatio=%.9g "
      "torqueX=%.9g torqueY=%.9g torqueZ=%.9g torqueMagnitude=%.9g "
      "torqueRms=%.9g maxTorqueMagnitude=%.9g "
      "forceMaxPositionError=%.9g forceMaxPositionErrorFrame=%u "
      "forceMaxRotationError=%.9g forceMaxLinearSpeed=%.9g "
      "forceMaxLinearSpeedFrame=%u forceSteadyMaxLinearSpeed=%.9g "
      "forceSteadyMaxLinearSpeedFrame=%u forceFinalPositionError=%.9g "
      "forceFinalLinearSpeed=%.9g "
      "forceMaxAngularSpeed=%.9g topologyDynamicActors=%u "
      "topologyStaticActors=%u topologyConstraints=%u "
      "finalDynamicActors=%u finalStaticActors=%u finalConstraints=%u "
      "cleanupDynamicActors=%u cleanupStaticActors=%u "
      "cleanupConstraints=%u cleanupComplete=%u forceRatioMin=%.9g "
      "forceRatioMax=%.9g forceDirectionDotMin=%.9g "
      "forceOrthogonalRatioMax=%.9g forceTorqueMax=%.9g "
      "forceFixture=%s forceAnchorOffsetX=%.9g forceAnchorOffsetY=%.9g "
      "forceAnchorOffsetZ=%.9g expectedTorqueX=%.9g expectedTorqueY=%.9g "
      "expectedTorqueZ=%.9g expectedTorqueMagnitude=%.9g torqueRatio=%.9g "
      "meanSampleTorqueRatio=%.9g torqueDirectionDot=%.9g "
      "torqueOrthogonalRmsRatio=%.9g actor0FramePositionError=%.9g "
      "actor1FramePositionError=%.9g torqueRatioMin=%.9g "
      "torqueRatioMax=%.9g torqueDirectionDotMin=%.9g "
      "torqueOrthogonalRatioMax=%.9g "
      "endpoint=%s endpointActor0=%s endpointActor1=%s endpointProbe=%u "
      "endpointActorOrderValid=%u endpointFrameWitnessValid=%u "
      "endpointFixtureWitnessValid=%u endpointLimitEnabled=%u "
      "endpointShapeCount=%u endpointShapeRadius=%.9g "
      "endpointWorldFramePositionError=%.9g "
      "endpointDynamicLocalPositionError=%.9g "
      "endpointWorldFrameRotationDot=%.9g "
      "endpointDynamicLocalRotationDot=%.9g "
      "endpointShapeLocalPositionError=%.9g "
      "endpointShapeLocalRotationDot=%.9g "
      "endpointInitialDynamicActors=%u endpointInitialStaticActors=%u "
      "endpointInitialConstraints=%u endpointFinalDynamicActors=%u "
      "endpointFinalStaticActors=%u endpointFinalConstraints=%u "
      "endpointCleanupDynamicActors=%u endpointCleanupStaticActors=%u "
      "endpointCleanupConstraints=%u endpointCleanupComplete=%u "
      "endpointExpectedAxisX=%.9g endpointExpectedAxisY=%.9g "
      "endpointExpectedAxisZ=%.9g endpointActor0AxisX=%.9g "
      "endpointActor0AxisY=%.9g endpointActor0AxisZ=%.9g "
      "endpointActor1AxisX=%.9g endpointActor1AxisY=%.9g "
      "endpointActor1AxisZ=%.9g endpointBodyRotationDot=%.9g "
      "endpointExpectedAxisDot=%.9g endpointDynamicLocalAxisX=%.9g "
      "endpointDynamicLocalAxisY=%.9g endpointDynamicLocalAxisZ=%.9g "
      "endpointDynamicLocalAxisDot=%.9g endpointLaunchFrame=%u "
      "endpointLaunchAxisDot=%.9g endpointStateSampleAttempts=%u "
      "endpointStateSamples=%u endpointBaselineSamples=%u "
      "endpointResponseSampleAttempts=%u endpointResponseSamples=%u "
      "endpointNonFiniteStateSamples=%u "
      "endpointNonFiniteResponseSamples=%u "
      "endpointBaselinePositionX=%.9g endpointBaselinePositionY=%.9g "
      "endpointBaselinePositionZ=%.9g endpointBaselineVelocityX=%.9g "
      "endpointBaselineVelocityY=%.9g endpointBaselineVelocityZ=%.9g "
      "endpointMaxPrecontactPositionDrift=%.9g "
      "endpointMaxPrecontactSpeed=%.9g "
      "endpointTargetDeltaX=%.9g endpointTargetDeltaY=%.9g "
      "endpointTargetDeltaZ=%.9g endpointTargetDeltaQuantity=%s "
      "endpointSignedTargetDelta=%.9g "
      "endpointDirectionDot=%.9g endpointOrthogonalDelta=%.9g "
      "endpointOrthogonalRms=%.9g endpointOrthogonalRatio=%.9g "
      "endpointMaxSampleOrthogonalDelta=%.9g "
      "endpointVelocityDeltaX=%.9g endpointVelocityDeltaY=%.9g "
      "endpointVelocityDeltaZ=%.9g endpointVelocityDeltaQuantity=%s "
      "endpointSignedVelocityDelta=%.9g "
      "endpointVelocityDirectionDot=%.9g "
      "endpointVelocityOrthogonalDelta=%.9g "
      "endpointVelocityOrthogonalRms=%.9g "
      "endpointVelocityOrthogonalRatio=%.9g "
      "endpointMaxSampleVelocityOrthogonalDelta=%.9g "
      "endpointMaxTransverseAnchor=%.9g endpointResponseGate=%s "
      "endpointPositionResponseMin=%.9g "
      "endpointVelocityResponseMin=%.9g endpointDirectionDotMin=%.9g "
      "endpointOrthogonalRatioMax=%.9g "
      "endpointPositionOrthogonalAbsoluteEpsilon=%.9g "
      "endpointVelocityOrthogonalAbsoluteEpsilon=%.9g "
      "endpointPrecontactPositionDriftMax=%.9g "
      "endpointPrecontactSpeedMax=%.9g "
      "endpointTransverseAnchorMax=%.9g endpointJoint=%s "
      "endpointExcitation=%s endpointCalibrationRequired=%u "
      "endpointProjectileCount=%u endpointDriveEnabled=%u "
      "endpointNonIdentityWitnessValid=%u "
      "endpointPerpendicularAxisX=%.9g endpointPerpendicularAxisY=%.9g "
      "endpointPerpendicularAxisZ=%.9g endpointBodyWorldAxisDot=%.9g "
      "endpointAngularLaunchAttempts=%u endpointAngularLaunchSuccesses=%u "
      "endpointLaunchWakeValid=%u endpointLaunchVelocityError=%.9g "
      "endpointRequestedAngularVelocityX=%.9g "
      "endpointRequestedAngularVelocityY=%.9g "
      "endpointRequestedAngularVelocityZ=%.9g "
      "endpointActualLaunchAngularVelocityX=%.9g "
      "endpointActualLaunchAngularVelocityY=%.9g "
      "endpointActualLaunchAngularVelocityZ=%.9g "
      "endpointResponseAggregationSamples=%u endpointApiSamples=%u "
      "endpointNonFiniteApiSamples=%u "
      "endpointBaselineOrientationX=%.9g endpointBaselineOrientationY=%.9g "
      "endpointBaselineOrientationZ=%.9g endpointBaselineOrientationW=%.9g "
      "endpointBaselineAngularVelocityX=%.9g "
      "endpointBaselineAngularVelocityY=%.9g "
      "endpointBaselineAngularVelocityZ=%.9g "
      "endpointMaxPrelaunchOrientationDrift=%.9g "
      "endpointMaxPrelaunchAngularSpeed=%.9g "
      "endpointMinSignedRotation=%.9g "
      "endpointMinSignedAngularVelocity=%.9g "
      "endpointPerSampleSignedGate=%s "
      "endpointMeanRawJointAngleDelta=%.9g "
      "endpointMeanSemanticJointAngleDelta=%.9g "
      "endpointLastRawJointAngleDelta=%.9g "
      "endpointLastSemanticJointAngleDelta=%.9g "
      "endpointMaxJointAnglePoseMismatch=%.9g "
      "endpointMeanApiVelocityMagnitude=%.9g "
      "endpointMaxApiVelocityMagnitudeMismatch=%.9g "
      "endpointMaxAnchorError=%.9g endpointMaxAxisMisalignment=%.9g "
      "endpointAngularResponseWindowFrames=%u endpointAngularSettleFrames=%u "
      "endpointAngularPoseResponseMin=%.9g "
      "endpointAngularVelocityResponseMin=%.9g "
      "endpointAngularDirectionDotMin=%.9g "
      "endpointAngularOrthogonalRatioMax=%.9g "
      "endpointAngularOrthogonalAbsoluteEpsilon=%.9g "
      "endpointAngularPrelaunchOrientationDriftMax=%.9g "
      "endpointAngularPrelaunchSpeedMax=%.9g "
      "endpointAngularAnchorMax=%.9g "
      "endpointAngularAxisMisalignmentMax=%.9g "
      "endpointAngularJointAngleMismatchMax=%.9g "
      "forceActorOrder=%s forceAppliedActor0X=%.9g "
      "forceAppliedActor0Y=%.9g forceAppliedActor0Z=%.9g "
      "forceAppliedActor1X=%.9g forceAppliedActor1Y=%.9g "
      "forceAppliedActor1Z=%.9g forceExpectedActor0X=%.9g "
      "forceExpectedActor0Y=%.9g forceExpectedActor0Z=%.9g "
      "forcePairActorOrderValid=%u forcePairInitialSeparation=%.9g "
      "forcePairMaxSeparationError=%.9g forcePairFinalSeparation=%.9g "
      "forcePairMaxRelativeSpeed=%.9g forcePairFinalRelativeSpeed=%.9g "
      "forcePairMaxCenterOfMassError=%.9g "
      "forcePairFinalCenterOfMassError=%.9g "
      "forcePairMaxTotalMomentum=%.9g "
      "forcePairFinalTotalMomentum=%.9g forcePairTotalMomentumMax=%.9g "
      "forcePairSeparationErrorMax=%.9g forcePairRelativeSpeedMax=%.9g "
      "forcePairCenterOfMassErrorMax=%.9g\n",
      getHeadlessCaseName(gHeadlessCase),
      getGateJointName(),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gGateStats.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, capability, validation,
      evaluation.status, evaluation.reason,
      gGateStats.nonFinite, physicsErrors, physicsWarnings,
      gGateStats.fetchFailures, gGateStats.fetchErrorState,
      gGateStats.launchFailures, evaluation.expectedHits, evaluation.hitChains,
      evaluation.responseProjectiles,
      double(evaluation.minProjectileResponseFraction),
      evaluation.expectedTargetResponses, evaluation.respondedTargetChains,
      evaluation.linearTargetResponses, evaluation.notApplicableTargetResponses,
      double(evaluation.minTargetResponseFraction),
      getAuthorityTargetResponseGateName(evaluation.expectedTargetResponses),
      evaluation.incompleteTargetResponses,
      evaluation.wrongFirstContacts, evaluation.crossChainContacts,
      evaluation.incompleteObservations,
      double(gGateStats.maxQuaternionNormError),
      double(gGateStats.maxAbsPosition), double(gGateStats.maxLinearSpeed),
      double(gGateStats.maxAngularSpeed), double(gGateStats.maxAnchorError),
      double(anchorRms), double(maxAngularLimitViolation),
      double(gGateStats.maxPrismaticLimitViolation),
      double(gGateStats.maxD6LockedLinearError),
      double(d6LateAngularSpeedMean), double(d6LateAngularSpeedMax),
      double(gGateStats.maxConstrainedAngularError), double(lateAwakeRatio),
      gFixedStats.brokenCount,
      gGateStats.breakCallbackCount,
      gGateStats.breakCallbackIdentityMatches,
      gGateStats.breakCallbackConstraintMismatches,
      gGateStats.breakCallbackExternalReferenceMismatches,
      gGateStats.breakCallbackTypeMismatches,
      gGateStats.breakCallbackBrokenFlagMismatches,
      gGateStats.breakCallbackDuplicateMismatches,
      gGateStats.breakCallbackPollMismatches, gFixedStats.firstBrokenFrame,
      double(gGateStats.peakKineticEnergy),
      double(evaluation.tailEnergyW1), double(evaluation.tailEnergyW2),
      double(evaluation.tailEnergyW3), double(tailToPeak),
      evaluation.jitterReproduced ? 1u : 0u, double(gAnchorErrorCap),
      double(gAngularLimitViolationCap),
      double(gLinearLimitViolationCap),
      double(gConstrainedAngularErrorCap),
      double(gMinProjectileResponseFraction),
      double(gMinTargetResponseFraction),
      double(gImpactLinearSpeedCap), double(gImpactAngularSpeedCap),
      gProjectileResponseWindowFrames, gForceStaticStats.forceReads,
      gForceStaticStats.stateSamples, gForceStaticStats.steadyBeginFrame,
      gForceStaticStats.expectedSteadySamples,
      gForceStaticStats.steadySampleAttempts,
      gForceStaticStats.steadySamples,
      gForceStaticStats.nonFiniteForceSamples,
      gForceStaticStats.nonFiniteSteadyForceSamples,
      gForceStaticStats.nonFiniteStateSamples, double(meanForce.x),
      double(meanForce.y), double(meanForce.z), double(meanForceMagnitude),
      double(meanSampleForceMagnitude),
      double(getForceStaticMeanSampleRatio()),
      double(gForceStaticStats.actualMass),
      double(gForceStaticStats.gravity.x),
      double(gForceStaticStats.gravity.y),
      double(gForceStaticStats.gravity.z),
      double(getForceStaticExpectedWeight()), double(getForceStaticRatio()),
      double(getForceStaticDirectionDot()),
      double(getForceStaticMeanVectorOrthogonalRatio()),
      double(getForceStaticOrthogonalRmsRatio()), double(meanTorque.x),
      double(meanTorque.y), double(meanTorque.z),
      double(meanTorqueMagnitude),
      double(getForceStaticTorqueRms()),
      double(gForceStaticStats.maxAngularMagnitude),
      double(gForceStaticStats.maxPositionError),
      gForceStaticStats.maxPositionErrorFrame,
      double(gForceStaticStats.maxRotationError),
      double(gForceStaticStats.maxLinearSpeed),
      gForceStaticStats.maxLinearSpeedFrame,
      double(gForceStaticStats.steadyMaxLinearSpeed),
      gForceStaticStats.steadyMaxLinearSpeedFrame,
      double(gForceStaticStats.finalPositionError),
      double(gForceStaticStats.finalLinearSpeed),
      double(gForceStaticStats.maxAngularSpeed),
      gForceStaticStats.topologyDynamicActors,
      gForceStaticStats.topologyStaticActors,
      gForceStaticStats.topologyConstraints,
      gForceStaticStats.finalDynamicActors,
      gForceStaticStats.finalStaticActors,
      gForceStaticStats.finalConstraints,
      gForceStaticStats.cleanupDynamicActors,
      gForceStaticStats.cleanupStaticActors,
      gForceStaticStats.cleanupConstraints,
      gForceStaticStats.cleanupComplete ? 1u : 0u,
      double(gForceRatioMinimum), double(gForceRatioMaximum),
      double(gForceDirectionDotMinimum),
      double(gForceOrthogonalRatioMaximum), double(gForceTorqueMaximum),
      getForceFixtureName(),
      double(gForceStaticStats.anchorOffset.x),
      double(gForceStaticStats.anchorOffset.y),
      double(gForceStaticStats.anchorOffset.z),
      double(gForceStaticStats.expectedTorque.x),
      double(gForceStaticStats.expectedTorque.y),
      double(gForceStaticStats.expectedTorque.z),
      double(getForceStaticExpectedTorqueMagnitude()),
      double(getForceStaticTorqueRatio()),
      double(getForceStaticMeanSampleTorqueRatio()),
      double(getForceStaticTorqueDirectionDot()),
      double(getForceStaticTorqueOrthogonalRmsRatio()),
      double(gForceStaticStats.actor0FramePositionError),
      double(gForceStaticStats.actor1FramePositionError),
      double(gForceTorqueRatioMinimum), double(gForceTorqueRatioMaximum),
      double(gForceTorqueDirectionDotMinimum),
      double(gForceTorqueOrthogonalRatioMaximum),
      getEndpointName(gEndpointKind), getEndpointActor0Name(),
      getEndpointActor1Name(), isEndpointProbe() ? 1u : 0u,
      gEndpointStats.actorOrderValid ? 1u : 0u,
      gEndpointStats.frameWitnessValid ? 1u : 0u,
      gEndpointStats.fixtureWitnessValid ? 1u : 0u,
      gEndpointStats.limitEnabled ? 1u : 0u,
      gEndpointStats.shapeCount, double(gEndpointStats.shapeRadius),
      double(gEndpointStats.worldFramePositionError),
      double(gEndpointStats.dynamicLocalPositionError),
      double(gEndpointStats.worldFrameRotationDot),
      double(gEndpointStats.dynamicLocalRotationDot),
      double(gEndpointStats.shapeLocalPositionError),
      double(gEndpointStats.shapeLocalRotationDot),
      gEndpointStats.initialDynamicActors,
      gEndpointStats.initialStaticActors,
      gEndpointStats.initialConstraints,
      gEndpointStats.finalDynamicActors,
      gEndpointStats.finalStaticActors,
      gEndpointStats.finalConstraints,
      gEndpointStats.cleanupDynamicActors,
      gEndpointStats.cleanupStaticActors,
      gEndpointStats.cleanupConstraints,
      gEndpointStats.cleanupComplete ? 1u : 0u,
      double(gEndpointStats.expectedAxis.x),
      double(gEndpointStats.expectedAxis.y),
      double(gEndpointStats.expectedAxis.z),
      double(gEndpointStats.actor0Axis.x),
      double(gEndpointStats.actor0Axis.y),
      double(gEndpointStats.actor0Axis.z),
      double(gEndpointStats.actor1Axis.x),
      double(gEndpointStats.actor1Axis.y),
      double(gEndpointStats.actor1Axis.z),
      double(gEndpointStats.bodyRotationDot),
      double(gEndpointStats.expectedAxisDot),
      double(gEndpointStats.dynamicLocalAxis.x),
      double(gEndpointStats.dynamicLocalAxis.y),
      double(gEndpointStats.dynamicLocalAxis.z),
      double(gEndpointStats.dynamicLocalAxisDot),
      gEndpointStats.actualLaunchFrame,
      double(gEndpointStats.launchDirection.dot(
          gEndpointStats.expectedAxis)),
      gEndpointStats.stateSampleAttempts, gEndpointStats.stateSamples,
      gEndpointStats.responseBaselineSamples,
      gEndpointStats.responseSampleAttempts,
      gEndpointStats.responseSamples,
      gEndpointStats.nonFiniteStateSamples,
      gEndpointStats.nonFiniteResponseSamples,
      double(gEndpointStats.responseBaselinePosition.x),
      double(gEndpointStats.responseBaselinePosition.y),
      double(gEndpointStats.responseBaselinePosition.z),
      double(gEndpointStats.responseBaselineVelocity.x),
      double(gEndpointStats.responseBaselineVelocity.y),
      double(gEndpointStats.responseBaselineVelocity.z),
      double(gEndpointStats.maxPrecontactPositionDrift),
      double(gEndpointStats.maxPrecontactSpeed),
      double(endpointDelta.x), double(endpointDelta.y),
      double(endpointDelta.z), getEndpointTargetDeltaQuantityName(),
      double(getEndpointSignedTargetDelta()),
      double(getEndpointDirectionDot()),
      double(getEndpointOrthogonalDelta()),
      double(getEndpointPositionOrthogonalRms()),
      double(getEndpointOrthogonalRatio()),
      double(getEndpointMaxTargetOrthogonalDelta()),
      double(endpointVelocityDelta.x), double(endpointVelocityDelta.y),
      double(endpointVelocityDelta.z), getEndpointVelocityDeltaQuantityName(),
      double(getEndpointSignedTargetVelocityDelta()),
      double(getEndpointVelocityDirectionDot()),
      double(getEndpointVelocityOrthogonalDelta()),
      double(getEndpointVelocityOrthogonalRms()),
      double(getEndpointVelocityOrthogonalRatio()),
      double(getEndpointMaxVelocityOrthogonalDelta()),
      double(getEndpointMaxAnchorError()),
      getEndpointResponseAuthorityName(),
      double(gEndpointPositionResponseMinimum),
      double(gEndpointVelocityResponseMinimum),
      double(gEndpointDirectionDotMinimum),
      double(gEndpointOrthogonalRatioMaximum),
      double(gEndpointPositionOrthogonalAbsoluteEpsilon),
      double(gEndpointVelocityOrthogonalAbsoluteEpsilon),
      double(gEndpointPrecontactPositionDriftMaximum),
      double(gEndpointPrecontactSpeedMaximum),
      double(gEndpointTransverseAnchorMaximum),
      getEndpointJointKindName(), getEndpointExcitationName(),
      isRevoluteEndpointProbe() ? 1u : 0u,
      static_cast<PxU32>(gProjectiles.size()),
      gEndpointAngularStats.driveEnabled ? 1u : 0u,
      gEndpointAngularStats.nonIdentityWitnessValid ? 1u : 0u,
      double(gEndpointAngularStats.perpendicularAxis.x),
      double(gEndpointAngularStats.perpendicularAxis.y),
      double(gEndpointAngularStats.perpendicularAxis.z),
      double(gEndpointAngularStats.bodyWorldAxisDot),
      gEndpointAngularStats.launchAttempts,
      gEndpointAngularStats.launchSuccesses,
      gEndpointAngularStats.launchWakeValid ? 1u : 0u,
      double(gEndpointAngularStats.launchVelocityError),
      double(gEndpointAngularStats.requestedAngularVelocity.x),
      double(gEndpointAngularStats.requestedAngularVelocity.y),
      double(gEndpointAngularStats.requestedAngularVelocity.z),
      double(gEndpointAngularStats.actualLaunchAngularVelocity.x),
      double(gEndpointAngularStats.actualLaunchAngularVelocity.y),
      double(gEndpointAngularStats.actualLaunchAngularVelocity.z),
      gEndpointAngularStats.tailSamples,
      gEndpointAngularStats.apiSamples,
      gEndpointAngularStats.nonFiniteApiSamples,
      double(gEndpointAngularStats.responseBaselineOrientation.x),
      double(gEndpointAngularStats.responseBaselineOrientation.y),
      double(gEndpointAngularStats.responseBaselineOrientation.z),
      double(gEndpointAngularStats.responseBaselineOrientation.w),
      double(gEndpointAngularStats.responseBaselineAngularVelocity.x),
      double(gEndpointAngularStats.responseBaselineAngularVelocity.y),
      double(gEndpointAngularStats.responseBaselineAngularVelocity.z),
      double(gEndpointAngularStats.maxPrelaunchOrientationDrift),
      double(gEndpointAngularStats.maxPrelaunchAngularSpeed),
      double(isRevoluteEndpointProbe()
                 ? gEndpointAngularStats.minSignedRotation
                 : 0.0f),
      double(isRevoluteEndpointProbe()
                 ? gEndpointAngularStats.minSignedAngularVelocity
                 : 0.0f),
      getEndpointPerSampleSignedGateName(),
      double(getEndpointMeanRawJointAngleDelta()),
      double(getEndpointMeanSemanticJointAngleDelta()),
      double(gEndpointAngularStats.lastRawJointAngleDelta),
      double(gEndpointAngularStats.lastSemanticJointAngleDelta),
      double(gEndpointAngularStats.maxJointAnglePoseMismatch),
      double(getEndpointMeanApiVelocityMagnitude()),
      double(gEndpointAngularStats.maxApiVelocityMagnitudeMismatch),
      double(getEndpointMaxAnchorError()),
      double(gEndpointAngularStats.maxAxisMisalignment),
      gEndpointAngularResponseWindowFrames,
      gEndpointAngularSettleFrames,
      double(gEndpointAngularPoseResponseMinimum),
      double(gEndpointAngularVelocityResponseMinimum),
      double(gEndpointAngularDirectionDotMinimum),
      double(gEndpointAngularOrthogonalRatioMaximum),
      double(gEndpointAngularOrthogonalAbsoluteEpsilon),
      double(gEndpointAngularPrelaunchOrientationDriftMaximum),
      double(gEndpointAngularPrelaunchSpeedMaximum),
      double(gEndpointAngularAnchorMaximum),
      double(gEndpointAngularAxisMisalignmentMaximum),
      double(gEndpointAngularJointAngleMismatchMaximum),
      getForceActorOrderName(),
      double(gForceStaticStats.appliedForceActor0.x),
      double(gForceStaticStats.appliedForceActor0.y),
      double(gForceStaticStats.appliedForceActor0.z),
      double(gForceStaticStats.appliedForceActor1.x),
      double(gForceStaticStats.appliedForceActor1.y),
      double(gForceStaticStats.appliedForceActor1.z),
      double(gForceStaticStats.expectedLinearForce.x),
      double(gForceStaticStats.expectedLinearForce.y),
      double(gForceStaticStats.expectedLinearForce.z),
      gForceStaticStats.pairActorOrderValid ? 1u : 0u,
      double(gForceStaticStats.pairInitialSeparation),
      double(gForceStaticStats.pairMaxSeparationError),
      double(gForceStaticStats.pairFinalSeparation),
      double(gForceStaticStats.pairMaxRelativeSpeed),
      double(gForceStaticStats.pairFinalRelativeSpeed),
      double(gForceStaticStats.pairMaxCenterOfMassError),
      double(gForceStaticStats.pairFinalCenterOfMassError),
      double(gForceStaticStats.pairMaxTotalMomentum),
      double(gForceStaticStats.pairFinalTotalMomentum),
      double(gForcePairTotalMomentumMaximum),
      double(gForcePairSeparationErrorMaximum),
      double(gForcePairRelativeSpeedMaximum),
      double(gForcePairCenterOfMassErrorMaximum));
}

static int reportConfigurationError(const Snippets::HeadlessOptions &options,
                                    const char *message) {
  const bool forceReaction =
      Snippets::equalsIgnoreCase(options.caseName.c_str(), "force-static") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(), "force-offset") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(), "force-pair") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "force-pair-disabled");
  const bool sphericalCone =
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "spherical-cone-inside") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "spherical-cone-outside");
  const bool nativeBreakReaction =
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "native-reaction") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "native-no-break") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "native-break");
  const bool revoluteMotor =
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-limit") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-freespin") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-ratio") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-contact") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-kinematic") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-off-principal") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-off-center") ||
      Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                 "revolute-motor-spatial") ||
      Snippets::equalsIgnoreCase(
          options.caseName.c_str(),
          "revolute-motor-dynamic-limit") ||
      Snippets::equalsIgnoreCase(
          options.caseName.c_str(),
          "revolute-motor-dynamic-freespin") ||
      Snippets::equalsIgnoreCase(
          options.caseName.c_str(),
          "revolute-motor-dynamic-off-principal") ||
      Snippets::equalsIgnoreCase(
          options.caseName.c_str(),
          "revolute-motor-dynamic-off-center") ||
      Snippets::equalsIgnoreCase(
          options.caseName.c_str(),
          "revolute-motor-dynamic-spatial");
  const bool partialProbe =
      forceReaction || sphericalCone || nativeBreakReaction ||
      revoluteMotor ||
      gEndpointOptionMentioned;
  std::printf("[AVBD_GATE_ERROR] snippet=SnippetJoint message=%s\n", message);
  std::printf(
      "[AVBD_GATE] schema=1 snippet=SnippetJoint case=config-error joint=unknown solver=%s "
      "execution=%s requestedFrames=%u completedFrames=0 dt=%.9g seed=%u "
      "dispatcherThreads=%u "
      "capability=%s validation=%s status=ERROR reason=config "
      "nonFinite=0 physicsErrors=0 physicsWarnings=0 endpoint=%s "
      "endpointActor0=%s endpointActor1=%s endpointProbe=%u\n",
      Snippets::getSolverTypeName(options.solverType),
      Snippets::getExecutionName(options.execution), options.frames,
      double(options.dt), options.seed, options.dispatcherThreads,
      partialProbe ? "PARTIAL" : "SUPPORTED",
      partialProbe ? "PROBE" : "GATED", getEndpointName(gEndpointKind),
      getEndpointActor0Name(), getEndpointActor1Name(),
      gEndpointOptionMentioned ? 1u : 0u);
  return Snippets::eHEADLESS_CONFIG_ERROR;
}

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  gEndpointKind = eENDPOINT_IMPLICIT;
  gEndpointOptionMentioned = false;
  gForceActorOrder = eFORCE_ACTOR_ORDER_NORMAL;
  gSphericalConeTopology = eSPHERICAL_CONE_STATIC_DYNAMIC;
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && Snippets::hasOptionPrefix(argv[i], "--endpoint="))
      gEndpointOptionMentioned = true;
  }

  Snippets::HeadlessOptions defaults;
  defaults.caseName = "passive";
  defaults.frames = 1400;
  defaults.seed = 1;
  defaults.dispatcherThreads = 2;
  defaults.dt = 1.0f / 60.0f;

  Snippets::HeadlessOptions options;
  std::string parseError;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
                                            parseError))
    return reportConfigurationError(options, parseError.c_str());

  bool jointSeen = false;
  bool endpointSeen = false;
  bool actorOrderSeen = false;
  bool topologySeen = false;
  bool headlessOnlyOptionSeen = false;
  JointKind impactJointKind = eJOINT_SPHERICAL;
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
    if (Snippets::hasOptionPrefix(arg, "--joint=")) {
      if (jointSeen)
        return reportConfigurationError(options, "duplicate_--joint");
      jointSeen = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseJointKind(arg + std::strlen("--joint="), impactJointKind))
        return reportConfigurationError(options, "invalid_--joint_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--endpoint=")) {
      if (endpointSeen)
        return reportConfigurationError(options, "duplicate_--endpoint");
      endpointSeen = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseEndpointKind(arg + std::strlen("--endpoint="),
                                gEndpointKind))
        return reportConfigurationError(options,
                                        "invalid_--endpoint_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--actor-order=")) {
      if (actorOrderSeen)
        return reportConfigurationError(options, "duplicate_--actor-order");
      actorOrderSeen = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseForceActorOrder(
              arg + std::strlen("--actor-order="), gForceActorOrder))
        return reportConfigurationError(options,
                                        "invalid_--actor-order_value");
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--topology=")) {
      if (topologySeen)
        return reportConfigurationError(options, "duplicate_--topology");
      topologySeen = true;
      headlessOnlyOptionSeen = true;
      if (!tryParseSphericalConeTopology(
              arg + std::strlen("--topology="),
              gSphericalConeTopology))
        return reportConfigurationError(options,
                                        "invalid_--topology_value");
      continue;
    }
    return reportConfigurationError(options, "unknown_argument");
  }

#ifndef RENDER_SNIPPET
  options.headless = true;
#endif

  JointHeadlessCase headlessCase = eCASE_PASSIVE;
  if (!tryParseHeadlessCase(options.caseName.c_str(), headlessCase))
    return reportConfigurationError(options, "invalid_--case_value");
  options.caseName = getHeadlessCaseName(headlessCase);

  const bool nativeBreakReactionCase =
      headlessCase == eCASE_NATIVE_REACTION ||
      headlessCase == eCASE_NATIVE_NO_BREAK ||
      headlessCase == eCASE_NATIVE_BREAK;
  if ((headlessCase == eCASE_IMPACT_SINGLE ||
       nativeBreakReactionCase) &&
      !jointSeen)
    return reportConfigurationError(
        options, nativeBreakReactionCase
                     ? "native_break_reaction_requires_--joint"
                     : "impact_requires_--joint");
  if (headlessCase != eCASE_IMPACT_SINGLE &&
      !nativeBreakReactionCase && jointSeen)
    return reportConfigurationError(
        options, "--joint_requires_impact_or_native_break_reaction");
  if (nativeBreakReactionCase &&
      impactJointKind != eJOINT_PRISMATIC &&
      impactJointKind != eJOINT_REVOLUTE)
    return reportConfigurationError(
        options,
        "native_break_reaction_requires_prismatic_or_revolute");
  if (endpointSeen && headlessCase != eCASE_IMPACT_SINGLE)
    return reportConfigurationError(options, "--endpoint_requires_impact");
  if (actorOrderSeen && headlessCase != eCASE_FORCE_PAIR &&
      headlessCase != eCASE_FORCE_PAIR_DISABLED)
    return reportConfigurationError(options,
                                    "--actor-order_requires_force-pair");
  const bool sphericalConeCase =
      headlessCase == eCASE_SPHERICAL_CONE_INSIDE ||
      headlessCase == eCASE_SPHERICAL_CONE_OUTSIDE;
  if (topologySeen && !sphericalConeCase)
    return reportConfigurationError(
        options, "--topology_requires_spherical-cone");
  if (sphericalConeCase && !topologySeen)
    return reportConfigurationError(
        options, "spherical-cone_requires_--topology");
  if (endpointSeen && impactJointKind != eJOINT_PRISMATIC &&
      impactJointKind != eJOINT_REVOLUTE)
    return reportConfigurationError(options,
                                    "--endpoint_requires_prismatic_or_revolute");
  if (!options.headless && headlessOnlyOptionSeen)
    return reportConfigurationError(options, "gate_option_requires_--headless");

  const bool forceReactionCase = headlessCase == eCASE_FORCE_STATIC ||
                                 headlessCase == eCASE_FORCE_OFFSET ||
                                 headlessCase == eCASE_FORCE_PAIR ||
                                 headlessCase == eCASE_FORCE_PAIR_DISABLED;
  PxU32 forceStaticFrequency = 0;
  if (forceReactionCase &&
      !getForceStaticFrequency(options.dt, forceStaticFrequency))
    return reportConfigurationError(
        options,
        headlessCase == eCASE_FORCE_OFFSET
            ? "force_offset_dt_requires_30_60_or_120hz"
            : ((headlessCase == eCASE_FORCE_PAIR ||
                headlessCase == eCASE_FORCE_PAIR_DISABLED)
                   ? "force_pair_dt_requires_30_60_or_120hz"
                   : "force_static_dt_requires_30_60_or_120hz"));

  if (!options.framesExplicit) {
    if (headlessCase == eCASE_IMPACT_ALL ||
        headlessCase == eCASE_IMPACT_SINGLE)
      options.frames = 1800;
    else if (headlessCase == eCASE_WIDE_JOINT_STRESS)
      options.frames = 600;
    else if (headlessCase == eCASE_FIXED_NO_BREAK ||
             headlessCase == eCASE_FIXED_BREAK)
      options.frames = 600;
    else if (sphericalConeCase)
      options.frames = 360;
    else if (nativeBreakReactionCase)
      options.frames = 600;
    else if (forceReactionCase)
      options.frames = forceStaticFrequency * gForceStaticDurationSeconds;
  }
  if (options.headless && headlessCase == eCASE_PASSIVE &&
      options.frames < 1300)
    return reportConfigurationError(options, "passive_frames_must_be_at_least_1300");
  if (headlessCase != eCASE_PASSIVE && !forceReactionCase &&
      options.frames < 360)
    return reportConfigurationError(options, "impact_frames_must_be_at_least_360");
  if (forceReactionCase &&
      options.frames != forceStaticFrequency * gForceStaticDurationSeconds)
    return reportConfigurationError(
        options,
        headlessCase == eCASE_FORCE_OFFSET
            ? "force_offset_requires_10_seconds"
            : ((headlessCase == eCASE_FORCE_PAIR ||
                headlessCase == eCASE_FORCE_PAIR_DISABLED)
                   ? "force_pair_requires_10_seconds"
                   : "force_static_requires_10_seconds"));
  if (options.execution == Snippets::eHEADLESS_SEQUENTIAL &&
      options.solverType != PxSolverType::eAVBD)
    return reportConfigurationError(options, "sequential_requires_avbd");
  if (!forceReactionCase &&
      PxAbs(options.dt - (1.0f / 60.0f)) > 1e-7f)
    return reportConfigurationError(options, "dt_requires_60hz_calibration");
  if (!Snippets::applyExecutionEnvironment(options))
    return reportConfigurationError(options, "execution_environment_failed");

  gHeadlessOptions = options;
  gHeadlessCase = headlessCase;
  gImpactJointKind = impactJointKind;
  gSolverType = options.solverType;
  gFixedLinearBreakForce = 1000.0f;
  gFixedAngularBreakForce = 100000.0f;
  if (gHeadlessCase == eCASE_FIXED_NO_BREAK) {
    gFixedLinearBreakForce = PX_MAX_F32;
    gFixedAngularBreakForce = PX_MAX_F32;
  } else if (gHeadlessCase == eCASE_FIXED_BREAK) {
    gFixedLinearBreakForce = 250.0f;
    gFixedAngularBreakForce = PX_MAX_F32;
  }

#ifdef RENDER_SNIPPET
  if (!options.headless) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  Snippets::printHeadlessConfig("SnippetJoint", gHeadlessOptions);
  if (isWideJointStressCase()) {
    std::printf(
        "[AVBD_JOINT_STRESS_CONFIG] chainCount=1 chainLength=32 "
        "joint=d6 gravity=negative-y ground=plane projectile=none\n");
  } else if (isNativeBreakReactionCase()) {
    const PxReal configuredThreshold =
        gHeadlessCase == eCASE_NATIVE_BREAK
            ? gNativeLowBreakThreshold
            : (gHeadlessCase == eCASE_NATIVE_NO_BREAK
                   ? gNativeHighBreakThreshold
                   : PX_MAX_F32);
    std::printf(
        "[PROBE] [SnippetJointNativeBreakReactionConfig] "
        "case=%s joint=%s loadKind=%s loadMagnitude=100 "
        "configuredThreshold=%.9g gravity=zero ground=none "
        "topology=static-dynamic actor0=world actor1=dynamic "
        "mass=1 inertia=(1,1,1) warmupFrames=%u "
        "reactionRatioRange=[%.9g,%.9g] "
        "reactionDirectionDotMin=%.9g "
        "reactionOrthogonalRatioMax=%.9g "
        "constrainedErrorMax=%.9g constrainedSpeedMax=%.9g\n",
        getHeadlessCaseName(gHeadlessCase),
        getJointKindName(gImpactJointKind),
        isNativeAngularReactionCase() ? "angular" : "linear",
        double(configuredThreshold), gNativeReactionWarmupFrames,
        double(gNativeReactionRatioMinimum),
        double(gNativeReactionRatioMaximum),
        double(gNativeReactionDirectionMinimum),
        double(gNativeReactionOrthogonalRatioMaximum),
        double(gNativeConstrainedErrorMaximum),
        double(gNativeConstrainedSpeedMaximum));
  } else if (isSphericalConeCase()) {
    const PxReal configuredSwingY =
        isSphericalConeInsideCase() ? gSphericalConeInsideY
                                    : gSphericalConeOutsideY;
    const PxReal configuredSwingZ =
        isSphericalConeInsideCase() ? gSphericalConeInsideZ
                                    : gSphericalConeOutsideZ;
    std::printf(
        "[PROBE] [SnippetJointSphericalConeConfig] case=%s topology=%s "
        "limitY=%.9g limitZ=%.9g initialSwingY=%.9g "
        "initialSwingZ=%.9g gravity=zero ground=none "
        "jointFrames=identity mass=1 inertia=(1,1,1) "
        "finalRadiusTolerance=%.9g lateRadiusTolerance=%.9g "
        "insideDeviationTolerance=%.9g "
        "minimumRadiusCorrection=%.9g "
        "angularMomentumMaximum=%.9g anchorSeparationMaximum=%.9g\n",
        getHeadlessCaseName(gHeadlessCase),
        getSphericalConeTopologyName(),
        double(gSphericalConeLimitY), double(gSphericalConeLimitZ),
        double(configuredSwingY), double(configuredSwingZ),
        double(gSphericalConeFinalRadiusTolerance),
        double(gSphericalConeLateRadiusTolerance),
        double(gSphericalConeInsideDeviationTolerance),
        double(gSphericalConeMinimumRadiusCorrection),
        double(gSphericalConeAngularMomentumMaximum),
        double(gSphericalConeAnchorSeparationMaximum));
  } else if (isEndpointProbe()) {
    if (isRevoluteEndpointProbe()) {
      std::printf(
          "[PROBE] [SnippetJointEndpointConfig] endpoint=%s actor0=%s "
          "actor1=%s joint=revolute gravity=zero ground=none "
          "targetShape=sphere projectileShape=none bodyRotationYDeg=45 "
          "worldFrameRotationYDeg=90 expectedAxis=negative-z "
          "perpendicularAxis=positive-y revoluteLimit=disabled "
          "revoluteDrive=disabled excitation=direct-angular-velocity "
          "launchFrame=%u axialAngularSpeed=%.9g "
          "transverseAngularSpeed=%.9g responseWindowFrames=%u "
          "settleFrames=%u poseResponseMin=%.9g "
          "angularVelocityResponseMin=%.9g directionDotMin=%.9g "
          "orthogonalRatioMax=%.9g orthogonalAbsoluteEpsilon=%.9g "
          "prelaunchPositionDriftMax=%.9g prelaunchLinearSpeedMax=%.9g "
          "prelaunchOrientationDriftMax=%.9g "
          "prelaunchAngularSpeedMax=%.9g anchorMax=%.9g "
          "axisMisalignmentMax=%.9g jointAnglePoseMismatchMax=%.9g "
          "calibrationRequired=1\n",
          getEndpointName(gEndpointKind), getEndpointActor0Name(),
          getEndpointActor1Name(), gImpactLaunchFrame,
          double(gEndpointAngularAxialLaunchSpeed),
          double(gEndpointAngularTransverseLaunchSpeed),
          gEndpointAngularResponseWindowFrames,
          gEndpointAngularSettleFrames,
          double(gEndpointAngularPoseResponseMinimum),
          double(gEndpointAngularVelocityResponseMinimum),
          double(gEndpointAngularDirectionDotMinimum),
          double(gEndpointAngularOrthogonalRatioMaximum),
          double(gEndpointAngularOrthogonalAbsoluteEpsilon),
          double(gEndpointPrecontactPositionDriftMaximum),
          double(gEndpointPrecontactSpeedMaximum),
          double(gEndpointAngularPrelaunchOrientationDriftMaximum),
          double(gEndpointAngularPrelaunchSpeedMaximum),
          double(gEndpointAngularAnchorMaximum),
          double(gEndpointAngularAxisMisalignmentMaximum),
          double(gEndpointAngularJointAngleMismatchMaximum));
    } else {
    std::printf(
        "[PROBE] [SnippetJointEndpointConfig] endpoint=%s actor0=%s "
        "actor1=%s joint=prismatic gravity=zero ground=none "
        "targetShape=sphere projectileShape=sphere bodyRotationYDeg=45 "
        "worldFrameRotationYDeg=90 expectedAxis=negative-z "
        "prismaticLimit=disabled "
        "dynamicLocalAxisDot=%.9g launchFrame=%u speed=%.9g "
        "responseWindowFrames=%u positionResponseMin=%.9g "
        "velocityResponseMin=%.9g directionDotMin=%.9g "
        "orthogonalRatioMax=%.9g "
        "positionOrthogonalAbsoluteEpsilon=%.9g "
        "velocityOrthogonalAbsoluteEpsilon=%.9g "
        "precontactPositionDriftMax=%.9g precontactSpeedMax=%.9g "
        "transverseAnchorMax=%.9g\n",
        getEndpointName(gEndpointKind), getEndpointActor0Name(),
        getEndpointActor1Name(), double(PxSqrt(0.5f)),
        gImpactLaunchFrame, double(gImpactSpeed),
        gEndpointResponseWindowFrames,
        double(gEndpointPositionResponseMinimum),
        double(gEndpointVelocityResponseMinimum),
        double(gEndpointDirectionDotMinimum),
        double(gEndpointOrthogonalRatioMaximum),
        double(gEndpointPositionOrthogonalAbsoluteEpsilon),
        double(gEndpointVelocityOrthogonalAbsoluteEpsilon),
        double(gEndpointPrecontactPositionDriftMaximum),
        double(gEndpointPrecontactSpeedMaximum),
        double(gEndpointTransverseAnchorMaximum));
    }
  } else if (isImpactCase()) {
    std::printf(
        "[SnippetJointImpactConfig] launchFrame=%u radius=%.9g height=%.9g "
        "speed=%.9g transverseOffset=%.9g observationFrames=%u "
        "direction=negative-z target=middle-body fixedLinearBreakForce=%.9g "
        "fixedAngularBreakForce=%.9g\n",
        gImpactLaunchFrame, double(gImpactRadius), double(gImpactHeight),
        double(gImpactSpeed), double(gImpactTransverseOffset),
        gProjectileObservationFrames, double(gFixedLinearBreakForce),
        double(gFixedAngularBreakForce));
  } else if (isForceReactionCase()) {
    const PxReal pairForceSign =
        isForcePairDisabledCase() ? -1.0f : 1.0f;
    const PxVec3 configuredActor0Applied =
        isForcePairCase()
            ? PxVec3(0.0f,
                     pairForceSign *
                         (gForceActorOrder == eFORCE_ACTOR_ORDER_SWAPPED
                              ? -gForcePairAppliedMagnitude
                              : gForcePairAppliedMagnitude),
                     0.0f)
            : PxVec3(0.0f);
    const PxVec3 configuredActor1Applied = -configuredActor0Applied;
    const PxVec3 configuredExpectedActor0Force =
        isForcePairCase()
            ? -configuredActor0Applied
            : PxVec3(0.0f, -gForceStaticMass * gGravityMagnitude, 0.0f);
    const PxVec3 configuredAnchorOffset =
        isForcePairCase()
            ? PxVec3(0.0f,
                     gForceActorOrder == eFORCE_ACTOR_ORDER_SWAPPED ? -1.0f
                                                                    : 1.0f,
                     0.0f)
            : (isForceOffsetCase() ? gForceOffsetAnchor : PxVec3(0.0f));
    std::printf(
        "[PROBE] [SnippetJointForceConfig] fixture=%s mass=%.9g gravity=%.9g "
        "expectedWeight=%.9g warmupSeconds=%u durationSeconds=%u "
        "anchorOffset=(%.9g,%.9g,%.9g) expectedTorqueMagnitude=%.9g "
        "actorOrder=%s appliedForceActor0=(%.9g,%.9g,%.9g) "
        "appliedForceActor1=(%.9g,%.9g,%.9g) "
        "expectedActor0Force=(%.9g,%.9g,%.9g) "
        "ground=none "
        "forceRatioRange=[%.9g,%.9g] directionDotMin=%.9g "
        "orthogonalRatioMax=%.9g torqueMax=%.9g "
        "torqueRatioRange=[%.9g,%.9g] torqueDirectionDotMin=%.9g "
        "torqueOrthogonalRatioMax=%.9g\n",
        getForceFixtureName(),
        double(gForceStaticMass), double(gGravityMagnitude),
        double(getForceStaticExpectedWeight()), gForceStaticWarmupSeconds,
        gForceStaticDurationSeconds,
        double(configuredAnchorOffset.x),
        double(configuredAnchorOffset.y),
        double(configuredAnchorOffset.z),
        double(isForceOffsetCase()
                   ? gForceStaticMass * gGravityMagnitude *
                         getSafeMagnitude(gForceOffsetAnchor)
                   : 0.0f),
        getForceActorOrderName(),
        double(configuredActor0Applied.x),
        double(configuredActor0Applied.y),
        double(configuredActor0Applied.z),
        double(configuredActor1Applied.x),
        double(configuredActor1Applied.y),
        double(configuredActor1Applied.z),
        double(configuredExpectedActor0Force.x),
        double(configuredExpectedActor0Force.y),
        double(configuredExpectedActor0Force.z),
        double(gForceRatioMinimum),
        double(gForceRatioMaximum), double(gForceDirectionDotMinimum),
        double(gForceOrthogonalRatioMaximum),
        double(gForceTorqueMaximum), double(gForceTorqueRatioMinimum),
        double(gForceTorqueRatioMaximum),
        double(gForceTorqueDirectionDotMinimum),
        double(gForceTorqueOrthogonalRatioMaximum));
  }

  initPhysics(false);
  if (gInitializationFailed) {
    GateEvaluation evaluation;
    setGateError(evaluation, "initialization");
    cleanupPhysics(false);
    printGateResult(evaluation, gErrorCallback.getFatalCount(),
                    gErrorCallback.getWarningCount());
    return evaluation.exitCode;
  }

  for (PxU32 i = 0; i < gHeadlessOptions.frames; i++) {
    stepPhysics(false);
    if (gGateStats.fetchFailures)
      break;
  }
  GateEvaluation evaluation = evaluateGate();
  printGateDetails();
  cleanupPhysics(false);

  if (isForceReactionCase() && !gForceStaticStats.cleanupComplete)
    setInfrastructureErrorOverFailure(evaluation, "force_cleanup");
  if (isRevoluteMotorCase() &&
      !gRevoluteMotorStats.cleanupComplete)
    setInfrastructureErrorOverFailure(evaluation,
                                      "revolute_motor_cleanup");
  if (isRevoluteMotorCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorStats.cleanupDynamicActors,
        gRevoluteMotorStats.cleanupStaticActors,
        gRevoluteMotorStats.cleanupConstraints,
        gRevoluteMotorStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorLimitCase() &&
      !gRevoluteMotorLimitStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_limit_cleanup");
  if (isRevoluteMotorLimitCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorLimitCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorLimitStats.cleanupDynamicActors,
        gRevoluteMotorLimitStats.cleanupStaticActors,
        gRevoluteMotorLimitStats.cleanupConstraints,
        gRevoluteMotorLimitStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorFreeSpinCase() &&
      !gRevoluteMotorFreeSpinStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_freespin_cleanup");
  if (isRevoluteMotorFreeSpinCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorFreeSpinCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorFreeSpinStats.cleanupDynamicActors,
        gRevoluteMotorFreeSpinStats.cleanupStaticActors,
        gRevoluteMotorFreeSpinStats.cleanupConstraints,
        gRevoluteMotorFreeSpinStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorRatioCase() &&
      !gRevoluteMotorRatioStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_ratio_cleanup");
  if (isRevoluteMotorRatioCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorRatioCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorRatioStats.cleanupDynamicActors,
        gRevoluteMotorRatioStats.cleanupStaticActors,
        gRevoluteMotorRatioStats.cleanupConstraints,
        gRevoluteMotorRatioStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorContactCase() &&
      !gRevoluteMotorContactStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_contact_cleanup");
  if (isRevoluteMotorContactCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorContactCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorContactStats.cleanupDynamicActors,
        gRevoluteMotorContactStats.cleanupStaticActors,
        gRevoluteMotorContactStats.cleanupConstraints,
        gRevoluteMotorContactStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorKinematicCase() &&
      !gRevoluteMotorKinematicStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_kinematic_cleanup");
  if (isRevoluteMotorKinematicCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorKinematicCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorKinematicStats.cleanupDynamicActors,
        gRevoluteMotorKinematicStats.cleanupStaticActors,
        gRevoluteMotorKinematicStats.cleanupConstraints,
        gRevoluteMotorKinematicStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorOffPrincipalCase() &&
      !gRevoluteMotorOffPrincipalStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_off_principal_cleanup");
  if (isRevoluteMotorOffPrincipalCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorOffPrincipalCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorOffPrincipalStats.cleanupDynamicActors,
        gRevoluteMotorOffPrincipalStats.cleanupStaticActors,
        gRevoluteMotorOffPrincipalStats.cleanupConstraints,
        gRevoluteMotorOffPrincipalStats.cleanupComplete ? 1u : 0u);
  }
  if (isRevoluteMotorOffCenterCase() &&
      !gRevoluteMotorOffCenterStats.cleanupComplete)
    setInfrastructureErrorOverFailure(
        evaluation, "revolute_motor_off_center_cleanup");
  if (isRevoluteMotorOffCenterCase()) {
    std::printf(
        "[PROBE] [SnippetJointRevoluteMotorOffCenterCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gRevoluteMotorOffCenterStats.cleanupDynamicActors,
        gRevoluteMotorOffCenterStats.cleanupStaticActors,
        gRevoluteMotorOffCenterStats.cleanupConstraints,
        gRevoluteMotorOffCenterStats.cleanupComplete ? 1u : 0u);
  }
  if (isNativeBreakReactionCase() &&
      !gNativeBreakReactionStats.cleanupComplete)
    setInfrastructureErrorOverFailure(evaluation, "native_cleanup");
  if (isNativeBreakReactionCase()) {
    std::printf(
        "[PROBE] [SnippetJointNativeBreakReactionCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gNativeBreakReactionStats.cleanupDynamicActors,
        gNativeBreakReactionStats.cleanupStaticActors,
        gNativeBreakReactionStats.cleanupConstraints,
        gNativeBreakReactionStats.cleanupComplete ? 1u : 0u);
  }
  if (isSphericalConeCase() &&
      !gSphericalConeStats.cleanupComplete)
    setInfrastructureErrorOverFailure(evaluation,
                                      "spherical_cone_cleanup");
  if (isSphericalConeCase()) {
    std::printf(
        "[PROBE] [SnippetJointSphericalConeCleanup] "
        "dynamicActors=%u staticActors=%u constraints=%u "
        "cleanupComplete=%u\n",
        gSphericalConeStats.cleanupDynamicActors,
        gSphericalConeStats.cleanupStaticActors,
        gSphericalConeStats.cleanupConstraints,
        gSphericalConeStats.cleanupComplete ? 1u : 0u);
  }
  if (isEndpointProbe() && !gEndpointStats.cleanupComplete)
    setInfrastructureErrorOverFailure(evaluation, "endpoint_cleanup");

  const PxU32 physicsErrors = gErrorCallback.getFatalCount();
  if (physicsErrors ||
      ((isForceReactionCase() || isEndpointProbe() ||
        isSphericalConeCase() || isNativeBreakReactionCase() ||
        isRevoluteMotorFamilyCase()) &&
       gGateStats.fetchErrorState)) {
    if (isForceReactionCase() || isEndpointProbe() ||
        isSphericalConeCase() || isNativeBreakReactionCase() ||
        isRevoluteMotorFamilyCase())
      setInfrastructureErrorOverFailure(evaluation, "physx_error");
    else if (evaluation.exitCode == Snippets::eHEADLESS_PASS)
      setGateFailure(evaluation, "physx_error");
  }
  printGateResult(evaluation, physicsErrors, gErrorCallback.getWarningCount());
  return static_cast<int>(evaluation.exitCode);
}
