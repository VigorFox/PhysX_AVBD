// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdNativeMotorVelocity.h"
#include "avbd/solver/joint/DyAvbdJointVelocityPolicies.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

AvbdJointMotorAdmissionState buildAvbdJointMotorAdmission(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const physx::PxVec3 &gravity, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  AvbdJointMotorAdmissionState state;
  const bool gearCandidate =
      isNativeRevoluteMotorGearVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, gearJoints,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts,
          state.nativeRevoluteMotorGearJointIndex);
  const bool nativeCandidate =
      isSingleNativeRevoluteMotorVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  bool transientContact = false;
  const bool contactCandidate =
      isContactCoupledNativeRevoluteMotorVelocityProjectionSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          gravity, numGear, numSoftParticles, numSoftBodies,
          numSoftContacts, transientContact);
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    if (joint.motorEnabled == 0u)
      continue;
    const bool supported =
        (nativeCandidate && jointIndex == 0u) ||
        (gearCandidate &&
         jointIndex == state.nativeRevoluteMotorGearJointIndex) ||
        (contactCandidate && jointIndex == 0u);
    assignAvbdJointObjective(
        joint.objectiveProgram,
        supported ? AvbdVelocityObjectiveOwner::JointFinalize
                  : AvbdVelocityObjectiveOwner::Unsupported,
        AvbdJointObjectiveKind::NativeRevoluteMotor, 1u,
        eJOINT_SOURCE_NATIVE_MOTOR, joint.cacheKey);
  }

  state.nativeRevoluteMotorVelocityProjectionIsland =
      nativeCandidate && numD6 > 0u &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::NativeRevoluteMotor);
  state.nativeRevoluteMotorGearVelocityProjectionIsland =
      gearCandidate && state.nativeRevoluteMotorGearJointIndex < numD6 &&
      hasAvbdJointObjective(
          d6Joints[state.nativeRevoluteMotorGearJointIndex].objectiveProgram,
          AvbdJointObjectiveKind::NativeRevoluteMotor);
  state.contactCoupledNativeRevoluteMotorVelocityProjectionIsland =
      contactCandidate && numD6 > 0u &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::NativeRevoluteMotor);

  if (contactCandidate || transientContact) {
    const AvbdVelocityObjectiveOwner owner =
        state.contactCoupledNativeRevoluteMotorVelocityProjectionIsland
            ? AvbdVelocityObjectiveOwner::JointFinalize
            : AvbdVelocityObjectiveOwner::Unsupported;
    bool assignmentSupported = true;
    physx::PxU64 objectiveKey = ~physx::PxU64(0);
    for (physx::PxU32 c = 0; c < numContacts; ++c)
      objectiveKey = physx::PxMin(objectiveKey, contacts[c].cacheKey);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      if (!canAssignAvbdVelocityObjective(
              contacts[c].objectiveProgram, owner,
              AvbdVelocityObjectiveKind::PassiveFriction,
              AvbdVelocityObjectiveSpan::NormalAndTangentCone,
              AvbdVelocityObjectiveReconstruction::SolveStartInertial,
              numContacts, objectiveKey)) {
        state.contactCoupledNativeRevoluteMotorVelocityProjectionIsland =
            false;
        assignmentSupported = false;
        break;
      }
    }
    if (assignmentSupported) {
      for (physx::PxU32 c = 0; c < numContacts; ++c)
        assignAvbdVelocityObjective(
            contacts[c].objectiveProgram, owner,
            AvbdVelocityObjectiveKind::PassiveFriction,
            AvbdVelocityObjectiveSpan::NormalAndTangentCone,
            AvbdVelocityObjectiveReconstruction::SolveStartInertial,
            numContacts, objectiveKey);
    } else {
      transientContact = false;
      if (numD6 > 0u)
        fallbackAvbdJointObjective(
            d6Joints[0].objectiveProgram,
            AvbdJointObjectiveKind::NativeRevoluteMotor);
      for (physx::PxU32 c = 0; c < numContacts; ++c)
        invalidateAvbdVelocityObjective(contacts[c].objectiveProgram);
    }
  }
  return state;
}

bool isSingleNativeRevoluteMotorVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies == 0 || numBodies > 2 ||
      numContacts != 0 || !d6Joints || numD6 != 1 ||
      numGear != 0 || numSoftParticles != 0 || numSoftBodies != 0 ||
      numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const physx::PxU32 twistMotion = joint.getAngularMotion(0);
  const bool limitedTwist = twistMotion == 1u;
  const bool freeSpin =
      (joint.sourceFlags & AvbdD6JointConstraint::
                               eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0;
  const bool nonUnitDriveRatio =
      physx::PxAbs(joint.motorGearRatio - 1.0f) > 1e-6f;
  if (joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
      joint.motorEnabled == 0u ||
      !(joint.motorMaxForce > 0.0f) ||
      !physx::PxIsFinite(joint.motorMaxForce) ||
      !physx::PxIsFinite(joint.motorTargetVelocity) ||
      !physx::PxIsFinite(joint.motorGearRatio) ||
      !(joint.motorGearRatio > 0.0f) ||
      joint.linearMotion != 0u ||
      (twistMotion != 1u && twistMotion != 2u) ||
      joint.getAngularMotion(1) != 0u ||
      joint.getAngularMotion(2) != 0u ||
      joint.driveFlags != 0u || joint.driveAccelerationFlags != 0u ||
      joint.coneAngleLimit > 0.0f ||
      !joint.externalAngularStepA.isFinite() ||
      !joint.externalAngularStepB.isFinite())
    return false;
  if (limitedTwist &&
      (!physx::PxIsFinite(joint.angularLimitLower.x) ||
       !physx::PxIsFinite(joint.angularLimitUpper.x) ||
       joint.angularLimitLower.x >= joint.angularLimitUpper.x))
    return false;

  const physx::PxU32 endpoint[2] = {joint.header.bodyIndexA,
                                    joint.header.bodyIndexB};
  const bool dynamicA =
      endpoint[0] < numBodies && bodies[endpoint[0]].invMass > 0.0f;
  const bool dynamicB =
      endpoint[1] < numBodies && bodies[endpoint[1]].invMass > 0.0f;
  if ((!dynamicA && !dynamicB) ||
      (dynamicA && dynamicB && endpoint[0] == endpoint[1]))
    return false;
  // A limited motor may own either one centered dynamic endpoint against a
  // stationary endpoint or one centered dynamic pair. Both topologies share
  // the same scalar twist derivative and unilateral limit active set.
  // Off-center, off-principal, and prescribed-endpoint variants need a wider
  // bounded objective and remain fail closed below.
  if (limitedTwist) {
    const bool oneDynamic =
        numBodies == 1u && dynamicA != dynamicB;
    const bool dynamicPair =
        numBodies == 2u && dynamicA && dynamicB;
    if ((!oneDynamic && !dynamicPair) ||
        joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
        joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
      return false;
    if (dynamicPair) {
      if (joint.anchorA.magnitudeSquared() > 1e-8f ||
          joint.anchorB.magnitudeSquared() > 1e-8f)
        return false;
    } else {
      const physx::PxVec3 &dynamicAnchor =
          dynamicA ? joint.anchorA : joint.anchorB;
      if (dynamicAnchor.magnitudeSquared() > 1e-8f)
        return false;
    }
  }
  // Free-spin may likewise own one centered dynamic endpoint or one centered
  // principal-response dynamic pair. Its one-sided impulse row owns the
  // complete physical hinge derivative, including preservation of a
  // super-target solve-entry speed.
  if (freeSpin) {
    const bool oneDynamic =
        numBodies == 1u && dynamicA != dynamicB;
    const bool dynamicPair =
        numBodies == 2u && dynamicA && dynamicB;
    if ((!oneDynamic && !dynamicPair) ||
        joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
        joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
      return false;
    if (dynamicPair) {
      if (joint.anchorA.magnitudeSquared() > 1e-8f ||
          joint.anchorB.magnitudeSquared() > 1e-8f)
        return false;
    } else {
      const physx::PxVec3 &dynamicAnchor =
          dynamicA ? joint.anchorA : joint.anchorB;
      if (dynamicAnchor.magnitudeSquared() > 1e-8f)
        return false;
    }
  }
  if (freeSpin && limitedTwist)
    return false;
  if ((limitedTwist || freeSpin) && nonUnitDriveRatio)
    return false;
  if (nonUnitDriveRatio) {
    if (twistMotion != 2u || freeSpin || numBodies != 2u ||
        !dynamicA || !dynamicB ||
        joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
        joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
      return false;
    physx::PxVec3 localAxisA =
        joint.localFrameA.rotate(
            physx::PxVec3(1.0f, 0.0f, 0.0f));
    physx::PxVec3 localAxisB =
        joint.localFrameB.rotate(
            physx::PxVec3(1.0f, 0.0f, 0.0f));
    if (localAxisA.normalize() <= 1e-6f ||
        localAxisB.normalize() <= 1e-6f ||
        joint.anchorA.cross(localAxisA).magnitudeSquared() > 1e-8f ||
        joint.anchorB.cross(localAxisB).magnitudeSquared() > 1e-8f)
      return false;
  }

  physx::PxVec3 worldAxis =
      dynamicA
          ? (bodies[endpoint[0]].rotation * joint.localFrameA)
                .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
          : joint.localFrameA.rotate(
                physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (worldAxis.normalize() <= 1e-6f)
    return false;

  const bool allowCoupledOffPrincipalResponse =
      ((numBodies == 1u && dynamicA != dynamicB) ||
       (numBodies == 2u && dynamicA && dynamicB)) &&
      twistMotion == 2u && !freeSpin && !nonUnitDriveRatio &&
      joint.externalAngularStepA.magnitudeSquared() <= 1e-12f &&
      joint.externalAngularStepB.magnitudeSquared() <= 1e-12f;
  const bool allowCoupledOffCenterResponse =
      ((numBodies == 1u && dynamicA != dynamicB) ||
       (numBodies == 2u && dynamicA && dynamicB)) &&
      twistMotion == 2u && !freeSpin && !nonUnitDriveRatio &&
      joint.externalAngularStepA.magnitudeSquared() <= 1e-12f &&
      joint.externalAngularStepB.magnitudeSquared() <= 1e-12f;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    const bool dynamic = side == 0 ? dynamicA : dynamicB;
    const physx::PxVec3 localAxis =
        (side == 0 ? joint.localFrameA : joint.localFrameB)
            .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    const physx::PxVec3 anchor =
        side == 0 ? joint.anchorA : joint.anchorB;
    const physx::PxVec3 externalStep =
        side == 0 ? joint.externalAngularStepA
                  : joint.externalAngularStepB;
    if (!dynamic) {
      const physx::PxReal stepScale =
          physx::PxMax(externalStep.magnitude(), 1e-8f);
      // A prescribed endpoint may participate only when its motion is the
      // same physical hinge derivative owned by this scalar objective.
      if (externalStep.cross(worldAxis).magnitude() >
          stepScale * 1e-4f)
        return false;
      continue;
    }
    const AvbdSolverBody &body = bodies[endpoint[side]];
    if (body.lockFlags != 0 ||
        externalStep.magnitudeSquared() > 1e-12f)
      return false;
    // A perpendicular dynamic anchor couples the motor torque to locked
    // translation. The one-body free-twist topology below owns the complete
    // fixed-anchor spatial derivative; every wider topology remains rejected.
    // Collinear offsets are rotation-invariant.
    if (anchor.cross(localAxis).magnitudeSquared() > 1e-8f &&
        !allowCoupledOffCenterResponse)
      return false;
    const physx::PxVec3 response =
        body.invInertiaWorld.transform(worldAxis);
    const physx::PxReal responseScale =
        physx::PxMax(response.magnitude(), 1e-8f);
    // A post-finalize scalar motor row may not change the locked swing
    // derivative. The one-body free-twist topology below owns motor plus both
    // locked-swing derivatives as one complete angular objective; every
    // wider off-principal topology remains rejected.
    if (response.cross(worldAxis).magnitude() >
            responseScale * 1e-4f &&
        !allowCoupledOffPrincipalResponse)
      return false;
  }
  return true;
}

bool isContactCoupledNativeRevoluteMotorVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const physx::PxVec3 &gravity, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts,
    bool &unsupportedTransientContact) {
  unsupportedTransientContact = false;
  if (!bodies || numBodies != 2u || !contacts || numContacts < 2u ||
      numContacts > 8u || !d6Joints || numD6 != 1u || numGear != 0u ||
      numSoftParticles != 0u || numSoftBodies != 0u ||
      numSoftContacts != 0u)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  if (bodyAIndex >= numBodies || bodyBIndex >= numBodies ||
      bodyAIndex == bodyBIndex ||
      bodies[bodyAIndex].invMass <= 0.0f ||
      bodies[bodyBIndex].invMass <= 0.0f ||
      bodies[bodyAIndex].lockFlags != 0u ||
      bodies[bodyBIndex].lockFlags != 0u ||
      bodies[bodyAIndex].linearDamping != 0.0f ||
      bodies[bodyBIndex].linearDamping != 0.0f ||
      bodies[bodyAIndex].angularDampingBody != 0.0f ||
      bodies[bodyBIndex].angularDampingBody != 0.0f ||
      physx::PxAbs(bodies[bodyAIndex].invMass -
                   bodies[bodyBIndex].invMass) >
          physx::PxMax(bodies[bodyAIndex].invMass,
                       bodies[bodyBIndex].invMass) *
              1e-5f ||
      joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
      joint.motorEnabled == 0u || !(joint.motorMaxForce > 0.0f) ||
      !physx::PxIsFinite(joint.motorMaxForce) ||
      !physx::PxIsFinite(joint.motorTargetVelocity) ||
      physx::PxAbs(joint.motorGearRatio - 1.0f) > 1e-6f ||
      joint.linearMotion != 0u || joint.getAngularMotion(0) != 2u ||
      joint.getAngularMotion(1) != 0u ||
      joint.getAngularMotion(2) != 0u ||
      joint.driveFlags != 0u || joint.driveAccelerationFlags != 0u ||
      joint.coneAngleLimit > 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0u ||
      joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
      joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
    return false;

  physx::PxVec3 localAxisA =
      joint.localFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  physx::PxVec3 localAxisB =
      joint.localFrameB.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (localAxisA.normalize() <= 1e-6f ||
      localAxisB.normalize() <= 1e-6f ||
      joint.anchorA.cross(localAxisA).magnitudeSquared() > 1e-8f ||
      joint.anchorB.cross(localAxisB).magnitudeSquared() > 1e-8f)
    return false;

  physx::PxVec3 worldAxisA =
      (bodies[bodyAIndex].rotation * joint.localFrameA)
          .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (worldAxisA.normalize() <= 1e-6f)
    return false;

  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  physx::PxVec3 referenceNormal(0.0f);
  bool haveReferenceNormal = false;
  bool haveUnsupportedTransientContact = false;
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contact) ||
        hasKinematicShellAnchor(contact) ||
        (contact.friction <= 0.0f && contact.staticFriction <= 0.0f) ||
        !physx::PxIsFinite(contact.friction) ||
        !physx::PxIsFinite(contact.staticFriction) ||
        !physx::PxIsFinite(contact.restitution) ||
        contact.restitution < 0.0f ||
        contact.targetVelocity.magnitudeSquared() > 1e-12f)
      return false;
    if (contact.persistentPointMatched == 0u)
      haveUnsupportedTransientContact = true;

    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicIsA ? contact.header.bodyIndexA
                   : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies ||
        physx::PxAbs((dynamicIsA ? contact.invMassScaleA
                                 : contact.invMassScaleB) -
                    1.0f) > 1e-6f ||
        physx::PxAbs((dynamicIsA ? contact.invInertiaScaleA
                                 : contact.invInertiaScaleB) -
                    1.0f) > 1e-6f)
      return false;
    contactsPerBody[bodyIndex]++;

    physx::PxVec3 dynamicNormal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    if (dynamicNormal.normalize() <= 1e-6f ||
        gravity.magnitudeSquared() <= 1e-12f ||
        -gravity.getNormalized().dot(dynamicNormal) < 0.9999f)
      return false;
    const physx::PxVec3 localPoint =
        dynamicIsA ? contact.contactPointA : contact.contactPointB;
    const physx::PxVec3 contactArm =
        bodies[bodyIndex].rotation.rotate(localPoint);
    const physx::PxVec3 pointVelocity =
        bodies[bodyIndex].linearVelocity +
        bodies[bodyIndex].angularVelocity.cross(contactArm);
    if (pointVelocity.dot(dynamicNormal) < -0.25f)
      haveUnsupportedTransientContact = true;
    if (!haveReferenceNormal) {
      referenceNormal = dynamicNormal;
      haveReferenceNormal = true;
    } else if (referenceNormal.dot(dynamicNormal) < 0.9999f) {
      return false;
    }
  }
  const bool completeSupport =
      contactsPerBody[0] != 0u && contactsPerBody[1] != 0u;
  unsupportedTransientContact =
      completeSupport && haveUnsupportedTransientContact;
  return completeSupport && !haveUnsupportedTransientContact;
}

struct ContactCoupledVelocityRow {
  physx::PxVec3 linearA;
  physx::PxVec3 angularA;
  physx::PxVec3 linearB;
  physx::PxVec3 angularB;
  physx::PxReal targetVelocity;
};

void projectContactCoupledNativeRevoluteMotorVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint &joint, const physx::PxVec3 &gravity,
    physx::PxReal dt) {
  if (!bodies || numBodies != 2u || !contacts || numContacts == 0u ||
      !(dt > 0.0f))
    return;

  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  if (bodyAIndex >= numBodies || bodyBIndex >= numBodies)
    return;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  physx::PxQuat frameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = frameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !physx::PxIsFinite(frameMagnitude))
    return;
  frameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  const physx::PxVec3 axes[3] = {
      frameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f)),
      frameA.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f)),
      frameA.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f))};
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);

  ContactCoupledVelocityRow hardRows[5];
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    hardRows[axis].linearA = -axes[axis];
    hardRows[axis].angularA = -rA.cross(axes[axis]);
    hardRows[axis].linearB = axes[axis];
    hardRows[axis].angularB = rB.cross(axes[axis]);
    hardRows[axis].targetVelocity = 0.0f;
  }
  for (physx::PxU32 swing = 0; swing < 2; ++swing) {
    hardRows[3u + swing].linearA = physx::PxVec3(0.0f);
    hardRows[3u + swing].angularA = -axes[1u + swing];
    hardRows[3u + swing].linearB = physx::PxVec3(0.0f);
    hardRows[3u + swing].angularB = axes[1u + swing];
    hardRows[3u + swing].targetVelocity = 0.0f;
  }
  ContactCoupledVelocityRow motorRow;
  motorRow.linearA = physx::PxVec3(0.0f);
  motorRow.angularA = -axes[0];
  motorRow.linearB = physx::PxVec3(0.0f);
  motorRow.angularB = axes[0];
  motorRow.targetVelocity = joint.motorTargetVelocity;

  auto rowVelocity =
      [&](const ContactCoupledVelocityRow &row) -> physx::PxReal {
    return row.linearA.dot(bodyA.linearVelocity) +
           row.angularA.dot(bodyA.angularVelocity) +
           row.linearB.dot(bodyB.linearVelocity) +
           row.angularB.dot(bodyB.angularVelocity);
  };
  auto rowResponse =
      [&](const ContactCoupledVelocityRow &row) -> physx::PxReal {
    return bodyA.invMass * row.linearA.magnitudeSquared() +
           row.angularA.dot(
               bodyA.invInertiaWorld.transform(row.angularA)) +
           bodyB.invMass * row.linearB.magnitudeSquared() +
           row.angularB.dot(
               bodyB.invInertiaWorld.transform(row.angularB));
  };
  auto applyRowImpulse =
      [&](const ContactCoupledVelocityRow &row,
          physx::PxReal impulse) {
    bodyA.linearVelocity += row.linearA * (bodyA.invMass * impulse);
    bodyA.angularVelocity +=
        bodyA.invInertiaWorld.transform(row.angularA * impulse);
    bodyB.linearVelocity += row.linearB * (bodyB.invMass * impulse);
    bodyB.angularVelocity +=
        bodyB.invInertiaWorld.transform(row.angularB * impulse);
  };

  physx::PxArray<physx::PxReal> frictionImpulse0(numContacts);
  physx::PxArray<physx::PxReal> frictionImpulse1(numContacts);
  physx::PxArray<physx::PxReal> normalImpulse(numContacts);
  physx::PxArray<physx::PxReal> frictionMu(numContacts);
  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    if (bodyIndex < numBodies)
      contactsPerBody[bodyIndex]++;
  }
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    AvbdContactConstraint &contact = contacts[contactIndex];
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicIsA ? contact.header.bodyIndexA
                   : contact.header.bodyIndexB;
    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxReal reportSign = dynamicIsA ? 1.0f : -1.0f;
    frictionImpulse0[contactIndex] =
        reportSign *
        contact.frictionSweepImpulse.dot(contact.tangent0);
    frictionImpulse1[contactIndex] =
        reportSign *
        contact.frictionSweepImpulse.dot(contact.tangent1);
    const physx::PxReal mu =
        contact.friction > 0.0f ? contact.friction
                               : contact.staticFriction;
    frictionMu[contactIndex] = mu;
    const physx::PxReal existingMagnitude = physx::PxSqrt(
        frictionImpulse0[contactIndex] *
            frictionImpulse0[contactIndex] +
        frictionImpulse1[contactIndex] *
            frictionImpulse1[contactIndex]);
    physx::PxVec3 dynamicNormal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    dynamicNormal.normalize();
    const physx::PxReal mass = 1.0f / body.invMass;
    normalImpulse[contactIndex] =
        mass * physx::PxAbs(gravity.dot(dynamicNormal)) * dt /
        physx::PxReal(physx::PxMax(
            1u, contactsPerBody[bodyIndex]));
    const physx::PxReal physicalFrictionLimit =
        mu * normalImpulse[contactIndex];
    physx::PxReal projected0 = frictionImpulse0[contactIndex];
    physx::PxReal projected1 = frictionImpulse1[contactIndex];
    avbdProjectImpulseCone(physicalFrictionLimit,
                           projected0, projected1);
    if (existingMagnitude > physicalFrictionLimit + 1e-12f) {
      const physx::PxReal delta0 =
          projected0 - frictionImpulse0[contactIndex];
      const physx::PxReal delta1 =
          projected1 - frictionImpulse1[contactIndex];
      const physx::PxVec3 r =
          body.rotation.rotate(dynamicIsA ? contact.contactPointA
                                          : contact.contactPointB);
      const physx::PxVec3 angular0 = r.cross(contact.tangent0);
      const physx::PxVec3 angular1 = r.cross(contact.tangent1);
      body.linearVelocity +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          body.invMass;
      body.angularVelocity += body.invInertiaWorld.transform(
          angular0 * delta0 + angular1 * delta1);
      contact.frictionSweepImpulse +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          reportSign;
    }
    frictionImpulse0[contactIndex] = projected0;
    frictionImpulse1[contactIndex] = projected1;
  }

  physx::PxReal motorImpulse = 0.0f;
  const physx::PxReal maximumMotorImpulse =
      joint.motorMaxForce * dt;
  // The body-static friction sweep is a warm start with zero motor impulse.
  // Repeating every hard joint row, the bounded motor row, and every retained
  // two-axis friction cone makes the final state one coupled velocity
  // objective rather than a post-contact motor replay.
  for (physx::PxU32 iteration = 0; iteration < 96u; ++iteration) {
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numContacts; ++contactIndex) {
      AvbdContactConstraint &contact = contacts[contactIndex];
      const bool dynamicIsA =
          contact.header.bodyIndexA < numBodies;
      const physx::PxU32 bodyIndex =
          dynamicIsA ? contact.header.bodyIndexA
                     : contact.header.bodyIndexB;
      AvbdSolverBody &body = bodies[bodyIndex];
      const physx::PxVec3 r =
          body.rotation.rotate(dynamicIsA ? contact.contactPointA
                                          : contact.contactPointB);
      const physx::PxVec3 dynamicNormal =
          contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
      const physx::PxVec3 normalAngular = r.cross(dynamicNormal);
      const physx::PxReal normalResponse =
          body.invMass +
          normalAngular.dot(
              body.invInertiaWorld.transform(normalAngular));
      if (normalResponse > 1e-12f) {
        const physx::PxVec3 pointVelocity =
            body.linearVelocity + body.angularVelocity.cross(r);
        const physx::PxReal candidate =
            normalImpulse[contactIndex] -
            pointVelocity.dot(dynamicNormal) / normalResponse;
        const physx::PxReal maximumNormalImpulse =
            contact.maxImpulse < PX_MAX_REAL
                ? physx::PxMax(0.0f, contact.maxImpulse)
                : PX_MAX_REAL;
        const physx::PxReal projectedNormal =
            physx::PxClamp(candidate, 0.0f,
                           maximumNormalImpulse);
        const physx::PxReal deltaNormal =
            projectedNormal - normalImpulse[contactIndex];
        body.linearVelocity +=
            dynamicNormal * (body.invMass * deltaNormal);
        body.angularVelocity +=
            body.invInertiaWorld.transform(
                normalAngular * deltaNormal);
        normalImpulse[contactIndex] = projectedNormal;
      }

      const physx::PxVec3 angular0 = r.cross(contact.tangent0);
      const physx::PxVec3 angular1 = r.cross(contact.tangent1);
      const physx::PxReal k00 =
          body.invMass +
          angular0.dot(body.invInertiaWorld.transform(angular0));
      const physx::PxReal k01 =
          angular0.dot(body.invInertiaWorld.transform(angular1));
      const physx::PxReal k11 =
          body.invMass +
          angular1.dot(body.invInertiaWorld.transform(angular1));
      const physx::PxReal determinant = k00 * k11 - k01 * k01;
      if (determinant <= 1e-12f ||
          !physx::PxIsFinite(determinant))
        continue;
      const physx::PxVec3 pointVelocity =
          body.linearVelocity + body.angularVelocity.cross(r);
      const physx::PxReal rhs0 =
          -pointVelocity.dot(contact.tangent0);
      const physx::PxReal rhs1 =
          -pointVelocity.dot(contact.tangent1);
      physx::PxReal projected0 =
          frictionImpulse0[contactIndex] +
          (rhs0 * k11 - rhs1 * k01) / determinant;
      physx::PxReal projected1 =
          frictionImpulse1[contactIndex] +
          (k00 * rhs1 - k01 * rhs0) / determinant;
      const physx::PxReal frictionLimit =
          frictionMu[contactIndex] *
          normalImpulse[contactIndex];
      avbdProjectImpulseCone(frictionLimit,
                             projected0, projected1);
      const physx::PxReal delta0 =
          projected0 - frictionImpulse0[contactIndex];
      const physx::PxReal delta1 =
          projected1 - frictionImpulse1[contactIndex];
      body.linearVelocity +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          body.invMass;
      body.angularVelocity += body.invInertiaWorld.transform(
          angular0 * delta0 + angular1 * delta1);
      const physx::PxReal reportSign =
          dynamicIsA ? 1.0f : -1.0f;
      contact.frictionSweepImpulse +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          reportSign;
      frictionImpulse0[contactIndex] = projected0;
      frictionImpulse1[contactIndex] = projected1;
    }

    for (physx::PxU32 rowIndex = 0; rowIndex < 5u; ++rowIndex) {
      const physx::PxReal response = rowResponse(hardRows[rowIndex]);
      if (response <= 1e-12f)
        continue;
      const physx::PxReal impulse =
          (hardRows[rowIndex].targetVelocity -
           rowVelocity(hardRows[rowIndex])) /
          response;
      applyRowImpulse(hardRows[rowIndex], impulse);
    }

    const physx::PxReal motorResponse = rowResponse(motorRow);
    if (motorResponse > 1e-12f) {
      const physx::PxReal candidate =
          motorImpulse +
          (motorRow.targetVelocity - rowVelocity(motorRow)) /
              motorResponse;
      const physx::PxReal projected =
          physx::PxClamp(candidate, -maximumMotorImpulse,
                         maximumMotorImpulse);
      applyRowImpulse(motorRow, projected - motorImpulse);
      motorImpulse = projected;
    }
  }
  bodyA.projectLockedVelocities();
  bodyB.projectLockedVelocities();
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex)
    contacts[contactIndex].velocityNormalImpulse =
        normalImpulse[contactIndex];
}

static bool solveNativeMotorDense6(
    const physx::PxReal response[6][6],
    const physx::PxReal rhs[6],
    bool motorImpulseClamped,
    physx::PxReal clampedMotorImpulse,
    physx::PxReal solution[6]) {
  physx::PxReal augmented[6][7] = {};
  for (physx::PxU32 row = 0; row < 6; ++row) {
    for (physx::PxU32 column = 0; column < 6; ++column)
      augmented[row][column] = response[row][column];
    augmented[row][6] = rhs[row];
  }
  if (motorImpulseClamped) {
    for (physx::PxU32 row = 0; row < 6; ++row) {
      if (row == 3)
        continue;
      augmented[row][6] -=
          augmented[row][3] * clampedMotorImpulse;
      augmented[row][3] = 0.0f;
    }
    for (physx::PxU32 column = 0; column < 7; ++column)
      augmented[3][column] = 0.0f;
    augmented[3][3] = 1.0f;
    augmented[3][6] = clampedMotorImpulse;
  }

  for (physx::PxU32 column = 0; column < 6; ++column) {
    physx::PxU32 pivot = column;
    physx::PxReal pivotMagnitude =
        physx::PxAbs(augmented[column][column]);
    for (physx::PxU32 row = column + 1; row < 6; ++row) {
      const physx::PxReal candidate =
          physx::PxAbs(augmented[row][column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!physx::PxIsFinite(pivotMagnitude) ||
        pivotMagnitude <= 1e-10f)
      return false;
    if (pivot != column) {
      for (physx::PxU32 entry = column; entry < 7; ++entry) {
        const physx::PxReal temporary =
            augmented[column][entry];
        augmented[column][entry] = augmented[pivot][entry];
        augmented[pivot][entry] = temporary;
      }
    }
    const physx::PxReal inversePivot =
        1.0f / augmented[column][column];
    for (physx::PxU32 entry = column; entry < 7; ++entry)
      augmented[column][entry] *= inversePivot;
    for (physx::PxU32 row = 0; row < 6; ++row) {
      if (row == column)
        continue;
      const physx::PxReal factor = augmented[row][column];
      for (physx::PxU32 entry = column; entry < 7; ++entry)
        augmented[row][entry] -=
            factor * augmented[column][entry];
    }
  }
  for (physx::PxU32 row = 0; row < 6; ++row) {
    solution[row] = augmented[row][6];
    if (!physx::PxIsFinite(solution[row]))
      return false;
  }
  return true;
}

void projectSingleNativeRevoluteMotorVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint, physx::PxReal dt,
    bool conserveDynamicPairAngularMomentum,
    physx::PxReal expectedAngularMomentumOnAxis,
    bool conserveDynamicPairAngularMomentumVector,
    const physx::PxVec3 &expectedAngularMomentumVector,
    bool conserveDynamicPairLinearMomentum,
    const physx::PxVec3 &expectedLinearMomentum,
    bool conserveDynamicPairSpatialMomentum,
    const physx::PxVec3 &expectedSpatialAngularMomentum,
    bool useSolveStartRelativeVelocity,
    physx::PxReal solveStartRelativeVelocity) {
  const physx::PxU32 endpoint[2] = {joint.header.bodyIndexA,
                                    joint.header.bodyIndexB};
  const bool dynamicA =
      endpoint[0] < numBodies && bodies[endpoint[0]].invMass > 0.0f;
  const bool dynamicB =
      endpoint[1] < numBodies && bodies[endpoint[1]].invMass > 0.0f;
  physx::PxVec3 worldAxis =
      dynamicA
          ? (bodies[endpoint[0]].rotation * joint.localFrameA)
                .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
          : joint.localFrameA.rotate(
                physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (worldAxis.normalize() <= 1e-6f)
    return;

  physx::PxVec3 responseA(0.0f);
  physx::PxVec3 responseB(0.0f);
  physx::PxReal unitResponse = 0.0f;
  physx::PxReal motorVelocity = 0.0f;
  const physx::PxReal driveRatio = joint.motorGearRatio;
  if (dynamicA) {
    responseA =
        bodies[endpoint[0]].invInertiaWorld.transform(worldAxis);
    unitResponse += worldAxis.dot(responseA);
    motorVelocity -=
        worldAxis.dot(bodies[endpoint[0]].angularVelocity);
  } else if (dt > 0.0f)
    motorVelocity -=
        worldAxis.dot(joint.externalAngularStepA) / dt;
  if (dynamicB) {
    responseB =
        bodies[endpoint[1]].invInertiaWorld.transform(worldAxis);
    unitResponse +=
        driveRatio * driveRatio * worldAxis.dot(responseB);
    motorVelocity +=
        driveRatio *
        worldAxis.dot(bodies[endpoint[1]].angularVelocity);
  } else if (dt > 0.0f)
    motorVelocity += driveRatio *
                     worldAxis.dot(joint.externalAngularStepB) / dt;
  if (!physx::PxIsFinite(unitResponse) || unitResponse <= 1e-10f)
    return;

  const bool freeSpin =
      (joint.sourceFlags & AvbdD6JointConstraint::
                               eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0;
  const physx::PxU32 dynamicSide = dynamicA ? 0u : 1u;
  const physx::PxVec3 dynamicAnchor =
      dynamicA ? joint.anchorA : joint.anchorB;
  const physx::PxQuat dynamicFrame =
      dynamicA ? joint.localFrameA : joint.localFrameB;
  physx::PxVec3 dynamicLocalAxis =
      dynamicFrame.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  const bool coupledOffCenterResponse =
      dynamicA != dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin && physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      dynamicLocalAxis.normalize() > 1e-6f &&
      dynamicAnchor.cross(dynamicLocalAxis).magnitudeSquared() > 1e-8f;
  if (coupledOffCenterResponse) {
    AvbdSolverBody &body = bodies[endpoint[dynamicSide]];
    if (!(body.invMass > 0.0f))
      return;

    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxMat33 inertia =
        body.invInertiaWorld.getInverse();
    const physx::PxVec3 worldLeverArm =
        body.rotation.rotate(dynamicAnchor);
    const physx::PxReal motorSign = dynamicA ? -1.0f : 1.0f;
    const auto computeMotorImpulse =
        [&](physx::PxReal angularSpeed) -> physx::PxReal {
      const physx::PxVec3 desiredAngular =
          worldAxis * angularSpeed;
      const physx::PxVec3 desiredLinear =
          -desiredAngular.cross(worldLeverArm);
      const physx::PxVec3 linearImpulse =
          (desiredLinear - body.linearVelocity) * mass;
      const physx::PxVec3 angularImpulse =
          inertia.transform(
              desiredAngular - body.angularVelocity);
      const physx::PxVec3 anchorTorque =
          angularImpulse -
          worldLeverArm.cross(linearImpulse);
      return motorSign * worldAxis.dot(anchorTorque);
    };

    const physx::PxReal zeroSpeedImpulse =
        computeMotorImpulse(0.0f);
    const physx::PxReal impulsePerAngularSpeed =
        computeMotorImpulse(1.0f) - zeroSpeedImpulse;
    if (!physx::PxIsFinite(zeroSpeedImpulse) ||
        !physx::PxIsFinite(impulsePerAngularSpeed) ||
        impulsePerAngularSpeed <= 1e-10f)
      return;
    const physx::PxReal targetAngularSpeed =
        motorSign * joint.motorTargetVelocity;
    const physx::PxReal requiredMotorImpulse =
        zeroSpeedImpulse +
        impulsePerAngularSpeed * targetAngularSpeed;
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal motorImpulse =
        physx::PxClamp(requiredMotorImpulse,
                       -maximumMotorImpulse,
                       maximumMotorImpulse);
    const physx::PxReal ownedAngularSpeed =
        (motorImpulse - zeroSpeedImpulse) /
        impulsePerAngularSpeed;
    const physx::PxVec3 candidateAngular =
        worldAxis * ownedAngularSpeed;
    const physx::PxVec3 candidateLinear =
        -candidateAngular.cross(worldLeverArm);
    if (!candidateAngular.isFinite() ||
        !candidateLinear.isFinite() ||
        (body.maxAngularVelocitySq > 0.0f &&
         candidateAngular.magnitudeSquared() >
             body.maxAngularVelocitySq) ||
        (body.maxLinearVelocitySq > 0.0f &&
         candidateLinear.magnitudeSquared() >
             body.maxLinearVelocitySq))
      return;
    body.angularVelocity = candidateAngular;
    body.linearVelocity = candidateLinear;
    return;
  }

  const bool coupledOffPrincipalResponse =
      dynamicA != dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin &&
      physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      ((dynamicA ? responseA : responseB)
           .cross(worldAxis)
           .magnitude() >
       physx::PxMax(
           (dynamicA ? responseA : responseB).magnitude(), 1e-8f) *
           1e-4f);
  if (coupledOffPrincipalResponse) {
    AvbdSolverBody &body =
        bodies[dynamicA ? endpoint[0] : endpoint[1]];
    const physx::PxReal motorSign = dynamicA ? -1.0f : 1.0f;
    const physx::PxVec3 motorJ = worldAxis * motorSign;
    physx::PxVec3 referenceAxis =
        physx::PxAbs(worldAxis.x) < 0.8f
            ? physx::PxVec3(1.0f, 0.0f, 0.0f)
            : physx::PxVec3(0.0f, 1.0f, 0.0f);
    physx::PxVec3 swingJ0 = worldAxis.cross(referenceAxis);
    if (swingJ0.normalize() <= 1e-6f)
      return;
    physx::PxVec3 swingJ1 = worldAxis.cross(swingJ0);
    if (swingJ1.normalize() <= 1e-6f)
      return;

    const physx::PxVec3 response[3] = {
        body.invInertiaWorld.transform(motorJ),
        body.invInertiaWorld.transform(swingJ0),
        body.invInertiaWorld.transform(swingJ1)};
    const physx::PxVec3 jacobian[3] = {
        motorJ, swingJ0, swingJ1};
    const physx::PxMat33 responseMatrix(
        physx::PxVec3(jacobian[0].dot(response[0]),
                      jacobian[1].dot(response[0]),
                      jacobian[2].dot(response[0])),
        physx::PxVec3(jacobian[0].dot(response[1]),
                      jacobian[1].dot(response[1]),
                      jacobian[2].dot(response[1])),
        physx::PxVec3(jacobian[0].dot(response[2]),
                      jacobian[1].dot(response[2]),
                      jacobian[2].dot(response[2])));
    const physx::PxReal determinant =
        responseMatrix.getDeterminant();
    if (!physx::PxIsFinite(determinant) ||
        physx::PxAbs(determinant) <= 1e-12f)
      return;

    const physx::PxVec3 rhs(
        joint.motorTargetVelocity -
            motorJ.dot(body.angularVelocity),
        -swingJ0.dot(body.angularVelocity),
        -swingJ1.dot(body.angularVelocity));
    physx::PxVec3 impulse =
        responseMatrix.getInverse().transform(rhs);
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal clampedMotorImpulse =
        physx::PxClamp(impulse.x, -maximumMotorImpulse,
                       maximumMotorImpulse);
    if (clampedMotorImpulse != impulse.x) {
      const physx::PxReal k11 =
          jacobian[1].dot(response[1]);
      const physx::PxReal k12 =
          jacobian[1].dot(response[2]);
      const physx::PxReal k22 =
          jacobian[2].dot(response[2]);
      const physx::PxReal swingDeterminant =
          k11 * k22 - k12 * k12;
      if (!physx::PxIsFinite(swingDeterminant) ||
          physx::PxAbs(swingDeterminant) <= 1e-12f)
        return;
      const physx::PxReal swingRhs0 =
          rhs.y - jacobian[1].dot(response[0]) *
                      clampedMotorImpulse;
      const physx::PxReal swingRhs1 =
          rhs.z - jacobian[2].dot(response[0]) *
                      clampedMotorImpulse;
      impulse.y =
          (swingRhs0 * k22 - swingRhs1 * k12) /
          swingDeterminant;
      impulse.z =
          (k11 * swingRhs1 - k12 * swingRhs0) /
          swingDeterminant;
      impulse.x = clampedMotorImpulse;
    }
    if (!impulse.isFinite())
      return;

    const physx::PxVec3 candidate =
        body.angularVelocity + response[0] * impulse.x +
        response[1] * impulse.y + response[2] * impulse.z;
    if (!candidate.isFinite() ||
        (body.maxAngularVelocitySq > 0.0f &&
         candidate.magnitudeSquared() >
             body.maxAngularVelocitySq))
      return;
    body.angularVelocity = candidate;
    return;
  }

  const bool coupledDynamicPairOffPrincipalResponse =
      dynamicA && dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin && physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      joint.anchorA.magnitudeSquared() <= 1e-8f &&
      joint.anchorB.magnitudeSquared() <= 1e-8f &&
      (responseA.cross(worldAxis).magnitude() >
           physx::PxMax(responseA.magnitude(), 1e-8f) * 1e-4f ||
       responseB.cross(worldAxis).magnitude() >
           physx::PxMax(responseB.magnitude(), 1e-8f) * 1e-4f);
  if (coupledDynamicPairOffPrincipalResponse) {
    AvbdSolverBody &bodyA = bodies[endpoint[0]];
    AvbdSolverBody &bodyB = bodies[endpoint[1]];
    physx::PxVec3 referenceAxis =
        physx::PxAbs(worldAxis.x) < 0.8f
            ? physx::PxVec3(1.0f, 0.0f, 0.0f)
            : physx::PxVec3(0.0f, 1.0f, 0.0f);
    physx::PxVec3 swingJ0 = worldAxis.cross(referenceAxis);
    if (swingJ0.normalize() <= 1e-6f)
      return;
    physx::PxVec3 swingJ1 = worldAxis.cross(swingJ0);
    if (swingJ1.normalize() <= 1e-6f)
      return;

    const physx::PxVec3 jacobian[3] = {
        worldAxis, swingJ0, swingJ1};
    physx::PxVec3 responseA3[3];
    physx::PxVec3 responseB3[3];
    physx::PxMat33 responseMatrix(physx::PxZero);
    for (physx::PxU32 column = 0; column < 3; ++column) {
      responseA3[column] =
          bodyA.invInertiaWorld.transform(jacobian[column]);
      responseB3[column] =
          bodyB.invInertiaWorld.transform(jacobian[column]);
      responseMatrix[column] = physx::PxVec3(
          jacobian[0].dot(responseA3[column] +
                          responseB3[column]),
          jacobian[1].dot(responseA3[column] +
                          responseB3[column]),
          jacobian[2].dot(responseA3[column] +
                          responseB3[column]));
    }
    const physx::PxReal determinant =
        responseMatrix.getDeterminant();
    if (!physx::PxIsFinite(determinant) ||
        physx::PxAbs(determinant) <= 1e-12f)
      return;

    const physx::PxVec3 relativeAngular =
        bodyB.angularVelocity - bodyA.angularVelocity;
    const physx::PxVec3 rhs(
        joint.motorTargetVelocity -
            jacobian[0].dot(relativeAngular),
        -jacobian[1].dot(relativeAngular),
        -jacobian[2].dot(relativeAngular));
    physx::PxVec3 impulse =
        responseMatrix.getInverse().transform(rhs);
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal clampedMotorImpulse =
        physx::PxClamp(impulse.x, -maximumMotorImpulse,
                       maximumMotorImpulse);
    if (clampedMotorImpulse != impulse.x) {
      const physx::PxReal k11 =
          jacobian[1].dot(responseA3[1] + responseB3[1]);
      const physx::PxReal k12 =
          jacobian[1].dot(responseA3[2] + responseB3[2]);
      const physx::PxReal k22 =
          jacobian[2].dot(responseA3[2] + responseB3[2]);
      const physx::PxReal swingDeterminant =
          k11 * k22 - k12 * k12;
      if (!physx::PxIsFinite(swingDeterminant) ||
          physx::PxAbs(swingDeterminant) <= 1e-12f)
        return;
      const physx::PxReal swingRhs0 =
          rhs.y - jacobian[1].dot(responseA3[0] +
                                  responseB3[0]) *
                      clampedMotorImpulse;
      const physx::PxReal swingRhs1 =
          rhs.z - jacobian[2].dot(responseA3[0] +
                                  responseB3[0]) *
                      clampedMotorImpulse;
      impulse.y =
          (swingRhs0 * k22 - swingRhs1 * k12) /
          swingDeterminant;
      impulse.z =
          (k11 * swingRhs1 - k12 * swingRhs0) /
          swingDeterminant;
      impulse.x = clampedMotorImpulse;
    }
    if (!impulse.isFinite())
      return;

    physx::PxVec3 candidateA = bodyA.angularVelocity;
    physx::PxVec3 candidateB = bodyB.angularVelocity;
    for (physx::PxU32 row = 0; row < 3; ++row) {
      candidateA -= responseA3[row] * impulse[row];
      candidateB += responseB3[row] * impulse[row];
    }
    if (conserveDynamicPairAngularMomentumVector) {
      const physx::PxMat33 inertiaA =
          bodyA.invInertiaWorld.getInverse();
      const physx::PxMat33 inertiaB =
          bodyB.invInertiaWorld.getInverse();
      const physx::PxMat33 inertiaSum(
          inertiaA.column0 + inertiaB.column0,
          inertiaA.column1 + inertiaB.column1,
          inertiaA.column2 + inertiaB.column2);
      const physx::PxReal inertiaSumDeterminant =
          inertiaSum.getDeterminant();
      const physx::PxVec3 currentAngularMomentum =
          inertiaA.transform(candidateA) +
          inertiaB.transform(candidateB);
      if (!physx::PxIsFinite(inertiaSumDeterminant) ||
          physx::PxAbs(inertiaSumDeterminant) <= 1e-12f ||
          !currentAngularMomentum.isFinite())
        return;
      const physx::PxVec3 commonAngularVelocity =
          inertiaSum.getInverse().transform(
              expectedAngularMomentumVector -
              currentAngularMomentum);
      if (!commonAngularVelocity.isFinite())
        return;
      candidateA += commonAngularVelocity;
      candidateB += commonAngularVelocity;
    }
    if (!candidateA.isFinite() || !candidateB.isFinite() ||
        (bodyA.maxAngularVelocitySq > 0.0f &&
         candidateA.magnitudeSquared() >
             bodyA.maxAngularVelocitySq) ||
        (bodyB.maxAngularVelocitySq > 0.0f &&
         candidateB.magnitudeSquared() >
             bodyB.maxAngularVelocitySq))
      return;
    bodyA.angularVelocity = candidateA;
    bodyB.angularVelocity = candidateB;
    return;
  }

  const physx::PxVec3 localAxisA =
      joint.localFrameA.rotate(
          physx::PxVec3(1.0f, 0.0f, 0.0f));
  const physx::PxVec3 localAxisB =
      joint.localFrameB.rotate(
          physx::PxVec3(1.0f, 0.0f, 0.0f));
  const bool coupledDynamicPairOffCenterResponse =
      dynamicA && dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin && physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      (joint.anchorA.cross(localAxisA).magnitudeSquared() > 1e-8f ||
       joint.anchorB.cross(localAxisB).magnitudeSquared() > 1e-8f);
  if (coupledDynamicPairOffCenterResponse) {
    AvbdSolverBody &bodyA = bodies[endpoint[0]];
    AvbdSolverBody &bodyB = bodies[endpoint[1]];
    const physx::PxVec3 rA =
        bodyA.rotation.rotate(joint.anchorA);
    const physx::PxVec3 rB =
        bodyB.rotation.rotate(joint.anchorB);
    physx::PxVec3 referenceAxis =
        physx::PxAbs(worldAxis.x) < 0.8f
            ? physx::PxVec3(1.0f, 0.0f, 0.0f)
            : physx::PxVec3(0.0f, 1.0f, 0.0f);
    physx::PxVec3 swingAxis0 =
        worldAxis.cross(referenceAxis);
    if (swingAxis0.normalize() <= 1e-6f)
      return;
    physx::PxVec3 swingAxis1 =
        worldAxis.cross(swingAxis0);
    if (swingAxis1.normalize() <= 1e-6f)
      return;
    const physx::PxVec3 worldAxes[3] = {
        physx::PxVec3(1.0f, 0.0f, 0.0f),
        physx::PxVec3(0.0f, 1.0f, 0.0f),
        physx::PxVec3(0.0f, 0.0f, 1.0f)};
    const physx::PxVec3 angularAxes[3] = {
        worldAxis, swingAxis0, swingAxis1};
    AvbdVec6 jacobianA[6];
    AvbdVec6 jacobianB[6];
    for (physx::PxU32 row = 0; row < 3; ++row) {
      jacobianA[row] =
          AvbdVec6(worldAxes[row], rA.cross(worldAxes[row]));
      jacobianB[row] =
          AvbdVec6(-worldAxes[row], -rB.cross(worldAxes[row]));
      jacobianA[3 + row] =
          AvbdVec6(physx::PxVec3(0.0f), angularAxes[row]);
      jacobianB[3 + row] =
          AvbdVec6(physx::PxVec3(0.0f), -angularAxes[row]);
    }

    const AvbdVec6 velocityA(
        bodyA.linearVelocity, bodyA.angularVelocity);
    const AvbdVec6 velocityB(
        bodyB.linearVelocity, bodyB.angularVelocity);
    physx::PxReal rhs[6] = {};
    physx::PxReal responseMatrix[6][6] = {};
    for (physx::PxU32 row = 0; row < 6; ++row) {
      const physx::PxReal current =
          jacobianA[row].dot(velocityA) +
          jacobianB[row].dot(velocityB);
      const physx::PxReal target =
          row == 3 ? -joint.motorTargetVelocity : 0.0f;
      rhs[row] = target - current;
      for (physx::PxU32 column = 0; column < 6; ++column) {
        const AvbdVec6 responseA6(
            jacobianA[column].linear * bodyA.invMass,
            bodyA.invInertiaWorld *
                jacobianA[column].angular);
        const AvbdVec6 responseB6(
            jacobianB[column].linear * bodyB.invMass,
            bodyB.invInertiaWorld *
                jacobianB[column].angular);
        responseMatrix[row][column] =
            jacobianA[row].dot(responseA6) +
            jacobianB[row].dot(responseB6);
  }
}
    physx::PxReal impulse[6] = {};
    if (!solveNativeMotorDense6(
            responseMatrix, rhs, false, 0.0f, impulse))
      return;
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal clampedMotorImpulse =
        physx::PxClamp(impulse[3], -maximumMotorImpulse,
                       maximumMotorImpulse);
    if (clampedMotorImpulse != impulse[3] &&
        !solveNativeMotorDense6(
            responseMatrix, rhs, true,
            clampedMotorImpulse, impulse))
      return;

    AvbdVec6 bodyImpulseA;
    AvbdVec6 bodyImpulseB;
    for (physx::PxU32 row = 0; row < 6; ++row) {
      bodyImpulseA.linear +=
          jacobianA[row].linear * impulse[row];
      bodyImpulseA.angular +=
          jacobianA[row].angular * impulse[row];
      bodyImpulseB.linear +=
          jacobianB[row].linear * impulse[row];
      bodyImpulseB.angular +=
          jacobianB[row].angular * impulse[row];
    }
    physx::PxVec3 candidateLinearA =
        bodyA.linearVelocity +
        bodyImpulseA.linear * bodyA.invMass;
    physx::PxVec3 candidateAngularA =
        bodyA.angularVelocity +
        bodyA.invInertiaWorld * bodyImpulseA.angular;
    physx::PxVec3 candidateLinearB =
        bodyB.linearVelocity +
        bodyImpulseB.linear * bodyB.invMass;
    physx::PxVec3 candidateAngularB =
        bodyB.angularVelocity +
        bodyB.invInertiaWorld * bodyImpulseB.angular;
    if (conserveDynamicPairSpatialMomentum &&
        conserveDynamicPairLinearMomentum) {
      const physx::PxReal massA = 1.0f / bodyA.invMass;
      const physx::PxReal massB = 1.0f / bodyB.invMass;
      const physx::PxMat33 inertiaA =
          bodyA.invInertiaWorld.getInverse();
      const physx::PxMat33 inertiaB =
          bodyB.invInertiaWorld.getInverse();
      const physx::PxVec3 currentLinearMomentum =
          candidateLinearA * massA + candidateLinearB * massB;
      const physx::PxVec3 currentAngularMomentum =
          bodyA.position.cross(candidateLinearA * massA) +
          inertiaA.transform(candidateAngularA) +
          bodyB.position.cross(candidateLinearB * massB) +
          inertiaB.transform(candidateAngularB);
      if (!currentLinearMomentum.isFinite() ||
          !currentAngularMomentum.isFinite())
        return;
      physx::PxReal spatialResponse[6][6] = {};
      const physx::PxVec3 basis[3] = {
          physx::PxVec3(1.0f, 0.0f, 0.0f),
          physx::PxVec3(0.0f, 1.0f, 0.0f),
          physx::PxVec3(0.0f, 0.0f, 1.0f)};
      for (physx::PxU32 column = 0; column < 6; ++column) {
        const physx::PxVec3 commonLinear =
            column < 3 ? basis[column] : physx::PxVec3(0.0f);
        const physx::PxVec3 commonAngular =
            column < 3 ? physx::PxVec3(0.0f)
                       : basis[column - 3];
        const physx::PxVec3 deltaLinearA =
            commonLinear +
            commonAngular.cross(bodyA.position);
        const physx::PxVec3 deltaLinearB =
            commonLinear +
            commonAngular.cross(bodyB.position);
        const physx::PxVec3 deltaLinearMomentum =
            deltaLinearA * massA + deltaLinearB * massB;
        const physx::PxVec3 deltaAngularMomentum =
            bodyA.position.cross(deltaLinearA * massA) +
            inertiaA.transform(commonAngular) +
            bodyB.position.cross(deltaLinearB * massB) +
            inertiaB.transform(commonAngular);
        for (physx::PxU32 row = 0; row < 3; ++row) {
          spatialResponse[row][column] =
              deltaLinearMomentum[row];
          spatialResponse[3 + row][column] =
              deltaAngularMomentum[row];
        }
      }
      physx::PxReal spatialRhs[6] = {};
      const physx::PxVec3 linearMomentumDelta =
          expectedLinearMomentum - currentLinearMomentum;
      const physx::PxVec3 angularMomentumDelta =
          expectedSpatialAngularMomentum -
          currentAngularMomentum;
      for (physx::PxU32 row = 0; row < 3; ++row) {
        spatialRhs[row] = linearMomentumDelta[row];
        spatialRhs[3 + row] = angularMomentumDelta[row];
      }
      physx::PxReal spatialCorrection[6] = {};
      if (!solveNativeMotorDense6(
              spatialResponse, spatialRhs, false, 0.0f,
              spatialCorrection))
        return;
      const physx::PxVec3 commonLinearVelocity(
          spatialCorrection[0], spatialCorrection[1],
          spatialCorrection[2]);
      const physx::PxVec3 commonAngularVelocity(
          spatialCorrection[3], spatialCorrection[4],
          spatialCorrection[5]);
      candidateLinearA +=
          commonLinearVelocity +
          commonAngularVelocity.cross(bodyA.position);
      candidateLinearB +=
          commonLinearVelocity +
          commonAngularVelocity.cross(bodyB.position);
      candidateAngularA += commonAngularVelocity;
      candidateAngularB += commonAngularVelocity;
    } else if (conserveDynamicPairLinearMomentum) {
      const physx::PxReal massA = 1.0f / bodyA.invMass;
      const physx::PxReal massB = 1.0f / bodyB.invMass;
      const physx::PxReal totalMass = massA + massB;
      const physx::PxVec3 currentLinearMomentum =
          candidateLinearA * massA + candidateLinearB * massB;
      if (!physx::PxIsFinite(totalMass) ||
          totalMass <= 1e-10f ||
          !currentLinearMomentum.isFinite())
        return;
      const physx::PxVec3 correction =
          (expectedLinearMomentum - currentLinearMomentum) /
          totalMass;
      if (!correction.isFinite())
        return;
      candidateLinearA += correction;
      candidateLinearB += correction;
    }
    if (!candidateLinearA.isFinite() ||
        !candidateAngularA.isFinite() ||
        !candidateLinearB.isFinite() ||
        !candidateAngularB.isFinite() ||
        (bodyA.maxLinearVelocitySq > 0.0f &&
         candidateLinearA.magnitudeSquared() >
             bodyA.maxLinearVelocitySq) ||
        (bodyA.maxAngularVelocitySq > 0.0f &&
         candidateAngularA.magnitudeSquared() >
             bodyA.maxAngularVelocitySq) ||
        (bodyB.maxLinearVelocitySq > 0.0f &&
         candidateLinearB.magnitudeSquared() >
             bodyB.maxLinearVelocitySq) ||
        (bodyB.maxAngularVelocitySq > 0.0f &&
         candidateAngularB.magnitudeSquared() >
             bodyB.maxAngularVelocitySq))
      return;
    bodyA.linearVelocity = candidateLinearA;
    bodyA.angularVelocity = candidateAngularA;
    bodyB.linearVelocity = candidateLinearB;
    bodyB.angularVelocity = candidateAngularB;
    return;
  }

  const physx::PxReal motorBaseVelocity =
      freeSpin && useSolveStartRelativeVelocity
          ? solveStartRelativeVelocity
          : motorVelocity;
  const physx::PxReal requiredMotorImpulse =
      (joint.motorTargetVelocity - motorBaseVelocity) / unitResponse;
  const physx::PxReal maximumImpulse = joint.motorMaxForce * dt;
  physx::PxReal minimumMotorImpulse = -maximumImpulse;
  physx::PxReal maximumMotorImpulse = maximumImpulse;
  // Match PxRevoluteJoint::eDRIVE_FREESPIN row bounds exactly. Positive
  // targets may only receive positive drive impulse, negative targets only
  // negative drive impulse, and a zero target remains bilateral.
  if (freeSpin && joint.motorTargetVelocity > 0.0f)
    minimumMotorImpulse = 0.0f;
  else if (freeSpin && joint.motorTargetVelocity < 0.0f)
    maximumMotorImpulse = 0.0f;
  const physx::PxReal motorImpulse = physx::PxClamp(
      requiredMotorImpulse, minimumMotorImpulse, maximumMotorImpulse);
  const physx::PxReal motorOwnedVelocity =
      motorBaseVelocity + unitResponse * motorImpulse;
  // The position path reconstructs velocity from its corrected pose. For the
  // strict free-spin owner, restore the solve-entry hinge derivative before
  // applying the authored one-sided motor impulse; otherwise a pose-derived
  // zero velocity incorrectly brakes a super-target body.
  physx::PxReal impulse =
      (motorOwnedVelocity - motorVelocity) / unitResponse;

  if (joint.getAngularMotion(0) == 1u) {
    const physx::PxQuat rotationA =
        dynamicA ? bodies[endpoint[0]].rotation
                 : physx::PxQuat(physx::PxIdentity);
    const physx::PxQuat rotationB =
        dynamicB ? bodies[endpoint[1]].rotation
                 : physx::PxQuat(physx::PxIdentity);
    const physx::PxReal angularError =
        joint.computeAngularError(rotationA, rotationB, 0);
    const physx::PxReal limitSpan =
        joint.angularLimitUpper.x - joint.angularLimitLower.x;
    const physx::PxReal activeTolerance =
        physx::PxMax(1e-5f, physx::PxAbs(limitSpan) * 1e-5f);
    const physx::PxReal motorOnlyVelocity =
        motorVelocity + unitResponse * impulse;
    // computeAngularError uses frameA * conjugate(frameB), so its derivative
    // is axis dot (wA - wB), the negative of the public motor velocity row.
    // At the internal lower bound the admissible public velocity is <= 0;
    // at the internal upper bound it is >= 0. First apply the bounded motor,
    // then add only the unilateral limit impulse needed to close an outward
    // derivative. A one-step speculative test activates the row before the
    // bounded motor can cross it. Position remains exclusively owned by the
    // hard AL row.
    const physx::PxReal predictedAngularError =
        angularError - motorOnlyVelocity * dt;
    const bool atLower =
        angularError <= joint.angularLimitLower.x + activeTolerance ||
        predictedAngularError <= joint.angularLimitLower.x;
    const bool atUpper =
        angularError >= joint.angularLimitUpper.x - activeTolerance ||
        predictedAngularError >= joint.angularLimitUpper.x;
    if ((atLower && motorOnlyVelocity > 0.0f) ||
        (atUpper && motorOnlyVelocity < 0.0f))
      impulse = -motorVelocity / unitResponse;
  }
  if (!physx::PxIsFinite(impulse))
    return;

  physx::PxVec3 candidateA =
      dynamicA
          ? bodies[endpoint[0]].angularVelocity - responseA * impulse
          : physx::PxVec3(0.0f);
  physx::PxVec3 candidateB =
      dynamicB
          ? bodies[endpoint[1]].angularVelocity +
                responseB * (driveRatio * impulse)
          : physx::PxVec3(0.0f);

  if (conserveDynamicPairAngularMomentum && dynamicA && dynamicB) {
    const physx::PxMat33 inertiaA =
        bodies[endpoint[0]].invInertiaWorld.getInverse();
    const physx::PxMat33 inertiaB =
        bodies[endpoint[1]].invInertiaWorld.getInverse();
    const physx::PxReal inertiaOnAxisA =
        worldAxis.dot(inertiaA.transform(worldAxis));
    const physx::PxReal inertiaOnAxisB =
        worldAxis.dot(inertiaB.transform(worldAxis));
    const physx::PxReal inertiaSum =
        driveRatio * driveRatio * inertiaOnAxisA +
        inertiaOnAxisB;
    const physx::PxReal currentAngularMomentumOnAxis =
        worldAxis.dot(
            inertiaA.transform(candidateA) * driveRatio +
            inertiaB.transform(candidateB));
    if (!physx::PxIsFinite(inertiaSum) ||
        !physx::PxIsFinite(currentAngularMomentumOnAxis) ||
        inertiaSum <= 1e-10f)
      return;
    // The hard joint position path may leave a small common angular mode.
    // Restore the solve-start conserved momentum without changing the
    // already-projected relative motor velocity.
    const physx::PxReal commonAngularVelocity =
        (expectedAngularMomentumOnAxis -
         currentAngularMomentumOnAxis) /
        inertiaSum;
    if (!physx::PxIsFinite(commonAngularVelocity))
      return;
    candidateA +=
        worldAxis * (driveRatio * commonAngularVelocity);
    candidateB += worldAxis * commonAngularVelocity;
  }
  if ((dynamicA &&
       (!candidateA.isFinite() ||
        (bodies[endpoint[0]].maxAngularVelocitySq > 0.0f &&
         candidateA.magnitudeSquared() >
             bodies[endpoint[0]].maxAngularVelocitySq))) ||
      (dynamicB &&
       (!candidateB.isFinite() ||
        (bodies[endpoint[1]].maxAngularVelocitySq > 0.0f &&
         candidateB.magnitudeSquared() >
             bodies[endpoint[1]].maxAngularVelocitySq))))
    return;

  if (dynamicA)
    bodies[endpoint[0]].angularVelocity = candidateA;
  if (dynamicB)
    bodies[endpoint[1]].angularVelocity = candidateB;
}

} // namespace Dy
} // namespace physx
