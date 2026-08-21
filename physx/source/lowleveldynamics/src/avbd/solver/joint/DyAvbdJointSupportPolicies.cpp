// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointSupportPolicies.h"

namespace physx {
namespace Dy {

bool areFrictionlessBodyVsStaticContactsSupported(
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 numBodies) {
  if (numContacts != 0 && !contacts)
    return false;
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        contact.friction > 0.0f || contact.staticFriction > 0.0f ||
        hasDeformableStaticAnchor(contact))
      return false;
  }
  return true;
}

bool areTorqueFreeBodyVsStaticContactsSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts) {
  if (!bodies ||
      !areFrictionlessBodyVsStaticContactsSupported(
          contacts, numContacts, numBodies))
    return false;
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const bool dynamicA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicA ? contact.header.bodyIndexA : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies)
      return false;
    const physx::PxVec3 localPoint =
        dynamicA ? contact.contactPointA : contact.contactPointB;
    const physx::PxVec3 momentArm =
        bodies[bodyIndex].rotation.rotate(localPoint);
    if (momentArm.cross(contact.contactNormal).magnitudeSquared() > 1e-10f)
      return false;
  }
  return true;
}

bool areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity) {
  if (!bodies || numBodies != 2u || !contacts ||
      numContacts < 2u || numContacts > 8u ||
      !gravity.isFinite() || gravity.magnitudeSquared() <= 1e-12f)
    return false;

  const physx::PxVec3 up = -gravity.getNormalized();
  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  physx::PxVec3 referenceNormal(0.0f);
  bool haveReferenceNormal = false;
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contact) ||
        hasKinematicShellAnchor(contact) ||
        (contact.friction <= 0.0f &&
         contact.staticFriction <= 0.0f) ||
        !PxIsFinite(contact.friction) ||
        !PxIsFinite(contact.staticFriction) ||
        !PxIsFinite(contact.restitution) ||
        physx::PxAbs(contact.restitution) > 1e-6f ||
        !contact.targetVelocity.isFinite() ||
        !contact.contactNormal.isFinite() ||
        !contact.tangent0.isFinite() ||
        !contact.tangent1.isFinite() ||
        physx::PxAbs(contact.tangent0.magnitudeSquared() - 1.0f) >
            1e-4f ||
        physx::PxAbs(contact.tangent1.magnitudeSquared() - 1.0f) >
            1e-4f ||
        physx::PxAbs(contact.tangent0.dot(contact.tangent1)) > 1e-4f ||
        contact.targetVelocity.magnitudeSquared() > 1e-12f)
      return false;

    const bool dynamicIsA =
        contact.header.bodyIndexA < numBodies;
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

    physx::PxVec3 dynamicNormal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    if (dynamicNormal.normalize() <= 1e-6f ||
        // A stationary upward-facing slope is still an ordinary persistent
        // rigid support. The complete position owner already uses the full
        // two-tangent basis and computes its Coulomb budget from
        // |gravity dot normal| below, so gravity alignment is not a solver
        // requirement. Retain only a numerical non-degeneracy boundary that
        // excludes vertical/downward surfaces from this strict authority.
        up.dot(dynamicNormal) <= 1e-6f)
      return false;
    if (!haveReferenceNormal) {
      referenceNormal = dynamicNormal;
      haveReferenceNormal = true;
    } else if (referenceNormal.dot(dynamicNormal) < 0.9999f) {
      return false;
    }

    const physx::PxVec3 localPoint =
        dynamicIsA ? contact.contactPointA
                   : contact.contactPointB;
    if (!localPoint.isFinite())
      return false;
    const physx::PxVec3 contactArm =
        bodies[bodyIndex].rotation.rotate(localPoint);
    if (contactArm.cross(dynamicNormal).magnitudeSquared() >
        1e-10f)
      return false;
    const physx::PxVec3 pointVelocity =
        bodies[bodyIndex].linearVelocity +
        bodies[bodyIndex].angularVelocity.cross(contactArm);
    if (pointVelocity.dot(dynamicNormal) < -0.25f)
      return false;
    contactsPerBody[bodyIndex]++;
  }
  return contactsPerBody[0] != 0u && contactsPerBody[1] != 0u;
}

bool isCoupledLinearDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || !d6Joints || numD6 != 1 ||
      (numContacts != 0 && !contacts) || numGear != 0 ||
      numSoftParticles != 0 || numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  if (joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexA].lockFlags != 0 ||
      bodies[joint.header.bodyIndexB].lockFlags != 0 ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      joint.driveFlags != 0x1u ||
      (joint.driveAccelerationFlags != 0 &&
       joint.driveAccelerationFlags != 0x1u) ||
      joint.linearStiffness.x != 0.0f || joint.linearDamping.x <= 0.0f ||
      joint.getLinearMotion(0) != 2 || joint.getLinearMotion(1) != 2 ||
      joint.getLinearMotion(2) != 2 || joint.getAngularMotion(0) != 2 ||
      joint.getAngularMotion(1) != 2 || joint.getAngularMotion(2) != 2 ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0)
    return false;

  return areFrictionlessBodyVsStaticContactsSupported(
      contacts, numContacts, numBodies);
}

bool isLinearPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      joint.driveFlags != 0x1u || joint.driveAccelerationFlags != 0 ||
      joint.linearStiffness.x <= 0.0f || joint.linearDamping.x <= 0.0f ||
      joint.getLinearMotion(0) != 2 || joint.getLinearMotion(1) != 0 ||
      joint.getLinearMotion(2) != 0 || joint.angularMotion != 0 ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveLinearPosition.isFinite() ||
      !joint.driveLinearVelocity.isFinite() ||
      !(joint.driveLinearForce.x > 0.0f) ||
      !PxIsFinite(joint.driveLinearForce.x))
    return false;
  return true;
}

static bool isIdentityJointFrame(const physx::PxQuat &frame) {
  if (!frame.isFinite())
    return false;
  const physx::PxReal magnitudeSquared = frame.magnitudeSquared();
  if (!(magnitudeSquared > 1e-8f) || !PxIsFinite(magnitudeSquared))
    return false;
  const physx::PxReal invMagnitude =
      1.0f / physx::PxSqrt(magnitudeSquared);
  return physx::PxAbs(frame.x * invMagnitude) <= 1e-6f &&
         physx::PxAbs(frame.y * invMagnitude) <= 1e-6f &&
         physx::PxAbs(frame.z * invMagnitude) <= 1e-6f &&
         physx::PxAbs(physx::PxAbs(frame.w * invMagnitude) - 1.0f) <=
             1e-6f;
}

static bool isStrictAsymmetricJointXZPositionOwner(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint,
    bool frictionlessContacts) {
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (!frictionlessContacts ||
      bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
      physx::PxAbs(bodies[bodyA].invMass -
                    bodies[bodyB].invMass) >
          1e-6f * physx::PxMax(bodies[bodyA].invMass,
                              bodies[bodyB].invMass) ||
      !isIdentityJointFrame(joint.localFrameA) ||
      !isIdentityJointFrame(joint.localFrameB) ||
      physx::PxAbs(joint.driveLinearVelocity.x) > 1e-6f ||
      joint.driveLinearForce.x != PX_MAX_F32)
    return false;

  const bool centeredA =
      joint.anchorA.magnitudeSquared() <= 1e-12f;
  const bool centeredB =
      joint.anchorB.magnitudeSquared() <= 1e-12f;
  const bool pureXA =
      physx::PxAbs(joint.anchorA.x) > 1e-6f &&
      physx::PxAbs(joint.anchorA.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorA.z) <= 1e-6f;
  const bool pureXB =
      physx::PxAbs(joint.anchorB.x) > 1e-6f &&
      physx::PxAbs(joint.anchorB.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorB.z) <= 1e-6f;
  const bool pureZA =
      physx::PxAbs(joint.anchorA.x) <= 1e-6f &&
      physx::PxAbs(joint.anchorA.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorA.z) > 1e-6f;
  const bool pureZB =
      physx::PxAbs(joint.anchorB.x) <= 1e-6f &&
      physx::PxAbs(joint.anchorB.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorB.z) > 1e-6f;
  const bool unequalPureZPair =
      pureZA && pureZB &&
      (joint.anchorA - joint.anchorB).magnitudeSquared() > 1e-12f;
  return (centeredA && (pureXB || pureZB)) ||
         ((pureXA || pureZA) && centeredB) ||
         unequalPureZPair;
}

bool isCoupledLinearPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const physx::PxVec3 &gravity, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || !contacts || numContacts == 0 ||
      !d6Joints || numD6 != 1 || numGear != 0 ||
      numSoftParticles != 0 || numSoftBodies != 0 ||
      numSoftContacts != 0)
    return false;
  const bool frictionlessContacts =
      areTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts);
  const bool frictionalContacts =
      areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts, gravity);
  if (!frictionlessContacts && !frictionalContacts)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool equalAnchors =
      (joint.anchorA - joint.anchorB).magnitudeSquared() <= 1e-12f;
  const bool strictAsymmetricJointXZ =
      isStrictAsymmetricJointXZPositionOwner(
          bodies, numBodies, joint, frictionlessContacts);
  // eOUTPUT_FORCE is observational only. The complete position owner remains
  // unchanged, and post-finalization writeback controls whether its physical
  // drive force contributes to the public reaction.
  if (joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexA].lockFlags != 0 ||
      bodies[joint.header.bodyIndexB].lockFlags != 0 ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexA].angularDampingBody != 0.0f ||
      bodies[joint.header.bodyIndexB].angularDampingBody != 0.0f ||
      joint.driveFlags != 0x1u ||
      joint.driveAccelerationFlags != 0 ||
      joint.linearStiffness.x <= 0.0f ||
      joint.linearDamping.x <= 0.0f ||
      joint.linearStiffness.y != 0.0f ||
      joint.linearStiffness.z != 0.0f ||
      joint.linearDamping.y != 0.0f ||
      joint.linearDamping.z != 0.0f ||
      joint.getLinearMotion(0) != 2 ||
      joint.getLinearMotion(1) != 0 ||
      joint.getLinearMotion(2) != 0 ||
      joint.angularMotion != 0 ||
      !joint.anchorA.isFinite() || !joint.anchorB.isFinite() ||
      (!equalAnchors && !strictAsymmetricJointXZ) ||
      joint.motorEnabled != 0 || joint.coneAngleLimit > 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveLinearPosition.isFinite() ||
      !joint.driveLinearVelocity.isFinite() ||
      physx::PxAbs(joint.driveLinearPosition.y) > 1e-6f ||
      physx::PxAbs(joint.driveLinearPosition.z) > 1e-6f ||
      physx::PxAbs(joint.driveLinearVelocity.y) > 1e-6f ||
      physx::PxAbs(joint.driveLinearVelocity.z) > 1e-6f ||
      !(joint.driveLinearForce.x > 0.0f) ||
      !PxIsFinite(joint.driveLinearForce.x))
    return false;

  physx::PxQuat frameA = joint.localFrameA;
  const physx::PxReal frameMagnitude = frameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  frameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  const physx::PxVec3 axis =
      (bodies[joint.header.bodyIndexA].rotation * frameA)
          .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  bool bodyHasContact[2] = {false, false};
  physx::PxVec3 supportNormal(0.0f);
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies ||
        physx::PxAbs(contact.contactNormal.dot(axis)) > 1e-5f)
      return false;
    bodyHasContact[bodyIndex] = true;
    physx::PxVec3 normal = contact.contactNormal;
    const physx::PxReal normalLength = normal.magnitude();
    if (!(normalLength > 1e-6f) || !PxIsFinite(normalLength))
      return false;
    normal *= 1.0f / normalLength;
    if (i == 0)
      supportNormal = normal;
    else if (physx::PxAbs(normal.dot(supportNormal)) < 0.9999f)
      return false;
  }
  return bodyHasContact[0] && bodyHasContact[1];
}

bool isAngularAxisVelocityDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  const physx::PxU32 driveIndex = joint.driveFlags == (1u << 3)
                                         ? 0u
                                         : (joint.driveFlags == (1u << 4)
                                                ? 1u
                                                : (joint.driveFlags == (1u << 5)
                                                       ? 2u
                                                       : PX_MAX_U32));
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      driveIndex == PX_MAX_U32 ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0)
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.angularStiffness[driveIndex] != 0.0f ||
      joint.angularDamping[driveIndex] <= 0.0f ||
      joint.linearMotion != 0x2au || joint.angularMotion != 0x2au ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularVelocity.isFinite() ||
      !(joint.driveAngularForce[driveIndex] > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce[driveIndex]))
    return false;
  return true;
}

bool isAngularAxisPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  const physx::PxU32 driveIndex =
      joint.driveFlags == (1u << 3)
          ? 0u
          : (joint.driveFlags == (1u << 4)
                 ? 1u
                 : (joint.driveFlags == (1u << 5) ? 2u : PX_MAX_U32));
  const physx::PxReal targetMagnitude =
      joint.driveAngularPosition.magnitudeSquared();
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      driveIndex == PX_MAX_U32 ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0)
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.driveOutputForceFlags != 0 || joint.linearMotion != 0 ||
      joint.getAngularMotion(driveIndex) != 2 ||
      joint.getAngularMotion((driveIndex + 1) % 3) != 0 ||
      joint.getAngularMotion((driveIndex + 2) % 3) != 0 ||
      joint.angularStiffness[driveIndex] <= 0.0f ||
      joint.angularDamping[driveIndex] <= 0.0f ||
      joint.angularStiffness[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularStiffness[(driveIndex + 2) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 2) % 3] != 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularPosition.isFinite() ||
      !joint.driveAngularVelocity.isFinite() ||
      physx::PxAbs(targetMagnitude - 1.0f) > 1e-4f ||
      (driveIndex != 0 &&
       physx::PxAbs(joint.driveAngularPosition.x) > 1e-6f) ||
      (driveIndex != 1 &&
       physx::PxAbs(joint.driveAngularPosition.y) > 1e-6f) ||
      (driveIndex != 2 &&
       physx::PxAbs(joint.driveAngularPosition.z) > 1e-6f) ||
      joint.driveAngularVelocity.magnitudeSquared() > 1e-12f ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      !(joint.driveAngularForce[driveIndex] > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce[driveIndex]))
    return false;
  return true;
}

bool isSlerpVelocityDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) == 0 ||
      joint.driveFlags != (1u << 5))
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.angularStiffness.z != 0.0f ||
      joint.angularDamping.z <= 0.0f || joint.linearMotion != 0x2au ||
      joint.angularMotion != 0x2au ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularVelocity.isFinite() ||
      !(joint.driveAngularForce.z > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce.z))
    return false;
  return true;
}

bool isSlerpPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  const physx::PxReal targetMagnitude =
      joint.driveAngularPosition.magnitudeSquared();
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) == 0 ||
      joint.driveFlags != (1u << 5))
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.driveOutputForceFlags != 0 || joint.linearMotion != 0 ||
      joint.angularMotion != 0x2au ||
      joint.angularStiffness.x != 0.0f ||
      joint.angularStiffness.y != 0.0f ||
      joint.angularStiffness.z <= 0.0f ||
      joint.angularDamping.x != 0.0f ||
      joint.angularDamping.y != 0.0f ||
      joint.angularDamping.z <= 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularPosition.isFinite() ||
      !joint.driveAngularVelocity.isFinite() ||
      physx::PxAbs(targetMagnitude - 1.0f) > 1e-4f ||
      joint.driveAngularVelocity.magnitudeSquared() > 1e-12f ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      !(joint.driveAngularForce.z > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce.z))
    return false;
  return true;
}

bool isCoupledAngularPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || !d6Joints || numD6 != 1 ||
      numGear != 0 || numSoftParticles != 0 || numSoftBodies != 0 ||
      numSoftContacts != 0 ||
      !areTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts))
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  if (joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexA].angularDampingBody != 0.0f ||
      bodies[joint.header.bodyIndexB].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.driveOutputForceFlags != 0 || joint.linearMotion != 0 ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      joint.motorEnabled != 0 || joint.coneAngleLimit > 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularPosition.isFinite() ||
      !joint.driveAngularVelocity.isFinite() ||
      physx::PxAbs(joint.driveAngularPosition.magnitudeSquared() - 1.0f) >
          1e-4f ||
      joint.driveAngularVelocity.magnitudeSquared() > 1e-12f)
    return false;

  const bool slerp =
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0;
  if (slerp) {
    return joint.driveFlags == (1u << 5) &&
           joint.angularMotion == 0x2au &&
           joint.angularStiffness.x == 0.0f &&
           joint.angularStiffness.y == 0.0f &&
           joint.angularStiffness.z > 0.0f &&
           joint.angularDamping.x == 0.0f &&
           joint.angularDamping.y == 0.0f &&
           joint.angularDamping.z > 0.0f &&
           joint.driveAngularForce.z > 0.0f &&
           PxIsFinite(joint.driveAngularForce.z);
  }

  const physx::PxU32 driveIndex =
      joint.driveFlags == (1u << 3)
          ? 0u
          : (joint.driveFlags == (1u << 4)
                 ? 1u
                 : (joint.driveFlags == (1u << 5) ? 2u : PX_MAX_U32));
  if (driveIndex == PX_MAX_U32 ||
      joint.getAngularMotion(driveIndex) != 2 ||
      joint.getAngularMotion((driveIndex + 1) % 3) != 0 ||
      joint.getAngularMotion((driveIndex + 2) % 3) != 0 ||
      joint.angularStiffness[driveIndex] <= 0.0f ||
      joint.angularDamping[driveIndex] <= 0.0f ||
      joint.angularStiffness[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularStiffness[(driveIndex + 2) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 2) % 3] != 0.0f ||
      !(joint.driveAngularForce[driveIndex] > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce[driveIndex]))
    return false;
  if ((driveIndex != 0 &&
       physx::PxAbs(joint.driveAngularPosition.x) > 1e-6f) ||
      (driveIndex != 1 &&
       physx::PxAbs(joint.driveAngularPosition.y) > 1e-6f) ||
      (driveIndex != 2 &&
       physx::PxAbs(joint.driveAngularPosition.z) > 1e-6f))
    return false;
  return true;
}

} // namespace Dy
} // namespace physx
