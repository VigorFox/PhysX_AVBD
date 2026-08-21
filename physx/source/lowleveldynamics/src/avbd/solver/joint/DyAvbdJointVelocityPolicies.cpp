// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointVelocityPolicies.h"
#include "avbd/solver/joint/DyAvbdJointDriveMath.h"
#include "PxConstraintDesc.h"

namespace physx {
namespace Dy {

bool isPassiveCenteredGearVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, const AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 2 || !gearJoints || numGear != 1 ||
      numSoftParticles != 0 || numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdGearJointConstraint &gear = gearJoints[0];
  const physx::PxU32 gearA = gear.header.bodyIndexA;
  const physx::PxU32 gearB = gear.header.bodyIndexB;
  if (gearA >= numBodies || gearB >= numBodies || gearA == gearB ||
      bodies[gearA].invMass <= 0.0f || bodies[gearB].invMass <= 0.0f ||
      !physx::PxIsFinite(gear.gearRatio) ||
      physx::PxAbs(gear.gearRatio) <= 1e-6f ||
      !gear.gearAxis0.isFinite() || !gear.gearAxis1.isFinite() ||
      gear.gearAxis0.magnitudeSquared() <= 1e-8f ||
      gear.gearAxis1.magnitudeSquared() <= 1e-8f)
    return false;

  bool ownsBody[2] = {false, false};
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdD6JointConstraint &joint = d6Joints[i];
    const bool aDynamic =
        joint.header.bodyIndexA < numBodies &&
        bodies[joint.header.bodyIndexA].invMass > 0.0f;
    const bool bDynamic =
        joint.header.bodyIndexB < numBodies &&
        bodies[joint.header.bodyIndexB].invMass > 0.0f;
    if (joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
        aDynamic == bDynamic || joint.linearMotion != 0u ||
        joint.angularMotion != 0x2u || joint.driveFlags != 0u ||
        joint.motorEnabled != 0u)
      return false;

    const physx::PxU32 dynamicIndex =
        aDynamic ? joint.header.bodyIndexA : joint.header.bodyIndexB;
    if (dynamicIndex >= 2 || ownsBody[dynamicIndex])
      return false;
    ownsBody[dynamicIndex] = true;

    const physx::PxVec3 &dynamicAnchor =
        aDynamic ? joint.anchorA : joint.anchorB;
    const physx::PxQuat &dynamicFrame =
        aDynamic ? joint.localFrameA : joint.localFrameB;
    if (dynamicAnchor.magnitudeSquared() > 1e-8f)
      return false;

    physx::PxVec3 hingeAxis =
        dynamicFrame.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    physx::PxVec3 gearAxis =
        dynamicIndex == gearA ? gear.gearAxis0 : gear.gearAxis1;
    if (hingeAxis.normalize() <= 1e-6f ||
        gearAxis.normalize() <= 1e-6f ||
        physx::PxAbs(hingeAxis.dot(gearAxis)) < 0.9999f)
      return false;
  }

  return ownsBody[gearA] && ownsBody[gearB];
}

bool isNativeRevoluteMotorGearVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, const AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts,
    physx::PxU32 &motorJointIndex) {
  motorJointIndex = PX_MAX_U32;
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 2 || !gearJoints || numGear != 1 ||
      numSoftParticles != 0 || numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdGearJointConstraint &gear = gearJoints[0];
  const physx::PxU32 gearA = gear.header.bodyIndexA;
  const physx::PxU32 gearB = gear.header.bodyIndexB;
  if (gearA >= numBodies || gearB >= numBodies || gearA == gearB ||
      bodies[gearA].invMass <= 0.0f || bodies[gearB].invMass <= 0.0f ||
      !physx::PxIsFinite(gear.gearRatio) ||
      physx::PxAbs(gear.gearRatio) <= 1e-6f ||
      !gear.gearAxis0.isFinite() || !gear.gearAxis1.isFinite() ||
      gear.gearAxis0.magnitudeSquared() <= 1e-8f ||
      gear.gearAxis1.magnitudeSquared() <= 1e-8f)
    return false;

  bool ownsBody[2] = {false, false};
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdD6JointConstraint &joint = d6Joints[i];
    const bool dynamicA =
        joint.header.bodyIndexA < numBodies &&
        bodies[joint.header.bodyIndexA].invMass > 0.0f;
    const bool dynamicB =
        joint.header.bodyIndexB < numBodies &&
        bodies[joint.header.bodyIndexB].invMass > 0.0f;
    if (joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
        dynamicA == dynamicB || joint.linearMotion != 0u ||
        joint.angularMotion != 0x2u || joint.driveFlags != 0u ||
        joint.driveAccelerationFlags != 0u ||
        joint.coneAngleLimit > 0.0f)
      return false;

    const physx::PxU32 dynamicIndex =
        dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
    if (dynamicIndex >= 2 || ownsBody[dynamicIndex] ||
        bodies[dynamicIndex].lockFlags != 0)
      return false;
    ownsBody[dynamicIndex] = true;

    const physx::PxVec3 &dynamicAnchor =
        dynamicA ? joint.anchorA : joint.anchorB;
    const physx::PxQuat &dynamicFrame =
        dynamicA ? joint.localFrameA : joint.localFrameB;
    if (dynamicAnchor.magnitudeSquared() > 1e-8f)
      return false;

    physx::PxVec3 hingeAxis =
        dynamicFrame.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    physx::PxVec3 gearAxis =
        dynamicIndex == gearA ? gear.gearAxis0 : gear.gearAxis1;
    if (hingeAxis.normalize() <= 1e-6f ||
        gearAxis.normalize() <= 1e-6f ||
        physx::PxAbs(hingeAxis.dot(gearAxis)) < 0.9999f)
      return false;

    physx::PxVec3 worldAxis =
        bodies[dynamicIndex].rotation.rotate(gearAxis);
    if (worldAxis.normalize() <= 1e-6f)
      return false;
    const physx::PxVec3 response =
        bodies[dynamicIndex].invInertiaWorld.transform(worldAxis);
    const physx::PxReal responseScale =
        physx::PxMax(response.magnitude(), 1e-8f);
    if (response.cross(worldAxis).magnitude() >
        responseScale * 1e-4f)
      return false;

    if (joint.motorEnabled != 0u) {
      if (motorJointIndex != PX_MAX_U32 ||
          !(joint.motorMaxForce > 0.0f) ||
          !physx::PxIsFinite(joint.motorMaxForce) ||
          !physx::PxIsFinite(joint.motorTargetVelocity) ||
          !physx::PxIsFinite(joint.motorGearRatio) ||
          physx::PxAbs(joint.motorGearRatio - 1.0f) > 1e-6f ||
          (joint.sourceFlags & AvbdD6JointConstraint::
                                   eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0)
        return false;
      motorJointIndex = i;
    }
  }

  return ownsBody[gearA] && ownsBody[gearB] &&
         motorJointIndex != PX_MAX_U32;
}

void projectPassiveCenteredGearVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdGearJointConstraint &gear,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart) {
  const physx::PxU32 bodyAIndex = gear.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = gear.header.bodyIndexB;
  if (bodyAIndex >= numBodies || bodyBIndex >= numBodies ||
      angularVelAtSolveStart.size() != numBodies)
    return;

  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxVec3 axisA = bodyA.rotation.rotate(gear.gearAxis0);
  physx::PxVec3 axisB = bodyB.rotation.rotate(gear.gearAxis1);
  if (axisA.normalize() <= 1e-6f || axisB.normalize() <= 1e-6f)
    return;

  const physx::PxVec3 jacobianA = axisA * gear.gearRatio;
  const physx::PxVec3 jacobianB = axisB;
  const physx::PxVec3 responseA =
      bodyA.invInertiaWorld.transform(jacobianA);
  const physx::PxVec3 responseB =
      bodyB.invInertiaWorld.transform(jacobianB);
  const physx::PxReal denominator =
      jacobianA.dot(responseA) + jacobianB.dot(responseB);
  if (!physx::PxIsFinite(denominator) || denominator <= 1e-10f)
    return;

  // This scoped path requires each revolute free axis to be a principal
  // inertia direction.  Otherwise an exact gear impulse also changes locked
  // swing components and must be solved together with the complete D6 row set.
  const physx::PxReal responseScaleA =
      physx::PxMax(responseA.magnitude(), 1e-8f);
  const physx::PxReal responseScaleB =
      physx::PxMax(responseB.magnitude(), 1e-8f);
  if (responseA.cross(axisA).magnitude() > responseScaleA * 1e-4f ||
      responseB.cross(axisB).magnitude() > responseScaleB * 1e-4f)
    return;

  const physx::PxVec3 &rawA = angularVelAtSolveStart[bodyAIndex];
  const physx::PxVec3 &rawB = angularVelAtSolveStart[bodyBIndex];
  const physx::PxReal residual =
      jacobianA.dot(rawA) + jacobianB.dot(rawB);
  const physx::PxReal lambda = -residual / denominator;
  if (!physx::PxIsFinite(lambda))
    return;

  const physx::PxVec3 expectedA = rawA + responseA * lambda;
  const physx::PxVec3 expectedB = rawB + responseB * lambda;
  physx::PxVec3 candidateA =
      bodyA.angularVelocity +
      axisA * (expectedA.dot(axisA) - bodyA.angularVelocity.dot(axisA));
  physx::PxVec3 candidateB =
      bodyB.angularVelocity +
      axisB * (expectedB.dot(axisB) - bodyB.angularVelocity.dot(axisB));
  if (!candidateA.isFinite() || !candidateB.isFinite())
    return;

  // Preserve authored angular-velocity caps.  A capped coupled projection
  // requires an active-set solve and is intentionally outside this predicate.
  if ((bodyA.maxAngularVelocitySq > 0.0f &&
       candidateA.magnitudeSquared() > bodyA.maxAngularVelocitySq) ||
      (bodyB.maxAngularVelocitySq > 0.0f &&
       candidateB.magnitudeSquared() > bodyB.maxAngularVelocitySq))
    return;

  bodyA.angularVelocity = candidateA;
  bodyB.angularVelocity = candidateB;
}

void projectNativeRevoluteMotorGearVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &motor,
    const AvbdGearJointConstraint &gear, physx::PxReal dt) {
  const physx::PxU32 gearAIndex = gear.header.bodyIndexA;
  const physx::PxU32 gearBIndex = gear.header.bodyIndexB;
  if (gearAIndex >= numBodies || gearBIndex >= numBodies ||
      gearAIndex == gearBIndex)
    return;

  const bool motorDynamicA =
      motor.header.bodyIndexA < numBodies &&
      bodies[motor.header.bodyIndexA].invMass > 0.0f;
  const bool motorDynamicB =
      motor.header.bodyIndexB < numBodies &&
      bodies[motor.header.bodyIndexB].invMass > 0.0f;
  if (motorDynamicA == motorDynamicB)
    return;
  const physx::PxU32 motorBodyIndex =
      motorDynamicA ? motor.header.bodyIndexA : motor.header.bodyIndexB;
  if (motorBodyIndex != gearAIndex && motorBodyIndex != gearBIndex)
    return;

  AvbdSolverBody &gearA = bodies[gearAIndex];
  AvbdSolverBody &gearB = bodies[gearBIndex];
  AvbdSolverBody &motorBody = bodies[motorBodyIndex];
  physx::PxVec3 gearAxisA = gearA.rotation.rotate(gear.gearAxis0);
  physx::PxVec3 gearAxisB = gearB.rotation.rotate(gear.gearAxis1);
  physx::PxVec3 motorAxis =
      motorDynamicA
          ? (motorBody.rotation * motor.localFrameA)
                .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
          : motor.localFrameA.rotate(
                physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (gearAxisA.normalize() <= 1e-6f ||
      gearAxisB.normalize() <= 1e-6f ||
      motorAxis.normalize() <= 1e-6f)
    return;

  // Native revolute target convention is axis dot (wB - wA).  The motor
  // Jacobian therefore changes sign when the dynamic endpoint is actor A.
  const physx::PxReal motorSign = motorDynamicA ? -1.0f : 1.0f;
  const physx::PxVec3 motorJacobian = motorAxis * motorSign;
  const physx::PxVec3 gearJacobianA =
      gearAxisA * gear.gearRatio;
  const physx::PxVec3 gearJacobianB = gearAxisB;
  const physx::PxVec3 motorResponse =
      motorBody.invInertiaWorld.transform(motorJacobian);
  const physx::PxVec3 gearResponseA =
      gearA.invInertiaWorld.transform(gearJacobianA);
  const physx::PxVec3 gearResponseB =
      gearB.invInertiaWorld.transform(gearJacobianB);

  const physx::PxReal kMotorMotor =
      motorJacobian.dot(motorResponse);
  const physx::PxReal kGearGear =
      gearJacobianA.dot(gearResponseA) +
      gearJacobianB.dot(gearResponseB);
  const physx::PxReal kMotorGear =
      motorBodyIndex == gearAIndex
          ? motorJacobian.dot(gearResponseA)
          : motorJacobian.dot(gearResponseB);
  const physx::PxReal determinant =
      kMotorMotor * kGearGear - kMotorGear * kMotorGear;
  if (!physx::PxIsFinite(kMotorMotor) ||
      !physx::PxIsFinite(kGearGear) ||
      !physx::PxIsFinite(kMotorGear) ||
      !physx::PxIsFinite(determinant) ||
      kMotorMotor <= 1e-10f || kGearGear <= 1e-10f ||
      determinant <= 1e-12f)
    return;

  const physx::PxReal motorVelocity =
      motorJacobian.dot(motorBody.angularVelocity);
  const physx::PxReal gearVelocity =
      gearJacobianA.dot(gearA.angularVelocity) +
      gearJacobianB.dot(gearB.angularVelocity);
  const physx::PxReal motorRhs =
      motor.motorTargetVelocity - motorVelocity;
  const physx::PxReal gearRhs = -gearVelocity;
  const physx::PxReal unconstrainedMotorImpulse =
      (motorRhs * kGearGear - gearRhs * kMotorGear) /
      determinant;
  const physx::PxReal maximumMotorImpulse =
      motor.motorMaxForce * dt;
  const physx::PxReal motorImpulse = physx::PxClamp(
      unconstrainedMotorImpulse, -maximumMotorImpulse,
      maximumMotorImpulse);
  // When the motor force limit is active, keep that active-set value and
  // close the passive gear row with the remaining gear impulse.
  const physx::PxReal gearImpulse =
      (gearRhs - kMotorGear * motorImpulse) / kGearGear;
  if (!physx::PxIsFinite(motorImpulse) ||
      !physx::PxIsFinite(gearImpulse))
    return;

  physx::PxVec3 candidateA =
      gearA.angularVelocity + gearResponseA * gearImpulse;
  physx::PxVec3 candidateB =
      gearB.angularVelocity + gearResponseB * gearImpulse;
  if (motorBodyIndex == gearAIndex)
    candidateA += motorResponse * motorImpulse;
  else
    candidateB += motorResponse * motorImpulse;
  if (!candidateA.isFinite() || !candidateB.isFinite())
    return;
  if ((gearA.maxAngularVelocitySq > 0.0f &&
       candidateA.magnitudeSquared() > gearA.maxAngularVelocitySq) ||
      (gearB.maxAngularVelocitySq > 0.0f &&
       candidateB.magnitudeSquared() > gearB.maxAngularVelocitySq))
    return;

  gearA.angularVelocity = candidateA;
  gearB.angularVelocity = candidateB;
}

bool isSinglePassiveGenericHard1DVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts, physx::PxU32 &genericIndex) {
  genericIndex = PX_MAX_U32;
  if (!bodies || numBodies == 0 || numContacts != 0 || !d6Joints ||
      numD6 == 0 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  for (physx::PxU32 i = 0; i < numD6; ++i) {
    if (!hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::GenericHard1D) &&
        !hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::ArticulationHardMimic) &&
        !hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::GenericRestitution1D))
      continue;
    if (genericIndex != PX_MAX_U32)
      return false;
    genericIndex = i;
  }
  if (genericIndex == PX_MAX_U32)
    return false;

  const AvbdD6JointConstraint &generic = d6Joints[genericIndex];
  if (generic.header.type != AvbdConstraintType::eJOINT_CUSTOM_1D ||
      !generic.genericLinearA.isFinite() ||
      !generic.genericAngularA.isFinite() ||
      !generic.genericLinearB.isFinite() ||
      !generic.genericAngularB.isFinite() ||
      !physx::PxIsFinite(generic.genericVelocityTarget) ||
      generic.genericMinImpulse >= 0.0f ||
      generic.genericMaxImpulse <= 0.0f)
    return false;

  const physx::PxU32 endpoint[2] = {generic.header.bodyIndexA,
                                    generic.header.bodyIndexB};
  if (endpoint[0] < numBodies && endpoint[1] < numBodies &&
      endpoint[0] == endpoint[1])
    return false;
  const physx::PxVec3 linearJ[2] = {generic.genericLinearA,
                                    generic.genericLinearB};
  const physx::PxVec3 angularJ[2] = {generic.genericAngularA,
                                     generic.genericAngularB};
  bool ownsDynamicEndpoint = false;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    if (bodies[endpoint[side]].invMass <= 0.0f)
      return false;
    ownsDynamicEndpoint = true;

    const bool hasLinear = linearJ[side].magnitudeSquared() > 1e-10f;
    const bool hasAngular = angularJ[side].magnitudeSquared() > 1e-10f;
    // A general spatial row needs a coupled velocity active-set solve.
    // This accepted subdomain has at most one response kind per endpoint.
    if (hasLinear && hasAngular)
      return false;
    if (hasAngular) {
      const physx::PxVec3 response =
          bodies[endpoint[side]].invInertiaWorld * angularJ[side];
      const physx::PxReal scale =
          physx::PxMax(response.magnitude(), 1e-8f);
      if (response.cross(angularJ[side]).magnitude() > scale * 1e-4f)
        return false;
    }
  }
  if (!ownsDynamicEndpoint)
    return false;

  // Any accompanying D6 row must be a centered, passive world attachment.
  // The generic Jacobian may only occupy DOFs that attachment leaves FREE.
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    if (i == genericIndex)
      continue;
    const AvbdD6JointConstraint &joint = d6Joints[i];
    const bool aDynamic = joint.header.bodyIndexA < numBodies;
    const bool bDynamic = joint.header.bodyIndexB < numBodies;
    if (aDynamic == bDynamic || joint.driveFlags != 0u ||
        joint.motorEnabled != 0u)
      return false;
    const physx::PxU32 dynamicIndex =
        aDynamic ? joint.header.bodyIndexA : joint.header.bodyIndexB;
    physx::PxU32 side = PX_MAX_U32;
    if (dynamicIndex == endpoint[0])
      side = 0;
    else if (dynamicIndex == endpoint[1])
      side = 1;
    if (side == PX_MAX_U32)
      return false;

    const physx::PxVec3 &dynamicAnchor =
        aDynamic ? joint.anchorA : joint.anchorB;
    if (dynamicAnchor.magnitudeSquared() > 1e-8f)
      return false;
    const physx::PxQuat &localFrame =
        aDynamic ? joint.localFrameA : joint.localFrameB;
    const physx::PxQuat worldFrame =
        bodies[dynamicIndex].rotation * localFrame;
    for (physx::PxU32 axisIndex = 0; axisIndex < 3; ++axisIndex) {
      physx::PxVec3 localAxis(0.0f);
      localAxis[axisIndex] = 1.0f;
      physx::PxVec3 worldAxis = worldFrame.rotate(localAxis);
      if (joint.getLinearMotion(axisIndex) != 2 &&
          physx::PxAbs(linearJ[side].dot(worldAxis)) > 1e-5f)
        return false;
      if (joint.getAngularMotion(axisIndex) != 2 &&
          physx::PxAbs(angularJ[side].dot(worldAxis)) > 1e-5f)
        return false;
    }
  }
  return true;
}

void projectSinglePassiveGenericHard1DVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &generic,
    const physx::PxArray<physx::PxVec3> &linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart) {
  if (linearVelAtSolveStart.size() != numBodies ||
      angularVelAtSolveStart.size() != numBodies)
    return;

  const physx::PxU32 endpoint[2] = {generic.header.bodyIndexA,
                                    generic.header.bodyIndexB};
  const physx::PxVec3 linearJ[2] = {generic.genericLinearA,
                                    generic.genericLinearB};
  const physx::PxVec3 angularJ[2] = {generic.genericAngularA,
                                     generic.genericAngularB};
  physx::PxVec3 linearResponse[2] = {physx::PxVec3(0.0f),
                                     physx::PxVec3(0.0f)};
  physx::PxVec3 angularResponse[2] = {physx::PxVec3(0.0f),
                                      physx::PxVec3(0.0f)};
  physx::PxReal denominator = 0.0f;
  physx::PxReal startSpeed = 0.0f;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    AvbdSolverBody &body = bodies[endpoint[side]];
    linearResponse[side] = linearJ[side] * body.invMass;
    angularResponse[side] =
        body.invInertiaWorld.transform(angularJ[side]);
    denominator += linearJ[side].dot(linearResponse[side]) +
                   angularJ[side].dot(angularResponse[side]);
    startSpeed +=
        linearJ[side].dot(linearVelAtSolveStart[endpoint[side]]) +
        angularJ[side].dot(angularVelAtSolveStart[endpoint[side]]);
  }
  if (!physx::PxIsFinite(denominator) || denominator <= 1e-10f)
    return;

  physx::PxReal velocityTarget = generic.genericVelocityTarget;
  if (hasAvbdJointObjective(
          generic.objectiveProgram,
          AvbdJointObjectiveKind::GenericRestitution1D)) {
    const physx::PxReal bounceVelocity =
        -generic.genericRestitution * startSpeed;
    if (-startSpeed > generic.genericBounceThreshold &&
        bounceVelocity * generic.genericGeometricError <= 0.0f)
      velocityTarget = bounceVelocity;
  }
  const physx::PxReal residual = startSpeed - velocityTarget;
  const physx::PxReal impulse = physx::PxClamp(
      -residual / denominator, generic.genericMinImpulse,
      generic.genericMaxImpulse);
  if (!physx::PxIsFinite(impulse))
    return;

  physx::PxVec3 candidateLinear[2];
  physx::PxVec3 candidateAngular[2];
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    AvbdSolverBody &body = bodies[endpoint[side]];
    const physx::PxVec3 expectedLinear =
        linearVelAtSolveStart[endpoint[side]] +
        linearResponse[side] * impulse;
    const physx::PxVec3 expectedAngular =
        angularVelAtSolveStart[endpoint[side]] +
        angularResponse[side] * impulse;
    candidateLinear[side] = body.linearVelocity;
    candidateAngular[side] = body.angularVelocity;
    if (linearJ[side].magnitudeSquared() > 1e-10f) {
      physx::PxVec3 axis = linearJ[side].getNormalized();
      candidateLinear[side] +=
          axis * (expectedLinear.dot(axis) -
                  candidateLinear[side].dot(axis));
    }
    if (angularJ[side].magnitudeSquared() > 1e-10f) {
      physx::PxVec3 axis = angularJ[side].getNormalized();
      candidateAngular[side] +=
          axis * (expectedAngular.dot(axis) -
                  candidateAngular[side].dot(axis));
    }
    if (!candidateLinear[side].isFinite() ||
        !candidateAngular[side].isFinite())
      return;
    if ((body.maxLinearVelocitySq > 0.0f &&
         candidateLinear[side].magnitudeSquared() >
             body.maxLinearVelocitySq) ||
        (body.maxAngularVelocitySq > 0.0f &&
         candidateAngular[side].magnitudeSquared() >
             body.maxAngularVelocitySq))
      return;
  }

  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    bodies[endpoint[side]].linearVelocity = candidateLinear[side];
    bodies[endpoint[side]].angularVelocity = candidateAngular[side];
  }

  const bool outputForce =
      (generic.genericRowFlags &
       static_cast<physx::PxU32>(Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;
  generic.writebackLinearImpulse =
      outputForce ? generic.genericLinearA * impulse : physx::PxVec3(0.0f);
  generic.writebackAngularImpulse =
      outputForce ? generic.genericAngularAWriteback * impulse
                  : physx::PxVec3(0.0f);
  generic.writebackLinearImpulseValid = 1;
  generic.writebackAngularImpulseValid = 1;
}

void projectArticulationMimicVelocity1D(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &mimic) {
  const physx::PxReal effectiveMass =
      computeGeneric1DEffectiveMass(mimic, bodies, numBodies);
  if (effectiveMass <= 0.0f || !physx::PxIsFinite(effectiveMass))
    return;
  const physx::PxReal unitResponse = 1.0f / effectiveMass;
  const physx::PxU32 endpoint[2] = {mimic.header.bodyIndexA,
                                    mimic.header.bodyIndexB};
  const physx::PxVec3 linearJ[2] = {mimic.genericLinearA,
                                    mimic.genericLinearB};
  const physx::PxVec3 angularJ[2] = {mimic.genericAngularA,
                                     mimic.genericAngularB};
  physx::PxReal velocityResidual = -mimic.genericVelocityTarget;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    velocityResidual +=
        linearJ[side].dot(bodies[endpoint[side]].linearVelocity) +
        angularJ[side].dot(bodies[endpoint[side]].angularVelocity);
  }
  const physx::PxReal velocityImpulse =
      -velocityResidual / unitResponse;
  if (!physx::PxIsFinite(velocityImpulse))
    return;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies ||
        bodies[endpoint[side]].invMass <= 0.0f)
      continue;
    AvbdSolverBody &body = bodies[endpoint[side]];
    body.linearVelocity +=
        linearJ[side] * (body.invMass * velocityImpulse);
    body.angularVelocity +=
        body.invInertiaWorld * angularJ[side] * velocityImpulse;
  }
}

bool findCoupledSpatialTendonRows(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts,
    physx::PxArray<physx::PxU32> &rowIndices) {
  rowIndices.clear();
  if (!bodies || numBodies < 2 || numContacts != 0 || !d6Joints ||
      numD6 == 0 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdD6JointConstraint &joint = d6Joints[i];
    if ((joint.sourceFlags &
         AvbdD6JointConstraint::eARTICULATION_SPATIAL_TENDON_ROW) == 0)
      continue;
    const physx::PxU32 bodyA = joint.header.bodyIndexA;
    const physx::PxU32 bodyB = joint.header.bodyIndexB;
    const physx::PxReal jacobianMagnitudeSquaredA =
        joint.genericLinearA.magnitudeSquared() +
        joint.genericAngularA.magnitudeSquared();
    const physx::PxReal jacobianMagnitudeSquaredB =
        joint.genericLinearB.magnitudeSquared() +
        joint.genericAngularB.magnitudeSquared();
    if (bodyA >= numBodies || bodyB >= numBodies ||
        bodyA == bodyB || bodies[bodyA].invMass <= 0.0f ||
        bodies[bodyB].invMass <= 0.0f ||
        !PxIsFinite(joint.header.rho) ||
        joint.header.rho < 0.0f ||
        !PxIsFinite(joint.header.damping) ||
        joint.header.damping < 0.0f ||
        !PxIsFinite(joint.genericTendonLimitStiffness) ||
        joint.genericTendonLimitStiffness < 0.0f ||
        (joint.header.rho <= 0.0f &&
         joint.genericTendonLimitStiffness <= 0.0f) ||
        jacobianMagnitudeSquaredA <= 1e-12f ||
        jacobianMagnitudeSquaredB <= 1e-12f ||
        !joint.genericLinearA.isFinite() ||
        !joint.genericAngularA.isFinite() ||
        !joint.genericLinearB.isFinite() ||
        !joint.genericAngularB.isFinite())
      return false;
    rowIndices.pushBack(i);
  }
  return !rowIndices.empty();
}

} // namespace Dy
} // namespace physx
