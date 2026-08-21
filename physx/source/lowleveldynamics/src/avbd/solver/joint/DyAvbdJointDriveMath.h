// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_DRIVE_MATH_H
#define DY_AVBD_JOINT_DRIVE_MATH_H

#include "avbd/solver/joint/DyAvbdJointGeometryPolicy.h"

namespace physx {
namespace Dy {

PX_FORCE_INLINE physx::PxVec3 computeGeneric1DRotationDelta(
    const physx::PxQuat &current, const physx::PxQuat &reference) {
  physx::PxQuat delta = current * reference.getConjugate();
  if (delta.w < 0.0f)
    delta = -delta;

  const physx::PxVec3 imaginary(delta.x, delta.y, delta.z);
  const physx::PxReal sinHalfSquared = imaginary.magnitudeSquared();
  if (sinHalfSquared <= 1e-20f)
    return imaginary * 2.0f;

  const physx::PxReal sinHalf = physx::PxSqrt(sinHalfSquared);
  const physx::PxReal angle = 2.0f * physx::PxAtan2(
      sinHalf, physx::PxClamp(delta.w, 0.0f, 1.0f));
  return imaginary * (angle / sinHalf);
}

PX_FORCE_INLINE physx::PxReal computeGeneric1DViolation(
    const AvbdD6JointConstraint &joint, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxReal dt) {
  physx::PxReal violation =
      joint.genericGeometricError - joint.genericVelocityTarget * dt;
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;

  if (bodyA < numBodies) {
    violation += joint.genericLinearA.dot(
        bodies[bodyA].position - joint.genericReferencePositionA);
    violation += joint.genericAngularA.dot(computeGeneric1DRotationDelta(
        bodies[bodyA].rotation, joint.genericReferenceRotationA));
  }
  if (bodyB < numBodies) {
    violation += joint.genericLinearB.dot(
        bodies[bodyB].position - joint.genericReferencePositionB);
    violation += joint.genericAngularB.dot(computeGeneric1DRotationDelta(
        bodies[bodyB].rotation, joint.genericReferenceRotationB));
  }
  return violation;
}

PX_FORCE_INLINE physx::PxReal computeGeneric1DEffectiveMass(
    const AvbdD6JointConstraint &joint, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies) {
  physx::PxReal unitResponse = 0.0f;
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (bodyA < numBodies) {
    unitResponse +=
        bodies[bodyA].invMass * joint.genericLinearA.magnitudeSquared();
    unitResponse += joint.genericAngularA.dot(
        bodies[bodyA].invInertiaWorld * joint.genericAngularA);
  }
  if (bodyB < numBodies) {
    unitResponse +=
        bodies[bodyB].invMass * joint.genericLinearB.magnitudeSquared();
    unitResponse += joint.genericAngularB.dot(
        bodies[bodyB].invInertiaWorld * joint.genericAngularB);
  }
  return unitResponse > 1e-10f ? 1.0f / unitResponse : 0.0f;
}

PX_FORCE_INLINE void computeMaxPoseDeltas(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<physx::PxVec3> &prevPos,
    const physx::PxArray<physx::PxQuat> &prevRot,
    physx::PxReal &maxPositionDelta, physx::PxReal &maxRotationDelta) {
  maxPositionDelta = 0.0f;
  maxRotationDelta = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;

    maxPositionDelta = physx::PxMax(
        maxPositionDelta, (bodies[i].position - prevPos[i]).magnitude());
    maxRotationDelta = physx::PxMax(
        maxRotationDelta,
        computeRotationDeltaMagnitude(bodies[i].rotation, prevRot[i]));
  }
}

PX_FORCE_INLINE physx::PxReal computeLinearDriveRecipResponse(
    const AvbdSolverBody *bodyA, const AvbdSolverBody *bodyB,
    const physx::PxVec3 &rA, const physx::PxVec3 &rB,
    const physx::PxVec3 &worldAxis) {
  physx::PxReal unitResponse = 0.0f;
  if (bodyA) {
    unitResponse += bodyA->invMass;
    const physx::PxVec3 angA = rA.cross(worldAxis);
    unitResponse += (bodyA->invInertiaWorld * angA).dot(angA);
  }
  if (bodyB) {
    unitResponse += bodyB->invMass;
    const physx::PxVec3 angB = rB.cross(worldAxis);
    unitResponse += (bodyB->invInertiaWorld * angB).dot(angB);
  }
  return unitResponse > 1e-8f ? (1.0f / unitResponse) : 0.0f;
}

PX_FORCE_INLINE physx::PxReal computeAngularDriveRecipResponse(
    const AvbdSolverBody *bodyA, const AvbdSolverBody *bodyB,
    const physx::PxVec3 &worldAxis) {
  physx::PxReal unitResponse = 0.0f;
  if (bodyA)
    unitResponse += (bodyA->invInertiaWorld * worldAxis).dot(worldAxis);
  if (bodyB)
    unitResponse += (bodyB->invInertiaWorld * worldAxis).dot(worldAxis);
  return unitResponse > 1e-8f ? (1.0f / unitResponse) : 0.0f;
}

} // namespace Dy
} // namespace physx

#endif
