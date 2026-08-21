// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdSpatialTendon.h"
#include "avbd/solver/joint/DyAvbdJointDriveMath.h"

namespace physx {
namespace Dy {

bool solveCoupledSpatialTendonRow(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2) {
  if (!bodies || dt <= 0.0f)
    return false;

  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB)
    return false;

  AvbdSolverBody &endpointA = bodies[bodyA];
  AvbdSolverBody &endpointB = bodies[bodyB];
  const AvbdVec6 jacobianA(
      joint.genericLinearA, joint.genericAngularA);
  const AvbdVec6 jacobianB(
      joint.genericLinearB, joint.genericAngularB);
  const physx::PxReal jacobianMagnitudeA = physx::PxSqrt(
      jacobianA.linear.magnitudeSquared() +
      jacobianA.angular.magnitudeSquared());
  const physx::PxReal jacobianMagnitudeB = physx::PxSqrt(
      jacobianB.linear.magnitudeSquared() +
      jacobianB.angular.magnitudeSquared());
  if (!(jacobianMagnitudeA > 1e-6f) ||
      !(jacobianMagnitudeB > 1e-6f))
    return false;
  const AvbdVec6 directionA(
      jacobianA.linear / jacobianMagnitudeA,
      jacobianA.angular / jacobianMagnitudeA);
  const AvbdVec6 directionB(
      jacobianB.linear / jacobianMagnitudeB,
      jacobianB.angular / jacobianMagnitudeB);

  auto computeInertialTerm =
      [invDt2](const AvbdSolverBody &body,
               const AvbdVec6 &direction,
               physx::PxReal &hessian,
               physx::PxReal &gradient) -> bool {
    if (body.invMass <= 0.0f)
      return false;
    const physx::PxReal mass = 1.0f / body.invMass;
    hessian =
        mass * direction.linear.magnitudeSquared() * invDt2;
    gradient =
        mass * (body.position - body.inertialPosition)
                   .dot(direction.linear) *
        invDt2;
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxMat33 inertia = body.invInertiaWorld.getInverse();
    const physx::PxReal angularHessian =
        direction.angular.dot(inertia * direction.angular) *
        invDt2;
    hessian += angularHessian;
    gradient += angularHessian *
                rotationError.dot(direction.angular);
    return hessian > 1e-8f && PxIsFinite(hessian) &&
           PxIsFinite(gradient);
  };

  physx::PxReal inertialHessianA = 0.0f;
  physx::PxReal inertialHessianB = 0.0f;
  physx::PxReal gradientA = 0.0f;
  physx::PxReal gradientB = 0.0f;
  if (!computeInertialTerm(endpointA, directionA, inertialHessianA,
                           gradientA) ||
      !computeInertialTerm(endpointB, directionB, inertialHessianB,
                           gradientB))
    return false;

  const physx::PxReal violation =
      computeGeneric1DViolation(joint, bodies, numBodies, dt);
  const physx::PxReal velocity =
      (violation - joint.genericGeometricError) / dt;
  physx::PxReal penalty =
      joint.header.rho + joint.header.damping / dt;
  physx::PxReal unclampedForce =
      joint.header.rho * violation + joint.header.damping * velocity;
  if (joint.genericTendonLimitStiffness > 0.0f) {
    physx::PxReal limitViolation = 0.0f;
    if (violation < joint.genericTendonLowLimit)
      limitViolation = violation - joint.genericTendonLowLimit;
    else if (violation > joint.genericTendonHighLimit)
      limitViolation = violation - joint.genericTendonHighLimit;
    if (limitViolation != 0.0f) {
      penalty +=
          joint.genericTendonLimitStiffness +
          joint.header.damping / dt;
      unclampedForce +=
          joint.genericTendonLimitStiffness * limitViolation;
    }
  }
  const physx::PxReal appliedImpulse = physx::PxClamp(
      -unclampedForce * dt, joint.genericMinImpulse,
      joint.genericMaxImpulse);
  const physx::PxReal force = -appliedImpulse / dt;
  gradientA += jacobianMagnitudeA * force;
  gradientB += jacobianMagnitudeB * force;

  const physx::PxReal hessianAA =
      inertialHessianA +
      penalty * jacobianMagnitudeA * jacobianMagnitudeA;
  const physx::PxReal hessianBB =
      inertialHessianB +
      penalty * jacobianMagnitudeB * jacobianMagnitudeB;
  const physx::PxReal hessianAB =
      penalty * jacobianMagnitudeA * jacobianMagnitudeB;
  const physx::PxReal determinant =
      hessianAA * hessianBB - hessianAB * hessianAB;
  if (!(determinant > 1e-12f) || !PxIsFinite(determinant))
    return false;
  const physx::PxReal solutionA =
      (gradientA * hessianBB - gradientB * hessianAB) / determinant;
  const physx::PxReal solutionB =
      (hessianAA * gradientB - hessianAB * gradientA) / determinant;
  if (!PxIsFinite(solutionA) || !PxIsFinite(solutionB))
    return false;

  endpointA.position -= directionA.linear * solutionA;
  endpointB.position -= directionB.linear * solutionB;
  const physx::PxVec3 angularSolutionA =
      directionA.angular * solutionA;
  const physx::PxVec3 angularSolutionB =
      directionB.angular * solutionB;
  if (angularSolutionA.magnitudeSquared() > 1e-12f) {
    const physx::PxQuat delta(angularSolutionA.x, angularSolutionA.y,
                              angularSolutionA.z, 0.0f);
    endpointA.rotation =
        (endpointA.rotation - delta * endpointA.rotation * 0.5f)
            .getNormalized();
  }
  if (angularSolutionB.magnitudeSquared() > 1e-12f) {
    const physx::PxQuat delta(angularSolutionB.x, angularSolutionB.y,
                              angularSolutionB.z, 0.0f);
    endpointB.rotation =
        (endpointB.rotation - delta * endpointB.rotation * 0.5f)
            .getNormalized();
  }
  return true;
}

} // namespace Dy
} // namespace physx
