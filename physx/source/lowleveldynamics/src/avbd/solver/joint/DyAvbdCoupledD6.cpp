// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdCoupledD6.h"
#include "avbd/solver/joint/DyAvbdJointCoupledSystem.h"
#include "avbd/solver/joint/DyAvbdJointGeometryPolicy.h"
#include <algorithm>
#include <cmath>

namespace physx {
namespace Dy {

bool isCoupledFixedD6IslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  if (joint.header.type != AvbdConstraintType::eJOINT_FIXED ||
      joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      physx::PxAbs(bodies[joint.header.bodyIndexA].invMass -
                    bodies[joint.header.bodyIndexB].invMass) >
          1e-6f * physx::PxMax(
                      bodies[joint.header.bodyIndexA].invMass,
                      bodies[joint.header.bodyIndexB].invMass) ||
      bodies[joint.header.bodyIndexA].lockFlags != 0 ||
      bodies[joint.header.bodyIndexB].lockFlags != 0 ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexA].angularDampingBody != 0.0f ||
      bodies[joint.header.bodyIndexB].angularDampingBody != 0.0f ||
      joint.linearMotion != 0 || joint.angularMotion != 0 ||
      joint.driveFlags != 0 || joint.driveAccelerationFlags != 0 ||
      joint.motorEnabled != 0 || joint.coneAngleLimit > 0.0f ||
      !(joint.header.rho > 0.0f) || !PxIsFinite(joint.header.rho) ||
      !joint.anchorA.isFinite() || !joint.anchorB.isFinite() ||
      !joint.localFrameA.isFinite() || !joint.localFrameB.isFinite())
    return false;
  return true;
}

bool isCoupledSphericalConeIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (joint.header.type != AvbdConstraintType::eJOINT_SPHERICAL ||
      bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
      bodies[bodyA].invMass <= 0.0f || bodies[bodyB].invMass <= 0.0f ||
      bodies[bodyA].lockFlags != 0 || bodies[bodyB].lockFlags != 0 ||
      bodies[bodyA].linearDamping != 0.0f ||
      bodies[bodyB].linearDamping != 0.0f ||
      bodies[bodyA].angularDampingBody != 0.0f ||
      bodies[bodyB].angularDampingBody != 0.0f ||
      joint.linearMotion != 0 || joint.angularMotion != 0x2Au ||
      joint.driveFlags != 0 || joint.driveAccelerationFlags != 0 ||
      joint.motorEnabled != 0 ||
      (joint.sourceFlags & AvbdD6JointConstraint::
           eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE) == 0 ||
      joint.coneAngleLimit <= 0.0f || joint.coneAngleLimitZ <= 0.0f ||
      !(joint.header.rho > 0.0f) || !PxIsFinite(joint.header.rho) ||
      !joint.anchorA.isFinite() || !joint.anchorB.isFinite() ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      !joint.localFrameA.isFinite() || !joint.localFrameB.isFinite())
    return false;
  return true;
}

// Solve all six bilateral rows of one native PxFixedJoint against both
// dynamic endpoints in a single frozen Newton system.  Per-body block descent
// omits the off-diagonal J_A^T J_B blocks and can therefore inject a common
// translation into a free island under equal-and-opposite loads.
bool solveCoupledFixedD6Island(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &joint, physx::PxReal invDt2) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(6);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal penalty =
      physx::PxMax(joint.header.rho, physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 linearViolation =
      bodyA.position + rA - bodyB.position - rB;
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(worldAxes[axis], rA.cross(worldAxes[axis]));
    row.jacobianB =
        AvbdVec6(-worldAxes[axis], -rB.cross(worldAxes[axis]));
    row.penalty = penalty;
    row.force = penalty * linearViolation.dot(worldAxes[axis]) +
                joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = worldFrameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(physx::PxVec3(0.0f), worldAxis);
    row.jacobianB = AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
    row.penalty = penalty;
    row.force =
        penalty *
            joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis) +
        joint.lambdaAngular[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) || !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual = 1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction, operatorDirection);
    const double denominator = dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) || !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) || !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] = preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                               solution[i].angular.y,
                               solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation - delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
}

// Solve the three spherical anchor rows and its active cone inequality against
// both dynamic endpoints in one frozen Newton system.  The cone row is an
// internal equal-and-opposite angular constraint; per-body block descent drops
// its off-diagonal block and can rotate both endpoints in the same direction.
bool solveCoupledSphericalConeIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &joint, physx::PxReal invDt2) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(4);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f,
                                      deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal penalty =
      physx::PxMax(joint.header.rho,
                   physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 linearViolation =
      bodyA.position + rA - bodyB.position - rB;
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(worldAxes[axis], rA.cross(worldAxes[axis]));
    row.jacobianB =
        AvbdVec6(-worldAxes[axis], -rB.cross(worldAxes[axis]));
    row.penalty = penalty;
    row.force = penalty * linearViolation.dot(worldAxes[axis]) +
                joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxVec3 coneAxis(0.0f);
  physx::PxReal coneViolation = 0.0f;
  if (!computeEllipticalConeConstraint(
          joint, bodyA.rotation, bodyB.rotation, coneAxis,
          coneViolation))
    return false;
  const physx::PxReal coneForce =
      penalty * coneViolation - joint.coneLambda;
  if (coneForce > 0.0f &&
      coneAxis.magnitudeSquared() > 1e-12f) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(physx::PxVec3(0.0f), -coneAxis);
    row.jacobianB =
        AvbdVec6(physx::PxVec3(0.0f), coneAxis);
    row.penalty = penalty;
    row.force = coneForce;
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(
            preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) || !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual =
      1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction,
                         operatorDirection);
    const double denominator =
        dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) || !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) || !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] =
            preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                               solution[i].angular.y,
                               solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation -
           delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
}

// Project the velocity counterpart of the same six fixed-joint rows.  This is
// one bilateral impulse solve, with no speed threshold and no common-mode
// momentum correction: J M^-1 J^T impulse = -J velocity.
bool projectCoupledFixedD6Velocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  if (!bodies || bodyAIndex >= numBodies || bodyBIndex >= numBodies)
    return false;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = worldFrameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameMagnitude);

  AvbdVec6 jacobianA[6];
  AvbdVec6 jacobianB[6];
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    jacobianA[axis] =
        AvbdVec6(worldAxes[axis], rA.cross(worldAxes[axis]));
    jacobianB[axis] =
        AvbdVec6(-worldAxes[axis], -rB.cross(worldAxes[axis]));
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    const physx::PxVec3 angularAxis = worldFrameA.rotate(localAxis);
    jacobianA[3 + axis] =
        AvbdVec6(physx::PxVec3(0.0f), angularAxis);
    jacobianB[3 + axis] =
        AvbdVec6(physx::PxVec3(0.0f), -angularAxis);
  }

  const AvbdVec6 velocityA(bodyA.linearVelocity, bodyA.angularVelocity);
  const AvbdVec6 velocityB(bodyB.linearVelocity, bodyB.angularVelocity);
  AvbdVec6 residual;
  for (physx::PxU32 row = 0; row < 6; ++row) {
    const physx::PxReal value =
        jacobianA[row].dot(velocityA) + jacobianB[row].dot(velocityB);
    if (row < 3)
      residual.linear[row] = value;
    else
      residual.angular[row - 3] = value;
  }

  AvbdBlock6x6 response;
  response.setZero();
  const auto setResponse = [&response](physx::PxU32 row,
                                       physx::PxU32 column,
                                       physx::PxReal value) {
    if (row < 3 && column < 3)
      response.linearLinear(row, column) = value;
    else if (row < 3)
      response.linearAngular(row, column - 3) = value;
    else if (column < 3)
      response.angularLinear(row - 3, column) = value;
    else
      response.angularAngular(row - 3, column - 3) = value;
  };
  for (physx::PxU32 row = 0; row < 6; ++row) {
    for (physx::PxU32 column = 0; column < 6; ++column) {
      const AvbdVec6 responseA(
          jacobianA[column].linear * bodyA.invMass,
          bodyA.invInertiaWorld * jacobianA[column].angular);
      const AvbdVec6 responseB(
          jacobianB[column].linear * bodyB.invMass,
          bodyB.invInertiaWorld * jacobianB[column].angular);
      setResponse(row, column,
                  jacobianA[row].dot(responseA) +
                      jacobianB[row].dot(responseB));
    }
  }

  AvbdLDLT responseLdlt;
  if (!responseLdlt.decomposeWithRegularization(response))
    return false;
  const AvbdVec6 impulse = responseLdlt.solve(-residual);
  if (!impulse.linear.isFinite() || !impulse.angular.isFinite())
    return false;

  AvbdVec6 bodyImpulseA;
  AvbdVec6 bodyImpulseB;
  for (physx::PxU32 row = 0; row < 6; ++row) {
    const physx::PxReal rowImpulse =
        row < 3 ? impulse.linear[row] : impulse.angular[row - 3];
    addScaled(bodyImpulseA, jacobianA[row], rowImpulse);
    addScaled(bodyImpulseB, jacobianB[row], rowImpulse);
  }
  bodyA.linearVelocity += bodyImpulseA.linear * bodyA.invMass;
  bodyA.angularVelocity +=
      bodyA.invInertiaWorld * bodyImpulseA.angular;
  bodyB.linearVelocity += bodyImpulseB.linear * bodyB.invMass;
  bodyB.angularVelocity +=
      bodyB.invInertiaWorld * bodyImpulseB.angular;
  return true;
}

} // namespace Dy
} // namespace physx
