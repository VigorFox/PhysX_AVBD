// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointPositionSolves.h"
#include "avbd/solver/joint/DyAvbdJointCoupledMath.h"
#include "avbd/solver/joint/DyAvbdJointCoupledSystem.h"
#include <algorithm>
#include <cmath>

namespace physx {
namespace Dy {

bool solveCoupledAngularPositionDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdSolverConfig &config) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(numContacts + 9);

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

  if (!addFrictionlessBodyVsStaticContactRows(
          bodies, numBodies, contacts, numContacts, config, invDt2,
          rows, gradient, preconditioner))
    return false;

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal hardPenalty =
      physx::PxMax(joint.header.rho, physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 linearViolation = bodyA.position - bodyB.position;
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(worldAxes[axis], physx::PxVec3(0.0f));
    row.jacobianB = AvbdVec6(-worldAxes[axis], physx::PxVec3(0.0f));
    row.penalty = hardPenalty;
    row.force = hardPenalty * linearViolation.dot(worldAxes[axis]) +
                joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  physx::PxQuat worldFrameB = bodyB.rotation * joint.localFrameB;
  const physx::PxReal frameAMagnitude = worldFrameA.magnitudeSquared();
  const physx::PxReal frameBMagnitude = worldFrameB.magnitudeSquared();
  if (!(frameAMagnitude > 1e-8f) || !(frameBMagnitude > 1e-8f) ||
      !PxIsFinite(frameAMagnitude) || !PxIsFinite(frameBMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameAMagnitude);
  worldFrameB *= 1.0f / physx::PxSqrt(frameBMagnitude);

  const bool slerp =
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0;
  const physx::PxU32 driveIndex =
      slerp ? PX_MAX_U32
            : (joint.driveFlags == (1u << 3)
                   ? 0u
                   : (joint.driveFlags == (1u << 4) ? 1u : 2u));
  if (!slerp && driveIndex == 0) {
    const physx::PxVec3 worldTwistA =
        worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    const physx::PxVec3 worldTwistB =
        worldFrameB.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    const physx::PxVec3 axisViolation = worldTwistA.cross(worldTwistB);
    physx::PxVec3 perp1 =
        physx::PxAbs(worldTwistA.x) < 0.9f
            ? worldTwistA.cross(physx::PxVec3(1.0f, 0.0f, 0.0f))
            : worldTwistA.cross(physx::PxVec3(0.0f, 1.0f, 0.0f));
    const physx::PxReal perp1Length = perp1.magnitude();
    if (!(perp1Length > 1e-6f) || !PxIsFinite(perp1Length))
      return false;
    perp1 *= 1.0f / perp1Length;
    physx::PxVec3 perp2 = worldTwistA.cross(perp1);
    const physx::PxReal perp2Length = perp2.magnitude();
    if (!(perp2Length > 1e-6f) || !PxIsFinite(perp2Length))
      return false;
    perp2 *= 1.0f / perp2Length;
    const physx::PxVec3 perpendicularAxes[2] = {perp1, perp2};
    for (physx::PxU32 rowIndex = 0; rowIndex < 2; ++rowIndex) {
      CoupledIslandRow row;
      row.bodyA = bodyAIndex;
      row.bodyB = bodyBIndex;
      row.jacobianA =
          AvbdVec6(physx::PxVec3(0.0f), -perpendicularAxes[rowIndex]);
      row.jacobianB =
          AvbdVec6(physx::PxVec3(0.0f), perpendicularAxes[rowIndex]);
      row.penalty = hardPenalty;
      row.force =
          hardPenalty * axisViolation.dot(perpendicularAxes[rowIndex]) +
          joint.lambdaAngular[rowIndex + 1];
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  } else if (!slerp) {
    for (physx::PxU32 axis = 0; axis < 3; ++axis) {
      if (axis == driveIndex)
        continue;
      physx::PxVec3 localAxis(0.0f);
      localAxis[axis] = 1.0f;
      const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
      CoupledIslandRow row;
      row.bodyA = bodyAIndex;
      row.bodyB = bodyBIndex;
      row.jacobianA = AvbdVec6(physx::PxVec3(0.0f), worldAxis);
      row.jacobianB = AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
      row.penalty = hardPenalty;
      row.force =
          hardPenalty *
              joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis) +
          joint.lambdaAngular[axis];
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  }

  physx::PxQuat displacementA =
      bodyA.rotation * bodyA.prevRotation.getConjugate();
  physx::PxQuat displacementB =
      bodyB.rotation * bodyB.prevRotation.getConjugate();
  if (displacementA.w < 0.0f)
    displacementA = -displacementA;
  if (displacementB.w < 0.0f)
    displacementB = -displacementB;
  const physx::PxVec3 relativeAngularDisplacement =
      physx::PxVec3(displacementB.x - displacementA.x,
                    displacementB.y - displacementA.y,
                    displacementB.z - displacementA.z) *
      2.0f;

  physx::PxQuat currentRelative =
      worldFrameA.getConjugate() * worldFrameB;
  currentRelative.normalize();
  physx::PxQuat targetRelative = joint.driveAngularPosition;
  if (currentRelative.dot(targetRelative) < 0.0f)
    targetRelative = -targetRelative;
  if (slerp) {
    const physx::PxQuat delta =
        targetRelative.getConjugate() * currentRelative;
    physx::PxVec3 driveAxes[3];
    computeSlerpJacobianAxes(driveAxes, worldFrameA * targetRelative,
                             worldFrameB);
    const physx::PxReal stiffness = joint.angularStiffness.z;
    const physx::PxReal damping = joint.angularDamping.z;
    const physx::PxReal limit = joint.driveAngularForce.z;
    for (physx::PxU32 rowIndex = 0; rowIndex < 3; ++rowIndex) {
      const physx::PxReal velocityError =
          relativeAngularDisplacement.dot(driveAxes[rowIndex]);
      const physx::PxReal rawTorque =
          stiffness * (&delta.x)[rowIndex] +
          (damping / dt) * velocityError;
      const physx::PxReal driveTorque =
          physx::PxClamp(rawTorque, -limit, limit);
      CoupledIslandRow row;
      row.bodyA = bodyAIndex;
      row.bodyB = bodyBIndex;
      row.jacobianA =
          AvbdVec6(physx::PxVec3(0.0f), -driveAxes[rowIndex]);
      row.jacobianB =
          AvbdVec6(physx::PxVec3(0.0f), driveAxes[rowIndex]);
      row.penalty =
          physx::PxAbs(rawTorque) >= limit ? 0.0f
                                           : stiffness + damping / dt;
      row.force = driveTorque;
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  } else {
    physx::PxVec3 localAxis(0.0f);
    localAxis[driveIndex] = 1.0f;
    const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
    const physx::PxQuat delta =
        currentRelative * targetRelative.getConjugate();
    physx::PxReal positionResidual = 0.0f;
    physx::PxReal positionTangent = 0.0f;
    if (driveIndex == 0) {
      positionResidual = 2.0f * delta.x;
      positionTangent = physx::PxAbs(delta.w);
    } else if (driveIndex == 1) {
      positionResidual = -delta.getBasisVector0().z;
      positionTangent =
          physx::PxAbs(1.0f - 2.0f * delta.y * delta.y);
    } else {
      positionResidual = delta.getBasisVector0().y;
      positionTangent =
          physx::PxAbs(1.0f - 2.0f * delta.z * delta.z);
    }
    const physx::PxReal stiffness = joint.angularStiffness[driveIndex];
    const physx::PxReal damping = joint.angularDamping[driveIndex];
    const physx::PxReal limit = joint.driveAngularForce[driveIndex];
    const physx::PxReal velocityError =
        relativeAngularDisplacement.dot(worldAxis);
    const physx::PxReal rawTorque =
        stiffness * positionResidual + (damping / dt) * velocityError;
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
    row.jacobianB = AvbdVec6(physx::PxVec3(0.0f), worldAxis);
    row.penalty =
        physx::PxAbs(rawTorque) >= limit
            ? 0.0f
            : stiffness * positionTangent + damping / dt;
    row.force = physx::PxClamp(rawTorque, -limit, limit);
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

bool solveCoupledLinearPositionDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const physx::PxVec3 &gravity,
    const AvbdSolverConfig &config) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(numContacts * 3u + 6u);

  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  physx::PxQuat worldFrameB = bodyB.rotation * joint.localFrameB;
  const physx::PxReal frameAMagnitude = worldFrameA.magnitudeSquared();
  const physx::PxReal frameBMagnitude = worldFrameB.magnitudeSquared();
  if (!(frameAMagnitude > 1e-8f) || !(frameBMagnitude > 1e-8f) ||
      !PxIsFinite(frameAMagnitude) || !PxIsFinite(frameBMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameAMagnitude);
  worldFrameB *= 1.0f / physx::PxSqrt(frameBMagnitude);
  const physx::PxVec3 linearAxes[3] = {
      worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f)),
      worldFrameA.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f)),
      worldFrameA.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f))};
  const physx::PxVec3 driveAxis = linearAxes[0];
  physx::PxVec3 supportNormal(0.0f);
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    if (physx::PxAbs(contacts[i].contactNormal.dot(driveAxis)) > 1e-5f)
      return false;
    physx::PxVec3 normal = contacts[i].contactNormal;
    const physx::PxReal normalLength = normal.magnitude();
    if (!(normalLength > 1e-6f) || !PxIsFinite(normalLength))
      return false;
    normal *= 1.0f / normalLength;
    if (i == 0)
      supportNormal = normal;
    else if (physx::PxAbs(normal.dot(supportNormal)) < 0.9999f)
      return false;
  }

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

  if (!addBodyVsStaticContactNormalRows(
          bodies, numBodies, contacts, numContacts, config, invDt2,
          rows, gradient, preconditioner, true))
    return false;
  const bool frictionalContacts =
      contacts[0].friction > 0.0f ||
      contacts[0].staticFriction > 0.0f;
  if (frictionalContacts &&
      !addStrictFrictionalBodyVsStaticContactPositionRows(
          bodies, numBodies, contacts, numContacts, gravity, config, invDt2,
          rows, gradient, preconditioner))
    return false;

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal hardPenalty =
      physx::PxMax(joint.header.rho, physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 worldAnchorA = bodyA.position + rA;
  const physx::PxVec3 worldAnchorB = bodyB.position + rB;
  const physx::PxVec3 lockedLinearViolation =
      worldAnchorA - worldAnchorB;
  for (physx::PxU32 axis = 1; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(linearAxes[axis], rA.cross(linearAxes[axis]));
    row.jacobianB =
        AvbdVec6(-linearAxes[axis], -rB.cross(linearAxes[axis]));
    row.penalty = hardPenalty;
    row.force =
        hardPenalty * lockedLinearViolation.dot(linearAxes[axis]) +
        joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(physx::PxVec3(0.0f), worldAxis);
    row.jacobianB =
        AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
    row.penalty = hardPenalty;
    row.force =
        hardPenalty *
            joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis) +
        joint.lambdaAngular[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  const physx::PxVec3 previousRA =
      bodyA.prevRotation.rotate(joint.anchorA);
  const physx::PxVec3 previousRB =
      bodyB.prevRotation.rotate(joint.anchorB);
  const physx::PxVec3 displacementA =
      worldAnchorA - (bodyA.prevPosition + previousRA);
  const physx::PxVec3 displacementB =
      worldAnchorB - (bodyB.prevPosition + previousRB);
  const physx::PxReal positionResidual =
      (worldAnchorB - worldAnchorA).dot(driveAxis) -
      joint.driveLinearPosition.x;
  const physx::PxReal displacementResidual =
      (displacementB - displacementA).dot(driveAxis) -
      joint.driveLinearVelocity.x * dt;
  const physx::PxReal stiffness = joint.linearStiffness.x;
  const physx::PxReal damping = joint.linearDamping.x;
  const physx::PxReal limit = joint.driveLinearForce.x;
  const physx::PxReal rawForce =
      stiffness * positionResidual +
      (damping / dt) * displacementResidual;
  CoupledIslandRow driveRow;
  driveRow.bodyA = bodyAIndex;
  driveRow.bodyB = bodyBIndex;
  driveRow.jacobianA =
      AvbdVec6(-driveAxis, -rA.cross(driveAxis));
  driveRow.jacobianB =
      AvbdVec6(driveAxis, rB.cross(driveAxis));
  driveRow.penalty =
      physx::PxAbs(rawForce) >= limit
          ? 0.0f
          : stiffness + damping / dt;
  driveRow.force = physx::PxClamp(rawForce, -limit, limit);
  addCoupledRow(driveRow, rows, gradient, preconditioner);

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
  if (!(residualProduct >= 0.0) ||
      !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual = 1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction, operatorDirection);
    const double denominator = dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) ||
        !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) ||
        !std::isfinite(nextProduct))
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

  if (!frictionalContacts) {
    physx::PxReal totalMass = 0.0f;
    physx::PxVec3 expectedWeightedDelta(0.0f);
    physx::PxVec3 solvedWeightedDelta(0.0f);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal mass = 1.0f / bodies[i].invMass;
      totalMass += mass;
      expectedWeightedDelta +=
          (bodies[i].position - bodies[i].inertialPosition) * mass;
      solvedWeightedDelta += solution[i].linear * mass;
    }
    if (!(totalMass > 0.0f) || !PxIsFinite(totalMass))
      return false;
    physx::PxVec3 translationRoundoff =
        solvedWeightedDelta - expectedWeightedDelta;
    translationRoundoff -=
        supportNormal * translationRoundoff.dot(supportNormal);
    const physx::PxVec3 commonRoundoff =
        translationRoundoff / totalMass;
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      solution[i].linear -= commonRoundoff;
  }

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

} // namespace Dy
} // namespace physx
