// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdLinearDriveSolve.h"
#include "avbd/solver/joint/DyAvbdJointCoupledSystem.h"
#include <algorithm>
#include <cmath>

namespace physx {
namespace Dy {

bool computeTwoBodySupportAxisAngularMomentum(
    const AvbdSolverBody &bodyA, const AvbdSolverBody &bodyB,
    const physx::PxVec3 &supportAxis,
    physx::PxReal linearScale, physx::PxReal angularScale,
    physx::PxReal &axisAngularMomentum) {
  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal totalMass = massA + massB;
  if (!(totalMass > 0.0f) || !PxIsFinite(totalMass) ||
      !PxIsFinite(linearScale) || !PxIsFinite(angularScale))
    return false;
  const physx::PxVec3 centerOfMass =
      (bodyA.position * massA + bodyB.position * massB) /
      totalMass;
  const physx::PxVec3 orbitalAngularMomentum =
      (bodyA.position - centerOfMass)
              .cross(bodyA.linearVelocity * massA) +
      (bodyB.position - centerOfMass)
              .cross(bodyB.linearVelocity * massB);
  const physx::PxMat33 inertiaA =
      bodyA.invInertiaWorld.getInverse();
  const physx::PxMat33 inertiaB =
      bodyB.invInertiaWorld.getInverse();
  const physx::PxVec3 spinAngularMomentum =
      inertiaA.transform(bodyA.angularVelocity) +
      inertiaB.transform(bodyB.angularVelocity);
  axisAngularMomentum =
      supportAxis.dot(orbitalAngularMomentum * linearScale +
                      spinAngularMomentum * angularScale);
  return PxIsFinite(axisAngularMomentum);
}

bool restoreTwoBodySupportAxisAngularMomentum(
    AvbdSolverBody &bodyA, AvbdSolverBody &bodyB,
    const physx::PxVec3 &supportAxis,
    physx::PxReal expectedAxisAngularMomentum) {
  physx::PxReal currentAxisAngularMomentum = 0.0f;
  if (!computeTwoBodySupportAxisAngularMomentum(
          bodyA, bodyB, supportAxis, 1.0f, 1.0f,
          currentAxisAngularMomentum))
    return false;
  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal totalMass = massA + massB;
  const physx::PxVec3 centerOfMass =
      (bodyA.position * massA + bodyB.position * massB) /
      totalMass;
  const physx::PxVec3 armA = bodyA.position - centerOfMass;
  const physx::PxVec3 armB = bodyB.position - centerOfMass;
  const physx::PxMat33 inertiaA =
      bodyA.invInertiaWorld.getInverse();
  const physx::PxMat33 inertiaB =
      bodyB.invInertiaWorld.getInverse();
  const physx::PxVec3 tangentArmA = supportAxis.cross(armA);
  const physx::PxVec3 tangentArmB = supportAxis.cross(armB);
  const physx::PxReal axisInertia =
      massA * tangentArmA.magnitudeSquared() +
      supportAxis.dot(inertiaA.transform(supportAxis)) +
      massB * tangentArmB.magnitudeSquared() +
      supportAxis.dot(inertiaB.transform(supportAxis));
  if (!(axisInertia > 1e-10f) || !PxIsFinite(axisInertia) ||
      !PxIsFinite(expectedAxisAngularMomentum))
    return false;

  const physx::PxReal angularCorrection =
      (expectedAxisAngularMomentum - currentAxisAngularMomentum) /
      axisInertia;
  const physx::PxVec3 commonAngularVelocity =
      supportAxis * angularCorrection;
  const physx::PxVec3 candidateLinearA =
      bodyA.linearVelocity + commonAngularVelocity.cross(armA);
  const physx::PxVec3 candidateLinearB =
      bodyB.linearVelocity + commonAngularVelocity.cross(armB);
  const physx::PxVec3 candidateAngularA =
      bodyA.angularVelocity + commonAngularVelocity;
  const physx::PxVec3 candidateAngularB =
      bodyB.angularVelocity + commonAngularVelocity;
  if (!candidateLinearA.isFinite() || !candidateLinearB.isFinite() ||
      !candidateAngularA.isFinite() || !candidateAngularB.isFinite() ||
      (bodyA.maxLinearVelocitySq > 0.0f &&
       candidateLinearA.magnitudeSquared() >
           bodyA.maxLinearVelocitySq) ||
      (bodyB.maxLinearVelocitySq > 0.0f &&
       candidateLinearB.magnitudeSquared() >
           bodyB.maxLinearVelocitySq) ||
      (bodyA.maxAngularVelocitySq > 0.0f &&
       candidateAngularA.magnitudeSquared() >
           bodyA.maxAngularVelocitySq) ||
      (bodyB.maxAngularVelocitySq > 0.0f &&
       candidateAngularB.magnitudeSquared() >
           bodyB.maxAngularVelocitySq))
    return false;

  bodyA.linearVelocity = candidateLinearA;
  bodyA.angularVelocity = candidateAngularA;
  bodyB.linearVelocity = candidateLinearB;
  bodyB.angularVelocity = candidateAngularB;
  return true;
}

bool solveCoupledLinearDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt, physx::PxReal invDt2,
    const AvbdSolverConfig &config) {
  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(numContacts + 1);

  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  const AvbdSolverBody &bodyA = bodies[bodyAIndex];
  const AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxQuat frameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = frameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  frameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  const physx::PxVec3 axis =
      frameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  physx::PxVec3 supportNormal(0.0f);
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    if (physx::PxAbs(contacts[i].contactNormal.dot(axis)) > 1e-5f)
      return false;
    if (i == 0) {
      supportNormal = contacts[i].contactNormal;
      supportNormal.normalize();
    } else if (physx::PxAbs(
                   contacts[i].contactNormal.dot(supportNormal)) < 0.9999f) {
      return false;
    }
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

  if (!addFrictionlessBodyVsStaticContactRows(
          bodies, numBodies, contacts, numContacts, config, invDt2,
          rows, gradient, preconditioner))
    return false;

  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 previousRA =
      bodyA.prevRotation.rotate(joint.anchorA);
  const physx::PxVec3 previousRB =
      bodyB.prevRotation.rotate(joint.anchorB);
  const physx::PxVec3 displacementA =
      (bodyA.position + rA) - (bodyA.prevPosition + previousRA);
  const physx::PxVec3 displacementB =
      (bodyB.position + rB) - (bodyB.prevPosition + previousRB);
  const physx::PxReal violation =
      (displacementB - displacementA).dot(axis) -
      joint.driveLinearVelocity.x * dt;
  physx::PxReal penalty = joint.linearDamping.x / dt;
  if (joint.driveAccelerationFlags == 0x1u) {
    // PhysX acceleration drives scale the force-mode spring coefficient by
    // the reciprocal unit response of the complete row.  This preserves the
    // implicit damping response across endpoint masses while retaining the
    // authored force-valued limit.  The island Hessian supplies the
    // 1/(1 + dt*damping) denominator; applying it here again would count the
    // implicit response twice.
    const physx::PxVec3 angularA = rA.cross(axis);
    const physx::PxVec3 angularB = rB.cross(axis);
    const physx::PxReal unitResponse =
        bodyA.invMass + bodyB.invMass +
        angularA.dot(bodyA.invInertiaWorld * angularA) +
        angularB.dot(bodyB.invInertiaWorld * angularB);
    if (!(unitResponse > 1e-8f) || !PxIsFinite(unitResponse))
      return false;
    penalty /= unitResponse;
  }
  const physx::PxReal limit = joint.driveLinearForce.x;
  const physx::PxReal rawForce = penalty * violation;
  const physx::PxReal force =
      physx::PxClamp(rawForce, -limit, limit);
  const bool saturated = physx::PxAbs(rawForce) >= limit;
  CoupledIslandRow driveRow;
  driveRow.bodyA = bodyAIndex;
  driveRow.bodyB = bodyBIndex;
  driveRow.jacobianA = AvbdVec6(-axis, -rA.cross(axis));
  driveRow.jacobianB = AvbdVec6(axis, rB.cross(axis));
  driveRow.penalty = saturated ? 0.0f : penalty;
  driveRow.force = force;
  addCoupledRow(driveRow, rows, gradient, preconditioner);

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
  const double targetResidual =
      1e-8 * std::max(1.0, initialResidual);
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

  // The drive is translation invariant and every accepted external contact
  // normal is orthogonal to its axis. Project only the accumulated roundoff
  // in the mass-weighted translational Newton step (or its support-tangent
  // subspace); this is the exact summed linear equation, not an additional
  // physical constraint.
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
  if (numContacts > 0)
    translationRoundoff -=
        supportNormal * translationRoundoff.dot(supportNormal);
  const physx::PxVec3 commonRoundoff = translationRoundoff / totalMass;
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    solution[i].linear -= commonRoundoff;

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
//=============================================================================
// Unified 6x6 System Solver with Joints -- True AVBD
//
// Extends solveLocalSystem to accumulate BOTH contact AND joint Jacobians
// into the same Hessian H and gradient g, then solve once:
//
//   H = M/h^2 + sum_contacts(pen * Jc^T * Jc) + sum_joints(pen * Jj^T * Jj)
//   g = (M/h^2)(x - x_tilde) + sum_contacts(f_c * Jc) + sum_joints(f_j * Jj)
//   delta = solve(H, g)
//   x -= delta
//
// Joint Jacobians (for body i being processed):
//   Spherical (3 rows per joint, position only):
//     C_k = (anchorA - anchorB) . e_k
//     Body A: gradPos = +e_k, gradRot = +(r_A x e_k)   [sign convention]
//     Body B: gradPos = -e_k, gradRot = -(r_B x e_k)
//
//   Fixed (6 rows: 3 position + 3 rotation):
//     Position: same as spherical
//     Rotation C_k = rotError . e_k:
//       Body A: gradPos = 0, gradRot = +e_k
//       Body B: gradPos = 0, gradRot = -e_k
//=============================================================================

// Solver with Joint Constraints
//=============================================================================

} // namespace Dy
} // namespace physx
