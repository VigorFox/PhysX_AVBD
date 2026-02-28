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
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "DyAvbdJointSolver.h"
#include "foundation/PxMath.h"

namespace physx {
namespace Dy {

//=============================================================================
// Spherical Joint Solver
//=============================================================================

void processSphericalJointConstraint(AvbdSphericalJointConstraint &joint,
                                     AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     const AvbdSolverConfig &config,
                                     physx::PxReal dt) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  if (isAStatic && isBStatic)
    return;

  static AvbdSolverBody staticBody;
  staticBody.invMass = 0.0f;
  staticBody.invInertiaWorld = physx::PxMat33(physx::PxZero);

  AvbdSolverBody &bodyA = isAStatic ? staticBody : bodies[idxA];
  AvbdSolverBody &bodyB = isBStatic ? staticBody : bodies[idxB];

  // Compute world anchor positions
  physx::PxVec3 worldAnchorA =
      isAStatic ? joint.anchorA
                : (bodyA.position + bodyA.rotation.rotate(joint.anchorA));
  physx::PxVec3 worldAnchorB =
      isBStatic ? joint.anchorB
                : (bodyB.position + bodyB.rotation.rotate(joint.anchorB));

  physx::PxVec3 error = worldAnchorA - worldAnchorB;
  physx::PxReal errorMag = error.magnitude();

  if (errorMag < AvbdConstants::AVBD_POSITION_ERROR_THRESHOLD)
    return;

  physx::PxVec3 rA =
      isAStatic ? physx::PxVec3(0) : bodyA.rotation.rotate(joint.anchorA);
  physx::PxVec3 rB =
      isBStatic ? physx::PxVec3(0) : bodyB.rotation.rotate(joint.anchorB);

  // XPBD fused primal+dual (equivalent to AVBD AL):
  //   alphaHat = 1 / (rho * h^2)
  //   deltaLambda = -(C + alphaHat * lambda) / (w + alphaHat)
  //   lambda += deltaLambda
  //   Both bodies updated simultaneously per constraint.
  const physx::PxReal rho = joint.header.rho;
  const physx::PxReal h2 = dt * dt;
  const physx::PxReal alphaHat = 1.0f / (rho * h2);

  for (int axis = 0; axis < 3; ++axis) {
    physx::PxVec3 n(0.0f);
    n[axis] = 1.0f;

    physx::PxReal C = error[axis];
    if (physx::PxAbs(C) < AvbdConstants::AVBD_POSITION_ERROR_THRESHOLD)
      continue;

    physx::PxReal w = computeEffectiveInverseMass(bodyA, bodyB, rA, rB, n);
    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // XPBD delta lambda (fused primal+dual)
    physx::PxReal deltaLambda =
        -(C + alphaHat * joint.lambda[axis]) / (w + alphaHat);
    deltaLambda = physx::PxClamp(deltaLambda, -config.maxPositionCorrection,
                                 config.maxPositionCorrection);

    // Fused dual: accumulate lambda
    joint.lambda[axis] += deltaLambda;

    physx::PxVec3 impulse = n * deltaLambda;

    if (!isAStatic) {
      bodyA.position += impulse * bodyA.invMass;
      physx::PxVec3 angImpulse = rA.cross(impulse);
      physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      bodyB.position -= impulse * bodyB.invMass;
      physx::PxVec3 angImpulse = rB.cross(impulse);
      physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }

  // Process cone limit if enabled (inequality constraint)
  if (joint.hasConeLimit && joint.coneAngleLimit > 0.0f) {
    physx::PxReal coneViolation =
        joint.computeConeViolation(bodyA.rotation, bodyB.rotation);
    if (coneViolation > 0.0f) {
      physx::PxVec3 worldAxisA = bodyA.rotation.rotate(joint.coneAxisA);
      physx::PxVec3 worldAxisB = bodyB.rotation.rotate(joint.coneAxisA);

      physx::PxVec3 corrAxis = worldAxisA.cross(worldAxisB);
      physx::PxReal corrAxisMag = corrAxis.magnitude();
      if (corrAxisMag > AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD) {
        corrAxis /= corrAxisMag;

        physx::PxReal angularW = 0.0f;
        if (!isAStatic)
          angularW += (bodyA.invInertiaWorld * corrAxis).dot(corrAxis);
        if (!isBStatic)
          angularW += (bodyB.invInertiaWorld * corrAxis).dot(corrAxis);

        if (angularW > AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD) {
          // XPBD for inequality constraint
          physx::PxReal deltaLambda =
              -(coneViolation + alphaHat * joint.coneLambda) /
              (angularW + alphaHat);

          // Inequality clamp: coneLambda <= 0 (only resist exceeding limit)
          physx::PxReal newLambda = joint.coneLambda + deltaLambda;
          newLambda = physx::PxMin(newLambda, 0.0f);
          deltaLambda = newLambda - joint.coneLambda;
          joint.coneLambda = newLambda;

          if (!isAStatic) {
            physx::PxVec3 angDelta =
                bodyA.invInertiaWorld * (corrAxis * (-deltaLambda));
            physx::PxQuat dq(
                angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
            bodyA.rotation = (dq * bodyA.rotation).getNormalized();
          }
          if (!isBStatic) {
            physx::PxVec3 angDelta =
                bodyB.invInertiaWorld * (corrAxis * deltaLambda);
            physx::PxQuat dq(
                angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
            bodyB.rotation = (dq * bodyB.rotation).getNormalized();
          }
        }
      }
    }
  }
}

//=============================================================================
// Fixed Joint Solver
//=============================================================================

void processFixedJointConstraint(AvbdFixedJointConstraint &joint,
                                 AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                 const AvbdSolverConfig &config,
                                 physx::PxReal dt) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  if (isAStatic && isBStatic)
    return;

  if (joint.isBroken)
    return;

  static AvbdSolverBody staticBody;
  staticBody.invMass = 0.0f;
  staticBody.invInertiaWorld = physx::PxMat33(physx::PxZero);
  staticBody.rotation = physx::PxQuat(physx::PxIdentity);
  staticBody.position = physx::PxVec3(0.0f);

  AvbdSolverBody &bodyA = isAStatic ? staticBody : bodies[idxA];
  AvbdSolverBody &bodyB = isBStatic ? staticBody : bodies[idxB];

  const bool useMassWeightedWeld =
      config.enableMassWeightedWeld &&
      joint.header.type == AvbdConstraintType::eJOINT_WELD;

  // XPBD fused primal+dual (equivalent to AVBD AL)
  const physx::PxReal rho = joint.header.rho;
  const physx::PxReal h2 = dt * dt;
  const physx::PxReal alphaHat = 1.0f / (rho * h2);

  // --- Position Constraint (3 DOF) ---
  physx::PxVec3 worldAnchorA =
      isAStatic ? joint.anchorA
                : (bodyA.position + bodyA.rotation.rotate(joint.anchorA));
  physx::PxVec3 worldAnchorB =
      isBStatic ? joint.anchorB
                : (bodyB.position + bodyB.rotation.rotate(joint.anchorB));

  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  physx::PxVec3 rA =
      isAStatic ? physx::PxVec3(0) : bodyA.rotation.rotate(joint.anchorA);
  physx::PxVec3 rB =
      isBStatic ? physx::PxVec3(0) : bodyB.rotation.rotate(joint.anchorB);

  for (int axis = 0; axis < 3; ++axis) {
    physx::PxVec3 n(0.0f);
    n[axis] = 1.0f;

    physx::PxReal C = posError[axis];
    if (physx::PxAbs(C) < AvbdConstants::AVBD_POSITION_ERROR_THRESHOLD)
      continue;

    physx::PxReal w = computeEffectiveInverseMass(bodyA, bodyB, rA, rB, n);
    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // XPBD delta lambda
    physx::PxReal deltaLambda =
        -(C + alphaHat * joint.lambdaPosition[axis]) / (w + alphaHat);
    deltaLambda = physx::PxClamp(deltaLambda, -config.maxPositionCorrection,
                                 config.maxPositionCorrection);

    joint.lambdaPosition[axis] += deltaLambda;

    physx::PxVec3 impulse = n * deltaLambda;

    if (!isAStatic) {
      const physx::PxReal linearScaleA =
          useMassWeightedWeld ? joint.massRatioA : bodyA.invMass;
      const physx::PxReal angularScaleA =
          useMassWeightedWeld ? joint.massRatioA : 1.0f;

      bodyA.position += impulse * linearScaleA;
      physx::PxVec3 angDelta =
          (bodyA.invInertiaWorld * rA.cross(impulse)) * angularScaleA;
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      const physx::PxReal linearScaleB =
          useMassWeightedWeld ? joint.massRatioB : bodyB.invMass;
      const physx::PxReal angularScaleB =
          useMassWeightedWeld ? joint.massRatioB : 1.0f;

      bodyB.position -= impulse * linearScaleB;
      physx::PxVec3 angDelta =
          (bodyB.invInertiaWorld * rB.cross(impulse)) * angularScaleB;
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }

  // --- Rotation Constraint (3 DOF) ---
  physx::PxVec3 rotError =
      joint.computeRotationViolation(bodyA.rotation, bodyB.rotation);

  for (int axis = 0; axis < 3; ++axis) {
    physx::PxVec3 n(0.0f);
    n[axis] = 1.0f;

    physx::PxReal C = rotError[axis];
    if (physx::PxAbs(C) < AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD)
      continue;

    physx::PxReal w = 0.0f;
    if (!isAStatic)
      w += (bodyA.invInertiaWorld * n).dot(n);
    if (!isBStatic)
      w += (bodyB.invInertiaWorld * n).dot(n);

    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // XPBD delta lambda
    physx::PxReal deltaLambda =
        -(C + alphaHat * joint.lambdaRotation[axis]) / (w + alphaHat);
    deltaLambda = physx::PxClamp(deltaLambda, -config.maxAngularCorrection,
                                 config.maxAngularCorrection);

    joint.lambdaRotation[axis] += deltaLambda;

    physx::PxVec3 angImpulse = n * deltaLambda;

    if (!isAStatic) {
      const physx::PxReal angularScaleA =
          useMassWeightedWeld ? joint.massRatioA : 1.0f;
      physx::PxVec3 angDelta =
          (bodyA.invInertiaWorld * angImpulse) * angularScaleA;
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      const physx::PxReal angularScaleB =
          useMassWeightedWeld ? joint.massRatioB : 1.0f;
      physx::PxVec3 angDelta =
          (bodyB.invInertiaWorld * angImpulse) * angularScaleB;
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }
}

//=============================================================================
// Revolute Joint Solver
//=============================================================================

void processRevoluteJointConstraint(AvbdRevoluteJointConstraint &joint,
                                    AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    const AvbdSolverConfig &config,
                                    physx::PxReal dt) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  if (isAStatic && isBStatic)
    return;

  static AvbdSolverBody staticBody;
  staticBody.invMass = 0.0f;
  staticBody.invInertiaWorld = physx::PxMat33(physx::PxZero);
  staticBody.rotation = physx::PxQuat(physx::PxIdentity);
  staticBody.position = physx::PxVec3(0.0f);

  AvbdSolverBody &bodyA = isAStatic ? staticBody : bodies[idxA];
  AvbdSolverBody &bodyB = isBStatic ? staticBody : bodies[idxB];

  // AVBD: Process using augmented Lagrangian energy minimization
  physx::PxReal rho = joint.header.rho;

  // --- Position Constraint (3 DOF) ---
  physx::PxVec3 worldAnchorA =
      isAStatic ? joint.anchorA
                : (bodyA.position + bodyA.rotation.rotate(joint.anchorA));
  physx::PxVec3 worldAnchorB =
      isBStatic ? joint.anchorB
                : (bodyB.position + bodyB.rotation.rotate(joint.anchorB));

  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  physx::PxVec3 rA =
      isAStatic ? physx::PxVec3(0) : bodyA.rotation.rotate(joint.anchorA);
  physx::PxVec3 rB =
      isBStatic ? physx::PxVec3(0) : bodyB.rotation.rotate(joint.anchorB);

  for (int axis = 0; axis < 3; ++axis) {
    physx::PxVec3 n(0.0f);
    n[axis] = 1.0f;

    physx::PxReal C = posError[axis];
    if (physx::PxAbs(C) < AvbdConstants::AVBD_POSITION_ERROR_THRESHOLD)
      continue;

    physx::PxReal w = computeEffectiveInverseMass(bodyA, bodyB, rA, rB, n);
    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // AVBD: Compute gradient of augmented Lagrangian energy
    physx::PxReal gradient = joint.lambdaPosition[axis] + rho * C;
    physx::PxReal stepSize = -gradient / w;
    stepSize = physx::PxClamp(stepSize, -config.maxPositionCorrection,
                              config.maxPositionCorrection);

    physx::PxVec3 impulse = n * stepSize;

    if (!isAStatic) {
      bodyA.position += impulse * bodyA.invMass;
      physx::PxVec3 angDelta = bodyA.invInertiaWorld * rA.cross(impulse);
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      bodyB.position -= impulse * bodyB.invMass;
      physx::PxVec3 angDelta = bodyB.invInertiaWorld * rB.cross(impulse);
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }

  // --- Axis Alignment Constraint (2 DOF) ---
  physx::PxVec3 worldAxisA = bodyA.rotation.rotate(joint.axisA);
  physx::PxVec3 worldAxisB = bodyB.rotation.rotate(joint.axisB);

  // The cross product of the two axes gives the alignment error
  physx::PxVec3 axisError = worldAxisA.cross(worldAxisB);

  // We need two perpendicular correction directions
  // Build a basis perpendicular to the joint axis
  physx::PxVec3 perp1, perp2;
  if (physx::PxAbs(worldAxisA.x) <
      AvbdConstants::AVBD_AXIS_SELECTION_THRESHOLD) {
    perp1 = worldAxisA.cross(physx::PxVec3(1, 0, 0)).getNormalized();
  } else {
    perp1 = worldAxisA.cross(physx::PxVec3(0, 1, 0)).getNormalized();
  }
  perp2 = worldAxisA.cross(perp1).getNormalized();

  // Project axis error onto the two perpendicular directions
  physx::PxReal err1 = axisError.dot(perp1);
  physx::PxReal err2 = axisError.dot(perp2);

  // Solve for each perpendicular direction
  for (int i = 0; i < 2; ++i) {
    physx::PxVec3 corrAxis = (i == 0) ? perp1 : perp2;
    physx::PxReal C = (i == 0) ? err1 : err2;

    if (physx::PxAbs(C) < AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD)
      continue;

    // Angular effective mass
    physx::PxReal w = 0.0f;
    if (!isAStatic)
      w += (bodyA.invInertiaWorld * corrAxis).dot(corrAxis);
    if (!isBStatic)
      w += (bodyB.invInertiaWorld * corrAxis).dot(corrAxis);

    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // AVBD: Compute gradient of augmented Lagrangian energy
    physx::PxReal gradient = joint.lambdaAxisAlign[i] + rho * C;
    physx::PxReal stepSize = -gradient / w;
    stepSize = physx::PxClamp(stepSize, -config.maxAngularCorrection,
                              config.maxAngularCorrection);

    physx::PxVec3 angImpulse = corrAxis * stepSize;

    if (!isAStatic) {
      physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }

  // --- Angle Limits ---
  if (joint.hasAngleLimit) {
    physx::PxReal angleLimitViolation =
        joint.computeAngleLimitViolation(bodyA.rotation, bodyB.rotation);

    if (physx::PxAbs(angleLimitViolation) >
        AvbdConstants::AVBD_ANGLE_LIMIT_THRESHOLD) {
      // Apply correction around the joint axis
      physx::PxVec3 corrAxis = worldAxisA;

      physx::PxReal w = 0.0f;
      if (!isAStatic)
        w += (bodyA.invInertiaWorld * corrAxis).dot(corrAxis);
      if (!isBStatic)
        w += (bodyB.invInertiaWorld * corrAxis).dot(corrAxis);

      if (w > AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD) {
        // AVBD: Compute gradient of augmented Lagrangian energy
        physx::PxReal gradient =
            joint.lambdaAngleLimit + rho * angleLimitViolation;
        physx::PxReal stepSize = -gradient / w;
        stepSize = physx::PxClamp(stepSize, -config.maxAngularCorrection,
                                  config.maxAngularCorrection);

        physx::PxVec3 angImpulse = corrAxis * stepSize;

        if (!isAStatic) {
          physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
          physx::PxQuat dq(
              angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
              angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
              angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
          bodyA.rotation = (dq * bodyA.rotation).getNormalized();
        }

        if (!isBStatic) {
          physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
          physx::PxQuat dq(
              -angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
              -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
              -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
          bodyB.rotation = (dq * bodyB.rotation).getNormalized();
        }
      }
    }
  }

  // --- Motor Drive ---
  if (joint.motorEnabled && joint.motorMaxForce > 0.0f) {
    // Compute current angular velocity difference around joint axis
    physx::PxReal relAngVelOnAxis = 0.0f;
    if (!isAStatic)
      relAngVelOnAxis += bodyA.angularVelocity.dot(worldAxisA);
    if (!isBStatic)
      relAngVelOnAxis -= bodyB.angularVelocity.dot(worldAxisA);

    physx::PxReal velocityError = joint.motorTargetVelocity - relAngVelOnAxis;

    // Simple velocity-level motor with max force limit
    physx::PxReal motorImpulse =
        velocityError * AvbdConstants::AVBD_MOTOR_GAIN; // Simple gain
    motorImpulse = physx::PxClamp(motorImpulse, -joint.motorMaxForce * dt,
                                  joint.motorMaxForce * dt);

    physx::PxVec3 motorAngImpulse = worldAxisA * motorImpulse;

    if (!isAStatic) {
      bodyA.angularVelocity += bodyA.invInertiaWorld * motorAngImpulse;
    }
    if (!isBStatic) {
      bodyB.angularVelocity -= bodyB.invInertiaWorld * motorAngImpulse;
    }
  }
}

//=============================================================================
// Lagrangian Multiplier Updates
//=============================================================================

void updateSphericalJointMultiplier(AvbdSphericalJointConstraint &joint,
                                    const AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    const AvbdSolverConfig &config) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  // Use consistent static body check with processSphericalJointConstraint
  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  // Compute world anchors directly -- do NOT use computeViolation() because
  // for static bodies anchorA/B is already a world position, and
  // computeViolation() would double-count it (posA + rotate(anchorA)).
  physx::PxVec3 worldAnchorA =
      isAStatic
          ? joint.anchorA
          : bodies[idxA].position + bodies[idxA].rotation.rotate(joint.anchorA);
  physx::PxVec3 worldAnchorB =
      isBStatic
          ? joint.anchorB
          : bodies[idxB].position + bodies[idxB].rotation.rotate(joint.anchorB);
  physx::PxVec3 violation = worldAnchorA - worldAnchorB;

  // Augmented Lagrangian update: lambda += rho * C(x)
  physx::PxReal rho = joint.header.rho;
  joint.lambda += violation * rho;

  // Clamp lambda to prevent explosion
  physx::PxReal maxLambda = config.maxLambda;
  for (int i = 0; i < 3; ++i) {
    joint.lambda[i] = physx::PxClamp(joint.lambda[i], -maxLambda, maxLambda);
  }

  if (joint.hasConeLimit && joint.coneAngleLimit > 0.0f) {
    physx::PxQuat rotA =
        isAStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxA].rotation;
    physx::PxQuat rotB =
        isBStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxB].rotation;
    physx::PxReal coneViolation = joint.computeConeViolation(rotA, rotB);

    // coneLambda decreases -> magnitude increases
    joint.coneLambda -= coneViolation * rho;
    joint.coneLambda = physx::PxMax(joint.coneLambda, -maxLambda);
    joint.coneLambda = physx::PxMin(joint.coneLambda, 0.0f);
  }
}

void updateFixedJointMultiplier(AvbdFixedJointConstraint &joint,
                                const AvbdSolverBody *bodies,
                                physx::PxU32 numBodies,
                                const AvbdSolverConfig &config) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  // Use consistent static body check with processFixedJointConstraint
  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  if (joint.isBroken)
    return;

  // Compute world anchors directly -- do NOT use computePositionViolation()
  // because for static bodies anchorA/B is already a world position, and
  // computePositionViolation() would double-count it (posA + rotate(anchorA)).
  physx::PxVec3 worldAnchorA =
      isAStatic
          ? joint.anchorA
          : bodies[idxA].position + bodies[idxA].rotation.rotate(joint.anchorA);
  physx::PxVec3 worldAnchorB =
      isBStatic
          ? joint.anchorB
          : bodies[idxB].position + bodies[idxB].rotation.rotate(joint.anchorB);
  physx::PxVec3 posViolation = worldAnchorA - worldAnchorB;

  physx::PxQuat rotA =
      isAStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxA].rotation;
  physx::PxQuat rotB =
      isBStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxB].rotation;
  physx::PxVec3 rotViolation = joint.computeRotationViolation(rotA, rotB);

  physx::PxReal rho = joint.header.rho;
  joint.lambdaPosition += posViolation * rho;
  joint.lambdaRotation += rotViolation * rho;

  physx::PxReal maxLambda = config.maxLambda;
  for (int i = 0; i < 3; ++i) {
    joint.lambdaPosition[i] =
        physx::PxClamp(joint.lambdaPosition[i], -maxLambda, maxLambda);
    joint.lambdaRotation[i] =
        physx::PxClamp(joint.lambdaRotation[i], -maxLambda, maxLambda);
  }

  if (joint.isBreakable && !joint.isBroken) {
    // Break detection: use the CURRENT STEP constraint force (rho * violation),
    // NOT the accumulated lambda. Lambda grows unboundedly and would falsely
    // trigger breakage even for gentle constraint forces.
    const physx::PxReal forceThisStep = (posViolation * rho).magnitude();
    const physx::PxReal torqueThisStep = (rotViolation * rho).magnitude();
    if (joint.checkBreak(forceThisStep, torqueThisStep)) {
      joint.lambdaPosition = physx::PxVec3(0.0f);
      joint.lambdaRotation = physx::PxVec3(0.0f);
    }
  }
}

void updateRevoluteJointMultiplier(AvbdRevoluteJointConstraint &joint,
                                   const AvbdSolverBody *bodies,
                                   physx::PxU32 numBodies,
                                   const AvbdSolverConfig &config) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  // Use consistent static body check with processRevoluteJointConstraint
  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  physx::PxVec3 worldAnchorA =
      isAStatic
          ? joint.anchorA
          : bodies[idxA].position + bodies[idxA].rotation.rotate(joint.anchorA);
  physx::PxVec3 worldAnchorB =
      isBStatic
          ? joint.anchorB
          : bodies[idxB].position + bodies[idxB].rotation.rotate(joint.anchorB);
  physx::PxVec3 posViolation = worldAnchorA - worldAnchorB;

  physx::PxQuat rotA =
      isAStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxA].rotation;
  physx::PxQuat rotB =
      isBStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxB].rotation;
  physx::PxVec3 axisViolation = joint.computeAxisViolation(rotA, rotB);

  physx::PxReal rho = joint.header.rho;
  joint.lambdaPosition += posViolation * rho;

  // Project axis violation onto our two perpendicular directions
  // For simplicity, we store the full cross product result
  joint.lambdaAxisAlign += axisViolation * rho;

  physx::PxReal maxLambda = config.maxLambda;
  for (int i = 0; i < 3; ++i) {
    joint.lambdaPosition[i] =
        physx::PxClamp(joint.lambdaPosition[i], -maxLambda, maxLambda);
    joint.lambdaAxisAlign[i] =
        physx::PxClamp(joint.lambdaAxisAlign[i], -maxLambda, maxLambda);
  }
}

//=============================================================================
// Prismatic Joint Solver
//=============================================================================

void processPrismaticJointConstraint(AvbdPrismaticJointConstraint &joint,
                                     AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     const AvbdSolverConfig &config,
                                     physx::PxReal dt) {
  PX_UNUSED(dt);
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  if (isAStatic && isBStatic)
    return;

  static AvbdSolverBody staticBody;
  staticBody.invMass = 0.0f;
  staticBody.invInertiaWorld = physx::PxMat33(physx::PxZero);
  staticBody.rotation = physx::PxQuat(physx::PxIdentity);
  staticBody.position = physx::PxVec3(0.0f);

  AvbdSolverBody &bodyA = isAStatic ? staticBody : bodies[idxA];
  AvbdSolverBody &bodyB = isBStatic ? staticBody : bodies[idxB];

  // AVBD: Process using augmented Lagrangian energy minimization
  physx::PxReal rho = joint.header.rho;

  // Compute world space anchor positions
  physx::PxVec3 worldAnchorA =
      isAStatic ? joint.anchorA
                : (bodyA.position + bodyA.rotation.rotate(joint.anchorA));
  physx::PxVec3 worldAnchorB =
      isBStatic ? joint.anchorB
                : (bodyB.position + bodyB.rotation.rotate(joint.anchorB));

  // Compute world space slide axis
  physx::PxVec3 worldAxis = joint.getWorldAxis(bodyA.rotation);

  // Build perpendicular basis to the slide axis
  physx::PxVec3 perp1, perp2;
  if (physx::PxAbs(worldAxis.x) < 0.9f) {
    perp1 = worldAxis.cross(physx::PxVec3(1, 0, 0)).getNormalized();
  } else {
    perp1 = worldAxis.cross(physx::PxVec3(0, 1, 0)).getNormalized();
  }
  perp2 = worldAxis.cross(perp1).getNormalized();

  // Compute anchor offsets from body centers (world space)
  physx::PxVec3 rA =
      isAStatic ? physx::PxVec3(0) : bodyA.rotation.rotate(joint.anchorA);
  physx::PxVec3 rB =
      isBStatic ? physx::PxVec3(0) : bodyB.rotation.rotate(joint.anchorB);

  // --- Position Constraint (2 DOF - perpendicular to slide axis) ---
  physx::PxVec3 posDiff = worldAnchorB - worldAnchorA;

  for (int i = 0; i < 2; ++i) {
    physx::PxVec3 n = (i == 0) ? perp1 : perp2;
    physx::PxReal C = posDiff.dot(n);

    if (physx::PxAbs(C) < AvbdConstants::AVBD_POSITION_ERROR_THRESHOLD)
      continue;

    // Compute effective inverse mass along this direction
    physx::PxReal w = computeEffectiveInverseMass(bodyA, bodyB, rA, rB, n);

    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // AVBD: Compute gradient of augmented Lagrangian energy
    physx::PxReal gradient = joint.lambdaPosition[i] + rho * C;
    physx::PxReal stepSize = -gradient / w;
    stepSize = physx::PxClamp(stepSize, -config.maxPositionCorrection,
                              config.maxPositionCorrection);

    // Compute position corrections
    physx::PxVec3 impulse = n * stepSize;

    if (!isAStatic) {
      bodyA.position += impulse * bodyA.invMass;
      physx::PxVec3 angImpulse = rA.cross(impulse);
      physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      bodyB.position -= impulse * bodyB.invMass;
      physx::PxVec3 angImpulse = rB.cross(impulse);
      physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }

  // --- Rotation Constraint (3 DOF - lock relative rotation) ---
  physx::PxVec3 rotError =
      joint.computeRotationViolation(bodyA.rotation, bodyB.rotation);

  for (int axis = 0; axis < 3; ++axis) {
    physx::PxVec3 n(0.0f);
    n[axis] = 1.0f;

    physx::PxReal C = rotError[axis];
    if (physx::PxAbs(C) < AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD)
      continue;

    // Angular effective mass
    physx::PxReal w = 0.0f;
    if (!isAStatic)
      w += (bodyA.invInertiaWorld * n).dot(n);
    if (!isBStatic)
      w += (bodyB.invInertiaWorld * n).dot(n);

    if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
      continue;

    // AVBD: Compute gradient of augmented Lagrangian energy
    physx::PxReal gradient = joint.lambdaRotation[axis] + rho * C;
    physx::PxReal stepSize = -gradient / w;
    stepSize = physx::PxClamp(stepSize, -config.maxAngularCorrection,
                              config.maxAngularCorrection);

    physx::PxVec3 angImpulse = n * stepSize;

    if (!isAStatic) {
      physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
      physx::PxQuat dq(angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyA.rotation = (dq * bodyA.rotation).getNormalized();
    }

    if (!isBStatic) {
      physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
      physx::PxQuat dq(-angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
                       1.0f);
      bodyB.rotation = (dq * bodyB.rotation).getNormalized();
    }
  }
}

//=============================================================================
// Prismatic Joint Multiplier Update
//=============================================================================

void updatePrismaticJointMultiplier(AvbdPrismaticJointConstraint &joint,
                                    const AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    const AvbdSolverConfig &config) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  // Use consistent static body check with processPrismaticJointConstraint
  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  // Compute world anchors directly -- for static bodies anchorA/B is already
  // a world position; do NOT pass it through computePositionViolation() which
  // would add rotate(anchor) again, double-counting the offset.
  physx::PxVec3 worldAnchorA =
      isAStatic
          ? joint.anchorA
          : bodies[idxA].position + bodies[idxA].rotation.rotate(joint.anchorA);
  physx::PxVec3 worldAnchorB =
      isBStatic
          ? joint.anchorB
          : bodies[idxB].position + bodies[idxB].rotation.rotate(joint.anchorB);
  physx::PxVec3 posDiff = worldAnchorB - worldAnchorA;

  physx::PxQuat rotA =
      isAStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxA].rotation;
  physx::PxQuat rotB =
      isBStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxB].rotation;

  physx::PxVec3 worldAxis = rotA.rotate(joint.axisA);

  // Build perpendicular basis
  physx::PxVec3 perp1, perp2;
  if (physx::PxAbs(worldAxis.x) < 0.9f) {
    perp1 = worldAxis.cross(physx::PxVec3(1, 0, 0)).getNormalized();
  } else {
    perp1 = worldAxis.cross(physx::PxVec3(0, 1, 0)).getNormalized();
  }
  perp2 = worldAxis.cross(perp1).getNormalized();

  // Project position error onto perpendicular directions
  physx::PxVec3 posViolation(0.0f);
  posViolation.x = posDiff.dot(perp1);
  posViolation.y = posDiff.dot(perp2);

  // Compute rotation violation
  physx::PxVec3 rotViolation = joint.computeRotationViolation(rotA, rotB);

  // Compute limit violation
  physx::PxReal limitViolation = 0.0f;
  if (joint.hasLimit) {
    // Compute slide position directly from already-correct world anchors
    physx::PxVec3 diff = worldAnchorB - worldAnchorA;
    physx::PxReal slidePos = diff.dot(worldAxis);
    limitViolation = joint.computeLimitViolation(slidePos);
  }

  // Augmented Lagrangian update: lambda += rho * C(x)
  physx::PxReal rho = joint.header.rho;
  joint.lambdaPosition += posViolation * rho;
  joint.lambdaRotation += rotViolation * rho;
  joint.lambdaLimit += limitViolation * rho;

  // Clamp lambda to prevent explosion
  physx::PxReal maxLambda = config.maxLambda;
  for (int i = 0; i < 3; ++i) {
    joint.lambdaPosition[i] =
        physx::PxClamp(joint.lambdaPosition[i], -maxLambda, maxLambda);
    joint.lambdaRotation[i] =
        physx::PxClamp(joint.lambdaRotation[i], -maxLambda, maxLambda);
  }
  joint.lambdaLimit = physx::PxClamp(joint.lambdaLimit, -maxLambda, maxLambda);
}

//=============================================================================
// D6 Joint Solver
//=============================================================================

void processD6JointConstraint(AvbdD6JointConstraint &joint,
                              AvbdSolverBody *bodies, physx::PxU32 numBodies,
                              const AvbdSolverConfig &config,
                              physx::PxReal dt) {
  PX_UNUSED(dt);
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  if (isAStatic && isBStatic)
    return;

  static AvbdSolverBody staticBody;
  staticBody.invMass = 0.0f;
  staticBody.invInertiaWorld = physx::PxMat33(physx::PxZero);
  staticBody.rotation = physx::PxQuat(physx::PxIdentity);
  staticBody.position = physx::PxVec3(0.0f);

  AvbdSolverBody &bodyA = isAStatic ? staticBody : bodies[idxA];
  AvbdSolverBody &bodyB = isBStatic ? staticBody : bodies[idxB];

  // AVBD: Process using augmented Lagrangian energy minimization
  physx::PxReal rho = joint.header.rho;

  // Compute anchor offsets from body centers (world space)
  physx::PxVec3 rA =
      isAStatic ? physx::PxVec3(0) : bodyA.rotation.rotate(joint.anchorA);
  physx::PxVec3 rB =
      isBStatic ? physx::PxVec3(0) : bodyB.rotation.rotate(joint.anchorB);

  // --- Process Linear DOFs (3 axes) ---
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    physx::PxU32 motionType = joint.getLinearMotion(axis);

    if (motionType == 2) // FREE
      continue;

    // Compute constraint direction in world space
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    physx::PxVec3 worldAxis = bodyA.rotation.rotate(localAxis);

    // Compute linear error
    physx::PxReal error = joint.computeLinearError(
        bodyA.position, bodyA.rotation, bodyB.position, bodyB.rotation, axis);

    if (motionType == 0) { // LOCKED
      if (physx::PxAbs(error) < AvbdConstants::AVBD_POSITION_ERROR_THRESHOLD)
        continue;

      // Compute effective inverse mass
      physx::PxReal w =
          computeEffectiveInverseMass(bodyA, bodyB, rA, rB, worldAxis);

      if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
        continue;

      // AVBD: Compute gradient of augmented Lagrangian energy
      physx::PxReal gradient = joint.lambdaLinear[axis] + rho * error;
      physx::PxReal stepSize = -gradient / w;
      stepSize = physx::PxClamp(stepSize, -config.maxPositionCorrection,
                                config.maxPositionCorrection);

      physx::PxVec3 impulse = worldAxis * stepSize;

      if (!isAStatic) {
        bodyA.position += impulse * bodyA.invMass;
        physx::PxVec3 angImpulse = rA.cross(impulse);
        physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyA.rotation = (dq * bodyA.rotation).getNormalized();
      }

      if (!isBStatic) {
        bodyB.position -= impulse * bodyB.invMass;
        physx::PxVec3 angImpulse = rB.cross(impulse);
        physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            -angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyB.rotation = (dq * bodyB.rotation).getNormalized();
      }
    } else if (motionType == 1) { // LIMITED
      physx::PxReal limitViolation =
          joint.computeLinearLimitViolation(error, axis);

      if (physx::PxAbs(limitViolation) <
          AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD)
        continue;

      // Compute effective inverse mass
      physx::PxReal w =
          computeEffectiveInverseMass(bodyA, bodyB, rA, rB, worldAxis);

      if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
        continue;

      // AVBD: Compute gradient of augmented Lagrangian energy
      physx::PxReal gradient = joint.lambdaLinear[axis] + rho * limitViolation;
      physx::PxReal stepSize = -gradient / w;
      stepSize = physx::PxClamp(stepSize, -config.maxPositionCorrection,
                                config.maxPositionCorrection);

      physx::PxVec3 impulse = worldAxis * stepSize;

      if (!isAStatic) {
        bodyA.position += impulse * bodyA.invMass;
        physx::PxVec3 angImpulse = rA.cross(impulse);
        physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyA.rotation = (dq * bodyA.rotation).getNormalized();
      }

      if (!isBStatic) {
        bodyB.position -= impulse * bodyB.invMass;
        physx::PxVec3 angImpulse = rB.cross(impulse);
        physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            -angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyB.rotation = (dq * bodyB.rotation).getNormalized();
      }
    }
  }

  // --- Process Angular DOFs (3 axes) ---
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    physx::PxU32 motionType = joint.getAngularMotion(axis);

    if (motionType == 2) // FREE
      continue;

    // Compute constraint direction in world space
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    physx::PxVec3 worldAxis = bodyA.rotation.rotate(localAxis);

    // Compute angular error
    physx::PxReal error =
        joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis);

    if (motionType == 0) { // LOCKED
      if (physx::PxAbs(error) < AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD)
        continue;

      // Angular effective mass
      physx::PxReal w = 0.0f;
      if (!isAStatic)
        w += (bodyA.invInertiaWorld * worldAxis).dot(worldAxis);
      if (!isBStatic)
        w += (bodyB.invInertiaWorld * worldAxis).dot(worldAxis);

      if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
        continue;

      // AVBD: Compute gradient of augmented Lagrangian energy
      physx::PxReal gradient = joint.lambdaAngular[axis] + rho * error;
      physx::PxReal stepSize = -gradient / w;
      stepSize = physx::PxClamp(stepSize, -config.maxAngularCorrection,
                                config.maxAngularCorrection);

      physx::PxVec3 angImpulse = worldAxis * stepSize;

      if (!isAStatic) {
        physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyA.rotation = (dq * bodyA.rotation).getNormalized();
      }

      if (!isBStatic) {
        physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            -angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyB.rotation = (dq * bodyB.rotation).getNormalized();
      }
    } else if (motionType == 1) { // LIMITED
      physx::PxReal limitViolation =
          joint.computeAngularLimitViolation(error, axis);

      if (physx::PxAbs(limitViolation) <
          AvbdConstants::AVBD_ROTATION_ERROR_THRESHOLD)
        continue;

      // Angular effective mass
      physx::PxReal w = 0.0f;
      if (!isAStatic)
        w += (bodyA.invInertiaWorld * worldAxis).dot(worldAxis);
      if (!isBStatic)
        w += (bodyB.invInertiaWorld * worldAxis).dot(worldAxis);

      if (w < AvbdConstants::AVBD_INFINITE_MASS_THRESHOLD)
        continue;

      // AVBD: Compute gradient of augmented Lagrangian energy
      physx::PxReal gradient = joint.lambdaAngular[axis] + rho * limitViolation;
      physx::PxReal stepSize = -gradient / w;
      stepSize = physx::PxClamp(stepSize, -config.maxAngularCorrection,
                                config.maxAngularCorrection);

      physx::PxVec3 angImpulse = worldAxis * stepSize;

      if (!isAStatic) {
        physx::PxVec3 angDelta = bodyA.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyA.rotation = (dq * bodyA.rotation).getNormalized();
      }

      if (!isBStatic) {
        physx::PxVec3 angDelta = bodyB.invInertiaWorld * angImpulse;
        physx::PxQuat dq(
            -angDelta.x * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.y * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR,
            -angDelta.z * AvbdConstants::AVBD_QUATERNION_HALF_FACTOR, 1.0f);
        bodyB.rotation = (dq * bodyB.rotation).getNormalized();
      }
    }
  }
}

//=============================================================================
// D6 Joint Multiplier Update
//=============================================================================

void updateD6JointMultiplier(AvbdD6JointConstraint &joint,
                             const AvbdSolverBody *bodies,
                             physx::PxU32 numBodies,
                             const AvbdSolverConfig &config) {
  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  // Use consistent static body check with processD6JointConstraint
  const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
  const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

  // Compute world anchors directly -- for static bodies anchorA/B is already
  // a world position; do NOT pass through computeLinearError() which
  // would add rotate(anchor) again, double-counting the offset.
  physx::PxVec3 worldAnchorA =
      isAStatic
          ? joint.anchorA
          : bodies[idxA].position + bodies[idxA].rotation.rotate(joint.anchorA);
  physx::PxVec3 worldAnchorB =
      isBStatic
          ? joint.anchorB
          : bodies[idxB].position + bodies[idxB].rotation.rotate(joint.anchorB);
  physx::PxQuat rotA =
      isAStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxA].rotation;
  physx::PxQuat rotB =
      isBStatic ? physx::PxQuat(physx::PxIdentity) : bodies[idxB].rotation;

  // Update linear multipliers for locked DOFs
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    if (joint.getLinearMotion(axis) == 0) { // LOCKED
      // Inline the linear error computation with correct world anchors
      physx::PxVec3 diff = worldAnchorB - worldAnchorA;
      physx::PxVec3 localDiff = rotA.rotateInv(diff);
      physx::PxReal error = localDiff[axis];
      physx::PxReal rho = joint.header.rho;
      joint.lambdaLinear[axis] += error * rho;
    }
  }

  // Update angular multipliers for locked DOFs
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    if (joint.getAngularMotion(axis) == 0) { // LOCKED
      physx::PxReal error = joint.computeAngularError(rotA, rotB, axis);
      physx::PxReal rho = joint.header.rho;
      joint.lambdaAngular[axis] += error * rho;
    }
  }

  // Clamp lambda to prevent explosion
  physx::PxReal maxLambda = config.maxLambda;
  for (int i = 0; i < 3; ++i) {
    joint.lambdaLinear[i] =
        physx::PxClamp(joint.lambdaLinear[i], -maxLambda, maxLambda);
    joint.lambdaAngular[i] =
        physx::PxClamp(joint.lambdaAngular[i], -maxLambda, maxLambda);
  }
}

//=============================================================================
// Gear Joint Multiplier Update
//=============================================================================

void updateGearJointMultiplier(AvbdGearJointConstraint &joint,
                               const AvbdSolverBody *bodies,
                               physx::PxU32 numBodies,
                               const AvbdSolverConfig &config) {
  PX_UNUSED(config);

  const physx::PxU32 idxA = joint.header.bodyIndexA;
  const physx::PxU32 idxB = joint.header.bodyIndexB;

  // Both bodies must be present and dynamic
  if (idxA >= numBodies || idxB >= numBodies)
    return;

  const AvbdSolverBody &bodyA = bodies[idxA];
  const AvbdSolverBody &bodyB = bodies[idxB];

  if (bodyA.invMass <= 0.f || bodyB.invMass <= 0.f)
    return;

  physx::PxVec3 worldAxis0 = bodyA.rotation.rotate(joint.gearAxis0);
  physx::PxVec3 worldAxis1 = bodyB.rotation.rotate(joint.gearAxis1);

  // Measure angular displacement from prevRotation (pre integration pose).
  // This captures the full velocity of the bodies to enforce the gear ratio.
  auto computeDeltaW = [](const AvbdSolverBody &b,
                          const physx::PxVec3 &axis) -> physx::PxReal {
    physx::PxQuat dq = b.rotation * b.prevRotation.getConjugate();
    if (dq.w < 0.0f)
      dq = -dq;
    return physx::PxVec3(dq.x, dq.y, dq.z).dot(axis) * 2.0f;
  };

  physx::PxReal dwA = computeDeltaW(bodyA, worldAxis0);
  physx::PxReal dwB = computeDeltaW(bodyB, worldAxis1);

  // Velocity-level C, matching the primal. Cross-frame error must be included
  // so the dual multiplier absorbs the static physical error to prevent drift.
  // Using the exact same formula: J*dw + bias
  physx::PxReal C = dwA * joint.gearRatio + dwB + joint.geometricError;
  // --- ADMM-safe dual step using axial inertia (not mass) ---
  // For a pure angular constraint, the correct effective stiffness is based
  // on the axial moment of inertia: I_axial = 1 / (axis^T * invI * axis)
  const physx::PxVec3 tmpA = bodyA.invInertiaWorld.transform(worldAxis0);
  const physx::PxReal invIA = worldAxis0.dot(tmpA);
  const physx::PxReal IA = (invIA > 1e-10f) ? (1.0f / invIA) : 0.0f;

  const physx::PxVec3 tmpB = bodyB.invInertiaWorld.transform(worldAxis1);
  const physx::PxReal invIB = worldAxis1.dot(tmpB);
  const physx::PxReal IB = (invIB > 1e-10f) ? (1.0f / invIB) : 0.0f;

  // Effective axial inertia (minimum -?the softer body limits the dual step)
  const physx::PxReal Ieff =
      (IA > 0.f && IB > 0.f) ? physx::PxMin(IA, IB) : physx::PxMax(IA, IB);
  physx::PxReal rho = joint.header.rho;
  physx::PxReal Mh2 = Ieff * rho;
  physx::PxReal admm_step = rho * rho / (rho + Mh2);
  physx::PxReal rhoDual = physx::PxMin(Mh2, admm_step);

  // --- Leaky integrator dual update (decay = 0.99) ---
  const physx::PxReal decay = 0.99f;
  joint.lambdaGear = joint.lambdaGear * decay + rhoDual * C;
}

} // namespace Dy
} // namespace physx
