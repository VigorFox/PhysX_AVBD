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

#include "DyAvbdSolver.h"
#include "DyAvbdJointProjection.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "PxAvbdParallelFor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// Enable detailed joint solver diagnostics (first N frames)
#ifndef AVBD_JOINT_DEBUG
#define AVBD_JOINT_DEBUG 0
#endif
#ifndef AVBD_JOINT_DEBUG_FRAMES
#define AVBD_JOINT_DEBUG_FRAMES 200
#endif

// External frame counter from DyAvbdDynamics.cpp (used by motor drives)
extern physx::PxU64 getAvbdMotorFrameCounter();

static physx::PxU32 s_avbdJointDebugFrame = 0;

namespace physx {
namespace Dy {

namespace {
static physx::PxReal computeRotationDeltaMagnitude(const physx::PxQuat& current,
                                                   const physx::PxQuat& previous) {
  physx::PxQuat deltaQ = current * previous.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  return 2.0f * physx::PxSqrt(deltaQ.x * deltaQ.x + deltaQ.y * deltaQ.y +
                              deltaQ.z * deltaQ.z);
}

static void computeMaxPoseDeltas(const AvbdSolverBody* bodies,
                                 physx::PxU32 numBodies,
                                 const physx::PxArray<physx::PxVec3>& prevPos,
                                 const physx::PxArray<physx::PxQuat>& prevRot,
                                 physx::PxReal& maxPositionDelta,
                                 physx::PxReal& maxRotationDelta) {
  maxPositionDelta = 0.0f;
  maxRotationDelta = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;

    maxPositionDelta = physx::PxMax(maxPositionDelta,
      (bodies[i].position - prevPos[i]).magnitude());
    maxRotationDelta = physx::PxMax(maxRotationDelta,
      computeRotationDeltaMagnitude(bodies[i].rotation, prevRot[i]));
  }
}

static physx::PxReal computeLinearDriveRecipResponse(
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

static physx::PxReal computeAngularDriveRecipResponse(
    const AvbdSolverBody *bodyA, const AvbdSolverBody *bodyB,
    const physx::PxVec3 &worldAxis) {
  physx::PxReal unitResponse = 0.0f;
  if (bodyA)
    unitResponse += (bodyA->invInertiaWorld * worldAxis).dot(worldAxis);
  if (bodyB)
    unitResponse += (bodyB->invInertiaWorld * worldAxis).dot(worldAxis);
  return unitResponse > 1e-8f ? (1.0f / unitResponse) : 0.0f;
}
} // namespace


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

void AvbdSolver::solveLocalSystemWithJoints(
    AvbdSolverBody &body, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap) {

  if (body.invMass <= 0.0f)
    return;

  PX_UNUSED(dt);

  const physx::PxU32 bodyIndex = body.nodeIndex;

  // =========================================================================
  // Step 1: Initialize LHS with mass matrix M/h^2
  // =========================================================================
  AvbdBlock6x6 A;
  A.initializeDiagonal(body.invMass, body.invInertiaWorld, invDt2);

  // =========================================================================
  // Step 2: Initialize RHS with inertia term
  // =========================================================================
  physx::PxReal mass = (body.invMass > 1e-8f) ? (1.0f / body.invMass) : 0.0f;
  physx::PxReal massInvDt2 = mass * invDt2;

  physx::PxVec3 gLinear = (body.position - body.inertialPosition) * massInvDt2;

  physx::PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  physx::PxMat33 inertiaTensor = body.invInertiaWorld.getInverse();
  physx::PxVec3 gAngular = (inertiaTensor * rotError) * invDt2;

  physx::PxU32 numTouching = 0;
  bool hasLinearCoupling =
      false; // Force 6x6 solve for bodies touching joints with pos-rot coupling

  // =========================================================================
  // Step 3a: Accumulate CONTACT contributions (same as solveLocalSystem)
  // =========================================================================
  {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (contactMap && contactMap->numBodies > 0)
      contactMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numContacts;

    for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
      const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
      const physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
      const physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;

      const bool isBodyA = (bodyAIdx == bodyIndex);

      AvbdSolverBody *otherBody = nullptr;
      if (isBodyA && bodyBIdx < numBodies)
        otherBody = &bodies[bodyBIdx];
      else if (!isBodyA && bodyAIdx < numBodies)
        otherBody = &bodies[bodyAIdx];

      physx::PxVec3 worldPosA, worldPosB;
      physx::PxVec3 r;

      if (isBodyA) {
        r = body.rotation.rotate(contacts[c].contactPointA);
        worldPosA = body.position + r;
        worldPosB = otherBody
                        ? otherBody->position + otherBody->rotation.rotate(
                                                    contacts[c].contactPointB)
                        : contacts[c].contactPointB;
      } else {
        r = body.rotation.rotate(contacts[c].contactPointB);
        worldPosA = otherBody
                        ? otherBody->position + otherBody->rotation.rotate(
                                                    contacts[c].contactPointA)
                        : contacts[c].contactPointA;
        worldPosB = body.position + r;
      }

      const physx::PxVec3 &normal = contacts[c].contactNormal;
      physx::PxReal violation =
          (worldPosA - worldPosB).dot(normal) + contacts[c].penetrationDepth;
      violation -= mConfig.avbdAlpha * contacts[c].C0;

      physx::PxReal pen = contacts[c].header.penalty;
      // Per-body primal boost: 2% of M/h^2 (gentle safety net)
      pen = physx::PxMax(pen, 0.005f * massInvDt2);
      physx::PxReal lambda = contacts[c].header.lambda;

      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      physx::PxVec3 rCrossN = r.cross(normal);
      physx::PxVec3 gradPos = normal * sign;
      physx::PxVec3 gradRot = rCrossN * sign;

      // LHS: H += pen * J^T * J
      A.addConstraintContribution(gradPos, gradRot, pen);
      numTouching++;

      // Force: f = clamp(pen * C + lambda, -inf, 0)
      physx::PxReal f = physx::PxMin(0.0f, pen * violation + lambda);

      // RHS: g += J * f
      if (f < 0.0f) {
        gLinear += gradPos * f;
        gAngular += gradRot * f;
      }

      // Friction (same as solveLocalSystem)
      if (contacts[c].friction > 0.0f) {
        physx::PxReal frictionBound =
            physx::PxAbs(contacts[c].header.lambda) * contacts[c].friction;

        physx::PxVec3 prevWorldPosA, prevWorldPosB;
        if (isBodyA) {
          prevWorldPosA = body.prevPosition +
                          body.prevRotation.rotate(contacts[c].contactPointA);
          prevWorldPosB = otherBody ? otherBody->prevPosition +
                                          otherBody->prevRotation.rotate(
                                              contacts[c].contactPointB)
                                    : contacts[c].contactPointB;
        } else {
          prevWorldPosA = otherBody ? otherBody->prevPosition +
                                          otherBody->prevRotation.rotate(
                                              contacts[c].contactPointA)
                                    : contacts[c].contactPointA;
          prevWorldPosB = body.prevPosition +
                          body.prevRotation.rotate(contacts[c].contactPointB);
        }
        physx::PxVec3 relDisp =
            (worldPosA - prevWorldPosA) - (worldPosB - prevWorldPosB);

        // Tangent 0
        {
          const physx::PxVec3 &t = contacts[c].tangent0;
          physx::PxVec3 rCrossT = r.cross(t);
          physx::PxVec3 tGradPos = t * sign;
          physx::PxVec3 tGradRot = rCrossT * sign;
          physx::PxReal tPen = contacts[c].tangentPenalty0;
          tPen = physx::PxMax(tPen, 0.005f * massInvDt2); // 0.5% primal boost
          physx::PxReal tLambda = contacts[c].tangentLambda0;
          physx::PxReal tC = relDisp.dot(t);
          A.addConstraintContribution(tGradPos, tGradRot, tPen);
          physx::PxReal ft = physx::PxClamp(tPen * tC + tLambda, -frictionBound,
                                            frictionBound);
          gLinear += tGradPos * ft;
          gAngular += tGradRot * ft;
        }
        // Tangent 1
        {
          const physx::PxVec3 &t = contacts[c].tangent1;
          physx::PxVec3 rCrossT = r.cross(t);
          physx::PxVec3 tGradPos = t * sign;
          physx::PxVec3 tGradRot = rCrossT * sign;
          physx::PxReal tPen = contacts[c].tangentPenalty1;
          tPen = physx::PxMax(tPen, 0.005f * massInvDt2); // 0.5% primal boost
          physx::PxReal tLambda = contacts[c].tangentLambda1;
          physx::PxReal tC = relDisp.dot(t);
          A.addConstraintContribution(tGradPos, tGradRot, tPen);
          physx::PxReal ft = physx::PxClamp(tPen * tC + tLambda, -frictionBound,
                                            frictionBound);
          gLinear += tGradPos * ft;
          gAngular += tGradRot * ft;
        }
      }
    }
  }

  // Step 3e: Accumulate D6 JOINT contributions

  //
  //   Locked linear DOFs: 3 position rows (same as spherical)
  //   Angular velocity damping (SLERP/axis drives): adds damping_eff to
  //     the angular diagonal of the Hessian, penalizing deviation from
  //     inertial rotation (which encodes current angular velocity).
  //   Locked angular DOFs: TODO (not used by SnippetJoint D6 config)
  // =========================================================================
  if (d6Joints && numD6 > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (d6Map && d6Map->numBodies > 0)
      d6Map->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numD6;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numD6)
        continue;
      const AvbdD6JointConstraint &jnt = d6Joints[j];
      const physx::PxU32 bodyAIdx = jnt.header.bodyIndexA;
      const physx::PxU32 bodyBIdx = jnt.header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;

      const bool isBodyA = (bodyAIdx == bodyIndex);
      const bool otherIsStatic =
          isBodyA ? (bodyBIdx == 0xFFFFFFFF || bodyBIdx >= numBodies)
                  : (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);

      physx::PxReal mA =
          (bodyAIdx < numBodies && bodies[bodyAIdx].invMass > 1e-8f)
              ? (1.0f / bodies[bodyAIdx].invMass)
              : 0.0f;
      physx::PxReal mB =
          (bodyBIdx < numBodies && bodies[bodyBIdx].invMass > 1e-8f)
              ? (1.0f / bodies[bodyBIdx].invMass)
              : 0.0f;
      physx::PxReal mEff = physx::PxMax(mA, mB);

      // Auto-boost penalty using symmetric effective mass
      physx::PxReal pen = physx::PxMax(jnt.header.rho, mEff * invDt2);
      physx::PxReal signJ = isBodyA ? 1.0f : -1.0f;

      // Lever arm from body COM to constraint anchor (used by linear DOFs
      // AND linear drive).  Computed once and reused.
      physx::PxVec3 rArm(0.0f);

      // --- Linear DOFs (LOCKED / LIMITED / FREE) ---
      // Axis selection matches avbd_standalone:
      //   All-LOCKED => world axes (well-conditioned Hessian)
      //   Otherwise  => joint-local axes from localFrameA
      {
        physx::PxVec3 worldAnchorA, worldAnchorB;
        physx::PxVec3 r;
        if (isBodyA) {
          r = body.rotation.rotate(jnt.anchorA);
          worldAnchorA = body.position + r;
          worldAnchorB =
              otherIsStatic ? jnt.anchorB
                            : bodies[bodyBIdx].position +
                                  bodies[bodyBIdx].rotation.rotate(jnt.anchorB);
        } else {
          r = body.rotation.rotate(jnt.anchorB);
          worldAnchorB = body.position + r;
          worldAnchorA =
              otherIsStatic ? jnt.anchorA
                            : bodies[bodyAIdx].position +
                                  bodies[bodyAIdx].rotation.rotate(jnt.anchorA);
        }

        rArm = r;  // export to outer scope for drive
        physx::PxVec3 posError = worldAnchorA - worldAnchorB;

        // Compute joint-frame axes in world space
        physx::PxQuat rotA_lin = isBodyA ? body.rotation
            : (otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                             : bodies[bodyAIdx].rotation);
        physx::PxQuat jointFrameA_lin =
            (otherIsStatic && isBodyA) ? jnt.localFrameA
                                       : rotA_lin * jnt.localFrameA;
        {
          physx::PxReal qm2 = jointFrameA_lin.magnitudeSquared();
          if (qm2 > 1e-8f && PxIsFinite(qm2))
            jointFrameA_lin *= 1.0f / physx::PxSqrt(qm2);
        }

        const bool linAllLocked = (jnt.linearMotion == 0);
        physx::PxVec3 linearAxes[3];
        if (linAllLocked) {
          linearAxes[0] = physx::PxVec3(1.0f, 0.0f, 0.0f);
          linearAxes[1] = physx::PxVec3(0.0f, 1.0f, 0.0f);
          linearAxes[2] = physx::PxVec3(0.0f, 0.0f, 1.0f);
        } else {
          linearAxes[0] = jointFrameA_lin.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          linearAxes[1] = jointFrameA_lin.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f));
          linearAxes[2] = jointFrameA_lin.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f));
        }

        for (int axis = 0; axis < 3; ++axis) {
          physx::PxU32 motion = jnt.getLinearMotion(axis);
          if (motion == 2) // FREE
            continue;

          const physx::PxVec3 &n = linearAxes[axis];
          physx::PxReal C = posError.dot(n);

          physx::PxVec3 rCrossN = r.cross(n);
          physx::PxVec3 gradPos = n * signJ;
          physx::PxVec3 gradRot = rCrossN * signJ;

          if (motion == 0) { // LOCKED
            A.addConstraintContribution(gradPos, gradRot, pen);

            physx::PxReal f = pen * C + jnt.lambdaLinear[axis];
            gLinear += gradPos * f;
            gAngular += gradRot * f;
          } else if (motion == 1) { // LIMITED
            // Baseline Hessian stiffness (prevents drift in free range)
            A.addConstraintContribution(gradPos, gradRot, pen);

            physx::PxReal dist = -posError.dot(n);
            physx::PxReal limitViolation = 0.0f;
            if (dist < jnt.linearLimitLower[axis])
              limitViolation = dist - jnt.linearLimitLower[axis];
            else if (dist > jnt.linearLimitUpper[axis])
              limitViolation = dist - jnt.linearLimitUpper[axis];

            if (physx::PxAbs(limitViolation) > 0.0f) {
              physx::PxReal f = pen * limitViolation + jnt.lambdaLinear[axis];
              physx::PxReal forceMag = 0.0f;

              if (jnt.linearLimitLower[axis] < jnt.linearLimitUpper[axis]) {
                if (limitViolation > 0.0f || jnt.lambdaLinear[axis] > 0.0f) {
                  forceMag = physx::PxMax(0.0f, f);
                } else if (limitViolation < 0.0f ||
                           jnt.lambdaLinear[axis] < 0.0f) {
                  forceMag = physx::PxMin(0.0f, f);
                }
              } else {
                forceMag = f;
              }

              if (physx::PxAbs(forceMag) > 0.0f) {
                // Limit Jacobian direction: use negative axis (gradient of
                // dist)
                physx::PxVec3 nLim = n * (-1.0f);
                physx::PxVec3 gradPosLim = nLim * signJ;
                physx::PxVec3 gradRotLim = r.cross(nLim) * signJ;
                A.addConstraintContribution(gradPosLim, gradRotLim, pen);
                gLinear += gradPosLim * forceMag;
                gAngular += gradRotLim * forceMag;
              }
            }
          }
        } // End of Linear DOFs for loop
      } // End of Linear DOFs scope

      // --- Angular DOFs (LOCKED and LIMITED) ---
      {
        physx::PxQuat rotA, rotB;
        if (isBodyA) {
          rotA = body.rotation;
          rotB = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                               : bodies[bodyBIdx].rotation;
        } else {
          rotA = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                               : bodies[bodyAIdx].rotation;
          rotB = body.rotation;
        }

        // Detect revolute pattern: twist(X) FREE or LIMITED, swing(Y,Z) LOCKED
        const physx::PxU32 twistMotion = jnt.getAngularMotion(0);
        const physx::PxU32 swing1Motion = jnt.getAngularMotion(1);
        const physx::PxU32 swing2Motion = jnt.getAngularMotion(2);
        const bool isRevolutePattern =
            (twistMotion != 0) && (swing1Motion == 0) && (swing2Motion == 0);

        if (isRevolutePattern) {
          // Cross-product axis alignment (2 rows) - matches reference revolute
          // solver. Unlike computeAngularError decomposition, this is immune to
          // large twist angles amplifying swing drift.
          physx::PxQuat worldFrameA = rotA * jnt.localFrameA;
          physx::PxQuat worldFrameB = rotB * jnt.localFrameB;
          physx::PxVec3 worldTwistA =
              worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          physx::PxVec3 worldTwistB =
              worldFrameB.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          physx::PxVec3 axisViolation = worldTwistA.cross(worldTwistB);

          // Build perpendicular basis from worldTwistA
          physx::PxVec3 perp1, perp2;
          if (physx::PxAbs(worldTwistA.x) < 0.9f)
            perp1 = worldTwistA.cross(physx::PxVec3(1.0f, 0.0f, 0.0f));
          else
            perp1 = worldTwistA.cross(physx::PxVec3(0.0f, 1.0f, 0.0f));
          physx::PxReal perp1Len = perp1.magnitude();
          if (perp1Len > 1e-6f)
            perp1 *= (1.0f / perp1Len);
          perp2 = worldTwistA.cross(perp1);
          physx::PxReal perp2Len = perp2.magnitude();
          if (perp2Len > 1e-6f)
            perp2 *= (1.0f / perp2Len);

          physx::PxReal err1 = axisViolation.dot(perp1);
          physx::PxReal err2 = axisViolation.dot(perp2);

          // Row 1 (stored in lambdaAngular[1])
          {
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = -perp1 * signJ;
            A.addConstraintContribution(gradPos, gradRot, pen);
            physx::PxReal f = pen * err1 + jnt.lambdaAngular[1];
            gAngular += gradRot * f;
          }
          // Row 2 (stored in lambdaAngular[2])
          {
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = -perp2 * signJ;
            A.addConstraintContribution(gradPos, gradRot, pen);
            physx::PxReal f = pen * err2 + jnt.lambdaAngular[2];
            gAngular += gradRot * f;
          }

          // Handle twist axis (0) if LIMITED
          if (twistMotion == 1) {
            physx::PxVec3 worldAxis =
                worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = worldAxis * signJ;

            physx::PxReal error =
                jnt.computeAngularError(rotA, rotB, 0);
            physx::PxReal limitViolation =
                jnt.computeAngularLimitViolation(error, 0);
            physx::PxReal f =
                pen * limitViolation + jnt.lambdaAngular[0];
            physx::PxReal forceMag = 0.0f;

            if (jnt.angularLimitLower[0] < jnt.angularLimitUpper[0]) {
              if (limitViolation > 0.0f || jnt.lambdaAngular[0] > 0.0f)
                forceMag = physx::PxMax(0.0f, f);
              else if (limitViolation < 0.0f || jnt.lambdaAngular[0] < 0.0f)
                forceMag = physx::PxMin(0.0f, f);
            } else {
              forceMag = f;
            }

            if (physx::PxAbs(forceMag) > 0.0f) {
              A.addConstraintContribution(gradPos, gradRot, pen);
              gAngular += gradRot * forceMag;
            }
          }
        } else {
          // Generic per-axis angular constraint handling
          for (int axis = 0; axis < 3; ++axis) {
            physx::PxU32 motion = jnt.getAngularMotion(axis);
            if (motion == 2) // FREE
              continue;

            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[axis] = 1.0f;
            physx::PxQuat worldFrameA = rotA * jnt.localFrameA;
            physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);

            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = worldAxis * signJ;

            if (motion == 0) { // LOCKED
              physx::PxReal C = jnt.computeAngularError(rotA, rotB, axis);
              A.addConstraintContribution(gradPos, gradRot, pen);

              physx::PxReal f = pen * C + jnt.lambdaAngular[axis];
              gAngular += gradRot * f;
            } else if (motion == 1) { // LIMITED
              physx::PxReal error =
                  jnt.computeAngularError(rotA, rotB, axis);
              physx::PxReal limitViolation =
                  jnt.computeAngularLimitViolation(error, axis);
              physx::PxReal f =
                  pen * limitViolation + jnt.lambdaAngular[axis];
              physx::PxReal forceMag = 0.0f;

              if (jnt.angularLimitLower[axis] < jnt.angularLimitUpper[axis]) {
                if (limitViolation > 0.0f || jnt.lambdaAngular[axis] > 0.0f) {
                  forceMag = physx::PxMax(0.0f, f);
                } else if (limitViolation < 0.0f ||
                           jnt.lambdaAngular[axis] < 0.0f) {
                  forceMag = physx::PxMin(0.0f, f);
                }
              } else {
                forceMag = f;
              }

              if (physx::PxAbs(forceMag) > 0.0f) {
                A.addConstraintContribution(gradPos, gradRot, pen);
                gAngular += gradRot * forceMag;
              }
            }
          }
        }
      }

      // --- Cone limit (single angular inequality, for spherical joints) ---
      // Uses the same geometric approach as the reference spherical solver:
      // a single cone constraint based on the angle between the joint X-axes.
      if (jnt.coneAngleLimit > 0.0f) {
        physx::PxQuat rotA_cone, rotB_cone;
        if (isBodyA) {
          rotA_cone = body.rotation;
          rotB_cone = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                                    : bodies[bodyBIdx].rotation;
        } else {
          rotA_cone = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                                    : bodies[bodyAIdx].rotation;
          rotB_cone = body.rotation;
        }

        // Cone axis = X-axis of each body's joint frame
        physx::PxVec3 worldAxisA = (rotA_cone * jnt.localFrameA).rotate(
            physx::PxVec3(1.0f, 0.0f, 0.0f));
        physx::PxVec3 worldAxisB = (rotB_cone * jnt.localFrameB).rotate(
            physx::PxVec3(1.0f, 0.0f, 0.0f));

        physx::PxReal dotAB = physx::PxClamp(worldAxisA.dot(worldAxisB),
                                              -1.0f, 1.0f);
        physx::PxReal coneAngle = physx::PxAcos(dotAB);
        physx::PxReal coneViolation = coneAngle - jnt.coneAngleLimit;

        // coneLambda <= 0 (unilateral): force = pen * violation - coneLambda
        physx::PxReal forceMag = pen * coneViolation - jnt.coneLambda;

        if (forceMag > 0.0f) {
          physx::PxVec3 corrAxis = worldAxisA.cross(worldAxisB);
          physx::PxReal corrAxisMag = corrAxis.magnitude();
          if (corrAxisMag > 1e-6f) {
            corrAxis *= (1.0f / corrAxisMag);

            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = -corrAxis * signJ;

            A.addConstraintContribution(gradPos, gradRot, pen);
            gAngular += gradRot * forceMag;
          }
        }
      }

      // --- Pure AVBD AL velocity drive constraints ---
      // Replaces ad-hoc damping. Each driven axis contributes an AL
      // velocity constraint:
      //   C = (-x_B - -x_A) - axis - v_target - dt   (linear)
      //   C = (--_B - --_A) - axis - -_target - dt   (angular)
      // Hessian: -_drive - (axis - axis)
      // RHS:     sign - (-_drive - C + -)
      {
        // Joint frame A in world space
        physx::PxQuat jointFrameA =
            otherIsStatic && isBodyA
                ? jnt.localFrameA
                : (isBodyA ? body.rotation * jnt.localFrameA
                           : (otherIsStatic ? jnt.localFrameA
                                            : bodies[bodyAIdx].rotation *
                                                  jnt.localFrameA));

        physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
        if (qMag2 > 1e-8f && PxIsFinite(qMag2))
          jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

        physx::PxReal dt2 = dt * dt;

        // Get "other body" for relative displacement
        const AvbdSolverBody *otherBody = nullptr;
        const AvbdSolverBody *bodyARef = nullptr;
        const AvbdSolverBody *bodyBRef = nullptr;
        if (isBodyA && bodyBIdx < numBodies)
          otherBody = &bodies[bodyBIdx];
        else if (!isBodyA && bodyAIdx < numBodies)
          otherBody = &bodies[bodyAIdx];
        bodyARef = (bodyAIdx < numBodies) ? &bodies[bodyAIdx] : nullptr;
        bodyBRef = (bodyBIdx < numBodies) ? &bodies[bodyBIdx] : nullptr;

        // --- Linear velocity drive (AL constraint) ---
        if ((jnt.driveFlags & 0x7) != 0) {
          for (int a = 0; a < 3; ++a) {
            if ((jnt.driveFlags & (1 << a)) == 0)
              continue;
            physx::PxReal damping = (&jnt.linearDamping.x)[a];
            if (damping <= 0.0f)
              continue;

            // World-space axis
            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[a] = 1.0f;
            physx::PxVec3 wAxis = jointFrameA.rotate(localAxis);

            // Displacement of each body from start-of-step
            physx::PxVec3 dxThis = body.position - body.prevPosition;
            physx::PxVec3 dxOther =
                otherBody ? (otherBody->position - otherBody->prevPosition)
                          : physx::PxVec3(0.0f);

            // Constraint: C = (-x_B - -x_A) - axis - v_target - dt
            physx::PxReal dxB_proj, dxA_proj;
            if (isBodyA) {
              dxA_proj = dxThis.dot(wAxis);
              dxB_proj = dxOther.dot(wAxis);
            } else {
              dxB_proj = dxThis.dot(wAxis);
              dxA_proj = dxOther.dot(wAxis);
            }
            physx::PxReal targetVel = (&jnt.driveLinearVelocity.x)[a];
            physx::PxReal C = (dxB_proj - dxA_proj) - targetVel * dt;

            const physx::PxVec3 rAWorld = bodyARef
              ? bodyARef->rotation.rotate(jnt.anchorA)
              : physx::PxVec3(0.0f);
            const physx::PxVec3 rBWorld = bodyBRef
              ? bodyBRef->rotation.rotate(jnt.anchorB)
              : physx::PxVec3(0.0f);
            physx::PxReal rho_drive = damping / dt2;
            if (jnt.isLinearAccelerationDrive(a)) {
              const physx::PxReal driveScale =
                computeLinearDriveRecipResponse(bodyARef, bodyBRef,
                               rAWorld, rBWorld, wAxis);
              const physx::PxReal stiffness = (&jnt.linearStiffness.x)[a];
              const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
              const physx::PxReal implicitScale =
                1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
              rho_drive *= driveScale * implicitScale;
            }
            physx::PxReal lam = (&jnt.lambdaDriveLinear.x)[a];
            physx::PxReal signAL = isBodyA ? -1.0f : 1.0f;
            physx::PxReal f = signAL * (rho_drive * C + lam);

            // Full 6D Jacobian Jd = (wAxis, rArm x wAxis), matching
            // standalone.  The drive force acts at the anchor point, so
            // the lever arm produces torque.
            physx::PxVec3 rCrossW = rArm.cross(wAxis);

            // Hessian: outer(Jd, Jd * rho_drive) -> all 4 blocks
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l) {
                A.linearLinear(k, l) +=
                    rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];
                A.linearAngular(k, l) +=
                    rho_drive * (&wAxis.x)[k] * (&rCrossW.x)[l];
                A.angularLinear(k, l) +=
                    rho_drive * (&rCrossW.x)[k] * (&wAxis.x)[l];
                A.angularAngular(k, l) +=
                    rho_drive * (&rCrossW.x)[k] * (&rCrossW.x)[l];
              }

            // RHS: gradient on both linear and angular
            gLinear += physx::PxVec3(f * wAxis.x, f * wAxis.y, f * wAxis.z);
            gAngular += physx::PxVec3(f * rCrossW.x, f * rCrossW.y, f * rCrossW.z);
          }
        }

        // --- Angular velocity drive (AL constraint) ---
        if ((jnt.driveFlags & 0x38) != 0) {
          // Angular displacement from start-of-step for this body
          physx::PxQuat dqThis =
              body.rotation * body.prevRotation.getConjugate();
          if (dqThis.w < 0.0f)
            dqThis = -dqThis;
          physx::PxVec3 dThetaThis(dqThis.x, dqThis.y, dqThis.z);
          dThetaThis *= 2.0f;

          physx::PxVec3 dThetaOther(0.0f);
          if (otherBody) {
            physx::PxQuat dqOther =
                otherBody->rotation * otherBody->prevRotation.getConjugate();
            if (dqOther.w < 0.0f)
              dqOther = -dqOther;
            dThetaOther = physx::PxVec3(dqOther.x, dqOther.y, dqOther.z) * 2.0f;
          }

          physx::PxVec3 dThetaA(0.0f), dThetaB(0.0f);
          if (isBodyA) {
            dThetaA = dThetaThis;
            dThetaB = dThetaOther;
          } else {
            dThetaA = dThetaOther;
            dThetaB = dThetaThis;
          }
          physx::PxVec3 relDW = dThetaB - dThetaA;
          physx::PxVec3 worldAngTarget =
              jointFrameA.rotate(jnt.driveAngularVelocity) * dt;
          physx::PxReal signAL = isBodyA ? -1.0f : 1.0f;

          bool slerpDrive = (jnt.driveFlags & 0x20) != 0;
          if (slerpDrive) {
            physx::PxReal damping =
                jnt.angularDamping.z; // SLERP uses Z damping slot
            if (damping > 0.0f) {
              physx::PxReal rho_drive = damping / dt2;
              if (jnt.isAngularAccelerationDrive(2)) {
                const physx::PxReal driveScale =
                    computeAngularDriveRecipResponse(bodyARef, bodyBRef,
                                                     physx::PxVec3(1.0f, 0.0f, 0.0f));
                const physx::PxReal stiffness = jnt.angularStiffness.z;
                const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
                const physx::PxReal implicitScale =
                    1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
                rho_drive *= driveScale * implicitScale;
              }
              for (int k = 0; k < 3; ++k) {
                physx::PxReal C = (&relDW.x)[k] - (&worldAngTarget.x)[k];
                physx::PxReal lam = (&jnt.lambdaDriveAngular.x)[k];
                physx::PxReal f = signAL * (rho_drive * C + lam);

                A.angularAngular(k, k) += rho_drive;
                (&gAngular.x)[k] += f;
              }
            }
          } else {
            // Axis mapping: bit3=twist(X), bit4=swing1(Y), bit5=swing2(Z)
            struct AxisDrive {
              int bit;
              int dampIdx;
              physx::PxVec3 localAxis;
            };
            const AxisDrive axes[3] = {
                {3, 0, physx::PxVec3(1.0f, 0.0f, 0.0f)}, // TWIST
                {4, 1, physx::PxVec3(0.0f, 1.0f, 0.0f)}, // SWING1
                {5, 2, physx::PxVec3(0.0f, 0.0f, 1.0f)}, // SWING2
            };

            for (int a = 0; a < 3; ++a) {
              if ((jnt.driveFlags & (1 << axes[a].bit)) == 0)
                continue;
              physx::PxReal damping = (&jnt.angularDamping.x)[axes[a].dampIdx];
              if (damping <= 0.0f)
                continue;

              physx::PxVec3 wAxis = jointFrameA.rotate(axes[a].localAxis);
              // PhysX TGS convention: Twist/Swing target velocities are
              // applied as (wA - wB), meaning wB - wA = -target. SLERP is
              // applied as wB
              // - wA = target, which is handled above.
              physx::PxReal targetOmega_dt = -worldAngTarget.dot(wAxis);
              physx::PxReal C = relDW.dot(wAxis) - targetOmega_dt;

                physx::PxReal rho_drive = damping / dt2;
                if (jnt.isAngularAccelerationDrive(axes[a].dampIdx)) {
                const physx::PxReal driveScale =
                  computeAngularDriveRecipResponse(bodyARef, bodyBRef, wAxis);
                const physx::PxReal stiffness =
                  (&jnt.angularStiffness.x)[axes[a].dampIdx];
                const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
                const physx::PxReal implicitScale =
                  1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
                rho_drive *= driveScale * implicitScale;
                }
              physx::PxReal lam = (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx];
              physx::PxReal f = signAL * (rho_drive * C + lam);

              // Hessian: -_drive - (wAxis - wAxis) on angular block
              for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                  A.angularAngular(k, l) +=
                      rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];

              // RHS
              gAngular += physx::PxVec3(f * wAxis.x, f * wAxis.y, f * wAxis.z);
            }
          }
        }
      }

      numTouching++;
      hasLinearCoupling = true; // D6 joints always create pos-rot coupling via lever arm
    }
  }

  // =========================================================================
  // Step 3g: Accumulate GEAR JOINT contributions (angular-only, position-level)
  //
  // Constraint: C = geometricError  (accumulated angle error, radians)
  //   Computed by GearJoint::updateError() each frame.
  //
  // Jacobians match GearJointSolverPrep (ExtGearJoint.cpp):
  //   Body A:  J_ang = +worldAxis0 * gearRatio   (con.angular0 = axis0*ratio)
  //   Body B:  J_ang = -worldAxis1               (con.angular1 = -axis1)
  //
  // gearAxis0/1 stored as BODY LOCAL vectors -> rotate to world with
  // body.rotation
  //
  //   LHS: A_ang += pen * J_ang - J_ang
  //   RHS: g_ang += J_ang * (pen*C + lambda)
  // =========================================================================
  if (gearJoints && numGear > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (gearMap && gearMap->numBodies > 0)
      gearMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numGear;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numGear)
        continue;
      const AvbdGearJointConstraint &gnt = gearJoints[j];
      const physx::PxU32 bodyAIdx = gnt.header.bodyIndexA;
      const physx::PxU32 bodyBIdx = gnt.header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;
      const bool isBodyA = (bodyAIdx == bodyIndex);
      const bool otherIsStatic =
          isBodyA ? (bodyBIdx == 0xFFFFFFFF || bodyBIdx >= numBodies)
                  : (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);

      physx::PxReal dwA = 0.0f;
      physx::PxReal dwB = 0.0f;

      auto computeDeltaW = [](const AvbdSolverBody &b,
                              const physx::PxVec3 &axis) -> physx::PxReal {
        physx::PxQuat dq = b.rotation * b.prevRotation.getConjugate();
        if (dq.w < 0.0f)
          dq = -dq;
        return physx::PxVec3(dq.x, dq.y, dq.z).dot(axis) * 2.0f;
      };

      physx::PxVec3 worldAxis0, worldAxis1;

      if (isBodyA) {
        worldAxis0 = body.rotation.rotate(gnt.gearAxis0);
        dwA = computeDeltaW(body, worldAxis0);
        // For static body B, axis is fixed in world space. (Ideally we'd use
        // the static rotation, but typically it rotates from identity)
        worldAxis1 = gnt.gearAxis1;
      } else {
        worldAxis1 = body.rotation.rotate(gnt.gearAxis1);
        dwB = computeDeltaW(body, worldAxis1);
        worldAxis0 = gnt.gearAxis0;
      }

      // If the other body IS dynamic, rotate its axis and fetch its dw
      if (!otherIsStatic) {
        if (isBodyA) {
          worldAxis1 = bodies[bodyBIdx].rotation.rotate(gnt.gearAxis1);
          dwB = computeDeltaW(bodies[bodyBIdx], worldAxis1);
        } else {
          worldAxis0 = bodies[bodyAIdx].rotation.rotate(gnt.gearAxis0);
          dwA = computeDeltaW(bodies[bodyAIdx], worldAxis0);
        }
      }

      physx::PxReal C = dwA * gnt.gearRatio + dwB + gnt.geometricError;

      const physx::PxVec3 rawAxis = isBodyA ? worldAxis0 : worldAxis1;
      const physx::PxVec3 tmpInvIAxis = body.invInertiaWorld.transform(rawAxis);
      const physx::PxReal invIaxial = rawAxis.dot(tmpInvIAxis);
      const physx::PxReal Iaxial =
          (invIaxial > 1e-10f) ? (1.0f / invIaxial) : 0.0f;
      physx::PxReal pen = physx::PxMax(gnt.header.rho, Iaxial * invDt2);

      // Jacobian for THIS body - Body B uses POSITIVE axis1 (matches TGS
      // algebraic summation)
      physx::PxVec3 J_ang =
          isBodyA ? (worldAxis0 * gnt.gearRatio) : (worldAxis1);

      // AL force: f = pen * C + gnt.lambdaGear
      physx::PxReal f = pen * C + gnt.lambdaGear;

#if AVBD_JOINT_DEBUG
      {
        static physx::PxU32 s_gearDebugCount = 0;
        if (s_gearDebugCount < 0) {
          printf("[Gear] frame=%u isA=%d body%u num=%u C=%.4f (err=%.4f "
                 "dwA=%.4f dwB=%.4f) f=%.1f pen=%.1f gearRatio=%.2f "
                 "axis==(%.1f,%.1f,%.1f)\n",
                 s_gearDebugCount, isBodyA, bodyIndex, ji, C,
                 gnt.geometricError, dwA, dwB, f, pen, gnt.gearRatio,
                 isBodyA ? worldAxis0.x : worldAxis1.x,
                 isBodyA ? worldAxis0.y : worldAxis1.y,
                 isBodyA ? worldAxis0.z : worldAxis1.z);
          if (!isBodyA)
            s_gearDebugCount++; // increment after both passes
        }
      }
#endif

      // Accumulate into 6x6 Hessian (linear part zero, angular part = J)
      A.addConstraintContribution(physx::PxVec3(0.0f), J_ang, pen);

      // RHS gradient
      gAngular += J_ang * f;

      numTouching++;
    }
  }

  // =========================================================================
  // Step 4: Handle bodies with no constraints at all
  // =========================================================================
  if (numTouching == 0) {
    body.position = body.inertialPosition;
    body.rotation = body.inertialRotation;
    return;
  }

  // =========================================================================
  // Step 5: Solve A * delta = g via LDLT
  // =========================================================================
  AvbdLDLT ldlt;
  AvbdVec6 rhs(gLinear, gAngular);

#if AVBD_JOINT_DEBUG
  {
    static physx::PxU32 s_debugSolveFrame = 0;
    bool doSolveDebug = (s_debugSolveFrame < 4);
    if (doSolveDebug &&
        (numD6 > 0 || numGear > 0)) {
      printf("  [solveUnified] body%u touching=%u gLin=(%.4f,%.4f,%.4f) "
             "gAng=(%.4f,%.4f,%.4f)\n",
             bodyIndex, numTouching, gLinear.x, gLinear.y, gLinear.z,
             gAngular.x, gAngular.y, gAngular.z);
      printf("    H_diag pos=(%.1f,%.1f,%.1f) rot=(%.1f,%.1f,%.1f)\n",
             A.linearLinear.column0.x, A.linearLinear.column1.y,
             A.linearLinear.column2.z, A.angularAngular.column0.x,
             A.angularAngular.column1.y, A.angularAngular.column2.z);
      printf("    inertialDelta pos=(%.6f,%.6f,%.6f)\n",
             body.position.x - body.inertialPosition.x,
             body.position.y - body.inertialPosition.y,
             body.position.z - body.inertialPosition.z);
      s_debugSolveFrame++;
    }
  }
#endif

  physx::PxVec3 deltaPos;
  physx::PxVec3 deltaTheta;

  // Force 6x6 solve for bodies touching Prismatic joints: the 3x3
  // decoupled solve is incompatible with Prismatic's axis-dependent
  // position projection, which creates divergent oscillation.
  const bool use6x6 = mConfig.enableLocal6x6Solve || hasLinearCoupling;
  if (use6x6) {
    if (ldlt.decomposeWithRegularization(A)) {
      AvbdVec6 delta = ldlt.solve(rhs);
      deltaPos = delta.linear;
      deltaTheta = delta.angular;
    } else {
      deltaPos = physx::PxVec3(0.0f);
      deltaTheta = physx::PxVec3(0.0f);
    }
  } else {
    // 3x3 Block-Diagonal Decoupled Solve Fallback
    physx::PxMat33 Alin = A.linearLinear;
    physx::PxMat33 Aang = A.angularAngular;

    bool linOk = (physx::PxAbs(Alin.getDeterminant()) > 1e-12f);
    bool angOk = (physx::PxAbs(Aang.getDeterminant()) > 1e-12f);

    if (linOk) {
      physx::PxMat33 AlinInv = Alin.getInverse();
      deltaPos = AlinInv * gLinear;
    } else {
      deltaPos = physx::PxVec3(0.0f);
    }

    if (angOk) {
      physx::PxMat33 AangInv = Aang.getInverse();
      deltaTheta = AangInv * gAngular;
    } else {
      deltaTheta = physx::PxVec3(0.0f);
    }
  }

#if AVBD_JOINT_DEBUG
  {
    static physx::PxU32 s_debugSolveFrame2 = 0;
    bool doSolveDebug = (s_debugSolveFrame2 < 2);
    if (doSolveDebug && (numD6 > 0 || numGear > 0)) {
      printf("    delta pos=(%.6f,%.6f,%.6f) rot=(%.6f,%.6f,%.6f)\n",
             deltaPos.x, deltaPos.y, deltaPos.z, deltaTheta.x, deltaTheta.y,
             deltaTheta.z);
      printf("    newPos=(%.4f,%.4f,%.4f)\n", body.position.x - deltaPos.x,
             body.position.y - deltaPos.y, body.position.z - deltaPos.z);
    }
    // Only increment once per full body loop (not per body)
    if (bodyIndex == 0 && (numD6 > 0 || numGear > 0)) {
      s_debugSolveFrame2++;
    }
  }
#endif

  // =========================================================================
  // Step 6: Apply update  x -= delta
  // =========================================================================
  body.position -= deltaPos;

  if (deltaTheta.magnitudeSquared() > 1e-12f) {
    physx::PxQuat dq(deltaTheta.x, deltaTheta.y, deltaTheta.z, 0.0f);
    body.rotation = (body.rotation - dq * body.rotation * 0.5f).getNormalized();
  }
}

//=============================================================================
// Block Descent Iteration - Position-Based Constraint Solving
//=============================================================================



/**
 * @brief Compute correction for D6 joint
 */
bool AvbdSolver::computeD6JointCorrection(const AvbdD6JointConstraint &joint,
                                          AvbdSolverBody *bodies,
                                          physx::PxU32 numBodies,
                                          physx::PxU32 bodyIndex,
                                          physx::PxVec3 &deltaPos,
                                          physx::PxVec3 &deltaTheta) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

  bool bodyAIsStatic = (bodyAIdx >= numBodies);

  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }

  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);

  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }

  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);

  bool hasCorrection = false;

  // Check if bodies are static (index >= numBodies means static body, frame
  // already in world space) Note: bodyAIsStatic already defined above for
  // debug purposes
  bool bodyBIsStatic = (bodyBIdx >= numBodies);

  // Get rotations for frame transforms
  physx::PxQuat rotA =
      bodyAIsStatic
          ? physx::PxQuat(physx::PxIdentity)
          : (isBodyA ? body.rotation
                     : (otherBody ? otherBody->rotation
                                  : physx::PxQuat(physx::PxIdentity)));
  physx::PxQuat rotB =
      bodyBIsStatic ? physx::PxQuat(physx::PxIdentity)
                    : (isBodyA ? (otherBody ? otherBody->rotation
                                            : physx::PxQuat(physx::PxIdentity))
                               : body.rotation);

  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB =
        otherBody
            ? otherBody->position + otherBody->rotation.rotate(joint.anchorB)
            : joint.anchorB; // anchorB already in world space for static
  } else {
    worldAnchorA =
        otherBody
            ? otherBody->position + otherBody->rotation.rotate(joint.anchorA)
            : joint.anchorA; // anchorA already in world space for static
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }

  physx::PxVec3 posError = worldAnchorA - worldAnchorB;

  // Position constraint (linear locked) - but skip axes with velocity drive
  // When velocity drive is active, we want the body to move, not be
  // constrained
  if (joint.linearMotion == 0) {
    // Determine which axes have velocity drive (we'll skip position
    // constraint on those)
    physx::PxU32 linearDriveAxes =
        joint.driveFlags & 0x7; // bits 0,1,2 for X,Y,Z

    // If we have velocity drive, project out the position error along driven
    // axes
    physx::PxVec3 constrainedPosError = posError;

    if (linearDriveAxes != 0 && !isBodyA) {
      // Get joint frame in world space
      physx::PxQuat jointFrameA =
          bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);
      physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
      if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
        jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

        // Remove position error component along driven axes
        for (int axis = 0; axis < 3; ++axis) {
          if ((linearDriveAxes & (1 << axis)) != 0) {
            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[axis] = 1.0f;
            physx::PxVec3 worldAxis = jointFrameA.rotate(localAxis);
            // Remove the component of position error along this driven axis
            constrainedPosError -=
                worldAxis * constrainedPosError.dot(worldAxis);
          }
        }
      }
    }

    physx::PxReal posErrorMag = constrainedPosError.magnitude();
    if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      physx::PxVec3 direction = constrainedPosError / posErrorMag;

      physx::PxVec3 r = isBodyA ? body.rotation.rotate(joint.anchorA)
                                : body.rotation.rotate(joint.anchorB);
      physx::PxVec3 rCrossD = r.cross(direction);
      physx::PxReal w =
          body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);

      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOther = isBodyA
                                   ? otherBody->rotation.rotate(joint.anchorB)
                                   : otherBody->rotation.rotate(joint.anchorA);
        physx::PxVec3 rOtherCrossD = rOther.cross(direction);
        w += otherBody->invMass +
             rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
      }

      if (w > 1e-6f) {
        physx::PxReal correctionMag = -posErrorMag / w;
        physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

        deltaPos = direction * (correctionMag * body.invMass * sign);
        deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
      }
      hasCorrection = true;
    }
  }

  // Drive constraints now handled in AVBD Hessian
  // (solveLocalSystemWithJoints/3x3) GS fallback for drives is disabled.

  return hasCorrection;
}

//=============================================================================
// Solver with Joint Constraints
//=============================================================================

void AvbdSolver::solveWithJoints(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const physx::PxVec3 &gravity, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors,
    physx::PxU32 iterationOverride,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts) {

  PX_PROFILE_ZONE("AVBD.solveWithJoints", 0);

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();
  mStats.numBodies = numBodies;
  mStats.numContacts = numContacts;
  mStats.numJoints = numD6 + numGear;

  const physx::PxReal invDt = 1.0f / dt;
  const physx::PxReal invDt2 = invDt * invDt;

#if AVBD_JOINT_DEBUG
  const bool doDebug = (s_avbdJointDebugFrame < AVBD_JOINT_DEBUG_FRAMES);
  if (doDebug) {
    printf("\n=== AVBD solveWithJoints FRAME %u === bodies=%u contacts=%u "
           "d6=%u gear=%u\n",
           s_avbdJointDebugFrame, numBodies, numContacts, numD6, numGear);
  }
#endif

  // =========================================================================
  // Stage 1: Prediction
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);

    // Soft particle prediction
    for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
      softParticles[i].computePrediction(dt, gravity);
  }

  // =========================================================================
  // Stage 2: Adaptive position warmstarting (ref: AVBD3D solver.cpp L76-98)
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.initPositions", 0);

    const physx::PxReal gravMag = gravity.magnitude();
    const physx::PxVec3 gravDir =
        (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      bodies[i].prevPosition = bodies[i].position;
      bodies[i].prevRotation = bodies[i].rotation;

      if (bodies[i].invMass > 0.0f) {
        physx::PxVec3 accel =
            (bodies[i].linearVelocity - bodies[i].prevLinearVelocity) * invDt;
        physx::PxReal accelWeight = 0.0f;
        if (gravMag > 1e-6f) {
          accelWeight =
              physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f);
        }
        bodies[i].position = bodies[i].prevPosition +
                             bodies[i].linearVelocity * dt +
                             gravity * (accelWeight * dt * dt);
        bodies[i].rotation = bodies[i].inertialRotation;
      }
    }

    // Soft particle adaptive warmstarting
    for (physx::PxU32 i = 0; i < numSoftParticles; ++i) {
      AvbdSoftParticle &sp = softParticles[i];
      if (sp.invMass <= 0.0f) continue;
      physx::PxVec3 accel = (sp.velocity - sp.prevVelocity) * invDt;
      physx::PxReal accelWeight = 0.0f;
      if (gravMag > 1e-6f)
        accelWeight = physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f);
      sp.position = sp.position + sp.velocity * dt + gravity * (accelWeight * dt * dt);
    }

    // Soft body AVBD warmstart (penalty only)
    for (physx::PxU32 sbi = 0; sbi < numSoftBodies; ++sbi) {
      AvbdSoftBody &sb = softBodies[sbi];
      for (physx::PxU32 ai = 0; ai < sb.attachments.size(); ++ai)
        sb.attachments[ai].k = physx::PxMax(1e3f, physx::PxMin(sb.attachments[ai].kMax, sb.attachments[ai].k * mConfig.avbdGamma));
      for (physx::PxU32 pi = 0; pi < sb.pins.size(); ++pi)
        sb.pins[pi].k = physx::PxMax(1e3f, physx::PxMin(sb.pins[pi].kMax, sb.pins[pi].k * mConfig.avbdGamma));
    }
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      softContacts[sci].k = physx::PxMin(1e4f, softContacts[sci].ke);
  }

  // =========================================================================
  // Stage 3: Penalty floor for contacts (graph-propagated effective mass)
  //
  // Two key improvements:
  //   1. Graph-propagated effective mass: instead of simple valence-based
  //      augmentation, we propagate mass through the joint graph using
  //      Jacobi iteration (Neumann series approximation of Schur complement).
  //      Interior mesh nodes accumulate the collective inertia of their
  //      D-hop neighborhood with exponential decay per hop.
  //   2. max(augA,augB) for dynamic-dynamic contacts: the penalty must be
  //      stiff enough to decelerate the HEAVIER body within one timestep.
  //      AVBD's implicit solve keeps this stable regardless of mass ratio.
  //   3. Two-tier scaling: body-ground uses 0.25 (stacking stiffness),
  //      dynamic-dynamic uses 0.05 (allows net deformation).
  // =========================================================================
  if (contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.penaltyFloor", 0);

    // Graph propagation parameters
    const int propagationDepth = 4;
    const physx::PxReal propagationDecay = 0.5f;

    // Step 1: Build adjacency list from joints
    physx::PxArray<physx::PxArray<physx::PxU32>> adj;
    adj.resize(numBodies);
    auto addEdge = [&](physx::PxU32 a, physx::PxU32 b) {
      if (a < numBodies && b < numBodies) {
        adj[a].pushBack(b);
        adj[b].pushBack(a);
      }
    };
    for (physx::PxU32 j = 0; j < numD6; ++j)
      addEdge(d6Joints[j].header.bodyIndexA, d6Joints[j].header.bodyIndexB);
    for (physx::PxU32 j = 0; j < numGear; ++j)
      addEdge(gearJoints[j].header.bodyIndexA, gearJoints[j].header.bodyIndexB);

    // Step 2: Jacobi propagation of effective mass
    physx::PxArray<physx::PxReal> mEff;
    mEff.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      mEff[i] = (bodies[i].invMass > 0.0f) ? (1.0f / bodies[i].invMass) : 0.0f;

    for (int d = 0; d < propagationDepth; ++d) {
      physx::PxArray<physx::PxReal> mNext;
      mNext.resize(numBodies);
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        physx::PxReal baseMass =
            (bodies[i].invMass > 0.0f) ? (1.0f / bodies[i].invMass) : 0.0f;
        physx::PxReal neighborSum = 0.0f;
        for (physx::PxU32 k = 0; k < adj[i].size(); ++k)
          neighborSum += mEff[adj[i][k]];
        mNext[i] = baseMass + propagationDecay * neighborSum;
      }
      mEff = mNext;
    }

    // Step 3: apply penalty floor with propagated effective mass
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      physx::PxReal massA = 0.0f, massB = 0.0f;
      if (bA < numBodies && bodies[bA].invMass > 0.0f)
        massA = 1.0f / bodies[bA].invMass;
      if (bB < numBodies && bodies[bB].invMass > 0.0f)
        massB = 1.0f / bodies[bB].invMass;

      const physx::PxReal augA = (bA < numBodies) ? mEff[bA] : 0.0f;
      const physx::PxReal augB = (bB < numBodies) ? mEff[bB] : 0.0f;

      physx::PxReal effectiveMass;
      physx::PxReal penScale;
      if (massA > 0.0f && massB > 0.0f) {
        // Dynamic-dynamic: heavier body determines floor
        effectiveMass = physx::PxMax(augA, augB);
        penScale = 0.05f;
      } else {
        // Body-vs-static: propagated mass of dynamic body, full scale
        effectiveMass = physx::PxMax(augA, augB);
        penScale = 0.25f;
      }

      const physx::PxReal penaltyFloor = penScale * effectiveMass * invDt2;
      if (contacts[c].header.penalty < penaltyFloor)
        contacts[c].header.penalty = penaltyFloor;
      if (contacts[c].tangentPenalty0 < penaltyFloor)
        contacts[c].tangentPenalty0 = penaltyFloor;
      if (contacts[c].tangentPenalty1 < penaltyFloor)
        contacts[c].tangentPenalty1 = penaltyFloor;
    }
  }

  // =========================================================================
  // Stage 4: Compute C0 for alpha blending at pre-warmstart positions
  // =========================================================================
  if (contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.computeC0", 0);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      physx::PxVec3 wA =
          (bA < numBodies)
              ? bodies[bA].prevPosition +
                    bodies[bA].prevRotation.rotate(contacts[c].contactPointA)
              : contacts[c].contactPointA;
      physx::PxVec3 wB =
          (bB < numBodies)
              ? bodies[bB].prevPosition +
                    bodies[bB].prevRotation.rotate(contacts[c].contactPointB)
              : contacts[c].contactPointB;
      physx::PxReal rawC0 = (wA - wB).dot(contacts[c].contactNormal) +
                       contacts[c].penetrationDepth;

      // Depth-adaptive C0 clamping (same as solve() path)
      const physx::PxReal c0Threshold = 0.05f;
      const physx::PxReal c0MaxDepth  = 0.20f;
      if (rawC0 < -c0Threshold) {
        physx::PxReal t = PxClamp(
            (c0MaxDepth + rawC0) / (c0MaxDepth - c0Threshold), 0.0f, 1.0f);
        rawC0 *= t;
      }
      contacts[c].C0 = rawC0;
    }
  }

  // Sort constraints for deterministic iteration order (same as solve())
  if (mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_CONSTRAINTS) &&
      numContacts > 1) {
    PX_PROFILE_ZONE("AVBD.sortConstraints", 0);
    std::sort(
        contacts, contacts + numContacts,
        [](const AvbdContactConstraint &a, const AvbdContactConstraint &b) {
          if (a.header.bodyIndexA != b.header.bodyIndexA)
            return a.header.bodyIndexA < b.header.bodyIndexA;
          if (a.header.bodyIndexB != b.header.bodyIndexB)
            return a.header.bodyIndexB < b.header.bodyIndexB;
          return a.header.type < b.header.type;
        });
  }

  // =========================================================================
  // Stage 4b: Pre-solve initialization for no-contact bodies
  //
  // Bodies without contacts don't go through solveLocalSystem, so they
  // need to be positioned at the inertial prediction (which includes
  // gravity) before the iteration loop. This is done ONCE, outside the
  // loop, so that joint GS corrections can converge without the position
  // being reset every iteration.
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.initNoContactBodies", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass <= 0.0f)
        continue;

      // Check if this body has contacts
      bool hasContacts = false;
      if (contactMap && contactMap->numBodies > 0) {
        const physx::PxU32 *cIdx = nullptr;
        physx::PxU32 cCnt = 0;
        contactMap->getBodyConstraints(i, cIdx, cCnt);
        hasContacts = (cCnt > 0);
      } else if (numContacts > 0) {
        for (physx::PxU32 c = 0; c < numContacts; ++c) {
          if (contacts[c].header.bodyIndexA == i ||
              contacts[c].header.bodyIndexB == i) {
            hasContacts = true;
            break;
          }
        }
      }

      if (!hasContacts) {
        // Snap to inertial prediction (includes gravity).
        // Joint GS in the iteration loop will refine from here.
        bodies[i].position = bodies[i].inertialPosition;
        bodies[i].rotation = bodies[i].inertialRotation;
      }
    }
  }

#if AVBD_JOINT_DEBUG
  if (doDebug) {
    printf("  After Stage 4b (init no-contact bodies):\n");
    for (physx::PxU32 i = 0; i < numBodies && i < 20; ++i) {
      if (bodies[i].invMass <= 0.0f)
        continue;
      printf("    body[%u] pos=(%.4f,%.4f,%.4f) inertial=(%.4f,%.4f,%.4f) "
             "invM=%.3f\n",
             i, bodies[i].position.x, bodies[i].position.y,
             bodies[i].position.z, bodies[i].inertialPosition.x,
             bodies[i].inertialPosition.y, bodies[i].inertialPosition.z,
             bodies[i].invMass);
    }
    // Print D6 joint info
    for (physx::PxU32 j = 0; j < numD6; ++j) {
      printf("    d6[%u] bodyA=%u bodyB=%u driveFlags=0x%X "
             "angDamping=(%.1f,%.1f,%.1f)\n",
             j, d6Joints[j].header.bodyIndexA, d6Joints[j].header.bodyIndexB,
             d6Joints[j].driveFlags, d6Joints[j].angularDamping.x,
             d6Joints[j].angularDamping.y, d6Joints[j].angularDamping.z);
    }
  }
#endif

  // =========================================================================
  // Stage 5: Main solver loop -- primal + dual per iteration (unified AL)
  //
  // Primal: Block Coordinate Descent over bodies
  //   (A) Contact constraints: full AVBD AL local system solve (3x3 or 6x6)
  //       Only for bodies WITH contacts. Bodies without contacts keep
  //       their current position (initialized above, then refined by GS).
  //   (B) Joint constraints:   Gauss-Seidel corrections (applied immediately)
  //       Each joint correction is applied to the body before processing
  //       the next joint, so subsequent joints see the updated state.
  //       Full correction (no relaxation) for equality constraints;
  //       the PBD generalized-mass denominator w naturally prevents
  //       overcorrection.
  //
  // Dual: AL multiplier updates for both contacts and joints
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.solveIterations", 0);

    // Chebyshev semi-iterative state
    const bool useChebyshev = (mConfig.chebyshevRho > 0.0f && mConfig.chebyshevRho < 1.0f);
    physx::PxReal chebyOmega = 1.0f;
    physx::PxArray<physx::PxVec3> chebyPrevPos, chebyPrevPrevPos;
    physx::PxArray<physx::PxQuat> chebyPrevRot, chebyPrevPrevRot;
    if (useChebyshev) {
      chebyPrevPos.resize(numBodies);
      chebyPrevPrevPos.resize(numBodies);
      chebyPrevRot.resize(numBodies);
      chebyPrevPrevRot.resize(numBodies);
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        chebyPrevPos[i] = bodies[i].position;
        chebyPrevPrevPos[i] = bodies[i].position;
        chebyPrevRot[i] = bodies[i].rotation;
        chebyPrevPrevRot[i] = bodies[i].rotation;
      }
    }

    // =====================================================================
    // Pre-compute body-level inertial targets for Newton-style body solve
    // (mirrors avbd_solver.cpp bodyComPred / bodyThetaPred / bodyAccumTheta)
    // =====================================================================
    physx::PxArray<physx::PxVec3> bodyComPred(numSoftBodies);
    physx::PxArray<physx::PxVec3> bodyThetaPred(numSoftBodies);
    physx::PxArray<physx::PxVec3> bodyAccumTheta(numSoftBodies);

    // Build per-particle soft contact index for O(1) lookup
    physx::PxArray<physx::PxU32> scStart(numSoftParticles + 1);
    physx::PxArray<physx::PxU32> scIdxBuf(numSoftContacts);
    if (numSoftBodies > 0) {
      physx::PxArray<physx::PxU32> scCount(numSoftParticles);
      for (physx::PxU32 i = 0; i <= numSoftParticles; ++i)
        scStart[i] = 0;
      for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
        scCount[i] = 0;
      for (physx::PxU32 ci = 0; ci < numSoftContacts; ++ci)
        scCount[softContacts[ci].particleIdx]++;
      for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
        scStart[i + 1] = scStart[i] + scCount[i];
      for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
        scCount[i] = 0;
      for (physx::PxU32 ci = 0; ci < numSoftContacts; ++ci) {
        physx::PxU32 pi = softContacts[ci].particleIdx;
        scIdxBuf[scStart[pi] + scCount[pi]++] = ci;
      }

      for (physx::PxU32 si = 0; si < numSoftBodies; ++si) {
        const AvbdSoftBody& sb = softBodies[si];
        physx::PxVec3 com(0.0f), comPred(0.0f), angMom(0.0f);
        physx::PxReal totalMass = 0.0f;
        for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
          physx::PxU32 pi = sb.particleStart + li;
          if (softParticles[pi].invMass <= 0.0f) continue;
          physx::PxReal m = 1.0f / softParticles[pi].invMass;
          com += softParticles[pi].position * m;
          comPred += softParticles[pi].predictedPosition * m;
          totalMass += m;
        }
        if (totalMass > 0.0f) {
          physx::PxReal invM = 1.0f / totalMass;
          com *= invM;
          comPred *= invM;
        }
        bodyComPred[si] = comPred;
        PxMat33 bodyI(PxZero);
        for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
          physx::PxU32 pi = sb.particleStart + li;
          if (softParticles[pi].invMass <= 0.0f) continue;
          physx::PxReal m = 1.0f / softParticles[pi].invMass;
          physx::PxVec3 r = softParticles[pi].position - com;
          physx::PxReal r2 = r.dot(r);
          bodyI += (PxMat33::createDiagonal(PxVec3(r2)) - avbdOuter(r, r)) * m;
          angMom += r.cross(softParticles[pi].velocity) * m;
        }
        physx::PxVec3 omega = bodyI.getInverse() * angMom;
        if (omega.x != omega.x) omega = PxVec3(0.0f);
        bodyThetaPred[si] = omega * dt;
        bodyAccumTheta[si] = PxVec3(0.0f);
      }
    }

    const physx::PxU32 baseIters = (iterationOverride > 0)
        ? iterationOverride : mConfig.innerIterations;
    const physx::PxU32 jointIterations =
        (mStats.numJoints > 0)
            ? physx::PxMax(baseIters, physx::PxU32(8))
            : baseIters;
    const bool enableEarlyStop =
      (mConfig.positionTolerance > 0.0f && jointIterations > 1);
    const physx::PxU32 minIterations =
      physx::PxMin(jointIterations, physx::PxU32(mStats.numJoints > 0 ? 8 : 4));
    const physx::PxReal rotationTolerance =
      physx::PxMax(4.0f * mConfig.positionTolerance, 1e-4f);
    physx::PxU32 consecutiveConvergedIterations = 0;
    physx::PxArray<physx::PxVec3> earlyStopPrevPos;
    physx::PxArray<physx::PxQuat> earlyStopPrevRot;
    if (enableEarlyStop) {
      earlyStopPrevPos.resize(numBodies);
      earlyStopPrevRot.resize(numBodies);
    }

    for (physx::PxU32 iter = 0; iter < jointIterations; ++iter) {
      // Save pre-iteration state for Chebyshev
      if (useChebyshev) {
        for (physx::PxU32 i = 0; i < numBodies; ++i) {
          chebyPrevPrevPos[i] = chebyPrevPos[i];
          chebyPrevPrevRot[i] = chebyPrevRot[i];
          chebyPrevPos[i] = bodies[i].position;
          chebyPrevRot[i] = bodies[i].rotation;
        }
      }
      if (enableEarlyStop) {
        for (physx::PxU32 i = 0; i < numBodies; ++i) {
          earlyStopPrevPos[i] = bodies[i].position;
          earlyStopPrevRot[i] = bodies[i].rotation;
        }
      }

      // --- Primal step: block descent over bodies ---
      {
        PX_PROFILE_ZONE("AVBD.blockDescentWithJoints", 0);

        // Deterministic body ordering (same as blockDescentIteration)
        const bool useDeterministicOrder =
            mConfig.isDeterministic() &&
            (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);

        physx::PxArray<physx::PxU32> bodyOrder;
        if (useDeterministicOrder) {
          bodyOrder.resize(numBodies);
          for (physx::PxU32 bi = 0; bi < numBodies; ++bi)
            bodyOrder[bi] = bi;
          std::sort(bodyOrder.begin(), bodyOrder.end(),
                    [&bodies](physx::PxU32 a, physx::PxU32 b) {
                      return bodies[a].invMass > bodies[b].invMass;
                    });
        }
        const physx::PxU32 *orderPtr =
            useDeterministicOrder ? bodyOrder.begin() : nullptr;

        const bool useParallel = mConfig.enableParallelization
            && !useDeterministicOrder
            && numBodies >= AVBD_PARALLEL_MIN_ITEMS;

        auto solveBody = [&](physx::PxU32 idx) {
          const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
          if (bodies[i].invMass <= 0.0f)
            return;
          solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                     numContacts, d6Joints, numD6, gearJoints,
                                     numGear, dt, invDt2, contactMap, d6Map,
                                     gearMap);
        };

        if (useParallel) {
          avbdParallelFor(0u, numBodies, solveBody);
        } else {
          for (physx::PxU32 idx = 0; idx < numBodies; ++idx)
            solveBody(idx);
        }

        // --- Body-level 6x6 solve for soft bodies (Newton-style) ---
        if (numSoftBodies > 0 && numSoftContacts > 0) {
          PX_PROFILE_ZONE("AVBD.softBodyLevel6x6", 0);
          for (physx::PxU32 si = 0; si < numSoftBodies; ++si) {
            const AvbdSoftBody& sb = softBodies[si];
            physx::PxVec3 com(0.0f);
            physx::PxReal bodyMass = 0.0f;
            for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
              physx::PxU32 pi = sb.particleStart + li;
              if (softParticles[pi].invMass <= 0.0f) continue;
              physx::PxReal m = 1.0f / softParticles[pi].invMass;
              com += softParticles[pi].position * m;
              bodyMass += m;
            }
            if (bodyMass <= 0.0f) continue;
            com *= (1.0f / bodyMass);

            physx::PxU32 bodyContactCount = 0;
            for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
              physx::PxU32 pi = sb.particleStart + li;
              bodyContactCount += scStart[pi + 1] - scStart[pi];
            }
            if (bodyContactCount == 0) continue;

            PxMat33 bodyInertia(PxZero);
            for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
              physx::PxU32 pi = sb.particleStart + li;
              if (softParticles[pi].invMass <= 0.0f) continue;
              physx::PxReal m = 1.0f / softParticles[pi].invMass;
              physx::PxVec3 r = softParticles[pi].position - com;
              physx::PxReal r2 = r.dot(r);
              bodyInertia += (PxMat33::createDiagonal(PxVec3(r2)) - avbdOuter(r, r)) * m;
            }

            physx::PxReal bodyMassDtSq = bodyMass * invDt2;
            PxMat33 A_ll = PxMat33::createDiagonal(PxVec3(bodyMassDtSq));
            PxMat33 A_la(PxZero), A_al(PxZero);
            physx::PxReal reg = 1e-4f * bodyMassDtSq;
            PxMat33 A_aa = bodyInertia * invDt2 + PxMat33::createDiagonal(PxVec3(reg));

            physx::PxVec3 g_l = (com - bodyComPred[si]) * bodyMassDtSq;
            physx::PxVec3 g_a = (bodyInertia * invDt2) * (bodyAccumTheta[si] - bodyThetaPred[si]);

            for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
              physx::PxU32 pi = sb.particleStart + li;
              physx::PxVec3 r = softParticles[pi].position - com;
              for (physx::PxU32 k = scStart[pi]; k < scStart[pi + 1]; ++k) {
                const AvbdSoftContact& sc = softContacts[scIdxBuf[k]];
                physx::PxVec3 n = sc.normal;
                physx::PxReal violation;
                if (sc.rigidBodyIdx == PX_MAX_U32)
                  violation = softParticles[pi].position.dot(n);
                else
                  violation = (softParticles[pi].position - sc.surfacePoint).dot(n) - sc.margin;

                physx::PxReal pen = sc.k;
                physx::PxVec3 rCrossN = r.cross(n);
                A_ll += avbdOuter(n, n) * pen;
                A_la += avbdOuter(n, rCrossN) * pen;
                A_al += avbdOuter(rCrossN, n) * pen;
                A_aa += avbdOuter(rCrossN, rCrossN) * pen;

                physx::PxReal f = physx::PxMin(0.0f, pen * violation);
                if (f < 0.0f) {
                  g_l += n * f;
                  g_a += rCrossN * f;
                }
              }
            }

            PxMat33 A_ll_inv = A_ll.getInverse();
            PxMat33 S = A_aa - A_al * A_ll_inv * A_la;
            physx::PxVec3 deltaTheta = S.getInverse() * (g_a - A_al * A_ll_inv * g_l);
            physx::PxVec3 deltaPos = A_ll_inv * (g_l - A_la * deltaTheta);

            if (deltaPos.x != deltaPos.x || deltaTheta.x != deltaTheta.x) continue;

            physx::PxReal thetaMag = deltaTheta.magnitude();
            if (thetaMag > 0.5f) deltaTheta *= (0.5f / thetaMag);

            for (physx::PxU32 li = 0; li < sb.particleCount; ++li) {
              physx::PxU32 pi = sb.particleStart + li;
              if (softParticles[pi].invMass <= 0.0f) continue;
              physx::PxVec3 r = softParticles[pi].position - com;
              softParticles[pi].position -= deltaPos + deltaTheta.cross(r);
            }
            bodyAccumTheta[si] -= deltaTheta;
          }
        }

        // --- VBD soft particle primal (3x3 block coordinate descent) ---
        if (numSoftParticles > 0 && numSoftBodies > 0) {
          PX_PROFILE_ZONE("AVBD.softParticlePrimal", 0);

          auto solveSP = [&](physx::PxU32 spi) {
            if (softParticles[spi].invMass <= 0.0f) return;
            solveSoftParticle(spi, softParticles, numSoftParticles,
                              bodies, numBodies, softBodies, numSoftBodies,
                              softContacts, numSoftContacts, dt, invDt2);
          };

          if (useParallel) {
            avbdParallelFor(0u, numSoftParticles, solveSP);
          } else {
            for (physx::PxU32 spi = 0; spi < numSoftParticles; ++spi)
              solveSP(spi);
          }
        }

        mStats.totalIterations++;
      }

      // --- Dual step: AL multiplier updates ---
      //
      // CONTACTS: update every iteration. The unilateral clamp
      //   f = min(0, pen*C + lambda) prevents overcorrection, so frequent
      //   dual updates are safe and improve convergence.
      //
      // JOINTS (D6 + Gear): 3-mechanism ADMM-safe AL dual.
      //   (A) Primal auto-boost: effectiveRho = max(rho, M/h^2) ensures
      //       penalty is always >= body inertia for good convergence.
      //   (B) ADMM-safe dual step: rhoDual = min(Mh2, rho^2/(rho+Mh2))
      //       prevents dual overshoot for both light and heavy bodies.
      //   (C) Lambda decay: lambda = 0.99*lambda + rhoDual*C acts as a
      //       leaky integrator that damps oscillation modes.
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt);

        // ---------------------------------------------------------------
        // D6, Gear: ADMM-safe dual + lambda decay
        //
        // Three mechanisms ensure stable AL convergence:
        //   (A) effectiveRho = max(rho, M/h^2) in primal (above)
        //   (B) rhoDual = min(Mh2, rho^2/(rho+Mh2)) -- safe step size
        //   (C) lambda = decay*lambda + rhoDual*C -- leaky integrator
        // ---------------------------------------------------------------
        {
          const physx::PxReal lambdaDecay = 0.99f;

          auto getBodyMass = [&](physx::PxU32 idx) -> physx::PxReal {
            return (idx == 0xFFFFFFFF || idx >= numBodies)
                       ? 0.0f
                       : (bodies[idx].invMass > 1e-8f
                              ? 1.0f / bodies[idx].invMass
                              : 0.0f);
          };
          auto computeRhoDual = [&](physx::PxU32 idxA, physx::PxU32 idxB,
                                    physx::PxReal rho) -> physx::PxReal {
            physx::PxReal mA = getBodyMass(idxA);
            physx::PxReal mB = getBodyMass(idxB);
            physx::PxReal mEff;
            if (mA <= 0.0f)
              mEff = mB;
            else if (mB <= 0.0f)
              mEff = mA;
            else
              mEff = physx::PxMin(mA, mB);
            if (mEff <= 0.0f)
              return 0.0f;
            physx::PxReal Mh2 = mEff * invDt2;
            physx::PxReal admm_step = rho * rho / (rho + Mh2);
            return physx::PxMin(Mh2, admm_step);
          };

          // D6 joints
          for (physx::PxU32 j = 0; j < numD6; ++j) {
            AvbdD6JointConstraint &jnt = d6Joints[j];
            physx::PxReal rhoDual = computeRhoDual(
                jnt.header.bodyIndexA, jnt.header.bodyIndexB, jnt.header.rho);
            if (rhoDual <= 0.0f)
              continue;
            bool aStatic = (jnt.header.bodyIndexA == 0xFFFFFFFF ||
                            jnt.header.bodyIndexA >= numBodies);
            bool bStatic = (jnt.header.bodyIndexB == 0xFFFFFFFF ||
                            jnt.header.bodyIndexB >= numBodies);
            physx::PxVec3 wA =
                aStatic ? jnt.anchorA
                        : bodies[jnt.header.bodyIndexA].position +
                              bodies[jnt.header.bodyIndexA].rotation.rotate(
                                  jnt.anchorA);
            physx::PxVec3 wB =
                bStatic ? jnt.anchorB
                        : bodies[jnt.header.bodyIndexB].position +
                              bodies[jnt.header.bodyIndexB].rotation.rotate(
                                  jnt.anchorB);
            physx::PxQuat rotA = aStatic
                                     ? physx::PxQuat(physx::PxIdentity)
                                     : bodies[jnt.header.bodyIndexA].rotation;
            physx::PxQuat rotB = bStatic
                                     ? physx::PxQuat(physx::PxIdentity)
                                     : bodies[jnt.header.bodyIndexB].rotation;
            physx::PxVec3 posViol = wA - wB;

            // Compute joint-frame axes (match primal axis selection)
            physx::PxQuat jointFrameA_dual =
                aStatic
                    ? jnt.localFrameA
                    : bodies[jnt.header.bodyIndexA].rotation * jnt.localFrameA;
            {
              physx::PxReal qm2 = jointFrameA_dual.magnitudeSquared();
              if (qm2 > 1e-8f && PxIsFinite(qm2))
                jointFrameA_dual *= 1.0f / physx::PxSqrt(qm2);
            }

            const bool linAllLocked = (jnt.linearMotion == 0);
            physx::PxVec3 linearAxes[3];
            if (linAllLocked) {
              linearAxes[0] = physx::PxVec3(1.0f, 0.0f, 0.0f);
              linearAxes[1] = physx::PxVec3(0.0f, 1.0f, 0.0f);
              linearAxes[2] = physx::PxVec3(0.0f, 0.0f, 1.0f);
            } else {
              linearAxes[0] = jointFrameA_dual.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
              linearAxes[1] = jointFrameA_dual.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f));
              linearAxes[2] = jointFrameA_dual.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f));
            }

            for (int axis = 0; axis < 3; ++axis) {
              physx::PxU32 motion = jnt.getLinearMotion(axis);
              if (motion == 2) // FREE
                continue;

              physx::PxReal Ck = posViol.dot(linearAxes[axis]);

              if (motion == 0) { // LOCKED
                jnt.lambdaLinear[axis] = jnt.lambdaLinear[axis] * lambdaDecay +
                                         Ck * rhoDual;
              } else if (motion == 1) { // LIMITED
                physx::PxReal dist = -posViol.dot(linearAxes[axis]);
                physx::PxReal limitViol = 0.0f;
                if (dist < jnt.linearLimitLower[axis])
                  limitViol = dist - jnt.linearLimitLower[axis];
                else if (dist > jnt.linearLimitUpper[axis])
                  limitViol = dist - jnt.linearLimitUpper[axis];

                physx::PxReal newLam =
                    jnt.lambdaLinear[axis] * lambdaDecay + limitViol * rhoDual;

                if (jnt.linearLimitLower[axis] < jnt.linearLimitUpper[axis]) {
                  physx::PxReal signRef =
                      (physx::PxAbs(limitViol) > 1e-6f)
                          ? limitViol
                          : ((physx::PxAbs(jnt.lambdaLinear[axis]) > 1e-6f)
                                 ? jnt.lambdaLinear[axis]
                                 : 0.0f);
                  if (signRef > 0.0f)
                    jnt.lambdaLinear[axis] = physx::PxMax(0.0f, newLam);
                  else if (signRef < 0.0f)
                    jnt.lambdaLinear[axis] = physx::PxMin(0.0f, newLam);
                  else
                    jnt.lambdaLinear[axis] = 0.0f;
                } else {
                  jnt.lambdaLinear[axis] = newLam;
                }
              }
            }

            // Detect revolute pattern for cross-product axis alignment
            const physx::PxU32 twistMotion_d = jnt.getAngularMotion(0);
            const physx::PxU32 swing1Motion_d = jnt.getAngularMotion(1);
            const physx::PxU32 swing2Motion_d = jnt.getAngularMotion(2);
            const bool isRevolutePattern_d =
                (twistMotion_d != 0) && (swing1Motion_d == 0) &&
                (swing2Motion_d == 0);

            if (isRevolutePattern_d) {
              // Cross-product axis alignment dual
              physx::PxQuat worldFrameA_d = rotA * jnt.localFrameA;
              physx::PxQuat worldFrameB_d = rotB * jnt.localFrameB;
              physx::PxVec3 worldTwistA =
                  worldFrameA_d.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
              physx::PxVec3 worldTwistB =
                  worldFrameB_d.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
              physx::PxVec3 axisViol = worldTwistA.cross(worldTwistB);

              physx::PxVec3 perp1, perp2;
              if (physx::PxAbs(worldTwistA.x) < 0.9f)
                perp1 = worldTwistA.cross(physx::PxVec3(1.0f, 0.0f, 0.0f));
              else
                perp1 = worldTwistA.cross(physx::PxVec3(0.0f, 1.0f, 0.0f));
              physx::PxReal p1Len = perp1.magnitude();
              if (p1Len > 1e-6f) perp1 *= (1.0f / p1Len);
              perp2 = worldTwistA.cross(perp1);
              physx::PxReal p2Len = perp2.magnitude();
              if (p2Len > 1e-6f) perp2 *= (1.0f / p2Len);

              physx::PxReal err1 = axisViol.dot(perp1);
              physx::PxReal err2 = axisViol.dot(perp2);

              jnt.lambdaAngular[1] =
                  jnt.lambdaAngular[1] * lambdaDecay + err1 * rhoDual;
              jnt.lambdaAngular[2] =
                  jnt.lambdaAngular[2] * lambdaDecay + err2 * rhoDual;

              // Twist axis (0) if LIMITED
              if (twistMotion_d == 1) {
                physx::PxReal angErr =
                    jnt.computeAngularError(rotA, rotB, 0);
                physx::PxReal limitViol =
                    jnt.computeAngularLimitViolation(angErr, 0);
                physx::PxReal newLam =
                    jnt.lambdaAngular[0] * lambdaDecay + limitViol * rhoDual;

                if (jnt.angularLimitLower[0] < jnt.angularLimitUpper[0]) {
                  if (limitViol > 0.0f || jnt.lambdaAngular[0] > 0.0f)
                    jnt.lambdaAngular[0] = physx::PxMax(0.0f, newLam);
                  else if (limitViol < 0.0f || jnt.lambdaAngular[0] < 0.0f)
                    jnt.lambdaAngular[0] = physx::PxMin(0.0f, newLam);
                  else
                    jnt.lambdaAngular[0] = 0.0f;
                } else {
                  jnt.lambdaAngular[0] = newLam;
                }
              }
            } else {
              // Generic per-axis dual
              for (int axis = 0; axis < 3; ++axis) {
                physx::PxU32 motion = jnt.getAngularMotion(axis);
                if (motion == 2) // FREE
                  continue;

                if (motion == 0) { // LOCKED
                  physx::PxReal angErr =
                      jnt.computeAngularError(rotA, rotB, axis);
                  jnt.lambdaAngular[axis] =
                      jnt.lambdaAngular[axis] * lambdaDecay + angErr * rhoDual;
                } else if (motion == 1) { // LIMITED
                  physx::PxReal angErr =
                      jnt.computeAngularError(rotA, rotB, axis);
                  physx::PxReal limitViol =
                      jnt.computeAngularLimitViolation(angErr, axis);
                  physx::PxReal newLam =
                      jnt.lambdaAngular[axis] * lambdaDecay +
                      limitViol * rhoDual;

                  if (jnt.angularLimitLower[axis] <
                      jnt.angularLimitUpper[axis]) {
                    if (limitViol > 0.0f || jnt.lambdaAngular[axis] > 0.0f) {
                      jnt.lambdaAngular[axis] = physx::PxMax(0.0f, newLam);
                    } else if (limitViol < 0.0f ||
                               jnt.lambdaAngular[axis] < 0.0f) {
                      jnt.lambdaAngular[axis] = physx::PxMin(0.0f, newLam);
                    } else {
                      jnt.lambdaAngular[axis] = 0.0f;
                    }
                  } else {
                    jnt.lambdaAngular[axis] = newLam;
                  }
                }
              }
            }

            // --- Cone limit dual update ---
            if (jnt.coneAngleLimit > 0.0f) {
              physx::PxVec3 worldAxisA = (rotA * jnt.localFrameA).rotate(
                  physx::PxVec3(1.0f, 0.0f, 0.0f));
              physx::PxVec3 worldAxisB = (rotB * jnt.localFrameB).rotate(
                  physx::PxVec3(1.0f, 0.0f, 0.0f));
              physx::PxReal dotAB = physx::PxClamp(worldAxisA.dot(worldAxisB),
                                                    -1.0f, 1.0f);
              physx::PxReal coneAngle = physx::PxAcos(dotAB);
              physx::PxReal coneViol = coneAngle - jnt.coneAngleLimit;

              // Unilateral: coneLambda -= violation * rhoDual, clamped to <= 0
              jnt.coneLambda -= coneViol * rhoDual;
              jnt.coneLambda =
                  physx::PxMax(-1e9f, physx::PxMin(0.0f, jnt.coneLambda));
            }

            // --- Drive AL dual update ---
            physx::PxReal dt2 = dt * dt;

            // Joint frame A in world space
            physx::PxQuat jointFrameA =
                aStatic
                    ? jnt.localFrameA
                    : bodies[jnt.header.bodyIndexA].rotation * jnt.localFrameA;
            physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
            if (qMag2 > 1e-8f && PxIsFinite(qMag2))
              jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

            // Linear velocity drive dual
            if ((jnt.driveFlags & 0x7) != 0) {
              // Body displacements from start-of-step
              physx::PxVec3 dxA =
                  aStatic ? physx::PxVec3(0.0f)
                          : (bodies[jnt.header.bodyIndexA].position -
                             bodies[jnt.header.bodyIndexA].prevPosition);
              physx::PxVec3 dxB =
                  bStatic ? physx::PxVec3(0.0f)
                          : (bodies[jnt.header.bodyIndexB].position -
                             bodies[jnt.header.bodyIndexB].prevPosition);

              for (int a = 0; a < 3; ++a) {
                if ((jnt.driveFlags & (1 << a)) == 0)
                  continue;
                physx::PxReal damping = (&jnt.linearDamping.x)[a];
                if (damping <= 0.0f)
                  continue;

                physx::PxVec3 localAxis(0.0f);
                (&localAxis.x)[a] = 1.0f;
                physx::PxVec3 wAxis = jointFrameA.rotate(localAxis);

                // Drive velocity is in joint local space, rotate to world
                // space
                physx::PxVec3 worldTarget =
                    jointFrameA.rotate(jnt.driveLinearVelocity) * dt;
                physx::PxReal C =
                    (dxB.dot(wAxis) - dxA.dot(wAxis)) - worldTarget.dot(wAxis);

                const physx::PxVec3 rAWorld =
                  aStatic ? physx::PxVec3(0.0f)
                      : bodies[jnt.header.bodyIndexA].rotation.rotate(
                          jnt.anchorA);
                const physx::PxVec3 rBWorld =
                  bStatic ? physx::PxVec3(0.0f)
                      : bodies[jnt.header.bodyIndexB].rotation.rotate(
                          jnt.anchorB);
                const AvbdSolverBody *bodyARef =
                  aStatic ? nullptr : &bodies[jnt.header.bodyIndexA];
                const AvbdSolverBody *bodyBRef =
                  bStatic ? nullptr : &bodies[jnt.header.bodyIndexB];
                physx::PxReal rhoDualDrive = physx::PxMin(damping / dt2, rhoDual);
                if (jnt.isLinearAccelerationDrive(a)) {
                  const physx::PxReal driveScale =
                    computeLinearDriveRecipResponse(bodyARef, bodyBRef,
                                   rAWorld, rBWorld, wAxis);
                  const physx::PxReal stiffness = (&jnt.linearStiffness.x)[a];
                  const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
                  const physx::PxReal implicitScale =
                    1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
                  rhoDualDrive = physx::PxMin((damping * driveScale * implicitScale) / dt2,
                                rhoDual);
                }
                (&jnt.lambdaDriveLinear.x)[a] =
                    (&jnt.lambdaDriveLinear.x)[a] * lambdaDecay +
                    rhoDualDrive * C;
              }
            }

            // Angular velocity drive dual
            if ((jnt.driveFlags & 0x38) != 0) {
              // Angular displacements from start-of-step
              physx::PxVec3 dThetaA(0.0f), dThetaB(0.0f);
              if (!aStatic) {
                physx::PxQuat dqA =
                    bodies[jnt.header.bodyIndexA].rotation *
                    bodies[jnt.header.bodyIndexA].prevRotation.getConjugate();
                if (dqA.w < 0.0f)
                  dqA = -dqA;
                dThetaA = physx::PxVec3(dqA.x, dqA.y, dqA.z) * 2.0f;
              }
              if (!bStatic) {
                physx::PxQuat dqB =
                    bodies[jnt.header.bodyIndexB].rotation *
                    bodies[jnt.header.bodyIndexB].prevRotation.getConjugate();
                if (dqB.w < 0.0f)
                  dqB = -dqB;
                dThetaB = physx::PxVec3(dqB.x, dqB.y, dqB.z) * 2.0f;
              }

              physx::PxVec3 relDW = dThetaB - dThetaA;
              physx::PxVec3 worldAngTarget =
                  jointFrameA.rotate(jnt.driveAngularVelocity) * dt;

              bool slerpDrive = (jnt.driveFlags & 0x20) != 0;
              if (slerpDrive) {
                physx::PxReal damping =
                    jnt.angularDamping.z; // SLERP uses Z damping slot
                if (damping > 0.0f) {
                  const AvbdSolverBody *bodyARef =
                    aStatic ? nullptr : &bodies[jnt.header.bodyIndexA];
                  const AvbdSolverBody *bodyBRef =
                    bStatic ? nullptr : &bodies[jnt.header.bodyIndexB];
                    physx::PxReal rhoDualDrive = physx::PxMin(damping / dt2, rhoDual);
                    if (jnt.isAngularAccelerationDrive(2)) {
                    const physx::PxReal driveScale =
                      computeAngularDriveRecipResponse(bodyARef, bodyBRef,
                                       physx::PxVec3(1.0f, 0.0f, 0.0f));
                    const physx::PxReal stiffness = jnt.angularStiffness.z;
                    const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
                    const physx::PxReal implicitScale =
                      1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
                    rhoDualDrive = physx::PxMin((damping * driveScale * implicitScale) / dt2,
                                  rhoDual);
                    }
                  for (int k = 0; k < 3; ++k) {
                    physx::PxReal C = (&relDW.x)[k] - (&worldAngTarget.x)[k];
                    (&jnt.lambdaDriveAngular.x)[k] =
                        (&jnt.lambdaDriveAngular.x)[k] * lambdaDecay +
                        rhoDualDrive * C;
                  }
                }
              } else {
                struct AxisDrive {
                  int bit;
                  int dampIdx;
                  physx::PxVec3 localAxis;
                };
                const AxisDrive axes[3] = {
                    {3, 0, physx::PxVec3(1.0f, 0.0f, 0.0f)},
                    {4, 1, physx::PxVec3(0.0f, 1.0f, 0.0f)},
                    {5, 2, physx::PxVec3(0.0f, 0.0f, 1.0f)},
                };

                for (int a = 0; a < 3; ++a) {
                  if ((jnt.driveFlags & (1 << axes[a].bit)) == 0)
                    continue;
                  physx::PxReal damping =
                      (&jnt.angularDamping.x)[axes[a].dampIdx];
                  if (damping <= 0.0f)
                    continue;

                  physx::PxVec3 wAxis = jointFrameA.rotate(axes[a].localAxis);
          const AvbdSolverBody *bodyARef =
            aStatic ? nullptr : &bodies[jnt.header.bodyIndexA];
          const AvbdSolverBody *bodyBRef =
            bStatic ? nullptr : &bodies[jnt.header.bodyIndexB];
                  // PhysX TGS convention: Twist/Swing target velocities are
                  // applied as (wA - wB), meaning wB - wA = -target. SLERP is
                  // applied as wB - wA = target, which is handled above.
                  physx::PxReal targetOmega_dt = -worldAngTarget.dot(wAxis);
                  physx::PxReal C = relDW.dot(wAxis) - targetOmega_dt;

            physx::PxReal rhoDualDrive = physx::PxMin(damping / dt2, rhoDual);
            if (jnt.isAngularAccelerationDrive(axes[a].dampIdx)) {
            const physx::PxReal driveScale =
              computeAngularDriveRecipResponse(bodyARef, bodyBRef, wAxis);
            const physx::PxReal stiffness =
              (&jnt.angularStiffness.x)[axes[a].dampIdx];
            const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
            const physx::PxReal implicitScale =
              1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
            rhoDualDrive = physx::PxMin((damping * driveScale * implicitScale) / dt2,
                          rhoDual);
            }
                  (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] =
                      (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] *
                          lambdaDecay +
                      rhoDualDrive * C;
                }
              }
            }
          }
        }

        // Gear joints: AL dual
        for (physx::PxU32 j = 0; j < numGear; ++j)
          updateGearJointMultiplier(gearJoints[j], bodies, numBodies, mConfig);

        // Soft body AVBD dual update (penalty growth only)
        if (numSoftParticles > 0 && numSoftBodies > 0) {
          PX_PROFILE_ZONE("AVBD.softDual", 0);
          updateSoftDual(softParticles, numSoftParticles, bodies, numBodies,
                         softBodies, numSoftBodies, softContacts, numSoftContacts,
                         mConfig.avbdBeta);
        }
      }

      // Chebyshev semi-iterative position/rotation relaxation
      if (useChebyshev && iter >= 2) {
        const physx::PxReal rhoSq = mConfig.chebyshevRho * mConfig.chebyshevRho;
        if (iter == 2)
          chebyOmega = 2.0f / (2.0f - rhoSq);
        else
          chebyOmega = 1.0f / (1.0f - rhoSq * chebyOmega / 4.0f);
        chebyOmega = physx::PxClamp(chebyOmega, 1.0f, 2.0f);

        for (physx::PxU32 i = 0; i < numBodies; ++i) {
          if (bodies[i].invMass <= 0.0f) continue;
          // Position relaxation
          bodies[i].position = chebyPrevPrevPos[i] +
              (bodies[i].position - chebyPrevPrevPos[i]) * chebyOmega;
          // Rotation: quaternion linear blend + normalize
          physx::PxQuat qPrev = chebyPrevPrevRot[i];
          physx::PxQuat qCur = bodies[i].rotation;
          if (qPrev.dot(qCur) < 0.0f) qCur = -qCur;
          physx::PxQuat qBlend(
              qPrev.x + chebyOmega * (qCur.x - qPrev.x),
              qPrev.y + chebyOmega * (qCur.y - qPrev.y),
              qPrev.z + chebyOmega * (qCur.z - qPrev.z),
              qPrev.w + chebyOmega * (qCur.w - qPrev.w));
          bodies[i].rotation = qBlend.getNormalized();
        }
      }

      if (enableEarlyStop) {
        physx::PxReal maxPositionDelta = 0.0f;
        physx::PxReal maxRotationDelta = 0.0f;
        computeMaxPoseDeltas(bodies, numBodies, earlyStopPrevPos,
                             earlyStopPrevRot, maxPositionDelta,
                             maxRotationDelta);

        if ((iter + 1) >= minIterations &&
            maxPositionDelta <= mConfig.positionTolerance &&
            maxRotationDelta <= rotationTolerance) {
          consecutiveConvergedIterations++;
          if (consecutiveConvergedIterations >= 2)
            break;
        } else {
          consecutiveConvergedIterations = 0;
        }
      }
    } // end iteration loop
  }

  // =========================================================================
  // Stage 5b: Post-solve motor drives for D6 joints (revolute motor)
  //
  // Matches the reference's "Stage 6: Motor drives for RevoluteJoints"
  // approach: after all constraint iterations converge, directly apply
  // clamped torque to bodies. This avoids coupling the motor with position/
  // gear constraints in the Hessian which causes ADMM oscillation.
  // =========================================================================
  if (d6Joints && numD6 > 0) {
    PX_PROFILE_ZONE("AVBD.motorDrives", 0);

    for (physx::PxU32 j = 0; j < numD6; ++j) {
      AvbdD6JointConstraint &jnt = d6Joints[j];
      if (!jnt.motorEnabled || jnt.motorMaxForce <= 0.0f)
        continue;

      const physx::PxU32 idxA = jnt.header.bodyIndexA;
      const physx::PxU32 idxB = jnt.header.bodyIndexB;
      const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
      const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);
      if (isAStatic && isBStatic)
        continue;

      // Twist axis = X-axis of localFrameA in world space
      physx::PxVec3 worldAxis =
          isAStatic ? jnt.localFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
                    : (bodies[idxA].rotation * jnt.localFrameA)
                          .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
      worldAxis.normalize();

      // Apply motor to body B (the dynamic gear body for world-anchored joints)
      if (!isBStatic) {
        AvbdSolverBody &bodyB = bodies[idxB];

        // Current angular velocity from position-level solver
        physx::PxQuat deltaQB =
            bodyB.rotation * bodyB.prevRotation.getConjugate();
        if (deltaQB.w < 0.0f)
          deltaQB = -deltaQB;
        physx::PxVec3 currentAngVel =
            physx::PxVec3(deltaQB.x, deltaQB.y, deltaQB.z) * (2.0f * invDt);
        physx::PxReal currentAxisVel = currentAngVel.dot(worldAxis);

        physx::PxReal velocityError =
            jnt.motorTargetVelocity - currentAxisVel;

        physx::PxVec3 invITimesAxis =
            bodyB.invInertiaWorld.transform(worldAxis);
        physx::PxReal effectiveInvInertia = worldAxis.dot(invITimesAxis);
        if (effectiveInvInertia < 1e-10f)
          continue;

        physx::PxReal effectiveInertia = 1.0f / effectiveInvInertia;
        physx::PxReal requiredTorque =
            effectiveInertia * velocityError * invDt;
        physx::PxReal clampedTorque = physx::PxClamp(
            requiredTorque, -jnt.motorMaxForce, jnt.motorMaxForce);
        physx::PxReal angularAccel = clampedTorque * effectiveInvInertia;
        physx::PxReal deltaAngle = angularAccel * dt * dt;

        // Apply rotation to body B
        physx::PxReal ha = deltaAngle * 0.5f;
        physx::PxQuat dRot(worldAxis.x * physx::PxSin(ha),
                           worldAxis.y * physx::PxSin(ha),
                           worldAxis.z * physx::PxSin(ha),
                           physx::PxCos(ha));
        bodyB.rotation = (dRot * bodyB.rotation).getNormalized();
      }

      // Apply opposite rotation to body A if dynamic
      if (!isAStatic) {
        AvbdSolverBody &bodyA = bodies[idxA];

        physx::PxQuat deltaQA =
            bodyA.rotation * bodyA.prevRotation.getConjugate();
        if (deltaQA.w < 0.0f)
          deltaQA = -deltaQA;
        physx::PxVec3 currentAngVelA =
            physx::PxVec3(deltaQA.x, deltaQA.y, deltaQA.z) * (2.0f * invDt);
        physx::PxReal currentAxisVelA = currentAngVelA.dot(worldAxis);
        physx::PxReal velocityErrorA =
            -jnt.motorTargetVelocity - currentAxisVelA;

        physx::PxVec3 invITimesAxisA =
            bodyA.invInertiaWorld.transform(worldAxis);
        physx::PxReal effectiveInvInertiaA = worldAxis.dot(invITimesAxisA);
        if (effectiveInvInertiaA > 1e-10f) {
          physx::PxReal effectiveInertiaA = 1.0f / effectiveInvInertiaA;
          physx::PxReal requiredTorqueA =
              effectiveInertiaA * velocityErrorA * invDt;
          physx::PxReal clampedTorqueA = physx::PxClamp(
              requiredTorqueA, -jnt.motorMaxForce, jnt.motorMaxForce);
          physx::PxReal deltaAngleA =
              clampedTorqueA * effectiveInvInertiaA * dt * dt;
          physx::PxReal haA = deltaAngleA * 0.5f;
          physx::PxQuat dRotA(worldAxis.x * physx::PxSin(haA),
                              worldAxis.y * physx::PxSin(haA),
                              worldAxis.z * physx::PxSin(haA),
                              physx::PxCos(haA));
          bodyA.rotation = (dRotA * bodyA.rotation).getNormalized();
        }
      }
    }
  }

  // =========================================================================
  // Stage 6: Velocity update from position change
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;

        // Linear velocity: v = (x_new - x_old) / dt
        bodies[i].linearVelocity =
            (bodies[i].position - bodies[i].prevPosition) * invDt;
        bodies[i].linearVelocity *= mConfig.velocityDamping;

        // Angular velocity from quaternion difference
        physx::PxQuat deltaQ =
            bodies[i].rotation * bodies[i].prevRotation.getConjugate();
        if (deltaQ.w < 0.0f)
          deltaQ = -deltaQ;
        bodies[i].angularVelocity =
            physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
        bodies[i].angularVelocity *= mConfig.angularDamping;

        // Per-body damping (exponential decay per timestep)
        if (bodies[i].linearDamping > 0.0f) {
          physx::PxReal linDecay =
              1.0f / (1.0f + bodies[i].linearDamping * dt);
          bodies[i].linearVelocity *= linDecay;
        }
        if (bodies[i].angularDampingBody > 0.0f) {
          physx::PxReal angDecay =
              1.0f / (1.0f + bodies[i].angularDampingBody * dt);
          bodies[i].angularVelocity *= angDecay;
        }

        // Per-body velocity capping
        physx::PxReal linVelSq =
            bodies[i].linearVelocity.magnitudeSquared();
        if (linVelSq > bodies[i].maxLinearVelocitySq &&
            bodies[i].maxLinearVelocitySq > 0.0f) {
          bodies[i].linearVelocity *=
              physx::PxSqrt(bodies[i].maxLinearVelocitySq / linVelSq);
        }
        physx::PxReal angVelSq =
            bodies[i].angularVelocity.magnitudeSquared();
        if (angVelSq > bodies[i].maxAngularVelocitySq &&
            bodies[i].maxAngularVelocitySq > 0.0f) {
          bodies[i].angularVelocity *=
              physx::PxSqrt(bodies[i].maxAngularVelocitySq / angVelSq);
        }
      }
    }

    // D6 joint angular drive damping is now handled entirely by the AVBD
    // AL constraint in solveLocalSystemWithJoints. No extra velocity
    // attenuation is needed here.

    // Soft particle velocity update
    for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
      softParticles[i].updateVelocityFromPosition(invDt);
  }

#if AVBD_JOINT_DEBUG
  if (doDebug) {
    printf("  After Stage 7 (final positions):\n");
    for (physx::PxU32 i = 0; i < numBodies && i < 20; ++i) {
      if (bodies[i].invMass <= 0.0f)
        continue;
      printf("    body[%u] pos=(%.4f,%.4f,%.4f) vel=(%.4f,%.4f,%.4f)\n", i,
             bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
             bodies[i].linearVelocity.x, bodies[i].linearVelocity.y,
             bodies[i].linearVelocity.z);
    }
  }
  s_avbdJointDebugFrame++;
#endif
}

//=============================================================================
// Soft body VBD: per-particle 3x3 block coordinate descent
//=============================================================================

void AvbdSolver::solveSoftParticle(
    PxU32 spi,
    AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
    AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
    AvbdSoftBody *softBodies, PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, PxU32 numSoftContacts,
    PxReal dt, PxReal invDt2)
{
  PX_UNUSED(numSoftParticles);
  PX_UNUSED(numRigidBodies);
  PX_UNUSED(dt);

  AvbdSoftParticle &sp = softParticles[spi];
  if (sp.invMass <= 0.0f) return;

  PxReal mOverDt2 = sp.mass * invDt2;

  // Inertial force and Hessian
  PxVec3 f3 = (sp.predictedPosition - sp.position) * mOverDt2;
  PxMat33 H3 = PxMat33::createDiagonal(PxVec3(mOverDt2));

  // Accumulate VBD element contributions using per-particle adjacency
  for (PxU32 sbi = 0; sbi < numSoftBodies; ++sbi)
  {
    const AvbdSoftBody &sb = softBodies[sbi];
    PxU32 localIdx = spi - sb.particleStart;
    if (localIdx >= sb.particleCount) continue;

    const AvbdParticleAdjacency &adj = sb.adjacency[localIdx];

    // StVK triangle contributions
    for (PxU32 ri = 0; ri < adj.triRefs.size(); ++ri)
    {
      const AvbdParticleElementRef &ref = adj.triRefs[ri];
      PxVec3 ft; PxMat33 Ht;
      avbdEvaluateStVKForceHessian(sb.triElements[ref.index], ref.vOrder,
                                    sb.mu, sb.lambda, softParticles, ft, Ht);
      f3 += ft; H3 += Ht;
    }

    // Neo-Hookean tet contributions
    for (PxU32 ri = 0; ri < adj.tetRefs.size(); ++ri)
    {
      const AvbdParticleElementRef &ref = adj.tetRefs[ri];
      PxVec3 ft; PxMat33 Ht;
      avbdEvaluateNeoHookeanForceHessian(sb.tetElements[ref.index], ref.vOrder,
                                          sb.mu, sb.lambda, softParticles, ft, Ht);
      f3 += ft; H3 += Ht;
    }

    // Bending contributions
    for (PxU32 ri = 0; ri < adj.bendRefs.size(); ++ri)
    {
      const AvbdParticleElementRef &ref = adj.bendRefs[ri];
      PxVec3 fb; PxMat33 Hb;
      avbdEvaluateBendingForceHessian(sb.bendElements[ref.index], ref.vOrder,
                                       sb.bendingStiffness, softParticles, fb, Hb);
      f3 += fb; H3 += Hb;
    }

    // Attachment (AVBD penalty)
    for (PxU32 ai = 0; ai < adj.attachmentIndices.size(); ++ai)
    {
      PxVec3 fa; PxMat33 Ha;
      avbdEvaluateAttachmentForceHessian_particle(
          sb.attachments[adj.attachmentIndices[ai]],
          softParticles, rigidBodies, fa, Ha);
      f3 += fa; H3 += Ha;
    }

    // Kinematic pin (AVBD penalty)
    for (PxU32 pi = 0; pi < adj.pinIndices.size(); ++pi)
    {
      PxVec3 fp; PxMat33 Hp;
      avbdEvaluatePinForceHessian(sb.pins[adj.pinIndices[pi]],
                                   softParticles, fp, Hp);
      f3 += fp; H3 += Hp;
    }
  }

  // Soft contacts (ground / rigid, AVBD penalty)
  for (PxU32 sci = 0; sci < numSoftContacts; ++sci)
  {
    if (softContacts[sci].particleIdx != spi) continue;
    PxVec3 fc; PxMat33 Hc;
    avbdEvaluateContactForceHessian(softContacts[sci], softParticles, fc, Hc);
    f3 += fc; H3 += Hc;
  }

  // Solve 3x3: displacement = inv(H) * f
  PxVec3 displacement = H3.getInverse() * f3;
  PxReal dispMag = displacement.magnitude();
  if (!PxIsFinite(dispMag))
    displacement = PxVec3(0.0f);

  sp.position += displacement;
}

//=============================================================================
// Soft body AVBD dual update (penalty growth only)
//=============================================================================

void AvbdSolver::updateSoftDual(
    AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
    AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
    AvbdSoftBody *softBodies, PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, PxU32 numSoftContacts,
    PxReal beta)
{
  PX_UNUSED(numSoftParticles);
  PX_UNUSED(numRigidBodies);

  for (PxU32 sbi = 0; sbi < numSoftBodies; ++sbi)
  {
    AvbdSoftBody &sb = softBodies[sbi];
    for (PxU32 ai = 0; ai < sb.attachments.size(); ++ai)
      avbdUpdateAttachmentDual(sb.attachments[ai], softParticles, rigidBodies, beta);
    for (PxU32 pi = 0; pi < sb.pins.size(); ++pi)
      avbdUpdatePinDual(sb.pins[pi], softParticles, beta);
  }
  for (PxU32 sci = 0; sci < numSoftContacts; ++sci)
    avbdUpdateSoftContactDual(softContacts[sci], softParticles, beta);
}


} // namespace Dy
} // namespace physx
