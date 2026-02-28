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
#include "DyAvbdJointSolver.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// Enable detailed joint solver diagnostics (first N frames)
#define AVBD_JOINT_DEBUG 1
#define AVBD_JOINT_DEBUG_FRAMES 5

// External frame counter from DyAvbdDynamics.cpp (used by motor drives)
extern physx::PxU64 getAvbdMotorFrameCounter();

static physx::PxU32 s_avbdJointDebugFrame = 0;

namespace physx {
namespace Dy {

namespace {
struct KahanSum {
  physx::PxReal sum{0.0f};
  physx::PxReal c{0.0f};

  void add(physx::PxReal value) {
    physx::PxReal y = value - c;
    physx::PxReal t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};
} // namespace

//=============================================================================
// Main Solver Entry Point
//=============================================================================

void AvbdSolver::solve(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                       const AvbdBodyConstraintMap *contactMap,
                       AvbdColorBatch *colorBatches, physx::PxU32 numColors) {
  PX_PROFILE_ZONE("AVBD.solve", 0);

  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();
  mStats.numBodies = numBodies;
  mStats.numContacts = numContacts;

  physx::PxReal invDt = 1.0f / dt;

  // Stage 1: Prediction
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  // Stage 2: Graph Coloring (skip if pre-computed coloring is provided)
  if (mConfig.enableParallelization && numColors == 0) {
    PX_PROFILE_ZONE("AVBD.graphColoring", 0);
    computeGraphColoring(bodies, numBodies, contacts, numContacts);
  }

  // Adaptive position warmstarting (ref: AVBD3D solver.cpp L76-98)
  //
  // The solver's inertia term RHS = M/h^2*(x - x_pred) drives the body
  // toward its prediction. The warmstart position controls the gravity drive:
  //
  //   x_warmstart = x_n + v*dt + accelWeight * g*dt^2
  //   x_pred      = x_n + v*dt + g*dt^2
  //   RHS = M/h^2 * (accelWeight - 1) * g*dt^2
  //
  //   accelWeight=0 (supported): RHS = -M*g  (full gravity drive)
  //   accelWeight=1 (freefall):  RHS = 0     (no gravity drive)
  //
  // accelWeight = clamp(dot(acceleration, gravDir) / |g|, 0, 1)
  //   acceleration = (v_current - v_previous) / dt
  //
  // Now that computePrediction does NOT modify linearVelocity:
  //   linearVelocity     = v_{N-1, postsolve}  (clean post-solve from last
  //   frame) prevLinearVelocity  = v_{N-2, postsolve}  (saved at end of frame
  //   N-2)
  {
    PX_PROFILE_ZONE("AVBD.initPositions", 0);

    const physx::PxReal gravMag = gravity.magnitude();
    const physx::PxVec3 gravDir =
        (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      // Save current position for velocity computation at end of solve.
      // In the reference this is "initialPosition".
      bodies[i].prevPosition = bodies[i].position;
      bodies[i].prevRotation = bodies[i].rotation;

      if (bodies[i].invMass > 0.0f) {
        // Compute acceleration from velocity change across frames
        // accel = (v_{N-1} - v_{N-2}) / dt
        physx::PxVec3 accel =
            (bodies[i].linearVelocity - bodies[i].prevLinearVelocity) * invDt;

        // Project acceleration onto gravity direction
        physx::PxReal accelWeight = 0.0f;
        if (gravMag > 1e-6f) {
          accelWeight =
              physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f);
        }

        // Warmstart position: x = x_n + v*dt + accelWeight * g*dt^2
        bodies[i].position = bodies[i].prevPosition +
                             bodies[i].linearVelocity * dt +
                             gravity * (accelWeight * dt * dt);
        bodies[i].rotation = bodies[i].inertialRotation;
      }
    }
  }

  // =========================================================================
  // Enforce penalty floor: penalty must be proportional to M/h^2
  //
  // In AVBD3D, PENALTY_MIN=1000 with mass~1.25 gives ratio~22%.
  // For PhysX scenes with heavier bodies (mass=640 => M/h^2=2.3e6),
  // PENALTY_MIN=1000 gives ratio=0.04%, making constraints invisible.
  // We enforce penalty >= 0.25*M/h^2 so that constraints can resist
  // inertia from the very first iteration.
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.penaltyFloor", 0);
    const physx::PxReal invDt2 = 1.0f / (dt * dt);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;

      // Compute effective mass using harmonic mean for two-body contacts
      // (ref: AVBD3D solver step(): effectiveMass = mA*mB/(mA+mB))
      // For body-vs-static, effectiveMass = mass of dynamic body.
      physx::PxReal massA = 0.0f, massB = 0.0f;
      if (bA < numBodies && bodies[bA].invMass > 0.0f) {
        massA = 1.0f / bodies[bA].invMass;
      }
      if (bB < numBodies && bodies[bB].invMass > 0.0f) {
        massB = 1.0f / bodies[bB].invMass;
      }

      physx::PxReal effectiveMass;
      physx::PxReal penScale;
      if (massA > 0.0f && massB > 0.0f) {
        // Two dynamic bodies: use max mass with SOFT scale (0.05).
        // max(mA,mB) ensures the penalty is stiff enough to decelerate
        // the heavier body, preventing tunneling at extreme mass ratios.
        // AVBD's implicit solve keeps this stable regardless of ratio.
        effectiveMass = physx::PxMax(massA, massB);
        penScale = 0.05f;
      } else {
        // Body-vs-static: full stiffness (0.25) for stable stacking.
        effectiveMass = physx::PxMax(massA, massB);
        penScale = 0.25f;
      }

      const physx::PxReal effectiveMassH2 = effectiveMass * invDt2;
      const physx::PxReal penaltyFloor = penScale * effectiveMassH2;
      if (contacts[c].header.penalty < penaltyFloor) {
        contacts[c].header.penalty = penaltyFloor;
      }
      // Also floor tangent penalties (ref: standalone floors all 3 rows)
      if (contacts[c].tangentPenalty0 < penaltyFloor) {
        contacts[c].tangentPenalty0 = penaltyFloor;
      }
      if (contacts[c].tangentPenalty1 < penaltyFloor) {
        contacts[c].tangentPenalty1 = penaltyFloor;
      }
    }
  }

  // =========================================================================
  // Compute C0 for alpha blending (ref: AVBD3D manifold.cpp computeC0)
  //
  // C0 = initial constraint violation at PRE-WARMSTART positions (the old
  // positions from end of previous step, saved as prevPosition/prevRotation).
  //
  // CRITICAL: C0 must be computed at old positions, NOT warmstart positions!
  // If C0 captures the gravity-induced predicted penetration, then
  // alpha blending (violation - alpha*C0) cancels 95% of the constraint
  // signal, causing bodies to fall through each other.
  //
  // At old positions, established contacts have C0 ~= 0, so alpha blending
  // is nearly a no-op (violation ~= violation - 0). For newly penetrating
  // contacts, C0 < 0 and the blending gradually corrects over frames.
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.computeC0", 0);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      // Use prevPosition/prevRotation = positions from START of step
      // (saved before warmstart body positions were applied)
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
      contacts[c].C0 = (wA - wB).dot(contacts[c].contactNormal) +
                       contacts[c].penetrationDepth;
    }
  }

  // Sort constraints for deterministic iteration order
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
  // Main solver loop (ref: AVBD3D solver.cpp L103-164)
  //
  // 6x6 path: Each iteration performs primal update + dual update.
  //   The reference uses iterations=10 with a single combined loop.
  //   lambda and penalty adapt between primal steps for progressive stiffening.
  //
  // Fast path: Keeps original outer(1) x inner(4) structure with dual update
  //   only after all inner iterations (backward compatible).
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.solveIterations", 0);

    // Both 6x6 and 3x3 paths use AL dual update => primal+dual each iteration
    for (physx::PxU32 iter = 0; iter < mConfig.innerIterations; ++iter) {
      {
        PX_PROFILE_ZONE("AVBD.blockDescent", 0);
        blockDescentIteration(bodies, numBodies, contacts, numContacts, dt,
                              contactMap, colorBatches, numColors);
        mStats.totalIterations++;
      }
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt);
      }
    }
  }

  // Stage 5: Update velocities from position/rotation change
  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;

        // Linear velocity: v = (x - x_old) / dt
        bodies[i].linearVelocity =
            (bodies[i].position - bodies[i].prevPosition) * invDt;

        // Angular velocity from quaternion difference
        physx::PxQuat deltaQ =
            bodies[i].rotation * bodies[i].prevRotation.getConjugate();
        if (deltaQ.w < 0.0f) {
          deltaQ = -deltaQ;
        }
        bodies[i].angularVelocity =
            physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);

        // Apply angular damping
        bodies[i].angularVelocity *= mConfig.angularDamping;
      }
    }
  }
}

//=============================================================================
// Graph Coloring
//=============================================================================

void AvbdSolver::computeGraphColoring(AvbdSolverBody *bodies,
                                      physx::PxU32 numBodies,
                                      AvbdContactConstraint *contacts,
                                      physx::PxU32 numContacts) {
  PX_ASSERT(mAllocator != nullptr);

  // Build adjacency from contacts
  // Two bodies are adjacent if they share a constraint

  // Reset colors
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].colorGroup = 0xFFFFFFFF; // Uncolored
  }

  // Simple greedy coloring
  physx::PxU32 numColors = 0;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].isStatic()) {
      bodies[i].colorGroup = 0; // Static bodies go to color 0
      continue;
    }

    // Find colors used by neighbors
    physx::PxU32 usedColors = 0;

    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
      physx::PxU32 bodyB = contacts[c].header.bodyIndexB;

      if (bodyA == i && bodyB < numBodies && bodies[bodyB].colorGroup < 32) {
        usedColors |= (1u << bodies[bodyB].colorGroup);
      }
      if (bodyB == i && bodyA < numBodies && bodies[bodyA].colorGroup < 32) {
        usedColors |= (1u << bodies[bodyA].colorGroup);
      }
    }

    // Find first available color (skip color 0 for static bodies)
    physx::PxU32 color = 1;
    while ((usedColors & (1u << color)) != 0 && color < 32) {
      color++;
    }

    bodies[i].colorGroup = color;
    if (color + 1 > numColors) {
      numColors = color + 1;
    }
  }

  mStats.numColorGroups = numColors;
}

//=============================================================================
// Body-Based Graph Coloring for Block Coordinate Descent
//=============================================================================

void AvbdSolver::computeBodyColoring(AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     AvbdContactConstraint *contacts,
                                     physx::PxU32 numContacts) {
  PX_ASSERT(mAllocator != nullptr);

  // Initialize body coloring if not already done
  if (!mBodyColoring.isInitialized()) {
    mBodyColoring.initialize(numBodies, *mAllocator);
  }

  // Perform body-based coloring
  physx::PxU32 numColors =
      mBodyColoring.colorBodies(contacts, numContacts, bodies, numBodies);

  mStats.numColorGroups = numColors;
}

//=============================================================================
// Augmented Lagrangian Multiplier Update
//
// 6x6 path (ref: AVBD3D solver.cpp L142-164):
//   lambda = clamp(penalty*C + lambda, fmin, fmax)
//   if lambda within bounds: penalty += beta * |C|
//   penalty = min(penalty, PENALTY_MAX)
//
// Fast path: XPBD formula (unchanged)
//=============================================================================

void AvbdSolver::updateLagrangianMultipliers(AvbdSolverBody *bodies,
                                             physx::PxU32 numBodies,
                                             AvbdContactConstraint *contacts,
                                             physx::PxU32 numContacts,
                                             physx::PxReal dt) {
  PX_UNUSED(dt);
  physx::PxReal totalError = 0.0f;
  KahanSum totalErrorKahan;
  const bool useKahan =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eUSE_KAHAN_SUMMATION);
  physx::PxU32 numActive = 0;

  // AVBD reference parameters for penalty growth
  const physx::PxReal beta = mConfig.avbdBeta;
  const physx::PxReal penaltyMax = mConfig.avbdPenaltyMax;

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;

    // Compute current violation
    physx::PxReal violation = 0.0f;
    AvbdSolverBody *bodyA = nullptr;
    AvbdSolverBody *bodyB = nullptr;

    if (bodyAIdx < numBodies && bodyBIdx < numBodies) {
      bodyA = &bodies[bodyAIdx];
      bodyB = &bodies[bodyBIdx];
      violation = computeContactViolation(contacts[c], *bodyA, *bodyB);
    } else if (bodyAIdx < numBodies) {
      bodyA = &bodies[bodyAIdx];
      physx::PxVec3 worldPointA =
          bodyA->position + bodyA->rotation.rotate(contacts[c].contactPointA);
      physx::PxVec3 worldPointB = contacts[c].contactPointB;
      violation = (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
    } else if (bodyBIdx < numBodies) {
      bodyB = &bodies[bodyBIdx];
      physx::PxVec3 worldPointA = contacts[c].contactPointA;
      physx::PxVec3 worldPointB =
          bodyB->position + bodyB->rotation.rotate(contacts[c].contactPointB);
      violation = (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
    }

    // Alpha blending (ref: AVBD3D manifold.cpp computeConstraint)
    violation -= mConfig.avbdAlpha * contacts[c].C0;

    physx::PxReal oldLambda = contacts[c].header.lambda;
    physx::PxReal newLambda = 0.0f;

    // =====================================================================
    // AL dual update with adaptive penalty growth
    // (ref: AVBD3D solver.cpp L147-162)
    //
    // lambda = clamp(penalty*C + lambda, fmin, fmax)
    //   For normal contacts: fmin=-inf, fmax=0
    //
    // Penalty growth (Eq. 16): if lambda is strictly within force bounds
    // (not clamped), the constraint is active and we increase penalty to
    // stiffen it. penalty += beta * |C|
    // =====================================================================
    {
      physx::PxReal pen = contacts[c].header.penalty;
      physx::PxReal rawLambda = pen * violation + oldLambda;

      // Clamp: normal contacts are unilateral => fmin=-inf, fmax=0
      newLambda = physx::PxMin(0.0f, rawLambda);

      // Penalty growth: only when lambda is within bounds (active constraint)
      if (newLambda < 0.0f) {
        physx::PxReal newPenalty = pen + beta * physx::PxAbs(violation);
        contacts[c].header.penalty = physx::PxMin(newPenalty, penaltyMax);
      }
    }

    contacts[c].header.lambda = newLambda;

    // =================================================================
    // Tangent dual update (3-row AL friction, ref: standalone L579-590)
    // Update tangentLambda0/1 with Coulomb-clamped bounds, and grow
    // tangent penalties when lambda is strictly within bounds.
    // =================================================================
    if (contacts[c].friction > 0.0f) {
      // Compute world-space contact points and prev-frame positions
      physx::PxVec3 worldPosA, worldPosB, prevWorldPosA, prevWorldPosB;
      if (bodyAIdx < numBodies) {
        worldPosA = bodies[bodyAIdx].position +
                    bodies[bodyAIdx].rotation.rotate(contacts[c].contactPointA);
        prevWorldPosA =
            bodies[bodyAIdx].prevPosition +
            bodies[bodyAIdx].prevRotation.rotate(contacts[c].contactPointA);
      } else {
        worldPosA = contacts[c].contactPointA;
        prevWorldPosA = contacts[c].contactPointA;
      }
      if (bodyBIdx < numBodies) {
        worldPosB = bodies[bodyBIdx].position +
                    bodies[bodyBIdx].rotation.rotate(contacts[c].contactPointB);
        prevWorldPosB =
            bodies[bodyBIdx].prevPosition +
            bodies[bodyBIdx].prevRotation.rotate(contacts[c].contactPointB);
      } else {
        worldPosB = contacts[c].contactPointB;
        prevWorldPosB = contacts[c].contactPointB;
      }

      physx::PxVec3 relDisp =
          (worldPosA - prevWorldPosA) - (worldPosB - prevWorldPosB);

      // Friction bound from normal lambda (Coulomb friction cone)
      physx::PxReal frictionBound =
          physx::PxAbs(newLambda) * contacts[c].friction;

      // Tangent 0
      {
        physx::PxReal tC = relDisp.dot(contacts[c].tangent0);
        physx::PxReal tPen = contacts[c].tangentPenalty0;
        physx::PxReal tLam = contacts[c].tangentLambda0;
        physx::PxReal rawTLam = tPen * tC + tLam;
        physx::PxReal newTLam =
            physx::PxClamp(rawTLam, -frictionBound, frictionBound);
        contacts[c].tangentLambda0 = newTLam;
        // Penalty growth if lambda is strictly within bounds (not clamped)
        if (newTLam > -frictionBound && newTLam < frictionBound) {
          contacts[c].tangentPenalty0 =
              physx::PxMin(tPen + beta * physx::PxAbs(tC), penaltyMax);
        }
      }

      // Tangent 1
      {
        physx::PxReal tC = relDisp.dot(contacts[c].tangent1);
        physx::PxReal tPen = contacts[c].tangentPenalty1;
        physx::PxReal tLam = contacts[c].tangentLambda1;
        physx::PxReal rawTLam = tPen * tC + tLam;
        physx::PxReal newTLam =
            physx::PxClamp(rawTLam, -frictionBound, frictionBound);
        contacts[c].tangentLambda1 = newTLam;
        if (newTLam > -frictionBound && newTLam < frictionBound) {
          contacts[c].tangentPenalty1 =
              physx::PxMin(tPen + beta * physx::PxAbs(tC), penaltyMax);
        }
      }
    }

    // Track convergence
    if (violation < 0.0f) {
      physx::PxReal err = violation * violation;
      if (useKahan) {
        totalErrorKahan.add(err);
      } else {
        totalError += err;
      }
      numActive++;
    }
  }

  if (useKahan) {
    totalError = totalErrorKahan.sum;
  }

  mStats.constraintError =
      (numActive > 0) ? sqrtf(totalError / (physx::PxReal)numActive) : 0.0f;
  mStats.activeConstraints = numActive;
}

//=============================================================================
// Local 6x6 System Solver -- AVBD Reference Algorithm
//
// Implements the AVBD primal update per body (ref: AVBD3D solver.cpp L107-138):
//
//   lhs = M/h^2
//   rhs = lhs * vec6{x - x_inertial, deltaW_inertial}
//   For each constraint on body:
//     f = clamp(penalty * C + lambda, fmin, fmax)
//     rhs += J * f               (Eq. 13)
//     lhs += outer(J, J*penalty)  (Eq. 17)
//   delta = solve(lhs, rhs)
//   x -= delta
//
// Key difference from old code: uses adaptive penalty (per-constraint,
// grows via beta*|C| in dual update) instead of fixed effectiveRho hack.
//=============================================================================

void AvbdSolver::solveLocalSystem(AvbdSolverBody &body, AvbdSolverBody *bodies,
                                  physx::PxU32 numBodies,
                                  AvbdContactConstraint *contacts,
                                  physx::PxU32 numContacts, physx::PxReal dt,
                                  physx::PxReal invDt2,
                                  const AvbdBodyConstraintMap *contactMap) {

  // Skip static bodies
  if (body.invMass <= 0.0f) {
    return;
  }
  PX_UNUSED(dt);

  const physx::PxU32 bodyIndex = body.nodeIndex;

  // =========================================================================
  // Step 1: Initialize LHS with mass matrix M/h^2
  // =========================================================================

  AvbdBlock6x6 A;
  A.initializeDiagonal(body.invMass, body.invInertiaWorld, invDt2);

  // =========================================================================
  // Step 2: Initialize RHS with inertia term
  //   rhs = (M/h^2) * vec6{x - x_inertial, deltaW_inertial}
  // =========================================================================

  physx::PxReal mass = (body.invMass > 1e-8f) ? (1.0f / body.invMass) : 0.0f;
  physx::PxReal massInvDt2 = mass * invDt2;

  physx::PxVec3 gLinear = (body.position - body.inertialPosition) * massInvDt2;

  // Angular inertia RHS: (I/h^2) * deltaW_inertial
  physx::PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
  if (deltaQ.w < 0.0f) {
    deltaQ = -deltaQ;
  }
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  physx::PxMat33 inertiaTensor = body.invInertiaWorld.getInverse();
  physx::PxVec3 gAngular = (inertiaTensor * rotError) * invDt2;

  // =========================================================================
  // Step 3: Iterate over all constraints affecting this body
  //         Accumulate penalty*J^T*J into LHS and J*f into RHS
  //
  //   Uses contact.header.penalty (adaptive, per-constraint) instead of a
  //   fixed rho. The penalty grows via beta*|C| in each dual update,
  //   automatically scaling up where constraints are violated.
  // =========================================================================

  physx::PxU32 numTouching = 0;

  // Use contactMap for O(K) lookup if available, else O(N) scan
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  if (contactMap && contactMap->numBodies > 0) {
    contactMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
  }
  const physx::PxU32 loopCount = mapIndices ? mapCount : numContacts;

  for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
    const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
    const physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;

    if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
      continue;
    }

    const bool isBodyA = (bodyAIdx == bodyIndex);

    // Get other body
    AvbdSolverBody *otherBody = nullptr;
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }

    // Compute world-space contact points
    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r;

    if (isBodyA) {
      r = body.rotation.rotate(contacts[c].contactPointA);
      worldPosA = body.position + r;
      worldPosB =
          otherBody ? otherBody->position +
                          otherBody->rotation.rotate(contacts[c].contactPointB)
                    : contacts[c].contactPointB;
    } else {
      r = body.rotation.rotate(contacts[c].contactPointB);
      worldPosA =
          otherBody ? otherBody->position +
                          otherBody->rotation.rotate(contacts[c].contactPointA)
                    : contacts[c].contactPointA;
      worldPosB = body.position + r;
    }

    // Constraint violation: C(x)
    const physx::PxVec3 &normal = contacts[c].contactNormal;
    physx::PxReal violation =
        (worldPosA - worldPosB).dot(normal) + contacts[c].penetrationDepth;

    // Alpha blending (ref: AVBD3D manifold.cpp computeConstraint)
    // C_blended = C_current - alpha * C0
    // This softens pre-existing penetration so only (1-alpha)*C0 is corrected
    // per step, preventing violent pop-out for stacked bodies.
    violation -= mConfig.avbdAlpha * contacts[c].C0;

    // Per-constraint adaptive penalty (ref AVBD3D: force->penalty[i])
    physx::PxReal pen = contacts[c].header.penalty;
    // Per-body primal boost: 2% of M/h^2, a gentle safety net that
    // prevents heavy bodies from slowly drifting through contacts
    // during oscillations where contact count drops temporarily.
    // The shared penalty floor (geometric mean) handles the primary
    // mass-ratio correction; this is just a small extra margin.
    pen = physx::PxMax(pen, 0.005f * massInvDt2);
    physx::PxReal lambda = contacts[c].header.lambda;

    // Jacobian: bodyA => +n, bodyB => -n
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxVec3 gradPos = normal * sign;
    physx::PxVec3 gradRot = rCrossN * sign;

    // -----------------------------------------------------------------------
    // LHS: lhs += outer(J, J * penalty)   (Eq. 17, unconditional)
    // -----------------------------------------------------------------------
    A.addConstraintContribution(gradPos, gradRot, pen);
    numTouching++;

    // -----------------------------------------------------------------------
    // Force: f = clamp(penalty * C + lambda, fmin, fmax)
    //   Normal contacts: fmin=-inf, fmax=0 (can only push, never pull)
    // -----------------------------------------------------------------------
    physx::PxReal f = physx::PxMin(0.0f, pen * violation + lambda);

    // -----------------------------------------------------------------------
    // RHS: rhs += J * f   (Eq. 13)
    //   Only when f != 0 (active constraint)
    // -----------------------------------------------------------------------
    if (f < 0.0f) {
      gLinear += gradPos * f;
      gAngular += gradRot * f;
    }

    // =====================================================================
    // Friction: 3-row AL model (ref: AVBD3D manifold.cpp)
    //   Two independent tangent constraint rows, each with its own
    //   lambda, penalty, and Coulomb-clamped force bounds.
    //   frictionBound = |lambda_normal| * mu
    //   Tangent constraint: C_t = relative tangent displacement
    // =====================================================================
    if (contacts[c].friction > 0.0f) {
      physx::PxReal frictionBound =
          physx::PxAbs(contacts[c].header.lambda) * contacts[c].friction;

      // Compute relative tangent displacement for constraint value
      physx::PxVec3 prevWorldPosA, prevWorldPosB;
      if (isBodyA) {
        prevWorldPosA = body.prevPosition +
                        body.prevRotation.rotate(contacts[c].contactPointA);
        prevWorldPosB =
            otherBody
                ? otherBody->prevPosition +
                      otherBody->prevRotation.rotate(contacts[c].contactPointB)
                : contacts[c].contactPointB;
      } else {
        prevWorldPosA =
            otherBody
                ? otherBody->prevPosition +
                      otherBody->prevRotation.rotate(contacts[c].contactPointA)
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
        physx::PxReal tC = relDisp.dot(t); // tangent constraint value

        // LHS += outer(Jt, Jt * penalty) -- UNCONDITIONAL
        A.addConstraintContribution(tGradPos, tGradRot, tPen);

        // Force: f_t = clamp(pen*C_t + lambda_t, -frictionBound,
        // +frictionBound)
        physx::PxReal ft = tPen * tC + tLambda;
        ft = physx::PxClamp(ft, -frictionBound, frictionBound);

        // RHS += Jt * ft
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

        physx::PxReal ft = tPen * tC + tLambda;
        ft = physx::PxClamp(ft, -frictionBound, frictionBound);

        gLinear += tGradPos * ft;
        gAngular += tGradRot * ft;
      }
    }
  }

  // No contacts: snap to inertial target
  if (numTouching == 0) {
    body.position = body.inertialPosition;
    body.rotation = body.inertialRotation;
    return;
  }

  // =========================================================================
  // Step 4: Solve A * delta = rhs via LDLT
  // =========================================================================

  AvbdLDLT ldlt;
  AvbdVec6 rhs(gLinear, gAngular);

  physx::PxVec3 deltaPos;
  physx::PxVec3 deltaTheta;

  if (ldlt.decomposeWithRegularization(A)) {
    AvbdVec6 delta = ldlt.solve(rhs);
    deltaPos = delta.linear;
    deltaTheta = delta.angular;
  } else {
    deltaPos = physx::PxVec3(0.0f);
    deltaTheta = physx::PxVec3(0.0f);
  }

  // =========================================================================
  // Step 5: Apply update  x -= delta
  //   (ref: solver.cpp L137-138)
  // =========================================================================

  body.position -= deltaPos;

  if (deltaTheta.magnitudeSquared() > 1e-12f) {
    physx::PxQuat dq(deltaTheta.x, deltaTheta.y, deltaTheta.z, 0.0f);
    body.rotation = (body.rotation - dq * body.rotation * 0.5f).getNormalized();
  }
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

void AvbdSolver::solveLocalSystemWithJoints(
    AvbdSolverBody &body, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *sphericalMap,
    const AvbdBodyConstraintMap *fixedMap,
    const AvbdBodyConstraintMap *revoluteMap,
    const AvbdBodyConstraintMap *prismaticMap,
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

  // =========================================================================
  // Step 3b: Accumulate SPHERICAL JOINT contributions (3 rows per joint)
  //
  //   C_k = (anchorA - anchorB) . e_k    for k = x, y, z
  //   Body A: gradPos = +e_k, gradRot = +(r_A x e_k)
  //   Body B: gradPos = -e_k, gradRot = -(r_B x e_k)
  //   Force: f = pen * C + lambda   (equality: no clamping)
  // =========================================================================
  if (sphericalJoints && numSpherical > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (sphericalMap && sphericalMap->numBodies > 0)
      sphericalMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numSpherical;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numSpherical)
        continue;
      const AvbdSphericalJointConstraint &jnt = sphericalJoints[j];
      const physx::PxU32 bodyAIdx = jnt.header.bodyIndexA;
      const physx::PxU32 bodyBIdx = jnt.header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;

      const bool isBodyA = (bodyAIdx == bodyIndex);
      const bool otherIsStatic =
          isBodyA ? (bodyBIdx == 0xFFFFFFFF || bodyBIdx >= numBodies)
                  : (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);

      // Compute world anchors
      physx::PxVec3 worldAnchorA, worldAnchorB;
      physx::PxVec3 r; // lever arm for THIS body
      if (isBodyA) {
        r = body.rotation.rotate(jnt.anchorA);
        worldAnchorA = body.position + r;
        worldAnchorB = otherIsStatic
                           ? jnt.anchorB
                           : bodies[bodyBIdx].position +
                                 bodies[bodyBIdx].rotation.rotate(jnt.anchorB);
      } else {
        r = body.rotation.rotate(jnt.anchorB);
        worldAnchorB = body.position + r;
        worldAnchorA = otherIsStatic
                           ? jnt.anchorA
                           : bodies[bodyAIdx].position +
                                 bodies[bodyAIdx].rotation.rotate(jnt.anchorA);
      }

      physx::PxVec3 posError = worldAnchorA - worldAnchorB;
      physx::PxReal mA =
          (bodyAIdx < numBodies && bodies[bodyAIdx].invMass > 1e-8f)
              ? (1.0f / bodies[bodyAIdx].invMass)
              : 0.0f;
      physx::PxReal mB =
          (bodyBIdx < numBodies && bodies[bodyBIdx].invMass > 1e-8f)
              ? (1.0f / bodies[bodyBIdx].invMass)
              : 0.0f;
      physx::PxReal mEff = physx::PxMax(mA, mB);

      // Auto-boost penalty using symmetric effective mass to satisfy Newton's
      // 3rd Law
      physx::PxReal pen = physx::PxMax(jnt.header.rho, mEff * invDt2);
      physx::PxReal signJ = isBodyA ? 1.0f : -1.0f;

#if AVBD_JOINT_DEBUG
      {
        static physx::PxU32 s_sphDbg = 0;
        if (s_sphDbg < 2) {
          printf("    sph[%u] body%u(%c) r=(%.3f,%.3f,%.3f) C=(%.6f,%.6f,%.6f) "
                 "rho=%.0f\n",
                 j, bodyIndex, isBodyA ? 'A' : 'B', r.x, r.y, r.z, posError.x,
                 posError.y, posError.z, pen);
          printf(
              "      anchorA_w=(%.4f,%.4f,%.4f) anchorB_w=(%.4f,%.4f,%.4f)\n",
              worldAnchorA.x, worldAnchorA.y, worldAnchorA.z, worldAnchorB.x,
              worldAnchorB.y, worldAnchorB.z);
          if (bodyIndex == 4)
            s_sphDbg++; // increment after last body
        }
      }
#endif

      for (int axis = 0; axis < 3; ++axis) {
        physx::PxReal C = posError[axis];

        physx::PxVec3 n(0.0f);
        (&n.x)[axis] = 1.0f;

        physx::PxVec3 rCrossN = r.cross(n);
        physx::PxVec3 gradPos = n * signJ;
        physx::PxVec3 gradRot = rCrossN * signJ;

        // LHS: H += pen * J^T * J (unconditional for equality constraints)
        A.addConstraintContribution(gradPos, gradRot, pen);

        // Force: f = pen * C + lambda (equality, no clamping)
        physx::PxReal f = pen * C + jnt.lambda[axis];

        // RHS: g += J * f
        gLinear += gradPos * f;
        gAngular += gradRot * f;
      }

      // Process cone limit (inequality constraint)
      if (jnt.hasConeLimit && jnt.coneAngleLimit > 0.0f) {
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

        physx::PxReal coneViolation = jnt.computeConeViolation(rotA, rotB);
        // coneLambda is <= 0 (negative means active force pushing back)
        physx::PxReal forceMag = pen * coneViolation - jnt.coneLambda;

        if (forceMag > 0.0f) {
          physx::PxVec3 worldAxisA = rotA.rotate(jnt.coneAxisA);
          physx::PxVec3 worldAxisB = rotB.rotate(jnt.coneAxisA);
          physx::PxVec3 corrAxis = worldAxisA.cross(worldAxisB);

          physx::PxReal corrAxisMag = corrAxis.magnitude();
          if (corrAxisMag > 1e-6f) {
            corrAxis /= corrAxisMag;

            // grad_theta_A C = -corrAxis, grad_theta_B C = +corrAxis
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = -corrAxis * signJ;

            A.addConstraintContribution(gradPos, gradRot, pen);
            gAngular += gradRot * forceMag;
          }
        }
      }

      numTouching++;
    }
  }

  // =========================================================================
  // Step 3c: Accumulate FIXED JOINT contributions (6 rows: 3 pos + 3 rot)
  //
  //   Position (same as spherical):
  //     C_k = (anchorA - anchorB) . e_k
  //
  //   Rotation:
  //     C_k = rotError . e_k
  //     Body A: gradPos = 0, gradRot = +e_k
  //     Body B: gradPos = 0, gradRot = -e_k
  // =========================================================================
  if (fixedJoints && numFixed > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (fixedMap && fixedMap->numBodies > 0)
      fixedMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numFixed;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numFixed)
        continue;
      const AvbdFixedJointConstraint &jnt = fixedJoints[j];
      const physx::PxU32 bodyAIdx = jnt.header.bodyIndexA;
      const physx::PxU32 bodyBIdx = jnt.header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;
      if (jnt.isBroken)
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

      // --- Position rows (3 DOF) ---
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

        physx::PxVec3 posError = worldAnchorA - worldAnchorB;

        for (int axis = 0; axis < 3; ++axis) {
          physx::PxReal C = posError[axis];
          physx::PxVec3 n(0.0f);
          (&n.x)[axis] = 1.0f;

          physx::PxVec3 rCrossN = r.cross(n);
          physx::PxVec3 gradPos = n * signJ;
          physx::PxVec3 gradRot = rCrossN * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);

          physx::PxReal f = pen * C + jnt.lambdaPosition[axis];
          gLinear += gradPos * f;
          gAngular += gradRot * f;
        }
      }

      // --- Rotation rows (3 DOF) ---
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

        physx::PxVec3 rotErr = jnt.computeRotationViolation(rotA, rotB);

        for (int axis = 0; axis < 3; ++axis) {
          physx::PxReal C = rotErr[axis];
          physx::PxVec3 n(0.0f);
          (&n.x)[axis] = 1.0f;

          // Rotation constraint Jacobian: pure angular
          physx::PxVec3 gradPos(0.0f);
          physx::PxVec3 gradRot = n * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);

          physx::PxReal f = pen * C + jnt.lambdaRotation[axis];
          // gradPos is zero, so only angular
          gAngular += gradRot * f;
        }
      }
      numTouching++;
    }
  }

  // =========================================================================
  // Step 3d: Accumulate REVOLUTE JOINT contributions
  //   Position (3 rows): same as spherical
  //   Axis Alignment (2 rows): enforces axisA == axisB
  //   Angle Limit (1 inequality row): if hasAngleLimit and violated
  // =========================================================================
  if (revoluteJoints && numRevolute > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (revoluteMap && revoluteMap->numBodies > 0)
      revoluteMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numRevolute;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numRevolute)
        continue;
      const AvbdRevoluteJointConstraint &jnt = revoluteJoints[j];
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

      physx::PxReal pen = physx::PxMax(jnt.header.rho, mEff * invDt2);
      physx::PxReal signJ = isBodyA ? 1.0f : -1.0f;

      // --- Position constraint (3 rows) ---
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

        physx::PxVec3 posError = worldAnchorA - worldAnchorB;

        for (int axis = 0; axis < 3; ++axis) {
          physx::PxReal C = posError[axis];
          physx::PxVec3 n(0.0f);
          (&n.x)[axis] = 1.0f;

          physx::PxVec3 rCrossN = r.cross(n);
          physx::PxVec3 gradPos = n * signJ;
          physx::PxVec3 gradRot = rCrossN * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);

          physx::PxReal f = pen * C + jnt.lambdaPosition[axis];
          gLinear += gradPos * f;
          gAngular += gradRot * f;

#if AVBD_JOINT_DEBUG
          {
            static physx::PxU32 s_revDebugCount = 0;
            if (s_revDebugCount < 30) {
              if (axis == 2) {
                printf("[Rev PosZ] frame=%u isA=%d body%u num=%u C=%.4f f=%.1f "
                       "gradRot=(%.1f,%.1f,%.1f)\n",
                       s_revDebugCount++, isBodyA, bodyIndex, ji, C, f,
                       gradRot.x, gradRot.y, gradRot.z);
              }
            }
          }
#endif
        }
      }

      // --- Axis alignment constraint (2 rows) ---
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

        physx::PxVec3 axisViolation = jnt.computeAxisViolation(rotA, rotB);

        // Build perpendicular basis from axisA
        physx::PxVec3 worldAxisA = rotA.rotate(jnt.axisA);
        physx::PxVec3 perp1, perp2;
        if (physx::PxAbs(worldAxisA.x) <
            AvbdConstants::AVBD_AXIS_SELECTION_THRESHOLD) {
          perp1 = worldAxisA.cross(physx::PxVec3(1, 0, 0)).getNormalized();
        } else {
          perp1 = worldAxisA.cross(physx::PxVec3(0, 1, 0)).getNormalized();
        }
        perp2 = worldAxisA.cross(perp1).getNormalized();

        physx::PxReal err1 = axisViolation.dot(perp1);
        physx::PxReal err2 = axisViolation.dot(perp2);

        for (int i = 0; i < 2; ++i) {
          physx::PxVec3 corrAxis = (i == 0) ? perp1 : perp2;
          physx::PxReal C = (i == 0) ? err1 : err2;

          // For A, d/dtheta ( (A x B).P ) = A x (B x P) = B(A.P) - P(A.B) =
          // -P(A.B) Since A ~ B, A.B ~ +1. So gradA is approximately -P.
          // Therefore, gradRot for A should be -corrAxis, and +corrAxis for B.
          physx::PxVec3 gradPos(0.0f);
          physx::PxVec3 gradRot = -corrAxis * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);

          physx::PxReal f = pen * C + jnt.lambdaAxisAlign.dot(corrAxis);
          gAngular += gradRot * f;
        }
      }

      // --- Angle limit inequality constraint (1 row) ---
      if (jnt.hasAngleLimit) {
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

        physx::PxReal angleViolation =
            jnt.computeAngleLimitViolation(rotA, rotB);
        physx::PxReal f = pen * angleViolation + jnt.lambdaAngleLimit;
        physx::PxReal forceMag = 0.0f;

        if (jnt.angleLimitLower < jnt.angleLimitUpper) {
          if (angleViolation > 0.0f || jnt.lambdaAngleLimit > 0.0f) {
            forceMag = physx::PxMax(0.0f, f);
          } else if (angleViolation < 0.0f || jnt.lambdaAngleLimit < 0.0f) {
            forceMag = physx::PxMin(0.0f, f);
          }
        } else {
          forceMag = f;
        }

        if (physx::PxAbs(forceMag) > 0.0f) {
          physx::PxVec3 worldAxisA = rotA.rotate(jnt.axisA);
          physx::PxVec3 gradPos(0.0f);
          physx::PxVec3 gradRot = worldAxisA * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);
          gAngular += gradRot * forceMag;
        }
      }

      numTouching++;
    }
  }

  // =========================================================================
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

      // --- Locked linear DOFs (position constraint, same as spherical) ---
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

        physx::PxVec3 posError = worldAnchorA - worldAnchorB;

        for (int axis = 0; axis < 3; ++axis) {
          physx::PxU32 motion = jnt.getLinearMotion(axis);
          if (motion == 2) // FREE
            continue;

          physx::PxReal C = posError[axis];
          physx::PxVec3 n(0.0f);
          (&n.x)[axis] = 1.0f;

          physx::PxVec3 rCrossN = r.cross(n);
          physx::PxVec3 gradPos = n * signJ;
          physx::PxVec3 gradRot = rCrossN * signJ;

          if (motion == 0) { // LOCKED
            A.addConstraintContribution(gradPos, gradRot, pen);

            physx::PxReal f = pen * C + jnt.lambdaLinear[axis];
            gLinear += gradPos * f;
            gAngular += gradRot * f;
          } else if (motion == 1) { // LIMITED
            physx::PxReal limitViolation =
                jnt.computeLinearLimitViolation(C, axis);
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
              A.addConstraintContribution(gradPos, gradRot, pen);
              gLinear += gradPos * forceMag;
              gAngular += gradRot * forceMag;
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
            physx::PxReal error = jnt.computeAngularError(rotA, rotB, axis);
            physx::PxReal limitViolation =
                jnt.computeAngularLimitViolation(error, axis);
            physx::PxReal f = pen * limitViolation + jnt.lambdaAngular[axis];
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
        if (isBodyA && bodyBIdx < numBodies)
          otherBody = &bodies[bodyBIdx];
        else if (!isBodyA && bodyAIdx < numBodies)
          otherBody = &bodies[bodyAIdx];

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

            physx::PxReal rho_drive = damping / dt2;
            physx::PxReal lam = (&jnt.lambdaDriveLinear.x)[a];
            physx::PxReal signAL = isBodyA ? -1.0f : 1.0f;
            physx::PxReal f = signAL * (rho_drive * C + lam);

            // Hessian: -_drive - (wAxis - wAxis) on linear block
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l)
                A.linearLinear(k, l) +=
                    rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];

            // RHS
            gLinear += physx::PxVec3(f * wAxis.x, f * wAxis.y, f * wAxis.z);
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
    }
  }

  // TODO: Add revolute, prismatic joint Jacobian accumulation
  // For now they remain as GS fallback in the body loop.
  PX_UNUSED(revoluteJoints);
  PX_UNUSED(numRevolute);
  PX_UNUSED(revoluteMap);
  PX_UNUSED(prismaticJoints);
  PX_UNUSED(numPrismatic);
  PX_UNUSED(prismaticMap);

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
        (numSpherical > 0 || numFixed > 0 || numRevolute > 0 || numGear > 0)) {
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

  if (mConfig.enableLocal6x6Solve) {
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
    if (doSolveDebug && (numSpherical > 0 || numFixed > 0)) {
      printf("    delta pos=(%.6f,%.6f,%.6f) rot=(%.6f,%.6f,%.6f)\n",
             deltaPos.x, deltaPos.y, deltaPos.z, deltaTheta.x, deltaTheta.y,
             deltaTheta.z);
      printf("    newPos=(%.4f,%.4f,%.4f)\n", body.position.x - deltaPos.x,
             body.position.y - deltaPos.y, body.position.z - deltaPos.z);
    }
    // Only increment once per full body loop (not per body)
    if (bodyIndex == 0 && (numSpherical > 0 || numFixed > 0)) {
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

void AvbdSolver::blockDescentIteration(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts, physx::PxReal dt,
    const AvbdBodyConstraintMap *contactMap, AvbdColorBatch *colorBatches,
    physx::PxU32 numColors) {

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  // True Block Coordinate Descent: iterate over bodies, not constraints
  // For each body, solve a local optimization problem considering all
  // constraints that affect this body.

  const bool useDeterministicOrder =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);

  physx::PxArray<physx::PxU32> bodyOrder;
  if (useDeterministicOrder) {
    bodyOrder.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      bodyOrder[i] = i;
    }
    // Sort bodies by index for deterministic processing
    std::sort(bodyOrder.begin(), bodyOrder.end(),
              [&bodies](physx::PxU32 a, physx::PxU32 b) {
                return bodies[a].invMass > bodies[b].invMass;
              });
  }

  const physx::PxReal invDt2 = 1.0f / (dt * dt);
  const physx::PxU32 *orderPtr =
      useDeterministicOrder ? bodyOrder.begin() : nullptr;

  for (physx::PxU32 idx = 0; idx < numBodies; ++idx) {
    const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
    if (bodies[i].invMass <= 0.0f) {
      continue;
    }

    if (mConfig.enableLocal6x6Solve) {
      // 6x6 fully-coupled solve via LDLT
      solveLocalSystem(bodies[i], bodies, numBodies, contacts, numContacts, dt,
                       invDt2, contactMap);
    } else {
      // 3x3 decoupled solve (position + rotation independently)
      solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                 numContacts, nullptr, 0, nullptr, 0, nullptr,
                                 0, nullptr, 0, nullptr, 0, nullptr, 0, dt,
                                 invDt2, contactMap);
    }
  }
}

/**
 * @brief Compute position/rotation correction for a spherical joint
 *
 * Uses fused AVBD primal+dual (XPBD equivalent):
 *   For each axis k:
 *     alpha = 1 / rho
 *     delta_lambda_k = -(C_k + alpha * lambda_k) / (w_k + alpha / h^2)
 *     lambda_k += delta_lambda_k
 *     dx = (1/m) * J^T * delta_lambda (split to this body's share)
 *
 * This is mathematically identical to the AVBD AL formulation but expressed
 * in constraint-space (XPBD dual form). The lambda update is fused into the
 * primal step, eliminating the need for a separate dual step.
 */
bool AvbdSolver::computeSphericalJointCorrection(
    AvbdSphericalJointConstraint &joint, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
    physx::PxVec3 &deltaTheta, physx::PxReal dt) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }

  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);

  // Get other body
  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }

  // Compute world anchor positions
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorB)
                             : joint.anchorB;
  } else {
    worldAnchorA = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorA)
                             : joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }

  // Compute constraint error
  physx::PxVec3 error = worldAnchorA - worldAnchorB;
  physx::PxReal errorMag = error.magnitude();

  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);
  bool hasCorrection = false;

  if (errorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    // AVBD/XPBD fused primal+dual:
    //   alpha = 1/rho  (compliance)
    //   alphaHat = alpha / h^2  (time-scaled compliance)
    //   delta_lambda = -(C + alphaHat * lambda) / (w + alphaHat)
    //   lambda += delta_lambda
    //   dx = invM * J^T * delta_lambda
    const physx::PxReal rho = joint.header.rho;
    const physx::PxReal alpha = 1.0f / rho;
    const physx::PxReal h2 = dt * dt;
    const physx::PxReal alphaHat = alpha / h2; // = 1/(rho * h^2)

    physx::PxVec3 r = isBodyA ? body.rotation.rotate(joint.anchorA)
                              : body.rotation.rotate(joint.anchorB);
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ? otherBody->rotation.rotate(joint.anchorB)
                       : otherBody->rotation.rotate(joint.anchorA);
    }
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

    for (int axis = 0; axis < 3; ++axis) {
      physx::PxReal C = error[axis];
      if (physx::PxAbs(C) < AvbdConstants::AVBD_NUMERICAL_EPSILON)
        continue;

      physx::PxVec3 n(0.0f);
      (&n.x)[axis] = 1.0f;

      // Effective inverse mass: w = J * M^{-1} * J^T
      physx::PxVec3 rCrossN = r.cross(n);
      physx::PxReal w =
          body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);

      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOtherCrossN = rOther.cross(n);
        w += otherBody->invMass +
             rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
      }

      if (w > 1e-6f) {
        // XPBD delta_lambda (== AVBD primal Newton step in constraint space)
        physx::PxReal deltaLambda =
            -(C + alphaHat * joint.lambda[axis]) / (w + alphaHat);
        deltaLambda =
            physx::PxClamp(deltaLambda, -mConfig.maxPositionCorrection,
                           mConfig.maxPositionCorrection);

        // Fused dual update: accumulate lambda
        joint.lambda[axis] += deltaLambda;

        // Primal position update for THIS body
        (&deltaPos.x)[axis] += deltaLambda * body.invMass * sign;
        deltaTheta += (body.invInertiaWorld * rCrossN) * (deltaLambda * sign);
        hasCorrection = true;
      }
    }
  }

  // Process cone limit if enabled
  if (joint.hasConeLimit && joint.coneAngleLimit > 0.0f) {
    physx::PxQuat rotA = isBodyA
                             ? body.rotation
                             : (otherBody ? otherBody->rotation
                                          : physx::PxQuat(physx::PxIdentity));
    physx::PxQuat rotB = isBodyA
                             ? (otherBody ? otherBody->rotation
                                          : physx::PxQuat(physx::PxIdentity))
                             : body.rotation;

    // Compute cone violation
    physx::PxVec3 worldAxisA = rotA.rotate(joint.coneAxisA);
    physx::PxVec3 worldAxisB = rotB.rotate(joint.coneAxisA);

    physx::PxReal cosAngle = worldAxisA.dot(worldAxisB);
    cosAngle = physx::PxClamp(cosAngle, -1.0f, 1.0f);
    physx::PxReal angle = physx::PxAcos(cosAngle);

    if (angle > joint.coneAngleLimit) {
      // Cone limit violated - inequality constraint, use XPBD with clamping
      physx::PxReal violation = angle - joint.coneAngleLimit;

      physx::PxVec3 corrAxis = worldAxisA.cross(worldAxisB);
      physx::PxReal corrAxisMag = corrAxis.magnitude();

      if (corrAxisMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
        corrAxis /= corrAxisMag;

        physx::PxReal angularW = 0.0f;
        angularW += (body.invInertiaWorld * corrAxis).dot(corrAxis);
        if (otherBody && otherBody->invMass > 0.0f) {
          angularW += (otherBody->invInertiaWorld * corrAxis).dot(corrAxis);
        }

        if (angularW > 1e-6f) {
          const physx::PxReal rho = joint.header.rho;
          const physx::PxReal alpha = 1.0f / rho;
          const physx::PxReal h2 = dt * dt;
          const physx::PxReal alphaHat = alpha / h2;

          physx::PxReal deltaLambda =
              -(violation + alphaHat * joint.coneLambda) /
              (angularW + alphaHat);

          // Cone is inequality: lambda <= 0 (resist separation only)
          physx::PxReal newLambda = joint.coneLambda + deltaLambda;
          newLambda = physx::PxMin(newLambda, 0.0f);
          deltaLambda = newLambda - joint.coneLambda;
          joint.coneLambda = newLambda;

          physx::PxReal bodySign = isBodyA ? 1.0f : -1.0f;
          physx::PxVec3 angCorrection = corrAxis * (deltaLambda * bodySign);
          deltaTheta += body.invInertiaWorld * angCorrection;
          hasCorrection = true;
        }
      }
    }
  }

  return hasCorrection;
}

/**
 * @brief Compute position/rotation correction for a fixed joint
 *
 * Uses fused AVBD primal+dual (XPBD equivalent) for both position (3 DOF)
 * and rotation (3 DOF) constraints. Lambda updated in-place.
 */
bool AvbdSolver::computeFixedJointCorrection(
    AvbdFixedJointConstraint &joint, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
    physx::PxVec3 &deltaTheta, physx::PxReal dt) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

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

  // XPBD compliance parameters
  const physx::PxReal rho = joint.header.rho;
  const physx::PxReal alpha = 1.0f / rho;
  const physx::PxReal h2 = dt * dt;
  const physx::PxReal alphaHat = alpha / h2;

  // Position constraint (same as spherical)
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorB)
                             : joint.anchorB;
  } else {
    worldAnchorA = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorA)
                             : joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }

  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  physx::PxReal posErrorMag = posError.magnitude();

  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);

  // --- Position constraint: per-axis XPBD ---
  if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 r = isBodyA ? body.rotation.rotate(joint.anchorA)
                              : body.rotation.rotate(joint.anchorB);
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ? otherBody->rotation.rotate(joint.anchorB)
                       : otherBody->rotation.rotate(joint.anchorA);
    }
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

    for (int axis = 0; axis < 3; ++axis) {
      physx::PxReal C = posError[axis];
      if (physx::PxAbs(C) < AvbdConstants::AVBD_NUMERICAL_EPSILON)
        continue;

      physx::PxVec3 n(0.0f);
      (&n.x)[axis] = 1.0f;

      physx::PxVec3 rCrossN = r.cross(n);
      physx::PxReal w =
          body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);

      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOtherCrossN = rOther.cross(n);
        w += otherBody->invMass +
             rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
      }

      if (w > 1e-6f) {
        physx::PxReal deltaLambda =
            -(C + alphaHat * joint.lambdaPosition[axis]) / (w + alphaHat);
        deltaLambda =
            physx::PxClamp(deltaLambda, -mConfig.maxPositionCorrection,
                           mConfig.maxPositionCorrection);

        joint.lambdaPosition[axis] += deltaLambda;

        (&deltaPos.x)[axis] += deltaLambda * body.invMass * sign;
        deltaTheta += (body.invInertiaWorld * rCrossN) * (deltaLambda * sign);
      }
    }
  }

  // --- Rotation constraint: per-axis XPBD ---
  physx::PxQuat rotA, rotB;
  if (isBodyA) {
    rotA = body.rotation;
    rotB = otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity);
  } else {
    rotA = otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity);
    rotB = body.rotation;
  }

  physx::PxVec3 rotError = joint.computeRotationViolation(rotA, rotB);
  physx::PxReal rotErrorMag = rotError.magnitude();

  if (rotErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

    for (int axis = 0; axis < 3; ++axis) {
      physx::PxReal C = rotError[axis];
      if (physx::PxAbs(C) < AvbdConstants::AVBD_NUMERICAL_EPSILON)
        continue;

      physx::PxVec3 n(0.0f);
      (&n.x)[axis] = 1.0f;

      physx::PxReal w = n.dot(body.invInertiaWorld * n);
      if (otherBody && otherBody->invMass > 0.0f) {
        w += n.dot(otherBody->invInertiaWorld * n);
      }

      if (w > 1e-6f) {
        physx::PxReal deltaLambda =
            -(C + alphaHat * joint.lambdaRotation[axis]) / (w + alphaHat);
        deltaLambda = physx::PxClamp(deltaLambda, -mConfig.maxAngularCorrection,
                                     mConfig.maxAngularCorrection);

        joint.lambdaRotation[axis] += deltaLambda;

        deltaTheta += (body.invInertiaWorld * n) * (deltaLambda * sign);
      }
    }
  }

  return (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON ||
          rotErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON);
}

/**
 * @brief Compute correction for revolute joint
 */
bool AvbdSolver::computeRevoluteJointCorrection(
    const AvbdRevoluteJointConstraint &joint, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
    physx::PxVec3 &deltaTheta) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

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

  // Position constraint
  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorB)
                             : joint.anchorB;
  } else {
    worldAnchorA = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorA)
                             : joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }

  physx::PxVec3 posError = worldAnchorA - worldAnchorB;
  physx::PxReal posErrorMag = posError.magnitude();

  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);

  if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 direction = posError / posErrorMag;

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
  }

  // Axis alignment constraint
  physx::PxVec3 worldAxisA =
      isBodyA
          ? body.rotation.rotate(joint.axisA)
          : (otherBody ? otherBody->rotation.rotate(joint.axisA) : joint.axisA);
  physx::PxVec3 worldAxisB =
      isBodyA
          ? (otherBody ? otherBody->rotation.rotate(joint.axisB) : joint.axisB)
          : body.rotation.rotate(joint.axisB);

  physx::PxVec3 axisCross = worldAxisA.cross(worldAxisB);
  physx::PxReal axisError = axisCross.magnitude();

  if (axisError > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxReal sign = isBodyA ? -1.0f : 1.0f;
    deltaTheta += axisCross * (0.5f * sign);
  }

  return (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON ||
          axisError > AvbdConstants::AVBD_NUMERICAL_EPSILON);
}

/**
 * @brief Compute correction for prismatic joint
 */
bool AvbdSolver::computePrismaticJointCorrection(
    const AvbdPrismaticJointConstraint &joint, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
    physx::PxVec3 &deltaTheta) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

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

  physx::PxVec3 worldAnchorA, worldAnchorB;
  physx::PxVec3 worldAxis;

  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorB)
                             : joint.anchorB;
    worldAxis = body.rotation.rotate(joint.axisA);
  } else {
    worldAnchorA = otherBody ? otherBody->position +
                                   otherBody->rotation.rotate(joint.anchorA)
                             : joint.anchorA;
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
    worldAxis =
        otherBody ? otherBody->rotation.rotate(joint.axisA) : joint.axisA;
  }

  physx::PxVec3 diff = worldAnchorA - worldAnchorB;
  physx::PxVec3 perpError = diff - worldAxis * diff.dot(worldAxis);
  physx::PxReal perpErrorMag = perpError.magnitude();

  if (perpErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    physx::PxVec3 direction = perpError / perpErrorMag;

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
      physx::PxReal correctionMag = -perpErrorMag / w;
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

      deltaPos = direction * (correctionMag * body.invMass * sign);
      deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
    }
  }

  return perpErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON;
}

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
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const physx::PxVec3 &gravity, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *sphericalMap,
    const AvbdBodyConstraintMap *fixedMap,
    const AvbdBodyConstraintMap *revoluteMap,
    const AvbdBodyConstraintMap *prismaticMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors) {

  PX_PROFILE_ZONE("AVBD.solveWithJoints", 0);

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();
  mStats.numBodies = numBodies;
  mStats.numContacts = numContacts;
  mStats.numJoints =
      numSpherical + numFixed + numRevolute + numPrismatic + numD6 + numGear;

  const physx::PxReal invDt = 1.0f / dt;
  const physx::PxReal invDt2 = invDt * invDt;

#if AVBD_JOINT_DEBUG
  const bool doDebug = (s_avbdJointDebugFrame < AVBD_JOINT_DEBUG_FRAMES);
  if (doDebug) {
    printf("\n=== AVBD solveWithJoints FRAME %u === bodies=%u contacts=%u "
           "sph=%u fix=%u rev=%u pri=%u d6=%u gear=%u\n",
           s_avbdJointDebugFrame, numBodies, numContacts, numSpherical,
           numFixed, numRevolute, numPrismatic, numD6, numGear);
  }
#endif

  // =========================================================================
  // Stage 1: Prediction
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
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
    for (physx::PxU32 j = 0; j < numSpherical; ++j)
      addEdge(sphericalJoints[j].header.bodyIndexA,
              sphericalJoints[j].header.bodyIndexB);
    for (physx::PxU32 j = 0; j < numFixed; ++j)
      addEdge(fixedJoints[j].header.bodyIndexA,
              fixedJoints[j].header.bodyIndexB);
    for (physx::PxU32 j = 0; j < numRevolute; ++j)
      addEdge(revoluteJoints[j].header.bodyIndexA,
              revoluteJoints[j].header.bodyIndexB);
    for (physx::PxU32 j = 0; j < numPrismatic; ++j)
      addEdge(prismaticJoints[j].header.bodyIndexA,
              prismaticJoints[j].header.bodyIndexB);
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
      contacts[c].C0 = (wA - wB).dot(contacts[c].contactNormal) +
                       contacts[c].penetrationDepth;
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
    // Print fixed joint lambda and broken status
    for (physx::PxU32 j = 0; j < numFixed; ++j) {
      printf("    fixed[%u] bodyA=%u bodyB=%u broken=%u "
             "lambdaPos=(%.2f,%.2f,%.2f) "
             "lambdaRot=(%.2f,%.2f,%.2f) rho=%.0f\n",
             j, fixedJoints[j].header.bodyIndexA,
             fixedJoints[j].header.bodyIndexB,
             (unsigned)fixedJoints[j].isBroken, fixedJoints[j].lambdaPosition.x,
             fixedJoints[j].lambdaPosition.y, fixedJoints[j].lambdaPosition.z,
             fixedJoints[j].lambdaRotation.x, fixedJoints[j].lambdaRotation.y,
             fixedJoints[j].lambdaRotation.z, fixedJoints[j].header.rho);
    }
    for (physx::PxU32 j = 0; j < numSpherical; ++j) {
      printf("    spherical[%u] bodyA=%u bodyB=%u lambda=(%.2f,%.2f,%.2f) "
             "rho=%.0f "
             "coneLimit=%.3f\n",
             j, sphericalJoints[j].header.bodyIndexA,
             sphericalJoints[j].header.bodyIndexB, sphericalJoints[j].lambda.x,
             sphericalJoints[j].lambda.y, sphericalJoints[j].lambda.z,
             sphericalJoints[j].header.rho, sphericalJoints[j].coneAngleLimit);
    }
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

    const physx::PxU32 jointIterations =
        (mStats.numJoints > 0)
            ? physx::PxMax(mConfig.innerIterations, physx::PxU32(8))
            : mConfig.innerIterations;

    for (physx::PxU32 iter = 0; iter < jointIterations; ++iter) {
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

        for (physx::PxU32 idx = 0; idx < numBodies; ++idx) {
          const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
          if (bodies[i].invMass <= 0.0f)
            continue;

          // ---------------------------------------------------------------
          // (A) Unified AVBD: contacts + spherical + fixed + D6 joints
          //     All constraints touching this body are accumulated into
          //     ONE per-body system and solved simultaneously.
          //     6x6: full LDLT.  3x3: decoupled diagonal blocks.
          //     Bodies with no constraints snap to inertialPosition.
          //     Revolute/prismatic/gear: TODO accumulate; currently
          //     still handled by GS in step (B) below.
          // ---------------------------------------------------------------
          solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                     numContacts, sphericalJoints, numSpherical,
                                     fixedJoints, numFixed, revoluteJoints,
                                     numRevolute, prismaticJoints, numPrismatic,
                                     d6Joints, numD6, gearJoints, numGear, dt,
                                     invDt2, contactMap, sphericalMap, fixedMap,
                                     revoluteMap, prismaticMap, d6Map, gearMap);

          // ---------------------------------------------------------------
          // (B) Joint constraints: Gauss-Seidel corrections
          //     Each correction is applied immediately so subsequent joints
          //     see the updated body state.
          //
          //     Full correction (no SOR relaxation) -- the PBD correction
          //     formula divides by generalized inverse mass w which already
          //     ensures the correction is properly scaled. Under-relaxation
          //     only slows chain convergence without improving stability.
          //
          //     For each joint affecting this body:
          //       1. Compute PBD correction (deltaPos, deltaTheta)
          //       2. Apply position += dp
          //       3. Apply rotation via quaternion update
          //
          //     With map: O(K) per body.  Without map: O(J) fallback.
          // ---------------------------------------------------------------

          // Spherical and Fixed joints are now handled by the unified
          // AVBD solver (solveLocalSystemWithJoints) above, not GS.
          // const physx::PxReal maxAngleRevolute = 0.20f; // Unused, disabled
          // GS fallback
          const physx::PxReal maxAnglePrismatic = 0.20f;
          // maxAngleD6 and maxAngleGear no longer needed (both handled by AVBD
          // Hessian)

          // Helper lambda: apply a single joint correction to body i
          auto applyJointGS = [&](const physx::PxVec3 &dp,
                                  const physx::PxVec3 &dth,
                                  physx::PxReal maxAngle) {
            bodies[i].position += dp;
            const physx::PxVec3 &dthR = dth;
            physx::PxReal angle = dthR.magnitude();
            if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
              angle = physx::PxMin(angle, maxAngle);
              physx::PxVec3 axis = dthR.getNormalized();
              physx::PxReal ha = angle * 0.5f;
              physx::PxQuat dq(axis.x * physx::PxSin(ha),
                               axis.y * physx::PxSin(ha),
                               axis.z * physx::PxSin(ha), physx::PxCos(ha));
              bodies[i].rotation = (dq * bodies[i].rotation).getNormalized();
            }
          };

          // Spherical and Fixed joints: handled in joint-centric pass below.
          // (Body-centric GS diverges for fixed joints due to
          //  position-rotation coupling + double lambda update.)

          // Revolute joints (DISABLED - now integrated into
          // solveLocalSystemWithJoints)
          /*
          if (revoluteJoints && numRevolute > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (revoluteMap) {
              revoluteMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numRevolute)
                  continue;
                physx::PxVec3 dp, dth;
                if (computeRevoluteJointCorrection(
                        revoluteJoints[jIdx[k]], bodies, numBodies, i, dp,
          dth)) applyJointGS(dp, dth, maxAngleRevolute);
              }
            } else {
              for (physx::PxU32 k = 0; k < numRevolute; ++k) {
                physx::PxVec3 dp, dth;
                if (computeRevoluteJointCorrection(revoluteJoints[k], bodies,
                                                   numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAngleRevolute);
              }
            }
          }
          */

          // Prismatic joints
          if (prismaticJoints && numPrismatic > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (prismaticMap) {
              prismaticMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numPrismatic)
                  continue;
                physx::PxVec3 dp, dth;
                if (computePrismaticJointCorrection(prismaticJoints[jIdx[k]],
                                                    bodies, numBodies, i, dp,
                                                    dth))
                  applyJointGS(dp, dth, maxAnglePrismatic);
              }
            } else {
              for (physx::PxU32 k = 0; k < numPrismatic; ++k) {
                physx::PxVec3 dp, dth;
                if (computePrismaticJointCorrection(prismaticJoints[k], bodies,
                                                    numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAnglePrismatic);
              }
            }
          }

          // D6 joints -- now handled in solveLocalSystemWithJoints (AVBD
          // Hessian) GS fallback disabled. if (d6Joints && numD6 > 0) { ... }

          // Gear joints -- now handled in solveLocalSystemWithJoints (AVBD
          // Hessian Step 3g) via velocity-level constraint accumulation.
          // GS fallback removed: it fought the Hessian and caused divergence.

        } // end body loop
        mStats.totalIterations++;
      }

      // -----------------------------------------------------------------
      // Joint-centric XPBD pass for spherical and fixed joints:
      //   DISABLED -- these are now integrated into the Hessian in
      //   solveLocalSystemWithJoints (true AVBD).
      //   Lambda is updated in the dual step below (not fused).
      // -----------------------------------------------------------------

      // --- Dual step: AL multiplier updates ---
      //
      // CONTACTS: update every iteration. The unilateral clamp
      //   f = min(0, pen*C + lambda) prevents overcorrection, so frequent
      //   dual updates are safe and improve convergence.
      //
      // JOINTS (spherical + fixed + D6): 3-mechanism ADMM-safe AL dual.
      //   (A) Primal auto-boost: effectiveRho = max(rho, M/h^2) ensures
      //       penalty is always >= body inertia for good convergence.
      //   (B) ADMM-safe dual step: rhoDual = min(Mh2, rho^2/(rho+Mh2))
      //       prevents dual overshoot for both light and heavy bodies.
      //   (C) Lambda decay: lambda = 0.99*lambda + rhoDual*C acts as a
      //       leaky integrator that damps oscillation modes.
      //
      // OTHER JOINTS (revolute, prismatic, gear): still use GS primal
      //   + AL dual. These don't form long chains in typical usage.
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt);

        // ---------------------------------------------------------------
        // Spherical, Fixed, D6: ADMM-safe dual + lambda decay
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

          // Spherical joints
          for (physx::PxU32 j = 0; j < numSpherical; ++j) {
            AvbdSphericalJointConstraint &jnt = sphericalJoints[j];
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
            jnt.lambda = jnt.lambda * lambdaDecay + (wA - wB) * rhoDual;

            if (jnt.hasConeLimit && jnt.coneAngleLimit > 0.0f) {
              physx::PxQuat rotA = aStatic
                                       ? physx::PxQuat(physx::PxIdentity)
                                       : bodies[jnt.header.bodyIndexA].rotation;
              physx::PxQuat rotB = bStatic
                                       ? physx::PxQuat(physx::PxIdentity)
                                       : bodies[jnt.header.bodyIndexB].rotation;
              physx::PxReal coneViol = jnt.computeConeViolation(rotA, rotB);

              jnt.coneLambda -= coneViol * rhoDual;
              jnt.coneLambda =
                  physx::PxMax(-1e9f, physx::PxMin(0.0f, jnt.coneLambda));
            }
          }

          // Fixed joints
          for (physx::PxU32 j = 0; j < numFixed; ++j) {
            AvbdFixedJointConstraint &jnt = fixedJoints[j];
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
            jnt.lambdaPosition =
                jnt.lambdaPosition * lambdaDecay + (wA - wB) * rhoDual;
            jnt.lambdaRotation =
                jnt.lambdaRotation * lambdaDecay +
                jnt.computeRotationViolation(rotA, rotB) * rhoDual;
          }

          // Revolute joints
          for (physx::PxU32 j = 0; j < numRevolute; ++j) {
            AvbdRevoluteJointConstraint &jnt = revoluteJoints[j];
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

            // Position
            jnt.lambdaPosition =
                jnt.lambdaPosition * lambdaDecay + (wA - wB) * rhoDual;

            // Axis Alignment
            physx::PxVec3 axisViol = jnt.computeAxisViolation(rotA, rotB);
            physx::PxVec3 worldAxisA = rotA.rotate(jnt.axisA);
            physx::PxVec3 perp1, perp2;
            if (physx::PxAbs(worldAxisA.x) <
                AvbdConstants::AVBD_AXIS_SELECTION_THRESHOLD) {
              perp1 = worldAxisA.cross(physx::PxVec3(1, 0, 0)).getNormalized();
            } else {
              perp1 = worldAxisA.cross(physx::PxVec3(0, 1, 0)).getNormalized();
            }
            perp2 = worldAxisA.cross(perp1).getNormalized();

            jnt.lambdaAxisAlign[0] = jnt.lambdaAxisAlign[0] * lambdaDecay +
                                     axisViol.dot(perp1) * rhoDual;
            jnt.lambdaAxisAlign[1] = jnt.lambdaAxisAlign[1] * lambdaDecay +
                                     axisViol.dot(perp2) * rhoDual;

            // Angle Limit
            if (jnt.hasAngleLimit) {
              physx::PxReal angleViol =
                  jnt.computeAngleLimitViolation(rotA, rotB);
              physx::PxReal newLam =
                  jnt.lambdaAngleLimit * lambdaDecay + angleViol * rhoDual;

              if (jnt.angleLimitLower < jnt.angleLimitUpper) {
                if (angleViol > 0.0f || jnt.lambdaAngleLimit > 0.0f) {
                  jnt.lambdaAngleLimit = physx::PxMax(0.0f, newLam);
                } else if (angleViol < 0.0f || jnt.lambdaAngleLimit < 0.0f) {
                  jnt.lambdaAngleLimit = physx::PxMin(0.0f, newLam);
                } else {
                  jnt.lambdaAngleLimit = 0.0f;
                }
              } else {
                jnt.lambdaAngleLimit = newLam;
              }
            }
          }

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
            for (int axis = 0; axis < 3; ++axis) {
              physx::PxU32 motion = jnt.getLinearMotion(axis);
              if (motion == 2) // FREE
                continue;

              if (motion == 0) { // LOCKED
                jnt.lambdaLinear[axis] = jnt.lambdaLinear[axis] * lambdaDecay +
                                         posViol[axis] * rhoDual;
              } else if (motion == 1) { // LIMITED
                physx::PxReal limitViol =
                    jnt.computeLinearLimitViolation(posViol[axis], axis);
                physx::PxReal newLam =
                    jnt.lambdaLinear[axis] * lambdaDecay + limitViol * rhoDual;

                if (jnt.linearLimitLower[axis] < jnt.linearLimitUpper[axis]) {
                  if (limitViol > 0.0f || jnt.lambdaLinear[axis] > 0.0f) {
                    jnt.lambdaLinear[axis] = physx::PxMax(0.0f, newLam);
                  } else if (limitViol < 0.0f ||
                             jnt.lambdaLinear[axis] < 0.0f) {
                    jnt.lambdaLinear[axis] = physx::PxMin(0.0f, newLam);
                  } else {
                    jnt.lambdaLinear[axis] = 0.0f;
                  }
                } else {
                  jnt.lambdaLinear[axis] = newLam;
                }
              }
            }

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
                    jnt.lambdaAngular[axis] * lambdaDecay + limitViol * rhoDual;

                if (jnt.angularLimitLower[axis] < jnt.angularLimitUpper[axis]) {
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

                physx::PxReal rhoDualDrive =
                    physx::PxMin(damping / dt2, rhoDual);
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
                  physx::PxReal rhoDualDrive =
                      physx::PxMin(damping / dt2, rhoDual);
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
                  // PhysX TGS convention: Twist/Swing target velocities are
                  // applied as (wA - wB), meaning wB - wA = -target. SLERP is
                  // applied as wB - wA = target, which is handled above.
                  physx::PxReal targetOmega_dt = -worldAngTarget.dot(wAxis);
                  physx::PxReal C = relDW.dot(wAxis) - targetOmega_dt;

                  physx::PxReal rhoDualDrive =
                      physx::PxMin(damping / dt2, rhoDual);
                  (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] =
                      (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] *
                          lambdaDecay +
                      rhoDualDrive * C;
                }
              }
            }
          }
        }

        // Other joint types: AL dual as before
        for (physx::PxU32 j = 0; j < numRevolute; ++j)
          updateRevoluteJointMultiplier(revoluteJoints[j], bodies, numBodies,
                                        mConfig);
        for (physx::PxU32 j = 0; j < numPrismatic; ++j)
          updatePrismaticJointMultiplier(prismaticJoints[j], bodies, numBodies,
                                         mConfig);
        for (physx::PxU32 j = 0; j < numGear; ++j)
          updateGearJointMultiplier(gearJoints[j], bodies, numBodies, mConfig);
      }
    } // end iteration loop
  }

  // =========================================================================
  // Stage 6: Motor drives for RevoluteJoints (post-solve)
  //
  // Motor applies torque (limited by maxForce) to accelerate toward target
  // velocity. This matches TGS behavior where motor gradually accelerates
  // bodies. Applied AFTER constraint iterations so joint corrections are
  // stable.
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.motorDrives", 0);

    physx::PxU64 currentFrame = getAvbdMotorFrameCounter();
    static physx::PxU64 lastMotorFrame = 0;
    static physx::PxU32 processedBodyFlags = 0;

    if (currentFrame != lastMotorFrame) {
      lastMotorFrame = currentFrame;
      processedBodyFlags = 0;
    }

    for (physx::PxU32 j = 0; j < numRevolute; ++j) {
      AvbdRevoluteJointConstraint &joint = revoluteJoints[j];
      if (!joint.motorEnabled || joint.motorMaxForce <= 0.0f)
        continue;

      const physx::PxU32 idxA = joint.header.bodyIndexA;
      const physx::PxU32 idxB = joint.header.bodyIndexB;
      const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
      const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);
      if (isAStatic && isBStatic)
        continue;

      if (idxB < 32 && (processedBodyFlags & (1u << idxB)))
        continue;
      if (idxB < 32)
        processedBodyFlags |= (1u << idxB);

      AvbdSolverBody &bodyB = bodies[idxB];

      physx::PxVec3 worldAxis =
          isAStatic ? joint.axisA : bodies[idxA].rotation.rotate(joint.axisA);
      worldAxis.normalize();

      // Current angular velocity from position-level solver
      physx::PxQuat deltaQ = bodyB.rotation * bodyB.prevRotation.getConjugate();
      if (deltaQ.w < 0.0f)
        deltaQ = -deltaQ;
      physx::PxVec3 currentAngVel =
          physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
      physx::PxReal currentAxisVel = currentAngVel.dot(worldAxis);

      physx::PxReal velocityError = joint.motorTargetVelocity - currentAxisVel;

      physx::PxVec3 invITimesAxis = bodyB.invInertiaWorld * worldAxis;
      physx::PxReal effectiveInvInertia = worldAxis.dot(invITimesAxis);
      if (effectiveInvInertia < 1e-10f)
        continue;

      physx::PxReal effectiveInertia = 1.0f / effectiveInvInertia;
      physx::PxReal requiredTorque = effectiveInertia * velocityError * invDt;
      physx::PxReal clampedTorque = physx::PxClamp(
          requiredTorque, -joint.motorMaxForce, joint.motorMaxForce);
      physx::PxReal angularAccel = clampedTorque * effectiveInvInertia;
      physx::PxReal deltaAngle = angularAccel * dt * dt;

      // Apply rotation to body B
      {
        physx::PxReal ha = deltaAngle * 0.5f;
        physx::PxQuat dRot(worldAxis.x * physx::PxSin(ha),
                           worldAxis.y * physx::PxSin(ha),
                           worldAxis.z * physx::PxSin(ha), physx::PxCos(ha));
        bodyB.rotation = (dRot * bodyB.rotation).getNormalized();
      }

      // Apply opposite rotation to body A if dynamic
      if (!isAStatic) {
        AvbdSolverBody &bodyA = bodies[idxA];
        physx::PxVec3 invITimesAxisA = bodyA.invInertiaWorld * worldAxis;
        physx::PxReal effectiveInvInertiaA = worldAxis.dot(invITimesAxisA);
        if (effectiveInvInertiaA > 1e-10f) {
          physx::PxReal deltaAngleA =
              clampedTorque * effectiveInvInertiaA * dt * dt;
          physx::PxReal ha = -deltaAngleA * 0.5f;
          physx::PxQuat dRotA(worldAxis.x * physx::PxSin(ha),
                              worldAxis.y * physx::PxSin(ha),
                              worldAxis.z * physx::PxSin(ha), physx::PxCos(ha));
          bodyA.rotation = (dRotA * bodyA.rotation).getNormalized();
        }
      }
    }
  }

  // =========================================================================
  // Stage 7: Velocity update from position change
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
      }
    }

    // D6 joint angular drive damping is now handled entirely by the AVBD
    // AL constraint in solveLocalSystemWithJoints. No extra velocity
    // attenuation is needed here.
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

} // namespace Dy
} // namespace physx
