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

// External frame counter from DyAvbdDynamics.cpp (used by motor drives)
extern physx::PxU64 getAvbdMotorFrameCounter();

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
  //   linearVelocity     = v_{N-1, postsolve}  (clean post-solve from last frame)
  //   prevLinearVelocity  = v_{N-2, postsolve}  (saved at end of frame N-2)
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
      if (massA > 0.0f && massB > 0.0f) {
        // Two dynamic bodies: harmonic mean
        effectiveMass = (massA * massB) / (massA + massB);
      } else {
        // One body is static: use the dynamic body's mass
        effectiveMass = physx::PxMax(massA, massB);
      }

      const physx::PxReal effectiveMassH2 = effectiveMass * invDt2;
      // Floor at 25% of effectiveMass/h^2 -- matches reference penaltyScale
      const physx::PxReal penaltyFloor = 0.25f * effectiveMassH2;
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
      physx::PxVec3 wA = (bA < numBodies)
          ? bodies[bA].prevPosition + bodies[bA].prevRotation.rotate(contacts[c].contactPointA)
          : contacts[c].contactPointA;
      physx::PxVec3 wB = (bB < numBodies)
          ? bodies[bB].prevPosition + bodies[bB].prevRotation.rotate(contacts[c].contactPointB)
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
        contacts[c].header.penalty =
            physx::PxMin(newPenalty, penaltyMax);
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
        prevWorldPosA = bodies[bodyAIdx].prevPosition +
                        bodies[bodyAIdx].prevRotation.rotate(contacts[c].contactPointA);
      } else {
        worldPosA = contacts[c].contactPointA;
        prevWorldPosA = contacts[c].contactPointA;
      }
      if (bodyBIdx < numBodies) {
        worldPosB = bodies[bodyBIdx].position +
                    bodies[bodyBIdx].rotation.rotate(contacts[c].contactPointB);
        prevWorldPosB = bodies[bodyBIdx].prevPosition +
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

  physx::PxVec3 gLinear =
      (body.position - body.inertialPosition) * massInvDt2;

  // Angular inertia RHS: (I/h^2) * deltaW_inertial
  physx::PxQuat deltaQ =
      body.rotation * body.inertialRotation.getConjugate();
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
      physx::PxVec3 relDisp = (worldPosA - prevWorldPosA) - (worldPosB - prevWorldPosB);

      // Tangent 0
      {
        const physx::PxVec3 &t = contacts[c].tangent0;
        physx::PxVec3 rCrossT = r.cross(t);
        physx::PxVec3 tGradPos = t * sign;
        physx::PxVec3 tGradRot = rCrossT * sign;
        physx::PxReal tPen = contacts[c].tangentPenalty0;
        physx::PxReal tLambda = contacts[c].tangentLambda0;
        physx::PxReal tC = relDisp.dot(t);  // tangent constraint value

        // LHS += outer(Jt, Jt * penalty) -- UNCONDITIONAL
        A.addConstraintContribution(tGradPos, tGradRot, tPen);

        // Force: f_t = clamp(pen*C_t + lambda_t, -frictionBound, +frictionBound)
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
// Decoupled 3x3 System Solver -- Block-Diagonal Approximation
//
// Solves position (3x3) and rotation (3x3) independently, dropping the
// off-diagonal coupling between linear and angular DOFs. Same AL framework,
// penalty/lambda, alpha blending, and 3-row friction as the full 6x6 path.
//
// Trade-off: ~40% cheaper per body (two 3x3 inverses vs one 6x6 LDLT)
// but less accurate for contacts with large moment arms.
//=============================================================================

void AvbdSolver::solveLocalSystem3x3(
    AvbdSolverBody &body, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, physx::PxReal invDt2,
    const AvbdBodyConstraintMap *contactMap) {

  if (body.invMass <= 0.0f) {
    return;
  }
  PX_UNUSED(dt);

  const physx::PxU32 bodyIndex = body.nodeIndex;

  // =========================================================================
  // Step 1: Initialize LHS with mass matrices
  // =========================================================================

  physx::PxReal mass = (body.invMass > 1e-8f) ? (1.0f / body.invMass) : 0.0f;
  physx::PxReal massInvDt2 = mass * invDt2;

  // Linear 3x3: Alin = (M/h^2) * I
  physx::PxMat33 Alin(physx::PxVec3(massInvDt2, 0, 0),
                      physx::PxVec3(0, massInvDt2, 0),
                      physx::PxVec3(0, 0, massInvDt2));

  // Angular 3x3: Aang = I_body / h^2
  physx::PxMat33 inertiaTensor = body.invInertiaWorld.getInverse();
  physx::PxMat33 Aang(inertiaTensor.column0 * invDt2,
                      inertiaTensor.column1 * invDt2,
                      inertiaTensor.column2 * invDt2);

  // =========================================================================
  // Step 2: Initialize RHS with inertia terms
  // =========================================================================

  physx::PxVec3 rhsLin =
      (body.position - body.inertialPosition) * massInvDt2;

  physx::PxQuat deltaQ =
      body.rotation * body.inertialRotation.getConjugate();
  if (deltaQ.w < 0.0f) {
    deltaQ = -deltaQ;
  }
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  physx::PxVec3 rhsAng = (inertiaTensor * rotError) * invDt2;

  // =========================================================================
  // Step 3: Iterate constraints -- accumulate into linear & angular
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

    AvbdSolverBody *otherBody = nullptr;
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }

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

    const physx::PxVec3 &normal = contacts[c].contactNormal;
    physx::PxReal violation =
        (worldPosA - worldPosB).dot(normal) + contacts[c].penetrationDepth;

    // Alpha blending
    violation -= mConfig.avbdAlpha * contacts[c].C0;

    physx::PxReal pen = contacts[c].header.penalty;
    physx::PxReal lambda = contacts[c].header.lambda;

    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxVec3 gradPos = normal * sign;
    physx::PxVec3 gradRot = rCrossN * sign;

    // LHS += pen * J * J^T (unconditional, decoupled into linear & angular)
    Alin.column0 += gradPos * (gradPos.x * pen);
    Alin.column1 += gradPos * (gradPos.y * pen);
    Alin.column2 += gradPos * (gradPos.z * pen);

    Aang.column0 += gradRot * (gradRot.x * pen);
    Aang.column1 += gradRot * (gradRot.y * pen);
    Aang.column2 += gradRot * (gradRot.z * pen);

    numTouching++;

    // Force: f = clamp(pen * C + lambda, -inf, 0)
    physx::PxReal f = physx::PxMin(0.0f, pen * violation + lambda);

    if (f < 0.0f) {
      rhsLin += gradPos * f;
      rhsAng += gradRot * f;
    }

    // Friction: 3-row AL (same as 6x6)
    if (contacts[c].friction > 0.0f) {
      physx::PxReal frictionBound =
          physx::PxAbs(contacts[c].header.lambda) * contacts[c].friction;

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
        physx::PxReal tLambda = contacts[c].tangentLambda0;
        physx::PxReal tC = relDisp.dot(t);

        Alin.column0 += tGradPos * (tGradPos.x * tPen);
        Alin.column1 += tGradPos * (tGradPos.y * tPen);
        Alin.column2 += tGradPos * (tGradPos.z * tPen);
        Aang.column0 += tGradRot * (tGradRot.x * tPen);
        Aang.column1 += tGradRot * (tGradRot.y * tPen);
        Aang.column2 += tGradRot * (tGradRot.z * tPen);

        physx::PxReal ft = tPen * tC + tLambda;
        ft = physx::PxClamp(ft, -frictionBound, frictionBound);

        rhsLin += tGradPos * ft;
        rhsAng += tGradRot * ft;
      }

      // Tangent 1
      {
        const physx::PxVec3 &t = contacts[c].tangent1;
        physx::PxVec3 rCrossT = r.cross(t);
        physx::PxVec3 tGradPos = t * sign;
        physx::PxVec3 tGradRot = rCrossT * sign;
        physx::PxReal tPen = contacts[c].tangentPenalty1;
        physx::PxReal tLambda = contacts[c].tangentLambda1;
        physx::PxReal tC = relDisp.dot(t);

        Alin.column0 += tGradPos * (tGradPos.x * tPen);
        Alin.column1 += tGradPos * (tGradPos.y * tPen);
        Alin.column2 += tGradPos * (tGradPos.z * tPen);
        Aang.column0 += tGradRot * (tGradRot.x * tPen);
        Aang.column1 += tGradRot * (tGradRot.y * tPen);
        Aang.column2 += tGradRot * (tGradRot.z * tPen);

        physx::PxReal ft = tPen * tC + tLambda;
        ft = physx::PxClamp(ft, -frictionBound, frictionBound);

        rhsLin += tGradPos * ft;
        rhsAng += tGradRot * ft;
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
  // Step 4: Solve two independent 3x3 systems
  // =========================================================================

  physx::PxVec3 deltaPos = Alin.getInverse().transform(rhsLin);
  physx::PxVec3 deltaTheta = Aang.getInverse().transform(rhsAng);

  // =========================================================================
  // Step 5: Apply updates
  // =========================================================================

  body.position -= deltaPos;

  if (deltaTheta.magnitudeSquared() > 1e-12f) {
    physx::PxQuat dq(deltaTheta.x, deltaTheta.y, deltaTheta.z, 0.0f);
    body.rotation =
        (body.rotation - dq * body.rotation * 0.5f).getNormalized();
  }
}

//=============================================================================
// Helper Methods for 6x6 System Solver
//=============================================================================

physx::PxU32 AvbdSolver::collectBodyConstraints(
    physx::PxU32 bodyIndex, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 *constraintIndices) {

  physx::PxU32 count = 0;
  for (physx::PxU32 c = 0; c < numContacts && count < 64; ++c) {
    physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyB = contacts[c].header.bodyIndexB;

    if (bodyA == bodyIndex || bodyB == bodyIndex) {
      constraintIndices[count++] = c;
    }
  }
  return count;
}

void AvbdSolver::buildHessianMatrix(const AvbdSolverBody &body,
                                    AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    AvbdContactConstraint *contacts,
                                    physx::PxU32 numContacts,
                                    physx::PxReal invDt2, AvbdBlock6x6 &H) {

  PX_UNUSED(bodies);
  PX_UNUSED(numBodies);

  // Initialize with mass/inertia contribution: M/h^2
  H.initializeDiagonal(body.invMass, body.invInertiaWorld, invDt2);

  // Add constraint contributions (unconditionally for ALL contacts)
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyB = contacts[c].header.bodyIndexB;

    // Skip if this body is not involved
    if (bodyA != body.nodeIndex && bodyB != body.nodeIndex) {
      continue;
    }

    // Compute constraint gradient (Jacobian)
    // Note: Hessian contribution (rho * J^T * J) is accumulated for ALL
    // contacts unconditionally, matching the reference AVBD3D.
    // No active set check needed here -- the Hessian does not depend on
    // whether the contact force is zero or not.
    physx::PxVec3 gradPos, gradRot;
    if (bodyA == body.nodeIndex) {
      gradPos = contacts[c].contactNormal;
      physx::PxVec3 rA = body.rotation.rotate(contacts[c].contactPointA);
      gradRot = rA.cross(contacts[c].contactNormal);
    } else {
      gradPos = -contacts[c].contactNormal;
      physx::PxVec3 rB = body.rotation.rotate(contacts[c].contactPointB);
      gradRot = -rB.cross(contacts[c].contactNormal);
    }

    // Add constraint contribution: H += penalty * J^T * J
    // Uses adaptive penalty that grows via beta*|C| in dual update
    H.addConstraintContribution(gradPos, gradRot, contacts[c].header.penalty);
  }
}

void AvbdSolver::buildGradientVector(const AvbdSolverBody &body,
                                     AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     AvbdContactConstraint *contacts,
                                     physx::PxU32 numContacts,
                                     physx::PxReal invDt2, AvbdVec6 &g) {

  // Initialize with inertia term: M/h^2 * (x - x_inertial)
  physx::PxReal massContrib =
      (body.invMass > 0.0f) ? (1.0f / body.invMass) * invDt2 : 0.0f;
  g.linear = (body.position - body.inertialPosition) * massContrib;

  // Angular inertia term: I/h^2 * (theta - theta_inertial)
  // Compute rotation difference as axis-angle: deltaQ = q * q_inertial^-1
  physx::PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
  // Ensure shortest path
  if (deltaQ.w < 0.0f) {
    deltaQ = -deltaQ;
  }
  // Extract axis-angle: theta ~= 2 * imaginary part for small angles
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  // Compute I = inv(invInertiaWorld) using full 3x3 inverse,
  // then multiply: g_angular = (I / h^2) * rotError
  physx::PxMat33 inertiaTensor = body.invInertiaWorld.getInverse();
  g.angular = (inertiaTensor * rotError) * invDt2;

  // Add constraint gradient contributions
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyB = contacts[c].header.bodyIndexB;

    // Skip if this body is not involved
    if (bodyA != body.nodeIndex && bodyB != body.nodeIndex) {
      continue;
    }

    // Compute constraint violation correctly handling static bodies
    // For static bodies, contactPoint is already in world coords
    physx::PxVec3 worldPointA, worldPointB;
    if (bodyA < numBodies) {
      worldPointA = bodies[bodyA].position +
                    bodies[bodyA].rotation.rotate(contacts[c].contactPointA);
    } else {
      worldPointA = contacts[c].contactPointA; // Static: already world coords
    }
    if (bodyB < numBodies) {
      worldPointB = bodies[bodyB].position +
                    bodies[bodyB].rotation.rotate(contacts[c].contactPointB);
    } else {
      worldPointB = contacts[c].contactPointB; // Static: already world coords
    }
    physx::PxReal violation =
        (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
        contacts[c].penetrationDepth;

    // Compute AL force with adaptive penalty, clamped for unilateral contacts
    physx::PxReal force =
        physx::PxMin(0.0f, contacts[c].header.penalty * violation + contacts[c].header.lambda);

    // Skip inactive contacts (no repulsive force)
    if (force >= 0.0f) {
      continue;
    }

    // Compute constraint gradient (Jacobian)
    physx::PxVec3 gradPos, gradRot;
    if (bodyA == body.nodeIndex) {
      gradPos = contacts[c].contactNormal;
      physx::PxVec3 rA = body.rotation.rotate(contacts[c].contactPointA);
      gradRot = rA.cross(contacts[c].contactNormal);
    } else {
      gradPos = -contacts[c].contactNormal;
      physx::PxVec3 rB = body.rotation.rotate(contacts[c].contactPointB);
      gradRot = -rB.cross(contacts[c].contactNormal);
    }

    // Add contribution: g += J^T * F
    g.linear += gradPos * force;
    g.angular += gradRot * force;
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
      solveLocalSystem3x3(bodies[i], bodies, numBodies, contacts, numContacts,
                          dt, invDt2, contactMap);
    }
  }
}

/**
 * @brief Solve all constraints affecting a single body (Block Descent step)
 *
 * This is the core of AVBD: for a single body, we consider all constraints
 * affecting it and compute the optimal position/rotation update.
 */
void AvbdSolver::solveBodyLocalConstraints(AvbdSolverBody *bodies,
                                           physx::PxU32 numBodies,
                                           physx::PxU32 bodyIndex,
                                           AvbdContactConstraint *contacts,
                                           physx::PxU32 numContacts) {

  AvbdSolverBody &body = bodies[bodyIndex];

  if (body.invMass <= 0.0f) {
    return;
  }

  // Jacobi: accumulate all contact corrections, then apply once.
  // This avoids asymmetric bias from sequential application order.
  physx::PxVec3 contactDeltaPos(0.0f);
  physx::PxVec3 contactDeltaTheta(0.0f);

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;

    // Skip if this body is not involved
    if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
      continue;
    }

    // Get the other body
    AvbdSolverBody *otherBody = nullptr;
    bool isBodyA = (bodyAIdx == bodyIndex);

    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }

    // Compute world positions of contact points (using CURRENT body position
    // which may have been updated by previous constraints in this loop)
    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r; // Contact arm for this body

    if (isBodyA) {
      r = body.rotation.rotate(contacts[c].contactPointA);
      worldPosA = body.position + r;
      if (otherBody) {
        worldPosB = otherBody->position +
                    otherBody->rotation.rotate(contacts[c].contactPointB);
      } else {
        worldPosB = contacts[c].contactPointB;
      }
    } else {
      r = body.rotation.rotate(contacts[c].contactPointB);
      if (otherBody) {
        worldPosA = otherBody->position +
                    otherBody->rotation.rotate(contacts[c].contactPointA);
      } else {
        worldPosA = contacts[c].contactPointA;
      }
      worldPosB = body.position + r;
    }

    // Compute constraint violation C(x) (negative = penetration)
    const physx::PxVec3 &normal = contacts[c].contactNormal;
    physx::PxReal violation =
        (worldPosA - worldPosB).dot(normal) + contacts[c].penetrationDepth;

    // AL target: inner solve drives C(x) toward lambda/rho instead of 0.
    // This lets the accumulated Lagrange multiplier provide persistent contact
    // force.
    physx::PxReal lambdaOverRho =
        contacts[c].header.lambda / physx::PxMax(contacts[c].header.rho, 1e-6f);

    // Skip if constraint is inactive: not penetrating AND no accumulated lambda
    if (violation >= lambdaOverRho && contacts[c].header.lambda <= 0.0f) {
      continue;
    }

    // Compute generalized inverse mass for normal direction
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxReal w =
        body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);

    // Add other body's contribution if dynamic
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ? otherBody->rotation.rotate(contacts[c].contactPointB)
                       : otherBody->rotation.rotate(contacts[c].contactPointA);
      physx::PxVec3 rOtherCrossN = rOther.cross(normal);
      w += otherBody->invMass +
           rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
    }

    if (w <= 1e-6f) {
      continue;
    }

    // AL-augmented Baumgarte correction: drive violation toward lambda/rho
    // For contacts, clamp to non-negative to never pull bodies together
    physx::PxReal normalCorrectionMag = physx::PxMax(
        0.0f, -(violation - lambdaOverRho) / w * mConfig.baumgarte);

    // Direction sign for this body
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

    // Normal correction with scaled angular component for rotational stiffness
    physx::PxVec3 deltaPos =
        normal * (normalCorrectionMag * body.invMass * sign);
    physx::PxVec3 deltaTheta =
        (body.invInertiaWorld * rCrossN) *
        (normalCorrectionMag * sign * mConfig.angularContactScale);

    //=========================================================================
    // FRICTION CONSTRAINT (Position-based using previous position delta)
    //=========================================================================
    if (contacts[c].friction > 0.0f) {
      // Compute relative position displacement at contact point
      physx::PxVec3 prevWorldPosA, prevWorldPosB;

      if (isBodyA) {
        prevWorldPosA = body.prevPosition +
                        body.prevRotation.rotate(contacts[c].contactPointA);
        if (otherBody) {
          prevWorldPosB =
              otherBody->prevPosition +
              otherBody->prevRotation.rotate(contacts[c].contactPointB);
        } else {
          prevWorldPosB = contacts[c].contactPointB;
        }
      } else {
        if (otherBody) {
          prevWorldPosA =
              otherBody->prevPosition +
              otherBody->prevRotation.rotate(contacts[c].contactPointA);
        } else {
          prevWorldPosA = contacts[c].contactPointA;
        }
        prevWorldPosB = body.prevPosition +
                        body.prevRotation.rotate(contacts[c].contactPointB);
      }

      // Relative displacement
      physx::PxVec3 dispA = worldPosA - prevWorldPosA;
      physx::PxVec3 dispB = worldPosB - prevWorldPosB;
      physx::PxVec3 relDisp = isBodyA ? (dispA - dispB) : (dispB - dispA);

      // Extract tangential component
      physx::PxVec3 tangentDisp = relDisp - normal * relDisp.dot(normal);
      physx::PxReal tangentDispMag = tangentDisp.magnitude();

      if (tangentDispMag > 1e-6f) {
        physx::PxVec3 tangent = tangentDisp / tangentDispMag;

        physx::PxVec3 rCrossT = r.cross(tangent);
        physx::PxReal wT =
            body.invMass + rCrossT.dot(body.invInertiaWorld * rCrossT);

        if (otherBody && otherBody->invMass > 0.0f) {
          physx::PxVec3 rOtherCrossT = rOther.cross(tangent);
          wT += otherBody->invMass +
                rOtherCrossT.dot(otherBody->invInertiaWorld * rOtherCrossT);
        }

        if (wT > 1e-6f) {
          physx::PxReal maxFrictionCorrection =
              contacts[c].friction * physx::PxAbs(normalCorrectionMag);
          physx::PxReal frictionCorrection = tangentDispMag / wT;
          frictionCorrection =
              physx::PxMin(frictionCorrection, maxFrictionCorrection);

          deltaPos += -tangent * (frictionCorrection * body.invMass);
          deltaTheta += -(body.invInertiaWorld * rCrossT) * frictionCorrection;
        }
      }
    }

    // Accumulate correction (Jacobi within body)
    contactDeltaPos += deltaPos;
    contactDeltaTheta += deltaTheta;
  }

  // Apply accumulated contact corrections with SOR over-relaxation
  body.position += contactDeltaPos * mConfig.omega;

  physx::PxReal angle = contactDeltaTheta.magnitude() * mConfig.omega;
  if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    angle = physx::PxMin(angle, 0.1f);
    physx::PxVec3 axis = contactDeltaTheta.getNormalized();
    physx::PxReal halfAngle = angle * 0.5f;
    physx::PxQuat deltaQ(
        axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
        axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
    body.rotation = (deltaQ * body.rotation).getNormalized();
  }
}

/**
 * @brief Solve all constraints affecting a single body
 *
 * This is the core AVBD block descent step. For a given body, we:
 * 1. Collect all constraint violations and gradients
 * 2. Compute the optimal position/rotation update
 * 3. Apply the update
 */
void AvbdSolver::solveBodyAllConstraints(
    AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6, physx::PxReal dt) {

  PX_UNUSED(dt);

  AvbdSolverBody &body = bodies[bodyIndex];

  if (body.invMass <= 0.0f) {
    return;
  }

  // Accumulate corrections from all constraint types
  physx::PxVec3 totalDeltaPos(0.0f);
  physx::PxVec3 totalDeltaTheta(0.0f);
  physx::PxU32 numActiveConstraints = 0;

  // Process contact constraints using Jacobi accumulation within body.
  // All contacts compute based on same state, then sum applied once.
  {
    physx::PxVec3 contactDeltaPos(0.0f);
    physx::PxVec3 contactDeltaTheta(0.0f);

    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeContactCorrection(contacts[c], bodies, numBodies, bodyIndex,
                                   deltaPos, deltaTheta)) {
        contactDeltaPos += deltaPos;
        contactDeltaTheta += deltaTheta;
      }
    }

    // Apply accumulated contact corrections (Jacobi within body)
    body.position += contactDeltaPos;
    physx::PxReal angle = contactDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f);
      physx::PxVec3 axis = contactDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(
          axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
          axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }

  // Process spherical joint constraints
  for (physx::PxU32 j = 0; j < numSpherical; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeSphericalJointCorrection(sphericalJoints[j], bodies, numBodies,
                                        bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }

  // Process fixed joint constraints
  for (physx::PxU32 j = 0; j < numFixed; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeFixedJointCorrection(fixedJoints[j], bodies, numBodies,
                                    bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }

  // Process revolute joint constraints
  for (physx::PxU32 j = 0; j < numRevolute; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeRevoluteJointCorrection(revoluteJoints[j], bodies, numBodies,
                                       bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }

  // Process prismatic joint constraints
  for (physx::PxU32 j = 0; j < numPrismatic; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computePrismaticJointCorrection(prismaticJoints[j], bodies, numBodies,
                                        bodyIndex, deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }

  // Process D6 joint constraints
  for (physx::PxU32 j = 0; j < numD6; ++j) {
    physx::PxVec3 deltaPos, deltaTheta;
    if (computeD6JointCorrection(d6Joints[j], bodies, numBodies, bodyIndex,
                                 deltaPos, deltaTheta)) {
      totalDeltaPos += deltaPos;
      totalDeltaTheta += deltaTheta;
      numActiveConstraints++;
    }
  }

  // Apply averaged joint corrections (joints use weighted average, contacts
  // already applied via Gauss-Seidel above)
  if (numActiveConstraints > 0) {
    physx::PxReal invCount =
        1.0f / static_cast<physx::PxReal>(numActiveConstraints);

    // Apply position correction with baumgarte for joints
    body.position += totalDeltaPos * invCount * mConfig.baumgarte;

    // Apply rotation correction
    physx::PxVec3 avgDeltaTheta =
        totalDeltaTheta * invCount * mConfig.baumgarte;
    physx::PxReal angle = avgDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f);
      physx::PxVec3 axis = avgDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(
          axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
          axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }
}

/**
 * @brief Compute position/rotation correction for a contact constraint
 * @return true if constraint is active and correction was computed
 */
bool AvbdSolver::computeContactCorrection(const AvbdContactConstraint &contact,
                                          AvbdSolverBody *bodies,
                                          physx::PxU32 numBodies,
                                          physx::PxU32 bodyIndex,
                                          physx::PxVec3 &deltaPos,
                                          physx::PxVec3 &deltaTheta) {

  const physx::PxU32 bodyAIdx = contact.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = contact.header.bodyIndexB;

  // Skip if this body is not involved
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

  // Compute world positions and contact arms
  physx::PxVec3 worldPosA, worldPosB;
  physx::PxVec3 r; // Contact arm for this body

  if (isBodyA) {
    r = body.rotation.rotate(contact.contactPointA);
    worldPosA = body.position + r;
    worldPosB = otherBody
                    ? otherBody->position +
                          otherBody->rotation.rotate(contact.contactPointB)
                    : contact.contactPointB;
  } else {
    r = body.rotation.rotate(contact.contactPointB);
    worldPosA = otherBody
                    ? otherBody->position +
                          otherBody->rotation.rotate(contact.contactPointA)
                    : contact.contactPointA;
    worldPosB = body.position + r;
  }

  // Compute constraint violation C(x) (negative = penetration)
  const physx::PxVec3 &normal = contact.contactNormal;
  physx::PxReal violation =
      (worldPosA - worldPosB).dot(normal) + contact.penetrationDepth;

  // AL target: inner solve drives C(x) toward lambda/rho instead of 0.
  physx::PxReal lambdaOverRho =
      contact.header.lambda / physx::PxMax(contact.header.rho, 1e-6f);

  // Skip if constraint is inactive: not penetrating AND no accumulated lambda
  if (violation >= lambdaOverRho && contact.header.lambda <= 0.0f) {
    return false;
  }

  // Compute generalized inverse mass for normal
  physx::PxVec3 rCrossN = r.cross(normal);
  physx::PxReal w = body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);

  physx::PxVec3 rOther(0.0f);
  if (otherBody && otherBody->invMass > 0.0f) {
    rOther = isBodyA ? otherBody->rotation.rotate(contact.contactPointB)
                     : otherBody->rotation.rotate(contact.contactPointA);
    physx::PxVec3 rOtherCrossN = rOther.cross(normal);
    w += otherBody->invMass +
         rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
  }

  if (w <= 1e-6f) {
    return false;
  }

  // AL-augmented Baumgarte correction: drive violation toward lambda/rho
  // Clamp to non-negative to never pull bodies together for contacts
  physx::PxReal correctionMag =
      physx::PxMax(0.0f, -(violation - lambdaOverRho) / w * mConfig.baumgarte);

  physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

  deltaPos = normal * (correctionMag * body.invMass * sign);
  // Scaled angular correction from contact normal. Reduced scale
  // (angularContactScale) prevents drift from asymmetric contact patches while
  // maintaining rotational stiffness for dynamic stability (impact response,
  // settling to flat).
  deltaTheta = (body.invInertiaWorld * rCrossN) *
               (correctionMag * sign * mConfig.angularContactScale);

  //=========================================================================
  // FRICTION CONSTRAINT (Position-based)
  //=========================================================================
  if (contact.friction > 0.0f) {
    // Compute relative position displacement at contact point
    physx::PxVec3 prevWorldPosA, prevWorldPosB;

    if (isBodyA) {
      // Use previous position to compute displacement
      prevWorldPosA =
          body.prevPosition + body.prevRotation.rotate(contact.contactPointA);
      if (otherBody) {
        prevWorldPosB = otherBody->prevPosition +
                        otherBody->prevRotation.rotate(contact.contactPointB);
      } else {
        prevWorldPosB = contact.contactPointB;
      }
    } else {
      if (otherBody) {
        prevWorldPosA = otherBody->prevPosition +
                        otherBody->prevRotation.rotate(contact.contactPointA);
      } else {
        prevWorldPosA = contact.contactPointA;
      }
      prevWorldPosB =
          body.prevPosition + body.prevRotation.rotate(contact.contactPointB);
    }

    // Relative displacement from previous to current position
    physx::PxVec3 dispA = worldPosA - prevWorldPosA;
    physx::PxVec3 dispB = worldPosB - prevWorldPosB;
    physx::PxVec3 relDisp = isBodyA ? (dispA - dispB) : (dispB - dispA);

    // Extract tangential component (remove normal component)
    physx::PxVec3 tangentDisp = relDisp - normal * relDisp.dot(normal);
    physx::PxReal tangentDispMag = tangentDisp.magnitude();

    if (tangentDispMag > 1e-6f) {
      physx::PxVec3 tangent = tangentDisp / tangentDispMag;

      // Compute generalized inverse mass for tangent direction
      physx::PxVec3 rCrossT = r.cross(tangent);
      physx::PxReal wT =
          body.invMass + rCrossT.dot(body.invInertiaWorld * rCrossT);

      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOtherCrossT = rOther.cross(tangent);
        wT += otherBody->invMass +
              rOtherCrossT.dot(otherBody->invInertiaWorld * rOtherCrossT);
      }

      if (wT > 1e-6f) {
        // Maximum friction correction based on Coulomb friction model
        // Friction force magnitude <= mu * normal force magnitude
        physx::PxReal maxFrictionCorrection =
            contact.friction * physx::PxAbs(correctionMag);

        // Compute friction correction to oppose tangent displacement
        physx::PxReal frictionCorrection = tangentDispMag / wT;
        frictionCorrection =
            physx::PxMin(frictionCorrection, maxFrictionCorrection);

        // Apply friction correction (opposes tangent displacement direction)
        deltaPos += -tangent * (frictionCorrection * body.invMass);
        deltaTheta += -(body.invInertiaWorld * rCrossT) * frictionCorrection;
      }
    }
  }

  return true;
}

/**
 * @brief Compute position/rotation correction for a spherical joint
 */
bool AvbdSolver::computeSphericalJointCorrection(
    const AvbdSphericalJointConstraint &joint, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
    physx::PxVec3 &deltaTheta) {

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
    physx::PxVec3 direction = error / errorMag;

    // Compute generalized inverse mass
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
      // Compute correction
      physx::PxReal correctionMag = -errorMag / w;
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

      deltaPos = direction * (correctionMag * body.invMass * sign);
      deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
      hasCorrection = true;
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
      // Cone limit violated
      physx::PxReal violation = angle - joint.coneAngleLimit;

      // Correction axis is perpendicular to both
      physx::PxVec3 corrAxis = worldAxisA.cross(worldAxisB);
      physx::PxReal corrAxisMag = corrAxis.magnitude();

      if (corrAxisMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
        corrAxis /= corrAxisMag;

        // Compute angular correction
        physx::PxReal angularW = 0.0f;
        angularW += (body.invInertiaWorld * corrAxis).dot(corrAxis);
        if (otherBody && otherBody->invMass > 0.0f) {
          angularW += (otherBody->invInertiaWorld * corrAxis).dot(corrAxis);
        }

        if (angularW > 1e-6f) {
          physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
          physx::PxVec3 angCorrection =
              corrAxis * (violation * 0.5f / angularW * sign);
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
 */
bool AvbdSolver::computeFixedJointCorrection(
    const AvbdFixedJointConstraint &joint, AvbdSolverBody *bodies,
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

  // Rotation constraint for fixed joint - lock all 3 rotation axes
  physx::PxQuat rotA, rotB;
  if (isBodyA) {
    rotA = body.rotation;
    rotB = otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity);
  } else {
    rotA = otherBody ? otherBody->rotation : physx::PxQuat(physx::PxIdentity);
    rotB = body.rotation;
  }

  // Target: rotB = rotA * relativeRotation
  // Current relative rotation vs target
  physx::PxQuat targetRotB = rotA * joint.relativeRotation;
  physx::PxQuat errorQ = targetRotB.getConjugate() * rotB;
  if (errorQ.w < 0.0f) {
    errorQ = -errorQ; // Shortest path
  }

  // Convert to axis-angle in world space (small angle approximation)
  // errorQ represents the rotation needed to align rotB with target
  physx::PxVec3 rotErrorLocal(errorQ.x * 2.0f, errorQ.y * 2.0f,
                              errorQ.z * 2.0f);
  // Transform to world space
  physx::PxVec3 rotError = rotB.rotate(rotErrorLocal);
  physx::PxReal rotErrorMag = rotError.magnitude();

  if (rotErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    // Compute effective mass for rotation
    physx::PxVec3 axis = rotError / rotErrorMag;
    physx::PxReal w = axis.dot(body.invInertiaWorld * axis);

    if (otherBody && otherBody->invMass > 0.0f) {
      w += axis.dot(otherBody->invInertiaWorld * axis);
    }

    if (w > 1e-6f) {
      // Apply damped correction to avoid oscillation
      physx::PxReal compliance = 0.0001f; // Small compliance for stability
      physx::PxReal correctionMag = rotErrorMag / (w + compliance);
      correctionMag =
          physx::PxMin(correctionMag, rotErrorMag * 0.8f); // Limit correction

      // Body B should rotate towards target (negative correction)
      // Body A should rotate away (positive correction)
      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
      physx::PxVec3 deltaOmega = axis * (correctionMag * sign);
      deltaTheta += body.invInertiaWorld * deltaOmega / w;
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
  // already in world space) Note: bodyAIsStatic already defined above for debug
  // purposes
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
  // When velocity drive is active, we want the body to move, not be constrained
  if (joint.linearMotion == 0) {
    // Determine which axes have velocity drive (we'll skip position constraint
    // on those)
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

  // Linear velocity drive
  // PxD6Drive: eX=0, eY=1, eZ=2 -> bits 0,1,2
  // Only apply delta when processing body B (the driven body)
  if ((joint.driveFlags & 0x7) != 0 && body.invMass > 0.0f) {
    // Drive target velocity is in joint frame A's space
    // If body A is static, use localFrameA directly; otherwise transform it
    physx::PxQuat jointFrameA =
        bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);

    // Validate and normalize quaternion
    physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
    if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
      jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

      for (int axis = 0; axis < 3; ++axis) {
        if ((joint.driveFlags & (1 << axis)) == 0)
          continue;

        physx::PxReal targetVel = (&joint.driveLinearVelocity.x)[axis];
        physx::PxReal damping = (&joint.linearDamping.x)[axis];

        // Skip if no damping or invalid values
        if (damping <= 0.0f || !PxIsFinite(targetVel) || !PxIsFinite(damping))
          continue;

        // Get axis direction in world space
        physx::PxVec3 localAxis(0.0f);
        (&localAxis.x)[axis] = 1.0f;
        physx::PxVec3 worldAxis = jointFrameA.rotate(localAxis);

        // Validate world axis
        physx::PxReal axisMag2 = worldAxis.magnitudeSquared();
        if (axisMag2 < 0.9f || axisMag2 > 1.1f || !PxIsFinite(axisMag2))
          continue;

        // Current relative velocity along this axis
        // Body B should move in +axis direction for positive target velocity
        physx::PxVec3 velA =
            otherBody ? otherBody->linearVelocity : physx::PxVec3(0.0f);
        physx::PxVec3 velB = body.linearVelocity;
        physx::PxReal relVel = (velB - velA).dot(worldAxis);

        if (!PxIsFinite(relVel))
          continue;

        // Velocity error: positive error means we need to speed up in target
        // direction
        physx::PxReal velError = targetVel - relVel;

        if (physx::PxAbs(velError) > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
          // Position-based velocity drive: deltaX = targetVel * dt
          // Scale by damping factor and dt
          physx::PxReal dt = 1.0f / 60.0f; // Approximate timestep
          // Normalize damping: 1000 -> factor for reasonable speed
          physx::PxReal dampingFactor = physx::PxMin(damping / 30000.0f, 0.05f);

          // Direct position change based on target velocity
          // Positive targetVel -> body B moves in +worldAxis direction
          physx::PxVec3 posDelta = worldAxis * (targetVel * dt * dampingFactor);

          // Only apply to body B, not body A
          // When processing body A (isBodyA=true), skip applying the drive
          if (!isBodyA) {
            // Validate correction before applying
            if (PxIsFinite(posDelta.x) && PxIsFinite(posDelta.y) &&
                PxIsFinite(posDelta.z)) {
              // Clamp max correction per frame
              physx::PxReal maxCorrection = 0.005f;
              posDelta.x =
                  physx::PxClamp(posDelta.x, -maxCorrection, maxCorrection);
              posDelta.y =
                  physx::PxClamp(posDelta.y, -maxCorrection, maxCorrection);
              posDelta.z =
                  physx::PxClamp(posDelta.z, -maxCorrection, maxCorrection);
              deltaPos += posDelta;
              hasCorrection = true;
            }
          }
        }
      }
    }
  }

  // Angular velocity drive
  // PxD6Drive: eTWIST=4 (bit 4=0x10), eSLERP=5 (bit 5=0x20),
  //            eSWING1=6 (bit 6=0x40), eSWING2=7 (bit 7=0x80)
  // Combined angular mask: 0xF0 (bits 4-7)
  if ((joint.driveFlags & 0xF0) != 0) {
    // Drive target velocity is in joint frame A's space
    // If body A is static, use localFrameA directly; otherwise transform it
    physx::PxQuat jointFrameA =
        bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);

    // Validate and normalize quaternion
    physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
    if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
      jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

      // SLERP drive (PxD6Drive::eSLERP = 5, bit 5 = 0x20)
      bool slerpDrive = (joint.driveFlags & 0x20) != 0;

      physx::PxVec3 targetAngVel = joint.driveAngularVelocity;

      // Validate target angular velocity
      if (!PxIsFinite(targetAngVel.x) || !PxIsFinite(targetAngVel.y) ||
          !PxIsFinite(targetAngVel.z)) {
        targetAngVel = physx::PxVec3(0.0f);
      }

      // Transform target velocity from joint frame to world frame
      physx::PxVec3 worldTargetAngVel = jointFrameA.rotate(targetAngVel);

      physx::PxVec3 thetaDelta(0.0f);
      physx::PxReal maxDamping = 0.0f;
      physx::PxReal dt = 1.0f / 60.0f;
      // Normalize damping: 1000 -> factor for reasonable speed

      if (slerpDrive) {
        // SLERP applies to all angular axes
        // angularDamping.z = SLERP damping
        maxDamping = joint.angularDamping.z;
        if (maxDamping > 0.0f && PxIsFinite(maxDamping)) {
          physx::PxReal dampingFactor =
              physx::PxMin(maxDamping / 8000.0f, 0.15f);
          // Apply angular velocity in world space
          thetaDelta = worldTargetAngVel * (dt * dampingFactor);
        }
      } else {
        // Individual axis drives: TWIST (bit 4), SWING1 (bit 6), SWING2 (bit 7)
        // TWIST = X axis (PxD6Drive::eTWIST = 4), damping from angularDamping.x
        if (joint.driveFlags & 0x10) { // bit 4 = TWIST
          physx::PxReal damping = joint.angularDamping.x;
          if (damping > 0.0f && PxIsFinite(damping)) {
            maxDamping = physx::PxMax(maxDamping, damping);
            physx::PxVec3 worldAxisX =
                jointFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
            physx::PxReal dampingFactor =
                physx::PxMin(damping / 8000.0f, 0.15f);
            physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxisX);
            // Negate for correct direction
            thetaDelta -= worldAxisX * (targetOnAxis * dt * dampingFactor);
          }
        }

        // SWING1 = Y axis (PxD6Drive::eSWING1 = 6), damping from
        // angularDamping.y
        if (joint.driveFlags & 0x40) { // bit 6 = SWING1
          physx::PxReal damping = joint.angularDamping.y;
          if (damping > 0.0f && PxIsFinite(damping)) {
            maxDamping = physx::PxMax(maxDamping, damping);
            physx::PxVec3 worldAxisY =
                jointFrameA.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f));
            physx::PxReal dampingFactor =
                physx::PxMin(damping / 8000.0f, 0.15f);
            physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxisY);
            // Negate for correct direction
            thetaDelta -= worldAxisY * (targetOnAxis * dt * dampingFactor);
          }
        }

        // SWING2 = Z axis (PxD6Drive::eSWING2 = 7), damping from
        // angularDamping.z
        if (joint.driveFlags & 0x80) { // bit 7 = SWING2
          physx::PxReal damping = joint.angularDamping.z;
          if (damping > 0.0f && PxIsFinite(damping)) {
            maxDamping = physx::PxMax(maxDamping, damping);
            physx::PxVec3 worldAxisZ =
                jointFrameA.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f));
            physx::PxReal dampingFactor =
                physx::PxMin(damping / 8000.0f, 0.15f);
            physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxisZ);
            // Negate for correct direction
            thetaDelta -= worldAxisZ * (targetOnAxis * dt * dampingFactor);
          }
        }
      }

      physx::PxReal thetaMag2 = thetaDelta.magnitudeSquared();
      if (thetaMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON &&
          PxIsFinite(thetaMag2)) {
        // Only apply to body B, not body A
        if (!isBodyA) {
          // Validate correction before applying
          if (PxIsFinite(thetaDelta.x) && PxIsFinite(thetaDelta.y) &&
              PxIsFinite(thetaDelta.z)) {
            // Clamp max angular correction
            physx::PxReal maxAngCorrection = 0.01f;
            thetaDelta.x = physx::PxClamp(thetaDelta.x, -maxAngCorrection,
                                          maxAngCorrection);
            thetaDelta.y = physx::PxClamp(thetaDelta.y, -maxAngCorrection,
                                          maxAngCorrection);
            thetaDelta.z = physx::PxClamp(thetaDelta.z, -maxAngCorrection,
                                          maxAngCorrection);
            deltaTheta += thetaDelta;
            hasCorrection = true;
          }
        }
      }
    }
  }

  return hasCorrection;
}

/**
 * @brief Compute correction for gear joint
 *
 * The gear joint constrains the angular velocities of two bodies around their
 * respective axes such that they maintain a fixed ratio:
 *   omega0 * gearRatio + omega1 = 0
 *
 * In position-based form, we constrain the angular displacement:
 *   deltaTheta0 * gearRatio + deltaTheta1 = 0
 *
 * Where deltaTheta is the change in angle from prevRotation to current
 * rotation.
 */
bool AvbdSolver::computeGearJointCorrection(
    const AvbdGearJointConstraint &joint, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
    physx::PxVec3 &deltaTheta) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

  // Check if this body is involved in the constraint
  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }

  // Check for valid body indices - both must be dynamic for gear to work
  if (bodyAIdx >= numBodies || bodyBIdx >= numBodies) {
    return false; // Gear joint needs both bodies to be dynamic
  }

  AvbdSolverBody &bodyA = bodies[bodyAIdx];
  AvbdSolverBody &bodyB = bodies[bodyBIdx];

  // Skip if either body is static
  if (bodyA.invMass <= 0.0f || bodyB.invMass <= 0.0f) {
    return false;
  }

  bool isBodyA = (bodyAIdx == bodyIndex);

  // Initialize outputs
  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);

  // Use joint.gearAxis0/1 as local axes, transform to world space using CURRENT
  // rotation
  physx::PxVec3 worldAxis0 = bodyA.rotation.rotate(joint.gearAxis0);
  physx::PxVec3 worldAxis1 = bodyB.rotation.rotate(joint.gearAxis1);

  // Normalize axes
  physx::PxReal axis0Len = worldAxis0.magnitude();
  physx::PxReal axis1Len = worldAxis1.magnitude();
  if (axis0Len < AvbdConstants::AVBD_NUMERICAL_EPSILON ||
      axis1Len < AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    return false;
  }
  worldAxis0 /= axis0Len;
  worldAxis1 /= axis1Len;

  // Compute angle changes since start of frame (prevRotation -> rotation)
  // deltaQ = rotation * prevRotation.conjugate()
  physx::PxQuat deltaQA = bodyA.rotation * bodyA.prevRotation.getConjugate();
  physx::PxQuat deltaQB = bodyB.rotation * bodyB.prevRotation.getConjugate();

  // Ensure shortest path
  if (deltaQA.w < 0.0f)
    deltaQA = -deltaQA;
  if (deltaQB.w < 0.0f)
    deltaQB = -deltaQB;

  // Extract angle around the gear axis
  // Use local axis for extraction since we want rotation around the hinge axis
  physx::PxVec3 localAxis0 = joint.gearAxis0;
  physx::PxVec3 localAxis1 = joint.gearAxis1;
  localAxis0.normalize();
  localAxis1.normalize();

  // Convert deltaQ to axis-angle and project onto gear axis
  physx::PxVec3 deltaAngA(deltaQA.x, deltaQA.y, deltaQA.z);
  physx::PxVec3 deltaAngB(deltaQB.x, deltaQB.y, deltaQB.z);

  // For small angles: angle ~= 2 * |imaginary part|, direction = imaginary part
  // / |imaginary part| So angular displacement vector ~= 2 * imaginary part
  deltaAngA *= 2.0f;
  deltaAngB *= 2.0f;

  // Transform to local space to get rotation around gear axis
  physx::PxVec3 localDeltaAngA =
      bodyA.prevRotation.getConjugate().rotate(deltaAngA);
  physx::PxVec3 localDeltaAngB =
      bodyB.prevRotation.getConjugate().rotate(deltaAngB);

  physx::PxReal thetaA =
      localDeltaAngA.dot(localAxis0); // Rotation of body A around its gear axis
  physx::PxReal thetaB =
      localDeltaAngB.dot(localAxis1); // Rotation of body B around its gear axis

  // Gear constraint: thetaA * gearRatio - thetaB = 0
  // This means: if A rotates by angle theta, B should rotate by theta *
  // gearRatio (OPPOSITE direction) The minus sign ensures opposite rotation
  // direction (gears mesh)
  physx::PxReal violation =
      thetaA * joint.gearRatio - thetaB + joint.geometricError;

  // Skip if violation is negligible
  if (physx::PxAbs(violation) < 1e-6f) {
    return false;
  }

  // Compute effective inertia for angular constraint along axis
  physx::PxVec3 Iinv_axis0 = bodyA.invInertiaWorld * worldAxis0;
  physx::PxVec3 Iinv_axis1 = bodyB.invInertiaWorld * worldAxis1;
  physx::PxReal w0 = worldAxis0.dot(Iinv_axis0);
  physx::PxReal w1 = worldAxis1.dot(Iinv_axis1);

  // Effective inverse inertia for gear constraint:
  // w_eff = gearRatio^2 * w0 + w1
  physx::PxReal wEff = joint.gearRatio * joint.gearRatio * w0 + w1;

  if (wEff < 1e-10f) {
    return false;
  }

  // Compute Lagrange multiplier (correction impulse magnitude)
  physx::PxReal lambda = -violation / wEff;

  // Apply correction to this body
  // Constraint: C = thetaA * gearRatio - thetaB = 0
  // Gradient for A: dC/dthetaA = +gearRatio
  // Gradient for B: dC/dthetaB = -1
  if (isBodyA) {
    // Body A gets impulse along axis0, scaled by gearRatio
    physx::PxReal impulseMag = joint.gearRatio * lambda;
    deltaTheta = (bodyA.invInertiaWorld * worldAxis0) * impulseMag;
  } else {
    // Body B gets impulse along axis1, with NEGATIVE sign (opposite direction)
    physx::PxReal impulseMag = -lambda; // Note: negative because gradient is -1
    deltaTheta = (bodyB.invInertiaWorld * worldAxis1) * impulseMag;
  }

  // Clamp the correction to prevent instability
  physx::PxReal maxAngCorrection = 0.1f;
  physx::PxReal thetaMag = deltaTheta.magnitude();
  if (thetaMag > maxAngCorrection) {
    deltaTheta *= maxAngCorrection / thetaMag;
  }

  return true;
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

  // Debug: Track solver entry
  if (!mInitialized || numBodies == 0) {
    return;
  }

  mStats.reset();
  mStats.numBodies = numBodies;
  mStats.numContacts = numContacts;
  mStats.numJoints =
      numSpherical + numFixed + numRevolute + numPrismatic + numD6 + numGear;

  physx::PxReal invDt = 1.0f / dt;

  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  // Adaptive position warmstarting (same as solve())
  const physx::PxReal gravMag = gravity.magnitude();
  const physx::PxVec3 gravDir =
      (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].prevPosition = bodies[i].position;
    bodies[i].prevRotation = bodies[i].rotation;

    if (bodies[i].invMass > 0.0f) {
      // Acceleration-based accelWeight (ref: AVBD3D solver.cpp L88-90)
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

  // Empty mappings for fallback when nullptr is passed
  static const AvbdBodyConstraintMap emptyMap;

  // =========================================================================
  // Penalty floor: ensure penalty >= 0.25 * effectiveMass / h^2
  // (same as solve() path)
  // =========================================================================
  if (contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.penaltyFloor", 0);
    const physx::PxReal invDt2 = 1.0f / (dt * dt);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      physx::PxReal massA = 0.0f, massB = 0.0f;
      if (bA < numBodies && bodies[bA].invMass > 0.0f)
        massA = 1.0f / bodies[bA].invMass;
      if (bB < numBodies && bodies[bB].invMass > 0.0f)
        massB = 1.0f / bodies[bB].invMass;
      physx::PxReal effectiveMass =
          (massA > 0.0f && massB > 0.0f)
              ? (massA * massB) / (massA + massB)
              : physx::PxMax(massA, massB);
      const physx::PxReal penaltyFloor = 0.25f * effectiveMass * invDt2;
      if (contacts[c].header.penalty < penaltyFloor)
        contacts[c].header.penalty = penaltyFloor;
      if (contacts[c].tangentPenalty0 < penaltyFloor)
        contacts[c].tangentPenalty0 = penaltyFloor;
      if (contacts[c].tangentPenalty1 < penaltyFloor)
        contacts[c].tangentPenalty1 = penaltyFloor;
    }
  }

  // =========================================================================
  // Compute C0 for alpha blending at old (pre-warmstart) positions
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

  // =========================================================================
  // Main iteration loop: primal + dual per iteration (unified AL)
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.solveIterations", 0);
    const physx::PxReal invDt2 = 1.0f / (dt * dt);

    for (physx::PxU32 iter = 0; iter < mConfig.innerIterations; ++iter) {
      // --- Primal step ---
      {
        PX_PROFILE_ZONE("AVBD.blockDescentWithJoints", 0);

        for (physx::PxU32 bodyIdx = 0; bodyIdx < numBodies; ++bodyIdx) {
          if (bodies[bodyIdx].invMass <= 0.0f) {
            continue;
          }

          // (A) Contact constraints: 6x6 or 3x3 system solve
          // Only call for bodies that actually have contacts; otherwise
          // the numTouching==0 path snaps body to inertialPosition,
          // erasing joint corrections from previous iterations.
          if (contacts && numContacts > 0 && contactMap) {
            const physx::PxU32 *cIdx = nullptr;
            physx::PxU32 cCnt = 0;
            contactMap->getBodyConstraints(bodyIdx, cIdx, cCnt);
            if (cCnt > 0) {
              if (mConfig.enableLocal6x6Solve) {
                solveLocalSystem(bodies[bodyIdx], bodies, numBodies, contacts,
                                 numContacts, dt, invDt2, contactMap);
              } else {
                solveLocalSystem3x3(bodies[bodyIdx], bodies, numBodies, contacts,
                                    numContacts, dt, invDt2, contactMap);
              }
            }
          } else if (contacts && numContacts > 0) {
            // No contactMap: fall back but still call (O(N) scan will skip
            // bodies with no contacts via the bodyIndex mismatch check)
            if (mConfig.enableLocal6x6Solve) {
              solveLocalSystem(bodies[bodyIdx], bodies, numBodies, contacts,
                               numContacts, dt, invDt2, nullptr);
            } else {
              solveLocalSystem3x3(bodies[bodyIdx], bodies, numBodies, contacts,
                                  numContacts, dt, invDt2, nullptr);
            }
          }

          // (B) Joint constraints: Jacobi corrections (using maps for O(K))
          physx::PxVec3 totalDeltaPos(0.0f);
          physx::PxVec3 totalDeltaTheta(0.0f);
          physx::PxU32 numActiveJoints = 0;

          // Spherical joints
          if (sphericalJoints && numSpherical > 0) {
            const physx::PxU32 *indices = nullptr;
            physx::PxU32 cnt = 0;
            (sphericalMap ? *sphericalMap : emptyMap)
                .getBodyConstraints(bodyIdx, indices, cnt);
            for (physx::PxU32 i = 0; i < cnt; ++i) {
              if (indices[i] >= numSpherical) continue;
              physx::PxVec3 dp, dth;
              if (computeSphericalJointCorrection(sphericalJoints[indices[i]],
                      bodies, numBodies, bodyIdx, dp, dth)) {
                totalDeltaPos += dp; totalDeltaTheta += dth;
                numActiveJoints++;
              }
            }
          }

          // Fixed joints
          if (fixedJoints && numFixed > 0) {
            const physx::PxU32 *indices = nullptr;
            physx::PxU32 cnt = 0;
            (fixedMap ? *fixedMap : emptyMap)
                .getBodyConstraints(bodyIdx, indices, cnt);
            for (physx::PxU32 i = 0; i < cnt; ++i) {
              if (indices[i] >= numFixed) continue;
              physx::PxVec3 dp, dth;
              if (computeFixedJointCorrection(fixedJoints[indices[i]], bodies,
                      numBodies, bodyIdx, dp, dth)) {
                totalDeltaPos += dp; totalDeltaTheta += dth;
                numActiveJoints++;
              }
            }
          }

          // Revolute joints
          if (revoluteJoints && numRevolute > 0) {
            const physx::PxU32 *indices = nullptr;
            physx::PxU32 cnt = 0;
            (revoluteMap ? *revoluteMap : emptyMap)
                .getBodyConstraints(bodyIdx, indices, cnt);
            for (physx::PxU32 i = 0; i < cnt; ++i) {
              if (indices[i] >= numRevolute) continue;
              physx::PxVec3 dp, dth;
              if (computeRevoluteJointCorrection(revoluteJoints[indices[i]],
                      bodies, numBodies, bodyIdx, dp, dth)) {
                totalDeltaPos += dp; totalDeltaTheta += dth;
                numActiveJoints++;
              }
            }
          }

          // Prismatic joints
          if (prismaticJoints && numPrismatic > 0) {
            const physx::PxU32 *indices = nullptr;
            physx::PxU32 cnt = 0;
            (prismaticMap ? *prismaticMap : emptyMap)
                .getBodyConstraints(bodyIdx, indices, cnt);
            for (physx::PxU32 i = 0; i < cnt; ++i) {
              if (indices[i] >= numPrismatic) continue;
              physx::PxVec3 dp, dth;
              if (computePrismaticJointCorrection(prismaticJoints[indices[i]],
                      bodies, numBodies, bodyIdx, dp, dth)) {
                totalDeltaPos += dp; totalDeltaTheta += dth;
                numActiveJoints++;
              }
            }
          }

          // D6 joints
          if (d6Joints && numD6 > 0) {
            const physx::PxU32 *indices = nullptr;
            physx::PxU32 cnt = 0;
            (d6Map ? *d6Map : emptyMap)
                .getBodyConstraints(bodyIdx, indices, cnt);
            for (physx::PxU32 i = 0; i < cnt; ++i) {
              if (indices[i] >= numD6) continue;
              physx::PxVec3 dp, dth;
              if (computeD6JointCorrection(d6Joints[indices[i]], bodies,
                      numBodies, bodyIdx, dp, dth)) {
                totalDeltaPos += dp; totalDeltaTheta += dth;
                numActiveJoints++;
              }
            }
          }

          // Gear joints
          if (gearJoints && numGear > 0) {
            const physx::PxU32 *indices = nullptr;
            physx::PxU32 cnt = 0;
            (gearMap ? *gearMap : emptyMap)
                .getBodyConstraints(bodyIdx, indices, cnt);
            for (physx::PxU32 i = 0; i < cnt; ++i) {
              if (indices[i] >= numGear) continue;
              physx::PxVec3 dp, dth;
              if (computeGearJointCorrection(gearJoints[indices[i]], bodies,
                      numBodies, bodyIdx, dp, dth)) {
                totalDeltaPos += dp; totalDeltaTheta += dth;
                numActiveJoints++;
              }
            }
          }

          // Apply accumulated joint corrections
          if (numActiveJoints > 0) {
            physx::PxReal invCount =
                1.0f / static_cast<physx::PxReal>(numActiveJoints);
            AvbdSolverBody &body = bodies[bodyIdx];
            body.position += totalDeltaPos * invCount * mConfig.baumgarte;
            physx::PxVec3 avgDT = totalDeltaTheta * invCount * mConfig.baumgarte;
            physx::PxReal angle = avgDT.magnitude();
            if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
              angle = physx::PxMin(angle, 0.1f);
              physx::PxVec3 axis = avgDT.getNormalized();
              physx::PxReal ha = angle * 0.5f;
              physx::PxQuat dq(axis.x * physx::PxSin(ha),
                               axis.y * physx::PxSin(ha),
                               axis.z * physx::PxSin(ha), physx::PxCos(ha));
              body.rotation = (dq * body.rotation).getNormalized();
            }
          }
        }
        mStats.totalIterations++;
      }

      // --- Dual step: AL update for contacts + joint multipliers ---
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt);
        for (physx::PxU32 j = 0; j < numSpherical; ++j)
          updateSphericalJointMultiplier(sphericalJoints[j], bodies, numBodies,
                                         mConfig);
        for (physx::PxU32 j = 0; j < numFixed; ++j)
          updateFixedJointMultiplier(fixedJoints[j], bodies, numBodies,
                                     mConfig);
        for (physx::PxU32 j = 0; j < numRevolute; ++j)
          updateRevoluteJointMultiplier(revoluteJoints[j], bodies, numBodies,
                                        mConfig);
        for (physx::PxU32 j = 0; j < numPrismatic; ++j)
          updatePrismaticJointMultiplier(prismaticJoints[j], bodies, numBodies,
                                         mConfig);
        for (physx::PxU32 j = 0; j < numD6; ++j)
          updateD6JointMultiplier(d6Joints[j], bodies, numBodies, mConfig);
        for (physx::PxU32 j = 0; j < numGear; ++j)
          updateGearJointMultiplier(gearJoints[j], bodies, numBodies, mConfig);
      }
    }
  }

  // Process motor drives for RevoluteJoints AFTER all constraint iterations
  // Motor applies torque (limited by maxForce) to accelerate toward target
  // velocity This matches TGS behavior where motor gradually accelerates bodies
  //
  // IMPORTANT: The solver may be called multiple times per simulation step
  // (once per island). We use a global frame counter to track which frame
  // we're in and only apply motor once per frame per body.
  {
    PX_PROFILE_ZONE("AVBD.motorDrives", 0);

    // Get current frame from global counter (incremented in
    // AvbdDynamicsContext::update)
    physx::PxU64 currentFrame = getAvbdMotorFrameCounter();

    // Static tracking for frame-based deduplication
    static physx::PxU64 lastMotorFrame = 0;
    static physx::PxU32 processedBodyFlags = 0; // Bitmask for up to 32 bodies

    // Reset flags when entering a new frame
    if (currentFrame != lastMotorFrame) {
      lastMotorFrame = currentFrame;
      processedBodyFlags = 0;
    }

    for (physx::PxU32 j = 0; j < numRevolute; ++j) {
      AvbdRevoluteJointConstraint &joint = revoluteJoints[j];
      if (joint.motorEnabled && joint.motorMaxForce > 0.0f) {
        const physx::PxU32 idxA = joint.header.bodyIndexA;
        const physx::PxU32 idxB = joint.header.bodyIndexB;

        const bool isAStatic = (idxA == 0xFFFFFFFF || idxA >= numBodies);
        const bool isBStatic = (idxB == 0xFFFFFFFF || idxB >= numBodies);

        if (isAStatic && isBStatic)
          continue;

        // Check if this body already had motor applied this frame
        if (idxB < 32 && (processedBodyFlags & (1u << idxB))) {
          continue; // Skip - already processed this frame
        }
        if (idxB < 32) {
          processedBodyFlags |= (1u << idxB); // Mark as processed
        }

        AvbdSolverBody &bodyB = bodies[idxB];

        // Get world-space joint axis
        physx::PxVec3 worldAxis =
            isAStatic ? joint.axisA : bodies[idxA].rotation.rotate(joint.axisA);
        worldAxis.normalize();

        // Compute current angular velocity around the joint axis
        // Note: In AVBD, velocity is computed from position difference
        // (prevPosition/rotation) We need to get the actual angular velocity
        // from the body
        physx::PxQuat deltaQ =
            bodyB.rotation * bodyB.prevRotation.getConjugate();
        if (deltaQ.w < 0.0f)
          deltaQ = -deltaQ;
        physx::PxVec3 currentAngVel =
            physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);

        // Project to joint axis to get current rotation speed around axis
        physx::PxReal currentAxisVel = currentAngVel.dot(worldAxis);

        // Velocity error
        physx::PxReal velocityError =
            joint.motorTargetVelocity - currentAxisVel;

        // Compute effective inertia around the joint axis
        // For a rotation around axis 'n', effective inertia = n^T * I * n
        // Since we have invInertiaWorld, effective invInertia = n^T * invI * n
        physx::PxVec3 invITimesAxis = bodyB.invInertiaWorld * worldAxis;
        physx::PxReal effectiveInvInertia = worldAxis.dot(invITimesAxis);

        if (effectiveInvInertia < 1e-10f)
          continue; // Body is static around this axis

        physx::PxReal effectiveInertia = 1.0f / effectiveInvInertia;

        // Required torque to achieve target velocity change
        // torque = inertia * angular_acceleration = inertia * (deltaVel / dt)
        physx::PxReal requiredTorque = effectiveInertia * velocityError * invDt;

        // Clamp torque to maxForce (which is actually max torque for revolute
        // joints)
        physx::PxReal clampedTorque = physx::PxClamp(
            requiredTorque, -joint.motorMaxForce, joint.motorMaxForce);

        // Angular acceleration from clamped torque
        physx::PxReal angularAccel = clampedTorque * effectiveInvInertia;

        // Delta angle = 0.5 * alpha * dt^2 (integration from torque to
        // position) But since we already have velocity changes, use: deltaAngle
        // = deltaVel * dt
        physx::PxReal deltaVel = angularAccel * dt;
        physx::PxReal deltaAngle = deltaVel * dt;

        // Apply rotation around the joint axis
        physx::PxReal halfAngle = deltaAngle * 0.5f;
        physx::PxReal sinHalf = physx::PxSin(halfAngle);
        physx::PxReal cosHalf = physx::PxCos(halfAngle);
        physx::PxQuat deltaRot(worldAxis.x * sinHalf, worldAxis.y * sinHalf,
                               worldAxis.z * sinHalf, cosHalf);

        // Apply rotation to body B
        bodyB.rotation = (deltaRot * bodyB.rotation).getNormalized();

        // If bodyA is also dynamic, apply opposite rotation (scaled by inertia
        // ratio)
        if (!isAStatic) {
          AvbdSolverBody &bodyA = bodies[idxA];
          physx::PxVec3 invITimesAxisA = bodyA.invInertiaWorld * worldAxis;
          physx::PxReal effectiveInvInertiaA = worldAxis.dot(invITimesAxisA);

          if (effectiveInvInertiaA > 1e-10f) {
            physx::PxReal angularAccelA = clampedTorque * effectiveInvInertiaA;
            physx::PxReal deltaAngleA = angularAccelA * dt * dt;

            physx::PxReal halfAngleA =
                -deltaAngleA * 0.5f; // Opposite direction
            physx::PxReal sinHalfA = physx::PxSin(halfAngleA);
            physx::PxReal cosHalfA = physx::PxCos(halfAngleA);
            physx::PxQuat deltaRotA(worldAxis.x * sinHalfA,
                                    worldAxis.y * sinHalfA,
                                    worldAxis.z * sinHalfA, cosHalfA);

            bodyA.rotation = (deltaRotA * bodyA.rotation).getNormalized();
          }
        }
      }
    }
  }

  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        // Save current velocity for next frame's adaptive warmstart
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;

        bodies[i].linearVelocity =
            (bodies[i].position - bodies[i].prevPosition) * invDt;
        bodies[i].linearVelocity *= mConfig.velocityDamping;
        // Pseudo-sleep: clamp very small velocities to zero to prevent
        // micro-drift from solver asymmetry and post-fall jitter
        if (bodies[i].linearVelocity.magnitudeSquared() < 1e-3f) { // ~0.032 m/s
          bodies[i].linearVelocity = physx::PxVec3(0.0f);
        }
        physx::PxQuat deltaQ =
            bodies[i].rotation * bodies[i].prevRotation.getConjugate();
        if (deltaQ.w < 0.0f) {
          deltaQ = -deltaQ;
        }
        bodies[i].angularVelocity =
            physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
        bodies[i].angularVelocity *= mConfig.angularDamping;
        if (bodies[i].angularVelocity.magnitudeSquared() <
            1e-4f) { // ~0.01 rad/s
          bodies[i].angularVelocity = physx::PxVec3(0.0f);
        }
      }
    }

    // Apply D6 joint angular damping to connected bodies
    for (physx::PxU32 j = 0; j < numD6; ++j) {
      const AvbdD6JointConstraint &joint = d6Joints[j];
      physx::PxReal maxDamping = physx::PxMax(
          joint.angularDamping.x,
          physx::PxMax(joint.angularDamping.y, joint.angularDamping.z));
      if (maxDamping > 0.0f && joint.driveFlags != 0) {
        // Compute damping factor (normalize to reasonable range)
        // For damping=1000, we want strong damping ~0.9
        physx::PxReal dampingFactor = 1.0f / (1.0f + maxDamping * dt * 0.1f);
        dampingFactor =
            physx::PxMax(dampingFactor, 0.1f); // Minimum 10% velocity retained

        // Apply damping to body A
        if (joint.header.bodyIndexA < numBodies &&
            bodies[joint.header.bodyIndexA].invMass > 0.0f) {
          bodies[joint.header.bodyIndexA].angularVelocity *= dampingFactor;
        }
        // Apply damping to body B
        if (joint.header.bodyIndexB < numBodies &&
            bodies[joint.header.bodyIndexB].invMass > 0.0f) {
          bodies[joint.header.bodyIndexB].angularVelocity *= dampingFactor;
        }
      }
    }
  }
}

//=============================================================================
// Energy Minimization Framework
//=============================================================================

physx::PxReal AvbdSolver::computeTotalEnergy(AvbdSolverBody *bodies,
                                             physx::PxU32 numBodies,
                                             AvbdContactConstraint *contacts,
                                             physx::PxU32 numContacts,
                                             const physx::PxVec3 &gravity) {
  physx::PxReal kineticEnergy = computeKineticEnergy(bodies, numBodies);
  physx::PxReal potentialEnergy =
      computePotentialEnergy(bodies, numBodies, gravity);
  physx::PxReal constraintEnergy =
      computeConstraintEnergy(contacts, numContacts, bodies, numBodies);
  return kineticEnergy + potentialEnergy + constraintEnergy;
}

physx::PxReal AvbdSolver::computeKineticEnergy(AvbdSolverBody *bodies,
                                               physx::PxU32 numBodies) {
  physx::PxReal totalEnergy = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;
    physx::PxReal mass = 1.0f / bodies[i].invMass;
    physx::PxReal linearKE =
        0.5f * mass * bodies[i].linearVelocity.magnitudeSquared();
    physx::PxVec3 angVel = bodies[i].angularVelocity;
    physx::PxVec3 angMomentum =
        bodies[i].invInertiaWorld.transformTranspose(angVel);
    physx::PxReal angularKE = 0.5f * angVel.dot(angMomentum);
    totalEnergy += linearKE + angularKE;
  }
  return totalEnergy;
}

physx::PxReal AvbdSolver::computePotentialEnergy(AvbdSolverBody *bodies,
                                                 physx::PxU32 numBodies,
                                                 const physx::PxVec3 &gravity) {
  physx::PxReal totalEnergy = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;
    physx::PxReal mass = 1.0f / bodies[i].invMass;
    physx::PxReal height = bodies[i].position.dot(gravity.getNormalized());
    totalEnergy -= mass * gravity.magnitude() * height;
  }
  return totalEnergy;
}

physx::PxReal AvbdSolver::computeConstraintEnergy(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSolverBody *bodies, physx::PxU32 numBodies) {
  physx::PxReal totalEnergy = 0.0f;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;
    physx::PxReal violation = 0.0f;
    // Compute violation correctly handling static bodies (contactPoint is world
    // coords for static)
    physx::PxVec3 worldPointA, worldPointB;
    if (bodyAIdx < numBodies) {
      worldPointA = bodies[bodyAIdx].position +
                    bodies[bodyAIdx].rotation.rotate(contacts[c].contactPointA);
    } else {
      worldPointA = contacts[c].contactPointA; // Static: already world coords
    }
    if (bodyBIdx < numBodies) {
      worldPointB = bodies[bodyBIdx].position +
                    bodies[bodyBIdx].rotation.rotate(contacts[c].contactPointB);
    } else {
      worldPointB = contacts[c].contactPointB; // Static: already world coords
    }
    violation = (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
                contacts[c].penetrationDepth;
    if (violation >= 0.0f && contacts[c].header.lambda <= 0.0f)
      continue;
    physx::PxReal rho = contacts[c].header.rho;
    physx::PxReal lambda = contacts[c].header.lambda;
    totalEnergy += 0.5f * rho * violation * violation + lambda * violation;
  }
  return totalEnergy;
}

void AvbdSolver::computeEnergyGradient(
    physx::PxU32 bodyIndex, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal invDt2, AvbdVec6 &gradient) {
  AvbdSolverBody &body = bodies[bodyIndex];
  physx::PxReal massContrib =
      (body.invMass > 0.0f) ? (1.0f / body.invMass) * invDt2 : 0.0f;
  gradient.linear = (body.position - body.inertialPosition) * massContrib;
  gradient.angular = physx::PxVec3(0.0f);
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
    if (bodyA != bodyIndex && bodyB != bodyIndex)
      continue;
    // Compute constraint violation correctly handling static bodies
    // For static bodies, contactPoint is already in world coords
    physx::PxVec3 worldPointA, worldPointB;
    if (bodyA < numBodies) {
      worldPointA = bodies[bodyA].position +
                    bodies[bodyA].rotation.rotate(contacts[c].contactPointA);
    } else {
      worldPointA = contacts[c].contactPointA; // Static: already world coords
    }
    if (bodyB < numBodies) {
      worldPointB = bodies[bodyB].position +
                    bodies[bodyB].rotation.rotate(contacts[c].contactPointB);
    } else {
      worldPointB = contacts[c].contactPointB; // Static: already world coords
    }
    physx::PxReal violation =
        (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
        contacts[c].penetrationDepth;
    // Clamp force for unilateral contacts (repulsive only), with adaptive penalty
    physx::PxReal force =
        physx::PxMin(0.0f, contacts[c].header.penalty * violation + contacts[c].header.lambda);
    if (force >= 0.0f)
      continue;
    physx::PxVec3 gradPos, gradRot;
    if (bodyA == bodyIndex) {
      gradPos = contacts[c].contactNormal;
      physx::PxVec3 rA = body.rotation.rotate(contacts[c].contactPointA);
      gradRot = rA.cross(contacts[c].contactNormal);
    } else {
      gradPos = -contacts[c].contactNormal;
      physx::PxVec3 rB = body.rotation.rotate(contacts[c].contactPointB);
      gradRot = -rB.cross(contacts[c].contactNormal);
    }
    gradient.linear += gradPos * force;
    gradient.angular += gradRot * force;
  }
}

bool AvbdSolver::checkEnergyConvergence(physx::PxReal oldEnergy,
                                        physx::PxReal newEnergy,
                                        physx::PxReal tolerance) const {
  physx::PxReal energyChange = physx::PxAbs(newEnergy - oldEnergy);
  physx::PxReal relativeChange =
      (physx::PxAbs(oldEnergy) > AvbdConstants::AVBD_LDLT_SINGULAR_THRESHOLD)
          ? energyChange / physx::PxAbs(oldEnergy)
          : energyChange;
  return relativeChange < tolerance;
}

physx::PxReal
AvbdSolver::performLineSearch(AvbdSolverBody &body, const AvbdVec6 &direction,
                              physx::PxReal initialStep, physx::PxReal energy,
                              physx::PxReal c1, physx::PxReal rho) {
  PX_UNUSED(energy);
  physx::PxReal alpha = initialStep;
  physx::PxReal maxIter = 10;
  physx::PxVec3 savedPosition = body.position;
  physx::PxQuat savedRotation = body.rotation;
  for (physx::PxU32 i = 0; i < maxIter; ++i) {
    body.position += direction.linear * alpha;
    physx::PxReal angle = (direction.angular * alpha).magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      physx::PxVec3 axis = direction.angular / angle;
      physx::PxReal sinHalf = physx::PxSin(angle * 0.5f);
      physx::PxReal cosHalf = physx::PxCos(angle * 0.5f);
      physx::PxQuat deltaQ(axis.x * sinHalf, axis.y * sinHalf, axis.z * sinHalf,
                           cosHalf);
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
    physx::PxReal positionChange = (body.position - savedPosition).magnitude();
    if (positionChange < initialStep * c1) {
      return alpha;
    }
    alpha *= rho;
    body.position = savedPosition;
    body.rotation = savedRotation;
  }
  body.position = savedPosition;
  body.rotation = savedRotation;
  return alpha;
}

//=============================================================================
// Optimized Constraint Mapping and Solving
//=============================================================================

void AvbdSolver::buildConstraintMapping(AvbdContactConstraint *contacts,
                                        physx::PxU32 numContacts,
                                        physx::PxU32 numBodies) {
  if (!mAllocator || numContacts == 0 || numBodies == 0) {
    return;
  }

  mContactMap.build(numBodies, contacts, numContacts, *mAllocator);
}

void AvbdSolver::solveBodyLocalConstraintsFast(
    AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts) {

  AvbdSolverBody &body = bodies[bodyIndex];

  if (body.invMass <= 0.0f) {
    return;
  }

  // Get only the constraints affecting this body - O(1) lookup!
  const physx::PxU32 *constraintIndices = nullptr;
  physx::PxU32 numBodyConstraints = 0;
  mContactMap.getBodyConstraints(bodyIndex, constraintIndices,
                                 numBodyConstraints);

  if (numBodyConstraints == 0 || constraintIndices == nullptr) {
    return;
  }

  // Jacobi accumulation: compute all corrections based on same body state,
  // then apply sum once to eliminate spurious rotational artifacts.
  physx::PxVec3 contactDeltaPos(0.0f);
  physx::PxVec3 contactDeltaTheta(0.0f);

  for (physx::PxU32 ci = 0; ci < numBodyConstraints; ++ci) {
    const physx::PxU32 c = constraintIndices[ci];
    AvbdContactConstraint &contact = contacts[c];

    const physx::PxU32 bodyAIdx = contact.header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contact.header.bodyIndexB;

    // Determine which body we are
    bool isBodyA = (bodyAIdx == bodyIndex);

    // Get the other body
    AvbdSolverBody *otherBody = nullptr;
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }

    // Compute world positions of contact points
    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r; // Contact arm for this body

    if (isBodyA) {
      r = body.rotation.rotate(contact.contactPointA);
      worldPosA = body.position + r;
      if (otherBody) {
        worldPosB = otherBody->position +
                    otherBody->rotation.rotate(contact.contactPointB);
      } else {
        worldPosB = contact.contactPointB;
      }
    } else {
      r = body.rotation.rotate(contact.contactPointB);
      if (otherBody) {
        worldPosA = otherBody->position +
                    otherBody->rotation.rotate(contact.contactPointA);
      } else {
        worldPosA = contact.contactPointA;
      }
      worldPosB = body.position + r;
    }

    // Compute constraint violation C(x) (negative = penetration)
    const physx::PxVec3 &normal = contact.contactNormal;
    physx::PxReal violation =
        (worldPosA - worldPosB).dot(normal) + contact.penetrationDepth;

    // AL target: inner solve drives C(x) toward lambda/rho instead of 0.
    physx::PxReal lambdaOverRho =
        contact.header.lambda / physx::PxMax(contact.header.rho, 1e-6f);

    // Skip if constraint is inactive: not penetrating AND no accumulated lambda
    if (violation >= lambdaOverRho && contact.header.lambda <= 0.0f) {
      continue;
    }

    // Compute generalized inverse mass for normal direction
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxReal w =
        body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);

    // Add other body's contribution if dynamic
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ? otherBody->rotation.rotate(contact.contactPointB)
                       : otherBody->rotation.rotate(contact.contactPointA);
      physx::PxVec3 rOtherCrossN = rOther.cross(normal);
      w += otherBody->invMass +
           rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
    }

    if (w <= 1e-6f) {
      continue;
    }

    // AL-augmented Baumgarte correction: drive violation toward lambda/rho
    // Clamp to non-negative to never pull bodies together for contacts
    physx::PxReal normalCorrectionMag = physx::PxMax(
        0.0f, -(violation - lambdaOverRho) / w * mConfig.baumgarte);

    // Direction sign for this body
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

    // Normal correction with scaled angular component for rotational stiffness
    physx::PxVec3 deltaPos =
        normal * (normalCorrectionMag * body.invMass * sign);
    physx::PxVec3 deltaTheta =
        (body.invInertiaWorld * rCrossN) *
        (normalCorrectionMag * sign * mConfig.angularContactScale);

    // Accumulate (Jacobi within body)
    contactDeltaPos += deltaPos;
    contactDeltaTheta += deltaTheta;
  }

  // Apply accumulated contact corrections
  body.position += contactDeltaPos;
  physx::PxReal angle = contactDeltaTheta.magnitude();
  if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    angle = physx::PxMin(angle, 0.1f);
    physx::PxVec3 axis = contactDeltaTheta.getNormalized();
    physx::PxReal halfAngle = angle * 0.5f;
    physx::PxQuat deltaQ(
        axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
        axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
    body.rotation = (deltaQ * body.rotation).getNormalized();
  }
}

//=============================================================================
// Thread-Safe Version Using External Constraint Map
//=============================================================================

void AvbdSolver::solveBodyLocalConstraintsFastWithMap(
    AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts, const AvbdBodyConstraintMap &contactMap) {

  AvbdSolverBody &body = bodies[bodyIndex];

  if (body.invMass <= 0.0f) {
    return;
  }

  // Get only the constraints affecting this body - O(1) lookup!
  // Uses external contactMap for thread safety
  const physx::PxU32 *constraintIndices = nullptr;
  physx::PxU32 numBodyConstraints = 0;
  contactMap.getBodyConstraints(bodyIndex, constraintIndices,
                                numBodyConstraints);

  if (numBodyConstraints == 0 || constraintIndices == nullptr) {
    return;
  }

  // Jacobi accumulation: compute all corrections based on same body state,
  // then apply sum once to eliminate spurious rotational artifacts.
  physx::PxVec3 contactDeltaPos(0.0f);
  physx::PxVec3 contactDeltaTheta(0.0f);

  for (physx::PxU32 ci = 0; ci < numBodyConstraints; ++ci) {
    const physx::PxU32 c = constraintIndices[ci];
    AvbdContactConstraint &contact = contacts[c];

    const physx::PxU32 bodyAIdx = contact.header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contact.header.bodyIndexB;

    // Determine which body we are
    bool isBodyA = (bodyAIdx == bodyIndex);

    // Get the other body
    AvbdSolverBody *otherBody = nullptr;
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }

    // Compute world positions of contact points
    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r; // Contact arm for this body

    if (isBodyA) {
      r = body.rotation.rotate(contact.contactPointA);
      worldPosA = body.position + r;
      if (otherBody) {
        worldPosB = otherBody->position +
                    otherBody->rotation.rotate(contact.contactPointB);
      } else {
        worldPosB = contact.contactPointB;
      }
    } else {
      r = body.rotation.rotate(contact.contactPointB);
      if (otherBody) {
        worldPosA = otherBody->position +
                    otherBody->rotation.rotate(contact.contactPointA);
      } else {
        worldPosA = contact.contactPointA;
      }
      worldPosB = body.position + r;
    }

    // Compute constraint violation C(x) (negative = penetration)
    const physx::PxVec3 &normal = contact.contactNormal;
    physx::PxReal violation =
        (worldPosA - worldPosB).dot(normal) + contact.penetrationDepth;

    // AL target: inner solve drives C(x) toward lambda/rho instead of 0.
    physx::PxReal lambdaOverRho =
        contact.header.lambda / physx::PxMax(contact.header.rho, 1e-6f);

    // Skip if constraint is inactive: not penetrating AND no accumulated lambda
    if (violation >= lambdaOverRho && contact.header.lambda <= 0.0f) {
      continue;
    }

    // Compute generalized inverse mass for normal direction
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxReal w =
        body.invMass + rCrossN.dot(body.invInertiaWorld * rCrossN);

    // Add other body's contribution if dynamic
    physx::PxVec3 rOther(0.0f);
    if (otherBody && otherBody->invMass > 0.0f) {
      rOther = isBodyA ? otherBody->rotation.rotate(contact.contactPointB)
                       : otherBody->rotation.rotate(contact.contactPointA);
      physx::PxVec3 rOtherCrossN = rOther.cross(normal);
      w += otherBody->invMass +
           rOtherCrossN.dot(otherBody->invInertiaWorld * rOtherCrossN);
    }

    if (w <= 1e-6f) {
      continue;
    }

    // AL-augmented Baumgarte correction: drive violation toward lambda/rho
    // Clamp to non-negative to never pull bodies together for contacts
    physx::PxReal normalCorrectionMag = physx::PxMax(
        0.0f, -(violation - lambdaOverRho) / w * mConfig.baumgarte);

    // Direction sign for this body
    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

    // Normal correction with scaled angular component for rotational stiffness
    physx::PxVec3 deltaPos =
        normal * (normalCorrectionMag * body.invMass * sign);
    physx::PxVec3 deltaTheta =
        (body.invInertiaWorld * rCrossN) *
        (normalCorrectionMag * sign * mConfig.angularContactScale);

    //=========================================================================
    // FRICTION CONSTRAINT (Position-based using previous position delta)
    //=========================================================================
    if (contact.friction > 0.0f) {
      // Compute relative position displacement at contact point
      physx::PxVec3 prevWorldPosA, prevWorldPosB;

      if (isBodyA) {
        prevWorldPosA =
            body.prevPosition + body.prevRotation.rotate(contact.contactPointA);
        if (otherBody) {
          prevWorldPosB = otherBody->prevPosition +
                          otherBody->prevRotation.rotate(contact.contactPointB);
        } else {
          prevWorldPosB = contact.contactPointB;
        }
      } else {
        if (otherBody) {
          prevWorldPosA = otherBody->prevPosition +
                          otherBody->prevRotation.rotate(contact.contactPointA);
        } else {
          prevWorldPosA = contact.contactPointA;
        }
        prevWorldPosB =
            body.prevPosition + body.prevRotation.rotate(contact.contactPointB);
      }

      // Relative displacement
      physx::PxVec3 dispA = worldPosA - prevWorldPosA;
      physx::PxVec3 dispB = worldPosB - prevWorldPosB;
      physx::PxVec3 relDisp = isBodyA ? (dispA - dispB) : (dispB - dispA);

      // Extract tangential component
      physx::PxVec3 tangentDisp = relDisp - normal * relDisp.dot(normal);
      physx::PxReal tangentDispMag = tangentDisp.magnitude();

      if (tangentDispMag > 1e-6f) {
        physx::PxVec3 tangent = tangentDisp / tangentDispMag;

        physx::PxVec3 rCrossT = r.cross(tangent);
        physx::PxReal wT =
            body.invMass + rCrossT.dot(body.invInertiaWorld * rCrossT);

        if (otherBody && otherBody->invMass > 0.0f) {
          physx::PxVec3 rOtherCrossT = rOther.cross(tangent);
          wT += otherBody->invMass +
                rOtherCrossT.dot(otherBody->invInertiaWorld * rOtherCrossT);
        }

        if (wT > 1e-6f) {
          physx::PxReal maxFrictionCorrection =
              contact.friction * physx::PxAbs(normalCorrectionMag);
          physx::PxReal frictionCorrection = tangentDispMag / wT;
          frictionCorrection =
              physx::PxMin(frictionCorrection, maxFrictionCorrection);

          deltaPos += -tangent * (frictionCorrection * body.invMass);
          deltaTheta += -(body.invInertiaWorld * rCrossT) * frictionCorrection;
        }
      }
    }

    // Accumulate (Jacobi within body)
    contactDeltaPos += deltaPos;
    contactDeltaTheta += deltaTheta;
  }

  // Apply accumulated contact corrections (no omega - matches original
  // solveBodyLocalConstraintsFast)
  body.position += contactDeltaPos;
  physx::PxReal angle = contactDeltaTheta.magnitude();
  if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
    angle = physx::PxMin(angle, 0.1f);
    physx::PxVec3 axis = contactDeltaTheta.getNormalized();
    physx::PxReal halfAngle = angle * 0.5f;
    physx::PxQuat deltaQ(
        axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
        axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
    body.rotation = (deltaQ * body.rotation).getNormalized();
  }
}

//=============================================================================
// Optimized Constraint Mapping Functions
//=============================================================================

void AvbdSolver::buildAllConstraintMappings(
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, AvbdSphericalJointConstraint *sphericalJoints,
    physx::PxU32 numSpherical, AvbdFixedJointConstraint *fixedJoints,
    physx::PxU32 numFixed, AvbdRevoluteJointConstraint *revoluteJoints,
    physx::PxU32 numRevolute, AvbdPrismaticJointConstraint *prismaticJoints,
    physx::PxU32 numPrismatic, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6) {

  if (!mAllocator || numBodies == 0)
    return;

  // IMPORTANT: Release all old mappings first to avoid stale data
  // This prevents accessing old indices when constraint counts change between
  // frames
  mContactMap.release(*mAllocator);
  mSphericalMap.release(*mAllocator);
  mFixedMap.release(*mAllocator);
  mRevoluteMap.release(*mAllocator);
  mPrismaticMap.release(*mAllocator);
  mD6Map.release(*mAllocator);

  // Build contact mapping
  if (numContacts > 0 && contacts) {
    mContactMap.build(numBodies, contacts, numContacts, *mAllocator);
  }

  // Build joint mappings
  if (numSpherical > 0 && sphericalJoints) {
    mSphericalMap.build(numBodies, sphericalJoints, numSpherical, *mAllocator);
  }
  if (numFixed > 0 && fixedJoints) {
    mFixedMap.build(numBodies, fixedJoints, numFixed, *mAllocator);
  }
  if (numRevolute > 0 && revoluteJoints) {
    mRevoluteMap.build(numBodies, revoluteJoints, numRevolute, *mAllocator);
  }
  if (numPrismatic > 0 && prismaticJoints) {
    mPrismaticMap.build(numBodies, prismaticJoints, numPrismatic, *mAllocator);
  }
  if (numD6 > 0 && d6Joints) {
    mD6Map.build(numBodies, d6Joints, numD6, *mAllocator);
  }
}

void AvbdSolver::solveBodyAllConstraintsFast(
    AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 bodyIndex,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
    AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
    AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
    AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const AvbdBodyConstraintMap &contactMap,
    const AvbdBodyConstraintMap &sphericalMap,
    const AvbdBodyConstraintMap &fixedMap,
    const AvbdBodyConstraintMap &revoluteMap,
    const AvbdBodyConstraintMap &prismaticMap,
    const AvbdBodyConstraintMap &d6Map, const AvbdBodyConstraintMap &gearMap,
    physx::PxReal dt) {

  PX_UNUSED(dt);

  AvbdSolverBody &body = bodies[bodyIndex];

  if (body.invMass <= 0.0f) {
    return;
  }

  // Accumulate corrections from all constraint types
  physx::PxVec3 totalDeltaPos(0.0f);
  physx::PxVec3 totalDeltaTheta(0.0f);
  physx::PxU32 numActiveConstraints = 0;

  // Process contact constraints using Jacobi accumulation within body.
  // All contacts compute corrections based on the SAME body state, then the sum
  // is applied once. This eliminates spurious rotational artifacts from
  // sequential processing that cause lateral drift in stacked objects.
  // Inter-body GS propagation is preserved because bodies are still processed
  // sequentially.
  if (contacts && numContacts > 0) {
    const physx::PxU32 *contactIndices = nullptr;
    physx::PxU32 numBodyContacts = 0;
    contactMap.getBodyConstraints(bodyIndex, contactIndices, numBodyContacts);

    physx::PxVec3 contactDeltaPos(0.0f);
    physx::PxVec3 contactDeltaTheta(0.0f);

    for (physx::PxU32 i = 0; i < numBodyContacts; ++i) {
      physx::PxU32 c = contactIndices[i];
      // Bounds check to prevent out-of-range access
      if (c >= numContacts)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeContactCorrection(contacts[c], bodies, numBodies, bodyIndex,
                                   deltaPos, deltaTheta)) {
        contactDeltaPos += deltaPos;
        contactDeltaTheta += deltaTheta;
      }
    }

    // Apply accumulated contact corrections with SOR over-relaxation
    body.position += contactDeltaPos * mConfig.omega;
    physx::PxReal angle = contactDeltaTheta.magnitude() * mConfig.omega;
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f);
      physx::PxVec3 axis = contactDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(
          axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
          axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }

  // Process spherical joint constraints
  if (sphericalJoints && numSpherical > 0) {
    const physx::PxU32 *sphericalIndices = nullptr;
    physx::PxU32 numBodySpherical = 0;
    sphericalMap.getBodyConstraints(bodyIndex, sphericalIndices,
                                    numBodySpherical);

    for (physx::PxU32 i = 0; i < numBodySpherical; ++i) {
      physx::PxU32 j = sphericalIndices[i];
      if (j >= numSpherical)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeSphericalJointCorrection(sphericalJoints[j], bodies, numBodies,
                                          bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }

  // Process fixed joint constraints
  if (fixedJoints && numFixed > 0) {
    const physx::PxU32 *fixedIndices = nullptr;
    physx::PxU32 numBodyFixed = 0;
    fixedMap.getBodyConstraints(bodyIndex, fixedIndices, numBodyFixed);

    for (physx::PxU32 i = 0; i < numBodyFixed; ++i) {
      physx::PxU32 j = fixedIndices[i];
      if (j >= numFixed)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeFixedJointCorrection(fixedJoints[j], bodies, numBodies,
                                      bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }

  // Process revolute joint constraints
  if (revoluteJoints && numRevolute > 0) {
    const physx::PxU32 *revoluteIndices = nullptr;
    physx::PxU32 numBodyRevolute = 0;
    revoluteMap.getBodyConstraints(bodyIndex, revoluteIndices, numBodyRevolute);

    for (physx::PxU32 i = 0; i < numBodyRevolute; ++i) {
      physx::PxU32 j = revoluteIndices[i];
      if (j >= numRevolute)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeRevoluteJointCorrection(revoluteJoints[j], bodies, numBodies,
                                         bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }

  // Process prismatic joint constraints
  if (prismaticJoints && numPrismatic > 0) {
    const physx::PxU32 *prismaticIndices = nullptr;
    physx::PxU32 numBodyPrismatic = 0;
    prismaticMap.getBodyConstraints(bodyIndex, prismaticIndices,
                                    numBodyPrismatic);

    for (physx::PxU32 i = 0; i < numBodyPrismatic; ++i) {
      physx::PxU32 j = prismaticIndices[i];
      if (j >= numPrismatic)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computePrismaticJointCorrection(prismaticJoints[j], bodies, numBodies,
                                          bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }

  // Process D6 joint constraints
  if (d6Joints && numD6 > 0) {
    const physx::PxU32 *d6Indices = nullptr;
    physx::PxU32 numBodyD6 = 0;
    d6Map.getBodyConstraints(bodyIndex, d6Indices, numBodyD6);

    for (physx::PxU32 i = 0; i < numBodyD6; ++i) {
      physx::PxU32 j = d6Indices[i];
      if (j >= numD6)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeD6JointCorrection(d6Joints[j], bodies, numBodies, bodyIndex,
                                   deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }

  // Process gear joint constraints
  if (gearJoints && numGear > 0) {
    const physx::PxU32 *gearIndices = nullptr;
    physx::PxU32 numBodyGear = 0;
    gearMap.getBodyConstraints(bodyIndex, gearIndices, numBodyGear);

    for (physx::PxU32 i = 0; i < numBodyGear; ++i) {
      physx::PxU32 j = gearIndices[i];
      if (j >= numGear)
        continue;
      physx::PxVec3 deltaPos, deltaTheta;
      if (computeGearJointCorrection(gearJoints[j], bodies, numBodies,
                                     bodyIndex, deltaPos, deltaTheta)) {
        totalDeltaPos += deltaPos;
        totalDeltaTheta += deltaTheta;
        numActiveConstraints++;
      }
    }
  }

  // Apply averaged joint corrections (contacts already applied via Gauss-Seidel
  // above)
  if (numActiveConstraints > 0) {
    physx::PxReal invCount =
        1.0f / static_cast<physx::PxReal>(numActiveConstraints);

    // Apply position correction with Baumgarte stabilization for joints
    body.position += totalDeltaPos * invCount * mConfig.baumgarte;

    // Apply rotation correction
    physx::PxVec3 avgDeltaTheta =
        totalDeltaTheta * invCount * mConfig.baumgarte;
    physx::PxReal angle = avgDeltaTheta.magnitude();
    if (angle > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      angle = physx::PxMin(angle, 0.1f);
      physx::PxVec3 axis = avgDeltaTheta.getNormalized();
      physx::PxReal halfAngle = angle * 0.5f;
      physx::PxQuat deltaQ(
          axis.x * physx::PxSin(halfAngle), axis.y * physx::PxSin(halfAngle),
          axis.z * physx::PxSin(halfAngle), physx::PxCos(halfAngle));
      body.rotation = (deltaQ * body.rotation).getNormalized();
    }
  }
}

} // namespace Dy
} // namespace physx
