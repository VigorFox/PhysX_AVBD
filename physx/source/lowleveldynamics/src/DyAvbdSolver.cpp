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
          // PBD correction: correctionMag = violation / w
          // The generalized inverse mass w already accounts for both bodies,
          // so we apply the full correction (no 0.5 factor).
          physx::PxReal correctionMag = violation / angularW;
          physx::PxVec3 angCorrection =
              corrAxis * (correctionMag * sign);
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
    // Compute effective inverse mass for rotation along error axis
    physx::PxVec3 axis = rotError / rotErrorMag;
    physx::PxReal w = axis.dot(body.invInertiaWorld * axis);

    if (otherBody && otherBody->invMass > 0.0f) {
      w += axis.dot(otherBody->invInertiaWorld * axis);
    }

    if (w > 1e-6f) {
      // PBD-style correction: correctionMag = error / w
      // (baumgarte relaxation is applied externally by applyJointGS)
      physx::PxReal correctionMag = rotErrorMag / w;

      physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

      // Distribute correction to this body proportional to its invInertia
      // Same pattern as position: deltaTheta = (invI * axis) * (corr * sign)
      deltaTheta += (body.invInertiaWorld * axis) * (correctionMag * sign);
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

  // Angular velocity drive (damping + stiffness)
  // PxD6Drive: eTWIST=4 (bit 4=0x10), eSLERP=5 (bit 5=0x20),
  //            eSWING1=6 (bit 6=0x40), eSWING2=7 (bit 7=0x80)
  // Combined angular mask: 0xF0 (bits 4-7)
  //
  // Drive model: F = stiffness*(targetOrient - currentOrient)
  //              + damping*(targetAngVel - currentAngVel)
  //
  // Position-based: deltaTheta = (targetAngVel - relAngVel) * dt * factor
  // When targetAngVel=0 and damping>0 (pure damper):
  //   deltaTheta = -relAngVel * dt * factor   (opposes current motion)
  if ((joint.driveFlags & 0xF0) != 0) {
    physx::PxQuat jointFrameA =
        bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);

    physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
    if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
      jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

      bool slerpDrive = (joint.driveFlags & 0x20) != 0;

      physx::PxVec3 targetAngVel = joint.driveAngularVelocity;
      if (!PxIsFinite(targetAngVel.x) || !PxIsFinite(targetAngVel.y) ||
          !PxIsFinite(targetAngVel.z)) {
        targetAngVel = physx::PxVec3(0.0f);
      }
      physx::PxVec3 worldTargetAngVel = jointFrameA.rotate(targetAngVel);

      physx::PxVec3 thetaDelta(0.0f);
      physx::PxReal dt = 1.0f / 60.0f;

      // Estimate current relative angular velocity from rotation delta
      auto estimateAngVel = [&](const AvbdSolverBody &b) -> physx::PxVec3 {
        physx::PxQuat dq = b.rotation * b.prevRotation.getConjugate();
        if (dq.w < 0.0f) {
          dq = -dq;
        }
        physx::PxVec3 v(dq.x, dq.y, dq.z);
        v *= 2.0f;
        return v * (1.0f / dt);
      };

      physx::PxVec3 angVelA =
          otherBody ? estimateAngVel(*otherBody) : physx::PxVec3(0.0f);
      physx::PxVec3 angVelB = estimateAngVel(body);
      physx::PxVec3 relAngVel =
          isBodyA ? (angVelA - angVelB) : (angVelB - angVelA);

      if (slerpDrive) {
        // SLERP applies to all angular axes uniformly
        physx::PxReal damping = joint.angularDamping.z;
        if (damping > 0.0f && PxIsFinite(damping)) {
          // Velocity error: drive toward target angular velocity
          physx::PxVec3 velError = worldTargetAngVel - relAngVel;
          // dampingFactor scales from physical damping to position correction
          // damping=1000 -> factor~=0.05, provides moderate viscous resistance
          physx::PxReal dampingFactor =
              physx::PxMin(damping / 20000.0f, 0.1f);
          thetaDelta = velError * (dt * dampingFactor);
        }
      } else {
        // Individual axis drives: TWIST(bit4), SWING1(bit6), SWING2(bit7)
        struct AxisDrive {
          physx::PxU32 bit;
          int dampIdx;     // index into angularDamping
          physx::PxVec3 localAxis;
        };
        const AxisDrive axes[3] = {
            {0x10, 0, physx::PxVec3(1.0f, 0.0f, 0.0f)}, // TWIST
            {0x40, 1, physx::PxVec3(0.0f, 1.0f, 0.0f)}, // SWING1
            {0x80, 2, physx::PxVec3(0.0f, 0.0f, 1.0f)}, // SWING2
        };

        for (int a = 0; a < 3; ++a) {
          if ((joint.driveFlags & axes[a].bit) == 0) continue;
          physx::PxReal damping = (&joint.angularDamping.x)[axes[a].dampIdx];
          if (damping <= 0.0f || !PxIsFinite(damping)) continue;

          physx::PxVec3 worldAxis = jointFrameA.rotate(axes[a].localAxis);
          physx::PxReal targetOnAxis = worldTargetAngVel.dot(worldAxis);
          physx::PxReal currentOnAxis = relAngVel.dot(worldAxis);
          physx::PxReal velError = targetOnAxis - currentOnAxis;

          if (physx::PxAbs(velError) > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
            physx::PxReal dampingFactor =
                physx::PxMin(damping / 20000.0f, 0.1f);
            thetaDelta += worldAxis * (velError * dt * dampingFactor);
          }
        }
      }

      physx::PxReal thetaMag2 = thetaDelta.magnitudeSquared();
      if (thetaMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON &&
          PxIsFinite(thetaMag2)) {
        // Apply to the current body with proper sign
        physx::PxReal sign = isBodyA ? -1.0f : 1.0f;
        physx::PxVec3 correction = thetaDelta * sign;
        if (PxIsFinite(correction.x) && PxIsFinite(correction.y) &&
            PxIsFinite(correction.z)) {
          // Clamp max angular correction per iteration
          physx::PxReal maxAngCorrection = 0.02f;
          correction.x = physx::PxClamp(correction.x, -maxAngCorrection,
                                        maxAngCorrection);
          correction.y = physx::PxClamp(correction.y, -maxAngCorrection,
                                        maxAngCorrection);
          correction.z = physx::PxClamp(correction.z, -maxAngCorrection,
                                        maxAngCorrection);
          deltaTheta += correction;
          hasCorrection = true;
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
  // Stage 3: Penalty floor for contacts
  // =========================================================================
  if (contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.penaltyFloor", 0);
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
          // (A) Contact constraints: AVBD AL local system solve
          //     Only for bodies WITH contacts. Bodies without contacts
          //     were initialized to inertialPosition before the loop and
          //     are refined only by joint GS (step B).
          // ---------------------------------------------------------------
          {
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

            if (hasContacts) {
              // Body has contacts: full AL local system solve
              if (mConfig.enableLocal6x6Solve) {
                solveLocalSystem(bodies[i], bodies, numBodies, contacts,
                                 numContacts, dt, invDt2, contactMap);
              } else {
                solveLocalSystem3x3(bodies[i], bodies, numBodies, contacts,
                                    numContacts, dt, invDt2, contactMap);
              }
            }
            // No contacts: position was set before the loop, joint GS refines.
          }

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

          const physx::PxReal maxAngleSpherical = 0.12f;
          const physx::PxReal maxAngleFixed = 0.35f;
          const physx::PxReal maxAngleRevolute = 0.20f;
          const physx::PxReal maxAnglePrismatic = 0.20f;
          const physx::PxReal maxAngleD6 = 0.35f;
          const physx::PxReal maxAngleGear = 0.30f;

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

          // Spherical joints
          if (sphericalJoints && numSpherical > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (sphericalMap) {
              sphericalMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numSpherical) continue;
                physx::PxVec3 dp, dth;
                    if (computeSphericalJointCorrection(
                    sphericalJoints[jIdx[k]], bodies, numBodies, i, dp,
                    dth))
                  applyJointGS(dp, dth, maxAngleSpherical);
              }
            } else {
              for (physx::PxU32 k = 0; k < numSpherical; ++k) {
                physx::PxVec3 dp, dth;
                if (computeSphericalJointCorrection(sphericalJoints[k], bodies,
                                                    numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAngleSpherical);
              }
            }
          }

          // Fixed joints
          if (fixedJoints && numFixed > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (fixedMap) {
              fixedMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numFixed) continue;
                physx::PxVec3 dp, dth;
                if (computeFixedJointCorrection(fixedJoints[jIdx[k]], bodies,
                                                numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAngleFixed);
              }
            } else {
              for (physx::PxU32 k = 0; k < numFixed; ++k) {
                physx::PxVec3 dp, dth;
                if (computeFixedJointCorrection(fixedJoints[k], bodies,
                                                numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAngleFixed);
              }
            }
          }

          // Revolute joints
          if (revoluteJoints && numRevolute > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (revoluteMap) {
              revoluteMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numRevolute) continue;
                physx::PxVec3 dp, dth;
                if (computeRevoluteJointCorrection(revoluteJoints[jIdx[k]],
                                                   bodies, numBodies, i, dp,
                                                   dth))
                  applyJointGS(dp, dth, maxAngleRevolute);
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

          // Prismatic joints
          if (prismaticJoints && numPrismatic > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (prismaticMap) {
              prismaticMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numPrismatic) continue;
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

          // D6 joints
          if (d6Joints && numD6 > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (d6Map) {
              d6Map->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numD6) continue;
                physx::PxVec3 dp, dth;
                if (computeD6JointCorrection(d6Joints[jIdx[k]], bodies,
                                             numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAngleD6);
              }
            } else {
              for (physx::PxU32 k = 0; k < numD6; ++k) {
                physx::PxVec3 dp, dth;
                if (computeD6JointCorrection(d6Joints[k], bodies, numBodies, i,
                                             dp, dth))
                  applyJointGS(dp, dth, maxAngleD6);
              }
            }
          }

          // Gear joints
          if (gearJoints && numGear > 0) {
            const physx::PxU32 *jIdx = nullptr;
            physx::PxU32 jCnt = 0;
            if (gearMap) {
              gearMap->getBodyConstraints(i, jIdx, jCnt);
              for (physx::PxU32 k = 0; k < jCnt; ++k) {
                if (jIdx[k] >= numGear) continue;
                physx::PxVec3 dp, dth;
                if (computeGearJointCorrection(gearJoints[jIdx[k]], bodies,
                                               numBodies, i, dp, dth))
                  applyJointGS(dp, dth, maxAngleGear);
              }
            } else {
              for (physx::PxU32 k = 0; k < numGear; ++k) {
                physx::PxVec3 dp, dth;
                if (computeGearJointCorrection(gearJoints[k], bodies, numBodies,
                                               i, dp, dth))
                  applyJointGS(dp, dth, maxAngleGear);
              }
            }
          }

        } // end body loop
        mStats.totalIterations++;
      }

      // --- Dual step: AL multiplier updates for contacts + joints ---
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
    } // end iteration loop
  }

  // =========================================================================
  // Stage 6: Motor drives for RevoluteJoints (post-solve)
  //
  // Motor applies torque (limited by maxForce) to accelerate toward target
  // velocity. This matches TGS behavior where motor gradually accelerates
  // bodies. Applied AFTER constraint iterations so joint corrections are stable.
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
      physx::PxQuat deltaQ =
          bodyB.rotation * bodyB.prevRotation.getConjugate();
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

    // Apply D6 joint angular damping to connected bodies
    for (physx::PxU32 j = 0; j < numD6; ++j) {
      const AvbdD6JointConstraint &joint = d6Joints[j];
      physx::PxReal maxDamping = physx::PxMax(
          joint.angularDamping.x,
          physx::PxMax(joint.angularDamping.y, joint.angularDamping.z));
      if (maxDamping > 0.0f && joint.driveFlags != 0) {
        physx::PxReal dampingFactor = 1.0f / (1.0f + maxDamping * dt * 0.1f);
        dampingFactor = physx::PxMax(dampingFactor, 0.1f);
        if (joint.header.bodyIndexA < numBodies &&
            bodies[joint.header.bodyIndexA].invMass > 0.0f) {
          bodies[joint.header.bodyIndexA].angularVelocity *= dampingFactor;
        }
        if (joint.header.bodyIndexB < numBodies &&
            bodies[joint.header.bodyIndexB].invMass > 0.0f) {
          bodies[joint.header.bodyIndexB].angularVelocity *= dampingFactor;
        }
      }
    }
  }
}

} // namespace Dy
} // namespace physx
