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
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"

#include "DyAvbdParallelFor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
} // namespace

//=============================================================================
// Main Solver Entry Point
//=============================================================================

void AvbdSolver::solve(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                       const AvbdBodyConstraintMap *contactMap,
                       AvbdColorBatch *colorBatches, physx::PxU32 numColors,
                       physx::PxU32 iterationOverride) {
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
        // Body-vs-static: high stiffness to compete with joint penalties
        // in articulation scenarios (joint rho ~1e6).
        effectiveMass = physx::PxMax(massA, massB);
        penScale = 2.0f;
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
      physx::PxReal rawC0 = (wA - wB).dot(contacts[c].contactNormal) +
                       contacts[c].penetrationDepth;

      // Depth-adaptive C0 clamping: for deep penetrations (fast impacts),
      // reduce C0 so that alpha blending does not over-soften the correction.
      const physx::PxReal c0Threshold = 0.05f;  // 50 mm
      const physx::PxReal c0MaxDepth  = 0.20f;  // 200 mm
      if (rawC0 < -c0Threshold) {
        physx::PxReal t = PxClamp(
            (c0MaxDepth + rawC0) / (c0MaxDepth - c0Threshold), 0.0f, 1.0f);
        rawC0 *= t;
      }
      contacts[c].C0 = rawC0;
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

    // Both 6x6 and 3x3 paths use AL dual update => primal+dual each iteration
    const physx::PxU32 iters = (iterationOverride > 0)
        ? iterationOverride : mConfig.innerIterations;
    const bool enableEarlyStop = (mConfig.positionTolerance > 0.0f && iters > 1);
    const physx::PxU32 minIterations = physx::PxMin(iters, physx::PxU32(4));
    const physx::PxReal rotationTolerance =
        physx::PxMax(4.0f * mConfig.positionTolerance, 1e-4f);
    physx::PxU32 consecutiveConvergedIterations = 0;
    physx::PxArray<physx::PxVec3> earlyStopPrevPos;
    physx::PxArray<physx::PxQuat> earlyStopPrevRot;
    if (enableEarlyStop) {
      earlyStopPrevPos.resize(numBodies);
      earlyStopPrevRot.resize(numBodies);
    }

    for (physx::PxU32 iter = 0; iter < iters; ++iter) {
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
  //
  // Parallelization: each body's local solve reads only its own position
  // (mutated) and neighbor positions (read-only). The AVBD proximal term
  // ensures convergence under Jacobi (parallel) updates.

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

  const bool useParallel = mConfig.enableParallelization && !useDeterministicOrder
      && numBodies >= AVBD_PARALLEL_MIN_ITEMS;

  auto solveBody = [&](physx::PxU32 idx) {
    const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
    if (bodies[i].invMass <= 0.0f)
      return;
    if (mConfig.enableLocal6x6Solve) {
      solveLocalSystem(bodies[i], bodies, numBodies, contacts, numContacts, dt,
                       invDt2, contactMap);
    } else {
      solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                 numContacts, nullptr, 0, nullptr, 0, dt,
                                 invDt2, contactMap, nullptr, nullptr);
    }
  };

  if (useParallel) {
    avbdParallelFor(0u, numBodies, solveBody);
  } else {
    for (physx::PxU32 idx = 0; idx < numBodies; ++idx)
      solveBody(idx);
  }
}

} // namespace Dy
} // namespace physx
