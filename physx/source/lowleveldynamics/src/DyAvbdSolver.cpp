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

static bool bodyTouchesDeformableAnchor(AvbdContactConstraint *contacts,
                                        physx::PxU32 numContacts,
                                        physx::PxU32 bodyIndex) {
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bB = contacts[c].header.bodyIndexB;
    if (!hasDeformableStaticAnchor(contacts[c]))
      continue;
    if (bA == bodyIndex || bB == bodyIndex)
      return true;
  }
  return false;
}

// Suppress pose-solve bounce only on fast normal approach (sphere shot).
static const physx::PxReal kBodyStaticFastImpactSpeed =
    AvbdConstants::AVBD_BODY_STATIC_FAST_IMPACT_SPEED;

// Near-surface band for e=0 / mesh-following (meters). After geometric depen
// clears overlap, residual pose-solve velocity still separates - must clamp.
static const physx::PxReal kBodyStaticNearSurface = 0.05f;

/**
 * Material normal-velocity response after pose finalize (friction already applied).
 * - Deformable: mesh-relative e=0 (heave).
 * - Rigid body-static: material restitution with scene bounce threshold.
 * - Dyn-dyn: same restitution on relative normal speed (linear mass split).
 * Friction mu is consumed elsewhere (dual cone + body-static friction post-pass).
 */
static void applyAvbdMaterialNormalVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold) {
  const physx::PxReal invDt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
  const physx::PxReal bounceThreshold =
      bounceApproachThreshold > 0.0f
          ? bounceApproachThreshold
          : AvbdConstants::AVBD_BOUNCE_THRESHOLD;

  // ---- Body-static (incl. deformable anchors) ----
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;

    physx::PxU32 dominant = 0xFFFFFFFFu;
    physx::PxReal worstViolation = 1e9f;
    physx::PxVec3 domWorldA(0.0f), domWorldB(0.0f);

    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      if (!isBodyVsStaticContact(bA, bB, numBodies))
        continue;
      if (bA != i && bB != i)
        continue;

      const bool dynIsA = (bA == i);
      physx::PxVec3 worldA, worldB;
      if (dynIsA) {
        worldA = bodies[i].position +
                 bodies[i].rotation.rotate(contacts[c].contactPointA);
        worldB = contacts[c].contactPointB;
      } else {
        worldA = contacts[c].contactPointA;
        worldB = bodies[i].position +
                 bodies[i].rotation.rotate(contacts[c].contactPointB);
      }
      physx::PxReal violation =
          (worldA - worldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      if (hasDeformableStaticAnchor(contacts[c]))
        violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
      if (violation < worstViolation) {
        worstViolation = violation;
        dominant = c;
        domWorldA = worldA;
        domWorldB = worldB;
      }
    }

    if (dominant == 0xFFFFFFFFu)
      continue;

    const bool isDeform = hasDeformableStaticAnchor(contacts[dominant]);
    const AvbdContactConstraint &cc = contacts[dominant];
    const bool dynIsA = (cc.header.bodyIndexA == i);
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);

    physx::PxReal approach = 0.0f;
    if (linearVelAtSolveStart && linearVelAtSolveStart->size() == numBodies) {
      approach = -(*linearVelAtSolveStart)[i].dot(nd);
      if (approach < 0.0f)
        approach = 0.0f;
    }

    const physx::PxReal vn = bodies[i].linearVelocity.dot(nd);

    if (isDeform) {
      const physx::PxReal nearLim = kBodyStaticNearSurface;
      if (worstViolation >= nearLim)
        continue;
      if (approach > kBodyStaticFastImpactSpeed)
        continue;

      physx::PxReal vMeshN = 0.0f;
      if (invDt > 0.0f) {
        const physx::PxVec3 staticNow = dynIsA ? domWorldB : domWorldA;
        const physx::PxVec3 meshStep = staticNow - cc.staticPrevWorldPoint;
        const physx::PxReal stepCap = AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
        if (meshStep.magnitudeSquared() <= stepCap * stepCap)
          vMeshN = (meshStep * invDt).dot(nd);
      }
      const physx::PxReal vRelN = vn - vMeshN;
      if (vRelN > 0.0f)
        bodies[i].linearVelocity -= nd * vRelN;
      continue;
    }

    // Rigid body-static: material e from NP-combined patch restitution.
    // Compliant contacts (e < 0) treated as inelastic for now.
    const physx::PxReal e =
        (cc.restitution > 0.0f) ? physx::PxMin(cc.restitution, 1.0f) : 0.0f;
    physx::PxReal approachEff = approach;
    if (e > 0.0f && vn < 0.0f)
      approachEff = physx::PxMax(approachEff, -vn);
    if (e > 0.0f && approachEff > bounceThreshold) {
      const physx::PxReal desiredVn = e * approachEff;
      bodies[i].linearVelocity += nd * (desiredVn - vn);
    } else if (worstViolation < -1e-5f) {
      // Inelastic / resting: kill separating while still penetrating.
      if (vn > 0.0f)
        bodies[i].linearVelocity -= nd * vn;
    }
  }

  // Dyn-dyn restitution: relative normal impulse with invMass split.
  // Apply only for free rigid pairs (no deformable); e and bounce threshold
  // from material/scene. Skip if either body already handled as body-static
  // dominant this frame would double-count; dyn-dyn contacts are exclusive.
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    if (hasDeformableStaticAnchor(cc))
      continue;
    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    if (bA >= numBodies || bB >= numBodies)
      continue;
    if (bodies[bA].invMass <= 0.0f || bodies[bB].invMass <= 0.0f)
      continue;
    const physx::PxReal e =
        (cc.restitution > 0.0f) ? physx::PxMin(cc.restitution, 1.0f) : 0.0f;
    if (e <= 1e-6f)
      continue;
    if (!linearVelAtSolveStart || linearVelAtSolveStart->size() != numBodies)
      continue;

    const physx::PxVec3 &n = cc.contactNormal;
    const physx::PxReal vrel0 =
        ((*linearVelAtSolveStart)[bA] - (*linearVelAtSolveStart)[bB]).dot(n);
    const physx::PxReal approach = (vrel0 < 0.0f) ? -vrel0 : 0.0f;
    if (approach <= bounceThreshold)
      continue;

    const physx::PxReal vrel =
        (bodies[bA].linearVelocity - bodies[bB].linearVelocity).dot(n);
    const physx::PxReal desiredVrel = e * approach;
    if (vrel >= desiredVrel)
      continue;
    const physx::PxReal invSum = bodies[bA].invMass + bodies[bB].invMass;
    if (invSum < 1e-12f)
      continue;
    const physx::PxReal j = (desiredVrel - vrel) / invSum;
    bodies[bA].linearVelocity += n * (j * bodies[bA].invMass);
    bodies[bB].linearVelocity -= n * (j * bodies[bB].invMass);
  }
}

// Backward-compatible name used by postAlStages call site.
static void clampBodyStaticInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold) {
  applyAvbdMaterialNormalVelocity(bodies, numBodies, contacts, numContacts,
                                  linearVelAtSolveStart, dt,
                                  bounceApproachThreshold);
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
                       physx::PxU32 iterationOverride,
                       AvbdSoftParticle *kinematicShellParticles,
                       physx::PxU32 numKinematicShellParticles,
                       AvbdSoftContact *kinematicShellContacts,
                       physx::PxU32 numKinematicShellContacts,
                       AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solve", 0);

  if (!mInitialized || numBodies == 0) {
    return;
  }

  stats.reset();
  stats.numBodies = numBodies;
  stats.numContacts = numContacts;

  physx::PxReal invDt = 1.0f / dt;

  const bool hasKinematicShellContacts =
      kinematicShellContacts && numKinematicShellContacts > 0 &&
      kinematicShellParticles && numKinematicShellParticles > 0;
  static const physx::PxReal kShellFastImpactSpeed =
      AvbdConstants::AVBD_SHELL_FAST_IMPACT_SPEED;

  physx::PxArray<bool> touchesKinematicShell(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    touchesKinematicShell[i] = false;
  if (hasKinematicShellContacts) {
    for (physx::PxU32 sci = 0; sci < numKinematicShellContacts; ++sci) {
      const physx::PxU32 bi = kinematicShellContacts[sci].rigidBodyIdx;
      if (bi < numBodies)
        touchesKinematicShell[bi] = true;
    }
  }

  physx::PxArray<physx::PxVec3> shellLinearVelAtSolveStart;
  if (hasKinematicShellContacts) {
    shellLinearVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      shellLinearVelAtSolveStart[i] = bodies[i].linearVelocity;
  }

  // Stage 1: Prediction
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  // Stage 2: Graph Coloring (skip if pre-computed coloring is provided)
  if (mConfig.enableParallelization && numColors == 0) {
    PX_PROFILE_ZONE("AVBD.graphColoring", 0);
    computeGraphColoring(bodies, numBodies, contacts, numContacts, stats);
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
  bool hasBodyStaticContact = false;
  bool hasDeformableAnchorContact = false;
  bool allBodyVsStatic = (numContacts > 0);
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    if (isBodyVsStaticContact(contacts[c].header.bodyIndexA,
                              contacts[c].header.bodyIndexB, numBodies)) {
      hasBodyStaticContact = true;
    } else {
      allBodyVsStatic = false;
    }
    if (hasDeformableStaticAnchor(contacts[c]))
      hasDeformableAnchorContact = true;
  }
  // Fast sphere-on-mesh islands: single dynamic + deformable static only.
  const bool deformableFastImpactIsland =
      allBodyVsStatic && hasDeformableAnchorContact;

  physx::PxArray<bool> touchingBodyStatic(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    touchingBodyStatic[i] = false;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bB = contacts[c].header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies))
      continue;
    if (bA < numBodies)
      touchingBodyStatic[bA] = true;
    if (bB < numBodies)
      touchingBodyStatic[bB] = true;
  }

  // Snapshot pre-solve velocity for material restitution (incl. pure dyn-dyn
  // islands) and deformable fast-impact blend.
  physx::PxArray<physx::PxVec3> linearVelAtSolveStart;
  if (numContacts > 0) {
    linearVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      linearVelAtSolveStart[i] = bodies[i].linearVelocity;
  }

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

        physx::PxReal accelWeight = 0.0f;
        if (!touchingBodyStatic[i] && gravMag > 1e-6f) {
          accelWeight =
              physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f);
        }

        // Warmstart position: x = x_n + v*dt + accelWeight * g*dt^2
        // Body-vs-static: start from inertial prediction only. Gravity
        // warmstart overshoots into the mesh on fast impacts without CCD;
        // the supported RHS (accelWeight=0) then fights contacts and ejects.
        if (touchingBodyStatic[i]) {
          const bool deformableTouch =
              bodyTouchesDeformableAnchor(contacts, numContacts, i);
          const bool fastImpact =
              bodies[i].linearVelocity.magnitude() > kBodyStaticFastImpactSpeed;
          // Slow support on heaving mesh: inertial init pulls bodies into the
          // surface (accelWeight=0 already removes gravity drive from RHS).
          // Fast deformable impact: inertial start avoids warmstart overshoot.
          if (deformableTouch && fastImpact) {
            bodies[i].position = bodies[i].inertialPosition;
          } else {
            bodies[i].position =
                bodies[i].prevPosition + bodies[i].linearVelocity * dt;
          }
          bodies[i].rotation = bodies[i].inertialRotation;
        } else if (touchesKinematicShell[i]) {
          const bool fastImpact =
              bodies[i].linearVelocity.magnitude() > kShellFastImpactSpeed;
          if (fastImpact)
            bodies[i].position = bodies[i].inertialPosition;
          else
            bodies[i].position =
                bodies[i].prevPosition + bodies[i].linearVelocity * dt;
          bodies[i].rotation = bodies[i].inertialRotation;
        } else {
          bodies[i].position = bodies[i].prevPosition +
                               bodies[i].linearVelocity * dt +
                               gravity * (accelWeight * dt * dt);
          bodies[i].rotation = bodies[i].inertialRotation;
        }
      }
    }
  }

  // =========================================================================
  // Kinematic shell AL warmstart (normal + tangent dual state on soft rows)
  // =========================================================================
  if (hasKinematicShellContacts) {
    PX_PROFILE_ZONE("AVBD.kinematicShellWarmstart", 0);
    for (physx::PxU32 sci = 0; sci < numKinematicShellContacts; ++sci) {
      AvbdSoftContact &sc = kinematicShellContacts[sci];
      if (sc.particleIdx >= numKinematicShellParticles ||
          kinematicShellParticles[sc.particleIdx].invMass > 0.0f)
        continue;
      sc.alLambda *= mConfig.avbdAlpha * mConfig.avbdGamma;
      sc.k = physx::PxMax(1000.0f,
                          physx::PxMin(sc.ke, sc.k * mConfig.avbdGamma));
      for (int ti = 0; ti < 2; ++ti) {
        sc.alLambdaTangent[ti] *= mConfig.avbdAlpha * mConfig.avbdGamma;
        sc.penTangent[ti] = physx::PxMax(
            1000.0f, physx::PxMin(mConfig.avbdPenaltyMax,
                                  sc.penTangent[ti] * mConfig.avbdGamma));
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
        penScale = AvbdConstants::AVBD_PEN_SCALE_DYN_DYN;
      } else {
        // Body-vs-static: high stiffness to compete with joint penalties
        // in articulation scenarios (joint rho ~1e6).
        effectiveMass = physx::PxMax(massA, massB);
        penScale = AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC;
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

    if (hasKinematicShellContacts) {
      const physx::PxReal shellBoostFloor =
          AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC * invDt2;
      for (physx::PxU32 sci = 0; sci < numKinematicShellContacts; ++sci) {
        AvbdSoftContact &sc = kinematicShellContacts[sci];
        if (sc.particleIdx >= numKinematicShellParticles ||
            kinematicShellParticles[sc.particleIdx].invMass > 0.0f)
          continue;
        if (sc.rigidBodyIdx >= numBodies || bodies[sc.rigidBodyIdx].invMass <= 0.0f)
          continue;
        const physx::PxReal mass = 1.0f / bodies[sc.rigidBodyIdx].invMass;
        const physx::PxReal floor =
            physx::PxMax(shellBoostFloor, 0.25f * mass * invDt2);
        sc.k = physx::PxMax(physx::PxMax(sc.k, floor), 1000.0f);
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
      if (hasDeformableStaticAnchor(contacts[c])) {
        // Moving mesh anchor: no alpha-soften on normals.
        contacts[c].C0 = 0.0f;
        continue;
      }

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
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.solveIterations", 0);

    // Chebyshev extrapolation overshoots on deep body-vs-static impacts (no CCD).
    const bool useChebyshev =
        !hasDeformableAnchorContact && mConfig.chebyshevRho > 0.0f &&
        mConfig.chebyshevRho < 1.0f;
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
                              contactMap, colorBatches, numColors,
                              kinematicShellParticles, numKinematicShellParticles,
                              kinematicShellContacts, numKinematicShellContacts);
        stats.totalIterations++;
      }
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt, stats);
      }
      if (hasKinematicShellContacts) {
        PX_PROFILE_ZONE("AVBD.kinematicShellDual", 0);
        updateKinematicShellDual(bodies, numBodies, kinematicShellContacts,
                                 numKinematicShellContacts);
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

  // Shared post-AL stage list (depen, Decision A friction, pose-split vel, e=0/e)
  postAlStages(dt, invDt, bodies, numBodies, contacts, numContacts, gravity,
               hasBodyStaticContact, deformableFastImpactIsland,
               touchingBodyStatic,
               numContacts > 0 ? &linearVelAtSolveStart : nullptr,
               kinematicShellParticles, numKinematicShellParticles,
               kinematicShellContacts, numKinematicShellContacts,
               touchesKinematicShell,
               hasKinematicShellContacts ? &shellLinearVelAtSolveStart : nullptr,
               nullptr, 0, false, nullptr, 0);

}

void AvbdSolver::postAlStages(
    physx::PxReal dt, physx::PxReal invDt, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const physx::PxVec3 &gravity,
    bool hasBodyStaticContact, bool deformableFastImpactIsland,
    const physx::PxArray<bool> &touchingBodyStatic,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    const physx::PxArray<bool> &touchesKinematicShell,
    const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    bool applyVelocityDamping, AvbdSoftParticle *softParticlesForVel,
    physx::PxU32 numSoftParticlesForVel) {

  const bool hasKinematicShellContacts =
      shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0;
  static const physx::PxReal kShellFastImpactSpeed =
      AvbdConstants::AVBD_SHELL_FAST_IMPACT_SPEED;

  // Snapshot pose after the block solve; depenetration is geometric correction
  // and must not become launch velocity (friction tangents may).
  physx::PxArray<physx::PxVec3> postBlockPos(numBodies);
  physx::PxArray<physx::PxQuat> postBlockRot(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    postBlockPos[i] = bodies[i].position;
    postBlockRot[i] = bodies[i].rotation;
  }

  const physx::PxArray<bool> *shellSkipDepen =
      hasKinematicShellContacts ? &touchesKinematicShell : nullptr;
  if (hasBodyStaticContact && contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.bodyStaticDepenetration", 0);
    physx::PxU32 anyDeform = 0;
    for (physx::PxU32 bi = 0; bi < numBodies && anyDeform == 0; ++bi) {
      if (bodies[bi].invMass > 0.0f &&
          bodyTouchesDeformableAnchor(contacts, numContacts, bi))
        anyDeform = 1;
    }
    const physx::PxU32 depenSweeps =
        deformableFastImpactIsland ? 8u
        : (anyDeform != 0 ? (numBodies > 2u ? 10u : 8u) : 6u);
    applyBodyStaticNormalDepenetrationSweeps(bodies, numBodies, contacts,
                                           numContacts, gravity, dt,
                                           depenSweeps, shellSkipDepen);
  }
  if (hasKinematicShellContacts) {
    PX_PROFILE_ZONE("AVBD.kinematicShellDepenetration", 0);
    applyKinematicShellNormalDepenetrationSweeps(
        bodies, numBodies, shellParticles, numShellParticles, shellContacts,
        numShellContacts, gravity, dt, 8u);
  }

  physx::PxArray<physx::PxVec3> postDepenPos(numBodies);
  physx::PxArray<physx::PxQuat> postDepenRot(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    postDepenPos[i] = bodies[i].position;
    postDepenRot[i] = bodies[i].rotation;
  }

  if (contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.bodyStaticFriction", 0);
    applyBodyStaticFrictionSweeps(bodies, numBodies, contacts, numContacts, dt,
                                  6u, &postDepenPos, &postDepenRot,
                                  shellSkipDepen);
  }
  if (hasKinematicShellContacts) {
    PX_PROFILE_ZONE("AVBD.kinematicShellFriction", 0);
    applyKinematicShellFrictionSweeps(
        bodies, numBodies, shellParticles, numShellParticles, shellContacts,
        numShellContacts, dt, 4u, &postBlockPos, &postBlockRot);
  }

  // Motors after friction (pose-level), before velocity finalize.
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

      physx::PxVec3 worldAxis =
          isAStatic ? jnt.localFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
                    : (bodies[idxA].rotation * jnt.localFrameA)
                          .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
      worldAxis.normalize();

      if (!isBStatic) {
        AvbdSolverBody &bodyB = bodies[idxB];
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
        physx::PxReal ha = deltaAngle * 0.5f;
        physx::PxQuat dRot(worldAxis.x * physx::PxSin(ha),
                           worldAxis.y * physx::PxSin(ha),
                           worldAxis.z * physx::PxSin(ha),
                           physx::PxCos(ha));
        bodyB.rotation = (dRot * bodyB.rotation).getNormalized();
      }

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

  // Finalize velocity: block motion + friction/motor tangents; exclude depen.
  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;

        const physx::PxVec3 vFromBlock =
            (postBlockPos[i] - bodies[i].prevPosition) * invDt;
        const physx::PxVec3 vFromFriction =
            (bodies[i].position - postDepenPos[i]) * invDt;
        const physx::PxVec3 vFromPose = vFromBlock + vFromFriction;
        bool fastNormalImpact = false;
        if (deformableFastImpactIsland && i < touchingBodyStatic.size() &&
            touchingBodyStatic[i] && linearVelAtSolveStart &&
            linearVelAtSolveStart->size() == numBodies) {
          for (physx::PxU32 c = 0; c < numContacts; ++c) {
            const physx::PxU32 bA = contacts[c].header.bodyIndexA;
            const physx::PxU32 bB = contacts[c].header.bodyIndexB;
            if (!isBodyVsStaticContact(bA, bB, numBodies))
              continue;
            if (bA != i && bB != i)
              continue;
            const bool dynIsA = (bA == i);
            const physx::PxVec3 nd =
                contacts[c].contactNormal * (dynIsA ? 1.0f : -1.0f);
            const physx::PxReal approach =
                -(*linearVelAtSolveStart)[i].dot(nd);
            if (approach > kBodyStaticFastImpactSpeed) {
              fastNormalImpact = true;
              break;
            }
          }
        }
        if (fastNormalImpact) {
          bodies[i].linearVelocity =
              (*linearVelAtSolveStart)[i] * 0.85f + vFromPose * 0.15f;
        } else if (i < touchesKinematicShell.size() && touchesKinematicShell[i] &&
                   shellLinearVelAtSolveStart &&
                   shellLinearVelAtSolveStart->size() == numBodies) {
          bool shellFast = false;
          for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
            if (shellContacts[sci].rigidBodyIdx != i)
              continue;
            const physx::PxReal approach =
                -(*shellLinearVelAtSolveStart)[i].dot(shellContacts[sci].normal);
            if (approach > kShellFastImpactSpeed) {
              shellFast = true;
              break;
            }
          }
          if (shellFast)
            bodies[i].linearVelocity =
                (*shellLinearVelAtSolveStart)[i] * 0.85f + vFromPose * 0.15f;
          else
            bodies[i].linearVelocity = vFromPose;
        } else {
          bodies[i].linearVelocity = vFromPose;
        }

        if (applyVelocityDamping)
          bodies[i].linearVelocity *= mConfig.velocityDamping;

        physx::PxQuat deltaQBlock =
            postBlockRot[i] * bodies[i].prevRotation.getConjugate();
        if (deltaQBlock.w < 0.0f)
          deltaQBlock = -deltaQBlock;
        physx::PxVec3 wBlock =
            physx::PxVec3(deltaQBlock.x, deltaQBlock.y, deltaQBlock.z) *
            (2.0f * invDt);
        physx::PxQuat deltaQFr =
            bodies[i].rotation * postDepenRot[i].getConjugate();
        if (deltaQFr.w < 0.0f)
          deltaQFr = -deltaQFr;
        physx::PxVec3 wFr =
            physx::PxVec3(deltaQFr.x, deltaQFr.y, deltaQFr.z) * (2.0f * invDt);
        bodies[i].angularVelocity = wBlock + wFr;
        bodies[i].angularVelocity *= mConfig.angularDamping;

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
    // Material normal response: body-static e / deformable e=0 / dyn-dyn bounce.
    // Gate on numContacts (not hasBodyStaticContact) so pure dyn-dyn islands
    // still consume restitution e (criterion 2 / Entry 160).
    if (contacts && numContacts > 0) {
      PX_PROFILE_ZONE("AVBD.materialNormalVelocity", 0);
      clampBodyStaticInelasticNormalVelocities(
          bodies, numBodies, contacts, numContacts, linearVelAtSolveStart, dt,
          mConfig.bounceApproachSpeedThreshold());
    }
    if (hasKinematicShellContacts) {
      PX_PROFILE_ZONE("AVBD.kinematicShellInelasticVel", 0);
      clampKinematicShellInelasticNormalVelocities(
          bodies, numBodies, shellParticles, numShellParticles, shellContacts,
          numShellContacts, shellLinearVelAtSolveStart, dt);
    }

    if (softParticlesForVel && numSoftParticlesForVel > 0) {
      for (physx::PxU32 i = 0; i < numSoftParticlesForVel; ++i) {
        if (softParticlesForVel[i].invMass > 0.0f)
          softParticlesForVel[i].updateVelocityFromPosition(invDt);
      }
    }
  }
}

void AvbdSolver::solveIsland(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors,
    physx::PxU32 iterationOverride, AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies, AvbdSoftContact *softContacts,
    physx::PxU32 numSoftContacts, bool kinematicShellBatch,
    AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solveIsland", 0);

  const bool hasJoints = (numD6 > 0 || numGear > 0);
  const bool hasDeformableSoftVbd =
      (numSoftParticles > 0 && numSoftBodies > 0 && softContacts &&
       numSoftContacts > 0 && !kinematicShellBatch);

  // One island entry: joint/soft-VBD module vs contact/shell module.
  // Both call postAlStages; contact primal uses accumulateBodyContactRows.
  if (hasJoints || hasDeformableSoftVbd) {
    solveWithJoints(dt, bodies, numBodies, contacts, numContacts, d6Joints,
                    numD6, gearJoints, numGear, gravity, contactMap, d6Map,
                    gearMap, colorBatches, numColors, iterationOverride,
                    softParticles, numSoftParticles, softBodies, numSoftBodies,
                    softContacts, numSoftContacts, stats);
  } else if (kinematicShellBatch && softContacts && numSoftContacts > 0 &&
             softParticles && numSoftParticles > 0) {
    solve(dt, bodies, numBodies, contacts, numContacts, gravity, contactMap,
          colorBatches, numColors, iterationOverride, softParticles,
          numSoftParticles, softContacts, numSoftContacts, stats);
  } else {
    solve(dt, bodies, numBodies, contacts, numContacts, gravity, contactMap,
          colorBatches, numColors, iterationOverride, nullptr, 0, nullptr, 0,
          stats);
  }
}

//=============================================================================
// Graph Coloring
//=============================================================================

void AvbdSolver::computeGraphColoring(AvbdSolverBody *bodies,
                                      physx::PxU32 numBodies,
                                      AvbdContactConstraint *contacts,
                                      physx::PxU32 numContacts,
                                      AvbdSolverStats &stats) {
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

  stats.numColorGroups = numColors;
}

//=============================================================================
// Body-Based Graph Coloring for Block Coordinate Descent
//=============================================================================

void AvbdSolver::computeBodyColoring(AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     AvbdContactConstraint *contacts,
                                     physx::PxU32 numContacts,
                                     AvbdSolverStats &stats) {
  PX_ASSERT(mAllocator != nullptr);

  // Initialize body coloring if not already done
  if (!mBodyColoring.isInitialized()) {
    mBodyColoring.initialize(numBodies, *mAllocator);
  }

  // Perform body-based coloring
  physx::PxU32 numColors =
      mBodyColoring.colorBodies(contacts, numContacts, bodies, numBodies);

  stats.numColorGroups = numColors;
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
                                             physx::PxReal dt,
                                             AvbdSolverStats &stats) {
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
    const bool deformableStaticAnchor = hasDeformableStaticAnchor(contacts[c]);

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
    if (deformableStaticAnchor) {
      violation = finalizeBodyVsStaticViolation(violation,
                                              contacts[c].penetrationDepth);
    }

    // =====================================================================
    // AL dual + Coulomb cone (avbd-demo3d manifold.cpp updateDual):
    //   F = K*C + ?;  Fn = min(0,F_n);  ||Ft|| <= ?|Fn|;  store F as ?.
    // Bound uses normal force F_n (not raw ? alone). Tangents are a 2D cone.
    // =====================================================================
    physx::PxReal newLambda = 0.0f;
    {
      const physx::PxReal pen = contacts[c].header.penalty;
      const physx::PxReal oldLambda = contacts[c].header.lambda;
      const physx::PxReal mu = contactCoulombMu(contacts[c]);

      physx::PxReal tC0 = 0.0f, tC1 = 0.0f;
      if (contacts[c].friction > 0.0f || contacts[c].staticFriction > 0.0f) {
        const bool bodyVsStatic =
            isBodyVsStaticContact(bodyAIdx, bodyBIdx, numBodies);
        physx::PxVec3 worldPosA, worldPosB, prevWorldPosA, prevWorldPosB;
        if (bodyAIdx < numBodies) {
          worldPosA =
              bodies[bodyAIdx].position +
              bodies[bodyAIdx].rotation.rotate(contacts[c].contactPointA);
          prevWorldPosA =
              bodies[bodyAIdx].prevPosition +
              bodies[bodyAIdx].prevRotation.rotate(contacts[c].contactPointA);
        } else {
          worldPosA = contacts[c].contactPointA;
          prevWorldPosA = deformableStaticAnchor
                              ? contacts[c].staticPrevWorldPoint
                              : contacts[c].contactPointA;
        }
        if (bodyBIdx < numBodies) {
          worldPosB =
              bodies[bodyBIdx].position +
              bodies[bodyBIdx].rotation.rotate(contacts[c].contactPointB);
          prevWorldPosB =
              bodies[bodyBIdx].prevPosition +
              bodies[bodyBIdx].prevRotation.rotate(contacts[c].contactPointB);
        } else {
          worldPosB = contacts[c].contactPointB;
          prevWorldPosB = deformableStaticAnchor
                              ? contacts[c].staticPrevWorldPoint
                              : contacts[c].contactPointB;
        }
        const physx::PxVec3 relDisp =
            bodyVsStatic
                ? computeBodyVsStaticRelDisp(worldPosA, prevWorldPosA, worldPosB,
                                             prevWorldPosB, contacts[c],
                                             numBodies)
                : (worldPosA - prevWorldPosA) - (worldPosB - prevWorldPosB);
        tC0 = relDisp.dot(contacts[c].tangent0);
        tC1 = relDisp.dot(contacts[c].tangent1);
      }

      physx::PxReal Fn = 0.0f, Ft0 = 0.0f, Ft1 = 0.0f;
      const physx::PxReal preLen = avbdEvaluateContactForcesCone(
          pen, violation, oldLambda, contacts[c].tangentPenalty0, tC0,
          contacts[c].tangentLambda0, contacts[c].tangentPenalty1, tC1,
          contacts[c].tangentLambda1, mu, Fn, Ft0, Ft1);
      // Coulomb bound uses Fn / prior ? only (demo3d). Do NOT inject m*g here:
      // per-contact weight floors multi-count box corners and glue HelloWorld
      // stacks under ball impact. Resting grip is the post-pass, impact-gated.
      const physx::PxReal nCap = physx::PxMax(
          -Fn, (oldLambda < 0.0f) ? -oldLambda : 0.0f);
      newLambda = Fn;
      contacts[c].header.lambda = Fn;
      contacts[c].tangentLambda0 = Ft0;
      contacts[c].tangentLambda1 = Ft1;

      if (newLambda < 0.0f) {
        physx::PxReal growthDist = physx::PxAbs(violation);
        if (deformableStaticAnchor ||
            (numContacts > 4u &&
             isBodyVsStaticContact(bodyAIdx, bodyBIdx, numBodies)))
          growthDist = physx::PxMin(growthDist, 0.15f);
        contacts[c].header.penalty =
            physx::PxMin(pen + beta * growthDist, penaltyMax);
      }
      const physx::PxReal bounds = nCap * mu;
      if (preLen <= bounds) {
        contacts[c].tangentPenalty0 = physx::PxMin(
            contacts[c].tangentPenalty0 + beta * physx::PxAbs(tC0),
            penaltyMax);
        contacts[c].tangentPenalty1 = physx::PxMin(
            contacts[c].tangentPenalty1 + beta * physx::PxAbs(tC1),
            penaltyMax);
      }
      setFrictionStick(contacts[c],
                       avbdFrictionStickFromDual(nCap, mu, preLen, tC0, tC1));
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

  stats.constraintError =
      (numActive > 0) ? sqrtf(totalError / (physx::PxReal)numActive) : 0.0f;
  stats.activeConstraints = numActive;
}

//=============================================================================
// Body-static normal depenetration (TGS-style capped geometric projection)
//=============================================================================
void AvbdSolver::applyBodyStaticNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
    const physx::PxArray<bool> *skipDepenForBodies) {
  if (numContacts == 0 || numBodies == 0 || dt <= 0.0f || sweeps == 0)
    return;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      if (!isBodyVsStaticContact(bA, bB, numBodies))
        continue;

      const bool dynIsA = (bA < numBodies);
      const bool dynIsB = (bB < numBodies);
      if (dynIsA == dynIsB)
        continue;
      const physx::PxU32 bi = dynIsA ? bA : bB;
      if (skipDepenForBodies && bi < skipDepenForBodies->size() &&
          (*skipDepenForBodies)[bi] &&
          hasDeformableStaticAnchor(contacts[c]))
        continue;
      AvbdSolverBody &body = bodies[bi];

      physx::PxVec3 worldA, worldB;
      if (dynIsA) {
        worldA = body.position + body.rotation.rotate(contacts[c].contactPointA);
        worldB = contacts[c].contactPointB;
      } else {
        worldA = contacts[c].contactPointA;
        worldB = body.position + body.rotation.rotate(contacts[c].contactPointB);
      }

      physx::PxReal violation =
          (worldA - worldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      if (hasDeformableStaticAnchor(contacts[c]))
        violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
      if (violation >= -1e-5f)
        continue;

      const physx::PxReal approachSpeed =
          body.linearVelocity.magnitude() + gravity.magnitude() * dt;
      physx::PxReal sweepCap = physx::PxMax(approachSpeed * dt * 0.5f, 0.01f);
      if (hasDeformableStaticAnchor(contacts[c])) {
        const physx::PxVec3 staticNow = dynIsA ? worldB : worldA;
        const physx::PxVec3 meshStep =
            staticNow - contacts[c].staticPrevWorldPoint;
        // Mesh step + deeper floor: prevent multi-cycle trough sink when the
        // heaving surface rises into resting stacks (was capped too soft).
        sweepCap = physx::PxMax(sweepCap, meshStep.magnitude() * 1.5f);
        sweepCap = physx::PxMax(sweepCap, 0.04f);
        if (violation < -0.05f)
          sweepCap = physx::PxMax(sweepCap, -violation * 0.6f);
      }
      const physx::PxReal corr = physx::PxMin(-violation, sweepCap);
      if (dynIsA)
        body.position += contacts[c].contactNormal * corr;
      else
        body.position -= contacts[c].contactNormal * corr;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

//=============================================================================
// Sequential body-static friction (deformable-mesh anchors + rigid static partners)
//
// TGS-style projected Gauss-Seidel friction, decoupled from the AVBD block
// solve. Rigid plane: all corner contacts per sweep. Deformable anchors: one
// dominant contact per body (shell bodies use shell friction instead of NP).
//=============================================================================
void AvbdSolver::applyBodyStaticFrictionSweeps(AvbdSolverBody *bodies,
                                               physx::PxU32 numBodies,
                                               AvbdContactConstraint *contacts,
                                               physx::PxU32 numContacts,
                                               physx::PxReal dt,
                                               physx::PxU32 sweeps,
                                               const physx::PxArray<physx::PxVec3> *velSeedPos,
                                               const physx::PxArray<physx::PxQuat> *velSeedRot,
                                               const physx::PxArray<bool> *skipForBodies) {
  if (numContacts == 0 || numBodies == 0 || dt <= 0.0f || sweeps == 0)
    return;

  const physx::PxReal invDt = 1.0f / dt;

  // Deformable anchors: one dominant contact per body (multiple mesh rows
  // over-constrain tangential DOF). Rigid static partners: all contacts in
  // sequential GS. Raw deformable contact counts gate mesh-velocity tracking.
  physx::PxArray<physx::PxU32> dominantDeformable(numBodies);
  physx::PxArray<physx::PxU32> bodyDeformRawCount(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    dominantDeformable[i] = 0xFFFFFFFFu;
    bodyDeformRawCount[i] = 0;
  }
  physx::PxArray<physx::PxU32> frContacts;
  physx::PxArray<physx::PxU32> bodyContactCount(numBodies);
  physx::PxArray<physx::PxReal> bodyContactNormalSum(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodyContactCount[i] = 0;
    bodyContactNormalSum[i] = 0.0f;
  }
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    if (cc.friction <= 0.0f && cc.staticFriction <= 0.0f)
      continue;
    const bool dynA = cc.header.bodyIndexA < numBodies;
    const bool dynB = cc.header.bodyIndexB < numBodies;
    if (dynA == dynB)
      continue;
    if (!isBodyVsStaticContact(cc.header.bodyIndexA, cc.header.bodyIndexB,
                               numBodies))
      continue;
    const physx::PxU32 bi = dynA ? cc.header.bodyIndexA : cc.header.bodyIndexB;
    if (hasDeformableStaticAnchor(cc) && skipForBodies &&
        bi < skipForBodies->size() && (*skipForBodies)[bi])
      continue;
    if (hasDeformableStaticAnchor(cc)) {
      bodyDeformRawCount[bi]++;
      const physx::PxU32 cur = dominantDeformable[bi];
      if (cur == 0xFFFFFFFFu ||
          physx::PxAbs(cc.header.lambda) >
              physx::PxAbs(contacts[cur].header.lambda))
        dominantDeformable[bi] = c;
    } else {
      frContacts.pushBack(c);
      bodyContactCount[bi]++;
      bodyContactNormalSum[bi] += physx::PxAbs(cc.header.lambda);
    }
  }
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (dominantDeformable[i] != 0xFFFFFFFFu) {
      frContacts.pushBack(dominantDeformable[i]);
      bodyContactCount[i] = 1;
      bodyContactNormalSum[i] =
          physx::PxAbs(contacts[dominantDeformable[i]].header.lambda);
    }
  }
  if (frContacts.empty())
    return;

  // Work on a separate velocity field seeded from this step's pose change, so
  // sweeps never feed position back into themselves (that caused divergence on
  // stacks where one base box carries several mesh contacts). The friction-only
  // velocity delta is converted to a tangential pose shift at the very end,
  // leaving the block solve's normal penetration resolution intact.
  physx::PxArray<physx::PxVec3> vLin(numBodies), vAng(numBodies), vLin0(numBodies),
      vAng0(numBodies);
  physx::PxArray<bool> touched(numBodies);
  physx::PxArray<physx::PxReal> bodySpeed(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    touched[i] = false;
    bodySpeed[i] = 0.0f;
    if (bodies[i].invMass <= 0.0f) {
      vLin[i] = vAng[i] = vLin0[i] = vAng0[i] = physx::PxVec3(0.0f);
      continue;
    }
    const physx::PxVec3 seedPos =
        velSeedPos && i < velSeedPos->size() ? (*velSeedPos)[i] : bodies[i].position;
    const physx::PxQuat seedRot =
        velSeedRot && i < velSeedRot->size() ? (*velSeedRot)[i] : bodies[i].rotation;
    physx::PxVec3 vl = (seedPos - bodies[i].prevPosition) * invDt;
    physx::PxQuat dq = seedRot * bodies[i].prevRotation.getConjugate();
    if (dq.w < 0.0f)
      dq = -dq;
    physx::PxVec3 va = physx::PxVec3(dq.x, dq.y, dq.z) * (2.0f * invDt);
    vLin[i] = vLin0[i] = vl;
    vAng[i] = vAng0[i] = va;
    bodySpeed[i] = vl.magnitude() + va.magnitude() * 0.5f;
  }

  // Resting weight floor only when quasi-static. Impact / ball-shot must use
  // dual normal force alone - m*g floors glued HelloWorld boxes and killed ball KE.
  static const physx::PxReal kRestSpeed = 1.5f;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    for (physx::PxU32 fi = 0; fi < frContacts.size(); ++fi) {
      AvbdContactConstraint &cc = contacts[frContacts[fi]];
      const bool dynIsA = cc.header.bodyIndexA < numBodies;
      const physx::PxU32 bi = dynIsA ? cc.header.bodyIndexA : cc.header.bodyIndexB;
      AvbdSolverBody &body = bodies[bi];
      touched[bi] = true;

      const physx::PxVec3 cpLocal = dynIsA ? cc.contactPointA : cc.contactPointB;
      const physx::PxVec3 r = body.rotation.rotate(cpLocal);
      const physx::PxMat33 &invI = body.invInertiaWorld;

      physx::PxVec3 worldA, worldB;
      if (dynIsA) {
        worldA = body.position + r;
        worldB = cc.contactPointB;
      } else {
        worldA = cc.contactPointA;
        worldB = body.position + r;
      }
      physx::PxReal viol =
          (worldA - worldB).dot(cc.contactNormal) + cc.penetrationDepth;
      if (hasDeformableStaticAnchor(cc))
        viol = finalizeBodyVsStaticViolation(viol, cc.penetrationDepth);

      // Mesh target velocity via SupportClass policy (solve-loop contract).
      // eRigidPlane / eDeformableMultiCorner -> vMesh=0; few-contact ride on.
      physx::PxVec3 vMesh(0.0f);
      if (cc.supportClass == AvbdSupportClass::eUnset) {
        if (hasDeformableStaticAnchor(cc)) {
          const physx::PxReal mass =
              (body.invMass > 1e-8f) ? (1.0f / body.invMass) : 1e8f;
          if (bodyDeformRawCount[bi] >=
                  AvbdConstants::AVBD_SUPPORT_MULTI_CORNER_MIN &&
              mass >= AvbdConstants::AVBD_SUPPORT_MULTI_CORNER_MASS)
            cc.supportClass = AvbdSupportClass::eDeformableMultiCorner;
          else
            cc.supportClass = AvbdSupportClass::eDeformableFewContact;
        } else {
          cc.supportClass = AvbdSupportClass::eRigidPlane;
        }
      }
      if (cc.supportClass == AvbdSupportClass::eDeformableFewContact ||
          cc.supportClass == AvbdSupportClass::eShell) {
        const physx::PxVec3 staticNow = dynIsA ? worldB : worldA;
        physx::PxVec3 vFull = (staticNow - cc.staticPrevWorldPoint) * invDt;
        const physx::PxReal stepCap = AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
        if ((staticNow - cc.staticPrevWorldPoint).magnitudeSquared() >
            stepCap * stepCap) {
          vFull = physx::PxVec3(0.0f);
        }
        const physx::PxVec3 &n = cc.contactNormal;
        vMesh = vFull - n * vFull.dot(n);
        const physx::PxReal vCap = AvbdConstants::AVBD_SURFACE_VMESH_CAP;
        const physx::PxReal vMag2 = vMesh.magnitudeSquared();
        if (vMag2 > vCap * vCap)
          vMesh *= vCap / physx::PxSqrt(vMag2);
      }

      // Normal force from dual / penalty depth only by default.
      physx::PxReal contactN = physx::PxMax(
          physx::PxAbs(cc.header.lambda),
          cc.header.penalty * physx::PxMax(0.0f, -viol));

      // Soft shared m*g fill only when resting (not under ball impact).
      if (body.invMass > 1e-8f && bodySpeed[bi] < kRestSpeed && viol <= 0.05f) {
        const physx::PxReal weight =
            (1.0f / body.invMass) * 9.81f /
            physx::PxReal(physx::PxMax(1u, bodyContactCount[bi]));
        contactN = physx::PxMax(contactN, weight);
      }

      // Velocity-level friction is dynamic ?; static ? is for dual stick only.
      const physx::PxReal mu =
          cc.friction > 0.0f ? cc.friction
                             : (cc.staticFriction > 0.0f ? cc.staticFriction
                                                         : 0.0f);
      const physx::PxReal jmax = contactN * mu * dt;
      if (jmax <= 0.0f)
        continue;

      const physx::PxVec3 tangents[2] = {cc.tangent0, cc.tangent1};
      physx::PxReal jUnc[2] = {0.0f, 0.0f};
      physx::PxReal kEff[2] = {0.0f, 0.0f};
      physx::PxVec3 rCrossT[2];
      for (physx::PxU32 a = 0; a < 2; ++a) {
        const physx::PxVec3 &t = tangents[a];
        rCrossT[a] = r.cross(t);
        kEff[a] = body.invMass + rCrossT[a].dot(invI * rCrossT[a]);
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxVec3 vRel = (vLin[bi] + vAng[bi].cross(r)) - vMesh;
        jUnc[a] = -vRel.dot(t) / kEff[a];
      }
      avbdProjectImpulseCone(jmax, jUnc[0], jUnc[1]);
      for (physx::PxU32 a = 0; a < 2; ++a) {
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxReal j = jUnc[a];
        vLin[bi] += tangents[a] * (j * body.invMass);
        vAng[bi] += invI * (rCrossT[a] * j);
      }
    }
  }

  // Apply only the friction-induced velocity delta as a tangential pose shift.
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!touched[i] || bodies[i].invMass <= 0.0f)
      continue;
    const physx::PxVec3 dPos = (vLin[i] - vLin0[i]) * dt;
    bodies[i].position += dPos;
    const physx::PxVec3 dTheta = (vAng[i] - vAng0[i]) * dt;
    if (dTheta.magnitudeSquared() > 1e-16f) {
      physx::PxQuat dqi(dTheta.x, dTheta.y, dTheta.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation + dqi * bodies[i].rotation * 0.5f).getNormalized();
    }
  }
}

void AvbdSolver::applyKinematicShellNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps) {
  if (!softContacts || numSoftContacts == 0 || !bodies || numBodies == 0 ||
      !softParticles || sweeps == 0 || dt <= 0.0f)
    return;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      AvbdSoftContact &sc = softContacts[sci];
      if (sc.rigidBodyIdx >= numBodies)
        continue;
      if (sc.particleIdx >= numSoftParticles ||
          softParticles[sc.particleIdx].invMass > 0.0f)
        continue;
      AvbdSolverBody &body = bodies[sc.rigidBodyIdx];
      if (body.invMass <= 0.0f)
        continue;

      const physx::PxReal violation =
          avbdKinematicShellContactViolation(sc, body);
      if (violation >= -1e-5f)
        continue;

      const physx::PxReal approachSpeed =
          body.linearVelocity.magnitude() + gravity.magnitude() * dt;
      physx::PxReal sweepCap =
          physx::PxMax(approachSpeed * dt * 0.5f, 0.04f);
      const physx::PxVec3 meshStep = sc.surfacePoint - sc.surfacePointPrev;
      sweepCap = physx::PxMax(sweepCap, meshStep.magnitude() * 1.5f);
      if (violation < -0.05f)
        sweepCap = physx::PxMax(sweepCap, -violation * 0.6f);
      const physx::PxReal corr = physx::PxMin(-violation, sweepCap);
      body.position += sc.normal * corr;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

void AvbdSolver::applyKinematicShellFrictionSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt, physx::PxU32 sweeps,
    const physx::PxArray<physx::PxVec3> *velSeedPos,
    const physx::PxArray<physx::PxQuat> *velSeedRot) {
  if (!softContacts || numSoftContacts == 0 || !bodies || numBodies == 0 ||
      !softParticles || sweeps == 0 || dt <= 0.0f)
    return;

  const physx::PxReal invDt = 1.0f / dt;

  // One dominant shell corner per body (largest penetration).
  physx::PxArray<physx::PxU32> dominantContact(numBodies);
  physx::PxArray<physx::PxReal> worstViol(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    dominantContact[i] = 0xFFFFFFFFu;
    worstViol[i] = 1e9f;
  }
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContact &sc = softContacts[sci];
    if (sc.rigidBodyIdx >= numBodies || sc.friction <= 0.0f)
      continue;
    if (sc.particleIdx >= numSoftParticles ||
        softParticles[sc.particleIdx].invMass > 0.0f)
      continue;
    const physx::PxReal viol =
        avbdKinematicShellContactViolation(sc, bodies[sc.rigidBodyIdx]);
    if (viol > 0.05f)
      continue;
    const physx::PxU32 bi = sc.rigidBodyIdx;
    if (viol < worstViol[bi]) {
      worstViol[bi] = viol;
      dominantContact[bi] = sci;
    }
  }
  physx::PxArray<physx::PxU32> frContacts;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (dominantContact[i] != 0xFFFFFFFFu)
      frContacts.pushBack(dominantContact[i]);
  }
  if (frContacts.empty())
    return;

  physx::PxArray<physx::PxVec3> vLin(numBodies), vAng(numBodies), vLin0(numBodies),
      vAng0(numBodies);
  physx::PxArray<bool> touched(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    touched[i] = false;
    if (bodies[i].invMass <= 0.0f) {
      vLin[i] = vAng[i] = vLin0[i] = vAng0[i] = physx::PxVec3(0.0f);
      continue;
    }
    const physx::PxVec3 seedPos =
        velSeedPos && i < velSeedPos->size() ? (*velSeedPos)[i] : bodies[i].position;
    const physx::PxQuat seedRot =
        velSeedRot && i < velSeedRot->size() ? (*velSeedRot)[i] : bodies[i].rotation;
    const physx::PxVec3 vl =
        (seedPos - bodies[i].prevPosition) * invDt;
    physx::PxQuat dq = seedRot * bodies[i].prevRotation.getConjugate();
    if (dq.w < 0.0f)
      dq = -dq;
    const physx::PxVec3 va = physx::PxVec3(dq.x, dq.y, dq.z) * (2.0f * invDt);
    vLin[i] = vLin0[i] = vl;
    vAng[i] = vAng0[i] = va;
  }

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    for (physx::PxU32 fi = 0; fi < frContacts.size(); ++fi) {
      AvbdSoftContact &sc = softContacts[frContacts[fi]];
      const physx::PxU32 bi = sc.rigidBodyIdx;
      AvbdSolverBody &body = bodies[bi];
      touched[bi] = true;

      const physx::PxVec3 r = body.rotation.rotate(sc.rigidLocalPoint);
      const physx::PxMat33 &invI = body.invInertiaWorld;
      // Shell path: one dominant contact per body. Always track tangential
      // mesh velocity (shot sphere mass can be >>5). Stack multi-corner
      // energy is limited by NP multi-corner gate + e=0 clamps.
      physx::PxVec3 vMesh = (sc.surfacePoint - sc.surfacePointPrev) * invDt;
      {
        const physx::PxVec3 &n = sc.normal;
        vMesh = vMesh - n * vMesh.dot(n);
        const physx::PxReal vCap = 12.0f;
        const physx::PxReal vMag2 = vMesh.magnitudeSquared();
        if (vMag2 > vCap * vCap)
          vMesh *= vCap / physx::PxSqrt(vMag2);
      }

      const physx::PxReal viol = avbdKinematicShellContactViolation(sc, body);
      const physx::PxReal normalForce =
          PxMax(PxAbs(sc.alLambda), sc.k * PxMax(0.0f, -viol));
      const physx::PxReal jmax = normalForce * sc.friction * dt;
      if (jmax <= 0.0f)
        continue;

      const physx::PxVec3 tangents[2] = {sc.tangent1, sc.tangent2};
      physx::PxReal jUnc[2] = {0.0f, 0.0f};
      physx::PxReal kEff[2] = {0.0f, 0.0f};
      physx::PxVec3 rCrossT[2];
      for (physx::PxU32 a = 0; a < 2; ++a) {
        const physx::PxVec3 &t = tangents[a];
        rCrossT[a] = r.cross(t);
        kEff[a] = body.invMass + rCrossT[a].dot(invI * rCrossT[a]);
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxVec3 vRel = (vLin[bi] + vAng[bi].cross(r)) - vMesh;
        jUnc[a] = -vRel.dot(t) / kEff[a];
      }
      avbdProjectImpulseCone(jmax, jUnc[0], jUnc[1]);
      for (physx::PxU32 a = 0; a < 2; ++a) {
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxReal j = jUnc[a];
        vLin[bi] += tangents[a] * (j * body.invMass);
        vAng[bi] += invI * (rCrossT[a] * j);
      }
    }
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!touched[i] || bodies[i].invMass <= 0.0f)
      continue;
    const physx::PxVec3 dPos = (vLin[i] - vLin0[i]) * dt;
    bodies[i].position += dPos;
    const physx::PxVec3 dTheta = (vAng[i] - vAng0[i]) * dt;
    if (dTheta.magnitudeSquared() > 1e-16f) {
      physx::PxQuat dqi(dTheta.x, dTheta.y, dTheta.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation + dqi * bodies[i].rotation * 0.5f).getNormalized();
    }
  }
}

void AvbdSolver::clampKinematicShellInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxArray<physx::PxVec3> * /*linearVelAtSolveStart*/,
    physx::PxReal dt) {
  if (!softContacts || numSoftContacts == 0 || !bodies || numBodies == 0 ||
      !softParticles)
    return;

  const physx::PxReal invDt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;

    physx::PxU32 dominant = 0xFFFFFFFFu;
    physx::PxReal worstViolation = 0.0f;
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContact &sc = softContacts[sci];
      if (sc.rigidBodyIdx != i)
        continue;
      if (sc.particleIdx >= numSoftParticles ||
          softParticles[sc.particleIdx].invMass > 0.0f)
        continue;
      const physx::PxReal viol =
          avbdKinematicShellContactViolation(sc, bodies[i]);
      if (viol < worstViolation) {
        worstViolation = viol;
        dominant = sci;
      }
    }
    if (dominant == 0xFFFFFFFFu)
      continue;
    // Near contact or penetration: e=0 normal clamp (depenetration may have
    // cleared overlap while pose-derived velocity still separates).
    if (worstViolation >= 0.05f)
      continue;

    const AvbdSoftContact &sc = softContacts[dominant];
    const physx::PxVec3 nd = sc.normal;
    const physx::PxReal vn = bodies[i].linearVelocity.dot(nd);
    const physx::PxReal vMeshN =
        invDt > 0.0f
            ? ((sc.surfacePoint - sc.surfacePointPrev) * invDt).dot(nd)
            : 0.0f;
    const physx::PxReal vRelN = vn - vMeshN;
    if (vRelN > 0.0f)
      bodies[i].linearVelocity -= nd * vRelN;
  }
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

void AvbdSolver::updateKinematicShellDual(AvbdSolverBody *bodies,
                                          physx::PxU32 numBodies,
                                          AvbdSoftContact *shellContacts,
                                          physx::PxU32 numShellContacts) {
  if (!shellContacts || numShellContacts == 0 || !bodies)
    return;
  const physx::PxReal beta = mConfig.avbdBeta;
  const physx::PxReal penaltyMax = mConfig.avbdPenaltyMax;
  for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
    AvbdSoftContact &sc = shellContacts[sci];
    if (sc.rigidBodyIdx >= numBodies)
      continue;
    avbdUpdateKinematicShellContactDual(sc, bodies[sc.rigidBodyIdx], beta,
                                        penaltyMax);
  }
}

void AvbdSolver::accumulateBodyContactRows(
    AvbdSolverBody &body, physx::PxU32 bodyIndex, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    physx::PxReal massInvDt2, AvbdBlock6x6 &A, physx::PxVec3 &gLinear,
    physx::PxVec3 &gAngular, physx::PxU32 &numTouching) {

  bool bodyUsesKinematicShellNormals = false;
  if (shellContacts && numShellContacts > 0) {
    for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
      if (shellContacts[sci].rigidBodyIdx == bodyIndex) {
        bodyUsesKinematicShellNormals = true;
        break;
      }
    }
  }

  // Use contactMap for O(K) lookup if available, else O(N) scan
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  if (contactMap && contactMap->numBodies > 0) {
    contactMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
  }
  const physx::PxU32 loopCount = mapIndices ? mapCount : numContacts;

  const physx::PxReal contactBoostFloor =
      AvbdConstants::AVBD_CONTACT_BOOST_FRACTION * massInvDt2;

  for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
    const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
    const physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;

    if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
      continue;
    }

    // Shell rows supply normals for boxes; keep NP anchors for post-pass friction.
    if (bodyUsesKinematicShellNormals &&
        hasDeformableStaticAnchor(contacts[c])) {
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

    violation -= mConfig.avbdAlpha * contacts[c].C0;
    if (hasDeformableStaticAnchor(contacts[c])) {
      violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
    }

    physx::PxReal pen = contacts[c].header.penalty;
    // Per-body primal boost: small fraction of M/h^2 safety net.
    pen = physx::PxMax(pen, contactBoostFloor);
    physx::PxReal lambda = contacts[c].header.lambda;

    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxVec3 gradPos = normal * sign;
    physx::PxVec3 gradRot = rCrossN * sign;

    A.addConstraintContribution(gradPos, gradRot, pen);
    numTouching++;

    // Normal force (unilateral) + optional Coulomb-cone tangents in 6x6.
    physx::PxReal f = physx::PxMin(0.0f, pen * violation + lambda);
    if (f < 0.0f) {
      gLinear += gradPos * f;
      gAngular += gradRot * f;
    }

    // Never stack body-static tangents in aggregated 6x6 (PhysX contract).
    if ((contacts[c].friction > 0.0f || contacts[c].staticFriction > 0.0f) &&
        useBodyVsStaticFrictionIn6x6(bodyAIdx, bodyBIdx, numBodies)) {
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
      const physx::PxVec3 relDisp =
          (worldPosA - prevWorldPosA) - (worldPosB - prevWorldPosB);

      const physx::PxReal tPen0 =
          physx::PxMax(contacts[c].tangentPenalty0, contactBoostFloor);
      const physx::PxReal tPen1 =
          physx::PxMax(contacts[c].tangentPenalty1, contactBoostFloor);
      const physx::PxReal tC0 = relDisp.dot(contacts[c].tangent0);
      const physx::PxReal tC1 = relDisp.dot(contacts[c].tangent1);
      const physx::PxReal mu = contactCoulombMu(contacts[c]);

      physx::PxReal Fn = 0.0f, Ft0 = 0.0f, Ft1 = 0.0f;
      (void)avbdEvaluateContactForcesCone(
          pen, violation, lambda, tPen0, tC0, contacts[c].tangentLambda0, tPen1,
          tC1, contacts[c].tangentLambda1, mu, Fn, Ft0, Ft1);

      {
        const physx::PxVec3 &t = contacts[c].tangent0;
        const physx::PxVec3 rCrossT = r.cross(t);
        const physx::PxVec3 tGradPos = t * sign;
        const physx::PxVec3 tGradRot = rCrossT * sign;
        A.addConstraintContribution(tGradPos, tGradRot, tPen0);
        gLinear += tGradPos * Ft0;
        gAngular += tGradRot * Ft0;
      }
      {
        const physx::PxVec3 &t = contacts[c].tangent1;
        const physx::PxVec3 rCrossT = r.cross(t);
        const physx::PxVec3 tGradPos = t * sign;
        const physx::PxVec3 tGradRot = rCrossT * sign;
        A.addConstraintContribution(tGradPos, tGradRot, tPen1);
        gLinear += tGradPos * Ft1;
        gAngular += tGradRot * Ft1;
      }
    }
  }

  if (shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0) {
    const physx::PxReal shellBoostFloor =
        AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC * massInvDt2;
    AvbdVec6 shellRhs;
    shellRhs.linear = physx::PxVec3(0.0f);
    shellRhs.angular = physx::PxVec3(0.0f);
    for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
      const AvbdSoftContact &sc = shellContacts[sci];
      if (sc.rigidBodyIdx != bodyIndex)
        continue;
      if (sc.particleIdx >= numShellParticles)
        continue;
      if (shellParticles[sc.particleIdx].invMass > 0.0f)
        continue;
      avbdAddKinematicShellContactContribution_rigid(
          shellContacts[sci], bodyIndex, body, shellBoostFloor, A, shellRhs);
      numTouching++;
    }
    gLinear += shellRhs.linear;
    gAngular += shellRhs.angular;
  }
}

void AvbdSolver::solveLocalSystem(AvbdSolverBody &body, AvbdSolverBody *bodies,
                                  physx::PxU32 numBodies,
                                  AvbdContactConstraint *contacts,
                                  physx::PxU32 numContacts, physx::PxReal dt,
                                  physx::PxReal invDt2,
                                  const AvbdBodyConstraintMap *contactMap,
                                  AvbdSoftParticle *kinematicShellParticles,
                                  physx::PxU32 numKinematicShellParticles,
                                  AvbdSoftContact *kinematicShellContacts,
                                  physx::PxU32 numKinematicShellContacts) {

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
  // Step 3: Shared contact + shell primal accumulation (body-static contract)
  // =========================================================================

  physx::PxU32 numTouching = 0;
  accumulateBodyContactRows(
      body, bodyIndex, bodies, numBodies, contacts, numContacts, contactMap,
      kinematicShellParticles, numKinematicShellParticles,
      kinematicShellContacts, numKinematicShellContacts, massInvDt2, A, gLinear,
      gAngular, numTouching);

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
    physx::PxU32 numColors, AvbdSoftParticle *kinematicShellParticles,
    physx::PxU32 numKinematicShellParticles,
    AvbdSoftContact *kinematicShellContacts,
    physx::PxU32 numKinematicShellContacts) {

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
                       invDt2, contactMap, kinematicShellParticles,
                       numKinematicShellParticles, kinematicShellContacts,
                       numKinematicShellContacts);
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
