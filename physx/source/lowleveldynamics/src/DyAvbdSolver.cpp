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
#include "DyAvbdBoundedProjection.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"

#include "DyAvbdParallelFor.h"

#include <algorithm>
#include <cmath>

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

// Enforce the velocity counterpart of body-vs-static locked D6 linear rows.
// Position-level AL convergence can leave a small first-step pose residual;
// reconstructing velocity directly from that residual creates a velocity that
// violates an otherwise hard joint.  This is a Jacobian/effective-mass
// projection, not a magnitude dead-zone.  Dynamic-dynamic, limited/free and
// driven rows remain outside this first body-vs-static correctness slice.
static void projectBodyStaticLockedD6LinearVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint *joints, physx::PxU32 numJoints) {
  if (!bodies || !joints)
    return;

  for (physx::PxU32 ji = 0; ji < numJoints; ++ji) {
    const AvbdD6JointConstraint &joint = joints[ji];
    const bool aDynamic = joint.header.bodyIndexA < numBodies;
    const bool bDynamic = joint.header.bodyIndexB < numBodies;
    if (aDynamic == bDynamic)
      continue;

    AvbdSolverBody &body =
        bodies[aDynamic ? joint.header.bodyIndexA : joint.header.bodyIndexB];
    if (body.invMass <= 0.0f)
      continue;

    physx::PxQuat worldFrameA =
        aDynamic ? body.rotation * joint.localFrameA : joint.localFrameA;
    const physx::PxReal frameMagnitudeSquared = worldFrameA.magnitudeSquared();
    if (frameMagnitudeSquared > 1e-8f &&
        physx::PxIsFinite(frameMagnitudeSquared))
      worldFrameA *= 1.0f / physx::PxSqrt(frameMagnitudeSquared);
    const physx::PxVec3 r = body.rotation.rotate(
        aDynamic ? joint.anchorA : joint.anchorB);
    const bool allLinearLocked = joint.linearMotion == 0;

    for (physx::PxU32 axis = 0; axis < 3; ++axis) {
      if (joint.getLinearMotion(axis) != 0 ||
          joint.isLinearDriveEnabled(axis))
        continue;

      physx::PxVec3 worldAxis(0.0f);
      worldAxis[axis] = 1.0f;
      if (!allLinearLocked)
        worldAxis = worldFrameA.rotate(worldAxis);

      const physx::PxVec3 rCrossAxis = r.cross(worldAxis);
      const physx::PxReal recipResponse =
          body.invMass +
          rCrossAxis.dot(body.invInertiaWorld.transform(rCrossAxis));
      if (recipResponse <= 1e-12f || !physx::PxIsFinite(recipResponse))
        continue;

      const physx::PxReal anchorSpeed =
          (body.linearVelocity + body.angularVelocity.cross(r)).dot(worldAxis);
      if (!physx::PxIsFinite(anchorSpeed))
        continue;

      // C = anchorA-anchorB, so the dynamic-B Jacobian is -J.
      const physx::PxReal dynamicSign = aDynamic ? 1.0f : -1.0f;
      const physx::PxReal impulse =
          -dynamicSign * anchorSpeed / recipResponse;
      body.linearVelocity += worldAxis * (dynamicSign * impulse * body.invMass);
      body.angularVelocity += body.invInertiaWorld.transform(
          rCrossAxis * (dynamicSign * impulse));
    }

    // A dynamic body fixed to a static/world endpoint has no admissible
    // spatial velocity.  Project the complete six-dimensional locked
    // subspace after pose-to-velocity reconstruction; the row-wise linear
    // projection above remains responsible for partially locked joints.
    if (allLinearLocked && joint.angularMotion == 0 &&
        joint.driveFlags == 0) {
      body.linearVelocity = physx::PxVec3(0.0f);
      body.angularVelocity = physx::PxVec3(0.0f);
    }

  }
}

// Suppress pose-solve bounce only on fast normal approach (sphere shot).
static const physx::PxReal kBodyStaticFastImpactSpeed =
    AvbdConstants::AVBD_BODY_STATIC_FAST_IMPACT_SPEED;

// Near-surface band for e=0 / mesh-following (meters). After geometric depen
// clears overlap, residual pose-solve velocity still separates - must clamp.
static const physx::PxReal kBodyStaticNearSurface = 0.05f;

// The validated dense complete-component owner is deliberately capped.
// Larger components remain entirely on the legacy fail-closed path until a
// scalable backend satisfies the same atomic accuracy and performance gates.
static const physx::PxU32 kMaxPassiveMaterialComponentContacts = 16;

struct AvbdPassiveMaterialComponentRow {
  physx::PxU32 bodyA{PX_MAX_U32};
  physx::PxU32 bodyB{PX_MAX_U32};
  physx::PxVec3 linearA{0.0f};
  physx::PxVec3 angularA{0.0f};
  physx::PxVec3 linearB{0.0f};
  physx::PxVec3 angularB{0.0f};
  physx::PxReal solveStartVelocity{0.0f};
};

static physx::PxReal passiveMaterialRowResponse(
    const AvbdPassiveMaterialComponentRow &a,
    const AvbdPassiveMaterialComponentRow &b,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies) {
  physx::PxReal response = 0.0f;
  const auto addTerm =
      [&](physx::PxU32 bodyA, const physx::PxVec3 &linearA,
          const physx::PxVec3 &angularA, physx::PxU32 bodyB,
          const physx::PxVec3 &linearB,
          const physx::PxVec3 &angularB) {
        if (bodyA >= numBodies || bodyA != bodyB)
          return;
        const AvbdSolverBody &body = bodies[bodyA];
        response +=
            body.invMass * linearA.dot(linearB) +
            angularA.dot(
                body.invInertiaWorld.transform(angularB));
      };
  addTerm(a.bodyA, a.linearA, a.angularA,
          b.bodyA, b.linearA, b.angularA);
  addTerm(a.bodyA, a.linearA, a.angularA,
          b.bodyB, b.linearB, b.angularB);
  addTerm(a.bodyB, a.linearB, a.angularB,
          b.bodyA, b.linearA, b.angularA);
  addTerm(a.bodyB, a.linearB, a.angularB,
          b.bodyB, b.linearB, b.angularB);
  return response;
}

/**
 * Close every material normal row and Coulomb disk in a connected rigid
 * zero-restitution contact component from one reconstructed baseline.
 *
 * Normal complementarity and tangent maximum dissipation are block solves:
 * every normal row is updated from the same iterate, every tangent row is
 * updated from the same iterate, and the two complete blocks iterate to a
 * common fixed point.  No point-wise/body-wise Gauss-Seidel budget replay is
 * performed.  State is committed only after the whole component is finite
 * and satisfies the projected fixed-point residual.
 */
static void applyAvbdPassiveFrictionComponents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, AvbdSolverStats *stats) {
  if (!bodies || !contacts || numBodies == 0 ||
      numContacts == 0 || dt <= 0.0f)
    return;

  physx::PxArray<physx::PxU8> visitedContacts(numContacts);
  for (physx::PxU32 c = 0; c < numContacts; ++c)
    visitedContacts[c] = 0;

  for (physx::PxU32 seed = 0; seed < numContacts; ++seed) {
    if (visitedContacts[seed] ||
        !hasVelocityPassiveFrictionComponentOwner(contacts[seed]))
      continue;

    physx::PxArray<physx::PxU32> componentContacts;
    physx::PxArray<physx::PxU32> bodyQueue;
    physx::PxArray<physx::PxU8> componentBodies(numBodies);
    for (physx::PxU32 body = 0; body < numBodies; ++body)
      componentBodies[body] = 0;

    const auto enqueueBody = [&](physx::PxU32 bodyIndex) {
      if (bodyIndex < numBodies && !componentBodies[bodyIndex]) {
        componentBodies[bodyIndex] = 1;
        bodyQueue.pushBack(bodyIndex);
      }
    };
    visitedContacts[seed] = 1;
    componentContacts.pushBack(seed);
    enqueueBody(contacts[seed].header.bodyIndexA);
    enqueueBody(contacts[seed].header.bodyIndexB);
    for (physx::PxU32 queueIndex = 0;
         queueIndex < bodyQueue.size(); ++queueIndex) {
      const physx::PxU32 bodyIndex = bodyQueue[queueIndex];
      for (physx::PxU32 c = 0; c < numContacts; ++c) {
        if (visitedContacts[c] ||
            !hasVelocityPassiveFrictionComponentOwner(contacts[c]) ||
            (contacts[c].header.bodyIndexA != bodyIndex &&
             contacts[c].header.bodyIndexB != bodyIndex))
          continue;
        visitedContacts[c] = 1;
        componentContacts.pushBack(c);
        enqueueBody(contacts[c].header.bodyIndexA);
        enqueueBody(contacts[c].header.bodyIndexB);
      }
    }
    bool supported = componentContacts.size() >= 2;
    for (physx::PxU32 index = 0;
         index < componentContacts.size(); ++index) {
      supported =
          supported &&
          contacts[componentContacts[index]].restitution == 0.0f;
    }
    if (!supported)
      continue;

    const auto solveStartWorldPoint =
        [&](const AvbdContactConstraint &contact, bool actorA) {
          const physx::PxU32 bodyIndex =
              actorA ? contact.header.bodyIndexA
                     : contact.header.bodyIndexB;
          if (bodyIndex < numBodies) {
            const AvbdSolverBody &body = bodies[bodyIndex];
            return body.prevPosition +
                   body.prevRotation.rotate(
                       actorA ? contact.contactPointA
                              : contact.contactPointB);
          }
          return actorA ? contact.contactPointA
                        : contact.contactPointB;
        };
    std::sort(
        componentContacts.begin(), componentContacts.end(),
        [&](physx::PxU32 lhs, physx::PxU32 rhs) {
          const physx::PxVec3 lhsPoint =
              (solveStartWorldPoint(contacts[lhs], true) +
               solveStartWorldPoint(contacts[lhs], false)) *
              0.5f;
          const physx::PxVec3 rhsPoint =
              (solveStartWorldPoint(contacts[rhs], true) +
               solveStartWorldPoint(contacts[rhs], false)) *
              0.5f;
          if (lhsPoint.x != rhsPoint.x)
            return lhsPoint.x < rhsPoint.x;
          if (lhsPoint.y != rhsPoint.y)
            return lhsPoint.y < rhsPoint.y;
          if (lhsPoint.z != rhsPoint.z)
            return lhsPoint.z < rhsPoint.z;
          return lhs < rhs;
        });

    const physx::PxU32 contactCount = componentContacts.size();
    const physx::PxU32 rowCount = contactCount * 3;
    physx::PxArray<AvbdPassiveMaterialComponentRow> rows(rowCount);
    bool finite = true;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      const AvbdContactConstraint &contact =
          contacts[componentContacts[contactSlot]];
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      const physx::PxVec3 rA =
          bodyA < numBodies
              ? bodies[bodyA].prevRotation.rotate(contact.contactPointA)
              : physx::PxVec3(0.0f);
      const physx::PxVec3 rB =
          bodyB < numBodies
              ? bodies[bodyB].prevRotation.rotate(contact.contactPointB)
              : physx::PxVec3(0.0f);
      const physx::PxVec3 velocityA =
          bodyA < numBodies
              ? bodies[bodyA].linearVelocity +
                    bodies[bodyA].angularVelocity.cross(rA)
              : physx::PxVec3(0.0f);
      const physx::PxVec3 velocityB =
          bodyB < numBodies
              ? bodies[bodyB].linearVelocity +
                    bodies[bodyB].angularVelocity.cross(rB)
              : physx::PxVec3(0.0f);
      const physx::PxVec3 relativeVelocity =
          velocityA - velocityB;
      const physx::PxVec3 axes[3] = {
          contact.contactNormal, contact.tangent0, contact.tangent1};
      for (physx::PxU32 component = 0;
           component < 3; ++component) {
        AvbdPassiveMaterialComponentRow &row =
            rows[contactSlot * 3 + component];
        row.bodyA = bodyA;
        row.bodyB = bodyB;
        if (bodyA < numBodies) {
          row.linearA = axes[component];
          row.angularA = rA.cross(axes[component]);
        }
        if (bodyB < numBodies) {
          row.linearB = -axes[component];
          row.angularB = rB.cross(-axes[component]);
        }
        row.solveStartVelocity =
            relativeVelocity.dot(axes[component]);
        finite = finite &&
                 row.linearA.isFinite() &&
                 row.angularA.isFinite() &&
                 row.linearB.isFinite() &&
                 row.angularB.isFinite() &&
                 physx::PxIsFinite(row.solveStartVelocity);
      }
    }
    if (!finite)
      continue;

    physx::PxReal normalLipschitz = 0.0f;
    physx::PxReal tangentLipschitz = 0.0f;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      physx::PxReal absoluteRowSum = 0.0f;
      const physx::PxU32 rowComponent = row % 3;
      for (physx::PxU32 column = rowComponent == 0 ? 0 : 1;
           column < rowCount; column += 3) {
        if (rowComponent != 0) {
          absoluteRowSum += physx::PxAbs(
              passiveMaterialRowResponse(
                  rows[row], rows[column], bodies, numBodies));
          if (column + 1 < rowCount)
            absoluteRowSum += physx::PxAbs(
                passiveMaterialRowResponse(
                    rows[row], rows[column + 1],
                    bodies, numBodies));
        } else {
          absoluteRowSum += physx::PxAbs(
              passiveMaterialRowResponse(
                  rows[row], rows[column], bodies, numBodies));
        }
      }
      if (rowComponent == 0)
        normalLipschitz =
            physx::PxMax(normalLipschitz, absoluteRowSum);
      else
        tangentLipschitz =
            physx::PxMax(tangentLipschitz, absoluteRowSum);
    }
    if (!physx::PxIsFinite(normalLipschitz) ||
        !physx::PxIsFinite(tangentLipschitz) ||
        normalLipschitz <= 1.0e-12f ||
        tangentLipschitz <= 1.0e-12f)
      continue;

    physx::PxArray<physx::PxReal> impulses(rowCount);
    physx::PxArray<physx::PxReal> nextImpulses(rowCount);
    physx::PxArray<physx::PxReal> responseVelocity(rowCount);
    physx::PxArray<physx::PxVec3> bodyLinearImpulse(numBodies);
    physx::PxArray<physx::PxVec3> bodyAngularImpulse(numBodies);
    physx::PxArray<physx::PxVec3> bodyLinearDelta(numBodies);
    physx::PxArray<physx::PxVec3> bodyAngularDelta(numBodies);
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      impulses[row] = 0.0f;
      nextImpulses[row] = 0.0f;
      responseVelocity[row] = 0.0f;
    }

    const auto multiplyResponse =
        [&](const physx::PxArray<physx::PxReal> &input) {
          for (physx::PxU32 bodySlot = 0;
               bodySlot < bodyQueue.size(); ++bodySlot) {
            const physx::PxU32 body = bodyQueue[bodySlot];
            bodyLinearImpulse[body] = physx::PxVec3(0.0f);
            bodyAngularImpulse[body] = physx::PxVec3(0.0f);
          }
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            const AvbdPassiveMaterialComponentRow &materialRow =
                rows[row];
            const physx::PxReal impulse = input[row];
            if (materialRow.bodyA < numBodies) {
              bodyLinearImpulse[materialRow.bodyA] +=
                  materialRow.linearA * impulse;
              bodyAngularImpulse[materialRow.bodyA] +=
                  materialRow.angularA * impulse;
            }
            if (materialRow.bodyB < numBodies) {
              bodyLinearImpulse[materialRow.bodyB] +=
                  materialRow.linearB * impulse;
              bodyAngularImpulse[materialRow.bodyB] +=
                  materialRow.angularB * impulse;
            }
          }
          for (physx::PxU32 bodySlot = 0;
               bodySlot < bodyQueue.size(); ++bodySlot) {
            const physx::PxU32 body = bodyQueue[bodySlot];
            bodyLinearDelta[body] =
                bodyLinearImpulse[body] * bodies[body].invMass;
            bodyAngularDelta[body] =
                bodies[body].invInertiaWorld.transform(
                    bodyAngularImpulse[body]);
          }
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            const AvbdPassiveMaterialComponentRow &materialRow =
                rows[row];
            physx::PxReal value = 0.0f;
            if (materialRow.bodyA < numBodies) {
              value +=
                  bodyLinearDelta[materialRow.bodyA].dot(
                      materialRow.linearA) +
                  bodyAngularDelta[materialRow.bodyA].dot(
                      materialRow.angularA);
            }
            if (materialRow.bodyB < numBodies) {
              value +=
                  bodyLinearDelta[materialRow.bodyB].dot(
                      materialRow.linearB) +
                  bodyAngularDelta[materialRow.bodyB].dot(
                      materialRow.angularB);
            }
            responseVelocity[row] = value;
          }
        };

    const physx::PxReal normalStep = 1.0f / normalLipschitz;
    const physx::PxReal tangentStep = 1.0f / tangentLipschitz;
    for (physx::PxU32 outer = 0; outer < 64; ++outer) {
      physx::PxReal outerDelta = 0.0f;
      for (physx::PxU32 iteration = 0; iteration < 256; ++iteration) {
        multiplyResponse(impulses);
        physx::PxReal maximumDelta = 0.0f;
        for (physx::PxU32 contactSlot = 0;
             contactSlot < contactCount; ++contactSlot) {
          const physx::PxU32 row = contactSlot * 3;
          nextImpulses[row] = physx::PxMax(
              0.0f, impulses[row] -
                        normalStep *
                            (rows[row].solveStartVelocity +
                             responseVelocity[row]));
          maximumDelta = physx::PxMax(
              maximumDelta,
              physx::PxAbs(nextImpulses[row] - impulses[row]));
        }
        for (physx::PxU32 contactSlot = 0;
             contactSlot < contactCount; ++contactSlot) {
          const physx::PxU32 row = contactSlot * 3;
          outerDelta = physx::PxMax(
              outerDelta,
              physx::PxAbs(nextImpulses[row] - impulses[row]));
          impulses[row] = nextImpulses[row];
        }
        if (maximumDelta <= 1.0e-7f)
          break;
      }

      for (physx::PxU32 iteration = 0; iteration < 256; ++iteration) {
        multiplyResponse(impulses);
        physx::PxReal maximumDelta = 0.0f;
        for (physx::PxU32 contactSlot = 0;
             contactSlot < contactCount; ++contactSlot) {
          const physx::PxU32 row = contactSlot * 3;
          nextImpulses[row + 1] =
              impulses[row + 1] -
              tangentStep *
                  (rows[row + 1].solveStartVelocity +
                   responseVelocity[row + 1]);
          nextImpulses[row + 2] =
              impulses[row + 2] -
              tangentStep *
                  (rows[row + 2].solveStartVelocity +
                   responseVelocity[row + 2]);
          const physx::PxReal cap =
              contactCoulombMu(
                  contacts[componentContacts[contactSlot]]) *
              impulses[row];
          avbdProjectImpulseCone(
              cap, nextImpulses[row + 1],
              nextImpulses[row + 2]);
          maximumDelta = physx::PxMax(
              maximumDelta,
              physx::PxMax(
                  physx::PxAbs(
                      nextImpulses[row + 1] -
                      impulses[row + 1]),
                  physx::PxAbs(
                      nextImpulses[row + 2] -
                      impulses[row + 2])));
        }
        for (physx::PxU32 contactSlot = 0;
             contactSlot < contactCount; ++contactSlot) {
          const physx::PxU32 row = contactSlot * 3;
          outerDelta = physx::PxMax(
              outerDelta,
              physx::PxMax(
                  physx::PxAbs(
                      nextImpulses[row + 1] -
                      impulses[row + 1]),
                  physx::PxAbs(
                      nextImpulses[row + 2] -
                      impulses[row + 2])));
          impulses[row + 1] = nextImpulses[row + 1];
          impulses[row + 2] = nextImpulses[row + 2];
        }
        if (maximumDelta <= 1.0e-7f)
          break;
      }
      if (outerDelta <= 1.0e-6f)
        break;
    }

    multiplyResponse(impulses);
    physx::PxReal maximumResidual = 0.0f;
    physx::PxReal impulseScale = 1.0f;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      const physx::PxU32 row = contactSlot * 3;
      nextImpulses[row] = physx::PxMax(
          0.0f, impulses[row] -
                    normalStep *
                        (rows[row].solveStartVelocity +
                         responseVelocity[row]));
      nextImpulses[row + 1] =
          impulses[row + 1] -
          tangentStep *
              (rows[row + 1].solveStartVelocity +
               responseVelocity[row + 1]);
      nextImpulses[row + 2] =
          impulses[row + 2] -
          tangentStep *
              (rows[row + 2].solveStartVelocity +
               responseVelocity[row + 2]);
      const physx::PxReal cap =
          contactCoulombMu(
              contacts[componentContacts[contactSlot]]) *
          impulses[row];
      avbdProjectImpulseCone(
          cap, nextImpulses[row + 1],
          nextImpulses[row + 2]);
      for (physx::PxU32 component = 0;
           component < 3; ++component) {
        maximumResidual = physx::PxMax(
            maximumResidual,
            physx::PxAbs(
                nextImpulses[row + component] -
                impulses[row + component]));
        impulseScale = physx::PxMax(
            impulseScale,
            physx::PxAbs(impulses[row + component]));
        finite = finite &&
                 physx::PxIsFinite(impulses[row + component]);
      }
    }
    if (!finite ||
        maximumResidual > 1.0e-4f * impulseScale)
      continue;

    for (physx::PxU32 bodySlot = 0;
         bodySlot < bodyQueue.size(); ++bodySlot) {
      const physx::PxU32 body = bodyQueue[bodySlot];
      bodies[body].linearVelocity += bodyLinearDelta[body];
      bodies[body].angularVelocity += bodyAngularDelta[body];
    }
    const physx::PxReal invDt = 1.0f / dt;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      AvbdContactConstraint &contact =
          contacts[componentContacts[contactSlot]];
      const physx::PxU32 row = contactSlot * 3;
      const physx::PxReal normalImpulse = impulses[row];
      const physx::PxReal tangent0 = impulses[row + 1];
      const physx::PxReal tangent1 = impulses[row + 2];
      contact.header.lambda = -normalImpulse * invDt;
      contact.frictionSweepImpulse +=
          contact.tangent0 * tangent0 +
          contact.tangent1 * tangent1;
      const physx::PxReal tangentMagnitude =
          physx::PxSqrt(tangent0 * tangent0 +
                        tangent1 * tangent1);
      if (stats) {
        stats->contactTargetNormalProjectionRows++;
        stats->contactTargetTangentRows++;
        if (normalImpulse > 1.0e-8f) {
          stats->contactTargetNormalCorrections++;
          stats->contactTargetNormalImpulse += normalImpulse;
        }
        if (tangentMagnitude > 1.0e-8f) {
          stats->contactTargetTangentCorrections++;
          stats->contactTargetTangentImpulse += tangentMagnitude;
        }
      }
    }
  }
}

/**
 * Project a strict multi-point rigid-static friction manifold as one
 * material-velocity objective.
 *
 * The block has at most eight scalar tangent rows. Projected-gradient steps
 * update every row from the same iterate and project each contact's pair onto
 * its Coulomb disk. This is a simultaneous whole-manifold projection, not a
 * point-wise velocity Gauss-Seidel replay.
 */
static void applyAvbdContactMaterialFrictionManifolds(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, AvbdSolverStats *stats) {
  if (!bodies || !contacts || dt <= 0.0f)
    return;

  applyAvbdPassiveFrictionComponents(
      bodies, numBodies, contacts, numContacts,
      dt, stats);

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    AvbdSolverBody &body = bodies[bodyIndex];
    if (body.invMass <= 0.0f)
      continue;

    physx::PxU32 contactIndices[4] = {};
    physx::PxU32 contactCount = 0;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const AvbdContactConstraint &contact = contacts[c];
      if (!hasVelocityFrictionManifoldOwner(contact) ||
          hasVelocityPassiveFrictionComponentOwner(contact) ||
          (contact.header.bodyIndexA != bodyIndex &&
           contact.header.bodyIndexB != bodyIndex))
        continue;
      if (contactCount < 4)
        contactIndices[contactCount++] = c;
    }
    if (contactCount < 2 || contactCount > 4)
      continue;

    // Rebuild the inelastic normal response as a coupled nonnegative block.
    // Position AL has already resolved geometry; its multipliers must not be
    // replayed as material impulses. Starting from the inertial velocity here
    // makes the normal and tangent material objectives share one velocity
    // owner without importing pose-derived angular velocity.
    physx::PxVec3 normalAxes[4];
    physx::PxVec3 normalAngularJacobians[4];
    physx::PxReal normalRhs[4] = {};
    physx::PxReal normalResponse[4][4] = {};
    physx::PxReal normalImpulses[4] = {};
    physx::PxReal nextNormalImpulses[4] = {};
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      const AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      const bool dynamicIsA =
          contact.header.bodyIndexA == bodyIndex;
      const physx::PxReal dynamicSign = dynamicIsA ? 1.0f : -1.0f;
      const physx::PxVec3 localPoint =
          dynamicIsA ? contact.contactPointA : contact.contactPointB;
      const physx::PxVec3 arm =
          body.prevRotation.rotate(localPoint);
      normalAxes[contactSlot] =
          contact.contactNormal * dynamicSign;
      normalAngularJacobians[contactSlot] =
          arm.cross(normalAxes[contactSlot]);
      normalRhs[contactSlot] =
          -(body.linearVelocity + body.angularVelocity.cross(arm))
               .dot(normalAxes[contactSlot]);
    }
    physx::PxReal normalLipschitz = 0.0f;
    for (physx::PxU32 row = 0; row < contactCount; ++row) {
      physx::PxReal absoluteRowSum = 0.0f;
      for (physx::PxU32 column = 0; column < contactCount; ++column) {
        normalResponse[row][column] =
            body.invMass *
                normalAxes[row].dot(normalAxes[column]) +
            normalAngularJacobians[row].dot(
                body.invInertiaWorld.transform(
                    normalAngularJacobians[column]));
        absoluteRowSum +=
            physx::PxAbs(normalResponse[row][column]);
      }
      normalLipschitz =
          physx::PxMax(normalLipschitz, absoluteRowSum);
    }
    if (!physx::PxIsFinite(normalLipschitz) ||
        normalLipschitz <= 1.0e-12f)
      continue;
    const physx::PxReal normalStep = 1.0f / normalLipschitz;
    for (physx::PxU32 iteration = 0; iteration < 96; ++iteration) {
      for (physx::PxU32 row = 0; row < contactCount; ++row) {
        physx::PxReal gradient = -normalRhs[row];
        for (physx::PxU32 column = 0; column < contactCount; ++column)
          gradient += normalResponse[row][column] *
                      normalImpulses[column];
        nextNormalImpulses[row] = physx::PxMax(
            0.0f, normalImpulses[row] - normalStep * gradient);
      }
      for (physx::PxU32 row = 0; row < contactCount; ++row)
        normalImpulses[row] = nextNormalImpulses[row];
    }
    physx::PxVec3 normalLinearImpulse(0.0f);
    physx::PxVec3 normalAngularImpulse(0.0f);
    const physx::PxReal invDt = 1.0f / dt;
    for (physx::PxU32 row = 0; row < contactCount; ++row) {
      normalLinearImpulse += normalAxes[row] * normalImpulses[row];
      normalAngularImpulse +=
          normalAngularJacobians[row] * normalImpulses[row];
      contacts[contactIndices[row]].header.lambda =
          -normalImpulses[row] * invDt;
    }
    body.linearVelocity += normalLinearImpulse * body.invMass;
    body.angularVelocity +=
        body.invInertiaWorld.transform(normalAngularImpulse);

    const physx::PxU32 rowCount = contactCount * 2;
    physx::PxVec3 axes[8];
    physx::PxVec3 angularJacobians[8];
    physx::PxReal rhs[8] = {};
    physx::PxReal caps[4] = {};
    physx::PxReal response[8][8] = {};
    physx::PxReal impulses[8] = {};
    physx::PxReal nextImpulses[8] = {};

    bool supported = true;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount && supported; ++contactSlot) {
      AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      const bool dynamicIsA =
          contact.header.bodyIndexA == bodyIndex;
      const physx::PxReal dynamicSign = dynamicIsA ? 1.0f : -1.0f;
      const physx::PxVec3 localPoint =
          dynamicIsA ? contact.contactPointA : contact.contactPointB;
      const physx::PxVec3 arm =
          body.prevRotation.rotate(localPoint);
      const physx::PxVec3 pointVelocity =
          body.linearVelocity + body.angularVelocity.cross(arm);
      const physx::PxReal linearScale =
          dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
      const physx::PxReal angularScale =
          dynamicIsA ? contact.invInertiaScaleA
                     : contact.invInertiaScaleB;
      if (physx::PxAbs(linearScale - 1.0f) > 1.0e-6f ||
          physx::PxAbs(angularScale - 1.0f) > 1.0e-6f) {
        supported = false;
        break;
      }

      const physx::PxVec3 contactAxes[2] = {
          contact.tangent0, contact.tangent1};
      for (physx::PxU32 tangent = 0; tangent < 2; ++tangent) {
        const physx::PxU32 row = contactSlot * 2 + tangent;
        axes[row] = contactAxes[tangent] * dynamicSign;
        angularJacobians[row] = arm.cross(axes[row]);
        rhs[row] =
            contact.targetVelocity.dot(contactAxes[tangent]) -
            pointVelocity.dot(axes[row]);
      }
      caps[contactSlot] =
          contactCoulombMu(contact) *
          physx::PxMax(0.0f, -contact.header.lambda) * dt;
      if (!physx::PxIsFinite(caps[contactSlot]) ||
          caps[contactSlot] < 0.0f)
        supported = false;
    }
    if (!supported)
      continue;

    physx::PxReal lipschitz = 0.0f;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      physx::PxReal absoluteRowSum = 0.0f;
      for (physx::PxU32 column = 0; column < rowCount; ++column) {
        response[row][column] =
            body.invMass * axes[row].dot(axes[column]) +
            angularJacobians[row].dot(
                body.invInertiaWorld.transform(
                    angularJacobians[column]));
        absoluteRowSum += physx::PxAbs(response[row][column]);
      }
      lipschitz = physx::PxMax(lipschitz, absoluteRowSum);
    }
    if (!physx::PxIsFinite(lipschitz) || lipschitz <= 1.0e-12f)
      continue;

    const physx::PxReal step = 1.0f / lipschitz;
    for (physx::PxU32 iteration = 0; iteration < 96; ++iteration) {
      for (physx::PxU32 row = 0; row < rowCount; ++row) {
        physx::PxReal gradient = -rhs[row];
        for (physx::PxU32 column = 0; column < rowCount; ++column)
          gradient += response[row][column] * impulses[column];
        nextImpulses[row] = impulses[row] - step * gradient;
      }
      for (physx::PxU32 contactSlot = 0;
           contactSlot < contactCount; ++contactSlot) {
        const physx::PxU32 row = contactSlot * 2;
        avbdProjectImpulseCone(caps[contactSlot],
                               nextImpulses[row],
                               nextImpulses[row + 1]);
      }
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        impulses[row] = nextImpulses[row];
    }

    physx::PxVec3 linearImpulse(0.0f);
    physx::PxVec3 angularImpulse(0.0f);
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      const physx::PxU32 row = contactSlot * 2;
      linearImpulse += axes[row] * impulses[row] +
                       axes[row + 1] * impulses[row + 1];
      angularImpulse +=
          angularJacobians[row] * impulses[row] +
          angularJacobians[row + 1] * impulses[row + 1];
      contact.frictionSweepImpulse +=
          contact.tangent0 * impulses[row] +
          contact.tangent1 * impulses[row + 1];
      const physx::PxReal magnitude = physx::PxSqrt(
          impulses[row] * impulses[row] +
          impulses[row + 1] * impulses[row + 1]);
      if (stats) {
        stats->contactTargetTangentRows++;
        if (magnitude > 1.0e-8f) {
          stats->contactTargetTangentCorrections++;
          stats->contactTargetTangentImpulse += magnitude;
        }
      }
    }
    body.linearVelocity += linearImpulse * body.invMass;
    body.angularVelocity +=
        body.invInertiaWorld.transform(angularImpulse);
  }
}

/**
 * Consume PxContactModifyCallback target velocity after pose-to-velocity
 * reconstruction.  The projection uses the same contact-local inverse
 * mass/inertia scales as PhysX's impulse solvers and remains unilateral on
 * the normal row.
 */
static void applyAvbdContactTargetVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, AvbdSolverStats *stats) {
  if (!bodies || !contacts || dt <= 0.0f)
    return;

  applyAvbdContactMaterialFrictionManifolds(
      bodies, numBodies, contacts, numContacts,
      dt, stats);

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &cc = contacts[c];
    if (hasVelocityFrictionManifoldOwner(cc))
      continue;
    if (cc.targetVelocity.magnitudeSquared() <= 1e-12f)
      continue;

    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    const bool dynA = bA < numBodies && bodies[bA].invMass > 0.0f;
    const bool dynB = bB < numBodies && bodies[bB].invMass > 0.0f;
    if (!dynA && !dynB)
      continue;

    const bool solveStartTangentOwner =
        hasVelocityTangentTargetNormalSpan(cc);
    const physx::PxVec3 rA =
        dynA ? (solveStartTangentOwner ? bodies[bA].prevRotation
                                      : bodies[bA].rotation)
                   .rotate(cc.contactPointA)
             : physx::PxVec3(0.0f);
    const physx::PxVec3 rB =
        dynB ? (solveStartTangentOwner ? bodies[bB].prevRotation
                                      : bodies[bB].rotation)
                   .rotate(cc.contactPointB)
             : physx::PxVec3(0.0f);
    const physx::PxReal invMassA =
        dynA ? bodies[bA].invMass * cc.invMassScaleA : 0.0f;
    const physx::PxReal invMassB =
        dynB ? bodies[bB].invMass * cc.invMassScaleB : 0.0f;
    const physx::PxMat33 invInertiaA =
        dynA ? bodies[bA].invInertiaWorld * cc.invInertiaScaleA
             : physx::PxMat33(physx::PxZero);
    const physx::PxMat33 invInertiaB =
        dynB ? bodies[bB].invInertiaWorld * cc.invInertiaScaleB
             : physx::PxMat33(physx::PxZero);

    auto pointVelocity = [&](bool bodyA) {
      if (bodyA) {
        return dynA ? bodies[bA].linearVelocity +
                          bodies[bA].angularVelocity.cross(rA)
                    : physx::PxVec3(0.0f);
      }
      return dynB ? bodies[bB].linearVelocity +
                        bodies[bB].angularVelocity.cross(rB)
                  : physx::PxVec3(0.0f);
    };
    auto response = [&](const physx::PxVec3 &axis) {
      const physx::PxVec3 rAx = rA.cross(axis);
      const physx::PxVec3 rBx = rB.cross(axis);
      return invMassA + invMassB +
             rAx.dot(invInertiaA * rAx) +
             rBx.dot(invInertiaB * rBx);
    };
    auto applyImpulse = [&](const physx::PxVec3 &axis,
                            physx::PxReal impulse) {
      if (dynA) {
        bodies[bA].linearVelocity += axis * (impulse * invMassA);
        bodies[bA].angularVelocity +=
            invInertiaA * (rA.cross(axis) * impulse);
      }
      if (dynB) {
        bodies[bB].linearVelocity -= axis * (impulse * invMassB);
        bodies[bB].angularVelocity -=
            invInertiaB * (rB.cross(axis) * impulse);
      }
    };

    const physx::PxVec3 &normal = cc.contactNormal;
    physx::PxReal normalImpulse = 0.0f;
    const physx::PxReal normalResponse = response(normal);
    const bool ownedCombinedNormalTarget =
        hasVelocityTangentTargetOwner(cc) &&
        physx::PxAbs(cc.targetVelocity.dot(normal)) > 1.0e-6f;
    if ((!hasVelocityTangentTargetOwner(cc) ||
         ownedCombinedNormalTarget) &&
        normalResponse > 1e-12f) {
      if (stats)
        stats->contactTargetNormalProjectionRows++;
      const physx::PxReal currentNormal =
          (pointVelocity(true) - pointVelocity(false)).dot(normal);
      const physx::PxReal requestedNormal =
          cc.targetVelocity.dot(normal);
      const physx::PxReal deltaNormal =
          requestedNormal - currentNormal;
      if (deltaNormal > 0.0f) {
        normalImpulse = deltaNormal / normalResponse;
        if (cc.maxImpulse < PX_MAX_REAL) {
          const physx::PxReal existingImpulse =
              physx::PxMax(0.0f, -cc.header.lambda) * dt;
          normalImpulse = physx::PxMin(
              normalImpulse,
              physx::PxMax(0.0f, cc.maxImpulse - existingImpulse));
        }
        if (normalImpulse > 0.0f) {
          applyImpulse(normal, normalImpulse);
          if (stats) {
            stats->contactTargetNormalCorrections++;
            stats->contactTargetNormalImpulse += normalImpulse;
          }
        }
      }
    }

    const physx::PxReal targetT0 =
        cc.targetVelocity.dot(cc.tangent0);
    const physx::PxReal targetT1 =
        cc.targetVelocity.dot(cc.tangent1);
    if (physx::PxAbs(targetT0) <= 1e-6f &&
        physx::PxAbs(targetT1) <= 1e-6f)
      continue;
    if (stats)
      stats->contactTargetTangentRows++;

    const physx::PxReal mu = contactCoulombMu(cc);
    const physx::PxReal existingNormalSupport =
        physx::PxMax(0.0f, -cc.header.lambda) * dt;
    const physx::PxReal normalSupport =
        hasVelocityTangentTargetOwner(cc)
            ? existingNormalSupport + normalImpulse
            : physx::PxMax(normalImpulse, existingNormalSupport);
    const physx::PxReal tangentLimit = mu * normalSupport;
    if (tangentLimit <= 0.0f)
      continue;

    const physx::PxVec3 relativeVelocity =
        pointVelocity(true) - pointVelocity(false);
    const physx::PxReal responseT0 = response(cc.tangent0);
    const physx::PxReal responseT1 = response(cc.tangent1);
    physx::PxReal impulseT0 =
        responseT0 > 1e-12f
            ? (targetT0 - relativeVelocity.dot(cc.tangent0)) / responseT0
            : 0.0f;
    physx::PxReal impulseT1 =
        responseT1 > 1e-12f
            ? (targetT1 - relativeVelocity.dot(cc.tangent1)) / responseT1
            : 0.0f;
    avbdProjectImpulseCone(tangentLimit, impulseT0, impulseT1);
    applyImpulse(cc.tangent0, impulseT0);
    applyImpulse(cc.tangent1, impulseT1);
    cc.frictionSweepImpulse +=
        cc.tangent0 * impulseT0 + cc.tangent1 * impulseT1;
    const physx::PxReal tangentImpulse =
        physx::PxSqrt(impulseT0 * impulseT0 + impulseT1 * impulseT1);
    if (stats && tangentImpulse > 1e-8f) {
      stats->contactTargetTangentCorrections++;
      stats->contactTargetTangentImpulse += tangentImpulse;
    }
  }
}

static bool isRigidDeepBodyStaticRecoverySplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex, physx::PxReal lengthScale) {
  if (!bodies || bodyIndex >= numBodies || !contacts)
    return false;

  bool foundContact = false;
  physx::PxReal worstInitialViolation = PX_MAX_REAL;
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    if (bodyA != bodyIndex && bodyB != bodyIndex)
      continue;
    foundContact = true;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
        hasDeformableStaticAnchor(contact) || contact.friction > 0.0f ||
        contact.staticFriction > 0.0f || contact.restitution > 0.0f ||
        contact.targetVelocity.magnitudeSquared() > 1e-12f ||
        contact.maxImpulse < PX_MAX_REAL)
      return false;

    const bool dynamicIsA = bodyA == bodyIndex;
    const physx::PxReal linearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal angularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (!physx::PxIsFinite(linearScale) ||
        !physx::PxIsFinite(angularScale) || linearScale < 0.0f ||
        angularScale < 0.0f ||
        physx::PxAbs(linearScale - 1.0f) > 1e-6f ||
        physx::PxAbs(angularScale - 1.0f) > 1e-6f)
      return false;

    const physx::PxVec3 initialWorldA =
        dynamicIsA
            ? bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointA)
            : contact.staticPrevWorldPoint;
    const physx::PxVec3 initialWorldB =
        dynamicIsA
            ? contact.staticPrevWorldPoint
            : bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointB);
    const physx::PxReal initialViolation =
        (initialWorldA - initialWorldB).dot(contact.contactNormal) +
        contact.penetrationDepth;
    worstInitialViolation =
        physx::PxMin(worstInitialViolation, initialViolation);
  }
  return foundContact &&
         worstInitialViolation <
             -kBodyStaticNearSurface *
                 physx::PxMax(lengthScale, physx::PxReal(1e-6f));
}

static bool isRigidFiniteBodyStaticMaterialSplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex, physx::PxReal lengthScale) {
  if (!bodies || bodyIndex >= numBodies || !contacts)
    return false;

  bool foundContact = false;
  physx::PxU32 contactCount = 0;
  physx::PxReal manifoldLinearScale = 0.0f;
  physx::PxReal manifoldAngularScale = 0.0f;
  const physx::PxReal deepLimit =
      -kBodyStaticNearSurface *
      physx::PxMax(lengthScale, physx::PxReal(1.0e-6f));
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    if (bodyA != bodyIndex && bodyB != bodyIndex)
      continue;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
        hasDeformableStaticAnchor(contact) || contact.friction > 0.0f ||
        contact.staticFriction > 0.0f || contact.restitution < 0.0f ||
        contact.targetVelocity.magnitudeSquared() > 1.0e-12f ||
        contact.maxImpulse >= PX_MAX_REAL ||
        !physx::PxIsFinite(contact.maxImpulse) ||
        contact.maxImpulse < 0.0f)
      return false;

    const bool dynamicIsA = bodyA == bodyIndex;
    const physx::PxReal linearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal angularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (!physx::PxIsFinite(linearScale) ||
        !physx::PxIsFinite(angularScale) || linearScale < 0.0f ||
        angularScale < 0.0f)
      return false;
    if (!foundContact) {
      manifoldLinearScale = linearScale;
      manifoldAngularScale = angularScale;
    } else if (
        physx::PxAbs(linearScale - manifoldLinearScale) > 1.0e-6f ||
        physx::PxAbs(angularScale - manifoldAngularScale) > 1.0e-6f) {
      return false;
    }
    foundContact = true;
    ++contactCount;

    const physx::PxVec3 initialWorldA =
        dynamicIsA
            ? bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointA)
            : contact.staticPrevWorldPoint;
    const physx::PxVec3 initialWorldB =
        dynamicIsA
            ? contact.staticPrevWorldPoint
            : bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointB);
    const physx::PxReal initialViolation =
        (initialWorldA - initialWorldB).dot(contact.contactNormal) +
        contact.penetrationDepth;
    if (contact.contactManagerEstablished == 0 &&
        initialViolation < deepLimit)
      return false;
  }
  return foundContact && contactCount >= 1 && contactCount <= 4;
}

/**
 * Material normal-velocity response after pose finalize (friction already applied).
 * - Deformable: mesh-relative e=0 (heave).
 * - Rigid body-static: material restitution with scene bounce threshold.
 * - Dyn-dyn: same restitution on relative normal speed (linear mass split).
 * Friction mu is consumed elsewhere (dual cone + body-static friction post-pass).
 */
static bool applyBodyStaticRestitutionSpatialRow(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    physx::PxReal &linearDeltaMagnitude) {
  linearDeltaMagnitude = 0.0f;
  if (!linearVelAtSolveStart || !angularVelAtSolveStart ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies || dt <= 0.0f)
    return false;

  AvbdSolverBody &body = bodies[bodyIndex];
  const physx::PxReal invDt = 1.0f / dt;
  physx::PxVec3 aggregateNormal(0.0f);
  physx::PxVec3 aggregateAngularJacobian(0.0f);
  physx::PxReal aggregateApproach = 0.0f;
  physx::PxReal aggregateRestitution = 0.0f;
  physx::PxReal aggregateStaticNormalVelocity = 0.0f;
  physx::PxReal aggregateLinearScale = 0.0f;
  physx::PxReal aggregateAngularScale = 0.0f;
  physx::PxU32 rowCount = 0;

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies) ||
        hasDeformableStaticAnchor(cc) || (bA != bodyIndex && bB != bodyIndex) ||
        cc.restitution <= 0.0f || cc.maxImpulse < PX_MAX_REAL)
      continue;

    const bool dynIsA = bA == bodyIndex;
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);
    const physx::PxVec3 localPoint =
        dynIsA ? cc.contactPointA : cc.contactPointB;
    const physx::PxVec3 r0 = body.prevRotation.rotate(localPoint);
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 staticNow =
        dynIsA ? cc.contactPointB : cc.contactPointA;
    const physx::PxReal staticNormalVelocity =
        ((staticNow - cc.staticPrevWorldPoint) * invDt).dot(nd);
    const physx::PxReal solveStartPointVn =
        (*linearVelAtSolveStart)[bodyIndex].dot(nd) +
        (*angularVelAtSolveStart)[bodyIndex].dot(r0.cross(nd)) -
        staticNormalVelocity;
    const physx::PxReal approach =
        physx::PxMax(-solveStartPointVn, physx::PxReal(0.0f));
    if (approach <= bounceThreshold)
      continue;

    aggregateNormal += nd;
    aggregateAngularJacobian += r.cross(nd);
    aggregateApproach += approach;
    aggregateRestitution += physx::PxMin(cc.restitution, physx::PxReal(1.0f));
    aggregateStaticNormalVelocity += staticNormalVelocity;
    aggregateLinearScale += dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
    aggregateAngularScale +=
        dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
    ++rowCount;
  }

  if (rowCount == 0)
    return false;

  const physx::PxReal invRowCount = 1.0f / physx::PxReal(rowCount);
  aggregateNormal *= invRowCount;
  aggregateAngularJacobian *= invRowCount;
  aggregateApproach *= invRowCount;
  aggregateRestitution *= invRowCount;
  aggregateStaticNormalVelocity *= invRowCount;
  aggregateLinearScale *= invRowCount;
  aggregateAngularScale *= invRowCount;

  const physx::PxVec3 angularResponse =
      body.invInertiaWorld.transform(aggregateAngularJacobian) *
      aggregateAngularScale;
  const physx::PxReal response =
      body.invMass * aggregateLinearScale *
          aggregateNormal.magnitudeSquared() +
      aggregateAngularJacobian.dot(angularResponse);
  if (!physx::PxIsFinite(response) || response <= 1.0e-12f)
    return false;

  const physx::PxReal currentRelativeVn =
      body.linearVelocity.dot(aggregateNormal) +
      body.angularVelocity.dot(aggregateAngularJacobian) -
      aggregateStaticNormalVelocity;
  const physx::PxReal desiredRelativeVn =
      aggregateRestitution * aggregateApproach;
  const physx::PxReal impulse =
      (desiredRelativeVn - currentRelativeVn) / response;
  if (!physx::PxIsFinite(impulse))
    return false;
  if (impulse <= 1.0e-8f)
    return true;

  const physx::PxVec3 linearDelta =
      aggregateNormal * (impulse * body.invMass * aggregateLinearScale);
  body.linearVelocity += linearDelta;
  body.angularVelocity += angularResponse * impulse;
  linearDeltaMagnitude = linearDelta.magnitude();
  return true;
}

/**
 * Solve the free block of a finite-contact active set directly. P1I is
 * deliberately limited to at most four rows, so the whole manifold can be
 * solved as one deterministic objective instead of replaying point-wise
 * velocity Gauss-Seidel after the position solve.
 */
static bool solveFiniteContactFreeSystem(
    const physx::PxReal response[4][4], const physx::PxReal rhs[4],
    const physx::PxU32 freeRows[4], physx::PxU32 freeCount,
    physx::PxReal solution[4]) {
  physx::PxReal augmented[4][5] = {};
  for (physx::PxU32 row = 0; row < freeCount; ++row) {
    for (physx::PxU32 column = 0; column < freeCount; ++column) {
      augmented[row][column] =
          response[freeRows[row]][freeRows[column]];
    }
    augmented[row][freeCount] = rhs[row];
  }

  for (physx::PxU32 column = 0; column < freeCount; ++column) {
    physx::PxU32 pivot = column;
    physx::PxReal pivotMagnitude =
        physx::PxAbs(augmented[column][column]);
    for (physx::PxU32 row = column + 1; row < freeCount; ++row) {
      const physx::PxReal candidate =
          physx::PxAbs(augmented[row][column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!physx::PxIsFinite(pivotMagnitude) ||
        pivotMagnitude <= 1.0e-10f)
      return false;
    if (pivot != column) {
      for (physx::PxU32 entry = column; entry <= freeCount; ++entry) {
        const physx::PxReal temporary = augmented[column][entry];
        augmented[column][entry] = augmented[pivot][entry];
        augmented[pivot][entry] = temporary;
      }
    }

    const physx::PxReal inversePivot =
        1.0f / augmented[column][column];
    for (physx::PxU32 entry = column; entry <= freeCount; ++entry)
      augmented[column][entry] *= inversePivot;
    for (physx::PxU32 row = 0; row < freeCount; ++row) {
      if (row == column)
        continue;
      const physx::PxReal factor = augmented[row][column];
      for (physx::PxU32 entry = column; entry <= freeCount; ++entry)
        augmented[row][entry] -= factor * augmented[column][entry];
    }
  }

  for (physx::PxU32 row = 0; row < freeCount; ++row) {
    solution[freeRows[row]] = augmented[row][freeCount];
    if (!physx::PxIsFinite(solution[freeRows[row]]))
      return false;
  }
  return true;
}

static bool solveFiniteContactObjective(
    const physx::PxReal response[4][4], const physx::PxReal q[4],
    const physx::PxReal caps[4], physx::PxU32 rowCount,
    physx::PxReal impulses[4]) {
  // Enumerate lower/free/upper status for the bounded convex objective.
  // At four rows this is at most 3^4 = 81 direct candidates.
  physx::PxU32 statusCount = 1;
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    statusCount *= 3;

  bool found = false;
  physx::PxReal bestObjective = PX_MAX_REAL;
  for (physx::PxU32 encoded = 0; encoded < statusCount; ++encoded) {
    physx::PxU32 code = encoded;
    physx::PxU8 status[4] = {};
    physx::PxU32 freeRows[4] = {};
    physx::PxU32 freeCount = 0;
    physx::PxReal candidate[4] = {};
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      status[row] = static_cast<physx::PxU8>(code % 3);
      code /= 3;
      if (status[row] == 1)
        freeRows[freeCount++] = row;
      else if (status[row] == 2)
        candidate[row] = caps[row];
    }

    physx::PxReal rhs[4] = {};
    for (physx::PxU32 freeIndex = 0; freeIndex < freeCount; ++freeIndex) {
      const physx::PxU32 row = freeRows[freeIndex];
      rhs[freeIndex] = -q[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column) {
        if (status[column] == 2)
          rhs[freeIndex] -= response[row][column] * caps[column];
      }
    }
    if (freeCount > 0 &&
        !solveFiniteContactFreeSystem(
            response, rhs, freeRows, freeCount, candidate))
      continue;

    physx::PxReal scale = 1.0f;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      scale = physx::PxMax(scale, physx::PxAbs(q[row]));
    const physx::PxReal tolerance = 1.0e-5f * scale;
    bool valid = true;
    for (physx::PxU32 row = 0; row < rowCount && valid; ++row) {
      if (candidate[row] < -tolerance ||
          candidate[row] > caps[row] + tolerance) {
        valid = false;
        break;
      }
      candidate[row] = physx::PxClamp(
          candidate[row], physx::PxReal(0.0f), caps[row]);
      physx::PxReal gradient = q[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column)
        gradient += response[row][column] * candidate[column];
      if ((status[row] == 0 && gradient < -tolerance) ||
          (status[row] == 1 && physx::PxAbs(gradient) > tolerance) ||
          (status[row] == 2 && gradient > tolerance))
        valid = false;
    }
    if (!valid)
      continue;

    physx::PxReal objective = 0.0f;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      objective += q[row] * candidate[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column) {
        objective += 0.5f * candidate[row] *
                     response[row][column] * candidate[column];
      }
    }
    if (!physx::PxIsFinite(objective))
      continue;
    if (!found || objective < bestObjective) {
      found = true;
      bestObjective = objective;
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        impulses[row] = candidate[row];
    }
  }
  return found;
}

static bool applyBodyStaticFiniteSpatialBudget(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    physx::PxReal &linearDeltaMagnitude) {
  linearDeltaMagnitude = 0.0f;
  if (!linearVelAtSolveStart || !angularVelAtSolveStart ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies || dt <= 0.0f)
    return false;

  AvbdSolverBody &body = bodies[bodyIndex];
  const physx::PxReal invDt = 1.0f / dt;
  physx::PxU32 rowIndices[4] = {};
  physx::PxVec3 normals[4] = {};
  physx::PxVec3 angularJacobians[4] = {};
  physx::PxReal targets[4] = {};
  physx::PxReal staticNormalVelocities[4] = {};
  physx::PxReal caps[4] = {};
  physx::PxU32 rowCount = 0;
  physx::PxReal linearScale = 0.0f;
  physx::PxReal angularScale = 0.0f;

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &cc = contacts[c];
    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies) ||
        hasDeformableStaticAnchor(cc) || (bA != bodyIndex && bB != bodyIndex) ||
        cc.maxImpulse >= PX_MAX_REAL)
      continue;

    const bool dynIsA = bA == bodyIndex;
    const physx::PxReal rowLinearScale =
        dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
    const physx::PxReal rowAngularScale =
        dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);
    const physx::PxVec3 localPoint =
        dynIsA ? cc.contactPointA : cc.contactPointB;
    const physx::PxVec3 r0 = body.prevRotation.rotate(localPoint);
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 angularJacobian = r.cross(nd);
    const physx::PxReal cap =
        physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f));
    const physx::PxVec3 staticNow =
        dynIsA ? cc.contactPointB : cc.contactPointA;
    const physx::PxReal staticNormalVelocity =
        ((staticNow - cc.staticPrevWorldPoint) * invDt).dot(nd);
    const physx::PxReal solveStartPointVn =
        (*linearVelAtSolveStart)[bodyIndex].dot(nd) +
        (*angularVelAtSolveStart)[bodyIndex].dot(r0.cross(nd)) -
        staticNormalVelocity;
    const physx::PxReal approach =
        physx::PxMax(-solveStartPointVn, physx::PxReal(0.0f));
    const physx::PxVec3 initialWorldA =
        dynIsA
            ? body.prevPosition +
                  body.prevRotation.rotate(cc.contactPointA)
            : cc.staticPrevWorldPoint;
    const physx::PxVec3 initialWorldB =
        dynIsA
            ? cc.staticPrevWorldPoint
            : body.prevPosition +
                  body.prevRotation.rotate(cc.contactPointB);
    const physx::PxReal initialViolation =
        (initialWorldA - initialWorldB).dot(cc.contactNormal) +
        cc.penetrationDepth;
    // Match TGS impact eligibility: restitution is active only when the
    // solve-start point speed exceeds the scene threshold and the point will
    // close its current separation within this step.
    const bool collidingWithinStep =
        approach > initialViolation * invDt;
    const physx::PxReal restitution =
        cc.restitution > 0.0f && approach > bounceThreshold &&
                collidingWithinStep
            ? physx::PxMin(cc.restitution, physx::PxReal(1.0f))
            : physx::PxReal(0.0f);

    if (rowCount >= 4)
      return false;
    if (rowCount == 0) {
      linearScale = rowLinearScale;
      angularScale = rowAngularScale;
    }
    rowIndices[rowCount] = c;
    normals[rowCount] = nd;
    angularJacobians[rowCount] = angularJacobian;
    targets[rowCount] = restitution * approach;
    staticNormalVelocities[rowCount] = staticNormalVelocity;
    caps[rowCount] = cap;
    ++rowCount;
  }

  if (rowCount == 0)
    return false;

  physx::PxReal response[4][4] = {};
  physx::PxReal q[4] = {};
  physx::PxReal impulses[4] = {};
  physx::PxReal totalCap = 0.0f;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    const physx::PxReal currentRelativeVn =
        body.linearVelocity.dot(normals[row]) +
        body.angularVelocity.dot(angularJacobians[row]) -
        staticNormalVelocities[row];
    q[row] = currentRelativeVn - targets[row];
    totalCap += caps[row];
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      response[row][column] =
          body.invMass * linearScale *
              normals[row].dot(normals[column]) +
          angularJacobians[row].dot(
              body.invInertiaWorld.transform(
                  angularJacobians[column]) *
              angularScale);
    }
  }

  if (totalCap <= 1.0e-8f)
    return true;
  if (!solveFiniteContactObjective(
          response, q, caps, rowCount, impulses))
    return false;

  physx::PxVec3 linearImpulse(0.0f);
  physx::PxVec3 angularImpulse(0.0f);
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    linearImpulse += normals[row] * impulses[row];
    angularImpulse += angularJacobians[row] * impulses[row];
  }

  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    contacts[rowIndices[row]].header.lambda = -impulses[row] * invDt;
  }
  const physx::PxVec3 linearDelta =
      linearImpulse * (body.invMass * linearScale);
  body.linearVelocity += linearDelta;
  body.angularVelocity +=
      body.invInertiaWorld.transform(angularImpulse) * angularScale;
  linearDeltaMagnitude = linearDelta.magnitude();
  return true;
}

struct SurfaceFinalizeTopologyNode {
  physx::PxU32 parent;
  physx::PxU32 bodyCount;
  physx::PxU32 rowCount;
  physx::PxReal firstLinearScale;
  physx::PxReal firstAngularScale;
  physx::PxU8 strictOwner;
  physx::PxU8 bodyStrictOwner;
  physx::PxU8 legacyStrictOwner;
  physx::PxU8 restitution;
  physx::PxU8 finiteImpulse;
  physx::PxU8 targetVelocity;
  physx::PxU8 mixedScale;
  physx::PxU8 rigidStatic;
  physx::PxU8 nonOwnerDeformable;
  physx::PxU8 scaleSeen;
  physx::PxU8 lockedDof;
  physx::PxU8 nonDynamicBody;
  physx::PxU8 fastImpact;
  physx::PxU8 snapshotUnsupported;
  physx::PxU32 budgetDiagNoCorrectionRows;
  physx::PxU32 budgetDiagZeroBudgetRequiredRows;
  physx::PxU32 budgetDiagWithinBudgetRows;
  physx::PxU32 budgetDiagOverBudgetRows;
  physx::PxU32 budgetDiagUnsupportedRows;
};

struct SurfaceFinalizeBudgetDiagSnapshot {
  physx::PxReal outwardVelocity;
  physx::PxReal maximumImpulse;
  physx::PxU8 classification;
  physx::PxU8 fastImpact;
  physx::PxU8 unsupported;

  SurfaceFinalizeBudgetDiagSnapshot()
      : outwardVelocity(0.0f), maximumImpulse(0.0f),
        classification(0), fastImpact(0),
        unsupported(0) {}
};

enum SurfaceFinalizeBudgetDiagClass {
  eBUDGET_DIAG_NOT_APPLICABLE = 0,
  eBUDGET_DIAG_NO_CORRECTION,
  eBUDGET_DIAG_ZERO_BUDGET_REQUIRED,
  eBUDGET_DIAG_WITHIN_BUDGET,
  eBUDGET_DIAG_OVER_BUDGET,
  eBUDGET_DIAG_UNSUPPORTED
};

struct SurfaceFinalizeMatrixFreeRow {
  physx::PxU32 bodies[2];
  physx::PxVec3 axes[2];
  physx::PxVec3 angularJacobians[2];
};

struct SurfaceFinalizeDoubleVec3 {
  double x;
  double y;
  double z;

  SurfaceFinalizeDoubleVec3() : x(0.0), y(0.0), z(0.0) {}
};

static SurfaceFinalizeDoubleVec3
transformSurfaceFinalizeDouble(
    const physx::PxMat33 &matrix,
    const SurfaceFinalizeDoubleVec3 &value) {
  SurfaceFinalizeDoubleVec3 result;
  result.x = double(matrix.column0.x) * value.x +
             double(matrix.column1.x) * value.y +
             double(matrix.column2.x) * value.z;
  result.y = double(matrix.column0.y) * value.x +
             double(matrix.column1.y) * value.y +
             double(matrix.column2.y) * value.z;
  result.z = double(matrix.column0.z) * value.x +
             double(matrix.column1.z) * value.y +
             double(matrix.column2.z) * value.z;
  return result;
}

// Matrix-free equivalent of the dense J M^-1 J^T bounded solve.  It is used
// only as a backend choice for broad components; capability and KKT semantics
// do not depend on row count.  Until the unbounded feasibility classifier is
// also scalable, a converged bounded optimum with residual fails closed as
// ResidualUnclassified instead of being guessed as BudgetExhausted or
// mislabeled as a numerical fault.
static AvbdBoundedProjectionResult
solveSurfaceFinalizeMatrixFreeBoundedProjection(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    physx::PxU32 root,
    const AvbdContactConstraint *contacts,
    const physx::PxArray<physx::PxU32> &orderedRows,
    const physx::PxArray<double> &outward,
    const physx::PxArray<double> &upperBounds,
    double relativeTolerance = 1.0e-6,
    const physx::PxArray<double> *denseOracleResponse = NULL,
    bool *oracleOperatorMatched = NULL) {
  using namespace AvbdBoundedProjectionDetail;
  AvbdBoundedProjectionResult result;
  const physx::PxU32 rowCount = orderedRows.size();
  result.candidateImpulses.resize(rowCount, 0.0);
  result.commitImpulses.resize(rowCount, 0.0);
  if (oracleOperatorMatched)
    *oracleOperatorMatched = false;
  if (!bodies || !contacts || nodes.size() != numBodies ||
      rowCount == 0 || outward.size() != rowCount ||
      upperBounds.size() != rowCount ||
      !std::isfinite(relativeTolerance) || relativeTolerance <= 0.0)
    return result;

  physx::PxArray<SurfaceFinalizeMatrixFreeRow> rows(rowCount);
  double velocityScale = 1.0;
  double impulseScale = 1.0;
  double trace = 0.0;
  double maximumDiagonal = 0.0;
  bool needsCorrection = false;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (!std::isfinite(outward[row]) ||
        !std::isfinite(upperBounds[row]) || upperBounds[row] < 0.0)
      return result;
    needsCorrection = needsCorrection || outward[row] > 0.0;
    velocityScale = std::max(velocityScale, std::fabs(outward[row]));
    impulseScale = std::max(impulseScale, upperBounds[row]);
    const AvbdContactConstraint &contact = contacts[orderedRows[row]];
    SurfaceFinalizeMatrixFreeRow &operatorRow = rows[row];
    operatorRow.bodies[0] = contact.header.bodyIndexA;
    operatorRow.bodies[1] = contact.header.bodyIndexB;
    operatorRow.axes[0] = contact.contactNormal;
    operatorRow.axes[1] = -contact.contactNormal;
    const physx::PxVec3 localPoints[2] = {
        contact.contactPointA, contact.contactPointB};
    double diagonal = 0.0;
    for (physx::PxU32 end = 0; end < 2; ++end) {
      const physx::PxU32 body = operatorRow.bodies[end];
      operatorRow.angularJacobians[end] = physx::PxVec3(0.0f);
      if (body >= numBodies)
        continue;
      if (nodes[body].parent != root)
        return result;
      const physx::PxVec3 arm =
          bodies[body].rotation.rotate(localPoints[end]);
      operatorRow.angularJacobians[end] =
          arm.cross(operatorRow.axes[end]);
      const double linearResponse =
          double(bodies[body].invMass * nodes[body].firstLinearScale);
      const double angularResponse = double(
          operatorRow.angularJacobians[end].dot(
              bodies[body].invInertiaWorld.transform(
                  operatorRow.angularJacobians[end])) *
          nodes[body].firstAngularScale);
      diagonal += linearResponse + angularResponse;
    }
    if (!std::isfinite(diagonal) || diagonal < 0.0)
      return result;
    trace += diagonal;
    maximumDiagonal = std::max(maximumDiagonal, diagonal);
  }
  if (!needsCorrection) {
    result.status = eAVBD_BOUNDED_NO_CORRECTION;
    result.lowerRows = rowCount;
    return result;
  }
  if (!std::isfinite(trace) || trace <= 1.0e-14) {
    result.status = eAVBD_BOUNDED_INFEASIBLE;
    result.maximumResidual = velocityScale;
    return result;
  }

  physx::PxArray<SurfaceFinalizeDoubleVec3> linearImpulses(
      numBodies);
  physx::PxArray<SurfaceFinalizeDoubleVec3> angularImpulses(
      numBodies);
  physx::PxArray<SurfaceFinalizeDoubleVec3> linearResponses(
      numBodies);
  physx::PxArray<SurfaceFinalizeDoubleVec3> angularResponses(
      numBodies);
  const auto applyResponse =
      [&](const physx::PxArray<double> &impulses,
          physx::PxArray<double> &values) {
        std::fill(
            linearImpulses.begin(), linearImpulses.end(),
            SurfaceFinalizeDoubleVec3());
        std::fill(
            angularImpulses.begin(), angularImpulses.end(),
            SurfaceFinalizeDoubleVec3());
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          const double impulse = impulses[row];
          for (physx::PxU32 end = 0; end < 2; ++end) {
            const physx::PxU32 body = rows[row].bodies[end];
            if (body >= numBodies)
              continue;
            linearImpulses[body].x +=
                double(rows[row].axes[end].x) * impulse;
            linearImpulses[body].y +=
                double(rows[row].axes[end].y) * impulse;
            linearImpulses[body].z +=
                double(rows[row].axes[end].z) * impulse;
            angularImpulses[body].x +=
                double(rows[row].angularJacobians[end].x) * impulse;
            angularImpulses[body].y +=
                double(rows[row].angularJacobians[end].y) * impulse;
            angularImpulses[body].z +=
                double(rows[row].angularJacobians[end].z) * impulse;
          }
        }
        for (physx::PxU32 body = 0; body < numBodies; ++body) {
          if (nodes[body].parent != root) {
            linearResponses[body] = SurfaceFinalizeDoubleVec3();
            angularResponses[body] = SurfaceFinalizeDoubleVec3();
            continue;
          }
          const double linearScale =
              double(bodies[body].invMass) *
              double(nodes[body].firstLinearScale);
          linearResponses[body].x =
              linearImpulses[body].x * linearScale;
          linearResponses[body].y =
              linearImpulses[body].y * linearScale;
          linearResponses[body].z =
              linearImpulses[body].z * linearScale;
          angularResponses[body] =
              transformSurfaceFinalizeDouble(
                  bodies[body].invInertiaWorld,
                  angularImpulses[body]);
          const double angularScale =
              double(nodes[body].firstAngularScale);
          angularResponses[body].x *= angularScale;
          angularResponses[body].y *= angularScale;
          angularResponses[body].z *= angularScale;
        }
        values.resize(rowCount);
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          double value = 0.0;
          for (physx::PxU32 end = 0; end < 2; ++end) {
            const physx::PxU32 body = rows[row].bodies[end];
            if (body >= numBodies)
              continue;
            value +=
                double(rows[row].axes[end].x) *
                    linearResponses[body].x +
                double(rows[row].axes[end].y) *
                    linearResponses[body].y +
                double(rows[row].axes[end].z) *
                    linearResponses[body].z +
                double(rows[row].angularJacobians[end].x) *
                    angularResponses[body].x +
                double(rows[row].angularJacobians[end].y) *
                    angularResponses[body].y +
                double(rows[row].angularJacobians[end].z) *
                    angularResponses[body].z;
          }
          values[row] = value;
        }
      };
  if (oracleOperatorMatched && denseOracleResponse &&
      denseOracleResponse->size() == rowCount * rowCount) {
    physx::PxArray<double> oracleImpulses(rowCount, 0.0);
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      oracleImpulses[row] =
          0.125 + double((row * 17 + 3) % 29) / 29.0;
    physx::PxArray<double> matrixFreeValues;
    applyResponse(oracleImpulses, matrixFreeValues);
    double maximumDelta = 0.0;
    double responseScale = 1.0;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      double denseValue = 0.0;
      for (physx::PxU32 column = 0; column < rowCount; ++column)
        denseValue +=
            (*denseOracleResponse)[row * rowCount + column] *
            oracleImpulses[column];
      if (!std::isfinite(denseValue) ||
          !std::isfinite(matrixFreeValues[row])) {
        maximumDelta = PX_MAX_F64;
        break;
      }
      responseScale =
          std::max(responseScale, std::fabs(denseValue));
      responseScale =
          std::max(responseScale, std::fabs(matrixFreeValues[row]));
      maximumDelta = std::max(
          maximumDelta,
          std::fabs(denseValue - matrixFreeValues[row]));
    }
    *oracleOperatorMatched =
        maximumDelta <= 8.0e-6 * responseScale;
  }
  const double feasibilityTolerance =
      relativeTolerance * velocityScale;
  const double boundTolerance =
      relativeTolerance * impulseScale;
  result.projectedGradientTolerance = feasibilityTolerance;
  double lipschitzBound = maximumDiagonal;
  physx::PxArray<double> impulses(rowCount, 0.0);
  physx::PxArray<double> extrapolated(rowCount, 0.0);
  physx::PxArray<double> next(rowCount, 0.0);
  physx::PxArray<double> responseValues;
  physx::PxArray<double> gradientValues(rowCount, 0.0);
  physx::PxArray<double> baseResponse;
  double acceleration = 1.0;
  double currentObjective = 0.0;
  const physx::PxU32 iterationLimit =
      physx::PxMax(
          physx::PxU32(4096),
          physx::PxU32(1024 + 128 * nodes[root].bodyCount));
  bool converged = false;
  const auto takeProjectedStep =
      [&](const physx::PxArray<double> &base,
          physx::PxArray<double> &candidate,
          physx::PxArray<double> &candidateResponse,
          double &candidateObjective) {
        applyResponse(base, baseResponse);
        double baseObjective = 0.0;
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          gradientValues[row] =
              baseResponse[row] - outward[row];
          baseObjective +=
              0.5 * base[row] * baseResponse[row] -
              outward[row] * base[row];
        }
        if (!std::isfinite(baseObjective))
          return false;
        for (;;) {
          const double inverseLipschitz =
              1.0 / lipschitzBound;
          double gradientStep = 0.0;
          double stepNormSquared = 0.0;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            candidate[row] = std::min(
                upperBounds[row],
                std::max(
                    0.0,
                    base[row] -
                        inverseLipschitz *
                            gradientValues[row]));
            const double delta =
                candidate[row] - base[row];
            gradientStep += gradientValues[row] * delta;
            stepNormSquared += delta * delta;
          }
          applyResponse(candidate, candidateResponse);
          candidateObjective = 0.0;
          for (physx::PxU32 row = 0; row < rowCount; ++row)
            candidateObjective +=
                0.5 * candidate[row] *
                    candidateResponse[row] -
                outward[row] * candidate[row];
          const double modelObjective =
              baseObjective + gradientStep +
              0.5 * lipschitzBound * stepNormSquared;
          const double modelSlack =
              1.0e-13 *
              std::max(
                  1.0,
                  std::max(
                      std::fabs(candidateObjective),
                      std::fabs(modelObjective)));
          if (std::isfinite(candidateObjective) &&
              std::isfinite(modelObjective) &&
              candidateObjective <=
                  modelObjective + modelSlack)
            return true;
          lipschitzBound *= 2.0;
          if (!std::isfinite(lipschitzBound))
            return false;
        }
      };
  for (physx::PxU32 iteration = 0;
       iteration < iterationLimit; ++iteration) {
    double nextObjective = 0.0;
    if (!takeProjectedStep(
            extrapolated, next, responseValues,
            nextObjective))
      return result;
    const double objectiveSlack =
        1.0e-13 * std::max(1.0, std::fabs(currentObjective));
    if (nextObjective > currentObjective + objectiveSlack) {
      extrapolated = impulses;
      acceleration = 1.0;
      if (!takeProjectedStep(
              extrapolated, next, responseValues,
              nextObjective) ||
          nextObjective > currentObjective + 16.0 * objectiveSlack)
        return result;
    }
    impulses.swap(next);
    currentObjective = nextObjective;
    result.iterations = iteration + 1;
    applyResponse(impulses, responseValues);
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      gradientValues[row] = responseValues[row] - outward[row];
    if (projectedGradientViolation(
            gradientValues, impulses, upperBounds, boundTolerance) <=
        feasibilityTolerance) {
      converged = true;
      break;
    }
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(
                         1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        (acceleration - 1.0) / nextAcceleration;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      extrapolated[row] =
          impulses[row] + momentum * (impulses[row] - next[row]);
    acceleration = nextAcceleration;
    if ((iteration + 1) % 64 == 0) {
      extrapolated = impulses;
      acceleration = 1.0;
    }
  }
  if (!converged) {
    result.maximumKktViolation = projectedGradientViolation(
        gradientValues, impulses, upperBounds, boundTolerance);
    result.status = eAVBD_BOUNDED_ITERATION_LIMIT;
    return result;
  }

  result.maximumKktViolation = projectedGradientViolation(
      gradientValues, impulses, upperBounds, boundTolerance);
  double maximumResidual = 0.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (upperBounds[row] <= boundTolerance ||
        upperBounds[row] - impulses[row] <= boundTolerance)
      ++result.upperRows;
    else if (impulses[row] <= boundTolerance)
      ++result.lowerRows;
    else
      ++result.freeRows;
    maximumResidual =
        std::max(maximumResidual, -gradientValues[row]);
  }
  result.maximumResidual = maximumResidual;
  result.candidateImpulses = impulses;
  if (!std::isfinite(maximumResidual) ||
      maximumResidual > 4.0 * feasibilityTolerance) {
    result.status = std::isfinite(maximumResidual)
                        ? eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED
                        : eAVBD_BOUNDED_NUMERICAL_FAILURE;
    return result;
  }
  result.commitImpulses = result.candidateImpulses;
  result.status = eAVBD_BOUNDED_SOLVED;
  return result;
}

static bool compareSurfaceFinalizeMatrixFreeOracle(
    const AvbdBoundedProjectionResult &dense,
    const AvbdBoundedProjectionResult &matrixFree,
    const physx::PxArray<double> &response,
    const physx::PxArray<double> &outward,
    bool operatorMatched, bool &comparable) {
  comparable = true;
  bool statusMatched = false;
  switch (dense.status) {
  case eAVBD_BOUNDED_SOLVED:
    statusMatched = matrixFree.status == eAVBD_BOUNDED_SOLVED;
    break;
  case eAVBD_BOUNDED_NO_CORRECTION:
    statusMatched =
        matrixFree.status == eAVBD_BOUNDED_NO_CORRECTION;
    return statusMatched;
  case eAVBD_BOUNDED_BUDGET_EXHAUSTED:
    statusMatched =
        matrixFree.status == eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED;
    break;
  case eAVBD_BOUNDED_INFEASIBLE:
    statusMatched =
        matrixFree.status == eAVBD_BOUNDED_INFEASIBLE ||
        matrixFree.status == eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED;
    break;
  default:
    comparable = false;
    return false;
  }
  if (!statusMatched || !operatorMatched)
    return false;

  const physx::PxU32 rowCount = outward.size();
  if (response.size() != rowCount * rowCount ||
      dense.candidateImpulses.size() != rowCount ||
      matrixFree.candidateImpulses.size() != rowCount)
    return false;
  double velocityScale = 1.0;
  double maximumResponseDelta = 0.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    velocityScale =
        std::max(velocityScale, std::fabs(outward[row]));
    double denseValue = 0.0;
    double matrixFreeValue = 0.0;
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      const double entry = response[row * rowCount + column];
      denseValue += entry * dense.candidateImpulses[column];
      matrixFreeValue +=
          entry * matrixFree.candidateImpulses[column];
    }
    if (!std::isfinite(denseValue) ||
        !std::isfinite(matrixFreeValue))
      return false;
    maximumResponseDelta = std::max(
        maximumResponseDelta,
        std::fabs(denseValue - matrixFreeValue));
  }
  return maximumResponseDelta <= 32.0e-6 * velocityScale;
}

static bool isSurfaceFinalizeContactNear(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint &contact);

static SurfaceFinalizeBudgetDiagSnapshot
classifySurfaceFinalizeBudgetDiag(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint &contact, physx::PxReal dt,
    physx::PxReal lengthScale,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart) {
  SurfaceFinalizeBudgetDiagSnapshot snapshot;
  if (!bodies || dt <= 0.0f ||
      !isSurfaceFinalizeContactNear(bodies, numBodies, contact))
    return snapshot;
  if (contact.restitution != 0.0f ||
      contact.targetVelocity.magnitudeSquared() > 1.0e-12f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  const bool dynamicA = bodyA < numBodies;
  const bool dynamicB = bodyB < numBodies;
  if (!dynamicA && !dynamicB)
    return snapshot;
  if ((dynamicA && (bodies[bodyA].invMass <= 0.0f ||
                    bodies[bodyA].lockFlags != 0)) ||
      (dynamicB && (bodies[bodyB].invMass <= 0.0f ||
                    bodies[bodyB].lockFlags != 0))) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const auto pointVelocity =
      [&](physx::PxU32 body, const physx::PxVec3 &localPoint) {
        if (body >= numBodies)
          return physx::PxVec3(0.0f);
        const physx::PxVec3 arm =
            bodies[body].rotation.rotate(localPoint);
        return bodies[body].linearVelocity +
               bodies[body].angularVelocity.cross(arm);
      };
  physx::PxVec3 velocityA =
      pointVelocity(bodyA, contact.contactPointA);
  physx::PxVec3 velocityB =
      pointVelocity(bodyB, contact.contactPointB);
  if (isBodyVsStaticContact(bodyA, bodyB, numBodies) &&
      hasDeformableStaticAnchor(contact)) {
    const physx::PxVec3 staticNow =
        dynamicA ? contact.contactPointB : contact.contactPointA;
    const physx::PxVec3 staticStep =
        staticNow - contact.staticPrevWorldPoint;
    const physx::PxReal aliasCap =
        AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
    if (!staticStep.isFinite() ||
        staticStep.magnitudeSquared() > aliasCap * aliasCap) {
      snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
      snapshot.unsupported = 1;
      return snapshot;
    }
    const physx::PxVec3 staticVelocity = staticStep / dt;
    if (dynamicA)
      velocityB = staticVelocity;
    else
      velocityA = staticVelocity;
  }

  const physx::PxReal outwardVelocity =
      (velocityA - velocityB).dot(contact.contactNormal);
  snapshot.outwardVelocity = outwardVelocity;
  const physx::PxReal velocityTolerance =
      1.0e-5f *
      physx::PxMax(lengthScale, physx::PxReal(1.0e-6f)) / dt;
  if (!physx::PxIsFinite(outwardVelocity)) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const bool haveSolveStart =
      linearVelAtSolveStart && angularVelAtSolveStart &&
      linearVelAtSolveStart->size() == numBodies &&
      angularVelAtSolveStart->size() == numBodies;
  if (haveSolveStart) {
    const auto solveStartPointVelocity =
        [&](physx::PxU32 body, const physx::PxVec3 &localPoint) {
          if (body >= numBodies)
            return physx::PxVec3(0.0f);
          const physx::PxVec3 arm =
              bodies[body].rotation.rotate(localPoint);
          return (*linearVelAtSolveStart)[body] +
                 (*angularVelAtSolveStart)[body].cross(arm);
        };
    physx::PxVec3 solveStartA =
        solveStartPointVelocity(bodyA, contact.contactPointA);
    physx::PxVec3 solveStartB =
        solveStartPointVelocity(bodyB, contact.contactPointB);
    if (isBodyVsStaticContact(bodyA, bodyB, numBodies) &&
        hasDeformableStaticAnchor(contact)) {
      const physx::PxVec3 staticNow =
          dynamicA ? contact.contactPointB : contact.contactPointA;
      const physx::PxVec3 staticVelocity =
          (staticNow - contact.staticPrevWorldPoint) / dt;
      if (dynamicA)
        solveStartB = staticVelocity;
      else
        solveStartA = staticVelocity;
    }
    const physx::PxReal solveStartRelative =
        (solveStartA - solveStartB).dot(contact.contactNormal);
    if (!physx::PxIsFinite(solveStartRelative)) {
      snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
      snapshot.unsupported = 1;
      return snapshot;
    }
    snapshot.fastImpact =
        -solveStartRelative > kBodyStaticFastImpactSpeed ? 1 : 0;
  }

  physx::PxReal budget =
      physx::PxMax(-contact.header.lambda, physx::PxReal(0.0f)) * dt;
  if (contact.maxImpulse < PX_MAX_REAL)
    budget = physx::PxMin(
        budget, physx::PxMax(contact.maxImpulse, physx::PxReal(0.0f)));
  snapshot.maximumImpulse = budget;
  if (!physx::PxIsFinite(budget) || budget < 0.0f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }
  if (outwardVelocity <= velocityTolerance) {
    snapshot.classification = eBUDGET_DIAG_NO_CORRECTION;
    return snapshot;
  }

  physx::PxReal response = 0.0f;
  const auto addResponse =
      [&](physx::PxU32 body, const physx::PxVec3 &localPoint,
          const physx::PxVec3 &axis, physx::PxReal linearScale,
          physx::PxReal angularScale) {
        if (body >= numBodies)
          return true;
        if (!physx::PxIsFinite(linearScale) ||
            !physx::PxIsFinite(angularScale) || linearScale < 0.0f ||
            angularScale < 0.0f)
          return false;
        const physx::PxVec3 arm =
            bodies[body].rotation.rotate(localPoint);
        const physx::PxVec3 angularJacobian = arm.cross(axis);
        response += bodies[body].invMass * linearScale +
                    angularJacobian.dot(
                        bodies[body].invInertiaWorld.transform(
                            angularJacobian)) *
                        angularScale;
        return true;
      };
  if (!addResponse(bodyA, contact.contactPointA,
                   contact.contactNormal, contact.invMassScaleA,
                   contact.invInertiaScaleA) ||
      !addResponse(bodyB, contact.contactPointB,
                   -contact.contactNormal, contact.invMassScaleB,
                   contact.invInertiaScaleB) ||
      !physx::PxIsFinite(response) || response <= 1.0e-12f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const physx::PxReal requiredImpulse = outwardVelocity / response;
  if (!physx::PxIsFinite(requiredImpulse) ||
      !physx::PxIsFinite(budget) || budget < 0.0f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }
  if (budget <= 1.0e-8f) {
    snapshot.classification = eBUDGET_DIAG_ZERO_BUDGET_REQUIRED;
    return snapshot;
  }
  const physx::PxReal impulseTolerance =
      1.0e-6f *
      physx::PxMax(physx::PxReal(1.0f),
                   physx::PxMax(requiredImpulse, budget));
  snapshot.classification = physx::PxU8(
      requiredImpulse <= budget + impulseTolerance
          ? eBUDGET_DIAG_WITHIN_BUDGET
          : eBUDGET_DIAG_OVER_BUDGET);
  return snapshot;
}

static bool isSurfaceFinalizeContactNear(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint &contact) {
  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  if (bodyA >= numBodies && bodyB >= numBodies)
    return false;
  const physx::PxVec3 worldA =
      bodyA < numBodies
          ? bodies[bodyA].position +
                bodies[bodyA].rotation.rotate(contact.contactPointA)
          : contact.contactPointA;
  const physx::PxVec3 worldB =
      bodyB < numBodies
          ? bodies[bodyB].position +
                bodies[bodyB].rotation.rotate(contact.contactPointB)
          : contact.contactPointB;
  physx::PxReal violation =
      (worldA - worldB).dot(contact.contactNormal) +
      contact.penetrationDepth;
  if (isBodyVsStaticContact(bodyA, bodyB, numBodies) &&
      hasDeformableStaticAnchor(contact)) {
    violation = finalizeBodyVsStaticViolation(
        violation, contact.penetrationDepth);
  }
  return physx::PxIsFinite(violation) &&
         violation < kBodyStaticNearSurface;
}

// Discover the strict P3E/P3K owner before P3K mutates velocity.  This is the
// extracted control predicate of the legacy manifold diagnostic below:
// dominant deformable/static contact, near-surface capability, non-fast
// solve-start COM approach, and at least one near position-tangent-owned row.
// Keeping the legacy marker separately lets the hidden gate prove exact
// equivalence before any production owner replacement is attempted.
static void discoverSurfaceFinalizeStrictOwnersPreP3K(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    physx::PxArray<SurfaceFinalizeTopologyNode> &nodes) {
  if (!bodies || !contacts || nodes.size() != numBodies)
    return;

  const bool haveSolveStart =
      linearVelAtSolveStart &&
      linearVelAtSolveStart->size() == numBodies;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    nodes[body].bodyStrictOwner = 0;
    if (bodies[body].invMass <= 0.0f)
      continue;

    physx::PxU32 dominant = PX_MAX_U32;
    physx::PxReal worstViolation = PX_MAX_REAL;
    for (physx::PxU32 row = 0; row < numContacts; ++row) {
      const AvbdContactConstraint &contact = contacts[row];
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
          (bodyA != body && bodyB != body))
        continue;
      const bool dynamicIsA = bodyA == body;
      const physx::PxVec3 worldA =
          dynamicIsA
              ? bodies[body].position +
                    bodies[body].rotation.rotate(contact.contactPointA)
              : contact.contactPointA;
      const physx::PxVec3 worldB =
          dynamicIsA
              ? contact.contactPointB
              : bodies[body].position +
                    bodies[body].rotation.rotate(contact.contactPointB);
      physx::PxReal violation =
          (worldA - worldB).dot(contact.contactNormal) +
          contact.penetrationDepth;
      if (hasDeformableStaticAnchor(contact))
        violation = finalizeBodyVsStaticViolation(
            violation, contact.penetrationDepth);
      if (violation < worstViolation) {
        worstViolation = violation;
        dominant = row;
      }
    }
    if (dominant == PX_MAX_U32)
      continue;

    const AvbdContactConstraint &dominantContact = contacts[dominant];
    if (!hasDeformableStaticAnchor(dominantContact) ||
        worstViolation >= kBodyStaticNearSurface)
      continue;
    if (haveSolveStart) {
      const bool dynamicIsA =
          dominantContact.header.bodyIndexA == body;
      const physx::PxVec3 outwardNormal =
          dominantContact.contactNormal *
          (dynamicIsA ? 1.0f : -1.0f);
      const physx::PxReal approach =
          -(*linearVelAtSolveStart)[body].dot(outwardNormal);
      if (approach > kBodyStaticFastImpactSpeed)
        continue;
    }

    for (physx::PxU32 row = 0; row < numContacts; ++row) {
      const AvbdContactConstraint &contact = contacts[row];
      if ((contact.header.bodyIndexA != body &&
           contact.header.bodyIndexB != body) ||
          !isBodyVsStaticContact(
              contact.header.bodyIndexA, contact.header.bodyIndexB,
              numBodies) ||
          !hasDeformableStaticAnchor(contact) ||
          !hasDeformablePositionTangentOwner(contact) ||
          !isSurfaceFinalizeContactNear(bodies, numBodies, contact))
        continue;
      nodes[body].bodyStrictOwner = 1;
      break;
    }
  }
}

static physx::PxU32 findFinalizeComponentRoot(
    physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    physx::PxU32 body) {
  physx::PxU32 root = body;
  while (nodes[root].parent != root)
    root = nodes[root].parent;
  while (nodes[body].parent != body) {
    const physx::PxU32 next = nodes[body].parent;
    nodes[body].parent = root;
    body = next;
  }
  return root;
}

static void recordSurfaceDeformableFinalizeComponentTopology(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    const physx::PxArray<SurfaceFinalizeBudgetDiagSnapshot>
        &budgetDiagSnapshots,
    bool hasJointConstraints, bool enableProductionProbe,
    bool enableMatrixFreeOracle,
    physx::PxArray<bool> &probeOwnedBodies,
    AvbdSolverStats *stats) {
  if (!stats || nodes.size() != numBodies || numBodies == 0)
    return;

  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    nodes[body].parent = body;
    nodes[body].bodyCount = 0;
    nodes[body].rowCount = 0;
    nodes[body].firstLinearScale = 0.0f;
    nodes[body].firstAngularScale = 0.0f;
    nodes[body].restitution = 0;
    nodes[body].finiteImpulse = 0;
    nodes[body].targetVelocity = 0;
    nodes[body].mixedScale = 0;
    nodes[body].rigidStatic = 0;
    nodes[body].nonOwnerDeformable = 0;
    nodes[body].scaleSeen = 0;
    nodes[body].lockedDof = 0;
    nodes[body].nonDynamicBody = 0;
    nodes[body].fastImpact = 0;
    nodes[body].snapshotUnsupported = 0;
    nodes[body].budgetDiagNoCorrectionRows = 0;
    nodes[body].budgetDiagZeroBudgetRequiredRows = 0;
    nodes[body].budgetDiagWithinBudgetRows = 0;
    nodes[body].budgetDiagOverBudgetRows = 0;
    nodes[body].budgetDiagUnsupportedRows = 0;
  }
  physx::PxArray<physx::PxArray<physx::PxU32> > componentRows(
      numBodies);
  for (physx::PxU32 row = 0; row < numContacts; ++row) {
    if (!isSurfaceFinalizeContactNear(
            bodies, numBodies, contacts[row]))
      continue;
    const physx::PxU32 bodyA = contacts[row].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[row].header.bodyIndexB;
    if (bodyA >= numBodies || bodyB >= numBodies)
      continue;
    const physx::PxU32 rootA =
        findFinalizeComponentRoot(nodes, bodyA);
    const physx::PxU32 rootB =
        findFinalizeComponentRoot(nodes, bodyB);
    if (rootA != rootB)
      nodes[rootB].parent = rootA;
  }
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    nodes[body].parent = findFinalizeComponentRoot(nodes, body);

  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 root = nodes[body].parent;
    ++nodes[root].bodyCount;
    if (nodes[body].bodyStrictOwner)
      nodes[root].strictOwner = 1;
    if (bodies[body].lockFlags != 0)
      nodes[root].lockedDof = 1;
    if (bodies[body].invMass <= 0.0f)
      nodes[root].nonDynamicBody = 1;
  }

  for (physx::PxU32 row = 0; row < numContacts; ++row) {
    const AvbdContactConstraint &contact = contacts[row];
    if (!isSurfaceFinalizeContactNear(
            bodies, numBodies, contact))
      continue;
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    const physx::PxU32 dynamicBody =
        bodyA < numBodies ? bodyA : bodyB;
    if (dynamicBody >= numBodies)
      continue;
    const physx::PxU32 root = nodes[dynamicBody].parent;
    if (!nodes[root].strictOwner)
      continue;

    ++nodes[root].rowCount;
    componentRows[root].pushBack(row);
    const SurfaceFinalizeBudgetDiagSnapshot snapshot =
        row < budgetDiagSnapshots.size()
            ? budgetDiagSnapshots[row]
            : SurfaceFinalizeBudgetDiagSnapshot();
    const physx::PxU8 budgetClass =
        row < budgetDiagSnapshots.size()
            ? snapshot.classification
            : physx::PxU8(eBUDGET_DIAG_UNSUPPORTED);
    if (snapshot.fastImpact)
      nodes[root].fastImpact = 1;
    if (snapshot.unsupported ||
        budgetClass == eBUDGET_DIAG_UNSUPPORTED)
      nodes[root].snapshotUnsupported = 1;
    ++stats->surfaceDeformableFinalizeBudgetDiagRows;
    switch (budgetClass) {
    case eBUDGET_DIAG_NO_CORRECTION:
      ++nodes[root].budgetDiagNoCorrectionRows;
      ++stats->surfaceDeformableFinalizeBudgetDiagNoCorrectionRows;
      break;
    case eBUDGET_DIAG_ZERO_BUDGET_REQUIRED:
      ++nodes[root].budgetDiagZeroBudgetRequiredRows;
      ++stats
            ->surfaceDeformableFinalizeBudgetDiagZeroBudgetRequiredRows;
      break;
    case eBUDGET_DIAG_WITHIN_BUDGET:
      ++nodes[root].budgetDiagWithinBudgetRows;
      ++stats->surfaceDeformableFinalizeBudgetDiagWithinBudgetRows;
      break;
    case eBUDGET_DIAG_OVER_BUDGET:
      ++nodes[root].budgetDiagOverBudgetRows;
      ++stats->surfaceDeformableFinalizeBudgetDiagOverBudgetRows;
      break;
    default:
      ++nodes[root].budgetDiagUnsupportedRows;
      ++stats->surfaceDeformableFinalizeBudgetDiagUnsupportedRows;
      break;
    }
    if (contact.restitution > 0.0f)
      nodes[root].restitution = 1;
    if (contact.maxImpulse < PX_MAX_REAL)
      nodes[root].finiteImpulse = 1;
    if (contact.targetVelocity.magnitudeSquared() > 1.0e-12f)
      nodes[root].targetVelocity = 1;
    if (isBodyVsStaticContact(bodyA, bodyB, numBodies)) {
      if (!hasDeformableStaticAnchor(contact))
        nodes[root].rigidStatic = 1;
      else if (!hasDeformablePositionTangentOwner(contact))
        nodes[root].nonOwnerDeformable = 1;
    }

    const auto recordScale =
        [&](physx::PxU32 body, physx::PxReal linearScale,
            physx::PxReal angularScale) {
          if (body >= numBodies)
            return;
          SurfaceFinalizeTopologyNode &bodyNode = nodes[body];
          if (!bodyNode.scaleSeen) {
            bodyNode.scaleSeen = 1;
            bodyNode.firstLinearScale = linearScale;
            bodyNode.firstAngularScale = angularScale;
            return;
          }
          const physx::PxReal linearTolerance =
              1.0e-6f *
              physx::PxMax(
                  physx::PxReal(1.0f),
                  physx::PxMax(
                      physx::PxAbs(bodyNode.firstLinearScale),
                               physx::PxAbs(linearScale)));
          const physx::PxReal angularTolerance =
              1.0e-6f *
              physx::PxMax(
                  physx::PxReal(1.0f),
                  physx::PxMax(
                      physx::PxAbs(bodyNode.firstAngularScale),
                               physx::PxAbs(angularScale)));
          if (physx::PxAbs(
                  linearScale - bodyNode.firstLinearScale) >
                  linearTolerance ||
              physx::PxAbs(
                  angularScale - bodyNode.firstAngularScale) >
                  angularTolerance)
            nodes[root].mixedScale = 1;
        };
    recordScale(bodyA, contact.invMassScaleA,
                contact.invInertiaScaleA);
    recordScale(bodyB, contact.invMassScaleB,
                contact.invInertiaScaleB);
  }

  for (physx::PxU32 root = 0; root < numBodies; ++root) {
    const SurfaceFinalizeTopologyNode &component = nodes[root];
    if (component.parent != root || !component.strictOwner)
      continue;
    ++stats->surfaceDeformableFinalizeComponents;

    const physx::PxU32 componentBodies = component.bodyCount;
    if (componentBodies == 1)
      ++stats->surfaceDeformableFinalizeComponentOneBody;
    else if (componentBodies == 2)
      ++stats->surfaceDeformableFinalizeComponentTwoBodies;
    else if (componentBodies <= 4)
      ++stats->surfaceDeformableFinalizeComponentThreeToFourBodies;
    else if (componentBodies <= 8)
      ++stats->surfaceDeformableFinalizeComponentFiveToEightBodies;
    else if (componentBodies <= 16)
      ++stats->surfaceDeformableFinalizeComponentNineToSixteenBodies;
    else if (componentBodies <= 32)
      ++stats
            ->surfaceDeformableFinalizeComponentSeventeenToThirtyTwoBodies;
    else
      ++stats->surfaceDeformableFinalizeComponentOverThirtyTwoBodies;

    const physx::PxU32 componentRowCount = component.rowCount;
    if (componentRowCount <= 8)
      ++stats->surfaceDeformableFinalizeComponentOneToEightRows;
    else if (componentRowCount <= 16)
      ++stats->surfaceDeformableFinalizeComponentNineToSixteenRows;
    else if (componentRowCount <= 32)
      ++stats
            ->surfaceDeformableFinalizeComponentSeventeenToThirtyTwoRows;
    else if (componentRowCount <= 64)
      ++stats
            ->surfaceDeformableFinalizeComponentThirtyThreeToSixtyFourRows;
    else
      ++stats->surfaceDeformableFinalizeComponentOverSixtyFourRows;

    if (component.restitution)
      ++stats->surfaceDeformableFinalizeComponentRestitution;
    if (component.finiteImpulse)
      ++stats->surfaceDeformableFinalizeComponentFiniteImpulse;
    if (component.targetVelocity)
      ++stats->surfaceDeformableFinalizeComponentTargetVelocity;
    if (component.mixedScale)
      ++stats->surfaceDeformableFinalizeComponentMixedScale;
    if (component.rigidStatic)
      ++stats->surfaceDeformableFinalizeComponentRigidStatic;
    if (component.nonOwnerDeformable)
      ++stats->surfaceDeformableFinalizeComponentNonOwnerDeformable;
    if (hasJointConstraints)
      ++stats->surfaceDeformableFinalizeComponentJointIsland;
    if (component.lockedDof)
      ++stats->surfaceDeformableFinalizeComponentLockedDof;
    if (component.nonDynamicBody)
      ++stats->surfaceDeformableFinalizeComponentNonDynamicBody;
    if (component.budgetDiagUnsupportedRows > 0) {
      ++stats
            ->surfaceDeformableFinalizeBudgetDiagComponentsUnsupported;
    } else if (
        component.budgetDiagZeroBudgetRequiredRows > 0 ||
        component.budgetDiagOverBudgetRows > 0) {
      ++stats
            ->surfaceDeformableFinalizeBudgetDiagComponentsOverBudget;
    } else {
      ++stats
            ->surfaceDeformableFinalizeBudgetDiagComponentsWithinBudget;
    }

    ++stats->surfaceDeformableFinalizeShadowComponents;
    stats->surfaceDeformableFinalizeShadowRows += component.rowCount;
    const bool shadowUnsupported =
        component.restitution || component.targetVelocity ||
        component.mixedScale || component.rigidStatic ||
        component.nonOwnerDeformable || hasJointConstraints ||
        component.lockedDof || component.nonDynamicBody ||
        component.fastImpact || component.snapshotUnsupported;
    if (component.fastImpact)
      ++stats->surfaceDeformableFinalizeShadowUnsupportedFastImpact;
    if (component.snapshotUnsupported)
      ++stats->surfaceDeformableFinalizeShadowUnsupportedSnapshot;
    if (shadowUnsupported) {
      ++stats->surfaceDeformableFinalizeShadowUnsupported;
      continue;
    }

    physx::PxArray<physx::PxU32> orderedRows = componentRows[root];
    std::sort(
        orderedRows.begin(), orderedRows.end(),
        [&](physx::PxU32 lhs, physx::PxU32 rhs) {
          const AvbdContactConstraint &a = contacts[lhs];
          const AvbdContactConstraint &b = contacts[rhs];
          if (a.cacheKey != b.cacheKey)
            return a.cacheKey < b.cacheKey;
          const physx::PxU32 aMin =
              physx::PxMin(a.header.bodyIndexA, a.header.bodyIndexB);
          const physx::PxU32 bMin =
              physx::PxMin(b.header.bodyIndexA, b.header.bodyIndexB);
          if (aMin != bMin)
            return aMin < bMin;
          const physx::PxU32 aMax =
              physx::PxMax(a.header.bodyIndexA, a.header.bodyIndexB);
          const physx::PxU32 bMax =
              physx::PxMax(b.header.bodyIndexA, b.header.bodyIndexB);
          if (aMax != bMax)
            return aMax < bMax;
          const physx::PxReal aValues[9] = {
              a.contactNormal.x, a.contactNormal.y, a.contactNormal.z,
              a.contactPointA.x, a.contactPointA.y, a.contactPointA.z,
              a.contactPointB.x, a.contactPointB.y, a.contactPointB.z};
          const physx::PxReal bValues[9] = {
              b.contactNormal.x, b.contactNormal.y, b.contactNormal.z,
              b.contactPointA.x, b.contactPointA.y, b.contactPointA.z,
              b.contactPointB.x, b.contactPointB.y, b.contactPointB.z};
          for (physx::PxU32 value = 0; value < 9; ++value) {
            if (aValues[value] != bValues[value])
              return aValues[value] < bValues[value];
          }
          return lhs < rhs;
        });

    const physx::PxU32 rowCount = orderedRows.size();
    physx::PxArray<double> outward(rowCount, 0.0);
    physx::PxArray<double> upperBounds(rowCount, 0.0);
    bool assemblyValid = rowCount == component.rowCount;
    for (physx::PxU32 row = 0; row < rowCount && assemblyValid; ++row) {
      const physx::PxU32 contactIndex = orderedRows[row];
      if (contactIndex >= budgetDiagSnapshots.size()) {
        assemblyValid = false;
        break;
      }
      outward[row] =
          double(budgetDiagSnapshots[contactIndex].outwardVelocity);
      upperBounds[row] =
          double(budgetDiagSnapshots[contactIndex].maximumImpulse);
      if (!std::isfinite(outward[row]) ||
          !std::isfinite(upperBounds[row]) ||
          upperBounds[row] < 0.0) {
        assemblyValid = false;
        break;
      }
    }

    if (!assemblyValid) {
      ++stats->surfaceDeformableFinalizeShadowNumericalFailure;
      continue;
    }
    const bool useMatrixFreeBackend = rowCount > 128;
    AvbdBoundedProjectionResult shadow;
    if (useMatrixFreeBackend) {
      ++stats->surfaceDeformableFinalizeShadowMatrixFreeComponents;
      stats->surfaceDeformableFinalizeShadowMatrixFreeRows += rowCount;
      shadow = solveSurfaceFinalizeMatrixFreeBoundedProjection(
          bodies, numBodies, nodes, root, contacts, orderedRows,
          outward, upperBounds);
    } else {
      physx::PxArray<double> response(rowCount * rowCount, 0.0);
      for (physx::PxU32 row = 0;
           row < rowCount && assemblyValid; ++row) {
        const AvbdContactConstraint &a =
            contacts[orderedRows[row]];
        for (physx::PxU32 column = 0;
             column < rowCount; ++column) {
          const AvbdContactConstraint &b =
              contacts[orderedRows[column]];
          double value = 0.0;
          const physx::PxU32 aBodies[2] = {
              a.header.bodyIndexA, a.header.bodyIndexB};
          const physx::PxU32 bBodies[2] = {
              b.header.bodyIndexA, b.header.bodyIndexB};
          const physx::PxVec3 aPoints[2] = {
              a.contactPointA, a.contactPointB};
          const physx::PxVec3 bPoints[2] = {
              b.contactPointA, b.contactPointB};
          const physx::PxVec3 aAxes[2] = {
              a.contactNormal, -a.contactNormal};
          const physx::PxVec3 bAxes[2] = {
              b.contactNormal, -b.contactNormal};
          for (physx::PxU32 aEnd = 0; aEnd < 2; ++aEnd) {
            const physx::PxU32 body = aBodies[aEnd];
            if (body >= numBodies)
              continue;
            for (physx::PxU32 bEnd = 0; bEnd < 2; ++bEnd) {
              if (bBodies[bEnd] != body)
                continue;
              const SurfaceFinalizeTopologyNode &bodyNode =
                  nodes[body];
              const physx::PxVec3 aArm =
                  bodies[body].rotation.rotate(aPoints[aEnd]);
              const physx::PxVec3 bArm =
                  bodies[body].rotation.rotate(bPoints[bEnd]);
              const physx::PxVec3 aAngular =
                  aArm.cross(aAxes[aEnd]);
              const physx::PxVec3 bAngular =
                  bArm.cross(bAxes[bEnd]);
              value +=
                  double(bodies[body].invMass *
                         bodyNode.firstLinearScale *
                         aAxes[aEnd].dot(bAxes[bEnd])) +
                  double(aAngular.dot(
                             bodies[body].invInertiaWorld.transform(
                                 bAngular)) *
                         bodyNode.firstAngularScale);
            }
          }
          if (!std::isfinite(value)) {
            assemblyValid = false;
            break;
          }
          response[row * rowCount + column] = value;
        }
      }
      if (!assemblyValid) {
        ++stats->surfaceDeformableFinalizeShadowNumericalFailure;
        continue;
      }
      shadow = solveAvbdBoundedProjection(
          response, outward, upperBounds, 6 * component.bodyCount);
      if (enableMatrixFreeOracle) {
        ++stats
              ->surfaceDeformableFinalizeShadowMatrixFreeOracleComponents;
        stats->surfaceDeformableFinalizeShadowMatrixFreeOracleRows +=
            rowCount;
        bool operatorMatched = false;
        const AvbdBoundedProjectionResult matrixFreeOracle =
            solveSurfaceFinalizeMatrixFreeBoundedProjection(
                bodies, numBodies, nodes, root, contacts, orderedRows,
                outward, upperBounds, 1.0e-6, &response,
                &operatorMatched);
        bool comparable = false;
        const bool matched =
            compareSurfaceFinalizeMatrixFreeOracle(
                shadow, matrixFreeOracle, response, outward,
                operatorMatched, comparable);
        if (!comparable)
          ++stats
                ->surfaceDeformableFinalizeShadowMatrixFreeOracleSkipped;
        else if (matched)
          ++stats
                ->surfaceDeformableFinalizeShadowMatrixFreeOracleMatched;
        else
          ++stats
                ->surfaceDeformableFinalizeShadowMatrixFreeOracleMismatched;
      }
    }
    stats->surfaceDeformableFinalizeShadowLowerRows += shadow.lowerRows;
    stats->surfaceDeformableFinalizeShadowFreeRows += shadow.freeRows;
    stats->surfaceDeformableFinalizeShadowUpperRows += shadow.upperRows;
    switch (shadow.status) {
    case eAVBD_BOUNDED_SOLVED:
      ++stats->surfaceDeformableFinalizeShadowSolved;
      if (useMatrixFreeBackend)
        ++stats->surfaceDeformableFinalizeShadowMatrixFreeSolved;
      if (shadow.commitImpulses.size() == rowCount)
        ++stats->surfaceDeformableFinalizeShadowCommitCapable;
      if (enableProductionProbe &&
          shadow.commitImpulses.size() == rowCount &&
          probeOwnedBodies.size() == numBodies) {
        ++stats->surfaceDeformableFinalizeProbeEligibleComponents;
        physx::PxArray<physx::PxVec3> linearImpulses(
            numBodies, physx::PxVec3(0.0f));
        physx::PxArray<physx::PxVec3> angularImpulses(
            numBodies, physx::PxVec3(0.0f));
        bool commitValid = true;
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          const double candidate = shadow.commitImpulses[row];
          if (!std::isfinite(candidate) || candidate < 0.0 ||
              candidate > double(PX_MAX_REAL)) {
            commitValid = false;
            break;
          }
          const physx::PxReal impulse = physx::PxReal(candidate);
          const AvbdContactConstraint &contact =
              contacts[orderedRows[row]];
          const physx::PxU32 rowBodies[2] = {
              contact.header.bodyIndexA, contact.header.bodyIndexB};
          const physx::PxVec3 rowPoints[2] = {
              contact.contactPointA, contact.contactPointB};
          const physx::PxVec3 rowAxes[2] = {
              contact.contactNormal, -contact.contactNormal};
          for (physx::PxU32 end = 0; end < 2; ++end) {
            const physx::PxU32 body = rowBodies[end];
            if (body >= numBodies)
              continue;
            const physx::PxVec3 arm =
                bodies[body].rotation.rotate(rowPoints[end]);
            linearImpulses[body] += rowAxes[end] * impulse;
            angularImpulses[body] +=
                arm.cross(rowAxes[end]) * impulse;
          }
        }
        physx::PxArray<physx::PxVec3> linearDeltas(
            numBodies, physx::PxVec3(0.0f));
        physx::PxArray<physx::PxVec3> angularDeltas(
            numBodies, physx::PxVec3(0.0f));
        if (commitValid) {
          for (physx::PxU32 body = 0; body < numBodies; ++body) {
            if (nodes[body].parent != root)
              continue;
            const SurfaceFinalizeTopologyNode &bodyNode = nodes[body];
            linearDeltas[body] =
                linearImpulses[body] *
                (bodies[body].invMass * bodyNode.firstLinearScale);
            angularDeltas[body] =
                bodies[body].invInertiaWorld.transform(
                    angularImpulses[body]) *
                bodyNode.firstAngularScale;
            if (!linearDeltas[body].isFinite() ||
                !angularDeltas[body].isFinite()) {
              commitValid = false;
              break;
            }
          }
        }
        if (commitValid) {
          double velocityScale = 1.0;
          for (physx::PxU32 row = 0; row < rowCount; ++row)
            velocityScale =
                std::max(velocityScale, std::fabs(outward[row]));
          const double residualTolerance = 8.0e-6 * velocityScale;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            const AvbdContactConstraint &contact =
                contacts[orderedRows[row]];
            const physx::PxU32 rowBodies[2] = {
                contact.header.bodyIndexA, contact.header.bodyIndexB};
            const physx::PxVec3 rowPoints[2] = {
                contact.contactPointA, contact.contactPointB};
            const physx::PxVec3 rowAxes[2] = {
                contact.contactNormal, -contact.contactNormal};
            double responseDelta = 0.0;
            for (physx::PxU32 end = 0; end < 2; ++end) {
              const physx::PxU32 body = rowBodies[end];
              if (body >= numBodies)
                continue;
              const physx::PxVec3 arm =
                  bodies[body].rotation.rotate(rowPoints[end]);
              const physx::PxVec3 pointDelta =
                  linearDeltas[body] +
                  angularDeltas[body].cross(arm);
              responseDelta +=
                  double(pointDelta.dot(rowAxes[end]));
            }
            const double postOutward =
                outward[row] - responseDelta;
            if (!std::isfinite(postOutward) ||
                postOutward > residualTolerance) {
              commitValid = false;
              break;
            }
          }
        }
        if (commitValid) {
          physx::PxU32 committedBodies = 0;
          physx::PxU32 replacedOwners = 0;
          for (physx::PxU32 body = 0; body < numBodies; ++body) {
            if (nodes[body].parent != root)
              continue;
            bodies[body].linearVelocity -= linearDeltas[body];
            bodies[body].angularVelocity -= angularDeltas[body];
            probeOwnedBodies[body] = true;
            ++committedBodies;
            if (nodes[body].bodyStrictOwner)
              ++replacedOwners;
          }
          ++stats->surfaceDeformableFinalizeProbeCommittedComponents;
          if (useMatrixFreeBackend)
            ++stats
                  ->surfaceDeformableFinalizeShadowMatrixFreeCommittedComponents;
          stats->surfaceDeformableFinalizeProbeCommittedRows += rowCount;
          stats->surfaceDeformableFinalizeProbeCommittedBodies +=
              committedBodies;
          stats->surfaceDeformableFinalizeProbeReplacedOwnerBodies +=
              replacedOwners;
        }
      }
      break;
    case eAVBD_BOUNDED_NO_CORRECTION:
      ++stats->surfaceDeformableFinalizeShadowNoCorrection;
      if (useMatrixFreeBackend)
        ++stats->surfaceDeformableFinalizeShadowMatrixFreeNoCorrection;
      break;
    case eAVBD_BOUNDED_BUDGET_EXHAUSTED:
      ++stats->surfaceDeformableFinalizeShadowBudgetExhausted;
      if (useMatrixFreeBackend)
        ++stats
              ->surfaceDeformableFinalizeShadowMatrixFreeBudgetExhausted;
      break;
    case eAVBD_BOUNDED_INFEASIBLE:
      ++stats->surfaceDeformableFinalizeShadowInfeasible;
      if (useMatrixFreeBackend)
        ++stats->surfaceDeformableFinalizeShadowMatrixFreeInfeasible;
      break;
    case eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED:
      ++stats->surfaceDeformableFinalizeShadowResidualUnclassified;
      if (useMatrixFreeBackend)
        ++stats
              ->surfaceDeformableFinalizeShadowMatrixFreeResidualUnclassified;
      break;
    case eAVBD_BOUNDED_ITERATION_LIMIT:
      ++stats->surfaceDeformableFinalizeShadowIterationLimit;
      if (useMatrixFreeBackend)
        ++stats
              ->surfaceDeformableFinalizeShadowMatrixFreeIterationLimit;
      break;
    default:
      ++stats->surfaceDeformableFinalizeShadowNumericalFailure;
      if (useMatrixFreeBackend)
        ++stats
              ->surfaceDeformableFinalizeShadowMatrixFreeNumericalFailure;
      break;
    }
    if (useMatrixFreeBackend) {
      stats->surfaceDeformableFinalizeShadowMatrixFreeIterations +=
          shadow.iterations;
      if (shadow.status == eAVBD_BOUNDED_ITERATION_LIMIT) {
        const double tolerance =
            shadow.projectedGradientTolerance;
        const double violation = shadow.maximumKktViolation;
        if (std::isfinite(violation) &&
            std::isfinite(tolerance) && tolerance > 0.0 &&
            violation <= 2.0 * tolerance) {
          ++stats
                ->surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost2x;
        } else if (
            std::isfinite(violation) &&
            std::isfinite(tolerance) && tolerance > 0.0 &&
            violation <= 16.0 * tolerance) {
          ++stats
                ->surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost16x;
        } else {
          ++stats
                ->surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktOver16x;
        }
      }
    }
  }
}

static void recordSurfaceFinalizeOwnerDiscoveryComparison(
    const physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    const physx::PxArray<bool> &probeOwnedBodies,
    AvbdSolverStats *stats) {
  if (!stats || probeOwnedBodies.size() != nodes.size())
    return;
  for (physx::PxU32 body = 0; body < nodes.size(); ++body) {
    const bool preOwner = nodes[body].bodyStrictOwner != 0;
    const bool legacyOwner = nodes[body].legacyStrictOwner != 0;
    const bool replacedOwner = preOwner && probeOwnedBodies[body];
    if (preOwner)
      ++stats->surfaceDeformableFinalizePreOwnerBodies;
    if (legacyOwner)
      ++stats->surfaceDeformableFinalizeLegacyOwnerBodies;
    if (preOwner != (legacyOwner || replacedOwner))
      ++stats->surfaceDeformableFinalizeOwnerDiscoveryMismatchBodies;
  }
}

static void applyAvbdMaterialNormalVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale,
    bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    bool enableMatrixFreeComponentOracle,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  const physx::PxReal invDt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
  const physx::PxReal bounceThreshold =
      bounceApproachThreshold > 0.0f
          ? bounceApproachThreshold
          : AvbdConstants::AVBD_BOUNCE_THRESHOLD;
  physx::PxArray<SurfaceFinalizeTopologyNode> finalizeTopologyNodes;
  physx::PxArray<SurfaceFinalizeBudgetDiagSnapshot>
      finalizeBudgetDiagSnapshots;
  physx::PxArray<bool> finalizeProbeOwnedBodies(numBodies, false);
  if (stats && deformableNormalStageMask) {
    finalizeTopologyNodes.resize(numBodies);
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      finalizeTopologyNodes[body].strictOwner = 0;
      finalizeTopologyNodes[body].bodyStrictOwner = 0;
      finalizeTopologyNodes[body].legacyStrictOwner = 0;
    }
    finalizeBudgetDiagSnapshots.resize(numContacts);
    for (physx::PxU32 row = 0; row < numContacts; ++row) {
      finalizeBudgetDiagSnapshots[row] =
          classifySurfaceFinalizeBudgetDiag(
              bodies, numBodies, contacts[row], dt, lengthScale,
              linearVelAtSolveStart, angularVelAtSolveStart);
    }
    discoverSurfaceFinalizeStrictOwnersPreP3K(
        bodies, numBodies, contacts, numContacts,
        linearVelAtSolveStart, finalizeTopologyNodes);
    recordSurfaceDeformableFinalizeComponentTopology(
        bodies, numBodies, contacts, numContacts,
        finalizeTopologyNodes, finalizeBudgetDiagSnapshots,
        hasJointConstraints, enableBoundedComponentProductionProbe,
        enableMatrixFreeComponentOracle,
        finalizeProbeOwnedBodies, stats);
  }
  // ---- Body-static (incl. deformable anchors) ----
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;
    if (finalizeProbeOwnedBodies[i])
      continue;
    bool passiveMaterialComponentOwned = false;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      if (hasVelocityPassiveFrictionComponentOwner(contacts[c]) &&
          (contacts[c].header.bodyIndexA == i ||
           contacts[c].header.bodyIndexB == i)) {
        passiveMaterialComponentOwned = true;
        break;
      }
    }
    if (passiveMaterialComponentOwned)
      continue;

    physx::PxU32 dominant = 0xFFFFFFFFu;
    physx::PxU32 initialDominant = 0xFFFFFFFFu;
    physx::PxReal worstViolation = 1e9f;
    physx::PxReal worstInitialViolation = 1e9f;
    physx::PxVec3 domWorldA(0.0f), domWorldB(0.0f);
    physx::PxVec3 initialDomWorldA(0.0f), initialDomWorldB(0.0f);

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
      const physx::PxVec3 initialWorldA =
          dynIsA
              ? bodies[i].prevPosition +
                    bodies[i].prevRotation.rotate(contacts[c].contactPointA)
              : contacts[c].staticPrevWorldPoint;
      const physx::PxVec3 initialWorldB =
          dynIsA
              ? contacts[c].staticPrevWorldPoint
              : bodies[i].prevPosition +
                    bodies[i].prevRotation.rotate(contacts[c].contactPointB);
      const physx::PxReal initialViolation =
          (initialWorldA - initialWorldB).dot(contacts[c].contactNormal) +
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
      if (initialViolation < worstInitialViolation) {
        worstInitialViolation = initialViolation;
        initialDominant = c;
        initialDomWorldA = worldA;
        initialDomWorldB = worldB;
      }
    }

    if (dominant == 0xFFFFFFFFu)
      continue;

    const bool splitDeepInitialDepenetration =
        initialDominant != 0xFFFFFFFFu &&
        !hasDeformableStaticAnchor(contacts[initialDominant]) &&
        worstInitialViolation <
            -kBodyStaticNearSurface *
                physx::PxMax(lengthScale, physx::PxReal(1e-6f));
    if (splitDeepInitialDepenetration) {
      dominant = initialDominant;
      domWorldA = initialDomWorldA;
      domWorldB = initialDomWorldB;
    }

    if (finiteMaterialPoseSplit &&
        finiteMaterialPoseSplit->size() == numBodies &&
        (*finiteMaterialPoseSplit)[i]) {
      physx::PxReal spatialLinearDelta = 0.0f;
      const bool finiteOwned = applyBodyStaticFiniteSpatialBudget(
          bodies, numBodies, contacts, numContacts, i,
          linearVelAtSolveStart, angularVelAtSolveStart, dt, bounceThreshold,
          spatialLinearDelta);
      if (finiteOwned) {
        if (stats && spatialLinearDelta > 1.0e-7f) {
          stats->bodyStaticMaterialVelocityCorrections++;
          stats->bodyStaticMaterialVelocityDelta += spatialLinearDelta;
          stats->bodyStaticMaterialVelocityMaxDelta = physx::PxMax(
              stats->bodyStaticMaterialVelocityMaxDelta,
              spatialLinearDelta);
        }
        continue;
      }
    }

    const bool isDeform = hasDeformableStaticAnchor(contacts[dominant]);
    const AvbdContactConstraint &cc = contacts[dominant];
    const bool established = cc.contactManagerEstablished != 0;
    const bool dynIsA = (cc.header.bodyIndexA == i);
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);

    physx::PxReal staticNormalVelocity = 0.0f;
    if (!isDeform && invDt > 0.0f) {
      const physx::PxVec3 staticNow = dynIsA ? domWorldB : domWorldA;
      staticNormalVelocity =
          ((staticNow - cc.staticPrevWorldPoint) * invDt).dot(nd);
    }

    const bool hasSolveStartVelocity =
        linearVelAtSolveStart &&
        linearVelAtSolveStart->size() == numBodies;
    const physx::PxReal vn = bodies[i].linearVelocity.dot(nd);
    const physx::PxReal relativeVn = vn - staticNormalVelocity;
    physx::PxReal solveStartRelativeVn = relativeVn;
    physx::PxReal approach = 0.0f;
    if (hasSolveStartVelocity) {
      solveStartRelativeVn =
          (*linearVelAtSolveStart)[i].dot(nd) - staticNormalVelocity;
      approach = -solveStartRelativeVn;
      if (approach < 0.0f)
        approach = 0.0f;
    }
    const bool hasFiniteMaxImpulse = cc.maxImpulse < PX_MAX_REAL;
    const physx::PxReal maxImpulseRelativeVn =
        hasSolveStartVelocity && hasFiniteMaxImpulse
            ? solveStartRelativeVn +
                  physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)) *
                      bodies[i].invMass *
                      (dynIsA ? cc.invMassScaleA : cc.invMassScaleB)
            : PX_MAX_REAL;
    if (stats) {
      const physx::PxReal poseSeparatingVelocity =
          physx::PxMax(relativeVn, physx::PxReal(0.0f));
      stats->bodyStaticMaterialPoseSeparatingVelocity +=
          poseSeparatingVelocity;
      if (!isDeform) {
        if (established) {
          stats->bodyStaticNormalSupportFinalizeBodies++;
          stats->bodyStaticNormalSupportPoseSeparatingVelocity +=
              poseSeparatingVelocity;
        } else {
          stats->bodyStaticNormalOnsetFinalizeBodies++;
          stats->bodyStaticNormalOnsetPoseSeparatingVelocity +=
              poseSeparatingVelocity;
        }
      }
      if (hasFiniteMaxImpulse) {
        stats->bodyStaticMaterialFiniteBudgetRows++;
        const physx::PxReal positionImpulse =
            physx::PxMax(-cc.header.lambda, physx::PxReal(0.0f)) * dt;
        stats->bodyStaticMaterialFiniteRemainingImpulse += physx::PxMax(
            physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)) -
                positionImpulse,
            physx::PxReal(0.0f));
      } else {
        stats->bodyStaticMaterialUnlimitedBudgetRows++;
      }
    }
    if (isDeform) {
      if (stats)
        stats->surfaceDeformableFinalizeBodies++;
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
      const physx::PxVec3 dynamicWorldPoint =
          dynIsA ? domWorldA : domWorldB;
      const physx::PxVec3 dynamicContactArm =
          dynamicWorldPoint - bodies[i].position;
      const physx::PxReal contactRelativeVnBefore =
          (bodies[i].linearVelocity +
           bodies[i].angularVelocity.cross(dynamicContactArm))
                  .dot(nd) -
          vMeshN;
      const bool spatialOwner = hasDeformablePositionTangentOwner(cc);
      const physx::PxReal comRelativeVn = vn - vMeshN;
      const physx::PxReal correctionRelativeVn =
          spatialOwner ? contactRelativeVnBefore : comRelativeVn;
      if (correctionRelativeVn > 0.0f) {
        physx::PxReal linearDeltaMagnitude = 0.0f;
        bool corrected = false;
        if (spatialOwner) {
          const physx::PxReal linearScale =
              dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
          const physx::PxReal angularScale =
              dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
          const physx::PxReal linearResponse =
              bodies[i].invMass * linearScale;
          const physx::PxVec3 angularJacobian =
              dynamicContactArm.cross(nd);
          const physx::PxVec3 angularResponse =
              bodies[i].invInertiaWorld.transform(angularJacobian);
          const physx::PxReal totalResponse =
              linearResponse +
              angularJacobian.dot(angularResponse) * angularScale;
          if (totalResponse > 1.0e-8f) {
            const physx::PxReal impulse =
                contactRelativeVnBefore / totalResponse;
            linearDeltaMagnitude = impulse * linearResponse;
            bodies[i].linearVelocity -=
                nd * linearDeltaMagnitude;
            bodies[i].angularVelocity -=
                angularResponse * (impulse * angularScale);
            corrected = true;
          }
        } else {
          bodies[i].linearVelocity -= nd * comRelativeVn;
          linearDeltaMagnitude = comRelativeVn;
          corrected = true;
        }
        if (corrected) {
          const bool collectContactPointDiagnostic =
              stats && deformableNormalStageMask &&
              dominant < deformableNormalStageMask->size();
          if (collectContactPointDiagnostic) {
            const physx::PxReal contactRelativeVnAfter =
                (bodies[i].linearVelocity +
                 bodies[i].angularVelocity.cross(dynamicContactArm))
                        .dot(nd) -
                vMeshN;
            const physx::PxReal diagnosticVelocityTolerance =
                1.0e-5f *
                physx::PxMax(lengthScale, physx::PxReal(1.0e-6f)) *
                invDt;
            stats->surfaceDeformableFinalizeContactPreSeparation +=
                physx::PxMax(contactRelativeVnBefore, physx::PxReal(0.0f));
            if (!spatialOwner &&
                contactRelativeVnBefore <=
                diagnosticVelocityTolerance)
              stats
                  ->surfaceDeformableFinalizeContactFalsePositiveCorrections++;
            if (contactRelativeVnAfter > diagnosticVelocityTolerance) {
              stats
                  ->surfaceDeformableFinalizeContactResidualSeparationCorrections++;
              stats->surfaceDeformableFinalizeContactPostSeparation +=
                  contactRelativeVnAfter;
            } else if (contactRelativeVnAfter <
                       -diagnosticVelocityTolerance) {
              stats->surfaceDeformableFinalizeContactReversalCorrections++;
              stats->surfaceDeformableFinalizeContactPostApproach +=
                  -contactRelativeVnAfter;
            }
          }
          if (deformableNormalStageMask &&
              dominant < deformableNormalStageMask->size())
            (*deformableNormalStageMask)[dominant] |= 4u;
          if (stats) {
            stats->surfaceDeformableFinalizeCorrections++;
            stats->surfaceDeformableFinalizeDelta += linearDeltaMagnitude;
            if (spatialOwner)
              stats->surfaceDeformableFinalizeSpatialCorrections++;
            else
              stats->surfaceDeformableFinalizeComFallbackCorrections++;
          }
        }
      }
      if (stats && deformableNormalStageMask) {
        const physx::PxReal diagnosticVelocityTolerance =
            1.0e-5f *
            physx::PxMax(lengthScale, physx::PxReal(1.0e-6f)) * invDt;
        physx::PxU32 manifoldRows = 0;
        physx::PxU32 manifoldRank = 0;
        physx::PxReal manifoldBasis[6][6] = {};
        physx::PxReal firstLinearScale = 0.0f;
        physx::PxReal firstAngularScale = 0.0f;
        bool mixedScale = false;
        for (physx::PxU32 c = 0; c < numContacts; ++c) {
          const AvbdContactConstraint &secondary = contacts[c];
          const physx::PxU32 secondaryBodyA =
              secondary.header.bodyIndexA;
          const physx::PxU32 secondaryBodyB =
              secondary.header.bodyIndexB;
          if (!isBodyVsStaticContact(
                  secondaryBodyA, secondaryBodyB, numBodies) ||
              (secondaryBodyA != i && secondaryBodyB != i) ||
              !hasDeformableStaticAnchor(secondary) ||
              !hasDeformablePositionTangentOwner(secondary))
            continue;

          const bool secondaryDynIsA = secondaryBodyA == i;
          const physx::PxVec3 secondaryNormal =
              secondary.contactNormal *
              (secondaryDynIsA ? 1.0f : -1.0f);
          const physx::PxVec3 secondaryDynamicWorldPoint =
              bodies[i].position +
              bodies[i].rotation.rotate(
                  secondaryDynIsA ? secondary.contactPointA
                                  : secondary.contactPointB);
          const physx::PxVec3 secondaryStaticWorldPoint =
              secondaryDynIsA ? secondary.contactPointB
                              : secondary.contactPointA;
          const physx::PxVec3 secondaryWorldA =
              secondaryDynIsA ? secondaryDynamicWorldPoint
                              : secondaryStaticWorldPoint;
          const physx::PxVec3 secondaryWorldB =
              secondaryDynIsA ? secondaryStaticWorldPoint
                              : secondaryDynamicWorldPoint;
          const physx::PxReal secondaryViolation =
              finalizeBodyVsStaticViolation(
                  (secondaryWorldA - secondaryWorldB)
                          .dot(secondary.contactNormal) +
                      secondary.penetrationDepth,
                  secondary.penetrationDepth);
          if (secondaryViolation >= nearLim)
            continue;

          const physx::PxReal secondaryLinearScale =
              secondaryDynIsA ? secondary.invMassScaleA
                              : secondary.invMassScaleB;
          const physx::PxReal secondaryAngularScale =
              secondaryDynIsA ? secondary.invInertiaScaleA
                              : secondary.invInertiaScaleB;
          if (manifoldRows == 0) {
            firstLinearScale = secondaryLinearScale;
            firstAngularScale = secondaryAngularScale;
          } else {
            const physx::PxReal linearScaleTolerance =
                1.0e-6f * physx::PxMax(
                              physx::PxReal(1.0f),
                              physx::PxMax(
                                  physx::PxAbs(firstLinearScale),
                                  physx::PxAbs(secondaryLinearScale)));
            const physx::PxReal angularScaleTolerance =
                1.0e-6f * physx::PxMax(
                              physx::PxReal(1.0f),
                              physx::PxMax(
                                  physx::PxAbs(firstAngularScale),
                                  physx::PxAbs(secondaryAngularScale)));
            if (physx::PxAbs(
                    secondaryLinearScale - firstLinearScale) >
                    linearScaleTolerance ||
                physx::PxAbs(
                    secondaryAngularScale - firstAngularScale) >
                    angularScaleTolerance)
              mixedScale = true;
          }

          const physx::PxVec3 secondaryContactArm =
              secondaryDynamicWorldPoint - bodies[i].position;
          const physx::PxVec3 secondaryAngularJacobian =
              secondaryContactArm.cross(secondaryNormal);
          physx::PxReal spatialRow[6] = {
              secondaryNormal.x, secondaryNormal.y, secondaryNormal.z,
              secondaryAngularJacobian.x,
              secondaryAngularJacobian.y,
              secondaryAngularJacobian.z};
          for (physx::PxU32 basisRow = 0;
               basisRow < manifoldRank; ++basisRow) {
            physx::PxReal projection = 0.0f;
            for (physx::PxU32 component = 0; component < 6; ++component)
              projection +=
                  spatialRow[component] *
                  manifoldBasis[basisRow][component];
            for (physx::PxU32 component = 0; component < 6; ++component)
              spatialRow[component] -=
                  projection * manifoldBasis[basisRow][component];
          }
          physx::PxReal spatialRowNormSquared = 0.0f;
          for (physx::PxU32 component = 0; component < 6; ++component)
            spatialRowNormSquared += spatialRow[component] *
                                     spatialRow[component];
          if (manifoldRank < 6 && spatialRowNormSquared > 1.0e-8f) {
            const physx::PxReal invNorm =
                1.0f / physx::PxSqrt(spatialRowNormSquared);
            for (physx::PxU32 component = 0; component < 6; ++component)
              manifoldBasis[manifoldRank][component] =
                  spatialRow[component] * invNorm;
            ++manifoldRank;
          }
          ++manifoldRows;

          physx::PxReal secondaryMeshNormalVelocity = 0.0f;
          if (invDt > 0.0f) {
            const physx::PxVec3 secondaryMeshStep =
                secondaryStaticWorldPoint -
                secondary.staticPrevWorldPoint;
            const physx::PxReal stepCap =
                AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
            if (secondaryMeshStep.magnitudeSquared() <= stepCap * stepCap)
              secondaryMeshNormalVelocity =
                  (secondaryMeshStep * invDt).dot(secondaryNormal);
            else
              stats->surfaceDeformableFinalizeManifoldAliasRows++;
          }
          if (c == dominant)
            continue;
          const physx::PxReal secondaryRelativeVn =
              (bodies[i].linearVelocity +
               bodies[i].angularVelocity.cross(secondaryContactArm))
                      .dot(secondaryNormal) -
              secondaryMeshNormalVelocity;
          stats->surfaceDeformableFinalizeSecondaryRows++;
          if (secondaryRelativeVn > diagnosticVelocityTolerance) {
            stats
                ->surfaceDeformableFinalizeSecondaryResidualSeparationRows++;
            stats->surfaceDeformableFinalizeSecondaryResidualSeparation +=
                secondaryRelativeVn;
          }
        }
        if (manifoldRows > 0) {
          if (finalizeTopologyNodes.size() == numBodies)
            finalizeTopologyNodes[i].legacyStrictOwner = 1;
          bool hasDynamicIncident = false;
          bool hasRigidStaticIncident = false;
          bool hasNonOwnerDeformableIncident = false;
          for (physx::PxU32 c = 0; c < numContacts; ++c) {
            const AvbdContactConstraint &incident = contacts[c];
            const physx::PxU32 incidentBodyA =
                incident.header.bodyIndexA;
            const physx::PxU32 incidentBodyB =
                incident.header.bodyIndexB;
            if (incidentBodyA != i && incidentBodyB != i)
              continue;
            if (!isBodyVsStaticContact(
                    incidentBodyA, incidentBodyB, numBodies)) {
              hasDynamicIncident = true;
            } else if (!hasDeformableStaticAnchor(incident)) {
              hasRigidStaticIncident = true;
            } else if (!hasDeformablePositionTangentOwner(incident)) {
              hasNonOwnerDeformableIncident = true;
            }
          }
          stats->surfaceDeformableFinalizeManifoldBodies++;
          if (manifoldRows == 1)
            stats->surfaceDeformableFinalizeManifoldOneRowBodies++;
          else if (manifoldRows == 2)
            stats->surfaceDeformableFinalizeManifoldTwoRowBodies++;
          else if (manifoldRows == 3)
            stats->surfaceDeformableFinalizeManifoldThreeRowBodies++;
          else if (manifoldRows == 4)
            stats->surfaceDeformableFinalizeManifoldFourRowBodies++;
          else {
            stats->surfaceDeformableFinalizeManifoldOverFourRowBodies++;
            if (manifoldRows <= 8)
              stats
                  ->surfaceDeformableFinalizeManifoldFiveToEightRowBodies++;
            else if (manifoldRows <= 16)
              stats
                  ->surfaceDeformableFinalizeManifoldNineToSixteenRowBodies++;
            else
              stats
                  ->surfaceDeformableFinalizeManifoldOverSixteenRowBodies++;
          }
          if (mixedScale)
            stats->surfaceDeformableFinalizeManifoldMixedScaleBodies++;
          if (manifoldRank <
              physx::PxMin(manifoldRows, physx::PxU32(6)))
            stats->surfaceDeformableFinalizeManifoldRankDeficientBodies++;
          if (hasDynamicIncident)
            stats
                ->surfaceDeformableFinalizeManifoldDynamicIncidentBodies++;
          if (hasRigidStaticIncident)
            stats
                ->surfaceDeformableFinalizeManifoldRigidStaticIncidentBodies++;
          if (hasNonOwnerDeformableIncident)
            stats
                ->surfaceDeformableFinalizeManifoldNonOwnerDeformableIncidentBodies++;
        }
      }
      continue;
    }

    // Rigid body-static: material e from NP-combined patch restitution.
    // Compliant contacts (e < 0) treated as inelastic for now.
    const physx::PxReal e =
        (cc.restitution > 0.0f) ? physx::PxMin(cc.restitution, 1.0f) : 0.0f;
    physx::PxReal approachEff = approach;
    if (e > 0.0f && relativeVn < 0.0f)
      approachEff = physx::PxMax(approachEff, -relativeVn);
    bool restitutionOwned = false;
    physx::PxReal restitutionLinearDelta = 0.0f;
    if (e > 0.0f && !hasFiniteMaxImpulse) {
      restitutionOwned = applyBodyStaticRestitutionSpatialRow(
          bodies, numBodies, contacts, numContacts, i,
          linearVelAtSolveStart, angularVelAtSolveStart, dt, bounceThreshold,
          restitutionLinearDelta);
      if (stats && restitutionOwned && restitutionLinearDelta > 1.0e-7f) {
        stats->bodyStaticMaterialVelocityCorrections++;
        stats->bodyStaticRestitutionCorrections++;
        stats->bodyStaticMaterialVelocityDelta += restitutionLinearDelta;
        if (established) {
          stats->bodyStaticNormalSupportFinalizeCorrections++;
          stats->bodyStaticNormalSupportFinalizeDelta += restitutionLinearDelta;
        } else {
          stats->bodyStaticNormalOnsetFinalizeCorrections++;
          stats->bodyStaticNormalOnsetFinalizeDelta += restitutionLinearDelta;
        }
        stats->bodyStaticMaterialVelocityMaxDelta = physx::PxMax(
            stats->bodyStaticMaterialVelocityMaxDelta,
            restitutionLinearDelta);
      }
    } else if (e > 0.0f && approachEff > bounceThreshold) {
      const physx::PxReal desiredRelativeVn =
          physx::PxMin(e * approachEff, maxImpulseRelativeVn);
      if (stats)
        stats->bodyStaticMaterialAllowedSeparatingVelocity +=
            physx::PxMax(desiredRelativeVn, physx::PxReal(0.0f));
      const physx::PxReal deltaV =
          staticNormalVelocity + desiredRelativeVn - vn;
      bodies[i].linearVelocity += nd * deltaV;
      if (stats && physx::PxAbs(deltaV) > 1e-7f) {
        const physx::PxReal absDeltaV = physx::PxAbs(deltaV);
        stats->bodyStaticMaterialVelocityCorrections++;
        stats->bodyStaticRestitutionCorrections++;
        stats->bodyStaticMaterialVelocityDelta += absDeltaV;
        if (established) {
          stats->bodyStaticNormalSupportFinalizeCorrections++;
          stats->bodyStaticNormalSupportFinalizeDelta += absDeltaV;
        } else {
          stats->bodyStaticNormalOnsetFinalizeCorrections++;
          stats->bodyStaticNormalOnsetFinalizeDelta += absDeltaV;
        }
        stats->bodyStaticMaterialVelocityMaxDelta = physx::PxMax(
            stats->bodyStaticMaterialVelocityMaxDelta, absDeltaV);
      }
      restitutionOwned = true;
    }
    if (!restitutionOwned) {
      // Inelastic / resting: the position solve may clear the narrow-phase
      // overlap in this step, but that geometric correction is not impact
      // velocity. Preserve any separating velocity the body already had at
      // solve start (so an authored take-off is not cancelled), and remove
      // only the separating speed created by the contact correction.
      const physx::PxReal allowedRelativeVn =
          hasSolveStartVelocity
              ? physx::PxMin(
                    physx::PxMax(solveStartRelativeVn, physx::PxReal(0.0f)),
                    maxImpulseRelativeVn)
              : physx::PxReal(0.0f);
      if (stats)
        stats->bodyStaticMaterialAllowedSeparatingVelocity +=
            physx::PxMax(allowedRelativeVn, physx::PxReal(0.0f));
      const bool shouldClamp =
          hasSolveStartVelocity || worstViolation < -1e-5f ||
          splitDeepInitialDepenetration;
      if (shouldClamp && relativeVn > allowedRelativeVn) {
        const physx::PxReal deltaV = relativeVn - allowedRelativeVn;
        bodies[i].linearVelocity -= nd * deltaV;
        if (stats && deltaV > 1e-7f) {
          stats->bodyStaticMaterialVelocityCorrections++;
          stats->bodyStaticMaterialVelocityDelta += deltaV;
          if (established) {
            stats->bodyStaticNormalSupportFinalizeCorrections++;
            stats->bodyStaticNormalSupportFinalizeDelta += deltaV;
          } else {
            stats->bodyStaticNormalOnsetFinalizeCorrections++;
            stats->bodyStaticNormalOnsetFinalizeDelta += deltaV;
          }
          stats->bodyStaticMaterialVelocityMaxDelta = physx::PxMax(
              stats->bodyStaticMaterialVelocityMaxDelta, deltaV);
        }
      }
    }
  }

  if (finalizeTopologyNodes.size() == numBodies)
    recordSurfaceFinalizeOwnerDiscoveryComparison(
        finalizeTopologyNodes, finalizeProbeOwnedBodies, stats);

  // Dyn-dyn restitution: relative normal impulse with invMass split.
  // Apply only for free rigid pairs (no deformable); e and bounce threshold
  // from material/scene. Skip if either body already handled as body-static
  // dominant this frame would double-count; dyn-dyn contacts are exclusive.
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    if (hasDeformableStaticAnchor(cc) ||
        hasVelocityPassiveFrictionComponentOwner(cc))
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
    const physx::PxReal invMassA =
        bodies[bA].invMass * cc.invMassScaleA;
    const physx::PxReal invMassB =
        bodies[bB].invMass * cc.invMassScaleB;
    const physx::PxReal invSum = invMassA + invMassB;
    if (invSum < 1e-12f)
      continue;
    const physx::PxReal maxImpulseVrel =
        cc.maxImpulse < PX_MAX_REAL
            ? vrel0 + physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)) *
                          invSum
            : PX_MAX_REAL;
    const physx::PxReal desiredVrel =
        physx::PxMin(e * approach, maxImpulseVrel);
    if (vrel >= desiredVrel)
      continue;
    physx::PxReal j = (desiredVrel - vrel) / invSum;
    if (cc.maxImpulse < PX_MAX_REAL)
      j = physx::PxMin(
          j, physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)));
    bodies[bA].linearVelocity += n * (j * invMassA);
    bodies[bB].linearVelocity -= n * (j * invMassB);
  }
}

// Backward-compatible name used by postAlStages call site.
static void clampBodyStaticInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale,
    bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    bool enableMatrixFreeComponentOracle,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  applyAvbdMaterialNormalVelocity(bodies, numBodies, contacts, numContacts,
                                  linearVelAtSolveStart,
                                  angularVelAtSolveStart,
                                  finiteMaterialPoseSplit, dt,
                                  bounceApproachThreshold, lengthScale,
                                  hasJointConstraints,
                                  enableBoundedComponentProductionProbe,
                                  enableMatrixFreeComponentOracle,
                                  deformableNormalStageMask, stats);
}

static void recordBodyStaticNormalAlOwnership(
    const AvbdSolverBody *bodies, const AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numBodies,
    physx::PxReal avbdAlpha,
    const physx::PxArray<bool> *touchesKinematicShell,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats &stats) {
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &contact = contacts[c];
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies))
      continue;

    const bool dynamicIsA = bodyA < numBodies;
    const physx::PxReal linearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal angularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (linearScale <= 0.0f && angularScale <= 0.0f)
      continue;

    if (hasDeformableStaticAnchor(contact)) {
      stats.surfaceDeformableAlRows++;
      if (deformableNormalStageMask &&
          c < deformableNormalStageMask->size())
        (*deformableNormalStageMask)[c] |= 1u;
      if (hasDeformablePositionTangentOwner(contact))
        stats.surfaceDeformablePositionTangentRows += 2;
      const physx::PxU32 dynamicBody = dynamicIsA ? bodyA : bodyB;
      if (touchesKinematicShell &&
          dynamicBody < touchesKinematicShell->size() &&
          (*touchesKinematicShell)[dynamicBody])
        stats.surfaceDeformableShellSuppressedPrimalRows++;
      continue;
    }

    stats.bodyStaticNormalAlRows++;
    const bool rowWarmstartHit =
        contact.diagnosticNormalWarmstartHit != 0;
    const bool established = contact.contactManagerEstablished != 0;
    if (rowWarmstartHit) {
      stats.bodyStaticNormalWarmstartHits++;
      switch (contact.diagnosticNormalWarmstartAge) {
      case 0:
        stats.bodyStaticNormalWarmstartAge0++;
        break;
      case 1:
        stats.bodyStaticNormalWarmstartAge1++;
        break;
      case 2:
        stats.bodyStaticNormalWarmstartAge2++;
        break;
      case 3:
        stats.bodyStaticNormalWarmstartAge3++;
        break;
      default:
        break;
      }
    } else {
      stats.bodyStaticNormalWarmstartMisses++;
    }
    if (established) {
      stats.bodyStaticNormalManagerSupportRows++;
      if (!rowWarmstartHit)
        stats.bodyStaticNormalRowMissOnManagerSupportRows++;
      switch (contact.contactManagerAge) {
      case 0:
        stats.bodyStaticNormalManagerAge0++;
        break;
      case 1:
        stats.bodyStaticNormalManagerAge1++;
        break;
      case 2:
        stats.bodyStaticNormalManagerAge2++;
        break;
      case 3:
        stats.bodyStaticNormalManagerAge3++;
        break;
      default:
        break;
      }
    } else {
      stats.bodyStaticNormalManagerOnsetRows++;
    }
    stats.bodyStaticNormalRestoredLambdaMax = physx::PxMax(
        stats.bodyStaticNormalRestoredLambdaMax,
        physx::PxAbs(contact.diagnosticRestoredNormalLambda));
    stats.bodyStaticNormalRestoredPenaltyMax = physx::PxMax(
        stats.bodyStaticNormalRestoredPenaltyMax,
        physx::PxAbs(contact.diagnosticRestoredNormalPenalty));
    stats.bodyStaticNormalInitialPenaltyMax = physx::PxMax(
        stats.bodyStaticNormalInitialPenaltyMax,
        physx::PxAbs(contact.diagnosticInitialNormalPenalty));

    const physx::PxVec3 worldA =
        bodyA < numBodies
            ? bodies[bodyA].position +
                  bodies[bodyA].rotation.rotate(contact.contactPointA)
            : contact.contactPointA;
    const physx::PxVec3 worldB =
        bodyB < numBodies
            ? bodies[bodyB].position +
                  bodies[bodyB].rotation.rotate(contact.contactPointB)
            : contact.contactPointB;
    const physx::PxReal postAlNormalCoordinate =
        (worldA - worldB).dot(contact.contactNormal);
    const physx::PxReal postAlRawViolation =
        postAlNormalCoordinate + contact.penetrationDepth;
    const physx::PxReal postAlViolation =
        postAlRawViolation - avbdAlpha * contact.C0;
    const physx::PxReal preAlRawPenetration = physx::PxMax(
        -contact.diagnosticPreAlRawViolation, physx::PxReal(0.0f));
    const physx::PxReal postAlRawPenetration =
        physx::PxMax(-postAlRawViolation, physx::PxReal(0.0f));
    const physx::PxReal alphaC0Offset = physx::PxAbs(
        contact.diagnosticPreAlRawViolation -
        contact.diagnosticPreAlViolation);
    const physx::PxReal preAlPenetration =
        physx::PxMax(-contact.diagnosticPreAlViolation, physx::PxReal(0.0f));
    const physx::PxReal postAlPenetration =
        physx::PxMax(-postAlViolation, physx::PxReal(0.0f));
    stats.bodyStaticNormalPreAlRawPenetration += preAlRawPenetration;
    stats.bodyStaticNormalPostAlRawPenetration += postAlRawPenetration;
    stats.bodyStaticNormalAlphaC0Offset += alphaC0Offset;
    stats.bodyStaticNormalPreAlPenetration += preAlPenetration;
    stats.bodyStaticNormalPostAlPenetration += postAlPenetration;
    stats.bodyStaticNormalPostAlSeparation +=
        physx::PxMax(postAlViolation, physx::PxReal(0.0f));
    const physx::PxReal alNormalDelta =
        postAlNormalCoordinate - contact.diagnosticPreAlNormalCoordinate;
    const physx::PxReal alOutwardDistance =
        physx::PxMax(alNormalDelta, physx::PxReal(0.0f));
    stats.bodyStaticNormalAlOutwardDistance += alOutwardDistance;
    stats.bodyStaticNormalAlInwardDistance +=
        physx::PxMax(-alNormalDelta, physx::PxReal(0.0f));
    if (established) {
      stats.bodyStaticNormalSupportPreAlRawPenetration +=
          preAlRawPenetration;
      stats.bodyStaticNormalSupportPreAlPenetration += preAlPenetration;
      stats.bodyStaticNormalSupportPostAlRawPenetration +=
          postAlRawPenetration;
      stats.bodyStaticNormalSupportPostAlPenetration += postAlPenetration;
      stats.bodyStaticNormalSupportAlphaC0Offset += alphaC0Offset;
      stats.bodyStaticNormalSupportAlOutwardDistance += alOutwardDistance;
    } else {
      stats.bodyStaticNormalOnsetPreAlRawPenetration += preAlRawPenetration;
      stats.bodyStaticNormalOnsetPreAlPenetration += preAlPenetration;
      stats.bodyStaticNormalOnsetPostAlRawPenetration +=
          postAlRawPenetration;
      stats.bodyStaticNormalOnsetPostAlPenetration += postAlPenetration;
      stats.bodyStaticNormalOnsetAlphaC0Offset += alphaC0Offset;
      stats.bodyStaticNormalOnsetAlOutwardDistance += alOutwardDistance;
    }
  }
  stats.bodyStaticNormalAlEvaluations =
      physx::PxU64(stats.bodyStaticNormalAlRows) *
      physx::PxU64(stats.totalIterations);
  stats.surfaceDeformableAlEvaluations =
      physx::PxU64(stats.surfaceDeformableAlRows) *
      physx::PxU64(stats.totalIterations);
  stats.surfaceDeformablePositionTangentEvaluations =
      physx::PxU64(stats.surfaceDeformablePositionTangentRows) *
      physx::PxU64(stats.totalIterations);
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

void AvbdSolver::captureBodyStaticNormalDiagnosticStart(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts) {
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &contact = contacts[c];
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    contact.diagnosticInitialNormalPenalty = contact.header.penalty;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
        hasDeformableStaticAnchor(contact)) {
      contact.diagnosticPreAlViolation = 0.0f;
      contact.diagnosticPreAlRawViolation = 0.0f;
      contact.diagnosticPreAlNormalCoordinate = 0.0f;
      continue;
    }

    const physx::PxVec3 worldA =
        bodyA < numBodies
            ? bodies[bodyA].position +
                  bodies[bodyA].rotation.rotate(contact.contactPointA)
            : contact.contactPointA;
    const physx::PxVec3 worldB =
        bodyB < numBodies
            ? bodies[bodyB].position +
                  bodies[bodyB].rotation.rotate(contact.contactPointB)
            : contact.contactPointB;
    contact.diagnosticPreAlNormalCoordinate =
        (worldA - worldB).dot(contact.contactNormal);
    contact.diagnosticPreAlRawViolation =
        contact.diagnosticPreAlNormalCoordinate + contact.penetrationDepth;
    contact.diagnosticPreAlViolation =
        contact.diagnosticPreAlRawViolation - mConfig.avbdAlpha * contact.C0;
  }
}

void AvbdSolver::solve(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                       const AvbdBodyConstraintMap *contactMap,
                       AvbdColorBatch *colorBatches, physx::PxU32 numColors,
                       physx::PxU32 iterationOverride,
                       AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solve", 0);

  if (!mInitialized || numBodies == 0) {
    return;
  }

  stats.numBodies = numBodies;
  stats.numContacts = numContacts;

  physx::PxReal invDt = 1.0f / dt;

  physx::PxArray<bool> touchesKinematicShell(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    touchesKinematicShell[i] = false;

  // Stage 1: Prediction
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  // The contact BCD path below uses body-level Jacobi snapshots and does not
  // consume the legacy solver-owned graph coloring.  Building that shared
  // coloring here is both redundant and unsafe when independent island tasks
  // enter the same solver concurrently.

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
  physx::PxArray<physx::PxVec3> angularVelAtSolveStart;
  if (numContacts > 0) {
    linearVelAtSolveStart.resize(numBodies);
    angularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      linearVelAtSolveStart[i] = bodies[i].linearVelocity;
      angularVelAtSolveStart[i] = bodies[i].angularVelocity;
    }
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
        } else {
          bodies[i].position = bodies[i].prevPosition +
                               bodies[i].linearVelocity * dt +
                               gravity * (accelWeight * dt * dt);
          bodies[i].rotation = bodies[i].inertialRotation;
        }
        bodies[i].projectLockedPose(bodies[i].prevPosition,
                                    bodies[i].prevRotation);
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
      // A freshly prepared row carries the reference unit-mass minimum.
      // Replace that sentinel with the mass/time-scaled floor even when the
      // physical floor is lower; otherwise sub-unit-mass scenes become
      // artificially stiffer and are not scale-equivalent.
      if (contacts[c].header.penalty <= mConfig.avbdPenaltyMin) {
        contacts[c].header.penalty = penaltyFloor;
      } else if (contacts[c].header.penalty < penaltyFloor) {
        contacts[c].header.penalty = penaltyFloor;
      }
      // Also floor tangent penalties (ref: standalone floors all 3 rows)
      if (contacts[c].tangentPenalty0 <= mConfig.avbdPenaltyMin) {
        contacts[c].tangentPenalty0 = penaltyFloor;
      } else if (contacts[c].tangentPenalty0 < penaltyFloor) {
        contacts[c].tangentPenalty0 = penaltyFloor;
      }
      if (contacts[c].tangentPenalty1 <= mConfig.avbdPenaltyMin) {
        contacts[c].tangentPenalty1 = penaltyFloor;
      } else if (contacts[c].tangentPenalty1 < penaltyFloor) {
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
      if (hasDeformableStaticAnchor(contacts[c])) {
        // Moving mesh anchor: no alpha-soften on normals.
        contacts[c].C0 = 0.0f;
        continue;
      }
      if (isBodyVsStaticContact(bA, bB, numBodies) &&
          contacts[c].contactManagerEstablished) {
        // An uninterrupted rigid support is owned by its raw position-level
        // normal row.  Alpha-softened C0 remains an onset stabilization rule.
        contacts[c].C0 = 0.0f;
        continue;
      }

      physx::PxReal rawC0 = (wA - wB).dot(contacts[c].contactNormal) +
                            contacts[c].penetrationDepth;

      // Depth-adaptive C0 clamping: for deep penetrations (fast impacts),
      // reduce C0 so that alpha blending does not over-soften the correction.
      const physx::PxReal c0Threshold = 0.05f * mConfig.lengthScale;
      const physx::PxReal c0MaxDepth = 0.20f * mConfig.lengthScale;
      if (rawC0 < -c0Threshold) {
        physx::PxReal t = PxClamp(
            (c0MaxDepth + rawC0) / (c0MaxDepth - c0Threshold), 0.0f, 1.0f);
        rawC0 *= t;
      }
      contacts[c].C0 = rawC0;
    }
  }

  captureBodyStaticNormalDiagnosticStart(bodies, numBodies, contacts,
                                         numContacts);

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

    // Chebyshev extrapolation can overshoot a deep quasi-static
    // body-vs-static overlap after the ordinary block step clears it.
    const bool useChebyshev =
        !hasDeformableAnchorContact &&
        mConfig.chebyshevRho > 0.0f &&
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
        physx::PxMax(4.0f * mConfig.positionTolerance /
                         physx::PxMax(mConfig.lengthScale, 1e-6f),
                     1e-4f);
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
        for (physx::PxU32 i = 0; i < numBodies; ++i) {
          if (bodies[i].invMass > 0.0f)
            bodies[i].projectLockedPose(bodies[i].prevPosition,
                                        bodies[i].prevRotation);
        }
        stats.totalIterations++;
      }
      {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt, stats);
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
          const physx::PxVec3 gsPosition = bodies[i].position;
          const physx::PxQuat gsRotation = bodies[i].rotation;
          // Position relaxation
          const physx::PxVec3 relaxedPosition = chebyPrevPrevPos[i] +
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
          const physx::PxQuat relaxedRotation = qBlend.getNormalized();

          // Chebyshev acceleration assumes a smooth equality system.  A
          // unilateral body-static active set has zero energy anywhere on
          // its satisfied side, so extrapolating after the ordinary block
          // step has cleared every row only creates artificial separation.
          // Keep Chebyshev while any row still needs correction; reject only
          // the no-benefit outward overshoot of an already-cleared set.
          bool rejectBodyStaticOvershoot = false;
          if (hasBodyStaticContact) {
            physx::PxReal minGsViolation = PX_MAX_REAL;
            physx::PxReal minRelaxedViolation = PX_MAX_REAL;
            bool foundBodyStatic = false;
            bool deepQuasistaticInitialOverlap = false;
            const physx::PxU32 *mapIndices = nullptr;
            physx::PxU32 mapCount = 0;
            if (contactMap && contactMap->numBodies > 0)
              contactMap->getBodyConstraints(i, mapIndices, mapCount);
            const physx::PxU32 loopCount =
                mapIndices ? mapCount : numContacts;
            for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
              const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
              const physx::PxU32 bA = contacts[c].header.bodyIndexA;
              const physx::PxU32 bB = contacts[c].header.bodyIndexB;
              if (!isBodyVsStaticContact(bA, bB, numBodies) ||
                  (bA != i && bB != i))
                continue;

              const bool dynIsA = (bA == i);
              const physx::PxVec3 gsWorldA =
                  dynIsA
                      ? gsPosition +
                            gsRotation.rotate(contacts[c].contactPointA)
                      : contacts[c].contactPointA;
              const physx::PxVec3 gsWorldB =
                  dynIsA
                      ? contacts[c].contactPointB
                      : gsPosition +
                            gsRotation.rotate(contacts[c].contactPointB);
              const physx::PxVec3 relaxedWorldA =
                  dynIsA
                      ? relaxedPosition +
                            relaxedRotation.rotate(contacts[c].contactPointA)
                      : contacts[c].contactPointA;
              const physx::PxVec3 relaxedWorldB =
                  dynIsA
                      ? contacts[c].contactPointB
                      : relaxedPosition +
                            relaxedRotation.rotate(contacts[c].contactPointB);
              const physx::PxReal gsViolation =
                  (gsWorldA - gsWorldB).dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
              const physx::PxReal relaxedViolation =
                  (relaxedWorldA - relaxedWorldB)
                          .dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
              minGsViolation =
                  physx::PxMin(minGsViolation, gsViolation);
              minRelaxedViolation =
                  physx::PxMin(minRelaxedViolation, relaxedViolation);
              foundBodyStatic = true;

              const physx::PxVec3 initialWorldA =
                  dynIsA
                      ? bodies[i].prevPosition +
                            bodies[i].prevRotation.rotate(
                                contacts[c].contactPointA)
                      : contacts[c].staticPrevWorldPoint;
              const physx::PxVec3 initialWorldB =
                  dynIsA
                      ? contacts[c].staticPrevWorldPoint
                      : bodies[i].prevPosition +
                            bodies[i].prevRotation.rotate(
                                contacts[c].contactPointB);
              const physx::PxReal initialViolation =
                  (initialWorldA - initialWorldB)
                          .dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
              const physx::PxVec3 outwardNormal =
                  contacts[c].contactNormal * (dynIsA ? 1.0f : -1.0f);
              const physx::PxReal approach =
                  linearVelAtSolveStart.size() == numBodies
                      ? physx::PxMax(
                            0.0f,
                            -linearVelAtSolveStart[i].dot(outwardNormal))
                      : 0.0f;
              const physx::PxReal deepOverlapThreshold =
                  0.05f * physx::PxMax(mConfig.lengthScale, 1e-6f);
              if (initialViolation < -deepOverlapThreshold &&
                  approach <= mConfig.bounceApproachSpeedThreshold())
                deepQuasistaticInitialOverlap = true;
            }
            const physx::PxReal activeSetTolerance =
                0.01f * physx::PxMax(mConfig.lengthScale, 1e-6f);
            rejectBodyStaticOvershoot =
                foundBodyStatic && deepQuasistaticInitialOverlap &&
                minGsViolation >= activeSetTolerance &&
                minRelaxedViolation >
                    minGsViolation + activeSetTolerance;
          }

          bodies[i].position =
              rejectBodyStaticOvershoot ? gsPosition : relaxedPosition;
          bodies[i].rotation =
              rejectBodyStaticOvershoot ? gsRotation : relaxedRotation;
        }
      }
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        if (bodies[i].invMass > 0.0f)
          bodies[i].projectLockedPose(bodies[i].prevPosition,
                                      bodies[i].prevRotation);
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
               numContacts > 0 ? &angularVelAtSolveStart : nullptr,
               /*allowRigidDeepPoseRecoverySplit=*/true,
               /*allowRigidFiniteMaterialPoseSplit=*/true,
               nullptr, 0, nullptr, 0, touchesKinematicShell, nullptr,
               nullptr, 0, false, false, false, nullptr, 0, stats);

}

void AvbdSolver::postAlStages(
    physx::PxReal dt, physx::PxReal invDt, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const physx::PxVec3 &gravity,
    bool hasBodyStaticContact, bool deformableFastImpactIsland,
    const physx::PxArray<bool> &touchingBodyStatic,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    bool allowRigidDeepPoseRecoverySplit,
    bool allowRigidFiniteMaterialPoseSplit,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    const physx::PxArray<bool> &touchesKinematicShell,
    const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    bool hasJointConstraints, bool skipBodyStaticFriction,
    bool applyVelocityDamping,
    AvbdSoftParticle *softParticlesForVel,
    physx::PxU32 numSoftParticlesForVel, AvbdSolverStats &stats) {

  const bool hasKinematicShellContacts =
      shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0;
  static const physx::PxReal kShellFastImpactSpeed =
      AvbdConstants::AVBD_SHELL_FAST_IMPACT_SPEED;

  // Deep initial overlap is an emergency geometric recovery.  Its nonlinear
  // rotation can contain motion outside the final contact-row span, so that
  // component cannot be removed later by any correct material impulse.  Mark
  // the strict contact-only capability slice now and exclude its block pose
  // recovery during pose-to-velocity reconstruction.  The inertial pose still
  // preserves authored motion and gravity; only the emergency contact offset
  // is split from velocity.
  physx::PxArray<bool> splitRigidDeepPoseRecovery(numBodies);
  physx::PxArray<bool> splitRigidFiniteMaterialPose(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    splitRigidDeepPoseRecovery[i] =
        allowRigidDeepPoseRecoverySplit &&
        isRigidDeepBodyStaticRecoverySplitSupported(
            bodies, numBodies, contacts, numContacts, i,
            mConfig.lengthScale);
    splitRigidFiniteMaterialPose[i] =
        allowRigidFiniteMaterialPoseSplit &&
        isRigidFiniteBodyStaticMaterialSplitSupported(
            bodies, numBodies, contacts, numContacts, i,
            mConfig.lengthScale);
  }

  // Diagnostic-only contact identity ledger. These bits never participate in
  // a solve decision; they correlate the three ordinary deformable/static
  // normal owners within this one island/substep.
  physx::PxArray<physx::PxU8> deformableNormalStageMask;
  physx::PxArray<physx::PxU8> *deformableNormalStageMaskPtr = nullptr;
  if (mConfig.enableStageOwnershipDiagnostics) {
    deformableNormalStageMask.resize(numContacts);
    for (physx::PxU32 c = 0; c < numContacts; ++c)
      deformableNormalStageMask[c] = 0u;
    deformableNormalStageMaskPtr = &deformableNormalStageMask;
  }

  recordBodyStaticNormalAlOwnership(
      bodies, contacts, numContacts, numBodies, mConfig.avbdAlpha,
      &touchesKinematicShell, deformableNormalStageMaskPtr, stats);

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
                                           depenSweeps, shellSkipDepen,
                                           deformableNormalStageMaskPtr,
                                           &stats);
  }
  if (hasKinematicShellContacts) {
    PX_PROFILE_ZONE("AVBD.kinematicShellDepenetration", 0);
    applyKinematicShellNormalDepenetrationSweeps(
        bodies, numBodies, shellParticles, numShellParticles, shellContacts,
        numShellContacts, gravity, dt, 8u, &stats);
  }
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  physx::PxArray<physx::PxVec3> postDepenPos(numBodies);
  physx::PxArray<physx::PxQuat> postDepenRot(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    postDepenPos[i] = bodies[i].position;
    postDepenRot[i] = bodies[i].rotation;
  }

  if (contacts && numContacts > 0 && !skipBodyStaticFriction) {
    PX_PROFILE_ZONE("AVBD.bodyStaticFriction", 0);
    applyBodyStaticFrictionSweeps(bodies, numBodies, contacts, numContacts,
                                  gravity, dt, 6u, &postDepenPos, &postDepenRot,
                                  shellSkipDepen, &stats);
  }
  if (hasKinematicShellContacts) {
    PX_PROFILE_ZONE("AVBD.kinematicShellFriction", 0);
    applyKinematicShellFrictionSweeps(
        bodies, numBodies, shellParticles, numShellParticles, shellContacts,
        numShellContacts, dt, 4u, &postBlockPos, &postBlockRot, &stats);
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  // Finalize velocity: block motion + friction/motor tangents; exclude depen.
  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;
        bool physicalContactTangentMaterialOwner = false;
        physx::PxU32 physicalContactTangentMaterialOwnerIndex =
            PX_MAX_U32;
        for (physx::PxU32 c = 0; c < numContacts; ++c) {
          if (hasVelocityTangentMaterialOwner(contacts[c]) &&
              (contacts[c].header.bodyIndexA == i ||
               contacts[c].header.bodyIndexB == i)) {
            physicalContactTangentMaterialOwner = true;
            physicalContactTangentMaterialOwnerIndex = c;
            break;
          }
        }

        const physx::PxVec3 blockPositionForVelocity =
            (splitRigidDeepPoseRecovery[i] ||
             splitRigidFiniteMaterialPose[i])
                ? bodies[i].inertialPosition
                : postBlockPos[i];
        const physx::PxVec3 vFromBlock =
            (blockPositionForVelocity - bodies[i].prevPosition) * invDt;
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

        if (applyVelocityDamping &&
            !physicalContactTangentMaterialOwner)
          bodies[i].linearVelocity *= mConfig.velocityDamping;

        const bool unconstrainedAngularMotion =
            numContacts == 0 && !hasKinematicShellContacts &&
            (!d6Joints || numD6 == 0);
        bool physicalSlerpPositionDrive = false;
        if (d6Joints) {
          for (physx::PxU32 j = 0; j < numD6; ++j) {
            const AvbdD6JointConstraint &joint = d6Joints[j];
            if (joint.header.bodyIndexA != i &&
                joint.header.bodyIndexB != i)
              continue;
            if ((joint.sourceFlags &
                 AvbdD6JointConstraint::
                     eSLERP_POSITION_DRIVE_ACTIVE) != 0)
              physicalSlerpPositionDrive = true;
            if (physicalSlerpPositionDrive)
              break;
          }
        }
        if (!unconstrainedAngularMotion) {
          const physx::PxQuat blockRotationForVelocity =
              (splitRigidDeepPoseRecovery[i] ||
               splitRigidFiniteMaterialPose[i])
                  ? bodies[i].inertialRotation
                  : postBlockRot[i];
          physx::PxQuat deltaQBlock =
              blockRotationForVelocity *
              bodies[i].prevRotation.getConjugate();
          if (deltaQBlock.w < 0.0f)
            deltaQBlock = -deltaQBlock;
          const physx::PxVec3 wBlock =
              physx::PxVec3(deltaQBlock.x, deltaQBlock.y, deltaQBlock.z) *
              (2.0f * invDt);
          physx::PxQuat deltaQFr =
              bodies[i].rotation * postDepenRot[i].getConjugate();
          if (deltaQFr.w < 0.0f)
            deltaQFr = -deltaQFr;
          const physx::PxVec3 wFr =
              physx::PxVec3(deltaQFr.x, deltaQFr.y, deltaQFr.z) *
              (2.0f * invDt);
          bodies[i].angularVelocity = wBlock + wFr;
          // Explicit position/velocity targets already own their damping and
          // material semantics. Applying solver-wide stabilization decay
          // again turns a constant-speed target into a frame-rate-dependent
          // lag and changes a passive manifold's inertial baseline.
          if (!physicalSlerpPositionDrive &&
              !physicalContactTangentMaterialOwner)
            bodies[i].angularVelocity *= mConfig.angularDamping;
        }

        if (physicalContactTangentMaterialOwnerIndex != PX_MAX_U32 &&
            hasVelocityFrictionManifoldOwner(
                contacts[physicalContactTangentMaterialOwnerIndex]) &&
            linearVelAtSolveStart && angularVelAtSolveStart &&
            linearVelAtSolveStart->size() == numBodies &&
            angularVelAtSolveStart->size() == numBodies) {
          // The position solve owns geometry, but its pose delta and AL
          // multipliers are not material impulses. Reconstruct this strict
          // manifold from solve-start inertial velocity. Its coupled
          // post-reconstruction owner rebuilds both the nonnegative normal
          // response and the tangent target from that single baseline.
          physx::PxVec3 baselineLinear =
              (*linearVelAtSolveStart)[i] + gravity * dt;
          physx::PxVec3 baselineAngular =
              (*angularVelAtSolveStart)[i];
          bodies[i].projectLockedLinearVector(baselineLinear);
          bodies[i].projectLockedAngularVector(baselineAngular);
          bodies[i].linearVelocity = baselineLinear;
          bodies[i].angularVelocity = baselineAngular;
          bodies[i].projectLockedVelocities();
        } else if (
            physicalContactTangentMaterialOwnerIndex != PX_MAX_U32 &&
            hasVelocityTangentTargetNormalSpan(
                contacts[physicalContactTangentMaterialOwnerIndex]) &&
            linearVelAtSolveStart && angularVelAtSolveStart &&
            linearVelAtSolveStart->size() == numBodies &&
            angularVelAtSolveStart->size() == numBodies) {
          const AvbdContactConstraint &targetContact =
              contacts[physicalContactTangentMaterialOwnerIndex];
          const bool dynamicIsA =
              targetContact.header.bodyIndexA == i;
          const physx::PxVec3 dynamicNormal =
              targetContact.contactNormal *
              (dynamicIsA ? 1.0f : -1.0f);
          const physx::PxVec3 localPoint =
              dynamicIsA ? targetContact.contactPointA
                         : targetContact.contactPointB;
          const physx::PxVec3 contactArm =
              bodies[i].prevRotation.rotate(localPoint);
          const physx::PxVec3 angularJacobian =
              contactArm.cross(dynamicNormal);
          const physx::PxReal linearScale =
              dynamicIsA ? targetContact.invMassScaleA
                         : targetContact.invMassScaleB;
          const physx::PxReal angularScale =
              dynamicIsA ? targetContact.invInertiaScaleA
                         : targetContact.invInertiaScaleB;
          const physx::PxVec3 normalLinearResponse =
              dynamicNormal * (bodies[i].invMass * linearScale);
          const physx::PxVec3 normalAngularResponse =
              bodies[i].invInertiaWorld *
              (angularJacobian * angularScale);
          const physx::PxReal normalResponse =
              dynamicNormal.dot(normalLinearResponse) +
              angularJacobian.dot(normalAngularResponse);
          if (normalResponse > 1.0e-12f) {
            physx::PxVec3 baselineLinear =
                (*linearVelAtSolveStart)[i] + gravity * dt;
            physx::PxVec3 baselineAngular =
                (*angularVelAtSolveStart)[i];
            bodies[i].projectLockedLinearVector(baselineLinear);
            bodies[i].projectLockedAngularVector(baselineAngular);
            const physx::PxVec3 poseDeltaLinear =
                bodies[i].linearVelocity - baselineLinear;
            const physx::PxVec3 poseDeltaAngular =
                bodies[i].angularVelocity - baselineAngular;
            const physx::PxReal normalImpulse = physx::PxMax(
                0.0f,
                (dynamicNormal.dot(poseDeltaLinear) +
                 angularJacobian.dot(poseDeltaAngular)) /
                    normalResponse);
            bodies[i].linearVelocity =
                baselineLinear + normalLinearResponse * normalImpulse;
            bodies[i].angularVelocity =
                baselineAngular + normalAngularResponse * normalImpulse;
            bodies[i].projectLockedVelocities();
          }
        }

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
    if (d6Joints && numD6 > 0) {
      PX_PROFILE_ZONE("AVBD.projectBodyStaticLockedD6LinearVelocity", 0);
      projectBodyStaticLockedD6LinearVelocities(bodies, numBodies, d6Joints,
                                                numD6);
    }
    // Material normal response: body-static e / deformable e=0 / dyn-dyn bounce.
    // Gate on numContacts (not hasBodyStaticContact) so pure dyn-dyn islands
    // still consume restitution e (criterion 2 / Entry 160).
    if (contacts && numContacts > 0) {
      PX_PROFILE_ZONE("AVBD.materialNormalVelocity", 0);
      clampBodyStaticInelasticNormalVelocities(
          bodies, numBodies, contacts, numContacts, linearVelAtSolveStart,
          angularVelAtSolveStart, &splitRigidFiniteMaterialPose, dt,
          mConfig.bounceApproachSpeedThreshold(), mConfig.lengthScale,
          hasJointConstraints,
          mConfig.enableBoundedComponentProductionProbe,
          mConfig.enableMatrixFreeComponentOracle,
          deformableNormalStageMaskPtr, &stats);
      if (deformableNormalStageMaskPtr) {
        for (physx::PxU32 c = 0; c < numContacts; ++c) {
          const physx::PxU8 mask = deformableNormalStageMask[c];
          if ((mask & 3u) == 3u)
            stats.surfaceDeformableAlDepenetrationRows++;
          if ((mask & 5u) == 5u)
            stats.surfaceDeformableAlFinalizeRows++;
          if ((mask & 6u) == 6u)
            stats.surfaceDeformableDepenetrationFinalizeRows++;
          if ((mask & 7u) == 7u)
            stats.surfaceDeformableAlDepenetrationFinalizeRows++;
        }
      }
      PX_PROFILE_ZONE("AVBD.contactTargetVelocity", 0);
      applyAvbdContactTargetVelocity(bodies, numBodies, contacts, numContacts,
                                     dt, &stats);
    }
    if (hasKinematicShellContacts) {
      PX_PROFILE_ZONE("AVBD.kinematicShellInelasticVel", 0);
      clampKinematicShellInelasticNormalVelocities(
          bodies, numBodies, shellParticles, numShellParticles, shellContacts,
          numShellContacts, shellLinearVelAtSolveStart, dt, &stats);
    }
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f)
        bodies[i].projectLockedVelocities();
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
    physx::PxU32 iterationOverride,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solveIsland", 0);

  // solveIsland is the sole public island entry and owns transient
  // classification before dispatching to either internal solve module.
  stats.reset();
  const bool hasJoints = (numD6 > 0 || numGear > 0);
  const bool hasDeformableSoftVbd =
      softParticles && numSoftParticles > 0 && softBodies &&
      numSoftBodies > 0 && softContacts && numSoftContacts > 0;
  const bool contactOnlyTargetOwnership =
      !hasJoints && !hasDeformableSoftVbd;
  physx::PxArray<physx::PxU32> rigidStaticContactsPerBody(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    rigidStaticContactsPerBody[i] = 0;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    contacts[c].header.flags = physx::PxU16(
        contacts[c].header.flags &
        ~(AvbdContactConstraintFlags::eVELOCITY_TANGENT_TARGET_OWNER |
          AvbdContactConstraintFlags::
              eVELOCITY_TANGENT_TARGET_NORMAL_SPAN |
          AvbdContactConstraintFlags::
              eVELOCITY_TANGENT_TARGET_MANIFOLD_OWNER |
          AvbdContactConstraintFlags::
              eVELOCITY_PASSIVE_FRICTION_MANIFOLD_OWNER |
          AvbdContactConstraintFlags::
              eVELOCITY_PASSIVE_FRICTION_COMPONENT_OWNER |
          AvbdContactConstraintFlags::
              eDEFORMABLE_POSITION_TANGENT_OWNER));
    if (!contactOnlyTargetOwnership ||
        !isBodyVsStaticContact(contacts[c].header.bodyIndexA,
                               contacts[c].header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contacts[c]))
      continue;
    const physx::PxU32 bodyIndex =
        contacts[c].header.bodyIndexA < numBodies
            ? contacts[c].header.bodyIndexA
            : contacts[c].header.bodyIndexB;
    rigidStaticContactsPerBody[bodyIndex]++;
  }
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &contact = contacts[c];
    if (!contactOnlyTargetOwnership ||
        !isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contact) ||
        hasKinematicShellAnchor(contact))
      continue;
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    const physx::PxReal targetNormal =
        contact.targetVelocity.dot(contact.contactNormal);
    const physx::PxReal targetTangent0 =
        contact.targetVelocity.dot(contact.tangent0);
    const physx::PxReal targetTangent1 =
        contact.targetVelocity.dot(contact.tangent1);
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxReal dynamicLinearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal dynamicAngularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    const bool hasTangentTarget =
        physx::PxAbs(targetTangent0) > 1e-6f ||
        physx::PxAbs(targetTangent1) > 1e-6f;
    const bool defaultDynamicScales =
        physx::PxAbs(dynamicLinearScale - 1.0f) <= 1e-6f &&
        physx::PxAbs(dynamicAngularScale - 1.0f) <= 1e-6f;
    const physx::PxU32 angularLocks =
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
    const bool allAngularMotionLocked =
        (bodies[bodyIndex].lockFlags & angularLocks) == angularLocks;
    const physx::PxVec3 staticPoint =
        dynamicIsA ? contact.contactPointB : contact.contactPointA;
    const physx::PxReal lengthTolerance =
        1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0f);
    const bool stationaryStatic =
        (staticPoint - contact.staticPrevWorldPoint).magnitudeSquared() <=
        lengthTolerance * lengthTolerance;
    const bool pureUnlimitedTangentTarget =
        physx::PxAbs(targetNormal) <= 1e-6f &&
        contact.maxImpulse > 1.0e20f;
    const bool strictFiniteCombinedTarget =
        targetNormal > 1e-6f && contact.maxImpulse >= 0.0f &&
        contact.maxImpulse < PX_MAX_REAL &&
        physx::PxIsFinite(contact.maxImpulse) && allAngularMotionLocked &&
        stationaryStatic;
    if (rigidStaticContactsPerBody[bodyIndex] == 1 &&
        (contact.friction > 0.0f || contact.staticFriction > 0.0f) &&
        hasTangentTarget && contact.restitution == 0.0f &&
        defaultDynamicScales &&
        (pureUnlimitedTangentTarget || strictFiniteCombinedTarget)) {
      contact.header.flags = physx::PxU16(
          contact.header.flags |
          AvbdContactConstraintFlags::eVELOCITY_TANGENT_TARGET_OWNER);

      // The nonlinear position solve may rotate a cached local contact point
      // while enforcing its normal row.  For a central contact on an
      // isotropic body, that row has no physical angular Jacobian and cannot
      // create tangent-space generalized velocity.  Mark this independently
      // so velocity reconstruction can retain only the normal impulse span
      // before the unique tangent target is applied.
      if (!pureUnlimitedTangentTarget)
        continue;

      const AvbdSolverBody &body = bodies[bodyIndex];
      const physx::PxVec3 dynamicNormal =
          contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
      const physx::PxVec3 localPoint =
          dynamicIsA ? contact.contactPointA : contact.contactPointB;
      const physx::PxVec3 normalAngularJacobian =
          body.rotation.rotate(localPoint).cross(dynamicNormal);
      const physx::PxMat33 &invInertia = body.invInertiaWorld;
      const physx::PxReal inertiaMagnitude = physx::PxMax(
          1.0f,
          physx::PxMax(
              physx::PxAbs(invInertia.column0.x),
              physx::PxMax(physx::PxAbs(invInertia.column1.y),
                           physx::PxAbs(invInertia.column2.z))));
      const physx::PxReal inertiaTolerance = 1.0e-5f * inertiaMagnitude;
      const bool isotropicInertia =
          physx::PxAbs(invInertia.column0.x - invInertia.column1.y) <=
              inertiaTolerance &&
          physx::PxAbs(invInertia.column0.x - invInertia.column2.z) <=
              inertiaTolerance &&
          physx::PxAbs(invInertia.column0.y) <= inertiaTolerance &&
          physx::PxAbs(invInertia.column0.z) <= inertiaTolerance &&
          physx::PxAbs(invInertia.column1.x) <= inertiaTolerance &&
          physx::PxAbs(invInertia.column1.z) <= inertiaTolerance &&
          physx::PxAbs(invInertia.column2.x) <= inertiaTolerance &&
          physx::PxAbs(invInertia.column2.y) <= inertiaTolerance;
      const bool centralNormal =
          normalAngularJacobian.magnitudeSquared() <=
          lengthTolerance * lengthTolerance;
      if (isotropicInertia && centralNormal && stationaryStatic) {
        contact.header.flags = physx::PxU16(
            contact.header.flags |
            AvbdContactConstraintFlags::
                eVELOCITY_TANGENT_TARGET_NORMAL_SPAN);
      }
    }
  }

  // Ordinary zero-restitution rigid support is a connected material
  // component once a body-static manifold is incident to a dynamic-dynamic
  // contact. Restitution components remain fail-closed until their complete
  // owner also preserves the full ToleranceScale stability gate.
  // Discover the complete topology before the narrower one-body manifold
  // owner. Any unsupported incident row rejects the whole component so no
  // subset can be consumed by a second owner.
  if (contactOnlyTargetOwnership && contacts && numContacts > 0) {
    physx::PxArray<physx::PxU8> visitedBodies(numBodies);
    physx::PxArray<physx::PxU8> visitedContacts(numContacts);
    for (physx::PxU32 body = 0; body < numBodies; ++body)
      visitedBodies[body] = 0;
    for (physx::PxU32 contact = 0; contact < numContacts; ++contact)
      visitedContacts[contact] = 0;

    for (physx::PxU32 seed = 0; seed < numBodies; ++seed) {
      if (visitedBodies[seed])
        continue;
      physx::PxArray<physx::PxU32> bodyQueue;
      physx::PxArray<physx::PxU32> componentContacts;
      bodyQueue.pushBack(seed);
      visitedBodies[seed] = 1;
      bool supported = true;
      bool haveRigidStatic = false;
      bool haveDynamicDynamic = false;
      bool haveRestitutionMaterial = false;

      for (physx::PxU32 queueIndex = 0;
           queueIndex < bodyQueue.size(); ++queueIndex) {
        const physx::PxU32 bodyIndex = bodyQueue[queueIndex];
        if (bodies[bodyIndex].invMass <= 0.0f ||
            bodies[bodyIndex].lockFlags != 0)
          supported = false;
        for (physx::PxU32 c = 0; c < numContacts; ++c) {
          AvbdContactConstraint &contact = contacts[c];
          const physx::PxU32 bodyA = contact.header.bodyIndexA;
          const physx::PxU32 bodyB = contact.header.bodyIndexB;
          if (bodyA != bodyIndex && bodyB != bodyIndex)
            continue;
          if (!visitedContacts[c]) {
            visitedContacts[c] = 1;
            componentContacts.pushBack(c);
          }

          const bool dynamicA = bodyA < numBodies;
          const bool dynamicB = bodyB < numBodies;
          if (!dynamicA && !dynamicB) {
            supported = false;
            continue;
          }
          if (dynamicA && dynamicB)
            haveDynamicDynamic = true;
          else
            haveRigidStatic = true;
          haveRestitutionMaterial =
              haveRestitutionMaterial ||
              contact.restitution > 0.0f;

          if (hasDeformableStaticAnchor(contact) ||
              hasKinematicShellAnchor(contact) ||
              (contact.friction <= 0.0f &&
               contact.staticFriction <= 0.0f) ||
              contact.targetVelocity.magnitudeSquared() > 1.0e-12f ||
              !physx::PxIsFinite(contact.restitution) ||
              contact.restitution < 0.0f ||
              contact.restitution > 1.0f ||
              contact.maxImpulse <= 1.0e20f) {
            supported = false;
          }
          if (dynamicA &&
              (physx::PxAbs(contact.invMassScaleA - 1.0f) > 1.0e-6f ||
               physx::PxAbs(contact.invInertiaScaleA - 1.0f) > 1.0e-6f))
            supported = false;
          if (dynamicB &&
              (physx::PxAbs(contact.invMassScaleB - 1.0f) > 1.0e-6f ||
               physx::PxAbs(contact.invInertiaScaleB - 1.0f) > 1.0e-6f))
            supported = false;

          if (dynamicA && !visitedBodies[bodyA]) {
            visitedBodies[bodyA] = 1;
            bodyQueue.pushBack(bodyA);
          }
          if (dynamicB && !visitedBodies[bodyB]) {
            visitedBodies[bodyB] = 1;
            bodyQueue.pushBack(bodyB);
          }

          if (dynamicA != dynamicB) {
            const bool dynamicIsA = dynamicA;
            const physx::PxVec3 staticPoint =
                dynamicIsA ? contact.contactPointB
                           : contact.contactPointA;
            const physx::PxReal lengthTolerance =
                1.0e-4f *
                physx::PxMax(mConfig.lengthScale, 1.0f);
            if ((staticPoint - contact.staticPrevWorldPoint)
                    .magnitudeSquared() >
                lengthTolerance * lengthTolerance)
              supported = false;
          }
        }
      }

      const bool passiveSupportComponent =
          haveRigidStatic && haveDynamicDynamic &&
          componentContacts.size() >= 2 &&
          !haveRestitutionMaterial;
      if (!supported ||
          componentContacts.size() >
              kMaxPassiveMaterialComponentContacts ||
          !passiveSupportComponent)
        continue;
      for (physx::PxU32 index = 0;
           index < componentContacts.size(); ++index) {
        AvbdContactConstraint &contact =
            contacts[componentContacts[index]];
        contact.header.flags = physx::PxU16(
            contact.header.flags |
            AvbdContactConstraintFlags::
                eVELOCITY_PASSIVE_FRICTION_COMPONENT_OWNER);
      }
    }
  }

  // A strict two-to-four-row rigid-static friction manifold has one coupled
  // material-velocity objective. This includes a shared explicit tangential
  // target or the passive zero-target case. Mark every physical row so
  // position friction and the body-static sweep cannot replay it, then
  // project the block once after inertial velocity reconstruction.
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    if (rigidStaticContactsPerBody[bodyIndex] < 2 ||
        rigidStaticContactsPerBody[bodyIndex] > 4)
      continue;

    bool supported = true;
    bool haveReferenceTarget = false;
    physx::PxVec3 referenceDynamicTarget(0.0f);
    for (physx::PxU32 c = 0; c < numContacts && supported; ++c) {
      AvbdContactConstraint &contact = contacts[c];
      if (contact.header.bodyIndexA != bodyIndex &&
          contact.header.bodyIndexB != bodyIndex)
        continue;
      if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                                 contact.header.bodyIndexB, numBodies) ||
          hasDeformableStaticAnchor(contact) ||
          hasKinematicShellAnchor(contact)) {
        supported = false;
        break;
      }
      const bool dynamicIsA = contact.header.bodyIndexA == bodyIndex;
      const physx::PxReal dynamicLinearScale =
          dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
      const physx::PxReal dynamicAngularScale =
          dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
      const physx::PxReal targetNormal =
          contact.targetVelocity.dot(contact.contactNormal);
      const physx::PxVec3 staticPoint =
          dynamicIsA ? contact.contactPointB : contact.contactPointA;
      const physx::PxReal lengthTolerance =
          1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0f);
      const bool stationaryStatic =
          (staticPoint - contact.staticPrevWorldPoint).magnitudeSquared() <=
          lengthTolerance * lengthTolerance;
      const physx::PxVec3 dynamicTarget =
          contact.targetVelocity * (dynamicIsA ? 1.0f : -1.0f);
      if ((contact.friction <= 0.0f &&
           contact.staticFriction <= 0.0f) ||
          physx::PxAbs(targetNormal) > 1.0e-6f ||
          contact.maxImpulse <= 1.0e20f ||
          contact.restitution != 0.0f ||
          physx::PxAbs(dynamicLinearScale - 1.0f) > 1.0e-6f ||
          physx::PxAbs(dynamicAngularScale - 1.0f) > 1.0e-6f ||
          !stationaryStatic) {
        supported = false;
        break;
      }
      if (!haveReferenceTarget) {
        referenceDynamicTarget = dynamicTarget;
        haveReferenceTarget = true;
      } else if ((dynamicTarget - referenceDynamicTarget).magnitudeSquared() >
                 1.0e-10f) {
        supported = false;
      }
    }
    if (!supported || !haveReferenceTarget)
      continue;
    const bool passiveFriction =
        referenceDynamicTarget.magnitudeSquared() <= 1.0e-12f;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      AvbdContactConstraint &contact = contacts[c];
      if (contact.header.bodyIndexA == bodyIndex ||
          contact.header.bodyIndexB == bodyIndex) {
        if (passiveFriction) {
          contact.header.flags = physx::PxU16(
              contact.header.flags |
              AvbdContactConstraintFlags::
                  eVELOCITY_PASSIVE_FRICTION_MANIFOLD_OWNER);
        } else {
          contact.header.flags = physx::PxU16(
              contact.header.flags |
              AvbdContactConstraintFlags::
                  eVELOCITY_TANGENT_TARGET_OWNER |
              AvbdContactConstraintFlags::
                  eVELOCITY_TANGENT_TARGET_MANIFOLD_OWNER);
        }
      }
    }
  }

  // Strict Phase-3 owner: ordinary zero-target deformable/static tangents use
  // the same position-level row in primal and dual. Joint-mixed islands remain
  // excluded until they have an independent capability fixture. NP contacts
  // cannot create a synthesized soft/direct-shell batch at this boundary.
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &contact = contacts[c];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        !hasDeformableStaticAnchor(contact) ||
        (contact.friction <= 0.0f &&
         contact.staticFriction <= 0.0f))
      continue;
    stats.surfaceDeformablePositionTangentCandidates++;
    if (hasJoints || hasDeformableSoftVbd) {
      stats.surfaceDeformablePositionTangentMixedRejectRows++;
      continue;
    }
    if (contact.restitution != 0.0f) {
      stats.surfaceDeformablePositionTangentRestitutionRejectRows++;
      continue;
    }
    if (contact.maxImpulse <= 1.0e20f) {
      stats.surfaceDeformablePositionTangentFiniteRejectRows++;
      continue;
    }
    const physx::PxReal targetNormal =
        contact.targetVelocity.dot(contact.contactNormal);
    const physx::PxReal targetTangent0 =
        contact.targetVelocity.dot(contact.tangent0);
    const physx::PxReal targetTangent1 =
        contact.targetVelocity.dot(contact.tangent1);
    if (physx::PxAbs(targetNormal) > 1.0e-6f ||
        physx::PxAbs(targetTangent0) > 1.0e-6f ||
        physx::PxAbs(targetTangent1) > 1.0e-6f) {
      stats.surfaceDeformablePositionTangentTargetRejectRows++;
      continue;
    }
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxReal dynamicLinearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal dynamicAngularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (physx::PxAbs(dynamicLinearScale - 1.0f) > 1.0e-6f ||
        physx::PxAbs(dynamicAngularScale - 1.0f) > 1.0e-6f) {
      stats.surfaceDeformablePositionTangentScaleRejectRows++;
      continue;
    }
    contact.header.flags = physx::PxU16(
        contact.header.flags |
        AvbdContactConstraintFlags::eDEFORMABLE_POSITION_TANGENT_OWNER);
  }

  // One island entry: joint/genuine-soft module vs contact-only module. NP
  // contact data cannot synthesize soft particles or route through a second
  // primal.
  if (hasJoints || hasDeformableSoftVbd) {
    solveWithJoints(dt, bodies, numBodies, contacts, numContacts, d6Joints,
                    numD6, gearJoints, numGear, gravity, contactMap, d6Map,
                    gearMap, colorBatches, numColors, iterationOverride,
                    softParticles, numSoftParticles, softBodies, numSoftBodies,
                    softContacts, numSoftContacts, stats);
  } else {
    solve(dt, bodies, numBodies, contacts, numContacts, gravity, contactMap,
          colorBatches, numColors, iterationOverride, stats);
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
  physx::PxReal totalError = 0.0f;
  KahanSum totalErrorKahan;
  const bool useKahan =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eUSE_KAHAN_SUMMATION);
  physx::PxU32 numActive = 0;

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

    // The reference beta is calibrated in unit-length coordinates.  Normalize
    // the raw world-space violation while preserving the established
    // lengthScale=1 behavior and its impact-energy/stability envelope.
    const physx::PxReal lengthScale =
        physx::PxMax(mConfig.lengthScale, 1e-6f);
    const physx::PxReal beta = mConfig.avbdBeta / lengthScale;
    const physx::PxReal penaltyMax = mConfig.avbdPenaltyMax;

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
      const physx::PxReal mu =
          hasVelocityTangentMaterialOwner(contacts[c])
              ? 0.0f
              : contactCoulombMu(contacts[c]);

      physx::PxReal tC0 = 0.0f, tC1 = 0.0f;
      if (contacts[c].friction > 0.0f || contacts[c].staticFriction > 0.0f) {
        const physx::PxReal targetT0 =
            contacts[c].targetVelocity.dot(contacts[c].tangent0);
        const physx::PxReal targetT1 =
            contacts[c].targetVelocity.dot(contacts[c].tangent1);
        if (!hasVelocityTangentTargetOwner(contacts[c]) &&
            (physx::PxAbs(targetT0) > 1e-6f ||
             physx::PxAbs(targetT1) > 1e-6f))
          stats.contactFrictionTargetAlEvaluations++;
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
      if (contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f) {
        const physx::PxReal maxNormalForce =
            physx::PxMax(0.0f, contacts[c].maxImpulse) / dt;
        Fn = physx::PxMax(Fn, -maxNormalForce);
        avbdProjectImpulseCone(maxNormalForce * mu, Ft0, Ft1);
      }
      // Coulomb bound uses Fn / prior ? only (demo3d). Do NOT inject m*g here:
      // per-contact weight floors multi-count box corners and glue HelloWorld
      // stacks under ball impact. Resting grip is the post-pass, impact-gated.
      const physx::PxReal nCap = physx::PxMax(
          -Fn, (oldLambda < 0.0f) ? -oldLambda : 0.0f);
      const physx::PxReal boundedNCap =
          contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f
              ? physx::PxMin(nCap, contacts[c].maxImpulse / dt)
              : nCap;
      newLambda = Fn;
      contacts[c].header.lambda = Fn;
      contacts[c].tangentLambda0 = Ft0;
      contacts[c].tangentLambda1 = Ft1;

      if (newLambda < 0.0f) {
        physx::PxReal growthDist = physx::PxAbs(violation);
        if (deformableStaticAnchor ||
            (numContacts > 4u &&
             isBodyVsStaticContact(bodyAIdx, bodyBIdx, numBodies)))
          growthDist = physx::PxMin(growthDist, 0.15f * lengthScale);
        contacts[c].header.penalty =
            physx::PxMin(pen + beta * growthDist, penaltyMax);
      }
      const physx::PxReal bounds = boundedNCap * mu;
      if (preLen <= bounds) {
        contacts[c].tangentPenalty0 = physx::PxMin(
            contacts[c].tangentPenalty0 + beta * physx::PxAbs(tC0),
            penaltyMax);
        contacts[c].tangentPenalty1 = physx::PxMin(
            contacts[c].tangentPenalty1 + beta * physx::PxAbs(tC1),
            penaltyMax);
      }
      setFrictionStick(contacts[c],
                       avbdFrictionStickFromDual(boundedNCap, mu, preLen,
                                                tC0, tC1,
                                                AVBD_FRICTION_STICK_THRESH *
                                                    lengthScale));
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
    const physx::PxArray<bool> *skipDepenForBodies,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
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
      const physx::PxReal linearResponseScale =
          dynIsA ? contacts[c].invMassScaleA
                 : contacts[c].invMassScaleB;
      if (linearResponseScale <= 0.0f)
        continue;
      // A finite contact impulse cannot also receive an unbounded split-pose
      // correction.  Let the capped AL force determine the pose response so
      // insufficient authored support can pass through, matching PhysX/TGS.
      if (contacts[c].maxImpulse < PX_MAX_REAL) {
        if (stats && sweep == 0) {
          stats->bodyStaticDepenetrationFiniteImpulseSkips++;
          if (contacts[c].maxImpulse < 1.0e20f)
            stats->bodyStaticDepenetrationAuthoredFiniteImpulseSkips++;
        }
        continue;
      }
      if (skipDepenForBodies && bi < skipDepenForBodies->size() &&
          (*skipDepenForBodies)[bi] &&
          hasDeformableStaticAnchor(contacts[c]))
        continue;
      if (stats && sweep == 0 && !hasDeformableStaticAnchor(contacts[c])) {
        stats->bodyStaticDepenetrationEligibleRows++;
        if (contacts[c].contactManagerEstablished)
          stats->bodyStaticNormalSupportDepenetrationEligibleRows++;
        else
          stats->bodyStaticNormalOnsetDepenetrationEligibleRows++;
      }
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
      const bool deformableAnchor =
          hasDeformableStaticAnchor(contacts[c]);
      if (deformableAnchor)
        violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
      const physx::PxReal lengthScale =
          physx::PxMax(mConfig.lengthScale, 1e-6f);
      if (violation >= -1e-5f * lengthScale)
        continue;
      const physx::PxVec3 initialWorldA =
          dynIsA
              ? body.prevPosition +
                    body.prevRotation.rotate(contacts[c].contactPointA)
              : contacts[c].staticPrevWorldPoint;
      const physx::PxVec3 initialWorldB =
          dynIsA
              ? contacts[c].staticPrevWorldPoint
              : body.prevPosition +
                    body.prevRotation.rotate(contacts[c].contactPointB);
      const physx::PxReal initialViolation =
          (initialWorldA - initialWorldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      const bool deepInitialViolation =
          initialViolation < -kBodyStaticNearSurface * lengthScale;
      // The retained normal AL row owns uninterrupted shallow support for
      // both rigid and deformable static anchors. Split-pose recovery is an
      // onset/deep-overlap emergency, never a second steady-support owner.
      if (contacts[c].contactManagerEstablished &&
          !deepInitialViolation)
        continue;

      const physx::PxReal approachSpeed =
          body.linearVelocity.magnitude() + gravity.magnitude() * dt;
      physx::PxReal sweepCap =
          physx::PxMax(approachSpeed * dt * 0.5f, 0.01f * lengthScale);
      if (deformableAnchor) {
        const physx::PxVec3 staticNow = dynIsA ? worldB : worldA;
        const physx::PxVec3 meshStep =
            staticNow - contacts[c].staticPrevWorldPoint;
        // Mesh step + deeper floor: prevent multi-cycle trough sink when the
        // heaving surface rises into resting stacks (was capped too soft).
        sweepCap = physx::PxMax(sweepCap, meshStep.magnitude() * 1.5f);
        sweepCap = physx::PxMax(sweepCap, 0.04f * lengthScale);
        if (violation < -0.05f * lengthScale)
          sweepCap = physx::PxMax(sweepCap, -violation * 0.6f);
      }
      const physx::PxReal corr = physx::PxMin(-violation, sweepCap);
      if (dynIsA)
        body.position += contacts[c].contactNormal * corr;
      else
        body.position -= contacts[c].contactNormal * corr;
      if (stats) {
        stats->bodyStaticDepenetrationCorrections++;
        stats->bodyStaticDepenetrationDistance += corr;
        if (deformableAnchor) {
          if (deformableNormalStageMask &&
              c < deformableNormalStageMask->size())
            (*deformableNormalStageMask)[c] |= 2u;
          stats->surfaceDeformableDepenetrationCorrections++;
          stats->surfaceDeformableDepenetrationDistance += corr;
        }
        if (!deformableAnchor) {
          if (contacts[c].contactManagerEstablished) {
            stats->bodyStaticNormalSupportDepenetrationCorrections++;
            stats->bodyStaticNormalSupportDepenetrationDistance += corr;
            if (deepInitialViolation) {
              stats->bodyStaticNormalSupportDeepDepenetrationCorrections++;
              stats->bodyStaticNormalSupportDeepDepenetrationDistance += corr;
            } else {
              stats->bodyStaticNormalSupportShallowDepenetrationCorrections++;
              stats->bodyStaticNormalSupportShallowDepenetrationDistance +=
                  corr;
            }
          } else {
            stats->bodyStaticNormalOnsetDepenetrationCorrections++;
            stats->bodyStaticNormalOnsetDepenetrationDistance += corr;
            if (deepInitialViolation) {
              stats->bodyStaticNormalOnsetDeepDepenetrationCorrections++;
              stats->bodyStaticNormalOnsetDeepDepenetrationDistance += corr;
            } else {
              stats->bodyStaticNormalOnsetShallowDepenetrationCorrections++;
              stats->bodyStaticNormalOnsetShallowDepenetrationDistance += corr;
            }
          }
        }
        stats->bodyStaticDepenetrationMaxCorrection = physx::PxMax(
            stats->bodyStaticDepenetrationMaxCorrection, corr);
      }
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

//=============================================================================
// Sequential body-static friction fallback (rigid static partners and deformable
// rows excluded from the position-level tangent owner)
//
// TGS-style projected Gauss-Seidel friction, decoupled from the AVBD block
// solve. Rigid plane: all corner contacts per sweep. Unsupported deformable
// rows retain the legacy dominant-contact fallback; position-owned deformable
// tangents are skipped here.
//=============================================================================
void AvbdSolver::applyBodyStaticFrictionSweeps(AvbdSolverBody *bodies,
                                               physx::PxU32 numBodies,
                                               AvbdContactConstraint *contacts,
                                               physx::PxU32 numContacts,
                                               const physx::PxVec3 &gravity,
                                               physx::PxReal dt,
                                               physx::PxU32 sweeps,
                                               const physx::PxArray<physx::PxVec3> *velSeedPos,
                                               const physx::PxArray<physx::PxQuat> *velSeedRot,
                                               const physx::PxArray<bool> *skipForBodies,
                                               AvbdSolverStats *stats) {
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
    if (hasVelocityTangentMaterialOwner(cc))
      continue;
    if (hasDeformablePositionTangentOwner(cc))
      continue;
    const physx::PxU32 bi = dynA ? cc.header.bodyIndexA : cc.header.bodyIndexB;
    if (hasDeformableStaticAnchor(cc) && skipForBodies &&
        bi < skipForBodies->size() && (*skipForBodies)[bi])
      continue;
    if (hasDeformableStaticAnchor(cc)) {
      bodyDeformRawCount[bi]++;
      if (stats)
        stats->surfaceDeformableFrictionRawRows++;
      const physx::PxU32 cur = dominantDeformable[bi];
      if (cur == 0xFFFFFFFFu ||
          physx::PxAbs(cc.header.lambda) >
              physx::PxAbs(contacts[cur].header.lambda))
        dominantDeformable[bi] = c;
    } else {
      frContacts.pushBack(c);
      bodyContactCount[bi]++;
      bodyContactNormalSum[bi] += physx::PxAbs(cc.header.lambda);
      if (stats)
        stats->bodyStaticFrictionFallbackRows++;
      const physx::PxReal targetT0 = cc.targetVelocity.dot(cc.tangent0);
      const physx::PxReal targetT1 = cc.targetVelocity.dot(cc.tangent1);
      if (stats &&
          (physx::PxAbs(targetT0) > 1e-6f ||
           physx::PxAbs(targetT1) > 1e-6f))
        stats->bodyStaticFrictionTargetRows++;
    }
  }
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (dominantDeformable[i] != 0xFFFFFFFFu) {
      frContacts.pushBack(dominantDeformable[i]);
      if (stats)
        stats->surfaceDeformableFrictionDominantRows++;
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
  const physx::PxReal lengthScale =
      physx::PxMax(mConfig.lengthScale, 1e-6f);
  const physx::PxReal restSpeed = 1.5f * lengthScale;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    for (physx::PxU32 fi = 0; fi < frContacts.size(); ++fi) {
      AvbdContactConstraint &cc = contacts[frContacts[fi]];
      const bool dynIsA = cc.header.bodyIndexA < numBodies;
      const physx::PxU32 bi = dynIsA ? cc.header.bodyIndexA : cc.header.bodyIndexB;
      AvbdSolverBody &body = bodies[bi];
      const physx::PxReal linearResponseScale =
          dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
      const physx::PxReal angularResponseScale =
          dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
      if (linearResponseScale <= 0.0f &&
          angularResponseScale <= 0.0f)
        continue;
      touched[bi] = true;

      const physx::PxVec3 cpLocal = dynIsA ? cc.contactPointA : cc.contactPointB;
      const physx::PxVec3 r = body.rotation.rotate(cpLocal);
      const physx::PxReal contactInvMass =
          body.invMass * linearResponseScale;
      const physx::PxMat33 contactInvI =
          body.invInertiaWorld * angularResponseScale;

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
      if (stats && sweep == 0 && hasDeformableStaticAnchor(cc)) {
        if (cc.supportClass == AvbdSupportClass::eDeformableFewContact)
          stats->surfaceDeformableFrictionFewContactRows++;
        else if (cc.supportClass ==
                 AvbdSupportClass::eDeformableMultiCorner)
          stats->surfaceDeformableFrictionMultiCornerRows++;
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
      if (body.invMass > 1e-8f && bodySpeed[bi] < restSpeed &&
          viol <= 0.05f * lengthScale) {
        const physx::PxReal weight =
            (1.0f / body.invMass) * gravity.magnitude() /
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
        kEff[a] =
            contactInvMass + rCrossT[a].dot(contactInvI * rCrossT[a]);
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxVec3 dynamicTargetVelocity =
            cc.targetVelocity * (dynIsA ? 1.0f : -1.0f);
        const physx::PxVec3 vRel =
            (vLin[bi] + vAng[bi].cross(r)) - vMesh -
            dynamicTargetVelocity;
        jUnc[a] = -vRel.dot(t) / kEff[a];
      }
      avbdProjectImpulseCone(jmax, jUnc[0], jUnc[1]);
      const physx::PxReal targetT0 =
          cc.targetVelocity.dot(cc.tangent0);
      const physx::PxReal targetT1 =
          cc.targetVelocity.dot(cc.tangent1);
      const bool hasTangentTarget =
          physx::PxAbs(targetT0) > 1e-6f ||
          physx::PxAbs(targetT1) > 1e-6f;
      const physx::PxReal frictionImpulseMagnitude =
          physx::PxSqrt(jUnc[0] * jUnc[0] + jUnc[1] * jUnc[1]);
      if (stats && hasDeformableStaticAnchor(cc) &&
          frictionImpulseMagnitude > 1e-8f) {
        stats->surfaceDeformableFrictionCorrections++;
        stats->surfaceDeformableFrictionImpulse +=
            frictionImpulseMagnitude;
      }
      if (stats && !hasDeformableStaticAnchor(cc) &&
          frictionImpulseMagnitude > 1e-8f) {
        stats->bodyStaticFrictionFallbackCorrections++;
        stats->bodyStaticFrictionFallbackImpulse +=
            frictionImpulseMagnitude;
      }
      if (stats && hasTangentTarget &&
          frictionImpulseMagnitude > 1e-8f) {
        stats->bodyStaticFrictionTargetCorrections++;
        stats->bodyStaticFrictionTargetImpulse +=
            frictionImpulseMagnitude;
      }
      for (physx::PxU32 a = 0; a < 2; ++a) {
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxReal j = jUnc[a];
        vLin[bi] += tangents[a] * (j * contactInvMass);
        vAng[bi] += contactInvI * (rCrossT[a] * j);
        // Public PxContactPair friction impulses use the impulse applied to
        // contact body A. The sweep updates whichever endpoint is dynamic, so
        // flip the recorded direction when that endpoint is body B.
        const physx::PxReal reportSign = dynIsA ? 1.0f : -1.0f;
        cc.frictionSweepImpulse += tangents[a] * (j * reportSign);
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
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
    AvbdSolverStats *stats) {
  if (!softContacts || numSoftContacts == 0 || !bodies || numBodies == 0 ||
      !softParticles || sweeps == 0 || dt <= 0.0f)
    return;

  if (stats) {
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContact &sc = softContacts[sci];
      if (sc.rigidBodyIdx < numBodies &&
          sc.particleIdx < numSoftParticles &&
          softParticles[sc.particleIdx].invMass <= 0.0f &&
          bodies[sc.rigidBodyIdx].invMass > 0.0f)
        stats->surfaceShellContacts++;
    }
  }

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
      if (stats) {
        stats->surfaceShellDepenetrationCorrections++;
        stats->surfaceShellDepenetrationDistance += corr;
      }
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
    const physx::PxArray<physx::PxQuat> *velSeedRot,
    AvbdSolverStats *stats) {
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
    if (dominantContact[i] != 0xFFFFFFFFu) {
      frContacts.pushBack(dominantContact[i]);
      if (stats)
        stats->surfaceShellFrictionRows++;
    }
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
      const physx::PxReal impulse =
          physx::PxSqrt(jUnc[0] * jUnc[0] + jUnc[1] * jUnc[1]);
      if (stats && impulse > 1e-8f) {
        stats->surfaceShellFrictionCorrections++;
        stats->surfaceShellFrictionImpulse += impulse;
      }
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
    physx::PxReal dt, AvbdSolverStats *stats) {
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

    if (stats)
      stats->surfaceShellFinalizeBodies++;
    const AvbdSoftContact &sc = softContacts[dominant];
    const physx::PxVec3 nd = sc.normal;
    const physx::PxReal vn = bodies[i].linearVelocity.dot(nd);
    const physx::PxReal vMeshN =
        invDt > 0.0f
            ? ((sc.surfacePoint - sc.surfacePointPrev) * invDt).dot(nd)
            : 0.0f;
    const physx::PxReal vRelN = vn - vMeshN;
    if (vRelN > 0.0f) {
      bodies[i].linearVelocity -= nd * vRelN;
      if (stats) {
        stats->surfaceShellFinalizeCorrections++;
        stats->surfaceShellFinalizeDelta += vRelN;
      }
    }
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

void AvbdSolver::accumulateBodyContactRows(
    AvbdSolverBody &body, physx::PxU32 bodyIndex, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt, physx::PxReal massInvDt2, AvbdBlock6x6 &A,
    physx::PxVec3 &gLinear, physx::PxVec3 &gAngular,
    physx::PxU32 &numTouching) {

  bool bodyUsesStaticParticleSoftNormals = false;
  if (softContacts && numSoftContacts > 0) {
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      if (softContacts[sci].rigidBodyIdx == bodyIndex) {
        bodyUsesStaticParticleSoftNormals = true;
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

    // A real rigid/soft contact owns the normal when both representations are
    // present. This is not reachable from the rigid NP-only solveIsland entry.
    if (bodyUsesStaticParticleSoftNormals &&
        hasDeformableStaticAnchor(contacts[c])) {
      continue;
    }

    const bool isBodyA = (bodyAIdx == bodyIndex);
    const physx::PxReal linearResponseScale =
        isBodyA ? contacts[c].invMassScaleA
                : contacts[c].invMassScaleB;
    const physx::PxReal angularResponseScale =
        isBodyA ? contacts[c].invInertiaScaleA
                : contacts[c].invInertiaScaleB;
    if (linearResponseScale <= 0.0f &&
        angularResponseScale <= 0.0f) {
      // Contact-local infinite mass/inertia: this row must not move this body,
      // but the peer still consumes the same row with its own response scales.
      continue;
    }
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

    // Normal force (unilateral) + optional Coulomb-cone tangents in 6x6.
    const physx::PxReal rawForce =
        physx::PxMin(0.0f, pen * violation + lambda);
    physx::PxReal f = rawForce;
    bool forceSaturated = false;
    if (contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f) {
      const physx::PxReal maxNormalForce =
          physx::PxMax(contacts[c].maxImpulse, physx::PxReal(0.0f)) / dt;
      f = physx::PxMax(f, -maxNormalForce);
      forceSaturated = rawForce < -maxNormalForce;
    }
    // The derivative of a clamped force is zero while saturated.  Keeping the
    // contact penalty in the local Hessian here would enforce the unilateral
    // row even though its authored impulse budget has already been exhausted.
    if (!forceSaturated) {
      A.addResponseScaledConstraintContribution(
          gradPos, gradRot, pen, linearResponseScale, angularResponseScale);
    }
    numTouching++;

    if (f < 0.0f) {
      gLinear += gradPos * (f * linearResponseScale);
      gAngular += gradRot * (f * angularResponseScale);
    }

    // Ordinary rigid-static tangents keep their dedicated material owner.
    // The strict deformable/static probe instead consumes its position dual
    // through this same body-level AVBD primal block.
    if ((contacts[c].friction > 0.0f || contacts[c].staticFriction > 0.0f) &&
        (useBodyVsStaticFrictionIn6x6(bodyAIdx, bodyBIdx, numBodies) ||
         hasDeformablePositionTangentOwner(contacts[c]))) {
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
          hasDeformablePositionTangentOwner(contacts[c])
              ? computeBodyVsStaticRelDisp(
                    worldPosA, prevWorldPosA, worldPosB, prevWorldPosB,
                    contacts[c], numBodies)
              : (worldPosA - prevWorldPosA) -
                    (worldPosB - prevWorldPosB);

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
        A.addResponseScaledConstraintContribution(
            tGradPos, tGradRot, tPen0, linearResponseScale,
            angularResponseScale);
        gLinear += tGradPos * (Ft0 * linearResponseScale);
        gAngular += tGradRot * (Ft0 * angularResponseScale);
      }
      {
        const physx::PxVec3 &t = contacts[c].tangent1;
        const physx::PxVec3 rCrossT = r.cross(t);
        const physx::PxVec3 tGradPos = t * sign;
        const physx::PxVec3 tGradRot = rCrossT * sign;
        A.addResponseScaledConstraintContribution(
            tGradPos, tGradRot, tPen1, linearResponseScale,
            angularResponseScale);
        gLinear += tGradPos * (Ft1 * linearResponseScale);
        gAngular += tGradRot * (Ft1 * angularResponseScale);
      }
    }
  }

  if (softContacts && numSoftContacts > 0 && softParticles &&
      numSoftParticles > 0) {
    const physx::PxReal shellBoostFloor =
        AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC * massInvDt2;
    AvbdVec6 shellRhs;
    shellRhs.linear = physx::PxVec3(0.0f);
    shellRhs.angular = physx::PxVec3(0.0f);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContact &sc = softContacts[sci];
      if (sc.rigidBodyIdx != bodyIndex)
        continue;
      if (sc.particleIdx >= numSoftParticles)
        continue;
      if (softParticles[sc.particleIdx].invMass > 0.0f)
        continue;
      avbdAddKinematicShellContactContribution_rigid(
          softContacts[sci], bodyIndex, body, shellBoostFloor, A, shellRhs);
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
  // Step 3: Shared rigid-contact primal accumulation (body-static contract)
  // =========================================================================

  physx::PxU32 numTouching = 0;
  accumulateBodyContactRows(
      body, bodyIndex, bodies, numBodies, contacts, numContacts, contactMap,
      nullptr, 0, nullptr, 0, dt, massInvDt2, A, gLinear, gAngular,
      numTouching);

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
  // Parallelization uses a read-only pose snapshot for every local solve and
  // writes each result to a distinct output body.  Reading the live body array
  // here would be asynchronous Gauss-Seidel with unsynchronized neighbor
  // reads, not Jacobi, and makes both results and scale invariance depend on
  // task scheduling.

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
    physx::PxArray<AvbdSolverBody> bodySnapshot(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      bodySnapshot[i] = bodies[i];

    auto solveBodyJacobi = [&](physx::PxU32 i) {
      if (bodySnapshot[i].invMass <= 0.0f)
        return;
      AvbdSolverBody updatedBody = bodySnapshot[i];
      if (mConfig.enableLocal6x6Solve) {
        solveLocalSystem(
            updatedBody, bodySnapshot.begin(), numBodies, contacts, numContacts,
            dt, invDt2, contactMap);
      } else {
        solveLocalSystemWithJoints(
            updatedBody, bodySnapshot.begin(), numBodies, contacts, numContacts,
            nullptr, 0, nullptr, 0, dt, invDt2, contactMap, nullptr, nullptr);
      }
      bodies[i] = updatedBody;
    };
    avbdParallelFor(0u, numBodies, solveBodyJacobi);
  } else {
    for (physx::PxU32 idx = 0; idx < numBodies; ++idx)
      solveBody(idx);
  }
}

} // namespace Dy
} // namespace physx
