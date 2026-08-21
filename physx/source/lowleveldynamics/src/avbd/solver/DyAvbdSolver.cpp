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

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/core/DyAvbdBoundedProjection.h"
#include "avbd/solver/post_al/DyAvbdPostAl.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace physx {
namespace Dy {

namespace {

struct KahanSum {
  physx::PxReal sum{0.0f};
  physx::PxReal c{0.0f};

  void add(physx::PxReal value) {
    const physx::PxReal y = value - c;
    const physx::PxReal t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

static physx::PxReal computeRotationDeltaMagnitude(
    const physx::PxQuat &current, const physx::PxQuat &previous) {
  physx::PxQuat deltaQ = current * previous.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  return 2.0f * physx::PxSqrt(deltaQ.x * deltaQ.x +
                              deltaQ.y * deltaQ.y +
                              deltaQ.z * deltaQ.z);
}

static PX_FORCE_INLINE bool getAvbdBodyContactRange(
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    const physx::PxU32 *&indices, physx::PxU32 &count);

static bool bodyTouchesDeformableAnchorImpl(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex, const AvbdBodyConstraintMap *contactMap) {
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
    const physx::PxU32 bA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bB = contacts[c].header.bodyIndexB;
    if (!hasDeformableStaticAnchor(contacts[c]))
      continue;
    if (bA == bodyIndex || bB == bodyIndex)
      return true;
  }
  return false;
}

// The contact map is built once per island.  Keep the fallback for callers
// that do not provide it (notably a few legacy/deformable paths), but make the
// hot per-body post-AL loops consume only incident rows when it is available.
static PX_FORCE_INLINE bool getAvbdBodyContactRange(
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    const physx::PxU32 *&indices, physx::PxU32 &count) {
  if (!contactMap || !contactMap->constraintOffsets ||
      !contactMap->constraintCounts || bodyIndex >= contactMap->numBodies) {
    indices = nullptr;
    count = 0;
    return false;
  }
  contactMap->getBodyConstraints(bodyIndex, indices, count);
  return true;
}

// Enforce the velocity counterpart of body-vs-static locked D6 linear rows.
// Position-level AL convergence can leave a small first-step pose residual;
// reconstructing velocity directly from that residual creates a velocity that
// violates an otherwise hard joint.  This is a Jacobian/effective-mass
// projection, not a magnitude dead-zone.  Dynamic-dynamic, limited/free and
// driven rows remain outside this first body-vs-static correctness slice.
static void projectBodyStaticLockedD6LinearVelocitiesImpl(
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
    const physx::PxU32 linearSourceRows[3] = {
        eJOINT_SOURCE_LINEAR_MOTION_X,
        eJOINT_SOURCE_LINEAR_MOTION_Y,
        eJOINT_SOURCE_LINEAR_MOTION_Z};
    const physx::PxU32 angularSourceRows[3] = {
        eJOINT_SOURCE_ANGULAR_MOTION_X,
        eJOINT_SOURCE_ANGULAR_MOTION_Y,
        eJOINT_SOURCE_ANGULAR_MOTION_Z};
    const auto isPositionGeometrySource =
        [&](physx::PxU32 sourceRow) -> bool {
      const AvbdCompiledJointObjective *objective =
          findAvbdJointObjectiveForSourceRow(
              joint.objectiveProgram, sourceRow);
      if (!objective ||
          objective->owner !=
              AvbdVelocityObjectiveOwner::PositionAL)
        return false;
      return objective->kind ==
                 AvbdJointObjectiveKind::OrdinaryD6Position ||
             objective->kind ==
                 AvbdJointObjectiveKind::CoupledFixedD6;
    };

    for (physx::PxU32 axis = 0; axis < 3; ++axis) {
      if (joint.getLinearMotion(axis) != 0 ||
          !isPositionGeometrySource(linearSourceRows[axis]))
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
    bool completeFixedPositionObjective =
        allLinearLocked && joint.angularMotion == 0;
    for (physx::PxU32 axis = 0;
         axis < 3 && completeFixedPositionObjective; ++axis) {
      completeFixedPositionObjective =
          isPositionGeometrySource(linearSourceRows[axis]) &&
          isPositionGeometrySource(angularSourceRows[axis]);
    }
    if (completeFixedPositionObjective) {
      body.linearVelocity = physx::PxVec3(0.0f);
      body.angularVelocity = physx::PxVec3(0.0f);
    }

  }
}

// Suppress pose-solve bounce only on fast normal approach (sphere shot).
static const physx::PxReal kBodyStaticFastImpactSpeed =
    AvbdConstants::AVBD_BODY_STATIC_FAST_IMPACT_SPEED;

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
static bool mayHavePostAlContactWork(
    const AvbdPostAlContactWorkPlan *workPlan, physx::PxU8 work) {
  return !workPlan || workPlan->mayHave(work);
}

// Classify one final, validated contact program for the three post-AL
// consumers below.  The point predicate deliberately mirrors its consumer's
// first three continues, including the NaN behavior of !(magnitudeSq <= eps).
static physx::PxU8 collectValidatedPostAlContactWork(
    const AvbdContactConstraint &contact, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies) {
  physx::PxU8 work = 0;
  bool velocityFrictionManifoldOwner = false;
  const AvbdCompiledContactObjectiveProgram &program =
      contact.objectiveProgram;
  for (physx::PxU32 entryIndex = 0; entryIndex < program.entryCount;
       ++entryIndex) {
    const AvbdCompiledVelocityObjective &entry = program.entries[entryIndex];
    if (entry.owner == AvbdVelocityObjectiveOwner::ComponentFinalize &&
        entry.kind == AvbdVelocityObjectiveKind::PassiveFriction) {
      work = physx::PxU8(
          work | AvbdPostAlContactWorkPlan::ePASSIVE_COMPONENT);
      velocityFrictionManifoldOwner = true;
    }
    if (entry.owner != AvbdVelocityObjectiveOwner::ManifoldFinalize)
      continue;
    if (entry.kind == AvbdVelocityObjectiveKind::TangentTarget)
      velocityFrictionManifoldOwner = true;
    if (entry.kind == AvbdVelocityObjectiveKind::PassiveFriction &&
        entry.span == AvbdVelocityObjectiveSpan::NormalAndTangentCone &&
        entry.reconstruction ==
            AvbdVelocityObjectiveReconstruction::SolveStartInertial)
      velocityFrictionManifoldOwner = true;
    if (entry.span == AvbdVelocityObjectiveSpan::NormalAndTangentCone &&
        entry.reconstruction ==
            AvbdVelocityObjectiveReconstruction::SolveStartInertial)
      work = physx::PxU8(
          work | AvbdPostAlContactWorkPlan::eCOMPLETE_MANIFOLD);
  }

  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  const bool dynamicA =
      bodyA < numBodies && bodies[bodyA].invMass > 0.0f;
  const bool dynamicB =
      bodyB < numBodies && bodies[bodyB].invMass > 0.0f;
  if (!velocityFrictionManifoldOwner &&
      !(contact.targetVelocity.magnitudeSquared() <= 1.0e-12f) &&
      (dynamicA || dynamicB))
    work = physx::PxU8(work | AvbdPostAlContactWorkPlan::ePOINT_TARGET);
  return work;
}

static void applyAvbdPassiveFrictionComponents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, const AvbdPostAlContactWorkPlan *workPlan) {
  if (!bodies || !contacts || numBodies == 0 ||
      numContacts == 0 || dt <= 0.0f ||
      !mayHavePostAlContactWork(
          workPlan, AvbdPostAlContactWorkPlan::ePASSIVE_COMPONENT))
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
    const AvbdCompiledVelocityObjective *seedObjective =
        findAvbdVelocityObjective(
            contacts[seed].objectiveProgram,
            AvbdVelocityObjectiveOwner::ComponentFinalize,
            AvbdVelocityObjectiveKind::PassiveFriction);
    if (!seedObjective)
      continue;
    const physx::PxU64 objectiveKey = seedObjective->objectiveKey;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const AvbdCompiledVelocityObjective *objective =
          findAvbdVelocityObjective(
              contacts[c].objectiveProgram,
              AvbdVelocityObjectiveOwner::ComponentFinalize,
              AvbdVelocityObjectiveKind::PassiveFriction);
      if (visitedContacts[c] ||
          !hasVelocityPassiveFrictionComponentOwner(contacts[c]) ||
          !objective || objective->objectiveKey != objectiveKey)
        continue;
      visitedContacts[c] = 1;
      componentContacts.pushBack(c);
      enqueueBody(contacts[c].header.bodyIndexA);
      enqueueBody(contacts[c].header.bodyIndexB);
    }
    bool supported =
        componentContacts.size() >= 2 &&
        componentContacts.size() ==
            seedObjective->objectiveRowCount;
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
    physx::PxReal dt, const AvbdPostAlContactWorkPlan *workPlan) {
  if (!bodies || !contacts || dt <= 0.0f)
    return;

  applyAvbdPassiveFrictionComponents(
      bodies, numBodies, contacts, numContacts,
      dt, workPlan);

  if (!mayHavePostAlContactWork(
          workPlan, AvbdPostAlContactWorkPlan::eCOMPLETE_MANIFOLD))
    return;

  physx::PxArray<physx::PxU8> visitedManifoldRows(numContacts);
  for (physx::PxU32 c = 0; c < numContacts; ++c)
    visitedManifoldRows[c] = 0;
  for (physx::PxU32 seed = 0; seed < numContacts; ++seed) {
    const AvbdCompiledVelocityObjective *seedObjective =
        findAvbdCompleteManifoldObjective(
            contacts[seed].objectiveProgram);
    if (visitedManifoldRows[seed] || !seedObjective)
      continue;
    const physx::PxU32 bodyIndex =
        contacts[seed].header.bodyIndexA < numBodies
            ? contacts[seed].header.bodyIndexA
            : contacts[seed].header.bodyIndexB;
    if (bodyIndex >= numBodies)
      continue;
    AvbdSolverBody &body = bodies[bodyIndex];
    if (body.invMass <= 0.0f)
      continue;

    physx::PxU32 contactIndices[4] = {};
    physx::PxU32 contactCount = 0;
    const physx::PxU64 objectiveKey = seedObjective->objectiveKey;
    bool supportedGroup = true;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const AvbdContactConstraint &contact = contacts[c];
      const AvbdCompiledVelocityObjective *objective =
          findAvbdCompleteManifoldObjective(
              contact.objectiveProgram);
      if (!objective || objective->objectiveKey != objectiveKey)
        continue;
      visitedManifoldRows[c] = 1;
      if (contact.header.bodyIndexA != bodyIndex &&
          contact.header.bodyIndexB != bodyIndex)
        supportedGroup = false;
      if (contactCount < 4)
        contactIndices[contactCount] = c;
      ++contactCount;
    }
    if (!supportedGroup || contactCount < 2 || contactCount > 4 ||
        contactCount != seedObjective->objectiveRowCount)
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
static void applyAvbdContactTargetVelocityImpl(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, const AvbdPostAlContactWorkPlan *workPlan) {
  if (!bodies || !contacts || dt <= 0.0f)
    return;

  applyAvbdContactMaterialFrictionManifolds(
      bodies, numBodies, contacts, numContacts,
      dt, workPlan);

  if (!mayHavePostAlContactWork(
          workPlan, AvbdPostAlContactWorkPlan::ePOINT_TARGET))
    return;

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
  }
}

static bool isRigidDeepBodyStaticRecoverySplitSupportedImpl(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  if (!bodies || bodyIndex >= numBodies || !contacts)
    return false;

  bool foundContact = false;
  physx::PxReal worstInitialViolation = PX_MAX_REAL;
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 contactIndex = hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
             -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE *
                 physx::PxMax(lengthScale, physx::PxReal(1e-6f));
}

static bool isRigidFiniteBodyStaticMaterialSplitSupportedImpl(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  if (!bodies || bodyIndex >= numBodies || !contacts)
    return false;

  bool foundContact = false;
  physx::PxU32 contactCount = 0;
  physx::PxReal manifoldLinearScale = 0.0f;
  physx::PxReal manifoldAngularScale = 0.0f;
  const physx::PxReal deepLimit =
      -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE *
      physx::PxMax(lengthScale, physx::PxReal(1.0e-6f));
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 contactIndex = hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
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

  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
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

  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
    double relativeTolerance = 1.0e-6) {
  using namespace AvbdBoundedProjectionDetail;
  AvbdBoundedProjectionResult result;
  const physx::PxU32 rowCount = orderedRows.size();
  result.candidateImpulses.resize(rowCount, 0.0);
  result.commitImpulses.resize(rowCount, 0.0);
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
         violation < AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE;
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
        worstViolation >= AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE)
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
    switch (budgetClass) {
    case eBUDGET_DIAG_NO_CORRECTION:
      ++nodes[root].budgetDiagNoCorrectionRows;
      break;
    case eBUDGET_DIAG_ZERO_BUDGET_REQUIRED:
      ++nodes[root].budgetDiagZeroBudgetRequiredRows;
      break;
    case eBUDGET_DIAG_WITHIN_BUDGET:
      ++nodes[root].budgetDiagWithinBudgetRows;
      break;
    case eBUDGET_DIAG_OVER_BUDGET:
      ++nodes[root].budgetDiagOverBudgetRows;
      break;
    default:
      ++nodes[root].budgetDiagUnsupportedRows;
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
    PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowComponents);
    const bool shadowUnsupported =
        component.restitution || component.targetVelocity ||
        component.mixedScale || component.rigidStatic ||
        component.nonOwnerDeformable || hasJointConstraints ||
        component.lockedDof || component.nonDynamicBody ||
        component.fastImpact || component.snapshotUnsupported;
    if (shadowUnsupported) {
      PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowUnsupported);
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
      continue;
    }
    const bool useMatrixFreeBackend = rowCount > 128;
    AvbdBoundedProjectionResult shadow;
    if (useMatrixFreeBackend) {
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
        continue;
      }
      shadow = solveAvbdBoundedProjection(
          response, outward, upperBounds, 6 * component.bodyCount);
    }
    switch (shadow.status) {
    case eAVBD_BOUNDED_SOLVED:
      PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowSolved);
      if (enableProductionProbe &&
          shadow.commitImpulses.size() == rowCount &&
          probeOwnedBodies.size() == numBodies) {
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
          PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeProbeCommittedComponents);
        }
      }
      break;
    case eAVBD_BOUNDED_NO_CORRECTION:
      break;
    case eAVBD_BOUNDED_BUDGET_EXHAUSTED:
      break;
    case eAVBD_BOUNDED_INFEASIBLE:
      break;
    case eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED:
      break;
    case eAVBD_BOUNDED_ITERATION_LIMIT:
      PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowIterationLimit);
      break;
    default:
      break;
    }
  }
}

static void applyAvbdMaterialNormalVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale,
    bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
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
  physx::PxArray<bool> finalizeProbeOwnedBodies;
  physx::PxArray<bool> *finalizeProbeOwnedBodiesPtr = nullptr;
  if (stats && deformableNormalStageMask) {
    finalizeProbeOwnedBodies.resize(numBodies);
    for (physx::PxU32 body = 0; body < numBodies; ++body)
      finalizeProbeOwnedBodies[body] = false;
    finalizeProbeOwnedBodiesPtr = &finalizeProbeOwnedBodies;
    finalizeTopologyNodes.resize(numBodies);
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      finalizeTopologyNodes[body].strictOwner = 0;
      finalizeTopologyNodes[body].bodyStrictOwner = 0;
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
        finalizeProbeOwnedBodies, stats);
  }
  // ---- Body-static (incl. deformable anchors) ----
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;
    if (finalizeProbeOwnedBodiesPtr && (*finalizeProbeOwnedBodiesPtr)[i])
      continue;
    bool passiveMaterialComponentOwned = false;
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    const bool hasMapRange = getAvbdBodyContactRange(
        contactMap, i, mapIndices, mapCount);
    const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
      if (hasVelocityPassiveFrictionComponentOwner(contacts[c])) {
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

    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
            -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE *
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
          bodies, numBodies, contacts, numContacts, contactMap, i,
          linearVelAtSolveStart, angularVelAtSolveStart, dt, bounceThreshold,
          spatialLinearDelta);
      if (finiteOwned) {
        continue;
      }
    }

    const bool isDeform = hasDeformableStaticAnchor(contacts[dominant]);
    const AvbdContactConstraint &cc = contacts[dominant];
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
    if (isDeform) {
      if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats)
        PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeBodies++);
      const physx::PxReal nearLim =
          AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE;
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
          if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats &&
              deformableNormalStageMask &&
              dominant < deformableNormalStageMask->size()) {
            const physx::PxReal contactRelativeVnAfter =
                (bodies[i].linearVelocity +
                 bodies[i].angularVelocity.cross(dynamicContactArm))
                        .dot(nd) -
                vMeshN;
            const physx::PxReal diagnosticVelocityTolerance =
                1.0e-5f *
                physx::PxMax(lengthScale, physx::PxReal(1.0e-6f)) *
                invDt;
            if (contactRelativeVnAfter < -diagnosticVelocityTolerance) {
              PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeContactReversalCorrections++);
            }
          }
          if (deformableNormalStageMask &&
              dominant < deformableNormalStageMask->size())
            (*deformableNormalStageMask)[dominant] |= 4u;
          if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats) {
            PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeCorrections++);
            PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeDelta += linearDeltaMagnitude);
          }
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
          bodies, numBodies, contacts, numContacts, contactMap, i,
          linearVelAtSolveStart, angularVelAtSolveStart, dt, bounceThreshold,
          restitutionLinearDelta);
    } else if (e > 0.0f && approachEff > bounceThreshold) {
      const physx::PxReal desiredRelativeVn =
          physx::PxMin(e * approachEff, maxImpulseRelativeVn);
      const physx::PxReal deltaV =
          staticNormalVelocity + desiredRelativeVn - vn;
      bodies[i].linearVelocity += nd * deltaV;
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
      const bool shouldClamp =
          hasSolveStartVelocity || worstViolation < -1e-5f ||
          splitDeepInitialDepenetration;
      if (shouldClamp && relativeVn > allowedRelativeVn) {
        const physx::PxReal deltaV = relativeVn - allowedRelativeVn;
        bodies[i].linearVelocity -= nd * deltaV;
      }
    }
  }

  // Dyn-dyn restitution: relative normal impulse with invMass split.
  // Apply only for free rigid pairs (no deformable); e and bounce threshold
  // from material/scene. Skip if either body already handled as body-static
  // dominant this frame would double-count; dyn-dyn contacts are exclusive.
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    const AvbdCompiledVelocityObjective *materialNormalObjective =
        findAvbdContactSourceObjective(
            cc.objectiveProgram,
            eCONTACT_SOURCE_MATERIAL_NORMAL);
    // A compiled material-normal source is consumed only by its unique
    // owner. Legacy rows retain the historical path until their compile
    // classification is made explicit.
    if (materialNormalObjective &&
        materialNormalObjective->owner !=
            AvbdVelocityObjectiveOwner::PointFinalize)
      continue;
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
            ? vrel0 + physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)) * invSum
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
static void clampBodyStaticInelasticNormalVelocitiesImpl(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale,
    bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  applyAvbdMaterialNormalVelocity(bodies, numBodies, contacts, numContacts,
                                  contactMap,
                                  linearVelAtSolveStart,
                                  angularVelAtSolveStart,
                                  finiteMaterialPoseSplit, dt,
                                  bounceApproachThreshold, lengthScale,
                                  hasJointConstraints,
                                  enableBoundedComponentProductionProbe,
                                  deformableNormalStageMask, stats);
}

static void recordBodyStaticNormalAlOwnershipImpl(
    const AvbdSolverBody *bodies, const AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numBodies,
    physx::PxReal /*avbdAlpha*/,
    const physx::PxArray<bool> * /*touchesKinematicShell*/,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats &stats) {
  (void)stats;
  if (!bodies || !contacts || !deformableNormalStageMask)
    return;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &contact = contacts[c];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        !hasDeformableStaticAnchor(contact))
      continue;
    PX_AVBD_PROFILE_STAT(stats.surfaceDeformableAlRows++);
    if (c < deformableNormalStageMask->size())
      (*deformableNormalStageMask)[c] |= 1u;
    if (hasDeformablePositionTangentOwner(contact))
      PX_AVBD_PROFILE_STAT(stats.surfaceDeformablePositionTangentRows += 2);
  }
}

static void computeMaxPoseDeltas(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<physx::PxVec3> &prevPos,
    const physx::PxArray<physx::PxQuat> &prevRot,
    physx::PxReal &maxPositionDelta, physx::PxReal &maxRotationDelta) {
  maxPositionDelta = 0.0f;
  maxRotationDelta = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;

    maxPositionDelta = physx::PxMax(
        maxPositionDelta, (bodies[i].position - prevPos[i]).magnitude());
    maxRotationDelta = physx::PxMax(
        maxRotationDelta,
        computeRotationDeltaMagnitude(bodies[i].rotation, prevRot[i]));
  }
}
} // namespace

bool bodyTouchesDeformableAnchor(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex, const AvbdBodyConstraintMap *contactMap) {
  return bodyTouchesDeformableAnchorImpl(contacts, numContacts, bodyIndex,
                                         contactMap);
}

void projectBodyStaticLockedD6LinearVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint *joints, physx::PxU32 numJoints) {
  projectBodyStaticLockedD6LinearVelocitiesImpl(bodies, numBodies, joints,
                                                numJoints);
}

void applyAvbdContactTargetVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, const AvbdPostAlContactWorkPlan *workPlan) {
  applyAvbdContactTargetVelocityImpl(bodies, numBodies, contacts, numContacts,
                                     dt, workPlan);
}

bool isRigidDeepBodyStaticRecoverySplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  return isRigidDeepBodyStaticRecoverySplitSupportedImpl(
      bodies, numBodies, contacts, numContacts, contactMap, bodyIndex,
      lengthScale);
}

bool isRigidFiniteBodyStaticMaterialSplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  return isRigidFiniteBodyStaticMaterialSplitSupportedImpl(
      bodies, numBodies, contacts, numContacts, contactMap, bodyIndex,
      lengthScale);
}

void clampBodyStaticInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale, bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  clampBodyStaticInelasticNormalVelocitiesImpl(
      bodies, numBodies, contacts, numContacts, contactMap,
      linearVelAtSolveStart, angularVelAtSolveStart,
      finiteMaterialPoseSplit, dt, bounceApproachThreshold, lengthScale,
      hasJointConstraints, enableBoundedComponentProductionProbe,
      deformableNormalStageMask, stats);
}

void recordBodyStaticNormalAlOwnership(
    const AvbdSolverBody *bodies, const AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numBodies,
    physx::PxReal avbdAlpha,
    const physx::PxArray<bool> *touchesKinematicShell,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats &stats) {
  recordBodyStaticNormalAlOwnershipImpl(
      bodies, contacts, numContacts, numBodies, avbdAlpha,
      touchesKinematicShell, deformableNormalStageMask, stats);
}

bool AvbdSolver::beginRigidSolveIteration(
    AvbdRigidSolveIterationState &state) {
  if (!state.bodies || !state.stats || state.iter >= state.iters ||
      state.iterationActive)
    return false;
  PX_ASSERT(!state.parallelDualComplete);

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  state.activeIteration = state.iter++;
  state.iterationActive = true;

  // Save pre-iteration state for Chebyshev relaxation and convergence tests.
  if (state.useChebyshev) {
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      state.chebyPrevPrevPos[i] = state.chebyPrevPos[i];
      state.chebyPrevPrevRot[i] = state.chebyPrevRot[i];
      state.chebyPrevPos[i] = bodies[i].position;
      state.chebyPrevRot[i] = bodies[i].rotation;
    }
  }
  if (state.enableEarlyStop) {
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      state.earlyStopPrevPos[i] = bodies[i].position;
      state.earlyStopPrevRot[i] = bodies[i].rotation;
    }
  }
  return true;
}

bool AvbdSolver::completeRigidSolveIteration(
    AvbdRigidSolveIterationState &state) {
  if (!state.bodies || !state.stats || !state.iterationActive)
    return false;

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  const physx::PxReal dt = state.dt;
  const AvbdBodyConstraintMap *contactMap = state.contactMap;
  AvbdSolverStats &stats = *state.stats;
  const physx::PxU32 iter = state.activeIteration;

  PX_AVBD_PROFILE_STAT(stats.totalIterations++);
  if (!state.parallelDualComplete) {
    PX_PROFILE_ZONE("AVBD.updateLambda", 0);
    updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                dt, stats);
  }
  state.parallelDualComplete = false;
  PX_PROFILE_ZONE("AVBD.postDualBody", 0);
  // Chebyshev semi-iterative position/rotation relaxation.
  if (state.useChebyshev && iter >= 2) {
    const physx::PxReal rhoSq = mConfig.chebyshevRho * mConfig.chebyshevRho;
    if (iter == 2)
      state.chebyOmega = 2.0f / (2.0f - rhoSq);
    else
      state.chebyOmega =
          1.0f / (1.0f - rhoSq * state.chebyOmega / 4.0f);
    state.chebyOmega = physx::PxClamp(state.chebyOmega, 1.0f, 2.0f);

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass <= 0.0f)
        continue;
      const physx::PxVec3 gsPosition = bodies[i].position;
      const physx::PxQuat gsRotation = bodies[i].rotation;
      const physx::PxVec3 relaxedPosition =
          state.chebyPrevPrevPos[i] +
          (bodies[i].position - state.chebyPrevPrevPos[i]) *
              state.chebyOmega;

      physx::PxQuat qPrev = state.chebyPrevPrevRot[i];
      physx::PxQuat qCur = bodies[i].rotation;
      if (qPrev.dot(qCur) < 0.0f)
        qCur = -qCur;
      physx::PxQuat qBlend(
          qPrev.x + state.chebyOmega * (qCur.x - qPrev.x),
          qPrev.y + state.chebyOmega * (qCur.y - qPrev.y),
          qPrev.z + state.chebyOmega * (qCur.z - qPrev.z),
          qPrev.w + state.chebyOmega * (qCur.w - qPrev.w));
      const physx::PxQuat relaxedRotation = qBlend.getNormalized();

      // A unilateral body-static active set has zero energy on its satisfied
      // side.  Reject only an outward extrapolation after a deep, quasi-static
      // overlap has already been cleared by the ordinary block step.
      bool rejectBodyStaticOvershoot = false;
      if (state.hasBodyStaticContact) {
        physx::PxReal minGsViolation = PX_MAX_REAL;
        physx::PxReal minRelaxedViolation = PX_MAX_REAL;
        bool foundBodyStatic = false;
        bool deepQuasistaticInitialOverlap = false;
        const physx::PxU32 *mapIndices = nullptr;
        physx::PxU32 mapCount = 0;
        if (contactMap && contactMap->numBodies > 0)
          contactMap->getBodyConstraints(i, mapIndices, mapCount);
        const physx::PxU32 loopCount = mapIndices ? mapCount : numContacts;
        for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
          const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
          const physx::PxU32 bA = contacts[c].header.bodyIndexA;
          const physx::PxU32 bB = contacts[c].header.bodyIndexB;
          if (!isBodyVsStaticContact(bA, bB, numBodies) ||
              (bA != i && bB != i))
            continue;

          const bool dynIsA = (bA == i);
          const physx::PxVec3 gsWorldA =
              dynIsA ? gsPosition + gsRotation.rotate(contacts[c].contactPointA)
                     : contacts[c].contactPointA;
          const physx::PxVec3 gsWorldB =
              dynIsA ? contacts[c].contactPointB
                     : gsPosition + gsRotation.rotate(contacts[c].contactPointB);
          const physx::PxVec3 relaxedWorldA =
              dynIsA ? relaxedPosition +
                           relaxedRotation.rotate(contacts[c].contactPointA)
                     : contacts[c].contactPointA;
          const physx::PxVec3 relaxedWorldB =
              dynIsA ? contacts[c].contactPointB
                     : relaxedPosition +
                           relaxedRotation.rotate(contacts[c].contactPointB);
          const physx::PxReal gsViolation =
              (gsWorldA - gsWorldB).dot(contacts[c].contactNormal) +
              contacts[c].penetrationDepth;
          const physx::PxReal relaxedViolation =
              (relaxedWorldA - relaxedWorldB).dot(contacts[c].contactNormal) +
              contacts[c].penetrationDepth;
          minGsViolation = physx::PxMin(minGsViolation, gsViolation);
          minRelaxedViolation =
              physx::PxMin(minRelaxedViolation, relaxedViolation);
          foundBodyStatic = true;

          const physx::PxVec3 initialWorldA =
              dynIsA ? bodies[i].prevPosition +
                           bodies[i].prevRotation.rotate(
                               contacts[c].contactPointA)
                     : contacts[c].staticPrevWorldPoint;
          const physx::PxVec3 initialWorldB =
              dynIsA ? contacts[c].staticPrevWorldPoint
                     : bodies[i].prevPosition +
                           bodies[i].prevRotation.rotate(
                               contacts[c].contactPointB);
          const physx::PxReal initialViolation =
              (initialWorldA - initialWorldB).dot(contacts[c].contactNormal) +
              contacts[c].penetrationDepth;
          const physx::PxVec3 outwardNormal =
              contacts[c].contactNormal * (dynIsA ? 1.0f : -1.0f);
          const physx::PxReal approach =
              state.linearVelAtSolveStart &&
                      state.linearVelAtSolveStart->size() == numBodies
                  ? physx::PxMax(
                        0.0f,
                        -(*state.linearVelAtSolveStart)[i].dot(outwardNormal))
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
            minRelaxedViolation > minGsViolation + activeSetTolerance;
      }

      bodies[i].position = rejectBodyStaticOvershoot ? gsPosition
                                                      : relaxedPosition;
      bodies[i].rotation = rejectBodyStaticOvershoot ? gsRotation
                                                      : relaxedRotation;
    }
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  if (state.enableEarlyStop) {
    physx::PxReal maxPositionDelta = 0.0f;
    physx::PxReal maxRotationDelta = 0.0f;
    computeMaxPoseDeltas(bodies, numBodies, state.earlyStopPrevPos,
                         state.earlyStopPrevRot, maxPositionDelta,
                         maxRotationDelta);
    if ((iter + 1) >= state.minIterations &&
        maxPositionDelta <= mConfig.positionTolerance &&
        maxRotationDelta <= state.rotationTolerance) {
      ++state.consecutiveConvergedIterations;
      if (state.consecutiveConvergedIterations >= 2)
        state.iter = state.iters;
    } else {
      state.consecutiveConvergedIterations = 0;
    }
  }
  state.iterationActive = false;
  return state.iter < state.iters;
}

bool AvbdSolver::advanceRigidSolveIterations(
    AvbdRigidSolveIterationState &state) {
  if (!beginRigidSolveIteration(state))
    return false;

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  const physx::PxReal dt = state.dt;
  const AvbdBodyConstraintMap *contactMap = state.contactMap;

  {
    PX_PROFILE_ZONE("AVBD.blockDescent", 0);
    blockDescentIteration(bodies, numBodies, contacts, numContacts, dt,
                          contactMap, state.colorBatches, state.numColors);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f)
        bodies[i].projectLockedPose(bodies[i].prevPosition,
                                    bodies[i].prevRotation);
    }
  }
  completeRigidSolveIteration(state);
  return true;
}

void AvbdSolver::buildRigidDependencyWaves(
    AvbdRigidSolveContext &context) {
  AvbdRigidSolveIterationState &state = context.iteration;
  const physx::PxU32 numBodies = state.numBodies;
  context.dependencyWaveOffsets.clear();
  context.dependencyWaveBodies.clear();
  context.dependencyWaveCount = 0;
  if (numBodies == 0)
    return;

  physx::PxArray<physx::PxU32> bodyOrder(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    bodyOrder[i] = i;

  const bool useDeterministicOrder =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);
  if (useDeterministicOrder) {
    std::sort(bodyOrder.begin(), bodyOrder.end(),
              [&state](physx::PxU32 a, physx::PxU32 b) {
                if (state.bodies[a].invMass != state.bodies[b].invMass)
                  return state.bodies[a].invMass > state.bodies[b].invMass;
                return a < b;
              });
  }

  physx::PxArray<physx::PxU32> orderPosition(numBodies);
  physx::PxArray<physx::PxU32> bodyWave(numBodies);
  for (physx::PxU32 position = 0; position < numBodies; ++position) {
    orderPosition[bodyOrder[position]] = position;
    bodyWave[bodyOrder[position]] = 0;
  }

  // The serial body sweep is a Gauss--Seidel order.  A body depends only on
  // incident dynamic bodies that have already appeared in that order; those
  // edges are acyclic by construction and can therefore be levelized in one
  // forward pass.
  for (physx::PxU32 position = 0; position < numBodies; ++position) {
    const physx::PxU32 body = bodyOrder[position];
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (state.contactMap)
      state.contactMap->getBodyConstraints(body, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : state.numContacts;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c = mapIndices ? mapIndices[loopIndex] : loopIndex;
      const AvbdContactConstraint &contact = state.contacts[c];
      const physx::PxU32 other =
          contact.header.bodyIndexA == body ? contact.header.bodyIndexB
                                             : contact.header.bodyIndexA;
      if (other >= numBodies || other == body ||
          orderPosition[other] >= position)
        continue;
      bodyWave[body] = physx::PxMax(bodyWave[body], bodyWave[other] + 1u);
    }
  }

  physx::PxU32 maxWave = 0;
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    maxWave = physx::PxMax(maxWave, bodyWave[i]);
  context.dependencyWaveCount = maxWave + 1u;
  context.dependencyWaveOffsets.resize(context.dependencyWaveCount + 1u);
  for (physx::PxU32 wave = 0; wave <= context.dependencyWaveCount; ++wave)
    context.dependencyWaveOffsets[wave] = 0;
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    ++context.dependencyWaveOffsets[bodyWave[i] + 1u];
  for (physx::PxU32 wave = 1; wave <= context.dependencyWaveCount; ++wave)
    context.dependencyWaveOffsets[wave] +=
        context.dependencyWaveOffsets[wave - 1u];

  context.dependencyWaveBodies.resize(numBodies);
  physx::PxArray<physx::PxU32> waveWriteOffsets(
      context.dependencyWaveCount);
  for (physx::PxU32 wave = 0; wave < context.dependencyWaveCount; ++wave)
    waveWriteOffsets[wave] = context.dependencyWaveOffsets[wave];
  for (physx::PxU32 position = 0; position < numBodies; ++position) {
    const physx::PxU32 body = bodyOrder[position];
    const physx::PxU32 wave = bodyWave[body];
    context.dependencyWaveBodies[waveWriteOffsets[wave]++] = body;
  }

}

bool AvbdSolver::buildRigidBodyColorPlan(
    AvbdRigidSolveContext &context) {
  PX_PROFILE_ZONE("AVBD.buildRigidBodyColorPlan", 0);
  AvbdRigidSolveIterationState &state = context.iteration;
  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  const AvbdBodyConstraintMap *contactMap = state.contactMap;

  context.bodyColorOffsets.clear();
  context.bodyColorBodies.clear();
  context.bodyColorCount = 0;
  context.maxBodyColorWidth = 0;

  // The fast schedule is deliberately fail-closed.  A partial map or a body
  // index that does not name its island-local slot would make two tasks read
  // and write an unproven ownership graph.
  if (!bodies || !contacts || numBodies == 0 || numContacts == 0 ||
      !contactMap || contactMap->numBodies != numBodies ||
      !contactMap->constraintOffsets || !contactMap->constraintCounts ||
      (contactMap->totalConstraintRefs > 0 &&
       !contactMap->constraintIndices) ||
      contactMap->constraintOffsets[numBodies] !=
          contactMap->totalConstraintRefs)
    return false;

  if (contactMap->constraintOffsets[0] != 0)
    return false;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 begin = contactMap->constraintOffsets[body];
    const physx::PxU32 end = contactMap->constraintOffsets[body + 1u];
    if (begin > end || end > contactMap->totalConstraintRefs ||
        contactMap->constraintCounts[body] != end - begin)
      return false;
  }
  physx::PxArray<physx::PxU32> bodyColors(numBodies);
  physx::PxArray<physx::PxU32> forbiddenColorStamp(numBodies);
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    bodyColors[body] = PX_MAX_U32;
    forbiddenColorStamp[body] = 0;
    if (bodies[body].nodeIndex != body)
      return false;
  }

  physx::PxU32 dynamicBodyCount = 0;
  physx::PxU32 colorCount = 0;
  physx::PxU32 stamp = 0;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    if (bodies[body].invMass <= 0.0f)
      continue;

    ++dynamicBodyCount;
    ++stamp;
    if (stamp == 0) {
      for (physx::PxU32 color = 0; color < numBodies; ++color)
        forbiddenColorStamp[color] = 0;
      stamp = 1;
    }

    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    contactMap->getBodyConstraints(body, mapIndices, mapCount);
    if (mapCount > 0 && !mapIndices)
      return false;
    for (physx::PxU32 ref = 0; ref < mapCount; ++ref) {
      const physx::PxU32 contactIndex = mapIndices[ref];
      if (contactIndex >= numContacts)
        return false;
      const AvbdContactConstraint &contact = contacts[contactIndex];
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      if (bodyA != body && bodyB != body)
        return false;
      const physx::PxU32 other = bodyA == body ? bodyB : bodyA;
      if (other >= numBodies || other == body ||
          bodies[other].invMass <= 0.0f)
        continue;
      const physx::PxU32 otherColor = bodyColors[other];
      if (otherColor < colorCount)
        forbiddenColorStamp[otherColor] = stamp;
    }

    physx::PxU32 color = 0;
    while (color < colorCount && forbiddenColorStamp[color] == stamp)
      ++color;
    if (color == colorCount)
      ++colorCount;
    if (color >= numBodies)
      return false;
    bodyColors[body] = color;
  }

  if (dynamicBodyCount == 0 || colorCount == 0)
    return false;
  // Validate the strict independent-set contract against the source rows,
  // independently of the CSR traversal used to build the plan.
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const physx::PxU32 bodyA = contacts[contactIndex].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[contactIndex].header.bodyIndexB;
    if (bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
        bodies[bodyA].invMass <= 0.0f || bodies[bodyB].invMass <= 0.0f)
      continue;
    if (bodyColors[bodyA] == PX_MAX_U32 ||
        bodyColors[bodyB] == PX_MAX_U32 ||
        bodyColors[bodyA] == bodyColors[bodyB])
      return false;
  }
  context.bodyColorOffsets.resize(colorCount + 1u);
  for (physx::PxU32 color = 0; color <= colorCount; ++color)
    context.bodyColorOffsets[color] = 0;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    if (bodyColors[body] < colorCount)
      ++context.bodyColorOffsets[bodyColors[body] + 1u];
  }
  for (physx::PxU32 color = 1; color <= colorCount; ++color)
    context.bodyColorOffsets[color] +=
        context.bodyColorOffsets[color - 1u];
  if (context.bodyColorOffsets[colorCount] != dynamicBodyCount)
    return false;

  context.bodyColorBodies.resize(dynamicBodyCount);
  physx::PxArray<physx::PxU32> writeOffsets(colorCount);
  for (physx::PxU32 color = 0; color < colorCount; ++color) {
    writeOffsets[color] = context.bodyColorOffsets[color];
    context.maxBodyColorWidth = physx::PxMax(
        context.maxBodyColorWidth,
        context.bodyColorOffsets[color + 1u] -
            context.bodyColorOffsets[color]);
  }
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 color = bodyColors[body];
    if (color < colorCount)
      context.bodyColorBodies[writeOffsets[color]++] = body;
  }

  context.bodyColorCount = colorCount;
  return true;
}
//=============================================================================
// Main Solver Entry Point
//=============================================================================

bool AvbdSolver::prepareRigidSolve(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, const AvbdBodyConstraintMap *contactMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors,
    physx::PxU32 iterationOverride, AvbdSolverStats &stats,
    AvbdRigidSolveContext &context) {
  PX_PROFILE_ZONE("AVBD.prepareRigidSolve", 0);
  context.postAlContactWork.reset();
  if (!mInitialized || numBodies == 0)
    return false;

  context.invDt = 1.0f / dt;
  context.invDt2 = context.invDt * context.invDt;
  context.gravity = gravity;
  context.hasBodyStaticContact = false;
  context.deformableFastImpactIsland = false;
  context.touchingBodyStatic.clear();
  context.linearVelAtSolveStart.clear();
  context.angularVelAtSolveStart.clear();
  AvbdRigidSolveIterationState &iterationState = context.iteration;
  iterationState.bodies = bodies;
  iterationState.numBodies = numBodies;
  iterationState.contacts = contacts;
  iterationState.numContacts = numContacts;
  iterationState.dt = dt;
  iterationState.contactMap = contactMap;
  iterationState.colorBatches = colorBatches;
  iterationState.numColors = numColors;
  iterationState.stats = &stats;
  iterationState.iter = 0;
  iterationState.activeIteration = 0;
  iterationState.iterationActive = false;
  PX_AVBD_PROFILE_STAT(stats.numBodies = numBodies);
  PX_AVBD_PROFILE_STAT(stats.numContacts = numContacts);

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
  context.hasBodyStaticContact = false;
  bool hasDeformableAnchorContact = false;
  bool allBodyVsStatic = (numContacts > 0);
  context.touchingBodyStatic.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    context.touchingBodyStatic[i] = false;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bB = contacts[c].header.bodyIndexB;
    if (isBodyVsStaticContact(bA, bB, numBodies)) {
      context.hasBodyStaticContact = true;
      if (bA < numBodies)
        context.touchingBodyStatic[bA] = true;
      if (bB < numBodies)
        context.touchingBodyStatic[bB] = true;
    } else {
      allBodyVsStatic = false;
    }
    if (hasDeformableStaticAnchor(contacts[c]))
      hasDeformableAnchorContact = true;
  }
  // Fast sphere-on-mesh islands: single dynamic + deformable static only.
  context.deformableFastImpactIsland =
      allBodyVsStatic && hasDeformableAnchorContact;

  // Snapshot pre-solve velocity for material restitution (incl. pure dyn-dyn
  // islands) and deformable fast-impact blend.
  context.linearVelAtSolveStart.clear();
  context.angularVelAtSolveStart.clear();
  if (numContacts > 0) {
    context.linearVelAtSolveStart.resize(numBodies);
    context.angularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      context.linearVelAtSolveStart[i] = bodies[i].linearVelocity;
      context.angularVelAtSolveStart[i] = bodies[i].angularVelocity;
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
            (bodies[i].linearVelocity - bodies[i].prevLinearVelocity) * context.invDt;

        physx::PxReal accelWeight = 0.0f;
        if (!context.touchingBodyStatic[i] && gravMag > 1e-6f) {
          accelWeight =
              physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f);
        }

        // Warmstart position: x = x_n + v*dt + accelWeight * g*dt^2
        // Body-vs-static: start from inertial prediction only. Gravity
        // warmstart overshoots into the mesh on fast impacts without CCD;
        // the supported RHS (accelWeight=0) then fights contacts and ejects.
        if (context.touchingBodyStatic[i]) {
          const bool deformableTouch =
              bodyTouchesDeformableAnchor(contacts, numContacts, i,
                                          contactMap);
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


  iterationState.hasBodyStaticContact = context.hasBodyStaticContact;
  iterationState.linearVelAtSolveStart =
      numContacts > 0 ? &context.linearVelAtSolveStart : nullptr;
  const bool useChebyshev =
      !hasDeformableAnchorContact &&
      mConfig.chebyshevRho > 0.0f &&
      mConfig.chebyshevRho < 1.0f;
  iterationState.useChebyshev = useChebyshev;
  iterationState.chebyOmega = 1.0f;
  if (useChebyshev) {
    iterationState.chebyPrevPos.resize(numBodies);
    iterationState.chebyPrevPrevPos.resize(numBodies);
    iterationState.chebyPrevRot.resize(numBodies);
    iterationState.chebyPrevPrevRot.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      iterationState.chebyPrevPos[i] = bodies[i].position;
      iterationState.chebyPrevPrevPos[i] = bodies[i].position;
      iterationState.chebyPrevRot[i] = bodies[i].rotation;
      iterationState.chebyPrevPrevRot[i] = bodies[i].rotation;
    }
  }
  const physx::PxU32 iters =
      physx::PxMax(mConfig.iterations, iterationOverride);
  iterationState.iters = iters;
  iterationState.minIterations =
      physx::PxMin(iters, physx::PxU32(4));
  iterationState.enableEarlyStop =
      mConfig.enableEarlyStop &&
      iters - iterationState.minIterations > 1;
  iterationState.rotationTolerance =
      physx::PxMax(4.0f * mConfig.positionTolerance /
                       physx::PxMax(mConfig.lengthScale, 1e-6f),
                   1e-4f);
  iterationState.consecutiveConvergedIterations = 0;
  if (iterationState.enableEarlyStop) {
    iterationState.earlyStopPrevPos.resize(numBodies);
    iterationState.earlyStopPrevRot.resize(numBodies);
  }
  return true;
}

void AvbdSolver::finishRigidSolve(AvbdRigidSolveContext &context) {
  AvbdRigidSolveIterationState &iterationState = context.iteration;
  if (!iterationState.bodies || !iterationState.stats)
    return;
  const physx::PxArray<bool> touchesKinematicShell;
  AvbdSolverStats &stats = *iterationState.stats;
  postAlStages(
      iterationState.dt, context.invDt, iterationState.bodies,
      iterationState.numBodies, iterationState.contacts,
      iterationState.numContacts, iterationState.contactMap, context.gravity,
      context.hasBodyStaticContact, context.deformableFastImpactIsland,
      context.touchingBodyStatic,
      iterationState.numContacts > 0
          ? &context.linearVelAtSolveStart
          : nullptr,
      iterationState.numContacts > 0
          ? &context.angularVelAtSolveStart
          : nullptr,
      true, true, nullptr, 0, nullptr, 0, nullptr, 0,
      touchesKinematicShell, nullptr,
      nullptr, nullptr, 0, false, false, false, nullptr, 0, stats,
      &context.postAlContactWork);
}

void AvbdSolver::solve(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                       const AvbdBodyConstraintMap *contactMap,
                       AvbdColorBatch *colorBatches, physx::PxU32 numColors,
                       physx::PxU32 iterationOverride,
                       AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solve", 0);
  AvbdRigidSolveContext context;
  if (!prepareRigidSolve(dt, bodies, numBodies, contacts, numContacts, gravity,
                         contactMap, colorBatches, numColors,
                         iterationOverride, stats, context))
    return;
  while (context.iteration.iter < context.iteration.iters) {
    if (!advanceRigidSolveIterations(context.iteration))
      break;
  }
  finishRigidSolve(context);
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
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    FeatherstoneArticulation *const *articulationForBody,
    const physx::PxU32 *linkIndexForBody,
    AvbdSolverStats &stats,
    AvbdRigidSolveContext *deferredRigidContext) {
  PX_PROFILE_ZONE("AVBD.solveIsland", 0);

  // solveIsland is the sole public island entry and owns transient
  // classification before dispatching to either internal solve module.
  stats.reset();
  if (deferredRigidContext)
    deferredRigidContext->postAlContactWork.reset();
  const bool hasJoints = (numD6 > 0 || numGear > 0);
  const bool hasDeformableSoftVbd =
      softParticles && numSoftParticles > 0 && softBodies &&
      numSoftBodies > 0 &&
      (numSoftContacts == 0 || softContacts);
  const bool contactOnlyTargetOwnership =
      !hasJoints && !hasDeformableSoftVbd;
  // This is the deferred non-ordered rigid path admitted by the task graph.
  // Keep ordered/deterministic and synchronous entries on the original
  // classification sequence even when their island data happens to match.
  const bool fastDeferredRigidClassification =
      deferredRigidContext && contactOnlyTargetOwnership &&
      mConfig.enableParallelization &&
      !mConfig.requiresOrderedBackend();
  physx::PxU8 postAlContactWorkMask = 0;
  bool postAlContactWorkKnown = fastDeferredRigidClassification;
  bool hasExactZeroRestitutionRow = false;
  physx::PxArray<physx::PxU32> rigidStaticContactsPerBody(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    rigidStaticContactsPerBody[i] = 0;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    // ComponentFinalize can only consume a fully supported all-zero-
    // restitution component.  Record the exact predicate before any early
    // continue so the fast path can conservatively avoid its otherwise-empty
    // topology walk.  -0.0f deliberately counts as zero.
    if (fastDeferredRigidClassification && contactOnlyTargetOwnership &&
        contacts[c].restitution == 0.0f)
      hasExactZeroRestitutionRow = true;
    resetAvbdContactObjectiveProgram(contacts[c].objectiveProgram);
    if (!assignAvbdVelocityObjective(
            contacts[c].objectiveProgram,
            AvbdVelocityObjectiveOwner::PositionAL,
            AvbdVelocityObjectiveKind::GeometryNormal,
            AvbdVelocityObjectiveSpan::Normal,
            AvbdVelocityObjectiveReconstruction::PoseDerived,
            1u,
            contacts[c].cacheKey))
      continue;
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
      if (!assignAvbdVelocityObjective(
              contact.objectiveProgram,
              AvbdVelocityObjectiveOwner::PointFinalize,
              AvbdVelocityObjectiveKind::TangentTarget,
              strictFiniteCombinedTarget
                  ? AvbdVelocityObjectiveSpan::NormalAndTangentCone
                  : AvbdVelocityObjectiveSpan::TangentCone,
              AvbdVelocityObjectiveReconstruction::PoseDerived,
              1u,
              contact.cacheKey))
        continue;

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
        AvbdCompiledVelocityObjective *objective =
            findAvbdVelocityObjective(
                contact.objectiveProgram,
                AvbdVelocityObjectiveOwner::PointFinalize,
                AvbdVelocityObjectiveKind::TangentTarget);
        if (objective)
          objective->reconstruction =
              AvbdVelocityObjectiveReconstruction::NormalResponseSpan;
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
  if (contactOnlyTargetOwnership && contacts && numContacts > 0 &&
      (!fastDeferredRigidClassification || hasExactZeroRestitutionRow)) {
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
        const physx::PxU32 *mapIndices = nullptr;
        physx::PxU32 mapCount = 0;
        const bool hasMapRange = getAvbdBodyContactRange(
            contactMap, bodyIndex, mapIndices, mapCount);
        const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
        for (physx::PxU32 loopIndex = 0; loopIndex < loopCount;
             ++loopIndex) {
          const physx::PxU32 c =
              hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
      physx::PxU64 objectiveKey = ~physx::PxU64(0);
      for (physx::PxU32 index = 0;
           index < componentContacts.size(); ++index) {
        objectiveKey =
            physx::PxMin(
                objectiveKey,
                contacts[componentContacts[index]].cacheKey);
      }
      for (physx::PxU32 index = 0;
           index < componentContacts.size(); ++index) {
        const AvbdContactConstraint &contact =
            contacts[componentContacts[index]];
        if (!canAssignAvbdVelocityObjective(
                contact.objectiveProgram,
                AvbdVelocityObjectiveOwner::ComponentFinalize,
                AvbdVelocityObjectiveKind::PassiveFriction,
                AvbdVelocityObjectiveSpan::NormalAndTangentCone,
                AvbdVelocityObjectiveReconstruction::
                    SolveStartInertial,
                componentContacts.size(),
                objectiveKey)) {
          supported = false;
          break;
        }
      }
      if (!supported) {
        for (physx::PxU32 index = 0;
             index < componentContacts.size(); ++index) {
          invalidateAvbdVelocityObjective(
              contacts[componentContacts[index]].objectiveProgram);
        }
        continue;
      }
      for (physx::PxU32 index = 0;
           index < componentContacts.size(); ++index) {
        assignAvbdVelocityObjective(
            contacts[componentContacts[index]].objectiveProgram,
            AvbdVelocityObjectiveOwner::ComponentFinalize,
            AvbdVelocityObjectiveKind::PassiveFriction,
            AvbdVelocityObjectiveSpan::NormalAndTangentCone,
            AvbdVelocityObjectiveReconstruction::SolveStartInertial,
            componentContacts.size(),
            objectiveKey);
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
    physx::PxU64 objectiveKey = ~physx::PxU64(0);
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    const bool hasMapRange = getAvbdBodyContactRange(
        contactMap, bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
    for (physx::PxU32 loopIndex = 0;
         loopIndex < loopCount && supported; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
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
      objectiveKey = physx::PxMin(objectiveKey, contact.cacheKey);
    }
    if (!supported || !haveReferenceTarget)
      continue;
    const bool passiveFriction =
        referenceDynamicTarget.magnitudeSquared() <= 1.0e-12f;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      const AvbdContactConstraint &contact = contacts[c];
      if ((contact.header.bodyIndexA == bodyIndex ||
           contact.header.bodyIndexB == bodyIndex) &&
          !canAssignAvbdVelocityObjective(
              contact.objectiveProgram,
              AvbdVelocityObjectiveOwner::ManifoldFinalize,
              passiveFriction
                  ? AvbdVelocityObjectiveKind::PassiveFriction
                  : AvbdVelocityObjectiveKind::TangentTarget,
              AvbdVelocityObjectiveSpan::NormalAndTangentCone,
              AvbdVelocityObjectiveReconstruction::SolveStartInertial,
              rigidStaticContactsPerBody[bodyIndex],
              objectiveKey)) {
        supported = false;
        break;
      }
    }
    if (!supported) {
      for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
        const physx::PxU32 c =
            hasMapRange ? mapIndices[loopIndex] : loopIndex;
        AvbdContactConstraint &contact = contacts[c];
        if (contact.header.bodyIndexA == bodyIndex ||
            contact.header.bodyIndexB == bodyIndex)
          invalidateAvbdVelocityObjective(contact.objectiveProgram);
      }
      continue;
    }
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      AvbdContactConstraint &contact = contacts[c];
      if (contact.header.bodyIndexA == bodyIndex ||
          contact.header.bodyIndexB == bodyIndex) {
        assignAvbdVelocityObjective(
            contact.objectiveProgram,
            AvbdVelocityObjectiveOwner::ManifoldFinalize,
            passiveFriction
                ? AvbdVelocityObjectiveKind::PassiveFriction
                : AvbdVelocityObjectiveKind::TangentTarget,
            AvbdVelocityObjectiveSpan::NormalAndTangentCone,
            AvbdVelocityObjectiveReconstruction::
                SolveStartInertial,
            rigidStaticContactsPerBody[bodyIndex],
            objectiveKey);
      }
    }
  }

  // Specialized target/manifold/component programs have first claim.
  // Compile all remaining ordinary rigid contact sources through the same
  // helper used by joint islands, so owner classification has one entry point.
  if (contactOnlyTargetOwnership)
    compileAvbdOrdinaryRigidContactObjectives(
        contacts, numContacts, numBodies, contactMap);

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
    if (hasJoints || hasDeformableSoftVbd)
      continue;
    if (contact.restitution != 0.0f)
      continue;
    if (contact.maxImpulse <= 1.0e20f)
      continue;
    const physx::PxReal targetNormal =
        contact.targetVelocity.dot(contact.contactNormal);
    const physx::PxReal targetTangent0 =
        contact.targetVelocity.dot(contact.tangent0);
    const physx::PxReal targetTangent1 =
        contact.targetVelocity.dot(contact.tangent1);
    if (physx::PxAbs(targetNormal) > 1.0e-6f ||
        physx::PxAbs(targetTangent0) > 1.0e-6f ||
        physx::PxAbs(targetTangent1) > 1.0e-6f) {
      continue;
    }
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxReal dynamicLinearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal dynamicAngularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (physx::PxAbs(dynamicLinearScale - 1.0f) > 1.0e-6f ||
        physx::PxAbs(dynamicAngularScale - 1.0f) > 1.0e-6f) {
      continue;
    }
    if (!assignAvbdVelocityObjective(
            contact.objectiveProgram,
            AvbdVelocityObjectiveOwner::PositionAL,
            AvbdVelocityObjectiveKind::PassiveFriction,
            AvbdVelocityObjectiveSpan::TangentCone,
            AvbdVelocityObjectiveReconstruction::PoseDerived,
            1u,
            contact.cacheKey))
      continue;
  }

  // Publish the remaining authored source slots as an explicit migration
  // backlog. Geometry normal is already compiled independently above.
  // Material normal exists for every contact; material tangent exists only
  // when friction or an authored tangential target is present.
  //
  // On the admitted deferred non-ordered rigid path, publication and
  // validation have no cross-contact dependency: both only read/write the
  // current program.  Fuse them to avoid one complete wide-contact walk.
  // Ordered/synchronous paths retain the original two-pass sequence.
  if (fastDeferredRigidClassification) {
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      AvbdContactConstraint &contact = contacts[c];
      physx::PxU8 authoredSourceSlots =
          eCONTACT_SOURCE_GEOMETRY_NORMAL |
          eCONTACT_SOURCE_MATERIAL_NORMAL;
      const physx::PxReal targetTangent0 =
          contact.targetVelocity.dot(contact.tangent0);
      const physx::PxReal targetTangent1 =
          contact.targetVelocity.dot(contact.tangent1);
      if (contact.friction > 0.0f || contact.staticFriction > 0.0f ||
          physx::PxAbs(targetTangent0) > 1.0e-6f ||
          physx::PxAbs(targetTangent1) > 1.0e-6f)
        authoredSourceSlots = physx::PxU8(
            authoredSourceSlots |
            eCONTACT_SOURCE_MATERIAL_TANGENT);
      setAvbdContactObjectiveLegacySources(
          contact.objectiveProgram, authoredSourceSlots);
      if (!isValidAvbdContactObjectiveProgram(contact.objectiveProgram)) {
        invalidateAvbdVelocityObjective(contact.objectiveProgram);
        postAlContactWorkKnown = false;
      } else {
        markAvbdContactObjectiveProgramValidated(contact.objectiveProgram);
        postAlContactWorkMask = physx::PxU8(
            postAlContactWorkMask |
            collectValidatedPostAlContactWork(contact, bodies, numBodies));
      }
    }
  } else {
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      AvbdContactConstraint &contact = contacts[c];
      physx::PxU8 authoredSourceSlots =
          eCONTACT_SOURCE_GEOMETRY_NORMAL |
          eCONTACT_SOURCE_MATERIAL_NORMAL;
      const physx::PxReal targetTangent0 =
          contact.targetVelocity.dot(contact.tangent0);
      const physx::PxReal targetTangent1 =
          contact.targetVelocity.dot(contact.tangent1);
      if (contact.friction > 0.0f || contact.staticFriction > 0.0f ||
          physx::PxAbs(targetTangent0) > 1.0e-6f ||
          physx::PxAbs(targetTangent1) > 1.0e-6f)
        authoredSourceSlots = physx::PxU8(
            authoredSourceSlots |
            eCONTACT_SOURCE_MATERIAL_TANGENT);
      setAvbdContactObjectiveLegacySources(
          contact.objectiveProgram, authoredSourceSlots);
    }

    // The compiled program is the only ownership authority consumed below.
    // Any internally inconsistent program is converted to the explicit
    // fail-closed state before position or velocity stages can inspect it.
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      if (!isValidAvbdContactObjectiveProgram(
              contacts[c].objectiveProgram)) {
        invalidateAvbdVelocityObjective(
            contacts[c].objectiveProgram);
      } else {
        markAvbdContactObjectiveProgramValidated(
            contacts[c].objectiveProgram);
      }
    }
  }

  // One island entry: joint/genuine-soft module vs contact-only module. NP
  // contact data cannot synthesize soft particles or route through a second
  // primal.
  if (deferredRigidContext) {
    if (hasJoints || hasDeformableSoftVbd)
      return;
    if (prepareRigidSolve(dt, bodies, numBodies, contacts, numContacts,
                          gravity, contactMap, colorBatches, numColors,
                          iterationOverride, stats, *deferredRigidContext) &&
        postAlContactWorkKnown) {
      deferredRigidContext->postAlContactWork.publish(postAlContactWorkMask);
    }
    return;
  }
  if (hasJoints || hasDeformableSoftVbd) {
    solveWithJoints(dt, bodies, numBodies, contacts, numContacts, d6Joints,
                    numD6, gearJoints, numGear, gravity, contactMap, d6Map,
                    gearMap, colorBatches, numColors, iterationOverride,
                    softParticles, numSoftParticles, softBodies, numSoftBodies,
                    softContacts, numSoftContacts, softExecutionPlan,
                    articulationForBody,
                    linkIndexForBody, stats);
  } else {
    solve(dt, bodies, numBodies, contacts, numContacts, gravity, contactMap,
          colorBatches, numColors, iterationOverride, stats);
  }

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
  (void)stats;
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

  PX_AVBD_PROFILE_STAT(stats.constraintError =       (numActive > 0) ? sqrtf(totalError / (physx::PxReal)numActive) : 0.0f);
}

void AvbdSolver::solveRigidDualRange(
    AvbdRigidSolveIterationState &state, physx::PxU32 begin,
    physx::PxU32 end) {
  PX_ASSERT(state.bodies && state.contacts);
  PX_ASSERT(begin < end && end <= state.numContacts);
  // The fast path deliberately reuses the established per-contact kernel
  // rather than cloning its numerically sensitive /fp:fast expressions.
  // Admission keeps every range wider than four rows, preserving the only
  // physical branch in that kernel that depends on the supplied row count.
  PX_ASSERT(end - begin > 4u);
  AvbdSolverStats rangeStats;
  rangeStats.reset();
  updateLagrangianMultipliers(
      state.bodies, state.numBodies, state.contacts + begin, end - begin,
      state.dt, rangeStats);
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
    physx::PxU32 &numTouching,
    const physx::PxU32 *rigidTargetContactStarts,
    const physx::PxU32 *rigidTargetContactRefs) {

  const bool useRigidTargetContactCsr =
      softContacts && numSoftContacts > 0 &&
      bodyIndex < numBodies && rigidTargetContactStarts &&
      (rigidTargetContactRefs ||
       rigidTargetContactStarts[bodyIndex] ==
           rigidTargetContactStarts[bodyIndex + 1]);
  bool bodyUsesSoftContactNormals = false;
  if (softContacts && numSoftContacts > 0) {
    if (useRigidTargetContactCsr) {
      bodyUsesSoftContactNormals =
          rigidTargetContactStarts[bodyIndex] !=
          rigidTargetContactStarts[bodyIndex + 1];
    } else {
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
        const AvbdSoftContactGeometry &geometry =
            softContacts[sci].geometry;
        if (geometry.hasRigidBodyTarget() &&
            geometry.targetIndex == bodyIndex) {
          bodyUsesSoftContactNormals = true;
          break;
        }
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
    if (bodyUsesSoftContactNormals &&
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
    AvbdVec6 softContactRhs;
    softContactRhs.linear = physx::PxVec3(0.0f);
    softContactRhs.angular = physx::PxVec3(0.0f);
    const auto accumulateSoftContact =
        [&](physx::PxU32 sci) {
      const AvbdSoftContact &sc = softContacts[sci];
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      const AvbdSoftContactAugmentedState &state = sc.state;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex != bodyIndex)
        return;
      if (avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles)) {
        avbdAddKinematicShellContactContribution_rigid(
            geometry, state, bodyIndex, body,
            shellBoostFloor, A, softContactRhs);
        numTouching++;
      } else if (
          avbdAddDynamicSoftRigidContactContribution_rigid(
              geometry, state, bodyIndex, softParticles,
              numSoftParticles, body, A, softContactRhs)) {
        numTouching++;
      }
    };
    if (useRigidTargetContactCsr) {
      for (physx::PxU32 refIndex =
               rigidTargetContactStarts[bodyIndex];
           refIndex < rigidTargetContactStarts[bodyIndex + 1];
           ++refIndex)
        accumulateSoftContact(rigidTargetContactRefs[refIndex]);
    } else {
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
        accumulateSoftContact(sci);
    }
    gLinear += softContactRhs.linear;
    gAngular += softContactRhs.angular;
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

void AvbdSolver::solveRigidBodyRange(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, physx::PxReal invDt2,
    const AvbdBodyConstraintMap *contactMap, const physx::PxU32 *bodyOrder,
    physx::PxU32 begin, physx::PxU32 end) {
  PX_PROFILE_ZONE("AVBD.solveRigidBodyRange", 0);
  PX_ASSERT(begin <= end && end <= numBodies);
  for (physx::PxU32 idx = begin; idx < end; ++idx) {
    const physx::PxU32 i = bodyOrder ? bodyOrder[idx] : idx;
    if (bodies[i].invMass <= 0.0f)
      continue;
    if (mConfig.enableLocal6x6Solve) {
      solveLocalSystem(bodies[i], bodies, numBodies, contacts, numContacts, dt,
                       invDt2, contactMap);
    } else {
      solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                 numContacts, nullptr, 0, nullptr, 0, dt,
                                 invDt2, contactMap, nullptr, nullptr);
    }
  }
}

bool AvbdSolver::solveRigidOwnerFallback(
    AvbdRigidSolveContext &context, const physx::PxU32 *ownerBodyOrder,
    physx::PxU32 lane) {
  if (!ownerBodyOrder || !context.iteration.bodies ||
      !context.iteration.contacts || !context.iteration.contactMap ||
      lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH)
    return false;
  solveRigidBodyRange(
      context.iteration.bodies, context.iteration.numBodies,
      context.iteration.contacts, context.iteration.numContacts,
      context.iteration.dt, context.invDt2, context.iteration.contactMap,
      ownerBodyOrder, lane, lane + 1u);
  return true;
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

  // P2 removes the AVBD-private worker path.  A non-conflicting colored body
  // stage will be submitted through the Scene taskgraph in P4; until then
  // retain the authoritative Gauss-Seidel body order rather than silently
  // changing the solve to an unscheduled Jacobi variant.
  solveRigidBodyRange(bodies, numBodies, contacts, numContacts, dt, invDt2,
                      contactMap, orderPtr, 0, numBodies);
}

} // namespace Dy
} // namespace physx
