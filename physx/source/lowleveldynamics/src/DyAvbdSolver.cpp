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
    physx::PxReal y = value - c;
    physx::PxReal t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

// Contact detection starts from a cooked collision proxy, while Scene expands
// the accepted row to a weighted simulation-particle query point before the
// solve.  A cached proxy face normal/surface is therefore not authoritative
// for a late safety projection.  Box contacts retain this compact descriptor
// so the recovery stages can query the actual, current OBB at the expanded
// point.  This is endpoint DCD only: it neither samples a swept segment nor
// uses the OGC shell margin as penetration.
struct AvbdCurrentRigidBoxSdf {
  physx::PxReal signedDistance{0.0f};
  physx::PxVec3 normal{0.0f};
  physx::PxVec3 surfacePoint{0.0f};
  // Surface point relative to the dynamic solver body's origin.  It is zero
  // for a world-static box, which has no movable endpoint.
  physx::PxVec3 rigidOffset{0.0f};
};

static bool queryCurrentRigidBoxSdf(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSolverBody *dynamicBody, const physx::PxVec3 &queryPoint,
    AvbdCurrentRigidBoxSdf &result) {
  if (!geometry.hasRigidBoxSdf || !queryPoint.isFinite())
    return false;

  const physx::PxVec3 halfExtent = geometry.rigidBoxHalfExtent;
  if (!halfExtent.isFinite() || halfExtent.x <= 0.0f ||
      halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;

  physx::PxTransform shapeToWorld = geometry.rigidBoxPose;
  if (geometry.hasRigidBodyTarget()) {
    if (!dynamicBody || !dynamicBody->position.isFinite() ||
        !dynamicBody->rotation.isFinite())
      return false;
    shapeToWorld = physx::PxTransform(dynamicBody->position,
                                      dynamicBody->rotation) *
                   geometry.rigidBoxPose;
  } else if (!geometry.hasWorldStaticTarget()) {
    return false;
  }

  const physx::PxVec3 localPoint = shapeToWorld.transformInv(queryPoint);
  if (!localPoint.isFinite())
    return false;

  const physx::PxVec3 q(physx::PxAbs(localPoint.x) - halfExtent.x,
                         physx::PxAbs(localPoint.y) - halfExtent.y,
                         physx::PxAbs(localPoint.z) - halfExtent.z);
  const bool inside = q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f;
  physx::PxReal signedDistance = 0.0f;
  physx::PxVec3 localNormal(0.0f);
  physx::PxVec3 surfaceLocal(0.0f);
  if (inside) {
    signedDistance = physx::PxMax(q.x, physx::PxMax(q.y, q.z));
    if (q.x > q.y && q.x > q.z)
      localNormal = physx::PxVec3(localPoint.x >= 0.0f ? 1.0f : -1.0f,
                                   0.0f, 0.0f);
    else if (q.y > q.z)
      localNormal = physx::PxVec3(0.0f,
                                   localPoint.y >= 0.0f ? 1.0f : -1.0f,
                                   0.0f);
    else
      localNormal = physx::PxVec3(0.0f, 0.0f,
                                   localPoint.z >= 0.0f ? 1.0f : -1.0f);
    surfaceLocal = localPoint - localNormal * signedDistance;
  } else {
    const physx::PxVec3 outside(
        physx::PxMax(q.x, 0.0f), physx::PxMax(q.y, 0.0f),
        physx::PxMax(q.z, 0.0f));
    signedDistance = outside.magnitude();
    if (signedDistance > 1.0e-10f) {
      localNormal = physx::PxVec3(
                        localPoint.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f) *
                    outside.x +
                    physx::PxVec3(0.0f,
                        localPoint.y >= 0.0f ? 1.0f : -1.0f, 0.0f) *
                    outside.y +
                    physx::PxVec3(0.0f, 0.0f,
                        localPoint.z >= 0.0f ? 1.0f : -1.0f) *
                    outside.z;
      localNormal *= 1.0f / signedDistance;
    } else {
      // The inside branch owns the exact boundary.  This only guards an
      // underflowing outside distance and keeps the query fail-safe.
      localNormal = physx::PxVec3(0.0f, 1.0f, 0.0f);
    }
    surfaceLocal = localPoint;
    surfaceLocal.x = physx::PxClamp(surfaceLocal.x, -halfExtent.x,
                                     halfExtent.x);
    surfaceLocal.y = physx::PxClamp(surfaceLocal.y, -halfExtent.y,
                                     halfExtent.y);
    surfaceLocal.z = physx::PxClamp(surfaceLocal.z, -halfExtent.z,
                                     halfExtent.z);
  }

  const physx::PxVec3 normal = shapeToWorld.q.rotate(localNormal);
  const physx::PxReal normalLengthSq = normal.magnitudeSquared();
  const physx::PxVec3 surfacePoint = shapeToWorld.transform(surfaceLocal);
  if (!physx::PxIsFinite(signedDistance) || !normal.isFinite() ||
      !physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f ||
      !surfacePoint.isFinite())
    return false;

  result.signedDistance = signedDistance;
  result.normal = normal * physx::PxRecipSqrt(normalLengthSq);
  result.surfacePoint = surfacePoint;
  result.rigidOffset = geometry.hasRigidBodyTarget()
                           ? dynamicBody->rotation.rotate(
                                 geometry.rigidBoxPose.transform(surfaceLocal))
                           : physx::PxVec3(0.0f);
  return result.rigidOffset.isFinite();
}

static bool getCurrentWorldStaticSoftContactGeometry(
    const AvbdSoftContactGeometry &geometry,
    const physx::PxVec3 &queryPoint, physx::PxVec3 &normal,
    physx::PxReal &trueGap) {
  if (geometry.hasRigidBoxSdf) {
    AvbdCurrentRigidBoxSdf boxQuery;
    if (!queryCurrentRigidBoxSdf(geometry, nullptr, queryPoint, boxQuery))
      return false;
    normal = boxQuery.normal;
    trueGap = boxQuery.signedDistance;
    return true;
  }

  const physx::PxReal normalLengthSq = geometry.normal.magnitudeSquared();
  if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f ||
      !geometry.surfacePoint.isFinite())
    return false;
  normal = geometry.normal * physx::PxRecipSqrt(normalLengthSq);
  trueGap = (queryPoint - geometry.surfacePoint).dot(normal);
  return physx::PxIsFinite(trueGap);
}

static bool getCurrentDynamicSoftRigidContactGeometry(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const physx::PxVec3 &queryPoint, physx::PxVec3 &normal,
    physx::PxVec3 &worldOffset, physx::PxReal &trueGap) {
  if (geometry.hasRigidBoxSdf) {
    AvbdCurrentRigidBoxSdf boxQuery;
    if (!queryCurrentRigidBoxSdf(geometry, &body, queryPoint, boxQuery))
      return false;
    normal = boxQuery.normal;
    worldOffset = boxQuery.rigidOffset;
    trueGap = boxQuery.signedDistance;
    return true;
  }

  const physx::PxReal normalLengthSq = geometry.normal.magnitudeSquared();
  if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f ||
      !body.position.isFinite() || !body.rotation.isFinite())
    return false;
  normal = geometry.normal * physx::PxRecipSqrt(normalLengthSq);
  worldOffset = body.rotation.rotate(geometry.rigidLocalPoint);
  const physx::PxVec3 surfacePoint = body.position + worldOffset;
  if (!worldOffset.isFinite() || !surfacePoint.isFinite())
    return false;
  trueGap = (queryPoint - surfacePoint).dot(normal);
  return physx::PxIsFinite(trueGap);
}

static void projectDynamicSoftRigidVelocityTangents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt) {
  if (!avbdUseVelocityTangentOwner() || !bodies || !softParticles ||
      !softBodies || !softContacts || dt <= 0.0f || !physx::PxIsFinite(dt))
    return;

  for (physx::PxU32 contactIndex = 0; contactIndex < numSoftContacts;
       ++contactIndex) {
    const AvbdSoftContact &contact = softContacts[contactIndex];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    const AvbdSoftContactAugmentedState &state = contact.state;
    if (geometry.tangentOwner != AvbdSoftContactTangentOwner::eVELOCITY ||
        !geometry.hasRigidBodyTarget() || geometry.targetIndex >= numBodies ||
        !avbdCanUseVelocityTangentOwner(geometry, softBodies, numSoftBodies,
                                        softParticles, numSoftParticles) ||
        !physx::PxIsFinite(state.alLambda) || state.alLambda >= 0.0f)
      continue;

    AvbdSolverBody &body = bodies[geometry.targetIndex];
    if (body.invMass <= 0.0f || !physx::PxIsFinite(body.invMass) ||
        !body.position.isFinite() || !body.rotation.isFinite() ||
        !body.linearVelocity.isFinite() || !body.angularVelocity.isFinite() ||
        !body.invInertiaWorld.column0.isFinite() ||
        !body.invInertiaWorld.column1.isFinite() ||
        !body.invInertiaWorld.column2.isFinite())
      continue;

    physx::PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
    const physx::PxU32 supportCount =
        avbdCollectSoftContactParticleIndices(geometry, supportIndices);
    if (supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
      continue;
    physx::PxReal softResponse = 0.0f;
    physx::PxVec3 queryVelocity(0.0f);
    bool valid = true;
    for (physx::PxU32 supportIndex = 0; supportIndex < supportCount;
         ++supportIndex) {
      const physx::PxU32 particleIndex = supportIndices[supportIndex];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight) ||
          !softParticles[particleIndex].velocity.isFinite()) {
        valid = false;
        break;
      }
      softResponse += weight * weight * softParticles[particleIndex].invMass;
      queryVelocity += softParticles[particleIndex].velocity * weight;
    }
    if (!valid || !physx::PxIsFinite(softResponse) ||
        softResponse <= 1.0e-12f || !queryVelocity.isFinite())
      continue;

    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    physx::PxVec3 normal(0.0f), worldOffset(0.0f);
    physx::PxReal trueGap = 0.0f;
    if (!queryPoint.isFinite() ||
        !getCurrentDynamicSoftRigidContactGeometry(
            geometry, body, queryPoint, normal, worldOffset, trueGap))
      continue;

    const physx::PxVec3 tangents[2] = {geometry.tangent1,
                                       geometry.tangent2};
    physx::PxVec3 linearJacobian[2];
    physx::PxVec3 linearDelta[2];
    physx::PxVec3 angularJacobian[2];
    physx::PxVec3 angularDelta[2];
    for (physx::PxU32 axis = 0; axis < 2; ++axis) {
      linearJacobian[axis] = -tangents[axis];
      body.projectLockedLinearVector(linearJacobian[axis]);
      linearDelta[axis] = linearJacobian[axis] * body.invMass;
      angularJacobian[axis] = -worldOffset.cross(tangents[axis]);
      body.projectLockedAngularVector(angularJacobian[axis]);
      angularDelta[axis] =
          body.invInertiaWorld * angularJacobian[axis];
      body.projectLockedAngularVector(angularDelta[axis]);
      if (!linearDelta[axis].isFinite() ||
          !angularDelta[axis].isFinite()) {
        valid = false;
        break;
      }
    }
    if (!valid)
      continue;

    const physx::PxVec3 rigidSurfaceVelocity =
        body.linearVelocity + body.angularVelocity.cross(worldOffset);
    const physx::PxVec3 relativeVelocity =
        queryVelocity - rigidSurfaceVelocity;
    const physx::PxReal velocity0 = relativeVelocity.dot(tangents[0]);
    const physx::PxReal velocity1 = relativeVelocity.dot(tangents[1]);
    const physx::PxReal k00 = softResponse +
        linearJacobian[0].dot(linearDelta[0]) +
        angularJacobian[0].dot(angularDelta[0]);
    const physx::PxReal k01 =
        linearJacobian[0].dot(linearDelta[1]) +
        angularJacobian[0].dot(angularDelta[1]);
    const physx::PxReal k11 = softResponse +
        linearJacobian[1].dot(linearDelta[1]) +
        angularJacobian[1].dot(angularDelta[1]);
    const physx::PxReal determinant = k00 * k11 - k01 * k01;
    if (!relativeVelocity.isFinite() || !physx::PxIsFinite(velocity0) ||
        !physx::PxIsFinite(velocity1) || !physx::PxIsFinite(determinant) ||
        determinant <= 1.0e-12f)
      continue;

    physx::PxReal impulse0 = (-k11 * velocity0 + k01 * velocity1) /
                             determinant;
    physx::PxReal impulse1 = (k01 * velocity0 - k00 * velocity1) /
                             determinant;
    const physx::PxReal tangentLimit = geometry.friction *
        physx::PxMax(-state.alLambda, 0.0f) * dt;
    const physx::PxReal impulseMagnitude = physx::PxSqrt(
        impulse0 * impulse0 + impulse1 * impulse1);
    if (!physx::PxIsFinite(tangentLimit) || tangentLimit < 0.0f ||
        !physx::PxIsFinite(impulseMagnitude))
      continue;
    if (impulseMagnitude > tangentLimit && impulseMagnitude > 1.0e-12f) {
      const physx::PxReal scale = tangentLimit / impulseMagnitude;
      impulse0 *= scale;
      impulse1 *= scale;
    }
    if (physx::PxAbs(impulse0) <= 1.0e-12f &&
        physx::PxAbs(impulse1) <= 1.0e-12f)
      continue;

    const physx::PxVec3 softImpulse =
        tangents[0] * impulse0 + tangents[1] * impulse1;
    physx::PxVec3 candidateSoftVelocity[AVBD_CONTACT_MAX_PARTICLES];
    for (physx::PxU32 supportIndex = 0; supportIndex < supportCount;
         ++supportIndex) {
      const physx::PxU32 particleIndex = supportIndices[supportIndex];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      candidateSoftVelocity[supportIndex] =
          softParticles[particleIndex].velocity +
          softImpulse * (softParticles[particleIndex].invMass * weight);
      if (!candidateSoftVelocity[supportIndex].isFinite()) {
        valid = false;
        break;
      }
    }
    physx::PxVec3 candidateLinearVelocity = body.linearVelocity +
        linearDelta[0] * impulse0 + linearDelta[1] * impulse1;
    physx::PxVec3 candidateAngularVelocity = body.angularVelocity +
        angularDelta[0] * impulse0 + angularDelta[1] * impulse1;
    body.projectLockedLinearVector(candidateLinearVelocity);
    body.projectLockedAngularVector(candidateAngularVelocity);
    if (!valid || !candidateLinearVelocity.isFinite() ||
        !candidateAngularVelocity.isFinite())
      continue;

    for (physx::PxU32 supportIndex = 0; supportIndex < supportCount;
         ++supportIndex)
      softParticles[supportIndices[supportIndex]].velocity =
          candidateSoftVelocity[supportIndex];
    body.linearVelocity = candidateLinearVelocity;
    body.angularVelocity = candidateAngularVelocity;
    body.projectLockedVelocities();
  }
}

// Compile a current-pose manifold into the same pair representation used by
// the prediction epoch.  The contact geometry is scratch-owned because it is
// rebuilt at t=dt; the state type and pair identity are shared.  A following
// bridge transfers terminal phase results back to the selection-owned epoch
// records without letting stale prediction representatives index this fresh
// manifold.
static void buildCurrentOgcPairStates(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxArray<AvbdOgcPairState> &pairStates,
    physx::PxArray<physx::PxU32> &pairIndices) {
  pairStates.clear();
  pairIndices.resize(numContacts);
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    pairIndices[contactIndex] = PX_MAX_U32;
    const AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
    const bool dynamicRigid =
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies;
    const bool worldStatic =
        (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
         geometry.source.type == AvbdSoftContactSource::eGROUND) &&
        geometry.hasWorldStaticTarget();
    if ((!dynamicRigid && !worldStatic) ||
        geometry.queryBodyIndex == PX_MAX_U32 ||
        !avbdHasSoftContactDynamicQuerySupport(geometry, softParticles,
                                               numSoftParticles))
      continue;

    physx::PxU32 pairIndex = PX_MAX_U32;
    for (physx::PxU32 candidateIndex = 0;
         candidateIndex < pairStates.size(); ++candidateIndex) {
      const AvbdOgcPairState &candidate = pairStates[candidateIndex];
      if (candidate.sourceType == geometry.source.type &&
          candidate.targetKind == geometry.targetKind &&
          candidate.sourceBodyIndex == geometry.queryBodyIndex &&
          candidate.targetBodyIndex == geometry.targetIndex &&
          candidate.primitiveKey == geometry.source.primitiveKey) {
        pairIndex = candidateIndex;
        break;
      }
    }
    if (pairIndex == PX_MAX_U32) {
      pairIndex = pairStates.size();
      AvbdOgcPairState pair;
      pair.sourceType = geometry.source.type;
      pair.targetKind = geometry.targetKind;
      pair.sourceBodyIndex = geometry.queryBodyIndex;
      pair.targetBodyIndex = geometry.targetIndex;
      pair.primitiveKey = geometry.source.primitiveKey;
      pair.active = true;
      pairStates.pushBack(pair);
    }

    AvbdOgcPairState &pair = pairStates[pairIndex];
    ++pair.contactCount;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    physx::PxVec3 normal(0.0f), rigidOffset(0.0f);
    physx::PxReal gap = 0.0f;
    const bool geometryValid = dynamicRigid
        ? queryPoint.isFinite() &&
              getCurrentDynamicSoftRigidContactGeometry(
                  geometry, bodies[geometry.targetIndex], queryPoint, normal,
                  rigidOffset, gap)
        : queryPoint.isFinite() &&
              getCurrentWorldStaticSoftContactGeometry(
                  geometry, queryPoint, normal, gap);
    if (geometryValid && physx::PxIsFinite(gap)) {
      if (pair.contactCount == 1u || gap < pair.referenceGap)
        pair.referenceGap = gap;
      if (pair.representativeContact == PX_MAX_U32 ||
          gap < pair.representativeGap) {
        pair.representativeContact = contactIndex;
        pair.representativeNormal = normal;
        pair.representativeRigidOffset = rigidOffset;
        pair.referenceRelativePoint = dynamicRigid
            ? queryPoint -
                  (bodies[geometry.targetIndex].position + rigidOffset)
            : queryPoint;
        pair.representativeGap = gap;
      }
      if (pair.representativeContact == contactIndex || gap < pair.safetyGap)
        pair.safetyGap = gap;
      pair.remainingSafeDisplacement = physx::PxMax(pair.safetyGap, 0.0f);
      pair.minimumGap = pair.safetyGap;
      pair.epoch = 1u;
    }
    pairIndices[contactIndex] = pairIndex;
  }
}

// Terminal geometry needs fresh row indices, but the OGC scheduler must still
// observe one pair lifecycle.  Link the transient manifold records to the
// selection-owned state by the complete body/shape key.  A missing parent is
// fail-closed for shared scheduling; the local current-pose manifold still
// owns only its own geometric repair.
static void linkCurrentOgcPairStates(
    physx::PxArray<AvbdOgcPairState> &currentPairStates,
    AvbdOgcPairState *selectionPairStates,
    physx::PxU32 numSelectionPairStates,
    physx::PxArray<physx::PxU32> &parentIndices) {
  parentIndices.resize(currentPairStates.size());
  for (physx::PxU32 currentIndex = 0;
       currentIndex < currentPairStates.size(); ++currentIndex) {
    parentIndices[currentIndex] = PX_MAX_U32;
    AvbdOgcPairState &current = currentPairStates[currentIndex];
    for (physx::PxU32 selectionIndex = 0;
         selectionIndex < numSelectionPairStates; ++selectionIndex) {
      AvbdOgcPairState &selection = selectionPairStates[selectionIndex];
      if (!selection.active || selection.sourceType != current.sourceType ||
          selection.targetKind != current.targetKind ||
          selection.sourceBodyIndex != current.sourceBodyIndex ||
          selection.targetBodyIndex != current.targetBodyIndex ||
          selection.primitiveKey != current.primitiveKey)
        continue;
      parentIndices[currentIndex] = selectionIndex;
      current.epoch = selection.epoch;
      current.admittedAtBoundary = selection.admittedAtBoundary;
      current.refreshRequested = selection.refreshRequested;
      selection.minimumGap =
          physx::PxMin(selection.minimumGap, current.minimumGap);
      break;
    }
  }
}

static void publishCurrentOgcPairStates(
    const physx::PxArray<AvbdOgcPairState> &currentPairStates,
    const physx::PxArray<physx::PxU32> &parentIndices,
    AvbdOgcPairState *selectionPairStates,
    physx::PxU32 numSelectionPairStates) {
  if (parentIndices.size() != currentPairStates.size())
    return;
  for (physx::PxU32 currentIndex = 0;
       currentIndex < currentPairStates.size(); ++currentIndex) {
    const physx::PxU32 parentIndex = parentIndices[currentIndex];
    if (parentIndex >= numSelectionPairStates)
      continue;
    const AvbdOgcPairState &current = currentPairStates[currentIndex];
    AvbdOgcPairState &selection = selectionPairStates[parentIndex];
    selection.minimumGap =
        physx::PxMin(selection.minimumGap, current.minimumGap);
    selection.hasTriangleCoreManifold =
        selection.hasTriangleCoreManifold || current.hasTriangleCoreManifold;
    selection.triangleCoreLocallyResolved =
        selection.triangleCoreLocallyResolved ||
        current.triangleCoreLocallyResolved;
    if (current.hasTriangleCoreManifold) {
      selection.triangleCoreFace = current.triangleCoreFace;
      selection.triangleCoreFaceExit = current.triangleCoreFaceExit;
    }
    // A terminal manifold was consumed at this same simulation time.  Its
    // refresh request must not leak into the next selection epoch.
    selection.refreshRequested = false;
  }
}

// Return the detector-time relative translation which moves the entire
// triangle through one selected box face.  This uses the complete triangle
// bounds rather than the centroid SDF witness: after translating by this
// amount, every point of the triangle is outside that face's supporting
// plane, so the convex triangle cannot still overlap the OBB.
static bool getRigidBoxTriangleCoreExitDistance(
    const AvbdSoftContactGeometry &geometry, physx::PxU32 face,
    physx::PxReal &distance) {
  if (!geometry.hasRigidBoxTriangleCoreExit ||
      !geometry.rigidBoxHalfExtent.isFinite() ||
      !geometry.rigidBoxTriangleCoreMinimumLocal.isFinite() ||
      !geometry.rigidBoxTriangleCoreMaximumLocal.isFinite() ||
      face >= 6u)
    return false;

  const physx::PxVec3 &halfExtent = geometry.rigidBoxHalfExtent;
  const physx::PxVec3 &minimum = geometry.rigidBoxTriangleCoreMinimumLocal;
  const physx::PxVec3 &maximum = geometry.rigidBoxTriangleCoreMaximumLocal;
  if (halfExtent.x <= 0.0f || halfExtent.y <= 0.0f ||
      halfExtent.z <= 0.0f || minimum.x > maximum.x ||
      minimum.y > maximum.y || minimum.z > maximum.z)
    return false;

  switch (face) {
  case 0u:
    distance = halfExtent.x - minimum.x;
    break;
  case 1u:
    distance = maximum.x + halfExtent.x;
    break;
  case 2u:
    distance = halfExtent.y - minimum.y;
    break;
  case 3u:
    distance = maximum.y + halfExtent.y;
    break;
  case 4u:
    distance = halfExtent.z - minimum.z;
    break;
  default:
    distance = maximum.z + halfExtent.z;
    break;
  }
  if (!physx::PxIsFinite(distance) || distance < 0.0f)
    return false;
  // The detector's existing one-face certificate adds the same small
  // current-pose DCD clearance.  Retain it here so an exactly coplanar
  // triangle is pushed strictly outside instead of cycling on inclusive SAT.
  distance += physx::PxMax(1.0e-5f, geometry.margin * 0.02f);
  return physx::PxIsFinite(distance) && distance > 0.0f;
}

static physx::PxVec3 getRigidBoxTriangleCoreExitNormalLocal(
    physx::PxU32 face) {
  switch (face) {
  case 0u:
    return physx::PxVec3(1.0f, 0.0f, 0.0f);
  case 1u:
    return physx::PxVec3(-1.0f, 0.0f, 0.0f);
  case 2u:
    return physx::PxVec3(0.0f, 1.0f, 0.0f);
  case 3u:
    return physx::PxVec3(0.0f, -1.0f, 0.0f);
  case 4u:
    return physx::PxVec3(0.0f, 0.0f, 1.0f);
  default:
    return physx::PxVec3(0.0f, 0.0f, -1.0f);
  }
}

// Current-pose validity test for a complete triangle-core certificate.  The
// prepared core row stores a centroid for AL, but the OGC scheduler owns the
// three independently embedded collision vertices.  A cached certificate is
// therefore a refresh hint only; it becomes a terminal overlap only when its
// complete support plane is actually crossed at the current solve pose.
static bool getCurrentRigidBoxTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody *body,
    const AvbdSoftParticle *particles, physx::PxU32 numParticles,
    physx::PxReal &faceGap,
    physx::PxU32 movedParticleIndex = PX_MAX_U32,
    const physx::PxVec3 &movedParticleDisplacement =
        physx::PxVec3(0.0f)) {
  if (!particles || !geometry.hasRigidBoxSdf ||
      !geometry.hasRigidBoxTriangleCoreExit ||
      !geometry.rigidBoxPose.isValid() ||
      !geometry.rigidBoxHalfExtent.isFinite() ||
      !geometry.rigidBoxTriangleCoreExitNormalLocal.isFinite())
    return false;

  physx::PxTransform boxToWorld = geometry.rigidBoxPose;
  if (geometry.hasRigidBodyTarget()) {
    if (!body || !body->position.isFinite() || !body->rotation.isFinite())
      return false;
    boxToWorld = physx::PxTransform(body->position, body->rotation) *
        geometry.rigidBoxPose;
  } else if (!geometry.hasWorldStaticTarget()) {
    return false;
  }
  if (!boxToWorld.isValid())
    return false;

  const physx::PxVec3 rawNormalLocal =
      geometry.rigidBoxTriangleCoreExitNormalLocal;
  const physx::PxReal localNormalLengthSq =
      rawNormalLocal.magnitudeSquared();
  const physx::PxVec3 halfExtent = geometry.rigidBoxHalfExtent;
  if (!physx::PxIsFinite(localNormalLengthSq) ||
      localNormalLengthSq <= 1.0e-12f || halfExtent.x <= 0.0f ||
      halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;
  const physx::PxVec3 normalLocal =
      rawNormalLocal * physx::PxRecipSqrt(localNormalLengthSq);
  const physx::PxVec3 rawNormalWorld = boxToWorld.q.rotate(normalLocal);
  const physx::PxReal worldNormalLengthSq =
      rawNormalWorld.magnitudeSquared();
  if (!rawNormalWorld.isFinite() ||
      !physx::PxIsFinite(worldNormalLengthSq) ||
      worldNormalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalWorld =
      rawNormalWorld * physx::PxRecipSqrt(worldNormalLengthSq);
  const physx::PxVec3 surfaceLocal(normalLocal.x * halfExtent.x,
                                    normalLocal.y * halfExtent.y,
                                    normalLocal.z * halfExtent.z);
  const physx::PxVec3 surfaceWorld =
      boxToWorld.transform(surfaceLocal);

  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
    const AvbdWeightedContactPoint &mapping =
        geometry.rigidBoxTriangleCorePoints[vertex];
    if (mapping.count == 0 || mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    physx::PxVec3 point(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 support = 0; support < mapping.count; ++support) {
      const physx::PxU32 particleIndex = mapping.particleIndices[support];
      const physx::PxReal weight = mapping.weights[support];
      if (particleIndex >= numParticles || !physx::PxIsFinite(weight) ||
          !particles[particleIndex].position.isFinite())
        return false;
      point += particles[particleIndex].position * weight;
      if (particleIndex == movedParticleIndex)
        point += movedParticleDisplacement * weight;
      weightSum += weight;
    }
    if (!point.isFinite() || !physx::PxIsFinite(weightSum) ||
        physx::PxAbs(weightSum - 1.0f) > 1.0e-3f)
      return false;
    const physx::PxReal gap = (point - surfaceWorld).dot(normalWorld);
    if (!physx::PxIsFinite(gap))
      return false;
    faceGap = physx::PxMin(faceGap, gap);
  }
  return physx::PxIsFinite(faceGap);
}

// Every post-AL position owner must consume the same pair trust region as the
// material and rigid GS updates.  In particular, a world-static recovery may
// not push a lower soft support through a dynamic box merely because its own
// target is a pedestal.  This helper applies the immutable Scene contact CSR
// to one proposed soft-particle displacement and returns the largest common
// current-pose DCD-safe fraction.  It never evaluates a swept pose or advances
// time.
static physx::PxReal limitPostAlSoftParticleOgcCandidate(
    const AvbdSoftIslandExecutionPlan *plan,
    const AvbdSoftContact *ogcContacts, physx::PxU32 numOgcContacts,
    AvbdSolverBody *rigidBodies, physx::PxU32 numRigidBodies,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 particleIndex,
    const physx::PxVec3 &candidateDisplacement) {
  if (!plan || !ogcContacts || !rigidBodies || !softParticles ||
      particleIndex >= numSoftParticles ||
      !candidateDisplacement.isFinite() ||
      !plan->isComplete(numSoftParticles) ||
      !plan->hasMixedOgcPairPlan(numOgcContacts))
    return 1.0f;

  const physx::PxU32 begin = plan->contactStarts[particleIndex];
  const physx::PxU32 end = plan->contactStarts[particleIndex + 1u];
  if (begin > end || end > plan->numContactRefs)
    return 0.0f;
  const physx::PxReal tolerance = 1.0e-6f;
  physx::PxReal alpha = 1.0f;
  auto consumeGap = [&](AvbdOgcPairState &pair, physx::PxReal currentGap,
                        physx::PxReal candidateGap) {
    if (!physx::PxIsFinite(currentGap) ||
        !physx::PxIsFinite(candidateGap) ||
        candidateGap >= currentGap - tolerance)
      return;
    physx::PxReal localAlpha = 0.0f;
    if (currentGap > tolerance) {
      const physx::PxReal denominator = currentGap - candidateGap;
      if (denominator <= tolerance)
        return;
      localAlpha = physx::PxClamp(
          (currentGap - tolerance) / denominator, 0.0f, 1.0f);
    }
    if (localAlpha < 1.0f - 1.0e-6f) {
      pair.refreshRequested = true;
      pair.remainingSafeDisplacement = 0.0f;
      const physx::PxReal admittedGap =
          currentGap + (candidateGap - currentGap) * localAlpha;
      pair.minimumGap = physx::PxMin(pair.minimumGap, admittedGap);
      pair.accumulatedRelativeDisplacement = physx::PxMax(
          pair.accumulatedRelativeDisplacement,
          candidateDisplacement.magnitude() * localAlpha);
    }
    alpha = physx::PxMin(alpha, localAlpha);
  };

  for (physx::PxU32 refIndex = begin; refIndex < end; ++refIndex) {
    const AvbdSoftContactParticleRef &ref = plan->contactRefs[refIndex];
    if (ref.contactIndex >= numOgcContacts ||
        !physx::PxIsFinite(ref.jacobianScale))
      continue;
    const physx::PxU32 pairIndex = plan->ogcPairIndices[ref.contactIndex];
    if (pairIndex >= plan->numOgcPairStates)
      continue;
    AvbdOgcPairState &pair = plan->ogcPairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        ogcContacts[ref.contactIndex].geometry;
    if (!pair.active || !geometry.hasRigidBodyTarget() ||
        geometry.targetIndex >= numRigidBodies ||
        geometry.targetIndex != pair.targetBodyIndex ||
        geometry.queryBodyIndex != pair.sourceBodyIndex ||
        geometry.source.primitiveKey != pair.primitiveKey)
      continue;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!queryPoint.isFinite())
      continue;
    physx::PxVec3 normal(0.0f), rigidOffset(0.0f);
    physx::PxReal currentGap = 0.0f;
    physx::PxReal candidateGap = 0.0f;
    if (!getCurrentDynamicSoftRigidContactGeometry(
            geometry, rigidBodies[geometry.targetIndex], queryPoint, normal,
            rigidOffset, currentGap) ||
        !getCurrentDynamicSoftRigidContactGeometry(
            geometry, rigidBodies[geometry.targetIndex],
            queryPoint + candidateDisplacement * ref.jacobianScale, normal,
            rigidOffset, candidateGap))
      continue;
    consumeGap(pair, currentGap, candidateGap);
  }

  if (plan->hasTriangleCoreSafetyPlan(numSoftParticles)) {
    const physx::PxU32 coreBegin =
        plan->triangleCoreSafetyStarts[particleIndex];
    const physx::PxU32 coreEnd =
        plan->triangleCoreSafetyStarts[particleIndex + 1u];
    if (coreBegin > coreEnd || coreEnd > plan->numTriangleCoreSafetyRefs)
      return 0.0f;
    for (physx::PxU32 refIndex = coreBegin; refIndex < coreEnd; ++refIndex) {
      const AvbdSoftContactParticleRef &ref =
          plan->triangleCoreSafetyRefs[refIndex];
      if (ref.contactIndex >= numOgcContacts)
        continue;
      const physx::PxU32 pairIndex = plan->ogcPairIndices[ref.contactIndex];
      if (pairIndex >= plan->numOgcPairStates)
        continue;
      AvbdOgcPairState &pair = plan->ogcPairStates[pairIndex];
      const AvbdSoftContactGeometry &geometry =
          ogcContacts[ref.contactIndex].geometry;
      if (!pair.active || !geometry.hasRigidBodyTarget() ||
          geometry.targetIndex >= numRigidBodies ||
          geometry.targetIndex != pair.targetBodyIndex ||
          geometry.queryBodyIndex != pair.sourceBodyIndex ||
          geometry.source.primitiveKey != pair.primitiveKey)
        continue;
      physx::PxReal currentGap = 0.0f;
      physx::PxReal candidateGap = 0.0f;
      if (!getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, &rigidBodies[geometry.targetIndex], softParticles,
              numSoftParticles, currentGap) ||
          !getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, &rigidBodies[geometry.targetIndex], softParticles,
              numSoftParticles, candidateGap, particleIndex,
              candidateDisplacement))
        continue;
      consumeGap(pair, currentGap, candidateGap);
    }
  }
  return physx::PxIsFinite(alpha) ? alpha : 0.0f;
}

// A post-AL correction for one soft/rigid pair may move the shared rigid into
// another soft body.  Treat every other pair targeting that rigid as an active
// OGC trust-region constraint and clip only the rigid endpoint of the proposed
// correction.  The current pair's soft endpoint remains free to deform, which
// is the block-GS response of an opposing contact rather than a whole-island
// position rollback.  All queries are exact current-pose DCD queries; no
// previous pose, swept shape, or time subdivision participates.
static physx::PxReal limitPostAlRigidOgcCandidate(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 targetBodyIndex, physx::PxU32 currentSourceBodyIndex,
    physx::PxU64 currentPrimitiveKey, const AvbdSolverBody &currentBody,
    const AvbdSolverBody &candidateBody) {
  if (!contacts || !softParticles || numContacts == 0u ||
      !currentBody.position.isFinite() || !currentBody.rotation.isFinite() ||
      !candidateBody.position.isFinite() ||
      !candidateBody.rotation.isFinite())
    return 1.0f;

  const physx::PxReal tolerance = 1.0e-6f;
  physx::PxReal alpha = 1.0f;
  auto interpolateBody = [&](physx::PxReal value) {
    AvbdSolverBody result = currentBody;
    result.position = currentBody.position +
        (candidateBody.position - currentBody.position) * value;
    result.rotation = physx::PxSlerp(
        value, currentBody.rotation, candidateBody.rotation);
    result.projectLockedPose(currentBody.position, currentBody.rotation);
    return result;
  };

  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdSoftContactGeometry &geometry =
        contacts[contactIndex].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() ||
        geometry.targetIndex != targetBodyIndex ||
        (geometry.queryBodyIndex == currentSourceBodyIndex &&
         geometry.source.primitiveKey == currentPrimitiveKey))
      continue;

    const bool triangleCore = geometry.hasRigidBoxTriangleCoreExit;
    const physx::PxVec3 queryPoint = triangleCore
        ? physx::PxVec3(0.0f)
        : avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!triangleCore && !queryPoint.isFinite())
      continue;

    auto evaluateGap = [&](const AvbdSolverBody &body,
                           physx::PxReal &gap) {
      if (triangleCore)
        return getCurrentRigidBoxTriangleCoreFaceGap(
            geometry, &body, softParticles, numSoftParticles, gap);
      physx::PxVec3 normal(0.0f), worldOffset(0.0f);
      return getCurrentDynamicSoftRigidContactGeometry(
          geometry, body, queryPoint, normal, worldOffset, gap);
    };

    physx::PxReal currentGap = 0.0f;
    physx::PxReal candidateGap = 0.0f;
    const AvbdSolverBody endpointBody = interpolateBody(alpha);
    if (!evaluateGap(currentBody, currentGap) ||
        !evaluateGap(endpointBody, candidateGap) ||
        !physx::PxIsFinite(currentGap) ||
        !physx::PxIsFinite(candidateGap) ||
        candidateGap >= currentGap - tolerance)
      continue;

    // A pair already on or behind its boundary may only stay neutral or
    // separate.  An inward candidate has no admissible positive fraction.
    if (currentGap <= tolerance) {
      alpha = 0.0f;
      break;
    }

    physx::PxReal lo = 0.0f;
    physx::PxReal hi = alpha;
    for (physx::PxU32 iteration = 0; iteration < 10u; ++iteration) {
      const physx::PxReal mid = 0.5f * (lo + hi);
      const AvbdSolverBody midBody = interpolateBody(mid);
      physx::PxReal midGap = 0.0f;
      if (evaluateGap(midBody, midGap) && physx::PxIsFinite(midGap) &&
          midGap >= tolerance)
        lo = mid;
      else
        hi = mid;
    }
    alpha = lo;
    if (alpha <= 0.0f)
      break;
  }
  return physx::PxIsFinite(alpha) ? alpha : 0.0f;
}

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

static bool bodyTouchesDeformableAnchor(AvbdContactConstraint *contacts,
                                        physx::PxU32 numContacts,
                                        physx::PxU32 bodyIndex,
                                        const AvbdBodyConstraintMap *contactMap = nullptr) {
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
static void applyAvbdContactTargetVelocity(
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

static bool isRigidDeepBodyStaticRecoverySplitSupported(
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
             -kBodyStaticNearSurface *
                 physx::PxMax(lengthScale, physx::PxReal(1e-6f));
}

static bool isRigidFiniteBodyStaticMaterialSplitSupported(
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
      -kBodyStaticNearSurface *
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
static void clampBodyStaticInelasticNormalVelocities(
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

static void recordBodyStaticNormalAlOwnership(
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

// Jolt-style terminal collision preparation: build a compact, private proxy
// batch once from the immutable Scene execution plan, then collide that batch
// against final-pose boxes.  This is a same-time DCD refresh, not a swept
// query or an extra simulation substep.  Keeping it here also means the task
// graph never has to call back into ScScene while an island owns its writes.
static bool avbdBuildTerminalCurrentPoseBoxContacts(
    const AvbdSoftIslandExecutionPlan *plan, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, const AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies,
    const physx::PxU8 *sourceBodyMask,
    physx::PxU32 numSourceBodyMask,
    const physx::PxU8 *rigidBoxMask,
    physx::PxU32 numRigidBoxMask,
    physx::PxArray<AvbdSoftParticle> &proxyParticles,
    physx::PxArray<AvbdSoftBody> &collisionBodies,
    physx::PxArray<AvbdRigidBox> &boxes,
    physx::PxArray<AvbdSoftContact> &contacts) {
  if (!plan || !plan->hasTerminalCurrentPoseBoxPlan(numSoftParticles) ||
      !softParticles || !softBodies || numSoftBodies == 0 ||
      !sourceBodyMask ||
      numSourceBodyMask != plan->numTerminalCollisionBodies ||
      !rigidBoxMask || numRigidBoxMask != plan->numTerminalRigidBoxes)
    return false;

  // This terminal path is specifically the non-CCD OGC owner.  A mixed
  // swept island keeps its existing authoritative stream and never quietly
  // consumes a current-only repair as if it were a CCD result.
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
    if (softBodies[bodyIndex].compiled.speculativeCCDEnabled)
      return false;

  proxyParticles.resize(plan->numTerminalCollisionVertexMappings);
  for (physx::PxU32 proxyIndex = 0;
       proxyIndex < plan->numTerminalCollisionVertexMappings; ++proxyIndex) {
    const AvbdWeightedContactPoint &mapping =
        plan->terminalCollisionVertexMappings[proxyIndex];
    if (mapping.count == 0)
      return false;
    AvbdSoftParticle proxy;
    proxy.position = physx::PxVec3(0.0f);
    proxy.velocity = physx::PxVec3(0.0f);
    bool dynamic = false;
    for (physx::PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint) {
      const physx::PxU32 particleIndex = mapping.particleIndices[endpoint];
      const physx::PxReal weight = mapping.weights[endpoint];
      if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight))
        return false;
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      if (!particle.position.isFinite())
        return false;
      proxy.position += particle.position * weight;
      proxy.velocity += particle.velocity * weight;
      dynamic = dynamic || particle.invMass > 0.0f;
    }
    proxy.initialPosition = proxy.position;
    proxy.predictedPosition = proxy.position;
    proxy.outerPosition = proxy.position;
    proxy.prevVelocity = proxy.velocity;
    proxy.invMass = dynamic ? 1.0f : 0.0f;
    proxy.mass = dynamic ? 1.0f : 0.0f;
    proxy.gravityScale = 1.0f;
    proxyParticles[proxyIndex] = proxy;
  }

  // A terminal epoch is a pair-owned narrowphase phase, not a second complete
  // island collision pass.  The OGC pair state has already identified exactly
  // which source soft bodies and box targets exhausted their current-pose
  // safety allowance.  Retain the original proxy indexing, but pass only
  // those immutable body/box descriptors to the detector.  This is the same
  // persistent-manifold work partition used by CPU XPBD solvers: contact
  // persistence narrows the expensive refresh without ever reusing a stale
  // contact row as geometry.
  collisionBodies.clear();
  for (physx::PxU32 bodyIndex = 0;
       bodyIndex < plan->numTerminalCollisionBodies; ++bodyIndex) {
    if (sourceBodyMask[bodyIndex] != 0u)
      collisionBodies.pushBack(plan->terminalCollisionBodies[bodyIndex]);
  }
  if (collisionBodies.empty())
    return false;

  boxes.clear();
  for (physx::PxU32 boxIndex = 0; boxIndex < plan->numTerminalRigidBoxes;
       ++boxIndex) {
    if (rigidBoxMask[boxIndex] == 0u)
      continue;
    AvbdRigidBox box = plan->terminalRigidBoxes[boxIndex];
    if (box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY) {
      if (box.targetIndex >= numBodies)
        return false;
      const physx::PxTransform bodyToWorld(
          bodies[box.targetIndex].position, bodies[box.targetIndex].rotation);
      const physx::PxTransform shapeToWorld =
          bodyToWorld * box.shapeToRigidBody;
      if (!shapeToWorld.isValid())
        return false;
      box.center = shapeToWorld.p;
      box.rotation = shapeToWorld.q;
    }
    // A terminal epoch has one pose only.  Pin previous to current so future
    // detector changes cannot accidentally discover a swept segment here.
    box.previousCenter = box.center;
    box.previousRotation = box.rotation;
    boxes.pushBack(box);
  }
  if (boxes.empty())
    return false;

  contacts.clear();
  avbdDetectSoftRigidSDF(
      proxyParticles.begin(), proxyParticles.size(), boxes.begin(),
      boxes.size(), contacts, plan->terminalContactRadius, nullptr, 0,
      collisionBodies.begin(), collisionBodies.size());
  avbdDetectSoftRigidOGCFeatures(
      proxyParticles.begin(), proxyParticles.size(), boxes.begin(),
      boxes.size(), collisionBodies.begin(), collisionBodies.size(), contacts,
      plan->terminalContactRadius);

  // The detector operates on collision-proxy vertices.  Expand only the
  // rigid-contact query support into simulation DOFs; no AL state is copied
  // or transferred because these contacts live solely in terminal scratch.
  for (physx::PxU32 contactIndex = 0; contactIndex < contacts.size();
       ++contactIndex) {
    AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
    const physx::PxU32 collisionFeatureParticle = geometry.particleIdx;
    physx::PxU32 collisionBodyIndex = PX_MAX_U32;
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < plan->numTerminalCollisionBodies; ++bodyIndex) {
      const AvbdSoftBodyCompiledData &compiled =
          plan->terminalCollisionBodies[bodyIndex].compiled;
      if (collisionFeatureParticle >= compiled.particleStart &&
          collisionFeatureParticle - compiled.particleStart <
              compiled.particleCount) {
        collisionBodyIndex = bodyIndex;
        break;
      }
    }
    if (collisionBodyIndex == PX_MAX_U32)
      return false;

    AvbdWeightedContactPoint expanded;
    physx::PxU32 queryCount = 1u;
    if (geometry.hasBarycentricQueryPoint()) {
      queryCount = 0u;
      while (queryCount < 3u &&
             geometry.queryParticleIndices[queryCount] != PX_MAX_U32)
        ++queryCount;
      if (queryCount == 0u)
        return false;
    }
    for (physx::PxU32 queryIndex = 0; queryIndex < queryCount; ++queryIndex) {
      const physx::PxU32 proxyIndex = geometry.hasBarycentricQueryPoint()
          ? geometry.queryParticleIndices[queryIndex]
          : collisionFeatureParticle;
      const physx::PxReal queryWeight = geometry.hasBarycentricQueryPoint()
          ? geometry.queryWeights[queryIndex]
          : 1.0f;
      if (proxyIndex >= plan->numTerminalCollisionVertexMappings ||
          !physx::PxIsFinite(queryWeight))
        return false;
      const AvbdWeightedContactPoint &mapping =
          plan->terminalCollisionVertexMappings[proxyIndex];
      for (physx::PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint)
        if (!expanded.appendMerged(mapping.particleIndices[endpoint],
                                   queryWeight * mapping.weights[endpoint]))
          return false;
    }
    expanded.removeNearZero();
    if (expanded.count == 0)
      return false;
    for (physx::PxU32 endpoint = 0; endpoint < expanded.count; ++endpoint)
      if (expanded.particleIndices[endpoint] >= numSoftParticles)
        return false;
    geometry.collisionFeatureParticleIdx = collisionFeatureParticle;
    geometry.queryBodyIndex = collisionBodyIndex;
    geometry.queryPoint = expanded;
    geometry.particleIdx = expanded.particleIndices[0];

    // A terminal triangle-core manifold must retain the three independently
    // embedded collision vertices, not merely its centroid AL query.  The
    // terminal pass is built from final positions, so these supports are the
    // authoritative local deformation endpoints for the same-time OGC repair.
    if (geometry.hasRigidBoxTriangleCoreExit) {
      for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
        const physx::PxU32 proxyIndex =
            geometry.rigidBoxTriangleCoreCollisionParticleIndices[vertex];
        if (proxyIndex >= plan->numTerminalCollisionVertexMappings)
          return false;
        const AvbdWeightedContactPoint &vertexMapping =
            plan->terminalCollisionVertexMappings[proxyIndex];
        if (vertexMapping.count == 0 ||
            vertexMapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
          return false;
        geometry.rigidBoxTriangleCorePoints[vertex] = vertexMapping;
        for (physx::PxU32 endpoint = 0;
             endpoint < vertexMapping.count; ++endpoint) {
          if (vertexMapping.particleIndices[endpoint] >= numSoftParticles ||
              !physx::PxIsFinite(vertexMapping.weights[endpoint]))
            return false;
        }
      }
    }
  }
  return true;
}

// Pair-state persistence avoids rebuilding a full manifold for resting pairs,
// but it cannot be the only terminal admission rule: the final material or
// static projection can create a *new* soft/box overlap that has no source
// row in the prediction epoch.  Refit the immutable collision proxy into a
// conservative AABB and mark only those final-pose body/box pairs that can
// enter the OGC shell.  This is the broadphase half of a same-time DCD epoch;
// it neither advances time nor uses a previous pose/swept query.
static bool avbdMarkTerminalCurrentPoseBroadphasePairs(
    const AvbdSoftIslandExecutionPlan *plan, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, physx::PxU8 *sourceBodyMask,
    physx::PxU32 numSourceBodyMask, physx::PxU8 *rigidBoxMask,
    physx::PxU32 numRigidBoxMask) {
  if (!plan || !plan->hasTerminalCurrentPoseBoxPlan(numSoftParticles) ||
      !bodies || !softParticles || !sourceBodyMask || !rigidBoxMask ||
      numSourceBodyMask != plan->numTerminalCollisionBodies ||
      numRigidBoxMask != plan->numTerminalRigidBoxes)
    return false;

  const physx::PxReal shellRadius =
      physx::PxMax(plan->terminalContactRadius, 1.0e-6f);
  bool markedAny = false;
  for (physx::PxU32 sourceBodyIndex = 0;
       sourceBodyIndex < plan->numTerminalCollisionBodies;
       ++sourceBodyIndex) {
    const AvbdSoftBodyCompiledData &compiled =
        plan->terminalCollisionBodies[sourceBodyIndex].compiled;
    if (compiled.particleStart >
            plan->numTerminalCollisionVertexMappings ||
        compiled.particleCount >
            plan->numTerminalCollisionVertexMappings -
                compiled.particleStart)
      return false;

    physx::PxVec3 minimum(PX_MAX_F32);
    physx::PxVec3 maximum(-PX_MAX_F32);
    bool hasDynamicProxyVertex = false;
    for (physx::PxU32 localVertex = 0;
         localVertex < compiled.particleCount; ++localVertex) {
      const physx::PxU32 proxyIndex =
          compiled.particleStart + localVertex;
      const AvbdWeightedContactPoint &mapping =
          plan->terminalCollisionVertexMappings[proxyIndex];
      if (mapping.count == 0)
        return false;
      physx::PxVec3 point(0.0f);
      bool dynamic = false;
      for (physx::PxU32 endpoint = 0; endpoint < mapping.count;
           ++endpoint) {
        const physx::PxU32 particleIndex =
            mapping.particleIndices[endpoint];
        const physx::PxReal weight = mapping.weights[endpoint];
        if (particleIndex >= numSoftParticles ||
            !physx::PxIsFinite(weight) ||
            !softParticles[particleIndex].position.isFinite())
          return false;
        point += softParticles[particleIndex].position * weight;
        dynamic = dynamic || softParticles[particleIndex].invMass > 0.0f;
      }
      if (!point.isFinite())
        return false;
      minimum.x = physx::PxMin(minimum.x, point.x);
      minimum.y = physx::PxMin(minimum.y, point.y);
      minimum.z = physx::PxMin(minimum.z, point.z);
      maximum.x = physx::PxMax(maximum.x, point.x);
      maximum.y = physx::PxMax(maximum.y, point.y);
      maximum.z = physx::PxMax(maximum.z, point.z);
      hasDynamicProxyVertex = hasDynamicProxyVertex || dynamic;
    }
    if (!hasDynamicProxyVertex || !minimum.isFinite() ||
        !maximum.isFinite())
      continue;

    for (physx::PxU32 boxIndex = 0;
         boxIndex < plan->numTerminalRigidBoxes; ++boxIndex) {
      const AvbdRigidBox &sourceBox = plan->terminalRigidBoxes[boxIndex];
      if (!sourceBox.halfExtent.isFinite() ||
          sourceBox.halfExtent.x <= 0.0f ||
          sourceBox.halfExtent.y <= 0.0f ||
          sourceBox.halfExtent.z <= 0.0f)
        return false;
      physx::PxVec3 center = sourceBox.center;
      if (sourceBox.targetKind ==
          AvbdSoftContactTargetKind::eRIGID_BODY) {
        if (sourceBox.targetIndex >= numBodies)
          return false;
        const physx::PxTransform shapeToWorld(
            bodies[sourceBox.targetIndex].position,
            bodies[sourceBox.targetIndex].rotation);
        const physx::PxTransform boxToWorld =
            shapeToWorld * sourceBox.shapeToRigidBody;
        if (!boxToWorld.isValid())
          return false;
        center = boxToWorld.p;
      }
      if (!center.isFinite())
        return false;
      // A sphere AABB is deliberately conservative for a rotated OBB.  It
      // only schedules the exact box SDF/feature detector below; it never
      // becomes a contact or a solver constraint itself.
      const physx::PxReal extent =
          sourceBox.halfExtent.magnitude() + shellRadius;
      if (!physx::PxIsFinite(extent) || extent <= 0.0f)
        return false;
      const physx::PxVec3 boxMinimum = center - physx::PxVec3(extent);
      const physx::PxVec3 boxMaximum = center + physx::PxVec3(extent);
      if (maximum.x < boxMinimum.x || minimum.x > boxMaximum.x ||
          maximum.y < boxMinimum.y || minimum.y > boxMaximum.y ||
          maximum.z < boxMinimum.z || minimum.z > boxMaximum.z)
        continue;
      sourceBodyMask[sourceBodyIndex] = 1u;
      rigidBoxMask[boxIndex] = 1u;
      markedAny = true;
    }
  }
  return markedAny;
}

void AvbdSolver::postAlStages(
    physx::PxReal dt, physx::PxReal invDt, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
    const physx::PxVec3 &gravity,
    bool hasBodyStaticContact, bool deformableFastImpactIsland,
    const physx::PxArray<bool> &touchingBodyStatic,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    bool allowRigidDeepPoseRecoverySplit,
    bool allowRigidFiniteMaterialPoseSplit,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    const physx::PxArray<bool> &touchesKinematicShell,
    const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
    const physx::PxArray<bool> *positionOwnedAngularBodies,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    bool hasJointConstraints, bool skipBodyStaticFriction,
    bool applyVelocityDamping,
    AvbdSoftParticle *softParticlesForVel,
    physx::PxU32 numSoftParticlesForVel,
    AvbdSolverStats &stats,
    const AvbdPostAlContactWorkPlan *postAlContactWork,
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan) {
  PX_PROFILE_ZONE("AVBD.postAlStages", 0);

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
            bodies, numBodies, contacts, numContacts, contactMap, i,
            mConfig.lengthScale);
    splitRigidFiniteMaterialPose[i] =
        allowRigidFiniteMaterialPoseSplit &&
        isRigidFiniteBodyStaticMaterialSplitSupported(
            bodies, numBodies, contacts, numContacts, contactMap, i,
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
          bodyTouchesDeformableAnchor(contacts, numContacts, bi, contactMap))
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
  // A world-static soft row has no solver-body endpoint, so body/static and
  // kinematic-shell recovery above cannot see it.  World-static and dynamic
  // soft/rigid recovery share soft support vertices, so solve them as one
  // deterministic Gauss--Seidel stage rather than completing all of one kind
  // before starting the other.  In particular, a dynamic correction can move
  // a support back into a pedestal, and an independent one-shot pass would
  // leave that last overlap unresolved until the next frame.
  physx::PxArray<physx::PxU8> recoveredWorldStaticSoftContacts;
  physx::PxArray<physx::PxU8> recoveredDynamicSoftRigidContacts;

  // Private terminal-epoch storage.  It never aliases shellContacts: those
  // retain the solver's AL state, whereas this compact batch is rebuilt from
  // the final pose and is consumed only by geometric post-stabilization.
  physx::PxArray<AvbdSoftParticle> terminalProxyParticles;
  physx::PxArray<AvbdSoftBody> terminalCollisionBodies;
  physx::PxArray<AvbdRigidBox> terminalRigidBoxes;
  physx::PxArray<AvbdSoftContact> terminalContacts;
  physx::PxArray<AvbdOgcPairState> terminalOgcPairStates;
  physx::PxArray<physx::PxU32> terminalOgcPairIndices;
  physx::PxArray<physx::PxU32> terminalOgcPairParentIndices;
  physx::PxArray<physx::PxU8> terminalRecoveredWorldStaticContacts;
  physx::PxArray<physx::PxU8> terminalRecoveredDynamicContacts;
  physx::PxArray<physx::PxU8> terminalRecoveredWorldStaticBodies;
  physx::PxArray<physx::PxVec3> terminalWorldStaticNormals;
  physx::PxArray<physx::PxVec3> terminalVelocityBasePos;
  physx::PxArray<physx::PxQuat> terminalVelocityBaseRot;
  physx::PxArray<physx::PxU8> terminalSourceBodyMask;
  physx::PxArray<physx::PxU8> terminalRigidBoxMask;
  bool terminalCurrentPoseEpochApplied = false;
  bool terminalCurrentPoseClosureUnresolved = false;
  physx::PxU32 terminalCurrentPoseProjectionPasses = 0u;
  physx::PxU32 terminalCurrentPoseLastContactCount = 0u;
	// The support-level recovery above preserves local compliance.  Its
	// coherent endpoint companion is only admitted for an entirely free soft
	// body, and therefore preserves every tet F while covering a static box
	// triangle-core overlap which individual vertices cannot safely repair.
	physx::PxArray<physx::PxU8> recoveredWorldStaticSoftBodies;
	physx::PxArray<physx::PxVec3> recoveredWorldStaticSoftBodyNormals;
  if (shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0 && softBodiesForRecovery &&
      numSoftBodiesForRecovery > 0) {
    recoveredWorldStaticSoftContacts.resize(numShellContacts);
    recoveredDynamicSoftRigidContacts.resize(numShellContacts);
    for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
      recoveredWorldStaticSoftContacts[sci] = 0u;
      recoveredDynamicSoftRigidContacts[sci] = 0u;
    }
    for (physx::PxU32 recoverySweep = 0; recoverySweep < 8u;
         ++recoverySweep) {
      {
        PX_PROFILE_ZONE("AVBD.worldStaticSoftDepenetration", 0);
        applyWorldStaticSoftNormalDepenetrationSweeps(
            shellParticles, numShellParticles, softBodiesForRecovery,
            numSoftBodiesForRecovery, shellContacts, numShellContacts, 1u,
            &recoveredWorldStaticSoftContacts, &stats,
            terminalSoftExecutionPlan, bodies, numBodies, shellContacts,
            numShellContacts);
      }
      // Preserve the existing dynamic recovery budget (six sweeps) while
      // interleaving each of those sweeps with the eight static sweeps.
      if (recoverySweep < 6u) {
        // The regular dynamic OGC rows have already been consumed by their
        // shared soft-particle / rigid-body Position-AL blocks.  This final
        // projection sees only a true current-pose SDF overlap, never the OGC
        // proximity shell or a swept/CCD candidate.  It remains before
        // postDepenPos so the paired rigid offset is pose-only.
        PX_PROFILE_ZONE("AVBD.dynamicSoftRigidDepenetration", 0);
        applyDynamicSoftRigidNormalDepenetrationSweeps(
            bodies, numBodies, shellParticles, numShellParticles,
            softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
            numShellContacts, 1u, &recoveredDynamicSoftRigidContacts,
            &stats,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->ogcPairStates : nullptr,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->numOgcPairStates : 0u,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->ogcPairIndices : nullptr,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->numOgcPairIndices : 0u,
            /*softComplianceResponseScale=*/1.0f,
            /*projectToCurrentPoseBoundary=*/false,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->ogcPairContactStarts : nullptr,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->numOgcPairContactStarts : 0u,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->ogcPairContactRefs : nullptr,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->numOgcPairContactRefs : 0u);
      }
    }
    // The local row projections above preserve the regular per-support
    // response.  If a deep endpoint has already made that det(F)-limited
    // path unable to separate a full body, use the much narrower coherent
    // paired fallback once it has exhausted those ordinary sweeps.  It stays
    // current-pose/non-CCD and leaves all soft tet F unchanged.
    {
      PX_PROFILE_ZONE("AVBD.dynamicSoftRigidBodyEndpointTranslation", 0);
      applyDynamicSoftRigidBodyEndpointTranslations(
          bodies, numBodies, shellParticles, numShellParticles,
          softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
          numShellContacts, 4u, &recoveredDynamicSoftRigidContacts,
          /*precedingStaticTranslations=*/nullptr,
          /*allowFreshTriangleCoreExit=*/false,
          /*preferLocalTriangleCoreManifold=*/false,
          /*allowCoherentEndpointFallback=*/false, &stats);
    }
    // Dynamic response can be directed towards world-static geometry.  Make
    // world-static the final owner in this stage so a dynamic projection
    // cannot leave a real static overlap at the end of the step.
    {
      PX_PROFILE_ZONE("AVBD.worldStaticTriangleCoreLocalManifold", 0);
      applyWorldStaticTriangleCoreLocalManifold(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, shellContacts, numShellContacts, 2u,
          &stats, terminalSoftExecutionPlan, bodies, numBodies,
          shellContacts, numShellContacts);
      PX_PROFILE_ZONE("AVBD.worldStaticSoftDepenetration", 0);
      applyWorldStaticSoftNormalDepenetrationSweeps(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, shellContacts, numShellContacts, 1u,
          &recoveredWorldStaticSoftContacts, &stats,
          terminalSoftExecutionPlan, bodies, numBodies, shellContacts,
          numShellContacts);
    }
		{
			PX_PROFILE_ZONE("AVBD.worldStaticSoftBodyEndpointTranslation", 0);
			applyWorldStaticSoftBodyEndpointTranslations(
				shellParticles, numShellParticles, softBodiesForRecovery,
				numSoftBodiesForRecovery, shellContacts, numShellContacts,
				&recoveredWorldStaticSoftBodies,
				&recoveredWorldStaticSoftBodyNormals,
				/*recoveryTranslations=*/nullptr,
				/*allowFreshTriangleCoreExit=*/false, &stats);
		}
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

  // Terminal current-pose OGC epoch.  This is intentionally after every
  // friction pose edit and before velocity reconstruction: it closes the
  // single-dt contact pipeline without advancing time, and its geometric
  // correction is excluded from pose-derived rigid velocity below.
  // Schedule final DCD like a narrowphase work phase, not an unconditional
  // per-island tax.  A pair at its DCD boundary (or one whose trust region was
  // consumed) needs a same-time refresh; so does a static/dynamic local
  // projection which may have invalidated the opposing endpoint.
  bool terminalCurrentPoseRefreshNeeded = false;
  if (terminalSoftExecutionPlan) {
    terminalSourceBodyMask.resize(
        terminalSoftExecutionPlan->numTerminalCollisionBodies);
    terminalRigidBoxMask.resize(
        terminalSoftExecutionPlan->numTerminalRigidBoxes);
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < terminalSourceBodyMask.size(); ++bodyIndex)
      terminalSourceBodyMask[bodyIndex] = 0u;
    for (physx::PxU32 boxIndex = 0;
         boxIndex < terminalRigidBoxMask.size(); ++boxIndex)
      terminalRigidBoxMask[boxIndex] = 0u;
  }

  // Keep the terminal DCD work set pair-local.  A contact epoch can have
  // hundreds of rows for one body/shape pair, but all of them map to the same
  // collision proxy and box descriptor; marking the descriptor once is enough
  // to rebuild a fresh manifold.  Matching the primitive key as well as the
  // target identity matters for a rigid actor carrying more than one shape.
  const auto markTerminalCurrentPosePair =
      [&](const AvbdSoftContactGeometry &geometry) {
        if (!terminalSoftExecutionPlan ||
            (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF &&
             geometry.source.type != AvbdSoftContactSource::eGROUND) ||
            geometry.queryBodyIndex >= terminalSourceBodyMask.size())
          return;
        terminalSourceBodyMask[geometry.queryBodyIndex] = 1u;
        for (physx::PxU32 boxIndex = 0;
             boxIndex < terminalSoftExecutionPlan->numTerminalRigidBoxes;
             ++boxIndex) {
          const AvbdRigidBox &box =
              terminalSoftExecutionPlan->terminalRigidBoxes[boxIndex];
          if (box.targetKind == geometry.targetKind &&
              box.targetIndex == geometry.targetIndex &&
              box.primitiveKey == geometry.source.primitiveKey) {
            terminalRigidBoxMask[boxIndex] = 1u;
            break;
          }
        }
      };
  if (terminalSoftExecutionPlan &&
      terminalSoftExecutionPlan->hasMixedOgcPairPlan(numShellContacts)) {
    // Re-query the prepared pair representatives at the final pose before
    // paying for proxy DCD.  An OGC shell admission is deliberately not by
    // itself a dirty marker: most admitted pairs remain separated after the
    // material solve, and rebuilding their complete proxy manifold was the
    // source of the 40+ ms spikes.  A resting zero-gap row remains owned by
    // the persistent pair manifold; only a true overlap or a prepared
    // triangle-core witness requests terminal narrowphase work.
    AvbdOgcPairState *pairStates =
        terminalSoftExecutionPlan->ogcPairStates;
    const physx::PxU32 *pairIndices =
        terminalSoftExecutionPlan->ogcPairIndices;
    const physx::PxU32 numPairStates =
        terminalSoftExecutionPlan->numOgcPairStates;
    const physx::PxReal refreshTolerance = physx::PxMax(
        1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));
    if (pairStates && pairIndices &&
        terminalSoftExecutionPlan->numOgcPairIndices == numShellContacts) {
      for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
        const physx::PxU32 pairIndex = pairIndices[sci];
        if (pairIndex >= numPairStates)
          continue;
        AvbdOgcPairState &pair = pairStates[pairIndex];
        const AvbdSoftContactGeometry &geometry = shellContacts[sci].geometry;
        const bool dynamicRigid =
            geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
            geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies;
        const bool worldStatic =
            (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
             geometry.source.type == AvbdSoftContactSource::eGROUND) &&
            geometry.hasWorldStaticTarget();
        if (!pair.active || (!dynamicRigid && !worldStatic) ||
            geometry.queryBodyIndex != pair.sourceBodyIndex ||
            geometry.targetIndex != pair.targetBodyIndex ||
            geometry.source.primitiveKey != pair.primitiveKey)
          continue;

        const physx::PxVec3 queryPoint =
            avbdGetSoftContactQueryPoint(geometry, shellParticles);
        physx::PxVec3 normal(0.0f), rigidOffset(0.0f);
        physx::PxReal currentGap = 0.0f;
        const bool geometryValid = dynamicRigid
            ? queryPoint.isFinite() &&
                  getCurrentDynamicSoftRigidContactGeometry(
                      geometry, bodies[geometry.targetIndex], queryPoint,
                      normal, rigidOffset, currentGap)
            : queryPoint.isFinite() &&
                  getCurrentWorldStaticSoftContactGeometry(
                      geometry, queryPoint, normal, currentGap);
        if (!geometryValid ||
            !physx::PxIsFinite(currentGap))
          continue;

        pair.minimumGap = physx::PxMin(pair.minimumGap, currentGap);

        // The pair state starts at a discrete, current-pose DCD epoch.  Its
        // representative carries the relative surface vector from that
        // epoch, so consume the pair safety budget with actual relative
        // motion rather than with an arbitrary iteration count.  This is the
        // OGC hand-off for a fast free rigid: even if a cached representative
        // has not yet crossed the surface, a relative displacement larger
        // than the DCD clearance means another final-pose manifold is now
        // required to cover the rest of the collision proxy.
        const physx::PxVec3 currentRelativePoint = dynamicRigid
            ? queryPoint -
                  (bodies[geometry.targetIndex].position + rigidOffset)
            : queryPoint - pair.referenceRelativePoint;
        const physx::PxReal relativeEpochDisplacement =
            dynamicRigid
                ? (currentRelativePoint - pair.referenceRelativePoint).magnitude()
                : 0.0f;
        const physx::PxReal safetyBudget = physx::PxMax(
            pair.safetyGap, refreshTolerance);
        if (dynamicRigid && currentRelativePoint.isFinite() &&
            pair.referenceRelativePoint.isFinite() &&
            physx::PxIsFinite(relativeEpochDisplacement) &&
            relativeEpochDisplacement >= safetyBudget) {
          pair.refreshRequested = true;
          pair.remainingSafeDisplacement = 0.0f;
          pair.accumulatedRelativeDisplacement = physx::PxMax(
              pair.accumulatedRelativeDisplacement, relativeEpochDisplacement);
          markTerminalCurrentPosePair(geometry);
        }
        bool coreOverlap = false;
        if (geometry.hasRigidBoxTriangleCoreExit) {
          physx::PxReal coreGap = 0.0f;
          // A prepared core certificate is immutable detector metadata, not
          // proof that the final pose still overlaps.  Re-evaluate its three
          // expanded vertices before scheduling expensive proxy DCD; otherwise
          // every resting face keeps the terminal closure hot forever.
          coreOverlap = !getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, dynamicRigid ? &bodies[geometry.targetIndex] : nullptr,
              shellParticles,
              numShellParticles, coreGap) ||
              coreGap < -refreshTolerance;
        }
        if (currentGap < -refreshTolerance || coreOverlap) {
          pair.refreshRequested = true;
          markTerminalCurrentPosePair(geometry);
        }
        // `refreshRequested` is raised only when the pair moves beyond the
        // published OGC safety domain (or a true current overlap was found).
        // `admittedAtBoundary` is deliberately *not* a terminal-DCD trigger:
        // it records a valid start-of-solve clip and is consumed later by the
        // inelastic velocity owner.  Treating every valid boundary admission
        // as dirty rebuilt the complete proxy manifold on every resting frame.
        if (pair.refreshRequested)
          markTerminalCurrentPosePair(geometry);
      }
    }
    for (physx::PxU32 pairIndex = 0;
         pairIndex < terminalSoftExecutionPlan->numOgcPairStates;
         ++pairIndex) {
      const AvbdOgcPairState &pair =
          terminalSoftExecutionPlan->ogcPairStates[pairIndex];
      if (pair.active && pair.refreshRequested) {
        terminalCurrentPoseRefreshNeeded = true;
        break;
      }
    }
  }
  for (physx::PxU32 sci = 0; sci < recoveredWorldStaticSoftContacts.size();
       ++sci) {
    if (recoveredWorldStaticSoftContacts[sci] != 0u ||
        recoveredDynamicSoftRigidContacts[sci] != 0u) {
      terminalCurrentPoseRefreshNeeded = true;
      markTerminalCurrentPosePair(shellContacts[sci].geometry);
    }
  }

  // Cover final-pose contacts born after the prediction manifold was built.
  // Existing OgcPairState rows still provide the cheap persistent path above;
  // this conservative AABB admission is only the missing-new-pair fallback.
  // The following detector remains exact current-pose SDF/feature DCD.
  if (terminalSoftExecutionPlan && shellParticles &&
      numShellParticles > 0 &&
      avbdMarkTerminalCurrentPoseBroadphasePairs(
          terminalSoftExecutionPlan, bodies, numBodies, shellParticles,
          numShellParticles, terminalSourceBodyMask.begin(),
          terminalSourceBodyMask.size(), terminalRigidBoxMask.begin(),
          terminalRigidBoxMask.size()))
    terminalCurrentPoseRefreshNeeded = true;

  if (terminalCurrentPoseRefreshNeeded && shellParticles &&
      numShellParticles > 0 && softBodiesForRecovery &&
      numSoftBodiesForRecovery > 0 && terminalSoftExecutionPlan) {
    terminalVelocityBasePos.resize(numBodies);
    terminalVelocityBaseRot.resize(numBodies);
    for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
      terminalVelocityBasePos[bodyIndex] = bodies[bodyIndex].position;
      terminalVelocityBaseRot[bodyIndex] = bodies[bodyIndex].rotation;
    }

    // This is a bounded detect -> project -> verify closure at the *same*
    // t=dt pose.  It is intentionally not a time microstep: no velocity is
    // integrated and the detector never receives a previous pose.  A raw
    // SDF witness can move a different collision triangle into the OBB just
    // as a triangle-core row can, so verify both kinds of true overlap rather
    // than treating core rows as a special-only retry.
    // A two-jaw manifold can alternate ownership through the shared 6DOF
    // rigid: projecting one side may close the other side again.  Run a
    // small, convergence-driven same-time closure and reserve the last pass
    // for verification.  This does not advance time or repeat the material
    // solve; it is the terminal narrowphase/project phase of the one dt OGC
    // epoch.  Never commit merely because a fixed number of projections was
    // exhausted without observing a clean final DCD epoch.
    // A two-sided dynamic manifold sharing a world-static support can need one
    // additional static/dynamic exchange after each jaw has first reached its
    // current-pose boundary.  Keep that nonlinear closure inside this same-time
    // terminal phase instead of committing an unresolved pose for the next
    // frame to eject.  This is a narrow, convergence-driven contact budget;
    // it does not repeat material iterations or advance the simulation clock.
    static const physx::PxU32 kMaxTerminalOgcProjectionPasses = 6u;
    for (physx::PxU32 closurePass = 0;
         closurePass <= kMaxTerminalOgcProjectionPasses; ++closurePass) {
      if (!avbdBuildTerminalCurrentPoseBoxContacts(
              terminalSoftExecutionPlan, bodies, numBodies, shellParticles,
              numShellParticles, softBodiesForRecovery,
              numSoftBodiesForRecovery, terminalSourceBodyMask.begin(),
              terminalSourceBodyMask.size(), terminalRigidBoxMask.begin(),
              terminalRigidBoxMask.size(), terminalProxyParticles,
              terminalCollisionBodies, terminalRigidBoxes, terminalContacts))
        break;
      if (terminalContacts.empty())
        break;
      terminalCurrentPoseLastContactCount = terminalContacts.size();

      const physx::PxReal terminalOverlapTolerance = physx::PxMax(
          1.0e-5f,
          1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));
      bool terminalHasTrueOverlap = false;
      for (physx::PxU32 contactIndex = 0;
           contactIndex < terminalContacts.size(); ++contactIndex) {
        const AvbdSoftContactGeometry &geometry =
            terminalContacts[contactIndex].geometry;
        if (geometry.hasRigidBoxTriangleCoreExit) {
          const AvbdSolverBody *coreBody = nullptr;
          if (geometry.hasRigidBodyTarget()) {
            if (geometry.targetIndex >= numBodies) {
              terminalHasTrueOverlap = true;
              break;
            }
            coreBody = &bodies[geometry.targetIndex];
          }
          physx::PxReal coreGap = 0.0f;
          if (!getCurrentRigidBoxTriangleCoreFaceGap(
                  geometry, coreBody, shellParticles, numShellParticles,
                  coreGap) || coreGap < -terminalOverlapTolerance) {
            if (std::getenv("PHYSX_AVBD_OGC_TERMINAL_TRACE"))
              std::printf(
                  "[AVBD_OGC_TERMINAL_OVERLAP] pass=%u contact=%u "
                  "sourceBody=%u targetKind=%u target=%u primitive=%llu "
                  "core=1 gap=%.9g\n",
                  closurePass, contactIndex, geometry.queryBodyIndex,
                  physx::PxU32(geometry.targetKind), geometry.targetIndex,
                  static_cast<unsigned long long>(geometry.source.primitiveKey),
                  double(coreGap));
            terminalHasTrueOverlap = true;
            break;
          }
        }
        const physx::PxVec3 queryPoint =
            avbdGetSoftContactQueryPoint(geometry, shellParticles);
        if (!queryPoint.isFinite())
          continue;
        physx::PxVec3 normal(0.0f), rigidOffset(0.0f);
        physx::PxReal trueGap = 0.0f;
        const bool valid = geometry.hasRigidBodyTarget()
            ? geometry.targetIndex < numBodies &&
                  getCurrentDynamicSoftRigidContactGeometry(
                      geometry, bodies[geometry.targetIndex], queryPoint,
                      normal, rigidOffset, trueGap)
            : geometry.hasWorldStaticTarget() &&
                  getCurrentWorldStaticSoftContactGeometry(
                      geometry, queryPoint, normal, trueGap);
        if (valid && physx::PxIsFinite(trueGap) &&
            trueGap < -terminalOverlapTolerance) {
          if (std::getenv("PHYSX_AVBD_OGC_TERMINAL_TRACE"))
            std::printf(
                "[AVBD_OGC_TERMINAL_OVERLAP] pass=%u contact=%u "
                "sourceBody=%u targetKind=%u target=%u primitive=%llu "
                "core=0 gap=%.9g\n",
                closurePass, contactIndex, geometry.queryBodyIndex,
                physx::PxU32(geometry.targetKind), geometry.targetIndex,
                static_cast<unsigned long long>(geometry.source.primitiveKey),
                double(trueGap));
          terminalHasTrueOverlap = true;
          break;
        }
      }
      // A narrow phase can return proximity rows for the OGC shell.  They
      // belong to the persistent Position-AL manifold; the terminal owner is
      // only permitted to touch an actual final-pose overlap.
      if (!terminalHasTrueOverlap)
        break;

      // The final pass is verification-only.  Leaving the loop with an
      // overlap here means the local projector did not converge; performing
      // one more unverified correction would only hide the unresolved state
      // until the next frame and recreate the observed penetrate-then-pop
      // behavior.
      if (closurePass == kMaxTerminalOgcProjectionPasses) {
        terminalCurrentPoseClosureUnresolved = true;
        break;
      }

      // Terminal contacts are fresh at t=dt, so their row indices and
      // manifold representatives are scratch-owned.  They retain the shared
      // OgcPairState type and link back to the prediction epoch immediately
      // below; this avoids using stale representatives while preserving one
      // scheduler lifecycle for the pair.
      buildCurrentOgcPairStates(
          terminalContacts.begin(), terminalContacts.size(), shellParticles,
          numShellParticles, bodies, numBodies, terminalOgcPairStates,
          terminalOgcPairIndices);
      linkCurrentOgcPairStates(
          terminalOgcPairStates, terminalSoftExecutionPlan->ogcPairStates,
          terminalSoftExecutionPlan->numOgcPairStates,
          terminalOgcPairParentIndices);

      terminalCurrentPoseEpochApplied = true;
      ++terminalCurrentPoseProjectionPasses;
      terminalRecoveredWorldStaticContacts.resize(terminalContacts.size());
      terminalRecoveredDynamicContacts.resize(terminalContacts.size());
      for (physx::PxU32 contactIndex = 0;
           contactIndex < terminalContacts.size(); ++contactIndex) {
        terminalRecoveredWorldStaticContacts[contactIndex] = 0u;
        terminalRecoveredDynamicContacts[contactIndex] = 0u;
      }
      terminalRecoveredWorldStaticBodies.resize(numSoftBodiesForRecovery);
      terminalWorldStaticNormals.resize(numSoftBodiesForRecovery);
      for (physx::PxU32 bodyIndex = 0;
           bodyIndex < numSoftBodiesForRecovery; ++bodyIndex) {
        terminalRecoveredWorldStaticBodies[bodyIndex] = 0u;
        terminalWorldStaticNormals[bodyIndex] = physx::PxVec3(0.0f);
      }

      PX_PROFILE_ZONE("AVBD.terminalCurrentPoseOgcProject", 0);
      applyWorldStaticTriangleCoreLocalManifold(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, terminalContacts.begin(),
          terminalContacts.size(), 4u, &stats);
      applyWorldStaticSoftNormalDepenetrationSweeps(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, terminalContacts.begin(),
          terminalContacts.size(), 1u, &terminalRecoveredWorldStaticContacts,
          &stats);
      applyDynamicSoftRigidNormalDepenetrationSweeps(
          bodies, numBodies, shellParticles, numShellParticles,
          softBodiesForRecovery, numSoftBodiesForRecovery,
          terminalContacts.begin(), terminalContacts.size(), 1u,
          &terminalRecoveredDynamicContacts, &stats,
          terminalOgcPairStates.empty() ? nullptr : terminalOgcPairStates.begin(),
          terminalOgcPairStates.size(),
          terminalOgcPairIndices.empty() ? nullptr : terminalOgcPairIndices.begin(),
          terminalOgcPairIndices.size(),
          // Terminal DCD has a freshly refit collision proxy and may project
          // exactly to its true boundary.  Bias the positional response
          // toward the deformable support rather than ejecting a light free
          // rigid; this represents the compliant endpoint already solved by
          // AVBD material rows, not a mass change or a new CCD owner.
          /*softComplianceResponseScale=*/4.0f,
          /*projectToCurrentPoseBoundary=*/true);
      applyWorldStaticSoftBodyEndpointTranslations(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, terminalContacts.begin(),
          terminalContacts.size(), &terminalRecoveredWorldStaticBodies,
          &terminalWorldStaticNormals, /*recoveryTranslations=*/nullptr,
          /*allowFreshTriangleCoreExit=*/false, &stats);
      applyDynamicSoftRigidBodyEndpointTranslations(
          bodies, numBodies, shellParticles, numShellParticles,
          softBodiesForRecovery, numSoftBodiesForRecovery,
          // A current-pose triangle manifold is a small coupled PGS block,
          // not a time microstep.  Four local sweeps let both jaws exchange
          // load through the same 6DOF rigid before its t=dt verification.
          terminalContacts.begin(), terminalContacts.size(), 4u,
          &terminalRecoveredDynamicContacts,
          /*precedingStaticTranslations=*/nullptr,
          /*allowFreshTriangleCoreExit=*/true,
          /*preferLocalTriangleCoreManifold=*/true,
          /*allowCoherentEndpointFallback=*/true, &stats,
          terminalOgcPairStates.empty() ? nullptr : terminalOgcPairStates.begin(),
          terminalOgcPairStates.size(),
          terminalOgcPairIndices.empty() ? nullptr : terminalOgcPairIndices.begin(),
          terminalOgcPairIndices.size());

      publishCurrentOgcPairStates(
          terminalOgcPairStates, terminalOgcPairParentIndices,
          terminalSoftExecutionPlan->ogcPairStates,
          terminalSoftExecutionPlan->numOgcPairStates);

    }
  }

  // Opt-in diagnostics for the same-time closure.  Keep this outside the
  // normal telemetry path so production runs pay no formatting cost, while a
  // regression can distinguish "terminal DCD never ran" from "fresh DCD ran
  // but its projector did not converge" without instrumenting Scene state.
  if (std::getenv("PHYSX_AVBD_OGC_TERMINAL_TRACE") &&
      (terminalCurrentPoseEpochApplied ||
       terminalCurrentPoseClosureUnresolved)) {
    std::printf(
        "[AVBD_OGC_TERMINAL] applied=%u projectionPasses=%u "
        "lastContacts=%u unresolved=%u\n",
        terminalCurrentPoseEpochApplied ? 1u : 0u,
        terminalCurrentPoseProjectionPasses,
        terminalCurrentPoseLastContactCount,
        terminalCurrentPoseClosureUnresolved ? 1u : 0u);
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  // Finalize velocity: block motion + friction/motor tangents; exclude depen.
  // These two per-body classifications used to rescan the complete contact
  // array inside the body loop.  Build them once in contact order instead;
  // preserving the first-owner rule while removing the O(numBodies*numContacts)
  // work from the post-AL hot path.
  physx::PxArray<physx::PxU32> physicalContactTangentOwnerIndex(numBodies,
                                                                PX_MAX_U32);
  physx::PxArray<bool> fastNormalImpactByBody(numBodies, false);
  const bool haveSolveStartLinear =
      deformableFastImpactIsland && linearVelAtSolveStart &&
      linearVelAtSolveStart->size() == numBodies;
  if (numContacts > 0) {
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const AvbdContactConstraint &contact = contacts[c];
      if (hasVelocityTangentMaterialOwner(contact) &&
          !hasVelocityBodyStaticFrictionSweepOwner(contact)) {
        const physx::PxU32 bodyA = contact.header.bodyIndexA;
        const physx::PxU32 bodyB = contact.header.bodyIndexB;
        if (bodyA < numBodies &&
            physicalContactTangentOwnerIndex[bodyA] == PX_MAX_U32)
          physicalContactTangentOwnerIndex[bodyA] = c;
        if (bodyB < numBodies &&
            physicalContactTangentOwnerIndex[bodyB] == PX_MAX_U32)
          physicalContactTangentOwnerIndex[bodyB] = c;
      }
      if (haveSolveStartLinear) {
        const physx::PxU32 bodyA = contact.header.bodyIndexA;
        const physx::PxU32 bodyB = contact.header.bodyIndexB;
        if (isBodyVsStaticContact(bodyA, bodyB, numBodies)) {
          const physx::PxU32 dynamicBody =
              bodyA < numBodies ? bodyA : bodyB;
          if (dynamicBody < numBodies &&
              dynamicBody < touchingBodyStatic.size() &&
              touchingBodyStatic[dynamicBody]) {
            const bool dynamicIsA = bodyA == dynamicBody;
            const physx::PxVec3 nd =
                contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
            if (-(*linearVelAtSolveStart)[dynamicBody].dot(nd) >
                kBodyStaticFastImpactSpeed)
              fastNormalImpactByBody[dynamicBody] = true;
          }
        }
      }
    }
  }
  {
    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;
        const physx::PxU32 physicalContactTangentMaterialOwnerIndex =
            physicalContactTangentOwnerIndex[i];
        const bool physicalContactTangentMaterialOwner =
            physicalContactTangentMaterialOwnerIndex != PX_MAX_U32;

        const physx::PxVec3 blockPositionForVelocity =
            (splitRigidDeepPoseRecovery[i] ||
             splitRigidFiniteMaterialPose[i])
                ? bodies[i].inertialPosition
                : postBlockPos[i];
        const physx::PxVec3 vFromBlock =
            (blockPositionForVelocity - bodies[i].prevPosition) * invDt;
        const physx::PxVec3 frictionPositionForVelocity =
            terminalCurrentPoseEpochApplied &&
                    terminalVelocityBasePos.size() == numBodies
                ? terminalVelocityBasePos[i]
                : bodies[i].position;
        const physx::PxVec3 vFromFriction =
            (frictionPositionForVelocity - postDepenPos[i]) * invDt;
        const physx::PxVec3 vFromPose = vFromBlock + vFromFriction;
        const bool fastNormalImpact =
            i < fastNormalImpactByBody.size() && fastNormalImpactByBody[i];
        if (fastNormalImpact) {
          bodies[i].linearVelocity =
              (*linearVelAtSolveStart)[i] * 0.85f + vFromPose * 0.15f;
        } else if (i < touchesKinematicShell.size() && touchesKinematicShell[i] &&
                   shellLinearVelAtSolveStart &&
                   shellLinearVelAtSolveStart->size() == numBodies) {
          bool shellFast = false;
          for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
            const AvbdSoftContactGeometry &geometry =
                shellContacts[sci].geometry;
            if (!geometry.hasRigidBodyTarget() ||
                geometry.targetIndex != i)
              continue;
            const physx::PxReal approach =
                -(*shellLinearVelAtSolveStart)[i].dot(geometry.normal);
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
            (!d6Joints || numD6 == 0) &&
            !(positionOwnedAngularBodies &&
              positionOwnedAngularBodies->size() == numBodies &&
              (*positionOwnedAngularBodies)[i]);
        bool physicalSlerpPositionDrive = false;
        if (d6Joints) {
          for (physx::PxU32 j = 0; j < numD6; ++j) {
            const AvbdD6JointConstraint &joint = d6Joints[j];
            if (joint.header.bodyIndexA != i &&
                joint.header.bodyIndexB != i)
              continue;
            if (hasAvbdJointObjective(
                    joint.objectiveProgram,
                    AvbdJointObjectiveKind::SlerpPositionDrive))
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
          const physx::PxQuat frictionRotationForVelocity =
              terminalCurrentPoseEpochApplied &&
                      terminalVelocityBaseRot.size() == numBodies
                  ? terminalVelocityBaseRot[i]
                  : bodies[i].rotation;
          physx::PxQuat deltaQFr =
              frictionRotationForVelocity * postDepenRot[i].getConjugate();
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
          bodies, numBodies, contacts, numContacts, contactMap,
          linearVelAtSolveStart,
          angularVelAtSolveStart, &splitRigidFiniteMaterialPose, dt,
          mConfig.bounceApproachSpeedThreshold(), mConfig.lengthScale,
          hasJointConstraints,
          mConfig.enableBoundedComponentProductionProbe,
          deformableNormalStageMaskPtr, &stats);
      PX_PROFILE_ZONE("AVBD.contactTargetVelocity", 0);
      applyAvbdContactTargetVelocity(bodies, numBodies, contacts,
                                     numContacts, dt, postAlContactWork);
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
        if (softParticlesForVel[i].invMass > 0.0f) {
          softParticlesForVel[i].updateVelocityFromPosition(invDt);
        }
      }
    }
    if (shellParticles && numShellParticles > 0 && shellContacts &&
        numShellContacts > 0 && softBodiesForRecovery &&
        numSoftBodiesForRecovery > 0) {
      PX_PROFILE_ZONE("AVBD.dynamicSoftRigidTangentVelocity", 0);
      projectDynamicSoftRigidVelocityTangents(
          bodies, numBodies, shellParticles, numShellParticles,
          softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
          numShellContacts, dt);
      PX_PROFILE_ZONE("AVBD.softContactTangentVelocity", 0);
      avbdProjectSoftContactVelocityTangents(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, shellContacts, numShellContacts, dt,
          nullptr);
    }
    if (!recoveredDynamicSoftRigidContacts.empty()) {
      PX_PROFILE_ZONE("AVBD.dynamicSoftRigidBodyEndpointInelasticVel", 0);
      clampDynamicSoftRigidBodyEndpointVelocities(
          bodies, numBodies, shellParticles, numShellParticles,
          softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
          numShellContacts, &recoveredDynamicSoftRigidContacts, &stats);
      PX_PROFILE_ZONE("AVBD.dynamicSoftRigidInelasticVel", 0);
      clampDynamicSoftRigidInelasticNormalVelocities(
          bodies, numBodies, shellParticles, numShellParticles, shellContacts,
          numShellContacts, &recoveredDynamicSoftRigidContacts, &stats);
    }
    // Endpoint projection can move a soft body which is also resting on a
    // world-static target.  Keep the static unilateral clamp last so that
    // this stage cannot reintroduce an inward velocity at the final owner.
    if (!recoveredWorldStaticSoftContacts.empty()) {
      PX_PROFILE_ZONE("AVBD.worldStaticSoftInelasticVel", 0);
      clampWorldStaticSoftInelasticNormalVelocities(
          shellParticles, numShellParticles, shellContacts, numShellContacts,
          &recoveredWorldStaticSoftContacts, &stats);
    }
		if (!recoveredWorldStaticSoftBodies.empty()) {
			// Keep the coherent world-static owner last: dynamic endpoint and
			// support-level clamps may otherwise restore an inward velocity on a
			// body that was translated clear of a static target.
			PX_PROFILE_ZONE("AVBD.worldStaticSoftBodyEndpointInelasticVel", 0);
			clampWorldStaticSoftBodyEndpointVelocities(
				shellParticles, numShellParticles, softBodiesForRecovery,
				numSoftBodiesForRecovery, &recoveredWorldStaticSoftBodies,
				&recoveredWorldStaticSoftBodyNormals, &stats);
		}
		// Terminal rows are fresh current-pose contacts, so their e=0 response
		// must run after position-derived soft velocities exist.  Keep this
		// separate from the persistent AL stream: it owns no warm-start state.
		if (terminalCurrentPoseEpochApplied && !terminalContacts.empty()) {
			PX_PROFILE_ZONE("AVBD.terminalCurrentPoseOgcVelocity", 0);
			if (!terminalRecoveredDynamicContacts.empty()) {
				clampDynamicSoftRigidBodyEndpointVelocities(
					bodies, numBodies, shellParticles, numShellParticles,
					softBodiesForRecovery, numSoftBodiesForRecovery,
					terminalContacts.begin(), terminalContacts.size(),
					&terminalRecoveredDynamicContacts, &stats);
				clampDynamicSoftRigidInelasticNormalVelocities(
					bodies, numBodies, shellParticles, numShellParticles,
					terminalContacts.begin(), terminalContacts.size(),
					&terminalRecoveredDynamicContacts, &stats);
			}
			if (!terminalRecoveredWorldStaticContacts.empty())
				clampWorldStaticSoftInelasticNormalVelocities(
					shellParticles, numShellParticles, terminalContacts.begin(),
					terminalContacts.size(),
					&terminalRecoveredWorldStaticContacts, &stats);
			if (!terminalRecoveredWorldStaticBodies.empty())
				clampWorldStaticSoftBodyEndpointVelocities(
					shellParticles, numShellParticles, softBodiesForRecovery,
					numSoftBodiesForRecovery,
					&terminalRecoveredWorldStaticBodies,
					&terminalWorldStaticNormals, &stats);
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
// Body-static normal depenetration (TGS-style capped geometric projection)
//=============================================================================
void AvbdSolver::applyBodyStaticNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
    const physx::PxArray<bool> *skipDepenForBodies,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  (void)stats;
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
      if (contacts[c].maxImpulse < PX_MAX_REAL)
        continue;
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
      if (deformableAnchor) {
        if (deformableNormalStageMask &&
            c < deformableNormalStageMask->size())
          (*deformableNormalStageMask)[c] |= 2u;
        PX_AVBD_PROFILE_STAT(stats->surfaceDeformableDepenetrationCorrections++);
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
    if (hasVelocityTangentMaterialOwner(cc) &&
        !hasVelocityBodyStaticFrictionSweepOwner(cc))
      continue;
    if (hasDeformablePositionTangentOwner(cc))
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
      if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats &&
          hasDeformableStaticAnchor(cc) &&
          (jUnc[0] * jUnc[0] + jUnc[1] * jUnc[1]) > 1.0e-16f)
        PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFrictionCorrections++);
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
  (void)stats;
  if (!softContacts || numSoftContacts == 0 || !bodies || numBodies == 0 ||
      !softParticles || sweeps == 0 || dt <= 0.0f)
    return;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      AvbdSoftContact &sc = softContacts[sci];
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      const AvbdSoftContactAugmentedState &state = sc.state;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex >= numBodies)
        continue;
      if (!avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles))
        continue;
      AvbdSolverBody &body = bodies[geometry.targetIndex];
      if (body.invMass <= 0.0f)
        continue;

      const physx::PxReal violation =
          avbdKinematicShellContactViolation(geometry, body);
      if (violation >= -1e-5f)
        continue;

      const physx::PxReal approachSpeed =
          body.linearVelocity.magnitude() + gravity.magnitude() * dt;
      physx::PxReal sweepCap =
          physx::PxMax(approachSpeed * dt * 0.5f, 0.04f);
      const physx::PxVec3 meshStep =
          geometry.surfacePoint - state.surfacePointPrev;
      sweepCap = physx::PxMax(sweepCap, meshStep.magnitude() * 1.5f);
      if (violation < -0.05f)
        sweepCap = physx::PxMax(sweepCap, -violation * 0.6f);
      const physx::PxReal corr = physx::PxMin(-violation, sweepCap);
      body.position += geometry.normal * corr;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

//=============================================================================
// Current-pose soft/world-static overlap recovery
//=============================================================================

void AvbdSolver::applyWorldStaticSoftBodyEndpointTranslations(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxArray<physx::PxU8> *recoveredBodies,
    physx::PxArray<physx::PxVec3> *recoveryNormals,
    physx::PxArray<physx::PxVec3> *recoveryTranslations,
    bool allowFreshTriangleCoreExit,
    AvbdSolverStats *stats) {
  (void)stats;
  if (!softParticles || numSoftParticles == 0 || !softBodies ||
      numSoftBodies == 0 || !softContacts || numSoftContacts == 0 ||
      !recoveredBodies || !recoveryNormals ||
      recoveredBodies->size() != numSoftBodies ||
      recoveryNormals->size() != numSoftBodies ||
      (recoveryTranslations && recoveryTranslations->size() != numSoftBodies))
    return;

  // Coherent whole-volume translation was useful as a temporary geometry
  // repair, but it is not a physical world-static contact response: it can
  // move an unsupported falling soft body clear of a plane/box and then hide
  // the corresponding velocity in `initialPosition`. Normal OGC ownership
  // belongs to the local mass-weighted rows below. Keep this narrow escape
  // facility opt-in for an explicit recovery caller only; no production
  // native path currently opts in.
  if (!allowFreshTriangleCoreExit)
    return;

  physx::PxArray<physx::PxReal> deepestOverlap(numSoftBodies, 0.0f);
  physx::PxArray<physx::PxVec3> deepestNormals(numSoftBodies,
                                                physx::PxVec3(0.0f));
  // A fresh TBIX certificate is exact for its collision triangle.  Multiple
  // collision triangles through one static OBB can still share a coherent
  // whole-body exit: aggregate their required distances on each of the six
  // OBB face axes.  Different static primitives remain fail-closed because a
  // single uniform translation cannot prove both exits at once.
  struct TriangleCoreGroup {
    physx::PxU32 sourceBodyIndex;
    physx::PxU32 targetIndex;
    physx::PxU64 primitiveKey;
    physx::PxU32 representativeContact;
  };
  physx::PxArray<TriangleCoreGroup> triangleCoreGroups;
  physx::PxArray<physx::PxReal> triangleCoreExitDistances(numSoftBodies,
                                                           0.0f);
  physx::PxArray<physx::PxVec3> triangleCoreExitNormals(
      numSoftBodies, physx::PxVec3(0.0f));
  physx::PxArray<physx::PxU8> triangleCoreGroupCounts(numSoftBodies, 0u);

    // This is endpoint DCD, never a swept test.  A coherent translation is
    // safe for the deliberately narrow all-dynamic/no-objective admission, so
    // it must start at the first real surface overlap rather than waiting for
    // a deep AL residual that may already have degraded an incident tet.
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if ((geometry.source.type != AvbdSoftContactSource::eGROUND &&
         geometry.source.type != AvbdSoftContactSource::eRIGID_SDF) ||
        !geometry.hasWorldStaticTarget() ||
        geometry.velocityOwner != AvbdVelocityObjectiveOwner::PositionAL ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    const physx::PxU32 representative =
        geometry.hasWeightedQueryPoint()
            ? geometry.queryPoint.particleIndices[0]
            : geometry.hasBarycentricQueryPoint()
                  ? geometry.queryParticleIndices[0]
                  : geometry.particleIdx;
    if (representative >= numSoftParticles)
      continue;
    const AvbdSoftBody *sourceBody =
        geometry.queryBodyIndex < numSoftBodies &&
                avbdSoftBodyContainsParticle(
                    softBodies[geometry.queryBodyIndex], representative,
                    numSoftParticles)
            ? &softBodies[geometry.queryBodyIndex]
            : avbdFindSoftBodyForParticle(softBodies, numSoftBodies,
                                           representative);
    if (!sourceBody || sourceBody->compiled.speculativeCCDEnabled ||
        !physx::PxIsFinite(sourceBody->compiled.maxDepenetrationVelocity) ||
        sourceBody->compiled.maxDepenetrationVelocity < 1.0e20f)
      continue;
    const physx::PxU32 bodyIndex =
        physx::PxU32(sourceBody - softBodies);
    if (bodyIndex >= numSoftBodies)
      continue;

    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!queryPoint.isFinite())
      continue;
    physx::PxVec3 normal(0.0f);
    physx::PxReal trueGap = 0.0f;
    if (!getCurrentWorldStaticSoftContactGeometry(
            geometry, queryPoint, normal, trueGap))
      continue;
    if (!physx::PxIsFinite(trueGap))
      continue;

    if (trueGap < -1.0e-5f) {
      const physx::PxReal overlap = -trueGap;
      if (overlap > deepestOverlap[bodyIndex]) {
        deepestOverlap[bodyIndex] = overlap;
        deepestNormals[bodyIndex] = normal;
      }
    }

    // This certificate comes only from the current-pose triangle/box core
    // clipper. It is not contact-shell depth and is never authored by a
    // swept/CCD detector.  Ground/plane rows deliberately remain on the
    // ordinary true-gap path above: only a world-static box has the local
    // triangle bounds needed for a common-axis aggregate.
    if (allowFreshTriangleCoreExit &&
        (*recoveredBodies)[bodyIndex] == 0u &&
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBoxSdf && geometry.hasRigidBoxTriangleCoreExit &&
        geometry.rigidBoxPose.p.isFinite() &&
        geometry.rigidBoxPose.q.isFinite() &&
        geometry.rigidBoxHalfExtent.isFinite() &&
        geometry.rigidBoxTriangleCoreMinimumLocal.isFinite() &&
        geometry.rigidBoxTriangleCoreMaximumLocal.isFinite()) {
      bool groupExists = false;
      for (physx::PxU32 groupIndex = 0;
           groupIndex < triangleCoreGroups.size(); ++groupIndex) {
        const TriangleCoreGroup &group = triangleCoreGroups[groupIndex];
        if (group.sourceBodyIndex == bodyIndex &&
            group.targetIndex == geometry.targetIndex &&
            group.primitiveKey == geometry.source.primitiveKey) {
          groupExists = true;
          break;
        }
      }
      if (!groupExists) {
        TriangleCoreGroup group;
        group.sourceBodyIndex = bodyIndex;
        group.targetIndex = geometry.targetIndex;
        group.primitiveKey = geometry.source.primitiveKey;
        group.representativeContact = sci;
        triangleCoreGroups.pushBack(group);
      }
    }
  }

  if (allowFreshTriangleCoreExit) {
    for (physx::PxU32 groupIndex = 0;
         groupIndex < triangleCoreGroups.size(); ++groupIndex) {
      const TriangleCoreGroup &group = triangleCoreGroups[groupIndex];
      if (group.sourceBodyIndex < numSoftBodies &&
          triangleCoreGroupCounts[group.sourceBodyIndex] < 2u)
        ++triangleCoreGroupCounts[group.sourceBodyIndex];
    }

    const physx::PxReal lengthScale =
        physx::PxMax(mConfig.lengthScale, 1.0e-6f);
    const physx::PxReal exitTolerance =
        physx::PxMax(1.0e-6f, 1.0e-5f * lengthScale);
    for (physx::PxU32 groupIndex = 0;
         groupIndex < triangleCoreGroups.size(); ++groupIndex) {
      const TriangleCoreGroup &group = triangleCoreGroups[groupIndex];
      if (group.sourceBodyIndex >= numSoftBodies ||
          group.representativeContact >= numSoftContacts ||
          triangleCoreGroupCounts[group.sourceBodyIndex] != 1u ||
          (*recoveredBodies)[group.sourceBodyIndex] != 0u)
        continue;

      const AvbdSoftContactGeometry &carrier =
          softContacts[group.representativeContact].geometry;
      if (!carrier.rigidBoxPose.p.isFinite() ||
          !carrier.rigidBoxPose.q.isFinite() ||
          !carrier.rigidBoxHalfExtent.isFinite())
        continue;
      const physx::PxQuat shapeRotation = carrier.rigidBoxPose.q;
      physx::PxReal aggregateExit[6] = {
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      bool validGroup = true;
      bool hasCoreTriangle = false;
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
        const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
        if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
            !geometry.hasWorldStaticTarget() ||
            geometry.targetIndex != group.targetIndex ||
            geometry.source.primitiveKey != group.primitiveKey ||
            !geometry.hasRigidBoxSdf ||
            !geometry.hasRigidBoxTriangleCoreExit)
          continue;

        const physx::PxU32 representative =
            geometry.hasWeightedQueryPoint()
                ? geometry.queryPoint.particleIndices[0]
                : geometry.hasBarycentricQueryPoint()
                      ? geometry.queryParticleIndices[0]
                      : geometry.particleIdx;
        if (representative >= numSoftParticles)
          continue;
        const AvbdSoftBody *sourceBody =
            geometry.queryBodyIndex < numSoftBodies &&
                    avbdSoftBodyContainsParticle(
                        softBodies[geometry.queryBodyIndex], representative,
                        numSoftParticles)
                ? &softBodies[geometry.queryBodyIndex]
                : avbdFindSoftBodyForParticle(softBodies, numSoftBodies,
                                               representative);
        if (!sourceBody ||
            physx::PxU32(sourceBody - softBodies) != group.sourceBodyIndex)
          continue;

        // One primitive group must retain a single immutable OBB frame.  A
        // mismatch means that a common-axis exit has no geometric proof.
        if (!geometry.rigidBoxPose.p.isFinite() ||
            !geometry.rigidBoxPose.q.isFinite() ||
            !geometry.rigidBoxHalfExtent.isFinite() ||
            geometry.rigidBoxPose.p != carrier.rigidBoxPose.p ||
            !(geometry.rigidBoxPose.q == carrier.rigidBoxPose.q) ||
            geometry.rigidBoxHalfExtent != carrier.rigidBoxHalfExtent) {
          validGroup = false;
          break;
        }

        hasCoreTriangle = true;
        for (physx::PxU32 face = 0; face < 6u; ++face) {
          physx::PxReal exitDistance = 0.0f;
          if (!getRigidBoxTriangleCoreExitDistance(geometry, face,
                                                   exitDistance)) {
            validGroup = false;
            break;
          }
          aggregateExit[face] =
              physx::PxMax(aggregateExit[face], exitDistance);
        }
        if (!validGroup)
          break;
      }
      if (!validGroup || !hasCoreTriangle)
        continue;

      physx::PxReal bestExitDistance = PX_MAX_F32;
      physx::PxVec3 bestExitNormal(0.0f);
      for (physx::PxU32 face = 0; face < 6u; ++face) {
        const physx::PxReal candidateExit = aggregateExit[face];
        if (!(candidateExit > exitTolerance) ||
            candidateExit >= bestExitDistance)
          continue;
        const physx::PxVec3 worldNormal = shapeRotation.rotate(
            getRigidBoxTriangleCoreExitNormalLocal(face));
        const physx::PxReal normalLengthSq = worldNormal.magnitudeSquared();
        if (!worldNormal.isFinite() || !physx::PxIsFinite(normalLengthSq) ||
            normalLengthSq <= 1.0e-12f)
          continue;
        bestExitDistance = candidateExit;
        bestExitNormal =
            worldNormal * physx::PxRecipSqrt(normalLengthSq);
      }
      if (bestExitDistance < PX_MAX_F32) {
        triangleCoreExitDistances[group.sourceBodyIndex] = bestExitDistance;
        triangleCoreExitNormals[group.sourceBodyIndex] = bestExitNormal;
      }
    }
  }

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies;
       ++bodyIndex) {
    physx::PxReal overlap = deepestOverlap[bodyIndex];
    physx::PxVec3 normal = deepestNormals[bodyIndex];
    if (allowFreshTriangleCoreExit &&
        triangleCoreGroupCounts[bodyIndex] == 1u &&
        triangleCoreExitDistances[bodyIndex] > 0.0f) {
      overlap = triangleCoreExitDistances[bodyIndex];
      normal = triangleCoreExitNormals[bodyIndex];
    }
    if (!(overlap > 0.0f) || !physx::PxIsFinite(overlap) ||
        !normal.isFinite())
      continue;

    const AvbdSoftBody &body = softBodies[bodyIndex];
    const physx::PxU32 particleStart = body.compiled.particleStart;
    const physx::PxU32 particleCount = body.compiled.particleCount;
    if (particleStart > numSoftParticles ||
        particleCount > numSoftParticles - particleStart ||
        body.runtime.objectiveAdjacency.size() != particleCount)
      continue;

    const physx::PxVec3 delta = normal * overlap;
    if (!delta.isFinite())
      continue;
    bool validBodyTranslation = true;
    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex) {
      const physx::PxU32 particleIndex = particleStart + localIndex;
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      if (particle.invMass <= 0.0f || !particle.position.isFinite() ||
          !particle.initialPosition.isFinite() ||
          !particle.predictedPosition.isFinite() ||
          !particle.outerPosition.isFinite() ||
          !body.runtime.objectiveAdjacency[localIndex]
               .objectiveIndices.empty() ||
          !(particle.position + delta).isFinite() ||
          !(particle.initialPosition + delta).isFinite() ||
          !(particle.predictedPosition + delta).isFinite() ||
          !(particle.outerPosition + delta).isFinite()) {
        validBodyTranslation = false;
        break;
      }
    }
    if (!validBodyTranslation)
      continue;

    // A uniform translation leaves every edge vector and tet F unchanged,
    // so unlike a per-support projection it cannot turn a healthy endpoint
    // configuration inside out before the material solve sees it.
    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex) {
      AvbdSoftParticle &particle =
          softParticles[particleStart + localIndex];
      particle.position += delta;
      particle.initialPosition += delta;
      particle.predictedPosition += delta;
      particle.outerPosition += delta;
    }
    (*recoveredBodies)[bodyIndex] = 1u;
    (*recoveryNormals)[bodyIndex] = normal;
    if (recoveryTranslations)
      (*recoveryTranslations)[bodyIndex] = delta;
  }
}

void AvbdSolver::clampWorldStaticSoftBodyEndpointVelocities(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const physx::PxArray<physx::PxU8> *recoveredBodies,
    const physx::PxArray<physx::PxVec3> *recoveryNormals,
    AvbdSolverStats *stats) {
  (void)stats;
  if (!softParticles || numSoftParticles == 0 || !softBodies ||
      numSoftBodies == 0 || !recoveredBodies || !recoveryNormals ||
      recoveredBodies->size() != numSoftBodies ||
      recoveryNormals->size() != numSoftBodies)
    return;

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies;
       ++bodyIndex) {
    if ((*recoveredBodies)[bodyIndex] == 0u)
      continue;
    const physx::PxVec3 rawNormal = (*recoveryNormals)[bodyIndex];
    const physx::PxReal normalLengthSq = rawNormal.magnitudeSquared();
    if (!rawNormal.isFinite() || !physx::PxIsFinite(normalLengthSq) ||
        normalLengthSq <= 1.0e-12f)
      continue;
    const physx::PxVec3 normal =
        rawNormal * physx::PxRecipSqrt(normalLengthSq);
    const AvbdSoftBody &body = softBodies[bodyIndex];
    const physx::PxU32 particleStart = body.compiled.particleStart;
    const physx::PxU32 particleCount = body.compiled.particleCount;
    if (particleStart > numSoftParticles ||
        particleCount > numSoftParticles - particleStart)
      continue;
    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex) {
      AvbdSoftParticle &particle =
          softParticles[particleStart + localIndex];
      if (particle.invMass <= 0.0f || !particle.velocity.isFinite())
        continue;
      const physx::PxReal inwardVelocity = particle.velocity.dot(normal);
      if (inwardVelocity >= -1.0e-6f)
        continue;
      const physx::PxVec3 candidateVelocity =
          particle.velocity - normal * inwardVelocity;
      if (candidateVelocity.isFinite())
        particle.velocity = candidateVelocity;
    }
  }
}

void AvbdSolver::applyWorldStaticTriangleCoreLocalManifold(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, AvbdSolverStats *stats,
    const AvbdSoftIslandExecutionPlan *ogcExecutionPlan,
    AvbdSolverBody *ogcRigidBodies,
    physx::PxU32 numOgcRigidBodies,
    const AvbdSoftContact *ogcContacts,
    physx::PxU32 numOgcContacts) {
  if (!softParticles || !softBodies || !softContacts ||
      numSoftParticles == 0 || numSoftBodies == 0 || numSoftContacts == 0 ||
      sweeps == 0)
    return;

  struct TriangleCoreGroup {
    physx::PxU32 sourceBodyIndex;
    physx::PxU32 targetIndex;
    physx::PxU64 primitiveKey;
    physx::PxU32 representativeContact;
  };
  physx::PxArray<TriangleCoreGroup> groups;
  auto isGroupContact = [](const AvbdSoftContactGeometry &geometry,
                           const TriangleCoreGroup &group) {
    return geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasWorldStaticTarget() &&
        geometry.queryBodyIndex == group.sourceBodyIndex &&
        geometry.targetIndex == group.targetIndex &&
        geometry.source.primitiveKey == group.primitiveKey &&
        geometry.hasRigidBoxSdf && geometry.hasRigidBoxTriangleCoreExit;
  };
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasWorldStaticTarget() ||
        geometry.queryBodyIndex >= numSoftBodies ||
        !geometry.hasRigidBoxSdf || !geometry.hasRigidBoxTriangleCoreExit ||
        !geometry.rigidBoxPose.isValid() ||
        !geometry.rigidBoxHalfExtent.isFinite() ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles) ||
        softBodies[geometry.queryBodyIndex].compiled.speculativeCCDEnabled)
      continue;
    bool exists = false;
    for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
      if (isGroupContact(geometry, groups[groupIndex])) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      TriangleCoreGroup group;
      group.sourceBodyIndex = geometry.queryBodyIndex;
      group.targetIndex = geometry.targetIndex;
      group.primitiveKey = geometry.source.primitiveKey;
      group.representativeContact = sci;
      groups.pushBack(group);
    }
  }
  if (groups.empty())
    return;

  auto evaluatePoint = [](const AvbdWeightedContactPoint &point,
                          const AvbdSoftParticle *particles,
                          physx::PxU32 numParticles,
                          physx::PxVec3 &result) {
    if (point.count == 0 || point.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    result = physx::PxVec3(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 i = 0; i < point.count; ++i) {
      const physx::PxU32 particleIndex = point.particleIndices[i];
      const physx::PxReal weight = point.weights[i];
      if (particleIndex >= numParticles || !physx::PxIsFinite(weight) ||
          !particles[particleIndex].position.isFinite())
        return false;
      result += particles[particleIndex].position * weight;
      weightSum += weight;
    }
    return result.isFinite() && physx::PxIsFinite(weightSum) &&
        weightSum > 1.0e-6f;
  };
  const physx::PxReal tolerance = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
      const TriangleCoreGroup &group = groups[groupIndex];
      if (group.representativeContact >= numSoftContacts)
        continue;
      const AvbdSoftContactGeometry &carrier =
          softContacts[group.representativeContact].geometry;
      const physx::PxTransform boxToWorld = carrier.rigidBoxPose;
      if (!boxToWorld.isValid())
        continue;

      physx::PxReal faceExit[6] = {0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f};
      bool validGroup = true;
      bool hasCore = false;
      for (physx::PxU32 sci = 0;
           sci < numSoftContacts && validGroup; ++sci) {
        const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
        if (!isGroupContact(geometry, group))
          continue;
        if (geometry.rigidBoxPose.p != carrier.rigidBoxPose.p ||
            !(geometry.rigidBoxPose.q == carrier.rigidBoxPose.q) ||
            geometry.rigidBoxHalfExtent != carrier.rigidBoxHalfExtent) {
          validGroup = false;
          break;
        }
        hasCore = true;
        physx::PxVec3 minimum(PX_MAX_F32);
        physx::PxVec3 maximum(-PX_MAX_F32);
        for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
          physx::PxVec3 point(0.0f);
          if (!evaluatePoint(geometry.rigidBoxTriangleCorePoints[vertex],
                             softParticles, numSoftParticles, point)) {
            validGroup = false;
            break;
          }
          const physx::PxVec3 localPoint = boxToWorld.transformInv(point);
          if (!localPoint.isFinite()) {
            validGroup = false;
            break;
          }
          minimum = minimum.minimum(localPoint);
          maximum = maximum.maximum(localPoint);
        }
        const physx::PxVec3 &extent = carrier.rigidBoxHalfExtent;
        const physx::PxReal exits[6] = {
            extent.x - minimum.x, maximum.x + extent.x,
            extent.y - minimum.y, maximum.y + extent.y,
            extent.z - minimum.z, maximum.z + extent.z};
        for (physx::PxU32 face = 0; face < 6 && validGroup; ++face) {
          if (!physx::PxIsFinite(exits[face])) {
            validGroup = false;
            break;
          }
          faceExit[face] =
              physx::PxMax(faceExit[face], physx::PxMax(0.0f, exits[face]));
        }
      }
      if (!validGroup || !hasCore)
        continue;

      physx::PxU32 face = PX_MAX_U32;
      physx::PxReal bestExit = PX_MAX_F32;
      for (physx::PxU32 candidate = 0; candidate < 6; ++candidate) {
        if (faceExit[candidate] <= tolerance) {
          face = PX_MAX_U32;
          bestExit = 0.0f;
          break;
        }
        if (faceExit[candidate] < bestExit) {
          face = candidate;
          bestExit = faceExit[candidate];
        }
      }
      if (face == PX_MAX_U32 || !physx::PxIsFinite(bestExit) ||
          bestExit <= tolerance)
        continue;

      const physx::PxVec3 localNormal =
          getRigidBoxTriangleCoreExitNormalLocal(face);
      const physx::PxVec3 worldNormal =
          boxToWorld.q.rotate(localNormal).getNormalized();
      const physx::PxU32 axis = face >> 1u;
      const physx::PxVec3 facePoint =
          boxToWorld.transform(localNormal * carrier.rigidBoxHalfExtent[axis]);
      if (!worldNormal.isFinite() || !facePoint.isFinite())
        continue;

      // Keep every actually penetrating collision-triangle vertex in this
      // local patch.  Selecting only the deepest vertex leaves the other two
      // vertices of a triangle-core intersection for a later frame; the
      // collision triangle can therefore remain inside the OBB even after a
      // successful single-vertex correction.  These rows share one freshly
      // selected OBB exit face and are solved through the existing
      // mass-weighted, positive-J guarded static projector, so this spreads
      // load through the contact patch without a body-wide teleport or a
      // second time step.
      physx::PxArray<AvbdSoftContact> patchContacts;
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
        if (!isGroupContact(softContacts[sci].geometry, group))
          continue;
        for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
          AvbdSoftContact localContact = softContacts[sci];
          AvbdSoftContactGeometry &localGeometry = localContact.geometry;
          localGeometry.queryPoint =
              localGeometry.rigidBoxTriangleCorePoints[vertex];
          localGeometry.hasRigidBoxSdf = false;
          localGeometry.hasRigidBoxTriangleCoreExit = false;
          localGeometry.normal = worldNormal;
          localGeometry.projNormal = worldNormal;
          localGeometry.surfacePoint = facePoint;
          physx::PxVec3 queryPoint(0.0f);
          if (!evaluatePoint(localGeometry.queryPoint, softParticles,
                             numSoftParticles, queryPoint))
            continue;
          const physx::PxReal gap = (queryPoint - facePoint).dot(worldNormal);
          if (!physx::PxIsFinite(gap) || gap >= -tolerance)
            continue;
          patchContacts.pushBack(localContact);
        }
      }
      if (patchContacts.empty())
        continue;
      applyWorldStaticSoftNormalDepenetrationSweeps(
          softParticles, numSoftParticles, softBodies, numSoftBodies,
          patchContacts.begin(), patchContacts.size(), 1u, nullptr, stats,
          ogcExecutionPlan, ogcRigidBodies, numOgcRigidBodies, ogcContacts,
          numOgcContacts);
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

void AvbdSolver::applyWorldStaticSoftNormalDepenetrationSweeps(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps,
    physx::PxArray<physx::PxU8> *recoveredContacts,
    AvbdSolverStats *stats,
    const AvbdSoftIslandExecutionPlan *ogcExecutionPlan,
    AvbdSolverBody *ogcRigidBodies,
    physx::PxU32 numOgcRigidBodies,
    const AvbdSoftContact *ogcContacts,
    physx::PxU32 numOgcContacts) {
  (void)stats;
  if (!softParticles || numSoftParticles == 0 || !softBodies ||
      numSoftBodies == 0 || !softContacts || numSoftContacts == 0 ||
      sweeps == 0)
    return;

  if (recoveredContacts && recoveredContacts->size() != numSoftContacts) {
    recoveredContacts->resize(numSoftContacts);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      (*recoveredContacts)[sci] = 0u;
  }

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      AvbdSoftContact &sc = softContacts[sci];
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      if ((geometry.source.type != AvbdSoftContactSource::eGROUND &&
           geometry.source.type != AvbdSoftContactSource::eRIGID_SDF) ||
          !geometry.hasWorldStaticTarget() ||
          geometry.velocityOwner !=
              AvbdVelocityObjectiveOwner::PositionAL ||
          !avbdHasSoftContactDynamicQuerySupport(
              geometry, softParticles, numSoftParticles))
        continue;

      const physx::PxU32 queryRepresentative =
          geometry.hasWeightedQueryPoint()
              ? geometry.queryPoint.particleIndices[0]
              : geometry.hasBarycentricQueryPoint()
                    ? geometry.queryParticleIndices[0]
                    : geometry.particleIdx;
      if (queryRepresentative >= numSoftParticles)
        continue;
      const AvbdSoftBody *sourceBody =
          geometry.queryBodyIndex < numSoftBodies &&
                  avbdSoftBodyContainsParticle(
                      softBodies[geometry.queryBodyIndex],
                      queryRepresentative, numSoftParticles)
              ? &softBodies[geometry.queryBodyIndex]
              : avbdFindSoftBodyForParticle(
                    softBodies, numSoftBodies, queryRepresentative);
      // A finite public depenetration cap deliberately permits residual
      // overlap.  This safety recovery must preserve that authored policy,
      // and it must never consume a speculative CCD source.
      if (!sourceBody || sourceBody->compiled.speculativeCCDEnabled ||
          !physx::PxIsFinite(sourceBody->compiled.maxDepenetrationVelocity) ||
          sourceBody->compiled.maxDepenetrationVelocity < 1.0e20f)
        continue;

      const physx::PxVec3 queryPoint =
          avbdGetSoftContactQueryPoint(geometry, softParticles);
      if (!queryPoint.isFinite())
        continue;

      // Unlike avbdEvaluateSoftContactNormalConstraint(), this omits the OGC
      // margin.  Only actual collision-surface overlap is recoverable here.
      physx::PxVec3 normal(0.0f);
      physx::PxReal trueGap = 0.0f;
      if (!getCurrentWorldStaticSoftContactGeometry(
              geometry, queryPoint, normal, trueGap))
        continue;
      if (!(trueGap < 0.0f) || !physx::PxIsFinite(trueGap))
        continue;

      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleCount =
          avbdCollectSoftContactParticleIndices(geometry, particleIndices);
      if (particleCount == 0 || particleCount > AVBD_CONTACT_MAX_PARTICLES)
        continue;

      const AvbdSoftBody *supportBodies[AVBD_CONTACT_MAX_PARTICLES];
      physx::PxReal softResponse = 0.0f;
      bool validSupport = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        if (particleIndex >= numSoftParticles ||
            !avbdSoftBodyContainsParticle(
                *sourceBody, particleIndex, numSoftParticles)) {
          validSupport = false;
          break;
        }
        const physx::PxReal weight =
            avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        if (!physx::PxIsFinite(weight) ||
            !physx::PxIsFinite(particle.invMass) ||
            !particle.position.isFinite() ||
            !particle.initialPosition.isFinite()) {
          validSupport = false;
          break;
        }
        if (particle.invMass > 0.0f)
          softResponse += particle.invMass * weight * weight;
        supportBodies[pi] = sourceBody;
      }
      if (!validSupport || !physx::PxIsFinite(softResponse) ||
          softResponse <= 1.0e-12f)
        continue;

      // World-static geometry has no opposing movable endpoint. Recover the
      // complete actual overlap and let common-alpha exact-J admission be the
      // displacement limiter; a fixed small cap cannot recover a 60 Hz fall.
      const physx::PxReal requestedCorrection = -trueGap;
      const physx::PxReal lambda = requestedCorrection / softResponse;
      if (!physx::PxIsFinite(lambda) || lambda <= 0.0f)
        continue;

      physx::PxVec3 particleDeltas[AVBD_CONTACT_MAX_PARTICLES];
      bool finiteCandidates = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        particleDeltas[pi] = physx::PxVec3(0.0f);
        if (particle.invMass <= 0.0f)
          continue;
        const physx::PxReal weight =
            avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
        const physx::PxVec3 delta =
            normal * (particle.invMass * weight * lambda);
        if (!delta.isFinite() || !(particle.position + delta).isFinite() ||
            !(particle.initialPosition + delta).isFinite()) {
          finiteCandidates = false;
          break;
        }
        particleDeltas[pi] = delta;
      }
      if (!finiteCandidates)
        continue;

      // Every support vertex and every incident tet shares the same alpha.
      // This is the only safe way to retain the coupled contact correction
      // while applying the scalar path's exact positive-J admission rule.
      physx::PxReal commonAlpha = 1.0f;
      bool acceptedCandidate = false;
      for (physx::PxU32 attempt = 0; attempt < 8 && !acceptedCandidate;
           ++attempt) {
        bool candidateValid = true;
        for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
          const AvbdSoftParticle &particle =
              softParticles[particleIndices[pi]];
          const physx::PxVec3 delta = particleDeltas[pi] * commonAlpha;
          if (!delta.isFinite() || !(particle.position + delta).isFinite() ||
              !(particle.initialPosition + delta).isFinite()) {
            candidateValid = false;
            break;
          }
        }

        auto candidatePositionFor =
            [&particleIndices, &particleDeltas, softParticles, particleCount,
             commonAlpha](physx::PxU32 particleIndex) -> physx::PxVec3 {
          for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
            if (particleIndices[pi] == particleIndex)
              return softParticles[particleIndex].position +
                     particleDeltas[pi] * commonAlpha;
          }
          return softParticles[particleIndex].position;
        };
        // If a preceding contact already left an incident tet below the
        // ordinary positive-J floor, an emergency separation step must be
        // allowed to *repair* it.  Requiring it to jump directly to .05
        // freezes recovery exactly when it is needed most.  Such a step is
        // still fail-closed: every healthy tet must remain healthy, every
        // unhealthy tet must be non-decreasing, and at least one unhealthy
        // tet must strictly improve.
        bool hasSubthresholdIncidentTet = false;
        bool improvesSubthresholdIncidentTet = false;
        const physx::PxReal detRecoveryTolerance = 1.0e-6f;
        for (physx::PxU32 pi = 0; pi < particleCount && candidateValid;
             ++pi) {
          const AvbdSoftBody *supportBody = supportBodies[pi];
          const physx::PxU32 particleIndex = particleIndices[pi];
          if (!supportBody ||
              !avbdSoftBodyContainsParticle(
                  *supportBody, particleIndex, numSoftParticles)) {
            candidateValid = false;
            break;
          }
          const physx::PxU32 localIndex =
              particleIndex - supportBody->compiled.particleStart;
          if (localIndex >= supportBody->compiled.elementAdjacency.size()) {
            candidateValid = false;
            break;
          }
          const AvbdParticleElementAdjacency &adjacency =
              supportBody->compiled.elementAdjacency[localIndex];
          for (physx::PxU32 refIndex = 0;
               refIndex < adjacency.tetRefs.size(); ++refIndex) {
            const AvbdParticleElementRef &ref = adjacency.tetRefs[refIndex];
            if (ref.index >= supportBody->compiled.tetElements.size()) {
              candidateValid = false;
              break;
            }
            const AvbdTetElement &tet =
                supportBody->compiled.tetElements[ref.index];
            if (tet.p0 >= numSoftParticles || tet.p1 >= numSoftParticles ||
                tet.p2 >= numSoftParticles || tet.p3 >= numSoftParticles) {
              candidateValid = false;
              break;
            }
            const physx::PxVec3 currentP0 = softParticles[tet.p0].position;
            const physx::PxVec3 currentE1 =
                softParticles[tet.p1].position - currentP0;
            const physx::PxVec3 currentE2 =
                softParticles[tet.p2].position - currentP0;
            const physx::PxVec3 currentE3 =
                softParticles[tet.p3].position - currentP0;
            physx::PxReal currentDeterminant;
            physx::PxVec3 unusedCurrentGradient;
            avbdEvaluateTetDeterminantAndGradient(
                tet, 0u, currentE1, currentE2, currentE3,
                currentDeterminant, unusedCurrentGradient);
            const physx::PxVec3 p0 = candidatePositionFor(tet.p0);
            const physx::PxVec3 e1 = candidatePositionFor(tet.p1) - p0;
            const physx::PxVec3 e2 = candidatePositionFor(tet.p2) - p0;
            const physx::PxVec3 e3 = candidatePositionFor(tet.p3) - p0;
            physx::PxReal determinant;
            physx::PxVec3 unusedGradient;
            avbdEvaluateTetDeterminantAndGradient(
                tet, 0u, e1, e2, e3, determinant, unusedGradient);
            if (!physx::PxIsFinite(currentDeterminant) ||
                !physx::PxIsFinite(determinant)) {
              candidateValid = false;
              break;
            }
            if (currentDeterminant >= 0.05f) {
              if (determinant < 0.05f) {
                candidateValid = false;
                break;
              }
            } else {
              hasSubthresholdIncidentTet = true;
              if (determinant + detRecoveryTolerance < currentDeterminant) {
                candidateValid = false;
                break;
              }
              if (determinant > currentDeterminant + detRecoveryTolerance)
                improvesSubthresholdIncidentTet = true;
            }
          }
        }

        if (candidateValid && hasSubthresholdIncidentTet &&
            !improvesSubthresholdIncidentTet)
          candidateValid = false;

        // A world-static correction is only one owner in the shared OGC
        // island.  Clip its complete support update against every dynamic
        // soft/rigid pair touched by those particles before committing it;
        // otherwise the pedestal can push the lower jaw through the box and
        // leave terminal recovery with an already-infeasible configuration.
        physx::PxReal ogcAlpha = 1.0f;
        if (candidateValid && ogcExecutionPlan && ogcRigidBodies &&
            ogcContacts && numOgcContacts > 0u) {
          for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
            const physx::PxVec3 delta = particleDeltas[pi] * commonAlpha;
            if (delta.magnitudeSquared() == 0.0f)
              continue;
            ogcAlpha = physx::PxMin(
                ogcAlpha,
                limitPostAlSoftParticleOgcCandidate(
                    ogcExecutionPlan, ogcContacts, numOgcContacts,
                    ogcRigidBodies, numOgcRigidBodies, softParticles,
                    numSoftParticles, particleIndices[pi], delta));
          }
          if (!physx::PxIsFinite(ogcAlpha) || ogcAlpha <= 0.0f) {
            candidateValid = false;
            commonAlpha = 0.0f;
          } else if (ogcAlpha < 1.0f - 1.0e-6f) {
            commonAlpha *= ogcAlpha;
            candidateValid = false;
          }
        }

        physx::PxVec3 candidateQueryPoint = queryPoint;
        for (physx::PxU32 pi = 0;
             pi < particleCount && candidateValid; ++pi) {
          const physx::PxReal weight =
              avbdGetSoftContactParticleJacobianScale(
                  geometry, particleIndices[pi]);
          candidateQueryPoint +=
              particleDeltas[pi] * (commonAlpha * weight);
          if (!physx::PxIsFinite(weight) || !candidateQueryPoint.isFinite())
            candidateValid = false;
        }
        physx::PxVec3 candidateNormal(0.0f);
        physx::PxReal candidateGap = 0.0f;
        const physx::PxReal gapImprovementTolerance = 1.0e-6f;
        if (candidateValid &&
            !getCurrentWorldStaticSoftContactGeometry(
                geometry, candidateQueryPoint, candidateNormal, candidateGap))
          candidateValid = false;
        if (candidateValid &&
            (!(candidateGap > trueGap + gapImprovementTolerance) ||
             !physx::PxIsFinite(candidateGap)))
          candidateValid = false;

        if (candidateValid)
          acceptedCandidate = true;
        else if (commonAlpha > 0.0f &&
                 !(ogcAlpha < 1.0f - 1.0e-6f))
          commonAlpha *= 0.5f;
      }
      if (!acceptedCandidate)
        continue;

      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxVec3 delta = particleDeltas[pi] * commonAlpha;
        if (delta.magnitudeSquared() == 0.0f)
          continue;
        AvbdSoftParticle &particle = softParticles[particleIndices[pi]];
        particle.position += delta;
        // The recovery is geometric, not an elastic launch.  Keep velocity
        // reconstruction anchored to the same corrected support position.
        particle.initialPosition += delta;
      }
      if (recoveredContacts && sci < recoveredContacts->size())
        (*recoveredContacts)[sci] = 1u;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

void AvbdSolver::clampWorldStaticSoftInelasticNormalVelocities(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxArray<physx::PxU8> *recoveredContacts,
    AvbdSolverStats *stats) {
  (void)stats;
  if (!softParticles || numSoftParticles == 0 || !softContacts ||
      numSoftContacts == 0 || !recoveredContacts ||
      recoveredContacts->size() != numSoftContacts)
    return;

  const physx::PxReal nearSurfaceTolerance = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    if ((*recoveredContacts)[sci] == 0u)
      continue;
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if ((geometry.source.type != AvbdSoftContactSource::eGROUND &&
         geometry.source.type != AvbdSoftContactSource::eRIGID_SDF) ||
        !geometry.hasWorldStaticTarget() ||
        geometry.velocityOwner != AvbdVelocityObjectiveOwner::PositionAL)
      continue;

    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!queryPoint.isFinite())
      continue;
    physx::PxVec3 normal(0.0f);
    physx::PxReal trueGap = 0.0f;
    if (!getCurrentWorldStaticSoftContactGeometry(
            geometry, queryPoint, normal, trueGap))
      continue;
    if (!physx::PxIsFinite(trueGap) || trueGap > nearSurfaceTolerance)
      continue;
    physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
    const physx::PxU32 particleCount =
        avbdCollectSoftContactParticleIndices(geometry, particleIndices);
    if (particleCount == 0 || particleCount > AVBD_CONTACT_MAX_PARTICLES)
      continue;

    physx::PxReal response = 0.0f;
    physx::PxVec3 queryVelocity(0.0f);
    bool validSupport = true;
    for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
      const physx::PxU32 particleIndex = particleIndices[pi];
      if (particleIndex >= numSoftParticles) {
        validSupport = false;
        break;
      }
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      if (!physx::PxIsFinite(weight) ||
          !physx::PxIsFinite(particle.invMass) ||
          !particle.velocity.isFinite()) {
        validSupport = false;
        break;
      }
      queryVelocity += particle.velocity * weight;
      if (particle.invMass > 0.0f)
        response += particle.invMass * weight * weight;
    }
    if (!validSupport || !queryVelocity.isFinite() ||
        !physx::PxIsFinite(response) || response <= 1.0e-12f)
      continue;

    const physx::PxReal normalVelocity = queryVelocity.dot(normal);
    if (!physx::PxIsFinite(normalVelocity) || normalVelocity >= -1.0e-6f)
      continue;
    const physx::PxReal impulse = -normalVelocity / response;
    if (!physx::PxIsFinite(impulse) || impulse <= 0.0f)
      continue;

    physx::PxVec3 velocityDeltas[AVBD_CONTACT_MAX_PARTICLES];
    bool finiteCandidates = true;
    for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
      const physx::PxU32 particleIndex = particleIndices[pi];
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      velocityDeltas[pi] = physx::PxVec3(0.0f);
      if (particle.invMass <= 0.0f)
        continue;
      const physx::PxVec3 delta =
          normal * (particle.invMass * weight * impulse);
      if (!delta.isFinite() || !(particle.velocity + delta).isFinite()) {
        finiteCandidates = false;
        break;
      }
      velocityDeltas[pi] = delta;
    }
    if (!finiteCandidates)
      continue;
    for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
      const physx::PxVec3 &delta = velocityDeltas[pi];
      if (delta.magnitudeSquared() > 0.0f)
        softParticles[particleIndices[pi]].velocity += delta;
    }
  }
}

//=============================================================================
// Dynamic soft/rigid SDF overlap recovery
//=============================================================================

void AvbdSolver::applyDynamicSoftRigidTriangleCoreLocalManifold(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps,
    physx::PxArray<physx::PxU8> *resolvedTriangleCoreContacts,
    AvbdSolverStats *stats, AvbdOgcPairState *ogcPairStates,
    physx::PxU32 numOgcPairStates,
    const physx::PxU32 *ogcPairIndices,
    physx::PxU32 numOgcPairIndices,
    const physx::PxU32 *ogcPairContactStarts,
    physx::PxU32 numOgcPairContactStarts,
    const physx::PxU32 *ogcPairContactRefs,
    physx::PxU32 numOgcPairContactRefs) {
  if (resolvedTriangleCoreContacts) {
    resolvedTriangleCoreContacts->resize(numSoftContacts);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      (*resolvedTriangleCoreContacts)[sci] = 0u;
  }
  if (!bodies || !softParticles || !softBodies || !softContacts ||
      numBodies == 0 || numSoftBodies == 0 || numSoftContacts == 0 ||
      sweeps == 0)
    return;

  struct TriangleCoreGroup {
    physx::PxU32 sourceBodyIndex;
    physx::PxU32 targetBodyIndex;
    physx::PxU64 primitiveKey;
    physx::PxU32 representativeContact;
    physx::PxU32 pairIndex;
  };
  physx::PxArray<TriangleCoreGroup> groups;
  bool usePairContactPlan =
      ogcPairStates && numOgcPairStates > 0 && ogcPairIndices &&
      numOgcPairIndices == numSoftContacts && ogcPairContactStarts &&
      numOgcPairContactStarts == numOgcPairStates + 1 &&
      (numOgcPairContactRefs == 0 || ogcPairContactRefs) &&
      ogcPairContactStarts[0] == 0 &&
      ogcPairContactStarts[numOgcPairStates] == numOgcPairContactRefs;
  if (usePairContactPlan) {
    // Validate the provider-owned inverse map once.  A malformed optional
    // acceleration plan must not change contact ownership: fall back to the
    // original direct scan below instead of partially using its ranges.
    for (physx::PxU32 pairIndex = 0;
         pairIndex < numOgcPairStates && usePairContactPlan; ++pairIndex) {
      const physx::PxU32 begin = ogcPairContactStarts[pairIndex];
      const physx::PxU32 end = ogcPairContactStarts[pairIndex + 1u];
      if (begin > end || end > numOgcPairContactRefs) {
        usePairContactPlan = false;
        break;
      }
      for (physx::PxU32 ref = begin; ref < end; ++ref) {
        const physx::PxU32 sci = ogcPairContactRefs[ref];
        if (sci >= numSoftContacts || ogcPairIndices[sci] != pairIndex) {
          usePairContactPlan = false;
          break;
        }
      }
    }
  }

  auto isEligibleTriangleCore = [&](physx::PxU32 sci) {
    if (sci >= numSoftContacts)
      return false;
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() ||
        geometry.targetIndex >= numBodies ||
        geometry.queryBodyIndex >= numSoftBodies ||
        !geometry.hasRigidBoxTriangleCoreExit ||
        !geometry.hasRigidBoxSdf ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      return false;
    return !softBodies[geometry.queryBodyIndex]
                .compiled.speculativeCCDEnabled &&
        bodies[geometry.targetIndex].invMass > 0.0f;
  };

  if (usePairContactPlan) {
    for (physx::PxU32 pairIndex = 0; pairIndex < numOgcPairStates;
         ++pairIndex) {
      const AvbdOgcPairState &pair = ogcPairStates[pairIndex];
      for (physx::PxU32 ref = ogcPairContactStarts[pairIndex];
           ref < ogcPairContactStarts[pairIndex + 1u]; ++ref) {
        const physx::PxU32 sci = ogcPairContactRefs[ref];
        if (!isEligibleTriangleCore(sci))
          continue;
        const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
        if (pair.sourceType != geometry.source.type ||
            pair.targetKind != geometry.targetKind ||
            pair.sourceBodyIndex != geometry.queryBodyIndex ||
            pair.targetBodyIndex != geometry.targetIndex ||
            pair.primitiveKey != geometry.source.primitiveKey)
          continue;
        TriangleCoreGroup group;
        group.sourceBodyIndex = geometry.queryBodyIndex;
        group.targetBodyIndex = geometry.targetIndex;
        group.primitiveKey = geometry.source.primitiveKey;
        group.representativeContact = sci;
        group.pairIndex = pairIndex;
        groups.pushBack(group);
        break;
      }
    }
  } else {
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      if (!isEligibleTriangleCore(sci))
        continue;
      const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
      bool exists = false;
      for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
           ++groupIndex) {
        const TriangleCoreGroup &group = groups[groupIndex];
        if (group.sourceBodyIndex == geometry.queryBodyIndex &&
            group.targetBodyIndex == geometry.targetIndex &&
            group.primitiveKey == geometry.source.primitiveKey) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        TriangleCoreGroup group;
        group.sourceBodyIndex = geometry.queryBodyIndex;
        group.targetBodyIndex = geometry.targetIndex;
        group.primitiveKey = geometry.source.primitiveKey;
        group.representativeContact = sci;
        group.pairIndex = PX_MAX_U32;
        groups.pushBack(group);
      }
    }
  }
  if (groups.empty())
    return;

  const physx::PxReal tolerance = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));

  auto isGroupContact = [](const AvbdSoftContactGeometry &geometry,
                           const TriangleCoreGroup &group) {
    return geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() &&
        geometry.queryBodyIndex == group.sourceBodyIndex &&
        geometry.targetIndex == group.targetBodyIndex &&
        geometry.source.primitiveKey == group.primitiveKey &&
        geometry.hasRigidBoxTriangleCoreExit && geometry.hasRigidBoxSdf;
  };
  auto evaluateWeightedPoint = [](const AvbdWeightedContactPoint &point,
                                  const AvbdSoftParticle *particles,
                                  physx::PxU32 numParticles,
                                  physx::PxVec3 &result) {
    if (point.count == 0 || point.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    result = physx::PxVec3(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 index = 0; index < point.count; ++index) {
      const physx::PxU32 particleIndex = point.particleIndices[index];
      const physx::PxReal weight = point.weights[index];
      if (particleIndex >= numParticles || !physx::PxIsFinite(weight) ||
          !particles[particleIndex].position.isFinite())
        return false;
      result += particles[particleIndex].position * weight;
      weightSum += weight;
    }
    return result.isFinite() && physx::PxIsFinite(weightSum) &&
        weightSum > 1.0e-6f;
  };
  auto markResolvedGroup = [&](const TriangleCoreGroup &group) {
    if (!resolvedTriangleCoreContacts)
      return;
    const physx::PxU32 begin =
        usePairContactPlan && group.pairIndex < numOgcPairStates
            ? ogcPairContactStarts[group.pairIndex]
            : 0u;
    const physx::PxU32 end =
        usePairContactPlan && group.pairIndex < numOgcPairStates
            ? ogcPairContactStarts[group.pairIndex + 1u]
            : numSoftContacts;
    for (physx::PxU32 index = begin; index < end; ++index) {
      const physx::PxU32 sci = usePairContactPlan
                                    ? ogcPairContactRefs[index]
                                    : index;
      if (isGroupContact(softContacts[sci].geometry, group))
        (*resolvedTriangleCoreContacts)[sci] = 1u;
    }
  };
  auto getPairForGroup = [&](const TriangleCoreGroup &group)
      -> AvbdOgcPairState * {
    if (usePairContactPlan && group.pairIndex < numOgcPairStates) {
      AvbdOgcPairState &pair = ogcPairStates[group.pairIndex];
      return pair.active && pair.sourceBodyIndex == group.sourceBodyIndex &&
              pair.targetBodyIndex == group.targetBodyIndex &&
              pair.primitiveKey == group.primitiveKey
          ? &pair
          : nullptr;
    }
    if (!ogcPairStates || !ogcPairIndices ||
        numOgcPairIndices != numSoftContacts ||
        group.representativeContact >= numSoftContacts)
      return nullptr;
    const physx::PxU32 pairIndex =
        ogcPairIndices[group.representativeContact];
    if (pairIndex >= numOgcPairStates)
      return nullptr;
    AvbdOgcPairState &pair = ogcPairStates[pairIndex];
    return pair.active && pair.sourceBodyIndex == group.sourceBodyIndex &&
            pair.targetBodyIndex == group.targetBodyIndex &&
            pair.primitiveKey == group.primitiveKey
        ? &pair
        : nullptr;
  };

  // This is a local manifold stage.  It deliberately reuses the exact
  // support-level positive-J projector below instead of translating an entire
  // volume.  Each group picks one face shared by every intersecting collision
  // triangle, so a successful pass proves all of those triangles lie outside
  // that face while subsequent material rows carry the compression inward.
  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
      const TriangleCoreGroup &group = groups[groupIndex];
      if (group.sourceBodyIndex >= numSoftBodies ||
          group.targetBodyIndex >= numBodies ||
          group.representativeContact >= numSoftContacts)
        continue;
      AvbdSolverBody &rigidBody = bodies[group.targetBodyIndex];
      const AvbdSoftContactGeometry &carrier =
          softContacts[group.representativeContact].geometry;
      if (rigidBody.invMass <= 0.0f || !rigidBody.position.isFinite() ||
          !rigidBody.rotation.isFinite() ||
          !carrier.rigidBoxPose.isValid() ||
          !carrier.rigidBoxHalfExtent.isFinite())
        continue;
      const physx::PxTransform shapeToWorld(
          rigidBody.position, rigidBody.rotation);
      const physx::PxTransform boxToWorld =
          shapeToWorld * carrier.rigidBoxPose;
      if (!boxToWorld.isValid())
        continue;

      physx::PxReal faceExit[6] = {0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f};
      bool validGroup = true;
      bool hasCore = false;
      const physx::PxU32 groupBegin =
          usePairContactPlan ? ogcPairContactStarts[group.pairIndex] : 0u;
      const physx::PxU32 groupEnd = usePairContactPlan
          ? ogcPairContactStarts[group.pairIndex + 1u]
          : numSoftContacts;
      for (physx::PxU32 index = groupBegin;
           index < groupEnd && validGroup; ++index) {
        const physx::PxU32 sci =
            usePairContactPlan ? ogcPairContactRefs[index] : index;
        const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
        if (!isGroupContact(geometry, group))
          continue;
        if (geometry.rigidBoxPose.p != carrier.rigidBoxPose.p ||
            !(geometry.rigidBoxPose.q == carrier.rigidBoxPose.q) ||
            geometry.rigidBoxHalfExtent != carrier.rigidBoxHalfExtent) {
          validGroup = false;
          break;
        }
        hasCore = true;
        physx::PxVec3 minimum(PX_MAX_F32);
        physx::PxVec3 maximum(-PX_MAX_F32);
        for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
          physx::PxVec3 point(0.0f);
          if (!evaluateWeightedPoint(
                  geometry.rigidBoxTriangleCorePoints[vertex], softParticles,
                  numSoftParticles, point)) {
            validGroup = false;
            break;
          }
          const physx::PxVec3 localPoint = boxToWorld.transformInv(point);
          if (!localPoint.isFinite()) {
            validGroup = false;
            break;
          }
          minimum = minimum.minimum(localPoint);
          maximum = maximum.maximum(localPoint);
        }
        if (!validGroup)
          break;
        const physx::PxVec3 &extent = carrier.rigidBoxHalfExtent;
        const physx::PxReal exits[6] = {
            extent.x - minimum.x, maximum.x + extent.x,
            extent.y - minimum.y, maximum.y + extent.y,
            extent.z - minimum.z, maximum.z + extent.z};
        for (physx::PxU32 face = 0; face < 6; ++face) {
          if (!physx::PxIsFinite(exits[face])) {
            validGroup = false;
            break;
          }
          faceExit[face] = physx::PxMax(faceExit[face],
                                         physx::PxMax(0.0f, exits[face]));
        }
      }
      if (!validGroup || !hasCore)
        continue;

      physx::PxU32 face = PX_MAX_U32;
      physx::PxReal bestExit = PX_MAX_F32;
      bool alreadySeparated = false;
      for (physx::PxU32 candidate = 0; candidate < 6; ++candidate) {
        if (faceExit[candidate] <= tolerance) {
          alreadySeparated = true;
          break;
        }
        if (faceExit[candidate] < bestExit) {
          bestExit = faceExit[candidate];
          face = candidate;
        }
      }
      if (alreadySeparated) {
        markResolvedGroup(group);
        if (AvbdOgcPairState *pair = getPairForGroup(group)) {
          pair->hasTriangleCoreManifold = true;
          pair->triangleCoreLocallyResolved = true;
          pair->triangleCoreFaceExit = 0.0f;
        }
        continue;
      }
      if (face == PX_MAX_U32 || !physx::PxIsFinite(bestExit))
        continue;

      if (AvbdOgcPairState *pair = getPairForGroup(group)) {
        pair->hasTriangleCoreManifold = true;
        pair->triangleCoreLocallyResolved = false;
        pair->triangleCoreFace = physx::PxU8(face);
        pair->triangleCoreFaceExit = bestExit;
      }

      const physx::PxVec3 localNormal =
          getRigidBoxTriangleCoreExitNormalLocal(face);
      const physx::PxVec3 worldNormal =
          boxToWorld.q.rotate(localNormal).getNormalized();
      const physx::PxReal normalLengthSq = worldNormal.magnitudeSquared();
      if (!worldNormal.isFinite() || !physx::PxIsFinite(normalLengthSq) ||
          normalLengthSq <= 1.0e-12f)
        continue;
      const physx::PxU32 axis = face >> 1u;
      const physx::PxVec3 faceLocal = localNormal *
          carrier.rigidBoxHalfExtent[axis];

      bool projectedGroup = false;
      const physx::PxU32 projectionGroupBegin =
          usePairContactPlan ? ogcPairContactStarts[group.pairIndex] : 0u;
      const physx::PxU32 projectionGroupEnd = usePairContactPlan
          ? ogcPairContactStarts[group.pairIndex + 1u]
          : numSoftContacts;
      // A collision triangle is not resolved when only its deepest vertex is
      // projected.  Its other vertices can still straddle the selected OBB
      // face, leaving a triangle-core intersection while concentrating the
      // reaction on the light rigid.  Build the complete local vertex patch
      // for this shared exit face instead.  The existing coupled
      // soft/6DOF projector applies each row with its exact support mass,
      // common-alpha positive-J test, and current-pose requery; this spreads
      // the correction into the deformable material without introducing an
      // additional time step or a swept/CCD query.
      physx::PxArray<AvbdSoftContact> patchContacts;
      for (physx::PxU32 index = projectionGroupBegin;
           index < projectionGroupEnd; ++index) {
        const physx::PxU32 sci =
            usePairContactPlan ? ogcPairContactRefs[index] : index;
        if (!isGroupContact(softContacts[sci].geometry, group))
          continue;
        for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
          AvbdSoftContact localContact = softContacts[sci];
          AvbdSoftContactGeometry &localGeometry = localContact.geometry;
          localGeometry.queryPoint =
              localGeometry.rigidBoxTriangleCorePoints[vertex];
          // Make the selected common OBB face a local plane row.  Leaving the
          // regular box-SDF flag enabled would let a different nearest face
          // split this manifold back into unrelated projections.
          localGeometry.hasRigidBoxSdf = false;
          localGeometry.hasRigidBoxTriangleCoreExit = false;
          localGeometry.normal = worldNormal;
          localGeometry.projNormal = worldNormal;
          localGeometry.rigidLocalPoint =
              carrier.rigidBoxPose.transform(faceLocal);
          localGeometry.surfacePoint = rigidBody.position +
              rigidBody.rotation.rotate(localGeometry.rigidLocalPoint);

          physx::PxVec3 queryPoint(0.0f);
          if (!evaluateWeightedPoint(localGeometry.queryPoint, softParticles,
                                     numSoftParticles, queryPoint))
            continue;
          const physx::PxVec3 currentSurface = rigidBody.position +
              rigidBody.rotation.rotate(localGeometry.rigidLocalPoint);
          const physx::PxReal planeGap =
              (queryPoint - currentSurface).dot(worldNormal);
          if (!physx::PxIsFinite(planeGap) || planeGap >= tolerance)
            continue;
          patchContacts.pushBack(localContact);
        }
      }
      if (!patchContacts.empty()) {
        applyDynamicSoftRigidNormalDepenetrationSweeps(
            bodies, numBodies, softParticles, numSoftParticles, softBodies,
            numSoftBodies, patchContacts.begin(), patchContacts.size(), 1u,
            nullptr, stats,
            nullptr, 0u, nullptr, 0u,
            /*softComplianceResponseScale=*/4.0f,
            /*projectToCurrentPoseBoundary=*/true);
        projectedGroup = true;
      }
      anyCorrection = anyCorrection || projectedGroup;
    }
    if (!anyCorrection)
      break;
  }

  // The endpoint fallback may only consume groups that failed the local
  // manifold.  Re-evaluate after all coupled local rows, rather than trusting
  // a per-vertex success bit from an earlier PGS pass.
  for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
       ++groupIndex) {
    const TriangleCoreGroup &group = groups[groupIndex];
    if (group.targetBodyIndex >= numBodies ||
        group.representativeContact >= numSoftContacts)
      continue;
    const AvbdSolverBody &rigidBody = bodies[group.targetBodyIndex];
    const AvbdSoftContactGeometry &carrier =
        softContacts[group.representativeContact].geometry;
    const physx::PxTransform boxToWorld(
        physx::PxTransform(rigidBody.position, rigidBody.rotation) *
        carrier.rigidBoxPose);
    if (!boxToWorld.isValid())
      continue;
    physx::PxReal faceExit[6] = {0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f};
    bool validGroup = true;
    const physx::PxU32 groupBegin =
        usePairContactPlan ? ogcPairContactStarts[group.pairIndex] : 0u;
    const physx::PxU32 groupEnd = usePairContactPlan
        ? ogcPairContactStarts[group.pairIndex + 1u]
        : numSoftContacts;
    for (physx::PxU32 index = groupBegin;
         index < groupEnd && validGroup; ++index) {
      const physx::PxU32 sci =
          usePairContactPlan ? ogcPairContactRefs[index] : index;
      const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
      if (!isGroupContact(geometry, group))
        continue;
      physx::PxVec3 minimum(PX_MAX_F32);
      physx::PxVec3 maximum(-PX_MAX_F32);
      for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
        physx::PxVec3 point(0.0f);
        if (!evaluateWeightedPoint(
                geometry.rigidBoxTriangleCorePoints[vertex], softParticles,
                numSoftParticles, point)) {
          validGroup = false;
          break;
        }
        const physx::PxVec3 localPoint = boxToWorld.transformInv(point);
        if (!localPoint.isFinite()) {
          validGroup = false;
          break;
        }
        minimum = minimum.minimum(localPoint);
        maximum = maximum.maximum(localPoint);
      }
      if (!validGroup)
        break;
      const physx::PxVec3 &extent = carrier.rigidBoxHalfExtent;
      const physx::PxReal exits[6] = {
          extent.x - minimum.x, maximum.x + extent.x,
          extent.y - minimum.y, maximum.y + extent.y,
          extent.z - minimum.z, maximum.z + extent.z};
      for (physx::PxU32 face = 0; face < 6; ++face)
        faceExit[face] = physx::PxMax(faceExit[face],
                                       physx::PxMax(0.0f, exits[face]));
    }
    if (!validGroup)
      continue;
    for (physx::PxU32 face = 0; face < 6; ++face) {
      if (faceExit[face] <= tolerance) {
        markResolvedGroup(group);
        if (AvbdOgcPairState *pair = getPairForGroup(group)) {
          pair->hasTriangleCoreManifold = true;
          pair->triangleCoreLocallyResolved = true;
          pair->triangleCoreFace = physx::PxU8(face);
          pair->triangleCoreFaceExit = 0.0f;
        }
        break;
      }
    }
  }
}

void AvbdSolver::applyDynamicSoftRigidBodyEndpointTranslations(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps,
    physx::PxArray<physx::PxU8> *recoveredContacts,
    const physx::PxArray<physx::PxVec3> *precedingStaticTranslations,
    bool allowFreshTriangleCoreExit,
    bool preferLocalTriangleCoreManifold,
    bool allowCoherentEndpointFallback,
    AvbdSolverStats *stats, AvbdOgcPairState *ogcPairStates,
    physx::PxU32 numOgcPairStates,
    const physx::PxU32 *ogcPairIndices,
    physx::PxU32 numOgcPairIndices,
    const physx::PxU32 *ogcPairContactStarts,
    physx::PxU32 numOgcPairContactStarts,
    const physx::PxU32 *ogcPairContactRefs,
    physx::PxU32 numOgcPairContactRefs) {
  (void)stats;
  if (!bodies || numBodies == 0 || !softParticles ||
      numSoftParticles == 0 || !softBodies || numSoftBodies == 0 ||
      !softContacts || numSoftContacts == 0 || sweeps == 0)
    return;

  if (recoveredContacts && recoveredContacts->size() != numSoftContacts) {
    recoveredContacts->resize(numSoftContacts);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      (*recoveredContacts)[sci] = 0u;
  }

  // Local triangle-manifold projection has to run at the terminal DCD epoch,
  // after every position-changing material/friction stage.  Earlier in the
  // frame it is only a useful candidate, not a final contact guarantee, so
  // the pre-AL safety path deliberately retains the coherent fallback.
  physx::PxArray<physx::PxU8> locallyResolvedTriangleCoreContacts;
  if (allowFreshTriangleCoreExit && preferLocalTriangleCoreManifold) {
    applyDynamicSoftRigidTriangleCoreLocalManifold(
        bodies, numBodies, softParticles, numSoftParticles, softBodies,
        numSoftBodies, softContacts, numSoftContacts, sweeps,
        &locallyResolvedTriangleCoreContacts, stats, ogcPairStates,
        numOgcPairStates, ogcPairIndices, numOgcPairIndices,
        ogcPairContactStarts, numOgcPairContactStarts,
        ogcPairContactRefs, numOgcPairContactRefs);

    // The local core manifold moves `initialPosition` with its geometric
    // correction, so it intentionally does not create a pose-derived bounce.
    // It still needs the usual e=0 velocity ownership at the resolved
    // surface; otherwise the unchanged inward velocity simply recreates the
    // same core overlap in the next frame.  Reuse the normal-recovery bit so
    // the existing shared soft/6DOF velocity clamp handles that endpoint
    // once, rather than adding a second triangle-core impulse path.
    if (recoveredContacts &&
        locallyResolvedTriangleCoreContacts.size() == numSoftContacts) {
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
        if (locallyResolvedTriangleCoreContacts[sci] != 0u)
          (*recoveredContacts)[sci] |= 1u;
      }
    }
  }

  // Normal mixed OGC keeps compression in the local coupled material/ridig
  // manifold.  A coherent full-body translation is a repair-only operation:
  // with a light free rigid it resolves geometry by launching the rigid rather
  // than by deforming the soft body.  Callers must opt into that behavior.
  if (!allowCoherentEndpointFallback)
    return;

  const physx::PxReal lengthScale =
      physx::PxMax(mConfig.lengthScale, 1.0e-6f);
  const physx::PxReal maxAngularStep = 0.25f;

  // Track only translations introduced by this coherent endpoint stage.  The
  // detector already used the current predicted pose, and certificates keep
  // their triangle bounds in that box frame.  Re-evaluating these relative
  // translations on every PGS sweep lets the upper and lower soft bodies
  // react to one another through the same free rigid without ever treating a
  // swept path as an input.
  physx::PxArray<physx::PxVec3> endpointSoftTranslations(numSoftBodies,
                                                           physx::PxVec3(0.0f));
  physx::PxArray<physx::PxVec3> endpointRigidTranslations(numBodies,
                                                            physx::PxVec3(0.0f));
  // Triangle-core bounds are expressed in the detector's box-local frame.
  // A preceding endpoint row may translate the box (which we account for
  // below), but a rotation would invalidate that frame.  Keep the entry
  // orientation so a core certificate fails closed after such a row instead
  // of applying a stale geometric proof.
  physx::PxArray<physx::PxQuat> endpointRigidRotations(numBodies);
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex)
    endpointRigidRotations[bodyIndex] = bodies[bodyIndex].rotation;
  if (precedingStaticTranslations &&
      precedingStaticTranslations->size() == numSoftBodies) {
    for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies;
         ++bodyIndex)
      endpointSoftTranslations[bodyIndex] =
          (*precedingStaticTranslations)[bodyIndex];
  }

  // Unlike the ordinary local recovery below, this endpoint-only fallback
  // moves the *complete* soft body.  That preserves every tet edge vector
  // exactly, which is the one safe escape hatch when a current-pose contact
  // first arrives after a fast fall has already made local det(F) admission
  // too restrictive. It is deliberately restricted to wholly dynamic,
  // objective-free bodies and starts at the first genuine surface overlap.
  // The only exception is a freshly detected current triangle/box core
  // certificate, which proves a collision triangle itself crosses the OBB;
  // it is admitted solely at the pre-AL boundary and is never a swept row.
  // The OGC shell itself remains exclusively Position-AL owned.
  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 sourceBodyIndex = 0;
         sourceBodyIndex < numSoftBodies; ++sourceBodyIndex) {
      const AvbdSoftBody &sourceBody = softBodies[sourceBodyIndex];
      const physx::PxU32 particleStart = sourceBody.compiled.particleStart;
      const physx::PxU32 particleCount = sourceBody.compiled.particleCount;
      if (sourceBody.compiled.speculativeCCDEnabled ||
          !physx::PxIsFinite(
              sourceBody.compiled.maxDepenetrationVelocity) ||
          sourceBody.compiled.maxDepenetrationVelocity < 1.0e20f ||
          particleStart > numSoftParticles ||
          particleCount == 0 ||
          particleCount > numSoftParticles - particleStart ||
          sourceBody.runtime.objectiveAdjacency.size() != particleCount)
        continue;

      physx::PxReal totalSoftMass = 0.0f;
      bool completeDynamicBody = true;
      for (physx::PxU32 localIndex = 0; localIndex < particleCount;
           ++localIndex) {
        const AvbdSoftParticle &particle =
            softParticles[particleStart + localIndex];
        if (particle.invMass <= 0.0f || !physx::PxIsFinite(particle.invMass) ||
            !physx::PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
            !particle.position.isFinite() ||
            !particle.initialPosition.isFinite() ||
            !particle.predictedPosition.isFinite() ||
            !particle.outerPosition.isFinite() ||
            !sourceBody.runtime.objectiveAdjacency[localIndex]
                 .objectiveIndices.empty()) {
          completeDynamicBody = false;
          break;
        }
        totalSoftMass += particle.mass;
      }
      if (!completeDynamicBody || !physx::PxIsFinite(totalSoftMass) ||
          totalSoftMass <= 1.0e-12f)
        continue;
      // Select exactly one deepest current-pose row for this source body per
      // sweep.  Re-evaluating after every preceding body correction gives a
      // deterministic body-level PGS ordering when two soft bodies press the
      // same free rigid from opposite sides.
      physx::PxU32 selectedContact = PX_MAX_U32;
      physx::PxReal selectedGap = 0.0f;
      physx::PxReal selectedQueryWeightSum = 0.0f;
      struct TriangleCoreGroup {
        physx::PxU32 targetIndex;
        physx::PxU64 primitiveKey;
        physx::PxU32 representativeContact;
      };
      physx::PxArray<TriangleCoreGroup> triangleCoreGroups;
      physx::PxU32 triangleCoreExitContact = PX_MAX_U32;
      physx::PxVec3 triangleCoreExitNormal(0.0f);
      physx::PxReal triangleCoreExitDistance = 0.0f;
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
        const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
        if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
            !geometry.hasRigidBodyTarget() ||
            geometry.targetIndex >= numBodies ||
            !avbdHasSoftContactDynamicQuerySupport(
                geometry, softParticles, numSoftParticles))
          continue;

        const physx::PxU32 representative =
            geometry.hasWeightedQueryPoint()
                ? geometry.queryPoint.particleIndices[0]
                : geometry.hasBarycentricQueryPoint()
                      ? geometry.queryParticleIndices[0]
                      : geometry.particleIdx;
        if (representative >= numSoftParticles)
          continue;
        const AvbdSoftBody *candidateSourceBody =
            geometry.queryBodyIndex < numSoftBodies &&
                    avbdSoftBodyContainsParticle(
                        softBodies[geometry.queryBodyIndex], representative,
                        numSoftParticles)
                ? &softBodies[geometry.queryBodyIndex]
                : avbdFindSoftBodyForParticle(softBodies, numSoftBodies,
                                               representative);
        if (candidateSourceBody != &sourceBody)
          continue;

        if (allowFreshTriangleCoreExit &&
            (locallyResolvedTriangleCoreContacts.size() != numSoftContacts ||
             locallyResolvedTriangleCoreContacts[sci] == 0u) &&
            geometry.hasRigidBoxSdf &&
            geometry.hasRigidBoxTriangleCoreExit &&
            geometry.rigidBoxPose.p.isFinite() &&
            geometry.rigidBoxPose.q.isFinite() &&
            geometry.rigidBoxHalfExtent.isFinite() &&
            geometry.rigidBoxTriangleCoreMinimumLocal.isFinite() &&
            geometry.rigidBoxTriangleCoreMaximumLocal.isFinite()) {
          bool groupExists = false;
          for (physx::PxU32 groupIndex = 0;
               groupIndex < triangleCoreGroups.size(); ++groupIndex) {
            const TriangleCoreGroup &group = triangleCoreGroups[groupIndex];
            if (group.targetIndex == geometry.targetIndex &&
                group.primitiveKey == geometry.source.primitiveKey) {
              groupExists = true;
              break;
            }
          }
          if (!groupExists) {
            TriangleCoreGroup group;
            group.targetIndex = geometry.targetIndex;
            group.primitiveKey = geometry.source.primitiveKey;
            group.representativeContact = sci;
            triangleCoreGroups.pushBack(group);
          }
        }

        physx::PxU32 queryParticleIndices[AVBD_CONTACT_MAX_PARTICLES];
        const physx::PxU32 queryParticleCount =
            avbdCollectSoftContactParticleIndices(geometry,
                                                  queryParticleIndices);
        if (queryParticleCount == 0 ||
            queryParticleCount > AVBD_CONTACT_MAX_PARTICLES)
          continue;
        physx::PxReal queryWeightSum = 0.0f;
        bool validQueryWeights = true;
        for (physx::PxU32 pi = 0; pi < queryParticleCount; ++pi) {
          const physx::PxU32 particleIndex = queryParticleIndices[pi];
          const physx::PxReal weight =
              avbdGetSoftContactParticleJacobianScale(geometry,
                                                       particleIndex);
          if (particleIndex >= numSoftParticles ||
              !avbdSoftBodyContainsParticle(sourceBody, particleIndex,
                                            numSoftParticles) ||
              !physx::PxIsFinite(weight)) {
            validQueryWeights = false;
            break;
          }
          queryWeightSum += weight;
        }
        if (!validQueryWeights || !physx::PxIsFinite(queryWeightSum) ||
            queryWeightSum <= 1.0e-6f)
          continue;

        const AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
        if (rigidBody.invMass <= 0.0f || !rigidBody.position.isFinite() ||
            !rigidBody.rotation.isFinite())
          continue;
        const physx::PxVec3 queryPoint =
            avbdGetSoftContactQueryPoint(geometry, softParticles);
        if (!queryPoint.isFinite())
          continue;
        physx::PxVec3 normal(0.0f);
        physx::PxVec3 worldOffset(0.0f);
        physx::PxReal trueGap = 0.0f;
        if (!getCurrentDynamicSoftRigidContactGeometry(
                geometry, rigidBody, queryPoint, normal, worldOffset,
                trueGap))
          continue;
        if (!physx::PxIsFinite(trueGap))
          continue;

        if (trueGap < -1.0e-5f &&
            (selectedContact == PX_MAX_U32 || trueGap < selectedGap)) {
          selectedContact = sci;
          selectedGap = trueGap;
          selectedQueryWeightSum = queryWeightSum;
        }

      }

      // A body can be crossed by core triangles on opposite box faces.  That
      // does not make a coherent escape impossible: evaluate all six common
      // box-face exits, take the maximum required distance over the complete
      // triangle set for each face, then choose the smallest valid aggregate.
      // This is an endpoint/current-pose DCD certificate, not a per-row AL
      // shell depth or a swept test.
      if (allowFreshTriangleCoreExit) {
        const physx::PxReal exitTolerance = physx::PxMax(
            1.0e-6f, 1.0e-5f * lengthScale);
        physx::PxReal bestExitDistance = PX_MAX_F32;
        for (physx::PxU32 groupIndex = 0;
             groupIndex < triangleCoreGroups.size(); ++groupIndex) {
          const TriangleCoreGroup &group = triangleCoreGroups[groupIndex];
          if (group.targetIndex >= numBodies ||
              group.representativeContact >= numSoftContacts)
            continue;
          const AvbdSoftContactGeometry &carrier =
              softContacts[group.representativeContact].geometry;
          AvbdSolverBody &carrierBody = bodies[group.targetIndex];
          if (carrierBody.invMass <= 0.0f ||
              !physx::PxIsFinite(carrierBody.invMass) ||
              !carrierBody.position.isFinite() ||
              !carrierBody.rotation.isFinite() ||
              !carrier.rigidBoxPose.p.isFinite() ||
              !carrier.rigidBoxPose.q.isFinite())
            continue;
          const physx::PxReal rotationSinceEndpoint =
              computeRotationDeltaMagnitude(
                  carrierBody.rotation,
                  endpointRigidRotations[group.targetIndex]);
          if (!physx::PxIsFinite(rotationSinceEndpoint) ||
              rotationSinceEndpoint > 1.0e-6f)
            continue;
          const physx::PxQuat shapeRotation =
              carrierBody.rotation * carrier.rigidBoxPose.q;
          if (!shapeRotation.isFinite())
            continue;

          physx::PxReal aggregateExit[6] = {
              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
          bool validGroup = true;
          bool hasCoreTriangle = false;
          for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
            const AvbdSoftContactGeometry &geometry =
                softContacts[sci].geometry;
            if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
                !geometry.hasRigidBodyTarget() ||
                geometry.targetIndex != group.targetIndex ||
                geometry.source.primitiveKey != group.primitiveKey ||
                (locallyResolvedTriangleCoreContacts.size() ==
                     numSoftContacts &&
                 locallyResolvedTriangleCoreContacts[sci] != 0u) ||
                !geometry.hasRigidBoxSdf ||
                !geometry.hasRigidBoxTriangleCoreExit)
              continue;
            const physx::PxU32 representative =
                geometry.hasWeightedQueryPoint()
                    ? geometry.queryPoint.particleIndices[0]
                    : geometry.hasBarycentricQueryPoint()
                          ? geometry.queryParticleIndices[0]
                          : geometry.particleIdx;
            if (representative >= numSoftParticles)
              continue;
            const AvbdSoftBody *candidateSourceBody =
                geometry.queryBodyIndex < numSoftBodies &&
                        avbdSoftBodyContainsParticle(
                            softBodies[geometry.queryBodyIndex],
                            representative, numSoftParticles)
                    ? &softBodies[geometry.queryBodyIndex]
                    : avbdFindSoftBodyForParticle(
                          softBodies, numSoftBodies, representative);
            if (candidateSourceBody != &sourceBody)
              continue;
            // A rigid body may have several shapes.  A primitive group must
            // use one immutable local OBB frame; otherwise fail closed.
            if (geometry.rigidBoxPose.p != carrier.rigidBoxPose.p ||
                !(geometry.rigidBoxPose.q == carrier.rigidBoxPose.q) ||
                geometry.rigidBoxHalfExtent != carrier.rigidBoxHalfExtent) {
              validGroup = false;
              break;
            }
            hasCoreTriangle = true;
            const physx::PxVec3 relativeTranslation =
                endpointSoftTranslations[sourceBodyIndex] -
                endpointRigidTranslations[group.targetIndex];
            if (!relativeTranslation.isFinite()) {
              validGroup = false;
              break;
            }
            for (physx::PxU32 face = 0; face < 6u; ++face) {
              physx::PxReal rawDistance = 0.0f;
              if (!getRigidBoxTriangleCoreExitDistance(
                      geometry, face, rawDistance)) {
                validGroup = false;
                break;
              }
              const physx::PxVec3 worldNormal = shapeRotation.rotate(
                  getRigidBoxTriangleCoreExitNormalLocal(face));
              const physx::PxReal normalLengthSq =
                  worldNormal.magnitudeSquared();
              if (!worldNormal.isFinite() ||
                  !physx::PxIsFinite(normalLengthSq) ||
                  normalLengthSq <= 1.0e-12f) {
                validGroup = false;
                break;
              }
              const physx::PxVec3 normal =
                  worldNormal * physx::PxRecipSqrt(normalLengthSq);
              const physx::PxReal residual = physx::PxMax(
                  0.0f, rawDistance - relativeTranslation.dot(normal));
              if (!physx::PxIsFinite(residual)) {
                validGroup = false;
                break;
              }
              aggregateExit[face] =
                  physx::PxMax(aggregateExit[face], residual);
            }
            if (!validGroup)
              break;
          }
          if (!validGroup || !hasCoreTriangle)
            continue;

          // Any one support face with zero residual proves that *every*
          // triangle in this group is already outside that face.  Do not
          // select another still-positive face on a later PGS sweep: that
          // would undo the completed certificate and make opposite faces
          // ping-pong the free rigid between the two soft bodies.
          bool groupAlreadySeparated = false;
          for (physx::PxU32 face = 0; face < 6u; ++face) {
            if (aggregateExit[face] <= exitTolerance) {
              groupAlreadySeparated = true;
              break;
            }
          }
          if (groupAlreadySeparated)
            continue;

          for (physx::PxU32 face = 0; face < 6u; ++face) {
            const physx::PxReal candidateExit = aggregateExit[face];
            if (!(candidateExit > exitTolerance) ||
                candidateExit >= bestExitDistance)
              continue;
            const physx::PxVec3 worldNormal = shapeRotation.rotate(
                getRigidBoxTriangleCoreExitNormalLocal(face));
            const physx::PxReal normalLengthSq =
                worldNormal.magnitudeSquared();
            if (!worldNormal.isFinite() ||
                !physx::PxIsFinite(normalLengthSq) ||
                normalLengthSq <= 1.0e-12f)
              continue;
            bestExitDistance = candidateExit;
            triangleCoreExitContact = group.representativeContact;
            triangleCoreExitDistance = candidateExit;
            triangleCoreExitNormal =
                worldNormal * physx::PxRecipSqrt(normalLengthSq);
          }
        }
      }

      const bool useTriangleCoreExit =
          triangleCoreExitContact != PX_MAX_U32;
      if (useTriangleCoreExit) {
        selectedContact = triangleCoreExitContact;
        selectedGap = -triangleCoreExitDistance;
        // A uniform soft-body translation moves the complete collision
        // triangle by the translation itself, not by an expanded query's
        // incidental weight sum. The certificate must therefore use one.
        selectedQueryWeightSum = 1.0f;
      }

      if (selectedContact == PX_MAX_U32)
        continue;

      const AvbdSoftContactGeometry &geometry =
          softContacts[selectedContact].geometry;

      const physx::PxReal softTranslationResponse = 1.0f / totalSoftMass;
      // This endpoint phase is the first dynamic pair owner.  Validate that
      // the selected manifold row belongs to the selection-owned pair before
      // moving either endpoint; no per-row recovery may manufacture a second
      // soft/rigid ownership domain.
      AvbdOgcPairState *pairState = nullptr;
      if (ogcPairStates || ogcPairIndices) {
        if (!ogcPairStates || !ogcPairIndices ||
            numOgcPairIndices != numSoftContacts ||
            selectedContact >= numOgcPairIndices)
          continue;
        const physx::PxU32 pairIndex = ogcPairIndices[selectedContact];
        if (pairIndex >= numOgcPairStates)
          continue;
        pairState = &ogcPairStates[pairIndex];
        if (!pairState->active ||
            pairState->sourceBodyIndex != sourceBodyIndex ||
            pairState->targetBodyIndex != geometry.targetIndex ||
            pairState->primitiveKey != geometry.source.primitiveKey)
          continue;
      }
      AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
      if (rigidBody.invMass <= 0.0f ||
          !physx::PxIsFinite(rigidBody.invMass) ||
          !rigidBody.position.isFinite() || !rigidBody.rotation.isFinite() ||
          !rigidBody.invInertiaWorld.column0.isFinite() ||
          !rigidBody.invInertiaWorld.column1.isFinite() ||
          !rigidBody.invInertiaWorld.column2.isFinite())
        continue;
      const physx::PxVec3 queryPoint =
          avbdGetSoftContactQueryPoint(geometry, softParticles);
      if (!queryPoint.isFinite())
        continue;
      physx::PxVec3 normal(0.0f);
      physx::PxVec3 worldOffset(0.0f);
      physx::PxReal trueGap = 0.0f;
      if (useTriangleCoreExit) {
        normal = triangleCoreExitNormal;
        worldOffset = physx::PxVec3(0.0f);
        trueGap = -triangleCoreExitDistance;
      } else {
        if (!getCurrentDynamicSoftRigidContactGeometry(
                geometry, rigidBody, queryPoint, normal, worldOffset,
                trueGap))
          continue;
        if (!physx::PxIsFinite(trueGap) || trueGap >= -1.0e-5f)
          continue;
      }

      physx::PxVec3 rigidLinearJacobian = -normal;
      rigidBody.projectLockedLinearVector(rigidLinearJacobian);
      const physx::PxVec3 rigidLinearDeltaPerLambda =
          rigidLinearJacobian * rigidBody.invMass;
      // The certificate is a complete-triangle geometric escape. Keep its
      // paired rigid response translational so the OBB orientation stays the
      // one used to certify separation; ordinary AL rows still own torque.
      physx::PxVec3 rigidAngularJacobian = useTriangleCoreExit
          ? physx::PxVec3(0.0f) : -worldOffset.cross(normal);
      rigidBody.projectLockedAngularVector(rigidAngularJacobian);
      physx::PxVec3 rigidAngularDeltaPerLambda =
          rigidBody.invInertiaWorld * rigidAngularJacobian;
      rigidBody.projectLockedAngularVector(rigidAngularDeltaPerLambda);
      const physx::PxReal rigidResponse =
          rigidLinearJacobian.dot(rigidLinearDeltaPerLambda) +
          rigidAngularJacobian.dot(rigidAngularDeltaPerLambda);
      const physx::PxReal weightedSoftTranslationResponse =
          selectedQueryWeightSum * selectedQueryWeightSum *
          softTranslationResponse;
      const physx::PxReal response =
          weightedSoftTranslationResponse + rigidResponse;
      if (!rigidLinearDeltaPerLambda.isFinite() ||
          !rigidAngularDeltaPerLambda.isFinite() ||
          !physx::PxIsFinite(response) || response <= 1.0e-12f)
        continue;

      const physx::PxReal lambda = -trueGap / response;
      if (!physx::PxIsFinite(lambda) || lambda <= 0.0f)
        continue;
      const physx::PxVec3 rawSoftDelta =
          normal * (selectedQueryWeightSum * softTranslationResponse *
                    lambda);
      const physx::PxVec3 rawAngularDelta =
          rigidAngularDeltaPerLambda * lambda;
      if (!rawSoftDelta.isFinite() || !rawAngularDelta.isFinite())
        continue;

      // A free, very light box can have a small angular inertia.  Keep every
      // generalized endpoint correction on one common scale if its rotation
      // would otherwise jump too far in this emergency stage.
      physx::PxReal commonAlpha = 1.0f;
      const physx::PxReal rawAngularMagnitude = rawAngularDelta.magnitude();
      if (!physx::PxIsFinite(rawAngularMagnitude))
        continue;
      if (rawAngularMagnitude > maxAngularStep)
        commonAlpha = maxAngularStep / rawAngularMagnitude;
      if (!(commonAlpha > 0.0f) || !physx::PxIsFinite(commonAlpha))
        continue;

      const physx::PxVec3 rigidPositionBefore = rigidBody.position;
      const physx::PxQuat rigidRotationBefore = rigidBody.rotation;
      physx::PxVec3 acceptedSoftDelta(0.0f);
      AvbdSolverBody acceptedRigidBody = rigidBody;
      bool acceptedCandidate = false;
      for (physx::PxU32 attempt = 0; attempt < 8u && !acceptedCandidate;
           ++attempt) {
        const physx::PxVec3 softDelta = rawSoftDelta * commonAlpha;
        AvbdSolverBody candidateRigidBody = rigidBody;
        candidateRigidBody.position +=
            rigidLinearDeltaPerLambda * (lambda * commonAlpha);
        const physx::PxVec3 angularDelta = rawAngularDelta * commonAlpha;
        if (angularDelta.magnitudeSquared() > 1.0e-16f) {
          const physx::PxQuat dq(angularDelta.x, angularDelta.y,
                                  angularDelta.z, 0.0f);
          candidateRigidBody.rotation =
              (candidateRigidBody.rotation +
               dq * candidateRigidBody.rotation * 0.5f)
                  .getNormalized();
        }
        candidateRigidBody.projectLockedPose(rigidPositionBefore,
                                              rigidRotationBefore);
        physx::PxQuat actualRotationDelta =
            candidateRigidBody.rotation * rigidRotationBefore.getConjugate();
        if (actualRotationDelta.w < 0.0f)
          actualRotationDelta = -actualRotationDelta;
        if (actualRotationDelta.isFinite()) {
          actualRotationDelta.normalize();
          const physx::PxMat33 rotationDelta(actualRotationDelta);
          candidateRigidBody.invInertiaWorld =
              rotationDelta * rigidBody.invInertiaWorld *
              rotationDelta.getTranspose();
        }

        bool finiteCandidates = actualRotationDelta.isFinite() &&
            softDelta.isFinite() && candidateRigidBody.position.isFinite() &&
            candidateRigidBody.rotation.isFinite() &&
            candidateRigidBody.invInertiaWorld.column0.isFinite() &&
            candidateRigidBody.invInertiaWorld.column1.isFinite() &&
            candidateRigidBody.invInertiaWorld.column2.isFinite();
        for (physx::PxU32 localIndex = 0;
             localIndex < particleCount && finiteCandidates; ++localIndex) {
          const AvbdSoftParticle &particle =
              softParticles[particleStart + localIndex];
          finiteCandidates = (particle.position + softDelta).isFinite() &&
              (particle.initialPosition + softDelta).isFinite() &&
              (particle.predictedPosition + softDelta).isFinite() &&
              (particle.outerPosition + softDelta).isFinite();
        }

        // Locks and the finite quaternion update can make the actual point
        // response differ slightly from the linearized denominator.  Commit
        // only a candidate which demonstrably increases the current true
        // surface gap; otherwise retry the same generalized correction at a
        // smaller common scale.
        bool candidateImprovesGap = false;
        if (useTriangleCoreExit) {
          const physx::PxVec3 rigidTranslation =
              candidateRigidBody.position - rigidPositionBefore;
          const physx::PxReal achievedExit =
              (softDelta - rigidTranslation).dot(normal);
          const physx::PxReal exitTolerance = physx::PxMax(
              1.0e-6f, 1.0e-5f * lengthScale);
          candidateImprovesGap = physx::PxIsFinite(achievedExit) &&
              achievedExit >= triangleCoreExitDistance - exitTolerance;
        } else {
          const physx::PxVec3 candidateQueryPoint =
              queryPoint + softDelta * selectedQueryWeightSum;
          physx::PxVec3 candidateNormal(0.0f);
          physx::PxVec3 candidateWorldOffset(0.0f);
          physx::PxReal candidateGap = 0.0f;
          const bool candidateQueryValid =
              candidateQueryPoint.isFinite() &&
              getCurrentDynamicSoftRigidContactGeometry(
                  geometry, candidateRigidBody, candidateQueryPoint,
                  candidateNormal, candidateWorldOffset, candidateGap);
          const physx::PxReal gapImprovementTolerance = physx::PxMax(
              1.0e-6f, 1.0e-5f * lengthScale);
          candidateImprovesGap = candidateQueryValid &&
              physx::PxIsFinite(candidateGap) &&
              candidateGap > trueGap + gapImprovementTolerance;
        }
        if (finiteCandidates && candidateImprovesGap) {
          acceptedSoftDelta = softDelta;
          acceptedRigidBody = candidateRigidBody;
          acceptedCandidate = true;
        } else {
          commonAlpha *= 0.5f;
        }
      }
      if (!acceptedCandidate)
        continue;

      const physx::PxVec3 acceptedRigidTranslation =
          acceptedRigidBody.position - rigidPositionBefore;
      const physx::PxVec3 accumulatedSoftTranslation =
          endpointSoftTranslations[sourceBodyIndex] + acceptedSoftDelta;
      const physx::PxVec3 accumulatedRigidTranslation =
          endpointRigidTranslations[geometry.targetIndex] +
          acceptedRigidTranslation;
      if (!acceptedRigidTranslation.isFinite() ||
          !accumulatedSoftTranslation.isFinite() ||
          !accumulatedRigidTranslation.isFinite())
        continue;

      // Uniform translation preserves every soft-body deformation gradient.
      // Translate the velocity/inertial anchors along with the endpoint so
      // this geometric projection is not reconstructed as a spurious bounce.
      for (physx::PxU32 localIndex = 0; localIndex < particleCount;
           ++localIndex) {
        AvbdSoftParticle &particle =
            softParticles[particleStart + localIndex];
        particle.position += acceptedSoftDelta;
        particle.initialPosition += acceptedSoftDelta;
        particle.predictedPosition += acceptedSoftDelta;
        particle.outerPosition += acceptedSoftDelta;
      }
      rigidBody.position = acceptedRigidBody.position;
      rigidBody.rotation = acceptedRigidBody.rotation;
      rigidBody.invInertiaWorld = acceptedRigidBody.invInertiaWorld;
      endpointSoftTranslations[sourceBodyIndex] = accumulatedSoftTranslation;
      endpointRigidTranslations[geometry.targetIndex] =
          accumulatedRigidTranslation;
      if (pairState) {
        // A pair epoch advances only after a transactional, paired 6DOF/soft
        // update.  This state is deliberately group-scoped, so the many
        // contact rows of a triangle manifold cannot each reset the epoch.
        ++pairState->epoch;
      }
      if (recoveredContacts && selectedContact < recoveredContacts->size()) {
        // Keep one endpoint velocity row per full soft body.  Several feature
        // rows can describe the same face; applying a body-wide e=0 impulse
        // once per such row would no longer match this generalized projector.
        for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
          if (((*recoveredContacts)[sci] & 2u) == 0u)
            continue;
          const AvbdSoftContactGeometry &prior = softContacts[sci].geometry;
          const physx::PxU32 representative =
              prior.hasWeightedQueryPoint()
                  ? prior.queryPoint.particleIndices[0]
                  : prior.hasBarycentricQueryPoint()
                        ? prior.queryParticleIndices[0]
                        : prior.particleIdx;
          if (representative >= numSoftParticles)
            continue;
          const AvbdSoftBody *priorSourceBody =
              prior.queryBodyIndex < numSoftBodies &&
                      avbdSoftBodyContainsParticle(
                          softBodies[prior.queryBodyIndex], representative,
                          numSoftParticles)
                  ? &softBodies[prior.queryBodyIndex]
                  : avbdFindSoftBodyForParticle(softBodies, numSoftBodies,
                                                 representative);
          if (priorSourceBody == &sourceBody)
            (*recoveredContacts)[sci] &= ~2u;
        }
        // Endpoint recovery supersedes a local recovery of the same row: the
        // final e=0 impulse must use this body's generalized mass response.
        (*recoveredContacts)[selectedContact] = 2u;
      }
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

void AvbdSolver::clampDynamicSoftRigidBodyEndpointVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxArray<physx::PxU8> *recoveredContacts,
    AvbdSolverStats *stats) {
  (void)stats;
  if (!bodies || numBodies == 0 || !softParticles ||
      numSoftParticles == 0 || !softBodies || numSoftBodies == 0 ||
      !softContacts || numSoftContacts == 0 || !recoveredContacts ||
      recoveredContacts->size() != numSoftContacts)
    return;

  const physx::PxReal nearSurfaceTolerance = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));

  // A whole-body endpoint recovery is only a geometrical escape hatch.  Its
  // final e=0 velocity response must nevertheless remain a *shared* mixed
  // OGC solve: applying the upper and lower rows serially gives the light
  // rigid two independent kicks and is the source of the one-frame lateral
  // ejection seen in a soft/rigid/soft squeeze.  Build one representative per
  // (source soft body, target rigid shape) pair and solve all representatives
  // for a target rigid as one small projected Schur block.  The rigid cross
  // terms retain the common 6DOF response, so opposing normals cancel at the
  // rigid while their corresponding soft bodies receive their own response.
  struct EndpointVelocityPairRow {
    physx::PxU32 contactIndex;
    physx::PxU32 sourceBodyIndex;
    physx::PxU64 primitiveKey;
    physx::PxReal totalSoftMass;
    physx::PxReal queryWeightSum;
    physx::PxVec3 queryVelocity;
    physx::PxVec3 normal;
    physx::PxVec3 worldOffset;
    physx::PxVec3 rigidLinearJacobian;
    physx::PxVec3 rigidAngularJacobian;
    physx::PxReal relativeNormalVelocity;
  };
  physx::PxArray<physx::PxU8> manifoldHandled(numSoftContacts, 0u);

  auto buildEndpointVelocityPairRow =
      [&](physx::PxU32 sci, EndpointVelocityPairRow &row) -> bool {
    if (sci >= numSoftContacts)
      return false;
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() || geometry.targetIndex >= numBodies ||
        geometry.queryBodyIndex >= numSoftBodies ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      return false;
    const AvbdSoftBody &sourceBody = softBodies[geometry.queryBodyIndex];
    if (sourceBody.compiled.speculativeCCDEnabled ||
        !physx::PxIsFinite(sourceBody.compiled.maxDepenetrationVelocity) ||
        sourceBody.compiled.maxDepenetrationVelocity < 1.0e20f)
      return false;
    const physx::PxU32 particleStart = sourceBody.compiled.particleStart;
    const physx::PxU32 particleCount = sourceBody.compiled.particleCount;
    if (particleStart > numSoftParticles || particleCount == 0u ||
        particleCount > numSoftParticles - particleStart ||
        sourceBody.runtime.objectiveAdjacency.size() != particleCount)
      return false;
    physx::PxReal totalSoftMass = 0.0f;
    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex) {
      const AvbdSoftParticle &particle =
          softParticles[particleStart + localIndex];
      if (particle.invMass <= 0.0f || !physx::PxIsFinite(particle.invMass) ||
          !physx::PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
          !particle.velocity.isFinite() ||
          !sourceBody.runtime.objectiveAdjacency[localIndex]
               .objectiveIndices.empty())
        return false;
      totalSoftMass += particle.mass;
    }
    if (!physx::PxIsFinite(totalSoftMass) || totalSoftMass <= 1.0e-12f)
      return false;

    physx::PxU32 queryParticleIndices[AVBD_CONTACT_MAX_PARTICLES];
    const physx::PxU32 queryParticleCount =
        avbdCollectSoftContactParticleIndices(geometry, queryParticleIndices);
    if (queryParticleCount == 0u ||
        queryParticleCount > AVBD_CONTACT_MAX_PARTICLES)
      return false;
    physx::PxReal queryWeightSum = 0.0f;
    physx::PxVec3 queryVelocity(0.0f);
    for (physx::PxU32 pi = 0; pi < queryParticleCount; ++pi) {
      const physx::PxU32 particleIndex = queryParticleIndices[pi];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      if (particleIndex >= numSoftParticles ||
          !avbdSoftBodyContainsParticle(sourceBody, particleIndex,
                                        numSoftParticles) ||
          !physx::PxIsFinite(weight) ||
          !softParticles[particleIndex].velocity.isFinite())
        return false;
      queryWeightSum += weight;
      queryVelocity += softParticles[particleIndex].velocity * weight;
    }
    if (!physx::PxIsFinite(queryWeightSum) || queryWeightSum <= 1.0e-6f ||
        !queryVelocity.isFinite())
      return false;

    AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
    if (rigidBody.invMass <= 0.0f || !rigidBody.position.isFinite() ||
        !rigidBody.rotation.isFinite() ||
        !rigidBody.linearVelocity.isFinite() ||
        !rigidBody.angularVelocity.isFinite() ||
        !rigidBody.invInertiaWorld.column0.isFinite() ||
        !rigidBody.invInertiaWorld.column1.isFinite() ||
        !rigidBody.invInertiaWorld.column2.isFinite())
      return false;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    physx::PxVec3 normal(0.0f), worldOffset(0.0f);
    physx::PxReal trueGap = 0.0f;
    if (!queryPoint.isFinite() ||
        !getCurrentDynamicSoftRigidContactGeometry(
            geometry, rigidBody, queryPoint, normal, worldOffset, trueGap) ||
        !physx::PxIsFinite(trueGap) || trueGap > nearSurfaceTolerance)
      return false;
    physx::PxVec3 rigidLinearJacobian = -normal;
    rigidBody.projectLockedLinearVector(rigidLinearJacobian);
    physx::PxVec3 rigidAngularJacobian = -worldOffset.cross(normal);
    rigidBody.projectLockedAngularVector(rigidAngularJacobian);
    const physx::PxVec3 rigidSurfaceVelocity =
        rigidBody.linearVelocity + rigidBody.angularVelocity.cross(worldOffset);
    const physx::PxReal relativeNormalVelocity =
        (queryVelocity - rigidSurfaceVelocity).dot(normal);
    if (!rigidLinearJacobian.isFinite() || !rigidAngularJacobian.isFinite() ||
        !physx::PxIsFinite(relativeNormalVelocity))
      return false;
    row.contactIndex = sci;
    row.sourceBodyIndex = geometry.queryBodyIndex;
    row.primitiveKey = geometry.source.primitiveKey;
    row.totalSoftMass = totalSoftMass;
    row.queryWeightSum = queryWeightSum;
    row.queryVelocity = queryVelocity;
    row.normal = normal;
    row.worldOffset = worldOffset;
    row.rigidLinearJacobian = rigidLinearJacobian;
    row.rigidAngularJacobian = rigidAngularJacobian;
    row.relativeNormalVelocity = relativeNormalVelocity;
    return true;
  };

  for (physx::PxU32 targetIndex = 0; targetIndex < numBodies;
       ++targetIndex) {
    physx::PxArray<EndpointVelocityPairRow> rows;
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
      if (!geometry.hasRigidBodyTarget() || geometry.targetIndex != targetIndex)
        continue;
      EndpointVelocityPairRow candidate;
      if (!buildEndpointVelocityPairRow(sci, candidate))
        continue;
      bool replaced = false;
      for (physx::PxU32 rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        EndpointVelocityPairRow &existing = rows[rowIndex];
        if (existing.sourceBodyIndex == candidate.sourceBodyIndex &&
            existing.primitiveKey == candidate.primitiveKey) {
          if (candidate.relativeNormalVelocity <
              existing.relativeNormalVelocity)
            existing = candidate;
          replaced = true;
          break;
        }
      }
      if (!replaced)
        rows.pushBack(candidate);
    }
    if (rows.size() < 2u)
      continue;

    // A fixed tiny block is enough for the intended soft/rigid/soft manifold;
    // leave unusually large manifolds to the scalar robust fallback below.
    if (rows.size() > 8u)
      continue;
    bool hasDistinctSources = false;
    for (physx::PxU32 i = 1; i < rows.size(); ++i)
      hasDistinctSources = hasDistinctSources ||
          rows[i].sourceBodyIndex != rows[0].sourceBodyIndex;
    if (!hasDistinctSources)
      continue;

    AvbdSolverBody &rigidBody = bodies[targetIndex];
    physx::PxArray<physx::PxReal> matrix(rows.size() * rows.size(), 0.0f);
    physx::PxArray<physx::PxReal> impulses(rows.size(), 0.0f);
    bool validBlock = true;
    for (physx::PxU32 i = 0; i < rows.size() && validBlock; ++i) {
      for (physx::PxU32 j = 0; j < rows.size(); ++j) {
        const EndpointVelocityPairRow &a = rows[i];
        const EndpointVelocityPairRow &b = rows[j];
        physx::PxReal response = 0.0f;
        if (a.sourceBodyIndex == b.sourceBodyIndex)
          response += a.queryWeightSum * b.queryWeightSum /
              a.totalSoftMass;
        const physx::PxVec3 rigidAngularResponse =
            rigidBody.invInertiaWorld * b.rigidAngularJacobian;
        response += a.rigidLinearJacobian.dot(
                        b.rigidLinearJacobian * rigidBody.invMass) +
            a.rigidAngularJacobian.dot(rigidAngularResponse);
        if (!physx::PxIsFinite(response)) {
          validBlock = false;
          break;
        }
        matrix[i * rows.size() + j] = response;
      }
      if (matrix[i * rows.size() + i] <= 1.0e-12f)
        validBlock = false;
    }
    if (!validBlock)
      continue;

    // Projected Gauss--Seidel on the dense pair Schur complement.  This is
    // simultaneous at the rigid endpoint even though the small linear system
    // is iterated for deterministic unilateral clamping.
    for (physx::PxU32 iteration = 0; iteration < 6u; ++iteration) {
      for (physx::PxU32 i = 0; i < rows.size(); ++i) {
        physx::PxReal relativeVelocity = rows[i].relativeNormalVelocity;
        for (physx::PxU32 j = 0; j < rows.size(); ++j)
          relativeVelocity += matrix[i * rows.size() + j] * impulses[j];
        const physx::PxReal diagonal = matrix[i * rows.size() + i];
        impulses[i] = physx::PxMax(
            0.0f, impulses[i] - relativeVelocity / diagonal);
      }
    }

    physx::PxArray<physx::PxVec3> softVelocityDeltas(numSoftBodies,
                                                      physx::PxVec3(0.0f));
    physx::PxVec3 rigidLinearVelocityDelta(0.0f);
    physx::PxVec3 rigidAngularVelocityDelta(0.0f);
    bool anyImpulse = false;
    for (physx::PxU32 i = 0; i < rows.size(); ++i) {
      const physx::PxReal impulse = impulses[i];
      if (!physx::PxIsFinite(impulse) || impulse <= 1.0e-8f)
        continue;
      const EndpointVelocityPairRow &row = rows[i];
      softVelocityDeltas[row.sourceBodyIndex] += row.normal *
          (row.queryWeightSum * impulse / row.totalSoftMass);
      rigidLinearVelocityDelta +=
          row.rigidLinearJacobian * (rigidBody.invMass * impulse);
      rigidAngularVelocityDelta +=
          (rigidBody.invInertiaWorld * row.rigidAngularJacobian) * impulse;
      anyImpulse = true;
    }
    if (!anyImpulse || !rigidLinearVelocityDelta.isFinite() ||
        !rigidAngularVelocityDelta.isFinite())
      continue;

    physx::PxVec3 candidateLinearVelocity =
        rigidBody.linearVelocity + rigidLinearVelocityDelta;
    physx::PxVec3 candidateAngularVelocity =
        rigidBody.angularVelocity + rigidAngularVelocityDelta;
    rigidBody.projectLockedLinearVector(candidateLinearVelocity);
    rigidBody.projectLockedAngularVector(candidateAngularVelocity);
    validBlock = candidateLinearVelocity.isFinite() &&
        candidateAngularVelocity.isFinite();
    for (physx::PxU32 sourceIndex = 0;
         sourceIndex < numSoftBodies && validBlock; ++sourceIndex) {
      const physx::PxVec3 &delta = softVelocityDeltas[sourceIndex];
      if (delta.magnitudeSquared() == 0.0f)
        continue;
      const AvbdSoftBody &sourceBody = softBodies[sourceIndex];
      const physx::PxU32 particleStart = sourceBody.compiled.particleStart;
      const physx::PxU32 particleCount = sourceBody.compiled.particleCount;
      if (particleStart > numSoftParticles ||
          particleCount > numSoftParticles - particleStart) {
        validBlock = false;
        break;
      }
      for (physx::PxU32 localIndex = 0; localIndex < particleCount;
           ++localIndex) {
        if (!(softParticles[particleStart + localIndex].velocity + delta)
                 .isFinite()) {
          validBlock = false;
          break;
        }
      }
    }
    if (!validBlock)
      continue;

    for (physx::PxU32 sourceIndex = 0; sourceIndex < numSoftBodies;
         ++sourceIndex) {
      const physx::PxVec3 &delta = softVelocityDeltas[sourceIndex];
      if (delta.magnitudeSquared() == 0.0f)
        continue;
      const AvbdSoftBody &sourceBody = softBodies[sourceIndex];
      for (physx::PxU32 localIndex = 0;
           localIndex < sourceBody.compiled.particleCount; ++localIndex)
        softParticles[sourceBody.compiled.particleStart + localIndex].velocity +=
            delta;
    }
    rigidBody.linearVelocity = candidateLinearVelocity;
    rigidBody.angularVelocity = candidateAngularVelocity;
    rigidBody.projectLockedVelocities();
    for (physx::PxU32 rowIndex = 0; rowIndex < rows.size(); ++rowIndex)
      manifoldHandled[rows[rowIndex].contactIndex] = 1u;
  }

  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    if (manifoldHandled[sci] != 0u)
      continue;
    if (((*recoveredContacts)[sci] & 2u) == 0u)
      continue;

    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() || geometry.targetIndex >= numBodies ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    const physx::PxU32 representative =
        geometry.hasWeightedQueryPoint()
            ? geometry.queryPoint.particleIndices[0]
            : geometry.hasBarycentricQueryPoint()
                  ? geometry.queryParticleIndices[0]
                  : geometry.particleIdx;
    if (representative >= numSoftParticles)
      continue;
    const AvbdSoftBody *sourceBody =
        geometry.queryBodyIndex < numSoftBodies &&
                avbdSoftBodyContainsParticle(
                    softBodies[geometry.queryBodyIndex], representative,
                    numSoftParticles)
            ? &softBodies[geometry.queryBodyIndex]
            : avbdFindSoftBodyForParticle(softBodies, numSoftBodies,
                                           representative);
    if (!sourceBody || sourceBody->compiled.speculativeCCDEnabled ||
        !physx::PxIsFinite(sourceBody->compiled.maxDepenetrationVelocity) ||
        sourceBody->compiled.maxDepenetrationVelocity < 1.0e20f)
      continue;

    const physx::PxU32 particleStart = sourceBody->compiled.particleStart;
    const physx::PxU32 particleCount = sourceBody->compiled.particleCount;
    if (particleStart > numSoftParticles || particleCount == 0 ||
        particleCount > numSoftParticles - particleStart ||
        sourceBody->runtime.objectiveAdjacency.size() != particleCount)
      continue;
    physx::PxReal totalSoftMass = 0.0f;
    bool completeDynamicBody = true;
    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex) {
      const AvbdSoftParticle &particle =
          softParticles[particleStart + localIndex];
      if (particle.invMass <= 0.0f || !physx::PxIsFinite(particle.invMass) ||
          !physx::PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
          !particle.velocity.isFinite() ||
          !sourceBody->runtime.objectiveAdjacency[localIndex]
               .objectiveIndices.empty()) {
        completeDynamicBody = false;
        break;
      }
      totalSoftMass += particle.mass;
    }
    if (!completeDynamicBody || !physx::PxIsFinite(totalSoftMass) ||
        totalSoftMass <= 1.0e-12f)
      continue;

    physx::PxU32 queryParticleIndices[AVBD_CONTACT_MAX_PARTICLES];
    const physx::PxU32 queryParticleCount =
        avbdCollectSoftContactParticleIndices(geometry, queryParticleIndices);
    if (queryParticleCount == 0 ||
        queryParticleCount > AVBD_CONTACT_MAX_PARTICLES)
      continue;
    physx::PxReal queryWeightSum = 0.0f;
    physx::PxVec3 queryVelocity(0.0f);
    bool validQuery = true;
    for (physx::PxU32 pi = 0; pi < queryParticleCount; ++pi) {
      const physx::PxU32 particleIndex = queryParticleIndices[pi];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      if (particleIndex >= numSoftParticles ||
          !avbdSoftBodyContainsParticle(*sourceBody, particleIndex,
                                        numSoftParticles) ||
          !physx::PxIsFinite(weight) ||
          !softParticles[particleIndex].velocity.isFinite()) {
        validQuery = false;
        break;
      }
      queryWeightSum += weight;
      queryVelocity += softParticles[particleIndex].velocity * weight;
    }
    if (!validQuery || !physx::PxIsFinite(queryWeightSum) ||
        queryWeightSum <= 1.0e-6f || !queryVelocity.isFinite())
      continue;

    AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
    if (rigidBody.invMass <= 0.0f || !rigidBody.position.isFinite() ||
        !rigidBody.rotation.isFinite() || !rigidBody.linearVelocity.isFinite() ||
        !rigidBody.angularVelocity.isFinite() ||
        !rigidBody.invInertiaWorld.column0.isFinite() ||
        !rigidBody.invInertiaWorld.column1.isFinite() ||
        !rigidBody.invInertiaWorld.column2.isFinite())
      continue;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!queryPoint.isFinite())
      continue;
    physx::PxVec3 normal(0.0f);
    physx::PxVec3 worldOffset(0.0f);
    physx::PxReal trueGap = 0.0f;
    if (!getCurrentDynamicSoftRigidContactGeometry(
            geometry, rigidBody, queryPoint, normal, worldOffset,
            trueGap))
      continue;
    if (!physx::PxIsFinite(trueGap) || trueGap > nearSurfaceTolerance)
      continue;

    physx::PxVec3 rigidLinearJacobian = -normal;
    rigidBody.projectLockedLinearVector(rigidLinearJacobian);
    const physx::PxVec3 rigidLinearDeltaPerImpulse =
        rigidLinearJacobian * rigidBody.invMass;
    physx::PxVec3 rigidAngularJacobian = -worldOffset.cross(normal);
    rigidBody.projectLockedAngularVector(rigidAngularJacobian);
    physx::PxVec3 rigidAngularDeltaPerImpulse =
        rigidBody.invInertiaWorld * rigidAngularJacobian;
    rigidBody.projectLockedAngularVector(rigidAngularDeltaPerImpulse);
    const physx::PxReal rigidResponse =
        rigidLinearJacobian.dot(rigidLinearDeltaPerImpulse) +
        rigidAngularJacobian.dot(rigidAngularDeltaPerImpulse);
    const physx::PxReal softResponse =
        queryWeightSum * queryWeightSum / totalSoftMass;
    const physx::PxReal response = softResponse + rigidResponse;
    if (!rigidLinearDeltaPerImpulse.isFinite() ||
        !rigidAngularDeltaPerImpulse.isFinite() ||
        !physx::PxIsFinite(response) || response <= 1.0e-12f)
      continue;

    const physx::PxVec3 rigidSurfaceVelocity =
        rigidBody.linearVelocity + rigidBody.angularVelocity.cross(worldOffset);
    const physx::PxReal relativeNormalVelocity =
        (queryVelocity - rigidSurfaceVelocity).dot(normal);
    if (!physx::PxIsFinite(relativeNormalVelocity) ||
        relativeNormalVelocity >= -1.0e-6f)
      continue;
    const physx::PxReal impulse = -relativeNormalVelocity / response;
    if (!physx::PxIsFinite(impulse) || impulse <= 0.0f)
      continue;

    const physx::PxVec3 softVelocityDelta =
        normal * (queryWeightSum * impulse / totalSoftMass);
    physx::PxVec3 candidateLinearVelocity =
        rigidBody.linearVelocity + rigidLinearDeltaPerImpulse * impulse;
    physx::PxVec3 candidateAngularVelocity =
        rigidBody.angularVelocity + rigidAngularDeltaPerImpulse * impulse;
    rigidBody.projectLockedLinearVector(candidateLinearVelocity);
    rigidBody.projectLockedAngularVector(candidateAngularVelocity);
    bool finiteCandidates = softVelocityDelta.isFinite() &&
        candidateLinearVelocity.isFinite() && candidateAngularVelocity.isFinite();
    for (physx::PxU32 localIndex = 0;
         localIndex < particleCount && finiteCandidates; ++localIndex) {
      finiteCandidates =
          (softParticles[particleStart + localIndex].velocity +
           softVelocityDelta)
              .isFinite();
    }
    if (!finiteCandidates)
      continue;

    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex)
      softParticles[particleStart + localIndex].velocity += softVelocityDelta;
    rigidBody.linearVelocity = candidateLinearVelocity;
    rigidBody.angularVelocity = candidateAngularVelocity;
    rigidBody.projectLockedVelocities();
  }
}

void AvbdSolver::applyDynamicSoftRigidNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps,
    physx::PxArray<physx::PxU8> *recoveredContacts,
    AvbdSolverStats *stats, const AvbdOgcPairState *ogcPairStates,
    physx::PxU32 numOgcPairStates,
    const physx::PxU32 *ogcPairIndices,
    physx::PxU32 numOgcPairIndices,
    physx::PxReal softComplianceResponseScale,
    bool projectToCurrentPoseBoundary,
    const physx::PxU32 *ogcPairContactStarts,
    physx::PxU32 numOgcPairContactStarts,
    const physx::PxU32 *ogcPairContactRefs,
    physx::PxU32 numOgcPairContactRefs) {
  (void)stats;
  if (!bodies || numBodies == 0 || !softParticles ||
      numSoftParticles == 0 || !softBodies || numSoftBodies == 0 ||
      !softContacts || numSoftContacts == 0 || sweeps == 0)
    return;

  if (recoveredContacts && recoveredContacts->size() != numSoftContacts) {
    recoveredContacts->resize(numSoftContacts);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      (*recoveredContacts)[sci] = 0u;
  }

  // This is a recovery, not a second OGC owner.  Cap each contact projection
  // to a small physical distance and let six Gauss--Seidel sweeps resolve
  // intersecting rows in their prepared order.
  const physx::PxReal lengthScale =
      physx::PxMax(mConfig.lengthScale, 1.0e-6f);
  const physx::PxReal maxCorrectionPerContact = 0.05f * lengthScale;
  // The material block supplies the compliant endpoint in a mixed contact.
  // A terminal core repair otherwise sees only particle masses and makes a
  // light rigid absorb nearly all positional correction.  Treat the supplied
  // factor as a local contact compliance (not a mass change): scale both the
  // soft response and its Jacobian displacement consistently.
  const physx::PxReal softResponseScale =
      physx::PxClamp(softComplianceResponseScale, 1.0f, 16.0f);
  const bool usePairGroups = ogcPairStates && ogcPairIndices &&
      numOgcPairStates > 0 && numOgcPairIndices == numSoftContacts;
  const bool usePairContactPlan = usePairGroups && ogcPairContactStarts &&
      numOgcPairContactStarts == numOgcPairStates + 1 &&
      (numOgcPairContactRefs == 0 || ogcPairContactRefs) &&
      ogcPairContactStarts[0] == 0 &&
      ogcPairContactStarts[numOgcPairStates] == numOgcPairContactRefs;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    physx::PxArray<physx::PxU32> deepestPairContacts;
    if (usePairGroups) {
      deepestPairContacts.resize(numOgcPairStates);
      physx::PxArray<physx::PxReal> deepestPairGaps(numOgcPairStates,
                                                     0.0f);
      for (physx::PxU32 pairIndex = 0; pairIndex < numOgcPairStates;
           ++pairIndex)
        deepestPairContacts[pairIndex] = PX_MAX_U32;
      const physx::PxU32 pairScanCount =
          usePairContactPlan ? numOgcPairStates : numSoftContacts;
      for (physx::PxU32 scan = 0; scan < pairScanCount; ++scan) {
        const physx::PxU32 pairIndex = usePairContactPlan
            ? scan
            : ogcPairIndices[scan];
        if (pairIndex >= numOgcPairStates ||
            !ogcPairStates[pairIndex].active)
          continue;
        const physx::PxU32 begin = usePairContactPlan
            ? ogcPairContactStarts[pairIndex]
            : scan;
        const physx::PxU32 end = usePairContactPlan
            ? ogcPairContactStarts[pairIndex + 1u]
            : scan + 1u;
        if (begin > end || end >
                (usePairContactPlan ? numOgcPairContactRefs
                                    : numSoftContacts))
          continue;
        for (physx::PxU32 ref = begin; ref < end; ++ref) {
          const physx::PxU32 sci =
              usePairContactPlan ? ogcPairContactRefs[ref] : ref;
          if (sci >= numSoftContacts || ogcPairIndices[sci] != pairIndex)
            continue;
          const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
          if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
              !geometry.hasRigidBodyTarget() ||
              geometry.targetIndex >= numBodies)
            continue;
          const physx::PxVec3 queryPoint =
              avbdGetSoftContactQueryPoint(geometry, softParticles);
          physx::PxVec3 normal(0.0f), worldOffset(0.0f);
          physx::PxReal trueGap = 0.0f;
          if (!queryPoint.isFinite() ||
              !getCurrentDynamicSoftRigidContactGeometry(
                  geometry, bodies[geometry.targetIndex], queryPoint, normal,
                  worldOffset, trueGap) ||
              !physx::PxIsFinite(trueGap) || trueGap >= 0.0f)
            continue;
          if (deepestPairContacts[pairIndex] == PX_MAX_U32 ||
              trueGap < deepestPairGaps[pairIndex]) {
            deepestPairContacts[pairIndex] = sci;
            deepestPairGaps[pairIndex] = trueGap;
          }
        }
      }
    }
    const physx::PxU32 correctionCount =
        usePairGroups ? numOgcPairStates : numSoftContacts;
    for (physx::PxU32 correctionIndex = 0;
         correctionIndex < correctionCount; ++correctionIndex) {
      const physx::PxU32 sci = usePairGroups
          ? deepestPairContacts[correctionIndex]
          : correctionIndex;
      if (sci == PX_MAX_U32 || sci >= numSoftContacts)
        continue;
      const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
      if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
          !geometry.hasRigidBodyTarget() ||
          geometry.targetIndex >= numBodies ||
          !avbdHasSoftContactDynamicQuerySupport(
              geometry, softParticles, numSoftParticles))
        continue;

      const physx::PxU32 queryRepresentative =
          geometry.hasWeightedQueryPoint()
              ? geometry.queryPoint.particleIndices[0]
              : geometry.hasBarycentricQueryPoint()
                    ? geometry.queryParticleIndices[0]
                    : geometry.particleIdx;
      if (queryRepresentative >= numSoftParticles)
        continue;
      const AvbdSoftBody *sourceBody =
          geometry.queryBodyIndex < numSoftBodies &&
                  avbdSoftBodyContainsParticle(
                      softBodies[geometry.queryBodyIndex],
                      queryRepresentative, numSoftParticles)
              ? &softBodies[geometry.queryBodyIndex]
              : avbdFindSoftBodyForParticle(
                    softBodies, numSoftBodies, queryRepresentative);
      // This pass must remain entirely outside speculative CCD ownership.
      // Current-pose OGC rows on a non-CCD source body are its only input.
      if (!sourceBody || sourceBody->compiled.speculativeCCDEnabled)
        continue;

      AvbdSolverBody &body = bodies[geometry.targetIndex];
      if (body.invMass <= 0.0f)
        continue;

      const physx::PxVec3 queryPoint =
          avbdGetSoftContactQueryPoint(geometry, softParticles);
      if (!queryPoint.isFinite())
        continue;

      // Do not use avbdEvaluateSoftContactNormalConstraint() here: it
      // subtracts geometry.margin and is negative throughout the OGC shell.
      // This is the actual collision-surface signed gap only.
      physx::PxVec3 normal(0.0f);
      physx::PxVec3 worldOffset(0.0f);
      physx::PxReal trueGap = 0.0f;
      if (!getCurrentDynamicSoftRigidContactGeometry(
              geometry, body, queryPoint, normal, worldOffset, trueGap))
        continue;
      if (!physx::PxIsFinite(trueGap))
        continue;

      if (!(trueGap < 0.0f))
        continue;

      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleCount =
          avbdCollectSoftContactParticleIndices(geometry, particleIndices);
      if (particleCount == 0 || particleCount > AVBD_CONTACT_MAX_PARTICLES)
        continue;

      const AvbdSoftBody *supportBodies[AVBD_CONTACT_MAX_PARTICLES];
      physx::PxReal softResponse = 0.0f;
      bool validSupport = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        if (particleIndex >= numSoftParticles ||
            !avbdSoftBodyContainsParticle(
                *sourceBody, particleIndex, numSoftParticles)) {
          validSupport = false;
          break;
        }
        const physx::PxReal weight =
            avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        if (!physx::PxIsFinite(weight) ||
            !physx::PxIsFinite(particle.invMass) ||
            !particle.position.isFinite() ||
            !particle.initialPosition.isFinite()) {
          validSupport = false;
          break;
        }
        if (particle.invMass > 0.0f)
          softResponse +=
              softResponseScale * particle.invMass * weight * weight;
        supportBodies[pi] = sourceBody;
      }
      if (!validSupport || softResponse <= 1.0e-12f ||
          !physx::PxIsFinite(softResponse))
        continue;

      // J_rigid = {-n, -(r x n)}.  Project both the force direction and the
      // resulting angular displacement through lock masks before forming the
      // effective response, so locked rigid DOFs neither move nor contribute
      // artificial compliance.
      physx::PxVec3 rigidLinearJacobian = -normal;
      body.projectLockedLinearVector(rigidLinearJacobian);
      const physx::PxVec3 rigidLinearDeltaPerLambda =
          rigidLinearJacobian * body.invMass;
      physx::PxVec3 rigidAngularJacobian = -worldOffset.cross(normal);
      body.projectLockedAngularVector(rigidAngularJacobian);
      physx::PxVec3 rigidAngularDeltaPerLambda =
          body.invInertiaWorld * rigidAngularJacobian;
      body.projectLockedAngularVector(rigidAngularDeltaPerLambda);
      const physx::PxReal rigidResponse =
          rigidLinearJacobian.dot(rigidLinearDeltaPerLambda) +
          rigidAngularJacobian.dot(rigidAngularDeltaPerLambda);
      const physx::PxReal effectiveResponse = softResponse + rigidResponse;
      if (effectiveResponse <= 1.0e-12f ||
          !physx::PxIsFinite(effectiveResponse))
        continue;

      // Ordinary recovery is deliberately capped so it remains a gentle
      // post-AL fallback.  A freshly detected triangle-core row is
      // different: its common face is an exact current-pose DCD certificate
      // for the whole collision triangle.  Let that narrow path request the
      // complete distance to the face, while the shared-alpha positive-J
      // line search remains the hard deformation safety gate.
      const physx::PxReal requestedCorrection =
          projectToCurrentPoseBoundary
              ? -trueGap
              : physx::PxMin(-trueGap, maxCorrectionPerContact);
      const physx::PxReal lambda =
          requestedCorrection / effectiveResponse;
      if (lambda <= 0.0f || !physx::PxIsFinite(lambda))
        continue;

      // Keep this row transactional.  A malformed query support or a bad
      // angular update must not leave the paired soft/rigid projection only
      // half applied.
      physx::PxVec3 particleDeltas[AVBD_CONTACT_MAX_PARTICLES];
      bool finiteCandidates = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        particleDeltas[pi] = physx::PxVec3(0.0f);
        if (particle.invMass <= 0.0f)
          continue;
        const physx::PxReal weight =
            avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
        const physx::PxVec3 delta =
          normal * (softResponseScale * particle.invMass * weight * lambda);
        if (!delta.isFinite() ||
            !(particle.position + delta).isFinite() ||
            !(particle.initialPosition + delta).isFinite()) {
          finiteCandidates = false;
          break;
        }
        particleDeltas[pi] = delta;
      }
      const physx::PxVec3 bodyPositionBefore = body.position;
      const physx::PxQuat bodyRotationBefore = body.rotation;
      if (!finiteCandidates)
        continue;

      // A contact support moves as one coupled endpoint.  Reuse the scalar
      // positive-J policy's exact candidate det(F) test with one shared
      // alpha; per-particle limiting would be invalid once two vertices of
      // the same tetrahedron move together.  The paired rigid correction is
      // scaled by that exact same alpha.
      physx::PxReal commonAlpha = 1.0f;
      bool acceptedCandidate = false;
      AvbdSolverBody acceptedBody = body;
      for (physx::PxU32 attempt = 0; attempt < 8 && !acceptedCandidate;
           ++attempt) {
        bool candidateValid = true;
        for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
          const AvbdSoftParticle &particle =
              softParticles[particleIndices[pi]];
          const physx::PxVec3 delta = particleDeltas[pi] * commonAlpha;
          if (!delta.isFinite() ||
              !(particle.position + delta).isFinite() ||
              !(particle.initialPosition + delta).isFinite()) {
            candidateValid = false;
            break;
          }
        }

        auto candidatePositionFor =
            [&particleIndices, &particleDeltas, softParticles, particleCount,
             commonAlpha](physx::PxU32 particleIndex) -> physx::PxVec3 {
          for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
            if (particleIndices[pi] == particleIndex)
              return softParticles[particleIndex].position +
                     particleDeltas[pi] * commonAlpha;
          }
          return softParticles[particleIndex].position;
        };
        // Keep a positive determinant floor for healthy incident elements,
        // but permit this emergency projection to monotonically repair an
        // element which was already below the floor before this row.  The
        // terminal current-pose DCD owner uses a lower, still-positive floor:
        // retaining the ordinary material-quality threshold (.05) can make
        // the final contact feasible set empty and leave the rigid inside for
        // one frame. Material rows restore quality in the following epoch.
        const physx::PxReal minimumRecoveryDeterminant =
            projectToCurrentPoseBoundary ? 0.035f : 0.05f;
        bool hasSubthresholdIncidentTet = false;
        bool improvesSubthresholdIncidentTet = false;
        const physx::PxReal detRecoveryTolerance = 1.0e-6f;
        for (physx::PxU32 pi = 0; pi < particleCount && candidateValid;
             ++pi) {
          const AvbdSoftBody *supportBody = supportBodies[pi];
          const physx::PxU32 particleIndex = particleIndices[pi];
          if (!supportBody ||
              !avbdSoftBodyContainsParticle(
                  *supportBody, particleIndex, numSoftParticles)) {
            candidateValid = false;
            break;
          }
          const physx::PxU32 localIndex =
              particleIndex - supportBody->compiled.particleStart;
          if (localIndex >= supportBody->compiled.elementAdjacency.size()) {
            candidateValid = false;
            break;
          }
          const AvbdParticleElementAdjacency &adjacency =
              supportBody->compiled.elementAdjacency[localIndex];
          for (physx::PxU32 refIndex = 0;
               refIndex < adjacency.tetRefs.size(); ++refIndex) {
            const AvbdParticleElementRef &ref = adjacency.tetRefs[refIndex];
            if (ref.index >= supportBody->compiled.tetElements.size()) {
              candidateValid = false;
              break;
            }
            const AvbdTetElement &tet =
                supportBody->compiled.tetElements[ref.index];
            if (tet.p0 >= numSoftParticles || tet.p1 >= numSoftParticles ||
                tet.p2 >= numSoftParticles || tet.p3 >= numSoftParticles) {
              candidateValid = false;
              break;
            }
            const physx::PxVec3 currentP0 = softParticles[tet.p0].position;
            const physx::PxVec3 currentE1 =
                softParticles[tet.p1].position - currentP0;
            const physx::PxVec3 currentE2 =
                softParticles[tet.p2].position - currentP0;
            const physx::PxVec3 currentE3 =
                softParticles[tet.p3].position - currentP0;
            physx::PxReal currentDeterminant;
            physx::PxVec3 unusedCurrentGradient;
            avbdEvaluateTetDeterminantAndGradient(
                tet, 0u, currentE1, currentE2, currentE3,
                currentDeterminant, unusedCurrentGradient);
            const physx::PxVec3 p0 = candidatePositionFor(tet.p0);
            const physx::PxVec3 e1 = candidatePositionFor(tet.p1) - p0;
            const physx::PxVec3 e2 = candidatePositionFor(tet.p2) - p0;
            const physx::PxVec3 e3 = candidatePositionFor(tet.p3) - p0;
            physx::PxReal determinant;
            physx::PxVec3 unusedGradient;
            avbdEvaluateTetDeterminantAndGradient(
                tet, 0u, e1, e2, e3, determinant, unusedGradient);
            if (!physx::PxIsFinite(currentDeterminant) ||
                !physx::PxIsFinite(determinant)) {
              candidateValid = false;
              break;
            }
            if (currentDeterminant >= minimumRecoveryDeterminant) {
              if (determinant < minimumRecoveryDeterminant) {
                candidateValid = false;
                break;
              }
            } else {
              hasSubthresholdIncidentTet = true;
              if (determinant + detRecoveryTolerance < currentDeterminant) {
                candidateValid = false;
                break;
              }
              if (determinant > currentDeterminant + detRecoveryTolerance)
                improvesSubthresholdIncidentTet = true;
            }
          }
        }

        if (candidateValid && hasSubthresholdIncidentTet &&
            !improvesSubthresholdIncidentTet)
          candidateValid = false;

        AvbdSolverBody candidateBody = body;
        const physx::PxReal scaledLambda = lambda * commonAlpha;
        candidateBody.position +=
            rigidLinearDeltaPerLambda * scaledLambda;
        const physx::PxVec3 angularDelta =
            rigidAngularDeltaPerLambda * scaledLambda;
        if (angularDelta.magnitudeSquared() > 1.0e-16f) {
          const physx::PxQuat dq(
              angularDelta.x, angularDelta.y, angularDelta.z, 0.0f);
          candidateBody.rotation =
              (candidateBody.rotation + dq * candidateBody.rotation * 0.5f)
                  .getNormalized();
        }
        candidateBody.projectLockedPose(bodyPositionBefore,
                                        bodyRotationBefore);

        // A rigid endpoint shared by two soft bodies is constrained by both
        // pair trust regions.  Clip its pose update against every *other*
        // current pair while retaining this row's soft correction.  This is
        // the active-set response which makes the contacted soft support take
        // additional deformation instead of pushing the rigid through the
        // opposite support and repairing it one frame later.
        const physx::PxReal rigidOgcAlpha =
            limitPostAlRigidOgcCandidate(
                softContacts, numSoftContacts, softParticles,
                numSoftParticles, geometry.targetIndex,
                geometry.queryBodyIndex, geometry.source.primitiveKey, body,
                candidateBody);
        if (!physx::PxIsFinite(rigidOgcAlpha) || rigidOgcAlpha < 0.0f) {
          candidateValid = false;
        } else if (rigidOgcAlpha < 1.0f - 1.0e-6f) {
          candidateBody.position = body.position +
              (candidateBody.position - body.position) * rigidOgcAlpha;
          candidateBody.rotation = physx::PxSlerp(
              rigidOgcAlpha, body.rotation, candidateBody.rotation);
          candidateBody.projectLockedPose(bodyPositionBefore,
                                          bodyRotationBefore);
        }

        // Solver bodies carry world-space inverse inertia.  The regular AVBD
        // inner iteration holds it fixed, but this out-of-band projection can
        // rotate a body between recovery sweeps; transport it with the
        // accepted pose so subsequent contacts get a consistent angular mass.
        physx::PxQuat actualRotationDelta =
            candidateBody.rotation * bodyRotationBefore.getConjugate();
        if (actualRotationDelta.w < 0.0f)
          actualRotationDelta = -actualRotationDelta;
        if (actualRotationDelta.isFinite()) {
          actualRotationDelta.normalize();
          const physx::PxMat33 rotationDelta(actualRotationDelta);
          candidateBody.invInertiaWorld =
              rotationDelta * body.invInertiaWorld *
              rotationDelta.getTranspose();
        } else {
          candidateValid = false;
        }
        candidateValid =
            candidateValid && candidateBody.position.isFinite() &&
            candidateBody.rotation.isFinite() &&
            candidateBody.invInertiaWorld.column0.isFinite() &&
            candidateBody.invInertiaWorld.column1.isFinite() &&
            candidateBody.invInertiaWorld.column2.isFinite();
        physx::PxVec3 candidateQueryPoint = queryPoint;
        for (physx::PxU32 pi = 0;
             pi < particleCount && candidateValid; ++pi) {
          const physx::PxReal weight =
              avbdGetSoftContactParticleJacobianScale(
                  geometry, particleIndices[pi]);
          candidateQueryPoint +=
              particleDeltas[pi] * (commonAlpha * weight);
          if (!physx::PxIsFinite(weight) || !candidateQueryPoint.isFinite())
            candidateValid = false;
        }
        physx::PxVec3 candidateNormal(0.0f);
        physx::PxVec3 candidateWorldOffset(0.0f);
        physx::PxReal candidateGap = 0.0f;
        const physx::PxReal gapImprovementTolerance = physx::PxMax(
            1.0e-6f, 1.0e-5f * lengthScale);
        if (candidateValid &&
            !getCurrentDynamicSoftRigidContactGeometry(
                geometry, candidateBody, candidateQueryPoint,
                candidateNormal, candidateWorldOffset, candidateGap))
          candidateValid = false;
        if (candidateValid &&
            (!(candidateGap > trueGap + gapImprovementTolerance) ||
             !physx::PxIsFinite(candidateGap)))
          candidateValid = false;
        if (candidateValid) {
          acceptedBody = candidateBody;
          acceptedCandidate = true;
        } else {
          commonAlpha *= 0.5f;
        }
      }
      if (!acceptedCandidate)
        continue;

      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        const physx::PxVec3 delta = particleDeltas[pi] * commonAlpha;
        if (delta.magnitudeSquared() == 0.0f)
          continue;
        AvbdSoftParticle &particle = softParticles[particleIndex];
        particle.position += delta;
        // This post-AL recovery has no material impulse.  Moving the velocity
        // anchor by exactly the same amount excludes it from the particle's
        // position-to-velocity reconstruction at the end of this stage.
        particle.initialPosition += delta;
      }
      body.position = acceptedBody.position;
      body.rotation = acceptedBody.rotation;
      body.invInertiaWorld = acceptedBody.invInertiaWorld;
      if (recoveredContacts && sci < recoveredContacts->size())
        (*recoveredContacts)[sci] |= 1u;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }

}

void AvbdSolver::clampDynamicSoftRigidInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxArray<physx::PxU8> *recoveredContacts,
    AvbdSolverStats *stats) {
  (void)stats;
  // Only recovery rows are admitted.  In particular, do not turn the OGC
  // proximity shell into a velocity constraint merely because it happens to
  // be present after the regular Position-AL solve.
  if (!bodies || numBodies == 0 || !softParticles ||
      numSoftParticles == 0 || !softContacts || numSoftContacts == 0 ||
      !recoveredContacts || recoveredContacts->size() != numSoftContacts)
    return;

  const physx::PxReal nearSurfaceTolerance = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    if (((*recoveredContacts)[sci] & 1u) == 0u)
      continue;

    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() || geometry.targetIndex >= numBodies ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    AvbdSolverBody &body = bodies[geometry.targetIndex];
    if (body.invMass <= 0.0f || !body.position.isFinite() ||
        !body.rotation.isFinite() || !body.linearVelocity.isFinite() ||
        !body.angularVelocity.isFinite() ||
        !body.invInertiaWorld.column0.isFinite() ||
        !body.invInertiaWorld.column1.isFinite() ||
        !body.invInertiaWorld.column2.isFinite())
      continue;

    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!queryPoint.isFinite())
      continue;

    // Recheck the true collision-surface gap after recovery.  This permits a
    // remaining actual overlap (or a numerically adjacent recovered row), but
    // never the much wider OGC shell.
    physx::PxVec3 normal(0.0f);
    physx::PxVec3 worldOffset(0.0f);
    physx::PxReal trueGap = 0.0f;
    if (!getCurrentDynamicSoftRigidContactGeometry(
            geometry, body, queryPoint, normal, worldOffset, trueGap))
      continue;
    if (!physx::PxIsFinite(trueGap) || trueGap > nearSurfaceTolerance)
      continue;

    physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
    const physx::PxU32 particleCount =
        avbdCollectSoftContactParticleIndices(geometry, particleIndices);
    if (particleCount == 0 || particleCount > AVBD_CONTACT_MAX_PARTICLES)
      continue;

    physx::PxReal softResponse = 0.0f;
    physx::PxVec3 queryVelocity(0.0f);
    bool validSupport = true;
    for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
      const physx::PxU32 particleIndex = particleIndices[pi];
      if (particleIndex >= numSoftParticles) {
        validSupport = false;
        break;
      }
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      if (!physx::PxIsFinite(weight) || !physx::PxIsFinite(particle.invMass) ||
          !particle.velocity.isFinite()) {
        validSupport = false;
        break;
      }
      queryVelocity += particle.velocity * weight;
      if (particle.invMass > 0.0f)
        softResponse += particle.invMass * weight * weight;
    }
    if (!validSupport || !queryVelocity.isFinite() ||
        !physx::PxIsFinite(softResponse) || softResponse <= 1.0e-12f)
      continue;

    physx::PxVec3 rigidLinearJacobian = -normal;
    body.projectLockedLinearVector(rigidLinearJacobian);
    const physx::PxVec3 rigidLinearDeltaPerImpulse =
        rigidLinearJacobian * body.invMass;
    physx::PxVec3 rigidAngularJacobian = -worldOffset.cross(normal);
    body.projectLockedAngularVector(rigidAngularJacobian);
    physx::PxVec3 rigidAngularDeltaPerImpulse =
        body.invInertiaWorld * rigidAngularJacobian;
    body.projectLockedAngularVector(rigidAngularDeltaPerImpulse);
    const physx::PxReal rigidResponse =
        rigidLinearJacobian.dot(rigidLinearDeltaPerImpulse) +
        rigidAngularJacobian.dot(rigidAngularDeltaPerImpulse);
    const physx::PxReal response = softResponse + rigidResponse;
    if (!rigidLinearDeltaPerImpulse.isFinite() ||
        !rigidAngularDeltaPerImpulse.isFinite() ||
        !physx::PxIsFinite(response) || response <= 1.0e-12f)
      continue;

    const physx::PxVec3 rigidSurfaceVelocity =
        body.linearVelocity + body.angularVelocity.cross(worldOffset);
    const physx::PxReal relativeNormalVelocity =
        (queryVelocity - rigidSurfaceVelocity).dot(normal);
    if (!physx::PxIsFinite(relativeNormalVelocity) ||
        relativeNormalVelocity >= -1.0e-6f)
      continue;
    const physx::PxReal impulse = -relativeNormalVelocity / response;
    if (!physx::PxIsFinite(impulse) || impulse <= 0.0f)
      continue;

    // Compute every endpoint candidate before writing either endpoint.  The
    // recovery does not own an elastic/material impulse, so this is a pure
    // e=0 unilateral normal correction with no dual update.
    physx::PxVec3 particleVelocityDeltas[AVBD_CONTACT_MAX_PARTICLES];
    bool finiteCandidates = true;
    for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
      const physx::PxU32 particleIndex = particleIndices[pi];
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      const physx::PxReal weight =
          avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
      particleVelocityDeltas[pi] = physx::PxVec3(0.0f);
      if (particle.invMass <= 0.0f)
        continue;
      const physx::PxVec3 delta =
          normal * (particle.invMass * weight * impulse);
      if (!delta.isFinite() || !(particle.velocity + delta).isFinite()) {
        finiteCandidates = false;
        break;
      }
      particleVelocityDeltas[pi] = delta;
    }
    physx::PxVec3 candidateLinearVelocity =
        body.linearVelocity + rigidLinearDeltaPerImpulse * impulse;
    physx::PxVec3 candidateAngularVelocity =
        body.angularVelocity + rigidAngularDeltaPerImpulse * impulse;
    body.projectLockedLinearVector(candidateLinearVelocity);
    body.projectLockedAngularVector(candidateAngularVelocity);
    if (!finiteCandidates || !candidateLinearVelocity.isFinite() ||
        !candidateAngularVelocity.isFinite())
      continue;

    for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
      const physx::PxVec3 &delta = particleVelocityDeltas[pi];
      if (delta.magnitudeSquared() > 0.0f)
        softParticles[particleIndices[pi]].velocity += delta;
    }
    body.linearVelocity = candidateLinearVelocity;
    body.angularVelocity = candidateAngularVelocity;
    body.projectLockedVelocities();
  }
}

void AvbdSolver::clampAdmittedMixedOgcPairNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *contactPairIndices,
    physx::PxU32 numContactPairIndices,
    const physx::PxU32 *pairContactStarts,
    physx::PxU32 numPairContactStarts,
    const physx::PxU32 *pairContactRefs,
    physx::PxU32 numPairContactRefs,
    AvbdSolverStats *stats) {
  (void)stats;
  if (!bodies || numBodies == 0 || !softParticles ||
      numSoftParticles == 0 || !softContacts || numSoftContacts == 0 ||
      !pairStates || numPairStates == 0)
    return;

  // Most native mixed frames have an OGC pair plan but no endpoint was
  // actually clipped by this dt.  In that normal case there is no withheld
  // impact to convert, so avoid even validating the inverse CSR.  This keeps
  // the pair scheduler proportional to active boundary events rather than to
  // every resting/proximity manifold in the island.
  bool hasAdmittedPair = false;
  for (physx::PxU32 pairIndex = 0; pairIndex < numPairStates; ++pairIndex) {
    const AvbdOgcPairState &pair = pairStates[pairIndex];
    if (pair.active && pair.admittedAtBoundary) {
      hasAdmittedPair = true;
      break;
    }
  }
  if (!hasAdmittedPair)
    return;

  // This is intentionally a velocity-only PGS block.  The pair reached this
  // band only because the same-dt DCD admission prevented a real crossing;
  // it is therefore safe to preserve the contact's unilateral e=0 condition
  // throughout the remainder of this frame without promoting arbitrary OGC
  // proximity rows into velocity owners.  Consume the complete prepared
  // manifold for that pair, rather than just the row which first reached the
  // boundary: a single representative vertex concentrates the withheld
  // impulse and makes a free rigid appear to bounce out of a broad soft
  // contact instead of loading the adjacent material patch.
  const physx::PxReal minimumBand = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0e-6f));
  static const physx::PxU32 kPairVelocitySweeps = 4u;

  // Pair state is the ownership unit, so consume the provider's immutable
  // inverse contact map here too.  The former pair-by-contact nested scan
  // was O(pairCount * contactCount) precisely in the broad-manifold cases
  // this block exists to stabilize.  Validate the optional program before
  // using it: a malformed accelerator must retain the direct-scan semantics,
  // never drop or duplicate a normal impulse.
  bool usePairContactPlan =
      contactPairIndices && numContactPairIndices == numSoftContacts &&
      pairContactStarts && numPairContactStarts == numPairStates + 1u &&
      (numPairContactRefs == 0u || pairContactRefs) &&
      pairContactStarts[0] == 0u &&
      pairContactStarts[numPairStates] == numPairContactRefs;
  physx::PxArray<physx::PxU8> pairPlanSeen;
  if (usePairContactPlan) {
    pairPlanSeen.resize(numSoftContacts);
    for (physx::PxU32 contactIndex = 0; contactIndex < numSoftContacts;
         ++contactIndex)
      pairPlanSeen[contactIndex] = 0u;
    for (physx::PxU32 pairIndex = 0;
         pairIndex < numPairStates && usePairContactPlan; ++pairIndex) {
      const physx::PxU32 begin = pairContactStarts[pairIndex];
      const physx::PxU32 end = pairContactStarts[pairIndex + 1u];
      if (begin > end || end > numPairContactRefs) {
        usePairContactPlan = false;
        break;
      }
      for (physx::PxU32 refIndex = begin; refIndex < end; ++refIndex) {
        const physx::PxU32 contactIndex = pairContactRefs[refIndex];
        if (contactIndex >= numSoftContacts ||
            pairPlanSeen[contactIndex] != 0u ||
            contactPairIndices[contactIndex] != pairIndex) {
          usePairContactPlan = false;
          break;
        }
        pairPlanSeen[contactIndex] = 1u;
      }
    }
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts && usePairContactPlan;
         ++contactIndex) {
      const physx::PxU32 pairIndex = contactPairIndices[contactIndex];
      if (pairIndex == PX_MAX_U32)
        continue;
      if (pairIndex >= numPairStates || pairPlanSeen[contactIndex] == 0u)
        usePairContactPlan = false;
    }
  }

  for (physx::PxU32 sweep = 0; sweep < kPairVelocitySweeps; ++sweep) {
    bool appliedImpulse = false;
    for (physx::PxU32 pairIndex = 0; pairIndex < numPairStates;
         ++pairIndex) {
      AvbdOgcPairState &pair = pairStates[pairIndex];
      if (!pair.active || !pair.admittedAtBoundary)
        continue;

      const physx::PxU32 begin =
          usePairContactPlan ? pairContactStarts[pairIndex] : 0u;
      const physx::PxU32 end = usePairContactPlan
          ? pairContactStarts[pairIndex + 1u] : numSoftContacts;
      for (physx::PxU32 refIndex = begin; refIndex < end; ++refIndex) {
      const physx::PxU32 contactIndex =
          usePairContactPlan ? pairContactRefs[refIndex] : refIndex;
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
          !geometry.hasRigidBodyTarget() ||
          geometry.queryBodyIndex != pair.sourceBodyIndex ||
          geometry.targetIndex != pair.targetBodyIndex ||
          geometry.source.primitiveKey != pair.primitiveKey ||
          geometry.targetIndex >= numBodies ||
          !avbdHasSoftContactDynamicQuerySupport(
              geometry, softParticles, numSoftParticles))
        continue;

      AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
      if (rigidBody.invMass <= 0.0f || !rigidBody.position.isFinite() ||
          !rigidBody.rotation.isFinite() ||
          !rigidBody.linearVelocity.isFinite() ||
          !rigidBody.angularVelocity.isFinite() ||
          !rigidBody.invInertiaWorld.column0.isFinite() ||
          !rigidBody.invInertiaWorld.column1.isFinite() ||
          !rigidBody.invInertiaWorld.column2.isFinite())
        continue;

      const physx::PxVec3 queryPoint =
          avbdGetSoftContactQueryPoint(geometry, softParticles);
      physx::PxVec3 normal(0.0f), worldOffset(0.0f);
      physx::PxReal trueGap = 0.0f;
      if (!queryPoint.isFinite() ||
          !getCurrentDynamicSoftRigidContactGeometry(
              geometry, rigidBody, queryPoint, normal, worldOffset,
              trueGap) ||
          !physx::PxIsFinite(trueGap))
        continue;

      // The actual signed distance remains the geometric authority.  The
      // contact margin only keeps a boundary-admitted impact owned until it
      // has visibly separated; it never pulls a distant pair back together.
      const physx::PxReal contactBand = physx::PxMax(
          minimumBand, physx::PxMax(0.0f, geometry.margin));
      if (trueGap > contactBand)
        continue;

      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleCount =
          avbdCollectSoftContactParticleIndices(geometry, particleIndices);
      if (particleCount == 0u ||
          particleCount > AVBD_CONTACT_MAX_PARTICLES)
        continue;

      physx::PxReal softResponse = 0.0f;
      physx::PxVec3 queryVelocity(0.0f);
      bool validSupport = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        if (particleIndex >= numSoftParticles) {
          validSupport = false;
          break;
        }
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        const physx::PxReal weight =
            avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
        if (!physx::PxIsFinite(weight) ||
            !physx::PxIsFinite(particle.invMass) ||
            !particle.velocity.isFinite()) {
          validSupport = false;
          break;
        }
        queryVelocity += particle.velocity * weight;
        if (particle.invMass > 0.0f)
          softResponse += particle.invMass * weight * weight;
      }
      if (!validSupport || !queryVelocity.isFinite() ||
          !physx::PxIsFinite(softResponse) || softResponse <= 1.0e-12f)
        continue;

      physx::PxVec3 rigidLinearJacobian = -normal;
      rigidBody.projectLockedLinearVector(rigidLinearJacobian);
      const physx::PxVec3 rigidLinearDelta =
          rigidLinearJacobian * rigidBody.invMass;
      physx::PxVec3 rigidAngularJacobian = -worldOffset.cross(normal);
      rigidBody.projectLockedAngularVector(rigidAngularJacobian);
      physx::PxVec3 rigidAngularDelta =
          rigidBody.invInertiaWorld * rigidAngularJacobian;
      rigidBody.projectLockedAngularVector(rigidAngularDelta);
      const physx::PxReal rigidResponse =
          rigidLinearJacobian.dot(rigidLinearDelta) +
          rigidAngularJacobian.dot(rigidAngularDelta);
      const physx::PxReal response = softResponse + rigidResponse;
      if (!rigidLinearDelta.isFinite() || !rigidAngularDelta.isFinite() ||
          !physx::PxIsFinite(response) || response <= 1.0e-12f)
        continue;

      const physx::PxVec3 rigidSurfaceVelocity =
          rigidBody.linearVelocity +
          rigidBody.angularVelocity.cross(worldOffset);
      const physx::PxReal relativeNormalVelocity =
          (queryVelocity - rigidSurfaceVelocity).dot(normal);
      if (!physx::PxIsFinite(relativeNormalVelocity) ||
          relativeNormalVelocity >= -1.0e-6f)
        continue;
      const physx::PxReal impulse = -relativeNormalVelocity / response;
      if (!physx::PxIsFinite(impulse) || impulse <= 0.0f)
        continue;

      physx::PxVec3 particleVelocityDeltas[AVBD_CONTACT_MAX_PARTICLES];
      bool finiteCandidates = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        const physx::PxReal weight =
            avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
        particleVelocityDeltas[pi] = physx::PxVec3(0.0f);
        if (particle.invMass <= 0.0f)
          continue;
        const physx::PxVec3 delta =
            normal * (particle.invMass * weight * impulse);
        if (!delta.isFinite() || !(particle.velocity + delta).isFinite()) {
          finiteCandidates = false;
          break;
        }
        particleVelocityDeltas[pi] = delta;
      }
      physx::PxVec3 candidateLinearVelocity =
          rigidBody.linearVelocity + rigidLinearDelta * impulse;
      physx::PxVec3 candidateAngularVelocity =
          rigidBody.angularVelocity + rigidAngularDelta * impulse;
      rigidBody.projectLockedLinearVector(candidateLinearVelocity);
      rigidBody.projectLockedAngularVector(candidateAngularVelocity);
      if (!finiteCandidates || !candidateLinearVelocity.isFinite() ||
          !candidateAngularVelocity.isFinite())
        continue;

      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxVec3 &delta = particleVelocityDeltas[pi];
        if (delta.magnitudeSquared() > 0.0f)
          softParticles[particleIndices[pi]].velocity += delta;
      }
      rigidBody.linearVelocity = candidateLinearVelocity;
      rigidBody.angularVelocity = candidateAngularVelocity;
      rigidBody.projectLockedVelocities();
      pair.accumulatedNormalLambda += impulse;
      pair.minimumGap = physx::PxMin(pair.minimumGap, trueGap);
      appliedImpulse = true;
      }
    }
    if (!appliedImpulse)
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
  (void)stats;
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
    const AvbdSoftContactGeometry &geometry = sc.geometry;
    if (!geometry.hasRigidBodyTarget() ||
        geometry.targetIndex >= numBodies ||
        geometry.friction <= 0.0f)
      continue;
    if (!avbdIsSoftContactQueryFullyKinematic(
            geometry, softParticles, numSoftParticles))
      continue;
    const physx::PxReal viol =
        avbdKinematicShellContactViolation(
            geometry, bodies[geometry.targetIndex]);
    if (viol > 0.05f)
      continue;
    const physx::PxU32 bi = geometry.targetIndex;
    if (viol < worstViol[bi]) {
      worstViol[bi] = viol;
      dominantContact[bi] = sci;
    }
  }
  physx::PxArray<physx::PxU32> frContacts;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (dominantContact[i] != 0xFFFFFFFFu) {
      frContacts.pushBack(dominantContact[i]);
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
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      const AvbdSoftContactAugmentedState &state = sc.state;
      const physx::PxU32 bi = geometry.targetIndex;
      AvbdSolverBody &body = bodies[bi];
      touched[bi] = true;

      const physx::PxVec3 r =
          body.rotation.rotate(geometry.rigidLocalPoint);
      const physx::PxMat33 &invI = body.invInertiaWorld;
      // Shell path: one dominant contact per body. Always track tangential
      // mesh velocity (shot sphere mass can be >>5). Stack multi-corner
      // energy is limited by NP multi-corner gate + e=0 clamps.
      physx::PxVec3 vMesh =
          (geometry.surfacePoint - state.surfacePointPrev) * invDt;
      {
        const physx::PxVec3 &n = geometry.normal;
        vMesh = vMesh - n * vMesh.dot(n);
        const physx::PxReal vCap = 12.0f;
        const physx::PxReal vMag2 = vMesh.magnitudeSquared();
        if (vMag2 > vCap * vCap)
          vMesh *= vCap / physx::PxSqrt(vMag2);
      }

      const physx::PxReal viol =
          avbdKinematicShellContactViolation(geometry, body);
      const physx::PxReal normalForce =
          PxMax(PxAbs(state.alLambda), state.k * PxMax(0.0f, -viol));
      const physx::PxReal jmax = normalForce * geometry.friction * dt;
      if (jmax <= 0.0f)
        continue;

      const physx::PxVec3 tangents[2] = {
          geometry.tangent1, geometry.tangent2};
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
    physx::PxReal dt, AvbdSolverStats *stats) {
  (void)stats;
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
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex != i)
        continue;
      if (!avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles))
        continue;
      const physx::PxReal viol =
          avbdKinematicShellContactViolation(geometry, bodies[i]);
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
    const AvbdSoftContactGeometry &geometry = sc.geometry;
    const AvbdSoftContactAugmentedState &state = sc.state;
    const physx::PxVec3 nd = geometry.normal;
    const physx::PxReal vn = bodies[i].linearVelocity.dot(nd);
    const physx::PxReal vMeshN =
        invDt > 0.0f
            ? ((geometry.surfacePoint - state.surfacePointPrev) * invDt).dot(nd)
            : 0.0f;
    const physx::PxReal vRelN = vn - vMeshN;
    if (vRelN > 0.0f) {
      bodies[i].linearVelocity -= nd * vRelN;
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
