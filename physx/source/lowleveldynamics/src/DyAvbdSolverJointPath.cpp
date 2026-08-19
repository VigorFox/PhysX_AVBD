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
#include "DyFeatherstoneArticulation.h"
#include "CmConeLimitHelper.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "PxConstraintDesc.h"
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
static bool computeEllipticalConeConstraint(
    const AvbdD6JointConstraint &joint, const physx::PxQuat &rotA,
    const physx::PxQuat &rotB, physx::PxVec3 &worldAxis,
    physx::PxReal &violation) {
  const physx::PxU32 ellipticalConeFlags =
      AvbdD6JointConstraint::eD6_LEGACY_CONE_LIMIT_ACTIVE |
      AvbdD6JointConstraint::eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE;
  if ((joint.sourceFlags & ellipticalConeFlags) == 0 ||
      joint.coneAngleLimit <= 0.0f || joint.coneAngleLimitZ <= 0.0f)
    return false;

  physx::PxQuat worldFrameA = rotA * joint.localFrameA;
  physx::PxQuat worldFrameB = rotB * joint.localFrameB;
  worldFrameA.normalize();
  worldFrameB.normalize();
  physx::PxQuat relative =
      worldFrameA.getConjugate() * worldFrameB;
  relative.normalize();
  if (relative.w < 0.0f)
    relative = -relative;

  physx::PxQuat swing, twist;
  physx::PxSeparateSwingTwist(relative, swing, twist);
  if (swing.w < 0.0f)
    swing = -swing;

  physx::Cm::ConeLimitHelperTanLess helper(
      joint.coneAngleLimit, joint.coneAngleLimitZ);
  physx::PxVec3 localAxis;
  physx::PxReal signedInsideError = 0.0f;
  helper.getLimit(swing, localAxis, signedInsideError);
  worldAxis = worldFrameA.rotate(localAxis);
  const physx::PxReal axisMagnitude = worldAxis.magnitude();
  if (!worldAxis.isFinite() || !PxIsFinite(signedInsideError) ||
      axisMagnitude <= 1e-6f)
    return false;
  worldAxis *= 1.0f / axisMagnitude;
  // ConeLimitHelperTanLess returns a positive error inside the cone and a
  // negative error outside.  AVBD's existing unilateral cone convention
  // uses a positive outward violation.
  violation = -signedInsideError;
  return PxIsFinite(violation);
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

static bool queryCurrentPairSignedDistance(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const physx::PxVec3 &queryPoint, physx::PxReal &signedDistance);
static bool queryCurrentPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    physx::PxReal &faceGap);
static bool queryCurrentWorldStaticPairSignedDistance(
    const AvbdSoftContactGeometry &geometry,
    const physx::PxVec3 &queryPoint, physx::PxReal &signedDistance);
static bool queryCurrentWorldStaticPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    physx::PxReal &faceGap);
static bool queryInterpolatedPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSolverBody &initialBody, const AvbdSolverBody &endpointBody,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal alpha, physx::PxReal &faceGap);

// Conservative single-particle part of the shared OGC trust region.  The
// rigid endpoint has already been updated for this nonlinear GS iteration, so
// evaluate the soft candidate against that *current* pose.  Every incident
// dynamic pair votes for one common alpha; the particle may still deform in
// any tangent direction, but it cannot spend more normal clearance than the
// pair owns.  This is a candidate filter, not a post-solve depenetration.
static physx::PxReal limitSoftParticleOgcCandidate(
    const AvbdOgcPairTrustRegionContext *context,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftContactParticleRef *contactRefs,
    physx::PxU32 contactRefBegin, physx::PxU32 contactRefEnd,
    physx::PxU32 particleIndex,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSolverBody *rigidBodies, physx::PxU32 numRigidBodies,
    const physx::PxVec3 &candidateDisplacement) {
  if (!context || !context->isComplete(numSoftContacts) ||
      !contactRefs || !softParticles ||
      !candidateDisplacement.isFinite())
    return 1.0f;

  physx::PxReal alpha = 1.0f;
  const physx::PxReal tolerance = 1.0e-6f;
  for (physx::PxU32 refIndex = contactRefBegin;
       refIndex < contactRefEnd; ++refIndex) {
    const AvbdSoftContactParticleRef &ref = contactRefs[refIndex];
    if (ref.contactIndex >= numSoftContacts)
      continue;
    const physx::PxU32 pairIndex =
        context->contactPairIndices[ref.contactIndex];
    if (pairIndex >= context->numPairStates ||
        !context->pairStates[pairIndex].active)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[ref.contactIndex].geometry;
    const bool hasDynamicRigid = geometry.hasRigidBodyTarget() &&
        rigidBodies && geometry.targetIndex < numRigidBodies;
    const bool hasWorldStatic = geometry.hasWorldStaticTarget();
    if ((!hasDynamicRigid && !hasWorldStatic) ||
        pair.sourceType != geometry.source.type ||
        pair.targetKind != geometry.targetKind ||
        pair.sourceBodyIndex != geometry.queryBodyIndex ||
        pair.targetBodyIndex != geometry.targetIndex ||
        pair.primitiveKey != geometry.source.primitiveKey)
      continue;
    if (!physx::PxIsFinite(ref.jacobianScale))
      continue;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    physx::PxReal gap = 0.0f;
    physx::PxReal candidateGap = 0.0f;
    const physx::PxVec3 candidateQuery =
        queryPoint + candidateDisplacement * ref.jacobianScale;
    const bool querySucceeded = hasDynamicRigid
        ? queryCurrentPairSignedDistance(
              geometry, rigidBodies[geometry.targetIndex], queryPoint, gap) &&
              queryCurrentPairSignedDistance(
                  geometry, rigidBodies[geometry.targetIndex], candidateQuery,
                  candidateGap)
        : queryCurrentWorldStaticPairSignedDistance(geometry, queryPoint, gap) &&
              queryCurrentWorldStaticPairSignedDistance(
                  geometry, candidateQuery, candidateGap);
    if (!querySucceeded)
      continue;
    // A clipped OGC support commonly begins the next nonlinear update exactly
    // on its current-pose boundary.  It is still a live trust-region
    // constraint: tangent and separating candidates are valid, but an inward
    // candidate must be rejected.  Treating `gap <= tolerance` as a reason to
    // skip the row re-opens the boundary after the first clip and lets a later
    // material/rigid GS update tunnel through it in the same dt.
    if (candidateGap >= gap - tolerance)
      continue;
    physx::PxReal localAlpha = 0.0f;
    if (gap > tolerance) {
      const physx::PxReal denominator = gap - candidateGap;
      if (denominator <= tolerance)
        continue;
      // Keep a tiny positive residual to avoid repeatedly landing on the
      // unilateral boundary through round-off.
      localAlpha =
          physx::PxClamp((gap - tolerance) / denominator, 0.0f, 1.0f);
    }
    // Reaching the pair trust-region boundary is a real OGC epoch event.
    // Preserve it until post-AL consumes it with a current-pose DCD refresh;
    // otherwise a triangle interior can cross the OBB while every cached
    // point witness remains just outside.  This is not a swept/CCD request:
    // the candidate and boundary are both evaluated at this same solve time.
    if (localAlpha < 1.0f - 1.0e-6f) {
      pair.refreshRequested = true;
      pair.remainingSafeDisplacement = 0.0f;
      const physx::PxReal admittedGap =
          gap + (candidateGap - gap) * localAlpha;
      pair.minimumGap = physx::PxMin(pair.minimumGap, admittedGap);
      pair.accumulatedRelativeDisplacement = physx::PxMax(
          pair.accumulatedRelativeDisplacement,
          physx::PxMax(0.0f, gap - admittedGap));
    }
    alpha = physx::PxMin(alpha, localAlpha);
  }

  // The ordinary contact CSR above contains only the compact AL query.  A
  // triangle-core row additionally publishes all three proxy-vertex
  // embeddings in this companion CSR, so a material update cannot move an
  // unreferenced triangle corner across an otherwise positive OGC boundary.
  // It is intentionally an admission-only pass: these refs do not duplicate
  // the AL force/Hessian row.
  if (!context->hasTriangleCoreSafetyPlan(numSoftParticles))
    return physx::PxIsFinite(alpha) ? alpha : 0.0f;
  const physx::PxU32 coreBegin =
      context->triangleCoreSafetyStarts[particleIndex];
  const physx::PxU32 coreEnd =
      context->triangleCoreSafetyStarts[particleIndex + 1u];
  for (physx::PxU32 refIndex = coreBegin; refIndex < coreEnd; ++refIndex) {
    const AvbdSoftContactParticleRef &ref =
        context->triangleCoreSafetyRefs[refIndex];
    if (ref.contactIndex >= numSoftContacts)
      continue;
    const physx::PxU32 pairIndex =
        context->contactPairIndices[ref.contactIndex];
    if (pairIndex >= context->numPairStates ||
        !context->pairStates[pairIndex].active)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[ref.contactIndex].geometry;
    const bool hasDynamicRigid = geometry.hasRigidBodyTarget() &&
        rigidBodies && geometry.targetIndex < numRigidBodies;
    const bool hasWorldStatic = geometry.hasWorldStaticTarget();
    if ((!hasDynamicRigid && !hasWorldStatic) ||
        pair.sourceType != geometry.source.type ||
        pair.targetKind != geometry.targetKind ||
        pair.sourceBodyIndex != geometry.queryBodyIndex ||
        pair.targetBodyIndex != geometry.targetIndex ||
        pair.primitiveKey != geometry.source.primitiveKey)
      continue;

    physx::PxReal gap = 0.0f;
    physx::PxReal candidateGap = 0.0f;
    const bool querySucceeded = hasDynamicRigid
        ? queryCurrentPairTriangleCoreFaceGap(
              geometry, rigidBodies[geometry.targetIndex], softParticles,
              numSoftParticles, PX_MAX_U32, physx::PxVec3(0.0f), gap) &&
              queryCurrentPairTriangleCoreFaceGap(
                  geometry, rigidBodies[geometry.targetIndex], softParticles,
                  numSoftParticles, particleIndex, candidateDisplacement,
                  candidateGap)
        : queryCurrentWorldStaticPairTriangleCoreFaceGap(
              geometry, softParticles, numSoftParticles, PX_MAX_U32,
              physx::PxVec3(0.0f), gap) &&
              queryCurrentWorldStaticPairTriangleCoreFaceGap(
                  geometry, softParticles, numSoftParticles, particleIndex,
                  candidateDisplacement, candidateGap);
    if (!querySucceeded)
      continue;
    // Keep the complete collision triangle on its fresh OGC face once that
    // face has been reached.  Otherwise an unreferenced corner can spend a
    // new inward displacement during the same material iteration.
    if (candidateGap >= gap - tolerance)
      continue;
    physx::PxReal localAlpha = 0.0f;
    if (gap > tolerance) {
      const physx::PxReal denominator = gap - candidateGap;
      if (denominator <= tolerance)
        continue;
      localAlpha =
          physx::PxClamp((gap - tolerance) / denominator, 0.0f, 1.0f);
    }
    if (localAlpha < 1.0f - 1.0e-6f) {
      pair.refreshRequested = true;
      pair.remainingSafeDisplacement = 0.0f;
      const physx::PxReal admittedGap =
          gap + (candidateGap - gap) * localAlpha;
      pair.minimumGap = physx::PxMin(pair.minimumGap, admittedGap);
      pair.accumulatedRelativeDisplacement = physx::PxMax(
          pair.accumulatedRelativeDisplacement,
          physx::PxMax(0.0f, gap - admittedGap));
    }
    alpha = physx::PxMin(alpha, localAlpha);
  }
  return physx::PxIsFinite(alpha) ? alpha : 0.0f;
}

// A deferred inertial target is intentionally released only through the
// contact-admitted soft body.  Unlike the ordinary material path, it can
// increase an already-large compressive load after the boundary clip.  Keep
// that release from degrading an incident tet: if the tet is healthy retain
// the existing 5% determinant floor; if it was already below the floor,
// accept only a non-worsening candidate so the regular material/contact
// equations still have a route to repair it.
// Exact point-to-current-OBB signed distance for the candidate trust region.
// The regular contact row still owns AL normal/penalty state; this query is
// only the geometric admission test and therefore must not reuse its cached
// detector normal when the rigid has already rotated in the same GS sweep.
static bool queryCurrentPairSignedDistance(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const physx::PxVec3 &queryPoint, physx::PxReal &signedDistance) {
  if (!queryPoint.isFinite())
    return false;
  if (!geometry.hasRigidBoxSdf) {
    const physx::PxReal normalLengthSq = geometry.normal.magnitudeSquared();
    if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f)
      return false;
    const physx::PxVec3 normal =
        geometry.normal * physx::PxRecipSqrt(normalLengthSq);
    signedDistance =
        (queryPoint - avbdGetRigidContactSurfacePoint(geometry, body)).dot(normal);
    return physx::PxIsFinite(signedDistance);
  }
  if (!body.position.isFinite() || !body.rotation.isFinite() ||
      !geometry.rigidBoxPose.p.isFinite() ||
      !geometry.rigidBoxPose.q.isFinite() ||
      !geometry.rigidBoxHalfExtent.isFinite())
    return false;
  const physx::PxQuat shapeRotation =
      body.rotation * geometry.rigidBoxPose.q;
  const physx::PxVec3 shapePosition =
      body.position + body.rotation.rotate(geometry.rigidBoxPose.p);
  const physx::PxVec3 halfExtent = geometry.rigidBoxHalfExtent;
  if (!shapeRotation.isFinite() || !shapePosition.isFinite() ||
      halfExtent.x <= 0.0f || halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;
  const physx::PxVec3 localPoint =
      shapeRotation.getConjugate().rotate(queryPoint - shapePosition);
  const physx::PxVec3 delta(
      physx::PxAbs(localPoint.x) - halfExtent.x,
      physx::PxAbs(localPoint.y) - halfExtent.y,
      physx::PxAbs(localPoint.z) - halfExtent.z);
  const physx::PxVec3 outside(
      physx::PxMax(delta.x, 0.0f), physx::PxMax(delta.y, 0.0f),
      physx::PxMax(delta.z, 0.0f));
  const physx::PxReal inside = physx::PxMin(
      physx::PxMax(delta.x, physx::PxMax(delta.y, delta.z)), 0.0f);
  signedDistance = outside.magnitude() + inside;
  return physx::PxIsFinite(signedDistance);
}

// Return the signed clearance of the complete expanded collision triangle to
// the OBB support plane selected by its discrete core certificate.  Moving one
// simulation particle is optional; this makes the same function usable for
// the soft candidate admission and the rigid candidate admission.  It is a
// static current-pose query -- no segment interpolation, TOI, or CCD state is
// involved.
static bool queryCurrentPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    physx::PxReal &faceGap) {
  if (!geometry.hasRigidBoxSdf ||
      !geometry.hasRigidBoxTriangleCoreExit || !softParticles ||
      !movedParticleDisplacement.isFinite() || !body.position.isFinite() ||
      !body.rotation.isFinite() || !geometry.rigidBoxPose.p.isFinite() ||
      !geometry.rigidBoxPose.q.isFinite() ||
      !geometry.rigidBoxHalfExtent.isFinite() ||
      !geometry.rigidBoxTriangleCoreExitNormalLocal.isFinite())
    return false;

  const physx::PxVec3 rawNormalLocal =
      geometry.rigidBoxTriangleCoreExitNormalLocal;
  const physx::PxReal normalLengthSq = rawNormalLocal.magnitudeSquared();
  if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalLocal =
      rawNormalLocal * physx::PxRecipSqrt(normalLengthSq);
  const physx::PxVec3 halfExtent = geometry.rigidBoxHalfExtent;
  if (halfExtent.x <= 0.0f || halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;

  const physx::PxQuat shapeRotation = body.rotation * geometry.rigidBoxPose.q;
  const physx::PxVec3 shapePosition =
      body.position + body.rotation.rotate(geometry.rigidBoxPose.p);
  const physx::PxVec3 rawNormalWorld = shapeRotation.rotate(normalLocal);
  const physx::PxReal worldNormalLengthSq = rawNormalWorld.magnitudeSquared();
  if (!shapeRotation.isFinite() || !shapePosition.isFinite() ||
      !rawNormalWorld.isFinite() || !physx::PxIsFinite(worldNormalLengthSq) ||
      worldNormalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalWorld =
      rawNormalWorld * physx::PxRecipSqrt(worldNormalLengthSq);
  const physx::PxVec3 surfaceLocal(normalLocal.x * halfExtent.x,
                                    normalLocal.y * halfExtent.y,
                                    normalLocal.z * halfExtent.z);
  const physx::PxVec3 surfaceWorld =
      shapePosition + shapeRotation.rotate(surfaceLocal);

  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
    const AvbdWeightedContactPoint &mapping =
        geometry.rigidBoxTriangleCorePoints[vertex];
    if (mapping.count == 0 || mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    physx::PxVec3 point(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 support = 0; support < mapping.count; ++support) {
      const physx::PxU32 particleIndex = mapping.particleIndices[support];
      const physx::PxReal weight = mapping.weights[support];
      if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight) ||
          !softParticles[particleIndex].position.isFinite())
        return false;
      point += softParticles[particleIndex].position * weight;
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

// Static targets participate in the same OGC epoch as a dynamic rigid, but
// their shape pose is already world-space and has no 6DOF endpoint.  These
// queries intentionally use only the current pose; they are not swept tests.
static bool queryCurrentWorldStaticPairSignedDistance(
    const AvbdSoftContactGeometry &geometry,
    const physx::PxVec3 &queryPoint, physx::PxReal &signedDistance) {
  if (!geometry.hasWorldStaticTarget() || !queryPoint.isFinite())
    return false;
  if (!geometry.hasRigidBoxSdf) {
    const physx::PxReal normalLengthSq = geometry.normal.magnitudeSquared();
    if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f ||
        !geometry.surfacePoint.isFinite())
      return false;
    signedDistance = (queryPoint - geometry.surfacePoint).dot(
        geometry.normal * physx::PxRecipSqrt(normalLengthSq));
    return physx::PxIsFinite(signedDistance);
  }
  if (!geometry.rigidBoxPose.isValid() ||
      !geometry.rigidBoxHalfExtent.isFinite() ||
      geometry.rigidBoxHalfExtent.x <= 0.0f ||
      geometry.rigidBoxHalfExtent.y <= 0.0f ||
      geometry.rigidBoxHalfExtent.z <= 0.0f)
    return false;
  const physx::PxVec3 localPoint =
      geometry.rigidBoxPose.transformInv(queryPoint);
  const physx::PxVec3 delta(
      physx::PxAbs(localPoint.x) - geometry.rigidBoxHalfExtent.x,
      physx::PxAbs(localPoint.y) - geometry.rigidBoxHalfExtent.y,
      physx::PxAbs(localPoint.z) - geometry.rigidBoxHalfExtent.z);
  const physx::PxVec3 outside(
      physx::PxMax(delta.x, 0.0f), physx::PxMax(delta.y, 0.0f),
      physx::PxMax(delta.z, 0.0f));
  signedDistance = outside.magnitude() + physx::PxMin(
      physx::PxMax(delta.x, physx::PxMax(delta.y, delta.z)), 0.0f);
  return physx::PxIsFinite(signedDistance);
}

static bool queryCurrentWorldStaticPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    physx::PxReal &faceGap) {
  if (!geometry.hasWorldStaticTarget() || !geometry.hasRigidBoxSdf ||
      !geometry.hasRigidBoxTriangleCoreExit || !softParticles ||
      !movedParticleDisplacement.isFinite() || !geometry.rigidBoxPose.isValid() ||
      !geometry.rigidBoxHalfExtent.isFinite() ||
      !geometry.rigidBoxTriangleCoreExitNormalLocal.isFinite())
    return false;
  const physx::PxReal normalLengthSq =
      geometry.rigidBoxTriangleCoreExitNormalLocal.magnitudeSquared();
  if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalLocal =
      geometry.rigidBoxTriangleCoreExitNormalLocal *
      physx::PxRecipSqrt(normalLengthSq);
  const physx::PxVec3 normalWorld =
      geometry.rigidBoxPose.q.rotate(normalLocal).getNormalized();
  const physx::PxU32 axis = normalLocal.x != 0.0f ? 0u :
                             (normalLocal.y != 0.0f ? 1u : 2u);
  const physx::PxVec3 surfaceWorld = geometry.rigidBoxPose.transform(
      normalLocal * geometry.rigidBoxHalfExtent[axis]);
  if (!normalWorld.isFinite() || !surfaceWorld.isFinite())
    return false;
  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
    const AvbdWeightedContactPoint &mapping =
        geometry.rigidBoxTriangleCorePoints[vertex];
    if (mapping.count == 0 || mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    physx::PxVec3 point(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 support = 0; support < mapping.count; ++support) {
      const physx::PxU32 index = mapping.particleIndices[support];
      const physx::PxReal weight = mapping.weights[support];
      if (index >= numSoftParticles || !physx::PxIsFinite(weight) ||
          !softParticles[index].position.isFinite())
        return false;
      point += softParticles[index].position * weight;
      if (index == movedParticleIndex)
        point += movedParticleDisplacement * weight;
      weightSum += weight;
    }
    if (!point.isFinite() || !physx::PxIsFinite(weightSum) ||
        physx::PxAbs(weightSum - 1.0f) > 1.0e-3f)
      return false;
    faceGap = physx::PxMin(faceGap, (point - surfaceWorld).dot(normalWorld));
  }
  return physx::PxIsFinite(faceGap);
}

// Evaluate the complete collision triangle on the same frame-start ->
// endpoint interpolation used by OGC warmstart admission.  This is a
// trust-region query, not swept CCD: it only finds the largest admissible
// position update within the already selected simulation dt.  Keeping the
// triangle here (rather than its detector centroid) makes the warmstart
// invariant match the terminal triangle-core manifold.
static bool queryInterpolatedPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSolverBody &initialBody, const AvbdSolverBody &endpointBody,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal alpha, physx::PxReal &faceGap) {
  if (!softParticles || !physx::PxIsFinite(alpha) || alpha < 0.0f ||
      alpha > 1.0f || !initialBody.position.isFinite() ||
      !initialBody.rotation.isFinite() || !endpointBody.position.isFinite() ||
      !endpointBody.rotation.isFinite() || !geometry.hasRigidBoxSdf ||
      !geometry.hasRigidBoxTriangleCoreExit ||
      !geometry.rigidBoxPose.p.isFinite() ||
      !geometry.rigidBoxPose.q.isFinite() ||
      !geometry.rigidBoxHalfExtent.isFinite() ||
      !geometry.rigidBoxTriangleCoreExitNormalLocal.isFinite())
    return false;

  const physx::PxVec3 rawNormalLocal =
      geometry.rigidBoxTriangleCoreExitNormalLocal;
  const physx::PxReal normalLengthSq = rawNormalLocal.magnitudeSquared();
  const physx::PxVec3 halfExtent = geometry.rigidBoxHalfExtent;
  if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f ||
      halfExtent.x <= 0.0f || halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;

  const physx::PxVec3 normalLocal =
      rawNormalLocal * physx::PxRecipSqrt(normalLengthSq);
  AvbdSolverBody body = endpointBody;
  body.position = initialBody.position +
      (endpointBody.position - initialBody.position) * alpha;
  body.rotation =
      physx::PxSlerp(alpha, initialBody.rotation, endpointBody.rotation);
  body.projectLockedPose(initialBody.position, initialBody.rotation);
  if (!body.position.isFinite() || !body.rotation.isFinite())
    return false;

  const physx::PxQuat shapeRotation = body.rotation * geometry.rigidBoxPose.q;
  const physx::PxVec3 shapePosition =
      body.position + body.rotation.rotate(geometry.rigidBoxPose.p);
  const physx::PxVec3 rawNormalWorld = shapeRotation.rotate(normalLocal);
  const physx::PxReal worldNormalLengthSq = rawNormalWorld.magnitudeSquared();
  if (!shapeRotation.isFinite() || !shapePosition.isFinite() ||
      !rawNormalWorld.isFinite() || !physx::PxIsFinite(worldNormalLengthSq) ||
      worldNormalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalWorld =
      rawNormalWorld * physx::PxRecipSqrt(worldNormalLengthSq);
  const physx::PxVec3 surfaceLocal(normalLocal.x * halfExtent.x,
                                    normalLocal.y * halfExtent.y,
                                    normalLocal.z * halfExtent.z);
  const physx::PxVec3 surfaceWorld =
      shapePosition + shapeRotation.rotate(surfaceLocal);

  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
    const AvbdWeightedContactPoint &mapping =
        geometry.rigidBoxTriangleCorePoints[vertex];
    if (mapping.count == 0 || mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    physx::PxVec3 point(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 support = 0; support < mapping.count; ++support) {
      const physx::PxU32 particleIndex = mapping.particleIndices[support];
      const physx::PxReal weight = mapping.weights[support];
      if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight) ||
          !softParticles[particleIndex].initialPosition.isFinite() ||
          !softParticles[particleIndex].position.isFinite())
        return false;
      const physx::PxVec3 position =
          softParticles[particleIndex].initialPosition +
          (softParticles[particleIndex].position -
           softParticles[particleIndex].initialPosition) * alpha;
      point += position * weight;
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

// OGC's admission point is the pair prediction, not a post-penetration
// ejection.  Restricting only a rigid (or only the four simulation vertices
// behind one collision vertex) transfers the load into the wrong endpoint:
// the former launches the rigid and the latter shears/inverts the tet mesh.
//
// This one-time same-dt line search instead advances every endpoint of the
// active mixed island to one common feasible DCD epoch.  A source volume is
// moved coherently from its accepted frame-start state to its warmstarted
// candidate, and its paired 6DOF rigid uses the same alpha.  The regular
// coupled AVBD/OGC rows then distribute subsequent pressure locally through
// the soft support.  It is neither a sweep/TOI query nor a physics substep.
//
// World-static targets use the companion routine below.  They have no rigid
// endpoint to interpolate, so their admissible initial guess must be clipped
// at the *collision support* level.  Scaling an entire falling volume here
// makes gravity look disabled; scaling just the weighted query/core supports
// is the OGC vertex trust-region operation that lets the material block form
// the visible landing deformation.
static void applyWorldStaticOgcInitialAdmission(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts) {
  if (!softParticles || !softBodies || !softContacts || numSoftParticles == 0)
    return;
  const physx::PxReal clearance = 1.0e-5f;
  physx::PxArray<physx::PxReal> admissionAlphas(numSoftParticles, 1.0f);
  auto evaluatePoint = [&](const AvbdWeightedContactPoint &mapping,
                           physx::PxReal alpha,
                           physx::PxVec3 &point) -> bool {
    if (mapping.count == 0 || mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    point = physx::PxVec3(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 i = 0; i < mapping.count; ++i) {
      const physx::PxU32 index = mapping.particleIndices[i];
      const physx::PxReal weight = mapping.weights[i];
      if (index >= numSoftParticles || !physx::PxIsFinite(weight) ||
          !softParticles[index].initialPosition.isFinite() ||
          !softParticles[index].position.isFinite())
        return false;
      point += (softParticles[index].initialPosition +
                (softParticles[index].position -
                 softParticles[index].initialPosition) * alpha) * weight;
      weightSum += weight;
    }
    return point.isFinite() && physx::PxIsFinite(weightSum) &&
        physx::PxAbs(weightSum - 1.0f) <= 1.0e-3f;
  };
  auto limitPoint = [&](const AvbdWeightedContactPoint &mapping,
                        physx::PxReal alpha) {
    for (physx::PxU32 i = 0; i < mapping.count; ++i) {
      const physx::PxU32 index = mapping.particleIndices[i];
      if (index >= numSoftParticles)
        continue;
      admissionAlphas[index] = physx::PxMin(admissionAlphas[index], alpha);
    }
  };
  auto sourceIsDynamic = [&](const AvbdSoftContactGeometry &geometry) {
    return geometry.queryBodyIndex < numSoftBodies &&
        !softBodies[geometry.queryBodyIndex].compiled.speculativeCCDEnabled &&
        avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles);
  };
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (!geometry.hasWorldStaticTarget() || !sourceIsDynamic(geometry))
      continue;

    auto admitPoint = [&](const AvbdWeightedContactPoint &mapping) {
      physx::PxVec3 initialPoint(0.0f), endpointPoint(0.0f);
      physx::PxReal initialGap = 0.0f, endpointGap = 0.0f;
      if (!evaluatePoint(mapping, 0.0f, initialPoint) ||
          !evaluatePoint(mapping, 1.0f, endpointPoint) ||
          !queryCurrentWorldStaticPairSignedDistance(
              geometry, initialPoint, initialGap) ||
          !queryCurrentWorldStaticPairSignedDistance(
              geometry, endpointPoint, endpointGap) ||
          initialGap <= clearance || endpointGap >= clearance)
        return;
      physx::PxReal lower = 0.0f;
      physx::PxReal upper = 1.0f;
      for (physx::PxU32 iteration = 0; iteration < 10u; ++iteration) {
        const physx::PxReal middle = 0.5f * (lower + upper);
        physx::PxVec3 point(0.0f);
        physx::PxReal gap = 0.0f;
        if (evaluatePoint(mapping, middle, point) &&
            queryCurrentWorldStaticPairSignedDistance(geometry, point, gap) &&
            gap >= clearance)
          lower = middle;
        else
          upper = middle;
      }
      limitPoint(mapping, lower);
    };

    // A TBIX certificate owns its complete collision triangle.  Clip all
    // three embedded vertices together on its selected support plane, which
    // prevents an all-outside triangle interior from crossing a static OBB.
    if (geometry.hasRigidBoxTriangleCoreExit && geometry.hasRigidBoxSdf) {
      auto coreGap = [&](physx::PxReal alpha, physx::PxReal &gap) -> bool {
        if (!geometry.rigidBoxPose.isValid() ||
            !geometry.rigidBoxTriangleCoreExitNormalLocal.isFinite())
          return false;
        const physx::PxReal normalLengthSq =
            geometry.rigidBoxTriangleCoreExitNormalLocal.magnitudeSquared();
        if (normalLengthSq <= 1.0e-12f || !physx::PxIsFinite(normalLengthSq))
          return false;
        const physx::PxVec3 normal = geometry.rigidBoxPose.q.rotate(
            geometry.rigidBoxTriangleCoreExitNormalLocal *
            physx::PxRecipSqrt(normalLengthSq));
        const physx::PxU32 axis =
            geometry.rigidBoxTriangleCoreExitNormalLocal.x != 0.0f ? 0u :
            (geometry.rigidBoxTriangleCoreExitNormalLocal.y != 0.0f ? 1u : 2u);
        const physx::PxVec3 surface = geometry.rigidBoxPose.transform(
            (geometry.rigidBoxTriangleCoreExitNormalLocal *
             physx::PxRecipSqrt(normalLengthSq)) *
            geometry.rigidBoxHalfExtent[axis]);
        gap = PX_MAX_F32;
        for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
          physx::PxVec3 point(0.0f);
          if (!evaluatePoint(geometry.rigidBoxTriangleCorePoints[vertex],
                             alpha, point))
            return false;
          gap = physx::PxMin(gap, (point - surface).dot(normal));
        }
        return physx::PxIsFinite(gap);
      };
      physx::PxReal initialGap = 0.0f, endpointGap = 0.0f;
      if (coreGap(0.0f, initialGap) && coreGap(1.0f, endpointGap) &&
          initialGap > clearance && endpointGap < clearance) {
        physx::PxReal lower = 0.0f, upper = 1.0f;
        for (physx::PxU32 iteration = 0; iteration < 10u; ++iteration) {
          const physx::PxReal middle = 0.5f * (lower + upper);
          physx::PxReal gap = 0.0f;
          if (coreGap(middle, gap) && gap >= clearance)
            lower = middle;
          else
            upper = middle;
        }
        for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex)
          limitPoint(geometry.rigidBoxTriangleCorePoints[vertex], lower);
      }
    }

    AvbdWeightedContactPoint query;
    if (geometry.hasWeightedQueryPoint())
      query = geometry.queryPoint;
    else if (geometry.hasBarycentricQueryPoint()) {
      for (physx::PxU32 i = 0; i < 3u; ++i) {
        if (geometry.queryParticleIndices[i] == PX_MAX_U32)
          break;
        if (!query.appendMerged(geometry.queryParticleIndices[i],
                                geometry.queryWeights[i])) {
          query.clear();
          break;
        }
      }
    } else
      query.appendMerged(geometry.particleIdx, 1.0f);
    if (query.count > 0)
      admitPoint(query);
  }
  // Commit each collision support once. Multiple feature rows may share an
  // embedded simulation vertex; their minimum admissible alpha is its OGC
  // trust-region bound for this epoch, independent of contact enumeration.
  for (physx::PxU32 index = 0; index < numSoftParticles; ++index) {
    const physx::PxReal alpha = admissionAlphas[index];
    if (!physx::PxIsFinite(alpha) || alpha >= 1.0f)
      continue;
    AvbdSoftParticle &particle = softParticles[index];
    const physx::PxVec3 initial = particle.initialPosition;
    particle.position = initial + (particle.position - initial) * alpha;
    particle.predictedPosition = initial +
        (particle.predictedPosition - initial) * alpha;
    particle.outerPosition = initial +
        (particle.outerPosition - initial) * alpha;
  }
}

static void applyOgcMixedWarmstartAdmission(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxArray<physx::PxU8> *admittedContacts,
    physx::PxArray<physx::PxReal> *admittedNormalDisplacements) {
  if (admittedContacts) {
    admittedContacts->resize(numSoftContacts);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      (*admittedContacts)[sci] = 0u;
  }
  if (admittedNormalDisplacements) {
    admittedNormalDisplacements->resize(numSoftContacts);
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
      (*admittedNormalDisplacements)[sci] = 0.0f;
  }
  if (!bodies || !softParticles || !softBodies || !softContacts ||
      numBodies == 0 || numSoftBodies == 0 || numSoftContacts == 0)
    return;

  // Boundary admission is a time-domain trust-region operation, not a
  // global scene rewind.  Build the bipartite soft/rigid pair graph first;
  // only endpoints connected through an actual mixed OGC pair may share an
  // alpha.  The former single alpha coupled unrelated islands and could
  // suppress a volume's gravity/material response merely because a different
  // free rigid crossed its own contact boundary.
  const physx::PxU32 nodeCount = numSoftBodies + numBodies;
  physx::PxArray<physx::PxU32> parent(nodeCount);
  physx::PxArray<physx::PxU8> participating(nodeCount, 0u);
  physx::PxArray<physx::PxReal> contactAlphas(numSoftContacts, 1.0f);
  for (physx::PxU32 node = 0; node < nodeCount; ++node)
    parent[node] = node;
  auto findRoot = [&parent](physx::PxU32 node) {
    physx::PxU32 root = node;
    while (parent[root] != root)
      root = parent[root];
    while (parent[node] != node) {
      const physx::PxU32 next = parent[node];
      parent[node] = root;
      node = next;
    }
    return root;
  };
  auto unite = [&findRoot, &parent](physx::PxU32 a, physx::PxU32 b) {
    const physx::PxU32 rootA = findRoot(a);
    const physx::PxU32 rootB = findRoot(b);
    if (rootA != rootB)
      parent[rootB] = rootA;
  };
  bool hasMixedPair = false;
  const physx::PxReal clearance = 1.0e-5f;

  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() ||
        geometry.targetIndex >= numBodies ||
        geometry.queryBodyIndex >= numSoftBodies ||
        softBodies[geometry.queryBodyIndex].compiled.speculativeCCDEnabled ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    const AvbdSolverBody &body = bodies[geometry.targetIndex];
    if (body.invMass <= 0.0f || !body.position.isFinite() ||
        !body.rotation.isFinite() || !body.prevPosition.isFinite() ||
        !body.prevRotation.isFinite())
      continue;

    const physx::PxU32 softNode = geometry.queryBodyIndex;
    const physx::PxU32 rigidNode = numSoftBodies + geometry.targetIndex;
    participating[softNode] = 1u;
    participating[rigidNode] = 1u;
    unite(softNode, rigidNode);
    hasMixedPair = true;

    const physx::PxVec3 initialQuery =
        avbdGetSoftContactInitialQueryPoint(geometry, softParticles);
    const physx::PxVec3 endpointQuery =
        avbdGetSoftContactQueryPoint(geometry, softParticles);

    AvbdSolverBody initialBody = body;
    initialBody.position = body.prevPosition;
    initialBody.rotation = body.prevRotation;
    physx::PxReal initialGap = 0.0f;
    physx::PxReal endpointGap = 0.0f;
    const bool hasPointGaps = initialQuery.isFinite() &&
        endpointQuery.isFinite() &&
        queryCurrentPairSignedDistance(geometry, initialBody, initialQuery,
                                       initialGap) &&
        queryCurrentPairSignedDistance(geometry, body, endpointQuery,
                                       endpointGap) &&
        physx::PxIsFinite(initialGap) && physx::PxIsFinite(endpointGap);
    const bool requiresPointAdmission = hasPointGaps &&
        ((initialGap > clearance && endpointGap < clearance) ||
         (initialGap <= clearance && endpointGap < initialGap - clearance));

    // The detector's triangle-core witness is not merely another centroid
    // row.  When the full face crosses a box while its compact AL query stays
    // outside, the old admission path accepted the illegal endpoint and paid
    // for a terminal full-proxy rebuild every frame.  Carry the complete
    // triangle through the same DCD trust region so the solve begins from a
    // valid OGC epoch for both endpoints.
    physx::PxReal initialCoreGap = 0.0f;
    physx::PxReal endpointCoreGap = 0.0f;
    const bool hasCoreGaps = geometry.hasRigidBoxTriangleCoreExit &&
        queryInterpolatedPairTriangleCoreFaceGap(
            geometry, initialBody, body, softParticles, numSoftParticles,
            0.0f, initialCoreGap) &&
        queryInterpolatedPairTriangleCoreFaceGap(
            geometry, initialBody, body, softParticles, numSoftParticles,
            1.0f, endpointCoreGap) &&
        physx::PxIsFinite(initialCoreGap) &&
        physx::PxIsFinite(endpointCoreGap);
    const bool requiresCoreAdmission = hasCoreGaps &&
        ((initialCoreGap > clearance && endpointCoreGap < clearance) ||
         (initialCoreGap <= clearance &&
          endpointCoreGap < initialCoreGap - clearance));
    if (!requiresPointAdmission && !requiresCoreAdmission)
      continue;

    auto hasClearance = [&](physx::PxReal alpha) {
      if (requiresPointAdmission) {
        AvbdSolverBody candidate = body;
        candidate.position = body.prevPosition +
            (body.position - body.prevPosition) * alpha;
        candidate.rotation = PxSlerp(alpha, body.prevRotation, body.rotation);
        candidate.projectLockedPose(body.prevPosition, body.prevRotation);
        const physx::PxVec3 query = initialQuery +
            (endpointQuery - initialQuery) * alpha;
        physx::PxReal gap = 0.0f;
        if (!queryCurrentPairSignedDistance(geometry, candidate, query, gap) ||
            !physx::PxIsFinite(gap) || gap < clearance)
          return false;
      }
      if (requiresCoreAdmission) {
        physx::PxReal coreGap = 0.0f;
        if (!queryInterpolatedPairTriangleCoreFaceGap(
                geometry, initialBody, body, softParticles, numSoftParticles,
                alpha, coreGap) || !physx::PxIsFinite(coreGap) ||
            coreGap < clearance)
          return false;
      }
      return true;
    };

    physx::PxReal lower = 0.0f;
    physx::PxReal upper = 1.0f;
    for (physx::PxU32 iteration = 0; iteration < 10u; ++iteration) {
      const physx::PxReal middle = 0.5f * (lower + upper);
      if (hasClearance(middle))
        lower = middle;
      else
        upper = middle;
    }
    contactAlphas[sci] = lower;
    // This row alone is eligible to own the e=0 impact response after the
    // position solve.  The marker records a real same-dt DCD boundary
    // crossing; it is not a generic OGC-shell or CCD flag.
    if (admittedContacts)
      (*admittedContacts)[sci] = 1u;
    if (admittedNormalDisplacements) {
      // Keep only the inward endpoint motion that was clipped by this DCD
      // boundary.  The value is a geometric work request; it is converted
      // to a bounded force by the pair-major soft solve below.
      physx::PxReal displacement = 0.0f;
      if (requiresPointAdmission && hasPointGaps)
        displacement = physx::PxMax(displacement,
                                    clearance - endpointGap);
      if (requiresCoreAdmission && hasCoreGaps)
        displacement = physx::PxMax(displacement,
                                    clearance - endpointCoreGap);
      (*admittedNormalDisplacements)[sci] =
          physx::PxClamp(displacement, 0.0f, 0.05f);
    }
  }

  if (!hasMixedPair)
    return;

  physx::PxArray<physx::PxReal> componentAlpha(nodeCount, 1.0f);
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const physx::PxReal contactAlpha = contactAlphas[sci];
    if (!physx::PxIsFinite(contactAlpha) || contactAlpha >= 1.0f)
      continue;
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() ||
        geometry.queryBodyIndex >= numSoftBodies ||
        geometry.targetIndex >= numBodies)
      continue;
    const physx::PxU32 root = findRoot(geometry.queryBodyIndex);
    componentAlpha[root] = physx::PxMin(componentAlpha[root],
                                         physx::PxMax(contactAlpha, 0.0f));
  }

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    const physx::PxU32 node = numSoftBodies + bodyIndex;
    if (!participating[node])
      continue;
    const physx::PxReal commonAlpha = componentAlpha[findRoot(node)];
    if (!physx::PxIsFinite(commonAlpha) || commonAlpha >= 1.0f)
      continue;
    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxVec3 inertialPosition = body.inertialPosition;
    const physx::PxQuat inertialRotation = body.inertialRotation;
    body.position = body.prevPosition +
        (body.position - body.prevPosition) * commonAlpha;
    body.rotation = PxSlerp(commonAlpha, body.prevRotation, body.rotation);
    body.inertialPosition = body.prevPosition +
        (inertialPosition - body.prevPosition) * commonAlpha;
    body.inertialRotation = PxSlerp(commonAlpha, body.prevRotation,
                                    inertialRotation);
    body.projectLockedPose(body.prevPosition, body.prevRotation);
  }

  for (physx::PxU32 sourceBodyIndex = 0;
       sourceBodyIndex < numSoftBodies; ++sourceBodyIndex) {
    if (!participating[sourceBodyIndex])
      continue;
    const physx::PxReal commonAlpha =
        componentAlpha[findRoot(sourceBodyIndex)];
    if (!physx::PxIsFinite(commonAlpha) || commonAlpha >= 1.0f)
      continue;
    const AvbdSoftBody &sourceBody = softBodies[sourceBodyIndex];
    const physx::PxU32 particleStart = sourceBody.compiled.particleStart;
    const physx::PxU32 particleCount = sourceBody.compiled.particleCount;
    if (particleStart > numSoftParticles ||
        particleCount > numSoftParticles - particleStart)
      continue;
    for (physx::PxU32 localIndex = 0; localIndex < particleCount;
         ++localIndex) {
      AvbdSoftParticle &particle = softParticles[particleStart + localIndex];
      const physx::PxVec3 initial = particle.initialPosition;
      particle.position = initial + (particle.position - initial) * commonAlpha;
      particle.predictedPosition =
          initial + (particle.predictedPosition - initial) * commonAlpha;
      particle.outerPosition =
          initial + (particle.outerPosition - initial) * commonAlpha;
    }
  }
}

static physx::PxReal limitRigidOgcCandidate(
    const AvbdOgcPairTrustRegionContext *context,
    physx::PxU32 bodyIndex, const AvbdSolverBody &body,
    const physx::PxVec3 &deltaPosition,
    const physx::PxVec3 &deltaTheta,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxU32 *rigidTargetContactStarts,
    const physx::PxU32 *rigidTargetContactRefs,
    const AvbdSoftParticle *softParticles) {
  if (!context || !context->isComplete(numSoftContacts) ||
      !rigidTargetContactStarts || !rigidTargetContactRefs ||
      !softParticles || !deltaPosition.isFinite() || !deltaTheta.isFinite())
    return 1.0f;

  const physx::PxU32 begin = rigidTargetContactStarts[bodyIndex];
  const physx::PxU32 end = rigidTargetContactStarts[bodyIndex + 1u];
  if (begin >= end)
    return 1.0f;

  const physx::PxReal tolerance = 1.0e-6f;
  physx::PxReal alpha = 1.0f;
  for (physx::PxU32 refIndex = begin; refIndex < end; ++refIndex) {
    const physx::PxU32 contactIndex = rigidTargetContactRefs[refIndex];
    if (contactIndex >= numSoftContacts)
      continue;
    const physx::PxU32 pairIndex = context->contactPairIndices[contactIndex];
    if (pairIndex >= context->numPairStates ||
        !context->pairStates[pairIndex].active)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    if (!geometry.hasRigidBodyTarget() || geometry.targetIndex != bodyIndex ||
        pair.sourceType != geometry.source.type ||
        pair.targetKind != geometry.targetKind ||
        pair.targetBodyIndex != geometry.targetIndex ||
        pair.primitiveKey != geometry.source.primitiveKey)
      continue;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    physx::PxReal currentGap = 0.0f;
    if (!queryCurrentPairSignedDistance(geometry, body, queryPoint,
                                        currentGap))
      continue;

    // Candidate rotations are evaluated directly (rather than through the
    // linearized r x n term), then a short monotone bisection finds the
    // largest coupled 6DOF step which retains positive clearance.
    auto candidateHasClearance = [&](physx::PxReal candidateAlpha) {
      AvbdSolverBody candidate = body;
      candidate.position -= deltaPosition * candidateAlpha;
      const physx::PxVec3 theta = deltaTheta * candidateAlpha;
      if (theta.magnitudeSquared() > 1.0e-12f) {
        const physx::PxQuat dq(theta.x, theta.y, theta.z, 0.0f);
        candidate.rotation =
            (candidate.rotation - dq * candidate.rotation * 0.5f)
                .getNormalized();
      }
      candidate.projectLockedPose(body.prevPosition, body.prevRotation);
      physx::PxReal candidateGap = 0.0f;
      return queryCurrentPairSignedDistance(geometry, candidate, queryPoint,
                                            candidateGap) &&
             candidateGap >= tolerance;
    };
    // A body already held on an OGC boundary must still reject an inward GS
    // update.  The ordinary bisection assumes alpha=0 has positive clearance;
    // when a previous row left a tiny negative round-off residual that premise
    // is false, so retain only a fully separating candidate and otherwise
    // hold this positional update.  Terminal current-pose recovery remains
    // responsible for repairing an already-invalid state.
    if (currentGap < tolerance) {
      if (!candidateHasClearance(alpha)) {
        pair.refreshRequested = true;
        pair.remainingSafeDisplacement = 0.0f;
        pair.minimumGap = physx::PxMin(pair.minimumGap, currentGap);
        pair.accumulatedRelativeDisplacement = physx::PxMax(
            pair.accumulatedRelativeDisplacement, deltaPosition.magnitude());
        alpha = 0.0f;
      }
      continue;
    }
    if (candidateHasClearance(alpha))
      continue;
    physx::PxReal lo = 0.0f;
    physx::PxReal hi = alpha;
    for (physx::PxU32 iteration = 0; iteration < 8u; ++iteration) {
      const physx::PxReal mid = 0.5f * (lo + hi);
      if (candidateHasClearance(mid))
        lo = mid;
      else
        hi = mid;
    }
    if (lo < 1.0f - 1.0e-6f) {
      pair.refreshRequested = true;
      pair.remainingSafeDisplacement = 0.0f;
      pair.minimumGap = physx::PxMin(pair.minimumGap, tolerance);
      pair.accumulatedRelativeDisplacement = physx::PxMax(
          pair.accumulatedRelativeDisplacement, deltaPosition.magnitude());
    }
    alpha = lo;
  }
  return physx::PxIsFinite(alpha) ? alpha : 0.0f;
}

static physx::PxVec3 computeGeneric1DRotationDelta(
    const physx::PxQuat &current, const physx::PxQuat &reference) {
  physx::PxQuat delta = current * reference.getConjugate();
  if (delta.w < 0.0f)
    delta = -delta;

  const physx::PxVec3 imaginary(delta.x, delta.y, delta.z);
  const physx::PxReal sinHalfSquared = imaginary.magnitudeSquared();
  if (sinHalfSquared <= 1e-20f)
    return imaginary * 2.0f;

  const physx::PxReal sinHalf = physx::PxSqrt(sinHalfSquared);
  const physx::PxReal angle = 2.0f * physx::PxAtan2(
      sinHalf, physx::PxClamp(delta.w, 0.0f, 1.0f));
  return imaginary * (angle / sinHalf);
}

static physx::PxReal computeGeneric1DViolation(
    const AvbdD6JointConstraint &joint, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxReal dt) {
  physx::PxReal violation =
      joint.genericGeometricError - joint.genericVelocityTarget * dt;
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;

  if (bodyA < numBodies) {
    violation += joint.genericLinearA.dot(
        bodies[bodyA].position - joint.genericReferencePositionA);
    violation += joint.genericAngularA.dot(computeGeneric1DRotationDelta(
        bodies[bodyA].rotation, joint.genericReferenceRotationA));
  }
  if (bodyB < numBodies) {
    violation += joint.genericLinearB.dot(
        bodies[bodyB].position - joint.genericReferencePositionB);
    violation += joint.genericAngularB.dot(computeGeneric1DRotationDelta(
        bodies[bodyB].rotation, joint.genericReferenceRotationB));
  }
  return violation;
}

static physx::PxReal computeGeneric1DEffectiveMass(
    const AvbdD6JointConstraint &joint, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies) {
  physx::PxReal unitResponse = 0.0f;
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (bodyA < numBodies) {
    unitResponse +=
        bodies[bodyA].invMass * joint.genericLinearA.magnitudeSquared();
    unitResponse += joint.genericAngularA.dot(
        bodies[bodyA].invInertiaWorld * joint.genericAngularA);
  }
  if (bodyB < numBodies) {
    unitResponse +=
        bodies[bodyB].invMass * joint.genericLinearB.magnitudeSquared();
    unitResponse += joint.genericAngularB.dot(
        bodies[bodyB].invInertiaWorld * joint.genericAngularB);
  }
  return unitResponse > 1e-10f ? 1.0f / unitResponse : 0.0f;
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

static bool isSingleNativeRevoluteMotorVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies == 0 || numBodies > 2 ||
      numContacts != 0 || !d6Joints || numD6 != 1 ||
      numGear != 0 || numSoftParticles != 0 || numSoftBodies != 0 ||
      numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const physx::PxU32 twistMotion = joint.getAngularMotion(0);
  const bool limitedTwist = twistMotion == 1u;
  const bool freeSpin =
      (joint.sourceFlags & AvbdD6JointConstraint::
                               eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0;
  const bool nonUnitDriveRatio =
      physx::PxAbs(joint.motorGearRatio - 1.0f) > 1e-6f;
  if (joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
      joint.motorEnabled == 0u ||
      !(joint.motorMaxForce > 0.0f) ||
      !physx::PxIsFinite(joint.motorMaxForce) ||
      !physx::PxIsFinite(joint.motorTargetVelocity) ||
      !physx::PxIsFinite(joint.motorGearRatio) ||
      !(joint.motorGearRatio > 0.0f) ||
      joint.linearMotion != 0u ||
      (twistMotion != 1u && twistMotion != 2u) ||
      joint.getAngularMotion(1) != 0u ||
      joint.getAngularMotion(2) != 0u ||
      joint.driveFlags != 0u || joint.driveAccelerationFlags != 0u ||
      joint.coneAngleLimit > 0.0f ||
      !joint.externalAngularStepA.isFinite() ||
      !joint.externalAngularStepB.isFinite())
    return false;
  if (limitedTwist &&
      (!physx::PxIsFinite(joint.angularLimitLower.x) ||
       !physx::PxIsFinite(joint.angularLimitUpper.x) ||
       joint.angularLimitLower.x >= joint.angularLimitUpper.x))
    return false;

  const physx::PxU32 endpoint[2] = {joint.header.bodyIndexA,
                                    joint.header.bodyIndexB};
  const bool dynamicA =
      endpoint[0] < numBodies && bodies[endpoint[0]].invMass > 0.0f;
  const bool dynamicB =
      endpoint[1] < numBodies && bodies[endpoint[1]].invMass > 0.0f;
  if ((!dynamicA && !dynamicB) ||
      (dynamicA && dynamicB && endpoint[0] == endpoint[1]))
    return false;
  // A limited motor may own either one centered dynamic endpoint against a
  // stationary endpoint or one centered dynamic pair. Both topologies share
  // the same scalar twist derivative and unilateral limit active set.
  // Off-center, off-principal, and prescribed-endpoint variants need a wider
  // bounded objective and remain fail closed below.
  if (limitedTwist) {
    const bool oneDynamic =
        numBodies == 1u && dynamicA != dynamicB;
    const bool dynamicPair =
        numBodies == 2u && dynamicA && dynamicB;
    if ((!oneDynamic && !dynamicPair) ||
        joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
        joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
      return false;
    if (dynamicPair) {
      if (joint.anchorA.magnitudeSquared() > 1e-8f ||
          joint.anchorB.magnitudeSquared() > 1e-8f)
        return false;
    } else {
      const physx::PxVec3 &dynamicAnchor =
          dynamicA ? joint.anchorA : joint.anchorB;
      if (dynamicAnchor.magnitudeSquared() > 1e-8f)
        return false;
    }
  }
  // Free-spin may likewise own one centered dynamic endpoint or one centered
  // principal-response dynamic pair. Its one-sided impulse row owns the
  // complete physical hinge derivative, including preservation of a
  // super-target solve-entry speed.
  if (freeSpin) {
    const bool oneDynamic =
        numBodies == 1u && dynamicA != dynamicB;
    const bool dynamicPair =
        numBodies == 2u && dynamicA && dynamicB;
    if ((!oneDynamic && !dynamicPair) ||
        joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
        joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
      return false;
    if (dynamicPair) {
      if (joint.anchorA.magnitudeSquared() > 1e-8f ||
          joint.anchorB.magnitudeSquared() > 1e-8f)
        return false;
    } else {
      const physx::PxVec3 &dynamicAnchor =
          dynamicA ? joint.anchorA : joint.anchorB;
      if (dynamicAnchor.magnitudeSquared() > 1e-8f)
        return false;
    }
  }
  if (freeSpin && limitedTwist)
    return false;
  if ((limitedTwist || freeSpin) && nonUnitDriveRatio)
    return false;
  if (nonUnitDriveRatio) {
    if (twistMotion != 2u || freeSpin || numBodies != 2u ||
        !dynamicA || !dynamicB ||
        joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
        joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
      return false;
    physx::PxVec3 localAxisA =
        joint.localFrameA.rotate(
            physx::PxVec3(1.0f, 0.0f, 0.0f));
    physx::PxVec3 localAxisB =
        joint.localFrameB.rotate(
            physx::PxVec3(1.0f, 0.0f, 0.0f));
    if (localAxisA.normalize() <= 1e-6f ||
        localAxisB.normalize() <= 1e-6f ||
        joint.anchorA.cross(localAxisA).magnitudeSquared() > 1e-8f ||
        joint.anchorB.cross(localAxisB).magnitudeSquared() > 1e-8f)
      return false;
  }

  physx::PxVec3 worldAxis =
      dynamicA
          ? (bodies[endpoint[0]].rotation * joint.localFrameA)
                .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
          : joint.localFrameA.rotate(
                physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (worldAxis.normalize() <= 1e-6f)
    return false;

  const bool allowCoupledOffPrincipalResponse =
      ((numBodies == 1u && dynamicA != dynamicB) ||
       (numBodies == 2u && dynamicA && dynamicB)) &&
      twistMotion == 2u && !freeSpin && !nonUnitDriveRatio &&
      joint.externalAngularStepA.magnitudeSquared() <= 1e-12f &&
      joint.externalAngularStepB.magnitudeSquared() <= 1e-12f;
  const bool allowCoupledOffCenterResponse =
      ((numBodies == 1u && dynamicA != dynamicB) ||
       (numBodies == 2u && dynamicA && dynamicB)) &&
      twistMotion == 2u && !freeSpin && !nonUnitDriveRatio &&
      joint.externalAngularStepA.magnitudeSquared() <= 1e-12f &&
      joint.externalAngularStepB.magnitudeSquared() <= 1e-12f;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    const bool dynamic = side == 0 ? dynamicA : dynamicB;
    const physx::PxVec3 localAxis =
        (side == 0 ? joint.localFrameA : joint.localFrameB)
            .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    const physx::PxVec3 anchor =
        side == 0 ? joint.anchorA : joint.anchorB;
    const physx::PxVec3 externalStep =
        side == 0 ? joint.externalAngularStepA
                  : joint.externalAngularStepB;
    if (!dynamic) {
      const physx::PxReal stepScale =
          physx::PxMax(externalStep.magnitude(), 1e-8f);
      // A prescribed endpoint may participate only when its motion is the
      // same physical hinge derivative owned by this scalar objective.
      if (externalStep.cross(worldAxis).magnitude() >
          stepScale * 1e-4f)
        return false;
      continue;
    }
    const AvbdSolverBody &body = bodies[endpoint[side]];
    if (body.lockFlags != 0 ||
        externalStep.magnitudeSquared() > 1e-12f)
      return false;
    // A perpendicular dynamic anchor couples the motor torque to locked
    // translation. The one-body free-twist topology below owns the complete
    // fixed-anchor spatial derivative; every wider topology remains rejected.
    // Collinear offsets are rotation-invariant.
    if (anchor.cross(localAxis).magnitudeSquared() > 1e-8f &&
        !allowCoupledOffCenterResponse)
      return false;
    const physx::PxVec3 response =
        body.invInertiaWorld.transform(worldAxis);
    const physx::PxReal responseScale =
        physx::PxMax(response.magnitude(), 1e-8f);
    // A post-finalize scalar motor row may not change the locked swing
    // derivative. The one-body free-twist topology below owns motor plus both
    // locked-swing derivatives as one complete angular objective; every
    // wider off-principal topology remains rejected.
    if (response.cross(worldAxis).magnitude() >
            responseScale * 1e-4f &&
        !allowCoupledOffPrincipalResponse)
      return false;
  }
  return true;
}

static bool isContactCoupledNativeRevoluteMotorVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const physx::PxVec3 &gravity, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts,
    bool &unsupportedTransientContact) {
  unsupportedTransientContact = false;
  if (!bodies || numBodies != 2u || !contacts || numContacts < 2u ||
      numContacts > 8u || !d6Joints || numD6 != 1u || numGear != 0u ||
      numSoftParticles != 0u || numSoftBodies != 0u ||
      numSoftContacts != 0u)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  if (bodyAIndex >= numBodies || bodyBIndex >= numBodies ||
      bodyAIndex == bodyBIndex ||
      bodies[bodyAIndex].invMass <= 0.0f ||
      bodies[bodyBIndex].invMass <= 0.0f ||
      bodies[bodyAIndex].lockFlags != 0u ||
      bodies[bodyBIndex].lockFlags != 0u ||
      bodies[bodyAIndex].linearDamping != 0.0f ||
      bodies[bodyBIndex].linearDamping != 0.0f ||
      bodies[bodyAIndex].angularDampingBody != 0.0f ||
      bodies[bodyBIndex].angularDampingBody != 0.0f ||
      physx::PxAbs(bodies[bodyAIndex].invMass -
                   bodies[bodyBIndex].invMass) >
          physx::PxMax(bodies[bodyAIndex].invMass,
                       bodies[bodyBIndex].invMass) *
              1e-5f ||
      joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
      joint.motorEnabled == 0u || !(joint.motorMaxForce > 0.0f) ||
      !physx::PxIsFinite(joint.motorMaxForce) ||
      !physx::PxIsFinite(joint.motorTargetVelocity) ||
      physx::PxAbs(joint.motorGearRatio - 1.0f) > 1e-6f ||
      joint.linearMotion != 0u || joint.getAngularMotion(0) != 2u ||
      joint.getAngularMotion(1) != 0u ||
      joint.getAngularMotion(2) != 0u ||
      joint.driveFlags != 0u || joint.driveAccelerationFlags != 0u ||
      joint.coneAngleLimit > 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0u ||
      joint.externalAngularStepA.magnitudeSquared() > 1e-12f ||
      joint.externalAngularStepB.magnitudeSquared() > 1e-12f)
    return false;

  physx::PxVec3 localAxisA =
      joint.localFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  physx::PxVec3 localAxisB =
      joint.localFrameB.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (localAxisA.normalize() <= 1e-6f ||
      localAxisB.normalize() <= 1e-6f ||
      joint.anchorA.cross(localAxisA).magnitudeSquared() > 1e-8f ||
      joint.anchorB.cross(localAxisB).magnitudeSquared() > 1e-8f)
    return false;

  physx::PxVec3 worldAxisA =
      (bodies[bodyAIndex].rotation * joint.localFrameA)
          .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (worldAxisA.normalize() <= 1e-6f)
    return false;

  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  physx::PxVec3 referenceNormal(0.0f);
  bool haveReferenceNormal = false;
  bool haveUnsupportedTransientContact = false;
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contact) ||
        hasKinematicShellAnchor(contact) ||
        (contact.friction <= 0.0f && contact.staticFriction <= 0.0f) ||
        !physx::PxIsFinite(contact.friction) ||
        !physx::PxIsFinite(contact.staticFriction) ||
        !physx::PxIsFinite(contact.restitution) ||
        contact.restitution < 0.0f ||
        contact.targetVelocity.magnitudeSquared() > 1e-12f)
      return false;
    if (contact.contactManagerEstablished == 0u)
      haveUnsupportedTransientContact = true;

    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicIsA ? contact.header.bodyIndexA
                   : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies ||
        physx::PxAbs((dynamicIsA ? contact.invMassScaleA
                                 : contact.invMassScaleB) -
                    1.0f) > 1e-6f ||
        physx::PxAbs((dynamicIsA ? contact.invInertiaScaleA
                                 : contact.invInertiaScaleB) -
                    1.0f) > 1e-6f)
      return false;
    contactsPerBody[bodyIndex]++;

    physx::PxVec3 dynamicNormal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    if (dynamicNormal.normalize() <= 1e-6f ||
        gravity.magnitudeSquared() <= 1e-12f ||
        -gravity.getNormalized().dot(dynamicNormal) < 0.9999f)
      return false;
    const physx::PxVec3 localPoint =
        dynamicIsA ? contact.contactPointA : contact.contactPointB;
    const physx::PxVec3 contactArm =
        bodies[bodyIndex].rotation.rotate(localPoint);
    const physx::PxVec3 pointVelocity =
        bodies[bodyIndex].linearVelocity +
        bodies[bodyIndex].angularVelocity.cross(contactArm);
    if (pointVelocity.dot(dynamicNormal) < -0.25f)
      haveUnsupportedTransientContact = true;
    if (!haveReferenceNormal) {
      referenceNormal = dynamicNormal;
      haveReferenceNormal = true;
    } else if (referenceNormal.dot(dynamicNormal) < 0.9999f) {
      return false;
    }
  }
  const bool completeSupport =
      contactsPerBody[0] != 0u && contactsPerBody[1] != 0u;
  unsupportedTransientContact =
      completeSupport && haveUnsupportedTransientContact;
  return completeSupport && !haveUnsupportedTransientContact;
}

struct ContactCoupledVelocityRow {
  physx::PxVec3 linearA;
  physx::PxVec3 angularA;
  physx::PxVec3 linearB;
  physx::PxVec3 angularB;
  physx::PxReal targetVelocity;
};

static void projectContactCoupledNativeRevoluteMotorVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint &joint, const physx::PxVec3 &gravity,
    physx::PxReal dt) {
  if (!bodies || numBodies != 2u || !contacts || numContacts == 0u ||
      !(dt > 0.0f))
    return;

  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  if (bodyAIndex >= numBodies || bodyBIndex >= numBodies)
    return;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  physx::PxQuat frameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = frameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !physx::PxIsFinite(frameMagnitude))
    return;
  frameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  const physx::PxVec3 axes[3] = {
      frameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f)),
      frameA.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f)),
      frameA.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f))};
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);

  ContactCoupledVelocityRow hardRows[5];
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    hardRows[axis].linearA = -axes[axis];
    hardRows[axis].angularA = -rA.cross(axes[axis]);
    hardRows[axis].linearB = axes[axis];
    hardRows[axis].angularB = rB.cross(axes[axis]);
    hardRows[axis].targetVelocity = 0.0f;
  }
  for (physx::PxU32 swing = 0; swing < 2; ++swing) {
    hardRows[3u + swing].linearA = physx::PxVec3(0.0f);
    hardRows[3u + swing].angularA = -axes[1u + swing];
    hardRows[3u + swing].linearB = physx::PxVec3(0.0f);
    hardRows[3u + swing].angularB = axes[1u + swing];
    hardRows[3u + swing].targetVelocity = 0.0f;
  }
  ContactCoupledVelocityRow motorRow;
  motorRow.linearA = physx::PxVec3(0.0f);
  motorRow.angularA = -axes[0];
  motorRow.linearB = physx::PxVec3(0.0f);
  motorRow.angularB = axes[0];
  motorRow.targetVelocity = joint.motorTargetVelocity;

  auto rowVelocity =
      [&](const ContactCoupledVelocityRow &row) -> physx::PxReal {
    return row.linearA.dot(bodyA.linearVelocity) +
           row.angularA.dot(bodyA.angularVelocity) +
           row.linearB.dot(bodyB.linearVelocity) +
           row.angularB.dot(bodyB.angularVelocity);
  };
  auto rowResponse =
      [&](const ContactCoupledVelocityRow &row) -> physx::PxReal {
    return bodyA.invMass * row.linearA.magnitudeSquared() +
           row.angularA.dot(
               bodyA.invInertiaWorld.transform(row.angularA)) +
           bodyB.invMass * row.linearB.magnitudeSquared() +
           row.angularB.dot(
               bodyB.invInertiaWorld.transform(row.angularB));
  };
  auto applyRowImpulse =
      [&](const ContactCoupledVelocityRow &row,
          physx::PxReal impulse) {
    bodyA.linearVelocity += row.linearA * (bodyA.invMass * impulse);
    bodyA.angularVelocity +=
        bodyA.invInertiaWorld.transform(row.angularA * impulse);
    bodyB.linearVelocity += row.linearB * (bodyB.invMass * impulse);
    bodyB.angularVelocity +=
        bodyB.invInertiaWorld.transform(row.angularB * impulse);
  };

  physx::PxArray<physx::PxReal> frictionImpulse0(numContacts);
  physx::PxArray<physx::PxReal> frictionImpulse1(numContacts);
  physx::PxArray<physx::PxReal> normalImpulse(numContacts);
  physx::PxArray<physx::PxReal> frictionMu(numContacts);
  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    if (bodyIndex < numBodies)
      contactsPerBody[bodyIndex]++;
  }
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    AvbdContactConstraint &contact = contacts[contactIndex];
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicIsA ? contact.header.bodyIndexA
                   : contact.header.bodyIndexB;
    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxReal reportSign = dynamicIsA ? 1.0f : -1.0f;
    frictionImpulse0[contactIndex] =
        reportSign *
        contact.frictionSweepImpulse.dot(contact.tangent0);
    frictionImpulse1[contactIndex] =
        reportSign *
        contact.frictionSweepImpulse.dot(contact.tangent1);
    const physx::PxReal mu =
        contact.friction > 0.0f ? contact.friction
                               : contact.staticFriction;
    frictionMu[contactIndex] = mu;
    const physx::PxReal existingMagnitude = physx::PxSqrt(
        frictionImpulse0[contactIndex] *
            frictionImpulse0[contactIndex] +
        frictionImpulse1[contactIndex] *
            frictionImpulse1[contactIndex]);
    physx::PxVec3 dynamicNormal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    dynamicNormal.normalize();
    const physx::PxReal mass = 1.0f / body.invMass;
    normalImpulse[contactIndex] =
        mass * physx::PxAbs(gravity.dot(dynamicNormal)) * dt /
        physx::PxReal(physx::PxMax(
            1u, contactsPerBody[bodyIndex]));
    const physx::PxReal physicalFrictionLimit =
        mu * normalImpulse[contactIndex];
    physx::PxReal projected0 = frictionImpulse0[contactIndex];
    physx::PxReal projected1 = frictionImpulse1[contactIndex];
    avbdProjectImpulseCone(physicalFrictionLimit,
                           projected0, projected1);
    if (existingMagnitude > physicalFrictionLimit + 1e-12f) {
      const physx::PxReal delta0 =
          projected0 - frictionImpulse0[contactIndex];
      const physx::PxReal delta1 =
          projected1 - frictionImpulse1[contactIndex];
      const physx::PxVec3 r =
          body.rotation.rotate(dynamicIsA ? contact.contactPointA
                                          : contact.contactPointB);
      const physx::PxVec3 angular0 = r.cross(contact.tangent0);
      const physx::PxVec3 angular1 = r.cross(contact.tangent1);
      body.linearVelocity +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          body.invMass;
      body.angularVelocity += body.invInertiaWorld.transform(
          angular0 * delta0 + angular1 * delta1);
      contact.frictionSweepImpulse +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          reportSign;
    }
    frictionImpulse0[contactIndex] = projected0;
    frictionImpulse1[contactIndex] = projected1;
  }

  physx::PxReal motorImpulse = 0.0f;
  const physx::PxReal maximumMotorImpulse =
      joint.motorMaxForce * dt;
  // The body-static friction sweep is a warm start with zero motor impulse.
  // Repeating every hard joint row, the bounded motor row, and every retained
  // two-axis friction cone makes the final state one coupled velocity
  // objective rather than a post-contact motor replay.
  for (physx::PxU32 iteration = 0; iteration < 96u; ++iteration) {
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numContacts; ++contactIndex) {
      AvbdContactConstraint &contact = contacts[contactIndex];
      const bool dynamicIsA =
          contact.header.bodyIndexA < numBodies;
      const physx::PxU32 bodyIndex =
          dynamicIsA ? contact.header.bodyIndexA
                     : contact.header.bodyIndexB;
      AvbdSolverBody &body = bodies[bodyIndex];
      const physx::PxVec3 r =
          body.rotation.rotate(dynamicIsA ? contact.contactPointA
                                          : contact.contactPointB);
      const physx::PxVec3 dynamicNormal =
          contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
      const physx::PxVec3 normalAngular = r.cross(dynamicNormal);
      const physx::PxReal normalResponse =
          body.invMass +
          normalAngular.dot(
              body.invInertiaWorld.transform(normalAngular));
      if (normalResponse > 1e-12f) {
        const physx::PxVec3 pointVelocity =
            body.linearVelocity + body.angularVelocity.cross(r);
        const physx::PxReal candidate =
            normalImpulse[contactIndex] -
            pointVelocity.dot(dynamicNormal) / normalResponse;
        const physx::PxReal maximumNormalImpulse =
            contact.maxImpulse < PX_MAX_REAL
                ? physx::PxMax(0.0f, contact.maxImpulse)
                : PX_MAX_REAL;
        const physx::PxReal projectedNormal =
            physx::PxClamp(candidate, 0.0f,
                           maximumNormalImpulse);
        const physx::PxReal deltaNormal =
            projectedNormal - normalImpulse[contactIndex];
        body.linearVelocity +=
            dynamicNormal * (body.invMass * deltaNormal);
        body.angularVelocity +=
            body.invInertiaWorld.transform(
                normalAngular * deltaNormal);
        normalImpulse[contactIndex] = projectedNormal;
      }

      const physx::PxVec3 angular0 = r.cross(contact.tangent0);
      const physx::PxVec3 angular1 = r.cross(contact.tangent1);
      const physx::PxReal k00 =
          body.invMass +
          angular0.dot(body.invInertiaWorld.transform(angular0));
      const physx::PxReal k01 =
          angular0.dot(body.invInertiaWorld.transform(angular1));
      const physx::PxReal k11 =
          body.invMass +
          angular1.dot(body.invInertiaWorld.transform(angular1));
      const physx::PxReal determinant = k00 * k11 - k01 * k01;
      if (determinant <= 1e-12f ||
          !physx::PxIsFinite(determinant))
        continue;
      const physx::PxVec3 pointVelocity =
          body.linearVelocity + body.angularVelocity.cross(r);
      const physx::PxReal rhs0 =
          -pointVelocity.dot(contact.tangent0);
      const physx::PxReal rhs1 =
          -pointVelocity.dot(contact.tangent1);
      physx::PxReal projected0 =
          frictionImpulse0[contactIndex] +
          (rhs0 * k11 - rhs1 * k01) / determinant;
      physx::PxReal projected1 =
          frictionImpulse1[contactIndex] +
          (k00 * rhs1 - k01 * rhs0) / determinant;
      const physx::PxReal frictionLimit =
          frictionMu[contactIndex] *
          normalImpulse[contactIndex];
      avbdProjectImpulseCone(frictionLimit,
                             projected0, projected1);
      const physx::PxReal delta0 =
          projected0 - frictionImpulse0[contactIndex];
      const physx::PxReal delta1 =
          projected1 - frictionImpulse1[contactIndex];
      body.linearVelocity +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          body.invMass;
      body.angularVelocity += body.invInertiaWorld.transform(
          angular0 * delta0 + angular1 * delta1);
      const physx::PxReal reportSign =
          dynamicIsA ? 1.0f : -1.0f;
      contact.frictionSweepImpulse +=
          (contact.tangent0 * delta0 +
           contact.tangent1 * delta1) *
          reportSign;
      frictionImpulse0[contactIndex] = projected0;
      frictionImpulse1[contactIndex] = projected1;
    }

    for (physx::PxU32 rowIndex = 0; rowIndex < 5u; ++rowIndex) {
      const physx::PxReal response = rowResponse(hardRows[rowIndex]);
      if (response <= 1e-12f)
        continue;
      const physx::PxReal impulse =
          (hardRows[rowIndex].targetVelocity -
           rowVelocity(hardRows[rowIndex])) /
          response;
      applyRowImpulse(hardRows[rowIndex], impulse);
    }

    const physx::PxReal motorResponse = rowResponse(motorRow);
    if (motorResponse > 1e-12f) {
      const physx::PxReal candidate =
          motorImpulse +
          (motorRow.targetVelocity - rowVelocity(motorRow)) /
              motorResponse;
      const physx::PxReal projected =
          physx::PxClamp(candidate, -maximumMotorImpulse,
                         maximumMotorImpulse);
      applyRowImpulse(motorRow, projected - motorImpulse);
      motorImpulse = projected;
    }
  }
  bodyA.projectLockedVelocities();
  bodyB.projectLockedVelocities();
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex)
    contacts[contactIndex].velocityNormalImpulse =
        normalImpulse[contactIndex];
}

static bool solveNativeMotorDense6(
    const physx::PxReal response[6][6],
    const physx::PxReal rhs[6],
    bool motorImpulseClamped,
    physx::PxReal clampedMotorImpulse,
    physx::PxReal solution[6]) {
  physx::PxReal augmented[6][7] = {};
  for (physx::PxU32 row = 0; row < 6; ++row) {
    for (physx::PxU32 column = 0; column < 6; ++column)
      augmented[row][column] = response[row][column];
    augmented[row][6] = rhs[row];
  }
  if (motorImpulseClamped) {
    for (physx::PxU32 row = 0; row < 6; ++row) {
      if (row == 3)
        continue;
      augmented[row][6] -=
          augmented[row][3] * clampedMotorImpulse;
      augmented[row][3] = 0.0f;
    }
    for (physx::PxU32 column = 0; column < 7; ++column)
      augmented[3][column] = 0.0f;
    augmented[3][3] = 1.0f;
    augmented[3][6] = clampedMotorImpulse;
  }

  for (physx::PxU32 column = 0; column < 6; ++column) {
    physx::PxU32 pivot = column;
    physx::PxReal pivotMagnitude =
        physx::PxAbs(augmented[column][column]);
    for (physx::PxU32 row = column + 1; row < 6; ++row) {
      const physx::PxReal candidate =
          physx::PxAbs(augmented[row][column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!physx::PxIsFinite(pivotMagnitude) ||
        pivotMagnitude <= 1e-10f)
      return false;
    if (pivot != column) {
      for (physx::PxU32 entry = column; entry < 7; ++entry) {
        const physx::PxReal temporary =
            augmented[column][entry];
        augmented[column][entry] = augmented[pivot][entry];
        augmented[pivot][entry] = temporary;
      }
    }
    const physx::PxReal inversePivot =
        1.0f / augmented[column][column];
    for (physx::PxU32 entry = column; entry < 7; ++entry)
      augmented[column][entry] *= inversePivot;
    for (physx::PxU32 row = 0; row < 6; ++row) {
      if (row == column)
        continue;
      const physx::PxReal factor = augmented[row][column];
      for (physx::PxU32 entry = column; entry < 7; ++entry)
        augmented[row][entry] -=
            factor * augmented[column][entry];
    }
  }
  for (physx::PxU32 row = 0; row < 6; ++row) {
    solution[row] = augmented[row][6];
    if (!physx::PxIsFinite(solution[row]))
      return false;
  }
  return true;
}

static void projectSingleNativeRevoluteMotorVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint, physx::PxReal dt,
    bool conserveDynamicPairAngularMomentum,
    physx::PxReal expectedAngularMomentumOnAxis,
    bool conserveDynamicPairAngularMomentumVector,
    const physx::PxVec3 &expectedAngularMomentumVector,
    bool conserveDynamicPairLinearMomentum,
    const physx::PxVec3 &expectedLinearMomentum,
    bool conserveDynamicPairSpatialMomentum,
    const physx::PxVec3 &expectedSpatialAngularMomentum,
    bool useSolveStartRelativeVelocity,
    physx::PxReal solveStartRelativeVelocity) {
  const physx::PxU32 endpoint[2] = {joint.header.bodyIndexA,
                                    joint.header.bodyIndexB};
  const bool dynamicA =
      endpoint[0] < numBodies && bodies[endpoint[0]].invMass > 0.0f;
  const bool dynamicB =
      endpoint[1] < numBodies && bodies[endpoint[1]].invMass > 0.0f;
  physx::PxVec3 worldAxis =
      dynamicA
          ? (bodies[endpoint[0]].rotation * joint.localFrameA)
                .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
          : joint.localFrameA.rotate(
                physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (worldAxis.normalize() <= 1e-6f)
    return;

  physx::PxVec3 responseA(0.0f);
  physx::PxVec3 responseB(0.0f);
  physx::PxReal unitResponse = 0.0f;
  physx::PxReal motorVelocity = 0.0f;
  const physx::PxReal driveRatio = joint.motorGearRatio;
  if (dynamicA) {
    responseA =
        bodies[endpoint[0]].invInertiaWorld.transform(worldAxis);
    unitResponse += worldAxis.dot(responseA);
    motorVelocity -=
        worldAxis.dot(bodies[endpoint[0]].angularVelocity);
  } else if (dt > 0.0f)
    motorVelocity -=
        worldAxis.dot(joint.externalAngularStepA) / dt;
  if (dynamicB) {
    responseB =
        bodies[endpoint[1]].invInertiaWorld.transform(worldAxis);
    unitResponse +=
        driveRatio * driveRatio * worldAxis.dot(responseB);
    motorVelocity +=
        driveRatio *
        worldAxis.dot(bodies[endpoint[1]].angularVelocity);
  } else if (dt > 0.0f)
    motorVelocity += driveRatio *
                     worldAxis.dot(joint.externalAngularStepB) / dt;
  if (!physx::PxIsFinite(unitResponse) || unitResponse <= 1e-10f)
    return;

  const bool freeSpin =
      (joint.sourceFlags & AvbdD6JointConstraint::
                               eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0;
  const physx::PxU32 dynamicSide = dynamicA ? 0u : 1u;
  const physx::PxVec3 dynamicAnchor =
      dynamicA ? joint.anchorA : joint.anchorB;
  const physx::PxQuat dynamicFrame =
      dynamicA ? joint.localFrameA : joint.localFrameB;
  physx::PxVec3 dynamicLocalAxis =
      dynamicFrame.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  const bool coupledOffCenterResponse =
      dynamicA != dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin && physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      dynamicLocalAxis.normalize() > 1e-6f &&
      dynamicAnchor.cross(dynamicLocalAxis).magnitudeSquared() > 1e-8f;
  if (coupledOffCenterResponse) {
    AvbdSolverBody &body = bodies[endpoint[dynamicSide]];
    if (!(body.invMass > 0.0f))
      return;

    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxMat33 inertia =
        body.invInertiaWorld.getInverse();
    const physx::PxVec3 worldLeverArm =
        body.rotation.rotate(dynamicAnchor);
    const physx::PxReal motorSign = dynamicA ? -1.0f : 1.0f;
    const auto computeMotorImpulse =
        [&](physx::PxReal angularSpeed) -> physx::PxReal {
      const physx::PxVec3 desiredAngular =
          worldAxis * angularSpeed;
      const physx::PxVec3 desiredLinear =
          -desiredAngular.cross(worldLeverArm);
      const physx::PxVec3 linearImpulse =
          (desiredLinear - body.linearVelocity) * mass;
      const physx::PxVec3 angularImpulse =
          inertia.transform(
              desiredAngular - body.angularVelocity);
      const physx::PxVec3 anchorTorque =
          angularImpulse -
          worldLeverArm.cross(linearImpulse);
      return motorSign * worldAxis.dot(anchorTorque);
    };

    const physx::PxReal zeroSpeedImpulse =
        computeMotorImpulse(0.0f);
    const physx::PxReal impulsePerAngularSpeed =
        computeMotorImpulse(1.0f) - zeroSpeedImpulse;
    if (!physx::PxIsFinite(zeroSpeedImpulse) ||
        !physx::PxIsFinite(impulsePerAngularSpeed) ||
        impulsePerAngularSpeed <= 1e-10f)
      return;
    const physx::PxReal targetAngularSpeed =
        motorSign * joint.motorTargetVelocity;
    const physx::PxReal requiredMotorImpulse =
        zeroSpeedImpulse +
        impulsePerAngularSpeed * targetAngularSpeed;
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal motorImpulse =
        physx::PxClamp(requiredMotorImpulse,
                       -maximumMotorImpulse,
                       maximumMotorImpulse);
    const physx::PxReal ownedAngularSpeed =
        (motorImpulse - zeroSpeedImpulse) /
        impulsePerAngularSpeed;
    const physx::PxVec3 candidateAngular =
        worldAxis * ownedAngularSpeed;
    const physx::PxVec3 candidateLinear =
        -candidateAngular.cross(worldLeverArm);
    if (!candidateAngular.isFinite() ||
        !candidateLinear.isFinite() ||
        (body.maxAngularVelocitySq > 0.0f &&
         candidateAngular.magnitudeSquared() >
             body.maxAngularVelocitySq) ||
        (body.maxLinearVelocitySq > 0.0f &&
         candidateLinear.magnitudeSquared() >
             body.maxLinearVelocitySq))
      return;
    body.angularVelocity = candidateAngular;
    body.linearVelocity = candidateLinear;
    return;
  }

  const bool coupledOffPrincipalResponse =
      dynamicA != dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin &&
      physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      ((dynamicA ? responseA : responseB)
           .cross(worldAxis)
           .magnitude() >
       physx::PxMax(
           (dynamicA ? responseA : responseB).magnitude(), 1e-8f) *
           1e-4f);
  if (coupledOffPrincipalResponse) {
    AvbdSolverBody &body =
        bodies[dynamicA ? endpoint[0] : endpoint[1]];
    const physx::PxReal motorSign = dynamicA ? -1.0f : 1.0f;
    const physx::PxVec3 motorJ = worldAxis * motorSign;
    physx::PxVec3 referenceAxis =
        physx::PxAbs(worldAxis.x) < 0.8f
            ? physx::PxVec3(1.0f, 0.0f, 0.0f)
            : physx::PxVec3(0.0f, 1.0f, 0.0f);
    physx::PxVec3 swingJ0 = worldAxis.cross(referenceAxis);
    if (swingJ0.normalize() <= 1e-6f)
      return;
    physx::PxVec3 swingJ1 = worldAxis.cross(swingJ0);
    if (swingJ1.normalize() <= 1e-6f)
      return;

    const physx::PxVec3 response[3] = {
        body.invInertiaWorld.transform(motorJ),
        body.invInertiaWorld.transform(swingJ0),
        body.invInertiaWorld.transform(swingJ1)};
    const physx::PxVec3 jacobian[3] = {
        motorJ, swingJ0, swingJ1};
    const physx::PxMat33 responseMatrix(
        physx::PxVec3(jacobian[0].dot(response[0]),
                      jacobian[1].dot(response[0]),
                      jacobian[2].dot(response[0])),
        physx::PxVec3(jacobian[0].dot(response[1]),
                      jacobian[1].dot(response[1]),
                      jacobian[2].dot(response[1])),
        physx::PxVec3(jacobian[0].dot(response[2]),
                      jacobian[1].dot(response[2]),
                      jacobian[2].dot(response[2])));
    const physx::PxReal determinant =
        responseMatrix.getDeterminant();
    if (!physx::PxIsFinite(determinant) ||
        physx::PxAbs(determinant) <= 1e-12f)
      return;

    const physx::PxVec3 rhs(
        joint.motorTargetVelocity -
            motorJ.dot(body.angularVelocity),
        -swingJ0.dot(body.angularVelocity),
        -swingJ1.dot(body.angularVelocity));
    physx::PxVec3 impulse =
        responseMatrix.getInverse().transform(rhs);
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal clampedMotorImpulse =
        physx::PxClamp(impulse.x, -maximumMotorImpulse,
                       maximumMotorImpulse);
    if (clampedMotorImpulse != impulse.x) {
      const physx::PxReal k11 =
          jacobian[1].dot(response[1]);
      const physx::PxReal k12 =
          jacobian[1].dot(response[2]);
      const physx::PxReal k22 =
          jacobian[2].dot(response[2]);
      const physx::PxReal swingDeterminant =
          k11 * k22 - k12 * k12;
      if (!physx::PxIsFinite(swingDeterminant) ||
          physx::PxAbs(swingDeterminant) <= 1e-12f)
        return;
      const physx::PxReal swingRhs0 =
          rhs.y - jacobian[1].dot(response[0]) *
                      clampedMotorImpulse;
      const physx::PxReal swingRhs1 =
          rhs.z - jacobian[2].dot(response[0]) *
                      clampedMotorImpulse;
      impulse.y =
          (swingRhs0 * k22 - swingRhs1 * k12) /
          swingDeterminant;
      impulse.z =
          (k11 * swingRhs1 - k12 * swingRhs0) /
          swingDeterminant;
      impulse.x = clampedMotorImpulse;
    }
    if (!impulse.isFinite())
      return;

    const physx::PxVec3 candidate =
        body.angularVelocity + response[0] * impulse.x +
        response[1] * impulse.y + response[2] * impulse.z;
    if (!candidate.isFinite() ||
        (body.maxAngularVelocitySq > 0.0f &&
         candidate.magnitudeSquared() >
             body.maxAngularVelocitySq))
      return;
    body.angularVelocity = candidate;
    return;
  }

  const bool coupledDynamicPairOffPrincipalResponse =
      dynamicA && dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin && physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      joint.anchorA.magnitudeSquared() <= 1e-8f &&
      joint.anchorB.magnitudeSquared() <= 1e-8f &&
      (responseA.cross(worldAxis).magnitude() >
           physx::PxMax(responseA.magnitude(), 1e-8f) * 1e-4f ||
       responseB.cross(worldAxis).magnitude() >
           physx::PxMax(responseB.magnitude(), 1e-8f) * 1e-4f);
  if (coupledDynamicPairOffPrincipalResponse) {
    AvbdSolverBody &bodyA = bodies[endpoint[0]];
    AvbdSolverBody &bodyB = bodies[endpoint[1]];
    physx::PxVec3 referenceAxis =
        physx::PxAbs(worldAxis.x) < 0.8f
            ? physx::PxVec3(1.0f, 0.0f, 0.0f)
            : physx::PxVec3(0.0f, 1.0f, 0.0f);
    physx::PxVec3 swingJ0 = worldAxis.cross(referenceAxis);
    if (swingJ0.normalize() <= 1e-6f)
      return;
    physx::PxVec3 swingJ1 = worldAxis.cross(swingJ0);
    if (swingJ1.normalize() <= 1e-6f)
      return;

    const physx::PxVec3 jacobian[3] = {
        worldAxis, swingJ0, swingJ1};
    physx::PxVec3 responseA3[3];
    physx::PxVec3 responseB3[3];
    physx::PxMat33 responseMatrix(physx::PxZero);
    for (physx::PxU32 column = 0; column < 3; ++column) {
      responseA3[column] =
          bodyA.invInertiaWorld.transform(jacobian[column]);
      responseB3[column] =
          bodyB.invInertiaWorld.transform(jacobian[column]);
      responseMatrix[column] = physx::PxVec3(
          jacobian[0].dot(responseA3[column] +
                          responseB3[column]),
          jacobian[1].dot(responseA3[column] +
                          responseB3[column]),
          jacobian[2].dot(responseA3[column] +
                          responseB3[column]));
    }
    const physx::PxReal determinant =
        responseMatrix.getDeterminant();
    if (!physx::PxIsFinite(determinant) ||
        physx::PxAbs(determinant) <= 1e-12f)
      return;

    const physx::PxVec3 relativeAngular =
        bodyB.angularVelocity - bodyA.angularVelocity;
    const physx::PxVec3 rhs(
        joint.motorTargetVelocity -
            jacobian[0].dot(relativeAngular),
        -jacobian[1].dot(relativeAngular),
        -jacobian[2].dot(relativeAngular));
    physx::PxVec3 impulse =
        responseMatrix.getInverse().transform(rhs);
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal clampedMotorImpulse =
        physx::PxClamp(impulse.x, -maximumMotorImpulse,
                       maximumMotorImpulse);
    if (clampedMotorImpulse != impulse.x) {
      const physx::PxReal k11 =
          jacobian[1].dot(responseA3[1] + responseB3[1]);
      const physx::PxReal k12 =
          jacobian[1].dot(responseA3[2] + responseB3[2]);
      const physx::PxReal k22 =
          jacobian[2].dot(responseA3[2] + responseB3[2]);
      const physx::PxReal swingDeterminant =
          k11 * k22 - k12 * k12;
      if (!physx::PxIsFinite(swingDeterminant) ||
          physx::PxAbs(swingDeterminant) <= 1e-12f)
        return;
      const physx::PxReal swingRhs0 =
          rhs.y - jacobian[1].dot(responseA3[0] +
                                  responseB3[0]) *
                      clampedMotorImpulse;
      const physx::PxReal swingRhs1 =
          rhs.z - jacobian[2].dot(responseA3[0] +
                                  responseB3[0]) *
                      clampedMotorImpulse;
      impulse.y =
          (swingRhs0 * k22 - swingRhs1 * k12) /
          swingDeterminant;
      impulse.z =
          (k11 * swingRhs1 - k12 * swingRhs0) /
          swingDeterminant;
      impulse.x = clampedMotorImpulse;
    }
    if (!impulse.isFinite())
      return;

    physx::PxVec3 candidateA = bodyA.angularVelocity;
    physx::PxVec3 candidateB = bodyB.angularVelocity;
    for (physx::PxU32 row = 0; row < 3; ++row) {
      candidateA -= responseA3[row] * impulse[row];
      candidateB += responseB3[row] * impulse[row];
    }
    if (conserveDynamicPairAngularMomentumVector) {
      const physx::PxMat33 inertiaA =
          bodyA.invInertiaWorld.getInverse();
      const physx::PxMat33 inertiaB =
          bodyB.invInertiaWorld.getInverse();
      const physx::PxMat33 inertiaSum(
          inertiaA.column0 + inertiaB.column0,
          inertiaA.column1 + inertiaB.column1,
          inertiaA.column2 + inertiaB.column2);
      const physx::PxReal inertiaSumDeterminant =
          inertiaSum.getDeterminant();
      const physx::PxVec3 currentAngularMomentum =
          inertiaA.transform(candidateA) +
          inertiaB.transform(candidateB);
      if (!physx::PxIsFinite(inertiaSumDeterminant) ||
          physx::PxAbs(inertiaSumDeterminant) <= 1e-12f ||
          !currentAngularMomentum.isFinite())
        return;
      const physx::PxVec3 commonAngularVelocity =
          inertiaSum.getInverse().transform(
              expectedAngularMomentumVector -
              currentAngularMomentum);
      if (!commonAngularVelocity.isFinite())
        return;
      candidateA += commonAngularVelocity;
      candidateB += commonAngularVelocity;
    }
    if (!candidateA.isFinite() || !candidateB.isFinite() ||
        (bodyA.maxAngularVelocitySq > 0.0f &&
         candidateA.magnitudeSquared() >
             bodyA.maxAngularVelocitySq) ||
        (bodyB.maxAngularVelocitySq > 0.0f &&
         candidateB.magnitudeSquared() >
             bodyB.maxAngularVelocitySq))
      return;
    bodyA.angularVelocity = candidateA;
    bodyB.angularVelocity = candidateB;
    return;
  }

  const physx::PxVec3 localAxisA =
      joint.localFrameA.rotate(
          physx::PxVec3(1.0f, 0.0f, 0.0f));
  const physx::PxVec3 localAxisB =
      joint.localFrameB.rotate(
          physx::PxVec3(1.0f, 0.0f, 0.0f));
  const bool coupledDynamicPairOffCenterResponse =
      dynamicA && dynamicB && joint.getAngularMotion(0) == 2u &&
      !freeSpin && physx::PxAbs(driveRatio - 1.0f) <= 1e-6f &&
      (joint.anchorA.cross(localAxisA).magnitudeSquared() > 1e-8f ||
       joint.anchorB.cross(localAxisB).magnitudeSquared() > 1e-8f);
  if (coupledDynamicPairOffCenterResponse) {
    AvbdSolverBody &bodyA = bodies[endpoint[0]];
    AvbdSolverBody &bodyB = bodies[endpoint[1]];
    const physx::PxVec3 rA =
        bodyA.rotation.rotate(joint.anchorA);
    const physx::PxVec3 rB =
        bodyB.rotation.rotate(joint.anchorB);
    physx::PxVec3 referenceAxis =
        physx::PxAbs(worldAxis.x) < 0.8f
            ? physx::PxVec3(1.0f, 0.0f, 0.0f)
            : physx::PxVec3(0.0f, 1.0f, 0.0f);
    physx::PxVec3 swingAxis0 =
        worldAxis.cross(referenceAxis);
    if (swingAxis0.normalize() <= 1e-6f)
      return;
    physx::PxVec3 swingAxis1 =
        worldAxis.cross(swingAxis0);
    if (swingAxis1.normalize() <= 1e-6f)
      return;
    const physx::PxVec3 worldAxes[3] = {
        physx::PxVec3(1.0f, 0.0f, 0.0f),
        physx::PxVec3(0.0f, 1.0f, 0.0f),
        physx::PxVec3(0.0f, 0.0f, 1.0f)};
    const physx::PxVec3 angularAxes[3] = {
        worldAxis, swingAxis0, swingAxis1};
    AvbdVec6 jacobianA[6];
    AvbdVec6 jacobianB[6];
    for (physx::PxU32 row = 0; row < 3; ++row) {
      jacobianA[row] =
          AvbdVec6(worldAxes[row], rA.cross(worldAxes[row]));
      jacobianB[row] =
          AvbdVec6(-worldAxes[row], -rB.cross(worldAxes[row]));
      jacobianA[3 + row] =
          AvbdVec6(physx::PxVec3(0.0f), angularAxes[row]);
      jacobianB[3 + row] =
          AvbdVec6(physx::PxVec3(0.0f), -angularAxes[row]);
    }

    const AvbdVec6 velocityA(
        bodyA.linearVelocity, bodyA.angularVelocity);
    const AvbdVec6 velocityB(
        bodyB.linearVelocity, bodyB.angularVelocity);
    physx::PxReal rhs[6] = {};
    physx::PxReal responseMatrix[6][6] = {};
    for (physx::PxU32 row = 0; row < 6; ++row) {
      const physx::PxReal current =
          jacobianA[row].dot(velocityA) +
          jacobianB[row].dot(velocityB);
      const physx::PxReal target =
          row == 3 ? -joint.motorTargetVelocity : 0.0f;
      rhs[row] = target - current;
      for (physx::PxU32 column = 0; column < 6; ++column) {
        const AvbdVec6 responseA6(
            jacobianA[column].linear * bodyA.invMass,
            bodyA.invInertiaWorld *
                jacobianA[column].angular);
        const AvbdVec6 responseB6(
            jacobianB[column].linear * bodyB.invMass,
            bodyB.invInertiaWorld *
                jacobianB[column].angular);
        responseMatrix[row][column] =
            jacobianA[row].dot(responseA6) +
            jacobianB[row].dot(responseB6);
      }
    }
    physx::PxReal impulse[6] = {};
    if (!solveNativeMotorDense6(
            responseMatrix, rhs, false, 0.0f, impulse))
      return;
    const physx::PxReal maximumMotorImpulse =
        joint.motorMaxForce * dt;
    const physx::PxReal clampedMotorImpulse =
        physx::PxClamp(impulse[3], -maximumMotorImpulse,
                       maximumMotorImpulse);
    if (clampedMotorImpulse != impulse[3] &&
        !solveNativeMotorDense6(
            responseMatrix, rhs, true,
            clampedMotorImpulse, impulse))
      return;

    AvbdVec6 bodyImpulseA;
    AvbdVec6 bodyImpulseB;
    for (physx::PxU32 row = 0; row < 6; ++row) {
      bodyImpulseA.linear +=
          jacobianA[row].linear * impulse[row];
      bodyImpulseA.angular +=
          jacobianA[row].angular * impulse[row];
      bodyImpulseB.linear +=
          jacobianB[row].linear * impulse[row];
      bodyImpulseB.angular +=
          jacobianB[row].angular * impulse[row];
    }
    physx::PxVec3 candidateLinearA =
        bodyA.linearVelocity +
        bodyImpulseA.linear * bodyA.invMass;
    physx::PxVec3 candidateAngularA =
        bodyA.angularVelocity +
        bodyA.invInertiaWorld * bodyImpulseA.angular;
    physx::PxVec3 candidateLinearB =
        bodyB.linearVelocity +
        bodyImpulseB.linear * bodyB.invMass;
    physx::PxVec3 candidateAngularB =
        bodyB.angularVelocity +
        bodyB.invInertiaWorld * bodyImpulseB.angular;
    if (conserveDynamicPairSpatialMomentum &&
        conserveDynamicPairLinearMomentum) {
      const physx::PxReal massA = 1.0f / bodyA.invMass;
      const physx::PxReal massB = 1.0f / bodyB.invMass;
      const physx::PxMat33 inertiaA =
          bodyA.invInertiaWorld.getInverse();
      const physx::PxMat33 inertiaB =
          bodyB.invInertiaWorld.getInverse();
      const physx::PxVec3 currentLinearMomentum =
          candidateLinearA * massA + candidateLinearB * massB;
      const physx::PxVec3 currentAngularMomentum =
          bodyA.position.cross(candidateLinearA * massA) +
          inertiaA.transform(candidateAngularA) +
          bodyB.position.cross(candidateLinearB * massB) +
          inertiaB.transform(candidateAngularB);
      if (!currentLinearMomentum.isFinite() ||
          !currentAngularMomentum.isFinite())
        return;
      physx::PxReal spatialResponse[6][6] = {};
      const physx::PxVec3 basis[3] = {
          physx::PxVec3(1.0f, 0.0f, 0.0f),
          physx::PxVec3(0.0f, 1.0f, 0.0f),
          physx::PxVec3(0.0f, 0.0f, 1.0f)};
      for (physx::PxU32 column = 0; column < 6; ++column) {
        const physx::PxVec3 commonLinear =
            column < 3 ? basis[column] : physx::PxVec3(0.0f);
        const physx::PxVec3 commonAngular =
            column < 3 ? physx::PxVec3(0.0f)
                       : basis[column - 3];
        const physx::PxVec3 deltaLinearA =
            commonLinear +
            commonAngular.cross(bodyA.position);
        const physx::PxVec3 deltaLinearB =
            commonLinear +
            commonAngular.cross(bodyB.position);
        const physx::PxVec3 deltaLinearMomentum =
            deltaLinearA * massA + deltaLinearB * massB;
        const physx::PxVec3 deltaAngularMomentum =
            bodyA.position.cross(deltaLinearA * massA) +
            inertiaA.transform(commonAngular) +
            bodyB.position.cross(deltaLinearB * massB) +
            inertiaB.transform(commonAngular);
        for (physx::PxU32 row = 0; row < 3; ++row) {
          spatialResponse[row][column] =
              deltaLinearMomentum[row];
          spatialResponse[3 + row][column] =
              deltaAngularMomentum[row];
        }
      }
      physx::PxReal spatialRhs[6] = {};
      const physx::PxVec3 linearMomentumDelta =
          expectedLinearMomentum - currentLinearMomentum;
      const physx::PxVec3 angularMomentumDelta =
          expectedSpatialAngularMomentum -
          currentAngularMomentum;
      for (physx::PxU32 row = 0; row < 3; ++row) {
        spatialRhs[row] = linearMomentumDelta[row];
        spatialRhs[3 + row] = angularMomentumDelta[row];
      }
      physx::PxReal spatialCorrection[6] = {};
      if (!solveNativeMotorDense6(
              spatialResponse, spatialRhs, false, 0.0f,
              spatialCorrection))
        return;
      const physx::PxVec3 commonLinearVelocity(
          spatialCorrection[0], spatialCorrection[1],
          spatialCorrection[2]);
      const physx::PxVec3 commonAngularVelocity(
          spatialCorrection[3], spatialCorrection[4],
          spatialCorrection[5]);
      candidateLinearA +=
          commonLinearVelocity +
          commonAngularVelocity.cross(bodyA.position);
      candidateLinearB +=
          commonLinearVelocity +
          commonAngularVelocity.cross(bodyB.position);
      candidateAngularA += commonAngularVelocity;
      candidateAngularB += commonAngularVelocity;
    } else if (conserveDynamicPairLinearMomentum) {
      const physx::PxReal massA = 1.0f / bodyA.invMass;
      const physx::PxReal massB = 1.0f / bodyB.invMass;
      const physx::PxReal totalMass = massA + massB;
      const physx::PxVec3 currentLinearMomentum =
          candidateLinearA * massA + candidateLinearB * massB;
      if (!physx::PxIsFinite(totalMass) ||
          totalMass <= 1e-10f ||
          !currentLinearMomentum.isFinite())
        return;
      const physx::PxVec3 correction =
          (expectedLinearMomentum - currentLinearMomentum) /
          totalMass;
      if (!correction.isFinite())
        return;
      candidateLinearA += correction;
      candidateLinearB += correction;
    }
    if (!candidateLinearA.isFinite() ||
        !candidateAngularA.isFinite() ||
        !candidateLinearB.isFinite() ||
        !candidateAngularB.isFinite() ||
        (bodyA.maxLinearVelocitySq > 0.0f &&
         candidateLinearA.magnitudeSquared() >
             bodyA.maxLinearVelocitySq) ||
        (bodyA.maxAngularVelocitySq > 0.0f &&
         candidateAngularA.magnitudeSquared() >
             bodyA.maxAngularVelocitySq) ||
        (bodyB.maxLinearVelocitySq > 0.0f &&
         candidateLinearB.magnitudeSquared() >
             bodyB.maxLinearVelocitySq) ||
        (bodyB.maxAngularVelocitySq > 0.0f &&
         candidateAngularB.magnitudeSquared() >
             bodyB.maxAngularVelocitySq))
      return;
    bodyA.linearVelocity = candidateLinearA;
    bodyA.angularVelocity = candidateAngularA;
    bodyB.linearVelocity = candidateLinearB;
    bodyB.angularVelocity = candidateAngularB;
    return;
  }

  const physx::PxReal motorBaseVelocity =
      freeSpin && useSolveStartRelativeVelocity
          ? solveStartRelativeVelocity
          : motorVelocity;
  const physx::PxReal requiredMotorImpulse =
      (joint.motorTargetVelocity - motorBaseVelocity) / unitResponse;
  const physx::PxReal maximumImpulse = joint.motorMaxForce * dt;
  physx::PxReal minimumMotorImpulse = -maximumImpulse;
  physx::PxReal maximumMotorImpulse = maximumImpulse;
  // Match PxRevoluteJoint::eDRIVE_FREESPIN row bounds exactly. Positive
  // targets may only receive positive drive impulse, negative targets only
  // negative drive impulse, and a zero target remains bilateral.
  if (freeSpin && joint.motorTargetVelocity > 0.0f)
    minimumMotorImpulse = 0.0f;
  else if (freeSpin && joint.motorTargetVelocity < 0.0f)
    maximumMotorImpulse = 0.0f;
  const physx::PxReal motorImpulse = physx::PxClamp(
      requiredMotorImpulse, minimumMotorImpulse, maximumMotorImpulse);
  const physx::PxReal motorOwnedVelocity =
      motorBaseVelocity + unitResponse * motorImpulse;
  // The position path reconstructs velocity from its corrected pose. For the
  // strict free-spin owner, restore the solve-entry hinge derivative before
  // applying the authored one-sided motor impulse; otherwise a pose-derived
  // zero velocity incorrectly brakes a super-target body.
  physx::PxReal impulse =
      (motorOwnedVelocity - motorVelocity) / unitResponse;

  if (joint.getAngularMotion(0) == 1u) {
    const physx::PxQuat rotationA =
        dynamicA ? bodies[endpoint[0]].rotation
                 : physx::PxQuat(physx::PxIdentity);
    const physx::PxQuat rotationB =
        dynamicB ? bodies[endpoint[1]].rotation
                 : physx::PxQuat(physx::PxIdentity);
    const physx::PxReal angularError =
        joint.computeAngularError(rotationA, rotationB, 0);
    const physx::PxReal limitSpan =
        joint.angularLimitUpper.x - joint.angularLimitLower.x;
    const physx::PxReal activeTolerance =
        physx::PxMax(1e-5f, physx::PxAbs(limitSpan) * 1e-5f);
    const physx::PxReal motorOnlyVelocity =
        motorVelocity + unitResponse * impulse;
    // computeAngularError uses frameA * conjugate(frameB), so its derivative
    // is axis dot (wA - wB), the negative of the public motor velocity row.
    // At the internal lower bound the admissible public velocity is <= 0;
    // at the internal upper bound it is >= 0. First apply the bounded motor,
    // then add only the unilateral limit impulse needed to close an outward
    // derivative. A one-step speculative test activates the row before the
    // bounded motor can cross it. Position remains exclusively owned by the
    // hard AL row.
    const physx::PxReal predictedAngularError =
        angularError - motorOnlyVelocity * dt;
    const bool atLower =
        angularError <= joint.angularLimitLower.x + activeTolerance ||
        predictedAngularError <= joint.angularLimitLower.x;
    const bool atUpper =
        angularError >= joint.angularLimitUpper.x - activeTolerance ||
        predictedAngularError >= joint.angularLimitUpper.x;
    if ((atLower && motorOnlyVelocity > 0.0f) ||
        (atUpper && motorOnlyVelocity < 0.0f))
      impulse = -motorVelocity / unitResponse;
  }
  if (!physx::PxIsFinite(impulse))
    return;

  physx::PxVec3 candidateA =
      dynamicA
          ? bodies[endpoint[0]].angularVelocity - responseA * impulse
          : physx::PxVec3(0.0f);
  physx::PxVec3 candidateB =
      dynamicB
          ? bodies[endpoint[1]].angularVelocity +
                responseB * (driveRatio * impulse)
          : physx::PxVec3(0.0f);

  if (conserveDynamicPairAngularMomentum && dynamicA && dynamicB) {
    const physx::PxMat33 inertiaA =
        bodies[endpoint[0]].invInertiaWorld.getInverse();
    const physx::PxMat33 inertiaB =
        bodies[endpoint[1]].invInertiaWorld.getInverse();
    const physx::PxReal inertiaOnAxisA =
        worldAxis.dot(inertiaA.transform(worldAxis));
    const physx::PxReal inertiaOnAxisB =
        worldAxis.dot(inertiaB.transform(worldAxis));
    const physx::PxReal inertiaSum =
        driveRatio * driveRatio * inertiaOnAxisA +
        inertiaOnAxisB;
    const physx::PxReal currentAngularMomentumOnAxis =
        worldAxis.dot(
            inertiaA.transform(candidateA) * driveRatio +
            inertiaB.transform(candidateB));
    if (!physx::PxIsFinite(inertiaSum) ||
        !physx::PxIsFinite(currentAngularMomentumOnAxis) ||
        inertiaSum <= 1e-10f)
      return;
    // The hard joint position path may leave a small common angular mode.
    // Restore the solve-start conserved momentum without changing the
    // already-projected relative motor velocity.
    const physx::PxReal commonAngularVelocity =
        (expectedAngularMomentumOnAxis -
         currentAngularMomentumOnAxis) /
        inertiaSum;
    if (!physx::PxIsFinite(commonAngularVelocity))
      return;
    candidateA +=
        worldAxis * (driveRatio * commonAngularVelocity);
    candidateB += worldAxis * commonAngularVelocity;
  }
  if ((dynamicA &&
       (!candidateA.isFinite() ||
        (bodies[endpoint[0]].maxAngularVelocitySq > 0.0f &&
         candidateA.magnitudeSquared() >
             bodies[endpoint[0]].maxAngularVelocitySq))) ||
      (dynamicB &&
       (!candidateB.isFinite() ||
        (bodies[endpoint[1]].maxAngularVelocitySq > 0.0f &&
         candidateB.magnitudeSquared() >
             bodies[endpoint[1]].maxAngularVelocitySq))))
    return;

  if (dynamicA)
    bodies[endpoint[0]].angularVelocity = candidateA;
  if (dynamicB)
    bodies[endpoint[1]].angularVelocity = candidateB;
}

static bool isPassiveCenteredGearVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, const AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 2 || !gearJoints || numGear != 1 ||
      numSoftParticles != 0 || numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdGearJointConstraint &gear = gearJoints[0];
  const physx::PxU32 gearA = gear.header.bodyIndexA;
  const physx::PxU32 gearB = gear.header.bodyIndexB;
  if (gearA >= numBodies || gearB >= numBodies || gearA == gearB ||
      bodies[gearA].invMass <= 0.0f || bodies[gearB].invMass <= 0.0f ||
      !physx::PxIsFinite(gear.gearRatio) ||
      physx::PxAbs(gear.gearRatio) <= 1e-6f ||
      !gear.gearAxis0.isFinite() || !gear.gearAxis1.isFinite() ||
      gear.gearAxis0.magnitudeSquared() <= 1e-8f ||
      gear.gearAxis1.magnitudeSquared() <= 1e-8f)
    return false;

  bool ownsBody[2] = {false, false};
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdD6JointConstraint &joint = d6Joints[i];
    const bool aDynamic =
        joint.header.bodyIndexA < numBodies &&
        bodies[joint.header.bodyIndexA].invMass > 0.0f;
    const bool bDynamic =
        joint.header.bodyIndexB < numBodies &&
        bodies[joint.header.bodyIndexB].invMass > 0.0f;
    if (joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
        aDynamic == bDynamic || joint.linearMotion != 0u ||
        joint.angularMotion != 0x2u || joint.driveFlags != 0u ||
        joint.motorEnabled != 0u)
      return false;

    const physx::PxU32 dynamicIndex =
        aDynamic ? joint.header.bodyIndexA : joint.header.bodyIndexB;
    if (dynamicIndex >= 2 || ownsBody[dynamicIndex])
      return false;
    ownsBody[dynamicIndex] = true;

    const physx::PxVec3 &dynamicAnchor =
        aDynamic ? joint.anchorA : joint.anchorB;
    const physx::PxQuat &dynamicFrame =
        aDynamic ? joint.localFrameA : joint.localFrameB;
    if (dynamicAnchor.magnitudeSquared() > 1e-8f)
      return false;

    physx::PxVec3 hingeAxis =
        dynamicFrame.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    physx::PxVec3 gearAxis =
        dynamicIndex == gearA ? gear.gearAxis0 : gear.gearAxis1;
    if (hingeAxis.normalize() <= 1e-6f ||
        gearAxis.normalize() <= 1e-6f ||
        physx::PxAbs(hingeAxis.dot(gearAxis)) < 0.9999f)
      return false;
  }

  return ownsBody[gearA] && ownsBody[gearB];
}

static bool isNativeRevoluteMotorGearVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, const AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts,
    physx::PxU32 &motorJointIndex) {
  motorJointIndex = PX_MAX_U32;
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 2 || !gearJoints || numGear != 1 ||
      numSoftParticles != 0 || numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdGearJointConstraint &gear = gearJoints[0];
  const physx::PxU32 gearA = gear.header.bodyIndexA;
  const physx::PxU32 gearB = gear.header.bodyIndexB;
  if (gearA >= numBodies || gearB >= numBodies || gearA == gearB ||
      bodies[gearA].invMass <= 0.0f || bodies[gearB].invMass <= 0.0f ||
      !physx::PxIsFinite(gear.gearRatio) ||
      physx::PxAbs(gear.gearRatio) <= 1e-6f ||
      !gear.gearAxis0.isFinite() || !gear.gearAxis1.isFinite() ||
      gear.gearAxis0.magnitudeSquared() <= 1e-8f ||
      gear.gearAxis1.magnitudeSquared() <= 1e-8f)
    return false;

  bool ownsBody[2] = {false, false};
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdD6JointConstraint &joint = d6Joints[i];
    const bool dynamicA =
        joint.header.bodyIndexA < numBodies &&
        bodies[joint.header.bodyIndexA].invMass > 0.0f;
    const bool dynamicB =
        joint.header.bodyIndexB < numBodies &&
        bodies[joint.header.bodyIndexB].invMass > 0.0f;
    if (joint.header.type != AvbdConstraintType::eJOINT_REVOLUTE ||
        dynamicA == dynamicB || joint.linearMotion != 0u ||
        joint.angularMotion != 0x2u || joint.driveFlags != 0u ||
        joint.driveAccelerationFlags != 0u ||
        joint.coneAngleLimit > 0.0f)
      return false;

    const physx::PxU32 dynamicIndex =
        dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
    if (dynamicIndex >= 2 || ownsBody[dynamicIndex] ||
        bodies[dynamicIndex].lockFlags != 0)
      return false;
    ownsBody[dynamicIndex] = true;

    const physx::PxVec3 &dynamicAnchor =
        dynamicA ? joint.anchorA : joint.anchorB;
    const physx::PxQuat &dynamicFrame =
        dynamicA ? joint.localFrameA : joint.localFrameB;
    if (dynamicAnchor.magnitudeSquared() > 1e-8f)
      return false;

    physx::PxVec3 hingeAxis =
        dynamicFrame.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    physx::PxVec3 gearAxis =
        dynamicIndex == gearA ? gear.gearAxis0 : gear.gearAxis1;
    if (hingeAxis.normalize() <= 1e-6f ||
        gearAxis.normalize() <= 1e-6f ||
        physx::PxAbs(hingeAxis.dot(gearAxis)) < 0.9999f)
      return false;

    physx::PxVec3 worldAxis =
        bodies[dynamicIndex].rotation.rotate(gearAxis);
    if (worldAxis.normalize() <= 1e-6f)
      return false;
    const physx::PxVec3 response =
        bodies[dynamicIndex].invInertiaWorld.transform(worldAxis);
    const physx::PxReal responseScale =
        physx::PxMax(response.magnitude(), 1e-8f);
    if (response.cross(worldAxis).magnitude() >
        responseScale * 1e-4f)
      return false;

    if (joint.motorEnabled != 0u) {
      if (motorJointIndex != PX_MAX_U32 ||
          !(joint.motorMaxForce > 0.0f) ||
          !physx::PxIsFinite(joint.motorMaxForce) ||
          !physx::PxIsFinite(joint.motorTargetVelocity) ||
          !physx::PxIsFinite(joint.motorGearRatio) ||
          physx::PxAbs(joint.motorGearRatio - 1.0f) > 1e-6f ||
          (joint.sourceFlags & AvbdD6JointConstraint::
                                   eNATIVE_REVOLUTE_MOTOR_FREESPIN) != 0)
        return false;
      motorJointIndex = i;
    }
  }

  return ownsBody[gearA] && ownsBody[gearB] &&
         motorJointIndex != PX_MAX_U32;
}

static void projectPassiveCenteredGearVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdGearJointConstraint &gear,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart) {
  const physx::PxU32 bodyAIndex = gear.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = gear.header.bodyIndexB;
  if (bodyAIndex >= numBodies || bodyBIndex >= numBodies ||
      angularVelAtSolveStart.size() != numBodies)
    return;

  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxVec3 axisA = bodyA.rotation.rotate(gear.gearAxis0);
  physx::PxVec3 axisB = bodyB.rotation.rotate(gear.gearAxis1);
  if (axisA.normalize() <= 1e-6f || axisB.normalize() <= 1e-6f)
    return;

  const physx::PxVec3 jacobianA = axisA * gear.gearRatio;
  const physx::PxVec3 jacobianB = axisB;
  const physx::PxVec3 responseA =
      bodyA.invInertiaWorld.transform(jacobianA);
  const physx::PxVec3 responseB =
      bodyB.invInertiaWorld.transform(jacobianB);
  const physx::PxReal denominator =
      jacobianA.dot(responseA) + jacobianB.dot(responseB);
  if (!physx::PxIsFinite(denominator) || denominator <= 1e-10f)
    return;

  // This scoped path requires each revolute free axis to be a principal
  // inertia direction.  Otherwise an exact gear impulse also changes locked
  // swing components and must be solved together with the complete D6 row set.
  const physx::PxReal responseScaleA =
      physx::PxMax(responseA.magnitude(), 1e-8f);
  const physx::PxReal responseScaleB =
      physx::PxMax(responseB.magnitude(), 1e-8f);
  if (responseA.cross(axisA).magnitude() > responseScaleA * 1e-4f ||
      responseB.cross(axisB).magnitude() > responseScaleB * 1e-4f)
    return;

  const physx::PxVec3 &rawA = angularVelAtSolveStart[bodyAIndex];
  const physx::PxVec3 &rawB = angularVelAtSolveStart[bodyBIndex];
  const physx::PxReal residual =
      jacobianA.dot(rawA) + jacobianB.dot(rawB);
  const physx::PxReal lambda = -residual / denominator;
  if (!physx::PxIsFinite(lambda))
    return;

  const physx::PxVec3 expectedA = rawA + responseA * lambda;
  const physx::PxVec3 expectedB = rawB + responseB * lambda;
  physx::PxVec3 candidateA =
      bodyA.angularVelocity +
      axisA * (expectedA.dot(axisA) - bodyA.angularVelocity.dot(axisA));
  physx::PxVec3 candidateB =
      bodyB.angularVelocity +
      axisB * (expectedB.dot(axisB) - bodyB.angularVelocity.dot(axisB));
  if (!candidateA.isFinite() || !candidateB.isFinite())
    return;

  // Preserve authored angular-velocity caps.  A capped coupled projection
  // requires an active-set solve and is intentionally outside this predicate.
  if ((bodyA.maxAngularVelocitySq > 0.0f &&
       candidateA.magnitudeSquared() > bodyA.maxAngularVelocitySq) ||
      (bodyB.maxAngularVelocitySq > 0.0f &&
       candidateB.magnitudeSquared() > bodyB.maxAngularVelocitySq))
    return;

  bodyA.angularVelocity = candidateA;
  bodyB.angularVelocity = candidateB;
}

static void projectNativeRevoluteMotorGearVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &motor,
    const AvbdGearJointConstraint &gear, physx::PxReal dt) {
  const physx::PxU32 gearAIndex = gear.header.bodyIndexA;
  const physx::PxU32 gearBIndex = gear.header.bodyIndexB;
  if (gearAIndex >= numBodies || gearBIndex >= numBodies ||
      gearAIndex == gearBIndex)
    return;

  const bool motorDynamicA =
      motor.header.bodyIndexA < numBodies &&
      bodies[motor.header.bodyIndexA].invMass > 0.0f;
  const bool motorDynamicB =
      motor.header.bodyIndexB < numBodies &&
      bodies[motor.header.bodyIndexB].invMass > 0.0f;
  if (motorDynamicA == motorDynamicB)
    return;
  const physx::PxU32 motorBodyIndex =
      motorDynamicA ? motor.header.bodyIndexA : motor.header.bodyIndexB;
  if (motorBodyIndex != gearAIndex && motorBodyIndex != gearBIndex)
    return;

  AvbdSolverBody &gearA = bodies[gearAIndex];
  AvbdSolverBody &gearB = bodies[gearBIndex];
  AvbdSolverBody &motorBody = bodies[motorBodyIndex];
  physx::PxVec3 gearAxisA = gearA.rotation.rotate(gear.gearAxis0);
  physx::PxVec3 gearAxisB = gearB.rotation.rotate(gear.gearAxis1);
  physx::PxVec3 motorAxis =
      motorDynamicA
          ? (motorBody.rotation * motor.localFrameA)
                .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
          : motor.localFrameA.rotate(
                physx::PxVec3(1.0f, 0.0f, 0.0f));
  if (gearAxisA.normalize() <= 1e-6f ||
      gearAxisB.normalize() <= 1e-6f ||
      motorAxis.normalize() <= 1e-6f)
    return;

  // Native revolute target convention is axis dot (wB - wA).  The motor
  // Jacobian therefore changes sign when the dynamic endpoint is actor A.
  const physx::PxReal motorSign = motorDynamicA ? -1.0f : 1.0f;
  const physx::PxVec3 motorJacobian = motorAxis * motorSign;
  const physx::PxVec3 gearJacobianA =
      gearAxisA * gear.gearRatio;
  const physx::PxVec3 gearJacobianB = gearAxisB;
  const physx::PxVec3 motorResponse =
      motorBody.invInertiaWorld.transform(motorJacobian);
  const physx::PxVec3 gearResponseA =
      gearA.invInertiaWorld.transform(gearJacobianA);
  const physx::PxVec3 gearResponseB =
      gearB.invInertiaWorld.transform(gearJacobianB);

  const physx::PxReal kMotorMotor =
      motorJacobian.dot(motorResponse);
  const physx::PxReal kGearGear =
      gearJacobianA.dot(gearResponseA) +
      gearJacobianB.dot(gearResponseB);
  const physx::PxReal kMotorGear =
      motorBodyIndex == gearAIndex
          ? motorJacobian.dot(gearResponseA)
          : motorJacobian.dot(gearResponseB);
  const physx::PxReal determinant =
      kMotorMotor * kGearGear - kMotorGear * kMotorGear;
  if (!physx::PxIsFinite(kMotorMotor) ||
      !physx::PxIsFinite(kGearGear) ||
      !physx::PxIsFinite(kMotorGear) ||
      !physx::PxIsFinite(determinant) ||
      kMotorMotor <= 1e-10f || kGearGear <= 1e-10f ||
      determinant <= 1e-12f)
    return;

  const physx::PxReal motorVelocity =
      motorJacobian.dot(motorBody.angularVelocity);
  const physx::PxReal gearVelocity =
      gearJacobianA.dot(gearA.angularVelocity) +
      gearJacobianB.dot(gearB.angularVelocity);
  const physx::PxReal motorRhs =
      motor.motorTargetVelocity - motorVelocity;
  const physx::PxReal gearRhs = -gearVelocity;
  const physx::PxReal unconstrainedMotorImpulse =
      (motorRhs * kGearGear - gearRhs * kMotorGear) /
      determinant;
  const physx::PxReal maximumMotorImpulse =
      motor.motorMaxForce * dt;
  const physx::PxReal motorImpulse = physx::PxClamp(
      unconstrainedMotorImpulse, -maximumMotorImpulse,
      maximumMotorImpulse);
  // When the motor force limit is active, keep that active-set value and
  // close the passive gear row with the remaining gear impulse.
  const physx::PxReal gearImpulse =
      (gearRhs - kMotorGear * motorImpulse) / kGearGear;
  if (!physx::PxIsFinite(motorImpulse) ||
      !physx::PxIsFinite(gearImpulse))
    return;

  physx::PxVec3 candidateA =
      gearA.angularVelocity + gearResponseA * gearImpulse;
  physx::PxVec3 candidateB =
      gearB.angularVelocity + gearResponseB * gearImpulse;
  if (motorBodyIndex == gearAIndex)
    candidateA += motorResponse * motorImpulse;
  else
    candidateB += motorResponse * motorImpulse;
  if (!candidateA.isFinite() || !candidateB.isFinite())
    return;
  if ((gearA.maxAngularVelocitySq > 0.0f &&
       candidateA.magnitudeSquared() > gearA.maxAngularVelocitySq) ||
      (gearB.maxAngularVelocitySq > 0.0f &&
       candidateB.magnitudeSquared() > gearB.maxAngularVelocitySq))
    return;

  gearA.angularVelocity = candidateA;
  gearB.angularVelocity = candidateB;
}

static bool isSinglePassiveGenericHard1DVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts, physx::PxU32 &genericIndex) {
  genericIndex = PX_MAX_U32;
  if (!bodies || numBodies == 0 || numContacts != 0 || !d6Joints ||
      numD6 == 0 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  for (physx::PxU32 i = 0; i < numD6; ++i) {
    if (!hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::GenericHard1D) &&
        !hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::ArticulationHardMimic) &&
        !hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::GenericRestitution1D))
      continue;
    if (genericIndex != PX_MAX_U32)
      return false;
    genericIndex = i;
  }
  if (genericIndex == PX_MAX_U32)
    return false;

  const AvbdD6JointConstraint &generic = d6Joints[genericIndex];
  if (generic.header.type != AvbdConstraintType::eJOINT_CUSTOM_1D ||
      !generic.genericLinearA.isFinite() ||
      !generic.genericAngularA.isFinite() ||
      !generic.genericLinearB.isFinite() ||
      !generic.genericAngularB.isFinite() ||
      !physx::PxIsFinite(generic.genericVelocityTarget) ||
      generic.genericMinImpulse >= 0.0f ||
      generic.genericMaxImpulse <= 0.0f)
    return false;

  const physx::PxU32 endpoint[2] = {generic.header.bodyIndexA,
                                    generic.header.bodyIndexB};
  if (endpoint[0] < numBodies && endpoint[1] < numBodies &&
      endpoint[0] == endpoint[1])
    return false;
  const physx::PxVec3 linearJ[2] = {generic.genericLinearA,
                                    generic.genericLinearB};
  const physx::PxVec3 angularJ[2] = {generic.genericAngularA,
                                     generic.genericAngularB};
  bool ownsDynamicEndpoint = false;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    if (bodies[endpoint[side]].invMass <= 0.0f)
      return false;
    ownsDynamicEndpoint = true;

    const bool hasLinear = linearJ[side].magnitudeSquared() > 1e-10f;
    const bool hasAngular = angularJ[side].magnitudeSquared() > 1e-10f;
    // A general spatial row needs a coupled velocity active-set solve.
    // This accepted subdomain has at most one response kind per endpoint.
    if (hasLinear && hasAngular)
      return false;
    if (hasAngular) {
      const physx::PxVec3 response =
          bodies[endpoint[side]].invInertiaWorld * angularJ[side];
      const physx::PxReal scale =
          physx::PxMax(response.magnitude(), 1e-8f);
      if (response.cross(angularJ[side]).magnitude() > scale * 1e-4f)
        return false;
    }
  }
  if (!ownsDynamicEndpoint)
    return false;

  // Any accompanying D6 row must be a centered, passive world attachment.
  // The generic Jacobian may only occupy DOFs that attachment leaves FREE.
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    if (i == genericIndex)
      continue;
    const AvbdD6JointConstraint &joint = d6Joints[i];
    const bool aDynamic = joint.header.bodyIndexA < numBodies;
    const bool bDynamic = joint.header.bodyIndexB < numBodies;
    if (aDynamic == bDynamic || joint.driveFlags != 0u ||
        joint.motorEnabled != 0u)
      return false;
    const physx::PxU32 dynamicIndex =
        aDynamic ? joint.header.bodyIndexA : joint.header.bodyIndexB;
    physx::PxU32 side = PX_MAX_U32;
    if (dynamicIndex == endpoint[0])
      side = 0;
    else if (dynamicIndex == endpoint[1])
      side = 1;
    if (side == PX_MAX_U32)
      return false;

    const physx::PxVec3 &dynamicAnchor =
        aDynamic ? joint.anchorA : joint.anchorB;
    if (dynamicAnchor.magnitudeSquared() > 1e-8f)
      return false;
    const physx::PxQuat &localFrame =
        aDynamic ? joint.localFrameA : joint.localFrameB;
    const physx::PxQuat worldFrame =
        bodies[dynamicIndex].rotation * localFrame;
    for (physx::PxU32 axisIndex = 0; axisIndex < 3; ++axisIndex) {
      physx::PxVec3 localAxis(0.0f);
      localAxis[axisIndex] = 1.0f;
      physx::PxVec3 worldAxis = worldFrame.rotate(localAxis);
      if (joint.getLinearMotion(axisIndex) != 2 &&
          physx::PxAbs(linearJ[side].dot(worldAxis)) > 1e-5f)
        return false;
      if (joint.getAngularMotion(axisIndex) != 2 &&
          physx::PxAbs(angularJ[side].dot(worldAxis)) > 1e-5f)
        return false;
    }
  }
  return true;
}

static void projectSinglePassiveGenericHard1DVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &generic,
    const physx::PxArray<physx::PxVec3> &linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart) {
  if (linearVelAtSolveStart.size() != numBodies ||
      angularVelAtSolveStart.size() != numBodies)
    return;

  const physx::PxU32 endpoint[2] = {generic.header.bodyIndexA,
                                    generic.header.bodyIndexB};
  const physx::PxVec3 linearJ[2] = {generic.genericLinearA,
                                    generic.genericLinearB};
  const physx::PxVec3 angularJ[2] = {generic.genericAngularA,
                                     generic.genericAngularB};
  physx::PxVec3 linearResponse[2] = {physx::PxVec3(0.0f),
                                     physx::PxVec3(0.0f)};
  physx::PxVec3 angularResponse[2] = {physx::PxVec3(0.0f),
                                      physx::PxVec3(0.0f)};
  physx::PxReal denominator = 0.0f;
  physx::PxReal startSpeed = 0.0f;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    AvbdSolverBody &body = bodies[endpoint[side]];
    linearResponse[side] = linearJ[side] * body.invMass;
    angularResponse[side] =
        body.invInertiaWorld.transform(angularJ[side]);
    denominator += linearJ[side].dot(linearResponse[side]) +
                   angularJ[side].dot(angularResponse[side]);
    startSpeed +=
        linearJ[side].dot(linearVelAtSolveStart[endpoint[side]]) +
        angularJ[side].dot(angularVelAtSolveStart[endpoint[side]]);
  }
  if (!physx::PxIsFinite(denominator) || denominator <= 1e-10f)
    return;

  physx::PxReal velocityTarget = generic.genericVelocityTarget;
  if (hasAvbdJointObjective(
          generic.objectiveProgram,
          AvbdJointObjectiveKind::GenericRestitution1D)) {
    const physx::PxReal bounceVelocity =
        -generic.genericRestitution * startSpeed;
    if (-startSpeed > generic.genericBounceThreshold &&
        bounceVelocity * generic.genericGeometricError <= 0.0f)
      velocityTarget = bounceVelocity;
  }
  const physx::PxReal residual = startSpeed - velocityTarget;
  const physx::PxReal impulse = physx::PxClamp(
      -residual / denominator, generic.genericMinImpulse,
      generic.genericMaxImpulse);
  if (!physx::PxIsFinite(impulse))
    return;

  physx::PxVec3 candidateLinear[2];
  physx::PxVec3 candidateAngular[2];
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    AvbdSolverBody &body = bodies[endpoint[side]];
    const physx::PxVec3 expectedLinear =
        linearVelAtSolveStart[endpoint[side]] +
        linearResponse[side] * impulse;
    const physx::PxVec3 expectedAngular =
        angularVelAtSolveStart[endpoint[side]] +
        angularResponse[side] * impulse;
    candidateLinear[side] = body.linearVelocity;
    candidateAngular[side] = body.angularVelocity;
    if (linearJ[side].magnitudeSquared() > 1e-10f) {
      physx::PxVec3 axis = linearJ[side].getNormalized();
      candidateLinear[side] +=
          axis * (expectedLinear.dot(axis) -
                  candidateLinear[side].dot(axis));
    }
    if (angularJ[side].magnitudeSquared() > 1e-10f) {
      physx::PxVec3 axis = angularJ[side].getNormalized();
      candidateAngular[side] +=
          axis * (expectedAngular.dot(axis) -
                  candidateAngular[side].dot(axis));
    }
    if (!candidateLinear[side].isFinite() ||
        !candidateAngular[side].isFinite())
      return;
    if ((body.maxLinearVelocitySq > 0.0f &&
         candidateLinear[side].magnitudeSquared() >
             body.maxLinearVelocitySq) ||
        (body.maxAngularVelocitySq > 0.0f &&
         candidateAngular[side].magnitudeSquared() >
             body.maxAngularVelocitySq))
      return;
  }

  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    bodies[endpoint[side]].linearVelocity = candidateLinear[side];
    bodies[endpoint[side]].angularVelocity = candidateAngular[side];
  }

  const bool outputForce =
      (generic.genericRowFlags &
       static_cast<physx::PxU32>(Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;
  generic.writebackLinearImpulse =
      outputForce ? generic.genericLinearA * impulse : physx::PxVec3(0.0f);
  generic.writebackAngularImpulse =
      outputForce ? generic.genericAngularAWriteback * impulse
                  : physx::PxVec3(0.0f);
  generic.writebackLinearImpulseValid = 1;
  generic.writebackAngularImpulseValid = 1;
}

static void projectArticulationMimicVelocity1D(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &mimic) {
  const physx::PxReal effectiveMass =
      computeGeneric1DEffectiveMass(mimic, bodies, numBodies);
  if (effectiveMass <= 0.0f || !physx::PxIsFinite(effectiveMass))
    return;
  const physx::PxReal unitResponse = 1.0f / effectiveMass;
  const physx::PxU32 endpoint[2] = {mimic.header.bodyIndexA,
                                    mimic.header.bodyIndexB};
  const physx::PxVec3 linearJ[2] = {mimic.genericLinearA,
                                    mimic.genericLinearB};
  const physx::PxVec3 angularJ[2] = {mimic.genericAngularA,
                                     mimic.genericAngularB};
  physx::PxReal velocityResidual = -mimic.genericVelocityTarget;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies)
      continue;
    velocityResidual +=
        linearJ[side].dot(bodies[endpoint[side]].linearVelocity) +
        angularJ[side].dot(bodies[endpoint[side]].angularVelocity);
  }
  const physx::PxReal velocityImpulse =
      -velocityResidual / unitResponse;
  if (!physx::PxIsFinite(velocityImpulse))
    return;
  for (physx::PxU32 side = 0; side < 2; ++side) {
    if (endpoint[side] >= numBodies ||
        bodies[endpoint[side]].invMass <= 0.0f)
      continue;
    AvbdSolverBody &body = bodies[endpoint[side]];
    body.linearVelocity +=
        linearJ[side] * (body.invMass * velocityImpulse);
    body.angularVelocity +=
        body.invInertiaWorld * angularJ[side] * velocityImpulse;
  }
}

// Matches joint::computeJacobianAxes in ExtConstraintHelper.h.  These rows are
// the world-space derivatives of imag(qa^-1*qb) with respect to wB-wA.  A
// SLERP spring uses them when stiffness is non-zero; only velocity-only SLERP
// keeps the fixed world X/Y/Z rows.
static void computeSlerpJacobianAxes(physx::PxVec3 rows[3],
                                     const physx::PxQuat &qa,
                                     const physx::PxQuat &qb) {
  const physx::PxReal wa = qa.w;
  const physx::PxReal wb = qb.w;
  const physx::PxVec3 va(qa.x, qa.y, qa.z);
  const physx::PxVec3 vb(qb.x, qb.y, qb.z);
  const physx::PxVec3 c = vb * wa + va * wb;
  const physx::PxReal d0 = wa * wb;
  const physx::PxReal d1 = va.dot(vb);
  const physx::PxReal d = d0 - d1;

  rows[0] =
      (va * vb.x + vb * va.x + physx::PxVec3(d, c.z, -c.y)) * 0.5f;
  rows[1] =
      (va * vb.y + vb * va.y + physx::PxVec3(-c.z, d, c.x)) * 0.5f;
  rows[2] =
      (va * vb.z + vb * va.z + physx::PxVec3(c.y, -c.x, d)) * 0.5f;

  if ((d0 + d1) == 0.0f) {
    rows[0].x += PX_EPS_F32;
    rows[1].y += PX_EPS_F32;
    rows[2].z += PX_EPS_F32;
  }
}

struct CoupledIslandRow {
  AvbdVec6 jacobianA;
  AvbdVec6 jacobianB;
  physx::PxU32 bodyA;
  physx::PxU32 bodyB;
  physx::PxReal penalty;
  physx::PxReal force;
};

static bool findCoupledSpatialTendonRows(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts,
    physx::PxArray<physx::PxU32> &rowIndices) {
  rowIndices.clear();
  if (!bodies || numBodies < 2 || numContacts != 0 || !d6Joints ||
      numD6 == 0 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdD6JointConstraint &joint = d6Joints[i];
    if ((joint.sourceFlags &
         AvbdD6JointConstraint::eARTICULATION_SPATIAL_TENDON_ROW) == 0)
      continue;
    const physx::PxU32 bodyA = joint.header.bodyIndexA;
    const physx::PxU32 bodyB = joint.header.bodyIndexB;
    const physx::PxReal jacobianMagnitudeSquaredA =
        joint.genericLinearA.magnitudeSquared() +
        joint.genericAngularA.magnitudeSquared();
    const physx::PxReal jacobianMagnitudeSquaredB =
        joint.genericLinearB.magnitudeSquared() +
        joint.genericAngularB.magnitudeSquared();
    if (bodyA >= numBodies || bodyB >= numBodies ||
        bodyA == bodyB || bodies[bodyA].invMass <= 0.0f ||
        bodies[bodyB].invMass <= 0.0f ||
        !PxIsFinite(joint.header.rho) ||
        joint.header.rho < 0.0f ||
        !PxIsFinite(joint.header.damping) ||
        joint.header.damping < 0.0f ||
        !PxIsFinite(joint.genericTendonLimitStiffness) ||
        joint.genericTendonLimitStiffness < 0.0f ||
        (joint.header.rho <= 0.0f &&
         joint.genericTendonLimitStiffness <= 0.0f) ||
        jacobianMagnitudeSquaredA <= 1e-12f ||
        jacobianMagnitudeSquaredB <= 1e-12f ||
        !joint.genericLinearA.isFinite() ||
        !joint.genericAngularA.isFinite() ||
        !joint.genericLinearB.isFinite() ||
        !joint.genericAngularB.isFinite())
      return false;
    rowIndices.pushBack(i);
  }
  return !rowIndices.empty();
}

static AvbdVec6 multiplyBlock(const AvbdBlock6x6 &block,
                              const AvbdVec6 &value) {
  return AvbdVec6(block.linearLinear * value.linear +
                      block.linearAngular * value.angular,
                  block.angularLinear * value.linear +
                      block.angularAngular * value.angular);
}

static void addScaled(AvbdVec6 &target, const AvbdVec6 &value,
                      physx::PxReal scale) {
  target.linear += value.linear * scale;
  target.angular += value.angular * scale;
}

static double dotVectors(const physx::PxArray<AvbdVec6> &a,
                         const physx::PxArray<AvbdVec6> &b) {
  double result = 0.0;
  for (physx::PxU32 i = 0; i < a.size(); ++i)
    result += static_cast<double>(a[i].linear.x) * b[i].linear.x +
              static_cast<double>(a[i].linear.y) * b[i].linear.y +
              static_cast<double>(a[i].linear.z) * b[i].linear.z +
              static_cast<double>(a[i].angular.x) * b[i].angular.x +
              static_cast<double>(a[i].angular.y) * b[i].angular.y +
              static_cast<double>(a[i].angular.z) * b[i].angular.z;
  return result;
}

static void addCoupledRow(const CoupledIslandRow &row,
                          physx::PxArray<CoupledIslandRow> &rows,
                          physx::PxArray<AvbdVec6> &gradient,
                          physx::PxArray<AvbdBlock6x6> &preconditioner) {
  rows.pushBack(row);
  if (row.bodyA != PX_MAX_U32) {
    addScaled(gradient[row.bodyA], row.jacobianA, row.force);
    preconditioner[row.bodyA].addConstraintContribution(
        row.jacobianA.linear, row.jacobianA.angular, row.penalty);
  }
  if (row.bodyB != PX_MAX_U32) {
    addScaled(gradient[row.bodyB], row.jacobianB, row.force);
    preconditioner[row.bodyB].addConstraintContribution(
        row.jacobianB.linear, row.jacobianB.angular, row.penalty);
  }
}

static void applyCoupledOperator(
    const physx::PxArray<AvbdBlock6x6> &inertialBlocks,
    const physx::PxArray<CoupledIslandRow> &rows,
    const physx::PxArray<AvbdVec6> &input,
    physx::PxArray<AvbdVec6> &output) {
  output.resize(input.size());
  for (physx::PxU32 i = 0; i < input.size(); ++i)
    output[i] = multiplyBlock(inertialBlocks[i], input[i]);
  for (physx::PxU32 i = 0; i < rows.size(); ++i) {
    const CoupledIslandRow &row = rows[i];
    physx::PxReal projection = 0.0f;
    if (row.bodyA != PX_MAX_U32)
      projection += row.jacobianA.dot(input[row.bodyA]);
    if (row.bodyB != PX_MAX_U32)
      projection += row.jacobianB.dot(input[row.bodyB]);
    const physx::PxReal scale = row.penalty * projection;
    if (row.bodyA != PX_MAX_U32)
      addScaled(output[row.bodyA], row.jacobianA, scale);
    if (row.bodyB != PX_MAX_U32)
      addScaled(output[row.bodyB], row.jacobianB, scale);
  }
}

static bool areFrictionlessBodyVsStaticContactsSupported(
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 numBodies) {
  if (numContacts != 0 && !contacts)
    return false;
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        contact.friction > 0.0f || contact.staticFriction > 0.0f ||
        hasDeformableStaticAnchor(contact))
      return false;
  }
  return true;
}

static bool areTorqueFreeBodyVsStaticContactsSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts) {
  if (!bodies ||
      !areFrictionlessBodyVsStaticContactsSupported(
          contacts, numContacts, numBodies))
    return false;
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const bool dynamicA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicA ? contact.header.bodyIndexA : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies)
      return false;
    const physx::PxVec3 localPoint =
        dynamicA ? contact.contactPointA : contact.contactPointB;
    const physx::PxVec3 momentArm =
        bodies[bodyIndex].rotation.rotate(localPoint);
    if (momentArm.cross(contact.contactNormal).magnitudeSquared() > 1e-10f)
      return false;
  }
  return true;
}

static bool areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity) {
  if (!bodies || numBodies != 2u || !contacts ||
      numContacts < 2u || numContacts > 8u ||
      !gravity.isFinite() || gravity.magnitudeSquared() <= 1e-12f)
    return false;

  const physx::PxVec3 up = -gravity.getNormalized();
  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  physx::PxVec3 referenceNormal(0.0f);
  bool haveReferenceNormal = false;
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contact) ||
        hasKinematicShellAnchor(contact) ||
        (contact.friction <= 0.0f &&
         contact.staticFriction <= 0.0f) ||
        !PxIsFinite(contact.friction) ||
        !PxIsFinite(contact.staticFriction) ||
        !PxIsFinite(contact.restitution) ||
        physx::PxAbs(contact.restitution) > 1e-6f ||
        !contact.targetVelocity.isFinite() ||
        !contact.contactNormal.isFinite() ||
        !contact.tangent0.isFinite() ||
        !contact.tangent1.isFinite() ||
        physx::PxAbs(contact.tangent0.magnitudeSquared() - 1.0f) >
            1e-4f ||
        physx::PxAbs(contact.tangent1.magnitudeSquared() - 1.0f) >
            1e-4f ||
        physx::PxAbs(contact.tangent0.dot(contact.tangent1)) > 1e-4f ||
        contact.targetVelocity.magnitudeSquared() > 1e-12f)
      return false;

    const bool dynamicIsA =
        contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicIsA ? contact.header.bodyIndexA
                   : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies ||
        physx::PxAbs((dynamicIsA ? contact.invMassScaleA
                                 : contact.invMassScaleB) -
                    1.0f) > 1e-6f ||
        physx::PxAbs((dynamicIsA ? contact.invInertiaScaleA
                                 : contact.invInertiaScaleB) -
                    1.0f) > 1e-6f)
      return false;

    physx::PxVec3 dynamicNormal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    if (dynamicNormal.normalize() <= 1e-6f ||
        // A stationary upward-facing slope is still an ordinary persistent
        // rigid support. The complete position owner already uses the full
        // two-tangent basis and computes its Coulomb budget from
        // |gravity dot normal| below, so gravity alignment is not a solver
        // requirement. Retain only a numerical non-degeneracy boundary that
        // excludes vertical/downward surfaces from this strict authority.
        up.dot(dynamicNormal) <= 1e-6f)
      return false;
    if (!haveReferenceNormal) {
      referenceNormal = dynamicNormal;
      haveReferenceNormal = true;
    } else if (referenceNormal.dot(dynamicNormal) < 0.9999f) {
      return false;
    }

    const physx::PxVec3 localPoint =
        dynamicIsA ? contact.contactPointA
                   : contact.contactPointB;
    if (!localPoint.isFinite())
      return false;
    const physx::PxVec3 contactArm =
        bodies[bodyIndex].rotation.rotate(localPoint);
    if (contactArm.cross(dynamicNormal).magnitudeSquared() >
        1e-10f)
      return false;
    const physx::PxVec3 pointVelocity =
        bodies[bodyIndex].linearVelocity +
        bodies[bodyIndex].angularVelocity.cross(contactArm);
    if (pointVelocity.dot(dynamicNormal) < -0.25f)
      return false;
    contactsPerBody[bodyIndex]++;
  }
  return contactsPerBody[0] != 0u && contactsPerBody[1] != 0u;
}

static bool addBodyVsStaticContactNormalRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner,
    bool allowFriction) {
  if (!allowFriction) {
    if (!areFrictionlessBodyVsStaticContactsSupported(
            contacts, numContacts, numBodies))
      return false;
  } else {
    if (!contacts || numContacts == 0u)
      return false;
    for (physx::PxU32 i = 0; i < numContacts; ++i) {
      if (!isBodyVsStaticContact(
              contacts[i].header.bodyIndexA,
              contacts[i].header.bodyIndexB, numBodies) ||
          hasDeformableStaticAnchor(contacts[i]) ||
          hasKinematicShellAnchor(contacts[i]))
        return false;
    }
  }

  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const bool dynamicA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicA ? contact.header.bodyIndexA : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies || bodies[bodyIndex].invMass <= 0.0f)
      return false;

    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxVec3 localPoint =
        dynamicA ? contact.contactPointA : contact.contactPointB;
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 worldDynamic = body.position + r;
    const physx::PxVec3 worldStatic =
        dynamicA ? contact.contactPointB : contact.contactPointA;
    physx::PxReal violation =
        (dynamicA ? worldDynamic - worldStatic : worldStatic - worldDynamic)
            .dot(contact.contactNormal) +
        contact.penetrationDepth;
    violation -= config.avbdAlpha * contact.C0;
    const physx::PxReal massInvDt2 =
        (1.0f / body.invMass) * invDt2;
    const physx::PxReal penalty = physx::PxMax(
        contact.header.penalty,
        AvbdConstants::AVBD_CONTACT_BOOST_FRACTION * massInvDt2);
    const physx::PxReal force =
        physx::PxMin(0.0f, penalty * violation + contact.header.lambda);
    const physx::PxReal sign = dynamicA ? 1.0f : -1.0f;
    const physx::PxVec3 contactAxis = contact.contactNormal * sign;
    CoupledIslandRow row;
    row.bodyA = bodyIndex;
    row.bodyB = PX_MAX_U32;
    row.jacobianA =
        AvbdVec6(contactAxis, r.cross(contact.contactNormal) * sign);
    row.jacobianB = AvbdVec6();
    row.penalty = penalty;
    row.force = force;
    addCoupledRow(row, rows, gradient, preconditioner);
  }
  return true;
}

static bool addFrictionlessBodyVsStaticContactRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner) {
  return addBodyVsStaticContactNormalRows(
      bodies, numBodies, contacts, numContacts, config, invDt2,
      rows, gradient, preconditioner, false);
}

static bool addStrictFrictionalBodyVsStaticContactPositionRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, const AvbdSolverConfig &config,
    physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner) {
  if (!areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts, gravity))
    return false;

  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies)
      return false;
    contactsPerBody[bodyIndex]++;
  }
  const physx::PxReal mass0 = 1.0f / bodies[0].invMass;
  const physx::PxReal mass1 = 1.0f / bodies[1].invMass;
  const bool unequalEndpointMasses =
      physx::PxAbs(mass0 - mass1) >
      1e-6f * physx::PxMax(mass0, mass1);

  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const bool dynamicA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicA ? contact.header.bodyIndexA
                 : contact.header.bodyIndexB;
    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxVec3 localPoint =
        dynamicA ? contact.contactPointA
                 : contact.contactPointB;
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 previousR =
        body.prevRotation.rotate(localPoint);
    const physx::PxVec3 displacement =
        (body.position + r) -
        (body.prevPosition + previousR);
    const physx::PxReal sign = dynamicA ? 1.0f : -1.0f;
    const physx::PxReal tangentViolation0 =
        sign * displacement.dot(contact.tangent0);
    const physx::PxReal tangentViolation1 =
        sign * displacement.dot(contact.tangent1);

    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxReal contactBoostFloor =
        AvbdConstants::AVBD_CONTACT_BOOST_FRACTION *
        mass * invDt2;
    physx::PxVec3 dynamicNormal =
        contact.contactNormal * sign;
    if (dynamicNormal.normalize() <= 1e-6f)
      return false;
    const physx::PxReal weightShare =
        mass * physx::PxAbs(gravity.dot(dynamicNormal)) /
        physx::PxReal(contactsPerBody[bodyIndex]);
    physx::PxReal normalCapacity = weightShare;
    if (!unequalEndpointMasses) {
      const physx::PxVec3 worldDynamic = body.position + r;
      const physx::PxVec3 worldStatic =
          dynamicA ? contact.contactPointB : contact.contactPointA;
      physx::PxReal normalViolation =
          (dynamicA ? worldDynamic - worldStatic
                    : worldStatic - worldDynamic)
              .dot(contact.contactNormal) +
          contact.penetrationDepth;
      normalViolation -= config.avbdAlpha * contact.C0;
      const physx::PxReal normalPenalty =
          physx::PxMax(contact.header.penalty, contactBoostFloor);
      const physx::PxReal normalForce =
          physx::PxMin(0.0f, normalPenalty * normalViolation +
                                contact.header.lambda);
      const physx::PxReal priorNormalForce =
          contact.header.lambda < 0.0f ? -contact.header.lambda : 0.0f;
      normalCapacity = physx::PxMax(
          weightShare,
          physx::PxMax(-normalForce, priorNormalForce));
    }
    // The two support normals and the locked joint-normal row are redundant.
    // With unequal endpoint masses the AL normal multiplier can transfer the
    // heavy body's reaction to the light endpoint, so only the exact
    // per-body weight share is an admissible Coulomb budget. Preserve the
    // accepted symmetric P4AB normal-history budget when both endpoint
    // masses are equal; no cross-mass attribution exists in that boundary.
    const physx::PxReal mu = contactCoulombMu(contact);
    const physx::PxReal tangentPenalty0 =
        physx::PxMax(contact.tangentPenalty0,
                     contactBoostFloor);
    const physx::PxReal tangentPenalty1 =
        physx::PxMax(contact.tangentPenalty1,
                     contactBoostFloor);
    physx::PxReal tangentForce0 =
        tangentPenalty0 * tangentViolation0 +
        contact.tangentLambda0;
    physx::PxReal tangentForce1 =
        tangentPenalty1 * tangentViolation1 +
        contact.tangentLambda1;
    const physx::PxReal unconstrainedTangentForce =
        physx::PxSqrt(tangentForce0 * tangentForce0 +
                      tangentForce1 * tangentForce1);
    const physx::PxReal tangentForceLimit =
        mu * normalCapacity;
    avbdProjectImpulseCone(tangentForceLimit,
                           tangentForce0, tangentForce1);
    const bool forceSaturated =
        unconstrainedTangentForce > tangentForceLimit;

    const physx::PxVec3 tangents[2] = {
        contact.tangent0, contact.tangent1};
    const physx::PxReal tangentPenalties[2] = {
        tangentPenalty0, tangentPenalty1};
    const physx::PxReal tangentForces[2] = {
        tangentForce0, tangentForce1};
    for (physx::PxU32 tangent = 0; tangent < 2u;
         ++tangent) {
      const physx::PxVec3 axis =
          tangents[tangent] * sign;
      CoupledIslandRow row;
      row.bodyA = bodyIndex;
      row.bodyB = PX_MAX_U32;
      row.jacobianA =
          AvbdVec6(axis, r.cross(tangents[tangent]) * sign);
      row.jacobianB = AvbdVec6();
      // Outside the Coulomb disk the projected force is locally bounded,
      // so retaining an unconstrained tangent Hessian would make the
      // saturated row artificially bilateral.
      row.penalty =
          forceSaturated ? 0.0f : tangentPenalties[tangent];
      row.force = tangentForces[tangent];
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  }
  return true;
}

static bool isCoupledFixedD6IslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  if (joint.header.type != AvbdConstraintType::eJOINT_FIXED ||
      joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      physx::PxAbs(bodies[joint.header.bodyIndexA].invMass -
                    bodies[joint.header.bodyIndexB].invMass) >
          1e-6f * physx::PxMax(
                      bodies[joint.header.bodyIndexA].invMass,
                      bodies[joint.header.bodyIndexB].invMass) ||
      bodies[joint.header.bodyIndexA].lockFlags != 0 ||
      bodies[joint.header.bodyIndexB].lockFlags != 0 ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexA].angularDampingBody != 0.0f ||
      bodies[joint.header.bodyIndexB].angularDampingBody != 0.0f ||
      joint.linearMotion != 0 || joint.angularMotion != 0 ||
      joint.driveFlags != 0 || joint.driveAccelerationFlags != 0 ||
      joint.motorEnabled != 0 || joint.coneAngleLimit > 0.0f ||
      !(joint.header.rho > 0.0f) || !PxIsFinite(joint.header.rho) ||
      !joint.anchorA.isFinite() || !joint.anchorB.isFinite() ||
      !joint.localFrameA.isFinite() || !joint.localFrameB.isFinite())
    return false;
  return true;
}

static bool isCoupledSphericalConeIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (joint.header.type != AvbdConstraintType::eJOINT_SPHERICAL ||
      bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
      bodies[bodyA].invMass <= 0.0f || bodies[bodyB].invMass <= 0.0f ||
      bodies[bodyA].lockFlags != 0 || bodies[bodyB].lockFlags != 0 ||
      bodies[bodyA].linearDamping != 0.0f ||
      bodies[bodyB].linearDamping != 0.0f ||
      bodies[bodyA].angularDampingBody != 0.0f ||
      bodies[bodyB].angularDampingBody != 0.0f ||
      joint.linearMotion != 0 || joint.angularMotion != 0x2Au ||
      joint.driveFlags != 0 || joint.driveAccelerationFlags != 0 ||
      joint.motorEnabled != 0 ||
      (joint.sourceFlags & AvbdD6JointConstraint::
           eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE) == 0 ||
      joint.coneAngleLimit <= 0.0f || joint.coneAngleLimitZ <= 0.0f ||
      !(joint.header.rho > 0.0f) || !PxIsFinite(joint.header.rho) ||
      !joint.anchorA.isFinite() || !joint.anchorB.isFinite() ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      !joint.localFrameA.isFinite() || !joint.localFrameB.isFinite())
    return false;
  return true;
}

// Solve all six bilateral rows of one native PxFixedJoint against both
// dynamic endpoints in a single frozen Newton system.  Per-body block descent
// omits the off-diagonal J_A^T J_B blocks and can therefore inject a common
// translation into a free island under equal-and-opposite loads.
static bool solveCoupledFixedD6Island(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &joint, physx::PxReal invDt2) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(6);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal penalty =
      physx::PxMax(joint.header.rho, physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 linearViolation =
      bodyA.position + rA - bodyB.position - rB;
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(worldAxes[axis], rA.cross(worldAxes[axis]));
    row.jacobianB =
        AvbdVec6(-worldAxes[axis], -rB.cross(worldAxes[axis]));
    row.penalty = penalty;
    row.force = penalty * linearViolation.dot(worldAxes[axis]) +
                joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = worldFrameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(physx::PxVec3(0.0f), worldAxis);
    row.jacobianB = AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
    row.penalty = penalty;
    row.force =
        penalty *
            joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis) +
        joint.lambdaAngular[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) || !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual = 1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction, operatorDirection);
    const double denominator = dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) || !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) || !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] = preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                               solution[i].angular.y,
                               solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation - delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
}

// Solve the three spherical anchor rows and its active cone inequality against
// both dynamic endpoints in one frozen Newton system.  The cone row is an
// internal equal-and-opposite angular constraint; per-body block descent drops
// its off-diagonal block and can rotate both endpoints in the same direction.
static bool solveCoupledSphericalConeIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &joint, physx::PxReal invDt2) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(4);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f,
                                      deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal penalty =
      physx::PxMax(joint.header.rho,
                   physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 linearViolation =
      bodyA.position + rA - bodyB.position - rB;
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(worldAxes[axis], rA.cross(worldAxes[axis]));
    row.jacobianB =
        AvbdVec6(-worldAxes[axis], -rB.cross(worldAxes[axis]));
    row.penalty = penalty;
    row.force = penalty * linearViolation.dot(worldAxes[axis]) +
                joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxVec3 coneAxis(0.0f);
  physx::PxReal coneViolation = 0.0f;
  if (!computeEllipticalConeConstraint(
          joint, bodyA.rotation, bodyB.rotation, coneAxis,
          coneViolation))
    return false;
  const physx::PxReal coneForce =
      penalty * coneViolation - joint.coneLambda;
  if (coneForce > 0.0f &&
      coneAxis.magnitudeSquared() > 1e-12f) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(physx::PxVec3(0.0f), -coneAxis);
    row.jacobianB =
        AvbdVec6(physx::PxVec3(0.0f), coneAxis);
    row.penalty = penalty;
    row.force = coneForce;
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(
            preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) || !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual =
      1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction,
                         operatorDirection);
    const double denominator =
        dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) || !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) || !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] =
            preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                               solution[i].angular.y,
                               solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation -
           delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
}

// Project the velocity counterpart of the same six fixed-joint rows.  This is
// one bilateral impulse solve, with no speed threshold and no common-mode
// momentum correction: J M^-1 J^T impulse = -J velocity.
static bool projectCoupledFixedD6Velocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  if (!bodies || bodyAIndex >= numBodies || bodyBIndex >= numBodies)
    return false;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];

  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = worldFrameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameMagnitude);

  AvbdVec6 jacobianA[6];
  AvbdVec6 jacobianB[6];
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    jacobianA[axis] =
        AvbdVec6(worldAxes[axis], rA.cross(worldAxes[axis]));
    jacobianB[axis] =
        AvbdVec6(-worldAxes[axis], -rB.cross(worldAxes[axis]));
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    const physx::PxVec3 angularAxis = worldFrameA.rotate(localAxis);
    jacobianA[3 + axis] =
        AvbdVec6(physx::PxVec3(0.0f), angularAxis);
    jacobianB[3 + axis] =
        AvbdVec6(physx::PxVec3(0.0f), -angularAxis);
  }

  const AvbdVec6 velocityA(bodyA.linearVelocity, bodyA.angularVelocity);
  const AvbdVec6 velocityB(bodyB.linearVelocity, bodyB.angularVelocity);
  AvbdVec6 residual;
  for (physx::PxU32 row = 0; row < 6; ++row) {
    const physx::PxReal value =
        jacobianA[row].dot(velocityA) + jacobianB[row].dot(velocityB);
    if (row < 3)
      residual.linear[row] = value;
    else
      residual.angular[row - 3] = value;
  }

  AvbdBlock6x6 response;
  response.setZero();
  const auto setResponse = [&response](physx::PxU32 row,
                                       physx::PxU32 column,
                                       physx::PxReal value) {
    if (row < 3 && column < 3)
      response.linearLinear(row, column) = value;
    else if (row < 3)
      response.linearAngular(row, column - 3) = value;
    else if (column < 3)
      response.angularLinear(row - 3, column) = value;
    else
      response.angularAngular(row - 3, column - 3) = value;
  };
  for (physx::PxU32 row = 0; row < 6; ++row) {
    for (physx::PxU32 column = 0; column < 6; ++column) {
      const AvbdVec6 responseA(
          jacobianA[column].linear * bodyA.invMass,
          bodyA.invInertiaWorld * jacobianA[column].angular);
      const AvbdVec6 responseB(
          jacobianB[column].linear * bodyB.invMass,
          bodyB.invInertiaWorld * jacobianB[column].angular);
      setResponse(row, column,
                  jacobianA[row].dot(responseA) +
                      jacobianB[row].dot(responseB));
    }
  }

  AvbdLDLT responseLdlt;
  if (!responseLdlt.decomposeWithRegularization(response))
    return false;
  const AvbdVec6 impulse = responseLdlt.solve(-residual);
  if (!impulse.linear.isFinite() || !impulse.angular.isFinite())
    return false;

  AvbdVec6 bodyImpulseA;
  AvbdVec6 bodyImpulseB;
  for (physx::PxU32 row = 0; row < 6; ++row) {
    const physx::PxReal rowImpulse =
        row < 3 ? impulse.linear[row] : impulse.angular[row - 3];
    addScaled(bodyImpulseA, jacobianA[row], rowImpulse);
    addScaled(bodyImpulseB, jacobianB[row], rowImpulse);
  }
  bodyA.linearVelocity += bodyImpulseA.linear * bodyA.invMass;
  bodyA.angularVelocity +=
      bodyA.invInertiaWorld * bodyImpulseA.angular;
  bodyB.linearVelocity += bodyImpulseB.linear * bodyB.invMass;
  bodyB.angularVelocity +=
      bodyB.invInertiaWorld * bodyImpulseB.angular;
  return true;
}

static bool solveCoupledSpatialTendonRow(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2) {
  if (!bodies || dt <= 0.0f)
    return false;

  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB)
    return false;

  AvbdSolverBody &endpointA = bodies[bodyA];
  AvbdSolverBody &endpointB = bodies[bodyB];
  const AvbdVec6 jacobianA(
      joint.genericLinearA, joint.genericAngularA);
  const AvbdVec6 jacobianB(
      joint.genericLinearB, joint.genericAngularB);
  const physx::PxReal jacobianMagnitudeA = physx::PxSqrt(
      jacobianA.linear.magnitudeSquared() +
      jacobianA.angular.magnitudeSquared());
  const physx::PxReal jacobianMagnitudeB = physx::PxSqrt(
      jacobianB.linear.magnitudeSquared() +
      jacobianB.angular.magnitudeSquared());
  if (!(jacobianMagnitudeA > 1e-6f) ||
      !(jacobianMagnitudeB > 1e-6f))
    return false;
  const AvbdVec6 directionA(
      jacobianA.linear / jacobianMagnitudeA,
      jacobianA.angular / jacobianMagnitudeA);
  const AvbdVec6 directionB(
      jacobianB.linear / jacobianMagnitudeB,
      jacobianB.angular / jacobianMagnitudeB);

  auto computeInertialTerm =
      [invDt2](const AvbdSolverBody &body,
               const AvbdVec6 &direction,
               physx::PxReal &hessian,
               physx::PxReal &gradient) -> bool {
    if (body.invMass <= 0.0f)
      return false;
    const physx::PxReal mass = 1.0f / body.invMass;
    hessian =
        mass * direction.linear.magnitudeSquared() * invDt2;
    gradient =
        mass * (body.position - body.inertialPosition)
                   .dot(direction.linear) *
        invDt2;
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxMat33 inertia = body.invInertiaWorld.getInverse();
    const physx::PxReal angularHessian =
        direction.angular.dot(inertia * direction.angular) *
        invDt2;
    hessian += angularHessian;
    gradient += angularHessian *
                rotationError.dot(direction.angular);
    return hessian > 1e-8f && PxIsFinite(hessian) &&
           PxIsFinite(gradient);
  };

  physx::PxReal inertialHessianA = 0.0f;
  physx::PxReal inertialHessianB = 0.0f;
  physx::PxReal gradientA = 0.0f;
  physx::PxReal gradientB = 0.0f;
  if (!computeInertialTerm(endpointA, directionA, inertialHessianA,
                           gradientA) ||
      !computeInertialTerm(endpointB, directionB, inertialHessianB,
                           gradientB))
    return false;

  const physx::PxReal violation =
      computeGeneric1DViolation(joint, bodies, numBodies, dt);
  const physx::PxReal velocity =
      (violation - joint.genericGeometricError) / dt;
  physx::PxReal penalty =
      joint.header.rho + joint.header.damping / dt;
  physx::PxReal unclampedForce =
      joint.header.rho * violation + joint.header.damping * velocity;
  if (joint.genericTendonLimitStiffness > 0.0f) {
    physx::PxReal limitViolation = 0.0f;
    if (violation < joint.genericTendonLowLimit)
      limitViolation = violation - joint.genericTendonLowLimit;
    else if (violation > joint.genericTendonHighLimit)
      limitViolation = violation - joint.genericTendonHighLimit;
    if (limitViolation != 0.0f) {
      penalty +=
          joint.genericTendonLimitStiffness +
          joint.header.damping / dt;
      unclampedForce +=
          joint.genericTendonLimitStiffness * limitViolation;
    }
  }
  const physx::PxReal appliedImpulse = physx::PxClamp(
      -unclampedForce * dt, joint.genericMinImpulse,
      joint.genericMaxImpulse);
  const physx::PxReal force = -appliedImpulse / dt;
  gradientA += jacobianMagnitudeA * force;
  gradientB += jacobianMagnitudeB * force;

  const physx::PxReal hessianAA =
      inertialHessianA +
      penalty * jacobianMagnitudeA * jacobianMagnitudeA;
  const physx::PxReal hessianBB =
      inertialHessianB +
      penalty * jacobianMagnitudeB * jacobianMagnitudeB;
  const physx::PxReal hessianAB =
      penalty * jacobianMagnitudeA * jacobianMagnitudeB;
  const physx::PxReal determinant =
      hessianAA * hessianBB - hessianAB * hessianAB;
  if (!(determinant > 1e-12f) || !PxIsFinite(determinant))
    return false;
  const physx::PxReal solutionA =
      (gradientA * hessianBB - gradientB * hessianAB) / determinant;
  const physx::PxReal solutionB =
      (hessianAA * gradientB - hessianAB * gradientA) / determinant;
  if (!PxIsFinite(solutionA) || !PxIsFinite(solutionB))
    return false;

  endpointA.position -= directionA.linear * solutionA;
  endpointB.position -= directionB.linear * solutionB;
  const physx::PxVec3 angularSolutionA =
      directionA.angular * solutionA;
  const physx::PxVec3 angularSolutionB =
      directionB.angular * solutionB;
  if (angularSolutionA.magnitudeSquared() > 1e-12f) {
    const physx::PxQuat delta(angularSolutionA.x, angularSolutionA.y,
                              angularSolutionA.z, 0.0f);
    endpointA.rotation =
        (endpointA.rotation - delta * endpointA.rotation * 0.5f)
            .getNormalized();
  }
  if (angularSolutionB.magnitudeSquared() > 1e-12f) {
    const physx::PxQuat delta(angularSolutionB.x, angularSolutionB.y,
                              angularSolutionB.z, 0.0f);
    endpointB.rotation =
        (endpointB.rotation - delta * endpointB.rotation * 0.5f)
            .getNormalized();
  }
  return true;
}

static bool isCoupledLinearDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || !d6Joints || numD6 != 1 ||
      (numContacts != 0 && !contacts) || numGear != 0 ||
      numSoftParticles != 0 || numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  if (joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexA].lockFlags != 0 ||
      bodies[joint.header.bodyIndexB].lockFlags != 0 ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      joint.driveFlags != 0x1u ||
      (joint.driveAccelerationFlags != 0 &&
       joint.driveAccelerationFlags != 0x1u) ||
      joint.linearStiffness.x != 0.0f || joint.linearDamping.x <= 0.0f ||
      joint.getLinearMotion(0) != 2 || joint.getLinearMotion(1) != 2 ||
      joint.getLinearMotion(2) != 2 || joint.getAngularMotion(0) != 2 ||
      joint.getAngularMotion(1) != 2 || joint.getAngularMotion(2) != 2 ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0)
    return false;

  return areFrictionlessBodyVsStaticContactsSupported(
      contacts, numContacts, numBodies);
}

static bool isLinearPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      joint.driveFlags != 0x1u || joint.driveAccelerationFlags != 0 ||
      joint.linearStiffness.x <= 0.0f || joint.linearDamping.x <= 0.0f ||
      joint.getLinearMotion(0) != 2 || joint.getLinearMotion(1) != 0 ||
      joint.getLinearMotion(2) != 0 || joint.angularMotion != 0 ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveLinearPosition.isFinite() ||
      !joint.driveLinearVelocity.isFinite() ||
      !(joint.driveLinearForce.x > 0.0f) ||
      !PxIsFinite(joint.driveLinearForce.x))
    return false;
  return true;
}

static bool isIdentityJointFrame(const physx::PxQuat &frame) {
  if (!frame.isFinite())
    return false;
  const physx::PxReal magnitudeSquared = frame.magnitudeSquared();
  if (!(magnitudeSquared > 1e-8f) || !PxIsFinite(magnitudeSquared))
    return false;
  const physx::PxReal invMagnitude =
      1.0f / physx::PxSqrt(magnitudeSquared);
  return physx::PxAbs(frame.x * invMagnitude) <= 1e-6f &&
         physx::PxAbs(frame.y * invMagnitude) <= 1e-6f &&
         physx::PxAbs(frame.z * invMagnitude) <= 1e-6f &&
         physx::PxAbs(physx::PxAbs(frame.w * invMagnitude) - 1.0f) <=
             1e-6f;
}

static bool isStrictAsymmetricJointXZPositionOwner(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint,
    bool frictionlessContacts) {
  const physx::PxU32 bodyA = joint.header.bodyIndexA;
  const physx::PxU32 bodyB = joint.header.bodyIndexB;
  if (!frictionlessContacts ||
      bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
      physx::PxAbs(bodies[bodyA].invMass -
                    bodies[bodyB].invMass) >
          1e-6f * physx::PxMax(bodies[bodyA].invMass,
                              bodies[bodyB].invMass) ||
      !isIdentityJointFrame(joint.localFrameA) ||
      !isIdentityJointFrame(joint.localFrameB) ||
      physx::PxAbs(joint.driveLinearVelocity.x) > 1e-6f ||
      joint.driveLinearForce.x != PX_MAX_F32)
    return false;

  const bool centeredA =
      joint.anchorA.magnitudeSquared() <= 1e-12f;
  const bool centeredB =
      joint.anchorB.magnitudeSquared() <= 1e-12f;
  const bool pureXA =
      physx::PxAbs(joint.anchorA.x) > 1e-6f &&
      physx::PxAbs(joint.anchorA.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorA.z) <= 1e-6f;
  const bool pureXB =
      physx::PxAbs(joint.anchorB.x) > 1e-6f &&
      physx::PxAbs(joint.anchorB.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorB.z) <= 1e-6f;
  const bool pureZA =
      physx::PxAbs(joint.anchorA.x) <= 1e-6f &&
      physx::PxAbs(joint.anchorA.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorA.z) > 1e-6f;
  const bool pureZB =
      physx::PxAbs(joint.anchorB.x) <= 1e-6f &&
      physx::PxAbs(joint.anchorB.y) <= 1e-6f &&
      physx::PxAbs(joint.anchorB.z) > 1e-6f;
  const bool unequalPureZPair =
      pureZA && pureZB &&
      (joint.anchorA - joint.anchorB).magnitudeSquared() > 1e-12f;
  return (centeredA && (pureXB || pureZB)) ||
         ((pureXA || pureZA) && centeredB) ||
         unequalPureZPair;
}

static bool isCoupledLinearPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const physx::PxVec3 &gravity, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || !contacts || numContacts == 0 ||
      !d6Joints || numD6 != 1 || numGear != 0 ||
      numSoftParticles != 0 || numSoftBodies != 0 ||
      numSoftContacts != 0)
    return false;
  const bool frictionlessContacts =
      areTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts);
  const bool frictionalContacts =
      areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts, gravity);
  if (!frictionlessContacts && !frictionalContacts)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool equalAnchors =
      (joint.anchorA - joint.anchorB).magnitudeSquared() <= 1e-12f;
  const bool strictAsymmetricJointXZ =
      isStrictAsymmetricJointXZPositionOwner(
          bodies, numBodies, joint, frictionlessContacts);
  // eOUTPUT_FORCE is observational only. The complete position owner remains
  // unchanged, and post-finalization writeback controls whether its physical
  // drive force contributes to the public reaction.
  if (joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexA].lockFlags != 0 ||
      bodies[joint.header.bodyIndexB].lockFlags != 0 ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexA].angularDampingBody != 0.0f ||
      bodies[joint.header.bodyIndexB].angularDampingBody != 0.0f ||
      joint.driveFlags != 0x1u ||
      joint.driveAccelerationFlags != 0 ||
      joint.linearStiffness.x <= 0.0f ||
      joint.linearDamping.x <= 0.0f ||
      joint.linearStiffness.y != 0.0f ||
      joint.linearStiffness.z != 0.0f ||
      joint.linearDamping.y != 0.0f ||
      joint.linearDamping.z != 0.0f ||
      joint.getLinearMotion(0) != 2 ||
      joint.getLinearMotion(1) != 0 ||
      joint.getLinearMotion(2) != 0 ||
      joint.angularMotion != 0 ||
      !joint.anchorA.isFinite() || !joint.anchorB.isFinite() ||
      (!equalAnchors && !strictAsymmetricJointXZ) ||
      joint.motorEnabled != 0 || joint.coneAngleLimit > 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveLinearPosition.isFinite() ||
      !joint.driveLinearVelocity.isFinite() ||
      physx::PxAbs(joint.driveLinearPosition.y) > 1e-6f ||
      physx::PxAbs(joint.driveLinearPosition.z) > 1e-6f ||
      physx::PxAbs(joint.driveLinearVelocity.y) > 1e-6f ||
      physx::PxAbs(joint.driveLinearVelocity.z) > 1e-6f ||
      !(joint.driveLinearForce.x > 0.0f) ||
      !PxIsFinite(joint.driveLinearForce.x))
    return false;

  physx::PxQuat frameA = joint.localFrameA;
  const physx::PxReal frameMagnitude = frameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  frameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  const physx::PxVec3 axis =
      (bodies[joint.header.bodyIndexA].rotation * frameA)
          .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  bool bodyHasContact[2] = {false, false};
  physx::PxVec3 supportNormal(0.0f);
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies ||
        physx::PxAbs(contact.contactNormal.dot(axis)) > 1e-5f)
      return false;
    bodyHasContact[bodyIndex] = true;
    physx::PxVec3 normal = contact.contactNormal;
    const physx::PxReal normalLength = normal.magnitude();
    if (!(normalLength > 1e-6f) || !PxIsFinite(normalLength))
      return false;
    normal *= 1.0f / normalLength;
    if (i == 0)
      supportNormal = normal;
    else if (physx::PxAbs(normal.dot(supportNormal)) < 0.9999f)
      return false;
  }
  return bodyHasContact[0] && bodyHasContact[1];
}

static bool isAngularAxisVelocityDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  const physx::PxU32 driveIndex = joint.driveFlags == (1u << 3)
                                         ? 0u
                                         : (joint.driveFlags == (1u << 4)
                                                ? 1u
                                                : (joint.driveFlags == (1u << 5)
                                                       ? 2u
                                                       : PX_MAX_U32));
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      driveIndex == PX_MAX_U32 ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0)
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.angularStiffness[driveIndex] != 0.0f ||
      joint.angularDamping[driveIndex] <= 0.0f ||
      joint.linearMotion != 0x2au || joint.angularMotion != 0x2au ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularVelocity.isFinite() ||
      !(joint.driveAngularForce[driveIndex] > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce[driveIndex]))
    return false;
  return true;
}

static bool isAngularAxisPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  const physx::PxU32 driveIndex =
      joint.driveFlags == (1u << 3)
          ? 0u
          : (joint.driveFlags == (1u << 4)
                 ? 1u
                 : (joint.driveFlags == (1u << 5) ? 2u : PX_MAX_U32));
  const physx::PxReal targetMagnitude =
      joint.driveAngularPosition.magnitudeSquared();
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      driveIndex == PX_MAX_U32 ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0)
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.driveOutputForceFlags != 0 || joint.linearMotion != 0 ||
      joint.getAngularMotion(driveIndex) != 2 ||
      joint.getAngularMotion((driveIndex + 1) % 3) != 0 ||
      joint.getAngularMotion((driveIndex + 2) % 3) != 0 ||
      joint.angularStiffness[driveIndex] <= 0.0f ||
      joint.angularDamping[driveIndex] <= 0.0f ||
      joint.angularStiffness[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularStiffness[(driveIndex + 2) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 2) % 3] != 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularPosition.isFinite() ||
      !joint.driveAngularVelocity.isFinite() ||
      physx::PxAbs(targetMagnitude - 1.0f) > 1e-4f ||
      (driveIndex != 0 &&
       physx::PxAbs(joint.driveAngularPosition.x) > 1e-6f) ||
      (driveIndex != 1 &&
       physx::PxAbs(joint.driveAngularPosition.y) > 1e-6f) ||
      (driveIndex != 2 &&
       physx::PxAbs(joint.driveAngularPosition.z) > 1e-6f) ||
      joint.driveAngularVelocity.magnitudeSquared() > 1e-12f ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      !(joint.driveAngularForce[driveIndex] > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce[driveIndex]))
    return false;
  return true;
}

static bool isSlerpVelocityDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) == 0 ||
      joint.driveFlags != (1u << 5))
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.angularStiffness.z != 0.0f ||
      joint.angularDamping.z <= 0.0f || joint.linearMotion != 0x2au ||
      joint.angularMotion != 0x2au ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularVelocity.isFinite() ||
      !(joint.driveAngularForce.z > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce.z))
    return false;
  return true;
}

static bool isSlerpPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 1 || numContacts != 0 || !d6Joints ||
      numD6 != 1 || numGear != 0 || numSoftParticles != 0 ||
      numSoftBodies != 0 || numSoftContacts != 0)
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  const bool dynamicA = joint.header.bodyIndexA < numBodies;
  const bool dynamicB = joint.header.bodyIndexB < numBodies;
  const physx::PxU32 dynamicIndex =
      dynamicA ? joint.header.bodyIndexA : joint.header.bodyIndexB;
  const physx::PxReal targetMagnitude =
      joint.driveAngularPosition.magnitudeSquared();
  if (dynamicA == dynamicB || dynamicIndex >= numBodies ||
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) == 0 ||
      joint.driveFlags != (1u << 5))
    return false;
  if (bodies[dynamicIndex].invMass <= 0.0f ||
      bodies[dynamicIndex].linearDamping != 0.0f ||
      bodies[dynamicIndex].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.driveOutputForceFlags != 0 || joint.linearMotion != 0 ||
      joint.angularMotion != 0x2au ||
      joint.angularStiffness.x != 0.0f ||
      joint.angularStiffness.y != 0.0f ||
      joint.angularStiffness.z <= 0.0f ||
      joint.angularDamping.x != 0.0f ||
      joint.angularDamping.y != 0.0f ||
      joint.angularDamping.z <= 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularPosition.isFinite() ||
      !joint.driveAngularVelocity.isFinite() ||
      physx::PxAbs(targetMagnitude - 1.0f) > 1e-4f ||
      joint.driveAngularVelocity.magnitudeSquared() > 1e-12f ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      !(joint.driveAngularForce.z > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce.z))
    return false;
  return true;
}

static bool isCoupledAngularPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  if (!bodies || numBodies != 2 || !d6Joints || numD6 != 1 ||
      numGear != 0 || numSoftParticles != 0 || numSoftBodies != 0 ||
      numSoftContacts != 0 ||
      !areTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts))
    return false;

  const AvbdD6JointConstraint &joint = d6Joints[0];
  if (joint.header.bodyIndexA >= numBodies ||
      joint.header.bodyIndexB >= numBodies ||
      joint.header.bodyIndexA == joint.header.bodyIndexB ||
      bodies[joint.header.bodyIndexA].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexB].invMass <= 0.0f ||
      bodies[joint.header.bodyIndexA].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexB].linearDamping != 0.0f ||
      bodies[joint.header.bodyIndexA].angularDampingBody != 0.0f ||
      bodies[joint.header.bodyIndexB].angularDampingBody != 0.0f ||
      joint.driveAccelerationFlags != 0 ||
      joint.driveOutputForceFlags != 0 || joint.linearMotion != 0 ||
      joint.anchorA.magnitudeSquared() > 1e-12f ||
      joint.anchorB.magnitudeSquared() > 1e-12f ||
      joint.motorEnabled != 0 || joint.coneAngleLimit > 0.0f ||
      (joint.sourceFlags &
       AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES) == 0 ||
      !joint.driveAngularPosition.isFinite() ||
      !joint.driveAngularVelocity.isFinite() ||
      physx::PxAbs(joint.driveAngularPosition.magnitudeSquared() - 1.0f) >
          1e-4f ||
      joint.driveAngularVelocity.magnitudeSquared() > 1e-12f)
    return false;

  const bool slerp =
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0;
  if (slerp) {
    return joint.driveFlags == (1u << 5) &&
           joint.angularMotion == 0x2au &&
           joint.angularStiffness.x == 0.0f &&
           joint.angularStiffness.y == 0.0f &&
           joint.angularStiffness.z > 0.0f &&
           joint.angularDamping.x == 0.0f &&
           joint.angularDamping.y == 0.0f &&
           joint.angularDamping.z > 0.0f &&
           joint.driveAngularForce.z > 0.0f &&
           PxIsFinite(joint.driveAngularForce.z);
  }

  const physx::PxU32 driveIndex =
      joint.driveFlags == (1u << 3)
          ? 0u
          : (joint.driveFlags == (1u << 4)
                 ? 1u
                 : (joint.driveFlags == (1u << 5) ? 2u : PX_MAX_U32));
  if (driveIndex == PX_MAX_U32 ||
      joint.getAngularMotion(driveIndex) != 2 ||
      joint.getAngularMotion((driveIndex + 1) % 3) != 0 ||
      joint.getAngularMotion((driveIndex + 2) % 3) != 0 ||
      joint.angularStiffness[driveIndex] <= 0.0f ||
      joint.angularDamping[driveIndex] <= 0.0f ||
      joint.angularStiffness[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularStiffness[(driveIndex + 2) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 1) % 3] != 0.0f ||
      joint.angularDamping[(driveIndex + 2) % 3] != 0.0f ||
      !(joint.driveAngularForce[driveIndex] > 0.0f) ||
      !PxIsFinite(joint.driveAngularForce[driveIndex]))
    return false;
  if ((driveIndex != 0 &&
       physx::PxAbs(joint.driveAngularPosition.x) > 1e-6f) ||
      (driveIndex != 1 &&
       physx::PxAbs(joint.driveAngularPosition.y) > 1e-6f) ||
      (driveIndex != 2 &&
       physx::PxAbs(joint.driveAngularPosition.z) > 1e-6f))
    return false;
  return true;
}

static bool solveCoupledAngularPositionDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdSolverConfig &config) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(numContacts + 9);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  if (!addFrictionlessBodyVsStaticContactRows(
          bodies, numBodies, contacts, numContacts, config, invDt2,
          rows, gradient, preconditioner))
    return false;

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal hardPenalty =
      physx::PxMax(joint.header.rho, physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 linearViolation = bodyA.position - bodyB.position;
  const physx::PxVec3 worldAxes[3] = {
      physx::PxVec3(1.0f, 0.0f, 0.0f),
      physx::PxVec3(0.0f, 1.0f, 0.0f),
      physx::PxVec3(0.0f, 0.0f, 1.0f)};
  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(worldAxes[axis], physx::PxVec3(0.0f));
    row.jacobianB = AvbdVec6(-worldAxes[axis], physx::PxVec3(0.0f));
    row.penalty = hardPenalty;
    row.force = hardPenalty * linearViolation.dot(worldAxes[axis]) +
                joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  physx::PxQuat worldFrameB = bodyB.rotation * joint.localFrameB;
  const physx::PxReal frameAMagnitude = worldFrameA.magnitudeSquared();
  const physx::PxReal frameBMagnitude = worldFrameB.magnitudeSquared();
  if (!(frameAMagnitude > 1e-8f) || !(frameBMagnitude > 1e-8f) ||
      !PxIsFinite(frameAMagnitude) || !PxIsFinite(frameBMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameAMagnitude);
  worldFrameB *= 1.0f / physx::PxSqrt(frameBMagnitude);

  const bool slerp =
      (joint.sourceFlags & AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0;
  const physx::PxU32 driveIndex =
      slerp ? PX_MAX_U32
            : (joint.driveFlags == (1u << 3)
                   ? 0u
                   : (joint.driveFlags == (1u << 4) ? 1u : 2u));
  if (!slerp && driveIndex == 0) {
    const physx::PxVec3 worldTwistA =
        worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    const physx::PxVec3 worldTwistB =
        worldFrameB.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
    const physx::PxVec3 axisViolation = worldTwistA.cross(worldTwistB);
    physx::PxVec3 perp1 =
        physx::PxAbs(worldTwistA.x) < 0.9f
            ? worldTwistA.cross(physx::PxVec3(1.0f, 0.0f, 0.0f))
            : worldTwistA.cross(physx::PxVec3(0.0f, 1.0f, 0.0f));
    const physx::PxReal perp1Length = perp1.magnitude();
    if (!(perp1Length > 1e-6f) || !PxIsFinite(perp1Length))
      return false;
    perp1 *= 1.0f / perp1Length;
    physx::PxVec3 perp2 = worldTwistA.cross(perp1);
    const physx::PxReal perp2Length = perp2.magnitude();
    if (!(perp2Length > 1e-6f) || !PxIsFinite(perp2Length))
      return false;
    perp2 *= 1.0f / perp2Length;
    const physx::PxVec3 perpendicularAxes[2] = {perp1, perp2};
    for (physx::PxU32 rowIndex = 0; rowIndex < 2; ++rowIndex) {
      CoupledIslandRow row;
      row.bodyA = bodyAIndex;
      row.bodyB = bodyBIndex;
      row.jacobianA =
          AvbdVec6(physx::PxVec3(0.0f), -perpendicularAxes[rowIndex]);
      row.jacobianB =
          AvbdVec6(physx::PxVec3(0.0f), perpendicularAxes[rowIndex]);
      row.penalty = hardPenalty;
      row.force =
          hardPenalty * axisViolation.dot(perpendicularAxes[rowIndex]) +
          joint.lambdaAngular[rowIndex + 1];
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  } else if (!slerp) {
    for (physx::PxU32 axis = 0; axis < 3; ++axis) {
      if (axis == driveIndex)
        continue;
      physx::PxVec3 localAxis(0.0f);
      localAxis[axis] = 1.0f;
      const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
      CoupledIslandRow row;
      row.bodyA = bodyAIndex;
      row.bodyB = bodyBIndex;
      row.jacobianA = AvbdVec6(physx::PxVec3(0.0f), worldAxis);
      row.jacobianB = AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
      row.penalty = hardPenalty;
      row.force =
          hardPenalty *
              joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis) +
          joint.lambdaAngular[axis];
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  }

  physx::PxQuat displacementA =
      bodyA.rotation * bodyA.prevRotation.getConjugate();
  physx::PxQuat displacementB =
      bodyB.rotation * bodyB.prevRotation.getConjugate();
  if (displacementA.w < 0.0f)
    displacementA = -displacementA;
  if (displacementB.w < 0.0f)
    displacementB = -displacementB;
  const physx::PxVec3 relativeAngularDisplacement =
      physx::PxVec3(displacementB.x - displacementA.x,
                    displacementB.y - displacementA.y,
                    displacementB.z - displacementA.z) *
      2.0f;

  physx::PxQuat currentRelative =
      worldFrameA.getConjugate() * worldFrameB;
  currentRelative.normalize();
  physx::PxQuat targetRelative = joint.driveAngularPosition;
  if (currentRelative.dot(targetRelative) < 0.0f)
    targetRelative = -targetRelative;
  if (slerp) {
    const physx::PxQuat delta =
        targetRelative.getConjugate() * currentRelative;
    physx::PxVec3 driveAxes[3];
    computeSlerpJacobianAxes(driveAxes, worldFrameA * targetRelative,
                             worldFrameB);
    const physx::PxReal stiffness = joint.angularStiffness.z;
    const physx::PxReal damping = joint.angularDamping.z;
    const physx::PxReal limit = joint.driveAngularForce.z;
    for (physx::PxU32 rowIndex = 0; rowIndex < 3; ++rowIndex) {
      const physx::PxReal velocityError =
          relativeAngularDisplacement.dot(driveAxes[rowIndex]);
      const physx::PxReal rawTorque =
          stiffness * (&delta.x)[rowIndex] +
          (damping / dt) * velocityError;
      const physx::PxReal driveTorque =
          physx::PxClamp(rawTorque, -limit, limit);
      CoupledIslandRow row;
      row.bodyA = bodyAIndex;
      row.bodyB = bodyBIndex;
      row.jacobianA =
          AvbdVec6(physx::PxVec3(0.0f), -driveAxes[rowIndex]);
      row.jacobianB =
          AvbdVec6(physx::PxVec3(0.0f), driveAxes[rowIndex]);
      row.penalty =
          physx::PxAbs(rawTorque) >= limit ? 0.0f
                                           : stiffness + damping / dt;
      row.force = driveTorque;
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  } else {
    physx::PxVec3 localAxis(0.0f);
    localAxis[driveIndex] = 1.0f;
    const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
    const physx::PxQuat delta =
        currentRelative * targetRelative.getConjugate();
    physx::PxReal positionResidual = 0.0f;
    physx::PxReal positionTangent = 0.0f;
    if (driveIndex == 0) {
      positionResidual = 2.0f * delta.x;
      positionTangent = physx::PxAbs(delta.w);
    } else if (driveIndex == 1) {
      positionResidual = -delta.getBasisVector0().z;
      positionTangent =
          physx::PxAbs(1.0f - 2.0f * delta.y * delta.y);
    } else {
      positionResidual = delta.getBasisVector0().y;
      positionTangent =
          physx::PxAbs(1.0f - 2.0f * delta.z * delta.z);
    }
    const physx::PxReal stiffness = joint.angularStiffness[driveIndex];
    const physx::PxReal damping = joint.angularDamping[driveIndex];
    const physx::PxReal limit = joint.driveAngularForce[driveIndex];
    const physx::PxReal velocityError =
        relativeAngularDisplacement.dot(worldAxis);
    const physx::PxReal rawTorque =
        stiffness * positionResidual + (damping / dt) * velocityError;
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA = AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
    row.jacobianB = AvbdVec6(physx::PxVec3(0.0f), worldAxis);
    row.penalty =
        physx::PxAbs(rawTorque) >= limit
            ? 0.0f
            : stiffness * positionTangent + damping / dt;
    row.force = physx::PxClamp(rawTorque, -limit, limit);
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) || !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual = 1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction, operatorDirection);
    const double denominator = dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) || !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) || !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] = preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                               solution[i].angular.y,
                               solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation - delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
}

static bool solveCoupledLinearPositionDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const physx::PxVec3 &gravity,
    const AvbdSolverConfig &config) {
  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  AvbdSolverBody &bodyA = bodies[bodyAIndex];
  AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(numContacts * 3u + 6u);

  physx::PxQuat worldFrameA = bodyA.rotation * joint.localFrameA;
  physx::PxQuat worldFrameB = bodyB.rotation * joint.localFrameB;
  const physx::PxReal frameAMagnitude = worldFrameA.magnitudeSquared();
  const physx::PxReal frameBMagnitude = worldFrameB.magnitudeSquared();
  if (!(frameAMagnitude > 1e-8f) || !(frameBMagnitude > 1e-8f) ||
      !PxIsFinite(frameAMagnitude) || !PxIsFinite(frameBMagnitude))
    return false;
  worldFrameA *= 1.0f / physx::PxSqrt(frameAMagnitude);
  worldFrameB *= 1.0f / physx::PxSqrt(frameBMagnitude);
  const physx::PxVec3 linearAxes[3] = {
      worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f)),
      worldFrameA.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f)),
      worldFrameA.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f))};
  const physx::PxVec3 driveAxis = linearAxes[0];
  physx::PxVec3 supportNormal(0.0f);
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    if (physx::PxAbs(contacts[i].contactNormal.dot(driveAxis)) > 1e-5f)
      return false;
    physx::PxVec3 normal = contacts[i].contactNormal;
    const physx::PxReal normalLength = normal.magnitude();
    if (!(normalLength > 1e-6f) || !PxIsFinite(normalLength))
      return false;
    normal *= 1.0f / normalLength;
    if (i == 0)
      supportNormal = normal;
    else if (physx::PxAbs(normal.dot(supportNormal)) < 0.9999f)
      return false;
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  if (!addBodyVsStaticContactNormalRows(
          bodies, numBodies, contacts, numContacts, config, invDt2,
          rows, gradient, preconditioner, true))
    return false;
  const bool frictionalContacts =
      contacts[0].friction > 0.0f ||
      contacts[0].staticFriction > 0.0f;
  if (frictionalContacts &&
      !addStrictFrictionalBodyVsStaticContactPositionRows(
          bodies, numBodies, contacts, numContacts, gravity, config, invDt2,
          rows, gradient, preconditioner))
    return false;

  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal hardPenalty =
      physx::PxMax(joint.header.rho, physx::PxMax(massA, massB) * invDt2);
  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 worldAnchorA = bodyA.position + rA;
  const physx::PxVec3 worldAnchorB = bodyB.position + rB;
  const physx::PxVec3 lockedLinearViolation =
      worldAnchorA - worldAnchorB;
  for (physx::PxU32 axis = 1; axis < 3; ++axis) {
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(linearAxes[axis], rA.cross(linearAxes[axis]));
    row.jacobianB =
        AvbdVec6(-linearAxes[axis], -rB.cross(linearAxes[axis]));
    row.penalty = hardPenalty;
    row.force =
        hardPenalty * lockedLinearViolation.dot(linearAxes[axis]) +
        joint.lambdaLinear[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  for (physx::PxU32 axis = 0; axis < 3; ++axis) {
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    const physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);
    CoupledIslandRow row;
    row.bodyA = bodyAIndex;
    row.bodyB = bodyBIndex;
    row.jacobianA =
        AvbdVec6(physx::PxVec3(0.0f), worldAxis);
    row.jacobianB =
        AvbdVec6(physx::PxVec3(0.0f), -worldAxis);
    row.penalty = hardPenalty;
    row.force =
        hardPenalty *
            joint.computeAngularError(bodyA.rotation, bodyB.rotation, axis) +
        joint.lambdaAngular[axis];
    addCoupledRow(row, rows, gradient, preconditioner);
  }

  const physx::PxVec3 previousRA =
      bodyA.prevRotation.rotate(joint.anchorA);
  const physx::PxVec3 previousRB =
      bodyB.prevRotation.rotate(joint.anchorB);
  const physx::PxVec3 displacementA =
      worldAnchorA - (bodyA.prevPosition + previousRA);
  const physx::PxVec3 displacementB =
      worldAnchorB - (bodyB.prevPosition + previousRB);
  const physx::PxReal positionResidual =
      (worldAnchorB - worldAnchorA).dot(driveAxis) -
      joint.driveLinearPosition.x;
  const physx::PxReal displacementResidual =
      (displacementB - displacementA).dot(driveAxis) -
      joint.driveLinearVelocity.x * dt;
  const physx::PxReal stiffness = joint.linearStiffness.x;
  const physx::PxReal damping = joint.linearDamping.x;
  const physx::PxReal limit = joint.driveLinearForce.x;
  const physx::PxReal rawForce =
      stiffness * positionResidual +
      (damping / dt) * displacementResidual;
  CoupledIslandRow driveRow;
  driveRow.bodyA = bodyAIndex;
  driveRow.bodyB = bodyBIndex;
  driveRow.jacobianA =
      AvbdVec6(-driveAxis, -rA.cross(driveAxis));
  driveRow.jacobianB =
      AvbdVec6(driveAxis, rB.cross(driveAxis));
  driveRow.penalty =
      physx::PxAbs(rawForce) >= limit
          ? 0.0f
          : stiffness + damping / dt;
  driveRow.force = physx::PxClamp(rawForce, -limit, limit);
  addCoupledRow(driveRow, rows, gradient, preconditioner);

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(
            preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) ||
      !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual = 1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction, operatorDirection);
    const double denominator = dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) ||
        !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) ||
        !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] = preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  if (!frictionalContacts) {
    physx::PxReal totalMass = 0.0f;
    physx::PxVec3 expectedWeightedDelta(0.0f);
    physx::PxVec3 solvedWeightedDelta(0.0f);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal mass = 1.0f / bodies[i].invMass;
      totalMass += mass;
      expectedWeightedDelta +=
          (bodies[i].position - bodies[i].inertialPosition) * mass;
      solvedWeightedDelta += solution[i].linear * mass;
    }
    if (!(totalMass > 0.0f) || !PxIsFinite(totalMass))
      return false;
    physx::PxVec3 translationRoundoff =
        solvedWeightedDelta - expectedWeightedDelta;
    translationRoundoff -=
        supportNormal * translationRoundoff.dot(supportNormal);
    const physx::PxVec3 commonRoundoff =
        translationRoundoff / totalMass;
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      solution[i].linear -= commonRoundoff;
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                                solution[i].angular.y,
                                solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation - delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
}

static bool computeTwoBodySupportAxisAngularMomentum(
    const AvbdSolverBody &bodyA, const AvbdSolverBody &bodyB,
    const physx::PxVec3 &supportAxis,
    physx::PxReal linearScale, physx::PxReal angularScale,
    physx::PxReal &axisAngularMomentum) {
  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal totalMass = massA + massB;
  if (!(totalMass > 0.0f) || !PxIsFinite(totalMass) ||
      !PxIsFinite(linearScale) || !PxIsFinite(angularScale))
    return false;
  const physx::PxVec3 centerOfMass =
      (bodyA.position * massA + bodyB.position * massB) /
      totalMass;
  const physx::PxVec3 orbitalAngularMomentum =
      (bodyA.position - centerOfMass)
              .cross(bodyA.linearVelocity * massA) +
      (bodyB.position - centerOfMass)
              .cross(bodyB.linearVelocity * massB);
  const physx::PxMat33 inertiaA =
      bodyA.invInertiaWorld.getInverse();
  const physx::PxMat33 inertiaB =
      bodyB.invInertiaWorld.getInverse();
  const physx::PxVec3 spinAngularMomentum =
      inertiaA.transform(bodyA.angularVelocity) +
      inertiaB.transform(bodyB.angularVelocity);
  axisAngularMomentum =
      supportAxis.dot(orbitalAngularMomentum * linearScale +
                      spinAngularMomentum * angularScale);
  return PxIsFinite(axisAngularMomentum);
}

static bool restoreTwoBodySupportAxisAngularMomentum(
    AvbdSolverBody &bodyA, AvbdSolverBody &bodyB,
    const physx::PxVec3 &supportAxis,
    physx::PxReal expectedAxisAngularMomentum) {
  physx::PxReal currentAxisAngularMomentum = 0.0f;
  if (!computeTwoBodySupportAxisAngularMomentum(
          bodyA, bodyB, supportAxis, 1.0f, 1.0f,
          currentAxisAngularMomentum))
    return false;
  const physx::PxReal massA = 1.0f / bodyA.invMass;
  const physx::PxReal massB = 1.0f / bodyB.invMass;
  const physx::PxReal totalMass = massA + massB;
  const physx::PxVec3 centerOfMass =
      (bodyA.position * massA + bodyB.position * massB) /
      totalMass;
  const physx::PxVec3 armA = bodyA.position - centerOfMass;
  const physx::PxVec3 armB = bodyB.position - centerOfMass;
  const physx::PxMat33 inertiaA =
      bodyA.invInertiaWorld.getInverse();
  const physx::PxMat33 inertiaB =
      bodyB.invInertiaWorld.getInverse();
  const physx::PxVec3 tangentArmA = supportAxis.cross(armA);
  const physx::PxVec3 tangentArmB = supportAxis.cross(armB);
  const physx::PxReal axisInertia =
      massA * tangentArmA.magnitudeSquared() +
      supportAxis.dot(inertiaA.transform(supportAxis)) +
      massB * tangentArmB.magnitudeSquared() +
      supportAxis.dot(inertiaB.transform(supportAxis));
  if (!(axisInertia > 1e-10f) || !PxIsFinite(axisInertia) ||
      !PxIsFinite(expectedAxisAngularMomentum))
    return false;

  const physx::PxReal angularCorrection =
      (expectedAxisAngularMomentum - currentAxisAngularMomentum) /
      axisInertia;
  const physx::PxVec3 commonAngularVelocity =
      supportAxis * angularCorrection;
  const physx::PxVec3 candidateLinearA =
      bodyA.linearVelocity + commonAngularVelocity.cross(armA);
  const physx::PxVec3 candidateLinearB =
      bodyB.linearVelocity + commonAngularVelocity.cross(armB);
  const physx::PxVec3 candidateAngularA =
      bodyA.angularVelocity + commonAngularVelocity;
  const physx::PxVec3 candidateAngularB =
      bodyB.angularVelocity + commonAngularVelocity;
  if (!candidateLinearA.isFinite() || !candidateLinearB.isFinite() ||
      !candidateAngularA.isFinite() || !candidateAngularB.isFinite() ||
      (bodyA.maxLinearVelocitySq > 0.0f &&
       candidateLinearA.magnitudeSquared() >
           bodyA.maxLinearVelocitySq) ||
      (bodyB.maxLinearVelocitySq > 0.0f &&
       candidateLinearB.magnitudeSquared() >
           bodyB.maxLinearVelocitySq) ||
      (bodyA.maxAngularVelocitySq > 0.0f &&
       candidateAngularA.magnitudeSquared() >
           bodyA.maxAngularVelocitySq) ||
      (bodyB.maxAngularVelocitySq > 0.0f &&
       candidateAngularB.magnitudeSquared() >
           bodyB.maxAngularVelocitySq))
    return false;

  bodyA.linearVelocity = candidateLinearA;
  bodyA.angularVelocity = candidateAngularA;
  bodyB.linearVelocity = candidateLinearB;
  bodyB.angularVelocity = candidateAngularB;
  return true;
}

static bool solveCoupledLinearDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt, physx::PxReal invDt2,
    const AvbdSolverConfig &config) {
  physx::PxArray<AvbdBlock6x6> inertialBlocks(numBodies);
  physx::PxArray<AvbdBlock6x6> preconditioner(numBodies);
  physx::PxArray<AvbdVec6> gradient(numBodies);
  physx::PxArray<CoupledIslandRow> rows;
  rows.reserve(numContacts + 1);

  const physx::PxU32 bodyAIndex = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIndex = joint.header.bodyIndexB;
  const AvbdSolverBody &bodyA = bodies[bodyAIndex];
  const AvbdSolverBody &bodyB = bodies[bodyBIndex];
  physx::PxQuat frameA = bodyA.rotation * joint.localFrameA;
  const physx::PxReal frameMagnitude = frameA.magnitudeSquared();
  if (!(frameMagnitude > 1e-8f) || !PxIsFinite(frameMagnitude))
    return false;
  frameA *= 1.0f / physx::PxSqrt(frameMagnitude);
  const physx::PxVec3 axis =
      frameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
  physx::PxVec3 supportNormal(0.0f);
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    if (physx::PxAbs(contacts[i].contactNormal.dot(axis)) > 1e-5f)
      return false;
    if (i == 0) {
      supportNormal = contacts[i].contactNormal;
      supportNormal.normalize();
    } else if (physx::PxAbs(
                   contacts[i].contactNormal.dot(supportNormal)) < 0.9999f) {
      return false;
    }
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    inertialBlocks[i].initializeDiagonal(body.invMass, body.invInertiaWorld,
                                         invDt2);
    preconditioner[i] = inertialBlocks[i];
    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxVec3 linear =
        (body.position - body.inertialPosition) * (mass * invDt2);
    physx::PxQuat deltaQ =
        body.rotation * body.inertialRotation.getConjugate();
    if (deltaQ.w < 0.0f)
      deltaQ = -deltaQ;
    const physx::PxVec3 rotationError(deltaQ.x * 2.0f, deltaQ.y * 2.0f,
                                      deltaQ.z * 2.0f);
    const physx::PxVec3 angular =
        (body.invInertiaWorld.getInverse() * rotationError) * invDt2;
    gradient[i] = AvbdVec6(linear, angular);
  }

  if (!addFrictionlessBodyVsStaticContactRows(
          bodies, numBodies, contacts, numContacts, config, invDt2,
          rows, gradient, preconditioner))
    return false;

  const physx::PxVec3 rA = bodyA.rotation.rotate(joint.anchorA);
  const physx::PxVec3 rB = bodyB.rotation.rotate(joint.anchorB);
  const physx::PxVec3 previousRA =
      bodyA.prevRotation.rotate(joint.anchorA);
  const physx::PxVec3 previousRB =
      bodyB.prevRotation.rotate(joint.anchorB);
  const physx::PxVec3 displacementA =
      (bodyA.position + rA) - (bodyA.prevPosition + previousRA);
  const physx::PxVec3 displacementB =
      (bodyB.position + rB) - (bodyB.prevPosition + previousRB);
  const physx::PxReal violation =
      (displacementB - displacementA).dot(axis) -
      joint.driveLinearVelocity.x * dt;
  physx::PxReal penalty = joint.linearDamping.x / dt;
  if (joint.driveAccelerationFlags == 0x1u) {
    // PhysX acceleration drives scale the force-mode spring coefficient by
    // the reciprocal unit response of the complete row.  This preserves the
    // implicit damping response across endpoint masses while retaining the
    // authored force-valued limit.  The island Hessian supplies the
    // 1/(1 + dt*damping) denominator; applying it here again would count the
    // implicit response twice.
    const physx::PxVec3 angularA = rA.cross(axis);
    const physx::PxVec3 angularB = rB.cross(axis);
    const physx::PxReal unitResponse =
        bodyA.invMass + bodyB.invMass +
        angularA.dot(bodyA.invInertiaWorld * angularA) +
        angularB.dot(bodyB.invInertiaWorld * angularB);
    if (!(unitResponse > 1e-8f) || !PxIsFinite(unitResponse))
      return false;
    penalty /= unitResponse;
  }
  const physx::PxReal limit = joint.driveLinearForce.x;
  const physx::PxReal rawForce = penalty * violation;
  const physx::PxReal force =
      physx::PxClamp(rawForce, -limit, limit);
  const bool saturated = physx::PxAbs(rawForce) >= limit;
  CoupledIslandRow driveRow;
  driveRow.bodyA = bodyAIndex;
  driveRow.bodyB = bodyBIndex;
  driveRow.jacobianA = AvbdVec6(-axis, -rA.cross(axis));
  driveRow.jacobianB = AvbdVec6(axis, rB.cross(axis));
  driveRow.penalty = saturated ? 0.0f : penalty;
  driveRow.force = force;
  addCoupledRow(driveRow, rows, gradient, preconditioner);

  physx::PxArray<AvbdLDLT> preconditionerLdlt(numBodies);
  physx::PxArray<AvbdVec6> residual = gradient;
  physx::PxArray<AvbdVec6> preconditioned(numBodies);
  physx::PxArray<AvbdVec6> direction(numBodies);
  physx::PxArray<AvbdVec6> operatorDirection;
  physx::PxArray<AvbdVec6> solution(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!preconditionerLdlt[i].decomposeWithRegularization(preconditioner[i]))
      return false;
    preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    direction[i] = preconditioned[i];
    solution[i] = AvbdVec6();
  }
  double residualProduct = dotVectors(residual, preconditioned);
  if (!(residualProduct >= 0.0) || !std::isfinite(residualProduct))
    return false;
  const double initialResidual = std::sqrt(residualProduct);
  const double targetResidual =
      1e-8 * std::max(1.0, initialResidual);
  bool converged = initialResidual <= targetResidual;
  const physx::PxU32 maxIterations = numBodies * 12u;
  for (physx::PxU32 iteration = 0;
       iteration < maxIterations && !converged; ++iteration) {
    applyCoupledOperator(inertialBlocks, rows, direction, operatorDirection);
    const double denominator = dotVectors(direction, operatorDirection);
    if (!(denominator > 1e-30) || !std::isfinite(denominator))
      return false;
    const physx::PxReal alpha =
        static_cast<physx::PxReal>(residualProduct / denominator);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      addScaled(solution[i], direction[i], alpha);
      addScaled(residual[i], operatorDirection[i], -alpha);
      preconditioned[i] = preconditionerLdlt[i].solve(residual[i]);
    }
    const double nextProduct = dotVectors(residual, preconditioned);
    if (!(nextProduct >= 0.0) || !std::isfinite(nextProduct))
      return false;
    converged = physx::PxSqrt(nextProduct) <= targetResidual;
    if (!converged) {
      if (!(residualProduct > 1e-30))
        return false;
      const physx::PxReal beta =
          static_cast<physx::PxReal>(nextProduct / residualProduct);
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        direction[i] = preconditioned[i] + direction[i] * beta;
    }
    residualProduct = nextProduct;
  }
  if (!converged)
    return false;

  // The drive is translation invariant and every accepted external contact
  // normal is orthogonal to its axis. Project only the accumulated roundoff
  // in the mass-weighted translational Newton step (or its support-tangent
  // subspace); this is the exact summed linear equation, not an additional
  // physical constraint.
  physx::PxReal totalMass = 0.0f;
  physx::PxVec3 expectedWeightedDelta(0.0f);
  physx::PxVec3 solvedWeightedDelta(0.0f);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    const physx::PxReal mass = 1.0f / bodies[i].invMass;
    totalMass += mass;
    expectedWeightedDelta +=
        (bodies[i].position - bodies[i].inertialPosition) * mass;
    solvedWeightedDelta += solution[i].linear * mass;
  }
  if (!(totalMass > 0.0f) || !PxIsFinite(totalMass))
    return false;
  physx::PxVec3 translationRoundoff =
      solvedWeightedDelta - expectedWeightedDelta;
  if (numContacts > 0)
    translationRoundoff -=
        supportNormal * translationRoundoff.dot(supportNormal);
  const physx::PxVec3 commonRoundoff = translationRoundoff / totalMass;
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    solution[i].linear -= commonRoundoff;

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].position -= solution[i].linear;
    if (solution[i].angular.magnitudeSquared() > 1e-12f) {
      const physx::PxQuat delta(solution[i].angular.x,
                               solution[i].angular.y,
                               solution[i].angular.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation - delta * bodies[i].rotation * 0.5f)
              .getNormalized();
    }
  }
  return true;
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
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const physx::PxU32 *rigidTargetContactStarts,
    const physx::PxU32 *rigidTargetContactRefs,
    const AvbdOgcPairTrustRegionContext *ogcPairContext) {

  if (body.invMass <= 0.0f)
    return;

  PX_UNUSED(softBodies);
  PX_UNUSED(numSoftBodies);
  PX_UNUSED(ogcPairContext);

  const physx::PxU32 bodyIndex = body.nodeIndex;

  const physx::PxReal scaledInvMass = body.invMass;
  const physx::PxMat33 scaledInvInertia = body.invInertiaWorld;

  // =========================================================================
  // Step 1: Initialize LHS with mass matrix M/h^2
  // =========================================================================
  AvbdBlock6x6 A;
  A.initializeDiagonal(scaledInvMass, scaledInvInertia, invDt2);

  // =========================================================================
  // Step 2: Initialize RHS with inertia term
  // =========================================================================
  physx::PxReal mass =
      (scaledInvMass > 1e-8f) ? (1.0f / scaledInvMass) : 0.0f;
  physx::PxReal massInvDt2 = mass * invDt2;

  physx::PxVec3 gLinear = (body.position - body.inertialPosition) * massInvDt2;

  physx::PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  physx::PxMat33 inertiaTensor = scaledInvInertia.getInverse();
  physx::PxVec3 gAngular = (inertiaTensor * rotError) * invDt2;

  physx::PxU32 numTouching = 0;
  bool hasLinearCoupling =
      false; // Force 6x6 solve for bodies touching joints with pos-rot coupling

  // =========================================================================
  // Step 3a/3b: Shared contact + shell primal (same body-static contract as
  // solveLocalSystem). Joint / gear / soft rows accumulate after this.
  // =========================================================================
  accumulateBodyContactRows(
      body, bodyIndex, bodies, numBodies, contacts, numContacts, contactMap,
      softParticles, numSoftParticles, softContacts, numSoftContacts,
      dt, massInvDt2, A, gLinear, gAngular, numTouching,
      rigidTargetContactStarts, rigidTargetContactRefs);

  // Rigid-vertex attachments are not accumulated into this one-body block.
  // Their compiled owner is the coupled rigid+particle positional block run
  // after the ordinary rigid and soft local objectives.

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
      const physx::PxU32 compiledSourceRows =
          getAvbdJointObjectiveSourceRows(jnt.objectiveProgram);
      const physx::PxU32 compiledDriveRows =
          compiledSourceRows & 0x3fu;
      const bool compiledConeObjective =
          (compiledSourceRows &
           eJOINT_SOURCE_ANGULAR_CONE) != 0;

      const bool genericHard1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::GenericHard1D) ||
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::ArticulationHardMimic);
      const bool genericAccelerationDamping1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::
                  GenericAccelerationDamping1D);
      const bool genericForceSpring1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::GenericForceSpring1D);
      const bool compliantMimic1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::
                  ArticulationCompliantMimic);
      const bool fixedTendon1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::ArticulationFixedTendon);
      const bool spatialTendon1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::ArticulationSpatialTendon);
      const bool coupledSpatialTendon1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::CoupledSpatialTendon);
      const bool compliantTendon1D =
          fixedTendon1D || spatialTendon1D;
      if (coupledSpatialTendon1D)
        continue;
      if (genericHard1D || genericAccelerationDamping1D ||
          genericForceSpring1D || compliantMimic1D ||
          compliantTendon1D) {
        const physx::PxReal effectiveMass =
            computeGeneric1DEffectiveMass(jnt, bodies, numBodies);
        if (effectiveMass <= 0.0f || dt <= 0.0f)
          continue;

        const physx::PxReal C =
            computeGeneric1DViolation(jnt, bodies, numBodies, dt);
        physx::PxReal pen = 0.0f;
        physx::PxReal unclampedForce = 0.0f;
        if (genericAccelerationDamping1D) {
          // For an acceleration spring, PhysX's implicit velocity update is
          // mass independent: dv = -v*(d*dt)/(1+d*dt).  Expressing that in
          // AVBD's position Hessian requires scaling the physical damping by
          // the row effective mass.
          pen = effectiveMass * jnt.header.damping / dt;
          unclampedForce = pen * C;
        } else if (genericForceSpring1D || compliantMimic1D ||
                   compliantTendon1D) {
          // Backward-Euler linearization of k*C + d*Cdot with respect to
          // the current position iterate.  genericGeometricError is C at the
          // start-of-step reference pose, so this uses the displacement
          // produced by this step rather than mixing the previous interval's
          // body velocity with a current-position damping Hessian.  A changed
          // public tendon offset is deliberately part of both values: the
          // reduced-coordinate tendon path damps joint motion, not an
          // externally authored offset velocity.
          physx::PxReal stiffness = jnt.header.rho;
          physx::PxReal damping = jnt.header.damping;
          if (compliantMimic1D) {
            stiffness = jnt.genericNaturalFrequency *
                        jnt.genericNaturalFrequency * effectiveMass;
            damping = 2.0f * jnt.genericNaturalFrequency *
                      jnt.genericDampingRatio * effectiveMass;
          }
          const physx::PxReal velocity =
              (C - jnt.genericGeometricError) / dt;
          pen = stiffness + damping / dt;
          unclampedForce = stiffness * C + damping * velocity;
          if (compliantTendon1D &&
              jnt.genericTendonLimitStiffness > 0.0f) {
            physx::PxReal limitViolation = 0.0f;
            if (C < jnt.genericTendonLowLimit)
              limitViolation = C - jnt.genericTendonLowLimit;
            else if (C > jnt.genericTendonHighLimit)
              limitViolation = C - jnt.genericTendonHighLimit;
            if (limitViolation != 0.0f) {
              // Tendon limit springs are additive to the rest-length spring.
              // PhysX uses the authored damping in the limit spring's implicit
              // response, but computeTendonImpulse applies the tendon-speed
              // damping force only to the rest-length spring.
              pen +=
                  jnt.genericTendonLimitStiffness + damping / dt;
              unclampedForce +=
                  jnt.genericTendonLimitStiffness * limitViolation;
            }
          }
        } else {
          pen = physx::PxMax(jnt.header.rho, effectiveMass * invDt2);
          unclampedForce = pen * C + jnt.lambdaLinear.x;
        }
        const physx::PxReal appliedImpulse = physx::PxClamp(
            -unclampedForce * dt, jnt.genericMinImpulse,
            jnt.genericMaxImpulse);
        const bool bilateral =
            jnt.genericMinImpulse < 0.0f && jnt.genericMaxImpulse > 0.0f;
        if (!bilateral && physx::PxAbs(appliedImpulse) <= 1e-12f)
          continue;

        const physx::PxReal force = -appliedImpulse / dt;
        const physx::PxVec3 &linearJacobian =
            isBodyA ? jnt.genericLinearA : jnt.genericLinearB;
        const physx::PxVec3 &angularJacobian =
            isBodyA ? jnt.genericAngularA : jnt.genericAngularB;
        A.addConstraintContribution(linearJacobian, angularJacobian, pen);
        gLinear += linearJacobian * force;
        gAngular += angularJacobian * force;
        hasLinearCoupling |=
            linearJacobian.magnitudeSquared() > 1e-12f &&
            angularJacobian.magnitudeSquared() > 1e-12f;
        ++numTouching;
        continue;
      }

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
        const bool bodyAIsStatic =
            (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);
        physx::PxQuat rotA_lin =
            bodyAIsStatic
                ? physx::PxQuat(physx::PxIdentity)
                : (isBodyA ? body.rotation : bodies[bodyAIdx].rotation);
        physx::PxQuat jointFrameA_lin =
            bodyAIsStatic ? jnt.localFrameA : rotA_lin * jnt.localFrameA;
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
            if (compiledConeObjective && axis >= 1)
              continue;
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

      // --- Cone limit (single angular inequality) ---
      // Public D6 legacy swing limits and native spherical limits both use
      // ConeLimitHelperTanLess so unequal Y/Z limits preserve the same
      // elliptical geometry as their Extensions solver preps.
      if (compiledConeObjective) {
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

        physx::PxVec3 corrAxis(0.0f);
        physx::PxReal coneViolation = 0.0f;
        const bool ellipticalCone = computeEllipticalConeConstraint(
            jnt, rotA_cone, rotB_cone, corrAxis, coneViolation);
        if (!ellipticalCone) {
          const physx::PxVec3 worldAxisA =
              (rotA_cone * jnt.localFrameA)
                  .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          const physx::PxVec3 worldAxisB =
              (rotB_cone * jnt.localFrameB)
                  .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          const physx::PxReal dotAB = physx::PxClamp(
              worldAxisA.dot(worldAxisB), -1.0f, 1.0f);
          const physx::PxReal coneAngle = physx::PxAcos(dotAB);
          coneViolation = coneAngle - jnt.coneAngleLimit;
          corrAxis = worldAxisA.cross(worldAxisB);
          const physx::PxReal corrAxisMag = corrAxis.magnitude();
          if (corrAxisMag > 1e-6f)
            corrAxis *= 1.0f / corrAxisMag;
          else
            corrAxis = physx::PxVec3(0.0f);
        }

        // coneLambda <= 0 (unilateral): force = pen * violation - coneLambda
        physx::PxReal forceMag = pen * coneViolation - jnt.coneLambda;

        if (forceMag > 0.0f && corrAxis.magnitudeSquared() > 1e-12f) {
          physx::PxVec3 gradPos(0.0f);
          physx::PxVec3 gradRot = -corrAxis * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);
          gAngular += gradRot * forceMag;
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
            otherIsStatic && !isBodyA
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
        if ((compiledDriveRows & 0x7u) != 0) {
          for (int a = 0; a < 3; ++a) {
            if ((compiledDriveRows & (1u << a)) == 0)
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

            // Constraint: C = (dx_B - dx_A) dot axis - v_target * dt
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
            const bool isAccelerationDrive =
                jnt.isLinearAccelerationDrive(a);
            const physx::PxReal stiffness = (&jnt.linearStiffness.x)[a];
            const bool usePhysicalVelocityObjective =
                stiffness <= 0.0f && (bodyARef == nullptr || bodyBRef == nullptr);
            const bool usePositionObjective =
                a == 0 &&
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::LinearPositionDrive);
            // In the verified exactly-one-dynamic, stiffness-zero subset,
            // PxD6JointDrive::damping is a force-per-velocity coefficient.
            // With C = (dxB-dxA)-targetVelocity*dt, its position objective
            // therefore uses damping/dt.  Wider dynamic-dynamic and spring
            // families retain the legacy AL path until they have independent
            // mixed-island and articulation gates.
            const physx::PxReal positionError =
                ((bodyBRef ? bodyBRef->position + rBWorld : jnt.anchorB) -
                 (bodyARef ? bodyARef->position + rAWorld : jnt.anchorA))
                    .dot(wAxis) -
                (&jnt.driveLinearPosition.x)[a];
            physx::PxReal rho_drive =
                usePositionObjective
                    ? stiffness + damping / dt
                    : (usePhysicalVelocityObjective ? damping / dt
                                                    : damping / dt2);
            if (isAccelerationDrive && usePhysicalVelocityObjective) {
              const physx::PxReal driveScale =
                computeLinearDriveRecipResponse(bodyARef, bodyBRef,
                               rAWorld, rBWorld, wAxis);
              rho_drive *= driveScale;
            } else if (isAccelerationDrive) {
              const physx::PxReal driveScale =
                computeLinearDriveRecipResponse(bodyARef, bodyBRef,
                               rAWorld, rBWorld, wAxis);
              const physx::PxReal dampingOnly =
                  physx::PxMax(0.0f, damping - stiffness);
              const physx::PxReal implicitScale =
                  1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
              rho_drive *= driveScale * implicitScale;
            }
            const physx::PxReal authoredLimit =
                (&jnt.driveLinearForce.x)[a];
            const bool limitsAreForces =
                (jnt.sourceFlags & AvbdD6JointConstraint::
                                       eD6_DRIVE_LIMITS_ARE_FORCES) != 0;
            const physx::PxReal maxForce = limitsAreForces
                ? authoredLimit
                : physx::PxMin(PX_MAX_F32, authoredLimit / dt);
            const bool usePhysicalObjective =
                usePhysicalVelocityObjective || usePositionObjective;
            const physx::PxReal lambda = usePhysicalObjective
                ? 0.0f
                : (&jnt.lambdaDriveLinear.x)[a];
            const physx::PxReal rawForce =
                usePositionObjective
                    ? stiffness * positionError + (damping / dt) * C
                    : rho_drive * C + lambda;
            const physx::PxReal driveForce = usePhysicalObjective
                ? physx::PxClamp(rawForce, -maxForce, maxForce)
                : rawForce;
            const bool saturated = usePhysicalObjective &&
                physx::PxAbs(rawForce) >= maxForce;
            physx::PxReal signAL = isBodyA ? -1.0f : 1.0f;
            physx::PxReal f = signAL * driveForce;
            if (saturated)
              rho_drive = 0.0f;

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
        if ((compiledDriveRows & 0x38u) != 0) {
          // Angular displacement from start-of-step for this body
          physx::PxQuat dqThis =
              body.rotation * body.prevRotation.getConjugate();
          if (dqThis.w < 0.0f)
            dqThis = -dqThis;
          physx::PxVec3 dThetaThis(dqThis.x, dqThis.y, dqThis.z);
          dThetaThis *= 2.0f;

          physx::PxVec3 dThetaOther =
              isBodyA ? jnt.externalAngularStepB
                      : jnt.externalAngularStepA;
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

          const bool slerpDrive =
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::SlerpVelocityDrive) ||
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::SlerpPositionDrive) ||
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::
                      CoupledAngularPositionDrive) ||
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::OrdinaryD6SlerpDrive);
          if (slerpDrive) {
            physx::PxReal damping =
                jnt.angularDamping.z; // SLERP uses Z damping slot
            if (damping > 0.0f) {
              const bool usePhysicalSlerpVelocityObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::SlerpVelocityDrive);
              const bool usePhysicalSlerpPositionObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::SlerpPositionDrive);
              if (usePhysicalSlerpPositionObjective) {
                physx::PxQuat worldFrameA =
                    bodyARef ? bodyARef->rotation * jnt.localFrameA
                             : jnt.localFrameA;
                physx::PxQuat worldFrameB =
                    bodyBRef ? bodyBRef->rotation * jnt.localFrameB
                             : jnt.localFrameB;
                worldFrameA.normalize();
                worldFrameB.normalize();
                physx::PxQuat currentRelative =
                    worldFrameA.getConjugate() * worldFrameB;
                currentRelative.normalize();
                physx::PxQuat targetRelative = jnt.driveAngularPosition;
                if (currentRelative.dot(targetRelative) < 0.0f)
                  targetRelative = -targetRelative;
                const physx::PxQuat delta =
                    targetRelative.getConjugate() * currentRelative;
                physx::PxVec3 rows[3];
                computeSlerpJacobianAxes(rows, worldFrameA * targetRelative,
                                         worldFrameB);
                const physx::PxReal stiffness = jnt.angularStiffness.z;
                for (int row = 0; row < 3; ++row) {
                  const physx::PxReal C = relDW.dot(rows[row]);
                  const physx::PxReal rawTorque =
                      stiffness * (&delta.x)[row] + (damping / dt) * C;
                  const physx::PxReal driveTorque = physx::PxClamp(
                      rawTorque, -jnt.driveAngularForce.z,
                      jnt.driveAngularForce.z);
                  const bool saturated =
                      physx::PxAbs(rawTorque) >= jnt.driveAngularForce.z;
                  const physx::PxReal rowTangent =
                      saturated ? 0.0f : stiffness + damping / dt;
                  for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                      A.angularAngular(k, l) +=
                          rowTangent * (&rows[row].x)[k] *
                          (&rows[row].x)[l];
                  gAngular += rows[row] * (signAL * driveTorque);
                }
              } else {
                physx::PxReal rho_drive =
                    usePhysicalSlerpVelocityObjective ? damping / dt
                                                      : damping / dt2;
                if (jnt.isAngularAccelerationDrive(2)) {
                  const physx::PxReal driveScale =
                      computeAngularDriveRecipResponse(
                          bodyARef, bodyBRef,
                          physx::PxVec3(1.0f, 0.0f, 0.0f));
                  const physx::PxReal stiffness = jnt.angularStiffness.z;
                  const physx::PxReal dampingOnly =
                      physx::PxMax(0.0f, damping - stiffness);
                  const physx::PxReal implicitScale =
                      1.0f /
                      (1.0f + dt * (dt * stiffness + dampingOnly));
                  rho_drive *= driveScale * implicitScale;
                }
                const physx::PxReal targetScale =
                    usePhysicalSlerpVelocityObjective &&
                            mConfig.angularDamping > 1e-6f
                        ? 1.0f / mConfig.angularDamping
                        : 1.0f;
                for (int k = 0; k < 3; ++k) {
                  physx::PxReal C =
                      (&relDW.x)[k] -
                      targetScale * (&worldAngTarget.x)[k];
                  const physx::PxReal lam =
                      usePhysicalSlerpVelocityObjective
                          ? 0.0f
                          : (&jnt.lambdaDriveAngular.x)[k];
                  const physx::PxReal rawTorque = rho_drive * C + lam;
                  const physx::PxReal driveTorque =
                      usePhysicalSlerpVelocityObjective
                          ? physx::PxClamp(rawTorque,
                                           -jnt.driveAngularForce.z,
                                           jnt.driveAngularForce.z)
                          : rawTorque;
                  const bool saturated =
                      usePhysicalSlerpVelocityObjective &&
                      physx::PxAbs(rawTorque) >=
                          jnt.driveAngularForce.z;
                  physx::PxReal f = signAL * driveTorque;

                  if (!saturated)
                    A.angularAngular(k, k) += rho_drive;
                  (&gAngular.x)[k] += f;
                }
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
              if ((compiledDriveRows &
                   (1u << axes[a].bit)) == 0)
                continue;
              const physx::PxReal damping =
                  (&jnt.angularDamping.x)[axes[a].dampIdx];
              const physx::PxReal stiffness =
                  (&jnt.angularStiffness.x)[axes[a].dampIdx];
              const bool isAccelerationDrive =
                  jnt.isAngularAccelerationDrive(axes[a].dampIdx);
              const physx::PxReal effectiveRate =
                  isAccelerationDrive ? dt * stiffness + damping : damping;
              if (effectiveRate <= 0.0f)
                continue;

              physx::PxVec3 wAxis = jointFrameA.rotate(axes[a].localAxis);
              // PhysX TGS convention: Twist/Swing target velocities are
              // applied as (wA - wB), meaning wB - wA = -target. SLERP is
              // applied as wB
              // - wA = target, which is handled above.
              physx::PxReal targetOmega_dt = -worldAngTarget.dot(wAxis);
              physx::PxReal C = relDW.dot(wAxis) - targetOmega_dt;

              const bool usePhysicalAngularAxisVelocityObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::
                          AngularAxisVelocityDrive);
              const bool usePhysicalAngularPositionObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::
                          AngularAxisPositionDrive);
              physx::PxReal positionResidual = 0.0f;
              physx::PxReal positionTangent = 0.0f;
              if (usePhysicalAngularPositionObjective) {
                physx::PxQuat worldFrameA =
                    bodyARef ? bodyARef->rotation * jnt.localFrameA
                             : jnt.localFrameA;
                physx::PxQuat worldFrameB =
                    bodyBRef ? bodyBRef->rotation * jnt.localFrameB
                             : jnt.localFrameB;
                worldFrameA.normalize();
                worldFrameB.normalize();
                physx::PxQuat currentRelative =
                    worldFrameA.getConjugate() * worldFrameB;
                currentRelative.normalize();
                physx::PxQuat targetRelative = jnt.driveAngularPosition;
                if (currentRelative.dot(targetRelative) < 0.0f)
                  targetRelative = -targetRelative;
                const physx::PxQuat delta =
                    currentRelative * targetRelative.getConjugate();

                if (axes[a].dampIdx == 0) {
                  // ExtD6Joint emits geometricError=-2*delta.x for TWIST.
                  // AVBD's gradient uses current-target, so this is the
                  // opposite sign. Its local derivative is delta.w.
                  positionResidual = 2.0f * delta.x;
                  positionTangent = physx::PxAbs(delta.w);
                } else if (axes[a].dampIdx == 1) {
                  // ExtD6Joint emits delta.getBasisVector0().z for SWING1.
                  // The AVBD gradient again uses the opposite sign. In the
                  // predicate-approved isolated SWING1 row this is a
                  // full-angle sine residual with a cosine tangent.
                  positionResidual = -delta.getBasisVector0().z;
                  positionTangent = physx::PxAbs(
                      1.0f - 2.0f * delta.y * delta.y);
                } else {
                  // ExtD6Joint emits -delta.getBasisVector0().y for SWING2.
                  // AVBD uses its opposite gradient. In the predicate-approved
                  // isolated SWING2 row this is again a full-angle sine with
                  // the corresponding cosine tangent.
                  positionResidual = delta.getBasisVector0().y;
                  positionTangent = physx::PxAbs(
                      1.0f - 2.0f * delta.z * delta.z);
                }
              }
              // In the scoped force-mode TWIST/SWING1/SWING2 subset, damping
              // has the physical units torque/(angular velocity).  C is an
              // angular displacement over the step, so damping/dt maps C back
              // to a torque.  The wider angular family retains its existing AL
              // objective until SLERP and spring semantics are gated.
              physx::PxReal rho_drive =
                  usePhysicalAngularPositionObjective
                      ? stiffness * positionTangent + damping / dt
                      : (usePhysicalAngularAxisVelocityObjective
                             ? damping / dt
                             : damping / dt2);
              if (isAccelerationDrive) {
                const physx::PxReal driveScale =
                  computeAngularDriveRecipResponse(bodyARef, bodyBRef, wAxis);
                const physx::PxReal implicitScale =
                  1.0f / (1.0f + dt * effectiveRate);
                rho_drive = driveScale * implicitScale * effectiveRate;
              }
              const bool usePhysicalAngularObjective =
                  usePhysicalAngularAxisVelocityObjective ||
                  usePhysicalAngularPositionObjective;
              const physx::PxReal lambda =
                  usePhysicalAngularObjective
                      ? 0.0f
                      : (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx];
              const physx::PxReal rawTorque =
                  usePhysicalAngularPositionObjective
                      ? stiffness * positionResidual + (damping / dt) * C
                      : rho_drive * C + lambda;
              const physx::PxReal driveTorque =
                  usePhysicalAngularObjective
                      ? physx::PxClamp(rawTorque,
                                       -jnt.driveAngularForce[axes[a].dampIdx],
                                       jnt.driveAngularForce[axes[a].dampIdx])
                      : rawTorque;
              const bool saturated =
                  usePhysicalAngularObjective &&
                  physx::PxAbs(rawTorque) >=
                      jnt.driveAngularForce[axes[a].dampIdx];
              physx::PxReal f = signAL * driveTorque;
              if (saturated)
                rho_drive = 0.0f;

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

  const physx::PxReal ogcAlpha = limitRigidOgcCandidate(
      ogcPairContext, bodyIndex, body, deltaPos, deltaTheta, softContacts,
      numSoftContacts, rigidTargetContactStarts, rigidTargetContactRefs,
      softParticles);
  deltaPos *= ogcAlpha;
  deltaTheta *= ogcAlpha;

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

// Validate provider-owned support arrays exactly once, before any mixed
// lifecycle stage consumes them.  Besides making the fallback boundary
// explicit, this lets prediction share the Scene's swept-contact preparation
// pass without trusting a merely shape-complete CSR.
static bool avbdValidateSoftIslandExecutionPlan(
    const AvbdSoftIslandExecutionPlan &plan,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftContacts) {
  if (!plan.isComplete(numSoftParticles) || !softBodies ||
      numSoftBodies == 0)
    return false;

  for (physx::PxU32 particleIndex = 0;
       particleIndex < numSoftParticles; ++particleIndex) {
    const physx::PxU32 bodyIndex = plan.particleBodyIndices[particleIndex];
    if (bodyIndex >= numSoftBodies)
      return false;
    const AvbdSoftBody &softBody = softBodies[bodyIndex];
    if (particleIndex < softBody.compiled.particleStart ||
        particleIndex - softBody.compiled.particleStart >=
            softBody.compiled.particleCount ||
        plan.contactStarts[particleIndex] >
            plan.contactStarts[particleIndex + 1])
      return false;
  }
  for (physx::PxU32 refIndex = 0; refIndex < plan.numContactRefs;
       ++refIndex) {
    if (plan.contactRefs[refIndex].contactIndex >= numSoftContacts)
      return false;
  }
  return true;
}

// The rigid-facing contact mirror is optional for provider compatibility.  A
// valid mirror contains each dynamic rigid target exactly once, in source
// order within its target-body range.  This proof lets the 6x6 blocks skip
// their former all-soft-contact scans without changing the fallback route.
#if PX_CHECKED
static bool avbdValidateRigidTargetContactPlan(
    const AvbdSoftIslandExecutionPlan &plan,
    const AvbdSolverBody *bodies, physx::PxU32 numRigidBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts) {
  if (!plan.hasRigidTargetContactPlan(numRigidBodies) ||
      !bodies || (numSoftContacts > 0 && !softContacts))
    return false;

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numRigidBodies;
       ++bodyIndex) {
    if (bodies[bodyIndex].nodeIndex != bodyIndex)
      return false;
  }

  physx::PxU32 expectedRefCount = 0;
  for (physx::PxU32 contactIndex = 0; contactIndex < numSoftContacts;
       ++contactIndex) {
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    if (geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numRigidBodies)
      ++expectedRefCount;
  }
  if (expectedRefCount != plan.numRigidTargetContactRefs)
    return false;

  for (physx::PxU32 rigidBodyIndex = 0;
       rigidBodyIndex < numRigidBodies; ++rigidBodyIndex) {
    const physx::PxU32 begin =
        plan.rigidTargetContactStarts[rigidBodyIndex];
    const physx::PxU32 end =
        plan.rigidTargetContactStarts[rigidBodyIndex + 1];
    if (begin > end)
      return false;
    physx::PxU32 previousContactIndex = 0;
    for (physx::PxU32 refIndex = begin; refIndex < end; ++refIndex) {
      const physx::PxU32 contactIndex =
          plan.rigidTargetContactRefs[refIndex];
      if (contactIndex >= numSoftContacts ||
          (refIndex > begin && previousContactIndex >= contactIndex))
        return false;
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex != rigidBodyIndex)
        return false;
      previousContactIndex = contactIndex;
    }
  }
  return true;
}
#endif

// A Scene-generated plan is immutable for the island solve.  Shipping builds
// retain only its O(numBodies) ownership proof so that the contact CSR removes
// work rather than moving the same work into validation.  Checked builds also
// verify every source row, which keeps provider-contract failures fail-closed
// during development.
static bool avbdCanUseRigidTargetContactPlan(
    const AvbdSoftIslandExecutionPlan &plan,
    const AvbdSolverBody *bodies, physx::PxU32 numRigidBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts) {
  if (!plan.hasRigidTargetContactPlan(numRigidBodies) || !bodies)
    return false;
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numRigidBodies;
       ++bodyIndex) {
    const physx::PxU32 begin =
        plan.rigidTargetContactStarts[bodyIndex];
    const physx::PxU32 end =
        plan.rigidTargetContactStarts[bodyIndex + 1];
    if (bodies[bodyIndex].nodeIndex != bodyIndex || begin > end ||
        end > plan.numRigidTargetContactRefs)
      return false;
  }
#if PX_CHECKED
  return avbdValidateRigidTargetContactPlan(
      plan, bodies, numRigidBodies, softContacts, numSoftContacts);
#else
  PX_UNUSED(softContacts);
  PX_UNUSED(numSoftContacts);
  return true;
#endif
}

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
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    FeatherstoneArticulation *const *articulationForBody,
    const physx::PxU32 *linkIndexForBody,
    AvbdSolverStats &stats) {

  PX_PROFILE_ZONE("AVBD.solveWithJoints", 0);

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  const bool hasCompleteSoftSelection =
      softParticles && numSoftParticles > 0 &&
      softBodies && numSoftBodies > 0 &&
      (numSoftContacts == 0 || softContacts);
  if (!mInitialized ||
      (numBodies == 0 && !hasCompleteSoftSelection)) {
    return;
  }

  // Validate once at the native island boundary.  No lifecycle stage is
  // allowed to consume a provider token before this semantic validation: a
  // shape-complete but malformed plan must still take the local fallback.
  const bool useProvidedSoftExecutionPlan =
      hasCompleteSoftSelection && softExecutionPlan &&
      avbdValidateSoftIslandExecutionPlan(
          *softExecutionPlan, softBodies, numSoftBodies, numSoftParticles,
          numSoftContacts);
  const bool useProvidedRigidTargetContactPlan =
      useProvidedSoftExecutionPlan && numBodies > 0 &&
      avbdCanUseRigidTargetContactPlan(
          *softExecutionPlan, bodies, numBodies, softContacts,
          numSoftContacts);
  // A Scene-selected mixed island has already predicted its particles while
  // compiling swept/OGC contact objectives.  Reusing that exact frame state
  // makes prediction one owner-stage shared by contact preparation and the
  // AVBD primal, without changing direct/standalone callers that do not
  // provide a valid complete plan.
  const bool hasPreparedSoftPrediction =
      useProvidedSoftExecutionPlan &&
      softExecutionPlan->softPredictionPrepared;

	physx::PxArray<physx::PxU8> endpointRecoveredWorldStaticBodies;
	physx::PxArray<physx::PxVec3> endpointWorldStaticRecoveryNormals;
	physx::PxArray<physx::PxU8> endpointRecoveredDynamicSoftRigidContacts;

  PX_AVBD_PROFILE_STAT(stats.numBodies = numBodies);
  PX_AVBD_PROFILE_STAT(stats.numContacts = numContacts);
  PX_AVBD_PROFILE_STAT(stats.numJoints = numD6 + numGear);

  const physx::PxReal invDt = 1.0f / dt;
  const physx::PxReal invDt2 = invDt * invDt;

  for (physx::PxU32 i = 0; i < numD6; ++i)
    resetAvbdJointObjectiveProgram(d6Joints[i].objectiveProgram);

  const auto getJointObjectiveSourceRowMask =
      [](const AvbdD6JointConstraint &joint,
         AvbdJointObjectiveKind kind) -> physx::PxU32 {
        const physx::PxU32 linearDriveRows =
            joint.driveFlags & 0x7u;
        const physx::PxU32 angularDriveRows =
            joint.driveFlags & 0x38u;
        switch (kind) {
        case AvbdJointObjectiveKind::CoupledLinearVelocityDrive:
        case AvbdJointObjectiveKind::LinearPositionDrive:
        case AvbdJointObjectiveKind::CoupledLinearPositionDrive:
        case AvbdJointObjectiveKind::OrdinaryD6LinearDrive:
          return linearDriveRows;
        case AvbdJointObjectiveKind::AngularAxisVelocityDrive:
        case AvbdJointObjectiveKind::AngularAxisPositionDrive:
        case AvbdJointObjectiveKind::SlerpVelocityDrive:
        case AvbdJointObjectiveKind::SlerpPositionDrive:
        case AvbdJointObjectiveKind::CoupledAngularPositionDrive:
        case AvbdJointObjectiveKind::OrdinaryD6AngularAxisDrive:
        case AvbdJointObjectiveKind::OrdinaryD6SlerpDrive:
          return angularDriveRows;
        case AvbdJointObjectiveKind::CoupledSpatialTendon:
        case AvbdJointObjectiveKind::GenericHard1D:
        case AvbdJointObjectiveKind::GenericAccelerationDamping1D:
        case AvbdJointObjectiveKind::GenericForceSpring1D:
        case AvbdJointObjectiveKind::GenericRestitution1D:
        case AvbdJointObjectiveKind::ArticulationHardMimic:
        case AvbdJointObjectiveKind::ArticulationCompliantMimic:
        case AvbdJointObjectiveKind::ArticulationFixedTendon:
        case AvbdJointObjectiveKind::ArticulationSpatialTendon:
          return eJOINT_SOURCE_GENERIC_ROW;
        case AvbdJointObjectiveKind::NativeRevoluteMotor:
          return eJOINT_SOURCE_NATIVE_MOTOR;
        case AvbdJointObjectiveKind::CoupledFixedD6:
          return eJOINT_SOURCE_LINEAR_MOTION_X |
                 eJOINT_SOURCE_LINEAR_MOTION_Y |
                 eJOINT_SOURCE_LINEAR_MOTION_Z |
                 eJOINT_SOURCE_ANGULAR_MOTION_X |
                 eJOINT_SOURCE_ANGULAR_MOTION_Y |
                 eJOINT_SOURCE_ANGULAR_MOTION_Z;
        case AvbdJointObjectiveKind::CoupledSphericalCone:
          return eJOINT_SOURCE_LINEAR_MOTION_X |
                 eJOINT_SOURCE_LINEAR_MOTION_Y |
                 eJOINT_SOURCE_LINEAR_MOTION_Z |
                 eJOINT_SOURCE_ANGULAR_CONE;
        case AvbdJointObjectiveKind::NativePassiveReaction: {
          physx::PxU32 sourceRows = 0;
          const physx::PxU32 linearRows[3] = {
              eJOINT_SOURCE_LINEAR_MOTION_X,
              eJOINT_SOURCE_LINEAR_MOTION_Y,
              eJOINT_SOURCE_LINEAR_MOTION_Z};
          const physx::PxU32 angularRows[3] = {
              eJOINT_SOURCE_ANGULAR_MOTION_X,
              eJOINT_SOURCE_ANGULAR_MOTION_Y,
              eJOINT_SOURCE_ANGULAR_MOTION_Z};
          for (physx::PxU32 axis = 0; axis < 3; ++axis) {
            if (joint.getLinearMotion(axis) != 2)
              sourceRows |= linearRows[axis];
            if (joint.getAngularMotion(axis) != 2)
              sourceRows |= angularRows[axis];
          }
          return sourceRows;
        }
        case AvbdJointObjectiveKind::OrdinaryD6Position: {
          physx::PxU32 sourceRows = 0;
          const physx::PxU32 linearRows[3] = {
              eJOINT_SOURCE_LINEAR_MOTION_X,
              eJOINT_SOURCE_LINEAR_MOTION_Y,
              eJOINT_SOURCE_LINEAR_MOTION_Z};
          const physx::PxU32 angularRows[3] = {
              eJOINT_SOURCE_ANGULAR_MOTION_X,
              eJOINT_SOURCE_ANGULAR_MOTION_Y,
              eJOINT_SOURCE_ANGULAR_MOTION_Z};
          const physx::PxU32 ellipticalConeFlags =
              AvbdD6JointConstraint::
                  eD6_LEGACY_CONE_LIMIT_ACTIVE |
              AvbdD6JointConstraint::
                  eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE;
          const bool ellipticalCone =
              (joint.sourceFlags & ellipticalConeFlags) != 0;
          for (physx::PxU32 axis = 0; axis < 3; ++axis) {
            if (joint.getLinearMotion(axis) != 2)
              sourceRows |= linearRows[axis];
            if (joint.getAngularMotion(axis) != 2 &&
                !(ellipticalCone && axis >= 1))
              sourceRows |= angularRows[axis];
          }
          if (joint.coneAngleLimit > 0.0f)
            sourceRows |= eJOINT_SOURCE_ANGULAR_CONE;
          return sourceRows;
        }
        case AvbdJointObjectiveKind::None:
        case AvbdJointObjectiveKind::Invalid:
          break;
        }
        return 0;
      };
  const auto countJointObjectiveSourceRows =
      [](physx::PxU32 sourceRowMask) -> physx::PxU16 {
        physx::PxU16 count = 0;
        while (sourceRowMask != 0) {
          count += physx::PxU16(sourceRowMask & 1u);
          sourceRowMask >>= 1;
        }
        return count;
      };

  const auto compileSingleJointObjective =
      [&](bool candidate, AvbdVelocityObjectiveOwner owner,
          AvbdJointObjectiveKind kind) {
        if (!candidate || numD6 == 0)
          return;
        const physx::PxU32 sourceRowMask =
            getJointObjectiveSourceRowMask(d6Joints[0], kind);
        const physx::PxU16 objectiveRowCount =
            kind == AvbdJointObjectiveKind::CoupledFixedD6
                ? 6u
                : (kind ==
                           AvbdJointObjectiveKind::
                               CoupledSphericalCone
                       ? 4u
                       : (kind ==
                                  AvbdJointObjectiveKind::
                                      NativePassiveReaction
                              ? countJointObjectiveSourceRows(
                                    sourceRowMask)
                              : 1u));
        assignAvbdJointObjective(
            d6Joints[0].objectiveProgram, owner, kind,
            objectiveRowCount,
            sourceRowMask,
            d6Joints[0].cacheKey);
      };

  const bool linearPositionDriveCandidate =
      isLinearPositionDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      linearPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::LinearPositionDrive);
  const bool coupledLinearPositionDriveCandidate =
      isCoupledLinearPositionDriveIslandSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          gravity, numGear, numSoftParticles, numSoftBodies,
          numSoftContacts);
  compileSingleJointObjective(
      coupledLinearPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledLinearPositionDrive);
  const bool angularAxisVelocityDriveCandidate =
      isAngularAxisVelocityDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      angularAxisVelocityDriveCandidate,
      AvbdVelocityObjectiveOwner::JointFinalize,
      AvbdJointObjectiveKind::AngularAxisVelocityDrive);
  const bool angularAxisPositionDriveCandidate =
      isAngularAxisPositionDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      angularAxisPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::AngularAxisPositionDrive);
  const bool slerpVelocityDriveCandidate =
      isSlerpVelocityDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      slerpVelocityDriveCandidate,
      AvbdVelocityObjectiveOwner::JointFinalize,
      AvbdJointObjectiveKind::SlerpVelocityDrive);
  const bool slerpPositionDriveCandidate =
      isSlerpPositionDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      slerpPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::SlerpPositionDrive);
  const bool coupledAngularPositionDriveCandidate =
      isCoupledAngularPositionDriveIslandSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledAngularPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledAngularPositionDrive);
  const bool coupledLinearDriveCandidate =
      isCoupledLinearDriveIslandSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledLinearDriveCandidate,
      AvbdVelocityObjectiveOwner::JointFinalize,
      AvbdJointObjectiveKind::CoupledLinearVelocityDrive);
  const bool coupledFixedD6Candidate =
      isCoupledFixedD6IslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledFixedD6Candidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledFixedD6);
  const bool coupledSphericalConeCandidate =
      isCoupledSphericalConeIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledSphericalConeCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledSphericalCone);
  const auto isNativePassiveReactionSource =
      [](const AvbdD6JointConstraint &joint) {
        return joint.header.type ==
                   AvbdConstraintType::eJOINT_FIXED ||
               (joint.header.type ==
                    AvbdConstraintType::eJOINT_PRISMATIC &&
                joint.linearMotion == 0x02u &&
                joint.angularMotion == 0u) ||
               (joint.header.type ==
                    AvbdConstraintType::eJOINT_REVOLUTE &&
                joint.linearMotion == 0u &&
                joint.angularMotion == 0x02u &&
                joint.motorEnabled == 0u);
      };
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    if (!isNativePassiveReactionSource(joint) ||
        hasAvbdJointObjective(
            joint.objectiveProgram,
            AvbdJointObjectiveKind::CoupledFixedD6))
      continue;
    const physx::PxU32 sourceRowMask =
        getJointObjectiveSourceRowMask(
            joint,
            AvbdJointObjectiveKind::NativePassiveReaction);
    assignAvbdJointObjective(
        joint.objectiveProgram,
        AvbdVelocityObjectiveOwner::PositionAL,
        AvbdJointObjectiveKind::NativePassiveReaction,
        countJointObjectiveSourceRows(sourceRowMask),
        sourceRowMask, joint.cacheKey);
  }

  bool coupledLinearPositionDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledLinearPositionDrive);
  bool coupledLinearPositionDriveFrictionPositionOwnerIsland =
      coupledLinearPositionDriveIsland &&
      areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts, gravity);
  const bool slerpVelocityDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::SlerpVelocityDrive);
  bool coupledAngularPositionDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledAngularPositionDrive);
  bool coupledLinearDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledLinearVelocityDrive);
  bool coupledFixedD6Island =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledFixedD6);
  bool coupledSphericalConeIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledSphericalCone);
  physx::PxArray<physx::PxU32> coupledSpatialTendonRowIndices;
  const bool coupledSpatialTendonCandidate = findCoupledSpatialTendonRows(
      bodies, numBodies, numContacts, d6Joints, numD6, numGear,
      numSoftParticles, numSoftBodies, numSoftContacts,
      coupledSpatialTendonRowIndices);
  if (coupledSpatialTendonCandidate) {
    bool supported =
        coupledSpatialTendonRowIndices.size() <= PX_MAX_U16;
    physx::PxU64 objectiveKey = ~physx::PxU64(0);
    for (physx::PxU32 row = 0;
         row < coupledSpatialTendonRowIndices.size(); ++row) {
      objectiveKey = physx::PxMin(
          objectiveKey,
          d6Joints[coupledSpatialTendonRowIndices[row]].cacheKey);
    }
    for (physx::PxU32 row = 0;
         row < coupledSpatialTendonRowIndices.size() && supported;
         ++row) {
      supported = canAssignAvbdJointObjective(
          d6Joints[coupledSpatialTendonRowIndices[row]]
              .objectiveProgram,
          AvbdVelocityObjectiveOwner::PositionAL,
          AvbdJointObjectiveKind::CoupledSpatialTendon,
          physx::PxU16(coupledSpatialTendonRowIndices.size()),
          eJOINT_SOURCE_GENERIC_ROW,
          objectiveKey);
    }
    if (supported) {
      for (physx::PxU32 row = 0;
           row < coupledSpatialTendonRowIndices.size(); ++row) {
        assignAvbdJointObjective(
            d6Joints[coupledSpatialTendonRowIndices[row]]
                .objectiveProgram,
            AvbdVelocityObjectiveOwner::PositionAL,
            AvbdJointObjectiveKind::CoupledSpatialTendon,
            physx::PxU16(coupledSpatialTendonRowIndices.size()),
            eJOINT_SOURCE_GENERIC_ROW,
            objectiveKey);
      }
    } else {
      for (physx::PxU32 row = 0;
           row < coupledSpatialTendonRowIndices.size(); ++row) {
        invalidateAvbdJointObjective(
            d6Joints[coupledSpatialTendonRowIndices[row]]
                .objectiveProgram);
      }
    }
  }
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    if (hasAvbdJointObjective(
            joint.objectiveProgram,
            AvbdJointObjectiveKind::CoupledSpatialTendon))
      continue;

    const physx::PxU32 sourceFlags = joint.sourceFlags;
    AvbdJointObjectiveKind kind = AvbdJointObjectiveKind::None;
    AvbdVelocityObjectiveOwner owner =
        AvbdVelocityObjectiveOwner::PositionAL;
    if ((sourceFlags &
         AvbdD6JointConstraint::eARTICULATION_MIMIC_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationHardMimic;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eARTICULATION_COMPLIANT_MIMIC_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationCompliantMimic;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eARTICULATION_FIXED_TENDON_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationFixedTendon;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eARTICULATION_SPATIAL_TENDON_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationSpatialTendon;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eGENERIC_ACCELERATION_DAMPING_1D_ROW) != 0)
      kind =
          AvbdJointObjectiveKind::GenericAccelerationDamping1D;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eGENERIC_FORCE_SPRING_1D_ROW) != 0)
      kind = AvbdJointObjectiveKind::GenericForceSpring1D;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eGENERIC_RESTITUTION_1D_ROW) != 0) {
      kind = AvbdJointObjectiveKind::GenericRestitution1D;
      owner = AvbdVelocityObjectiveOwner::JointFinalize;
    } else if ((sourceFlags &
                AvbdD6JointConstraint::
                    eGENERIC_HARD_1D_ROW) != 0)
      kind = AvbdJointObjectiveKind::GenericHard1D;

    if (kind == AvbdJointObjectiveKind::None)
      continue;
    assignAvbdJointObjective(
        joint.objectiveProgram, owner, kind, 1u,
        eJOINT_SOURCE_GENERIC_ROW, joint.cacheKey);
  }
  const physx::PxU32 genericSourceFlags =
      AvbdD6JointConstraint::eGENERIC_HARD_1D_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_MIMIC_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_FIXED_TENDON_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_SPATIAL_TENDON_ROW |
      AvbdD6JointConstraint::
          eGENERIC_ACCELERATION_DAMPING_1D_ROW |
      AvbdD6JointConstraint::
          eGENERIC_FORCE_SPRING_1D_ROW |
      AvbdD6JointConstraint::
          eGENERIC_RESTITUTION_1D_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_COMPLIANT_MIMIC_ROW;
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    AvbdCompiledJointObjectiveProgram &program =
        joint.objectiveProgram;
    if (program.invalid)
      continue;

    physx::PxU32 compiledSourceRows = 0;
    for (physx::PxU32 entryIndex = 0;
         entryIndex < program.entryCount; ++entryIndex)
      compiledSourceRows |=
          program.entries[entryIndex].sourceRowMask;

    const physx::PxU32 positionSourceRows =
        getJointObjectiveSourceRowMask(
            joint,
            AvbdJointObjectiveKind::OrdinaryD6Position);
    const physx::PxU32 remainingPositionSourceRows =
        positionSourceRows & ~compiledSourceRows;
    if (remainingPositionSourceRows != 0) {
      assignAvbdJointObjective(
          program, AvbdVelocityObjectiveOwner::PositionAL,
          AvbdJointObjectiveKind::OrdinaryD6Position,
          countJointObjectiveSourceRows(
              remainingPositionSourceRows),
          remainingPositionSourceRows, joint.cacheKey);
      if (program.invalid)
        continue;
      compiledSourceRows |= remainingPositionSourceRows;
    }

    const physx::PxU32 remainingLinearDriveSourceRows =
        (joint.driveFlags & 0x7u) & ~compiledSourceRows;
    if (remainingLinearDriveSourceRows != 0) {
      assignAvbdJointObjective(
          program, AvbdVelocityObjectiveOwner::PositionAL,
          AvbdJointObjectiveKind::OrdinaryD6LinearDrive,
          countJointObjectiveSourceRows(
              remainingLinearDriveSourceRows),
          remainingLinearDriveSourceRows, joint.cacheKey);
      if (program.invalid)
        continue;
      compiledSourceRows |= remainingLinearDriveSourceRows;
    }

    const physx::PxU32 remainingAngularDriveSourceRows =
        (joint.driveFlags & 0x38u) & ~compiledSourceRows;
    if (remainingAngularDriveSourceRows != 0) {
      const bool slerpDrive =
          (joint.sourceFlags &
           AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0;
      assignAvbdJointObjective(
          program, AvbdVelocityObjectiveOwner::PositionAL,
          slerpDrive
              ? AvbdJointObjectiveKind::OrdinaryD6SlerpDrive
              : AvbdJointObjectiveKind::OrdinaryD6AngularAxisDrive,
          countJointObjectiveSourceRows(
              remainingAngularDriveSourceRows),
          remainingAngularDriveSourceRows, joint.cacheKey);
    }
  }
  bool coupledSpatialTendonIsland =
      coupledSpatialTendonCandidate;
  for (physx::PxU32 row = 0;
       row < coupledSpatialTendonRowIndices.size() &&
       coupledSpatialTendonIsland;
       ++row) {
    coupledSpatialTendonIsland = hasAvbdJointObjective(
        d6Joints[coupledSpatialTendonRowIndices[row]]
            .objectiveProgram,
        AvbdJointObjectiveKind::CoupledSpatialTendon);
  }
  const bool passiveCenteredGearVelocityProjectionIsland =
      isPassiveCenteredGearVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, gearJoints,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts);
  physx::PxU32 nativeRevoluteMotorGearJointIndex = PX_MAX_U32;
  const bool nativeRevoluteMotorGearVelocityProjectionCandidate =
      isNativeRevoluteMotorGearVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, gearJoints,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts,
          nativeRevoluteMotorGearJointIndex);
  const bool nativeRevoluteMotorVelocityProjectionCandidate =
      isSingleNativeRevoluteMotorVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  bool contactCoupledNativeRevoluteMotorUnsupportedTransientContact =
      false;
  const bool contactCoupledNativeRevoluteMotorVelocityProjectionCandidate =
      isContactCoupledNativeRevoluteMotorVelocityProjectionSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          gravity, numGear, numSoftParticles, numSoftBodies,
          numSoftContacts,
          contactCoupledNativeRevoluteMotorUnsupportedTransientContact);

  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    if (joint.motorEnabled == 0u)
      continue;
    const bool supported =
        (nativeRevoluteMotorVelocityProjectionCandidate &&
         jointIndex == 0u) ||
        (nativeRevoluteMotorGearVelocityProjectionCandidate &&
         jointIndex == nativeRevoluteMotorGearJointIndex) ||
        (contactCoupledNativeRevoluteMotorVelocityProjectionCandidate &&
         jointIndex == 0u);
    assignAvbdJointObjective(
        joint.objectiveProgram,
        supported ? AvbdVelocityObjectiveOwner::JointFinalize
                  : AvbdVelocityObjectiveOwner::Unsupported,
        AvbdJointObjectiveKind::NativeRevoluteMotor, 1u,
        eJOINT_SOURCE_NATIVE_MOTOR, joint.cacheKey);
  }

  bool nativeRevoluteMotorVelocityProjectionIsland =
      nativeRevoluteMotorVelocityProjectionCandidate && numD6 > 0u &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::NativeRevoluteMotor);
  bool nativeRevoluteMotorGearVelocityProjectionIsland =
      nativeRevoluteMotorGearVelocityProjectionCandidate &&
      nativeRevoluteMotorGearJointIndex < numD6 &&
      hasAvbdJointObjective(
          d6Joints[nativeRevoluteMotorGearJointIndex].objectiveProgram,
          AvbdJointObjectiveKind::NativeRevoluteMotor);
  bool contactCoupledNativeRevoluteMotorVelocityProjectionIsland =
      contactCoupledNativeRevoluteMotorVelocityProjectionCandidate &&
      numD6 > 0u &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::NativeRevoluteMotor);
  const bool compileContactCoupledNativeRevoluteMotorObjective =
      contactCoupledNativeRevoluteMotorVelocityProjectionCandidate ||
      contactCoupledNativeRevoluteMotorUnsupportedTransientContact;
  if (compileContactCoupledNativeRevoluteMotorObjective) {
    const AvbdVelocityObjectiveOwner objectiveOwner =
        contactCoupledNativeRevoluteMotorVelocityProjectionIsland
            ? AvbdVelocityObjectiveOwner::JointFinalize
            : AvbdVelocityObjectiveOwner::Unsupported;
    bool objectiveAssignmentSupported = true;
    physx::PxU64 objectiveKey = ~physx::PxU64(0);
    for (physx::PxU32 c = 0; c < numContacts; ++c)
      objectiveKey =
          physx::PxMin(objectiveKey, contacts[c].cacheKey);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      if (!canAssignAvbdVelocityObjective(
              contacts[c].objectiveProgram,
              objectiveOwner,
              AvbdVelocityObjectiveKind::PassiveFriction,
              AvbdVelocityObjectiveSpan::NormalAndTangentCone,
              AvbdVelocityObjectiveReconstruction::
                  SolveStartInertial,
              numContacts,
              objectiveKey)) {
        contactCoupledNativeRevoluteMotorVelocityProjectionIsland =
            false;
        objectiveAssignmentSupported = false;
        break;
      }
    }
    if (objectiveAssignmentSupported) {
      for (physx::PxU32 c = 0; c < numContacts; ++c) {
        assignAvbdVelocityObjective(
            contacts[c].objectiveProgram,
            objectiveOwner,
            AvbdVelocityObjectiveKind::PassiveFriction,
            AvbdVelocityObjectiveSpan::NormalAndTangentCone,
            AvbdVelocityObjectiveReconstruction::SolveStartInertial,
            numContacts,
            objectiveKey);
      }
    } else {
      contactCoupledNativeRevoluteMotorUnsupportedTransientContact =
          false;
      if (numD6 > 0u)
        fallbackAvbdJointObjective(
            d6Joints[0].objectiveProgram,
            AvbdJointObjectiveKind::NativeRevoluteMotor);
      for (physx::PxU32 c = 0; c < numContacts; ++c) {
        invalidateAvbdVelocityObjective(
            contacts[c].objectiveProgram);
      }
    }
  }

  // Specialized contact-coupled joint objectives have first claim on their
  // physical contact sources. Compile every remaining ordinary rigid contact
  // only after that priority decision, so later solve stages consume an
  // immutable unique-owner program rather than overriding earlier flags.
  if (numSoftParticles == 0 && numSoftBodies == 0 &&
      numSoftContacts == 0) {
    compileAvbdOrdinaryRigidContactObjectives(
        contacts, numContacts, numBodies, contactMap);
  }

  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    AvbdCompiledJointObjectiveProgram &program =
        joint.objectiveProgram;
    if (program.invalid)
      continue;

    physx::PxU32 compiledSourceRows = 0;
    for (physx::PxU32 entryIndex = 0;
         entryIndex < program.entryCount; ++entryIndex)
      compiledSourceRows |=
          program.entries[entryIndex].sourceRowMask;

    const physx::PxU32 positionSourceRows =
        getJointObjectiveSourceRowMask(
            joint,
            AvbdJointObjectiveKind::OrdinaryD6Position);
    physx::PxU32 allSourceRows =
        positionSourceRows | (joint.driveFlags & 0x3fu);
    if ((joint.sourceFlags & genericSourceFlags) != 0)
      allSourceRows |= eJOINT_SOURCE_GENERIC_ROW;
    if (joint.motorEnabled != 0u)
      allSourceRows |= eJOINT_SOURCE_NATIVE_MOTOR;
    program.legacySourceRowMask =
        allSourceRows & ~compiledSourceRows;
  }
  bool conserveNativeRevoluteMotorAngularMomentum = false;
  physx::PxReal nativeRevoluteMotorExpectedAngularMomentum = 0.0f;
  bool conserveNativeRevoluteMotorAngularMomentumVector = false;
  physx::PxVec3 nativeRevoluteMotorExpectedAngularMomentumVector(
      0.0f);
  bool conserveNativeRevoluteMotorLinearMomentum = false;
  physx::PxVec3 nativeRevoluteMotorExpectedLinearMomentum(0.0f);
  bool conserveNativeRevoluteMotorSpatialMomentum = false;
  physx::PxVec3 nativeRevoluteMotorExpectedSpatialAngularMomentum(
      0.0f);
  bool useNativeRevoluteMotorSolveStartRelativeVelocity = false;
  physx::PxReal nativeRevoluteMotorSolveStartRelativeVelocity = 0.0f;
  if (nativeRevoluteMotorVelocityProjectionIsland) {
    const AvbdD6JointConstraint &motor = d6Joints[0];
    const physx::PxU32 bodyA = motor.header.bodyIndexA;
    const physx::PxU32 bodyB = motor.header.bodyIndexB;
    const bool dynamicA =
        bodyA < numBodies && bodies[bodyA].invMass > 0.0f;
    const bool dynamicB =
        bodyB < numBodies && bodies[bodyB].invMass > 0.0f;
    physx::PxVec3 motorAxis =
        dynamicA
            ? (bodies[bodyA].rotation * motor.localFrameA)
                  .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
            : motor.localFrameA.rotate(
                  physx::PxVec3(1.0f, 0.0f, 0.0f));
    if (motorAxis.normalize() > 1e-6f) {
      if (dynamicA)
        nativeRevoluteMotorSolveStartRelativeVelocity -=
            motorAxis.dot(bodies[bodyA].angularVelocity);
      if (dynamicB)
        nativeRevoluteMotorSolveStartRelativeVelocity +=
            motorAxis.dot(bodies[bodyB].angularVelocity);
      useNativeRevoluteMotorSolveStartRelativeVelocity =
          physx::PxIsFinite(
              nativeRevoluteMotorSolveStartRelativeVelocity);
    }
    if (bodyA < numBodies && bodyB < numBodies &&
        bodyA != bodyB && bodies[bodyA].invMass > 0.0f &&
        bodies[bodyB].invMass > 0.0f) {
      physx::PxVec3 axis =
          (bodies[bodyA].rotation * motor.localFrameA)
              .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
      if (axis.normalize() > 1e-6f) {
        const physx::PxMat33 inertiaA =
            bodies[bodyA].invInertiaWorld.getInverse();
        const physx::PxMat33 inertiaB =
            bodies[bodyB].invInertiaWorld.getInverse();
        nativeRevoluteMotorExpectedAngularMomentum =
            axis.dot(
                inertiaA.transform(bodies[bodyA].angularVelocity) *
                    motor.motorGearRatio +
                inertiaB.transform(bodies[bodyB].angularVelocity));
        conserveNativeRevoluteMotorAngularMomentum =
            physx::PxIsFinite(
                nativeRevoluteMotorExpectedAngularMomentum);
        if (physx::PxAbs(motor.motorGearRatio - 1.0f) <= 1e-6f &&
            motor.anchorA.magnitudeSquared() <= 1e-8f &&
            motor.anchorB.magnitudeSquared() <= 1e-8f) {
          nativeRevoluteMotorExpectedAngularMomentumVector =
              inertiaA.transform(bodies[bodyA].angularVelocity) +
              inertiaB.transform(bodies[bodyB].angularVelocity);
          conserveNativeRevoluteMotorAngularMomentumVector =
              nativeRevoluteMotorExpectedAngularMomentumVector
                  .isFinite();
        }
        const physx::PxReal massA = 1.0f / bodies[bodyA].invMass;
        const physx::PxReal massB = 1.0f / bodies[bodyB].invMass;
        nativeRevoluteMotorExpectedLinearMomentum =
            bodies[bodyA].linearVelocity * massA +
            bodies[bodyB].linearVelocity * massB;
        conserveNativeRevoluteMotorLinearMomentum =
            physx::PxIsFinite(massA) && physx::PxIsFinite(massB) &&
            nativeRevoluteMotorExpectedLinearMomentum.isFinite();
        nativeRevoluteMotorExpectedSpatialAngularMomentum =
            bodies[bodyA].position.cross(
                bodies[bodyA].linearVelocity * massA) +
            inertiaA.transform(bodies[bodyA].angularVelocity) +
            bodies[bodyB].position.cross(
                bodies[bodyB].linearVelocity * massB) +
            inertiaB.transform(bodies[bodyB].angularVelocity);
        conserveNativeRevoluteMotorSpatialMomentum =
            conserveNativeRevoluteMotorLinearMomentum &&
            nativeRevoluteMotorExpectedSpatialAngularMomentum
                .isFinite();
      }
    }
  }
  physx::PxU32 passiveGenericHard1DIndex = PX_MAX_U32;
  const bool passiveGenericHard1DVelocityProjectionIsland =
      isSinglePassiveGenericHard1DVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts,
          passiveGenericHard1DIndex);
  physx::PxVec3 coupledExpectedMomentum(0.0f);
  physx::PxVec3 coupledExpectedAngularMomentum(0.0f);
  bool conserveCoupledLinearPositionSupportAxisMomentum =
      coupledLinearPositionDriveIsland &&
      !coupledLinearPositionDriveFrictionPositionOwnerIsland &&
      (d6Joints[0].anchorA - d6Joints[0].anchorB)
              .magnitudeSquared() >
          1e-12f;
  physx::PxVec3 coupledLinearPositionSupportAxis(0.0f);
  physx::PxReal
      coupledExpectedLinearPositionSupportAxisAngularMomentum = 0.0f;
  if (conserveCoupledLinearPositionSupportAxisMomentum) {
    coupledLinearPositionSupportAxis = contacts[0].contactNormal;
    const physx::PxReal supportAxisLength =
        coupledLinearPositionSupportAxis.normalize();
    const physx::PxU32 bodyA = d6Joints[0].header.bodyIndexA;
    const physx::PxU32 bodyB = d6Joints[0].header.bodyIndexB;
    conserveCoupledLinearPositionSupportAxisMomentum =
        supportAxisLength > 1e-6f &&
        computeTwoBodySupportAxisAngularMomentum(
            bodies[bodyA], bodies[bodyB],
            coupledLinearPositionSupportAxis,
            // A shared support-axis rotation contains both orbital and spin
            // momentum. Apply the angular damping factor uniformly to that
            // complete rigid mode; using the separate linear damping factor
            // on only its orbital part destroys an exact spin/orbit
            // cancellation and injects support-axis angular momentum.
            mConfig.angularDamping, mConfig.angularDamping,
            coupledExpectedLinearPositionSupportAxisAngularMomentum);
  }
  if ((coupledLinearDriveIsland ||
       coupledLinearPositionDriveIsland) &&
      !coupledLinearPositionDriveFrictionPositionOwnerIsland) {
    physx::PxReal totalMass = 0.0f;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal mass = 1.0f / bodies[i].invMass;
      totalMass += mass;
      coupledExpectedMomentum += bodies[i].linearVelocity * mass;
    }
    coupledExpectedMomentum =
        (coupledExpectedMomentum + gravity * (totalMass * dt)) *
        mConfig.velocityDamping;
  }
  if (coupledAngularPositionDriveIsland ||
      coupledSphericalConeIsland) {
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      coupledExpectedAngularMomentum +=
          bodies[i].invInertiaWorld.getInverse() *
          bodies[i].angularVelocity;
    coupledExpectedAngularMomentum *= mConfig.velocityDamping;
  }
  physx::PxArray<bool> touchesKinematicShell(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    touchesKinematicShell[i] = false;
  bool hasKinematicShellContacts = false;
  if (softContacts && numSoftContacts > 0 && softParticles) {
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[sci].geometry;
      const physx::PxU32 bi = geometry.targetIndex;
      if (geometry.hasRigidBodyTarget() &&
          bi < numBodies &&
          avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles)) {
        touchesKinematicShell[bi] = true;
        hasKinematicShellContacts = true;
      }
    }
  }
  static const physx::PxReal kShellFastImpactSpeed =
      AvbdConstants::AVBD_SHELL_FAST_IMPACT_SPEED;

  physx::PxArray<physx::PxVec3> shellLinearVelAtSolveStart;
  if (hasKinematicShellContacts) {
    shellLinearVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      shellLinearVelAtSolveStart[i] = bodies[i].linearVelocity;
  }

  // Island classification for shared post-AL (parity with contact path).
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
  const bool deformableFastImpactIsland =
      allBodyVsStatic && hasDeformableAnchorContact && numD6 == 0 &&
      numGear == 0;

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

  physx::PxArray<physx::PxVec3> linearVelAtSolveStart;
  if (numContacts > 0) {
    linearVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      linearVelAtSolveStart[i] = bodies[i].linearVelocity;
  }
  physx::PxArray<physx::PxVec3> angularVelAtSolveStart;
  if (numContacts > 0 || passiveCenteredGearVelocityProjectionIsland) {
    angularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      angularVelAtSolveStart[i] = bodies[i].angularVelocity;
  }
  physx::PxArray<physx::PxVec3> genericLinearVelAtSolveStart;
  physx::PxArray<physx::PxVec3> genericAngularVelAtSolveStart;
  if (passiveGenericHard1DVelocityProjectionIsland) {
    genericLinearVelAtSolveStart.resize(numBodies);
    genericAngularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      genericLinearVelAtSolveStart[i] = bodies[i].linearVelocity;
      genericAngularVelAtSolveStart[i] = bodies[i].angularVelocity;
    }
  }

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

    // Soft particle prediction.  The native Scene provider precomputes this
    // only when it needs the pose for contact preparation; all other solver
    // users retain the local prediction authority.
    if (!hasPreparedSoftPrediction) {
      for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
        softParticles[i].computePrediction(dt, gravity);
    }
  }

  // Split world-static ownership before any positional contact iteration.
  // The normal remains in Position-AL; tangential Coulomb response is rebuilt
  // from final velocities so an expanded collision-proxy row cannot pin the
  // whole deformable component through a persistent position spring.
  avbdAssignVelocityTangentOwners(
      softContacts, numSoftContacts, softBodies, numSoftBodies,
      softParticles, numSoftParticles);

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
        if (touchesKinematicShell[i]) {
          const bool fastImpact =
              bodies[i].linearVelocity.magnitude() > kShellFastImpactSpeed;
          if (fastImpact)
            bodies[i].position = bodies[i].inertialPosition;
          else
            bodies[i].position =
                bodies[i].prevPosition + bodies[i].linearVelocity * dt;
          bodies[i].rotation = bodies[i].inertialRotation;
        } else {
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
        bodies[i].projectLockedPose(bodies[i].prevPosition,
                                    bodies[i].prevRotation);
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

    // Soft body AVBD warmstart. Equality multipliers and their penalties
    // remain paired when carried across timesteps.
    for (physx::PxU32 sbi = 0; sbi < numSoftBodies; ++sbi) {
      AvbdSoftBody &sb = softBodies[sbi];
      PX_ASSERT(sb.runtime.isObjectiveProgramCurrent(
          sb.compiled.particleStart, sb.compiled.particleCount));
      for (physx::PxU32 oi = 0;
           oi < sb.runtime.compiledObjectives.size(); ++oi) {
        const AvbdCompiledSoftObjective &objective =
            sb.runtime.compiledObjectives[oi];
        switch (objective.owner) {
        case AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL:
        case AvbdSoftObjectiveOwner::
            eARTICULATION_ATTACHMENT_POSITION_AL:
        case AvbdSoftObjectiveOwner::
            eSOFT_PAIR_ATTACHMENT_POSITION_AL:
          avbdWarmstartAttachmentState(
              sb.runtime.attachments[objective.runtimeStateIndex],
              mConfig.avbdAlpha, mConfig.avbdGamma, 1e3f);
          break;
        case AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL:
        case AvbdSoftObjectiveOwner::
            eKINEMATIC_ATTACHMENT_POSITION_AL:
          avbdWarmstartPinState(
              sb.runtime.pins[objective.runtimeStateIndex],
              mConfig.avbdAlpha, mConfig.avbdGamma, 1e3f);
          break;
        case AvbdSoftObjectiveOwner::
            eDEFORMABLE_KINEMATIC_POSITION_AL:
          // A public deformable kinematic target is a persistent prescribed
          // particle equality. Decaying its multiplier or penalty each
          // timestep leaves a load-dependent steady-state position error.
          avbdWarmstartPinState(
              sb.runtime.pins[objective.runtimeStateIndex],
              1.0f, 1.0f, 1.0e8f);
          break;
        default:
          PX_ASSERT(false);
          break;
        }
      }
    }

    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      AvbdSoftContact &sc = softContacts[sci];
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      AvbdSoftContactAugmentedState &state = sc.state;
      if (avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles)) {
        state.alLambda *= mConfig.avbdAlpha * mConfig.avbdGamma;
        state.k = physx::PxMax(
            1000.0f, physx::PxMin(state.ke, state.k * mConfig.avbdGamma));
        for (int ti = 0; ti < 2; ++ti) {
          state.alLambdaTangent[ti] *=
              mConfig.avbdAlpha * mConfig.avbdGamma;
          state.penTangent[ti] = physx::PxMax(
              1000.0f, physx::PxMin(mConfig.avbdPenaltyMax,
                                    state.penTangent[ti] * mConfig.avbdGamma));
        }
        continue;
      }
      // Current-pose dynamic soft/rigid SDF and OGC-feature rows are
      // prepared with a 1e5 normal penalty.  Do not silently demote them to
      // the static-shell 1e4 baseline here: under a large compressive load
      // that permits the rigid to enter the volume before the coupled AL
      // system can distribute displacement through the soft support, after
      // which the emergency whole-body escape path looks like a launch.
      // Preserve the prepared dynamic normal scale (still bounded by ke);
      // ordinary world-static shell rows retain the softer 1e4 baseline.
      const bool dynamicRigidOgc =
          geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
          geometry.hasRigidBodyTarget() &&
          geometry.targetIndex < numBodies &&
          bodies[geometry.targetIndex].invMass > 0.0f;
      state.k = physx::PxMin(dynamicRigidOgc ? 1e5f : 1e4f, state.ke);
      if (geometry.hasRigidBodyTarget() &&
          geometry.targetIndex < numBodies) {
        const AvbdSolverBody &rb = bodies[geometry.targetIndex];
        if (rb.invMass > 0.0f) {
          const physx::PxReal mass = 1.0f / rb.invMass;
          const physx::PxReal floor = 0.25f * mass * invDt2;
          state.k =
              physx::PxMax(physx::PxMax(state.k, floor), 1000.0f);
        }
      }
    }
  }

  // Establish one feasible endpoint epoch before pair-state construction.
  // The following local AVBD updates are then admitted by the same shared
  // pair state instead of beginning from a rigid already inside a soft mesh.
  applyWorldStaticOgcInitialAdmission(
      softParticles, numSoftParticles, softBodies, numSoftBodies,
      softContacts, numSoftContacts);
  physx::PxArray<physx::PxU8> mixedOgcAdmissionContacts;
  physx::PxArray<physx::PxReal> mixedOgcAdmissionDisplacements;
  applyOgcMixedWarmstartAdmission(
      bodies, numBodies, softParticles, numSoftParticles, softBodies,
      numSoftBodies, softContacts, numSoftContacts,
      &mixedOgcAdmissionContacts, &mixedOgcAdmissionDisplacements);

  // Initialize one shared OGC pair epoch after warmstart poses are current.
  // A static target has no 6DOF response, but it still owns the same positive
  // current-pose gap budget as a dynamic rigid.  Excluding it here lets a
  // material update cross the ground or a world box before post-AL repair.
  Dy::AvbdOgcPairState *mixedOgcPairStates = nullptr;
  physx::PxU32 numMixedOgcPairStates = 0;
  AvbdOgcPairTrustRegionContext mixedOgcPairContext;
  if (useProvidedSoftExecutionPlan &&
      softExecutionPlan->hasMixedOgcPairPlan(numSoftContacts)) {
    mixedOgcPairStates = softExecutionPlan->ogcPairStates;
    numMixedOgcPairStates = softExecutionPlan->numOgcPairStates;
    for (physx::PxU32 pairIndex = 0; pairIndex < numMixedOgcPairStates;
         ++pairIndex) {
      AvbdOgcPairState &pair = mixedOgcPairStates[pairIndex];
      // Pair identity is selection-owned and survives across solve calls;
      // only the DCD epoch values are per-step mutable.
      pair.representativeContact = PX_MAX_U32;
      pair.admissionContact = PX_MAX_U32;
      pair.triangleCoreFace = PX_MAX_U8;
      pair.triangleCoreFaceExit = 0.0f;
      pair.hasTriangleCoreManifold = false;
      pair.triangleCoreLocallyResolved = false;
      pair.representativeNormal = physx::PxVec3(0.0f);
      pair.representativeRigidOffset = physx::PxVec3(0.0f);
	  pair.referenceRelativePoint = physx::PxVec3(0.0f);
      pair.representativeGap = PX_MAX_F32;
      pair.accumulatedNormalLambda = 0.0f;
      pair.admittedNormalDisplacement = 0.0f;
      pair.admittedNormalLoad = 0.0f;
      // Positive separations must remain positive in the epoch state.  Zero
      // is a real boundary value, not an "unset" sentinel: using it here
      // schedules a terminal DCD rebuild for every merely nearby pair.
      pair.safetyGap = PX_MAX_F32;
      pair.remainingSafeDisplacement = 0.0f;
      pair.accumulatedRelativeDisplacement = 0.0f;
      pair.referenceGap = PX_MAX_F32;
      pair.minimumGap = PX_MAX_F32;
      ++pair.epoch;
      pair.active = false;
      pair.admittedAtBoundary = false;
      pair.refreshRequested = false;
    }
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const physx::PxU32 pairIndex = softExecutionPlan->ogcPairIndices[sci];
      if (pairIndex >= numMixedOgcPairStates)
        continue;
      AvbdOgcPairState &pair = mixedOgcPairStates[pairIndex];
      const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
      const bool dynamicRigid =
          geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
          geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies &&
          bodies[geometry.targetIndex].invMass > 0.0f;
      const bool worldStatic =
          (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
           geometry.source.type == AvbdSoftContactSource::eGROUND) &&
          geometry.hasWorldStaticTarget();
      if ((!dynamicRigid && !worldStatic) ||
		  pair.sourceType != geometry.source.type ||
		  pair.targetKind != geometry.targetKind ||
		  pair.sourceBodyIndex != geometry.queryBodyIndex ||
          pair.targetBodyIndex != geometry.targetIndex ||
          pair.primitiveKey != geometry.source.primitiveKey ||
          !avbdHasSoftContactDynamicQuerySupport(
              geometry, softParticles, numSoftParticles))
        continue;
      const physx::PxVec3 queryPoint =
          avbdGetSoftContactQueryPoint(geometry, softParticles);
      const physx::PxReal normalLengthSq = geometry.normal.magnitudeSquared();
      if (!queryPoint.isFinite() || !physx::PxIsFinite(normalLengthSq) ||
          normalLengthSq <= 1.0e-12f)
        continue;
      const physx::PxVec3 normal =
          geometry.normal * physx::PxRecipSqrt(normalLengthSq);
      physx::PxReal gap = 0.0f;
      physx::PxVec3 surfacePoint(0.0f);
      physx::PxVec3 rigidOffset(0.0f);
      if (dynamicRigid) {
        const AvbdSolverBody &rigid = bodies[geometry.targetIndex];
        rigidOffset = rigid.rotation.rotate(geometry.rigidLocalPoint);
        surfacePoint = rigid.position + rigidOffset;
        gap = (queryPoint - surfacePoint).dot(normal);
      } else {
        if (!queryCurrentWorldStaticPairSignedDistance(
                geometry, queryPoint, gap))
          continue;
        surfacePoint = geometry.hasRigidBoxSdf
            ? geometry.rigidBoxPose.p : geometry.surfacePoint;
      }
      if (!physx::PxIsFinite(gap))
        continue;
      if (!pair.active || gap < pair.referenceGap)
        pair.referenceGap = gap;
      if (pair.representativeContact == PX_MAX_U32 ||
          gap < pair.representativeGap) {
        pair.representativeContact = sci;
        pair.representativeNormal = normal;
        pair.representativeRigidOffset = rigidOffset;
		pair.referenceRelativePoint = queryPoint - surfacePoint;
        pair.representativeGap = gap;
      }
      if (pair.representativeContact == sci || gap < pair.safetyGap)
        pair.safetyGap = gap;
      if (mixedOgcAdmissionContacts.size() == numSoftContacts &&
          mixedOgcAdmissionContacts[sci] != 0u) {
        pair.admissionContact = sci;
        pair.admittedAtBoundary = true;
        if (mixedOgcAdmissionDisplacements.size() == numSoftContacts)
          pair.admittedNormalDisplacement = physx::PxMax(
              pair.admittedNormalDisplacement,
              mixedOgcAdmissionDisplacements[sci]);
      }
      pair.minimumGap = pair.safetyGap;
      pair.active = true;
      pair.remainingSafeDisplacement = physx::PxMax(pair.safetyGap, 0.0f);
      if (pair.admittedAtBoundary && pair.admittedNormalDisplacement > 0.0f)
        pair.admittedNormalLoad = physx::PxMax(
            pair.admittedNormalLoad,
            pair.admittedNormalDisplacement *
                physx::PxMin(1.0e5f, physx::PxMax(1.0e3f, softContacts[sci].state.k)));
      // A static triangle-core row is an actual discrete triangle/OBB
      // intersection, not mere shell proximity.  Request one terminal
      // same-time verification so material deformation after this prediction
      // cannot leave an unobserved core through the static target.
      if (worldStatic && geometry.hasRigidBoxTriangleCoreExit)
        pair.refreshRequested = true;
    }
    mixedOgcPairContext.pairStates = mixedOgcPairStates;
    mixedOgcPairContext.numPairStates = numMixedOgcPairStates;
    mixedOgcPairContext.contactPairIndices =
        softExecutionPlan->ogcPairIndices;
    mixedOgcPairContext.numContactPairIndices =
        softExecutionPlan->numOgcPairIndices;

  }


  // maxDepenetrationVelocity is a contact-bias policy. Build one shifted
  // normal target from the frame-start relative gap after prediction state
  // has been captured, and keep that target through all nonlinear sweeps.
  avbdResetSoftContactDepenetrationLimits(
      softContacts, numSoftContacts);
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    AvbdSoftContact &contact = softContacts[sci];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    physx::PxVec3 initialSurfacePoint =
        avbdGetSoftContactInitialSurfacePoint(
            geometry, softParticles);
    if (geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numBodies) {
      const AvbdSolverBody &rigidBody =
          bodies[geometry.targetIndex];
      initialSurfacePoint =
          rigidBody.prevPosition +
          rigidBody.prevRotation.rotate(
              geometry.rigidLocalPoint);
    }
    avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
        contact, softParticles, softBodies, numSoftBodies,
        initialSurfacePoint, dt);
  }

  // If a non-CCD current-pose endpoint first appears deeply inside a static
  // target, move the entire unconstrained source body coherently before the
  // nonlinear material sweep.  This preserves its tet F exactly and avoids
  // asking a local support correction to repair an inversion it has not yet
  // created.  Ordinary shell contacts remain exclusively Position-AL owned.
  if (hasCompleteSoftSelection && numSoftContacts > 0) {
    endpointRecoveredWorldStaticBodies.resize(numSoftBodies);
    endpointWorldStaticRecoveryNormals.resize(numSoftBodies);
    physx::PxArray<physx::PxVec3> endpointWorldStaticRecoveryTranslations;
    endpointWorldStaticRecoveryTranslations.resize(numSoftBodies);
    for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies;
         ++bodyIndex) {
      endpointRecoveredWorldStaticBodies[bodyIndex] = 0u;
      endpointWorldStaticRecoveryNormals[bodyIndex] = physx::PxVec3(0.0f);
      endpointWorldStaticRecoveryTranslations[bodyIndex] =
          physx::PxVec3(0.0f);
    }
    // Resolve fresh triangle/box cores through their collision support.  This
    // is deliberately local: a world-static target must resist a falling
    // volume through deformation, never by translating the full body out of
    // contact and thereby cancelling gravity for the whole volume.
    PX_PROFILE_ZONE("AVBD.worldStaticTriangleCoreLocalManifold", 0);
    applyWorldStaticTriangleCoreLocalManifold(
        softParticles, numSoftParticles, softBodies, numSoftBodies,
        softContacts, numSoftContacts, 1u, &stats);

		// Begin fresh triangle cores in the local shared manifold.  It couples the
		// dynamic rigid to the same support rows but never translates an entire
		// soft body/rigid pair, which would turn compression into rigid ejection.
		endpointRecoveredDynamicSoftRigidContacts.resize(numSoftContacts);
		for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
			endpointRecoveredDynamicSoftRigidContacts[sci] = 0u;
		PX_PROFILE_ZONE("AVBD.endpointDynamicSoftRigidTranslation", 0);
		applyDynamicSoftRigidBodyEndpointTranslations(
            bodies, numBodies, softParticles, numSoftParticles, softBodies,
            numSoftBodies, softContacts, numSoftContacts, 4u,
            &endpointRecoveredDynamicSoftRigidContacts,
            &endpointWorldStaticRecoveryTranslations,
            /*allowFreshTriangleCoreExit=*/true,
            /*preferLocalTriangleCoreManifold=*/true,
            /*allowCoherentEndpointFallback=*/false, &stats,
            mixedOgcPairStates, numMixedOgcPairStates,
            useProvidedSoftExecutionPlan
                ? softExecutionPlan->ogcPairIndices : nullptr,
            useProvidedSoftExecutionPlan
                ? softExecutionPlan->numOgcPairIndices : 0u,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->ogcPairContactStarts : nullptr,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->numOgcPairContactStarts : 0u,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->ogcPairContactRefs : nullptr,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->numOgcPairContactRefs : 0u);

		// The paired dynamic correction can move the same local soft support
		// towards a static target. Re-evaluate the cached triangle vertices and
		// project only their deepest support point back through the local static
		// manifold before material iterations begin.
		PX_PROFILE_ZONE("AVBD.worldStaticTriangleCoreLocalManifold", 0);
		applyWorldStaticTriangleCoreLocalManifold(
            softParticles, numSoftParticles, softBodies, numSoftBodies,
            softContacts, numSoftContacts, 1u, &stats);

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
        penScale = AvbdConstants::AVBD_PEN_SCALE_DYN_DYN;
      } else {
        // Body-vs-static: match contact-path floor scale (Phase 1 parity)
        effectiveMass = physx::PxMax(augA, augB);
        penScale = AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC;
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
      if (hasDeformableStaticAnchor(contacts[c])) {
        // Moving mesh anchor: no alpha-soften on normals (match solve()).
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

  // Scene owns the compiled support program whenever the island came from a
  // soft selection.  The early validation above keeps its particle ownership
  // and contact CSR safe to reuse here; standalone and malformed-provider
  // callers build the same source-ordered representation locally.
  physx::PxArray<physx::PxU32> localParticleBodyIndices;
  physx::PxArray<physx::PxU8> localParticleBodyConflicts;
  physx::PxArray<physx::PxU32> localSoftContactStarts;
  physx::PxArray<physx::PxU32> localSoftContactCounts;
  physx::PxArray<AvbdSoftContactParticleRef> localSoftContactRefs;
  physx::PxArray<physx::PxU32> localTriangleCoreSafetyStarts;
  physx::PxArray<physx::PxU32> localTriangleCoreSafetyCounts;
  physx::PxArray<AvbdSoftContactParticleRef> localTriangleCoreSafetyRefs;
  const physx::PxU32 *softParticleBodyIndices = nullptr;
  const physx::PxU32 *softContactStarts = nullptr;
  const AvbdSoftContactParticleRef *softContactRefs = nullptr;
  const physx::PxU32 *triangleCoreSafetyStarts = nullptr;
  const AvbdSoftContactParticleRef *triangleCoreSafetyRefs = nullptr;
  physx::PxU32 numTriangleCoreSafetyStarts = 0;
  physx::PxU32 numTriangleCoreSafetyRefs = 0;
  const physx::PxU32 *rigidTargetContactStarts = nullptr;
  const physx::PxU32 *rigidTargetContactRefs = nullptr;

  if (useProvidedSoftExecutionPlan) {
    softParticleBodyIndices = softExecutionPlan->particleBodyIndices;
    softContactStarts = softExecutionPlan->contactStarts;
    softContactRefs = softExecutionPlan->contactRefs;
    if (softExecutionPlan->hasTriangleCoreSafetyPlan(numSoftParticles)) {
      triangleCoreSafetyStarts =
          softExecutionPlan->triangleCoreSafetyStarts;
      numTriangleCoreSafetyStarts =
          softExecutionPlan->numTriangleCoreSafetyStarts;
      triangleCoreSafetyRefs = softExecutionPlan->triangleCoreSafetyRefs;
      numTriangleCoreSafetyRefs =
          softExecutionPlan->numTriangleCoreSafetyRefs;
    }
  } else {
    localParticleBodyIndices.resize(numSoftParticles);
    localParticleBodyConflicts.resize(numSoftParticles);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex) {
      localParticleBodyIndices[particleIndex] = PX_MAX_U32;
      localParticleBodyConflicts[particleIndex] = 0;
    }
    for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies;
         ++bodyIndex) {
      const AvbdSoftBody &softBody = softBodies[bodyIndex];
      const physx::PxU32 particleStart =
          softBody.compiled.particleStart;
      const physx::PxU32 particleCount =
          softBody.compiled.particleCount;
      if (particleStart > numSoftParticles ||
          particleCount > numSoftParticles - particleStart)
        continue;
      for (physx::PxU32 localParticleIndex = 0;
           localParticleIndex < particleCount; ++localParticleIndex) {
        const physx::PxU32 particleIndex =
            particleStart + localParticleIndex;
        // Overlapping body ranges are invalid provider input.  Keep that
        // state sticky: a third overlapping range must not make a particle
        // look unowned and silently claim it again.
        if (localParticleBodyConflicts[particleIndex] ||
            localParticleBodyIndices[particleIndex] != PX_MAX_U32) {
          localParticleBodyConflicts[particleIndex] = 1;
          localParticleBodyIndices[particleIndex] = PX_MAX_U32;
          continue;
        }
        localParticleBodyIndices[particleIndex] = bodyIndex;
      }
    }

    // Build the same per-particle contact adjacency used by the standalone
    // soft component.  Scanning every soft contact from every particle block
    // makes a resting surface O(particles * contacts * iterations), which is
    // effectively quadratic when most vertices touch a support.  Preserve
    // contact order in each particle range so this remains numerically
    // equivalent to the former full scan.
    localSoftContactStarts.resize(numSoftParticles + 1);
    localSoftContactCounts.resize(numSoftParticles);
    for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
      localSoftContactCounts[i] = 0;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleIndexCount =
          avbdCollectSoftContactParticleIndices(
              geometry, particleIndices);
      for (physx::PxU32 i = 0; i < particleIndexCount; ++i) {
        const physx::PxU32 particleIndex = particleIndices[i];
        if (particleIndex >= numSoftParticles)
          continue;
        if (physx::PxAbs(avbdGetSoftContactParticleJacobianScale(
                geometry, particleIndex)) > 1e-12f)
          localSoftContactCounts[particleIndex]++;
      }
    }
    localSoftContactStarts[0] = 0;
    for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
      localSoftContactStarts[i + 1] =
          localSoftContactStarts[i] + localSoftContactCounts[i];
    localSoftContactRefs.resize(
        localSoftContactStarts[numSoftParticles]);
    for (physx::PxU32 i = 0; i < numSoftParticles; ++i)
      localSoftContactCounts[i] = 0;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleIndexCount =
          avbdCollectSoftContactParticleIndices(
              geometry, particleIndices);
      for (physx::PxU32 i = 0; i < particleIndexCount; ++i) {
        const physx::PxU32 particleIndex = particleIndices[i];
        if (particleIndex >= numSoftParticles)
          continue;
        const physx::PxReal jacobianScale =
            avbdGetSoftContactParticleJacobianScale(
                geometry, particleIndex);
        if (physx::PxAbs(jacobianScale) <= 1e-12f)
          continue;
        localSoftContactRefs[
            localSoftContactStarts[particleIndex] +
            localSoftContactCounts[particleIndex]++] =
                AvbdSoftContactParticleRef(
                    contactIndex, jacobianScale);
      }
    }
    softParticleBodyIndices = localParticleBodyIndices.begin();
    softContactStarts = localSoftContactStarts.begin();
    softContactRefs = localSoftContactRefs.begin();
  }

  // Direct/native callers that do not supply the Scene-owned geometry CSR
  // compile the same compact core-support program locally.  This is separate
  // from the AL particle CSR above: each contact appears at most once per
  // simulation particle even when a proxy triangle's three embeddings share
  // a tet vertex.
  if (!triangleCoreSafetyStarts) {
    const physx::PxU32 maxCoreSafetyParticles =
        3u * AVBD_CONTACT_POINT_MAX_SUPPORT;
    auto collectTriangleCoreSafetyParticles =
        [&](const AvbdSoftContactGeometry &geometry,
            physx::PxU32 *particleIndices) -> physx::PxU32 {
      if (!geometry.hasRigidBoxTriangleCoreExit ||
          !geometry.hasRigidBodyTarget())
        return 0;
      physx::PxU32 count = 0;
      for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
        const AvbdWeightedContactPoint &mapping =
            geometry.rigidBoxTriangleCorePoints[vertex];
        if (mapping.count == 0 ||
            mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
          return PX_MAX_U32;
        for (physx::PxU32 support = 0; support < mapping.count; ++support) {
          const physx::PxU32 particleIndex =
              mapping.particleIndices[support];
          if (particleIndex >= numSoftParticles)
            return PX_MAX_U32;
          bool duplicate = false;
          for (physx::PxU32 prior = 0; prior < count; ++prior)
            duplicate |= particleIndices[prior] == particleIndex;
          if (!duplicate) {
            if (count >= maxCoreSafetyParticles)
              return PX_MAX_U32;
            particleIndices[count++] = particleIndex;
          }
        }
      }
      return count;
    };

    localTriangleCoreSafetyStarts.resize(numSoftParticles + 1);
    localTriangleCoreSafetyCounts.resize(numSoftParticles);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      localTriangleCoreSafetyCounts[particleIndex] = 0;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      physx::PxU32 particleIndices[3 * AVBD_CONTACT_POINT_MAX_SUPPORT];
      const physx::PxU32 particleCount = collectTriangleCoreSafetyParticles(
          softContacts[contactIndex].geometry, particleIndices);
      if (particleCount == PX_MAX_U32)
        continue;
      for (physx::PxU32 index = 0; index < particleCount; ++index)
        localTriangleCoreSafetyCounts[particleIndices[index]]++;
    }
    localTriangleCoreSafetyStarts[0] = 0;
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      localTriangleCoreSafetyStarts[particleIndex + 1] =
          localTriangleCoreSafetyStarts[particleIndex] +
          localTriangleCoreSafetyCounts[particleIndex];
    localTriangleCoreSafetyRefs.resize(
        localTriangleCoreSafetyStarts[numSoftParticles]);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      localTriangleCoreSafetyCounts[particleIndex] = 0;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      physx::PxU32 particleIndices[3 * AVBD_CONTACT_POINT_MAX_SUPPORT];
      const physx::PxU32 particleCount =
          collectTriangleCoreSafetyParticles(
              softContacts[contactIndex].geometry, particleIndices);
      if (particleCount == PX_MAX_U32)
        continue;
      for (physx::PxU32 index = 0; index < particleCount; ++index) {
        const physx::PxU32 particleIndex = particleIndices[index];
        localTriangleCoreSafetyRefs[
            localTriangleCoreSafetyStarts[particleIndex] +
            localTriangleCoreSafetyCounts[particleIndex]++] =
            AvbdSoftContactParticleRef(contactIndex, 1.0f);
      }
    }
    triangleCoreSafetyStarts = localTriangleCoreSafetyStarts.begin();
    numTriangleCoreSafetyStarts = localTriangleCoreSafetyStarts.size();
    triangleCoreSafetyRefs = localTriangleCoreSafetyRefs.begin();
    numTriangleCoreSafetyRefs = localTriangleCoreSafetyRefs.size();
  }

  if (useProvidedRigidTargetContactPlan) {
    rigidTargetContactStarts = softExecutionPlan->rigidTargetContactStarts;
    rigidTargetContactRefs = softExecutionPlan->rigidTargetContactRefs;
  }

  if (mixedOgcPairContext.isComplete(numSoftContacts) &&
      triangleCoreSafetyStarts &&
      numTriangleCoreSafetyStarts == numSoftParticles + 1 &&
      (numTriangleCoreSafetyRefs == 0 || triangleCoreSafetyRefs) &&
      triangleCoreSafetyStarts[0] == 0 &&
      triangleCoreSafetyStarts[numSoftParticles] ==
          numTriangleCoreSafetyRefs) {
    mixedOgcPairContext.triangleCoreSafetyStarts =
        triangleCoreSafetyStarts;
    mixedOgcPairContext.numTriangleCoreSafetyStarts =
        numTriangleCoreSafetyStarts;
    mixedOgcPairContext.triangleCoreSafetyRefs = triangleCoreSafetyRefs;
    mixedOgcPairContext.numTriangleCoreSafetyRefs =
        numTriangleCoreSafetyRefs;
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

    // Chebyshev semi-iterative state
    // A component-clamped SLERP drive is non-smooth at each row limit.
    // Chebyshev extrapolation assumes a smooth stationary iteration and can
    // repeatedly overshoot that active set, losing drive authority at coarse
    // timesteps.  Keep the scoped physical SLERP island on plain block descent.
    bool hasDynamicSoftRigidContact = false;
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
      if (geometry.hasRigidBodyTarget() &&
          geometry.targetIndex < numBodies &&
          bodies[geometry.targetIndex].invMass > 0.0f) {
        hasDynamicSoftRigidContact = true;
        break;
      }
    }
    // A unilateral soft-rigid manifold changes its active row set as the
    // bodies separate and re-enter the OGC shell.  Chebyshev acceleration
    // assumes a smooth stationary iteration; extrapolating across that active
    // set can cross the inertial minimizer after the contact has gone inactive
    // and give the rigid endpoint motion with the wrong sign.  The per-row OGC
    // trust-region only admits the local solve candidate, so it cannot bound a
    // later whole-island extrapolation.  Keep mixed dynamic contact islands on
    // the stable block Gauss-Seidel schedule.
    const bool useChebyshev =
        !slerpVelocityDriveIsland && !coupledFixedD6Island &&
        !coupledSphericalConeIsland &&
        !coupledSpatialTendonIsland &&
        !hasDynamicSoftRigidContact &&
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

    const physx::PxU32 baseIterations =
        physx::PxMax(mConfig.iterations, iterationOverride);
    const bool hasExplicitJointConstraints = numD6 > 0 || numGear > 0;
    const physx::PxU32 jointIterations =
        hasExplicitJointConstraints && mConfig.jointIterationOverride > 0
            ? physx::PxMax(baseIterations,
                           mConfig.jointIterationOverride)
            : baseIterations;
    const physx::PxU32 minIterations =
        hasExplicitJointConstraints && mConfig.jointIterationOverride > 0
            ? physx::PxMin(jointIterations,
                           mConfig.jointIterationOverride)
            : physx::PxMin(jointIterations, physx::PxU32(4));
    const bool enableEarlyStop =
        mConfig.enableEarlyStop && !hasCompleteSoftSelection &&
        jointIterations - minIterations > 1;
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

    // The packet material backend is a relaxed production fast path: it may
    // use ISA-specific arithmetic and therefore is intentionally not part of
    // the ordered/enhanced-determinism authority.  The material program is
    // immutable for this island solve, so choose its backend once rather than
    // scanning the body list from every primal sweep.
    const AvbdCpuIsaCorotationalTetPacket8Fn corotationalTetPacketKernel =
        numSoftParticles > 0 && numSoftBodies > 0 &&
                !mConfig.requiresOrderedBackend()
            ? avbdSelectCorotationalTetPacketKernel(softBodies,
                                                     numSoftBodies)
            : NULL;

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

        bool coupledSolved = false;
        if (coupledFixedD6Island) {
          coupledSolved = solveCoupledFixedD6Island(
              bodies, numBodies, d6Joints[0], invDt2);
          if (!coupledSolved) {
            coupledFixedD6Island = false;
            fallbackAvbdJointObjective(
                d6Joints[0].objectiveProgram,
                AvbdJointObjectiveKind::CoupledFixedD6);
          }
        } else if (coupledSphericalConeIsland) {
          coupledSolved = solveCoupledSphericalConeIsland(
              bodies, numBodies, d6Joints[0], invDt2);
          if (!coupledSolved) {
            coupledSphericalConeIsland = false;
            fallbackAvbdJointObjective(
                d6Joints[0].objectiveProgram,
                AvbdJointObjectiveKind::
                    CoupledSphericalCone);
          }
        } else if (coupledLinearPositionDriveIsland) {
          coupledSolved = solveCoupledLinearPositionDriveIsland(
              bodies, numBodies, contacts, numContacts, d6Joints[0], dt,
              invDt2, gravity, mConfig);
          if (!coupledSolved) {
            coupledLinearPositionDriveIsland = false;
            coupledLinearPositionDriveFrictionPositionOwnerIsland =
                false;
            fallbackAvbdJointObjective(
                d6Joints[0].objectiveProgram,
                AvbdJointObjectiveKind::
                    CoupledLinearPositionDrive);
          }
        } else if (coupledLinearDriveIsland) {
          coupledSolved = solveCoupledLinearDriveIsland(
              bodies, numBodies, contacts, numContacts, d6Joints[0], dt,
              invDt2, mConfig);
          if (!coupledSolved) {
            coupledLinearDriveIsland = false;
            fallbackAvbdJointObjective(
                d6Joints[0].objectiveProgram,
                AvbdJointObjectiveKind::
                    CoupledLinearVelocityDrive);
          }
        } else if (coupledAngularPositionDriveIsland) {
          coupledSolved = solveCoupledAngularPositionDriveIsland(
              bodies, numBodies, contacts, numContacts, d6Joints[0], dt,
              invDt2, mConfig);
          if (!coupledSolved) {
            coupledAngularPositionDriveIsland = false;
            fallbackAvbdJointObjective(
                d6Joints[0].objectiveProgram,
                AvbdJointObjectiveKind::
                    CoupledAngularPositionDrive);
          }
        }

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

        auto solveBody = [&](physx::PxU32 idx) {
          const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
          if (bodies[i].invMass <= 0.0f)
            return;
          solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                     numContacts, d6Joints, numD6, gearJoints,
                                     numGear, dt, invDt2, contactMap, d6Map,
                                     gearMap, softParticles, numSoftParticles,
                                     softContacts, numSoftContacts,
                                     softBodies, numSoftBodies,
                                     rigidTargetContactStarts,
                                     rigidTargetContactRefs,
                                     mixedOgcPairContext.isComplete(
                                         numSoftContacts)
                                         ? &mixedOgcPairContext : nullptr);
        };

        if (coupledSolved) {
          // Both endpoints were updated from one frozen island objective.
        } else {
          for (physx::PxU32 idx = 0; idx < numBodies; ++idx)
            solveBody(idx);
        }

        if (coupledSpatialTendonIsland) {
          bool tendonSolved = true;
          for (physx::PxU32 row = 0;
               row < coupledSpatialTendonRowIndices.size();
               ++row) {
            const physx::PxU32 rowIndex =
                coupledSpatialTendonRowIndices[row];
            tendonSolved =
                solveCoupledSpatialTendonRow(
                    bodies, numBodies, d6Joints[rowIndex], dt,
                    invDt2) &&
                tendonSolved;
          }
          if (!tendonSolved) {
            coupledSpatialTendonIsland = false;
            for (physx::PxU32 row = 0;
                 row < coupledSpatialTendonRowIndices.size();
                 ++row) {
              fallbackAvbdJointObjective(
                  d6Joints[coupledSpatialTendonRowIndices[row]]
                      .objectiveProgram,
                  AvbdJointObjectiveKind::CoupledSpatialTendon);
            }
          }
        }

        // --- VBD soft particle primal (3x3 block coordinate descent) ---
        if (numSoftParticles > 0 && numSoftBodies > 0) {
          PX_PROFILE_ZONE("AVBD.softParticlePrimal", 0);

          auto solveSP = [&](physx::PxU32 spi) {
            if (softParticles[spi].invMass <= 0.0f) return;
            const physx::PxU32 softBodyIndex =
                softParticleBodyIndices[spi];
            if (softBodyIndex >= numSoftBodies)
              return;
            solveSoftParticle(spi, softParticles, numSoftParticles,
                              bodies, numBodies,
                              softBodies[softBodyIndex],
                              softContacts, numSoftContacts,
                              softContactRefs,
                              softContactStarts[spi],
                              softContactStarts[spi + 1],
                              dt, invDt2,
                              corotationalTetPacketKernel,
                              mixedOgcPairContext.isComplete(numSoftContacts)
                                  ? &mixedOgcPairContext : nullptr);
          };

          // The soft-particle primal is nonlinear GS. It deliberately stays
          // serial until P4 supplies a conflict-colored taskgraph schedule;
          // `numBodies` is not a proxy for soft work and no private worker
          // pool may be entered from this stage.
          for (physx::PxU32 spi = 0; spi < numSoftParticles; ++spi)
            solveSP(spi);

          solveSoftRigidAttachmentsCoupled(
              softParticles, numSoftParticles,
              bodies, numBodies,
              softBodies, numSoftBodies, dt,
              articulationForBody, linkIndexForBody);
          solveSoftPairAttachmentsCoupled(
              softParticles, numSoftParticles,
              softBodies, numSoftBodies, dt);
        }

        for (physx::PxU32 i = 0; i < numBodies; ++i) {
          if (bodies[i].invMass > 0.0f)
            bodies[i].projectLockedPose(bodies[i].prevPosition,
                                        bodies[i].prevRotation);
        }
        PX_AVBD_PROFILE_STAT(stats.totalIterations++);
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
                                    dt, stats);

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
            jnt.writebackLinearImpulse = physx::PxVec3(0.0f);
            jnt.writebackLinearImpulseValid = 0;
            jnt.writebackAngularImpulse = physx::PxVec3(0.0f);
            jnt.writebackAngularImpulseValid = 0;
            const physx::PxU32 compiledSourceRows =
                getAvbdJointObjectiveSourceRows(
                    jnt.objectiveProgram);
            const physx::PxU32 compiledDriveRows =
                compiledSourceRows & 0x3fu;
            const bool compiledConeObjective =
                (compiledSourceRows &
                 eJOINT_SOURCE_ANGULAR_CONE) != 0;

            const bool noDualObjective =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationFixedTendon) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationSpatialTendon) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        CoupledSpatialTendon) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationCompliantMimic) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        GenericAccelerationDamping1D) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        GenericRestitution1D);
            if (noDualObjective) {
              jnt.lambdaLinear = physx::PxVec3(0.0f);
              jnt.lambdaAngular = physx::PxVec3(0.0f);
              continue;
            }

            if (hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        GenericForceSpring1D)) {
              if (dt <= 0.0f)
                continue;
              const physx::PxReal C =
                  computeGeneric1DViolation(jnt, bodies, numBodies, dt);
              const physx::PxReal velocity =
                  (C - jnt.genericGeometricError) / dt;
              const physx::PxReal totalForce =
                  jnt.header.rho * C + jnt.header.damping * velocity;
              const physx::PxReal appliedImpulse = physx::PxClamp(
                  -totalForce * dt, jnt.genericMinImpulse,
                  jnt.genericMaxImpulse);
              const bool outputForce =
                  (jnt.genericRowFlags &
                   static_cast<physx::PxU32>(
                       Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;
              jnt.writebackLinearImpulse =
                  outputForce ? jnt.genericLinearA * appliedImpulse
                              : physx::PxVec3(0.0f);
              jnt.writebackAngularImpulse =
                  outputForce
                      ? jnt.genericAngularAWriteback * appliedImpulse
                      : physx::PxVec3(0.0f);
              jnt.writebackLinearImpulseValid = 1;
              jnt.writebackAngularImpulseValid = 1;
              jnt.lambdaLinear = physx::PxVec3(0.0f);
              jnt.lambdaAngular = physx::PxVec3(0.0f);
              continue;
            }

            if (hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::GenericHard1D) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationHardMimic)) {
              const physx::PxReal effectiveMass =
                  computeGeneric1DEffectiveMass(jnt, bodies, numBodies);
              if (effectiveMass <= 0.0f || dt <= 0.0f)
                continue;

              const physx::PxReal Mh2 = effectiveMass * invDt2;
              const physx::PxReal rho = jnt.header.rho;
              const physx::PxReal rhoDual =
                  physx::PxMin(Mh2, rho * rho / (rho + Mh2));
              const physx::PxReal C =
                  computeGeneric1DViolation(jnt, bodies, numBodies, dt);
              const physx::PxReal pen = physx::PxMax(rho, Mh2);
              const physx::PxReal totalForce =
                  pen * C + jnt.lambdaLinear.x;
              const physx::PxReal appliedImpulse = physx::PxClamp(
                  -totalForce * dt, jnt.genericMinImpulse,
                  jnt.genericMaxImpulse);

              const bool outputForce =
                  (jnt.genericRowFlags &
                   static_cast<physx::PxU32>(
                       Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;
              jnt.writebackLinearImpulse =
                  outputForce ? jnt.genericLinearA * appliedImpulse
                              : physx::PxVec3(0.0f);
              jnt.writebackAngularImpulse =
                  outputForce
                      ? jnt.genericAngularAWriteback * appliedImpulse
                              : physx::PxVec3(0.0f);
              jnt.writebackLinearImpulseValid = 1;
              jnt.writebackAngularImpulseValid = 1;

              const physx::PxReal newLambda =
                  jnt.lambdaLinear.x * lambdaDecay + C * rhoDual;
              const physx::PxReal clampedDualImpulse = physx::PxClamp(
                  -newLambda * dt, jnt.genericMinImpulse,
                  jnt.genericMaxImpulse);
              jnt.lambdaLinear =
                  physx::PxVec3(-clampedDualImpulse / dt, 0.0f, 0.0f);
              jnt.lambdaAngular = physx::PxVec3(0.0f);
              continue;
            }

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
            bool hasLockedLinearRow = false;
            bool hasUnsupportedLinearRow = false;
            for (int axis = 0; axis < 3; ++axis) {
              hasLockedLinearRow |= (jnt.getLinearMotion(axis) == 0);
              hasUnsupportedLinearRow |= (jnt.getLinearMotion(axis) == 1) ||
                                         jnt.isLinearDriveEnabled(axis);
            }
            const physx::PxReal mA =
                getBodyMass(jnt.header.bodyIndexA);
            const physx::PxReal mB =
                getBodyMass(jnt.header.bodyIndexB);
            const physx::PxReal pen = physx::PxMax(
                jnt.header.rho, physx::PxMax(mA, mB) * invDt2);
            physx::PxVec3 actor0LinearForce(0.0f);
            physx::PxVec3 actor0PositionDriveForce(0.0f);
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
                // Match the primal row force f = pen*C + lambda, using the
                // pre-update solver multiplier.  The row convention is
                // C=xA-xB, so actor0's public reaction is -axis*f.
                const physx::PxReal totalForce =
                    pen * Ck + jnt.lambdaLinear[axis];
                actor0LinearForce -= linearAxes[axis] * totalForce;
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

            const bool positionDriveActive =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::LinearPositionDrive) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        CoupledLinearPositionDrive);
            if (positionDriveActive) {
              const physx::PxVec3 axis = linearAxes[0];
              const physx::PxVec3 dxA =
                  aStatic ? physx::PxVec3(0.0f)
                          : ((bodies[jnt.header.bodyIndexA].position +
                              rotA.rotate(jnt.anchorA)) -
                             (bodies[jnt.header.bodyIndexA].prevPosition +
                              bodies[jnt.header.bodyIndexA]
                                  .prevRotation.rotate(jnt.anchorA)));
              const physx::PxVec3 dxB =
                  bStatic ? physx::PxVec3(0.0f)
                          : ((bodies[jnt.header.bodyIndexB].position +
                              rotB.rotate(jnt.anchorB)) -
                             (bodies[jnt.header.bodyIndexB].prevPosition +
                              bodies[jnt.header.bodyIndexB]
                                  .prevRotation.rotate(jnt.anchorB)));
              const physx::PxReal positionError =
                  (wB - wA).dot(axis) - jnt.driveLinearPosition.x;
              const physx::PxReal velocityError =
                  (dxB - dxA).dot(axis) / dt -
                  jnt.driveLinearVelocity.x;
              const physx::PxReal driveForce = physx::PxClamp(
                  jnt.linearStiffness.x * positionError +
                      jnt.linearDamping.x * velocityError,
                  -jnt.driveLinearForce.x, jnt.driveLinearForce.x);
              actor0PositionDriveForce = axis * driveForce;
              if ((jnt.driveOutputForceFlags & 0x1u) != 0)
                actor0LinearForce += actor0PositionDriveForce;
            }

            if ((hasLockedLinearRow && !hasUnsupportedLinearRow) ||
                positionDriveActive) {
              // ConstraintWriteback stores impulses; Sc::ConstraintSim turns
              // them back into public force by multiplying by 1/dt.
              jnt.writebackLinearImpulse = actor0LinearForce * dt;
              jnt.writebackLinearImpulseValid = 1;
            }

            // Detect revolute pattern for cross-product axis alignment
            const physx::PxU32 twistMotion_d = jnt.getAngularMotion(0);
            const physx::PxU32 swing1Motion_d = jnt.getAngularMotion(1);
            const physx::PxU32 swing2Motion_d = jnt.getAngularMotion(2);
            const bool isRevolutePattern_d =
                (twistMotion_d != 0) && (swing1Motion_d == 0) &&
                (swing2Motion_d == 0);
            physx::PxVec3 actor0AngularTorque(0.0f);

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

              const physx::PxReal totalTorque1 =
                  pen * err1 + jnt.lambdaAngular[1];
              const physx::PxReal totalTorque2 =
                  pen * err2 + jnt.lambdaAngular[2];
              actor0AngularTorque +=
                  perp1 * totalTorque1 + perp2 * totalTorque2;
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
                if (compiledConeObjective && axis >= 1)
                  continue;
                physx::PxU32 motion = jnt.getAngularMotion(axis);
                if (motion == 2) // FREE
                  continue;

                if (motion == 0) { // LOCKED
                  physx::PxReal angErr =
                      jnt.computeAngularError(rotA, rotB, axis);
                  physx::PxVec3 localAxis(0.0f);
                  localAxis[axis] = 1.0f;
                  const physx::PxVec3 worldAxis =
                      jointFrameA_dual.rotate(localAxis);
                  const physx::PxReal totalTorque =
                      pen * angErr + jnt.lambdaAngular[axis];
                  // C = rotation(A)-rotation(B); actor0/A's public torque is
                  // the negative world row force in this AVBD convention.
                  actor0AngularTorque -= worldAxis * totalTorque;
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

            if (positionDriveActive && jnt.angularMotion == 0) {
              // The finite linear drive acts at the dynamic endpoint's
              // anchor.  All angular rows are locked in this scoped island,
              // so their public reaction is the opposite lever-arm torque.
              // PxConstraint reports linear rows about bodyAWorldOffset;
              // eOUTPUT_FORCE must therefore not add another COM moment.
              const physx::PxVec3 dynamicArm =
                  aStatic
                      ? bodies[jnt.header.bodyIndexB].rotation.rotate(
                            jnt.anchorB)
                      : bodies[jnt.header.bodyIndexA].rotation.rotate(
                            jnt.anchorA);
              actor0AngularTorque =
                  -dynamicArm.cross(actor0PositionDriveForce);
            }

            const bool angularAxisVelocityDriveActive =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        AngularAxisVelocityDrive);
            const bool slerpVelocityDriveActive =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::SlerpVelocityDrive);
            if (angularAxisVelocityDriveActive ||
                slerpVelocityDriveActive) {
              physx::PxVec3 dThetaA = aStatic
                                          ? jnt.externalAngularStepA
                                          : physx::PxVec3(0.0f);
              physx::PxVec3 dThetaB = bStatic
                                          ? jnt.externalAngularStepB
                                          : physx::PxVec3(0.0f);
              if (!aStatic) {
                physx::PxQuat dqA =
                    bodies[jnt.header.bodyIndexA].rotation *
                    bodies[jnt.header.bodyIndexA]
                        .prevRotation.getConjugate();
                if (dqA.w < 0.0f)
                  dqA = -dqA;
                dThetaA =
                    physx::PxVec3(dqA.x, dqA.y, dqA.z) * 2.0f;
              }
              if (!bStatic) {
                physx::PxQuat dqB =
                    bodies[jnt.header.bodyIndexB].rotation *
                    bodies[jnt.header.bodyIndexB]
                        .prevRotation.getConjugate();
                if (dqB.w < 0.0f)
                  dqB = -dqB;
                dThetaB =
                    physx::PxVec3(dqB.x, dqB.y, dqB.z) * 2.0f;
              }
              const physx::PxVec3 worldAngularTarget =
                  jointFrameA_dual.rotate(jnt.driveAngularVelocity) * dt;
              if (slerpVelocityDriveActive) {
                // TGS emits SLERP as three fixed world rows.  The shared
                // scalar limit clamps each row independently and actor0's
                // public torque is their aggregate, independent of which
                // endpoint owns the dynamic body.
                const physx::PxReal targetScale =
                    mConfig.angularDamping > 1e-6f
                        ? 1.0f / mConfig.angularDamping
                        : 1.0f;
                const physx::PxVec3 residual =
                    (dThetaB - dThetaA) - worldAngularTarget * targetScale;
                const physx::PxReal scale = jnt.angularDamping.z / dt;
                physx::PxVec3 driveTorque(0.0f);
                for (int k = 0; k < 3; ++k)
                  (&driveTorque.x)[k] = physx::PxClamp(
                      scale * (&residual.x)[k],
                      -jnt.driveAngularForce.z,
                      jnt.driveAngularForce.z);
                if ((jnt.driveOutputForceFlags & (1u << 5)) != 0)
                  actor0AngularTorque += driveTorque;
              } else {
                const physx::PxU32 driveIndex =
                    compiledDriveRows == (1u << 3)
                        ? 0u
                        : (compiledDriveRows == (1u << 4) ? 1u
                                                         : 2u);
                physx::PxVec3 localDriveAxis(0.0f);
                localDriveAxis[driveIndex] = 1.0f;
                const physx::PxVec3 worldDriveAxis =
                    jointFrameA_dual.rotate(localDriveAxis);
                // TWIST/SWING use wA-wB=target.  With the solver's
                // relDW=dThetaB-dThetaA convention this is
                // C=relDW+target*dt.  Positive physical torque acts on actor0
                // along the actor-A authored drive axis, independent of which
                // endpoint is the dynamic body.
                const physx::PxReal C =
                    (dThetaB - dThetaA).dot(worldDriveAxis) +
                    worldAngularTarget.dot(worldDriveAxis);
                const physx::PxReal driveTorque = physx::PxClamp(
                    (jnt.angularDamping[driveIndex] / dt) * C,
                    -jnt.driveAngularForce[driveIndex],
                    jnt.driveAngularForce[driveIndex]);
                if ((jnt.driveOutputForceFlags &
                     (1u << (3u + driveIndex))) != 0)
                  actor0AngularTorque += worldDriveAxis * driveTorque;
              }
            }

            const bool passiveNativeReaction =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        NativePassiveReaction) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::CoupledFixedD6);
            if ((jnt.angularMotion == 0 &&
                 (compiledDriveRows & 0x38u) == 0) ||
                passiveNativeReaction ||
                angularAxisVelocityDriveActive ||
                slerpVelocityDriveActive) {
              jnt.writebackAngularImpulse = actor0AngularTorque * dt;
              jnt.writebackAngularImpulseValid = 1;
            }

            // --- Cone limit dual update ---
            if (compiledConeObjective) {
              physx::PxVec3 coneAxis(0.0f);
              physx::PxReal coneViol = 0.0f;
              const bool ellipticalCone =
                  computeEllipticalConeConstraint(
                      jnt, rotA, rotB, coneAxis, coneViol);
              PX_UNUSED(coneAxis);
              if (!ellipticalCone) {
                const physx::PxVec3 worldAxisA =
                    (rotA * jnt.localFrameA)
                        .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
                const physx::PxVec3 worldAxisB =
                    (rotB * jnt.localFrameB)
                        .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
                const physx::PxReal dotAB = physx::PxClamp(
                    worldAxisA.dot(worldAxisB), -1.0f, 1.0f);
                const physx::PxReal coneAngle =
                    physx::PxAcos(dotAB);
                coneViol = coneAngle - jnt.coneAngleLimit;
              }

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
            if ((compiledDriveRows & 0x7u) != 0) {
              // Body displacements from start-of-step
              const physx::PxVec3 dxA =
                  aStatic ? physx::PxVec3(0.0f)
                          : (bodies[jnt.header.bodyIndexA].position -
                             bodies[jnt.header.bodyIndexA].prevPosition);
              const physx::PxVec3 dxB =
                  bStatic ? physx::PxVec3(0.0f)
                          : (bodies[jnt.header.bodyIndexB].position -
                             bodies[jnt.header.bodyIndexB].prevPosition);

              for (int a = 0; a < 3; ++a) {
                if ((compiledDriveRows & (1u << a)) == 0)
                  continue;
                const physx::PxReal stiffness =
                    (&jnt.linearStiffness.x)[a];
                const bool usePhysicalVelocityObjective =
                    stiffness <= 0.0f &&
                    ((aStatic || bStatic) ||
                     hasAvbdJointObjective(
                         jnt.objectiveProgram,
                         AvbdJointObjectiveKind::
                             CoupledLinearVelocityDrive));
                const bool usePositionObjective =
                    a == 0 &&
                    (hasAvbdJointObjective(
                         jnt.objectiveProgram,
                         AvbdJointObjectiveKind::
                             LinearPositionDrive) ||
                     hasAvbdJointObjective(
                         jnt.objectiveProgram,
                         AvbdJointObjectiveKind::
                             CoupledLinearPositionDrive));
                if (usePhysicalVelocityObjective || usePositionObjective) {
                  // The scoped force/acceleration objective carries its mass
                  // distinction in the primal penalty, not in an AL dual.
                  (&jnt.lambdaDriveLinear.x)[a] = 0.0f;
                  continue;
                }

                const physx::PxReal damping = (&jnt.linearDamping.x)[a];
                if (damping <= 0.0f)
                  continue;

                physx::PxVec3 localAxis(0.0f);
                (&localAxis.x)[a] = 1.0f;
                const physx::PxVec3 wAxis = jointFrameA.rotate(localAxis);
                const physx::PxVec3 worldTarget =
                    jointFrameA.rotate(jnt.driveLinearVelocity) * dt;
                const physx::PxReal C =
                    (dxB.dot(wAxis) - dxA.dot(wAxis)) -
                    worldTarget.dot(wAxis);
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
                physx::PxReal rhoDualDrive =
                    physx::PxMin(damping / dt2, rhoDual);
                if (jnt.isLinearAccelerationDrive(a)) {
                  const physx::PxReal driveScale =
                      computeLinearDriveRecipResponse(
                          bodyARef, bodyBRef, rAWorld, rBWorld, wAxis);
                  const physx::PxReal dampingOnly =
                      physx::PxMax(0.0f, damping - stiffness);
                  const physx::PxReal implicitScale =
                      1.0f /
                      (1.0f + dt * (dt * stiffness + dampingOnly));
                  rhoDualDrive = physx::PxMin(
                      (damping * driveScale * implicitScale) / dt2,
                      rhoDual);
                }
                (&jnt.lambdaDriveLinear.x)[a] =
                    (&jnt.lambdaDriveLinear.x)[a] * lambdaDecay +
                    rhoDualDrive * C;
              }
            }

            // Angular velocity drive dual
            if ((compiledDriveRows & 0x38u) != 0) {
              // Angular displacements from start-of-step
              physx::PxVec3 dThetaA = aStatic
                                          ? jnt.externalAngularStepA
                                          : physx::PxVec3(0.0f);
              physx::PxVec3 dThetaB = bStatic
                                          ? jnt.externalAngularStepB
                                          : physx::PxVec3(0.0f);
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

              const bool slerpDrive =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::SlerpVelocityDrive) ||
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::SlerpPositionDrive) ||
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::
                          CoupledAngularPositionDrive) ||
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::
                          OrdinaryD6SlerpDrive);
              if (slerpDrive) {
                const bool usePhysicalSlerpVelocityObjective =
                    hasAvbdJointObjective(
                        jnt.objectiveProgram,
                        AvbdJointObjectiveKind::SlerpVelocityDrive);
                const bool usePhysicalSlerpPositionObjective =
                    hasAvbdJointObjective(
                        jnt.objectiveProgram,
                        AvbdJointObjectiveKind::SlerpPositionDrive);
                const bool useCoupledAngularPositionObjective =
                    hasAvbdJointObjective(
                        jnt.objectiveProgram,
                        AvbdJointObjectiveKind::
                            CoupledAngularPositionDrive);
                if (usePhysicalSlerpVelocityObjective ||
                    usePhysicalSlerpPositionObjective ||
                    useCoupledAngularPositionObjective) {
                  // The force-mode objective is solved directly in the
                  // primal path.  An AL multiplier would bypass its authored
                  // per-row torque limit on later iterations.
                  jnt.lambdaDriveAngular = physx::PxVec3(0.0f);
                } else {
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
                  if ((compiledDriveRows &
                       (1u << axes[a].bit)) == 0)
                    continue;
                  const bool usePhysicalAngularAxisVelocityObjective =
                      hasAvbdJointObjective(
                          jnt.objectiveProgram,
                          AvbdJointObjectiveKind::
                              AngularAxisVelocityDrive);
                  const bool usePhysicalAngularPositionObjective =
                      hasAvbdJointObjective(
                          jnt.objectiveProgram,
                          AvbdJointObjectiveKind::
                              AngularAxisPositionDrive);
                  const bool useCoupledAngularPositionObjective =
                      hasAvbdJointObjective(
                          jnt.objectiveProgram,
                          AvbdJointObjectiveKind::
                              CoupledAngularPositionDrive);
                  if (usePhysicalAngularAxisVelocityObjective ||
                      usePhysicalAngularPositionObjective ||
                      useCoupledAngularPositionObjective) {
                    // The physical force-mode objective is solved directly
                    // in the primal path; carrying an AL multiplier would
                    // add non-physical torque on top of the authored limit.
                    (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] = 0.0f;
                    continue;
                  }
                  const physx::PxReal damping =
                      (&jnt.angularDamping.x)[axes[a].dampIdx];
                  const physx::PxReal stiffness =
                      (&jnt.angularStiffness.x)[axes[a].dampIdx];
                  const bool isAccelerationDrive =
                      jnt.isAngularAccelerationDrive(axes[a].dampIdx);
                  const physx::PxReal effectiveRate =
                      isAccelerationDrive ? dt * stiffness + damping : damping;
                  if (effectiveRate <= 0.0f)
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

                  physx::PxReal rhoDualDrive =
                      physx::PxMin(damping / dt2, rhoDual);
                  if (isAccelerationDrive) {
                    const physx::PxReal driveScale =
                        computeAngularDriveRecipResponse(bodyARef, bodyBRef,
                                                         wAxis);
                    const physx::PxReal implicitScale =
                        1.0f / (1.0f + dt * effectiveRate);
                    rhoDualDrive = physx::PxMin(
                        driveScale * implicitScale * effectiveRate, rhoDual);
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
          ++consecutiveConvergedIterations;
          if (consecutiveConvergedIterations >= 2)
            break;
        } else {
          consecutiveConvergedIterations = 0;
        }
      }
    } // end iteration loop
  }

  // =========================================================================
  // Shared post-AL stages (same contract as contact path): depen, Decision A
  // velocity friction, pose-split finalize, mesh-relative e=0.  The strict
  // frictional linear-position island already owns both contact tangents in
  // its position PCG, so it skips the legacy body-static velocity-friction
  // stage.  Strict native motor velocity ownership runs below, after velocity
  // reconstruction.
  // =========================================================================
  physx::PxArray<bool> positionOwnedAngularBodies(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    positionOwnedAngularBodies[i] = false;
  for (physx::PxU32 softBodyIndex = 0;
       softBodyIndex < numSoftBodies; ++softBodyIndex) {
    const AvbdSoftBodyRuntimeState &runtime =
        softBodies[softBodyIndex].runtime;
    for (physx::PxU32 objectiveIndex = 0;
         objectiveIndex < runtime.compiledObjectives.size();
         ++objectiveIndex) {
      const AvbdCompiledSoftObjective &objective =
          runtime.compiledObjectives[objectiveIndex];
      if (objective.owner ==
              AvbdSoftObjectiveOwner::
                  eRIGID_ATTACHMENT_POSITION_AL &&
          objective.rigidBodyIdx < numBodies)
        positionOwnedAngularBodies[objective.rigidBodyIdx] = true;
      else if (objective.owner ==
                   AvbdSoftObjectiveOwner::
                       eARTICULATION_ATTACHMENT_POSITION_AL &&
               objective.rigidBodyIdx < numBodies &&
               articulationForBody) {
        FeatherstoneArticulation *const articulation =
            articulationForBody[objective.rigidBodyIdx];
        if (!articulation)
          continue;
        for (physx::PxU32 linkBodyIndex = 0;
             linkBodyIndex < numBodies; ++linkBodyIndex) {
          if (articulationForBody[linkBodyIndex] == articulation)
            positionOwnedAngularBodies[linkBodyIndex] = true;
        }
      }
    }
  }
  postAlStages(
      dt, invDt, bodies, numBodies, contacts, numContacts, contactMap, gravity,
      hasBodyStaticContact, deformableFastImpactIsland, touchingBodyStatic,
      numContacts > 0 ? &linearVelAtSolveStart : nullptr,
      numContacts > 0 ? &angularVelAtSolveStart : nullptr,
      /*allowRigidDeepPoseRecoverySplit=*/false,
      /*allowRigidFiniteMaterialPoseSplit=*/false, softParticles,
      numSoftParticles, softBodies, numSoftBodies, softContacts,
      numSoftContacts, touchesKinematicShell,
      hasKinematicShellContacts ? &shellLinearVelAtSolveStart : nullptr,
      &positionOwnedAngularBodies,
      d6Joints, numD6, /*hasJointConstraints=*/true,
      contactCoupledNativeRevoluteMotorVelocityProjectionIsland ||
          coupledLinearPositionDriveFrictionPositionOwnerIsland,
      /*applyVelocityDamping=*/true, softParticles, numSoftParticles, stats,
      /*postAlContactWork=*/nullptr,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr);
  // The DCD admission above may have clipped an incoming mixed endpoint at
  // the physical boundary.  Convert only that withheld normal motion into a
  // shared e=0 pair impulse after both soft and rigid velocities have been
  // reconstructed.  This keeps the regular Position-AL rows responsible for
  // local deformation, while preventing the next frame from treating the
  // withheld approach as a spurious elastic launch.
  if (mixedOgcPairContext.isComplete(numSoftContacts)) {
    PX_PROFILE_ZONE("AVBD.admittedMixedOgcPairInelasticVel", 0);
    clampAdmittedMixedOgcPairNormalVelocities(
        bodies, numBodies, softParticles, numSoftParticles, softContacts,
        numSoftContacts, mixedOgcPairStates, numMixedOgcPairStates,
        useProvidedSoftExecutionPlan
            ? softExecutionPlan->ogcPairIndices : nullptr,
        useProvidedSoftExecutionPlan
            ? softExecutionPlan->numOgcPairIndices : 0u,
        useProvidedSoftExecutionPlan &&
                softExecutionPlan->hasMixedOgcPairContactPlan(
                    numSoftContacts)
            ? softExecutionPlan->ogcPairContactStarts : nullptr,
        useProvidedSoftExecutionPlan &&
                softExecutionPlan->hasMixedOgcPairContactPlan(
                    numSoftContacts)
            ? softExecutionPlan->numOgcPairContactStarts : 0u,
        useProvidedSoftExecutionPlan &&
                softExecutionPlan->hasMixedOgcPairContactPlan(
                    numSoftContacts)
            ? softExecutionPlan->ogcPairContactRefs : nullptr,
        useProvidedSoftExecutionPlan &&
                softExecutionPlan->hasMixedOgcPairContactPlan(
                    numSoftContacts)
            ? softExecutionPlan->numOgcPairContactRefs : 0u,
        &stats);
  }
  if (endpointRecoveredWorldStaticBodies.size() == numSoftBodies &&
      endpointWorldStaticRecoveryNormals.size() == numSoftBodies) {
		if (endpointRecoveredDynamicSoftRigidContacts.size() ==
			numSoftContacts) {
			PX_PROFILE_ZONE("AVBD.endpointDynamicSoftRigidInelasticVel", 0);
			clampDynamicSoftRigidBodyEndpointVelocities(
				bodies, numBodies, softParticles, numSoftParticles, softBodies,
				numSoftBodies, softContacts, numSoftContacts,
				&endpointRecoveredDynamicSoftRigidContacts, &stats);
		}
    // The coherent endpoint shift deliberately carries initialPosition with
    // it.  Remove the pre-impact normal velocity only after postAlStages has
    // completed the ordinary position-to-velocity reconstruction.
    clampWorldStaticSoftBodyEndpointVelocities(
        softParticles, numSoftParticles, softBodies, numSoftBodies,
        &endpointRecoveredWorldStaticBodies,
        &endpointWorldStaticRecoveryNormals, &stats);
  }
  avbdApplyBendingDamping(
      softParticles, softBodies, numSoftBodies, dt);

  if (nativeRevoluteMotorVelocityProjectionIsland) {
    PX_PROFILE_ZONE("AVBD.projectNativeRevoluteMotorVelocity", 0);
    projectSingleNativeRevoluteMotorVelocity(
        bodies, numBodies, d6Joints[0], dt,
        conserveNativeRevoluteMotorAngularMomentum,
        nativeRevoluteMotorExpectedAngularMomentum,
        conserveNativeRevoluteMotorAngularMomentumVector,
        nativeRevoluteMotorExpectedAngularMomentumVector,
        conserveNativeRevoluteMotorLinearMomentum,
        nativeRevoluteMotorExpectedLinearMomentum,
        conserveNativeRevoluteMotorSpatialMomentum,
        nativeRevoluteMotorExpectedSpatialAngularMomentum,
        useNativeRevoluteMotorSolveStartRelativeVelocity,
        nativeRevoluteMotorSolveStartRelativeVelocity);
  }

  if (contactCoupledNativeRevoluteMotorVelocityProjectionIsland) {
    PX_PROFILE_ZONE(
        "AVBD.projectContactCoupledNativeRevoluteMotorVelocity", 0);
    projectContactCoupledNativeRevoluteMotorVelocity(
        bodies, numBodies, contacts, numContacts, d6Joints[0], gravity,
        dt);
  }

  if (nativeRevoluteMotorGearVelocityProjectionIsland &&
      nativeRevoluteMotorGearJointIndex < numD6) {
    PX_PROFILE_ZONE("AVBD.projectNativeRevoluteMotorGearVelocity", 0);
    projectNativeRevoluteMotorGearVelocity(
        bodies, numBodies, d6Joints[nativeRevoluteMotorGearJointIndex],
        gearJoints[0], dt);
  }

  if (coupledFixedD6Island) {
    PX_PROFILE_ZONE("AVBD.projectCoupledFixedD6Velocity", 0);
    projectCoupledFixedD6Velocity(
        bodies, numBodies, d6Joints[0]);
  }

  if (passiveCenteredGearVelocityProjectionIsland) {
    PX_PROFILE_ZONE("AVBD.projectPassiveCenteredGearVelocity", 0);
    projectPassiveCenteredGearVelocity(
        bodies, numBodies, gearJoints[0], angularVelAtSolveStart);
  }
  if (passiveGenericHard1DVelocityProjectionIsland &&
      passiveGenericHard1DIndex < numD6) {
    PX_PROFILE_ZONE("AVBD.projectPassiveGenericHard1DVelocity", 0);
    projectSinglePassiveGenericHard1DVelocity(
        bodies, numBodies, d6Joints[passiveGenericHard1DIndex],
        genericLinearVelAtSolveStart, genericAngularVelAtSolveStart);
  }
  if ((coupledLinearDriveIsland ||
       coupledLinearPositionDriveIsland) &&
      !coupledLinearPositionDriveFrictionPositionOwnerIsland) {
    physx::PxReal totalMass = 0.0f;
    physx::PxVec3 finalMomentum(0.0f);
    bool velocityLimitActive = false;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal mass = 1.0f / bodies[i].invMass;
      totalMass += mass;
      finalMomentum += bodies[i].linearVelocity * mass;
      const physx::PxReal maxSpeedSquared = bodies[i].maxLinearVelocitySq;
      velocityLimitActive |=
          maxSpeedSquared > 0.0f &&
          bodies[i].linearVelocity.magnitudeSquared() >=
              maxSpeedSquared * (1.0f - 1e-5f);
    }
    if (!velocityLimitActive) {
      // Close only the translation-invariant equation after the shared
      // velocity finalize. Both the velocity-drive and position-drive
      // objectives are internal, so neither may create a shared tangent
      // momentum mode. The strict frictional support owner is excluded:
      // ground friction is an external impulse, and restoring the island's
      // solve-start tangent momentum would erase its physical effect.
      // The target includes gravity and global damping; per-body damping is
      // excluded by the ownership predicate, and an active authored velocity
      // cap suppresses this numerical projection.
      physx::PxVec3 momentumError =
          finalMomentum - coupledExpectedMomentum;
      if (numContacts > 0) {
        physx::PxVec3 supportNormal = contacts[0].contactNormal;
        supportNormal.normalize();
        momentumError -= supportNormal * momentumError.dot(supportNormal);
      }
      const physx::PxVec3 commonVelocity = momentumError / totalMass;
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        bodies[i].linearVelocity -= commonVelocity;
        bodies[i].position -= commonVelocity * dt;
      }
    }
  }

  if (coupledLinearPositionDriveIsland &&
      conserveCoupledLinearPositionSupportAxisMomentum) {
    const physx::PxU32 bodyA = d6Joints[0].header.bodyIndexA;
    const physx::PxU32 bodyB = d6Joints[0].header.bodyIndexB;
    // This is a shared rigid null-mode closure, not a second drive solve:
    // it leaves every rotating-frame D6 coordinate derivative and every
    // support-normal point velocity unchanged.
    restoreTwoBodySupportAxisAngularMomentum(
        bodies[bodyA], bodies[bodyB],
        coupledLinearPositionSupportAxis,
        coupledExpectedLinearPositionSupportAxisAngularMomentum);
  }

  if (coupledAngularPositionDriveIsland ||
      coupledSphericalConeIsland) {
    physx::PxMat33 totalInertia(physx::PxZero);
    physx::PxVec3 finalAngularMomentum(0.0f);
    bool velocityLimitActive = false;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxMat33 inertia =
          bodies[i].invInertiaWorld.getInverse();
      totalInertia += inertia;
      finalAngularMomentum += inertia * bodies[i].angularVelocity;
      const physx::PxReal maxSpeedSquared =
          bodies[i].maxAngularVelocitySq;
      velocityLimitActive |=
          maxSpeedSquared > 0.0f &&
          bodies[i].angularVelocity.magnitudeSquared() >=
              maxSpeedSquared * (1.0f - 1e-5f);
    }
    if (!velocityLimitActive) {
      // An internal centered angular row cannot change island angular
      // momentum.  Remove only the shared world-angular mode after velocity
      // finalize; applying the same correction to both endpoints preserves
      // their relative angular velocity and either the D6 target or the
      // spherical cone state.
      const physx::PxVec3 angularMomentumError =
          finalAngularMomentum - coupledExpectedAngularMomentum;
      const physx::PxVec3 commonAngularVelocity =
          totalInertia.getInverse() * angularMomentumError;
      const physx::PxReal correctionSpeed =
          commonAngularVelocity.magnitude();
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        bodies[i].angularVelocity -= commonAngularVelocity;
      if (correctionSpeed > 1e-10f && PxIsFinite(correctionSpeed)) {
        const physx::PxQuat correction(
            -correctionSpeed * dt,
            commonAngularVelocity * (1.0f / correctionSpeed));
        for (physx::PxU32 i = 0; i < numBodies; ++i)
          bodies[i].rotation =
              (correction * bodies[i].rotation).getNormalized();
      }
    }
  }

  // Mimic velocity iterations are hard even when the position equation is
  // compliant. Run them after all drive conservation projections so later
  // post-processing cannot undo the velocity derivative. Position ownership
  // remains in the generic-hard AL path for hard mimic rows and in the
  // compliant position path for compliant mimic rows.
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    if (hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::ArticulationHardMimic)) {
      PX_PROFILE_ZONE("AVBD.projectArticulationMimicVelocity1D", 0);
      projectArticulationMimicVelocity1D(
          bodies, numBodies, d6Joints[i]);
    } else if (hasAvbdJointObjective(
                   d6Joints[i].objectiveProgram,
                   AvbdJointObjectiveKind::
                       ArticulationCompliantMimic)) {
      PX_PROFILE_ZONE("AVBD.projectArticulationMimicVelocity1D", 0);
      projectArticulationMimicVelocity1D(
          bodies, numBodies, d6Joints[i]);
    }
  }
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f) {
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
      bodies[i].projectLockedVelocities();
    }
  }

  // Preserve an auditable partition of every published objective-program
  // entry after runtime fallback decisions. An empty valid program contributes
  // one Legacy sentinel for its source D6 record until prep enumerates that
  // record's ordinary physical objectives.
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    const AvbdCompiledJointObjectiveProgram &program =
        d6Joints[i].objectiveProgram;
    (void)0;
    if (!isValidAvbdJointObjectiveProgram(program)) {
      (void)0;
      continue;
    }
    if (program.entryCount == 0 ||
        program.legacySourceRowMask != 0) {
      (void)0;
    }
    for (physx::PxU32 entryIndex = 0;
         entryIndex < program.entryCount; ++entryIndex) {
      const AvbdCompiledJointObjective &objective =
          program.entries[entryIndex];
      switch (objective.owner) {
      case AvbdVelocityObjectiveOwner::PositionAL:
        (void)0;
        break;
      case AvbdVelocityObjectiveOwner::JointFinalize:
        (void)0;
        break;
      case AvbdVelocityObjectiveOwner::Unsupported:
        (void)0;
        break;
      case AvbdVelocityObjectiveOwner::PointFinalize:
      case AvbdVelocityObjectiveOwner::ManifoldFinalize:
      case AvbdVelocityObjectiveOwner::ComponentFinalize:
        (void)0;
        break;
      }
    }
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
    const AvbdSoftBody &sb,
    AvbdSoftContact *softContacts, PxU32 numSoftContacts,
    const AvbdSoftContactParticleRef *softContactRefs,
    PxU32 softContactRefBegin, PxU32 softContactRefEnd,
    PxReal dt, PxReal invDt2,
    AvbdCpuIsaCorotationalTetPacket8Fn corotationalTetPacketKernel,
    const AvbdOgcPairTrustRegionContext *ogcPairContext)
{
  PX_UNUSED(numSoftParticles);
  PX_UNUSED(dt);

  AvbdSoftParticle &sp = softParticles[spi];
  if (sp.invMass <= 0.0f) return;

  PxReal mOverDt2 = sp.mass * invDt2;

  // Inertial force and Hessian
  PxVec3 f3 = (sp.predictedPosition - sp.position) * mOverDt2;
  PxMat33 H3 = PxMat33::createDiagonal(PxVec3(mOverDt2));

  // The Scene-compiled execution plan gives every active particle one body
  // owner.  That removes the former O(particles * softBodies) membership
  // scan while preserving the material/objective accumulation order inside
  // the owning body.
  if (spi < sb.compiled.particleStart ||
      spi - sb.compiled.particleStart >= sb.compiled.particleCount)
    return;
  const PxU32 localIdx = spi - sb.compiled.particleStart;
  const AvbdParticleElementAdjacency &elementAdjacency =
      sb.compiled.elementAdjacency[localIdx];
  const AvbdParticleObjectiveAdjacency &objectiveAdjacency =
      sb.runtime.objectiveAdjacency[localIdx];

    // StVK triangle contributions
    for (PxU32 ri = 0; ri < elementAdjacency.triRefs.size(); ++ri)
    {
      const AvbdParticleElementRef &ref = elementAdjacency.triRefs[ri];
      PxVec3 ft; PxMat33 Ht;
      avbdEvaluateStVKForceHessian(sb.compiled.triElements[ref.index], ref.vOrder,
                                    sb.material.mu, sb.material.lambda, softParticles, ft, Ht);
      f3 += ft; H3 += Ht;
    }

    // Tetrahedral material-model contributions.  A complete packet program
    // only changes the local material evaluation; it never changes particle
    // ownership, attachment ownership, or the rigid/soft phase barriers.
    // Ordered scenes intentionally retain the scalar evaluation above the
    // call site, while the relaxed fast path uses the same packet backend as
    // the component solver and falls back lane-by-lane when necessary.
    const bool useCorotationalTetPackets =
        corotationalTetPacketKernel &&
        sb.material.coRotationalVolumeModel &&
        sb.compiled.tetIncidencePacketProgramValid &&
        localIdx < sb.compiled.tetIncidencePacketRanges.size() &&
        elementAdjacency.tetRefs.size() >=
            eAVBD_TET_INCIDENCE_PACKET_WIDTH;
    if (useCorotationalTetPackets) {
      avbdAccumulateCorotationalTetPacketContributions(
          sb, localIdx, softParticles, corotationalTetPacketKernel,
          false, NULL, f3, H3);
    } else {
      for (PxU32 ri = 0; ri < elementAdjacency.tetRefs.size(); ++ri)
      {
        const AvbdParticleElementRef &ref = elementAdjacency.tetRefs[ri];
        PxVec3 ft; PxMat33 Ht;
        if (sb.material.coRotationalVolumeModel)
          avbdEvaluateCorotationalForceHessianPrepared(
              sb.compiled.tetElements[ref.index], ref.vOrder,
              sb.material.mu, sb.material.lambda,
              softParticles, ft, Ht);
        else
          avbdEvaluateNeoHookeanForceHessianPrepared(
              sb.compiled.tetElements[ref.index], ref.vOrder,
              sb.material.mu, sb.material.lambda, sb.material.neoHookeanAlpha,
              softParticles, ft, Ht);
        f3 += ft; H3 += Ht;
      }
    }

    // Bending contributions
    for (PxU32 ri = 0; ri < elementAdjacency.bendRefs.size(); ++ri)
    {
      const AvbdParticleElementRef &ref = elementAdjacency.bendRefs[ri];
      PxVec3 fb; PxMat33 Hb;
      avbdEvaluateBendingForceHessian(sb.compiled.bendElements[ref.index], ref.vOrder,
                                       sb.material.bendingStiffness, softParticles, fb, Hb);
      f3 += fb; H3 += Hb;
    }

    // Prep assigns each physical soft equality objective one owner.
    for (PxU32 oi = 0;
         oi < objectiveAdjacency.objectiveIndices.size(); ++oi)
    {
      const PxU32 objectiveIndex =
          objectiveAdjacency.objectiveIndices[oi];
      const AvbdCompiledSoftObjective &objective =
          sb.runtime.compiledObjectives[objectiveIndex];
      PxVec3 objectiveForce;
      PxMat33 objectiveHessian;
      switch (objective.owner)
      {
      case AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eARTICULATION_ATTACHMENT_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eSOFT_PAIR_ATTACHMENT_POSITION_AL:
        // The coupled 9-DOF position block is the sole primal owner.
        continue;
      case AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eDEFORMABLE_KINEMATIC_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eKINEMATIC_ATTACHMENT_POSITION_AL:
        avbdEvaluatePinForceHessian(
            objective.point,
            sb.runtime.pins[objective.runtimeStateIndex],
            softParticles, spi, objectiveForce, objectiveHessian);
        break;
      default:
        PX_ASSERT(false);
        continue;
      }
      f3 += objectiveForce;
      H3 += objectiveHessian;
    }

  // Soft contacts (ground / rigid, AVBD penalty). Contact prep groups only
  // the rows incident to this particle while preserving source order.
  for (PxU32 contactRefIndex = softContactRefBegin;
       contactRefIndex < softContactRefEnd; ++contactRefIndex)
  {
    const AvbdSoftContactParticleRef &contactRef =
        softContactRefs[contactRefIndex];
    if (contactRef.contactIndex >= numSoftContacts)
      continue;
    PxVec3 fc; PxMat33 Hc;
    const AvbdSoftContact& contact =
        softContacts[contactRef.contactIndex];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    if (geometry.hasRigidBodyTarget()) {
      if (geometry.targetIndex >= numRigidBodies)
        continue;
      avbdEvaluateContactParticleBlockAtSurfacePoint(
          geometry, contact.state, softParticles,
          avbdGetRigidContactSurfacePoint(
              geometry, rigidBodies[geometry.targetIndex]),
          contactRef.jacobianScale, fc, Hc);
    } else {
      avbdEvaluateContactParticleBlock(
          geometry, contact.state, softParticles,
          contactRef.jacobianScale, fc, Hc);
    }
    f3 += fc; H3 += Hc;

    // Endpoint admission clips a mixed pair to its current DCD boundary so
    // the rigid cannot enter the soft mesh.  The clipped normal work must
    // still reach the material solve; otherwise the rigid simply stops at
    // the boundary and the soft volume never develops the compressive
    // deformation requested by the motion.  Apply one shared pair load,
    // distributed over the prepared manifold, to the soft side only.  The
    // rigid side remains owned by the coupled body solve and its trust-region
    // limiter, preventing the old high-speed rebound.
    if (ogcPairContext &&
        ogcPairContext->isComplete(numSoftContacts) &&
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numRigidBodies) {
      const PxU32 pairIndex =
          ogcPairContext->contactPairIndices[contactRef.contactIndex];
      if (pairIndex < ogcPairContext->numPairStates) {
        const AvbdOgcPairState &pair =
            ogcPairContext->pairStates[pairIndex];
        const PxReal load = avbdGetOgcPairNormalLoadPerContact(pair);
        if (load > 0.0f) {
          const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
          if (PxIsFinite(load) && load > 0.0f &&
              PxIsFinite(normalLengthSq) && normalLengthSq > 1.0e-12f) {
            const PxVec3 normal = geometry.normal *
                PxRecipSqrt(normalLengthSq);
            const PxReal weight = PxAbs(contactRef.jacobianScale);
            if (normal.isFinite() && PxIsFinite(weight))
              f3 += normal * (load * weight);
          }
        }
      }
    }
  }

  // Solve 3x3: displacement = inv(H) * f
  PxVec3 displacement = avbdSolveSymmetric33(H3, f3);
  PxReal dispMag = displacement.magnitude();
  if (!PxIsFinite(dispMag))
    displacement = PxVec3(0.0f);

  const PxReal ogcAlpha = limitSoftParticleOgcCandidate(
      ogcPairContext, softContacts, numSoftContacts, softContactRefs,
      softContactRefBegin, softContactRefEnd, spi, softParticles,
      numSoftParticles, rigidBodies,
      numRigidBodies, displacement);
  displacement *= ogcAlpha;

  // The mixed OGC limiter constrains motion against the rigid trust region,
  // but it does not protect the incident tetrahedra from a local material
  // solve collapsing their positive Jacobian.  This is especially important
  // for a dynamic box pressing two soft volumes: if a particle crosses the
  // J floor first, the later OGC response has no valid soft deformation left
  // to absorb the load and the rigid body receives an artificial rebound.
  // Apply the same analytic incident-tet limiter used by the component path
  // to the final candidate (after OGC scaling, before committing position).
  if(displacement.magnitudeSquared() > 0.0f)
  {
    const AvbdSoftTetDisplacementLimitResult positiveJLimit =
        avbdLimitTetDisplacementObserved(
            sb, spi, softParticles, displacement);
    displacement = positiveJLimit.appliedDisplacement;
  }

  sp.position += displacement;
}

void AvbdSolver::solveSoftRigidAttachmentsCoupled(
    AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
    AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
    AvbdSoftBody *softBodies, PxU32 numSoftBodies,
    PxReal dt,
    FeatherstoneArticulation *const *articulationForBody,
    const PxU32 *linkIndexForBody)
{
  if (!softParticles || !rigidBodies || !softBodies ||
      numSoftParticles == 0 || numRigidBodies == 0 ||
      numSoftBodies == 0 || dt <= 0.0f)
    return;

  for (PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
  {
    AvbdSoftBodyRuntimeState &runtime =
        softBodies[bodyIndex].runtime;
    for (PxU32 objectiveIndex = 0;
         objectiveIndex < runtime.compiledObjectives.size();
         ++objectiveIndex)
    {
      const AvbdCompiledSoftObjective &objective =
          runtime.compiledObjectives[objectiveIndex];
      const bool dynamicRigidOwner =
          objective.owner ==
          AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL;
      const bool articulationOwner =
          objective.owner ==
          AvbdSoftObjectiveOwner::
              eARTICULATION_ATTACHMENT_POSITION_AL;
      if (!dynamicRigidOwner && !articulationOwner)
        continue;
      if (objective.runtimeStateIndex >=
          runtime.attachments.size())
        continue;

      AvbdSoftAttachment &attachment =
          runtime.attachments[objective.runtimeStateIndex];
      if (attachment.rigidBodyIdx >= numRigidBodies ||
          attachment.k <= 0.0f)
        continue;
      const PxReal softPointInverseMass =
          avbdGetSoftPointInverseMass(
              objective.point, softParticles, numSoftParticles);
      if (softPointInverseMass <= 0.0f)
        continue;
      AvbdSolverBody &rigidBody =
          rigidBodies[attachment.rigidBodyIdx];
      if (articulationOwner)
      {
        if (!articulationForBody || !linkIndexForBody)
          continue;
        FeatherstoneArticulation *const articulation =
            articulationForBody[attachment.rigidBodyIdx];
        const PxU32 targetLinkIndex =
            linkIndexForBody[attachment.rigidBodyIdx];
        if (!articulation ||
            targetLinkIndex >=
                articulation->getArticulationData().getLinkCount())
          continue;

        const PxVec3 worldOffset =
            rigidBody.rotation.rotate(attachment.localOffset);
        const PxVec3 worldAnchor =
            rigidBody.position + worldOffset;
        const PxVec3 constraint =
            avbdGetSoftPointPosition(
                objective.point, softParticles) - worldAnchor;
        const PxVec3 basis[3] = {
            PxVec3(1.0f, 0.0f, 0.0f),
            PxVec3(0.0f, 1.0f, 0.0f),
            PxVec3(0.0f, 0.0f, 1.0f)};
        PxVec3 pointResponseColumns[3];
        bool responseIsFinite = true;
        for (PxU32 axis = 0; axis < 3; ++axis)
        {
          const Cm::SpatialVector pointImpulse(
              basis[axis], worldOffset.cross(basis[axis]));
          Cm::SpatialVector targetResponse =
              Cm::SpatialVector::zero();
          articulation->getImpulseResponse(
              targetLinkIndex, pointImpulse, targetResponse);
          pointResponseColumns[axis] =
              targetResponse.linear +
              targetResponse.angular.cross(worldOffset);
          responseIsFinite =
              responseIsFinite &&
              pointResponseColumns[axis].isFinite();
        }
        if (!responseIsFinite)
          continue;

        PxMat33 articulationPointInverseMass(
            pointResponseColumns[0], pointResponseColumns[1],
            pointResponseColumns[2]);
        articulationPointInverseMass =
            (articulationPointInverseMass +
             articulationPointInverseMass.getTranspose()) *
            0.5f;
        const PxReal dt2 = dt * dt;
        const PxMat33 unit =
            PxMat33::createDiagonal(PxVec3(1.0f));
        const PxReal compliance =
            1.0f / PxMax(attachment.k, 1.0e-6f);
        const PxMat33 effectiveMass =
            (unit * softPointInverseMass +
             articulationPointInverseMass) *
                dt2 +
            unit * compliance;
        const PxVec3 multiplier = avbdSolveSymmetric33(
            effectiveMass,
            -(constraint + attachment.alLambda * compliance));
        if (!multiplier.isFinite())
          continue;

        PxVec3 particleCorrections[4] = {
            PxVec3(0.0f), PxVec3(0.0f),
            PxVec3(0.0f), PxVec3(0.0f)};
        bool particleCorrectionsAreFinite = true;
        for (PxU32 endpoint = 0;
             endpoint < objective.point.particleCount; ++endpoint)
        {
          const PxU32 particleIndex =
              objective.point.particleIndices[endpoint];
          particleCorrections[endpoint] =
              multiplier *
              (dt2 * objective.point.weights[endpoint] *
               softParticles[particleIndex].invMass);
          particleCorrectionsAreFinite =
              particleCorrectionsAreFinite &&
              particleCorrections[endpoint].isFinite();
        }
        const PxVec3 articulationPointLoad =
            multiplier * (-dt2);
        const Cm::SpatialVector articulationLoad(
            articulationPointLoad,
            worldOffset.cross(articulationPointLoad));
        Cm::SpatialVector targetPoseResponse =
            Cm::SpatialVector::zero();
        articulation->getImpulseResponse(
            targetLinkIndex, articulationLoad,
            targetPoseResponse);
        if (!particleCorrectionsAreFinite ||
            !targetPoseResponse.isFinite())
          continue;

        for (PxU32 endpoint = 0;
             endpoint < objective.point.particleCount; ++endpoint)
        {
          softParticles[
              objective.point.particleIndices[endpoint]].position +=
              particleCorrections[endpoint];
        }
        for (PxU32 linkBodyIndex = 0;
             linkBodyIndex < numRigidBodies; ++linkBodyIndex)
        {
          if (articulationForBody[linkBodyIndex] != articulation)
            continue;
          const PxU32 responseLinkIndex =
              linkIndexForBody[linkBodyIndex];
          if (responseLinkIndex >=
              articulation->getArticulationData().getLinkCount())
            continue;

          Cm::SpatialVector linkPoseResponse =
              Cm::SpatialVector::zero();
          if (responseLinkIndex == targetLinkIndex)
          {
            linkPoseResponse = targetPoseResponse;
          }
          else
          {
            Cm::SpatialVector ignoredTargetResponse =
                Cm::SpatialVector::zero();
            articulation->getImpulseSelfResponse(
                targetLinkIndex, responseLinkIndex,
                articulationLoad, Cm::SpatialVector::zero(),
                ignoredTargetResponse, linkPoseResponse);
          }
          if (!linkPoseResponse.isFinite())
            continue;

          AvbdSolverBody &linkBody = rigidBodies[linkBodyIndex];
          const PxVec3 linkPositionBefore = linkBody.position;
          const PxQuat linkRotationBefore = linkBody.rotation;
          linkBody.position += linkPoseResponse.linear;
          if (linkPoseResponse.angular.magnitudeSquared() >
              1.0e-16f)
          {
            const PxQuat angularStep(
                linkPoseResponse.angular.x,
                linkPoseResponse.angular.y,
                linkPoseResponse.angular.z, 0.0f);
            linkBody.rotation =
                (linkBody.rotation +
                 angularStep * linkBody.rotation * 0.5f).
                    getNormalized();
          }
          linkBody.projectLockedPose(
              linkPositionBefore, linkRotationBefore);
        }
        continue;
      }

      AvbdSoftRigidAttachmentCoupledStep step;
      if (!avbdEvaluateSoftRigidAttachmentCoupledStep(
              attachment, objective.point, softParticles,
              numSoftParticles, rigidBody, dt, step))
        continue;

      const PxVec3 rigidPositionBefore = rigidBody.position;
      const PxQuat rigidRotationBefore = rigidBody.rotation;
      for (PxU32 endpoint = 0;
           endpoint < objective.point.particleCount; ++endpoint)
      {
        softParticles[
            objective.point.particleIndices[endpoint]].position +=
            step.particleCorrections[endpoint];
      }
      rigidBody.position += step.rigidLinearCorrection;
      if (step.rigidAngularCorrection.magnitudeSquared() > 1.0e-16f)
      {
        const PxQuat angularStep(
            step.rigidAngularCorrection.x,
            step.rigidAngularCorrection.y,
            step.rigidAngularCorrection.z, 0.0f);
        rigidBody.rotation =
            (rigidBody.rotation +
             angularStep * rigidBody.rotation * 0.5f).
                getNormalized();
      }
      rigidBody.projectLockedPose(
          rigidPositionBefore, rigidRotationBefore);
    }
  }
}

void AvbdSolver::solveSoftPairAttachmentsCoupled(
    AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, PxU32 numSoftBodies,
    PxReal dt)
{
  if (!softParticles || !softBodies || numSoftParticles == 0 ||
      numSoftBodies == 0 || dt <= 0.0f)
    return;

  const PxReal dt2 = dt * dt;
  for (PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
  {
    AvbdSoftBodyRuntimeState &runtime =
        softBodies[bodyIndex].runtime;
    for (PxU32 objectiveIndex = 0;
         objectiveIndex < runtime.compiledObjectives.size();
         ++objectiveIndex)
    {
      const AvbdCompiledSoftObjective &objective =
          runtime.compiledObjectives[objectiveIndex];
      if (objective.owner !=
          AvbdSoftObjectiveOwner::
              eSOFT_PAIR_ATTACHMENT_POSITION_AL)
        continue;
      if (objective.runtimeStateIndex >=
          runtime.attachments.size())
        continue;

      AvbdSoftAttachment &attachment =
          runtime.attachments[objective.runtimeStateIndex];
      if (attachment.k <= 0.0f)
        continue;
      const PxReal sourceInverseMass =
          avbdGetSoftPointInverseMass(
              objective.point, softParticles, numSoftParticles);
      const PxReal targetInverseMass =
          avbdGetSoftPointInverseMass(
              objective.targetPoint,
              softParticles, numSoftParticles);
      const PxReal combinedInverseMass =
          sourceInverseMass + targetInverseMass;
      if (combinedInverseMass <= 0.0f)
        continue;

      const PxVec3 constraint =
          avbdGetSoftPointPosition(
              objective.point, softParticles) -
          avbdGetSoftPointPosition(
              objective.targetPoint, softParticles);
      const PxReal compliance =
          1.0f / PxMax(attachment.k, 1.0e-6f);
      const PxReal effectiveMass =
          combinedInverseMass * dt2 + compliance;
      if (effectiveMass <= 1.0e-12f ||
          !PxIsFinite(effectiveMass))
        continue;
      const PxVec3 multiplier =
          -(constraint + attachment.alLambda * compliance) /
          effectiveMass;
      if (!multiplier.isFinite())
        continue;

      for (PxU32 endpoint = 0;
           endpoint < objective.point.particleCount; ++endpoint)
      {
        const PxU32 particleIndex =
            objective.point.particleIndices[endpoint];
        softParticles[particleIndex].position +=
            multiplier *
            (dt2 * objective.point.weights[endpoint] *
             softParticles[particleIndex].invMass);
      }
      for (PxU32 endpoint = 0;
           endpoint < objective.targetPoint.particleCount; ++endpoint)
      {
        const PxU32 particleIndex =
            objective.targetPoint.particleIndices[endpoint];
        softParticles[particleIndex].position -=
            multiplier *
            (dt2 * objective.targetPoint.weights[endpoint] *
             softParticles[particleIndex].invMass);
      }
    }
  }
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
  for (PxU32 sbi = 0; sbi < numSoftBodies; ++sbi)
  {
    AvbdSoftBody &sb = softBodies[sbi];
    for (PxU32 oi = 0;
         oi < sb.runtime.compiledObjectives.size(); ++oi)
    {
      const AvbdCompiledSoftObjective &objective =
          sb.runtime.compiledObjectives[oi];
      switch (objective.owner)
      {
      case AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eARTICULATION_ATTACHMENT_POSITION_AL:
        avbdUpdateAttachmentDual(
            sb.runtime.attachments[objective.runtimeStateIndex],
            objective.point, softParticles, rigidBodies, beta);
        break;
      case AvbdSoftObjectiveOwner::
          eSOFT_PAIR_ATTACHMENT_POSITION_AL:
        avbdUpdateSoftPairAttachmentDual(
            sb.runtime.attachments[objective.runtimeStateIndex],
            objective.point, objective.targetPoint,
            softParticles, beta);
        break;
      case AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eDEFORMABLE_KINEMATIC_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eKINEMATIC_ATTACHMENT_POSITION_AL:
        avbdUpdatePinDual(
            sb.runtime.pins[objective.runtimeStateIndex],
            objective.point, softParticles, beta);
        break;
      default:
        PX_ASSERT(false);
        break;
      }
    }
  }
  for (PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    AvbdSoftContact &sc = softContacts[sci];
    const AvbdSoftContactGeometry& geometry = sc.geometry;
    AvbdSoftContactAugmentedState& state = sc.state;
    if (avbdIsSoftContactQueryFullyKinematic(
            geometry, softParticles, numSoftParticles) &&
        geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numRigidBodies) {
      avbdUpdateKinematicShellContactDual(
          geometry, state, rigidBodies[geometry.targetIndex],
          beta, mConfig.avbdPenaltyMax);
    } else if (geometry.hasRigidBodyTarget() &&
               geometry.targetIndex < numRigidBodies) {
      avbdUpdateSoftContactDualAtSurfacePoint(
          geometry, state, softParticles,
          avbdGetRigidContactSurfacePoint(
              geometry, rigidBodies[geometry.targetIndex]),
          beta);
    } else {
      avbdUpdateSoftContactDual(
          geometry, state,
          softParticles, beta);
    }
  }
}


} // namespace Dy
} // namespace physx
