// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcResponse.h"

#include "avbd/solver/rigid/DyAvbdSolverBody.h"
#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/soft/DyAvbdSoftBodyEpochSafety.h"
#include "avbd/solver/soft/DyAvbdSoftBodyMechanics.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"
#include "avbd/solver/soft/DyAvbdSoftContactGeometry.h"

namespace physx {
namespace Dy {

#include "avbd/solver/soft/DyAvbdSoftBodyMechanics.inl"

bool compileCurrentOgcNormalResponse(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSolverBody *dynamicTarget,
    physx::PxReal softResponseScale,
    AvbdOgcNormalResponse &response,
    const AvbdOgcRigidBoxGeometry *rigidBox) {
  response.current = AvbdOgcCurrentPairGeometry();
  response.sourceMobility =
      AvbdOgcNormalSourceMobility::eDYNAMIC_SOFT;
  response.normal = physx::PxVec3(0.0f);
  response.constraintValue = 0.0f;
  response.queryPoint = physx::PxVec3(0.0f);
  response.particleCount = 0u;
  response.softResponse = 0.0f;
  response.targetLinearJacobian = physx::PxVec3(0.0f);
  response.targetAngularJacobian = physx::PxVec3(0.0f);
  response.targetLinearDeltaPerLambda = physx::PxVec3(0.0f);
  response.targetAngularDeltaPerLambda = physx::PxVec3(0.0f);
  response.targetResponse = 0.0f;
  response.effectiveResponse = 0.0f;
  if (!softParticles || numSoftParticles == 0u ||
      !physx::PxIsFinite(softResponseScale) || softResponseScale <= 0.0f)
    return false;

  const bool dynamic = geometry.hasRigidBodyTarget();
  const bool worldStatic = geometry.hasWorldStaticTarget();
  const bool deformable = geometry.hasDeformableSurfaceTarget();
  const physx::PxU32 targetKindCount = physx::PxU32(dynamic) +
      physx::PxU32(worldStatic) + physx::PxU32(deformable);
  if (targetKindCount != 1u || dynamic != (dynamicTarget != nullptr))
    return false;

  response.queryPoint =
      avbdGetSoftContactQueryPoint(geometry, softParticles);
  if (!response.queryPoint.isFinite())
    return false;
  if (deformable) {
    const physx::PxVec3 targetPoint =
        avbdGetSoftContactSurfacePoint(geometry, softParticles);
    const physx::PxReal normalLengthSq =
        geometry.normal.magnitudeSquared();
    if (!targetPoint.isFinite() || !physx::PxIsFinite(normalLengthSq) ||
        normalLengthSq <= 1.0e-12f)
      return false;
    response.current.normal =
        geometry.normal * physx::PxRecipSqrt(normalLengthSq);
    response.current.targetOffset = physx::PxVec3(0.0f);
    response.current.signedGap =
        (response.queryPoint - targetPoint).dot(response.current.normal);
    if (!physx::PxIsFinite(response.current.signedGap))
      return false;
  } else if (!getCurrentOgcPairGeometry(
                 geometry, dynamicTarget, response.queryPoint,
                 response.current, rigidBox)) {
    return false;
  }

  response.particleCount = avbdCollectSoftContactParticleIndices(
      geometry, response.particleIndices);
  if (response.particleCount == 0u ||
      response.particleCount > AVBD_CONTACT_MAX_PARTICLES)
    return false;

  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    if (particleIndex >= numSoftParticles)
      return false;
    const AvbdSoftParticle &particle = softParticles[particleIndex];
    const physx::PxReal weight =
        avbdGetSoftContactParticleJacobianScale(geometry, particleIndex);
    if (!physx::PxIsFinite(weight) ||
        !physx::PxIsFinite(particle.invMass))
      return false;
    response.particleWeights[supportIndex] = weight;
    if (particle.invMass > 0.0f)
      response.softResponse +=
          softResponseScale * particle.invMass * weight * weight;
  }
  if (!physx::PxIsFinite(response.softResponse))
    return false;

  const bool dynamicSoftSource = response.softResponse > 1.0e-12f;
  const bool kinematicSoftSource = !dynamicSoftSource && dynamic &&
      avbdIsSoftContactQueryFullyKinematic(
          geometry, softParticles, numSoftParticles);
  if (!dynamicSoftSource && !kinematicSoftSource)
    return false;
  response.sourceMobility = dynamicSoftSource
      ? AvbdOgcNormalSourceMobility::eDYNAMIC_SOFT
      : AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT;
  if (kinematicSoftSource) {
    const physx::PxReal normalLengthSq =
        geometry.normal.magnitudeSquared();
    if (!physx::PxIsFinite(normalLengthSq) ||
        normalLengthSq <= 1.0e-12f)
      return false;
    response.normal =
        geometry.normal * physx::PxRecipSqrt(normalLengthSq);
    response.current.targetOffset =
        dynamicTarget->rotation.rotate(geometry.rigidLocalPoint);
    if (!response.current.targetOffset.isFinite())
      return false;
  } else {
    response.normal = response.current.normal;
  }

  if (dynamic) {
    if (!physx::PxIsFinite(dynamicTarget->invMass) ||
        !dynamicTarget->invInertiaWorld.column0.isFinite() ||
        !dynamicTarget->invInertiaWorld.column1.isFinite() ||
        !dynamicTarget->invInertiaWorld.column2.isFinite())
      return false;
    const physx::PxReal targetJacobianSign =
        kinematicSoftSource ? 1.0f : -1.0f;
    response.targetLinearJacobian =
        response.normal * targetJacobianSign;
    dynamicTarget->projectLockedLinearVector(
        response.targetLinearJacobian);
    response.targetLinearDeltaPerLambda =
        response.targetLinearJacobian * dynamicTarget->invMass;
    response.targetAngularJacobian =
        response.current.targetOffset.cross(response.normal) *
        targetJacobianSign;
    dynamicTarget->projectLockedAngularVector(
        response.targetAngularJacobian);
    response.targetAngularDeltaPerLambda =
        dynamicTarget->invInertiaWorld * response.targetAngularJacobian;
    dynamicTarget->projectLockedAngularVector(
        response.targetAngularDeltaPerLambda);
    response.targetResponse =
        response.targetLinearJacobian.dot(
            response.targetLinearDeltaPerLambda) +
        response.targetAngularJacobian.dot(
            response.targetAngularDeltaPerLambda);
    if (!response.targetLinearJacobian.isFinite() ||
        !response.targetAngularJacobian.isFinite() ||
        !response.targetLinearDeltaPerLambda.isFinite() ||
        !response.targetAngularDeltaPerLambda.isFinite() ||
        !physx::PxIsFinite(response.targetResponse) ||
        response.targetResponse < 0.0f)
      return false;
  }

  if (deformable) {
    response.constraintValue = response.current.signedGap;
  } else if (!evaluateCurrentOgcNormalConstraint(
                 geometry, response, dynamicTarget, response.queryPoint,
                 response.constraintValue)) {
    return false;
  }

  response.effectiveResponse =
      response.softResponse + response.targetResponse;
  return physx::PxIsFinite(response.effectiveResponse) &&
      response.effectiveResponse > 1.0e-12f;
}

bool getOgcNormalResponseQueryVelocity(
    const AvbdOgcNormalResponse &response,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    physx::PxVec3 &queryVelocity) {
  queryVelocity = physx::PxVec3(0.0f);
  if (!softParticles || response.particleCount == 0u ||
      response.particleCount > AVBD_CONTACT_MAX_PARTICLES)
    return false;
  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    const physx::PxReal weight = response.particleWeights[supportIndex];
    if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight) ||
        !softParticles[particleIndex].velocity.isFinite())
      return false;
    queryVelocity += softParticles[particleIndex].velocity * weight;
  }
  return queryVelocity.isFinite();
}

bool compileCurrentOgcTangentResponse(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSolverBody *dynamicTarget,
    AvbdOgcTangentResponse &response,
    const AvbdOgcRigidBoxGeometry *rigidBox) {
  response = AvbdOgcTangentResponse();
  if (!compileCurrentOgcNormalResponse(
          geometry, softParticles, numSoftParticles, dynamicTarget,
          1.0f, response.normalResponse, rigidBox))
    return false;

  // Velocity friction is an isotropic disk, so its result is invariant to
  // tangent-frame rotation.  Rebuild from the authoritative current normal
  // instead of trusting a prepared basis that may belong to an adjacent box
  // face after current-pose re-query.
  if (physx::PxAbs(response.normalResponse.normal.x) < 0.9f)
    response.tangents[0] = response.normalResponse.normal.cross(
        physx::PxVec3(1.0f, 0.0f, 0.0f)).getNormalized();
  else
    response.tangents[0] = response.normalResponse.normal.cross(
        physx::PxVec3(0.0f, 1.0f, 0.0f)).getNormalized();
  response.tangents[1] =
      response.normalResponse.normal.cross(response.tangents[0]);
  if (!response.tangents[0].isFinite() ||
      !response.tangents[1].isFinite())
    return false;

  physx::PxVec3 targetLinearJacobian[2] = {
      physx::PxVec3(0.0f), physx::PxVec3(0.0f)};
  physx::PxVec3 targetAngularJacobian[2] = {
      physx::PxVec3(0.0f), physx::PxVec3(0.0f)};
  if (dynamicTarget) {
    for (physx::PxU32 axis = 0u; axis < 2u; ++axis) {
      // Relative tangent velocity is always source minus target.  This sign
      // is therefore independent of whether the source is dynamic or
      // prescribed; source mobility only controls the soft response term.
      targetLinearJacobian[axis] = -response.tangents[axis];
      dynamicTarget->projectLockedLinearVector(
          targetLinearJacobian[axis]);
      response.targetLinearDeltaPerImpulse[axis] =
          targetLinearJacobian[axis] * dynamicTarget->invMass;
      targetAngularJacobian[axis] =
          -response.normalResponse.current.targetOffset.cross(
              response.tangents[axis]);
      dynamicTarget->projectLockedAngularVector(
          targetAngularJacobian[axis]);
      response.targetAngularDeltaPerImpulse[axis] =
          dynamicTarget->invInertiaWorld * targetAngularJacobian[axis];
      dynamicTarget->projectLockedAngularVector(
          response.targetAngularDeltaPerImpulse[axis]);
      if (!targetLinearJacobian[axis].isFinite() ||
          !targetAngularJacobian[axis].isFinite() ||
          !response.targetLinearDeltaPerImpulse[axis].isFinite() ||
          !response.targetAngularDeltaPerImpulse[axis].isFinite())
        return false;
    }
  }

  const physx::PxReal softResponse = response.normalResponse.softResponse;
  response.response00 = softResponse +
      targetLinearJacobian[0].dot(
          response.targetLinearDeltaPerImpulse[0]) +
      targetAngularJacobian[0].dot(
          response.targetAngularDeltaPerImpulse[0]);
  response.response01 =
      targetLinearJacobian[0].dot(
          response.targetLinearDeltaPerImpulse[1]) +
      targetAngularJacobian[0].dot(
          response.targetAngularDeltaPerImpulse[1]);
  response.response11 = softResponse +
      targetLinearJacobian[1].dot(
          response.targetLinearDeltaPerImpulse[1]) +
      targetAngularJacobian[1].dot(
          response.targetAngularDeltaPerImpulse[1]);
  response.determinant =
      response.response00 * response.response11 -
      response.response01 * response.response01;
  return physx::PxIsFinite(response.response00) &&
      physx::PxIsFinite(response.response01) &&
      physx::PxIsFinite(response.response11) &&
      physx::PxIsFinite(response.determinant) &&
      response.response00 >= 0.0f && response.response11 >= 0.0f &&
      response.determinant >= -1.0e-8f &&
      (response.determinant > 1.0e-12f ||
       (physx::PxAbs(response.response01) <= 1.0e-8f &&
        (response.response00 > 1.0e-12f ||
         response.response11 > 1.0e-12f)));
}

bool applyOgcTangentVelocityResponse(
    const AvbdOgcTangentResponse &response,
    AvbdSoftContact &contact,
    AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    AvbdSolverBody *dynamicTarget,
    physx::PxReal dt) {
  const AvbdSoftContactGeometry &geometry = contact.geometry;
  AvbdSoftContactAugmentedState &state = contact.state;
  if (!softParticles || numSoftParticles == 0u ||
      !physx::PxIsFinite(dt) || dt <= 0.0f ||
      geometry.tangentOwner != AvbdSoftContactTangentOwner::eVELOCITY ||
      !physx::PxIsFinite(geometry.friction) || geometry.friction <= 0.0f ||
      !physx::PxIsFinite(state.alLambda) || !physx::PxIsFinite(state.k))
    return false;

  physx::PxVec3 sourceVelocity(0.0f);
  if (response.normalResponse.sourceMobility ==
      AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT) {
    if (!geometry.surfacePoint.isFinite() ||
        !state.surfacePointPrev.isFinite())
      return false;
    sourceVelocity =
        (geometry.surfacePoint - state.surfacePointPrev) / dt;
    sourceVelocity -= response.normalResponse.normal *
        sourceVelocity.dot(response.normalResponse.normal);
    const physx::PxReal speedSquared = sourceVelocity.magnitudeSquared();
    const physx::PxReal maxKinematicSurfaceSpeed = 12.0f;
    if (!sourceVelocity.isFinite() || !physx::PxIsFinite(speedSquared))
      return false;
    if (speedSquared >
        maxKinematicSurfaceSpeed * maxKinematicSurfaceSpeed)
      sourceVelocity *= maxKinematicSurfaceSpeed /
          physx::PxSqrt(speedSquared);
    // The prescribed surface witness is velocity-epoch state, not an AL
    // spring anchor.  Consume it exactly once even when the target is locked
    // or the Coulomb disk later produces a zero impulse.
    state.surfacePointPrev = geometry.surfacePoint;
  } else if (!getOgcNormalResponseQueryVelocity(
                 response.normalResponse, softParticles,
                 numSoftParticles, sourceVelocity)) {
    return false;
  }

  physx::PxVec3 targetVelocity(0.0f);
  if (dynamicTarget) {
    if (!dynamicTarget->linearVelocity.isFinite() ||
        !dynamicTarget->angularVelocity.isFinite())
      return false;
    targetVelocity = dynamicTarget->linearVelocity +
        dynamicTarget->angularVelocity.cross(
            response.normalResponse.current.targetOffset);
  }
  const physx::PxVec3 relativeVelocity = sourceVelocity - targetVelocity;
  const physx::PxReal velocity0 =
      relativeVelocity.dot(response.tangents[0]);
  const physx::PxReal velocity1 =
      relativeVelocity.dot(response.tangents[1]);
  if (!relativeVelocity.isFinite() || !physx::PxIsFinite(velocity0) ||
      !physx::PxIsFinite(velocity1))
    return false;

  physx::PxReal impulse0 = 0.0f;
  physx::PxReal impulse1 = 0.0f;
  if (response.determinant > 1.0e-12f) {
    impulse0 =
        (-response.response11 * velocity0 +
         response.response01 * velocity1) /
        response.determinant;
    impulse1 =
        (response.response01 * velocity0 -
         response.response00 * velocity1) /
        response.determinant;
  } else {
    // A target may lock exactly one tangent axis.  Preserve the admissible
    // one-dimensional response instead of rejecting the complete contact.
    if (response.response00 > 1.0e-12f)
      impulse0 = -velocity0 / response.response00;
    if (response.response11 > 1.0e-12f)
      impulse1 = -velocity1 / response.response11;
  }
  physx::PxReal normalLoad = physx::PxMax(-state.alLambda, 0.0f);
  if (response.normalResponse.sourceMobility ==
      AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT) {
    normalLoad = physx::PxMax(
        physx::PxAbs(state.alLambda),
        state.k * physx::PxMax(
            0.0f, -response.normalResponse.constraintValue));
  }
  const physx::PxReal tangentImpulseLimit =
      geometry.friction * normalLoad * dt;
  if (!physx::PxIsFinite(impulse0) || !physx::PxIsFinite(impulse1) ||
      !physx::PxIsFinite(normalLoad) || normalLoad < 0.0f ||
      !physx::PxIsFinite(tangentImpulseLimit) ||
      tangentImpulseLimit < 0.0f)
    return false;
  avbdProjectImpulseCone(tangentImpulseLimit, impulse0, impulse1);
  if (physx::PxAbs(impulse0) <= 1.0e-12f &&
      physx::PxAbs(impulse1) <= 1.0e-12f)
    return false;

  const physx::PxVec3 softImpulse =
      response.tangents[0] * impulse0 +
      response.tangents[1] * impulse1;
  physx::PxVec3 candidateSoftVelocities[AVBD_CONTACT_MAX_PARTICLES];
  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.normalResponse.particleCount;
       ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.normalResponse.particleIndices[supportIndex];
    const physx::PxReal weight =
        response.normalResponse.particleWeights[supportIndex];
    if (particleIndex >= numSoftParticles)
      return false;
    candidateSoftVelocities[supportIndex] =
        softParticles[particleIndex].velocity +
        softImpulse *
            (softParticles[particleIndex].invMass * weight);
    const physx::PxVec3 &candidate =
        candidateSoftVelocities[supportIndex];
    if (!candidate.isFinite() || physx::PxAbs(candidate.x) > 1.0e6f ||
        physx::PxAbs(candidate.y) > 1.0e6f ||
        physx::PxAbs(candidate.z) > 1.0e6f)
      return false;
  }

  physx::PxVec3 candidateLinearVelocity(0.0f);
  physx::PxVec3 candidateAngularVelocity(0.0f);
  if (dynamicTarget) {
    candidateLinearVelocity = dynamicTarget->linearVelocity +
        response.targetLinearDeltaPerImpulse[0] * impulse0 +
        response.targetLinearDeltaPerImpulse[1] * impulse1;
    candidateAngularVelocity = dynamicTarget->angularVelocity +
        response.targetAngularDeltaPerImpulse[0] * impulse0 +
        response.targetAngularDeltaPerImpulse[1] * impulse1;
    dynamicTarget->projectLockedLinearVector(candidateLinearVelocity);
    dynamicTarget->projectLockedAngularVector(candidateAngularVelocity);
    if (!candidateLinearVelocity.isFinite() ||
        !candidateAngularVelocity.isFinite())
      return false;
  }

  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.normalResponse.particleCount;
       ++supportIndex) {
    softParticles[response.normalResponse.particleIndices[supportIndex]]
        .velocity = candidateSoftVelocities[supportIndex];
  }
  if (dynamicTarget) {
    dynamicTarget->linearVelocity = candidateLinearVelocity;
    dynamicTarget->angularVelocity = candidateAngularVelocity;
    dynamicTarget->projectLockedVelocities();
  }
  return true;
}

bool buildOgcSoftPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    physx::PxReal softResponseScale,
    physx::PxReal lambda,
    AvbdOgcSoftPositionCandidate &candidate) {
  if (!softParticles || response.particleCount == 0u ||
      response.particleCount > AVBD_CONTACT_MAX_PARTICLES ||
      !physx::PxIsFinite(softResponseScale) || softResponseScale <= 0.0f ||
      !physx::PxIsFinite(lambda) || lambda <= 0.0f)
    return false;

  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    if (particleIndex >= numSoftParticles ||
        !avbdSoftBodyContainsParticle(
            sourceBody, particleIndex, numSoftParticles))
      return false;
    const AvbdSoftParticle &particle = softParticles[particleIndex];
    const physx::PxReal weight = response.particleWeights[supportIndex];
    if (!particle.position.isFinite() ||
        !particle.initialPosition.isFinite() ||
        !physx::PxIsFinite(particle.invMass) ||
        !physx::PxIsFinite(weight))
      return false;
    candidate.particleDeltas[supportIndex] = physx::PxVec3(0.0f);
    if (particle.invMass <= 0.0f)
      continue;
    const physx::PxVec3 delta = response.normal *
        (softResponseScale * particle.invMass * weight * lambda);
    if (!delta.isFinite() || !(particle.position + delta).isFinite() ||
        !(particle.initialPosition + delta).isFinite())
      return false;
    candidate.particleDeltas[supportIndex] = delta;
  }
  return true;
}

bool buildOgcDeformablePairPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    const AvbdSoftBody &targetBody,
    physx::PxReal lambda,
    AvbdOgcSoftPositionCandidate &candidate) {
  if (!softParticles || response.particleCount == 0u ||
      response.particleCount > AVBD_CONTACT_MAX_PARTICLES ||
      !physx::PxIsFinite(lambda) || lambda <= 0.0f)
    return false;

  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    if (particleIndex >= numSoftParticles)
      return false;
    const bool belongsToSource = avbdSoftBodyContainsParticle(
        sourceBody, particleIndex, numSoftParticles);
    const bool belongsToTarget = avbdSoftBodyContainsParticle(
        targetBody, particleIndex, numSoftParticles);
    if (belongsToSource == belongsToTarget)
      return false;
    const AvbdSoftParticle &particle = softParticles[particleIndex];
    const physx::PxReal weight = response.particleWeights[supportIndex];
    if (!particle.position.isFinite() ||
        !particle.initialPosition.isFinite() ||
        !physx::PxIsFinite(particle.invMass) ||
        !physx::PxIsFinite(weight))
      return false;
    candidate.particleDeltas[supportIndex] = physx::PxVec3(0.0f);
    if (particle.invMass <= 0.0f)
      continue;
    const physx::PxVec3 delta = response.normal *
        (particle.invMass * weight * lambda);
    if (!delta.isFinite() || !(particle.position + delta).isFinite() ||
        !(particle.initialPosition + delta).isFinite())
      return false;
    candidate.particleDeltas[supportIndex] = delta;
  }
  return true;
}

static bool admitOgcPositionCandidateForBody(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    physx::PxReal alpha,
    physx::PxReal minimumDeterminant,
    bool requireAllSupports) {
  if (!softParticles || response.particleCount == 0u ||
      response.particleCount > AVBD_CONTACT_MAX_PARTICLES ||
      !physx::PxIsFinite(alpha) || alpha <= 0.0f ||
      !physx::PxIsFinite(minimumDeterminant) || minimumDeterminant <= 0.0f)
    return false;

  bool hasBodySupport = false;
  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    if (particleIndex >= numSoftParticles)
      return false;
    if (!avbdSoftBodyContainsParticle(
            sourceBody, particleIndex, numSoftParticles)) {
      if (requireAllSupports)
        return false;
      continue;
    }
    hasBodySupport = true;
    const AvbdSoftParticle &particle = softParticles[particleIndex];
    const physx::PxVec3 delta =
        candidate.particleDeltas[supportIndex] * alpha;
    if (!delta.isFinite() || !(particle.position + delta).isFinite() ||
        !(particle.initialPosition + delta).isFinite())
      return false;
  }

  const auto candidatePositionFor =
      [&response, &candidate, softParticles,
       alpha](physx::PxU32 particleIndex) -> physx::PxVec3 {
    for (physx::PxU32 supportIndex = 0u;
         supportIndex < response.particleCount; ++supportIndex) {
      if (response.particleIndices[supportIndex] == particleIndex)
        return softParticles[particleIndex].position +
            candidate.particleDeltas[supportIndex] * alpha;
    }
    return softParticles[particleIndex].position;
  };

  bool hasSubthresholdIncidentTet = false;
  bool improvesSubthresholdIncidentTet = false;
  const physx::PxReal recoveryTolerance = 1.0e-6f;
  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    if (!avbdSoftBodyContainsParticle(
            sourceBody, particleIndex, numSoftParticles))
      continue;
    const physx::PxU32 localIndex =
        particleIndex - sourceBody.compiled.particleStart;
    if (localIndex >= sourceBody.compiled.elementAdjacency.size())
      return false;
    const AvbdParticleElementAdjacency &adjacency =
        sourceBody.compiled.elementAdjacency[localIndex];
    for (physx::PxU32 refIndex = 0u;
         refIndex < adjacency.tetRefs.size(); ++refIndex) {
      const AvbdParticleElementRef &ref = adjacency.tetRefs[refIndex];
      if (ref.index >= sourceBody.compiled.tetElements.size())
        return false;
      const AvbdTetElement &tet = sourceBody.compiled.tetElements[ref.index];
      if (tet.p0 >= numSoftParticles || tet.p1 >= numSoftParticles ||
          tet.p2 >= numSoftParticles || tet.p3 >= numSoftParticles)
        return false;

      const physx::PxVec3 currentP0 = softParticles[tet.p0].position;
      const physx::PxVec3 currentE1 =
          softParticles[tet.p1].position - currentP0;
      const physx::PxVec3 currentE2 =
          softParticles[tet.p2].position - currentP0;
      const physx::PxVec3 currentE3 =
          softParticles[tet.p3].position - currentP0;
      physx::PxReal currentDeterminant = 0.0f;
      physx::PxVec3 unusedCurrentGradient;
      avbdEvaluateTetDeterminantAndGradient(
          tet, 0u, currentE1, currentE2, currentE3,
          currentDeterminant, unusedCurrentGradient);

      const physx::PxVec3 p0 = candidatePositionFor(tet.p0);
      const physx::PxVec3 e1 = candidatePositionFor(tet.p1) - p0;
      const physx::PxVec3 e2 = candidatePositionFor(tet.p2) - p0;
      const physx::PxVec3 e3 = candidatePositionFor(tet.p3) - p0;
      physx::PxReal determinant = 0.0f;
      physx::PxVec3 unusedGradient;
      avbdEvaluateTetDeterminantAndGradient(
          tet, 0u, e1, e2, e3, determinant, unusedGradient);
      if (!physx::PxIsFinite(currentDeterminant) ||
          !physx::PxIsFinite(determinant))
        return false;
      if (currentDeterminant >= minimumDeterminant) {
        if (determinant < minimumDeterminant)
          return false;
      } else {
        hasSubthresholdIncidentTet = true;
        if (determinant + recoveryTolerance < currentDeterminant)
          return false;
        if (determinant > currentDeterminant + recoveryTolerance)
          improvesSubthresholdIncidentTet = true;
      }
    }
  }
  return hasBodySupport &&
      (!hasSubthresholdIncidentTet || improvesSubthresholdIncidentTet);
}

bool admitOgcSoftPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    physx::PxReal alpha,
    physx::PxReal minimumDeterminant) {
  return admitOgcPositionCandidateForBody(
      response, candidate, softParticles, numSoftParticles, sourceBody,
      alpha, minimumDeterminant, true);
}

bool admitOgcDeformablePairPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    const AvbdSoftBody &targetBody,
    physx::PxReal alpha,
    physx::PxReal minimumDeterminant) {
  return admitOgcPositionCandidateForBody(
             response, candidate, softParticles, numSoftParticles,
             sourceBody, alpha, minimumDeterminant, false) &&
      admitOgcPositionCandidateForBody(
             response, candidate, softParticles, numSoftParticles,
             targetBody, alpha, minimumDeterminant, false);
}

bool evaluateOgcSoftPositionCandidateQueryPoint(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    physx::PxReal alpha,
    physx::PxVec3 &queryPoint) {
  if (!physx::PxIsFinite(alpha) || alpha <= 0.0f ||
      !response.queryPoint.isFinite() || response.particleCount == 0u ||
      response.particleCount > AVBD_CONTACT_MAX_PARTICLES)
    return false;
  queryPoint = response.queryPoint;
  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxReal weight = response.particleWeights[supportIndex];
    if (!physx::PxIsFinite(weight))
      return false;
    queryPoint += candidate.particleDeltas[supportIndex] * (alpha * weight);
    if (!queryPoint.isFinite())
      return false;
  }
  return true;
}

bool finalizeOgcRigidPositionCandidate(
    const AvbdSolverBody &currentBody,
    AvbdSolverBody &candidateBody) {
  candidateBody.projectLockedPose(
      currentBody.position, currentBody.rotation);
  physx::PxQuat actualRotationDelta =
      candidateBody.rotation * currentBody.rotation.getConjugate();
  if (actualRotationDelta.w < 0.0f)
    actualRotationDelta = -actualRotationDelta;
  if (!actualRotationDelta.isFinite())
    return false;
  actualRotationDelta.normalize();
  const physx::PxMat33 rotationDelta(actualRotationDelta);
  candidateBody.invInertiaWorld =
      rotationDelta * currentBody.invInertiaWorld *
      rotationDelta.getTranspose();
  return candidateBody.position.isFinite() &&
      candidateBody.rotation.isFinite() &&
      candidateBody.invInertiaWorld.column0.isFinite() &&
      candidateBody.invInertiaWorld.column1.isFinite() &&
      candidateBody.invInertiaWorld.column2.isFinite();
}

bool buildOgcRigidPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdSolverBody &currentBody,
    physx::PxReal lambda,
    physx::PxReal alpha,
    AvbdSolverBody &candidateBody) {
  if (!physx::PxIsFinite(lambda) || lambda <= 0.0f ||
      !physx::PxIsFinite(alpha) || alpha <= 0.0f)
    return false;
  const physx::PxReal scaledLambda = lambda * alpha;
  candidateBody = currentBody;
  candidateBody.position +=
      response.targetLinearDeltaPerLambda * scaledLambda;
  const physx::PxVec3 angularDelta =
      response.targetAngularDeltaPerLambda * scaledLambda;
  if (!candidateBody.position.isFinite() || !angularDelta.isFinite())
    return false;
  if (angularDelta.magnitudeSquared() > 1.0e-16f) {
    const physx::PxQuat dq(
        angularDelta.x, angularDelta.y, angularDelta.z, 0.0f);
    candidateBody.rotation =
        (candidateBody.rotation + dq * candidateBody.rotation * 0.5f)
            .getNormalized();
  }
  return finalizeOgcRigidPositionCandidate(currentBody, candidateBody);
}

void commitOgcRigidPositionCandidate(
    const AvbdSolverBody &candidateBody,
    AvbdSolverBody &body) {
  body.position = candidateBody.position;
  body.rotation = candidateBody.rotation;
  body.invInertiaWorld = candidateBody.invInertiaWorld;
}

bool evaluateCurrentOgcNormalConstraint(
    const AvbdSoftContactGeometry &geometry,
    const AvbdOgcNormalResponse &response,
    const AvbdSolverBody *dynamicTarget,
    const physx::PxVec3 &queryPoint,
    physx::PxReal &constraintValue) {
  constraintValue = 0.0f;
  if (!response.normal.isFinite() ||
      response.normal.magnitudeSquared() <= 1.0e-12f ||
      !queryPoint.isFinite())
    return false;
  if (response.sourceMobility ==
      AvbdOgcNormalSourceMobility::eDYNAMIC_SOFT) {
    AvbdOgcCurrentPairGeometry current;
    if (!getCurrentOgcPairGeometry(
            geometry, dynamicTarget, queryPoint, current) ||
        !physx::PxIsFinite(current.signedGap))
      return false;
    constraintValue = current.signedGap;
    return true;
  }

  if (!dynamicTarget || !geometry.hasRigidBodyTarget() ||
      !dynamicTarget->position.isFinite() ||
      !dynamicTarget->rotation.isFinite() ||
      !geometry.rigidLocalPoint.isFinite() ||
      !geometry.surfacePoint.isFinite() ||
      !physx::PxIsFinite(geometry.depth))
    return false;
  const physx::PxVec3 targetOffset =
      dynamicTarget->rotation.rotate(geometry.rigidLocalPoint);
  const physx::PxVec3 targetPoint =
      dynamicTarget->position + targetOffset;
  if (!targetOffset.isFinite() || !targetPoint.isFinite())
    return false;
  constraintValue =
      (targetPoint - geometry.surfacePoint).dot(response.normal) -
      geometry.depth;
  // The AL objective may floor an active shell row at -depth to retain its
  // penalty load.  A geometric recovery must not inherit that floor: doing
  // so makes every partial correction appear unchanged and defeats the
  // candidate-improvement transaction.  This owner evaluates the actual
  // current target-to-shell separation only.
  return physx::PxIsFinite(constraintValue);
}

void commitOgcSoftPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    physx::PxReal alpha) {
  PX_ASSERT(softParticles);
  PX_ASSERT(physx::PxIsFinite(alpha) && alpha > 0.0f);
  for (physx::PxU32 supportIndex = 0u;
       supportIndex < response.particleCount; ++supportIndex) {
    const physx::PxU32 particleIndex =
        response.particleIndices[supportIndex];
    PX_ASSERT(particleIndex < numSoftParticles);
    if (particleIndex >= numSoftParticles)
      continue;
    const physx::PxVec3 delta =
        candidate.particleDeltas[supportIndex] * alpha;
    if (delta.magnitudeSquared() == 0.0f)
      continue;
    softParticles[particleIndex].position += delta;
    softParticles[particleIndex].initialPosition += delta;
  }
}

void clampRecoveredOgcPairNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    AvbdOgcVelocityContactDomain contactDomain,
    AvbdOgcNormalTargetMobility targetMobility,
    physx::PxReal configLengthScale, AvbdSolverStats *stats) {
  (void)stats;
  if (!softParticles || numSoftParticles == 0u || !softContacts ||
      numSoftContacts == 0u || !pairStates || numPairStates == 0u)
    return;

  const physx::PxReal nearSurfaceTolerance = physx::PxMax(
      1.0e-5f,
      1.0e-4f * physx::PxMax(configLengthScale, 1.0e-6f));
  for (physx::PxU32 pairIndex = 0u; pairIndex < numPairStates;
       ++pairIndex) {
    AvbdOgcPairState &pair = pairStates[pairIndex];
    if (!pair.geometry.active ||
        !pair.solve.hasPendingLocalVelocity(contactDomain))
      continue;
    const bool dynamicPair =
        pair.key.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY;
    const bool worldStaticPair =
        pair.key.targetKind == AvbdSoftContactTargetKind::eWORLD_STATIC;
    const bool deformablePair =
        pair.key.targetKind ==
            AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE;
    if ((targetMobility == AvbdOgcNormalTargetMobility::eDYNAMIC_RIGID &&
         !dynamicPair) ||
        (targetMobility == AvbdOgcNormalTargetMobility::eWORLD_STATIC &&
         !worldStaticPair) ||
        (targetMobility ==
             AvbdOgcNormalTargetMobility::eDEFORMABLE_SURFACE &&
         !deformablePair))
      continue;

    // This typed pair result has exactly one consumer. Malformed or stale
    // contact geometry fails closed instead of remaining available to a
    // later phase with a different contact domain.
    pair.solve.localVelocityConsumed = true;
    const physx::PxU32 contactIndex = pair.solve.localVelocityContact;
    if (contactIndex >= numSoftContacts)
      continue;
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    if (!pair.matches(geometry.source.type, geometry.targetKind,
                      geometry.queryBodyIndex, geometry.targetIndex,
                      geometry.source.primitiveKey) ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    AvbdSolverBody *dynamicTarget = nullptr;
    if (dynamicPair) {
      if (!bodies || geometry.targetIndex >= numBodies)
        continue;
      dynamicTarget = &bodies[geometry.targetIndex];
      if (dynamicTarget->invMass <= 0.0f ||
          !dynamicTarget->position.isFinite() ||
          !dynamicTarget->rotation.isFinite() ||
          !dynamicTarget->linearVelocity.isFinite() ||
          !dynamicTarget->angularVelocity.isFinite() ||
          !dynamicTarget->invInertiaWorld.column0.isFinite() ||
          !dynamicTarget->invInertiaWorld.column1.isFinite() ||
          !dynamicTarget->invInertiaWorld.column2.isFinite())
        continue;
    }

    AvbdOgcNormalResponse response;
    if (!compileCurrentOgcNormalResponse(
            geometry, softParticles, numSoftParticles, dynamicTarget,
            /*softResponseScale=*/1.0f, response,
            pair.geometry.rigidBox.valid
                ? &pair.geometry.rigidBox : nullptr) ||
        !physx::PxIsFinite(response.current.signedGap) ||
        response.current.signedGap > nearSurfaceTolerance)
      continue;

    physx::PxVec3 queryVelocity(0.0f);
    if (!getOgcNormalResponseQueryVelocity(
            response, softParticles, numSoftParticles, queryVelocity))
      continue;
    physx::PxVec3 targetVelocity(0.0f);
    if (dynamicTarget) {
      targetVelocity = dynamicTarget->linearVelocity +
          dynamicTarget->angularVelocity.cross(
              response.current.targetOffset);
    }
    const physx::PxReal relativeNormalVelocity =
        (queryVelocity - targetVelocity).dot(response.normal);
    if (!physx::PxIsFinite(relativeNormalVelocity) ||
        relativeNormalVelocity >= -1.0e-6f)
      continue;
    const physx::PxReal impulse =
        -relativeNormalVelocity / response.effectiveResponse;
    if (!physx::PxIsFinite(impulse) || impulse <= 0.0f)
      continue;

    physx::PxVec3 particleVelocityDeltas[AVBD_CONTACT_MAX_PARTICLES];
    bool finiteCandidates = true;
    for (physx::PxU32 supportIndex = 0u;
         supportIndex < response.particleCount; ++supportIndex) {
      const physx::PxU32 particleIndex =
          response.particleIndices[supportIndex];
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      particleVelocityDeltas[supportIndex] = physx::PxVec3(0.0f);
      if (particle.invMass <= 0.0f)
        continue;
      const physx::PxVec3 delta = response.normal *
          (particle.invMass * response.particleWeights[supportIndex] *
           impulse);
      if (!delta.isFinite() || !(particle.velocity + delta).isFinite()) {
        finiteCandidates = false;
        break;
      }
      particleVelocityDeltas[supportIndex] = delta;
    }

    physx::PxVec3 candidateLinearVelocity(0.0f);
    physx::PxVec3 candidateAngularVelocity(0.0f);
    if (dynamicTarget) {
      candidateLinearVelocity = dynamicTarget->linearVelocity +
          response.targetLinearDeltaPerLambda * impulse;
      candidateAngularVelocity = dynamicTarget->angularVelocity +
          response.targetAngularDeltaPerLambda * impulse;
      dynamicTarget->projectLockedLinearVector(candidateLinearVelocity);
      dynamicTarget->projectLockedAngularVector(candidateAngularVelocity);
      finiteCandidates = finiteCandidates &&
          candidateLinearVelocity.isFinite() &&
          candidateAngularVelocity.isFinite();
    }
    if (!finiteCandidates)
      continue;

    for (physx::PxU32 supportIndex = 0u;
         supportIndex < response.particleCount; ++supportIndex) {
      const physx::PxVec3 &delta =
          particleVelocityDeltas[supportIndex];
      if (delta.magnitudeSquared() > 0.0f)
        softParticles[response.particleIndices[supportIndex]].velocity +=
            delta;
    }
    if (dynamicTarget) {
      dynamicTarget->linearVelocity = candidateLinearVelocity;
      dynamicTarget->angularVelocity = candidateAngularVelocity;
      dynamicTarget->projectLockedVelocities();
    }
  }
}

void applyKinematicOgcNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt,
    physx::PxU32 sweeps, AvbdSolverStats *stats) {
  (void)stats;
  if (!bodies || numBodies == 0u || !softParticles ||
      numSoftParticles == 0u || !softContacts || numSoftContacts == 0u ||
      sweeps == 0u || !physx::PxIsFinite(dt) || dt <= 0.0f)
    return;

  for (physx::PxU32 sweep = 0u; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 contactIndex = 0u;
         contactIndex < numSoftContacts; ++contactIndex) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex >= numBodies)
        continue;
      AvbdSolverBody &body = bodies[geometry.targetIndex];
      if (body.invMass <= 0.0f)
        continue;

      AvbdOgcNormalResponse response;
      if (!compileCurrentOgcNormalResponse(
              geometry, softParticles, numSoftParticles, &body,
              /*softResponseScale=*/1.0f, response) ||
          response.sourceMobility !=
              AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT ||
          !physx::PxIsFinite(response.constraintValue) ||
          response.constraintValue >= -1.0e-5f)
        continue;

      const physx::PxReal approachSpeed =
          body.linearVelocity.magnitude() +
          body.angularVelocity.magnitude() *
              response.current.targetOffset.magnitude() +
          gravity.magnitude() * dt;
      physx::PxReal sweepCap =
          physx::PxMax(approachSpeed * dt * 0.5f, 0.04f);
      const physx::PxVec3 meshStep = geometry.surfacePoint -
          softContacts[contactIndex].state.surfacePointPrev;
      sweepCap = physx::PxMax(sweepCap, meshStep.magnitude() * 1.5f);
      if (response.constraintValue < -0.05f)
        sweepCap = physx::PxMax(
            sweepCap, -response.constraintValue * 0.6f);
      const physx::PxReal requestedCorrection =
          physx::PxMin(-response.constraintValue, sweepCap);
      const physx::PxReal lambda =
          requestedCorrection / response.effectiveResponse;
      if (!physx::PxIsFinite(lambda) || lambda <= 0.0f)
        continue;

      physx::PxReal alpha = 1.0f;
      bool accepted = false;
      AvbdSolverBody acceptedBody = body;
      const physx::PxReal improvementTolerance = 1.0e-6f;
      for (physx::PxU32 attempt = 0u; attempt < 8u && !accepted;
           ++attempt) {
        AvbdSolverBody candidateBody;
        physx::PxReal candidateConstraint = 0.0f;
        if (buildOgcRigidPositionCandidate(
                response, body, lambda, alpha, candidateBody) &&
            evaluateCurrentOgcNormalConstraint(
                geometry, response, &candidateBody,
                response.queryPoint, candidateConstraint) &&
            physx::PxIsFinite(candidateConstraint) &&
            candidateConstraint >
                response.constraintValue + improvementTolerance) {
          acceptedBody = candidateBody;
          accepted = true;
        } else {
          alpha *= 0.5f;
        }
      }
      if (!accepted)
        continue;
      commitOgcRigidPositionCandidate(acceptedBody, body);
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
}

void clampKinematicOgcInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt, AvbdSolverStats *stats) {
  (void)stats;
  if (!bodies || numBodies == 0u || !softParticles ||
      numSoftParticles == 0u || !softContacts || numSoftContacts == 0u ||
      !physx::PxIsFinite(dt) || dt <= 0.0f)
    return;
  const physx::PxReal invDt = 1.0f / dt;

  for (physx::PxU32 bodyIndex = 0u; bodyIndex < numBodies;
       ++bodyIndex) {
    AvbdSolverBody &body = bodies[bodyIndex];
    if (body.invMass <= 0.0f)
      continue;
    physx::PxU32 dominantContact = PX_MAX_U32;
    physx::PxReal worstConstraint = 0.0f;
    for (physx::PxU32 contactIndex = 0u;
         contactIndex < numSoftContacts; ++contactIndex) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex != bodyIndex)
        continue;
      AvbdOgcNormalResponse response;
      if (!compileCurrentOgcNormalResponse(
              geometry, softParticles, numSoftParticles, &body,
              /*softResponseScale=*/1.0f, response) ||
          response.sourceMobility !=
              AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT ||
          !physx::PxIsFinite(response.constraintValue))
        continue;
      if (response.constraintValue < worstConstraint) {
        worstConstraint = response.constraintValue;
        dominantContact = contactIndex;
      }
    }
    if (dominantContact == PX_MAX_U32)
      continue;

    const AvbdSoftContact &contact = softContacts[dominantContact];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    AvbdOgcNormalResponse response;
    if (!compileCurrentOgcNormalResponse(
            geometry, softParticles, numSoftParticles, &body,
            /*softResponseScale=*/1.0f, response) ||
        response.sourceMobility !=
            AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT)
      continue;
    const physx::PxVec3 meshVelocity =
        (geometry.surfacePoint - contact.state.surfacePointPrev) * invDt;
    const physx::PxVec3 targetVelocity =
        body.linearVelocity +
        body.angularVelocity.cross(response.current.targetOffset);
    const physx::PxReal relativeNormalVelocity =
        (targetVelocity - meshVelocity).dot(response.normal);
    if (!physx::PxIsFinite(relativeNormalVelocity) ||
        relativeNormalVelocity <= 0.0f)
      continue;
    const physx::PxReal impulse =
        -relativeNormalVelocity / response.effectiveResponse;
    if (!physx::PxIsFinite(impulse) || impulse >= 0.0f)
      continue;

    physx::PxVec3 candidateLinearVelocity =
        body.linearVelocity +
        response.targetLinearDeltaPerLambda * impulse;
    physx::PxVec3 candidateAngularVelocity =
        body.angularVelocity +
        response.targetAngularDeltaPerLambda * impulse;
    body.projectLockedLinearVector(candidateLinearVelocity);
    body.projectLockedAngularVector(candidateAngularVelocity);
    if (!candidateLinearVelocity.isFinite() ||
        !candidateAngularVelocity.isFinite())
      continue;
    body.linearVelocity = candidateLinearVelocity;
    body.angularVelocity = candidateAngularVelocity;
    body.projectLockedVelocities();
  }
}

} // namespace Dy
} // namespace physx
