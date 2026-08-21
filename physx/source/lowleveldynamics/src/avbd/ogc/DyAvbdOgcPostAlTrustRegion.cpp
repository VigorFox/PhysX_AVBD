// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

// Every post-AL position owner must consume the same pair trust region as the
// material and rigid GS updates.  In particular, a world-static recovery may
// not push a lower soft support through a dynamic box merely because its own
// target is a pedestal.  This helper applies the immutable Scene contact CSR
// to one proposed soft-particle displacement and returns the largest common
// current-pose DCD-safe fraction.  It never evaluates a swept pose or advances
// time.
physx::PxReal limitPostAlSoftParticleOgcCandidate(
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
      pair.trustRegion.refreshRequested = true;
      pair.trustRegion.remainingSafeDisplacement = 0.0f;
      const physx::PxReal admittedGap =
          currentGap + (candidateGap - currentGap) * localAlpha;
      pair.geometry.minimumGap =
          physx::PxMin(pair.geometry.minimumGap, admittedGap);
      pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
          pair.trustRegion.accumulatedRelativeDisplacement,
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
    if (!pair.geometry.active || !geometry.hasRigidBodyTarget() ||
        geometry.targetIndex >= numRigidBodies ||
        geometry.targetIndex != pair.key.targetBodyIndex ||
        geometry.queryBodyIndex != pair.key.sourceBodyIndex ||
        geometry.source.primitiveKey != pair.key.primitiveKey)
      continue;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!queryPoint.isFinite())
      continue;
    AvbdOgcCurrentPairGeometry currentGeometry;
    AvbdOgcCurrentPairGeometry candidateGeometry;
    if (!getCurrentOgcPairGeometry(
            geometry, &rigidBodies[geometry.targetIndex], queryPoint,
            currentGeometry,
            pair.geometry.rigidBox.valid
                ? &pair.geometry.rigidBox : nullptr) ||
        !getCurrentOgcPairGeometry(
            geometry, &rigidBodies[geometry.targetIndex],
            queryPoint + candidateDisplacement * ref.jacobianScale,
            candidateGeometry,
            pair.geometry.rigidBox.valid
                ? &pair.geometry.rigidBox : nullptr))
      continue;
    consumeGap(pair, currentGeometry.signedGap,
               candidateGeometry.signedGap);
  }

  if (plan->hasTriangleCoreSafetyPlan(numSoftParticles)) {
    if (!plan->hasOgcTriangleCoreGeometryPlan(numOgcContacts))
      return 0.0f;
    const AvbdOgcGeometryEpochView geometryEpoch =
        makeOgcGeometryEpochView(plan);
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
      const AvbdOgcTriangleCoreCertificate *certificate =
          geometryEpoch.getTriangleCore(ref.contactIndex, numOgcContacts);
      if (!certificate)
        return 0.0f;
      const physx::PxU32 pairIndex = plan->ogcPairIndices[ref.contactIndex];
      if (pairIndex >= plan->numOgcPairStates)
        continue;
      AvbdOgcPairState &pair = plan->ogcPairStates[pairIndex];
      const AvbdSoftContactGeometry &geometry =
          ogcContacts[ref.contactIndex].geometry;
      if (!pair.geometry.active || !geometry.hasRigidBodyTarget() ||
          geometry.targetIndex >= numRigidBodies ||
          geometry.targetIndex != pair.key.targetBodyIndex ||
          geometry.queryBodyIndex != pair.key.sourceBodyIndex ||
          geometry.source.primitiveKey != pair.key.primitiveKey)
        continue;
      physx::PxReal currentGap = 0.0f;
      physx::PxReal candidateGap = 0.0f;
      if (!getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, &rigidBodies[geometry.targetIndex], softParticles,
              numSoftParticles, currentGap, PX_MAX_U32,
              physx::PxVec3(0.0f), &pair.geometry.rigidBox,
              certificate) ||
          !getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, &rigidBodies[geometry.targetIndex], softParticles,
              numSoftParticles, candidateGap, particleIndex,
              candidateDisplacement, &pair.geometry.rigidBox,
              certificate))
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
physx::PxReal limitPostAlRigidOgcCandidate(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 targetBodyIndex, physx::PxU32 currentSourceBodyIndex,
    physx::PxU64 currentPrimitiveKey, const AvbdSolverBody &currentBody,
    const AvbdSolverBody &candidateBody,
    const AvbdOgcGeometryEpochView *geometryEpoch) {
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

    const AvbdOgcRigidBoxGeometry *rigidBox = geometryEpoch
        ? geometryEpoch->getRigidBox(contactIndex, numContacts) : nullptr;
    const AvbdOgcTriangleCoreCertificate *certificate = geometryEpoch
        ? geometryEpoch->getTriangleCore(contactIndex, numContacts) : nullptr;
    const bool triangleCore = certificate != nullptr;
    const physx::PxVec3 queryPoint = triangleCore
        ? physx::PxVec3(0.0f)
        : avbdGetSoftContactQueryPoint(geometry, softParticles);
    if (!triangleCore && !queryPoint.isFinite())
      continue;

    auto evaluateGap = [&](const AvbdSolverBody &body,
                           physx::PxReal &gap) {
      if (triangleCore)
        return getCurrentRigidBoxTriangleCoreFaceGap(
            geometry, &body, softParticles, numSoftParticles, gap,
            PX_MAX_U32, physx::PxVec3(0.0f), rigidBox, certificate);
      AvbdOgcCurrentPairGeometry currentGeometry;
      const bool valid = getCurrentOgcPairGeometry(
          geometry, &body, queryPoint, currentGeometry, rigidBox);
      gap = currentGeometry.signedGap;
      return valid;
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

} // namespace Dy
} // namespace physx
