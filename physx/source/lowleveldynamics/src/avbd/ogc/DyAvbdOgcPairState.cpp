// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcPairState.h"
#include "avbd/contact/DyAvbdContactRigidPrimitives.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool compileOgcPairProviderPlan(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const AvbdRigidBox *rigidBoxes, physx::PxU32 numRigidBoxes,
    physx::PxU32 numSoftBodies, physx::PxU32 numDynamicRigidBodies,
    physx::PxU32 targetMask,
    physx::PxArray<AvbdOgcPairState> &pairStates,
    physx::PxArray<physx::PxU32> &pairIndices) {
  pairStates.clear();
  pairIndices.clear();
  if ((numContacts > 0u && !contacts) ||
      (numRigidBoxes > 0u && !rigidBoxes))
    return false;
  pairIndices.resize(numContacts);
  for (physx::PxU32 contactIndex = 0u; contactIndex < numContacts;
       ++contactIndex) {
    pairIndices[contactIndex] = PX_MAX_U32;
    const AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
    if (geometry.queryBodyIndex >= numSoftBodies)
      continue;
    const bool dynamicRigid =
        (targetMask & eOGC_PAIR_PROVIDER_DYNAMIC_RIGID) != 0u &&
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numDynamicRigidBodies;
    const bool worldStatic =
        (targetMask & eOGC_PAIR_PROVIDER_WORLD_STATIC) != 0u &&
        (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
         geometry.source.type == AvbdSoftContactSource::eGROUND) &&
        geometry.hasWorldStaticTarget();
    const bool deformable =
        (targetMask & eOGC_PAIR_PROVIDER_DEFORMABLE) != 0u &&
        geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
        geometry.targetKind ==
            AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
        geometry.targetIndex < numSoftBodies &&
        geometry.targetIndex != geometry.queryBodyIndex;
    if (!dynamicRigid && !worldStatic && !deformable)
      continue;

    physx::PxU32 pairIndex = PX_MAX_U32;
    for (physx::PxU32 candidateIndex = 0u;
         candidateIndex < pairStates.size(); ++candidateIndex) {
      if (pairStates[candidateIndex].matches(
              geometry.source.type, geometry.targetKind,
              geometry.queryBodyIndex, geometry.targetIndex,
              geometry.source.primitiveKey)) {
        pairIndex = candidateIndex;
        break;
      }
    }
    if (pairIndex == PX_MAX_U32) {
      pairIndex = pairStates.size();
      AvbdOgcPairState pair;
      pair.initializeKey(
          geometry.source.type, geometry.targetKind,
          geometry.queryBodyIndex, geometry.targetIndex,
          geometry.source.primitiveKey);
      pairStates.pushBack(pair);
    }

    AvbdOgcPairState &pair = pairStates[pairIndex];
    if (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF) {
      const AvbdRigidBox *rigidBox = nullptr;
      for (physx::PxU32 boxIndex = 0u; boxIndex < numRigidBoxes;
           ++boxIndex) {
        const AvbdRigidBox &candidate = rigidBoxes[boxIndex];
        if (candidate.targetKind == geometry.targetKind &&
            candidate.targetIndex == geometry.targetIndex &&
            candidate.primitiveKey == geometry.source.primitiveKey) {
          rigidBox = &candidate;
          break;
        }
      }
      if (rigidBox) {
        const physx::PxTransform shapeToTarget = dynamicRigid
            ? rigidBox->shapeToRigidBody
            : physx::PxTransform(rigidBox->center, rigidBox->rotation);
        if (!pair.geometry.rigidBox.bind(
                rigidBox->halfExtent, shapeToTarget)) {
          pairStates.clear();
          pairIndices.clear();
          return false;
        }
      }
    }
    pair.addContact();
    pairIndices[contactIndex] = pairIndex;
  }
  return pairIndices.size() == numContacts;
}

bool refreshCurrentOgcPairRegistry(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const AvbdRigidBox *rigidBoxes, physx::PxU32 numRigidBoxes,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numSoftBodies,
    physx::PxArray<AvbdOgcPairState> &pairRegistry,
    physx::PxArray<AvbdOgcPairState> &detectedPairScratch,
    physx::PxArray<physx::PxU32> &detectedPairIndexScratch,
    physx::PxArray<physx::PxU32> &detectedPairToRegistryScratch,
    physx::PxArray<physx::PxU32> &pairIndices) {
  if ((numContacts > 0u && !contacts) ||
      (numRigidBoxes > 0u && !rigidBoxes) ||
      (numSoftParticles > 0u && !softParticles) ||
      (numBodies > 0u && !bodies))
    return false;

  if (!compileOgcPairProviderPlan(
          contacts, numContacts, rigidBoxes, numRigidBoxes, numSoftBodies,
          numBodies,
          eOGC_PAIR_PROVIDER_WORLD_STATIC |
              eOGC_PAIR_PROVIDER_DYNAMIC_RIGID |
              eOGC_PAIR_PROVIDER_DEFORMABLE,
          detectedPairScratch, detectedPairIndexScratch))
    return false;
  if (detectedPairIndexScratch.size() != numContacts)
    return false;

  detectedPairToRegistryScratch.resize(detectedPairScratch.size());
  for (physx::PxU32 detectedIndex = 0;
       detectedIndex < detectedPairScratch.size(); ++detectedIndex) {
    const AvbdOgcPairState &detectedPair =
        detectedPairScratch[detectedIndex];
    physx::PxU32 registryIndex = PX_MAX_U32;
    for (physx::PxU32 candidateIndex = 0;
         candidateIndex < pairRegistry.size(); ++candidateIndex) {
      if (pairRegistry[candidateIndex].matches(
              detectedPair.key.sourceType, detectedPair.key.targetKind,
              detectedPair.key.sourceBodyIndex,
              detectedPair.key.targetBodyIndex,
              detectedPair.key.primitiveKey)) {
        registryIndex = candidateIndex;
        break;
      }
    }
    if (registryIndex == PX_MAX_U32) {
      registryIndex = pairRegistry.size();
      pairRegistry.pushBack(detectedPair);
    }

    AvbdOgcPairState &registryPair = pairRegistry[registryIndex];
    AvbdOgcRigidBoxGeometry rigidBox = registryPair.geometry.rigidBox;
    if (detectedPair.geometry.rigidBox.valid &&
        !rigidBox.bind(detectedPair.geometry.rigidBox.halfExtent,
                       detectedPair.geometry.rigidBox.shapeToTarget))
      return false;
    const AvbdOgcPairSolveState solveState = registryPair.solve;
    const physx::PxU32 nextEpoch = registryPair.geometry.epoch + 1u;
    registryPair.geometry = AvbdOgcPairGeometryState();
    registryPair.geometry.contactCount =
        detectedPair.geometry.contactCount;
    registryPair.geometry.rigidBox = rigidBox;
    registryPair.geometry.epoch = nextEpoch;
    registryPair.trustRegion = AvbdOgcPairTrustRegionState();
    registryPair.solve = solveState;
    // Local triangle resolution belongs to the detector geometry epoch that
    // produced its complete collision-to-simulation witness. Preserve the
    // remaining solve handoff state, but never carry this geometry verdict
    // across a fresh current-pose manifold.
    registryPair.solve.triangleCoreLocallyResolved = false;
    detectedPairToRegistryScratch[detectedIndex] = registryIndex;
  }

  pairIndices.resize(numContacts);
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    pairIndices[contactIndex] = PX_MAX_U32;
    const physx::PxU32 detectedIndex =
        detectedPairIndexScratch[contactIndex];
    if (detectedIndex >= detectedPairToRegistryScratch.size())
      continue;
    const physx::PxU32 pairIndex =
        detectedPairToRegistryScratch[detectedIndex];
    if (pairIndex >= pairRegistry.size())
      return false;

    const AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
    const bool dynamicRigid =
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies;
    const bool worldStatic =
        (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
         geometry.source.type == AvbdSoftContactSource::eGROUND) &&
        geometry.hasWorldStaticTarget();
    const bool deformable =
        geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
        geometry.hasDeformableSurfaceTarget() &&
        geometry.targetIndex < numSoftBodies;
    if ((!dynamicRigid && !worldStatic && !deformable) ||
        geometry.queryBodyIndex >= numSoftBodies ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    AvbdOgcPairState &pair = pairRegistry[pairIndex];
    const AvbdOgcRigidBoxGeometry *rigidBox =
        pair.geometry.rigidBox.valid ? &pair.geometry.rigidBox : nullptr;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    const AvbdSolverBody *dynamicTarget =
        dynamicRigid ? &bodies[geometry.targetIndex] : nullptr;
    AvbdOgcCurrentPairGeometry currentGeometry;
    physx::PxVec3 deformableTargetPoint(0.0f);
    bool geometryValid = false;
    if (deformable) {
      deformableTargetPoint =
          avbdGetSoftContactSurfacePoint(geometry, softParticles);
      const physx::PxReal normalSq = geometry.normal.magnitudeSquared();
      if (queryPoint.isFinite() && deformableTargetPoint.isFinite() &&
          physx::PxIsFinite(normalSq) && normalSq > 1.0e-12f) {
        currentGeometry.normal =
            geometry.normal * physx::PxRecipSqrt(normalSq);
        currentGeometry.targetOffset = physx::PxVec3(0.0f);
        currentGeometry.signedGap =
            (queryPoint - deformableTargetPoint).dot(currentGeometry.normal);
        geometryValid = physx::PxIsFinite(currentGeometry.signedGap);
      }
    } else {
      geometryValid = queryPoint.isFinite() &&
          getCurrentOgcPairGeometry(
              geometry, dynamicTarget, queryPoint, currentGeometry,
              rigidBox);
    }
    if (!geometryValid || !physx::PxIsFinite(currentGeometry.signedGap))
      continue;

    const physx::PxVec3 surfacePoint = dynamicRigid
        ? dynamicTarget->position + currentGeometry.targetOffset
        : (deformable ? deformableTargetPoint
                      : queryPoint - currentGeometry.normal *
                            currentGeometry.signedGap);
    if (!surfacePoint.isFinite())
      continue;
    if (!pair.geometry.active ||
        currentGeometry.signedGap < pair.geometry.referenceGap)
      pair.geometry.referenceGap = currentGeometry.signedGap;
    if (pair.geometry.representativeContact == PX_MAX_U32 ||
        currentGeometry.signedGap < pair.geometry.representativeGap) {
      pair.geometry.representativeContact = contactIndex;
      pair.geometry.representativeNormal = currentGeometry.normal;
      pair.geometry.representativeRigidOffset = currentGeometry.targetOffset;
      pair.geometry.referenceRelativePoint = queryPoint - surfacePoint;
      pair.geometry.representativeGap = currentGeometry.signedGap;
    }
    pair.trustRegion.safetyGap = physx::PxMin(
        pair.trustRegion.safetyGap, currentGeometry.signedGap);
    pair.geometry.minimumGap = physx::PxMin(
        pair.geometry.minimumGap, currentGeometry.signedGap);
    pair.geometry.active = true;
    pair.trustRegion.remainingSafeDisplacement =
        physx::PxMax(pair.trustRegion.safetyGap, 0.0f);
    pair.trustRegion.refreshRequested = true;
    pairIndices[contactIndex] = pairIndex;
  }

  // Position corrections are published before the verify DCD pass. Rebind
  // their frame-local witness to the representative of this latest manifold
  // so terminal velocity handoff never consumes a stale contact index.
  for (physx::PxU32 detectedIndex = 0;
       detectedIndex < detectedPairToRegistryScratch.size();
       ++detectedIndex) {
    const physx::PxU32 pairIndex =
        detectedPairToRegistryScratch[detectedIndex];
    if (pairIndex >= pairRegistry.size())
      return false;
    AvbdOgcPairState &pair = pairRegistry[pairIndex];
    const physx::PxU32 representative =
        pair.geometry.representativeContact;
    if (representative == PX_MAX_U32)
      continue;
    if (pair.solve.hasPendingLocalVelocity(
            AvbdOgcVelocityContactDomain::eTERMINAL))
      pair.solve.localVelocityContact = representative;
  }
  return true;
}

void consumeCurrentOgcPairRefreshRequests(
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *pairIndices, physx::PxU32 numPairIndices) {
  if (!pairStates || !pairIndices)
    return;
  for (physx::PxU32 contactIndex = 0;
       contactIndex < numPairIndices; ++contactIndex) {
    const physx::PxU32 pairIndex = pairIndices[contactIndex];
    if (pairIndex >= numPairStates)
      continue;
    // This same-time current-pose manifold was consumed. The persistent pair
    // already owns every geometry/solve result written by the projector.
    pairStates[pairIndex].trustRegion.refreshRequested = false;
  }
}

bool publishLocalOgcPairPositionResult(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    physx::PxU32 contactIndex, physx::PxReal correction,
    AvbdOgcVelocityContactDomain contactDomain,
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *pairIndices, physx::PxU32 numPairIndices) {
  if (!contacts || contactIndex >= numContacts || !pairStates ||
      !pairIndices || numPairIndices != numContacts ||
      !physx::PxIsFinite(correction) || correction <= 0.0f)
    return false;
  const physx::PxU32 pairIndex = pairIndices[contactIndex];
  if (pairIndex >= numPairStates)
    return false;
  const AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
  AvbdOgcPairState &pair = pairStates[pairIndex];
  if (!pair.geometry.active ||
      !pair.matches(geometry.source.type, geometry.targetKind,
                    geometry.queryBodyIndex, geometry.targetIndex,
                    geometry.source.primitiveKey))
    return false;
  pair.solve.publishLocalPositionResult(contactIndex, correction,
                                        contactDomain);
  return true;
}

} // namespace Dy
} // namespace physx
