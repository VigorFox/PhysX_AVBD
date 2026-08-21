// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

// Dynamic soft/rigid SDF overlap recovery
//=============================================================================

physx::PxU32 applyDynamicSoftRigidTriangleCoreLocalManifold(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, physx::PxReal configLengthScale,
    AvbdSolverStats *stats,
    AvbdOgcPairState *ogcPairStates,
    physx::PxU32 numOgcPairStates,
    const physx::PxU32 *ogcPairIndices,
    physx::PxU32 numOgcPairIndices,
    const physx::PxU32 *ogcPairContactStarts,
    physx::PxU32 numOgcPairContactStarts,
    const physx::PxU32 *ogcPairContactRefs,
    physx::PxU32 numOgcPairContactRefs,
    AvbdOgcVelocityContactDomain contactDomain,
    const AvbdOgcGeometryEpochView *geometryEpoch) {
  if (!bodies || !softParticles || !softBodies || !softContacts ||
      numBodies == 0 || numSoftBodies == 0 || numSoftContacts == 0 ||
      sweeps == 0)
    return 0u;

  physx::PxU32 appliedCorrections = 0u;

  auto getRigidBox = [&](physx::PxU32 contactIndex)
      -> const AvbdOgcRigidBoxGeometry * {
    return geometryEpoch
        ? geometryEpoch->getRigidBox(contactIndex, numSoftContacts)
        : nullptr;
  };
  auto getTriangleCore = [&](physx::PxU32 contactIndex) {
    return geometryEpoch
        ? geometryEpoch->getTriangleCore(contactIndex, numSoftContacts)
        : static_cast<const AvbdOgcTriangleCoreCertificate *>(nullptr);
  };

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
    const AvbdOgcRigidBoxGeometry *rigidBox = getRigidBox(sci);
    const AvbdOgcTriangleCoreCertificate *certificate =
        getTriangleCore(sci);
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasRigidBodyTarget() ||
        geometry.targetIndex >= numBodies ||
        geometry.queryBodyIndex >= numSoftBodies ||
        !rigidBox || !certificate ||
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
        if (pair.key.sourceType != geometry.source.type ||
            pair.key.targetKind != geometry.targetKind ||
            pair.key.sourceBodyIndex != geometry.queryBodyIndex ||
            pair.key.targetBodyIndex != geometry.targetIndex ||
            pair.key.primitiveKey != geometry.source.primitiveKey)
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
    return 0u;

  const physx::PxReal tolerance = physx::PxMax(
      1.0e-5f,
      1.0e-4f * physx::PxMax(configLengthScale, 1.0e-6f));

  auto isGroupContact = [&](physx::PxU32 contactIndex,
                            const TriangleCoreGroup &group) {
    if (contactIndex >= numSoftContacts)
      return false;
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    const AvbdOgcRigidBoxGeometry *rigidBox = getRigidBox(contactIndex);
    const AvbdOgcTriangleCoreCertificate *certificate =
        getTriangleCore(contactIndex);
    return geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() &&
        geometry.queryBodyIndex == group.sourceBodyIndex &&
        geometry.targetIndex == group.targetBodyIndex &&
        geometry.source.primitiveKey == group.primitiveKey &&
        rigidBox && certificate;
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
  auto getPairForGroup = [&](const TriangleCoreGroup &group)
      -> AvbdOgcPairState * {
    if (usePairContactPlan && group.pairIndex < numOgcPairStates) {
      AvbdOgcPairState &pair = ogcPairStates[group.pairIndex];
      return pair.geometry.active &&
              pair.key.sourceBodyIndex == group.sourceBodyIndex &&
              pair.key.targetBodyIndex == group.targetBodyIndex &&
              pair.key.primitiveKey == group.primitiveKey
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
    return pair.geometry.active &&
            pair.key.sourceBodyIndex == group.sourceBodyIndex &&
            pair.key.targetBodyIndex == group.targetBodyIndex &&
            pair.key.primitiveKey == group.primitiveKey
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
      const AvbdOgcRigidBoxGeometry *carrierBox =
          getRigidBox(group.representativeContact);
      if (rigidBody.invMass <= 0.0f || !rigidBody.position.isFinite() ||
          !rigidBody.rotation.isFinite() || !carrierBox ||
          !carrierBox->valid)
        continue;
      const physx::PxTransform shapeToWorld(
          rigidBody.position, rigidBody.rotation);
      const physx::PxTransform boxToWorld =
          shapeToWorld * carrierBox->shapeToTarget;
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
        if (!isGroupContact(sci, group))
          continue;
        const AvbdOgcRigidBoxGeometry *rowBox = getRigidBox(sci);
        const AvbdOgcTriangleCoreCertificate *certificate =
            getTriangleCore(sci);
        if (!rowBox) {
          validGroup = false;
          break;
        }
        const bool sameRotation =
            rowBox->shapeToTarget.q == carrierBox->shapeToTarget.q ||
            rowBox->shapeToTarget.q == -carrierBox->shapeToTarget.q;
        if (rowBox->shapeToTarget.p != carrierBox->shapeToTarget.p ||
            !sameRotation || rowBox->halfExtent != carrierBox->halfExtent) {
          validGroup = false;
          break;
        }
        hasCore = true;
        physx::PxVec3 minimum(PX_MAX_F32);
        physx::PxVec3 maximum(-PX_MAX_F32);
        validGroup = getCurrentRigidBoxTriangleCoreLocalBounds(
            geometry, boxToWorld, softParticles, numSoftParticles,
            minimum, maximum, PX_MAX_U32, physx::PxVec3(0.0f),
            certificate) &&
            accumulateRigidBoxTriangleCoreFaceExits(
                carrierBox->halfExtent, minimum, maximum, faceExit);
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
        if (AvbdOgcPairState *pair = getPairForGroup(group)) {
          pair->geometry.publishTriangleCoreManifold(
              geometryEpoch ? geometryEpoch->geometryEpoch : 0u,
              PX_MAX_U8, 0.0f);
          pair->solve.triangleCoreLocallyResolved = true;
        }
        continue;
      }
      if (face == PX_MAX_U32 || !physx::PxIsFinite(bestExit))
        continue;

      if (AvbdOgcPairState *pair = getPairForGroup(group)) {
        pair->geometry.publishTriangleCoreManifold(
            geometryEpoch ? geometryEpoch->geometryEpoch : 0u,
            physx::PxU8(face), bestExit);
        pair->solve.triangleCoreLocallyResolved = false;
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
          carrierBox->halfExtent[axis];

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
        if (!isGroupContact(sci, group))
          continue;
        const AvbdOgcTriangleCoreCertificate *certificate =
            getTriangleCore(sci);
        for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
          AvbdSoftContact localContact = softContacts[sci];
          AvbdSoftContactGeometry &localGeometry = localContact.geometry;
          if (!resolveOgcTriangleCorePoint(
                  localGeometry, certificate, vertex,
                  localGeometry.queryPoint))
            continue;
          // Make the selected common OBB face a local plane row so a different
          // nearest face cannot split this manifold into unrelated projections.
          localGeometry.normal = worldNormal;
          localGeometry.projNormal = worldNormal;
          localGeometry.rigidLocalPoint =
              carrierBox->shapeToTarget.transform(faceLocal);
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
        const physx::PxU32 patchCorrections =
            applyDynamicSoftRigidNormalDepenetrationSweeps(
            bodies, numBodies, softParticles, numSoftParticles, softBodies,
            numSoftBodies, patchContacts.begin(), patchContacts.size(), 1u,
            configLengthScale, stats,
            nullptr, 0u, nullptr, 0u,
            /*softComplianceResponseScale=*/4.0f,
            /*projectToCurrentPoseBoundary=*/true);
        projectedGroup = patchCorrections > 0u;
        appliedCorrections += patchCorrections;
        if (projectedGroup) {
          if (AvbdOgcPairState *pair = getPairForGroup(group))
            pair->solve.publishLocalPositionResult(
                group.representativeContact, bestExit, contactDomain);
        }
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
    const AvbdOgcRigidBoxGeometry *carrierBox =
        getRigidBox(group.representativeContact);
    if (!carrierBox || !carrierBox->valid)
      continue;
    const physx::PxTransform boxToWorld(
        physx::PxTransform(rigidBody.position, rigidBody.rotation) *
        carrierBox->shapeToTarget);
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
      if (!isGroupContact(sci, group))
        continue;
      const AvbdOgcTriangleCoreCertificate *certificate =
          getTriangleCore(sci);
      physx::PxVec3 minimum(PX_MAX_F32);
      physx::PxVec3 maximum(-PX_MAX_F32);
      validGroup = getCurrentRigidBoxTriangleCoreLocalBounds(
          geometry, boxToWorld, softParticles, numSoftParticles,
          minimum, maximum, PX_MAX_U32, physx::PxVec3(0.0f),
          certificate) &&
          accumulateRigidBoxTriangleCoreFaceExits(
              carrierBox->halfExtent, minimum, maximum, faceExit);
    }
    if (!validGroup)
      continue;
    for (physx::PxU32 face = 0; face < 6; ++face) {
      if (faceExit[face] <= tolerance) {
        if (AvbdOgcPairState *pair = getPairForGroup(group)) {
          pair->geometry.publishTriangleCoreManifold(
              geometryEpoch ? geometryEpoch->geometryEpoch : 0u,
              physx::PxU8(face), 0.0f);
          pair->solve.triangleCoreLocallyResolved = true;
        }
        break;
      }
    }
  }
  return appliedCorrections;
}

} // namespace Dy
} // namespace physx
