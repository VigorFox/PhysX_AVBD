// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcStaticResponse.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPairState.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

physx::PxU32 applyWorldStaticTriangleCoreLocalManifold(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, physx::PxReal configLengthScale,
    AvbdSolverStats *stats,
    const AvbdSoftIslandExecutionPlan *ogcExecutionPlan,
    AvbdSolverBody *ogcRigidBodies,
    physx::PxU32 numOgcRigidBodies,
    const AvbdSoftContact *ogcContacts,
    physx::PxU32 numOgcContacts,
    const AvbdOgcGeometryEpochView *geometryEpoch) {
  if (!softParticles || !softBodies || !softContacts ||
      numSoftParticles == 0 || numSoftBodies == 0 || numSoftContacts == 0 ||
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
    physx::PxU32 targetIndex;
    physx::PxU64 primitiveKey;
    physx::PxU32 representativeContact;
  };
  physx::PxArray<TriangleCoreGroup> groups;
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
        geometry.hasWorldStaticTarget() &&
        geometry.queryBodyIndex == group.sourceBodyIndex &&
        geometry.targetIndex == group.targetIndex &&
        geometry.source.primitiveKey == group.primitiveKey &&
        rigidBox && certificate;
  };
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    const AvbdOgcRigidBoxGeometry *rigidBox = getRigidBox(sci);
    const AvbdOgcTriangleCoreCertificate *certificate =
        getTriangleCore(sci);
    if (geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
        !geometry.hasWorldStaticTarget() ||
        geometry.queryBodyIndex >= numSoftBodies ||
        !rigidBox || !certificate ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles) ||
        softBodies[geometry.queryBodyIndex].compiled.speculativeCCDEnabled)
      continue;
    bool exists = false;
    for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
      if (isGroupContact(sci, groups[groupIndex])) {
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
    return 0u;

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
      1.0e-5f, 1.0e-4f * physx::PxMax(configLengthScale, 1.0e-6f));

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
      const TriangleCoreGroup &group = groups[groupIndex];
      if (group.representativeContact >= numSoftContacts)
        continue;
      const AvbdOgcRigidBoxGeometry *carrierBox =
          getRigidBox(group.representativeContact);
      if (!carrierBox || !carrierBox->valid)
        continue;
      const physx::PxTransform boxToWorld = carrierBox->shapeToTarget;

      physx::PxReal faceExit[6] = {0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f};
      bool validGroup = true;
      bool hasCore = false;
      for (physx::PxU32 sci = 0;
           sci < numSoftContacts && validGroup; ++sci) {
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
          boxToWorld.transform(localNormal * carrierBox->halfExtent[axis]);
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
      const physx::PxU32 patchCorrections =
          applyWorldStaticSoftNormalDepenetrationSweeps(
          softParticles, numSoftParticles, softBodies, numSoftBodies,
          patchContacts.begin(), patchContacts.size(), 1u, stats,
          ogcExecutionPlan, ogcRigidBodies, numOgcRigidBodies, ogcContacts,
          numOgcContacts);
      appliedCorrections += patchCorrections;
      anyCorrection = anyCorrection || patchCorrections > 0u;
    }
    if (!anyCorrection)
      break;
  }
  return appliedCorrections;
}

physx::PxU32 applyWorldStaticSoftNormalDepenetrationSweeps(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, AvbdSolverStats *stats,
    const AvbdSoftIslandExecutionPlan *ogcExecutionPlan,
    AvbdSolverBody *ogcRigidBodies,
    physx::PxU32 numOgcRigidBodies,
    const AvbdSoftContact *ogcContacts,
    physx::PxU32 numOgcContacts,
    AvbdOgcPairState *pairStates,
    physx::PxU32 numPairStates,
    const physx::PxU32 *contactPairIndices,
    physx::PxU32 numContactPairIndices,
    AvbdOgcVelocityContactDomain contactDomain,
    const AvbdOgcGeometryEpochView *geometryEpoch) {
  (void)stats;
  if (!softParticles || numSoftParticles == 0 || !softBodies ||
      numSoftBodies == 0 || !softContacts || numSoftContacts == 0 ||
      sweeps == 0)
    return 0u;

  physx::PxU32 appliedCorrections = 0u;

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

      // Unlike avbdEvaluateSoftContactNormalConstraint(), this omits the OGC
      // margin.  Only actual collision-surface overlap is recoverable here.
      const AvbdOgcRigidBoxGeometry *rigidBox = geometryEpoch
          ? geometryEpoch->getRigidBox(sci, numSoftContacts) : nullptr;
      AvbdOgcNormalResponse response;
      if (!compileCurrentOgcNormalResponse(
              geometry, softParticles, numSoftParticles,
              /*dynamicTarget=*/nullptr,
              /*softResponseScale=*/1.0f, response, rigidBox))
        continue;
      if (!(response.current.signedGap < 0.0f) ||
          !physx::PxIsFinite(response.current.signedGap))
        continue;
      const physx::PxReal trueGap = response.current.signedGap;
      const physx::PxU32 *particleIndices = response.particleIndices;
      const physx::PxU32 particleCount = response.particleCount;

      // World-static geometry has no opposing movable endpoint. Recover the
      // complete actual overlap and let common-alpha exact-J admission be the
      // displacement limiter; a fixed small cap cannot recover a 60 Hz fall.
      const physx::PxReal requestedCorrection = -trueGap;
      const physx::PxReal lambda =
          requestedCorrection / response.effectiveResponse;
      if (!physx::PxIsFinite(lambda) || lambda <= 0.0f)
        continue;

      AvbdOgcSoftPositionCandidate softCandidate;
      if (!buildOgcSoftPositionCandidate(
              response, softParticles, numSoftParticles, *sourceBody,
              /*softResponseScale=*/1.0f, lambda, softCandidate))
        continue;

      // Every support vertex and every incident tet shares the same alpha.
      // This is the only safe way to retain the coupled contact correction
      // while applying the scalar path's exact positive-J admission rule.
      physx::PxReal commonAlpha = 1.0f;
      bool acceptedCandidate = false;
      for (physx::PxU32 attempt = 0; attempt < 8 && !acceptedCandidate;
           ++attempt) {
        bool candidateValid = admitOgcSoftPositionCandidate(
            response, softCandidate, softParticles, numSoftParticles,
            *sourceBody, commonAlpha, 0.05f);

        // A world-static correction is only one owner in the shared OGC
        // island.  Clip its complete support update against every dynamic
        // soft/rigid pair touched by those particles before committing it;
        // otherwise the pedestal can push the lower jaw through the box and
        // leave terminal recovery with an already-infeasible configuration.
        physx::PxReal ogcAlpha = 1.0f;
        if (candidateValid && ogcExecutionPlan && ogcRigidBodies &&
            ogcContacts && numOgcContacts > 0u) {
          for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
            const physx::PxVec3 delta =
                softCandidate.particleDeltas[pi] * commonAlpha;
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

        physx::PxVec3 candidateQueryPoint(0.0f);
        if (candidateValid &&
            !evaluateOgcSoftPositionCandidateQueryPoint(
                response, softCandidate, commonAlpha,
                candidateQueryPoint))
          candidateValid = false;
        AvbdOgcCurrentPairGeometry candidateGeometry;
        const physx::PxReal gapImprovementTolerance = 1.0e-6f;
        if (candidateValid &&
            !getCurrentOgcPairGeometry(
                geometry, nullptr, candidateQueryPoint,
                candidateGeometry, rigidBox))
          candidateValid = false;
        if (candidateValid &&
            (!(candidateGeometry.signedGap >
               trueGap + gapImprovementTolerance) ||
             !physx::PxIsFinite(candidateGeometry.signedGap)))
          candidateValid = false;

        if (candidateValid)
          acceptedCandidate = true;
        else if (commonAlpha > 0.0f &&
                 !(ogcAlpha < 1.0f - 1.0e-6f))
          commonAlpha *= 0.5f;
      }
      if (!acceptedCandidate)
        continue;

      commitOgcSoftPositionCandidate(
          response, softCandidate, softParticles, numSoftParticles,
          commonAlpha);
      publishLocalOgcPairPositionResult(
          softContacts, numSoftContacts, sci,
          requestedCorrection * commonAlpha, contactDomain,
          pairStates, numPairStates, contactPairIndices,
          numContactPairIndices);
      ++appliedCorrections;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }
  return appliedCorrections;
}

//=============================================================================

} // namespace Dy
} // namespace physx
