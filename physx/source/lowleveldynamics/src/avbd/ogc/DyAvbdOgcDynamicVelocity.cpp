// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/ogc/DyAvbdOgcPairState.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

physx::PxU32 applyDynamicSoftRigidNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps,
    physx::PxReal configLengthScale, AvbdSolverStats *stats,
    AvbdOgcPairState *ogcPairStates,
    physx::PxU32 numOgcPairStates,
    const physx::PxU32 *ogcPairIndices,
    physx::PxU32 numOgcPairIndices,
    physx::PxReal softComplianceResponseScale,
    bool projectToCurrentPoseBoundary,
    const physx::PxU32 *ogcPairContactStarts,
    physx::PxU32 numOgcPairContactStarts,
    const physx::PxU32 *ogcPairContactRefs,
    physx::PxU32 numOgcPairContactRefs,
    AvbdOgcVelocityContactDomain contactDomain,
    const AvbdOgcGeometryEpochView *geometryEpoch) {
  (void)stats;
  if (!bodies || numBodies == 0 || !softParticles ||
      numSoftParticles == 0 || !softBodies || numSoftBodies == 0 ||
      !softContacts || numSoftContacts == 0 || sweeps == 0)
    return 0u;

  physx::PxU32 appliedCorrections = 0u;

  // This is a recovery, not a second OGC owner.  Cap each contact projection
  // to a small physical distance and let six Gauss--Seidel sweeps resolve
  // intersecting rows in their prepared order.
  const physx::PxReal lengthScale =
      physx::PxMax(configLengthScale, 1.0e-6f);
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
            !ogcPairStates[pairIndex].geometry.active)
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
          const AvbdOgcRigidBoxGeometry *rigidBox = geometryEpoch
              ? geometryEpoch->getRigidBox(sci, numSoftContacts) : nullptr;
          AvbdOgcCurrentPairGeometry currentGeometry;
          if (!queryPoint.isFinite() ||
              !getCurrentOgcPairGeometry(
                  geometry, &bodies[geometry.targetIndex], queryPoint,
                  currentGeometry, rigidBox) ||
              !physx::PxIsFinite(currentGeometry.signedGap) ||
              currentGeometry.signedGap >= 0.0f)
            continue;
          if (deepestPairContacts[pairIndex] == PX_MAX_U32 ||
              currentGeometry.signedGap < deepestPairGaps[pairIndex]) {
            deepestPairContacts[pairIndex] = sci;
            deepestPairGaps[pairIndex] = currentGeometry.signedGap;
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

      // Do not use avbdEvaluateSoftContactNormalConstraint() here: it
      // subtracts geometry.margin and is negative throughout the OGC shell.
      // This is the actual collision-surface signed gap only.
      const AvbdOgcRigidBoxGeometry *rigidBox = geometryEpoch
          ? geometryEpoch->getRigidBox(sci, numSoftContacts) : nullptr;
      AvbdOgcNormalResponse response;
      if (!compileCurrentOgcNormalResponse(
              geometry, softParticles, numSoftParticles, &body,
              softResponseScale, response, rigidBox))
        continue;
      if (!physx::PxIsFinite(response.constraintValue))
        continue;

      if (!(response.constraintValue < 0.0f))
        continue;
      const physx::PxReal trueGap = response.constraintValue;

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
          requestedCorrection / response.effectiveResponse;
      if (lambda <= 0.0f || !physx::PxIsFinite(lambda))
        continue;

      AvbdOgcSoftPositionCandidate softCandidate;
      if (!buildOgcSoftPositionCandidate(
              response, softParticles, numSoftParticles, *sourceBody,
              softResponseScale, lambda, softCandidate))
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
        // Keep a positive determinant floor for healthy incident elements,
        // but permit this emergency projection to monotonically repair an
        // element which was already below the floor before this row.  The
        // terminal current-pose DCD owner uses a lower, still-positive floor:
        // retaining the ordinary material-quality threshold (.05) can make
        // the final contact feasible set empty and leave the rigid inside for
        // one frame. Material rows restore quality in the following epoch.
        const physx::PxReal minimumRecoveryDeterminant =
            projectToCurrentPoseBoundary ? 0.035f : 0.05f;
        bool candidateValid = admitOgcSoftPositionCandidate(
            response, softCandidate, softParticles, numSoftParticles,
            *sourceBody, commonAlpha, minimumRecoveryDeterminant);

        AvbdSolverBody candidateBody;
        if (!buildOgcRigidPositionCandidate(
                response, body, lambda, commonAlpha,
                candidateBody))
          candidateValid = false;
        if (!candidateValid) {
          commonAlpha *= 0.5f;
          continue;
        }

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
                candidateBody, geometryEpoch);
        if (!physx::PxIsFinite(rigidOgcAlpha) || rigidOgcAlpha < 0.0f) {
          candidateValid = false;
        } else if (rigidOgcAlpha < 1.0f - 1.0e-6f) {
          candidateBody.position = body.position +
              (candidateBody.position - body.position) * rigidOgcAlpha;
          candidateBody.rotation = physx::PxSlerp(
              rigidOgcAlpha, body.rotation, candidateBody.rotation);
        }

        candidateValid = candidateValid &&
            finalizeOgcRigidPositionCandidate(body, candidateBody);
        physx::PxVec3 candidateQueryPoint(0.0f);
        if (candidateValid &&
            !evaluateOgcSoftPositionCandidateQueryPoint(
                response, softCandidate, commonAlpha,
                candidateQueryPoint))
          candidateValid = false;
        physx::PxReal candidateConstraint = 0.0f;
        const physx::PxReal gapImprovementTolerance = physx::PxMax(
            1.0e-6f, 1.0e-5f * lengthScale);
        if (candidateValid &&
            !evaluateCurrentOgcNormalConstraint(
                geometry, response, &candidateBody,
                candidateQueryPoint, candidateConstraint))
          candidateValid = false;
        if (candidateValid &&
            (!(candidateConstraint >
               trueGap + gapImprovementTolerance) ||
             !physx::PxIsFinite(candidateConstraint)))
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

      commitOgcSoftPositionCandidate(
          response, softCandidate, softParticles, numSoftParticles,
          commonAlpha);
      commitOgcRigidPositionCandidate(acceptedBody, body);
      publishLocalOgcPairPositionResult(
          softContacts, numSoftContacts, sci,
          requestedCorrection * commonAlpha, contactDomain,
          ogcPairStates, numOgcPairStates, ogcPairIndices,
          numOgcPairIndices);
      ++appliedCorrections;
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }

  return appliedCorrections;
}

void clampAdmittedMixedOgcPairNormalVelocities(
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
    physx::PxReal configLengthScale, AvbdSolverStats *stats) {
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
    if (pair.geometry.active && pair.solve.admittedAtBoundary) {
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
      1.0e-5f,
      1.0e-4f * physx::PxMax(configLengthScale, 1.0e-6f));
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
      if (!pair.geometry.active || !pair.solve.admittedAtBoundary)
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
          geometry.queryBodyIndex != pair.key.sourceBodyIndex ||
          geometry.targetIndex != pair.key.targetBodyIndex ||
          geometry.source.primitiveKey != pair.key.primitiveKey ||
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

      AvbdOgcNormalResponse response;
      if (!compileCurrentOgcNormalResponse(
              geometry, softParticles, numSoftParticles, &rigidBody,
              /*softResponseScale=*/1.0f, response) ||
          !physx::PxIsFinite(response.current.signedGap))
        continue;

      // The actual signed distance remains the geometric authority.  The
      // contact margin only keeps a boundary-admitted impact owned until it
      // has visibly separated; it never pulls a distant pair back together.
      const physx::PxReal contactBand = physx::PxMax(
          minimumBand, physx::PxMax(0.0f, geometry.margin));
      if (response.current.signedGap > contactBand)
        continue;
      const physx::PxVec3 &normal = response.normal;
      const physx::PxVec3 &worldOffset = response.current.targetOffset;
      const physx::PxU32 *particleIndices = response.particleIndices;
      const physx::PxU32 particleCount = response.particleCount;

      physx::PxVec3 queryVelocity(0.0f);
      if (!getOgcNormalResponseQueryVelocity(
              response, softParticles, numSoftParticles, queryVelocity))
        continue;

      const physx::PxVec3 &rigidLinearDelta =
          response.targetLinearDeltaPerLambda;
      const physx::PxVec3 &rigidAngularDelta =
          response.targetAngularDeltaPerLambda;

      const physx::PxVec3 rigidSurfaceVelocity =
          rigidBody.linearVelocity +
          rigidBody.angularVelocity.cross(worldOffset);
      const physx::PxReal relativeNormalVelocity =
          (queryVelocity - rigidSurfaceVelocity).dot(normal);
      if (!physx::PxIsFinite(relativeNormalVelocity) ||
          relativeNormalVelocity >= -1.0e-6f)
        continue;
      const physx::PxReal impulse =
          -relativeNormalVelocity / response.effectiveResponse;
      if (!physx::PxIsFinite(impulse) || impulse <= 0.0f)
        continue;

      physx::PxVec3 particleVelocityDeltas[AVBD_CONTACT_MAX_PARTICLES];
      bool finiteCandidates = true;
      for (physx::PxU32 pi = 0; pi < particleCount; ++pi) {
        const physx::PxU32 particleIndex = particleIndices[pi];
        const AvbdSoftParticle &particle = softParticles[particleIndex];
        particleVelocityDeltas[pi] = physx::PxVec3(0.0f);
        if (particle.invMass <= 0.0f)
          continue;
        const physx::PxVec3 delta =
            normal *
            (particle.invMass * response.particleWeights[pi] * impulse);
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
      pair.solve.accumulatedNormalLambda += impulse;
      pair.geometry.minimumGap =
          physx::PxMin(pair.geometry.minimumGap,
                       response.current.signedGap);
      appliedImpulse = true;
      }
    }
    if (!appliedImpulse)
      break;
  }
}

} // namespace Dy
} // namespace physx
