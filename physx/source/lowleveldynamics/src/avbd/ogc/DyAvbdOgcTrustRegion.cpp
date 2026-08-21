// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/ogc/DyAvbdOgcGeometryQueries.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

physx::PxReal limitSoftParticleOgcCandidate(
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
        !context->pairStates[pairIndex].geometry.active)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[ref.contactIndex].geometry;
    const bool hasDynamicRigid = geometry.hasRigidBodyTarget() &&
        rigidBodies && geometry.targetIndex < numRigidBodies;
    const bool hasWorldStatic = geometry.hasWorldStaticTarget();
    if ((!hasDynamicRigid && !hasWorldStatic) ||
        pair.key.sourceType != geometry.source.type ||
        pair.key.targetKind != geometry.targetKind ||
        pair.key.sourceBodyIndex != geometry.queryBodyIndex ||
        pair.key.targetBodyIndex != geometry.targetIndex ||
        pair.key.primitiveKey != geometry.source.primitiveKey)
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
      pair.trustRegion.refreshRequested = true;
      pair.trustRegion.remainingSafeDisplacement = 0.0f;
      const physx::PxReal admittedGap =
          gap + (candidateGap - gap) * localAlpha;
      pair.geometry.minimumGap =
          physx::PxMin(pair.geometry.minimumGap, admittedGap);
      pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
          pair.trustRegion.accumulatedRelativeDisplacement,
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
        !context->pairStates[pairIndex].geometry.active)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[ref.contactIndex].geometry;
    const bool hasDynamicRigid = geometry.hasRigidBodyTarget() &&
        rigidBodies && geometry.targetIndex < numRigidBodies;
    const bool hasWorldStatic = geometry.hasWorldStaticTarget();
    if ((!hasDynamicRigid && !hasWorldStatic) ||
        pair.key.sourceType != geometry.source.type ||
        pair.key.targetKind != geometry.targetKind ||
        pair.key.sourceBodyIndex != geometry.queryBodyIndex ||
        pair.key.targetBodyIndex != geometry.targetIndex ||
        pair.key.primitiveKey != geometry.source.primitiveKey)
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
      pair.trustRegion.refreshRequested = true;
      pair.trustRegion.remainingSafeDisplacement = 0.0f;
      const physx::PxReal admittedGap =
          gap + (candidateGap - gap) * localAlpha;
      pair.geometry.minimumGap =
          physx::PxMin(pair.geometry.minimumGap, admittedGap);
      pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
          pair.trustRegion.accumulatedRelativeDisplacement,
          physx::PxMax(0.0f, gap - admittedGap));
    }
    alpha = physx::PxMin(alpha, localAlpha);
  }
  return physx::PxIsFinite(alpha) ? alpha : 0.0f;
}

} // namespace Dy
} // namespace physx
