// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/DyAvbdSolver.h"
#include "DyFeatherstoneArticulation.h"
#include "avbd/ogc/DyAvbdOgcGeometryQueries.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"

namespace physx
{
namespace Dy
{

namespace
{
}

void AvbdSolver::solveSoftParticle(
    PxU32 spi,
    AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
    AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
    const AvbdSoftBody &sb,
    AvbdSoftContact *softContacts, PxU32 numSoftContacts,
    const AvbdSoftContactParticleRef *softContactRefs,
    PxU32 softContactRefBegin, PxU32 softContactRefEnd,
    PxReal dt, PxReal invDt2,
    const AvbdTetMaterialPacketKernels &tetMaterialPacketKernels,
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
    const bool useTetMaterialPackets =
        (sb.material.coRotationalVolumeModel
             ? tetMaterialPacketKernels.corotational != NULL
             : tetMaterialPacketKernels.neoHookean != NULL) &&
        sb.compiled.tetIncidencePacketProgramValid &&
        localIdx < sb.compiled.tetIncidencePacketRanges.size() &&
        elementAdjacency.tetRefs.size() >=
            eAVBD_TET_INCIDENCE_PACKET_WIDTH;
    if (useTetMaterialPackets) {
      avbdAccumulateTetMaterialPacketContributions(
          sb, localIdx, softParticles, tetMaterialPacketKernels,
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

} // namespace Dy
} // namespace physx
