// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/ogc/DyAvbdOgcGeometryQueries.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

namespace {

bool buildOgcQueryMapping(
    const AvbdSoftContactGeometry &geometry,
    AvbdWeightedContactPoint &mapping) {
  mapping.clear();
  if (geometry.hasWeightedQueryPoint()) {
    mapping = geometry.queryPoint;
    return mapping.count > 0u;
  }
  if (geometry.hasBarycentricQueryPoint()) {
    for (physx::PxU32 index = 0u; index < 3u; ++index) {
      if (geometry.queryParticleIndices[index] == PX_MAX_U32)
        break;
      if (!mapping.appendMerged(geometry.queryParticleIndices[index],
                                geometry.queryWeights[index])) {
        mapping.clear();
        return false;
      }
    }
    return mapping.count > 0u;
  }
  return geometry.particleIdx != PX_MAX_U32 &&
      mapping.appendMerged(geometry.particleIdx, 1.0f);
}

bool evaluateOgcPoseWritePoint(
    const AvbdWeightedContactPoint &mapping,
    const physx::PxVec3 *positionsBefore,
    physx::PxU32 numPositionsBefore,
    const AvbdSoftParticle *endpointParticles,
    physx::PxU32 numEndpointParticles,
    physx::PxReal alpha,
    physx::PxVec3 &point) {
  if (!positionsBefore || !endpointParticles ||
      mapping.count == 0u ||
      mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT ||
      !physx::PxIsFinite(alpha) || alpha < 0.0f || alpha > 1.0f)
    return false;
  point = physx::PxVec3(0.0f);
  physx::PxReal weightSum = 0.0f;
  for (physx::PxU32 support = 0u; support < mapping.count; ++support) {
    const physx::PxU32 particleIndex = mapping.particleIndices[support];
    const physx::PxReal weight = mapping.weights[support];
    if (particleIndex >= numPositionsBefore ||
        particleIndex >= numEndpointParticles ||
        !physx::PxIsFinite(weight) ||
        !positionsBefore[particleIndex].isFinite() ||
        !endpointParticles[particleIndex].position.isFinite())
      return false;
    const physx::PxVec3 position = positionsBefore[particleIndex] +
        (endpointParticles[particleIndex].position -
         positionsBefore[particleIndex]) * alpha;
    point += position * weight;
    weightSum += weight;
  }
  return point.isFinite() && physx::PxIsFinite(weightSum) &&
      physx::PxAbs(weightSum - 1.0f) <= 1.0e-3f;
}

bool evaluateOgcPoseWriteTetDeterminant(
    const AvbdTetElement &tet,
    const physx::PxVec3 *positionsBefore,
    physx::PxU32 numPositionsBefore,
    const AvbdSoftParticle *endpointParticles,
    physx::PxU32 numEndpointParticles,
    physx::PxReal alpha,
    physx::PxReal &determinant) {
  if (!positionsBefore || !endpointParticles ||
      tet.p0 >= numPositionsBefore || tet.p1 >= numPositionsBefore ||
      tet.p2 >= numPositionsBefore || tet.p3 >= numPositionsBefore ||
      tet.p0 >= numEndpointParticles || tet.p1 >= numEndpointParticles ||
      tet.p2 >= numEndpointParticles || tet.p3 >= numEndpointParticles ||
      !physx::PxIsFinite(alpha) || alpha < 0.0f || alpha > 1.0f ||
      !physx::PxIsFinite(tet.inverseRestDeterminant))
    return false;

  const physx::PxU32 indices[4] = {tet.p0, tet.p1, tet.p2, tet.p3};
  physx::PxVec3 positions[4];
  for (physx::PxU32 vertex = 0u; vertex < 4u; ++vertex) {
    const physx::PxU32 particleIndex = indices[vertex];
    const physx::PxVec3 &before = positionsBefore[particleIndex];
    const physx::PxVec3 &endpoint =
        endpointParticles[particleIndex].position;
    if (!before.isFinite() || !endpoint.isFinite())
      return false;
    positions[vertex] = before + (endpoint - before) * alpha;
  }
  const physx::PxVec3 e1 = positions[1] - positions[0];
  const physx::PxVec3 e2 = positions[2] - positions[0];
  const physx::PxVec3 e3 = positions[3] - positions[0];
  determinant = e1.dot(e2.cross(e3)) * tet.inverseRestDeterminant;
  return physx::PxIsFinite(determinant);
}

bool isOgcPoseWriteBodyJacobianAdmissible(
    const AvbdSoftBody &body,
    const physx::PxVec3 *positionsBefore,
    physx::PxU32 numPositionsBefore,
    const AvbdSoftParticle *endpointParticles,
    physx::PxU32 numEndpointParticles,
    physx::PxReal alpha) {
  const physx::PxReal determinantFloor = 0.05f;
  const physx::PxReal monotonicTolerance = 1.0e-6f;
  for (physx::PxU32 tetIndex = 0u;
       tetIndex < body.compiled.tetElements.size(); ++tetIndex) {
    const AvbdTetElement &tet = body.compiled.tetElements[tetIndex];
    physx::PxReal determinantBefore = 0.0f;
    physx::PxReal determinantCandidate = 0.0f;
    if (!evaluateOgcPoseWriteTetDeterminant(
            tet, positionsBefore, numPositionsBefore, endpointParticles,
            numEndpointParticles, 0.0f, determinantBefore) ||
        !evaluateOgcPoseWriteTetDeterminant(
            tet, positionsBefore, numPositionsBefore, endpointParticles,
            numEndpointParticles, alpha, determinantCandidate))
      return false;
    if (determinantBefore >= determinantFloor) {
      if (determinantCandidate < determinantFloor)
        return false;
    } else if (determinantCandidate + monotonicTolerance <
               determinantBefore) {
      // A phase may operate on a body that was already below the nominal
      // floor.  It may leave an unrelated bad tet unchanged or improve it,
      // but it may never make that state worse.
      return false;
    }
  }
  return true;
}

bool hasOgcCoupledSoftPoseWriter(const AvbdSoftBody &body) {
  for (physx::PxU32 objectiveIndex = 0u;
       objectiveIndex < body.runtime.compiledObjectives.size();
       ++objectiveIndex) {
    if (avbdIsAttachmentPositionOwner(
            body.runtime.compiledObjectives[objectiveIndex].owner))
      return true;
  }
  return false;
}

AvbdSolverBody interpolateOgcPoseWriteBody(
    const AvbdSolverBody &endpointBody,
    const physx::PxVec3 &positionBefore,
    const physx::PxQuat &rotationBefore,
    physx::PxReal alpha) {
  AvbdSolverBody body = endpointBody;
  body.position = positionBefore +
      (endpointBody.position - positionBefore) * alpha;
  body.rotation = physx::PxSlerp(
      alpha, rotationBefore, endpointBody.rotation);
  body.projectLockedPose(positionBefore, rotationBefore);
  return body;
}

bool queryOgcPoseWriteTriangleCoreGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdOgcRigidBoxGeometry &rigidBox,
    const AvbdOgcTriangleCoreCertificate &certificate,
    const AvbdSolverBody *endpointBody,
    const physx::PxVec3 *rigidPositionsBefore,
    const physx::PxQuat *rigidRotationsBefore,
    physx::PxU32 numRigidBodies,
    const physx::PxVec3 *softPositionsBefore,
    physx::PxU32 numSoftPositionsBefore,
    const AvbdSoftParticle *endpointParticles,
    physx::PxU32 numEndpointParticles,
    physx::PxReal alpha,
    physx::PxReal &faceGap) {
  if (!rigidBox.valid || !certificate.isValid())
    return false;

  physx::PxTransform boxToWorld = rigidBox.shapeToTarget;
  if (geometry.hasRigidBodyTarget()) {
    if (!endpointBody || !rigidPositionsBefore || !rigidRotationsBefore ||
        geometry.targetIndex >= numRigidBodies)
      return false;
    const AvbdSolverBody body = interpolateOgcPoseWriteBody(
        *endpointBody, rigidPositionsBefore[geometry.targetIndex],
        rigidRotationsBefore[geometry.targetIndex], alpha);
    if (!body.position.isFinite() || !body.rotation.isFinite())
      return false;
    boxToWorld = physx::PxTransform(body.position, body.rotation) *
        rigidBox.shapeToTarget;
  } else if (!geometry.hasWorldStaticTarget()) {
    return false;
  }
  if (!boxToWorld.isValid())
    return false;

  physx::PxVec3 points[3];
  physx::PxVec3 minimumLocal(PX_MAX_F32);
  physx::PxVec3 maximumLocal(-PX_MAX_F32);
  for (physx::PxU32 vertex = 0u; vertex < 3u; ++vertex) {
    AvbdWeightedContactPoint mapping;
    if (!resolveOgcTriangleCorePoint(
            geometry, &certificate, vertex, mapping) ||
        !evaluateOgcPoseWritePoint(
            mapping, softPositionsBefore, numSoftPositionsBefore,
            endpointParticles, numEndpointParticles, alpha, points[vertex]))
      return false;
    const physx::PxVec3 local = boxToWorld.transformInv(points[vertex]);
    if (!local.isFinite())
      return false;
    minimumLocal = minimumLocal.minimum(local);
    maximumLocal = maximumLocal.maximum(local);
  }

  physx::PxU32 exitFace = PX_MAX_U32;
  physx::PxReal exitDistance = 0.0f;
  if (!getRigidBoxTriangleCoreMinimumExitFace(
          rigidBox.halfExtent, minimumLocal, maximumLocal, geometry.margin,
          exitFace, exitDistance))
    return false;
  const physx::PxVec3 normalLocal =
      getRigidBoxTriangleCoreExitNormalLocal(exitFace);
  const physx::PxVec3 rawNormal = boxToWorld.q.rotate(normalLocal);
  const physx::PxReal normalLengthSq = rawNormal.magnitudeSquared();
  if (!rawNormal.isFinite() || !physx::PxIsFinite(normalLengthSq) ||
      normalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normal =
      rawNormal * physx::PxRecipSqrt(normalLengthSq);
  const physx::PxU32 axis = exitFace / 2u;
  const physx::PxVec3 surface = boxToWorld.transform(
      normalLocal * rigidBox.halfExtent[axis]);
  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0u; vertex < 3u; ++vertex)
    faceGap = physx::PxMin(
        faceGap, (points[vertex] - surface).dot(normal));
  return physx::PxIsFinite(faceGap);
}

} // namespace

bool initializeOgcPairTrustRegionContextView(
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    physx::PxU32 numSoftContacts,
    AvbdOgcPairTrustRegionContext &context) {
  context = AvbdOgcPairTrustRegionContext();
  if (!softExecutionPlan ||
      !softExecutionPlan->hasMixedOgcPairPlan(numSoftContacts))
    return false;
  context.pairStates = softExecutionPlan->ogcPairStates;
  context.numPairStates = softExecutionPlan->numOgcPairStates;
  context.contactPairIndices = softExecutionPlan->ogcPairIndices;
  context.numContactPairIndices = softExecutionPlan->numOgcPairIndices;
  context.triangleCoreSafetyStarts =
      softExecutionPlan->triangleCoreSafetyStarts;
  context.numTriangleCoreSafetyStarts =
      softExecutionPlan->numTriangleCoreSafetyStarts;
  context.triangleCoreSafetyRefs =
      softExecutionPlan->triangleCoreSafetyRefs;
  context.numTriangleCoreSafetyRefs =
      softExecutionPlan->numTriangleCoreSafetyRefs;
  return context.isComplete(numSoftContacts);
}

void AvbdOgcPoseWritePhaseState::capture(
    const AvbdOgcPairTrustRegionContext *context,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies) {
  active = context && context->isComplete(numSoftContacts) && softContacts &&
      softParticles && bodies && numSoftParticles > 0u && numBodies > 0u;
  if (!active)
    return;
  softPositionBefore.resize(numSoftParticles);
  for (physx::PxU32 particleIndex = 0u;
       particleIndex < numSoftParticles; ++particleIndex)
    softPositionBefore[particleIndex] =
        softParticles[particleIndex].position;
  rigidPositionBefore.resize(numBodies);
  rigidRotationBefore.resize(numBodies);
  rigidInvInertiaBefore.resize(numBodies);
  for (physx::PxU32 bodyIndex = 0u; bodyIndex < numBodies; ++bodyIndex) {
    rigidPositionBefore[bodyIndex] = bodies[bodyIndex].position;
    rigidRotationBefore[bodyIndex] = bodies[bodyIndex].rotation;
    rigidInvInertiaBefore[bodyIndex] = bodies[bodyIndex].invInertiaWorld;
  }
}

bool admitOgcPoseWritePhase(
    AvbdOgcPoseWritePhaseState &state,
    AvbdOgcPairTrustRegionContext *context,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan) {
  if (!state.active || !context || !context->isComplete(numSoftContacts) ||
      !bodies || !softParticles || !softBodies || !softContacts ||
      state.softPositionBefore.size() != numSoftParticles ||
      state.rigidPositionBefore.size() != numBodies ||
      state.rigidRotationBefore.size() != numBodies ||
      state.rigidInvInertiaBefore.size() != numBodies) {
    state.active = false;
    return false;
  }
  state.active = false;

  const physx::PxU32 nodeCount = numSoftBodies + numBodies;
  AvbdOgcAdmissionWorkspace &scratch =
      softExecutionPlan && softExecutionPlan->ogcAdmissionWorkspace
          ? *softExecutionPlan->ogcAdmissionWorkspace
          : state.scratch;
  scratch.componentParents.resize(nodeCount);
  scratch.participatingComponents.resize(nodeCount);
  scratch.contactAlphas.resize(numSoftContacts);
  scratch.componentAlphas.resize(nodeCount);
  physx::PxArray<physx::PxU32> &parent = scratch.componentParents;
  physx::PxArray<physx::PxU8> &participating =
      scratch.participatingComponents;
  physx::PxArray<physx::PxReal> &contactAlphas = scratch.contactAlphas;
  physx::PxArray<physx::PxReal> &componentAlpha = scratch.componentAlphas;
  for (physx::PxU32 node = 0u; node < nodeCount; ++node) {
    participating[node] = 0u;
    componentAlpha[node] = 1.0f;
  }
  for (physx::PxU32 contactIndex = 0u;
       contactIndex < numSoftContacts; ++contactIndex)
    contactAlphas[contactIndex] = 1.0f;
  for (physx::PxU32 node = 0u; node < nodeCount; ++node)
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

  const physx::PxReal clearance = 1.0e-6f;
  bool hasPair = false;
  bool limited = false;
  for (physx::PxU32 contactIndex = 0u;
       contactIndex < numSoftContacts; ++contactIndex) {
    const physx::PxU32 pairIndex =
        context->contactPairIndices[contactIndex];
    if (pairIndex >= context->numPairStates)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    const bool dynamicRigid = geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numBodies &&
        bodies[geometry.targetIndex].invMass > 0.0f;
    const bool worldStatic = geometry.hasWorldStaticTarget();
    if (!pair.geometry.active || (!dynamicRigid && !worldStatic) ||
        geometry.queryBodyIndex >= numSoftBodies ||
        pair.key.sourceType != geometry.source.type ||
        pair.key.targetKind != geometry.targetKind ||
        pair.key.sourceBodyIndex != geometry.queryBodyIndex ||
        pair.key.targetBodyIndex != geometry.targetIndex ||
        pair.key.primitiveKey != geometry.source.primitiveKey ||
        softBodies[geometry.queryBodyIndex].compiled.speculativeCCDEnabled)
      continue;

    const physx::PxU32 softNode = geometry.queryBodyIndex;
    participating[softNode] = 1u;
    if (dynamicRigid) {
      const physx::PxU32 rigidNode =
          numSoftBodies + geometry.targetIndex;
      participating[rigidNode] = 1u;
      unite(softNode, rigidNode);
    }
    hasPair = true;

    const AvbdOgcRigidBoxGeometry *rigidBox =
        getOgcRigidBoxGeometry(
            softExecutionPlan, contactIndex, numSoftContacts);
    const AvbdOgcTriangleCoreCertificate *certificate =
        getOgcTriangleCoreCertificate(
            softExecutionPlan, contactIndex, numSoftContacts);
    AvbdWeightedContactPoint queryMapping;
    const bool hasQueryMapping =
        buildOgcQueryMapping(geometry, queryMapping);
    physx::PxVec3 initialQuery(0.0f), endpointQuery(0.0f);
    const bool hasQueryPoints = hasQueryMapping &&
        evaluateOgcPoseWritePoint(
            queryMapping, state.softPositionBefore.begin(),
            state.softPositionBefore.size(), softParticles,
            numSoftParticles, 0.0f, initialQuery) &&
        evaluateOgcPoseWritePoint(
            queryMapping, state.softPositionBefore.begin(),
            state.softPositionBefore.size(), softParticles,
            numSoftParticles, 1.0f, endpointQuery);
    AvbdSolverBody initialBody;
    const AvbdSolverBody *endpointBody = nullptr;
    if (dynamicRigid) {
      endpointBody = &bodies[geometry.targetIndex];
      initialBody = *endpointBody;
      initialBody.position = state.rigidPositionBefore[geometry.targetIndex];
      initialBody.rotation = state.rigidRotationBefore[geometry.targetIndex];
      initialBody.invInertiaWorld =
          state.rigidInvInertiaBefore[geometry.targetIndex];
    }

    auto queryPointGap = [&](physx::PxReal alpha,
                             physx::PxReal &gap) {
      if (!hasQueryPoints)
        return false;
      const physx::PxVec3 query = initialQuery +
          (endpointQuery - initialQuery) * alpha;
      if (dynamicRigid) {
        const AvbdSolverBody body = interpolateOgcPoseWriteBody(
            *endpointBody,
            state.rigidPositionBefore[geometry.targetIndex],
            state.rigidRotationBefore[geometry.targetIndex], alpha);
        return queryCurrentPairSignedDistance(
            geometry, body, query, gap, rigidBox);
      }
      return queryCurrentWorldStaticPairSignedDistance(
          geometry, query, gap, rigidBox);
    };
    auto queryCoreGap = [&](physx::PxReal alpha,
                            physx::PxReal &gap) {
      return rigidBox && certificate &&
          queryOgcPoseWriteTriangleCoreGap(
              geometry, *rigidBox, *certificate, endpointBody,
              state.rigidPositionBefore.begin(),
              state.rigidRotationBefore.begin(), numBodies,
              state.softPositionBefore.begin(),
              state.softPositionBefore.size(), softParticles,
              numSoftParticles, alpha, gap);
    };

    physx::PxReal initialPointGap = 0.0f;
    physx::PxReal endpointPointGap = 0.0f;
    const bool hasPointGaps =
        queryPointGap(0.0f, initialPointGap) &&
        queryPointGap(1.0f, endpointPointGap) &&
        physx::PxIsFinite(initialPointGap) &&
        physx::PxIsFinite(endpointPointGap);
    const bool requiresPointAdmission = hasPointGaps &&
        ((initialPointGap > clearance && endpointPointGap < clearance) ||
         (initialPointGap <= clearance &&
          endpointPointGap < initialPointGap - clearance));
    physx::PxReal initialCoreGap = 0.0f;
    physx::PxReal endpointCoreGap = 0.0f;
    const bool hasCoreGaps = queryCoreGap(0.0f, initialCoreGap) &&
        queryCoreGap(1.0f, endpointCoreGap) &&
        physx::PxIsFinite(initialCoreGap) &&
        physx::PxIsFinite(endpointCoreGap);
    const bool requiresCoreAdmission = hasCoreGaps &&
        ((initialCoreGap > clearance && endpointCoreGap < clearance) ||
         (initialCoreGap <= clearance &&
          endpointCoreGap < initialCoreGap - clearance));
    if (!requiresPointAdmission && !requiresCoreAdmission)
      continue;

    auto hasClearance = [&](physx::PxReal alpha) {
      physx::PxReal gap = 0.0f;
      if (requiresPointAdmission &&
          (!queryPointGap(alpha, gap) || !physx::PxIsFinite(gap) ||
           gap < clearance))
        return false;
      if (requiresCoreAdmission &&
          (!queryCoreGap(alpha, gap) || !physx::PxIsFinite(gap) ||
           gap < clearance))
        return false;
      return true;
    };
    physx::PxReal lower = 0.0f;
    physx::PxReal upper = 1.0f;
    for (physx::PxU32 iteration = 0u; iteration < 10u; ++iteration) {
      const physx::PxReal middle = 0.5f * (lower + upper);
      if (hasClearance(middle))
        lower = middle;
      else
        upper = middle;
    }
    contactAlphas[contactIndex] = lower;
    if (lower < 1.0f - 1.0e-6f) {
      limited = true;
      pair.trustRegion.refreshRequested = true;
      pair.trustRegion.remainingSafeDisplacement = 0.0f;
      pair.geometry.minimumGap = physx::PxMin(
          pair.geometry.minimumGap, clearance);
      pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
          pair.trustRegion.accumulatedRelativeDisplacement,
          physx::PxMax(
              (endpointQuery - initialQuery).magnitude(),
              dynamicRigid
                  ? (bodies[geometry.targetIndex].position -
                     state.rigidPositionBefore[geometry.targetIndex]).
                        magnitude()
                  : 0.0f));
    }
  }
  if (!hasPair)
    return false;

  for (physx::PxU32 contactIndex = 0u;
       contactIndex < numSoftContacts; ++contactIndex) {
    const physx::PxReal alpha = contactAlphas[contactIndex];
    if (!physx::PxIsFinite(alpha) || alpha >= 1.0f)
      continue;
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    if (geometry.queryBodyIndex >= numSoftBodies)
      continue;
    const physx::PxU32 root = findRoot(geometry.queryBodyIndex);
    componentAlpha[root] = physx::PxMin(
        componentAlpha[root], physx::PxMax(alpha, 0.0f));
  }

  // Contact geometry and tet orientation are one transaction.  Coupled
  // attachment/joint blocks can move several vertices at once, so the
  // ordinary per-particle positive-J limiter is not sufficient here.  Reduce
  // the same connected-component alpha until every participating body keeps
  // its healthy tets above the determinant floor and never worsens an already
  // subthreshold tet.  Applying the same alpha to the paired rigid endpoint
  // preserves the block's relative response instead of ejecting the rigid.
  for (physx::PxU32 sourceBodyIndex = 0u;
       sourceBodyIndex < numSoftBodies; ++sourceBodyIndex) {
    if (!participating[sourceBodyIndex] ||
        !hasOgcCoupledSoftPoseWriter(softBodies[sourceBodyIndex]))
      continue;
    const physx::PxU32 root = findRoot(sourceBodyIndex);
    physx::PxReal alpha = componentAlpha[root];
    if (!physx::PxIsFinite(alpha))
      alpha = 0.0f;
    bool admissible = isOgcPoseWriteBodyJacobianAdmissible(
        softBodies[sourceBodyIndex], state.softPositionBefore.begin(),
        state.softPositionBefore.size(), softParticles, numSoftParticles,
        alpha);
    for (physx::PxU32 attempt = 0u;
         attempt < 12u && !admissible; ++attempt) {
      alpha *= 0.5f;
      admissible = isOgcPoseWriteBodyJacobianAdmissible(
          softBodies[sourceBodyIndex], state.softPositionBefore.begin(),
          state.softPositionBefore.size(), softParticles,
          numSoftParticles, alpha);
    }
    if (!admissible)
      alpha = 0.0f;
    if (alpha < componentAlpha[root] - 1.0e-7f)
      limited = true;
    componentAlpha[root] = physx::PxMin(componentAlpha[root], alpha);
  }

  for (physx::PxU32 sourceBodyIndex = 0u;
       sourceBodyIndex < numSoftBodies; ++sourceBodyIndex) {
    if (!participating[sourceBodyIndex])
      continue;
    const physx::PxReal alpha = componentAlpha[findRoot(sourceBodyIndex)];
    if (!physx::PxIsFinite(alpha) || alpha >= 1.0f)
      continue;
    const AvbdSoftBody &sourceBody = softBodies[sourceBodyIndex];
    const physx::PxU32 particleStart = sourceBody.compiled.particleStart;
    const physx::PxU32 particleCount = sourceBody.compiled.particleCount;
    if (particleStart > numSoftParticles ||
        particleCount > numSoftParticles - particleStart)
      continue;
    for (physx::PxU32 localIndex = 0u;
         localIndex < particleCount; ++localIndex) {
      const physx::PxU32 particleIndex = particleStart + localIndex;
      softParticles[particleIndex].position =
          state.softPositionBefore[particleIndex] +
          (softParticles[particleIndex].position -
           state.softPositionBefore[particleIndex]) * alpha;
    }
  }

  for (physx::PxU32 bodyIndex = 0u; bodyIndex < numBodies; ++bodyIndex) {
    const physx::PxU32 node = numSoftBodies + bodyIndex;
    if (!participating[node])
      continue;
    const physx::PxReal alpha = componentAlpha[findRoot(node)];
    if (!physx::PxIsFinite(alpha) || alpha >= 1.0f)
      continue;
    const AvbdSolverBody endpointBody = bodies[bodyIndex];
    AvbdSolverBody initialBody = endpointBody;
    initialBody.position = state.rigidPositionBefore[bodyIndex];
    initialBody.rotation = state.rigidRotationBefore[bodyIndex];
    initialBody.invInertiaWorld = state.rigidInvInertiaBefore[bodyIndex];
    AvbdSolverBody acceptedBody = endpointBody;
    acceptedBody.position = state.rigidPositionBefore[bodyIndex] +
        (endpointBody.position - state.rigidPositionBefore[bodyIndex]) *
            alpha;
    acceptedBody.rotation = physx::PxSlerp(
        alpha, state.rigidRotationBefore[bodyIndex], endpointBody.rotation);
    if (finalizeOgcRigidPositionCandidate(initialBody, acceptedBody))
      commitOgcRigidPositionCandidate(acceptedBody, bodies[bodyIndex]);
    else
      commitOgcRigidPositionCandidate(initialBody, bodies[bodyIndex]);
  }
  return limited;
}

void applyWorldStaticOgcInitialAdmission(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan) {
  if (!softParticles || !softBodies || !softContacts || numSoftParticles == 0)
    return;
  const physx::PxReal clearance = 1.0e-5f;
  AvbdOgcAdmissionWorkspace localScratch;
  AvbdOgcAdmissionWorkspace &scratch =
      softExecutionPlan && softExecutionPlan->ogcAdmissionWorkspace
          ? *softExecutionPlan->ogcAdmissionWorkspace
          : localScratch;
  scratch.particleAlphas.resize(numSoftParticles);
  physx::PxArray<physx::PxReal> &admissionAlphas = scratch.particleAlphas;
  for (physx::PxU32 particleIndex = 0u;
       particleIndex < numSoftParticles; ++particleIndex)
    admissionAlphas[particleIndex] = 1.0f;
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
    const AvbdOgcRigidBoxGeometry *rigidBox =
        getOgcRigidBoxGeometry(softExecutionPlan, sci, numSoftContacts);
    const AvbdOgcTriangleCoreCertificate *certificate =
        getOgcTriangleCoreCertificate(
            softExecutionPlan, sci, numSoftContacts);

    auto admitPoint = [&](const AvbdWeightedContactPoint &mapping) {
      physx::PxVec3 initialPoint(0.0f), endpointPoint(0.0f);
      physx::PxReal initialGap = 0.0f, endpointGap = 0.0f;
      if (!evaluatePoint(mapping, 0.0f, initialPoint) ||
          !evaluatePoint(mapping, 1.0f, endpointPoint) ||
          !queryCurrentWorldStaticPairSignedDistance(
              geometry, initialPoint, initialGap, rigidBox) ||
          !queryCurrentWorldStaticPairSignedDistance(
              geometry, endpointPoint, endpointGap, rigidBox) ||
          initialGap <= clearance || endpointGap >= clearance)
        return;
      physx::PxReal lower = 0.0f;
      physx::PxReal upper = 1.0f;
      for (physx::PxU32 iteration = 0; iteration < 10u; ++iteration) {
        const physx::PxReal middle = 0.5f * (lower + upper);
        physx::PxVec3 point(0.0f);
        physx::PxReal gap = 0.0f;
        if (evaluatePoint(mapping, middle, point) &&
            queryCurrentWorldStaticPairSignedDistance(
                geometry, point, gap, rigidBox) &&
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
    if (rigidBox && certificate) {
      AvbdWeightedContactPoint coreMappings[3];
      bool hasCoreMappings = true;
      for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex)
        hasCoreMappings &= resolveOgcTriangleCorePoint(
            geometry, certificate, vertex, coreMappings[vertex]);
      physx::PxVec3 minimumLocal(0.0f), maximumLocal(0.0f);
      physx::PxU32 exitFace = PX_MAX_U32;
      physx::PxReal exitDistance = 0.0f;
      const bool hasExitFace =
          rigidBox->shapeToTarget.isValid() &&
          getCurrentRigidBoxTriangleCoreLocalBounds(
              geometry, rigidBox->shapeToTarget, softParticles,
              numSoftParticles, minimumLocal, maximumLocal,
              PX_MAX_U32, physx::PxVec3(0.0f), certificate) &&
          getRigidBoxTriangleCoreMinimumExitFace(
              rigidBox->halfExtent, minimumLocal, maximumLocal,
              geometry.margin, exitFace, exitDistance);
      auto coreGap = [&](physx::PxReal alpha, physx::PxReal &gap) -> bool {
        if (!hasExitFace)
          return false;
        const physx::PxVec3 normalLocal =
            getRigidBoxTriangleCoreExitNormalLocal(exitFace);
        const physx::PxVec3 normal =
            rigidBox->shapeToTarget.q.rotate(normalLocal);
        const physx::PxU32 axis = exitFace / 2u;
        const physx::PxVec3 surface = rigidBox->shapeToTarget.transform(
            normalLocal * rigidBox->halfExtent[axis]);
        gap = PX_MAX_F32;
        for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
          physx::PxVec3 point(0.0f);
          if (!hasCoreMappings || !evaluatePoint(coreMappings[vertex],
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
        if (hasCoreMappings)
          for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex)
            limitPoint(coreMappings[vertex], lower);
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

physx::PxReal limitRigidOgcCandidate(
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
        !context->pairStates[pairIndex].geometry.active)
      continue;
    AvbdOgcPairState &pair = context->pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    if (!geometry.hasRigidBodyTarget() || geometry.targetIndex != bodyIndex ||
        pair.key.sourceType != geometry.source.type ||
        pair.key.targetKind != geometry.targetKind ||
        pair.key.targetBodyIndex != geometry.targetIndex ||
        pair.key.primitiveKey != geometry.source.primitiveKey)
      continue;
    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    physx::PxReal currentGap = 0.0f;
    if (!queryCurrentPairSignedDistance(geometry, body, queryPoint,
                                        currentGap,
                                        pair.geometry.rigidBox.valid
                                            ? &pair.geometry.rigidBox
                                            : nullptr))
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
      return queryCurrentPairSignedDistance(
                 geometry, candidate, queryPoint, candidateGap,
                 pair.geometry.rigidBox.valid
                     ? &pair.geometry.rigidBox : nullptr) &&
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
        pair.trustRegion.refreshRequested = true;
        pair.trustRegion.remainingSafeDisplacement = 0.0f;
        pair.geometry.minimumGap =
            physx::PxMin(pair.geometry.minimumGap, currentGap);
        pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
            pair.trustRegion.accumulatedRelativeDisplacement,
            deltaPosition.magnitude());
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
      pair.trustRegion.refreshRequested = true;
      pair.trustRegion.remainingSafeDisplacement = 0.0f;
      pair.geometry.minimumGap =
          physx::PxMin(pair.geometry.minimumGap, tolerance);
      pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
          pair.trustRegion.accumulatedRelativeDisplacement,
          deltaPosition.magnitude());
    }
    alpha = lo;
  }
  return physx::PxIsFinite(alpha) ? alpha : 0.0f;
}

} // namespace Dy
} // namespace physx

namespace physx {
namespace Dy {

bool initializeMixedOgcPairEpoch(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    bool useProvidedSoftExecutionPlan,
    const physx::PxArray<physx::PxU8> &admissionContacts,
    const physx::PxArray<physx::PxReal> &admissionDisplacements,
    AvbdOgcPairTrustRegionContext &context,
    AvbdOgcPairState *&pairStates, physx::PxU32 &numPairStates) {
  pairStates = nullptr;
  numPairStates = 0u;
  context = AvbdOgcPairTrustRegionContext();
  if (!useProvidedSoftExecutionPlan || !softExecutionPlan || !bodies ||
      !softParticles || !softContacts ||
      !softExecutionPlan->hasMixedOgcPairPlan(numSoftContacts))
    return false;

  pairStates = softExecutionPlan->ogcPairStates;
  numPairStates = softExecutionPlan->numOgcPairStates;
  if (!pairStates || numPairStates == 0u)
    return false;

  for (physx::PxU32 pairIndex = 0; pairIndex < numPairStates;
       ++pairIndex) {
    AvbdOgcPairState &pair = pairStates[pairIndex];
    // Pair identity and compiled contact count are selection-owned. All
    // current-pose geometry, trust-region and solve values begin a new epoch.
    pair.beginSolveEpoch();
  }

  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const physx::PxU32 pairIndex = softExecutionPlan->ogcPairIndices[sci];
    if (pairIndex >= numPairStates)
      continue;
    AvbdOgcPairState &pair = pairStates[pairIndex];
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    const bool dynamicRigid =
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies &&
        bodies[geometry.targetIndex].invMass > 0.0f;
    const bool worldStatic =
        (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
         geometry.source.type == AvbdSoftContactSource::eGROUND) &&
         geometry.hasWorldStaticTarget();
    const bool deformable =
        geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
        geometry.hasDeformableSurfaceTarget();
    if ((!dynamicRigid && !worldStatic && !deformable) ||
        pair.key.sourceType != geometry.source.type ||
        pair.key.targetKind != geometry.targetKind ||
        pair.key.sourceBodyIndex != geometry.queryBodyIndex ||
        pair.key.targetBodyIndex != geometry.targetIndex ||
        pair.key.primitiveKey != geometry.source.primitiveKey ||
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;
    const AvbdOgcRigidBoxGeometry *rigidBox =
        pair.geometry.rigidBox.valid ? &pair.geometry.rigidBox : nullptr;
    const AvbdOgcTriangleCoreCertificate *certificate =
        getOgcTriangleCoreCertificate(
            softExecutionPlan, sci, numSoftContacts);

    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, softParticles);
    AvbdOgcCurrentPairGeometry currentGeometry;
    const AvbdSolverBody *dynamicBody =
        dynamicRigid ? &bodies[geometry.targetIndex] : nullptr;
    bool geometryValid = false;
    physx::PxVec3 deformableTargetPoint(0.0f);
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
              geometry, dynamicBody, queryPoint, currentGeometry, rigidBox);
    }
    if (!geometryValid)
      continue;
    const physx::PxVec3 normal = currentGeometry.normal;
    const physx::PxReal gap = currentGeometry.signedGap;
    const physx::PxVec3 rigidOffset = currentGeometry.targetOffset;
    const physx::PxVec3 surfacePoint = dynamicRigid
        ? dynamicBody->position + rigidOffset
        : (deformable ? deformableTargetPoint
                      : queryPoint - normal * gap);
    if (!pair.geometry.active || gap < pair.geometry.referenceGap)
      pair.geometry.referenceGap = gap;
    if (pair.geometry.representativeContact == PX_MAX_U32 ||
        gap < pair.geometry.representativeGap) {
      pair.geometry.representativeContact = sci;
      pair.geometry.representativeNormal = normal;
      pair.geometry.representativeRigidOffset = rigidOffset;
      pair.geometry.referenceRelativePoint = queryPoint - surfacePoint;
      pair.geometry.representativeGap = gap;
    }
    if (pair.geometry.representativeContact == sci ||
        gap < pair.trustRegion.safetyGap)
      pair.trustRegion.safetyGap = gap;
    if (admissionContacts.size() == numSoftContacts &&
        admissionContacts[sci] != 0u) {
      pair.geometry.admissionContact = sci;
      pair.solve.admittedAtBoundary = true;
      if (admissionDisplacements.size() == numSoftContacts)
        pair.solve.admittedNormalDisplacement = physx::PxMax(
            pair.solve.admittedNormalDisplacement,
            admissionDisplacements[sci]);
    }
    pair.geometry.minimumGap = pair.trustRegion.safetyGap;
    pair.geometry.active = true;
    pair.trustRegion.remainingSafeDisplacement =
        physx::PxMax(pair.trustRegion.safetyGap, 0.0f);
    if (pair.solve.admittedAtBoundary &&
        pair.solve.admittedNormalDisplacement > 0.0f)
      pair.solve.admittedNormalLoad = physx::PxMax(
          pair.solve.admittedNormalLoad,
          pair.solve.admittedNormalDisplacement *
              physx::PxMin(1.0e5f,
                           physx::PxMax(1.0e3f, softContacts[sci].state.k)));
    // A static triangle-core row is an actual discrete triangle/OBB
    // intersection, not mere shell proximity.
    if (worldStatic && certificate)
      pair.trustRegion.refreshRequested = true;
  }

  context.pairStates = pairStates;
  context.numPairStates = numPairStates;
  context.contactPairIndices = softExecutionPlan->ogcPairIndices;
  context.numContactPairIndices = softExecutionPlan->numOgcPairIndices;
  return context.isComplete(numSoftContacts);
}

void initializeAvbdSoftContactDepenetrationTargets(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt) {
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
      const AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
      initialSurfacePoint =
          rigidBody.prevPosition +
          rigidBody.prevRotation.rotate(geometry.rigidLocalPoint);
    }
    avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
        contact, softParticles, softBodies, numSoftBodies,
        initialSurfacePoint, dt);
  }
}

} // namespace Dy
} // namespace physx

namespace physx {
namespace Dy {

void applyOgcMixedWarmstartAdmission(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
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
  AvbdOgcAdmissionWorkspace localScratch;
  AvbdOgcAdmissionWorkspace &scratch =
      softExecutionPlan && softExecutionPlan->ogcAdmissionWorkspace
          ? *softExecutionPlan->ogcAdmissionWorkspace
          : localScratch;
  scratch.componentParents.resize(nodeCount);
  scratch.participatingComponents.resize(nodeCount);
  scratch.contactAlphas.resize(numSoftContacts);
  scratch.componentAlphas.resize(nodeCount);
  physx::PxArray<physx::PxU32> &parent = scratch.componentParents;
  physx::PxArray<physx::PxU8> &participating =
      scratch.participatingComponents;
  physx::PxArray<physx::PxReal> &contactAlphas = scratch.contactAlphas;
  physx::PxArray<physx::PxReal> &componentAlpha = scratch.componentAlphas;
  for (physx::PxU32 node = 0u; node < nodeCount; ++node) {
    participating[node] = 0u;
    componentAlpha[node] = 1.0f;
  }
  for (physx::PxU32 contactIndex = 0u;
       contactIndex < numSoftContacts; ++contactIndex)
    contactAlphas[contactIndex] = 1.0f;
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
        !avbdHasSoftContactDynamicQuerySupport(
            geometry, softParticles, numSoftParticles))
      continue;

    const AvbdOgcRigidBoxGeometry *rigidBox =
        getOgcRigidBoxGeometry(softExecutionPlan, sci, numSoftContacts);
    const AvbdOgcTriangleCoreCertificate *certificate =
        getOgcTriangleCoreCertificate(
            softExecutionPlan, sci, numSoftContacts);

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
                                       initialGap, rigidBox) &&
        queryCurrentPairSignedDistance(geometry, body, endpointQuery,
                                       endpointGap, rigidBox) &&
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
    const bool hasCoreGaps = certificate &&
        queryInterpolatedPairTriangleCoreFaceGap(
            geometry, initialBody, body, softParticles, numSoftParticles,
            0.0f, initialCoreGap, rigidBox, certificate) &&
        queryInterpolatedPairTriangleCoreFaceGap(
            geometry, initialBody, body, softParticles, numSoftParticles,
            1.0f, endpointCoreGap, rigidBox, certificate) &&
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
        if (!queryCurrentPairSignedDistance(
                geometry, candidate, query, gap, rigidBox) ||
            !physx::PxIsFinite(gap) || gap < clearance)
          return false;
      }
      if (requiresCoreAdmission) {
        physx::PxReal coreGap = 0.0f;
        if (!queryInterpolatedPairTriangleCoreFaceGap(
                geometry, initialBody, body, softParticles, numSoftParticles,
                alpha, coreGap, rigidBox, certificate) ||
            !physx::PxIsFinite(coreGap) ||
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

} // namespace Dy
} // namespace physx
