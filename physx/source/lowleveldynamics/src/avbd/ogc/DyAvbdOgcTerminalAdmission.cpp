// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcTerminal.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "avbd/contact/DyAvbdContactDetection.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

// Terminal current-pose OGC admission remains private to LowLevelDynamics.
// The terminal epoch owns only pair admission; it never advances time and
// never invokes a swept/CCD detector.

// Build a fresh terminal manifold from the immutable geometry provider. The
// proxy and every movable primitive use the final solver pose, with previous
// pose pinned to the same value. This helper is strictly discrete OGC.
bool avbdBuildTerminalCurrentPoseContacts(
    const AvbdSoftIslandExecutionPlan *plan, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, const AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies, const physx::PxU8 *sourceBodyMask,
    physx::PxU32 numSourceBodyMask,
    physx::PxArray<AvbdSoftParticle> &proxyParticles,
    physx::PxArray<AvbdSoftBody> &collisionBodies,
    physx::PxArray<AvbdRigidBox> &boxes,
    physx::PxArray<AvbdRigidSphere> &spheres,
    physx::PxArray<AvbdRigidCapsule> &capsules,
    physx::PxArray<AvbdRigidConvex> &convexes,
    physx::PxArray<AvbdRigidTriangleSurface> &triangleSurfaces,
    physx::PxArray<AvbdSoftContact> &contacts,
    AvbdSoftContactWorkspace &contactWorkspace,
    AvbdOgcGeometryEpochSidecar &geometrySidecar) {
  if (!plan || !plan->hasTerminalCurrentPoseGeometryPlan(numSoftParticles) ||
      !softParticles || !softBodies || numSoftBodies == 0 ||
      !sourceBodyMask ||
      numSourceBodyMask !=
          plan->terminalGeometryProvider.numCollisionBodies)
    return false;
  const AvbdOgcCurrentPoseGeometryProvider &provider =
      plan->terminalGeometryProvider;

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
    if (softBodies[bodyIndex].compiled.speculativeCCDEnabled)
      return false;

  proxyParticles.resize(provider.numCollisionVertexMappings);
  for (physx::PxU32 proxyIndex = 0;
       proxyIndex < provider.numCollisionVertexMappings; ++proxyIndex) {
    const AvbdWeightedContactPoint &mapping =
        provider.collisionVertexMappings[proxyIndex];
    if (mapping.count == 0)
      return false;
    AvbdSoftParticle proxy;
    proxy.position = physx::PxVec3(0.0f);
    proxy.velocity = physx::PxVec3(0.0f);
    bool dynamic = false;
    for (physx::PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint) {
      const physx::PxU32 particleIndex = mapping.particleIndices[endpoint];
      const physx::PxReal weight = mapping.weights[endpoint];
      if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight))
        return false;
      const AvbdSoftParticle &particle = softParticles[particleIndex];
      if (!particle.position.isFinite())
        return false;
      proxy.position += particle.position * weight;
      proxy.velocity += particle.velocity * weight;
      dynamic = dynamic || particle.invMass > 0.0f;
    }
    proxy.initialPosition = proxy.position;
    proxy.predictedPosition = proxy.position;
    proxy.outerPosition = proxy.position;
    proxy.prevVelocity = proxy.velocity;
    proxy.invMass = dynamic ? 1.0f : 0.0f;
    proxy.mass = dynamic ? 1.0f : 0.0f;
    proxy.gravityScale = 1.0f;
    proxyParticles[proxyIndex] = proxy;
  }

  collisionBodies.clear();
  for (physx::PxU32 bodyIndex = 0;
       bodyIndex < provider.numCollisionBodies; ++bodyIndex) {
    if (sourceBodyMask[bodyIndex] != 0u)
      collisionBodies.pushBack(provider.collisionBodies[bodyIndex]);
  }
  if (collisionBodies.empty())
    return false;

  boxes.clear();
  for (physx::PxU32 boxIndex = 0; boxIndex < provider.numRigidBoxes;
       ++boxIndex) {
    AvbdRigidBox box = provider.rigidBoxes[boxIndex];
    if (box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY) {
      if (box.targetIndex >= numBodies)
        return false;
      const physx::PxTransform bodyToWorld(
          bodies[box.targetIndex].position, bodies[box.targetIndex].rotation);
      const physx::PxTransform shapeToWorld =
          bodyToWorld * box.shapeToRigidBody;
      if (!shapeToWorld.isValid())
        return false;
      box.center = shapeToWorld.p;
      box.rotation = shapeToWorld.q;
    }
    box.previousCenter = box.center;
    box.previousRotation = box.rotation;
    boxes.pushBack(box);
  }
  const auto currentShapePose =
      [bodies, numBodies](AvbdSoftContactTargetKind targetKind,
                         physx::PxU32 targetIndex,
                         const physx::PxTransform &shapeToRigidBody,
                         const physx::PxVec3 &staticCenter,
                         const physx::PxQuat &staticRotation,
                         physx::PxTransform &shapeToWorld) {
        if (targetKind == AvbdSoftContactTargetKind::eRIGID_BODY) {
          if (targetIndex >= numBodies)
            return false;
          shapeToWorld =
              physx::PxTransform(bodies[targetIndex].position,
                                 bodies[targetIndex].rotation) *
              shapeToRigidBody;
        } else {
          shapeToWorld = physx::PxTransform(staticCenter, staticRotation);
        }
        return shapeToWorld.isValid();
      };

  spheres.clear();
  for (physx::PxU32 index = 0; index < provider.numRigidSpheres; ++index) {
    AvbdRigidSphere sphere = provider.rigidSpheres[index];
    physx::PxTransform pose;
    if (!currentShapePose(sphere.targetKind, sphere.targetIndex,
                          sphere.shapeToRigidBody, sphere.center,
                          sphere.rotation, pose))
      return false;
    sphere.center = pose.p;
    sphere.rotation = pose.q;
    sphere.previousCenter = pose.p;
    sphere.previousRotation = pose.q;
    sphere.predictedCenter = pose.p;
    sphere.predictedRotation = pose.q;
    sphere.predictedPoseValid = true;
    spheres.pushBack(sphere);
  }

  capsules.clear();
  for (physx::PxU32 index = 0; index < provider.numRigidCapsules; ++index) {
    AvbdRigidCapsule capsule = provider.rigidCapsules[index];
    physx::PxTransform pose;
    if (!currentShapePose(capsule.targetKind, capsule.targetIndex,
                          capsule.shapeToRigidBody, capsule.center,
                          capsule.rotation, pose))
      return false;
    capsule.center = pose.p;
    capsule.rotation = pose.q;
    capsule.previousCenter = pose.p;
    capsule.previousRotation = pose.q;
    capsule.predictedCenter = pose.p;
    capsule.predictedRotation = pose.q;
    capsule.predictedPoseValid = true;
    capsules.pushBack(capsule);
  }

  convexes.clear();
  for (physx::PxU32 index = 0; index < provider.numRigidConvexes; ++index) {
    AvbdRigidConvex convex = provider.rigidConvexes[index];
    physx::PxTransform pose;
    if (!currentShapePose(convex.targetKind, convex.targetIndex,
                          convex.shapeToRigidBody, convex.center,
                          convex.rotation, pose))
      return false;
    convex.center = pose.p;
    convex.rotation = pose.q;
    convex.previousCenter = pose.p;
    convex.previousRotation = pose.q;
    convex.predictedCenter = pose.p;
    convex.predictedRotation = pose.q;
    convex.predictedPoseValid = true;
    convexes.pushBack(convex);
  }

  triangleSurfaces.clear();
  for (physx::PxU32 index = 0;
       index < provider.numRigidTriangleSurfaces; ++index) {
    AvbdRigidTriangleSurface surface =
        provider.rigidTriangleSurfaces[index];
    physx::PxTransform pose;
    if (!currentShapePose(surface.targetKind, surface.targetIndex,
                          surface.shapeToRigidBody, surface.center,
                          surface.rotation, pose))
      return false;
    surface.center = pose.p;
    surface.rotation = pose.q;
    surface.previousCenter = pose.p;
    surface.previousRotation = pose.q;
    triangleSurfaces.pushBack(surface);
  }

  AvbdSoftContactDetectionView detectionView;
  detectionView.particles = proxyParticles.begin();
  detectionView.numParticles = proxyParticles.size();
  detectionView.softBodies = collisionBodies.begin();
  detectionView.numSoftBodies = collisionBodies.size();
  detectionView.worldPlanes = provider.worldPlanes;
  detectionView.numWorldPlanes = provider.numWorldPlanes;
  detectionView.includeLegacyGround = false;
  detectionView.rigidBoxes = boxes.begin();
  detectionView.numRigidBoxes = boxes.size();
  detectionView.rigidSpheres = spheres.begin();
  detectionView.numRigidSpheres = spheres.size();
  detectionView.rigidCapsules = capsules.begin();
  detectionView.numRigidCapsules = capsules.size();
  detectionView.rigidConvexes = convexes.begin();
  detectionView.numRigidConvexes = convexes.size();
  detectionView.rigidTriangleSurfaces = triangleSurfaces.begin();
  detectionView.numRigidTriangleSurfaces = triangleSurfaces.size();
  detectionView.includeSoftTargets = provider.includeSoftTargets;
  AvbdOGCParams currentPoseParams;
  currentPoseParams.contactRadius = provider.contactRadius;
  if (!avbdDetectCurrentPoseOGCContacts(
          detectionView, contacts, currentPoseParams, nullptr,
          &contactWorkspace, &geometrySidecar))
    return false;

  for (physx::PxU32 contactIndex = 0; contactIndex < contacts.size();
       ++contactIndex) {
    AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
    const physx::PxU32 collisionFeatureParticle = geometry.particleIdx;
    physx::PxU32 collisionBodyIndex = PX_MAX_U32;
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < provider.numCollisionBodies; ++bodyIndex) {
      const AvbdSoftBodyCompiledData &compiled =
          provider.collisionBodies[bodyIndex].compiled;
      if (collisionFeatureParticle >= compiled.particleStart &&
          collisionFeatureParticle - compiled.particleStart <
              compiled.particleCount) {
        collisionBodyIndex = bodyIndex;
        break;
      }
    }
    if (collisionBodyIndex == PX_MAX_U32)
      return false;

    AvbdWeightedContactPoint expanded;
    physx::PxU32 queryCount = 1u;
    if (geometry.hasBarycentricQueryPoint()) {
      queryCount = 0u;
      while (queryCount < 3u &&
             geometry.queryParticleIndices[queryCount] != PX_MAX_U32)
        ++queryCount;
      if (queryCount == 0u)
        return false;
    }
    for (physx::PxU32 queryIndex = 0; queryIndex < queryCount; ++queryIndex) {
      const physx::PxU32 proxyIndex = geometry.hasBarycentricQueryPoint()
          ? geometry.queryParticleIndices[queryIndex]
          : collisionFeatureParticle;
      const physx::PxReal queryWeight = geometry.hasBarycentricQueryPoint()
          ? geometry.queryWeights[queryIndex]
          : 1.0f;
      if (proxyIndex >= provider.numCollisionVertexMappings ||
          !physx::PxIsFinite(queryWeight))
        return false;
      const AvbdWeightedContactPoint &mapping =
          provider.collisionVertexMappings[proxyIndex];
      for (physx::PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint)
        if (!expanded.appendMerged(mapping.particleIndices[endpoint],
                                   queryWeight * mapping.weights[endpoint]))
          return false;
    }
    expanded.removeNearZero();
    if (expanded.count == 0)
      return false;
    for (physx::PxU32 endpoint = 0; endpoint < expanded.count; ++endpoint)
      if (expanded.particleIndices[endpoint] >= numSoftParticles)
        return false;
    geometry.collisionFeatureParticleIdx = collisionFeatureParticle;
    geometry.queryBodyIndex = collisionBodyIndex;
    geometry.queryPoint = expanded;
    geometry.particleIdx = expanded.particleIndices[0];

    if (geometry.hasDeformableSurfaceTarget()) {
      physx::PxU32 targetProxyCount = 0u;
      while (targetProxyCount < 3u &&
             geometry.surfaceParticleIndices[targetProxyCount] != PX_MAX_U32)
        ++targetProxyCount;
      if (targetProxyCount == 0u)
        return false;
      AvbdWeightedContactPoint expandedTarget;
      for (physx::PxU32 targetIndex = 0u;
           targetIndex < targetProxyCount; ++targetIndex) {
        const physx::PxU32 proxyIndex =
            geometry.surfaceParticleIndices[targetIndex];
        const physx::PxReal proxyWeight =
            geometry.surfaceWeights[targetIndex];
        if (proxyIndex >= provider.numCollisionVertexMappings ||
            !physx::PxIsFinite(proxyWeight))
          return false;
        const AvbdWeightedContactPoint &mapping =
            provider.collisionVertexMappings[proxyIndex];
        for (physx::PxU32 endpoint = 0u; endpoint < mapping.count;
             ++endpoint)
          if (!expandedTarget.appendMerged(
                  mapping.particleIndices[endpoint],
                  proxyWeight * mapping.weights[endpoint]))
            return false;
      }
      expandedTarget.removeNearZero();
      if (expandedTarget.count == 0u)
        return false;
      for (physx::PxU32 endpoint = 0u;
           endpoint < expandedTarget.count; ++endpoint)
        if (expandedTarget.particleIndices[endpoint] >= numSoftParticles)
          return false;
      const physx::PxU32 targetProxyParticle =
          geometry.surfaceParticleIndices[0];
      physx::PxU32 targetBodyIndex = PX_MAX_U32;
      for (physx::PxU32 bodyIndex = 0u;
           bodyIndex < provider.numCollisionBodies; ++bodyIndex) {
        const AvbdSoftBodyCompiledData &compiled =
            provider.collisionBodies[bodyIndex].compiled;
        if (targetProxyParticle >= compiled.particleStart &&
            targetProxyParticle - compiled.particleStart <
                compiled.particleCount) {
          targetBodyIndex = bodyIndex;
          break;
        }
      }
      if (targetBodyIndex == PX_MAX_U32)
        return false;
      geometry.targetPoint = expandedTarget;
      geometry.source.targetBodyIndex = targetBodyIndex;
      geometry.targetIndex = targetBodyIndex;
    }

    AvbdOgcTriangleCoreCertificate *certificate =
        geometrySidecar.getTriangleCoreMutable(contactIndex);
    if (certificate) {
      for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
        const AvbdWeightedContactPoint proxyPoint =
            certificate->points[vertex];
        AvbdWeightedContactPoint expandedPoint;
        for (physx::PxU32 support = 0; support < proxyPoint.count;
             ++support) {
          const physx::PxU32 proxyIndex =
              proxyPoint.particleIndices[support];
          const physx::PxReal proxyWeight = proxyPoint.weights[support];
          if (proxyIndex >= provider.numCollisionVertexMappings ||
              !physx::PxIsFinite(proxyWeight))
            return false;
          const AvbdWeightedContactPoint &vertexMapping =
              provider.collisionVertexMappings[proxyIndex];
          for (physx::PxU32 endpoint = 0;
               endpoint < vertexMapping.count; ++endpoint)
            if (!expandedPoint.appendMerged(
                    vertexMapping.particleIndices[endpoint],
                    proxyWeight * vertexMapping.weights[endpoint]))
              return false;
        }
        expandedPoint.removeNearZero();
        certificate->points[vertex] = expandedPoint;
        for (physx::PxU32 endpoint = 0;
             endpoint < expandedPoint.count; ++endpoint)
          if (expandedPoint.particleIndices[endpoint] >= numSoftParticles ||
              !physx::PxIsFinite(expandedPoint.weights[endpoint]))
            return false;
      }
      if (!certificate->isValid())
        return false;
    }
  }
  return true;
}

// Pair-state persistence avoids rebuilding a full manifold for resting pairs,
// but it cannot be the only terminal admission rule: the final material or
// static projection can create a new soft/box overlap that has no source row
// in the prediction epoch.  Refit the immutable collision proxy into a
// conservative AABB and mark only those final-pose body/box pairs that can
// enter the OGC shell.  This is the broadphase half of a same-time DCD epoch;
// it neither advances time nor uses a previous pose/swept query.
static bool avbdMarkTerminalCurrentPoseBroadphaseBodies(
    const AvbdSoftIslandExecutionPlan *plan, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, physx::PxU8 *sourceBodyMask,
    physx::PxU32 numSourceBodyMask,
    physx::PxArray<physx::PxVec3> &bodyMinimum,
    physx::PxArray<physx::PxVec3> &bodyMaximum) {
  if (!plan || !plan->hasTerminalCurrentPoseGeometryPlan(numSoftParticles) ||
      !softParticles || !sourceBodyMask)
    return false;
  const AvbdOgcCurrentPoseGeometryProvider &provider =
      plan->terminalGeometryProvider;
  if (numSourceBodyMask != provider.numCollisionBodies ||
      (numBodies > 0 && !bodies))
    return false;

  const physx::PxReal shellRadius =
      physx::PxMax(provider.contactRadius, 1.0e-6f);
  bodyMinimum.resize(provider.numCollisionBodies);
  bodyMaximum.resize(provider.numCollisionBodies);
  bool markedAny = false;
  for (physx::PxU32 sourceBodyIndex = 0;
       sourceBodyIndex < provider.numCollisionBodies;
       ++sourceBodyIndex) {
    const AvbdSoftBodyCompiledData &compiled =
        provider.collisionBodies[sourceBodyIndex].compiled;
    if (compiled.particleStart >
            provider.numCollisionVertexMappings ||
        compiled.particleCount >
            provider.numCollisionVertexMappings -
                compiled.particleStart)
      return false;

    physx::PxVec3 minimum(PX_MAX_F32);
    physx::PxVec3 maximum(-PX_MAX_F32);
    bool hasDynamicProxyVertex = false;
    for (physx::PxU32 localVertex = 0;
         localVertex < compiled.particleCount; ++localVertex) {
      const physx::PxU32 proxyIndex =
          compiled.particleStart + localVertex;
      const AvbdWeightedContactPoint &mapping =
          provider.collisionVertexMappings[proxyIndex];
      if (mapping.count == 0)
        return false;
      physx::PxVec3 point(0.0f);
      bool dynamic = false;
      for (physx::PxU32 endpoint = 0; endpoint < mapping.count;
           ++endpoint) {
        const physx::PxU32 particleIndex = mapping.particleIndices[endpoint];
        const physx::PxReal weight = mapping.weights[endpoint];
        if (particleIndex >= numSoftParticles ||
            !physx::PxIsFinite(weight) ||
            !softParticles[particleIndex].position.isFinite())
          return false;
        point += softParticles[particleIndex].position * weight;
        dynamic = dynamic || softParticles[particleIndex].invMass > 0.0f;
      }
      if (!point.isFinite())
        return false;
      minimum.x = physx::PxMin(minimum.x, point.x);
      minimum.y = physx::PxMin(minimum.y, point.y);
      minimum.z = physx::PxMin(minimum.z, point.z);
      maximum.x = physx::PxMax(maximum.x, point.x);
      maximum.y = physx::PxMax(maximum.y, point.y);
      maximum.z = physx::PxMax(maximum.z, point.z);
      hasDynamicProxyVertex = hasDynamicProxyVertex || dynamic;
    }
    if (!hasDynamicProxyVertex || !minimum.isFinite() ||
        !maximum.isFinite()) {
      bodyMinimum[sourceBodyIndex] = physx::PxVec3(PX_MAX_F32);
      bodyMaximum[sourceBodyIndex] = physx::PxVec3(-PX_MAX_F32);
      continue;
    }
    bodyMinimum[sourceBodyIndex] = minimum;
    bodyMaximum[sourceBodyIndex] = maximum;
  }

  const auto overlapsSphereBounds =
      [shellRadius](const physx::PxVec3 &minimum,
                    const physx::PxVec3 &maximum,
                    const physx::PxVec3 &center,
                    physx::PxReal radius) {
        if (!center.isFinite() || !physx::PxIsFinite(radius) || radius < 0.0f)
          return false;
        const physx::PxReal extent = radius + shellRadius;
        const physx::PxVec3 targetMinimum = center - physx::PxVec3(extent);
        const physx::PxVec3 targetMaximum = center + physx::PxVec3(extent);
        return !(maximum.x < targetMinimum.x || minimum.x > targetMaximum.x ||
                 maximum.y < targetMinimum.y || minimum.y > targetMaximum.y ||
                 maximum.z < targetMinimum.z || minimum.z > targetMaximum.z);
      };
  const auto currentCenter =
      [bodies, numBodies](AvbdSoftContactTargetKind targetKind,
                         physx::PxU32 targetIndex,
                         const physx::PxTransform &shapeToRigidBody,
                         const physx::PxVec3 &staticCenter,
                         physx::PxVec3 &center) {
        center = staticCenter;
        if (targetKind != AvbdSoftContactTargetKind::eRIGID_BODY)
          return center.isFinite();
        if (!bodies || targetIndex >= numBodies)
          return false;
        const physx::PxTransform pose =
            physx::PxTransform(bodies[targetIndex].position,
                               bodies[targetIndex].rotation) *
            shapeToRigidBody;
        center = pose.p;
        return pose.isValid();
      };

  for (physx::PxU32 sourceBodyIndex = 0;
       sourceBodyIndex < provider.numCollisionBodies; ++sourceBodyIndex) {
    const physx::PxVec3 &minimum = bodyMinimum[sourceBodyIndex];
    const physx::PxVec3 &maximum = bodyMaximum[sourceBodyIndex];
    if (!minimum.isFinite() || !maximum.isFinite())
      continue;
    bool overlaps = false;

    for (physx::PxU32 planeIndex = 0;
         !overlaps && planeIndex < provider.numWorldPlanes; ++planeIndex) {
      const AvbdWorldPlane &plane = provider.worldPlanes[planeIndex];
      const physx::PxReal normalSq = plane.normal.magnitudeSquared();
      if (!physx::PxIsFinite(normalSq) || normalSq <= 1.0e-12f)
        continue;
      const physx::PxVec3 normal =
          plane.normal * physx::PxRecipSqrt(normalSq);
      const physx::PxVec3 center = (minimum + maximum) * 0.5f;
      const physx::PxVec3 extent = (maximum - minimum) * 0.5f;
      const physx::PxReal minimumDistance = center.dot(normal) -
          (physx::PxAbs(normal.x) * extent.x +
           physx::PxAbs(normal.y) * extent.y +
           physx::PxAbs(normal.z) * extent.z) - plane.offset;
      overlaps = physx::PxIsFinite(minimumDistance) &&
          minimumDistance <= shellRadius;
    }
    for (physx::PxU32 index = 0;
         !overlaps && index < provider.numRigidBoxes; ++index) {
      const AvbdRigidBox &box = provider.rigidBoxes[index];
      physx::PxVec3 center;
      if (!currentCenter(box.targetKind, box.targetIndex,
                         box.shapeToRigidBody, box.center, center))
        return false;
      overlaps = overlapsSphereBounds(
          minimum, maximum, center, box.halfExtent.magnitude());
    }
    for (physx::PxU32 index = 0;
         !overlaps && index < provider.numRigidSpheres; ++index) {
      const AvbdRigidSphere &sphere = provider.rigidSpheres[index];
      physx::PxVec3 center;
      if (!currentCenter(sphere.targetKind, sphere.targetIndex,
                         sphere.shapeToRigidBody, sphere.center, center))
        return false;
      overlaps = overlapsSphereBounds(minimum, maximum, center, sphere.radius);
    }
    for (physx::PxU32 index = 0;
         !overlaps && index < provider.numRigidCapsules; ++index) {
      const AvbdRigidCapsule &capsule = provider.rigidCapsules[index];
      physx::PxVec3 center;
      if (!currentCenter(capsule.targetKind, capsule.targetIndex,
                         capsule.shapeToRigidBody, capsule.center, center))
        return false;
      overlaps = overlapsSphereBounds(
          minimum, maximum, center, capsule.radius + capsule.halfHeight);
    }
    for (physx::PxU32 index = 0;
         !overlaps && index < provider.numRigidConvexes; ++index) {
      const AvbdRigidConvex &convex = provider.rigidConvexes[index];
      physx::PxVec3 center;
      if (!currentCenter(convex.targetKind, convex.targetIndex,
                         convex.shapeToRigidBody, convex.center, center))
        return false;
      overlaps = overlapsSphereBounds(
          minimum, maximum, center, convex.localRadius);
    }
    for (physx::PxU32 index = 0;
         !overlaps && index < provider.numRigidTriangleSurfaces; ++index) {
      const AvbdRigidTriangleSurface &surface =
          provider.rigidTriangleSurfaces[index];
      physx::PxVec3 center;
      if (!currentCenter(surface.targetKind, surface.targetIndex,
                         surface.shapeToRigidBody, surface.center, center))
        return false;
      overlaps = overlapsSphereBounds(
          minimum, maximum, center, surface.localRadius);
    }
    if (!overlaps && provider.includeSoftTargets) {
      for (physx::PxU32 targetBodyIndex = 0;
           targetBodyIndex < provider.numCollisionBodies; ++targetBodyIndex) {
        if (targetBodyIndex == sourceBodyIndex ||
            !bodyMinimum[targetBodyIndex].isFinite() ||
            !bodyMaximum[targetBodyIndex].isFinite())
          continue;
        const physx::PxVec3 targetMinimum =
            bodyMinimum[targetBodyIndex] - physx::PxVec3(shellRadius);
        const physx::PxVec3 targetMaximum =
            bodyMaximum[targetBodyIndex] + physx::PxVec3(shellRadius);
        if (!(maximum.x < targetMinimum.x || minimum.x > targetMaximum.x ||
              maximum.y < targetMinimum.y || minimum.y > targetMaximum.y ||
              maximum.z < targetMinimum.z || minimum.z > targetMaximum.z)) {
          overlaps = true;
          sourceBodyMask[targetBodyIndex] = 1u;
          break;
        }
      }
    }
    if (overlaps) {
      sourceBodyMask[sourceBodyIndex] = 1u;
      markedAny = true;
    }
  }
  return markedAny;
}

// Compute the terminal DCD dirty set from persistent pair state and recovery
// results.  This admission phase may mark pair refresh state, but it does not
// build contacts or project positions.
bool avbdPrepareTerminalCurrentPoseAdmission(
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    physx::PxArray<physx::PxU8> &terminalSourceBodyMask,
    physx::PxArray<physx::PxVec3> &broadphaseBodyMinimum,
    physx::PxArray<physx::PxVec3> &broadphaseBodyMaximum,
    physx::PxReal lengthScale) {
  bool terminalCurrentPoseRefreshNeeded = false;
  if (terminalSoftExecutionPlan) {
    terminalSourceBodyMask.resize(
        terminalSoftExecutionPlan->terminalGeometryProvider.
            numCollisionBodies);
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < terminalSourceBodyMask.size(); ++bodyIndex)
      terminalSourceBodyMask[bodyIndex] = 0u;
  }

  // The terminal epoch is a discrete current-pose OGC owner. A soft body
  // which opted into speculative CCD is owned by the swept manifold through
  // position and velocity handoff, so starting the terminal transaction for
  // an island containing such a body would create two incompatible owners.
  // Keep this admission precondition aligned with
  // avbdBuildTerminalCurrentPoseContacts(): "not applicable" must stop here,
  // rather than being reported later as a malformed terminal manifold and
  // rolling a valid swept response back to its solve-entry pose.
  if ((numSoftBodies > 0u && !softBodies) ||
      (terminalSoftExecutionPlan &&
       numSoftBodies != terminalSoftExecutionPlan->terminalGeometryProvider.
                            numCollisionBodies))
    return false;
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
    if (softBodies[bodyIndex].compiled.speculativeCCDEnabled)
      return false;

  // Keep the terminal DCD work set pair-local.  A contact epoch can have
  // hundreds of rows for one body/shape pair, but all of them map to the same
  // collision proxy and box descriptor; marking the descriptor once is enough
  // to rebuild a fresh manifold.  Matching the primitive key as well as the
  // target identity matters for a rigid actor carrying more than one shape.
  const auto markTerminalCurrentPosePair =
      [&](const AvbdSoftContactGeometry &geometry) {
        if (!terminalSoftExecutionPlan ||
            geometry.queryBodyIndex >= terminalSourceBodyMask.size())
          return;
        terminalSourceBodyMask[geometry.queryBodyIndex] = 1u;
        if (geometry.hasDeformableSurfaceTarget() &&
            geometry.targetIndex < terminalSourceBodyMask.size())
          terminalSourceBodyMask[geometry.targetIndex] = 1u;
      };
  if (terminalSoftExecutionPlan &&
      terminalSoftExecutionPlan->hasMixedOgcPairPlan(numShellContacts)) {
    // Re-query the prepared pair representatives at the final pose before
    // paying for proxy DCD.  An OGC shell admission is deliberately not by
    // itself a dirty marker: most admitted pairs remain separated after the
    // material solve, and rebuilding their complete proxy manifold was the
    // source of the 40+ ms spikes.  A resting zero-gap row remains owned by
    // the persistent pair manifold; only a true overlap or a prepared
    // triangle-core witness requests terminal narrowphase work.
    AvbdOgcPairState *pairStates =
        terminalSoftExecutionPlan->ogcPairStates;
    const physx::PxU32 *pairIndices =
        terminalSoftExecutionPlan->ogcPairIndices;
    const physx::PxU32 numPairStates =
        terminalSoftExecutionPlan->numOgcPairStates;
    const physx::PxReal refreshTolerance = physx::PxMax(
        1.0e-5f, 1.0e-4f * physx::PxMax(lengthScale, 1.0e-6f));
	const AvbdOgcGeometryEpochView geometryEpoch =
		makeOgcGeometryEpochView(terminalSoftExecutionPlan);
    if (pairStates && pairIndices &&
        terminalSoftExecutionPlan->numOgcPairIndices == numShellContacts) {
      for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
        const physx::PxU32 pairIndex = pairIndices[sci];
        if (pairIndex >= numPairStates)
          continue;
        AvbdOgcPairState &pair = pairStates[pairIndex];
        const AvbdSoftContactGeometry &geometry = shellContacts[sci].geometry;
        const bool dynamicRigid =
            geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
            geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies;
        const bool worldStatic =
            (geometry.source.type == AvbdSoftContactSource::eRIGID_SDF ||
             geometry.source.type == AvbdSoftContactSource::eGROUND) &&
            geometry.hasWorldStaticTarget();
        const bool deformable =
            geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
            geometry.hasDeformableSurfaceTarget();
        if (!pair.geometry.active ||
            (!dynamicRigid && !worldStatic && !deformable) ||
            geometry.queryBodyIndex != pair.key.sourceBodyIndex ||
            geometry.targetIndex != pair.key.targetBodyIndex ||
            geometry.source.primitiveKey != pair.key.primitiveKey)
          continue;

        const physx::PxVec3 queryPoint =
            avbdGetSoftContactQueryPoint(geometry, shellParticles);
        AvbdOgcCurrentPairGeometry currentGeometry;
        const AvbdSolverBody *dynamicTarget =
            dynamicRigid ? &bodies[geometry.targetIndex] : nullptr;
		const AvbdOgcRigidBoxGeometry* rigidBox =
			geometryEpoch.getRigidBox(sci, numShellContacts);
        bool geometryValid = false;
        if (deformable) {
          const physx::PxVec3 targetPoint =
              avbdGetSoftContactSurfacePoint(geometry, shellParticles);
          const physx::PxReal normalSq = geometry.normal.magnitudeSquared();
          if (queryPoint.isFinite() && targetPoint.isFinite() &&
              physx::PxIsFinite(normalSq) && normalSq > 1.0e-12f) {
            currentGeometry.normal =
                geometry.normal * physx::PxRecipSqrt(normalSq);
            currentGeometry.targetOffset = physx::PxVec3(0.0f);
            currentGeometry.signedGap =
                (queryPoint - targetPoint).dot(currentGeometry.normal);
            geometryValid = physx::PxIsFinite(currentGeometry.signedGap);
          }
        } else {
          geometryValid = queryPoint.isFinite() &&
              getCurrentOgcPairGeometry(
                  geometry, dynamicTarget, queryPoint, currentGeometry,
				  rigidBox);
        }
        if (!geometryValid ||
            !physx::PxIsFinite(currentGeometry.signedGap))
          continue;

        pair.geometry.minimumGap =
            physx::PxMin(pair.geometry.minimumGap,
                         currentGeometry.signedGap);

        // The pair state starts at a discrete, current-pose DCD epoch.  Its
        // representative carries the relative surface vector from that
        // epoch, so consume the pair safety budget with actual relative
        // motion rather than with an arbitrary iteration count.  This is the
        // OGC hand-off for a fast free rigid: even if a cached representative
        // has not yet crossed the surface, a relative displacement larger
        // than the DCD clearance means another final-pose manifold is now
        // required to cover the rest of the collision proxy.
        const physx::PxVec3 currentRelativePoint = dynamicRigid
            ? queryPoint -
                  (bodies[geometry.targetIndex].position +
                   currentGeometry.targetOffset)
            : queryPoint - pair.geometry.referenceRelativePoint;
        const physx::PxReal relativeEpochDisplacement =
            dynamicRigid
                ? (currentRelativePoint -
                   pair.geometry.referenceRelativePoint).magnitude()
                : 0.0f;
        const physx::PxReal safetyBudget = physx::PxMax(
            pair.trustRegion.safetyGap, refreshTolerance);
        if (dynamicRigid && currentRelativePoint.isFinite() &&
            pair.geometry.referenceRelativePoint.isFinite() &&
            physx::PxIsFinite(relativeEpochDisplacement) &&
            relativeEpochDisplacement >= safetyBudget) {
          pair.trustRegion.refreshRequested = true;
          pair.trustRegion.remainingSafeDisplacement = 0.0f;
          pair.trustRegion.accumulatedRelativeDisplacement = physx::PxMax(
              pair.trustRegion.accumulatedRelativeDisplacement,
              relativeEpochDisplacement);
          markTerminalCurrentPosePair(geometry);
        }
        bool coreOverlap = false;
		const AvbdOgcTriangleCoreCertificate* triangleCore =
			geometryEpoch.getTriangleCore(sci, numShellContacts);
        if (triangleCore) {
          physx::PxReal coreGap = 0.0f;
          // A prepared core certificate is immutable detector metadata, not
          // proof that the final pose still overlaps.  Re-evaluate its three
          // expanded vertices before scheduling expensive proxy DCD; otherwise
          // every resting face keeps the terminal closure hot forever.
          coreOverlap = !getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, dynamicRigid ? &bodies[geometry.targetIndex] : nullptr,
              shellParticles,
			  numShellParticles, coreGap, PX_MAX_U32,
			  physx::PxVec3(0.0f), rigidBox, triangleCore) ||
              coreGap < -refreshTolerance;
        }
        if (currentGeometry.signedGap < -refreshTolerance || coreOverlap) {
          pair.trustRegion.refreshRequested = true;
          markTerminalCurrentPosePair(geometry);
        }
        // `refreshRequested` is raised only when the pair moves beyond the
        // published OGC safety domain (or a true current overlap was found).
        // `admittedAtBoundary` is deliberately *not* a terminal-DCD trigger:
        // it records a valid start-of-solve clip and is consumed later by the
        // inelastic velocity owner.  Treating every valid boundary admission
        // as dirty rebuilt the complete proxy manifold on every resting frame.
        if (pair.trustRegion.refreshRequested)
          markTerminalCurrentPosePair(geometry);
      }
    }
    for (physx::PxU32 pairIndex = 0;
         pairIndex < terminalSoftExecutionPlan->numOgcPairStates;
         ++pairIndex) {
      const AvbdOgcPairState &pair =
          terminalSoftExecutionPlan->ogcPairStates[pairIndex];
      if (pair.geometry.active && pair.trustRegion.refreshRequested) {
        terminalCurrentPoseRefreshNeeded = true;
        break;
      }
    }
  }
  if (terminalSoftExecutionPlan &&
      terminalSoftExecutionPlan->ogcPairStates) {
    for (physx::PxU32 pairIndex = 0;
         pairIndex < terminalSoftExecutionPlan->numOgcPairStates;
         ++pairIndex) {
      AvbdOgcPairState &pair =
          terminalSoftExecutionPlan->ogcPairStates[pairIndex];
      physx::PxU32 contactIndex = PX_MAX_U32;
      if (pair.solve.hasPendingLocalVelocity(
              AvbdOgcVelocityContactDomain::eSELECTION))
        contactIndex = pair.solve.localVelocityContact;
      if (contactIndex >= numShellContacts)
        continue;
      pair.trustRegion.refreshRequested = true;
      terminalCurrentPoseRefreshNeeded = true;
      markTerminalCurrentPosePair(shellContacts[contactIndex].geometry);
    }
  }

  // Cover final-pose contacts born after the prediction manifold was built.
  // Existing OgcPairState rows still provide the cheap persistent path above;
  // this conservative AABB admission is only the missing-new-pair fallback.
  // The following detector remains exact current-pose SDF/feature DCD.
  if (terminalSoftExecutionPlan && shellParticles &&
      numShellParticles > 0 &&
      avbdMarkTerminalCurrentPoseBroadphaseBodies(
          terminalSoftExecutionPlan, bodies, numBodies, shellParticles,
          numShellParticles, terminalSourceBodyMask.begin(),
          terminalSourceBodyMask.size(), broadphaseBodyMinimum,
          broadphaseBodyMaximum))
    terminalCurrentPoseRefreshNeeded = true;


  return terminalCurrentPoseRefreshNeeded;
}

} // namespace Dy
} // namespace physx
