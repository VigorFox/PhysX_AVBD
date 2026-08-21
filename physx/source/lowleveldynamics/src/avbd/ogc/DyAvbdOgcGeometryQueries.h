// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_GEOMETRY_QUERIES_H
#define DY_AVBD_OGC_GEOMETRY_QUERIES_H

#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "avbd/solver/soft/DyAvbdSoftBody.h"

namespace physx {
namespace Dy {

PX_FORCE_INLINE bool queryCurrentPairSignedDistance(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const physx::PxVec3 &queryPoint, physx::PxReal &signedDistance,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr) {
  AvbdOgcCurrentPairGeometry currentGeometry;
  if (!getCurrentOgcPairGeometry(
          geometry, &body, queryPoint, currentGeometry, rigidBox))
    return false;
  signedDistance = currentGeometry.signedGap;
  return physx::PxIsFinite(signedDistance);
}

// Return the signed clearance of the complete expanded collision triangle to
// the OBB support plane selected by its discrete core certificate.  Moving one
// simulation particle is optional; this makes the same function usable for
// the soft candidate admission and the rigid candidate admission.  It is a
// static current-pose query -- no segment interpolation, TOI, or CCD state is
// involved.
PX_FORCE_INLINE bool queryCurrentPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody &body,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    physx::PxReal &faceGap,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr,
    const AvbdOgcTriangleCoreCertificate *certificate = nullptr) {
  return getCurrentRigidBoxTriangleCoreFaceGap(
      geometry, &body, softParticles, numSoftParticles, faceGap,
      movedParticleIndex, movedParticleDisplacement, rigidBox, certificate);
}

// Static targets participate in the same OGC epoch as a dynamic rigid, but
// their shape pose is already world-space and has no 6DOF endpoint.  These
// queries intentionally use only the current pose; they are not swept tests.
PX_FORCE_INLINE bool queryCurrentWorldStaticPairSignedDistance(
    const AvbdSoftContactGeometry &geometry,
    const physx::PxVec3 &queryPoint, physx::PxReal &signedDistance,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr) {
  AvbdOgcCurrentPairGeometry currentGeometry;
  if (!getCurrentOgcPairGeometry(
          geometry, nullptr, queryPoint, currentGeometry, rigidBox))
    return false;
  signedDistance = currentGeometry.signedGap;
  return physx::PxIsFinite(signedDistance);
}

PX_FORCE_INLINE bool queryCurrentWorldStaticPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    physx::PxReal &faceGap,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr,
    const AvbdOgcTriangleCoreCertificate *certificate = nullptr) {
  return getCurrentRigidBoxTriangleCoreFaceGap(
      geometry, nullptr, softParticles, numSoftParticles, faceGap,
      movedParticleIndex, movedParticleDisplacement, rigidBox, certificate);
}

// Evaluate the complete collision triangle on the same frame-start ->
// endpoint interpolation used by OGC warmstart admission.  This is a
// trust-region query, not swept CCD: it only finds the largest admissible
// position update within the already selected simulation dt.  Keeping the
// triangle here (rather than its detector centroid) makes the warmstart
// invariant match the terminal triangle-core manifold.
PX_FORCE_INLINE bool queryInterpolatedPairTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSolverBody &initialBody, const AvbdSolverBody &endpointBody,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal alpha, physx::PxReal &faceGap,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr,
    const AvbdOgcTriangleCoreCertificate *certificate = nullptr) {
  if (!softParticles || !physx::PxIsFinite(alpha) || alpha < 0.0f ||
      alpha > 1.0f || !initialBody.position.isFinite() ||
      !initialBody.rotation.isFinite() || !endpointBody.position.isFinite() ||
      !endpointBody.rotation.isFinite() ||
      !rigidBox || !certificate)
    return false;
  if (!rigidBox->valid || (certificate && !certificate->isValid()))
    return false;
  const physx::PxVec3 halfExtent = rigidBox->halfExtent;
  const physx::PxTransform shapeToBody = rigidBox->shapeToTarget;

  const physx::PxQuat endpointShapeRotation =
      endpointBody.rotation * shapeToBody.q;
  const physx::PxVec3 endpointShapePosition =
      endpointBody.position +
      endpointBody.rotation.rotate(shapeToBody.p);
  physx::PxVec3 minimumLocal(0.0f), maximumLocal(0.0f);
  if (!getCurrentRigidBoxTriangleCoreLocalBounds(
          geometry,
          physx::PxTransform(endpointShapePosition, endpointShapeRotation),
          softParticles, numSoftParticles, minimumLocal, maximumLocal,
          PX_MAX_U32, physx::PxVec3(0.0f), certificate))
    return false;
  physx::PxU32 exitFace = PX_MAX_U32;
  physx::PxReal exitDistance = 0.0f;
  if (!getRigidBoxTriangleCoreMinimumExitFace(
          halfExtent, minimumLocal, maximumLocal, geometry.margin,
          exitFace, exitDistance))
    return false;

  const physx::PxVec3 normalLocal =
      getRigidBoxTriangleCoreExitNormalLocal(exitFace);
  AvbdSolverBody body = endpointBody;
  body.position = initialBody.position +
      (endpointBody.position - initialBody.position) * alpha;
  body.rotation =
      physx::PxSlerp(alpha, initialBody.rotation, endpointBody.rotation);
  body.projectLockedPose(initialBody.position, initialBody.rotation);
  if (!body.position.isFinite() || !body.rotation.isFinite())
    return false;

  const physx::PxQuat shapeRotation = body.rotation * shapeToBody.q;
  const physx::PxVec3 shapePosition =
      body.position + body.rotation.rotate(shapeToBody.p);
  const physx::PxVec3 rawNormalWorld = shapeRotation.rotate(normalLocal);
  const physx::PxReal worldNormalLengthSq = rawNormalWorld.magnitudeSquared();
  if (!shapeRotation.isFinite() || !shapePosition.isFinite() ||
      !rawNormalWorld.isFinite() || !physx::PxIsFinite(worldNormalLengthSq) ||
      worldNormalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalWorld =
      rawNormalWorld * physx::PxRecipSqrt(worldNormalLengthSq);
  const physx::PxVec3 surfaceLocal(normalLocal.x * halfExtent.x,
                                    normalLocal.y * halfExtent.y,
                                    normalLocal.z * halfExtent.z);
  const physx::PxVec3 surfaceWorld =
      shapePosition + shapeRotation.rotate(surfaceLocal);

  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0; vertex < 3; ++vertex) {
    AvbdWeightedContactPoint mapping;
    if (!resolveOgcTriangleCorePoint(
            geometry, certificate, vertex, mapping))
      return false;
    if (mapping.count == 0 || mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
      return false;
    physx::PxVec3 point(0.0f);
    physx::PxReal weightSum = 0.0f;
    for (physx::PxU32 support = 0; support < mapping.count; ++support) {
      const physx::PxU32 particleIndex = mapping.particleIndices[support];
      const physx::PxReal weight = mapping.weights[support];
      if (particleIndex >= numSoftParticles || !physx::PxIsFinite(weight) ||
          !softParticles[particleIndex].initialPosition.isFinite() ||
          !softParticles[particleIndex].position.isFinite())
        return false;
      const physx::PxVec3 position =
          softParticles[particleIndex].initialPosition +
          (softParticles[particleIndex].position -
           softParticles[particleIndex].initialPosition) * alpha;
      point += position * weight;
      weightSum += weight;
    }
    if (!point.isFinite() || !physx::PxIsFinite(weightSum) ||
        physx::PxAbs(weightSum - 1.0f) > 1.0e-3f)
      return false;
    const physx::PxReal gap = (point - surfaceWorld).dot(normalWorld);
    if (!physx::PxIsFinite(gap))
      return false;
    faceGap = physx::PxMin(faceGap, gap);
  }
  return physx::PxIsFinite(faceGap);
}

} // namespace Dy
} // namespace physx

#endif
