// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

// Return the detector-time relative translation which moves the entire
// triangle through one selected box face.  This uses the complete triangle
// bounds rather than the centroid SDF witness: after translating by this
// amount, every point of the triangle is outside that face's supporting
// plane, so the convex triangle cannot still overlap the OBB.
bool getRigidBoxTriangleCoreExitDistance(
    const physx::PxVec3 &halfExtent, const physx::PxVec3 &minimum,
    const physx::PxVec3 &maximum, physx::PxReal margin,
    physx::PxU32 face,
    physx::PxReal &distance) {
  if (!halfExtent.isFinite() || !minimum.isFinite() ||
      !maximum.isFinite() || !physx::PxIsFinite(margin) || face >= 6u)
    return false;

  if (halfExtent.x <= 0.0f || halfExtent.y <= 0.0f ||
      halfExtent.z <= 0.0f || minimum.x > maximum.x ||
      minimum.y > maximum.y || minimum.z > maximum.z)
    return false;

  switch (face) {
  case 0u:
    distance = halfExtent.x - minimum.x;
    break;
  case 1u:
    distance = maximum.x + halfExtent.x;
    break;
  case 2u:
    distance = halfExtent.y - minimum.y;
    break;
  case 3u:
    distance = maximum.y + halfExtent.y;
    break;
  case 4u:
    distance = halfExtent.z - minimum.z;
    break;
  default:
    distance = maximum.z + halfExtent.z;
    break;
  }
  if (!physx::PxIsFinite(distance) || distance < 0.0f)
    return false;
  // The detector's existing one-face certificate adds the same small
  // current-pose DCD clearance.  Retain it here so an exactly coplanar
  // triangle is pushed strictly outside instead of cycling on inclusive SAT.
  distance += physx::PxMax(1.0e-5f, margin * 0.02f);
  return physx::PxIsFinite(distance) && distance > 0.0f;
}

bool getRigidBoxTriangleCoreMinimumExitFace(
    const physx::PxVec3 &halfExtent, const physx::PxVec3 &minimumLocal,
    const physx::PxVec3 &maximumLocal, physx::PxReal margin,
    physx::PxU32 &face,
    physx::PxReal &distance) {
  face = PX_MAX_U32;
  distance = PX_MAX_F32;
  for (physx::PxU32 candidateFace = 0; candidateFace < 6u;
       ++candidateFace) {
    physx::PxReal candidateDistance = 0.0f;
    if (!getRigidBoxTriangleCoreExitDistance(
            halfExtent, minimumLocal, maximumLocal, margin,
            candidateFace, candidateDistance))
      return false;
    if (candidateDistance < distance) {
      face = candidateFace;
      distance = candidateDistance;
    }
  }
  return face < 6u && physx::PxIsFinite(distance) && distance > 0.0f;
}

bool accumulateRigidBoxTriangleCoreFaceExits(
    const physx::PxVec3 &halfExtent,
    const physx::PxVec3 &minimumLocal,
    const physx::PxVec3 &maximumLocal,
    physx::PxReal faceExits[6]) {
  if (!faceExits || !halfExtent.isFinite() || !minimumLocal.isFinite() ||
      !maximumLocal.isFinite() || halfExtent.x <= 0.0f ||
      halfExtent.y <= 0.0f || halfExtent.z <= 0.0f ||
      minimumLocal.x > maximumLocal.x ||
      minimumLocal.y > maximumLocal.y ||
      minimumLocal.z > maximumLocal.z)
    return false;

  const physx::PxReal exits[6] = {
      halfExtent.x - minimumLocal.x, maximumLocal.x + halfExtent.x,
      halfExtent.y - minimumLocal.y, maximumLocal.y + halfExtent.y,
      halfExtent.z - minimumLocal.z, maximumLocal.z + halfExtent.z};
  for (physx::PxU32 face = 0u; face < 6u; ++face) {
    if (!physx::PxIsFinite(faceExits[face]) || faceExits[face] < 0.0f ||
        !physx::PxIsFinite(exits[face]))
      return false;
    faceExits[face] = physx::PxMax(
        faceExits[face], physx::PxMax(0.0f, exits[face]));
  }
  return true;
}

bool getCurrentRigidBoxTriangleCoreLocalBounds(
    const AvbdSoftContactGeometry &geometry,
    const physx::PxTransform &boxToWorld,
    const AvbdSoftParticle *particles, physx::PxU32 numParticles,
    physx::PxVec3 &minimumLocal, physx::PxVec3 &maximumLocal,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    const AvbdOgcTriangleCoreCertificate *certificate) {
  if (!particles || !certificate ||
      !boxToWorld.isValid() || !movedParticleDisplacement.isFinite())
    return false;
  if (certificate && !certificate->isValid())
    return false;

  minimumLocal = physx::PxVec3(PX_MAX_F32);
  maximumLocal = physx::PxVec3(-PX_MAX_F32);
  for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
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
      if (particleIndex >= numParticles || !physx::PxIsFinite(weight) ||
          !particles[particleIndex].position.isFinite())
        return false;
      point += particles[particleIndex].position * weight;
      if (particleIndex == movedParticleIndex)
        point += movedParticleDisplacement * weight;
      weightSum += weight;
    }
    if (!point.isFinite() || !physx::PxIsFinite(weightSum) ||
        physx::PxAbs(weightSum - 1.0f) > 1.0e-3f)
      return false;
    const physx::PxVec3 localPoint = boxToWorld.transformInv(point);
    if (!localPoint.isFinite())
      return false;
    minimumLocal = minimumLocal.minimum(localPoint);
    maximumLocal = maximumLocal.maximum(localPoint);
  }
  return minimumLocal.isFinite() && maximumLocal.isFinite() &&
      minimumLocal.x <= maximumLocal.x &&
      minimumLocal.y <= maximumLocal.y &&
      minimumLocal.z <= maximumLocal.z;
}

physx::PxVec3 getRigidBoxTriangleCoreExitNormalLocal(
    physx::PxU32 face) {
  switch (face) {
  case 0u:
    return physx::PxVec3(1.0f, 0.0f, 0.0f);
  case 1u:
    return physx::PxVec3(-1.0f, 0.0f, 0.0f);
  case 2u:
    return physx::PxVec3(0.0f, 1.0f, 0.0f);
  case 3u:
    return physx::PxVec3(0.0f, -1.0f, 0.0f);
  case 4u:
    return physx::PxVec3(0.0f, 0.0f, 1.0f);
  default:
    return physx::PxVec3(0.0f, 0.0f, -1.0f);
  }
}

// Current-pose validity test for a complete triangle-core certificate.  The
// prepared core row stores a centroid for AL, but the OGC scheduler owns the
// three independently embedded collision vertices.  A cached certificate is
// therefore a refresh hint only; it becomes a terminal overlap only when its
// complete support plane is actually crossed at the current solve pose.
bool getCurrentRigidBoxTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody *body,
    const AvbdSoftParticle *particles, physx::PxU32 numParticles,
    physx::PxReal &faceGap,
    physx::PxU32 movedParticleIndex,
    const physx::PxVec3 &movedParticleDisplacement,
    const AvbdOgcRigidBoxGeometry *rigidBox,
    const AvbdOgcTriangleCoreCertificate *certificate) {
  if (!particles || !rigidBox || !certificate)
    return false;
  if (!rigidBox->valid)
    return false;

  physx::PxTransform boxToWorld = rigidBox->shapeToTarget;
  if (geometry.hasRigidBodyTarget()) {
    if (!body || !body->position.isFinite() || !body->rotation.isFinite())
      return false;
    boxToWorld = physx::PxTransform(body->position, body->rotation) *
        rigidBox->shapeToTarget;
  } else if (!geometry.hasWorldStaticTarget()) {
    return false;
  }
  if (!boxToWorld.isValid())
    return false;

  physx::PxVec3 minimumLocal(0.0f), maximumLocal(0.0f);
  if (!getCurrentRigidBoxTriangleCoreLocalBounds(
          geometry, boxToWorld, particles, numParticles,
          minimumLocal, maximumLocal, movedParticleIndex,
          movedParticleDisplacement, certificate))
    return false;
  physx::PxU32 exitFace = PX_MAX_U32;
  physx::PxReal exitDistance = 0.0f;
  if (!getRigidBoxTriangleCoreMinimumExitFace(
          rigidBox->halfExtent, minimumLocal, maximumLocal,
          geometry.margin, exitFace, exitDistance))
    return false;
  const physx::PxVec3 halfExtent = rigidBox->halfExtent;
  if (halfExtent.x <= 0.0f ||
      halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;
  const physx::PxVec3 normalLocal =
      getRigidBoxTriangleCoreExitNormalLocal(exitFace);
  const physx::PxVec3 rawNormalWorld = boxToWorld.q.rotate(normalLocal);
  const physx::PxReal worldNormalLengthSq =
      rawNormalWorld.magnitudeSquared();
  if (!rawNormalWorld.isFinite() ||
      !physx::PxIsFinite(worldNormalLengthSq) ||
      worldNormalLengthSq <= 1.0e-12f)
    return false;
  const physx::PxVec3 normalWorld =
      rawNormalWorld * physx::PxRecipSqrt(worldNormalLengthSq);
  const physx::PxVec3 surfaceLocal(normalLocal.x * halfExtent.x,
                                    normalLocal.y * halfExtent.y,
                                    normalLocal.z * halfExtent.z);
  const physx::PxVec3 surfaceWorld =
      boxToWorld.transform(surfaceLocal);

  faceGap = PX_MAX_F32;
  for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
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
      if (particleIndex >= numParticles || !physx::PxIsFinite(weight) ||
          !particles[particleIndex].position.isFinite())
        return false;
      point += particles[particleIndex].position * weight;
      if (particleIndex == movedParticleIndex)
        point += movedParticleDisplacement * weight;
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
