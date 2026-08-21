// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

namespace {
// Contact detection starts from a cooked collision proxy, while Scene expands
// the accepted row to a weighted simulation-particle query point before the
// solve.  A cached proxy face normal/surface is therefore not authoritative
// for a late safety projection.  Box contacts retain this compact descriptor
// so the recovery stages can query the actual, current OBB at the expanded
// point.  This is endpoint DCD only: it neither samples a swept segment nor
// uses the OGC shell margin as penetration.
struct AvbdCurrentRigidBoxSdf {
  physx::PxReal signedDistance{0.0f};
  physx::PxVec3 normal{0.0f};
  physx::PxVec3 surfacePoint{0.0f};
  // Surface point relative to the dynamic solver body's origin.  It is zero
  // for a world-static box, which has no movable endpoint.
  physx::PxVec3 rigidOffset{0.0f};
};

static bool queryCurrentRigidBoxSdf(
    const AvbdSoftContactGeometry &geometry,
    const AvbdOgcRigidBoxGeometry &boxGeometry,
    const AvbdSolverBody *dynamicBody, const physx::PxVec3 &queryPoint,
    AvbdCurrentRigidBoxSdf &result) {
  if (!boxGeometry.valid || !queryPoint.isFinite())
    return false;

  const physx::PxVec3 halfExtent = boxGeometry.halfExtent;
  if (!halfExtent.isFinite() || halfExtent.x <= 0.0f ||
      halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
    return false;

  physx::PxTransform shapeToWorld = boxGeometry.shapeToTarget;
  if (geometry.hasRigidBodyTarget()) {
    if (!dynamicBody || !dynamicBody->position.isFinite() ||
        !dynamicBody->rotation.isFinite())
      return false;
    shapeToWorld = physx::PxTransform(dynamicBody->position,
                                      dynamicBody->rotation) *
                   boxGeometry.shapeToTarget;
  } else if (!geometry.hasWorldStaticTarget()) {
    return false;
  }

  const physx::PxVec3 localPoint = shapeToWorld.transformInv(queryPoint);
  if (!localPoint.isFinite())
    return false;

  const physx::PxVec3 q(physx::PxAbs(localPoint.x) - halfExtent.x,
                         physx::PxAbs(localPoint.y) - halfExtent.y,
                         physx::PxAbs(localPoint.z) - halfExtent.z);
  const bool inside = q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f;
  physx::PxReal signedDistance = 0.0f;
  physx::PxVec3 localNormal(0.0f);
  physx::PxVec3 surfaceLocal(0.0f);
  if (inside) {
    signedDistance = physx::PxMax(q.x, physx::PxMax(q.y, q.z));
    if (q.x > q.y && q.x > q.z)
      localNormal = physx::PxVec3(localPoint.x >= 0.0f ? 1.0f : -1.0f,
                                   0.0f, 0.0f);
    else if (q.y > q.z)
      localNormal = physx::PxVec3(0.0f,
                                   localPoint.y >= 0.0f ? 1.0f : -1.0f,
                                   0.0f);
    else
      localNormal = physx::PxVec3(0.0f, 0.0f,
                                   localPoint.z >= 0.0f ? 1.0f : -1.0f);
    surfaceLocal = localPoint - localNormal * signedDistance;
  } else {
    const physx::PxVec3 outside(
        physx::PxMax(q.x, 0.0f), physx::PxMax(q.y, 0.0f),
        physx::PxMax(q.z, 0.0f));
    signedDistance = outside.magnitude();
    if (signedDistance > 1.0e-10f) {
      localNormal = physx::PxVec3(
                        localPoint.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f) *
                    outside.x +
                    physx::PxVec3(0.0f,
                        localPoint.y >= 0.0f ? 1.0f : -1.0f, 0.0f) *
                    outside.y +
                    physx::PxVec3(0.0f, 0.0f,
                        localPoint.z >= 0.0f ? 1.0f : -1.0f) *
                    outside.z;
      localNormal *= 1.0f / signedDistance;
    } else {
      // The inside branch owns the exact boundary.  This only guards an
      // underflowing outside distance and keeps the query fail-safe.
      localNormal = physx::PxVec3(0.0f, 1.0f, 0.0f);
    }
    surfaceLocal = localPoint;
    surfaceLocal.x = physx::PxClamp(surfaceLocal.x, -halfExtent.x,
                                     halfExtent.x);
    surfaceLocal.y = physx::PxClamp(surfaceLocal.y, -halfExtent.y,
                                     halfExtent.y);
    surfaceLocal.z = physx::PxClamp(surfaceLocal.z, -halfExtent.z,
                                     halfExtent.z);
  }

  const physx::PxVec3 normal = shapeToWorld.q.rotate(localNormal);
  const physx::PxReal normalLengthSq = normal.magnitudeSquared();
  const physx::PxVec3 surfacePoint = shapeToWorld.transform(surfaceLocal);
  if (!physx::PxIsFinite(signedDistance) || !normal.isFinite() ||
      !physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f ||
      !surfacePoint.isFinite())
    return false;

  result.signedDistance = signedDistance;
  result.normal = normal * physx::PxRecipSqrt(normalLengthSq);
  result.surfacePoint = surfacePoint;
  result.rigidOffset = geometry.hasRigidBodyTarget()
                           ? dynamicBody->rotation.rotate(
                                 boxGeometry.shapeToTarget.transform(
                                     surfaceLocal))
                           : physx::PxVec3(0.0f);
  return result.rigidOffset.isFinite();
}

} // namespace

bool getCurrentOgcPairGeometry(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSolverBody *dynamicTarget,
    const physx::PxVec3 &queryPoint,
    AvbdOgcCurrentPairGeometry &result,
    const AvbdOgcRigidBoxGeometry *rigidBox) {
  const bool dynamic = geometry.hasRigidBodyTarget();
  const bool worldStatic = geometry.hasWorldStaticTarget();
  if (dynamic == worldStatic || (dynamic && !dynamicTarget) ||
      (worldStatic && dynamicTarget))
    return false;

  if (rigidBox && rigidBox->valid) {
    AvbdCurrentRigidBoxSdf boxQuery;
    if (!queryCurrentRigidBoxSdf(
            geometry, *rigidBox, dynamicTarget, queryPoint, boxQuery))
      return false;
    result.normal = boxQuery.normal;
    result.targetOffset = boxQuery.rigidOffset;
    result.signedGap = boxQuery.signedDistance;
    return true;
  }

  const physx::PxReal normalLengthSq = geometry.normal.magnitudeSquared();
  if (!physx::PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f)
    return false;
  result.normal = geometry.normal * physx::PxRecipSqrt(normalLengthSq);
  if (worldStatic) {
    if (!geometry.surfacePoint.isFinite())
      return false;
    result.targetOffset = physx::PxVec3(0.0f);
    result.signedGap =
        (queryPoint - geometry.surfacePoint).dot(result.normal);
    return physx::PxIsFinite(result.signedGap);
  }

  if (!dynamicTarget->position.isFinite() ||
      !dynamicTarget->rotation.isFinite())
    return false;
  result.targetOffset =
      dynamicTarget->rotation.rotate(geometry.rigidLocalPoint);
  const physx::PxVec3 surfacePoint =
      dynamicTarget->position + result.targetOffset;
  if (!result.targetOffset.isFinite() || !surfacePoint.isFinite())
    return false;
  result.signedGap = (queryPoint - surfacePoint).dot(result.normal);
  return physx::PxIsFinite(result.signedGap);
}


} // namespace Dy
} // namespace physx
