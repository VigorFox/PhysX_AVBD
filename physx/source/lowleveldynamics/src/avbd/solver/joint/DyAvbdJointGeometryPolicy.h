// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_GEOMETRY_POLICY_H
#define DY_AVBD_JOINT_GEOMETRY_POLICY_H

#include "avbd/solver/DyAvbdSolver.h"
#include "CmConeLimitHelper.h"

namespace physx {
namespace Dy {

PX_FORCE_INLINE bool computeEllipticalConeConstraint(
    const AvbdD6JointConstraint &joint, const physx::PxQuat &rotA,
    const physx::PxQuat &rotB, physx::PxVec3 &worldAxis,
    physx::PxReal &violation) {
  const physx::PxU32 ellipticalConeFlags =
      AvbdD6JointConstraint::eD6_LEGACY_CONE_LIMIT_ACTIVE |
      AvbdD6JointConstraint::eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE;
  if ((joint.sourceFlags & ellipticalConeFlags) == 0 ||
      joint.coneAngleLimit <= 0.0f || joint.coneAngleLimitZ <= 0.0f)
    return false;

  physx::PxQuat worldFrameA = rotA * joint.localFrameA;
  physx::PxQuat worldFrameB = rotB * joint.localFrameB;
  worldFrameA.normalize();
  worldFrameB.normalize();
  physx::PxQuat relative = worldFrameA.getConjugate() * worldFrameB;
  relative.normalize();
  if (relative.w < 0.0f)
    relative = -relative;

  physx::PxQuat swing, twist;
  physx::PxSeparateSwingTwist(relative, swing, twist);
  if (swing.w < 0.0f)
    swing = -swing;

  physx::Cm::ConeLimitHelperTanLess helper(
      joint.coneAngleLimit, joint.coneAngleLimitZ);
  physx::PxVec3 localAxis;
  physx::PxReal signedInsideError = 0.0f;
  helper.getLimit(swing, localAxis, signedInsideError);
  worldAxis = worldFrameA.rotate(localAxis);
  const physx::PxReal axisMagnitude = worldAxis.magnitude();
  if (!worldAxis.isFinite() || !PxIsFinite(signedInsideError) ||
      axisMagnitude <= 1e-6f)
    return false;
  worldAxis *= 1.0f / axisMagnitude;
  // ConeLimitHelperTanLess returns a positive error inside the cone and a
  // negative error outside. AVBD's unilateral cone convention uses a
  // positive outward violation.
  violation = -signedInsideError;
  return PxIsFinite(violation);
}

PX_FORCE_INLINE physx::PxReal computeRotationDeltaMagnitude(
    const physx::PxQuat &current, const physx::PxQuat &previous) {
  physx::PxQuat deltaQ = current * previous.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  return 2.0f * physx::PxSqrt(deltaQ.x * deltaQ.x +
                              deltaQ.y * deltaQ.y +
                              deltaQ.z * deltaQ.z);
}

} // namespace Dy
} // namespace physx

#endif
