// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_COUPLED_MATH_H
#define DY_AVBD_JOINT_COUPLED_MATH_H

#include "avbd/core/DyAvbdTypes.h"
#include "foundation/PxMath.h"
#include "foundation/PxQuat.h"

namespace physx {
namespace Dy {

// Matches joint::computeJacobianAxes in ExtConstraintHelper.h.  These rows are
// the world-space derivatives of imag(qa^-1*qb) with respect to wB-wA.  A
// SLERP spring uses them when stiffness is non-zero; only velocity-only SLERP
// keeps the fixed world X/Y/Z rows.
PX_FORCE_INLINE void computeSlerpJacobianAxes(physx::PxVec3 rows[3],
                                              const physx::PxQuat &qa,
                                              const physx::PxQuat &qb) {
  const physx::PxReal wa = qa.w;
  const physx::PxReal wb = qb.w;
  const physx::PxVec3 va(qa.x, qa.y, qa.z);
  const physx::PxVec3 vb(qb.x, qb.y, qb.z);
  const physx::PxVec3 c = vb * wa + va * wb;
  const physx::PxReal d0 = wa * wb;
  const physx::PxReal d1 = va.dot(vb);
  const physx::PxReal d = d0 - d1;

  rows[0] =
      (va * vb.x + vb * va.x + physx::PxVec3(d, c.z, -c.y)) * 0.5f;
  rows[1] =
      (va * vb.y + vb * va.y + physx::PxVec3(-c.z, d, c.x)) * 0.5f;
  rows[2] =
      (va * vb.z + vb * va.z + physx::PxVec3(c.y, -c.x, d)) * 0.5f;

  if ((d0 + d1) == 0.0f) {
    rows[0].x += PX_EPS_F32;
    rows[1].y += PX_EPS_F32;
    rows[2].z += PX_EPS_F32;
  }
}

struct CoupledIslandRow {
  AvbdVec6 jacobianA;
  AvbdVec6 jacobianB;
  physx::PxU32 bodyA;
  physx::PxU32 bodyB;
  physx::PxReal penalty;
  physx::PxReal force;
};

PX_FORCE_INLINE AvbdVec6 multiplyBlock(const AvbdBlock6x6 &block,
                                       const AvbdVec6 &value) {
  return AvbdVec6(block.linearLinear * value.linear +
                      block.linearAngular * value.angular,
                  block.angularLinear * value.linear +
                      block.angularAngular * value.angular);
}

} // namespace Dy
} // namespace physx

#endif
