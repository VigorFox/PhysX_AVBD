// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_COUPLED_SYSTEM_H
#define DY_AVBD_JOINT_COUPLED_SYSTEM_H

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/solver/joint/DyAvbdJointCoupledMath.h"

namespace physx {
namespace Dy {

PX_FORCE_INLINE void addScaled(AvbdVec6 &target, const AvbdVec6 &value,
                               physx::PxReal scale) {
  target.linear += value.linear * scale;
  target.angular += value.angular * scale;
}

double dotVectors(const physx::PxArray<AvbdVec6> &a,
                  const physx::PxArray<AvbdVec6> &b);

void addCoupledRow(const CoupledIslandRow &row,
                   physx::PxArray<CoupledIslandRow> &rows,
                   physx::PxArray<AvbdVec6> &gradient,
                   physx::PxArray<AvbdBlock6x6> &preconditioner);

void applyCoupledOperator(
    const physx::PxArray<AvbdBlock6x6> &inertialBlocks,
    const physx::PxArray<CoupledIslandRow> &rows,
    const physx::PxArray<AvbdVec6> &input,
    physx::PxArray<AvbdVec6> &output);

bool addBodyVsStaticContactNormalRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner,
    bool allowFriction);

bool addFrictionlessBodyVsStaticContactRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner);

bool addStrictFrictionalBodyVsStaticContactPositionRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, const AvbdSolverConfig &config,
    physx::PxReal invDt2, physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner);

} // namespace Dy
} // namespace physx

#endif
