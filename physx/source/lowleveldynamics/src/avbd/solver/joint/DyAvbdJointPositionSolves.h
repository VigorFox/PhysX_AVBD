// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_POSITION_SOLVES_H
#define DY_AVBD_JOINT_POSITION_SOLVES_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool solveCoupledAngularPositionDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdSolverConfig &config);

bool solveCoupledLinearPositionDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const physx::PxVec3 &gravity,
    const AvbdSolverConfig &config);

} // namespace Dy
} // namespace physx

#endif
