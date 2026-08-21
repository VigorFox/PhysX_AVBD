// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_LINEAR_DRIVE_SOLVE_H
#define DY_AVBD_LINEAR_DRIVE_SOLVE_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool computeTwoBodySupportAxisAngularMomentum(
    const AvbdSolverBody &bodyA, const AvbdSolverBody &bodyB,
    const physx::PxVec3 &supportAxis,
    physx::PxReal linearScale, physx::PxReal angularScale,
    physx::PxReal &axisAngularMomentum);

bool restoreTwoBodySupportAxisAngularMomentum(
    AvbdSolverBody &bodyA, AvbdSolverBody &bodyB,
    const physx::PxVec3 &supportAxis,
    physx::PxReal expectedAxisAngularMomentum);

bool solveCoupledLinearDriveIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdSolverConfig &config);

} // namespace Dy
} // namespace physx

#endif
