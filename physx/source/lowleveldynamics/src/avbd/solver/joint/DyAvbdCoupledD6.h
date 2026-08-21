// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_COUPLED_D6_H
#define DY_AVBD_COUPLED_D6_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool isCoupledFixedD6IslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isCoupledSphericalConeIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool solveCoupledFixedD6Island(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &joint, physx::PxReal invDt2);

bool solveCoupledSphericalConeIsland(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &joint, physx::PxReal invDt2);

bool projectCoupledFixedD6Velocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint);

} // namespace Dy
} // namespace physx

#endif
