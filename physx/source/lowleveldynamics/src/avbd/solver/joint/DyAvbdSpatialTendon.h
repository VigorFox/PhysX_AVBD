// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_SPATIAL_TENDON_H
#define DY_AVBD_SPATIAL_TENDON_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool solveCoupledSpatialTendonRow(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint, physx::PxReal dt,
    physx::PxReal invDt2);

} // namespace Dy
} // namespace physx

#endif
