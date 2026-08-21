// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_CURRENT_POSE_H
#define DY_AVBD_OGC_CURRENT_POSE_H

#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

struct AvbdSoftContactGeometry;
struct AvbdOgcRigidBoxGeometry;
struct AvbdSolverBody;

struct AvbdOgcCurrentPairGeometry {
  physx::PxVec3 normal{0.0f};
  // Surface point relative to a movable target's solver-body origin.  World
  // static targets have no movable endpoint and keep this at zero.
  physx::PxVec3 targetOffset{0.0f};
  physx::PxReal signedGap{0.0f};
};

bool getCurrentOgcPairGeometry(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSolverBody *dynamicTarget,
    const physx::PxVec3 &queryPoint,
    AvbdOgcCurrentPairGeometry &result,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr);

} // namespace Dy
} // namespace physx

#endif
