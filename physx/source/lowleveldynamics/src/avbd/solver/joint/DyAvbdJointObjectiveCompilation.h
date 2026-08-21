// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_OBJECTIVE_COMPILATION_H
#define DY_AVBD_JOINT_OBJECTIVE_COMPILATION_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

struct AvbdJointObjectiveCompilationState {
  bool coupledLinearPositionDriveIsland = false;
  bool coupledLinearPositionDriveFrictionPositionOwnerIsland = false;
  bool slerpVelocityDriveIsland = false;
  bool coupledAngularPositionDriveIsland = false;
  bool coupledLinearDriveIsland = false;
  bool coupledFixedD6Island = false;
  bool coupledSphericalConeIsland = false;
  bool coupledSpatialTendonIsland = false;
  physx::PxArray<physx::PxU32> coupledSpatialTendonRowIndices;
};

AvbdJointObjectiveCompilationState compileAvbdJointObjectives(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, const physx::PxVec3 &gravity,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

} // namespace Dy
} // namespace physx

#endif
