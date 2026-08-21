// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_ITERATION_H
#define DY_AVBD_JOINT_ITERATION_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

struct AvbdJointIterationPhaseInput {
  const AvbdSolverConfig &config;
  physx::PxReal dt;
  physx::PxReal invDt2;
  AvbdSolverBody *bodies;
  physx::PxU32 numBodies;
  AvbdContactConstraint *contacts;
  physx::PxU32 numContacts;
  AvbdD6JointConstraint *d6Joints;
  physx::PxU32 numD6;
  AvbdGearJointConstraint *gearJoints;
  physx::PxU32 numGear;
  const physx::PxVec3 &gravity;
  const AvbdBodyConstraintMap *contactMap;
  const AvbdBodyConstraintMap *d6Map;
  const AvbdBodyConstraintMap *gearMap;
  physx::PxU32 iterationOverride;
  bool hasCompleteSoftSelection;
  AvbdSoftParticle *softParticles;
  physx::PxU32 numSoftParticles;
  AvbdSoftBody *softBodies;
  physx::PxU32 numSoftBodies;
  AvbdSoftContact *softContacts;
  physx::PxU32 numSoftContacts;
  const physx::PxU32 *softParticleBodyIndices;
  const physx::PxU32 *softContactStarts;
  const AvbdSoftContactParticleRef *softContactRefs;
  const physx::PxU32 *rigidTargetContactStarts;
  const physx::PxU32 *rigidTargetContactRefs;
  AvbdOgcPairTrustRegionContext *mixedOgcPairContext;
  const AvbdSoftIslandExecutionPlan *softExecutionPlan;
  FeatherstoneArticulation *const *articulationForBody;
  const physx::PxU32 *linkIndexForBody;
  bool slerpVelocityDriveIsland;
  bool &coupledFixedD6Island;
  bool &coupledSphericalConeIsland;
  bool &coupledLinearPositionDriveIsland;
  bool &coupledLinearPositionDriveFrictionPositionOwnerIsland;
  bool &coupledLinearDriveIsland;
  bool &coupledAngularPositionDriveIsland;
  bool &coupledSpatialTendonIsland;
  const physx::PxArray<physx::PxU32> &coupledSpatialTendonRowIndices;
  AvbdSolverStats &stats;
};

} // namespace Dy
} // namespace physx

#endif
