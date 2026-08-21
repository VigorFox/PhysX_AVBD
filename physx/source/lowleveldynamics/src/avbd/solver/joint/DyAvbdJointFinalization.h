// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_FINALIZATION_H
#define DY_AVBD_JOINT_FINALIZATION_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

struct AvbdJointContactPhaseState;
struct AvbdJointMotorAdmissionState;
struct AvbdJointOgcAdmissionState;

struct AvbdJointVelocityPhaseState {
  bool conserveNativeRevoluteMotorAngularMomentum;
  physx::PxReal nativeRevoluteMotorExpectedAngularMomentum;
  bool conserveNativeRevoluteMotorAngularMomentumVector;
  physx::PxVec3 nativeRevoluteMotorExpectedAngularMomentumVector;
  bool conserveNativeRevoluteMotorLinearMomentum;
  physx::PxVec3 nativeRevoluteMotorExpectedLinearMomentum;
  bool conserveNativeRevoluteMotorSpatialMomentum;
  physx::PxVec3 nativeRevoluteMotorExpectedSpatialAngularMomentum;
  bool useNativeRevoluteMotorSolveStartRelativeVelocity;
  physx::PxReal nativeRevoluteMotorSolveStartRelativeVelocity;
  physx::PxU32 passiveGenericHard1DIndex;
  bool passiveGenericHard1DVelocityProjectionIsland;
  physx::PxVec3 coupledExpectedMomentum;
  physx::PxVec3 coupledExpectedAngularMomentum;
  bool conserveCoupledLinearPositionSupportAxisMomentum;
  physx::PxVec3 coupledLinearPositionSupportAxis;
  physx::PxReal coupledExpectedLinearPositionSupportAxisAngularMomentum;
};

void initializeAvbdJointVelocityPhaseState(
    AvbdJointVelocityPhaseState &state, const AvbdSolverConfig &config,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, const physx::PxVec3 &gravity, physx::PxReal dt,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts,
    bool nativeRevoluteMotorVelocityProjectionIsland,
    bool coupledLinearDriveIsland,
    bool coupledLinearPositionDriveIsland,
    bool coupledLinearPositionDriveFrictionPositionOwnerIsland,
    bool coupledAngularPositionDriveIsland,
    bool coupledSphericalConeIsland);

struct AvbdJointPostAlPhaseState {
  physx::PxArray<bool> positionOwnedAngularBodies;

  void run(
      AvbdSolver &solver, physx::PxReal dt, physx::PxReal invDt,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      const AvbdBodyConstraintMap *contactMap,
      const physx::PxVec3 &gravity,
      const AvbdJointContactPhaseState &contactPhase,
      const AvbdJointVelocityPhaseState &velocityPhase,
      const AvbdJointMotorAdmissionState &motorAdmission,
      AvbdJointOgcAdmissionState &ogcAdmission,
      bool passiveCenteredGearVelocityProjectionIsland,
      bool coupledFixedD6Island, bool coupledLinearDriveIsland,
      bool coupledLinearPositionDriveIsland,
      bool coupledLinearPositionDriveFrictionPositionOwnerIsland,
      bool coupledAngularPositionDriveIsland,
      bool coupledSphericalConeIsland, bool hasJointConstraints,
      bool skipBodyStaticFriction, bool applyVelocityDamping,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const AvbdSoftIslandExecutionPlan *softExecutionPlan,
      bool useProvidedSoftExecutionPlan,
      FeatherstoneArticulation *const *articulationForBody,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      AvbdGearJointConstraint *gearJoints,
      const physx::PxArray<physx::PxVec3> &genericLinearVelAtSolveStart,
      const physx::PxArray<physx::PxVec3> &genericAngularVelAtSolveStart,
      AvbdSolverStats &stats);
};

} // namespace Dy
} // namespace physx

#endif
