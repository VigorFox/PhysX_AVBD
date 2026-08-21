// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_VELOCITY_POLICIES_H
#define DY_AVBD_JOINT_VELOCITY_POLICIES_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool isPassiveCenteredGearVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, const AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts);

bool isNativeRevoluteMotorGearVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, const AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts,
    physx::PxU32 &motorJointIndex);

void projectPassiveCenteredGearVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdGearJointConstraint &gear,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart);

void projectNativeRevoluteMotorGearVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &motor,
    const AvbdGearJointConstraint &gear, physx::PxReal dt);

bool isSinglePassiveGenericHard1DVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts, physx::PxU32 &genericIndex);

void projectSinglePassiveGenericHard1DVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &generic,
    const physx::PxArray<physx::PxVec3> &linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart);

void projectArticulationMimicVelocity1D(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint &mimic);

bool findCoupledSpatialTendonRows(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts,
    physx::PxArray<physx::PxU32> &rowIndices);

} // namespace Dy
} // namespace physx

#endif
