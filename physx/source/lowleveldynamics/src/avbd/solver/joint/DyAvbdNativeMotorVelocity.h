// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_NATIVE_MOTOR_VELOCITY_H
#define DY_AVBD_NATIVE_MOTOR_VELOCITY_H

#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

struct AvbdContactConstraint;
struct AvbdD6JointConstraint;
struct AvbdGearJointConstraint;
struct AvbdSolverBody;

struct AvbdJointMotorAdmissionState {
  physx::PxU32 nativeRevoluteMotorGearJointIndex = PX_MAX_U32;
  bool nativeRevoluteMotorVelocityProjectionIsland = false;
  bool nativeRevoluteMotorGearVelocityProjectionIsland = false;
  bool contactCoupledNativeRevoluteMotorVelocityProjectionIsland = false;
};

AvbdJointMotorAdmissionState buildAvbdJointMotorAdmission(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const physx::PxVec3 &gravity, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts);

bool isSingleNativeRevoluteMotorVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, const AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isContactCoupledNativeRevoluteMotorVelocityProjectionSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const physx::PxVec3 &gravity, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts,
    bool &unsupportedTransientContact);

void projectContactCoupledNativeRevoluteMotorVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint &joint, const physx::PxVec3 &gravity,
    physx::PxReal dt);

void projectSingleNativeRevoluteMotorVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint &joint, physx::PxReal dt,
    bool conserveDynamicPairAngularMomentum,
    physx::PxReal expectedAngularMomentumOnAxis,
    bool conserveDynamicPairAngularMomentumVector,
    const physx::PxVec3 &expectedAngularMomentumVector,
    bool conserveDynamicPairLinearMomentum,
    const physx::PxVec3 &expectedLinearMomentum,
    bool conserveDynamicPairSpatialMomentum,
    const physx::PxVec3 &expectedSpatialAngularMomentum,
    bool useSolveStartRelativeVelocity,
    physx::PxReal solveStartRelativeVelocity);

} // namespace Dy
} // namespace physx

#endif
