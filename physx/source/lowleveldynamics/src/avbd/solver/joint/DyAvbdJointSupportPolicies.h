// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_SUPPORT_POLICIES_H
#define DY_AVBD_JOINT_SUPPORT_POLICIES_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

bool areFrictionlessBodyVsStaticContactsSupported(
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 numBodies);

bool areTorqueFreeBodyVsStaticContactsSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts);

bool areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity);

bool isCoupledLinearDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts);

bool isLinearPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isCoupledLinearPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const physx::PxVec3 &gravity, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isAngularAxisVelocityDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isAngularAxisPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isSlerpVelocityDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isSlerpPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, physx::PxU32 numGear,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts);

bool isCoupledAngularPositionDriveIslandSupported(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, physx::PxU32 numSoftParticles,
    physx::PxU32 numSoftBodies, physx::PxU32 numSoftContacts);

} // namespace Dy
} // namespace physx

#endif
