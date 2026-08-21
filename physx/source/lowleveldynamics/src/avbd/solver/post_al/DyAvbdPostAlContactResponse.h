// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_POST_AL_CONTACT_RESPONSE_H
#define DY_AVBD_POST_AL_CONTACT_RESPONSE_H

#include "foundation/PxArray.h"
#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

struct AvbdContactConstraint;
struct AvbdSoftContact;
struct AvbdSoftParticle;
struct AvbdSolverBody;
struct AvbdSolverStats;

// Per-island storage for the body-static friction phase.  The response kernel
// overwrites active ranges every solve while the owning post-AL workspace
// retains capacity across frames.
struct AvbdBodyStaticFrictionWorkspace {
  physx::PxArray<physx::PxU32> dominantDeformable;
  physx::PxArray<physx::PxU32> bodyDeformRawCount;
  physx::PxArray<physx::PxU32> contactIndices;
  physx::PxArray<physx::PxU32> bodyContactCount;
  physx::PxArray<physx::PxReal> bodyContactNormalSum;
  physx::PxArray<physx::PxVec3> linearVelocity;
  physx::PxArray<physx::PxVec3> angularVelocity;
  physx::PxArray<physx::PxVec3> initialLinearVelocity;
  physx::PxArray<physx::PxVec3> initialAngularVelocity;
  physx::PxArray<bool> touched;
  physx::PxArray<physx::PxReal> bodySpeed;
};

void applyBodyStaticNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
    const physx::PxArray<bool> *skipDepenForBodies,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    physx::PxReal configLengthScale, AvbdSolverStats *stats);

void applyBodyStaticFrictionSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
    const physx::PxArray<physx::PxVec3> *velSeedPos,
    const physx::PxArray<physx::PxQuat> *velSeedRot,
    const physx::PxArray<bool> *skipForBodies,
    physx::PxReal configLengthScale,
    AvbdBodyStaticFrictionWorkspace &workspace,
    AvbdSolverStats *stats);

} // namespace Dy
} // namespace physx

#endif
