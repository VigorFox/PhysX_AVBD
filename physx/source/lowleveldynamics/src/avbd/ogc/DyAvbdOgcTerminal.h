// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_TERMINAL_H
#define DY_AVBD_OGC_TERMINAL_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

struct AvbdSolverBody;
struct AvbdSoftBody;
struct AvbdSoftParticle;
struct AvbdSoftContact;
struct AvbdSoftIslandExecutionPlan;
struct AvbdRigidBox;
struct AvbdRigidSphere;
struct AvbdRigidCapsule;
struct AvbdRigidConvex;
struct AvbdRigidTriangleSurface;
struct AvbdSoftContactWorkspace;
struct AvbdSolverStats;
struct AvbdTerminalOgcState;
struct AvbdOgcGeometryEpochSidecar;

bool avbdBuildTerminalCurrentPoseContacts(
    const AvbdSoftIslandExecutionPlan *plan, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, const AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies, const physx::PxU8 *sourceBodyMask,
    physx::PxU32 numSourceBodyMask,
    physx::PxArray<AvbdSoftParticle> &proxyParticles,
    physx::PxArray<AvbdSoftBody> &collisionBodies,
    physx::PxArray<AvbdRigidBox> &boxes,
    physx::PxArray<AvbdRigidSphere> &spheres,
    physx::PxArray<AvbdRigidCapsule> &capsules,
    physx::PxArray<AvbdRigidConvex> &convexes,
    physx::PxArray<AvbdRigidTriangleSurface> &triangleSurfaces,
    physx::PxArray<AvbdSoftContact> &contacts,
    AvbdSoftContactWorkspace &contactWorkspace,
    AvbdOgcGeometryEpochSidecar &geometrySidecar);

bool avbdPrepareTerminalCurrentPoseAdmission(
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    physx::PxArray<physx::PxU8> &terminalSourceBodyMask,
    physx::PxArray<physx::PxVec3> &broadphaseBodyMinimum,
    physx::PxArray<physx::PxVec3> &broadphaseBodyMaximum,
    physx::PxReal lengthScale);

void runTerminalCurrentPoseClosure(
    AvbdTerminalOgcState &terminalState,
    bool terminalCurrentPoseRefreshNeeded,
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery, physx::PxReal lengthScale,
    AvbdSolverStats &stats);

} // namespace Dy
} // namespace physx

#endif
