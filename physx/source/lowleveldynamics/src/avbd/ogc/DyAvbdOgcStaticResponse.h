// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_STATIC_RESPONSE_H
#define DY_AVBD_OGC_STATIC_RESPONSE_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "avbd/ogc/DyAvbdOgcPair.h"

namespace physx {
namespace Dy {

struct AvbdSoftBody;
struct AvbdSoftContact;
struct AvbdSoftIslandExecutionPlan;
struct AvbdOgcPairState;
struct AvbdOgcGeometryEpochView;
struct AvbdSoftParticle;
struct AvbdSolverBody;
struct AvbdSolverStats;

physx::PxU32 applyWorldStaticSoftNormalDepenetrationSweeps(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, AvbdSolverStats *stats,
    const AvbdSoftIslandExecutionPlan *ogcExecutionPlan = nullptr,
    AvbdSolverBody *ogcRigidBodies = nullptr,
    physx::PxU32 numOgcRigidBodies = 0,
    const AvbdSoftContact *ogcContacts = nullptr,
    physx::PxU32 numOgcContacts = 0,
    AvbdOgcPairState *pairStates = nullptr,
    physx::PxU32 numPairStates = 0,
    const physx::PxU32 *contactPairIndices = nullptr,
    physx::PxU32 numContactPairIndices = 0,
    AvbdOgcVelocityContactDomain contactDomain =
        AvbdOgcVelocityContactDomain::eNONE,
    const AvbdOgcGeometryEpochView *geometryEpoch = nullptr);

physx::PxU32 applyWorldStaticTriangleCoreLocalManifold(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, physx::PxReal lengthScale,
    AvbdSolverStats *stats,
    const AvbdSoftIslandExecutionPlan *ogcExecutionPlan = nullptr,
    AvbdSolverBody *ogcRigidBodies = nullptr,
    physx::PxU32 numOgcRigidBodies = 0,
    const AvbdSoftContact *ogcContacts = nullptr,
    physx::PxU32 numOgcContacts = 0,
    const AvbdOgcGeometryEpochView *geometryEpoch = nullptr);

} // namespace Dy
} // namespace physx

#endif
