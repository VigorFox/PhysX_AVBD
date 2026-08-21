// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_DYNAMIC_RESPONSE_H
#define DY_AVBD_OGC_DYNAMIC_RESPONSE_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "avbd/ogc/DyAvbdOgcPair.h"

namespace physx {
namespace Dy {

struct AvbdOgcPairState;
struct AvbdOgcGeometryEpochView;
struct AvbdSoftBody;
struct AvbdSoftContact;
struct AvbdSoftParticle;
struct AvbdSolverBody;
struct AvbdSolverStats;

physx::PxU32 applyDynamicSoftRigidTriangleCoreLocalManifold(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, physx::PxReal lengthScale,
    AvbdSolverStats *stats,
    AvbdOgcPairState *ogcPairStates = nullptr,
    physx::PxU32 numOgcPairStates = 0,
    const physx::PxU32 *ogcPairIndices = nullptr,
    physx::PxU32 numOgcPairIndices = 0,
    const physx::PxU32 *ogcPairContactStarts = nullptr,
    physx::PxU32 numOgcPairContactStarts = 0,
    const physx::PxU32 *ogcPairContactRefs = nullptr,
    physx::PxU32 numOgcPairContactRefs = 0,
    AvbdOgcVelocityContactDomain contactDomain =
        AvbdOgcVelocityContactDomain::eNONE,
    const AvbdOgcGeometryEpochView *geometryEpoch = nullptr);

physx::PxU32 applyDynamicSoftRigidNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps,
    physx::PxReal lengthScale, AvbdSolverStats *stats,
    AvbdOgcPairState *ogcPairStates = nullptr,
    physx::PxU32 numOgcPairStates = 0,
    const physx::PxU32 *ogcPairIndices = nullptr,
    physx::PxU32 numOgcPairIndices = 0,
    physx::PxReal softComplianceResponseScale = 1.0f,
    bool projectToCurrentPoseBoundary = false,
    const physx::PxU32 *ogcPairContactStarts = nullptr,
    physx::PxU32 numOgcPairContactStarts = 0,
    const physx::PxU32 *ogcPairContactRefs = nullptr,
    physx::PxU32 numOgcPairContactRefs = 0,
    AvbdOgcVelocityContactDomain contactDomain =
        AvbdOgcVelocityContactDomain::eNONE,
    const AvbdOgcGeometryEpochView *geometryEpoch = nullptr);

void clampAdmittedMixedOgcPairNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *contactPairIndices,
    physx::PxU32 numContactPairIndices,
    const physx::PxU32 *pairContactStarts,
    physx::PxU32 numPairContactStarts,
    const physx::PxU32 *pairContactRefs,
    physx::PxU32 numPairContactRefs,
    physx::PxReal lengthScale, AvbdSolverStats *stats);

void projectDynamicTargetOgcVelocityTangents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt);

} // namespace Dy
} // namespace physx

#endif
