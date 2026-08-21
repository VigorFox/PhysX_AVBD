// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_TRUST_REGION_H
#define DY_AVBD_OGC_TRUST_REGION_H

#include "avbd/ogc/DyAvbdOgcPair.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

struct AvbdSoftContact;
struct AvbdOgcGeometryEpochView;
struct AvbdSoftContactParticleRef;
struct AvbdSoftIslandExecutionPlan;
struct AvbdSoftParticle;
struct AvbdSolverBody;

struct AvbdOgcPairTrustRegionContext {
  AvbdOgcPairState *pairStates;
  physx::PxU32 numPairStates;
  const physx::PxU32 *contactPairIndices;
  physx::PxU32 numContactPairIndices;
  const physx::PxU32 *triangleCoreSafetyStarts;
  physx::PxU32 numTriangleCoreSafetyStarts;
  const AvbdSoftContactParticleRef *triangleCoreSafetyRefs;
  physx::PxU32 numTriangleCoreSafetyRefs;

  AvbdOgcPairTrustRegionContext()
      : pairStates(nullptr), numPairStates(0), contactPairIndices(nullptr),
        numContactPairIndices(0), triangleCoreSafetyStarts(nullptr),
        numTriangleCoreSafetyStarts(0), triangleCoreSafetyRefs(nullptr),
        numTriangleCoreSafetyRefs(0) {}

  PX_FORCE_INLINE bool isComplete(physx::PxU32 numContacts) const {
    return pairStates && numPairStates > 0 && contactPairIndices &&
        numContactPairIndices == numContacts;
  }

  PX_FORCE_INLINE bool hasTriangleCoreSafetyPlan(
      physx::PxU32 numParticles) const {
    return triangleCoreSafetyStarts &&
        numTriangleCoreSafetyStarts == numParticles + 1 &&
        (numTriangleCoreSafetyRefs == 0 || triangleCoreSafetyRefs) &&
        triangleCoreSafetyStarts[0] == 0 &&
        triangleCoreSafetyStarts[numParticles] == numTriangleCoreSafetyRefs;
  }

  PX_FORCE_INLINE bool publishTriangleCoreSafetyPlan(
      const physx::PxU32 *starts, physx::PxU32 numStarts,
      const AvbdSoftContactParticleRef *refs, physx::PxU32 numRefs,
      physx::PxU32 numParticles) {
    triangleCoreSafetyStarts = nullptr;
    numTriangleCoreSafetyStarts = 0u;
    triangleCoreSafetyRefs = nullptr;
    numTriangleCoreSafetyRefs = 0u;
    if (!starts || numStarts != numParticles + 1u ||
        (numRefs != 0u && !refs) || starts[0] != 0u ||
        starts[numParticles] != numRefs)
      return false;

    triangleCoreSafetyStarts = starts;
    numTriangleCoreSafetyStarts = numStarts;
    triangleCoreSafetyRefs = refs;
    numTriangleCoreSafetyRefs = numRefs;
    return true;
  }
};

physx::PxReal limitSoftParticleOgcCandidate(
    const AvbdOgcPairTrustRegionContext *context,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftContactParticleRef *contactRefs,
    physx::PxU32 contactRefBegin, physx::PxU32 contactRefEnd,
    physx::PxU32 particleIndex,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSolverBody *rigidBodies, physx::PxU32 numRigidBodies,
    const physx::PxVec3 &candidateDisplacement);

physx::PxReal limitPostAlSoftParticleOgcCandidate(
    const AvbdSoftIslandExecutionPlan *plan,
    const AvbdSoftContact *ogcContacts, physx::PxU32 numOgcContacts,
    AvbdSolverBody *rigidBodies, physx::PxU32 numRigidBodies,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 particleIndex,
    const physx::PxVec3 &candidateDisplacement);

physx::PxReal limitPostAlRigidOgcCandidate(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxU32 targetBodyIndex,
    physx::PxU32 currentSourceBodyIndex,
    physx::PxU64 currentPrimitiveKey,
    const AvbdSolverBody &currentBody,
    const AvbdSolverBody &candidateBody,
    const AvbdOgcGeometryEpochView *geometryEpoch = nullptr);

} // namespace Dy
} // namespace physx

#endif
