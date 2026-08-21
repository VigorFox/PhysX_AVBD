// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_RESPONSE_H
#define DY_AVBD_OGC_RESPONSE_H

#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx {
namespace Dy {

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
    defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
#define DY_AVBD_OGC_RESPONSE_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
#define DY_AVBD_OGC_RESPONSE_API PX_UNIX_EXPORT
#else
#define DY_AVBD_OGC_RESPONSE_API
#endif

struct AvbdSoftContactGeometry;
struct AvbdSoftContact;
struct AvbdSoftBody;
struct AvbdSolverBody;
struct AvbdSolverStats;

enum class AvbdOgcNormalTargetMobility : physx::PxU8 {
  eWORLD_STATIC,
  eDYNAMIC_RIGID,
  eDEFORMABLE_SURFACE
};

enum class AvbdOgcNormalSourceMobility : physx::PxU8 {
  eDYNAMIC_SOFT,
  eKINEMATIC_SOFT
};

// Stateless normal-row IR shared by every OGC endpoint mobility.  Contact
// detection owns geometry, pair state owns lifetime, and this structure only
// compiles the current response needed by one projector invocation.
struct AvbdOgcNormalResponse {
  AvbdOgcCurrentPairGeometry current;
  AvbdOgcNormalSourceMobility sourceMobility{
      AvbdOgcNormalSourceMobility::eDYNAMIC_SOFT};
  physx::PxVec3 normal{0.0f};
  physx::PxReal constraintValue{0.0f};
  physx::PxVec3 queryPoint{0.0f};
  physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
  physx::PxReal particleWeights[AVBD_CONTACT_MAX_PARTICLES];
  physx::PxU32 particleCount{0u};
  physx::PxReal softResponse{0.0f};
  physx::PxVec3 targetLinearJacobian{0.0f};
  physx::PxVec3 targetAngularJacobian{0.0f};
  physx::PxVec3 targetLinearDeltaPerLambda{0.0f};
  physx::PxVec3 targetAngularDeltaPerLambda{0.0f};
  physx::PxReal targetResponse{0.0f};
  physx::PxReal effectiveResponse{0.0f};
};

// Position-side transaction for the deformable endpoint of one compiled
// normal response.  Target mobility is deliberately absent: world-static and
// movable-rigid projectors must share exactly one support displacement,
// positive-J admission and velocity-anchor commit protocol.
struct AvbdOgcSoftPositionCandidate {
  physx::PxVec3 particleDeltas[AVBD_CONTACT_MAX_PARTICLES];
};

// Stateless tangent-row IR shared by world-static, dynamic-rigid and
// prescribed-soft/dynamic-rigid OGC contacts.  The normal compiler owns
// endpoint mobility and support identity; this layer only adds the tangent
// Schur block needed by the end-step velocity transaction.
struct AvbdOgcTangentResponse {
  AvbdOgcNormalResponse normalResponse;
  physx::PxVec3 tangents[2];
  physx::PxVec3 targetLinearDeltaPerImpulse[2];
  physx::PxVec3 targetAngularDeltaPerImpulse[2];
  physx::PxReal response00{0.0f};
  physx::PxReal response01{0.0f};
  physx::PxReal response11{0.0f};
  physx::PxReal determinant{0.0f};
};

DY_AVBD_OGC_RESPONSE_API bool compileCurrentOgcNormalResponse(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSolverBody *dynamicTarget,
    physx::PxReal softResponseScale,
    AvbdOgcNormalResponse &response,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr);

DY_AVBD_OGC_RESPONSE_API bool getOgcNormalResponseQueryVelocity(
    const AvbdOgcNormalResponse &response,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    physx::PxVec3 &queryVelocity);

DY_AVBD_OGC_RESPONSE_API bool compileCurrentOgcTangentResponse(
    const AvbdSoftContactGeometry &geometry,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSolverBody *dynamicTarget,
    AvbdOgcTangentResponse &response,
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr);

DY_AVBD_OGC_RESPONSE_API bool applyOgcTangentVelocityResponse(
    const AvbdOgcTangentResponse &response,
    AvbdSoftContact &contact,
    AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    AvbdSolverBody *dynamicTarget,
    physx::PxReal dt);

DY_AVBD_OGC_RESPONSE_API bool buildOgcSoftPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    physx::PxReal softResponseScale,
    physx::PxReal lambda,
    AvbdOgcSoftPositionCandidate &candidate);

DY_AVBD_OGC_RESPONSE_API bool buildOgcDeformablePairPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    const AvbdSoftBody &targetBody,
    physx::PxReal lambda,
    AvbdOgcSoftPositionCandidate &candidate);

DY_AVBD_OGC_RESPONSE_API bool admitOgcSoftPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    physx::PxReal alpha,
    physx::PxReal minimumDeterminant);

DY_AVBD_OGC_RESPONSE_API bool admitOgcDeformablePairPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    const AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    const AvbdSoftBody &sourceBody,
    const AvbdSoftBody &targetBody,
    physx::PxReal alpha,
    physx::PxReal minimumDeterminant);

DY_AVBD_OGC_RESPONSE_API bool evaluateOgcSoftPositionCandidateQueryPoint(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    physx::PxReal alpha,
    physx::PxVec3 &queryPoint);

DY_AVBD_OGC_RESPONSE_API bool buildOgcRigidPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdSolverBody &currentBody,
    physx::PxReal lambda,
    physx::PxReal alpha,
    AvbdSolverBody &candidateBody);

DY_AVBD_OGC_RESPONSE_API bool finalizeOgcRigidPositionCandidate(
    const AvbdSolverBody &currentBody,
    AvbdSolverBody &candidateBody);

DY_AVBD_OGC_RESPONSE_API void commitOgcRigidPositionCandidate(
    const AvbdSolverBody &candidateBody,
    AvbdSolverBody &body);

DY_AVBD_OGC_RESPONSE_API bool evaluateCurrentOgcNormalConstraint(
    const AvbdSoftContactGeometry &geometry,
    const AvbdOgcNormalResponse &response,
    const AvbdSolverBody *dynamicTarget,
    const physx::PxVec3 &queryPoint,
    physx::PxReal &constraintValue);

DY_AVBD_OGC_RESPONSE_API void commitOgcSoftPositionCandidate(
    const AvbdOgcNormalResponse &response,
    const AvbdOgcSoftPositionCandidate &candidate,
    AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles,
    physx::PxReal alpha);

DY_AVBD_OGC_RESPONSE_API void clampRecoveredOgcPairNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    AvbdOgcVelocityContactDomain contactDomain,
    AvbdOgcNormalTargetMobility targetMobility,
    physx::PxReal lengthScale, AvbdSolverStats *stats);

physx::PxU32 applyDeformableOgcNormalDepenetrationSweeps(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 sweeps, AvbdOgcPairState *pairStates,
    physx::PxU32 numPairStates, const physx::PxU32 *pairIndices,
    physx::PxU32 numPairIndices,
    AvbdOgcVelocityContactDomain contactDomain,
    AvbdSolverStats *stats);

DY_AVBD_OGC_RESPONSE_API void applyKinematicOgcNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt,
    physx::PxU32 sweeps, AvbdSolverStats *stats);

DY_AVBD_OGC_RESPONSE_API void clampKinematicOgcInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt, AvbdSolverStats *stats);

#undef DY_AVBD_OGC_RESPONSE_API

} // namespace Dy
} // namespace physx

#endif
