// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"

#include "avbd/solver/rigid/DyAvbdSolverBody.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPolicy.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.h"

namespace physx {
namespace Dy {

void projectDynamicTargetOgcVelocityTangents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt) {
  if (!avbdUseVelocityTangentOwner() || !bodies || !softParticles ||
      !softBodies || !softContacts || !physx::PxIsFinite(dt) || dt <= 0.0f)
    return;

  // Prescribed soft shells have no source inverse-mass response.  Preserve
  // the former one-corner-per-rigid admission without preserving its
  // position-level friction implementation: select the deepest current row,
  // then consume it through the same velocity transaction as dynamic soft.
  physx::PxArray<physx::PxU32> dominantKinematicContact(numBodies);
  physx::PxArray<physx::PxReal> dominantConstraint(numBodies);
  for (physx::PxU32 bodyIndex = 0u; bodyIndex < numBodies; ++bodyIndex) {
    dominantKinematicContact[bodyIndex] = PX_MAX_U32;
    dominantConstraint[bodyIndex] = PX_MAX_F32;
  }
  for (physx::PxU32 contactIndex = 0u; contactIndex < numSoftContacts;
       ++contactIndex) {
    const AvbdSoftContact &contact = softContacts[contactIndex];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    if (geometry.tangentOwner != AvbdSoftContactTangentOwner::eVELOCITY ||
        !geometry.hasRigidBodyTarget() || geometry.targetIndex >= numBodies ||
        !avbdCanUseVelocityTangentOwner(
            geometry, softBodies, numSoftBodies, softParticles,
            numSoftParticles))
      continue;
    AvbdOgcTangentResponse response;
    if (!compileCurrentOgcTangentResponse(
            geometry, softParticles, numSoftParticles,
            &bodies[geometry.targetIndex], response) ||
        response.normalResponse.sourceMobility !=
            AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT)
      continue;
    const physx::PxU32 bodyIndex = geometry.targetIndex;
    if (response.normalResponse.constraintValue <
        dominantConstraint[bodyIndex]) {
      dominantConstraint[bodyIndex] =
          response.normalResponse.constraintValue;
      dominantKinematicContact[bodyIndex] = contactIndex;
    }
  }

  for (physx::PxU32 contactIndex = 0u; contactIndex < numSoftContacts;
       ++contactIndex) {
    AvbdSoftContact &contact = softContacts[contactIndex];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    if (geometry.tangentOwner != AvbdSoftContactTangentOwner::eVELOCITY ||
        !geometry.hasRigidBodyTarget() || geometry.targetIndex >= numBodies ||
        !avbdCanUseVelocityTangentOwner(
            geometry, softBodies, numSoftBodies, softParticles,
            numSoftParticles))
      continue;
    AvbdOgcTangentResponse response;
    if (!compileCurrentOgcTangentResponse(
            geometry, softParticles, numSoftParticles,
            &bodies[geometry.targetIndex], response))
      continue;
    if (response.normalResponse.sourceMobility ==
            AvbdOgcNormalSourceMobility::eKINEMATIC_SOFT &&
        dominantKinematicContact[geometry.targetIndex] != contactIndex)
      continue;
    applyOgcTangentVelocityResponse(
        response, contact, softParticles, numSoftParticles,
        &bodies[geometry.targetIndex], dt);
  }
}

} // namespace Dy
} // namespace physx
