// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/ogc/DyAvbdOgcStaticResponse.h"
#include "avbd/solver/post_al/DyAvbdPostAl.h"
#include "avbd/solver/soft/DyAvbdSoftBodyFinalization.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

// Soft-particle velocity reconstruction and OGC endpoint handoff after the
// rigid velocity/material phase.  This phase owns no position correction.

void finalizePostAlSoftVelocities(
    AvbdSoftParticle *softParticlesForVel,
    physx::PxU32 numSoftParticlesForVel, physx::PxReal invDt,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery, AvbdSoftContact *shellContacts,
    physx::PxU32 numShellContacts, physx::PxReal dt,
	const AvbdSoftIslandExecutionPlan *geometryPlan,
    AvbdOgcPairState *selectionPairStates,
    physx::PxU32 numSelectionPairStates,
    AvbdSoftContact *terminalContacts, physx::PxU32 numTerminalContacts,
    bool terminalCurrentPoseEpochApplied,
    const AvbdTerminalOgcState *terminalOgcState,
    AvbdOgcPairState *terminalPairStates,
    physx::PxU32 numTerminalPairStates,
    physx::PxReal lengthScale, AvbdSolverStats &stats) {
  if (softParticlesForVel && numSoftParticlesForVel > 0) {
    for (physx::PxU32 i = 0; i < numSoftParticlesForVel; ++i) {
      if (softParticlesForVel[i].invMass > 0.0f)
        softParticlesForVel[i].updateVelocityFromPosition(invDt);
    }
  }
  // A fail-closed pose rollback rejects only the unsafe normal advance. Keep
  // the terminal-entry velocity so tangential motion and gravity response do
  // not freeze; the terminal pair owners below remove renewed inward speed.
  if (terminalOgcState && terminalOgcState->failClosed && shellParticles &&
      softBodiesForRecovery &&
      terminalOgcState->acceptedSoftVelocities.size() ==
          numShellParticles) {
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < numSoftBodiesForRecovery; ++bodyIndex) {
      if (bodyIndex >= terminalOgcState->failClosedSoftBodyMask.size() ||
          terminalOgcState->failClosedSoftBodyMask[bodyIndex] == 0u)
        continue;
      const AvbdSoftBody &body = softBodiesForRecovery[bodyIndex];
      const physx::PxU32 start = body.compiled.particleStart;
      const physx::PxU32 count = body.compiled.particleCount;
      if (start > numShellParticles || count > numShellParticles - start)
        continue;
      for (physx::PxU32 local = 0; local < count; ++local) {
        const physx::PxU32 particleIndex = start + local;
        const physx::PxVec3 acceptedVelocity =
            terminalOgcState->acceptedSoftVelocities[particleIndex];
        if (!acceptedVelocity.isFinite())
          continue;
        shellParticles[particleIndex].velocity = acceptedVelocity;
        shellParticles[particleIndex].prevVelocity = acceptedVelocity;
      }
    }
  }
  if (shellParticles && numShellParticles > 0 && shellContacts &&
      numShellContacts > 0 && softBodiesForRecovery &&
      numSoftBodiesForRecovery > 0) {
	const AvbdOgcGeometryEpochView geometryEpoch =
		makeOgcGeometryEpochView(geometryPlan);
    PX_PROFILE_ZONE("AVBD.dynamicSoftRigidTangentVelocity", 0);
    projectDynamicTargetOgcVelocityTangents(
        bodies, numBodies, shellParticles, numShellParticles,
        softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
        numShellContacts, dt);
    PX_PROFILE_ZONE("AVBD.softContactTangentVelocity", 0);
    avbdProjectSoftContactVelocityTangents(
        shellParticles, numShellParticles, softBodiesForRecovery,
        numSoftBodiesForRecovery, shellContacts, numShellContacts, dt,
		nullptr, &geometryEpoch);
  }
  PX_PROFILE_ZONE("AVBD.dynamicSoftRigidInelasticVel", 0);
  clampRecoveredOgcPairNormalVelocities(
      bodies, numBodies, shellParticles, numShellParticles, shellContacts,
      numShellContacts, selectionPairStates, numSelectionPairStates,
      AvbdOgcVelocityContactDomain::eSELECTION,
      AvbdOgcNormalTargetMobility::eDYNAMIC_RIGID, lengthScale, &stats);
  PX_PROFILE_ZONE("AVBD.deformablePairInelasticVel", 0);
  clampRecoveredOgcPairNormalVelocities(
      bodies, numBodies, shellParticles, numShellParticles, shellContacts,
      numShellContacts, selectionPairStates, numSelectionPairStates,
      AvbdOgcVelocityContactDomain::eSELECTION,
      AvbdOgcNormalTargetMobility::eDEFORMABLE_SURFACE, lengthScale, &stats);
  // A dynamic response can load a source which also rests on static geometry;
  // keep the static unilateral owner last without maintaining another mask.
  PX_PROFILE_ZONE("AVBD.worldStaticSoftInelasticVel", 0);
  clampRecoveredOgcPairNormalVelocities(
      bodies, numBodies, shellParticles, numShellParticles, shellContacts,
      numShellContacts, selectionPairStates, numSelectionPairStates,
      AvbdOgcVelocityContactDomain::eSELECTION,
      AvbdOgcNormalTargetMobility::eWORLD_STATIC, lengthScale, &stats);
  // Terminal rows are fresh current-pose contacts and own no warm-start
  // state, so their e=0 response follows position-derived soft velocities.
  if (terminalCurrentPoseEpochApplied && terminalContacts &&
      numTerminalContacts > 0) {
    PX_PROFILE_ZONE("AVBD.terminalCurrentPoseOgcVelocity", 0);
    clampRecoveredOgcPairNormalVelocities(
        bodies, numBodies, shellParticles, numShellParticles,
        terminalContacts, numTerminalContacts, terminalPairStates,
        numTerminalPairStates, AvbdOgcVelocityContactDomain::eTERMINAL,
        AvbdOgcNormalTargetMobility::eDYNAMIC_RIGID, lengthScale, &stats);
    clampRecoveredOgcPairNormalVelocities(
        bodies, numBodies, shellParticles, numShellParticles,
        terminalContacts, numTerminalContacts, terminalPairStates,
        numTerminalPairStates, AvbdOgcVelocityContactDomain::eTERMINAL,
        AvbdOgcNormalTargetMobility::eDEFORMABLE_SURFACE, lengthScale,
        &stats);
    clampRecoveredOgcPairNormalVelocities(
        bodies, numBodies, shellParticles, numShellParticles,
        terminalContacts, numTerminalContacts, terminalPairStates,
        numTerminalPairStates, AvbdOgcVelocityContactDomain::eTERMINAL,
        AvbdOgcNormalTargetMobility::eWORLD_STATIC, lengthScale, &stats);
  }
  avbdProjectWorldFixedPins(
      shellParticles, numShellParticles, softBodiesForRecovery,
      numSoftBodiesForRecovery);
}

} // namespace Dy
} // namespace physx
