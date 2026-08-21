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

#include "avbd/solver/post_al/DyAvbdPostAl.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/solver/post_al/DyAvbdPostAlContactResponse.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

// Post-AL pose/depenetration phase owner.  This phase records the block-solve
// anchor, applies the existing static and kinematic-shell geometric recovery,
// and materializes the split-policy masks consumed by velocity handoff.

void AvbdPostAlPoseState::run(
    AvbdSolver &solver, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, const physx::PxVec3 &gravity,
    physx::PxReal dt,
    bool hasBodyStaticContact, bool deformableFastImpactIsland,
    bool allowRigidDeepPoseRecoverySplit,
    bool allowRigidFiniteMaterialPoseSplit,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    const physx::PxArray<bool> &touchesKinematicShell,
    AvbdSolverStats &stats) {
  const AvbdSolverConfig &config = solver.getConfig();
  splitRigidDeepPoseRecovery.resize(numBodies);
  splitRigidFiniteMaterialPose.resize(numBodies);
  postBlockPos.resize(numBodies);
  postBlockRot.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    splitRigidDeepPoseRecovery[i] =
        allowRigidDeepPoseRecoverySplit &&
        isRigidDeepBodyStaticRecoverySplitSupported(
            bodies, numBodies, contacts, numContacts, contactMap, i,
            config.lengthScale);
    splitRigidFiniteMaterialPose[i] =
        allowRigidFiniteMaterialPoseSplit &&
        isRigidFiniteBodyStaticMaterialSplitSupported(
            bodies, numBodies, contacts, numContacts, contactMap, i,
            config.lengthScale);
  }

  hasNormalStageMask = config.enableStageOwnershipDiagnostics;
  if (hasNormalStageMask) {
    deformableNormalStageMask.resize(numContacts);
    for (physx::PxU32 c = 0; c < numContacts; ++c)
      deformableNormalStageMask[c] = 0u;
  }
  physx::PxArray<physx::PxU8> *stageMask = normalStageMaskPtr();
  recordBodyStaticNormalAlOwnership(
      bodies, contacts, numContacts, numBodies, config.avbdAlpha,
      &touchesKinematicShell, stageMask, stats);

  // Preserve the original coordinator order: the block-solve anchor is
  // sampled after ownership bookkeeping and immediately before geometric
  // recovery begins.  The bookkeeping call is diagnostic-only, but keeping
  // this order makes the phase boundary mechanically auditable.
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    postBlockPos[i] = bodies[i].position;
    postBlockRot[i] = bodies[i].rotation;
  }

  const bool hasKinematicShellContacts =
      shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0;
  const physx::PxArray<bool> *shellSkipDepen =
      hasKinematicShellContacts ? &touchesKinematicShell : nullptr;
  if (hasBodyStaticContact && contacts && numContacts > 0) {
    PX_PROFILE_ZONE("AVBD.bodyStaticDepenetration", 0);
    physx::PxU32 anyDeform = 0;
    for (physx::PxU32 bi = 0; bi < numBodies && anyDeform == 0; ++bi) {
      if (bodies[bi].invMass > 0.0f &&
          bodyTouchesDeformableAnchor(contacts, numContacts, bi,
                                       contactMap))
        anyDeform = 1;
    }
    const physx::PxU32 depenSweeps =
        deformableFastImpactIsland
            ? 8u
            : (anyDeform != 0 ? (numBodies > 2u ? 10u : 8u) : 6u);
    applyBodyStaticNormalDepenetrationSweeps(
        bodies, numBodies, contacts, numContacts, gravity, dt,
        depenSweeps, shellSkipDepen, stageMask, config.lengthScale, &stats);
  }
  if (hasKinematicShellContacts) {
    PX_PROFILE_ZONE("AVBD.kinematicShellDepenetration", 0);
    applyKinematicOgcNormalDepenetrationSweeps(
        bodies, numBodies, shellParticles, numShellParticles, shellContacts,
        numShellContacts, gravity, dt, 8u, &stats);
  }
}

} // namespace Dy
} // namespace physx
