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
#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcStaticResponse.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

void AvbdPostAlRecoveryState::run(
    AvbdSolver &solver, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    AvbdSolverStats &stats,
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan) {
  if (shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0 && softBodiesForRecovery &&
      numSoftBodiesForRecovery > 0) {
    const bool hasPairPlan = terminalSoftExecutionPlan &&
        terminalSoftExecutionPlan->hasMixedOgcPairPlan(numShellContacts);
    AvbdOgcPairState *pairStates =
        hasPairPlan ? terminalSoftExecutionPlan->ogcPairStates : nullptr;
    const physx::PxU32 numPairStates =
        hasPairPlan ? terminalSoftExecutionPlan->numOgcPairStates : 0u;
    const physx::PxU32 *pairIndices =
        hasPairPlan ? terminalSoftExecutionPlan->ogcPairIndices : nullptr;
    const physx::PxU32 numPairIndices =
        hasPairPlan ? terminalSoftExecutionPlan->numOgcPairIndices : 0u;
    const AvbdOgcGeometryEpochView geometryEpoch =
        makeOgcGeometryEpochView(terminalSoftExecutionPlan);
    const AvbdOgcGeometryEpochView *geometryEpochView =
        terminalSoftExecutionPlan ? &geometryEpoch : nullptr;
    for (physx::PxU32 recoverySweep = 0; recoverySweep < 8u;
         ++recoverySweep) {
      {
        PX_PROFILE_ZONE("AVBD.worldStaticSoftDepenetration", 0);
        applyWorldStaticSoftNormalDepenetrationSweeps(
            shellParticles, numShellParticles, softBodiesForRecovery,
            numSoftBodiesForRecovery, shellContacts, numShellContacts, 1u,
            &stats,
            terminalSoftExecutionPlan, bodies, numBodies, shellContacts,
            numShellContacts, pairStates, numPairStates, pairIndices,
            numPairIndices, AvbdOgcVelocityContactDomain::eSELECTION,
            geometryEpochView);
      }
      // Preserve the existing dynamic recovery budget (six sweeps) while
      // interleaving each of those sweeps with the eight static sweeps.
      if (recoverySweep < 6u) {
        // The regular dynamic OGC rows have already been consumed by their
        // shared soft-particle / rigid-body Position-AL blocks.  This final
        // projection sees only a true current-pose SDF overlap, never the OGC
        // proximity shell or a swept/CCD candidate.  It remains before
        // postDepenPos so the paired rigid offset is pose-only.
        PX_PROFILE_ZONE("AVBD.dynamicSoftRigidDepenetration", 0);
        applyDynamicSoftRigidNormalDepenetrationSweeps(
            bodies, numBodies, shellParticles, numShellParticles,
            softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
            numShellContacts, 1u,
            solver.getConfig().lengthScale, &stats,
            pairStates, numPairStates, pairIndices, numPairIndices,
            /*softComplianceResponseScale=*/1.0f,
            /*projectToCurrentPoseBoundary=*/false,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->ogcPairContactStarts : nullptr,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->numOgcPairContactStarts : 0u,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->ogcPairContactRefs : nullptr,
            terminalSoftExecutionPlan &&
                    terminalSoftExecutionPlan->hasMixedOgcPairContactPlan(
                        numShellContacts)
                ? terminalSoftExecutionPlan->numOgcPairContactRefs : 0u,
            AvbdOgcVelocityContactDomain::eSELECTION,
            geometryEpochView);
      }
    }
    // Dynamic response can be directed towards world-static geometry.  Make
    // world-static the final owner in this stage so a dynamic projection
    // cannot leave a real static overlap at the end of the step.
    {
      PX_PROFILE_ZONE("AVBD.worldStaticTriangleCoreLocalManifold", 0);
      applyWorldStaticTriangleCoreLocalManifold(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, shellContacts, numShellContacts, 2u,
          solver.getConfig().lengthScale, &stats,
          terminalSoftExecutionPlan, bodies, numBodies,
          shellContacts, numShellContacts, geometryEpochView);
      PX_PROFILE_ZONE("AVBD.worldStaticSoftDepenetration", 0);
      applyWorldStaticSoftNormalDepenetrationSweeps(
          shellParticles, numShellParticles, softBodiesForRecovery,
          numSoftBodiesForRecovery, shellContacts, numShellContacts, 1u,
          &stats,
          terminalSoftExecutionPlan, bodies, numBodies, shellContacts,
          numShellContacts, pairStates, numPairStates, pairIndices,
          numPairIndices, AvbdOgcVelocityContactDomain::eSELECTION,
          geometryEpochView);
    }
  }
}

} // namespace Dy
} // namespace physx
