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

#include "avbd/ogc/DyAvbdOgcTerminal.h"
#include "avbd/solver/post_al/DyAvbdPostAl.h"

namespace physx {
namespace Dy {

// Post-AL terminal OGC phase owner.  This is a frame-local coordinator for
// admission and same-time closure; it does not own persistent pair state or
// advance time.

void AvbdPostAlTerminalState::run(
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    physx::PxReal lengthScale, AvbdSolverStats &stats) {
  const bool refreshNeeded = avbdPrepareTerminalCurrentPoseAdmission(
      terminalSoftExecutionPlan, bodies, numBodies, shellParticles,
      numShellParticles, softBodiesForRecovery, numSoftBodiesForRecovery,
      shellContacts, numShellContacts,
      ogc.sourceBodyMask, ogc.broadphaseBodyMinimum,
      ogc.broadphaseBodyMaximum, lengthScale);

  runTerminalCurrentPoseClosure(
      ogc, refreshNeeded, terminalSoftExecutionPlan, bodies, numBodies,
      shellParticles, numShellParticles, softBodiesForRecovery,
      numSoftBodiesForRecovery, lengthScale, stats);
}

} // namespace Dy
} // namespace physx
