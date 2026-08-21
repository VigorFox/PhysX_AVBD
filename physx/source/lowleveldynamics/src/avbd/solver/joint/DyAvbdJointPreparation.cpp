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

#include "avbd/solver/joint/DyAvbdJointPreparation.h"
#include "avbd/contact/DyAvbdContactPrep.h"
#include "avbd/solver/rigid/DyAvbdRigidPhases.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.h"
#include "avbd/solver/soft/DyAvbdSoftWarmstart.h"
#include "common/PxProfileZone.h"

#include <algorithm>

namespace physx {
namespace Dy {

void prepareAvbdJointConstraintPhase(
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const AvbdBodyConstraintMap *contactMap) {
  applyAvbdPenaltyFloor(
      contacts, numContacts, bodies, numBodies, d6Joints, numD6,
      gearJoints, numGear, invDt2);

  initializeAvbdContactC0(
      contacts, numContacts, bodies, numBodies, config);

  if (config.isDeterministic() &&
      (config.determinismFlags & AvbdDeterminismFlags::eSORT_CONSTRAINTS) &&
      numContacts > 1) {
    PX_PROFILE_ZONE("AVBD.sortConstraints", 0);
    std::sort(
        contacts, contacts + numContacts,
        [](const AvbdContactConstraint &a, const AvbdContactConstraint &b) {
          if (a.header.bodyIndexA != b.header.bodyIndexA)
            return a.header.bodyIndexA < b.header.bodyIndexA;
          if (a.header.bodyIndexB != b.header.bodyIndexB)
            return a.header.bodyIndexB < b.header.bodyIndexB;
          return a.header.type < b.header.type;
        });
  }

  initializeAvbdNoContactBodies(
      bodies, numBodies, contacts, numContacts, contactMap);
}

void AvbdSolver::runAvbdJointWarmstartPhase(
    const AvbdSolverConfig &config, physx::PxReal dt,
    physx::PxReal invDt, physx::PxReal invDt2, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, const physx::PxVec3 &gravity,
    const physx::PxArray<bool> &touchesKinematicShell,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    bool hasPreparedSoftPrediction) {
  // Stage 1: prediction. Provider-prepared soft poses remain authoritative;
  // direct callers use the same local fallback inside the phase helper.
  computePrediction(bodies, numBodies, dt, gravity);
  runAvbdSoftPredictionPhase(
      softParticles, numSoftParticles, dt, gravity,
      hasPreparedSoftPrediction);

  // Split world-static ownership before any positional contact iteration.
  // Normal response remains Position-AL owned; tangential Coulomb response is
  // rebuilt from final velocities rather than pinning the deformable body.
  avbdAssignVelocityTangentOwners(
      softContacts, numSoftContacts, softBodies, numSoftBodies,
      softParticles, numSoftParticles);

  PX_PROFILE_ZONE("AVBD.initPositions", 0);
  static const physx::PxReal kShellFastImpactSpeed =
      AvbdConstants::AVBD_SHELL_FAST_IMPACT_SPEED;
  warmstartAvbdRigidBodies(
      bodies, numBodies, touchesKinematicShell, dt, invDt, gravity,
      kShellFastImpactSpeed);
  warmstartAvbdSoftParticles(
      softParticles, numSoftParticles, dt, invDt, gravity);
  warmstartAvbdSoftObjectives(softBodies, numSoftBodies, config);
  warmstartAvbdSoftContacts(
      softContacts, numSoftContacts, softParticles, numSoftParticles,
      bodies, numBodies, invDt2, config);
}

} // namespace Dy
} // namespace physx
