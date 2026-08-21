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

#include "avbd/solver/soft/DyAvbdSoftWarmstart.h"

namespace physx {
namespace Dy {

void warmstartAvbdSoftParticles(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal dt, physx::PxReal invDt,
    const physx::PxVec3 &gravity) {
  const physx::PxReal gravMag = gravity.magnitude();
  const physx::PxVec3 gravDir =
      (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);
  for (physx::PxU32 particleIndex = 0;
       particleIndex < numSoftParticles; ++particleIndex) {
    AvbdSoftParticle &particle = softParticles[particleIndex];
    if (particle.invMass <= 0.0f)
      continue;
    const physx::PxVec3 accel =
        (particle.velocity - particle.prevVelocity) * invDt;
    const physx::PxReal accelWeight =
        (gravMag > 1e-6f)
            ? physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f)
            : 0.0f;
    particle.position += particle.velocity * dt +
                         gravity * (accelWeight * dt * dt);
  }
}

void warmstartAvbdSoftObjectives(
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSolverConfig &config) {
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex) {
    AvbdSoftBody &softBody = softBodies[bodyIndex];
    PX_ASSERT(softBody.runtime.isObjectiveProgramCurrent(
        softBody.compiled.particleStart, softBody.compiled.particleCount));
    for (physx::PxU32 objectiveIndex = 0;
         objectiveIndex < softBody.runtime.compiledObjectives.size();
         ++objectiveIndex) {
      const AvbdCompiledSoftObjective &objective =
          softBody.runtime.compiledObjectives[objectiveIndex];
      switch (objective.owner) {
      case AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL:
      case AvbdSoftObjectiveOwner::eARTICULATION_ATTACHMENT_POSITION_AL:
      case AvbdSoftObjectiveOwner::eSOFT_PAIR_ATTACHMENT_POSITION_AL:
        avbdWarmstartAttachmentState(
            softBody.runtime.attachments[objective.runtimeStateIndex],
            config.avbdAlpha, config.avbdGamma, 1e3f);
        break;
      case AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL:
      case AvbdSoftObjectiveOwner::eKINEMATIC_ATTACHMENT_POSITION_AL:
        avbdWarmstartPinState(
            softBody.runtime.pins[objective.runtimeStateIndex],
            config.avbdAlpha, config.avbdGamma, 1e3f);
        break;
      case AvbdSoftObjectiveOwner::eDEFORMABLE_KINEMATIC_POSITION_AL:
        avbdWarmstartPinState(
            softBody.runtime.pins[objective.runtimeStateIndex],
            1.0f, 1.0f, 1.0e8f);
        break;
      default:
        PX_ASSERT(false);
        break;
      }
    }
  }
}

void warmstartAvbdSoftContacts(
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxReal invDt2, const AvbdSolverConfig &config) {
  for (physx::PxU32 contactIndex = 0;
       contactIndex < numSoftContacts; ++contactIndex) {
    AvbdSoftContact &contact = softContacts[contactIndex];
    const AvbdSoftContactGeometry &geometry = contact.geometry;
    AvbdSoftContactAugmentedState &state = contact.state;
    if (avbdIsSoftContactQueryFullyKinematic(
            geometry, softParticles, numSoftParticles)) {
      state.alLambda *= config.avbdAlpha * config.avbdGamma;
      state.k = physx::PxMax(
          1000.0f, physx::PxMin(state.ke, state.k * config.avbdGamma));
      for (int tangent = 0; tangent < 2; ++tangent) {
        state.alLambdaTangent[tangent] *=
            config.avbdAlpha * config.avbdGamma;
        state.penTangent[tangent] = physx::PxMax(
            1000.0f,
            physx::PxMin(config.avbdPenaltyMax,
                         state.penTangent[tangent] * config.avbdGamma));
      }
      continue;
    }

    const bool dynamicRigidOgc =
        geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
        geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies &&
        bodies[geometry.targetIndex].invMass > 0.0f;
    state.k = physx::PxMin(dynamicRigidOgc ? 1e5f : 1e4f, state.ke);
    if (geometry.hasRigidBodyTarget() && geometry.targetIndex < numBodies) {
      const AvbdSolverBody &rigidBody = bodies[geometry.targetIndex];
      if (rigidBody.invMass > 0.0f) {
        const physx::PxReal mass = 1.0f / rigidBody.invMass;
        const physx::PxReal floor = 0.25f * mass * invDt2;
        state.k = physx::PxMax(physx::PxMax(state.k, floor), 1000.0f);
      }
    }
  }
}

} // namespace Dy
} // namespace physx
