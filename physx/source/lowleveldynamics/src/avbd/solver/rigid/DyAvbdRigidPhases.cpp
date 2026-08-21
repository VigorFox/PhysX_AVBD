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

#include "avbd/solver/rigid/DyAvbdRigidPhases.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

void runAvbdSoftPredictionPhase(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal dt, const physx::PxVec3 &gravity,
    bool hasPreparedSoftPrediction) {
  PX_PROFILE_ZONE("AVBD.prediction", 0);
  if (!hasPreparedSoftPrediction) {
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      softParticles[particleIndex].computePrediction(dt, gravity);
  }
}

void initializeAvbdNoContactBodies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap) {
  PX_PROFILE_ZONE("AVBD.initNoContactBodies", 0);
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    if (bodies[bodyIndex].invMass <= 0.0f)
      continue;

    bool hasContacts = false;
    if (contactMap && contactMap->numBodies > 0) {
      const physx::PxU32 *constraintIndices = nullptr;
      physx::PxU32 constraintCount = 0;
      contactMap->getBodyConstraints(
          bodyIndex, constraintIndices, constraintCount);
      hasContacts = constraintCount > 0;
    } else if (numContacts > 0) {
      for (physx::PxU32 contactIndex = 0;
           contactIndex < numContacts; ++contactIndex) {
        if (contacts[contactIndex].header.bodyIndexA == bodyIndex ||
            contacts[contactIndex].header.bodyIndexB == bodyIndex) {
          hasContacts = true;
          break;
        }
      }
    }

    if (!hasContacts) {
      bodies[bodyIndex].position = bodies[bodyIndex].inertialPosition;
      bodies[bodyIndex].rotation = bodies[bodyIndex].inertialRotation;
    }
  }
}

void warmstartAvbdRigidBodies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<bool> &touchesKinematicShell, physx::PxReal dt,
    physx::PxReal invDt, const physx::PxVec3 &gravity,
    physx::PxReal shellFastImpactSpeed) {
  const physx::PxReal gravMag = gravity.magnitude();
  const physx::PxVec3 gravDir =
      (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    body.prevPosition = body.position;
    body.prevRotation = body.rotation;

    if (body.invMass <= 0.0f)
      continue;

    if (touchesKinematicShell[i]) {
      const bool fastImpact =
          body.linearVelocity.magnitude() > shellFastImpactSpeed;
      body.position = fastImpact
                          ? body.inertialPosition
                          : body.prevPosition + body.linearVelocity * dt;
      body.rotation = body.inertialRotation;
    } else {
      const physx::PxVec3 accel =
          (body.linearVelocity - body.prevLinearVelocity) * invDt;
      const physx::PxReal accelWeight =
          (gravMag > 1e-6f)
              ? physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f)
              : 0.0f;
      body.position = body.prevPosition + body.linearVelocity * dt +
                      gravity * (accelWeight * dt * dt);
      body.rotation = body.inertialRotation;
    }
    body.projectLockedPose(body.prevPosition, body.prevRotation);
  }
}

void applyAvbdPenaltyFloor(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    physx::PxReal invDt2) {
  if (!contacts || numContacts == 0u)
    return;

  PX_PROFILE_ZONE("AVBD.penaltyFloor", 0);

  const int propagationDepth = 4;
  const physx::PxReal propagationDecay = 0.5f;

  physx::PxArray<physx::PxArray<physx::PxU32>> adj;
  adj.resize(numBodies);
  auto addEdge = [&](physx::PxU32 a, physx::PxU32 b) {
    if (a < numBodies && b < numBodies) {
      adj[a].pushBack(b);
      adj[b].pushBack(a);
    }
  };
  for (physx::PxU32 j = 0; j < numD6; ++j)
    addEdge(d6Joints[j].header.bodyIndexA, d6Joints[j].header.bodyIndexB);
  for (physx::PxU32 j = 0; j < numGear; ++j)
    addEdge(gearJoints[j].header.bodyIndexA,
            gearJoints[j].header.bodyIndexB);

  physx::PxArray<physx::PxReal> effectiveMassByBody;
  effectiveMassByBody.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    effectiveMassByBody[i] =
        (bodies[i].invMass > 0.0f) ? (1.0f / bodies[i].invMass) : 0.0f;

  for (int depth = 0; depth < propagationDepth; ++depth) {
    physx::PxArray<physx::PxReal> nextEffectiveMass;
    nextEffectiveMass.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal baseMass =
          (bodies[i].invMass > 0.0f) ? (1.0f / bodies[i].invMass) : 0.0f;
      physx::PxReal neighborSum = 0.0f;
      for (physx::PxU32 k = 0; k < adj[i].size(); ++k)
        neighborSum += effectiveMassByBody[adj[i][k]];
      nextEffectiveMass[i] =
          baseMass + propagationDecay * neighborSum;
    }
    effectiveMassByBody = nextEffectiveMass;
  }

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
    physx::PxReal massA = 0.0f;
    physx::PxReal massB = 0.0f;
    if (bodyA < numBodies && bodies[bodyA].invMass > 0.0f)
      massA = 1.0f / bodies[bodyA].invMass;
    if (bodyB < numBodies && bodies[bodyB].invMass > 0.0f)
      massB = 1.0f / bodies[bodyB].invMass;

    const physx::PxReal augmentedA =
        (bodyA < numBodies) ? effectiveMassByBody[bodyA] : 0.0f;
    const physx::PxReal augmentedB =
        (bodyB < numBodies) ? effectiveMassByBody[bodyB] : 0.0f;
    const physx::PxReal effectiveMass =
        physx::PxMax(augmentedA, augmentedB);
    const physx::PxReal penaltyScale =
        (massA > 0.0f && massB > 0.0f)
            ? AvbdConstants::AVBD_PEN_SCALE_DYN_DYN
            : AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC;
    const physx::PxReal penaltyFloor =
        penaltyScale * effectiveMass * invDt2;
    if (contacts[c].header.penalty < penaltyFloor)
      contacts[c].header.penalty = penaltyFloor;
    if (contacts[c].tangentPenalty0 < penaltyFloor)
      contacts[c].tangentPenalty0 = penaltyFloor;
    if (contacts[c].tangentPenalty1 < penaltyFloor)
      contacts[c].tangentPenalty1 = penaltyFloor;
  }
}

} // namespace Dy
} // namespace physx
