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

namespace physx {
namespace Dy {

// Post-AL velocity classification state.  The state is built once in contact
// order and then consumed by the velocity reconstruction loop; it does not
// own any velocity mutation or contact policy.

void AvbdPostAlVelocityState::build(
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts,
    const physx::PxArray<bool> &touchingBodyStatic,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    bool deformableFastImpactIsland) {
  physicalContactTangentOwnerIndex.resize(numBodies);
  fastNormalImpactByBody.resize(numBodies);
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    physicalContactTangentOwnerIndex[bodyIndex] = PX_MAX_U32;
    fastNormalImpactByBody[bodyIndex] = false;
  }
  haveSolveStartLinear = deformableFastImpactIsland &&
                         linearVelAtSolveStart &&
                         linearVelAtSolveStart->size() == numBodies;
  if (numContacts == 0 || !contacts)
    return;

  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdContactConstraint &contact = contacts[contactIndex];
    if (hasVelocityTangentMaterialOwner(contact) &&
        !hasVelocityBodyStaticFrictionSweepOwner(contact)) {
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      if (bodyA < numBodies &&
          physicalContactTangentOwnerIndex[bodyA] == PX_MAX_U32)
        physicalContactTangentOwnerIndex[bodyA] = contactIndex;
      if (bodyB < numBodies &&
          physicalContactTangentOwnerIndex[bodyB] == PX_MAX_U32)
        physicalContactTangentOwnerIndex[bodyB] = contactIndex;
    }
    if (!haveSolveStartLinear)
      continue;

    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies))
      continue;
    const physx::PxU32 dynamicBody = bodyA < numBodies ? bodyA : bodyB;
    if (dynamicBody >= numBodies || dynamicBody >= touchingBodyStatic.size() ||
        !touchingBodyStatic[dynamicBody])
      continue;
    const bool dynamicIsA = bodyA == dynamicBody;
    const physx::PxVec3 normal =
        contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
    if (-(*linearVelAtSolveStart)[dynamicBody].dot(normal) >
        AvbdConstants::AVBD_BODY_STATIC_FAST_IMPACT_SPEED)
      fastNormalImpactByBody[dynamicBody] = true;
  }
}

} // namespace Dy
} // namespace physx
