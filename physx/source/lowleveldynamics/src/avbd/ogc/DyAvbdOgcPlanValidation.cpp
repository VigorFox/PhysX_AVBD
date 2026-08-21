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

#include "avbd/ogc/DyAvbdOgcPlanValidation.h"
#include "avbd/solver/rigid/DyAvbdSolverBody.h"
#include "avbd/solver/soft/DyAvbdSoftBody.h"
#include "avbd/solver/soft/DyAvbdSoftIslandPlan.h"

namespace physx {
namespace Dy {

bool avbdValidateSoftIslandExecutionPlan(
    const AvbdSoftIslandExecutionPlan &plan,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftContacts) {
  if (!plan.isComplete(numSoftParticles) || !softBodies ||
      numSoftBodies == 0)
    return false;

  for (physx::PxU32 particleIndex = 0;
       particleIndex < numSoftParticles; ++particleIndex) {
    const physx::PxU32 bodyIndex = plan.particleBodyIndices[particleIndex];
    if (bodyIndex >= numSoftBodies)
      return false;
    const AvbdSoftBody &softBody = softBodies[bodyIndex];
    if (particleIndex < softBody.compiled.particleStart ||
        particleIndex - softBody.compiled.particleStart >=
            softBody.compiled.particleCount ||
        plan.contactStarts[particleIndex] >
            plan.contactStarts[particleIndex + 1])
      return false;
  }
  for (physx::PxU32 refIndex = 0; refIndex < plan.numContactRefs;
       ++refIndex) {
    if (plan.contactRefs[refIndex].contactIndex >= numSoftContacts)
      return false;
  }
  return true;
}

#if PX_CHECKED
static bool avbdValidateRigidTargetContactPlan(
    const AvbdSoftIslandExecutionPlan &plan,
    const AvbdSolverBody *bodies, physx::PxU32 numRigidBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts) {
  if (!plan.hasRigidTargetContactPlan(numRigidBodies) ||
      !bodies || (numSoftContacts > 0 && !softContacts))
    return false;

  for (physx::PxU32 bodyIndex = 0; bodyIndex < numRigidBodies;
       ++bodyIndex) {
    if (bodies[bodyIndex].nodeIndex != bodyIndex)
      return false;
  }

  physx::PxU32 expectedRefCount = 0;
  for (physx::PxU32 contactIndex = 0; contactIndex < numSoftContacts;
       ++contactIndex) {
    const AvbdSoftContactGeometry &geometry =
        softContacts[contactIndex].geometry;
    if (geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numRigidBodies)
      ++expectedRefCount;
  }
  if (expectedRefCount != plan.numRigidTargetContactRefs)
    return false;

  for (physx::PxU32 rigidBodyIndex = 0;
       rigidBodyIndex < numRigidBodies; ++rigidBodyIndex) {
    const physx::PxU32 begin =
        plan.rigidTargetContactStarts[rigidBodyIndex];
    const physx::PxU32 end =
        plan.rigidTargetContactStarts[rigidBodyIndex + 1];
    if (begin > end)
      return false;
    physx::PxU32 previousContactIndex = 0;
    for (physx::PxU32 refIndex = begin; refIndex < end; ++refIndex) {
      const physx::PxU32 contactIndex =
          plan.rigidTargetContactRefs[refIndex];
      if (contactIndex >= numSoftContacts ||
          (refIndex > begin && previousContactIndex >= contactIndex))
        return false;
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex != rigidBodyIndex)
        return false;
      previousContactIndex = contactIndex;
    }
  }
  return true;
}
#endif

bool avbdCanUseRigidTargetContactPlan(
    const AvbdSoftIslandExecutionPlan &plan,
    const AvbdSolverBody *bodies, physx::PxU32 numRigidBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts) {
  if (!plan.hasRigidTargetContactPlan(numRigidBodies) || !bodies)
    return false;
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numRigidBodies;
       ++bodyIndex) {
    const physx::PxU32 begin =
        plan.rigidTargetContactStarts[bodyIndex];
    const physx::PxU32 end =
        plan.rigidTargetContactStarts[bodyIndex + 1];
    if (bodies[bodyIndex].nodeIndex != bodyIndex || begin > end ||
        end > plan.numRigidTargetContactRefs)
      return false;
  }
#if PX_CHECKED
  return avbdValidateRigidTargetContactPlan(
      plan, bodies, numRigidBodies, softContacts, numSoftContacts);
#else
  PX_UNUSED(softContacts);
  PX_UNUSED(numSoftContacts);
  return true;
#endif
}

} // namespace Dy
} // namespace physx
