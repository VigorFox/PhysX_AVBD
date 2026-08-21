// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/solver/joint/DyAvbdJointSoftExecutionData.h"

namespace physx
{
namespace Dy
{

void initializeAvbdSoftExecutionData(
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    bool useProvidedSoftExecutionPlan,
    bool useProvidedRigidTargetContactPlan,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxU32 numSoftParticles, AvbdSoftExecutionData &data) {
  if (useProvidedSoftExecutionPlan && softExecutionPlan) {
    data.particleBodyIndices = softExecutionPlan->particleBodyIndices;
    data.contactStarts = softExecutionPlan->contactStarts;
    data.contactRefs = softExecutionPlan->contactRefs;
    if (softExecutionPlan->hasTriangleCoreSafetyPlan(numSoftParticles)) {
      data.triangleCoreSafetyStarts =
          softExecutionPlan->triangleCoreSafetyStarts;
      data.numTriangleCoreSafetyStarts =
          softExecutionPlan->numTriangleCoreSafetyStarts;
      data.triangleCoreSafetyRefs = softExecutionPlan->triangleCoreSafetyRefs;
      data.numTriangleCoreSafetyRefs =
          softExecutionPlan->numTriangleCoreSafetyRefs;
    }
  } else {
    data.particleBodyIndicesStorage.resize(numSoftParticles);
    data.particleBodyConflictsStorage.resize(numSoftParticles);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex) {
      data.particleBodyIndicesStorage[particleIndex] = PX_MAX_U32;
      data.particleBodyConflictsStorage[particleIndex] = 0u;
    }
    for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies;
         ++bodyIndex) {
      const AvbdSoftBody &softBody = softBodies[bodyIndex];
      const physx::PxU32 particleStart = softBody.compiled.particleStart;
      const physx::PxU32 particleCount = softBody.compiled.particleCount;
      if (particleStart > numSoftParticles ||
          particleCount > numSoftParticles - particleStart)
        continue;
      for (physx::PxU32 localParticleIndex = 0;
           localParticleIndex < particleCount; ++localParticleIndex) {
        const physx::PxU32 particleIndex =
            particleStart + localParticleIndex;
        if (data.particleBodyConflictsStorage[particleIndex] ||
            data.particleBodyIndicesStorage[particleIndex] != PX_MAX_U32) {
          data.particleBodyConflictsStorage[particleIndex] = 1u;
          data.particleBodyIndicesStorage[particleIndex] = PX_MAX_U32;
          continue;
        }
        data.particleBodyIndicesStorage[particleIndex] = bodyIndex;
      }
    }

    data.contactStartsStorage.resize(numSoftParticles + 1u);
    data.contactCountsStorage.resize(numSoftParticles);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      data.contactCountsStorage[particleIndex] = 0u;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleIndexCount =
          avbdCollectSoftContactParticleIndices(geometry, particleIndices);
      for (physx::PxU32 i = 0; i < particleIndexCount; ++i) {
        const physx::PxU32 particleIndex = particleIndices[i];
        if (particleIndex >= numSoftParticles)
          continue;
        if (physx::PxAbs(avbdGetSoftContactParticleJacobianScale(
                geometry, particleIndex)) > 1e-12f)
          data.contactCountsStorage[particleIndex]++;
      }
    }
    data.contactStartsStorage[0] = 0u;
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      data.contactStartsStorage[particleIndex + 1u] =
          data.contactStartsStorage[particleIndex] +
          data.contactCountsStorage[particleIndex];
    data.contactRefsStorage.resize(
        data.contactStartsStorage[numSoftParticles]);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      data.contactCountsStorage[particleIndex] = 0u;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      physx::PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
      const physx::PxU32 particleIndexCount =
          avbdCollectSoftContactParticleIndices(geometry, particleIndices);
      for (physx::PxU32 i = 0; i < particleIndexCount; ++i) {
        const physx::PxU32 particleIndex = particleIndices[i];
        if (particleIndex >= numSoftParticles)
          continue;
        const physx::PxReal jacobianScale =
            avbdGetSoftContactParticleJacobianScale(
                geometry, particleIndex);
        if (physx::PxAbs(jacobianScale) <= 1e-12f)
          continue;
        data.contactRefsStorage[
            data.contactStartsStorage[particleIndex] +
            data.contactCountsStorage[particleIndex]++] =
            AvbdSoftContactParticleRef(contactIndex, jacobianScale);
      }
    }
    data.particleBodyIndices = data.particleBodyIndicesStorage.begin();
    data.contactStarts = data.contactStartsStorage.begin();
    data.contactRefs = data.contactRefsStorage.begin();
  }

  if (!data.triangleCoreSafetyStarts) {
    const physx::PxU32 maxCoreSafetyParticles =
        3u * AVBD_CONTACT_POINT_MAX_SUPPORT;
    auto collectTriangleCoreSafetyParticles =
        [&](physx::PxU32 contactIndex,
            physx::PxU32 *particleIndices) -> physx::PxU32 {
      const AvbdSoftContactGeometry &geometry =
          softContacts[contactIndex].geometry;
      const AvbdOgcTriangleCoreCertificate *certificate =
          getOgcTriangleCoreCertificate(
              useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr,
              contactIndex, numSoftContacts);
      if (!certificate || !geometry.hasRigidBodyTarget())
        return 0u;
      physx::PxU32 count = 0u;
      for (physx::PxU32 vertex = 0; vertex < 3u; ++vertex) {
        AvbdWeightedContactPoint mapping;
        if (!resolveOgcTriangleCorePoint(
                geometry, certificate, vertex, mapping))
          return PX_MAX_U32;
        if (mapping.count == 0u ||
            mapping.count > AVBD_CONTACT_POINT_MAX_SUPPORT)
          return PX_MAX_U32;
        for (physx::PxU32 support = 0; support < mapping.count; ++support) {
          const physx::PxU32 particleIndex =
              mapping.particleIndices[support];
          if (particleIndex >= numSoftParticles)
            return PX_MAX_U32;
          bool duplicate = false;
          for (physx::PxU32 prior = 0; prior < count; ++prior)
            duplicate |= particleIndices[prior] == particleIndex;
          if (!duplicate) {
            if (count >= maxCoreSafetyParticles)
              return PX_MAX_U32;
            particleIndices[count++] = particleIndex;
          }
        }
      }
      return count;
    };

    data.triangleCoreSafetyStartsStorage.resize(numSoftParticles + 1u);
    data.triangleCoreSafetyCountsStorage.resize(numSoftParticles);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      data.triangleCoreSafetyCountsStorage[particleIndex] = 0u;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      physx::PxU32 particleIndices[3 * AVBD_CONTACT_POINT_MAX_SUPPORT];
      const physx::PxU32 particleCount =
          collectTriangleCoreSafetyParticles(contactIndex, particleIndices);
      if (particleCount == PX_MAX_U32)
        continue;
      for (physx::PxU32 index = 0; index < particleCount; ++index)
        data.triangleCoreSafetyCountsStorage[particleIndices[index]]++;
    }
    data.triangleCoreSafetyStartsStorage[0] = 0u;
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      data.triangleCoreSafetyStartsStorage[particleIndex + 1u] =
          data.triangleCoreSafetyStartsStorage[particleIndex] +
          data.triangleCoreSafetyCountsStorage[particleIndex];
    data.triangleCoreSafetyRefsStorage.resize(
        data.triangleCoreSafetyStartsStorage[numSoftParticles]);
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      data.triangleCoreSafetyCountsStorage[particleIndex] = 0u;
    for (physx::PxU32 contactIndex = 0;
         contactIndex < numSoftContacts; ++contactIndex) {
      physx::PxU32 particleIndices[3 * AVBD_CONTACT_POINT_MAX_SUPPORT];
      const physx::PxU32 particleCount =
          collectTriangleCoreSafetyParticles(contactIndex, particleIndices);
      if (particleCount == PX_MAX_U32)
        continue;
      for (physx::PxU32 index = 0; index < particleCount; ++index) {
        const physx::PxU32 particleIndex = particleIndices[index];
        data.triangleCoreSafetyRefsStorage[
            data.triangleCoreSafetyStartsStorage[particleIndex] +
            data.triangleCoreSafetyCountsStorage[particleIndex]++] =
            AvbdSoftContactParticleRef(contactIndex, 1.0f);
      }
    }
    data.triangleCoreSafetyStarts =
        data.triangleCoreSafetyStartsStorage.begin();
    data.numTriangleCoreSafetyStarts =
        data.triangleCoreSafetyStartsStorage.size();
    data.triangleCoreSafetyRefs = data.triangleCoreSafetyRefsStorage.begin();
    data.numTriangleCoreSafetyRefs = data.triangleCoreSafetyRefsStorage.size();
  }

  if (useProvidedRigidTargetContactPlan && softExecutionPlan) {
    data.rigidTargetContactStarts = softExecutionPlan->rigidTargetContactStarts;
    data.rigidTargetContactRefs = softExecutionPlan->rigidTargetContactRefs;
  }
}

} // namespace Dy
} // namespace physx
