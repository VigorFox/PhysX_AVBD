// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_TRIANGLE_CORE_GEOMETRY_H
#define DY_AVBD_OGC_TRIANGLE_CORE_GEOMETRY_H

#include "foundation/PxSimpleTypes.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

struct AvbdSoftContactGeometry;
struct AvbdOgcRigidBoxGeometry;
struct AvbdOgcTriangleCoreCertificate;
struct AvbdSoftParticle;
struct AvbdSolverBody;

bool getRigidBoxTriangleCoreExitDistance(
    const physx::PxVec3 &halfExtent, const physx::PxVec3 &minimumLocal,
    const physx::PxVec3 &maximumLocal, physx::PxReal margin,
    physx::PxU32 face,
    physx::PxReal &distance);

bool getRigidBoxTriangleCoreMinimumExitFace(
    const physx::PxVec3 &halfExtent, const physx::PxVec3 &minimumLocal,
    const physx::PxVec3 &maximumLocal, physx::PxReal margin,
    physx::PxU32 &face,
    physx::PxReal &distance);

// Merge one complete collision triangle into the six common OBB face exits
// used by both static and dynamic local-manifold projectors.
bool accumulateRigidBoxTriangleCoreFaceExits(
    const physx::PxVec3 &halfExtent,
    const physx::PxVec3 &minimumLocal,
    const physx::PxVec3 &maximumLocal,
    physx::PxReal faceExits[6]);

bool getCurrentRigidBoxTriangleCoreLocalBounds(
    const AvbdSoftContactGeometry &geometry,
    const physx::PxTransform &boxToWorld,
    const AvbdSoftParticle *particles, physx::PxU32 numParticles,
    physx::PxVec3 &minimumLocal, physx::PxVec3 &maximumLocal,
    physx::PxU32 movedParticleIndex = PX_MAX_U32,
    const physx::PxVec3 &movedParticleDisplacement =
        physx::PxVec3(0.0f),
    const AvbdOgcTriangleCoreCertificate *certificate = nullptr);

physx::PxVec3 getRigidBoxTriangleCoreExitNormalLocal(physx::PxU32 face);

bool getCurrentRigidBoxTriangleCoreFaceGap(
    const AvbdSoftContactGeometry &geometry, const AvbdSolverBody *body,
    const AvbdSoftParticle *particles, physx::PxU32 numParticles,
    physx::PxReal &faceGap,
    physx::PxU32 movedParticleIndex = PX_MAX_U32,
    const physx::PxVec3 &movedParticleDisplacement =
        physx::PxVec3(0.0f),
    const AvbdOgcRigidBoxGeometry *rigidBox = nullptr,
    const AvbdOgcTriangleCoreCertificate *certificate = nullptr);

} // namespace Dy
} // namespace physx

#endif
