// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_SOFT_BODY_EPOCH_SAFETY_H
#define DY_AVBD_SOFT_BODY_EPOCH_SAFETY_H

#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyWorkspace.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API
#endif

// Shared access predicate used by hot OGC kernels. Keep this small function
// inline; epoch construction and trust-region policy live in the .cpp owner.
PX_FORCE_INLINE bool avbdSoftBodyContainsParticle(
	const AvbdSoftBody& body, PxU32 particleIndex, PxU32 numParticles)
{
	return body.compiled.particleStart <= numParticles &&
		body.compiled.particleCount <=
			numParticles - body.compiled.particleStart &&
		particleIndex >= body.compiled.particleStart &&
		particleIndex - body.compiled.particleStart <
			body.compiled.particleCount;
}

DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API void avbdSnapshotOuterPositionsScalar(
	AvbdSoftParticle* particles, PxU32 numParticles,
	PxReal* selfCollisionSafetyBounds);

DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API bool avbdCanReuseComponentOgcEpoch(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles);

DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API bool
avbdBuildComponentOgcGeometryEpoch(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles,
	AvbdSoftBodyWorkspace& workspace);

DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API bool
avbdApplyComponentOgcEpochSafetyBounds(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles,
	PxReal contactRadius, PxReal safetyRelax,
	PxReal* particleSafetyBounds, PxU32 numParticles,
	AvbdSoftBodyWorkspace& workspace);

#undef DY_AVBD_SOFT_BODY_EPOCH_SAFETY_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_EPOCH_SAFETY_H
