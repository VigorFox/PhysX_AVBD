// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_SELF_H
#define DY_AVBD_CONTACT_SELF_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/contact/DyAvbdSoftContactWorkspace.h"
#include "avbd/ogc/DyAvbdOgcParameters.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_SELF_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_SELF_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_SELF_API
#endif

PX_FORCE_INLINE void avbdTruncateDisplacement(
	AvbdSoftParticle& particle,
	const PxVec3& previousPosition,
	PxReal bound)
{
	const PxVec3 displacement = particle.position - previousPosition;
	const PxReal magnitude = displacement.magnitude();
	if(magnitude > bound && magnitude > 1.0e-10f)
		particle.position = previousPosition + displacement * (bound / magnitude);
}

DY_AVBD_CONTACT_SELF_API void avbdComputeSafetyBounds(
	const AvbdSoftBody& softBody,
	const AvbdSoftParticle* particles,
	const PxArray<PxArray<PxU32> >& adjacency,
	PxReal queryRadius,
	PxReal gammaP,
	PxArray<PxReal>& bounds,
	AvbdSoftContactWorkspace& workspace);

DY_AVBD_CONTACT_SELF_API void avbdDetectSelfCollisionOGC(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& softBody,
	PxU32 softBodyIndex,
	const PxArray<PxArray<PxU32> >& adjacency,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	const AvbdSoftContactWorkspace* preparedBvhWorkspace = NULL,
	PxU32 vertexLoopBegin = 0,
	PxU32 vertexLoopEnd = PX_MAX_U32,
	PxU32 edgeLoopBegin = 0,
	PxU32 edgeLoopEnd = PX_MAX_U32);

DY_AVBD_CONTACT_SELF_API bool avbdCanUseSelfCollisionOGCBvhRanges(
	const AvbdSoftBody& softBody);

DY_AVBD_CONTACT_SELF_API bool avbdPrepareSelfCollisionOGCBvhRanges(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& softBody,
	PxU32 softBodyIndex,
	const PxArray<PxArray<PxU32> >& adjacency,
	const AvbdOGCParams& params,
	AvbdSoftContactWorkspace& parentWorkspace,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_SELF_API void avbdDetectSelfCollisionOGCBvhRange(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& softBody,
	PxU32 softBodyIndex,
	const PxArray<PxArray<PxU32> >& adjacency,
	const AvbdSoftContactWorkspace& parentWorkspace,
	AvbdSoftContactWorkspace& rangeWorkspace,
	PxU32 vertexLoopBegin,
	PxU32 vertexLoopEnd,
	PxU32 edgeLoopBegin,
	PxU32 edgeLoopEnd,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL);

#undef DY_AVBD_CONTACT_SELF_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_SELF_H
