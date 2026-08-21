// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_SOFT_PAIR_H
#define DY_AVBD_CONTACT_SOFT_PAIR_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/contact/DyAvbdSoftContactWorkspace.h"
#include "avbd/ogc/DyAvbdOgcParameters.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftBody;
struct AvbdSoftParticle;

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_SOFT_PAIR_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_SOFT_PAIR_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_SOFT_PAIR_API
#endif

DY_AVBD_CONTACT_SOFT_PAIR_API void avbdBuildSoftSoftOGCDetectionPlan(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace& workspace);

DY_AVBD_CONTACT_SOFT_PAIR_API bool avbdRefitSoftSoftOGCDetectionPlan(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace& workspace);

DY_AVBD_CONTACT_SOFT_PAIR_API void avbdDetectSoftSoftOGCPlanRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContactWorkspace& refitWorkspace,
	AvbdSoftContactWorkspace* serialScratchWorkspace,
	AvbdSoftSoftPairQueryScratch& queryScratch,
	bool useSurfaceTriangleBvh,
	PxU32 planBegin, PxU32 planEnd,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_SOFT_PAIR_API void avbdDetectSoftSoftOGC(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	AvbdSoftSoftPairQueryScratch* queryScratchOverride = NULL);

#undef DY_AVBD_CONTACT_SOFT_PAIR_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_SOFT_PAIR_H
