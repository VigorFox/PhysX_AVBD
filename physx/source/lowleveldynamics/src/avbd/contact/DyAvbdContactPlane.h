// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_PLANE_H
#define DY_AVBD_CONTACT_PLANE_H

#include "foundation/PxArray.h"
#include "foundation/PxVec3.h"
#include "PxMaterial.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftBody;
struct AvbdSoftContact;
struct AvbdSoftParticle;

struct AvbdWorldPlane
{
	PxVec3 normal;
	PxReal offset;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;

	AvbdWorldPlane()
		: normal(0.0f, 1.0f, 0.0f), offset(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)), primitiveKey(0)
	{
	}
};

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_PLANE_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_PLANE_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_PLANE_API
#endif

DY_AVBD_CONTACT_PLANE_API void avbdDetectSoftWorldPlaneContactsRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.02f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_PLANE_API void avbdDetectSoftWorldPlaneContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.02f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_PLANE_API void avbdDetectSoftGroundContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxArray<AvbdSoftContact>& contacts,
	PxReal groundY = 0.0f, PxReal margin = 0.02f,
	PxReal friction = 0.5f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

#undef DY_AVBD_CONTACT_PLANE_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_PLANE_H
