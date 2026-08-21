// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_CREATION_H
#define DY_AVBD_SOFT_BODY_CREATION_H

#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_CREATION_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_CREATION_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_CREATION_API
#endif

DY_AVBD_SOFT_BODY_CREATION_API PxU32 avbdCreateSoftBody(
	const PxVec3* vertices, PxU32 numVertices,
	const PxU32* tets, PxU32 numTetIndices,
	const PxU32* tris, PxU32 numTriIndices,
	PxReal youngsModulus, PxReal poissonsRatio,
	PxReal density, PxReal damping,
	PxReal bendingStiffness, PxReal thickness,
	PxArray<AvbdSoftParticle>& outParticles,
	PxArray<AvbdSoftBody>& outSoftBodies,
	bool flatteningEnabled = false,
	PxReal selfCollisionFilterDistance = 0.0f,
	PxReal dynamicFriction = 0.5f,
	bool coRotationalVolumeModel = true);

#undef DY_AVBD_SOFT_BODY_CREATION_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_CREATION_H
