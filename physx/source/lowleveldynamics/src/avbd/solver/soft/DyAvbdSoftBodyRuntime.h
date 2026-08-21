// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_RUNTIME_H
#define DY_AVBD_SOFT_BODY_RUNTIME_H

#include "avbd/solver/soft/DyAvbdSoftBodyCompiledData.h"
#include "avbd/solver/soft/DyAvbdSoftBodyData.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
namespace Dy
{

// Runtime object joining immutable compiled topology, material parameters and
// mutable objective/contact state for one soft body.
struct AvbdSoftBody
{
	AvbdSoftBodyCompiledData compiled;
	AvbdSoftBodyMaterialData material;
	AvbdSoftBodyRuntimeState runtime;

	PX_FORCE_INLINE void buildElements(
		const PxArray<AvbdSoftParticle>& particles)
	{
		compiled.buildElements(particles, material, runtime);
	}
};

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_RUNTIME_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_RUNTIME_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_RUNTIME_API
#endif

DY_AVBD_SOFT_BODY_RUNTIME_API bool
avbdCanUseSoftAdaptivePrimalInitialization(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies);

DY_AVBD_SOFT_BODY_RUNTIME_API bool
avbdCanUseSoftRigidPrimalInitialization(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies);

#undef DY_AVBD_SOFT_BODY_RUNTIME_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_RUNTIME_H
