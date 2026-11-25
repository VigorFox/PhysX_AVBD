// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Smoke/host diagnostics for the AVBD backend. Deformables themselves use the
// public PxDeformableSurface/PxDeformableVolume Host and IG path.

#include "PxPhysicsAPI.h"

extern "C"
{

// P4a Host unlock: active IG::Node::eDEFORMABLE_VOLUME count with non-null Dy object.
PX_PHYSX_CORE_API physx::PxU32 avbdHostActiveDeformableVolumeNodeCount(physx::PxScene* scene);

// P5 Kernel table (Dy API, not Scene *Avbd*). deviceKind is the selected Dy::DeviceKind.
PX_PHYSX_CORE_API bool avbdHostKernelTableInfo(physx::PxScene* scene, physx::PxU32* deviceKind,
	bool* ldlt6Bound, bool* ldlt3Bound);
PX_PHYSX_CORE_API bool avbdHostKernelLdltSelfCheck();

}
