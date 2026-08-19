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
// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "DyAvbdOwnerWaveContract.h"
#include "PxgAvbdDynamicsContext.h"
#include "PxgAvbdDynamicsContextImpl.h"
#include "PxgAvbdOwnerWaveSolverCore.h"

#include <type_traits>

namespace physx
{

// Host-side ABI assertions for the active owner-wave batch contract. The
// factory and solver entry points remain in their dedicated translation units;
// this file keeps CPU/GPU packet layout drift compile-time visible.
static_assert(PXG_AVBD_BACKEND_CONTRACT_VERSION == 1u, "unexpected AVBD GPU contract version");
static_assert(PXG_AVBD_OWNER_WAVE_WIDTH == 8u, "unexpected AVBD owner wave width");
static_assert(sizeof(PxgAvbdOwnerWaveDesc) == 48u, "AVBD owner descriptor ABI drift");
static_assert(sizeof(PxgAvbdRigidOwnerWavePacket8) == 1136u, "AVBD owner packet ABI drift");
static_assert(sizeof(PxgAvbdRigidOwnerWaveSolution8) == 200u, "AVBD owner solution ABI drift");
static_assert(std::is_abstract<PxgAvbdDynamicsContext>::value,
	"AVBD GPU context base must keep its lifecycle seam abstract");
static_assert(std::is_base_of<Dy::Context, PxgAvbdDynamicsContext>::value,
	"AVBD GPU context must use the low-level context lifecycle");
static_assert(!std::is_abstract<PxgAvbdDynamicsContextImpl>::value,
	"AVBD GPU context implementation must close the abstract lifecycle seam");
static_assert(std::is_base_of<PxgAvbdDynamicsContext, PxgAvbdDynamicsContextImpl>::value,
	"AVBD GPU context implementation must use the dedicated AVBD seam");
static_assert(std::is_base_of<Dy::AvbdRigidGpuWaveCallbackSink,
	PxgAvbdDynamicsContext>::value,
	"AVBD GPU context must expose only the opaque callback sink");

} // namespace physx
