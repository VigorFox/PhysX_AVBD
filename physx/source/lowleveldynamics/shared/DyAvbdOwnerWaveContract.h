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

#ifndef DY_AVBD_OWNER_WAVE_CONTRACT_H
#define DY_AVBD_OWNER_WAVE_CONTRACT_H

#include "PxPhysXConfig.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{

// Device-neutral owner-major packet ABI shared by the CPU producer and the
// CUDA batch solver. It contains no allocator, task, atomic, or host pointer.
static const PxU32 PXG_AVBD_BACKEND_CONTRACT_VERSION = 1u;
static const PxU32 PXG_AVBD_OWNER_WAVE_WIDTH = 8u;
static const PxU32 PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS = 8u;
static const PxU32 PXG_AVBD_OWNER_WAVE_MAX_OWNERS =
	PXG_AVBD_OWNER_WAVE_WIDTH * PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS;

struct PxgAvbdOwnerWaveDesc
{
	PxU32 waveEpoch;
	PxU32 ownerCount;
	PxU32 bodyCount;
	PxU32 contactCount;
	PxF32 dt;
	PxF32 invDt2;
	PxF32 avbdAlpha;
	PxF32 regularizationCoefficient;
	PxF32 singularThreshold;
	PxF32 conditionNumberThreshold;
	PxU32 maxRegularizationAttempts;
	PxU8 activeMask;
	PxU8 touchingMask;
	PxU8 padding[2];
};

struct PxgAvbdRigidOwnerWavePacket8
{
	PxgAvbdOwnerWaveDesc desc;
	PxU32 ownerBodyIndex[PXG_AVBD_OWNER_WAVE_WIDTH];
	PxF32 linearLinear[3][3][PXG_AVBD_OWNER_WAVE_WIDTH];
	PxF32 angularLinear[3][3][PXG_AVBD_OWNER_WAVE_WIDTH];
	PxF32 angularAngular[3][3][PXG_AVBD_OWNER_WAVE_WIDTH];
	PxF32 rhsLinear[3][PXG_AVBD_OWNER_WAVE_WIDTH];
	PxF32 rhsAngular[3][PXG_AVBD_OWNER_WAVE_WIDTH];
};

struct PxgAvbdRigidOwnerWaveSolution8
{
	PxF32 linear[3][PXG_AVBD_OWNER_WAVE_WIDTH];
	PxF32 angular[3][PXG_AVBD_OWNER_WAVE_WIDTH];
	PxU8 validLane[PXG_AVBD_OWNER_WAVE_WIDTH];
};

} // namespace physx

#endif // DY_AVBD_OWNER_WAVE_CONTRACT_H
