// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "avbdOwnerWave.cuh"

using namespace physx;

extern "C" __host__ void initSolverKernels14() {}

extern "C" __global__ void avbdSolveRigidOwnerWaveBatch(
	const PxgAvbdRigidOwnerWavePacket8* PX_RESTRICT packets,
	PxgAvbdRigidOwnerWaveSolution8* PX_RESTRICT solutions,
	PxU32 packetCount)
{
	const PxU32 packetIndex = static_cast<PxU32>(blockIdx.x);
	const PxU32 lane = static_cast<PxU32>(threadIdx.x);
	if (packetIndex >= packetCount || lane >= PXG_AVBD_OWNER_WAVE_WIDTH)
		return;

	PxF32 output[avbdOwnerWave::kDofs] = {};
	const bool valid = avbdOwnerWave::solveLane(
		packets[packetIndex], static_cast<int>(lane), output);
	for (int i = 0; i < 3; ++i)
	{
		solutions[packetIndex].linear[i][lane] = output[i];
		solutions[packetIndex].angular[i][lane] = output[i + 3];
	}
	solutions[packetIndex].validLane[lane] = valid ? 1u : 0u;
}
