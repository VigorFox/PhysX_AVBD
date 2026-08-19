// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "DyAvbdGpuWaveBackend.h"
#include "DyAvbdGpuWaveBridge.h"
#include "DyAvbdSolver.h"
#include "DyAvbdOwnerWaveContract.h"

namespace physx
{
namespace Dy
{
static_assert(eAVBD_RIGID_LDLT_PACKET_WIDTH == PXG_AVBD_OWNER_WAVE_WIDTH,
	"CPU producer and GPU owner-wave packet widths must match");

namespace
{
	bool prepareRigidOwnerWavePacket(
		void* userData, const void* solveContext, PxU32 waveIndex,
		PxU32 waveBodyOffset, PxU32 epoch, PxF32 avbdAlpha,
		void* opaquePacket)
	{
		PX_UNUSED(userData);
		if(!solveContext || !opaquePacket)
			return false;
		AvbdRigidLocalSystemAoSoA8 scratch;
		return avbdBuildRigidOwnerWavePacket(
			*static_cast<const AvbdRigidSolveContext*>(solveContext),
			waveIndex, waveBodyOffset, epoch, avbdAlpha,
			*static_cast<PxgAvbdRigidOwnerWavePacket8*>(opaquePacket), scratch);
	}

	bool executeRigidOwnerWaveScalarFallback(
		void* userData, void* scalarSolver, void* solveContext,
		const void* opaquePacket, PxU8 validMask)
	{
		PX_UNUSED(userData);
		if(!scalarSolver || !solveContext || !opaquePacket)
			return false;
		AvbdSolver& solver = *static_cast<AvbdSolver*>(scalarSolver);
		AvbdRigidSolveContext& context =
			*static_cast<AvbdRigidSolveContext*>(solveContext);
		const PxgAvbdRigidOwnerWavePacket8& packet =
			*static_cast<const PxgAvbdRigidOwnerWavePacket8*>(opaquePacket);
		const PxU8 activeMask = packet.desc.activeMask;
		if((validMask & PxU8(~activeMask)) != 0u ||
			!context.iteration.bodies || !context.iteration.contacts ||
			!context.iteration.contactMap)
			return false;
		// Validate the complete packet before the first scalar lane mutates the
		// wave. This keeps the backend's false-means-untouched contract exact.
		for(PxU32 lane = 0; lane < PXG_AVBD_OWNER_WAVE_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((activeMask & bit) != 0u && (validMask & bit) == 0u &&
				packet.ownerBodyIndex[lane] >= context.iteration.numBodies)
				return false;
		}
		for(PxU32 lane = 0; lane < PXG_AVBD_OWNER_WAVE_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((activeMask & bit) == 0 || (validMask & bit) != 0)
				continue;
			if(!solver.solveRigidOwnerFallback(
					context, packet.ownerBodyIndex, lane))
				return false;
		}
		return true;
	}

	bool commitRigidOwnerWaveWriteback(
		void* userData, void* solveContext, const void* opaquePacket,
		const void* opaqueSolution)
	{
		PX_UNUSED(userData);
		if(!solveContext || !opaquePacket || !opaqueSolution)
			return false;
		AvbdRigidSolveContext& context =
			*static_cast<AvbdRigidSolveContext*>(solveContext);
		const PxgAvbdRigidOwnerWavePacket8& packet =
			*static_cast<const PxgAvbdRigidOwnerWavePacket8*>(opaquePacket);
		const PxgAvbdRigidOwnerWaveSolution8& solution =
			*static_cast<const PxgAvbdRigidOwnerWaveSolution8*>(opaqueSolution);
		return avbdApplyRigidOwnerWaveSolution(
			context.iteration.bodies, context.iteration.numBodies,
			packet, solution);
	}
}

void avbdGetRigidGpuWaveCallbackTable(
	AvbdRigidGpuWaveCallbackTable& table, void* userData)
{
	table = AvbdRigidGpuWaveCallbackTable();
	table.version = AVBD_RIGID_GPU_WAVE_CALLBACK_VERSION;
	table.userData = userData;
	table.preparePacket = prepareRigidOwnerWavePacket;
	table.executeScalarFallback = executeRigidOwnerWaveScalarFallback;
	table.commitWriteback = commitRigidOwnerWaveWriteback;
}

} // namespace Dy
} // namespace physx
