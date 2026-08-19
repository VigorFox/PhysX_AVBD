// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#ifndef PXG_AVBD_OWNER_WAVE_SOLVER_CORE_H
#define PXG_AVBD_OWNER_WAVE_SOLVER_CORE_H

#include <cuda.h>

#include "DyAvbdOwnerWaveContract.h"

namespace physx
{
	class PxCudaContext;
	class PxCudaContextManager;
	class PxgCudaKernelWranglerManager;

	/**
	 * Standalone AVBD owner-wave GPU core. It owns no PGS/TGS state and is not
	 * a scene context; PxgAvbdDynamicsContextImpl composes it for one bounded
	 * owner-wave packet batch at a time.
	 */
	class PxgAvbdOwnerWaveSolverCore
	{
		PX_NOCOPY(PxgAvbdOwnerWaveSolverCore)

		PxgCudaKernelWranglerManager* mKernelWrangler;
		PxCudaContextManager* mCudaContextManager;
		PxCudaContext* mCudaContext;
		CUdeviceptr mDevicePacket;
		CUdeviceptr mDeviceSolution;

		CUresult ensureWaveBuffers();

	public:
		PxgAvbdOwnerWaveSolverCore(PxgCudaKernelWranglerManager* kernelWrangler,
			PxCudaContextManager* cudaContextManager);
		~PxgAvbdOwnerWaveSolverCore();

		bool isReady() const { return mKernelWrangler != NULL && mCudaContext != NULL; }
		CUresult launchBatch(const PxgAvbdRigidOwnerWavePacket8* devicePackets,
			PxgAvbdRigidOwnerWaveSolution8* deviceSolutions, PxU32 packetCount,
			CUstream stream) const;
		CUresult uploadPackets(const PxgAvbdRigidOwnerWavePacket8* packets,
			PxU32 packetCount, CUstream stream);
		CUresult downloadSolutions(PxgAvbdRigidOwnerWaveSolution8* solutions,
			PxU32 packetCount, CUstream stream);
		CUresult synchronize(CUstream stream) const;
		PxgAvbdRigidOwnerWavePacket8* devicePackets() const
		{
			return reinterpret_cast<PxgAvbdRigidOwnerWavePacket8*>(
				mDevicePacket);
		}
		PxgAvbdRigidOwnerWaveSolution8* deviceSolutions() const
		{
			return reinterpret_cast<PxgAvbdRigidOwnerWaveSolution8*>(
				mDeviceSolution);
		}
	};
}

#endif // PXG_AVBD_OWNER_WAVE_SOLVER_CORE_H
