// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "PxgAvbdOwnerWaveSolverCore.h"
#include "PxgKernelIndices.h"
#include "PxgKernelWrangler.h"
#include "CudaKernelWrangler.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"

namespace physx
{
	PxgAvbdOwnerWaveSolverCore::PxgAvbdOwnerWaveSolverCore(
		PxgCudaKernelWranglerManager* kernelWrangler,
		PxCudaContextManager* cudaContextManager)
		: mKernelWrangler(kernelWrangler),
		  mCudaContextManager(cudaContextManager),
		  mCudaContext(cudaContextManager ? cudaContextManager->getCudaContext() : NULL),
		  mDevicePacket(0), mDeviceSolution(0)
	{
	}

	PxgAvbdOwnerWaveSolverCore::~PxgAvbdOwnerWaveSolverCore()
	{
		const bool contextAcquired = mCudaContextManager &&
			mCudaContextManager->tryAcquireContext();
		if(mCudaContext && contextAcquired)
		{
			if(mDevicePacket)
				mCudaContext->memFree(mDevicePacket);
			if(mDeviceSolution)
				mCudaContext->memFree(mDeviceSolution);
		}
		if(contextAcquired)
			mCudaContextManager->releaseContext();
		mDevicePacket = 0;
		mDeviceSolution = 0;
	}

	CUresult PxgAvbdOwnerWaveSolverCore::ensureWaveBuffers()
	{
		if(!isReady())
			return CUDA_ERROR_INVALID_CONTEXT;
		if(mDevicePacket && mDeviceSolution)
			return CUDA_SUCCESS;
		if(!mDevicePacket && mCudaContext->memAlloc(
			&mDevicePacket, sizeof(PxgAvbdRigidOwnerWavePacket8) *
				PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS) != CUDA_SUCCESS)
			return CUDA_ERROR_OUT_OF_MEMORY;
		if(!mDeviceSolution && mCudaContext->memAlloc(
			&mDeviceSolution, sizeof(PxgAvbdRigidOwnerWaveSolution8) *
				PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS) != CUDA_SUCCESS)
		{
			mCudaContext->memFree(mDevicePacket);
			mDevicePacket = 0;
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		return CUDA_SUCCESS;
	}

	CUresult PxgAvbdOwnerWaveSolverCore::uploadPackets(
		const PxgAvbdRigidOwnerWavePacket8* packets, PxU32 packetCount,
		CUstream stream)
	{
		if(!packets || packetCount == 0 ||
			packetCount > PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS)
			return CUDA_ERROR_INVALID_VALUE;
		const CUresult result = ensureWaveBuffers();
		if(result != CUDA_SUCCESS)
			return result;
		return mCudaContext->memcpyHtoDAsync(
			mDevicePacket, packets,
			sizeof(PxgAvbdRigidOwnerWavePacket8) * packetCount, stream);
	}

	CUresult PxgAvbdOwnerWaveSolverCore::downloadSolutions(
		PxgAvbdRigidOwnerWaveSolution8* solutions, PxU32 packetCount,
		CUstream stream)
	{
		if(!solutions || packetCount == 0 ||
			packetCount > PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS ||
			!mDeviceSolution)
			return CUDA_ERROR_INVALID_VALUE;
		return mCudaContext->memcpyDtoHAsync(
			solutions, mDeviceSolution,
			sizeof(PxgAvbdRigidOwnerWaveSolution8) * packetCount, stream);
	}

	CUresult PxgAvbdOwnerWaveSolverCore::synchronize(CUstream stream) const
	{
		if(!mCudaContext)
			return CUDA_ERROR_INVALID_CONTEXT;
		return mCudaContext->streamSynchronize(stream);
	}

	CUresult PxgAvbdOwnerWaveSolverCore::launchBatch(
		const PxgAvbdRigidOwnerWavePacket8* devicePackets,
		PxgAvbdRigidOwnerWaveSolution8* deviceSolutions,
		PxU32 packetCount, CUstream stream) const
	{
		if(!isReady() || !devicePackets || !deviceSolutions ||
			packetCount == 0 ||
			packetCount > PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS)
			return CUDA_ERROR_INVALID_VALUE;
		const CUfunction function = mKernelWrangler->getKernelWrangler()->getCuFunction(
			PxgKernelIds::AVBD_SOLVE_RIGID_OWNER_WAVE_BATCH);
		if(!function)
			return CUDA_ERROR_NOT_FOUND;
		PxgAvbdRigidOwnerWavePacket8* packets =
			const_cast<PxgAvbdRigidOwnerWavePacket8*>(devicePackets);
		PxU32 count = packetCount;
		PxgAvbdRigidOwnerWaveSolution8* solutions = deviceSolutions;
		PxCudaKernelParam kernelParams[] =
		{
			PX_CUDA_KERNEL_PARAM(packets),
			PX_CUDA_KERNEL_PARAM(solutions),
			PX_CUDA_KERNEL_PARAM(count)
		};
		return mCudaContext->launchKernel(function, packetCount, 1, 1,
			PXG_AVBD_OWNER_WAVE_WIDTH, 1, 1, 0, stream, kernelParams,
			sizeof(kernelParams), 0, PX_FL);
	}

}
