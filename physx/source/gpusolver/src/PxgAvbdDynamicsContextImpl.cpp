// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "PxgAvbdDynamicsContextImpl.h"
#include "PxgKernelWrangler.h"
#include "avbd/pipeline/DyAvbdDynamics.h"
#include "foundation/PxAssert.h"
#include "cudamanager/PxCudaContextManager.h"

namespace physx
{
	namespace
	{
		class AvbdCudaExecutionScope
		{
			PxCudaContextManager* mManager;
			bool mAcquired;

		public:
			explicit AvbdCudaExecutionScope(PxCudaContextManager* manager)
				: mManager(manager),
				  mAcquired(manager && manager->tryAcquireContext())
			{
			}

			~AvbdCudaExecutionScope()
			{
				if(mAcquired)
					mManager->releaseContext();
			}

			bool acquired() const { return mAcquired; }
		};
	}

	PxgAvbdDynamicsContextImpl::PxgAvbdDynamicsContextImpl(
		PxsKernelWranglerManager* gpuKernelWrangler,
		PxCudaContextManager* cudaContextManager,
		IG::SimpleIslandManager& islandManager,
		Cm::VirtualAllocatorCallback& allocator,
		Cm::VirtualAllocatorCallback& mappedAllocator,
		PxvSimStats& simStats, PxReal maxBiasCoefficient,
		PxReal lengthScale, PxU64 contextID, PxSceneFlags sceneFlags)
		: PxgAvbdDynamicsContext(islandManager, allocator, mappedAllocator,
			simStats, maxBiasCoefficient, lengthScale, contextID, sceneFlags)
		, mOwnerWaveCore(
			static_cast<PxgCudaKernelWranglerManager*>(gpuKernelWrangler),
			cudaContextManager)
		, mCudaContextManager(cudaContextManager)
		, mOwnerWaveBackendReady(false)
		, mAttachedCpuContext(NULL)
	{
	}

	bool PxgAvbdDynamicsContextImpl::attachToCpuAvbdContext(
		Dy::AvbdDynamicsContext& cpuContext)
	{
		if(mAttachedCpuContext && mAttachedCpuContext != &cpuContext)
			return false;
		Dy::AvbdRigidGpuWaveBackend* existing =
			cpuContext.getRigidGpuWaveBackend();
		if((existing && existing != this) || !mCpuWaveCallbacks.isComplete())
			return false;
		cpuContext.setRigidGpuWaveBackend(this);
		mAttachedCpuContext = &cpuContext;
		return true;
	}

	bool PxgAvbdDynamicsContextImpl::setCpuWaveCallbacks(
		const Dy::AvbdRigidGpuWaveCallbackTable& callbacks)
	{
		if(!callbacks.isComplete())
			return false;
		mCpuWaveCallbacks = callbacks;
		return true;
	}

	void PxgAvbdDynamicsContextImpl::clearCpuWaveCallbacks()
	{
		mOwnerWaveBackendReady = false;
		mCpuWaveCallbacks = Dy::AvbdRigidGpuWaveCallbackTable();
	}

	bool PxgAvbdDynamicsContextImpl::enableOwnerWaveBackend()
	{
		if(!mOwnerWaveCore.isReady() || !mAttachedCpuContext ||
			!mCpuWaveCallbacks.isComplete())
			return false;
		mOwnerWaveBackendReady = true;
		return true;
	}

	bool PxgAvbdDynamicsContextImpl::detachFromCpuAvbdContext(
		Dy::AvbdDynamicsContext& cpuContext)
	{
		if(mAttachedCpuContext != &cpuContext ||
			cpuContext.getRigidGpuWaveBackend() != this)
			return false;
		mOwnerWaveBackendReady = false;
		cpuContext.setRigidGpuWaveBackend(NULL);
		mAttachedCpuContext = NULL;
		clearCpuWaveCallbacks();
		return true;
	}

	bool PxgAvbdDynamicsContextImpl::solveRigidOwnerWave(
		Dy::AvbdSolver& scalarSolver, Dy::AvbdRigidSolveContext& context,
		PxU32 waveIndex, PxU32 waveBodyOffset, PxU32 epoch,
		PxF32 avbdAlpha)
	{
		if(!mOwnerWaveBackendReady || !mOwnerWaveCore.isReady() ||
			!mCpuWaveCallbacks.isComplete() || epoch == 0u ||
			!(avbdAlpha >= 0.0f) || !context.iteration.bodies ||
			!context.iteration.contacts || !context.iteration.contactMap ||
			!(context.iteration.dt > 0.0f) || !(context.invDt2 > 0.0f) ||
			waveIndex >= context.dependencyWaveCount ||
			context.dependencyWaveOffsets.size() <
				context.dependencyWaveCount + 1u)
			return false;

		const PxU32 waveBegin = context.dependencyWaveOffsets[waveIndex];
		const PxU32 waveEnd = context.dependencyWaveOffsets[waveIndex + 1u];
		if(waveBegin > waveEnd ||
			waveEnd > context.dependencyWaveBodies.size() ||
			waveBodyOffset >= waveEnd - waveBegin)
			return false;

		const PxU32 waveBodyCount = waveEnd - waveBegin - waveBodyOffset;
		return solveRigidOwnerWavePackets(
			scalarSolver, context, waveIndex, waveBodyOffset, waveBodyCount,
			epoch, avbdAlpha);
	}

	bool PxgAvbdDynamicsContextImpl::solveRigidOwnerWavePackets(
		Dy::AvbdSolver& scalarSolver, Dy::AvbdRigidSolveContext& context,
		PxU32 waveIndex, PxU32 waveBodyOffset, PxU32 waveBodyCount,
		PxU32 epoch, PxF32 avbdAlpha)
	{
		const PxU32 packetWidth = PXG_AVBD_OWNER_WAVE_WIDTH;
		const PxU32 maxPackets = PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS;
		if(waveBodyCount == 0u || waveBodyCount > maxPackets * packetWidth)
			return false;

		const PxU32 waveBegin = context.dependencyWaveOffsets[waveIndex];
		const PxU32 packetCount =
			(waveBodyCount + packetWidth - 1u) / packetWidth;
		PxgAvbdRigidOwnerWavePacket8 packets[
			PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS];
		PxgAvbdRigidOwnerWaveSolution8 solutions[
			PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS];
		PxU8 packetGpu[PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS];
		PxU8 validMasks[PXG_AVBD_OWNER_WAVE_MAX_BATCH_PACKETS] = {};

		// Prepare and validate the complete CPU transaction before the first
		// scalar fallback or device writeback can mutate solver state.
		for(PxU32 packet = 0; packet < packetCount; ++packet)
		{
			const PxU32 packetOffset = waveBodyOffset + packet * packetWidth;
			const PxU32 packetBodies = PxMin(
				packetWidth, waveBodyCount - packet * packetWidth);
			const PxU8 expectedMask = PxU8(
				(PxU32(1u) << packetBodies) - 1u);
			packetGpu[packet] = 1u;
			if(!mCpuWaveCallbacks.preparePacket(
				mCpuWaveCallbacks.userData, &context, waveIndex, packetOffset,
				epoch, avbdAlpha, &packets[packet]))
			{
				packets[packet] = PxgAvbdRigidOwnerWavePacket8();
				packets[packet].desc.waveEpoch = epoch;
				packets[packet].desc.activeMask = expectedMask;
				for(PxU32 lane = 0; lane < packetBodies; ++lane)
					packets[packet].ownerBodyIndex[lane] =
						context.dependencyWaveBodies[
							waveBegin + packetOffset + lane];
				packetGpu[packet] = 0u;
			}

			if(packets[packet].desc.waveEpoch != epoch ||
				packets[packet].desc.activeMask != expectedMask)
				return false;
			for(PxU32 lane = 0; lane < packetBodies; ++lane)
				if(packets[packet].ownerBodyIndex[lane] >=
					context.iteration.numBodies)
					return false;
		}

		bool deviceBatchComplete = false;
		{
			PxMutex::ScopedLock ownerWaveLock(mOwnerWaveMutex);
			AvbdCudaExecutionScope cudaScope(mCudaContextManager);
			if(cudaScope.acquired())
			{
				deviceBatchComplete =
					mOwnerWaveCore.uploadPackets(
						packets, packetCount, NULL) == CUDA_SUCCESS &&
					mOwnerWaveCore.launchBatch(
						mOwnerWaveCore.devicePackets(),
						mOwnerWaveCore.deviceSolutions(), packetCount,
						NULL) == CUDA_SUCCESS &&
					mOwnerWaveCore.downloadSolutions(
						solutions, packetCount, NULL) == CUDA_SUCCESS &&
					mOwnerWaveCore.synchronize(NULL) == CUDA_SUCCESS;
			}
		}

		if(deviceBatchComplete)
		{
			for(PxU32 packet = 0; packet < packetCount; ++packet)
			{
				for(PxU32 lane = 0; lane < packetWidth; ++lane)
					if(solutions[packet].validLane[lane])
						validMasks[packet] = PxU8(
							validMasks[packet] | PxU8(1u << lane));
				if((validMasks[packet] &
					PxU8(~packets[packet].desc.activeMask)) != 0u)
					return false;
			}
		}

		for(PxU32 packet = 0; packet < packetCount; ++packet)
		{
			const PxU8 validMask =
				deviceBatchComplete && packetGpu[packet]
					? validMasks[packet] : 0u;
			if(!mCpuWaveCallbacks.executeScalarFallback(
				mCpuWaveCallbacks.userData, &scalarSolver, &context,
				&packets[packet], validMask))
				return false;
		}

		if(deviceBatchComplete)
		{
			for(PxU32 packet = 0; packet < packetCount; ++packet)
				if(packetGpu[packet] &&
					!mCpuWaveCallbacks.commitWriteback(
						mCpuWaveCallbacks.userData, &context,
						&packets[packet], &solutions[packet]))
					return false;
		}
		return true;
	}

	void PxgAvbdDynamicsContextImpl::destroy()
	{
		PX_ASSERT(!mAttachedCpuContext);
		this->~PxgAvbdDynamicsContextImpl();
		PX_FREE_THIS;
	}

	void PxgAvbdDynamicsContextImpl::update(
		Cm::FlushPool& flushPool, PxBaseTask* continuation,
		PxBaseTask* postPartitioningTask, PxBaseTask* processLostTouchTask,
		PxvNphaseImplementationContext* nPhaseContext, PxU32 maxPatchesPerCM,
		PxU32 maxArticulationLinks, PxReal dt, const PxVec3& gravity,
		Cm::PinnableBitMap& changedHandleMap)
	{
		PX_UNUSED(flushPool);
		PX_UNUSED(continuation);
		PX_UNUSED(postPartitioningTask);
		PX_UNUSED(processLostTouchTask);
		PX_UNUSED(nPhaseContext);
		PX_UNUSED(maxPatchesPerCM);
		PX_UNUSED(maxArticulationLinks);
		PX_UNUSED(dt);
		PX_UNUSED(gravity);
		PX_UNUSED(changedHandleMap);
		// This context is a hybrid owner-wave backend, not a scene-wide AVBD
		// dynamics context. Keep direct scene lifecycle entry fail-closed.
		PX_ASSERT(!"PxgAvbdDynamicsContextImpl scene lifecycle is not wired");
	}

	void PxgAvbdDynamicsContextImpl::mergeResults()
	{
		PX_ASSERT(!"PxgAvbdDynamicsContextImpl result merge is not wired");
	}

	void PxgAvbdDynamicsContextImpl::setSimulationController(
		PxsSimulationController* simulationController)
	{
		mSimulationController = simulationController;
	}
}
