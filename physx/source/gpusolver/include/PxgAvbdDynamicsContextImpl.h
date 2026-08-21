// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#ifndef PXG_AVBD_DYNAMICS_CONTEXT_IMPL_H
#define PXG_AVBD_DYNAMICS_CONTEXT_IMPL_H

#include "PxgAvbdDynamicsContext.h"
#include "PxgAvbdOwnerWaveSolverCore.h"
#include "avbd/backend/gpu/DyAvbdGpuWaveBackend.h"
#include "foundation/PxMutex.h"

namespace physx
{
	namespace Dy
	{
		class AvbdDynamicsContext;
	}

	class PxCudaContextManager;
	class PxsKernelWranglerManager;

	/**
	 * Concrete AVBD context composition for the GPU owner-wave seam.
	 *
	 * This type owns the dedicated AVBD solver core and exposes only device
	 * packet submission.  The Dy::Context lifecycle remains deliberately
	 * fail-closed until the CPU owner installs the callback table and explicitly
	 * enables the owner-wave backend. The context is created only as a hybrid
	 * owner-wave component; it is never passed to PxgSimulationController.
	 */
	class PxgAvbdDynamicsContextImpl final : public PxgAvbdDynamicsContext,
		public Dy::AvbdRigidGpuWaveBackend
	{
		PX_NOCOPY(PxgAvbdDynamicsContextImpl)

		PxgAvbdOwnerWaveSolverCore mOwnerWaveCore;
		PxCudaContextManager* mCudaContextManager;
		PxMutex mOwnerWaveMutex;
		bool mOwnerWaveBackendReady;
		Dy::AvbdDynamicsContext* mAttachedCpuContext;
		Dy::AvbdRigidGpuWaveCallbackTable mCpuWaveCallbacks;

		bool solveRigidOwnerWavePackets(
			Dy::AvbdSolver& scalarSolver,
			Dy::AvbdRigidSolveContext& context, PxU32 waveIndex,
			PxU32 waveBodyOffset, PxU32 waveBodyCount, PxU32 epoch,
			PxF32 avbdAlpha);

	public:
		PxgAvbdDynamicsContextImpl(
			PxsKernelWranglerManager* gpuKernelWrangler,
			PxCudaContextManager* cudaContextManager,
			IG::SimpleIslandManager& islandManager,
			Cm::VirtualAllocatorCallback& allocator,
			Cm::VirtualAllocatorCallback& mappedAllocator,
			PxvSimStats& simStats, PxReal maxBiasCoefficient,
			PxReal lengthScale, PxU64 contextID, PxSceneFlags sceneFlags);

		bool isAvailable() const PX_OVERRIDE
		{
			// P218 owner-wave arithmetic is admitted only behind the scene-owned
			// capability and unchanged public CPU/device differential gates. The
			// owner-wave executor acquires the scene CUDA context only while it
			// owns the shared packet/solution buffers.
			return mOwnerWaveBackendReady && mOwnerWaveCore.isReady() &&
				mCudaContextManager != NULL;
		}
		bool solveRigidOwnerWave(
			Dy::AvbdSolver& scalarSolver,
			Dy::AvbdRigidSolveContext& context, PxU32 waveIndex,
			PxU32 waveBodyOffset, PxU32 epoch,
			PxF32 avbdAlpha) PX_OVERRIDE;
		// Explicit hybrid attach point; the caller owns both contexts and must
		// detach before destroying this backend.  Detach is explicit so a scene
		// teardown cannot leave a dangling optional hook in AvbdSolveIslandTask.
		bool attachToCpuAvbdContext(
			Dy::AvbdDynamicsContext& cpuContext) PX_OVERRIDE;
		bool detachFromCpuAvbdContext(
			Dy::AvbdDynamicsContext& cpuContext) PX_OVERRIDE;
		// Configure the opaque CPU producer/fallback/writeback boundary.  The
		// table is copied by value and remains owned by this context.  A complete
		// table is mandatory before attach; there is no direct CPU bridge in this
		// GPU object.
		bool setCpuWaveCallbacks(
			const Dy::AvbdRigidGpuWaveCallbackTable& callbacks) PX_OVERRIDE;
		void clearCpuWaveCallbacks() PX_OVERRIDE;
		bool enableOwnerWaveBackend() PX_OVERRIDE;

		void destroy() PX_OVERRIDE;
		void update(Cm::FlushPool& flushPool, PxBaseTask* continuation,
			PxBaseTask* postPartitioningTask, PxBaseTask* processLostTouchTask,
			PxvNphaseImplementationContext* nPhaseContext, PxU32 maxPatchesPerCM,
			PxU32 maxArticulationLinks, PxReal dt, const PxVec3& gravity,
			Cm::PinnableBitMap& changedHandleMap) PX_OVERRIDE;
		void mergeResults() PX_OVERRIDE;
		void setSimulationController(PxsSimulationController* simulationController) PX_OVERRIDE;

	protected:
		~PxgAvbdDynamicsContextImpl() PX_OVERRIDE {}
	};
}

#endif // PXG_AVBD_DYNAMICS_CONTEXT_IMPL_H
