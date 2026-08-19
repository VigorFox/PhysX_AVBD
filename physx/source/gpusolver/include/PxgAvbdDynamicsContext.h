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

#ifndef PXG_AVBD_DYNAMICS_CONTEXT_H
#define PXG_AVBD_DYNAMICS_CONTEXT_H

#include "DyContext.h"
#include "DyAvbdGpuWaveBackend.h"
#include "DyAvbdOwnerWaveContract.h"

namespace physx
{

/**
 * Dedicated AVBD GPU context seam.
 *
 * This is intentionally an abstract Dy::Context rather than a subclass of
 * PxgGpuContext: the latter owns the existing PGS/TGS solver core and would
 * silently execute the wrong algorithm.  A concrete implementation must own
 * its own solver core, implement the Dy::Context lifecycle, and expose a
 * packet submission path before the factory may route eAVBD here.
 */
class PxgAvbdDynamicsContext : public Dy::Context,
	public Dy::AvbdRigidGpuWaveCallbackSink
{
	PX_NOCOPY(PxgAvbdDynamicsContext)

protected:
	PxgAvbdDynamicsContext(IG::SimpleIslandManager& islandManager,
		Cm::VirtualAllocatorCallback& allocator,
		Cm::VirtualAllocatorCallback& mappedAllocator,
		PxvSimStats& simStats, PxReal maxBiasCoefficient, PxReal lengthScale,
		PxU64 contextID, PxSceneFlags sceneFlags)
		: Dy::Context(islandManager, allocator, mappedAllocator, simStats,
			maxBiasCoefficient, lengthScale, contextID, sceneFlags)
	{
	}

	virtual ~PxgAvbdDynamicsContext() {}

public:
	virtual PxSolverType::Enum getSolverType() const PX_OVERRIDE PX_FINAL
	{
		return PxSolverType::eAVBD;
	}

	// Keep the complete Dy::Context lifecycle abstract.  No PGS/TGS default
	// implementation is permitted to satisfy the AVBD seam accidentally.
	virtual void destroy() PX_OVERRIDE = 0;
	virtual void update(Cm::FlushPool& flushPool, PxBaseTask* continuation,
		PxBaseTask* postPartitioningTask, PxBaseTask* processLostTouchTask,
		PxvNphaseImplementationContext* nPhaseContext, PxU32 maxPatchesPerCM,
		PxU32 maxArticulationLinks, PxReal dt, const PxVec3& gravity,
		Cm::PinnableBitMap& changedHandleMap) PX_OVERRIDE = 0;
	virtual void mergeResults() PX_OVERRIDE = 0;
	virtual void setSimulationController(PxsSimulationController* simulationController) PX_OVERRIDE = 0;
};

} // namespace physx

#endif // PXG_AVBD_DYNAMICS_CONTEXT_H
