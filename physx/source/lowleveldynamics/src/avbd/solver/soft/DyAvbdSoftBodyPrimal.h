// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_PRIMAL_H
#define DY_AVBD_SOFT_BODY_PRIMAL_H

#include "avbd/backend/cpu/DyAvbdCpuIsa.h"
#include "avbd/contact/DyAvbdContact.h"
#include "avbd/solver/soft/DyAvbdSoftBodyMechanics.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyWorkspace.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_PRIMAL_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_PRIMAL_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_PRIMAL_API
#endif

struct AvbdParticlePrimalRangeObservation
{
	AvbdSoftSweepConvergenceObservation sweepObservation;
	PxU64 tetLinearizationCacheFallbackParticleSteps;

	AvbdParticlePrimalRangeObservation()
		: tetLinearizationCacheFallbackParticleSteps(0)
	{
	}

	PX_FORCE_INLINE void merge(
		const AvbdParticlePrimalRangeObservation& other)
	{
		sweepObservation.merge(other.sweepObservation);
		tetLinearizationCacheFallbackParticleSteps +=
			other.tetLinearizationCacheFallbackParticleSteps;
	}
};

DY_AVBD_SOFT_BODY_PRIMAL_API void avbdAccumulateTetMaterialPacketContributions(
	const AvbdSoftBody& softBody, PxU32 localParticleIndex,
	const AvbdSoftParticle* particles,
	const AvbdTetMaterialPacketKernels& packetKernels,
	bool cacheTetLinearizations,
	AvbdTetVertexLinearization* tetLinearizations,
	PxVec3& force, PxMat33& hessian);

struct AvbdParticlePrimalSolveContext
{
	AvbdSoftParticle* particles;
	const AvbdSoftContact* contacts;
	const PxU32* contactStarts;
	const AvbdSoftContactParticleRef* contactIndices;
	const PxReal* selfCollisionSafetyBounds;
	PxReal invDt;
	PxReal invDtSq;
	AvbdTetMaterialPacketKernels tetMaterialPacketKernels;

	PX_FORCE_INLINE bool canUseTetMaterialPackets(
		const AvbdSoftBody& sb, PxU32 localParticleIndex) const
	{
		const bool hasMaterialKernel =
			sb.material.coRotationalVolumeModel
				? tetMaterialPacketKernels.corotational != NULL
				: tetMaterialPacketKernels.neoHookean != NULL;
		return hasMaterialKernel &&
			sb.compiled.tetIncidencePacketProgramValid &&
			localParticleIndex <
				sb.compiled.tetIncidencePacketRanges.size() &&
			sb.compiled.elementAdjacency[localParticleIndex].tetRefs.size() >=
				eAVBD_TET_INCIDENCE_PACKET_WIDTH;
	}

	// Keep the candidate instantiation behind a real call boundary.  In
	// particular, do not let its packet eligibility and fallback control flow
	// enter the canonical scalar step section.
	DY_AVBD_SOFT_BODY_PRIMAL_API void solveWithTetMaterialPackets(
		const AvbdSoftBody& sb, PxU32 localParticleIndex,
		AvbdParticlePrimalRangeObservation& observation) const;

	template<
		bool enableTetMaterialPackets = false,
		bool tetMaterialPacketEligibilityProven = false>
	PX_FORCE_INLINE void solve(
		const AvbdSoftBody& sb, PxU32 localParticleIndex,
		AvbdParticlePrimalRangeObservation& observation) const;
};

DY_AVBD_SOFT_BODY_PRIMAL_API void avbdSolveParticlePrimalTetMaterialPacketBodyRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdParticlePrimalRangeObservation& observation);

DY_AVBD_SOFT_BODY_PRIMAL_API void avbdSolveParticlePrimalIndependentBodyRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 bodyBegin, PxU32 bodyEnd,
	AvbdParticlePrimalRangeObservation& observation);

DY_AVBD_SOFT_BODY_PRIMAL_API void avbdSolveParticlePrimalPackedRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxU32* particleBodyIndices, PxU32 numParticles,
	const PxU32* packedParticleIndices,
	PxU32 packedBegin, PxU32 packedEnd,
	AvbdParticlePrimalRangeObservation& observation);

struct AvbdParticlePrimalCausalLayerState
{
	DY_AVBD_SOFT_BODY_PRIMAL_API AvbdParticlePrimalCausalLayerState();

	DY_AVBD_SOFT_BODY_PRIMAL_API bool begin(
		const AvbdParticlePrimalSolveContext& inputSolveContext,
		const AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
		const PxU32* inputParticleBodyIndices, PxU32 inputNumParticles,
		const PxU32* inputPackedParticleIndices,
		const PxU32* inputLayerOffsets, PxU32 inputLayerCount);

	DY_AVBD_SOFT_BODY_PRIMAL_API bool hasPublishedLayer() const;

	DY_AVBD_SOFT_BODY_PRIMAL_API PxU32 getPublishedLayerIndex() const;

	DY_AVBD_SOFT_BODY_PRIMAL_API void getPublishedPackedRange(
		PxU32& packedBegin, PxU32& packedEnd) const;

	// This is the one-worker reference consumer for a published layer.  The
	// future Scene task route performs this same range operation in children,
	// then calls completePublishedLayer() once on its fan-in parent.
	DY_AVBD_SOFT_BODY_PRIMAL_API void solvePublishedLayerSerial();

	// Parent-only deterministic reduction. Observation order is the stable
	// child-range order constructed by Scene, never task completion order.
	DY_AVBD_SOFT_BODY_PRIMAL_API bool completePublishedLayer(
		const AvbdParticlePrimalRangeObservation* observations,
		PxU32 observationCount);

	DY_AVBD_SOFT_BODY_PRIMAL_API
	const AvbdParticlePrimalRangeObservation& getSweepObservation() const;

private:
	const AvbdParticlePrimalSolveContext* solveContext;
	const AvbdSoftBody* softBodies;
	PxU32 numSoftBodies;
	const PxU32* particleBodyIndices;
	PxU32 numParticles;
	const PxU32* packedParticleIndices;
	const PxU32* layerOffsets;
	PxU32 layerCount;
	PxU32 currentLayer;
	AvbdParticlePrimalRangeObservation sweepObservation;
	bool active;
};

#undef DY_AVBD_SOFT_BODY_PRIMAL_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_PRIMAL_H
