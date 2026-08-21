// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_MECHANICS_H
#define DY_AVBD_SOFT_BODY_MECHANICS_H

#include "avbd/backend/cpu/DyAvbdCpuIsa.h"
#include "foundation/PxMath.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftBody;
struct AvbdSoftParticle;

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_MECHANICS_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_MECHANICS_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_MECHANICS_API
#endif

DY_AVBD_SOFT_BODY_MECHANICS_API AvbdTetMaterialPacketKernels
avbdSelectTetMaterialPacketKernels(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies);

DY_AVBD_SOFT_BODY_MECHANICS_API bool
avbdApplySoftBodyRigidPrimalInitialGuess(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody& body);

DY_AVBD_SOFT_BODY_MECHANICS_API void avbdApplyBendingDamping(
	AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxReal dt);

enum class AvbdSoftTetDisplacementLimitReason : PxU8
{
	eNONE,
	ePOSITIVE_J_LIMITED,
	ePOSITIVE_J_REJECTED,
	eNONFINITE_REJECTED
};

struct AvbdSoftTetDisplacementLimitResult
{
	PxVec3 appliedDisplacement;
	PxReal fraction;
	AvbdSoftTetDisplacementLimitReason reason;

	AvbdSoftTetDisplacementLimitResult()
		: appliedDisplacement(0.0f), fraction(0.0f),
		  reason(AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED)
	{
	}

	AvbdSoftTetDisplacementLimitResult(
		const PxVec3& displacement, PxReal appliedFraction,
		AvbdSoftTetDisplacementLimitReason limitReason)
		: appliedDisplacement(displacement),
		  fraction(appliedFraction), reason(limitReason)
	{
	}
};

// A rejected candidate is a feasibility observation, not a convergence
// certificate: the applied displacement may be zero while H^-1 f is not.
struct AvbdSoftSweepConvergenceObservation
{
	PxReal maxLocalSolveDisplacementSq;
	PxReal maxAppliedDisplacementSq;
	PxU32 trustRegionLimitedSteps;
	PxU32 positiveJLimitedSteps;
	PxU32 positiveJRejectedSteps;
	PxU32 nonFiniteRejectedSteps;

	AvbdSoftSweepConvergenceObservation()
		: maxLocalSolveDisplacementSq(0.0f),
		  maxAppliedDisplacementSq(0.0f),
		  trustRegionLimitedSteps(0), positiveJLimitedSteps(0),
		  positiveJRejectedSteps(0), nonFiniteRejectedSteps(0)
	{
	}

	PX_FORCE_INLINE void observe(
		const PxVec3& localSolveDisplacement,
		bool trustRegionLimited,
		const AvbdSoftTetDisplacementLimitResult& limitResult)
	{
		const PxReal localSolveDisplacementSq =
			localSolveDisplacement.magnitudeSquared();
		if(localSolveDisplacement.isFinite() &&
			PxIsFinite(localSolveDisplacementSq))
		{
			maxLocalSolveDisplacementSq = PxMax(
				maxLocalSolveDisplacementSq,
				localSolveDisplacementSq);
		}
		else
			nonFiniteRejectedSteps++;

		const PxReal appliedDisplacementSq =
			limitResult.appliedDisplacement.magnitudeSquared();
		if(limitResult.appliedDisplacement.isFinite() &&
			PxIsFinite(appliedDisplacementSq))
		{
			maxAppliedDisplacementSq = PxMax(
				maxAppliedDisplacementSq,
				appliedDisplacementSq);
		}
		else
			nonFiniteRejectedSteps++;

		if(trustRegionLimited)
			trustRegionLimitedSteps++;

		switch(limitResult.reason)
		{
		case AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_LIMITED:
			positiveJLimitedSteps++;
			break;
		case AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_REJECTED:
			positiveJRejectedSteps++;
			break;
		case AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED:
			if(localSolveDisplacement.isFinite() &&
				PxIsFinite(localSolveDisplacementSq))
				nonFiniteRejectedSteps++;
			break;
		case AvbdSoftTetDisplacementLimitReason::eNONE:
			break;
		}
	}

	PX_FORCE_INLINE void merge(
		const AvbdSoftSweepConvergenceObservation& other)
	{
		maxLocalSolveDisplacementSq = PxMax(
			maxLocalSolveDisplacementSq,
			other.maxLocalSolveDisplacementSq);
		maxAppliedDisplacementSq = PxMax(
			maxAppliedDisplacementSq,
			other.maxAppliedDisplacementSq);
		trustRegionLimitedSteps += other.trustRegionLimitedSteps;
		positiveJLimitedSteps += other.positiveJLimitedSteps;
		positiveJRejectedSteps += other.positiveJRejectedSteps;
		nonFiniteRejectedSteps += other.nonFiniteRejectedSteps;
	}

	PX_FORCE_INLINE bool isAppliedDisplacementConverged(
		PxReal toleranceSq) const
	{
		return maxAppliedDisplacementSq < toleranceSq;
	}

	PX_FORCE_INLINE bool isResidualConverged(PxReal toleranceSq) const
	{
		return maxLocalSolveDisplacementSq < toleranceSq &&
			trustRegionLimitedSteps == 0 &&
			positiveJLimitedSteps == 0 &&
			positiveJRejectedSteps == 0 &&
			nonFiniteRejectedSteps == 0;
	}
};

struct AvbdSoftResidualConvergenceTracker
{
	PxReal toleranceSq;
	PxU32 requiredConsecutiveSweeps;
	PxU32 consecutiveSweeps;

	AvbdSoftResidualConvergenceTracker(
		PxReal solveToleranceSq, PxU32 requiredSweeps)
		: toleranceSq(solveToleranceSq),
		  requiredConsecutiveSweeps(PxMax(requiredSweeps, 1u)),
		  consecutiveSweeps(0)
	{
	}

	PX_FORCE_INLINE bool observe(
		const AvbdSoftSweepConvergenceObservation& observation)
	{
		consecutiveSweeps = observation.isResidualConverged(toleranceSq)
			? consecutiveSweeps + 1
			: 0;
		return consecutiveSweeps >= requiredConsecutiveSweeps;
	}
};

#undef DY_AVBD_SOFT_BODY_MECHANICS_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_MECHANICS_H
