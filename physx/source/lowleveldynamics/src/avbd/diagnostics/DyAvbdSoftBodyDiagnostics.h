// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_DIAGNOSTICS_H
#define DY_AVBD_SOFT_BODY_DIAGNOSTICS_H

#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyStep.h"

namespace physx
{
namespace Dy
{

struct AvbdParticlePrimalWorkCensus
{
	PxU64 dynamicParticleSolves;
	PxU64 triangleEvaluations;
	PxU64 corotationalTetEvaluations;
	PxU64 neoHookeanTetEvaluations;
	PxU64 bendingEvaluations;
	PxU64 contactEvaluations;
	PxU64 tetPacket8FullPackets;
	PxU64 tetPacket8TailLanes;

	AvbdParticlePrimalWorkCensus()
		: dynamicParticleSolves(0), triangleEvaluations(0),
		  corotationalTetEvaluations(0), neoHookeanTetEvaluations(0),
		  bendingEvaluations(0), contactEvaluations(0),
		  tetPacket8FullPackets(0), tetPacket8TailLanes(0)
	{
	}
};

void avbdPublishTetMaterialPacketIrStats(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftBodyStepStats* stepStats);

void avbdAccumulateParticlePrimalWorkCensus(
	AvbdSoftBodyStepStats& stepStats,
	const AvbdParticlePrimalWorkCensus& census);

void avbdRecordParticlePrimalWorkCensusForSweep(
	const AvbdSoftParticle* particles, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, const PxU32* contactStarts,
	AvbdParticlePrimalWorkCensus& census);

void avbdAccumulateParticlePrimalWorkCensusForOuterEpoch(
	AvbdSoftBodyStepStats& stepStats,
	const AvbdSoftParticle* particles, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, const PxU32* contactStarts, PxU64 sweepCount);

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_DIAGNOSTICS_H
