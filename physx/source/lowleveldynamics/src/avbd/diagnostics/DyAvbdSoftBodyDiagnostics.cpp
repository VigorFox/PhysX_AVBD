/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of NVIDIA CORPORATION nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "avbd/diagnostics/DyAvbdSoftBodyDiagnostics.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPolicy.h"

namespace physx
{
namespace Dy
{

// Publish topology-owned IR facts once per component step. This remains out
// of every particle range and does not imply that a packet evaluator runs.
void avbdPublishTetMaterialPacketIrStats(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftBodyStepStats* stepStats)
{
	if(!stepStats || !avbdUseTetMaterialPacketIr())
		return;
	const AvbdCpuIsaCorotationalTetPacket8Fn corotational =
		avbdGetCorotationalTetPacketKernel();
	const AvbdCpuIsaNeoHookeanTetPacket8Fn neoHookean =
		avbdGetNeoHookeanTetPacketKernel();
	stepStats->particlePrimalTetPacketIrBodies = 0;
	stepStats->particlePrimalTetPacketIrPackets = 0;
	stepStats->particlePrimalTetPacketIrActiveLanes = 0;
	stepStats->particlePrimalTetPacketIrTailLanes = 0;
	stepStats->particlePrimalTetPacketIrActiveTailLanes = 0;
	stepStats->particlePrimalTetPacketIrInvalidBodies = 0;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const AvbdSoftBodyCompiledData& compiled =
			body.compiled;
		const bool hasMaterialBackend =
			body.material.coRotationalVolumeModel
				? corotational != NULL : neoHookean != NULL;
		if(compiled.tetElements.empty() || !hasMaterialBackend)
			continue;
		// Topology compilation already validated the full packet mapping. A
		// step must only read that immutable result; rescanning every tet ref
		// here would distort the very material-stage timing P8 is preparing.
		if(!compiled.tetIncidencePacketProgramValid)
		{
			stepStats->particlePrimalTetPacketIrInvalidBodies++;
			continue;
		}
		stepStats->particlePrimalTetPacketIrBodies++;
		stepStats->particlePrimalTetPacketIrPackets +=
			compiled.tetIncidencePackets.size();
		for(PxU32 localParticleIndex = 0;
			localParticleIndex < compiled.particleCount;
			localParticleIndex++)
		{
			const PxU32 activeLanes =
				compiled.elementAdjacency[localParticleIndex].tetRefs.size();
			const PxU32 packets =
				compiled.tetIncidencePacketRanges[localParticleIndex].packetCount;
			stepStats->particlePrimalTetPacketIrActiveLanes += activeLanes;
			stepStats->particlePrimalTetPacketIrTailLanes +=
				packets * eAVBD_TET_INCIDENCE_PACKET_WIDTH - activeLanes;
			stepStats->particlePrimalTetPacketIrActiveTailLanes +=
				activeLanes % eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		}
	}
}

// P8.1 is a diagnostic transaction, not a particle-range dependency.  Keep
// its counters physically separate from the task-local convergence record so
// a disabled census cannot change the scalar primal's stack/register layout.
// This is deliberately separate from convergence telemetry. P8.1 uses the
// count-only result to choose a packet boundary, and does not change any
// convergence, limiter or early-out decision.
void avbdAccumulateParticlePrimalWorkCensus(
	AvbdSoftBodyStepStats& stepStats,
	const AvbdParticlePrimalWorkCensus& census)
{
	stepStats.particlePrimalCensusDynamicParticleSolves +=
		census.dynamicParticleSolves;
	stepStats.particlePrimalCensusTriangleEvaluations +=
		census.triangleEvaluations;
	stepStats.particlePrimalCensusCorotationalTetEvaluations +=
		census.corotationalTetEvaluations;
	stepStats.particlePrimalCensusNeoHookeanTetEvaluations +=
		census.neoHookeanTetEvaluations;
	stepStats.particlePrimalCensusBendingEvaluations +=
		census.bendingEvaluations;
	stepStats.particlePrimalCensusContactEvaluations +=
		census.contactEvaluations;
	stepStats.particlePrimalCensusTetPacket8FullPackets +=
		census.tetPacket8FullPackets;
	stepStats.particlePrimalCensusTetPacket8TailLanes +=
		census.tetPacket8TailLanes;
}

// P8.1 instrumentation is intentionally outside the scalar solve kernel.
// The count is a diagnostic-only replay of immutable sweep inputs.  The
// topology and contact index are fixed for one outer epoch, so one replay per
// epoch scaled by its executed sweep count is exactly equivalent to a replay
// after every inner sweep and leaves no census control in the scalar loop.
void avbdRecordParticlePrimalWorkCensusForSweep(
	const AvbdSoftParticle* particles, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, const PxU32* contactStarts,
	AvbdParticlePrimalWorkCensus& census)
{
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localIndex;
			if(particles[particleIndex].isStatic())
				continue;
			const AvbdParticleElementAdjacency& elementAdjacency =
				body.compiled.elementAdjacency[localIndex];
			const PxU32 tetIncidenceCount =
				elementAdjacency.tetRefs.size();
			census.dynamicParticleSolves++;
			census.triangleEvaluations +=
				elementAdjacency.triRefs.size();
			if(body.material.coRotationalVolumeModel)
				census.corotationalTetEvaluations +=
					tetIncidenceCount;
			else
				census.neoHookeanTetEvaluations +=
					tetIncidenceCount;
			census.bendingEvaluations +=
				elementAdjacency.bendRefs.size();
			census.contactEvaluations +=
				contactStarts[particleIndex + 1] -
				contactStarts[particleIndex];
			census.tetPacket8FullPackets +=
				tetIncidenceCount / 8;
			census.tetPacket8TailLanes +=
				tetIncidenceCount % 8;
		}
	}
}

// Keep the diagnostic replay entirely outside the scalar sweep's generated
// code.  The enabled path is intentionally one cold transaction per outer
// epoch; the default path has no inner-loop diagnostic control or call edge.
PX_NOINLINE void avbdAccumulateParticlePrimalWorkCensusForOuterEpoch(
	AvbdSoftBodyStepStats& stepStats,
	const AvbdSoftParticle* particles, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, const PxU32* contactStarts, PxU64 sweepCount)
{
	AvbdParticlePrimalWorkCensus workCensus;
	avbdRecordParticlePrimalWorkCensusForSweep(
		particles, softBodies, numSoftBodies, contactStarts, workCensus);
	workCensus.dynamicParticleSolves *= sweepCount;
	workCensus.triangleEvaluations *= sweepCount;
	workCensus.corotationalTetEvaluations *= sweepCount;
	workCensus.neoHookeanTetEvaluations *= sweepCount;
	workCensus.bendingEvaluations *= sweepCount;
	workCensus.contactEvaluations *= sweepCount;
	workCensus.tetPacket8FullPackets *= sweepCount;
	workCensus.tetPacket8TailLanes *= sweepCount;
	avbdAccumulateParticlePrimalWorkCensus(stepStats, workCensus);
}

} // namespace Dy
} // namespace physx
