// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

PX_NOINLINE void avbdAccumulateTetMaterialPacketContributions(
	const AvbdSoftBody& softBody, PxU32 localParticleIndex,
	const AvbdSoftParticle* particles,
	const AvbdTetMaterialPacketKernels& packetKernels,
	bool cacheTetLinearizations,
	AvbdTetVertexLinearization* tetLinearizations,
	PxVec3& force, PxMat33& hessian)
{
	const bool coRotational =
		softBody.material.coRotationalVolumeModel;
	PX_ASSERT(coRotational
		? packetKernels.corotational != NULL
		: packetKernels.neoHookean != NULL);
	const AvbdSoftBodyCompiledData& compiled = softBody.compiled;
	const AvbdParticleElementAdjacency& adjacency =
		compiled.elementAdjacency[localParticleIndex];
	const PxU32 tetIncidenceCount = adjacency.tetRefs.size();
	PX_ASSERT(compiled.tetIncidencePacketProgramValid);
	PX_ASSERT(localParticleIndex <
		compiled.tetIncidencePacketRanges.size());
	const AvbdTetIncidencePacketRange& packetRange =
		compiled.tetIncidencePacketRanges[localParticleIndex];
	const PxU32 fullPacketCount = tetIncidenceCount /
		eAVBD_TET_INCIDENCE_PACKET_WIDTH;
	const PxU32 tailLaneCount = tetIncidenceCount %
		eAVBD_TET_INCIDENCE_PACKET_WIDTH;
	const bool useTailPacket = tailLaneCount >= 4;
	const PxU32 vectorPacketCount =
		fullPacketCount + PxU32(useTailPacket);
	const PxU32 vectorIncidenceCount =
		fullPacketCount * eAVBD_TET_INCIDENCE_PACKET_WIDTH +
		(useTailPacket ? tailLaneCount : 0);
	PX_ASSERT(packetRange.packetCount >= vectorPacketCount);

	for(PxU32 packetOrdinal = 0;
		packetOrdinal < vectorPacketCount; packetOrdinal++)
	{
		const AvbdTetIncidencePacket8& incidencePacket =
			compiled.tetIncidencePackets[
				packetRange.packetStart + packetOrdinal];
		const PxU32 firstTetRefIndex = packetOrdinal *
			eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		const PxU32 packetLaneCount = PxMin(
			eAVBD_TET_INCIDENCE_PACKET_WIDTH,
			tetIncidenceCount - firstTetRefIndex);
		PX_ASSERT(incidencePacket.validMask ==
			(packetLaneCount == eAVBD_TET_INCIDENCE_PACKET_WIDTH
				? PxU8(0xffu)
				: PxU8((1u << packetLaneCount) - 1u)));
		AvbdTetMaterialPacket8Input input = {};
		AvbdTetMaterialPacket8Output output = {};
		for(PxU32 lane = 0;
			lane < packetLaneCount; lane++)
		{
			PX_ASSERT(adjacency.tetRefs[firstTetRefIndex + lane].index ==
				incidencePacket.tetIndices[lane]);
			PX_ASSERT(adjacency.tetRefs[firstTetRefIndex + lane].vOrder ==
				incidencePacket.vertexOrders[lane]);
			const AvbdTetElement& tet =
				compiled.tetElements[incidencePacket.tetIndices[lane]];
			const PxU32 vertexOrder =
				incidencePacket.vertexOrders[lane];
			const PxVec3 p0 = particles[tet.p0].position;
			const PxVec3 e1 = particles[tet.p1].position - p0;
			const PxVec3 e2 = particles[tet.p2].position - p0;
			const PxVec3 e3 = particles[tet.p3].position - p0;
			input.e1X[lane] = e1.x;
			input.e1Y[lane] = e1.y;
			input.e1Z[lane] = e1.z;
			input.e2X[lane] = e2.x;
			input.e2Y[lane] = e2.y;
			input.e2Z[lane] = e2.z;
			input.e3X[lane] = e3.x;
			input.e3Y[lane] = e3.y;
			input.e3Z[lane] = e3.z;
			input.dm0X[lane] = tet.DmInv.column0.x;
			input.dm0Y[lane] = tet.DmInv.column0.y;
			input.dm0Z[lane] = tet.DmInv.column0.z;
			input.dm1X[lane] = tet.DmInv.column1.x;
			input.dm1Y[lane] = tet.DmInv.column1.y;
			input.dm1Z[lane] = tet.DmInv.column1.z;
			input.dm2X[lane] = tet.DmInv.column2.x;
			input.dm2Y[lane] = tet.DmInv.column2.y;
			input.dm2Z[lane] = tet.DmInv.column2.z;
			input.shapeX[lane] = tet.shapeGradients[vertexOrder].x;
			input.shapeY[lane] = tet.shapeGradients[vertexOrder].y;
			input.shapeZ[lane] = tet.shapeGradients[vertexOrder].z;
			input.shapeNormSq[lane] =
				tet.shapeGradientNormSq[vertexOrder];
			input.restVolume[lane] = tet.restVolume;
		}

		if(coRotational)
			packetKernels.corotational(
				input, softBody.material.mu, softBody.material.lambda,
				output);
		else
			packetKernels.neoHookean(
				input, softBody.material.mu, softBody.material.lambda,
				softBody.material.neoHookeanAlpha, output);
		for(PxU32 lane = 0;
			lane < packetLaneCount; lane++)
		{
			PxVec3 elementForce;
			PxMat33 elementHessian;
			if((output.validMask & PxU8(1u << lane)) != 0u)
			{
				if(cacheTetLinearizations)
				{
					tetLinearizations[firstTetRefIndex + lane].determinant =
						output.determinant[lane];
					tetLinearizations[firstTetRefIndex + lane].determinantGradient =
						PxVec3(
							output.determinantGradientX[lane],
							output.determinantGradientY[lane],
							output.determinantGradientZ[lane]);
				}
				elementForce = PxVec3(
					output.forceX[lane], output.forceY[lane],
					output.forceZ[lane]);
				elementHessian = PxMat33(
					PxVec3(output.hessianXX[lane],
						output.hessianXY[lane], output.hessianXZ[lane]),
					PxVec3(output.hessianXY[lane],
						output.hessianYY[lane], output.hessianYZ[lane]),
					PxVec3(output.hessianXZ[lane],
						output.hessianYZ[lane], output.hessianZZ[lane]));
			}
			else
			{
				const PxU32 tetRefIndex = firstTetRefIndex + lane;
				const AvbdParticleElementRef& ref =
					adjacency.tetRefs[tetRefIndex];
				if(coRotational)
					avbdEvaluateCorotationalForceHessianPrepared(
						compiled.tetElements[ref.index], int(ref.vOrder),
						softBody.material.mu, softBody.material.lambda,
						particles, elementForce, elementHessian,
						cacheTetLinearizations
							? &tetLinearizations[tetRefIndex] : NULL);
				else
					avbdEvaluateNeoHookeanForceHessianPrepared(
						compiled.tetElements[ref.index], int(ref.vOrder),
						softBody.material.mu, softBody.material.lambda,
						softBody.material.neoHookeanAlpha, particles,
						elementForce, elementHessian,
						cacheTetLinearizations
							? &tetLinearizations[tetRefIndex] : NULL);
			}
			force = force + elementForce;
			hessian = hessian + elementHessian;
		}
	}

	for(PxU32 tetRefIndex = vectorIncidenceCount;
		tetRefIndex < tetIncidenceCount; tetRefIndex++)
	{
		const AvbdParticleElementRef& ref =
			adjacency.tetRefs[tetRefIndex];
		PxVec3 elementForce;
		PxMat33 elementHessian;
		if(coRotational)
			avbdEvaluateCorotationalForceHessianPrepared(
				compiled.tetElements[ref.index], int(ref.vOrder),
				softBody.material.mu, softBody.material.lambda,
				particles, elementForce, elementHessian,
				cacheTetLinearizations
					? &tetLinearizations[tetRefIndex] : NULL);
		else
			avbdEvaluateNeoHookeanForceHessianPrepared(
				compiled.tetElements[ref.index], int(ref.vOrder),
				softBody.material.mu, softBody.material.lambda,
				softBody.material.neoHookeanAlpha, particles,
				elementForce, elementHessian,
				cacheTetLinearizations
					? &tetLinearizations[tetRefIndex] : NULL);
		force = force + elementForce;
		hessian = hessian + elementHessian;
	}
}

PX_NOINLINE void AvbdParticlePrimalSolveContext::solveWithTetMaterialPackets(
	const AvbdSoftBody& sb, PxU32 localParticleIndex,
	AvbdParticlePrimalRangeObservation& observation) const
{
	PX_ASSERT(tetMaterialPacketKernels.hasAny());
	PX_ASSERT(canUseTetMaterialPackets(sb, localParticleIndex));
	solve<true, true>(sb, localParticleIndex, observation);
}

PX_NOINLINE void avbdSolveParticlePrimalTetMaterialPacketBodyRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdParticlePrimalRangeObservation& observation)
{
	PX_ASSERT(solveContext.tetMaterialPacketKernels.hasAny());
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const bool useBodyPackets =
			(body.material.coRotationalVolumeModel
				? solveContext.tetMaterialPacketKernels.corotational != NULL
				: solveContext.tetMaterialPacketKernels.neoHookean != NULL) &&
			body.compiled.tetIncidencePacketProgramValid;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; localIndex++)
		{
			if(useBodyPackets &&
				body.compiled.elementAdjacency[localIndex].tetRefs.size() >=
					eAVBD_TET_INCIDENCE_PACKET_WIDTH)
				solveContext.solveWithTetMaterialPackets(
					body, localIndex, observation);
			else
				solveContext.solve(body, localIndex, observation);
		}
	}
}

PX_NOINLINE void avbdSolveParticlePrimalIndependentBodyRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 bodyBegin, PxU32 bodyEnd,
	AvbdParticlePrimalRangeObservation& observation)
{
	PX_UNUSED(numSoftBodies);
	PX_ASSERT(softBodies && bodyBegin < bodyEnd &&
		bodyEnd <= numSoftBodies);
	for(PxU32 bodyIndex = bodyBegin; bodyIndex < bodyEnd; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const bool useBodyPackets =
			(body.material.coRotationalVolumeModel
				? solveContext.tetMaterialPacketKernels.corotational != NULL
				: solveContext.tetMaterialPacketKernels.neoHookean != NULL) &&
			body.compiled.tetIncidencePacketProgramValid;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; localIndex++)
		{
			if(useBodyPackets &&
				body.compiled.elementAdjacency[localIndex].tetRefs.size() >=
					eAVBD_TET_INCIDENCE_PACKET_WIDTH)
				solveContext.solveWithTetMaterialPackets(
					body, localIndex, observation);
			else
				solveContext.solve(body, localIndex, observation);
		}
	}
}

void avbdSolveParticlePrimalPackedRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxU32* particleBodyIndices, PxU32 numParticles,
	const PxU32* packedParticleIndices,
	PxU32 packedBegin, PxU32 packedEnd,
	AvbdParticlePrimalRangeObservation& observation)
{
	PX_UNUSED(numSoftBodies);
	PX_UNUSED(numParticles);
	PX_ASSERT(packedBegin <= packedEnd);
	if(solveContext.tetMaterialPacketKernels.hasAny())
	{
		for(PxU32 packedIndex = packedBegin;
			packedIndex < packedEnd; packedIndex++)
		{
			const PxU32 particleIndex = packedParticleIndices[packedIndex];
			PX_ASSERT(particleIndex < numParticles);
			const PxU32 bodyIndex = particleBodyIndices[particleIndex];
			PX_ASSERT(bodyIndex < numSoftBodies);
			const AvbdSoftBody& body = softBodies[bodyIndex];
			PX_ASSERT(particleIndex >= body.compiled.particleStart &&
				particleIndex - body.compiled.particleStart <
					body.compiled.particleCount);
			const PxU32 localParticleIndex =
				particleIndex - body.compiled.particleStart;
			if(solveContext.canUseTetMaterialPackets(
				body, localParticleIndex))
				solveContext.solveWithTetMaterialPackets(
					body, localParticleIndex, observation);
			else
				solveContext.solve(
					body, localParticleIndex, observation);
		}
	}
	else
	{
		for(PxU32 packedIndex = packedBegin;
			packedIndex < packedEnd; packedIndex++)
		{
			const PxU32 particleIndex = packedParticleIndices[packedIndex];
			PX_ASSERT(particleIndex < numParticles);
			const PxU32 bodyIndex = particleBodyIndices[particleIndex];
			PX_ASSERT(bodyIndex < numSoftBodies);
			const AvbdSoftBody& body = softBodies[bodyIndex];
			PX_ASSERT(particleIndex >= body.compiled.particleStart &&
				particleIndex - body.compiled.particleStart <
					body.compiled.particleCount);
			solveContext.solve(
				body, particleIndex - body.compiled.particleStart,
				observation);
		}
	}
}

AvbdParticlePrimalCausalLayerState::AvbdParticlePrimalCausalLayerState()
: solveContext(NULL), softBodies(NULL), numSoftBodies(0),
	  particleBodyIndices(NULL), numParticles(0),
	  packedParticleIndices(NULL), layerOffsets(NULL),
	  layerCount(0), currentLayer(0), active(false)
{
}

bool AvbdParticlePrimalCausalLayerState::begin(
	const AvbdParticlePrimalSolveContext& inputSolveContext,
	const AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
	const PxU32* inputParticleBodyIndices, PxU32 inputNumParticles,
	const PxU32* inputPackedParticleIndices,
	const PxU32* inputLayerOffsets, PxU32 inputLayerCount)
{
	if(!inputSoftBodies || inputNumSoftBodies == 0 ||
		!inputParticleBodyIndices || inputNumParticles == 0 ||
		!inputPackedParticleIndices || !inputLayerOffsets ||
		inputLayerCount == 0)
		return false;
	if(inputLayerOffsets[0] != 0 ||
		inputLayerOffsets[inputLayerCount] != inputNumParticles)
		return false;
	for(PxU32 layer = 0; layer < inputLayerCount; layer++)
	{
		if(inputLayerOffsets[layer] > inputLayerOffsets[layer + 1] ||
			inputLayerOffsets[layer + 1] > inputNumParticles)
			return false;
	}
	solveContext = &inputSolveContext;
	softBodies = inputSoftBodies;
	numSoftBodies = inputNumSoftBodies;
	particleBodyIndices = inputParticleBodyIndices;
	numParticles = inputNumParticles;
	packedParticleIndices = inputPackedParticleIndices;
	layerOffsets = inputLayerOffsets;
	layerCount = inputLayerCount;
	currentLayer = 0;
	sweepObservation = AvbdParticlePrimalRangeObservation();
	active = true;
	return true;
}

bool AvbdParticlePrimalCausalLayerState::hasPublishedLayer() const
{
	return active && currentLayer < layerCount;
}

PxU32 AvbdParticlePrimalCausalLayerState::getPublishedLayerIndex() const
{
	PX_ASSERT(hasPublishedLayer());
	return currentLayer;
}

void AvbdParticlePrimalCausalLayerState::getPublishedPackedRange(
	PxU32& packedBegin, PxU32& packedEnd) const
{
	PX_ASSERT(hasPublishedLayer());
	packedBegin = layerOffsets[currentLayer];
	packedEnd = layerOffsets[currentLayer + 1];
}

void AvbdParticlePrimalCausalLayerState::solvePublishedLayerSerial()
{
	PX_ASSERT(hasPublishedLayer());
	PxU32 packedBegin = 0;
	PxU32 packedEnd = 0;
	getPublishedPackedRange(packedBegin, packedEnd);
	AvbdParticlePrimalRangeObservation observation;
	avbdSolveParticlePrimalPackedRange(
		*solveContext, softBodies, numSoftBodies,
		particleBodyIndices, numParticles, packedParticleIndices,
		packedBegin, packedEnd, observation);
	completePublishedLayer(&observation, 1);
}

bool AvbdParticlePrimalCausalLayerState::completePublishedLayer(
	const AvbdParticlePrimalRangeObservation* observations,
	PxU32 observationCount)
{
	if(!hasPublishedLayer() || !observations || observationCount == 0)
		return false;
	for(PxU32 observationIndex = 0;
		observationIndex < observationCount; observationIndex++)
		sweepObservation.merge(observations[observationIndex]);
	currentLayer++;
	if(currentLayer == layerCount)
		active = false;
	return true;
}

const AvbdParticlePrimalRangeObservation&
AvbdParticlePrimalCausalLayerState::getSweepObservation() const
{
	PX_ASSERT(!active);
	return sweepObservation;
}

} // namespace Dy
} // namespace physx
