// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

AvbdTetMaterialPacketKernels avbdSelectTetMaterialPacketKernels(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	AvbdTetMaterialPacketKernels kernels = { NULL, NULL };
	const AvbdCpuIsaCorotationalTetPacket8Fn corotational =
		avbdGetCorotationalTetPacketKernel();
	const AvbdCpuIsaNeoHookeanTetPacket8Fn neoHookean =
		avbdGetNeoHookeanTetPacketKernel();
	if(!corotational && !neoHookean)
		return kernels;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.tetIncidencePacketProgramValid ||
			!body.compiled.tetIncidenceFullPacketCount)
			continue;
		if(body.material.coRotationalVolumeModel)
			kernels.corotational = corotational;
		else
			kernels.neoHookean = neoHookean;
	}
	return kernels;
}

bool avbdApplySoftBodyRigidPrimalInitialGuess(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody& body)
{
	const PxU32 particleStart = body.compiled.particleStart;
	const PxU32 particleCount = body.compiled.particleCount;
	if(!particles || particleCount == 0 || particleStart > numParticles ||
		particleCount > numParticles - particleStart)
		return false;

	PxReal totalMass = 0.0f;
	PxVec3 initialCentroid(0.0f);
	PxVec3 predictedCentroid(0.0f);
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		const AvbdSoftParticle& particle = particles[
			particleStart + localIndex];
		if(!PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
			!avbdIsFiniteVector(particle.initialPosition) ||
			!avbdIsFiniteVector(particle.predictedPosition))
			return false;
		totalMass += particle.mass;
		initialCentroid += particle.initialPosition * particle.mass;
		predictedCentroid += particle.predictedPosition * particle.mass;
	}
	if(!PxIsFinite(totalMass) || totalMass <= 1.0e-12f ||
		!avbdIsFiniteVector(initialCentroid) ||
		!avbdIsFiniteVector(predictedCentroid))
		return false;
	const PxReal inverseMass = 1.0f / totalMass;
	initialCentroid *= inverseMass;
	predictedCentroid *= inverseMass;

	PxMat33 covariance(PxZero);
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		const AvbdSoftParticle& particle = particles[
			particleStart + localIndex];
		covariance += avbdOuter(
			particle.predictedPosition - predictedCentroid,
			particle.initialPosition - initialCentroid) * particle.mass;
	}
	if(!avbdIsFiniteVector(covariance.column0) ||
		!avbdIsFiniteVector(covariance.column1) ||
		!avbdIsFiniteVector(covariance.column2))
		return false;
	const PxMat33 rotation = avbdExtractCorotationalRotation(covariance);
	if(!avbdIsFiniteVector(rotation.column0) ||
		!avbdIsFiniteVector(rotation.column1) ||
		!avbdIsFiniteVector(rotation.column2))
		return false;

	// Validate every transformed point before writing any one of them: a bad
	// diagnostic fit must retain the ordinary prediction start wholesale.
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		const AvbdSoftParticle& particle = particles[
			particleStart + localIndex];
		const PxVec3 rigidPosition = predictedCentroid + rotation *
			(particle.initialPosition - initialCentroid);
		if(!avbdIsFiniteVector(rigidPosition))
			return false;
	}
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		AvbdSoftParticle& particle = particles[particleStart + localIndex];
		particle.position = predictedCentroid + rotation *
			(particle.initialPosition - initialCentroid);
	}
	return true;
}

void avbdApplyBendingDamping(
	AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxReal dt)
{
	if(!particles || !softBodies || dt <= 0.0f)
		return;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const PxReal dampingFactor = PxClamp(
			body.material.bendingDamping * dt,
			0.0f, 1.0f);
		if(dampingFactor <= 0.0f ||
			body.material.bendingStiffness <= 0.0f ||
			body.compiled.bendElements.empty())
			continue;
		PxArray<PxVec3> deltaVelocities(
			body.compiled.particleCount);
		for(PxU32 localIndex = 0;
			localIndex < deltaVelocities.size(); localIndex++)
			deltaVelocities[localIndex] = PxVec3(0.0f);

		for(PxU32 bendingIndex = 0;
			bendingIndex < body.compiled.bendElements.size();
			bendingIndex++)
		{
			const AvbdBendingElement& bending =
				body.compiled.bendElements[bendingIndex];
			const PxU32 edgeStart = bending.edgeStart;
			const PxU32 edgeEnd = bending.edgeEnd;
			const PxU32 tip0 = bending.opp0;
			const PxU32 tip1 = bending.opp1;
			const PxU32 bodyEnd =
				body.compiled.particleStart +
				body.compiled.particleCount;
			if(edgeStart < body.compiled.particleStart ||
				edgeEnd < body.compiled.particleStart ||
				tip0 < body.compiled.particleStart ||
				tip1 < body.compiled.particleStart ||
				edgeStart >= bodyEnd || edgeEnd >= bodyEnd ||
				tip0 >= bodyEnd || tip1 >= bodyEnd)
				continue;

			const PxVec3 linearVelocity =
				(particles[edgeStart].velocity +
				 particles[edgeEnd].velocity) * 0.5f;
			PxVec3 edgeDirection =
				particles[edgeEnd].position -
				particles[edgeStart].position;
			if(edgeDirection.normalize() < 1.0e-6f)
				continue;
			PxVec3 tipDirection0 =
				edgeDirection.cross(
					particles[tip0].position -
					particles[edgeStart].position);
			PxVec3 tipDirection1 =
				edgeDirection.cross(
					particles[tip1].position -
					particles[edgeStart].position);
			const PxReal tipDistance0 = tipDirection0.normalize();
			const PxReal tipDistance1 = tipDirection1.normalize();
			if(tipDistance0 < 1.0e-6f ||
				tipDistance1 < 1.0e-6f)
				continue;
			const PxReal angularVelocity0 =
				tipDirection0.dot(
					particles[tip0].velocity -
					linearVelocity) /
				tipDistance0;
			const PxReal angularVelocity1 =
				tipDirection1.dot(
					particles[tip1].velocity -
					linearVelocity) /
				tipDistance1;
			const PxReal dampedAngularDifference =
				(angularVelocity1 - angularVelocity0) *
				dampingFactor;
			PxVec3 deltaEdgeStart(0.0f);
			PxVec3 deltaEdgeEnd(0.0f);
			PxVec3 deltaTip0 =
				tipDirection0 *
					(dampedAngularDifference * tipDistance0);
			PxVec3 deltaTip1 =
				tipDirection1 *
					(-dampedAngularDifference * tipDistance1);
			const PxReal inverseMassSum =
				particles[edgeStart].invMass +
				particles[edgeEnd].invMass +
				particles[tip0].invMass +
				particles[tip1].invMass;
			if(inverseMassSum <= 1.0e-12f)
				continue;
			const PxVec3 averageDelta =
				(deltaEdgeStart + deltaEdgeEnd +
				 deltaTip0 + deltaTip1) * 0.25f;
			deltaEdgeStart -= averageDelta;
			deltaEdgeEnd -= averageDelta;
			deltaTip0 -= averageDelta;
			deltaTip1 -= averageDelta;
			const PxReal weightFactor =
				1.0f / inverseMassSum;
			deltaVelocities[
				edgeStart - body.compiled.particleStart] +=
					deltaEdgeStart *
					(particles[edgeStart].invMass * weightFactor);
			deltaVelocities[
				edgeEnd - body.compiled.particleStart] +=
					deltaEdgeEnd *
					(particles[edgeEnd].invMass * weightFactor);
			deltaVelocities[
				tip0 - body.compiled.particleStart] +=
					deltaTip0 *
					(particles[tip0].invMass * weightFactor);
			deltaVelocities[
				tip1 - body.compiled.particleStart] +=
					deltaTip1 *
					(particles[tip1].invMass * weightFactor);
		}
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount;
			localIndex++)
		{
			AvbdSoftParticle& particle =
				particles[
					body.compiled.particleStart + localIndex];
			if(particle.invMass > 0.0f)
				particle.velocity +=
					deltaVelocities[localIndex];
		}
	}
}

} // namespace Dy
} // namespace physx
