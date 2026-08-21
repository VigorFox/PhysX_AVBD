// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyCreation.h"

namespace physx
{
namespace Dy
{

PxU32 avbdCreateSoftBody(
	const PxVec3* vertices, PxU32 numVertices,
	const PxU32* tets, PxU32 numTetIndices,
	const PxU32* tris, PxU32 numTriIndices,
	PxReal youngsModulus, PxReal poissonsRatio,
	PxReal density, PxReal damping,
	PxReal bendingStiffness, PxReal thickness,
	PxArray<AvbdSoftParticle>& outParticles,
	PxArray<AvbdSoftBody>& outSoftBodies,
	bool flatteningEnabled,
	PxReal selfCollisionFilterDistance,
	PxReal dynamicFriction,
	bool coRotationalVolumeModel)
{
	const PxU32 particleStart = outParticles.size();

	PxArray<PxReal> vertexMass;
	vertexMass.resize(numVertices, 0.0f);

	if(numTetIndices > 0)
	{
		for(PxU32 i = 0; i + 3 < numTetIndices; i += 4)
		{
			const PxVec3 e1 = vertices[tets[i + 1]] - vertices[tets[i]];
			const PxVec3 e2 = vertices[tets[i + 2]] - vertices[tets[i]];
			const PxVec3 e3 = vertices[tets[i + 3]] - vertices[tets[i]];
			const PxReal volume = PxAbs(e1.dot(e2.cross(e3)) / 6.0f);
			const PxReal perVertexMass = volume * density * 0.25f;
			vertexMass[tets[i]] += perVertexMass;
			vertexMass[tets[i + 1]] += perVertexMass;
			vertexMass[tets[i + 2]] += perVertexMass;
			vertexMass[tets[i + 3]] += perVertexMass;
		}
	}
	else if(numTriIndices > 0)
	{
		for(PxU32 i = 0; i + 2 < numTriIndices; i += 3)
		{
			const PxVec3 e1 = vertices[tris[i + 1]] - vertices[tris[i]];
			const PxVec3 e2 = vertices[tris[i + 2]] - vertices[tris[i]];
			const PxReal area = e1.cross(e2).magnitude() * 0.5f;
			const PxReal perVertexMass = area * thickness * density / 3.0f;
			vertexMass[tris[i]] += perVertexMass;
			vertexMass[tris[i + 1]] += perVertexMass;
			vertexMass[tris[i + 2]] += perVertexMass;
		}
	}

	for(PxU32 i = 0; i < numVertices; ++i)
		vertexMass[i] = PxMax(vertexMass[i], 1.0e-4f);

	PxReal maxMass = 0.0f;
	for(PxU32 i = 0; i < numVertices; ++i)
		maxMass = PxMax(maxMass, vertexMass[i]);
	const PxReal massFloor = maxMass / 50.0f;
	for(PxU32 i = 0; i < numVertices; ++i)
		vertexMass[i] = PxMax(vertexMass[i], massFloor);

	for(PxU32 i = 0; i < numVertices; ++i)
	{
		AvbdSoftParticle particle;
		particle.position = vertices[i];
		particle.velocity = PxVec3(0.0f);
		particle.prevVelocity = PxVec3(0.0f);
		particle.initialPosition = vertices[i];
		particle.predictedPosition = vertices[i];
		particle.mass = vertexMass[i];
		particle.invMass = 1.0f / particle.mass;
		particle.damping = damping;
		outParticles.pushBack(particle);
	}

	AvbdSoftBody body;
	body.compiled.particleStart = particleStart;
	body.compiled.particleCount = numVertices;
	body.compiled.selfCollisionRestPositions.resize(numVertices);
	for(PxU32 i = 0; i < numVertices; ++i)
		body.compiled.selfCollisionRestPositions[i] = vertices[i];
	body.compiled.selfCollisionFilterDistance = PxMax(selfCollisionFilterDistance, 0.0f);

	for(PxU32 i = 0; i < numTetIndices; ++i)
		body.compiled.tetrahedra.pushBack(tets[i]);
	for(PxU32 i = 0; i < numTriIndices; ++i)
		body.compiled.triangles.pushBack(tris[i]);

	body.material.youngsModulus = youngsModulus;
	body.material.poissonsRatio = poissonsRatio;
	body.material.density = density;
	body.material.damping = damping;
	body.material.bendingStiffness = bendingStiffness;
	body.material.thickness = thickness;
	body.material.dynamicFriction = PxMax(dynamicFriction, 0.0f);
	body.material.coRotationalVolumeModel = coRotationalVolumeModel;

	body.buildElements(outParticles);
	body.compiled.compileBendingRestAngles(flatteningEnabled);

	outSoftBodies.pushBack(body);
	return particleStart;
}

} // namespace Dy
} // namespace physx
