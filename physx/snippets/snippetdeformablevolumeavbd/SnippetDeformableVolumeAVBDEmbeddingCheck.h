#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_EMBEDDING_CHECK_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_EMBEDDING_CHECK_H

#include "PxPhysicsAPI.h"
#include "extensions/PxDeformableVolumeExt.h"
#include <cfloat>
#include <limits>

// Headless-only publication oracle. The SDK extension evaluates the cooked
// mapping independently of AVBD's Host writeback, into private scratch buffers.
// This checks publication, not the correctness of the cooking weights themselves.
struct VolumeEmbeddingCheck
{
	physx::PxArray<physx::PxVec4> simulation, reference;
	physx::PxF64 maxVertexError = 0.0, maxErrorRatio = 0.0;
	physx::PxU64 samples = 0;
	physx::PxU32 invalid = 0;

	void compare(const physx::PxVec4* actual, const physx::PxVec4* expected,
		physx::PxU32 count)
	{
		using namespace physx;
		if(!actual || !expected || !count) { ++invalid; return; }
		for(PxU32 vertex = 0; vertex < count; ++vertex)
		{
			++samples;
			if(!actual[vertex].isFinite() || !expected[vertex].isFinite())
			{ ++invalid; continue; }
			PxF64 squaredError = 0.0;
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				const PxF64 a = actual[vertex][axis], b = expected[vertex][axis];
				const PxF64 error = PxAbs(a - b);
				// Eight float roundoff units allow contraction/operation ordering
				// in a four-term interpolation. This is not a physical tolerance.
				const PxF64 bound = 8.0 * FLT_EPSILON * PxMax(1.0, PxMax(PxAbs(a), PxAbs(b)));
				maxErrorRatio = PxMax(maxErrorRatio, error / bound);
				squaredError += error * error;
			}
			maxVertexError = PxMax(maxVertexError, PxSqrt(squaredError));
		}
	}

	void sample(physx::PxDeformableVolume& volume)
	{
		using namespace physx;
		const PxTetrahedronMesh* simMesh = volume.getSimulationMesh();
		const PxTetrahedronMesh* collMesh = volume.getCollisionMesh();
		const PxVec4* sim = volume.getSimPositionInvMassBufferD();
		const PxVec4* coll = volume.getPositionInvMassBufferD();
		if(!simMesh || !collMesh || !sim || !coll || !volume.getDeformableVolumeAuxData())
		{ ++invalid; return; }
		simulation.resize(simMesh->getNbVertices());
		reference.resize(collMesh->getNbVertices());
		for(PxU32 vertex = 0; vertex < simulation.size(); ++vertex)
		{
			if(!sim[vertex].isFinite()) { ++invalid; return; }
			simulation[vertex] = sim[vertex];
		}
		PxDeformableVolumeExt::updateEmbeddedCollisionMesh(volume,
			simulation.begin(), reference.begin());
		compare(coll, reference.begin(), reference.size());
	}

	// Independent rest geometry and affine partition checks. These use raw mesh
	// vertices, not collision positions already initialized by the same mapping.
	void sampleRest(physx::PxDeformableVolume& volume, bool partitionOnly)
	{
		using namespace physx;
		const PxTetrahedronMesh* sim = volume.getSimulationMesh();
		const PxTetrahedronMesh* coll = volume.getCollisionMesh();
		if(!sim || !coll || !sim->getVertices() || !coll->getVertices())
		{ ++invalid; return; }
		simulation.resize(sim->getNbVertices()); reference.resize(coll->getNbVertices());
		PxArray<PxVec4> expected(coll->getNbVertices());
		for(PxU32 vertex = 0; vertex < simulation.size(); ++vertex)
			simulation[vertex] = PxVec4(partitionOnly ? PxVec3(1.0f) : sim->getVertices()[vertex], 1.0f);
		for(PxU32 vertex = 0; vertex < expected.size(); ++vertex)
			expected[vertex] = PxVec4(partitionOnly ? PxVec3(1.0f) : coll->getVertices()[vertex], 1.0f);
		PxDeformableVolumeExt::updateEmbeddedCollisionMesh(volume, simulation.begin(), reference.begin());
		compare(reference.begin(), expected.begin(), reference.size());
	}

	bool passed() const { return samples > 0 && invalid == 0 && maxErrorRatio <= 1.0; }

	static bool detectsCorruption()
	{
		using namespace physx;
		const PxVec4 expected[2] = {PxVec4(0.0f, 1.0f, 0.0f, 1.0f),
			PxVec4(0.0f, 2.0f, 0.0f, 1.0f)};
		VolumeEmbeddingCheck clean, cancelling, nonFinite, empty;
		clean.compare(expected, expected, 2);
		PxVec4 actual[2] = {expected[0], expected[1]};
		actual[0].x += 0.01f;
		actual[1].x -= 0.01f; // Same centroid: the former oracle misses this.
		cancelling.compare(actual, expected, 2);
		actual[0].x = std::numeric_limits<PxReal>::quiet_NaN();
		nonFinite.compare(actual, expected, 2);
		empty.compare(NULL, expected, 2);
		return clean.passed() && !cancelling.passed() && !nonFinite.passed() && !empty.passed();
	}
};

#endif
