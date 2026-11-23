#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_PVD_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_PVD_H

#include "PxPhysicsAPI.h"
#include "foundation/PxSort.h"
#include "pvd/PxPvdSceneClient.h"

// Publish unique simulation tetrahedron edges from the live Particle3 positions.
class VolumeAvbdPvdMeshVisualizer
{
	struct VolumeCache
	{
		const physx::PxTetrahedronMesh* mesh;
		physx::PxArray<physx::PxU64> edges;
	};

	physx::PxArray<VolumeCache> mVolumes;
	physx::PxArray<physx::PxDebugLine> mDebugLines;

	const physx::PxArray<physx::PxU64>* getVolumeEdges(
		const physx::PxTetrahedronMesh& mesh)
	{
		for(physx::PxU32 i = 0; i < mVolumes.size(); ++i)
			if(mVolumes[i].mesh == &mesh)
				return &mVolumes[i].edges;

		VolumeCache cache;
		cache.mesh = &mesh;
		cache.edges.reserve(mesh.getNbTetrahedrons() * 6);
		const bool indices16 = mesh.getTetrahedronMeshFlags().isSet(
			physx::PxTetrahedronMeshFlag::e16_BIT_INDICES);
		const physx::PxU16* tetrahedra16 = indices16
			? static_cast<const physx::PxU16*>(mesh.getTetrahedrons()) : NULL;
		const physx::PxU32* tetrahedra32 = indices16
			? NULL : static_cast<const physx::PxU32*>(mesh.getTetrahedrons());
		static const physx::PxU32 edgeVertices[6][2] =
		{
			{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
		};
		for(physx::PxU32 tetrahedron = 0;
			tetrahedron < mesh.getNbTetrahedrons(); ++tetrahedron)
		{
			physx::PxU32 vertices[4];
			for(physx::PxU32 vertex = 0; vertex < 4; ++vertex)
				vertices[vertex] = indices16
					? physx::PxU32(tetrahedra16[4 * tetrahedron + vertex])
					: tetrahedra32[4 * tetrahedron + vertex];
			for(physx::PxU32 edge = 0; edge < 6; ++edge)
			{
				const physx::PxU32 a = vertices[edgeVertices[edge][0]];
				const physx::PxU32 b = vertices[edgeVertices[edge][1]];
				const physx::PxU32 lo = physx::PxMin(a, b);
				const physx::PxU32 hi = physx::PxMax(a, b);
				cache.edges.pushBack((physx::PxU64(lo) << 32) | hi);
			}
		}
		physx::PxSort(cache.edges.begin(), cache.edges.size());
		physx::PxU32 uniqueCount = 0;
		for(physx::PxU32 edge = 0; edge < cache.edges.size(); ++edge)
		{
			if(uniqueCount && cache.edges[edge] == cache.edges[uniqueCount - 1])
				continue;
			cache.edges[uniqueCount++] = cache.edges[edge];
		}
		cache.edges.resize(uniqueCount);
		mVolumes.pushBack(cache);
		return &mVolumes.back().edges;
	}

public:
	void clear()
	{
		// PxArray uses the PhysX foundation allocator.  Release capacity while the
		// foundation is still alive; a size-only clear would defer deallocation to
		// this process-global helper's destructor, after PxFoundation::release().
		mDebugLines.clear();
		mDebugLines.shrink();
		for(physx::PxU32 i = 0; i < mVolumes.size(); ++i)
		{
			mVolumes[i].edges.clear();
			mVolumes[i].edges.shrink();
		}
		mVolumes.clear();
		mVolumes.shrink();
	}

	void send(physx::PxScene& scene, physx::PxDeformableVolume*const* volumes,
		physx::PxU32 volumeCount)
	{
		physx::PxPvdSceneClient* client = scene.getScenePvdClient();
		if(!client)
			return;
		static const physx::PxU32 colors[] =
		{
			0xFFFF8000, 0xFF0080FF, 0xFF00FF80, 0xFFFFD020, 0xFFCC4DB3
		};
		mDebugLines.clear();
		for(physx::PxU32 volumeIndex = 0; volumeIndex < volumeCount; ++volumeIndex)
		{
			physx::PxDeformableVolume* volume = volumes[volumeIndex];
			if(!volume)
				continue;
			const physx::PxU32 color = colors[volumeIndex % PX_ARRAY_SIZE(colors)];
			const physx::PxTetrahedronMesh* simulationMesh =
				volume->getSimulationMesh();
			const physx::PxVec4* simulationPositions =
				volume->getSimPositionInvMassBufferD();
			if(!simulationMesh || !simulationPositions)
				continue;
			const physx::PxArray<physx::PxU64>* volumeEdges =
				getVolumeEdges(*simulationMesh);
			const physx::PxU32 simulationVertexCount =
				simulationMesh->getNbVertices();
			for(physx::PxU32 edge = 0; edge < volumeEdges->size(); ++edge)
			{
				const physx::PxU32 i0 = physx::PxU32((*volumeEdges)[edge] >> 32);
				const physx::PxU32 i1 = physx::PxU32((*volumeEdges)[edge]);
				if(i0 >= simulationVertexCount || i1 >= simulationVertexCount)
					continue;
				const physx::PxVec3 p0 = simulationPositions[i0].getXYZ();
				const physx::PxVec3 p1 = simulationPositions[i1].getXYZ();
				if(p0.isFinite() && p1.isFinite())
					mDebugLines.pushBack(
						physx::PxDebugLine(p0, p1, color));
			}
		}
		if(mDebugLines.size())
			client->drawLines(mDebugLines.begin(), mDebugLines.size());
	}
};

#endif
