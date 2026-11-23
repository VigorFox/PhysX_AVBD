#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_COOKED_MESH_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_COOKED_MESH_H

#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "extensions/PxTetMakerExt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Headless fixture IO only. All geometry generation, cooking and deserialization
// are public SDK operations. Replay supplies the same bytes to the same scene.
namespace VolumeAvbdCookedMesh
{
using namespace physx;

struct Header
{
	char magic[8];
	PxU32 sdkVersion, voxels;
	PxU64 authoredHash, payloadHash;
	PxU32 payloadBytes, reserved;
};
static_assert(sizeof(Header) == 40, "Cooked fixture header layout changed");

inline PxU64 hashBytes(const void* data, size_t size, PxU64 hash = 14695981039346656037ull)
{
	const PxU8* bytes = static_cast<const PxU8*>(data);
	for(size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
	return hash;
}

inline bool enabled()
{
	return std::getenv("PHYSX_AVBD_MESH_CAPTURE_DIR") ||
		std::getenv("PHYSX_AVBD_MESH_REPLAY_DIR");
}

inline PxDeformableVolumeMesh* error(const char* name, const char* reason)
{
	std::fprintf(stderr, "[AVBD_MESH_IO_ERROR] name=%s reason=%s\n", name, reason);
	return NULL;
}

inline PxDeformableVolumeMesh* create(PxPhysics& physics,
	const PxArray<PxVec3>& vertices, const PxArray<PxU32>& triangles,
	PxU32 voxels, const char* name)
{
	const char* capture = std::getenv("PHYSX_AVBD_MESH_CAPTURE_DIR");
	const char* replay = std::getenv("PHYSX_AVBD_MESH_REPLAY_DIR");
	if((capture && replay) || (!capture && !replay) ||
		(capture && !capture[0]) || (replay && !replay[0]))
		return error(name, "invalid_mode");
	const std::string path = std::string(capture ? capture : replay) + "/" + name + ".pxvol";
	const PxU64 authoredHash = hashBytes(triangles.begin(), triangles.size() * sizeof(PxU32),
		hashBytes(vertices.begin(), vertices.size() * sizeof(PxVec3)));
	Header header = {};
	PxArray<PxU8> payload;
	if(capture)
	{
		// The SDK extension's public surface -> conforming tets -> voxel tets
		// recipe, with the final cook directed to a stream instead of insertion.
		PxSimpleTriangleMesh surface;
		surface.points.count = vertices.size(); surface.points.stride = sizeof(PxVec3);
		surface.points.data = vertices.begin();
		surface.triangles.count = triangles.size() / 3;
		surface.triangles.stride = 3 * sizeof(PxU32); surface.triangles.data = triangles.begin();
		PxArray<PxVec3> collisionVertices, simulationVertices;
		PxArray<PxU32> collisionIndices, simulationIndices;
		if(!PxTetMaker::createConformingTetrahedronMesh(surface, collisionVertices, collisionIndices, true))
			return error(name, "tetrahedralization");
		PxTetrahedronMeshDesc collision(collisionVertices, collisionIndices);
		PxArray<PxI32> vertexToTet;
		vertexToTet.resize(collision.points.count);
		if(!PxTetMaker::createVoxelTetrahedronMesh(collision, voxels,
			simulationVertices, simulationIndices, vertexToTet.begin()))
			return error(name, "voxelization");
		PxTetrahedronMeshDesc simulation(simulationVertices, simulationIndices,
			PxTetrahedronMeshDesc::PxMeshFormat::eHEX_MESH);
		PxCookingParams params(physics.getTolerancesScale());
		const char* gpu = std::getenv("PHYSX_AVBD_MESH_CAPTURE_GPU_DATA");
		params.buildGPUData = gpu && std::strcmp(gpu, "1") == 0;
		params.meshWeldTolerance = 0.001f;
		params.meshPreprocessParams = PxMeshPreprocessingFlag::eWELD_VERTICES;
		PxDefaultMemoryOutputStream stream;
		if(!PxCookDeformableVolumeMesh(params, simulation, collision,
			PxDeformableVolumeSimulationDataDesc(vertexToTet), stream))
			return error(name, "cooking");
		if(stream.getSize() == 0 || stream.getSize() > PX_MAX_U32)
			return error(name, "payload_size");
		const PxU32 payloadSize = static_cast<PxU32>(stream.getSize());
		std::memcpy(header.magic, "AVBDMV1", sizeof(header.magic));
		header.sdkVersion = PX_PHYSICS_VERSION; header.voxels = voxels;
		header.authoredHash = authoredHash; header.payloadBytes = payloadSize;
		header.payloadHash = hashBytes(stream.getData(), payloadSize);
		payload.resize(payloadSize);
		std::memcpy(payload.begin(), stream.getData(), payloadSize);
		FILE* existing = std::fopen(path.c_str(), "rb");
		if(existing) { std::fclose(existing); return error(name, "capture_exists"); }
		FILE* file = std::fopen(path.c_str(), "wb");
		if(!file) return error(name, "capture_open");
		const bool written = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
			std::fwrite(payload.begin(), 1, payload.size(), file) == payload.size();
		const bool closed = std::fclose(file) == 0;
		if(!written || !closed) return error(name, "capture_write");
	}
	else
	{
		FILE* file = std::fopen(path.c_str(), "rb");
		if(!file) return error(name, "replay_open");
		const bool readHeader = std::fread(&header, sizeof(header), 1, file) == 1;
		std::fseek(file, 0, SEEK_END);
		const long size = std::ftell(file);
		const bool valid = readHeader && size >= long(sizeof(header)) &&
			std::memcmp(header.magic, "AVBDMV1", sizeof(header.magic)) == 0 &&
			header.sdkVersion == PX_PHYSICS_VERSION && header.voxels == voxels &&
			header.authoredHash == authoredHash && header.reserved == 0 &&
			header.payloadBytes > 0 && PxU64(size) == sizeof(header) + PxU64(header.payloadBytes);
		if(!valid) { std::fclose(file); return error(name, "replay_header"); }
		payload.resize(header.payloadBytes);
		std::fseek(file, sizeof(header), SEEK_SET);
		const bool readPayload = std::fread(payload.begin(), 1, payload.size(), file) == payload.size();
		std::fclose(file);
		if(!readPayload || hashBytes(payload.begin(), payload.size()) != header.payloadHash)
			return error(name, "replay_checksum");
	}
	PxDefaultMemoryInputData input(payload.begin(), payload.size());
	PxDeformableVolumeMesh* mesh = physics.createDeformableVolumeMesh(input);
	if(!mesh) return error(name, "deserialization");
	std::printf("[AVBD_COOKED_MESH] name=%s mode=%s voxels=%u authoredHash=%016llx bytes=%u payloadHash=%016llx\n",
		name, capture ? "capture" : "replay", voxels,
		static_cast<unsigned long long>(authoredHash), header.payloadBytes,
		static_cast<unsigned long long>(header.payloadHash));
	std::fflush(stdout);
	return mesh;
}
}
#endif
