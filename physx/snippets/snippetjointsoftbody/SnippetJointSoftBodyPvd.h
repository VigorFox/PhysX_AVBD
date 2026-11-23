// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "PxPhysicsAPI.h"
#include "foundation/PxSort.h"
#include "../snippetcommon/SnippetPVD.h"
#include <cstdio>
#include <cstdlib>

namespace JointSoftBodySnippet
{
// Presentation only. Physics actors/shapes use the standard SDK PVD stream;
// the same read-only display skin used by the window uses unique wire edges.
class PvdVisualization
{
	physx::PxPvd* mPvd;
	physx::PxPvdTransport* mTransport;
	struct MeshEdges
	{
		const void* topology;
		physx::PxArray<physx::PxU64> edges;
	};
	physx::PxArray<MeshEdges> mMeshes;
	physx::PxArray<physx::PxDebugLine> mLines;
	static physx::PxVec3 xyz(const physx::PxVec3& p) { return p; }
	static physx::PxVec3 xyz(const physx::PxVec4& p) { return p.getXYZ(); }
public:
	PvdVisualization() : mPvd(NULL),mTransport(NULL) {}
	physx::PxPvd* get() const { return mPvd; }
	bool connected() const { return mPvd && mPvd->isConnected(); }
	bool init(physx::PxFoundation& foundation,bool live)
	{
		using namespace physx;
		const char* path=std::getenv("PHYSX_AVBD_PVD_CAPTURE_PATH");
		const bool capture=path && path[0];
		if(!live && !capture) return true;
		mPvd=PxCreatePvd(foundation);
		mTransport=capture?PxDefaultPvdFileTransportCreate(path):PxDefaultPvdSocketTransportCreate(PVD_HOST,5425,10);
		const bool ok=mPvd && mTransport && mPvd->connect(*mTransport,PxPvdInstrumentationFlag::eDEBUG);
		std::printf("[JOINT_PVD] transport=%s target=%s connected=%u\n",capture?"file":"socket",capture?path:PVD_HOST ":5425",unsigned(ok));
		if(!ok)
		{
			release();
			if(capture) std::fprintf(stderr,"[JOINT_PVD_ERROR] Cannot open the requested PVD capture.\n");
		}
		// A visual snippet also runs when no debugger is listening. An explicitly
		// requested file capture must not silently succeed without a recording.
		return ok || !capture;
	}
	void release()
	{
		mLines.clear();mLines.shrink();
		mMeshes.clear();mMeshes.shrink();
		PX_RELEASE(mPvd);
		PX_RELEASE(mTransport);
	}
	void configureScene(physx::PxScene& scene)
	{
		using namespace physx;
		if(PxPvdSceneClient* client=scene.getScenePvdClient())
		{
			client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS,true);
			client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS,true);
			client->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES,true);
		}
	}
	void wireMesh(physx::PxScene& scene,const physx::PxArray<physx::PxVec3>& vertices,
		const physx::PxArray<physx::PxU32>& indices,physx::PxU32 color)
	{
		drawWireMesh(scene, &indices, vertices.begin(), indices.begin(), indices.size(), color);
	}
	void wireMesh(physx::PxScene& scene, const physx::PxTriangleMesh& mesh,
		const physx::PxVec4* positions, physx::PxU32 color)
	{
		using namespace physx;
		const PxU32 count = 3 * mesh.getNbTriangles();
		if(mesh.getTriangleMeshFlags().isSet(PxTriangleMeshFlag::e16_BIT_INDICES))
			drawWireMesh(scene, &mesh, positions, static_cast<const PxU16*>(mesh.getTriangles()), count, color);
		else
			drawWireMesh(scene, &mesh, positions, static_cast<const PxU32*>(mesh.getTriangles()), count, color);
	}
private:
	template<class Vertex, class Index>
	void drawWireMesh(physx::PxScene& scene, const void* topology, const Vertex* vertices,
		const Index* indices, physx::PxU32 count, physx::PxU32 color)
	{
		using namespace physx;
		if(!connected() || !vertices || !indices || !count) return;
		PxPvdSceneClient* client=scene.getScenePvdClient();
		if(!client) return;
		PxU32 mesh=0;
		while(mesh<mMeshes.size() && mMeshes[mesh].topology!=topology) ++mesh;
		if(mesh==mMeshes.size())
		{
			// Authored display topology is immutable for a World lifetime. Cache
			// only its unique edges; release() invalidates all entries on reset.
			mMeshes.pushBack(MeshEdges());mMeshes.back().topology=topology;
			PxArray<PxU64>& edges=mMeshes.back().edges;
			for(PxU32 i=0;i+2<count;i+=3) for(PxU32 e=0;e<3;++e)
			{
				const PxU32 a=indices[i+e],b=indices[i+(e+1)%3];
				edges.pushBack((PxU64(PxMin(a,b))<<32)|PxMax(a,b));
			}
			PxSort(edges.begin(),edges.size());
			PxU32 unique=0;
			for(PxU32 i=0;i<edges.size();++i)
				if(!unique || edges[i]!=edges[unique-1]) edges[unique++]=edges[i];
			edges.resize(unique);
		}
		mLines.clear();
		const PxArray<PxU64>& edges=mMeshes[mesh].edges;
		for(PxU32 i=0;i<edges.size();++i)
			mLines.pushBack(PxDebugLine(xyz(vertices[PxU32(edges[i]>>32)]),xyz(vertices[PxU32(edges[i])]),color));
		client->drawLines(mLines.begin(),mLines.size());
	}
};
}
