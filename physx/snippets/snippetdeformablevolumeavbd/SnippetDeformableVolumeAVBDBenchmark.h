#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_BENCHMARK_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_BENCHMARK_H

#include "PxPhysicsAPI.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Diagnostic export of the actual cooked Scene input. No solver or scene policy
// is changed. The independent Jolt runner reads this file; it does not recook it.
class VolumeAvbdBenchmark
{
public:
	VolumeAvbdBenchmark() : mTrace(NULL), mVolume(NULL), mOk(true) {}
	~VolumeAvbdBenchmark() { if(mTrace) std::fclose(mTrace); }
	bool begin(physx::PxScene& scene, physx::PxDeformableVolume& volume,
		const char* caseName, float dt)
	{
		using namespace physx;
		const char* scenePath = std::getenv("PHYSX_AVBD_BENCH_SCENE");
		const char* tracePath = std::getenv("PHYSX_AVBD_BENCH_TRACE");
		if(!scenePath && !tracePath) return true;
		if(!scenePath || !tracePath || (std::strcmp(caseName, "rigid-on-soft") &&
			std::strcmp(caseName, "rigid-on-soft-separated") &&
			std::strcmp(caseName, "embedded-sync"))) return false;
		mVolume = &volume;
		const PxActorTypeFlags types = PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC;
		std::vector<PxActor*> actors(scene.getNbActors(types));
		scene.getActors(types, actors.data(), PxU32(actors.size()));
		for(PxActor* actor : actors) mRigids.push_back(static_cast<PxRigidActor*>(actor));
		FILE* file = std::fopen(scenePath, "w");
		if(!file) return false;
		const PxTetrahedronMesh& mesh = *volume.getSimulationMesh();
		PxDeformableVolumeMaterial* material = NULL;
		volume.getShape()->getDeformableVolumeMaterials(&material, 1);
		if(!material) { std::fclose(file); return false; }
		const PxVec3 gravity = scene.getGravity();
		std::fprintf(file, "AVBD_JOLT_SCENE 1\ncase %s\ndt %.9g\ngravity %.9g %.9g %.9g\n",
			caseName, double(dt), double(gravity.x), double(gravity.y), double(gravity.z));
		std::fprintf(file, "soft %u %u %u %u %.9g %.9g %.9g %.9g %.9g %.9g\nvertices\n",
			mesh.getNbVertices(), mesh.getNbTetrahedrons(), volume.getCollisionMesh()->getNbVertices(),
			volume.getCollisionMesh()->getNbTetrahedrons(), double(material->getYoungsModulus()),
			double(material->getPoissons()), double(material->getDynamicFriction()),
			double(material->getElasticityDamping()), double(volume.getLinearDamping()),
			double(volume.getMaxLinearVelocity()));
		const PxVec4* positions = volume.getSimPositionInvMassBufferD();
		const PxVec4* velocities = volume.getSimVelocityBufferD();
		for(PxU32 i = 0; i < mesh.getNbVertices(); ++i)
			std::fprintf(file, "%.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
				double(positions[i].x), double(positions[i].y), double(positions[i].z), double(positions[i].w),
				double(velocities[i].x), double(velocities[i].y), double(velocities[i].z));
		std::fprintf(file, "tets\n");
		const bool small = mesh.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES;
		for(PxU32 i = 0; i < mesh.getNbTetrahedrons(); ++i)
		{
			PxU32 ids[4];
			for(PxU32 j = 0; j < 4; ++j)
				ids[j] = small ? static_cast<const PxU16*>(mesh.getTetrahedrons())[4*i+j] :
					static_cast<const PxU32*>(mesh.getTetrahedrons())[4*i+j];
			std::fprintf(file, "%u %u %u %u\n", ids[0], ids[1], ids[2], ids[3]);
		}
		std::fprintf(file, "rigids %u\n", unsigned(mRigids.size()));
		for(PxRigidActor* actor : mRigids)
		{
			PxShape* shape = NULL;
			actor->getShapes(&shape, 1);
			if(!shape || actor->getNbShapes() != 1) { mOk = false; break; }
			const PxGeometry& geometry = shape->getGeometry();
			const bool box = geometry.getType() == PxGeometryType::eBOX;
			if(!box && geometry.getType() != PxGeometryType::ePLANE) { mOk = false; break; }
			const PxVec3 half = box ? static_cast<const PxBoxGeometry&>(geometry).halfExtents : PxVec3(0);
			const PxTransform pose = actor->getGlobalPose() * shape->getLocalPose();
			PxRigidDynamic* body = actor->is<PxRigidDynamic>();
			PxMaterial* mat = NULL;
			shape->getMaterials(&mat, 1);
			const PxVec3 inertia = body ? body->getMassSpaceInertiaTensor() : PxVec3(0);
			std::fprintf(file, "%s %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
				box ? "box" : "plane", body ? 1u : 0u,
				double(pose.p.x), double(pose.p.y), double(pose.p.z), double(pose.q.x), double(pose.q.y), double(pose.q.z), double(pose.q.w),
				double(half.x), double(half.y), double(half.z), double(body ? body->getMass() : 0),
				double(mat->getDynamicFriction()), double(mat->getRestitution()),
				double(body ? body->getLinearDamping() : 0), double(body ? body->getAngularDamping() : 0),
				double(inertia.x), double(inertia.y), double(inertia.z));
		}
		mOk = !std::ferror(file) && mOk;
		if(std::fclose(file)) mOk = false;
		mTrace = std::fopen(tracePath, "wb");
		mFrame.reserve(2 + mesh.getNbVertices() * 3 + mRigids.size() * 7);
		return mOk && mTrace;
	}
	bool frame(double stepMs, bool activeAtStart)
	{
		using namespace physx;
		if(!mTrace) return mOk;
		mFrame.clear();
		mFrame.push_back(float(stepMs));
		mFrame.push_back(activeAtStart ? 1.0f : 0.0f);
		const PxVec4* positions = mVolume->getSimPositionInvMassBufferD();
		for(PxU32 i = 0; i < mVolume->getSimulationMesh()->getNbVertices(); ++i)
			for(PxU32 j = 0; j < 3; ++j) mFrame.push_back(positions[i][j]);
		for(PxRigidActor* actor : mRigids)
		{
			const PxTransform pose = actor->getGlobalPose();
			mFrame.push_back(pose.p.x); mFrame.push_back(pose.p.y); mFrame.push_back(pose.p.z);
			mFrame.push_back(pose.q.x); mFrame.push_back(pose.q.y); mFrame.push_back(pose.q.z); mFrame.push_back(pose.q.w);
		}
		mOk = std::fwrite(mFrame.data(), sizeof(float), mFrame.size(), mTrace) == mFrame.size();
		return mOk;
	}
	bool finish() { if(mTrace) { if(std::fclose(mTrace)) mOk = false; mTrace = NULL; } return mOk; }
private:
	FILE* mTrace;
	physx::PxDeformableVolume* mVolume;
	std::vector<physx::PxRigidActor*> mRigids;
	std::vector<float> mFrame;
	bool mOk;
};
#endif
