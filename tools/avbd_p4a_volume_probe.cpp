// P4a gate: CPU Host PxDeformableVolume create/attach/addActor + IG node + host buffers.
#include <cstdio>
#include "PxPhysicsAPI.h"
#include "extensions/PxDeformableVolumeExt.h"
#include "cooking/PxCooking.h"
#include "cooking/PxTetrahedronMeshDesc.h"
#include "avbd_host_access.h"

using namespace physx;

class ProbeErrorCallback : public PxErrorCallback
{
public:
	virtual void reportError(PxErrorCode::Enum code, const char* message,
		const char* file, int line) PX_OVERRIDE
	{
		std::printf("PhysX [%d] %s (%s:%d)\n", int(code), message, file, line);
		std::fflush(stdout);
	}
};

static void hostFree(void* p)
{
	if(p)
		PxGetFoundation().getAllocatorCallback().deallocate(p);
}

// 8-corner cube, 5 tetrahedra (standard corner-cutting decomposition).
static PxDeformableVolumeMesh* cookUnitCubeVolumeMesh(PxPhysics& physics, const PxVec3& origin, PxReal size)
{
	PxArray<PxVec3> verts;
	verts.reserve(8);
	const PxReal h = size * 0.5f;
	for(PxU32 i = 0; i < 8; ++i)
	{
		verts.pushBack(origin + PxVec3(
			(i & 1) ? h : -h,
			(i & 2) ? h : -h,
			(i & 4) ? h : -h));
	}

	// Corners: 0(--+) wait: bit0=x, bit1=y, bit2=z
	// 0:(-h,-h,-h) 1:(+h,-h,-h) 2:(-h,+h,-h) 3:(+h,+h,-h)
	// 4:(-h,-h,+h) 5:(+h,-h,+h) 6:(-h,+h,+h) 7:(+h,+h,+h)
	PxArray<PxU32> tets;
	const PxU32 tet5[5][4] = {
		{0, 1, 2, 4},
		{1, 3, 2, 7},
		{1, 4, 5, 7},
		{2, 4, 6, 7},
		{1, 2, 4, 7}
	};
	for(PxU32 i = 0; i < 5; ++i)
	{
		tets.pushBack(tet5[i][0]);
		tets.pushBack(tet5[i][1]);
		tets.pushBack(tet5[i][2]);
		tets.pushBack(tet5[i][3]);
	}

	PxTetrahedronMeshDesc meshDesc(verts, tets);
	PxDeformableVolumeSimulationDataDesc simDesc;
	PxCookingParams params(physics.getTolerancesScale());
	std::printf("PxCreateDeformableVolumeMesh...\n");
	return PxCreateDeformableVolumeMesh(params, meshDesc, meshDesc, simDesc, physics.getPhysicsInsertionCallback());
}

int main()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	std::printf("P4a CPU DeformableVolume Host probe start\n");

	ProbeErrorCallback err;
	PxDefaultAllocator alloc;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), true, NULL);
	if(!physics || !PxInitExtensions(*physics, NULL))
	{
		std::printf("FAIL physics\n");
		return 1;
	}

	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(2);
	PxSceneDesc desc(physics->getTolerancesScale());
	desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = PxDefaultSimulationFilterShader;
	desc.solverType = PxSolverType::eAVBD;
	PxScene* scene = physics->createScene(desc);
	if(!scene)
	{
		std::printf("FAIL scene\n");
		return 1;
	}
	std::printf("scene ok\n");

	PxDeformableVolume* volume = physics->createDeformableVolume();
	if(!volume || volume->getCudaContextManager())
	{
		std::printf("FAIL createDeformableVolume cpu host\n");
		return 1;
	}
	std::printf("volume ok %p\n", (void*)volume);

	PxDeformableVolumeMesh* mesh = cookUnitCubeVolumeMesh(*physics, PxVec3(0.0f, 1.0f, 0.0f), 1.0f);
	if(!mesh)
	{
		std::printf("FAIL cook mesh\n");
		return 1;
	}
	std::printf("mesh ok collVerts=%u simVerts=%u\n",
		mesh->getCollisionMesh()->getNbVertices(),
		mesh->getSimulationMesh()->getNbVertices());

	PxDeformableVolumeMaterial* mat = physics->createDeformableVolumeMaterial(2.e5f, 0.3f, 0.1f);
	PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
	PxTetrahedronMeshGeometry geom(mesh->getCollisionMesh());
	PxShape* shape = physics->createShape(geom, &mat, 1, true, shapeFlags);
	std::printf("shape %p mat %p\n", (void*)shape, (void*)mat);
	if(!shape || !volume->attachShape(*shape) ||
		!volume->attachSimulationMesh(*mesh->getSimulationMesh(), *mesh->getDeformableVolumeAuxData()))
	{
		std::printf("FAIL attach\n");
		return 1;
	}
	std::printf("attach ok\n");

	PxVec4* simPos = NULL;
	PxVec4* simVel = NULL;
	PxVec4* collPos = NULL;
	PxVec4* restPos = NULL;
	PxDeformableVolumeExt::allocateAndInitializeHostMirror(*volume, NULL, simPos, simVel, collPos, restPos);
	PxDeformableVolumeExt::updateMass(*volume, 100.0f, 50.0f, simPos);
	PxDeformableVolumeExt::copyToDevice(*volume, PxDeformableVolumeDataFlag::eALL, simPos, simVel, collPos, restPos);
	hostFree(simPos);
	hostFree(simVel);
	hostFree(collPos);
	hostFree(restPos);
	std::printf("buffers filled\n");

	if(!scene->addActor(*volume))
	{
		std::printf("FAIL addActor\n");
		return 1;
	}
	std::printf("addActor ok\n");

	const PxU32 nbActors = scene->getNbDeformableVolumes();
	const PxU32 nbNodesPre = avbdHostActiveDeformableVolumeNodeCount(scene);
	PxVec4* hostSim = volume->getSimPositionInvMassBufferD();
	PxVec4* hostVel = volume->getSimVelocityBufferD();
	const PxU32 nbSimVerts = volume->getSimulationMesh()->getNbVertices();
	bool buffersOk = hostSim && hostVel && nbSimVerts > 0;
	if(buffersOk)
	{
		const PxVec4 saved = hostSim[0];
		hostSim[0] = PxVec4(saved.x + 0.001f, saved.y, saved.z, saved.w);
		buffersOk = (hostSim[0].x == saved.x + 0.001f);
		hostSim[0] = saved;
	}
	std::printf("pre-sim actors=%u nodes=%u simVerts=%u buffers=%d\n",
		nbActors, nbNodesPre, nbSimVerts, int(buffersOk));

	std::printf("simulate start\n");
	scene->simulate(1.0f / 60.0f);
	scene->fetchResults(true);
	std::printf("simulate ok\n");

	const PxU32 nbNodes = avbdHostActiveDeformableVolumeNodeCount(scene);
	std::printf("actors=%u nodes=%u simVerts=%u buffers=%d\n",
		nbActors, nbNodes, nbSimVerts, int(buffersOk));
	const bool pass = (nbActors >= 1) && (nbNodesPre >= 1 || nbNodes >= 1) && buffersOk;
	std::printf(pass ? "PASS P4a CPU DeformableVolume Host\n" : "FAIL P4a Host unlock\n");

	scene->removeActor(*volume);
	volume->release();
	mesh->release();
	mat->release();
	scene->release();
	dispatcher->release();
	PxCloseExtensions();
	physics->release();
	foundation->release();
	return pass ? 0 : 1;
}
