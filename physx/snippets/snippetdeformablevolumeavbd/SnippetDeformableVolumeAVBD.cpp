#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetAllocationStats.h"
#include "../snippetdeformablevolume/MeshGenerator.h"
#include "SnippetDeformableVolumeAVBD.h"
#include "SnippetDeformableVolumeAVBDCommon.h"
#include "SnippetDeformableVolumeAVBDHeadless.h"
#include "SnippetDeformableVolumeAVBDPvd.h"
#include "../snippetjointsoftbody/SnippetJointSoftBodyApp.h"
#include "../snippetcommon/SnippetDeformableAvbdLaunch.h"
#include "../snippetcommon/SnippetHeadless.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace physx;

PxScene* gVolumeAvbdScene = NULL;

void createVolumeAvbdSphereMesh(PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles, PxReal radius, PxReal maxEdgeLength)
{
	meshgenerator::createSphere(vertices, triangles, PxVec3(0.0f), radius,
		maxEdgeLength);
}

namespace
{
Snippets::AllocationStats gAllocator;
Snippets::TrackingErrorCallback gErrorCallback;
PxFoundation* gFoundation = NULL;
PxPhysics* gPhysics = NULL;
PxDefaultCpuDispatcher* gDispatcher = NULL;
PxPvd* gPvd = NULL;
PxMaterial* gRigidMaterial = NULL;
PxDeformableVolumeMaterial* gVolumeMaterial = NULL;
PxDeformableVolumeMesh* gMeshes[4] = {};
PxDeformableVolume* gVolumes[5] = { NULL, NULL, NULL, NULL, NULL };
bool gExtensionsInitialized = false;
bool gPause = false;
bool gOneFrame = false;
VolumeAvbdPvdMeshVisualizer gPvdMeshVisualizer;

void reportCookingInput(const char* name,const PxArray<PxVec3>& vertices,
	const PxArray<PxU32>& triangles)
{
	const char* enabled=std::getenv("PHYSX_AVBD_MESH_DIAGNOSTICS");
	if(!enabled || !enabled[0] || enabled[0]=='0') return;
	PxU64 hash=14695981039346656037ull;
	auto hashU32=[&](PxU32 value)
	{
		for(PxU32 byte=0;byte<4;++byte)
		{
			hash^=(value>>(8*byte))&0xff;
			hash*=1099511628211ull;
		}
	};
	for(PxU32 vertex=0;vertex<vertices.size();++vertex)
		for(PxU32 axis=0;axis<3;++axis)
		{
			PxU32 bits=0;
			std::memcpy(&bits,&vertices[vertex][axis],sizeof(bits));
			hashU32(bits);
		}
	for(PxU32 index=0;index<triangles.size();++index)
		hashU32(triangles[index]);
	std::printf("[AVBD_COOK_INPUT] name=%s vertices=%u triangles=%u hash=%016llx\n",
		name,vertices.size(),triangles.size()/3,
		static_cast<unsigned long long>(hash));
}

bool createShowcase(VolumeAvbdMeshFactory meshFactory)
{
	PxRigidStatic* ground = PxCreatePlane(*gPhysics,
		PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gRigidMaterial);
	PxRigidStatic* pedestal = gPhysics->createRigidStatic(
		PxTransform(PxVec3(7.6f, 0.55f, 0.0f)));
	if(!ground || !pedestal || !PxRigidActorExt::createExclusiveShape(
		*pedestal, PxBoxGeometry(0.7f, 0.55f, 2.2f), *gRigidMaterial) ||
		!gVolumeAvbdScene->addActor(*ground) ||
		!gVolumeAvbdScene->addActor(*pedestal))
		return false;

	// One elastic material for all five volumes. Stiffness carries the impact
	// load; relaxation and surface friction control distinct losses.
	gVolumeMaterial = gPhysics->createDeformableVolumeMaterial(3.0e5f, 0.42f, 0.1f, 0.009f);
	if(!gVolumeMaterial)
		return false;
	gVolumeMaterial->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);

	PxArray<PxVec3> vertices;
	PxArray<PxU32> triangles;
	auto cook = [&](PxU32 voxels, const char* name)
	{
		return meshFactory ? meshFactory(*gPhysics, vertices, triangles, voxels, name) :
			VolumeAvbdSnippet::cookVisualMesh(*gPhysics, vertices, triangles, voxels);
	};
	VolumeAvbdSnippet::createSubdividedCubeSurface(vertices, triangles, 2.0f, 4);
    const PxVec3 primaryScale(1.8f,.65f,.9f),jawScale(1.7f,.7f,.9f);
    {
        // The material reads the cooked inverse rest matrices. Cook the
        // authored aspect ratios before placing the actors in the world.
        const PxArray<PxVec3> cube(vertices);
        for(PxU32 i=0;i<cube.size();++i)vertices[i]=cube[i].multiply(primaryScale);
        gMeshes[0]=cook(5,"primary-cube");
        for(PxU32 i=0;i<cube.size();++i)vertices[i]=cube[i].multiply(jawScale);
        gMeshes[3]=cook(5,"jaw");
        if(!gMeshes[3])return false;
    }
	meshgenerator::createSphere(vertices, triangles, PxVec3(0.0f), 1.8f, 0.8f);
	reportCookingInput("sphere",vertices,triangles);
	// A 3.6 m diameter, 0.8 m surface edge target spans five material cells.
	// The collision surface and the authored sphere remain the same.
	gMeshes[1] = cook(5, "sphere");
	VolumeAvbdSnippet::createLayeredConeSurface(vertices, triangles,
		0.8f, 3.0f, 32, 3);
	reportCookingInput("cone",vertices,triangles);
	gMeshes[2] = cook(8, "cone");
	if(!gMeshes[0] || !gMeshes[1] || !gMeshes[2])
		return false;

	const PxQuat primaryOrientation(-0.62f,
		PxVec3(0.18f, 0.22f, 0.96f).getNormalized());
	const PxQuat coneOrientation(0.48f,
		PxVec3(0.35f, 0.82f, 0.45f).getNormalized());
	const PxQuat jawOrientation(0.36f, PxVec3(0.0f, 1.0f, 0.0f));
	const PxVec3 rigidCenter(6.4f, 25.0f, 0.3f);
	const PxVec3 jawAxis = jawOrientation.rotate(PxVec3(0.0f, 1.0f, 0.0f));
	gVolumes[0] = VolumeAvbdSnippet::createVisualVolume(*gPhysics,
		*gVolumeAvbdScene, *gMeshes[0], *gVolumeMaterial,
		PxTransform(PxVec3(-3.2f, 8.0f, 0.0f), primaryOrientation),
		PxVec3(1), 160.0f);
	gVolumes[1] = VolumeAvbdSnippet::createVisualVolume(*gPhysics,
		*gVolumeAvbdScene, *gMeshes[1], *gVolumeMaterial,
		PxTransform(PxVec3(-0.2f, 2.0f, 0.0f)), PxVec3(1.0f), 130.0f);
	gVolumes[2] = VolumeAvbdSnippet::createVisualVolume(*gPhysics,
		*gVolumeAvbdScene, *gMeshes[2], *gVolumeMaterial,
		PxTransform(PxVec3(3.2f, 11.0f, 1.2f), coneOrientation),
		PxVec3(1.0f), 160.0f);
	gVolumes[3] = VolumeAvbdSnippet::createVisualVolume(*gPhysics,
		*gVolumeAvbdScene, *gMeshes[3], *gVolumeMaterial,
		PxTransform(rigidCenter + jawAxis * 1.10f, jawOrientation),
		PxVec3(1), 160.0f);
	gVolumes[4] = VolumeAvbdSnippet::createVisualVolume(*gPhysics,
		*gVolumeAvbdScene, *gMeshes[3], *gVolumeMaterial,
		PxTransform(rigidCenter - jawAxis * 1.10f, jawOrientation),
		PxVec3(1), 160.0f);
	for(PxU32 i = 0; i < PX_ARRAY_SIZE(gVolumes); ++i)
		if(!gVolumes[i])
			return false;
	if(!VolumeAvbdSnippet::setVisualVelocity(*gVolumes[0], PxVec3(2.5f, 0.0f, 0.0f)) ||
		!VolumeAvbdSnippet::setVisualVelocity(*gVolumes[2], PxVec3(0.045f, 0.0f, 0.02f)))
		return false;

	PxRigidDynamic* box = gPhysics->createRigidDynamic(
		PxTransform(rigidCenter, jawOrientation));
	if(!box || !PxRigidActorExt::createExclusiveShape(*box,
		PxBoxGeometry(0.5f, 0.35f, 0.7f), *gRigidMaterial) ||
		!PxRigidBodyExt::setMassAndUpdateInertia(*box, 10.0f))
		return false;
	box->setLinearDamping(0.015f);
	box->setSolverIterationCounts(16, 1);
	return gVolumeAvbdScene->addActor(*box);
}
}

void reportVolumeAvbdAllocations(const char* phase) { gAllocator.report(phase); }
PxU32 getVolumeAvbdRenderCount()
{
	return PX_ARRAY_SIZE(gVolumes);
}

PxDeformableVolume* getVolumeAvbdRenderVolume(PxU32 index)
{
	return index < PX_ARRAY_SIZE(gVolumes) ? gVolumes[index] : NULL;
}

bool initPhysics(bool interactive, PxU32 dispatcherThreads, VolumeAvbdMeshFactory meshFactory)
{
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	const char* capturePath = std::getenv("PHYSX_AVBD_PVD_CAPTURE_PATH");
	const bool capturePvd = capturePath && capturePath[0];
	if(interactive || capturePvd)
	{
		gPvd = PxCreatePvd(*gFoundation);
		if(gPvd)
		{
			PxPvdTransport* transport = capturePvd ?
				PxDefaultPvdFileTransportCreate(capturePath) :
				PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if(transport)
				gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
		}
	}
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
		PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.solverType = PxSolverType::eAVBD;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	gVolumeAvbdScene = gPhysics->createScene(sceneDesc);
	PxPvdSceneClient* pvdClient = gVolumeAvbdScene ?
		gVolumeAvbdScene->getScenePvdClient() : NULL;
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	gRigidMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	gPause = false;
	gOneFrame = false;
	return gExtensionsInitialized && gDispatcher && gVolumeAvbdScene &&
		gRigidMaterial && createShowcase(meshFactory);
}

void sendVolumeAvbdPvdMesh()
{
	if(gPvd && gPvd->isConnected())
		gPvdMeshVisualizer.send(*gVolumeAvbdScene, gVolumes, PX_ARRAY_SIZE(gVolumes));
}

void stepPhysics(bool)
{
	if(!gVolumeAvbdScene || (gPause && !gOneFrame))
		return;
	gOneFrame = false;
	sendVolumeAvbdPvdMesh();
	if(!gVolumeAvbdScene->simulate(1.0f / 60.0f) || !gVolumeAvbdScene->fetchResults(true) || gErrorCallback.getFatalCount())
	{
		gPause = true;
		std::fprintf(stderr, "[AVBD_VOLUME_VISUAL] simulation failed; paused.\n");
		return;
	}
}

void cleanupPhysics(bool)
{
	gPvdMeshVisualizer.clear();
	for(PxU32 i = 0; i < PX_ARRAY_SIZE(gVolumes); ++i)
	{
		if(gVolumes[i])
		{
			PxScene* scene = gVolumes[i]->getScene();
			if(scene)
				scene->removeActor(*gVolumes[i]);
			gVolumes[i]->release();
			gVolumes[i] = NULL;
		}
	}
	PX_RELEASE(gVolumeAvbdScene);
	for(PxU32 i = 0; i < PX_ARRAY_SIZE(gMeshes); ++i)
		PX_RELEASE(gMeshes[i]);
	PX_RELEASE(gVolumeMaterial);
	PX_RELEASE(gRigidMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
}

void keyPress(unsigned char key, const PxTransform&)
{
	if(key == 'p' || key == 'P' || key == ' ')
		gPause = !gPause;
	else if(key == 'o' || key == 'O')
	{
		gPause = true;
		gOneFrame = true;
	}
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::DeformableAvbdLaunch launch;
	if(!launch.parse(argc, argv)) return 2;
	argc = int(launch.args.size());
	argv = launch.args.data();
	if(launch.model == Snippets::DeformableAvbdLaunch::Rigid6)
		return JointSoftBodySnippet::runApplication(argc, argv, true);
	if(!launch.configureParticle3()) return 2;
	std::printf("[AVBD_VOLUME_ROUTE] model=particle3 stateDof=3 commonStep=budget iterations=4\n");
	if(snippetDeformableVolumeAvbdHeadlessShouldRun(argc, argv))
		return snippetDeformableVolumeAvbdHeadlessMain(argc, argv);
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	if(initPhysics(false))
		for(PxU32 i = 0; i < 300; ++i)
			stepPhysics(false);
	cleanupPhysics(false);
#endif
	return 0;
}
