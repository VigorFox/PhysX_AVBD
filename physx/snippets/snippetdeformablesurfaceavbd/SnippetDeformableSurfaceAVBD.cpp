#include "PxPhysicsAPI.h"
#include "../snippetjointsoftbody/SnippetJointSoftBodyPvd.h"
#include "SnippetDeformableSurfaceAVBD.h"
#include "SnippetDeformableSurfaceAVBDCommon.h"
#include "SnippetDeformableSurfaceAVBDHeadless.h"
#include "SnippetJointSurface.h"
#include "../snippetcommon/SnippetDeformableAvbdLaunch.h"
#include "../snippetcommon/SnippetHeadless.h"

using namespace physx;

PxScene* gSurfaceAvbdScene = NULL;
PxDeformableSurface* gSurfaceAvbdRenderSurface = NULL;
PxTriangleMesh* gSurfaceAvbdRenderMesh = NULL;

namespace
{
PxDefaultAllocator gAllocator;
Snippets::TrackingErrorCallback gErrorCallback;
PxFoundation* gFoundation = NULL;
PxPhysics* gPhysics = NULL;
PxDefaultCpuDispatcher* gDispatcher = NULL;
JointSoftBodySnippet::PvdVisualization gPvd;
PxMaterial* gRigidMaterial = NULL;
PxDeformableSurfaceMaterial* gSurfaceMaterial = NULL;
bool gExtensionsInitialized = false;
bool gPause = false;
bool gOneFrame = false;
}

bool initPhysics(bool interactive)
{
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	if(!gPvd.init(*gFoundation, interactive))
		return false;
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
		PxTolerancesScale(), true, gPvd.get());
	if(!gPhysics)
		return false;
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd.get());
	gDispatcher = PxDefaultCpuDispatcherCreate(4);
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.solverType = PxSolverType::eAVBD;
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	gSurfaceAvbdScene = gPhysics->createScene(sceneDesc);
	gRigidMaterial = gPhysics->createMaterial(0.7f, 0.6f, 0.0f);
	if(!gExtensionsInitialized || !gDispatcher || !gSurfaceAvbdScene || !gRigidMaterial)
		return false;
	gPvd.configureScene(*gSurfaceAvbdScene);

	PxRigidStatic* ground = PxCreatePlane(*gPhysics,
		PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *gRigidMaterial);
	PxRigidStatic* obstacle = PxCreateStatic(*gPhysics,
		PxTransform(PxVec3(0.0f, 1.0f, 0.0f)),
		PxBoxGeometry(1.15f, 1.0f, 1.15f), *gRigidMaterial);
	if(!ground || !obstacle || !gSurfaceAvbdScene->addActor(*ground) ||
		!gSurfaceAvbdScene->addActor(*obstacle))
		return false;

	gSurfaceAvbdRenderSurface = SurfaceAvbdSnippet::createDrapedSurface(
		*gPhysics, *gSurfaceAvbdScene, gSurfaceAvbdRenderMesh, gSurfaceMaterial);
	gPause = false;
	gOneFrame = false;
	return gSurfaceAvbdRenderSurface != NULL;
}

void stepPhysics(bool)
{
	if(!gSurfaceAvbdScene || (gPause && !gOneFrame))
		return;
	gOneFrame = false;
	if(gPvd.connected() && gSurfaceAvbdRenderMesh && gSurfaceAvbdRenderSurface)
		gPvd.wireMesh(*gSurfaceAvbdScene, *gSurfaceAvbdRenderMesh,
			gSurfaceAvbdRenderSurface->getPositionInvMassBufferH(), 0xfff25729);
	if(!gSurfaceAvbdScene->simulate(1.0f / 60.0f) || !gSurfaceAvbdScene->fetchResults(true) || gErrorCallback.getFatalCount())
	{
		gPause = true;
		std::fprintf(stderr, "[AVBD_SURFACE_VISUAL] simulation failed; paused.\n");
		return;
	}
}

void cleanupPhysics(bool)
{
	PX_RELEASE(gSurfaceAvbdRenderSurface);
	PX_RELEASE(gSurfaceAvbdScene);
	PX_RELEASE(gSurfaceAvbdRenderMesh);
	PX_RELEASE(gSurfaceMaterial);
	PX_RELEASE(gRigidMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	gPvd.release();
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
	else if((key == 'r' || key == 'R') && gSurfaceAvbdRenderSurface &&
		gSurfaceAvbdRenderMesh)
		SurfaceAvbdSnippet::resetDrapedSurface(
			*gSurfaceAvbdRenderSurface, *gSurfaceAvbdRenderMesh);
}

int snippetMain(int argc, const char*const* argv)
{
	Snippets::DeformableAvbdLaunch launch;
	if(!launch.parse(argc, argv)) return 2;
	argc = int(launch.args.size());
	argv = launch.args.data();
	if(launch.model == Snippets::DeformableAvbdLaunch::Rigid6)
		return JointSurfaceSnippet::runApplication(argc, argv);
	if(!launch.configureParticle3()) return 2;
	std::printf("[AVBD_SURFACE_ROUTE] model=particle3 stateDof=3 commonStep=budget iterations=4\n");
	if(snippetDeformableSurfaceAvbdHeadlessShouldRun(argc, argv))
		return snippetDeformableSurfaceAvbdHeadlessMain(argc, argv);
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
