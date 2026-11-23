#ifdef RENDER_SNIPPET

#include "SnippetDeformableSurfaceAVBD.h"
#include "../snippetrender/SnippetCamera.h"
#include "../snippetrender/SnippetRender.h"

#include <cstdlib>

using namespace physx;

namespace
{
Snippets::Camera* gCamera = NULL;
PxU32 gFrameCount = 0;
PxU32 gFrameLimit = 0;

void renderCallback()
{
	if(gFrameLimit && gFrameCount >= gFrameLimit)
	{
		glutLeaveMainLoop();
		return;
	}
	stepPhysics(true);
	++gFrameCount;
	Snippets::startRender(gCamera);
	if(gSurfaceAvbdScene)
	{
		const PxActorTypeFlags types = PxActorTypeFlag::eRIGID_DYNAMIC |
			PxActorTypeFlag::eRIGID_STATIC;
		const PxU32 count = gSurfaceAvbdScene->getNbActors(types);
		if(count)
		{
			PxArray<PxRigidActor*> actors(count);
			gSurfaceAvbdScene->getActors(types,
				reinterpret_cast<PxActor**>(actors.begin()), count);
			Snippets::renderActors(actors.begin(), count, true,
				PxVec3(0.65f, 0.72f, 0.82f));
		}
	}
	if(gSurfaceAvbdRenderSurface && gSurfaceAvbdRenderMesh)
	{
		Snippets::renderMesh(gSurfaceAvbdRenderMesh->getNbVertices(),
			gSurfaceAvbdRenderSurface->getPositionInvMassBufferH(),
			gSurfaceAvbdRenderMesh->getNbTriangles(),
			gSurfaceAvbdRenderMesh->getTriangles(),
			gSurfaceAvbdRenderMesh->getTriangleMeshFlags().isSet(
				PxTriangleMeshFlag::e16_BIT_INDICES),
			PxVec3(0.95f, 0.34f, 0.16f));
	}
	Snippets::finishRender();
}

void exitCallback()
{
	delete gCamera;
	gCamera = NULL;
	cleanupPhysics(true);
}
}

void renderLoop()
{
	if(!initPhysics(true))
	{
		cleanupPhysics(true);
		return;
	}
	gCamera = new Snippets::Camera(PxVec3(7.5f, 5.5f, 8.0f),
		PxVec3(-0.62f, -0.28f, -0.73f));
	gFrameCount = 0;
	const char* frameLimit = std::getenv("PHYSX_AVBD_VISUAL_MAX_FRAMES");
	gFrameLimit = frameLimit ? PxU32(std::strtoul(frameLimit, NULL, 10)) : 0;
	Snippets::setupDefault("PhysX Snippet Deformable Surface AVBD",
		gCamera, keyPress, renderCallback, exitCallback);
	glutMainLoop();
}

#endif
