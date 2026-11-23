#ifdef RENDER_SNIPPET

#include "SnippetDeformableVolumeAVBD.h"
#include "../snippetrender/SnippetCamera.h"
#include "../snippetrender/SnippetRender.h"

#include <cstdlib>

using namespace physx;

namespace
{
Snippets::Camera* gCamera = NULL;
PxU32 gFrameCount = 0;
PxU32 gFrameLimit = 0;
const PxVec3 gColors[] =
{
	PxVec3(1.0f, 0.5f, 0.25f), PxVec3(0.45f, 0.6f, 0.75f),
	PxVec3(0.6f, 0.9f, 0.4f), PxVec3(0.9f, 0.85f, 0.2f),
	PxVec3(0.8f, 0.3f, 0.7f)
};

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
	if(gVolumeAvbdScene)
	{
		const PxActorTypeFlags types = PxActorTypeFlag::eRIGID_DYNAMIC |
			PxActorTypeFlag::eRIGID_STATIC;
		const PxU32 count = gVolumeAvbdScene->getNbActors(types);
		if(count)
		{
			PxArray<PxRigidActor*> actors(count);
			gVolumeAvbdScene->getActors(types,
				reinterpret_cast<PxActor**>(actors.begin()), count);
			Snippets::renderActors(actors.begin(), count, true,
				PxVec3(0.65f, 0.72f, 0.82f));
		}
	}
	for(PxU32 i = 0; i < getVolumeAvbdRenderCount(); ++i)
	{
		PxDeformableVolume* volume = getVolumeAvbdRenderVolume(i);
		if(volume)
			Snippets::renderDeformableVolume(volume,
				volume->getPositionInvMassBufferD(), true,
				gColors[i % PX_ARRAY_SIZE(gColors)]);
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
	gCamera = new Snippets::Camera(PxVec3(20.0f, 26.0f, 24.0f),
		PxVec3(-0.52f, -0.45f, -0.73f));
	gFrameCount = 0;
	const char* frameLimit = std::getenv("PHYSX_AVBD_VISUAL_MAX_FRAMES");
	gFrameLimit = frameLimit ? PxU32(std::strtoul(frameLimit, NULL, 10)) : 0;
	Snippets::setupDefault("PhysX Snippet Deformable Volume AVBD",
		gCamera, keyPress, renderCallback, exitCallback);
	glutMainLoop();
}

#endif
