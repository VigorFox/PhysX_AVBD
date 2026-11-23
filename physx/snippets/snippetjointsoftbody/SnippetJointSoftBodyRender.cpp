// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifdef RENDER_SNIPPET
#include "PxPhysicsAPI.h"
#include "SnippetJointSoftBodyShowcase.h"
#include "SnippetJointSoftBodyApp.h"
#include "../snippetrender/SnippetRender.h"
#include "../snippetrender/SnippetCamera.h"
#include "../snippetcommon/SnippetPrint.h"
#include <cstdlib>
using namespace physx;
namespace JointSoftBodySnippet
{
namespace
{
Snippets::Camera* camera=NULL;
PxU32 frameCount=0,frameLimit=0;
void renderCallback()
{
	if(frameLimit && frameCount>=frameLimit) { glutLeaveMainLoop();return; }
	stepPhysics();++frameCount;Snippets::startRender(camera);
	PxScene* scene=getJointSoftBodyScene();
	if(scene)
	{
		const PxActorTypeFlags types=PxActorTypeFlag::eRIGID_DYNAMIC|PxActorTypeFlag::eRIGID_STATIC;
		const PxU32 count=scene->getNbActors(types);
		PxArray<PxRigidActor*> actors(count);
		scene->getActors(types,reinterpret_cast<PxActor**>(actors.begin()),count);
		JointSoftBodySnippet::ShowcaseWorld* world=getJointSoftBodyShowcase();
		PxU32 ordinary=0;
		for(PxU32 i=0;i<count;++i) { PxShape* shape=NULL;actors[i]->getShapes(&shape,1);if(shape && (!world || !shape->getSimulationFilterData().word0)) actors[ordinary++]=actors[i]; }
		Snippets::renderActors(actors.begin(),ordinary,true,PxVec3(0.6f,0.6f,0.6f));
		const PxVec3 colors[5]={PxVec3(0.4f,0.7f,0.9f),PxVec3(0.7f,0.45f,0.9f),PxVec3(0.9f,0.6f,0.3f),PxVec3(0.3f,0.8f,0.6f),PxVec3(0.3f,0.6f,0.8f)};
		for(PxU32 i=0;world && i<world->bodies.size();++i)
		{
			JointSoftBodySnippet::JointAsset& asset=*world->bodies[i];asset.updateSkin();
			Snippets::renderMesh(asset.skinVertices.size(),asset.skinVertices.begin(),asset.skinTriangles.size()/3,asset.skinTriangles.begin(),colors[i%5]);
		}
	}
	Snippets::print(getJointSoftBodyShowcase()?"AVBD joint soft body: five-body showcase (long-run qualification OPEN).":"Joint soft body: single lattice. --visual-showcase selects five-body stress scene.");
	if(getJointSoftBodyShowcase()) Snippets::print("Display skin follows cells; collisions use their convex union.");
	Snippets::print(simulationFailed()?"Simulation stopped after an error. See console; restart to retry.":simulationPaused()?"Paused. P: resume. O: one step.":"P: pause. O: one step. Space: impulse the center cell.");
	Snippets::finishRender();
}
void exitCallback() { delete camera; camera=NULL; cleanupPhysics(); }
}
int renderLoop()
{
	if(!initPhysics()) { cleanupPhysics();return 2; }
	camera=getJointSoftBodyShowcase()?new Snippets::Camera(PxVec3(20,26,24),PxVec3(-0.52f,-0.45f,-0.73f)):new Snippets::Camera(PxVec3(6,5,7),PxVec3(-0.6f,-0.35f,-0.7f));
	frameCount=0;
	const char* limit=std::getenv("PHYSX_AVBD_VISUAL_MAX_FRAMES");
	frameLimit=limit?PxU32(std::strtoul(limit,NULL,10)):0;
	Snippets::setupDefault("PhysX Snippet Joint Soft Body",camera,keyPress,renderCallback,exitCallback);
	glutMainLoop();
	return simulationFailed()?1:0;
}
}
#endif
