// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifdef RENDER_SNIPPET
#include "SnippetJointSurface.h"
#include "../snippetrender/SnippetCamera.h"
#include "../snippetrender/SnippetRender.h"
#include "../snippetcommon/SnippetPrint.h"
#include <cstdlib>
#include <chrono>

namespace JointSurfaceSnippet
{
namespace
{
SheetWorld* world=NULL;
Snippets::Camera* camera=NULL;
Options config;
bool paused=false,oneFrame=false,failed=false;
bool inFlight=false,resetRequested=false;
PxGeometryHolder geometry[3];PxTransform poses[3];PxU32 geometryCount=0;
PxU32 frameCount=0,frameLimit=0;
void snapshot()
{
	world->updateSkin();geometryCount=0;
	PxRigidActor* actors[3]={world->host.ground,world->obstacle,world->host.impactor};
	for(auto* actor:actors)if(actor)
	{
		PxShape* shape=NULL;if(actor->getShapes(&shape,1)!=1)continue;
		geometry[geometryCount]=shape->getGeometry();poses[geometryCount]=actor->getGlobalPose()*shape->getLocalPose();++geometryCount;
	}
}
void draw()
{
	if(!world) { glutLeaveMainLoop();return; }
	if(inFlight && world->host.scene->checkResults(false))
	{
		PxU32 error=0;
		failed=!world->host.scene->fetchResults(false,&error) || error!=0 || world->host.errors.fatal.load()!=0;
		inFlight=false;++frameCount;snapshot();
		if(failed) { paused=true;std::fprintf(stderr,"[JOINT_SURFACE_VISUAL] simulation failed; paused without fallback.\n"); }
	}
	if(resetRequested && !inFlight)
	{
		delete world;world=new SheetWorld();failed=!world->init(config);paused=failed;oneFrame=false;frameCount=0;resetRequested=false;
		if(failed) { glutLeaveMainLoop();return; }snapshot();
	}
	if(frameLimit && frameCount>=frameLimit) { glutLeaveMainLoop();return; }
	// Only the last fetched snapshot is read while the SDK owns the scene.
	Snippets::startRender(camera);
	Snippets::renderGeoms(geometryCount,geometry,poses,true,PxVec3(.65f,.72f,.82f));
	Snippets::renderMesh(world->vertices.size(),world->vertices.begin(),world->triangles.size()/3,world->triangles.begin(),PxVec3(.95f,.34f,.16f));
	Snippets::print("Joint cloth: thin patches with finite joints (full drape qualification OPEN).");
	Snippets::print(failed?"Simulation stopped after an error. R: restart. See console.":
		resetRequested?"Reset queued; waiting for the current physics step.":
		inFlight?(paused?"Finishing current step, then pausing.":"Simulating... showing the last completed frame. P: pause. R: reset."):
		paused?"Paused. P: resume. O: one step. R: reset.":"P / Space: pause. O: one step. R: reset.");
	Snippets::finishRender();
	if(std::getenv("PHYSX_AVBD_VISUAL_PROFILE"))
	{
		static auto last=std::chrono::steady_clock::now();static PxU32 draws=0;++draws;
		const auto now=std::chrono::steady_clock::now();const double seconds=std::chrono::duration<double>(now-last).count();
		if(seconds>=1) { std::fprintf(stderr,"[JOINT_SURFACE_UI] completed=%u pending=%u draws=%u seconds=%.3f\n",frameCount,unsigned(inFlight),draws,seconds);draws=0;last=now; }
	}
	if(!failed && !inFlight && !resetRequested && (!paused || oneFrame))
	{
		world->sendPvdSkin();
		oneFrame=false;inFlight=world->host.scene->simulate(1.0f/60);
		if(!inFlight) { failed=true;paused=true; }
	}
}
void key(unsigned char k,const PxTransform&)
{
	if(k=='r' || k=='R')
	{
		// Rebuild authored state including constraints, so reset cannot retain
		// old contact/material history behind new actor poses.
		resetRequested=true;paused=true;oneFrame=false;return;
	}
	if(failed) return;
	if(k=='p' || k=='P' || k==' ') paused=!paused;
	if(k=='o' || k=='O') { paused=true;oneFrame=true; }
}
void cleanup()
{
	// Releasing an SDK scene with an outstanding simulate is not permitted.
	if(inFlight && world) { glutHideWindow();world->host.scene->fetchResults(true);inFlight=false; }
	delete camera;camera=NULL;delete world;world=NULL;
}
}
int renderLoop(const Options& options)
{
	config=options;world=new SheetWorld();if(!world->init(config)) { cleanup();return 2; }
	paused=false;oneFrame=false;failed=false;inFlight=false;resetRequested=false;frameCount=0;snapshot();
	const char* limit=std::getenv("PHYSX_AVBD_VISUAL_MAX_FRAMES");frameLimit=limit?PxU32(std::strtoul(limit,NULL,10)):0;
	camera=new Snippets::Camera(PxVec3(7.5f,5.5f,8),PxVec3(-.62f,-.28f,-.73f));
	Snippets::setupDefault("PhysX Snippet Deformable Surface AVBD - Joint",camera,key,draw,cleanup);
	Snippets::enableVSync(true);
	glutMainLoop();return failed?1:0;
}
}
#endif
