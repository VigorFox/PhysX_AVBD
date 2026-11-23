// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "SnippetJointSoftBodyApp.h"
#include "SnippetJointSoftBodyShowcase.h"
#include "SnippetJointSoftBodyHeadless.h"
#include "SnippetJointSoftBodyPreset.h"
#include <cctype>
#include <cstdlib>
#include <vector>

namespace JointSoftBodySnippet
{
namespace
{
ShowcaseWorld* gWorld=NULL;
World* gLattice=NULL;
Options gOptions;
PxU32 gFrame=0;
bool gPause=false,gOneFrame=false,gFailed=false;
}
PxScene* getJointSoftBodyScene() { return gWorld?gWorld->host.scene:gLattice?gLattice->scene:NULL; }
ShowcaseWorld* getJointSoftBodyShowcase() { return gWorld; }
bool simulationFailed() { return gFailed; }
bool simulationPaused() { return gPause; }
bool initPhysics()
{
	gPause=false;gOneFrame=false;gFailed=false;gFrame=0;
	if(std::strcmp(gOptions.caseName,"visual-showcase")==0)
	{
		gWorld=new ShowcaseWorld();
		return gWorld->init(gOptions.settings,gOptions.caseName);
	}
	gLattice=new World();
	return gLattice->init(gOptions.settings,gOptions.caseName);
}
void stepPhysics()
{
	if(gFailed || (gPause && !gOneFrame)) return;
	gOneFrame=false;
	const bool fetched=gWorld?gWorld->step():gLattice->step(gFrame);
	const PxU32 errors=gWorld?gWorld->host.errors.fatal.load():gLattice->errors.fatal.load();
	if(!fetched || errors)
	{
		gFailed=true;gPause=true;
		std::fprintf(stderr,"[JOINT_SOFT_VISUAL] stopped at frame=%u after simulation failure; no fallback.\n",gFrame+1);
		return;
	}
	++gFrame;
}
void cleanupPhysics() { delete gWorld;gWorld=NULL;delete gLattice;gLattice=NULL; }
void keyPress(unsigned char key,const PxTransform&)
{
	if(gFailed) return;
	if(std::toupper(key)=='P') gPause=!gPause;
	if(std::toupper(key)=='O') { gPause=true;gOneFrame=true; }
	if(key==' ')
	{
		Lattice* lattice=gWorld && !gWorld->bodies.empty()?&gWorld->bodies[0]->lattice:gLattice?&gLattice->lattice:NULL;
		if(lattice && !lattice->nodes.empty())
			lattice->nodes[lattice->nodes.size()/2]->addForce(PxVec3(0,60,0),PxForceMode::eIMPULSE);
	}
}
int runApplication(int argc,const char*const* argv,bool nativeVolumeDefault)
{
	// Detect the user's mode before adding the native showcase's default case.
	const bool headless=snippetJointSoftBodyHeadlessShouldRun(argc,argv);
	std::vector<const char*> args;
	args.push_back(argv[0]);
	if(nativeVolumeDefault)
	{
		args.push_back("--case=visual-showcase");
		args.push_back("--volume-material");
	}
	for(int i=1;i<argc;++i) args.push_back(argv[i]);
	gOptions=Options();
	if(!parseOptions(int(args.size()),args.data(),gOptions))
	{
		std::fprintf(stderr,"Invalid joint-soft options. Use --headless for regression cases.\n");
		return 2;
	}
	if(nativeVolumeDefault)
	{
		if(gOptions.settings.solver!=PxSolverType::eAVBD)
		{
			std::fprintf(stderr,"The native volume preset requires --solver=avbd. Use SnippetJointSoftBody for PGS/TGS comparisons.\n");
			return 2;
		}
		// Application preset only; no SDK-wide solver default or persistent environment change.
		if(!configureNativeJointStep()) return 2;
		std::printf("[AVBD_VOLUME_ROUTE] model=joint volumeMaterial=1 commonStep=%s npCertificates=2 case=%s grid=%u workers=%u\n",
			std::getenv("PHYSX_AVBD_COMMON_STEP"),gOptions.caseName,gOptions.settings.grid,gOptions.settings.workers);
		std::fflush(stdout);
	}
	if(headless) return snippetJointSoftBodyHeadlessMain(int(args.size()),args.data());
	gOptions.settings.pvd=true;
	if(gOptions.disableSleep || (std::strcmp(gOptions.caseName,"showcase") && std::strcmp(gOptions.caseName,"visual-showcase"))) return 2;
#ifdef RENDER_SNIPPET
	return renderLoop();
#else
	if(!initPhysics()) { cleanupPhysics();return 2; }
	for(PxU32 i=0;i<gOptions.frames && !gFailed;++i) stepPhysics();
	cleanupPhysics();return gFailed?1:0;
#endif
}
}
