// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_JOINT_SOFT_BODY_APP_H
#define SNIPPET_JOINT_SOFT_BODY_APP_H
#include "SnippetJointSoftBodyCommon.h"

namespace JointSoftBodySnippet
{
struct ShowcaseWorld;
struct Options
{
	Settings settings;
	const char* caseName;
	PxU32 frames;
	bool disableSleep;
	Options() : caseName("showcase"),frames(360),disableSleep(false) {}
};
bool parseOptions(int argc,const char*const* argv,Options& options);
int runApplication(int argc,const char*const* argv,bool nativeVolumeDefault);
PxScene* getJointSoftBodyScene();
ShowcaseWorld* getJointSoftBodyShowcase();
bool initPhysics();
void stepPhysics();
void cleanupPhysics();
void keyPress(unsigned char key,const PxTransform&);
bool simulationFailed();
bool simulationPaused();
int renderLoop();
}
#endif
