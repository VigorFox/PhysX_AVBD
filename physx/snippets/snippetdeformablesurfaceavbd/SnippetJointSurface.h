// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "../snippetjointsoftbody/SnippetJointSoftBodyCommon.h"

namespace JointSurfaceSnippet
{
using namespace physx;
struct Options
{
	JointSoftBodySnippet::Settings solver;
	PxU32 grid,frames;
	const char* caseName;
	bool headless;
	Options() : grid(24),frames(360),caseName("static-box"),headless(false) {}
};
struct SheetWorld
{
	JointSoftBodySnippet::World host;
	PxRigidStatic* obstacle;
	Options options;
	PxArray<PxVec3> vertices,restVertices;
	PxArray<PxU32> triangles;
	SheetWorld() : obstacle(NULL) {}
	~SheetWorld() { PX_RELEASE(obstacle); }
	bool init(const Options& config);
	bool step();
	void updateSkin();
	void sendPvdSkin();
	static PxReal size() { return 4.8f; }
	static PxReal thickness() { return .02f; }
	static PxReal mass() { return 625; }
};
int runApplication(int argc,const char*const* argv);
int runHeadless(const Options& options);
int renderLoop(const Options& options);
}
