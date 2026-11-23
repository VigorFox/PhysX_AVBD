// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_JOINT_SOFT_BODY_SHOWCASE_H
#define SNIPPET_JOINT_SOFT_BODY_SHOWCASE_H
#include "SnippetJointSoftBodyCommon.h"

namespace JointSoftBodySnippet
{
// Scene assets and observations; none of these arrays enters the solver.
struct SkinBinding { PxU32 nodes[8]; PxReal weights[8]; };
struct JointAsset
{
	Lattice lattice;
	PxArray<PxVec3> skinRest, skinVertices;
	PxArray<PxU32> skinTriangles;
	PxArray<SkinBinding> skinBindings;
	PxArray<PxTransform> poseView;
	PxReal sourceVolume, proxyVolume, discardedVolume, restSkinError, envelopeError, envelopeExpansion, rigidSkinError;
	JointAsset() : sourceVolume(0),proxyVolume(0),discardedVolume(0),restSkinError(0),envelopeError(0),envelopeExpansion(0),rigidSkinError(0) {}
	bool build(PxPhysics&,PxScene&,PxMaterial&,const Settings&,const PxTransform&,
		const PxArray<PxVec3>&,const PxArray<PxU32>&,PxU32 identity);
	void updateSkin();
	void release() { lattice.release(); }
	PxU64 skinBytes() const;
};

struct ShowcaseWorld
{
	World host;
	PxArray<JointAsset*> bodies;
	PxRigidStatic* pedestal;
	PxMaterial* rigidMaterial;
	ShowcaseWorld() : pedestal(NULL),rigidMaterial(NULL) {}
	~ShowcaseWorld() { release(); }
	bool init(const Settings&,const char* caseName);
	bool step();
	void release();
};
int runJointSoftBodyMultiHeadless(const Settings&,const char*,PxU32 frames);
}
#endif
