// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

AvbdCpuSoftScene::SelfBvhContactTask::SelfBvhContactTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mParams(), mRangeWorkspace(), mRange()
{
}

void AvbdCpuSoftScene::SelfBvhContactTask::reserveQueryScratch(
	const Dy::AvbdSoftBody& body)
{
	mRangeWorkspace.reserveSelfCollisionSweep(
		body.compiled.tetElements.size(),
		body.compiled.surfaceTriangles.size() / 3,
		body.compiled.surfaceVertices.size(),
		body.compiled.surfaceEdges.size());
}

void AvbdCpuSoftScene::SelfBvhContactTask::configure(
	const Dy::AvbdSoftParticle* particles, const Dy::AvbdSoftBody& body,
	PxU32 softBodyIndex, const Dy::AvbdSelfCollisionAdjacency& adjacency,
	const Dy::AvbdSoftContactWorkspace& parentWorkspace,
	PxU32 vertexBegin, PxU32 vertexEnd, PxU32 edgeBegin, PxU32 edgeEnd,
	PxArray<Dy::AvbdSoftContact>& contacts,
	const Dy::AvbdOGCParams& params,
	Dy::AvbdSoftCollisionStats* collisionStats)
{
	PX_ASSERT(particles && (vertexBegin < vertexEnd || edgeBegin < edgeEnd));
	mParams = params;
	mRange.particles = particles;
	mRange.body = &body;
	mRange.softBodyIndex = softBodyIndex;
	mRange.adjacency = &adjacency;
	mRange.parentWorkspace = &parentWorkspace;
	mRange.rangeWorkspace = &mRangeWorkspace;
	mRange.vertexBegin = vertexBegin;
	mRange.vertexEnd = vertexEnd;
	mRange.edgeBegin = edgeBegin;
	mRange.edgeEnd = edgeEnd;
	mRange.contacts = &contacts;
	mRange.params = &mParams;
	mRange.collisionStats = collisionStats;
}

void AvbdCpuSoftScene::SelfBvhContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginSelfBvhContactTask();
	executeAvbdSelfBvhContactRange(mRange);
	mScene.mStandaloneTaskGraphTelemetry.endSelfBvhContactTask();
}

void AvbdCpuSoftScene::SelfBvhContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleSelfBvhContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::SelfBvhContactTask::getName() const
{
	return "ScScene.avbdCpuSelfBvhContact";
}

AvbdCpuSoftScene::SelfBvhContactFinishTask::SelfBvhContactFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::SelfBvhContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordSelfBvhContactFanIn();
	mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::SelfBvhContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleSelfBvhContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::SelfBvhContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSelfBvhContactFinish";
}

AvbdCpuSoftScene::StaticWorldSelfOgcContactTask::
	StaticWorldSelfOgcContactTask(
		PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex), mParams(),
	  mRangeWorkspace(), mWorldRange(), mBoxRange(), mSelfVertexRange(),
	  mSelfEdgeRange()
{
}

void AvbdCpuSoftScene::StaticWorldSelfOgcContactTask::reserveQueryScratch(
	const Dy::AvbdSoftBody& body)
{
	mRangeWorkspace.reserveSelfCollisionSweep(
		body.compiled.tetElements.size(),
		body.compiled.surfaceTriangles.size() / 3,
		body.compiled.surfaceVertices.size(),
		body.compiled.surfaceEdges.size());
}

void AvbdCpuSoftScene::StaticWorldSelfOgcContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdWorldPlane* planes, PxU32 numPlanes,
	const Dy::AvbdRigidBox* boxes, PxU32 numBoxes,
	const Dy::AvbdSoftContact* previousContacts,
	PxU32 numPreviousContacts, const Dy::AvbdSoftBody& body,
	const Dy::AvbdSelfCollisionAdjacency& adjacency,
	const Dy::AvbdSoftContactWorkspace& preparedWorkspace,
	PxU32 vertexBegin, PxU32 vertexEnd, PxU32 edgeBegin, PxU32 edgeEnd,
	PxArray<Dy::AvbdSoftContact>& worldContacts,
	PxArray<Dy::AvbdSoftContact>& boxContacts,
	PxArray<Dy::AvbdSoftContact>& boxSweptContacts,
	PxArray<Dy::AvbdSoftContact>& selfVertexContacts,
	PxArray<Dy::AvbdSoftContact>& selfEdgeContacts,
	const Dy::AvbdOGCParams& params,
	Dy::AvbdSoftCollisionStats* taskStats,
	PxReal margin)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles && planes && numPlanes > 0);
	PX_ASSERT(boxes && numBoxes > 0 && vertexBegin < vertexEnd);
	PX_ASSERT(edgeBegin < edgeEnd);
	mParams = params;
	mWorldRange.particles = particles;
	mWorldRange.numParticles = numParticles;
	mWorldRange.particleBegin = particleBegin;
	mWorldRange.particleEnd = particleEnd;
	mWorldRange.planes = planes;
	mWorldRange.numPlanes = numPlanes;
	mWorldRange.bodies = &body;
	mWorldRange.numBodies = 1;
	mWorldRange.contacts = &worldContacts;
	mWorldRange.margin = margin;
	mBoxRange.particles = particles;
	mBoxRange.numParticles = numParticles;
	mBoxRange.particleBegin = particleBegin;
	mBoxRange.particleEnd = particleEnd;
	mBoxRange.boxes = boxes;
	mBoxRange.numBoxes = numBoxes;
	mBoxRange.previousContacts = previousContacts;
	mBoxRange.numPreviousContacts = numPreviousContacts;
	mBoxRange.bodies = &body;
	mBoxRange.numBodies = 1;
	mBoxRange.contacts = &boxContacts;
	mBoxRange.sweptContacts = &boxSweptContacts;
	mBoxRange.margin = margin;
	mSelfVertexRange.particles = particles;
	mSelfVertexRange.body = &body;
	mSelfVertexRange.softBodyIndex = 0;
	mSelfVertexRange.adjacency = &adjacency;
	mSelfVertexRange.parentWorkspace = &preparedWorkspace;
	mSelfVertexRange.rangeWorkspace = &mRangeWorkspace;
	mSelfVertexRange.vertexBegin = vertexBegin;
	mSelfVertexRange.vertexEnd = vertexEnd;
	mSelfVertexRange.edgeBegin = 0;
	mSelfVertexRange.edgeEnd = 0;
	mSelfVertexRange.contacts = &selfVertexContacts;
	mSelfVertexRange.params = &mParams;
	mSelfVertexRange.collisionStats = taskStats;
	mSelfEdgeRange = mSelfVertexRange;
	mSelfEdgeRange.vertexBegin = 0;
	mSelfEdgeRange.vertexEnd = 0;
	mSelfEdgeRange.edgeBegin = edgeBegin;
	mSelfEdgeRange.edgeEnd = edgeEnd;
	mSelfEdgeRange.contacts = &selfEdgeContacts;
}

void AvbdCpuSoftScene::StaticWorldSelfOgcContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginWorldPlaneContactTask();
	executeAvbdWorldPlaneContactRange(mWorldRange);
	mScene.mStandaloneTaskGraphTelemetry.endWorldPlaneContactTask();
	mScene.mStandaloneTaskGraphTelemetry.beginRigidBoxSdfContactTask();
	executeAvbdRigidBoxSdfContactRange(mBoxRange);
	mScene.mStandaloneTaskGraphTelemetry.endRigidBoxSdfContactTask();
	mScene.mStandaloneTaskGraphTelemetry.beginSelfBvhContactTask();
	executeAvbdSelfBvhContactRange(mSelfVertexRange);
	executeAvbdSelfBvhContactRange(mSelfEdgeRange);
	mScene.mStandaloneTaskGraphTelemetry.endSelfBvhContactTask();
}

void AvbdCpuSoftScene::StaticWorldSelfOgcContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleStaticWorldSelfOgcContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::StaticWorldSelfOgcContactTask::getName() const
{
	return "ScScene.avbdCpuStaticWorldSelfOgcContact";
}

AvbdCpuSoftScene::StaticWorldSelfOgcContactFinishTask::
	StaticWorldSelfOgcContactFinishTask(
		PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner,
		PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::StaticWorldSelfOgcContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordWorldPlaneContactFanIn();
	mScene.mStandaloneTaskGraphTelemetry.recordRigidBoxSdfContactFanIn();
	mScene.mStandaloneTaskGraphTelemetry.recordSelfBvhContactFanIn();
	mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::StaticWorldSelfOgcContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleStaticWorldSelfOgcContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char*
AvbdCpuSoftScene::StaticWorldSelfOgcContactFinishTask::getName() const
{
	return "ScScene.avbdCpuStaticWorldSelfOgcContactFinish";
}

} // namespace Sc
} // namespace physx
