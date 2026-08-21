// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

AvbdCpuSoftScene::WorldPlaneContactTask::WorldPlaneContactTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mRange()
{
}

void AvbdCpuSoftScene::WorldPlaneContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdWorldPlane* planes, PxU32 numPlanes,
	const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts, PxReal margin)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_ASSERT(planes && numPlanes > 0 && bodies && numBodies > 0);
	mRange.particles = particles;
	mRange.numParticles = numParticles;
	mRange.particleBegin = particleBegin;
	mRange.particleEnd = particleEnd;
	mRange.planes = planes;
	mRange.numPlanes = numPlanes;
	mRange.bodies = bodies;
	mRange.numBodies = numBodies;
	mRange.contacts = &contacts;
	mRange.margin = margin;
}

void AvbdCpuSoftScene::WorldPlaneContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginWorldPlaneContactTask();
	executeAvbdWorldPlaneContactRange(mRange);
	mScene.mStandaloneTaskGraphTelemetry.endWorldPlaneContactTask();
}

void AvbdCpuSoftScene::WorldPlaneContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleWorldPlaneContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::WorldPlaneContactTask::getName() const
{
	return "ScScene.avbdCpuSoftWorldPlaneContact";
}

AvbdCpuSoftScene::WorldPlaneContactFinishTask::WorldPlaneContactFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::WorldPlaneContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordWorldPlaneContactFanIn();
	mOwner.avbdCpuSoftComponentWorldPlaneContactFinish(mCont);
}

void AvbdCpuSoftScene::WorldPlaneContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleWorldPlaneContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::WorldPlaneContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftWorldPlaneContactFinish";
}

AvbdCpuSoftScene::RigidBoxSdfContactTask::RigidBoxSdfContactTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mRange()
{
}

void AvbdCpuSoftScene::RigidBoxSdfContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdRigidBox* boxes, PxU32 numBoxes,
	const Dy::AvbdSoftContact* previousContacts, PxU32 numPreviousContacts,
	const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts,
	PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_ASSERT(boxes && numBoxes > 0 && bodies && numBodies > 0);
	mRange.particles = particles;
	mRange.numParticles = numParticles;
	mRange.particleBegin = particleBegin;
	mRange.particleEnd = particleEnd;
	mRange.boxes = boxes;
	mRange.numBoxes = numBoxes;
	mRange.previousContacts = previousContacts;
	mRange.numPreviousContacts = numPreviousContacts;
	mRange.bodies = bodies;
	mRange.numBodies = numBodies;
	mRange.contacts = &contacts;
	mRange.sweptContacts = &sweptContacts;
	mRange.margin = margin;
}

void AvbdCpuSoftScene::RigidBoxSdfContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginRigidBoxSdfContactTask();
	executeAvbdRigidBoxSdfContactRange(mRange);
	mScene.mStandaloneTaskGraphTelemetry.endRigidBoxSdfContactTask();
}

void AvbdCpuSoftScene::RigidBoxSdfContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidBoxSdfContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidBoxSdfContactTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidBoxSdfContact";
}

AvbdCpuSoftScene::RigidBoxSdfContactFinishTask::RigidBoxSdfContactFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::RigidBoxSdfContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordRigidBoxSdfContactFanIn();
	mOwner.avbdCpuSoftComponentRigidBoxSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::RigidBoxSdfContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidBoxSdfContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidBoxSdfContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidBoxSdfContactFinish";
}

AvbdCpuSoftScene::RigidSphereSdfContactTask::RigidSphereSdfContactTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mRange()
{
}

void AvbdCpuSoftScene::RigidSphereSdfContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdRigidSphere* spheres, PxU32 numSpheres,
	const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts,
	PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_ASSERT(spheres && numSpheres > 0 && bodies && numBodies > 0);
	mRange.particles = particles;
	mRange.numParticles = numParticles;
	mRange.particleBegin = particleBegin;
	mRange.particleEnd = particleEnd;
	mRange.spheres = spheres;
	mRange.numSpheres = numSpheres;
	mRange.bodies = bodies;
	mRange.numBodies = numBodies;
	mRange.contacts = &contacts;
	mRange.sweptContacts = &sweptContacts;
	mRange.margin = margin;
}

void AvbdCpuSoftScene::RigidSphereSdfContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginRigidSphereSdfContactTask();
	executeAvbdRigidSphereSdfContactRange(mRange);
	mScene.mStandaloneTaskGraphTelemetry.endRigidSphereSdfContactTask();
}

void AvbdCpuSoftScene::RigidSphereSdfContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidSphereSdfContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidSphereSdfContactTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidSphereSdfContact";
}

AvbdCpuSoftScene::RigidSphereSdfContactFinishTask::RigidSphereSdfContactFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::RigidSphereSdfContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordRigidSphereSdfContactFanIn();
	mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::RigidSphereSdfContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidSphereSdfContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidSphereSdfContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidSphereSdfContactFinish";
}

AvbdCpuSoftScene::RigidCapsuleSdfContactTask::RigidCapsuleSdfContactTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mRange()
{
}

void AvbdCpuSoftScene::RigidCapsuleSdfContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts,
	PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_ASSERT(capsules && numCapsules > 0 && bodies && numBodies > 0);
	mRange.particles = particles;
	mRange.numParticles = numParticles;
	mRange.particleBegin = particleBegin;
	mRange.particleEnd = particleEnd;
	mRange.capsules = capsules;
	mRange.numCapsules = numCapsules;
	mRange.bodies = bodies;
	mRange.numBodies = numBodies;
	mRange.contacts = &contacts;
	mRange.sweptContacts = &sweptContacts;
	mRange.margin = margin;
}

void AvbdCpuSoftScene::RigidCapsuleSdfContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginRigidCapsuleSdfContactTask();
	executeAvbdRigidCapsuleSdfContactRange(mRange);
	mScene.mStandaloneTaskGraphTelemetry.endRigidCapsuleSdfContactTask();
}

void AvbdCpuSoftScene::RigidCapsuleSdfContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidCapsuleSdfContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidCapsuleSdfContactTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidCapsuleSdfContact";
}

AvbdCpuSoftScene::RigidCapsuleSdfContactFinishTask::RigidCapsuleSdfContactFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::RigidCapsuleSdfContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordRigidCapsuleSdfContactFanIn();
	// Sphere, capsule and convex transactions share a mutually-exclusive
	// continuation slot whose owner dispatches by the pending transaction.
	mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::RigidCapsuleSdfContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidCapsuleSdfContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidCapsuleSdfContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidCapsuleSdfContactFinish";
}

AvbdCpuSoftScene::RigidConvexSdfContactTask::RigidConvexSdfContactTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mRange()
{
}

void AvbdCpuSoftScene::RigidConvexSdfContactTask::configure(
	const Dy::AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const Dy::AvbdRigidConvex* convexes, PxU32 numConvexes,
	const Dy::AvbdSoftBody* bodies, PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts,
	PxArray<Dy::AvbdSoftContact>& sweptContacts, PxReal margin)
{
	PX_ASSERT(particles && particleBegin < particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_ASSERT(convexes && numConvexes > 0 && bodies && numBodies > 0);
	mRange.particles = particles;
	mRange.numParticles = numParticles;
	mRange.particleBegin = particleBegin;
	mRange.particleEnd = particleEnd;
	mRange.convexes = convexes;
	mRange.numConvexes = numConvexes;
	mRange.bodies = bodies;
	mRange.numBodies = numBodies;
	mRange.contacts = &contacts;
	mRange.sweptContacts = &sweptContacts;
	mRange.margin = margin;
}

void AvbdCpuSoftScene::RigidConvexSdfContactTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginRigidConvexSdfContactTask();
	executeAvbdRigidConvexSdfContactRange(mRange);
	mScene.mStandaloneTaskGraphTelemetry.endRigidConvexSdfContactTask();
}

void AvbdCpuSoftScene::RigidConvexSdfContactTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidConvexSdfContactTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidConvexSdfContactTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidConvexSdfContact";
}

AvbdCpuSoftScene::RigidConvexSdfContactFinishTask::RigidConvexSdfContactFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::RigidConvexSdfContactFinishTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.recordRigidConvexSdfContactFanIn();
	mOwner.avbdCpuSoftComponentRigidSphereSdfContactFinish(mCont);
}

void AvbdCpuSoftScene::RigidConvexSdfContactFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleRigidConvexSdfContactFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::RigidConvexSdfContactFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftRigidConvexSdfContactFinish";
}

} // namespace Sc
} // namespace physx
