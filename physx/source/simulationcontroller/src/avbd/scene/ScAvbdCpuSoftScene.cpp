// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

AvbdCpuSoftScene::WriteBackTask::WriteBackTask(
	PxU64 contextId, AvbdCpuSoftScene& scene)
	: Cm::Task(contextId), mScene(scene), mEntryBegin(0), mEntryEnd(0)
{
}

void AvbdCpuSoftScene::WriteBackTask::configure(
	PxU32 entryBegin, PxU32 entryEnd)
{
	PX_ASSERT(entryBegin < entryEnd);
	mEntryBegin = entryBegin;
	mEntryEnd = entryEnd;
}

const char* AvbdCpuSoftScene::WriteBackTask::getName() const
{
	return "ScScene.avbdCpuSoftWriteBack";
}

void AvbdCpuSoftScene::WriteBackTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginWriteBackTask();
	mScene.writeBackStandaloneComponentRange(mEntryBegin, mEntryEnd);
	mScene.mStandaloneTaskGraphTelemetry.endWriteBackTask();
}

AvbdCpuSoftScene::PredictionTask::PredictionTask(
	PxU64 contextId, AvbdCpuSoftScene& scene)
	: Cm::Task(contextId), mScene(scene), mEntryBegin(0), mEntryEnd(0),
	  mDt(0.0f), mGravity(0.0f)
{
}

void AvbdCpuSoftScene::PredictionTask::configure(
	PxU32 entryBegin, PxU32 entryEnd, PxReal dt,
	const PxVec3& gravity)
{
	PX_ASSERT(entryBegin < entryEnd);
	mEntryBegin = entryBegin;
	mEntryEnd = entryEnd;
	mDt = dt;
	mGravity = gravity;
}

const char* AvbdCpuSoftScene::PredictionTask::getName() const
{
	return "ScScene.avbdCpuSoftPrediction";
}

void AvbdCpuSoftScene::PredictionTask::runInternal()
{
	mScene.mStandaloneTaskGraphTelemetry.beginPredictionTask();
	mScene.predictStandaloneComponentRange(
		mEntryBegin, mEntryEnd, mDt, mGravity);
	mScene.mStandaloneTaskGraphTelemetry.endPredictionTask();
}

AvbdCpuSoftScene::CausalLayerTask::CausalLayerTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mPoolIndex(poolIndex),
	  mSolveContext(NULL), mBodies(NULL), mBodyCount(0),
	  mParticleBodyIndices(NULL), mNumParticles(0),
	  mPackedParticleIndices(NULL), mPackedBegin(0), mPackedEnd(0),
	  mIndependentBodyRange(false), mBodyBegin(0), mBodyEnd(0),
	  mObservation(NULL)
{
}

void AvbdCpuSoftScene::CausalLayerTask::configure(
	const Dy::AvbdParticlePrimalSolveContext& solveContext,
	const Dy::AvbdSoftBody* bodies, PxU32 bodyCount,
	const PxU32* particleBodyIndices, PxU32 numParticles,
	const PxU32* packedParticleIndices, PxU32 packedBegin, PxU32 packedEnd,
	Dy::AvbdParticlePrimalRangeObservation& observation)
{
	PX_ASSERT(bodies && bodyCount > 0);
	PX_ASSERT(particleBodyIndices && packedParticleIndices);
	PX_ASSERT(packedBegin < packedEnd);
	mSolveContext = &solveContext;
	mBodies = bodies;
	mBodyCount = bodyCount;
	mParticleBodyIndices = particleBodyIndices;
	mNumParticles = numParticles;
	mPackedParticleIndices = packedParticleIndices;
	mPackedBegin = packedBegin;
	mPackedEnd = packedEnd;
	mIndependentBodyRange = false;
	mBodyBegin = 0;
	mBodyEnd = 0;
	mObservation = &observation;
}

void AvbdCpuSoftScene::CausalLayerTask::configureIndependentBodyRange(
	const Dy::AvbdParticlePrimalSolveContext& solveContext,
	const Dy::AvbdSoftBody* bodies, PxU32 bodyCount,
	PxU32 bodyBegin, PxU32 bodyEnd,
	Dy::AvbdParticlePrimalRangeObservation& observation)
{
	PX_ASSERT(bodies && bodyCount > 1 &&
		bodyBegin < bodyEnd && bodyEnd <= bodyCount);
	mSolveContext = &solveContext;
	mBodies = bodies;
	mBodyCount = bodyCount;
	mParticleBodyIndices = NULL;
	mNumParticles = 0;
	mPackedParticleIndices = NULL;
	mPackedBegin = 0;
	mPackedEnd = 0;
	mIndependentBodyRange = true;
	mBodyBegin = bodyBegin;
	mBodyEnd = bodyEnd;
	mObservation = &observation;
}

void AvbdCpuSoftScene::CausalLayerTask::release()
{
	// Keep the continuation local before returning this persistent task to the
	// pool: the final child can immediately wake its parent on another worker.
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleCausalLayerTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::CausalLayerTask::getName() const
{
	return "ScScene.avbdCpuSoftCausalLayer";
}

void AvbdCpuSoftScene::CausalLayerTask::runInternal()
{
	PX_ASSERT(mSolveContext && mBodies);
	PX_ASSERT(mObservation);
	PX_ASSERT(mIndependentBodyRange ?
		(mBodyBegin < mBodyEnd && mBodyEnd <= mBodyCount) :
		(mParticleBodyIndices && mPackedParticleIndices &&
			mPackedBegin < mPackedEnd));
	mScene.mStandaloneTaskGraphTelemetry.beginCausalLayerTask();
	// The fan-in observations are stored contiguously.  Updating those shared
	// slots once per particle makes otherwise independent body tasks contend on
	// the same cache line.  Keep the reduction private for the duration of the
	// task and publish it once after the primal range is complete.
	Dy::AvbdParticlePrimalRangeObservation localObservation;
	if(mIndependentBodyRange)
		Dy::avbdSolveParticlePrimalIndependentBodyRange(
			*mSolveContext, mBodies, mBodyCount,
			mBodyBegin, mBodyEnd, localObservation);
	else
		Dy::avbdSolveParticlePrimalPackedRange(
			*mSolveContext, mBodies, mBodyCount, mParticleBodyIndices,
			mNumParticles, mPackedParticleIndices,
			mPackedBegin, mPackedEnd, localObservation);
	*mObservation = localObservation;
	mScene.mStandaloneTaskGraphTelemetry.endCausalLayerTask();
}

AvbdCpuSoftScene::CausalLayerFinishTask::CausalLayerFinishTask(
	PxU64 contextId, AvbdCpuSoftScene& scene, Scene& owner, PxU32 poolIndex)
	: Cm::Task(contextId), mScene(scene), mOwner(owner), mPoolIndex(poolIndex)
{
}

void AvbdCpuSoftScene::CausalLayerFinishTask::runInternal()
{
	mOwner.avbdCpuSoftComponentCausalLayerFinish(mCont);
}

void AvbdCpuSoftScene::CausalLayerFinishTask::release()
{
	PxBaseTask* const continuation = mCont;
	mCont = NULL;
	mScene.recycleCausalLayerFinishTask(mPoolIndex);
	if(continuation)
		continuation->removeReference();
}

const char* AvbdCpuSoftScene::CausalLayerFinishTask::getName() const
{
	return "ScScene.avbdCpuSoftCausalLayerFinish";
}

} // namespace Sc
} // namespace physx
