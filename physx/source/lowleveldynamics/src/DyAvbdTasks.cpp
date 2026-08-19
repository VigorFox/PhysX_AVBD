// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "DyAvbdTasks.h"
#include "DyAvbdBodyConversion.h"
#include "DyAvbdDynamics.h"
#include "DyAvbdOwnerWaveContract.h"
#include "DyFeatherstoneArticulation.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DySleep.h"
#include "DyVArticulation.h"
#include "PxsRigidBody.h"
#include "common/PxProfileZone.h"
namespace physx {
namespace Dy {

struct AvbdRigidExecutionPolicy {
  enum : PxU32 {
    eMIN_PARALLEL_ISLAND_BODIES = 512u,
    eTASK_GRAIN_BODIES = 256u,
    eCOLOR_TASK_GRAIN_BODIES = 64u,
    eDUAL_TASK_GRAIN_CONTACTS = 256u,
    eGPU_MIN_WAVE_BODIES = 32u,
    eGPU_MAX_WAVE_BODIES = PXG_AVBD_OWNER_WAVE_MAX_OWNERS
  };
};

static void syncSingleDofArticulationJointState(
    ArticulationData &artData, PxU32 linkIndex) {
  if (linkIndex == 0 || linkIndex >= artData.getLinkCount())
    return;
  const ArticulationLink &link = artData.getLink(linkIndex);
  if (link.parent == DY_ARTICULATION_LINK_NONE ||
      link.parent >= artData.getLinkCount() || !link.inboundJoint ||
      !link.bodyCore)
    return;
  const ArticulationLink &parent = artData.getLink(link.parent);
  if (!parent.bodyCore)
    return;

  ArticulationJointCore &joint = *link.inboundJoint;
  ArticulationJointCoreData &jointData =
      artData.getJointData(linkIndex);
  if (jointData.nbDof != 1 || jointData.jointOffset == PX_MAX_U32)
    return;
  const PxArticulationAxis::Enum axis =
      static_cast<PxArticulationAxis::Enum>(joint.dofIds[0]);
  PxU32 component = 0;
  switch (axis) {
  case PxArticulationAxis::eX:
  case PxArticulationAxis::eTWIST:
    component = 0;
    break;
  case PxArticulationAxis::eY:
  case PxArticulationAxis::eSWING1:
    component = 1;
    break;
  case PxArticulationAxis::eZ:
  case PxArticulationAxis::eSWING2:
    component = 2;
    break;
  default:
    return;
  }

  const PxTransform parentFrame =
      parent.bodyCore->body2World * joint.parentPose;
  const PxTransform childFrame =
      link.bodyCore->body2World * joint.childPose;
  PxVec3 localAxis(0.0f);
  localAxis[component] = 1.0f;
  const PxVec3 worldAxis = parentFrame.q.rotate(localAxis);
  const bool linearAxis = axis >= PxArticulationAxis::eX;
  PxReal position = 0.0f;
  PxReal velocity = 0.0f;
  if (linearAxis) {
    position = (childFrame.p - parentFrame.p).dot(worldAxis);
    const PxVec3 parentPointVelocity =
        parent.bodyCore->linearVelocity +
        parent.bodyCore->angularVelocity.cross(
            parentFrame.p - parent.bodyCore->body2World.p);
    const PxVec3 childPointVelocity =
        link.bodyCore->linearVelocity +
        link.bodyCore->angularVelocity.cross(
            childFrame.p - link.bodyCore->body2World.p);
    velocity =
        (childPointVelocity - parentPointVelocity).dot(worldAxis);
  } else {
    PxQuat relative = parentFrame.q.getConjugate() * childFrame.q;
    if (relative.w < 0.0f)
      relative = -relative;
    position =
        2.0f * PxAtan2((&relative.x)[component], relative.w);
    velocity =
        (link.bodyCore->angularVelocity -
         parent.bodyCore->angularVelocity).dot(worldAxis);
  }
  if (!PxIsFinite(position) || !PxIsFinite(velocity))
    return;

  artData.getJointPositions()[jointData.jointOffset] = position;
  artData.getJointVelocities()[jointData.jointOffset] = velocity;
  artData.getPosIterJointVelocities()[jointData.jointOffset] = velocity;
  joint.jointPos[axis] = position;
  joint.jointVel[axis] = velocity;
}

void AvbdRigidBodyRangeTask::run() {
  AvbdRigidSolveIterationState &state = mSolveContext.iteration;
  mSolver.solveRigidBodyRange(
      state.bodies, state.numBodies, state.contacts, state.numContacts,
      state.dt, mSolveContext.invDt2, state.contactMap,
      mBodyOrder, mBegin, mEnd);
}

void AvbdRigidDualRangeTask::run() {
  PX_PROFILE_ZONE("AVBD.updateLambdaRange", 0);
  mSolver.solveRigidDualRange(mSolveContext.iteration, mBegin, mEnd);
}

static void releaseAvbdMapStorage(AvbdMapStorage &storage,
                                  PxAllocatorCallback &allocator) {
  if (storage.counts)
    allocator.deallocate(storage.counts);
  if (storage.offsets)
    allocator.deallocate(storage.offsets);
  if (storage.indices)
    allocator.deallocate(storage.indices);
  storage = AvbdMapStorage();
}

void AvbdSolveIslandTask::materializeDeferredContacts() {
  AvbdDeferredContactPrep &prep = mBatch.deferredContactPrep;
  if (!prep.snapshots)
    return;

  const PxU32 emitted = prepareAvbdContactSnapshots(
      mDt, mBatch.bodies, mBatch.numBodies, mBatch.constraints,
      prep.constraintCapacity, prep.startContactIdx, prep.numContactManagers,
      prep.bodyOffset, prep.snapshots, nullptr, 0, prep.lengthScale,
      prep.enableLambdaWarmStart, prep.frameStamp);

  bool cardinalityMatch = emitted == prep.constraintCapacity;
  for (PxU32 contactOrder = 0;
       cardinalityMatch && contactOrder < prep.numContactManagers;
       ++contactOrder) {
    const AvbdContactPrepSnapshot &snapshot =
        prep.snapshots[prep.startContactIdx + contactOrder];
    cardinalityMatch =
        snapshot.expectedResponseRows == snapshot.emittedResponseRows;
  }

  if (!cardinalityMatch) {
    for (PxU32 contactOrder = 0; contactOrder < prep.numContactManagers;
         ++contactOrder)
      prep.snapshots[prep.startContactIdx + contactOrder].managerStateCommit =
          0;
    mBatch.numConstraints = 0;
    prep.snapshots = nullptr;
    PxGetFoundation().error(
        PxErrorCode::eINTERNAL_ERROR, PX_FL,
        "AVBD deferred contact preparation violated its frozen row contract");
    PX_ALWAYS_ASSERT_MESSAGE(
        "AVBD deferred contact preparation violated its frozen row contract");
    return;
  }

  mBatch.numConstraints = emitted;
  mContext.commitContactPrepSnapshots(prep.snapshots, prep.startContactIdx,
                                      prep.numContactManagers,
                                      prep.frameStamp);
  prep.snapshots = nullptr;
}

template <typename ConstraintType>
static void buildDeferredMap(AvbdBodyConstraintMap &map,
                             AvbdMapStorage &storage,
                             const ConstraintType *constraints,
                             PxU32 numBodies, PxU32 numConstraints,
                             PxAllocatorCallback &allocator) {
  if (numBodies == 0 || numConstraints == 0 || !constraints) {
    releaseAvbdMapStorage(storage, allocator);
    return;
  }
  if (storage.counts && storage.offsets && storage.indices &&
      map.buildInPlace(numBodies, constraints, numConstraints, storage.counts,
                       storage.offsets, storage.indices,
                       storage.indexCapacity)) {
    storage = AvbdMapStorage();
    return;
  }
  releaseAvbdMapStorage(storage, allocator);
  map.build(numBodies, constraints, numConstraints, allocator);
}

void AvbdSolveIslandTask::buildDeferredMaps() {
  PxAllocatorCallback &allocator = mContext.getAllocator();
  buildDeferredMap(mBatch.contactMap, mBatch.mapStorage[0],
                   mBatch.constraints, mBatch.numBodies,
                   mBatch.numConstraints, allocator);
  buildDeferredMap(mBatch.d6Map, mBatch.mapStorage[1], mBatch.d6Joints,
                   mBatch.numBodies, mBatch.numD6, allocator);
  buildDeferredMap(mBatch.gearMap, mBatch.mapStorage[2], mBatch.gearJoints,
                   mBatch.numBodies, mBatch.numGear, allocator);
}

void AvbdSolveIslandTask::releaseDeferredMapStorage() {
  PxAllocatorCallback &allocator = mContext.getAllocator();
  for (PxU32 mapIndex = 0; mapIndex < 3; ++mapIndex)
    releaseAvbdMapStorage(mBatch.mapStorage[mapIndex], allocator);
}

bool AvbdSolveIslandTask::canUseRigidWaveTasks() const {
  // Keep wave fan-out for genuinely large islands. The pressure fixture has
  // many independent medium islands, where island-level task parallelism is
  // cheaper and exact than nested wave barriers.
  if (mBatch.numBodies <
          AvbdRigidExecutionPolicy::eMIN_PARALLEL_ISLAND_BODIES ||
      mBatch.numConstraints == 0 ||
      mBatch.hasArticulationBodies ||
      mBatch.numD6 != 0 || mBatch.numGear != 0 ||
      mBatch.numSoftParticles != 0 || mBatch.numSoftBodies != 0 ||
      mBatch.numSoftContacts != 0 ||
      !mBatch.contactMap.constraintOffsets || !getTaskManager() ||
      !getTaskManager()->getCpuDispatcher() ||
      getTaskManager()->getCpuDispatcher()->getWorkerCount() < 2)
    return false;

  // The dependency planner can preserve the configured order, but the
  // deterministic body sort historically used an unspecified tie order.
  // Keep the first dispatcher promotion fail-closed for that mode.
  const AvbdSolverConfig &config = mSolver.getConfig();
  return config.enableParallelization && !config.requiresOrderedBackend();
}

void AvbdSolveIslandTask::submitRigidWave() {
  const AvbdRigidSolveContext &context = mRigidContext;
  const PxU32 begin = context.dependencyWaveOffsets[mCurrentWave];
  const PxU32 end = context.dependencyWaveOffsets[mCurrentWave + 1u];
  const PxU32 count = end - begin;
  if (count == 0) {
    mRigidWaiting = false;
    return;
  }

  // The optional backend receives the already-prepared real island context
  // and the complete disjoint dependency wave. It may batch fixed-width
  // packets internally, but the call remains synchronous: false is
  // transactional and leaves the wave untouched, so the exact scalar owner
  // range can take over without changing wave order.
  AvbdRigidGpuWaveBackend *gpuBackend =
      mContext.getRigidGpuWaveBackend();
  // A device transaction has a fixed upload/launch/readback cost. Keep narrow
  // waves on the scalar authority. The upper bound is also fail-closed: GPU
  // writeback has only been proven exact through the current 64-owner wave
  // boundary; wider waves remain scalar until their ordering contract is
  // independently reproduced.
  if (gpuBackend && gpuBackend->isAvailable() &&
      count >= AvbdRigidExecutionPolicy::eGPU_MIN_WAVE_BODIES &&
      count <= AvbdRigidExecutionPolicy::eGPU_MAX_WAVE_BODIES) {
    PxU32 epoch = ++mGpuWaveEpoch;
    if (epoch == 0)
      epoch = ++mGpuWaveEpoch;
    if (!gpuBackend->solveRigidOwnerWave(
            mSolver, mRigidContext, mCurrentWave, 0u, epoch,
            mSolver.getConfig().avbdAlpha)) {
      mSolver.solveRigidBodyRange(
          mRigidContext.iteration.bodies,
          mRigidContext.iteration.numBodies,
          mRigidContext.iteration.contacts,
          mRigidContext.iteration.numContacts,
          mRigidContext.iteration.dt,
          mRigidContext.invDt2,
          mRigidContext.iteration.contactMap,
          mRigidContext.dependencyWaveBodies.begin(), begin, end);
    }
    mRigidWaiting = false;
    return;
  }

  const PxU32 workers = PxMax(
      1u, getTaskManager()->getCpuDispatcher()->getWorkerCount());
  // Islands are already dispatched independently. Keep the dependency-wave
  // fan-out coarse enough that medium stack islands do not pay a child-task
  // and continuation barrier for every 64 bodies.
  const PxU32 targetBodiesPerTask =
      AvbdRigidExecutionPolicy::eTASK_GRAIN_BODIES;
  // A narrow wave already has no useful fan-out: the existing partitioning
  // would create one child and pay a full parent/continuation round trip for
  // work that still runs on one worker.  Execute that wave in the parent so
  // the dependency barrier remains exact without turning every narrow layer
  // into a task-graph barrier.  Wider waves retain the normal disjoint range
  // fan-in and therefore the multi-worker path is unchanged.
  if (count <= targetBodiesPerTask) {
    mSolver.solveRigidBodyRange(
        mRigidContext.iteration.bodies, mRigidContext.iteration.numBodies,
        mRigidContext.iteration.contacts,
        mRigidContext.iteration.numContacts, mRigidContext.iteration.dt,
        mRigidContext.invDt2, mRigidContext.iteration.contactMap,
        mRigidContext.dependencyWaveBodies.begin(), begin, end);
    mRigidWaiting = false;
    return;
  }
  const PxU32 desiredTasks = (count + targetBodiesPerTask - 1u) /
                             targetBodiesPerTask;
  const PxU32 taskCount = PxMin(workers, PxMax(1u, desiredTasks));
  const PxU32 chunk = (count + taskCount - 1u) / taskCount;
  // Keep one parent reference while the fan-in children are being created.
  // A child may finish inline on the submitting worker; without this hold,
  // the last early child could resubmit the parent before this run returns.
  addReference();
  mRigidSubmitHold = true;
  mRigidWaiting = false;
  for (PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
    const PxU32 taskBegin = begin + PxMin(count, taskIndex * chunk);
    const PxU32 taskEnd = begin + PxMin(count, (taskIndex + 1u) * chunk);
    if (taskBegin >= taskEnd)
      continue;
    AvbdRigidBodyRangeTask *task =
        mContext.getTaskFactory().createRigidBodyRangeTask(
            mContext, mSolver, mRigidContext,
            mRigidContext.dependencyWaveBodies.begin(), taskBegin, taskEnd);
    task->setContinuation(this);
    task->removeReference();
    mRigidWaiting = true;
  }
}

void AvbdSolveIslandTask::submitRigidColor() {
  PX_PROFILE_ZONE("AVBD.submitRigidColor", 0);
  const AvbdRigidSolveContext &context = mRigidContext;
  const PxU32 begin = context.bodyColorOffsets[mCurrentColor];
  const PxU32 end = context.bodyColorOffsets[mCurrentColor + 1u];
  const PxU32 count = end - begin;
  if (count == 0) {
    mRigidWaiting = false;
    return;
  }

  // Unlike the exact dependency waves, a strict body color normally exposes
  // hundreds or thousands of independent owners behind only a few barriers.
  // Use a smaller grain to feed modern CPU dispatchers without recreating the
  // old many-wave tiny-task failure mode.
  const PxU32 targetBodiesPerTask =
      AvbdRigidExecutionPolicy::eCOLOR_TASK_GRAIN_BODIES;
  const PxU32 workers = PxMax(
      1u, getTaskManager()->getCpuDispatcher()->getWorkerCount());
  const PxU32 desiredTasks =
      (count + targetBodiesPerTask - 1u) / targetBodiesPerTask;
  const PxU32 taskCount = count <= targetBodiesPerTask
      ? 1u
      : PxMin(workers, PxMax(1u, desiredTasks));
  const PxU32 chunk = count <= targetBodiesPerTask
      ? count
      : (count + taskCount - 1u) / taskCount;

  // This is a lab-only, opt-in parent-side capture point.  The color plan and
  // prepared CSR are real production data, while no body in this color has
  // begun solving yet.  The context reserved all storage before task
  // submission; this call only validates and copies, and is unreachable from
  // the GPU owner-wave path.
  mContext.captureKernelLabCpuColorPreRange(
      mKernelLabCaptureTicket, mRigidContext, mBatch.islandStart,
      mCurrentColor, mRigidContext.bodyColorBodies.begin(), begin, end,
      workers, targetBodiesPerTask, taskCount, chunk);
  if (count <= targetBodiesPerTask) {
    AvbdRigidSolveIterationState &state = mRigidContext.iteration;
    mSolver.solveRigidBodyRange(
        state.bodies, state.numBodies, state.contacts, state.numContacts,
        state.dt, mRigidContext.invDt2, state.contactMap,
        mRigidContext.bodyColorBodies.begin(), begin, end);
    mContext.captureKernelLabCpuColorPostRange(mKernelLabCaptureTicket,
                                               mRigidContext, mCurrentColor);
    mRigidWaiting = false;
    return;
  }

  addReference();
  mRigidSubmitHold = true;
  mRigidWaiting = false;
  for (PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
    const PxU32 taskBegin = begin + PxMin(count, taskIndex * chunk);
    const PxU32 taskEnd =
        begin + PxMin(count, (taskIndex + 1u) * chunk);
    if (taskBegin >= taskEnd)
      continue;
    AvbdRigidBodyRangeTask *task =
        mContext.getTaskFactory().createRigidBodyRangeTask(
            mContext, mSolver, mRigidContext,
            mRigidContext.bodyColorBodies.begin(), taskBegin, taskEnd);
    task->setContinuation(this);
    task->removeReference();
    mRigidWaiting = true;
  }
}

bool AvbdSolveIslandTask::submitRigidDual() {
#if PX_AVBD_ENABLE_SOLVER_PROFILE
  // The profile payload includes a global RMS constraint error. Keep that
  // uncommon diagnostic build on the scalar reduction until it has a
  // task-slot reduction contract of its own.
  return false;
#else
  const PxU32 count = mRigidContext.iteration.numContacts;
  const PxU32 workers = PxMax(
      1u, getTaskManager()->getCpuDispatcher()->getWorkerCount());
  // Use a floor here so every balanced range retains at least the grain. In
  // particular, every range remains wider than four contacts and therefore
  // preserves the scalar dual pass's numContacts>4 penalty-growth branch.
  const PxU32 taskCount = PxMin(
      workers, count / AvbdRigidExecutionPolicy::eDUAL_TASK_GRAIN_CONTACTS);
  if (taskCount < 2u)
    return false;

  PX_PROFILE_ZONE("AVBD.submitRigidDual", 0);
  mRigidPhase = eRIGID_DUAL;
  addReference();
  mRigidSubmitHold = true;
  mRigidWaiting = false;
  for (PxU32 taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
    const PxU32 begin = static_cast<PxU32>(
        (static_cast<PxU64>(count) * taskIndex) / taskCount);
    const PxU32 end = static_cast<PxU32>(
        (static_cast<PxU64>(count) * (taskIndex + 1u)) / taskCount);
    PX_ASSERT(end > begin);
    PX_ASSERT(end - begin >=
              AvbdRigidExecutionPolicy::eDUAL_TASK_GRAIN_CONTACTS);
    AvbdRigidDualRangeTask *task =
        mContext.getTaskFactory().createRigidDualRangeTask(
            mContext, mSolver, mRigidContext, begin, end);
    task->setContinuation(this);
    task->removeReference();
    mRigidWaiting = true;
  }
  return mRigidWaiting;
#endif
}

void AvbdSolveIslandTask::finishPreparedRigidSolveSynchronously() {
  while (mSolver.advanceRigidSolveIterations(mRigidContext.iteration)) {
  }
  mSolver.finishRigidSolve(mRigidContext);
  mRigidAsync = false;
}

void AvbdSolveIslandTask::run() {
  if (!mRigidStarted) {
    materializeDeferredContacts();
    buildDeferredMaps();
  }
  if (!mRigidStarted) {
    mRigidStarted = true;
    if (!canUseRigidWaveTasks()) {
      // Single island schedule (classification + shared post-AL inside
      // solver).  This is the authoritative fallback for joints, soft data,
      // small islands, deterministic mode and single-worker scenes.
      mSolver.solveIsland(
          mDt, mBatch.bodies, mBatch.numBodies, mBatch.constraints,
          mBatch.numConstraints, mGravity, mBatch.d6Joints, mBatch.numD6,
          mBatch.gearJoints, mBatch.numGear, &mBatch.contactMap,
          &mBatch.d6Map, &mBatch.gearMap, mBatch.colorBatches,
          mBatch.numColors, mBatch.iterationOverride, mBatch.softParticles,
          mBatch.numSoftParticles, mBatch.softBodies, mBatch.numSoftBodies,
          mBatch.softContacts, mBatch.numSoftContacts,
          &mBatch.softExecutionPlan,
          mBatch.articulationForBody, mBatch.linkIndexForBody, mStats);
      return;
    }

    mRigidAsync = true;
    // Run the full objective classification, then stop at the prepared rigid
    // state instead of entering the serial body loop.
    mSolver.solveIsland(
        mDt, mBatch.bodies, mBatch.numBodies, mBatch.constraints,
        mBatch.numConstraints, mGravity, mBatch.d6Joints, mBatch.numD6,
        mBatch.gearJoints, mBatch.numGear, &mBatch.contactMap, &mBatch.d6Map,
        &mBatch.gearMap, mBatch.colorBatches, mBatch.numColors,
        mBatch.iterationOverride, mBatch.softParticles,
        mBatch.numSoftParticles, mBatch.softBodies, mBatch.numSoftBodies,
        mBatch.softContacts, mBatch.numSoftContacts,
        &mBatch.softExecutionPlan,
        mBatch.articulationForBody, mBatch.linkIndexForBody, mStats,
        &mRigidContext);
    if (!mRigidContext.iteration.bodies) {
      mRigidAsync = false;
      return;
    }
    // Keep the explicitly-installed GPU owner-wave backend on its proven
    // exact schedule.  The normal CPU product path uses a strict compact body
    // coloring and never passes through the GPU interception in
    // submitRigidWave().
    AvbdRigidGpuWaveBackend *gpuBackend =
        mContext.getRigidGpuWaveBackend();
    mRigidUsesBodyColors = !(gpuBackend && gpuBackend->isAvailable());
    if (mRigidUsesBodyColors) {
      if (!mSolver.buildRigidBodyColorPlan(mRigidContext) ||
          mRigidContext.maxBodyColorWidth <=
              AvbdRigidExecutionPolicy::eCOLOR_TASK_GRAIN_BODIES) {
        finishPreparedRigidSolveSynchronously();
        return;
      }
    } else {
      mSolver.buildRigidDependencyWaves(mRigidContext);
    }
    if (!mSolver.beginRigidSolveIteration(mRigidContext.iteration)) {
      mSolver.finishRigidSolve(mRigidContext);
      mRigidAsync = false;
      return;
    }
    mCurrentWave = 0;
    mCurrentColor = 0;
    mRigidPhase = eRIGID_PRIMAL;
  } else if (mRigidWaiting) {
    // The last child in the current wave/color resubmitted this parent through
    // its PxLightCpuTask continuation.
    mRigidWaiting = false;
    if (mRigidPhase == eRIGID_DUAL) {
      mRigidContext.iteration.parallelDualComplete = true;
      mRigidPhase = eRIGID_POST_DUAL;
    } else if (mRigidUsesBodyColors) {
      mContext.captureKernelLabCpuColorPostRange(mKernelLabCaptureTicket,
                                                 mRigidContext,
                                                 mCurrentColor);
      ++mCurrentColor;
    } else {
      ++mCurrentWave;
    }
  }

  while (mRigidAsync) {
    if (mRigidPhase == eRIGID_PRIMAL) {
      if (mRigidUsesBodyColors &&
          mCurrentColor < mRigidContext.bodyColorCount) {
        submitRigidColor();
        if (mRigidWaiting)
          return;
        ++mCurrentColor;
        continue;
      }
      if (!mRigidUsesBodyColors &&
          mCurrentWave < mRigidContext.dependencyWaveCount) {
        submitRigidWave();
        if (mRigidWaiting)
          return;
        ++mCurrentWave;
        continue;
      }

      // Dual rows read the final primal poses, so lock projection remains
      // behind the last color/wave fan-in and ahead of contact fan-out.
      for (PxU32 i = 0; i < mBatch.numBodies; ++i) {
        if (mBatch.bodies[i].invMass > 0.0f)
          mBatch.bodies[i].projectLockedPose(mBatch.bodies[i].prevPosition,
                                             mBatch.bodies[i].prevRotation);
      }
      if (mRigidUsesBodyColors && submitRigidDual())
        return;
      mRigidPhase = eRIGID_POST_DUAL;
    }

    PX_ASSERT(mRigidPhase == eRIGID_POST_DUAL);
    const bool moreIterations =
        mSolver.completeRigidSolveIteration(mRigidContext.iteration);
    if (!moreIterations) {
      mSolver.finishRigidSolve(mRigidContext);
      mRigidAsync = false;
      return;
    }
    if (!mSolver.beginRigidSolveIteration(mRigidContext.iteration)) {
      mSolver.finishRigidSolve(mRigidContext);
      mRigidAsync = false;
      return;
    }
    mCurrentWave = 0;
    mCurrentColor = 0;
    mRigidPhase = eRIGID_PRIMAL;
  }
}

void AvbdCoordinatorTask::run() {
  // The continuation is the join between all submitted island solve tasks
  // and AVBD writeback. Profile reduction, when enabled, belongs here and
  // uses task-local records rather than shared atomics.
}

void AvbdTask::release() {
  // CRITICAL: Must call base class release() which calls
  // mCont->removeReference() to notify the continuation that this task is done.
  PxLightCpuTask::release();
  // After base class release, destroy ourselves
  mContext.destroyTask(this);
}

void AvbdSolveIslandTask::release() {
  if (mRigidAsync && mRigidWaiting) {
    if (mRigidSubmitHold) {
      mRigidSubmitHold = false;
      removeReference();
    }
    return;
  }

  // Write back lambda values to the cache for warm-starting next frame
  // This is thread-safe because each island writes to disjoint cache indices
  {
    AvbdContactConstraint *constraints = mBatch.constraints;
    PxU32 numConstraints = mBatch.numConstraints;

    // Function declared as friend in AvbdDynamicsContext class
    writeLambdaToCache(mContext, constraints, numConstraints, mBatch.numBodies);
    if (mBatch.contactOutputTokens && mBatch.contactOutputResults) {
      writeContactImpulseToOutputTokens(
          constraints, mBatch.contactOutputTokens,
          mBatch.contactOutputResults, numConstraints, mBatch.numBodies, mDt);
    } else {
      writeContactImpulseToOutput(constraints, numConstraints, mBatch.numBodies,
                                  mDt);
    }
    writeJointLambdaToCache(mContext, mBatch.d6Joints, mBatch.numD6);
  }

  // Shared by the task and forced-sequential paths so constraint force and
  // breakage semantics do not depend on scheduling mode. Scoped drive
  // families with a validated physical writeback contract compare their
  // force*dt or torque*dt impulse divided by dt; every other family retains
  // the legacy lambda comparison until it has an independent reaction gate.
  writeJointConstraintWriteback(mContext, mBatch.d6Joints, mBatch.numD6,
                                mDt);

  // Release constraint maps to prevent memory leak
  // Each frame builds new maps, so we must free them when task completes
  PxAllocatorCallback &allocator = mContext.getAllocator();
  mBatch.contactMap.release(allocator);
  mBatch.d6Map.release(allocator);
  mBatch.gearMap.release(allocator);
  releaseDeferredMapStorage();

  // Call base class release
  AvbdTask::release();
}

void AvbdWriteBackTask::run() {
  // Contact report targets are committed only after every island solve has
  // released its worker-owned result tokens.
  commitContactOutputTokens(mContactOutputTargets, mContactOutputResults,
                            mContactOutputCount);

  for (PxU32 i = 0; i < mNumBodies; ++i) {
    if (mRigidBodies[i]) {
      // Regular rigid body - writeback to body core
      if (!mAvbdBodies[i].isStatic()) {
        PxsRigidBody &rigidBody = *mRigidBodies[i];
        PxsBodyCore &bodyCore = rigidBody.getCore();
        const PxTransform oldTransform = bodyCore.body2World;

        // Match the PGS/TGS writeback contract: CCD consumes the pre-step COM
        // pose, while sleeping consumes the motion that actually changed the
        // COM pose this step (which is not necessarily AVBD's final API
        // velocity after depenetration, damping, and velocity limiting).
        rigidBody.mLastTransform = oldTransform;
        writeBackAvbdSolverBody(mAvbdBodies[i], bodyCore);

        if (!mSleepingDisabled) {
          PxVec3 linearMotionVelocity(0.0f);
          PxVec3 angularMotionVelocity(0.0f);
          if (mDt > 0.0f) {
            calculateNewVelocity(bodyCore.body2World, oldTransform, mDt,
                                 linearMotionVelocity,
                                 angularMotionVelocity);
          }

          const Cm::SpatialVector motionVelocity(linearMotionVelocity,
                                                  angularMotionVelocity);
          const PxU32 staticTouchCount =
              mStaticTouchCounts ? mStaticTouchCounts[i] : 0;
          sleepCheck(&rigidBody, mDt, mEnableStabilization, motionVelocity,
                     staticTouchCount);
        }
      }
    } else if (mArticulationForBody && mLinkIndexForBody) {
      // Articulation link - writeback to articulation link body core
      FeatherstoneArticulation *articulation = mArticulationForBody[i];
      PxU32 linkIndex = mLinkIndexForBody[i];

      if (articulation && linkIndex != PX_MAX_U32) {
        ArticulationData &artData = articulation->getArticulationData();
        if (linkIndex < artData.getLinkCount()) {
          const ArticulationLink &link = artData.getLink(linkIndex);
          PxsBodyCore *bodyCore = link.bodyCore;

          if (bodyCore) {
            // Write back position and rotation
            bodyCore->body2World.p = mAvbdBodies[i].position;
            bodyCore->body2World.q = mAvbdBodies[i].rotation;

            // Write back velocities
            bodyCore->linearVelocity = mAvbdBodies[i].linearVelocity;
            bodyCore->angularVelocity = mAvbdBodies[i].angularVelocity;

            // PhysX 5.9 public link-velocity queries consume mMotionVelocities,
            // while articulation sleeping consumes mPosIterMotionVelocities.
            // Keep both buffers and PxsBodyCore synchronized with AVBD's final
            // velocity so driven/falling links cannot sleep on stale zeros.
            Cm::SpatialVectorF &motionVelocity =
                artData.getMotionVelocity(linkIndex);
            motionVelocity.top = mAvbdBodies[i].angularVelocity;
            motionVelocity.bottom = mAvbdBodies[i].linearVelocity;
            artData.getPosIterMotionVelocity(linkIndex) = motionVelocity;
          }
        }
      }
    }
  }

  // Joint coordinates are relative parent/child state. Rebuild them only
  // after every AVBD link pose and velocity has been written back; doing this
  // inside the loop can observe a new child against a parent that has not yet
  // reached its final pose when body ordering changes.
  FeatherstoneArticulation *lastArticulation = nullptr;
  for (PxU32 i = 0; i < mNumBodies; ++i) {
    if (!mArticulationForBody || !mLinkIndexForBody)
      break;
    FeatherstoneArticulation *articulation = mArticulationForBody[i];
    if (!articulation || articulation == lastArticulation)
      continue;
    lastArticulation = articulation;
    ArticulationData &artData = articulation->getArticulationData();
    for (PxU32 linkIndex = 1; linkIndex < artData.getLinkCount();
         ++linkIndex)
      syncSingleDofArticulationJointState(artData, linkIndex);
  }

}

} // namespace Dy
} // namespace physx
