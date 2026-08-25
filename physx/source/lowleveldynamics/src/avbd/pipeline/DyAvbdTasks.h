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
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_TASKS_H
#define DY_AVBD_TASKS_H

#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/backend/gpu/DyAvbdGpuWaveBackend.h"
#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/solver/rigid/DyAvbdSolverBody.h"
#include "avbd/core/DyAvbdTypes.h"
#include "DyFeatherstoneArticulation.h"
#include "foundation/PxSimpleTypes.h"
#include "task/PxTask.h"

namespace physx {

class PxTaskManager;
class PxsRigidBody;
struct AvbdContactPrepSnapshot;

namespace IG {
class IslandSim;
}

namespace Dy {

class AvbdDynamicsContext;
class FeatherstoneArticulation;

//=============================================================================
// Task Data Structures
//=============================================================================

// Per-island caller-owned map buffers.  The update thread reserves these
// before dispatch; the solve task consumes exactly one disjoint slot and then
// transfers ownership to AvbdBodyConstraintMap (or falls back to build()).
struct AvbdMapStorage {
  PxU32 *counts;
  PxU32 *offsets;
  PxU32 *indices;
  PxU32 indexCapacity;

  AvbdMapStorage()
      : counts(nullptr), offsets(nullptr), indices(nullptr), indexCapacity(0) {}
};

// Immutable contact-preparation input owned by the parent update and consumed
// once by the island task before it builds its contact map. A null snapshot
// pointer keeps the established parent-preparation path unchanged.
struct AvbdDeferredContactPrep {
  AvbdContactPrepSnapshot *snapshots;
  PxReal lengthScale;
  PxU32 startContactIdx;
  PxU32 numContactManagers;
  PxU32 bodyOffset;
  PxU32 constraintCapacity;
  PxU16 frameStamp;
  bool enableLambdaWarmStart;

  AvbdDeferredContactPrep()
      : snapshots(nullptr), lengthScale(1.0f), startContactIdx(0),
        numContactManagers(0), bodyOffset(0), constraintCapacity(0),
        frameStamp(0), enableLambdaWarmStart(false) {}
};

// Contact report writeback is split into a worker result and a parent-owned
// target. Workers receive only target indices; the parent/writeback owner is
// the sole code that dereferences PhysX output addresses.
struct AvbdContactOutputToken {
  enum : PxU8 {
    eNORMAL_IMPULSE = 1u << 0,
    eFRICTION_IMPULSE = 1u << 1
  };

  PxU32 targetIndex;
  PxU8 flags;
  PxU8 padding[3];

  AvbdContactOutputToken()
      : targetIndex(PX_MAX_U32), flags(0), padding{0, 0, 0} {}
};

struct AvbdContactOutputTarget {
  PxReal *normalImpulse;
  PxVec3 *frictionImpulse;

  AvbdContactOutputTarget()
      : normalImpulse(nullptr), frictionImpulse(nullptr) {}
};

struct AvbdContactOutputResult {
  PxReal normalImpulse;
  PxVec3 frictionImpulse;
  PxU8 flags;
  PxU8 padding[3];

  AvbdContactOutputResult()
      : normalImpulse(0.0f), frictionImpulse(PxVec3(0.0f)), flags(0),
        padding{0, 0, 0} {}
};

// Parent-owned description of one island's final D6 row range. Island tasks
// only mutate the rows themselves; the post-join writeback task walks these
// ranges linearly and is the sole owner of persistent joint-cache mutation.
struct AvbdJointCacheCommitRange {
  const AvbdD6JointConstraint *constraints;
  PxU32 numConstraints;

  AvbdJointCacheCommitRange()
      : constraints(nullptr), numConstraints(0) {}
};

struct AvbdIslandBatch {
  AvbdSolverBody *bodies;
  PxU32 numBodies;
  bool hasArticulationBodies;
  FeatherstoneArticulation **articulationForBody;
  PxU32 *linkIndexForBody;

  AvbdContactConstraint *constraints;
  PxU32 numConstraints;
  AvbdContactOutputToken *contactOutputTokens;
  AvbdContactOutputResult *contactOutputResults;
  AvbdDeferredContactPrep deferredContactPrep;

  // Joint Constraints
  AvbdD6JointConstraint *d6Joints;
  PxU32 numD6;

  AvbdGearJointConstraint *gearJoints;
  PxU32 numGear;

  // Complete soft/VBD tuple. Rigid NP contact prep leaves these null; no
  // synthetic-shell discriminator or partial soft representation is allowed.
  AvbdSoftParticle *softParticles;
  PxU32 numSoftParticles;

  AvbdSoftBody *softBodies;
  PxU32 numSoftBodies;

  AvbdSoftContact *softContacts;
  PxU32 numSoftContacts;
  AvbdSoftIslandExecutionPlan softExecutionPlan;

  PxU32 islandStart;
  PxU32 islandEnd;

  // Per-island requested iteration budget (0 = use the Scene-wide default).
  // Articulations and soft bodies may raise, but never lower, the Scene-wide
  // budget. Convergence may still end the solve before exhausting it.
  PxU32 iterationOverride;

  // Pre-computed constraint coloring (for large islands)
  // These are computed in the single-threaded update() phase to avoid
  // race conditions when multiple island tasks run concurrently.
  AvbdColorBatch
      *colorBatches; //!< Array of color batches (nullptr if not colored)
  PxU32 numColors;   //!< Number of colors used (0 if not colored)

  // Pre-computed constraint-to-body mappings for O(1) lookup
  // These are built once per island and reused across solver iterations
  AvbdBodyConstraintMap contactMap;
  AvbdBodyConstraintMap d6Map;
  AvbdBodyConstraintMap gearMap;
  AvbdMapStorage mapStorage[3]; // contact, D6, gear
};

//=============================================================================
// AVBD Task Base Class
//=============================================================================

class AvbdTask : public PxLightCpuTask {
public:
  AvbdTask(AvbdDynamicsContext &context) : mContext(context) {}
  virtual void
  release() override; // Implemented in cpp to call context.destroyTask

  virtual const char *getName() const override { return "AvbdTask"; }

protected:
  AvbdDynamicsContext &mContext;
};


// One disjoint body slice of the contact-only rigid primal sweep. The parent
// island task owns the context and remains alive through the child
// continuation fan-in; the order may name either an exact wave or a strict
// independent-set color.
class AvbdRigidBodyRangeTask : public AvbdTask {
public:
  AvbdRigidBodyRangeTask(AvbdDynamicsContext &context, AvbdSolver &solver,
                         AvbdRigidSolveContext &solveContext,
                         const PxU32 *bodyOrder, PxU32 begin, PxU32 end)
      : AvbdTask(context), mSolver(solver), mSolveContext(solveContext),
        mBodyOrder(bodyOrder), mBegin(begin), mEnd(end) {}

  virtual void run() override;

  virtual const char *getName() const override {
    return "AvbdRigidBodyRangeTask";
  }

private:
  AvbdSolver &mSolver;
  AvbdRigidSolveContext &mSolveContext;
  const PxU32 *mBodyOrder;
  PxU32 mBegin;
  PxU32 mEnd;
};

// One disjoint contact slice of the fast CPU dual/penalty pass. Body poses are
// read-only after the final primal color barrier and each task writes only its
// own contact rows.
class AvbdRigidDualRangeTask : public AvbdTask {
public:
  AvbdRigidDualRangeTask(AvbdDynamicsContext &context, AvbdSolver &solver,
                         AvbdRigidSolveContext &solveContext, PxU32 begin,
                         PxU32 end)
      : AvbdTask(context), mSolver(solver), mSolveContext(solveContext),
        mBegin(begin), mEnd(end) {}

  virtual void run() override;

  virtual const char *getName() const override {
    return "AvbdRigidDualRangeTask";
  }

private:
  AvbdSolver &mSolver;
  AvbdRigidSolveContext &mSolveContext;
  PxU32 mBegin;
  PxU32 mEnd;
};

//=============================================================================
// Island Solve Task
//=============================================================================

class AvbdSolveIslandTask : public AvbdTask {
public:
  enum RigidAsyncPhase {
    eRIGID_PRIMAL,
    eRIGID_DUAL,
    eRIGID_POST_DUAL
  };

  AvbdSolveIslandTask(AvbdDynamicsContext &context, AvbdSolver &solver,
                      const AvbdIslandBatch &batch, PxReal dt,
                      const PxVec3 &gravity)
      : AvbdTask(context), mSolver(solver), mBatch(batch), mDt(dt),
        mGravity(gravity), mCurrentWave(0), mCurrentColor(0),
        mGpuWaveEpoch(0), mRigidPhase(eRIGID_PRIMAL),
        mRigidStarted(false),
        mRigidAsync(false), mRigidWaiting(false), mRigidSubmitHold(false),
        mRigidUsesBodyColors(false) {}

  virtual void run() override;

  virtual void release() override;

  virtual const char *getName() const override { return "AvbdSolveIslandTask"; }

private:
  void materializeDeferredContacts();
  void buildDeferredMaps();
  void releaseDeferredMapStorage();
  bool canUseRigidWaveTasks() const;
  void submitRigidWave();
  void submitRigidColor();
  bool submitRigidDual();
  void finishPreparedRigidSolveSynchronously();

  AvbdSolver &mSolver;
  AvbdIslandBatch mBatch;
  PxReal mDt;
  PxVec3 mGravity;
  AvbdRigidSolveContext mRigidContext;
  AvbdSolverStats mStats;
  PxU32 mCurrentWave;
  PxU32 mCurrentColor;
  PxU32 mGpuWaveEpoch;
  RigidAsyncPhase mRigidPhase;
  bool mRigidStarted;
  bool mRigidAsync;
  bool mRigidWaiting;
  bool mRigidSubmitHold;
  bool mRigidUsesBodyColors;
};

//=============================================================================
// Write Back Task
//=============================================================================

class AvbdWriteBackTask : public AvbdTask {
public:
  AvbdWriteBackTask(AvbdDynamicsContext &context, AvbdSolverBody *avbdBodies,
                    PxsRigidBody **rigidBodies,
                    const PxU32 *staticTouchCounts, PxU32 numBodies, PxReal dt,
                    bool enableStabilization, bool sleepingDisabled,
                    FeatherstoneArticulation **articulationForBody = nullptr,
                    PxU32 *linkIndexForBody = nullptr,
                    AvbdContactOutputTarget *contactOutputTargets = nullptr,
                    AvbdContactOutputResult *contactOutputResults = nullptr,
                    PxU32 contactOutputCount = 0,
                    const AvbdJointCacheCommitRange *jointCacheRanges = nullptr,
                    PxU32 jointCacheRangeCount = 0)
      : AvbdTask(context), mAvbdBodies(avbdBodies), mRigidBodies(rigidBodies),
        mStaticTouchCounts(staticTouchCounts), mNumBodies(numBodies), mDt(dt),
        mEnableStabilization(enableStabilization),
        mSleepingDisabled(sleepingDisabled),
        mArticulationForBody(articulationForBody),
        mLinkIndexForBody(linkIndexForBody),
        mContactOutputTargets(contactOutputTargets),
        mContactOutputResults(contactOutputResults),
        mContactOutputCount(contactOutputCount),
        mJointCacheRanges(jointCacheRanges),
        mJointCacheRangeCount(jointCacheRangeCount) {}

  virtual void run() override; // Implemented in cpp

  virtual const char *getName() const override { return "AvbdWriteBackTask"; }

private:
  AvbdSolverBody *mAvbdBodies;
  PxsRigidBody **mRigidBodies;
  const PxU32 *mStaticTouchCounts;
  PxU32 mNumBodies;
  PxReal mDt;
  bool mEnableStabilization;
  bool mSleepingDisabled;
  FeatherstoneArticulation **mArticulationForBody;
  PxU32 *mLinkIndexForBody;
  AvbdContactOutputTarget *mContactOutputTargets;
  AvbdContactOutputResult *mContactOutputResults;
  PxU32 mContactOutputCount;
  const AvbdJointCacheCommitRange *mJointCacheRanges;
  PxU32 mJointCacheRangeCount;
};

//=============================================================================
// Coordinator Task
//=============================================================================

class AvbdCoordinatorTask : public AvbdTask {
public:
  AvbdCoordinatorTask(AvbdDynamicsContext &context, PxBaseTask *continuation)
      : AvbdTask(context), mContinuation(continuation) {}

  virtual void run() override;

  virtual const char *getName() const override { return "AvbdCoordinatorTask"; }

  PxBaseTask *getContinuation() const { return mContinuation; }

private:
  PxBaseTask *mContinuation;
};

//=============================================================================
// Task Factory
//=============================================================================

class AvbdTaskFactory {
public:
  AvbdTaskFactory(PxTaskManager *taskManager, PxAllocatorCallback &allocator)
      : mTaskManager(taskManager), mAllocator(allocator) {}

  AvbdSolveIslandTask *createSolveTask(AvbdDynamicsContext &context,
                                       AvbdSolver &solver,
                                       const AvbdIslandBatch &batch, PxReal dt,
                                       const PxVec3 &gravity) {
    void *mem = mAllocator.allocate(sizeof(AvbdSolveIslandTask),
                                    "AvbdSolveIslandTask", __FILE__, __LINE__);
    return PX_PLACEMENT_NEW(mem, AvbdSolveIslandTask)(
        context, solver, batch, dt, gravity);
  }

  AvbdRigidBodyRangeTask *createRigidBodyRangeTask(
      AvbdDynamicsContext &context, AvbdSolver &solver,
      AvbdRigidSolveContext &solveContext, const PxU32 *bodyOrder,
      PxU32 begin, PxU32 end) {
    void *mem = mAllocator.allocate(sizeof(AvbdRigidBodyRangeTask),
                                    "AvbdRigidBodyRangeTask", __FILE__,
                                    __LINE__);
    return PX_PLACEMENT_NEW(mem, AvbdRigidBodyRangeTask)(
        context, solver, solveContext, bodyOrder, begin, end);
  }

  AvbdRigidDualRangeTask *createRigidDualRangeTask(
      AvbdDynamicsContext &context, AvbdSolver &solver,
      AvbdRigidSolveContext &solveContext, PxU32 begin, PxU32 end) {
    void *mem = mAllocator.allocate(sizeof(AvbdRigidDualRangeTask),
                                    "AvbdRigidDualRangeTask", __FILE__,
                                    __LINE__);
    return PX_PLACEMENT_NEW(mem, AvbdRigidDualRangeTask)(
        context, solver, solveContext, begin, end);
  }

  AvbdWriteBackTask *
  createWriteBackTask(AvbdDynamicsContext &context, AvbdSolverBody *avbdBodies,
                      PxsRigidBody **rigidBodies,
                      const PxU32 *staticTouchCounts, PxU32 numBodies,
                      PxReal dt, bool enableStabilization,
                      bool sleepingDisabled,
                      FeatherstoneArticulation **articulationForBody = nullptr,
                      PxU32 *linkIndexForBody = nullptr,
                      AvbdContactOutputTarget *contactOutputTargets = nullptr,
                      AvbdContactOutputResult *contactOutputResults = nullptr,
                      PxU32 contactOutputCount = 0,
                      const AvbdJointCacheCommitRange *jointCacheRanges = nullptr,
                      PxU32 jointCacheRangeCount = 0) {
    void *mem = mAllocator.allocate(sizeof(AvbdWriteBackTask),
                                    "AvbdWriteBackTask", __FILE__, __LINE__);
    return PX_PLACEMENT_NEW(mem, AvbdWriteBackTask)(
        context, avbdBodies, rigidBodies, staticTouchCounts, numBodies, dt,
        enableStabilization, sleepingDisabled, articulationForBody,
        linkIndexForBody, contactOutputTargets, contactOutputResults,
        contactOutputCount, jointCacheRanges, jointCacheRangeCount);
  }

  AvbdCoordinatorTask *createCoordinatorTask(AvbdDynamicsContext &context,
                                             PxBaseTask *continuation) {
    void *mem = mAllocator.allocate(sizeof(AvbdCoordinatorTask),
                                    "AvbdCoordinatorTask", __FILE__, __LINE__);
    return PX_PLACEMENT_NEW(mem, AvbdCoordinatorTask)(context, continuation);
  }

  template <typename T> void destroyTask(T *task) {
    if (task) {
      task->~T();
      mAllocator.deallocate(task);
    }
  }

  PxTaskManager *getTaskManager() const { return mTaskManager; }

private:
  PxTaskManager *mTaskManager;
  PxAllocatorCallback &mAllocator;
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_TASKS_H
