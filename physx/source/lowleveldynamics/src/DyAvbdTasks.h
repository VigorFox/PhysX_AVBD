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

#include "DyAvbdConstraint.h"
#include "DyAvbdSolver.h"
#include "DyAvbdSolverBody.h"
#include "foundation/PxSimpleTypes.h"
#include "task/PxTask.h"
#include <cstdio>
#include "DyFeatherstoneArticulation.h"

namespace physx {

class PxTaskManager;
class PxsRigidBody;

namespace Dy {

class AvbdDynamicsContext;
class FeatherstoneArticulation;

//=============================================================================
// Task Data Structures
//=============================================================================

struct AvbdIslandBatch {
  AvbdSolverBody *bodies;
  PxU32 numBodies;

  AvbdContactConstraint *constraints;
  PxU32 numConstraints;

  // Joint Constraints
  AvbdSphericalJointConstraint *sphericalJoints;
  PxU32 numSpherical;

  AvbdFixedJointConstraint *fixedJoints;
  PxU32 numFixed;

  AvbdRevoluteJointConstraint *revoluteJoints;
  PxU32 numRevolute;

  AvbdPrismaticJointConstraint *prismaticJoints;
  PxU32 numPrismatic;

  AvbdD6JointConstraint *d6Joints;
  PxU32 numD6;

  PxU32 islandStart;
  PxU32 islandEnd;

  // Pre-computed constraint coloring (for large islands)
  // These are computed in the single-threaded update() phase to avoid
  // race conditions when multiple island tasks run concurrently.
  AvbdColorBatch
      *colorBatches; //!< Array of color batches (nullptr if not colored)
  PxU32 numColors;   //!< Number of colors used (0 if not colored)
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

//=============================================================================
// Island Solve Task
//=============================================================================

class AvbdSolveIslandTask : public AvbdTask {
public:
  AvbdSolveIslandTask(AvbdDynamicsContext &context, AvbdSolver &solver,
                      const AvbdIslandBatch &batch, PxReal dt,
                      const PxVec3 &gravity)
      : AvbdTask(context), mSolver(solver), mBatch(batch), mDt(dt),
        mGravity(gravity) {}

  virtual void run() override {
    printf("[AVBD] AvbdSolveIslandTask::run() START - island=%u, bodies=%u, "
           "contacts=%u, spherical=%u, fixed=%u, revolute=%u, prismatic=%u, d6=%u\n",
           mBatch.islandStart, mBatch.numBodies, mBatch.numConstraints,
           mBatch.numSpherical, mBatch.numFixed, mBatch.numRevolute,
           mBatch.numPrismatic, mBatch.numD6);
    fflush(stdout);

    // Use solveWithJoints to handle all constraint types
    mSolver.solveWithJoints(
        mDt, mBatch.bodies, mBatch.numBodies, mBatch.constraints,
        mBatch.numConstraints, mBatch.sphericalJoints, mBatch.numSpherical,
        mBatch.fixedJoints, mBatch.numFixed, mBatch.revoluteJoints,
        mBatch.numRevolute, mBatch.prismaticJoints, mBatch.numPrismatic,
        mBatch.d6Joints, mBatch.numD6,
        mGravity, mBatch.colorBatches, mBatch.numColors);

    printf("[AVBD] AvbdSolveIslandTask::run() END - island=%u\n",
           mBatch.islandStart);
    fflush(stdout);
  }

  virtual const char *getName() const override { return "AvbdSolveIslandTask"; }

private:
  AvbdSolver &mSolver;
  AvbdIslandBatch mBatch;
  PxReal mDt;
  PxVec3 mGravity;
};

//=============================================================================
// Write Back Task
//=============================================================================

class AvbdWriteBackTask : public AvbdTask {
public:
  AvbdWriteBackTask(AvbdDynamicsContext &context, AvbdSolverBody *avbdBodies,
                    PxsRigidBody **rigidBodies, PxU32 numBodies,
                    FeatherstoneArticulation** articulationForBody = nullptr,
                    PxU32* linkIndexForBody = nullptr)
      : AvbdTask(context), mAvbdBodies(avbdBodies), mRigidBodies(rigidBodies),
        mNumBodies(numBodies), 
        mArticulationForBody(articulationForBody),
        mLinkIndexForBody(linkIndexForBody) {}

  virtual void run() override; // Implemented in cpp

  virtual const char *getName() const override { return "AvbdWriteBackTask"; }

private:
  AvbdSolverBody *mAvbdBodies;
  PxsRigidBody **mRigidBodies;
  PxU32 mNumBodies;
  FeatherstoneArticulation** mArticulationForBody;
  PxU32* mLinkIndexForBody;
};

//=============================================================================
// Coordinator Task
//=============================================================================

class AvbdCoordinatorTask : public AvbdTask {
public:
  AvbdCoordinatorTask(AvbdDynamicsContext &context, PxBaseTask *continuation)
      : AvbdTask(context), mContinuation(continuation) {}

  virtual void run() override {
    printf("[AVBD] AvbdCoordinatorTask::run() - sync point reached\n");
    fflush(stdout);
  }

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
    return PX_PLACEMENT_NEW(mem, AvbdSolveIslandTask)(context, solver, batch,
                                                      dt, gravity);
  }

  AvbdWriteBackTask *createWriteBackTask(AvbdDynamicsContext &context,
                                         AvbdSolverBody *avbdBodies,
                                         PxsRigidBody **rigidBodies,
                                         PxU32 numBodies,
                                         FeatherstoneArticulation** articulationForBody = nullptr,
                                         PxU32* linkIndexForBody = nullptr) {
    void *mem = mAllocator.allocate(sizeof(AvbdWriteBackTask),
                                    "AvbdWriteBackTask", __FILE__, __LINE__);
    return PX_PLACEMENT_NEW(mem, AvbdWriteBackTask)(context, avbdBodies,
                                                    rigidBodies, numBodies,
                                                    articulationForBody, linkIndexForBody);
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
