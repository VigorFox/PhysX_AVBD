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

#ifndef DY_AVBD_DYNAMICS_H
#define DY_AVBD_DYNAMICS_H

#include "DyAvbdParallel.h"
#include "DyAvbdSolver.h"
#include "DyAvbdTasks.h"
#include "DyDynamicsBase.h"

namespace physx {

namespace Dy {

/**
 * @brief AVBD Dynamics Context
 *
 * Implements the dynamics pipeline using the AVBD (Augmented Variable Block
 * Descent) solver. This is an alternative to PGS and TGS solvers with different
 * convergence characteristics.
 */
class AvbdDynamicsContext : public DynamicsContextBase {
  PX_NOCOPY(AvbdDynamicsContext)
public:
  AvbdDynamicsContext(PxcNpMemBlockPool *memBlockPool,
                      PxcScratchAllocator &scratchAllocator,
                      Cm::FlushPool &taskPool, PxvSimStats &simStats,
                      PxTaskManager *taskManager,
                      PxVirtualAllocatorCallback *allocatorCallback,
                      PxsMaterialManager *materialManager,
                      IG::SimpleIslandManager &islandManager, PxU64 contextID,
                      bool enableStabilization, bool useEnhancedDeterminism,
                      bool solveArticulationContactLast,
                      PxReal maxBiasCoefficient, bool frictionEveryIteration,
                      PxReal lengthScale, bool isResidualReportingEnabled);

  virtual ~AvbdDynamicsContext();

  //-------------------------------------------------------------------------
  // Context Virtual Methods
  //-------------------------------------------------------------------------

  virtual void destroy() override;

  /**
   * @brief Destroy AVBD task (called by task->release())
   */
  void destroyTask(AvbdTask *task);
  
  /**
   * @brief Get allocator callback for constraint map cleanup
   */
  PxAllocatorCallback& getAllocator() { return *reinterpret_cast<PxAllocatorCallback*>(mAllocatorCallback); }

  virtual void update(Cm::FlushPool &flushPool, PxBaseTask *continuation,
                      PxBaseTask *postPartitioningTask,
                      PxBaseTask *processLostTouchTask,
                      PxvNphaseImplementationContext *nPhaseContext,
                      PxU32 maxPatchesPerCM, PxU32 maxArticulationLinks,
                      PxReal dt, const PxVec3 &gravity,
                      PxBitMapPinned &changedHandleMap) override;

  virtual void mergeResults() override;

  virtual void setSimulationController(
      PxsSimulationController *simulationController) override;

  virtual PxSolverType::Enum getSolverType() const override {
    return PxSolverType::eAVBD;
  }

private:
  //-------------------------------------------------------------------------
  // Internal Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Solve constraints for a single island using AVBD algorithm
   */
  void solveIsland(const IG::IslandSim &islandSim, PxReal dt,
                   const PxVec3 &gravity);

  /**
   * @brief Convert PhysX contacts to AVBD contact constraints
   */
  void prepareContacts(const IG::IslandSim &islandSim);

  /**
   * @brief Convert PhysX bodies to AVBD solver bodies
   */
  void setupBodies(AvbdSolverBody *avbdBodies, PxsRigidBody **rigidBodies,
                   PxU32 numBodies, PxReal dt, const PxVec3 &gravity);

  /**
   * @brief Write AVBD solver results back to PhysX bodies
   */
  void writeBackBodies(AvbdSolverBody *avbdBodies, PxsRigidBody **rigidBodies,
                       PxU32 numBodies);

  /**
   * @brief Convert PhysX contacts to AVBD contact constraints
   * @param avbdBodies Island-local bodies array (starts at bodyOffset in
   * global)
   * @param islandBodyCount Number of bodies in this island
   * @param constraints Output constraint array
   * @param maxConstraints Maximum constraints to write
   * @param startContactIdx Start index in mContactList
   * @param numContactsToProcess Number of contacts to process
   * @param bodyOffset Global index of first body in this island
   * @return Number of constraints created
   */
  PxU32 prepareAvbdContacts(AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
                            AvbdContactConstraint *constraints,
                            PxU32 maxConstraints, PxU32 startContactIdx,
                            PxU32 numContactsToProcess, PxU32 bodyOffset);

  /**
   * @brief Convert PhysX constraints (joints) to AVBD joint constraints
   */
  void prepareAvbdConstraints(
      const IG::IslandSim &islandSim, AvbdSolverBody *avbdBodies,
      PxU32 islandBodyCount, PxU32 bodyOffset,
      AvbdSphericalJointConstraint *sphericalConstraints, PxU32 &numSpherical,
      PxU32 maxSpherical, AvbdFixedJointConstraint *fixedConstraints,
      PxU32 &numFixed, PxU32 maxFixed,
      AvbdRevoluteJointConstraint *revoluteConstraints, PxU32 &numRevolute,
      PxU32 maxRevolute, AvbdPrismaticJointConstraint *prismaticConstraints,
      PxU32 &numPrismatic, PxU32 maxPrismatic,
      AvbdD6JointConstraint *d6Constraints, PxU32 &numD6,
      PxU32 maxD6, PxU32 islandIndex,
      PxU32 *bodyRemapTable);

  //-------------------------------------------------------------------------
  // Member Variables
  //-------------------------------------------------------------------------

  AvbdSolver mSolver;                       //!< AVBD solver instance
  AvbdParallelColoring mConstraintColoring; //!< Constraint graph coloring
  PxcScratchAllocator &mScratchAllocator;   //!< Scratch memory allocator

  class ScratchAllocatorAdapter : public PxAllocatorCallback {
  public:
    ScratchAllocatorAdapter(PxcScratchAllocator &scratch);
    virtual void *allocate(size_t size, const char *, const char *,
                           int) override;
    virtual void deallocate(void *) override;
    PxcScratchAllocator &mScratch;
  };
  ScratchAllocatorAdapter mScratchAdapter;

  PxTaskManager *mTaskManager;   //!< Task manager for parallel execution
  AvbdTaskFactory *mTaskFactory; //!< Factory for creating AVBD tasks
  PxVirtualAllocatorCallback *mAllocatorCallback; //!< Memory allocator
  bool mFrictionEveryIteration; //!< Apply friction every iteration
  bool mSolverInitialized;      //!< Whether solver has been initialized
  
  //!< Track heap fallback allocations for cleanup at frame end
  //!< No mutex needed since update() and mergeResults() are called from single-threaded contexts
  PxArray<void*> mHeapFallbackAllocations;
};

/**
 * @brief Factory function to create AVBD dynamics context
 */
Context *createAVBDDynamicsContext(
  PxcNpMemBlockPool *memBlockPool, PxcScratchAllocator &scratchAllocator,
  Cm::FlushPool &taskPool, PxvSimStats &simStats, PxTaskManager *taskManager,
  PxVirtualAllocatorCallback *allocatorCallback,
  PxsMaterialManager *materialManager, IG::SimpleIslandManager &islandManager,
  PxU64 contextID, bool enableStabilization, bool useEnhancedDeterminism,
  bool solveArticulationContactLast, PxReal maxBiasCoefficient,
  bool frictionEveryIteration, PxReal lengthScale,
  bool isResidualReportingEnabled);

} // namespace Dy

} // namespace physx

#endif // DY_AVBD_DYNAMICS_H
