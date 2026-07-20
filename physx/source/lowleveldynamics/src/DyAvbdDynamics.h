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

#include <atomic>

namespace physx {

struct PxsBodyCore;

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

  // Friend function for lambda cache write-back from task (needs access to
  // private mLambdaCache)
  friend void writeLambdaToCache(AvbdDynamicsContext &ctx,
                                 AvbdContactConstraint *constraints,
                                 PxU32 numConstraints, PxU32 numBodies);
  friend void writeJointLambdaToCache(AvbdDynamicsContext &ctx,
                                      AvbdD6JointConstraint *constraints,
                                      PxU32 numConstraints);

public:
  AvbdDynamicsContext(PxcNpMemBlockPool *memBlockPool,
                      PxcScratchAllocator &scratchAllocator,
                      Cm::FlushPool &taskPool, PxvSimStats &simStats,
                      PxTaskManager *taskManager,
                      Cm::VirtualAllocatorCallback &allocator,
                      PxsMaterialManager *materialManager,
                      IG::SimpleIslandManager &islandManager, PxU64 contextID,
                      PxReal maxBiasCoefficient, PxReal lengthScale,
                      PxSceneFlags sceneFlags);

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
  PxAllocatorCallback &getAllocator() { return mAllocatorAdapter; }

  void beginIterationDiagnosticsFrame();
  void recordIterationDiagnostics(PxU32 requestedIterations,
                                  const AvbdSolverStats &stats,
                                  bool hasJointConstraints,
                                  const AvbdD6JointConstraint *d6Joints = nullptr,
                                  PxU32 numD6 = 0);
  void flushIterationDiagnosticsFrame();

  virtual void update(Cm::FlushPool &flushPool, PxBaseTask *continuation,
                      PxBaseTask *postPartitioningTask,
                      PxBaseTask *processLostTouchTask,
                      PxvNphaseImplementationContext *nPhaseContext,
                      PxU32 maxPatchesPerCM, PxU32 maxArticulationLinks,
                      PxReal dt, const PxVec3 &gravity,
                      Cm::PinnableBitMap &changedHandleMap) override;

  virtual void mergeResults() override;

  virtual void setSimulationController(
      PxsSimulationController *simulationController) override;

  virtual PxSolverType::Enum getSolverType() const override {
    return PxSolverType::eAVBD;
  }

  virtual void setConstraintConcreteType(PxU32 id, PxU16 type) override {
    if (id >= mConstraintConcreteTypes.size()) {
      const PxU32 oldSize = mConstraintConcreteTypes.size();
      mConstraintConcreteTypes.resize(id + 1);
      for (PxU32 i = oldSize; i <= id; ++i)
        mConstraintConcreteTypes[i] = 0;
    }
    mConstraintConcreteTypes[id] = type;
  }

  virtual void clearConstraintConcreteType(PxU32 id) override {
    if (id < mConstraintConcreteTypes.size())
      mConstraintConcreteTypes[id] = 0;
  }

  //-------------------------------------------------------------------------
  // Lambda Warm-Starting Cache (Public for external access)
  //-------------------------------------------------------------------------

  /**
   * @brief Cached lambda values for warm-starting across frames
   *
   * Each contact manager owns a fixed, disjoint group of direct-mapped slots.
   * Rows are selected by a stable contact-identity hash and the stored key is
   * validated before restore.  This prevents patch reordering and recycled
   * contact-manager indices from applying another row's dual state.
   */
  struct CachedLambda {
    PxU64 key;             //!< Stable contact identity for slot validation
    PxReal lambda;         //!< Normal constraint lambda
    PxReal tangentLambda0; //!< Friction lambda 1
    PxReal tangentLambda1; //!< Friction lambda 2
    PxReal penalty; //!< Adaptive penalty for normal (persists across frames)
    PxReal tangentPenalty0; //!< Adaptive penalty for tangent 0
    PxReal tangentPenalty1; //!< Adaptive penalty for tangent 1
    PxVec3 prevStaticWorldPoint; //!< Deformable/static anchor for friction
    PxU8 frameAge;          //!< Frames since last update (0 = current frame)
    PxU8 stick;             //!< Coulomb stick flag (static μ next frame)
    PxU8 padding[2];
  };

  PxArray<CachedLambda> mLambdaCache; //!< Per-contact lambda storage
  bool mEnableLambdaWarmStart; //!< Enable lambda warm-starting (default: true)

  struct CachedJointLambda {
    PxU64 key; //!< Stable joint identity used to validate hashed cache slots
    PxVec3 lambdaLinear;
    PxVec3 lambdaAngular;
    PxVec3 lambdaDriveLinear;
    PxVec3 lambdaDriveAngular;
    PxReal coneLambda;
    PxU8 frameAge;
    PxU8 padding[3];
  };

  PxArray<CachedJointLambda> mJointLambdaCache; //!< Per-D6 warm-start storage

  static const PxU32 CONTACT_CACHE_SLOTS_PER_CM =
      64; //!< Direct-mapped cache slots owned by each ContactManager
  static const PxU32 JOINT_LAMBDA_CACHE_SIZE =
      8192; //!< Fixed-size hashed cache for D6 joints
  static const PxU8 LAMBDA_MAX_AGE = 3; //!< Max frames to keep cached lambda
  static constexpr PxReal LAMBDA_WARMSTART_SCALE =
      0.9f; //!< Damping for warm-started lambda

private:
  PX_FORCE_INLINE PxU16 getConstraintConcreteType(PxU32 id) const {
    return id < mConstraintConcreteTypes.size() ? mConstraintConcreteTypes[id]
                                                : PxU16(0);
  }

  struct CachedBodyVelocityHistory {
    PxU64 bodyCoreKey;
    PxU64 lastSeenFrame;
    PxVec3 linearVelocity;
  };

  // Fixed-size, direct-mapped storage keeps stale body identities bounded.
  // The exact bodyCore pointer is still validated before any history is used.
  static const PxU32 BODY_VELOCITY_HISTORY_CACHE_SIZE = 16384;
  PxArray<CachedBodyVelocityHistory> mBodyVelocityHistoryCache;
  PxU64 mBodyVelocityHistoryFrame;
  PxArray<PxU16> mConstraintConcreteTypes;

  //-------------------------------------------------------------------------
  // Internal Methods
  //-------------------------------------------------------------------------

  void restoreAndUpdateBodyVelocityHistory(const PxsBodyCore &bodyCore,
                                           AvbdSolverBody &solverBody);

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
  void prepareAvbdConstraints(const IG::IslandSim &islandSim,
                              AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
                              PxU32 bodyOffset,
                              AvbdD6JointConstraint *d6Constraints,
                              PxU32 &numD6, PxU32 maxD6,
                              AvbdGearJointConstraint *gearConstraints,
                              PxU32 &numGear, PxU32 maxGear, PxU32 islandIndex,
                              PxU32 *bodyRemapTable,
                              PxU32 *articulationFirstLinkIndex,
                              FeatherstoneArticulation **articulationByActiveIdx,
                              PxU32 numArticulations);

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

  class VirtualAllocatorAdapter : public PxAllocatorCallback {
  public:
    explicit VirtualAllocatorAdapter(Cm::VirtualAllocatorCallback &allocator);
    virtual void *allocate(size_t size, const char *, const char *file,
                           int line) override;
    virtual void deallocate(void *ptr) override;

  private:
    Cm::VirtualAllocatorCallback &mAllocator;
  };
  VirtualAllocatorAdapter mAllocatorAdapter;

  PxTaskManager *mTaskManager;   //!< Task manager for parallel execution
  AvbdTaskFactory *mTaskFactory; //!< Factory for creating AVBD tasks
  bool mFrictionEveryIteration; //!< Apply friction every iteration
  bool mSolverInitialized;      //!< Whether solver has been initialized
  bool mIterationDiagnosticsEnabled; //!< Print AVBD iteration summaries when enabled via env
  bool mIterationDiagnosticsSequential; //!< Force sequential island solve for trustworthy diagnostics
  PxU32 mIterationDiagnosticsEvery; //!< Diagnostic print cadence in frames
  std::atomic<PxU64> mDiagIslandCount;
  std::atomic<PxU64> mDiagJointIslandCount;
  std::atomic<PxU64> mDiagRequestedIterations;
  std::atomic<PxU64> mDiagExecutedIterations;
  std::atomic<PxU64> mDiagEarlyStopIslands;
  std::atomic<PxU64> mDiagJointRequestedIterations;
  std::atomic<PxU64> mDiagJointExecutedIterations;
  std::atomic<PxU64> mDiagJointBudgetHitIslands;
  std::atomic<PxU64> mDiagJointEarlyStopIslands;
  std::atomic<PxU64> mDiagJointContactCount;
  std::atomic<PxU64> mDiagJointConstraintCount;
  std::atomic<PxU64> mDiagJointLockedLinearRows;
  std::atomic<PxU64> mDiagJointLimitedLinearRows;
  std::atomic<PxU64> mDiagJointLockedAngularRows;
  std::atomic<PxU64> mDiagJointLimitedAngularRows;
  std::atomic<PxU64> mDiagJointLinearDriveRows;
  std::atomic<PxU64> mDiagJointAngularDriveRows;
  std::atomic<PxU64> mDiagJointConeRows;
  std::atomic<PxU32> mDiagMaxRequestedIterations;
  std::atomic<PxU32> mDiagMaxExecutedIterations;
  std::atomic<PxU32> mDiagJointMaxExecutedIterations;
  std::atomic<PxU32> mDiagJointMaxLinearLambdaMilli;
  std::atomic<PxU32> mDiagJointMaxAngularLambdaMilli;
  std::atomic<PxU32> mDiagJointMaxLinearDriveLambdaMilli;
  std::atomic<PxU32> mDiagJointMaxAngularDriveLambdaMilli;
  std::atomic<PxU32> mDiagJointMaxConeLambdaMilli;
  PxU32 mDiagSeqMaxLinearJointIndex;
  PxU32 mDiagSeqMaxLinearJointBodyA;
  PxU32 mDiagSeqMaxLinearJointBodyB;
  PxU32 mDiagSeqMaxLinearDriveJointIndex;
  PxU32 mDiagSeqMaxLinearDriveJointBodyA;
  PxU32 mDiagSeqMaxLinearDriveJointBodyB;

  //!< Track heap fallback allocations for cleanup at frame end
  //!< No mutex needed since update() and mergeResults() are called from
  //!< single-threaded contexts
  PxArray<void *> mHeapFallbackAllocations;
};

// Lambda cache write-back function (declared friend in AvbdDynamicsContext)
void writeLambdaToCache(AvbdDynamicsContext &ctx,
                        AvbdContactConstraint *constraints,
                        PxU32 numConstraints, PxU32 numBodies);

void restoreJointLambdaFromCache(AvbdDynamicsContext &ctx,
                                 AvbdD6JointConstraint &constraint,
                                 PxU64 cacheKey);

void writeJointLambdaToCache(AvbdDynamicsContext &ctx,
                             AvbdD6JointConstraint *constraints,
                             PxU32 numConstraints);

/**
 * @brief Factory function to create AVBD dynamics context
 */
Context *createAVBDDynamicsContext(
    PxcNpMemBlockPool *memBlockPool, PxcScratchAllocator &scratchAllocator,
    Cm::FlushPool &taskPool, PxvSimStats &simStats, PxTaskManager *taskManager,
    Cm::VirtualAllocatorCallback &allocator,
    PxsMaterialManager *materialManager, IG::SimpleIslandManager &islandManager,
    PxU64 contextID, PxReal maxBiasCoefficient, PxReal lengthScale,
    PxSceneFlags sceneFlags);

} // namespace Dy

} // namespace physx

#endif // DY_AVBD_DYNAMICS_H
