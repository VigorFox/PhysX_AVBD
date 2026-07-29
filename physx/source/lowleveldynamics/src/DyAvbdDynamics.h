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

  struct CachedContactManagerState {
    PxU64 key;      //!< Exact manager/body identity for slot validation
    PxU8 frameAge; //!< 1 means the manager was active last frame
    PxU8 padding[7];
  };

  PxArray<CachedContactManagerState>
      mContactManagerStateCache; //!< One persistent state per CM index

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

  // Open-addressed storage starts at a bounded minimum and grows before
  // gather to keep load at or below 0.5.  The exact bodyCore pointer is
  // validated before any history is used, and serial gather resolves
  // collisions without evicting another active body's previous-frame state.
  static const PxU32 BODY_VELOCITY_HISTORY_CACHE_SIZE = 16384;
  PxArray<CachedBodyVelocityHistory> mBodyVelocityHistoryCache;
  PxU64 mBodyVelocityHistoryFrame;
  PxArray<PxU16> mConstraintConcreteTypes;

  //-------------------------------------------------------------------------
  // Internal Methods
  //-------------------------------------------------------------------------

  void restoreAndUpdateBodyVelocityHistory(const PxsBodyCore &bodyCore,
                                           AvbdSolverBody &solverBody);
  void ensureBodyVelocityHistoryCapacity(PxU32 bodyCount);

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
  PxU32 prepareAvbdContacts(const IG::IslandSim &islandSim, PxReal dt,
                            AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
                            AvbdContactConstraint *constraints,
                            PxU32 maxConstraints, PxU32 startContactIdx,
                            PxU32 numContactsToProcess, PxU32 bodyOffset);

  /**
   * @brief Convert PhysX constraints (joints) to AVBD joint constraints
   */
  void prepareAvbdConstraints(const IG::IslandSim &islandSim,
                              PxReal dt,
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
  bool mNormalRowDiagnosticsEnabled; //!< Emit detailed body-static row evidence
  PxU32 mIterationDiagnosticsEvery; //!< Diagnostic print cadence in frames
  std::atomic<PxU64> mDiagIslandCount;
  std::atomic<PxU64> mDiagJointIslandCount;
  std::atomic<PxU64> mDiagRequestedIterations;
  std::atomic<PxU64> mDiagExecutedIterations;
  std::atomic<PxU64> mDiagEarlyStopIslands;
  std::atomic<PxU64> mDiagVelocityObjectivePositionRows;
  std::atomic<PxU64> mDiagVelocityObjectivePointRows;
  std::atomic<PxU64> mDiagVelocityObjectiveManifoldRows;
  std::atomic<PxU64> mDiagVelocityObjectiveComponentRows;
  std::atomic<PxU64> mDiagVelocityObjectiveJointRows;
  std::atomic<PxU64> mDiagVelocityObjectiveUnsupportedRows;
  std::atomic<PxU64> mDiagVelocityObjectiveLegacyRows;
  std::atomic<PxU64> mDiagVelocityObjectiveInvalidRows;
  std::atomic<PxU64> mDiagVelocityObjectiveFingerprint;
  std::atomic<PxU64> mDiagContactObjectivePositionSlots;
  std::atomic<PxU64> mDiagContactObjectivePointSlots;
  std::atomic<PxU64> mDiagContactObjectiveManifoldSlots;
  std::atomic<PxU64> mDiagContactObjectiveComponentSlots;
  std::atomic<PxU64> mDiagContactObjectiveJointSlots;
  std::atomic<PxU64> mDiagContactObjectiveUnsupportedSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacySlots;
  std::atomic<PxU64> mDiagContactObjectiveInvalidSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyNormalSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyTangentSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyRigidStaticTangentSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyDynamicTangentSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyDeformableTangentSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyJointMixedTangentSlots;
  std::atomic<PxU64> mDiagContactObjectiveLegacyOtherTangentSlots;
  std::atomic<PxU64> mDiagContactObjectiveFingerprint;
  std::atomic<PxU64> mDiagJointObjectivePositionRows;
  std::atomic<PxU64> mDiagJointObjectiveFinalizeRows;
  std::atomic<PxU64> mDiagJointObjectiveUnsupportedRows;
  std::atomic<PxU64> mDiagJointObjectiveLegacyRows;
  std::atomic<PxU64> mDiagJointObjectiveInvalidRows;
  std::atomic<PxU64> mDiagJointObjectiveFingerprint;
  std::atomic<PxU64> mDiagBodyStaticNormalAlRows;
  std::atomic<PxU64> mDiagBodyStaticNormalAlEvaluations;
  std::atomic<PxU64> mDiagBodyStaticDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticDepenetrationEligibleRows;
  std::atomic<PxU64> mDiagBodyStaticDepenetrationFiniteImpulseSkips;
  std::atomic<PxU64> mDiagBodyStaticDepenetrationAuthoredFiniteImpulseSkips;
  std::atomic<PxU64> mDiagBodyStaticMaterialVelocityCorrections;
  std::atomic<PxU64> mDiagBodyStaticRestitutionCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalWarmstartHits;
  std::atomic<PxU64> mDiagBodyStaticNormalWarmstartMisses;
  std::atomic<PxU64> mDiagBodyStaticNormalWarmstartAge0;
  std::atomic<PxU64> mDiagBodyStaticNormalWarmstartAge1;
  std::atomic<PxU64> mDiagBodyStaticNormalWarmstartAge2;
  std::atomic<PxU64> mDiagBodyStaticNormalWarmstartAge3;
  std::atomic<PxU64> mDiagBodyStaticNormalManagerOnsetRows;
  std::atomic<PxU64> mDiagBodyStaticNormalManagerSupportRows;
  std::atomic<PxU64> mDiagBodyStaticNormalManagerAge0;
  std::atomic<PxU64> mDiagBodyStaticNormalManagerAge1;
  std::atomic<PxU64> mDiagBodyStaticNormalManagerAge2;
  std::atomic<PxU64> mDiagBodyStaticNormalManagerAge3;
  std::atomic<PxU64> mDiagBodyStaticNormalRowMissOnManagerSupportRows;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetFinalizeBodies;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportFinalizeBodies;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetFinalizeCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportFinalizeCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetDepenetrationEligibleRows;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportDepenetrationEligibleRows;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetShallowDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetDeepDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportShallowDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportDeepDepenetrationCorrections;
  std::atomic<PxU64> mDiagBodyStaticMaterialFiniteBudgetRows;
  std::atomic<PxU64> mDiagBodyStaticMaterialUnlimitedBudgetRows;
  std::atomic<PxU64> mDiagContactFrictionTargetAlEvaluations;
  std::atomic<PxU64> mDiagBodyStaticFrictionTargetRows;
  std::atomic<PxU64> mDiagBodyStaticFrictionTargetCorrections;
  std::atomic<PxU64> mDiagBodyStaticFrictionFallbackRows;
  std::atomic<PxU64> mDiagBodyStaticFrictionFallbackCorrections;
  std::atomic<PxU64> mDiagContactTargetNormalProjectionRows;
  std::atomic<PxU64> mDiagContactTargetNormalCorrections;
  std::atomic<PxU64> mDiagContactTargetTangentRows;
  std::atomic<PxU64> mDiagContactTargetTangentCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableAlRows;
  std::atomic<PxU64> mDiagSurfaceDeformableAlEvaluations;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentCandidates;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentRows;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentEvaluations;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentMixedRejectRows;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentShellRejectRows;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentTargetRejectRows;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentRestitutionRejectRows;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentFiniteRejectRows;
  std::atomic<PxU64> mDiagSurfaceDeformablePositionTangentScaleRejectRows;
  std::atomic<PxU64> mDiagSurfaceDeformableStrippedRows;
  std::atomic<PxU64> mDiagSurfaceDeformableShellSuppressedPrimalRows;
  std::atomic<PxU64> mDiagSurfaceDeformableDepenetrationCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFrictionRawRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFrictionDominantRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFrictionFewContactRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFrictionMultiCornerRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFrictionCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeBodies;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeSpatialCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeComFallbackCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeSecondaryRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeSecondaryResidualSeparationRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeManifoldBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldOneRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldTwoRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldThreeRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldFourRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldOverFourRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldFiveToEightRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldNineToSixteenRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldOverSixteenRowBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldMixedScaleBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldRankDeficientBodies;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeManifoldAliasRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldDynamicIncidentBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldRigidStaticIncidentBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeManifoldNonOwnerDeformableIncidentBodies;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeComponents;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeComponentOneBody;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeComponentTwoBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentThreeToFourBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentFiveToEightBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentNineToSixteenBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentSeventeenToThirtyTwoBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentOverThirtyTwoBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentOneToEightRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentNineToSixteenRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentSeventeenToThirtyTwoRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentThirtyThreeToSixtyFourRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentOverSixtyFourRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentRestitution;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentFiniteImpulse;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentTargetVelocity;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentMixedScale;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentRigidStatic;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentNonOwnerDeformable;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentJointIsland;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentLockedDof;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeComponentNonDynamicBody;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeBudgetDiagRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagNoCorrectionRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagZeroBudgetRequiredRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagWithinBudgetRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagOverBudgetRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagUnsupportedRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagComponentsWithinBudget;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagComponentsOverBudget;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeBudgetDiagComponentsUnsupported;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowComponents;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowNoCorrection;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowSolved;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowCommitCapable;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowBudgetExhausted;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowInfeasible;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowResidualUnclassified;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowNumericalFailure;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowIterationLimit;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowUnsupported;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowUnsupportedFastImpact;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowUnsupportedSnapshot;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowLowerRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowFreeRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowUpperRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeComponents;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeShadowMatrixFreeRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeNoCorrection;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeSolved;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeBudgetExhausted;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeInfeasible;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeResidualUnclassified;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeNumericalFailure;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeIterationLimit;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeIterations;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost2x;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost16x;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktOver16x;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeCommittedComponents;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeOracleComponents;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeOracleRows;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeOracleMatched;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeOracleMismatched;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeShadowMatrixFreeOracleSkipped;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizePreOwnerBodies;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeLegacyOwnerBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeOwnerDiscoveryMismatchBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeProbeEligibleComponents;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeProbeCommittedComponents;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeProbeCommittedRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeProbeCommittedBodies;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeProbeReplacedOwnerBodies;
  std::atomic<PxU64> mDiagSurfaceDeformableAlDepenetrationRows;
  std::atomic<PxU64> mDiagSurfaceDeformableAlFinalizeRows;
  std::atomic<PxU64> mDiagSurfaceDeformableDepenetrationFinalizeRows;
  std::atomic<PxU64> mDiagSurfaceDeformableAlDepenetrationFinalizeRows;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeContactFalsePositiveCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeContactResidualSeparationCorrections;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeContactReversalCorrections;
  std::atomic<PxU64> mDiagSurfaceShellContacts;
  std::atomic<PxU64> mDiagSurfaceShellDepenetrationCorrections;
  std::atomic<PxU64> mDiagSurfaceShellFrictionRows;
  std::atomic<PxU64> mDiagSurfaceShellFrictionCorrections;
  std::atomic<PxU64> mDiagSurfaceShellFinalizeBodies;
  std::atomic<PxU64> mDiagSurfaceShellFinalizeCorrections;
  std::atomic<PxU64> mDiagBodyStaticDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticMaterialVelocityDeltaNanos;
  std::atomic<PxU64> mDiagBodyStaticFrictionTargetImpulseNanos;
  std::atomic<PxU64> mDiagBodyStaticFrictionFallbackImpulseNanos;
  std::atomic<PxU64> mDiagContactTargetNormalImpulseNanos;
  std::atomic<PxU64> mDiagContactTargetTangentImpulseNanos;
  std::atomic<PxU64> mDiagSurfaceDeformableDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagSurfaceDeformableFrictionImpulseNanos;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeDeltaNanos;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeContactPreSeparationNanos;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeContactPostSeparationNanos;
  std::atomic<PxU64> mDiagSurfaceDeformableFinalizeContactPostApproachNanos;
  std::atomic<PxU64>
      mDiagSurfaceDeformableFinalizeSecondaryResidualSeparationNanos;
  std::atomic<PxU64> mDiagSurfaceShellDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagSurfaceShellFrictionImpulseNanos;
  std::atomic<PxU64> mDiagSurfaceShellFinalizeDeltaNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalRestoredLambdaMaxNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalRestoredPenaltyMaxNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalInitialPenaltyMaxNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalPreAlRawPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalPostAlRawPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalAlphaC0OffsetNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalPreAlPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalPostAlPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalPostAlSeparationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalAlOutwardDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalAlInwardDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticMaterialPoseSeparatingVelocityNanos;
  std::atomic<PxU64> mDiagBodyStaticMaterialAllowedSeparatingVelocityNanos;
  std::atomic<PxU64> mDiagBodyStaticMaterialFiniteRemainingImpulseNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetPreAlRawPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetPreAlPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetPostAlRawPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetPostAlPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetAlphaC0OffsetNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetAlOutwardDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportPreAlRawPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportPreAlPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportPostAlRawPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportPostAlPenetrationNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportAlphaC0OffsetNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportAlOutwardDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetPoseSeparatingVelocityNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportPoseSeparatingVelocityNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetFinalizeDeltaNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportFinalizeDeltaNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetShallowDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalOnsetDeepDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportShallowDepenetrationDistanceNanos;
  std::atomic<PxU64> mDiagBodyStaticNormalSupportDeepDepenetrationDistanceNanos;
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
  std::atomic<PxU64> mDiagBodyStaticDepenetrationMaxCorrectionNanos;
  std::atomic<PxU64> mDiagBodyStaticMaterialVelocityMaxDeltaNanos;
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

void writeContactImpulseToOutput(const AvbdContactConstraint *constraints,
                                 PxU32 numConstraints, PxU32 numBodies,
                                 PxReal dt);

void restoreJointLambdaFromCache(AvbdDynamicsContext &ctx,
                                 AvbdD6JointConstraint &constraint,
                                 PxU64 cacheKey);

void writeJointLambdaToCache(AvbdDynamicsContext &ctx,
                             AvbdD6JointConstraint *constraints,
                             PxU32 numConstraints);

void writeJointConstraintWriteback(AvbdDynamicsContext &ctx,
                                   const AvbdD6JointConstraint *constraints,
                                   PxU32 numConstraints, PxReal dt);

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
