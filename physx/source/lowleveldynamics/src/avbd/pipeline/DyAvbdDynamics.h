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

#include "avbd/scheduling/DyAvbdParallel.h"
#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/pipeline/DyAvbdTasks.h"
#include "avbd/backend/gpu/DyAvbdGpuWaveBackend.h"
#include "DyDynamicsBase.h"

// Reading map:
//   1. Solver knobs: AvbdSolverConfig in DyAvbdTypes.h.
//   2. Public lifecycle: constructor/destroy/update/mergeResults below.
//   3. Runtime storage: private members at the end of AvbdDynamicsContext.
//      Profile data is task-local and reduced after the coordinator join; it
//      is not context state.
//
// The context owns the runtime bridge; it is not the configuration object.

namespace physx {

struct PxsBodyCore;
class PxsRigidBody;

// Immutable, serially captured narrow-phase input for the first contact-prep
// differential.  The payload deliberately contains no worker-owned output
// pointers: report/force streams use parent-owned target tokens, while all
// unsupported endpoint kinds fail closed to the legacy preparation path until
// their commit ABI is explicit.
struct AvbdContactPrepSnapshot {
  PxU64 contactManagerIdentity;
  PxReal restDistance;
  PxU32 contactManagerIndex;
  PxU32 npIndex;
  PxU64 solverBody0;
  PxU64 solverBody1;
  PxU8 indexType0;
  PxU8 indexType1;
  PxU8 nbPatches;
  PxU8 contactManagerAge;
  PxU16 nbContacts;
  PxU16 contactPointSize;
  // These byte streams are immutable for the update and borrowed from the
  // parent-owned narrow-phase output. A worker may only read them; the parent
  // keeps the source buffers alive through the post-solver continuation.
  const PxU8 *contactPatches;
  const PxU8 *contactPoints;
  // Parent-owned compact manifold points for this manager, borrowed read-only
  // until the island solve writes the same disjoint range back.
  const void *contactManifoldPoints;
  PxU32 outputTargetBase;
  PxU16 outputTargetCount;
  // Row-cardinality contract: captured from the immutable NP stream and
  // filled by the pure parser. Serial snapshot preparation can rebuild on the
  // parent; deferred island preparation reports an internal error and drops
  // the invalid frozen range instead of consuming partial rows.
  PxU16 expectedResponseRows;
  PxU16 emittedResponseRows;
  PxU8 outputTargetFlags;
  // Single-consumer commit token: set when this CM emitted at least one
  // response row. Serial preparation commits it on the parent; deferred
  // preparation commits its disjoint manager slot in the owning island task.
  PxU8 managerStateCommit;
  PxU8 eligible;
  PxU8 padding[2];

  AvbdContactPrepSnapshot()
      : contactManagerIdentity(0), restDistance(0.0f),
        contactManagerIndex(PX_MAX_U32), npIndex(PX_MAX_U32), solverBody0(0),
        solverBody1(0), indexType0(0), indexType1(0), nbPatches(0),
        contactManagerAge(255), nbContacts(0), contactPointSize(0),
        contactPatches(nullptr), contactPoints(nullptr),
        contactManifoldPoints(nullptr),
        outputTargetBase(0), outputTargetCount(0), expectedResponseRows(0),
        emittedResponseRows(0), outputTargetFlags(0),
        managerStateCommit(0), eligible(0), padding{0, 0} {}
};

namespace Dy {

// Complete soft/VBD tuple selected for exactly one already-gathered rigid
// island.  The provider must either return every pointer/count here or return
// false; partial soft representations are never routed into the solver.
struct AvbdSoftIslandSelection {
  AvbdSoftParticle *particles;
  PxU32 numParticles;
  AvbdSoftBody *bodies;
  PxU32 numBodies;
  AvbdSoftContact *contacts;
  PxU32 numContacts;
  PxU32 islandIndex;
  PxU32 iterationOverride;
  AvbdSoftIslandExecutionPlan executionPlan;

  AvbdSoftIslandSelection()
      : particles(nullptr), numParticles(0), bodies(nullptr), numBodies(0),
        contacts(nullptr), numContacts(0), islandIndex(PX_MAX_U32),
        iterationOverride(0) {}

  PX_FORCE_INLINE bool isComplete() const {
    return particles && numParticles > 0 && bodies && numBodies > 0 &&
           (numContacts == 0 || contacts) && islandIndex != PX_MAX_U32;
  }
};

// Scene-owned bridge into the main AVBD island solve.  It is invoked during
// serial gather, before any island task is submitted, so returned storage must
// remain stable until the Scene post-solver phase.
class AvbdSoftIslandProvider {
public:
  virtual ~AvbdSoftIslandProvider() {}

  virtual bool prepareSoftIslandSelections(
      AvbdSolverBody *solverBodies, PxsRigidBody *const *rigidBodies,
      FeatherstoneArticulation *const *articulationForBody,
      const PxU32 *linkIndexForBody,
      const PxU32 *islandBodyStarts, const PxU32 *islandBodyCounts,
      const PxU32 *activeIslandIds, PxU32 islandCount, PxReal dt,
      const PxVec3 &gravity,
      PxArray<AvbdSoftIslandSelection> &selections) = 0;
};

/**
 * @brief AVBD Dynamics Context
 *
 * Implements the dynamics pipeline using the AVBD (Augmented Variable Block
 * Descent) solver. This is an alternative to PGS and TGS solvers with different
 * convergence characteristics.
 */
class AvbdDynamicsContext : public DynamicsContextBase {
  PX_NOCOPY(AvbdDynamicsContext)

  // Friend function for persistent-manifold write-back from an island task.
  friend void writeLambdaToCache(AvbdDynamicsContext &ctx,
                                 AvbdContactConstraint *constraints,
                                 PxU32 numConstraints, PxU32 numBodies);
  friend void commitJointLambdaCacheRanges(
      AvbdDynamicsContext &ctx,
      const AvbdJointCacheCommitRange *ranges, PxU32 rangeCount);
public:
  AvbdDynamicsContext(PxcNpMemBlockPool *memBlockPool,
                      PxcScratchAllocator &scratchAllocator,
                      Cm::FlushPool &taskPool, PxvSimStats &simStats,
                      PxTaskManager *taskManager,
                      Cm::VirtualAllocatorCallback &allocator,
                      PxsMaterialManager *materialManager,
                      IG::SimpleIslandManager &islandManager, PxU64 contextID,
                      PxReal maxBiasCoefficient, PxReal lengthScale,
                      PxU32 avbdIterations,
                      PxU32 avbdJointIterationOverride,
                      bool avbdEnableEarlyStop, PxSceneFlags sceneFlags);

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

  // Commits contact-manager lifecycle tokens after validated snapshot
  // preparation and before the solve consumes the rows.
  void commitContactPrepSnapshots(AvbdContactPrepSnapshot *snapshots,
                                  PxU32 startContactIdx,
                                  PxU32 numContacts, PxU16 frameStamp);

  /** Task factory used by an island task's dispatcher fan-in stages. */
  AvbdTaskFactory &getTaskFactory() { return *mTaskFactory; }

  /** Current per-context warm-start cache epoch (zero is reserved invalid). */
  PX_FORCE_INLINE PxU16 getAvbdFrameStamp() const {
    // Use the nonzero 1..65535 range so the first frame after wrap still has
    // an unambiguous age of one relative to the wrapped frame.
    return static_cast<PxU16>(
        ((mBodyVelocityHistoryFrame - 1u) % 65535u) + 1u);
  }

  /** Scheduling policy query consumed by the Scene task-graph bridge. */
  PX_FORCE_INLINE bool isTaskGraphSerialMode() const {
    return mTaskGraphSerialMode;
  }

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
    ensureJointLambdaCacheCapacity(id + 1u);
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
    invalidateJointLambdaCacheSlot(id);
  }

  PX_FORCE_INLINE void setSoftIslandProvider(
      AvbdSoftIslandProvider *provider) {
    mSoftIslandProvider = provider;
  }

  // Optional hybrid CPU-frontend/GPU-wave backend. Null is the canonical CPU
  // path; attaching a backend never changes island ownership or writeback
  // ordering, it only replaces one prepared owner-wave chunk transactionally.
  PX_FORCE_INLINE void setRigidGpuWaveBackend(
      AvbdRigidGpuWaveBackend *backend) {
    mRigidGpuWaveBackend = backend;
  }

  PX_FORCE_INLINE AvbdRigidGpuWaveBackend *getRigidGpuWaveBackend() const {
    return mRigidGpuWaveBackend;
  }

  // CPU-owned opaque packet producer/fallback/writeback table.  The table
  // carries no GPU context or CPU object pointer in its packet ABI; the
  // caller owns the returned lifetime token and must clear any attached GPU
  // backend before destroying this context.
  PX_FORCE_INLINE bool getRigidGpuWaveCallbackTable(
      AvbdRigidGpuWaveCallbackTable &table) {
    avbdGetRigidGpuWaveCallbackTable(table, this);
    return table.isComplete();
  }

  //-------------------------------------------------------------------------
  // Persistent contact manifold state (Public for preparation/writeback)
  //-------------------------------------------------------------------------

  /**
   * @brief One compact persistent point owned by a ContactManager
   *
   * Detection anchors are refreshed every frame and used for deterministic
   * one-to-one geometric matching. Constraint anchors and their normal offset
   * are a single tuple: sticking points retain that tuple, while sliding
   * points adopt the fresh narrow-phase tuple. No hashed point lookup is used.
   */
  struct CachedContactManifoldPoint {
    PxVec3 detectionPointA;
    PxVec3 detectionPointB;
    PxVec3 constraintPointA;
    PxVec3 constraintPointB;
    PxVec3 normal;
    PxVec3 tangent0;
    PxReal normalOffset;   //!< Offset paired with the constraint anchors
    PxReal lambda;         //!< Normal constraint lambda
    PxReal tangentLambda0; //!< Friction lambda 1
    PxReal tangentLambda1; //!< Friction lambda 2
    PxReal penalty; //!< Adaptive penalty for normal (persists across frames)
    PxReal tangentPenalty0; //!< Adaptive penalty along transported tangent 0
    PxReal tangentPenalty1; //!< Adaptive penalty along transported tangent 1
    PxU32 materialKey;
    PxU16 frameStamp;
    PxU8 stick;             //!< Coulomb stick flag (static μ next frame)
    PxU8 padding[1];
  };

  PxArray<CachedContactManifoldPoint>
      mContactManifoldPoints; //!< Fixed compact ranges indexed by CM
  bool mEnableLambdaWarmStart; //!< Enable lambda warm-starting (default: true)

  struct CachedContactManagerState {
    PxU64 identity; //!< Exact manager identity for slot validation
    PxU16 frameStamp;
    PxU8 padding[6];
  };

  PxArray<CachedContactManagerState>
      mContactManagerStateCache; //!< One persistent state per CM index

  struct CachedJointLambda {
    PxU64 identity; //!< Exact source identity validating the direct slot
    PxVec3 lambdaLinear;
    PxVec3 lambdaAngular;
    PxVec3 lambdaDriveLinear;
    PxVec3 lambdaDriveAngular;
    PxReal coneLambda;
    PxU16 frameStamp;
    PxU8 padding[2];
  };

  PxArray<CachedJointLambda>
      mJointLambdaCache; //!< Direct sidecar indexed by Constraint::index

  static const PxU32 CONTACT_MANIFOLD_POINT_CAPACITY =
      8; //!< Matches the reference AVBD manifold bound
  static const PxU8 LAMBDA_MAX_AGE = 3; //!< Max frames to keep cached lambda
  static constexpr PxReal LAMBDA_WARMSTART_SCALE =
      0.9f; //!< Damping for warm-started lambda

private:
  PX_FORCE_INLINE PxU16 getConstraintConcreteType(PxU32 id) const {
    return id < mConstraintConcreteTypes.size() ? mConstraintConcreteTypes[id]
                                                : PxU16(0);
  }

  struct BodyVelocityHistoryHeader {
    PxU64 frame;
    PxU64 generation;
    PxU32 offset;
    PxU32 count;
  };

  // Direct node-indexed headers validate the node lifetime generation and
  // address a compact, contiguous sample range. The two buffers alternate:
  // serial gather only reads the previous buffer and appends to the current
  // one, which is frozen before island tasks are submitted.
  PxArray<BodyVelocityHistoryHeader> mBodyVelocityHistoryHeaders[2];
  PxArray<PxVec3> mBodyVelocityHistorySamples[2];
  PxU64 mBodyVelocityHistoryFrame;
  PxU32 mBodyVelocityHistoryWriteBuffer;
  PxArray<PxU16> mConstraintConcreteTypes;

  //-------------------------------------------------------------------------
  // Internal Methods
  //-------------------------------------------------------------------------

  void beginBodyVelocityHistoryFrame(PxU32 nodeCapacity,
                                     PxU32 sampleCapacity);
  void restoreAndUpdateBodyVelocityHistory(
      const PxsBodyCore &bodyCore, AvbdSolverBody &solverBody,
      PxNodeIndex nodeIndex, PxU32 sampleIndex, PxU32 sampleCount);
  void ensureJointLambdaCacheCapacity(PxU32 requiredCapacity);
  void invalidateJointLambdaCacheSlot(PxU32 index);

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
                   const PxNodeIndex *bodyNodeIndices, PxU32 numBodies,
                   PxReal dt, const PxVec3 &gravity);

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
                            PxU32 numContactsToProcess, PxU32 bodyOffset,
                            AvbdContactPrepSnapshot *snapshots = nullptr,
                            AvbdContactOutputToken *outputTokens = nullptr,
                            PxU32 outputTokenBase = 0);

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
  // AVBD context storage. Keep runtime knobs in AvbdSolverConfig
  // (DyAvbdTypes.h); profiling state must remain task-local/cold.
  AvbdSolver mSolver; //!< AVBD solver instance
  PxcScratchAllocator &mScratchAllocator; //!< Scratch memory allocator
  AvbdSoftIslandProvider
      *mSoftIslandProvider; //!< Scene-owned complete soft tuple provider
  AvbdRigidGpuWaveBackend
      *mRigidGpuWaveBackend; //!< Optional transactional owner-wave backend
  PxU32 mAvbdIterations; //!< Scene-wide complete AVBD iteration budget
  PxU32 mAvbdJointIterationOverride; //!< Optional joint-island minimum budget
  bool mAvbdEnableEarlyStop; //!< Allow pose-delta convergence termination

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
  // P2 only supports the Scene PxTaskManager path. `true` is the explicit
  // serial reference mode selected by PHYSX_AVBD_TASKGRAPH_SERIAL.
  bool mTaskGraphSerialMode;
  // Task-graph and iteration diagnostics are intentionally kept out of the
  // solver context. Profile data is local to tasks and reduced after join.
  bool mFrictionEveryIteration; //!< Apply friction every iteration
  bool mSolverInitialized;      //!< Whether solver has been initialized

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

void writeContactImpulseToOutputTokens(
    const AvbdContactConstraint *constraints,
    const AvbdContactOutputToken *tokens,
    AvbdContactOutputResult *results, PxU32 numConstraints, PxU32 numBodies,
    PxReal dt);

// Context-free snapshot parser. The caller owns all input/output ranges and
// supplies the immutable configuration captured by the parent update.
PxU32 prepareAvbdContactSnapshots(
    PxReal dt, AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
    AvbdContactConstraint *constraints, PxU32 maxConstraints,
    PxU32 startContactIdx, PxU32 numContactsToProcess, PxU32 bodyOffset,
    AvbdContactPrepSnapshot *snapshots, AvbdContactOutputToken *outputTokens,
    PxU32 outputTokenBase, PxReal lengthScale, bool enableLambdaWarmStart,
    PxU16 frameStamp);

void commitContactOutputTokens(const AvbdContactOutputTarget *targets,
                               AvbdContactOutputResult *results,
                               PxU32 count);

void restoreJointLambdaFromCache(AvbdDynamicsContext &ctx,
                                 AvbdD6JointConstraint &constraint,
                                 PxU32 cacheIndex, PxU64 cacheIdentity,
                                 PxU64 objectiveKey);

void commitJointLambdaCacheRanges(
    AvbdDynamicsContext &ctx,
    const AvbdJointCacheCommitRange *ranges, PxU32 rangeCount);

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
    PxU32 avbdIterations, PxU32 avbdJointIterationOverride,
    bool avbdEnableEarlyStop, PxSceneFlags sceneFlags);

} // namespace Dy

} // namespace physx

#endif // DY_AVBD_DYNAMICS_H
