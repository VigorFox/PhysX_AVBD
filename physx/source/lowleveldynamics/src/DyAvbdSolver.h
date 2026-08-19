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

#ifndef DY_AVBD_SOLVER_H
#define DY_AVBD_SOLVER_H

#include "DyAvbdConstraint.h"
#include "DyAvbdParallel.h"
#include "DyAvbdSoftBody.h"
#include "DyAvbdSolverBody.h"
#include "DyAvbdTypes.h"

#pragma warning(push)
#pragma warning(disable : 4324)

namespace physx {

namespace Dy {

class FeatherstoneArticulation;
class AvbdRigidBodyRangeTask;
class AvbdRigidDualRangeTask;
// Test-owned accessor for the standalone AVBD kernel lab.  The friend does
// not change the solver ABI; it lets the lab invoke the existing contact-only
// scalar authority without copying its local 6x6 assembly mathematics.
class AvbdKernelLabSolverAccess;

struct AvbdOgcPairTrustRegionContext {
  AvbdOgcPairState *pairStates;
  physx::PxU32 numPairStates;
  const physx::PxU32 *contactPairIndices;
  physx::PxU32 numContactPairIndices;
  // Separate particle-to-triangle-core incidence for geometric admission.
  // These refs never enter the AL force/Hessian program: they let a particle
  // candidate preserve the complete expanded collision triangle carried by a
  // TBIX row, rather than only that row's compact centroid objective.
  const physx::PxU32 *triangleCoreSafetyStarts;
  physx::PxU32 numTriangleCoreSafetyStarts;
  const AvbdSoftContactParticleRef *triangleCoreSafetyRefs;
  physx::PxU32 numTriangleCoreSafetyRefs;

  AvbdOgcPairTrustRegionContext()
      : pairStates(nullptr), numPairStates(0), contactPairIndices(nullptr),
        numContactPairIndices(0), triangleCoreSafetyStarts(nullptr),
        numTriangleCoreSafetyStarts(0), triangleCoreSafetyRefs(nullptr),
        numTriangleCoreSafetyRefs(0) {}

  PX_FORCE_INLINE bool isComplete(physx::PxU32 numContacts) const {
    return pairStates && numPairStates > 0 && contactPairIndices &&
        numContactPairIndices == numContacts;
  }

  PX_FORCE_INLINE bool hasTriangleCoreSafetyPlan(
      physx::PxU32 numParticles) const {
    return triangleCoreSafetyStarts &&
        numTriangleCoreSafetyStarts == numParticles + 1 &&
        (numTriangleCoreSafetyRefs == 0 || triangleCoreSafetyRefs) &&
        triangleCoreSafetyStarts[0] == 0 &&
        triangleCoreSafetyStarts[numParticles] == numTriangleCoreSafetyRefs;
  }
};

// Provider-owned, immutable-for-one-island-solve support program.  It keeps
// the native mixed path from re-discovering soft ownership and contact
// incidence after Scene has already compiled the same information while
// preparing the island.  The arrays are deliberately non-owning: their
// lifetime is the provider's selection storage, which is required to outlive
// the solver task by the soft-island provider contract.
struct AvbdSoftIslandExecutionPlan {
  const physx::PxU32 *particleBodyIndices;
  physx::PxU32 numParticleBodyIndices;
  const physx::PxU32 *contactStarts;
  physx::PxU32 numContactStarts;
  const AvbdSoftContactParticleRef *contactRefs;
  physx::PxU32 numContactRefs;

  // Geometry-only companion CSR for triangle/OBB core rows.  It intentionally
  // does not change contact-force ownership; it is consulted only by the OGC
  // candidate trust-region filter before a particle position is committed.
  const physx::PxU32 *triangleCoreSafetyStarts;
  physx::PxU32 numTriangleCoreSafetyStarts;
  const AvbdSoftContactParticleRef *triangleCoreSafetyRefs;
  physx::PxU32 numTriangleCoreSafetyRefs;

  // Mirror the particle-contact CSR for the rigid endpoint of a dynamic
  // soft/rigid contact.  Each range preserves source contact order and lets
  // the 6x6 rigid block consume the same immutable contact-ownership program
  // instead of scanning every soft row once per body and iteration.
  const physx::PxU32 *rigidTargetContactStarts;
  physx::PxU32 numRigidTargetContactStarts;
  const physx::PxU32 *rigidTargetContactRefs;
  physx::PxU32 numRigidTargetContactRefs;

  // Immutable collision-domain view for the optional terminal OGC epoch.
  // Scene owns these proxy bodies and embeddings for the entire native island
  // task.  The solver rebuilds only their positions from the final simulation
  // pose, then performs current-pose box DCD into private scratch; this is
  // deliberately not an AL-state or swept/CCD contact stream.
  const AvbdRigidBox *terminalRigidBoxes;
  physx::PxU32 numTerminalRigidBoxes;
  const AvbdSoftBody *terminalCollisionBodies;
  physx::PxU32 numTerminalCollisionBodies;
  const AvbdWeightedContactPoint *terminalCollisionVertexMappings;
  physx::PxU32 numTerminalCollisionVertexMappings;
  physx::PxReal terminalContactRadius;

  // Shared mutable OGC pair state.  This is intentionally separate from the
  // persistent AL contact state: it governs geometric trust-region epochs,
  // while the contact stream retains multiplier/penalty history.
  AvbdOgcPairState *ogcPairStates;
  physx::PxU32 numOgcPairStates;
  const physx::PxU32 *ogcPairIndices;
  physx::PxU32 numOgcPairIndices;

  // Pair-major view of the same dynamic soft/rigid contact stream.  The
  // contact-to-pair map above is convenient for a particle update; this CSR
  // is the inverse schedule for a pair-owned manifold block.  Keeping both
  // views immutable avoids per-sweep all-contact scans once a pair is active.
  const physx::PxU32 *ogcPairContactStarts;
  physx::PxU32 numOgcPairContactStarts;
  const physx::PxU32 *ogcPairContactRefs;
  physx::PxU32 numOgcPairContactRefs;

  // The Scene provider may need the predicted soft pose to compile swept
  // contact objectives before the native island task is submitted.  When it
  // has done so, carry that lifecycle fact with the same immutable support
  // program instead of making the mixed solver repeat a full particle pass.
  // This is a per-solve Scene-provider token: it is never mutated by a
  // worker, and the solver consumes it only after semantic plan validation.
  bool softPredictionPrepared;

  AvbdSoftIslandExecutionPlan()
      : particleBodyIndices(nullptr), numParticleBodyIndices(0),
        contactStarts(nullptr), numContactStarts(0), contactRefs(nullptr),
        numContactRefs(0), triangleCoreSafetyStarts(nullptr),
        numTriangleCoreSafetyStarts(0), triangleCoreSafetyRefs(nullptr),
        numTriangleCoreSafetyRefs(0), rigidTargetContactStarts(nullptr),
        numRigidTargetContactStarts(0), rigidTargetContactRefs(nullptr),
        numRigidTargetContactRefs(0), terminalRigidBoxes(nullptr),
        numTerminalRigidBoxes(0), terminalCollisionBodies(nullptr),
        numTerminalCollisionBodies(0),
        terminalCollisionVertexMappings(nullptr),
        numTerminalCollisionVertexMappings(0), terminalContactRadius(0.0f),
        ogcPairStates(nullptr), numOgcPairStates(0), ogcPairIndices(nullptr),
        numOgcPairIndices(0), ogcPairContactStarts(nullptr),
        numOgcPairContactStarts(0), ogcPairContactRefs(nullptr),
        numOgcPairContactRefs(0),
        softPredictionPrepared(false) {}

  PX_FORCE_INLINE bool isComplete(physx::PxU32 numParticles) const {
    return particleBodyIndices && numParticleBodyIndices == numParticles &&
           contactStarts && numContactStarts == numParticles + 1 &&
           (numContactRefs == 0 || contactRefs) &&
           contactStarts[0] == 0 &&
           contactStarts[numParticles] == numContactRefs;
  }

  PX_FORCE_INLINE bool hasRigidTargetContactPlan(
      physx::PxU32 numRigidBodies) const {
    return numRigidBodies > 0 && rigidTargetContactStarts &&
           numRigidTargetContactStarts == numRigidBodies + 1 &&
           (numRigidTargetContactRefs == 0 || rigidTargetContactRefs) &&
           rigidTargetContactStarts[0] == 0 &&
           rigidTargetContactStarts[numRigidBodies] ==
               numRigidTargetContactRefs;
  }

  PX_FORCE_INLINE bool hasTriangleCoreSafetyPlan(
      physx::PxU32 numParticles) const {
    return triangleCoreSafetyStarts &&
           numTriangleCoreSafetyStarts == numParticles + 1 &&
           (numTriangleCoreSafetyRefs == 0 || triangleCoreSafetyRefs) &&
           triangleCoreSafetyStarts[0] == 0 &&
           triangleCoreSafetyStarts[numParticles] ==
               numTriangleCoreSafetyRefs;
  }

  PX_FORCE_INLINE bool hasTerminalCurrentPoseBoxPlan(
      physx::PxU32 numSimulationParticles) const {
    return terminalRigidBoxes && numTerminalRigidBoxes > 0 &&
           terminalCollisionBodies && numTerminalCollisionBodies > 0 &&
           terminalCollisionVertexMappings &&
           numTerminalCollisionVertexMappings > 0 &&
           physx::PxIsFinite(terminalContactRadius) &&
           terminalContactRadius > 0.0f &&
           numSimulationParticles > 0;
  }

  PX_FORCE_INLINE bool hasMixedOgcPairPlan(physx::PxU32 numContacts) const {
    return ogcPairStates && numOgcPairStates > 0 && ogcPairIndices &&
           numOgcPairIndices == numContacts && numContacts > 0;
  }

  PX_FORCE_INLINE bool hasMixedOgcPairContactPlan(
      physx::PxU32 numContacts) const {
    return hasMixedOgcPairPlan(numContacts) && ogcPairContactStarts &&
           numOgcPairContactStarts == numOgcPairStates + 1 &&
           (numOgcPairContactRefs == 0 || ogcPairContactRefs) &&
           ogcPairContactStarts[0] == 0 &&
           ogcPairContactStarts[numOgcPairStates] ==
               numOgcPairContactRefs;
  }
};

// Persistent state for the rigid position/dual iteration loop.  The state is
// deliberately limited to iteration control and transient pose snapshots;
// contact/body ownership remains in the caller-owned island batch.  Keeping
// this seam explicit allows a future PhysX-dispatcher fan-in to suspend after
// one exact Gauss--Seidel dependency wave without adding solver-path atomics.
struct AvbdRigidSolveIterationState {
  AvbdSolverBody *bodies;
  physx::PxU32 numBodies;
  AvbdContactConstraint *contacts;
  physx::PxU32 numContacts;
  physx::PxReal dt;
  const AvbdBodyConstraintMap *contactMap;
  AvbdColorBatch *colorBatches;
  physx::PxU32 numColors;
  bool hasBodyStaticContact;
  const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart;
  AvbdSolverStats *stats;

  bool useChebyshev;
  bool enableEarlyStop;
  physx::PxReal chebyOmega;
  physx::PxU32 iters;
  physx::PxU32 minIterations;
  physx::PxReal rotationTolerance;
  physx::PxU32 consecutiveConvergedIterations;
  physx::PxU32 iter;
  physx::PxU32 activeIteration;
  bool iterationActive;
  // Set only by the fast CPU task path after every disjoint contact range has
  // completed. The ordered authority always leaves this false and executes
  // the established scalar dual pass inside completeRigidSolveIteration().
  bool parallelDualComplete;

  physx::PxArray<physx::PxVec3> chebyPrevPos;
  physx::PxArray<physx::PxVec3> chebyPrevPrevPos;
  physx::PxArray<physx::PxQuat> chebyPrevRot;
  physx::PxArray<physx::PxQuat> chebyPrevPrevRot;
  physx::PxArray<physx::PxVec3> earlyStopPrevPos;
  physx::PxArray<physx::PxQuat> earlyStopPrevRot;

  AvbdRigidSolveIterationState()
      : bodies(nullptr), numBodies(0), contacts(nullptr), numContacts(0),
        dt(0.0f), contactMap(nullptr), colorBatches(nullptr), numColors(0),
        hasBodyStaticContact(false), linearVelAtSolveStart(nullptr),
        stats(nullptr), useChebyshev(false), enableEarlyStop(false),
        chebyOmega(1.0f), iters(0), minIterations(0),
        rotationTolerance(0.0f), consecutiveConvergedIterations(0), iter(0),
        activeIteration(0), iterationActive(false),
        parallelDualComplete(false) {}
};

// A fast deferred rigid island can publish exactly which post-AL contact
// consumers may have work after final objective validation.  Unknown is the
// fail-closed default used by ordered, synchronous, joint, and soft paths;
// zero is a valid, proven-empty work set.
struct AvbdPostAlContactWorkPlan {
  enum : physx::PxU8 {
    eUNKNOWN = 0xff,
    ePASSIVE_COMPONENT = 1u << 0,
    eCOMPLETE_MANIFOLD = 1u << 1,
    ePOINT_TARGET = 1u << 2,
  };

  physx::PxU8 mask;

  AvbdPostAlContactWorkPlan() : mask(eUNKNOWN) {}

  void reset() { mask = eUNKNOWN; }
  void publish(physx::PxU8 knownMask) { mask = knownMask; }
  bool isKnown() const { return mask != eUNKNOWN; }
  bool mayHave(physx::PxU8 work) const {
    return !isKnown() || (mask & work) != 0;
  }
};

// Contact-only rigid solve lifetime shared by the synchronous entry and the
// dispatcher-driven wave task path.  All arrays are island-owned state; no
// solver-global mutable scratch is used while islands overlap.
struct AvbdRigidSolveContext {
  AvbdRigidSolveIterationState iteration;
  physx::PxReal invDt;
  physx::PxReal invDt2;
  physx::PxVec3 gravity;
  bool hasBodyStaticContact;
  bool deformableFastImpactIsland;
  AvbdPostAlContactWorkPlan postAlContactWork;
  physx::PxArray<bool> touchingBodyStatic;
  physx::PxArray<physx::PxVec3> linearVelAtSolveStart;
  physx::PxArray<physx::PxVec3> angularVelAtSolveStart;

  physx::PxArray<physx::PxU32> dependencyWaveOffsets;
  physx::PxArray<physx::PxU32> dependencyWaveBodies;
  physx::PxU32 dependencyWaveCount;

  // CPU-native non-deterministic schedule.  Colors are compact island-local
  // independent sets: every writable dynamic body appears exactly once and
  // no dynamic--dynamic contact has both endpoints in the same color.
  physx::PxArray<physx::PxU32> bodyColorOffsets;
  physx::PxArray<physx::PxU32> bodyColorBodies;
  physx::PxU32 bodyColorCount;
  physx::PxU32 maxBodyColorWidth;

  AvbdRigidSolveContext()
      : invDt(0.0f), invDt2(0.0f), gravity(0.0f),
        hasBodyStaticContact(false),
        deformableFastImpactIsland(false),
        dependencyWaveCount(0),
        bodyColorCount(0), maxBodyColorWidth(0) {}
};

/**
 * @brief Main AVBD Solver class implementing the Block Coordinate Descent
 * algorithm
 *
 * The AVBD solver operates on position-level variables and uses:
 * 1. Prediction integration (explicit Euler)
 * 2. Block descent solve for each body's local 6x6 system
 * 3. Augmented Lagrangian multiplier updates for constraint satisfaction
 */
class AvbdSolver {
public:
  AvbdSolver();
  ~AvbdSolver();

  //-------------------------------------------------------------------------
  // Initialization
  //-------------------------------------------------------------------------

  /**
   * @brief Initialize solver with configuration
   */
  void initialize(const AvbdSolverConfig &config,
                  physx::PxAllocatorCallback &allocator);

  /**
   * @brief Release all allocated resources
   */
  void release();

  //-------------------------------------------------------------------------
  // Solver Main Loop
  //-------------------------------------------------------------------------

  /**
   * @brief Execute one simulation step (contacts only)
   * @param dt Time step
   * @param bodies Array of solver bodies
   * @param numBodies Number of bodies
   * @param contacts Array of contact constraints
   * @param numContacts Number of contacts
   * @param gravity Gravity vector
   * @param colorBatches Pre-computed color batches (nullptr for no coloring)
   * @param numColors Number of colors in colorBatches (0 if not colored)
   */
  void solve(physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
             AvbdContactConstraint *contacts, physx::PxU32 numContacts,
             const physx::PxVec3 &gravity,
             const AvbdBodyConstraintMap *contactMap,
             AvbdColorBatch *colorBatches,
             physx::PxU32 numColors,
             physx::PxU32 iterationOverride,
             AvbdSolverStats &stats);

  /** Prepare the contact-only rigid path up to the first body iteration. */
  bool prepareRigidSolve(physx::PxReal dt, AvbdSolverBody *bodies,
                         physx::PxU32 numBodies,
                         AvbdContactConstraint *contacts,
                         physx::PxU32 numContacts,
                         const physx::PxVec3 &gravity,
                         const AvbdBodyConstraintMap *contactMap,
                         AvbdColorBatch *colorBatches,
                         physx::PxU32 numColors,
                         physx::PxU32 iterationOverride,
                         AvbdSolverStats &stats,
                         AvbdRigidSolveContext &context);

  /** Run the shared post-iteration stages after a prepared rigid solve. */
  void finishRigidSolve(AvbdRigidSolveContext &context);

  /** Build exact order-preserving dependency waves for a prepared island. */
  void buildRigidDependencyWaves(AvbdRigidSolveContext &context);

  /** Build a compact strict body-color plan for the fast CPU backend. */
  bool buildRigidBodyColorPlan(AvbdRigidSolveContext &context);

  /** Begin one prepared iteration before body-range tasks are submitted. */
  bool beginRigidSolveIteration(AvbdRigidSolveIterationState &state);

  /** Complete one prepared iteration after all body ranges have finished. */
  bool completeRigidSolveIteration(AvbdRigidSolveIterationState &state);

  /**
   * @brief Execute one simulation step with joint constraints (unified D6 + gear)
   * @param dt Time step
   * @param bodies Array of solver bodies
   * @param numBodies Number of bodies
   * @param contacts Array of contact constraints
   * @param numContacts Number of contacts
   * @param d6Joints Array of D6 joint constraints (all joint types unified)
   * @param numD6 Number of D6 joints
   * @param gearJoints Array of gear joint constraints
   * @param numGear Number of gear joints
   * @param gravity Gravity vector
   * @param contactMap Pre-computed contact-to-body mapping (optional)
   * @param d6Map Pre-computed D6 joint mapping (optional)
   * @param gearMap Pre-computed gear joint mapping (optional)
   * @param colorBatches Pre-computed color batches (nullptr for no coloring)
   * @param numColors Number of colors in colorBatches (0 if not colored)
   */
  void solveWithJoints(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts,
                       AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
                       AvbdGearJointConstraint *gearJoints,
                       physx::PxU32 numGear, const physx::PxVec3 &gravity,
                       const AvbdBodyConstraintMap *contactMap,
                       const AvbdBodyConstraintMap *d6Map,
                       const AvbdBodyConstraintMap *gearMap,
                       AvbdColorBatch *colorBatches,
                       physx::PxU32 numColors,
                       physx::PxU32 iterationOverride,
                       // Soft body VBD parameters
                       AvbdSoftParticle *softParticles,
                       physx::PxU32 numSoftParticles,
                       AvbdSoftBody *softBodies,
                       physx::PxU32 numSoftBodies,
                       AvbdSoftContact *softContacts,
                       physx::PxU32 numSoftContacts,
                       const AvbdSoftIslandExecutionPlan *softExecutionPlan,
                       FeatherstoneArticulation *const *articulationForBody,
                       const physx::PxU32 *linkIndexForBody,
                       AvbdSolverStats &stats);

  /**
   * @brief Single island solve entry.
   *
   * Classifies the batch and runs the shared stage list. Joint rows and a
   * complete genuine soft/VBD tuple use the joint/VBD module; contact-only
   * rows remain on the rigid contact path. Soft data is never synthesized
   * from NP contacts here.
   */
  void solveIsland(physx::PxReal dt, AvbdSolverBody *bodies,
                   physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                   physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                   AvbdD6JointConstraint *d6Joints,
                   physx::PxU32 numD6,
                   AvbdGearJointConstraint *gearJoints,
                   physx::PxU32 numGear,
                   const AvbdBodyConstraintMap *contactMap,
                   const AvbdBodyConstraintMap *d6Map,
                   const AvbdBodyConstraintMap *gearMap,
                   AvbdColorBatch *colorBatches,
                   physx::PxU32 numColors,
                   physx::PxU32 iterationOverride,
                   AvbdSoftParticle *softParticles,
                   physx::PxU32 numSoftParticles,
                   AvbdSoftBody *softBodies,
                   physx::PxU32 numSoftBodies,
                   AvbdSoftContact *softContacts,
                   physx::PxU32 numSoftContacts,
                   const AvbdSoftIslandExecutionPlan *softExecutionPlan,
                   FeatherstoneArticulation *const *articulationForBody,
                   const physx::PxU32 *linkIndexForBody,
                   AvbdSolverStats &stats,
                   AvbdRigidSolveContext *deferredRigidContext = nullptr);

  /**
   * @brief Get solver configuration
   */
  const AvbdSolverConfig &getConfig() const { return mConfig; }

  /** Mutable config (scene bounce threshold, material gates). */
  AvbdSolverConfig &getConfigMutable() { return mConfig; }

  /** Execute one rejected owner lane through the authoritative scalar path. */
  bool solveRigidOwnerFallback(AvbdRigidSolveContext &context,
                               const physx::PxU32 *ownerBodyOrder,
                               physx::PxU32 lane);

private:
  friend class AvbdRigidBodyRangeTask;
  friend class AvbdRigidDualRangeTask;
  friend class AvbdSolveIslandTask;
  friend class AvbdKernelLabSolverAccess;

  //-------------------------------------------------------------------------
  // Algorithm Stages
  //-------------------------------------------------------------------------

  /**
   * @brief Stage 1: Compute predicted positions using explicit Euler
   * x_tilde = x_n + h*v + h^2*f_ext/m
   */
  void computePrediction(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                         physx::PxReal dt, const physx::PxVec3 &gravity);

  /**
   * @brief Stage 3: Block coordinate descent iteration
   * For each color group (in parallel):
   *   For each body in group:
   *     Solve local 6x6 system to minimize energy
   * @param colorBatches Pre-computed color batches (nullptr for sequential)
   * @param numColors Number of colors (0 for sequential processing)
   */
  void blockDescentIteration(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                             AvbdContactConstraint *contacts,
                             physx::PxU32 numContacts, physx::PxReal dt,
                             const AvbdBodyConstraintMap *contactMap = nullptr,
                             AvbdColorBatch *colorBatches = nullptr,
                             physx::PxU32 numColors = 0);

  /** Solve a contiguous body-index range using the live Gauss--Seidel pose. */
  void solveRigidBodyRange(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      physx::PxReal dt, physx::PxReal invDt2,
      const AvbdBodyConstraintMap *contactMap, const physx::PxU32 *bodyOrder,
      physx::PxU32 begin, physx::PxU32 end);

  /** Update one disjoint contact range for the fast CPU dual fan-out. */
  void solveRigidDualRange(AvbdRigidSolveIterationState &state,
                           physx::PxU32 begin, physx::PxU32 end);

  /** Execute one or more exact serial AVBD iterations from resumable state. */
  bool advanceRigidSolveIterations(AvbdRigidSolveIterationState &state);

  /**
   * @brief Shared contact primal rows for one body (Phase 1).
   *
   * Body-static contract (both contact and joint local solvers):
   *   - dominant body-static normal only when contact count > 4
   *   - finalizeBodyVsStaticViolation for deformable anchors
   *   - no multi-contact body-static tangents in aggregated 6x6
   *   - static-particle soft rows only from a complete genuine soft/VBD tuple
   */
  void accumulateBodyContactRows(
      AvbdSolverBody &body, physx::PxU32 bodyIndex, AvbdSolverBody *bodies,
      physx::PxU32 numBodies, AvbdContactConstraint *contacts,
      physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxReal dt, physx::PxReal massInvDt2, AvbdBlock6x6 &A,
      physx::PxVec3 &gLinear, physx::PxVec3 &gAngular,
      physx::PxU32 &numTouching,
      const physx::PxU32 *rigidTargetContactStarts = nullptr,
      const physx::PxU32 *rigidTargetContactRefs = nullptr);

  /**
   * @brief Solve local 6x6 system for a single body
   * Minimizes: 1/(2h^2) * ||M(x - x_tilde)||^2 + Sum constraint_energy
   */
  void solveLocalSystem(AvbdSolverBody &body, AvbdSolverBody *bodies,
                        physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                        physx::PxU32 numContacts, physx::PxReal dt,
                        physx::PxReal invDt2,
                        const AvbdBodyConstraintMap *contactMap = nullptr);

  /**
   * @brief Solve local 6x6 system for a single body with BOTH contacts AND
   * joints
   *
   * True AVBD: accumulates both contact and joint Jacobians into the same
   * Hessian matrix H = M/h^2 + sum(rho_c * Jc^T * Jc) + sum(rho_j * Jj^T * Jj)
   * and gradient g, then solves the 6x6 system in one shot.
   *
   * For joints: Jacobian per constraint row is computed and accumulated
   * the same way as contacts -- pen * J^T * J into LHS, f * J into RHS.
   */
  void solveLocalSystemWithJoints(
      AvbdSolverBody &body, AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
      physx::PxReal dt, physx::PxReal invDt2,
      const AvbdBodyConstraintMap *contactMap = nullptr,
      const AvbdBodyConstraintMap *d6Map = nullptr,
      const AvbdBodyConstraintMap *gearMap = nullptr,
      AvbdSoftParticle *softParticles = nullptr,
      physx::PxU32 numSoftParticles = 0,
      AvbdSoftContact *softContacts = nullptr,
      physx::PxU32 numSoftContacts = 0,
      AvbdSoftBody *softBodies = nullptr,
      physx::PxU32 numSoftBodies = 0,
      const physx::PxU32 *rigidTargetContactStarts = nullptr,
      const physx::PxU32 *rigidTargetContactRefs = nullptr,
      const AvbdOgcPairTrustRegionContext *ogcPairContext = nullptr);

  /**
   * @brief Solve decoupled 3x3 system for a single body
   * Block-diagonal approximation of the 6x6 system:
   *   Same accumulation as solveLocalSystemWithJoints (contacts + joints
   *   into per-body LHS/RHS), but solves two independent 3x3 systems
   *   (Alin, Aang) instead of one coupled 6x6 LDLT.
   *
   * KNOWN LIMITATION: dropping the off-diagonal B block loses the
   * linear-angular coupling from joints with offset anchors. For dense
   * mesh + impact scenarios (e.g. chainmail), this makes contact
   * response ~42x weaker. Joint chains work fine.
   */
  void solveLocalSystem3x3(AvbdSolverBody &body, AvbdSolverBody *bodies,
                           physx::PxU32 numBodies,
                           AvbdContactConstraint *contacts,
                           physx::PxU32 numContacts,
                           AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
                           physx::PxReal dt, physx::PxReal invDt2,
                           const AvbdBodyConstraintMap *contactMap = nullptr,
                           const AvbdBodyConstraintMap *d6Map = nullptr);

  /**
   * @brief Stage 4: Update Augmented Lagrangian multipliers with XPBD
   * compliance Uses XPBD formula: dLambda = (-C - alphaTilde*lambda) / (w + alphaTilde) where alphaTilde = alpha/dt2
   */
  void updateLagrangianMultipliers(AvbdSolverBody *bodies,
                                   physx::PxU32 numBodies,
                                   AvbdContactConstraint *contacts,
                                   physx::PxU32 numContacts, physx::PxReal dt,
                                   AvbdSolverStats &stats);

  /**
   * @brief Sequential (Gauss-Seidel) velocity-level friction for body-vs-static
   *        contacts (rigid plane and deformable-mesh anchors).
   *
   * This is the fallback for body-static tangents that are intentionally not
   * owned by the position-level local system. Rigid-static rows and unsupported
   * deformable rows use the decoupled projected pass; position-owned deformable
   * tangents are excluded.
   */
  void applyBodyStaticFrictionSweeps(AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     AvbdContactConstraint *contacts,
                                     physx::PxU32 numContacts,
                                     const physx::PxVec3 &gravity,
                                     physx::PxReal dt,
                                     physx::PxU32 sweeps,
                                     const physx::PxArray<physx::PxVec3> *velSeedPos,
                                     const physx::PxArray<physx::PxQuat> *velSeedRot,
                                     const physx::PxArray<bool> *skipForBodies,
                                     AvbdSolverStats *stats);

  /** Post-pass friction for static-particle AvbdSoftContact rows. */
  void applyKinematicShellFrictionSweeps(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxReal dt, physx::PxU32 sweeps,
      const physx::PxArray<physx::PxVec3> *velSeedPos,
      const physx::PxArray<physx::PxQuat> *velSeedRot,
      AvbdSolverStats *stats);

  /** TGS-style capped normal projection on static-particle soft rows. */
  void applyKinematicShellNormalDepenetrationSweeps(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
      AvbdSolverStats *stats);

  /**
   * @brief Conservative current-pose recovery for genuine soft/world-static
   * SDF or ground overlap.
   *
   * This is deliberately a geometric safety stage, not another owner of the
   * OGC margin.  It accepts only a negative true collision-surface gap from a
   * non-CCD source body, applies a common-alpha support correction guarded by
   * exact incident-tet determinants, and moves the velocity anchor together
   * with position.
   */
  void applyWorldStaticSoftNormalDepenetrationSweeps(
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxU32 sweeps,
      physx::PxArray<physx::PxU8> *recoveredContacts,
      AvbdSolverStats *stats,
      const AvbdSoftIslandExecutionPlan *ogcExecutionPlan = nullptr,
      AvbdSolverBody *ogcRigidBodies = nullptr,
      physx::PxU32 numOgcRigidBodies = 0,
      const AvbdSoftContact *ogcContacts = nullptr,
      physx::PxU32 numOgcContacts = 0);

  // Fresh current-pose triangle/OBB cores use the same local support
  // scheduling as dynamic pairs, but have no movable rigid endpoint. This
  // intentionally deforms only the soft collision support; it never performs
  // a coherent whole-volume escape from a plane or world-static box.
  void applyWorldStaticTriangleCoreLocalManifold(
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxU32 sweeps, AvbdSolverStats *stats,
      const AvbdSoftIslandExecutionPlan *ogcExecutionPlan = nullptr,
      AvbdSolverBody *ogcRigidBodies = nullptr,
      physx::PxU32 numOgcRigidBodies = 0,
      const AvbdSoftContact *ogcContacts = nullptr,
      physx::PxU32 numOgcContacts = 0);

  /**
   * @brief Coherent endpoint fallback for a deep non-CCD soft/world-static
   *        overlap.
   *
   * A current-pose OGC row which first appears deeply inside a static target
   * may be too late for independent support rows: correcting only those
   * vertices can invert their incident tetrahedra before the material solve.
   * This narrow fallback translates an otherwise unconstrained source body as
   * one rigidly coherent soft configuration, preserving every tet F exactly.
   */
  void applyWorldStaticSoftBodyEndpointTranslations(
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxArray<physx::PxU8> *recoveredBodies,
      physx::PxArray<physx::PxVec3> *recoveryNormals,
      physx::PxArray<physx::PxVec3> *recoveryTranslations,
      bool allowFreshTriangleCoreExit,
      AvbdSolverStats *stats);

  /** e=0 body-wide normal clamp for endpoint fallback translations. */
  void clampWorldStaticSoftBodyEndpointVelocities(
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      const physx::PxArray<physx::PxU8> *recoveredBodies,
      const physx::PxArray<physx::PxVec3> *recoveryNormals,
      AvbdSolverStats *stats);

  /** e=0 normal clamp for recovered soft/world-static contacts. */
  void clampWorldStaticSoftInelasticNormalVelocities(
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const physx::PxArray<physx::PxU8> *recoveredContacts,
      AvbdSolverStats *stats);

  /**
   * @brief Build a local, common-face OGC manifold for freshly detected
   *        dynamic box/triangle-core intersections.
   *
   * A triangle-core row carries three independently embedded collision
   * vertices.  Before resorting to a body-wide escape, this stage chooses one
   * OBB support face for the complete soft-body/rigid-shape pair and projects
   * those three weighted supports through the ordinary coupled soft/rigid
   * response.  The material solve can therefore absorb the load as local
   * deformation.  `resolvedTriangleCoreContacts` is set only when every core
   * triangle in a group is outside the selected face, letting the coherent
   * endpoint translation remain a fail-closed fallback rather than the normal
   * contact response.
   */
  void applyDynamicSoftRigidTriangleCoreLocalManifold(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxU32 sweeps,
      physx::PxArray<physx::PxU8> *resolvedTriangleCoreContacts,
      AvbdSolverStats *stats,
      AvbdOgcPairState *ogcPairStates = nullptr,
      physx::PxU32 numOgcPairStates = 0,
      const physx::PxU32 *ogcPairIndices = nullptr,
      physx::PxU32 numOgcPairIndices = 0,
      const physx::PxU32 *ogcPairContactStarts = nullptr,
      physx::PxU32 numOgcPairContactStarts = 0,
      const physx::PxU32 *ogcPairContactRefs = nullptr,
      physx::PxU32 numOgcPairContactRefs = 0);

  /**
   * @brief Coherent paired endpoint recovery for deep current-pose
   *        soft/dynamic-rigid SDF overlap.
   *
   * This is the dynamic counterpart to the world-static coherent endpoint
   * fallback.  It is intentionally much narrower than the ordinary
   * Position-AL contact solve: it accepts only a non-CCD source which is
   * already deeply inside the true collision surface, translates an entirely
   * unconstrained soft body rigidly (preserving every tet F), and applies the
   * equal generalized linear/angular response to the dynamic rigid target.
   */
  void applyDynamicSoftRigidBodyEndpointTranslations(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxU32 sweeps,
      physx::PxArray<physx::PxU8> *recoveredContacts,
       const physx::PxArray<physx::PxVec3> *precedingStaticTranslations,
       bool allowFreshTriangleCoreExit,
       bool preferLocalTriangleCoreManifold,
       bool allowCoherentEndpointFallback,
       AvbdSolverStats *stats,
      AvbdOgcPairState *ogcPairStates = nullptr,
      physx::PxU32 numOgcPairStates = 0,
      const physx::PxU32 *ogcPairIndices = nullptr,
      physx::PxU32 numOgcPairIndices = 0,
      const physx::PxU32 *ogcPairContactStarts = nullptr,
      physx::PxU32 numOgcPairContactStarts = 0,
      const physx::PxU32 *ogcPairContactRefs = nullptr,
      physx::PxU32 numOgcPairContactRefs = 0);

  /** e=0 generalized velocity clamp for coherent dynamic endpoint recovery. */
  void clampDynamicSoftRigidBodyEndpointVelocities(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const physx::PxArray<physx::PxU8> *recoveredContacts,
      AvbdSolverStats *stats);

  /**
   * @brief Conservative current-pose recovery for genuine dynamic soft/rigid
   * SDF overlap.
   *
   * This intentionally consumes neither the OGC shell nor any swept/CCD
   * objective: it is an after-solve geometric safety projection for a
   * prepared eRIGID_SDF row whose actual surface gap is negative.  The soft
   * correction is paired with an equal generalized rigid response and its
   * initialPosition anchor moves with that correction, so the recovery is
   * excluded from the subsequent soft velocity reconstruction.
   */
  void applyDynamicSoftRigidNormalDepenetrationSweeps(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      physx::PxU32 sweeps,
      physx::PxArray<physx::PxU8> *recoveredContacts,
      AvbdSolverStats *stats,
      const AvbdOgcPairState *ogcPairStates = nullptr,
      physx::PxU32 numOgcPairStates = 0,
      const physx::PxU32 *ogcPairIndices = nullptr,
      physx::PxU32 numOgcPairIndices = 0,
      physx::PxReal softComplianceResponseScale = 1.0f,
      bool projectToCurrentPoseBoundary = false,
      const physx::PxU32 *ogcPairContactStarts = nullptr,
      physx::PxU32 numOgcPairContactStarts = 0,
      const physx::PxU32 *ogcPairContactRefs = nullptr,
      physx::PxU32 numOgcPairContactRefs = 0);

  /** e=0 normal clamp for dynamic soft/rigid recovery contacts. */
  void clampDynamicSoftRigidInelasticNormalVelocities(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const physx::PxArray<physx::PxU8> *recoveredContacts,
      AvbdSolverStats *stats);

  /**
   * @brief Shared e=0 velocity block for pairs whose same-dt OGC admission
   *        clipped an incoming endpoint at the physical collision boundary.
   *
   * Unlike geometric recovery this consumes neither an overlap nor every
   * proximity row.  It transfers only the normal momentum that would have
   * crossed an already-admitted DCD boundary through the weighted soft
   * support and the target rigid's locked 6DOF response.  All pairs for a
   * target are swept together, so opposing soft contacts balance on the same
   * rigid instead of producing serial launch impulses.
   */
  void clampAdmittedMixedOgcPairNormalVelocities(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
      const physx::PxU32 *contactPairIndices,
      physx::PxU32 numContactPairIndices,
      const physx::PxU32 *pairContactStarts,
      physx::PxU32 numPairContactStarts,
      const physx::PxU32 *pairContactRefs,
      physx::PxU32 numPairContactRefs,
      AvbdSolverStats *stats);

  /** e=0 normal clamp for rigid/static-particle soft contacts. */
  void clampKinematicShellInelasticNormalVelocities(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
      physx::PxReal dt, AvbdSolverStats *stats);

  /**
   * @brief Gauss-Seidel geometric normal depenetration for body-vs-static.
   *
   * TGS limits separation speed via maxDepenetrationVelocity; AVBD must bleed
   * deep narrow-phase overlap off over sweeps so Stage-5 velocity does not
   * encode a full bounce from one substep's position solve (no CCD).
   */
  void applyBodyStaticNormalDepenetrationSweeps(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
      const physx::PxArray<bool> *skipDepenForBodies,
      physx::PxArray<physx::PxU8> *deformableNormalStageMask,
      AvbdSolverStats *stats);

  /**
   * @brief Stage 5: Update velocities from position change
   * v = (x_new - x_n) / dt
   */
  void updateVelocities(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                        physx::PxReal invDt);

  /**
   * Shared post-AL stage list (Decision A + depen + e=0 + pose-split velocity).
   * Called from both contact and joint island paths after primal/dual iterations.
   * A strict complete velocity owner may suppress the legacy body-static
   * friction pose sweep and consume those contact derivatives after velocity
   * reconstruction. Soft particle velocity write is optional.
   */
  void postAlStages(
      physx::PxReal dt, physx::PxReal invDt, AvbdSolverBody *bodies,
      physx::PxU32 numBodies, AvbdContactConstraint *contacts,
      physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
      const physx::PxVec3 &gravity,
      bool hasBodyStaticContact, bool deformableFastImpactIsland,
      const physx::PxArray<bool> &touchingBodyStatic,
      const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
      const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
      bool allowRigidDeepPoseRecoverySplit,
      bool allowRigidFiniteMaterialPoseSplit,
      AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
      const AvbdSoftBody *softBodiesForRecovery,
      physx::PxU32 numSoftBodiesForRecovery,
      AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
      const physx::PxArray<bool> &touchesKinematicShell,
      const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
      const physx::PxArray<bool> *positionOwnedAngularBodies,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      bool hasJointConstraints, bool skipBodyStaticFriction,
      bool applyVelocityDamping,
      AvbdSoftParticle *softParticlesForVel,
      physx::PxU32 numSoftParticlesForVel,
      AvbdSolverStats &stats,
      const AvbdPostAlContactWorkPlan *postAlContactWork = nullptr,
      const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan = nullptr);

  //-------------------------------------------------------------------------
  // Energy Minimization Framework
  //-------------------------------------------------------------------------

  /**
   * @brief Compute total system energy (kinetic + potential + constraint)
   *
   * The total energy in AVBD is:
   * E_total = E_kinetic + E_potential + E_constraint
   *
   * Where:
   * - E_kinetic = 0.5 * v^T * M * v
   * - E_potential = -m * g * h (gravity potential)
   * - E_constraint = Sum(0.5 * rho * C(x)^2 + lambda * C(x))
   */
  physx::PxReal computeTotalEnergy(AvbdSolverBody *bodies,
                                   physx::PxU32 numBodies,
                                   AvbdContactConstraint *contacts,
                                   physx::PxU32 numContacts,
                                   const physx::PxVec3 &gravity);

  /**
   * @brief Compute kinetic energy of the system
   * E_kinetic = 0.5 * Sum(m * v^2 + I * omega^2)
   */
  physx::PxReal computeKineticEnergy(AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies);

  /**
   * @brief Compute potential energy of the system
   * E_potential = -Sum(m * g * h)
   */
  physx::PxReal computePotentialEnergy(AvbdSolverBody *bodies,
                                       physx::PxU32 numBodies,
                                       const physx::PxVec3 &gravity);

  /**
   * @brief Compute augmented Lagrangian constraint energy
   * E_constraint = Sum(0.5 * rho * C(x)^2 + lambda * C(x))
   */
  physx::PxReal computeConstraintEnergy(AvbdContactConstraint *contacts,
                                        physx::PxU32 numContacts,
                                        AvbdSolverBody *bodies,
                                        physx::PxU32 numBodies);

  //-------------------------------------------------------------------------
  // Helper Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Compute constraint violation for a contact
   */
  physx::PxReal computeContactViolation(const AvbdContactConstraint &contact,
                                        const AvbdSolverBody &bodyA,
                                        const AvbdSolverBody &bodyB);

  /**
   * @brief Compute constraint energy contribution
   */
  physx::PxReal computeContactEnergy(const AvbdContactConstraint &contact,
                                     const AvbdSolverBody &bodyA,
                                     const AvbdSolverBody &bodyB);

  /**
   * @brief Compute gradient of constraint energy w.r.t. body position
   */
  void computeContactGradient(const AvbdContactConstraint &contact,
                              const AvbdSolverBody &bodyA,
                              const AvbdSolverBody &bodyB,
                              physx::PxVec3 &gradPosA, physx::PxVec3 &gradRotA,
                              physx::PxVec3 &gradPosB, physx::PxVec3 &gradRotB);

  //-------------------------------------------------------------------------
  // Block Coordinate Descent - Body-Centric Constraint Solving
  //-------------------------------------------------------------------------

  /**
   * @brief Compute position/rotation correction for a D6 joint
   */
  bool computeD6JointCorrection(const AvbdD6JointConstraint &joint,
                                AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                physx::PxU32 bodyIndex, physx::PxVec3 &deltaPos,
                                physx::PxVec3 &deltaTheta);

  //-------------------------------------------------------------------------
  // Soft Body VBD Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Solve local 3x3 VBD system for a single soft particle
   * Accumulates VBD elastic forces (StVK, Neo-Hookean, bending) and
   * AVBD penalty forces (contacts and pins) into a 3x3 system, then applies
   * displacement = H^{-1} * f. Rigid attachments have a coupled owner.
   */
  void solveSoftParticle(
      PxU32 particleGlobalIdx,
      AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
      AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
      const AvbdSoftBody &softBody,
      AvbdSoftContact *softContacts, PxU32 numSoftContacts,
      const AvbdSoftContactParticleRef *softContactRefs,
      PxU32 softContactRefBegin, PxU32 softContactRefEnd,
      PxReal dt, PxReal invDt2,
      AvbdCpuIsaCorotationalTetPacket8Fn corotationalTetPacketKernel,
      const AvbdOgcPairTrustRegionContext *ogcPairContext = nullptr);

  /**
   * @brief Solve every rigid-vertex attachment as one coupled positional
   * block. This is the sole primal owner of
   * eRIGID_ATTACHMENT_POSITION_AL.
   */
  void solveSoftRigidAttachmentsCoupled(
      AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
      AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
      AvbdSoftBody *softBodies, PxU32 numSoftBodies,
      PxReal dt,
      FeatherstoneArticulation *const *articulationForBody,
      const PxU32 *linkIndexForBody);

  /**
   * @brief Solve each two-weighted-soft-point equality once as a coupled
   * positional block. This is the sole primal owner of
   * eSOFT_PAIR_ATTACHMENT_POSITION_AL.
   */
  void solveSoftPairAttachmentsCoupled(
      AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
      AvbdSoftBody *softBodies, PxU32 numSoftBodies,
      PxReal dt);

  /**
   * @brief Dual update for soft body AVBD constraints (penalty growth)
   */
  void updateSoftDual(
      AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
      AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
      AvbdSoftBody *softBodies, PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, PxU32 numSoftContacts,
      PxReal beta);

  //-------------------------------------------------------------------------
  // Member Variables
  //-------------------------------------------------------------------------

  AvbdSolverConfig mConfig;

  bool mInitialized;

};

//=============================================================================
// Inline Implementation
//=============================================================================

inline AvbdSolver::AvbdSolver() : mInitialized(false) {}

inline AvbdSolver::~AvbdSolver() { release(); }

inline void AvbdSolver::initialize(const AvbdSolverConfig &config,
                                   physx::PxAllocatorCallback &allocator) {
  PX_UNUSED(allocator);
  mConfig = config;
  mInitialized = true;
}

inline void AvbdSolver::release() {
  mInitialized = false;
}

inline void AvbdSolver::computePrediction(AvbdSolverBody *bodies,
                                          physx::PxU32 numBodies,
                                          physx::PxReal dt,
                                          const physx::PxVec3 &gravity) {
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].computePrediction(dt, gravity);
  }
}

inline void AvbdSolver::updateVelocities(AvbdSolverBody *bodies,
                                         physx::PxU32 numBodies,
                                         physx::PxReal invDt) {
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodies[i].updateVelocityFromPosition(invDt);
  }
}

inline physx::PxReal
AvbdSolver::computeContactViolation(const AvbdContactConstraint &contact,
                                    const AvbdSolverBody &bodyA,
                                    const AvbdSolverBody &bodyB) {
  // Use fullViolation (geometric gap + penetrationDepth) so that the AL
  // update sees the SAME constraint function as the inner solve.
  // Without this, the inner solve drives fullViolation->0 which makes
  // geometricViolation > 0, causing lambda to be clamped to 0 forever.
  return contact.computeFullViolation(bodyA.position, bodyA.rotation,
                                      bodyB.position, bodyB.rotation);
}

inline physx::PxReal
AvbdSolver::computeContactEnergy(const AvbdContactConstraint &contact,
                                 const AvbdSolverBody &bodyA,
                                 const AvbdSolverBody &bodyB) {
  physx::PxReal violation = computeContactViolation(contact, bodyA, bodyB);

  // For inequality constraint (contact), only penalize if violation < 0
  if (violation >= 0.0f && contact.header.lambda <= 0.0f) {
    return 0.0f;
  }

  // Augmented Lagrangian energy: E = 0.5 * rho * C^2 + lambda * C
  physx::PxReal rho = contact.header.rho;
  physx::PxReal lambda = contact.header.lambda;

  return 0.5f * rho * violation * violation + lambda * violation;
}

inline void AvbdSolver::computeContactGradient(
    const AvbdContactConstraint &contact, const AvbdSolverBody &bodyA,
    const AvbdSolverBody &bodyB, physx::PxVec3 &gradPosA,
    physx::PxVec3 &gradRotA, physx::PxVec3 &gradPosB, physx::PxVec3 &gradRotB) {
  contact.computeGradient(bodyA.rotation, bodyB.rotation, gradPosA, gradPosB,
                          gradRotA, gradRotB);

  // Scale by constraint force
  physx::PxReal violation = computeContactViolation(contact, bodyA, bodyB);
  physx::PxReal force = contact.header.rho * violation + contact.header.lambda;

  // For inequality: only apply if active
  if (violation >= 0.0f && contact.header.lambda <= 0.0f) {
    gradPosA = physx::PxVec3(0.0f);
    gradRotA = physx::PxVec3(0.0f);
    gradPosB = physx::PxVec3(0.0f);
    gradRotB = physx::PxVec3(0.0f);
    return;
  }

  gradPosA *= force;
  gradRotA *= force;
  gradPosB *= force;
  gradRotB *= force;
}

} // namespace Dy

} // namespace physx

#pragma warning(pop)

#endif // DY_AVBD_SOLVER_H
