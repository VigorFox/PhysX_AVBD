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

#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/scheduling/DyAvbdParallel.h"
#include "avbd/solver/soft/DyAvbdSoftBody.h"
#include "avbd/solver/soft/DyAvbdSoftIslandPlan.h"
#include "avbd/solver/rigid/DyAvbdSolverBody.h"
#include "avbd/core/DyAvbdTypes.h"

#pragma warning(push)
#pragma warning(disable : 4324)

namespace physx {

namespace Dy {

class FeatherstoneArticulation;
class AvbdRigidBodyRangeTask;
class AvbdRigidDualRangeTask;
struct AvbdPostAlVelocityState;
struct AvbdPostAlFrictionState;
struct AvbdPostAlPoseState;
struct AvbdOgcPairTrustRegionContext;

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
struct AvbdJointOgcAdmissionState;
struct AvbdJointPostAlPhaseState;
struct AvbdJointIterationPhaseInput;
struct AvbdJointPrimalPhaseInput;
struct AvbdJointSoftParticlePhaseInput;
struct AvbdJointOgcVelocityHandoffInput;
struct AvbdPostAlRecoveryState;

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
  friend struct AvbdJointOgcAdmissionState;
  friend struct AvbdJointPostAlPhaseState;
  friend struct AvbdPostAlRecoveryState;
  friend struct AvbdPostAlFrictionState;
  friend struct AvbdPostAlPoseState;
  friend class AvbdRigidBodyRangeTask;
  friend class AvbdRigidDualRangeTask;
  friend class AvbdSolveIslandTask;

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
   * @brief Stage 5: update contact, joint and soft augmented-Lagrangian duals.
   *
   * The implementation preserves the existing D6/gear/soft ordering while
   * giving solveWithJoints an explicit dual-update phase boundary.
   */
  void runAvbdJointDualPhase(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
      physx::PxReal dt, physx::PxReal invDt2,
      const AvbdSolverConfig &config,
      AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
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
  /** Final velocity handoff for same-dt OGC endpoint corrections. */
  void applyAvbdJointOgcVelocityHandoff(
      const AvbdJointOgcVelocityHandoffInput &input);

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
      const AvbdTetMaterialPacketKernels &tetMaterialPacketKernels,
      const AvbdOgcPairTrustRegionContext *ogcPairContext = nullptr);

  /** Execute the serial nonlinear soft-particle primal and coupled owners. */
  void runAvbdJointSoftParticlePhase(
      const AvbdJointSoftParticlePhaseInput &input);

  /** Execute one mixed joint/soft primal iteration. */
  void runAvbdJointPrimalIteration(
      const AvbdJointPrimalPhaseInput &input);

  /** Execute the mixed position primal/dual iteration lifecycle. */
  void runAvbdJointIterationPhase(
      const AvbdJointIterationPhaseInput &input);

  /** Execute prediction, tangent ownership, and adaptive warmstart. */
  void runAvbdJointWarmstartPhase(
      const AvbdSolverConfig &config, PxReal dt, PxReal invDt,
      PxReal invDt2, AvbdSolverBody *bodies, PxU32 numBodies,
      const PxVec3 &gravity,
      const physx::PxArray<bool> &touchesKinematicShell,
      AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
      AvbdSoftBody *softBodies, PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, PxU32 numSoftContacts,
      bool hasPreparedSoftPrediction);

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
