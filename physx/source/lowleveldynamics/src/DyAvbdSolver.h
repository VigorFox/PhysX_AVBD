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

/**
 * @brief Main AVBD Solver class implementing the Block Coordinate Descent
 * algorithm
 *
 * The AVBD solver operates on position-level variables and uses:
 * 1. Prediction integration (explicit Euler)
 * 2. Graph coloring for parallel body updates
 * 3. Block descent solve for each body's local 6x6 system
 * 4. Augmented Lagrangian multiplier updates for constraint satisfaction
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
                   AvbdSolverStats &stats);

  /**
   * @brief Get solver configuration
   */
  const AvbdSolverConfig &getConfig() const { return mConfig; }

  /** Mutable config (scene bounce threshold, material gates). */
  AvbdSolverConfig &getConfigMutable() { return mConfig; }

private:
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
   * @brief Stage 2: Build constraint graph and compute body coloring
   */
  void computeGraphColoring(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                            AvbdContactConstraint *contacts,
                            physx::PxU32 numContacts,
                            AvbdSolverStats &stats);

  /**
   * @brief Stage 2b: Compute body-based coloring for block coordinate descent
   * Bodies sharing constraints get different colors, enabling parallel BCD.
   */
  void computeBodyColoring(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                           AvbdContactConstraint *contacts,
                           physx::PxU32 numContacts,
                           AvbdSolverStats &stats);

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
      physx::PxU32 &numTouching);

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
      physx::PxU32 numSoftContacts = 0);

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

  /** Capture the exact normal-row state consumed by the first AL iteration. */
  void captureBodyStaticNormalDiagnosticStart(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts);

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
      physx::PxU32 numContacts, const physx::PxVec3 &gravity,
      bool hasBodyStaticContact, bool deformableFastImpactIsland,
      const physx::PxArray<bool> &touchingBodyStatic,
      const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
      const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
      bool allowRigidDeepPoseRecoverySplit,
      bool allowRigidFiniteMaterialPoseSplit,
      AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
      AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
      const physx::PxArray<bool> &touchesKinematicShell,
      const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      bool hasJointConstraints, bool skipBodyStaticFriction,
      bool applyVelocityDamping,
      AvbdSoftParticle *softParticlesForVel,
      physx::PxU32 numSoftParticlesForVel, AvbdSolverStats &stats);

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
   * AVBD penalty forces (contacts, attachments, pins) into a 3x3 system,
   * then applies displacement = H^{-1} * f.
   */
  void solveSoftParticle(
      PxU32 particleGlobalIdx,
      AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
      AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
      AvbdSoftBody *softBodies, PxU32 numSoftBodies,
      AvbdSoftContact *softContacts, PxU32 numSoftContacts,
      PxReal dt, PxReal invDt2);

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
  AvbdGraphColoring mColoring;
  AvbdParallelColoring
      mParallelColoring; //!< Constraint-based parallel coloring
  AvbdBodyParallelColoring
      mBodyColoring; //!< Body-based parallel coloring for BCD
  AvbdBodyConstraintMap
      mContactMap; //!< Pre-computed contact-to-body mapping for O(1) lookup
  AvbdBodyConstraintMap mD6Map;        //!< Pre-computed D6 joint mapping

  physx::PxAllocatorCallback *mAllocator;
  bool mInitialized;

  //-------------------------------------------------------------------------
  // Optimized solving with pre-computed constraint mapping
  //-------------------------------------------------------------------------

  /**
   * @brief Build constraint-to-body mapping for efficient lookup
   */
  void buildConstraintMapping(AvbdContactConstraint *contacts,
                              physx::PxU32 numContacts, physx::PxU32 numBodies);

  /**
   * @brief Optimized version using pre-computed constraint map -
   * O(constraints per body)
   */
  void solveBodyLocalConstraintsFast(AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     physx::PxU32 bodyIndex,
                                     AvbdContactConstraint *contacts);

  /**
   * @brief Thread-safe version using external constraint map -
   * O(constraints per body)
   */
  void solveBodyLocalConstraintsFastWithMap(
      AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 bodyIndex,
      AvbdContactConstraint *contacts, const AvbdBodyConstraintMap &contactMap);

  /**
   * @brief Build all constraint mappings for joints (called once before
   * solve iterations)
   */
  void buildAllConstraintMappings(
      physx::PxU32 numBodies, AvbdContactConstraint *contacts,
      physx::PxU32 numContacts, AvbdD6JointConstraint *d6Joints,
      physx::PxU32 numD6);
};

//=============================================================================
// Inline Implementation
//=============================================================================

inline AvbdSolver::AvbdSolver() : mAllocator(nullptr), mInitialized(false) {
  // Initialize coloring to safe defaults
  mColoring.colorGroups = nullptr;
  mColoring.numColors = 0;
  mColoring.maxColors = 0;

  // Explicitly initialize all constraint mappings to safe defaults
  // (redundant if default constructors work, but safer)
  mContactMap = AvbdBodyConstraintMap();
  mD6Map = AvbdBodyConstraintMap();
}

inline AvbdSolver::~AvbdSolver() { release(); }

inline void AvbdSolver::initialize(const AvbdSolverConfig &config,
                                   physx::PxAllocatorCallback &allocator) {
  mConfig = config;
  mAllocator = &allocator;
  mInitialized = true;
}

inline void AvbdSolver::release() {
  if (mInitialized && mAllocator) {
    if (mColoring.colorGroups != nullptr) {
      mColoring.release(*mAllocator);
    }
    // Release all constraint mappings
    mContactMap.release(*mAllocator);
    mD6Map.release(*mAllocator);
  }
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
