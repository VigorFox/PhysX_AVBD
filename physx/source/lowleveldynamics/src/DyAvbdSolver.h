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
             AvbdColorBatch *colorBatches = nullptr,
             physx::PxU32 numColors = 0);

  /**
   * @brief Execute one simulation step with joint constraints
   * @param dt Time step
   * @param bodies Array of solver bodies
   * @param numBodies Number of bodies
   * @param contacts Array of contact constraints
   * @param numContacts Number of contacts
   * @param sphericalJoints Array of spherical joint constraints
   * @param numSpherical Number of spherical joints
   * @param fixedJoints Array of fixed joint constraints
   * @param numFixed Number of fixed joints
   * @param revoluteJoints Array of revolute joint constraints
   * @param numRevolute Number of revolute joints
   * @param prismaticJoints Array of prismatic joint constraints
   * @param numPrismatic Number of prismatic joints
   * @param d6Joints Array of D6 joint constraints
   * @param numD6 Number of D6 joints
   * @param gearJoints Array of gear joint constraints
   * @param numGear Number of gear joints
   * @param gravity Gravity vector
   * @param contactMap Pre-computed contact-to-body mapping (optional)
   * @param sphericalMap Pre-computed spherical joint mapping (optional)
   * @param fixedMap Pre-computed fixed joint mapping (optional)
   * @param revoluteMap Pre-computed revolute joint mapping (optional)
   * @param prismaticMap Pre-computed prismatic joint mapping (optional)
   * @param d6Map Pre-computed D6 joint mapping (optional)
   * @param gearMap Pre-computed gear joint mapping (optional)
   * @param colorBatches Pre-computed color batches (nullptr for no coloring)
   * @param numColors Number of colors in colorBatches (0 if not colored)
   */
  void solveWithJoints(
      physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
      AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
      AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
      AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
      const physx::PxVec3 &gravity, 
      const AvbdBodyConstraintMap *contactMap = nullptr,
      const AvbdBodyConstraintMap *sphericalMap = nullptr,
      const AvbdBodyConstraintMap *fixedMap = nullptr,
      const AvbdBodyConstraintMap *revoluteMap = nullptr,
      const AvbdBodyConstraintMap *prismaticMap = nullptr,
      const AvbdBodyConstraintMap *d6Map = nullptr,
      const AvbdBodyConstraintMap *gearMap = nullptr,
      AvbdColorBatch *colorBatches = nullptr,
      physx::PxU32 numColors = 0);

  /**
   * @brief Get solver statistics from last solve
   */
  const AvbdSolverStats &getStats() const { return mStats; }

  /**
   * @brief Get solver configuration
   */
  const AvbdSolverConfig &getConfig() const { return mConfig; }

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
                           physx::PxU32 numContacts);

  /**
   * @brief Stage 2b: Compute body-based coloring for block coordinate descent
   * Bodies sharing constraints get different colors, enabling parallel BCD.
   */
  void computeBodyColoring(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                          AvbdContactConstraint *contacts,
                          physx::PxU32 numContacts);

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
                             AvbdColorBatch *colorBatches = nullptr,
                             physx::PxU32 numColors = 0);

  /**
   * @brief Solve local 6x6 system for a single body
   * Minimizes: 1/(2h^2) * ||M(x - x_tilde)||^2 + Sum constraint_energy
   */
  void solveLocalSystem(AvbdSolverBody &body, AvbdContactConstraint *contacts,
                        physx::PxU32 numContacts, physx::PxReal dt,
                        physx::PxReal invDt2);

  /**
   * @brief Stage 4: Update Augmented Lagrangian multipliers
   * lambda <- lambda + rho * C(x)
   */
  void updateLagrangianMultipliers(AvbdSolverBody *bodies,
                                   physx::PxU32 numBodies,
                                   AvbdContactConstraint *contacts,
                                   physx::PxU32 numContacts);

  /**
   * @brief Stage 5: Update velocities from position change
   * v = (x_new - x_n) / dt
   */
  void updateVelocities(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                       physx::PxReal invDt);

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
  physx::PxReal computeTotalEnergy(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                                    const physx::PxVec3 &gravity);

  /**
   * @brief Compute kinetic energy of the system
   * E_kinetic = 0.5 * Sum(m * v^2 + I * omega^2)
   */
  physx::PxReal computeKineticEnergy(AvbdSolverBody *bodies, physx::PxU32 numBodies);

  /**
   * @brief Compute potential energy of the system
   * E_potential = -Sum(m * g * h)
   */
  physx::PxReal computePotentialEnergy(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                        const physx::PxVec3 &gravity);

  /**
   * @brief Compute augmented Lagrangian constraint energy
   * E_constraint = Sum(0.5 * rho * C(x)^2 + lambda * C(x))
   */
  physx::PxReal computeConstraintEnergy(AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                                         AvbdSolverBody *bodies, physx::PxU32 numBodies);

  /**
   * @brief Compute energy gradient for a single body
   * Returns the 6D gradient vector [dE/dp, dE/dtheta]
   */
  void computeEnergyGradient(physx::PxU32 bodyIndex, AvbdSolverBody *bodies, physx::PxU32 numBodies,
                            AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                            physx::PxReal invDt2, AvbdVec6 &gradient);

  /**
   * @brief Check convergence based on energy change
   * Returns true if |E_new - E_old| < tolerance
   */
  bool checkEnergyConvergence(physx::PxReal oldEnergy, physx::PxReal newEnergy,
                             physx::PxReal tolerance) const;

  /**
   * @brief Perform line search for optimal step size
   * Uses Armijo backtracking to ensure sufficient energy decrease
   */
  physx::PxReal performLineSearch(AvbdSolverBody &body, const AvbdVec6 &direction,
                                   physx::PxReal initialStep, physx::PxReal energy,
                                   physx::PxReal c1, physx::PxReal rho);

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
                              physx::PxVec3 &gradPosA,
                              physx::PxVec3 &gradRotA,
                              physx::PxVec3 &gradPosB,
                              physx::PxVec3 &gradRotB);

  /**
   * @brief Build Hessian matrix for a body's local system
   */
  void buildHessianMatrix(const AvbdSolverBody &body,
                          AvbdContactConstraint *contacts,
                          physx::PxU32 numContacts,
                          physx::PxReal invDt2,
                          AvbdBlock6x6 &H);

  /**
   * @brief Build gradient vector for a body's local system
   */
  void buildGradientVector(const AvbdSolverBody &body,
                          AvbdContactConstraint *contacts,
                          physx::PxU32 numContacts,
                          physx::PxReal invDt2,
                          AvbdVec6 &g);

  /**
   * @brief Collect all constraints affecting a specific body
   */
  physx::PxU32 collectBodyConstraints(physx::PxU32 bodyIndex,
                                       AvbdContactConstraint *contacts,
                                       physx::PxU32 numContacts,
                                       physx::PxU32 *constraintIndices);

  //-------------------------------------------------------------------------
  // Block Coordinate Descent - Body-Centric Constraint Solving
  //-------------------------------------------------------------------------

  /**
   * @brief Solve all constraints (contacts only) affecting a single body
   * This is the core AVBD block descent step for contact-only solving.
   */
  void solveBodyLocalConstraints(AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                 physx::PxU32 bodyIndex,
                                 AvbdContactConstraint *contacts,
                                 physx::PxU32 numContacts);

  /**
   * @brief Solve all constraints (contacts + joints) affecting a single body
   * This is the core AVBD block descent step for joint solving.
   */
  void solveBodyAllConstraints(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
      AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
      AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
      AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      physx::PxReal dt);

  /**
   * @brief Compute position/rotation correction for a contact constraint
   * @return true if constraint is active and correction was computed
   */
  bool computeContactCorrection(
      const AvbdContactConstraint &contact,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  /**
   * @brief Compute position/rotation correction for a spherical joint
   */
  bool computeSphericalJointCorrection(
      const AvbdSphericalJointConstraint &joint,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  /**
   * @brief Compute position/rotation correction for a fixed joint
   */
  bool computeFixedJointCorrection(
      const AvbdFixedJointConstraint &joint,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  /**
   * @brief Compute position/rotation correction for a revolute joint
   */
  bool computeRevoluteJointCorrection(
      const AvbdRevoluteJointConstraint &joint,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  /**
   * @brief Compute position/rotation correction for a prismatic joint
   */
  bool computePrismaticJointCorrection(
      const AvbdPrismaticJointConstraint &joint,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  /**
   * @brief Compute position/rotation correction for a D6 joint
   */
  bool computeD6JointCorrection(
      const AvbdD6JointConstraint &joint,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  /**
   * @brief Compute angular correction for a gear joint
   */
  bool computeGearJointCorrection(
      const AvbdGearJointConstraint &joint,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      physx::PxVec3 &deltaPos, physx::PxVec3 &deltaTheta);

  //-------------------------------------------------------------------------
  // Member Variables
  //-------------------------------------------------------------------------

  AvbdSolverConfig mConfig;
  AvbdSolverStats mStats;
  AvbdGraphColoring mColoring;
  AvbdParallelColoring
      mParallelColoring; //!< Constraint-based parallel coloring
  AvbdBodyParallelColoring
      mBodyColoring; //!< Body-based parallel coloring for BCD
  AvbdBodyConstraintMap
      mContactMap;    //!< Pre-computed contact-to-body mapping for O(1) lookup
  AvbdBodyConstraintMap
      mSphericalMap;  //!< Pre-computed spherical joint mapping
  AvbdBodyConstraintMap
      mFixedMap;      //!< Pre-computed fixed joint mapping
  AvbdBodyConstraintMap
      mRevoluteMap;   //!< Pre-computed revolute joint mapping
  AvbdBodyConstraintMap
      mPrismaticMap;  //!< Pre-computed prismatic joint mapping
  AvbdBodyConstraintMap
      mD6Map;         //!< Pre-computed D6 joint mapping

  physx::PxAllocatorCallback *mAllocator;
  bool mInitialized;

  //-------------------------------------------------------------------------
  // Optimized solving with pre-computed constraint mapping
  //-------------------------------------------------------------------------
  
  /**
   * @brief Build constraint-to-body mapping for efficient lookup
   */
  void buildConstraintMapping(AvbdContactConstraint *contacts, physx::PxU32 numContacts,
                              physx::PxU32 numBodies);
  
  /**
   * @brief Optimized version using pre-computed constraint map - O(constraints per body)
   */
  void solveBodyLocalConstraintsFast(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      AvbdContactConstraint *contacts);
  
  /**
   * @brief Build all constraint mappings for joints (called once before solve iterations)
   */
  void buildAllConstraintMappings(
      physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
      AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
      AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
      AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6);
  
  /**
   * @brief Optimized version of solveBodyAllConstraints using pre-computed mappings
   * Complexity: O(constraints connected to this body) instead of O(all constraints)
   */
  void solveBodyAllConstraintsFast(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      physx::PxU32 bodyIndex,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      AvbdSphericalJointConstraint *sphericalJoints, physx::PxU32 numSpherical,
      AvbdFixedJointConstraint *fixedJoints, physx::PxU32 numFixed,
      AvbdRevoluteJointConstraint *revoluteJoints, physx::PxU32 numRevolute,
      AvbdPrismaticJointConstraint *prismaticJoints, physx::PxU32 numPrismatic,
      AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
      AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
      const AvbdBodyConstraintMap &contactMap,
      const AvbdBodyConstraintMap &sphericalMap,
      const AvbdBodyConstraintMap &fixedMap,
      const AvbdBodyConstraintMap &revoluteMap,
      const AvbdBodyConstraintMap &prismaticMap,
      const AvbdBodyConstraintMap &d6Map,
      const AvbdBodyConstraintMap &gearMap,
      physx::PxReal dt);
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
  mSphericalMap = AvbdBodyConstraintMap();
  mFixedMap = AvbdBodyConstraintMap();
  mRevoluteMap = AvbdBodyConstraintMap();
  mPrismaticMap = AvbdBodyConstraintMap();
  mD6Map = AvbdBodyConstraintMap();
}

inline AvbdSolver::~AvbdSolver() { release(); }

inline void AvbdSolver::initialize(const AvbdSolverConfig &config,
                                   physx::PxAllocatorCallback &allocator) {
  mConfig = config;
  mAllocator = &allocator;
  mStats.reset();
  mInitialized = true;
}

inline void AvbdSolver::release() {
  if (mInitialized && mAllocator) {
    if (mColoring.colorGroups != nullptr) {
      mColoring.release(*mAllocator);
    }
    // Release all constraint mappings
    mContactMap.release(*mAllocator);
    mSphericalMap.release(*mAllocator);
    mFixedMap.release(*mAllocator);
    mRevoluteMap.release(*mAllocator);
    mPrismaticMap.release(*mAllocator);
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
  return contact.computeViolation(bodyA.position, bodyA.rotation,
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

  // Augmented Lagrangian energy: E = 0.5 * rho * C² + lambda * C
  physx::PxReal rho = contact.header.rho;
  physx::PxReal lambda = contact.header.lambda;

  return 0.5f * rho * violation * violation + lambda * violation;
}

inline void AvbdSolver::computeContactGradient(
    const AvbdContactConstraint &contact, const AvbdSolverBody &bodyA,
    const AvbdSolverBody &bodyB, physx::PxVec3 &gradPosA,
    physx::PxVec3 &gradRotA, physx::PxVec3 &gradPosB,
    physx::PxVec3 &gradRotB) {
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
