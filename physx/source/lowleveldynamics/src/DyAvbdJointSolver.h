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

#ifndef DY_AVBD_JOINT_SOLVER_H
#define DY_AVBD_JOINT_SOLVER_H

#include "DyAvbdConstraint.h"
#include "DyAvbdSolverBody.h"
#include "DyAvbdTypes.h"

namespace physx {
namespace Dy {

//=============================================================================
// Joint Solver Functions (Static Polymorphism Pattern)
//
// These functions implement the XPBD-style position correction for each
// joint type. They follow the pattern used by processContactConstraint().
//=============================================================================

/**
 * @brief Process a spherical (ball) joint constraint
 *
 * The spherical joint constrains two anchor points to be coincident.
 * This is a 3-DOF position constraint.
 *
 * Algorithm:
 * 1. Compute world anchor positions
 * 2. Compute position error (violation)
 * 3. Compute effective mass (including rotational contribution)
 * 4. Apply XPBD-style position correction
 * 5. Update Lagrangian multiplier
 *
 * @param joint The spherical joint constraint
 * @param bodies Array of all solver bodies
 * @param numBodies Total number of bodies
 * @param config Solver configuration
 * @param dt Time step
 */
void processSphericalJointConstraint(AvbdSphericalJointConstraint &joint,
                                     AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     const AvbdSolverConfig &config,
                                     physx::PxReal dt);

/**
 * @brief Process a fixed joint constraint
 *
 * The fixed joint locks all 6 DOF between two bodies:
 * - 3 DOF position constraint (anchor points coincident)
 * - 3 DOF rotation constraint (orientations match)
 *
 * @param joint The fixed joint constraint
 * @param bodies Array of all solver bodies
 * @param numBodies Total number of bodies
 * @param config Solver configuration
 * @param dt Time step
 */
void processFixedJointConstraint(AvbdFixedJointConstraint &joint,
                                 AvbdSolverBody *bodies, physx::PxU32 numBodies,
                                 const AvbdSolverConfig &config,
                                 physx::PxReal dt);

/**
 * @brief Process a revolute (hinge) joint constraint
 *
 * The revolute joint allows rotation around a single axis:
 * - 3 DOF position constraint (anchor points coincident)
 * - 2 DOF axis alignment constraint (axes must be parallel)
 * - Optional angle limit constraint
 * - Optional motor drive
 *
 * @param joint The revolute joint constraint
 * @param bodies Array of all solver bodies
 * @param numBodies Total number of bodies
 * @param config Solver configuration
 * @param dt Time step
 */
void processRevoluteJointConstraint(AvbdRevoluteJointConstraint &joint,
                                    AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    const AvbdSolverConfig &config,
                                    physx::PxReal dt);

/**
 * @brief Process a prismatic (slider) joint constraint
 *
 * The prismatic joint allows translation along a single axis:
 * - 2 DOF position constraint (perpendicular to slide axis)
 * - 3 DOF rotation constraint (orientations locked)
 * - Optional linear limit constraint
 * - Optional motor drive
 *
 * @param joint The prismatic joint constraint
 * @param bodies Array of all solver bodies
 * @param numBodies Total number of bodies
 * @param config Solver configuration
 * @param dt Time step
 */
void processPrismaticJointConstraint(AvbdPrismaticJointConstraint &joint,
                                     AvbdSolverBody *bodies,
                                     physx::PxU32 numBodies,
                                     const AvbdSolverConfig &config,
                                     physx::PxReal dt);

/**
 * @brief Process a D6 (configurable) joint constraint
 *
 * The D6 joint allows independent control of all 6 degrees of freedom:
 * - 3 Linear DOFs (X, Y, Z translation)
 * - 3 Angular DOFs (X, Y, Z rotation)
 *
 * Each DOF can be configured as:
 * - LOCKED: Constrained to zero
 * - FREE: No constraint
 * - LIMITED: Constrained within a range
 *
 * @param joint The D6 joint constraint
 * @param bodies Array of all solver bodies
 * @param numBodies Total number of bodies
 * @param config Solver configuration
 * @param dt Time step
 */
void processD6JointConstraint(AvbdD6JointConstraint &joint,
                              AvbdSolverBody *bodies, physx::PxU32 numBodies,
                              const AvbdSolverConfig &config, physx::PxReal dt);

//=============================================================================
// Augmented Lagrangian Multiplier Updates
//=============================================================================

/**
 * @brief Update Lagrangian multipliers for spherical joint
 */
void updateSphericalJointMultiplier(AvbdSphericalJointConstraint &joint,
                                    const AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    const AvbdSolverConfig &config);

/**
 * @brief Update Lagrangian multipliers for fixed joint
 */
void updateFixedJointMultiplier(AvbdFixedJointConstraint &joint,
                                const AvbdSolverBody *bodies,
                                physx::PxU32 numBodies,
                                const AvbdSolverConfig &config);

/**
 * @brief Update Lagrangian multipliers for revolute joint
 */
void updateRevoluteJointMultiplier(AvbdRevoluteJointConstraint &joint,
                                   const AvbdSolverBody *bodies,
                                   physx::PxU32 numBodies,
                                   const AvbdSolverConfig &config);

/**
 * @brief Update Lagrangian multipliers for prismatic joint
 */
void updatePrismaticJointMultiplier(AvbdPrismaticJointConstraint &joint,
                                    const AvbdSolverBody *bodies,
                                    physx::PxU32 numBodies,
                                    const AvbdSolverConfig &config);

/**
 * @brief Update Lagrangian multipliers for D6 joint
 */
void updateD6JointMultiplier(AvbdD6JointConstraint &joint,
                             const AvbdSolverBody *bodies,
                             physx::PxU32 numBodies,
                             const AvbdSolverConfig &config);

/**
 * @brief Process a gear joint constraint
 *
 * @brief Update Lagrangian multipliers for gear joint
 */
void updateGearJointMultiplier(AvbdGearJointConstraint &joint,
                               const AvbdSolverBody *bodies,
                               physx::PxU32 numBodies,
                               const AvbdSolverConfig &config);

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Compute effective inverse mass for a position constraint
 *
 * For and anchor-based constraint between two bodies:
 *   w_eff = invMassA + (rA x n)^T * I_A^-1 * (rA x n)
 *         + invMassB + (rB x n)^T * I_B^-1 * (rB x n)
 *
 * @param bodyA First body
 * @param bodyB Second body
 * @param rA Anchor offset from body A center (world space)
 * @param rB Anchor offset from body B center (world space)
 * @param n Constraint direction (normalized)
 * @return Effective inverse mass along direction n
 */
PX_FORCE_INLINE physx::PxReal computeEffectiveInverseMass(
    const AvbdSolverBody &bodyA, const AvbdSolverBody &bodyB,
    const physx::PxVec3 &rA, const physx::PxVec3 &rB, const physx::PxVec3 &n) {
  // Linear contribution
  physx::PxReal wLin = bodyA.invMass + bodyB.invMass;

  // Angular contribution for body A
  physx::PxVec3 rAxN = rA.cross(n);
  physx::PxVec3 angA = bodyA.invInertiaWorld * rAxN;
  physx::PxReal wAngA = rAxN.dot(angA);

  // Angular contribution for body B
  physx::PxVec3 rBxN = rB.cross(n);
  physx::PxVec3 angB = bodyB.invInertiaWorld * rBxN;
  physx::PxReal wAngB = rBxN.dot(angB);

  return wLin + wAngA + wAngB;
}

/**
 * @brief Apply position correction to a body
 *
 * Updates position and rotation based on the correction impulse.
 *
 * @param body The body to correct
 * @param linearCorrection Linear correction impulse
 * @param angularCorrection Angular correction impulse
 */
PX_FORCE_INLINE void
applyPositionCorrection(AvbdSolverBody &body,
                        const physx::PxVec3 &linearCorrection,
                        const physx::PxVec3 &angularCorrection) {
  // Apply linear correction
  body.position += linearCorrection * body.invMass;

  // Apply angular correction
  physx::PxVec3 angularDelta = body.invInertiaWorld * angularCorrection;

  // Update rotation using quaternion derivative
  // q' = q + 0.5 * dt * omega_quat * q
  // For small angles: dq ~= [0.5 * delta, 1]
  physx::PxQuat deltaQ(angularDelta.x * 0.5f, angularDelta.y * 0.5f,
                       angularDelta.z * 0.5f, 1.0f);
  body.rotation = (deltaQ * body.rotation).getNormalized();
}

/**
 * @brief Compute the skew-symmetric matrix from a vector
 * Used for cross product: [v]x * u = v x u
 */
PX_FORCE_INLINE physx::PxMat33 skewMatrix(const physx::PxVec3 &v) {
  return physx::PxMat33(physx::PxVec3(0, v.z, -v.y),
                        physx::PxVec3(-v.z, 0, v.x),
                        physx::PxVec3(v.y, -v.x, 0));
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_JOINT_SOLVER_H
