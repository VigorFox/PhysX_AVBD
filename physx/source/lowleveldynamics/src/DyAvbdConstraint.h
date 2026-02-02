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

#ifndef DY_AVBD_CONSTRAINT_H
#define DY_AVBD_CONSTRAINT_H

#include "foundation/PxMat33.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

#pragma warning(push)
#pragma warning(                                                               \
    disable : 4324) // Structure was padded due to alignment specifier

namespace physx {

namespace Dy {

// Forward declarations
struct AvbdSolverBody;

/**
 * @brief Constraint type enumeration for AVBD solver
 */
struct AvbdConstraintType {
  enum Enum {
    eNONE = 0,        //!< Invalid/uninitialized constraint
    eCONTACT,         //!< Contact constraint (inequality, unilateral)
    eFRICTION,        //!< Friction constraint
    eJOINT_SPHERICAL, //!< Spherical (ball) joint
    eJOINT_REVOLUTE,  //!< Revolute (hinge) joint
    eJOINT_PRISMATIC, //!< Prismatic (slider) joint
    eJOINT_FIXED,     //!< Fixed joint (all DOF locked)
    eJOINT_D6,        //!< Configurable 6-DOF joint
    eJOINT_WELD,      //!< Weld joint (optimized for runtime attachment, e.g., "Ultrahand")
    eSOFT_DISTANCE,   //!< Soft body distance constraint
    eSOFT_VOLUME,     //!< Soft body volume preservation

    eCOUNT
  };
};

/**
 * @brief Base structure for AVBD constraints
 *
 * AVBD constraints are fundamentally different from PhysX's Jacobian-based
 * constraints. Instead of velocity-level Jacobian rows, AVBD constraints
 * define position-level energy functions and their derivatives.
 *
 * The constraint energy has the form:
 *   E(x) = 0.5 * compliance^-1 * C(x)^2 + lambda * C(x)
 *
 * Where:
 *   - C(x) is the constraint function (violation)
 *   - compliance is the inverse stiffness
 *   - lambda is the Augmented Lagrangian multiplier
 */
struct PX_ALIGN_PREFIX(16) AvbdConstraintHeader {
  physx::PxU32 bodyIndexA; //!< Index of first body (PX_MAX_U32 if world)
  physx::PxU32 bodyIndexB; //!< Index of second body (PX_MAX_U32 if world)
  physx::PxU16 type;       //!< Constraint type (AvbdConstraintType::Enum)
  physx::PxU16 flags;      //!< Constraint flags

  physx::PxReal compliance; //!< Constraint compliance (inverse stiffness), 0
                             //!< for hard constraint
  physx::PxReal damping; //!< Damping coefficient for velocity-dependent forces
  physx::PxReal lambda;  //!< Augmented Lagrangian multiplier
  physx::PxReal rho;     //!< Penalty parameter for ALM

  physx::PxU16 colorGroup; //!< Color group for parallel solving
  physx::PxU16 padding0;   //!< Padding for alignment

  PX_FORCE_INLINE AvbdConstraintHeader()
      : bodyIndexA(0xFFFFFFFF), bodyIndexB(0xFFFFFFFF),
        type(AvbdConstraintType::eNONE), flags(0), compliance(0.0f),
        damping(0.0f), lambda(0.0f), rho(0.0f), colorGroup(0), padding0(0) {}
} PX_ALIGN_SUFFIX(16);

/**
 * @brief Contact constraint for AVBD solver
 *
 * Represents a unilateral (inequality) contact constraint:
 *   C(x) = (pA - pB) · n - d >= 0
 *
 * Where:
 *   - pA, pB are contact points on bodies A and B
 *   - n is the contact normal (from B to A)
 *   - d is the rest distance (usually 0)
 */
struct PX_ALIGN_PREFIX(16) AvbdContactConstraint {
  AvbdConstraintHeader header;

  //-------------------------------------------------------------------------
  // Contact geometry
  //-------------------------------------------------------------------------

  physx::PxVec3 contactPointA; //!< Contact point on body A (local space)
  physx::PxReal
      penetrationDepth; //!< Current penetration depth (negative = separated)

  physx::PxVec3 contactPointB; //!< Contact point on body B (local space)
  physx::PxReal restitution;   //!< Coefficient of restitution

  physx::PxVec3 contactNormal; //!< Contact normal (world space, from B to A)
  physx::PxReal friction;      //!< Friction coefficient

  //-------------------------------------------------------------------------
  // Friction tangents
  //-------------------------------------------------------------------------

  physx::PxVec3 tangent0;       //!< First friction tangent direction
  physx::PxReal tangentLambda0; //!< Lambda for first friction direction

  physx::PxVec3 tangent1;       //!< Second friction tangent direction
  physx::PxReal tangentLambda1; //!< Lambda for second friction direction

  //-------------------------------------------------------------------------
  // Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Compute constraint violation C(x)
   * @param posA World position of body A
   * @param rotA World rotation of body A
   * @param posB World position of body B
   * @param rotB World rotation of body B
   * @return Constraint violation (negative = penetration)
   */
  PX_FORCE_INLINE physx::PxReal
  computeViolation(const physx::PxVec3 &posA, const physx::PxQuat &rotA,
                   const physx::PxVec3 &posB,
                   const physx::PxQuat &rotB) const {
    physx::PxVec3 worldPointA = posA + rotA.rotate(contactPointA);
    physx::PxVec3 worldPointB = posB + rotB.rotate(contactPointB);
    return (worldPointA - worldPointB).dot(contactNormal);
  }

  /**
   * @brief Compute gradient of constraint function
   * @param rotA World rotation of body A
   * @param rotB World rotation of body B
   * @param[out] gradPosA Gradient w.r.t. position of A
   * @param[out] gradPosB Gradient w.r.t. position of B
   * @param[out] gradRotA Gradient w.r.t. rotation of A (as angular)
   * @param[out] gradRotB Gradient w.r.t. rotation of B (as angular)
   */
  PX_FORCE_INLINE void
  computeGradient(const physx::PxQuat &rotA, const physx::PxQuat &rotB,
                  physx::PxVec3 &gradPosA, physx::PxVec3 &gradPosB,
                  physx::PxVec3 &gradRotA, physx::PxVec3 &gradRotB) const {
    // Gradient w.r.t. position is simply the contact normal
    gradPosA = contactNormal;
    gradPosB = -contactNormal;

    // Gradient w.r.t. rotation: d(R*r)/dq contributes to the constraint
    physx::PxVec3 rA = rotA.rotate(contactPointA);
    physx::PxVec3 rB = rotB.rotate(contactPointB);

    gradRotA = rA.cross(contactNormal);
    gradRotB = -rB.cross(contactNormal);
  }

  /**
   * @brief Check if this is an active (penetrating) contact
   */
  PX_FORCE_INLINE bool isActive() const {
    return penetrationDepth < 0.0f || header.lambda > 0.0f;
  }

  /**
   * @brief Compute augmented Lagrangian energy
   * E = 0.5 * rho * C^2 + lambda * C
   */
  PX_FORCE_INLINE physx::PxReal computeAugmentedLagrangianEnergy(
      const physx::PxVec3 &posA,
      const physx::PxQuat &rotA,
      const physx::PxVec3 &posB,
      const physx::PxQuat &rotB) const {
    
    physx::PxReal violation = computeViolation(posA, rotA, posB, rotB);
    
    // For inequality constraint (contact), only penalize if violating
    if (violation >= 0.0f && header.lambda <= 0.0f) {
      return 0.0f;
    }
    
    // Augmented Lagrangian energy: E = 0.5 * rho * C^2 + lambda * C
    physx::PxReal rho = header.rho;
    physx::PxReal lambda = header.lambda;
    
    return 0.5f * rho * violation * violation + lambda * violation;
  }

  /**
   * @brief Compute energy gradient w.r.t. body positions and rotations
   */
  PX_FORCE_INLINE void computeEnergyGradient(
      const physx::PxVec3 &posA,
      const physx::PxQuat &rotA,
      const physx::PxVec3 &posB,
      const physx::PxQuat &rotB,
      physx::PxVec3 &gradPosA,
      physx::PxVec3 &gradRotA,
      physx::PxVec3 &gradPosB,
      physx::PxVec3 &gradRotB) const {
    
    physx::PxReal violation = computeViolation(posA, rotA, posB, rotB);
    
    // Constraint force: F = rho * C + lambda
    physx::PxReal force = header.rho * violation + header.lambda;
    
    // For inequality constraint, only apply if active
    if (violation >= 0.0f && header.lambda <= 0.0f) {
      gradPosA = physx::PxVec3(0.0f);
      gradRotA = physx::PxVec3(0.0f);
      gradPosB = physx::PxVec3(0.0f);
      gradRotB = physx::PxVec3(0.0f);
      return;
    }
    
    // Gradient = force * Jacobian
    computeGradient(rotA, rotB, gradPosA, gradPosB, gradRotA, gradRotB);
    gradPosA *= force;
    gradRotA *= force;
    gradPosB *= force;
    gradRotB *= force;
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief Weld joint constraint for AVBD solver (Unified Fixed/Weld Joint)
 *
 * This is the primary constraint for locking all 6 DOF between two bodies.
 * Optimized for both:
 *   1. Pre-authored fixed joints (scene setup)
 *   2. Runtime attachment scenarios ("Ultrahand" style)
 *
 * Key features:
 *   - Handles large mass ratios gracefully (small object on large object)
 *   - Uses mass-weighted corrections for stability
 *   - Supports "breakable" mode with force threshold
 *   - Optimized for frequent creation/destruction
 *   - Pre-computed effective mass for faster solving
 *
 * Constraint functions:
 *   C_pos(x) = pA + R_A * anchorA - pB - R_B * anchorB = 0
 *   C_rot(x) = relative rotation error = 0
 *
 * Note: eJOINT_FIXED and eJOINT_WELD both map to this constraint type.
 *       Use eJOINT_FIXED for pre-authored joints, eJOINT_WELD for runtime creation.
 */
struct PX_ALIGN_PREFIX(16) AvbdWeldJointConstraint {
  AvbdConstraintHeader header;

  //-------------------------------------------------------------------------
  // Joint anchors (in local coordinates of each body)
  //-------------------------------------------------------------------------

  physx::PxVec3 anchorA;     //!< Weld point on body A (local space)
  physx::PxReal massRatioA;  //!< Cached mass contribution ratio for body A

  physx::PxVec3 anchorB;     //!< Weld point on body B (local space)
  physx::PxReal massRatioB;  //!< Cached mass contribution ratio for body B

  //-------------------------------------------------------------------------
  // Relative orientation at weld time
  //-------------------------------------------------------------------------

  physx::PxQuat relativeRotation;  //!< Target relative rotation (B relative to A)

  //-------------------------------------------------------------------------
  // Lagrangian multipliers (6 DOF)
  //-------------------------------------------------------------------------

  physx::PxVec3 lambdaPosition;  //!< Position constraint multipliers
  physx::PxReal breakForce;      //!< Force threshold for breaking (0 = unbreakable)

  physx::PxVec3 lambdaRotation;  //!< Rotation constraint multipliers
  physx::PxReal breakTorque;     //!< Torque threshold for breaking (0 = unbreakable)

  //-------------------------------------------------------------------------
  // Pre-computed effective mass (for faster solving)
  //-------------------------------------------------------------------------

  physx::PxReal effectiveLinearMass;   //!< Combined effective mass for position
  physx::PxReal effectiveAngularMass;  //!< Combined effective mass for rotation
  physx::PxReal accumulatedForce;      //!< Accumulated constraint force magnitude (for break check)
  physx::PxReal accumulatedTorque;     //!< Accumulated constraint torque magnitude (for break check)

  //-------------------------------------------------------------------------
  // Flags
  //-------------------------------------------------------------------------

  physx::PxU16 isBreakable;    //!< Whether this weld can break
  physx::PxU16 isBroken;       //!< Whether this weld has broken
  physx::PxU16 isNewlyCreated; //!< Flag for first-frame handling
  physx::PxU16 padding0;

  //-------------------------------------------------------------------------
  // Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Initialize as a fixed joint (pre-authored, equal mass distribution)
   * 
   * Use this for scene-authored fixed joints where mass ratios are not critical.
   */
  PX_FORCE_INLINE void initFixed(
      const physx::PxVec3& localAnchorA,
      const physx::PxVec3& localAnchorB,
      const physx::PxQuat& localFrameA = physx::PxQuat(physx::PxIdentity),
      const physx::PxQuat& localFrameB = physx::PxQuat(physx::PxIdentity)) {
    
    header.type = AvbdConstraintType::eJOINT_FIXED;
    header.compliance = 0.0f;
    header.rho = 1e4f;
    
    anchorA = localAnchorA;
    anchorB = localAnchorB;
    
    // For fixed joints, use target relative rotation from local frames
    relativeRotation = localFrameA.getConjugate() * localFrameB;
    
    // Equal mass distribution for pre-authored joints
    massRatioA = 0.5f;
    massRatioB = 0.5f;
    effectiveLinearMass = 0.0f;
    effectiveAngularMass = 1.0f;
    
    lambdaPosition = physx::PxVec3(0.0f);
    lambdaRotation = physx::PxVec3(0.0f);
    
    // Fixed joints are not breakable by default
    breakForce = 0.0f;
    breakTorque = 0.0f;
    isBreakable = 0;
    isBroken = 0;
    isNewlyCreated = 0;
    accumulatedForce = 0.0f;
    accumulatedTorque = 0.0f;
    padding0 = 0;
  }

  /**
   * @brief Initialize weld constraint between two bodies at runtime
   * 
   * Use this for "Ultrahand" style runtime attachment.
   * 
   * @param posA World position of body A
   * @param rotA World rotation of body A
   * @param invMassA Inverse mass of body A
   * @param posB World position of body B
   * @param rotB World rotation of body B
   * @param invMassB Inverse mass of body B
   * @param worldWeldPoint World-space point where bodies are welded
   */
  PX_FORCE_INLINE void initializeWeld(
      const physx::PxVec3& posA, const physx::PxQuat& rotA, physx::PxReal invMassA,
      const physx::PxVec3& posB, const physx::PxQuat& rotB, physx::PxReal invMassB,
      const physx::PxVec3& worldWeldPoint) {
    
    header.type = AvbdConstraintType::eJOINT_WELD;
    header.compliance = 0.0f;  // Hard constraint
    header.rho = 1e5f;         // Higher rho for stiffer welds
    
    // Convert world weld point to local coordinates
    anchorA = rotA.rotateInv(worldWeldPoint - posA);
    anchorB = rotB.rotateInv(worldWeldPoint - posB);
    
    // Store relative rotation at weld time
    relativeRotation = rotA.getConjugate() * rotB;
    
    // Compute mass ratios for stable correction distribution
    // The heavier object moves less
    physx::PxReal totalInvMass = invMassA + invMassB;
    if (totalInvMass > 1e-10f) {
      massRatioA = invMassA / totalInvMass;
      massRatioB = invMassB / totalInvMass;
    } else {
      massRatioA = 0.5f;
      massRatioB = 0.5f;
    }
    
    // Pre-compute effective mass
    if (totalInvMass > 1e-10f) {
      effectiveLinearMass = 1.0f / totalInvMass;
    } else {
      effectiveLinearMass = 0.0f;
    }
    effectiveAngularMass = 1.0f;  // Simplified, could be computed from inertias
    
    // Initialize multipliers
    lambdaPosition = physx::PxVec3(0.0f);
    lambdaRotation = physx::PxVec3(0.0f);
    
    // Default: unbreakable
    breakForce = 0.0f;
    breakTorque = 0.0f;
    isBreakable = 0;
    isBroken = 0;
    isNewlyCreated = 1;
    
    accumulatedForce = 0.0f;
    accumulatedTorque = 0.0f;
    padding0 = 0;
  }

  /**
   * @brief Set breakable parameters
   */
  PX_FORCE_INLINE void setBreakable(physx::PxReal maxForce, physx::PxReal maxTorque) {
    breakForce = maxForce;
    breakTorque = maxTorque;
    isBreakable = (maxForce > 0.0f || maxTorque > 0.0f) ? 1 : 0;
  }

  /**
   * @brief Compute position constraint violation (3 components)
   */
  PX_FORCE_INLINE physx::PxVec3 computePositionViolation(
      const physx::PxVec3& posA, const physx::PxQuat& rotA,
      const physx::PxVec3& posB, const physx::PxQuat& rotB) const {
    physx::PxVec3 worldAnchorA = posA + rotA.rotate(anchorA);
    physx::PxVec3 worldAnchorB = posB + rotB.rotate(anchorB);
    return worldAnchorA - worldAnchorB;
  }

  /**
   * @brief Compute rotation constraint violation (3 components as angular error)
   * 
   * We want: rotA.inverse() * rotB == relativeRotation
   * Error: rotA * relativeRotation * rotB.inverse()
   */
  PX_FORCE_INLINE physx::PxVec3 computeRotationViolation(
      const physx::PxQuat& rotA, const physx::PxQuat& rotB) const {
    
    // Current relative rotation
    physx::PxQuat currentRelRot = rotA.getConjugate() * rotB;
    
    // Error quaternion
    physx::PxQuat errorQ = currentRelRot * relativeRotation.getConjugate();
    if (errorQ.w < 0.0f) {
      errorQ = -errorQ;  // Shortest path
    }
    
    // Convert to axis-angle (small angle approximation)
    return physx::PxVec3(errorQ.x, errorQ.y, errorQ.z) * 2.0f;
  }

  /**
   * @brief Compute position corrections with mass-weighted distribution
   * 
   * This is the key optimization for "Ultrahand" scenarios:
   * - Light objects move more, heavy objects move less
   * - Prevents small attached objects from destabilizing large structures
   */
  PX_FORCE_INLINE void computeMassWeightedCorrections(
      const physx::PxVec3& violation,
      physx::PxVec3& correctionA,
      physx::PxVec3& correctionB) const {
    
    // Body A gets correction proportional to its inverse mass ratio
    // (heavier bodies have lower inverse mass, so they move less)
    correctionA = -violation * massRatioA;
    correctionB = violation * massRatioB;
  }

  /**
   * @brief Check if weld should break based on accumulated forces
   * @return true if weld is broken
   */
  PX_FORCE_INLINE bool checkBreak(physx::PxReal forceThisFrame, physx::PxReal torqueThisFrame) {
    if (!isBreakable || isBroken) {
      return isBroken != 0;
    }
    
    // Accumulate forces (with some smoothing)
    accumulatedForce = accumulatedForce * 0.9f + forceThisFrame * 0.1f;
    accumulatedTorque = accumulatedTorque * 0.9f + torqueThisFrame * 0.1f;
    
    // Check break threshold
    if ((breakForce > 0.0f && accumulatedForce > breakForce) ||
        (breakTorque > 0.0f && accumulatedTorque > breakTorque)) {
      isBroken = 1;
      return true;
    }
    
    return false;
  }

  /**
   * @brief Compute total constraint energy
   */
  PX_FORCE_INLINE physx::PxReal
  computeEnergy(const physx::PxVec3 &posViolation,
                const physx::PxVec3 &rotViolation) const {
    physx::PxReal invCompliance =
        (header.compliance > 0.0f) ? (1.0f / header.compliance) : header.rho;

    // Quadratic penalty + Lagrangian term
    physx::PxReal posEnergy =
        0.5f * invCompliance * posViolation.magnitudeSquared() +
        lambdaPosition.dot(posViolation);
    physx::PxReal rotEnergy =
        0.5f * invCompliance * rotViolation.magnitudeSquared() +
        lambdaRotation.dot(rotViolation);

    return posEnergy + rotEnergy;
  }

  /**
   * @brief Compute augmented Lagrangian energy
   */
  PX_FORCE_INLINE physx::PxReal computeAugmentedLagrangianEnergy(
      const physx::PxVec3& posA, const physx::PxQuat& rotA,
      const physx::PxVec3& posB, const physx::PxQuat& rotB) const {
    
    if (isBroken) return 0.0f;
    
    physx::PxVec3 posViolation = computePositionViolation(posA, rotA, posB, rotB);
    physx::PxVec3 rotViolation = computeRotationViolation(rotA, rotB);
    
    return computeEnergy(posViolation, rotViolation);
  }

  /**
   * @brief Initialize default values
   */
  PX_FORCE_INLINE void initDefaults() {
    header.type = AvbdConstraintType::eJOINT_FIXED;
    header.compliance = 0.0f;
    header.rho = 1e4f;
    anchorA = physx::PxVec3(0.0f);
    anchorB = physx::PxVec3(0.0f);
    massRatioA = 0.5f;
    massRatioB = 0.5f;
    relativeRotation = physx::PxQuat(physx::PxIdentity);
    lambdaPosition = physx::PxVec3(0.0f);
    lambdaRotation = physx::PxVec3(0.0f);
    breakForce = 0.0f;
    breakTorque = 0.0f;
    effectiveLinearMass = 0.0f;
    effectiveAngularMass = 1.0f;
    accumulatedForce = 0.0f;
    accumulatedTorque = 0.0f;
    isBreakable = 0;
    isBroken = 0;
    isNewlyCreated = 0;
    padding0 = 0;
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief Alias for backward compatibility
 * 
 * AvbdFixedJointConstraint is now unified with AvbdWeldJointConstraint.
 * Both eJOINT_FIXED and eJOINT_WELD use the same underlying structure.
 * 
 * For new code, prefer using AvbdWeldJointConstraint directly.
 */
typedef AvbdWeldJointConstraint AvbdFixedJointConstraint;

/**
 * @brief Spherical (Ball) joint constraint for AVBD solver
 *
 * Constrains two anchor points to be coincident:
 *   C(x) = pA + R_A * anchorA - pB - R_B * anchorB = 0
 *
 * Optionally supports cone angle limit to restrict relative rotation.
 */
struct PX_ALIGN_PREFIX(16) AvbdSphericalJointConstraint {
  AvbdConstraintHeader header;

  //-------------------------------------------------------------------------
  // Joint anchors
  //-------------------------------------------------------------------------

  physx::PxVec3 anchorA;        //!< Anchor point on body A (local space)
  physx::PxReal coneAngleLimit; //!< Cone angle limit in radians (0 = no limit)

  physx::PxVec3 anchorB;    //!< Anchor point on body B (local space)
  physx::PxReal coneLambda; //!< Lambda for cone constraint

  physx::PxVec3 lambda; //!< Lagrangian multipliers (3 DOF position)
  physx::PxReal padding0;

  //-------------------------------------------------------------------------
  // Cone limit axis (local frame on body A)
  //-------------------------------------------------------------------------

  physx::PxVec3 coneAxisA;   //!< Cone axis on body A (local space)
  physx::PxU16 hasConeLimit; //!< Whether cone limit is active
  physx::PxU16 padding1;

  //-------------------------------------------------------------------------
  // Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Compute position constraint violation (3 components)
   */
  PX_FORCE_INLINE physx::PxVec3
  computeViolation(const physx::PxVec3 &posA, const physx::PxQuat &rotA,
                   const physx::PxVec3 &posB,
                   const physx::PxQuat &rotB) const {
    physx::PxVec3 worldAnchorA = posA + rotA.rotate(anchorA);
    physx::PxVec3 worldAnchorB = posB + rotB.rotate(anchorB);
    return worldAnchorA - worldAnchorB;
  }

  /**
   * @brief Compute cone angle violation
   * @return Angle violation in radians (positive = exceeded limit)
   */
  PX_FORCE_INLINE physx::PxReal
  computeConeViolation(const physx::PxQuat &rotA,
                       const physx::PxQuat &rotB) const {
    if (!hasConeLimit || coneAngleLimit <= 0.0f)
      return 0.0f;

    // Get world space cone axes
    physx::PxVec3 worldAxisA = rotA.rotate(coneAxisA);
    physx::PxVec3 worldAxisB =
        rotB.rotate(coneAxisA); // Use same reference axis

    // Compute angle between axes
    physx::PxReal dotProduct = worldAxisA.dot(worldAxisB);
    dotProduct = physx::PxClamp(dotProduct, -1.0f, 1.0f);
    physx::PxReal angle = physx::PxAcos(dotProduct);

    return angle - coneAngleLimit; // Positive if exceeded
  }

  /**
   * @brief Initialize default values
   */
  PX_FORCE_INLINE void initDefaults() {
    header.type = AvbdConstraintType::eJOINT_SPHERICAL;
    header.compliance = 0.0f; // Hard constraint by default
    header.rho = 1e4f;
    anchorA = physx::PxVec3(0.0f);
    anchorB = physx::PxVec3(0.0f);
    lambda = physx::PxVec3(0.0f);
    coneAngleLimit = 0.0f;
    coneLambda = 0.0f;
    coneAxisA = physx::PxVec3(0.0f, 1.0f, 0.0f);
    hasConeLimit = 0;
    padding0 = 0.0f;
    padding1 = 0;
  }

  /**
   * @brief Compute augmented Lagrangian energy
   * E = 0.5 * rho * ||C||^2 + lambda^T * C
   */
  PX_FORCE_INLINE physx::PxReal computeAugmentedLagrangianEnergy(
      const physx::PxVec3 &posA,
      const physx::PxQuat &rotA,
      const physx::PxVec3 &posB,
      const physx::PxQuat &rotB) const {
    
    physx::PxVec3 violation = computeViolation(posA, rotA, posB, rotB);
    physx::PxReal rho = header.rho;
    
    physx::PxReal energy = 0.0f;
    for (int i = 0; i < 3; ++i) {
      energy += 0.5f * rho * violation[i] * violation[i] + lambda[i] * violation[i];
    }
    
    return energy;
  }

  /**
   * @brief Compute energy gradient w.r.t. body positions and rotations
   */
  PX_FORCE_INLINE void computeEnergyGradient(
      const physx::PxVec3 &posA,
      const physx::PxQuat &rotA,
      const physx::PxVec3 &posB,
      const physx::PxQuat &rotB,
      physx::PxVec3 &gradPosA,
      physx::PxVec3 &gradRotA,
      physx::PxVec3 &gradPosB,
      physx::PxVec3 &gradRotB) const {
    
    physx::PxVec3 violation = computeViolation(posA, rotA, posB, rotB);
    physx::PxReal rho = header.rho;
    
    // Gradient = (rho * C + lambda) * Jacobian
    // For spherical joint, Jacobian is identity for position
    physx::PxVec3 force(0.0f);
    for (int i = 0; i < 3; ++i) {
      force[i] = rho * violation[i] + lambda[i];
    }
    
    gradPosA = force;
    gradPosB = -force;
    
    // Rotation gradient: d(R*r)/dq
    physx::PxVec3 rA = rotA.rotate(anchorA);
    physx::PxVec3 rB = rotB.rotate(anchorB);
    
    gradRotA = rA.cross(force);
    gradRotB = -rB.cross(force);
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief Revolute (Hinge) joint constraint for AVBD solver
 *
 * Constrains two bodies to rotate around a single shared axis:
 *   - Position constraint (3 DOF): anchor points must coincide
 *   - Axis alignment (2 DOF): rotation axes must align
 *   - Free axis: 1 rotation DOF around the shared axis
 *
 * Total: 5 constraints (3 position + 2 axis alignment)
 */
struct PX_ALIGN_PREFIX(16) AvbdRevoluteJointConstraint {
  AvbdConstraintHeader header;

  //-------------------------------------------------------------------------
  // Joint geometry
  //-------------------------------------------------------------------------

  physx::PxVec3 anchorA;         //!< Anchor point on body A (local space)
  physx::PxReal angleLimitLower; //!< Lower angle limit (radians)

  physx::PxVec3 anchorB;         //!< Anchor point on body B (local space)
  physx::PxReal angleLimitUpper; //!< Upper angle limit (radians)

  physx::PxVec3 axisA; //!< Rotation axis on body A (local space, normalized)
  physx::PxReal motorTargetVelocity; //!< Motor target angular velocity

  physx::PxVec3 axisB; //!< Rotation axis on body B (local space, normalized)
  physx::PxReal motorMaxForce; //!< Motor maximum force

  //-------------------------------------------------------------------------
  // Reference frames for angle measurement
  //-------------------------------------------------------------------------

  physx::PxVec3
      refAxisA; //!< Reference axis on body A (perpendicular to axisA)
  physx::PxReal padding0;

  physx::PxVec3
      refAxisB; //!< Reference axis on body B (perpendicular to axisB)
  physx::PxReal padding1;

  //-------------------------------------------------------------------------
  // Lagrangian multipliers
  //-------------------------------------------------------------------------

  physx::PxVec3 lambdaPosition;   //!< Position constraint multipliers (3 DOF)
  physx::PxReal lambdaAngleLimit; //!< Angle limit multiplier

  physx::PxVec3 lambdaAxisAlign; //!< Axis alignment multipliers (2 DOF stored
                                  //!< as Vec3, z unused)
  physx::PxReal padding2;

  //-------------------------------------------------------------------------
  // Flags
  //-------------------------------------------------------------------------

  physx::PxU16 hasAngleLimit; //!< Whether angle limits are active
  physx::PxU16 motorEnabled;  //!< Whether motor is enabled
  physx::PxU16 padding3;
  physx::PxU16 padding4;

  //-------------------------------------------------------------------------
  // Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Compute position constraint violation (3 components)
   */
  PX_FORCE_INLINE physx::PxVec3 computePositionViolation(
      const physx::PxVec3 &posA, const physx::PxQuat &rotA,
      const physx::PxVec3 &posB, const physx::PxQuat &rotB) const {
    physx::PxVec3 worldAnchorA = posA + rotA.rotate(anchorA);
    physx::PxVec3 worldAnchorB = posB + rotB.rotate(anchorB);
    return worldAnchorA - worldAnchorB;
  }

  /**
   * @brief Compute axis alignment violation (2 components)
   *
   * The two axes should be parallel. We measure the cross product
   * and project onto two perpendicular directions.
   */
  PX_FORCE_INLINE physx::PxVec3
  computeAxisViolation(const physx::PxQuat &rotA,
                       const physx::PxQuat &rotB) const {
    physx::PxVec3 worldAxisA = rotA.rotate(axisA);
    physx::PxVec3 worldAxisB = rotB.rotate(axisB);

    // Cross product gives axis alignment error
    // When aligned, cross product is zero
    return worldAxisA.cross(worldAxisB);
  }

  /**
   * @brief Compute current joint angle
   */
  PX_FORCE_INLINE physx::PxReal
  computeAngle(const physx::PxQuat &rotA, const physx::PxQuat &rotB) const {
    // Transform reference axes to world space
    physx::PxVec3 worldRefA = rotA.rotate(refAxisA);
    physx::PxVec3 worldRefB = rotB.rotate(refAxisB);
    physx::PxVec3 worldAxisA = rotA.rotate(axisA);

    // Project refB onto plane perpendicular to axis
    physx::PxVec3 projB = worldRefB - worldAxisA * worldRefB.dot(worldAxisA);
    projB.normalize();

    // Compute angle using atan2
    physx::PxReal cosAngle = worldRefA.dot(projB);
    physx::PxReal sinAngle = worldAxisA.dot(worldRefA.cross(projB));

    return physx::PxAtan2(sinAngle, cosAngle);
  }

  /**
   * @brief Compute angle limit violation
   * @return Positive if exceeded lower limit, negative if exceeded upper limit
   */
  PX_FORCE_INLINE physx::PxReal
  computeAngleLimitViolation(const physx::PxQuat &rotA,
                             const physx::PxQuat &rotB) const {
    if (!hasAngleLimit)
      return 0.0f;

    physx::PxReal angle = computeAngle(rotA, rotB);

    if (angle < angleLimitLower)
      return angle - angleLimitLower; // Negative
    if (angle > angleLimitUpper)
      return angle - angleLimitUpper; // Positive

    return 0.0f;
  }

  /**
   * @brief Initialize default values
   */
  PX_FORCE_INLINE void initDefaults() {
    header.type = AvbdConstraintType::eJOINT_REVOLUTE;
    header.compliance = 0.0f;
    header.rho = 1e4f;
    anchorA = physx::PxVec3(0.0f);
    anchorB = physx::PxVec3(0.0f);
    axisA = physx::PxVec3(0.0f, 0.0f, 1.0f); // Default Z axis
    axisB = physx::PxVec3(0.0f, 0.0f, 1.0f);
    refAxisA = physx::PxVec3(1.0f, 0.0f, 0.0f); // Default X axis
    refAxisB = physx::PxVec3(1.0f, 0.0f, 0.0f);
    angleLimitLower = -physx::PxPi;
    angleLimitUpper = physx::PxPi;
    motorTargetVelocity = 0.0f;
    motorMaxForce = 0.0f;
    lambdaPosition = physx::PxVec3(0.0f);
    lambdaAxisAlign = physx::PxVec3(0.0f);
    lambdaAngleLimit = 0.0f;
    hasAngleLimit = 0;
    motorEnabled = 0;
    padding0 = 0.0f;
    padding1 = 0.0f;
    padding2 = 0;
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief Prismatic (Slider) joint constraint for AVBD solver
 *
 * Constrains two bodies to move along a single shared axis:
 *   - Position constraint (2 DOF): perpendicular to slide axis
 *   - Rotation constraint (3 DOF): orientations locked
 *   - Free axis: 1 translation DOF along the slide axis
 *
 * The constraint energy is:
 *   E = 0.5/α * ||C||² + λᵀC
 *
 * Where C includes:
 *   - Perpendicular position error (2D)
 *   - Relative rotation error (quaternion difference)
 *   - Optional limit constraint (when slide is out of range)
 */
struct PX_ALIGN_PREFIX(16) AvbdPrismaticJointConstraint {
  AvbdConstraintHeader header;

  // Anchor points in local coordinates
  physx::PxVec3 anchorA;
  physx::PxVec3 anchorB;

  // Slide axis in local coordinates (body A frame)
  physx::PxVec3 axisA;

  // Local frame orientations for full rotation lock
  physx::PxQuat localFrameA;
  physx::PxQuat localFrameB;

  // Slide limits
  physx::PxReal limitLower;
  physx::PxReal limitUpper;

  // Motor
  physx::PxReal motorTargetVelocity;
  physx::PxReal motorMaxForce;

  // Lagrange multipliers
  physx::PxVec3 lambdaPosition; // Perpendicular position (2D -> stored in x,y)
  physx::PxVec3 lambdaRotation; // Rotation alignment (3D)
  physx::PxReal lambdaLimit;    // Slide limit

  // Flags
  physx::PxU8 hasLimit;
  physx::PxU8 motorEnabled;
  physx::PxU16 padding0;
  physx::PxReal padding1;

  /**
   * @brief Compute world space slide axis
   */
  PX_FORCE_INLINE physx::PxVec3
  getWorldAxis(const physx::PxQuat &rotA) const {
    return rotA.rotate(axisA);
  }

  /**
   * @brief Compute current slide position along axis
   */
  PX_FORCE_INLINE physx::PxReal
  computeSlidePosition(const physx::PxVec3 &posA, const physx::PxQuat &rotA,
                       const physx::PxVec3 &posB,
                       const physx::PxQuat &rotB) const {
    physx::PxVec3 worldAnchorA = posA + rotA.rotate(anchorA);
    physx::PxVec3 worldAnchorB = posB + rotB.rotate(anchorB);
    physx::PxVec3 worldAxis = getWorldAxis(rotA);
    physx::PxVec3 diff = worldAnchorB - worldAnchorA;
    return diff.dot(worldAxis);
  }

  /**
   * @brief Compute limit violation
   * @return Positive = exceeded upper, Negative = below lower, 0 = within
   * limits
   */
  PX_FORCE_INLINE physx::PxReal
  computeLimitViolation(physx::PxReal slidePos) const {
    if (!hasLimit)
      return 0.0f;
    if (slidePos < limitLower)
      return slidePos - limitLower;
    if (slidePos > limitUpper)
      return slidePos - limitUpper;
    return 0.0f;
  }

  /**
   * @brief Compute rotation constraint violation (3 components as angular
   * error)
   */
  PX_FORCE_INLINE physx::PxVec3
  computeRotationViolation(const physx::PxQuat &rotA,
                           const physx::PxQuat &rotB) const {
    physx::PxQuat worldFrameA = rotA * localFrameA;
    physx::PxQuat worldFrameB = rotB * localFrameB;

    // Compute relative rotation error
    physx::PxQuat errorQ = worldFrameB.getConjugate() * worldFrameA;
    if (errorQ.w < 0.0f)
      errorQ = -errorQ;

    // Return axis-angle representation (small angle approximation)
    return physx::PxVec3(errorQ.x, errorQ.y, errorQ.z) * 2.0f;
  }

  /**
   * @brief Initialize default values
   */
  PX_FORCE_INLINE void initDefaults() {
    header.type = AvbdConstraintType::eJOINT_PRISMATIC;
    header.compliance = 0.0f;
    header.rho = 1e4f;
    anchorA = physx::PxVec3(0.0f);
    anchorB = physx::PxVec3(0.0f);
    axisA = physx::PxVec3(1.0f, 0.0f, 0.0f); // Default X axis
    localFrameA = physx::PxQuat(0.0f, 0.0f, 0.0f, 1.0f);
    localFrameB = physx::PxQuat(0.0f, 0.0f, 0.0f, 1.0f);
    limitLower = -PX_MAX_F32;
    limitUpper = PX_MAX_F32;
    motorTargetVelocity = 0.0f;
    motorMaxForce = 0.0f;
    lambdaPosition = physx::PxVec3(0.0f);
    lambdaRotation = physx::PxVec3(0.0f);
    lambdaLimit = 0.0f;
    hasLimit = 0;
    motorEnabled = 0;
    padding0 = 0;
    padding1 = 0.0f;
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief D6 (Configurable) joint constraint for AVBD solver
 *
 * The D6 joint is the most flexible joint type, allowing independent control
 * of all 6 degrees of freedom. Each DOF can be configured as:
 *   - LOCKED: Constrained to zero
 *   - FREE: No constraint
 *   - LIMITED: Constrained within a range
 *
 * Linear DOFs (3): Translation along X, Y, Z axes
 * Angular DOFs (3): Rotation around X, Y, Z axes
 *
 * The constraint energy is:
 *   E = 0.5/α * ||C||² + λᵀC
 *
 * Where C includes:
 *   - Locked linear DOFs (position error)
 *   - Locked angular DOFs (rotation error)
 *   - Limited linear DOFs (limit violation)
 *   - Limited angular DOFs (limit violation)
 *   - Spring forces (optional)
 */
struct PX_ALIGN_PREFIX(16) AvbdD6JointConstraint {
  AvbdConstraintHeader header;

  //-------------------------------------------------------------------------
  // Joint geometry
  //-------------------------------------------------------------------------

  physx::PxVec3 anchorA; //!< Anchor point on body A (local space)
  physx::PxReal padding0;

  physx::PxVec3 anchorB; //!< Anchor point on body B (local space)
  physx::PxReal padding1;

  physx::PxQuat localFrameA; //!< Local frame on body A
  physx::PxQuat localFrameB; //!< Local frame on body B

  //-------------------------------------------------------------------------
  // Linear DOF configuration (3 axes)
  //-------------------------------------------------------------------------

  physx::PxVec3 linearLimitLower; //!< Lower limits for X, Y, Z translation
  physx::PxVec3 linearLimitUpper; //!< Upper limits for X, Y, Z translation

  physx::PxVec3 linearStiffness;  //!< Spring stiffness for X, Y, Z
  physx::PxVec3 linearDamping;    //!< Spring damping for X, Y, Z

  //-------------------------------------------------------------------------
  // Angular DOF configuration (3 axes)
  //-------------------------------------------------------------------------

  physx::PxVec3 angularLimitLower; //!< Lower limits for X, Y, Z rotation (radians)
  physx::PxVec3 angularLimitUpper; //!< Upper limits for X, Y, Z rotation (radians)

  physx::PxVec3 angularStiffness;  //!< Spring stiffness for X, Y, Z rotation
  physx::PxVec3 angularDamping;    //!< Spring damping for X, Y, Z rotation

  //-------------------------------------------------------------------------
  // Drive (motor) configuration
  //-------------------------------------------------------------------------

  physx::PxVec3 driveLinearVelocity;  //!< Target linear velocity (X, Y, Z)
  physx::PxVec3 driveLinearForce;     //!< Max linear drive force (X, Y, Z)

  physx::PxVec3 driveAngularVelocity; //!< Target angular velocity (X, Y, Z)
  physx::PxVec3 driveAngularForce;    //!< Max angular drive force (X, Y, Z)

  //-------------------------------------------------------------------------
  // Lagrangian multipliers (6 DOF)
  //-------------------------------------------------------------------------

  physx::PxVec3 lambdaLinear;  //!< Linear constraint multipliers (X, Y, Z)
  physx::PxVec3 lambdaAngular; //!< Angular constraint multipliers (X, Y, Z)

  //-------------------------------------------------------------------------
  // DOF motion flags (bitmask)
  //-------------------------------------------------------------------------

  physx::PxU32 linearMotion;  //!< Linear motion flags (3 bits: 0=LOCKED, 1=LIMITED, 2=FREE)
  physx::PxU32 angularMotion; //!< Angular motion flags (3 bits: 0=LOCKED, 1=LIMITED, 2=FREE)

  physx::PxU32 driveFlags;    //!< Drive enable flags (6 bits: bit 0-2 linear, bit 3-5 angular)
  physx::PxU32 padding2;

  //-------------------------------------------------------------------------
  // Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Get linear motion type for an axis
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return Motion type: 0=LOCKED, 1=LIMITED, 2=FREE
   */
  PX_FORCE_INLINE physx::PxU32 getLinearMotion(physx::PxU32 axis) const {
    return (linearMotion >> (axis * 2)) & 0x3;
  }

  /**
   * @brief Get angular motion type for an axis
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return Motion type: 0=LOCKED, 1=LIMITED, 2=FREE
   */
  PX_FORCE_INLINE physx::PxU32 getAngularMotion(physx::PxU32 axis) const {
    return (angularMotion >> (axis * 2)) & 0x3;
  }

  /**
   * @brief Check if linear drive is enabled for an axis
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return True if drive is enabled
   */
  PX_FORCE_INLINE bool isLinearDriveEnabled(physx::PxU32 axis) const {
    return (driveFlags & (1 << axis)) != 0;
  }

  /**
   * @brief Check if angular drive is enabled for an axis
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return True if drive is enabled
   */
  PX_FORCE_INLINE bool isAngularDriveEnabled(physx::PxU32 axis) const {
    return (driveFlags & (1 << (axis + 3))) != 0;
  }

  /**
   * @brief Compute linear position error for an axis
   * @param posA World position of body A
   * @param rotA World rotation of body A
   * @param posB World position of body B
   * @param rotB World rotation of body B
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return Position error along the axis
   */
  PX_FORCE_INLINE physx::PxReal computeLinearError(
      const physx::PxVec3 &posA, const physx::PxQuat &rotA,
      const physx::PxVec3 &posB, const physx::PxQuat &rotB,
      physx::PxU32 axis) const {
    physx::PxVec3 worldAnchorA = posA + rotA.rotate(anchorA);
    physx::PxVec3 worldAnchorB = posB + rotB.rotate(anchorB);
    physx::PxVec3 diff = worldAnchorB - worldAnchorA;

    // Transform to local frame A
    physx::PxVec3 localDiff = rotA.rotateInv(diff);

    return localDiff[axis];
  }

  /**
   * @brief Compute angular error for an axis
   * @param rotA World rotation of body A
   * @param rotB World rotation of body B
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return Angular error around the axis (radians)
   */
  PX_FORCE_INLINE physx::PxReal computeAngularError(
      const physx::PxQuat &rotA, const physx::PxQuat &rotB,
      physx::PxU32 axis) const {
    physx::PxQuat worldFrameA = rotA * localFrameA;
    physx::PxQuat worldFrameB = rotB * localFrameB;

    // Compute relative rotation
    physx::PxQuat relRot = worldFrameB.getConjugate() * worldFrameA;
    if (relRot.w < 0.0f)
      relRot = -relRot;

    // Convert to axis-angle
    physx::PxReal angle = 2.0f * physx::PxAcos(physx::PxClamp(relRot.w, -1.0f, 1.0f));
    if (angle < 1e-6f)
      return 0.0f;

    physx::PxVec3 axisVec(relRot.x, relRot.y, relRot.z);
    axisVec.normalize();

    // Project onto the requested axis in local frame A
    physx::PxVec3 localAxis(0.0f);
    localAxis[axis] = 1.0f;
    physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);

    return angle * axisVec.dot(worldAxis);
  }

  /**
   * @brief Compute linear limit violation for an axis
   * @param error Current linear error
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return Limit violation (0 if within limits)
   */
  PX_FORCE_INLINE physx::PxReal computeLinearLimitViolation(
      physx::PxReal error, physx::PxU32 axis) const {
    if (error < linearLimitLower[axis])
      return error - linearLimitLower[axis];
    if (error > linearLimitUpper[axis])
      return error - linearLimitUpper[axis];
    return 0.0f;
  }

  /**
   * @brief Compute angular limit violation for an axis
   * @param error Current angular error
   * @param axis Axis index (0=X, 1=Y, 2=Z)
   * @return Limit violation (0 if within limits)
   */
  PX_FORCE_INLINE physx::PxReal computeAngularLimitViolation(
      physx::PxReal error, physx::PxU32 axis) const {
    if (error < angularLimitLower[axis])
      return error - angularLimitLower[axis];
    if (error > angularLimitUpper[axis])
      return error - angularLimitUpper[axis];
    return 0.0f;
  }

  /**
   * @brief Initialize default values
   */
  PX_FORCE_INLINE void initDefaults() {
    header.type = AvbdConstraintType::eJOINT_D6;
    header.compliance = 0.0f;
    header.rho = 1e4f;
    anchorA = physx::PxVec3(0.0f);
    anchorB = physx::PxVec3(0.0f);
    localFrameA = physx::PxQuat(0.0f, 0.0f, 0.0f, 1.0f);
    localFrameB = physx::PxQuat(0.0f, 0.0f, 0.0f, 1.0f);
    linearLimitLower = physx::PxVec3(0.0f);
    linearLimitUpper = physx::PxVec3(0.0f);
    linearStiffness = physx::PxVec3(0.0f);
    linearDamping = physx::PxVec3(0.0f);
    angularLimitLower = physx::PxVec3(-physx::PxPi);
    angularLimitUpper = physx::PxVec3(physx::PxPi);
    angularStiffness = physx::PxVec3(0.0f);
    angularDamping = physx::PxVec3(0.0f);
    driveLinearVelocity = physx::PxVec3(0.0f);
    driveLinearForce = physx::PxVec3(0.0f);
    driveAngularVelocity = physx::PxVec3(0.0f);
    driveAngularForce = physx::PxVec3(0.0f);
    lambdaLinear = physx::PxVec3(0.0f);
    lambdaAngular = physx::PxVec3(0.0f);
    linearMotion = 0; // All locked by default
    angularMotion = 0; // All locked by default
    driveFlags = 0;
    padding0 = 0.0f;
    padding1 = 0.0f;
    padding2 = 0;
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief Constraint batch for SIMD processing
 *
 * Groups constraints of the same type for vectorized processing.
 */
struct AvbdConstraintBatch {
  AvbdConstraintType::Enum type; //!< Type of constraints in this batch
  physx::PxU32 startIndex;      //!< Start index in the constraint pool
  physx::PxU32 count;           //!< Number of constraints
  physx::PxU32 padding;
};

} // namespace Dy

} // namespace physx

#pragma warning(pop)

#endif // DY_AVBD_CONSTRAINT_H
