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

#ifndef DY_AVBD_TYPES_H
#define DY_AVBD_TYPES_H

#include "foundation/PxAllocator.h"
#include "foundation/PxMat33.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx {

/**
 * @brief AVBD numerical constants
 *
 * These constants define the numerical parameters used throughout the AVBD
 * solver. They are centralized here to avoid magic numbers in the code.
 */
namespace AvbdConstants {
// Position step size limits
static const PxReal AVBD_MAX_POSITION_STEP = 0.1f;

// Friction correction coefficient
static const PxReal AVBD_FRICTION_CORRECTION = 0.1f;

// Constraint damping
static const PxReal AVBD_CONSTRAINT_DAMPING = 0.1f;

// Numerical precision
static const PxReal AVBD_NUMERICAL_EPSILON = 1e-6f;

// Position error threshold (for early exit when constraint is satisfied)
static const PxReal AVBD_POSITION_ERROR_THRESHOLD = 1e-7f;

// Rotation error threshold (for early exit when constraint is satisfied)
static const PxReal AVBD_ROTATION_ERROR_THRESHOLD = 1e-6f;

// Angle limit violation threshold
static const PxReal AVBD_ANGLE_LIMIT_THRESHOLD = 1e-4f;

// Infinite mass threshold (for detecting static bodies)
static const PxReal AVBD_INFINITE_MASS_THRESHOLD = 1e-10f;

// Motor gain coefficient
static const PxReal AVBD_MOTOR_GAIN = 0.5f;

// Axis selection threshold (for building perpendicular basis)
static const PxReal AVBD_AXIS_SELECTION_THRESHOLD = 0.9f;

// Quaternion half factor (for converting angular velocity to quaternion)
static const PxReal AVBD_QUATERNION_HALF_FACTOR = 0.5f;

// Default penalty parameters
static const PxReal AVBD_DEFAULT_PENALTY_RHO_LOW = 1e5f; // Increased from 1e4
static const PxReal AVBD_DEFAULT_PENALTY_RHO_HIGH = 1e6f;

// Maximum penalty parameter
static const PxReal AVBD_MAX_PENALTY_RHO = 1e8f;

// Minimum penalty parameter
static const PxReal AVBD_MIN_PENALTY_RHO = 1e2f;

// Penalty parameter increase factor
static const PxReal AVBD_RHO_INCREASE_FACTOR = 2.0f;

// Penalty parameter decrease factor
static const PxReal AVBD_RHO_DECREASE_FACTOR = 0.5f;

// Violation threshold
static const PxReal AVBD_VIOLATION_THRESHOLD =
    1e-2f; // Increased from 1e-3 to skip near-satisfied constraints

// Maximum Lagrangian multiplier
static const PxReal AVBD_MAX_LAMBDA = 1e6f;

// LDLT decomposition threshold for singularity detection
static const PxReal AVBD_LDLT_SINGULAR_THRESHOLD = 1e-10f;

// Condition number threshold for ill-conditioned matrices
static const PxReal AVBD_CONDITION_NUMBER_THRESHOLD = 1e8f;

// Regularization coefficient for ill-conditioned matrices
static const PxReal AVBD_REGULARIZATION_COEFFICIENT = 1e-6f;

// Shared body-vs-static constants (kept aligned with avbd_standalone).
static const PxReal AVBD_PEN_SCALE_BODY_VS_STATIC = 2.0f;
static const PxReal AVBD_PEN_SCALE_DYN_DYN = 0.05f;
static const PxReal AVBD_CONTACT_BOOST_FRACTION = 0.005f;
static const PxU32 AVBD_MIN_INNER_ITERS_BODY_VS_STATIC = 16u;
// Surface-motion alias reject + friction ride caps (solve-loop contract Phase 4)
static const PxReal AVBD_SURFACE_STEP_ALIAS_M = 0.5f;
static const PxReal AVBD_SURFACE_VMESH_CAP = 8.0f;
static const PxReal AVBD_BODY_STATIC_FAST_IMPACT_SPEED = 60.0f;
static const PxReal AVBD_SHELL_FAST_IMPACT_SPEED = 60.0f;
// Fallback approach-speed gate (m/s) when scene bounceThreshold is unset.
// PhysX scene bounceThresholdVelocity is typically -2 (signed relative speed).
static const PxReal AVBD_BOUNCE_THRESHOLD = 2.0f;
// SupportClass fill: multi-corner heavy box disables mesh ride
static const PxU32 AVBD_SUPPORT_MULTI_CORNER_MIN = 4u;
static const PxReal AVBD_SUPPORT_MULTI_CORNER_MASS = 40.0f;
} // namespace AvbdConstants

/**
 * Body-static support classification used by friction and surface-motion policy.
 * Filled once per contact for friction/e=0 consumers; replaces ad-hoc mass ifs.
 */
struct AvbdSupportClass {
  enum Enum : physx::PxU8 {
    eUnset = 0,
    eRigidPlane = 1,
    eDeformableFewContact = 2,
    eDeformableMultiCorner = 3,
    eShell = 4,
  };
};

namespace Dy {

/**
 * @brief Configuration flags for deterministic simulation
 */
struct AvbdDeterminismFlags {
  enum Enum {
    eNONE = 0,
    eSORT_CONSTRAINTS =
        (1 << 0), //!< Sort constraints by body pair for consistent ordering
    eSORT_BODIES = (1 << 1),         //!< Sort bodies by ID before iteration
    eUSE_KAHAN_SUMMATION = (1 << 2), //!< Use Kahan summation for accumulation
    eFIXED_POINT_MATH =
        (1 << 3), //!< Use fixed-point math where possible (future)

    eDETERMINISTIC_DEFAULT =
        eSORT_CONSTRAINTS | eSORT_BODIES | eUSE_KAHAN_SUMMATION
  };
};

/**
 * @brief AVBD solver configuration parameters
 */
struct AvbdSolverConfig {
  /**
   * Scene length scale from PxTolerancesScale.
   *
   * Every solver threshold with length dimensions must be expressed relative
   * to this value so geometrically equivalent scenes do not select different
   * AVBD control paths merely because their units differ.
   */
  physx::PxReal lengthScale;

  //-------------------------------------------------------------------------
  // Iteration control
  //-------------------------------------------------------------------------

  physx::PxU32
      outerIterations; //!< Number of outer ALM iterations (typically 1-4)
  physx::PxU32 innerIterations; //!< Number of inner block descent iterations
                                //!< per outer (typically 2-8)

  //-------------------------------------------------------------------------
  // Augmented Lagrangian parameters
  //-------------------------------------------------------------------------

  physx::PxReal initialRho; //!< Initial penalty parameter for ALM
  physx::PxReal
      rhoScale;         //!< Scale factor for rho adaptation per outer iteration
  physx::PxReal maxRho; //!< Maximum penalty parameter

  //-------------------------------------------------------------------------
  // Compliance and damping
  //-------------------------------------------------------------------------

  physx::PxReal defaultCompliance; //!< Default compliance for soft constraints
  physx::PxReal contactCompliance; //!< Compliance for contact constraints
                                   //!< (usually 0 or very small)
  physx::PxReal jointCompliance;   //!< Default compliance for joint constraints
  physx::PxReal contactDamping; //!< XPBD-style damping for contact constraints
                                //!< (0-1, higher = more energy dissipation)
  physx::PxReal damping; //!< Step size damping for gradient descent (0-1)

  //-------------------------------------------------------------------------
  // Rotation dynamics
  //-------------------------------------------------------------------------

  physx::PxReal angularDamping;    //!< Angular velocity damping per frame (0-1,
                                   //!< default 0.98)
  physx::PxReal rotationThreshold; //!< Penetration threshold to trigger
                                   //!< rotation (meters, default 0.001)
  physx::PxReal angularScale;      //!< Scale factor for angular velocity from
                                   //!< torque (default 800)
  physx::PxReal angularContactScale; //!< Scale for angular correction from
                                     //!< contact normals (0-1, default 0.2).
                                     //!< Reduced scale prevents drift from
                                     //!< asymmetric contact patches while
                                     //!< maintaining rotational stiffness.
  physx::PxReal
      baumgarte; //!< Baumgarte position correction factor (0-1, default 0.2)
  physx::PxReal
      chebyshevRho; //!< Chebyshev semi-iterative spectral radius (0-1, default 0.92)
                    //!< Set to 0 to disable Chebyshev acceleration.
                    //!< Higher values give faster convergence but risk instability.

  //-------------------------------------------------------------------------
  // Convergence
  //-------------------------------------------------------------------------

  physx::PxReal
      positionTolerance; //!< Position error tolerance for early termination
  physx::PxReal velocityDamping; //!< Global velocity damping factor (0-1)

  /**
   * Scene bounce threshold (PhysX: typically negative, e.g. -2).
   * Restitution applies only when approach speed exceeds |bounceThresholdVelocity|.
   * Copied each step from Dy::Context::getBounceThreshold().
   */
  physx::PxReal bounceThresholdVelocity;

  /** Positive approach speed (m/s) required for material restitution bounce. */
  PX_FORCE_INLINE physx::PxReal bounceApproachSpeedThreshold() const {
    return bounceThresholdVelocity < 0.0f ? -bounceThresholdVelocity
                                          : bounceThresholdVelocity;
  }

  //-------------------------------------------------------------------------
  // Constraint correction limits
  //-------------------------------------------------------------------------

  physx::PxReal maxPositionCorrection; //!< Maximum position correction per
                                       //!< iteration (meters, default 0.2)
  physx::PxReal maxAngularCorrection;  //!< Maximum angular correction per
                                       //!< iteration (radians, default 0.5)
  physx::PxReal maxLambda;             //!< Maximum Lagrangian multiplier
                                       //!< magnitude (default 1e6)

  //-------------------------------------------------------------------------
  // Parallelization
  //-------------------------------------------------------------------------

  bool enableParallelization; //!< Enable graph coloring for parallel body
                              //!< updates

  bool enableLocal6x6Solve; //!< Use 6x6 local system solve in block descent
                            //!< (fallback to Gauss-Seidel when false)

  bool enableMassWeightedWeld; //!< Use mass-ratio weighted corrections for weld
                               //!< joints (runtime attachment stability)

  bool enableStageOwnershipDiagnostics; //!< Collect per-contact post-stage
                                        //!< ownership masks for opt-in gates
  bool enableBoundedComponentProductionProbe; //!< Opt-in complete-component
                                              //!< P3K owner replacement probe
  bool enableMatrixFreeComponentOracle; //!< Opt-in read-only dense/matrix-free
                                        //!< same-component comparison

  physx::PxU32 largeIslandThreshold; //!< Constraint count threshold to trigger
                                     //!< internal island parallelization.
                                     //!< Islands with more constraints use
                                     //!< constraint coloring for better cache
                                     //!< locality. Default: 128

  //-------------------------------------------------------------------------
  // AVBD Reference Parameters (from AVBD3D solver.cpp)
  //   alpha: Stabilization parameter for constraint error correction (0-1).
  //          Higher = slower/smoother correction. Reference default: 0.95
  //   beta:  Penalty growth rate. penalty += beta * |C| each dual update.
  //          Reference default: 1000
  //   gamma: Warmstart decay factor for penalty and lambda each frame.
  //          lambda *= alpha*gamma, penalty = clamp(penalty*gamma, MIN, MAX).
  //          Reference default: 0.99
  //   penaltyMin/Max: Bounds for adaptive penalty parameter.
  //          Reference: 1000 / 1e9
  //-------------------------------------------------------------------------

  physx::PxReal avbdAlpha;      //!< Stabilization (error correction speed)
  physx::PxReal avbdBeta;       //!< Penalty growth rate (dual update)
  physx::PxReal avbdGamma;      //!< Warmstart decay factor
  physx::PxReal avbdPenaltyMin; //!< Minimum adaptive penalty
  physx::PxReal avbdPenaltyMax; //!< Maximum adaptive penalty

  //-------------------------------------------------------------------------
  // Determinism (for multi-platform synchronization)
  //-------------------------------------------------------------------------

  physx::PxU32 determinismFlags; //!< Bitmask of AvbdDeterminismFlags::Enum
                                 //!< for cross-platform determinism

  //-------------------------------------------------------------------------
  // Defaults
  //-------------------------------------------------------------------------

  AvbdSolverConfig()
      : lengthScale(1.0f), outerIterations(1), innerIterations(4), initialRho(1e4f),
        rhoScale(2.0f), maxRho(1e8f), defaultCompliance(1e-6f),
        contactCompliance(1e-4f), jointCompliance(1e-8f), contactDamping(0.5f),
        damping(0.5f), angularDamping(0.95f), rotationThreshold(0.001f),
        angularScale(400.0f), angularContactScale(0.2f), baumgarte(0.3f),
        chebyshevRho(0.92f), positionTolerance(1e-4f), velocityDamping(0.99f),
        bounceThresholdVelocity(-2.0f),
        maxPositionCorrection(0.2f), maxAngularCorrection(0.5f),
        maxLambda(1e6f), enableParallelization(true),
        enableLocal6x6Solve(false), enableMassWeightedWeld(false),
        enableStageOwnershipDiagnostics(false),
        enableBoundedComponentProductionProbe(false),
        enableMatrixFreeComponentOracle(false),
        largeIslandThreshold(128),
        avbdAlpha(0.95f), avbdBeta(1000.0f), avbdGamma(0.99f),
        avbdPenaltyMin(1000.0f), avbdPenaltyMax(1e9f),
        determinismFlags(0) {}

  /**
   * @brief Enable deterministic simulation for cross-platform synchronization
   *
   * When enabled, the solver will:
   * - Sort constraints by body pair indices for consistent iteration order
   * - Sort bodies by node index before each solve iteration
   * - Use Kahan summation to reduce floating-point rounding errors
   *
   * Note: This may reduce performance slightly but ensures identical
   * results across different platforms (x86, ARM, etc.)
   */
  void enableDeterminism() {
    determinismFlags = AvbdDeterminismFlags::eDETERMINISTIC_DEFAULT;
    enableParallelization =
        false; // Disable parallelization for strict determinism
  }

  /**
   * @brief Check if determinism is enabled
   */
  bool isDeterministic() const {
    return (determinismFlags & AvbdDeterminismFlags::eDETERMINISTIC_DEFAULT) !=
           0;
  }
};

/**
 * @brief AVBD solver statistics for debugging and profiling
 */
struct AvbdSolverStats {
  physx::PxU32 numBodies;      //!< Number of dynamic bodies solved
  physx::PxU32 numContacts;    //!< Number of contact constraints
  physx::PxU32 numJoints;      //!< Number of joint constraints
  physx::PxU32 numColorGroups; //!< Number of color groups for parallelization
  physx::PxU32 activeConstraints; //!< Number of active (violating) constraints

  physx::PxU32 totalIterations; //!< Total inner iterations executed
  physx::PxU32 bodyStaticNormalAlRows;
  physx::PxU64 bodyStaticNormalAlEvaluations;
  physx::PxU32 bodyStaticDepenetrationCorrections;
  physx::PxU32 bodyStaticDepenetrationEligibleRows;
  physx::PxU32 bodyStaticDepenetrationFiniteImpulseSkips;
  physx::PxU32 bodyStaticDepenetrationAuthoredFiniteImpulseSkips;
  physx::PxU32 bodyStaticMaterialVelocityCorrections;
  physx::PxU32 bodyStaticRestitutionCorrections;
  physx::PxU32 bodyStaticNormalWarmstartHits;
  physx::PxU32 bodyStaticNormalWarmstartMisses;
  physx::PxU32 bodyStaticNormalWarmstartAge0;
  physx::PxU32 bodyStaticNormalWarmstartAge1;
  physx::PxU32 bodyStaticNormalWarmstartAge2;
  physx::PxU32 bodyStaticNormalWarmstartAge3;
  physx::PxU32 bodyStaticNormalManagerOnsetRows;
  physx::PxU32 bodyStaticNormalManagerSupportRows;
  physx::PxU32 bodyStaticNormalManagerAge0;
  physx::PxU32 bodyStaticNormalManagerAge1;
  physx::PxU32 bodyStaticNormalManagerAge2;
  physx::PxU32 bodyStaticNormalManagerAge3;
  physx::PxU32 bodyStaticNormalRowMissOnManagerSupportRows;
  physx::PxU32 bodyStaticNormalOnsetFinalizeBodies;
  physx::PxU32 bodyStaticNormalSupportFinalizeBodies;
  physx::PxU32 bodyStaticNormalOnsetFinalizeCorrections;
  physx::PxU32 bodyStaticNormalSupportFinalizeCorrections;
  physx::PxU32 bodyStaticNormalOnsetDepenetrationEligibleRows;
  physx::PxU32 bodyStaticNormalSupportDepenetrationEligibleRows;
  physx::PxU32 bodyStaticNormalOnsetDepenetrationCorrections;
  physx::PxU32 bodyStaticNormalSupportDepenetrationCorrections;
  physx::PxU32 bodyStaticNormalOnsetShallowDepenetrationCorrections;
  physx::PxU32 bodyStaticNormalOnsetDeepDepenetrationCorrections;
  physx::PxU32 bodyStaticNormalSupportShallowDepenetrationCorrections;
  physx::PxU32 bodyStaticNormalSupportDeepDepenetrationCorrections;
  physx::PxU32 bodyStaticMaterialFiniteBudgetRows;
  physx::PxU32 bodyStaticMaterialUnlimitedBudgetRows;
  physx::PxU32 contactFrictionTargetAlEvaluations;
  physx::PxU32 bodyStaticFrictionTargetRows;
  physx::PxU32 bodyStaticFrictionTargetCorrections;
  physx::PxU32 bodyStaticFrictionFallbackRows;
  physx::PxU32 bodyStaticFrictionFallbackCorrections;
  physx::PxU32 contactTargetNormalProjectionRows;
  physx::PxU32 contactTargetNormalCorrections;
  physx::PxU32 contactTargetTangentRows;
  physx::PxU32 contactTargetTangentCorrections;
  physx::PxU32 surfaceDeformableAlRows;
  physx::PxU64 surfaceDeformableAlEvaluations;
  physx::PxU32 surfaceDeformablePositionTangentCandidates;
  physx::PxU32 surfaceDeformablePositionTangentRows;
  physx::PxU64 surfaceDeformablePositionTangentEvaluations;
  physx::PxU32 surfaceDeformablePositionTangentMixedRejectRows;
  physx::PxU32 surfaceDeformablePositionTangentShellRejectRows;
  physx::PxU32 surfaceDeformablePositionTangentTargetRejectRows;
  physx::PxU32 surfaceDeformablePositionTangentRestitutionRejectRows;
  physx::PxU32 surfaceDeformablePositionTangentFiniteRejectRows;
  physx::PxU32 surfaceDeformablePositionTangentScaleRejectRows;
  physx::PxU32 surfaceDeformableStrippedRows;
  physx::PxU32 surfaceDeformableShellSuppressedPrimalRows;
  physx::PxU32 surfaceDeformableDepenetrationCorrections;
  physx::PxU32 surfaceDeformableFrictionRawRows;
  physx::PxU32 surfaceDeformableFrictionDominantRows;
  physx::PxU32 surfaceDeformableFrictionFewContactRows;
  physx::PxU32 surfaceDeformableFrictionMultiCornerRows;
  physx::PxU32 surfaceDeformableFrictionCorrections;
  physx::PxU32 surfaceDeformableFinalizeBodies;
  physx::PxU32 surfaceDeformableFinalizeCorrections;
  physx::PxU32 surfaceDeformableFinalizeSpatialCorrections;
  physx::PxU32 surfaceDeformableFinalizeComFallbackCorrections;
  physx::PxU32 surfaceDeformableFinalizeSecondaryRows;
  physx::PxU32 surfaceDeformableFinalizeSecondaryResidualSeparationRows;
  physx::PxU32 surfaceDeformableFinalizeManifoldBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldOneRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldTwoRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldThreeRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldFourRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldOverFourRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldFiveToEightRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldNineToSixteenRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldOverSixteenRowBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldMixedScaleBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldRankDeficientBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldAliasRows;
  physx::PxU32 surfaceDeformableFinalizeManifoldDynamicIncidentBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldRigidStaticIncidentBodies;
  physx::PxU32 surfaceDeformableFinalizeManifoldNonOwnerDeformableIncidentBodies;
  physx::PxU32 surfaceDeformableFinalizeComponents;
  physx::PxU32 surfaceDeformableFinalizeComponentOneBody;
  physx::PxU32 surfaceDeformableFinalizeComponentTwoBodies;
  physx::PxU32 surfaceDeformableFinalizeComponentThreeToFourBodies;
  physx::PxU32 surfaceDeformableFinalizeComponentFiveToEightBodies;
  physx::PxU32 surfaceDeformableFinalizeComponentNineToSixteenBodies;
  physx::PxU32 surfaceDeformableFinalizeComponentSeventeenToThirtyTwoBodies;
  physx::PxU32 surfaceDeformableFinalizeComponentOverThirtyTwoBodies;
  physx::PxU32 surfaceDeformableFinalizeComponentOneToEightRows;
  physx::PxU32 surfaceDeformableFinalizeComponentNineToSixteenRows;
  physx::PxU32 surfaceDeformableFinalizeComponentSeventeenToThirtyTwoRows;
  physx::PxU32 surfaceDeformableFinalizeComponentThirtyThreeToSixtyFourRows;
  physx::PxU32 surfaceDeformableFinalizeComponentOverSixtyFourRows;
  physx::PxU32 surfaceDeformableFinalizeComponentRestitution;
  physx::PxU32 surfaceDeformableFinalizeComponentFiniteImpulse;
  physx::PxU32 surfaceDeformableFinalizeComponentTargetVelocity;
  physx::PxU32 surfaceDeformableFinalizeComponentMixedScale;
  physx::PxU32 surfaceDeformableFinalizeComponentRigidStatic;
  physx::PxU32 surfaceDeformableFinalizeComponentNonOwnerDeformable;
  physx::PxU32 surfaceDeformableFinalizeComponentJointIsland;
  physx::PxU32 surfaceDeformableFinalizeComponentLockedDof;
  physx::PxU32 surfaceDeformableFinalizeComponentNonDynamicBody;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagRows;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagNoCorrectionRows;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagZeroBudgetRequiredRows;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagWithinBudgetRows;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagOverBudgetRows;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagUnsupportedRows;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagComponentsWithinBudget;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagComponentsOverBudget;
  physx::PxU32 surfaceDeformableFinalizeBudgetDiagComponentsUnsupported;
  physx::PxU32 surfaceDeformableFinalizeShadowComponents;
  physx::PxU32 surfaceDeformableFinalizeShadowRows;
  physx::PxU32 surfaceDeformableFinalizeShadowNoCorrection;
  physx::PxU32 surfaceDeformableFinalizeShadowSolved;
  physx::PxU32 surfaceDeformableFinalizeShadowCommitCapable;
  physx::PxU32 surfaceDeformableFinalizeShadowBudgetExhausted;
  physx::PxU32 surfaceDeformableFinalizeShadowInfeasible;
  physx::PxU32 surfaceDeformableFinalizeShadowResidualUnclassified;
  physx::PxU32 surfaceDeformableFinalizeShadowNumericalFailure;
  physx::PxU32 surfaceDeformableFinalizeShadowIterationLimit;
  physx::PxU32 surfaceDeformableFinalizeShadowUnsupported;
  physx::PxU32 surfaceDeformableFinalizeShadowUnsupportedFastImpact;
  physx::PxU32 surfaceDeformableFinalizeShadowUnsupportedSnapshot;
  physx::PxU32 surfaceDeformableFinalizeShadowLowerRows;
  physx::PxU32 surfaceDeformableFinalizeShadowFreeRows;
  physx::PxU32 surfaceDeformableFinalizeShadowUpperRows;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeComponents;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeRows;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeNoCorrection;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeSolved;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeBudgetExhausted;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeInfeasible;
  physx::PxU32
      surfaceDeformableFinalizeShadowMatrixFreeResidualUnclassified;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeNumericalFailure;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeIterationLimit;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeIterations;
  physx::PxU32
      surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost2x;
  physx::PxU32
      surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost16x;
  physx::PxU32
      surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktOver16x;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeCommittedComponents;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeOracleComponents;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeOracleRows;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeOracleMatched;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeOracleMismatched;
  physx::PxU32 surfaceDeformableFinalizeShadowMatrixFreeOracleSkipped;
  physx::PxU32 surfaceDeformableFinalizePreOwnerBodies;
  physx::PxU32 surfaceDeformableFinalizeLegacyOwnerBodies;
  physx::PxU32 surfaceDeformableFinalizeOwnerDiscoveryMismatchBodies;
  physx::PxU32 surfaceDeformableFinalizeProbeEligibleComponents;
  physx::PxU32 surfaceDeformableFinalizeProbeCommittedComponents;
  physx::PxU32 surfaceDeformableFinalizeProbeCommittedRows;
  physx::PxU32 surfaceDeformableFinalizeProbeCommittedBodies;
  physx::PxU32 surfaceDeformableFinalizeProbeReplacedOwnerBodies;
  physx::PxU32 surfaceDeformableAlDepenetrationRows;
  physx::PxU32 surfaceDeformableAlFinalizeRows;
  physx::PxU32 surfaceDeformableDepenetrationFinalizeRows;
  physx::PxU32 surfaceDeformableAlDepenetrationFinalizeRows;
  physx::PxU32 surfaceDeformableFinalizeContactFalsePositiveCorrections;
  physx::PxU32 surfaceDeformableFinalizeContactResidualSeparationCorrections;
  physx::PxU32 surfaceDeformableFinalizeContactReversalCorrections;
  physx::PxU32 surfaceShellContacts;
  physx::PxU32 surfaceShellDepenetrationCorrections;
  physx::PxU32 surfaceShellFrictionRows;
  physx::PxU32 surfaceShellFrictionCorrections;
  physx::PxU32 surfaceShellFinalizeBodies;
  physx::PxU32 surfaceShellFinalizeCorrections;

  physx::PxReal constraintError;        //!< RMS constraint error
  physx::PxReal maxPositionError;       //!< Maximum position error after solve
  physx::PxReal avgPositionError;       //!< Average position error after solve
  physx::PxReal maxConstraintViolation; //!< Maximum constraint violation
  physx::PxReal bodyStaticDepenetrationDistance;
  physx::PxReal bodyStaticDepenetrationMaxCorrection;
  physx::PxReal bodyStaticMaterialVelocityDelta;
  physx::PxReal bodyStaticMaterialVelocityMaxDelta;
  physx::PxReal bodyStaticNormalRestoredLambdaMax;
  physx::PxReal bodyStaticNormalRestoredPenaltyMax;
  physx::PxReal bodyStaticNormalInitialPenaltyMax;
  physx::PxReal bodyStaticNormalPreAlRawPenetration;
  physx::PxReal bodyStaticNormalPostAlRawPenetration;
  physx::PxReal bodyStaticNormalAlphaC0Offset;
  physx::PxReal bodyStaticNormalPreAlPenetration;
  physx::PxReal bodyStaticNormalPostAlPenetration;
  physx::PxReal bodyStaticNormalPostAlSeparation;
  physx::PxReal bodyStaticNormalAlOutwardDistance;
  physx::PxReal bodyStaticNormalAlInwardDistance;
  physx::PxReal bodyStaticMaterialPoseSeparatingVelocity;
  physx::PxReal bodyStaticMaterialAllowedSeparatingVelocity;
  physx::PxReal bodyStaticMaterialFiniteRemainingImpulse;
  physx::PxReal bodyStaticNormalOnsetPreAlRawPenetration;
  physx::PxReal bodyStaticNormalOnsetPreAlPenetration;
  physx::PxReal bodyStaticNormalOnsetPostAlRawPenetration;
  physx::PxReal bodyStaticNormalOnsetPostAlPenetration;
  physx::PxReal bodyStaticNormalOnsetAlphaC0Offset;
  physx::PxReal bodyStaticNormalOnsetAlOutwardDistance;
  physx::PxReal bodyStaticNormalSupportPreAlRawPenetration;
  physx::PxReal bodyStaticNormalSupportPreAlPenetration;
  physx::PxReal bodyStaticNormalSupportPostAlRawPenetration;
  physx::PxReal bodyStaticNormalSupportPostAlPenetration;
  physx::PxReal bodyStaticNormalSupportAlphaC0Offset;
  physx::PxReal bodyStaticNormalSupportAlOutwardDistance;
  physx::PxReal bodyStaticNormalOnsetPoseSeparatingVelocity;
  physx::PxReal bodyStaticNormalSupportPoseSeparatingVelocity;
  physx::PxReal bodyStaticNormalOnsetFinalizeDelta;
  physx::PxReal bodyStaticNormalSupportFinalizeDelta;
  physx::PxReal bodyStaticNormalOnsetDepenetrationDistance;
  physx::PxReal bodyStaticNormalSupportDepenetrationDistance;
  physx::PxReal bodyStaticNormalOnsetShallowDepenetrationDistance;
  physx::PxReal bodyStaticNormalOnsetDeepDepenetrationDistance;
  physx::PxReal bodyStaticNormalSupportShallowDepenetrationDistance;
  physx::PxReal bodyStaticNormalSupportDeepDepenetrationDistance;
  physx::PxReal bodyStaticFrictionTargetImpulse;
  physx::PxReal bodyStaticFrictionFallbackImpulse;
  physx::PxReal contactTargetNormalImpulse;
  physx::PxReal contactTargetTangentImpulse;
  physx::PxReal surfaceDeformableDepenetrationDistance;
  physx::PxReal surfaceDeformableFrictionImpulse;
  physx::PxReal surfaceDeformableFinalizeDelta;
  physx::PxReal surfaceDeformableFinalizeContactPreSeparation;
  physx::PxReal surfaceDeformableFinalizeContactPostSeparation;
  physx::PxReal surfaceDeformableFinalizeContactPostApproach;
  physx::PxReal surfaceDeformableFinalizeSecondaryResidualSeparation;
  physx::PxReal surfaceShellDepenetrationDistance;
  physx::PxReal surfaceShellFrictionImpulse;
  physx::PxReal surfaceShellFinalizeDelta;

  physx::PxReal totalEnergy; //!< Total system energy (kinetic + potential)

  physx::PxU64 solveTimeUs; //!< Solve time in microseconds

  void reset() {
    numBodies = 0;
    numContacts = 0;
    numJoints = 0;
    numColorGroups = 0;
    activeConstraints = 0;
    totalIterations = 0;
    bodyStaticNormalAlRows = 0;
    bodyStaticNormalAlEvaluations = 0;
    bodyStaticDepenetrationCorrections = 0;
    bodyStaticDepenetrationEligibleRows = 0;
    bodyStaticDepenetrationFiniteImpulseSkips = 0;
    bodyStaticDepenetrationAuthoredFiniteImpulseSkips = 0;
    bodyStaticMaterialVelocityCorrections = 0;
    bodyStaticRestitutionCorrections = 0;
    bodyStaticNormalWarmstartHits = 0;
    bodyStaticNormalWarmstartMisses = 0;
    bodyStaticNormalWarmstartAge0 = 0;
    bodyStaticNormalWarmstartAge1 = 0;
    bodyStaticNormalWarmstartAge2 = 0;
    bodyStaticNormalWarmstartAge3 = 0;
    bodyStaticNormalManagerOnsetRows = 0;
    bodyStaticNormalManagerSupportRows = 0;
    bodyStaticNormalManagerAge0 = 0;
    bodyStaticNormalManagerAge1 = 0;
    bodyStaticNormalManagerAge2 = 0;
    bodyStaticNormalManagerAge3 = 0;
    bodyStaticNormalRowMissOnManagerSupportRows = 0;
    bodyStaticNormalOnsetFinalizeBodies = 0;
    bodyStaticNormalSupportFinalizeBodies = 0;
    bodyStaticNormalOnsetFinalizeCorrections = 0;
    bodyStaticNormalSupportFinalizeCorrections = 0;
    bodyStaticNormalOnsetDepenetrationEligibleRows = 0;
    bodyStaticNormalSupportDepenetrationEligibleRows = 0;
    bodyStaticNormalOnsetDepenetrationCorrections = 0;
    bodyStaticNormalSupportDepenetrationCorrections = 0;
    bodyStaticNormalOnsetShallowDepenetrationCorrections = 0;
    bodyStaticNormalOnsetDeepDepenetrationCorrections = 0;
    bodyStaticNormalSupportShallowDepenetrationCorrections = 0;
    bodyStaticNormalSupportDeepDepenetrationCorrections = 0;
    bodyStaticMaterialFiniteBudgetRows = 0;
    bodyStaticMaterialUnlimitedBudgetRows = 0;
    contactFrictionTargetAlEvaluations = 0;
    bodyStaticFrictionTargetRows = 0;
    bodyStaticFrictionTargetCorrections = 0;
    bodyStaticFrictionFallbackRows = 0;
    bodyStaticFrictionFallbackCorrections = 0;
    contactTargetNormalProjectionRows = 0;
    contactTargetNormalCorrections = 0;
    contactTargetTangentRows = 0;
    contactTargetTangentCorrections = 0;
    surfaceDeformableAlRows = 0;
    surfaceDeformableAlEvaluations = 0;
    surfaceDeformablePositionTangentCandidates = 0;
    surfaceDeformablePositionTangentRows = 0;
    surfaceDeformablePositionTangentEvaluations = 0;
    surfaceDeformablePositionTangentMixedRejectRows = 0;
    surfaceDeformablePositionTangentShellRejectRows = 0;
    surfaceDeformablePositionTangentTargetRejectRows = 0;
    surfaceDeformablePositionTangentRestitutionRejectRows = 0;
    surfaceDeformablePositionTangentFiniteRejectRows = 0;
    surfaceDeformablePositionTangentScaleRejectRows = 0;
    surfaceDeformableStrippedRows = 0;
    surfaceDeformableShellSuppressedPrimalRows = 0;
    surfaceDeformableDepenetrationCorrections = 0;
    surfaceDeformableFrictionRawRows = 0;
    surfaceDeformableFrictionDominantRows = 0;
    surfaceDeformableFrictionFewContactRows = 0;
    surfaceDeformableFrictionMultiCornerRows = 0;
    surfaceDeformableFrictionCorrections = 0;
    surfaceDeformableFinalizeBodies = 0;
    surfaceDeformableFinalizeCorrections = 0;
    surfaceDeformableFinalizeSpatialCorrections = 0;
    surfaceDeformableFinalizeComFallbackCorrections = 0;
    surfaceDeformableFinalizeSecondaryRows = 0;
    surfaceDeformableFinalizeSecondaryResidualSeparationRows = 0;
    surfaceDeformableFinalizeManifoldBodies = 0;
    surfaceDeformableFinalizeManifoldOneRowBodies = 0;
    surfaceDeformableFinalizeManifoldTwoRowBodies = 0;
    surfaceDeformableFinalizeManifoldThreeRowBodies = 0;
    surfaceDeformableFinalizeManifoldFourRowBodies = 0;
    surfaceDeformableFinalizeManifoldOverFourRowBodies = 0;
    surfaceDeformableFinalizeManifoldFiveToEightRowBodies = 0;
    surfaceDeformableFinalizeManifoldNineToSixteenRowBodies = 0;
    surfaceDeformableFinalizeManifoldOverSixteenRowBodies = 0;
    surfaceDeformableFinalizeManifoldMixedScaleBodies = 0;
    surfaceDeformableFinalizeManifoldRankDeficientBodies = 0;
    surfaceDeformableFinalizeManifoldAliasRows = 0;
    surfaceDeformableFinalizeManifoldDynamicIncidentBodies = 0;
    surfaceDeformableFinalizeManifoldRigidStaticIncidentBodies = 0;
    surfaceDeformableFinalizeManifoldNonOwnerDeformableIncidentBodies = 0;
    surfaceDeformableFinalizeComponents = 0;
    surfaceDeformableFinalizeComponentOneBody = 0;
    surfaceDeformableFinalizeComponentTwoBodies = 0;
    surfaceDeformableFinalizeComponentThreeToFourBodies = 0;
    surfaceDeformableFinalizeComponentFiveToEightBodies = 0;
    surfaceDeformableFinalizeComponentNineToSixteenBodies = 0;
    surfaceDeformableFinalizeComponentSeventeenToThirtyTwoBodies = 0;
    surfaceDeformableFinalizeComponentOverThirtyTwoBodies = 0;
    surfaceDeformableFinalizeComponentOneToEightRows = 0;
    surfaceDeformableFinalizeComponentNineToSixteenRows = 0;
    surfaceDeformableFinalizeComponentSeventeenToThirtyTwoRows = 0;
    surfaceDeformableFinalizeComponentThirtyThreeToSixtyFourRows = 0;
    surfaceDeformableFinalizeComponentOverSixtyFourRows = 0;
    surfaceDeformableFinalizeComponentRestitution = 0;
    surfaceDeformableFinalizeComponentFiniteImpulse = 0;
    surfaceDeformableFinalizeComponentTargetVelocity = 0;
    surfaceDeformableFinalizeComponentMixedScale = 0;
    surfaceDeformableFinalizeComponentRigidStatic = 0;
    surfaceDeformableFinalizeComponentNonOwnerDeformable = 0;
    surfaceDeformableFinalizeComponentJointIsland = 0;
    surfaceDeformableFinalizeComponentLockedDof = 0;
    surfaceDeformableFinalizeComponentNonDynamicBody = 0;
    surfaceDeformableFinalizeBudgetDiagRows = 0;
    surfaceDeformableFinalizeBudgetDiagNoCorrectionRows = 0;
    surfaceDeformableFinalizeBudgetDiagZeroBudgetRequiredRows = 0;
    surfaceDeformableFinalizeBudgetDiagWithinBudgetRows = 0;
    surfaceDeformableFinalizeBudgetDiagOverBudgetRows = 0;
    surfaceDeformableFinalizeBudgetDiagUnsupportedRows = 0;
    surfaceDeformableFinalizeBudgetDiagComponentsWithinBudget = 0;
    surfaceDeformableFinalizeBudgetDiagComponentsOverBudget = 0;
    surfaceDeformableFinalizeBudgetDiagComponentsUnsupported = 0;
    surfaceDeformableFinalizeShadowComponents = 0;
    surfaceDeformableFinalizeShadowRows = 0;
    surfaceDeformableFinalizeShadowNoCorrection = 0;
    surfaceDeformableFinalizeShadowSolved = 0;
    surfaceDeformableFinalizeShadowCommitCapable = 0;
    surfaceDeformableFinalizeShadowBudgetExhausted = 0;
    surfaceDeformableFinalizeShadowInfeasible = 0;
    surfaceDeformableFinalizeShadowResidualUnclassified = 0;
    surfaceDeformableFinalizeShadowNumericalFailure = 0;
    surfaceDeformableFinalizeShadowIterationLimit = 0;
    surfaceDeformableFinalizeShadowUnsupported = 0;
    surfaceDeformableFinalizeShadowUnsupportedFastImpact = 0;
    surfaceDeformableFinalizeShadowUnsupportedSnapshot = 0;
    surfaceDeformableFinalizeShadowLowerRows = 0;
    surfaceDeformableFinalizeShadowFreeRows = 0;
    surfaceDeformableFinalizeShadowUpperRows = 0;
    surfaceDeformableFinalizeShadowMatrixFreeComponents = 0;
    surfaceDeformableFinalizeShadowMatrixFreeRows = 0;
    surfaceDeformableFinalizeShadowMatrixFreeNoCorrection = 0;
    surfaceDeformableFinalizeShadowMatrixFreeSolved = 0;
    surfaceDeformableFinalizeShadowMatrixFreeBudgetExhausted = 0;
    surfaceDeformableFinalizeShadowMatrixFreeInfeasible = 0;
    surfaceDeformableFinalizeShadowMatrixFreeResidualUnclassified = 0;
    surfaceDeformableFinalizeShadowMatrixFreeNumericalFailure = 0;
    surfaceDeformableFinalizeShadowMatrixFreeIterationLimit = 0;
    surfaceDeformableFinalizeShadowMatrixFreeIterations = 0;
    surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost2x = 0;
    surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktAtMost16x = 0;
    surfaceDeformableFinalizeShadowMatrixFreeIterationLimitKktOver16x = 0;
    surfaceDeformableFinalizeShadowMatrixFreeCommittedComponents = 0;
    surfaceDeformableFinalizeShadowMatrixFreeOracleComponents = 0;
    surfaceDeformableFinalizeShadowMatrixFreeOracleRows = 0;
    surfaceDeformableFinalizeShadowMatrixFreeOracleMatched = 0;
    surfaceDeformableFinalizeShadowMatrixFreeOracleMismatched = 0;
    surfaceDeformableFinalizeShadowMatrixFreeOracleSkipped = 0;
    surfaceDeformableFinalizePreOwnerBodies = 0;
    surfaceDeformableFinalizeLegacyOwnerBodies = 0;
    surfaceDeformableFinalizeOwnerDiscoveryMismatchBodies = 0;
    surfaceDeformableFinalizeProbeEligibleComponents = 0;
    surfaceDeformableFinalizeProbeCommittedComponents = 0;
    surfaceDeformableFinalizeProbeCommittedRows = 0;
    surfaceDeformableFinalizeProbeCommittedBodies = 0;
    surfaceDeformableFinalizeProbeReplacedOwnerBodies = 0;
    surfaceDeformableAlDepenetrationRows = 0;
    surfaceDeformableAlFinalizeRows = 0;
    surfaceDeformableDepenetrationFinalizeRows = 0;
    surfaceDeformableAlDepenetrationFinalizeRows = 0;
    surfaceDeformableFinalizeContactFalsePositiveCorrections = 0;
    surfaceDeformableFinalizeContactResidualSeparationCorrections = 0;
    surfaceDeformableFinalizeContactReversalCorrections = 0;
    surfaceShellContacts = 0;
    surfaceShellDepenetrationCorrections = 0;
    surfaceShellFrictionRows = 0;
    surfaceShellFrictionCorrections = 0;
    surfaceShellFinalizeBodies = 0;
    surfaceShellFinalizeCorrections = 0;
    constraintError = 0.0f;
    maxPositionError = 0.0f;
    avgPositionError = 0.0f;
    maxConstraintViolation = 0.0f;
    bodyStaticDepenetrationDistance = 0.0f;
    bodyStaticDepenetrationMaxCorrection = 0.0f;
    bodyStaticMaterialVelocityDelta = 0.0f;
    bodyStaticMaterialVelocityMaxDelta = 0.0f;
    bodyStaticNormalRestoredLambdaMax = 0.0f;
    bodyStaticNormalRestoredPenaltyMax = 0.0f;
    bodyStaticNormalInitialPenaltyMax = 0.0f;
    bodyStaticNormalPreAlRawPenetration = 0.0f;
    bodyStaticNormalPostAlRawPenetration = 0.0f;
    bodyStaticNormalAlphaC0Offset = 0.0f;
    bodyStaticNormalPreAlPenetration = 0.0f;
    bodyStaticNormalPostAlPenetration = 0.0f;
    bodyStaticNormalPostAlSeparation = 0.0f;
    bodyStaticNormalAlOutwardDistance = 0.0f;
    bodyStaticNormalAlInwardDistance = 0.0f;
    bodyStaticMaterialPoseSeparatingVelocity = 0.0f;
    bodyStaticMaterialAllowedSeparatingVelocity = 0.0f;
    bodyStaticMaterialFiniteRemainingImpulse = 0.0f;
    bodyStaticNormalOnsetPreAlRawPenetration = 0.0f;
    bodyStaticNormalOnsetPreAlPenetration = 0.0f;
    bodyStaticNormalOnsetPostAlRawPenetration = 0.0f;
    bodyStaticNormalOnsetPostAlPenetration = 0.0f;
    bodyStaticNormalOnsetAlphaC0Offset = 0.0f;
    bodyStaticNormalOnsetAlOutwardDistance = 0.0f;
    bodyStaticNormalSupportPreAlRawPenetration = 0.0f;
    bodyStaticNormalSupportPreAlPenetration = 0.0f;
    bodyStaticNormalSupportPostAlRawPenetration = 0.0f;
    bodyStaticNormalSupportPostAlPenetration = 0.0f;
    bodyStaticNormalSupportAlphaC0Offset = 0.0f;
    bodyStaticNormalSupportAlOutwardDistance = 0.0f;
    bodyStaticNormalOnsetPoseSeparatingVelocity = 0.0f;
    bodyStaticNormalSupportPoseSeparatingVelocity = 0.0f;
    bodyStaticNormalOnsetFinalizeDelta = 0.0f;
    bodyStaticNormalSupportFinalizeDelta = 0.0f;
    bodyStaticNormalOnsetDepenetrationDistance = 0.0f;
    bodyStaticNormalSupportDepenetrationDistance = 0.0f;
    bodyStaticNormalOnsetShallowDepenetrationDistance = 0.0f;
    bodyStaticNormalOnsetDeepDepenetrationDistance = 0.0f;
    bodyStaticNormalSupportShallowDepenetrationDistance = 0.0f;
    bodyStaticNormalSupportDeepDepenetrationDistance = 0.0f;
    bodyStaticFrictionTargetImpulse = 0.0f;
    bodyStaticFrictionFallbackImpulse = 0.0f;
    contactTargetNormalImpulse = 0.0f;
    contactTargetTangentImpulse = 0.0f;
    surfaceDeformableDepenetrationDistance = 0.0f;
    surfaceDeformableFrictionImpulse = 0.0f;
    surfaceDeformableFinalizeDelta = 0.0f;
    surfaceDeformableFinalizeContactPreSeparation = 0.0f;
    surfaceDeformableFinalizeContactPostSeparation = 0.0f;
    surfaceDeformableFinalizeContactPostApproach = 0.0f;
    surfaceDeformableFinalizeSecondaryResidualSeparation = 0.0f;
    surfaceShellDepenetrationDistance = 0.0f;
    surfaceShellFrictionImpulse = 0.0f;
    surfaceShellFinalizeDelta = 0.0f;
    totalEnergy = 0.0f;
    solveTimeUs = 0;
  }
};

//-----------------------------------------------------------------------------
// Graph Coloring Types
//-----------------------------------------------------------------------------

/**
 * @brief Represents a color group for parallel solving
 */
struct AvbdBodyColorGroup {
  physx::PxU32 *bodyIndices; //!< Indices of bodies in this color group
  physx::PxU32 numBodies;    //!< Number of bodies in this group
  physx::PxU32 capacity;     //!< Allocated capacity
};

/**
 * @brief Helper for graph coloring bodies that share constraints
 */
struct AvbdGraphColoring {
  AvbdBodyColorGroup *colorGroups; //!< Array of color groups
  physx::PxU32 numColors;          //!< Number of colors used
  physx::PxU32 maxColors;          //!< Maximum colors allocated

  /**
   * @brief Initialize coloring structure
   */
  inline void initialize(physx::PxU32 maxColorsIn,
                         physx::PxAllocatorCallback &allocator) {
    maxColors = maxColorsIn;
    numColors = 0;
    colorGroups = static_cast<AvbdBodyColorGroup *>(
        allocator.allocate(sizeof(AvbdBodyColorGroup) * maxColors,
                           "AvbdColorGroups", __FILE__, __LINE__));
    for (physx::PxU32 i = 0; i < maxColors; ++i) {
      colorGroups[i].bodyIndices = nullptr;
      colorGroups[i].numBodies = 0;
      colorGroups[i].capacity = 0;
    }
  }

  /**
   * @brief Release coloring structure
   */
  inline void release(physx::PxAllocatorCallback &allocator) {
    if (colorGroups) {
      for (physx::PxU32 i = 0; i < maxColors; ++i) {
        if (colorGroups[i].bodyIndices) {
          allocator.deallocate(colorGroups[i].bodyIndices);
          colorGroups[i].bodyIndices = nullptr;
        }
      }
      allocator.deallocate(colorGroups);
      colorGroups = nullptr;
    }
    numColors = 0;
    maxColors = 0;
  }

  /**
   * @brief Perform greedy graph coloring on bodies
   */
  inline physx::PxU32
  computeColoring(const physx::PxU32 *const * /*adjacencyList*/,
                  const physx::PxU32 * /*adjacencyListSizes*/,
                  physx::PxU32 /*numBodies*/) {
    // Placeholder implementation - actual coloring is done in
    // AvbdSolver::computeGraphColoring
    return numColors;
  }
};

/**
 * @brief 6x6 block matrix for rigid body local solve
 *
 * Represents the local Hessian contribution for a single body:
 *   H = [ M/h^2  0    ]
 *       [ 0     I/h^2 ]
 * Plus constraint contributions from connected bodies.
 */
struct PX_ALIGN_PREFIX(16) AvbdBlock6x6 {
  // Upper-left 3x3 (linear-linear coupling)
  physx::PxMat33 linearLinear;

  // Upper-right 3x3 (linear-angular coupling)
  physx::PxMat33 linearAngular;

  // Lower-left 3x3 (angular-linear coupling)
  physx::PxMat33 angularLinear;

  // Lower-right 3x3 (angular-angular coupling)
  physx::PxMat33 angularAngular;

  /**
   * @brief Set to identity matrix
   */
  PX_FORCE_INLINE void setIdentity() {
    linearLinear = physx::PxMat33(physx::PxIdentity);
    linearAngular = physx::PxMat33(physx::PxZero);
    angularLinear = physx::PxMat33(physx::PxZero);
    angularAngular = physx::PxMat33(physx::PxIdentity);
  }

  /**
   * @brief Set to zero matrix
   */
  PX_FORCE_INLINE void setZero() {
    linearLinear = physx::PxMat33(physx::PxZero);
    linearAngular = physx::PxMat33(physx::PxZero);
    angularLinear = physx::PxMat33(physx::PxZero);
    angularAngular = physx::PxMat33(physx::PxZero);
  }

  /**
   * @brief Initialize diagonal blocks from inverse mass and inertia
   */
  PX_FORCE_INLINE void initializeDiagonal(physx::PxReal invMass,
                                          const physx::PxMat33 &invInertia,
                                          physx::PxReal invDtSq) {
    // M/h^2 on linear diagonal
    physx::PxReal massContrib =
        (invMass > 0.0f) ? (1.0f / invMass) * invDtSq : 0.0f;
    linearLinear = physx::PxMat33(physx::PxVec3(massContrib, 0, 0),
                                  physx::PxVec3(0, massContrib, 0),
                                  physx::PxVec3(0, 0, massContrib));

    // I/h^2 on angular diagonal: compute I = inv(invInertia) using full
    // 3x3 matrix inverse, since invInertiaWorld has off-diagonal terms
    // after rotation to world space.
    linearAngular = physx::PxMat33(physx::PxZero);
    angularLinear = physx::PxMat33(physx::PxZero);

    // Compute inertia tensor I = inv(invInertia) for angular block
    physx::PxMat33 inertiaTensor = invInertia.getInverse();
    angularAngular = inertiaTensor * invDtSq;
  }

  /**
   * @brief Add contribution from constraint to diagonal
   */
  PX_FORCE_INLINE void addConstraintContribution(const physx::PxVec3 &gradPos,
                                                 const physx::PxVec3 &gradRot,
                                                 physx::PxReal invCompliance) {
    // H += invCompliance * grad * grad^T
    for (physx::PxU32 i = 0; i < 3; ++i) {
      for (physx::PxU32 j = 0; j < 3; ++j) {
        linearLinear(i, j) += invCompliance * gradPos[i] * gradPos[j];
        linearAngular(i, j) += invCompliance * gradPos[i] * gradRot[j];
        angularLinear(i, j) += invCompliance * gradRot[i] * gradPos[j];
        angularAngular(i, j) += invCompliance * gradRot[i] * gradRot[j];
      }
    }
  }

  /**
   * Add a contact row whose linear and angular response are scaled locally.
   *
   * The geometric-mean cross scale keeps the local Hessian symmetric and
   * positive semidefinite.  A zero scale removes only that response component,
   * which is the contact-modification infinite-mass/inertia contract.
   */
  PX_FORCE_INLINE void addResponseScaledConstraintContribution(
      const physx::PxVec3 &gradPos, const physx::PxVec3 &gradRot,
      physx::PxReal invCompliance, physx::PxReal linearScale,
      physx::PxReal angularScale) {
    const physx::PxReal nonnegativeLinear =
        physx::PxMax(0.0f, linearScale);
    const physx::PxReal nonnegativeAngular =
        physx::PxMax(0.0f, angularScale);
    const physx::PxReal crossScale =
        physx::PxSqrt(nonnegativeLinear * nonnegativeAngular);
    for (physx::PxU32 i = 0; i < 3; ++i) {
      for (physx::PxU32 j = 0; j < 3; ++j) {
        linearLinear(i, j) +=
            invCompliance * nonnegativeLinear * gradPos[i] * gradPos[j];
        linearAngular(i, j) +=
            invCompliance * crossScale * gradPos[i] * gradRot[j];
        angularLinear(i, j) +=
            invCompliance * crossScale * gradRot[i] * gradPos[j];
        angularAngular(i, j) +=
            invCompliance * nonnegativeAngular * gradRot[i] * gradRot[j];
      }
    }
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief 6D vector for rigid body state (position + rotation as axis-angle)
 */
struct PX_ALIGN_PREFIX(16) AvbdVec6 {
  physx::PxVec3 linear; //!< Linear component (position or linear velocity)
  physx::PxReal padding0;
  physx::PxVec3 angular; //!< Angular component (rotation or angular velocity)
  physx::PxReal padding1;

  PX_FORCE_INLINE AvbdVec6()
      : linear(physx::PxZero), padding0(0), angular(physx::PxZero),
        padding1(0) {}

  PX_FORCE_INLINE AvbdVec6(const physx::PxVec3 &lin, const physx::PxVec3 &ang)
      : linear(lin), padding0(0), angular(ang), padding1(0) {}

  PX_FORCE_INLINE AvbdVec6 operator+(const AvbdVec6 &other) const {
    return AvbdVec6(linear + other.linear, angular + other.angular);
  }

  PX_FORCE_INLINE AvbdVec6 operator-(const AvbdVec6 &other) const {
    return AvbdVec6(linear - other.linear, angular - other.angular);
  }

  PX_FORCE_INLINE AvbdVec6 operator-() const {
    return AvbdVec6(-linear, -angular);
  }

  PX_FORCE_INLINE AvbdVec6 operator*(physx::PxReal s) const {
    return AvbdVec6(linear * s, angular * s);
  }

  PX_FORCE_INLINE physx::PxReal dot(const AvbdVec6 &other) const {
    return linear.dot(other.linear) + angular.dot(other.angular);
  }

} PX_ALIGN_SUFFIX(16);

/**
 * @brief LDLT decomposition for 6x6 symmetric positive definite matrices
 *
 * Decomposes A = L * D * L^T where:
 * - L is lower triangular with unit diagonal
 * - D is diagonal
 *
 * Used for solving H * x = b in the local system solver.
 */
struct PX_ALIGN_PREFIX(16) AvbdLDLT {
  AvbdBlock6x6 L;                // Lower triangular matrix (diagonal = 1)
  AvbdVec6 D;                    // Diagonal matrix (stored as vector)
  physx::PxReal conditionNumber; // Condition number of the matrix

  /**
   * @brief Compute condition number from diagonal D
   * @return Condition number (max(D) / min(D))
   */
  PX_FORCE_INLINE physx::PxReal computeConditionNumber() const {
    physx::PxReal minD = PX_MAX_F32;
    physx::PxReal maxD = 0.0f;

    for (int i = 0; i < 3; ++i) {
      if (D.linear[i] > 0.0f) {
        minD = physx::PxMin(minD, D.linear[i]);
        maxD = physx::PxMax(maxD, D.linear[i]);
      }
      if (D.angular[i] > 0.0f) {
        minD = physx::PxMin(minD, D.angular[i]);
        maxD = physx::PxMax(maxD, D.angular[i]);
      }
    }

    return (minD > 0.0f) ? (maxD / minD) : PX_MAX_F32;
  }

  /**
   * @brief Regularize matrix A by adding small diagonal terms
   * @param A Input matrix to regularize
   * @param reg Regularization coefficient
   * @return Regularized matrix
   */
  PX_FORCE_INLINE AvbdBlock6x6 regularizeMatrix(const AvbdBlock6x6 &A,
                                                physx::PxReal reg) const {
    AvbdBlock6x6 A_reg = A;

    // Add regularization to diagonal blocks
    for (int i = 0; i < 3; ++i) {
      A_reg.linearLinear(i, i) += reg;
      A_reg.angularAngular(i, i) += reg;
    }

    return A_reg;
  }

  /**
   * @brief Decompose matrix A into L * D * L^T with numerical stability checks
   * @param A Input symmetric positive definite matrix
   * @return true if decomposition succeeded, false if matrix is singular
   */
  PX_FORCE_INLINE bool decompose(const AvbdBlock6x6 &A) {
    // For a 6x6 matrix stored as 4 3x3 blocks:
    // A = [A11 A12; A21 A22] where A21 = A12^T
    // We need to compute L = [L11 0; L21 L22] and D = [D1 0; 0 D2]

    // First, decompose the top-left 3x3 block
    // L11 * D1 * L11^T = A11
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j <= i; ++j) {
        physx::PxReal sum = A.linearLinear(i, j);
        for (int k = 0; k < j; ++k) {
          sum -= L.linearLinear(i, k) * D.linear[k] * L.linearLinear(j, k);
        }
        if (i == j) {
          D.linear[i] = sum;
          if (D.linear[i] <= AvbdConstants::AVBD_LDLT_SINGULAR_THRESHOLD) {
            return false; // Singular matrix
          }
          L.linearLinear(i, j) = 1.0f;
        } else {
          L.linearLinear(i, j) = sum / D.linear[j];
        }
      }
    }

    // Compute L21: A21 = L21 * D1 * L11^T
    // Solving for L21 row by row: L21[i][j] = (A21[i][j] - sum_{k<j} L21[i][k] * D1[k] * L11[j][k]) / D1[j]
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        physx::PxReal sum = A.angularLinear(i, j);
        for (int k = 0; k < j; ++k) {
          sum -= L.angularLinear(i, k) * D.linear[k] * L.linearLinear(j, k);
        }
        L.angularLinear(i, j) = sum / D.linear[j];
      }
    }

    // Compute L22: L22 * D2 * L22^T = A22 - L21 * D1 * L21^T
    AvbdBlock6x6 S; // Schur complement
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        physx::PxReal sum = A.angularAngular(i, j);
        for (int k = 0; k < 3; ++k) {
          sum -= L.angularLinear(i, k) * D.linear[k] * L.angularLinear(j, k);
        }
        S.angularAngular(i, j) = sum;
      }
    }

    // Decompose the Schur complement
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j <= i; ++j) {
        physx::PxReal sum = S.angularAngular(i, j);
        for (int k = 0; k < j; ++k) {
          sum -= L.angularAngular(i, k) * D.angular[k] * L.angularAngular(j, k);
        }
        if (i == j) {
          D.angular[i] = sum;
          if (D.angular[i] <= AvbdConstants::AVBD_LDLT_SINGULAR_THRESHOLD) {
            return false; // Singular matrix
          }
          L.angularAngular(i, j) = 1.0f;
        } else {
          L.angularAngular(i, j) = sum / D.angular[j];
        }
      }
    }

    // L12 = 0 (upper triangular is zero)
    L.linearAngular = physx::PxMat33(physx::PxZero);

    // Compute condition number for numerical stability check
    conditionNumber = computeConditionNumber();

    return true;
  }

  /**
   * @brief Decompose matrix A with automatic regularization for ill-conditioned
   * matrices
   * @param A Input symmetric positive definite matrix
   * @param maxRegAttempts Maximum number of regularization attempts
   * @return true if decomposition succeeded, false if matrix is singular
   */
  PX_FORCE_INLINE bool decomposeWithRegularization(const AvbdBlock6x6 &A,
                                                   int maxRegAttempts = 3) {
    AvbdBlock6x6 A_reg = A;
    physx::PxReal reg = AvbdConstants::AVBD_REGULARIZATION_COEFFICIENT;

    for (int attempt = 0; attempt <= maxRegAttempts; ++attempt) {
      if (decompose(A_reg)) {
        // Check condition number
        if (conditionNumber < AvbdConstants::AVBD_CONDITION_NUMBER_THRESHOLD) {
          return true; // Well-conditioned matrix
        }
        // Ill-conditioned but decomposable - try regularization
        if (attempt < maxRegAttempts) {
          reg *= 10.0f; // Increase regularization
          A_reg = regularizeMatrix(A, reg);
        } else {
          // Last attempt succeeded but still ill-conditioned
          // Accept it but warn (in production, could log this)
          return true;
        }
      } else {
        // Decomposition failed - try regularization
        if (attempt < maxRegAttempts) {
          reg *= 10.0f; // Increase regularization
          A_reg = regularizeMatrix(A, reg);
        } else {
          return false; // Failed even with regularization
        }
      }
    }

    return false;
  }

  /**
   * @brief Solve L * D * L^T * x = b
   * @param b Right-hand side vector
   * @return Solution vector x
   */
  PX_FORCE_INLINE AvbdVec6 solve(const AvbdVec6 &b) const {
    AvbdVec6 y, x;

    // Forward substitution: L * y = b
    for (int i = 0; i < 3; ++i) {
      physx::PxReal sum = b.linear[i];
      for (int j = 0; j < i; ++j) {
        sum -= L.linearLinear(i, j) * y.linear[j];
      }
      y.linear[i] = sum;
    }

    for (int i = 0; i < 3; ++i) {
      physx::PxReal sum = b.angular[i];
      for (int j = 0; j < 3; ++j) {
        sum -= L.angularLinear(i, j) * y.linear[j];
      }
      for (int j = 0; j < i; ++j) {
        sum -= L.angularAngular(i, j) * y.angular[j];
      }
      y.angular[i] = sum;
    }

    // Scale by D: D * z = y => z = D^-1 * y
    for (int i = 0; i < 3; ++i) {
      y.linear[i] /= D.linear[i];
    }
    for (int i = 0; i < 3; ++i) {
      y.angular[i] /= D.angular[i];
    }

    // Backward substitution: L^T * x = z
    for (int i = 2; i >= 0; --i) {
      physx::PxReal sum = y.angular[i];
      for (int j = i + 1; j < 3; ++j) {
        sum -= L.angularAngular(j, i) * x.angular[j];
      }
      x.angular[i] = sum;
    }

    for (int i = 2; i >= 0; --i) {
      physx::PxReal sum = y.linear[i];
      for (int j = i + 1; j < 3; ++j) {
        sum -= L.linearLinear(j, i) * x.linear[j];
      }
      for (int j = 0; j < 3; ++j) {
        sum -= L.angularLinear(j, i) * x.angular[j];
      }
      x.linear[i] = sum;
    }

    return x;
  }

  PX_FORCE_INLINE AvbdLDLT() {
    L.setZero();
    D = AvbdVec6();
    conditionNumber = 0.0f;
  }
} PX_ALIGN_SUFFIX(16);

/**
 * @brief Kahan summation accumulator for deterministic floating-point addition
 *
 * This helps ensure identical results across different platforms by reducing
 * floating-point rounding errors and making the summation order-independent.
 */
struct AvbdKahanAccumulator {
  physx::PxVec3 sum;
  physx::PxVec3 compensation;

  PX_FORCE_INLINE AvbdKahanAccumulator() : sum(0.0f), compensation(0.0f) {}

  PX_FORCE_INLINE void add(const physx::PxVec3 &value) {
    physx::PxVec3 y = value - compensation;
    physx::PxVec3 t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }

  PX_FORCE_INLINE physx::PxVec3 getSum() const { return sum; }
};

/**
 * @brief Pre-computed constraint-to-body mapping for O(1) constraint lookup
 *
 * This structure eliminates O(N^2) complexity in the solver by pre-computing
 * which constraints affect each body. Instead of iterating all constraints
 * for each body, we can directly access only the relevant constraints.
 */
struct AvbdBodyConstraintMap {
  physx::PxU32
      *constraintOffsets; //!< Per-body start offset into constraintIndices
  physx::PxU32 *constraintCounts;  //!< Per-body constraint count
  physx::PxU32 *constraintIndices; //!< Packed array of constraint indices
  physx::PxU32 numBodies;
  physx::PxU32 totalConstraintRefs; //!< Total entries in constraintIndices
  physx::PxU32 capacity;

  PX_FORCE_INLINE AvbdBodyConstraintMap()
      : constraintOffsets(nullptr), constraintCounts(nullptr),
        constraintIndices(nullptr), numBodies(0), totalConstraintRefs(0),
        capacity(0) {}

  /**
   * @brief Build the mapping from constraint array
   * @param numBodiesIn Number of bodies
   * @param numConstraints Number of constraints
   * @param bodyIndicesA Array of bodyIndexA for each constraint
   * @param bodyIndicesB Array of bodyIndexB for each constraint
   * @param allocator Allocator for memory
   */
  template <typename ConstraintType>
  void build(physx::PxU32 numBodiesIn, const ConstraintType *constraints,
             physx::PxU32 numConstraints,
             physx::PxAllocatorCallback &allocator) {

    // Release old data if any
    if (constraintOffsets) {
      release(allocator);
    }

    numBodies = numBodiesIn;

    // Allocate count array
    constraintCounts = static_cast<physx::PxU32 *>(allocator.allocate(
        sizeof(physx::PxU32) * numBodies, "AvbdBodyConstraintMap::counts",
        __FILE__, __LINE__));

    // First pass: count constraints per body
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      constraintCounts[i] = 0;
    }

    for (physx::PxU32 c = 0; c < numConstraints; ++c) {
      physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
      physx::PxU32 bodyB = constraints[c].header.bodyIndexB;
      if (bodyA < numBodies)
        constraintCounts[bodyA]++;
      if (bodyB < numBodies)
        constraintCounts[bodyB]++;
    }

    // Compute offsets (prefix sum)
    constraintOffsets = static_cast<physx::PxU32 *>(allocator.allocate(
        sizeof(physx::PxU32) * (numBodies + 1),
        "AvbdBodyConstraintMap::offsets", __FILE__, __LINE__));

    constraintOffsets[0] = 0;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      constraintOffsets[i + 1] = constraintOffsets[i] + constraintCounts[i];
    }
    totalConstraintRefs = constraintOffsets[numBodies];

    // Allocate constraint indices array
    if (totalConstraintRefs > 0) {
      constraintIndices = static_cast<physx::PxU32 *>(allocator.allocate(
          sizeof(physx::PxU32) * totalConstraintRefs,
          "AvbdBodyConstraintMap::indices", __FILE__, __LINE__));
    }

    // Reset counts for second pass
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      constraintCounts[i] = 0;
    }

    // Second pass: fill constraint indices
    for (physx::PxU32 c = 0; c < numConstraints; ++c) {
      physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
      physx::PxU32 bodyB = constraints[c].header.bodyIndexB;

      if (bodyA < numBodies) {
        physx::PxU32 idx = constraintOffsets[bodyA] + constraintCounts[bodyA];
        constraintIndices[idx] = c;
        constraintCounts[bodyA]++;
      }
      if (bodyB < numBodies) {
        physx::PxU32 idx = constraintOffsets[bodyB] + constraintCounts[bodyB];
        constraintIndices[idx] = c;
        constraintCounts[bodyB]++;
      }
    }

    capacity = numBodies;
  }

  /**
   * @brief Get constraints for a specific body
   * @param bodyIndex Body index
   * @param outIndices Output pointer to constraint indices
   * @param outCount Output count of constraints
   */
  PX_FORCE_INLINE void getBodyConstraints(physx::PxU32 bodyIndex,
                                          const physx::PxU32 *&outIndices,
                                          physx::PxU32 &outCount) const {
    // Safety check: ensure all required pointers are valid
    if (constraintOffsets && constraintCounts && constraintIndices &&
        bodyIndex < numBodies) {
      outIndices = constraintIndices + constraintOffsets[bodyIndex];
      outCount = constraintCounts[bodyIndex];
    } else {
      outIndices = nullptr;
      outCount = 0;
    }
  }

  /**
   * @brief Release allocated memory
   */
  void release(physx::PxAllocatorCallback &allocator) {
    if (constraintOffsets) {
      allocator.deallocate(constraintOffsets);
      constraintOffsets = nullptr;
    }
    if (constraintCounts) {
      allocator.deallocate(constraintCounts);
      constraintCounts = nullptr;
    }
    if (constraintIndices) {
      allocator.deallocate(constraintIndices);
      constraintIndices = nullptr;
    }
    numBodies = 0;
    totalConstraintRefs = 0;
    capacity = 0;
  }
};

} // namespace Dy

} // namespace physx

#endif // DY_AVBD_TYPES_H
