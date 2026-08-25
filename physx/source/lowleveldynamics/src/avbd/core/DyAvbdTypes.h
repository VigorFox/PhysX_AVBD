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

#if PX_X64
#include <emmintrin.h>
#endif

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

// Shared augmented-Lagrangian profile. Alpha and gamma track the current
// avbd-demo3d defaults. The linear beta and penalty bounds retain PhysX's
// established unit scale until impact and joint regressions justify changing
// them. Contact preparation, rigid/soft solves and persistent joint state must
// consume this one cross-frame stabilization contract.
static const PxReal AVBD_AL_ALPHA = 0.99f;
static const PxReal AVBD_AL_BETA_LINEAR = 1000.0f;
static const PxReal AVBD_AL_GAMMA = 0.999f;
static const PxReal AVBD_AL_PENALTY_MIN = 1000.0f;
static const PxReal AVBD_AL_PENALTY_MAX = 1e9f;

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

// Deformable shell fallback and local primal safety-net constants.
static const PxReal AVBD_PEN_SCALE_BODY_VS_STATIC = 2.0f;
static const PxReal AVBD_CONTACT_BOOST_FRACTION = 0.005f;
// Surface-motion alias reject + friction ride caps (solve-loop contract Phase 4)
static const PxReal AVBD_SURFACE_STEP_ALIAS_M = 0.5f;
static const PxReal AVBD_SURFACE_VMESH_CAP = 8.0f;
static const PxReal AVBD_BODY_STATIC_FAST_IMPACT_SPEED = 60.0f;
static const PxReal AVBD_SHELL_FAST_IMPACT_SPEED = 60.0f;
static const PxReal AVBD_BODY_STATIC_NEAR_SURFACE = 0.05f;
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

  //! Number of complete AVBD iterations. Each iteration performs the primal
  //! body sweep and its dual-variable/stiffness updates.
  physx::PxU32 iterations;

  //! Minimum complete-iteration budget and early-stop floor for islands
  //! containing D6 or gear joints. Zero disables the joint-specific override.
  physx::PxU32 jointIterationOverride;

  //! Allow converged rigid-only AVBD islands to stop before exhausting their
  //! effective complete-iteration budget.
  bool enableEarlyStop;

  //-------------------------------------------------------------------------
  // Augmented Lagrangian parameters
  //-------------------------------------------------------------------------

  physx::PxReal initialRho; //!< Initial penalty parameter for ALM
  physx::PxReal
      rhoScale;         //!< Scale factor for rho adaptation per AVBD iteration
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

  physx::PxReal
      positionTolerance; //!< Pose-delta threshold for early termination
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

  // Select the legacy ordered body sweep as the scene's reproducibility
  // authority.  This is deliberately separate from determinismFlags: the
  // latter enables an older, still separately-audited sort/Kahan experiment,
  // while PxSceneFlag::eENABLE_ENHANCED_DETERMINISM must preserve the current
  // ordered solver trajectory instead of silently changing its math.
  bool useOrderedBackend;

  bool enableLocal6x6Solve; //!< Use 6x6 local system solve in block descent
                            //!< (fallback to Gauss-Seidel when false)

  bool enableMassWeightedWeld; //!< Use mass-ratio weighted corrections for weld
                               //!< joints (runtime attachment stability)

  bool enableStageOwnershipDiagnostics; //!< Collect per-contact post-stage
                                        //!< ownership masks for opt-in gates
  bool enableBoundedComponentProductionProbe; //!< Opt-in complete-component
                                              //!< P3K owner replacement probe

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
      : lengthScale(1.0f), iterations(4), jointIterationOverride(8),
        enableEarlyStop(true), initialRho(1e4f),
        rhoScale(2.0f), maxRho(1e8f), defaultCompliance(1e-6f),
        contactCompliance(1e-4f), jointCompliance(1e-8f), contactDamping(0.5f),
        damping(0.5f), angularDamping(0.95f), rotationThreshold(0.001f),
        angularScale(400.0f), angularContactScale(0.2f), baumgarte(0.3f),
        chebyshevRho(0.92f), positionTolerance(1e-4f),
        velocityDamping(0.99f),
        bounceThresholdVelocity(-2.0f),
        maxPositionCorrection(0.2f), maxAngularCorrection(0.5f),
        maxLambda(1e6f), enableParallelization(true),
        useOrderedBackend(false),
        enableLocal6x6Solve(false), enableMassWeightedWeld(false),
        enableStageOwnershipDiagnostics(false),
        enableBoundedComponentProductionProbe(false),
        largeIslandThreshold(128),
        avbdAlpha(AvbdConstants::AVBD_AL_ALPHA),
        avbdBeta(AvbdConstants::AVBD_AL_BETA_LINEAR),
        avbdGamma(AvbdConstants::AVBD_AL_GAMMA),
        avbdPenaltyMin(AvbdConstants::AVBD_AL_PENALTY_MIN),
        avbdPenaltyMax(AvbdConstants::AVBD_AL_PENALTY_MAX),
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

  /** Select the current ordered scalar/GS backend without changing its math. */
  void enableOrderedBackend() {
    useOrderedBackend = true;
    enableParallelization = false;
  }

  /**
   * @brief Check if determinism is enabled
   */
  bool isDeterministic() const {
    return (determinismFlags & AvbdDeterminismFlags::eDETERMINISTIC_DEFAULT) !=
           0;
  }

  /** True when task scheduling must retain the ordered solver authority. */
  bool requiresOrderedBackend() const {
    return useOrderedBackend || isDeterministic();
  }
};

/**
 * @brief Small cold-path AVBD solver profile payload.
 *
 * The default build uses the empty sink below.  Keep this payload limited to
 * workload, deformable-contact ownership, and finalization health signals;
 * per-row classifications and oracle/probe detail do not belong in the
 * solver/task hot path.
 */
// Full solver telemetry is an explicit cold/profile payload. The default
// solver build uses an empty sink so island tasks do not carry or clear
// diagnostic state in the hot path.
#ifndef PX_AVBD_ENABLE_SOLVER_PROFILE
#define PX_AVBD_ENABLE_SOLVER_PROFILE 0
#endif

#if PX_AVBD_ENABLE_SOLVER_PROFILE
#define PX_AVBD_PROFILE_STAT(...)                                           \
  do {                                                                      \
    __VA_ARGS__;                                                            \
  } while (false)
#else
#define PX_AVBD_PROFILE_STAT(...)                                           \
  do {                                                                      \
  } while (false)
#endif

struct AvbdSolverStatsPayload {
  // Workload counters used to correlate solver cost with deformable contact
  // topology. Keep this list intentionally small; detailed probes are retired.
  physx::PxU32 numBodies;
  physx::PxU32 numContacts;
  physx::PxU32 numJoints;
  physx::PxU32 totalIterations;

  physx::PxU32 surfaceDeformableAlRows;
  physx::PxU32 surfaceDeformablePositionTangentRows;
  physx::PxU32 surfaceDeformableDepenetrationCorrections;
  physx::PxU32 surfaceDeformableFrictionCorrections;

  // Finalization health: invocation, applied correction, and bounded-shadow
  // outcomes are the only retained signals needed for correctness triage.
  physx::PxU32 surfaceDeformableFinalizeBodies;
  physx::PxU32 surfaceDeformableFinalizeCorrections;
  physx::PxU32 surfaceDeformableFinalizeShadowComponents;
  physx::PxU32 surfaceDeformableFinalizeShadowSolved;
  physx::PxU32 surfaceDeformableFinalizeShadowUnsupported;
  physx::PxU32 surfaceDeformableFinalizeShadowIterationLimit;
  physx::PxU32 surfaceDeformableFinalizeProbeCommittedComponents;
  physx::PxU32 surfaceDeformableFinalizeContactReversalCorrections;

  physx::PxReal constraintError;
  physx::PxReal surfaceDeformableFinalizeDelta;

  PX_FORCE_INLINE void reset() { *this = AvbdSolverStatsPayload(); }
};
static_assert(sizeof(AvbdSolverStatsPayload) <= 80,
              "AVBD profile payload must remain a small cold-path record");

#if PX_AVBD_ENABLE_SOLVER_PROFILE
using AvbdSolverStats = AvbdSolverStatsPayload;
#else
struct AvbdSolverStats {
  PX_FORCE_INLINE void reset() {}
};
static_assert(sizeof(AvbdSolverStats) <= 1,
              "default AVBD stats sink must remain empty");
#endif

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
#if PX_X64
    const __m128 scale = _mm_set1_ps(invCompliance);
    addConstraintContributionSse2(linearLinear, gradPos, gradPos, scale,
                                  invCompliance);
    addConstraintContributionSse2(linearAngular, gradPos, gradRot, scale,
                                  invCompliance);
    addConstraintContributionSse2(angularLinear, gradRot, gradPos, scale,
                                  invCompliance);
    addConstraintContributionSse2(angularAngular, gradRot, gradRot, scale,
                                  invCompliance);
#else
    // H += invCompliance * grad * grad^T
    for (physx::PxU32 i = 0; i < 3; ++i) {
      for (physx::PxU32 j = 0; j < 3; ++j) {
        linearLinear(i, j) += invCompliance * gradPos[i] * gradPos[j];
        linearAngular(i, j) += invCompliance * gradPos[i] * gradRot[j];
        angularLinear(i, j) += invCompliance * gradRot[i] * gradPos[j];
        angularAngular(i, j) += invCompliance * gradRot[i] * gradRot[j];
      }
    }
#endif
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
    if (linearScale == 1.0f && angularScale == 1.0f) {
      addConstraintContribution(gradPos, gradRot, invCompliance);
      return;
    }
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

#if PX_X64
  PX_FORCE_INLINE static void addConstraintContributionSse2(
      physx::PxMat33 &matrix, const physx::PxVec3 &row,
      const physx::PxVec3 &column, const __m128 scale,
      physx::PxReal scalarScale) {
    // PxMat33 is column-major and contains exactly nine contiguous PxReal
    // values.  The first eight values can be updated as two SSE2 vectors;
    // the final (column2.z) element stays scalar so no adjacent matrix bytes
    // are read or written.  Each lane retains the scalar order
    // (invCompliance * row[i]) * column[j].
    const __m128 row01 = _mm_set_ps(row.x, row.z, row.y, row.x);
    const __m128 column01 =
        _mm_set_ps(column.y, column.x, column.x, column.x);
    const __m128 row12 = _mm_set_ps(row.y, row.x, row.z, row.y);
    const __m128 column12 =
        _mm_set_ps(column.z, column.z, column.y, column.y);
    const __m128 delta01 = _mm_mul_ps(_mm_mul_ps(scale, row01), column01);
    const __m128 delta12 = _mm_mul_ps(_mm_mul_ps(scale, row12), column12);
    physx::PxReal *values = &matrix.column0.x;
    _mm_storeu_ps(values,
                  _mm_add_ps(_mm_loadu_ps(values), delta01));
    _mm_storeu_ps(values + 4,
                  _mm_add_ps(_mm_loadu_ps(values + 4), delta12));
    values[8] += scalarScale * row.z * column.z;
  }
#endif

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
   * @brief Build the mapping into caller-owned, preallocated storage.
   *
   * The storage must be reserved before a task is submitted.  This keeps the
   * preparation path allocation-free once island work is dispatched and lets
   * a future task own one disjoint map slot without touching a shared heap.
   * On failure the map remains empty; the caller may release the supplied
   * buffers and use build() as the serial fallback.
   */
  template <typename ConstraintType>
  bool buildInPlace(physx::PxU32 numBodiesIn,
                    const ConstraintType *constraints,
                    physx::PxU32 numConstraints,
                    physx::PxU32 *counts, physx::PxU32 *offsets,
                    physx::PxU32 *indices,
                    physx::PxU32 indexCapacity) {
    if (constraintOffsets || constraintCounts || constraintIndices ||
        !counts || !offsets || (indexCapacity > 0 && !indices))
      return false;

    for (physx::PxU32 i = 0; i < numBodiesIn; ++i)
      counts[i] = 0;

    for (physx::PxU32 c = 0; c < numConstraints; ++c) {
      const physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
      const physx::PxU32 bodyB = constraints[c].header.bodyIndexB;
      if (bodyA < numBodiesIn)
        ++counts[bodyA];
      if (bodyB < numBodiesIn)
        ++counts[bodyB];
    }

    offsets[0] = 0;
    for (physx::PxU32 i = 0; i < numBodiesIn; ++i) {
      if (counts[i] > PX_MAX_U32 - offsets[i])
        return false;
      offsets[i + 1] = offsets[i] + counts[i];
    }
    const physx::PxU32 totalRefs = offsets[numBodiesIn];
    if (totalRefs > indexCapacity || (totalRefs > 0 && !indices))
      return false;

    // Preserve the same packed order as build(): counts become write cursors
    // for the second pass and are final per-body counts after the fill.
    for (physx::PxU32 i = 0; i < numBodiesIn; ++i)
      counts[i] = 0;
    for (physx::PxU32 c = 0; c < numConstraints; ++c) {
      const physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
      const physx::PxU32 bodyB = constraints[c].header.bodyIndexB;
      if (bodyA < numBodiesIn) {
        const physx::PxU32 idx = offsets[bodyA] + counts[bodyA];
        indices[idx] = c;
        ++counts[bodyA];
      }
      if (bodyB < numBodiesIn) {
        const physx::PxU32 idx = offsets[bodyB] + counts[bodyB];
        indices[idx] = c;
        ++counts[bodyB];
      }
    }

    constraintOffsets = offsets;
    constraintCounts = counts;
    constraintIndices = indices;
    numBodies = numBodiesIn;
    totalConstraintRefs = totalRefs;
    capacity = numBodiesIn;
    return true;
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
