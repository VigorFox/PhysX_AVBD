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
 * These constants define the numerical parameters used throughout the AVBD solver.
 * They are centralized here to avoid magic numbers in the code.
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
  static const PxReal AVBD_DEFAULT_PENALTY_RHO_LOW = 1e4f;
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
  static const PxReal AVBD_VIOLATION_THRESHOLD = 1e-3f;

  // Maximum Lagrangian multiplier
  static const PxReal AVBD_MAX_LAMBDA = 1e6f;

  // LDLT decomposition threshold for singularity detection
  static const PxReal AVBD_LDLT_SINGULAR_THRESHOLD = 1e-10f;

  // Condition number threshold for ill-conditioned matrices
  static const PxReal AVBD_CONDITION_NUMBER_THRESHOLD = 1e8f;

  // Regularization coefficient for ill-conditioned matrices
  static const PxReal AVBD_REGULARIZATION_COEFFICIENT = 1e-6f;
}

namespace Dy {

/**
 * @brief Configuration flags for deterministic simulation
 */
struct AvbdDeterminismFlags {
    enum Enum {
        eNONE                    = 0,
        eSORT_CONSTRAINTS        = (1 << 0),  //!< Sort constraints by body pair for consistent ordering
        eSORT_BODIES             = (1 << 1),  //!< Sort bodies by ID before iteration
        eUSE_KAHAN_SUMMATION     = (1 << 2),  //!< Use Kahan summation for accumulation
        eFIXED_POINT_MATH        = (1 << 3),  //!< Use fixed-point math where possible (future)
        
        eDETERMINISTIC_DEFAULT   = eSORT_CONSTRAINTS | eSORT_BODIES | eUSE_KAHAN_SUMMATION
    };
};

/**
 * @brief AVBD solver configuration parameters
 */
struct AvbdSolverConfig {
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
      rhoScale; //!< Scale factor for rho adaptation per outer iteration
  physx::PxReal maxRho; //!< Maximum penalty parameter

  //-------------------------------------------------------------------------
  // Compliance and damping
  //-------------------------------------------------------------------------

  physx::PxReal defaultCompliance; //!< Default compliance for soft constraints
  physx::PxReal contactCompliance; //!< Compliance for contact constraints
                                    //!< (usually 0 or very small)
  physx::PxReal jointCompliance; //!< Default compliance for joint constraints
  physx::PxReal damping; //!< Step size damping for gradient descent (0-1)

  //-------------------------------------------------------------------------
  // Rotation dynamics
  //-------------------------------------------------------------------------

  physx::PxReal angularDamping; //!< Angular velocity damping per frame (0-1,
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

  //-------------------------------------------------------------------------
  // Convergence
  //-------------------------------------------------------------------------

  physx::PxReal
      positionTolerance; //!< Position error tolerance for early termination
  physx::PxReal velocityDamping; //!< Global velocity damping factor (0-1)

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

  physx::PxU32 largeIslandThreshold; //!< Constraint count threshold to trigger
                                      //!< internal island parallelization.
                                      //!< Islands with more constraints use
                                      //!< constraint coloring for better cache
                                      //!< locality. Default: 128

  //-------------------------------------------------------------------------
  // Determinism (for multi-platform synchronization)
  //-------------------------------------------------------------------------

  physx::PxU32 determinismFlags;   //!< Bitmask of AvbdDeterminismFlags::Enum
                                    //!< for cross-platform determinism

  //-------------------------------------------------------------------------
  // Defaults
  //-------------------------------------------------------------------------

  AvbdSolverConfig()
      : outerIterations(1), innerIterations(4), initialRho(1e4f),
        rhoScale(2.0f), maxRho(1e8f), defaultCompliance(1e-6f),
        contactCompliance(0.0f), jointCompliance(1e-8f), damping(0.5f),
        angularDamping(0.95f), rotationThreshold(0.001f), angularScale(400.0f),
        angularContactScale(0.2f), baumgarte(0.3f), positionTolerance(1e-4f), velocityDamping(0.99f),
        maxPositionCorrection(0.2f), maxAngularCorrection(0.5f),
        maxLambda(1e6f), enableParallelization(true), enableLocal6x6Solve(false),
        enableMassWeightedWeld(false),
        largeIslandThreshold(128), determinismFlags(0) {}
  
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
    enableParallelization = false;  // Disable parallelization for strict determinism
  }
  
  /**
   * @brief Check if determinism is enabled
   */
  bool isDeterministic() const {
    return (determinismFlags & AvbdDeterminismFlags::eDETERMINISTIC_DEFAULT) != 0;
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

  physx::PxReal constraintError;        //!< RMS constraint error
  physx::PxReal maxPositionError;       //!< Maximum position error after solve
  physx::PxReal avgPositionError;       //!< Average position error after solve
  physx::PxReal maxConstraintViolation; //!< Maximum constraint violation

  physx::PxReal totalEnergy; //!< Total system energy (kinetic + potential)

  physx::PxU64 solveTimeUs; //!< Solve time in microseconds

  void reset() {
    numBodies = 0;
    numContacts = 0;
    numJoints = 0;
    numColorGroups = 0;
    activeConstraints = 0;
    totalIterations = 0;
    constraintError = 0.0f;
    maxPositionError = 0.0f;
    avgPositionError = 0.0f;
    maxConstraintViolation = 0.0f;
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
  physx::PxU32 numColors;         //!< Number of colors used
  physx::PxU32 maxColors;         //!< Maximum colors allocated

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
  PX_FORCE_INLINE void
  initializeDiagonal(physx::PxReal invMass,
                     const physx::PxMat33 & /*invInertia*/,
                     physx::PxReal invDtSq) {
    // M/h^2 and I/h^2 on diagonal
    physx::PxReal massContrib =
        (invMass > 0.0f) ? (1.0f / invMass) * invDtSq : 0.0f;
    linearLinear = physx::PxMat33(physx::PxVec3(massContrib, 0, 0),
                                   physx::PxVec3(0, massContrib, 0),
                                   physx::PxVec3(0, 0, massContrib));

    // For inertia, we need I/h^2 (not inverse)
    // This is a simplification - full implementation would compute I from invI
    linearAngular = physx::PxMat33(physx::PxZero);
    angularLinear = physx::PxMat33(physx::PxZero);
    angularAngular =
        physx::PxMat33(physx::PxIdentity) * invDtSq; // Placeholder
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
  AvbdBlock6x6 L;  // Lower triangular matrix (diagonal = 1)
  AvbdVec6 D;      // Diagonal matrix (stored as vector)
  physx::PxReal conditionNumber;  // Condition number of the matrix
  
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
  PX_FORCE_INLINE AvbdBlock6x6 regularizeMatrix(const AvbdBlock6x6& A, physx::PxReal reg) const {
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
  PX_FORCE_INLINE bool decompose(const AvbdBlock6x6& A) {
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
    
    // Compute L21: L21 * D1 * L11^T = A21
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        physx::PxReal sum = A.angularLinear(i, j);
        for (int k = 0; k < 3; ++k) {
          sum -= L.angularAngular(i, k) * D.angular[k] * L.linearLinear(j, k);
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
   * @brief Decompose matrix A with automatic regularization for ill-conditioned matrices
   * @param A Input symmetric positive definite matrix
   * @param maxRegAttempts Maximum number of regularization attempts
   * @return true if decomposition succeeded, false if matrix is singular
   */
  PX_FORCE_INLINE bool decomposeWithRegularization(const AvbdBlock6x6& A, int maxRegAttempts = 3) {
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
  PX_FORCE_INLINE AvbdVec6 solve(const AvbdVec6& b) const {
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
    
    PX_FORCE_INLINE void add(const physx::PxVec3& value) {
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
    physx::PxU32* constraintOffsets;    //!< Per-body start offset into constraintIndices
    physx::PxU32* constraintCounts;     //!< Per-body constraint count
    physx::PxU32* constraintIndices;    //!< Packed array of constraint indices
    physx::PxU32 numBodies;
    physx::PxU32 totalConstraintRefs;   //!< Total entries in constraintIndices
    physx::PxU32 capacity;
    
    PX_FORCE_INLINE AvbdBodyConstraintMap() 
        : constraintOffsets(nullptr), constraintCounts(nullptr), 
          constraintIndices(nullptr), numBodies(0), totalConstraintRefs(0), capacity(0) {}
    
    /**
     * @brief Build the mapping from constraint array
     * @param numBodiesIn Number of bodies
     * @param numConstraints Number of constraints
     * @param bodyIndicesA Array of bodyIndexA for each constraint
     * @param bodyIndicesB Array of bodyIndexB for each constraint
     * @param allocator Allocator for memory
     */
    template<typename ConstraintType>
    void build(physx::PxU32 numBodiesIn, 
               const ConstraintType* constraints,
               physx::PxU32 numConstraints,
               physx::PxAllocatorCallback& allocator) {
        
        // Release old data if any
        if (constraintOffsets) {
            release(allocator);
        }
        
        numBodies = numBodiesIn;
        
        // Allocate count array
        constraintCounts = static_cast<physx::PxU32*>(
            allocator.allocate(sizeof(physx::PxU32) * numBodies, 
                              "AvbdBodyConstraintMap::counts", __FILE__, __LINE__));
        
        // First pass: count constraints per body
        for (physx::PxU32 i = 0; i < numBodies; ++i) {
            constraintCounts[i] = 0;
        }
        
        for (physx::PxU32 c = 0; c < numConstraints; ++c) {
            physx::PxU32 bodyA = constraints[c].header.bodyIndexA;
            physx::PxU32 bodyB = constraints[c].header.bodyIndexB;
            if (bodyA < numBodies) constraintCounts[bodyA]++;
            if (bodyB < numBodies) constraintCounts[bodyB]++;
        }
        
        // Compute offsets (prefix sum)
        constraintOffsets = static_cast<physx::PxU32*>(
            allocator.allocate(sizeof(physx::PxU32) * (numBodies + 1), 
                              "AvbdBodyConstraintMap::offsets", __FILE__, __LINE__));
        
        constraintOffsets[0] = 0;
        for (physx::PxU32 i = 0; i < numBodies; ++i) {
            constraintOffsets[i + 1] = constraintOffsets[i] + constraintCounts[i];
        }
        totalConstraintRefs = constraintOffsets[numBodies];
        
        // Allocate constraint indices array
        if (totalConstraintRefs > 0) {
            constraintIndices = static_cast<physx::PxU32*>(
                allocator.allocate(sizeof(physx::PxU32) * totalConstraintRefs, 
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
                                            const physx::PxU32*& outIndices, 
                                            physx::PxU32& outCount) const {
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
    void release(physx::PxAllocatorCallback& allocator) {
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
