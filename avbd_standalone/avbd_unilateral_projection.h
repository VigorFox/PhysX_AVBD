#pragma once

#include "avbd_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace AvbdRef {

struct UnilateralProjectionRow {
  Vec3 linearJacobian;
  Vec3 angularJacobian;
  float outwardVelocity = 0.0f;
  uint64_t stableKey = 0;
};

enum class UnilateralProjectionStatus {
  Solved,
  NoCorrection,
  Infeasible,
  NumericalFailure,
  IterationLimit
};

struct UnilateralProjectionResult {
  UnilateralProjectionStatus status =
      UnilateralProjectionStatus::NumericalFailure;
  std::vector<float> impulses;
  Vec6 velocityDelta;
  int activeRows = 0;
  int iterations = 0;
  int multiplierRemovals = 0;
  int dependentPivots = 0;
  float maxResidual = 0.0f;
};

namespace UnilateralProjectionDetail {

struct OrderedRow {
  UnilateralProjectionRow row;
  size_t originalIndex = 0;
};

inline bool finiteVec3(const Vec3 &v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline bool rowLess(const OrderedRow &a, const OrderedRow &b) {
  if (a.row.stableKey != b.row.stableKey)
    return a.row.stableKey < b.row.stableKey;
  const float av[7] = {
      a.row.linearJacobian.x,  a.row.linearJacobian.y,
      a.row.linearJacobian.z,  a.row.angularJacobian.x,
      a.row.angularJacobian.y, a.row.angularJacobian.z,
      a.row.outwardVelocity};
  const float bv[7] = {
      b.row.linearJacobian.x,  b.row.linearJacobian.y,
      b.row.linearJacobian.z,  b.row.angularJacobian.x,
      b.row.angularJacobian.y, b.row.angularJacobian.z,
      b.row.outwardVelocity};
  for (int i = 0; i < 7; ++i) {
    if (av[i] != bv[i])
      return av[i] < bv[i];
  }
  // Exact duplicate rows are physically interchangeable. The original index
  // is only a strict-weak-order fallback; it cannot change the aggregate
  // generalized impulse.
  return a.originalIndex < b.originalIndex;
}

inline double response(const UnilateralProjectionRow &a,
                       const UnilateralProjectionRow &b,
                       float inverseMassResponse,
                       const Mat33 &inverseInertiaResponse) {
  const Vec3 angularResponse =
      inverseInertiaResponse * b.angularJacobian;
  return static_cast<double>(inverseMassResponse) *
             static_cast<double>(a.linearJacobian.dot(b.linearJacobian)) +
         static_cast<double>(a.angularJacobian.dot(angularResponse));
}

inline bool solveDense(const double matrix[6][6], const double rhs[6],
                       int count, double relativeTolerance,
                       double solution[6]) {
  double augmented[6][7] = {};
  double scale = 1.0;
  for (int row = 0; row < count; ++row) {
    for (int column = 0; column < count; ++column) {
      augmented[row][column] = matrix[row][column];
      scale = std::max(scale, std::fabs(matrix[row][column]));
    }
    augmented[row][count] = rhs[row];
  }

  const double pivotTolerance = relativeTolerance * scale;
  for (int column = 0; column < count; ++column) {
    int pivot = column;
    double pivotMagnitude = std::fabs(augmented[column][column]);
    for (int row = column + 1; row < count; ++row) {
      const double candidate = std::fabs(augmented[row][column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!std::isfinite(pivotMagnitude) ||
        pivotMagnitude <= pivotTolerance)
      return false;
    if (pivot != column) {
      for (int entry = column; entry <= count; ++entry)
        std::swap(augmented[column][entry], augmented[pivot][entry]);
    }

    const double inversePivot = 1.0 / augmented[column][column];
    for (int entry = column; entry <= count; ++entry)
      augmented[column][entry] *= inversePivot;
    for (int row = 0; row < count; ++row) {
      if (row == column)
        continue;
      const double factor = augmented[row][column];
      for (int entry = column; entry <= count; ++entry)
        augmented[row][entry] -= factor * augmented[column][entry];
    }
  }

  for (int row = 0; row < count; ++row) {
    solution[row] = augmented[row][count];
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

inline int activePosition(const std::vector<int> &active, int row) {
  for (size_t i = 0; i < active.size(); ++i) {
    if (active[i] == row)
      return static_cast<int>(i);
  }
  return -1;
}

} // namespace UnilateralProjectionDetail

/**
 * Solve the minimum kinetic-energy correction for
 *
 *   outwardVelocity_i + J_i * velocityDelta <= 0
 *
 * with a single rigid body's inverse mass/inertia response. The dual is the
 * monotone LCP
 *
 *   impulse >= 0
 *   W * impulse - outwardVelocity >= 0
 *   impulse' * (W * impulse - outwardVelocity) = 0
 *
 * where W = J M^-1 J'. The working set contains at most six independent rows,
 * but every input inequality participates in residual selection and final
 * verification.
 */
inline UnilateralProjectionResult solveBodyUnilateralProjection(
    const std::vector<UnilateralProjectionRow> &inputRows,
    float inverseMassResponse, const Mat33 &inverseInertiaResponse,
    float relativeTolerance = 1.0e-6f) {
  using namespace UnilateralProjectionDetail;

  UnilateralProjectionResult result;
  result.impulses.assign(inputRows.size(), 0.0f);
  if (!std::isfinite(inverseMassResponse) ||
      inverseMassResponse < 0.0f || relativeTolerance <= 0.0f ||
      !std::isfinite(relativeTolerance)) {
    return result;
  }

  std::vector<OrderedRow> ordered;
  ordered.reserve(inputRows.size());
  double velocityScale = 1.0;
  bool needsCorrection = false;
  for (size_t i = 0; i < inputRows.size(); ++i) {
    const UnilateralProjectionRow &row = inputRows[i];
    if (!finiteVec3(row.linearJacobian) ||
        !finiteVec3(row.angularJacobian) ||
        !std::isfinite(row.outwardVelocity)) {
      return result;
    }
    ordered.push_back({row, i});
    velocityScale =
        std::max(velocityScale,
                 std::fabs(static_cast<double>(row.outwardVelocity)));
    needsCorrection = needsCorrection || row.outwardVelocity > 0.0f;
  }
  if (!needsCorrection || ordered.empty()) {
    result.status = UnilateralProjectionStatus::NoCorrection;
    return result;
  }
  std::sort(ordered.begin(), ordered.end(), rowLess);

  const int rowCount = static_cast<int>(ordered.size());
  std::vector<double> lambda(static_cast<size_t>(rowCount), 0.0);
  std::vector<int> active;
  active.reserve(6);
  const double feasibilityTolerance =
      static_cast<double>(relativeTolerance) * velocityScale;
  const double multiplierTolerance =
      static_cast<double>(relativeTolerance) * 0.1;
  // Rows originate as float contact geometry. A nominally exact spatial
  // dependency (for example 1/sqrt(2) directions) carries O(1e-8) Gram
  // error before the double-precision working-set algebra begins.
  const double rankTolerance = 1.0e-7;
  const int iterationLimit = 64 + 32 * rowCount;

  const auto matrixEntry = [&](int row, int column) {
    return response(ordered[static_cast<size_t>(row)].row,
                    ordered[static_cast<size_t>(column)].row,
                    inverseMassResponse, inverseInertiaResponse);
  };
  const auto residual = [&](int row) {
    double value =
        static_cast<double>(
            ordered[static_cast<size_t>(row)].row.outwardVelocity);
    for (int activeRow : active)
      value -= matrixEntry(row, activeRow) *
               lambda[static_cast<size_t>(activeRow)];
    return value;
  };

  while (result.iterations < iterationLimit) {
    ++result.iterations;
    int candidate = -1;
    double maximumViolation = feasibilityTolerance;
    for (int row = 0; row < rowCount; ++row) {
      if (activePosition(active, row) >= 0)
        continue;
      const double rowResidual = residual(row);
      if (rowResidual > maximumViolation) {
        candidate = row;
        maximumViolation = rowResidual;
      }
    }

    if (candidate < 0) {
      double maximumResidual = 0.0;
      for (int row = 0; row < rowCount; ++row)
        maximumResidual = std::max(maximumResidual, residual(row));
      result.maxResidual = static_cast<float>(maximumResidual);
      if (!std::isfinite(maximumResidual) ||
          maximumResidual > 4.0 * feasibilityTolerance) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }

      Vec3 linearImpulse(0.0f, 0.0f, 0.0f);
      Vec3 angularImpulse(0.0f, 0.0f, 0.0f);
      for (int row = 0; row < rowCount; ++row) {
        const double rowLambda = lambda[static_cast<size_t>(row)];
        if (rowLambda < -multiplierTolerance ||
            !std::isfinite(rowLambda)) {
          result.status = UnilateralProjectionStatus::NumericalFailure;
          return result;
        }
        const float impulse =
            static_cast<float>(std::max(0.0, rowLambda));
        const OrderedRow &orderedRow =
            ordered[static_cast<size_t>(row)];
        result.impulses[orderedRow.originalIndex] = impulse;
        linearImpulse += orderedRow.row.linearJacobian * impulse;
        angularImpulse += orderedRow.row.angularJacobian * impulse;
      }
      const Vec3 linearDelta =
          linearImpulse * (-inverseMassResponse);
      const Vec3 angularDelta =
          (inverseInertiaResponse * angularImpulse) * -1.0f;
      result.velocityDelta = Vec6(linearDelta, angularDelta);
      result.activeRows = static_cast<int>(active.size());
      result.status = UnilateralProjectionStatus::Solved;
      return result;
    }

    bool candidateAdded = false;
    if (active.empty()) {
      const double diagonal = matrixEntry(candidate, candidate);
      if (!std::isfinite(diagonal) || diagonal <= rankTolerance) {
        result.status = UnilateralProjectionStatus::Infeasible;
        return result;
      }
      active.push_back(candidate);
      candidateAdded = true;
    } else {
      const int activeCount = static_cast<int>(active.size());
      double matrix[6][6] = {};
      double cross[6] = {};
      double coefficients[6] = {};
      double diagonalScale =
          std::max(1.0, std::fabs(matrixEntry(candidate, candidate)));
      for (int row = 0; row < activeCount; ++row) {
        cross[row] = matrixEntry(active[row], candidate);
        for (int column = 0; column < activeCount; ++column) {
          matrix[row][column] =
              matrixEntry(active[row], active[column]);
          diagonalScale =
              std::max(diagonalScale,
                       std::fabs(matrix[row][column]));
        }
      }
      if (!solveDense(matrix, cross, activeCount, rankTolerance,
                      coefficients)) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }
      double schur = matrixEntry(candidate, candidate);
      for (int row = 0; row < activeCount; ++row)
        schur -= cross[row] * coefficients[row];
      const double independenceTolerance =
          rankTolerance * diagonalScale;

      if (schur > independenceTolerance) {
        if (active.size() >= 6) {
          result.status = UnilateralProjectionStatus::NumericalFailure;
          return result;
        }
        active.push_back(candidate);
        candidateAdded = true;
      } else if (schur < -16.0 * independenceTolerance ||
                 !std::isfinite(schur)) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      } else {
        // The new row lies in the current response span. A positive residual
        // supplies a zero-curvature descent direction:
        //   d_candidate = 1, d_active = -coefficients.
        // Move until an existing multiplier reaches zero, replace that row,
        // and continue with an independent working set. With no blocker the
        // dual is unbounded, which is the explicit infeasible-primal case.
        ++result.dependentPivots;
        double step = 0.0;
        bool haveBlocker = false;
        for (int row = 0; row < activeCount; ++row) {
          if (coefficients[row] <= multiplierTolerance)
            continue;
          const double ratio =
              lambda[static_cast<size_t>(active[row])] /
              coefficients[row];
          if (!haveBlocker || ratio < step) {
            haveBlocker = true;
            step = ratio;
          }
        }
        if (!haveBlocker || !std::isfinite(step)) {
          result.status = UnilateralProjectionStatus::Infeasible;
          return result;
        }
        step = std::max(0.0, step);
        for (int row = 0; row < activeCount; ++row) {
          lambda[static_cast<size_t>(active[row])] -=
              step * coefficients[row];
        }
        lambda[static_cast<size_t>(candidate)] = step;

        std::vector<int> retained;
        retained.reserve(6);
        for (int row = 0; row < activeCount; ++row) {
          const int activeRow = active[row];
          if (coefficients[row] > multiplierTolerance &&
              lambda[static_cast<size_t>(activeRow)] <=
                  multiplierTolerance) {
            lambda[static_cast<size_t>(activeRow)] = 0.0;
            ++result.multiplierRemovals;
          } else {
            retained.push_back(activeRow);
          }
        }
        if (retained.size() == active.size()) {
          result.status = UnilateralProjectionStatus::NumericalFailure;
          return result;
        }
        active.swap(retained);
        active.push_back(candidate);
        candidateAdded = true;
      }
    }

    if (!candidateAdded) {
      result.status = UnilateralProjectionStatus::NumericalFailure;
      return result;
    }

    bool workingSetSolved = false;
    while (!workingSetSolved && result.iterations < iterationLimit) {
      ++result.iterations;
      const int activeCount = static_cast<int>(active.size());
      if (activeCount == 0) {
        workingSetSolved = true;
        break;
      }
      double matrix[6][6] = {};
      double rhs[6] = {};
      double solution[6] = {};
      for (int row = 0; row < activeCount; ++row) {
        rhs[row] = static_cast<double>(
            ordered[static_cast<size_t>(active[row])].row.outwardVelocity);
        for (int column = 0; column < activeCount; ++column)
          matrix[row][column] =
              matrixEntry(active[row], active[column]);
      }
      if (!solveDense(matrix, rhs, activeCount, rankTolerance,
                      solution)) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }

      bool allPositive = true;
      for (int row = 0; row < activeCount; ++row) {
        if (solution[row] <= multiplierTolerance) {
          allPositive = false;
          break;
        }
      }
      if (allPositive) {
        for (int row = 0; row < activeCount; ++row)
          lambda[static_cast<size_t>(active[row])] = solution[row];
        workingSetSolved = true;
        break;
      }

      double alpha = 1.0;
      bool haveBlocker = false;
      for (int row = 0; row < activeCount; ++row) {
        if (solution[row] > multiplierTolerance)
          continue;
        const double current =
            lambda[static_cast<size_t>(active[row])];
        const double denominator = current - solution[row];
        const double ratio =
            denominator > 0.0 ? current / denominator : 0.0;
        if (!haveBlocker || ratio < alpha) {
          alpha = ratio;
          haveBlocker = true;
        }
      }
      if (!haveBlocker || !std::isfinite(alpha)) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }
      alpha = std::max(0.0, std::min(1.0, alpha));
      for (int row = 0; row < activeCount; ++row) {
        double &current = lambda[static_cast<size_t>(active[row])];
        current += alpha * (solution[row] - current);
      }

      std::vector<int> retained;
      retained.reserve(6);
      for (int row = 0; row < activeCount; ++row) {
        const int activeRow = active[row];
        if (solution[row] <= multiplierTolerance &&
            lambda[static_cast<size_t>(activeRow)] <=
                multiplierTolerance) {
          lambda[static_cast<size_t>(activeRow)] = 0.0;
          ++result.multiplierRemovals;
        } else {
          retained.push_back(activeRow);
        }
      }
      if (retained.size() == active.size()) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }
      active.swap(retained);
    }
  }

  result.status = UnilateralProjectionStatus::IterationLimit;
  return result;
}

} // namespace AvbdRef
