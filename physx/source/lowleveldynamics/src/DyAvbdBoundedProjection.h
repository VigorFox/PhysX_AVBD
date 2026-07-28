#pragma once

#include "foundation/PxArray.h"

#include <algorithm>
#include <cmath>

namespace physx {
namespace Dy {

enum AvbdBoundedProjectionStatus {
  eAVBD_BOUNDED_SOLVED,
  eAVBD_BOUNDED_NO_CORRECTION,
  eAVBD_BOUNDED_BUDGET_EXHAUSTED,
  eAVBD_BOUNDED_INFEASIBLE,
  eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED,
  eAVBD_BOUNDED_NUMERICAL_FAILURE,
  eAVBD_BOUNDED_ITERATION_LIMIT
};

struct AvbdBoundedProjectionResult {
  AvbdBoundedProjectionStatus status;
  PxArray<double> candidateImpulses;
  PxArray<double> commitImpulses;
  PxU32 lowerRows;
  PxU32 freeRows;
  PxU32 upperRows;
  PxU32 iterations;
  double maximumResidual;
  double maximumKktViolation;
  double projectedGradientTolerance;

  AvbdBoundedProjectionResult()
      : status(eAVBD_BOUNDED_NUMERICAL_FAILURE), lowerRows(0), freeRows(0),
        upperRows(0), iterations(0), maximumResidual(0.0),
        maximumKktViolation(0.0), projectedGradientTolerance(0.0) {}
};

namespace AvbdBoundedProjectionDetail {

PX_FORCE_INLINE double denseEntry(const PxArray<double> &matrix,
                                  PxU32 count, PxU32 row,
                                  PxU32 column) {
  return matrix[row * count + column];
}

static bool solveDense(const PxArray<double> &matrix,
                       const PxArray<double> &rhs, PxU32 count,
                       double relativeTolerance,
                       PxArray<double> &solution) {
  if (count == 0) {
    solution.clear();
    return true;
  }
  const PxU32 width = count + 1;
  PxArray<double> augmented(count * width, 0.0);
  double scale = 1.0;
  for (PxU32 row = 0; row < count; ++row) {
    for (PxU32 column = 0; column < count; ++column) {
      const double value = matrix[row * count + column];
      augmented[row * width + column] = value;
      scale = std::max(scale, std::fabs(value));
    }
    augmented[row * width + count] = rhs[row];
  }

  const double pivotTolerance = relativeTolerance * scale;
  for (PxU32 column = 0; column < count; ++column) {
    PxU32 pivot = column;
    double pivotMagnitude =
        std::fabs(augmented[column * width + column]);
    for (PxU32 row = column + 1; row < count; ++row) {
      const double candidate =
          std::fabs(augmented[row * width + column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!std::isfinite(pivotMagnitude) ||
        pivotMagnitude <= pivotTolerance)
      return false;
    if (pivot != column) {
      for (PxU32 entry = column; entry <= count; ++entry)
        std::swap(augmented[column * width + entry],
                  augmented[pivot * width + entry]);
    }

    const double inversePivot =
        1.0 / augmented[column * width + column];
    for (PxU32 entry = column; entry <= count; ++entry)
      augmented[column * width + entry] *= inversePivot;
    for (PxU32 row = 0; row < count; ++row) {
      if (row == column)
        continue;
      const double factor = augmented[row * width + column];
      for (PxU32 entry = column; entry <= count; ++entry)
        augmented[row * width + entry] -=
            factor * augmented[column * width + entry];
    }
  }

  solution.resize(count);
  for (PxU32 row = 0; row < count; ++row) {
    solution[row] = augmented[row * width + count];
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

PX_FORCE_INLINE PxI32 activePosition(const PxArray<PxU32> &active,
                                     PxU32 row) {
  for (PxU32 position = 0; position < active.size(); ++position) {
    if (active[position] == row)
      return PxI32(position);
  }
  return -1;
}

enum UnboundedFeasibilityStatus {
  eUNBOUNDED_SOLVED,
  eUNBOUNDED_INFEASIBLE,
  eUNBOUNDED_NUMERICAL_FAILURE,
  eUNBOUNDED_ITERATION_LIMIT
};

// Active-set solve of the unbounded unilateral objective.  This dense path is
// used only to distinguish a valid same-row budget shortage from structural
// infeasibility; its multipliers are never exposed as a candidate.
static UnboundedFeasibilityStatus classifyUnboundedFeasibility(
    const PxArray<double> &response, const PxArray<double> &outward,
    PxU32 generalizedDimension, double feasibilityTolerance) {
  const PxU32 rowCount = outward.size();
  const auto responseEntry =
      [&](PxU32 row, PxU32 column) {
        return denseEntry(response, rowCount, row, column);
      };
  PxArray<double> lambda(rowCount, 0.0);
  PxArray<PxU32> active;
  active.reserve(PxMin(rowCount, generalizedDimension));
  const double multiplierTolerance = feasibilityTolerance * 0.1;
  const double rankTolerance = 1.0e-7;
  const PxU32 iterationLimit = 128 + 64 * rowCount;

  const auto residual = [&](PxU32 row) {
    double value = outward[row];
    for (PxU32 position = 0; position < active.size(); ++position) {
      const PxU32 activeRow = active[position];
      value -= responseEntry(row, activeRow) * lambda[activeRow];
    }
    return value;
  };

  PxU32 iterations = 0;
  while (iterations < iterationLimit) {
    ++iterations;
    PxU32 candidate = PX_MAX_U32;
    double maximumViolation = feasibilityTolerance;
    for (PxU32 row = 0; row < rowCount; ++row) {
      if (activePosition(active, row) >= 0)
        continue;
      const double value = residual(row);
      if (value > maximumViolation) {
        candidate = row;
        maximumViolation = value;
      }
    }
    if (candidate == PX_MAX_U32) {
      double maximumResidual = 0.0;
      for (PxU32 row = 0; row < rowCount; ++row)
        maximumResidual = std::max(maximumResidual, residual(row));
      return std::isfinite(maximumResidual) &&
                     maximumResidual <= 4.0 * feasibilityTolerance
                 ? eUNBOUNDED_SOLVED
                 : eUNBOUNDED_NUMERICAL_FAILURE;
    }

    if (active.empty()) {
      const double diagonal = responseEntry(candidate, candidate);
      if (!std::isfinite(diagonal) || diagonal <= rankTolerance)
        return eUNBOUNDED_INFEASIBLE;
      active.pushBack(candidate);
    } else {
      const PxU32 activeCount = active.size();
      PxArray<double> matrix(activeCount * activeCount, 0.0);
      PxArray<double> cross(activeCount, 0.0);
      PxArray<double> coefficients;
      double diagonalScale = std::max(
          1.0, std::fabs(responseEntry(candidate, candidate)));
      for (PxU32 row = 0; row < activeCount; ++row) {
        cross[row] = responseEntry(active[row], candidate);
        for (PxU32 column = 0; column < activeCount; ++column) {
          const double value =
              responseEntry(active[row], active[column]);
          matrix[row * activeCount + column] = value;
          diagonalScale = std::max(diagonalScale, std::fabs(value));
        }
      }
      if (!solveDense(matrix, cross, activeCount, rankTolerance,
                      coefficients))
        return eUNBOUNDED_NUMERICAL_FAILURE;
      double schur = responseEntry(candidate, candidate);
      for (PxU32 row = 0; row < activeCount; ++row)
        schur -= cross[row] * coefficients[row];
      const double independenceTolerance =
          rankTolerance * diagonalScale;
      if (schur > independenceTolerance) {
        if (active.size() >= generalizedDimension)
          return eUNBOUNDED_NUMERICAL_FAILURE;
        active.pushBack(candidate);
      } else if (!std::isfinite(schur) ||
                 schur < -16.0 * independenceTolerance) {
        return eUNBOUNDED_NUMERICAL_FAILURE;
      } else {
        double step = 0.0;
        bool haveBlocker = false;
        for (PxU32 row = 0; row < activeCount; ++row) {
          if (coefficients[row] <= multiplierTolerance)
            continue;
          const double ratio =
              lambda[active[row]] / coefficients[row];
          if (!haveBlocker || ratio < step) {
            haveBlocker = true;
            step = ratio;
          }
        }
        if (!haveBlocker || !std::isfinite(step))
          return eUNBOUNDED_INFEASIBLE;
        step = std::max(0.0, step);
        PxArray<PxU32> retained;
        retained.reserve(active.size());
        for (PxU32 row = 0; row < activeCount; ++row) {
          lambda[active[row]] -= step * coefficients[row];
          if (coefficients[row] > multiplierTolerance &&
              lambda[active[row]] <= multiplierTolerance) {
            lambda[active[row]] = 0.0;
          } else {
            retained.pushBack(active[row]);
          }
        }
        if (retained.size() == active.size())
          return eUNBOUNDED_NUMERICAL_FAILURE;
        active.swap(retained);
        lambda[candidate] = step;
        active.pushBack(candidate);
      }
    }

    bool workingSetSolved = false;
    while (!workingSetSolved && iterations < iterationLimit) {
      ++iterations;
      const PxU32 activeCount = active.size();
      if (activeCount == 0) {
        workingSetSolved = true;
        break;
      }
      PxArray<double> matrix(activeCount * activeCount, 0.0);
      PxArray<double> rhs(activeCount, 0.0);
      PxArray<double> solution;
      for (PxU32 row = 0; row < activeCount; ++row) {
        rhs[row] = outward[active[row]];
        for (PxU32 column = 0; column < activeCount; ++column)
          matrix[row * activeCount + column] =
              responseEntry(active[row], active[column]);
      }
      if (!solveDense(matrix, rhs, activeCount, rankTolerance, solution))
        return eUNBOUNDED_NUMERICAL_FAILURE;

      bool allPositive = true;
      for (PxU32 row = 0; row < activeCount; ++row) {
        if (solution[row] <= multiplierTolerance) {
          allPositive = false;
          break;
        }
      }
      if (allPositive) {
        for (PxU32 row = 0; row < activeCount; ++row)
          lambda[active[row]] = solution[row];
        workingSetSolved = true;
        break;
      }

      double alpha = 1.0;
      bool haveBlocker = false;
      for (PxU32 row = 0; row < activeCount; ++row) {
        if (solution[row] > multiplierTolerance)
          continue;
        const double current = lambda[active[row]];
        const double denominator = current - solution[row];
        const double ratio =
            denominator > 0.0 ? current / denominator : 0.0;
        if (!haveBlocker || ratio < alpha) {
          alpha = ratio;
          haveBlocker = true;
        }
      }
      if (!haveBlocker || !std::isfinite(alpha))
        return eUNBOUNDED_NUMERICAL_FAILURE;
      alpha = std::max(0.0, std::min(1.0, alpha));
      PxArray<PxU32> retained;
      retained.reserve(active.size());
      for (PxU32 row = 0; row < activeCount; ++row) {
        double &current = lambda[active[row]];
        current += alpha * (solution[row] - current);
        if (solution[row] <= multiplierTolerance &&
            current <= multiplierTolerance) {
          current = 0.0;
        } else {
          retained.pushBack(active[row]);
        }
      }
      if (retained.size() == active.size())
        return eUNBOUNDED_NUMERICAL_FAILURE;
      active.swap(retained);
    }
  }
  return eUNBOUNDED_ITERATION_LIMIT;
}

static void gradient(const PxArray<double> &response,
                     const PxArray<double> &outward,
                     const PxArray<double> &impulses,
                     PxArray<double> &values) {
  const PxU32 rowCount = outward.size();
  values.resize(rowCount);
  for (PxU32 row = 0; row < rowCount; ++row) {
    double value = -outward[row];
    for (PxU32 column = 0; column < rowCount; ++column)
      value += denseEntry(response, rowCount, row, column) *
               impulses[column];
    values[row] = value;
  }
}

static double objective(const PxArray<double> &response,
                        const PxArray<double> &outward,
                        const PxArray<double> &impulses) {
  const PxU32 rowCount = outward.size();
  double value = 0.0;
  for (PxU32 row = 0; row < rowCount; ++row) {
    double responseImpulse = 0.0;
    for (PxU32 column = 0; column < rowCount; ++column)
      responseImpulse +=
          denseEntry(response, rowCount, row, column) *
          impulses[column];
    value += 0.5 * impulses[row] * responseImpulse -
             outward[row] * impulses[row];
  }
  return value;
}

static double projectedGradientViolation(
    const PxArray<double> &gradientValues,
    const PxArray<double> &impulses,
    const PxArray<double> &upperBounds, double boundTolerance) {
  double maximum = 0.0;
  for (PxU32 row = 0; row < impulses.size(); ++row) {
    double violation = 0.0;
    if (upperBounds[row] <= boundTolerance) {
      violation = 0.0;
    } else if (impulses[row] <= boundTolerance) {
      violation = std::max(0.0, -gradientValues[row]);
    } else if (upperBounds[row] - impulses[row] <= boundTolerance) {
      violation = std::max(0.0, gradientValues[row]);
    } else {
      violation = std::fabs(gradientValues[row]);
    }
    maximum = std::max(maximum, violation);
  }
  return maximum;
}

} // namespace AvbdBoundedProjectionDetail

// Solve a complete-component box-QP.  The response matrix and objectives are
// immutable inputs.  candidateImpulses exposes the converged box optimum for
// diagnostics; commitImpulses stays zero unless all rows are feasible.
static AvbdBoundedProjectionResult solveAvbdBoundedProjection(
    const PxArray<double> &response, const PxArray<double> &outward,
    const PxArray<double> &upperBounds, PxU32 generalizedDimension,
    double relativeTolerance = 1.0e-6) {
  using namespace AvbdBoundedProjectionDetail;
  AvbdBoundedProjectionResult result;
  const PxU32 rowCount = outward.size();
  result.candidateImpulses.resize(rowCount, 0.0);
  result.commitImpulses.resize(rowCount, 0.0);
  if (rowCount == 0 || response.size() != rowCount * rowCount ||
      upperBounds.size() != rowCount || generalizedDimension == 0 ||
      !std::isfinite(relativeTolerance) || relativeTolerance <= 0.0)
    return result;

  bool needsCorrection = false;
  double velocityScale = 1.0;
  double impulseScale = 1.0;
  double lipschitzBound = 0.0;
  for (PxU32 row = 0; row < rowCount; ++row) {
    if (!std::isfinite(outward[row]) ||
        !std::isfinite(upperBounds[row]) || upperBounds[row] < 0.0)
      return result;
    needsCorrection = needsCorrection || outward[row] > 0.0;
    velocityScale = std::max(velocityScale, std::fabs(outward[row]));
    impulseScale = std::max(impulseScale, upperBounds[row]);
    double absoluteRowSum = 0.0;
    for (PxU32 column = 0; column < rowCount; ++column) {
      const double value =
          denseEntry(response, rowCount, row, column);
      if (!std::isfinite(value))
        return result;
      absoluteRowSum += std::fabs(value);
    }
    lipschitzBound = std::max(lipschitzBound, absoluteRowSum);
  }
  if (!needsCorrection) {
    result.status = eAVBD_BOUNDED_NO_CORRECTION;
    result.lowerRows = rowCount;
    return result;
  }

  const double feasibilityTolerance =
      relativeTolerance * velocityScale;
  const double boundTolerance = relativeTolerance * impulseScale;
  if (!std::isfinite(lipschitzBound) || lipschitzBound <= 1.0e-14) {
    result.status = eAVBD_BOUNDED_INFEASIBLE;
    result.maximumResidual = velocityScale;
    return result;
  }

  const double step = 1.0 / lipschitzBound;
  PxArray<double> impulses(rowCount, 0.0);
  PxArray<double> extrapolated(rowCount, 0.0);
  PxArray<double> next(rowCount, 0.0);
  PxArray<double> gradientValues;
  double acceleration = 1.0;
  double currentObjective = 0.0;
  const PxU32 iterationLimit =
      PxMax(PxU32(4096), PxU32(1024 + 512 * rowCount));
  bool converged = false;
  for (PxU32 iteration = 0; iteration < iterationLimit; ++iteration) {
    gradient(response, outward, extrapolated, gradientValues);
    for (PxU32 row = 0; row < rowCount; ++row)
      next[row] = std::min(
          upperBounds[row],
          std::max(0.0, extrapolated[row] - step * gradientValues[row]));
    double nextObjective = objective(response, outward, next);
    if (!std::isfinite(nextObjective)) {
      result.status = eAVBD_BOUNDED_NUMERICAL_FAILURE;
      return result;
    }
    const double objectiveSlack =
        1.0e-13 * std::max(1.0, std::fabs(currentObjective));
    if (nextObjective > currentObjective + objectiveSlack) {
      extrapolated = impulses;
      acceleration = 1.0;
      gradient(response, outward, extrapolated, gradientValues);
      for (PxU32 row = 0; row < rowCount; ++row)
        next[row] = std::min(
            upperBounds[row],
            std::max(0.0,
                     extrapolated[row] - step * gradientValues[row]));
      nextObjective = objective(response, outward, next);
      if (!std::isfinite(nextObjective) ||
          nextObjective > currentObjective + 16.0 * objectiveSlack) {
        result.status = eAVBD_BOUNDED_NUMERICAL_FAILURE;
        return result;
      }
    }

    impulses.swap(next);
    currentObjective = nextObjective;
    result.iterations = iteration + 1;
    gradient(response, outward, impulses, gradientValues);
    if (projectedGradientViolation(
            gradientValues, impulses, upperBounds, boundTolerance) <=
        feasibilityTolerance) {
      converged = true;
      break;
    }
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(
                         1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        (acceleration - 1.0) / nextAcceleration;
    for (PxU32 row = 0; row < rowCount; ++row)
      extrapolated[row] =
          impulses[row] + momentum * (impulses[row] - next[row]);
    acceleration = nextAcceleration;
    if ((iteration + 1) % 64 == 0) {
      extrapolated = impulses;
      acceleration = 1.0;
    }
  }
  if (!converged) {
    result.status = eAVBD_BOUNDED_ITERATION_LIMIT;
    return result;
  }

  gradient(response, outward, impulses, gradientValues);
  result.maximumKktViolation = projectedGradientViolation(
      gradientValues, impulses, upperBounds, boundTolerance);
  double maximumResidual = 0.0;
  for (PxU32 row = 0; row < rowCount; ++row) {
    if (upperBounds[row] <= boundTolerance ||
        upperBounds[row] - impulses[row] <= boundTolerance)
      ++result.upperRows;
    else if (impulses[row] <= boundTolerance)
      ++result.lowerRows;
    else
      ++result.freeRows;
    maximumResidual =
        std::max(maximumResidual, -gradientValues[row]);
  }
  result.maximumResidual = maximumResidual;
  result.candidateImpulses = impulses;
  if (!std::isfinite(maximumResidual) ||
      maximumResidual > 4.0 * feasibilityTolerance) {
    const UnboundedFeasibilityStatus unbounded =
        classifyUnboundedFeasibility(
            response, outward, generalizedDimension,
            feasibilityTolerance);
    if (unbounded == eUNBOUNDED_SOLVED)
      result.status = eAVBD_BOUNDED_BUDGET_EXHAUSTED;
    else if (unbounded == eUNBOUNDED_INFEASIBLE)
      result.status = eAVBD_BOUNDED_INFEASIBLE;
    else if (unbounded == eUNBOUNDED_ITERATION_LIMIT)
      result.status = eAVBD_BOUNDED_ITERATION_LIMIT;
    else
      result.status = eAVBD_BOUNDED_NUMERICAL_FAILURE;
    return result;
  }

  result.commitImpulses = result.candidateImpulses;
  result.status = eAVBD_BOUNDED_SOLVED;
  return result;
}

} // namespace Dy
} // namespace physx
