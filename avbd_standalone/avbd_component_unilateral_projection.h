#pragma once

#include "avbd_unilateral_projection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace AvbdRef {

struct ComponentProjectionBody {
  float inverseMassResponse = 0.0f;
  Mat33 inverseInertiaResponse;
  uint64_t stableKey = 0;
};

struct ComponentProjectionTerm {
  size_t bodyIndex = 0;
  Vec3 linearJacobian;
  Vec3 angularJacobian;
};

struct ComponentProjectionRow {
  std::vector<ComponentProjectionTerm> terms;
  float outwardVelocity = 0.0f;
  uint64_t stableKey = 0;
};

struct ComponentProjectionResult {
  UnilateralProjectionStatus status =
      UnilateralProjectionStatus::NumericalFailure;
  std::vector<float> impulses;
  std::vector<Vec6> velocityDeltas;
  int activeRows = 0;
  int iterations = 0;
  int multiplierRemovals = 0;
  int dependentPivots = 0;
  float maxResidual = 0.0f;
};

namespace ComponentProjectionDetail {

struct OrderedTerm {
  ComponentProjectionTerm term;
  uint64_t bodyStableKey = 0;
};

struct OrderedRow {
  std::vector<OrderedTerm> terms;
  float outwardVelocity = 0.0f;
  uint64_t stableKey = 0;
  size_t originalIndex = 0;
};

inline bool finiteBodyResponse(const ComponentProjectionBody &body) {
  if (!std::isfinite(body.inverseMassResponse) ||
      body.inverseMassResponse < 0.0f)
    return false;
  return UnilateralProjectionDetail::finiteVec3(
             body.inverseInertiaResponse * Vec3(1.0f, 0.0f, 0.0f)) &&
         UnilateralProjectionDetail::finiteVec3(
             body.inverseInertiaResponse * Vec3(0.0f, 1.0f, 0.0f)) &&
         UnilateralProjectionDetail::finiteVec3(
             body.inverseInertiaResponse * Vec3(0.0f, 0.0f, 1.0f));
}

inline bool orderedTermLess(const OrderedTerm &a, const OrderedTerm &b) {
  if (a.bodyStableKey != b.bodyStableKey)
    return a.bodyStableKey < b.bodyStableKey;
  return a.term.bodyIndex < b.term.bodyIndex;
}

inline bool orderedRowLess(const OrderedRow &a, const OrderedRow &b) {
  if (a.stableKey != b.stableKey)
    return a.stableKey < b.stableKey;
  if (a.terms.size() != b.terms.size())
    return a.terms.size() < b.terms.size();
  for (size_t term = 0; term < a.terms.size(); ++term) {
    const OrderedTerm &at = a.terms[term];
    const OrderedTerm &bt = b.terms[term];
    if (at.bodyStableKey != bt.bodyStableKey)
      return at.bodyStableKey < bt.bodyStableKey;
    const float av[6] = {
        at.term.linearJacobian.x, at.term.linearJacobian.y,
        at.term.linearJacobian.z, at.term.angularJacobian.x,
        at.term.angularJacobian.y, at.term.angularJacobian.z};
    const float bv[6] = {
        bt.term.linearJacobian.x, bt.term.linearJacobian.y,
        bt.term.linearJacobian.z, bt.term.angularJacobian.x,
        bt.term.angularJacobian.y, bt.term.angularJacobian.z};
    for (int component = 0; component < 6; ++component) {
      if (av[component] != bv[component])
        return av[component] < bv[component];
    }
  }
  if (a.outwardVelocity != b.outwardVelocity)
    return a.outwardVelocity < b.outwardVelocity;
  return a.originalIndex < b.originalIndex;
}

inline double response(const OrderedRow &a, const OrderedRow &b,
                       const std::vector<ComponentProjectionBody> &bodies) {
  double value = 0.0;
  for (const OrderedTerm &at : a.terms) {
    for (const OrderedTerm &bt : b.terms) {
      if (at.term.bodyIndex != bt.term.bodyIndex)
        continue;
      const ComponentProjectionBody &body =
          bodies[at.term.bodyIndex];
      value +=
          static_cast<double>(body.inverseMassResponse) *
              static_cast<double>(at.term.linearJacobian.dot(
                  bt.term.linearJacobian)) +
          static_cast<double>(at.term.angularJacobian.dot(
              body.inverseInertiaResponse *
              bt.term.angularJacobian));
    }
  }
  return value;
}

inline bool solveDenseDynamic(const std::vector<double> &matrix,
                              const std::vector<double> &rhs, int count,
                              double relativeTolerance,
                              std::vector<double> &solution) {
  if (count <= 0) {
    solution.clear();
    return true;
  }
  const size_t width = static_cast<size_t>(count + 1);
  std::vector<double> augmented(static_cast<size_t>(count) * width, 0.0);
  double scale = 1.0;
  for (int row = 0; row < count; ++row) {
    for (int column = 0; column < count; ++column) {
      const double value =
          matrix[static_cast<size_t>(row * count + column)];
      augmented[static_cast<size_t>(row) * width +
                static_cast<size_t>(column)] = value;
      scale = std::max(scale, std::fabs(value));
    }
    augmented[static_cast<size_t>(row) * width +
              static_cast<size_t>(count)] = rhs[static_cast<size_t>(row)];
  }

  const double pivotTolerance = relativeTolerance * scale;
  for (int column = 0; column < count; ++column) {
    int pivot = column;
    double pivotMagnitude = std::fabs(
        augmented[static_cast<size_t>(column) * width +
                  static_cast<size_t>(column)]);
    for (int row = column + 1; row < count; ++row) {
      const double candidate = std::fabs(
          augmented[static_cast<size_t>(row) * width +
                    static_cast<size_t>(column)]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!std::isfinite(pivotMagnitude) ||
        pivotMagnitude <= pivotTolerance)
      return false;
    if (pivot != column) {
      for (int entry = column; entry <= count; ++entry) {
        std::swap(
            augmented[static_cast<size_t>(column) * width +
                      static_cast<size_t>(entry)],
            augmented[static_cast<size_t>(pivot) * width +
                      static_cast<size_t>(entry)]);
      }
    }

    const double inversePivot =
        1.0 / augmented[static_cast<size_t>(column) * width +
                        static_cast<size_t>(column)];
    for (int entry = column; entry <= count; ++entry) {
      augmented[static_cast<size_t>(column) * width +
                static_cast<size_t>(entry)] *= inversePivot;
    }
    for (int row = 0; row < count; ++row) {
      if (row == column)
        continue;
      const double factor =
          augmented[static_cast<size_t>(row) * width +
                    static_cast<size_t>(column)];
      for (int entry = column; entry <= count; ++entry) {
        augmented[static_cast<size_t>(row) * width +
                  static_cast<size_t>(entry)] -=
            factor *
            augmented[static_cast<size_t>(column) * width +
                      static_cast<size_t>(entry)];
      }
    }
  }

  solution.resize(static_cast<size_t>(count));
  for (int row = 0; row < count; ++row) {
    solution[static_cast<size_t>(row)] =
        augmented[static_cast<size_t>(row) * width +
                  static_cast<size_t>(count)];
    if (!std::isfinite(solution[static_cast<size_t>(row)]))
      return false;
  }
  return true;
}

inline int activePosition(const std::vector<int> &active, int row) {
  for (size_t position = 0; position < active.size(); ++position) {
    if (active[position] == row)
      return static_cast<int>(position);
  }
  return -1;
}

} // namespace ComponentProjectionDetail

/**
 * Solve one minimum kinetic-energy unilateral objective over a connected set
 * of dynamic rigid bodies. Every row may reference one or more bodies, and the
 * response is the block-diagonal per-body inverse mass/inertia operator.
 *
 * Input row count is unrestricted. The active independent set is bounded only
 * by the generalized response rank (at most six times the body count), while
 * every inequality participates in residual selection and final verification.
 */
inline ComponentProjectionResult solveComponentUnilateralProjection(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<ComponentProjectionRow> &inputRows,
    float relativeTolerance = 1.0e-6f) {
  using namespace ComponentProjectionDetail;

  ComponentProjectionResult result;
  result.impulses.assign(inputRows.size(), 0.0f);
  result.velocityDeltas.assign(bodies.size(), Vec6());
  if (bodies.empty() || relativeTolerance <= 0.0f ||
      !std::isfinite(relativeTolerance))
    return result;

  std::vector<uint64_t> bodyKeys;
  bodyKeys.reserve(bodies.size());
  for (const ComponentProjectionBody &body : bodies) {
    if (!finiteBodyResponse(body))
      return result;
    bodyKeys.push_back(body.stableKey);
  }
  std::sort(bodyKeys.begin(), bodyKeys.end());
  if (std::adjacent_find(bodyKeys.begin(), bodyKeys.end()) !=
      bodyKeys.end())
    return result;

  std::vector<OrderedRow> ordered;
  ordered.reserve(inputRows.size());
  double velocityScale = 1.0;
  bool needsCorrection = false;
  for (size_t rowIndex = 0; rowIndex < inputRows.size(); ++rowIndex) {
    const ComponentProjectionRow &input = inputRows[rowIndex];
    if (!std::isfinite(input.outwardVelocity))
      return result;
    OrderedRow row;
    row.outwardVelocity = input.outwardVelocity;
    row.stableKey = input.stableKey;
    row.originalIndex = rowIndex;
    row.terms.reserve(input.terms.size());
    for (const ComponentProjectionTerm &term : input.terms) {
      if (term.bodyIndex >= bodies.size() ||
          !UnilateralProjectionDetail::finiteVec3(
              term.linearJacobian) ||
          !UnilateralProjectionDetail::finiteVec3(
              term.angularJacobian))
        return result;
      row.terms.push_back({term, bodies[term.bodyIndex].stableKey});
    }
    std::sort(row.terms.begin(), row.terms.end(), orderedTermLess);
    for (size_t term = 1; term < row.terms.size(); ++term) {
      if (row.terms[term - 1].term.bodyIndex ==
          row.terms[term].term.bodyIndex)
        return result;
    }
    ordered.push_back(row);
    velocityScale =
        std::max(velocityScale,
                 std::fabs(static_cast<double>(input.outwardVelocity)));
    needsCorrection = needsCorrection || input.outwardVelocity > 0.0f;
  }
  if (!needsCorrection || ordered.empty()) {
    result.status = UnilateralProjectionStatus::NoCorrection;
    return result;
  }
  std::sort(ordered.begin(), ordered.end(), orderedRowLess);

  const int rowCount = static_cast<int>(ordered.size());
  const size_t generalizedDimension = 6u * bodies.size();
  std::vector<double> lambda(static_cast<size_t>(rowCount), 0.0);
  std::vector<int> active;
  active.reserve(
      std::min(static_cast<size_t>(rowCount), generalizedDimension));
  const double feasibilityTolerance =
      static_cast<double>(relativeTolerance) * velocityScale;
  const double multiplierTolerance =
      static_cast<double>(relativeTolerance) * 0.1;
  const double rankTolerance = 1.0e-7;
  const int iterationLimit = 128 + 64 * rowCount;

  const auto matrixEntry = [&](int row, int column) {
    return response(ordered[static_cast<size_t>(row)],
                    ordered[static_cast<size_t>(column)], bodies);
  };
  const auto residual = [&](int row) {
    double value = static_cast<double>(
        ordered[static_cast<size_t>(row)].outwardVelocity);
    for (int activeRow : active) {
      value -= matrixEntry(row, activeRow) *
               lambda[static_cast<size_t>(activeRow)];
    }
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

      std::vector<Vec3> linearImpulses(
          bodies.size(), Vec3(0.0f, 0.0f, 0.0f));
      std::vector<Vec3> angularImpulses(
          bodies.size(), Vec3(0.0f, 0.0f, 0.0f));
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
        for (const OrderedTerm &term : orderedRow.terms) {
          linearImpulses[term.term.bodyIndex] +=
              term.term.linearJacobian * impulse;
          angularImpulses[term.term.bodyIndex] +=
              term.term.angularJacobian * impulse;
        }
      }
      for (size_t body = 0; body < bodies.size(); ++body) {
        const Vec3 linearDelta =
            linearImpulses[body] *
            (-bodies[body].inverseMassResponse);
        const Vec3 angularDelta =
            (bodies[body].inverseInertiaResponse *
             angularImpulses[body]) *
            -1.0f;
        result.velocityDeltas[body] =
            Vec6(linearDelta, angularDelta);
      }
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
      std::vector<double> matrix(
          static_cast<size_t>(activeCount * activeCount), 0.0);
      std::vector<double> cross(static_cast<size_t>(activeCount), 0.0);
      std::vector<double> coefficients;
      double diagonalScale =
          std::max(1.0, std::fabs(matrixEntry(candidate, candidate)));
      for (int row = 0; row < activeCount; ++row) {
        cross[static_cast<size_t>(row)] =
            matrixEntry(active[static_cast<size_t>(row)], candidate);
        for (int column = 0; column < activeCount; ++column) {
          const double value = matrixEntry(
              active[static_cast<size_t>(row)],
              active[static_cast<size_t>(column)]);
          matrix[static_cast<size_t>(row * activeCount + column)] =
              value;
          diagonalScale = std::max(diagonalScale, std::fabs(value));
        }
      }
      if (!solveDenseDynamic(matrix, cross, activeCount,
                             rankTolerance, coefficients)) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }
      double schur = matrixEntry(candidate, candidate);
      for (int row = 0; row < activeCount; ++row) {
        schur -= cross[static_cast<size_t>(row)] *
                 coefficients[static_cast<size_t>(row)];
      }
      const double independenceTolerance =
          rankTolerance * diagonalScale;

      if (schur > independenceTolerance) {
        if (active.size() >= generalizedDimension) {
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
        ++result.dependentPivots;
        double step = 0.0;
        bool haveBlocker = false;
        for (int row = 0; row < activeCount; ++row) {
          if (coefficients[static_cast<size_t>(row)] <=
              multiplierTolerance)
            continue;
          const double ratio =
              lambda[static_cast<size_t>(
                  active[static_cast<size_t>(row)])] /
              coefficients[static_cast<size_t>(row)];
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
          lambda[static_cast<size_t>(
              active[static_cast<size_t>(row)])] -=
              step * coefficients[static_cast<size_t>(row)];
        }
        lambda[static_cast<size_t>(candidate)] = step;

        std::vector<int> retained;
        retained.reserve(active.size());
        for (int row = 0; row < activeCount; ++row) {
          const int activeRow = active[static_cast<size_t>(row)];
          if (coefficients[static_cast<size_t>(row)] >
                  multiplierTolerance &&
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
    while (!workingSetSolved &&
           result.iterations < iterationLimit) {
      ++result.iterations;
      const int activeCount = static_cast<int>(active.size());
      if (activeCount == 0) {
        workingSetSolved = true;
        break;
      }
      std::vector<double> matrix(
          static_cast<size_t>(activeCount * activeCount), 0.0);
      std::vector<double> rhs(static_cast<size_t>(activeCount), 0.0);
      std::vector<double> solution;
      for (int row = 0; row < activeCount; ++row) {
        rhs[static_cast<size_t>(row)] =
            static_cast<double>(
                ordered[static_cast<size_t>(
                    active[static_cast<size_t>(row)])]
                    .outwardVelocity);
        for (int column = 0; column < activeCount; ++column) {
          matrix[static_cast<size_t>(row * activeCount + column)] =
              matrixEntry(active[static_cast<size_t>(row)],
                          active[static_cast<size_t>(column)]);
        }
      }
      if (!solveDenseDynamic(matrix, rhs, activeCount,
                             rankTolerance, solution)) {
        result.status = UnilateralProjectionStatus::NumericalFailure;
        return result;
      }

      bool allPositive = true;
      for (int row = 0; row < activeCount; ++row) {
        if (solution[static_cast<size_t>(row)] <=
            multiplierTolerance) {
          allPositive = false;
          break;
        }
      }
      if (allPositive) {
        for (int row = 0; row < activeCount; ++row) {
          lambda[static_cast<size_t>(
              active[static_cast<size_t>(row)])] =
              solution[static_cast<size_t>(row)];
        }
        workingSetSolved = true;
        break;
      }

      double alpha = 1.0;
      bool haveBlocker = false;
      for (int row = 0; row < activeCount; ++row) {
        if (solution[static_cast<size_t>(row)] >
            multiplierTolerance)
          continue;
        const double current =
            lambda[static_cast<size_t>(
                active[static_cast<size_t>(row)])];
        const double denominator =
            current - solution[static_cast<size_t>(row)];
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
        double &current =
            lambda[static_cast<size_t>(
                active[static_cast<size_t>(row)])];
        current +=
            alpha *
            (solution[static_cast<size_t>(row)] - current);
      }

      std::vector<int> retained;
      retained.reserve(active.size());
      for (int row = 0; row < activeCount; ++row) {
        const int activeRow = active[static_cast<size_t>(row)];
        if (solution[static_cast<size_t>(row)] <=
                multiplierTolerance &&
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
