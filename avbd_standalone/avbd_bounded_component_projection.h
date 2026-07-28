#pragma once

#include "avbd_component_unilateral_projection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace AvbdRef {

/**
 * Convert the AVBD normal dual to the only admissible material-finalize
 * impulse budget for the same row.
 *
 * PhysX AVBD stores normal compression as a negative force multiplier.
 * Multiplying its magnitude by dt converts force to the per-step impulse
 * unit used by the velocity objective.  The authored maxImpulse, when finite,
 * remains an additional upper bound; it never creates support that the
 * position solve did not own.
 */
inline bool makePositionNormalImpulseBudget(
    float signedNormalForce, float dt, float authoredMaximumImpulse,
    float &budget) {
  budget = 0.0f;
  if (!std::isfinite(signedNormalForce) || !std::isfinite(dt) ||
      dt <= 0.0f || std::isnan(authoredMaximumImpulse) ||
      authoredMaximumImpulse < 0.0f)
    return false;

  const double positionImpulse =
      std::max(0.0, -static_cast<double>(signedNormalForce)) *
      static_cast<double>(dt);
  const double authoredLimit =
      std::isfinite(authoredMaximumImpulse)
          ? static_cast<double>(authoredMaximumImpulse)
          : std::numeric_limits<double>::infinity();
  const double boundedImpulse = std::min(positionImpulse, authoredLimit);
  if (!std::isfinite(boundedImpulse) ||
      boundedImpulse >
          static_cast<double>(std::numeric_limits<float>::max()))
    return false;
  budget = static_cast<float>(boundedImpulse);
  return true;
}

struct BoundedComponentProjectionRow {
  ComponentProjectionRow row;

  /**
   * Per-step impulse upper bound in N*s.  A production caller must derive
   * this from this same row's compressive position force via
   * makePositionNormalImpulseBudget(); borrowing another row's budget is not
   * admissible.
   */
  float maximumImpulse = 0.0f;
};

enum class BoundedComponentProjectionStatus {
  Solved,
  NoCorrection,
  BudgetExhausted,
  Infeasible,
  NumericalFailure,
  IterationLimit
};

enum class BoundedComponentRowState {
  Lower,
  Free,
  Upper
};

struct BoundedComponentProjectionResult {
  BoundedComponentProjectionStatus status =
      BoundedComponentProjectionStatus::NumericalFailure;

  /**
   * The box-QP minimizer is always exposed for audit.  It is not permission
   * to mutate velocities when status is BudgetExhausted or Infeasible.
   */
  std::vector<float> candidateImpulses;
  std::vector<Vec6> candidateVelocityDeltas;
  std::vector<BoundedComponentRowState> rowStates;

  /**
   * Atomic commit payload.  These remain zero unless every row passes the
   * final residual check and status is Solved.
   */
  std::vector<float> impulses;
  std::vector<Vec6> velocityDeltas;

  int lowerRows = 0;
  int freeRows = 0;
  int upperRows = 0;
  int iterations = 0;
  int restarts = 0;
  float maximumResidual = 0.0f;
  float maximumKktViolation = 0.0f;
  double objective = 0.0;
};

namespace BoundedComponentProjectionDetail {

struct OrderedRow {
  ComponentProjectionDetail::OrderedRow component;
  float maximumImpulse = 0.0f;
};

inline bool orderedRowLess(const OrderedRow &a, const OrderedRow &b) {
  // Row keys are validated unique below, so they alone define the canonical
  // component order.  Never use originalIndex here: that would make a caller
  // permutation observable even when the physical component is unchanged.
  return a.component.stableKey < b.component.stableKey;
}

inline double objective(const std::vector<double> &response,
                        const std::vector<double> &outwardVelocity,
                        const std::vector<double> &impulses, int rowCount) {
  double value = 0.0;
  for (int row = 0; row < rowCount; ++row) {
    double responseImpulse = 0.0;
    for (int column = 0; column < rowCount; ++column) {
      responseImpulse +=
          response[static_cast<size_t>(row * rowCount + column)] *
          impulses[static_cast<size_t>(column)];
    }
    value += 0.5 * impulses[static_cast<size_t>(row)] *
                 responseImpulse -
             outwardVelocity[static_cast<size_t>(row)] *
                 impulses[static_cast<size_t>(row)];
  }
  return value;
}

inline void gradient(const std::vector<double> &response,
                     const std::vector<double> &outwardVelocity,
                     const std::vector<double> &impulses, int rowCount,
                     std::vector<double> &result) {
  result.assign(static_cast<size_t>(rowCount), 0.0);
  for (int row = 0; row < rowCount; ++row) {
    double value = -outwardVelocity[static_cast<size_t>(row)];
    for (int column = 0; column < rowCount; ++column) {
      value +=
          response[static_cast<size_t>(row * rowCount + column)] *
          impulses[static_cast<size_t>(column)];
    }
    result[static_cast<size_t>(row)] = value;
  }
}

inline double projectedGradientViolation(
    const std::vector<double> &gradientValues,
    const std::vector<double> &impulses,
    const std::vector<double> &upperBounds, double boundTolerance) {
  double maximum = 0.0;
  for (size_t row = 0; row < impulses.size(); ++row) {
    const double impulse = impulses[row];
    const double upper = upperBounds[row];
    const double gradientValue = gradientValues[row];
    double violation = 0.0;
    if (upper <= boundTolerance) {
      // A fixed zero-width interval has the full real normal cone.
      violation = 0.0;
    } else if (impulse <= boundTolerance) {
      violation = std::max(0.0, -gradientValue);
    } else if (upper - impulse <= boundTolerance) {
      violation = std::max(0.0, gradientValue);
    } else {
      violation = std::fabs(gradientValue);
    }
    maximum = std::max(maximum, violation);
  }
  return maximum;
}

inline void buildVelocityDeltas(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<OrderedRow> &ordered,
    const std::vector<double> &orderedImpulses,
    std::vector<float> &impulses, std::vector<Vec6> &velocityDeltas) {
  impulses.assign(ordered.size(), 0.0f);
  velocityDeltas.assign(bodies.size(), Vec6());
  std::vector<Vec3> linearImpulses(
      bodies.size(), Vec3(0.0f, 0.0f, 0.0f));
  std::vector<Vec3> angularImpulses(
      bodies.size(), Vec3(0.0f, 0.0f, 0.0f));
  for (size_t orderedIndex = 0; orderedIndex < ordered.size();
       ++orderedIndex) {
    const float impulse =
        static_cast<float>(orderedImpulses[orderedIndex]);
    const ComponentProjectionDetail::OrderedRow &row =
        ordered[orderedIndex].component;
    impulses[row.originalIndex] = impulse;
    for (const ComponentProjectionDetail::OrderedTerm &term : row.terms) {
      linearImpulses[term.term.bodyIndex] +=
          term.term.linearJacobian * impulse;
      angularImpulses[term.term.bodyIndex] +=
          term.term.angularJacobian * impulse;
    }
  }
  for (size_t body = 0; body < bodies.size(); ++body) {
    velocityDeltas[body] = Vec6(
        linearImpulses[body] *
            (-bodies[body].inverseMassResponse),
        (bodies[body].inverseInertiaResponse *
         angularImpulses[body]) *
            -1.0f);
  }
}

} // namespace BoundedComponentProjectionDetail

/**
 * Solve the complete-component bounded dual objective
 *
 *   minimize 0.5 * impulse' * W * impulse
 *            - outwardVelocity' * impulse
 *   subject to 0 <= impulse_i <= maximumImpulse_i
 *
 * where W is the block-diagonal rigid-body response assembled over every
 * body in the connected component.  The numerical solve is simultaneous
 * projected gradient with deterministic ordering and periodic restart; it is
 * not a row-wise physical velocity replay.
 *
 * The box KKT states are:
 *   lower: gradient >= 0
 *   free:  gradient == 0
 *   upper: gradient <= 0
 *
 * A converged box optimum is committed only when every final normal residual
 * outwardVelocity - W*impulse is non-positive within tolerance.  Otherwise
 * the whole component returns BudgetExhausted/Infeasible and its commit
 * payload remains zero.
 */
inline BoundedComponentProjectionResult
solveBoundedComponentUnilateralProjection(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<BoundedComponentProjectionRow> &inputRows,
    float relativeTolerance = 1.0e-6f) {
  using namespace BoundedComponentProjectionDetail;

  BoundedComponentProjectionResult result;
  result.candidateImpulses.assign(inputRows.size(), 0.0f);
  result.candidateVelocityDeltas.assign(bodies.size(), Vec6());
  result.rowStates.assign(inputRows.size(),
                          BoundedComponentRowState::Lower);
  result.impulses.assign(inputRows.size(), 0.0f);
  result.velocityDeltas.assign(bodies.size(), Vec6());
  if (bodies.empty() || relativeTolerance <= 0.0f ||
      !std::isfinite(relativeTolerance))
    return result;

  std::vector<uint64_t> bodyKeys;
  bodyKeys.reserve(bodies.size());
  for (const ComponentProjectionBody &body : bodies) {
    if (!ComponentProjectionDetail::finiteBodyResponse(body))
      return result;
    bodyKeys.push_back(body.stableKey);
  }
  std::sort(bodyKeys.begin(), bodyKeys.end());
  if (std::adjacent_find(bodyKeys.begin(), bodyKeys.end()) !=
      bodyKeys.end())
    return result;

  std::vector<OrderedRow> ordered;
  ordered.reserve(inputRows.size());
  std::vector<uint64_t> rowKeys;
  rowKeys.reserve(inputRows.size());
  double velocityScale = 1.0;
  bool needsCorrection = false;
  for (size_t rowIndex = 0; rowIndex < inputRows.size(); ++rowIndex) {
    const BoundedComponentProjectionRow &input =
        inputRows[rowIndex];
    if (!std::isfinite(input.row.outwardVelocity) ||
        !std::isfinite(input.maximumImpulse) ||
        input.maximumImpulse < 0.0f)
      return result;

    OrderedRow orderedRow;
    orderedRow.maximumImpulse = input.maximumImpulse;
    orderedRow.component.outwardVelocity =
        input.row.outwardVelocity;
    orderedRow.component.stableKey = input.row.stableKey;
    orderedRow.component.originalIndex = rowIndex;
    orderedRow.component.terms.reserve(input.row.terms.size());
    for (const ComponentProjectionTerm &term : input.row.terms) {
      if (term.bodyIndex >= bodies.size() ||
          !UnilateralProjectionDetail::finiteVec3(
              term.linearJacobian) ||
          !UnilateralProjectionDetail::finiteVec3(
              term.angularJacobian))
        return result;
      orderedRow.component.terms.push_back(
          {term, bodies[term.bodyIndex].stableKey});
    }
    std::sort(orderedRow.component.terms.begin(),
              orderedRow.component.terms.end(),
              ComponentProjectionDetail::orderedTermLess);
    for (size_t term = 1;
         term < orderedRow.component.terms.size(); ++term) {
      if (orderedRow.component.terms[term - 1].term.bodyIndex ==
          orderedRow.component.terms[term].term.bodyIndex)
        return result;
    }

    ordered.push_back(orderedRow);
    rowKeys.push_back(input.row.stableKey);
    velocityScale =
        std::max(velocityScale,
                 std::fabs(static_cast<double>(
                     input.row.outwardVelocity)));
    needsCorrection =
        needsCorrection || input.row.outwardVelocity > 0.0f;
  }
  if (!needsCorrection || ordered.empty()) {
    result.status = BoundedComponentProjectionStatus::NoCorrection;
    return result;
  }
  std::sort(rowKeys.begin(), rowKeys.end());
  if (std::adjacent_find(rowKeys.begin(), rowKeys.end()) !=
      rowKeys.end())
    return result;
  std::sort(ordered.begin(), ordered.end(), orderedRowLess);

  const int rowCount = static_cast<int>(ordered.size());
  std::vector<double> response(
      static_cast<size_t>(rowCount * rowCount), 0.0);
  std::vector<double> outwardVelocity(
      static_cast<size_t>(rowCount), 0.0);
  std::vector<double> upperBounds(
      static_cast<size_t>(rowCount), 0.0);
  double lipschitzBound = 0.0;
  double impulseScale = 1.0;
  for (int row = 0; row < rowCount; ++row) {
    outwardVelocity[static_cast<size_t>(row)] =
        static_cast<double>(
            ordered[static_cast<size_t>(row)]
                .component.outwardVelocity);
    upperBounds[static_cast<size_t>(row)] =
        static_cast<double>(
            ordered[static_cast<size_t>(row)].maximumImpulse);
    impulseScale =
        std::max(impulseScale,
                 upperBounds[static_cast<size_t>(row)]);
    double absoluteRowSum = 0.0;
    for (int column = 0; column < rowCount; ++column) {
      const double value = ComponentProjectionDetail::response(
          ordered[static_cast<size_t>(row)].component,
          ordered[static_cast<size_t>(column)].component, bodies);
      if (!std::isfinite(value))
        return result;
      response[static_cast<size_t>(row * rowCount + column)] =
          value;
      absoluteRowSum += std::fabs(value);
    }
    lipschitzBound = std::max(lipschitzBound, absoluteRowSum);
  }

  const double feasibilityTolerance =
      static_cast<double>(relativeTolerance) * velocityScale;
  const double kktTolerance = feasibilityTolerance;
  const double boundTolerance =
      static_cast<double>(relativeTolerance) * impulseScale;
  if (!std::isfinite(lipschitzBound) ||
      lipschitzBound <= 1.0e-14) {
    result.status = BoundedComponentProjectionStatus::Infeasible;
    result.maximumResidual =
        static_cast<float>(velocityScale);
    return result;
  }

  const double step = 1.0 / lipschitzBound;
  std::vector<double> impulses(static_cast<size_t>(rowCount), 0.0);
  std::vector<double> extrapolated = impulses;
  std::vector<double> next(static_cast<size_t>(rowCount), 0.0);
  std::vector<double> gradientValues;
  double acceleration = 1.0;
  double currentObjective = 0.0;
  const int iterationLimit =
      std::max(4096, 1024 + 512 * rowCount);
  bool converged = false;

  for (int iteration = 0; iteration < iterationLimit; ++iteration) {
    gradient(response, outwardVelocity, extrapolated, rowCount,
             gradientValues);
    for (int row = 0; row < rowCount; ++row) {
      next[static_cast<size_t>(row)] = std::min(
          upperBounds[static_cast<size_t>(row)],
          std::max(0.0,
                   extrapolated[static_cast<size_t>(row)] -
                       step *
                           gradientValues[static_cast<size_t>(row)]));
    }
    double nextObjective =
        objective(response, outwardVelocity, next, rowCount);
    if (!std::isfinite(nextObjective)) {
      result.status =
          BoundedComponentProjectionStatus::NumericalFailure;
      return result;
    }

    const double objectiveSlack =
        1.0e-13 * std::max(1.0, std::fabs(currentObjective));
    if (nextObjective > currentObjective + objectiveSlack) {
      // Monotone restart from the last accepted iterate.
      extrapolated = impulses;
      acceleration = 1.0;
      ++result.restarts;
      gradient(response, outwardVelocity, extrapolated, rowCount,
               gradientValues);
      for (int row = 0; row < rowCount; ++row) {
        next[static_cast<size_t>(row)] = std::min(
            upperBounds[static_cast<size_t>(row)],
            std::max(0.0,
                     extrapolated[static_cast<size_t>(row)] -
                         step *
                             gradientValues[static_cast<size_t>(row)]));
      }
      nextObjective =
          objective(response, outwardVelocity, next, rowCount);
      if (!std::isfinite(nextObjective) ||
          nextObjective > currentObjective + 16.0 * objectiveSlack) {
        result.status =
            BoundedComponentProjectionStatus::NumericalFailure;
        return result;
      }
    }

    impulses.swap(next);
    currentObjective = nextObjective;
    result.iterations = iteration + 1;
    gradient(response, outwardVelocity, impulses, rowCount,
             gradientValues);
    const double kktViolation = projectedGradientViolation(
        gradientValues, impulses, upperBounds, boundTolerance);
    if (kktViolation <= kktTolerance) {
      converged = true;
      break;
    }

    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(
                         1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        (acceleration - 1.0) / nextAcceleration;
    for (int row = 0; row < rowCount; ++row) {
      extrapolated[static_cast<size_t>(row)] =
          impulses[static_cast<size_t>(row)] +
          momentum *
              (impulses[static_cast<size_t>(row)] -
               next[static_cast<size_t>(row)]);
    }
    acceleration = nextAcceleration;

    // Periodic restart bounds drift on rank-deficient Gram matrices.
    if ((iteration + 1) % 64 == 0) {
      extrapolated = impulses;
      acceleration = 1.0;
      ++result.restarts;
    }
  }

  if (!converged) {
    result.status =
        BoundedComponentProjectionStatus::IterationLimit;
    return result;
  }

  gradient(response, outwardVelocity, impulses, rowCount,
           gradientValues);
  result.objective = objective(response, outwardVelocity, impulses,
                               rowCount);
  result.maximumKktViolation = static_cast<float>(
      projectedGradientViolation(gradientValues, impulses,
                                 upperBounds, boundTolerance));

  std::vector<BoundedComponentRowState> orderedStates(
      static_cast<size_t>(rowCount),
      BoundedComponentRowState::Lower);
  double maximumResidual = 0.0;
  for (int row = 0; row < rowCount; ++row) {
    const double impulse = impulses[static_cast<size_t>(row)];
    const double upper = upperBounds[static_cast<size_t>(row)];
    BoundedComponentRowState state =
        BoundedComponentRowState::Free;
    if (upper <= boundTolerance ||
        upper - impulse <= boundTolerance) {
      state = BoundedComponentRowState::Upper;
      ++result.upperRows;
    } else if (impulse <= boundTolerance) {
      state = BoundedComponentRowState::Lower;
      ++result.lowerRows;
    } else {
      ++result.freeRows;
    }
    orderedStates[static_cast<size_t>(row)] = state;
    result.rowStates[
        ordered[static_cast<size_t>(row)].component.originalIndex] =
        state;
    maximumResidual =
        std::max(maximumResidual,
                 -gradientValues[static_cast<size_t>(row)]);
  }
  result.maximumResidual =
      static_cast<float>(maximumResidual);
  buildVelocityDeltas(bodies, ordered, impulses,
                      result.candidateImpulses,
                      result.candidateVelocityDeltas);

  if (!std::isfinite(maximumResidual) ||
      maximumResidual > 4.0 * feasibilityTolerance) {
    std::vector<ComponentProjectionRow> unboundedRows;
    unboundedRows.reserve(ordered.size());
    for (const OrderedRow &row : ordered) {
      ComponentProjectionRow original;
      original.outwardVelocity = row.component.outwardVelocity;
      original.stableKey = row.component.stableKey;
      for (const ComponentProjectionDetail::OrderedTerm &term :
           row.component.terms)
        original.terms.push_back(term.term);
      unboundedRows.push_back(original);
    }
    const ComponentProjectionResult unbounded =
        solveComponentUnilateralProjection(
            bodies, unboundedRows, relativeTolerance);
    if (unbounded.status == UnilateralProjectionStatus::Infeasible) {
      result.status = BoundedComponentProjectionStatus::Infeasible;
    } else if (unbounded.status == UnilateralProjectionStatus::Solved) {
      result.status =
          BoundedComponentProjectionStatus::BudgetExhausted;
    } else {
      result.status =
          BoundedComponentProjectionStatus::NumericalFailure;
    }
    return result;
  }

  result.impulses = result.candidateImpulses;
  result.velocityDeltas = result.candidateVelocityDeltas;
  result.status = BoundedComponentProjectionStatus::Solved;
  return result;
}

} // namespace AvbdRef
