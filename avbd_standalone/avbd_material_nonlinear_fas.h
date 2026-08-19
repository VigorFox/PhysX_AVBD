#pragma once

#include "avbd_material_graph_multilevel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace AvbdRef {

struct MaterialNonlinearFasLevel {
  size_t stateCount = 0;
  size_t coarseCount = 0;
  // Row-major stateCount x coarseCount.
  std::vector<double> prolongation;
  // Row-major coarseCount x stateCount. This is the Galerkin-energy
  // left inverse Q=(P^T A P)^-1 P^T A used to restrict absolute state.
  std::vector<double> stateRestriction;
  // Row-major coarseCount x stateCount. This is P^T and restricts
  // residual/forcing in the work-adjoint metric.
  std::vector<double> residualRestriction;
};

struct MaterialNonlinearFasHierarchy {
  std::vector<MaterialNonlinearFasLevel> levels;
  size_t componentCount = 0;
  double maximumLeftInverseError = 0.0;
  int failureLevel = -1;
  int failureCode = 0;
  bool finite = false;
};

struct MaterialNonlinearFasStats {
  int cycles = 0;
  int operatorEvaluations = 0;
  int smoothingSweeps = 0;
  int coarseSolves = 0;
  int coarseVisits = 0;
  int maximumCoarseIterations = 0;
  int correctionTrials = 0;
  int acceptedCorrections = 0;
  int rejectedCorrections = 0;
  double maximumTau = 0.0;
  double maximumRestrictedDefect = 0.0;
  bool finite = true;
};

namespace MaterialNonlinearFasDetail {

inline bool finiteVector(const std::vector<double> &values) {
  for (double value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

inline void multiplyRectangular(
    const std::vector<double> &matrix, size_t rowCount,
    size_t columnCount, const std::vector<double> &input,
    std::vector<double> &output) {
  output.assign(rowCount, 0.0);
  for (size_t row = 0; row < rowCount; ++row) {
    for (size_t column = 0; column < columnCount; ++column) {
      output[row] +=
          matrix[row * columnCount + column] * input[column];
    }
  }
}

template <typename Evaluate, typename Smooth,
          typename CoarseSolve, typename Merit>
bool cycle(const MaterialNonlinearFasHierarchy &hierarchy,
           size_t levelIndex, const std::vector<double> &rhs,
           std::vector<double> &state, int preSmooth,
           int postSmooth, int maxCoarseCycles,
           double coarseReduction, Evaluate &evaluate,
           Smooth &smooth,
           CoarseSolve &coarseSolve, Merit &merit,
           MaterialNonlinearFasStats &stats) {
  const MaterialNonlinearFasLevel &level =
      hierarchy.levels[levelIndex];
  if (rhs.size() != level.stateCount)
    return false;
  if (state.size() != level.stateCount)
    state.assign(level.stateCount, 0.0);

  if (levelIndex + 1u == hierarchy.levels.size()) {
    ++stats.coarseSolves;
    if (!coarseSolve(levelIndex, rhs, state, stats) ||
        !finiteVector(state)) {
      return false;
    }
    return true;
  }

  if (preSmooth > 0) {
    if (!smooth(levelIndex, rhs, state, preSmooth, stats) ||
        !finiteVector(state)) {
      return false;
    }
    stats.smoothingSweeps += preSmooth;
  }

  std::vector<double> applied;
  ++stats.operatorEvaluations;
  if (!evaluate(levelIndex, state, applied) ||
      applied.size() != level.stateCount ||
      !finiteVector(applied)) {
    return false;
  }
  std::vector<double> defect(level.stateCount, 0.0);
  for (size_t entry = 0; entry < level.stateCount; ++entry)
    defect[entry] = rhs[entry] - applied[entry];

  std::vector<double> coarseState;
  multiplyRectangular(
      level.stateRestriction, level.coarseCount,
      level.stateCount, state, coarseState);
  std::vector<double> restrictedDefect;
  multiplyRectangular(
      level.residualRestriction, level.coarseCount,
      level.stateCount, defect, restrictedDefect);
  for (double value : restrictedDefect) {
    stats.maximumRestrictedDefect =
        std::max(stats.maximumRestrictedDefect,
                 std::fabs(value));
  }

  std::vector<double> coarseApplied;
  ++stats.operatorEvaluations;
  if (!evaluate(levelIndex + 1u, coarseState,
                coarseApplied) ||
      coarseApplied.size() != level.coarseCount ||
      !finiteVector(coarseApplied)) {
    return false;
  }
  std::vector<double> restrictedFineApplied;
  multiplyRectangular(
      level.residualRestriction, level.coarseCount,
      level.stateCount, applied, restrictedFineApplied);
  for (size_t entry = 0; entry < level.coarseCount; ++entry) {
    const double tau =
        coarseApplied[entry] - restrictedFineApplied[entry];
    stats.maximumTau =
        std::max(stats.maximumTau, std::fabs(tau));
  }

  // F_H(Q x_h) + P^T (b_h - F_h(x_h)).
  std::vector<double> coarseRhs(level.coarseCount, 0.0);
  for (size_t entry = 0; entry < level.coarseCount; ++entry) {
    coarseRhs[entry] =
        coarseApplied[entry] + restrictedDefect[entry];
  }
  std::vector<double> correctedCoarseState = coarseState;
  double initialCoarseDefect = 0.0;
  for (double value : restrictedDefect) {
    initialCoarseDefect =
        std::max(initialCoarseDefect, std::fabs(value));
  }
  const double coarseTarget =
      std::max(1.0e-13,
               coarseReduction * initialCoarseDefect);
  bool coarseConverged =
      initialCoarseDefect <= coarseTarget;
  for (int coarseIteration = 0;
       coarseIteration < maxCoarseCycles &&
       !coarseConverged;
       ++coarseIteration) {
    ++stats.coarseVisits;
    stats.maximumCoarseIterations =
        std::max(stats.maximumCoarseIterations,
                 coarseIteration + 1);
    if (!cycle(hierarchy, levelIndex + 1u, coarseRhs,
               correctedCoarseState, preSmooth, postSmooth,
               maxCoarseCycles, coarseReduction, evaluate,
               smooth, coarseSolve, merit, stats)) {
      return false;
    }
    std::vector<double> correctedCoarseApplied;
    ++stats.operatorEvaluations;
    if (!evaluate(levelIndex + 1u, correctedCoarseState,
                  correctedCoarseApplied) ||
        correctedCoarseApplied.size() != level.coarseCount ||
        !finiteVector(correctedCoarseApplied)) {
      return false;
    }
    double coarseDefect = 0.0;
    for (size_t entry = 0; entry < level.coarseCount;
         ++entry) {
      coarseDefect =
          std::max(
              coarseDefect,
              std::fabs(
                  coarseRhs[entry] -
                  correctedCoarseApplied[entry]));
    }
    coarseConverged = coarseDefect <= coarseTarget;
  }
  if (!coarseConverged)
    return false;

  std::vector<double> coarseCorrection(level.coarseCount, 0.0);
  for (size_t entry = 0; entry < level.coarseCount; ++entry) {
    coarseCorrection[entry] =
        correctedCoarseState[entry] - coarseState[entry];
  }
  std::vector<double> fineCorrection;
  multiplyRectangular(
      level.prolongation, level.stateCount,
      level.coarseCount, coarseCorrection, fineCorrection);
  const std::vector<double> preCorrectionState = state;
  double preCorrectionMerit = 0.0;
  if (!merit(levelIndex, rhs, preCorrectionState,
             preCorrectionMerit) ||
      !std::isfinite(preCorrectionMerit)) {
    return false;
  }
  bool acceptedCorrection = false;
  double relaxation = 1.0;
  for (int trialIndex = 0; trialIndex < 12; ++trialIndex) {
    ++stats.correctionTrials;
    std::vector<double> trial = preCorrectionState;
    for (size_t entry = 0; entry < level.stateCount; ++entry)
      trial[entry] += relaxation * fineCorrection[entry];
    double trialMerit = 0.0;
    if (!merit(levelIndex, rhs, trial, trialMerit) ||
        !std::isfinite(trialMerit)) {
      relaxation *= 0.5;
      continue;
    }
    if (trialMerit <=
        preCorrectionMerit +
            1.0e-14 *
                std::max(1.0,
                         std::fabs(preCorrectionMerit))) {
      state.swap(trial);
      acceptedCorrection = true;
      ++stats.acceptedCorrections;
      break;
    }
    relaxation *= 0.5;
  }
  if (!acceptedCorrection) {
    state = preCorrectionState;
    ++stats.rejectedCorrections;
  }

  if (postSmooth > 0) {
    if (!smooth(levelIndex, rhs, state, postSmooth, stats) ||
        !finiteVector(state)) {
      return false;
    }
    stats.smoothingSweeps += postSmooth;
  }
  return true;
}

} // namespace MaterialNonlinearFasDetail

inline MaterialNonlinearFasHierarchy
buildMaterialNonlinearFasHierarchy(
    const MaterialGraphMultilevelHierarchy &graphHierarchy,
    size_t componentCount) {
  MaterialNonlinearFasHierarchy result;
  if (!graphHierarchy.finite || graphHierarchy.levels.empty() ||
      componentCount == 0u) {
    return result;
  }
  result.componentCount = componentCount;
  result.levels.resize(graphHierarchy.levels.size());
  for (size_t levelIndex = 0;
       levelIndex < graphHierarchy.levels.size();
       ++levelIndex) {
    const MaterialGraphMultilevelLevel &graphLevel =
        graphHierarchy.levels[levelIndex];
    MaterialNonlinearFasLevel &level =
        result.levels[levelIndex];
    const size_t fineBodyCount = graphLevel.stableKeys.size();
    level.stateCount = fineBodyCount * componentCount;
    if (levelIndex + 1u == graphHierarchy.levels.size())
      continue;

    const size_t coarseBodyCount = graphLevel.coarseCount;
    if (coarseBodyCount == 0u ||
        graphLevel.prolongation.size() !=
            fineBodyCount * coarseBodyCount) {
      result.failureLevel = static_cast<int>(levelIndex);
      result.failureCode = 1;
      return result;
    }
    level.coarseCount = coarseBodyCount * componentCount;
    level.prolongation.assign(
        level.stateCount * level.coarseCount, 0.0);
    level.residualRestriction.assign(
        level.coarseCount * level.stateCount, 0.0);
    for (size_t fineBody = 0; fineBody < fineBodyCount;
         ++fineBody) {
      for (size_t coarseBody = 0;
           coarseBody < coarseBodyCount; ++coarseBody) {
        const double value =
            graphLevel.prolongation[
                fineBody * coarseBodyCount + coarseBody];
        for (size_t component = 0;
             component < componentCount; ++component) {
          const size_t fine =
              fineBody * componentCount + component;
          const size_t coarse =
              coarseBody * componentCount + component;
          level.prolongation[
              fine * level.coarseCount + coarse] = value;
          level.residualRestriction[
              coarse * level.stateCount + fine] = value;
        }
      }
    }

    const std::vector<double> &coarseMatrix =
        graphHierarchy.levels[levelIndex + 1u].matrix;
    if (coarseMatrix.size() !=
        coarseBodyCount * coarseBodyCount) {
      result.failureLevel = static_cast<int>(levelIndex);
      result.failureCode = 2;
      return result;
    }
    level.stateRestriction.assign(
        level.coarseCount * level.stateCount, 0.0);
    for (size_t fineBody = 0; fineBody < fineBodyCount;
         ++fineBody) {
      std::vector<double> rhs(coarseBodyCount, 0.0);
      for (size_t coarseBody = 0;
           coarseBody < coarseBodyCount; ++coarseBody) {
        for (size_t appliedBody = 0;
             appliedBody < fineBodyCount; ++appliedBody) {
          rhs[coarseBody] +=
              graphLevel.prolongation[
                  appliedBody * coarseBodyCount + coarseBody] *
              graphLevel.matrix[
                  appliedBody * fineBodyCount + fineBody];
        }
      }
      std::vector<double> scalarColumn;
      if (!MaterialGraphMultilevelDetail::solveDense(
              coarseMatrix, rhs, scalarColumn)) {
        result.failureLevel = static_cast<int>(levelIndex);
        result.failureCode = 3;
        return result;
      }
      for (size_t coarseBody = 0;
           coarseBody < coarseBodyCount; ++coarseBody) {
        for (size_t component = 0;
             component < componentCount; ++component) {
          const size_t fine =
              fineBody * componentCount + component;
          const size_t coarse =
              coarseBody * componentCount + component;
          level.stateRestriction[
              coarse * level.stateCount + fine] =
              scalarColumn[coarseBody];
        }
      }
    }

    for (size_t row = 0; row < level.coarseCount; ++row) {
      for (size_t column = 0; column < level.coarseCount;
           ++column) {
        double value = 0.0;
        for (size_t fine = 0; fine < level.stateCount;
             ++fine) {
          value +=
              level.stateRestriction[
                  row * level.stateCount + fine] *
              level.prolongation[
                  fine * level.coarseCount + column];
        }
        result.maximumLeftInverseError =
            std::max(
                result.maximumLeftInverseError,
                std::fabs(
                    value - (row == column ? 1.0 : 0.0)));
      }
    }
  }
  result.finite =
      std::isfinite(result.maximumLeftInverseError) &&
      result.maximumLeftInverseError <= 1.0e-10;
  if (!result.finite) {
    result.failureLevel =
        static_cast<int>(result.levels.size()) - 1;
    result.failureCode = 4;
  }
  return result;
}

template <typename Evaluate, typename Smooth,
          typename CoarseSolve, typename Merit>
inline bool applyMaterialNonlinearFasCycle(
    const MaterialNonlinearFasHierarchy &hierarchy,
    const std::vector<double> &rhs, std::vector<double> &state,
    Evaluate evaluate, Smooth smooth, CoarseSolve coarseSolve,
    Merit merit, MaterialNonlinearFasStats &stats,
    int preSmooth = 2, int postSmooth = 2,
    int maxCoarseCycles = 4,
    double coarseReduction = 0.1) {
  if (!hierarchy.finite || hierarchy.levels.empty() ||
      rhs.size() != hierarchy.levels.front().stateCount ||
      preSmooth < 0 || postSmooth < 0 ||
      maxCoarseCycles <= 0 ||
      !(coarseReduction > 0.0 && coarseReduction < 1.0) ||
      !MaterialNonlinearFasDetail::finiteVector(rhs)) {
    stats.finite = false;
    return false;
  }
  if (state.size() != rhs.size())
    state.assign(rhs.size(), 0.0);
  ++stats.cycles;
  if (!MaterialNonlinearFasDetail::cycle(
          hierarchy, 0u, rhs, state, preSmooth, postSmooth,
          maxCoarseCycles, coarseReduction, evaluate, smooth,
          coarseSolve, merit, stats)) {
    stats.finite = false;
    return false;
  }
  stats.finite =
      MaterialNonlinearFasDetail::finiteVector(state);
  return stats.finite;
}

} // namespace AvbdRef
