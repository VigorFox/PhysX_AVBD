#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace AvbdRef {

struct MaterialGraphMultilevelLevel {
  std::vector<uint64_t> stableKeys;
  // Row-major square operator at this level.
  std::vector<double> matrix;
  // Row-major fineCount x coarseCount prolongation. Empty on the last level.
  std::vector<double> prolongation;
  size_t coarseCount = 0;
  double jacobiWeight = 0.0;
};

struct MaterialGraphMultilevelHierarchy {
  std::vector<MaterialGraphMultilevelLevel> levels;
  // canonical index -> caller input index.
  std::vector<size_t> canonicalToInput;
  bool finite = false;
};

namespace MaterialGraphMultilevelDetail {

inline bool finiteMatrix(const std::vector<double> &matrix) {
  for (double value : matrix) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

inline bool solveDense(std::vector<double> matrix,
                       std::vector<double> rhs,
                       std::vector<double> &solution) {
  const size_t count = rhs.size();
  if (count == 0u || matrix.size() != count * count ||
      !finiteMatrix(matrix)) {
    return false;
  }
  for (size_t column = 0; column < count; ++column) {
    size_t pivot = column;
    double pivotMagnitude =
        std::fabs(matrix[column * count + column]);
    for (size_t row = column + 1u; row < count; ++row) {
      const double magnitude =
          std::fabs(matrix[row * count + column]);
      if (magnitude > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = magnitude;
      }
    }
    if (!(pivotMagnitude > 1.0e-18) ||
        !std::isfinite(pivotMagnitude)) {
      return false;
    }
    if (pivot != column) {
      for (size_t entry = column; entry < count; ++entry) {
        std::swap(matrix[column * count + entry],
                  matrix[pivot * count + entry]);
      }
      std::swap(rhs[column], rhs[pivot]);
    }
    const double inversePivot =
        1.0 / matrix[column * count + column];
    for (size_t row = column + 1u; row < count; ++row) {
      const double factor =
          matrix[row * count + column] * inversePivot;
      matrix[row * count + column] = 0.0;
      for (size_t entry = column + 1u; entry < count;
           ++entry) {
        matrix[row * count + entry] -=
            factor * matrix[column * count + entry];
      }
      rhs[row] -= factor * rhs[column];
    }
  }
  solution.assign(count, 0.0);
  for (size_t reverse = 0; reverse < count; ++reverse) {
    const size_t row = count - 1u - reverse;
    double value = rhs[row];
    for (size_t column = row + 1u; column < count; ++column)
      value -= matrix[row * count + column] * solution[column];
    solution[row] = value / matrix[row * count + row];
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

inline void multiply(const std::vector<double> &matrix,
                     const std::vector<double> &input,
                     std::vector<double> &output) {
  const size_t count = input.size();
  output.assign(count, 0.0);
  for (size_t row = 0; row < count; ++row) {
    for (size_t column = 0; column < count; ++column) {
      output[row] +=
          matrix[row * count + column] * input[column];
    }
  }
}

inline double rowScaledSpectralBound(
    const std::vector<double> &matrix, size_t count) {
  double bound = 0.0;
  for (size_t row = 0; row < count; ++row) {
    const double diagonal =
        std::fabs(matrix[row * count + row]);
    if (!(diagonal > 0.0))
      return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (size_t column = 0; column < count; ++column)
      sum += std::fabs(matrix[row * count + column]);
    bound = std::max(bound, sum / diagonal);
  }
  return bound;
}

inline bool buildAggregation(
    const MaterialGraphMultilevelLevel &fine,
    std::vector<double> &prolongation,
    std::vector<uint64_t> &coarseKeys, size_t &coarseCount) {
  const size_t count = fine.stableKeys.size();
  if (count <= 2u || fine.matrix.size() != count * count)
    return false;
  std::vector<int> aggregate(count, -1);
  std::vector<double> sign(count, 1.0);
  coarseKeys.clear();
  for (size_t row = 0; row < count; ++row) {
    if (aggregate[row] >= 0)
      continue;
    double bestStrength = 0.0;
    const double diagonalRow =
        std::fabs(fine.matrix[row * count + row]);
    for (size_t column = 0; column < count; ++column) {
      if (column == row || aggregate[column] >= 0)
        continue;
      const double diagonalColumn =
          std::fabs(fine.matrix[column * count + column]);
      const double denominator =
          std::sqrt(diagonalRow * diagonalColumn);
      if (!(denominator > 0.0))
        continue;
      const double coupling =
          fine.matrix[row * count + column];
      const double strength = std::fabs(coupling) / denominator;
      bestStrength = std::max(bestStrength, strength);
    }
    const int coarse = static_cast<int>(coarseKeys.size());
    aggregate[row] = coarse;
    coarseKeys.push_back(fine.stableKeys[row]);
    if (bestStrength <= 1.0e-14)
      continue;
    const double admissionStrength = 0.25 * bestStrength;
    for (size_t column = 0; column < count; ++column) {
      if (column == row || aggregate[column] >= 0)
        continue;
      const double diagonalColumn =
          std::fabs(fine.matrix[column * count + column]);
      const double denominator =
          std::sqrt(diagonalRow * diagonalColumn);
      if (!(denominator > 0.0))
        continue;
      const double coupling =
          fine.matrix[row * count + column];
      const double strength = std::fabs(coupling) / denominator;
      if (strength + 1.0e-15 < admissionStrength)
        continue;
      aggregate[column] = coarse;
      sign[column] = coupling > 0.0 ? -1.0 : 1.0;
      coarseKeys.back() =
          std::min(coarseKeys.back(), fine.stableKeys[column]);
    }
  }
  coarseCount = coarseKeys.size();
  if (coarseCount == 0u || coarseCount >= count)
    return false;

  std::vector<double> tentative(count * coarseCount, 0.0);
  for (size_t row = 0; row < count; ++row) {
    tentative[row * coarseCount +
              static_cast<size_t>(aggregate[row])] = sign[row];
  }

  const double spectralBound =
      rowScaledSpectralBound(fine.matrix, count);
  if (!(spectralBound > 0.0) ||
      !std::isfinite(spectralBound)) {
    return false;
  }
  const double smoothingWeight =
      std::min(0.8, 4.0 / (3.0 * spectralBound));
  prolongation = tentative;
  for (size_t row = 0; row < count; ++row) {
    const double diagonal = fine.matrix[row * count + row];
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
      return false;
    for (size_t coarse = 0; coarse < coarseCount; ++coarse) {
      double applied = 0.0;
      for (size_t column = 0; column < count; ++column) {
        applied +=
            fine.matrix[row * count + column] *
            tentative[column * coarseCount + coarse];
      }
      prolongation[row * coarseCount + coarse] -=
          smoothingWeight * applied / diagonal;
    }
  }
  return finiteMatrix(prolongation);
}

inline bool galerkinCoarse(
    const std::vector<double> &fineMatrix, size_t fineCount,
    const std::vector<double> &prolongation, size_t coarseCount,
    std::vector<double> &coarseMatrix) {
  if (fineMatrix.size() != fineCount * fineCount ||
      prolongation.size() != fineCount * coarseCount)
    return false;
  std::vector<double> applied(fineCount * coarseCount, 0.0);
  for (size_t row = 0; row < fineCount; ++row) {
    for (size_t coarse = 0; coarse < coarseCount; ++coarse) {
      for (size_t column = 0; column < fineCount; ++column) {
        applied[row * coarseCount + coarse] +=
            fineMatrix[row * fineCount + column] *
            prolongation[column * coarseCount + coarse];
      }
    }
  }
  coarseMatrix.assign(coarseCount * coarseCount, 0.0);
  for (size_t row = 0; row < coarseCount; ++row) {
    for (size_t column = 0; column < coarseCount; ++column) {
      for (size_t fine = 0; fine < fineCount; ++fine) {
        coarseMatrix[row * coarseCount + column] +=
            prolongation[fine * coarseCount + row] *
            applied[fine * coarseCount + column];
      }
    }
  }
  return finiteMatrix(coarseMatrix);
}

inline void smooth(const MaterialGraphMultilevelLevel &level,
                   const std::vector<double> &rhs,
                   std::vector<double> &solution,
                   int iterations) {
  const size_t count = rhs.size();
  std::vector<double> applied;
  std::vector<double> next = solution;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    multiply(level.matrix, solution, applied);
    for (size_t row = 0; row < count; ++row) {
      const double diagonal =
          level.matrix[row * count + row];
      next[row] =
          solution[row] +
          level.jacobiWeight *
              (rhs[row] - applied[row]) / diagonal;
    }
    solution.swap(next);
  }
}

inline bool vCycle(const MaterialGraphMultilevelHierarchy &hierarchy,
                   size_t levelIndex,
                   const std::vector<double> &rhs,
                   std::vector<double> &solution,
                   int preSmooth, int postSmooth) {
  const MaterialGraphMultilevelLevel &level =
      hierarchy.levels[levelIndex];
  const size_t count = level.stableKeys.size();
  if (rhs.size() != count)
    return false;
  if (solution.size() != count)
    solution.assign(count, 0.0);
  if (levelIndex + 1u == hierarchy.levels.size())
    return solveDense(level.matrix, rhs, solution);

  smooth(level, rhs, solution, preSmooth);
  std::vector<double> applied;
  multiply(level.matrix, solution, applied);
  std::vector<double> residual(count, 0.0);
  for (size_t row = 0; row < count; ++row)
    residual[row] = rhs[row] - applied[row];

  const size_t coarseCount = level.coarseCount;
  std::vector<double> coarseRhs(coarseCount, 0.0);
  for (size_t coarse = 0; coarse < coarseCount; ++coarse) {
    for (size_t fine = 0; fine < count; ++fine) {
      coarseRhs[coarse] +=
          level.prolongation[fine * coarseCount + coarse] *
          residual[fine];
    }
  }
  std::vector<double> coarseCorrection(coarseCount, 0.0);
  if (!vCycle(hierarchy, levelIndex + 1u, coarseRhs,
              coarseCorrection, preSmooth, postSmooth)) {
    return false;
  }
  for (size_t fine = 0; fine < count; ++fine) {
    for (size_t coarse = 0; coarse < coarseCount; ++coarse) {
      solution[fine] +=
          level.prolongation[fine * coarseCount + coarse] *
          coarseCorrection[coarse];
    }
  }
  smooth(level, rhs, solution, postSmooth);
  return finiteMatrix(solution);
}

} // namespace MaterialGraphMultilevelDetail

inline MaterialGraphMultilevelHierarchy
buildMaterialGraphMultilevelHierarchy(
    const std::vector<uint64_t> &inputStableKeys,
    const std::vector<double> &inputMatrix) {
  MaterialGraphMultilevelHierarchy result;
  const size_t count = inputStableKeys.size();
  if (count == 0u || inputMatrix.size() != count * count ||
      !MaterialGraphMultilevelDetail::finiteMatrix(inputMatrix)) {
    return result;
  }
  std::vector<std::pair<uint64_t, size_t>> order(count);
  for (size_t index = 0; index < count; ++index)
    order[index] = {inputStableKeys[index], index};
  std::sort(order.begin(), order.end());
  for (size_t index = 1; index < count; ++index) {
    if (order[index - 1u].first == order[index].first)
      return result;
  }
  result.canonicalToInput.resize(count);
  MaterialGraphMultilevelLevel fine;
  fine.stableKeys.resize(count);
  fine.matrix.assign(count * count, 0.0);
  for (size_t row = 0; row < count; ++row) {
    result.canonicalToInput[row] = order[row].second;
    fine.stableKeys[row] = order[row].first;
    for (size_t column = 0; column < count; ++column) {
      const double a =
          inputMatrix[order[row].second * count +
                      order[column].second];
      const double b =
          inputMatrix[order[column].second * count +
                      order[row].second];
      const double scale =
          std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
      if (std::fabs(a - b) > 1.0e-12 * scale)
        return result;
      fine.matrix[row * count + column] = 0.5 * (a + b);
    }
    if (!(fine.matrix[row * count + row] > 0.0))
      return result;
  }

  for (int depth = 0; depth < 32; ++depth) {
    const size_t levelCount = fine.stableKeys.size();
    const double spectralBound =
        MaterialGraphMultilevelDetail::rowScaledSpectralBound(
            fine.matrix, levelCount);
    if (!(spectralBound > 0.0) ||
        !std::isfinite(spectralBound)) {
      return MaterialGraphMultilevelHierarchy{};
    }
    fine.jacobiWeight =
        std::min(0.8, 1.2 / spectralBound);
    if (levelCount <= 2u) {
      result.levels.push_back(fine);
      result.finite = true;
      return result;
    }

    std::vector<uint64_t> coarseKeys;
    size_t coarseCount = 0u;
    if (!MaterialGraphMultilevelDetail::buildAggregation(
            fine, fine.prolongation, coarseKeys,
            coarseCount)) {
      return MaterialGraphMultilevelHierarchy{};
    }
    fine.coarseCount = coarseCount;
    std::vector<double> coarseMatrix;
    if (!MaterialGraphMultilevelDetail::galerkinCoarse(
            fine.matrix, levelCount, fine.prolongation,
            coarseCount, coarseMatrix)) {
      return MaterialGraphMultilevelHierarchy{};
    }
    result.levels.push_back(fine);
    fine = MaterialGraphMultilevelLevel{};
    fine.stableKeys = coarseKeys;
    fine.matrix = coarseMatrix;
  }
  return MaterialGraphMultilevelHierarchy{};
}

inline bool applyMaterialGraphMultilevelVCycle(
    const MaterialGraphMultilevelHierarchy &hierarchy,
    const std::vector<double> &inputRhs,
    std::vector<double> &inputSolution, int preSmooth = 2,
    int postSmooth = 2) {
  if (!hierarchy.finite || hierarchy.levels.empty() ||
      inputRhs.size() != hierarchy.canonicalToInput.size() ||
      preSmooth < 0 || postSmooth < 0) {
    return false;
  }
  const size_t count = inputRhs.size();
  if (inputSolution.size() != count)
    inputSolution.assign(count, 0.0);
  std::vector<double> canonicalRhs(count, 0.0);
  std::vector<double> canonicalSolution(count, 0.0);
  for (size_t canonical = 0; canonical < count; ++canonical) {
    const size_t input = hierarchy.canonicalToInput[canonical];
    canonicalRhs[canonical] = inputRhs[input];
    canonicalSolution[canonical] = inputSolution[input];
  }
  if (!MaterialGraphMultilevelDetail::vCycle(
          hierarchy, 0u, canonicalRhs, canonicalSolution,
          preSmooth, postSmooth)) {
    return false;
  }
  for (size_t canonical = 0; canonical < count; ++canonical) {
    inputSolution[hierarchy.canonicalToInput[canonical]] =
        canonicalSolution[canonical];
  }
  return true;
}

} // namespace AvbdRef
