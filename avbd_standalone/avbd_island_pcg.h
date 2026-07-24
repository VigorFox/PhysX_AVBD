#pragma once

#include "avbd_math.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace AvbdRef {

enum class IslandRowOwner : uint8_t {
  D6,
  Contact,
  Gear,
  Articulation,
  Attachment,
  Other
};

/**
 * One frozen scalar row in an island Newton linearization.
 *
 * bodyA/bodyB are compact island slots, not Solver::bodies indices.
 * UINT32_MAX denotes a world/static endpoint and requires a zero Jacobian.
 */
struct IslandPcgRow {
  IslandRowOwner owner = IslandRowOwner::Other;
  uint32_t ownerIndex = UINT32_MAX;
  uint16_t rowSlot = 0;
  uint16_t activeMode = 0;
  uint32_t bodyA = UINT32_MAX;
  uint32_t bodyB = UINT32_MAX;
  Vec6 jacobianA;
  Vec6 jacobianB;
  float violation = 0.0f;
  float penalty = 0.0f;
  float force = 0.0f;
  bool internalTranslationInvariant = false;
};

struct IslandPcgStats {
  int iterations = 0;
  double initialPreconditionedResidual = 0.0;
  double finalPreconditionedResidual = 0.0;
  bool converged = false;
  bool breakdown = false;
  bool finite = true;
};

inline Vec6 islandVec6Subtract(const Vec6 &a, const Vec6 &b) {
  Vec6 result;
  for (int i = 0; i < 6; ++i)
    result[i] = a[i] - b[i];
  return result;
}

inline void islandVec6AddScaled(Vec6 &target, const Vec6 &value,
                                float scale) {
  for (int i = 0; i < 6; ++i)
    target[i] += value[i] * scale;
}

inline double islandVectorDot(const std::vector<Vec6> &a,
                              const std::vector<Vec6> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    for (int k = 0; k < 6; ++k)
      result += double(a[i][k]) * double(b[i][k]);
  return result;
}

inline bool islandVectorFinite(const std::vector<Vec6> &values) {
  for (const Vec6 &value : values)
    for (int k = 0; k < 6; ++k)
      if (!std::isfinite(value[k]))
        return false;
  return true;
}

/**
 * Matrix-free SPD system
 *
 *   H = blockdiag(K_i) + sum_r p_r J_r^T J_r
 *   g = K d + sum_r J_r^T f_r
 *
 * The row list is frozen for the duration of solvePcg().
 */
class IslandPcgSystem {
public:
  void initialize(const std::vector<Mat66> &inertialBlocks,
                  const std::vector<Vec6> &inertialGradient) {
    mInertialBlocks = inertialBlocks;
    mGradient = inertialGradient;
    mRows.clear();
    mPreconditioner = inertialBlocks;
  }

  size_t bodyCount() const { return mInertialBlocks.size(); }
  const std::vector<IslandPcgRow> &rows() const { return mRows; }
  const std::vector<Vec6> &gradient() const { return mGradient; }
  const std::vector<Mat66> &preconditioner() const {
    return mPreconditioner;
  }

  bool addRow(const IslandPcgRow &row) {
    if (!(row.penalty >= 0.0f) || !std::isfinite(row.penalty) ||
        !std::isfinite(row.force) || !std::isfinite(row.violation))
      return false;
    if (row.bodyA >= bodyCount() && row.bodyA != UINT32_MAX)
      return false;
    if (row.bodyB >= bodyCount() && row.bodyB != UINT32_MAX)
      return false;
    if (row.bodyA == UINT32_MAX && row.bodyB == UINT32_MAX)
      return false;

    if (row.internalTranslationInvariant && row.bodyA != UINT32_MAX &&
        row.bodyB != UINT32_MAX) {
      const Vec3 sum = row.jacobianA.linear() + row.jacobianB.linear();
      if (sum.length() > 1e-5f)
        return false;
    }

    mRows.push_back(row);
    if (row.bodyA != UINT32_MAX) {
      islandVec6AddScaled(mGradient[row.bodyA], row.jacobianA, row.force);
      mPreconditioner[row.bodyA] +=
          outer(row.jacobianA, row.jacobianA * row.penalty);
    }
    if (row.bodyB != UINT32_MAX) {
      islandVec6AddScaled(mGradient[row.bodyB], row.jacobianB, row.force);
      mPreconditioner[row.bodyB] +=
          outer(row.jacobianB, row.jacobianB * row.penalty);
    }
    return true;
  }

  void apply(const std::vector<Vec6> &input,
             std::vector<Vec6> &output) const {
    output.assign(bodyCount(), Vec6());
    for (size_t body = 0; body < bodyCount(); ++body)
      output[body] = mInertialBlocks[body] * input[body];

    for (const IslandPcgRow &row : mRows) {
      double projection = 0.0;
      if (row.bodyA != UINT32_MAX)
        projection += dot(row.jacobianA, input[row.bodyA]);
      if (row.bodyB != UINT32_MAX)
        projection += dot(row.jacobianB, input[row.bodyB]);
      const float scale = row.penalty * float(projection);
      if (row.bodyA != UINT32_MAX)
        islandVec6AddScaled(output[row.bodyA], row.jacobianA, scale);
      if (row.bodyB != UINT32_MAX)
        islandVec6AddScaled(output[row.bodyB], row.jacobianB, scale);
    }
  }

  IslandPcgStats solvePcg(std::vector<Vec6> &solution,
                          double relativeTolerance = 1e-7,
                          int maxIterations = 0) const {
    IslandPcgStats stats;
    const size_t count = bodyCount();
    solution.assign(count, Vec6());
    if (count == 0) {
      stats.converged = true;
      return stats;
    }
    if (mGradient.size() != count || mPreconditioner.size() != count ||
        !islandVectorFinite(mGradient)) {
      stats.finite = false;
      return stats;
    }

    if (maxIterations <= 0)
      maxIterations = std::max(1, int(count) * 6);

    std::vector<Vec6> residual = mGradient;
    std::vector<Vec6> preconditioned(count);
    std::vector<Vec6> direction(count);
    std::vector<Vec6> operatorDirection;
    for (size_t body = 0; body < count; ++body)
      preconditioned[body] = solveLDLT(mPreconditioner[body], residual[body]);
    direction = preconditioned;

    double residualDotPreconditioned =
        islandVectorDot(residual, preconditioned);
    if (!std::isfinite(residualDotPreconditioned) ||
        residualDotPreconditioned < 0.0) {
      stats.breakdown = true;
      stats.finite = false;
      return stats;
    }
    stats.initialPreconditionedResidual =
        std::sqrt(residualDotPreconditioned);
    stats.finalPreconditionedResidual =
        stats.initialPreconditionedResidual;
    const double target =
        relativeTolerance * std::max(1.0, stats.initialPreconditionedResidual);
    if (stats.initialPreconditionedResidual <= target) {
      stats.converged = true;
      return stats;
    }

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
      apply(direction, operatorDirection);
      const double denominator =
          islandVectorDot(direction, operatorDirection);
      if (!std::isfinite(denominator) || denominator <= 1e-30) {
        stats.breakdown = true;
        break;
      }
      const double alpha = residualDotPreconditioned / denominator;
      if (!std::isfinite(alpha)) {
        stats.breakdown = true;
        stats.finite = false;
        break;
      }
      for (size_t body = 0; body < count; ++body) {
        islandVec6AddScaled(solution[body], direction[body], float(alpha));
        islandVec6AddScaled(residual[body], operatorDirection[body],
                            float(-alpha));
        preconditioned[body] =
            solveLDLT(mPreconditioner[body], residual[body]);
      }
      const double nextResidualDotPreconditioned =
          islandVectorDot(residual, preconditioned);
      stats.iterations = iteration + 1;
      if (!std::isfinite(nextResidualDotPreconditioned) ||
          nextResidualDotPreconditioned < 0.0) {
        stats.breakdown = true;
        stats.finite = false;
        break;
      }
      stats.finalPreconditionedResidual =
          std::sqrt(nextResidualDotPreconditioned);
      if (stats.finalPreconditionedResidual <= target) {
        stats.converged = true;
        break;
      }
      if (residualDotPreconditioned <= 1e-30) {
        stats.breakdown = true;
        break;
      }
      const double beta =
          nextResidualDotPreconditioned / residualDotPreconditioned;
      for (size_t body = 0; body < count; ++body) {
        for (int k = 0; k < 6; ++k)
          direction[body][k] =
              preconditioned[body][k] + float(beta) * direction[body][k];
      }
      residualDotPreconditioned = nextResidualDotPreconditioned;
    }
    stats.finite = stats.finite && islandVectorFinite(solution) &&
                   islandVectorFinite(residual);
    return stats;
  }

private:
  std::vector<Mat66> mInertialBlocks;
  std::vector<IslandPcgRow> mRows;
  std::vector<Vec6> mGradient;
  std::vector<Mat66> mPreconditioner;
};

} // namespace AvbdRef
