#include "avbd_component_unilateral_projection.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using namespace AvbdRef;

extern int gTestsPassed;
extern int gTestsFailed;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      gTestsFailed++;                                                          \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define PASS(msg)                                                              \
  do {                                                                         \
    printf("  PASS: %s\n", msg);                                               \
    gTestsPassed++;                                                            \
    return true;                                                               \
  } while (0)

namespace {

static constexpr size_t kStaticBody = std::numeric_limits<size_t>::max();

struct BroadMaterialContact {
  size_t bodyA = kStaticBody;
  size_t bodyB = kStaticBody;
  Vec3 armA;
  Vec3 armB;
  Vec3 normal;
  Vec3 tangent[2];
  ComponentProjectionRow rows[3];
  float normalTarget = 0.0f;
  float friction = 0.85f;
  uint64_t stableKey = 0;
};

struct BroadMaterialLane {
  std::vector<Vec6> initialVelocity;
  std::vector<Vec6> committedVelocity;
  std::vector<double> committedImpulse;
  size_t bodyCount = 0;
  size_t contactCount = 0;
  size_t rowCount = 0;
  size_t restitutionContactCount = 0;
  int outerIterations = 0;
  int denseNormalSweeps = 0;
  int denseTangentSweeps = 0;
  double maximumProjectedResidual = 0.0;
  double projectedResidualTolerance = 0.0;
  double maximumNormalTargetError = 0.0;
  double bodyVelocityOracleDelta = 0.0;
  double linearMomentumResidual = 0.0;
  double angularMomentumResidual = 0.0;
  double initialEnergy = 0.0;
  double finalEnergy = 0.0;
  double responseDiagonalRatio = 0.0;
  double maximumCoulombViolation = 0.0;
  std::vector<Vec6> scalableVelocity;
  std::vector<double> scalableLayerNormalResidual;
  std::vector<double> scalableLayerTangentResidual;
  int scalableOuterIterations = 0;
  int scalableBlockIterations = 0;
  int scalableMatvecs = 0;
  int scalableResidualEvaluations = 0;
  int scalableProjectionResponseRebuilds = 0;
  int scalableProjectionCorrectionRows = 0;
  int scalableMappedChoices = 0;
  int scalableRelaxationChoices = 0;
  int scalableAndersonChoices = 0;
  double scalableInitialResidual = 0.0;
  double scalableProjectedResidual = 0.0;
  double scalableBodyVelocityDelta = 0.0;
  bool scalableCandidateConverged = false;
  bool candidateConverged = false;
  bool committed = false;
  bool finite = true;
};

static Vec3 rotateAboutY(const Vec3 &value, float angle) {
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  return Vec3(c * value.x - s * value.z, value.y,
              s * value.x + c * value.z);
}

static ComponentProjectionTerm makeBroadMaterialTerm(
    size_t body, const Vec3 &arm, const Vec3 &axis) {
  ComponentProjectionTerm term;
  term.bodyIndex = body;
  term.linearJacobian = axis;
  term.angularJacobian = arm.cross(axis);
  return term;
}

static double broadMaterialRowVelocity(
    const ComponentProjectionRow &row,
    const std::vector<Vec6> &velocities) {
  double value = 0.0;
  for (const ComponentProjectionTerm &term : row.terms) {
    value += static_cast<double>(term.linearJacobian.dot(
                 velocities[term.bodyIndex].linear())) +
             static_cast<double>(term.angularJacobian.dot(
                 velocities[term.bodyIndex].angular()));
  }
  return value;
}

static double broadMaterialRowResponse(
    const ComponentProjectionRow &a,
    const ComponentProjectionRow &b,
    const std::vector<ComponentProjectionBody> &bodies) {
  double value = 0.0;
  for (const ComponentProjectionTerm &at : a.terms) {
    for (const ComponentProjectionTerm &bt : b.terms) {
      if (at.bodyIndex != bt.bodyIndex)
        continue;
      const ComponentProjectionBody &body = bodies[at.bodyIndex];
      value +=
          static_cast<double>(body.inverseMassResponse) *
              static_cast<double>(
                  at.linearJacobian.dot(bt.linearJacobian)) +
          static_cast<double>(at.angularJacobian.dot(
              body.inverseInertiaResponse * bt.angularJacobian));
    }
  }
  return value;
}

static void addBroadMaterialRowImpulse(
    const ComponentProjectionRow &row, double impulse,
    const std::vector<ComponentProjectionBody> &bodies,
    std::vector<Vec6> &velocities) {
  for (const ComponentProjectionTerm &term : row.terms) {
    const ComponentProjectionBody &body = bodies[term.bodyIndex];
    velocities[term.bodyIndex] +=
        Vec6(term.linearJacobian * body.inverseMassResponse,
             body.inverseInertiaResponse * term.angularJacobian) *
        static_cast<float>(impulse);
  }
}

static double maximumVelocityDelta(const std::vector<Vec6> &a,
                                   const std::vector<Vec6> &b) {
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  for (size_t body = 0; body < a.size(); ++body) {
    for (int component = 0; component < 6; ++component) {
      maximum = std::max(
          maximum,
          std::fabs(static_cast<double>(a[body][component]) -
                    static_cast<double>(b[body][component])));
    }
  }
  return maximum;
}

static bool solveSmallDenseSystem(std::vector<double> matrix,
                                  std::vector<double> rhs,
                                  std::vector<double> &solution) {
  const size_t size = rhs.size();
  if (matrix.size() != size * size || size == 0u)
    return false;
  for (size_t column = 0; column < size; ++column) {
    size_t pivot = column;
    double pivotMagnitude =
        std::fabs(matrix[column * size + column]);
    for (size_t row = column + 1u; row < size; ++row) {
      const double magnitude =
          std::fabs(matrix[row * size + column]);
      if (magnitude > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = magnitude;
      }
    }
    if (!(pivotMagnitude > 1.0e-20) ||
        !std::isfinite(pivotMagnitude))
      return false;
    if (pivot != column) {
      for (size_t entry = column; entry < size; ++entry) {
        std::swap(matrix[column * size + entry],
                  matrix[pivot * size + entry]);
      }
      std::swap(rhs[column], rhs[pivot]);
    }
    const double inversePivot =
        1.0 / matrix[column * size + column];
    for (size_t row = column + 1u; row < size; ++row) {
      const double factor =
          matrix[row * size + column] * inversePivot;
      matrix[row * size + column] = 0.0;
      for (size_t entry = column + 1u; entry < size;
           ++entry) {
        matrix[row * size + entry] -=
            factor * matrix[column * size + entry];
      }
      rhs[row] -= factor * rhs[column];
    }
  }
  solution.assign(size, 0.0);
  for (size_t reverse = 0; reverse < size; ++reverse) {
    const size_t row = size - 1u - reverse;
    double value = rhs[row];
    for (size_t column = row + 1u; column < size; ++column)
      value -= matrix[row * size + column] * solution[column];
    solution[row] = value / matrix[row * size + row];
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

struct BroadAcceleratedBlockStats {
  int iterations = 0;
  int matvecs = 0;
  double projectedResidual = 0.0;
  bool converged = false;
  bool finite = true;
};

static void multiplyBroadAllRows(
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<double> &input,
    std::vector<double> &output, int *matvecCounter);

static double projectedMaterialResidual(
    const std::vector<BroadMaterialContact> &contacts,
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<double> &response,
    const std::vector<double> &impulses);

static void multiplyBroadSelectedRows(
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<size_t> &selectedRows,
    const std::vector<double> &input,
    std::vector<double> &output, int *matvecCounter = nullptr) {
  const size_t bodyCount = bodies.size();
  std::vector<double> linearImpulse(bodyCount * 3u, 0.0);
  std::vector<double> angularImpulse(bodyCount * 3u, 0.0);
  std::vector<double> linearDelta(bodyCount * 3u, 0.0);
  std::vector<double> angularDelta(bodyCount * 3u, 0.0);
  for (size_t selected = 0; selected < selectedRows.size();
       ++selected) {
    const ComponentProjectionRow &row = rows[selectedRows[selected]];
    const double impulse = input[selected];
    for (const ComponentProjectionTerm &term : row.terms) {
      const size_t offset = term.bodyIndex * 3u;
      linearImpulse[offset] +=
          static_cast<double>(term.linearJacobian.x) * impulse;
      linearImpulse[offset + 1u] +=
          static_cast<double>(term.linearJacobian.y) * impulse;
      linearImpulse[offset + 2u] +=
          static_cast<double>(term.linearJacobian.z) * impulse;
      angularImpulse[offset] +=
          static_cast<double>(term.angularJacobian.x) * impulse;
      angularImpulse[offset + 1u] +=
          static_cast<double>(term.angularJacobian.y) * impulse;
      angularImpulse[offset + 2u] +=
          static_cast<double>(term.angularJacobian.z) * impulse;
    }
  }
  for (size_t body = 0; body < bodyCount; ++body) {
    const size_t offset = body * 3u;
    const double inverseMass =
        static_cast<double>(bodies[body].inverseMassResponse);
    for (int component = 0; component < 3; ++component) {
      linearDelta[offset + static_cast<size_t>(component)] =
          inverseMass *
          linearImpulse[offset + static_cast<size_t>(component)];
      double value = 0.0;
      for (int column = 0; column < 3; ++column) {
        value +=
            static_cast<double>(
                bodies[body].inverseInertiaResponse
                    .m[component][column]) *
            angularImpulse[
                offset + static_cast<size_t>(column)];
      }
      angularDelta[offset + static_cast<size_t>(component)] =
          value;
    }
  }
  output.assign(selectedRows.size(), 0.0);
  for (size_t selected = 0; selected < selectedRows.size();
       ++selected) {
    const ComponentProjectionRow &row = rows[selectedRows[selected]];
    double value = 0.0;
    for (const ComponentProjectionTerm &term : row.terms) {
      const size_t offset = term.bodyIndex * 3u;
      value +=
          static_cast<double>(term.linearJacobian.x) *
              linearDelta[offset] +
          static_cast<double>(term.linearJacobian.y) *
              linearDelta[offset + 1u] +
          static_cast<double>(term.linearJacobian.z) *
              linearDelta[offset + 2u] +
          static_cast<double>(term.angularJacobian.x) *
              angularDelta[offset] +
          static_cast<double>(term.angularJacobian.y) *
              angularDelta[offset + 1u] +
          static_cast<double>(term.angularJacobian.z) *
              angularDelta[offset + 2u];
    }
    output[selected] = value;
  }
  if (matvecCounter)
    ++*matvecCounter;
}

static void multiplyBroadAllRows(
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<double> &input,
    std::vector<double> &output, int *matvecCounter = nullptr) {
  std::vector<size_t> allRows(rows.size());
  for (size_t row = 0; row < rows.size(); ++row)
    allRows[row] = row;
  multiplyBroadSelectedRows(rows, bodies, allRows, input, output,
                            matvecCounter);
}

static double broadMaterialRowDeltaVelocity(
    const ComponentProjectionRow &row,
    const std::vector<double> &linearDelta,
    const std::vector<double> &angularDelta) {
  double value = 0.0;
  for (const ComponentProjectionTerm &term : row.terms) {
    const size_t offset = term.bodyIndex * 3u;
    value +=
        static_cast<double>(term.linearJacobian.x) *
            linearDelta[offset] +
        static_cast<double>(term.linearJacobian.y) *
            linearDelta[offset + 1u] +
        static_cast<double>(term.linearJacobian.z) *
            linearDelta[offset + 2u] +
        static_cast<double>(term.angularJacobian.x) *
            angularDelta[offset] +
        static_cast<double>(term.angularJacobian.y) *
            angularDelta[offset + 1u] +
        static_cast<double>(term.angularJacobian.z) *
            angularDelta[offset + 2u];
  }
  return value;
}

static void addBroadMaterialRowDelta(
    const ComponentProjectionRow &row, double impulseDelta,
    const std::vector<ComponentProjectionBody> &bodies,
    std::vector<double> &linearDelta,
    std::vector<double> &angularDelta) {
  if (impulseDelta == 0.0)
    return;
  for (const ComponentProjectionTerm &term : row.terms) {
    const ComponentProjectionBody &body = bodies[term.bodyIndex];
    const size_t offset = term.bodyIndex * 3u;
    linearDelta[offset] +=
        static_cast<double>(body.inverseMassResponse) *
        static_cast<double>(term.linearJacobian.x) * impulseDelta;
    linearDelta[offset + 1u] +=
        static_cast<double>(body.inverseMassResponse) *
        static_cast<double>(term.linearJacobian.y) * impulseDelta;
    linearDelta[offset + 2u] +=
        static_cast<double>(body.inverseMassResponse) *
        static_cast<double>(term.linearJacobian.z) * impulseDelta;
    for (int component = 0; component < 3; ++component) {
      double response = 0.0;
      for (int column = 0; column < 3; ++column) {
        const double angularJacobian =
            column == 0
                ? static_cast<double>(term.angularJacobian.x)
                : (column == 1
                       ? static_cast<double>(
                             term.angularJacobian.y)
                       : static_cast<double>(
                             term.angularJacobian.z));
        response +=
            static_cast<double>(
                body.inverseInertiaResponse.m[component][column]) *
            angularJacobian;
      }
      angularDelta[offset + static_cast<size_t>(component)] +=
          response * impulseDelta;
    }
  }
}

static bool correctBroadMaterialResponseAfterProjection(
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<double> &unprojected,
    const std::vector<double> &projected,
    std::vector<double> &responseVelocity,
    BroadMaterialLane &result) {
  if (unprojected.size() != rows.size() ||
      projected.size() != rows.size() ||
      responseVelocity.size() != rows.size())
    return false;
  std::vector<double> linearDelta(bodies.size() * 3u, 0.0);
  std::vector<double> angularDelta(bodies.size() * 3u, 0.0);
  bool changed = false;
  for (size_t row = 0; row < rows.size(); ++row) {
    const double delta = projected[row] - unprojected[row];
    if (delta == 0.0)
      continue;
    changed = true;
    ++result.scalableProjectionCorrectionRows;
    addBroadMaterialRowDelta(rows[row], delta, bodies,
                             linearDelta, angularDelta);
  }
  if (!changed)
    return true;
  for (size_t row = 0; row < rows.size(); ++row) {
    responseVelocity[row] +=
        broadMaterialRowDeltaVelocity(
            rows[row], linearDelta, angularDelta);
  }
  ++result.scalableProjectionResponseRebuilds;
  return true;
}

static BroadAcceleratedBlockStats solveBroadMaterialAcceleratedBlock(
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<size_t> &selectedRows,
    const std::vector<double> &q,
    const std::vector<double> &caps, bool tangentBlock,
    double physicalTolerance, std::vector<double> &z) {
  BroadAcceleratedBlockStats stats;
  const size_t count = selectedRows.size();
  if (count == 0u || q.size() != count ||
      (tangentBlock && caps.size() * 2u != count)) {
    stats.finite = false;
    return stats;
  }

  std::vector<double> variableScale(count, 1.0);
  for (size_t row = 0; row < count; ++row) {
    const double diagonal = broadMaterialRowResponse(
        rows[selectedRows[row]], rows[selectedRows[row]], bodies);
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
      stats.finite = false;
      return stats;
    }
    variableScale[row] = std::sqrt(diagonal);
  }
  if (tangentBlock) {
    for (size_t contact = 0; contact < caps.size(); ++contact) {
      const size_t row = contact * 2u;
      const double sharedScale =
          std::sqrt(0.5 *
                    (variableScale[row] * variableScale[row] +
                     variableScale[row + 1u] *
                         variableScale[row + 1u]));
      variableScale[row] = sharedScale;
      variableScale[row + 1u] = sharedScale;
    }
  }

  if (z.size() != count)
    z.assign(count, 0.0);
  std::vector<double> scaledQ(count, 0.0);
  std::vector<double> scaledCaps = caps;
  std::vector<double> scaledZ(count, 0.0);
  for (size_t row = 0; row < count; ++row) {
    scaledQ[row] = q[row] / variableScale[row];
    scaledZ[row] = z[row] * variableScale[row];
  }
  if (tangentBlock) {
    for (size_t contact = 0; contact < caps.size(); ++contact) {
      scaledCaps[contact] *= variableScale[contact * 2u];
    }
  }
  double lipschitz = 1.0;
  std::vector<double> y = scaledZ;
  std::vector<double> previous = scaledZ;
  std::vector<double> ay(count, 0.0);
  std::vector<double> scaledAz(count, 0.0);
  std::vector<double> previousAz(count, 0.0);
  std::vector<double> gradient(count, 0.0);
  std::vector<double> trial(count, 0.0);
  std::vector<double> atrial(count, 0.0);
  double momentumState = 1.0;

  const auto multiplyScaled =
      [&](const std::vector<double> &input,
          std::vector<double> &output) {
        std::vector<double> physicalInput(count, 0.0);
        for (size_t row = 0; row < count; ++row)
          physicalInput[row] = input[row] / variableScale[row];
        multiplyBroadSelectedRows(rows, bodies, selectedRows,
                                  physicalInput, output,
                                  &stats.matvecs);
        for (size_t row = 0; row < count; ++row)
          output[row] /= variableScale[row];
      };
  const auto project =
      [&](std::vector<double> &values) {
        if (!tangentBlock) {
          for (double &value : values)
            value = std::max(0.0, value);
          return;
        }
        for (size_t contact = 0; contact < caps.size();
             ++contact) {
          const size_t row = contact * 2u;
          const double magnitude = std::sqrt(
              values[row] * values[row] +
              values[row + 1u] * values[row + 1u]);
          if (magnitude > scaledCaps[contact] &&
              magnitude > 0.0) {
            const double scale =
                scaledCaps[contact] / magnitude;
            values[row] *= scale;
            values[row + 1u] *= scale;
          }
        }
      };
  multiplyScaled(scaledZ, scaledAz);
  ay = scaledAz;

  for (int iteration = 0; iteration < 2048; ++iteration) {
    double objectiveY = 0.0;
    for (size_t row = 0; row < count; ++row) {
      gradient[row] = ay[row] + scaledQ[row];
      objectiveY +=
          0.5 * y[row] * ay[row] + scaledQ[row] * y[row];
    }

    bool accepted = false;
    for (int backtrack = 0; backtrack < 24; ++backtrack) {
      double gradientModel = objectiveY;
      for (size_t row = 0; row < count; ++row)
        trial[row] = y[row] - gradient[row] / lipschitz;
      project(trial);
      for (size_t row = 0; row < count; ++row) {
        const double delta = trial[row] - y[row];
        gradientModel +=
            gradient[row] * delta +
            0.5 * lipschitz * delta * delta;
      }
      multiplyScaled(trial, atrial);
      double trialObjective = 0.0;
      for (size_t row = 0; row < count; ++row) {
        trialObjective +=
            0.5 * trial[row] * atrial[row] +
            scaledQ[row] * trial[row];
      }
      if (std::isfinite(trialObjective) &&
          trialObjective <=
              gradientModel +
                  1.0e-12 * std::max(1.0, std::fabs(objectiveY))) {
        accepted = true;
        break;
      }
      lipschitz *= 2.0;
    }
    if (!accepted) {
      stats.finite = false;
      break;
    }

    stats.iterations = iteration + 1;
    stats.projectedResidual = 0.0;
    if (!tangentBlock) {
      for (size_t row = 0; row < count; ++row) {
        const double physicalTrial =
            trial[row] / variableScale[row];
        const double rowGradient =
            variableScale[row] * atrial[row] + q[row];
        const double projected =
            std::max(0.0, physicalTrial - rowGradient);
        stats.projectedResidual =
            std::max(stats.projectedResidual,
                     std::fabs(projected - physicalTrial));
      }
    } else {
      for (size_t contact = 0; contact < caps.size(); ++contact) {
        const size_t row = contact * 2u;
        const double physicalTangent0 =
            trial[row] / variableScale[row];
        const double physicalTangent1 =
            trial[row + 1u] / variableScale[row + 1u];
        double tangent0 =
            physicalTangent0 -
            (variableScale[row] * atrial[row] + q[row]);
        double tangent1 =
            physicalTangent1 -
            (variableScale[row + 1u] * atrial[row + 1u] +
             q[row + 1u]);
        const double magnitude =
            std::sqrt(tangent0 * tangent0 +
                      tangent1 * tangent1);
        if (magnitude > caps[contact] && magnitude > 0.0) {
          const double scale = caps[contact] / magnitude;
          tangent0 *= scale;
          tangent1 *= scale;
        }
        stats.projectedResidual =
            std::max(stats.projectedResidual,
                     std::fabs(tangent0 - physicalTangent0));
        stats.projectedResidual =
            std::max(stats.projectedResidual,
                     std::fabs(tangent1 - physicalTangent1));
      }
    }
    if (!std::isfinite(stats.projectedResidual)) {
      stats.finite = false;
      break;
    }
    double restartDot = 0.0;
    for (size_t row = 0; row < count; ++row) {
      restartDot +=
          (y[row] - trial[row]) *
          (trial[row] - scaledZ[row]);
    }
    previous = scaledZ;
    previousAz = scaledAz;
    scaledZ = trial;
    scaledAz = atrial;
    if (stats.projectedResidual <= physicalTolerance) {
      stats.converged = true;
      break;
    }

    const double nextMomentumState =
        0.5 *
        (1.0 + std::sqrt(1.0 +
                         4.0 * momentumState * momentumState));
    const double momentum =
        (momentumState - 1.0) / nextMomentumState;
    if (restartDot > 0.0) {
      y = scaledZ;
      ay = scaledAz;
      momentumState = 1.0;
    } else {
      for (size_t row = 0; row < count; ++row) {
        y[row] =
            scaledZ[row] +
            momentum * (scaledZ[row] - previous[row]);
        ay[row] =
            scaledAz[row] +
            momentum * (scaledAz[row] - previousAz[row]);
      }
      momentumState = nextMomentumState;
    }
    lipschitz =
        std::max(lipschitz * 0.9,
                 std::numeric_limits<double>::min());
  }
  for (size_t row = 0; row < count; ++row)
    z[row] = scaledZ[row] / variableScale[row];
  return stats;
}

static double projectedMaterialResidual(
    const std::vector<BroadMaterialContact> &contacts,
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<double> &response,
    const std::vector<double> &impulses) {
  const size_t rowCount = rows.size();
  std::vector<double> gradient(rowCount, 0.0);
  for (size_t row = 0; row < rowCount; ++row) {
    gradient[row] = static_cast<double>(rows[row].outwardVelocity);
    for (size_t column = 0; column < rowCount; ++column) {
      gradient[row] +=
          response[row * rowCount + column] * impulses[column];
    }
  }

  double maximum = 0.0;
  for (size_t contact = 0; contact < contacts.size(); ++contact) {
    const size_t row = contact * 3;
    const double projectedNormal =
        std::max(0.0, impulses[row] - gradient[row]);
    maximum =
        std::max(maximum, std::fabs(projectedNormal - impulses[row]));

    double tangent0 = impulses[row + 1] - gradient[row + 1];
    double tangent1 = impulses[row + 2] - gradient[row + 2];
    const double magnitude =
        std::sqrt(tangent0 * tangent0 + tangent1 * tangent1);
    const double cap =
        static_cast<double>(contacts[contact].friction) *
        impulses[row];
    if (magnitude > cap && magnitude > 0.0) {
      const double scale = cap / magnitude;
      tangent0 *= scale;
      tangent1 *= scale;
    }
    maximum =
        std::max(maximum, std::fabs(tangent0 - impulses[row + 1]));
    maximum =
        std::max(maximum, std::fabs(tangent1 - impulses[row + 2]));
  }
  return maximum;
}

static double projectedMaterialResidualFromResponse(
    const std::vector<BroadMaterialContact> &contacts,
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<double> &impulses,
    const std::vector<double> &responseVelocity,
    int *evaluationCounter = nullptr) {
  if (impulses.size() != rows.size() ||
      responseVelocity.size() != rows.size())
    return std::numeric_limits<double>::infinity();
  if (evaluationCounter)
    ++*evaluationCounter;
  double maximum = 0.0;
  for (size_t contact = 0; contact < contacts.size(); ++contact) {
    const size_t row = contact * 3u;
    const double normalGradient =
        static_cast<double>(rows[row].outwardVelocity) +
        responseVelocity[row];
    const double projectedNormal =
        std::max(0.0, impulses[row] - normalGradient);
    maximum =
        std::max(maximum,
                 std::fabs(projectedNormal - impulses[row]));

    double tangent0 =
        impulses[row + 1u] -
        (static_cast<double>(rows[row + 1u].outwardVelocity) +
         responseVelocity[row + 1u]);
    double tangent1 =
        impulses[row + 2u] -
        (static_cast<double>(rows[row + 2u].outwardVelocity) +
         responseVelocity[row + 2u]);
    const double magnitude =
        std::sqrt(tangent0 * tangent0 + tangent1 * tangent1);
    const double cap =
        static_cast<double>(contacts[contact].friction) *
        impulses[row];
    if (magnitude > cap && magnitude > 0.0) {
      const double scale = cap / magnitude;
      tangent0 *= scale;
      tangent1 *= scale;
    }
    maximum =
        std::max(maximum,
                 std::fabs(tangent0 - impulses[row + 1u]));
    maximum =
        std::max(maximum,
                 std::fabs(tangent1 - impulses[row + 2u]));
  }
  return maximum;
}

static double projectedMaterialResidualMatrixFree(
    const std::vector<BroadMaterialContact> &contacts,
    const std::vector<ComponentProjectionRow> &rows,
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<double> &impulses, int *matvecCounter,
    int *evaluationCounter = nullptr) {
  std::vector<double> responseVelocity;
  multiplyBroadAllRows(rows, bodies, impulses, responseVelocity,
                       matvecCounter);
  return projectedMaterialResidualFromResponse(
      contacts, rows, impulses, responseVelocity,
      evaluationCounter);
}

static BroadMaterialLane runBroadMaterialLane(
    bool includeTransientImpact, float yaw, bool reverseContacts,
    bool reverseBodyStorage, int outerBudget,
    bool runScalableCandidate = false,
    size_t bodyCountOverride = 0u) {
  BroadMaterialLane result;
  const size_t logicalBodyCount =
      bodyCountOverride != 0u
          ? bodyCountOverride
          : (includeTransientImpact ? 6u : 5u);
  result.bodyCount = logicalBodyCount;
  const float dt = 1.0f / 60.0f;
  const float supportSpeed = 9.81f * dt;
  const float slideSpeed = 3.0f;
  const float impactSpeed = 5.0f;
  const float restitution = 0.5f;
  const float bounceThreshold = 2.0f;
  const float referenceMasses[6] = {
      10.0f, 7.0f, 5.0f, 3.0f, 2.0f, 1.0f};
  const Vec3 normal(0.0f, 1.0f, 0.0f);
  const Vec3 driveAxis =
      rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), yaw);
  const Vec3 lateralAxis =
      rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), yaw);
  const Vec3 bottomArms[4] = {
      Vec3(-0.5f, -0.5f, -0.5f),
      Vec3(-0.5f, -0.5f, 0.5f),
      Vec3(0.5f, -0.5f, -0.5f),
      Vec3(0.5f, -0.5f, 0.5f)};
  const Vec3 topArms[4] = {
      Vec3(-0.5f, 0.5f, -0.5f),
      Vec3(-0.5f, 0.5f, 0.5f),
      Vec3(0.5f, 0.5f, -0.5f),
      Vec3(0.5f, 0.5f, 0.5f)};

  const auto storageIndex = [&](size_t logicalBody) {
    return reverseBodyStorage
               ? logicalBodyCount - 1u - logicalBody
               : logicalBody;
  };

  std::vector<ComponentProjectionBody> bodies(logicalBodyCount);
  std::vector<Vec3> bodyPositions(logicalBodyCount);
  std::vector<float> bodyMasses(logicalBodyCount, 0.0f);
  std::vector<Vec6> velocities(logicalBodyCount);
  for (size_t logicalBody = 0; logicalBody < logicalBodyCount;
       ++logicalBody) {
    const size_t body = storageIndex(logicalBody);
    const float mass =
        logicalBody < 6u ? referenceMasses[logicalBody] : 1.0f;
    bodies[body].inverseMassResponse = 1.0f / mass;
    const float inverseInertia = 6.0f / mass;
    bodies[body].inverseInertiaResponse =
        Mat33::diag(inverseInertia, inverseInertia,
                    inverseInertia);
    bodies[body].stableKey =
        static_cast<uint64_t>(100 + logicalBody);
    bodyMasses[body] = mass;
    bodyPositions[body] =
        Vec3(0.0f, 0.5f + static_cast<float>(logicalBody),
             0.0f);
    Vec3 linearVelocity =
        driveAxis * slideSpeed - normal * supportSpeed;
    if (logicalBody + 1u == logicalBodyCount &&
        includeTransientImpact) {
      linearVelocity =
          driveAxis * -1.0f + lateralAxis * 0.75f -
          normal * (supportSpeed + impactSpeed);
    }
    velocities[body] = Vec6(linearVelocity, Vec3());
  }
  const std::vector<Vec6> initialVelocities = velocities;

  std::vector<BroadMaterialContact> contacts;
  contacts.reserve(logicalBodyCount * 4u);
  const size_t layerCount = logicalBodyCount;
  for (size_t layer = 0; layer < layerCount; ++layer) {
    const bool groundLayer = layer == 0u;
    const bool restitutionLayer =
        includeTransientImpact && layer + 1u == layerCount;
    const size_t logicalBodyA = groundLayer ? 0u : layer;
    const size_t logicalBodyB =
        groundLayer ? kStaticBody : layer - 1u;
    for (size_t corner = 0; corner < 4u; ++corner) {
      BroadMaterialContact contact;
      contact.bodyA = storageIndex(logicalBodyA);
      contact.bodyB =
          logicalBodyB == kStaticBody
              ? kStaticBody
              : storageIndex(logicalBodyB);
      contact.armA = rotateAboutY(bottomArms[corner], yaw);
      if (contact.bodyB != kStaticBody)
        contact.armB = rotateAboutY(topArms[corner], yaw);
      contact.normal = normal;
      contact.tangent[0] = driveAxis;
      contact.tangent[1] = lateralAxis;
      contact.stableKey =
          static_cast<uint64_t>(1000 + layer * 16u + corner);

      for (int component = 0; component < 3; ++component) {
        const Vec3 axis =
            component == 0 ? contact.normal
                           : contact.tangent[component - 1];
        contact.rows[component].stableKey =
            contact.stableKey * 3u +
            static_cast<uint64_t>(component);
        contact.rows[component].terms.push_back(
            makeBroadMaterialTerm(contact.bodyA,
                                  contact.armA, axis));
        if (contact.bodyB != kStaticBody) {
          contact.rows[component].terms.push_back(
              makeBroadMaterialTerm(contact.bodyB,
                                    contact.armB, -axis));
        }
        if (reverseContacts) {
          std::reverse(contact.rows[component].terms.begin(),
                       contact.rows[component].terms.end());
        }
        std::sort(
            contact.rows[component].terms.begin(),
            contact.rows[component].terms.end(),
            [&](const ComponentProjectionTerm &a,
                const ComponentProjectionTerm &b) {
              return bodies[a.bodyIndex].stableKey <
                     bodies[b.bodyIndex].stableKey;
            });
      }

      const double preNormalVelocity =
          broadMaterialRowVelocity(contact.rows[0],
                                   initialVelocities);
      if (restitutionLayer &&
          preNormalVelocity < -bounceThreshold) {
        contact.normalTarget =
            static_cast<float>(-restitution *
                               preNormalVelocity);
        ++result.restitutionContactCount;
      }
      contact.rows[0].outwardVelocity =
          static_cast<float>(
              preNormalVelocity - contact.normalTarget);
      contact.rows[1].outwardVelocity =
          static_cast<float>(broadMaterialRowVelocity(
              contact.rows[1], initialVelocities));
      contact.rows[2].outwardVelocity =
          static_cast<float>(broadMaterialRowVelocity(
              contact.rows[2], initialVelocities));
      contacts.push_back(contact);
    }
  }
  if (reverseContacts)
    std::reverse(contacts.begin(), contacts.end());
  std::sort(contacts.begin(), contacts.end(),
            [](const BroadMaterialContact &a,
               const BroadMaterialContact &b) {
              return a.stableKey < b.stableKey;
            });

  std::vector<ComponentProjectionRow> rows;
  rows.reserve(contacts.size() * 3u);
  for (const BroadMaterialContact &contact : contacts) {
    rows.push_back(contact.rows[0]);
    rows.push_back(contact.rows[1]);
    rows.push_back(contact.rows[2]);
  }
  result.contactCount = contacts.size();
  result.rowCount = rows.size();

  const size_t rowCount = rows.size();
  std::vector<double> response(rowCount * rowCount, 0.0);
  double minimumDiagonal = std::numeric_limits<double>::infinity();
  double maximumDiagonal = 0.0;
  for (size_t row = 0; row < rowCount; ++row) {
    for (size_t column = 0; column < rowCount; ++column) {
      const double value =
          broadMaterialRowResponse(rows[row], rows[column],
                                   bodies);
      response[row * rowCount + column] = value;
    }
    const double diagonal = response[row * rowCount + row];
    minimumDiagonal = std::min(minimumDiagonal, diagonal);
    maximumDiagonal = std::max(maximumDiagonal, diagonal);
  }
  result.responseDiagonalRatio =
      maximumDiagonal / minimumDiagonal;

  std::vector<double> impulses(rowCount, 0.0);
  std::vector<double> gradient(rowCount, 0.0);
  for (size_t row = 0; row < rowCount; ++row)
    gradient[row] = static_cast<double>(rows[row].outwardVelocity);

  const auto applyImpulseDelta =
      [&](size_t row, double delta) {
        if (delta == 0.0)
          return;
        impulses[row] += delta;
        for (size_t affected = 0; affected < rowCount;
             ++affected) {
          gradient[affected] +=
              response[affected * rowCount + row] * delta;
        }
      };

  const int innerBudget = outerBudget <= 1 ? 1 : 8192;
  const double velocityScale =
      std::max(1.0, static_cast<double>(
                        supportSpeed + impactSpeed));
  const double strictTolerance = 2.0e-8 * velocityScale;
  result.projectedResidualTolerance = strictTolerance;
  for (int outer = 0; outer < outerBudget; ++outer) {
    for (int iteration = 0; iteration < innerBudget;
         ++iteration) {
      ++result.denseNormalSweeps;
      double maximumDelta = 0.0;
      for (size_t contact = 0; contact < contacts.size();
           ++contact) {
        const size_t row = contact * 3u;
        const double diagonal =
            response[row * rowCount + row];
        const double candidate =
            std::max(0.0,
                     impulses[row] - gradient[row] / diagonal);
        const double delta = candidate - impulses[row];
        maximumDelta =
            std::max(maximumDelta, std::fabs(delta));
        applyImpulseDelta(row, delta);
      }
      if (maximumDelta <= 1.0e-13)
        break;
    }

    for (int iteration = 0; iteration < innerBudget;
         ++iteration) {
      ++result.denseTangentSweeps;
      double maximumDelta = 0.0;
      for (size_t contact = 0; contact < contacts.size();
           ++contact) {
        const size_t row = contact * 3u;
        const size_t tangent0Row = row + 1u;
        const size_t tangent1Row = row + 2u;
        const double a00 =
            response[tangent0Row * rowCount + tangent0Row];
        const double a11 =
            response[tangent1Row * rowCount + tangent1Row];
        const double a01 =
            0.5 *
            (response[tangent0Row * rowCount + tangent1Row] +
             response[tangent1Row * rowCount + tangent0Row]);
        const double trace = a00 + a11;
        const double discriminant =
            std::sqrt(std::max(
                0.0, (a00 - a11) * (a00 - a11) +
                         4.0 * a01 * a01));
        const double blockLipschitz =
            0.5 * (trace + discriminant);
        const double step = 1.0 / blockLipschitz;
        double candidate0 =
            impulses[tangent0Row] -
            step * gradient[tangent0Row];
        double candidate1 =
            impulses[tangent1Row] -
            step * gradient[tangent1Row];
        const double magnitude =
            std::sqrt(candidate0 * candidate0 +
                      candidate1 * candidate1);
        const double cap =
            static_cast<double>(contacts[contact].friction) *
            impulses[row];
        if (magnitude > cap && magnitude > 0.0) {
          const double scale = cap / magnitude;
          candidate0 *= scale;
          candidate1 *= scale;
        }
        const double delta0 =
            candidate0 - impulses[tangent0Row];
        const double delta1 =
            candidate1 - impulses[tangent1Row];
        maximumDelta =
            std::max(maximumDelta, std::fabs(delta0));
        maximumDelta =
            std::max(maximumDelta, std::fabs(delta1));
        applyImpulseDelta(tangent0Row, delta0);
        applyImpulseDelta(tangent1Row, delta1);
      }
      if (maximumDelta <= 1.0e-13)
        break;
    }

    result.outerIterations = outer + 1;
    result.maximumProjectedResidual =
        projectedMaterialResidual(contacts, rows, response,
                                  impulses);
    if (!std::isfinite(result.maximumProjectedResidual)) {
      result.finite = false;
      break;
    }
    if (result.maximumProjectedResidual <= strictTolerance) {
      result.candidateConverged = true;
      break;
    }
  }

  result.committedImpulse.assign(rowCount, 0.0);
  result.initialVelocity.resize(logicalBodyCount);
  result.committedVelocity.resize(logicalBodyCount);
  for (size_t logicalBody = 0; logicalBody < logicalBodyCount;
       ++logicalBody) {
    const size_t body = storageIndex(logicalBody);
    result.initialVelocity[logicalBody] =
        initialVelocities[body];
    result.committedVelocity[logicalBody] =
        initialVelocities[body];
  }
  if ((!result.candidateConverged || !result.finite) &&
      !runScalableCandidate)
    return result;

  std::vector<Vec6> scatteredVelocities = initialVelocities;
  for (size_t row = 0; row < rowCount; ++row) {
    addBroadMaterialRowImpulse(
        rows[row], impulses[row], bodies,
        scatteredVelocities);
  }

  std::vector<Vec6> oracleVelocities = initialVelocities;
  for (size_t contact = 0; contact < contacts.size();
       ++contact) {
    const size_t row = contact * 3u;
    const Vec3 impulse =
        contacts[contact].normal *
            static_cast<float>(impulses[row]) +
        contacts[contact].tangent[0] *
            static_cast<float>(impulses[row + 1u]) +
        contacts[contact].tangent[1] *
            static_cast<float>(impulses[row + 2u]);
    const auto applyContactImpulse =
        [&](size_t body, const Vec3 &arm,
            const Vec3 &bodyImpulse) {
          const ComponentProjectionBody &responseBody =
              bodies[body];
          oracleVelocities[body] +=
              Vec6(bodyImpulse *
                       responseBody.inverseMassResponse,
                   responseBody.inverseInertiaResponse *
                       arm.cross(bodyImpulse));
        };
    applyContactImpulse(contacts[contact].bodyA,
                        contacts[contact].armA, impulse);
    if (contacts[contact].bodyB != kStaticBody) {
      applyContactImpulse(contacts[contact].bodyB,
                          contacts[contact].armB, -impulse);
    }
  }
  result.bodyVelocityOracleDelta =
      maximumVelocityDelta(scatteredVelocities,
                           oracleVelocities);
  result.committedImpulse = impulses;
  result.committed =
      result.candidateConverged && result.finite;
  for (size_t logicalBody = 0; logicalBody < logicalBodyCount;
       ++logicalBody) {
    result.committedVelocity[logicalBody] =
        scatteredVelocities[storageIndex(logicalBody)];
  }

  Vec3 initialLinearMomentum;
  Vec3 finalLinearMomentum;
  Vec3 initialAngularMomentum;
  Vec3 finalAngularMomentum;
  double initialEnergy = 0.0;
  double finalEnergy = 0.0;
  for (size_t body = 0; body < logicalBodyCount; ++body) {
    const float mass = bodyMasses[body];
    const float inertia = mass / 6.0f;
    const Vec3 initialLinear = initialVelocities[body].linear();
    const Vec3 initialAngular = initialVelocities[body].angular();
    const Vec3 finalLinear = scatteredVelocities[body].linear();
    const Vec3 finalAngular = scatteredVelocities[body].angular();
    initialLinearMomentum += initialLinear * mass;
    finalLinearMomentum += finalLinear * mass;
    initialAngularMomentum +=
        bodyPositions[body].cross(initialLinear * mass) +
        initialAngular * inertia;
    finalAngularMomentum +=
        bodyPositions[body].cross(finalLinear * mass) +
        finalAngular * inertia;
    initialEnergy +=
        0.5 * static_cast<double>(mass) *
            static_cast<double>(initialLinear.length2()) +
        0.5 * static_cast<double>(inertia) *
            static_cast<double>(initialAngular.length2());
    finalEnergy +=
        0.5 * static_cast<double>(mass) *
            static_cast<double>(finalLinear.length2()) +
        0.5 * static_cast<double>(inertia) *
            static_cast<double>(finalAngular.length2());
  }

  Vec3 externalLinearImpulse;
  Vec3 externalAngularImpulse;
  for (size_t contact = 0; contact < contacts.size();
       ++contact) {
    const size_t row = contact * 3u;
    const double tangentMagnitude =
        std::sqrt(impulses[row + 1u] *
                      impulses[row + 1u] +
                  impulses[row + 2u] *
                      impulses[row + 2u]);
    const double cap =
        static_cast<double>(contacts[contact].friction) *
        impulses[row];
    result.maximumCoulombViolation =
        std::max(result.maximumCoulombViolation,
                 tangentMagnitude - cap);
    const double finalNormalVelocity =
        broadMaterialRowVelocity(contacts[contact].rows[0],
                                 scatteredVelocities);
    result.maximumNormalTargetError =
        std::max(result.maximumNormalTargetError,
                 static_cast<double>(
                     contacts[contact].normalTarget) -
                     finalNormalVelocity);
    if (contacts[contact].bodyB != kStaticBody)
      continue;
    const Vec3 impulse =
        contacts[contact].normal *
            static_cast<float>(impulses[row]) +
        contacts[contact].tangent[0] *
            static_cast<float>(impulses[row + 1u]) +
        contacts[contact].tangent[1] *
            static_cast<float>(impulses[row + 2u]);
    externalLinearImpulse += impulse;
    const Vec3 worldPoint =
        bodyPositions[contacts[contact].bodyA] +
        contacts[contact].armA;
    externalAngularImpulse += worldPoint.cross(impulse);
  }
  result.linearMomentumResidual =
      (finalLinearMomentum -
       initialLinearMomentum -
       externalLinearImpulse)
          .length();
  result.angularMomentumResidual =
      (finalAngularMomentum -
       initialAngularMomentum -
       externalAngularImpulse)
          .length();
  result.initialEnergy = initialEnergy;
  result.finalEnergy = finalEnergy;

  if (runScalableCandidate) {
    std::vector<size_t> normalRows(contacts.size());
    std::vector<size_t> tangentRows(contacts.size() * 2u);
    for (size_t contact = 0; contact < contacts.size();
         ++contact) {
      normalRows[contact] = contact * 3u;
      tangentRows[contact * 2u] = contact * 3u + 1u;
      tangentRows[contact * 2u + 1u] = contact * 3u + 2u;
    }

    std::vector<double> candidate(rowCount, 0.0);
    std::vector<double> other(rowCount, 0.0);
    std::vector<double> otherResponse;
    std::vector<double> normalQ(normalRows.size(), 0.0);
    std::vector<double> tangentQ(tangentRows.size(), 0.0);
    std::vector<double> normalImpulse(normalRows.size(), 0.0);
    std::vector<double> tangentImpulse(tangentRows.size(), 0.0);
    std::vector<double> tangentCaps(contacts.size(), 0.0);
    std::vector<std::vector<double>> mappedHistory;
    std::vector<std::vector<double>> mappedResponseHistory;
    std::vector<std::vector<double>> fixedResidualHistory;
    std::vector<double> candidateResponse(rowCount, 0.0);
    double outerPhysicalResidual =
        projectedMaterialResidualFromResponse(
            contacts, rows, candidate, candidateResponse,
            &result.scalableResidualEvaluations);
    result.scalableInitialResidual = outerPhysicalResidual;
    result.scalableProjectedResidual = outerPhysicalResidual;
    const auto projectCompleteCandidate =
        [&](std::vector<double> &values) {
          bool changed = false;
          for (size_t contact = 0;
               contact < contacts.size(); ++contact) {
            const size_t row = contact * 3u;
            const double normal = std::max(0.0, values[row]);
            changed = changed || normal != values[row];
            values[row] = normal;
            double tangent0 = values[row + 1u];
            double tangent1 = values[row + 2u];
            const double magnitude =
                std::sqrt(tangent0 * tangent0 +
                          tangent1 * tangent1);
            const double cap =
                static_cast<double>(
                    contacts[contact].friction) *
                values[row];
            if (magnitude > cap && magnitude > 0.0) {
              const double scale = cap / magnitude;
              tangent0 *= scale;
              tangent1 *= scale;
            }
            changed =
                changed || tangent0 != values[row + 1u] ||
                tangent1 != values[row + 2u];
            values[row + 1u] = tangent0;
            values[row + 2u] = tangent1;
          }
          return changed;
        };
    for (int outer = 0; outer < 256; ++outer) {
      const std::vector<double> outerStart = candidate;
      const std::vector<double> outerStartResponse =
          candidateResponse;
      const double blockTolerance = std::max(
          strictTolerance * 0.25,
          std::min(5.0e-2, 5.0e-2 * outerPhysicalResidual));
      {
        other = candidate;
      for (size_t contact = 0; contact < contacts.size();
           ++contact)
        other[contact * 3u] = 0.0;
      multiplyBroadAllRows(rows, bodies, other, otherResponse,
                           &result.scalableMatvecs);
      for (size_t contact = 0; contact < contacts.size();
           ++contact) {
        const size_t row = contact * 3u;
        normalQ[contact] =
            static_cast<double>(rows[row].outwardVelocity) +
            otherResponse[row];
        normalImpulse[contact] = candidate[row];
      }
      const BroadAcceleratedBlockStats normalStats =
          solveBroadMaterialAcceleratedBlock(
              rows, bodies, normalRows, normalQ,
              std::vector<double>(), false, blockTolerance,
              normalImpulse);
      result.scalableBlockIterations += normalStats.iterations;
      result.scalableMatvecs += normalStats.matvecs;
      if (!normalStats.finite || !normalStats.converged)
        break;
      for (size_t contact = 0; contact < contacts.size();
           ++contact)
        candidate[contact * 3u] = normalImpulse[contact];

      other = candidate;
      for (size_t contact = 0; contact < contacts.size();
           ++contact) {
        const size_t row = contact * 3u;
        other[row + 1u] = 0.0;
        other[row + 2u] = 0.0;
      }
      multiplyBroadAllRows(rows, bodies, other, otherResponse,
                           &result.scalableMatvecs);
      for (size_t contact = 0; contact < contacts.size();
           ++contact) {
        const size_t row = contact * 3u;
        tangentQ[contact * 2u] =
            static_cast<double>(
                rows[row + 1u].outwardVelocity) +
            otherResponse[row + 1u];
        tangentQ[contact * 2u + 1u] =
            static_cast<double>(
                rows[row + 2u].outwardVelocity) +
            otherResponse[row + 2u];
        tangentImpulse[contact * 2u] =
            candidate[row + 1u];
        tangentImpulse[contact * 2u + 1u] =
            candidate[row + 2u];
        tangentCaps[contact] =
            static_cast<double>(contacts[contact].friction) *
            candidate[row];
      }
      const BroadAcceleratedBlockStats tangentStats =
          solveBroadMaterialAcceleratedBlock(
              rows, bodies, tangentRows, tangentQ,
              tangentCaps, true, blockTolerance,
              tangentImpulse);
      result.scalableBlockIterations += tangentStats.iterations;
      result.scalableMatvecs += tangentStats.matvecs;
      if (!tangentStats.finite || !tangentStats.converged)
        break;
      for (size_t contact = 0; contact < contacts.size();
           ++contact) {
        const size_t row = contact * 3u;
        candidate[row + 1u] =
            tangentImpulse[contact * 2u];
        candidate[row + 2u] =
            tangentImpulse[contact * 2u + 1u];
      }
      }
      multiplyBroadAllRows(
          rows, bodies, candidate, candidateResponse,
          &result.scalableMatvecs);

      const std::vector<double> mappedCandidate = candidate;
      const std::vector<double> mappedResponse =
          candidateResponse;
      std::vector<double> fixedResidual(rowCount, 0.0);
      for (size_t row = 0; row < rowCount; ++row) {
        fixedResidual[row] =
            mappedCandidate[row] - outerStart[row];
      }
      mappedHistory.push_back(mappedCandidate);
      mappedResponseHistory.push_back(mappedResponse);
      fixedResidualHistory.push_back(fixedResidual);
      if (mappedHistory.size() > 7u) {
        mappedHistory.erase(mappedHistory.begin());
        mappedResponseHistory.erase(
            mappedResponseHistory.begin());
        fixedResidualHistory.erase(
            fixedResidualHistory.begin());
      }

      std::vector<double> bestCandidate = mappedCandidate;
      std::vector<double> bestResponse = mappedResponse;
      int bestChoice = 0;
      double bestResidual =
          projectedMaterialResidualFromResponse(
              contacts, rows, bestCandidate, bestResponse,
              &result.scalableResidualEvaluations);
      const double relaxationCandidates[8] = {
          1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0};
      for (size_t relaxationIndex = 0;
           relaxationIndex < 8u; ++relaxationIndex) {
        const double relaxation =
            relaxationCandidates[relaxationIndex];
        std::vector<double> trial(rowCount, 0.0);
        std::vector<double> trialResponse(rowCount, 0.0);
        for (size_t row = 0; row < rowCount; ++row) {
          trial[row] =
              outerStart[row] +
              relaxation *
                  (mappedCandidate[row] - outerStart[row]);
          trialResponse[row] =
              outerStartResponse[row] +
              relaxation *
                  (mappedResponse[row] -
                   outerStartResponse[row]);
        }
        const std::vector<double> unprojectedTrial = trial;
        if (projectCompleteCandidate(trial) &&
            !correctBroadMaterialResponseAfterProjection(
                rows, bodies, unprojectedTrial, trial,
                trialResponse, result)) {
          break;
        }
        const double trialResidual =
            projectedMaterialResidualFromResponse(
                contacts, rows, trial, trialResponse,
                &result.scalableResidualEvaluations);
        if (std::isfinite(trialResidual) &&
            trialResidual < bestResidual) {
          bestResidual = trialResidual;
          bestCandidate.swap(trial);
          bestResponse.swap(trialResponse);
          bestChoice = 1;
        }
      }

      if (mappedHistory.size() >= 2u) {
        const size_t memory =
            std::min<size_t>(5u, mappedHistory.size() - 1u);
        const size_t historyStart =
            mappedHistory.size() - memory - 1u;
        std::vector<std::vector<double>> deltaFixed(
            memory, std::vector<double>(rowCount, 0.0));
        std::vector<std::vector<double>> deltaMapped(
            memory, std::vector<double>(rowCount, 0.0));
        std::vector<std::vector<double>> deltaMappedResponse(
            memory, std::vector<double>(rowCount, 0.0));
        for (size_t column = 0; column < memory; ++column) {
          const size_t previous = historyStart + column;
          const size_t next = previous + 1u;
          for (size_t row = 0; row < rowCount; ++row) {
            deltaFixed[column][row] =
                fixedResidualHistory[next][row] -
                fixedResidualHistory[previous][row];
            deltaMapped[column][row] =
                mappedHistory[next][row] -
                mappedHistory[previous][row];
            deltaMappedResponse[column][row] =
                mappedResponseHistory[next][row] -
                mappedResponseHistory[previous][row];
          }
        }
        std::vector<double> normalMatrix(
            memory * memory, 0.0);
        std::vector<double> normalRhs(memory, 0.0);
        double trace = 0.0;
        for (size_t row = 0; row < memory; ++row) {
          for (size_t column = 0; column < memory;
               ++column) {
            double value = 0.0;
            for (size_t component = 0;
                 component < rowCount; ++component) {
              value +=
                  deltaFixed[row][component] *
                  deltaFixed[column][component];
            }
            normalMatrix[row * memory + column] = value;
          }
          trace += normalMatrix[row * memory + row];
          for (size_t component = 0;
               component < rowCount; ++component) {
            normalRhs[row] +=
                deltaFixed[row][component] *
                fixedResidualHistory.back()[component];
          }
        }
        const double regularization =
            std::max(1.0e-18, trace * 1.0e-12);
        for (size_t diagonal = 0; diagonal < memory;
             ++diagonal) {
          normalMatrix[diagonal * memory + diagonal] +=
              regularization;
        }
        std::vector<double> gamma;
        if (solveSmallDenseSystem(normalMatrix, normalRhs,
                                  gamma)) {
          std::vector<double> anderson = mappedCandidate;
          std::vector<double> andersonResponse =
              mappedResponse;
          for (size_t column = 0; column < memory; ++column) {
            for (size_t row = 0; row < rowCount; ++row) {
              anderson[row] -=
                  gamma[column] * deltaMapped[column][row];
              andersonResponse[row] -=
                  gamma[column] *
                  deltaMappedResponse[column][row];
            }
          }
          const std::vector<double> unprojectedAnderson =
              anderson;
          if (projectCompleteCandidate(anderson) &&
              !correctBroadMaterialResponseAfterProjection(
                  rows, bodies, unprojectedAnderson, anderson,
                  andersonResponse, result)) {
            break;
          }
          const double andersonResidual =
              projectedMaterialResidualFromResponse(
                  contacts, rows, anderson, andersonResponse,
                  &result.scalableResidualEvaluations);
          if (std::isfinite(andersonResidual) &&
              andersonResidual < bestResidual) {
            bestResidual = andersonResidual;
            bestCandidate.swap(anderson);
            bestResponse.swap(andersonResponse);
            bestChoice = 2;
          }
        }
      }
      if (bestChoice == 0)
        ++result.scalableMappedChoices;
      else if (bestChoice == 1)
        ++result.scalableRelaxationChoices;
      else
        ++result.scalableAndersonChoices;
      candidate.swap(bestCandidate);
      candidateResponse.swap(bestResponse);
      outerPhysicalResidual = bestResidual;
      result.scalableProjectedResidual = bestResidual;
      result.scalableOuterIterations = outer + 1;
      if (bestResidual <= strictTolerance) {
        result.scalableCandidateConverged = true;
        break;
      }
    }

    result.scalableLayerNormalResidual.assign(
        logicalBodyCount, 0.0);
    result.scalableLayerTangentResidual.assign(
        logicalBodyCount, 0.0);
    for (size_t contact = 0; contact < contacts.size();
         ++contact) {
      const size_t row = contact * 3u;
      const double normalProjected =
          std::max(0.0,
                   candidate[row] -
                       (static_cast<double>(
                            rows[row].outwardVelocity) +
                        candidateResponse[row]));
      const double normalResidual =
          std::fabs(normalProjected - candidate[row]);
      double tangent0 =
          candidate[row + 1u] -
          (static_cast<double>(rows[row + 1u].outwardVelocity) +
           candidateResponse[row + 1u]);
      double tangent1 =
          candidate[row + 2u] -
          (static_cast<double>(rows[row + 2u].outwardVelocity) +
           candidateResponse[row + 2u]);
      const double tangentMagnitude =
          std::sqrt(tangent0 * tangent0 + tangent1 * tangent1);
      const double tangentCap =
          static_cast<double>(contacts[contact].friction) *
          candidate[row];
      if (tangentMagnitude > tangentCap &&
          tangentMagnitude > 0.0) {
        const double scale = tangentCap / tangentMagnitude;
        tangent0 *= scale;
        tangent1 *= scale;
      }
      const double tangentResidual =
          std::max(std::fabs(tangent0 - candidate[row + 1u]),
                   std::fabs(tangent1 - candidate[row + 2u]));
      const size_t layer = contact / 4u;
      result.scalableLayerNormalResidual[layer] =
          std::max(result.scalableLayerNormalResidual[layer],
                   normalResidual);
      result.scalableLayerTangentResidual[layer] =
          std::max(result.scalableLayerTangentResidual[layer],
                   tangentResidual);
    }

    result.scalableVelocity = result.initialVelocity;
    if (result.scalableCandidateConverged) {
      std::vector<Vec6> scalableStorageVelocity =
          initialVelocities;
      for (size_t row = 0; row < rowCount; ++row) {
        addBroadMaterialRowImpulse(
            rows[row], candidate[row], bodies,
            scalableStorageVelocity);
      }
      for (size_t logicalBody = 0;
           logicalBody < logicalBodyCount; ++logicalBody) {
        result.scalableVelocity[logicalBody] =
            scalableStorageVelocity[storageIndex(logicalBody)];
      }
      if (result.committed) {
        result.scalableBodyVelocityDelta =
            maximumVelocityDelta(result.committedVelocity,
                                 result.scalableVelocity);
      }
    }

  }
  return result;
}

static double broadLaneInvariantDelta(
    const BroadMaterialLane &canonical,
    const BroadMaterialLane &other, float otherYaw) {
  if (canonical.committedVelocity.size() !=
      other.committedVelocity.size()) {
    return std::numeric_limits<double>::infinity();
  }
  const Vec3 canonicalDrive(1.0f, 0.0f, 0.0f);
  const Vec3 canonicalNormal(0.0f, 1.0f, 0.0f);
  const Vec3 canonicalLateral(0.0f, 0.0f, 1.0f);
  const Vec3 otherDrive =
      rotateAboutY(canonicalDrive, otherYaw);
  const Vec3 otherLateral =
      rotateAboutY(canonicalLateral, otherYaw);
  double maximum = 0.0;
  for (size_t body = 0;
       body < canonical.committedVelocity.size(); ++body) {
    const Vec6 &a = canonical.committedVelocity[body];
    const Vec6 &b = other.committedVelocity[body];
    maximum = std::max(
        maximum,
        std::fabs(static_cast<double>(
                      a.linear().dot(canonicalDrive)) -
                  static_cast<double>(
                      b.linear().dot(otherDrive))));
    maximum = std::max(
        maximum,
        std::fabs(static_cast<double>(
                      a.linear().dot(canonicalNormal)) -
                  static_cast<double>(
                      b.linear().dot(canonicalNormal))));
    maximum = std::max(
        maximum,
        std::fabs(static_cast<double>(
                      a.linear().dot(canonicalLateral)) -
                  static_cast<double>(
                      b.linear().dot(otherLateral))));
    maximum = std::max(
        maximum,
        std::fabs(static_cast<double>(
                      a.angular().dot(canonicalDrive)) -
                  static_cast<double>(
                      b.angular().dot(otherDrive))));
    maximum = std::max(
        maximum,
        std::fabs(static_cast<double>(
                      a.angular().dot(canonicalNormal)) -
                  static_cast<double>(
                      b.angular().dot(canonicalNormal))));
    maximum = std::max(
        maximum,
        std::fabs(static_cast<double>(
                      a.angular().dot(canonicalLateral)) -
                  static_cast<double>(
                      b.angular().dot(otherLateral))));
  }
  return maximum;
}

} // namespace

// Test 152: a five-body supported stack already forms a broad 20-contact
// material component.  A sixth impacting body merges four restitution
// contacts into the same 24-contact/72-row objective.  This is the standalone
// correctness authority required before another scalable PhysX backend may
// own components above the accepted 16-contact production cap.
bool test152_transientRestitutionBroadMaterialComponentAuthority() {
  printf("\n--- Test 152: Transient restitution broad material component authority ---\n");

  const float yaw = 0.37f;
  const BroadMaterialLane steady =
      runBroadMaterialLane(false, 0.0f, false, false, 512);
  const BroadMaterialLane canonical =
      runBroadMaterialLane(true, 0.0f, false, false, 512);
  const BroadMaterialLane reverse =
      runBroadMaterialLane(true, 0.0f, true, false, 512);
  const BroadMaterialLane reversedStorage =
      runBroadMaterialLane(true, 0.0f, false, true, 512);
  const BroadMaterialLane yawed =
      runBroadMaterialLane(true, yaw, false, false, 512);
  const BroadMaterialLane yawedReverseStorage =
      runBroadMaterialLane(true, yaw, true, true, 512);
  const BroadMaterialLane rejected =
      runBroadMaterialLane(true, 0.0f, false, false, 1);

  const double reverseDelta =
      broadLaneInvariantDelta(canonical, reverse, 0.0f);
  const double storageDelta =
      broadLaneInvariantDelta(canonical, reversedStorage,
                              0.0f);
  const double yawDelta =
      broadLaneInvariantDelta(canonical, yawed, yaw);
  const double yawReverseStorageDelta =
      broadLaneInvariantDelta(canonical,
                              yawedReverseStorage, yaw);
  const double rejectedVelocityDelta =
      maximumVelocityDelta(rejected.initialVelocity,
                           rejected.committedVelocity);
  double rejectedImpulseMagnitude = 0.0;
  for (double impulse : rejected.committedImpulse) {
    rejectedImpulseMagnitude =
        std::max(rejectedImpulseMagnitude,
                 std::fabs(impulse));
  }

  printf("  topology steady=(bodies=%zu contacts=%zu rows=%zu) "
         "merged=(bodies=%zu contacts=%zu rows=%zu restitution=%zu) "
         "diagRatio=%.9g\n",
         steady.bodyCount, steady.contactCount, steady.rowCount,
         canonical.bodyCount, canonical.contactCount,
         canonical.rowCount, canonical.restitutionContactCount,
         canonical.responseDiagonalRatio);
  printf("  canonical committed=%d outer=%d residual=%.9g/%.9g "
         "targetError=%.9g bodyOracle=%.9g coulomb=%.9g "
         "momentum=(%.9g,%.9g) energy=(%.9g,%.9g)\n",
         canonical.committed ? 1 : 0,
         canonical.outerIterations,
         canonical.maximumProjectedResidual,
         canonical.projectedResidualTolerance,
         canonical.maximumNormalTargetError,
         canonical.bodyVelocityOracleDelta,
         canonical.maximumCoulombViolation,
         canonical.linearMomentumResidual,
         canonical.angularMomentumResidual,
         canonical.initialEnergy, canonical.finalEnergy);
  printf("  invariance reverse=%.9g storage=%.9g yaw=%.9g "
         "yawReverseStorage=%.9g\n",
         reverseDelta, storageDelta, yawDelta,
         yawReverseStorageDelta);
  printf("  lanes reverse=(%d,%d,%.9g) "
         "storage=(%d,%d,%.9g) yaw=(%d,%d,%.9g) "
         "yawReverseStorage=(%d,%d,%.9g)\n",
         reverse.committed ? 1 : 0, reverse.outerIterations,
         reverse.maximumProjectedResidual,
         reversedStorage.committed ? 1 : 0,
         reversedStorage.outerIterations,
         reversedStorage.maximumProjectedResidual,
         yawed.committed ? 1 : 0, yawed.outerIterations,
         yawed.maximumProjectedResidual,
         yawedReverseStorage.committed ? 1 : 0,
         yawedReverseStorage.outerIterations,
         yawedReverseStorage.maximumProjectedResidual);
  printf("  rejected committed=%d residual=%.9g "
         "velocityDelta=%.9g committedImpulse=%.9g\n",
         rejected.committed ? 1 : 0,
         rejected.maximumProjectedResidual,
         rejectedVelocityDelta, rejectedImpulseMagnitude);

  CHECK(steady.committed && steady.contactCount == 20u &&
            steady.rowCount == 60u,
        "pre-impact broad support component did not solve");
  CHECK(canonical.contactCount == 24u &&
            canonical.rowCount == 72u &&
            canonical.restitutionContactCount == 4u,
        "transient impact did not create the required broad component");
  CHECK(canonical.contactCount > 16u,
        "authority fixture does not exceed the production owner cap");
  CHECK(canonical.responseDiagonalRatio >= 5.0,
        "authority fixture lacks broad response conditioning: %.9g",
        canonical.responseDiagonalRatio);

  for (const BroadMaterialLane *lane :
       {&canonical, &reverse, &reversedStorage, &yawed,
        &yawedReverseStorage}) {
    CHECK(lane->finite && lane->candidateConverged &&
              lane->committed,
          "broad material candidate did not converge and commit");
    CHECK(lane->maximumProjectedResidual <=
              lane->projectedResidualTolerance,
          "broad material owner left unscaled projected residual: %.9g/%.9g",
          lane->maximumProjectedResidual,
          lane->projectedResidualTolerance);
    CHECK(lane->maximumNormalTargetError <= 2.0e-5,
          "broad material owner missed a normal target: %.9g",
          lane->maximumNormalTargetError);
    CHECK(lane->bodyVelocityOracleDelta <= 2.0e-5,
          "row solve disagrees with independent body velocity oracle: %.9g",
          lane->bodyVelocityOracleDelta);
    CHECK(lane->maximumCoulombViolation <= 2.0e-8,
          "broad material owner violated a Coulomb disk: %.9g",
          lane->maximumCoulombViolation);
    CHECK(lane->linearMomentumResidual <= 5.0e-5 &&
              lane->angularMomentumResidual <= 5.0e-5,
          "broad material owner changed momentum: %.9g %.9g",
          lane->linearMomentumResidual,
          lane->angularMomentumResidual);
    CHECK(lane->finalEnergy <= lane->initialEnergy + 5.0e-5,
          "broad restitution component added kinetic energy: %.9g %.9g",
          lane->initialEnergy, lane->finalEnergy);
  }
  CHECK(reverseDelta <= 2.0e-5 &&
            storageDelta <= 2.0e-5 &&
            yawDelta <= 2.0e-5 &&
            yawReverseStorageDelta <= 2.0e-5,
        "broad material body velocity depends on row/body order or yaw");
  CHECK(!rejected.candidateConverged && !rejected.committed,
        "under-converged candidate incorrectly became an owner");
  CHECK(rejectedVelocityDelta == 0.0 &&
            rejectedImpulseMagnitude == 0.0,
        "numerical rejection leaked a partial body/impulse commit");
  CHECK(rejected.maximumProjectedResidual >
            rejected.projectedResidualTolerance,
        "limited candidate did not exercise numerical rejection");

  PASS("transient broad restitution components require strict residual, body-space oracle, and atomic ownership");
}

bool probe153_scalableBroadMaterialComponentAccelerated() {
  printf("\n--- Probe 153: Scalable broad material component accelerated blocks ---\n");
  const float yaw = 0.37f;
  const BroadMaterialLane canonical =
      runBroadMaterialLane(true, 0.0f, false, false, 512, true);
  const BroadMaterialLane reverse =
      runBroadMaterialLane(true, 0.0f, true, false, 512, true);
  const BroadMaterialLane reversedStorage =
      runBroadMaterialLane(true, 0.0f, false, true, 512, true);
  const BroadMaterialLane yawed =
      runBroadMaterialLane(true, yaw, false, false, 512, true);
  const BroadMaterialLane yawedReverseStorage =
      runBroadMaterialLane(true, yaw, true, true, 512, true);
  printf("  dense=(outer=%d normalSweeps=%d tangentSweeps=%d "
         "residual=%.9g/%.9g) "
         "accelerated=(converged=%d outer=%d blockIterations=%d "
         "matvec=%d residualEvals=%d projectionPasses=%d "
         "correctionRows=%d initialResidual=%.9g residual=%.9g "
         "velocityDelta=%.9g)\n",
         canonical.outerIterations, canonical.denseNormalSweeps,
         canonical.denseTangentSweeps,
         canonical.maximumProjectedResidual,
         canonical.projectedResidualTolerance,
         canonical.scalableCandidateConverged ? 1 : 0,
         canonical.scalableOuterIterations,
         canonical.scalableBlockIterations,
         canonical.scalableMatvecs,
         canonical.scalableResidualEvaluations,
         canonical.scalableProjectionResponseRebuilds,
         canonical.scalableProjectionCorrectionRows,
         canonical.scalableInitialResidual,
         canonical.scalableProjectedResidual,
         canonical.scalableBodyVelocityDelta);
  printf("  choices mapped=%d relaxation=%d anderson=%d\n",
         canonical.scalableMappedChoices,
         canonical.scalableRelaxationChoices,
         canonical.scalableAndersonChoices);
  for (const BroadMaterialLane *lane :
       {&canonical, &reverse, &reversedStorage, &yawed,
        &yawedReverseStorage}) {
    CHECK(lane->committed,
          "dense broad-component authority did not commit");
    CHECK(lane->scalableCandidateConverged,
          "matrix-free accelerated blocks did not reach the strict physical residual");
    CHECK(lane->scalableProjectedResidual <=
              lane->projectedResidualTolerance,
          "matrix-free accelerated residual exceeds authority: %.9g/%.9g",
          lane->scalableProjectedResidual,
          lane->projectedResidualTolerance);
    CHECK(lane->scalableBodyVelocityDelta <= 2.0e-5,
          "matrix-free accelerated body velocity disagrees with authority: %.9g",
          lane->scalableBodyVelocityDelta);
  }
  BroadMaterialLane scalableCanonical = canonical;
  BroadMaterialLane scalableReverse = reverse;
  BroadMaterialLane scalableStorage = reversedStorage;
  BroadMaterialLane scalableYawed = yawed;
  BroadMaterialLane scalableYawedReverseStorage =
      yawedReverseStorage;
  scalableCanonical.committedVelocity =
      canonical.scalableVelocity;
  scalableReverse.committedVelocity = reverse.scalableVelocity;
  scalableStorage.committedVelocity =
      reversedStorage.scalableVelocity;
  scalableYawed.committedVelocity = yawed.scalableVelocity;
  scalableYawedReverseStorage.committedVelocity =
      yawedReverseStorage.scalableVelocity;
  const double reverseDelta = broadLaneInvariantDelta(
      scalableCanonical, scalableReverse, 0.0f);
  const double storageDelta = broadLaneInvariantDelta(
      scalableCanonical, scalableStorage, 0.0f);
  const double yawDelta = broadLaneInvariantDelta(
      scalableCanonical, scalableYawed, yaw);
  const double yawReverseStorageDelta =
      broadLaneInvariantDelta(
          scalableCanonical, scalableYawedReverseStorage, yaw);
  printf("  scalable invariance reverse=%.9g storage=%.9g "
         "yaw=%.9g yawReverseStorage=%.9g\n",
         reverseDelta, storageDelta, yawDelta,
         yawReverseStorageDelta);
  CHECK(reverseDelta <= 2.0e-5 &&
            storageDelta <= 2.0e-5 &&
            yawDelta <= 2.0e-5 &&
            yawReverseStorageDelta <= 2.0e-5,
        "matrix-free accelerated result depends on row/body order or yaw");
  PASS("matrix-free accelerated blocks match the transient broad-component authority");
}

bool probe154_broadMaterialComponentScalingAuthority() {
  printf("\n--- Probe 154: Broad material component scaling authority ---\n");
  const BroadMaterialLane twelve =
      runBroadMaterialLane(true, 0.0f, false, false, 2048, true, 12u);
  const BroadMaterialLane twentyFour =
      runBroadMaterialLane(true, 0.0f, false, false, 2048, true, 24u);

  const auto printLane = [](const char *name,
                            const BroadMaterialLane &lane) {
    printf("  %s topology=(bodies=%zu contacts=%zu rows=%zu) "
           "dense=(converged=%d outer=%d residual=%.9g/%.9g "
           "bodyOracle=%.9g) "
           "accelerated=(converged=%d outer=%d blockIterations=%d "
           "matvec=%d residual=%.9g bodyDelta=%.9g)\n",
           name, lane.bodyCount, lane.contactCount, lane.rowCount,
           lane.candidateConverged ? 1 : 0, lane.outerIterations,
           lane.maximumProjectedResidual,
           lane.projectedResidualTolerance,
           lane.bodyVelocityOracleDelta,
           lane.scalableCandidateConverged ? 1 : 0,
           lane.scalableOuterIterations,
           lane.scalableBlockIterations, lane.scalableMatvecs,
           lane.scalableProjectedResidual,
           lane.scalableBodyVelocityDelta);
  };
  printLane("12-body", twelve);
  printLane("24-body", twentyFour);
  printf("  24-body residual by layer:");
  for (size_t layer = 0;
       layer < twentyFour.scalableLayerNormalResidual.size();
       ++layer) {
    printf(" %zu=(%.3g,%.3g)", layer,
           twentyFour.scalableLayerNormalResidual[layer],
           twentyFour.scalableLayerTangentResidual[layer]);
  }
  printf("\n");

  CHECK(twelve.bodyCount == 12u && twelve.contactCount == 48u &&
            twelve.rowCount == 144u &&
            twelve.restitutionContactCount == 4u,
        "12-body authority topology changed");
  CHECK(twentyFour.bodyCount == 24u &&
            twentyFour.contactCount == 96u &&
            twentyFour.rowCount == 288u &&
            twentyFour.restitutionContactCount == 4u,
        "24-body authority topology changed");

  for (const BroadMaterialLane *lane : {&twelve, &twentyFour}) {
    CHECK(lane->finite && lane->candidateConverged &&
              lane->committed,
          "dense scaling authority did not converge and commit");
    CHECK(lane->maximumProjectedResidual <=
              lane->projectedResidualTolerance,
          "dense scaling authority left physical residual: %.9g/%.9g",
          lane->maximumProjectedResidual,
          lane->projectedResidualTolerance);
    CHECK(lane->maximumNormalTargetError <= 5.0e-5,
          "dense scaling authority missed a normal target: %.9g",
          lane->maximumNormalTargetError);
    CHECK(lane->bodyVelocityOracleDelta <= 5.0e-5,
          "dense scaling authority disagrees with body oracle: %.9g",
          lane->bodyVelocityOracleDelta);
    CHECK(lane->maximumCoulombViolation <= 2.0e-8,
          "dense scaling authority violated a Coulomb disk: %.9g",
          lane->maximumCoulombViolation);
    CHECK(lane->linearMomentumResidual <= 2.0e-4 &&
              lane->angularMomentumResidual <= 2.0e-4,
          "dense scaling authority changed momentum: %.9g %.9g",
          lane->linearMomentumResidual,
          lane->angularMomentumResidual);
    CHECK(lane->finalEnergy <= lane->initialEnergy + 2.0e-4,
          "dense scaling authority added energy: %.9g %.9g",
          lane->initialEnergy, lane->finalEnergy);
  }
  CHECK(twelve.scalableCandidateConverged &&
            twelve.scalableProjectedResidual <=
                twelve.projectedResidualTolerance &&
            twelve.scalableBodyVelocityDelta <= 5.0e-5,
        "accelerated baseline lost the 12-body scaling authority");
  CHECK(!twentyFour.scalableCandidateConverged &&
            twentyFour.scalableProjectedResidual >
                twentyFour.projectedResidualTolerance &&
            maximumVelocityDelta(twentyFour.initialVelocity,
                                 twentyFour.scalableVelocity) == 0.0,
        "24-body failure-first boundary no longer rejects atomically");

  PASS("12-/24-body fixtures preserve the strict material authority and expose candidate scaling");
}
