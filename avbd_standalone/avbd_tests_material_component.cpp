#include "avbd_component_unilateral_projection.h"
#include "avbd_material_graph_multilevel.h"
#include "avbd_material_interface_wrench.h"
#include "avbd_material_spatial_transfer.h"
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

bool probe155_materialInterfaceWrenchAuthority() {
  printf("\n--- Probe 155: Material interface-wrench restriction/prolongation authority ---\n");

  const auto makePoints =
      [](float yaw, bool reverseOrder) {
        const Vec3 normal(0.0f, 1.0f, 0.0f);
        const Vec3 tangent0 =
            rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), yaw);
        const Vec3 tangent1 =
            rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), yaw);
        const Vec3 localPoints[4] = {
            Vec3(-0.5f, 0.0f, -0.5f),
            Vec3(-0.5f, 0.0f, 0.5f),
            Vec3(0.5f, 0.0f, -0.5f),
            Vec3(0.5f, 0.0f, 0.5f)};
        std::vector<MaterialInterfacePoint> points;
        for (size_t point = 0; point < 4u; ++point) {
          MaterialInterfacePoint value;
          value.worldPoint = rotateAboutY(localPoints[point], yaw);
          value.normal = normal;
          value.tangent0 = tangent0;
          value.tangent1 = tangent1;
          value.stableKey = 1000u + static_cast<uint64_t>(point);
          points.push_back(value);
        }
        if (reverseOrder)
          std::reverse(points.begin(), points.end());
        return points;
      };
  const auto maximumVectorDelta =
      [](const std::vector<double> &a,
         const std::vector<double> &b) {
        if (a.size() != b.size())
          return std::numeric_limits<double>::infinity();
        double maximum = 0.0;
        for (size_t index = 0; index < a.size(); ++index) {
          maximum =
              std::max(maximum, std::fabs(a[index] - b[index]));
        }
        return maximum;
      };
  const auto maximumWrenchDelta =
      [](const MaterialSpatialWrench &a,
         const MaterialSpatialWrench &b) {
        double maximum = 0.0;
        for (size_t component = 0; component < 6u; ++component) {
          maximum =
              std::max(maximum,
                       std::fabs(a[component] - b[component]));
        }
        return maximum;
      };
  const auto squaredNorm =
      [](const std::vector<double> &values) {
        double norm = 0.0;
        for (double value : values)
          norm += value * value;
        return norm;
      };
  const auto rotateWrench =
      [](const MaterialSpatialWrench &wrench, float yaw) {
        const Vec3 force = rotateAboutY(
            Vec3(static_cast<float>(wrench[0]),
                 static_cast<float>(wrench[1]),
                 static_cast<float>(wrench[2])),
            yaw);
        const Vec3 moment = rotateAboutY(
            Vec3(static_cast<float>(wrench[3]),
                 static_cast<float>(wrench[4]),
                 static_cast<float>(wrench[5])),
            yaw);
        MaterialSpatialWrench result = {
            force.x, force.y, force.z,
            moment.x, moment.y, moment.z};
        return result;
      };
  const auto applyPointImpulses =
      [](const MaterialInterfaceWrenchMap &map,
         const std::vector<double> &impulses,
         const Vec3 &bodyPosition, float inverseMass,
         const Mat33 &inverseInertia, float sign) {
        Vec3 linear;
        Vec3 angular;
        for (size_t point = 0; point < map.points.size(); ++point) {
          const size_t row = point * 3u;
          const Vec3 impulse =
              map.points[point].normal *
                  static_cast<float>(impulses[row]) +
              map.points[point].tangent0 *
                  static_cast<float>(impulses[row + 1u]) +
              map.points[point].tangent1 *
                  static_cast<float>(impulses[row + 2u]);
          const Vec3 signedImpulse = impulse * sign;
          linear += signedImpulse * inverseMass;
          angular +=
              inverseInertia *
              ((map.points[point].worldPoint - bodyPosition)
                   .cross(signedImpulse));
        }
        return Vec6(linear, angular);
      };
  const auto applyWrench =
      [](const MaterialSpatialWrench &wrench,
         const Vec3 &bodyPosition, float inverseMass,
         const Mat33 &inverseInertia, float sign) {
        const Vec3 force(
            static_cast<float>(wrench[0]),
            static_cast<float>(wrench[1]),
            static_cast<float>(wrench[2]));
        const Vec3 moment(
            static_cast<float>(wrench[3]),
            static_cast<float>(wrench[4]),
            static_cast<float>(wrench[5]));
        const Vec3 signedForce = force * sign;
        const Vec3 signedMoment = moment * sign;
        return Vec6(
            signedForce * inverseMass,
            inverseInertia *
                (signedMoment -
                 bodyPosition.cross(signedForce)));
      };

  const MaterialInterfaceWrenchMap canonical =
      buildMaterialInterfaceWrenchMap(makePoints(0.0f, false));
  const MaterialInterfaceWrenchMap reverse =
      buildMaterialInterfaceWrenchMap(makePoints(0.0f, true));
  CHECK(canonical.finite && reverse.finite &&
            canonical.rank == 6 && reverse.rank == 6,
        "four-point interface did not produce a full-rank wrench map");

  double rangeIdentityError = 0.0;
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      const double expected = row == column ? 1.0 : 0.0;
      rangeIdentityError =
          std::max(rangeIdentityError,
                   std::fabs(canonical.rangeProjector[
                                 row * 6 + column] -
                             expected));
    }
  }
  const double orderRestrictionDelta =
      maximumVectorDelta(canonical.restriction,
                         reverse.restriction);
  const double orderProlongationDelta =
      maximumVectorDelta(canonical.prolongation,
                         reverse.prolongation);

  std::vector<double> feasibleImpulse(
      canonical.points.size() * 3u, 0.0);
  for (size_t point = 0; point < canonical.points.size();
       ++point) {
    feasibleImpulse[point * 3u] = 2.0;
    feasibleImpulse[point * 3u + 1u] = 0.25;
    feasibleImpulse[point * 3u + 2u] = -0.1;
  }
  MaterialSpatialWrench feasibleWrench{};
  CHECK(restrictMaterialPointImpulses(
            canonical, feasibleImpulse, feasibleWrench),
        "failed to restrict a feasible point impulse");
  std::vector<double> minimumImpulse;
  CHECK(prolongMaterialInterfaceWrench(
            canonical, feasibleWrench, minimumImpulse),
        "failed to prolong a full-rank interface wrench");
  MaterialSpatialWrench reconstructedWrench{};
  CHECK(restrictMaterialPointImpulses(
            canonical, minimumImpulse, reconstructedWrench),
        "failed to reconstruct the prolonged wrench");
  const double wrenchRoundTripError =
      maximumWrenchDelta(feasibleWrench, reconstructedWrench);

  std::vector<double> arbitrary(feasibleImpulse.size(), 0.0);
  for (size_t coordinate = 0; coordinate < arbitrary.size();
       ++coordinate) {
    arbitrary[coordinate] =
        std::sin(0.73 * static_cast<double>(coordinate + 1u));
  }
  MaterialSpatialWrench arbitraryWrench{};
  CHECK(restrictMaterialPointImpulses(
            canonical, arbitrary, arbitraryWrench),
        "failed to restrict the nullspace witness");
  std::vector<double> arbitraryMinimum;
  CHECK(prolongMaterialInterfaceWrench(
            canonical, arbitraryWrench, arbitraryMinimum),
        "failed to remove the point-contact nullspace");
  std::vector<double> nullspace(arbitrary.size(), 0.0);
  double minimumNullDot = 0.0;
  for (size_t coordinate = 0; coordinate < arbitrary.size();
       ++coordinate) {
    nullspace[coordinate] =
        arbitrary[coordinate] - arbitraryMinimum[coordinate];
    minimumNullDot +=
        arbitraryMinimum[coordinate] * nullspace[coordinate];
  }
  MaterialSpatialWrench nullWrench{};
  CHECK(restrictMaterialPointImpulses(
            canonical, nullspace, nullWrench),
        "failed to restrict the removed nullspace");
  MaterialSpatialWrench zeroWrench{};
  const double nullWrenchError =
      maximumWrenchDelta(nullWrench, zeroWrench);

  const Vec3 bodyPositionA(0.0f, 0.5f, 0.0f);
  const Vec3 bodyPositionB(0.0f, -0.5f, 0.0f);
  const Mat33 inverseInertiaA = Mat33::diag(0.4f, 0.3f, 0.2f);
  const Mat33 inverseInertiaB = Mat33::diag(0.7f, 0.5f, 0.25f);
  const Vec6 pointDeltaA = applyPointImpulses(
      canonical, minimumImpulse, bodyPositionA, 0.25f,
      inverseInertiaA, 1.0f);
  const Vec6 pointDeltaB = applyPointImpulses(
      canonical, minimumImpulse, bodyPositionB, 0.5f,
      inverseInertiaB, -1.0f);
  const Vec6 wrenchDeltaA = applyWrench(
      reconstructedWrench, bodyPositionA, 0.25f,
      inverseInertiaA, 1.0f);
  const Vec6 wrenchDeltaB = applyWrench(
      reconstructedWrench, bodyPositionB, 0.5f,
      inverseInertiaB, -1.0f);
  double bodyResponseError = 0.0;
  for (int component = 0; component < 6; ++component) {
    bodyResponseError =
        std::max(bodyResponseError,
                 std::fabs(static_cast<double>(
                               pointDeltaA[component]) -
                           static_cast<double>(
                               wrenchDeltaA[component])));
    bodyResponseError =
        std::max(bodyResponseError,
                 std::fabs(static_cast<double>(
                               pointDeltaB[component]) -
                           static_cast<double>(
                               wrenchDeltaB[component])));
  }

  const float yaw = 0.41f;
  const MaterialInterfaceWrenchMap yawed =
      buildMaterialInterfaceWrenchMap(makePoints(yaw, false));
  MaterialSpatialWrench yawedWrench{};
  CHECK(yawed.finite && yawed.rank == 6 &&
            restrictMaterialPointImpulses(
                yawed, feasibleImpulse, yawedWrench),
        "yawed interface map is not full-rank");
  const double yawWrenchError = maximumWrenchDelta(
      rotateWrench(feasibleWrench, yaw), yawedWrench);
  std::vector<double> yawedMinimumImpulse;
  CHECK(prolongMaterialInterfaceWrench(
            yawed, yawedWrench, yawedMinimumImpulse),
        "failed to prolong the yawed interface wrench");
  const double yawImpulseError =
      maximumVectorDelta(minimumImpulse, yawedMinimumImpulse);

  MaterialInterfacePoint degeneratePoint;
  degeneratePoint.worldPoint = Vec3(0.25f, 0.0f, 0.5f);
  degeneratePoint.normal = Vec3(0.0f, 1.0f, 0.0f);
  degeneratePoint.tangent0 = Vec3(1.0f, 0.0f, 0.0f);
  degeneratePoint.tangent1 = Vec3(0.0f, 0.0f, 1.0f);
  degeneratePoint.stableKey = 42u;
  const MaterialInterfaceWrenchMap degenerate =
      buildMaterialInterfaceWrenchMap({degeneratePoint});
  const std::vector<double> degenerateImpulse = {1.0, -0.25, 0.5};
  MaterialSpatialWrench degenerateWrench{};
  std::vector<double> degenerateRoundTrip;
  CHECK(degenerate.finite && degenerate.rank == 3 &&
            restrictMaterialPointImpulses(
                degenerate, degenerateImpulse,
                degenerateWrench) &&
            prolongMaterialInterfaceWrench(
                degenerate, degenerateWrench,
                degenerateRoundTrip),
        "rank-deficient interface map did not preserve its range");
  const double degenerateImpulseError =
      maximumVectorDelta(degenerateImpulse,
                         degenerateRoundTrip);
  MaterialSpatialWrench unavailableWrench = {
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  const MaterialSpatialWrench projectedUnavailable =
      projectMaterialInterfaceWrenchToRange(
          degenerate, unavailableWrench);
  const MaterialSpatialWrench projectedTwice =
      projectMaterialInterfaceWrenchToRange(
          degenerate, projectedUnavailable);
  const double degenerateIdempotenceError =
      maximumWrenchDelta(projectedUnavailable, projectedTwice);
  const double unavailableProjectionGap =
      maximumWrenchDelta(unavailableWrench,
                         projectedUnavailable);

  std::vector<double> projectedCorrection = minimumImpulse;
  projectedCorrection[0] = -1.0;
  projectedCorrection[1] = 10.0;
  projectedCorrection[2] = -7.0;
  const std::vector<double> friction(
      canonical.points.size(), 0.8);
  CHECK(projectMaterialPointImpulses(
            canonical, friction, projectedCorrection),
        "failed to project a coarse point correction");
  double feasibilityViolation = 0.0;
  for (size_t point = 0; point < canonical.points.size();
       ++point) {
    const size_t row = point * 3u;
    const double magnitude =
        std::sqrt(projectedCorrection[row + 1u] *
                      projectedCorrection[row + 1u] +
                  projectedCorrection[row + 2u] *
                      projectedCorrection[row + 2u]);
    feasibilityViolation =
        std::max(feasibilityViolation,
                 -projectedCorrection[row]);
    feasibilityViolation =
        std::max(feasibilityViolation,
                 magnitude -
                     friction[point] * projectedCorrection[row]);
  }
  MaterialSpatialWrench projectedCorrectionWrench{};
  CHECK(restrictMaterialPointImpulses(
            canonical, projectedCorrection,
            projectedCorrectionWrench),
        "projected correction did not produce a finite wrench");
  const Vec6 projectedPointDelta = applyPointImpulses(
      canonical, projectedCorrection, bodyPositionA, 0.25f,
      inverseInertiaA, 1.0f);
  const Vec6 projectedWrenchDelta = applyWrench(
      projectedCorrectionWrench, bodyPositionA, 0.25f,
      inverseInertiaA, 1.0f);
  double projectedResponseError = 0.0;
  for (int component = 0; component < 6; ++component) {
    projectedResponseError =
        std::max(projectedResponseError,
                 std::fabs(static_cast<double>(
                               projectedPointDelta[component]) -
                           static_cast<double>(
                               projectedWrenchDelta[component])));
  }

  uint32_t randomState = 0x155A11CEu;
  const auto randomUnit =
      [&randomState]() {
        randomState =
            randomState * 1664525u + 1013904223u;
        return static_cast<double>(randomState >> 8u) /
               16777216.0;
      };
  const auto randomSigned =
      [&randomUnit]() {
        return randomUnit() * 2.0 - 1.0;
      };
  double randomOrderError = 0.0;
  double randomRoundTripError = 0.0;
  double randomNullWrenchError = 0.0;
  double randomMinimumOrthogonality = 0.0;
  double randomBodyResponseError = 0.0;
  double randomFeasibilityViolation = 0.0;
  double randomProjectedResponseError = 0.0;
  for (uint32_t sample = 0; sample < 64u; ++sample) {
    const size_t pointCount =
        4u + static_cast<size_t>(sample % 5u);
    const float randomYaw =
        static_cast<float>(randomSigned() * 3.0);
    const Vec3 randomNormal(0.0f, 1.0f, 0.0f);
    const Vec3 randomTangent0 =
        rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), randomYaw);
    const Vec3 randomTangent1 =
        rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), randomYaw);
    const Vec3 translation(
        static_cast<float>(randomSigned()),
        static_cast<float>(0.25 * randomSigned()),
        static_cast<float>(randomSigned()));
    std::vector<MaterialInterfacePoint> randomPoints;
    randomPoints.reserve(pointCount);
    for (size_t point = 0; point < pointCount; ++point) {
      const double angle =
          6.28318530717958647692 *
              static_cast<double>(point) /
              static_cast<double>(pointCount) +
          0.08 * randomSigned();
      const float radius =
          static_cast<float>(0.35 + 0.45 * randomUnit());
      const Vec3 local(
          radius * static_cast<float>(std::cos(angle)),
          static_cast<float>(0.04 * randomSigned()),
          radius * static_cast<float>(std::sin(angle)));
      MaterialInterfacePoint value;
      value.worldPoint =
          translation + rotateAboutY(local, randomYaw);
      value.normal = randomNormal;
      value.tangent0 = randomTangent0;
      value.tangent1 = randomTangent1;
      value.stableKey =
          10000u + static_cast<uint64_t>(sample) * 16u +
          static_cast<uint64_t>(point);
      randomPoints.push_back(value);
    }
    std::vector<MaterialInterfacePoint> randomReordered =
        randomPoints;
    std::reverse(randomReordered.begin(),
                 randomReordered.end());
    const MaterialInterfaceWrenchMap randomMap =
        buildMaterialInterfaceWrenchMap(randomPoints);
    const MaterialInterfaceWrenchMap randomOrderMap =
        buildMaterialInterfaceWrenchMap(randomReordered);
    CHECK(randomMap.finite && randomMap.rank == 6 &&
              randomOrderMap.finite &&
              randomOrderMap.rank == 6,
          "random interface %u did not produce a full-rank map",
          sample);
    randomOrderError =
        std::max(randomOrderError,
                 maximumVectorDelta(
                     randomMap.restriction,
                     randomOrderMap.restriction));
    randomOrderError =
        std::max(randomOrderError,
                 maximumVectorDelta(
                     randomMap.prolongation,
                     randomOrderMap.prolongation));

    std::vector<double> randomImpulse(
        pointCount * 3u, 0.0);
    for (double &value : randomImpulse)
      value = 2.0 * randomSigned();
    MaterialSpatialWrench randomWrench{};
    std::vector<double> randomMinimum;
    MaterialSpatialWrench randomReconstructed{};
    CHECK(restrictMaterialPointImpulses(
              randomMap, randomImpulse, randomWrench) &&
              prolongMaterialInterfaceWrench(
                  randomMap, randomWrench, randomMinimum) &&
              restrictMaterialPointImpulses(
                  randomMap, randomMinimum,
                  randomReconstructed),
          "random interface %u failed its wrench round trip",
          sample);
    randomRoundTripError =
        std::max(randomRoundTripError,
                 maximumWrenchDelta(
                     randomWrench, randomReconstructed));

    std::vector<double> randomNull(
        randomImpulse.size(), 0.0);
    double randomDot = 0.0;
    for (size_t coordinate = 0;
         coordinate < randomImpulse.size(); ++coordinate) {
      randomNull[coordinate] =
          randomImpulse[coordinate] -
          randomMinimum[coordinate];
      randomDot += randomMinimum[coordinate] *
                   randomNull[coordinate];
    }
    MaterialSpatialWrench randomNullWrench{};
    CHECK(restrictMaterialPointImpulses(
              randomMap, randomNull, randomNullWrench),
          "random interface %u failed to restrict nullspace",
          sample);
    randomNullWrenchError =
        std::max(randomNullWrenchError,
                 maximumWrenchDelta(
                     randomNullWrench, zeroWrench));
    randomMinimumOrthogonality =
        std::max(randomMinimumOrthogonality,
                 std::fabs(randomDot));

    const Vec3 randomBodyPosition =
        translation +
        Vec3(static_cast<float>(0.5 * randomSigned()),
             static_cast<float>(0.5 + 0.5 * randomUnit()),
             static_cast<float>(0.5 * randomSigned()));
    const float randomInverseMass =
        static_cast<float>(0.1 + randomUnit());
    const Mat33 randomInverseInertia = Mat33::diag(
        static_cast<float>(0.1 + randomUnit()),
        static_cast<float>(0.1 + randomUnit()),
        static_cast<float>(0.1 + randomUnit()));
    const Vec6 randomPointDelta = applyPointImpulses(
        randomMap, randomMinimum, randomBodyPosition,
        randomInverseMass, randomInverseInertia, 1.0f);
    const Vec6 randomWrenchDelta = applyWrench(
        randomReconstructed, randomBodyPosition,
        randomInverseMass, randomInverseInertia, 1.0f);
    for (int component = 0; component < 6; ++component) {
      randomBodyResponseError =
          std::max(
              randomBodyResponseError,
              std::fabs(
                  static_cast<double>(
                      randomPointDelta[component]) -
                  static_cast<double>(
                      randomWrenchDelta[component])));
    }

    std::vector<double> randomFriction(pointCount, 0.0);
    std::vector<double> randomProjected(pointCount * 3u, 0.0);
    for (size_t point = 0; point < pointCount; ++point) {
      randomFriction[point] = 0.1 + randomUnit();
      randomProjected[point * 3u] =
          3.0 * randomSigned();
      randomProjected[point * 3u + 1u] =
          5.0 * randomSigned();
      randomProjected[point * 3u + 2u] =
          5.0 * randomSigned();
    }
    CHECK(projectMaterialPointImpulses(
              randomMap, randomFriction, randomProjected),
          "random interface %u failed material projection",
          sample);
    for (size_t point = 0; point < pointCount; ++point) {
      const size_t row = point * 3u;
      const double tangentMagnitude =
          std::sqrt(
              randomProjected[row + 1u] *
                  randomProjected[row + 1u] +
              randomProjected[row + 2u] *
                  randomProjected[row + 2u]);
      randomFeasibilityViolation =
          std::max(randomFeasibilityViolation,
                   -randomProjected[row]);
      randomFeasibilityViolation =
          std::max(
              randomFeasibilityViolation,
              tangentMagnitude -
                  randomFriction[point] *
                      randomProjected[row]);
    }
    MaterialSpatialWrench randomProjectedWrench{};
    CHECK(restrictMaterialPointImpulses(
              randomMap, randomProjected,
              randomProjectedWrench),
          "random interface %u projected wrench is not finite",
          sample);
    const Vec6 randomProjectedPointDelta =
        applyPointImpulses(
            randomMap, randomProjected, randomBodyPosition,
            randomInverseMass, randomInverseInertia, -1.0f);
    const Vec6 randomProjectedWrenchDelta =
        applyWrench(
            randomProjectedWrench, randomBodyPosition,
            randomInverseMass, randomInverseInertia, -1.0f);
    for (int component = 0; component < 6; ++component) {
      randomProjectedResponseError =
          std::max(
              randomProjectedResponseError,
              std::fabs(
                  static_cast<double>(
                      randomProjectedPointDelta[component]) -
                  static_cast<double>(
                      randomProjectedWrenchDelta[component])));
    }
  }

  printf("  full rank=%d RP=%.9g wrench=%.9g order=(%.9g,%.9g) "
         "yaw=(%.9g,%.9g)\n",
         canonical.rank, rangeIdentityError,
         wrenchRoundTripError, orderRestrictionDelta,
         orderProlongationDelta, yawWrenchError,
         yawImpulseError);
  printf("  null wrench=%.9g orthogonality=%.9g norms=(%.9g,%.9g) "
         "body=%.9g\n",
         nullWrenchError, minimumNullDot,
         squaredNorm(arbitraryMinimum), squaredNorm(arbitrary),
         bodyResponseError);
  printf("  degenerate rank=%d impulse=%.9g idempotence=%.9g "
         "rangeGap=%.9g projected=(feasibility=%.9g body=%.9g)\n",
         degenerate.rank, degenerateImpulseError,
         degenerateIdempotenceError, unavailableProjectionGap,
         feasibilityViolation, projectedResponseError);
  printf("  random64 order=%.9g roundtrip=%.9g null=%.9g "
         "orthogonality=%.9g body=%.9g feasibility=%.9g "
         "projectedBody=%.9g\n",
         randomOrderError, randomRoundTripError,
         randomNullWrenchError, randomMinimumOrthogonality,
         randomBodyResponseError, randomFeasibilityViolation,
         randomProjectedResponseError);

  CHECK(rangeIdentityError <= 2.0e-12 &&
            wrenchRoundTripError <= 2.0e-12,
        "full-rank R*P authority failed: %.9g %.9g",
        rangeIdentityError, wrenchRoundTripError);
  CHECK(orderRestrictionDelta == 0.0 &&
            orderProlongationDelta == 0.0,
        "wrench map depends on point storage order");
  CHECK(nullWrenchError <= 2.0e-12 &&
            std::fabs(minimumNullDot) <= 2.0e-12 &&
            squaredNorm(arbitraryMinimum) <=
                squaredNorm(arbitrary) + 2.0e-12,
        "minimum-norm prolongation retained point nullspace");
  CHECK(bodyResponseError <= 2.0e-6,
        "interface wrench does not reproduce direct body response: %.9g",
        bodyResponseError);
  CHECK(yawWrenchError <= 2.0e-6 &&
            yawImpulseError <= 2.0e-6,
        "wrench map depends on world yaw: %.9g %.9g",
        yawWrenchError, yawImpulseError);
  CHECK(degenerateImpulseError <= 2.0e-12 &&
            degenerateIdempotenceError <= 2.0e-12 &&
            unavailableProjectionGap >= 1.0e-2,
        "rank-deficient range projector is not authoritative");
  CHECK(feasibilityViolation <= 2.0e-12 &&
            projectedResponseError <= 2.0e-6,
        "projected coarse correction is infeasible or changes body response");
  CHECK(randomOrderError == 0.0 &&
            randomRoundTripError <= 2.0e-10 &&
            randomNullWrenchError <= 2.0e-10 &&
            randomMinimumOrthogonality <= 2.0e-10,
        "random interface algebra authority failed: %.9g %.9g %.9g %.9g",
        randomOrderError, randomRoundTripError,
        randomNullWrenchError, randomMinimumOrthogonality);
  CHECK(randomBodyResponseError <= 2.0e-5 &&
            randomFeasibilityViolation <= 2.0e-12 &&
            randomProjectedResponseError <= 2.0e-5,
        "random interface physical authority failed: %.9g %.9g %.9g",
        randomBodyResponseError, randomFeasibilityViolation,
        randomProjectedResponseError);

  PASS("interface-wrench restriction/prolongation removes point nullspace and preserves physical body response");
}

bool probe156_materialMultilevelGraphAuthority() {
  printf("\n--- Probe 156: Material multilevel graph authority ---\n");

  struct GraphEdge {
    int bodyA = -1;
    int bodyB = -1;
    uint64_t stableKey = 0;
  };
  struct SolveResult {
    int cycles = 0;
    double relativeResidual = 0.0;
    double solutionError = 0.0;
    bool converged = false;
  };
  const auto makeBodyOperator =
      [](size_t bodyCount, const std::vector<GraphEdge> &edges) {
        std::vector<double> matrix(
            bodyCount * bodyCount, 0.0);
        for (size_t edge = 0; edge < edges.size(); ++edge) {
          const double weight =
              0.5 + 0.05 * static_cast<double>(edge % 7u);
          if (edges[edge].bodyA >= 0) {
            const size_t bodyA =
                static_cast<size_t>(edges[edge].bodyA);
            matrix[bodyA * bodyCount + bodyA] += weight;
          }
          if (edges[edge].bodyB >= 0) {
            const size_t bodyB =
                static_cast<size_t>(edges[edge].bodyB);
            matrix[bodyB * bodyCount + bodyB] += weight;
            if (edges[edge].bodyA >= 0) {
              const size_t bodyA =
                  static_cast<size_t>(edges[edge].bodyA);
              matrix[bodyA * bodyCount + bodyB] -= weight;
              matrix[bodyB * bodyCount + bodyA] -= weight;
            }
          }
        }
        return matrix;
      };
  const auto bodyKeys =
      [](size_t bodyCount, uint64_t base) {
        std::vector<uint64_t> keys(bodyCount, 0u);
        for (size_t body = 0; body < bodyCount; ++body)
          keys[body] = base + body;
        return keys;
      };
  const auto makeChain =
      [](size_t count, uint64_t keyBase) {
        std::vector<GraphEdge> edges;
        edges.reserve(count);
        for (size_t body = 0; body < count; ++body) {
          GraphEdge edge;
          edge.bodyA = static_cast<int>(body);
          edge.bodyB =
              body == 0u ? -1 : static_cast<int>(body - 1u);
          edge.stableKey = keyBase + body;
          edges.push_back(edge);
        }
        return edges;
      };
  const auto multiply =
      [](const std::vector<double> &matrix,
         const std::vector<double> &input) {
        std::vector<double> output(input.size(), 0.0);
        for (size_t row = 0; row < input.size(); ++row) {
          for (size_t column = 0; column < input.size();
               ++column) {
            output[row] +=
                matrix[row * input.size() + column] *
                input[column];
          }
        }
        return output;
      };
  const auto maximumVectorDelta =
      [](const std::vector<double> &a,
         const std::vector<double> &b) {
        if (a.size() != b.size())
          return std::numeric_limits<double>::infinity();
        double maximum = 0.0;
        for (size_t index = 0; index < a.size(); ++index) {
          maximum =
              std::max(maximum, std::fabs(a[index] - b[index]));
        }
        return maximum;
      };
  const auto solve =
      [&](const std::vector<uint64_t> &keys,
          const std::vector<double> &matrix) {
        SolveResult result;
        const MaterialGraphMultilevelHierarchy hierarchy =
            buildMaterialGraphMultilevelHierarchy(keys, matrix);
        if (!hierarchy.finite)
          return result;
        std::vector<double> expected(keys.size(), 0.0);
        for (size_t row = 0; row < keys.size(); ++row) {
          expected[row] =
              0.35 +
              std::sin(3.14159265358979323846 *
                       (static_cast<double>(keys[row] % 1000u) +
                        0.5) /
                       (static_cast<double>(keys.size()) + 1.0));
        }
        const std::vector<double> rhs = multiply(matrix, expected);
        double initialResidual = 0.0;
        for (double value : rhs)
          initialResidual =
              std::max(initialResidual, std::fabs(value));
        std::vector<double> solution(keys.size(), 0.0);
        for (int cycle = 1; cycle <= 64; ++cycle) {
          if (!applyMaterialGraphMultilevelVCycle(
                  hierarchy, rhs, solution, 2, 2)) {
            return SolveResult{};
          }
          const std::vector<double> applied =
              multiply(matrix, solution);
          double residual = 0.0;
          for (size_t row = 0; row < keys.size(); ++row) {
            residual =
                std::max(residual,
                         std::fabs(rhs[row] - applied[row]));
          }
          result.cycles = cycle;
          result.relativeResidual =
              residual / std::max(1.0e-30, initialResidual);
          if (result.relativeResidual <= 1.0e-10) {
            result.converged = true;
            break;
          }
        }
        result.solutionError =
            maximumVectorDelta(solution, expected);
        return result;
      };
  const auto reverseFixture =
      [](const std::vector<uint64_t> &keys,
         const std::vector<double> &matrix,
         std::vector<uint64_t> &reverseKeys,
         std::vector<double> &reverseMatrix) {
        const size_t count = keys.size();
        reverseKeys.resize(count);
        reverseMatrix.assign(count * count, 0.0);
        for (size_t row = 0; row < count; ++row) {
          reverseKeys[row] = keys[count - 1u - row];
          for (size_t column = 0; column < count; ++column) {
            reverseMatrix[row * count + column] =
                matrix[(count - 1u - row) * count +
                       (count - 1u - column)];
          }
        }
      };
  const auto hierarchyDelta =
      [&](const MaterialGraphMultilevelHierarchy &a,
          const MaterialGraphMultilevelHierarchy &b) {
        if (!a.finite || !b.finite ||
            a.levels.size() != b.levels.size()) {
          return std::numeric_limits<double>::infinity();
        }
        double maximum = 0.0;
        for (size_t level = 0; level < a.levels.size();
             ++level) {
          if (a.levels[level].stableKeys !=
                  b.levels[level].stableKeys ||
              a.levels[level].coarseCount !=
                  b.levels[level].coarseCount) {
            return std::numeric_limits<double>::infinity();
          }
          maximum =
              std::max(maximum,
                       maximumVectorDelta(
                           a.levels[level].matrix,
                           b.levels[level].matrix));
          maximum =
              std::max(maximum,
                       maximumVectorDelta(
                           a.levels[level].prolongation,
                           b.levels[level].prolongation));
        }
        return maximum;
      };
  const auto maximumGalerkinEnergyError =
      [&](const MaterialGraphMultilevelHierarchy &hierarchy) {
        double maximum = 0.0;
        for (size_t level = 0;
             level + 1u < hierarchy.levels.size(); ++level) {
          const MaterialGraphMultilevelLevel &fine =
              hierarchy.levels[level];
          const MaterialGraphMultilevelLevel &coarse =
              hierarchy.levels[level + 1u];
          std::vector<double> coarseVector(
              fine.coarseCount, 0.0);
          for (size_t entry = 0;
               entry < coarseVector.size(); ++entry) {
            coarseVector[entry] =
                std::sin(0.61 *
                         static_cast<double>(entry + 1u));
          }
          std::vector<double> fineVector(
              fine.stableKeys.size(), 0.0);
          for (size_t row = 0; row < fineVector.size(); ++row) {
            for (size_t column = 0;
                 column < coarseVector.size(); ++column) {
              fineVector[row] +=
                  fine.prolongation[
                      row * coarseVector.size() + column] *
                  coarseVector[column];
            }
          }
          const std::vector<double> fineApplied =
              multiply(fine.matrix, fineVector);
          const std::vector<double> coarseApplied =
              multiply(coarse.matrix, coarseVector);
          double fineEnergy = 0.0;
          double coarseEnergy = 0.0;
          for (size_t row = 0; row < fineVector.size(); ++row)
            fineEnergy += fineVector[row] * fineApplied[row];
          for (size_t row = 0; row < coarseVector.size(); ++row)
            coarseEnergy +=
                coarseVector[row] * coarseApplied[row];
          maximum =
              std::max(maximum,
                       std::fabs(fineEnergy - coarseEnergy) /
                           std::max(
                               1.0,
                               std::max(std::fabs(fineEnergy),
                                        std::fabs(coarseEnergy))));
        }
        return maximum;
      };

  const std::vector<GraphEdge> chain6 = makeChain(6u, 1000u);
  const std::vector<GraphEdge> chain12 = makeChain(12u, 1000u);
  const std::vector<GraphEdge> chain24 = makeChain(24u, 1000u);
  const std::vector<double> chain6Matrix =
      makeBodyOperator(6u, chain6);
  const std::vector<double> chain12Matrix =
      makeBodyOperator(12u, chain12);
  const std::vector<double> chain24Matrix =
      makeBodyOperator(24u, chain24);
  const std::vector<uint64_t> chain6Keys =
      bodyKeys(6u, 1000u);
  const std::vector<uint64_t> chain12Keys =
      bodyKeys(12u, 1000u);
  const std::vector<uint64_t> chain24Keys =
      bodyKeys(24u, 1000u);
  double phase2aMatrixHash = 0.0;
  for (size_t entry = 0; entry < chain24Matrix.size(); ++entry)
    phase2aMatrixHash +=
        static_cast<double>(entry + 1u) * chain24Matrix[entry];
  printf("  [FasDiag] phase2a matrixHash=%.17g\n",
         phase2aMatrixHash);
  const MaterialGraphMultilevelHierarchy chain24Hierarchy =
      buildMaterialGraphMultilevelHierarchy(
          chain24Keys, chain24Matrix);
  const SolveResult chain6Solve =
      solve(chain6Keys, chain6Matrix);
  const SolveResult chain12Solve =
      solve(chain12Keys, chain12Matrix);
  const SolveResult chain24Solve =
      solve(chain24Keys, chain24Matrix);

  std::vector<uint64_t> reverseKeys;
  std::vector<double> reverseMatrix;
  reverseFixture(chain24Keys, chain24Matrix,
                 reverseKeys, reverseMatrix);
  const MaterialGraphMultilevelHierarchy reverseHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          reverseKeys, reverseMatrix);
  const double orderError =
      hierarchyDelta(chain24Hierarchy, reverseHierarchy);
  const SolveResult reverseSolve =
      solve(reverseKeys, reverseMatrix);

  std::vector<GraphEdge> branch;
  branch.reserve(15u);
  for (size_t body = 0; body < 15u; ++body) {
    GraphEdge edge;
    edge.bodyA = static_cast<int>(body);
    edge.bodyB =
        body == 0u
            ? -1
            : static_cast<int>((body - 1u) / 2u);
    edge.stableKey = 2000u + body;
    branch.push_back(edge);
  }
  const std::vector<uint64_t> branchKeys =
      bodyKeys(15u, 2000u);
  const std::vector<double> branchMatrix =
      makeBodyOperator(15u, branch);
  const MaterialGraphMultilevelHierarchy branchHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          branchKeys, branchMatrix);
  const SolveResult branchSolve =
      solve(branchKeys, branchMatrix);

  std::vector<GraphEdge> loop = makeChain(8u, 3000u);
  GraphEdge closing;
  closing.bodyA = 0;
  closing.bodyB = 7;
  closing.stableKey = 3008u;
  loop.push_back(closing);
  const std::vector<uint64_t> loopKeys =
      bodyKeys(8u, 3000u);
  const std::vector<double> loopMatrix =
      makeBodyOperator(8u, loop);
  const MaterialGraphMultilevelHierarchy loopHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          loopKeys, loopMatrix);
  const SolveResult loopSolve = solve(loopKeys, loopMatrix);

  const size_t nullCount = 12u;
  std::vector<uint64_t> nullKeys(nullCount, 0u);
  std::vector<double> nullLaplacian(
      nullCount * nullCount, 0.0);
  for (size_t row = 0; row < nullCount; ++row) {
    nullKeys[row] = 4000u + row;
    const size_t previous =
        (row + nullCount - 1u) % nullCount;
    const size_t next = (row + 1u) % nullCount;
    nullLaplacian[row * nullCount + row] = 2.0;
    nullLaplacian[row * nullCount + previous] = -1.0;
    nullLaplacian[row * nullCount + next] = -1.0;
  }
  const MaterialGraphMultilevelHierarchy nullHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          nullKeys, nullLaplacian);
  double rigidNullError = 0.0;
  if (nullHierarchy.finite) {
    for (const MaterialGraphMultilevelLevel &level :
         nullHierarchy.levels) {
      std::vector<double> ones(level.stableKeys.size(), 1.0);
      const std::vector<double> response =
          multiply(level.matrix, ones);
      for (double value : response) {
        rigidNullError =
            std::max(rigidNullError, std::fabs(value));
      }
    }
  } else {
    rigidNullError = std::numeric_limits<double>::infinity();
  }

  std::vector<GraphEdge> beforeMerge;
  beforeMerge.push_back({0, -1, 5000u});
  beforeMerge.push_back({1, 0, 5001u});
  beforeMerge.push_back({2, 1, 5002u});
  beforeMerge.push_back({4, 3, 5010u});
  beforeMerge.push_back({5, 4, 5011u});
  std::vector<GraphEdge> afterMerge = beforeMerge;
  afterMerge.push_back({3, 2, 5020u});
  const std::vector<uint64_t> mergeKeys =
      bodyKeys(6u, 5000u);
  const MaterialGraphMultilevelHierarchy beforeMergeHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          mergeKeys,
          makeBodyOperator(6u, beforeMerge));
  const MaterialGraphMultilevelHierarchy afterMergeHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          mergeKeys,
          makeBodyOperator(6u, afterMerge));
  const SolveResult afterMergeSolve =
      solve(mergeKeys, makeBodyOperator(6u, afterMerge));

  const float yaw = 0.73f;
  const Vec3 axis(1.0f, 0.0f, 0.0f);
  const Vec3 yawAxis = rotateAboutY(axis, yaw);
  const double yawScale =
      static_cast<double>(yawAxis.dot(yawAxis)) /
      static_cast<double>(axis.dot(axis));
  std::vector<double> yawMatrix = chain24Matrix;
  for (double &value : yawMatrix)
    value *= yawScale;
  const MaterialGraphMultilevelHierarchy yawHierarchy =
      buildMaterialGraphMultilevelHierarchy(
          chain24Keys, yawMatrix);
  const double yawError =
      hierarchyDelta(chain24Hierarchy, yawHierarchy);

  const double energyError =
      std::max(
          maximumGalerkinEnergyError(chain24Hierarchy),
          std::max(
              maximumGalerkinEnergyError(branchHierarchy),
              maximumGalerkinEnergyError(loopHierarchy)));

  printf("  chain cycles=(%d,%d,%d) residual=(%.3g,%.3g,%.3g) "
         "solution=(%.3g,%.3g,%.3g) levels=%zu\n",
         chain6Solve.cycles, chain12Solve.cycles,
         chain24Solve.cycles, chain6Solve.relativeResidual,
         chain12Solve.relativeResidual,
         chain24Solve.relativeResidual,
         chain6Solve.solutionError, chain12Solve.solutionError,
         chain24Solve.solutionError,
         chain24Hierarchy.levels.size());
  printf("  branch cycles=%d residual=%.3g "
         "loop=(cycles=%d residual=%.3g) rigidNull=%.3g "
         "merge=(%zu->%zu cycles=%d residual=%.3g)\n",
         branchSolve.cycles, branchSolve.relativeResidual,
         loopSolve.cycles, loopSolve.relativeResidual,
         rigidNullError,
         beforeMergeHierarchy.levels.size(),
         afterMergeHierarchy.levels.size(),
         afterMergeSolve.cycles,
         afterMergeSolve.relativeResidual);
  printf("  invariance order=%.3g reverse=(cycles=%d residual=%.3g) "
         "yaw=%.3g energy=%.3g\n",
         orderError, reverseSolve.cycles,
         reverseSolve.relativeResidual, yawError, energyError);

  CHECK(chain24Hierarchy.finite &&
            branchHierarchy.finite &&
            loopHierarchy.finite &&
            beforeMergeHierarchy.finite &&
            afterMergeHierarchy.finite &&
            yawHierarchy.finite,
        "multilevel hierarchy construction failed");
  CHECK(chain6Solve.converged && chain12Solve.converged &&
            chain24Solve.converged && branchSolve.converged &&
            loopSolve.converged && afterMergeSolve.converged,
        "multilevel V-cycle did not close an anchored graph");
  CHECK(chain24Solve.cycles <= chain6Solve.cycles + 4 &&
            chain24Solve.cycles <= 20,
        "chain V-cycle still scales with graph diameter: %d %d %d",
        chain6Solve.cycles, chain12Solve.cycles,
        chain24Solve.cycles);
  CHECK(branchSolve.cycles <= 20 &&
            loopSolve.cycles <= 20 &&
            afterMergeSolve.cycles <= 20,
        "branch, loop, or transient-merge V-cycle remains topology-bound: %d %d %d",
        branchSolve.cycles, loopSolve.cycles,
        afterMergeSolve.cycles);
  CHECK(orderError <= 2.0e-12 &&
            yawError <= 2.0e-6 &&
            reverseSolve.converged &&
            reverseSolve.cycles == chain24Solve.cycles,
        "multilevel graph depends on input order or yaw");
  CHECK(energyError <= 2.0e-12,
        "Galerkin hierarchy does not preserve coarse work: %.9g",
        energyError);
  CHECK(rigidNullError <= 2.0e-12,
        "rigid null mode was not preserved");
  CHECK(chain24Solve.solutionError <= 2.0e-8 &&
            branchSolve.solutionError <= 2.0e-8 &&
            loopSolve.solutionError <= 2.0e-8 &&
            afterMergeSolve.solutionError <= 2.0e-8,
        "multilevel solution disagrees with the dense graph authority");

  PASS("multilevel body/manifold aggregation preserves work and removes chain-diameter low-frequency scaling");
}

bool probe157_materialSpatialTransferAuthority() {
  printf("\n--- Probe 157: Material spatial transfer authority ---\n");

  const auto rotateInertia =
      [](const Mat33 &inertia, float yaw) {
        const float cosine = std::cos(yaw);
        const float sine = std::sin(yaw);
        Mat33 rotation;
        rotation.m[0][0] = cosine;
        rotation.m[0][2] = -sine;
        rotation.m[1][1] = 1.0f;
        rotation.m[2][0] = sine;
        rotation.m[2][2] = cosine;
        return rotation.mul(inertia).mul(rotation.transpose());
      };
  const auto rotateSpatial =
      [](const MaterialWorldSpatialVector &value, float yaw) {
        const Vec3 linear = rotateAboutY(
            Vec3(static_cast<float>(value[0]),
                 static_cast<float>(value[1]),
                 static_cast<float>(value[2])),
            yaw);
        const Vec3 angular = rotateAboutY(
            Vec3(static_cast<float>(value[3]),
                 static_cast<float>(value[4]),
                 static_cast<float>(value[5])),
            yaw);
        return MaterialWorldSpatialVector{
            linear.x, linear.y, linear.z,
            angular.x, angular.y, angular.z};
      };
  const auto maximumSpatialDelta =
      [](const std::vector<MaterialWorldSpatialVector> &a,
         const std::vector<MaterialWorldSpatialVector> &b) {
        if (a.size() != b.size())
          return std::numeric_limits<double>::infinity();
        double maximum = 0.0;
        for (size_t entry = 0; entry < a.size(); ++entry) {
          for (size_t component = 0; component < 6u;
               ++component) {
            maximum =
                std::max(maximum,
                         std::fabs(a[entry][component] -
                                   b[entry][component]));
          }
        }
        return maximum;
      };
  const auto reverseSpatial =
      [](const std::vector<MaterialWorldSpatialVector> &values) {
        std::vector<MaterialWorldSpatialVector> result = values;
        std::reverse(result.begin(), result.end());
        return result;
      };

  std::vector<MaterialSpatialBody> bodies(5u);
  const uint64_t bodyKeys[5] = {100u, 200u, 300u, 400u, 500u};
  const Vec3 positions[5] = {
      Vec3(-1.0f, 0.25f, 0.5f),
      Vec3(0.2f, 1.0f, -0.4f),
      Vec3(1.25f, 1.8f, 0.75f),
      Vec3(-0.6f, 2.6f, -1.1f),
      Vec3(0.9f, 3.4f, 0.3f)};
  for (size_t body = 0; body < bodies.size(); ++body) {
    bodies[body].stableKey = bodyKeys[body];
    bodies[body].worldPosition = positions[body];
    bodies[body].inverseMass =
        0.2 + 0.15 * static_cast<double>(body);
    bodies[body].worldInverseInertia = rotateInertia(
        Mat33::diag(
            0.25f + 0.05f * static_cast<float>(body),
            0.4f + 0.03f * static_cast<float>(body),
            0.65f + 0.04f * static_cast<float>(body)),
        0.13f * static_cast<float>(body + 1u));
  }

  std::vector<MaterialSpatialInterface> interfaces;
  interfaces.push_back({1000u, 100u, 200u, false});
  interfaces.push_back({1010u, 200u, 300u, false});
  interfaces.push_back({1020u, 200u, 400u, false});
  interfaces.push_back({1030u, 300u, 400u, false});
  interfaces.push_back({1040u, 400u, 0u, true});
  interfaces.push_back({1050u, 400u, 500u, false});
  const MaterialSpatialTransfer transfer =
      buildMaterialSpatialTransfer(bodies, interfaces);
  CHECK(transfer.finite,
        "failed to build canonical spatial transfer");

  std::vector<MaterialWorldSpatialVector> interfaceWrenches(
      interfaces.size());
  std::vector<MaterialWorldSpatialVector> bodyTwists(
      bodies.size());
  for (size_t entry = 0; entry < interfaceWrenches.size();
       ++entry) {
    for (size_t component = 0; component < 6u; ++component) {
      interfaceWrenches[entry][component] =
          std::sin(0.37 *
                   static_cast<double>(
                       1u + entry * 6u + component));
    }
  }
  for (size_t entry = 0; entry < bodyTwists.size(); ++entry) {
    for (size_t component = 0; component < 6u; ++component) {
      bodyTwists[entry][component] =
          std::cos(0.29 *
                   static_cast<double>(
                       1u + entry * 6u + component));
    }
  }

  std::vector<MaterialWorldSpatialVector> bodyLoads;
  std::vector<MaterialWorldSpatialVector> relativeTwists;
  CHECK(scatterMaterialInterfaceWrenchesToBodies(
            transfer, interfaceWrenches, bodyLoads) &&
            gatherMaterialBodyTwistsToInterfaces(
                transfer, bodyTwists, relativeTwists),
        "spatial scatter/gather failed");
  const double interfaceWork =
      materialSpatialWork(interfaceWrenches, relativeTwists);
  const double bodyWork =
      materialSpatialWork(bodyLoads, bodyTwists);
  const double adjointWorkError =
      std::fabs(interfaceWork - bodyWork);

  double comWork = 0.0;
  double twistRoundTripError = 0.0;
  for (size_t body = 0; body < bodies.size(); ++body) {
    const Vec6 comTwist =
        materialBodyWorldOriginTwistToCom(
            bodyTwists[body], bodies[body].worldPosition);
    const MaterialWorldSpatialVector roundTrip =
        materialBodyComTwistToWorldOrigin(
            comTwist, bodies[body].worldPosition);
    for (size_t component = 0; component < 6u; ++component) {
      twistRoundTripError =
          std::max(twistRoundTripError,
                   std::fabs(roundTrip[component] -
                             bodyTwists[body][component]));
    }
    const Vec3 force(
        static_cast<float>(bodyLoads[body][0]),
        static_cast<float>(bodyLoads[body][1]),
        static_cast<float>(bodyLoads[body][2]));
    const Vec3 worldMoment(
        static_cast<float>(bodyLoads[body][3]),
        static_cast<float>(bodyLoads[body][4]),
        static_cast<float>(bodyLoads[body][5]));
    const Vec3 comTorque =
        worldMoment - bodies[body].worldPosition.cross(force);
    comWork +=
        static_cast<double>(force.dot(comTwist.linear())) +
        static_cast<double>(comTorque.dot(comTwist.angular()));
  }
  const double comWorkError = std::fabs(comWork - bodyWork);

  std::vector<MaterialWorldSpatialVector> secondWrenches(
      interfaces.size());
  for (size_t entry = 0; entry < secondWrenches.size(); ++entry) {
    for (size_t component = 0; component < 6u; ++component) {
      secondWrenches[entry][component] =
          0.25 +
          std::cos(0.43 *
                   static_cast<double>(
                       1u + entry * 6u + component));
    }
  }
  std::vector<MaterialWorldSpatialVector> secondLoads;
  std::vector<MaterialWorldSpatialVector> responseBodiesA;
  std::vector<MaterialWorldSpatialVector> responseBodiesB;
  std::vector<MaterialWorldSpatialVector> responseInterfacesA;
  std::vector<MaterialWorldSpatialVector> responseInterfacesB;
  CHECK(scatterMaterialInterfaceWrenchesToBodies(
            transfer, secondWrenches, secondLoads) &&
            applyMaterialBodyWorldOriginMobility(
                transfer, bodyLoads, responseBodiesA) &&
            applyMaterialBodyWorldOriginMobility(
                transfer, secondLoads, responseBodiesB) &&
            gatherMaterialBodyTwistsToInterfaces(
                transfer, responseBodiesA,
                responseInterfacesA) &&
            gatherMaterialBodyTwistsToInterfaces(
                transfer, responseBodiesB,
                responseInterfacesB),
        "spatial mobility response failed");
  const double mobilitySymmetryError =
      std::fabs(
          materialSpatialWork(
              interfaceWrenches, responseInterfacesB) -
          materialSpatialWork(
              secondWrenches, responseInterfacesA));
  const double mobilityEnergy =
      materialSpatialWork(
          interfaceWrenches, responseInterfacesA);

  std::vector<MaterialSpatialInterface> freeInterfaces =
      interfaces;
  freeInterfaces.erase(freeInterfaces.begin() + 4);
  const MaterialSpatialTransfer freeTransfer =
      buildMaterialSpatialTransfer(bodies, freeInterfaces);
  CHECK(freeTransfer.finite,
        "failed to build free spatial transfer");
  const MaterialWorldSpatialVector rigidMode = {
      0.3, -0.2, 0.1, 0.25, -0.4, 0.15};
  std::vector<MaterialWorldSpatialVector> rigidBodyTwists(
      bodies.size(), rigidMode);
  std::vector<MaterialWorldSpatialVector>
      rigidInterfaceTwists;
  CHECK(gatherMaterialBodyTwistsToInterfaces(
            freeTransfer, rigidBodyTwists,
            rigidInterfaceTwists),
        "failed to gather rigid spatial mode");
  double rigidModeError = 0.0;
  for (const MaterialWorldSpatialVector &value :
       rigidInterfaceTwists) {
    for (double component : value)
      rigidModeError =
          std::max(rigidModeError, std::fabs(component));
  }

  std::vector<MaterialWorldSpatialVector> freeWrenches(
      freeInterfaces.size());
  for (size_t entry = 0; entry < freeWrenches.size(); ++entry)
    freeWrenches[entry] = interfaceWrenches[
        entry < 4u ? entry : entry + 1u];
  std::vector<MaterialWorldSpatialVector> freeBodyLoads;
  CHECK(scatterMaterialInterfaceWrenchesToBodies(
            freeTransfer, freeWrenches, freeBodyLoads),
        "failed to scatter free interface wrenches");
  double freeWrenchBalanceError = 0.0;
  for (size_t component = 0; component < 6u; ++component) {
    double sum = 0.0;
    for (const MaterialWorldSpatialVector &load :
         freeBodyLoads) {
      sum += load[component];
    }
    freeWrenchBalanceError =
        std::max(freeWrenchBalanceError, std::fabs(sum));
  }

  std::vector<MaterialSpatialBody> reverseBodies = bodies;
  std::vector<MaterialSpatialInterface> reverseInterfaces =
      interfaces;
  std::reverse(reverseBodies.begin(), reverseBodies.end());
  std::reverse(reverseInterfaces.begin(),
               reverseInterfaces.end());
  const MaterialSpatialTransfer reverseTransfer =
      buildMaterialSpatialTransfer(
          reverseBodies, reverseInterfaces);
  const std::vector<MaterialWorldSpatialVector> reverseWrenches =
      reverseSpatial(interfaceWrenches);
  const std::vector<MaterialWorldSpatialVector> reverseTwists =
      reverseSpatial(bodyTwists);
  std::vector<MaterialWorldSpatialVector> reverseLoads;
  std::vector<MaterialWorldSpatialVector> reverseRelative;
  CHECK(reverseTransfer.finite &&
            scatterMaterialInterfaceWrenchesToBodies(
                reverseTransfer, reverseWrenches,
                reverseLoads) &&
            gatherMaterialBodyTwistsToInterfaces(
                reverseTransfer, reverseTwists,
                reverseRelative),
        "reversed spatial transfer failed");
  reverseLoads = reverseSpatial(reverseLoads);
  reverseRelative = reverseSpatial(reverseRelative);
  const double orderLoadError =
      maximumSpatialDelta(bodyLoads, reverseLoads);
  const double orderGatherError =
      maximumSpatialDelta(relativeTwists, reverseRelative);

  const float yaw = 0.61f;
  std::vector<MaterialSpatialBody> yawBodies = bodies;
  for (MaterialSpatialBody &body : yawBodies) {
    body.worldPosition =
        rotateAboutY(body.worldPosition, yaw);
    body.worldInverseInertia =
        rotateInertia(body.worldInverseInertia, yaw);
  }
  const MaterialSpatialTransfer yawTransfer =
      buildMaterialSpatialTransfer(yawBodies, interfaces);
  std::vector<MaterialWorldSpatialVector> yawWrenches(
      interfaceWrenches.size());
  std::vector<MaterialWorldSpatialVector> yawBodyTwists(
      bodyTwists.size());
  for (size_t entry = 0; entry < yawWrenches.size(); ++entry)
    yawWrenches[entry] =
        rotateSpatial(interfaceWrenches[entry], yaw);
  for (size_t entry = 0; entry < yawBodyTwists.size(); ++entry)
    yawBodyTwists[entry] = rotateSpatial(bodyTwists[entry], yaw);
  std::vector<MaterialWorldSpatialVector> yawLoads;
  std::vector<MaterialWorldSpatialVector> yawRelative;
  std::vector<MaterialWorldSpatialVector> yawResponseBodies;
  CHECK(yawTransfer.finite &&
            scatterMaterialInterfaceWrenchesToBodies(
                yawTransfer, yawWrenches, yawLoads) &&
            gatherMaterialBodyTwistsToInterfaces(
                yawTransfer, yawBodyTwists, yawRelative) &&
            applyMaterialBodyWorldOriginMobility(
                yawTransfer, yawLoads, yawResponseBodies),
        "yawed spatial transfer failed");
  std::vector<MaterialWorldSpatialVector> expectedYawLoads(
      bodyLoads.size());
  std::vector<MaterialWorldSpatialVector> expectedYawRelative(
      relativeTwists.size());
  std::vector<MaterialWorldSpatialVector>
      expectedYawResponseBodies(responseBodiesA.size());
  for (size_t entry = 0; entry < bodyLoads.size(); ++entry) {
    expectedYawLoads[entry] = rotateSpatial(bodyLoads[entry], yaw);
    expectedYawResponseBodies[entry] =
        rotateSpatial(responseBodiesA[entry], yaw);
  }
  for (size_t entry = 0; entry < relativeTwists.size(); ++entry) {
    expectedYawRelative[entry] =
        rotateSpatial(relativeTwists[entry], yaw);
  }
  const double yawLoadError =
      maximumSpatialDelta(yawLoads, expectedYawLoads);
  const double yawGatherError =
      maximumSpatialDelta(yawRelative, expectedYawRelative);
  const double yawMobilityError =
      maximumSpatialDelta(
          yawResponseBodies, expectedYawResponseBodies);

  const size_t nullCount = 8u;
  std::vector<uint64_t> nullKeys(nullCount, 0u);
  std::vector<double> nullLaplacian(
      nullCount * nullCount, 0.0);
  for (size_t row = 0; row < nullCount; ++row) {
    nullKeys[row] = 6000u + row;
    const size_t previous =
        (row + nullCount - 1u) % nullCount;
    const size_t next = (row + 1u) % nullCount;
    nullLaplacian[row * nullCount + row] = 2.0;
    nullLaplacian[row * nullCount + previous] = -1.0;
    nullLaplacian[row * nullCount + next] = -1.0;
  }
  const MaterialGraphMultilevelHierarchy hierarchy =
      buildMaterialGraphMultilevelHierarchy(
          nullKeys, nullLaplacian);
  double tensorRigidModeError = 0.0;
  if (hierarchy.finite) {
    for (size_t level = 0;
         level + 1u < hierarchy.levels.size(); ++level) {
      const MaterialGraphMultilevelLevel &value =
          hierarchy.levels[level];
      for (size_t row = 0; row < value.stableKeys.size(); ++row) {
        double sum = 0.0;
        for (size_t coarse = 0;
             coarse < value.coarseCount; ++coarse) {
          sum += value.prolongation[
              row * value.coarseCount + coarse];
        }
        for (size_t component = 0; component < 6u;
             ++component) {
          tensorRigidModeError =
              std::max(tensorRigidModeError,
                       std::fabs(sum - 1.0));
        }
      }
    }
  } else {
    tensorRigidModeError =
        std::numeric_limits<double>::infinity();
  }

  std::vector<MaterialInterfacePoint> points(4u);
  const Vec3 pointCenter(0.1f, 1.35f, -0.2f);
  const Vec3 pointOffsets[4] = {
      Vec3(-0.3f, 0.0f, -0.25f),
      Vec3(-0.3f, 0.0f, 0.25f),
      Vec3(0.3f, 0.0f, -0.25f),
      Vec3(0.3f, 0.0f, 0.25f)};
  for (size_t point = 0; point < points.size(); ++point) {
    points[point].worldPoint = pointCenter + pointOffsets[point];
    points[point].normal = Vec3(0.0f, 1.0f, 0.0f);
    points[point].tangent0 = Vec3(1.0f, 0.0f, 0.0f);
    points[point].tangent1 = Vec3(0.0f, 0.0f, 1.0f);
    points[point].stableKey = 7000u + point;
  }
  const MaterialInterfaceWrenchMap pointMap =
      buildMaterialInterfaceWrenchMap(points);
  const std::vector<double> pointImpulses = {
      1.0, 0.2, -0.1, 0.7, -0.1, 0.15,
      1.3, 0.05, 0.2, 0.9, -0.2, -0.05};
  MaterialSpatialWrench pointWrench{};
  CHECK(pointMap.finite &&
            restrictMaterialPointImpulses(
                pointMap, pointImpulses, pointWrench),
        "point-to-interface bridge failed");
  std::vector<MaterialWorldSpatialVector> pointBridgeWrenches(
      interfaces.size());
  pointBridgeWrenches[0] = pointWrench;
  std::vector<MaterialWorldSpatialVector> pointBridgeLoads;
  CHECK(scatterMaterialInterfaceWrenchesToBodies(
            transfer, pointBridgeWrenches,
            pointBridgeLoads),
        "interface-to-body bridge failed");
  double pointBridgeError = 0.0;
  for (size_t component = 0; component < 6u; ++component) {
    pointBridgeError =
        std::max(pointBridgeError,
                 std::fabs(pointBridgeLoads[0][component] -
                           pointWrench[component]));
    pointBridgeError =
        std::max(pointBridgeError,
                 std::fabs(pointBridgeLoads[1][component] +
                           pointWrench[component]));
  }

  printf("  work adjoint=%.9g com=%.9g roundtrip=%.9g "
         "mobility=(sym=%.9g energy=%.9g)\n",
         adjointWorkError, comWorkError, twistRoundTripError,
         mobilitySymmetryError, mobilityEnergy);
  printf("  rigid=(interface=%.9g balance=%.9g tensor=%.9g) "
         "order=(%.9g,%.9g) pointBridge=%.9g\n",
         rigidModeError, freeWrenchBalanceError,
         tensorRigidModeError, orderLoadError,
         orderGatherError, pointBridgeError);
  printf("  yaw=(load=%.9g gather=%.9g mobility=%.9g)\n",
         yawLoadError, yawGatherError, yawMobilityError);

  CHECK(adjointWorkError <= 2.0e-12 &&
            comWorkError <= 2.0e-5 &&
            twistRoundTripError <= 2.0e-6,
        "world-origin transfer is not work-adjoint to COM dynamics");
  CHECK(mobilitySymmetryError <= 2.0e-5 &&
            mobilityEnergy >= -2.0e-8 &&
            std::isfinite(mobilityEnergy),
        "world-origin body mobility is not symmetric positive");
  CHECK(rigidModeError <= 2.0e-12 &&
            freeWrenchBalanceError <= 2.0e-12 &&
            tensorRigidModeError <= 2.0e-12,
        "spatial transfer does not preserve rigid modes or internal balance");
  CHECK(orderLoadError == 0.0 && orderGatherError == 0.0,
        "spatial transfer depends on input storage order");
  CHECK(yawLoadError <= 2.0e-6 &&
            yawGatherError <= 2.0e-6 &&
            yawMobilityError <= 2.0e-5,
        "spatial transfer or body mobility depends on world yaw");
  CHECK(pointBridgeError <= 2.0e-12,
        "point/interface/body spatial bridge changed the net wrench");

  PASS("world-origin spatial transfers are work-adjoint and preserve six rigid body modes");
}
