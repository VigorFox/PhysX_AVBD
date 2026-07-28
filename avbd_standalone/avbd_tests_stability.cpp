#include "avbd_test_utils.h"
#include "avbd_bounded_component_projection.h"
#include "avbd_component_unilateral_projection.h"
#include "avbd_unilateral_projection.h"
#include <cmath>
#include <cstdio>
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

bool test1_singleBoxOnGround() {
  printf("\n--- Test 1: Single box on ground ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  float density = 10.0f;
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, density, 0.5f);

  addBoxGroundContacts(solver, box, halfExt);

  for (int frame = 0; frame < 120; frame++) {
    solver.verbose = (frame == 0);
    solver.step(solver.dt);
  }

  float finalY = solver.bodies[box].position.y;
  CHECK(fabsf(finalY - 1.0f) < 0.1f, "box drifted too much: y=%.4f", finalY);
  PASS("single box stable on ground");
}

bool test2_twoBoxStack() {
  printf("\n--- Test 2: Two boxes stacked ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  float density = 10.0f;
  uint32_t bottom = solver.addBody({0, 1, 0}, Quat(), halfExt, density, 0.5f);
  uint32_t top = solver.addBody({0, 3, 0}, Quat(), halfExt, density, 0.5f);

  addBoxGroundContacts(solver, bottom, halfExt);
  addBoxOnBoxContacts(solver, top, bottom, halfExt, halfExt);

  for (int frame = 0; frame < 120; frame++)
    solver.step(solver.dt);

  CHECK(fabsf(solver.bodies[bottom].position.y - 1.0f) < 0.2f,
        "bottom drifted");
  CHECK(fabsf(solver.bodies[top].position.y - 3.0f) < 0.2f, "top drifted");
  PASS("two boxes stable");
}

bool test3_fiveBoxTower() {
  printf("\n--- Test 3: 5-box tower ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  float density = 10.0f;
  const int N = 5;
  uint32_t boxIds[N];
  for (int i = 0; i < N; i++) {
    boxIds[i] =
        solver.addBody({0, 1.0f + 2.0f * i, 0}, Quat(), halfExt, density, 0.5f);
  }

  addBoxGroundContacts(solver, boxIds[0], halfExt);
  for (int i = 1; i < N; i++)
    addBoxOnBoxContacts(solver, boxIds[i], boxIds[i - 1], halfExt, halfExt);

  for (int frame = 0; frame < 240; frame++)
    solver.step(solver.dt);

  for (int i = 0; i < N; i++) {
    CHECK(fabsf(solver.bodies[boxIds[i]].position.y - (1.0f + 2.0f * i)) < 0.5f,
          "tower collapsed");
  }
  PASS("5-box tower stable");
}

bool test4_pyramid() {
  printf("\n--- Test 4: 2-layer pyramid ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  float density = 10.0f;
  uint32_t b0 = solver.addBody({-1, 1, 0}, Quat(), halfExt, density, 0.5f);
  uint32_t b1 = solver.addBody({1, 1, 0}, Quat(), halfExt, density, 0.5f);
  uint32_t b2 = solver.addBody({0, 3, 0}, Quat(), halfExt, density, 0.5f);

  addBoxGroundContacts(solver, b0, halfExt);
  addBoxGroundContacts(solver, b1, halfExt);
  addBoxOnBoxContacts(solver, b2, b0, halfExt, halfExt);
  addBoxOnBoxContacts(solver, b2, b1, halfExt, halfExt);

  for (int frame = 0; frame < 240; frame++)
    solver.step(solver.dt);

  CHECK(fabsf(solver.bodies[b2].position.y - 3.0f) < 0.3f, "top box drifted");
  PASS("2-layer pyramid stable");
}

bool test5_dropFromHeight() {
  printf("\n--- Test 5: Drop from height (dynamic contacts + cache) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 3, 0}, Quat(), halfExt, 10.0f, 0.5f);
  ContactCache cache;

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    addBoxGroundContactsDynamic(solver, box, halfExt, 0.15f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  CHECK(fabsf(solver.bodies[box].position.y - 1.0f) < 0.15f,
        "box didn't settle");
  PASS("drop from height settled");
}

bool test6_perFrameRegenWithCache() {
  printf("\n--- Test 6: Per-frame contact regen + cache ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
  ContactCache cache;

  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    addBoxGroundContacts(solver, box, halfExt);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  CHECK(fabsf(solver.bodies[box].position.y - 1.0f) < 0.01f,
        "regen mode drifted");
  PASS("per-frame regen with cache stable");
}

bool test7_physxScale() {
  printf("\n--- Test 7: PhysX-scale (4x4x4 box, mass=640) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(2, 2, 2);
  float density = 10.0f;
  uint32_t box = solver.addBody({0, 2, 0}, Quat(), halfExt, density, 0.5f);
  ContactCache cache;

  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    addBoxGroundContacts(solver, box, halfExt);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }
  CHECK(fabsf(solver.bodies[box].position.y - 2.0f) < 0.05f,
        "PhysX-scale single box drifted");
  PASS("PhysX-scale stable");
}

bool test8_asymmetricMass() {
  printf("\n--- Test 8: Asymmetric mass ratio (10:1) ---\n");
  {
    Solver solver;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t bot = solver.addBody({0, 1, 0}, Quat(), halfExt, 100.0f, 0.5f);
    uint32_t top = solver.addBody({0, 3, 0}, Quat(), halfExt, 10.0f, 0.5f);
    ContactCache cache;
    for (int frame = 0; frame < 180; frame++) {
      solver.contacts.clear();
      addBoxGroundContacts(solver, bot, halfExt);
      addBoxOnBoxContacts(solver, top, bot, halfExt, halfExt);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    CHECK(fabsf(solver.bodies[bot].position.y - 1.0f) < 0.1f, "A: bot drifted");
  }
  PASS("asymmetric mass ratio stable");
}

bool test9_tenBoxTower() {
  printf("\n--- Test 9: 10-box tower (stress test) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 15;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  const int N = 10;
  uint32_t boxIds[N];
  for (int i = 0; i < N; i++)
    boxIds[i] =
        solver.addBody({0, 1.0f + 2.0f * i, 0}, Quat(), halfExt, 10.0f, 0.5f);

  ContactCache cache;
  bool candidatePcgOk = true;
  int candidateMaxPcgIterations = 0;
  double candidateWorstPcgResidual = 0.0;
  for (int frame = 0; frame < 360; frame++) {
    solver.contacts.clear();
    addBoxGroundContacts(solver, boxIds[0], halfExt);
    for (int i = 1; i < N; i++)
      addBoxOnBoxContacts(solver, boxIds[i], boxIds[i - 1], halfExt, halfExt);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    if (solver.useContactIslandPcgProbe) {
      candidatePcgOk = candidatePcgOk &&
                       solver.contactIslandPcgLastStats.converged &&
                       !solver.contactIslandPcgLastStats.breakdown &&
                       solver.contactIslandPcgLastStats.finite;
      candidateMaxPcgIterations =
          std::max(candidateMaxPcgIterations,
                   solver.contactIslandPcgLastStats.iterations);
      candidateWorstPcgResidual =
          std::max(candidateWorstPcgResidual,
                   solver.contactIslandPcgLastStats
                       .finalPreconditionedResidual);
    }
  }

  if (solver.useContactIslandPcgProbe) {
    float maxHeightError = 0.0f;
    float minBottom = INFINITY;
    for (int i = 0; i < N; ++i) {
      maxHeightError =
          std::max(maxHeightError,
                   std::fabs(solver.bodies[boxIds[i]].position.y -
                             (1.0f + 2.0f * i)));
      minBottom =
          std::min(minBottom, solver.bodies[boxIds[i]].position.y - 1.0f);
    }
    printf("[ContactCandidateTower] maxHeightError=%.7g minBottom=%.7g "
           "topY=%.7g pcg=(%d,%d,%.7g)\n",
           maxHeightError, minBottom,
           solver.bodies[boxIds[N - 1]].position.y,
           candidatePcgOk ? 1 : 0, candidateMaxPcgIterations,
           candidateWorstPcgResidual);
  }

  for (int i = 0; i < N; i++) {
    CHECK(fabsf(solver.bodies[boxIds[i]].position.y - (1.0f + 2.0f * i)) < 0.5f,
          "10-box tower collapsed");
  }
  PASS("10-box tower stable");
}

bool test10_longTermStability() {
  printf("\n--- Test 10: Long-term stability (10 seconds) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t b0 = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
  uint32_t b1 = solver.addBody({0, 3, 0}, Quat(), halfExt, 10.0f, 0.5f);
  uint32_t b2 = solver.addBody({0, 5, 0}, Quat(), halfExt, 10.0f, 0.5f);
  ContactCache cache;

  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    addBoxGroundContacts(solver, b0, halfExt);
    addBoxOnBoxContacts(solver, b1, b0, halfExt, halfExt);
    addBoxOnBoxContacts(solver, b2, b1, halfExt, halfExt);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  CHECK(fabsf(solver.bodies[b2].position.y - 5.0f) < 0.1f, "b2 drifted");
  PASS("long-term stable");
}

static UnilateralProjectionRow makeProjectionRow(
    const Vec3 &linearJacobian, const Vec3 &angularJacobian,
    float outwardVelocity, uint64_t stableKey) {
  UnilateralProjectionRow row;
  row.linearJacobian = linearJacobian;
  row.angularJacobian = angularJacobian;
  row.outwardVelocity = outwardVelocity;
  row.stableKey = stableKey;
  return row;
}

static float projectionMaximumResidual(
    const std::vector<UnilateralProjectionRow> &rows,
    const UnilateralProjectionResult &result) {
  float maximum = -INFINITY;
  for (const UnilateralProjectionRow &row : rows) {
    const float residual =
        row.outwardVelocity +
        row.linearJacobian.dot(result.velocityDelta.linear()) +
        row.angularJacobian.dot(result.velocityDelta.angular());
    maximum = std::max(maximum, residual);
  }
  return maximum;
}

static float projectionDeltaDifference(
    const UnilateralProjectionResult &a,
    const UnilateralProjectionResult &b) {
  float maximum = 0.0f;
  for (int component = 0; component < 6; ++component) {
    maximum =
        std::max(maximum,
                 std::fabs(a.velocityDelta[component] -
                           b.velocityDelta[component]));
  }
  return maximum;
}

static bool bruteForceProjectionOracle(
    const std::vector<UnilateralProjectionRow> &rows,
    UnilateralProjectionResult &result) {
  if (rows.size() > 8)
    return false;
  const uint32_t subsetCount = 1u << static_cast<uint32_t>(rows.size());
  bool found = false;
  double bestObjective = INFINITY;
  std::vector<double> bestLambda(rows.size(), 0.0);
  for (uint32_t subset = 0; subset < subsetCount; ++subset) {
    int activeRows[6] = {};
    int activeCount = 0;
    for (size_t row = 0; row < rows.size(); ++row) {
      if ((subset & (1u << static_cast<uint32_t>(row))) == 0)
        continue;
      if (activeCount >= 6) {
        activeCount = 7;
        break;
      }
      activeRows[activeCount++] = static_cast<int>(row);
    }
    if (activeCount > 6)
      continue;

    double matrix[6][6] = {};
    double rhs[6] = {};
    double activeSolution[6] = {};
    for (int row = 0; row < activeCount; ++row) {
      rhs[row] = rows[static_cast<size_t>(activeRows[row])]
                     .outwardVelocity;
      for (int column = 0; column < activeCount; ++column) {
        const UnilateralProjectionRow &a =
            rows[static_cast<size_t>(activeRows[row])];
        const UnilateralProjectionRow &b =
            rows[static_cast<size_t>(activeRows[column])];
        matrix[row][column] =
            static_cast<double>(a.linearJacobian.dot(
                b.linearJacobian)) +
            static_cast<double>(a.angularJacobian.dot(
                b.angularJacobian));
      }
    }
    if (activeCount > 0 &&
        !UnilateralProjectionDetail::solveDense(
            matrix, rhs, activeCount, 1.0e-7, activeSolution))
      continue;

    std::vector<double> lambda(rows.size(), 0.0);
    bool valid = true;
    for (int row = 0; row < activeCount; ++row) {
      if (activeSolution[row] < -1.0e-6 ||
          !std::isfinite(activeSolution[row])) {
        valid = false;
        break;
      }
      lambda[static_cast<size_t>(activeRows[row])] =
          std::max(0.0, activeSolution[row]);
    }
    double objective = 0.0;
    for (size_t row = 0; row < rows.size() && valid; ++row) {
      double responseImpulse = 0.0;
      for (size_t column = 0; column < rows.size(); ++column) {
        responseImpulse +=
            (static_cast<double>(rows[row].linearJacobian.dot(
                 rows[column].linearJacobian)) +
             static_cast<double>(rows[row].angularJacobian.dot(
                 rows[column].angularJacobian))) *
            lambda[column];
      }
      const double residual =
          static_cast<double>(rows[row].outwardVelocity) -
          responseImpulse;
      if (residual > 2.0e-5)
        valid = false;
      objective -=
          static_cast<double>(rows[row].outwardVelocity) *
          lambda[row];
      objective += 0.5 * lambda[row] * responseImpulse;
    }
    if (!valid || !std::isfinite(objective))
      continue;
    if (!found || objective < bestObjective) {
      found = true;
      bestObjective = objective;
      bestLambda = lambda;
    }
  }
  if (!found)
    return false;

  Vec3 linearImpulse(0.0f, 0.0f, 0.0f);
  Vec3 angularImpulse(0.0f, 0.0f, 0.0f);
  result.impulses.assign(rows.size(), 0.0f);
  for (size_t row = 0; row < rows.size(); ++row) {
    result.impulses[row] = static_cast<float>(bestLambda[row]);
    linearImpulse += rows[row].linearJacobian * result.impulses[row];
    angularImpulse += rows[row].angularJacobian * result.impulses[row];
  }
  result.velocityDelta =
      Vec6(linearImpulse * -1.0f, angularImpulse * -1.0f);
  result.status = UnilateralProjectionStatus::Solved;
  return true;
}

static float projectionRandomSigned(uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  const float unit =
      static_cast<float>((state >> 8) & 0x00ffffffu) /
      static_cast<float>(0x00ffffffu);
  return unit * 2.0f - 1.0f;
}

bool test141_bodyUnilateralProjectionAuthority() {
  printf("\n--- Test 141: Body-level 6D unilateral projection authority ---\n");
  const Mat33 identityInertia = Mat33::diag(1.0f, 1.0f, 1.0f);
  const Mat33 zeroInertia;
  const float residualTolerance = 2.0e-5f;

  {
    const std::vector<UnilateralProjectionRow> rows = {
        makeProjectionRow({0, 1, 0}, {0, 0, 0}, 2.0f, 1)};
    const UnilateralProjectionResult result =
        solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
    CHECK(result.status == UnilateralProjectionStatus::Solved,
          "single-row projection did not solve");
    CHECK(std::fabs(result.impulses[0] - 2.0f) <= 1.0e-5f,
          "single-row impulse mismatch: %.9g", result.impulses[0]);
    CHECK(projectionMaximumResidual(rows, result) <= residualTolerance,
          "single-row residual remained positive");
  }

  std::vector<UnilateralProjectionRow> fourRows;
  {
    const float x[4] = {-1.0f, 1.0f, 1.0f, -1.0f};
    const float z[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
    for (int row = 0; row < 4; ++row) {
      fourRows.push_back(
          makeProjectionRow({0, 1, 0}, {-z[row], 0, x[row]},
                            1.0f, static_cast<uint64_t>(10 + row)));
    }
    const UnilateralProjectionResult result =
        solveBodyUnilateralProjection(fourRows, 1.0f, identityInertia);
    CHECK(result.status == UnilateralProjectionStatus::Solved,
          "rank-deficient four-row projection did not solve");
    CHECK(result.activeRows <= 3,
          "rank-deficient four-row projection retained %d active rows",
          result.activeRows);
    CHECK(projectionMaximumResidual(fourRows, result) <= residualTolerance,
          "rank-deficient four-row residual remained positive: %.9g",
          projectionMaximumResidual(fourRows, result));
  }

  {
    std::vector<UnilateralProjectionRow> eightRows = fourRows;
    for (int row = 0; row < 4; ++row) {
      UnilateralProjectionRow duplicate = fourRows[static_cast<size_t>(row)];
      duplicate.stableKey = static_cast<uint64_t>(20 + row);
      eightRows.push_back(duplicate);
    }
    std::vector<UnilateralProjectionRow> reversed = eightRows;
    std::reverse(reversed.begin(), reversed.end());
    const UnilateralProjectionResult forward =
        solveBodyUnilateralProjection(eightRows, 1.0f, identityInertia);
    const UnilateralProjectionResult reverse =
        solveBodyUnilateralProjection(reversed, 1.0f, identityInertia);
    CHECK(forward.status == UnilateralProjectionStatus::Solved &&
              reverse.status == UnilateralProjectionStatus::Solved,
          "rank-deficient eight-row projection did not solve");
    CHECK(projectionMaximumResidual(eightRows, forward) <=
              residualTolerance &&
              projectionMaximumResidual(reversed, reverse) <=
                  residualTolerance,
          "rank-deficient eight-row residual remained positive");
    CHECK(projectionDeltaDifference(forward, reverse) <= 1.0e-6f,
          "eight-row input order changed generalized delta: %.9g",
          projectionDeltaDifference(forward, reverse));
  }

  int dependentPivots = 0;
  int multiplierRemovals = 0;
  {
    const float inverseSqrtTwo = 0.7071067811865475f;
    const std::vector<UnilateralProjectionRow> rows = {
        makeProjectionRow({1, 0, 0}, {0, 0, 0}, 1.2f, 30),
        makeProjectionRow({0, 1, 0}, {0, 0, 0}, 1.2f, 31),
        makeProjectionRow({inverseSqrtTwo, inverseSqrtTwo, 0},
                          {0, 0, 0}, 1.5f, 32)};
    const UnilateralProjectionResult result =
        solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
    dependentPivots += result.dependentPivots;
    multiplierRemovals += result.multiplierRemovals;
    CHECK(result.status == UnilateralProjectionStatus::Solved,
          "dependent-row replacement did not solve: status=%d "
          "iterations=%d pivots=%d removals=%d residual=%.9g",
          static_cast<int>(result.status), result.iterations,
          result.dependentPivots, result.multiplierRemovals,
          result.maxResidual);
    CHECK(result.dependentPivots > 0 &&
              result.multiplierRemovals > 0,
          "dependent-row authority did not exercise replacement");
    CHECK(projectionMaximumResidual(rows, result) <= residualTolerance,
          "dependent-row replacement left a positive residual");
  }

  {
    std::vector<UnilateralProjectionRow> rows;
    const int rowCount = 32;
    for (int row = 0; row < rowCount; ++row) {
      const float angle =
          6.283185307179586f * static_cast<float>(row) /
          static_cast<float>(rowCount);
      const float x = std::cos(angle);
      const float z = std::sin(angle);
      rows.push_back(makeProjectionRow(
          {0, 1, 0}, {-z, 0, x},
          1.0f + 0.2f * x - 0.1f * z,
          static_cast<uint64_t>(100 + row)));
    }
    std::vector<UnilateralProjectionRow> permuted;
    permuted.reserve(rows.size());
    for (int row = 0; row < rowCount; ++row)
      permuted.push_back(rows[static_cast<size_t>((row * 11) % rowCount)]);
    const UnilateralProjectionResult canonical =
        solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
    const UnilateralProjectionResult shuffled =
        solveBodyUnilateralProjection(permuted, 1.0f, identityInertia);
    dependentPivots += canonical.dependentPivots;
    multiplierRemovals += canonical.multiplierRemovals;
    CHECK(canonical.status == UnilateralProjectionStatus::Solved &&
              shuffled.status == UnilateralProjectionStatus::Solved,
          "general 32-row projection did not solve");
    CHECK(canonical.activeRows <= 6 && shuffled.activeRows <= 6,
          "general projection exceeded six active rows");
    CHECK(projectionMaximumResidual(rows, canonical) <=
              residualTolerance &&
              projectionMaximumResidual(permuted, shuffled) <=
                  residualTolerance,
          "general 32-row projection left a positive residual");
    CHECK(projectionDeltaDifference(canonical, shuffled) <= 1.0e-5f,
          "32-row input order changed generalized delta: %.9g",
          projectionDeltaDifference(canonical, shuffled));
  }

  int randomizedCases = 0;
  float randomizedMaximumDifference = 0.0f;
  {
    uint32_t randomState = 0x714a93d5u;
    for (int sample = 0; sample < 256; ++sample) {
      const int rowCount = 1 + (sample % 8);
      const Vec3 currentLinear(
          projectionRandomSigned(randomState),
          projectionRandomSigned(randomState),
          projectionRandomSigned(randomState));
      const Vec3 currentAngular(
          projectionRandomSigned(randomState),
          projectionRandomSigned(randomState),
          projectionRandomSigned(randomState));
      std::vector<UnilateralProjectionRow> rows;
      rows.reserve(static_cast<size_t>(rowCount));
      bool hasPositive = false;
      for (int row = 0; row < rowCount; ++row) {
        Vec3 normal(projectionRandomSigned(randomState),
                    projectionRandomSigned(randomState),
                    projectionRandomSigned(randomState));
        const float normalLength = normal.length();
        if (normalLength < 1.0e-4f)
          normal = Vec3(0.0f, 1.0f, 0.0f);
        else
          normal = normal * (1.0f / normalLength);
        const Vec3 arm(projectionRandomSigned(randomState),
                       projectionRandomSigned(randomState),
                       projectionRandomSigned(randomState));
        const Vec3 angularJacobian = arm.cross(normal);
        const float outwardVelocity =
            normal.dot(currentLinear) +
            angularJacobian.dot(currentAngular);
        hasPositive = hasPositive || outwardVelocity > 1.0e-5f;
        rows.push_back(makeProjectionRow(
            normal, angularJacobian, outwardVelocity,
            static_cast<uint64_t>(1000 + sample * 8 + row)));
      }
      if (!hasPositive) {
        for (UnilateralProjectionRow &row : rows)
          row.outwardVelocity = -row.outwardVelocity;
      }

      UnilateralProjectionResult oracle;
      CHECK(bruteForceProjectionOracle(rows, oracle),
            "randomized brute-force oracle failed at sample %d", sample);
      const UnilateralProjectionResult candidate =
          solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
      CHECK(candidate.status == UnilateralProjectionStatus::Solved,
            "randomized active set failed at sample %d: status=%d",
            sample, static_cast<int>(candidate.status));
      CHECK(projectionMaximumResidual(rows, candidate) <=
                residualTolerance,
            "randomized active set left residual at sample %d: %.9g",
            sample, projectionMaximumResidual(rows, candidate));
      const float difference =
          projectionDeltaDifference(candidate, oracle);
      randomizedMaximumDifference =
          std::max(randomizedMaximumDifference, difference);
      CHECK(difference <= 2.0e-4f,
            "randomized active set differs from oracle at sample %d: %.9g",
            sample, difference);

      std::reverse(rows.begin(), rows.end());
      const UnilateralProjectionResult reverse =
          solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
      CHECK(reverse.status == UnilateralProjectionStatus::Solved,
            "randomized reverse order failed at sample %d", sample);
      CHECK(projectionDeltaDifference(candidate, reverse) <= 2.0e-5f,
            "randomized input order changed result at sample %d: %.9g",
            sample, projectionDeltaDifference(candidate, reverse));
      dependentPivots += candidate.dependentPivots;
      multiplierRemovals += candidate.multiplierRemovals;
      ++randomizedCases;
    }
  }

  {
    const std::vector<UnilateralProjectionRow> rows = {
        makeProjectionRow({1, 0, 0}, {0, 0, 0}, 1.0f, 200),
        makeProjectionRow({-1, 0, 0}, {0, 0, 0}, 1.0f, 201)};
    const UnilateralProjectionResult result =
        solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
    CHECK(result.status == UnilateralProjectionStatus::Infeasible,
          "contradictory half-spaces did not fail closed");
  }

  {
    const std::vector<UnilateralProjectionRow> rows = {
        makeProjectionRow({1, 0, 0}, {0, 0, 0}, 1.0f, 210)};
    const UnilateralProjectionResult result =
        solveBodyUnilateralProjection(rows, 0.0f, zeroInertia);
    CHECK(result.status == UnilateralProjectionStatus::Infeasible,
          "zero-response violated row did not fail closed");
  }

  {
    const std::vector<UnilateralProjectionRow> rows = {
        makeProjectionRow({1, 0, 0}, {0, 0, 0}, -1.0f, 220)};
    const UnilateralProjectionResult result =
        solveBodyUnilateralProjection(rows, 1.0f, identityInertia);
    CHECK(result.status == UnilateralProjectionStatus::NoCorrection,
          "satisfied manifold did not remain a no-op");
  }

  CHECK(dependentPivots > 0 && multiplierRemovals > 0,
        "authority did not exercise working-set deletion");
  printf("[BodyUnilateralProjection] dependentPivots=%d removals=%d "
         "fourRows=%zu generalRows=32 randomized=%d oracleDiff=%.9g\n",
         dependentPivots, multiplierRemovals, fourRows.size(),
         randomizedCases, randomizedMaximumDifference);
  PASS("body-level 6D unilateral projection authority");
}

static ComponentProjectionTerm makeComponentProjectionTerm(
    size_t bodyIndex, const Vec3 &linear, const Vec3 &angular) {
  ComponentProjectionTerm term;
  term.bodyIndex = bodyIndex;
  term.linearJacobian = linear;
  term.angularJacobian = angular;
  return term;
}

static ComponentProjectionRow makeComponentProjectionRow(
    const std::vector<ComponentProjectionTerm> &terms,
    float outwardVelocity, uint64_t stableKey) {
  ComponentProjectionRow row;
  row.terms = terms;
  row.outwardVelocity = outwardVelocity;
  row.stableKey = stableKey;
  return row;
}

static double componentProjectionResponse(
    const ComponentProjectionRow &a, const ComponentProjectionRow &b,
    const std::vector<ComponentProjectionBody> &bodies) {
  double response = 0.0;
  for (const ComponentProjectionTerm &termA : a.terms) {
    for (const ComponentProjectionTerm &termB : b.terms) {
      if (termA.bodyIndex != termB.bodyIndex)
        continue;
      const ComponentProjectionBody &body =
          bodies[termA.bodyIndex];
      response +=
          static_cast<double>(body.inverseMassResponse) *
              static_cast<double>(termA.linearJacobian.dot(
                  termB.linearJacobian)) +
          static_cast<double>(termA.angularJacobian.dot(
              body.inverseInertiaResponse *
              termB.angularJacobian));
    }
  }
  return response;
}

static float componentProjectionMaximumResidual(
    const std::vector<ComponentProjectionRow> &rows,
    const ComponentProjectionResult &result) {
  float maximum = 0.0f;
  for (const ComponentProjectionRow &row : rows) {
    float residual = row.outwardVelocity;
    for (const ComponentProjectionTerm &term : row.terms) {
      const Vec6 &delta = result.velocityDeltas[term.bodyIndex];
      residual += term.linearJacobian.dot(delta.linear()) +
                  term.angularJacobian.dot(delta.angular());
    }
    maximum = std::max(maximum, residual);
  }
  return maximum;
}

static float componentProjectionDeltaDifference(
    const ComponentProjectionResult &a,
    const ComponentProjectionResult &b) {
  if (a.velocityDeltas.size() != b.velocityDeltas.size())
    return INFINITY;
  float maximum = 0.0f;
  for (size_t body = 0; body < a.velocityDeltas.size(); ++body) {
    maximum = std::max(
        maximum,
        (a.velocityDeltas[body].linear() -
         b.velocityDeltas[body].linear())
            .length());
    maximum = std::max(
        maximum,
        (a.velocityDeltas[body].angular() -
         b.velocityDeltas[body].angular())
            .length());
  }
  return maximum;
}

static bool bruteForceComponentProjectionOracle(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<ComponentProjectionRow> &rows,
    ComponentProjectionResult &result) {
  if (rows.size() > 8)
    return false;
  const uint32_t subsetCount =
      1u << static_cast<uint32_t>(rows.size());
  bool found = false;
  double bestObjective = INFINITY;
  std::vector<double> bestLambda(rows.size(), 0.0);
  for (uint32_t subset = 0; subset < subsetCount; ++subset) {
    std::vector<int> activeRows;
    for (size_t row = 0; row < rows.size(); ++row) {
      if (subset & (1u << static_cast<uint32_t>(row)))
        activeRows.push_back(static_cast<int>(row));
    }
    const int activeCount = static_cast<int>(activeRows.size());
    std::vector<double> matrix(
        static_cast<size_t>(activeCount * activeCount), 0.0);
    std::vector<double> rhs(static_cast<size_t>(activeCount), 0.0);
    std::vector<double> activeSolution;
    for (int row = 0; row < activeCount; ++row) {
      rhs[static_cast<size_t>(row)] =
          rows[static_cast<size_t>(
                   activeRows[static_cast<size_t>(row)])]
              .outwardVelocity;
      for (int column = 0; column < activeCount; ++column) {
        matrix[static_cast<size_t>(row * activeCount + column)] =
            componentProjectionResponse(
                rows[static_cast<size_t>(
                    activeRows[static_cast<size_t>(row)])],
                rows[static_cast<size_t>(
                    activeRows[static_cast<size_t>(column)])],
                bodies);
      }
    }
    if (activeCount > 0 &&
        !ComponentProjectionDetail::solveDenseDynamic(
            matrix, rhs, activeCount, 1.0e-7, activeSolution))
      continue;

    std::vector<double> lambda(rows.size(), 0.0);
    bool valid = true;
    for (int row = 0; row < activeCount; ++row) {
      const double impulse =
          activeSolution[static_cast<size_t>(row)];
      if (impulse < -1.0e-6 || !std::isfinite(impulse)) {
        valid = false;
        break;
      }
      lambda[static_cast<size_t>(
          activeRows[static_cast<size_t>(row)])] =
          std::max(0.0, impulse);
    }

    double objective = 0.0;
    for (size_t row = 0; row < rows.size() && valid; ++row) {
      double responseImpulse = 0.0;
      for (size_t column = 0; column < rows.size(); ++column) {
        responseImpulse +=
            componentProjectionResponse(rows[row], rows[column],
                                        bodies) *
            lambda[column];
      }
      const double residual =
          static_cast<double>(rows[row].outwardVelocity) -
          responseImpulse;
      if (residual > 2.0e-5)
        valid = false;
      objective -=
          static_cast<double>(rows[row].outwardVelocity) *
          lambda[row];
      objective += 0.5 * lambda[row] * responseImpulse;
    }
    if (!valid || !std::isfinite(objective))
      continue;
    if (!found || objective < bestObjective) {
      found = true;
      bestObjective = objective;
      bestLambda = lambda;
    }
  }
  if (!found)
    return false;

  result.status = UnilateralProjectionStatus::Solved;
  result.impulses.assign(rows.size(), 0.0f);
  result.velocityDeltas.assign(bodies.size(), Vec6());
  std::vector<Vec3> linearImpulses(
      bodies.size(), Vec3(0.0f, 0.0f, 0.0f));
  std::vector<Vec3> angularImpulses(
      bodies.size(), Vec3(0.0f, 0.0f, 0.0f));
  for (size_t row = 0; row < rows.size(); ++row) {
    const float impulse = static_cast<float>(bestLambda[row]);
    result.impulses[row] = impulse;
    for (const ComponentProjectionTerm &term : rows[row].terms) {
      linearImpulses[term.bodyIndex] +=
          term.linearJacobian * impulse;
      angularImpulses[term.bodyIndex] +=
          term.angularJacobian * impulse;
    }
  }
  for (size_t body = 0; body < bodies.size(); ++body) {
    result.velocityDeltas[body] = Vec6(
        linearImpulses[body] *
            (-bodies[body].inverseMassResponse),
        (bodies[body].inverseInertiaResponse *
         angularImpulses[body]) *
            -1.0f);
  }
  return true;
}

bool test142_componentUnilateralProjectionAuthority() {
  printf("\n--- Test 142: Coupled component unilateral projection authority ---\n");
  const Mat33 identityInertia = Mat33::diag(1.0f, 1.0f, 1.0f);
  const float residualTolerance = 3.0e-5f;

  std::vector<ComponentProjectionBody> twoBodies(2);
  for (size_t body = 0; body < twoBodies.size(); ++body) {
    twoBodies[body].inverseMassResponse = 1.0f;
    twoBodies[body].inverseInertiaResponse = identityInertia;
    twoBodies[body].stableKey = static_cast<uint64_t>(10 + body);
  }
  {
    const std::vector<ComponentProjectionRow> rows = {
        makeComponentProjectionRow(
            {makeComponentProjectionTerm(
                 0, Vec3(1.0f, 0.0f, 0.0f), Vec3()),
             makeComponentProjectionTerm(
                 1, Vec3(-1.0f, 0.0f, 0.0f), Vec3())},
            2.0f, 1)};
    const ComponentProjectionResult result =
        solveComponentUnilateralProjection(twoBodies, rows);
    CHECK(result.status == UnilateralProjectionStatus::Solved,
          "two-body contact did not solve");
    CHECK(std::fabs(result.impulses[0] - 1.0f) <= 1.0e-5f,
          "two-body impulse mismatch: %.9g", result.impulses[0]);
    CHECK((result.velocityDeltas[0].linear() +
           result.velocityDeltas[1].linear())
                  .length() <= 1.0e-6f,
          "two-body projection changed internal linear momentum");
    CHECK(componentProjectionMaximumResidual(rows, result) <=
              residualTolerance,
          "two-body projection left positive residual");
  }

  std::vector<ComponentProjectionBody> chainBodies(3);
  for (size_t body = 0; body < chainBodies.size(); ++body) {
    chainBodies[body].inverseMassResponse =
        0.5f + 0.5f * static_cast<float>(body);
    chainBodies[body].inverseInertiaResponse =
        Mat33::diag(0.75f + static_cast<float>(body),
                    1.0f + 0.5f * static_cast<float>(body),
                    1.25f + 0.25f * static_cast<float>(body));
    chainBodies[body].stableKey =
        static_cast<uint64_t>(100 + body * 10);
  }
  const std::vector<ComponentProjectionRow> chainRows = {
      makeComponentProjectionRow(
          {makeComponentProjectionTerm(
               0, Vec3(0.0f, 1.0f, 0.0f),
               Vec3(-0.4f, 0.0f, 0.2f))},
          1.1f, 10),
      makeComponentProjectionRow(
          {makeComponentProjectionTerm(
               0, Vec3(1.0f, 0.0f, 0.0f),
               Vec3(0.0f, 0.3f, -0.2f)),
           makeComponentProjectionTerm(
               1, Vec3(-1.0f, 0.0f, 0.0f),
               Vec3(0.0f, -0.1f, 0.4f))},
          0.8f, 11),
      makeComponentProjectionRow(
          {makeComponentProjectionTerm(
               1, Vec3(0.0f, 0.0f, 1.0f),
               Vec3(0.2f, -0.3f, 0.0f)),
           makeComponentProjectionTerm(
               2, Vec3(0.0f, 0.0f, -1.0f),
               Vec3(-0.1f, 0.5f, 0.0f))},
          0.6f, 12),
      makeComponentProjectionRow(
          {makeComponentProjectionTerm(
               2, Vec3(0.0f, 1.0f, 0.0f),
               Vec3(0.3f, 0.0f, -0.2f))},
          -0.2f, 13)};
  {
    ComponentProjectionResult oracle;
    CHECK(bruteForceComponentProjectionOracle(
              chainBodies, chainRows, oracle),
          "three-body exhaustive oracle failed");
    const ComponentProjectionResult result =
        solveComponentUnilateralProjection(chainBodies, chainRows);
    CHECK(result.status == UnilateralProjectionStatus::Solved,
          "three-body component did not solve");
    CHECK(componentProjectionMaximumResidual(chainRows, result) <=
              residualTolerance,
          "three-body component left positive residual: %.9g",
          componentProjectionMaximumResidual(chainRows, result));
    CHECK(componentProjectionDeltaDifference(result, oracle) <=
              2.0e-5f,
          "three-body result differs from exhaustive oracle: %.9g",
          componentProjectionDeltaDifference(result, oracle));
  }

  {
    const size_t permutation[3] = {2, 0, 1};
    size_t oldToNew[3] = {};
    std::vector<ComponentProjectionBody> permutedBodies(3);
    for (size_t newIndex = 0; newIndex < 3; ++newIndex) {
      const size_t oldIndex = permutation[newIndex];
      oldToNew[oldIndex] = newIndex;
      permutedBodies[newIndex] = chainBodies[oldIndex];
    }
    std::vector<ComponentProjectionRow> permutedRows = chainRows;
    std::reverse(permutedRows.begin(), permutedRows.end());
    for (ComponentProjectionRow &row : permutedRows) {
      for (ComponentProjectionTerm &term : row.terms)
        term.bodyIndex = oldToNew[term.bodyIndex];
      std::reverse(row.terms.begin(), row.terms.end());
    }
    const ComponentProjectionResult canonical =
        solveComponentUnilateralProjection(chainBodies, chainRows);
    const ComponentProjectionResult permuted =
        solveComponentUnilateralProjection(permutedBodies, permutedRows);
    CHECK(canonical.status == UnilateralProjectionStatus::Solved &&
              permuted.status == UnilateralProjectionStatus::Solved,
          "actor/row permutation authority did not solve");
    float maximumDifference = 0.0f;
    for (size_t oldIndex = 0; oldIndex < 3; ++oldIndex) {
      const Vec6 &a = canonical.velocityDeltas[oldIndex];
      const Vec6 &b =
          permuted.velocityDeltas[oldToNew[oldIndex]];
      maximumDifference =
          std::max(maximumDifference,
                   (a.linear() - b.linear()).length());
      maximumDifference =
          std::max(maximumDifference,
                   (a.angular() - b.angular()).length());
    }
    CHECK(maximumDifference <= 2.0e-5f,
          "actor/row permutation changed component result: %.9g",
          maximumDifference);
  }

  int randomizedCases = 0;
  float randomizedMaximumDifference = 0.0f;
  {
    uint32_t randomState = 0x6d284f91u;
    for (int sample = 0; sample < 256; ++sample) {
      const int bodyCount = 2 + (sample % 3);
      const int rowCount = 1 + (sample % 8);
      std::vector<ComponentProjectionBody> bodies(
          static_cast<size_t>(bodyCount));
      std::vector<Vec6> velocities(
          static_cast<size_t>(bodyCount));
      for (int body = 0; body < bodyCount; ++body) {
        bodies[static_cast<size_t>(body)].inverseMassResponse =
            0.5f + 0.35f * static_cast<float>(body + 1);
        bodies[static_cast<size_t>(body)].inverseInertiaResponse =
            Mat33::diag(0.75f + 0.2f * static_cast<float>(body),
                        1.0f + 0.15f * static_cast<float>(body),
                        1.25f + 0.1f * static_cast<float>(body));
        bodies[static_cast<size_t>(body)].stableKey =
            static_cast<uint64_t>(1000 + body);
        velocities[static_cast<size_t>(body)] = Vec6(
            Vec3(projectionRandomSigned(randomState),
                 projectionRandomSigned(randomState),
                 projectionRandomSigned(randomState)),
            Vec3(projectionRandomSigned(randomState),
                 projectionRandomSigned(randomState),
                 projectionRandomSigned(randomState)));
      }

      std::vector<ComponentProjectionRow> rows;
      bool hasPositive = false;
      for (int row = 0; row < rowCount; ++row) {
        const size_t bodyA =
            static_cast<size_t>(row % bodyCount);
        const size_t bodyB =
            static_cast<size_t>((row + 1) % bodyCount);
        Vec3 normal(projectionRandomSigned(randomState),
                    projectionRandomSigned(randomState),
                    projectionRandomSigned(randomState));
        const float normalLength = normal.length();
        normal = normalLength > 1.0e-4f
                     ? normal * (1.0f / normalLength)
                     : Vec3(0.0f, 1.0f, 0.0f);
        const Vec3 armA(projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState));
        const Vec3 armB(projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState));
        const std::vector<ComponentProjectionTerm> terms = {
            makeComponentProjectionTerm(
                bodyA, normal, armA.cross(normal)),
            makeComponentProjectionTerm(
                bodyB, normal * -1.0f,
                armB.cross(normal * -1.0f))};
        float outwardVelocity = 0.0f;
        for (const ComponentProjectionTerm &term : terms) {
          outwardVelocity +=
              term.linearJacobian.dot(
                  velocities[term.bodyIndex].linear()) +
              term.angularJacobian.dot(
                  velocities[term.bodyIndex].angular());
        }
        hasPositive = hasPositive || outwardVelocity > 1.0e-5f;
        rows.push_back(makeComponentProjectionRow(
            terms, outwardVelocity,
            static_cast<uint64_t>(2000 + sample * 8 + row)));
      }
      if (!hasPositive) {
        for (ComponentProjectionRow &row : rows)
          row.outwardVelocity = -row.outwardVelocity;
      }

      ComponentProjectionResult oracle;
      CHECK(bruteForceComponentProjectionOracle(
                bodies, rows, oracle),
            "component randomized oracle failed at sample %d", sample);
      const ComponentProjectionResult candidate =
          solveComponentUnilateralProjection(bodies, rows);
      CHECK(candidate.status == UnilateralProjectionStatus::Solved,
            "component randomized solve failed at sample %d: %d",
            sample, static_cast<int>(candidate.status));
      CHECK(componentProjectionMaximumResidual(rows, candidate) <=
                residualTolerance,
            "component randomized residual at sample %d: %.9g",
            sample,
            componentProjectionMaximumResidual(rows, candidate));
      const float difference =
          componentProjectionDeltaDifference(candidate, oracle);
      randomizedMaximumDifference =
          std::max(randomizedMaximumDifference, difference);
      CHECK(difference <= 3.0e-4f,
            "component randomized oracle difference at sample %d: %.9g",
            sample, difference);

      std::reverse(rows.begin(), rows.end());
      const ComponentProjectionResult reverse =
          solveComponentUnilateralProjection(bodies, rows);
      CHECK(reverse.status == UnilateralProjectionStatus::Solved,
            "component reverse order failed at sample %d", sample);
      CHECK(componentProjectionDeltaDifference(candidate, reverse) <=
                3.0e-5f,
            "component row order changed result at sample %d: %.9g",
            sample,
            componentProjectionDeltaDifference(candidate, reverse));
      ++randomizedCases;
    }
  }

  int broadActiveRows = 0;
  {
    const int bodyCount = 40;
    const int rowCount = 140;
    std::vector<ComponentProjectionBody> bodies(
        static_cast<size_t>(bodyCount));
    std::vector<Vec6> velocities(
        static_cast<size_t>(bodyCount));
    for (int body = 0; body < bodyCount; ++body) {
      bodies[static_cast<size_t>(body)].inverseMassResponse =
          0.5f + 0.02f * static_cast<float>(body);
      bodies[static_cast<size_t>(body)].inverseInertiaResponse =
          identityInertia;
      bodies[static_cast<size_t>(body)].stableKey =
          static_cast<uint64_t>(5000 + body);
      velocities[static_cast<size_t>(body)] = Vec6(
          Vec3(std::sin(0.31f * static_cast<float>(body)),
               std::cos(0.17f * static_cast<float>(body)),
               std::sin(0.23f * static_cast<float>(body))),
          Vec3(std::cos(0.11f * static_cast<float>(body)),
               std::sin(0.29f * static_cast<float>(body)),
               std::cos(0.37f * static_cast<float>(body))));
    }
    std::vector<ComponentProjectionRow> rows;
    for (int row = 0; row < rowCount; ++row) {
      const size_t bodyA =
          static_cast<size_t>(row % (bodyCount - 1));
      const size_t bodyB = bodyA + 1;
      Vec3 normal(std::sin(0.19f * static_cast<float>(row)),
                  1.0f,
                  std::cos(0.13f * static_cast<float>(row)));
      normal = normal * (1.0f / normal.length());
      const Vec3 armA(0.1f * static_cast<float>((row % 5) - 2),
                      0.2f, -0.15f);
      const Vec3 armB(-0.12f, -0.1f,
                      0.08f * static_cast<float>((row % 7) - 3));
      const std::vector<ComponentProjectionTerm> terms = {
          makeComponentProjectionTerm(
              bodyA, normal, armA.cross(normal)),
          makeComponentProjectionTerm(
              bodyB, normal * -1.0f,
              armB.cross(normal * -1.0f))};
      float outwardVelocity = 0.0f;
      for (const ComponentProjectionTerm &term : terms) {
        outwardVelocity +=
            term.linearJacobian.dot(
                velocities[term.bodyIndex].linear()) +
            term.angularJacobian.dot(
                velocities[term.bodyIndex].angular());
      }
      rows.push_back(makeComponentProjectionRow(
          terms, outwardVelocity,
          static_cast<uint64_t>(6000 + row)));
    }
    const ComponentProjectionResult result =
        solveComponentUnilateralProjection(bodies, rows);
    CHECK(result.status == UnilateralProjectionStatus::Solved,
          "70-row/24-body authority failed: status=%d iterations=%d",
          static_cast<int>(result.status), result.iterations);
    CHECK(componentProjectionMaximumResidual(rows, result) <=
              5.0e-5f,
          "70-row/24-body authority left residual: %.9g",
          componentProjectionMaximumResidual(rows, result));
    CHECK(result.activeRows > 6,
          "broad component did not exceed the old single-body rank");
    broadActiveRows = result.activeRows;
  }

  {
    const std::vector<ComponentProjectionBody> body(1, twoBodies[0]);
    const std::vector<ComponentProjectionRow> rows = {
        makeComponentProjectionRow(
            {makeComponentProjectionTerm(
                0, Vec3(1.0f, 0.0f, 0.0f), Vec3())},
            1.0f, 9000),
        makeComponentProjectionRow(
            {makeComponentProjectionTerm(
                0, Vec3(-1.0f, 0.0f, 0.0f), Vec3())},
            1.0f, 9001)};
    const ComponentProjectionResult result =
        solveComponentUnilateralProjection(body, rows);
    CHECK(result.status == UnilateralProjectionStatus::Infeasible,
          "component contradictory half-spaces did not fail closed");
  }

  printf("[ComponentUnilateralProjection] randomized=%d "
         "oracleDiff=%.9g broadBodies=24 broadRows=70 active=%d\n",
         randomizedCases, randomizedMaximumDifference, broadActiveRows);
  PASS("coupled component unilateral projection authority");
}

static BoundedComponentProjectionRow makeBoundedComponentProjectionRow(
    const ComponentProjectionRow &row, float maximumImpulse) {
  BoundedComponentProjectionRow bounded;
  bounded.row = row;
  bounded.maximumImpulse = maximumImpulse;
  return bounded;
}

static double boundedComponentObjective(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<BoundedComponentProjectionRow> &rows,
    const std::vector<float> &impulses) {
  double objective = 0.0;
  for (size_t row = 0; row < rows.size(); ++row) {
    double responseImpulse = 0.0;
    for (size_t column = 0; column < rows.size(); ++column) {
      responseImpulse +=
          componentProjectionResponse(rows[row].row,
                                      rows[column].row, bodies) *
          static_cast<double>(impulses[column]);
    }
    objective +=
        0.5 * static_cast<double>(impulses[row]) *
            responseImpulse -
        static_cast<double>(rows[row].row.outwardVelocity) *
            static_cast<double>(impulses[row]);
  }
  return objective;
}

static float boundedComponentMaximumResidual(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<BoundedComponentProjectionRow> &rows,
    const std::vector<float> &impulses) {
  float maximum = 0.0f;
  for (size_t row = 0; row < rows.size(); ++row) {
    double residual =
        static_cast<double>(rows[row].row.outwardVelocity);
    for (size_t column = 0; column < rows.size(); ++column) {
      residual -=
          componentProjectionResponse(rows[row].row,
                                      rows[column].row, bodies) *
          static_cast<double>(impulses[column]);
    }
    maximum = std::max(maximum, static_cast<float>(residual));
  }
  return maximum;
}

static bool bruteForceBoundedComponentProjectionOracle(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<BoundedComponentProjectionRow> &rows,
    std::vector<float> &bestImpulses, double &bestObjective,
    float tolerance = 2.0e-5f) {
  if (rows.size() > 8)
    return false;

  uint32_t stateCount = 1;
  for (size_t row = 0; row < rows.size(); ++row)
    stateCount *= 3u;
  bool found = false;
  bestObjective = INFINITY;
  bestImpulses.assign(rows.size(), 0.0f);

  for (uint32_t encoded = 0; encoded < stateCount; ++encoded) {
    uint32_t stateCode = encoded;
    std::vector<int> states(rows.size(), 0);
    std::vector<int> freeRows;
    std::vector<double> candidate(rows.size(), 0.0);
    bool valid = true;
    for (size_t row = 0; row < rows.size(); ++row) {
      states[row] = static_cast<int>(stateCode % 3u);
      stateCode /= 3u;
      if (states[row] == 1) {
        freeRows.push_back(static_cast<int>(row));
      } else if (states[row] == 2) {
        candidate[row] =
            static_cast<double>(rows[row].maximumImpulse);
      }
    }

    const int freeCount = static_cast<int>(freeRows.size());
    std::vector<double> matrix(
        static_cast<size_t>(freeCount * freeCount), 0.0);
    std::vector<double> rhs(static_cast<size_t>(freeCount), 0.0);
    std::vector<double> solution;
    for (int freeRow = 0; freeRow < freeCount; ++freeRow) {
      const size_t row =
          static_cast<size_t>(freeRows[static_cast<size_t>(freeRow)]);
      rhs[static_cast<size_t>(freeRow)] =
          static_cast<double>(rows[row].row.outwardVelocity);
      for (size_t fixed = 0; fixed < rows.size(); ++fixed) {
        if (states[fixed] == 2) {
          rhs[static_cast<size_t>(freeRow)] -=
              componentProjectionResponse(rows[row].row,
                                          rows[fixed].row, bodies) *
              candidate[fixed];
        }
      }
      for (int freeColumn = 0; freeColumn < freeCount;
           ++freeColumn) {
        const size_t column = static_cast<size_t>(
            freeRows[static_cast<size_t>(freeColumn)]);
        matrix[static_cast<size_t>(
            freeRow * freeCount + freeColumn)] =
            componentProjectionResponse(rows[row].row,
                                        rows[column].row, bodies);
      }
    }
    if (freeCount > 0 &&
        !ComponentProjectionDetail::solveDenseDynamic(
            matrix, rhs, freeCount, 1.0e-9, solution))
      continue;
    for (int freeRow = 0; freeRow < freeCount; ++freeRow) {
      const size_t row =
          static_cast<size_t>(freeRows[static_cast<size_t>(freeRow)]);
      candidate[row] = solution[static_cast<size_t>(freeRow)];
      if (!std::isfinite(candidate[row]) ||
          candidate[row] < -tolerance ||
          candidate[row] >
              static_cast<double>(rows[row].maximumImpulse) +
                  tolerance) {
        valid = false;
        break;
      }
      candidate[row] = std::max(
          0.0, std::min(
                   candidate[row],
                   static_cast<double>(rows[row].maximumImpulse)));
    }

    double objective = 0.0;
    for (size_t row = 0; row < rows.size() && valid; ++row) {
      double gradient =
          -static_cast<double>(rows[row].row.outwardVelocity);
      for (size_t column = 0; column < rows.size(); ++column) {
        gradient +=
            componentProjectionResponse(rows[row].row,
                                        rows[column].row, bodies) *
            candidate[column];
      }
      if ((states[row] == 0 && gradient < -tolerance) ||
          (states[row] == 1 && std::fabs(gradient) > tolerance) ||
          (states[row] == 2 && gradient > tolerance))
        valid = false;
      objective +=
          0.5 * candidate[row] *
              (gradient +
               static_cast<double>(
                   rows[row].row.outwardVelocity)) -
          static_cast<double>(rows[row].row.outwardVelocity) *
              candidate[row];
    }
    if (!valid || !std::isfinite(objective))
      continue;
    if (!found || objective < bestObjective) {
      found = true;
      bestObjective = objective;
      for (size_t row = 0; row < rows.size(); ++row)
        bestImpulses[row] = static_cast<float>(candidate[row]);
    }
  }
  return found;
}

static float boundedComponentDeltaDifference(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<BoundedComponentProjectionRow> &rows,
    const std::vector<float> &a, const std::vector<float> &b) {
  std::vector<Vec3> linearA(bodies.size(), Vec3());
  std::vector<Vec3> angularA(bodies.size(), Vec3());
  std::vector<Vec3> linearB(bodies.size(), Vec3());
  std::vector<Vec3> angularB(bodies.size(), Vec3());
  for (size_t row = 0; row < rows.size(); ++row) {
    for (const ComponentProjectionTerm &term : rows[row].row.terms) {
      linearA[term.bodyIndex] += term.linearJacobian * a[row];
      angularA[term.bodyIndex] += term.angularJacobian * a[row];
      linearB[term.bodyIndex] += term.linearJacobian * b[row];
      angularB[term.bodyIndex] += term.angularJacobian * b[row];
    }
  }
  float maximum = 0.0f;
  for (size_t body = 0; body < bodies.size(); ++body) {
    maximum = std::max(
        maximum,
        ((linearA[body] - linearB[body]) *
         bodies[body].inverseMassResponse)
            .length());
    maximum = std::max(
        maximum,
        (bodies[body].inverseInertiaResponse *
         (angularA[body] - angularB[body]))
            .length());
  }
  return maximum;
}

enum class MatrixFreeBoundedOracleStatus {
  Solved,
  NoCorrection,
  ResidualUnclassified,
  Infeasible,
  NumericalFailure,
  IterationLimit
};

struct MatrixFreeBoundedOracleResult {
  MatrixFreeBoundedOracleStatus status =
      MatrixFreeBoundedOracleStatus::NumericalFailure;
  std::vector<float> candidateImpulses;
  float maximumKktViolation = 0.0f;
  int iterations = 0;
};

static MatrixFreeBoundedOracleResult solveMatrixFreeBoundedOracle(
    const std::vector<ComponentProjectionBody> &bodies,
    const std::vector<BoundedComponentProjectionRow> &rows,
    float relativeTolerance = 2.0e-6f) {
  MatrixFreeBoundedOracleResult result;
  const size_t rowCount = rows.size();
  result.candidateImpulses.assign(rowCount, 0.0f);
  if (rowCount == 0 || bodies.empty() ||
      !std::isfinite(relativeTolerance) ||
      relativeTolerance <= 0.0f)
    return result;

  double velocityScale = 1.0;
  double impulseScale = 1.0;
  double trace = 0.0;
  double maximumDiagonal = 0.0;
  bool needsCorrection = false;
  std::vector<double> outward(rowCount, 0.0);
  std::vector<double> upper(rowCount, 0.0);
  for (size_t row = 0; row < rowCount; ++row) {
    outward[row] =
        static_cast<double>(rows[row].row.outwardVelocity);
    upper[row] = static_cast<double>(rows[row].maximumImpulse);
    if (!std::isfinite(outward[row]) ||
        !std::isfinite(upper[row]) || upper[row] < 0.0)
      return result;
    needsCorrection = needsCorrection || outward[row] > 0.0;
    velocityScale =
        std::max(velocityScale, std::fabs(outward[row]));
    impulseScale = std::max(impulseScale, upper[row]);
    const double diagonal = componentProjectionResponse(
        rows[row].row, rows[row].row, bodies);
    if (!std::isfinite(diagonal) || diagonal < 0.0)
      return result;
    trace += diagonal;
    maximumDiagonal = std::max(maximumDiagonal, diagonal);
  }
  if (!needsCorrection) {
    result.status = MatrixFreeBoundedOracleStatus::NoCorrection;
    return result;
  }
  if (!std::isfinite(trace) || trace <= 1.0e-14) {
    result.status = MatrixFreeBoundedOracleStatus::Infeasible;
    return result;
  }

  const auto applyResponse =
      [&](const std::vector<double> &impulses,
          std::vector<double> &values) {
        values.assign(rowCount, 0.0);
        for (size_t row = 0; row < rowCount; ++row) {
          for (size_t column = 0; column < rowCount; ++column) {
            values[row] +=
                componentProjectionResponse(
                    rows[row].row, rows[column].row, bodies) *
                impulses[column];
          }
        }
      };
  const double feasibilityTolerance =
      static_cast<double>(relativeTolerance) * velocityScale;
  const double boundTolerance =
      static_cast<double>(relativeTolerance) * impulseScale;
  double lipschitzBound = maximumDiagonal;
  std::vector<double> impulses(rowCount, 0.0);
  std::vector<double> extrapolated(rowCount, 0.0);
  std::vector<double> next(rowCount, 0.0);
  std::vector<double> responseValues;
  std::vector<double> gradientValues(rowCount, 0.0);
  double acceleration = 1.0;
  double currentObjective = 0.0;
  const int iterationLimit =
      std::max(4096, 1024 + 128 * static_cast<int>(bodies.size()));
  bool converged = false;
  const auto takeProjectedStep =
      [&](const std::vector<double> &base,
          std::vector<double> &candidate,
          std::vector<double> &candidateResponse,
          double &candidateObjective) {
        std::vector<double> baseResponse;
        applyResponse(base, baseResponse);
        double baseObjective = 0.0;
        for (size_t row = 0; row < rowCount; ++row) {
          gradientValues[row] = baseResponse[row] - outward[row];
          baseObjective +=
              0.5 * base[row] * baseResponse[row] -
              outward[row] * base[row];
        }
        if (!std::isfinite(baseObjective))
          return false;
        for (;;) {
          const double inverseLipschitz = 1.0 / lipschitzBound;
          double gradientStep = 0.0;
          double stepNormSquared = 0.0;
          for (size_t row = 0; row < rowCount; ++row) {
            candidate[row] = std::min(
                upper[row],
                std::max(
                    0.0,
                    base[row] -
                        inverseLipschitz * gradientValues[row]));
            const double delta = candidate[row] - base[row];
            gradientStep += gradientValues[row] * delta;
            stepNormSquared += delta * delta;
          }
          applyResponse(candidate, candidateResponse);
          candidateObjective = 0.0;
          for (size_t row = 0; row < rowCount; ++row)
            candidateObjective +=
                0.5 * candidate[row] * candidateResponse[row] -
                outward[row] * candidate[row];
          const double modelObjective =
              baseObjective + gradientStep +
              0.5 * lipschitzBound * stepNormSquared;
          const double modelSlack =
              1.0e-13 *
              std::max(
                  1.0,
                  std::max(
                      std::fabs(candidateObjective),
                      std::fabs(modelObjective)));
          if (std::isfinite(candidateObjective) &&
              std::isfinite(modelObjective) &&
              candidateObjective <= modelObjective + modelSlack)
            return true;
          lipschitzBound *= 2.0;
          if (!std::isfinite(lipschitzBound))
            return false;
        }
      };
  for (int iteration = 0; iteration < iterationLimit; ++iteration) {
    double nextObjective = 0.0;
    if (!takeProjectedStep(
            extrapolated, next, responseValues, nextObjective))
      return result;
    const double objectiveSlack =
        1.0e-13 * std::max(1.0, std::fabs(currentObjective));
    if (nextObjective > currentObjective + objectiveSlack) {
      extrapolated = impulses;
      acceleration = 1.0;
      if (!takeProjectedStep(
              extrapolated, next, responseValues,
              nextObjective) ||
          nextObjective > currentObjective + 16.0 * objectiveSlack)
        return result;
    }
    impulses.swap(next);
    currentObjective = nextObjective;
    result.iterations = iteration + 1;
    applyResponse(impulses, responseValues);
    for (size_t row = 0; row < rowCount; ++row)
      gradientValues[row] = responseValues[row] - outward[row];
    const double violation =
        BoundedComponentProjectionDetail::projectedGradientViolation(
            gradientValues, impulses, upper, boundTolerance);
    if (violation <= feasibilityTolerance) {
      converged = true;
      break;
    }
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(
                         1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        (acceleration - 1.0) / nextAcceleration;
    for (size_t row = 0; row < rowCount; ++row)
      extrapolated[row] =
          impulses[row] + momentum * (impulses[row] - next[row]);
    acceleration = nextAcceleration;
    if ((iteration + 1) % 64 == 0) {
      extrapolated = impulses;
      acceleration = 1.0;
    }
  }
  if (!converged) {
    result.status = MatrixFreeBoundedOracleStatus::IterationLimit;
    return result;
  }

  applyResponse(impulses, responseValues);
  for (size_t row = 0; row < rowCount; ++row)
    gradientValues[row] = responseValues[row] - outward[row];
  result.maximumKktViolation = static_cast<float>(
      BoundedComponentProjectionDetail::projectedGradientViolation(
          gradientValues, impulses, upper, boundTolerance));
  double maximumResidual = 0.0;
  for (size_t row = 0; row < rowCount; ++row) {
    maximumResidual =
        std::max(maximumResidual, -gradientValues[row]);
    result.candidateImpulses[row] =
        static_cast<float>(impulses[row]);
  }
  if (!std::isfinite(maximumResidual)) {
    result.status =
        MatrixFreeBoundedOracleStatus::NumericalFailure;
  } else if (maximumResidual >
             4.0 * feasibilityTolerance) {
    result.status =
        MatrixFreeBoundedOracleStatus::ResidualUnclassified;
  } else {
    result.status = MatrixFreeBoundedOracleStatus::Solved;
  }
  return result;
}

bool test143_boundedComponentPositionImpulseAuthority() {
  printf("\n--- Test 143: Bounded component position-impulse authority ---\n");
  const Mat33 identityInertia = Mat33::diag(1.0f, 1.0f, 1.0f);

  float budget = -1.0f;
  CHECK(makePositionNormalImpulseBudget(
            -120.0f, 1.0f / 60.0f, INFINITY, budget) &&
            std::fabs(budget - 2.0f) <= 1.0e-6f,
        "force-to-impulse conversion mismatch: %.9g", budget);
  CHECK(makePositionNormalImpulseBudget(
            -120.0f, 1.0f / 60.0f, 0.75f, budget) &&
            std::fabs(budget - 0.75f) <= 1.0e-6f,
        "authored impulse limit did not tighten position budget");
  CHECK(makePositionNormalImpulseBudget(
            120.0f, 1.0f / 60.0f, INFINITY, budget) &&
            budget == 0.0f,
        "tensile dual incorrectly created a normal impulse budget");
  CHECK(!makePositionNormalImpulseBudget(
            -120.0f, 0.0f, INFINITY, budget),
        "zero dt did not fail closed");

  std::vector<ComponentProjectionBody> oneBody(1);
  oneBody[0].inverseMassResponse = 1.0f;
  oneBody[0].inverseInertiaResponse = identityInertia;
  oneBody[0].stableKey = 10;
  const std::vector<BoundedComponentProjectionRow> threeStates = {
      makeBoundedComponentProjectionRow(
          makeComponentProjectionRow(
              {makeComponentProjectionTerm(
                  0, Vec3(1.0f, 0.0f, 0.0f), Vec3())},
              -0.5f, 101),
          1.0f),
      makeBoundedComponentProjectionRow(
          makeComponentProjectionRow(
              {makeComponentProjectionTerm(
                  0, Vec3(0.0f, 1.0f, 0.0f), Vec3())},
              1.0f, 102),
          2.0f),
      makeBoundedComponentProjectionRow(
          makeComponentProjectionRow(
              {makeComponentProjectionTerm(
                  0, Vec3(0.0f, 0.0f, 1.0f), Vec3())},
              2.0f, 103),
          1.0f)};
  {
    const BoundedComponentProjectionResult result =
        solveBoundedComponentUnilateralProjection(oneBody,
                                                   threeStates);
    CHECK(result.status ==
              BoundedComponentProjectionStatus::BudgetExhausted,
          "lower/free/upper budget case returned status=%d",
          static_cast<int>(result.status));
    CHECK(result.lowerRows == 1 && result.freeRows == 1 &&
              result.upperRows == 1,
          "KKT states mismatch: lower=%d free=%d upper=%d",
          result.lowerRows, result.freeRows, result.upperRows);
    CHECK(std::fabs(result.candidateImpulses[0]) <= 2.0e-5f &&
              std::fabs(result.candidateImpulses[1] - 1.0f) <=
                  2.0e-5f &&
              std::fabs(result.candidateImpulses[2] - 1.0f) <=
                  2.0e-5f,
          "lower/free/upper candidate mismatch");
    CHECK(result.impulses[0] == 0.0f &&
              result.impulses[1] == 0.0f &&
              result.impulses[2] == 0.0f &&
              result.velocityDeltas[0].linear().length() == 0.0f,
          "budget-exhausted component exposed a partial commit");
  }

  {
    std::vector<BoundedComponentProjectionRow> solved = threeStates;
    solved[2].row.outwardVelocity = 1.0f;
    const BoundedComponentProjectionResult result =
        solveBoundedComponentUnilateralProjection(oneBody, solved);
    CHECK(result.status ==
              BoundedComponentProjectionStatus::Solved,
          "exactly saturated component did not solve: status=%d residual=%.9g",
          static_cast<int>(result.status), result.maximumResidual);
    CHECK(result.upperRows == 1 && result.freeRows == 1 &&
              result.lowerRows == 1,
          "exact saturation did not retain all KKT states");
    CHECK(boundedComponentMaximumResidual(
              oneBody, solved, result.impulses) <= 4.0e-5f,
          "solved bounded component left residual");
  }

  {
    const std::vector<BoundedComponentProjectionRow> contradictory = {
        makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                    0, Vec3(1.0f, 0.0f, 0.0f), Vec3())},
                1.0f, 201),
            4.0f),
        makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                    0, Vec3(-1.0f, 0.0f, 0.0f), Vec3())},
                1.0f, 202),
            4.0f)};
    const BoundedComponentProjectionResult result =
        solveBoundedComponentUnilateralProjection(
            oneBody, contradictory);
    CHECK(result.status ==
              BoundedComponentProjectionStatus::Infeasible,
          "contradictory component was not classified infeasible: %d",
          static_cast<int>(result.status));
    CHECK(result.impulses[0] == 0.0f &&
              result.impulses[1] == 0.0f,
          "infeasible component exposed a commit");
  }

  {
    const std::vector<BoundedComponentProjectionRow> dependent = {
        makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                    0, Vec3(1.0f, 0.0f, 0.0f), Vec3())},
                1.0f, 301),
            2.0f),
        makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                    0, Vec3(1.0f, 0.0f, 0.0f), Vec3())},
                1.0f, 302),
            2.0f)};
    const BoundedComponentProjectionResult result =
        solveBoundedComponentUnilateralProjection(oneBody, dependent);
    CHECK(result.status ==
              BoundedComponentProjectionStatus::Solved,
          "rank-deficient duplicate rows did not solve: %d",
          static_cast<int>(result.status));
    CHECK(std::fabs(result.candidateImpulses[0] +
                        result.candidateImpulses[1] -
                    1.0f) <= 3.0e-5f,
          "rank-deficient aggregate impulse mismatch");
  }

  int oracleCases = 0;
  int matrixFreeOracleCases = 0;
  double maximumObjectiveDifference = 0.0;
  double maximumMatrixFreeObjectiveDifference = 0.0;
  float maximumDeltaDifference = 0.0f;
  float maximumMatrixFreeDeltaDifference = 0.0f;
  {
    uint32_t randomState = 0x8a52d12bu;
    for (int sample = 0; sample < 192; ++sample) {
      const int bodyCount = 2 + sample % 3;
      const int rowCount = 1 + sample % 7;
      std::vector<ComponentProjectionBody> bodies(
          static_cast<size_t>(bodyCount));
      for (int body = 0; body < bodyCount; ++body) {
        bodies[static_cast<size_t>(body)].inverseMassResponse =
            0.6f + 0.2f * static_cast<float>(body);
        bodies[static_cast<size_t>(body)].inverseInertiaResponse =
            Mat33::diag(0.8f + 0.1f * static_cast<float>(body),
                        1.0f + 0.15f * static_cast<float>(body),
                        1.2f + 0.05f * static_cast<float>(body));
        bodies[static_cast<size_t>(body)].stableKey =
            static_cast<uint64_t>(1000 + body);
      }
      std::vector<BoundedComponentProjectionRow> rows;
      for (int row = 0; row < rowCount; ++row) {
        const size_t bodyA =
            static_cast<size_t>(row % bodyCount);
        const size_t bodyB =
            static_cast<size_t>((row + 1) % bodyCount);
        Vec3 normal(projectionRandomSigned(randomState),
                    projectionRandomSigned(randomState),
                    projectionRandomSigned(randomState));
        const float normalLength = normal.length();
        normal = normalLength > 1.0e-4f
                     ? normal * (1.0f / normalLength)
                     : Vec3(0.0f, 1.0f, 0.0f);
        const Vec3 armA(projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState));
        const Vec3 armB(projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState),
                        projectionRandomSigned(randomState));
        const float outwardVelocity =
            1.5f * projectionRandomSigned(randomState);
        const float cap =
            1.5f *
            (0.5f + 0.5f * projectionRandomSigned(randomState));
        rows.push_back(makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                     bodyA, normal, armA.cross(normal)),
                 makeComponentProjectionTerm(
                     bodyB, normal * -1.0f,
                     armB.cross(normal * -1.0f))},
                outwardVelocity,
                static_cast<uint64_t>(2000 + sample * 8 + row)),
            std::max(0.0f, cap)));
      }

      std::vector<float> oracleImpulses;
      double oracleObjective = 0.0;
      CHECK(bruteForceBoundedComponentProjectionOracle(
                bodies, rows, oracleImpulses, oracleObjective),
            "bounded exhaustive oracle failed at sample %d", sample);
      const BoundedComponentProjectionResult candidate =
          solveBoundedComponentUnilateralProjection(
              bodies, rows, 2.0e-6f);
      CHECK(candidate.status !=
                BoundedComponentProjectionStatus::NumericalFailure &&
                candidate.status !=
                    BoundedComponentProjectionStatus::IterationLimit,
            "bounded candidate failed at sample %d: status=%d iterations=%d",
            sample, static_cast<int>(candidate.status),
            candidate.iterations);
      const double candidateObjective =
          boundedComponentObjective(
              bodies, rows, candidate.candidateImpulses);
      const double objectiveDifference =
          std::fabs(candidateObjective - oracleObjective);
      maximumObjectiveDifference =
          std::max(maximumObjectiveDifference, objectiveDifference);
      CHECK(objectiveDifference <=
                2.0e-4 * std::max(1.0, std::fabs(oracleObjective)),
            "bounded objective differs from oracle at sample %d: %.9g",
            sample, objectiveDifference);
      const float deltaDifference =
          boundedComponentDeltaDifference(
              bodies, rows, candidate.candidateImpulses,
              oracleImpulses);
      maximumDeltaDifference =
          std::max(maximumDeltaDifference, deltaDifference);
      CHECK(deltaDifference <= 8.0e-4f,
            "bounded generalized delta differs at sample %d: %.9g",
            sample, deltaDifference);
      CHECK(candidate.maximumKktViolation <= 2.0e-5f,
            "bounded KKT violation at sample %d: %.9g",
            sample, candidate.maximumKktViolation);
      const MatrixFreeBoundedOracleResult matrixFree =
          solveMatrixFreeBoundedOracle(bodies, rows, 2.0e-6f);
      const bool expectedMatrixFreeStatus =
          (candidate.status ==
               BoundedComponentProjectionStatus::Solved &&
           matrixFree.status ==
               MatrixFreeBoundedOracleStatus::Solved) ||
          (candidate.status ==
               BoundedComponentProjectionStatus::NoCorrection &&
           matrixFree.status ==
               MatrixFreeBoundedOracleStatus::NoCorrection) ||
          ((candidate.status ==
                BoundedComponentProjectionStatus::BudgetExhausted ||
            candidate.status ==
                BoundedComponentProjectionStatus::Infeasible) &&
           (matrixFree.status ==
                MatrixFreeBoundedOracleStatus::ResidualUnclassified ||
            matrixFree.status ==
                MatrixFreeBoundedOracleStatus::Infeasible));
      CHECK(expectedMatrixFreeStatus,
            "matrix-free status mismatch at sample %d: dense=%d "
            "matrixFree=%d iterations=%d",
            sample, static_cast<int>(candidate.status),
            static_cast<int>(matrixFree.status),
            matrixFree.iterations);
      const double matrixFreeObjective =
          boundedComponentObjective(
              bodies, rows, matrixFree.candidateImpulses);
      const double matrixFreeObjectiveDifference =
          std::fabs(matrixFreeObjective - oracleObjective);
      maximumMatrixFreeObjectiveDifference = std::max(
          maximumMatrixFreeObjectiveDifference,
          matrixFreeObjectiveDifference);
      CHECK(matrixFreeObjectiveDifference <=
                2.0e-4 *
                    std::max(1.0, std::fabs(oracleObjective)),
            "matrix-free objective differs from oracle at sample %d: "
            "%.9g",
            sample, matrixFreeObjectiveDifference);
      const float matrixFreeDeltaDifference =
          boundedComponentDeltaDifference(
              bodies, rows, matrixFree.candidateImpulses,
              oracleImpulses);
      maximumMatrixFreeDeltaDifference = std::max(
          maximumMatrixFreeDeltaDifference,
          matrixFreeDeltaDifference);
      CHECK(matrixFreeDeltaDifference <= 8.0e-4f,
            "matrix-free generalized delta differs at sample %d: %.9g",
            sample, matrixFreeDeltaDifference);
      CHECK(matrixFree.maximumKktViolation <= 2.0e-5f,
            "matrix-free KKT violation at sample %d: %.9g",
            sample, matrixFree.maximumKktViolation);
      ++matrixFreeOracleCases;
      ++oracleCases;
    }
  }

  int broadIterations = 0;
  int broadMatrixFreeIterations = 0;
  {
    const int bodyCount = 24;
    const int rowCount = 70;
    std::vector<ComponentProjectionBody> bodies(
        static_cast<size_t>(bodyCount));
    std::vector<Vec6> velocities(static_cast<size_t>(bodyCount));
    for (int body = 0; body < bodyCount; ++body) {
      bodies[static_cast<size_t>(body)].inverseMassResponse =
          0.5f + 0.02f * static_cast<float>(body);
      bodies[static_cast<size_t>(body)].inverseInertiaResponse =
          identityInertia;
      bodies[static_cast<size_t>(body)].stableKey =
          static_cast<uint64_t>(5000 + body);
      velocities[static_cast<size_t>(body)] = Vec6(
          Vec3(std::sin(0.31f * static_cast<float>(body)),
               std::cos(0.17f * static_cast<float>(body)),
               std::sin(0.23f * static_cast<float>(body))),
          Vec3(std::cos(0.11f * static_cast<float>(body)),
               std::sin(0.29f * static_cast<float>(body)),
               std::cos(0.37f * static_cast<float>(body))));
    }
    std::vector<ComponentProjectionRow> unboundedRows;
    for (int row = 0; row < rowCount; ++row) {
      const size_t bodyA =
          static_cast<size_t>(row % (bodyCount - 1));
      const size_t bodyB = bodyA + 1;
      Vec3 normal(std::sin(0.19f * static_cast<float>(row)),
                  1.0f,
                  std::cos(0.13f * static_cast<float>(row)));
      normal = normal * (1.0f / normal.length());
      const Vec3 armA(0.1f * static_cast<float>((row % 5) - 2),
                      0.2f, -0.15f);
      const Vec3 armB(-0.12f, -0.1f,
                      0.08f * static_cast<float>((row % 7) - 3));
      const std::vector<ComponentProjectionTerm> terms = {
          makeComponentProjectionTerm(
              bodyA, normal, armA.cross(normal)),
          makeComponentProjectionTerm(
              bodyB, normal * -1.0f,
              armB.cross(normal * -1.0f))};
      float outwardVelocity = 0.0f;
      for (const ComponentProjectionTerm &term : terms) {
        outwardVelocity +=
            term.linearJacobian.dot(
                velocities[term.bodyIndex].linear()) +
            term.angularJacobian.dot(
                velocities[term.bodyIndex].angular());
      }
      unboundedRows.push_back(makeComponentProjectionRow(
          terms, outwardVelocity,
          static_cast<uint64_t>(6000 + row)));
    }
    const ComponentProjectionResult unbounded =
        solveComponentUnilateralProjection(bodies, unboundedRows);
    CHECK(unbounded.status == UnilateralProjectionStatus::Solved,
          "broad unbounded authority did not produce budgets");
    std::vector<BoundedComponentProjectionRow> boundedRows;
    for (int row = 0; row < rowCount; ++row) {
      boundedRows.push_back(makeBoundedComponentProjectionRow(
          unboundedRows[static_cast<size_t>(row)],
          0.05f +
              1.25f *
                  unbounded.impulses[static_cast<size_t>(row)]));
    }
    const BoundedComponentProjectionResult bounded =
        solveBoundedComponentUnilateralProjection(
            bodies, boundedRows, 2.0e-5f);
    CHECK(bounded.status ==
              BoundedComponentProjectionStatus::Solved,
          "140-row/40-body bounded authority failed: status=%d "
          "iterations=%d residual=%.9g kkt=%.9g",
          static_cast<int>(bounded.status), bounded.iterations,
          bounded.maximumResidual, bounded.maximumKktViolation);
    CHECK(boundedComponentMaximumResidual(
              bodies, boundedRows, bounded.impulses) <= 1.0e-4f,
          "broad bounded authority left residual");
    const MatrixFreeBoundedOracleResult matrixFree =
        solveMatrixFreeBoundedOracle(
            bodies, boundedRows, 2.0e-5f);
    CHECK(matrixFree.status ==
              MatrixFreeBoundedOracleStatus::Solved,
          "140-row/40-body matrix-free oracle failed: status=%d "
          "iterations=%d kkt=%.9g",
          static_cast<int>(matrixFree.status),
          matrixFree.iterations, matrixFree.maximumKktViolation);
    CHECK(boundedComponentDeltaDifference(
              bodies, boundedRows, bounded.candidateImpulses,
              matrixFree.candidateImpulses) <= 2.0e-3f,
          "140-row/40-body matrix-free response differs from dense");
    broadIterations = bounded.iterations;
    broadMatrixFreeIterations = matrixFree.iterations;
  }

  {
    std::vector<ComponentProjectionBody> bodies(2);
    for (size_t body = 0; body < bodies.size(); ++body) {
      bodies[body].inverseMassResponse = 1.0f;
      bodies[body].inverseInertiaResponse = identityInertia;
      bodies[body].stableKey = static_cast<uint64_t>(700 + body);
    }
    std::vector<BoundedComponentProjectionRow> rows = {
        makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                     0, Vec3(1.0f, 0.0f, 0.0f), Vec3()),
                 makeComponentProjectionTerm(
                     1, Vec3(-1.0f, 0.0f, 0.0f), Vec3())},
                1.0f, 7010),
            2.0f),
        makeBoundedComponentProjectionRow(
            makeComponentProjectionRow(
                {makeComponentProjectionTerm(
                    0, Vec3(0.0f, 1.0f, 0.0f), Vec3())},
                0.4f, 7011),
            1.0f)};
    const BoundedComponentProjectionResult canonical =
        solveBoundedComponentUnilateralProjection(bodies, rows);
    std::reverse(rows.begin(), rows.end());
    const BoundedComponentProjectionResult reversed =
        solveBoundedComponentUnilateralProjection(bodies, rows);
    CHECK(canonical.status ==
              BoundedComponentProjectionStatus::Solved &&
              reversed.status ==
                  BoundedComponentProjectionStatus::Solved,
          "bounded row permutation did not solve");
    CHECK((canonical.velocityDeltas[0].linear() -
           reversed.velocityDeltas[0].linear())
                  .length() <= 3.0e-5f &&
              (canonical.velocityDeltas[1].linear() -
               reversed.velocityDeltas[1].linear())
                      .length() <= 3.0e-5f,
          "bounded row permutation changed body deltas");
  }

  printf("[BoundedComponentProjection] oracle=%d objectiveDiff=%.9g "
         "deltaDiff=%.9g matrixFreeOracle=%d "
         "matrixFreeObjectiveDiff=%.9g matrixFreeDeltaDiff=%.9g "
         "broadBodies=40 broadRows=140 iterations=%d "
         "matrixFreeIterations=%d\n",
         oracleCases, maximumObjectiveDifference,
         maximumDeltaDifference, matrixFreeOracleCases,
         maximumMatrixFreeObjectiveDifference,
         maximumMatrixFreeDeltaDifference, broadIterations,
         broadMatrixFreeIterations);
  PASS("bounded component position-impulse authority");
}
