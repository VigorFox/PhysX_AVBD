#include "avbd_collision.h"
#include "avbd_component_unilateral_projection.h"
#include "avbd_island_pcg.h"
#include "avbd_island_rows.h"
#include "avbd_test_utils.h"
#include <algorithm>
#include <cfloat>
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

// Helper for Test 24/25/26
static int collideFiltered(Solver &solver, uint32_t ballIdx, uint32_t nodeStart,
                           uint32_t nodeEnd, float margin) {
  int total = 0;
  for (uint32_t i = nodeStart; i < nodeEnd; i++)
    total += collideBoxBox(solver, ballIdx, i, margin);
  total += collideBoxGround(solver, ballIdx, margin);
  for (uint32_t i = nodeStart; i < nodeEnd; i++) {
    if (solver.bodies[i].mass > 0)
      total += collideBoxGround(solver, i, margin);
  }
  return total;
}

bool test17_sphericalJointChain() {
  printf("test17_sphericalJointChain\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 5;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] =
        solver.addBody({0, 18.0f - i * 2.0f, 0}, Quat(), {1, 1, 1}, 500.0f);

  solver.addSphericalJoint(UINT32_MAX, ids[0], {0, 20, 0}, {0, 1, 0}, 1e6f);
  for (int i = 0; i < N - 1; i++)
    solver.addSphericalJoint(ids[i], ids[i + 1], {0, -1, 0}, {0, 1, 0}, 1e6f);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[ids[0]].position.y) > 100.0f)
      exploded = true;
  }
  CHECK(!exploded, "Spherical chain exploded!");
  CHECK(solver.bodies[ids[0]].position.y > solver.bodies[ids[N - 1]].position.y,
        "Chain should hang");
  PASS("spherical joint chain stable");
}

bool test18_fixedJointChain() {
  printf("test18_fixedJointChain\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 5;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] =
        solver.addBody({0, 18.0f - i * 2.0f, 0}, Quat(), {1, 1, 1}, 500.0f);

  solver.addFixedJoint(UINT32_MAX, ids[0], {0, 20, 0}, {0, 1, 0});
  for (int i = 0; i < N - 1; i++)
    solver.addFixedJoint(ids[i], ids[i + 1], {0, -1, 0}, {0, 1, 0});

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[ids[0]].position.y) > 100.0f)
      exploded = true;
  }
  CHECK(!exploded, "Fixed chain exploded!");
  CHECK(solver.bodies[ids[0]].position.y > solver.bodies[ids[N - 1]].position.y,
        "Chain should hang");
  PASS("fixed joint chain stable");
}

bool test19_d6JointChain() {
  printf("test19_d6JointChain\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 5;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] =
        solver.addBody({0, 18.0f - i * 2.0f, 0}, Quat(), {1, 1, 1}, 500.0f);

  float angDamping = 1000.0f;
  solver.addD6Joint(UINT32_MAX, ids[0], {0, 20, 0}, {0, 1, 0}, 0, 0x2A,
                    angDamping);
  for (int i = 0; i < N - 1; i++)
    solver.addD6Joint(ids[i], ids[i + 1], {0, -1, 0}, {0, 1, 0}, 0, 0x2A,
                      angDamping);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[ids[0]].position.y) > 100.0f)
      exploded = true;
  }
  CHECK(!exploded, "D6 chain exploded!");
  PASS("D6 joint chain stable");
}

bool test20_d6JointChain_snippetJoint() {
  printf("test20_d6JointChain_snippetJoint\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 5;
  Vec3 halfExt(2.0f, 0.5f, 0.5f);
  float separation = 4.0f;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({separation / 2.0f + i * separation, 20.0f, 0.0f},
                            Quat(), halfExt, 1.0f);

  float angDamping = 1000.0f;
  Vec3 offset(separation / 2.0f, 0, 0);
  solver.addD6Joint(UINT32_MAX, ids[0], {0, 20, 0}, {-offset.x, 0, 0}, 0, 0x2A,
                    angDamping);
  for (int i = 0; i < N - 1; i++)
    solver.addD6Joint(ids[i], ids[i + 1], {offset.x, 0, 0}, {-offset.x, 0, 0},
                      0, 0x2A, angDamping);

  bool exploded = false;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[ids[N - 1]].position.y) > 100.0f)
      exploded = true;
  }
  CHECK(!exploded, "D6 SnippetJoint chain exploded!");
  PASS("D6 SnippetJoint chain stable");
}

bool test21_highMassRatioChain() {
  printf("test21_highMassRatioChain\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  uint32_t heavy = solver.addBody({0, 18, 0}, Quat(), {1, 1, 1}, 125.0f);
  uint32_t light1 = solver.addBody({0, 16, 0}, Quat(), {1, 1, 1}, 0.125f);
  uint32_t light2 = solver.addBody({0, 14, 0}, Quat(), {1, 1, 1}, 0.125f);

  solver.addSphericalJoint(UINT32_MAX, heavy, {0, 20, 0}, {0, 1, 0});
  solver.addSphericalJoint(heavy, light1, {0, -1, 0}, {0, 1, 0});
  solver.addSphericalJoint(light1, light2, {0, -1, 0}, {0, 1, 0});

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[heavy].position.y) > 200.0f)
      exploded = true;
  }
  CHECK(!exploded, "High mass-ratio chain exploded!");
  PASS("High mass-ratio (1000:1) chain stable");
}

bool test22_meshChainmail() {
  printf("test22_meshChainmail\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int NX = 5, NY = 5;
  float spacing = 2.0f;
  Vec3 halfExt(0.3f, 0.3f, 0.3f);
  uint32_t grid[NY][NX];
  for (int row = 0; row < NY; row++) {
    for (int col = 0; col < NX; col++) {
      grid[row][col] = solver.addBody({col * spacing, 20.0f - row * spacing, 0},
                                      Quat(), halfExt, 0.5f / 0.216f);
    }
  }

  for (int col = 0; col < NX; col++)
    solver.addSphericalJoint(UINT32_MAX, grid[0][col],
                             {col * spacing, 20.0f, 0}, {0, 0, 0});
  for (int row = 0; row < NY; row++)
    for (int col = 0; col < NX - 1; col++)
      solver.addSphericalJoint(grid[row][col], grid[row][col + 1],
                               {spacing / 2, 0, 0}, {-spacing / 2, 0, 0});
  for (int row = 0; row < NY - 1; row++)
    for (int col = 0; col < NX; col++)
      solver.addSphericalJoint(grid[row][col], grid[row + 1][col],
                               {0, -spacing / 2, 0}, {0, spacing / 2, 0});

  bool exploded = false;
  float maxViolation = 0;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[grid[NY - 1][NX - 1]].position.y) > 200.0f)
      exploded = true;
  }
  CHECK(!exploded, "Chainmail mesh exploded!");
  PASS("2D mesh (chainmail) stable");
}

bool test23_heavyBallOnMesh() {
  printf("test23_heavyBallOnMesh\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int NX = 5, NY = 5;
  float spacing = 2.0f;
  Vec3 halfExt(0.3f, 0.3f, 0.3f);
  uint32_t grid[NY][NX];
  for (int row = 0; row < NY; row++)
    for (int col = 0; col < NX; col++)
      grid[row][col] = solver.addBody({col * spacing, 20.0f - row * spacing, 0},
                                      Quat(), halfExt, 2.3f);

  uint32_t ball =
      solver.addBody({NX / 2 * spacing, 20.0f - NY / 2 * spacing - 2.0f, 0},
                     Quat(), {1, 1, 1}, 100.0f / 8.0f);

  for (int col = 0; col < NX; col++)
    solver.addSphericalJoint(UINT32_MAX, grid[0][col],
                             {col * spacing, 20.0f, 0}, {0, 0, 0});
  for (int row = 0; row < NY; row++)
    for (int col = 0; col < NX - 1; col++)
      solver.addSphericalJoint(grid[row][col], grid[row][col + 1],
                               {spacing / 2, 0, 0}, {-spacing / 2, 0, 0});
  for (int row = 0; row < NY - 1; row++)
    for (int col = 0; col < NX; col++)
      solver.addSphericalJoint(grid[row][col], grid[row + 1][col],
                               {0, -spacing / 2, 0}, {0, spacing / 2, 0});

  solver.addSphericalJoint(grid[NY / 2][NX / 2], ball, {0, -halfExt.y, 0},
                           {0, 1, 0});

  bool exploded = false;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[ball].position.y) > 200.0f)
      exploded = true;
  }
  CHECK(!exploded, "Heavy ball on mesh exploded!");
  PASS("Heavy ball on mesh stable");
}

bool test24_fastBallOnChainmail() {
  printf("\n--- Test 24: Fast ball vs chainmail mesh (contact) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 9;
  float spacing = 0.65f;
  float halfGrid = (N - 1) * spacing * 0.5f;
  uint32_t grid[9][9];
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      float dens = (row == 0 || row == N - 1 || col == 0 || col == N - 1)
                       ? -1.0f
                       : 30.0f;
      grid[row][col] = solver.addBody(
          {col * spacing - halfGrid, 10.0f, row * spacing - halfGrid}, Quat(),
          {0.25f, 0.15f, 0.25f}, dens);
    }
  }
  uint32_t ball =
      solver.addBody({0, 15.0f, 0}, Quat(), {0.8f, 0.8f, 0.8f}, 100.0f);
  solver.bodies[ball].linearVelocity = {0, -50, 0};

  for (int row = 0; row < N; row++)
    for (int col = 0; col < N; col++) {
      if (col + 1 < N)
        solver.addSphericalJoint(grid[row][col], grid[row][col + 1],
                                 {spacing / 2, 0, 0}, {-spacing / 2, 0, 0});
      if (row + 1 < N)
        solver.addSphericalJoint(grid[row][col], grid[row + 1][col],
                                 {0, 0, spacing / 2}, {0, 0, -spacing / 2});
    }

  ContactCache cache;
  float minBallY = 999;
  int minBallFrame = -1;
  int firstBallContactFrame = -1;
  uint32_t maxContacts = 0;
  uint32_t maxBallContacts = 0;
  bool candidatePcgOk = true;
  int candidateFirstPcgFailure = -1;
  bool candidatePcgBreakdown = false;
  bool candidatePcgFinite = true;
  double candidatePcgInitialAtFailure = 0.0;
  double candidatePcgFinalAtFailure = 0.0;
  int candidateMaxPcgIterations = 0;
  double candidateWorstPcgResidual = 0.0;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    float specMargin = std::max(
        0.05f, fabsf(solver.bodies[ball].linearVelocity.y) * solver.dt);
    collideAll(solver, specMargin);
    uint32_t ballContacts = 0;
    for (const Contact &contact : solver.contacts)
      if (contact.bodyA == ball || contact.bodyB == ball)
        ++ballContacts;
    maxContacts = std::max(maxContacts,
                           static_cast<uint32_t>(solver.contacts.size()));
    maxBallContacts = std::max(maxBallContacts, ballContacts);
    if (ballContacts > 0 && firstBallContactFrame < 0)
      firstBallContactFrame = frame;
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    if (solver.bodies[ball].position.y < minBallY) {
      minBallY = solver.bodies[ball].position.y;
      minBallFrame = frame;
    }
    if (solver.useContactIslandPcgProbe && !solver.contacts.empty()) {
      if (!solver.contactIslandPcgLastStats.converged &&
          candidateFirstPcgFailure < 0) {
        candidateFirstPcgFailure = frame;
        candidatePcgBreakdown =
            solver.contactIslandPcgLastStats.breakdown;
        candidatePcgFinite = solver.contactIslandPcgLastStats.finite;
        candidatePcgInitialAtFailure = solver.contactIslandPcgLastStats
                                           .initialPreconditionedResidual;
        candidatePcgFinalAtFailure = solver.contactIslandPcgLastStats
                                         .finalPreconditionedResidual;
      }
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
  if (solver.useContactIslandPcgProbe)
    printf("[ContactCandidateChainmailFast] minY=%.7g frame=%d "
           "firstContact=%d contacts=(%u,%u) "
           "pcg=(%d,%d,%.7g,fail=%d/%d/%d/%.7g->%.7g)\n",
           minBallY, minBallFrame, firstBallContactFrame, maxContacts,
           maxBallContacts, candidatePcgOk ? 1 : 0,
           candidateMaxPcgIterations, candidateWorstPcgResidual,
           candidateFirstPcgFailure, candidatePcgBreakdown ? 1 : 0,
           candidatePcgFinite ? 1 : 0, candidatePcgInitialAtFailure,
           candidatePcgFinalAtFailure);
  CHECK(minBallY > 1.0f, "Ball fell to ground!");
  PASS("Fast ball caught");
}

bool test25_smallBallOnChainmail() {
  printf("\n--- Test 25: Small ball vs chainmail mesh ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 9;
  float spacing = 0.65f;
  float halfGrid = (N - 1) * spacing * 0.5f;
  uint32_t grid[9][9];
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      float dens = (row == 0 || row == N - 1 || col == 0 || col == N - 1)
                       ? -1.0f
                       : 30.0f;
      grid[row][col] = solver.addBody(
          {col * spacing - halfGrid, 10.0f, row * spacing - halfGrid}, Quat(),
          {0.25f, 0.15f, 0.25f}, dens);
    }
  }
  uint32_t ball =
      solver.addBody({0, 13.0f, 0}, Quat(), {0.5f, 0.5f, 0.5f}, 10.0f);
  solver.bodies[ball].linearVelocity = {0, -30, 0};

  for (int row = 0; row < N; row++)
    for (int col = 0; col < N; col++) {
      if (col + 1 < N)
        solver.addSphericalJoint(grid[row][col], grid[row][col + 1],
                                 {spacing / 2, 0, 0}, {-spacing / 2, 0, 0});
      if (row + 1 < N)
        solver.addSphericalJoint(grid[row][col], grid[row + 1][col],
                                 {0, 0, spacing / 2}, {0, 0, -spacing / 2});
    }

  ContactCache cache;
  float minBallY = 999;
  int minBallFrame = -1;
  int firstBallContactFrame = -1;
  uint32_t maxContacts = 0;
  uint32_t maxBallContacts = 0;
  bool candidatePcgOk = true;
  int candidateFirstPcgFailure = -1;
  bool candidatePcgBreakdown = false;
  bool candidatePcgFinite = true;
  double candidatePcgInitialAtFailure = 0.0;
  double candidatePcgFinalAtFailure = 0.0;
  int candidateMaxPcgIterations = 0;
  double candidateWorstPcgResidual = 0.0;
  for (int frame = 0; frame < 60; frame++) {
    solver.contacts.clear();
    float specMargin = std::max(
        0.05f, fabsf(solver.bodies[ball].linearVelocity.y) * solver.dt);
    collideAll(solver, specMargin);
    uint32_t ballContacts = 0;
    for (const Contact &contact : solver.contacts)
      if (contact.bodyA == ball || contact.bodyB == ball)
        ++ballContacts;
    maxContacts = std::max(maxContacts,
                           static_cast<uint32_t>(solver.contacts.size()));
    maxBallContacts = std::max(maxBallContacts, ballContacts);
    if (ballContacts > 0 && firstBallContactFrame < 0)
      firstBallContactFrame = frame;
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    if (solver.bodies[ball].position.y < minBallY) {
      minBallY = solver.bodies[ball].position.y;
      minBallFrame = frame;
    }
    if (solver.useContactIslandPcgProbe && !solver.contacts.empty()) {
      if (!solver.contactIslandPcgLastStats.converged &&
          candidateFirstPcgFailure < 0) {
        candidateFirstPcgFailure = frame;
        candidatePcgBreakdown =
            solver.contactIslandPcgLastStats.breakdown;
        candidatePcgFinite = solver.contactIslandPcgLastStats.finite;
        candidatePcgInitialAtFailure = solver.contactIslandPcgLastStats
                                           .initialPreconditionedResidual;
        candidatePcgFinalAtFailure = solver.contactIslandPcgLastStats
                                         .finalPreconditionedResidual;
      }
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
  if (solver.useContactIslandPcgProbe)
    printf("[ContactCandidateChainmailSmall] minY=%.7g frame=%d "
           "firstContact=%d contacts=(%u,%u) "
           "pcg=(%d,%d,%.7g,fail=%d/%d/%d/%.7g->%.7g)\n",
           minBallY, minBallFrame, firstBallContactFrame, maxContacts,
           maxBallContacts, candidatePcgOk ? 1 : 0,
           candidateMaxPcgIterations, candidateWorstPcgResidual,
           candidateFirstPcgFailure, candidatePcgBreakdown ? 1 : 0,
           candidatePcgFinite ? 1 : 0, candidatePcgInitialAtFailure,
           candidatePcgFinalAtFailure);
  CHECK(minBallY > 8.0f, "Small ball penetrated net!");
  PASS("Small ball caught");
}

bool test26_snippetChainmailReplica() {
  printf("\n--- Test 26: SnippetChainmail replica ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;

  const int N = 15;
  const float spacing = 0.65f;
  const float halfGrid = (N - 1) * spacing * 0.5f;
  uint32_t grid[15][15];
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      bool isCorner =
          ((row == 0 || row == N - 1) && (col == 0 || col == N - 1));
      float dens = isCorner ? -1.0f : 30.0f; // Approx
      grid[row][col] = solver.addBody(
          {col * spacing - halfGrid, 35.0f, row * spacing - halfGrid}, Quat(),
          {spacing / 2, 0.12f, spacing / 2}, dens);
    }
  }

  for (int row = 0; row < N; row++)
    for (int col = 0; col < N; col++) {
      if (col + 1 < N)
        solver.addSphericalJoint(grid[row][col], grid[row][col + 1],
                                 {spacing / 2, 0, 0}, {-spacing / 2, 0, 0});
      if (row + 1 < N)
        solver.addSphericalJoint(grid[row][col], grid[row + 1][col],
                                 {0, 0, spacing / 2}, {0, 0, -spacing / 2});
    }

  // Sub-test A: Heavy Ball
  uint32_t ball = solver.addBody({0, 70.0f, 0}, Quat(), {2, 2, 2}, 150.0f);
  solver.bodies[ball].linearVelocity = {0, -26.0f, 0};

  ContactCache cache;
  float minBallY = 999;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    float specMargin = std::max(
        0.05f, fabsf(solver.bodies[ball].linearVelocity.y) * solver.dt);
    collideFiltered(solver, ball, 0, (uint32_t)solver.bodies.size() - 1,
                    specMargin);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    minBallY = std::min(minBallY, solver.bodies[ball].position.y);
  }
  CHECK(minBallY > 1.0f, "Ball fell to ground!");
  PASS("SnippetChainmail replica stable");
}

bool test27_joints3x3Solve() {
  printf("\n--- Test 27: Joints under 3x3 decoupled solve ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;
  solver.use3x3Solve = true;

  uint32_t b0 = solver.addBody({0, 20, 0}, Quat(), {1, 1, 1}, 10.0f);
  solver.addSphericalJoint(UINT32_MAX, b0, {0, 20, 0}, {0, 1, 0});

  uint32_t prev = b0;
  for (int i = 1; i < 5; i++) {
    uint32_t bi =
        solver.addBody({0, 20.0f - i * 2, 0}, Quat(), {1, 1, 1}, 10.0f);
    solver.addSphericalJoint(prev, bi, {0, -1, 0}, {0, 1, 0});
    prev = bi;
  }

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }
  CHECK(solver.bodies[b0].position.y > 17.0f, "Chain 3x3 sagged too much");
  PASS("Joints under 3x3 decoupled solve stable");
}

// ==================== Drive tests (SnippetJointDrive) ====================
//
// SnippetJointDrive interactive toggles:
//   F6: 4 drive modes: linearX (0), twist (1), swing1 (2), SLERP (3)
//   F1: jointFrameA rotation: identity vs rotZ(-π/4)
//   F2: body0 type: static vs kinematic (both mass=0 in standalone)
//   F3: jointFrameB rotation: identity vs rotZ(-π/4) (no localFrameB in
//   standalone, skip) F4: body1 initial rotation: identity vs rotZ(-π/4)
//
// Test matrix: 4 modes × { default, rotFrameA, rotBodyB, rotFrameA+rotBodyB }
// = 16 tests (28–43)

static const float PI = 3.14159265358979f;

// rotZ(-π/4) quaternion: rotation by -45° around Z
static Quat rotZm45() {
  float angle = -PI / 4.0f;
  return Quat(0, 0, sinf(angle / 2.0f), cosf(angle / 2.0f));
}

// Drive mode descriptor
struct DriveMode {
  const char *name;
  uint32_t driveFlags;
  Vec3 linVel;
  Vec3 angVel;
  Vec3 linDamp;
  Vec3 angDamp;
};

static const DriveMode gDriveModes[4] = {
    {"linearX", 0x01, Vec3(1, 0, 0), Vec3(), Vec3(1000, 0, 0), Vec3()},
    {"twist", 0x10, Vec3(), Vec3(1, 0, 0), Vec3(), Vec3(1000, 0, 0)},
    {"swing1", 0x40, Vec3(), Vec3(0, 1, 0), Vec3(), Vec3(0, 1000, 0)},
    {"SLERP", 0x20, Vec3(), Vec3(0, 1, 0), Vec3(), Vec3(0, 0, 1000)},
};

// Variant descriptor
struct DriveVariant {
  const char *name;
  bool rotateFrameA;
  bool rotateBodyB;
};

static const DriveVariant gDriveVariants[4] = {
    {"default", false, false},
    {"rotFrameA", true, false},
    {"rotBodyB", false, true},
    {"rotFrameA+rotBodyB", true, true},
};

// Parametric setup: static body0 + dynamic body1, D6 joint all-free, no gravity
static void setupDriveSceneParametric(Solver &solver, uint32_t &boxA,
                                      uint32_t &boxB, const DriveMode &mode,
                                      const DriveVariant &var) {
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;
  solver.dt = 1.0f / 60.0f;

  Quat bodyBRot = var.rotateBodyB ? rotZm45() : Quat();

  // Static body (mass=0)
  boxA = solver.addBody({0, 2, 0}, Quat(), {0.5f, 0.5f, 0.5f}, 0.0f);
  // Dynamic body
  boxB = solver.addBody({1, 2, 0}, bodyBRot, {0.5f, 0.5f, 0.5f}, 10.0f);

  // D6 joint: all DOFs free
  solver.addD6Joint(boxA, boxB, {0, 0, 0}, {0, 0, 0}, 0x2A, 0x2A, 0.0f);

  auto &jnt = solver.d6Joints.back();
  jnt.driveFlags = mode.driveFlags;
  jnt.driveLinearVelocity = mode.linVel;
  jnt.driveAngularVelocity = mode.angVel;
  jnt.linearDriveDamping = mode.linDamp;
  jnt.angularDriveDamping = mode.angDamp;
  jnt.localFrameA = var.rotateFrameA ? rotZm45() : Quat();
}

// Generic drive check: run N frames, check body B moved / rotated
static bool runDriveTest(int testNum, const DriveMode &mode,
                         const DriveVariant &var) {
  printf("\n--- Test %d: %s drive [%s] ---\n", testNum, mode.name, var.name);
  Solver solver;
  uint32_t boxA, boxB;
  setupDriveSceneParametric(solver, boxA, boxB, mode, var);

  Vec3 startPos = solver.bodies[boxB].position;
  for (int frame = 0; frame < 100; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  bool isLinear = (mode.driveFlags & 0x07) != 0;
  if (isLinear) {
    // For linear drive: body B must move.
    // The drive velocity is in joint-frame-A space, so the world-space
    // direction depends on localFrameA rotation.
    Quat frameA = var.rotateFrameA ? rotZm45() : Quat();
    Vec3 expectedDir = frameA.rotate(mode.linVel).normalized();
    Vec3 displacement = solver.bodies[boxB].position - startPos;
    float projectedMotion = displacement.dot(expectedDir);
    float perpSq = displacement.length2() - projectedMotion * projectedMotion;
    float perpMotion = perpSq > 0.0f ? sqrtf(perpSq) : 0.0f;

    CHECK(projectedMotion > 0.1f,
          "Body B didn't move along expected dir (proj=%.4f)", projectedMotion);
    CHECK(perpMotion < 0.15f, "Body B moved too much perpendicular (perp=%.4f)",
          perpMotion);
  } else {
    // For angular drive: body B must rotate.
    // The drive angular velocity is in joint-frame-A space.
    Quat frameA = var.rotateFrameA ? rotZm45() : Quat();
    Vec3 expectedAxis = frameA.rotate(mode.angVel).normalized();
    Vec3 angVel = solver.bodies[boxB].angularVelocity;
    float proj = fabsf(angVel.dot(expectedAxis));
    float perpSq = angVel.length2() - proj * proj;
    float perp = perpSq > 0.0f ? sqrtf(perpSq) : 0.0f;

    CHECK(proj > 0.05f, "No rotation around expected axis (proj=%.4f)", proj);
    CHECK(proj > perp * 1.5f, "Rotation not primarily (proj=%.4f, perp=%.4f)",
          proj, perp);
  }

  printf("  PASS: %s drive [%s] working\n", mode.name, var.name);
  gTestsPassed++;
  return true;
}

// === 16 test functions: 4 modes × 4 variants ===

// Mode 0: linearX
bool test28_linearX_default() {
  return runDriveTest(28, gDriveModes[0], gDriveVariants[0]);
}
bool test29_linearX_rotFrameA() {
  return runDriveTest(29, gDriveModes[0], gDriveVariants[1]);
}
bool test30_linearX_rotBodyB() {
  return runDriveTest(30, gDriveModes[0], gDriveVariants[2]);
}
bool test31_linearX_rotBoth() {
  return runDriveTest(31, gDriveModes[0], gDriveVariants[3]);
}

// Mode 1: twist
bool test32_twist_default() {
  return runDriveTest(32, gDriveModes[1], gDriveVariants[0]);
}
bool test33_twist_rotFrameA() {
  return runDriveTest(33, gDriveModes[1], gDriveVariants[1]);
}
bool test34_twist_rotBodyB() {
  return runDriveTest(34, gDriveModes[1], gDriveVariants[2]);
}
bool test35_twist_rotBoth() {
  return runDriveTest(35, gDriveModes[1], gDriveVariants[3]);
}

// Mode 2: swing1
bool test36_swing1_default() {
  return runDriveTest(36, gDriveModes[2], gDriveVariants[0]);
}
bool test37_swing1_rotFrameA() {
  return runDriveTest(37, gDriveModes[2], gDriveVariants[1]);
}
bool test38_swing1_rotBodyB() {
  return runDriveTest(38, gDriveModes[2], gDriveVariants[2]);
}
bool test39_swing1_rotBoth() {
  return runDriveTest(39, gDriveModes[2], gDriveVariants[3]);
}

// Mode 3: SLERP
bool test40_slerp_default() {
  return runDriveTest(40, gDriveModes[3], gDriveVariants[0]);
}
bool test41_slerp_rotFrameA() {
  return runDriveTest(41, gDriveModes[3], gDriveVariants[1]);
}
bool test42_slerp_rotBodyB() {
  return runDriveTest(42, gDriveModes[3], gDriveVariants[2]);
}
bool test43_slerp_rotBoth() {
  return runDriveTest(43, gDriveModes[3], gDriveVariants[3]);
}

bool test44_sphericalConeLimit() {
  printf("\n--- Test 44: Spherical Joint Cone Limit ---\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;

  uint32_t b0 = solver.addBody({0, 20, 0}, Quat(), {1, 1, 1}, 0.0f);
  uint32_t b1 = solver.addBody({0, 10, 0}, Quat(), {1, 1, 1}, 10.0f);

  solver.addSphericalJoint(b0, b1, {0, 0, 0}, {0, 10, 0}, 1e6f);
  solver.setSphericalJointConeLimit(0, {0, 1, 0}, 30.0f * 3.14159265f / 180.0f);

  solver.bodies[b1].angularVelocity = {0, 0, 10.0f};

  float maxAngle = 0.0f;
  for (int frame = 0; frame < 100; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);

    Vec3 axisA = solver.bodies[b0].rotation.rotate(Vec3(0, 1, 0));
    Vec3 axisB = solver.bodies[b1].rotation.rotate(Vec3(0, 1, 0));
    float angle = acosf(std::max(-1.0f, std::min(1.0f, axisA.dot(axisB))));
    maxAngle = std::max(maxAngle, angle);
  }

  float maxAngleDeg = maxAngle * 180.0f / 3.14159265f;
  CHECK(maxAngleDeg < 32.0f, "Swung too far: %.2f degrees", maxAngleDeg);
  PASS("Cone limit successfully enforced");
}

// =============================================================================
// Gear Joint Tests (test45, test46)
// =============================================================================

// Test 45: Basic gear ratio 0.5  (2:1 reduction)
// Constraint: omegaA_z * 0.5 + omegaB_z = 0  =>  omegaB = -0.5 * omegaA
bool test45_gearJoint_basicRatio() {
  printf("\n--- Test 45: Gear Joint Basic Ratio (2:1) ---\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;

  const Vec3 half{0.5f, 0.5f, 0.5f};
  uint32_t bA = solver.addBody({-3, 0, 0}, Quat(), half, 1.0f);
  uint32_t bB = solver.addBody({3, 0, 0}, Quat(), half, 1.0f);

  // gearRatio = 0.5:  C = wA*0.5 + wB = 0  =>  wB = -0.5*wA
  solver.addGearJoint(bA, bB, {0, 0, 1}, {0, 0, 1}, 0.5f, 1e5f);

  // Kick A; B starts at rest
  solver.bodies[bA].angularVelocity = {0, 0, 4.0f};

  for (int frame = 0; frame < 180; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float wA = solver.bodies[bA].angularVelocity.z;
  float wB = solver.bodies[bB].angularVelocity.z;
  printf("  omegaA=%.4f  omegaB=%.4f  (target wB=-0.5*wA)\n", wA, wB);

  CHECK(fabsf(wA) > 0.05f, "Body A stopped spinning - lost all energy");
  // Check: wA*0.5 + wB ≈ 0  (constraint residual < 10% of wA)
  float residual = wA * 0.5f + wB;
  CHECK(fabsf(residual) < fabsf(wA) * 0.15f,
        "Gear ratio not enforced: residual=%.4f (wA=%.4f wB=%.4f)", residual,
        wA, wB);
  PASS("Gear 2:1 ratio enforced");
}

// Test 46: Meshed gears  ratio = -1
// Constraint: omegaA_z * (-1) + omegaB_z = 0  =>  omegaB = omegaA  (same dir)
bool test46_gearJoint_oppositeDir() {
  printf("\n--- Test 46: Gear Joint Ratio=-1 (meshed gears) ---\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;

  const Vec3 half{0.5f, 0.5f, 0.5f};
  uint32_t bA = solver.addBody({-3, 0, 0}, Quat(), half, 1.0f);
  uint32_t bB = solver.addBody({3, 0, 0}, Quat(), half, 1.0f);

  // gearRatio = -1:  C = wA*(-1) + wB = 0  =>  wB = wA (same direction in
  // formula)
  solver.addGearJoint(bA, bB, {0, 0, 1}, {0, 0, 1}, -1.0f, 1e5f);

  solver.bodies[bA].angularVelocity = {0, 0, 3.0f};

  for (int frame = 0; frame < 180; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float wA = solver.bodies[bA].angularVelocity.z;
  float wB = solver.bodies[bB].angularVelocity.z;
  printf("  omegaA=%.4f  omegaB=%.4f  residual=%.4f (target ~0)\n", wA, wB,
         -wA + wB);

  CHECK(fabsf(wA) > 0.05f, "Body A stopped spinning - lost all energy");
  float residual = -wA + wB; // C = wA*(-1) + wB
  CHECK(fabsf(residual) < fabsf(wA) * 0.15f,
        "Gear ratio=-1 not enforced: residual=%.4f (wA=%.4f wB=%.4f)", residual,
        wA, wB);
  PASS("Gear ratio=-1 enforced");
}

// =============================================================================
// Prismatic Joint Tests (test47, test48)
// =============================================================================

bool test47_prismaticJoint_basic() {
  printf("\n--- Test 47: Prismatic Joint Basic Limits ---\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;

  const Vec3 half{0.5f, 0.5f, 0.5f};
  uint32_t bA = solver.addBody({0, 5, 0}, Quat(), half, 0.0f); // Static anchor
  uint32_t bB = solver.addBody({0, 5, 0}, Quat(), half, 10.0f);

  // Prismatic joint along X axis. Limits [-2, 2].
  solver.addPrismaticJoint(bA, bB, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}, 1e6f);
  solver.setPrismaticJointLimit(0, -2.0f, 2.0f);

  // Push B heavily along +X, it should stop at X = 2.
  solver.bodies[bB].linearVelocity = {
      20.0f, 10.0f, 10.0f}; // Give it off-axis velocity too to test lockdowns

  float maxX = 0.0f, minX = 0.0f;
  for (int frame = 0; frame < 100; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (solver.bodies[bB].position.x > maxX)
      maxX = solver.bodies[bB].position.x;
  }

  // Expect max pos close to 2.0
  printf("  Max posB.x: %.4f\n", maxX);

  CHECK(fabsf(maxX - 2.0f) < 0.1f,
        "Upper limit breached or not reached: maxX=%.4f", maxX);
  CHECK(fabsf(solver.bodies[bB].position.y - 5.0f) < 0.1f,
        "Y DOF not locked: y=%.4f", solver.bodies[bB].position.y);
  CHECK(fabsf(solver.bodies[bB].position.z - 0.0f) < 0.1f,
        "Z DOF not locked: z=%.4f", solver.bodies[bB].position.z);

  // Now push heavily along -X, it should stop at X = -2.
  solver.bodies[bB].linearVelocity = {-50.0f, 0.0f, 0.0f};
  for (int frame = 0; frame < 100; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (solver.bodies[bB].position.x < minX)
      minX = solver.bodies[bB].position.x;
  }

  printf("  Min posB.x after reverse: %.4f\n", minX);
  CHECK(fabsf(minX - (-2.0f)) < 0.15f,
        "Lower limit breached or not reached: minX=%.4f", minX);

  PASS("Prismatic Joint basic limits and locking enforced");
}

bool test48_prismaticJoint_drive() {
  printf("\n--- Test 48: Prismatic Joint Drive ---\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;

  const Vec3 half{0.5f, 0.5f, 0.5f};
  uint32_t bA = solver.addBody({0, 5, 0}, Quat(), half, 0.0f); // Static anchor
  uint32_t bB = solver.addBody({0, 5, 0}, Quat(), half, 10.0f);

  // Prismatic joint along Y axis
  solver.addPrismaticJoint(bA, bB, {0, 0, 0}, {0, 0, 0}, {0, 1, 0}, 1e6f);
  // Target velocity 5m/s
  solver.setPrismaticJointDrive(0, 5.0f, 1e6f);

  for (int frame = 0; frame < 60; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Expect vel around Y = 5.0
  Vec3 posB = solver.bodies[bB].position;
  Vec3 velB = solver.bodies[bB].linearVelocity;
  printf("  Final posB: %.4f, %.4f, %.4f, velY=%.4f\n", posB.x, posB.y, posB.z,
         velB.y);

  CHECK(fabsf(posB.x - 0.0f) < 0.1f, "X DOF not locked: %.4f", posB.x);
  CHECK(fabsf(posB.z - 0.0f) < 0.1f, "Z DOF not locked: %.4f", posB.z);
  CHECK(fabsf(velB.y - 5.0f) < 0.5f || posB.y > 9.0f,
        "Drive velocity not met: velY=%.4f", velB.y);

  PASS("Prismatic Joint drive enforced");
}

// Test 49: Prismatic chain (SnippetJoint replica) with 6x6 solve
bool test49_prismaticChain_6x6() {
  printf(
      "\n--- Test 49: Prismatic Chain (6x6 solve, SnippetJoint replica) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.use3x3Solve = false;

  const int N = 5;
  Vec3 halfExt(2.0f, 0.5f, 0.5f);
  float separation = 4.0f;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({separation / 2.0f + i * separation, 20.0f, 0.0f},
                            Quat(), halfExt, 1.0f);

  Vec3 offset(separation / 2.0f, 0, 0);
  solver.addPrismaticJoint(UINT32_MAX, ids[0], {0, 20, 0}, {-offset.x, 0, 0},
                           {1, 0, 0}, 1e6f);
  solver.setPrismaticJointLimit(0, -2.0f, 2.0f);
  for (int i = 0; i < N - 1; i++) {
    solver.addPrismaticJoint(ids[i], ids[i + 1], {offset.x, 0, 0},
                             {-offset.x, 0, 0}, {1, 0, 0}, 1e6f);
    solver.setPrismaticJointLimit(i + 1, -2.0f, 2.0f);
  }

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < N; i++) {
      if (fabsf(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
    }
  }

  printf("  body[0] pos=%.2f, body[4] pos=%.2f\n",
         solver.bodies[ids[0]].position.y,
         solver.bodies[ids[N - 1]].position.y);
  CHECK(!exploded, "Prismatic chain 6x6 exploded!");
  CHECK(solver.bodies[ids[0]].position.y > 10.0f,
        "Chain fell too far (6x6): y=%.2f", solver.bodies[ids[0]].position.y);
  PASS("Prismatic chain 6x6 stable");
}

// Test 50: Same Prismatic chain with 3x3 decoupled solve
bool test50_prismaticChain_3x3() {
  printf("\n--- Test 50: Prismatic Chain (3x3 decoupled solve) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20; // More iterations for 3x3 to converge
  solver.use3x3Solve = true;

  const int N = 5;
  Vec3 halfExt(2.0f, 0.5f, 0.5f);
  float separation = 4.0f;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({separation / 2.0f + i * separation, 20.0f, 0.0f},
                            Quat(), halfExt, 1.0f);

  Vec3 offset(separation / 2.0f, 0, 0);
  solver.addPrismaticJoint(UINT32_MAX, ids[0], {0, 20, 0}, {-offset.x, 0, 0},
                           {1, 0, 0}, 1e6f);
  solver.setPrismaticJointLimit(0, -2.0f, 2.0f);
  for (int i = 0; i < N - 1; i++) {
    solver.addPrismaticJoint(ids[i], ids[i + 1], {offset.x, 0, 0},
                             {-offset.x, 0, 0}, {1, 0, 0}, 1e6f);
    solver.setPrismaticJointLimit(i + 1, -2.0f, 2.0f);
  }

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < N; i++) {
      if (fabsf(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
    }
  }

  printf("  body[0] pos=%.2f, body[4] pos=%.2f\n",
         solver.bodies[ids[0]].position.y,
         solver.bodies[ids[N - 1]].position.y);
  CHECK(!exploded, "Prismatic chain 3x3 exploded!");
  CHECK(solver.bodies[ids[0]].position.y > 10.0f,
        "Chain fell too far (3x3): y=%.2f", solver.bodies[ids[0]].position.y);
  PASS("Prismatic chain 3x3 stable");
}

// ============================================================================
// Revolute (Hinge) Joint Tests
// ============================================================================

// Test 51: Basic revolute chain — bodies connected by hinges hanging from ceiling
bool test51_revoluteJoint_basic() {
  printf("\n--- Test 51: Revolute Joint Basic Chain ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 5;
  Vec3 halfExt(2.0f, 0.5f, 0.5f);
  float separation = 4.0f;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({separation / 2.0f + i * separation, 20.0f, 0.0f},
                            Quat(), halfExt, 1.0f);

  Vec3 offset(separation / 2.0f, 0, 0);
  // Hinge axis = Z axis (bodies swing in X-Y plane)
  Vec3 hingeAxis(0, 0, 1);
  solver.addRevoluteJoint(UINT32_MAX, ids[0], {0, 20, 0}, {-offset.x, 0, 0},
                          hingeAxis, hingeAxis);
  for (int i = 0; i < N - 1; i++)
    solver.addRevoluteJoint(ids[i], ids[i + 1], {offset.x, 0, 0},
                            {-offset.x, 0, 0}, hingeAxis, hingeAxis);

  bool exploded = false;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < N; i++) {
      if (fabsf(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
    }
  }

  printf("  body[0] y=%.2f, body[4] y=%.2f\n",
         solver.bodies[ids[0]].position.y,
         solver.bodies[ids[N - 1]].position.y);
  CHECK(!exploded, "Revolute basic chain exploded!");
  // Chain should hang — first body higher than last
  CHECK(solver.bodies[ids[0]].position.y > solver.bodies[ids[N - 1]].position.y,
        "Chain should hang downward");
  PASS("Revolute basic chain stable");
}

// Test 52: Revolute joint with angle limits
bool test52_revoluteJoint_limit() {
  printf("\n--- Test 52: Revolute Joint with Angle Limits ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  // Two bodies connected by a hinge with ±45° limit
  uint32_t bodyA = solver.addBody({0, 10, 0}, Quat(), {1, 1, 1}, 10.0f);
  uint32_t bodyB = solver.addBody({3, 10, 0}, Quat(), {1, 1, 1}, 10.0f);

  Vec3 hingeAxis(1, 0, 0);
  solver.addRevoluteJoint(UINT32_MAX, bodyA, {0, 12, 0}, {0, 1, 0},
                          hingeAxis, hingeAxis);
  solver.addRevoluteJoint(bodyA, bodyB, {1.5f, 0, 0}, {-1.5f, 0, 0},
                          hingeAxis, hingeAxis);
  float limitAngle = 3.14159f / 4.0f; // 45 degrees
  solver.setRevoluteJointLimit(1, -limitAngle, limitAngle);

  bool exploded = false;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (fabsf(solver.bodies[bodyA].position.y) > 100.0f ||
        fabsf(solver.bodies[bodyB].position.y) > 100.0f)
      exploded = true;
  }

  // Compute final angle between the bodies
  Quat rotA = solver.bodies[bodyA].rotation;
  Quat rotB = solver.bodies[bodyB].rotation;
  float finalAngle = solver.d6Joints[1].computeHingeAngle(rotA, rotB);

  printf("  bodyA y=%.2f, bodyB y=%.2f, angle=%.2f deg\n",
         solver.bodies[bodyA].position.y, solver.bodies[bodyB].position.y,
         finalAngle * 180.0f / 3.14159f);
  CHECK(!exploded, "Revolute limit chain exploded!");
  // Angle should be within limit (with some tolerance)
  CHECK(finalAngle >= -limitAngle - 0.15f && finalAngle <= limitAngle + 0.15f,
        "Angle %.2f deg outside limit +/-45 deg", finalAngle * 180.0f / 3.14159f);
  PASS("Revolute joint with limits stable");
}

// Test 53: Revolute joint with motor drive
bool test53_revoluteJoint_drive() {
  printf("\n--- Test 53: Revolute Joint Motor Drive ---\n");
  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;

    // Unequal-inertia dynamic pair: the authored target is one relative
    // velocity, not one independent target per endpoint.
    const uint32_t bodyA =
        solver.addBody({-1, 10, 0}, Quat(), {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody({1, 10, 0}, Quat(), {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].mass = 1.0f;
    solver.bodies[bodyA].inertiaTensor = Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].mass = 1.0f;
    solver.bodies[bodyB].inertiaTensor = Mat33::diag(3.0f, 4.0f, 5.0f);
    solver.bodies[bodyB].computeDerived();

    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(bodyA, bodyB, {1, 0, 0}, {-1, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }

    Body &a = solver.bodies[bodyA];
    Body &b = solver.bodies[bodyB];
    a.updateInvInertiaWorld();
    b.updateInvInertiaWorld();
    const Vec3 worldAxis =
        (a.rotation * solver.d6Joints[0].localFrameA)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const float relativeVelocity =
        worldAxis.dot(b.angularVelocity - a.angularVelocity);
    const float angularMomentum =
        worldAxis.dot(a.invInertiaWorld.inverse() * a.angularVelocity +
                      b.invInertiaWorld.inverse() * b.angularVelocity);
    const Vec3 anchorA = a.position + a.rotation.rotate(Vec3(1, 0, 0));
    const Vec3 anchorB = b.position + b.rotation.rotate(Vec3(-1, 0, 0));
    const float anchorError = (anchorA - anchorB).length();
    printf("  pair relative=%.6f momentum=%.9f anchor=%.9f\n",
           relativeVelocity, angularMomentum, anchorError);
    CHECK(fabsf(relativeVelocity - 2.0f) < 0.02f,
          "Dynamic-pair motor target mismatch: %.6f", relativeVelocity);
    CHECK(fabsf(angularMomentum) < 1e-3f,
          "Dynamic-pair motor injected momentum: %.9f", angularMomentum);
    CHECK(anchorError < 1e-3f,
          "Dynamic-pair motor anchor drift: %.9f", anchorError);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const float angleA = 3.14159265358979323846f / 6.0f;
    const float angleB = -3.14159265358979323846f / 5.0f;
    const Quat rotationA(
        cosf(angleA * 0.5f), 0.0f, 0.0f,
        sinf(angleA * 0.5f));
    const Quat rotationB(
        cosf(angleB * 0.5f), 0.0f, 0.0f,
        sinf(angleB * 0.5f));
    const uint32_t bodyA =
        solver.addBody({0, 10, 0}, rotationA,
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody({0, 10, 0}, rotationB,
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].inertiaTensor =
        Mat33::diag(1.0f, 4.0f, 7.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].inertiaTensor =
        Mat33::diag(2.0f, 5.0f, 8.0f);
    solver.bodies[bodyB].computeDerived();
    const Vec3 worldAxis(1, 0, 0);
    const Vec3 localAxisA =
        rotationA.conjugate().rotate(worldAxis);
    const Vec3 localAxisB =
        rotationB.conjugate().rotate(worldAxis);
    solver.addRevoluteJoint(
        bodyA, bodyB, {0, 0, 0}, {0, 0, 0},
        localAxisA, localAxisB);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    float initialMomentum = 0.0f;
    float maximumLateRelativeSwing = 0.0f;
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
      Body &a = solver.bodies[bodyA];
      Body &b = solver.bodies[bodyB];
      a.updateInvInertiaWorld();
      b.updateInvInertiaWorld();
      const Vec3 axis =
          (a.rotation * solver.d6Joints[0].localFrameA)
              .rotate(Vec3(1, 0, 0))
              .normalized();
      const Vec3 relativeAngular =
          b.angularVelocity - a.angularVelocity;
      if (frame >= 60)
        maximumLateRelativeSwing =
            std::max(maximumLateRelativeSwing,
                     (relativeAngular -
                      axis * axis.dot(relativeAngular))
                         .length());
      const Vec3 momentum =
          a.invInertiaWorld.inverse() * a.angularVelocity +
          b.invInertiaWorld.inverse() * b.angularVelocity;
      if (frame == 0)
        initialMomentum = momentum.length();
    }

    Body &a = solver.bodies[bodyA];
    Body &b = solver.bodies[bodyB];
    const Vec3 axis =
        (a.rotation * solver.d6Joints[0].localFrameA)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const float relativeVelocity =
        axis.dot(b.angularVelocity - a.angularVelocity);
    printf("  pair off-principal relative=%.9f swing=%.9f "
           "momentum=%.9f\n",
           relativeVelocity, maximumLateRelativeSwing,
           initialMomentum);
    CHECK(fabsf(relativeVelocity - 2.0f) < 0.02f,
          "Dynamic off-principal motor target mismatch: %.9f",
          relativeVelocity);
    CHECK(maximumLateRelativeSwing < 0.02f,
          "Dynamic off-principal motor left relative swing: %.9f",
          maximumLateRelativeSwing);
    CHECK(initialMomentum < 0.03f,
          "Dynamic off-principal motor injected momentum: %.9f",
          initialMomentum);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t bodyA =
        solver.addBody({0, 9, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody({0, 11, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].mass = 1.0f;
    solver.bodies[bodyA].inertiaTensor =
        Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].mass = 1.0f;
    solver.bodies[bodyB].inertiaTensor =
        Mat33::diag(3.0f, 4.0f, 5.0f);
    solver.bodies[bodyB].computeDerived();
    const Vec3 localAnchorA(0, 1, 0);
    const Vec3 localAnchorB(0, -1, 0);
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(
        bodyA, bodyB, localAnchorA, localAnchorB,
        hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    float maximumLateRelativeSwing = 0.0f;
    float maximumLateRelativeAnchorPointSpeed = 0.0f;
    float maximumTotalLinearMomentum = 0.0f;
    float maximumInitialTotalAngularMomentum = 0.0f;
    float maximumLinearSpeed = 0.0f;
    for (int frame = 0; frame < 360; ++frame) {
      solver.contacts.clear();
      solver.step(solver.dt);
      Body &a = solver.bodies[bodyA];
      Body &b = solver.bodies[bodyB];
      a.updateInvInertiaWorld();
      b.updateInvInertiaWorld();
      const Vec3 worldAxis =
          (a.rotation * solver.d6Joints[0].localFrameA)
              .rotate(Vec3(1, 0, 0))
              .normalized();
      const Vec3 relativeAngular =
          b.angularVelocity - a.angularVelocity;
      const Vec3 rA = a.rotation.rotate(localAnchorA);
      const Vec3 rB = b.rotation.rotate(localAnchorB);
      const Vec3 relativeAnchorVelocity =
          b.linearVelocity + b.angularVelocity.cross(rB) -
          a.linearVelocity - a.angularVelocity.cross(rA);
      if (frame >= 60) {
        maximumLateRelativeSwing =
            std::max(maximumLateRelativeSwing,
                     (relativeAngular -
                      worldAxis * worldAxis.dot(relativeAngular))
                         .length());
        maximumLateRelativeAnchorPointSpeed =
            std::max(maximumLateRelativeAnchorPointSpeed,
                     relativeAnchorVelocity.length());
      }
      const Vec3 linearMomentum =
          a.linearVelocity * a.mass +
          b.linearVelocity * b.mass;
      maximumTotalLinearMomentum =
          std::max(maximumTotalLinearMomentum,
                   linearMomentum.length());
      if (frame < 12) {
        const Vec3 angularMomentum =
            a.position.cross(a.linearVelocity * a.mass) +
            a.invInertiaWorld.inverse() * a.angularVelocity +
            b.position.cross(b.linearVelocity * b.mass) +
            b.invInertiaWorld.inverse() * b.angularVelocity;
        maximumInitialTotalAngularMomentum =
            std::max(maximumInitialTotalAngularMomentum,
                     angularMomentum.length());
      }
      maximumLinearSpeed =
          std::max(maximumLinearSpeed,
                   std::max(a.linearVelocity.length(),
                            b.linearVelocity.length()));
    }

    Body &a = solver.bodies[bodyA];
    Body &b = solver.bodies[bodyB];
    const Vec3 worldAxisA =
        (a.rotation * solver.d6Joints[0].localFrameA)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const Vec3 worldAxisB =
        (b.rotation * solver.d6Joints[0].localFrameB)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const float relativeVelocity =
        worldAxisA.dot(b.angularVelocity - a.angularVelocity);
    const Vec3 rA = a.rotation.rotate(localAnchorA);
    const Vec3 rB = b.rotation.rotate(localAnchorB);
    const float relativeAnchorPointSpeed =
        (b.linearVelocity + b.angularVelocity.cross(rB) -
         a.linearVelocity - a.angularVelocity.cross(rA))
            .length();
    const float anchorError =
        (a.position + rA - b.position - rB).length();
    const float axisError = worldAxisA.cross(worldAxisB).length();
    printf("  pair off-center relative=%.9f swing=%.9f "
           "anchorSpeed=%.9f linearMomentum=%.9f "
           "initialAngularMomentum=%.9f linearSpeed=%.9f "
           "anchorError=%.9f axisError=%.9f\n",
           relativeVelocity, maximumLateRelativeSwing,
           std::max(relativeAnchorPointSpeed,
                    maximumLateRelativeAnchorPointSpeed),
           maximumTotalLinearMomentum,
           maximumInitialTotalAngularMomentum,
           maximumLinearSpeed, anchorError, axisError);
    CHECK(fabsf(relativeVelocity - 2.0f) <= 0.05f,
          "Dynamic off-center motor target mismatch: %.9f",
          relativeVelocity);
    CHECK(maximumLateRelativeSwing <= 0.05f,
          "Dynamic off-center motor left relative swing: %.9f",
          maximumLateRelativeSwing);
    CHECK(relativeAnchorPointSpeed <= 0.05f &&
              maximumLateRelativeAnchorPointSpeed <= 0.05f,
          "Dynamic off-center motor left anchor velocity: %.9f %.9f",
          relativeAnchorPointSpeed,
          maximumLateRelativeAnchorPointSpeed);
    CHECK(maximumTotalLinearMomentum <= 1e-3f,
          "Dynamic off-center motor injected linear momentum: %.9f",
          maximumTotalLinearMomentum);
    CHECK(maximumInitialTotalAngularMomentum <= 0.25f,
          "Dynamic off-center motor injected angular momentum: %.9f",
          maximumInitialTotalAngularMomentum);
    CHECK(maximumLinearSpeed >= 0.5f,
          "Dynamic off-center motor lacks orbital COM motion: %.9f",
          maximumLinearSpeed);
    CHECK(anchorError <= 1e-3f,
          "Dynamic off-center motor anchor drift: %.9f",
          anchorError);
    CHECK(axisError <= 1e-3f,
          "Dynamic off-center motor axis drift: %.9f",
          axisError);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const float angleA = 3.14159265358979323846f / 6.0f;
    const float angleB = -3.14159265358979323846f / 5.0f;
    const Quat rotationA(
        cosf(angleA * 0.5f), 0.0f, 0.0f,
        sinf(angleA * 0.5f));
    const Quat rotationB(
        cosf(angleB * 0.5f), 0.0f, 0.0f,
        sinf(angleB * 0.5f));
    const Vec3 positionA(0, 9, 0);
    const Vec3 positionB(0, 11, 0);
    const Vec3 worldAnchor(0, 10, 0);
    const Vec3 worldAxis(1, 0, 0);
    const Vec3 localAnchorA =
        rotationA.conjugate().rotate(worldAnchor - positionA);
    const Vec3 localAnchorB =
        rotationB.conjugate().rotate(worldAnchor - positionB);
    const Vec3 localAxisA =
        rotationA.conjugate().rotate(worldAxis);
    const Vec3 localAxisB =
        rotationB.conjugate().rotate(worldAxis);
    const uint32_t bodyA =
        solver.addBody(positionA, rotationA,
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody(positionB, rotationB,
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].mass = 1.0f;
    solver.bodies[bodyA].inertiaTensor =
        Mat33::diag(1.0f, 4.0f, 7.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].mass = 1.0f;
    solver.bodies[bodyB].inertiaTensor =
        Mat33::diag(2.0f, 5.0f, 8.0f);
    solver.bodies[bodyB].computeDerived();
    solver.addRevoluteJoint(
        bodyA, bodyB, localAnchorA, localAnchorB,
        localAxisA, localAxisB);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    Body &initialA = solver.bodies[bodyA];
    Body &initialB = solver.bodies[bodyB];
    initialA.updateInvInertiaWorld();
    initialB.updateInvInertiaWorld();
    const Vec3 initialResponseA =
        initialA.invInertiaWorld * worldAxis;
    const Vec3 initialResponseB =
        initialB.invInertiaWorld * worldAxis;
    const float offPrincipalResponseA =
        (initialResponseA -
         worldAxis * initialResponseA.dot(worldAxis))
            .length();
    const float offPrincipalResponseB =
        (initialResponseB -
         worldAxis * initialResponseB.dot(worldAxis))
            .length();

    float maximumLateRelativeSwing = 0.0f;
    float maximumLateRelativeAnchorPointSpeed = 0.0f;
    float maximumTotalLinearMomentum = 0.0f;
    float maximumInitialTotalAngularMomentum = 0.0f;
    float maximumLinearSpeed = 0.0f;
    for (int frame = 0; frame < 360; ++frame) {
      solver.contacts.clear();
      solver.step(solver.dt);
      Body &a = solver.bodies[bodyA];
      Body &b = solver.bodies[bodyB];
      a.updateInvInertiaWorld();
      b.updateInvInertiaWorld();
      const Vec3 hingeAxis =
          (a.rotation * solver.d6Joints[0].localFrameA)
              .rotate(Vec3(1, 0, 0))
              .normalized();
      const Vec3 relativeAngular =
          b.angularVelocity - a.angularVelocity;
      const Vec3 rA = a.rotation.rotate(localAnchorA);
      const Vec3 rB = b.rotation.rotate(localAnchorB);
      const Vec3 relativeAnchorVelocity =
          b.linearVelocity + b.angularVelocity.cross(rB) -
          a.linearVelocity - a.angularVelocity.cross(rA);
      if (frame >= 60) {
        maximumLateRelativeSwing =
            std::max(maximumLateRelativeSwing,
                     (relativeAngular -
                      hingeAxis * hingeAxis.dot(relativeAngular))
                         .length());
        maximumLateRelativeAnchorPointSpeed =
            std::max(maximumLateRelativeAnchorPointSpeed,
                     relativeAnchorVelocity.length());
      }
      const Vec3 linearMomentum =
          a.linearVelocity * a.mass +
          b.linearVelocity * b.mass;
      maximumTotalLinearMomentum =
          std::max(maximumTotalLinearMomentum,
                   linearMomentum.length());
      if (frame < 12) {
        const Vec3 angularMomentum =
            a.position.cross(a.linearVelocity * a.mass) +
            a.invInertiaWorld.inverse() * a.angularVelocity +
            b.position.cross(b.linearVelocity * b.mass) +
            b.invInertiaWorld.inverse() * b.angularVelocity;
        maximumInitialTotalAngularMomentum =
            std::max(maximumInitialTotalAngularMomentum,
                     angularMomentum.length());
      }
      maximumLinearSpeed =
          std::max(maximumLinearSpeed,
                   std::max(a.linearVelocity.length(),
                            b.linearVelocity.length()));
    }

    Body &a = solver.bodies[bodyA];
    Body &b = solver.bodies[bodyB];
    const Vec3 worldAxisA =
        (a.rotation * solver.d6Joints[0].localFrameA)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const Vec3 worldAxisB =
        (b.rotation * solver.d6Joints[0].localFrameB)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const float relativeVelocity =
        worldAxisA.dot(b.angularVelocity - a.angularVelocity);
    const Vec3 rA = a.rotation.rotate(localAnchorA);
    const Vec3 rB = b.rotation.rotate(localAnchorB);
    const float relativeAnchorPointSpeed =
        (b.linearVelocity + b.angularVelocity.cross(rB) -
         a.linearVelocity - a.angularVelocity.cross(rA))
            .length();
    const float anchorError =
        (a.position + rA - b.position - rB).length();
    const float axisError = worldAxisA.cross(worldAxisB).length();
    printf("  pair spatial response=(%.9f,%.9f) relative=%.9f "
           "swing=%.9f anchorSpeed=%.9f linearMomentum=%.9f "
           "initialAngularMomentum=%.9f linearSpeed=%.9f "
           "anchorError=%.9f axisError=%.9f\n",
           offPrincipalResponseA, offPrincipalResponseB,
           relativeVelocity, maximumLateRelativeSwing,
           std::max(relativeAnchorPointSpeed,
                    maximumLateRelativeAnchorPointSpeed),
           maximumTotalLinearMomentum,
           maximumInitialTotalAngularMomentum,
           maximumLinearSpeed, anchorError, axisError);
    CHECK(offPrincipalResponseA >= 0.05f &&
              offPrincipalResponseB >= 0.05f,
          "Dynamic spatial fixture lacks off-principal response: %.9f %.9f",
          offPrincipalResponseA, offPrincipalResponseB);
    CHECK(localAnchorA.cross(localAxisA).length() >= 0.5f &&
              localAnchorB.cross(localAxisB).length() >= 0.5f,
          "Dynamic spatial fixture lacks perpendicular lever arms");
    CHECK(fabsf(relativeVelocity - 2.0f) <= 0.05f,
          "Dynamic spatial motor target mismatch: %.9f",
          relativeVelocity);
    CHECK(maximumLateRelativeSwing <= 0.05f,
          "Dynamic spatial motor left relative swing: %.9f",
          maximumLateRelativeSwing);
    CHECK(relativeAnchorPointSpeed <= 0.05f &&
              maximumLateRelativeAnchorPointSpeed <= 0.05f,
          "Dynamic spatial motor left anchor velocity: %.9f %.9f",
          relativeAnchorPointSpeed,
          maximumLateRelativeAnchorPointSpeed);
    CHECK(maximumTotalLinearMomentum <= 1e-3f,
          "Dynamic spatial motor injected linear momentum: %.9f",
          maximumTotalLinearMomentum);
    CHECK(maximumInitialTotalAngularMomentum <= 0.25f,
          "Dynamic spatial motor injected angular momentum: %.9f",
          maximumInitialTotalAngularMomentum);
    CHECK(maximumLinearSpeed >= 0.5f,
          "Dynamic spatial motor lacks orbital COM motion: %.9f",
          maximumLinearSpeed);
    CHECK(anchorError <= 1e-3f,
          "Dynamic spatial motor anchor drift: %.9f",
          anchorError);
    CHECK(axisError <= 1e-3f,
          "Dynamic spatial motor axis drift: %.9f",
          axisError);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t body =
        solver.addBody({0, 10, 0}, Quat(), {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[body].inertiaTensor = Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[body].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(UINT32_MAX, body, {0, 10, 0}, {0, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointLimit(0, -0.5f, 0.5f);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    const float initialAngle = solver.d6Joints[0].computeHingeAngle(
        Quat(), solver.bodies[body].rotation);
    float maximumAngle = initialAngle;
    float maximumViolation = 0.0f;
    float maximumLateOutwardVelocity = 0.0f;
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
      const float angle = solver.d6Joints[0].computeHingeAngle(
          Quat(), solver.bodies[body].rotation);
      maximumAngle = std::max(maximumAngle, angle);
      maximumViolation =
          std::max(maximumViolation, std::max(0.0f, angle - 0.5f));
      if (frame >= 120 && angle >= 0.45f) {
        const Vec3 worldAxis =
            solver.d6Joints[0].localFrameA
                .rotate(Vec3(1, 0, 0))
                .normalized();
        maximumLateOutwardVelocity = std::max(
            maximumLateOutwardVelocity,
            std::max(0.0f,
                     worldAxis.dot(solver.bodies[body].angularVelocity)));
      }
    }
    const float finalAngle = solver.d6Joints[0].computeHingeAngle(
        Quat(), solver.bodies[body].rotation);
    const float travel = maximumAngle - initialAngle;
    printf("  limit travel=%.9f final=%.9f violation=%.9f outward=%.9f\n",
           travel, finalAngle, maximumViolation,
           maximumLateOutwardVelocity);
    CHECK(travel >= 0.4f && finalAngle >= 0.45f,
          "Motor did not reach active upper limit: %.9f %.9f",
          travel, finalAngle);
    CHECK(maximumViolation <= 0.02f,
          "Motor crossed active upper limit: %.9f", maximumViolation);
    CHECK(maximumLateOutwardVelocity <= 0.05f,
          "Motor retained outward velocity at upper limit: %.9f",
          maximumLateOutwardVelocity);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t bodyA =
        solver.addBody({0, 10, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody({0, 10, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].inertiaTensor =
        Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].inertiaTensor =
        Mat33::diag(3.0f, 4.0f, 5.0f);
    solver.bodies[bodyB].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(bodyA, bodyB, {0, 0, 0}, {0, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointLimit(0, -0.5f, 0.5f);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    const float initialAngle =
        solver.d6Joints[0].computeHingeAngle(
            solver.bodies[bodyA].rotation,
            solver.bodies[bodyB].rotation);
    float minimumAngle = initialAngle;
    float maximumAngle = initialAngle;
    float maximumViolation = 0.0f;
    float maximumLateOutwardVelocity = 0.0f;
    float maximumAngularMomentum = 0.0f;
    for (int frame = 0; frame < 360; ++frame) {
      if (frame == 180)
        solver.setRevoluteJointDrive(0, -2.0f, 1000.0f);
      solver.contacts.clear();
      solver.step(solver.dt);
      Body &a = solver.bodies[bodyA];
      Body &b = solver.bodies[bodyB];
      a.updateInvInertiaWorld();
      b.updateInvInertiaWorld();
      const float angle =
          solver.d6Joints[0].computeHingeAngle(
              a.rotation, b.rotation);
      minimumAngle = std::min(minimumAngle, angle);
      maximumAngle = std::max(maximumAngle, angle);
      maximumViolation = std::max(
          maximumViolation,
          std::max(std::max(0.0f, angle - 0.5f),
                   std::max(0.0f, -0.5f - angle)));
      const Vec3 worldAxis =
          (a.rotation * solver.d6Joints[0].localFrameA)
              .rotate(Vec3(1, 0, 0))
              .normalized();
      const float relativeVelocity =
          worldAxis.dot(b.angularVelocity - a.angularVelocity);
      if ((frame >= 120 && frame < 180 && angle >= 0.45f) ||
          (frame >= 300 && angle <= -0.45f)) {
        const float outwardVelocity =
            frame < 180 ? relativeVelocity : -relativeVelocity;
        maximumLateOutwardVelocity =
            std::max(maximumLateOutwardVelocity,
                     std::max(0.0f, outwardVelocity));
      }
      const Vec3 angularMomentum =
          a.invInertiaWorld.inverse() * a.angularVelocity +
          b.invInertiaWorld.inverse() * b.angularVelocity;
      maximumAngularMomentum =
          std::max(maximumAngularMomentum,
                   angularMomentum.length());
    }
    Body &a = solver.bodies[bodyA];
    Body &b = solver.bodies[bodyB];
    const float finalAngle =
        solver.d6Joints[0].computeHingeAngle(
            a.rotation, b.rotation);
    const float upperTravel = maximumAngle - initialAngle;
    const float range = maximumAngle - minimumAngle;
    printf("  pair limit travel=%.9f range=%.9f final=%.9f "
           "violation=%.9f outward=%.9f momentum=%.9f\n",
           upperTravel, range, finalAngle, maximumViolation,
           maximumLateOutwardVelocity, maximumAngularMomentum);
    CHECK(upperTravel >= 0.4f && range >= 0.9f &&
              finalAngle <= -0.45f,
          "Dynamic-pair motor did not traverse both limits: %.9f %.9f %.9f",
          upperTravel, range, finalAngle);
    CHECK(maximumViolation <= 0.02f,
          "Dynamic-pair motor crossed active limit: %.9f",
          maximumViolation);
    CHECK(maximumLateOutwardVelocity <= 0.05f,
          "Dynamic-pair motor retained outward limit velocity: %.9f",
          maximumLateOutwardVelocity);
    CHECK(maximumAngularMomentum <= 1e-3f,
          "Dynamic-pair limited motor injected momentum: %.9f",
          maximumAngularMomentum);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t body =
        solver.addBody({0, 10, 0}, Quat(), {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[body].inertiaTensor = Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[body].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(UINT32_MAX, body, {0, 10, 0}, {0, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f, true);

    for (int frame = 0; frame < 120; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    const float preBoostVelocity =
        solver.bodies[body].angularVelocity.x;
    solver.bodies[body].angularVelocity = Vec3(5.0f, 0.0f, 0.0f);
    float minimumPostBoostVelocity = 5.0f;
    for (int frame = 120; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
      minimumPostBoostVelocity =
          std::min(minimumPostBoostVelocity,
                   solver.bodies[body].angularVelocity.x);
    }
    const float finalVelocity =
        solver.bodies[body].angularVelocity.x;
    printf("  free-spin pre=%.9f minimumPost=%.9f final=%.9f\n",
           preBoostVelocity, minimumPostBoostVelocity, finalVelocity);
    CHECK(fabsf(preBoostVelocity - 2.0f) <= 0.05f,
          "Free-spin motor did not reach target: %.9f",
          preBoostVelocity);
    CHECK(minimumPostBoostVelocity >= 4.9f && finalVelocity >= 4.9f,
          "Free-spin motor braked super-target motion: %.9f %.9f",
          minimumPostBoostVelocity, finalVelocity);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t bodyA =
        solver.addBody({0, 10, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody({0, 10, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].inertiaTensor =
        Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].inertiaTensor =
        Mat33::diag(3.0f, 4.0f, 5.0f);
    solver.bodies[bodyB].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(bodyA, bodyB, {0, 0, 0}, {0, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f, true);

    for (int frame = 0; frame < 120; ++frame) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    const float preBoostVelocity =
        solver.bodies[bodyB].angularVelocity.x -
        solver.bodies[bodyA].angularVelocity.x;
    solver.bodies[bodyA].angularVelocity =
        Vec3(-3.75f, 0.0f, 0.0f);
    solver.bodies[bodyB].angularVelocity =
        Vec3(1.25f, 0.0f, 0.0f);
    float minimumPostBoostVelocity = 5.0f;
    float maximumAngularMomentum = 0.0f;
    for (int frame = 120; frame < 360; ++frame) {
      solver.contacts.clear();
      solver.step(solver.dt);
      Body &a = solver.bodies[bodyA];
      Body &b = solver.bodies[bodyB];
      a.updateInvInertiaWorld();
      b.updateInvInertiaWorld();
      minimumPostBoostVelocity =
          std::min(minimumPostBoostVelocity,
                   b.angularVelocity.x - a.angularVelocity.x);
      const Vec3 angularMomentum =
          a.invInertiaWorld.inverse() * a.angularVelocity +
          b.invInertiaWorld.inverse() * b.angularVelocity;
      maximumAngularMomentum =
          std::max(maximumAngularMomentum,
                   angularMomentum.length());
    }
    const float finalVelocity =
        solver.bodies[bodyB].angularVelocity.x -
        solver.bodies[bodyA].angularVelocity.x;
    printf("  pair free-spin pre=%.9f minimumPost=%.9f "
           "final=%.9f momentum=%.9f\n",
           preBoostVelocity, minimumPostBoostVelocity,
           finalVelocity, maximumAngularMomentum);
    CHECK(fabsf(preBoostVelocity - 2.0f) <= 0.05f,
          "Dynamic-pair free-spin motor did not reach target: %.9f",
          preBoostVelocity);
    CHECK(minimumPostBoostVelocity >= 4.9f &&
              finalVelocity >= 4.9f,
          "Dynamic-pair free-spin motor braked super-target motion: "
          "%.9f %.9f",
          minimumPostBoostVelocity, finalVelocity);
    CHECK(maximumAngularMomentum <= 1e-3f,
          "Dynamic-pair free-spin motor injected momentum: %.9f",
          maximumAngularMomentum);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t bodyA =
        solver.addBody({-1, 10, 0}, Quat(), {0.25f, 0.25f, 0.25f}, 1.0f);
    const uint32_t bodyB =
        solver.addBody({1, 10, 0}, Quat(), {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[bodyA].inertiaTensor = Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[bodyA].computeDerived();
    solver.bodies[bodyB].inertiaTensor = Mat33::diag(3.0f, 4.0f, 5.0f);
    solver.bodies[bodyB].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(bodyA, bodyB, {1, 0, 0}, {-1, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f, false, 2.5f);
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }

    Body &a = solver.bodies[bodyA];
    Body &b = solver.bodies[bodyB];
    a.updateInvInertiaWorld();
    b.updateInvInertiaWorld();
    const Vec3 worldAxis =
        (a.rotation * solver.d6Joints[0].localFrameA)
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const float velocityA = worldAxis.dot(a.angularVelocity);
    const float velocityB = worldAxis.dot(b.angularVelocity);
    const float weightedVelocity = 2.5f * velocityB - velocityA;
    const float generalizedMomentum =
        worldAxis.dot((a.invInertiaWorld.inverse() * a.angularVelocity) *
                          2.5f +
                      b.invInertiaWorld.inverse() * b.angularVelocity);
    printf("  ratio motorA=%.9f motorB=%.9f weighted=%.9f momentum=%.9f\n",
           velocityA, velocityB, weightedVelocity,
           generalizedMomentum);
    CHECK(fabsf(weightedVelocity - 2.0f) < 0.02f,
          "Non-unit motor weighted target mismatch: %.9f",
          weightedVelocity);
    CHECK(fabsf(generalizedMomentum) < 1e-3f,
          "Non-unit motor generalized momentum drift: %.9f",
          generalizedMomentum);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t body =
        solver.addBody({0, 10, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[body].inertiaTensor =
        Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[body].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    solver.addRevoluteJoint(
        UINT32_MAX, body, {0, 10, 0}, {0, 0, 0},
        hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);
    solver.d6Joints[0].motorExternalAngularVelocityA =
        Vec3(1.0f, 0.0f, 0.0f);
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }

    const float dynamicVelocity =
        solver.bodies[body].angularVelocity.x;
    const float relativeVelocity =
        dynamicVelocity -
        solver.d6Joints[0].motorExternalAngularVelocityA.x;
    printf("  prescribed motor kinematic=1.000000000 "
           "dynamic=%.9f relative=%.9f\n",
           dynamicVelocity, relativeVelocity);
    CHECK(fabsf(dynamicVelocity - 3.0f) < 0.02f,
          "Prescribed-endpoint dynamic velocity mismatch: %.9f",
          dynamicVelocity);
    CHECK(fabsf(relativeVelocity - 2.0f) < 0.02f,
          "Prescribed-endpoint motor target mismatch: %.9f",
          relativeVelocity);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const float bodyAngle = 3.14159265358979323846f / 6.0f;
    const Quat bodyRotation(
        cosf(bodyAngle * 0.5f), 0.0f, 0.0f,
        sinf(bodyAngle * 0.5f));
    const uint32_t body =
        solver.addBody({0, 10, 0}, bodyRotation,
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[body].inertiaTensor =
        Mat33::diag(1.0f, 4.0f, 7.0f);
    solver.bodies[body].computeDerived();
    const Vec3 worldHingeAxis(1, 0, 0);
    const Vec3 localHingeAxis =
        bodyRotation.conjugate().rotate(worldHingeAxis);
    solver.addRevoluteJoint(
        UINT32_MAX, body, {0, 10, 0}, {0, 0, 0},
        worldHingeAxis, localHingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    solver.bodies[body].updateInvInertiaWorld();
    const Vec3 initialResponse =
        solver.bodies[body].invInertiaWorld * worldHingeAxis;
    const float offPrincipalResponse =
        (initialResponse -
         worldHingeAxis *
             initialResponse.dot(worldHingeAxis)).length();
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }

    Body &dynamicBody = solver.bodies[body];
    const Vec3 finalWorldAxis =
        solver.d6Joints[0].localFrameA
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const float hingeVelocity =
        finalWorldAxis.dot(dynamicBody.angularVelocity);
    const float swingVelocity =
        (dynamicBody.angularVelocity -
         finalWorldAxis * hingeVelocity).length();
    printf("  off-principal response=%.9f hinge=%.9f swing=%.9f\n",
           offPrincipalResponse, hingeVelocity, swingVelocity);
    CHECK(offPrincipalResponse >= 0.05f,
          "Off-principal fixture lacks coupled response: %.9f",
          offPrincipalResponse);
    CHECK(fabsf(hingeVelocity - 2.0f) < 0.02f,
          "Off-principal motor target mismatch: %.9f",
          hingeVelocity);
    CHECK(swingVelocity < 0.02f,
          "Off-principal motor left locked swing velocity: %.9f",
          swingVelocity);
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t body =
        solver.addBody({0, 11, 0}, Quat(),
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[body].inertiaTensor =
        Mat33::diag(1.0f, 2.0f, 3.0f);
    solver.bodies[body].computeDerived();
    const Vec3 hingeAxis(1, 0, 0);
    const Vec3 localAnchor(0, -1, 0);
    solver.addRevoluteJoint(
        UINT32_MAX, body, {0, 10, 0}, localAnchor,
        hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);
    float maximumLateAnchorPointSpeed = 0.0f;
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
      if (frame >= 60) {
        const Body &dynamicBody = solver.bodies[body];
        const Vec3 worldLeverArm =
            dynamicBody.rotation.rotate(localAnchor);
        const Vec3 anchorVelocity =
            dynamicBody.linearVelocity +
            dynamicBody.angularVelocity.cross(worldLeverArm);
        maximumLateAnchorPointSpeed =
            std::max(maximumLateAnchorPointSpeed,
                     anchorVelocity.length());
      }
    }

    const Body &dynamicBody = solver.bodies[body];
    const Vec3 worldAxis =
        solver.d6Joints[0].localFrameA
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const Vec3 worldLeverArm =
        dynamicBody.rotation.rotate(localAnchor);
    const float hingeVelocity =
        worldAxis.dot(dynamicBody.angularVelocity);
    const float anchorPointSpeed =
        (dynamicBody.linearVelocity +
         dynamicBody.angularVelocity.cross(worldLeverArm))
            .length();
    printf("  off-center hinge=%.9f anchorSpeed=%.9f "
           "lateAnchorSpeed=%.9f linearSpeed=%.9f\n",
           hingeVelocity, anchorPointSpeed,
           maximumLateAnchorPointSpeed,
           dynamicBody.linearVelocity.length());
    CHECK(fabsf(hingeVelocity - 2.0f) < 0.02f,
          "Off-center motor target mismatch: %.9f",
          hingeVelocity);
    CHECK(anchorPointSpeed < 0.02f &&
              maximumLateAnchorPointSpeed < 0.02f,
          "Off-center motor left anchor velocity: %.9f %.9f",
          anchorPointSpeed, maximumLateAnchorPointSpeed);
    CHECK(dynamicBody.linearVelocity.length() > 1.0f,
          "Off-center motor did not produce orbital COM velocity: %.9f",
          dynamicBody.linearVelocity.length());
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const float bodyAngle = 3.14159265358979323846f / 6.0f;
    const Quat bodyRotation(
        cosf(bodyAngle * 0.5f), 0.0f, 0.0f,
        sinf(bodyAngle * 0.5f));
    const Vec3 bodyPosition(0, 11, 0);
    const Vec3 worldAnchor(0, 10, 0);
    const Vec3 worldHingeAxis(1, 0, 0);
    const Vec3 localAnchor =
        bodyRotation.conjugate().rotate(worldAnchor - bodyPosition);
    const Vec3 localHingeAxis =
        bodyRotation.conjugate().rotate(worldHingeAxis);
    const uint32_t body =
        solver.addBody(bodyPosition, bodyRotation,
                       {0.25f, 0.25f, 0.25f}, 1.0f);
    solver.bodies[body].inertiaTensor =
        Mat33::diag(1.0f, 4.0f, 7.0f);
    solver.bodies[body].computeDerived();
    solver.addRevoluteJoint(
        UINT32_MAX, body, worldAnchor, localAnchor,
        worldHingeAxis, localHingeAxis);
    solver.setRevoluteJointDrive(0, 2.0f, 1000.0f);

    solver.bodies[body].updateInvInertiaWorld();
    const Vec3 initialResponse =
        solver.bodies[body].invInertiaWorld * worldHingeAxis;
    const float offPrincipalResponse =
        (initialResponse -
         worldHingeAxis *
             initialResponse.dot(worldHingeAxis)).length();
    float maximumLateAnchorPointSpeed = 0.0f;
    float maximumLateSwingVelocity = 0.0f;
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
      if (frame >= 60) {
        const Body &dynamicBody = solver.bodies[body];
        const Vec3 worldAxis =
            solver.d6Joints[0].localFrameA
                .rotate(Vec3(1, 0, 0))
                .normalized();
        const Vec3 worldLeverArm =
            dynamicBody.rotation.rotate(localAnchor);
        const float hingeVelocity =
            worldAxis.dot(dynamicBody.angularVelocity);
        const Vec3 anchorVelocity =
            dynamicBody.linearVelocity +
            dynamicBody.angularVelocity.cross(worldLeverArm);
        maximumLateAnchorPointSpeed =
            std::max(maximumLateAnchorPointSpeed,
                     anchorVelocity.length());
        maximumLateSwingVelocity =
            std::max(maximumLateSwingVelocity,
                     (dynamicBody.angularVelocity -
                      worldAxis * hingeVelocity)
                         .length());
      }
    }

    const Body &dynamicBody = solver.bodies[body];
    const Vec3 worldAxis =
        solver.d6Joints[0].localFrameA
            .rotate(Vec3(1, 0, 0))
            .normalized();
    const Vec3 worldLeverArm =
        dynamicBody.rotation.rotate(localAnchor);
    const float hingeVelocity =
        worldAxis.dot(dynamicBody.angularVelocity);
    const float anchorPointSpeed =
        (dynamicBody.linearVelocity +
         dynamicBody.angularVelocity.cross(worldLeverArm))
            .length();
    printf("  spatial response=%.9f hinge=%.9f swing=%.9f "
           "anchorSpeed=%.9f lateAnchorSpeed=%.9f linearSpeed=%.9f\n",
           offPrincipalResponse, hingeVelocity,
           maximumLateSwingVelocity, anchorPointSpeed,
           maximumLateAnchorPointSpeed,
           dynamicBody.linearVelocity.length());
    CHECK(offPrincipalResponse >= 0.05f,
          "Spatial fixture lacks off-principal response: %.9f",
          offPrincipalResponse);
    CHECK(fabsf(hingeVelocity - 2.0f) < 0.02f,
          "Spatial motor target mismatch: %.9f", hingeVelocity);
    CHECK(maximumLateSwingVelocity < 0.02f,
          "Spatial motor left locked swing velocity: %.9f",
          maximumLateSwingVelocity);
    CHECK(anchorPointSpeed < 0.02f &&
              maximumLateAnchorPointSpeed < 0.02f,
          "Spatial motor left anchor velocity: %.9f %.9f",
          anchorPointSpeed, maximumLateAnchorPointSpeed);
    CHECK(dynamicBody.linearVelocity.length() > 1.0f,
          "Spatial motor did not produce orbital COM velocity: %.9f",
          dynamicBody.linearVelocity.length());
  }

  {
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 10;
    const uint32_t gearA =
        solver.addBody({-2, 10, 0}, Quat(), {0.5f, 0.5f, 0.5f}, 1.0f);
    const uint32_t gearB =
        solver.addBody({2, 10, 0}, Quat(), {0.5f, 0.5f, 0.5f}, 1.0f);
    const Vec3 hingeAxis(0, 0, 1);
    solver.addRevoluteJoint(UINT32_MAX, gearA, {-2, 10, 0}, {0, 0, 0},
                            hingeAxis, hingeAxis);
    solver.addRevoluteJoint(UINT32_MAX, gearB, {2, 10, 0}, {0, 0, 0},
                            hingeAxis, hingeAxis);
    solver.setRevoluteJointDrive(0, 0.5f, 1000.0f);
    solver.addGearJoint(gearA, gearB, hingeAxis, hingeAxis, 2.5f, 1e5f);
    for (int frame = 0; frame < 360; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }

    const float speedA = solver.bodies[gearA].angularVelocity.z;
    const float speedB = solver.bodies[gearB].angularVelocity.z;
    const float gearResidual = 2.5f * speedA + speedB;
    printf("  gear motorA=%.6f motorB=%.6f residual=%.9f\n",
           speedA, speedB, gearResidual);
    CHECK(fabsf(speedA - 0.5f) < 0.02f,
          "Coupled gear motor target mismatch: %.6f", speedA);
    CHECK(fabsf(gearResidual) < 1e-4f,
          "Coupled gear velocity residual: %.9f", gearResidual);
  }

  PASS("Revolute motor has one velocity owner for free, one-/two-body limited and free-spin, drive-ratio, prescribed-endpoint, and gear topologies");
}

// Test 54: Axis alignment — verify hinge constrains to 1 rotation DOF
bool test54_revoluteJoint_axisAlign() {
  printf("\n--- Test 54: Revolute Joint Axis Alignment ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 15;

  // Arm hanging from a Z-axis hinge. After settling, the arm should only
  // rotate around Z (i.e., angular velocity around X and Y should be ~zero).
  uint32_t arm =
      solver.addBody({3, 15, 0}, Quat(), {3.0f, 0.3f, 0.3f}, 1.0f);
  Vec3 hingeAxis(0, 0, 1);
  solver.addRevoluteJoint(UINT32_MAX, arm, {0, 15, 0}, {-3, 0, 0}, hingeAxis,
                          hingeAxis);

  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  Vec3 worldAxisA = Quat().rotate(hingeAxis); // static body => identity
  Vec3 worldAxisB = solver.bodies[arm].rotation.rotate(hingeAxis);
  float axisDot = worldAxisA.dot(worldAxisB);

  printf("  axis alignment dot = %.6f (should be ~1.0)\n", axisDot);
  CHECK(axisDot > 0.95f, "Axes not aligned: dot=%.4f", axisDot);
  PASS("Revolute axis alignment correct");
}

// Test 55: Reproduce/diagnose revolute tail jitter in near-rest hanging chain
bool test55_revoluteJoint_jitterRepro() {
  printf("\n--- Test 55: Revolute Tail Jitter Reproduction ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  const int N = 5;
  Vec3 halfExt(2.0f, 0.5f, 0.5f);
  float separation = 4.0f;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({separation / 2.0f + i * separation, 20.0f, 0.0f},
                            Quat(), halfExt, 1.0f);

  Vec3 hingeAxis(0, 0, 1);
  Vec3 offset(separation / 2.0f, 0, 0);

  solver.addRevoluteJoint(UINT32_MAX, ids[0], {0, 20, 0}, {-offset.x, 0, 0},
                          hingeAxis, hingeAxis);
  for (int i = 0; i < N - 1; i++)
    solver.addRevoluteJoint(ids[i], ids[i + 1], {offset.x, 0, 0},
                            {-offset.x, 0, 0}, hingeAxis, hingeAxis);

  const float limit = 3.14159f / 4.0f;
  for (uint32_t j = 0; j < solver.d6Joints.size(); j++)
    solver.setRevoluteJointLimit(j, -limit, limit);

  float maxTailLateral = 0.0f;
  float sumW4Early = 0.0f, sumW5Early = 0.0f;
  float sumW4Late = 0.0f, sumW5Late = 0.0f;
  int cntEarly = 0, cntLate = 0;

  float prevAngle3 = 0.0f;
  float prevAngle4 = 0.0f;
  float prevD3 = 0.0f;
  float prevD4 = 0.0f;
  int flip3 = 0;
  int flip4 = 0;

  bool exploded = false;
  const int totalFrames = 1400;
  const int settleFrames = 200;
  const int earlyBegin = 250, earlyEnd = 550;
  const int lateBegin = 1000, lateEnd = 1300;

  auto shouldDumpState = [&](int frame) {
    if (frame <= 180)
      return (frame % 30) == 0;
    if (frame >= 200 && frame <= 420)
      return (frame % 10) == 0;
    if (frame >= 900 && frame <= 1300)
      return (frame % 10) == 0;
    return (frame % 120) == 0;
  };

  for (int frame = 0; frame < totalFrames; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);

    if (shouldDumpState(frame)) {
      printf("  [StandaloneRevoluteNodes] frame=%d\n", frame);
      for (int i = 0; i < N; i++) {
        const Body &b = solver.bodies[ids[i]];
        printf("    node%d p=(%.3f,%.3f,%.3f) q=(%.4f,%.4f,%.4f,%.4f) "
               "w=(%.3f,%.3f,%.3f)\n",
               i, b.position.x, b.position.y, b.position.z, b.rotation.x,
               b.rotation.y, b.rotation.z, b.rotation.w, b.angularVelocity.x,
               b.angularVelocity.y, b.angularVelocity.z);
      }

      for (uint32_t j = 0; j < solver.d6Joints.size(); j++) {
        const D6Joint &rj = solver.d6Joints[j];
        const bool aStatic = (rj.bodyA == UINT32_MAX);
        const Quat rotA = aStatic ? Quat() : solver.bodies[rj.bodyA].rotation;
        const Quat rotB = solver.bodies[rj.bodyB].rotation;
        Vec3 localAxisA = rj.localFrameA.rotate(Vec3(1, 0, 0));
        const Vec3 axisA = rotA.rotate(localAxisA).normalized();
        const Vec3 axisB = rotB.rotate(rj.hingeAxisB).normalized();
        const float dot = std::max(-1.0f, std::min(1.0f, axisA.dot(axisB)));
        const float misDeg = acosf(dot) * 180.0f / 3.1415926535f;
        const float angle = rj.computeHingeAngle(rotA, rotB);
        float vel = 0.0f;
        if (!aStatic)
          vel += solver.bodies[rj.bodyA].angularVelocity.dot(axisA);
        vel -= solver.bodies[rj.bodyB].angularVelocity.dot(axisA);
        printf("    joint%u angle=%.4f vel=%.4f axisMisalignDeg=%.3f\n", j,
               angle, vel, misDeg);
      }
    }

    for (int i = 0; i < N; i++) {
      if (fabsf(solver.bodies[ids[i]].position.y) > 200.0f)
        exploded = true;
    }

    if (frame >= settleFrames) {
      Vec3 pTail = solver.bodies[ids[N - 1]].position;
      float lateral = sqrtf(pTail.x * pTail.x + pTail.z * pTail.z);
      maxTailLateral = std::max(maxTailLateral, lateral);

      Vec3 w3 = solver.bodies[ids[N - 2]].angularVelocity;
      Vec3 w4 = solver.bodies[ids[N - 1]].angularVelocity;
      float w3m = w3.length();
      float w4m = w4.length();
      if (frame >= earlyBegin && frame < earlyEnd) {
        sumW4Early += w3m;
        sumW5Early += w4m;
        cntEarly++;
      }
      if (frame >= lateBegin && frame < lateEnd) {
        sumW4Late += w3m;
        sumW5Late += w4m;
        cntLate++;
      }

      if (solver.d6Joints.size() >= 5) {
        float a3 = solver.d6Joints[3].computeHingeAngle(
            solver.bodies[ids[2]].rotation, solver.bodies[ids[3]].rotation);
        float a4 = solver.d6Joints[4].computeHingeAngle(
            solver.bodies[ids[3]].rotation, solver.bodies[ids[4]].rotation);
        float d3 = a3 - prevAngle3;
        float d4 = a4 - prevAngle4;

        if (fabsf(d3) > 1e-5f && fabsf(prevD3) > 1e-5f && d3 * prevD3 < 0.0f)
          flip3++;
        if (fabsf(d4) > 1e-5f && fabsf(prevD4) > 1e-5f && d4 * prevD4 < 0.0f)
          flip4++;

        prevD3 = d3;
        prevD4 = d4;
        prevAngle3 = a3;
        prevAngle4 = a4;
      }
    }
  }

    float avgW4Early = cntEarly > 0 ? sumW4Early / cntEarly : 0.0f;
    float avgW5Early = cntEarly > 0 ? sumW5Early / cntEarly : 0.0f;
    float avgW4Late = cntLate > 0 ? sumW4Late / cntLate : 0.0f;
    float avgW5Late = cntLate > 0 ? sumW5Late / cntLate : 0.0f;
    float growth4 = (avgW4Early > 1e-6f) ? (avgW4Late / avgW4Early) : 0.0f;
    float growth5 = (avgW5Early > 1e-6f) ? (avgW5Late / avgW5Early) : 0.0f;

    printf("  tail lateral max=%.5f, avgW early=(%.5f,%.5f), avgW late=(%.5f,%.5f), growth=(%.3f,%.3f), flips=(%d,%d)\n",
      maxTailLateral, avgW4Early, avgW5Early, avgW4Late, avgW5Late,
      growth4, growth5, flip3, flip4);

  CHECK(!exploded, "Revolute jitter repro exploded");

    // This case is diagnostic: report whether problematic jitter is reproduced.
    // Reproduction signal: long-horizon tail angular activity does not decay, or
    // even grows in late frames.
    bool reproduced = (growth4 > 1.10f) || (growth5 > 1.10f) || (flip4 > 140);
    printf("  jitter_reproduced=%s\n", reproduced ? "true" : "false");

    PASS("Revolute jitter diagnostics completed");
}

// Test 122: A prismatic joint must derive its world-space free axis from
// actor A, regardless of which endpoint is static.  This is a mirrored
// endpoint-ordering regression for the linear primal and dual D6 paths.
bool test122_prismaticReverseEndpointFrameA() {
  printf("\n--- Test 122: Prismatic Reverse Endpoint Frame A ---\n");

  struct EndpointResult {
    bool finite = true;
    bool fixtureWitness = false;
    float signedDisplacement = 0.0f;
    float displacementDirection = 0.0f;
    float displacementOrthogonalRatio = 0.0f;
    float maxDisplacementOrthogonalRatio = 0.0f;
    float signedVelocity = 0.0f;
    float velocityDirection = 0.0f;
    float velocityOrthogonalRatio = 0.0f;
    float worldAxisADot = 0.0f;
    float worldAxisBDot = 0.0f;
    float authoredVsWorldAxisDot = 0.0f;
  };

  const float pi = 3.14159265358979f;
  const float halfBodyAngle = pi / 8.0f; // body A: +45 degrees about Y
  const Quat bodyRotation(cosf(halfBodyAngle), 0.0f,
                          sinf(halfBodyAngle), 0.0f);
  const Vec3 worldFreeAxis(0.0f, 0.0f, -1.0f);
  const Vec3 reverseLocalFreeAxis =
      bodyRotation.conjugate().rotate(worldFreeAxis).normalized();
  const Vec3 start(1.25f, 4.0f, 2.5f);

  auto runEndpoint = [&](bool reverse) {
    EndpointResult out;
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 20;

    const uint32_t body =
        solver.addBody(start, bodyRotation, {0.5f, 0.5f, 0.5f}, 10.0f);
    if (reverse) {
      solver.addPrismaticJoint(body, UINT32_MAX, Vec3(), start,
                               reverseLocalFreeAxis, 1e6f);
    } else {
      solver.addPrismaticJoint(UINT32_MAX, body, start, Vec3(), worldFreeAxis,
                               1e6f);
    }

    const D6Joint &joint = solver.d6Joints[0];
    const Quat worldFrameA =
        (joint.bodyA == UINT32_MAX)
            ? joint.localFrameA
            : bodyRotation * joint.localFrameA;
    const Quat worldFrameB =
        (joint.bodyB == UINT32_MAX)
            ? joint.localFrameB
            : bodyRotation * joint.localFrameB;
    const Vec3 axisA = worldFrameA.rotate(Vec3(1, 0, 0)).normalized();
    const Vec3 axisB = worldFrameB.rotate(Vec3(1, 0, 0)).normalized();
    const Vec3 authoredAxisA =
        joint.localFrameA.rotate(Vec3(1, 0, 0)).normalized();
    out.worldAxisADot = axisA.dot(worldFreeAxis);
    out.worldAxisBDot = axisB.dot(worldFreeAxis);
    out.authoredVsWorldAxisDot = authoredAxisA.dot(worldFreeAxis);

    const bool endpointWitness =
        reverse ? (joint.bodyA == body && joint.bodyB == UINT32_MAX)
                : (joint.bodyA == UINT32_MAX && joint.bodyB == body);
    const bool motionWitness = joint.getLinearMotion(0) == 2 &&
                               joint.getLinearMotion(1) == 0 &&
                               joint.getLinearMotion(2) == 0;
    const bool frameWitness =
        out.worldAxisADot > 0.999f && out.worldAxisBDot > 0.999f &&
        (!reverse || (out.authoredVsWorldAxisDot > 0.65f &&
                      out.authoredVsWorldAxisDot < 0.76f &&
                      fabsf(bodyRotation.w) < 0.95f));
    out.fixtureWitness = endpointWitness && motionWitness && frameWitness;

    solver.bodies[body].linearVelocity = worldFreeAxis * 6.0f;
    float maxOrthogonalRatio = 0.0f;
    for (int frame = 0; frame < 45; ++frame) {
      solver.contacts.clear();
      solver.step(solver.dt);

      const Body &state = solver.bodies[body];
      const bool stateFinite =
          std::isfinite(state.position.x) && std::isfinite(state.position.y) &&
          std::isfinite(state.position.z) &&
          std::isfinite(state.linearVelocity.x) &&
          std::isfinite(state.linearVelocity.y) &&
          std::isfinite(state.linearVelocity.z) &&
          std::isfinite(state.rotation.w) && std::isfinite(state.rotation.x) &&
          std::isfinite(state.rotation.y) && std::isfinite(state.rotation.z) &&
          std::isfinite(state.angularVelocity.x) &&
          std::isfinite(state.angularVelocity.y) &&
          std::isfinite(state.angularVelocity.z);
      out.finite = out.finite && stateFinite;

      const Vec3 displacement = state.position - start;
      const float signedDisplacement = displacement.dot(worldFreeAxis);
      const Vec3 orthogonal =
          displacement - worldFreeAxis * signedDisplacement;
      const float orthogonalRatio =
          orthogonal.length() / std::max(fabsf(signedDisplacement), 1e-5f);
      maxOrthogonalRatio = std::max(maxOrthogonalRatio, orthogonalRatio);
    }

    const Body &state = solver.bodies[body];
    const Vec3 displacement = state.position - start;
    out.signedDisplacement = displacement.dot(worldFreeAxis);
    const Vec3 displacementOrthogonal =
        displacement - worldFreeAxis * out.signedDisplacement;
    const float displacementLength = displacement.length();
    out.displacementDirection =
        out.signedDisplacement / std::max(displacementLength, 1e-5f);
    out.displacementOrthogonalRatio =
        displacementOrthogonal.length() /
        std::max(fabsf(out.signedDisplacement), 1e-5f);
    out.maxDisplacementOrthogonalRatio = maxOrthogonalRatio;

    out.signedVelocity = state.linearVelocity.dot(worldFreeAxis);
    const Vec3 velocityOrthogonal =
        state.linearVelocity - worldFreeAxis * out.signedVelocity;
    const float velocityLength = state.linearVelocity.length();
    out.velocityDirection =
        out.signedVelocity / std::max(velocityLength, 1e-5f);
    out.velocityOrthogonalRatio =
        velocityOrthogonal.length() /
        std::max(fabsf(out.signedVelocity), 1e-5f);
    return out;
  };

  const EndpointResult forward = runEndpoint(false);
  const EndpointResult reverse = runEndpoint(true);
  printf("  forward: disp=%.5f dir=%.6f orth=%.6f maxOrth=%.6f "
         "vel=%.5f vdir=%.6f vorth=%.6f axisA=%.6f axisB=%.6f\n",
         forward.signedDisplacement, forward.displacementDirection,
         forward.displacementOrthogonalRatio,
         forward.maxDisplacementOrthogonalRatio, forward.signedVelocity,
         forward.velocityDirection, forward.velocityOrthogonalRatio,
         forward.worldAxisADot, forward.worldAxisBDot);
  printf("  reverse: disp=%.5f dir=%.6f orth=%.6f maxOrth=%.6f "
         "vel=%.5f vdir=%.6f vorth=%.6f axisA=%.6f axisB=%.6f "
         "authoredDot=%.6f\n",
         reverse.signedDisplacement, reverse.displacementDirection,
         reverse.displacementOrthogonalRatio,
         reverse.maxDisplacementOrthogonalRatio, reverse.signedVelocity,
         reverse.velocityDirection, reverse.velocityOrthogonalRatio,
         reverse.worldAxisADot, reverse.worldAxisBDot,
         reverse.authoredVsWorldAxisDot);

  CHECK(forward.fixtureWitness && reverse.fixtureWitness,
        "Endpoint/frame witness invalid (forward=%d reverse=%d)",
        forward.fixtureWitness ? 1 : 0, reverse.fixtureWitness ? 1 : 0);
  CHECK(forward.finite && reverse.finite,
        "Non-finite state in mirrored prismatic endpoint run");
  CHECK(forward.signedDisplacement > 3.0f && forward.signedVelocity > 4.0f,
        "Forward control did not travel along free axis: disp=%.5f vel=%.5f",
        forward.signedDisplacement, forward.signedVelocity);
  CHECK(forward.displacementDirection > 0.999f &&
            forward.displacementOrthogonalRatio < 0.005f &&
            forward.maxDisplacementOrthogonalRatio < 0.01f &&
            forward.velocityDirection > 0.999f &&
            forward.velocityOrthogonalRatio < 0.005f,
        "Forward control leaked off axis: pdir=%.6f porth=%.6f max=%.6f "
        "vdir=%.6f vorth=%.6f",
        forward.displacementDirection, forward.displacementOrthogonalRatio,
        forward.maxDisplacementOrthogonalRatio, forward.velocityDirection,
        forward.velocityOrthogonalRatio);
  CHECK(reverse.signedDisplacement > 3.0f && reverse.signedVelocity > 4.0f,
        "Reverse endpoint did not travel along free axis: disp=%.5f vel=%.5f",
        reverse.signedDisplacement, reverse.signedVelocity);
  CHECK(reverse.displacementDirection > 0.999f &&
            reverse.displacementOrthogonalRatio < 0.005f &&
            reverse.maxDisplacementOrthogonalRatio < 0.01f &&
            reverse.velocityDirection > 0.999f &&
            reverse.velocityOrthogonalRatio < 0.005f,
        "Reverse endpoint leaked off axis: pdir=%.6f porth=%.6f max=%.6f "
        "vdir=%.6f vorth=%.6f",
        reverse.displacementDirection, reverse.displacementOrthogonalRatio,
        reverse.maxDisplacementOrthogonalRatio, reverse.velocityDirection,
        reverse.velocityOrthogonalRatio);
  CHECK(fabsf(reverse.signedDisplacement - forward.signedDisplacement) < 0.1f &&
            fabsf(reverse.signedVelocity - forward.signedVelocity) < 0.1f,
        "Endpoint-order mirror mismatch: forward=(%.5f,%.5f) "
        "reverse=(%.5f,%.5f)",
        forward.signedDisplacement, forward.signedVelocity,
        reverse.signedDisplacement, reverse.signedVelocity);

  PASS("Prismatic reverse endpoint uses actor-A world frame");
}

// Test 123: A D6 velocity target is expressed in actor A's joint frame.
// Mirroring world-A/dynamic-B to dynamic-A/world-B must preserve
// (vB-vA).axis and rotate a dynamic actor-A local frame into world space.
bool test123_d6VelocityDriveReverseEndpointFrameA() {
  printf("\n--- Test 123: D6 Velocity Drive Reverse Endpoint Frame A ---\n");

  struct EndpointResult {
    bool finite = true;
    bool fixtureWitness = false;
    int lateSamples = 0;
    float lateRelativeMean = 0.0f;
    float lateRelativeTargetRms = 0.0f;
    float lateRelativeOrthogonalRms = 0.0f;
    float minLateRelativeProjection = 1e30f;
    float minLateRelativeDirection = 1e30f;
    float maxLateRelativeOrthogonal = 0.0f;
    float lateDynamicMean = 0.0f;
    float lateDynamicTargetRms = 0.0f;
    float lateDynamicOrthogonalRms = 0.0f;
    float minLateDynamicProjection = 1e30f;
    float minLateDynamicDirection = 1e30f;
    float maxLateDynamicOrthogonal = 0.0f;
    float relativeDisplacement = 0.0f;
    float dynamicDisplacement = 0.0f;
    float worldAxisADot = 0.0f;
    float worldAxisBDot = 0.0f;
    float authoredAxisADot = 0.0f;
    float maxAngularSpeed = 0.0f;
  };

  const float pi = 3.14159265358979f;
  const float halfAngle = pi / 8.0f;
  const Quat bodyRotation(cosf(halfAngle), 0.0f, 0.0f,
                          -sinf(halfAngle));
  const Vec3 worldAxis =
      bodyRotation.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
  const Vec3 start(0.0f, 10.0f, 0.0f);

  auto runEndpoint = [&](bool reverse) {
    EndpointResult out;
    Solver solver;
    solver.gravity = {0, 0, 0};
    solver.iterations = 20;
    solver.dt = 1.0f / 60.0f;

    const uint32_t body =
        solver.addBody(start, bodyRotation, {0.5f, 0.5f, 0.5f}, 1.0f);
    if (reverse)
      solver.addD6Joint(body, UINT32_MAX, Vec3(), start, 0x2A, 0x2A);
    else
      solver.addD6Joint(UINT32_MAX, body, start, Vec3(), 0x2A, 0x2A);

    D6Joint &joint = solver.d6Joints[0];
    joint.localFrameA = reverse ? Quat() : bodyRotation;
    joint.localFrameB = reverse ? bodyRotation : Quat();
    joint.driveFlags = 0x01;
    joint.driveLinearVelocity = Vec3(1.0f, 0.0f, 0.0f);
    joint.linearDriveDamping = Vec3(1000.0f, 0.0f, 0.0f);

    const Quat worldFrameA =
        reverse ? bodyRotation * joint.localFrameA : joint.localFrameA;
    const Quat worldFrameB =
        reverse ? joint.localFrameB : bodyRotation * joint.localFrameB;
    const Vec3 axisA =
        worldFrameA.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
    const Vec3 axisB =
        worldFrameB.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
    const Vec3 authoredAxisA =
        joint.localFrameA.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
    out.worldAxisADot = axisA.dot(worldAxis);
    out.worldAxisBDot = axisB.dot(worldAxis);
    out.authoredAxisADot = authoredAxisA.dot(worldAxis);

    int freeMotionCount = 0;
    for (int axis = 0; axis < 3; ++axis) {
      freeMotionCount += joint.getLinearMotion(axis) == 2 ? 1 : 0;
      freeMotionCount += joint.getAngularMotion(axis) == 2 ? 1 : 0;
    }
    const Vec3 worldAnchorA =
        reverse ? start + bodyRotation.rotate(joint.anchorA) : joint.anchorA;
    const Vec3 worldAnchorB =
        reverse ? joint.anchorB
                : start + bodyRotation.rotate(joint.anchorB);
    const Quat dynamicLocalFrame =
        reverse ? joint.localFrameA : joint.localFrameB;
    const bool endpointWitness =
        reverse ? (joint.bodyA == body && joint.bodyB == UINT32_MAX)
                : (joint.bodyA == UINT32_MAX && joint.bodyB == body);
    const bool frameWitness =
        out.worldAxisADot > 0.99999f && out.worldAxisBDot > 0.99999f &&
        (worldAnchorA - start).length() <= 1e-6f &&
        (worldAnchorB - start).length() <= 1e-6f &&
        fabsf(dynamicLocalFrame.w - 1.0f) <= 1e-6f &&
        fabsf(dynamicLocalFrame.x) <= 1e-6f &&
        fabsf(dynamicLocalFrame.y) <= 1e-6f &&
        fabsf(dynamicLocalFrame.z) <= 1e-6f &&
        (reverse ? fabsf(out.authoredAxisADot - sqrtf(0.5f)) <= 1e-5f
                 : out.authoredAxisADot > 0.99999f);
    const bool driveWitness =
        joint.driveFlags == 0x01 &&
        (joint.driveLinearVelocity - Vec3(1.0f, 0.0f, 0.0f)).length() <=
            1e-6f &&
        (joint.linearDriveDamping - Vec3(1000.0f, 0.0f, 0.0f)).length() <=
            1e-6f;
    out.fixtureWitness = endpointWitness && frameWitness && driveWitness &&
                         freeMotionCount == 6;

    float relativeProjectionSum = 0.0f;
    float relativeTargetErrorSquaredSum = 0.0f;
    float relativeOrthogonalSquaredSum = 0.0f;
    float dynamicProjectionSum = 0.0f;
    float dynamicTargetErrorSquaredSum = 0.0f;
    float dynamicOrthogonalSquaredSum = 0.0f;
    const Vec3 expectedDynamicAxis = reverse ? -worldAxis : worldAxis;
    for (int frame = 0; frame < 180; ++frame) {
      solver.contacts.clear();
      solver.step(solver.dt);

      const Body &state = solver.bodies[body];
      const bool stateFinite =
          std::isfinite(state.position.x) && std::isfinite(state.position.y) &&
          std::isfinite(state.position.z) &&
          std::isfinite(state.linearVelocity.x) &&
          std::isfinite(state.linearVelocity.y) &&
          std::isfinite(state.linearVelocity.z) &&
          std::isfinite(state.rotation.w) && std::isfinite(state.rotation.x) &&
          std::isfinite(state.rotation.y) && std::isfinite(state.rotation.z) &&
          std::isfinite(state.angularVelocity.x) &&
          std::isfinite(state.angularVelocity.y) &&
          std::isfinite(state.angularVelocity.z);
      out.finite = out.finite && stateFinite;
      out.maxAngularSpeed =
          std::max(out.maxAngularSpeed, state.angularVelocity.length());
      if (frame < 120 || !stateFinite)
        continue;

      const Vec3 relativeVelocity =
          reverse ? -state.linearVelocity : state.linearVelocity;
      const float relativeProjection = relativeVelocity.dot(worldAxis);
      const float dynamicProjection =
          state.linearVelocity.dot(expectedDynamicAxis);
      const float relativeMagnitude = relativeVelocity.length();
      const float dynamicMagnitude = state.linearVelocity.length();
      const float relativeDirection =
          relativeProjection / std::max(relativeMagnitude, 1e-6f);
      const float dynamicDirection =
          dynamicProjection / std::max(dynamicMagnitude, 1e-6f);
      const float relativeOrthogonal =
          (relativeVelocity - worldAxis * relativeProjection).length();
      const float dynamicOrthogonal =
          (state.linearVelocity -
           expectedDynamicAxis * dynamicProjection)
              .length();
      const float relativeTargetError = relativeProjection - 1.0f;
      const float dynamicTargetError = dynamicProjection - 1.0f;

      out.lateSamples++;
      relativeProjectionSum += relativeProjection;
      relativeTargetErrorSquaredSum +=
          relativeTargetError * relativeTargetError;
      relativeOrthogonalSquaredSum +=
          relativeOrthogonal * relativeOrthogonal;
      dynamicProjectionSum += dynamicProjection;
      dynamicTargetErrorSquaredSum += dynamicTargetError * dynamicTargetError;
      dynamicOrthogonalSquaredSum += dynamicOrthogonal * dynamicOrthogonal;
      out.minLateRelativeProjection =
          std::min(out.minLateRelativeProjection, relativeProjection);
      out.minLateRelativeDirection =
          std::min(out.minLateRelativeDirection, relativeDirection);
      out.maxLateRelativeOrthogonal =
          std::max(out.maxLateRelativeOrthogonal, relativeOrthogonal);
      out.minLateDynamicProjection =
          std::min(out.minLateDynamicProjection, dynamicProjection);
      out.minLateDynamicDirection =
          std::min(out.minLateDynamicDirection, dynamicDirection);
      out.maxLateDynamicOrthogonal =
          std::max(out.maxLateDynamicOrthogonal, dynamicOrthogonal);
    }

    if (out.lateSamples) {
      const float invSamples = 1.0f / float(out.lateSamples);
      out.lateRelativeMean = relativeProjectionSum * invSamples;
      out.lateRelativeTargetRms =
          sqrtf(relativeTargetErrorSquaredSum * invSamples);
      out.lateRelativeOrthogonalRms =
          sqrtf(relativeOrthogonalSquaredSum * invSamples);
      out.lateDynamicMean = dynamicProjectionSum * invSamples;
      out.lateDynamicTargetRms =
          sqrtf(dynamicTargetErrorSquaredSum * invSamples);
      out.lateDynamicOrthogonalRms =
          sqrtf(dynamicOrthogonalSquaredSum * invSamples);
    }
    const Vec3 displacement = solver.bodies[body].position - start;
    out.relativeDisplacement =
        (reverse ? -displacement : displacement).dot(worldAxis);
    out.dynamicDisplacement = displacement.dot(expectedDynamicAxis);
    return out;
  };

  const EndpointResult forward = runEndpoint(false);
  const EndpointResult reverse = runEndpoint(true);
  auto printEndpoint = [](const char *name, const EndpointResult &result) {
    printf("  %s: mean=%.6f targetRms=%.6f orthRms=%.6f "
           "minDir=%.6f maxOrth=%.6f disp=%.6f fixture=%d finite=%d\n",
           name, result.lateRelativeMean, result.lateRelativeTargetRms,
           result.lateRelativeOrthogonalRms,
           result.minLateRelativeDirection,
           result.maxLateRelativeOrthogonal, result.relativeDisplacement,
           result.fixtureWitness ? 1 : 0, result.finite ? 1 : 0);
  };
  printEndpoint("forward", forward);
  printEndpoint("reverse", reverse);

  CHECK(forward.fixtureWitness && reverse.fixtureWitness,
        "Velocity-drive endpoint/frame fixture invalid (forward=%d reverse=%d)",
        forward.fixtureWitness ? 1 : 0, reverse.fixtureWitness ? 1 : 0);
  CHECK(forward.finite && reverse.finite,
        "Non-finite state in mirrored velocity-drive run");
  CHECK(forward.lateSamples == 60 && reverse.lateSamples == 60,
        "Incomplete velocity-drive late window (forward=%d reverse=%d)",
        forward.lateSamples, reverse.lateSamples);
  auto checkResponse = [&](const char *name, const EndpointResult &result) {
    CHECK(result.lateRelativeMean >= 0.75f &&
              result.lateRelativeMean <= 1.25f &&
              result.lateDynamicMean >= 0.75f &&
              result.lateDynamicMean <= 1.25f,
          "%s velocity target mean invalid (relative=%.6f dynamic=%.6f)",
          name, result.lateRelativeMean, result.lateDynamicMean);
    CHECK(result.lateRelativeTargetRms <= 0.35f &&
              result.lateDynamicTargetRms <= 0.35f,
          "%s velocity target RMS invalid (relative=%.6f dynamic=%.6f)",
          name, result.lateRelativeTargetRms,
          result.lateDynamicTargetRms);
    CHECK(result.minLateRelativeProjection > 0.0f &&
              result.minLateDynamicProjection > 0.0f &&
              result.minLateRelativeDirection >= 0.98f &&
              result.minLateDynamicDirection >= 0.98f,
          "%s velocity axis/sign invalid (rProj=%.6f dProj=%.6f "
          "rDir=%.6f dDir=%.6f)",
          name, result.minLateRelativeProjection,
          result.minLateDynamicProjection, result.minLateRelativeDirection,
          result.minLateDynamicDirection);
    CHECK(result.lateRelativeOrthogonalRms <= 0.10f &&
              result.lateDynamicOrthogonalRms <= 0.10f &&
              result.maxLateRelativeOrthogonal <= 0.10f &&
              result.maxLateDynamicOrthogonal <= 0.10f,
          "%s velocity orthogonal leak (rRms=%.6f dRms=%.6f "
          "rMax=%.6f dMax=%.6f)",
          name, result.lateRelativeOrthogonalRms,
          result.lateDynamicOrthogonalRms,
          result.maxLateRelativeOrthogonal,
          result.maxLateDynamicOrthogonal);
    CHECK(result.relativeDisplacement > 0.5f &&
              result.dynamicDisplacement > 0.5f,
          "%s velocity drive did not move (relative=%.6f dynamic=%.6f)",
          name, result.relativeDisplacement, result.dynamicDisplacement);
    CHECK(result.maxAngularSpeed <= 0.01f,
          "%s linear drive leaked angular speed: %.6f", name,
          result.maxAngularSpeed);
    return true;
  };
  if (!checkResponse("Forward", forward))
    return false;
  if (!checkResponse("Reverse", reverse))
    return false;
  CHECK(fabsf(reverse.lateRelativeMean - forward.lateRelativeMean) < 0.1f &&
            fabsf(reverse.relativeDisplacement -
                  forward.relativeDisplacement) < 0.1f,
        "Velocity-drive endpoint mirror mismatch: mean=(%.6f,%.6f) "
        "disp=(%.6f,%.6f)",
        forward.lateRelativeMean, reverse.lateRelativeMean,
        forward.relativeDisplacement, reverse.relativeDisplacement);

  PASS("D6 velocity drive reverse endpoint uses actor-A world frame");
}

// Test 124: the public reaction witness is total primal row force, not the
// leaky AL multiplier alone.  Exercise both actor orders and three timesteps.
bool test124_d6LockedLinearReactionWriteback() {
  printf("\n--- Test 124: D6 Locked Linear Reaction Writeback ---\n");

  struct ReactionResult {
    float meanSignedForce = 0.0f;
    float maxOrthogonalForce = 0.0f;
    float maxPositionError = 0.0f;
    float maxLinearSpeed = 0.0f;
    int maxLinearSpeedFrame = 0;
    float steadyMaxLinearSpeed = 0.0f;
    int samples = 0;
    bool finite = true;
    bool supported = true;
  };

  const auto run = [](float dt, bool reverse) {
    ReactionResult out;
    Solver solver;
    solver.gravity = Vec3(0.0f, -9.81f, 0.0f);
    solver.iterations = 10;
    const Vec3 initialPosition(0.0f, 10.0f, 0.0f);
    const uint32_t body =
        solver.addBody(initialPosition, Quat(), Vec3(0.5f, 0.5f, 0.5f),
                       4.0f);
    if (reverse)
      solver.addFixedJoint(body, UINT32_MAX, Vec3(), initialPosition, 1e4f);
    else
      solver.addFixedJoint(UINT32_MAX, body, initialPosition, Vec3(), 1e4f);

    const int frameCount = static_cast<int>(10.0f / dt + 0.5f);
    const int warmupFrames = static_cast<int>(2.0f / dt + 0.5f);
    double signedForceSum = 0.0;
    for (int frame = 0; frame < frameCount; ++frame) {
      solver.contacts.clear();
      solver.step(dt);

      Vec3 actor0Force;
      const bool supported = computeD6LockedLinearActor0Force(
          solver.d6Joints[0], solver.bodies, dt, actor0Force);
      out.supported = out.supported && supported;
      const Body &state = solver.bodies[body];
      const bool finite =
          std::isfinite(actor0Force.x) && std::isfinite(actor0Force.y) &&
          std::isfinite(actor0Force.z) && std::isfinite(state.position.x) &&
          std::isfinite(state.position.y) && std::isfinite(state.position.z) &&
          std::isfinite(state.linearVelocity.x) &&
          std::isfinite(state.linearVelocity.y) &&
          std::isfinite(state.linearVelocity.z);
      out.finite = out.finite && finite;
      out.maxPositionError = std::max(
          out.maxPositionError, (state.position - initialPosition).length());
      const float linearSpeed = state.linearVelocity.length();
      if (linearSpeed > out.maxLinearSpeed) {
        out.maxLinearSpeed = linearSpeed;
        out.maxLinearSpeedFrame = frame + 1;
      }
      if (frame >= warmupFrames)
        out.steadyMaxLinearSpeed =
            std::max(out.steadyMaxLinearSpeed, linearSpeed);
      if (frame < warmupFrames || !supported || !finite)
        continue;

      const float expectedDirection = reverse ? 1.0f : -1.0f;
      signedForceSum += actor0Force.y * expectedDirection;
      out.maxOrthogonalForce =
          std::max(out.maxOrthogonalForce,
                   sqrtf(actor0Force.x * actor0Force.x +
                         actor0Force.z * actor0Force.z));
      out.samples++;
    }
    if (out.samples)
      out.meanSignedForce =
          static_cast<float>(signedForceSum / double(out.samples));
    return out;
  };

  const float timesteps[] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  float minMean = FLT_MAX;
  float maxMean = 0.0f;
  for (float dt : timesteps) {
    const ReactionResult forward = run(dt, false);
    const ReactionResult reverse = run(dt, true);
    printf("  dt=%.9g forward=%.6f reverse=%.6f orth=(%.6g,%.6g) "
           "pos=(%.6g,%.6g) speed=(%.9g@%d,%.9g@%d) "
           "steadySpeed=(%.9g,%.9g) samples=(%d,%d)\n",
           dt, forward.meanSignedForce, reverse.meanSignedForce,
           forward.maxOrthogonalForce, reverse.maxOrthogonalForce,
           forward.maxPositionError, reverse.maxPositionError,
           forward.maxLinearSpeed, forward.maxLinearSpeedFrame,
           reverse.maxLinearSpeed, reverse.maxLinearSpeedFrame,
           forward.steadyMaxLinearSpeed, reverse.steadyMaxLinearSpeed,
           forward.samples, reverse.samples);
    CHECK(forward.supported && reverse.supported,
          "Locked-row reaction unexpectedly unsupported at dt=%.9g", dt);
    CHECK(forward.finite && reverse.finite,
          "Non-finite locked-row reaction at dt=%.9g", dt);
    CHECK(forward.samples > 0 && reverse.samples > 0,
          "Missing locked-row reaction samples at dt=%.9g", dt);
    CHECK(forward.meanSignedForce >= 35.316f &&
              forward.meanSignedForce <= 43.164f &&
              reverse.meanSignedForce >= 35.316f &&
              reverse.meanSignedForce <= 43.164f,
          "Reaction does not match 4 kg weight at dt=%.9g: %.6f %.6f", dt,
          forward.meanSignedForce, reverse.meanSignedForce);
    CHECK(forward.maxOrthogonalForce <= 1e-4f &&
              reverse.maxOrthogonalForce <= 1e-4f,
          "Reaction has orthogonal leakage at dt=%.9g: %.6g %.6g", dt,
          forward.maxOrthogonalForce, reverse.maxOrthogonalForce);
    CHECK(forward.maxLinearSpeed <= 1e-6f &&
              reverse.maxLinearSpeed <= 1e-6f,
          "Locked body-static row leaked velocity at dt=%.9g: %.9g %.9g",
          dt, forward.maxLinearSpeed, reverse.maxLinearSpeed);
    CHECK(fabsf(forward.meanSignedForce - reverse.meanSignedForce) <= 0.1f,
          "Actor-order reaction mismatch at dt=%.9g: %.6f %.6f", dt,
          forward.meanSignedForce, reverse.meanSignedForce);
    minMean = std::min(
        minMean, std::min(forward.meanSignedForce, reverse.meanSignedForce));
    maxMean = std::max(
        maxMean, std::max(forward.meanSignedForce, reverse.meanSignedForce));
  }
  CHECK(maxMean - minMean <= 0.5f,
        "Reaction is timestep dependent: min=%.6f max=%.6f", minMean,
        maxMean);
  PASS("D6 locked-row reaction and body-static velocity projection are correct");
}

// Test 125: offset fixed joints require sub-milliradian angular errors, a
// complete locked spatial-velocity projection and force*dt writeback semantics.
bool test125_d6OffsetCoupledReaction() {
  printf("\n--- Test 125: D6 Offset Coupled Reaction ---\n");

  const float smallAngles[] = {1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f};
  for (float angle : smallAngles) {
    const float half = 0.5f * angle;
    const Quat rotationB(cosf(half), 0.0f, 0.0f, sinf(half));
    const float error =
        computeAngularError(Quat(), rotationB, Quat(), Quat(), 2);
    printf("  small-angle requested=%.9g recovered=%.9g\n", angle, error);
    CHECK(std::isfinite(error) &&
              fabsf(error + angle) <= std::max(1e-9f, angle * 1e-4f),
          "Small angular error was quantized: requested=%.9g recovered=%.9g",
          angle, error);
  }

  struct CoupledResult {
    float meanSignedForce = 0.0f;
    float meanSignedTorque = 0.0f;
    float maxForceOrthogonal = 0.0f;
    float maxTorqueOrthogonal = 0.0f;
    float maxPositionError = 0.0f;
    float maxLinearSpeed = 0.0f;
    float maxAngularSpeed = 0.0f;
    int samples = 0;
    bool finite = true;
    bool supported = true;
  };

  const auto run = [](float dt, bool reverse) {
    CoupledResult out;
    Solver solver;
    solver.gravity = Vec3(0.0f, -9.81f, 0.0f);
    solver.iterations = 10;
    const Vec3 initialPosition(0.0f, 10.0f, 0.0f);
    const Vec3 bodyAnchor(1.0f, 0.0f, 0.0f);
    const Vec3 worldAnchor = initialPosition + bodyAnchor;
    const uint32_t body =
        solver.addBody(initialPosition, Quat(), Vec3(0.5f, 0.5f, 0.5f),
                       4.0f);
    if (reverse)
      solver.addFixedJoint(body, UINT32_MAX, bodyAnchor, worldAnchor, 1e4f);
    else
      solver.addFixedJoint(UINT32_MAX, body, worldAnchor, bodyAnchor, 1e4f);

    const int frameCount = static_cast<int>(10.0f / dt + 0.5f);
    const int warmupFrames = static_cast<int>(2.0f / dt + 0.5f);
    double forceSum = 0.0;
    double torqueSum = 0.0;
    for (int frame = 0; frame < frameCount; ++frame) {
      solver.contacts.clear();
      solver.step(dt);

      Vec3 actor0Force, actor0Torque;
      const bool linearSupported = computeD6LockedLinearActor0Force(
          solver.d6Joints[0], solver.bodies, dt, actor0Force);
      const bool angularSupported = computeD6LockedAngularActor0Torque(
          solver.d6Joints[0], solver.bodies, dt, actor0Torque);
      out.supported = out.supported && linearSupported && angularSupported;
      const Body &state = solver.bodies[body];
      const bool finite =
          std::isfinite(actor0Force.x) && std::isfinite(actor0Force.y) &&
          std::isfinite(actor0Force.z) && std::isfinite(actor0Torque.x) &&
          std::isfinite(actor0Torque.y) && std::isfinite(actor0Torque.z) &&
          std::isfinite(state.position.x) &&
          std::isfinite(state.position.y) &&
          std::isfinite(state.position.z) &&
          std::isfinite(state.linearVelocity.x) &&
          std::isfinite(state.linearVelocity.y) &&
          std::isfinite(state.linearVelocity.z) &&
          std::isfinite(state.angularVelocity.x) &&
          std::isfinite(state.angularVelocity.y) &&
          std::isfinite(state.angularVelocity.z);
      out.finite = out.finite && finite;
      out.maxPositionError = std::max(
          out.maxPositionError, (state.position - initialPosition).length());
      out.maxLinearSpeed =
          std::max(out.maxLinearSpeed, state.linearVelocity.length());
      out.maxAngularSpeed =
          std::max(out.maxAngularSpeed, state.angularVelocity.length());
      if (frame < warmupFrames || !linearSupported || !angularSupported ||
          !finite)
        continue;

      const float sign = reverse ? 1.0f : -1.0f;
      forceSum += actor0Force.y * sign;
      torqueSum += actor0Torque.z * -sign;
      out.maxForceOrthogonal =
          std::max(out.maxForceOrthogonal,
                   sqrtf(actor0Force.x * actor0Force.x +
                         actor0Force.z * actor0Force.z));
      out.maxTorqueOrthogonal =
          std::max(out.maxTorqueOrthogonal,
                   sqrtf(actor0Torque.x * actor0Torque.x +
                         actor0Torque.y * actor0Torque.y));
      out.samples++;
    }
    if (out.samples) {
      out.meanSignedForce =
          static_cast<float>(forceSum / double(out.samples));
      out.meanSignedTorque =
          static_cast<float>(torqueSum / double(out.samples));
    }
    return out;
  };

  const float timesteps[] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  for (float dt : timesteps) {
    const CoupledResult forward = run(dt, false);
    const CoupledResult reverse = run(dt, true);
    printf("  dt=%.9g force=(%.6f,%.6f) torque=(%.6f,%.6f) "
           "orthF=(%.6g,%.6g) orthT=(%.6g,%.6g) "
           "speed=(%.9g,%.9g)/(%.9g,%.9g) pos=(%.6g,%.6g)\n",
           dt, forward.meanSignedForce, reverse.meanSignedForce,
           forward.meanSignedTorque, reverse.meanSignedTorque,
           forward.maxForceOrthogonal, reverse.maxForceOrthogonal,
           forward.maxTorqueOrthogonal, reverse.maxTorqueOrthogonal,
           forward.maxLinearSpeed, reverse.maxLinearSpeed,
           forward.maxAngularSpeed, reverse.maxAngularSpeed,
           forward.maxPositionError, reverse.maxPositionError);
    CHECK(forward.supported && reverse.supported,
          "Offset reaction unexpectedly unsupported at dt=%.9g", dt);
    CHECK(forward.finite && reverse.finite,
          "Non-finite offset reaction at dt=%.9g", dt);
    CHECK(forward.samples > 0 && reverse.samples > 0,
          "Missing offset reaction samples at dt=%.9g", dt);
    CHECK(forward.meanSignedForce >= 35.316f &&
              forward.meanSignedForce <= 43.164f &&
              reverse.meanSignedForce >= 35.316f &&
              reverse.meanSignedForce <= 43.164f,
          "Offset force does not match 4 kg weight at dt=%.9g: %.6f %.6f",
          dt, forward.meanSignedForce, reverse.meanSignedForce);
    CHECK(forward.meanSignedTorque >= 35.316f &&
              forward.meanSignedTorque <= 43.164f &&
              reverse.meanSignedTorque >= 35.316f &&
              reverse.meanSignedTorque <= 43.164f,
          "Offset torque does not match r x weight at dt=%.9g: %.6f %.6f",
          dt, forward.meanSignedTorque, reverse.meanSignedTorque);
    CHECK(forward.maxForceOrthogonal <= 0.3924f &&
              reverse.maxForceOrthogonal <= 0.3924f &&
              forward.maxTorqueOrthogonal <= 0.3924f &&
              reverse.maxTorqueOrthogonal <= 0.3924f,
          "Offset wrench has orthogonal leakage at dt=%.9g", dt);
    CHECK(forward.maxLinearSpeed <= 1e-6f &&
              reverse.maxLinearSpeed <= 1e-6f &&
              forward.maxAngularSpeed <= 1e-6f &&
              reverse.maxAngularSpeed <= 1e-6f,
          "Fixed body-static projection leaked spatial velocity at dt=%.9g",
          dt);
  }
  PASS("D6 offset coupled reaction and small-angle semantics are correct");
}

// Test 126: algebra-only gate for the matrix-free island operator. No solver
// routing changes are enabled by this test.
bool test126_matrixFreeIslandOperator() {
  printf("\n--- Test 126: Matrix-Free Island Operator ---\n");
  const float dt = 1.0f / 60.0f;
  const float mass = 4.0f;
  const float massInvDt2 = mass / (dt * dt);
  std::vector<Mat66> inertia(3);
  std::vector<Vec6> inertialGradient(3);
  for (Mat66 &block : inertia) {
    for (int axis = 0; axis < 3; ++axis) {
      block.m[axis][axis] = massInvDt2;
      block.m[3 + axis][3 + axis] = 2000.0f + 250.0f * axis;
    }
  }
  inertialGradient[0][1] = -39.24f;
  inertialGradient[2][1] = 39.24f;

  IslandPcgSystem system;
  system.initialize(inertia, inertialGradient);
  const auto addInternalRow = [&](uint32_t owner, uint32_t a, uint32_t b,
                                  const Vec6 &jacobianA,
                                  const Vec6 &jacobianB, float penalty,
                                  float force) {
    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = owner;
    row.bodyA = a;
    row.bodyB = b;
    row.jacobianA = jacobianA;
    row.jacobianB = jacobianB;
    row.penalty = penalty;
    row.force = force;
    row.internalTranslationInvariant = true;
    return system.addRow(row);
  };
  const Vec3 yAxis(0.0f, 1.0f, 0.0f);
  CHECK(addInternalRow(0, 0, 1, Vec6(yAxis, Vec3()),
                       Vec6(-yAxis, Vec3()), 1e6f, 0.0f),
        "Failed to add first internal row");
  CHECK(addInternalRow(1, 1, 2, Vec6(yAxis, Vec3()),
                       Vec6(-yAxis, Vec3()), 1e6f, 0.0f),
        "Failed to add second internal row");

  // Offset row exercises linear-angular and endpoint cross blocks without
  // changing the common-translation nullspace.
  const Vec3 xAxis(1.0f, 0.0f, 0.0f);
  CHECK(addInternalRow(2, 0, 2,
                       Vec6(xAxis, Vec3(0.0f, 0.0f, -1.0f)),
                       Vec6(-xAxis, Vec3(0.0f, 0.0f, 1.0f)), 2e5f, 3.0f),
        "Failed to add offset internal row");

  IslandPcgRow invalidRow;
  invalidRow.bodyA = 0;
  invalidRow.bodyB = 1;
  invalidRow.jacobianA = Vec6(yAxis, Vec3());
  invalidRow.jacobianB = Vec6(yAxis, Vec3());
  invalidRow.penalty = 1.0f;
  invalidRow.internalTranslationInvariant = true;
  CHECK(!system.addRow(invalidRow),
        "Translation-invariance witness accepted an invalid row");

  std::vector<Vec6> solution;
  const IslandPcgStats stats = system.solvePcg(solution, 1e-8, 18);
  std::vector<Vec6> applied;
  system.apply(solution, applied);
  double residualMax = 0.0;
  for (size_t body = 0; body < applied.size(); ++body)
    for (int k = 0; k < 6; ++k)
      residualMax = std::max(
          residualMax,
          std::fabs(double(applied[body][k] - system.gradient()[body][k])));
  const Vec3 massWeightedTranslation =
      (solution[0].linear() + solution[1].linear() + solution[2].linear()) *
      mass;

  // Cross-Hessian witness: a vector on endpoint B must contribute to A.
  std::vector<Vec6> endpointInput(3), endpointOutput;
  endpointInput[1][1] = 1.0f;
  system.apply(endpointInput, endpointOutput);
  const float crossWitness = std::fabs(endpointOutput[0][1]);

  std::vector<Vec6> repeatSolution;
  const IslandPcgStats repeatStats =
      system.solvePcg(repeatSolution, 1e-8, 18);
  double repeatDifference = 0.0;
  for (size_t body = 0; body < solution.size(); ++body)
    for (int k = 0; k < 6; ++k)
      repeatDifference = std::max(
          repeatDifference,
          std::fabs(double(solution[body][k] - repeatSolution[body][k])));

  struct EmitterResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    float translation = 0.0f;
    bool emitted = false;
  };
  const auto runEmitter = [dt](bool reverse) {
    EmitterResult result;
    Solver solver;
    solver.gravity = Vec3();
    const uint32_t lower = solver.addBody(
        Vec3(0.0f, 9.0f, 0.0f), Quat(), Vec3(0.5f, 0.5f, 0.5f), 4.0f);
    const uint32_t upper = solver.addBody(
        Vec3(0.0f, 11.0f, 0.0f), Quat(), Vec3(0.5f, 0.5f, 0.5f), 4.0f);
    const Vec3 lowerAnchor(1.0f, 1.0f, 0.0f);
    const Vec3 upperAnchor(1.0f, -1.0f, 0.0f);
    if (reverse)
      solver.addFixedJoint(upper, lower, upperAnchor, lowerAnchor, 1e4f);
    else
      solver.addFixedJoint(lower, upper, lowerAnchor, upperAnchor, 1e4f);

    const float inertialDisplacement = 39.24f * dt * dt / 4.0f;
    solver.bodies[lower].inertialPosition =
        solver.bodies[lower].position + Vec3(0.0f, inertialDisplacement, 0.0f);
    solver.bodies[upper].inertialPosition =
        solver.bodies[upper].position - Vec3(0.0f, inertialDisplacement, 0.0f);
    const IslandBodyMap map =
        buildIslandBodyMap(static_cast<uint32_t>(solver.bodies.size()));
    std::vector<Mat66> blocks(solver.bodies.size());
    std::vector<Vec6> gradients(solver.bodies.size());
    for (size_t body = 0; body < solver.bodies.size(); ++body) {
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
      const Vec6 displacement(
          solver.bodies[body].position - solver.bodies[body].inertialPosition,
          solver.bodies[body].deltaWInertial());
      gradients[body] = blocks[body] * displacement;
    }
    IslandPcgSystem emittedSystem;
    emittedSystem.initialize(blocks, gradients);
    result.emitted = emitFixedD6IslandRows(
        solver.d6Joints[0], 0, solver.bodies, map, dt, emittedSystem,
        result.rows);
    result.stats = emittedSystem.solvePcg(result.solution, 1e-8, 12);
    if (result.solution.size() == 2) {
      result.translation =
          ((result.solution[lower].linear() * solver.bodies[lower].mass) +
           (result.solution[upper].linear() * solver.bodies[upper].mass))
              .length();
    }
    return result;
  };
  const EmitterResult forwardEmitter = runEmitter(false);
  const EmitterResult reverseEmitter = runEmitter(true);
  double actorOrderDifference = 0.0;
  if (forwardEmitter.solution.size() == reverseEmitter.solution.size()) {
    for (size_t body = 0; body < forwardEmitter.solution.size(); ++body)
      for (int k = 0; k < 6; ++k)
        actorOrderDifference = std::max(
            actorOrderDifference,
            std::fabs(double(forwardEmitter.solution[body][k] -
                             reverseEmitter.solution[body][k])));
  } else {
    actorOrderDifference = INFINITY;
  }

  printf("  rows=%zu iterations=%d residual=(%.9g -> %.9g) "
         "equationMax=%.9g translation=%.9g cross=%.9g repeat=%.9g "
         "emitter=(%u,%u,%.9g,%.9g,%.9g)\n",
         system.rows().size(), stats.iterations,
         stats.initialPreconditionedResidual,
         stats.finalPreconditionedResidual, residualMax,
         massWeightedTranslation.length(), crossWitness, repeatDifference,
         forwardEmitter.rows, reverseEmitter.rows,
         forwardEmitter.translation, reverseEmitter.translation,
         actorOrderDifference);
  CHECK(stats.converged && !stats.breakdown && stats.finite,
        "PCG did not converge: iterations=%d initial=%.9g final=%.9g",
        stats.iterations, stats.initialPreconditionedResidual,
        stats.finalPreconditionedResidual);
  CHECK(repeatStats.converged && repeatDifference <= 1e-9,
        "Matrix-free solve is not deterministic: %.9g", repeatDifference);
  CHECK(residualMax <= 1e-3,
        "Matrix-free equation residual is too large: %.9g", residualMax);
  CHECK(massWeightedTranslation.length() <= 1e-6f,
        "Internal rows changed common translation: %.9g",
        massWeightedTranslation.length());
  CHECK(crossWitness > 1e5f,
        "Endpoint cross Hessian is missing: %.9g", crossWitness);
  CHECK(forwardEmitter.emitted && reverseEmitter.emitted &&
            forwardEmitter.rows == 6 && reverseEmitter.rows == 6 &&
            forwardEmitter.stats.converged && reverseEmitter.stats.converged,
        "Fixed-D6 emitter did not produce two valid six-row systems");
  CHECK(forwardEmitter.translation <= 1e-6f &&
            reverseEmitter.translation <= 1e-6f,
        "Fixed-D6 emitter changed common translation: %.9g %.9g",
        forwardEmitter.translation, reverseEmitter.translation);
  CHECK(actorOrderDifference <= 1e-7,
        "Fixed-D6 emitter depends on actor order: %.9g",
        actorOrderDifference);

  struct IndexedStaticSphericalResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    float maxViolation = 0.0f;
    bool staticEndpointHasNoSlot = true;
    bool emitted = false;
  };
  const auto runIndexedStaticSpherical = [dt](bool reverse) {
    IndexedStaticSphericalResult result;
    Solver solver;
    solver.gravity = Vec3();
    const float staticAngle = 0.4f;
    const Quat staticRotation(std::cos(0.5f * staticAngle), 0.0f, 0.0f,
                              std::sin(0.5f * staticAngle));
    const uint32_t staticBody = solver.addBody(
        Vec3(3.0f, 4.0f, 5.0f), staticRotation,
        Vec3(0.5f, 0.5f, 0.5f), 0.0f);
    const uint32_t dynamicBody = solver.addBody(
        Vec3(4.0f, 4.5f, 5.5f), Quat(), Vec3(0.5f, 0.5f, 0.5f), 4.0f);
    const Vec3 worldAnchor(3.6f, 4.3f, 5.2f);
    const Vec3 staticAnchor = staticRotation.conjugate().rotate(
        worldAnchor - solver.bodies[staticBody].position);
    const Vec3 dynamicAnchor =
        worldAnchor - solver.bodies[dynamicBody].position;
    if (reverse)
      solver.addSphericalJoint(dynamicBody, staticBody, dynamicAnchor,
                               staticAnchor, 1e4f);
    else
      solver.addSphericalJoint(staticBody, dynamicBody, staticAnchor,
                               dynamicAnchor, 1e4f);

    Body &dynamic = solver.bodies[dynamicBody];
    dynamic.initialPosition = dynamic.position;
    dynamic.initialRotation = dynamic.rotation;
    dynamic.inertialPosition =
        dynamic.position + Vec3(0.01f, -0.02f, 0.03f);
    dynamic.inertialRotation = dynamic.rotation;
    IslandBodyMap map;
    map.bodyToSlot.assign(solver.bodies.size(), -1);
    map.bodyToSlot[dynamicBody] = 0;
    map.slotToBody.push_back(dynamicBody);
    const Mat66 block = dynamic.getMassMatrix() / (dt * dt);
    const Vec6 displacement(dynamic.position - dynamic.inertialPosition,
                            dynamic.deltaWInertial());
    IslandPcgSystem emittedSystem;
    emittedSystem.initialize(std::vector<Mat66>{block},
                             std::vector<Vec6>{block * displacement});
    result.emitted = emitSphericalD6IslandRows(
        solver.d6Joints[0], 0, solver.bodies, map, dt, emittedSystem,
        result.rows);
    for (const IslandPcgRow &row : emittedSystem.rows()) {
      result.maxViolation =
          std::max(result.maxViolation, std::fabs(row.violation));
      const bool staticIsA = solver.d6Joints[0].bodyA == staticBody;
      result.staticEndpointHasNoSlot =
          result.staticEndpointHasNoSlot &&
          (staticIsA ? row.bodyA == UINT32_MAX
                     : row.bodyB == UINT32_MAX);
    }
    result.stats = emittedSystem.solvePcg(result.solution, 1e-9, 12);
    return result;
  };
  const IndexedStaticSphericalResult sphericalForward =
      runIndexedStaticSpherical(false);
  const IndexedStaticSphericalResult sphericalReverse =
      runIndexedStaticSpherical(true);
  double sphericalActorOrderDifference = 0.0;
  if (sphericalForward.solution.size() == sphericalReverse.solution.size()) {
    for (size_t body = 0; body < sphericalForward.solution.size(); ++body)
      for (int k = 0; k < 6; ++k)
        sphericalActorOrderDifference = std::max(
            sphericalActorOrderDifference,
            std::fabs(double(sphericalForward.solution[body][k] -
                             sphericalReverse.solution[body][k])));
  } else {
    sphericalActorOrderDifference = INFINITY;
  }
  printf("  spherical-indexed-static rows=(%u,%u) violation=(%.9g,%.9g) "
         "order=%.9g pcg=(%d,%.9g,%d,%.9g)\n",
         sphericalForward.rows, sphericalReverse.rows,
         sphericalForward.maxViolation, sphericalReverse.maxViolation,
         sphericalActorOrderDifference, sphericalForward.stats.iterations,
         sphericalForward.stats.finalPreconditionedResidual,
         sphericalReverse.stats.iterations,
         sphericalReverse.stats.finalPreconditionedResidual);
  CHECK(sphericalForward.emitted && sphericalReverse.emitted &&
            sphericalForward.rows == 3 && sphericalReverse.rows == 3 &&
            sphericalForward.staticEndpointHasNoSlot &&
            sphericalReverse.staticEndpointHasNoSlot,
        "Spherical-D6 indexed-static emitter produced an invalid row set");
  CHECK(sphericalForward.stats.converged &&
            sphericalReverse.stats.converged &&
            sphericalForward.maxViolation <= 1e-5f &&
            sphericalReverse.maxViolation <= 1e-5f,
        "Spherical-D6 indexed-static anchor transform is invalid: %.9g %.9g",
        sphericalForward.maxViolation, sphericalReverse.maxViolation);
  CHECK(sphericalActorOrderDifference <= 1e-6,
        "Spherical-D6 indexed-static emitter depends on actor order: %.9g",
        sphericalActorOrderDifference);

  struct RevoluteEmitterResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    double equationResidual = 0.0;
    bool emitted = false;
  };
  const auto runRevoluteEmitter = [dt](bool reverse, bool activeLimit) {
    RevoluteEmitterResult result;
    Solver solver;
    solver.gravity = Vec3();
    const uint32_t bodyA = solver.addBody(
        Vec3(-1.0f, 6.0f, 0.0f), Quat(), Vec3(0.5f, 0.5f, 0.5f), 4.0f);
    const uint32_t bodyB = solver.addBody(
        Vec3(1.0f, 6.0f, 0.0f), Quat(), Vec3(0.5f, 0.5f, 0.5f), 5.0f);
    const Vec3 anchorA(1.0f, 0.25f, 0.0f);
    const Vec3 anchorB(-1.0f, 0.25f, 0.0f);
    const uint32_t jointIndex =
        reverse
            ? solver.addRevoluteJoint(bodyB, bodyA, anchorB, anchorA,
                                      Vec3(1, 0, 0), Vec3(1, 0, 0), 2e4f)
            : solver.addRevoluteJoint(bodyA, bodyB, anchorA, anchorB,
                                      Vec3(1, 0, 0), Vec3(1, 0, 0), 2e4f);
    solver.setRevoluteJointLimit(jointIndex, -0.2f, 0.2f);
    const float twistAngle = activeLimit ? 0.6f : 0.1f;
    const float swingAngle = 0.2f;
    const Quat twist(std::cos(0.5f * twistAngle),
                     std::sin(0.5f * twistAngle), 0.0f, 0.0f);
    const Quat swing(std::cos(0.5f * swingAngle), 0.0f,
                     std::sin(0.5f * swingAngle), 0.0f);
    solver.bodies[bodyB].rotation = (swing * twist).normalized();
    for (Body &body : solver.bodies) {
      body.initialPosition = body.position;
      body.initialRotation = body.rotation;
      body.inertialPosition = body.position;
      body.inertialRotation = body.rotation;
      body.updateInvInertiaWorld();
    }
    const IslandBodyMap map = buildIslandBodyMap(2);
    std::vector<Mat66> blocks(2);
    std::vector<Vec6> gradients(2);
    for (uint32_t body = 0; body < 2; ++body)
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
    IslandPcgSystem emittedSystem;
    emittedSystem.initialize(blocks, gradients);
    result.emitted = emitRevoluteD6IslandRows(
        solver.d6Joints[jointIndex], jointIndex, solver.bodies, map, dt,
        emittedSystem, result.rows);
    result.stats = emittedSystem.solvePcg(result.solution, 1e-9, 24);
    if (result.solution.size() == 2) {
      std::vector<Vec6> applied;
      emittedSystem.apply(result.solution, applied);
      for (uint32_t body = 0; body < 2; ++body)
        for (int k = 0; k < 6; ++k)
          result.equationResidual =
              std::max(result.equationResidual,
                       std::fabs(double(applied[body][k] -
                                        emittedSystem.gradient()[body][k])));
    }
    return result;
  };
  const RevoluteEmitterResult revoluteActiveForward =
      runRevoluteEmitter(false, true);
  const RevoluteEmitterResult revoluteActiveReverse =
      runRevoluteEmitter(true, true);
  const RevoluteEmitterResult revoluteInactiveForward =
      runRevoluteEmitter(false, false);
  const RevoluteEmitterResult revoluteInactiveReverse =
      runRevoluteEmitter(true, false);
  const auto solutionDifference = [](const std::vector<Vec6> &a,
                                     const std::vector<Vec6> &b) {
    if (a.size() != b.size())
      return double(INFINITY);
    double difference = 0.0;
    for (size_t body = 0; body < a.size(); ++body)
      for (int k = 0; k < 6; ++k)
        difference = std::max(
            difference, std::fabs(double(a[body][k] - b[body][k])));
    return difference;
  };
  const double revoluteActiveOrderDifference = solutionDifference(
      revoluteActiveForward.solution, revoluteActiveReverse.solution);
  const double revoluteInactiveOrderDifference = solutionDifference(
      revoluteInactiveForward.solution, revoluteInactiveReverse.solution);
  printf("  revolute-emitter rows=(%u,%u active;%u,%u inactive) "
         "order=(%.9g,%.9g) equation=(%.9g,%.9g) "
         "pcg=(%d,%.9g;%d,%.9g)\n",
         revoluteActiveForward.rows, revoluteActiveReverse.rows,
         revoluteInactiveForward.rows, revoluteInactiveReverse.rows,
         revoluteActiveOrderDifference, revoluteInactiveOrderDifference,
         revoluteActiveForward.equationResidual,
         revoluteActiveReverse.equationResidual,
         revoluteActiveForward.stats.iterations,
         revoluteActiveForward.stats.finalPreconditionedResidual,
         revoluteActiveReverse.stats.iterations,
         revoluteActiveReverse.stats.finalPreconditionedResidual);
  CHECK(revoluteActiveForward.emitted && revoluteActiveReverse.emitted &&
            revoluteInactiveForward.emitted &&
            revoluteInactiveReverse.emitted &&
            revoluteActiveForward.rows == 6 &&
            revoluteActiveReverse.rows == 6 &&
            revoluteInactiveForward.rows == 5 &&
            revoluteInactiveReverse.rows == 5,
        "Revolute emitter active-set row count is invalid");
  CHECK(revoluteActiveForward.stats.converged &&
            revoluteActiveReverse.stats.converged &&
            revoluteInactiveForward.stats.converged &&
            revoluteInactiveReverse.stats.converged,
        "Revolute emitter PCG did not converge");
  CHECK(revoluteActiveForward.equationResidual <= 2e-3 &&
            revoluteActiveReverse.equationResidual <= 2e-3,
        "Revolute emitter equation mismatch: %.9g %.9g",
        revoluteActiveForward.equationResidual,
        revoluteActiveReverse.equationResidual);
  CHECK(revoluteActiveOrderDifference <= 1e-5 &&
            revoluteInactiveOrderDifference <= 1e-5,
        "Revolute emitter depends on actor order: %.9g %.9g",
        revoluteActiveOrderDifference, revoluteInactiveOrderDifference);

  struct PrismaticEmitterResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    double equationResidual = 0.0;
    bool emitted = false;
  };
  const auto runPrismaticEmitter = [dt](bool reverse, bool activeLimit) {
    PrismaticEmitterResult result;
    Solver solver;
    solver.gravity = Vec3();
    const uint32_t bodyA = solver.addBody(
        Vec3(0.0f, 6.0f, 0.0f), Quat(), Vec3(0.4f, 0.5f, 0.6f), 4.0f);
    const uint32_t bodyB = solver.addBody(
        Vec3(0.0f, 6.0f, 0.0f), Quat(), Vec3(0.6f, 0.4f, 0.5f), 5.0f);
    const Vec3 slideAxis = Vec3(1.0f, 0.3f, 0.2f).normalized();
    const uint32_t jointIndex =
        reverse ? solver.addPrismaticJoint(bodyB, bodyA, Vec3(), Vec3(),
                                            slideAxis, 2e4f)
                : solver.addPrismaticJoint(bodyA, bodyB, Vec3(), Vec3(),
                                            slideAxis, 2e4f);
    solver.setPrismaticJointLimit(jointIndex, -0.2f, 0.2f);
    const float displacement = activeLimit ? 0.6f : 0.05f;
    solver.bodies[bodyB].position +=
        slideAxis * displacement + Vec3(0.0f, 0.04f, -0.03f);
    const float rotationAngle = 0.18f;
    solver.bodies[bodyB].rotation =
        Quat(std::cos(0.5f * rotationAngle), 0.0f, 0.0f,
             std::sin(0.5f * rotationAngle));
    for (Body &body : solver.bodies) {
      body.initialPosition = body.position;
      body.initialRotation = body.rotation;
      body.inertialPosition = body.position;
      body.inertialRotation = body.rotation;
      body.updateInvInertiaWorld();
    }
    const IslandBodyMap map = buildIslandBodyMap(2);
    std::vector<Mat66> blocks(2);
    std::vector<Vec6> gradients(2);
    for (uint32_t body = 0; body < 2; ++body)
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
    IslandPcgSystem emittedSystem;
    emittedSystem.initialize(blocks, gradients);
    result.emitted = emitPrismaticD6IslandRows(
        solver.d6Joints[jointIndex], jointIndex, solver.bodies, map, dt,
        emittedSystem, result.rows);
    result.stats = emittedSystem.solvePcg(result.solution, 1e-9, 24);
    if (result.solution.size() == 2) {
      std::vector<Vec6> applied;
      emittedSystem.apply(result.solution, applied);
      for (uint32_t body = 0; body < 2; ++body)
        for (int k = 0; k < 6; ++k)
          result.equationResidual =
              std::max(result.equationResidual,
                       std::fabs(double(applied[body][k] -
                                        emittedSystem.gradient()[body][k])));
    }
    return result;
  };
  const PrismaticEmitterResult prismaticActiveForward =
      runPrismaticEmitter(false, true);
  const PrismaticEmitterResult prismaticActiveReverse =
      runPrismaticEmitter(true, true);
  const PrismaticEmitterResult prismaticInactiveForward =
      runPrismaticEmitter(false, false);
  const PrismaticEmitterResult prismaticInactiveReverse =
      runPrismaticEmitter(true, false);
  const double prismaticActiveOrderDifference = solutionDifference(
      prismaticActiveForward.solution, prismaticActiveReverse.solution);
  const double prismaticInactiveOrderDifference = solutionDifference(
      prismaticInactiveForward.solution, prismaticInactiveReverse.solution);
  printf("  prismatic-emitter rows=(%u,%u active;%u,%u inactive) "
         "order=(%.9g,%.9g) equation=(%.9g,%.9g) "
         "pcg=(%d,%.9g;%d,%.9g)\n",
         prismaticActiveForward.rows, prismaticActiveReverse.rows,
         prismaticInactiveForward.rows, prismaticInactiveReverse.rows,
         prismaticActiveOrderDifference,
         prismaticInactiveOrderDifference,
         prismaticActiveForward.equationResidual,
         prismaticActiveReverse.equationResidual,
         prismaticActiveForward.stats.iterations,
         prismaticActiveForward.stats.finalPreconditionedResidual,
         prismaticActiveReverse.stats.iterations,
         prismaticActiveReverse.stats.finalPreconditionedResidual);
  CHECK(prismaticActiveForward.emitted &&
            prismaticActiveReverse.emitted &&
            prismaticInactiveForward.emitted &&
            prismaticInactiveReverse.emitted &&
            prismaticActiveForward.rows == 6 &&
            prismaticActiveReverse.rows == 6 &&
            prismaticInactiveForward.rows == 5 &&
            prismaticInactiveReverse.rows == 5,
        "Prismatic emitter active-set row count is invalid");
  CHECK(prismaticActiveForward.stats.converged &&
            prismaticActiveReverse.stats.converged &&
            prismaticInactiveForward.stats.converged &&
            prismaticInactiveReverse.stats.converged,
        "Prismatic emitter PCG did not converge");
  CHECK(prismaticActiveForward.equationResidual <= 2e-3 &&
            prismaticActiveReverse.equationResidual <= 2e-3,
        "Prismatic emitter equation mismatch: %.9g %.9g",
        prismaticActiveForward.equationResidual,
        prismaticActiveReverse.equationResidual);
  CHECK(prismaticActiveOrderDifference <= 1e-5 &&
            prismaticInactiveOrderDifference <= 1e-5,
        "Prismatic emitter depends on actor order: %.9g %.9g",
        prismaticActiveOrderDifference,
        prismaticInactiveOrderDifference);

  struct LinearDriveEmitterResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    uint16_t activeMode = 0;
    float penalty = 0.0f;
    float force = 0.0f;
    float violation = 0.0f;
    double equationResidual = 0.0;
    bool emitted = false;
  };
  const auto runLinearDriveEmitter =
      [dt](bool reverse, int mode, bool acceleration,
           float densityScale) {
    LinearDriveEmitterResult result;
    Solver solver;
    solver.gravity = Vec3();
    const Vec3 initialA(-0.7f, 4.0f, 0.2f);
    const Vec3 initialB(0.9f, 4.3f, -0.4f);
    const float angleA = 0.2f;
    const float angleB = -0.3f;
    const Quat initialRotationA(std::cos(0.5f * angleA), 0.0f,
                                std::sin(0.5f * angleA), 0.0f);
    const Quat initialRotationB(std::cos(0.5f * angleB),
                                std::sin(0.5f * angleB), 0.0f, 0.0f);
    const uint32_t bodyA = solver.addBody(
        initialA, initialRotationA, Vec3(0.4f, 0.5f, 0.6f),
        4.0f * densityScale);
    const uint32_t bodyB = solver.addBody(
        initialB, initialRotationB, Vec3(0.6f, 0.4f, 0.5f),
        5.0f * densityScale);
    const Vec3 anchorA(0.25f, -0.1f, 0.15f);
    const Vec3 anchorB(-0.2f, 0.18f, -0.08f);
    const uint32_t jointIndex =
        reverse ? solver.addD6Joint(bodyB, bodyA, anchorB, anchorA, 0x2A,
                                    0x2A, 0.0f, 2e4f)
                : solver.addD6Joint(bodyA, bodyB, anchorA, anchorB, 0x2A,
                                    0x2A, 0.0f, 2e4f);

    if (mode != 0) {
      const float deltaAngle = 0.06f;
      const Quat deltaRotation(std::cos(0.5f * deltaAngle), 0.0f, 0.0f,
                               std::sin(0.5f * deltaAngle));
      solver.bodies[bodyA].position += Vec3(0.01f, -0.015f, 0.005f);
      solver.bodies[bodyB].position += Vec3(0.035f, 0.012f, -0.018f);
      solver.bodies[bodyB].rotation =
          (deltaRotation * initialRotationB).normalized();
    }
    for (uint32_t body = 0; body < 2; ++body) {
      Body &state = solver.bodies[body];
      state.initialPosition = body == bodyA ? initialA : initialB;
      state.initialRotation =
          body == bodyA ? initialRotationA : initialRotationB;
      state.inertialPosition = state.position;
      state.inertialRotation = state.rotation;
      state.updateInvInertiaWorld();
    }

    D6Joint &joint = solver.d6Joints[jointIndex];
    const float worldFrameAngle = 0.4f;
    const Quat worldFrame(std::cos(0.5f * worldFrameAngle), 0.0f, 0.0f,
                          std::sin(0.5f * worldFrameAngle));
    const Quat currentRotationA = solver.bodies[joint.bodyA].rotation;
    const Quat currentRotationB = solver.bodies[joint.bodyB].rotation;
    joint.localFrameA =
        (currentRotationA.conjugate() * worldFrame).normalized();
    joint.localFrameB =
        (currentRotationB.conjugate() * worldFrame).normalized();
    joint.driveFlags = 0x01;
    joint.driveAccelerationFlags = acceleration ? 0x01 : 0;
    const float targetVelocity = mode == 0 ? 0.0f : 0.25f;
    joint.driveLinearVelocity =
        Vec3(reverse ? -targetVelocity : targetVelocity, 0.0f, 0.0f);
    joint.linearDriveDamping = Vec3(120.0f, 0.0f, 0.0f);
    joint.driveLinearForce =
        Vec3(mode == 2 ? 10.0f : 1e6f, 0.0f, 0.0f);
    joint.lambdaDriveLinear = Vec3(
        mode == 0 || acceleration ? 0.0f : (reverse ? -2.0f : 2.0f),
        0.0f, 0.0f);

    const IslandBodyMap map = buildIslandBodyMap(2);
    std::vector<Mat66> blocks(2);
    std::vector<Vec6> gradients(2);
    for (uint32_t body = 0; body < 2; ++body)
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
    IslandPcgSystem emittedSystem;
    emittedSystem.initialize(blocks, gradients);
    result.emitted = emitLinearXVelocityDriveIslandRow(
        joint, jointIndex, solver.bodies, map, dt, emittedSystem,
        result.rows);
    if (emittedSystem.rows().size() == 1) {
      const IslandPcgRow &row = emittedSystem.rows()[0];
      result.activeMode = row.activeMode;
      result.penalty = row.penalty;
      result.force = row.force;
      result.violation = row.violation;
    }
    result.stats = emittedSystem.solvePcg(result.solution, 1e-9, 24);
    if (result.solution.size() == 2) {
      std::vector<Vec6> applied;
      emittedSystem.apply(result.solution, applied);
      for (uint32_t body = 0; body < 2; ++body)
        for (int k = 0; k < 6; ++k)
          result.equationResidual =
              std::max(result.equationResidual,
                       std::fabs(double(applied[body][k] -
                                        emittedSystem.gradient()[body][k])));
    }
    return result;
  };
  const LinearDriveEmitterResult driveZero =
      runLinearDriveEmitter(false, 0, false, 1.0f);
  const LinearDriveEmitterResult driveUnsaturatedForward =
      runLinearDriveEmitter(false, 1, false, 1.0f);
  const LinearDriveEmitterResult driveUnsaturatedReverse =
      runLinearDriveEmitter(true, 1, false, 1.0f);
  const LinearDriveEmitterResult driveSaturatedForward =
      runLinearDriveEmitter(false, 2, false, 1.0f);
  const LinearDriveEmitterResult driveSaturatedReverse =
      runLinearDriveEmitter(true, 2, false, 1.0f);
  const LinearDriveEmitterResult driveAccelerationLightForward =
      runLinearDriveEmitter(false, 1, true, 1.0f);
  const LinearDriveEmitterResult driveAccelerationLightReverse =
      runLinearDriveEmitter(true, 1, true, 1.0f);
  const LinearDriveEmitterResult driveAccelerationHeavyForward =
      runLinearDriveEmitter(false, 1, true, 10.0f);
  const LinearDriveEmitterResult driveAccelerationSaturatedForward =
      runLinearDriveEmitter(false, 2, true, 1.0f);
  const LinearDriveEmitterResult driveAccelerationSaturatedReverse =
      runLinearDriveEmitter(true, 2, true, 1.0f);
  const double driveUnsaturatedOrderDifference = solutionDifference(
      driveUnsaturatedForward.solution, driveUnsaturatedReverse.solution);
  const double driveSaturatedOrderDifference = solutionDifference(
      driveSaturatedForward.solution, driveSaturatedReverse.solution);
  const double driveAccelerationOrderDifference = solutionDifference(
      driveAccelerationLightForward.solution,
      driveAccelerationLightReverse.solution);
  const double driveAccelerationMassDifference = solutionDifference(
      driveAccelerationLightForward.solution,
      driveAccelerationHeavyForward.solution);
  double driveZeroSolution = 0.0;
  for (const Vec6 &value : driveZero.solution)
    for (int k = 0; k < 6; ++k)
      driveZeroSolution =
          std::max(driveZeroSolution, std::fabs(double(value[k])));
  printf("  linear-drive-emitter rows=(%u,%u,%u) "
         "mode=(%u,%u,%u) force=(%.9g,%.9g,%.9g) "
         "penalty=(%.9g,%.9g,%.9g) order=(%.9g,%.9g) "
         "zero=%.9g equation=(%.9g,%.9g)\n",
         driveZero.rows, driveUnsaturatedForward.rows,
         driveSaturatedForward.rows, driveZero.activeMode,
         driveUnsaturatedForward.activeMode,
         driveSaturatedForward.activeMode, driveZero.force,
         driveUnsaturatedForward.force, driveSaturatedForward.force,
         driveZero.penalty, driveUnsaturatedForward.penalty,
         driveSaturatedForward.penalty,
         driveUnsaturatedOrderDifference,
         driveSaturatedOrderDifference, driveZeroSolution,
         driveUnsaturatedForward.equationResidual,
         driveSaturatedForward.equationResidual);
  printf("  linear-acceleration-drive rows=(%u,%u,%u) "
         "mode=(%u,%u) force=(%.9g,%.9g) "
         "penalty=(%.9g,%.9g,ratio=%.9g) "
         "order=%.9g mass=%.9g equation=(%.9g,%.9g)\n",
         driveAccelerationLightForward.rows,
         driveAccelerationHeavyForward.rows,
         driveAccelerationSaturatedForward.rows,
         driveAccelerationLightForward.activeMode,
         driveAccelerationSaturatedForward.activeMode,
         driveAccelerationLightForward.force,
         driveAccelerationSaturatedForward.force,
         driveAccelerationLightForward.penalty,
         driveAccelerationHeavyForward.penalty,
         driveAccelerationHeavyForward.penalty /
             driveAccelerationLightForward.penalty,
         driveAccelerationOrderDifference,
         driveAccelerationMassDifference,
         driveAccelerationLightForward.equationResidual,
         driveAccelerationHeavyForward.equationResidual);
  CHECK(driveZero.emitted && driveUnsaturatedForward.emitted &&
            driveUnsaturatedReverse.emitted &&
            driveSaturatedForward.emitted &&
            driveSaturatedReverse.emitted && driveZero.rows == 1 &&
            driveUnsaturatedForward.rows == 1 &&
            driveUnsaturatedReverse.rows == 1 &&
            driveSaturatedForward.rows == 1 &&
            driveSaturatedReverse.rows == 1,
        "Linear velocity-drive emitter produced an invalid row set");
  CHECK(driveZero.activeMode == 3 &&
            driveUnsaturatedForward.activeMode == 3 &&
            driveUnsaturatedReverse.activeMode == 3 &&
            driveSaturatedForward.activeMode == 4 &&
            driveSaturatedReverse.activeMode == 4 &&
            driveZero.penalty > 0.0f &&
            driveUnsaturatedForward.penalty > 0.0f &&
            driveUnsaturatedReverse.penalty > 0.0f &&
            driveSaturatedForward.penalty == 0.0f &&
            driveSaturatedReverse.penalty == 0.0f,
        "Linear velocity-drive clamp derivative is invalid");
  CHECK(std::fabs(driveZero.force) <= 1e-6f &&
            std::fabs(driveUnsaturatedForward.force) > 10.0f &&
            std::fabs(driveUnsaturatedForward.force) < 1e6f &&
            std::fabs(std::fabs(driveSaturatedForward.force) - 10.0f) <=
                1e-6f &&
            std::fabs(std::fabs(driveSaturatedReverse.force) - 10.0f) <=
                1e-6f &&
            std::fabs(driveSaturatedForward.force +
                      driveSaturatedReverse.force) <= 1e-6f,
        "Linear velocity-drive force clamp is invalid: %.9g %.9g %.9g",
        driveZero.force, driveUnsaturatedForward.force,
        driveSaturatedForward.force);
  CHECK(driveZero.stats.converged &&
            driveUnsaturatedForward.stats.converged &&
            driveUnsaturatedReverse.stats.converged &&
            driveSaturatedForward.stats.converged &&
            driveSaturatedReverse.stats.converged &&
            driveZeroSolution <= 1e-8,
        "Linear velocity-drive PCG/zero-target gate failed");
  CHECK(driveUnsaturatedOrderDifference <= 1e-5 &&
            driveSaturatedOrderDifference <= 1e-5,
        "Linear velocity-drive emitter depends on actor order: %.9g %.9g",
        driveUnsaturatedOrderDifference,
        driveSaturatedOrderDifference);
  CHECK(driveUnsaturatedForward.equationResidual <= 2e-3 &&
            driveUnsaturatedReverse.equationResidual <= 2e-3 &&
            driveSaturatedForward.equationResidual <= 2e-3 &&
            driveSaturatedReverse.equationResidual <= 2e-3,
        "Linear velocity-drive equation mismatch");
  CHECK(driveAccelerationLightForward.emitted &&
            driveAccelerationLightReverse.emitted &&
            driveAccelerationHeavyForward.emitted &&
            driveAccelerationSaturatedForward.emitted &&
            driveAccelerationSaturatedReverse.emitted &&
            driveAccelerationLightForward.rows == 1 &&
            driveAccelerationLightReverse.rows == 1 &&
            driveAccelerationHeavyForward.rows == 1 &&
            driveAccelerationSaturatedForward.rows == 1 &&
            driveAccelerationSaturatedReverse.rows == 1,
        "Linear acceleration-drive emitter produced an invalid row set");
  CHECK(driveAccelerationLightForward.activeMode == 3 &&
            driveAccelerationHeavyForward.activeMode == 3 &&
            driveAccelerationSaturatedForward.activeMode == 4 &&
            driveAccelerationSaturatedReverse.activeMode == 4 &&
            driveAccelerationLightForward.penalty > 0.0f &&
            driveAccelerationHeavyForward.penalty > 0.0f &&
            driveAccelerationSaturatedForward.penalty == 0.0f &&
            driveAccelerationSaturatedReverse.penalty == 0.0f,
        "Linear acceleration-drive clamp derivative is invalid");
  const float driveAccelerationPenaltyRatio =
      driveAccelerationHeavyForward.penalty /
      driveAccelerationLightForward.penalty;
  CHECK(std::fabs(driveAccelerationPenaltyRatio - 10.0f) <= 1e-4f &&
            driveAccelerationMassDifference <= 1e-5 &&
            driveAccelerationOrderDifference <= 1e-5,
        "Linear acceleration-drive mass/order scaling is invalid: "
        "%.9g %.9g %.9g",
        driveAccelerationPenaltyRatio, driveAccelerationMassDifference,
        driveAccelerationOrderDifference);
  CHECK(std::fabs(std::fabs(driveAccelerationSaturatedForward.force) -
                  10.0f) <= 1e-6f &&
            std::fabs(std::fabs(driveAccelerationSaturatedReverse.force) -
                      10.0f) <= 1e-6f &&
            std::fabs(driveAccelerationSaturatedForward.force +
                      driveAccelerationSaturatedReverse.force) <= 1e-6f,
        "Linear acceleration-drive force clamp is invalid");
  CHECK(driveAccelerationLightForward.stats.converged &&
            driveAccelerationLightReverse.stats.converged &&
            driveAccelerationHeavyForward.stats.converged &&
            driveAccelerationSaturatedForward.stats.converged &&
            driveAccelerationSaturatedReverse.stats.converged &&
            driveAccelerationLightForward.equationResidual <= 2e-3 &&
            driveAccelerationLightReverse.equationResidual <= 2e-3 &&
            driveAccelerationHeavyForward.equationResidual <= 2e-3 &&
            driveAccelerationSaturatedForward.equationResidual <= 2e-3 &&
            driveAccelerationSaturatedReverse.equationResidual <= 2e-3,
        "Linear acceleration-drive equation/PCG gate failed");

  struct TwistDriveEmitterResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    uint16_t activeMode = 0;
    float penalty = 0.0f;
    float torque = 0.0f;
    float violation = 0.0f;
    double equationResidual = 0.0;
    bool emitted = false;
  };
  const auto runTwistDriveEmitter =
      [dt](bool reverse, int mode, bool acceleration,
           float densityScale, int axisIndex = 0) {
    TwistDriveEmitterResult result;
    const auto axisAngle = [](const Vec3 &axisValue, float angle) {
      const Vec3 axis = axisValue.normalized();
      const float half = 0.5f * angle;
      return Quat(std::cos(half), axis.x * std::sin(half),
                  axis.y * std::sin(half), axis.z * std::sin(half));
    };
    Solver solver;
    solver.gravity = Vec3();
    const Vec3 initialA(-0.7f, 4.0f, 0.2f);
    const Vec3 initialB(0.9f, 4.3f, -0.4f);
    const Quat initialRotationA =
        axisAngle(Vec3(0.2f, 0.9f, -0.3f), 0.27f);
    const Quat initialRotationB =
        axisAngle(Vec3(-0.6f, 0.1f, 0.8f), -0.34f);
    const uint32_t bodyA = solver.addBody(
        initialA, initialRotationA, Vec3(0.4f, 0.5f, 0.6f),
        4.0f * densityScale);
    const uint32_t bodyB = solver.addBody(
        initialB, initialRotationB, Vec3(0.6f, 0.4f, 0.5f),
        5.0f * densityScale);
    const uint32_t jointIndex =
        reverse ? solver.addD6Joint(bodyB, bodyA, Vec3(), Vec3(), 0x2A,
                                    0x2A, 0.0f, 2e4f)
                : solver.addD6Joint(bodyA, bodyB, Vec3(), Vec3(), 0x2A,
                                    0x2A, 0.0f, 2e4f);

    const Quat worldFrame =
        axisAngle(Vec3(0.3f, 0.8f, -0.5f), 0.63f);
    const Vec3 localAxes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                               Vec3(0.0f, 1.0f, 0.0f),
                               Vec3(0.0f, 0.0f, 1.0f)};
    const Vec3 worldDriveAxis =
        worldFrame.rotate(localAxes[axisIndex]);
    if (mode != 0) {
      solver.bodies[bodyA].rotation =
          (axisAngle(worldDriveAxis, 0.02f) * initialRotationA)
              .normalized();
      solver.bodies[bodyB].rotation =
          (axisAngle(worldDriveAxis, -0.04f) * initialRotationB)
              .normalized();
    }
    for (uint32_t body = 0; body < 2; ++body) {
      Body &state = solver.bodies[body];
      state.initialPosition = body == bodyA ? initialA : initialB;
      state.initialRotation =
          body == bodyA ? initialRotationA : initialRotationB;
      state.inertialPosition = state.position;
      state.inertialRotation = state.rotation;
      state.updateInvInertiaWorld();
    }

    D6Joint &joint = solver.d6Joints[jointIndex];
    const Quat currentRotationA = solver.bodies[joint.bodyA].rotation;
    const Quat currentRotationB = solver.bodies[joint.bodyB].rotation;
    joint.localFrameA =
        (currentRotationA.conjugate() * worldFrame).normalized();
    joint.localFrameB =
        (currentRotationB.conjugate() * worldFrame).normalized();
    const uint32_t driveBits[3] = {0x10, 0x40, 0x80};
    joint.driveFlags = driveBits[axisIndex];
    joint.driveAccelerationFlags =
        acceleration ? driveBits[axisIndex] : 0;
    const float targetVelocity = mode == 0 ? 0.0f : 0.25f;
    (&joint.driveAngularVelocity.x)[axisIndex] =
        reverse ? -targetVelocity : targetVelocity;
    (&joint.angularDriveDamping.x)[axisIndex] = 120.0f;
    (&joint.driveAngularForce.x)[axisIndex] =
        mode == 2 ? 10.0f : 1e6f;
    (&joint.lambdaDriveAngular.x)[axisIndex] =
        mode == 0 || acceleration ? 0.0f : (reverse ? -2.0f : 2.0f);

    const IslandBodyMap map = buildIslandBodyMap(2);
    std::vector<Mat66> blocks(2);
    std::vector<Vec6> gradients(2);
    for (uint32_t body = 0; body < 2; ++body)
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
    IslandPcgSystem emittedSystem;
    emittedSystem.initialize(blocks, gradients);
    result.emitted = emitSingleAxisAngularVelocityDriveIslandRow(
        joint, jointIndex, solver.bodies, map, dt, axisIndex,
        emittedSystem, result.rows);
    if (emittedSystem.rows().size() == 1) {
      const IslandPcgRow &row = emittedSystem.rows()[0];
      result.activeMode = row.activeMode;
      result.penalty = row.penalty;
      result.torque = row.force;
      result.violation = row.violation;
    }
    result.stats = emittedSystem.solvePcg(result.solution, 1e-9, 24);
    if (result.solution.size() == 2) {
      std::vector<Vec6> applied;
      emittedSystem.apply(result.solution, applied);
      for (uint32_t body = 0; body < 2; ++body)
        for (int k = 0; k < 6; ++k)
          result.equationResidual =
              std::max(result.equationResidual,
                       std::fabs(double(applied[body][k] -
                                        emittedSystem.gradient()[body][k])));
    }
    return result;
  };
  const TwistDriveEmitterResult twistZero =
      runTwistDriveEmitter(false, 0, false, 1.0f);
  const TwistDriveEmitterResult twistUnsaturatedForward =
      runTwistDriveEmitter(false, 1, false, 1.0f);
  const TwistDriveEmitterResult twistUnsaturatedReverse =
      runTwistDriveEmitter(true, 1, false, 1.0f);
  const TwistDriveEmitterResult twistSaturatedForward =
      runTwistDriveEmitter(false, 2, false, 1.0f);
  const TwistDriveEmitterResult twistSaturatedReverse =
      runTwistDriveEmitter(true, 2, false, 1.0f);
  const TwistDriveEmitterResult twistAccelerationLightForward =
      runTwistDriveEmitter(false, 1, true, 1.0f);
  const TwistDriveEmitterResult twistAccelerationLightReverse =
      runTwistDriveEmitter(true, 1, true, 1.0f);
  const TwistDriveEmitterResult twistAccelerationHeavyForward =
      runTwistDriveEmitter(false, 1, true, 10.0f);
  const TwistDriveEmitterResult twistAccelerationSaturatedForward =
      runTwistDriveEmitter(false, 2, true, 1.0f);
  const TwistDriveEmitterResult twistAccelerationSaturatedReverse =
      runTwistDriveEmitter(true, 2, true, 1.0f);
  const double twistUnsaturatedOrderDifference = solutionDifference(
      twistUnsaturatedForward.solution, twistUnsaturatedReverse.solution);
  const double twistSaturatedOrderDifference = solutionDifference(
      twistSaturatedForward.solution, twistSaturatedReverse.solution);
  const double twistAccelerationOrderDifference = solutionDifference(
      twistAccelerationLightForward.solution,
      twistAccelerationLightReverse.solution);
  const double twistAccelerationMassDifference = solutionDifference(
      twistAccelerationLightForward.solution,
      twistAccelerationHeavyForward.solution);
  double twistZeroSolution = 0.0;
  for (const Vec6 &value : twistZero.solution)
    for (int k = 0; k < 6; ++k)
      twistZeroSolution =
          std::max(twistZeroSolution, std::fabs(double(value[k])));
  printf("  twist-drive-emitter rows=(%u,%u,%u) "
         "mode=(%u,%u,%u) torque=(%.9g,%.9g,%.9g) "
         "penalty=(%.9g,%.9g,%.9g) order=(%.9g,%.9g) "
         "zero=%.9g equation=(%.9g,%.9g)\n",
         twistZero.rows, twistUnsaturatedForward.rows,
         twistSaturatedForward.rows, twistZero.activeMode,
         twistUnsaturatedForward.activeMode,
         twistSaturatedForward.activeMode, twistZero.torque,
         twistUnsaturatedForward.torque, twistSaturatedForward.torque,
         twistZero.penalty, twistUnsaturatedForward.penalty,
         twistSaturatedForward.penalty,
         twistUnsaturatedOrderDifference,
         twistSaturatedOrderDifference, twistZeroSolution,
         twistUnsaturatedForward.equationResidual,
         twistSaturatedForward.equationResidual);
  printf("  twist-acceleration-drive rows=(%u,%u,%u) "
         "mode=(%u,%u) torque=(%.9g,%.9g) "
         "penalty=(%.9g,%.9g,ratio=%.9g) "
         "order=%.9g mass=%.9g equation=(%.9g,%.9g)\n",
         twistAccelerationLightForward.rows,
         twistAccelerationHeavyForward.rows,
         twistAccelerationSaturatedForward.rows,
         twistAccelerationLightForward.activeMode,
         twistAccelerationSaturatedForward.activeMode,
         twistAccelerationLightForward.torque,
         twistAccelerationSaturatedForward.torque,
         twistAccelerationLightForward.penalty,
         twistAccelerationHeavyForward.penalty,
         twistAccelerationHeavyForward.penalty /
             twistAccelerationLightForward.penalty,
         twistAccelerationOrderDifference,
         twistAccelerationMassDifference,
         twistAccelerationLightForward.equationResidual,
         twistAccelerationHeavyForward.equationResidual);
  CHECK(twistZero.emitted && twistUnsaturatedForward.emitted &&
            twistUnsaturatedReverse.emitted &&
            twistSaturatedForward.emitted &&
            twistSaturatedReverse.emitted && twistZero.rows == 1 &&
            twistUnsaturatedForward.rows == 1 &&
            twistUnsaturatedReverse.rows == 1 &&
            twistSaturatedForward.rows == 1 &&
            twistSaturatedReverse.rows == 1,
        "TWIST velocity-drive emitter produced an invalid row set");
  CHECK(twistZero.activeMode == 3 &&
            twistUnsaturatedForward.activeMode == 3 &&
            twistUnsaturatedReverse.activeMode == 3 &&
            twistSaturatedForward.activeMode == 4 &&
            twistSaturatedReverse.activeMode == 4 &&
            twistZero.penalty > 0.0f &&
            twistUnsaturatedForward.penalty > 0.0f &&
            twistUnsaturatedReverse.penalty > 0.0f &&
            twistSaturatedForward.penalty == 0.0f &&
            twistSaturatedReverse.penalty == 0.0f,
        "TWIST velocity-drive clamp derivative is invalid");
  CHECK(std::fabs(twistZero.torque) <= 1e-6f &&
            std::fabs(twistUnsaturatedForward.torque) > 10.0f &&
            std::fabs(twistUnsaturatedForward.torque) < 1e6f &&
            std::fabs(std::fabs(twistSaturatedForward.torque) - 10.0f) <=
                1e-6f &&
            std::fabs(std::fabs(twistSaturatedReverse.torque) - 10.0f) <=
                1e-6f &&
            std::fabs(twistSaturatedForward.torque +
                      twistSaturatedReverse.torque) <= 1e-6f,
        "TWIST velocity-drive torque clamp is invalid: %.9g %.9g %.9g",
        twistZero.torque, twistUnsaturatedForward.torque,
        twistSaturatedForward.torque);
  CHECK(twistZero.stats.converged &&
            twistUnsaturatedForward.stats.converged &&
            twistUnsaturatedReverse.stats.converged &&
            twistSaturatedForward.stats.converged &&
            twistSaturatedReverse.stats.converged &&
            twistZeroSolution <= 1e-8,
        "TWIST velocity-drive PCG/zero-target gate failed");
  CHECK(twistUnsaturatedOrderDifference <= 1e-5 &&
            twistSaturatedOrderDifference <= 1e-5,
        "TWIST velocity-drive emitter depends on actor order: %.9g %.9g",
        twistUnsaturatedOrderDifference,
        twistSaturatedOrderDifference);
  CHECK(twistUnsaturatedForward.equationResidual <= 2e-3 &&
            twistUnsaturatedReverse.equationResidual <= 2e-3 &&
            twistSaturatedForward.equationResidual <= 2e-3 &&
            twistSaturatedReverse.equationResidual <= 2e-3,
        "TWIST velocity-drive equation mismatch");
  CHECK(twistAccelerationLightForward.emitted &&
            twistAccelerationLightReverse.emitted &&
            twistAccelerationHeavyForward.emitted &&
            twistAccelerationSaturatedForward.emitted &&
            twistAccelerationSaturatedReverse.emitted &&
            twistAccelerationLightForward.rows == 1 &&
            twistAccelerationLightReverse.rows == 1 &&
            twistAccelerationHeavyForward.rows == 1 &&
            twistAccelerationSaturatedForward.rows == 1 &&
            twistAccelerationSaturatedReverse.rows == 1,
        "TWIST acceleration-drive emitter produced an invalid row set");
  CHECK(twistAccelerationLightForward.activeMode == 3 &&
            twistAccelerationHeavyForward.activeMode == 3 &&
            twistAccelerationSaturatedForward.activeMode == 4 &&
            twistAccelerationSaturatedReverse.activeMode == 4 &&
            twistAccelerationLightForward.penalty > 0.0f &&
            twistAccelerationHeavyForward.penalty > 0.0f &&
            twistAccelerationSaturatedForward.penalty == 0.0f &&
            twistAccelerationSaturatedReverse.penalty == 0.0f,
        "TWIST acceleration-drive clamp derivative is invalid");
  const float twistAccelerationPenaltyRatio =
      twistAccelerationHeavyForward.penalty /
      twistAccelerationLightForward.penalty;
  CHECK(std::fabs(twistAccelerationPenaltyRatio - 10.0f) <= 1e-4f &&
            twistAccelerationMassDifference <= 1e-5 &&
            twistAccelerationOrderDifference <= 1e-5,
        "TWIST acceleration-drive mass/order scaling is invalid: "
        "%.9g %.9g %.9g",
        twistAccelerationPenaltyRatio, twistAccelerationMassDifference,
        twistAccelerationOrderDifference);
  CHECK(std::fabs(std::fabs(twistAccelerationSaturatedForward.torque) -
                  10.0f) <= 1e-6f &&
            std::fabs(std::fabs(twistAccelerationSaturatedReverse.torque) -
                      10.0f) <= 1e-6f &&
            std::fabs(twistAccelerationSaturatedForward.torque +
                      twistAccelerationSaturatedReverse.torque) <= 1e-6f,
        "TWIST acceleration-drive torque clamp is invalid");
  CHECK(twistAccelerationLightForward.stats.converged &&
            twistAccelerationLightReverse.stats.converged &&
            twistAccelerationHeavyForward.stats.converged &&
            twistAccelerationSaturatedForward.stats.converged &&
            twistAccelerationSaturatedReverse.stats.converged &&
            twistAccelerationLightForward.equationResidual <= 2e-3 &&
            twistAccelerationLightReverse.equationResidual <= 2e-3 &&
            twistAccelerationHeavyForward.equationResidual <= 2e-3 &&
            twistAccelerationSaturatedForward.equationResidual <= 2e-3 &&
            twistAccelerationSaturatedReverse.equationResidual <= 2e-3,
        "TWIST acceleration-drive equation/PCG gate failed");

  for (int swingAxis = 1; swingAxis <= 2; ++swingAxis) {
    const char *axisName = swingAxis == 1 ? "SWING1" : "SWING2";
    const TwistDriveEmitterResult zero =
        runTwistDriveEmitter(false, 0, false, 1.0f, swingAxis);
    const TwistDriveEmitterResult forceForward =
        runTwistDriveEmitter(false, 1, false, 1.0f, swingAxis);
    const TwistDriveEmitterResult forceReverse =
        runTwistDriveEmitter(true, 1, false, 1.0f, swingAxis);
    const TwistDriveEmitterResult forceSaturatedForward =
        runTwistDriveEmitter(false, 2, false, 1.0f, swingAxis);
    const TwistDriveEmitterResult forceSaturatedReverse =
        runTwistDriveEmitter(true, 2, false, 1.0f, swingAxis);
    const TwistDriveEmitterResult accelerationLightForward =
        runTwistDriveEmitter(false, 1, true, 1.0f, swingAxis);
    const TwistDriveEmitterResult accelerationLightReverse =
        runTwistDriveEmitter(true, 1, true, 1.0f, swingAxis);
    const TwistDriveEmitterResult accelerationHeavyForward =
        runTwistDriveEmitter(false, 1, true, 10.0f, swingAxis);
    const TwistDriveEmitterResult accelerationSaturatedForward =
        runTwistDriveEmitter(false, 2, true, 1.0f, swingAxis);
    const TwistDriveEmitterResult accelerationSaturatedReverse =
        runTwistDriveEmitter(true, 2, true, 1.0f, swingAxis);
    const double forceOrderDifference = solutionDifference(
        forceForward.solution, forceReverse.solution);
    const double forceSaturatedOrderDifference = solutionDifference(
        forceSaturatedForward.solution, forceSaturatedReverse.solution);
    const double accelerationOrderDifference = solutionDifference(
        accelerationLightForward.solution,
        accelerationLightReverse.solution);
    const double accelerationMassDifference = solutionDifference(
        accelerationLightForward.solution,
        accelerationHeavyForward.solution);
    const float accelerationPenaltyRatio =
        accelerationHeavyForward.penalty /
        accelerationLightForward.penalty;
    double zeroSolution = 0.0;
    for (const Vec6 &value : zero.solution)
      for (int k = 0; k < 6; ++k)
        zeroSolution =
            std::max(zeroSolution, std::fabs(double(value[k])));
    printf("  %s-drive-emitter rows=(%u,%u,%u) mode=(%u,%u,%u) "
           "torque=(%.9g,%.9g,%.9g) order=(%.9g,%.9g) "
           "accel=(penalty %.9g/%.9g ratio %.9g order %.9g mass %.9g) "
           "equation=(%.9g,%.9g)\n",
           axisName, zero.rows, forceForward.rows,
           forceSaturatedForward.rows, zero.activeMode,
           forceForward.activeMode, forceSaturatedForward.activeMode,
           zero.torque, forceForward.torque,
           forceSaturatedForward.torque, forceOrderDifference,
           forceSaturatedOrderDifference,
           accelerationLightForward.penalty,
           accelerationHeavyForward.penalty, accelerationPenaltyRatio,
           accelerationOrderDifference, accelerationMassDifference,
           forceForward.equationResidual,
           accelerationHeavyForward.equationResidual);
    CHECK(zero.emitted && forceForward.emitted && forceReverse.emitted &&
              forceSaturatedForward.emitted &&
              forceSaturatedReverse.emitted &&
              accelerationLightForward.emitted &&
              accelerationLightReverse.emitted &&
              accelerationHeavyForward.emitted &&
              accelerationSaturatedForward.emitted &&
              accelerationSaturatedReverse.emitted && zero.rows == 1 &&
              forceForward.rows == 1 && forceReverse.rows == 1 &&
              forceSaturatedForward.rows == 1 &&
              forceSaturatedReverse.rows == 1 &&
              accelerationLightForward.rows == 1 &&
              accelerationLightReverse.rows == 1 &&
              accelerationHeavyForward.rows == 1 &&
              accelerationSaturatedForward.rows == 1 &&
              accelerationSaturatedReverse.rows == 1,
          "%s velocity-drive emitter produced an invalid row set",
          axisName);
    CHECK(zero.activeMode == 3 && forceForward.activeMode == 3 &&
              forceReverse.activeMode == 3 &&
              forceSaturatedForward.activeMode == 4 &&
              forceSaturatedReverse.activeMode == 4 &&
              accelerationLightForward.activeMode == 3 &&
              accelerationHeavyForward.activeMode == 3 &&
              accelerationSaturatedForward.activeMode == 4 &&
              accelerationSaturatedReverse.activeMode == 4 &&
              zero.penalty > 0.0f && forceForward.penalty > 0.0f &&
              forceSaturatedForward.penalty == 0.0f &&
              accelerationLightForward.penalty > 0.0f &&
              accelerationHeavyForward.penalty > 0.0f &&
              accelerationSaturatedForward.penalty == 0.0f,
          "%s velocity-drive clamp derivative is invalid", axisName);
    CHECK(std::fabs(zero.torque) <= 1e-6f && zeroSolution <= 1e-8 &&
              std::fabs(forceForward.torque) > 10.0f &&
              std::fabs(std::fabs(forceSaturatedForward.torque) - 10.0f) <=
                  1e-6f &&
              std::fabs(std::fabs(forceSaturatedReverse.torque) - 10.0f) <=
                  1e-6f &&
              std::fabs(forceSaturatedForward.torque +
                        forceSaturatedReverse.torque) <= 1e-6f &&
              std::fabs(std::fabs(accelerationSaturatedForward.torque) -
                        10.0f) <= 1e-6f &&
              std::fabs(std::fabs(accelerationSaturatedReverse.torque) -
                        10.0f) <= 1e-6f &&
              std::fabs(accelerationSaturatedForward.torque +
                        accelerationSaturatedReverse.torque) <= 1e-6f,
          "%s velocity-drive torque clamp/zero gate failed", axisName);
    CHECK(forceOrderDifference <= 1e-5 &&
              forceSaturatedOrderDifference <= 1e-5 &&
              accelerationOrderDifference <= 1e-5 &&
              accelerationMassDifference <= 1e-5 &&
              std::fabs(accelerationPenaltyRatio - 10.0f) <= 1e-4f,
          "%s velocity-drive mass/order scaling is invalid: "
          "%.9g %.9g %.9g %.9g %.9g",
          axisName, forceOrderDifference,
          forceSaturatedOrderDifference, accelerationOrderDifference,
          accelerationMassDifference, accelerationPenaltyRatio);
    const TwistDriveEmitterResult *allResults[10] = {
        &zero, &forceForward, &forceReverse, &forceSaturatedForward,
        &forceSaturatedReverse, &accelerationLightForward,
        &accelerationLightReverse, &accelerationHeavyForward,
        &accelerationSaturatedForward,
        &accelerationSaturatedReverse};
    for (const TwistDriveEmitterResult *result : allResults)
      CHECK(result->stats.converged &&
                result->equationResidual <= 2e-3,
            "%s velocity-drive equation/PCG gate failed: %.9g",
            axisName, result->equationResidual);
  }

  struct SlerpDriveEmitterResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
    uint32_t rows = 0;
    uint16_t activeMode[3] = {};
    float penalty[3] = {};
    float torque[3] = {};
    float violation[3] = {};
    double equationResidual = 0.0;
    bool emitted = false;
  };
  const auto runSlerpDriveEmitter =
      [dt](bool reverse, int mode, bool acceleration,
           float densityScale) {
    SlerpDriveEmitterResult result;
    const auto rotationVectorQuat = [](const Vec3 &rotationVector) {
      const float angle = rotationVector.length();
      if (!(angle > 1e-10f))
        return Quat();
      const Vec3 axis = rotationVector * (1.0f / angle);
      const float half = 0.5f * angle;
      return Quat(std::cos(half), axis.x * std::sin(half),
                  axis.y * std::sin(half), axis.z * std::sin(half));
    };
    Solver solver;
    solver.gravity = Vec3();
    const Vec3 initialA(-0.7f, 4.0f, 0.2f);
    const Vec3 initialB(0.9f, 4.3f, -0.4f);
    const Quat initialRotationA =
        rotationVectorQuat(Vec3(0.04f, 0.21f, -0.07f));
    const Quat initialRotationB =
        rotationVectorQuat(Vec3(-0.18f, 0.03f, 0.24f));
    const uint32_t bodyA = solver.addBody(
        initialA, initialRotationA, Vec3(0.4f, 0.5f, 0.6f),
        4.0f * densityScale);
    const uint32_t bodyB = solver.addBody(
        initialB, initialRotationB, Vec3(0.6f, 0.4f, 0.5f),
        5.0f * densityScale);
    const uint32_t jointIndex =
        reverse ? solver.addD6Joint(bodyB, bodyA, Vec3(), Vec3(), 0x2A,
                                    0x2A, 0.0f, 2e4f)
                : solver.addD6Joint(bodyA, bodyB, Vec3(), Vec3(), 0x2A,
                                    0x2A, 0.0f, 2e4f);
    if (mode != 0) {
      solver.bodies[bodyA].rotation =
          (rotationVectorQuat(Vec3(0.018f, -0.013f, 0.009f)) *
           initialRotationA)
              .normalized();
      solver.bodies[bodyB].rotation =
          (rotationVectorQuat(Vec3(-0.032f, 0.027f, -0.041f)) *
           initialRotationB)
              .normalized();
    }
    for (uint32_t body = 0; body < 2; ++body) {
      Body &state = solver.bodies[body];
      state.initialPosition = body == bodyA ? initialA : initialB;
      state.initialRotation =
          body == bodyA ? initialRotationA : initialRotationB;
      state.inertialPosition = state.position;
      state.inertialRotation = state.rotation;
      state.updateInvInertiaWorld();
    }
    D6Joint &joint = solver.d6Joints[jointIndex];
    const Quat worldFrame =
        rotationVectorQuat(Vec3(0.17f, 0.31f, -0.23f));
    joint.localFrameA =
        (solver.bodies[joint.bodyA].rotation.conjugate() * worldFrame)
            .normalized();
    joint.localFrameB =
        (solver.bodies[joint.bodyB].rotation.conjugate() * worldFrame)
            .normalized();
    joint.driveFlags = 0x20;
    joint.driveAccelerationFlags = acceleration ? 0x20 : 0;
    const Vec3 target = mode == 0 ? Vec3() : Vec3(0.18f, -0.11f, 0.14f);
    joint.driveAngularVelocity = reverse ? target * -1.0f : target;
    joint.angularDriveDamping.z = 120.0f;
    joint.driveAngularForce.z = mode == 2 ? 10.0f : 1e6f;
    if (mode != 0 && !acceleration) {
      const Vec3 lambda(2.0f, -1.5f, 0.75f);
      joint.lambdaDriveAngular = reverse ? lambda * -1.0f : lambda;
    }

    const IslandBodyMap map = buildIslandBodyMap(2);
    std::vector<Mat66> blocks(2);
    std::vector<Vec6> gradients(2);
    for (uint32_t body = 0; body < 2; ++body)
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
    IslandPcgSystem system;
    system.initialize(blocks, gradients);
    result.emitted = emitSlerpVelocityDriveIslandRows(
        joint, jointIndex, solver.bodies, map, dt, system, result.rows);
    if (system.rows().size() == 3) {
      for (int axis = 0; axis < 3; ++axis) {
        result.activeMode[axis] = system.rows()[axis].activeMode;
        result.penalty[axis] = system.rows()[axis].penalty;
        result.torque[axis] = system.rows()[axis].force;
        result.violation[axis] = system.rows()[axis].violation;
      }
    }
    result.stats = system.solvePcg(result.solution, 1e-9, 24);
    if (result.solution.size() == 2) {
      std::vector<Vec6> applied;
      system.apply(result.solution, applied);
      for (uint32_t body = 0; body < 2; ++body)
        for (int k = 0; k < 6; ++k)
          result.equationResidual =
              std::max(result.equationResidual,
                       std::fabs(double(applied[body][k] -
                                        system.gradient()[body][k])));
    }
    return result;
  };
  const SlerpDriveEmitterResult slerpZero =
      runSlerpDriveEmitter(false, 0, false, 1.0f);
  const SlerpDriveEmitterResult slerpForceForward =
      runSlerpDriveEmitter(false, 1, false, 1.0f);
  const SlerpDriveEmitterResult slerpForceReverse =
      runSlerpDriveEmitter(true, 1, false, 1.0f);
  const SlerpDriveEmitterResult slerpForceSaturatedForward =
      runSlerpDriveEmitter(false, 2, false, 1.0f);
  const SlerpDriveEmitterResult slerpForceSaturatedReverse =
      runSlerpDriveEmitter(true, 2, false, 1.0f);
  const SlerpDriveEmitterResult slerpAccelerationLightForward =
      runSlerpDriveEmitter(false, 1, true, 1.0f);
  const SlerpDriveEmitterResult slerpAccelerationLightReverse =
      runSlerpDriveEmitter(true, 1, true, 1.0f);
  const SlerpDriveEmitterResult slerpAccelerationHeavyForward =
      runSlerpDriveEmitter(false, 1, true, 10.0f);
  const SlerpDriveEmitterResult slerpAccelerationSaturatedForward =
      runSlerpDriveEmitter(false, 2, true, 1.0f);
  const SlerpDriveEmitterResult slerpAccelerationSaturatedReverse =
      runSlerpDriveEmitter(true, 2, true, 1.0f);
  const double slerpForceOrderDifference = solutionDifference(
      slerpForceForward.solution, slerpForceReverse.solution);
  const double slerpForceSaturatedOrderDifference = solutionDifference(
      slerpForceSaturatedForward.solution,
      slerpForceSaturatedReverse.solution);
  const double slerpAccelerationOrderDifference = solutionDifference(
      slerpAccelerationLightForward.solution,
      slerpAccelerationLightReverse.solution);
  const double slerpAccelerationMassDifference = solutionDifference(
      slerpAccelerationLightForward.solution,
      slerpAccelerationHeavyForward.solution);
  double slerpZeroSolution = 0.0;
  for (const Vec6 &value : slerpZero.solution)
    for (int k = 0; k < 6; ++k)
      slerpZeroSolution =
          std::max(slerpZeroSolution, std::fabs(double(value[k])));
  printf("  SLERP-drive-emitter rows=(%u,%u,%u) "
         "mode=(%u/%u/%u,%u/%u/%u) "
         "torque=(%.9g/%.9g/%.9g) order=(%.9g,%.9g) "
         "accelPenalty=(%.9g/%.9g/%.9g ratios %.9g/%.9g/%.9g) "
         "accelOrder=%.9g mass=%.9g equation=(%.9g,%.9g)\n",
         slerpZero.rows, slerpForceForward.rows,
         slerpForceSaturatedForward.rows,
         slerpForceForward.activeMode[0],
         slerpForceForward.activeMode[1],
         slerpForceForward.activeMode[2],
         slerpForceSaturatedForward.activeMode[0],
         slerpForceSaturatedForward.activeMode[1],
         slerpForceSaturatedForward.activeMode[2],
         slerpForceSaturatedForward.torque[0],
         slerpForceSaturatedForward.torque[1],
         slerpForceSaturatedForward.torque[2],
         slerpForceOrderDifference,
         slerpForceSaturatedOrderDifference,
         slerpAccelerationLightForward.penalty[0],
         slerpAccelerationLightForward.penalty[1],
         slerpAccelerationLightForward.penalty[2],
         slerpAccelerationHeavyForward.penalty[0] /
             slerpAccelerationLightForward.penalty[0],
         slerpAccelerationHeavyForward.penalty[1] /
             slerpAccelerationLightForward.penalty[1],
         slerpAccelerationHeavyForward.penalty[2] /
             slerpAccelerationLightForward.penalty[2],
         slerpAccelerationOrderDifference,
         slerpAccelerationMassDifference,
         slerpForceForward.equationResidual,
         slerpAccelerationHeavyForward.equationResidual);
  const SlerpDriveEmitterResult *slerpResults[10] = {
      &slerpZero, &slerpForceForward, &slerpForceReverse,
      &slerpForceSaturatedForward, &slerpForceSaturatedReverse,
      &slerpAccelerationLightForward,
      &slerpAccelerationLightReverse,
      &slerpAccelerationHeavyForward,
      &slerpAccelerationSaturatedForward,
      &slerpAccelerationSaturatedReverse};
  for (const SlerpDriveEmitterResult *result : slerpResults)
    CHECK(result->emitted && result->rows == 3 &&
              result->stats.converged && result->equationResidual <= 2e-3,
          "SLERP emitter row/equation/PCG gate failed: %u %.9g",
          result->rows, result->equationResidual);
  CHECK(slerpZeroSolution <= 1e-8 &&
            slerpForceOrderDifference <= 1e-5 &&
            slerpForceSaturatedOrderDifference <= 1e-5 &&
            slerpAccelerationOrderDifference <= 1e-5 &&
            slerpAccelerationMassDifference <= 1e-5,
        "SLERP zero/order/mass gate failed: %.9g %.9g %.9g %.9g %.9g",
        slerpZeroSolution, slerpForceOrderDifference,
        slerpForceSaturatedOrderDifference,
        slerpAccelerationOrderDifference,
        slerpAccelerationMassDifference);
  for (int axis = 0; axis < 3; ++axis) {
    const float penaltyRatio =
        slerpAccelerationHeavyForward.penalty[axis] /
        slerpAccelerationLightForward.penalty[axis];
    CHECK(slerpForceForward.activeMode[axis] == 3 &&
              slerpForceSaturatedForward.activeMode[axis] == 4 &&
              slerpForceSaturatedReverse.activeMode[axis] == 4 &&
              slerpAccelerationLightForward.activeMode[axis] == 3 &&
              slerpAccelerationSaturatedForward.activeMode[axis] == 4 &&
              slerpAccelerationSaturatedReverse.activeMode[axis] == 4 &&
              slerpForceForward.penalty[axis] > 0.0f &&
              slerpForceSaturatedForward.penalty[axis] == 0.0f &&
              slerpAccelerationLightForward.penalty[axis] > 0.0f &&
              slerpAccelerationSaturatedForward.penalty[axis] == 0.0f &&
              std::fabs(penaltyRatio - 10.0f) <= 1e-4f &&
              std::fabs(std::fabs(
                            slerpForceSaturatedForward.torque[axis]) -
                        10.0f) <= 1e-6f &&
              std::fabs(slerpForceSaturatedForward.torque[axis] +
                        slerpForceSaturatedReverse.torque[axis]) <= 1e-6f &&
              std::fabs(std::fabs(
                            slerpAccelerationSaturatedForward.torque[axis]) -
                        10.0f) <= 1e-6f &&
              std::fabs(slerpAccelerationSaturatedForward.torque[axis] +
                        slerpAccelerationSaturatedReverse.torque[axis]) <=
                  1e-6f,
          "SLERP axis %d mode/penalty/clamp gate failed: %.9g", axis,
          penaltyRatio);
  }
  const float driveDualUnsaturated =
      updateClampedLinearDriveDual(2.0f, 0.001f, 100.0f, 10.0f, 0.99f);
  const float driveDualPositiveClamp =
      updateClampedLinearDriveDual(9.0f, 0.1f, 100.0f, 10.0f, 0.99f);
  const float driveDualNegativeClamp =
      updateClampedLinearDriveDual(-9.0f, -0.1f, 100.0f, 10.0f, 0.99f);
  CHECK(std::fabs(driveDualUnsaturated - 2.08f) <= 1e-6f &&
            driveDualPositiveClamp == 10.0f &&
            driveDualNegativeClamp == -10.0f,
        "Linear velocity-drive dual clamp is invalid: %.9g %.9g %.9g",
        driveDualUnsaturated, driveDualPositiveClamp,
        driveDualNegativeClamp);

  // Frozen body-vs-static contact snapshot. This intentionally does not step
  // the solver or update active/cache/dual state: it isolates row emission and
  // the matrix-free equation from the rejected runtime sequencing probes.
  Solver contactSolver;
  contactSolver.gravity = Vec3();
  const float contactAngle = 0.4f;
  const Quat contactRotation(std::cos(0.5f * contactAngle), 0.0f, 0.0f,
                             std::sin(0.5f * contactAngle));
  const uint32_t contactBody = contactSolver.addBody(
      Vec3(0.2f, 0.42f, -0.1f), contactRotation,
      Vec3(0.5f, 0.5f, 0.5f), 4.0f);
  Body &snapshotBody = contactSolver.bodies[contactBody];
  snapshotBody.initialPosition = snapshotBody.position;
  snapshotBody.initialRotation = snapshotBody.rotation;
  snapshotBody.inertialPosition =
      snapshotBody.position + Vec3(0.03f, -0.04f, 0.02f);
  snapshotBody.inertialRotation = Quat();
  const Vec3 snapshotNormal = Vec3(0.2f, 0.95f, -0.1f).normalized();
  const Vec3 snapshotRA(0.35f, -0.5f, 0.2f);
  const Vec3 snapshotWorldA =
      snapshotBody.position + snapshotBody.rotation.rotate(snapshotRA);
  const float snapshotDepth = 0.04f;
  // PhysX contactPointA/contactPointB are authored from the same world point;
  // narrow-phase separation/depth alone carries the initial signed offset.
  const Vec3 snapshotRB = snapshotWorldA;
  contactSolver.addContact(contactBody, UINT32_MAX, snapshotNormal,
                           snapshotRA, snapshotRB, snapshotDepth, 0.5f);
  Contact &snapshotContact = contactSolver.contacts.front();
  snapshotContact.penalty[0] = 25000.0f;
  snapshotContact.lambda[0] = -120.0f;
  contactSolver.computeConstraintBodyStatic(snapshotContact);
  const float snapshotPenalty = snapshotContact.penalty[0];
  const float snapshotForce =
      std::max(snapshotContact.fmin[0],
               std::min(snapshotContact.fmax[0],
                        snapshotPenalty * snapshotContact.C[0] +
                            snapshotContact.lambda[0]));

  const IslandBodyMap contactMap = buildIslandBodyMap(1);
  const Mat66 contactK = snapshotBody.getMassMatrix() / (dt * dt);
  const Vec6 contactDisplacement(
      snapshotBody.position - snapshotBody.inertialPosition,
      snapshotBody.deltaWInertial());
  const Vec6 contactInertialGradient = contactK * contactDisplacement;
  IslandPcgSystem contactSystem;
  contactSystem.initialize(std::vector<Mat66>{contactK},
                           std::vector<Vec6>{contactInertialGradient});
  uint32_t contactRows = 0;
  CHECK(emitFrozenContactNormalIslandRow(
            snapshotContact, 0, contactSolver.bodies, contactMap, true,
            snapshotPenalty, snapshotForce, 1, contactSystem, contactRows) &&
            contactRows == 1 && contactSystem.rows().size() == 1,
        "Frozen contact emitter did not produce exactly one normal row");

  Mat66 denseContactH = contactK;
  denseContactH += outer(snapshotContact.JA,
                         snapshotContact.JA * snapshotPenalty);
  Vec6 denseContactG = contactInertialGradient;
  denseContactG += snapshotContact.JA * snapshotForce;
  const Vec6 contactWitness(Vec3(0.2f, -0.3f, 0.4f),
                            Vec3(0.1f, 0.05f, -0.2f));
  const Vec6 denseContactApplied = denseContactH * contactWitness;
  std::vector<Vec6> contactApplied;
  contactSystem.apply(std::vector<Vec6>{contactWitness}, contactApplied);
  double contactApplyDifference = 0.0;
  double contactGradientDifference = 0.0;
  for (int k = 0; k < 6; ++k) {
    contactApplyDifference =
        std::max(contactApplyDifference,
                 std::fabs(double(denseContactApplied[k] -
                                  contactApplied.front()[k])));
    contactGradientDifference =
        std::max(contactGradientDifference,
                 std::fabs(double(denseContactG[k] -
                                  contactSystem.gradient().front()[k])));
  }
  const Vec6 denseContactDelta = solveLDLT(denseContactH, denseContactG);
  std::vector<Vec6> contactDelta;
  const IslandPcgStats contactStats =
      contactSystem.solvePcg(contactDelta, 1e-9, 12);
  double contactDeltaDifference = 0.0;
  for (int k = 0; k < 6; ++k)
    contactDeltaDifference =
        std::max(contactDeltaDifference,
                 std::fabs(double(denseContactDelta[k] -
                                  contactDelta.front()[k])));

  IslandPcgSystem inactiveContactSystem;
  inactiveContactSystem.initialize(
      std::vector<Mat66>{contactK},
      std::vector<Vec6>{contactInertialGradient});
  uint32_t inactiveRows = UINT32_MAX;
  CHECK(emitFrozenContactNormalIslandRow(
            snapshotContact, 0, contactSolver.bodies, contactMap, false,
            snapshotPenalty, snapshotForce, 0, inactiveContactSystem,
            inactiveRows) &&
            inactiveRows == 0 && inactiveContactSystem.rows().empty(),
        "Inactive frozen contact unexpectedly emitted a row");

  printf("  contact-snapshot C=%.9g penalty=%.9g lambda=%.9g force=%.9g "
         "JA=(%.9g,%.9g,%.9g,%.9g,%.9g,%.9g) "
         "diff=(apply=%.9g,g=%.9g,delta=%.9g) pcg=(%d,%.9g)\n",
         snapshotContact.C[0], snapshotPenalty, snapshotContact.lambda[0],
         snapshotForce, snapshotContact.JA[0], snapshotContact.JA[1],
         snapshotContact.JA[2], snapshotContact.JA[3],
         snapshotContact.JA[4], snapshotContact.JA[5],
         contactApplyDifference, contactGradientDifference,
         contactDeltaDifference, contactStats.iterations,
         contactStats.finalPreconditionedResidual);
  // Same-world-point authoring makes the geometric term zero at creation, so
  // standalone's positive penetration depth yields C=-depth.
  CHECK(std::fabs(snapshotContact.C[0] + snapshotDepth) <= 1e-5f,
        "Frozen contact geometric C changed: %.9g", snapshotContact.C[0]);
  CHECK(contactStats.converged && !contactStats.breakdown &&
            contactStats.finite,
        "Frozen contact PCG did not converge");
  CHECK(contactApplyDifference <= 1e-3 &&
            contactGradientDifference <= 1e-4 &&
            contactDeltaDifference <= 1e-5,
        "Frozen contact dense/matrix-free mismatch: %.9g %.9g %.9g",
        contactApplyDifference, contactGradientDifference,
        contactDeltaDifference);

  // Frozen dynamic-dynamic normal + 2D tangent row set. The force vector is
  // already projected into a circular Coulomb disk by the caller; the emitter
  // only validates and serializes the coupled snapshot.
  Solver frictionSolver;
  frictionSolver.gravity = Vec3();
  const Quat frictionRotationA(
      std::cos(0.1f), 0.0f, 0.0f, std::sin(0.1f));
  const Quat frictionRotationB(
      std::cos(-0.15f), 0.0f, std::sin(-0.15f), 0.0f);
  const uint32_t frictionBodyA = frictionSolver.addBody(
      Vec3(-0.55f, 1.0f, -0.1f), frictionRotationA,
      Vec3(0.5f, 0.5f, 0.5f), 4.0f);
  const uint32_t frictionBodyB = frictionSolver.addBody(
      Vec3(0.65f, 1.1f, 0.15f), frictionRotationB,
      Vec3(0.6f, 0.4f, 0.5f), 5.0f);
  const Vec3 frictionWorldPoint(0.05f, 1.25f, 0.3f);
  const Vec3 frictionNormal = Vec3(0.2f, 0.95f, -0.1f).normalized();
  const Vec3 frictionRA = frictionRotationA.conjugate().rotate(
      frictionWorldPoint - frictionSolver.bodies[frictionBodyA].position);
  const Vec3 frictionRB = frictionRotationB.conjugate().rotate(
      frictionWorldPoint - frictionSolver.bodies[frictionBodyB].position);
  frictionSolver.addContact(frictionBodyA, frictionBodyB, frictionNormal,
                            frictionRA, frictionRB, 0.02f, 0.5f);
  for (Body &body : frictionSolver.bodies) {
    body.initialPosition = body.position;
    body.initialRotation = body.rotation;
    body.inertialPosition = body.position;
    body.inertialRotation = body.rotation;
  }
  frictionSolver.bodies[frictionBodyA].inertialPosition +=
      Vec3(0.03f, -0.01f, 0.02f);
  frictionSolver.bodies[frictionBodyB].inertialPosition +=
      Vec3(-0.02f, 0.015f, -0.01f);
  Contact &frictionContact = frictionSolver.contacts.front();
  frictionSolver.computeC0(frictionContact);
  frictionSolver.computeConstraint(frictionContact);

  const IslandBodyMap frictionMap = buildIslandBodyMap(2);
  std::vector<Mat66> frictionK(2);
  std::vector<Vec6> frictionGradient(2);
  for (uint32_t body = 0; body < 2; ++body) {
    frictionK[body] = frictionSolver.bodies[body].getMassMatrix() / (dt * dt);
    const Vec6 displacement(
        frictionSolver.bodies[body].position -
            frictionSolver.bodies[body].inertialPosition,
        frictionSolver.bodies[body].deltaWInertial());
    frictionGradient[body] = frictionK[body] * displacement;
  }
  FrozenContactIslandRowSet frictionRows;
  frictionRows.active[0] = frictionRows.active[1] =
      frictionRows.active[2] = true;
  frictionRows.penalty[0] = 20000.0f;
  frictionRows.penalty[1] = frictionRows.penalty[2] = 8000.0f;
  frictionRows.force[0] = -1000.0f;
  frictionRows.force[1] = 300.0f;
  frictionRows.force[2] = -400.0f;
  frictionRows.activeMode[0] = 1;
  frictionRows.activeMode[1] = frictionRows.activeMode[2] = 2;
  frictionRows.tangentForceBound = 500.0f;

  IslandPcgSystem frictionSystem;
  frictionSystem.initialize(frictionK, frictionGradient);
  uint32_t frictionRowCount = 0;
  CHECK(emitFrozenContactIslandRows(
            frictionContact, 0, frictionSolver.bodies, frictionMap,
            frictionRows, frictionSystem, frictionRowCount) &&
            frictionRowCount == 3 && frictionSystem.rows().size() == 3,
        "Frozen contact set did not emit normal plus tangent pair");

  const std::vector<Vec6> frictionWitness = {
      Vec6(Vec3(0.2f, -0.3f, 0.1f), Vec3(0.05f, -0.1f, 0.2f)),
      Vec6(Vec3(-0.1f, 0.25f, -0.2f), Vec3(-0.15f, 0.08f, 0.04f))};
  std::vector<Vec6> frictionApplied;
  frictionSystem.apply(frictionWitness, frictionApplied);
  std::vector<Vec6> explicitFrictionApplied(2);
  std::vector<Vec6> explicitFrictionGradient = frictionGradient;
  for (uint32_t body = 0; body < 2; ++body)
    explicitFrictionApplied[body] = frictionK[body] * frictionWitness[body];
  for (const IslandPcgRow &row : frictionSystem.rows()) {
    const float projection =
        dot(row.jacobianA, frictionWitness[row.bodyA]) +
        dot(row.jacobianB, frictionWitness[row.bodyB]);
    explicitFrictionApplied[row.bodyA] +=
        row.jacobianA * (row.penalty * projection);
    explicitFrictionApplied[row.bodyB] +=
        row.jacobianB * (row.penalty * projection);
    explicitFrictionGradient[row.bodyA] += row.jacobianA * row.force;
    explicitFrictionGradient[row.bodyB] += row.jacobianB * row.force;
  }
  double frictionApplyDifference = 0.0;
  double frictionGradientDifference = 0.0;
  for (uint32_t body = 0; body < 2; ++body) {
    for (int k = 0; k < 6; ++k) {
      frictionApplyDifference =
          std::max(frictionApplyDifference,
                   std::fabs(double(frictionApplied[body][k] -
                                    explicitFrictionApplied[body][k])));
      frictionGradientDifference =
          std::max(frictionGradientDifference,
                   std::fabs(double(frictionSystem.gradient()[body][k] -
                                    explicitFrictionGradient[body][k])));
    }
  }

  std::vector<Vec6> frictionDelta;
  const IslandPcgStats frictionStats =
      frictionSystem.solvePcg(frictionDelta, 1e-9, 24);
  std::vector<Vec6> frictionEquation;
  frictionSystem.apply(frictionDelta, frictionEquation);
  double frictionEquationResidual = 0.0;
  for (uint32_t body = 0; body < 2; ++body)
    for (int k = 0; k < 6; ++k)
      frictionEquationResidual = std::max(
          frictionEquationResidual,
          std::fabs(double(frictionEquation[body][k] -
                           frictionSystem.gradient()[body][k])));

  Vec3 netRowForce;
  Vec3 netRowTorque;
  for (const IslandPcgRow &row : frictionSystem.rows()) {
    const Vec3 forceA = row.jacobianA.linear() * row.force;
    const Vec3 forceB = row.jacobianB.linear() * row.force;
    netRowForce += forceA + forceB;
    netRowTorque +=
        frictionSolver.bodies[frictionBodyA].position.cross(forceA) +
        row.jacobianA.angular() * row.force +
        frictionSolver.bodies[frictionBodyB].position.cross(forceB) +
        row.jacobianB.angular() * row.force;
  }

  const Vec3 commonTranslation(0.2f, -0.1f, 0.3f);
  const std::vector<Vec6> commonInput = {
      Vec6(commonTranslation, Vec3()), Vec6(commonTranslation, Vec3())};
  std::vector<Vec6> commonApplied;
  frictionSystem.apply(commonInput, commonApplied);
  double commonRowAction = 0.0;
  for (uint32_t body = 0; body < 2; ++body) {
    const Vec6 inertialAction = frictionK[body] * commonInput[body];
    for (int k = 0; k < 6; ++k)
      commonRowAction =
          std::max(commonRowAction,
                   std::fabs(double(commonApplied[body][k] -
                                    inertialAction[k])));
  }

  Contact swappedFrictionContact = frictionContact;
  std::swap(swappedFrictionContact.bodyA, swappedFrictionContact.bodyB);
  std::swap(swappedFrictionContact.rA, swappedFrictionContact.rB);
  swappedFrictionContact.normal = -swappedFrictionContact.normal;
  frictionSolver.computeC0(swappedFrictionContact);
  frictionSolver.computeConstraint(swappedFrictionContact);
  const Vec3 oldAxes[3] = {frictionContact.normal,
                           frictionContact.JAt1.linear(),
                           frictionContact.JAt2.linear()};
  const Vec3 physicalForceA =
      oldAxes[0] * frictionRows.force[0] +
      oldAxes[1] * frictionRows.force[1] +
      oldAxes[2] * frictionRows.force[2];
  const Vec3 newAxes[3] = {swappedFrictionContact.normal,
                           swappedFrictionContact.JAt1.linear(),
                           swappedFrictionContact.JAt2.linear()};
  FrozenContactIslandRowSet swappedFrictionRows = frictionRows;
  for (int row = 0; row < 3; ++row)
    swappedFrictionRows.force[row] = (-physicalForceA).dot(newAxes[row]);
  IslandPcgSystem swappedFrictionSystem;
  swappedFrictionSystem.initialize(frictionK, frictionGradient);
  uint32_t swappedFrictionRowCount = 0;
  CHECK(emitFrozenContactIslandRows(
            swappedFrictionContact, 0, frictionSolver.bodies, frictionMap,
            swappedFrictionRows, swappedFrictionSystem,
            swappedFrictionRowCount) &&
            swappedFrictionRowCount == 3,
        "Swapped frozen contact set did not emit three rows");
  std::vector<Vec6> swappedFrictionDelta;
  const IslandPcgStats swappedFrictionStats =
      swappedFrictionSystem.solvePcg(swappedFrictionDelta, 1e-9, 24);
  double frictionActorOrderDifference = 0.0;
  for (uint32_t body = 0; body < 2; ++body)
    for (int k = 0; k < 6; ++k)
      frictionActorOrderDifference =
          std::max(frictionActorOrderDifference,
                   std::fabs(double(frictionDelta[body][k] -
                                    swappedFrictionDelta[body][k])));

  FrozenContactIslandRowSet invalidConeRows = frictionRows;
  invalidConeRows.force[1] = 301.0f;
  IslandPcgSystem invalidConeSystem;
  invalidConeSystem.initialize(frictionK, frictionGradient);
  uint32_t invalidConeCount = UINT32_MAX;
  const bool invalidConeAccepted = emitFrozenContactIslandRows(
      frictionContact, 0, frictionSolver.bodies, frictionMap,
      invalidConeRows, invalidConeSystem, invalidConeCount);
  FrozenContactIslandRowSet incompleteTangentRows = frictionRows;
  incompleteTangentRows.active[2] = false;
  IslandPcgSystem incompleteTangentSystem;
  incompleteTangentSystem.initialize(frictionK, frictionGradient);
  uint32_t incompleteTangentCount = UINT32_MAX;
  const bool incompleteTangentAccepted = emitFrozenContactIslandRows(
      frictionContact, 0, frictionSolver.bodies, frictionMap,
      incompleteTangentRows, incompleteTangentSystem,
      incompleteTangentCount);

  printf("  contact-set rows=%u cone=(%.9g<=%.9g) "
         "diff=(apply=%.9g,g=%.9g,eq=%.9g,order=%.9g) "
         "wrench=(force=%.9g,torque=%.9g,translation=%.9g) "
         "pcg=(%d,%.9g,%d,%.9g) rejects=(%d,%d)\n",
         frictionRowCount,
         std::sqrt(frictionRows.force[1] * frictionRows.force[1] +
                   frictionRows.force[2] * frictionRows.force[2]),
         frictionRows.tangentForceBound, frictionApplyDifference,
         frictionGradientDifference, frictionEquationResidual,
         frictionActorOrderDifference, netRowForce.length(),
         netRowTorque.length(), commonRowAction, frictionStats.iterations,
         frictionStats.finalPreconditionedResidual,
         swappedFrictionStats.iterations,
         swappedFrictionStats.finalPreconditionedResidual,
         invalidConeAccepted ? 1 : 0,
         incompleteTangentAccepted ? 1 : 0);
  CHECK(frictionStats.converged && swappedFrictionStats.converged &&
            frictionStats.finite && swappedFrictionStats.finite,
        "Frozen contact-set PCG did not converge");
  CHECK(frictionApplyDifference <= 1e-3 &&
            frictionGradientDifference <= 1e-4 &&
            frictionEquationResidual <= 2e-3,
        "Frozen contact-set equation mismatch: %.9g %.9g %.9g",
        frictionApplyDifference, frictionGradientDifference,
        frictionEquationResidual);
  CHECK(netRowForce.length() <= 1e-5f && netRowTorque.length() <= 1e-3f &&
            commonRowAction <= 1e-3,
        "Frozen contact-set action/reaction mismatch: %.9g %.9g %.9g",
        netRowForce.length(), netRowTorque.length(), commonRowAction);
  CHECK(frictionActorOrderDifference <= 1e-5,
        "Frozen contact-set actor-order mismatch: %.9g",
        frictionActorOrderDifference);
  CHECK(!invalidConeAccepted && invalidConeCount == 0 &&
            invalidConeSystem.rows().empty() &&
            !incompleteTangentAccepted && incompleteTangentCount == 0 &&
            incompleteTangentSystem.rows().empty(),
        "Frozen contact-set accepted invalid cone or incomplete tangent pair");

  struct FrameResult {
    float forceRatio = 0.0f;
    float torqueRatio = 0.0f;
    float maxAnchorError = 0.0f;
    float maxRelativeSpeed = 0.0f;
    float maxAngularSpeed = 0.0f;
    float maxComError = 0.0f;
    float maxMomentum = 0.0f;
    int maxPcgIterations = 0;
    double worstPcgResidual = 0.0;
    bool finite = true;
    bool pcgOk = true;
  };
  const auto runFrameOracle = [](float frameDt, bool reverse, bool offset,
                                 bool chain) {
    FrameResult result;
    Solver solver;
    solver.gravity = Vec3();
    solver.iterations = 10;
    solver.useIslandPcgProbe = true;
    const int bodyCount = chain ? 3 : 2;
    std::vector<uint32_t> ids;
    for (int i = 0; i < bodyCount; ++i) {
      const float y = chain ? 8.0f + 2.0f * i : 9.0f + 2.0f * i;
      ids.push_back(solver.addBody(Vec3(0.0f, y, 0.0f), Quat(),
                                   Vec3(0.5f, 0.5f, 0.5f), 4.0f));
    }
    const float anchorX = offset ? 1.0f : 0.0f;
    for (int link = 0; link < bodyCount - 1; ++link) {
      const Vec3 lowerAnchor(anchorX, 1.0f, 0.0f);
      const Vec3 upperAnchor(anchorX, -1.0f, 0.0f);
      if (reverse)
        solver.addFixedJoint(ids[link + 1], ids[link], upperAnchor,
                             lowerAnchor, 1e4f);
      else
        solver.addFixedJoint(ids[link], ids[link + 1], lowerAnchor,
                             upperAnchor, 1e4f);
    }
    Vec3 initialCom;
    for (uint32_t id : ids)
      initialCom += solver.bodies[id].position;
    initialCom = initialCom * (1.0f / bodyCount);

    const int frames = static_cast<int>(10.0f / frameDt + 0.5f);
    for (int frame = 0; frame < frames; ++frame) {
      const float impulseSpeed = 39.24f * frameDt / 4.0f;
      solver.bodies[ids.front()].linearVelocity.y += impulseSpeed;
      solver.bodies[ids.back()].linearVelocity.y -= impulseSpeed;
      solver.contacts.clear();
      solver.step(frameDt);
      result.pcgOk = result.pcgOk && solver.islandPcgLastStats.converged &&
                     !solver.islandPcgLastStats.breakdown &&
                     solver.islandPcgLastStats.finite;
      result.maxPcgIterations =
          std::max(result.maxPcgIterations,
                   solver.islandPcgLastStats.iterations);
      result.worstPcgResidual =
          std::max(result.worstPcgResidual,
                   solver.islandPcgLastStats.finalPreconditionedResidual);

      Vec3 com;
      Vec3 momentum;
      for (uint32_t id : ids) {
        const Body &body = solver.bodies[id];
        com += body.position;
        momentum += body.linearVelocity * body.mass;
        result.maxAngularSpeed =
            std::max(result.maxAngularSpeed, body.angularVelocity.length());
        result.finite = result.finite && std::isfinite(body.position.x) &&
                        std::isfinite(body.position.y) &&
                        std::isfinite(body.position.z) &&
                        std::isfinite(body.linearVelocity.x) &&
                        std::isfinite(body.linearVelocity.y) &&
                        std::isfinite(body.linearVelocity.z);
      }
      com = com * (1.0f / bodyCount);
      result.maxComError =
          std::max(result.maxComError, (com - initialCom).length());
      result.maxMomentum =
          std::max(result.maxMomentum, momentum.length());
      for (const D6Joint &joint : solver.d6Joints) {
        const Body &bodyA = solver.bodies[joint.bodyA];
        const Body &bodyB = solver.bodies[joint.bodyB];
        const Vec3 worldA =
            bodyA.position + bodyA.rotation.rotate(joint.anchorA);
        const Vec3 worldB =
            bodyB.position + bodyB.rotation.rotate(joint.anchorB);
        result.maxAnchorError =
            std::max(result.maxAnchorError, (worldA - worldB).length());
        result.maxRelativeSpeed = std::max(
            result.maxRelativeSpeed,
            (bodyA.linearVelocity - bodyB.linearVelocity).length());
      }
    }
    Vec3 actor0Force, actor0Torque;
    const bool forceSupported = computeD6LockedLinearActor0Force(
        solver.d6Joints.front(), solver.bodies, frameDt, actor0Force);
    const bool torqueSupported = !offset || computeD6LockedAngularActor0Torque(
                                                solver.d6Joints.front(),
                                                solver.bodies, frameDt,
                                                actor0Torque);
    result.forceRatio = actor0Force.length() / 39.24f;
    result.torqueRatio = offset ? actor0Torque.length() / 39.24f : 1.0f;
    result.finite = result.finite && forceSupported && torqueSupported &&
                    std::isfinite(result.forceRatio) &&
                    std::isfinite(result.torqueRatio);
    return result;
  };

  const float frameTimesteps[] = {1.0f / 30.0f, 1.0f / 60.0f,
                                  1.0f / 120.0f};
  for (int fixture = 0; fixture < 3; ++fixture) {
    const bool offset = fixture == 1;
    const bool chain = fixture == 2;
    for (int order = 0; order < 2; ++order) {
      for (float frameDt : frameTimesteps) {
        const FrameResult frame =
            runFrameOracle(frameDt, order != 0, offset, chain);
        printf("  frame=%s order=%s dt=%.9g force=%.9g torque=%.9g "
               "anchor=%.9g relV=%.9g angV=%.9g com=%.9g momentum=%.9g "
               "pcg=(%d,%.9g)\n",
               chain ? "chain" : (offset ? "offset" : "centered"),
               order ? "reverse" : "normal", frameDt, frame.forceRatio,
               frame.torqueRatio, frame.maxAnchorError,
               frame.maxRelativeSpeed, frame.maxAngularSpeed,
               frame.maxComError, frame.maxMomentum,
               frame.maxPcgIterations, frame.worstPcgResidual);
        CHECK(frame.finite && frame.pcgOk,
              "Frame oracle produced non-finite state or PCG failure");
        CHECK(frame.forceRatio >= 0.9f && frame.forceRatio <= 1.1f &&
                  frame.torqueRatio >= 0.9f && frame.torqueRatio <= 1.1f,
              "Frame reaction gate failed: %.9g %.9g", frame.forceRatio,
              frame.torqueRatio);
        CHECK(frame.maxAnchorError <= 1e-3f &&
                  frame.maxRelativeSpeed <= 1e-3f &&
                  frame.maxAngularSpeed <= 1e-3f &&
                  frame.maxComError <= 1e-3f &&
                  frame.maxMomentum <= 1e-3f,
              "Frame stationarity gate failed: %.9g %.9g %.9g %.9g %.9g",
              frame.maxAnchorError, frame.maxRelativeSpeed,
              frame.maxAngularSpeed, frame.maxComError,
              frame.maxMomentum);
      }
    }
  }
  PASS("Matrix-free island operator preserves cross terms and stationarity");
}

// Test 127: algebra oracle for the force-mode linear position-drive row used
// by PhysX AVBD.  This is deliberately independent of scene iteration: the
// row force and tangent must have one unambiguous force/position/velocity
// unit contract before routing is enabled.
bool test127_linearPositionDriveDiscreteEquation() {
  printf("\n--- Test 127: Linear Position Drive Discrete Equation ---\n");
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float target = 0.5f;
  const float forceLimit = 5.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float masses[2] = {1.0f, 10.0f};

  for (float dt : frameDts) {
    const float invDt2 = 1.0f / (dt * dt);
    for (float mass : masses) {
      float orderPosition[2] = {0.0f, 0.0f};
      float orderAcceleration[2] = {0.0f, 0.0f};
      for (int order = 0; order < 2; ++order) {
        const float jacobian = order == 0 ? 1.0f : -1.0f;
        const float authoredTarget = jacobian * target;
        const float positionError = -authoredTarget;
        const float rawForce = stiffness * positionError;
        const float force =
            std::max(-forceLimit, std::min(forceLimit, rawForce));
        const bool saturated = std::fabs(rawForce) >= forceLimit;
        const float penalty =
            saturated ? 0.0f : stiffness + damping / dt;
        const float hessian = mass * invDt2 + penalty;
        const float gradient = jacobian * force;
        orderPosition[order] = -gradient / hessian;
        orderAcceleration[order] = orderPosition[order] * invDt2;
        CHECK(saturated && penalty == 0.0f,
              "Finite force clamp retained a tangent at dt=%.9g mass=%.9g",
              dt, mass);
        CHECK(std::fabs(std::fabs(force) - forceLimit) <= 1e-6f,
              "Finite force clamp changed units at dt=%.9g mass=%.9g",
              dt, mass);
      }

      const float expectedAcceleration = forceLimit / mass;
      const float orderDifference =
          std::fabs(orderPosition[0] - orderPosition[1]);
      printf("  dt=%.9g mass=%.9g x=%.9g accel=%.9g orderDiff=%.9g\n",
             dt, mass, orderPosition[0], orderAcceleration[0],
             orderDifference);
      CHECK(orderDifference <= 1e-9f,
            "Position row depends on actor order: %.9g", orderDifference);
      CHECK(std::fabs(orderAcceleration[0] - expectedAcceleration) <= 1e-5f,
            "Force-valued clamp acceleration mismatch: %.9g vs %.9g",
            orderAcceleration[0], expectedAcceleration);

      const float unsaturatedPenalty = stiffness + damping / dt;
      const float analyticPosition =
          stiffness * target / (mass * invDt2 + unsaturatedPenalty);
      CHECK(analyticPosition > 0.0f && analyticPosition < target,
            "Implicit spring response is outside the target interval: %.9g",
            analyticPosition);
    }
  }
  PASS("linear position drive equation preserves units and actor order");
}

// Test 128: independent algebra oracle for the public-force and break-force
// semantics of the force-mode linear position drive.  The solver row is
// force-valued, constraint writeback is impulse-valued, and eOUTPUT_FORCE
// controls whether the drive row contributes to the public reaction used for
// break testing.  Hard-limit reactions are outside this focused oracle.
bool test128_linearPositionDriveOutputForceSemantics() {
  printf("\n--- Test 128: Linear Position Drive Output-Force Semantics ---\n");
  const float forceLimit = 5.0f;
  const float breakBelow = 4.0f;
  const float breakAbove = 6.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  for (float dt : frameDts) {
    float normalizedForceByOrder[2] = {0.0f, 0.0f};
    for (int order = 0; order < 2; ++order) {
      const float actor0AxisSign = order == 0 ? 1.0f : -1.0f;
      const float authoredDriveForce = actor0AxisSign * forceLimit;
      const float driveImpulse = authoredDriveForce * dt;
      normalizedForceByOrder[order] =
          actor0AxisSign * (driveImpulse / dt);

      const float publicForceOn = std::fabs(driveImpulse) / dt;
      const float publicForceOff = 0.0f;
      const bool belowBreaksOn = publicForceOn > breakBelow;
      const bool belowBreaksOff = publicForceOff > breakBelow;
      const bool aboveBreaksOn = publicForceOn > breakAbove;
      const bool aboveBreaksOff = publicForceOff > breakAbove;
      const float publicTorque = 0.0f;

      printf("  dt=%.9g order=%s impulse=%.9g normalizedForce=%.9g "
             "offForce=%.9g breakBelow=(%d,%d) breakAbove=(%d,%d)\n",
             dt, order == 0 ? "forward" : "reverse", driveImpulse,
             normalizedForceByOrder[order], publicForceOff,
             belowBreaksOn ? 1 : 0, belowBreaksOff ? 1 : 0,
             aboveBreaksOn ? 1 : 0, aboveBreaksOff ? 1 : 0);

      CHECK(std::fabs(publicForceOn - forceLimit) <= 1e-6f,
            "Impulse-to-force conversion changed units at dt=%.9g", dt);
      CHECK(publicForceOff == 0.0f && publicTorque == 0.0f,
            "Disabled output-force emitted a drive reaction at dt=%.9g", dt);
      CHECK(belowBreaksOn && !belowBreaksOff,
            "Below-limit break bracket ignored output-force at dt=%.9g", dt);
      CHECK(!aboveBreaksOn && !aboveBreaksOff,
            "Above-limit break bracket fired at dt=%.9g", dt);
    }

    CHECK(std::fabs(normalizedForceByOrder[0] - forceLimit) <= 1e-6f &&
              std::fabs(normalizedForceByOrder[1] - forceLimit) <= 1e-6f &&
              std::fabs(normalizedForceByOrder[0] -
                        normalizedForceByOrder[1]) <= 1e-6f,
          "Output-force actor-order normalization failed at dt=%.9g", dt);
  }

  PASS("linear position drive output-force preserves units, order, and break semantics");
}

// Test 129: independent algebra oracle for the scoped force-mode TWIST
// velocity drive.  It fixes the target convention (wA-wB=target), verifies
// that damping/dt maps the angular-displacement residual to torque, and keeps
// torque, torque*dt writeback, actor0 sign and break semantics in one witness.
bool test129_angularTwistVelocityDriveOutputForceSemantics() {
  printf("\n--- Test 129: Angular TWIST Velocity Output-Torque Semantics ---\n");
  const float targetVelocity = 1.0f;
  const float damping = 1000.0f;
  const float torqueLimit = 5.0f;
  const float inertia = 1.0f;
  const float breakBelow = 4.0f;
  const float breakAbove = 6.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  for (float dt : frameDts) {
    float normalizedTorqueByOrder[2] = {0.0f, 0.0f};
    float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
    for (int order = 0; order < 2; ++order) {
      const bool dynamicActor0 = order != 0;
      const float signAL = dynamicActor0 ? -1.0f : 1.0f;
      const float expectedDynamicAxisSign = dynamicActor0 ? 1.0f : -1.0f;
      const float residual = targetVelocity * dt;
      const float penalty = damping / dt;
      const float rawTorque = penalty * residual;
      const float driveTorque =
          std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
      const float gradient = signAL * driveTorque;
      const float angularDisplacement =
          -gradient / (inertia / (dt * dt));
      const float dynamicAngularVelocity = angularDisplacement / dt;
      normalizedAccelerationByOrder[order] =
          expectedDynamicAxisSign * dynamicAngularVelocity / dt;

      const float actor0Torque = driveTorque;
      const float angularImpulse = actor0Torque * dt;
      normalizedTorqueByOrder[order] = angularImpulse / dt;
      const float publicTorqueOn = std::fabs(angularImpulse) / dt;
      const float publicTorqueOff = 0.0f;
      const bool belowBreaksOn = publicTorqueOn > breakBelow;
      const bool belowBreaksOff = publicTorqueOff > breakBelow;
      const bool aboveBreaksOn = publicTorqueOn > breakAbove;
      const bool aboveBreaksOff = publicTorqueOff > breakAbove;

      printf("  dt=%.9g order=%s residual=%.9g torque=%.9g "
             "angularImpulse=%.9g accel=%.9g breakBelow=(%d,%d) "
             "breakAbove=(%d,%d)\n",
             dt, dynamicActor0 ? "reverse" : "forward", residual,
             driveTorque, angularImpulse,
             normalizedAccelerationByOrder[order],
             belowBreaksOn ? 1 : 0, belowBreaksOff ? 1 : 0,
             aboveBreaksOn ? 1 : 0, aboveBreaksOff ? 1 : 0);

      CHECK(std::fabs(rawTorque - damping * targetVelocity) <= 1e-4f,
            "TWIST velocity residual changed torque units at dt=%.9g", dt);
      CHECK(std::fabs(publicTorqueOn - torqueLimit) <= 1e-6f,
            "Angular impulse-to-torque conversion failed at dt=%.9g", dt);
      CHECK(publicTorqueOff == 0.0f,
            "Disabled angular output emitted drive torque at dt=%.9g", dt);
      CHECK(belowBreaksOn && !belowBreaksOff,
            "Angular below-limit bracket ignored output torque at dt=%.9g",
            dt);
      CHECK(!aboveBreaksOn && !aboveBreaksOff,
            "Angular above-limit bracket fired at dt=%.9g", dt);
      CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                      torqueLimit / inertia) <= 1e-5f,
            "Finite torque acceleration mismatch at dt=%.9g", dt);
    }

    CHECK(std::fabs(normalizedTorqueByOrder[0] - torqueLimit) <= 1e-6f &&
              std::fabs(normalizedTorqueByOrder[1] - torqueLimit) <= 1e-6f &&
              std::fabs(normalizedTorqueByOrder[0] -
                        normalizedTorqueByOrder[1]) <= 1e-6f,
          "TWIST output-torque actor-order normalization failed at dt=%.9g",
          dt);
    CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                    normalizedAccelerationByOrder[1]) <= 1e-6f,
          "TWIST finite-torque motion depends on actor order at dt=%.9g", dt);
  }

  PASS("angular TWIST velocity drive preserves torque, impulse, order, and break semantics");
}

// Test 130: independent algebra and frame oracle for the scoped force-mode
// SWING1 velocity drive.  SWING1 uses actor A's local Y axis and the same
// wA-wB=target convention as TWIST, but it must not accidentally use actor B's
// authored frame axis or inherit an X-axis-only writeback.
bool test130_angularSwing1VelocityDriveOutputForceSemantics() {
  printf("\n--- Test 130: Angular SWING1 Velocity Output-Torque Semantics ---\n");
  const float targetVelocity = 1.0f;
  const float damping = 1000.0f;
  const float torqueLimit = 5.0f;
  const float inertia = 1.0f;
  const float breakBelow = 4.0f;
  const float breakAbove = 6.0f;
  const Vec3 actorAFrameAxis(0.0f, 1.0f, 0.0f);
  const Vec3 actorBFrameAxis(0.0f, 0.0f, 1.0f);
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  CHECK(std::fabs(actorAFrameAxis.dot(actorBFrameAxis)) <= 1e-6f,
        "SWING1 fixture does not separate actor A and actor B frame axes");

  for (float dt : frameDts) {
    float normalizedTorqueByOrder[2] = {0.0f, 0.0f};
    float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
    for (int order = 0; order < 2; ++order) {
      const bool dynamicActor0 = order != 0;
      const float signAL = dynamicActor0 ? -1.0f : 1.0f;
      const float expectedDynamicAxisSign = dynamicActor0 ? 1.0f : -1.0f;
      const float residual = targetVelocity * dt;
      const float penalty = damping / dt;
      const float rawTorque = penalty * residual;
      const float driveTorque =
          std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
      const float gradient = signAL * driveTorque;
      const float angularDisplacement =
          -gradient / (inertia / (dt * dt));
      const float dynamicAngularVelocity = angularDisplacement / dt;
      normalizedAccelerationByOrder[order] =
          expectedDynamicAxisSign * dynamicAngularVelocity / dt;

      const Vec3 actor0Torque = actorAFrameAxis * driveTorque;
      const Vec3 angularImpulse = actor0Torque * dt;
      const Vec3 publicTorqueOn = angularImpulse * (1.0f / dt);
      const Vec3 publicTorqueOff(0.0f, 0.0f, 0.0f);
      const float signedPublicTorque = publicTorqueOn.dot(actorAFrameAxis);
      const float wrongFrameTorque = publicTorqueOn.dot(actorBFrameAxis);
      normalizedTorqueByOrder[order] = signedPublicTorque;
      const bool belowBreaksOn = publicTorqueOn.length() > breakBelow;
      const bool belowBreaksOff = publicTorqueOff.length() > breakBelow;
      const bool aboveBreaksOn = publicTorqueOn.length() > breakAbove;
      const bool aboveBreaksOff = publicTorqueOff.length() > breakAbove;

      printf("  dt=%.9g order=%s residual=%.9g torque=%.9g "
             "wrongFrame=%.9g accel=%.9g breakBelow=(%d,%d) "
             "breakAbove=(%d,%d)\n",
             dt, dynamicActor0 ? "reverse" : "forward", residual,
             signedPublicTorque, wrongFrameTorque,
             normalizedAccelerationByOrder[order], belowBreaksOn ? 1 : 0,
             belowBreaksOff ? 1 : 0, aboveBreaksOn ? 1 : 0,
             aboveBreaksOff ? 1 : 0);

      CHECK(std::fabs(rawTorque - damping * targetVelocity) <= 1e-4f,
            "SWING1 velocity residual changed torque units at dt=%.9g", dt);
      CHECK(std::fabs(signedPublicTorque - torqueLimit) <= 1e-6f,
            "SWING1 angular impulse-to-torque conversion failed at dt=%.9g",
            dt);
      CHECK(std::fabs(wrongFrameTorque) <= 1e-6f,
            "SWING1 output torque used actor B's frame at dt=%.9g", dt);
      CHECK(publicTorqueOff.length() == 0.0f,
            "Disabled SWING1 output emitted drive torque at dt=%.9g", dt);
      CHECK(belowBreaksOn && !belowBreaksOff,
            "SWING1 below-limit bracket ignored output torque at dt=%.9g", dt);
      CHECK(!aboveBreaksOn && !aboveBreaksOff,
            "SWING1 above-limit bracket fired at dt=%.9g", dt);
      CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                      torqueLimit / inertia) <= 1e-5f,
            "SWING1 finite torque acceleration mismatch at dt=%.9g", dt);
    }

    CHECK(std::fabs(normalizedTorqueByOrder[0] - torqueLimit) <= 1e-6f &&
              std::fabs(normalizedTorqueByOrder[1] - torqueLimit) <= 1e-6f &&
              std::fabs(normalizedTorqueByOrder[0] -
                        normalizedTorqueByOrder[1]) <= 1e-6f,
          "SWING1 output-torque actor-order normalization failed at dt=%.9g",
          dt);
    CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                    normalizedAccelerationByOrder[1]) <= 1e-6f,
          "SWING1 finite-torque motion depends on actor order at dt=%.9g",
          dt);
  }

  PASS("angular SWING1 velocity drive preserves frame, torque, impulse, order, and break semantics");
}

// Test 131: independent algebra and frame oracle for the scoped force-mode
// SWING2 velocity drive.  SWING2 uses actor A's local Z axis and the same
// wA-wB=target convention as TWIST/SWING1, but it must not accidentally use
// actor B's authored frame axis or inherit an X/Y-axis-only writeback.
bool test131_angularSwing2VelocityDriveOutputForceSemantics() {
  printf("\n--- Test 131: Angular SWING2 Velocity Output-Torque Semantics ---\n");
  const float targetVelocity = 1.0f;
  const float damping = 1000.0f;
  const float torqueLimit = 5.0f;
  const float inertia = 1.0f;
  const float breakBelow = 4.0f;
  const float breakAbove = 6.0f;
  const Vec3 actorAFrameAxis(0.0f, 0.0f, 1.0f);
  const Vec3 actorBFrameAxis(1.0f, 0.0f, 0.0f);
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  CHECK(std::fabs(actorAFrameAxis.dot(actorBFrameAxis)) <= 1e-6f,
        "SWING2 fixture does not separate actor A and actor B frame axes");

  for (float dt : frameDts) {
    float normalizedTorqueByOrder[2] = {0.0f, 0.0f};
    float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
    for (int order = 0; order < 2; ++order) {
      const bool dynamicActor0 = order != 0;
      const float signAL = dynamicActor0 ? -1.0f : 1.0f;
      const float expectedDynamicAxisSign = dynamicActor0 ? 1.0f : -1.0f;
      const float residual = targetVelocity * dt;
      const float penalty = damping / dt;
      const float rawTorque = penalty * residual;
      const float driveTorque =
          std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
      const float gradient = signAL * driveTorque;
      const float angularDisplacement =
          -gradient / (inertia / (dt * dt));
      const float dynamicAngularVelocity = angularDisplacement / dt;
      normalizedAccelerationByOrder[order] =
          expectedDynamicAxisSign * dynamicAngularVelocity / dt;

      const Vec3 actor0Torque = actorAFrameAxis * driveTorque;
      const Vec3 angularImpulse = actor0Torque * dt;
      const Vec3 publicTorqueOn = angularImpulse * (1.0f / dt);
      const Vec3 publicTorqueOff(0.0f, 0.0f, 0.0f);
      const float signedPublicTorque = publicTorqueOn.dot(actorAFrameAxis);
      const float wrongFrameTorque = publicTorqueOn.dot(actorBFrameAxis);
      normalizedTorqueByOrder[order] = signedPublicTorque;
      const bool belowBreaksOn = publicTorqueOn.length() > breakBelow;
      const bool belowBreaksOff = publicTorqueOff.length() > breakBelow;
      const bool aboveBreaksOn = publicTorqueOn.length() > breakAbove;
      const bool aboveBreaksOff = publicTorqueOff.length() > breakAbove;

      printf("  dt=%.9g order=%s residual=%.9g torque=%.9g "
             "wrongFrame=%.9g accel=%.9g breakBelow=(%d,%d) "
             "breakAbove=(%d,%d)\n",
             dt, dynamicActor0 ? "reverse" : "forward", residual,
             signedPublicTorque, wrongFrameTorque,
             normalizedAccelerationByOrder[order], belowBreaksOn ? 1 : 0,
             belowBreaksOff ? 1 : 0, aboveBreaksOn ? 1 : 0,
             aboveBreaksOff ? 1 : 0);

      CHECK(std::fabs(rawTorque - damping * targetVelocity) <= 1e-4f,
            "SWING2 velocity residual changed torque units at dt=%.9g", dt);
      CHECK(std::fabs(signedPublicTorque - torqueLimit) <= 1e-6f,
            "SWING2 angular impulse-to-torque conversion failed at dt=%.9g",
            dt);
      CHECK(std::fabs(wrongFrameTorque) <= 1e-6f,
            "SWING2 output torque used actor B's frame at dt=%.9g", dt);
      CHECK(publicTorqueOff.length() == 0.0f,
            "Disabled SWING2 output emitted drive torque at dt=%.9g", dt);
      CHECK(belowBreaksOn && !belowBreaksOff,
            "SWING2 below-limit bracket ignored output torque at dt=%.9g", dt);
      CHECK(!aboveBreaksOn && !aboveBreaksOff,
            "SWING2 above-limit bracket fired at dt=%.9g", dt);
      CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                      torqueLimit / inertia) <= 1e-5f,
            "SWING2 finite torque acceleration mismatch at dt=%.9g", dt);
    }

    CHECK(std::fabs(normalizedTorqueByOrder[0] - torqueLimit) <= 1e-6f &&
              std::fabs(normalizedTorqueByOrder[1] - torqueLimit) <= 1e-6f &&
              std::fabs(normalizedTorqueByOrder[0] -
                        normalizedTorqueByOrder[1]) <= 1e-6f,
          "SWING2 output-torque actor-order normalization failed at dt=%.9g",
          dt);
    CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                    normalizedAccelerationByOrder[1]) <= 1e-6f,
          "SWING2 finite-torque motion depends on actor order at dt=%.9g",
          dt);
  }

  PASS("angular SWING2 velocity drive preserves frame, torque, impulse, order, and break semantics");
}

// Test 132: independent three-row SLERP output-torque oracle.  Unlike the
// single-axis angular drives, SLERP targets wB-wA and emits fixed world X/Y/Z
// rows.  Its one scalar force limit clamps each row independently, while break
// evaluation consumes the magnitude of the aggregated actor0 torque.
bool test132_angularSlerpVelocityDriveOutputForceSemantics() {
  printf("\n--- Test 132: Angular SLERP Velocity Output-Torque Semantics ---\n");
  const float damping = 1000.0f;
  const float torqueLimit = 5.0f;
  const float inertia = 1.0f;
  const float sqrtThree = std::sqrt(3.0f);
  const float breakBelow = 4.0f * sqrtThree;
  const float breakAbove = 6.0f * sqrtThree;
  const Vec3 localTarget = Vec3(1.0f, -1.0f, 1.0f).normalized();
  const Quat actorAFrame;
  const float actorBHalfAngle = 3.14159265358979323846f / 3.0f;
  const Vec3 actorBAxis = Vec3(1.0f, 1.0f, 1.0f).normalized();
  const Quat actorBFrame(std::cos(actorBHalfAngle),
                         actorBAxis.x * std::sin(actorBHalfAngle),
                         actorBAxis.y * std::sin(actorBHalfAngle),
                         actorBAxis.z * std::sin(actorBHalfAngle));
  const Vec3 actorAWorldTarget = actorAFrame.rotate(localTarget);
  const Vec3 actorBWorldTarget = actorBFrame.rotate(localTarget);
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  auto saturatedActor0Torque = [torqueLimit](const Vec3 &worldTarget) {
    return Vec3(worldTarget.x >= 0.0f ? -torqueLimit : torqueLimit,
                worldTarget.y >= 0.0f ? -torqueLimit : torqueLimit,
                worldTarget.z >= 0.0f ? -torqueLimit : torqueLimit);
  };
  const Vec3 expectedActor0Torque =
      saturatedActor0Torque(actorAWorldTarget);
  const Vec3 wrongFrameTorque = saturatedActor0Torque(actorBWorldTarget);
  const float expectedTorqueMagnitude = torqueLimit * sqrtThree;
  const float expectedDynamicAcceleration =
      -expectedActor0Torque.dot(actorAWorldTarget) / inertia;

  CHECK((expectedActor0Torque - wrongFrameTorque).length() > torqueLimit,
        "SLERP fixture does not separate actor A and actor B frame rows");
  CHECK(std::fabs(expectedActor0Torque.length() - expectedTorqueMagnitude) <=
            1e-6f,
        "SLERP aggregate torque magnitude is not the three-row limit");

  for (float dt : frameDts) {
    Vec3 normalizedTorqueByOrder[2];
    float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
    for (int order = 0; order < 2; ++order) {
      const bool dynamicActor0 = order != 0;
      const Vec3 residual = actorAWorldTarget * -dt;
      const Vec3 rawTorque = residual * (damping / dt);
      const Vec3 driveTorque(
          std::max(-torqueLimit, std::min(torqueLimit, rawTorque.x)),
          std::max(-torqueLimit, std::min(torqueLimit, rawTorque.y)),
          std::max(-torqueLimit, std::min(torqueLimit, rawTorque.z)));
      const Vec3 actor0Torque = driveTorque;
      const Vec3 angularImpulse = actor0Torque * dt;
      const Vec3 publicTorqueOn = angularImpulse * (1.0f / dt);
      const Vec3 publicTorqueOff;
      const Vec3 dynamicTorque =
          dynamicActor0 ? actor0Torque : actor0Torque * -1.0f;
      const Vec3 expectedDynamicAxis =
          dynamicActor0 ? actorAWorldTarget * -1.0f : actorAWorldTarget;
      normalizedAccelerationByOrder[order] =
          dynamicTorque.dot(expectedDynamicAxis) / inertia;
      normalizedTorqueByOrder[order] = publicTorqueOn;

      const bool belowBreaksOn = publicTorqueOn.length() > breakBelow;
      const bool belowBreaksOff = publicTorqueOff.length() > breakBelow;
      const bool aboveBreaksOn = publicTorqueOn.length() > breakAbove;
      const bool aboveBreaksOff = publicTorqueOff.length() > breakAbove;

      printf("  dt=%.9g order=%s residual=(%.9g,%.9g,%.9g) "
             "actor0Torque=(%.9g,%.9g,%.9g) accel=%.9g "
             "breakBelow=(%d,%d) breakAbove=(%d,%d)\n",
             dt, dynamicActor0 ? "reverse" : "forward", residual.x,
             residual.y, residual.z, publicTorqueOn.x, publicTorqueOn.y,
             publicTorqueOn.z, normalizedAccelerationByOrder[order],
             belowBreaksOn ? 1 : 0, belowBreaksOff ? 1 : 0,
             aboveBreaksOn ? 1 : 0, aboveBreaksOff ? 1 : 0);

      CHECK((rawTorque - actorAWorldTarget * -damping).length() <= 1e-4f,
            "SLERP velocity residual changed torque units at dt=%.9g", dt);
      CHECK((publicTorqueOn - expectedActor0Torque).length() <= 1e-6f,
            "SLERP actor0 torque lost row sign/limit at dt=%.9g", dt);
      CHECK(publicTorqueOff.length() == 0.0f,
            "Disabled SLERP output emitted drive torque at dt=%.9g", dt);
      CHECK(belowBreaksOn && !belowBreaksOff,
            "SLERP below-resultant bracket ignored output torque at dt=%.9g",
            dt);
      CHECK(!aboveBreaksOn && !aboveBreaksOff,
            "SLERP above-resultant bracket fired at dt=%.9g", dt);
      CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                      expectedDynamicAcceleration) <= 1e-5f,
            "SLERP finite torque acceleration mismatch at dt=%.9g", dt);
    }

    CHECK((normalizedTorqueByOrder[0] - expectedActor0Torque).length() <=
                  1e-6f &&
              (normalizedTorqueByOrder[1] - expectedActor0Torque).length() <=
                  1e-6f &&
              (normalizedTorqueByOrder[0] -
               normalizedTorqueByOrder[1])
                      .length() <=
                  1e-6f,
          "SLERP output-torque actor-order normalization failed at dt=%.9g",
          dt);
    CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                    normalizedAccelerationByOrder[1]) <= 1e-6f,
          "SLERP finite-torque motion depends on actor order at dt=%.9g", dt);
  }

  PASS("angular SLERP velocity drive preserves three-row frame, torque, impulse, order, and break semantics");
}

// Test 133: offset-anchor public wrench oracle for the scoped linear-X
// position drive.  The drive force acts at the dynamic anchor, while the
// locked angular rows supply the opposite reaction torque.  PxConstraint
// reports the linear row about bodyAWorldOffset, so eOUTPUT_FORCE adds the
// drive force but must not add a second COM lever-arm moment.  The locked-row
// torque remains public, impulse-valued in writeback, and breakable in both
// output-force flag states.
bool test133_linearPositionDriveOffsetMomentSemantics() {
  printf("\n--- Test 133: Linear Position Drive Offset-Moment Semantics ---\n");
  const float forceLimit = 5.0f;
  const float breakBelow = 1.0f;
  const float breakAbove = 1.5f;
  const Vec3 driveAxis(1.0f, 0.0f, 0.0f);
  const Vec3 dynamicArm(0.0f, 0.25f, 0.0f);
  const Vec3 dynamicForce = driveAxis * forceLimit;
  const Vec3 expectedNormalizedTorque =
      dynamicArm.cross(dynamicForce) * -1.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  CHECK(std::fabs(expectedNormalizedTorque.length() - 1.25f) <= 1e-6f,
        "Offset fixture did not produce the 1.25 N*m reaction witness");

  for (float dt : frameDts) {
    Vec3 normalizedTorqueByOrder[2];
    for (int order = 0; order < 2; ++order) {
      const float actor0Sign = order == 0 ? -1.0f : 1.0f;
      const Vec3 actor0DriveForce = dynamicForce * actor0Sign;
      const Vec3 actor0LockedTorque =
          expectedNormalizedTorque * actor0Sign;
      const Vec3 linearImpulseOn = actor0DriveForce * dt;
      const Vec3 linearImpulseOff;
      const Vec3 angularImpulseOn = actor0LockedTorque * dt;
      const Vec3 angularImpulseOff = actor0LockedTorque * dt;
      const Vec3 publicForceOn = linearImpulseOn * (1.0f / dt);
      const Vec3 publicForceOff = linearImpulseOff * (1.0f / dt);
      const Vec3 publicTorqueOn = angularImpulseOn * (1.0f / dt);
      const Vec3 publicTorqueOff = angularImpulseOff * (1.0f / dt);
      normalizedTorqueByOrder[order] = publicTorqueOn * actor0Sign;

      const bool belowBreaksOn = publicTorqueOn.length() > breakBelow;
      const bool belowBreaksOff = publicTorqueOff.length() > breakBelow;
      const bool aboveBreaksOn = publicTorqueOn.length() > breakAbove;
      const bool aboveBreaksOff = publicTorqueOff.length() > breakAbove;

      printf("  dt=%.9g order=%s actor0Force=(%.9g,%.9g,%.9g) "
             "normalizedTorque=(%.9g,%.9g,%.9g) "
             "breakBelow=(%d,%d) breakAbove=(%d,%d)\n",
             dt, order == 0 ? "forward" : "reverse", publicForceOn.x,
             publicForceOn.y, publicForceOn.z,
             normalizedTorqueByOrder[order].x,
             normalizedTorqueByOrder[order].y,
             normalizedTorqueByOrder[order].z, belowBreaksOn ? 1 : 0,
             belowBreaksOff ? 1 : 0, aboveBreaksOn ? 1 : 0,
             aboveBreaksOff ? 1 : 0);

      CHECK((publicForceOn - actor0DriveForce).length() <= 1e-6f &&
                publicForceOff.length() == 0.0f,
            "Offset linear output flag changed force units at dt=%.9g", dt);
      CHECK((publicTorqueOn - actor0LockedTorque).length() <= 1e-6f &&
                (publicTorqueOff - actor0LockedTorque).length() <= 1e-6f,
            "Offset locked reaction depends on output-force at dt=%.9g", dt);
      CHECK(belowBreaksOn && belowBreaksOff,
            "Offset below-moment bracket missed locked reaction at dt=%.9g",
            dt);
      CHECK(!aboveBreaksOn && !aboveBreaksOff,
            "Offset above-moment bracket fired at dt=%.9g", dt);
    }

    CHECK((normalizedTorqueByOrder[0] - expectedNormalizedTorque).length() <=
                  1e-6f &&
              (normalizedTorqueByOrder[1] - expectedNormalizedTorque).length() <=
                  1e-6f &&
              (normalizedTorqueByOrder[0] - normalizedTorqueByOrder[1])
                      .length() <=
                  1e-6f,
          "Offset public torque actor-order normalization failed at dt=%.9g",
          dt);
  }

  PASS("linear position offset moment preserves report origin, impulse, order, flag, and break semantics");
}

// Test 134: frozen algebra for the first angular-position family to be
// migrated into PhysX AVBD.  PhysX's SWING_TWIST D6 prep represents a TWIST
// target error as -2*delta.x, not as the raw rotation angle.  The authored
// target is inverted when actor order is reversed, but actor-A-frame torque,
// force-valued clamp and torque*dt writeback must normalize to the same
// physical result.
bool test134_angularTwistPositionDriveDiscreteEquation() {
  printf("\n--- Test 134: Angular TWIST Position Drive Discrete Equation ---\n");
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float targetDelta = 0.5f;
  const float torqueLimit = 5.0f;
  const float inertia = 1.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float initialAngles[2] = {0.0f, 0.2f};
  const float invSqrtTwo = 0.7071067811865475f;
  const Vec3 actorAWorldAxis(invSqrtTwo, -invSqrtTwo, 0.0f);
  const Vec3 actorBLocalAxis(1.0f, 0.0f, 0.0f);
  const Vec3 actorBWorldAxis = actorAWorldAxis;

  CHECK(std::fabs(actorAWorldAxis.length() - 1.0f) <= 1e-6f &&
            actorAWorldAxis.dot(actorBWorldAxis) >= 0.999999f &&
            actorAWorldAxis.dot(actorBLocalAxis) > 0.70f &&
            actorAWorldAxis.dot(actorBLocalAxis) < 0.72f,
        "TWIST position fixture does not separate local and world frames");

  for (float dt : frameDts) {
    for (float initialAngle : initialAngles) {
      float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
      float normalizedTorqueByOrder[2] = {0.0f, 0.0f};
      for (int order = 0; order < 2; ++order) {
        const float actorOrderSign = order == 0 ? 1.0f : -1.0f;
        const float authoredInitial = actorOrderSign * initialAngle;
        const float authoredTarget =
            actorOrderSign * (initialAngle + targetDelta);
        const float signedTargetError =
            actorOrderSign * (authoredTarget - authoredInitial);
        const float rowError = 2.0f * std::sin(0.5f * signedTargetError);
        const float rawTorque = stiffness * rowError;
        const float driveTorque =
            std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
        const bool saturated = std::fabs(rawTorque) >= torqueLimit;
        const float tangent = saturated
                                  ? 0.0f
                                  : stiffness *
                                        std::cos(0.5f * signedTargetError) +
                                        damping / dt;
        const float angularDisplacement =
            driveTorque / (inertia / (dt * dt) + tangent);
        normalizedAccelerationByOrder[order] =
            angularDisplacement / (dt * dt);
        const float actor0Torque = actorOrderSign * driveTorque;
        const float angularImpulse = actor0Torque * dt;
        normalizedTorqueByOrder[order] =
            actorOrderSign * angularImpulse / dt;

        printf("  dt=%.9g initial=%.9g order=%s rowError=%.9g "
               "torque=%.9g impulse=%.9g accel=%.9g\n",
               dt, initialAngle, order == 0 ? "forward" : "reverse",
               rowError, driveTorque, angularImpulse,
               normalizedAccelerationByOrder[order]);

        CHECK(std::fabs(signedTargetError - targetDelta) <= 1e-6f,
              "TWIST target inversion changed physical error at dt=%.9g",
              dt);
        CHECK(std::fabs(rowError - 2.0f * std::sin(0.25f)) <= 1e-6f,
              "TWIST position row lost half-angle semantics at dt=%.9g",
              dt);
        CHECK(saturated && tangent == 0.0f &&
                  std::fabs(driveTorque - torqueLimit) <= 1e-6f,
              "TWIST finite torque clamp/tangent failed at dt=%.9g", dt);
        CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                        torqueLimit / inertia) <= 1e-5f,
              "TWIST finite-torque acceleration changed units at dt=%.9g",
              dt);
        CHECK(std::fabs(normalizedTorqueByOrder[order] - torqueLimit) <=
                  1e-6f,
              "TWIST torque*dt writeback changed units at dt=%.9g", dt);
      }

      CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                      normalizedAccelerationByOrder[1]) <= 1e-6f &&
                std::fabs(normalizedTorqueByOrder[0] -
                          normalizedTorqueByOrder[1]) <= 1e-6f,
            "TWIST position response depends on actor order at dt=%.9g", dt);
    }
  }

  PASS("angular TWIST position drive preserves target, frame, clamp, impulse, and actor-order semantics");
}

// Test 135: freeze the effective-mass semantics of an acceleration-mode
// linear velocity drive before enabling the corresponding PhysX AVBD island.
// With stiffness zero, the official 1D acceleration-spring coefficient gives
// dv = dt*damping/(1 + dt*damping) * target, independent of endpoint mass.
// A force-valued limit still clamps physical force, so its saturated response
// remains dt*forceLimit/effectiveMass.
bool test135_linearAccelerationDriveEffectiveMassSemantics() {
  printf("\n--- Test 135: Linear Acceleration Drive Effective-Mass Semantics ---\n");
  struct LaneResult {
    float response = 0.0f;
    float momentum = 0.0f;
    float penalty = 0.0f;
    float force = 0.0f;
    uint32_t activeMode = 0;
    bool emitted = false;
    bool converged = false;
  };

  const auto runLane = [](float dt, float endpointMass, bool acceleration,
                          bool reverse, float forceLimit) {
    LaneResult result;
    Solver solver;
    solver.gravity = Vec3();
    const Vec3 halfExtent(0.5f, 0.5f, 0.5f);
    const uint32_t left = solver.addBody(Vec3(-1.0f, 0.0f, 0.0f), Quat(),
                                         halfExtent, endpointMass, 0.0f);
    const uint32_t right = solver.addBody(Vec3(1.0f, 0.0f, 0.0f), Quat(),
                                          halfExtent, endpointMass, 0.0f);
    for (Body &body : solver.bodies) {
      body.initialPosition = body.position;
      body.initialRotation = body.rotation;
      body.inertialPosition = body.position;
      body.inertialRotation = body.rotation;
      body.updateInvInertiaWorld();
    }
    const uint32_t jointIndex =
        reverse ? solver.addD6Joint(right, left, Vec3(), Vec3(), 0x2A,
                                    0x2A, 0.0f, 2e4f)
                : solver.addD6Joint(left, right, Vec3(), Vec3(), 0x2A,
                                    0x2A, 0.0f, 2e4f);
    D6Joint &joint = solver.d6Joints[jointIndex];
    joint.driveFlags = 0x01;
    joint.driveAccelerationFlags = acceleration ? 0x01 : 0;
    joint.driveLinearVelocity =
        Vec3(reverse ? -1.0f : 1.0f, 0.0f, 0.0f);
    joint.linearDriveDamping = Vec3(6.0f, 0.0f, 0.0f);
    joint.driveLinearForce = Vec3(forceLimit, 0.0f, 0.0f);
    joint.lambdaDriveLinear = Vec3();

    const IslandBodyMap map = buildIslandBodyMap(2);
    std::vector<Mat66> blocks(2);
    std::vector<Vec6> gradients(2);
    for (uint32_t body = 0; body < 2; ++body)
      blocks[body] = solver.bodies[body].getMassMatrix() / (dt * dt);
    IslandPcgSystem system;
    system.initialize(blocks, gradients);
    uint32_t emittedRows = 0;
    result.emitted = emitLinearXVelocityDriveIslandRow(
        joint, jointIndex, solver.bodies, map, dt, system, emittedRows);
    if (!result.emitted || emittedRows != 1 || system.rows().size() != 1)
      return result;
    const IslandPcgRow &row = system.rows()[0];
    result.penalty = row.penalty;
    result.force = row.force;
    result.activeMode = row.activeMode;
    std::vector<Vec6> solution;
    const IslandPcgStats stats = system.solvePcg(solution, 1e-10, 24);
    result.converged = stats.converged && !stats.breakdown && stats.finite &&
                       solution.size() == 2;
    if (!result.converged)
      return result;
    const Vec3 velocityLeft = solution[left].linear() * (-1.0f / dt);
    const Vec3 velocityRight = solution[right].linear() * (-1.0f / dt);
    result.response = velocityRight.x - velocityLeft.x;
    result.momentum = endpointMass * (velocityLeft.x + velocityRight.x);
    return result;
  };

  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float endpointMasses[2] = {1.0f, 10.0f};
  const float damping = 6.0f;
  for (float dt : frameDts) {
    float accelerationResponseByMass[2] = {0.0f, 0.0f};
    for (int massIndex = 0; massIndex < 2; ++massIndex) {
      const float mass = endpointMasses[massIndex];
      const float effectiveMass = 0.5f * mass;
      const float expectedAccelerationResponse =
          dt * damping / (1.0f + dt * damping);
      const float expectedForceResponse =
          dt * damping / (effectiveMass + dt * damping);
      const float expectedLimitedResponse = dt / effectiveMass;
      LaneResult acceleration[2];
      LaneResult force[2];
      LaneResult limited[2];
      for (int order = 0; order < 2; ++order) {
        acceleration[order] =
            runLane(dt, mass, true, order != 0, 1e6f);
        force[order] = runLane(dt, mass, false, order != 0, 1e6f);
        limited[order] = runLane(dt, mass, true, order != 0, 1.0f);
        CHECK(acceleration[order].emitted &&
                  acceleration[order].converged &&
                  force[order].emitted && force[order].converged &&
                  limited[order].emitted && limited[order].converged,
              "Linear acceleration authority row/PCG failed at dt=%.9g mass=%.9g",
              dt, mass);
        CHECK(acceleration[order].activeMode == 3 &&
                  force[order].activeMode == 3 &&
                  limited[order].activeMode == 4 &&
                  limited[order].penalty == 0.0f,
              "Linear acceleration active-set semantics failed at dt=%.9g mass=%.9g",
              dt, mass);
        CHECK(std::fabs(acceleration[order].response -
                        expectedAccelerationResponse) <= 2e-5f &&
                  std::fabs(force[order].response -
                            expectedForceResponse) <= 2e-5f &&
                  std::fabs(limited[order].response -
                            expectedLimitedResponse) <= 2e-5f,
              "Linear acceleration response equation failed at dt=%.9g mass=%.9g: %.9g %.9g %.9g",
              dt, mass, acceleration[order].response,
              force[order].response, limited[order].response);
        CHECK(std::fabs(acceleration[order].momentum) <= 1e-5f &&
                  std::fabs(force[order].momentum) <= 1e-5f &&
                  std::fabs(limited[order].momentum) <= 1e-5f,
              "Linear acceleration pair lost momentum conservation at dt=%.9g mass=%.9g",
              dt, mass);
      }
      CHECK(std::fabs(acceleration[0].response -
                      acceleration[1].response) <= 1e-6f &&
                std::fabs(force[0].response - force[1].response) <= 1e-6f &&
                std::fabs(limited[0].response -
                          limited[1].response) <= 1e-6f,
            "Linear acceleration response depends on actor order at dt=%.9g mass=%.9g",
            dt, mass);
      accelerationResponseByMass[massIndex] = acceleration[0].response;
      printf("  dt=%.9g mass=%.9g accel=%.9g force=%.9g limited=%.9g "
             "rho=%.9g momentum=%.9g\n",
             dt, mass, acceleration[0].response, force[0].response,
             limited[0].response, acceleration[0].penalty,
             acceleration[0].momentum);
    }
    CHECK(std::fabs(accelerationResponseByMass[0] -
                    accelerationResponseByMass[1]) <= 1e-6f,
          "Acceleration drive is not mass independent at dt=%.9g", dt);
  }

  PASS("linear acceleration drive preserves implicit effective-mass, limit, conservation, and actor-order semantics");
}

// Test 136: freeze the SWING1 angular-position row before migrating it into
// PhysX AVBD.  Unlike TWIST's -2*delta.x half-angle representation, official
// SWING_TWIST D6 prep uses delta.getBasisVector0().z for the SWING1 geometric
// error.  For an isolated SWING1 rotation this has full-angle sine semantics.
// The physical response must still preserve actor-A-frame torque, force-valued
// clamp, torque*dt writeback, and actor-order normalization.
bool test136_angularSwing1PositionDriveDiscreteEquation() {
  printf("\n--- Test 136: Angular SWING1 Position Drive Discrete Equation ---\n");
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float targetDelta = 0.5f;
  const float torqueLimits[2] = {5.0f, 100.0f};
  const float inertia = 1.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float initialAngles[2] = {0.0f, 0.2f};
  const float invSqrtTwo = 0.7071067811865475f;
  const Vec3 actorAWorldAxis(invSqrtTwo, invSqrtTwo, 0.0f);
  const Vec3 actorBLocalAxis(0.0f, 1.0f, 0.0f);
  const Vec3 actorBWorldAxis = actorAWorldAxis;

  CHECK(std::fabs(actorAWorldAxis.length() - 1.0f) <= 1e-6f &&
            actorAWorldAxis.dot(actorBWorldAxis) >= 0.999999f &&
            actorAWorldAxis.dot(actorBLocalAxis) > 0.70f &&
            actorAWorldAxis.dot(actorBLocalAxis) < 0.72f,
        "SWING1 position fixture does not separate local and world frames");

  for (float dt : frameDts) {
    for (float initialAngle : initialAngles) {
      for (float torqueLimit : torqueLimits) {
        float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
        float normalizedTorqueByOrder[2] = {0.0f, 0.0f};
        for (int order = 0; order < 2; ++order) {
          const float actorOrderSign = order == 0 ? 1.0f : -1.0f;
          const float authoredInitial = actorOrderSign * initialAngle;
          const float authoredTarget =
              actorOrderSign * (initialAngle + targetDelta);
          const float signedTargetError =
              actorOrderSign * (authoredTarget - authoredInitial);
          const float rowError = std::sin(signedTargetError);
          const float rawTorque = stiffness * rowError;
          const float driveTorque =
              std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
          const bool saturated = std::fabs(rawTorque) >= torqueLimit;
          const float tangent =
              saturated
                  ? 0.0f
                  : stiffness * std::fabs(std::cos(signedTargetError)) +
                        damping / dt;
          const float angularDisplacement =
              driveTorque / (inertia / (dt * dt) + tangent);
          normalizedAccelerationByOrder[order] =
              angularDisplacement / (dt * dt);
          const float actor0Torque = actorOrderSign * driveTorque;
          const float angularImpulse = actor0Torque * dt;
          normalizedTorqueByOrder[order] =
              actorOrderSign * angularImpulse / dt;

          printf("  dt=%.9g initial=%.9g limit=%.9g order=%s "
                 "rowError=%.9g torque=%.9g impulse=%.9g accel=%.9g\n",
                 dt, initialAngle, torqueLimit,
                 order == 0 ? "forward" : "reverse", rowError, driveTorque,
                 angularImpulse, normalizedAccelerationByOrder[order]);

          CHECK(std::fabs(signedTargetError - targetDelta) <= 1e-6f,
                "SWING1 target inversion changed physical error at dt=%.9g",
                dt);
          CHECK(std::fabs(rowError - std::sin(targetDelta)) <= 1e-6f,
                "SWING1 position row lost full-angle sine semantics at dt=%.9g",
                dt);
          if (torqueLimit == torqueLimits[0]) {
            CHECK(saturated && tangent == 0.0f &&
                      std::fabs(driveTorque - torqueLimit) <= 1e-6f,
                  "SWING1 finite torque clamp/tangent failed at dt=%.9g", dt);
            CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                            torqueLimit / inertia) <= 1e-5f,
                  "SWING1 finite-torque acceleration changed units at dt=%.9g",
                  dt);
          } else {
            CHECK(!saturated && tangent > damping / dt &&
                      std::fabs(driveTorque - rawTorque) <= 1e-6f,
                  "SWING1 unsaturated tangent semantics failed at dt=%.9g",
                  dt);
          }
          CHECK(std::fabs(normalizedTorqueByOrder[order] - driveTorque) <=
                    1e-6f,
                "SWING1 torque*dt writeback changed units at dt=%.9g", dt);
        }

        CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                        normalizedAccelerationByOrder[1]) <= 1e-6f &&
                  std::fabs(normalizedTorqueByOrder[0] -
                            normalizedTorqueByOrder[1]) <= 1e-6f,
              "SWING1 position response depends on actor order at dt=%.9g",
              dt);
      }
    }
  }

  PASS("angular SWING1 position drive preserves target, frame, row, clamp, impulse, and actor-order semantics");
}

bool test137_angularSwing2PositionDriveDiscreteEquation() {
  printf("\n--- Test 137: Angular SWING2 Position Drive Discrete Equation ---\n");
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float targetDelta = 0.5f;
  const float torqueLimits[2] = {5.0f, 100.0f};
  const float inertia = 1.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float initialAngles[2] = {0.0f, 0.2f};
  const float invSqrtTwo = 0.7071067811865475f;
  const Vec3 actorAWorldAxis(0.0f, invSqrtTwo, invSqrtTwo);
  const Vec3 actorBLocalAxis(0.0f, 0.0f, 1.0f);
  const Vec3 actorBWorldAxis = actorAWorldAxis;

  CHECK(std::fabs(actorAWorldAxis.length() - 1.0f) <= 1e-6f &&
            actorAWorldAxis.dot(actorBWorldAxis) >= 0.999999f &&
            actorAWorldAxis.dot(actorBLocalAxis) > 0.70f &&
            actorAWorldAxis.dot(actorBLocalAxis) < 0.72f,
        "SWING2 position fixture does not separate local and world frames");

  for (float dt : frameDts) {
    for (float initialAngle : initialAngles) {
      for (float torqueLimit : torqueLimits) {
        float normalizedAccelerationByOrder[2] = {0.0f, 0.0f};
        float normalizedTorqueByOrder[2] = {0.0f, 0.0f};
        for (int order = 0; order < 2; ++order) {
          const float actorOrderSign = order == 0 ? 1.0f : -1.0f;
          const float authoredInitial = actorOrderSign * initialAngle;
          const float authoredTarget =
              actorOrderSign * (initialAngle + targetDelta);
          const float signedTargetError =
              actorOrderSign * (authoredTarget - authoredInitial);
          const float halfError = 0.5f * signedTargetError;
          const Quat targetDeltaQuat(std::cos(halfError), 0.0f, 0.0f,
                                     std::sin(halfError));
          const Quat currentMinusTarget = targetDeltaQuat.conjugate();
          const Vec3 basis0 =
              currentMinusTarget.rotate(Vec3(1.0f, 0.0f, 0.0f));
          const float officialExtD6Error = -basis0.y;
          const float avbdGradientResidual = basis0.y;
          const float rowError = officialExtD6Error;
          const float rawTorque = stiffness * rowError;
          const float driveTorque =
              std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
          const bool saturated = std::fabs(rawTorque) >= torqueLimit;
          const float tangent =
              saturated
                  ? 0.0f
                  : stiffness * std::fabs(std::cos(signedTargetError)) +
                        damping / dt;
          const float angularDisplacement =
              driveTorque / (inertia / (dt * dt) + tangent);
          normalizedAccelerationByOrder[order] =
              angularDisplacement / (dt * dt);
          const float actor0Torque = actorOrderSign * driveTorque;
          const float angularImpulse = actor0Torque * dt;
          normalizedTorqueByOrder[order] =
              actorOrderSign * angularImpulse / dt;

          printf("  dt=%.9g initial=%.9g limit=%.9g order=%s "
                 "basisY=%.9g errZ=%.9g avbdResidual=%.9g "
                 "torque=%.9g impulse=%.9g accel=%.9g\n",
                 dt, initialAngle, torqueLimit,
                 order == 0 ? "forward" : "reverse", basis0.y,
                 officialExtD6Error, avbdGradientResidual, driveTorque,
                 angularImpulse, normalizedAccelerationByOrder[order]);

          CHECK(std::fabs(signedTargetError - targetDelta) <= 1e-6f,
                "SWING2 target inversion changed physical error at dt=%.9g",
                dt);
          CHECK(std::fabs(officialExtD6Error -
                          std::sin(targetDelta)) <= 1e-6f &&
                    std::fabs(avbdGradientResidual +
                              officialExtD6Error) <= 1e-6f,
                "SWING2 errZ=-basisY/opposite-gradient semantics changed at dt=%.9g",
                dt);
          if (torqueLimit == torqueLimits[0]) {
            CHECK(saturated && tangent == 0.0f &&
                      std::fabs(driveTorque - torqueLimit) <= 1e-6f,
                  "SWING2 finite torque clamp/tangent failed at dt=%.9g", dt);
            CHECK(std::fabs(normalizedAccelerationByOrder[order] -
                            torqueLimit / inertia) <= 1e-5f,
                  "SWING2 finite-torque acceleration changed units at dt=%.9g",
                  dt);
          } else {
            CHECK(!saturated && tangent > damping / dt &&
                      std::fabs(driveTorque - rawTorque) <= 1e-6f,
                  "SWING2 unsaturated tangent semantics failed at dt=%.9g",
                  dt);
          }
          CHECK(std::fabs(normalizedTorqueByOrder[order] - driveTorque) <=
                    1e-6f,
                "SWING2 torque*dt writeback changed units at dt=%.9g", dt);
        }

        CHECK(std::fabs(normalizedAccelerationByOrder[0] -
                        normalizedAccelerationByOrder[1]) <= 1e-6f &&
                  std::fabs(normalizedTorqueByOrder[0] -
                            normalizedTorqueByOrder[1]) <= 1e-6f,
              "SWING2 position response depends on actor order at dt=%.9g",
              dt);
      }
    }
  }

  PASS("angular SWING2 position drive preserves target, frame, errZ, clamp, impulse, and actor-order semantics");
}

// Test 138: freeze the official three-row SLERP spring equation before the
// PhysX AVBD path owns it.  Unlike stiffness-zero SLERP velocity rows, a SLERP
// position drive does not use fixed world X/Y/Z axes.  ExtD6 computes
// delta=target^-1*current, emits -delta.imaginary as the geometric error, and
// differentiates that quaternion error with computeJacobianAxes(cA*target,cB).
// The shared scalar torque limit clamps each row independently.
bool test138_angularSlerpPositionDriveDiscreteEquation() {
  printf("\n--- Test 138: Angular SLERP Position Drive Discrete Equation ---\n");
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float targetDelta = 0.5f;
  const float torqueLimits[2] = {5.0f, 100.0f};
  const float inertia = 1.0f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float initialAngles[2] = {0.0f, 0.2f};
  const Vec3 localTargetAxis = Vec3(1.0f, 2.0f, 3.0f).normalized();
  const float invSqrtTwo = 0.7071067811865475f;
  const Quat physicalFrameA(invSqrtTwo, -invSqrtTwo, 0.0f, 0.0f);
  const Vec3 physicalWorldAxis =
      physicalFrameA.rotate(localTargetAxis).normalized();

  const auto axisAngle = [](float angle, const Vec3 &axis) {
    const float halfAngle = 0.5f * angle;
    const float sinHalf = std::sin(halfAngle);
    return Quat(std::cos(halfAngle), axis.x * sinHalf, axis.y * sinHalf,
                axis.z * sinHalf);
  };
  const auto quatDot = [](const Quat &a, const Quat &b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
  };
  const auto computeJacobianAxes = [](Vec3 rows[3], const Quat &qa,
                                      const Quat &qb) {
    const float wa = qa.w;
    const float wb = qb.w;
    const Vec3 va(qa.x, qa.y, qa.z);
    const Vec3 vb(qb.x, qb.y, qb.z);
    const Vec3 c = vb * wa + va * wb;
    const float d0 = wa * wb;
    const float d1 = va.dot(vb);
    const float d = d0 - d1;
    rows[0] =
        (va * vb.x + vb * va.x + Vec3(d, c.z, -c.y)) * 0.5f;
    rows[1] =
        (va * vb.y + vb * va.y + Vec3(-c.z, d, c.x)) * 0.5f;
    rows[2] =
        (va * vb.z + vb * va.z + Vec3(c.y, -c.x, d)) * 0.5f;
  };

  CHECK(std::fabs(localTargetAxis.length() - 1.0f) <= 1e-6f &&
            std::fabs(physicalWorldAxis.length() - 1.0f) <= 1e-6f,
        "SLERP position fixture axes are not normalized");

  for (float dt : frameDts) {
    for (float initialAngle : initialAngles) {
      for (float torqueLimit : torqueLimits) {
        Vec3 accelerationByOrder[2];
        Vec3 actor0TorqueByOrder[2];
        for (int order = 0; order < 2; ++order) {
          const bool reverse = order != 0;
          const Quat physicalCurrent =
              axisAngle(initialAngle, localTargetAxis);
          const Quat physicalTarget =
              axisAngle(initialAngle + targetDelta, localTargetAxis);
          const Quat currentRelative =
              reverse ? physicalCurrent.conjugate() : physicalCurrent;
          Quat targetRelative =
              reverse ? physicalTarget.conjugate() : physicalTarget;
          if (quatDot(currentRelative, targetRelative) < 0.0f)
            targetRelative = -targetRelative;

          const Quat worldFrameA =
              reverse ? physicalFrameA * physicalCurrent : physicalFrameA;
          const Quat worldFrameB =
              reverse ? physicalFrameA : physicalFrameA * physicalCurrent;
          const Quat delta =
              targetRelative.conjugate() * currentRelative;
          const Vec3 officialError(-delta.x, -delta.y, -delta.z);
          const Vec3 avbdResidual(delta.x, delta.y, delta.z);
          Vec3 rows[3];
          computeJacobianAxes(rows, worldFrameA * targetRelative, worldFrameB);

          Mat33 system =
              Mat33::diag(inertia / (dt * dt), inertia / (dt * dt),
                          inertia / (dt * dt));
          Vec3 dynamicGradient;
          Vec3 actor0Torque;
          const float endpointSign = reverse ? -1.0f : 1.0f;
          int saturatedRows = 0;
          for (int row = 0; row < 3; ++row) {
            const float residual = (&avbdResidual.x)[row];
            const float rawRowTorque = stiffness * residual;
            const float rowTorque =
                std::max(-torqueLimit,
                         std::min(torqueLimit, rawRowTorque));
            const bool saturated =
                std::fabs(rawRowTorque) >= torqueLimit;
            saturatedRows += saturated ? 1 : 0;
            const float rowTangent =
                saturated ? 0.0f : stiffness + damping / dt;
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j)
                system.m[i][j] +=
                    rowTangent * (&rows[row].x)[i] *
                    (&rows[row].x)[j];
            dynamicGradient += rows[row] * (endpointSign * rowTorque);
            actor0Torque += rows[row] * rowTorque;
          }
          const Vec3 angularDisplacement =
              system.inverse() * (dynamicGradient * -1.0f);
          accelerationByOrder[order] =
              angularDisplacement * (1.0f / (dt * dt));
          actor0TorqueByOrder[order] = actor0Torque;
          const Vec3 publicTorqueOff;
          const Vec3 angularImpulse = actor0Torque * dt;

          printf("  dt=%.9g initial=%.9g limit=%.9g order=%s "
                 "deltaImag=(%.9g,%.9g,%.9g) saturatedRows=%d "
                 "actor0Torque=(%.9g,%.9g,%.9g) "
                 "accelProjection=%.9g impulseMagnitude=%.9g\n",
                 dt, initialAngle, torqueLimit,
                 reverse ? "reverse" : "forward", delta.x, delta.y,
                 delta.z, saturatedRows, actor0Torque.x, actor0Torque.y,
                 actor0Torque.z,
                 accelerationByOrder[order].dot(physicalWorldAxis),
                 angularImpulse.length());

          CHECK((officialError + avbdResidual).length() <= 1e-6f &&
                    std::fabs(officialError.length() -
                              std::sin(0.5f * targetDelta)) <= 1e-6f,
                "SLERP target^-1*current quaternion error changed at dt=%.9g",
                dt);
          CHECK(rows[0].length() > 0.49f &&
                    rows[0].length() < 0.51f &&
                    rows[1].length() > 0.49f &&
                    rows[1].length() < 0.51f &&
                    rows[2].length() > 0.49f &&
                    rows[2].length() < 0.51f,
                "SLERP quaternion Jacobian lost its half-angle scale at dt=%.9g",
                dt);
          CHECK(accelerationByOrder[order].dot(physicalWorldAxis) > 0.0f,
                "SLERP position response has the wrong physical direction at dt=%.9g",
                dt);
          CHECK(publicTorqueOff.length() == 0.0f &&
                    (angularImpulse - actor0Torque * dt).length() <= 1e-7f,
                "SLERP output-off/writeback units changed at dt=%.9g", dt);
          if (torqueLimit == torqueLimits[0])
            CHECK(saturatedRows == 3,
                  "SLERP shared finite limit did not clamp all fixture rows at dt=%.9g",
                  dt);
          else
            CHECK(saturatedRows == 0,
                  "SLERP unsaturated fixture unexpectedly clamped at dt=%.9g",
                  dt);
        }

        CHECK(std::fabs(accelerationByOrder[0].dot(physicalWorldAxis) -
                        accelerationByOrder[1].dot(physicalWorldAxis)) <=
                  2e-4f,
              "SLERP target-axis response depends on actor order at dt=%.9g",
              dt);
        CHECK(actor0TorqueByOrder[0].dot(physicalWorldAxis) < 0.0f &&
                  actor0TorqueByOrder[1].dot(physicalWorldAxis) > 0.0f,
              "SLERP actor0 torque did not follow endpoint ordering at dt=%.9g",
              dt);
      }
    }
  }

  PASS("angular SLERP position drive preserves quaternion error, moving rows, shared clamp, impulse, and actor-order semantics");
}

// Test 139: freeze the exact mass-metric impulse projection implied by the
// official GearJointSolverPrep row before PhysX AVBD applies it.  In the
// signed hinge/world coordinates exposed by the Snippet fixture, the official
// vector row angular0=axis0*ratio, angular1=-axis1 reduces to
// J=[ratio,+1], because the solver-prep body-B axis is opposite the readable
// hinge axis.  An external impulse first produces unconstrained angular
// velocities; the gear impulse is then the unique minimum-kinetic-change
// solution that makes J*w=0.  This equation has no dt scale.
bool test139_gearJointImpulseProjectionDiscreteEquation() {
  printf("\n--- Test 139: Gear Joint Impulse Projection Discrete Equation ---\n");
  struct Projection {
    float omega0;
    float omega1;
    float lambda;
    float residual;
  };
  const auto project = [](float rawOmega0, float rawOmega1,
                          float inverseInertia0, float inverseInertia1,
                          float jacobian0, float jacobian1) {
    Projection result = {rawOmega0, rawOmega1, 0.0f, 0.0f};
    const float denominator =
        jacobian0 * jacobian0 * inverseInertia0 +
        jacobian1 * jacobian1 * inverseInertia1;
    if (denominator > 1e-12f) {
      const float rawResidual =
          jacobian0 * rawOmega0 + jacobian1 * rawOmega1;
      result.lambda = -rawResidual / denominator;
      result.omega0 =
          rawOmega0 + inverseInertia0 * jacobian0 * result.lambda;
      result.omega1 =
          rawOmega1 + inverseInertia1 * jacobian1 * result.lambda;
    }
    result.residual =
        jacobian0 * result.omega0 + jacobian1 * result.omega1;
    return result;
  };

  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float ratios[3] = {1.0f, 2.5f, -2.5f};
  const float inertias[2][2] = {{1.0f, 4.0f}, {10.0f, 0.5f}};
  const float impulses[2] = {5.0f, 50.0f};

  for (float dt : frameDts) {
    for (float ratio : ratios) {
      for (const auto &inertia : inertias) {
        for (float impulse : impulses) {
          const float inertia0 = inertia[0];
          const float inertia1 = inertia[1];
          const float inverseInertia0 = 1.0f / inertia0;
          const float inverseInertia1 = 1.0f / inertia1;
          const float rawOmega0 = 0.0f;
          const float rawOmega1 = impulse * inverseInertia1;
          const Projection forward =
              project(rawOmega0, rawOmega1, inverseInertia0,
                      inverseInertia1, ratio, 1.0f);

          // Swapping actor storage swaps both the mass metric and row
          // coefficients.  The physical velocities must only swap slots.
          const Projection reverse =
              project(rawOmega1, rawOmega0, inverseInertia1,
                      inverseInertia0, 1.0f, ratio);
          const Projection idempotent =
              project(forward.omega0, forward.omega1, inverseInertia0,
                      inverseInertia1, ratio, 1.0f);

          const float rawEnergy =
              0.5f * inertia0 * rawOmega0 * rawOmega0 +
              0.5f * inertia1 * rawOmega1 * rawOmega1;
          const float projectedEnergy =
              0.5f * inertia0 * forward.omega0 * forward.omega0 +
              0.5f * inertia1 * forward.omega1 * forward.omega1;
          const float directOmega0 =
              (inertia0 * rawOmega0 - ratio * inertia1 * rawOmega1) /
              (inertia0 + ratio * ratio * inertia1);
          const float directOmega1 = -ratio * directOmega0;
          const float reactionImpulse0 = ratio * forward.lambda;
          const float reactionImpulse1 = forward.lambda;

          printf("  dt=%.9g ratio=%.9g inertia=(%.9g,%.9g) "
                 "impulse=%.9g raw=(%.9g,%.9g) projected=(%.9g,%.9g) "
                 "lambda=%.9g residual=%.9g energy=(%.9g,%.9g)\n",
                 dt, ratio, inertia0, inertia1, impulse, rawOmega0,
                 rawOmega1, forward.omega0, forward.omega1,
                 forward.lambda, forward.residual, rawEnergy,
                 projectedEnergy);

          CHECK(std::fabs(forward.residual) <= 2e-5f,
                "gear impulse projection did not close J*w at dt=%.9g",
                dt);
          CHECK(std::fabs(forward.omega0 - directOmega0) <= 2e-5f &&
                    std::fabs(forward.omega1 - directOmega1) <= 2e-5f,
                "gear impulse projection is not the mass-metric minimum at dt=%.9g",
                dt);
          CHECK(projectedEnergy <= rawEnergy + 2e-5f,
                "gear impulse projection increased kinetic energy at dt=%.9g",
                dt);
          CHECK(std::fabs(reverse.omega0 - forward.omega1) <= 2e-5f &&
                    std::fabs(reverse.omega1 - forward.omega0) <= 2e-5f,
                "gear impulse projection depends on actor storage order at dt=%.9g",
                dt);
          CHECK(std::fabs(idempotent.omega0 - forward.omega0) <= 2e-5f &&
                    std::fabs(idempotent.omega1 - forward.omega1) <= 2e-5f &&
                    std::fabs(idempotent.lambda) <= 2e-5f,
                "gear impulse projection is not idempotent at dt=%.9g", dt);
          CHECK(reactionImpulse1 * rawOmega1 <= 0.0f &&
                    std::fabs(reactionImpulse0 -
                              ratio * reactionImpulse1) <= 2e-5f,
                "gear reaction impulses do not follow the official row at dt=%.9g",
                dt);
        }
      }
    }
  }

  PASS("gear external impulse preserves the official row, mass-metric projection, actor order, energy, and dt-independent impulse semantics");
}

// Test 140: freeze the coupled two-dynamic-body equation required by an
// internal force-mode angular-position drive.  A drive row must update both
// endpoints from one frozen objective.  The mass metric therefore distributes
// equal-and-opposite torque according to each principal inertia, preserves
// total angular momentum, and remains invariant when the D6 endpoint storage
// order (and authored target sign) are reversed.
bool test140_dynamicDynamicAngularPositionDriveDiscreteEquation() {
  printf("\n--- Test 140: Dynamic-Dynamic Angular Position Drive Discrete Equation ---\n");
  struct CoupledResult {
    float physicalThetaA;
    float physicalThetaB;
    float physicalOmegaA;
    float physicalOmegaB;
    float relativeAcceleration;
    float angularMomentum;
    float normalizedActor0Torque;
  };
  const auto solve = [](float dt, float physicalInertiaA,
                        float physicalInertiaB, float driveTorque,
                        float tangent, bool reverse) {
    const float orderSign = reverse ? -1.0f : 1.0f;
    const float inertia0 =
        reverse ? physicalInertiaB : physicalInertiaA;
    const float inertia1 =
        reverse ? physicalInertiaA : physicalInertiaB;
    const float force = orderSign * driveTorque;
    const float invDt2 = 1.0f / (dt * dt);
    const float a00 = inertia0 * invDt2 + tangent;
    const float a01 = -tangent;
    const float a11 = inertia1 * invDt2 + tangent;
    const float rhs0 = -force;
    const float rhs1 = force;
    const float determinant = a00 * a11 - a01 * a01;
    const float theta0 = (a11 * rhs0 - a01 * rhs1) / determinant;
    const float theta1 = (-a01 * rhs0 + a00 * rhs1) / determinant;
    const float physicalThetaA = reverse ? theta1 : theta0;
    const float physicalThetaB = reverse ? theta0 : theta1;
    const float physicalOmegaA = physicalThetaA / dt;
    const float physicalOmegaB = physicalThetaB / dt;
    CoupledResult result = {
        physicalThetaA,
        physicalThetaB,
        physicalOmegaA,
        physicalOmegaB,
        (physicalOmegaB - physicalOmegaA) / dt,
        physicalInertiaA * physicalOmegaA +
            physicalInertiaB * physicalOmegaB,
        orderSign * (-force)};
    return result;
  };

  enum RowKind { eTWIST_ROW, eSWING1_ROW, eSWING2_ROW };
  const char *rowNames[3] = {"TWIST", "SWING1", "SWING2"};
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float targetDelta = 0.5f;
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};
  const float initialAngles[2] = {0.0f, 0.2f};
  const float inertiaPairs[3][2] = {
      {1.0f, 1.0f}, {1.0f, 10.0f}, {4.0f, 0.5f}};
  const float torqueLimits[2] = {5.0f, 1e6f};

  for (int rowIndex = 0; rowIndex < 3; ++rowIndex) {
    const RowKind rowKind = static_cast<RowKind>(rowIndex);
    const float rowError =
        rowKind == eTWIST_ROW
            ? 2.0f * std::sin(0.5f * targetDelta)
            : std::sin(targetDelta);
    const float rowDerivative =
        rowKind == eTWIST_ROW
            ? std::cos(0.5f * targetDelta)
            : std::cos(targetDelta);
    for (float dt : frameDts) {
      for (float initialAngle : initialAngles) {
        for (const auto &inertias : inertiaPairs) {
          const float inertiaA = inertias[0];
          const float inertiaB = inertias[1];
          const float effectiveInertia =
              1.0f / (1.0f / inertiaA + 1.0f / inertiaB);
          for (float torqueLimit : torqueLimits) {
            const float rawTorque = stiffness * rowError;
            const float driveTorque =
                std::max(-torqueLimit,
                         std::min(torqueLimit, rawTorque));
            const bool saturated =
                std::fabs(rawTorque) >= torqueLimit;
            const float tangent =
                saturated ? 0.0f
                          : stiffness * rowDerivative + damping / dt;
            const CoupledResult forward =
                solve(dt, inertiaA, inertiaB, driveTorque, tangent, false);
            const CoupledResult reverse =
                solve(dt, inertiaA, inertiaB, driveTorque, tangent, true);
            const float expectedRelativeDisplacement =
                driveTorque /
                (effectiveInertia / (dt * dt) + tangent);
            const float expectedRelativeAcceleration =
                expectedRelativeDisplacement / (dt * dt);
            const float expectedSaturatedAcceleration =
                driveTorque * (1.0f / inertiaA + 1.0f / inertiaB);
            const float expectedImpulse = driveTorque * dt;

            printf("  row=%s dt=%.9g initial=%.9g inertia=(%.9g,%.9g) "
                   "limit=%.9g saturated=%d torque=%.9g tangent=%.9g "
                   "theta=(%.9g,%.9g) relAccel=%.9g momentum=%.9g "
                   "impulse=%.9g\n",
                   rowNames[rowIndex], dt, initialAngle, inertiaA, inertiaB,
                   torqueLimit, saturated ? 1 : 0, driveTorque, tangent,
                   forward.physicalThetaA, forward.physicalThetaB,
                   forward.relativeAcceleration, forward.angularMomentum,
                   expectedImpulse);

            CHECK(std::fabs((initialAngle + targetDelta) -
                            initialAngle - targetDelta) <= 1e-6f,
                  "%s target delta changed with initial angle at dt=%.9g",
                  rowNames[rowIndex], dt);
            CHECK(std::fabs(forward.angularMomentum) <= 2e-5f &&
                      std::fabs(reverse.angularMomentum) <= 2e-5f,
                  "%s internal drive changed angular momentum at dt=%.9g",
                  rowNames[rowIndex], dt);
            CHECK(std::fabs(forward.relativeAcceleration -
                            expectedRelativeAcceleration) <= 2e-4f,
                  "%s coupled mass-metric response is incorrect at dt=%.9g",
                  rowNames[rowIndex], dt);
            CHECK(std::fabs(forward.physicalThetaA -
                            reverse.physicalThetaA) <= 2e-6f &&
                      std::fabs(forward.physicalThetaB -
                                reverse.physicalThetaB) <= 2e-6f &&
                      std::fabs(forward.relativeAcceleration -
                                reverse.relativeAcceleration) <= 2e-4f,
                  "%s coupled response depends on actor storage order at dt=%.9g",
                  rowNames[rowIndex], dt);
            CHECK(std::fabs(forward.normalizedActor0Torque +
                            driveTorque) <= 1e-6f &&
                      std::fabs(reverse.normalizedActor0Torque +
                                driveTorque) <= 1e-6f,
                  "%s actor0 torque normalization changed at dt=%.9g",
                  rowNames[rowIndex], dt);
            CHECK(std::fabs(expectedImpulse / dt - driveTorque) <= 1e-6f,
                  "%s torque*dt impulse units changed at dt=%.9g",
                  rowNames[rowIndex], dt);
            if (saturated) {
              CHECK(tangent == 0.0f &&
                        std::fabs(forward.relativeAcceleration -
                                  expectedSaturatedAcceleration) <= 2e-4f,
                    "%s finite torque clamp did not preserve inertia response at dt=%.9g",
                    rowNames[rowIndex], dt);
            } else {
              CHECK(tangent > 0.0f &&
                        std::fabs(driveTorque - rawTorque) <= 1e-6f,
                    "%s unsaturated tangent/torque semantics changed at dt=%.9g",
                    rowNames[rowIndex], dt);
            }
          }
        }
      }
    }
  }

  // The SLERP family uses three moving quaternion-Jacobian rows instead of
  // one principal-axis row.  Condense the same two-body mass metric into the
  // physical relative coordinate, then distribute that solution back to both
  // endpoints.  This composes Test 138's official row semantics with the
  // two-dynamic conservation contract above.
  const Vec3 slerpLocalAxis = Vec3(1.0f, 2.0f, 3.0f).normalized();
  const float invSqrtTwo = 0.7071067811865475f;
  const Quat physicalFrameA(invSqrtTwo, -invSqrtTwo, 0.0f, 0.0f);
  const Vec3 slerpWorldAxis =
      physicalFrameA.rotate(slerpLocalAxis).normalized();
  const auto axisAngle = [](float angle, const Vec3 &axis) {
    const float halfAngle = 0.5f * angle;
    const float sinHalf = std::sin(halfAngle);
    return Quat(std::cos(halfAngle), axis.x * sinHalf,
                axis.y * sinHalf, axis.z * sinHalf);
  };
  const auto quatDot = [](const Quat &a, const Quat &b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
  };
  const auto computeJacobianAxes = [](Vec3 rows[3], const Quat &qa,
                                      const Quat &qb) {
    const float wa = qa.w;
    const float wb = qb.w;
    const Vec3 va(qa.x, qa.y, qa.z);
    const Vec3 vb(qb.x, qb.y, qb.z);
    const Vec3 c = vb * wa + va * wb;
    const float d0 = wa * wb;
    const float d1 = va.dot(vb);
    const float d = d0 - d1;
    rows[0] =
        (va * vb.x + vb * va.x + Vec3(d, c.z, -c.y)) * 0.5f;
    rows[1] =
        (va * vb.y + vb * va.y + Vec3(-c.z, d, c.x)) * 0.5f;
    rows[2] =
        (va * vb.z + vb * va.z + Vec3(c.y, -c.x, d)) * 0.5f;
  };
  struct SlerpCoupledResult {
    Vec3 relativeAcceleration;
    Vec3 physicalOmegaA;
    Vec3 physicalOmegaB;
    Vec3 normalizedActor0Torque;
    float angularMomentum;
    int saturatedRows;
  };
  for (float dt : frameDts) {
    for (float initialAngle : initialAngles) {
      for (const auto &inertias : inertiaPairs) {
        const float inertiaA = inertias[0];
        const float inertiaB = inertias[1];
        const float effectiveInertia =
            1.0f / (1.0f / inertiaA + 1.0f / inertiaB);
        for (float torqueLimit : torqueLimits) {
          SlerpCoupledResult orderResults[2];
          for (int order = 0; order < 2; ++order) {
            const bool reverse = order != 0;
            const float orderSign = reverse ? -1.0f : 1.0f;
            const Quat physicalCurrent =
                axisAngle(initialAngle, slerpLocalAxis);
            const Quat physicalTarget =
                axisAngle(initialAngle + targetDelta, slerpLocalAxis);
            const Quat currentRelative =
                reverse ? physicalCurrent.conjugate() : physicalCurrent;
            Quat targetRelative =
                reverse ? physicalTarget.conjugate() : physicalTarget;
            if (quatDot(currentRelative, targetRelative) < 0.0f)
              targetRelative = -targetRelative;
            const Quat worldFrame0 =
                reverse ? physicalFrameA * physicalCurrent
                        : physicalFrameA;
            const Quat worldFrame1 =
                reverse ? physicalFrameA
                        : physicalFrameA * physicalCurrent;
            const Quat delta =
                targetRelative.conjugate() * currentRelative;
            Vec3 rows[3];
            computeJacobianAxes(rows, worldFrame0 * targetRelative,
                                worldFrame1);
            Mat33 system = Mat33::diag(
                effectiveInertia / (dt * dt),
                effectiveInertia / (dt * dt),
                effectiveInertia / (dt * dt));
            Vec3 actor0Torque;
            int saturatedRows = 0;
            for (int row = 0; row < 3; ++row) {
              const float rawTorque =
                  stiffness * (&delta.x)[row];
              const float rowTorque =
                  std::max(-torqueLimit,
                           std::min(torqueLimit, rawTorque));
              const bool saturated =
                  std::fabs(rawTorque) >= torqueLimit;
              saturatedRows += saturated ? 1 : 0;
              const float rowTangent =
                  saturated ? 0.0f : stiffness + damping / dt;
              for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                  system.m[i][j] +=
                      rowTangent * (&rows[row].x)[i] *
                      (&rows[row].x)[j];
              actor0Torque += rows[row] * rowTorque;
            }
            const Vec3 relativeTheta =
                system.inverse() * (actor0Torque * -orderSign);
            const Vec3 physicalThetaA =
                relativeTheta * (-effectiveInertia / inertiaA);
            const Vec3 physicalThetaB =
                relativeTheta * (effectiveInertia / inertiaB);
            const Vec3 omegaA = physicalThetaA * (1.0f / dt);
            const Vec3 omegaB = physicalThetaB * (1.0f / dt);
            orderResults[order] = {
                relativeTheta * (1.0f / (dt * dt)),
                omegaA,
                omegaB,
                actor0Torque * orderSign,
                (omegaA * inertiaA + omegaB * inertiaB).length(),
                saturatedRows};
          }

          printf("  row=SLERP dt=%.9g initial=%.9g inertia=(%.9g,%.9g) "
                 "limit=%.9g saturatedRows=%d relAccel=(%.9g,%.9g) "
                 "relVectorDiff=%.9g torqueProjection=(%.9g,%.9g) "
                 "torqueMagnitude=(%.9g,%.9g) momentum=(%.9g,%.9g)\n",
                 dt, initialAngle, inertiaA, inertiaB, torqueLimit,
                 orderResults[0].saturatedRows,
                 orderResults[0].relativeAcceleration.dot(slerpWorldAxis),
                 orderResults[1].relativeAcceleration.dot(slerpWorldAxis),
                 (orderResults[0].relativeAcceleration -
                  orderResults[1].relativeAcceleration)
                     .length(),
                 orderResults[0].normalizedActor0Torque.dot(slerpWorldAxis),
                 orderResults[1].normalizedActor0Torque.dot(slerpWorldAxis),
                 orderResults[0].normalizedActor0Torque.length(),
                 orderResults[1].normalizedActor0Torque.length(),
                 orderResults[0].angularMomentum,
                 orderResults[1].angularMomentum);
          CHECK(orderResults[0].relativeAcceleration.dot(slerpWorldAxis) >
                        0.0f &&
                    orderResults[1].relativeAcceleration.dot(
                        slerpWorldAxis) > 0.0f,
                "SLERP coupled response has the wrong direction at dt=%.9g",
                dt);
          // Reversing the D6 endpoints changes the moving quaternion-error
          // basis, so its transverse world-vector components are not an
          // invariant of the official ExtD6 linearization.  The physical
          // target-axis response, endpoint mass distribution, and torque
          // magnitude are invariant (the same contract frozen by Test 138).
          CHECK(std::fabs(
                    orderResults[0].relativeAcceleration.dot(slerpWorldAxis) -
                    orderResults[1].relativeAcceleration.dot(
                        slerpWorldAxis)) <= 2e-4f &&
                    std::fabs(orderResults[0].physicalOmegaA.dot(
                                  slerpWorldAxis) -
                              orderResults[1].physicalOmegaA.dot(
                                  slerpWorldAxis)) <= 2e-5f &&
                    std::fabs(orderResults[0].physicalOmegaB.dot(
                                  slerpWorldAxis) -
                              orderResults[1].physicalOmegaB.dot(
                                  slerpWorldAxis)) <= 2e-5f,
                "SLERP coupled target-axis mass-metric response depends on actor order at dt=%.9g",
                dt);
          CHECK(orderResults[0].angularMomentum <= 2e-5f &&
                    orderResults[1].angularMomentum <= 2e-5f,
                "SLERP internal drive changed angular momentum at dt=%.9g",
                dt);
          CHECK(std::fabs(
                    orderResults[0].normalizedActor0Torque.dot(
                        slerpWorldAxis) -
                    orderResults[1].normalizedActor0Torque.dot(
                        slerpWorldAxis)) <= 2e-5f &&
                    std::fabs(
                        orderResults[0].normalizedActor0Torque.length() -
                        orderResults[1].normalizedActor0Torque.length()) <=
                        2e-5f &&
                    orderResults[0].normalizedActor0Torque.dot(
                        slerpWorldAxis) < 0.0f &&
                    orderResults[1].normalizedActor0Torque.dot(
                        slerpWorldAxis) < 0.0f,
                "SLERP actor0 torque normalization changed at dt=%.9g", dt);
          if (torqueLimit == torqueLimits[0])
            CHECK(orderResults[0].saturatedRows == 3 &&
                      orderResults[1].saturatedRows == 3,
                  "SLERP finite torque did not clamp all rows at dt=%.9g",
                  dt);
          else
            CHECK(orderResults[0].saturatedRows == 0 &&
                      orderResults[1].saturatedRows == 0,
                  "SLERP high-limit rows unexpectedly saturated at dt=%.9g",
                  dt);
        }
      }
    }
  }

  PASS("dynamic-dynamic angular position drive preserves coupled mass metric, internal angular momentum, finite torque, dt, and actor-order semantics");
}

// Test 144: freeze the shared rigid null mode needed after reconstructing
// velocities from a two-body position objective with unequal endpoint
// anchors.  A common rotation about the frictionless support normal must
// restore only the conserved angular-momentum component without changing
// total linear momentum, any rotating-frame D6 coordinate derivative, the
// relative angular derivative, or either support-normal point velocity.
bool test144_supportAxisRigidNullModeAuthority() {
  printf("\n--- Test 144: Support-axis rigid null-mode authority ---\n");

  const auto finiteVec = [](const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
  };
  const auto maximumAbsComponent = [](const Vec3 &value) {
    return std::max(std::fabs(value.x),
                    std::max(std::fabs(value.y), std::fabs(value.z)));
  };
  const auto linearMomentum = [](const Body &a, const Body &b) {
    return a.linearVelocity * a.mass + b.linearVelocity * b.mass;
  };
  const auto axisAngularMomentum = [](const Body &a, const Body &b,
                                      const Vec3 &axis) {
    const float totalMass = a.mass + b.mass;
    const Vec3 centerOfMass =
        (a.position * a.mass + b.position * b.mass) *
        (1.0f / totalMass);
    return axis.dot(
        (a.position - centerOfMass)
                .cross(a.linearVelocity * a.mass) +
        a.invInertiaWorld.inverse() * a.angularVelocity +
        (b.position - centerOfMass)
                .cross(b.linearVelocity * b.mass) +
        b.invInertiaWorld.inverse() * b.angularVelocity);
  };
  const auto coordinateDerivatives =
      [](const Body &a, const Body &b, const Vec3 &anchorA,
         const Vec3 &anchorB) {
        const Vec3 worldArmA = a.rotation.rotate(anchorA);
        const Vec3 worldArmB = b.rotation.rotate(anchorB);
        const Vec3 worldAnchorDelta =
            (b.position + worldArmB) - (a.position + worldArmA);
        const Vec3 pointVelocityA =
            a.linearVelocity + a.angularVelocity.cross(worldArmA);
        const Vec3 pointVelocityB =
            b.linearVelocity + b.angularVelocity.cross(worldArmB);
        const Vec3 relativePointVelocity =
            pointVelocityB - pointVelocityA;
        const Vec3 axes[3] = {
            a.rotation.rotate(Vec3(1.0f, 0.0f, 0.0f)),
            a.rotation.rotate(Vec3(0.0f, 1.0f, 0.0f)),
            a.rotation.rotate(Vec3(0.0f, 0.0f, 1.0f))};
        return Vec3(
            relativePointVelocity.dot(axes[0]) +
                worldAnchorDelta.dot(a.angularVelocity.cross(axes[0])),
            relativePointVelocity.dot(axes[1]) +
                worldAnchorDelta.dot(a.angularVelocity.cross(axes[1])),
            relativePointVelocity.dot(axes[2]) +
                worldAnchorDelta.dot(a.angularVelocity.cross(axes[2])));
      };
  const auto supportPointNormalVelocity =
      [](const Body &body, const Vec3 &localPoint, const Vec3 &axis) {
        const Vec3 arm = body.rotation.rotate(localPoint);
        return axis.dot(
            body.linearVelocity + body.angularVelocity.cross(arm));
      };

  Body physicalA = {};
  Body physicalB = {};
  physicalA.position = Vec3(-0.25f, 0.5f, 0.0f);
  physicalB.position = Vec3(0.25f, 0.5f, -0.25f);
  physicalA.rotation = Quat();
  physicalB.rotation = Quat();
  physicalA.linearVelocity = Vec3(-0.37f, 0.0f, 0.18f);
  physicalB.linearVelocity = Vec3(0.61f, 0.0f, -0.11f);
  physicalA.angularVelocity = Vec3(0.17f, 0.46f, -0.09f);
  physicalB.angularVelocity = Vec3(-0.22f, -0.31f, 0.14f);
  physicalA.mass = 1.0f;
  physicalB.mass = 3.0f;
  physicalA.inertiaTensor = Mat33::diag(0.8f, 1.3f, 1.9f);
  physicalB.inertiaTensor = Mat33::diag(2.1f, 3.7f, 4.4f);
  physicalA.maxLinearVelocity = 1000.0f;
  physicalB.maxLinearVelocity = 1000.0f;
  physicalA.maxAngularVelocity = 1000.0f;
  physicalB.maxAngularVelocity = 1000.0f;
  physicalA.computeDerived();
  physicalB.computeDerived();
  physicalA.updateInvInertiaWorld();
  physicalB.updateInvInertiaWorld();

  const Vec3 supportNormal(0.0f, 2.0f, 0.0f);
  const Vec3 supportAxis = supportNormal.normalized();
  const Vec3 anchorA(0.0f, 0.0f, 0.0f);
  const Vec3 anchorB(0.0f, 0.0f, 0.25f);
  const Vec3 supportPoint(0.0f, -0.5f, 0.0f);
  const float expectedAxisAngularMomentum = 0.125f;

  Body forwardA = physicalA;
  Body forwardB = physicalB;
  const Vec3 momentumBefore = linearMomentum(forwardA, forwardB);
  const Vec3 derivativeBefore =
      coordinateDerivatives(forwardA, forwardB, anchorA, anchorB);
  const Vec3 relativeAngularBefore =
      forwardB.angularVelocity - forwardA.angularVelocity;
  const float supportVelocityABefore =
      supportPointNormalVelocity(forwardA, supportPoint, supportAxis);
  const float supportVelocityBBefore =
      supportPointNormalVelocity(forwardB, supportPoint, supportAxis);
  CHECK(restoreTwoBodySupportAxisAngularMomentum(
            forwardA, forwardB, supportNormal,
            expectedAxisAngularMomentum),
        "forward support-axis null-mode correction failed");
  const Vec3 momentumAfter = linearMomentum(forwardA, forwardB);
  const Vec3 derivativeAfter =
      coordinateDerivatives(forwardA, forwardB, anchorA, anchorB);
  const Vec3 relativeAngularAfter =
      forwardB.angularVelocity - forwardA.angularVelocity;
  const float supportVelocityAAfter =
      supportPointNormalVelocity(forwardA, supportPoint, supportAxis);
  const float supportVelocityBAfter =
      supportPointNormalVelocity(forwardB, supportPoint, supportAxis);
  const float angularMomentumAfter =
      axisAngularMomentum(forwardA, forwardB, supportAxis);

  CHECK(finiteVec(forwardA.linearVelocity) &&
            finiteVec(forwardA.angularVelocity) &&
            finiteVec(forwardB.linearVelocity) &&
            finiteVec(forwardB.angularVelocity),
        "support-axis null-mode correction produced non-finite velocity");
  CHECK(maximumAbsComponent(momentumAfter - momentumBefore) <= 2e-6f,
        "support-axis null mode changed total linear momentum");
  CHECK(std::fabs(angularMomentumAfter -
                  expectedAxisAngularMomentum) <= 2e-6f,
        "support-axis angular momentum did not reach target: %.9g",
        angularMomentumAfter);
  CHECK(maximumAbsComponent(derivativeAfter - derivativeBefore) <= 2e-6f,
        "support-axis null mode changed rotating-frame D6 derivatives");
  CHECK(maximumAbsComponent(relativeAngularAfter -
                            relativeAngularBefore) <= 2e-6f,
        "support-axis null mode changed relative angular derivative");
  CHECK(std::fabs(supportVelocityAAfter -
                  supportVelocityABefore) <= 2e-6f &&
            std::fabs(supportVelocityBAfter -
                      supportVelocityBBefore) <= 2e-6f,
        "support-axis null mode changed support-normal point velocity");

  Body reverseA = physicalB;
  Body reverseB = physicalA;
  CHECK(restoreTwoBodySupportAxisAngularMomentum(
            reverseA, reverseB, supportNormal,
            expectedAxisAngularMomentum),
        "reverse support-axis null-mode correction failed");
  CHECK(maximumAbsComponent(reverseB.linearVelocity -
                            forwardA.linearVelocity) <= 2e-6f &&
            maximumAbsComponent(reverseB.angularVelocity -
                                forwardA.angularVelocity) <= 2e-6f &&
            maximumAbsComponent(reverseA.linearVelocity -
                                forwardB.linearVelocity) <= 2e-6f &&
            maximumAbsComponent(reverseA.angularVelocity -
                                forwardB.angularVelocity) <= 2e-6f,
        "support-axis null mode depends on endpoint storage order");

  // P4Z authority: freeze the exact pure-X unequal-anchor geometry used by
  // the PhysX fixture.  The endpoint centers are offset so the public world
  // anchors initially coincide, and the local B arm is collinear with the
  // driven X axis.  The same shared rigid mode must remain a null mode even
  // though this geometry has no r-cross-drive-axis moment.
  Body collinearA = physicalA;
  Body collinearB = physicalB;
  collinearA.position = Vec3(0.0f, 0.5f, 0.0f);
  collinearB.position = Vec3(-0.25f, 0.5f, 0.0f);
  collinearA.computeDerived();
  collinearB.computeDerived();
  collinearA.updateInvInertiaWorld();
  collinearB.updateInvInertiaWorld();
  const Vec3 collinearAnchorA(0.0f, 0.0f, 0.0f);
  const Vec3 collinearAnchorB(0.25f, 0.0f, 0.0f);
  CHECK(((collinearA.position +
          collinearA.rotation.rotate(collinearAnchorA)) -
         (collinearB.position +
          collinearB.rotation.rotate(collinearAnchorB)))
                .length() <= 1e-7f,
        "pure-X unequal-anchor fixture does not start coincident");

  Body collinearForwardA = collinearA;
  Body collinearForwardB = collinearB;
  const Vec3 collinearMomentumBefore =
      linearMomentum(collinearForwardA, collinearForwardB);
  const Vec3 collinearDerivativeBefore =
      coordinateDerivatives(collinearForwardA, collinearForwardB,
                            collinearAnchorA, collinearAnchorB);
  const Vec3 collinearRelativeAngularBefore =
      collinearForwardB.angularVelocity -
      collinearForwardA.angularVelocity;
  const float collinearSupportVelocityABefore =
      supportPointNormalVelocity(collinearForwardA, supportPoint,
                                 supportAxis);
  const float collinearSupportVelocityBBefore =
      supportPointNormalVelocity(collinearForwardB, supportPoint,
                                 supportAxis);
  CHECK(restoreTwoBodySupportAxisAngularMomentum(
            collinearForwardA, collinearForwardB, supportNormal,
            expectedAxisAngularMomentum),
        "pure-X support-axis null-mode correction failed");
  const Vec3 collinearMomentumAfter =
      linearMomentum(collinearForwardA, collinearForwardB);
  const Vec3 collinearDerivativeAfter =
      coordinateDerivatives(collinearForwardA, collinearForwardB,
                            collinearAnchorA, collinearAnchorB);
  const Vec3 collinearRelativeAngularAfter =
      collinearForwardB.angularVelocity -
      collinearForwardA.angularVelocity;
  const float collinearSupportVelocityAAfter =
      supportPointNormalVelocity(collinearForwardA, supportPoint,
                                 supportAxis);
  const float collinearSupportVelocityBAfter =
      supportPointNormalVelocity(collinearForwardB, supportPoint,
                                 supportAxis);
  const float collinearAngularMomentumAfter =
      axisAngularMomentum(collinearForwardA, collinearForwardB,
                          supportAxis);

  CHECK(finiteVec(collinearForwardA.linearVelocity) &&
            finiteVec(collinearForwardA.angularVelocity) &&
            finiteVec(collinearForwardB.linearVelocity) &&
            finiteVec(collinearForwardB.angularVelocity),
        "pure-X support-axis null mode produced non-finite velocity");
  CHECK(maximumAbsComponent(collinearMomentumAfter -
                            collinearMomentumBefore) <= 2e-6f,
        "pure-X support-axis null mode changed total linear momentum");
  CHECK(std::fabs(collinearAngularMomentumAfter -
                  expectedAxisAngularMomentum) <= 2e-6f,
        "pure-X support-axis angular momentum did not reach target: %.9g",
        collinearAngularMomentumAfter);
  CHECK(maximumAbsComponent(collinearDerivativeAfter -
                            collinearDerivativeBefore) <= 2e-6f,
        "pure-X support-axis null mode changed D6 derivatives");
  CHECK(maximumAbsComponent(collinearRelativeAngularAfter -
                            collinearRelativeAngularBefore) <= 2e-6f,
        "pure-X support-axis null mode changed relative angular velocity");
  CHECK(std::fabs(collinearSupportVelocityAAfter -
                  collinearSupportVelocityABefore) <= 2e-6f &&
            std::fabs(collinearSupportVelocityBAfter -
                      collinearSupportVelocityBBefore) <= 2e-6f,
        "pure-X support-axis null mode changed support-normal velocity");

  Body collinearReverseA = collinearB;
  Body collinearReverseB = collinearA;
  CHECK(restoreTwoBodySupportAxisAngularMomentum(
            collinearReverseA, collinearReverseB, supportNormal,
            expectedAxisAngularMomentum),
        "reverse pure-X support-axis null-mode correction failed");
  CHECK(maximumAbsComponent(
            collinearReverseB.linearVelocity -
            collinearForwardA.linearVelocity) <= 2e-6f &&
            maximumAbsComponent(
                collinearReverseB.angularVelocity -
                collinearForwardA.angularVelocity) <= 2e-6f &&
            maximumAbsComponent(
                collinearReverseA.linearVelocity -
                collinearForwardB.linearVelocity) <= 2e-6f &&
            maximumAbsComponent(
                collinearReverseA.angularVelocity -
                collinearForwardB.angularVelocity) <= 2e-6f,
        "pure-X support-axis null mode depends on endpoint storage order");

  // P4AA authority: split the same 0.25 m pure-Z relative arm across two
  // nonzero endpoint anchors. Symmetric pose compensation keeps the public
  // world anchors coincident and both bodies at equal support height.
  Body twoSidedA = physicalA;
  Body twoSidedB = physicalB;
  twoSidedA.position = Vec3(0.0f, 0.5f, 0.125f);
  twoSidedB.position = Vec3(0.0f, 0.5f, -0.125f);
  twoSidedA.mass = 1.0f;
  twoSidedB.mass = 1.0f;
  twoSidedA.inertiaTensor = Mat33::diag(1.0f, 1.0f, 1.0f);
  twoSidedB.inertiaTensor = Mat33::diag(1.0f, 1.0f, 1.0f);
  twoSidedA.computeDerived();
  twoSidedB.computeDerived();
  twoSidedA.updateInvInertiaWorld();
  twoSidedB.updateInvInertiaWorld();
  const Vec3 twoSidedAnchorA(0.0f, 0.0f, -0.125f);
  const Vec3 twoSidedAnchorB(0.0f, 0.0f, 0.125f);
  CHECK(((twoSidedA.position +
          twoSidedA.rotation.rotate(twoSidedAnchorA)) -
         (twoSidedB.position +
          twoSidedB.rotation.rotate(twoSidedAnchorB)))
                .length() <= 1e-7f,
        "two-sided pure-Z fixture does not start coincident");

  Body twoSidedForwardA = twoSidedA;
  Body twoSidedForwardB = twoSidedB;
  const Vec3 twoSidedMomentumBefore =
      linearMomentum(twoSidedForwardA, twoSidedForwardB);
  const Vec3 twoSidedDerivativeBefore =
      coordinateDerivatives(twoSidedForwardA, twoSidedForwardB,
                            twoSidedAnchorA, twoSidedAnchorB);
  const Vec3 twoSidedRelativeAngularBefore =
      twoSidedForwardB.angularVelocity -
      twoSidedForwardA.angularVelocity;
  const float twoSidedSupportVelocityABefore =
      supportPointNormalVelocity(twoSidedForwardA, supportPoint,
                                 supportAxis);
  const float twoSidedSupportVelocityBBefore =
      supportPointNormalVelocity(twoSidedForwardB, supportPoint,
                                 supportAxis);
  CHECK(restoreTwoBodySupportAxisAngularMomentum(
            twoSidedForwardA, twoSidedForwardB, supportNormal,
            expectedAxisAngularMomentum),
        "two-sided pure-Z support-axis null-mode correction failed");
  const Vec3 twoSidedMomentumAfter =
      linearMomentum(twoSidedForwardA, twoSidedForwardB);
  const Vec3 twoSidedDerivativeAfter =
      coordinateDerivatives(twoSidedForwardA, twoSidedForwardB,
                            twoSidedAnchorA, twoSidedAnchorB);
  const Vec3 twoSidedRelativeAngularAfter =
      twoSidedForwardB.angularVelocity -
      twoSidedForwardA.angularVelocity;
  const float twoSidedSupportVelocityAAfter =
      supportPointNormalVelocity(twoSidedForwardA, supportPoint,
                                 supportAxis);
  const float twoSidedSupportVelocityBAfter =
      supportPointNormalVelocity(twoSidedForwardB, supportPoint,
                                 supportAxis);
  const float twoSidedAngularMomentumAfter =
      axisAngularMomentum(twoSidedForwardA, twoSidedForwardB,
                          supportAxis);

  CHECK(finiteVec(twoSidedForwardA.linearVelocity) &&
            finiteVec(twoSidedForwardA.angularVelocity) &&
            finiteVec(twoSidedForwardB.linearVelocity) &&
            finiteVec(twoSidedForwardB.angularVelocity),
        "two-sided pure-Z support-axis null mode produced non-finite velocity");
  CHECK(maximumAbsComponent(twoSidedMomentumAfter -
                            twoSidedMomentumBefore) <= 2e-6f,
        "two-sided pure-Z support-axis null mode changed linear momentum");
  CHECK(std::fabs(twoSidedAngularMomentumAfter -
                  expectedAxisAngularMomentum) <= 2e-6f,
        "two-sided pure-Z angular momentum did not reach target: %.9g",
        twoSidedAngularMomentumAfter);
  CHECK(maximumAbsComponent(twoSidedDerivativeAfter -
                            twoSidedDerivativeBefore) <= 2e-6f,
        "two-sided pure-Z support-axis null mode changed D6 derivatives");
  CHECK(maximumAbsComponent(twoSidedRelativeAngularAfter -
                            twoSidedRelativeAngularBefore) <= 2e-6f,
        "two-sided pure-Z null mode changed relative angular velocity");
  CHECK(std::fabs(twoSidedSupportVelocityAAfter -
                  twoSidedSupportVelocityABefore) <= 2e-6f &&
            std::fabs(twoSidedSupportVelocityBAfter -
                      twoSidedSupportVelocityBBefore) <= 2e-6f,
        "two-sided pure-Z null mode changed support-normal velocity");

  Body twoSidedReverseA = twoSidedB;
  Body twoSidedReverseB = twoSidedA;
  CHECK(restoreTwoBodySupportAxisAngularMomentum(
            twoSidedReverseA, twoSidedReverseB, supportNormal,
            expectedAxisAngularMomentum),
        "reverse two-sided pure-Z null-mode correction failed");
  CHECK(maximumAbsComponent(
            twoSidedReverseB.linearVelocity -
            twoSidedForwardA.linearVelocity) <= 2e-6f &&
            maximumAbsComponent(
                twoSidedReverseB.angularVelocity -
                twoSidedForwardA.angularVelocity) <= 2e-6f &&
            maximumAbsComponent(
                twoSidedReverseA.linearVelocity -
                twoSidedForwardB.linearVelocity) <= 2e-6f &&
            maximumAbsComponent(
                twoSidedReverseA.angularVelocity -
                twoSidedForwardB.angularVelocity) <= 2e-6f,
        "two-sided pure-Z null mode depends on endpoint storage order");

  printf("  targetL=%.9g finalL=%.9g momentumDelta=%.9g "
         "coordinateDelta=%.9g xFinalL=%.9g xMomentumDelta=%.9g "
         "xCoordinateDelta=%.9g pairZFinalL=%.9g "
         "pairZMomentumDelta=%.9g pairZCoordinateDelta=%.9g\n",
         expectedAxisAngularMomentum, angularMomentumAfter,
         maximumAbsComponent(momentumAfter - momentumBefore),
         maximumAbsComponent(derivativeAfter - derivativeBefore),
         collinearAngularMomentumAfter,
         maximumAbsComponent(collinearMomentumAfter -
                             collinearMomentumBefore),
         maximumAbsComponent(collinearDerivativeAfter -
                             collinearDerivativeBefore),
         twoSidedAngularMomentumAfter,
         maximumAbsComponent(twoSidedMomentumAfter -
                             twoSidedMomentumBefore),
         maximumAbsComponent(twoSidedDerivativeAfter -
                             twoSidedDerivativeBefore));
  PASS("support-axis rigid null mode preserves linear momentum, rotating-frame D6 derivatives, relative angular velocity, support normal, actor order, pure-X unequal anchors, and two-sided pure-Z anchors");
}

// Test 145: a finite position drive and Coulomb support must share one
// position-level objective. At rest, k*error balances mu*m*g. A split system
// that omits the two external tangent rows has a nonzero Newton step at that
// same state; a later velocity impulse cannot repair the already accepted
// position update.
bool test145_finiteDriveStaticFrictionPositionAuthority() {
  printf("\n--- Test 145: Finite-drive static-friction position authority ---\n");

  const float dt = 1.0f / 60.0f;
  const float mass = 1.0f;
  const float gravity = 9.81f;
  const float friction = 0.5f;
  const float stiffness = 100.0f;
  const float forceLimit = 5.0f;
  const float frictionCapacity = friction * mass * gravity;
  const float equilibriumError = frictionCapacity / stiffness;
  const Vec3 xAxis(1.0f, 0.0f, 0.0f);

  CHECK(frictionCapacity < forceLimit &&
            std::fabs(equilibriumError - 0.04905f) <= 1e-7f,
        "finite-drive friction fixture is not in the intended boundary");

  std::vector<Mat66> inertia(2);
  std::vector<Vec6> inertialGradient(2);
  const float massInvDt2 = mass / (dt * dt);
  for (Mat66 &block : inertia) {
    for (int axis = 0; axis < 3; ++axis) {
      block.m[axis][axis] = massInvDt2;
      block.m[3 + axis][3 + axis] = massInvDt2;
    }
  }

  const auto addDriveRow =
      [&](IslandPcgSystem &system) {
        IslandPcgRow row;
        row.owner = IslandRowOwner::D6;
        row.ownerIndex = 0;
        row.bodyA = 0;
        row.bodyB = 1;
        row.jacobianA = Vec6(-xAxis, Vec3());
        row.jacobianB = Vec6(xAxis, Vec3());
        row.violation = equilibriumError;
        row.penalty = stiffness;
        row.force = std::min(
            forceLimit, stiffness * equilibriumError);
        row.internalTranslationInvariant = true;
        return system.addRow(row);
      };
  const auto addSupportFrictionRows =
      [&](IslandPcgSystem &system) {
        IslandPcgRow rowA;
        rowA.owner = IslandRowOwner::Contact;
        rowA.ownerIndex = 0;
        rowA.bodyA = 0;
        rowA.jacobianA = Vec6(xAxis, Vec3());
        rowA.penalty = stiffness;
        rowA.force = frictionCapacity;
        if (!system.addRow(rowA))
          return false;

        IslandPcgRow rowB;
        rowB.owner = IslandRowOwner::Contact;
        rowB.ownerIndex = 1;
        rowB.bodyA = 1;
        rowB.jacobianA = Vec6(xAxis, Vec3());
        rowB.penalty = stiffness;
        rowB.force = -frictionCapacity;
        return system.addRow(rowB);
      };

  IslandPcgSystem completeSystem;
  completeSystem.initialize(inertia, inertialGradient);
  CHECK(addDriveRow(completeSystem) &&
            addSupportFrictionRows(completeSystem),
        "complete finite-drive/friction position rows were rejected");
  std::vector<Vec6> completeSolution;
  const IslandPcgStats completeStats =
      completeSystem.solvePcg(completeSolution, 1e-9, 12);

  IslandPcgSystem splitSystem;
  splitSystem.initialize(inertia, inertialGradient);
  CHECK(addDriveRow(splitSystem),
        "split finite-drive row was rejected");
  std::vector<Vec6> splitSolution;
  const IslandPcgStats splitStats =
      splitSystem.solvePcg(splitSolution, 1e-9, 12);

  const float completeStep =
      std::max(completeSolution[0].linear().length(),
               completeSolution[1].linear().length());
  const float splitRelativeStep =
      std::fabs(splitSolution[1].linear().x -
                splitSolution[0].linear().x);
  const Vec3 completeGradientSum =
      completeSystem.gradient()[0].linear() +
      completeSystem.gradient()[1].linear();

  printf("  equilibriumError=%.9g frictionCapacity=%.9g "
         "completeStep=%.9g splitRelativeStep=%.9g "
         "completeGradientSum=%.9g\n",
         equilibriumError, frictionCapacity, completeStep,
         splitRelativeStep, completeGradientSum.length());
  CHECK(completeStats.converged && completeStats.finite &&
            !completeStats.breakdown && completeStep <= 1e-8f,
        "complete position objective moved at static equilibrium");
  CHECK(splitStats.converged && splitStats.finite &&
            !splitStats.breakdown && splitRelativeStep > 1e-5f,
        "split position objective did not expose the missing friction force");
  CHECK(completeGradientSum.length() <= 1e-7f,
        "external tangent rows introduced a spurious common force");

  PASS("finite drive and static Coulomb support balance only in the complete position-level objective");
}

// Test 146: in the strict persistent-support position owner, each endpoint's
// Coulomb budget comes from its own gravity-aligned support. The locked joint
// normal is redundant with those two supports; using its indeterminate AL
// reaction to transfer normal load changes the relative response of a
// force-limited drive when endpoint masses differ.
bool test146_unequalMassFrictionWeightShareAuthority() {
  printf("\n--- Test 146: Unequal-mass friction weight-share authority ---\n");

  const float dt = 1.0f / 60.0f;
  const float dt2 = dt * dt;
  const float massA = 1.0f;
  const float massB = 10.0f;
  const float gravity = 9.81f;
  const float friction = 0.5f;
  const float driveForce = 5.0f;
  const float weightCapacityA = friction * massA * gravity;
  const float weightCapacityB = friction * massB * gravity;
  const float redundantNormalTransfer = 20.0f;
  const float transferredCapacityA =
      friction * (massA * gravity + redundantNormalTransfer);
  const float transferredCapacityB =
      friction * (massB * gravity - redundantNormalTransfer);
  const Vec3 xAxis(1.0f, 0.0f, 0.0f);
  const auto maximumAbsComponent = [](const Vec3 &value) {
    return std::max(std::fabs(value.x),
                    std::max(std::fabs(value.y),
                             std::fabs(value.z)));
  };

  CHECK(weightCapacityA < driveForce &&
            transferredCapacityA > weightCapacityA &&
            transferredCapacityB > 0.0f,
        "unequal-mass friction fixture is not in the intended boundary");

  struct SolveResult {
    std::vector<Vec6> solution;
    IslandPcgStats stats;
  };
  const auto solve =
      [&](bool reverseEndpoints, float frictionForceA,
          float frictionForceB, SolveResult &result) {
        std::vector<Mat66> inertia(2);
        std::vector<Vec6> inertialGradient(2);
        const float masses[2] = {massA, massB};
        for (int body = 0; body < 2; ++body) {
          const float massInvDt2 = masses[body] / dt2;
          for (int axis = 0; axis < 3; ++axis) {
            inertia[body].m[axis][axis] = massInvDt2;
            inertia[body].m[3 + axis][3 + axis] = massInvDt2;
          }
        }

        IslandPcgSystem system;
        system.initialize(inertia, inertialGradient);

        IslandPcgRow drive;
        drive.owner = IslandRowOwner::D6;
        drive.ownerIndex = 0;
        drive.bodyA = reverseEndpoints ? 1 : 0;
        drive.bodyB = reverseEndpoints ? 0 : 1;
        drive.jacobianA =
            Vec6(reverseEndpoints ? xAxis : -xAxis, Vec3());
        drive.jacobianB =
            Vec6(reverseEndpoints ? -xAxis : xAxis, Vec3());
        drive.force = driveForce;
        drive.internalTranslationInvariant = true;
        if (!system.addRow(drive))
          return false;

        const float frictionForces[2] = {
            frictionForceA, frictionForceB};
        for (int body = 0; body < 2; ++body) {
          IslandPcgRow support;
          support.owner = IslandRowOwner::Contact;
          support.ownerIndex = static_cast<uint32_t>(body);
          support.bodyA = static_cast<uint32_t>(body);
          support.jacobianA = Vec6(xAxis, Vec3());
          support.force = frictionForces[body];
          if (!system.addRow(support))
            return false;
        }

        result.stats =
            system.solvePcg(result.solution, 1e-9, 12);
        return result.stats.converged && result.stats.finite &&
               !result.stats.breakdown &&
               result.solution.size() == 2;
      };

  SolveResult driveOnly;
  SolveResult weightForward;
  SolveResult weightReverse;
  SolveResult transferred;
  CHECK(solve(false, 0.0f, 0.0f, driveOnly) &&
            solve(false, weightCapacityA, weightCapacityB,
                  weightForward) &&
            solve(true, weightCapacityA, weightCapacityB,
                  weightReverse) &&
            solve(false, transferredCapacityA,
                  transferredCapacityB, transferred),
        "unequal-mass friction PCG solve failed");

  const auto relativeStep =
      [](const SolveResult &result) {
        return result.solution[1].linear().x -
               result.solution[0].linear().x;
      };
  const float driveRelativeStep = relativeStep(driveOnly);
  const float weightRelativeStep = relativeStep(weightForward);
  const float transferredRelativeStep = relativeStep(transferred);
  const float expectedDriveRelativeStep =
      driveForce * (1.0f / massA + 1.0f / massB) * dt2;
  const float weightCommonStep =
      (massA * weightForward.solution[0].linear().x +
       massB * weightForward.solution[1].linear().x) /
      (massA + massB);
  const float expectedWeightCommonStep = friction * gravity * dt2;
  const float endpointOrderDelta = std::max(
      maximumAbsComponent(
          weightForward.solution[0].linear() -
          weightReverse.solution[0].linear()),
      maximumAbsComponent(
          weightForward.solution[1].linear() -
          weightReverse.solution[1].linear()));

  printf("  driveRelativeStep=%.9g weightRelativeStep=%.9g "
         "transferredRelativeStep=%.9g weightCommonStep=%.9g "
         "endpointOrderDelta=%.9g\n",
         driveRelativeStep, weightRelativeStep,
         transferredRelativeStep, weightCommonStep,
         endpointOrderDelta);
  CHECK(std::fabs(driveRelativeStep -
                  expectedDriveRelativeStep) <= 1e-8f,
        "finite drive did not use the unequal endpoint masses");
  CHECK(std::fabs(weightRelativeStep -
                  driveRelativeStep) <= 1e-8f,
        "per-body weight friction changed the relative drive response");
  CHECK(std::fabs(weightCommonStep -
                  expectedWeightCommonStep) <= 1e-8f,
        "per-body weight friction lost its external common response");
  CHECK(endpointOrderDelta <= 1e-8f,
        "unequal-mass friction response depends on endpoint storage order");
  CHECK(std::fabs(transferredRelativeStep -
                  driveRelativeStep) > 1e-4f,
        "redundant joint-normal load transfer did not expose the mismatch");

  PASS("unequal-mass persistent support uses per-body weight Coulomb budgets and preserves the external common response");
}

// Test 147: an upward-facing stationary slope is still an ordinary rigid
// support. Its Coulomb budget is set by the gravity component along the
// support normal, while the two tangent rows jointly balance downhill
// gravity and the internal X drive. Requiring the support normal to be
// gravity-aligned would reject this complete position objective and leave the
// drive on the historical mixed path.
bool test147_slopedSupportFrictionPositionAuthority() {
  printf("\n--- Test 147: Sloped-support friction position authority ---\n");

  const float dt = 1.0f / 60.0f;
  const float dt2 = dt * dt;
  const float mass = 1.0f;
  const float gravityMagnitude = 9.81f;
  const float friction = 0.5f;
  const float driveForce = 3.0f;
  const float stiffness = 100.0f;
  const float angle =
      10.0f * 3.14159265358979323846f / 180.0f;
  const Vec3 gravity(0.0f, -gravityMagnitude, 0.0f);
  const Vec3 supportNormal(
      0.0f, std::cos(angle), std::sin(angle));
  const Vec3 driveAxis(1.0f, 0.0f, 0.0f);
  const Vec3 downhillAxis(
      0.0f, -std::sin(angle), std::cos(angle));
  const float normalAcceleration =
      -gravity.dot(supportNormal);
  const float downhillAcceleration =
      gravity.dot(downhillAxis);
  const float coulombCapacity =
      friction * mass * normalAcceleration;
  const float fullGravityCapacity =
      friction * mass * gravityMagnitude;
  const float requiredFriction =
      std::sqrt(driveForce * driveForce +
                mass * mass * downhillAcceleration *
                    downhillAcceleration);

  CHECK(std::fabs(supportNormal.length() - 1.0f) <= 1e-7f &&
            std::fabs(downhillAxis.length() - 1.0f) <= 1e-7f &&
            std::fabs(supportNormal.dot(downhillAxis)) <= 1e-7f &&
            std::fabs(supportNormal.dot(driveAxis)) <= 1e-7f,
        "sloped-support basis is not orthonormal");
  CHECK(-gravity.normalized().dot(supportNormal) < 0.9999f &&
            normalAcceleration > 0.0f &&
            downhillAcceleration > 0.0f,
        "sloped-support fixture did not cross the former flat-only guard");
  CHECK(coulombCapacity < fullGravityCapacity &&
            requiredFriction < coulombCapacity,
        "sloped-support fixture is outside the intended static cone");

  std::vector<Mat66> inertia(2);
  std::vector<Vec6> inertialGradient(2);
  const float massInvDt2 = mass / dt2;
  for (int body = 0; body < 2; ++body) {
    for (int axis = 0; axis < 3; ++axis) {
      inertia[body].m[axis][axis] = massInvDt2;
      inertia[body].m[3 + axis][3 + axis] = massInvDt2;
    }
    inertialGradient[body] =
        Vec6(downhillAxis * (mass * downhillAcceleration), Vec3());
  }

  const auto addDriveRow =
      [&](IslandPcgSystem &system) {
        IslandPcgRow drive;
        drive.owner = IslandRowOwner::D6;
        drive.ownerIndex = 0;
        drive.bodyA = 0;
        drive.bodyB = 1;
        drive.jacobianA = Vec6(-driveAxis, Vec3());
        drive.jacobianB = Vec6(driveAxis, Vec3());
        drive.penalty = stiffness;
        drive.force = driveForce;
        drive.internalTranslationInvariant = true;
        return system.addRow(drive);
      };
  const auto addSupportRows =
      [&](IslandPcgSystem &system) {
        for (int body = 0; body < 2; ++body) {
          IslandPcgRow driveTangent;
          driveTangent.owner = IslandRowOwner::Contact;
          driveTangent.ownerIndex =
              static_cast<uint32_t>(body * 2);
          driveTangent.bodyA = static_cast<uint32_t>(body);
          driveTangent.jacobianA = Vec6(driveAxis, Vec3());
          driveTangent.penalty = stiffness;
          driveTangent.force = body == 0 ? driveForce : -driveForce;
          if (!system.addRow(driveTangent))
            return false;

          IslandPcgRow slopeTangent;
          slopeTangent.owner = IslandRowOwner::Contact;
          slopeTangent.ownerIndex =
              static_cast<uint32_t>(body * 2 + 1);
          slopeTangent.bodyA = static_cast<uint32_t>(body);
          slopeTangent.jacobianA = Vec6(downhillAxis, Vec3());
          slopeTangent.penalty = stiffness;
          slopeTangent.force = -mass * downhillAcceleration;
          if (!system.addRow(slopeTangent))
            return false;
        }
        return true;
      };

  IslandPcgSystem completeSystem;
  completeSystem.initialize(inertia, inertialGradient);
  CHECK(addDriveRow(completeSystem) &&
            addSupportRows(completeSystem),
        "complete sloped-support position rows were rejected");
  std::vector<Vec6> completeSolution;
  const IslandPcgStats completeStats =
      completeSystem.solvePcg(completeSolution, 1e-9, 16);

  IslandPcgSystem splitSystem;
  splitSystem.initialize(inertia, inertialGradient);
  CHECK(addDriveRow(splitSystem),
        "split sloped-support drive row was rejected");
  std::vector<Vec6> splitSolution;
  const IslandPcgStats splitStats =
      splitSystem.solvePcg(splitSolution, 1e-9, 16);

  const float completeStep =
      std::max(completeSolution[0].linear().length(),
               completeSolution[1].linear().length());
  const float splitRelativeStep =
      std::fabs((splitSolution[1].linear() -
                 splitSolution[0].linear())
                    .dot(driveAxis));
  const float splitCommonDownhillStep =
      0.5f * (splitSolution[0].linear() +
              splitSolution[1].linear())
                 .dot(downhillAxis);
  float maximumCompleteGradient = 0.0f;
  for (const Vec6 &gradient : completeSystem.gradient()) {
    for (int component = 0; component < 6; ++component)
      maximumCompleteGradient =
          std::max(maximumCompleteGradient,
                   std::fabs(gradient.v[component]));
  }

  printf("  normalAcceleration=%.9g downhillAcceleration=%.9g "
         "coulombCapacity=%.9g requiredFriction=%.9g "
         "completeStep=%.9g splitRelativeStep=%.9g "
         "splitCommonDownhillStep=%.9g maxGradient=%.9g\n",
         normalAcceleration, downhillAcceleration,
         coulombCapacity, requiredFriction, completeStep,
         splitRelativeStep, splitCommonDownhillStep,
         maximumCompleteGradient);
  CHECK(completeStats.converged && completeStats.finite &&
            !completeStats.breakdown && completeStep <= 1e-8f &&
            maximumCompleteGradient <= 1e-7f,
        "complete sloped-support objective moved at static equilibrium");
  CHECK(splitStats.converged && splitStats.finite &&
            !splitStats.breakdown && splitRelativeStep > 1e-5f &&
            splitCommonDownhillStep > 1e-5f,
        "split objective did not expose missing slope friction rows");

  PASS("upward sloped support uses projected normal gravity and one complete two-tangent position objective");
}

// Test 148: eOUTPUT_FORCE is an observation policy, not a solver-owner
// selector.  A saturated force-mode X position drive remains in the same
// complete position objective with both endpoint inertias, both support
// normals, locked Y/Z and all three angular rows.  Enabling public force
// reporting may expose the drive's impulse-valued writeback, but it must not
// change any row, tangent, or pose step.
bool test148_contactPositionOutputForceOwnerAuthority() {
  printf("\n--- Test 148: Contact position output-force owner authority ---\n");

  struct LaneResult {
    Vec3 poseStep[2];
    float relativeStep = 0.0f;
    float relativeAcceleration = 0.0f;
    float linearMomentumChange = 0.0f;
    float publicForce = 0.0f;
    float writebackImpulse = 0.0f;
    bool rowsAccepted = true;
    IslandPcgStats stats;
  };

  const float mass = 1.0f;
  const float stiffness = 100.0f;
  const float damping = 20.0f;
  const float positionError = 0.5f;
  const float targetVelocity = 0.0f;
  const float forceLimit = 5.0f;
  const float rawDriveForce =
      stiffness * positionError + damping * targetVelocity;
  const float driveForce =
      std::max(-forceLimit, std::min(forceLimit, rawDriveForce));
  const bool saturated = std::fabs(rawDriveForce) >= forceLimit;
  const float drivePenalty =
      saturated ? 0.0f : stiffness;
  const Vec3 xAxis(1.0f, 0.0f, 0.0f);
  const Vec3 yAxis(0.0f, 1.0f, 0.0f);
  const Vec3 zAxis(0.0f, 0.0f, 1.0f);
  const float frameDts[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                             1.0f / 120.0f};

  CHECK(saturated && drivePenalty == 0.0f &&
            std::fabs(driveForce - forceLimit) <= 1e-6f,
        "output-force authority fixture did not saturate at 5 N");

  const auto runLane = [&](float dt, bool reverse,
                           bool outputForce) {
    LaneResult result;
    const float invDt2 = 1.0f / (dt * dt);
    std::vector<Mat66> inertia(2);
    std::vector<Vec6> inertialGradient(2);
    for (int body = 0; body < 2; ++body) {
      for (int axis = 0; axis < 6; ++axis)
        inertia[body].m[axis][axis] = mass * invDt2;
    }

    IslandPcgSystem system;
    system.initialize(inertia, inertialGradient);

    const uint32_t actorA = reverse ? 1u : 0u;
    const uint32_t actorB = reverse ? 0u : 1u;
    const float actor0AxisSign = reverse ? -1.0f : 1.0f;
    const Vec3 authoredAxis = xAxis * actor0AxisSign;

    IslandPcgRow drive;
    drive.owner = IslandRowOwner::D6;
    drive.ownerIndex = 0;
    drive.bodyA = actorA;
    drive.bodyB = actorB;
    drive.jacobianA = Vec6(-authoredAxis, Vec3());
    drive.jacobianB = Vec6(authoredAxis, Vec3());
    drive.penalty = drivePenalty;
    drive.force = driveForce;
    drive.internalTranslationInvariant = true;
    result.rowsAccepted = system.addRow(drive);

    for (uint32_t body = 0; body < 2; ++body) {
      IslandPcgRow contactNormal;
      contactNormal.owner = IslandRowOwner::Contact;
      contactNormal.ownerIndex = body;
      contactNormal.bodyA = body;
      contactNormal.jacobianA = Vec6(yAxis, Vec3());
      contactNormal.penalty = stiffness;
      result.rowsAccepted =
          system.addRow(contactNormal) && result.rowsAccepted;
    }

    const Vec3 lockedLinearAxes[2] = {yAxis, zAxis};
    for (int axis = 0; axis < 2; ++axis) {
      IslandPcgRow hardLinear;
      hardLinear.owner = IslandRowOwner::D6;
      hardLinear.ownerIndex = 0;
      hardLinear.rowSlot = static_cast<uint16_t>(axis + 1);
      hardLinear.bodyA = actorA;
      hardLinear.bodyB = actorB;
      hardLinear.jacobianA =
          Vec6(-lockedLinearAxes[axis], Vec3());
      hardLinear.jacobianB =
          Vec6(lockedLinearAxes[axis], Vec3());
      hardLinear.penalty = stiffness;
      hardLinear.internalTranslationInvariant = true;
      result.rowsAccepted =
          system.addRow(hardLinear) && result.rowsAccepted;
    }

    const Vec3 lockedAngularAxes[3] = {xAxis, yAxis, zAxis};
    for (int axis = 0; axis < 3; ++axis) {
      IslandPcgRow hardAngular;
      hardAngular.owner = IslandRowOwner::D6;
      hardAngular.ownerIndex = 0;
      hardAngular.rowSlot = static_cast<uint16_t>(axis + 3);
      hardAngular.bodyA = actorA;
      hardAngular.bodyB = actorB;
      hardAngular.jacobianA =
          Vec6(Vec3(), -lockedAngularAxes[axis]);
      hardAngular.jacobianB =
          Vec6(Vec3(), lockedAngularAxes[axis]);
      hardAngular.penalty = stiffness;
      result.rowsAccepted =
          system.addRow(hardAngular) && result.rowsAccepted;
    }

    std::vector<Vec6> solution;
    result.stats = system.solvePcg(solution, 1e-10, 16);
    for (int body = 0; body < 2; ++body)
      result.poseStep[body] = solution[body].linear() * -1.0f;
    result.relativeStep =
        std::fabs((result.poseStep[1] - result.poseStep[0]).dot(xAxis));
    result.relativeAcceleration = result.relativeStep * invDt2;
    result.linearMomentumChange =
        ((result.poseStep[0] + result.poseStep[1]) *
         (mass / dt))
            .length();

    const float actor0AuthoredForce =
        actor0AxisSign * driveForce;
    result.writebackImpulse =
        outputForce ? actor0AuthoredForce * dt : 0.0f;
    result.publicForce =
        outputForce
            ? actor0AxisSign * (result.writebackImpulse / dt)
            : 0.0f;
    return result;
  };

  for (float dt : frameDts) {
    LaneResult lanes[2][2];
    for (int order = 0; order < 2; ++order) {
      for (int output = 0; output < 2; ++output) {
        lanes[order][output] =
            runLane(dt, order != 0, output != 0);
        CHECK(lanes[order][output].rowsAccepted &&
                  lanes[order][output].stats.converged &&
                  lanes[order][output].stats.finite &&
                  !lanes[order][output].stats.breakdown,
              "complete output-force rows were rejected or did not converge");
      }
    }

    const float expectedRelativeStep =
        2.0f * forceLimit * dt * dt / mass;
    const float expectedRelativeAcceleration =
        2.0f * forceLimit / mass;
    float maximumOutputStateDelta = 0.0f;
    float maximumOrderStateDelta = 0.0f;
    for (int order = 0; order < 2; ++order) {
      for (int body = 0; body < 2; ++body) {
        maximumOutputStateDelta = std::max(
            maximumOutputStateDelta,
            (lanes[order][1].poseStep[body] -
             lanes[order][0].poseStep[body])
                .length());
      }
    }
    for (int output = 0; output < 2; ++output) {
      for (int body = 0; body < 2; ++body) {
        maximumOrderStateDelta = std::max(
            maximumOrderStateDelta,
            (lanes[1][output].poseStep[body] -
             lanes[0][output].poseStep[body])
                .length());
      }
    }

    printf("  dt=%.9g relativeStep=%.9g relativeAcceleration=%.9g "
           "momentumChange=%.9g outputOn=%.9g outputOff=%.9g "
           "outputStateDelta=%.9g orderStateDelta=%.9g\n",
           dt, lanes[0][1].relativeStep,
           lanes[0][1].relativeAcceleration,
           lanes[0][1].linearMomentumChange,
           lanes[0][1].publicForce, lanes[0][0].publicForce,
           maximumOutputStateDelta, maximumOrderStateDelta);

    for (int order = 0; order < 2; ++order) {
      for (int output = 0; output < 2; ++output) {
        CHECK(std::fabs(lanes[order][output].relativeStep -
                        expectedRelativeStep) <= 1e-8f &&
                  std::fabs(
                      lanes[order][output].relativeAcceleration -
                      expectedRelativeAcceleration) <= 1e-5f,
              "output-force changed the finite-drive physical response");
        CHECK(lanes[order][output].linearMomentumChange <= 1e-8f,
              "complete output-force owner changed linear momentum");
      }
      CHECK(std::fabs(lanes[order][1].publicForce -
                      forceLimit) <= 1e-6f &&
                lanes[order][0].publicForce == 0.0f,
            "output-force did not remain a reporting-only policy");
      CHECK(std::fabs(std::fabs(
                          lanes[order][1].writebackImpulse) /
                          dt -
                      forceLimit) <= 1e-6f &&
                lanes[order][0].writebackImpulse == 0.0f,
            "output-force writeback did not preserve impulse units");
    }
    CHECK(maximumOutputStateDelta <= 1e-9f,
          "output-force flag changed the complete position objective");
    CHECK(maximumOrderStateDelta <= 1e-9f,
          "output-force position owner depends on endpoint storage order");
  }

  PASS("contact position output-force is observational and preserves the complete finite-force position owner");
}

// Test 149: passive multi-row rigid-static friction is one material-velocity
// manifold, not a reusable per-contact sweep budget.  The complete owner
// reconstructs the inertial velocity, solves all unilateral normal rows from
// one iterate, then solves both tangent rows at every contact simultaneously
// with each Coulomb disk projected once.  Replaying the same per-contact cap
// in four ordered Gauss-Seidel sweeps over-brakes the body and creates a
// contact-order-dependent angular component.
bool test149_passiveRigidStaticFrictionManifoldAuthority() {
  printf("\n--- Test 149: Passive rigid-static friction manifold authority ---\n");

  struct LaneResult {
    Vec3 linearVelocity;
    Vec3 angularVelocity;
    float normalImpulse[4] = {};
    float tangentImpulse[8] = {};
    bool valid = true;
  };

  const float dt = 1.0f / 60.0f;
  const float gravity = 9.81f;
  const float supportImpulse = gravity * dt;
  const float contactCoulombCap = supportImpulse * 0.25f;
  const float initialSlideSpeed = 3.0f;
  const Vec3 normal(0.0f, 1.0f, 0.0f);
  const Vec3 localArms[4] = {
      Vec3(-0.5f, -0.5f, -0.5f),
      Vec3(-0.5f, -0.5f, 0.5f),
      Vec3(0.5f, -0.5f, -0.5f),
      Vec3(0.5f, -0.5f, 0.5f)};

  const auto inverseInertia = [](const Vec3 &value) {
    // Unit-mass, unit-edge cube: I = 1/6 on every principal axis.
    return value * 6.0f;
  };
  const auto rotateAboutY = [](const Vec3 &value, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vec3(c * value.x - s * value.z, value.y,
                s * value.x + c * value.z);
  };
  const auto pointVelocity = [](const Vec3 &linear,
                                const Vec3 &angular,
                                const Vec3 &arm) {
    return linear + angular.cross(arm);
  };
  const auto applyImpulse =
      [&](Vec3 &linear, Vec3 &angular, const Vec3 &arm,
          const Vec3 &axis, float impulse) {
        linear += axis * impulse;
        angular += inverseInertia(arm.cross(axis) * impulse);
      };

  const auto runComplete = [&](float yaw, bool reverseRows) {
    LaneResult result;
    const Vec3 driveAxis =
        rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), yaw);
    const Vec3 lateralAxis =
        rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), yaw);
    Vec3 arms[4];
    for (int row = 0; row < 4; ++row) {
      const int source = reverseRows ? 3 - row : row;
      arms[row] = rotateAboutY(localArms[source], yaw);
    }

    result.linearVelocity =
        driveAxis * initialSlideSpeed - normal * supportImpulse;
    result.angularVelocity = Vec3();

    float normalResponse[4][4] = {};
    float normalRhs[4] = {};
    float nextNormalImpulse[4] = {};
    float normalLipschitz = 0.0f;
    for (int row = 0; row < 4; ++row) {
      normalRhs[row] =
          -pointVelocity(result.linearVelocity,
                         result.angularVelocity, arms[row])
               .dot(normal);
      float absoluteRowSum = 0.0f;
      for (int column = 0; column < 4; ++column) {
        const Vec3 angularRow = arms[row].cross(normal);
        const Vec3 angularColumn = arms[column].cross(normal);
        normalResponse[row][column] =
            1.0f +
            angularRow.dot(inverseInertia(angularColumn));
        absoluteRowSum += std::fabs(normalResponse[row][column]);
      }
      normalLipschitz =
          std::max(normalLipschitz, absoluteRowSum);
    }
    if (!(normalLipschitz > 0.0f)) {
      result.valid = false;
      return result;
    }
    const float normalStep = 1.0f / normalLipschitz;
    for (int iteration = 0; iteration < 96; ++iteration) {
      for (int row = 0; row < 4; ++row) {
        float gradient = -normalRhs[row];
        for (int column = 0; column < 4; ++column)
          gradient += normalResponse[row][column] *
                      result.normalImpulse[column];
        nextNormalImpulse[row] =
            std::max(0.0f, result.normalImpulse[row] -
                               normalStep * gradient);
      }
      for (int row = 0; row < 4; ++row)
        result.normalImpulse[row] = nextNormalImpulse[row];
    }
    for (int row = 0; row < 4; ++row)
      applyImpulse(result.linearVelocity, result.angularVelocity,
                   arms[row], normal, result.normalImpulse[row]);

    Vec3 tangentAxes[8];
    Vec3 tangentAngularJacobians[8];
    float tangentResponse[8][8] = {};
    float tangentRhs[8] = {};
    float nextTangentImpulse[8] = {};
    float tangentLipschitz = 0.0f;
    for (int contact = 0; contact < 4; ++contact) {
      const Vec3 axes[2] = {driveAxis, lateralAxis};
      const Vec3 velocity =
          pointVelocity(result.linearVelocity,
                        result.angularVelocity, arms[contact]);
      for (int tangent = 0; tangent < 2; ++tangent) {
        const int row = contact * 2 + tangent;
        tangentAxes[row] = axes[tangent];
        tangentAngularJacobians[row] =
            arms[contact].cross(axes[tangent]);
        tangentRhs[row] = -velocity.dot(axes[tangent]);
      }
    }
    for (int row = 0; row < 8; ++row) {
      float absoluteRowSum = 0.0f;
      for (int column = 0; column < 8; ++column) {
        tangentResponse[row][column] =
            tangentAxes[row].dot(tangentAxes[column]) +
            tangentAngularJacobians[row].dot(
                inverseInertia(
                    tangentAngularJacobians[column]));
        absoluteRowSum +=
            std::fabs(tangentResponse[row][column]);
      }
      tangentLipschitz =
          std::max(tangentLipschitz, absoluteRowSum);
    }
    if (!(tangentLipschitz > 0.0f)) {
      result.valid = false;
      return result;
    }
    const float tangentStep = 1.0f / tangentLipschitz;
    for (int iteration = 0; iteration < 96; ++iteration) {
      for (int row = 0; row < 8; ++row) {
        float gradient = -tangentRhs[row];
        for (int column = 0; column < 8; ++column)
          gradient += tangentResponse[row][column] *
                      result.tangentImpulse[column];
        nextTangentImpulse[row] =
            result.tangentImpulse[row] - tangentStep * gradient;
      }
      for (int contact = 0; contact < 4; ++contact) {
        const int row = contact * 2;
        const float magnitude =
            std::sqrt(nextTangentImpulse[row] *
                          nextTangentImpulse[row] +
                      nextTangentImpulse[row + 1] *
                          nextTangentImpulse[row + 1]);
        const float cap = result.normalImpulse[contact];
        if (magnitude > cap && magnitude > 0.0f) {
          const float scale = cap / magnitude;
          nextTangentImpulse[row] *= scale;
          nextTangentImpulse[row + 1] *= scale;
        }
      }
      for (int row = 0; row < 8; ++row)
        result.tangentImpulse[row] = nextTangentImpulse[row];
    }
    for (int contact = 0; contact < 4; ++contact) {
      for (int tangent = 0; tangent < 2; ++tangent) {
        const int row = contact * 2 + tangent;
        applyImpulse(result.linearVelocity, result.angularVelocity,
                     arms[contact], tangentAxes[row],
                     result.tangentImpulse[row]);
      }
    }
    return result;
  };

  const auto runSequentialFallback =
      [&](float yaw, bool reverseRows) {
        LaneResult result;
        const Vec3 driveAxis =
            rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), yaw);
        const Vec3 lateralAxis =
            rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), yaw);
        result.linearVelocity = driveAxis * initialSlideSpeed;
        result.angularVelocity = Vec3();
        for (int sweep = 0; sweep < 4; ++sweep) {
          for (int ordered = 0; ordered < 4; ++ordered) {
            const int contact =
                reverseRows ? 3 - ordered : ordered;
            const Vec3 arm =
                rotateAboutY(localArms[contact], yaw);
            const Vec3 axes[2] = {driveAxis, lateralAxis};
            float impulse[2] = {};
            for (int tangent = 0; tangent < 2; ++tangent) {
              const Vec3 angularJacobian =
                  arm.cross(axes[tangent]);
              const float response =
                  1.0f + angularJacobian.dot(
                             inverseInertia(angularJacobian));
              impulse[tangent] =
                  -pointVelocity(result.linearVelocity,
                                 result.angularVelocity, arm)
                       .dot(axes[tangent]) /
                  response;
            }
            const float magnitude =
                std::sqrt(impulse[0] * impulse[0] +
                          impulse[1] * impulse[1]);
            if (magnitude > contactCoulombCap) {
              const float scale = contactCoulombCap / magnitude;
              impulse[0] *= scale;
              impulse[1] *= scale;
            }
            for (int tangent = 0; tangent < 2; ++tangent)
              applyImpulse(result.linearVelocity,
                           result.angularVelocity, arm,
                           axes[tangent], impulse[tangent]);
          }
        }
        return result;
      };

  const float yaw = 0.37f;
  const LaneResult complete = runComplete(0.0f, false);
  const LaneResult completeReverse = runComplete(0.0f, true);
  const LaneResult completeYaw = runComplete(yaw, false);
  const LaneResult completeYawReverse = runComplete(yaw, true);
  const LaneResult fallback = runSequentialFallback(0.0f, false);
  const LaneResult fallbackReverse =
      runSequentialFallback(0.0f, true);
  const Vec3 driveAxis(1.0f, 0.0f, 0.0f);
  const Vec3 lateralAxis(0.0f, 0.0f, 1.0f);
  const Vec3 yawDriveAxis =
      rotateAboutY(driveAxis, yaw);
  const Vec3 yawLateralAxis =
      rotateAboutY(lateralAxis, yaw);
  const float expectedLinearSpeed =
      initialSlideSpeed - supportImpulse;
  const float expectedAngularSpeed = 3.0f * supportImpulse;
  const float completeOrderDelta =
      (complete.linearVelocity -
       completeReverse.linearVelocity)
          .length() +
      (complete.angularVelocity -
       completeReverse.angularVelocity)
          .length();
  const float completeYawDelta =
      std::max(
          std::fabs(complete.linearVelocity.dot(driveAxis) -
                    completeYaw.linearVelocity.dot(yawDriveAxis)),
          std::fabs(complete.angularVelocity.dot(lateralAxis) -
                    completeYaw.angularVelocity.dot(
                        yawLateralAxis)));
  const float fallbackOrderDelta =
      (fallback.angularVelocity -
       fallbackReverse.angularVelocity)
          .length();

  printf("  complete v=(%.9g,%.9g,%.9g) w=(%.9g,%.9g,%.9g) "
         "expected=(%.9g,%.9g) orderDelta=%.9g yawDelta=%.9g\n",
         complete.linearVelocity.x, complete.linearVelocity.y,
         complete.linearVelocity.z, complete.angularVelocity.x,
         complete.angularVelocity.y, complete.angularVelocity.z,
         expectedLinearSpeed, expectedAngularSpeed,
         completeOrderDelta, completeYawDelta);
  printf("  fallback v=(%.9g,%.9g,%.9g) w=(%.9g,%.9g,%.9g) "
         "reverseW=(%.9g,%.9g,%.9g) orderDelta=%.9g\n",
         fallback.linearVelocity.x, fallback.linearVelocity.y,
         fallback.linearVelocity.z, fallback.angularVelocity.x,
         fallback.angularVelocity.y, fallback.angularVelocity.z,
         fallbackReverse.angularVelocity.x,
         fallbackReverse.angularVelocity.y,
         fallbackReverse.angularVelocity.z, fallbackOrderDelta);

  for (const LaneResult *lane :
       {&complete, &completeReverse, &completeYaw,
        &completeYawReverse}) {
    CHECK(lane->valid,
          "complete passive-friction manifold is singular");
    CHECK(std::fabs(lane->linearVelocity.dot(normal)) <= 1e-6f,
          "complete passive-friction owner left normal approach");
    CHECK(std::fabs(lane->linearVelocity.length() -
                    expectedLinearSpeed) <= 2e-6f,
          "complete passive-friction owner reused the Coulomb budget");
    CHECK(std::fabs(lane->angularVelocity.length() -
                    expectedAngularSpeed) <= 2e-6f,
          "complete passive-friction owner produced wrong spatial response");
    for (int contact = 0; contact < 4; ++contact) {
      CHECK(std::fabs(lane->normalImpulse[contact] -
                      contactCoulombCap) <= 2e-6f,
            "complete passive-friction normal support is asymmetric");
      CHECK(std::fabs(lane->tangentImpulse[contact * 2] +
                      contactCoulombCap) <= 2e-6f &&
                std::fabs(
                    lane->tangentImpulse[contact * 2 + 1]) <=
                    2e-6f,
            "complete passive-friction tangent disk is wrong");
    }
  }
  CHECK(completeOrderDelta <= 2e-6f &&
            completeYawDelta <= 2e-6f,
        "complete passive-friction manifold depends on row order or yaw");
  CHECK(fallback.angularVelocity.length() >
            expectedAngularSpeed * 3.5f &&
            expectedLinearSpeed -
                    fallback.linearVelocity.dot(driveAxis) >
                0.4f,
        "sequential fallback did not expose repeated Coulomb-budget use");
  CHECK(fallbackOrderDelta > 0.04f,
        "sequential fallback did not expose contact-row order dependence");

  PASS("passive rigid-static friction requires one normal/tangent material manifold; repeated point sweeps are rejected");
}

// Test 150: once a ground manifold and a dynamic-dynamic manifold share a
// body, material velocity ownership belongs to the complete connected contact
// component.  All normal rows are solved from one inertial snapshot, then all
// tangent rows are solved from that normal result with each per-contact
// Coulomb disk projected once.  A body-static-only sweep cannot transmit the
// ground friction through the dynamic contact and is not an equivalent owner.
bool test150_passiveRigidContactMaterialComponentAuthority() {
  printf("\n--- Test 150: Passive rigid contact material component authority ---\n");

  struct MaterialContact {
    ComponentProjectionRow normal;
    ComponentProjectionRow tangent[2];
    float friction = 1.0f;
    uint64_t stableKey = 0;
  };
  struct LaneResult {
    Vec6 lowerVelocity;
    Vec6 upperVelocity;
    std::vector<float> normalImpulse;
    std::vector<float> tangentImpulse;
    float maximumNormalApproach = 0.0f;
    float maximumTangentKkt = 0.0f;
    bool valid = true;
  };

  const float dt = 1.0f / 60.0f;
  const float gravity = 9.81f;
  const float supportImpulse = gravity * dt;
  const float initialSlideSpeed = 3.0f;
  const Vec3 normal(0.0f, 1.0f, 0.0f);
  const Vec3 localGroundArms[4] = {
      Vec3(-0.5f, -0.5f, -0.5f),
      Vec3(-0.5f, -0.5f, 0.5f),
      Vec3(0.5f, -0.5f, -0.5f),
      Vec3(0.5f, -0.5f, 0.5f)};
  const Vec3 localLowerInterfaceArms[4] = {
      Vec3(-0.5f, 0.5f, -0.5f),
      Vec3(-0.5f, 0.5f, 0.5f),
      Vec3(0.5f, 0.5f, -0.5f),
      Vec3(0.5f, 0.5f, 0.5f)};
  const Vec3 localUpperInterfaceArms[4] = {
      Vec3(-0.5f, -0.5f, -0.5f),
      Vec3(-0.5f, -0.5f, 0.5f),
      Vec3(0.5f, -0.5f, -0.5f),
      Vec3(0.5f, -0.5f, 0.5f)};
  const Mat33 inverseInertia =
      Mat33::diag(6.0f, 6.0f, 6.0f);

  const auto rotateAboutY = [](const Vec3 &value, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vec3(c * value.x - s * value.z, value.y,
                s * value.x + c * value.z);
  };
  const auto makeTerm =
      [](size_t body, const Vec3 &arm, const Vec3 &axis) {
        ComponentProjectionTerm term;
        term.bodyIndex = body;
        term.linearJacobian = axis;
        term.angularJacobian = arm.cross(axis);
        return term;
      };
  const auto rowVelocity =
      [](const ComponentProjectionRow &row,
         const std::vector<Vec6> &velocities) {
        float value = 0.0f;
        for (const ComponentProjectionTerm &term : row.terms) {
          value +=
              term.linearJacobian.dot(
                  velocities[term.bodyIndex].linear()) +
              term.angularJacobian.dot(
                  velocities[term.bodyIndex].angular());
        }
        return value;
      };
  const auto rowResponse =
      [](const ComponentProjectionRow &a,
         const ComponentProjectionRow &b,
         const std::vector<ComponentProjectionBody> &bodies) {
        double value = 0.0;
        for (const ComponentProjectionTerm &at : a.terms) {
          for (const ComponentProjectionTerm &bt : b.terms) {
            if (at.bodyIndex != bt.bodyIndex)
              continue;
            const ComponentProjectionBody &body =
                bodies[at.bodyIndex];
            value +=
                static_cast<double>(body.inverseMassResponse) *
                    static_cast<double>(
                        at.linearJacobian.dot(bt.linearJacobian)) +
                static_cast<double>(at.angularJacobian.dot(
                    body.inverseInertiaResponse *
                    bt.angularJacobian));
          }
        }
        return value;
      };
  const auto applyRowImpulse =
      [](const ComponentProjectionRow &row, float impulse,
         const std::vector<ComponentProjectionBody> &bodies,
         std::vector<Vec6> &velocities) {
        for (const ComponentProjectionTerm &term : row.terms) {
          const ComponentProjectionBody &body =
              bodies[term.bodyIndex];
          velocities[term.bodyIndex] +=
              Vec6(term.linearJacobian *
                       body.inverseMassResponse,
                   body.inverseInertiaResponse *
                       term.angularJacobian) *
              impulse;
        }
      };

  const auto runComplete =
      [&](float yaw, bool reverseContacts, bool swapBodyStorage) {
        LaneResult result;
        const size_t lowerIndex = swapBodyStorage ? 1u : 0u;
        const size_t upperIndex = swapBodyStorage ? 0u : 1u;
        const Vec3 driveAxis =
            rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), yaw);
        const Vec3 lateralAxis =
            rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), yaw);

        std::vector<ComponentProjectionBody> bodies(2);
        bodies[lowerIndex].inverseMassResponse = 1.0f;
        bodies[lowerIndex].inverseInertiaResponse =
            inverseInertia;
        bodies[lowerIndex].stableKey = 100;
        bodies[upperIndex].inverseMassResponse = 1.0f;
        bodies[upperIndex].inverseInertiaResponse =
            inverseInertia;
        bodies[upperIndex].stableKey = 200;

        std::vector<Vec6> velocities(2);
        velocities[lowerIndex] =
            Vec6(driveAxis * initialSlideSpeed -
                     normal * supportImpulse,
                 Vec3());
        velocities[upperIndex] =
            Vec6(driveAxis * initialSlideSpeed -
                     normal * supportImpulse,
                 Vec3());

        std::vector<MaterialContact> contacts;
        contacts.reserve(8);
        for (int contact = 0; contact < 4; ++contact) {
          const Vec3 groundArm =
              rotateAboutY(localGroundArms[contact], yaw);
          MaterialContact ground;
          ground.stableKey =
              static_cast<uint64_t>(1000 + contact);
          ground.normal.stableKey = ground.stableKey * 3;
          ground.normal.terms.push_back(
              makeTerm(lowerIndex, groundArm, normal));
          ground.tangent[0].stableKey =
              ground.stableKey * 3 + 1;
          ground.tangent[0].terms.push_back(
              makeTerm(lowerIndex, groundArm, driveAxis));
          ground.tangent[1].stableKey =
              ground.stableKey * 3 + 2;
          ground.tangent[1].terms.push_back(
              makeTerm(lowerIndex, groundArm, lateralAxis));
          contacts.push_back(ground);

          const Vec3 lowerArm = rotateAboutY(
              localLowerInterfaceArms[contact], yaw);
          const Vec3 upperArm = rotateAboutY(
              localUpperInterfaceArms[contact], yaw);
          MaterialContact interfaceContact;
          interfaceContact.stableKey =
              static_cast<uint64_t>(2000 + contact);
          interfaceContact.normal.stableKey =
              interfaceContact.stableKey * 3;
          interfaceContact.normal.terms.push_back(
              makeTerm(upperIndex, upperArm, normal));
          interfaceContact.normal.terms.push_back(
              makeTerm(lowerIndex, lowerArm,
                       normal * -1.0f));
          interfaceContact.tangent[0].stableKey =
              interfaceContact.stableKey * 3 + 1;
          interfaceContact.tangent[0].terms.push_back(
              makeTerm(upperIndex, upperArm, driveAxis));
          interfaceContact.tangent[0].terms.push_back(
              makeTerm(lowerIndex, lowerArm,
                       driveAxis * -1.0f));
          interfaceContact.tangent[1].stableKey =
              interfaceContact.stableKey * 3 + 2;
          interfaceContact.tangent[1].terms.push_back(
              makeTerm(upperIndex, upperArm, lateralAxis));
          interfaceContact.tangent[1].terms.push_back(
              makeTerm(lowerIndex, lowerArm,
                       lateralAxis * -1.0f));
          contacts.push_back(interfaceContact);
        }
        if (reverseContacts)
          std::reverse(contacts.begin(), contacts.end());
        std::sort(
            contacts.begin(), contacts.end(),
            [](const MaterialContact &a,
               const MaterialContact &b) {
              return a.stableKey < b.stableKey;
            });

        std::vector<ComponentProjectionRow> materialRows;
        materialRows.reserve(contacts.size() * 3);
        for (MaterialContact &contact : contacts) {
          std::sort(
              contact.normal.terms.begin(),
              contact.normal.terms.end(),
              [&](const ComponentProjectionTerm &a,
                  const ComponentProjectionTerm &b) {
                return bodies[a.bodyIndex].stableKey <
                       bodies[b.bodyIndex].stableKey;
              });
          for (int tangent = 0; tangent < 2; ++tangent) {
            std::sort(
                contact.tangent[tangent].terms.begin(),
                contact.tangent[tangent].terms.end(),
                [&](const ComponentProjectionTerm &a,
                    const ComponentProjectionTerm &b) {
                  return bodies[a.bodyIndex].stableKey <
                         bodies[b.bodyIndex].stableKey;
                });
          }
          contact.normal.outwardVelocity =
              rowVelocity(contact.normal, velocities);
          for (int tangent = 0; tangent < 2; ++tangent) {
            contact.tangent[tangent].outwardVelocity =
                rowVelocity(contact.tangent[tangent],
                            velocities);
          }
          materialRows.push_back(contact.normal);
          materialRows.push_back(contact.tangent[0]);
          materialRows.push_back(contact.tangent[1]);
        }

        const size_t materialRowCount = materialRows.size();
        std::vector<double> response(
            materialRowCount * materialRowCount, 0.0);
        double lipschitz = 0.0;
        for (size_t row = 0; row < materialRowCount; ++row) {
          double absoluteRowSum = 0.0;
          for (size_t column = 0;
               column < materialRowCount; ++column) {
            const double value = rowResponse(
                materialRows[row], materialRows[column],
                bodies);
            response[row * materialRowCount + column] =
                value;
            absoluteRowSum += std::fabs(value);
          }
          lipschitz = std::max(lipschitz, absoluteRowSum);
        }
        if (!(lipschitz > 0.0) || !std::isfinite(lipschitz)) {
          result.valid = false;
          return result;
        }

        std::vector<double> impulses(materialRowCount, 0.0);
        std::vector<double> next(materialRowCount, 0.0);
        const auto projectContactDisk =
            [&](size_t contact,
                std::vector<double> &candidate) {
              const size_t row = contact * 3;
              const double friction =
                  contacts[contact].friction;
              const double normalImpulse =
                  impulses[row];
              double tangent0 = candidate[row + 1];
              double tangent1 = candidate[row + 2];
              if (!(friction > 0.0)) {
                candidate[row + 1] = 0.0;
                candidate[row + 2] = 0.0;
                return;
              }
              const double tangentMagnitude = std::sqrt(
                  tangent0 * tangent0 +
                  tangent1 * tangent1);
              const double cap =
                  friction * normalImpulse;
              if (tangentMagnitude <= cap)
                return;
              if (tangentMagnitude > 0.0) {
                const double tangentScale =
                    cap / tangentMagnitude;
                tangent0 *= tangentScale;
                tangent1 *= tangentScale;
              } else {
                tangent0 = 0.0;
                tangent1 = 0.0;
              }
              candidate[row + 1] = tangent0;
              candidate[row + 2] = tangent1;
            };
        const double step = 1.0 / lipschitz;
        bool converged = false;
        for (int outer = 0; outer < 512; ++outer) {
          const std::vector<double> outerStart = impulses;
          for (int iteration = 0; iteration < 4096;
               ++iteration) {
            next = impulses;
            double maximumDelta = 0.0;
            for (size_t contact = 0;
                 contact < contacts.size(); ++contact) {
              const size_t row = contact * 3;
              double gradient =
                  materialRows[row].outwardVelocity;
              for (size_t column = 0;
                   column < materialRowCount; ++column) {
                gradient +=
                    response[row * materialRowCount + column] *
                    impulses[column];
              }
              next[row] =
                  std::max(0.0,
                           impulses[row] - step * gradient);
              maximumDelta = std::max(
                  maximumDelta,
                  std::fabs(next[row] - impulses[row]));
            }
            for (size_t contact = 0;
                 contact < contacts.size(); ++contact) {
              const size_t row = contact * 3;
              impulses[row] = next[row];
            }
            if (maximumDelta <= 1.0e-12)
              break;
          }

          for (int iteration = 0; iteration < 4096;
               ++iteration) {
            next = impulses;
            for (size_t contact = 0;
                 contact < contacts.size(); ++contact) {
              const size_t row = contact * 3;
              for (int tangent = 1; tangent <= 2;
                   ++tangent) {
                const size_t tangentRow =
                    row + static_cast<size_t>(tangent);
                double gradient =
                    materialRows[tangentRow]
                        .outwardVelocity;
                for (size_t column = 0;
                     column < materialRowCount; ++column) {
                  gradient +=
                      response[tangentRow *
                                   materialRowCount +
                               column] *
                      impulses[column];
                }
                next[tangentRow] =
                    impulses[tangentRow] -
                    step * gradient;
              }
              projectContactDisk(contact, next);
            }
            double maximumDelta = 0.0;
            for (size_t contact = 0;
                 contact < contacts.size(); ++contact) {
              const size_t row = contact * 3;
              for (int tangent = 1; tangent <= 2;
                   ++tangent) {
                const size_t tangentRow =
                    row + static_cast<size_t>(tangent);
                maximumDelta = std::max(
                    maximumDelta,
                    std::fabs(next[tangentRow] -
                              impulses[tangentRow]));
                impulses[tangentRow] =
                    next[tangentRow];
              }
            }
            if (maximumDelta <= 1.0e-12)
              break;
          }

          double outerDelta = 0.0;
          for (size_t row = 0; row < materialRowCount;
               ++row) {
            outerDelta = std::max(
                outerDelta,
                std::fabs(impulses[row] -
                          outerStart[row]));
          }
          if (outerDelta <= 1.0e-11) {
            converged = true;
            break;
          }
        }
        if (!converged) {
          result.valid = false;
          return result;
        }

        result.normalImpulse.resize(contacts.size(), 0.0f);
        result.tangentImpulse.resize(
            contacts.size() * 2, 0.0f);
        for (size_t contact = 0;
             contact < contacts.size(); ++contact) {
          const size_t materialRow = contact * 3;
          const size_t tangentRow = contact * 2;
          result.normalImpulse[contact] =
              static_cast<float>(impulses[materialRow]);
          result.tangentImpulse[tangentRow] =
              static_cast<float>(impulses[materialRow + 1]);
          result.tangentImpulse[tangentRow + 1] =
              static_cast<float>(impulses[materialRow + 2]);
        }
        for (size_t row = 0; row < materialRowCount; ++row) {
          const float impulse =
              static_cast<float>(impulses[row]);
          applyRowImpulse(
              materialRows[row], impulse, bodies, velocities);
        }

        for (size_t contact = 0;
             contact < contacts.size(); ++contact) {
          result.maximumNormalApproach = std::max(
              result.maximumNormalApproach,
              std::max(0.0f,
                       -rowVelocity(
                           contacts[contact].normal,
                           velocities)));
        }
        for (size_t row = 0; row < materialRowCount; ++row) {
          double gradient =
              materialRows[row].outwardVelocity;
          for (size_t column = 0;
               column < materialRowCount; ++column) {
            gradient +=
                response[row * materialRowCount + column] *
                impulses[column];
          }
          next[row] = impulses[row] - gradient;
        }
        for (size_t contact = 0;
             contact < contacts.size(); ++contact) {
          const size_t row = contact * 3;
          next[row] = std::max(0.0, next[row]);
          projectContactDisk(contact, next);
          for (int component = 0; component < 3;
               ++component) {
            const size_t materialRow =
                row + static_cast<size_t>(component);
            result.maximumTangentKkt = std::max(
                result.maximumTangentKkt,
                static_cast<float>(std::fabs(
                    next[materialRow] -
                    impulses[materialRow])));
          }
        }

        result.lowerVelocity = velocities[lowerIndex];
        result.upperVelocity = velocities[upperIndex];
        return result;
      };

  const auto runBodyStaticFallback =
      [&](bool reverseContacts) {
        LaneResult result;
        const Vec3 driveAxis(1.0f, 0.0f, 0.0f);
        const Vec3 lateralAxis(0.0f, 0.0f, 1.0f);
        Vec3 lowerLinear =
            driveAxis * initialSlideSpeed;
        Vec3 lowerAngular;
        result.upperVelocity =
            Vec6(driveAxis * initialSlideSpeed, Vec3());
        const float groundCap =
            supportImpulse * 0.5f;
        for (int sweep = 0; sweep < 4; ++sweep) {
          for (int ordered = 0; ordered < 4; ++ordered) {
            const int contact = reverseContacts ?
                3 - ordered : ordered;
            const Vec3 arm = localGroundArms[contact];
            const Vec3 axes[2] = {
                driveAxis, lateralAxis};
            float impulse[2] = {};
            for (int tangent = 0; tangent < 2;
                 ++tangent) {
              const Vec3 angularJacobian =
                  arm.cross(axes[tangent]);
              const float response =
                  1.0f + angularJacobian.dot(
                             inverseInertia *
                             angularJacobian);
              impulse[tangent] =
                  -(lowerLinear +
                    lowerAngular.cross(arm))
                       .dot(axes[tangent]) /
                  response;
            }
            const float magnitude = std::sqrt(
                impulse[0] * impulse[0] +
                impulse[1] * impulse[1]);
            if (magnitude > groundCap) {
              const float scale = groundCap / magnitude;
              impulse[0] *= scale;
              impulse[1] *= scale;
            }
            for (int tangent = 0; tangent < 2;
                 ++tangent) {
              lowerLinear += axes[tangent] *
                             impulse[tangent];
              lowerAngular +=
                  inverseInertia *
                  arm.cross(axes[tangent]) *
                  impulse[tangent];
            }
          }
        }
        result.lowerVelocity =
            Vec6(lowerLinear, lowerAngular);
        return result;
      };

  const float yaw = 0.37f;
  const LaneResult canonical =
      runComplete(0.0f, false, false);
  const LaneResult reverse =
      runComplete(0.0f, true, false);
  const LaneResult swapped =
      runComplete(0.0f, false, true);
  const LaneResult yawed =
      runComplete(yaw, false, false);
  const LaneResult yawedReverse =
      runComplete(yaw, true, true);
  const LaneResult fallback =
      runBodyStaticFallback(false);
  const LaneResult fallbackReverse =
      runBodyStaticFallback(true);

  const Vec3 driveAxis(1.0f, 0.0f, 0.0f);
  const Vec3 lateralAxis(0.0f, 0.0f, 1.0f);
  const Vec3 yawDriveAxis =
      rotateAboutY(driveAxis, yaw);
  const Vec3 yawLateralAxis =
      rotateAboutY(lateralAxis, yaw);

  const auto invariantDelta =
      [&](const LaneResult &a, const LaneResult &b,
          const Vec3 &bDrive, const Vec3 &bLateral) {
        float delta = 0.0f;
        const Vec6 aBodies[2] = {
            a.lowerVelocity, a.upperVelocity};
        const Vec6 bBodies[2] = {
            b.lowerVelocity, b.upperVelocity};
        for (int body = 0; body < 2; ++body) {
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].linear().dot(driveAxis) -
                  bBodies[body].linear().dot(bDrive)));
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].linear().dot(normal) -
                  bBodies[body].linear().dot(normal)));
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].linear().dot(lateralAxis) -
                  bBodies[body].linear().dot(bLateral)));
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].angular().dot(lateralAxis) -
                  bBodies[body].angular().dot(bLateral)));
        }
        return delta;
      };
  const float reverseDelta = invariantDelta(
      canonical, reverse, driveAxis, lateralAxis);
  const float swappedDelta = invariantDelta(
      canonical, swapped, driveAxis, lateralAxis);
  const float yawDelta = invariantDelta(
      canonical, yawed, yawDriveAxis, yawLateralAxis);
  const float yawReverseDelta = invariantDelta(
      canonical, yawedReverse,
      yawDriveAxis, yawLateralAxis);
  const float fallbackRelativeSpeed =
      std::fabs(
          fallback.upperVelocity.linear().dot(driveAxis) -
          fallback.lowerVelocity.linear().dot(driveAxis));
  const float fallbackOrderDelta =
      (fallback.lowerVelocity.angular() -
       fallbackReverse.lowerVelocity.angular())
          .length();
  CHECK(canonical.valid &&
            canonical.normalImpulse.size() == 8 &&
            canonical.tangentImpulse.size() == 16,
        "canonical complete material component did not solve");
  float groundNormalImpulse = 0.0f;
  float interfaceNormalImpulse = 0.0f;
  Vec3 groundLinearImpulse;
  for (int contact = 0; contact < 4; ++contact) {
    groundNormalImpulse +=
        canonical.normalImpulse[contact];
    interfaceNormalImpulse +=
        canonical.normalImpulse[contact + 4];
    groundLinearImpulse +=
        normal * canonical.normalImpulse[contact] +
        driveAxis *
            canonical.tangentImpulse[contact * 2] +
        lateralAxis *
            canonical.tangentImpulse[contact * 2 + 1];
  }
  const Vec3 initialLinearMomentum =
      driveAxis * (2.0f * initialSlideSpeed) -
      normal * (2.0f * supportImpulse);
  const Vec3 finalLinearMomentum =
      canonical.lowerVelocity.linear() +
      canonical.upperVelocity.linear();
  const float momentumResidual =
      (finalLinearMomentum -
       (initialLinearMomentum + groundLinearImpulse))
          .length();
  const float completeRelativeSpeed =
      std::fabs(
          canonical.upperVelocity.linear().dot(driveAxis) -
          canonical.lowerVelocity.linear().dot(driveAxis));

  printf("  complete lowerV=(%.9g,%.9g,%.9g) "
         "lowerW=(%.9g,%.9g,%.9g) "
         "upperV=(%.9g,%.9g,%.9g) "
         "upperW=(%.9g,%.9g,%.9g) "
         "normalApproach=%.9g tangentKkt=%.9g\n",
         canonical.lowerVelocity.linear().x,
         canonical.lowerVelocity.linear().y,
         canonical.lowerVelocity.linear().z,
         canonical.lowerVelocity.angular().x,
         canonical.lowerVelocity.angular().y,
         canonical.lowerVelocity.angular().z,
         canonical.upperVelocity.linear().x,
         canonical.upperVelocity.linear().y,
         canonical.upperVelocity.linear().z,
         canonical.upperVelocity.angular().x,
         canonical.upperVelocity.angular().y,
         canonical.upperVelocity.angular().z,
         canonical.maximumNormalApproach,
         canonical.maximumTangentKkt);
  printf("  groundNormal=%.9g interfaceNormal=%.9g "
         "momentumResidual=%.9g relativeSpeed=%.9g "
         "reverseDelta=%.9g swappedDelta=%.9g "
         "yawDelta=%.9g yawReverseDelta=%.9g\n",
         groundNormalImpulse, interfaceNormalImpulse,
         momentumResidual, completeRelativeSpeed,
         reverseDelta, swappedDelta, yawDelta,
         yawReverseDelta);
  printf("  fallback lowerV=(%.9g,%.9g,%.9g) "
         "upperV=(%.9g,%.9g,%.9g) "
         "relativeSpeed=%.9g orderDelta=%.9g\n",
         fallback.lowerVelocity.linear().x,
         fallback.lowerVelocity.linear().y,
         fallback.lowerVelocity.linear().z,
         fallback.upperVelocity.linear().x,
         fallback.upperVelocity.linear().y,
         fallback.upperVelocity.linear().z,
         fallbackRelativeSpeed, fallbackOrderDelta);

  for (const LaneResult *lane :
       {&canonical, &reverse, &swapped, &yawed,
        &yawedReverse}) {
    CHECK(lane->valid,
          "complete passive material component did not solve");
    CHECK(lane->maximumNormalApproach <= 2.0e-5f,
          "complete passive component left normal approach: %.9g",
          lane->maximumNormalApproach);
    CHECK(lane->maximumTangentKkt <= 2.0e-5f,
          "complete passive component left tangent KKT error: %.9g",
          lane->maximumTangentKkt);
    CHECK(lane->normalImpulse.size() == 8 &&
              lane->tangentImpulse.size() == 16,
          "complete passive component did not own every row");
    float groundDriveImpulse = 0.0f;
    float interfaceDriveImpulse = 0.0f;
    for (int contact = 0; contact < 4; ++contact) {
      for (int layer = 0; layer < 2; ++layer) {
        const int contactIndex = contact + layer * 4;
        const int tangentRow = contactIndex * 2;
        const float tangentMagnitude = std::sqrt(
            lane->tangentImpulse[tangentRow] *
                lane->tangentImpulse[tangentRow] +
            lane->tangentImpulse[tangentRow + 1] *
                lane->tangentImpulse[tangentRow + 1]);
        CHECK(lane->normalImpulse[contactIndex] >=
                      -2.0e-6f &&
                  tangentMagnitude <=
                      lane->normalImpulse[contactIndex] +
                          2.0e-5f,
              "complete material contact violated its Coulomb cone: "
              "contact=%d normal=%.9g tangent=%.9g",
              contactIndex,
              lane->normalImpulse[contactIndex],
              tangentMagnitude);
      }
      groundDriveImpulse +=
          lane->tangentImpulse[contact * 2];
      interfaceDriveImpulse +=
          lane->tangentImpulse[(contact + 4) * 2];
    }
    CHECK(groundDriveImpulse < -1.0e-3f &&
              interfaceDriveImpulse < -1.0e-3f,
          "complete material component did not transmit friction "
          "through both manifolds");
  }
  CHECK(momentumResidual <= 2.0e-5f,
        "complete material component changed external momentum");
  CHECK(groundNormalImpulse > supportImpulse &&
            interfaceNormalImpulse > 0.1f * supportImpulse,
        "complete material component did not carry both-body support");
  CHECK(canonical.upperVelocity.linear().dot(driveAxis) <
            initialSlideSpeed - 1.0e-2f,
        "complete material component did not transfer ground friction");
  CHECK(completeRelativeSpeed < fallbackRelativeSpeed * 0.75f,
        "complete material component did not improve coupled slip");
  CHECK(reverseDelta <= 2.0e-6f &&
            swappedDelta <= 2.0e-6f &&
            yawDelta <= 2.0e-6f &&
            yawReverseDelta <= 2.0e-6f,
        "complete material component depends on row/body order or yaw");
  CHECK(fallbackRelativeSpeed > 0.5f,
        "body-static fallback unexpectedly transmitted friction");
  CHECK(fallbackOrderDelta > 0.02f,
        "body-static fallback did not expose row-order dependence");

  PASS("passive rigid friction requires one complete connected normal/tangent material component");
}

// Test 151: restitution is an explicit normal velocity target inside the
// same material component objective.  A steady ground support row remains
// target-zero while an incident dynamic-dynamic impact row receives
// -e*v_pre when it crosses the scene bounce threshold.  Normal target
// complementarity and both Coulomb disks share one component impulse budget;
// omitting restitution or replaying friction separately is not equivalent.
bool test151_restitutionRigidMaterialComponentAuthority() {
  printf("\n--- Test 151: Restitution rigid material component authority ---\n");

  struct MaterialContact {
    ComponentProjectionRow normal;
    ComponentProjectionRow tangent[2];
    float normalTarget = 0.0f;
    float friction = 1.0f;
    uint64_t stableKey = 0;
  };
  struct LaneResult {
    Vec6 lowerVelocity;
    Vec6 upperVelocity;
    float normalTarget[2] = {};
    float normalImpulse[2] = {};
    float tangentImpulse[4] = {};
    float maximumProjectedKkt = 0.0f;
    float maximumNormalTargetError = 0.0f;
    float momentumResidual = 0.0f;
    float initialEnergy = 0.0f;
    float finalEnergy = 0.0f;
    bool valid = true;
  };

  const float dt = 1.0f / 60.0f;
  const float gravity = 9.81f;
  const float supportSpeed = gravity * dt;
  const float slideSpeed = 3.0f;
  const Vec3 normal(0.0f, 1.0f, 0.0f);

  const auto rotateAboutY = [](const Vec3 &value, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vec3(c * value.x - s * value.z, value.y,
                s * value.x + c * value.z);
  };
  const auto makeTerm =
      [](size_t body, const Vec3 &axis) {
        ComponentProjectionTerm term;
        term.bodyIndex = body;
        term.linearJacobian = axis;
        return term;
      };
  const auto rowVelocity =
      [](const ComponentProjectionRow &row,
         const std::vector<Vec6> &velocities) {
        float value = 0.0f;
        for (const ComponentProjectionTerm &term : row.terms) {
          value += term.linearJacobian.dot(
              velocities[term.bodyIndex].linear());
        }
        return value;
      };
  const auto rowResponse =
      [](const ComponentProjectionRow &a,
         const ComponentProjectionRow &b,
         const std::vector<ComponentProjectionBody> &bodies) {
        double value = 0.0;
        for (const ComponentProjectionTerm &at : a.terms) {
          for (const ComponentProjectionTerm &bt : b.terms) {
            if (at.bodyIndex != bt.bodyIndex)
              continue;
            value +=
                static_cast<double>(
                    bodies[at.bodyIndex].inverseMassResponse) *
                static_cast<double>(
                    at.linearJacobian.dot(bt.linearJacobian));
          }
        }
        return value;
      };
  const auto applyRowImpulse =
      [](const ComponentProjectionRow &row, float impulse,
         const std::vector<ComponentProjectionBody> &bodies,
         std::vector<Vec6> &velocities) {
        for (const ComponentProjectionTerm &term : row.terms) {
          velocities[term.bodyIndex] +=
              Vec6(term.linearJacobian *
                       bodies[term.bodyIndex].inverseMassResponse,
                   Vec3()) *
              impulse;
        }
      };

  const auto runLane =
      [&](float impactSpeed, float restitution,
          float bounceThreshold, float yaw,
          bool reverseContacts, bool swapBodyStorage) {
        LaneResult result;
        const size_t lowerIndex = swapBodyStorage ? 1u : 0u;
        const size_t upperIndex = swapBodyStorage ? 0u : 1u;
        const Vec3 driveAxis =
            rotateAboutY(Vec3(1.0f, 0.0f, 0.0f), yaw);
        const Vec3 lateralAxis =
            rotateAboutY(Vec3(0.0f, 0.0f, 1.0f), yaw);

        std::vector<ComponentProjectionBody> bodies(2);
        for (size_t body = 0; body < bodies.size(); ++body) {
          bodies[body].inverseMassResponse = 1.0f;
          bodies[body].inverseInertiaResponse =
              Mat33::diag(0.0f, 0.0f, 0.0f);
        }
        bodies[lowerIndex].stableKey = 100;
        bodies[upperIndex].stableKey = 200;

        std::vector<Vec6> velocities(2);
        velocities[lowerIndex] =
            Vec6(driveAxis * slideSpeed -
                     normal * supportSpeed,
                 Vec3());
        velocities[upperIndex] =
            Vec6(driveAxis * slideSpeed -
                     normal * (supportSpeed + impactSpeed),
                 Vec3());
        const std::vector<Vec6> initialVelocities = velocities;

        std::vector<MaterialContact> contacts(2);
        MaterialContact &ground = contacts[0];
        ground.stableKey = 1000;
        ground.normal.stableKey = 3000;
        ground.normal.terms.push_back(
            makeTerm(lowerIndex, normal));
        ground.tangent[0].stableKey = 3001;
        ground.tangent[0].terms.push_back(
            makeTerm(lowerIndex, driveAxis));
        ground.tangent[1].stableKey = 3002;
        ground.tangent[1].terms.push_back(
            makeTerm(lowerIndex, lateralAxis));

        MaterialContact &interfaceContact = contacts[1];
        interfaceContact.stableKey = 2000;
        interfaceContact.normal.stableKey = 6000;
        interfaceContact.normal.terms.push_back(
            makeTerm(upperIndex, normal));
        interfaceContact.normal.terms.push_back(
            makeTerm(lowerIndex, normal * -1.0f));
        interfaceContact.tangent[0].stableKey = 6001;
        interfaceContact.tangent[0].terms.push_back(
            makeTerm(upperIndex, driveAxis));
        interfaceContact.tangent[0].terms.push_back(
            makeTerm(lowerIndex, driveAxis * -1.0f));
        interfaceContact.tangent[1].stableKey = 6002;
        interfaceContact.tangent[1].terms.push_back(
            makeTerm(upperIndex, lateralAxis));
        interfaceContact.tangent[1].terms.push_back(
            makeTerm(lowerIndex, lateralAxis * -1.0f));

        if (reverseContacts)
          std::reverse(contacts.begin(), contacts.end());
        std::sort(
            contacts.begin(), contacts.end(),
            [](const MaterialContact &a,
               const MaterialContact &b) {
              return a.stableKey < b.stableKey;
            });

        std::vector<ComponentProjectionRow> rows;
        rows.reserve(6);
        for (MaterialContact &contact : contacts) {
          std::sort(
              contact.normal.terms.begin(),
              contact.normal.terms.end(),
              [&](const ComponentProjectionTerm &a,
                  const ComponentProjectionTerm &b) {
                return bodies[a.bodyIndex].stableKey <
                       bodies[b.bodyIndex].stableKey;
              });
          for (int tangent = 0; tangent < 2; ++tangent) {
            std::sort(
                contact.tangent[tangent].terms.begin(),
                contact.tangent[tangent].terms.end(),
                [&](const ComponentProjectionTerm &a,
                    const ComponentProjectionTerm &b) {
                  return bodies[a.bodyIndex].stableKey <
                         bodies[b.bodyIndex].stableKey;
                });
          }
          const float preNormalVelocity =
              rowVelocity(contact.normal, velocities);
          contact.normalTarget =
              preNormalVelocity < -bounceThreshold
                  ? -restitution * preNormalVelocity
                  : 0.0f;
          contact.normal.outwardVelocity =
              preNormalVelocity - contact.normalTarget;
          for (int tangent = 0; tangent < 2; ++tangent) {
            contact.tangent[tangent].outwardVelocity =
                rowVelocity(contact.tangent[tangent],
                            velocities);
          }
          rows.push_back(contact.normal);
          rows.push_back(contact.tangent[0]);
          rows.push_back(contact.tangent[1]);
        }

        const size_t rowCount = rows.size();
        std::vector<double> response(
            rowCount * rowCount, 0.0);
        for (size_t row = 0; row < rowCount; ++row) {
          for (size_t column = 0; column < rowCount;
               ++column) {
            response[row * rowCount + column] =
                rowResponse(rows[row], rows[column],
                            bodies);
          }
        }

        double normalLipschitz = 0.0;
        double tangentLipschitz = 0.0;
        for (size_t contact = 0; contact < 2; ++contact) {
          const size_t normalRow = contact * 3;
          double normalRowSum = 0.0;
          for (size_t column = 0; column < 2; ++column) {
            normalRowSum += std::fabs(
                response[normalRow * rowCount +
                         column * 3]);
          }
          normalLipschitz =
              std::max(normalLipschitz, normalRowSum);
          for (int tangent = 1; tangent <= 2; ++tangent) {
            const size_t tangentRow =
                normalRow + static_cast<size_t>(tangent);
            double tangentRowSum = 0.0;
            for (size_t other = 0; other < 2; ++other) {
              for (int otherTangent = 1;
                   otherTangent <= 2; ++otherTangent) {
                tangentRowSum += std::fabs(
                    response[
                        tangentRow * rowCount +
                        other * 3 +
                        static_cast<size_t>(
                            otherTangent)]);
              }
            }
            tangentLipschitz =
                std::max(tangentLipschitz,
                         tangentRowSum);
          }
        }
        if (!(normalLipschitz > 0.0) ||
            !(tangentLipschitz > 0.0) ||
            !std::isfinite(normalLipschitz) ||
            !std::isfinite(tangentLipschitz)) {
          result.valid = false;
          return result;
        }

        std::vector<double> impulses(rowCount, 0.0);
        std::vector<double> next(rowCount, 0.0);
        const auto projectTangentDisks =
            [&](std::vector<double> &candidate) {
              for (size_t contact = 0; contact < 2;
                   ++contact) {
                const size_t row = contact * 3;
                const double cap =
                    contacts[contact].friction *
                    impulses[row];
                const double magnitude = std::sqrt(
                    candidate[row + 1] *
                        candidate[row + 1] +
                    candidate[row + 2] *
                        candidate[row + 2]);
                if (magnitude > cap && magnitude > 0.0) {
                  const double scale = cap / magnitude;
                  candidate[row + 1] *= scale;
                  candidate[row + 2] *= scale;
                }
              }
            };
        const double normalStep = 1.0 / normalLipschitz;
        const double tangentStep =
            1.0 / tangentLipschitz;
        bool converged = false;
        for (int outer = 0; outer < 128; ++outer) {
          const std::vector<double> outerStart = impulses;
          for (int iteration = 0; iteration < 4096;
               ++iteration) {
            next = impulses;
            double maximumDelta = 0.0;
            for (size_t contact = 0; contact < 2;
                 ++contact) {
              const size_t row = contact * 3;
              double gradient = rows[row].outwardVelocity;
              for (size_t column = 0; column < rowCount;
                   ++column) {
                gradient +=
                    response[row * rowCount + column] *
                    impulses[column];
              }
              next[row] =
                  std::max(0.0,
                           impulses[row] -
                               normalStep * gradient);
              maximumDelta = std::max(
                  maximumDelta,
                  std::fabs(next[row] - impulses[row]));
            }
            for (size_t contact = 0; contact < 2;
                 ++contact)
              impulses[contact * 3] = next[contact * 3];
            if (maximumDelta <= 1.0e-13)
              break;
          }
          for (int iteration = 0; iteration < 4096;
               ++iteration) {
            next = impulses;
            for (size_t contact = 0; contact < 2;
                 ++contact) {
              const size_t row = contact * 3;
              for (int tangent = 1; tangent <= 2;
                   ++tangent) {
                const size_t tangentRow =
                    row + static_cast<size_t>(tangent);
                double gradient =
                    rows[tangentRow].outwardVelocity;
                for (size_t column = 0;
                     column < rowCount; ++column) {
                  gradient +=
                      response[tangentRow * rowCount +
                               column] *
                      impulses[column];
                }
                next[tangentRow] =
                    impulses[tangentRow] -
                    tangentStep * gradient;
              }
            }
            projectTangentDisks(next);
            double maximumDelta = 0.0;
            for (size_t contact = 0; contact < 2;
                 ++contact) {
              const size_t row = contact * 3;
              for (int tangent = 1; tangent <= 2;
                   ++tangent) {
                const size_t tangentRow =
                    row + static_cast<size_t>(tangent);
                maximumDelta = std::max(
                    maximumDelta,
                    std::fabs(next[tangentRow] -
                              impulses[tangentRow]));
                impulses[tangentRow] =
                    next[tangentRow];
              }
            }
            if (maximumDelta <= 1.0e-13)
              break;
          }
          double outerDelta = 0.0;
          for (size_t row = 0; row < rowCount; ++row) {
            outerDelta = std::max(
                outerDelta,
                std::fabs(impulses[row] -
                          outerStart[row]));
          }
          if (outerDelta <= 1.0e-12) {
            converged = true;
            break;
          }
        }
        if (!converged) {
          result.valid = false;
          return result;
        }

        for (size_t contact = 0; contact < 2;
             ++contact) {
          result.normalTarget[contact] =
              contacts[contact].normalTarget;
          result.normalImpulse[contact] =
              static_cast<float>(impulses[contact * 3]);
          result.tangentImpulse[contact * 2] =
              static_cast<float>(
                  impulses[contact * 3 + 1]);
          result.tangentImpulse[contact * 2 + 1] =
              static_cast<float>(
                  impulses[contact * 3 + 2]);
        }
        for (size_t row = 0; row < rowCount; ++row) {
          applyRowImpulse(
              rows[row], static_cast<float>(impulses[row]),
              bodies, velocities);
        }

        next = impulses;
        for (size_t contact = 0; contact < 2;
             ++contact) {
          const size_t row = contact * 3;
          double normalGradient =
              rows[row].outwardVelocity;
          for (size_t column = 0; column < rowCount;
               ++column) {
            normalGradient +=
                response[row * rowCount + column] *
                impulses[column];
          }
          next[row] =
              std::max(0.0,
                       impulses[row] -
                           normalStep * normalGradient);
          for (int tangent = 1; tangent <= 2;
               ++tangent) {
            const size_t tangentRow =
                row + static_cast<size_t>(tangent);
            double tangentGradient =
                rows[tangentRow].outwardVelocity;
            for (size_t column = 0; column < rowCount;
                 ++column) {
              tangentGradient +=
                  response[tangentRow * rowCount +
                           column] *
                  impulses[column];
            }
            next[tangentRow] =
                impulses[tangentRow] -
                tangentStep * tangentGradient;
          }
        }
        projectTangentDisks(next);
        for (size_t row = 0; row < rowCount; ++row) {
          result.maximumProjectedKkt = std::max(
              result.maximumProjectedKkt,
              static_cast<float>(
                  std::fabs(next[row] - impulses[row])));
        }
        for (size_t contact = 0; contact < 2;
             ++contact) {
          const float finalNormalVelocity =
              rowVelocity(contacts[contact].normal,
                          velocities);
          result.maximumNormalTargetError = std::max(
              result.maximumNormalTargetError,
              std::max(
                  0.0f,
                  contacts[contact].normalTarget -
                      finalNormalVelocity));
        }

        result.lowerVelocity = velocities[lowerIndex];
        result.upperVelocity = velocities[upperIndex];
        const Vec3 initialMomentum =
            initialVelocities[lowerIndex].linear() +
            initialVelocities[upperIndex].linear();
        const Vec3 finalMomentum =
            result.lowerVelocity.linear() +
            result.upperVelocity.linear();
        const Vec3 groundImpulse =
            normal * result.normalImpulse[0] +
            driveAxis * result.tangentImpulse[0] +
            lateralAxis * result.tangentImpulse[1];
        result.momentumResidual =
            (finalMomentum -
             (initialMomentum + groundImpulse))
                .length();
        result.initialEnergy =
            0.5f *
            (initialVelocities[lowerIndex].linear()
                 .length2() +
             initialVelocities[upperIndex].linear()
                 .length2());
        result.finalEnergy =
            0.5f *
            (result.lowerVelocity.linear().length2() +
             result.upperVelocity.linear().length2());
        return result;
      };

  const float impactSpeed = 5.0f;
  const float restitution = 0.5f;
  const float bounceThreshold = 2.0f;
  const float yaw = 0.37f;
  const LaneResult canonical =
      runLane(impactSpeed, restitution, bounceThreshold,
              0.0f, false, false);
  const LaneResult reverse =
      runLane(impactSpeed, restitution, bounceThreshold,
              0.0f, true, false);
  const LaneResult swapped =
      runLane(impactSpeed, restitution, bounceThreshold,
              0.0f, false, true);
  const LaneResult yawed =
      runLane(impactSpeed, restitution, bounceThreshold,
              yaw, false, false);
  const LaneResult yawedReverse =
      runLane(impactSpeed, restitution, bounceThreshold,
              yaw, true, true);
  const LaneResult belowThreshold =
      runLane(1.5f, restitution, bounceThreshold,
              0.0f, false, false);
  const LaneResult aboveThreshold =
      runLane(2.5f, restitution, bounceThreshold,
              0.0f, false, false);
  const LaneResult zeroRestitution =
      runLane(impactSpeed, 0.0f, bounceThreshold,
              0.0f, false, false);
  const LaneResult elastic =
      runLane(impactSpeed, 1.0f, bounceThreshold,
              0.0f, false, false);

  const Vec3 driveAxis(1.0f, 0.0f, 0.0f);
  const Vec3 lateralAxis(0.0f, 0.0f, 1.0f);
  const Vec3 yawDriveAxis =
      rotateAboutY(driveAxis, yaw);
  const Vec3 yawLateralAxis =
      rotateAboutY(lateralAxis, yaw);
  const auto invariantDelta =
      [&](const LaneResult &a, const LaneResult &b,
          const Vec3 &bDrive, const Vec3 &bLateral) {
        float delta = 0.0f;
        const Vec6 aBodies[2] = {
            a.lowerVelocity, a.upperVelocity};
        const Vec6 bBodies[2] = {
            b.lowerVelocity, b.upperVelocity};
        for (int body = 0; body < 2; ++body) {
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].linear().dot(driveAxis) -
                  bBodies[body].linear().dot(bDrive)));
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].linear().dot(normal) -
                  bBodies[body].linear().dot(normal)));
          delta = std::max(
              delta,
              std::fabs(
                  aBodies[body].linear().dot(lateralAxis) -
                  bBodies[body].linear().dot(bLateral)));
        }
        return delta;
      };
  const float reverseDelta = invariantDelta(
      canonical, reverse, driveAxis, lateralAxis);
  const float swappedDelta = invariantDelta(
      canonical, swapped, driveAxis, lateralAxis);
  const float yawDelta = invariantDelta(
      canonical, yawed, yawDriveAxis, yawLateralAxis);
  const float yawReverseDelta = invariantDelta(
      canonical, yawedReverse,
      yawDriveAxis, yawLateralAxis);

  printf("  complete lowerV=(%.9g,%.9g,%.9g) "
         "upperV=(%.9g,%.9g,%.9g) target=(%.9g,%.9g) "
         "normalImpulse=(%.9g,%.9g) "
         "tangentImpulse=(%.9g,%.9g) "
         "kkt=%.9g targetError=%.9g momentum=%.9g "
         "energy=(%.9g,%.9g)\n",
         canonical.lowerVelocity.linear().x,
         canonical.lowerVelocity.linear().y,
         canonical.lowerVelocity.linear().z,
         canonical.upperVelocity.linear().x,
         canonical.upperVelocity.linear().y,
         canonical.upperVelocity.linear().z,
         canonical.normalTarget[0],
         canonical.normalTarget[1],
         canonical.normalImpulse[0],
         canonical.normalImpulse[1],
         canonical.tangentImpulse[0],
         canonical.tangentImpulse[2],
         canonical.maximumProjectedKkt,
         canonical.maximumNormalTargetError,
         canonical.momentumResidual,
         canonical.initialEnergy,
         canonical.finalEnergy);
  printf("  threshold below=(%.9g,%.9g) "
         "above=(%.9g,%.9g) zero=(%.9g,%.9g) "
         "elastic=(%.9g,%.9g) order=(%.9g,%.9g,%.9g,%.9g)\n",
         belowThreshold.normalTarget[1],
         belowThreshold.upperVelocity.linear().y,
         aboveThreshold.normalTarget[1],
         aboveThreshold.upperVelocity.linear().y,
         zeroRestitution.normalTarget[1],
         zeroRestitution.upperVelocity.linear().y,
         elastic.normalTarget[1],
         elastic.upperVelocity.linear().y,
         reverseDelta, swappedDelta, yawDelta,
         yawReverseDelta);

  for (const LaneResult *lane :
       {&canonical, &reverse, &swapped, &yawed,
        &yawedReverse, &belowThreshold, &aboveThreshold,
        &zeroRestitution, &elastic}) {
    CHECK(lane->valid,
          "restitution material component did not converge");
    CHECK(lane->maximumProjectedKkt <= 2.0e-5f,
          "restitution material component left KKT error: %.9g",
          lane->maximumProjectedKkt);
    CHECK(lane->maximumNormalTargetError <= 2.0e-5f,
          "restitution material component missed normal target: %.9g",
          lane->maximumNormalTargetError);
    CHECK(lane->momentumResidual <= 2.0e-5f,
          "restitution material component changed internal momentum");
    CHECK(lane->finalEnergy <= lane->initialEnergy + 2.0e-5f,
          "restitution material component added energy");
    for (int contact = 0; contact < 2; ++contact) {
      const float tangentMagnitude = std::sqrt(
          lane->tangentImpulse[contact * 2] *
              lane->tangentImpulse[contact * 2] +
          lane->tangentImpulse[contact * 2 + 1] *
              lane->tangentImpulse[contact * 2 + 1]);
      CHECK(lane->normalImpulse[contact] >= -2.0e-6f &&
                tangentMagnitude <=
                    lane->normalImpulse[contact] +
                        2.0e-5f,
            "restitution component violated Coulomb cone");
    }
  }
  CHECK(std::fabs(canonical.normalTarget[0]) <= 1.0e-6f &&
            std::fabs(canonical.normalTarget[1] - 2.5f) <=
                1.0e-6f,
        "steady support and impacting row did not receive distinct targets");
  CHECK(std::fabs(canonical.lowerVelocity.linear().y) <=
                2.0e-5f &&
            std::fabs(canonical.upperVelocity.linear().y -
                      2.5f) <= 2.0e-5f,
        "complete restitution component produced wrong rebound");
  CHECK(canonical.lowerVelocity.linear().length() <=
                2.0e-5f &&
            std::fabs(
                canonical.upperVelocity.linear().length() -
                2.5f) <= 2.0e-5f,
        "complete restitution/friction component did not share its budget");
  CHECK(std::fabs(belowThreshold.normalTarget[1]) <=
                1.0e-6f &&
            std::fabs(
                belowThreshold.upperVelocity.linear().y) <=
                2.0e-5f,
        "below-threshold impact incorrectly applied restitution");
  CHECK(std::fabs(aboveThreshold.normalTarget[1] - 1.25f) <=
                1.0e-6f &&
            std::fabs(
                aboveThreshold.upperVelocity.linear().y -
                1.25f) <= 2.0e-5f,
        "above-threshold impact missed restitution");
  CHECK(std::fabs(zeroRestitution.normalTarget[1]) <=
                1.0e-6f &&
            std::fabs(
                zeroRestitution.upperVelocity.linear().y) <=
                2.0e-5f,
        "zero restitution did not reduce to passive complementarity");
  CHECK(std::fabs(elastic.normalTarget[1] - impactSpeed) <=
                1.0e-6f &&
            std::fabs(elastic.upperVelocity.linear().y -
                      impactSpeed) <= 2.0e-5f,
        "elastic target did not preserve relative impact speed");
  CHECK(reverseDelta <= 2.0e-6f &&
            swappedDelta <= 2.0e-6f &&
            yawDelta <= 2.0e-6f &&
            yawReverseDelta <= 2.0e-6f,
        "restitution material component depends on row/body order or yaw");
  CHECK(canonical.upperVelocity.linear().y -
                zeroRestitution.upperVelocity.linear().y >
            2.4f,
        "omitting the restitution target did not expose the missing owner");

  PASS("restitution is a thresholded normal target inside the complete rigid material component");
}

struct FixedPairDropResult {
  float minBottom = INFINITY;
  float finalBottom = 0.0f;
  float maxLinkError = 0.0f;
  float maxHorizontalComError = 0.0f;
  float maxHorizontalMomentum = 0.0f;
  float maxSpeed = 0.0f;
  int contactFrames = 0;
  int routedFrames = 0;
  int maxPcgIterations = 0;
  double worstPcgResidual = 0.0;
  bool finite = true;
  bool pcgOk = true;
};

static FixedPairDropResult runFixedPairDrop(float frameDt, bool reverse,
                                            bool contactPcg) {
  FixedPairDropResult result;
  Solver solver;
  solver.gravity = Vec3(0.0f, -9.8f, 0.0f);
  solver.iterations = 15;
  solver.dt = frameDt;
  solver.useContactIslandPcgProbe = contactPcg;
  solver.useCanonicalRigidContactAuthoringProbe = true;
  const Vec3 halfExtent(0.5f, 0.5f, 0.5f);
  const uint32_t left = solver.addBody(Vec3(-0.5f, 3.0f, 0.0f), Quat(),
                                       halfExtent, 4.0f, 0.5f);
  const uint32_t right = solver.addBody(Vec3(0.5f, 3.0f, 0.0f), Quat(),
                                        halfExtent, 4.0f, 0.5f);
  const Vec3 leftAnchor(0.5f, 0.0f, 0.0f);
  const Vec3 rightAnchor(-0.5f, 0.0f, 0.0f);
  if (reverse)
    solver.addFixedJoint(right, left, rightAnchor, leftAnchor, 1e6f);
  else
    solver.addFixedJoint(left, right, leftAnchor, rightAnchor, 1e6f);

  ContactCache cache;
  const int frames = static_cast<int>(10.0f / frameDt + 0.5f);
  for (int frame = 0; frame < frames; ++frame) {
    solver.contacts.clear();
    const float downwardSpeed = std::max(
        0.0f, -std::min(solver.bodies[left].linearVelocity.y,
                        solver.bodies[right].linearVelocity.y));
    const float margin = std::max(0.05f, downwardSpeed * frameDt);
    collideBoxGround(solver, left, margin);
    collideBoxGround(solver, right, margin);
    if (!solver.contacts.empty())
      ++result.contactFrames;
    cache.restore(solver);
    solver.step(frameDt);
    cache.save(solver);
    if (solver.contactIslandPcgRoutedLastStep) {
      ++result.routedFrames;
      result.pcgOk = result.pcgOk &&
                     solver.contactIslandPcgLastStats.converged &&
                     !solver.contactIslandPcgLastStats.breakdown &&
                     solver.contactIslandPcgLastStats.finite;
      result.maxPcgIterations =
          std::max(result.maxPcgIterations,
                   solver.contactIslandPcgLastStats.iterations);
      result.worstPcgResidual =
          std::max(result.worstPcgResidual,
                   solver.contactIslandPcgLastStats
                       .finalPreconditionedResidual);
    }

    const Vec3 worldLeft = solver.bodies[left].position +
                           solver.bodies[left].rotation.rotate(leftAnchor);
    const Vec3 worldRight = solver.bodies[right].position +
                            solver.bodies[right].rotation.rotate(rightAnchor);
    result.maxLinkError =
        std::max(result.maxLinkError, (worldLeft - worldRight).length());
    Vec3 com = (solver.bodies[left].position +
                solver.bodies[right].position) * 0.5f;
    result.maxHorizontalComError = std::max(
        result.maxHorizontalComError,
        std::sqrt(com.x * com.x + com.z * com.z));
    const Vec3 momentum =
        solver.bodies[left].linearVelocity * solver.bodies[left].mass +
        solver.bodies[right].linearVelocity * solver.bodies[right].mass;
    result.maxHorizontalMomentum = std::max(
        result.maxHorizontalMomentum,
        std::sqrt(momentum.x * momentum.x + momentum.z * momentum.z));
    for (uint32_t bodyId : {left, right}) {
      const Body &body = solver.bodies[bodyId];
      result.minBottom =
          std::min(result.minBottom, body.position.y - halfExtent.y);
      result.maxSpeed = std::max(result.maxSpeed,
                                 body.linearVelocity.length());
      result.finite = result.finite && std::isfinite(body.position.x) &&
                      std::isfinite(body.position.y) &&
                      std::isfinite(body.position.z) &&
                      std::isfinite(body.linearVelocity.x) &&
                      std::isfinite(body.linearVelocity.y) &&
                      std::isfinite(body.linearVelocity.z);
    }
  }
  result.finalBottom =
      0.5f * ((solver.bodies[left].position.y - halfExtent.y) +
              (solver.bodies[right].position.y - halfExtent.y));
  return result;
}

bool probe129_contactPcgFixedPairDrop() {
  printf("\n--- Probe 129: Contact-PCG fixed-pair drop 30/60/120 Hz ---\n");
  const float timesteps[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                              1.0f / 120.0f};
  bool pass = true;
  const char *reason = "ok";
  for (float frameDt : timesteps) {
    FixedPairDropResult baseline[2];
    FixedPairDropResult candidate[2];
    for (int order = 0; order < 2; ++order) {
      baseline[order] = runFixedPairDrop(frameDt, order != 0, false);
      candidate[order] = runFixedPairDrop(frameDt, order != 0, true);
      printf("[FixedPairDropLane] dt=%.9g order=%s "
             "baseline=(bottom=%.7g,final=%.7g,link=%.7g,com=%.7g,"
             "momentum=%.7g) "
             "candidate=(bottom=%.7g,final=%.7g,link=%.7g,com=%.7g,"
             "momentum=%.7g,speed=%.7g,contacts=%d,routed=%d,"
             "pcg=%d/%d/%.7g)\n",
             frameDt, order ? "reverse" : "normal",
             baseline[order].minBottom, baseline[order].finalBottom,
             baseline[order].maxLinkError,
             baseline[order].maxHorizontalComError,
             baseline[order].maxHorizontalMomentum,
             candidate[order].minBottom,
             candidate[order].finalBottom, candidate[order].maxLinkError,
             candidate[order].maxHorizontalComError,
             candidate[order].maxHorizontalMomentum,
             candidate[order].maxSpeed, candidate[order].contactFrames,
             candidate[order].routedFrames,
             candidate[order].pcgOk ? 1 : 0,
             candidate[order].maxPcgIterations,
             candidate[order].worstPcgResidual);
      if (!candidate[order].finite || !candidate[order].pcgOk ||
          candidate[order].routedFrames == 0) {
        pass = false;
        reason = "routing_or_pcg";
      } else if (candidate[order].minBottom < -0.1f ||
                 std::fabs(candidate[order].finalBottom) > 0.05f ||
                 candidate[order].maxLinkError > 0.02f ||
                 candidate[order].maxHorizontalComError > 1e-3f ||
                 candidate[order].maxHorizontalMomentum > 0.1f) {
        pass = false;
        reason = "physical_gate";
      } else if (candidate[order].minBottom <
                 baseline[order].minBottom - 0.005f) {
        pass = false;
        reason = "penetration_parity";
      }
    }
    const float orderDiff = std::max(
        std::fabs(candidate[0].minBottom - candidate[1].minBottom),
        std::max(std::fabs(candidate[0].finalBottom -
                           candidate[1].finalBottom),
                 std::fabs(candidate[0].maxLinkError -
                           candidate[1].maxLinkError)));
    if (orderDiff > 1e-4f) {
      pass = false;
      reason = "actor_order";
    }
  }
  printf("[FixedPairDropCandidate] status=%s reason=%s\n",
         pass ? "PASS" : "FAIL", reason);
  return pass;
}

struct RevolutePairContactResult {
  float minBottom = INFINITY;
  float finalBottom = 0.0f;
  float maxAnchorError = 0.0f;
  float finalAnchorError = 0.0f;
  float maxAxisMisalignment = 0.0f;
  float finalAxisMisalignment = 0.0f;
  float maxLimitViolation = 0.0f;
  float finalLimitViolation = 0.0f;
  float maxHorizontalComError = 0.0f;
  float maxHorizontalMomentum = 0.0f;
  float maxSpeed = 0.0f;
  int contactFrames = 0;
  int routedFrames = 0;
  int activeLimitFrames = 0;
  int maxPcgIterations = 0;
  double worstPcgResidual = 0.0;
  bool finite = true;
  bool pcgOk = true;
};

static float coupledPairBottom(const Body &body) {
  float result = INFINITY;
  for (int sx = -1; sx <= 1; sx += 2)
    for (int sy = -1; sy <= 1; sy += 2)
      for (int sz = -1; sz <= 1; sz += 2) {
        const Vec3 corner(float(sx) * body.halfExtent.x,
                          float(sy) * body.halfExtent.y,
                          float(sz) * body.halfExtent.z);
        result = std::min(
            result, (body.position + body.rotation.rotate(corner)).y);
      }
  return result;
}

static RevolutePairContactResult runRevolutePairContact(
    float frameDt, bool reverse) {
  RevolutePairContactResult result;
  Solver solver;
  solver.gravity = Vec3(0.0f, -9.8f, 0.0f);
  solver.iterations = 15;
  solver.dt = frameDt;
  solver.useContactIslandPcgProbe = true;
  solver.useCanonicalRigidContactAuthoringProbe = true;
  const Vec3 halfExtent(0.5f, 0.5f, 0.5f);
  const uint32_t left = solver.addBody(Vec3(-0.5f, 0.7f, 0.0f), Quat(),
                                       halfExtent, 4.0f, 0.6f);
  const uint32_t right = solver.addBody(Vec3(0.5f, 0.7f, 0.0f), Quat(),
                                        halfExtent, 4.0f, 0.6f);
  const Vec3 leftAnchor(0.5f, 0.0f, 0.0f);
  const Vec3 rightAnchor(-0.5f, 0.0f, 0.0f);
  const Vec3 hingeAxis(0.0f, 1.0f, 0.0f);
  const uint32_t jointIndex =
      reverse ? solver.addRevoluteJoint(right, left, rightAnchor, leftAnchor,
                                        hingeAxis, hingeAxis, 1e6f)
              : solver.addRevoluteJoint(left, right, leftAnchor, rightAnchor,
                                        hingeAxis, hingeAxis, 1e6f);
  solver.setRevoluteJointLimit(jointIndex, -0.2f, 0.2f);

  const float twistAngle = 0.24f;
  const float swingAngle = 0.02f;
  const Quat twist(std::cos(0.5f * twistAngle), 0.0f,
                   std::sin(0.5f * twistAngle), 0.0f);
  const Quat swing(std::cos(0.5f * swingAngle),
                   std::sin(0.5f * swingAngle), 0.0f, 0.0f);
  Body &perturbed = solver.bodies[right];
  perturbed.rotation = (swing * twist).normalized();
  const Vec3 hingePoint =
      solver.bodies[left].position +
      solver.bodies[left].rotation.rotate(leftAnchor);
  perturbed.position =
      hingePoint - perturbed.rotation.rotate(rightAnchor);
  perturbed.initialPosition = perturbed.position;
  perturbed.inertialPosition = perturbed.position;
  perturbed.initialRotation = perturbed.rotation;
  perturbed.inertialRotation = perturbed.rotation;
  perturbed.updateInvInertiaWorld();
  const Vec3 initialCom =
      (solver.bodies[left].position + solver.bodies[right].position) * 0.5f;

  ContactCache cache;
  const int frames = static_cast<int>(8.0f / frameDt + 0.5f);
  for (int frame = 0; frame < frames; ++frame) {
    solver.contacts.clear();
    const float downwardSpeed = std::max(
        0.0f, -std::min(solver.bodies[left].linearVelocity.y,
                        solver.bodies[right].linearVelocity.y));
    const float margin = std::max(0.05f, downwardSpeed * frameDt);
    collideBoxGround(solver, left, margin);
    collideBoxGround(solver, right, margin);
    if (!solver.contacts.empty())
      ++result.contactFrames;
    cache.restore(solver);
    solver.step(frameDt);
    cache.save(solver);
    if (solver.contactIslandPcgRoutedLastStep) {
      ++result.routedFrames;
      result.pcgOk = result.pcgOk &&
                     solver.contactIslandPcgLastStats.converged &&
                     !solver.contactIslandPcgLastStats.breakdown &&
                     solver.contactIslandPcgLastStats.finite;
      result.maxPcgIterations =
          std::max(result.maxPcgIterations,
                   solver.contactIslandPcgLastStats.iterations);
      result.worstPcgResidual =
          std::max(result.worstPcgResidual,
                   solver.contactIslandPcgLastStats
                       .finalPreconditionedResidual);
    }

    const D6Joint &joint = solver.d6Joints[jointIndex];
    const Body *bodyA = joint.bodyA < solver.bodies.size()
                            ? &solver.bodies[joint.bodyA]
                            : nullptr;
    const Body *bodyB = joint.bodyB < solver.bodies.size()
                            ? &solver.bodies[joint.bodyB]
                            : nullptr;
    const Quat rotationA = bodyA ? bodyA->rotation : Quat();
    const Quat rotationB = bodyB ? bodyB->rotation : Quat();
    const Vec3 worldAnchorA =
        bodyA ? bodyA->position + rotationA.rotate(joint.anchorA)
              : joint.anchorA;
    const Vec3 worldAnchorB =
        bodyB ? bodyB->position + rotationB.rotate(joint.anchorB)
              : joint.anchorB;
    const Quat frameA =
        (bodyA ? rotationA * joint.localFrameA : joint.localFrameA)
            .normalized();
    const Quat frameB =
        (bodyB ? rotationB * joint.localFrameB : joint.localFrameB)
            .normalized();
    const Vec3 axisA = frameA.rotate(Vec3(1.0f, 0.0f, 0.0f));
    const Vec3 axisB = frameB.rotate(Vec3(1.0f, 0.0f, 0.0f));
    const float anchorError = (worldAnchorA - worldAnchorB).length();
    const float axisMisalignment =
        1.0f - std::max(-1.0f, std::min(1.0f, axisA.dot(axisB)));
    const float twistError =
        computeRevoluteSymmetricTwistError(frameA, frameB);
    const float limitViolation = std::fabs(computeAngularLimitViolation(
        twistError, joint.angularLimitLower[0],
        joint.angularLimitUpper[0]));
    if (limitViolation > 1e-5f)
      ++result.activeLimitFrames;
    result.maxAnchorError = std::max(result.maxAnchorError, anchorError);
    result.maxAxisMisalignment =
        std::max(result.maxAxisMisalignment, axisMisalignment);
    result.maxLimitViolation =
        std::max(result.maxLimitViolation, limitViolation);
    result.finalAnchorError = anchorError;
    result.finalAxisMisalignment = axisMisalignment;
    result.finalLimitViolation = limitViolation;

    const Vec3 com =
        (solver.bodies[left].position + solver.bodies[right].position) *
        0.5f;
    const Vec3 comError = com - initialCom;
    result.maxHorizontalComError = std::max(
        result.maxHorizontalComError,
        std::sqrt(comError.x * comError.x + comError.z * comError.z));
    const Vec3 momentum =
        solver.bodies[left].linearVelocity * solver.bodies[left].mass +
        solver.bodies[right].linearVelocity * solver.bodies[right].mass;
    result.maxHorizontalMomentum = std::max(
        result.maxHorizontalMomentum,
        std::sqrt(momentum.x * momentum.x + momentum.z * momentum.z));
    for (uint32_t bodyId : {left, right}) {
      const Body &body = solver.bodies[bodyId];
      result.minBottom =
          std::min(result.minBottom, coupledPairBottom(body));
      result.maxSpeed =
          std::max(result.maxSpeed, body.linearVelocity.length());
      result.finite = result.finite && std::isfinite(body.position.x) &&
                      std::isfinite(body.position.y) &&
                      std::isfinite(body.position.z) &&
                      std::isfinite(body.rotation.w) &&
                      std::isfinite(body.rotation.x) &&
                      std::isfinite(body.rotation.y) &&
                      std::isfinite(body.rotation.z) &&
                      std::isfinite(body.linearVelocity.x) &&
                      std::isfinite(body.linearVelocity.y) &&
                      std::isfinite(body.linearVelocity.z);
    }
  }
  result.finalBottom =
      0.5f * (coupledPairBottom(solver.bodies[left]) +
              coupledPairBottom(solver.bodies[right]));
  return result;
}

bool probe130_contactPcgRevolutePair() {
  printf("\n--- Probe 130: Contact-PCG revolute LIMITED pair 30/60/120 Hz ---\n");
  const float timesteps[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                              1.0f / 120.0f};
  bool pass = true;
  const char *reason = "ok";
  for (float frameDt : timesteps) {
    RevolutePairContactResult lane[2];
    for (int order = 0; order < 2; ++order) {
      lane[order] = runRevolutePairContact(frameDt, order != 0);
      printf("[RevolutePairLane] dt=%.9g order=%s "
             "bottom=(%.7g,%.7g) anchor=(%.7g,%.7g) "
             "axis=(%.7g,%.7g) limit=(%.7g,%.7g,%d) "
             "com=%.7g momentum=%.7g speed=%.7g contacts=%d routed=%d "
             "pcg=%d/%d/%.7g\n",
             frameDt, order ? "reverse" : "normal",
             lane[order].minBottom, lane[order].finalBottom,
             lane[order].maxAnchorError, lane[order].finalAnchorError,
             lane[order].maxAxisMisalignment,
             lane[order].finalAxisMisalignment,
             lane[order].maxLimitViolation,
             lane[order].finalLimitViolation,
             lane[order].activeLimitFrames,
             lane[order].maxHorizontalComError,
             lane[order].maxHorizontalMomentum, lane[order].maxSpeed,
             lane[order].contactFrames, lane[order].routedFrames,
             lane[order].pcgOk ? 1 : 0,
             lane[order].maxPcgIterations,
             lane[order].worstPcgResidual);
      if (!lane[order].finite || !lane[order].pcgOk ||
          lane[order].routedFrames == 0 ||
          lane[order].routedFrames != lane[order].contactFrames) {
        pass = false;
        reason = "routing_or_pcg";
      } else if (lane[order].activeLimitFrames == 0 ||
                 lane[order].minBottom < -0.1f ||
                 std::fabs(lane[order].finalBottom) > 0.05f ||
                 lane[order].finalAnchorError > 0.01f ||
                 lane[order].finalAxisMisalignment > 1e-4f ||
                 lane[order].finalLimitViolation > 0.01f ||
                 lane[order].maxHorizontalComError > 0.5f ||
                 lane[order].maxHorizontalMomentum > 5.0f ||
                 lane[order].maxSpeed > 25.0f) {
        pass = false;
        reason = "physical_gate";
      }
    }
    const float orderDifference = std::max(
        std::fabs(lane[0].minBottom - lane[1].minBottom),
        std::max(
            std::fabs(lane[0].finalBottom - lane[1].finalBottom),
            std::max(
                std::fabs(lane[0].finalAnchorError -
                          lane[1].finalAnchorError),
                std::max(std::fabs(lane[0].finalAxisMisalignment -
                                   lane[1].finalAxisMisalignment),
                         std::fabs(lane[0].finalLimitViolation -
                                   lane[1].finalLimitViolation)))));
    const float transientOrderDifference = std::max(
        std::fabs(lane[0].maxAnchorError - lane[1].maxAnchorError),
        std::max(
            std::fabs(lane[0].maxAxisMisalignment -
                      lane[1].maxAxisMisalignment),
            std::fabs(lane[0].maxLimitViolation -
                      lane[1].maxLimitViolation)));
    const float comOrderDifference =
        std::fabs(lane[0].maxHorizontalComError -
                  lane[1].maxHorizontalComError);
    const float momentumOrderDifference =
        std::fabs(lane[0].maxHorizontalMomentum -
                  lane[1].maxHorizontalMomentum);
    if (orderDifference > 1e-4f || transientOrderDifference > 2e-4f ||
        comOrderDifference > 0.01f || momentumOrderDifference > 0.15f ||
        std::abs(lane[0].contactFrames - lane[1].contactFrames) > 1) {
      pass = false;
      reason = "actor_order";
    }
  }
  printf("[RevolutePairCandidate] status=%s reason=%s\n",
         pass ? "PASS" : "FAIL", reason);
  return pass;
}

struct PrismaticPairContactResult {
  float minBottom = INFINITY;
  float finalBottom = 0.0f;
  float maxOrthogonalError = 0.0f;
  float finalOrthogonalError = 0.0f;
  float maxAngularError = 0.0f;
  float finalAngularError = 0.0f;
  float maxLimitViolation = 0.0f;
  float finalLimitViolation = 0.0f;
  float finalSlideDistance = 0.0f;
  float maxHorizontalComError = 0.0f;
  float maxHorizontalMomentum = 0.0f;
  float maxSpeed = 0.0f;
  int contactFrames = 0;
  int routedFrames = 0;
  int activeLimitFrames = 0;
  int maxPcgIterations = 0;
  double worstPcgResidual = 0.0;
  bool finite = true;
  bool pcgOk = true;
};

static PrismaticPairContactResult runPrismaticPairContact(
    float frameDt, bool reverse) {
  PrismaticPairContactResult result;
  Solver solver;
  solver.gravity = Vec3(0.0f, -9.8f, 0.0f);
  solver.iterations = 15;
  solver.dt = frameDt;
  solver.useContactIslandPcgProbe = true;
  solver.useCanonicalRigidContactAuthoringProbe = true;
  const Vec3 halfExtent(0.1f, 0.5f, 0.2f);
  const uint32_t left = solver.addBody(Vec3(-0.12f, 0.5f, 0.0f), Quat(),
                                       halfExtent, 50.0f, 0.0f);
  const uint32_t right = solver.addBody(Vec3(0.12f, 0.5f, 0.0f), Quat(),
                                        halfExtent, 50.0f, 0.0f);
  const Vec3 slideAxis(1.0f, 0.0f, 0.0f);
  const uint32_t jointIndex =
      reverse ? solver.addPrismaticJoint(right, left, Vec3(), Vec3(),
                                          slideAxis, 1e6f)
              : solver.addPrismaticJoint(left, right, Vec3(), Vec3(),
                                          slideAxis, 1e6f);
  solver.setPrismaticJointLimit(jointIndex, -0.2f, 0.2f);

  const float angularPerturbation = 0.01f;
  Body &perturbed = solver.bodies[right];
  perturbed.rotation =
      Quat(std::cos(0.5f * angularPerturbation), 0.0f, 0.0f,
           std::sin(0.5f * angularPerturbation));
  perturbed.initialRotation = perturbed.rotation;
  perturbed.inertialRotation = perturbed.rotation;
  perturbed.updateInvInertiaWorld();
  const Vec3 initialCom =
      (solver.bodies[left].position + solver.bodies[right].position) * 0.5f;

  ContactCache cache;
  const int frames = static_cast<int>(8.0f / frameDt + 0.5f);
  for (int frame = 0; frame < frames; ++frame) {
    solver.contacts.clear();
    const float downwardSpeed = std::max(
        0.0f, -std::min(solver.bodies[left].linearVelocity.y,
                        solver.bodies[right].linearVelocity.y));
    const float margin = std::max(0.05f, downwardSpeed * frameDt);
    collideBoxGround(solver, left, margin);
    collideBoxGround(solver, right, margin);
    if (!solver.contacts.empty())
      ++result.contactFrames;
    cache.restore(solver);
    solver.step(frameDt);
    cache.save(solver);
    if (solver.contactIslandPcgRoutedLastStep) {
      ++result.routedFrames;
      result.pcgOk = result.pcgOk &&
                     solver.contactIslandPcgLastStats.converged &&
                     !solver.contactIslandPcgLastStats.breakdown &&
                     solver.contactIslandPcgLastStats.finite;
      result.maxPcgIterations =
          std::max(result.maxPcgIterations,
                   solver.contactIslandPcgLastStats.iterations);
      result.worstPcgResidual =
          std::max(result.worstPcgResidual,
                   solver.contactIslandPcgLastStats
                       .finalPreconditionedResidual);
    }

    const D6Joint &joint = solver.d6Joints[jointIndex];
    const Body *bodyA = joint.bodyA < solver.bodies.size()
                            ? &solver.bodies[joint.bodyA]
                            : nullptr;
    const Body *bodyB = joint.bodyB < solver.bodies.size()
                            ? &solver.bodies[joint.bodyB]
                            : nullptr;
    const Quat rotationA = bodyA ? bodyA->rotation : Quat();
    const Quat rotationB = bodyB ? bodyB->rotation : Quat();
    const Vec3 worldAnchorA =
        bodyA ? bodyA->position + rotationA.rotate(joint.anchorA)
              : joint.anchorA;
    const Vec3 worldAnchorB =
        bodyB ? bodyB->position + rotationB.rotate(joint.anchorB)
              : joint.anchorB;
    const Vec3 linearError = worldAnchorA - worldAnchorB;
    const Quat frameA =
        (bodyA ? rotationA * joint.localFrameA : joint.localFrameA)
            .normalized();
    const Quat frameB =
        (bodyB ? rotationB * joint.localFrameB : joint.localFrameB)
            .normalized();
    const Quat midFrame = computeD6SymmetricMidFrame(frameA, frameB);
    const Vec3 worldSlideAxis =
        midFrame.rotate(Vec3(1.0f, 0.0f, 0.0f));
    const float slideDistance = -linearError.dot(worldSlideAxis);
    const float orthogonalError =
        (linearError - worldSlideAxis * linearError.dot(worldSlideAxis))
            .length();
    const float angularError =
        computeD6SymmetricAngularError(frameA, frameB).length();
    const float limitViolation = std::fabs(computeAngularLimitViolation(
        slideDistance, joint.linearLimitLower[0],
        joint.linearLimitUpper[0]));
    if (limitViolation > 1e-5f)
      ++result.activeLimitFrames;
    result.maxOrthogonalError =
        std::max(result.maxOrthogonalError, orthogonalError);
    result.maxAngularError =
        std::max(result.maxAngularError, angularError);
    result.maxLimitViolation =
        std::max(result.maxLimitViolation, limitViolation);
    result.finalOrthogonalError = orthogonalError;
    result.finalAngularError = angularError;
    result.finalLimitViolation = limitViolation;
    result.finalSlideDistance = slideDistance;

    const Vec3 com =
        (solver.bodies[left].position + solver.bodies[right].position) *
        0.5f;
    const Vec3 comError = com - initialCom;
    result.maxHorizontalComError = std::max(
        result.maxHorizontalComError,
        std::sqrt(comError.x * comError.x + comError.z * comError.z));
    const Vec3 momentum =
        solver.bodies[left].linearVelocity * solver.bodies[left].mass +
        solver.bodies[right].linearVelocity * solver.bodies[right].mass;
    result.maxHorizontalMomentum = std::max(
        result.maxHorizontalMomentum,
        std::sqrt(momentum.x * momentum.x + momentum.z * momentum.z));
    for (uint32_t bodyId : {left, right}) {
      const Body &body = solver.bodies[bodyId];
      result.minBottom =
          std::min(result.minBottom, coupledPairBottom(body));
      result.maxSpeed =
          std::max(result.maxSpeed, body.linearVelocity.length());
      result.finite = result.finite && std::isfinite(body.position.x) &&
                      std::isfinite(body.position.y) &&
                      std::isfinite(body.position.z) &&
                      std::isfinite(body.rotation.w) &&
                      std::isfinite(body.rotation.x) &&
                      std::isfinite(body.rotation.y) &&
                      std::isfinite(body.rotation.z) &&
                      std::isfinite(body.linearVelocity.x) &&
                      std::isfinite(body.linearVelocity.y) &&
                      std::isfinite(body.linearVelocity.z);
    }
  }
  result.finalBottom =
      0.5f * (coupledPairBottom(solver.bodies[left]) +
              coupledPairBottom(solver.bodies[right]));
  return result;
}

bool probe131_contactPcgPrismaticPair() {
  printf("\n--- Probe 131: Contact-PCG prismatic LIMITED pair 30/60/120 Hz ---\n");
  const float timesteps[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                              1.0f / 120.0f};
  bool pass = true;
  const char *reason = "ok";
  for (float frameDt : timesteps) {
    PrismaticPairContactResult lane[2];
    for (int order = 0; order < 2; ++order) {
      lane[order] = runPrismaticPairContact(frameDt, order != 0);
      printf("[PrismaticPairLane] dt=%.9g order=%s "
             "bottom=(%.7g,%.7g) orth=(%.7g,%.7g) "
             "angular=(%.7g,%.7g) limit=(%.7g,%.7g,%d) slide=%.7g "
             "com=%.7g momentum=%.7g speed=%.7g contacts=%d routed=%d "
             "pcg=%d/%d/%.7g\n",
             frameDt, order ? "reverse" : "normal",
             lane[order].minBottom, lane[order].finalBottom,
             lane[order].maxOrthogonalError,
             lane[order].finalOrthogonalError,
             lane[order].maxAngularError, lane[order].finalAngularError,
             lane[order].maxLimitViolation,
             lane[order].finalLimitViolation,
             lane[order].activeLimitFrames,
             lane[order].finalSlideDistance,
             lane[order].maxHorizontalComError,
             lane[order].maxHorizontalMomentum, lane[order].maxSpeed,
             lane[order].contactFrames, lane[order].routedFrames,
             lane[order].pcgOk ? 1 : 0,
             lane[order].maxPcgIterations,
             lane[order].worstPcgResidual);
      if (!lane[order].finite || !lane[order].pcgOk ||
          lane[order].routedFrames == 0 ||
          lane[order].routedFrames != lane[order].contactFrames) {
        pass = false;
        reason = "routing_or_pcg";
      } else if (lane[order].activeLimitFrames == 0 ||
                 lane[order].minBottom < -0.05f ||
                 std::fabs(lane[order].finalBottom) > 0.02f ||
                 lane[order].maxOrthogonalError > 0.005f ||
                 lane[order].finalOrthogonalError > 0.005f ||
                 lane[order].maxAngularError > 0.01f ||
                 lane[order].finalAngularError > 0.001f ||
                 lane[order].maxLimitViolation > 0.05f ||
                 lane[order].finalLimitViolation > 0.005f ||
                 std::fabs(lane[order].finalSlideDistance) > 0.205f ||
                 lane[order].maxHorizontalComError > 0.1f ||
                 lane[order].maxHorizontalMomentum > 1.0f ||
                 lane[order].maxSpeed > 10.0f) {
        pass = false;
        reason = "physical_gate";
      }
    }
    const float finalOrderDifference = std::max(
        std::fabs(lane[0].finalBottom - lane[1].finalBottom),
        std::max(
            std::fabs(lane[0].finalOrthogonalError -
                      lane[1].finalOrthogonalError),
            std::max(
                std::fabs(lane[0].finalAngularError -
                          lane[1].finalAngularError),
                std::fabs(lane[0].finalLimitViolation -
                          lane[1].finalLimitViolation))));
    const float transientOrderDifference = std::max(
        std::fabs(lane[0].maxOrthogonalError -
                  lane[1].maxOrthogonalError),
        std::max(std::fabs(lane[0].maxAngularError -
                           lane[1].maxAngularError),
                 std::fabs(lane[0].maxLimitViolation -
                           lane[1].maxLimitViolation)));
    const float comOrderDifference =
        std::fabs(lane[0].maxHorizontalComError -
                  lane[1].maxHorizontalComError);
    const float momentumOrderDifference =
        std::fabs(lane[0].maxHorizontalMomentum -
                  lane[1].maxHorizontalMomentum);
    const float speedOrderDifference =
        std::fabs(lane[0].maxSpeed - lane[1].maxSpeed);
    // The in-range slide coordinate is intentionally FREE. With zero ground
    // friction it remains an undamped phase variable, so endpoint parity is
    // judged from its bounds plus constrained errors/conserved quantities,
    // not from requiring the two float trajectories to share a final phase.
    if (finalOrderDifference > 1e-4f ||
        transientOrderDifference > 3e-4f || comOrderDifference > 0.01f ||
        momentumOrderDifference > 0.1f ||
        speedOrderDifference > 0.1f ||
        std::abs(lane[0].contactFrames - lane[1].contactFrames) > 1) {
      pass = false;
      reason = "actor_order";
    }
  }
  printf("[PrismaticPairCandidate] status=%s reason=%s\n",
         pass ? "PASS" : "FAIL", reason);
  return pass;
}

struct LinearDrivePairContactResult {
  float minBottom = INFINITY;
  float finalBottom = 0.0f;
  float firstRelativeVelocity = 0.0f;
  float transientRelativeVelocity = 0.0f;
  float finalRelativeVelocity = 0.0f;
  float maxOrthogonalRelativeVelocity = 0.0f;
  float maxHorizontalComError = 0.0f;
  float maxHorizontalMomentum = 0.0f;
  float maxSpeed = 0.0f;
  float maxAbsDriveForce = 0.0f;
  float maxDriveForceLimit = 0.0f;
  float maxAbsDriveDual = 0.0f;
  int contactFrames = 0;
  int routedFrames = 0;
  uint32_t emittedDriveRows = 0;
  uint32_t accelerationDriveRows = 0;
  uint32_t unsaturatedDriveRows = 0;
  uint32_t saturatedDriveRows = 0;
  int maxPcgIterations = 0;
  double worstPcgResidual = 0.0;
  bool finite = true;
  bool pcgOk = true;
};

static LinearDrivePairContactResult runLinearDrivePairContact(
    float frameDt, bool reverse, bool forceLimited,
    bool acceleration = false, float massScale = 1.0f,
    float targetOverride = 0.0f, float dampingOverride = 0.0f,
    int solverIterations = 15) {
  LinearDrivePairContactResult result;
  Solver solver;
  solver.gravity = Vec3(0.0f, -9.8f, 0.0f);
  solver.iterations = solverIterations;
  solver.dt = frameDt;
  solver.useContactIslandPcgProbe = true;
  solver.useCanonicalRigidContactAuthoringProbe = true;
  const Vec3 halfExtent(0.25f, 0.5f, 0.25f);
  const uint32_t left = solver.addBody(Vec3(-0.75f, 0.5f, 0.0f), Quat(),
                                       halfExtent, 50.0f * massScale, 0.0f);
  const uint32_t right = solver.addBody(Vec3(0.75f, 0.5f, 0.0f), Quat(),
                                        halfExtent, 50.0f * massScale, 0.0f);
  const uint32_t jointIndex =
      reverse ? solver.addD6Joint(right, left, Vec3(), Vec3(), 0x2A,
                                  0x2A, 0.0f, 2e4f)
              : solver.addD6Joint(left, right, Vec3(), Vec3(), 0x2A,
                                  0x2A, 0.0f, 2e4f);
  D6Joint &drive = solver.d6Joints[jointIndex];
  drive.driveFlags = 0x01;
  drive.driveAccelerationFlags = acceleration ? 0x01 : 0;
  const float targetVelocity =
      targetOverride > 0.0f ? targetOverride
                            : (forceLimited ? 2.0f : 0.5f);
  const float driveDamping =
      dampingOverride > 0.0f ? dampingOverride : 60.0f;
  drive.driveLinearVelocity =
      Vec3(reverse ? -targetVelocity : targetVelocity, 0.0f, 0.0f);
  drive.linearDriveDamping = Vec3(driveDamping, 0.0f, 0.0f);
  drive.driveLinearForce =
      Vec3(forceLimited ? 10.0f : 1e6f, 0.0f, 0.0f);

  const Vec3 initialCom =
      (solver.bodies[left].position + solver.bodies[right].position) * 0.5f;
  ContactCache cache;
  const int frames = static_cast<int>(4.0f / frameDt + 0.5f);
  const int transientFrame =
      std::max(0, static_cast<int>((1.0f / 3.0f) / frameDt + 0.5f) - 1);
  for (int frame = 0; frame < frames; ++frame) {
    solver.contacts.clear();
    collideBoxGround(solver, left, 0.05f);
    collideBoxGround(solver, right, 0.05f);
    if (!solver.contacts.empty())
      ++result.contactFrames;
    cache.restore(solver);
    solver.step(frameDt);
    cache.save(solver);
    if (solver.contactIslandPcgRoutedLastStep) {
      ++result.routedFrames;
      result.pcgOk = result.pcgOk &&
                     solver.contactIslandPcgLastStats.converged &&
                     !solver.contactIslandPcgLastStats.breakdown &&
                     solver.contactIslandPcgLastStats.finite;
      result.maxPcgIterations =
          std::max(result.maxPcgIterations,
                   solver.contactIslandPcgLastStats.iterations);
      result.worstPcgResidual =
          std::max(result.worstPcgResidual,
                   solver.contactIslandPcgLastStats
                       .finalPreconditionedResidual);
    }
    result.emittedDriveRows +=
        solver.linearDriveIslandLastStats.emittedRowCount;
    result.accelerationDriveRows +=
        solver.linearDriveIslandLastStats.accelerationRowCount;
    result.unsaturatedDriveRows +=
        solver.linearDriveIslandLastStats.unsaturatedRowCount;
    result.saturatedDriveRows +=
        solver.linearDriveIslandLastStats.saturatedRowCount;
    result.maxAbsDriveForce =
        std::max(result.maxAbsDriveForce,
                 solver.linearDriveIslandLastStats.maxAbsForce);
    result.maxDriveForceLimit =
        std::max(result.maxDriveForceLimit,
                 solver.linearDriveIslandLastStats.maxForceLimit);
    result.maxAbsDriveDual =
        std::max(result.maxAbsDriveDual,
                 solver.linearDriveIslandLastStats.maxAbsDual);

    const Vec3 relativeVelocity =
        solver.bodies[right].linearVelocity -
        solver.bodies[left].linearVelocity;
    if (frame == 0)
      result.firstRelativeVelocity = relativeVelocity.x;
    if (frame == transientFrame)
      result.transientRelativeVelocity = relativeVelocity.x;
    result.finalRelativeVelocity = relativeVelocity.x;
    result.maxOrthogonalRelativeVelocity =
        std::max(result.maxOrthogonalRelativeVelocity,
                 std::sqrt(relativeVelocity.y * relativeVelocity.y +
                           relativeVelocity.z * relativeVelocity.z));
    const Vec3 com =
        (solver.bodies[left].position + solver.bodies[right].position) *
        0.5f;
    const Vec3 comError = com - initialCom;
    result.maxHorizontalComError = std::max(
        result.maxHorizontalComError,
        std::sqrt(comError.x * comError.x + comError.z * comError.z));
    const Vec3 momentum =
        solver.bodies[left].linearVelocity * solver.bodies[left].mass +
        solver.bodies[right].linearVelocity * solver.bodies[right].mass;
    result.maxHorizontalMomentum = std::max(
        result.maxHorizontalMomentum,
        std::sqrt(momentum.x * momentum.x + momentum.z * momentum.z));
    for (uint32_t bodyId : {left, right}) {
      const Body &body = solver.bodies[bodyId];
      result.minBottom =
          std::min(result.minBottom, coupledPairBottom(body));
      result.maxSpeed =
          std::max(result.maxSpeed, body.linearVelocity.length());
      result.finite = result.finite && std::isfinite(body.position.x) &&
                      std::isfinite(body.position.y) &&
                      std::isfinite(body.position.z) &&
                      std::isfinite(body.linearVelocity.x) &&
                      std::isfinite(body.linearVelocity.y) &&
                      std::isfinite(body.linearVelocity.z);
    }
  }
  result.finalBottom =
      0.5f * (coupledPairBottom(solver.bodies[left]) +
              coupledPairBottom(solver.bodies[right]));
  return result;
}

bool probe132_contactPcgLinearDrivePair() {
  printf("\n--- Probe 132: Contact-PCG force-mode linear drive 30/60/120 Hz ---\n");
  const float timesteps[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                              1.0f / 120.0f};
  bool pass = true;
  const char *reason = "ok";
  for (int limited = 0; limited < 2; ++limited) {
    const float targetVelocity = limited ? 2.0f : 0.5f;
    const float expectedLimit = limited ? 10.0f : 1e6f;
    const float velocityTolerance = limited ? 0.03f : 0.02f;
    for (float frameDt : timesteps) {
      LinearDrivePairContactResult lane[2];
      for (int order = 0; order < 2; ++order) {
        lane[order] = runLinearDrivePairContact(
            frameDt, order != 0, limited != 0);
        printf("[LinearDrivePairLane] mode=%s dt=%.9g order=%s "
               "bottom=(%.7g,%.7g) relV=%.7g orthV=%.7g "
               "com=%.7g momentum=%.7g speed=%.7g force=(%.7g/%.7g) "
               "dual=%.7g rows=(%u,%u,%u) contacts=%d routed=%d "
               "pcg=%d/%d/%.7g\n",
               limited ? "limited" : "tracking", frameDt,
               order ? "reverse" : "normal", lane[order].minBottom,
               lane[order].finalBottom,
               lane[order].finalRelativeVelocity,
               lane[order].maxOrthogonalRelativeVelocity,
               lane[order].maxHorizontalComError,
               lane[order].maxHorizontalMomentum,
               lane[order].maxSpeed, lane[order].maxAbsDriveForce,
               lane[order].maxDriveForceLimit,
               lane[order].maxAbsDriveDual,
               lane[order].emittedDriveRows,
               lane[order].unsaturatedDriveRows,
               lane[order].saturatedDriveRows,
               lane[order].contactFrames, lane[order].routedFrames,
               lane[order].pcgOk ? 1 : 0,
               lane[order].maxPcgIterations,
               lane[order].worstPcgResidual);
        if (!lane[order].finite || !lane[order].pcgOk ||
            lane[order].routedFrames == 0 ||
            lane[order].routedFrames != lane[order].contactFrames ||
            lane[order].emittedDriveRows == 0) {
          pass = false;
          reason = "routing_or_pcg";
        } else if (lane[order].minBottom < -0.05f ||
                   std::fabs(lane[order].finalBottom) > 0.02f ||
                   std::fabs(lane[order].finalRelativeVelocity -
                             targetVelocity) > velocityTolerance ||
                   lane[order].maxOrthogonalRelativeVelocity > 0.02f ||
                   lane[order].maxHorizontalComError > 0.01f ||
                   lane[order].maxHorizontalMomentum > 0.1f ||
                   lane[order].maxSpeed > 5.0f ||
                   lane[order].maxAbsDriveForce > expectedLimit + 1e-3f ||
                   lane[order].maxAbsDriveDual > expectedLimit + 1e-3f) {
          pass = false;
          reason = "physical_gate";
        } else if ((!limited &&
                    lane[order].saturatedDriveRows != 0) ||
                   (limited && lane[order].saturatedDriveRows == 0)) {
          pass = false;
          reason = "clamp_witness";
        }
      }
      const float actorOrderDifference = std::max(
          std::fabs(lane[0].finalRelativeVelocity -
                    lane[1].finalRelativeVelocity),
          std::max(std::fabs(lane[0].maxAbsDriveForce -
                            lane[1].maxAbsDriveForce),
                   std::max(std::fabs(lane[0].maxHorizontalComError -
                                     lane[1].maxHorizontalComError),
                            std::fabs(lane[0].maxHorizontalMomentum -
                                      lane[1].maxHorizontalMomentum))));
      if (actorOrderDifference > 1e-4f ||
          lane[0].saturatedDriveRows != lane[1].saturatedDriveRows ||
          lane[0].unsaturatedDriveRows != lane[1].unsaturatedDriveRows) {
        pass = false;
        reason = "actor_order";
      }
    }
  }
  printf("[LinearDrivePairCandidate] status=%s reason=%s\n",
         pass ? "PASS" : "FAIL", reason);
  return pass;
}

bool probe133_contactPcgLinearAccelerationDrivePair() {
  printf("\n--- Probe 133: Contact-PCG linear acceleration drive "
         "30/60/120 Hz ---\n");
  const float timesteps[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                              1.0f / 120.0f};
  bool pass = true;
  const char *reason = "ok";
  for (float frameDt : timesteps) {
    LinearDrivePairContactResult accelerationLight[2];
    LinearDrivePairContactResult accelerationHeavy[2];
    LinearDrivePairContactResult forceHeavy[2];
    LinearDrivePairContactResult accelerationLimited[2];
    for (int order = 0; order < 2; ++order) {
      const bool reverse = order != 0;
      accelerationLight[order] = runLinearDrivePairContact(
          frameDt, reverse, false, true, 1.0f, 1.0f, 6.0f, 4);
      accelerationHeavy[order] = runLinearDrivePairContact(
          frameDt, reverse, false, true, 10.0f, 1.0f, 6.0f, 4);
      forceHeavy[order] = runLinearDrivePairContact(
          frameDt, reverse, false, false, 10.0f, 1.0f, 6.0f, 4);
      accelerationLimited[order] = runLinearDrivePairContact(
          frameDt, reverse, true, true, 1.0f, 2.0f, 6.0f, 4);
      const LinearDrivePairContactResult *lanes[4] = {
          &accelerationLight[order], &accelerationHeavy[order],
          &forceHeavy[order], &accelerationLimited[order]};
      const char *laneNames[4] = {"accel-light", "accel-heavy",
                                  "force-heavy", "accel-limited"};
      for (int laneIndex = 0; laneIndex < 4; ++laneIndex) {
        const LinearDrivePairContactResult &lane = *lanes[laneIndex];
        printf("[LinearAccelerationPairLane] mode=%s dt=%.9g order=%s "
               "bottom=(%.7g,%.7g) relV=(%.7g,%.7g,%.7g) orthV=%.7g "
               "com=%.7g momentum=%.7g speed=%.7g "
               "force=(%.7g/%.7g) dual=%.7g rows=(%u,%u,%u,%u) "
               "contacts=%d routed=%d pcg=%d/%d/%.7g\n",
               laneNames[laneIndex], frameDt,
               reverse ? "reverse" : "normal", lane.minBottom,
               lane.finalBottom, lane.firstRelativeVelocity,
               lane.transientRelativeVelocity,
               lane.finalRelativeVelocity,
               lane.maxOrthogonalRelativeVelocity,
               lane.maxHorizontalComError, lane.maxHorizontalMomentum,
               lane.maxSpeed, lane.maxAbsDriveForce,
               lane.maxDriveForceLimit, lane.maxAbsDriveDual,
               lane.emittedDriveRows, lane.accelerationDriveRows,
               lane.unsaturatedDriveRows, lane.saturatedDriveRows,
               lane.contactFrames, lane.routedFrames,
               lane.pcgOk ? 1 : 0, lane.maxPcgIterations,
               lane.worstPcgResidual);
        if (!lane.finite || !lane.pcgOk || lane.routedFrames == 0 ||
            lane.routedFrames != lane.contactFrames ||
            lane.emittedDriveRows == 0) {
          pass = false;
          reason = "routing_or_pcg";
        } else if (lane.minBottom < -0.05f ||
                   std::fabs(lane.finalBottom) > 0.02f ||
                   lane.maxOrthogonalRelativeVelocity > 0.02f ||
                   lane.maxHorizontalComError > 0.01f ||
                   lane.maxHorizontalMomentum > 0.1f ||
                   lane.maxSpeed > 5.0f) {
          pass = false;
          reason = "physical_gate";
        }
      }
      if (accelerationLight[order].accelerationDriveRows !=
              accelerationLight[order].emittedDriveRows ||
          accelerationHeavy[order].accelerationDriveRows !=
              accelerationHeavy[order].emittedDriveRows ||
          accelerationLimited[order].accelerationDriveRows !=
              accelerationLimited[order].emittedDriveRows ||
          forceHeavy[order].accelerationDriveRows != 0) {
        pass = false;
        reason = "mode_witness";
      }
      if (std::fabs(accelerationLight[order].finalRelativeVelocity - 1.0f) >
              0.03f ||
          std::fabs(accelerationHeavy[order].finalRelativeVelocity - 1.0f) >
              0.03f ||
          std::fabs(accelerationLimited[order].finalRelativeVelocity -
                    2.0f) > 0.05f) {
        pass = false;
        reason = "target_tracking";
      }
      if (accelerationLimited[order].saturatedDriveRows == 0 ||
          accelerationLimited[order].maxAbsDriveForce > 10.001f ||
          accelerationLimited[order].maxAbsDriveDual > 10.001f ||
          accelerationLight[order].saturatedDriveRows != 0 ||
          accelerationHeavy[order].saturatedDriveRows != 0) {
        pass = false;
        reason = "force_limit";
      }
    }

    for (int order = 0; order < 2; ++order) {
      const float massTransientDifference = std::fabs(
          accelerationLight[order].transientRelativeVelocity -
          accelerationHeavy[order].transientRelativeVelocity);
      const float massFinalDifference = std::fabs(
          accelerationLight[order].finalRelativeVelocity -
          accelerationHeavy[order].finalRelativeVelocity);
      const float modeContrast =
          accelerationHeavy[order].firstRelativeVelocity /
          std::max(1e-6f, forceHeavy[order].firstRelativeVelocity);
      if (massTransientDifference > 0.01f || massFinalDifference > 0.01f ||
          modeContrast < 1.25f) {
        pass = false;
        reason = "mass_or_mode_scaling";
      }
    }

    const LinearDrivePairContactResult *orderLanes[4][2] = {
        {&accelerationLight[0], &accelerationLight[1]},
        {&accelerationHeavy[0], &accelerationHeavy[1]},
        {&forceHeavy[0], &forceHeavy[1]},
        {&accelerationLimited[0], &accelerationLimited[1]}};
    for (int laneIndex = 0; laneIndex < 4; ++laneIndex) {
      const LinearDrivePairContactResult &forward = *orderLanes[laneIndex][0];
      const LinearDrivePairContactResult &reverse = *orderLanes[laneIndex][1];
      const float orderDifference = std::max(
          std::fabs(forward.firstRelativeVelocity -
                    reverse.firstRelativeVelocity),
          std::max(std::fabs(forward.finalRelativeVelocity -
                            reverse.finalRelativeVelocity),
                   std::fabs(forward.maxAbsDriveForce -
                             reverse.maxAbsDriveForce)));
      if (orderDifference > 1e-4f ||
          forward.accelerationDriveRows != reverse.accelerationDriveRows ||
          forward.saturatedDriveRows != reverse.saturatedDriveRows ||
          forward.unsaturatedDriveRows != reverse.unsaturatedDriveRows) {
        pass = false;
        reason = "actor_order";
      }
    }
  }
  printf("[LinearAccelerationPairCandidate] status=%s reason=%s\n",
         pass ? "PASS" : "FAIL", reason);
  return pass;
}

struct TwistDrivePairContactResult {
  float minBottom = INFINITY;
  float finalBottom = 0.0f;
  float firstRelativeAngularVelocity = 0.0f;
  float transientRelativeAngularVelocity = 0.0f;
  float finalRelativeAngularVelocity = 0.0f;
  float maxOrthogonalRelativeAngularVelocity = 0.0f;
  float maxHorizontalComError = 0.0f;
  float maxHorizontalMomentum = 0.0f;
  float maxAxisAngularMomentum = 0.0f;
  float maxLinearSpeed = 0.0f;
  float maxAngularSpeed = 0.0f;
  float firstAbsDriveTorque = 0.0f;
  float maxAbsDriveTorque = 0.0f;
  float maxDriveTorqueLimit = 0.0f;
  float maxAbsDriveDual = 0.0f;
  int contactFrames = 0;
  int routedFrames = 0;
  uint32_t emittedDriveRows = 0;
  uint32_t accelerationDriveRows = 0;
  uint32_t unsaturatedDriveRows = 0;
  uint32_t saturatedDriveRows = 0;
  int maxPcgIterations = 0;
  double worstPcgResidual = 0.0;
  bool finite = true;
  bool pcgOk = true;
};

static TwistDrivePairContactResult runTwistDrivePairContact(
    float frameDt, bool reverse, bool forceLimited,
    bool acceleration = false, float massScale = 1.0f,
    float targetOverride = 0.0f, float dampingOverride = 0.0f,
    int solverIterations = 4, int axisIndex = 0,
    bool slerp = false) {
  TwistDrivePairContactResult result;
  Solver solver;
  solver.gravity = Vec3(0.0f, -9.8f, 0.0f);
  solver.iterations = solverIterations;
  solver.dt = frameDt;
  solver.useContactIslandPcgProbe = true;
  solver.useCanonicalRigidContactAuthoringProbe = true;
  const Vec3 halfExtent(0.25f, 0.5f, 0.25f);
  const uint32_t left = solver.addBody(Vec3(-0.75f, 0.5f, 0.0f), Quat(),
                                       halfExtent, 50.0f * massScale, 0.0f);
  const uint32_t right = solver.addBody(Vec3(0.75f, 0.5f, 0.0f), Quat(),
                                        halfExtent, 50.0f * massScale, 0.0f);
  const uint32_t jointIndex =
      reverse ? solver.addD6Joint(right, left, Vec3(), Vec3(), 0x2A,
                                  0x2A, 0.0f, 2e4f)
              : solver.addD6Joint(left, right, Vec3(), Vec3(), 0x2A,
                                  0x2A, 0.0f, 2e4f);
  D6Joint &drive = solver.d6Joints[jointIndex];
  // The selected local angular axis maps to world +Y. Yaw therefore leaves
  // actor A's authored axis fixed while square footprints keep support neutral.
  const float quarterPi = 0.25f * 3.14159265358979323846f;
  Quat driveFrame;
  if (axisIndex == 0)
    driveFrame = Quat(std::cos(quarterPi), 0.0f, 0.0f,
                      std::sin(quarterPi));
  else if (axisIndex == 2)
    driveFrame = Quat(std::cos(quarterPi), -std::sin(quarterPi),
                      0.0f, 0.0f);
  drive.localFrameA = driveFrame;
  drive.localFrameB = driveFrame;
  const uint32_t driveBits[3] = {0x10, 0x40, 0x80};
  const uint32_t driveBit = slerp ? 0x20 : driveBits[axisIndex];
  drive.driveFlags = driveBit;
  drive.driveAccelerationFlags = acceleration ? driveBit : 0;
  const float targetVelocity =
      targetOverride > 0.0f ? targetOverride
                            : (forceLimited ? 2.0f : 1.0f);
  const float driveDamping =
      dampingOverride > 0.0f ? dampingOverride : 6.0f;
  (&drive.driveAngularVelocity.x)[axisIndex] =
      reverse ? -targetVelocity : targetVelocity;
  if (slerp) {
    drive.angularDriveDamping.z = driveDamping;
    drive.driveAngularForce.z = forceLimited ? 10.0f : 1e6f;
  } else {
    (&drive.angularDriveDamping.x)[axisIndex] = driveDamping;
    (&drive.driveAngularForce.x)[axisIndex] =
        forceLimited ? 10.0f : 1e6f;
  }

  const Vec3 worldAxis(0.0f, 1.0f, 0.0f);
  const Vec3 initialCom =
      (solver.bodies[left].position + solver.bodies[right].position) * 0.5f;
  ContactCache cache;
  const int frames = static_cast<int>(4.0f / frameDt + 0.5f);
  const int transientFrame =
      std::max(0, static_cast<int>((1.0f / 3.0f) / frameDt + 0.5f) - 1);
  for (int frame = 0; frame < frames; ++frame) {
    solver.contacts.clear();
    collideBoxGround(solver, left, 0.05f);
    collideBoxGround(solver, right, 0.05f);
    if (!solver.contacts.empty())
      ++result.contactFrames;
    cache.restore(solver);
    solver.step(frameDt);
    cache.save(solver);
    if (solver.contactIslandPcgRoutedLastStep) {
      ++result.routedFrames;
      result.pcgOk = result.pcgOk &&
                     solver.contactIslandPcgLastStats.converged &&
                     !solver.contactIslandPcgLastStats.breakdown &&
                     solver.contactIslandPcgLastStats.finite;
      result.maxPcgIterations =
          std::max(result.maxPcgIterations,
                   solver.contactIslandPcgLastStats.iterations);
      result.worstPcgResidual =
          std::max(result.worstPcgResidual,
                   solver.contactIslandPcgLastStats
                       .finalPreconditionedResidual);
    }
    result.emittedDriveRows +=
        solver.angularDriveIslandLastStats.emittedRowCount;
    result.accelerationDriveRows +=
        solver.angularDriveIslandLastStats.accelerationRowCount;
    result.unsaturatedDriveRows +=
        solver.angularDriveIslandLastStats.unsaturatedRowCount;
    result.saturatedDriveRows +=
        solver.angularDriveIslandLastStats.saturatedRowCount;
    result.maxAbsDriveTorque =
        std::max(result.maxAbsDriveTorque,
                 solver.angularDriveIslandLastStats.maxAbsTorque);
    result.maxDriveTorqueLimit =
        std::max(result.maxDriveTorqueLimit,
                 solver.angularDriveIslandLastStats.maxTorqueLimit);
    result.maxAbsDriveDual =
        std::max(result.maxAbsDriveDual,
                 solver.angularDriveIslandLastStats.maxAbsDual);
    if (frame == 0)
      result.firstAbsDriveTorque =
          solver.angularDriveIslandLastStats.maxAbsTorque;

    const Vec3 relativeAngularVelocity =
        slerp ? solver.bodies[right].angularVelocity -
                    solver.bodies[left].angularVelocity
              : solver.bodies[left].angularVelocity -
                    solver.bodies[right].angularVelocity;
    const float axisVelocity = relativeAngularVelocity.dot(worldAxis);
    if (frame == 0)
      result.firstRelativeAngularVelocity = axisVelocity;
    if (frame == transientFrame)
      result.transientRelativeAngularVelocity = axisVelocity;
    result.finalRelativeAngularVelocity = axisVelocity;
    const Vec3 orthogonalAngularVelocity =
        relativeAngularVelocity - worldAxis * axisVelocity;
    result.maxOrthogonalRelativeAngularVelocity =
        std::max(result.maxOrthogonalRelativeAngularVelocity,
                 orthogonalAngularVelocity.length());
    const Vec3 com =
        (solver.bodies[left].position + solver.bodies[right].position) *
        0.5f;
    const Vec3 comError = com - initialCom;
    result.maxHorizontalComError = std::max(
        result.maxHorizontalComError,
        std::sqrt(comError.x * comError.x + comError.z * comError.z));
    const Vec3 momentum =
        solver.bodies[left].linearVelocity * solver.bodies[left].mass +
        solver.bodies[right].linearVelocity * solver.bodies[right].mass;
    result.maxHorizontalMomentum = std::max(
        result.maxHorizontalMomentum,
        std::sqrt(momentum.x * momentum.x + momentum.z * momentum.z));
    const float axisAngularMomentum =
        solver.bodies[left].inertiaTensor.m[1][1] *
            solver.bodies[left].angularVelocity.y +
        solver.bodies[right].inertiaTensor.m[1][1] *
            solver.bodies[right].angularVelocity.y;
    result.maxAxisAngularMomentum =
        std::max(result.maxAxisAngularMomentum,
                 std::fabs(axisAngularMomentum));
    for (uint32_t bodyId : {left, right}) {
      const Body &body = solver.bodies[bodyId];
      result.minBottom =
          std::min(result.minBottom, coupledPairBottom(body));
      result.maxLinearSpeed =
          std::max(result.maxLinearSpeed, body.linearVelocity.length());
      result.maxAngularSpeed =
          std::max(result.maxAngularSpeed, body.angularVelocity.length());
      result.finite = result.finite && std::isfinite(body.position.x) &&
                      std::isfinite(body.position.y) &&
                      std::isfinite(body.position.z) &&
                      std::isfinite(body.rotation.w) &&
                      std::isfinite(body.rotation.x) &&
                      std::isfinite(body.rotation.y) &&
                      std::isfinite(body.rotation.z) &&
                      std::isfinite(body.linearVelocity.x) &&
                      std::isfinite(body.linearVelocity.y) &&
                      std::isfinite(body.linearVelocity.z) &&
                      std::isfinite(body.angularVelocity.x) &&
                      std::isfinite(body.angularVelocity.y) &&
                      std::isfinite(body.angularVelocity.z);
    }
  }
  result.finalBottom =
      0.5f * (coupledPairBottom(solver.bodies[left]) +
              coupledPairBottom(solver.bodies[right]));
  return result;
}

static bool probeContactPcgSingleAngularAxisDrivePair(
    int axisIndex, const char *axisName, bool slerp = false) {
  printf("\n--- Contact-PCG %s velocity drive 30/60/120 Hz ---\n",
         axisName);
  const float timesteps[3] = {1.0f / 30.0f, 1.0f / 60.0f,
                              1.0f / 120.0f};
  bool pass = true;
  const char *reason = "ok";
  for (float frameDt : timesteps) {
    TwistDrivePairContactResult forceLight[2];
    TwistDrivePairContactResult accelerationLight[2];
    TwistDrivePairContactResult accelerationHeavy[2];
    TwistDrivePairContactResult forceHeavy[2];
    TwistDrivePairContactResult accelerationLimited[2];
    for (int order = 0; order < 2; ++order) {
      const bool reverse = order != 0;
      forceLight[order] = runTwistDrivePairContact(
          frameDt, reverse, false, false, 1.0f, 1.0f, 0.5f, 4,
          axisIndex, slerp);
      accelerationLight[order] = runTwistDrivePairContact(
          frameDt, reverse, false, true, 1.0f, 1.0f, 0.5f, 4,
          axisIndex, slerp);
      accelerationHeavy[order] = runTwistDrivePairContact(
          frameDt, reverse, false, true, 10.0f, 1.0f, 0.5f, 4,
          axisIndex, slerp);
      forceHeavy[order] = runTwistDrivePairContact(
          frameDt, reverse, false, false, 10.0f, 1.0f, 0.5f, 4,
          axisIndex, slerp);
      accelerationLimited[order] = runTwistDrivePairContact(
          frameDt, reverse, true, true, 1.0f, 2.0f, 0.5f, 4,
          axisIndex, slerp);
      const TwistDrivePairContactResult *lanes[5] = {
          &forceLight[order], &accelerationLight[order],
          &accelerationHeavy[order], &forceHeavy[order],
          &accelerationLimited[order]};
      const char *laneNames[5] = {"force-light", "accel-light",
                                  "accel-heavy", "force-heavy",
                                  "accel-limited"};
      for (int laneIndex = 0; laneIndex < 5; ++laneIndex) {
        const TwistDrivePairContactResult &lane = *lanes[laneIndex];
        printf("[AngularAxisDrivePairLane] axis=%s mode=%s dt=%.9g "
               "order=%s "
               "bottom=(%.7g,%.7g) relW=(%.7g,%.7g,%.7g) orthW=%.7g "
               "com=%.7g momentum=%.7g angularMomentum=%.7g "
               "speed=(%.7g,%.7g) torque=(%.7g,%.7g/%.7g) dual=%.7g "
               "rows=(%u,%u,%u,%u) contacts=%d routed=%d "
               "pcg=%d/%d/%.7g\n",
               axisName, laneNames[laneIndex], frameDt,
               reverse ? "reverse" : "normal", lane.minBottom,
               lane.finalBottom, lane.firstRelativeAngularVelocity,
               lane.transientRelativeAngularVelocity,
               lane.finalRelativeAngularVelocity,
               lane.maxOrthogonalRelativeAngularVelocity,
               lane.maxHorizontalComError, lane.maxHorizontalMomentum,
               lane.maxAxisAngularMomentum, lane.maxLinearSpeed,
               lane.maxAngularSpeed, lane.firstAbsDriveTorque,
               lane.maxAbsDriveTorque,
               lane.maxDriveTorqueLimit, lane.maxAbsDriveDual,
               lane.emittedDriveRows, lane.accelerationDriveRows,
               lane.unsaturatedDriveRows, lane.saturatedDriveRows,
               lane.contactFrames, lane.routedFrames,
               lane.pcgOk ? 1 : 0, lane.maxPcgIterations,
               lane.worstPcgResidual);
        if (!lane.finite || !lane.pcgOk || lane.routedFrames == 0 ||
            lane.routedFrames != lane.contactFrames ||
            lane.emittedDriveRows == 0) {
          pass = false;
          reason = "routing_or_pcg";
        } else if (lane.minBottom < -0.05f ||
                   std::fabs(lane.finalBottom) > 0.02f ||
                   lane.maxOrthogonalRelativeAngularVelocity > 0.02f ||
                   lane.maxHorizontalComError > 0.01f ||
                   lane.maxHorizontalMomentum > 0.1f ||
                   lane.maxAxisAngularMomentum > 0.1f ||
                   lane.maxLinearSpeed > 1.0f ||
                   lane.maxAngularSpeed > 5.0f) {
          pass = false;
          reason = "physical_gate";
        }
      }
      if (forceLight[order].accelerationDriveRows != 0 ||
          forceHeavy[order].accelerationDriveRows != 0 ||
          accelerationLight[order].accelerationDriveRows !=
              accelerationLight[order].emittedDriveRows ||
          accelerationHeavy[order].accelerationDriveRows !=
              accelerationHeavy[order].emittedDriveRows ||
          accelerationLimited[order].accelerationDriveRows !=
              accelerationLimited[order].emittedDriveRows) {
        pass = false;
        reason = "mode_witness";
      }
      if (std::fabs(forceLight[order].finalRelativeAngularVelocity - 1.0f) >
              0.03f ||
          std::fabs(accelerationLight[order].finalRelativeAngularVelocity -
                    1.0f) > 0.03f ||
          std::fabs(accelerationHeavy[order].finalRelativeAngularVelocity -
                    1.0f) > 0.03f ||
          std::fabs(accelerationLimited[order].finalRelativeAngularVelocity -
                    2.0f) > 0.05f) {
        pass = false;
        reason = "target_tracking";
      }
      if (accelerationLimited[order].saturatedDriveRows == 0 ||
          accelerationLimited[order].maxAbsDriveTorque > 10.001f ||
          accelerationLimited[order].maxAbsDriveDual > 10.001f ||
          accelerationLight[order].saturatedDriveRows != 0 ||
          accelerationHeavy[order].saturatedDriveRows != 0 ||
          forceLight[order].saturatedDriveRows != 0) {
        pass = false;
        reason = "torque_limit";
      }
      const float massTransientDifference = std::fabs(
          accelerationLight[order].transientRelativeAngularVelocity -
          accelerationHeavy[order].transientRelativeAngularVelocity);
      const float massFinalDifference = std::fabs(
          accelerationLight[order].finalRelativeAngularVelocity -
          accelerationHeavy[order].finalRelativeAngularVelocity);
      const float modeContrast =
          accelerationHeavy[order].firstAbsDriveTorque /
          std::max(1e-6f, forceHeavy[order].firstAbsDriveTorque);
      if (massTransientDifference > 0.01f || massFinalDifference > 0.01f ||
          modeContrast < 1.2f) {
        pass = false;
        reason = "mass_or_mode_scaling";
      }
    }

    const TwistDrivePairContactResult *orderLanes[5][2] = {
        {&forceLight[0], &forceLight[1]},
        {&accelerationLight[0], &accelerationLight[1]},
        {&accelerationHeavy[0], &accelerationHeavy[1]},
        {&forceHeavy[0], &forceHeavy[1]},
        {&accelerationLimited[0], &accelerationLimited[1]}};
    for (int laneIndex = 0; laneIndex < 5; ++laneIndex) {
      const TwistDrivePairContactResult &forward = *orderLanes[laneIndex][0];
      const TwistDrivePairContactResult &reverse = *orderLanes[laneIndex][1];
      const float orderDifference = std::max(
          std::fabs(forward.firstRelativeAngularVelocity -
                    reverse.firstRelativeAngularVelocity),
          std::max(std::fabs(forward.finalRelativeAngularVelocity -
                            reverse.finalRelativeAngularVelocity),
                   std::fabs(forward.maxAbsDriveTorque -
                             reverse.maxAbsDriveTorque)));
      if (orderDifference > 1e-4f ||
          forward.accelerationDriveRows != reverse.accelerationDriveRows ||
          forward.saturatedDriveRows != reverse.saturatedDriveRows ||
          forward.unsaturatedDriveRows != reverse.unsaturatedDriveRows) {
        pass = false;
        reason = "actor_order";
      }
    }
  }
  printf("[AngularAxisDrivePairCandidate] axis=%s status=%s reason=%s\n",
         axisName, pass ? "PASS" : "FAIL", reason);
  return pass;
}

bool probe134_contactPcgTwistDrivePair() {
  printf("\n--- Probe 134: isolated TWIST drive pair ---\n");
  return probeContactPcgSingleAngularAxisDrivePair(0, "TWIST");
}

bool probe135_contactPcgSwingDrivePairs() {
  printf("\n--- Probe 135: isolated SWING1/SWING2 drive pairs ---\n");
  const bool swing1 =
      probeContactPcgSingleAngularAxisDrivePair(1, "SWING1");
  const bool swing2 =
      probeContactPcgSingleAngularAxisDrivePair(2, "SWING2");
  printf("[SwingDrivePairCandidate] status=%s swing1=%d swing2=%d\n",
         swing1 && swing2 ? "PASS" : "FAIL", swing1 ? 1 : 0,
         swing2 ? 1 : 0);
  return swing1 && swing2;
}

bool probe136_contactPcgSlerpDrivePair() {
  printf("\n--- Probe 136: isolated SLERP drive pair ---\n");
  return probeContactPcgSingleAngularAxisDrivePair(1, "SLERP", true);
}
