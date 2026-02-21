#include "avbd_collision.h"
#include "avbd_test_utils.h"
#include <algorithm>
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
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    float specMargin = std::max(
        0.05f, fabsf(solver.bodies[ball].linearVelocity.y) * solver.dt);
    collideAll(solver, specMargin);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    minBallY = std::min(minBallY, solver.bodies[ball].position.y);
  }
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
  for (int frame = 0; frame < 60; frame++) {
    solver.contacts.clear();
    float specMargin = std::max(
        0.05f, fabsf(solver.bodies[ball].linearVelocity.y) * solver.dt);
    collideAll(solver, specMargin);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    minBallY = std::min(minBallY, solver.bodies[ball].position.y);
  }
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
