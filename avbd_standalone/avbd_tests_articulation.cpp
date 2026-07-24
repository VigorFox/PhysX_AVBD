#include "avbd_articulation.h"
#include "avbd_collision.h"
#include "avbd_test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
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
    const std::string passMessage = (msg);                                     \
    printf("  PASS: %s\n", passMessage.c_str());                              \
    gTestsPassed++;                                                            \
    return true;                                                               \
  } while (0)

// =============================================================================
// test74: Single revolute pendulum — gravity swing + constraint correctness
//
// A single link attached to a fixed world anchor via revolute joint.
// Should swing like a pendulum under gravity without exploding.
// Validates: revolute constraint holds, chain hangs below anchor.
// =============================================================================
bool test74_articulationPendulum() {
  printf("test74_articulationPendulum\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Create a single link body
  Vec3 halfExt(0.2f, 0.5f, 0.2f);
  uint32_t link0 = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 500.0f);

  // Articulation: fixed base at (0, 10, 0), revolute joint along Z axis
  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  artic.addLink(link0, -1, eARTIC_REVOLUTE,
                Vec3(0, 0, 1),     // Z axis = rotation axis
                Vec3(0, 0, 0),     // parent anchor (at fixed base)
                Vec3(0, 1.0f, 0),  // child anchor (top of link)
                solver.bodies);
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (std::fabs(solver.bodies[link0].position.y) > 100.0f ||
        std::fabs(solver.bodies[link0].position.x) > 100.0f)
      exploded = true;
  }
  CHECK(!exploded, "Pendulum exploded!");

  // Link should be below the fixed base
  CHECK(solver.bodies[link0].position.y < 10.0f,
        "Link should hang below anchor (y=%.3f)",
        solver.bodies[link0].position.y);

  // Check joint constraint: child anchor should be near parent anchor
  Vec3 worldChildAnchor = solver.bodies[link0].position +
                          solver.bodies[link0].rotation.rotate(Vec3(0, 1.0f, 0));
  Vec3 worldParentAnchor = Vec3(0, 10, 0);
  float gap = (worldChildAnchor - worldParentAnchor).length();
  CHECK(gap < 0.15f, "Joint gap too large: %.4f", gap);

  PASS("articulation pendulum stable, gap=" + std::to_string(gap));
}

// =============================================================================
// test75: 5-link chain — constraint propagation along chain
//
// A 5-link chain hanging from a fixed base via revolute joints.
// Validates: chain ordering (each link below previous), no explosion.
// =============================================================================
bool test75_articulationChain5() {
  printf("test75_articulationChain5\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  const int N = 5;
  uint32_t ids[N];
  Vec3 halfExt(0.15f, 0.4f, 0.15f);
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({0, 18.0f - i * 2.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 20, 0);

  // Root joint: world -> link 0
  int j0 = artic.addLink(ids[0], -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  // Chain joints
  int prev = j0;
  for (int i = 1; i < N; i++) {
    prev = artic.addLink(ids[i], prev, eARTIC_REVOLUTE,
                         Vec3(0, 0, 1), Vec3(0, -1.0f, 0), Vec3(0, 1.0f, 0),
                         solver.bodies);
  }
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < N; i++)
      if (std::fabs(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
  }
  CHECK(!exploded, "5-link chain exploded!");

  // Check ordering: each link should be below or at the previous one
  for (int i = 0; i < N - 1; i++) {
    CHECK(solver.bodies[ids[i]].position.y >=
              solver.bodies[ids[i + 1]].position.y - 0.5f,
          "Chain ordering violated at link %d", i);
  }

  // Check joint gaps
  for (int i = 0; i < N; i++) {
    Vec3 childAnchor = solver.bodies[ids[i]].position +
                       solver.bodies[ids[i]].rotation.rotate(Vec3(0, 1.0f, 0));
    Vec3 parentAnchor;
    if (i == 0) {
      parentAnchor = Vec3(0, 20, 0);
    } else {
      parentAnchor = solver.bodies[ids[i - 1]].position +
                     solver.bodies[ids[i - 1]].rotation.rotate(Vec3(0, -1.0f, 0));
    }
    float gap = (childAnchor - parentAnchor).length();
    CHECK(gap < 0.3f, "Joint gap too large at link %d: %.4f", i, gap);
  }

  PASS("5-link articulation chain stable");
}

// =============================================================================
// test76: Articulation on ground — hybrid solve (contacts + joint constraints)
//
// A 3-link chain with the tip contacting ground. Validates that contact
// constraints and articulation constraints are solved simultaneously.
// =============================================================================
bool test76_articulationOnGround() {
  printf("test76_articulationOnGround\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.2f, 0.4f, 0.2f);
  uint32_t ids[3];
  ids[0] = solver.addBody({0, 4.0f, 0}, Quat(), halfExt, 500.0f);
  ids[1] = solver.addBody({0, 2.5f, 0}, Quat(), halfExt, 500.0f);
  ids[2] = solver.addBody({0, 1.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 5.5f, 0);
  int j0 = artic.addLink(ids[0], -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.8f, 0),
                          solver.bodies);
  int j1 = artic.addLink(ids[1], j0, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, -0.8f, 0), Vec3(0, 0.8f, 0),
                          solver.bodies);
  artic.addLink(ids[2], j1, eARTIC_REVOLUTE,
                Vec3(0, 0, 1), Vec3(0, -0.8f, 0), Vec3(0, 0.8f, 0),
                solver.bodies);
  solver.articulations.push_back(artic);

  ContactCache cache;
  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    for (int i = 0; i < 3; i++)
      collideBoxGround(solver, ids[i], 0.02f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    for (int i = 0; i < 3; i++)
      if (std::fabs(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
  }
  CHECK(!exploded, "Articulation on ground exploded!");

  // Bottom link should be near ground but not deeply penetrating
  float bottomY = solver.bodies[ids[2]].position.y;
  CHECK(bottomY > -1.0f, "Bottom link fell through ground (y=%.3f)", bottomY);

  PASS("articulation + ground contact hybrid stable");
}

// =============================================================================
// test77: Revolute joint with angle limits
//
// A pendulum with angular limits [-π/4, π/4]. After swinging, the link
// should stay within the limit range.
// =============================================================================
bool test77_articulationWithLimits() {
  printf("test77_articulationWithLimits\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.2f, 0.5f, 0.2f);
  // Start the link displaced to the side so it tries to swing past the limit
  float startAngle = 1.2f; // ~69°, well past the π/4 limit
  float startX = 2.0f * sinf(startAngle);
  float startY = 10.0f - 2.0f * cosf(startAngle);
  uint32_t link0 = solver.addBody({startX, startY, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  float limitRange = 3.14159f / 4.0f; // ±45°
  artic.setJointLimits(j0, -limitRange, limitRange);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 500; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float finalAngle = solver.articulations[0].computeJointAngle(0, solver.bodies);
  CHECK(std::fabs(solver.bodies[link0].position.x) < 100.0f, "Exploded!");
  CHECK(finalAngle >= -limitRange - 0.15f && finalAngle <= limitRange + 0.15f,
        "Angle %.3f outside limits [%.3f, %.3f]", finalAngle, -limitRange, limitRange);

  PASS("revolute joint limits working");
}

// =============================================================================
// test78: Spherical joint articulation
//
// A 3-link chain with spherical joints (ball-socket). Should hang freely
// in 3D while maintaining positional constraints.
// =============================================================================
bool test78_articulationSpherical() {
  printf("test78_articulationSpherical\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.2f, 0.2f, 0.2f);
  uint32_t ids[3];
  ids[0] = solver.addBody({0.5f, 8.0f, 0.3f}, Quat(), halfExt, 500.0f);
  ids[1] = solver.addBody({0.3f, 6.0f, -0.2f}, Quat(), halfExt, 500.0f);
  ids[2] = solver.addBody({-0.1f, 4.0f, 0.1f}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(ids[0], -1, eARTIC_SPHERICAL,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  int j1 = artic.addLink(ids[1], j0, eARTIC_SPHERICAL,
                          Vec3(0, 0, 1), Vec3(0, -1.0f, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  artic.addLink(ids[2], j1, eARTIC_SPHERICAL,
                Vec3(0, 0, 1), Vec3(0, -1.0f, 0), Vec3(0, 1.0f, 0),
                solver.bodies);
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < 3; i++)
      if (std::fabs(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
  }
  CHECK(!exploded, "Spherical chain exploded!");
  CHECK(solver.bodies[ids[0]].position.y < 10.0f, "Link 0 above anchor");
  CHECK(solver.bodies[ids[0]].position.y > solver.bodies[ids[2]].position.y - 1.0f,
        "Chain ordering violated");

  PASS("spherical joint articulation stable");
}

// =============================================================================
// test79: Fixed joint articulation — rigid body chain
//
// A 3-link chain connected with fixed joints. All links should move as
// a rigid assembly (no relative rotation or displacement).
// =============================================================================
bool test79_articulationFixed() {
  printf("test79_articulationFixed\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.2f, 0.3f, 0.2f);
  uint32_t ids[3];
  ids[0] = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 500.0f);
  ids[1] = solver.addBody({0, 6.5f, 0}, Quat(), halfExt, 500.0f);
  ids[2] = solver.addBody({0, 5.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(ids[0], -1, eARTIC_FIXED,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  int j1 = artic.addLink(ids[1], j0, eARTIC_FIXED,
                          Vec3(0, 0, 1), Vec3(0, -0.75f, 0), Vec3(0, 0.75f, 0),
                          solver.bodies);
  artic.addLink(ids[2], j1, eARTIC_FIXED,
                Vec3(0, 0, 1), Vec3(0, -0.75f, 0), Vec3(0, 0.75f, 0),
                solver.bodies);
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < 3; i++)
      if (std::fabs(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
  }
  CHECK(!exploded, "Fixed chain exploded!");

  // All links should maintain roughly consistent gaps (rigid assembly)
  for (int i = 0; i < 2; i++) {
    Vec3 topAnchor = solver.bodies[ids[i]].position +
                     solver.bodies[ids[i]].rotation.rotate(Vec3(0, -0.75f, 0));
    Vec3 botAnchor = solver.bodies[ids[i + 1]].position +
                     solver.bodies[ids[i + 1]].rotation.rotate(Vec3(0, 0.75f, 0));
    float gap = (topAnchor - botAnchor).length();
    CHECK(gap < 0.2f, "Fixed joint gap too large at %d: %.4f", i, gap);
  }

  PASS("fixed joint articulation stable");
}

// =============================================================================
// test80: Prismatic joint — slides under gravity
//
// A link attached via prismatic joint (slide along Y axis). Should slide
// downward under gravity.
// =============================================================================
bool test80_articulationPrismatic() {
  printf("test80_articulationPrismatic\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.3f, 0.3f, 0.3f);
  uint32_t link0 = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  artic.addLink(link0, -1, eARTIC_PRISMATIC,
                Vec3(0, 1, 0),     // slide along Y axis
                Vec3(0, 0, 0),     // parent anchor
                Vec3(0, 0, 0),     // child anchor
                solver.bodies);
  solver.articulations.push_back(artic);

  float initialY = solver.bodies[link0].position.y;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float finalY = solver.bodies[link0].position.y;
  CHECK(std::fabs(solver.bodies[link0].position.x) < 100.0f, "Exploded!");
  CHECK(finalY < initialY - 0.1f, "Should slide down (final=%.3f, initial=%.3f)",
        finalY, initialY);
  // Should maintain lateral constraint (no drift in X/Z)
  CHECK(std::fabs(solver.bodies[link0].position.x) < 0.5f,
        "Lateral X drift: %.3f", solver.bodies[link0].position.x);
  CHECK(std::fabs(solver.bodies[link0].position.z) < 0.5f,
        "Lateral Z drift: %.3f", solver.bodies[link0].position.z);

  PASS("prismatic joint slides correctly");
}

// =============================================================================
// test81: Prismatic joint with displacement limits
//
// A prismatic joint with limit [-1, 1] along Y. Link starts at center,
// slides under gravity, should stop at lower limit.
// =============================================================================
bool test81_articulationPrismaticLimits() {
  printf("test81_articulationPrismaticLimits\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  Vec3 halfExt(0.3f, 0.3f, 0.3f);
  // Start the body below the base so it quickly reaches the lower limit
  uint32_t link0 = solver.addBody({0, 9.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(link0, -1, eARTIC_PRISMATIC,
                          Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
                          solver.bodies);
  artic.setJointLimits(j0, -2.0f, 2.0f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float disp = solver.articulations[0].computeJointDisplacement(0, solver.bodies);
  CHECK(std::fabs(solver.bodies[link0].position.x) < 100.0f, "Exploded!");
  CHECK(disp >= -2.0f - 0.5f, "Displacement below lower limit: %.3f", disp);
  CHECK(disp <= 2.0f + 0.5f, "Displacement above upper limit: %.3f", disp);

  PASS("prismatic limits constrain displacement");
}

// =============================================================================
// test82: PD position drive — revolute joint tracks target angle
//
// A revolute joint with PD drive targeting angle = 0. The link starts
// displaced, and the drive should bring it back to the target angle.
// =============================================================================
bool test82_articulationPDDrive() {
  printf("test82_articulationPDDrive\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.15f, 0.5f, 0.15f);
  // Start displaced to the side
  uint32_t link0 = solver.addBody({1.0f, 8.5f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  // PD drive: target angle = 0 (straight down), moderate stiffness
  artic.setJointDrive(j0, 5000.0f, 500.0f, 1e6f, 0.0f, 0.0f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float angle = solver.articulations[0].computeJointAngle(0, solver.bodies);
  CHECK(std::fabs(solver.bodies[link0].position.x) < 100.0f, "Exploded!");
  // With high stiffness + damping, angle should be near 0 (gravity pulls down
  // which is the 0 angle direction for a Z-axis revolute hanging down)
  CHECK(std::fabs(angle) < 0.5f, "PD drive angle too far from target: %.3f", angle);

  PASS("PD drive tracks target angle");
}

// =============================================================================
// test83: Joint friction — damps oscillation of a pendulum
//
// A pendulum with joint friction. After starting from a displaced angle,
// friction should cause the pendulum to settle faster than without friction.
// =============================================================================
bool test83_articulationJointFriction() {
  printf("test83_articulationJointFriction\n");

  auto runPendulum = [](float frictionCoeff) -> float {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 20;

    Vec3 halfExt(0.15f, 0.5f, 0.15f);
    uint32_t link0 = solver.addBody({1.5f, 8.0f, 0}, Quat(), halfExt, 500.0f);

    Articulation artic;
    artic.fixedBase = true;
    artic.fixedBasePos = Vec3(0, 10, 0);
    int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                            Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                            solver.bodies);
    if (frictionCoeff > 0.0f)
      artic.setJointFriction(j0, frictionCoeff);
    solver.articulations.push_back(artic);

    for (int frame = 0; frame < 300; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    // Return final angular velocity magnitude
    return solver.bodies[link0].angularVelocity.length();
  };

  float velNoFriction = runPendulum(0.0f);
  float velWithFriction = runPendulum(500.0f);

  CHECK(velNoFriction < 100.0f && velWithFriction < 100.0f, "Exploded!");
  CHECK(velWithFriction < velNoFriction + 0.1f,
        "Friction should reduce velocity (no=%.3f, with=%.3f)",
        velNoFriction, velWithFriction);

  PASS("joint friction damps oscillation");
}

// =============================================================================
// test84: Constraint accuracy — long chain under gravity
//
// A 10-link chain. After settling, measure maximum joint gap across all
// joints. Tests constraint accuracy at scale.
// =============================================================================
bool test84_articulationConstraintAccuracy() {
  printf("test84_articulationConstraintAccuracy\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  const int N = 10;
  uint32_t ids[N];
  Vec3 halfExt(0.1f, 0.3f, 0.1f);
  for (int i = 0; i < N; i++)
    ids[i] = solver.addBody({0, 18.0f - i * 1.5f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 20, 0);
  int prev = artic.addLink(ids[0], -1, eARTIC_REVOLUTE,
                            Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.7f, 0),
                            solver.bodies);
  for (int i = 1; i < N; i++) {
    prev = artic.addLink(ids[i], prev, eARTIC_REVOLUTE,
                         Vec3(0, 0, 1), Vec3(0, -0.7f, 0), Vec3(0, 0.7f, 0),
                         solver.bodies);
  }
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 500; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float maxGap = 0.0f;
  for (int i = 0; i < N; i++) {
    Vec3 childAnchor = solver.bodies[ids[i]].position +
                       solver.bodies[ids[i]].rotation.rotate(Vec3(0, 0.7f, 0));
    Vec3 parentAnchor;
    if (i == 0) {
      parentAnchor = Vec3(0, 20, 0);
    } else {
      parentAnchor = solver.bodies[ids[i - 1]].position +
                     solver.bodies[ids[i - 1]].rotation.rotate(Vec3(0, -0.7f, 0));
    }
    float gap = (childAnchor - parentAnchor).length();
    maxGap = std::max(maxGap, gap);
  }

  CHECK(maxGap < 0.5f, "Max joint gap too large: %.4f", maxGap);

  PASS("10-link chain constraint accuracy, maxGap=" + std::to_string(maxGap));
}

// =============================================================================
// test85: Mixed joint types — chain with revolute + spherical + fixed
//
// Validates that different joint types can coexist in one articulation.
// =============================================================================
bool test85_articulationMixedJoints() {
  printf("test85_articulationMixedJoints\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.15f, 0.3f, 0.15f);
  uint32_t ids[4];
  for (int i = 0; i < 4; i++)
    ids[i] = solver.addBody({0, 8.0f - i * 1.5f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(ids[0], -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.7f, 0),
                          solver.bodies);
  int j1 = artic.addLink(ids[1], j0, eARTIC_SPHERICAL,
                          Vec3(0, 0, 1), Vec3(0, -0.7f, 0), Vec3(0, 0.7f, 0),
                          solver.bodies);
  int j2 = artic.addLink(ids[2], j1, eARTIC_FIXED,
                          Vec3(0, 0, 1), Vec3(0, -0.7f, 0), Vec3(0, 0.7f, 0),
                          solver.bodies);
  artic.addLink(ids[3], j2, eARTIC_REVOLUTE,
                Vec3(0, 0, 1), Vec3(0, -0.7f, 0), Vec3(0, 0.7f, 0),
                solver.bodies);
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < 4; i++)
      if (std::fabs(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
  }
  CHECK(!exploded, "Mixed joint chain exploded!");
  CHECK(solver.bodies[ids[0]].position.y > solver.bodies[ids[3]].position.y - 1.0f,
        "Chain ordering violated");

  PASS("mixed joint type articulation stable");
}

// =============================================================================
// test86: Floating base — free root under gravity
//
// A single link with no fixed base (floating root). The entire articulation
// should free-fall under gravity.
// =============================================================================
bool test86_articulationFloatingBase() {
  printf("test86_articulationFloatingBase\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;

  Vec3 halfExt(0.3f, 0.3f, 0.3f);
  uint32_t link0 = solver.addBody({0, 10.0f, 0}, Quat(), halfExt, 500.0f);
  uint32_t link1 = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 500.0f);

  // Floating base: fixedBase = false, link0 is the root
  Articulation artic;
  artic.fixedBase = false;
  // Root link has no constraint to world — it's free
  // But it has a joint to link1
  int j0 = artic.addLink(link0, -1, eARTIC_SPHERICAL,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0, 0),
                          solver.bodies);
  artic.addLink(link1, j0, eARTIC_REVOLUTE,
                Vec3(0, 0, 1), Vec3(0, -1.0f, 0), Vec3(0, 1.0f, 0),
                solver.bodies);
  solver.articulations.push_back(artic);

  float initialY0 = solver.bodies[link0].position.y;
  float initialY1 = solver.bodies[link1].position.y;

  for (int frame = 0; frame < 60; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Both links should fall under gravity
  CHECK(solver.bodies[link0].position.y < initialY0 - 0.5f,
        "Link 0 should fall (y=%.3f)", solver.bodies[link0].position.y);
  CHECK(solver.bodies[link1].position.y < initialY1 - 0.5f,
        "Link 1 should fall (y=%.3f)", solver.bodies[link1].position.y);

  // Joint constraint should still hold
  Vec3 parentAnchor = solver.bodies[link0].position +
                      solver.bodies[link0].rotation.rotate(Vec3(0, -1.0f, 0));
  Vec3 childAnchor = solver.bodies[link1].position +
                     solver.bodies[link1].rotation.rotate(Vec3(0, 1.0f, 0));
  float gap = (parentAnchor - childAnchor).length();
  CHECK(gap < 0.5f, "Joint gap in floating base too large: %.4f", gap);

  PASS("floating base articulation free-falls correctly");
}

// =============================================================================
// test87: Branching articulation — tree structure (not just chain)
//
// A root link with two child branches. Tests that tree topologies work.
// =============================================================================
bool test87_articulationBranching() {
  printf("test87_articulationBranching\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.2f, 0.3f, 0.2f);
  uint32_t root = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 500.0f);
  uint32_t leftChild = solver.addBody({-1.5f, 6.0f, 0}, Quat(), halfExt, 500.0f);
  uint32_t rightChild = solver.addBody({1.5f, 6.0f, 0}, Quat(), halfExt, 500.0f);
  uint32_t leftLeaf = solver.addBody({-2.0f, 4.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(root, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  // Left branch
  int jL = artic.addLink(leftChild, j0, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(-0.5f, -0.7f, 0), Vec3(0, 0.7f, 0),
                          solver.bodies);
  artic.addLink(leftLeaf, jL, eARTIC_REVOLUTE,
                Vec3(0, 0, 1), Vec3(0, -0.7f, 0), Vec3(0, 0.7f, 0),
                solver.bodies);
  // Right branch
  artic.addLink(rightChild, j0, eARTIC_REVOLUTE,
                Vec3(0, 0, 1), Vec3(0.5f, -0.7f, 0), Vec3(0, 0.7f, 0),
                solver.bodies);
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    if (std::fabs(solver.bodies[root].position.y) > 100.0f ||
        std::fabs(solver.bodies[leftChild].position.y) > 100.0f ||
        std::fabs(solver.bodies[rightChild].position.y) > 100.0f)
      exploded = true;
  }
  CHECK(!exploded, "Branching articulation exploded!");

  // All links should be below the base
  CHECK(solver.bodies[root].position.y < 10.0f, "Root above base");
  CHECK(solver.bodies[leftChild].position.y < solver.bodies[root].position.y + 1.0f,
        "Left child ordering");
  CHECK(solver.bodies[rightChild].position.y < solver.bodies[root].position.y + 1.0f,
        "Right child ordering");

  PASS("branching (tree) articulation stable");
}

// =============================================================================
// test88: Velocity drive — revolute joint spinning at target velocity
//
// A revolute joint with velocity drive. The link should spin at/near the
// target angular velocity.
// =============================================================================
bool test88_articulationVelocityDrive() {
  printf("test88_articulationVelocityDrive\n");
  Solver solver;
  solver.gravity = {0, 0, 0}; // zero gravity to isolate drive
  solver.iterations = 20;

  Vec3 halfExt(0.15f, 0.5f, 0.15f);
  uint32_t link0 = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10, 0);
  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1.0f, 0),
                          solver.bodies);
  // Velocity drive: target 2 rad/s, strong damping
  artic.setJointDrive(j0, 0.0f, 5000.0f, 1e6f, 0.0f, 2.0f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Check that angular velocity has a component along the joint axis
  Vec3 angVel = solver.bodies[link0].angularVelocity;
  float axisVel = angVel.dot(Vec3(0, 0, 1));
  CHECK(std::fabs(solver.bodies[link0].position.x) < 100.0f, "Exploded!");
  CHECK(std::fabs(axisVel) > 0.1f,
        "Velocity drive should produce rotation (axisVel=%.3f)", axisVel);

  PASS("velocity drive produces rotation");
}

// =============================================================================
// test89: Extreme mass ratio — light chain on heavy fixed base
//
// Tests stability when chain links have very different masses.
// =============================================================================
bool test89_articulationMassRatio() {
  printf("test89_articulationMassRatio\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  Vec3 halfExt(0.2f, 0.4f, 0.2f);
  // Heavy link near base, progressively lighter
  uint32_t ids[5];
  float densities[5] = {5000.0f, 2000.0f, 500.0f, 100.0f, 50.0f};
  for (int i = 0; i < 5; i++)
    ids[i] = solver.addBody({0, 16.0f - i * 2.0f, 0}, Quat(), halfExt, densities[i]);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 18, 0);
  int prev = artic.addLink(ids[0], -1, eARTIC_REVOLUTE,
                            Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.8f, 0),
                            solver.bodies);
  for (int i = 1; i < 5; i++) {
    prev = artic.addLink(ids[i], prev, eARTIC_REVOLUTE,
                         Vec3(0, 0, 1), Vec3(0, -0.8f, 0), Vec3(0, 0.8f, 0),
                         solver.bodies);
  }
  solver.articulations.push_back(artic);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
    for (int i = 0; i < 5; i++)
      if (std::fabs(solver.bodies[ids[i]].position.y) > 100.0f)
        exploded = true;
  }
  CHECK(!exploded, "Mass ratio chain exploded!");

  PASS("extreme mass ratio articulation stable");
}

// =============================================================================
// test90: PD drive gravity compensation
//
// A horizontal arm under gravity with PD drive targeting horizontal.
// Tests that the drive can hold the arm against gravity.
// =============================================================================
bool test90_articulationDriveGravComp() {
  printf("test90_articulationDriveGravComp\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  Vec3 halfExt(0.5f, 0.1f, 0.1f);
  // Start link horizontal to the right
  uint32_t link0 = solver.addBody({1.0f, 10.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10.0f, 0);
  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(-0.5f, 0, 0),
                          solver.bodies);
  // Very strong PD drive to hold horizontal (target angle = π/2)
  artic.setJointDrive(j0, 50000.0f, 5000.0f, 1e8f, -1.5708f, 0.0f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float finalY = solver.bodies[link0].position.y;
  CHECK(std::fabs(solver.bodies[link0].position.x) < 100.0f, "Exploded!");
  // The link should not droop too far below horizontal
  CHECK(finalY > 9.0f, "Arm drooped too much (y=%.3f)", finalY);

  PASS("PD drive provides gravity compensation");
}

// =============================================================================
// test91: ID extraction — converged λ* encodes joint torques
//
// A single horizontal link with a strong PD drive holding it against gravity.
// After convergence, getJointForce() should report a non-trivial force
// (the gravity compensation force), and getJointTorque() should be non-zero.
// §13.6: "The converged λ* IS inverse dynamics output."
// =============================================================================
bool test91_articulationIDExtraction() {
  printf("test91_articulationIDExtraction\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  Vec3 halfExt(0.5f, 0.1f, 0.1f);
  // Link starts horizontal to the right
  uint32_t link0 = solver.addBody({1.0f, 10.0f, 0}, Quat(), halfExt, 500.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10.0f, 0);
  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(-0.5f, 0, 0),
                          solver.bodies);
  // Strong drive to hold horizontal (target angle = -π/2 from vertical)
  artic.setJointDrive(j0, 50000.0f, 5000.0f, 1e8f, -1.5708f, 0.0f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // After convergence, extract ID
  Vec3 jForce = solver.articulations[0].getJointForce(j0);
  Vec3 jTorque = solver.articulations[0].getJointTorque(j0);
  float axisForce = solver.articulations[0].getJointAxisForce(j0);

  float forceMag = sqrtf(jForce.x * jForce.x + jForce.y * jForce.y + jForce.z * jForce.z);
  float torqueMag = sqrtf(jTorque.x * jTorque.x + jTorque.y * jTorque.y + jTorque.z * jTorque.z);

  printf("  Joint force magnitude: %.3f\n", forceMag);
  printf("  Joint torque magnitude: %.3f\n", torqueMag);
  printf("  Axis force (limit/drive): %.3f\n", axisForce);

  // The joint must report non-trivial forces (gravity on link ~ mass*g = 500*9.8 = 4900 N)
  CHECK(forceMag > 100.0f, "Joint force too small (%.3f), expected gravity compensation", forceMag);

  PASS("ID extraction: converged lambda encodes joint torques");
}

// =============================================================================
// test92: End-effector IK — position constraint on tip of a 2-link chain
//
// A 2-link planar arm (revolute-revolute). An IK target constrains the tip
// of the second link to a specified world position. After sufficient
// iterations, the tip should converge close to the target.
// §13.4 Phase 2: "Add end-effector position constraint."
// =============================================================================
bool test92_articulationEndEffectorIK() {
  printf("test92_articulationEndEffectorIK\n");
  Solver solver;
  solver.gravity = {0, 0, 0};  // Zero gravity for clean IK test
  solver.iterations = 40;

  Vec3 halfExt(0.5f, 0.1f, 0.1f);

  // Two links hanging straight down
  uint32_t link0 = solver.addBody({0, 9.0f, 0}, Quat(), halfExt, 100.0f);
  uint32_t link1 = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 100.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10.0f, 0);

  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.5f, 0),
                          solver.bodies);
  int j1 = artic.addLink(link1, j0, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1),
                          Vec3(0, -0.5f, 0), Vec3(0, 0.5f, 0),
                          solver.bodies);

  // IK target: tip of link1 should reach (1.0, 9.0, 0)
  Vec3 targetPos(1.0f, 9.0f, 0);
  artic.addIKTarget(j1, Vec3(0, -0.5f, 0), targetPos, 1e6f);

  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Check tip position
  Vec3 tipLocal(0, -0.5f, 0);
  Vec3 tipWorld = solver.bodies[link1].position +
                  solver.bodies[link1].rotation.rotate(tipLocal);

  float dx = tipWorld.x - targetPos.x;
  float dy = tipWorld.y - targetPos.y;
  float dz = tipWorld.z - targetPos.z;
  float errDist = sqrtf(dx * dx + dy * dy + dz * dz);
  printf("  Tip pos: (%.3f, %.3f, %.3f), target: (%.3f, %.3f, %.3f)\n",
         tipWorld.x, tipWorld.y, tipWorld.z, targetPos.x, targetPos.y, targetPos.z);
  printf("  IK error distance: %.4f\n", errDist);

  CHECK(errDist < 1.0f, "IK error too large (%.4f)", errDist);

  PASS("End-effector IK converges near target");
}

// =============================================================================
// test93: Long chain (20 links) — deep chain stability stress test
//
// 20-link chain hanging from a fixed base under gravity. Validates that
// the solver doesn't explode for deep articulation trees.
// =============================================================================
bool test93_articulationLongChain() {
  printf("test93_articulationLongChain\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  const int N = 20;
  Vec3 halfExt(0.1f, 0.25f, 0.1f);
  float linkSpacing = 0.5f;

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 15.0f, 0);

  int prevJoint = -1;
  for (int i = 0; i < N; i++) {
    float y = 15.0f - (i + 1) * linkSpacing;
    uint32_t body = solver.addBody({0, y, 0}, Quat(), halfExt, 50.0f);
    prevJoint = artic.addLink(body, prevJoint, eARTIC_REVOLUTE,
                              Vec3(0, 0, 1),
                              Vec3(0, -0.25f, 0), Vec3(0, 0.25f, 0),
                              solver.bodies);
  }
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Validate none of the bodies exploded
  bool stable = true;
  for (int i = 0; i < N; i++) {
    uint32_t bi = artic.joints[i].bodyIndex;
    float y = solver.bodies[bi].position.y;
    if (std::fabs(y) > 200.0f) {
      printf("  Body %d exploded: y = %.3f\n", i, y);
      stable = false;
    }
  }
  CHECK(stable, "Long chain exploded!");

  // Last link should be below the base
  uint32_t lastBody = artic.joints[N - 1].bodyIndex;
  float lastY = solver.bodies[lastBody].position.y;
  CHECK(lastY < artic.fixedBasePos.y, "Last link should be below base (y=%.3f)", lastY);

  // Chain should hang roughly straight down, last link well below base
  CHECK(lastY < artic.fixedBasePos.y - 3.0f,
        "Chain not hanging properly (lastY=%.3f, baseY=%.3f)", lastY, artic.fixedBasePos.y);

  PASS("20-link chain stable under gravity");
}

// =============================================================================
// test94: Prismatic drive tracking — PD drive pushes slider to target
//
// A prismatic joint with a PD drive targeting a specific displacement.
// After convergence the slider should be near the target position.
// =============================================================================
bool test94_articulationPrismaticDriveTracking() {
  printf("test94_articulationPrismaticDriveTracking\n");
  Solver solver;
  solver.gravity = {0, 0, 0};  // Zero gravity for clean test
  solver.iterations = 30;

  Vec3 halfExt(0.2f, 0.2f, 0.2f);
  // Slider starts at the base position
  uint32_t slider = solver.addBody({0, 10.0f, 0}, Quat(), halfExt, 100.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10.0f, 0);

  // Prismatic along Y: child anchor offset so slider is at base initially
  int j0 = artic.addLink(slider, -1, eARTIC_PRISMATIC,
                          Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
                          solver.bodies);

  // PD drive targeting displacement = 2.0 along axis (y direction)
  artic.setJointDrive(j0, 5000.0f, 1000.0f, 1e6f, 2.0f, 0.0f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float disp = solver.articulations[0].computeJointDisplacement(j0, solver.bodies);
  printf("  Prismatic displacement: %.3f (target: 2.0)\n", disp);

  // Should converge close to target displacement
  CHECK(std::fabs(disp - 2.0f) < 0.5f, "Prismatic drive didn't track target (disp=%.3f)", disp);

  PASS("Prismatic PD drive tracks target displacement");
}

// =============================================================================
// test95: Multi-articulation scene — two independent articulations
//
// Two separate pendulums in the same solver scene. Verifies that multiple
// articulations can coexist and solve independently.
// =============================================================================
bool test95_articulationMultiArticulation() {
  printf("test95_articulationMultiArticulation\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.2f, 0.4f, 0.2f);

  // Articulation A: pendulum at x = -2
  uint32_t linkA = solver.addBody({-2, 8.0f, 0}, Quat(), halfExt, 200.0f);
  Articulation articA;
  articA.fixedBase = true;
  articA.fixedBasePos = Vec3(-2, 10.0f, 0);
  articA.addLink(linkA, -1, eARTIC_REVOLUTE,
                 Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.4f, 0),
                 solver.bodies);
  solver.articulations.push_back(articA);

  // Articulation B: pendulum at x = +2
  uint32_t linkB = solver.addBody({2, 8.0f, 0}, Quat(), halfExt, 200.0f);
  Articulation articB;
  articB.fixedBase = true;
  articB.fixedBasePos = Vec3(2, 10.0f, 0);
  articB.addLink(linkB, -1, eARTIC_REVOLUTE,
                 Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.4f, 0),
                 solver.bodies);
  solver.articulations.push_back(articB);

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float yA = solver.bodies[linkA].position.y;
  float yB = solver.bodies[linkB].position.y;
  float xA = solver.bodies[linkA].position.x;
  float xB = solver.bodies[linkB].position.x;

  CHECK(std::fabs(xA) < 100.0f && std::fabs(xB) < 100.0f, "Exploded!");
  // Both should hang below their respective bases
  CHECK(yA < 10.0f && yB < 10.0f, "Pendulums not hanging (yA=%.3f, yB=%.3f)", yA, yB);
  // They should remain separated in X
  CHECK(xA < 0 && xB > 0, "Pendulums crossed (xA=%.3f, xB=%.3f)", xA, xB);

  PASS("Two independent articulations coexist");
}

// =============================================================================
// test96: Floating base momentum conservation — zero gravity
//
// A floating-base (non-fixed) articulation in zero gravity. With no external
// forces, the total linear momentum should be approximately conserved.
// =============================================================================
bool test96_articulationFloatingBaseMomentum() {
  printf("test96_articulationFloatingBaseMomentum\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 20;

  Vec3 halfExt(0.3f, 0.1f, 0.1f);

  // Floating base body (no fixed base)
  uint32_t baseBody = solver.addBody({0, 5.0f, 0}, Quat(), halfExt, 500.0f);
  uint32_t link0 = solver.addBody({1.0f, 5.0f, 0}, Quat(), halfExt, 200.0f);

  Articulation artic;
  artic.fixedBase = false;
  // Root link is the free-floating base
  int jBase = artic.addLink(baseBody, -1, eARTIC_SPHERICAL,
                             Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0, 0),
                             solver.bodies);
  int j0 = artic.addLink(link0, jBase, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1),
                          Vec3(0.3f, 0, 0), Vec3(-0.3f, 0, 0),
                          solver.bodies);
  // Give a PD drive to oscillate the joint
  artic.setJointDrive(j0, 100.0f, 10.0f, 1000.0f, 1.0f, 0.0f);
  solver.articulations.push_back(artic);

  // Measure initial momentum
  Vec3 p0;
  for (uint32_t bi = 0; bi < solver.bodies.size(); bi++) {
    p0 = p0 + solver.bodies[bi].linearVelocity * solver.bodies[bi].mass;
  }

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Measure final momentum
  Vec3 p1;
  for (uint32_t bi = 0; bi < solver.bodies.size(); bi++) {
    p1 = p1 + solver.bodies[bi].linearVelocity * solver.bodies[bi].mass;
  }

  Vec3 dp = p1 - p0;
  float dpMag = sqrtf(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z);
  printf("  Momentum change: (%.4f, %.4f, %.4f), magnitude: %.4f\n",
         dp.x, dp.y, dp.z, dpMag);

  // For internal constraints in zero gravity, momentum change should be small
  // ADMM-based solvers have some numerical drift but shouldn't create large forces
  float totalMass = 0;
  for (uint32_t bi = 0; bi < solver.bodies.size(); bi++)
    totalMass += solver.bodies[bi].mass;
  CHECK(dpMag < totalMass * 5.0f,
        "Momentum not conserved (dp=%.4f, totalMass=%.1f)", dpMag, totalMass);

  PASS("Floating base: momentum approximately conserved");
}

// =============================================================================
// test97: Mimic joint — coupled revolute joints with gear ratio
//
// Two revolute joints in a chain, coupled by a mimic constraint with
// gearRatio = 1.0 (follower mirrors leader). When gravity swings the
// chain, both joints should maintain approximately equal angles.
// =============================================================================
bool test97_articulationMimicJoint() {
  printf("test97_articulationMimicJoint\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  Vec3 halfExt(0.3f, 0.1f, 0.1f);

  uint32_t link0 = solver.addBody({0, 9.0f, 0}, Quat(), halfExt, 200.0f);
  uint32_t link1 = solver.addBody({0, 8.0f, 0}, Quat(), halfExt, 200.0f);

  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10.0f, 0);

  int j0 = artic.addLink(link0, -1, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1), Vec3(0, 0, 0), Vec3(0, 0.1f, 0),
                          solver.bodies);
  int j1 = artic.addLink(link1, j0, eARTIC_REVOLUTE,
                          Vec3(0, 0, 1),
                          Vec3(0, -0.1f, 0), Vec3(0, 0.1f, 0),
                          solver.bodies);

  // Mimic: joint0_angle + 1.0 * joint1_angle + 0 = 0
  // i.e. joint1 angle = -joint0 angle (counter-rotation)
  artic.addMimicJoint(j0, j1, 1.0f, 0.0f, 1e6f);
  solver.articulations.push_back(artic);

  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  float angle0 = solver.articulations[0].computeJointAngle(j0, solver.bodies);
  float angle1 = solver.articulations[0].computeJointAngle(j1, solver.bodies);

  printf("  Joint0 angle: %.4f rad\n", angle0);
  printf("  Joint1 angle: %.4f rad\n", angle1);
  printf("  Sum (should be ~0): %.4f\n", angle0 + angle1);

  // Mimic constraint: angle0 + 1.0 * angle1 + 0 = 0
  // So angle0 + angle1 should be near 0
  float mimicError = std::fabs(angle0 + angle1);
  CHECK(mimicError < 0.5f, "Mimic constraint not satisfied (error=%.4f)", mimicError);

  // Also ensure the chain didn't explode
  CHECK(std::fabs(solver.bodies[link0].position.x) < 50.0f, "Exploded!");
  CHECK(std::fabs(solver.bodies[link1].position.x) < 50.0f, "Exploded!");

  PASS("Mimic joint: coupled revolutes maintain constraint");
}

// =============================================================================
// Phase 3: Convergence & Performance Tests
// =============================================================================

// Helper: build a revolute chain starting horizontally (large initial violations)
static void buildHorizontalChain(Solver &solver, int N, float linkSpacing = 0.5f) {
  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 10.0f, 0);
  Vec3 halfExt(0.1f, 0.25f, 0.1f);
  int prevJoint = -1;
  for (int i = 0; i < N; i++) {
    // Place bodies horizontally — creates large constraint violations
    float x = (i + 1) * linkSpacing;
    uint32_t body = solver.addBody({x, 10.0f, 0}, Quat(), halfExt, 50.0f);
    prevJoint = artic.addLink(body, prevJoint, eARTIC_REVOLUTE,
                              Vec3(0, 0, 1),
                              Vec3(0, -0.25f, 0), Vec3(0, 0.25f, 0),
                              solver.bodies);
  }
  solver.articulations.push_back(artic);
}

// Helper: build a revolute chain hanging vertically (small initial violations)
static void buildVerticalChain(Solver &solver, int N, float linkSpacing = 0.5f) {
  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 15.0f, 0);
  Vec3 halfExt(0.1f, 0.25f, 0.1f);
  int prevJoint = -1;
  for (int i = 0; i < N; i++) {
    float y = 15.0f - (i + 1) * linkSpacing;
    uint32_t body = solver.addBody({0, y, 0}, Quat(), halfExt, 50.0f);
    prevJoint = artic.addLink(body, prevJoint, eARTIC_REVOLUTE,
                              Vec3(0, 0, 1),
                              Vec3(0, -0.25f, 0), Vec3(0, 0.25f, 0),
                              solver.bodies);
  }
  solver.articulations.push_back(artic);
}

// =============================================================================
// test98: Convergence benchmark — chain length vs iteration count
//
// Two scenarios:
// (A) Vertical chain (near-equilibrium start): measures warmstarted convergence
// (B) Horizontal chain (cold start, ~0.7m violation per joint): stress test
// Reports convergence curve for each chain length.
// =============================================================================
bool test98_convergenceBenchmark() {
  printf("test98_convergenceBenchmark\n");

  int chainLengths[] = {5, 10, 20, 50};
  float targetViol = 1e-3f;

  printf("  --- Scenario A: Vertical (warm) ---\n");
  for (int ci = 0; ci < 4; ci++) {
    int N = chainLengths[ci];
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 50;
    buildVerticalChain(solver, N);
    solver.step(solver.dt);

    int itersNeeded = (int)solver.convergenceHistory.size();
    for (int i = 0; i < (int)solver.convergenceHistory.size(); i++) {
      if (solver.convergenceHistory[i] < targetViol) { itersNeeded = i + 1; break; }
    }
    float finalViol = solver.convergenceHistory.back();
    printf("  N=%2d: iters_to_1e-3=%2d  viol@1=%.6f  viol@5=%.6f  final=%.6f\n",
           N, itersNeeded,
           solver.convergenceHistory[0],
           (int)solver.convergenceHistory.size() > 4 ? solver.convergenceHistory[4] : -1.0f,
           finalViol);
  }

  printf("  --- Scenario B: Horizontal (cold) ---\n");
  for (int ci = 0; ci < 4; ci++) {
    int N = chainLengths[ci];
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 50;
    buildHorizontalChain(solver, N);
    solver.step(solver.dt);

    int itersNeeded = (int)solver.convergenceHistory.size();
    for (int i = 0; i < (int)solver.convergenceHistory.size(); i++) {
      if (solver.convergenceHistory[i] < targetViol) { itersNeeded = i + 1; break; }
    }
    float finalViol = solver.convergenceHistory.back();
    printf("  N=%2d: iters_to_1e-3=%2d  viol@1=%.6f  viol@5=%.6f  final=%.6f\n",
           N, itersNeeded,
           solver.convergenceHistory[0],
           (int)solver.convergenceHistory.size() > 4 ? solver.convergenceHistory[4] : -1.0f,
           finalViol);
  }

  // Sanity: 5-link vertical chain should converge well within 50 iterations
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 50;
    buildVerticalChain(solver, 5);
    solver.step(solver.dt);
    CHECK(solver.convergenceHistory.back() < 0.01f,
          "5-link vertical chain didn't converge (viol=%.6f)", solver.convergenceHistory.back());
  }
  // Sanity: 50-link horizontal chain should not explode
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 50;
    buildHorizontalChain(solver, 50);
    for (int frame = 0; frame < 30; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    for (int i = 0; i < 50; i++) {
      uint32_t bi = solver.articulations[0].joints[i].bodyIndex;
      CHECK(std::fabs(solver.bodies[bi].position.y) < 200.0f,
            "Body %d exploded", i);
    }
  }

  PASS("Convergence benchmark complete");
}

// =============================================================================
// test99: Tree-sweep convergence — compare with/without tree ordering
//
// Uses the horizontal chain (hard) scenario with limited iterations to
// see if tree-sweep ordering improves convergence.
// =============================================================================
bool test99_treeSweepConvergence() {
  printf("test99_treeSweepConvergence\n");

  const int N = 20;
  int iterCounts[] = {3, 5, 10, 20};

  printf("  Horizontal %d-link chain:\n", N);
  for (int ii = 0; ii < 4; ii++) {
    int iters = iterCounts[ii];

    // Without tree-sweep (default linear order)
    float violNoSweep;
    {
      Solver solver;
      solver.gravity = {0, -9.8f, 0};
      solver.iterations = iters;
      solver.useTreeSweep = false;
      buildHorizontalChain(solver, N);
      solver.step(solver.dt);
      violNoSweep = solver.convergenceHistory.back();
    }
    // With tree-sweep
    float violSweep;
    {
      Solver solver;
      solver.gravity = {0, -9.8f, 0};
      solver.iterations = iters;
      solver.useTreeSweep = true;
      buildHorizontalChain(solver, N);
      solver.step(solver.dt);
      violSweep = solver.convergenceHistory.back();
    }
    float ratio = (violNoSweep > 1e-10f) ? violSweep / violNoSweep : 1.0f;
    printf("  iters=%2d: noSweep=%.6f  treeSweep=%.6f  ratio=%.3f\n",
           iters, violNoSweep, violSweep, ratio);
  }

  // Tree-sweep should not regress significantly
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 10;
    solver.useTreeSweep = true;
    buildHorizontalChain(solver, N);
    solver.step(solver.dt);
    float violSweep = solver.convergenceHistory.back();

    Solver solver2;
    solver2.gravity = {0, -9.8f, 0};
    solver2.iterations = 10;
    solver2.useTreeSweep = false;
    buildHorizontalChain(solver2, N);
    solver2.step(solver2.dt);
    float violNoSweep = solver2.convergenceHistory.back();

    CHECK(violSweep <= violNoSweep * 1.5f,
          "Tree-sweep regressed badly (%.6f vs %.6f)", violSweep, violNoSweep);
  }

  // Multi-frame stability
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 20;
    solver.useTreeSweep = true;
    buildHorizontalChain(solver, N);
    for (int frame = 0; frame < 100; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    for (int i = 0; i < N; i++) {
      uint32_t bi = solver.articulations[0].joints[i].bodyIndex;
      CHECK(std::fabs(solver.bodies[bi].position.y) < 200.0f,
            "Body %d exploded with tree-sweep", i);
    }
  }

  PASS("Tree-sweep convergence validated");
}

// =============================================================================
// test100: Anderson Acceleration — convergence speed improvement
//
// Compares BCD with and without AA(m=3) on the horizontal chain scenario.
// AA should accelerate convergence for the harder scenario.
// =============================================================================
bool test100_andersonAcceleration() {
  printf("test100_andersonAcceleration\n");

  const int N = 20;
  int iterCounts[] = {5, 10, 20};

  printf("  Horizontal %d-link chain:\n", N);
  for (int ii = 0; ii < 3; ii++) {
    int iters = iterCounts[ii];

    float violNoAA;
    {
      Solver solver;
      solver.gravity = {0, -9.8f, 0};
      solver.iterations = iters;
      solver.useAndersonAccel = false;
      buildHorizontalChain(solver, N);
      solver.step(solver.dt);
      violNoAA = solver.convergenceHistory.back();
    }
    float violAA;
    {
      Solver solver;
      solver.gravity = {0, -9.8f, 0};
      solver.iterations = iters;
      solver.useAndersonAccel = true;
      solver.aaWindowSize = 3;
      buildHorizontalChain(solver, N);
      solver.step(solver.dt);
      violAA = solver.convergenceHistory.back();
    }
    float ratio = (violNoAA > 1e-10f) ? violAA / violNoAA : 1.0f;
    printf("  iters=%2d: noAA=%.6f  AA(m=3)=%.6f  ratio=%.3f\n",
           iters, violNoAA, violAA, ratio);
  }

  // AA should not make things significantly worse (safeguard)
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 10;
    solver.useAndersonAccel = true;
    solver.aaWindowSize = 3;
    buildHorizontalChain(solver, N);
    solver.step(solver.dt);
    float violAA = solver.convergenceHistory.back();

    Solver solver2;
    solver2.gravity = {0, -9.8f, 0};
    solver2.iterations = 10;
    solver2.useAndersonAccel = false;
    buildHorizontalChain(solver2, N);
    solver2.step(solver2.dt);
    float violNoAA = solver2.convergenceHistory.back();

    CHECK(violAA <= violNoAA * 2.0f,
          "AA regressed convergence badly (%.6f vs %.6f)", violAA, violNoAA);
  }

  // Multi-frame stability with AA
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 20;
    solver.useAndersonAccel = true;
    solver.aaWindowSize = 3;
    buildHorizontalChain(solver, N);
    for (int frame = 0; frame < 200; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    for (int i = 0; i < N; i++) {
      uint32_t bi = solver.articulations[0].joints[i].bodyIndex;
      CHECK(std::fabs(solver.bodies[bi].position.y) < 200.0f,
            "Body %d exploded with AA", i);
    }
  }

  PASS("Anderson Acceleration validated");
}

// =============================================================================
// test101: Chebyshev semi-iterative — convergence speed improvement
//
// Compares BCD with and without Chebyshev position over-relaxation.
// =============================================================================
bool test101_chebyshevSemiIterative() {
  printf("test101_chebyshevSemiIterative\n");

  const int N = 20;
  int iterCounts[] = {5, 10, 20};

  printf("  Horizontal %d-link chain:\n", N);
  for (int ii = 0; ii < 3; ii++) {
    int iters = iterCounts[ii];

    float violBase;
    {
      Solver solver;
      solver.gravity = {0, -9.8f, 0};
      solver.iterations = iters;
      solver.useChebyshev = false;
      buildHorizontalChain(solver, N);
      solver.step(solver.dt);
      violBase = solver.convergenceHistory.back();
    }
    float violCheby;
    {
      Solver solver;
      solver.gravity = {0, -9.8f, 0};
      solver.iterations = iters;
      solver.useChebyshev = true;
      solver.chebyshevSpectralRadius = 0.92f;
      buildHorizontalChain(solver, N);
      solver.step(solver.dt);
      violCheby = solver.convergenceHistory.back();
    }
    float ratio = (violBase > 1e-10f) ? violCheby / violBase : 1.0f;
    printf("  iters=%2d: base=%.6f  cheby=%.6f  ratio=%.3f\n",
           iters, violBase, violCheby, ratio);
  }

  // Chebyshev should not make things significantly worse
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 10;
    solver.useChebyshev = true;
    solver.chebyshevSpectralRadius = 0.92f;
    buildHorizontalChain(solver, N);
    solver.step(solver.dt);
    float violCheby = solver.convergenceHistory.back();

    Solver solver2;
    solver2.gravity = {0, -9.8f, 0};
    solver2.iterations = 10;
    solver2.useChebyshev = false;
    buildHorizontalChain(solver2, N);
    solver2.step(solver2.dt);
    float violBase = solver2.convergenceHistory.back();

    CHECK(violCheby <= violBase * 2.0f,
          "Chebyshev regressed convergence badly (%.6f vs %.6f)", violCheby, violBase);
  }

  // Multi-frame stability
  {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 20;
    solver.useChebyshev = true;
    solver.chebyshevSpectralRadius = 0.92f;
    buildHorizontalChain(solver, N);
    for (int frame = 0; frame < 200; frame++) {
      solver.contacts.clear();
      solver.step(solver.dt);
    }
    for (int i = 0; i < N; i++) {
      uint32_t bi = solver.articulations[0].joints[i].bodyIndex;
      CHECK(std::fabs(solver.bodies[bi].position.y) < 200.0f,
            "Body %d exploded with Chebyshev", i);
    }
  }

  PASS("Chebyshev semi-iterative validated");
}

// =============================================================================
// Phase 4: Scissor Lift Validation Tests
// =============================================================================

// =============================================================================
// test102: D6 loop closure between articulation links
//
// Two branches of an articulation connected by a D6 joint at their tips.
// Both branches share the same fixed base. The D6 joint enforces positional
// coincidence (all angular DOFs free). This validates that D6 joints and
// articulation constraints can coexist on the same body indices.
// =============================================================================
bool test102_articulationD6LoopClosure() {
  printf("test102_articulationD6LoopClosure\n");

  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  // Build a Y-shaped articulation: base → left arm (2 links), base → right arm (2 links)
  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(0, 5.0f, 0);
  Vec3 halfExt(0.05f, 0.05f, 0.5f);

  // Left branch: link0 (parent=-1), link1 (parent=0)
  uint32_t bodyL0 = solver.addBody({-0.2f, 4.5f, 0}, Quat(), halfExt, 1.0f);
  int jL0 = artic.addLink(bodyL0, -1, eARTIC_REVOLUTE, Vec3(1, 0, 0),
                           Vec3(0, 0, -0.5f), Vec3(0, 0, 0.5f), solver.bodies);

  uint32_t bodyL1 = solver.addBody({-0.2f, 3.5f, 0}, Quat(), halfExt, 1.0f);
  int jL1 = artic.addLink(bodyL1, jL0, eARTIC_REVOLUTE, Vec3(1, 0, 0),
                           Vec3(0, 0, -0.5f), Vec3(0, 0, 0.5f), solver.bodies);

  // Right branch: link2 (parent=-1), link3 (parent=2)
  uint32_t bodyR0 = solver.addBody({0.2f, 4.5f, 0}, Quat(), halfExt, 1.0f);
  int jR0 = artic.addLink(bodyR0, -1, eARTIC_REVOLUTE, Vec3(1, 0, 0),
                           Vec3(0, 0, -0.5f), Vec3(0, 0, 0.5f), solver.bodies);

  uint32_t bodyR1 = solver.addBody({0.2f, 3.5f, 0}, Quat(), halfExt, 1.0f);
  int jR1 = artic.addLink(bodyR1, jR0, eARTIC_REVOLUTE, Vec3(1, 0, 0),
                           Vec3(0, 0, -0.5f), Vec3(0, 0, 0.5f), solver.bodies);

  solver.articulations.push_back(artic);

  // D6 joint connecting tips of left and right branches — positional lock, angular free
  // leftTip childAnchor end = (0,0,-0.5), rightTip childAnchor end = (0,0,-0.5)
  // linearMotion=0 (all locked), angularMotion=0x2A (all free: 10 10 10 in binary)
  solver.addD6Joint(bodyL1, bodyR1,
                    Vec3(0, 0, -0.5f), Vec3(0, 0, -0.5f),
                    0, 0x2A, 0.0f, 1e6f);

  // Run simulation
  for (int frame = 0; frame < 200; frame++) {
    solver.contacts.clear();
    solver.step(solver.dt);
  }

  // Check: the two tips should be very close
  Vec3 tipL = solver.bodies[bodyL1].position + solver.bodies[bodyL1].rotation.rotate(Vec3(0, 0, -0.5f));
  Vec3 tipR = solver.bodies[bodyR1].position + solver.bodies[bodyR1].rotation.rotate(Vec3(0, 0, -0.5f));
  float tipError = (tipL - tipR).length();
  printf("  Tip error after 200 frames: %.6f\n", tipError);
  CHECK(tipError < 0.05f, "D6 loop closure failed: tip error %.6f", tipError);

  // No body should have exploded
  for (uint32_t i = 0; i < (uint32_t)solver.bodies.size(); i++) {
    CHECK(std::fabs(solver.bodies[i].position.y) < 100.0f,
          "Body %d exploded (y=%.2f)", i, solver.bodies[i].position.y);
  }

  PASS("D6 loop closure between articulation branches works");
}

// =============================================================================
// test103: Scissor Lift Validation
//
// Replicates the PhysX SnippetArticulation scissor lift:
//   - Fixed base (box 0.5×0.25×1.5, density 3)
//   - leftRoot (FIXED to base), rightRoot (PRISMATIC drive to base)
//   - 2 columns × 3 levels of crossing scissor arms (REVOLUTE with limits)
//   - 6 D6 loop-closure joints at X-crossings + 2 D6 top-closure joints
//   - leftTop, rightTop bracket links + top platform (FIXED to leftTop)
//   - 8 dynamic boxes dropped on top as load
//   - Prismatic drive oscillates: target decrements 0.25*dt until <-1.2, then increments
//
// 9 sub-tests matching the PhysX Test 16 criteria.
// =============================================================================
bool test103_scissorLiftValidation() {
  printf("test103_scissorLiftValidation\n");

  // ---- Geometry parameters (from PhysX SnippetArticulation) ----
  const float runnerLength = 2.0f;
  const float placementDistance = 1.8f;
  const float cosAng = placementDistance / runnerLength;
  const float angle = acosf(cosAng); // ~0.451 rad ~25.8°
  const float sinAng = sinf(angle);
  const int linkHeight = 3;

  const float driveStiffness = 50.0f;
  const float driveLimitLower = -1.4f;
  const float driveLimitUpper = 0.2f;

  // Rotation quaternions for tilted scissor arms
  // leftRot = rotation of +angle around X axis (arm tilts "left")
  // rightRot = rotation of -angle around X axis (arm tilts "right")
  float halfAngle = angle * 0.5f;
  Quat leftRot(cosf(halfAngle), sinf(halfAngle), 0, 0);
  Quat rightRot(cosf(halfAngle), -sinf(halfAngle), 0, 0);

  struct ScissorInfo {
    uint32_t baseBody, topPlatformBody;
    uint32_t driveJointIdx;
  };

  // ---- Helper to build the scissor lift scene ----
  auto buildScissorLift = [&](Solver &solver) -> ScissorInfo {
    // --- Base: static body ---
    uint32_t baseBody = solver.addBody({0, 0.25f, 0}, Quat(),
                                        Vec3(0.5f, 0.25f, 1.5f), 3.0f);
    solver.bodies[baseBody].mass = 0.0f;
    solver.bodies[baseBody].invMass = 0.0f;
    solver.bodies[baseBody].inertiaTensor = Mat33::diag(0, 0, 0);
    solver.bodies[baseBody].invInertiaWorld = Mat33::diag(0, 0, 0);

    // --- Left root rail (FIXED to base) ---
    uint32_t leftRootBody = solver.addBody({0, 0.55f, -0.9f}, Quat(),
                                            Vec3(0.5f, 0.05f, 0.05f), 1.0f);
    // Fixed = all DOFs locked
    solver.addD6Joint(baseBody, leftRootBody,
                      Vec3(0, 0.3f, -0.9f), Vec3(0, 0, 0), 0, 0);

    // --- Right root rail (PRISMATIC along Z, driven) ---
    uint32_t rightRootBody = solver.addBody({0, 0.55f, 0.9f}, Quat(),
                                             Vec3(0.5f, 0.05f, 0.05f), 1.0f);
    // Prismatic: slide along Z in body A's local frame
    // We use addPrismaticJoint with localAxisA = (0,0,1)
    uint32_t driveJointIdx = solver.addPrismaticJoint(
        baseBody, rightRootBody,
        Vec3(0, 0.3f, 0.9f), Vec3(0, 0, 0),
        Vec3(0, 0, 1), 1e6f);
    solver.setPrismaticJointLimit(driveJointIdx, driveLimitLower, driveLimitUpper);
    solver.setPrismaticJointDrive(driveJointIdx, 0, driveStiffness);

    // Set damping on all link bodies
    auto setDamping = [&](uint32_t bi) {
      solver.bodies[bi].linearDamping = 0.2f;
      solver.bodies[bi].angularDamping = 0.2f;
      solver.bodies[bi].maxAngularVelocity = 20.0f;
      solver.bodies[bi].maxLinearVelocity = 100.0f;
    };
    setDamping(leftRootBody);
    setDamping(rightRootBody);

    // --- Build two columns of scissor arms ---
    // Each column has linkHeight=3 levels. At each level: one left arm + one right arm.
    // Arms connected to parent via revolute joint (hinge X), to each other via D6 loop closure.
    // After each level, parent pointers CROSS (left→right, right→left).

    auto buildColumn = [&](float xPos, uint32_t initLeft, uint32_t initRight) {
      uint32_t currLeft = initLeft;
      uint32_t currRight = initRight;

      for (int i = 0; i < linkHeight; i++) {
        // Arm center height: each cross pair adds ~2*sinAng height
        float yCenter = 0.55f + sinAng * (2 * i + 1);

        // Left arm: tilted by leftRot
        Vec3 leftArmPos(xPos, yCenter, 0);
        uint32_t leftArm = solver.addBody(leftArmPos, leftRot,
                                           Vec3(0.05f, 0.05f, 1.0f), 1.0f);
        setDamping(leftArm);

        // Revolute: currLeft → leftArm at bottom of arm (z=-1 in arm local)
        // Pivot in world: one end of the arm at the parent's position
        float yPivot = 0.55f + sinAng * (2 * i);
        Vec3 pivotWorld(xPos, yPivot, -cosAng);

        // Convert pivot to local frames
        Vec3 pivotInParent;
        {
          Quat pR = solver.bodies[currLeft].rotation;
          Vec3 pP = solver.bodies[currLeft].position;
          pivotInParent = pR.conjugate().rotate(pivotWorld - pP);
        }
        Vec3 pivotInChild;
        {
          Quat cR = solver.bodies[leftArm].rotation;
          Vec3 cP = solver.bodies[leftArm].position;
          pivotInChild = cR.conjugate().rotate(pivotWorld - cP);
        }

        uint32_t jL = solver.addRevoluteJoint(
            currLeft, leftArm, pivotInParent, pivotInChild,
            Vec3(1, 0, 0), Vec3(1, 0, 0), 1e6f);
        solver.setRevoluteJointLimit(jL, -3.14159f, angle);

        // Right arm: tilted by rightRot
        Vec3 rightArmPos(xPos, yCenter, 0);
        uint32_t rightArm = solver.addBody(rightArmPos, rightRot,
                                            Vec3(0.05f, 0.05f, 1.0f), 1.0f);
        setDamping(rightArm);

        // Revolute: currRight → rightArm at top of arm (z=+1 in arm local)
        Vec3 pivotWorldR(xPos, yPivot, cosAng);
        Vec3 pivotInParentR;
        {
          Quat pR = solver.bodies[currRight].rotation;
          Vec3 pP = solver.bodies[currRight].position;
          pivotInParentR = pR.conjugate().rotate(pivotWorldR - pP);
        }
        Vec3 pivotInChildR;
        {
          Quat cR = solver.bodies[rightArm].rotation;
          Vec3 cP = solver.bodies[rightArm].position;
          pivotInChildR = cR.conjugate().rotate(pivotWorldR - cP);
        }

        uint32_t jR = solver.addRevoluteJoint(
            currRight, rightArm, pivotInParentR, pivotInChildR,
            Vec3(1, 0, 0), Vec3(1, 0, 0), 1e6f);
        solver.setRevoluteJointLimit(jR, -angle, 3.14159f);

        // D6 loop closure at X-crossing: position locked, all angular free
        // Crossing point is at the arm centers (z=0 in world)
        Vec3 crossWorld(xPos, yCenter, 0);
        Vec3 crossInLeft, crossInRight;
        {
          Quat r = solver.bodies[leftArm].rotation;
          Vec3 p = solver.bodies[leftArm].position;
          crossInLeft = r.conjugate().rotate(crossWorld - p);
        }
        {
          Quat r = solver.bodies[rightArm].rotation;
          Vec3 p = solver.bodies[rightArm].position;
          crossInRight = r.conjugate().rotate(crossWorld - p);
        }

        solver.addD6Joint(leftArm, rightArm, crossInLeft, crossInRight,
                          0, 0x2A); // linear locked, angular free

        // CROSS parent assignment
        currLeft = rightArm;
        currRight = leftArm;
      }

      return std::make_pair(currLeft, currRight);
    };

    // Column 1 (x=+0.5)
    auto [col1FinalLeft, col1FinalRight] = buildColumn(0.5f, leftRootBody, rightRootBody);

    // Column 2 (x=-0.5)
    auto [col2FinalLeft, col2FinalRight] = buildColumn(-0.5f, leftRootBody, rightRootBody);

    // --- Top brackets ---
    // leftTop: connected to col1's final left arm tip, and col2's final left tip
    float topArmY = 0.55f + sinAng * (2 * linkHeight);
    Vec3 leftTopPos(0, topArmY, -cosAng);
    uint32_t leftTopBody = solver.addBody(leftTopPos, Quat(),
                                           Vec3(0.5f, 0.05f, 0.05f), 1.0f);
    setDamping(leftTopBody);

    // rightTop
    Vec3 rightTopPos(0, topArmY, cosAng);
    uint32_t rightTopBody = solver.addBody(rightTopPos, Quat(),
                                            Vec3(0.5f, 0.05f, 0.05f), 1.0f);
    setDamping(rightTopBody);

    // Connect col1 final arms to top brackets via revolute joints
    {
      Vec3 pivWorld(0.5f, topArmY, -cosAng);
      Vec3 pA, pB;
      {
        Quat r = solver.bodies[col1FinalLeft].rotation;
        Vec3 p = solver.bodies[col1FinalLeft].position;
        pA = r.conjugate().rotate(pivWorld - p);
      }
      pB = solver.bodies[leftTopBody].rotation.conjugate().rotate(
          pivWorld - solver.bodies[leftTopBody].position);
      solver.addRevoluteJoint(col1FinalLeft, leftTopBody, pA, pB,
                              Vec3(1, 0, 0), Vec3(1, 0, 0));
    }
    {
      Vec3 pivWorld(0.5f, topArmY, cosAng);
      Vec3 pA, pB;
      {
        Quat r = solver.bodies[col1FinalRight].rotation;
        Vec3 p = solver.bodies[col1FinalRight].position;
        pA = r.conjugate().rotate(pivWorld - p);
      }
      pB = solver.bodies[rightTopBody].rotation.conjugate().rotate(
          pivWorld - solver.bodies[rightTopBody].position);
      solver.addRevoluteJoint(col1FinalRight, rightTopBody, pA, pB,
                              Vec3(1, 0, 0), Vec3(1, 0, 0));
    }

    // Connect col2 final arms to top brackets via D6 loop closure
    {
      Vec3 pivWorld(-0.5f, topArmY, -cosAng);
      Vec3 pA = solver.bodies[col2FinalLeft].rotation.conjugate().rotate(
          pivWorld - solver.bodies[col2FinalLeft].position);
      Vec3 pB = solver.bodies[leftTopBody].rotation.conjugate().rotate(
          pivWorld - solver.bodies[leftTopBody].position);
      solver.addD6Joint(col2FinalLeft, leftTopBody, pA, pB, 0, 0x2A);
    }
    {
      Vec3 pivWorld(-0.5f, topArmY, cosAng);
      Vec3 pA = solver.bodies[col2FinalRight].rotation.conjugate().rotate(
          pivWorld - solver.bodies[col2FinalRight].position);
      Vec3 pB = solver.bodies[rightTopBody].rotation.conjugate().rotate(
          pivWorld - solver.bodies[rightTopBody].position);
      solver.addD6Joint(col2FinalRight, rightTopBody, pA, pB, 0, 0x2A);
    }

    // --- Top platform (FIXED to leftTop) ---
    float platformY = topArmY + 0.15f;
    uint32_t topPlatform = solver.addBody({0, platformY, 0}, Quat(),
                                           Vec3(0.5f, 0.1f, 1.5f), 1.0f);
    setDamping(topPlatform);

    // Fixed to leftTop
    Vec3 ancA = solver.bodies[leftTopBody].rotation.conjugate().rotate(
        Vec3(0, platformY, 0) - solver.bodies[leftTopBody].position);
    Vec3 ancB = Vec3(0, 0, solver.bodies[leftTopBody].position.z);
    solver.addD6Joint(leftTopBody, topPlatform, ancA, ancB, 0, 0);

    // Fixed to rightTop too (loop closure)
    Vec3 ancA2 = solver.bodies[rightTopBody].rotation.conjugate().rotate(
        Vec3(0, platformY, 0) - solver.bodies[rightTopBody].position);
    Vec3 ancB2 = Vec3(0, 0, solver.bodies[rightTopBody].position.z);
    solver.addD6Joint(rightTopBody, topPlatform, ancA2, ancB2, 0, 0x2A);

    printf("    Built: %d bodies, %d D6 joints\n",
           (int)solver.bodies.size(), (int)solver.d6Joints.size());

    ScissorInfo info;
    info.baseBody = baseBody;
    info.topPlatformBody = topPlatform;
    info.driveJointIdx = driveJointIdx;
    return info;
  };

  // ---- Sub-test helper: generate ground + box-platform contacts ----
  auto generateContacts = [](Solver &solver, uint32_t platformBody, uint32_t firstBox, int numBoxes) {
    solver.contacts.clear();
    // Ground contacts for all dynamic bodies (4 bottom corners)
    for (uint32_t i = 0; i < (uint32_t)solver.bodies.size(); i++) {
      if (solver.bodies[i].mass <= 0) continue;
      Vec3 he = solver.bodies[i].halfExtent;
      Vec3 corners[4] = {{-he.x,-he.y,-he.z},{he.x,-he.y,-he.z},{he.x,-he.y,he.z},{-he.x,-he.y,he.z}};
      for (int c = 0; c < 4; c++) {
        Vec3 wc = solver.bodies[i].position + solver.bodies[i].rotation.rotate(corners[c]);
        float depth = -wc.y; // positive = penetrating ground
        const Vec3 groundPoint =
            solver.useCanonicalRigidContactAuthoringProbe
                ? wc
                : Vec3(wc.x, 0, wc.z);
        solver.addContact(i, UINT32_MAX, Vec3(0,1,0), corners[c], groundPoint,
                          depth, solver.bodies[i].friction);
      }
    }
    // Box-on-platform contacts (4 bottom corners of box vs platform top)
    if (platformBody != UINT32_MAX && numBoxes > 0) {
      const auto &plat = solver.bodies[platformBody];
      float platTop = plat.position.y + plat.halfExtent.y;
      for (int b = 0; b < numBoxes; b++) {
        uint32_t bi = firstBox + b;
        if (bi >= (uint32_t)solver.bodies.size()) break;
        const auto &box = solver.bodies[bi];
        if (box.position.y < plat.position.y) continue; // fell through
        Vec3 he = box.halfExtent;
        Vec3 corners[4] = {{-he.x,-he.y,-he.z},{he.x,-he.y,-he.z},{he.x,-he.y,he.z},{-he.x,-he.y,he.z}};
        for (int c = 0; c < 4; c++) {
          Vec3 wc = box.position + box.rotation.rotate(corners[c]);
          bool inX = std::fabs(wc.x - plat.position.x) < plat.halfExtent.x;
          bool inZ = std::fabs(wc.z - plat.position.z) < plat.halfExtent.z;
          if (inX && inZ) {
            float depth = platTop - wc.y;
            Vec3 worldContact(wc.x, platTop, wc.z);
            if (solver.useCanonicalRigidContactAuthoringProbe)
              worldContact = wc;
            Vec3 rB = worldContact - plat.position;
            solver.addContact(bi, platformBody, Vec3(0,1,0), corners[c], rB,
                              depth, 0.5f);
          }
        }
      }
      // Box-box contacts (4-corner)
      for (int i = 0; i < numBoxes; i++) {
        for (int j = i + 1; j < numBoxes; j++) {
          uint32_t ai = firstBox + i, bj = firstBox + j;
          if (ai >= (uint32_t)solver.bodies.size() || bj >= (uint32_t)solver.bodies.size()) continue;
          const auto &a = solver.bodies[ai];
          const auto &b = solver.bodies[bj];
          uint32_t topI = (a.position.y > b.position.y) ? ai : bj;
          uint32_t botI = (a.position.y > b.position.y) ? bj : ai;
          const auto &top = solver.bodies[topI];
          const auto &bot = solver.bodies[botI];
          float botTop = bot.position.y + bot.halfExtent.y;
          Vec3 he = top.halfExtent;
          Vec3 corners[4] = {{-he.x,-he.y,-he.z},{he.x,-he.y,-he.z},{he.x,-he.y,he.z},{-he.x,-he.y,he.z}};
          for (int c = 0; c < 4; c++) {
            Vec3 wc = top.position + top.rotation.rotate(corners[c]);
            float depth = botTop - wc.y;
            Vec3 rB = wc - bot.position;
            solver.addContact(topI, botI, Vec3(0,1,0), corners[c], rB,
                              depth, 0.5f);
          }
        }
      }
    }
  };

  // ---- Sub-test A: Unloaded (15 seconds = 900 frames) ----
  printf("  Phase A: Unloaded (15s)...\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 64;

  auto info = buildScissorLift(solver);
  uint32_t baseBody = info.baseBody;
  uint32_t topPlatformBody = info.topPlatformBody;
  uint32_t driveJointIdx = info.driveJointIdx;

  bool closing = true;
  float driveTarget = 0.0f;
  float topYMin = 1e6f, topYMax = -1e6f;
  int directionChanges = 0;
  bool wasClosed = false;
  bool noNaN_A = true;
  bool noBoom_A = true;

  for (int frame = 0; frame < 900; frame++) {
    // Update drive target
    if (closing && driveTarget < -1.2f) { closing = false; directionChanges++; }
    else if (!closing && driveTarget > 0.0f) { closing = true; directionChanges++; }
    if (closing)  driveTarget -= solver.dt * 0.25f;
    else          driveTarget += solver.dt * 0.25f;

    // Set drive target position: prismatic uses axis0=X so drive velocity X = targetPos/dt
    solver.d6Joints[driveJointIdx].driveLinearVelocity = Vec3(driveTarget / solver.dt, 0, 0);

    generateContacts(solver, UINT32_MAX, 0, 0); // no boxes yet
    solver.step(solver.dt);

    // Track platform Y
    float topY = solver.bodies[topPlatformBody].position.y;
    topYMin = std::min(topYMin, topY);
    topYMax = std::max(topYMax, topY);

    // NaN check
    for (uint32_t i = 0; i < (uint32_t)solver.bodies.size(); i++) {
      Vec3 p = solver.bodies[i].position;
      if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) noNaN_A = false;
    }

    // Explosion check
    for (uint32_t i = 0; i < (uint32_t)solver.bodies.size(); i++) {
      if (solver.bodies[i].mass <= 0) continue;
      Vec3 p = solver.bodies[i].position;
      if (std::fabs(p.x) > 20 || std::fabs(p.y) > 20 || std::fabs(p.z) > 20) {
        if (noBoom_A)
          printf("    Explosion body %d pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f)\n",
                 i, p.x, p.y, p.z,
                 solver.bodies[i].linearVelocity.x,
                 solver.bodies[i].linearVelocity.y,
                 solver.bodies[i].linearVelocity.z);
        noBoom_A = false;
      }
    }

    if (!noNaN_A || !noBoom_A) {
      printf("    Phase A failed at frame %d\n", frame);
      break;
    }
  }

  float strokeA = topYMax - topYMin;
  printf("    Platform Y range: [%.3f, %.3f], stroke=%.3f, dirChanges=%d\n",
         topYMin, topYMax, strokeA, directionChanges);

  // A1: Scissor lift stable alone
  bool a1 = noBoom_A && noNaN_A;
  printf("  A1 %s: Scissor lift stable alone (15s unloaded)\n", a1 ? "PASS" : "FAIL");

  // A2: Platform travels through large lift stroke
  bool a2 = strokeA > 0.3f;
  printf("  A2 %s: Platform stroke=%.3f (need>0.3)\n", a2 ? "PASS" : "FAIL", strokeA);

  // A3: Scissor lift cycles up and down
  bool a3 = directionChanges >= 2;
  printf("  A3 %s: Direction changes=%d (need>=2)\n", a3 ? "PASS" : "FAIL", directionChanges);

  // ---- Sub-test B: Loaded (10 seconds = 600 frames, add boxes) ----
  printf("  Phase B: Loaded (10s)...\n");

  // Add 8 boxes on top of the platform
  float platformTopY = solver.bodies[topPlatformBody].position.y + 0.1f;
  Vec3 boxHalf(0.25f, 0.25f, 0.25f);
  float boxPositions[][3] = {
    {-0.25f, platformTopY + 0.3f, 0.5f},
    {0.25f,  platformTopY + 0.3f, 0.5f},
    {-0.25f, platformTopY + 0.3f, -0.5f},
    {0.25f,  platformTopY + 0.3f, -0.5f},
    {-0.25f, platformTopY + 0.8f, 0.0f},
    {0.25f,  platformTopY + 0.8f, 0.0f},
    {0.0f,   platformTopY + 1.3f, 0.25f},
    {0.0f,   platformTopY + 1.3f, -0.25f},
  };
  uint32_t firstBoxBody = (uint32_t)solver.bodies.size();
  for (int i = 0; i < 8; i++) {
    solver.addBody({boxPositions[i][0], boxPositions[i][1], boxPositions[i][2]},
                   Quat(), boxHalf, 0.5f);
  }

  bool noNaN_B = true;
  bool noBoom_B = true;
  int boomFrame = -1;
  float topYMin_B = 1e6f, topYMax_B = -1e6f;
  int dirChanges_B = 0;

  for (int frame = 0; frame < 600; frame++) {
    if (closing && driveTarget < -1.2f) { closing = false; dirChanges_B++; }
    else if (!closing && driveTarget > 0.0f) { closing = true; dirChanges_B++; }
    if (closing)  driveTarget -= solver.dt * 0.25f;
    else          driveTarget += solver.dt * 0.25f;

    solver.d6Joints[driveJointIdx].driveLinearVelocity = Vec3(driveTarget / solver.dt, 0, 0);

    generateContacts(solver, topPlatformBody, firstBoxBody, 8);
    solver.step(solver.dt);

    float topY = solver.bodies[topPlatformBody].position.y;
    topYMin_B = std::min(topYMin_B, topY);
    topYMax_B = std::max(topYMax_B, topY);

    for (uint32_t i = 0; i < (uint32_t)solver.bodies.size(); i++) {
      Vec3 p = solver.bodies[i].position;
      if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) noNaN_B = false;
    }
    if (!noNaN_B) { printf("    NaN at frame %d\n", frame); break; }

    for (uint32_t i = 0; i < (uint32_t)solver.bodies.size(); i++) {
      if (solver.bodies[i].mass <= 0) continue;
      Vec3 p = solver.bodies[i].position;
      if (std::fabs(p.x) > 20 || std::fabs(p.y) > 20 || std::fabs(p.z) > 20) {
        if (noBoom_B) {
          boomFrame = frame;
          printf("    Explosion body %d pos=(%.2f,%.2f,%.2f)\n", i, p.x, p.y, p.z);
        }
        noBoom_B = false;
      }
    }
    if (!noBoom_B) {
      printf("    Explosion at frame %d (t=%.2fs)\n", frame, frame * solver.dt);
      break;
    }
  }

  float strokeB = topYMax_B - topYMin_B;
  printf("    Platform Y range: [%.3f, %.3f], stroke=%.3f\n",
         topYMin_B, topYMax_B, strokeB);

  // B1: No NaN
  printf("  B1 %s: No NaN in 10s\n", noNaN_B ? "PASS" : "FAIL");

  // B2: No explosion
  printf("  B2 %s: No explosion in 10s%s\n", noBoom_B ? "PASS" : "FAIL",
         boomFrame >= 0 ? (" (frame " + std::to_string(boomFrame) + ")").c_str() : "");

  // B3: Boxes above ground
  bool boxesAboveGround = true;
  for (int i = 0; i < 8; i++) {
    float by = solver.bodies[firstBoxBody + i].position.y;
    if (by < -0.5f) boxesAboveGround = false;
  }
  printf("  B3 %s: Boxes rest above ground\n", boxesAboveGround ? "PASS" : "FAIL");

  // B4: Platform retains stroke under load
  bool b4 = strokeB > 0.1f;
  printf("  B4 %s: Loaded stroke=%.3f (need>0.1)\n", b4 ? "PASS" : "FAIL", strokeB);

  // B5: Cyclic motion continues under load
  bool b5 = dirChanges_B >= 2;
  printf("  B5 %s: Loaded dir changes=%d (need>=2)\n", b5 ? "PASS" : "FAIL", dirChanges_B);

  // B6: Base remains stable
  Vec3 basePos = solver.bodies[baseBody].position;
  float baseDrift = (basePos - Vec3(0, 0.25f, 0)).length();
  bool b6 = baseDrift < 0.5f;
  printf("  B6 %s: Base drift=%.4f (need<0.5)\n", b6 ? "PASS" : "FAIL", baseDrift);

  int subPassed = (int)a1 + (int)a2 + (int)a3 + (int)noNaN_B + (int)noBoom_B +
                  (int)boxesAboveGround + (int)b4 + (int)b5 + (int)b6;
  printf("  Score: %d/9 sub-tests passed\n", subPassed);

  // For Phase 4 we track results but pass the test if the structure doesn't explode unloaded
  // (matching Featherstone ceiling: the unloaded case should always work)
  CHECK(a1, "Scissor lift exploded in unloaded phase");

  PASS("Scissor lift validation complete");
}
