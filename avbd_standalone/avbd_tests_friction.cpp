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

// =============================================================================
// Helper: run N frames with collision detection
// =============================================================================
static void runFrames(Solver &solver, int nFrames, float margin = 0.05f) {
  ContactCache cache;
  for (int frame = 0; frame < nFrames; frame++) {
    solver.contacts.clear();
    collideAll(solver, margin);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }
}

// =============================================================================
// Test 56: Box on tilted plane — zero friction → slides freely
//
// A box on a 30° slope with friction=0 should slide down under gravity.
// Verify significant X displacement after simulation.
// =============================================================================
bool test56_tiltedPlane_zeroFriction() {
  printf("\n--- Test 56: Angled gravity, zero friction → slides ---\n");
  Solver solver;
  // Simulate a 30° slope via tilted gravity on flat ground.
  // Lateral component of gravity drives the box sideways;
  // the normal component keeps it pressed to the ground.
  float angle = 30.0f * 3.14159265f / 180.0f;
  solver.gravity = {9.81f * sinf(angle), -9.81f * cosf(angle), 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.0f);

  ContactCache cache;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float finalX = solver.bodies[box].position.x;
  printf("  finalX=%.4f (expected significant positive slide)\n", finalX);
  CHECK(finalX > 1.0f, "zero-friction box didn't slide: x=%.4f", finalX);

  PASS("zero friction: box slides under angled gravity");
}

// =============================================================================
// Test 57: Box on tilted plane — high friction → stays put
//
// Same 30° slope but with friction=1.0 (tan(30°)≈0.577 < 1.0).
// High friction should prevent sliding.
// =============================================================================
bool test57_tiltedPlane_highFriction() {
  printf("\n--- Test 57: Angled gravity, high friction → stays ---\n");
  Solver solver;
  float angle = 30.0f * 3.14159265f / 180.0f;
  solver.gravity = {9.81f * sinf(angle), -9.81f * cosf(angle), 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 1.0f);

  ContactCache cache;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float finalX = solver.bodies[box].position.x;
  printf("  finalX=%.4f (expected near 0)\n", finalX);
  CHECK(fabsf(finalX) < 0.5f, "high-friction box slid too far: x=%.4f", finalX);

  PASS("high friction: box stays under angled gravity");
}

// =============================================================================
// Test 58: Friction coefficient comparison — low vs high
//
// Two boxes on flat ground given identical lateral push (initial velocity).
// Low friction box should slide further than high friction box.
// =============================================================================
bool test58_frictionComparison_lowVsHigh() {
  printf("\n--- Test 58: Friction comparison: low vs high ---\n");

  float finalX_low, finalX_high;

  // Low friction run
  {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;

    Vec3 halfExt(1, 1, 1);
    uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.1f);
    solver.bodies[box].linearVelocity = {5.0f, 0, 0};

    ContactCache cache;
    for (int frame = 0; frame < 300; frame++) {
      solver.contacts.clear();
      collideBoxGround(solver, box, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    finalX_low = solver.bodies[box].position.x;
  }

  // High friction run
  {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;

    Vec3 halfExt(1, 1, 1);
    uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.8f);
    solver.bodies[box].linearVelocity = {5.0f, 0, 0};

    ContactCache cache;
    for (int frame = 0; frame < 300; frame++) {
      solver.contacts.clear();
      collideBoxGround(solver, box, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    finalX_high = solver.bodies[box].position.x;
  }

  printf("  low friction finalX=%.4f, high friction finalX=%.4f\n",
         finalX_low, finalX_high);
  CHECK(finalX_low > finalX_high,
        "low friction should slide further: low=%.4f high=%.4f", finalX_low,
        finalX_high);

  PASS("low friction slides further than high friction");
}

// =============================================================================
// Test 59: Zero friction → box slides indefinitely (no lateral deceleration)
//
// A box on flat ground with zero friction and lateral velocity should
// maintain its velocity (no tangential force to slow it down).
// =============================================================================
bool test59_zeroFriction_noDeceleration() {
  printf("\n--- Test 59: Zero friction → no lateral deceleration ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.0f);
  solver.bodies[box].linearVelocity = {3.0f, 0, 0};

  ContactCache cache;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float finalX = solver.bodies[box].position.x;
  // With no friction, box should travel ~3.0 * 2.0s = 6.0 units (120 frames at 60Hz = 2s)
  // Allow for some numerical damping but should be significant
  printf("  finalX=%.4f (expected ~6.0)\n", finalX);
  CHECK(finalX > 3.0f, "zero-friction box decelerated too much: x=%.4f", finalX);

  PASS("zero friction: box slides freely");
}

// =============================================================================
// Test 60: High friction → box stops quickly
//
// Same initial velocity but friction=1.0 → box should come to rest quickly.
// =============================================================================
bool test60_highFriction_stopsQuickly() {
  printf("\n--- Test 60: High friction → stops quickly ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 1.0f);
  solver.bodies[box].linearVelocity = {3.0f, 0, 0};

  ContactCache cache;
  // Record position at frame 60 and 120
  float x60 = 0, x120 = 0;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    if (frame == 59)
      x60 = solver.bodies[box].position.x;
  }
  x120 = solver.bodies[box].position.x;

  float drift = fabsf(x120 - x60);
  printf("  x@60=%.4f x@120=%.4f drift=%.4f\n", x60, x120, drift);
  CHECK(drift < 0.5f, "high-friction box still moving: drift=%.4f", drift);

  PASS("high friction: box stops quickly");
}

// =============================================================================
// Test 61: Friction prevents pyramid collapse
//
// A 5-level pyramid with friction=0.5 should remain stable.
// Compare lateral spread with and without friction to verify friction prevents
// the pyramid from spreading.
// =============================================================================
bool test61_pyramidFrictionStability() {
  printf("\n--- Test 61: Pyramid stability with friction ---\n");

  float maxLateralWith, maxLateralWithout;

  // With friction
  {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 12;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    const int size = 5;
    std::vector<uint32_t> ids;
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size - i; j++) {
        float x = (float)(j * 2 - (size - i - 1)) * halfExt.x;
        float y = (float)(i * 2 + 1) * halfExt.y;
        ids.push_back(solver.addBody({x, y, 0}, Quat(), halfExt, 10.0f, 0.5f));
      }
    runFrames(solver, 300);
    maxLateralWith = 0;
    for (size_t bi = 0; bi < ids.size(); bi++) {
      float x = fabsf(solver.bodies[ids[bi]].position.x);
      if (x > maxLateralWith)
        maxLateralWith = x;
    }
  }

  // Without friction
  {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 12;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    const int size = 5;
    std::vector<uint32_t> ids;
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size - i; j++) {
        float x = (float)(j * 2 - (size - i - 1)) * halfExt.x;
        float y = (float)(i * 2 + 1) * halfExt.y;
        ids.push_back(solver.addBody({x, y, 0}, Quat(), halfExt, 10.0f, 0.0f));
      }
    runFrames(solver, 300);
    maxLateralWithout = 0;
    for (size_t bi = 0; bi < ids.size(); bi++) {
      float x = fabsf(solver.bodies[ids[bi]].position.x);
      if (x > maxLateralWithout)
        maxLateralWithout = x;
    }
  }

  printf("  maxLateral: withFriction=%.4f withoutFriction=%.4f\n",
         maxLateralWith, maxLateralWithout);
  CHECK(maxLateralWith < maxLateralWithout,
        "friction should reduce lateral spread: with=%.4f without=%.4f",
        maxLateralWith, maxLateralWithout);

  PASS("pyramid friction stability");
}

// =============================================================================
// Test 62: Stacked boxes — friction slows top box lateral sliding
//
// Top box given lateral velocity on a stationary bottom box.
// High friction should reduce how far the top box slides compared to zero friction.
// =============================================================================
bool test62_stackedBoxOffset_frictionHolds() {
  printf("\n--- Test 62: Stacked box lateral slide — friction holds ---\n");

  auto runStack = [](float fric) -> float {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 12;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t bot = solver.addBody({0, 1, 0}, Quat(), halfExt, 100.0f, fric);
    uint32_t top = solver.addBody({0, 3, 0}, Quat(), halfExt, 10.0f, fric);
    solver.bodies[top].linearVelocity = {3.0f, 0, 0};
    ContactCache cache;
    for (int frame = 0; frame < 300; frame++) {
      solver.contacts.clear();
      collideAll(solver, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    return solver.bodies[top].position.x;
  };

  float xFric = runStack(0.8f);
  float xNoFric = runStack(0.0f);

  printf("  top box X: withFric=%.4f noFric=%.4f\n", xFric, xNoFric);
  CHECK(xNoFric > xFric,
        "friction should reduce top box slide: fric=%.4f noFric=%.4f",
        xFric, xNoFric);

  PASS("stacked box: friction holds top box");
}

// =============================================================================
// Test 63: Box pushed by lateral force — friction opposes sliding
//
// Apply a constant lateral push via velocity kick each frame.
// Lower friction → slides more, higher friction → slides less.
// Use small friction values so the push can still overcome friction.
// =============================================================================
bool test63_lateralPush_frictionResists() {
  printf("\n--- Test 63: Lateral push resisted by friction ---\n");

  auto runPush = [](float fric) -> float {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, fric);
    ContactCache cache;
    for (int frame = 0; frame < 180; frame++) {
      solver.contacts.clear();
      collideBoxGround(solver, box, 0.1f);
      cache.restore(solver);
      // Larger lateral kick to overcome friction
      solver.bodies[box].linearVelocity.x += 3.0f * solver.dt;
      solver.step(solver.dt);
      cache.save(solver);
    }
    return solver.bodies[box].position.x;
  };

  float x_zero = runPush(0.0f);
  float x_low = runPush(0.1f);
  float x_high = runPush(0.5f);

  printf("  fric=0: x=%.4f, fric=0.1: x=%.4f, fric=0.5: x=%.4f\n", x_zero,
         x_low, x_high);
  CHECK(x_zero > x_low, "zero fric should slide more: 0=%.4f 0.1=%.4f",
        x_zero, x_low);
  CHECK(x_low > x_high, "low fric should slide more than high: 0.1=%.4f 0.5=%.4f",
        x_low, x_high);

  PASS("lateral push: friction ordering correct");
}

// =============================================================================
// Test 64: Friction isotropy — box slides equally in X and Z
//
// Give a box diagonal velocity (X+Z). Friction should decelerate both
// components equally. Final X and Z displacements should be similar.
// =============================================================================
bool test64_frictionIsotropy() {
  printf("\n--- Test 64: Friction isotropy (X vs Z) ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
  solver.bodies[box].linearVelocity = {3.0f, 0, 3.0f};

  ContactCache cache;
  for (int frame = 0; frame < 180; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float finalX = solver.bodies[box].position.x;
  float finalZ = solver.bodies[box].position.z;
  float ratio = (fabsf(finalX) > 0.01f && fabsf(finalZ) > 0.01f)
                    ? fabsf(finalX / finalZ)
                    : 1.0f;

  printf("  finalX=%.4f finalZ=%.4f ratio=%.4f\n", finalX, finalZ, ratio);
  CHECK(ratio > 0.5f && ratio < 2.0f,
        "friction not isotropic: X=%.4f Z=%.4f ratio=%.4f", finalX, finalZ,
        ratio);

  PASS("friction isotropy");
}

// =============================================================================
// Test 65: Dynamic-dynamic friction — two boxes sliding against each other
//
// Bottom box sits on frictionless ground. Top box has lateral velocity.
// Friction between top and bottom should slow the top while dragging the bottom
// (Newton's third law). Bottom box uses zero ground-friction so it can slide.
// =============================================================================
bool test65_dynamicDynamicFriction() {
  printf("\n--- Test 65: Dynamic-dynamic friction (Newton's 3rd law) ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 12;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  // Both bodies have friction=0.5 so box-box contacts get friction.
  // We manually override ground contacts with fric=0 so the bottom can slide.
  uint32_t bot = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
  uint32_t top = solver.addBody({0, 3, 0}, Quat(), halfExt, 10.0f, 0.5f);
  solver.bodies[top].linearVelocity = {4.0f, 0, 0};

  // Use manual collision: ground contacts with zero friction, box-box with 0.5
  ContactCache cache;
  for (int frame = 0; frame < 240; frame++) {
    solver.contacts.clear();
    // Ground contacts for bottom box — zero friction
    {
      Body &b = solver.bodies[bot];
      Vec3 corners[4] = {
          {-halfExt.x, -halfExt.y, -halfExt.z},
          {halfExt.x, -halfExt.y, -halfExt.z},
          {halfExt.x, -halfExt.y, halfExt.z},
          {-halfExt.x, -halfExt.y, halfExt.z},
      };
      for (int i = 0; i < 4; i++) {
        Vec3 wc = b.position + b.rotation.rotate(corners[i]);
        if (wc.y < 0.1f) {
          solver.addContact(bot, UINT32_MAX, {0, 1, 0}, corners[i],
                            {wc.x, 0, wc.z}, -wc.y, 0.0f);
        }
      }
    }
    // Box-box contacts with friction
    addBoxOnBoxContacts(solver, top, bot, halfExt, halfExt);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float topX = solver.bodies[top].position.x;
  float botX = solver.bodies[bot].position.x;

  printf("  topX=%.4f botX=%.4f\n", topX, botX);
  CHECK(topX < 12.0f, "top box didn't decelerate: topX=%.4f", topX);
  CHECK(botX > 0.05f, "bottom box wasn't dragged: botX=%.4f", botX);

  PASS("dynamic-dynamic friction: action-reaction");
}

// =============================================================================
// Test 66: Mass ratio effect on friction — heavy vs light box
//
// Same friction coefficient, same initial velocity, different masses.
// Both should stop at similar distances (friction force ~ m*g*mu, deceleration ~ g*mu).
// =============================================================================
bool test66_massRatioFriction() {
  printf("\n--- Test 66: Mass ratio — friction deceleration independent of mass ---\n");

  auto runMass = [](float density) -> float {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, density, 0.5f);
    solver.bodies[box].linearVelocity = {3.0f, 0, 0};
    ContactCache cache;
    for (int frame = 0; frame < 240; frame++) {
      solver.contacts.clear();
      collideBoxGround(solver, box, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    return solver.bodies[box].position.x;
  };

  float xLight = runMass(1.0f);
  float xHeavy = runMass(100.0f);

  printf("  light(1kg/m3) x=%.4f, heavy(100kg/m3) x=%.4f\n", xLight, xHeavy);
  // Coulomb friction: a = mu*g, independent of mass → similar stopping distance
  float ratio = (xHeavy > 0.01f) ? xLight / xHeavy : 0;
  CHECK(ratio > 0.3f && ratio < 3.0f,
        "mass-independent friction violated: light=%.4f heavy=%.4f ratio=%.4f",
        xLight, xHeavy, ratio);

  PASS("friction deceleration roughly mass-independent");
}

// =============================================================================
// Test 67: Friction coefficient sweep — monotonic behavior
//
// Sweep friction from 0.0 to 1.0, verify that sliding distance decreases
// monotonically with increasing friction.
// =============================================================================
bool test67_frictionSweep_monotonic() {
  printf("\n--- Test 67: Friction sweep (0→1) monotonic ---\n");

  const int N = 5;
  float fricValues[N] = {0.0f, 0.2f, 0.4f, 0.6f, 1.0f};
  float results[N];

  for (int k = 0; k < N; k++) {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t box =
        solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, fricValues[k]);
    solver.bodies[box].linearVelocity = {5.0f, 0, 0};
    ContactCache cache;
    for (int frame = 0; frame < 300; frame++) {
      solver.contacts.clear();
      collideBoxGround(solver, box, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    results[k] = solver.bodies[box].position.x;
  }

  printf("  sweep:");
  for (int k = 0; k < N; k++)
    printf(" fric=%.1f→x=%.3f", fricValues[k], results[k]);
  printf("\n");

  bool monotonic = true;
  for (int k = 1; k < N; k++) {
    if (results[k] >= results[k - 1]) {
      printf("  FAIL monotonicity: fric=%.1f x=%.4f >= fric=%.1f x=%.4f\n",
             fricValues[k], results[k], fricValues[k - 1], results[k - 1]);
      monotonic = false;
    }
  }
  CHECK(monotonic, "friction sweep not monotonic");

  PASS("friction sweep monotonic");
}

// =============================================================================
// Test 68: Friction with rotation — spinning box on ground
//
// Give a box angular velocity about Y-axis (spin on ground).
// Friction should slow the spin. Zero friction should maintain spin.
// =============================================================================
bool test68_rotationalFriction() {
  printf("\n--- Test 68: Rotational friction (spinning box) ---\n");

  auto runSpin = [](float fric) -> float {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, fric);
    solver.bodies[box].angularVelocity = {0, 5.0f, 0}; // spin around Y

    ContactCache cache;
    for (int frame = 0; frame < 240; frame++) {
      solver.contacts.clear();
      collideBoxGround(solver, box, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    // Measure residual spin rate
    Quat dq = solver.bodies[box].rotation *
              solver.bodies[box].initialRotation.conjugate();
    if (dq.w < 0)
      dq = dq * (-1.0f);
    return fabsf(dq.y);
  };

  float spin_noFric = runSpin(0.0f);
  float spin_hiFric = runSpin(0.8f);

  printf("  spin_noFric_qy=%.4f spin_hiFric_qy=%.4f\n", spin_noFric,
         spin_hiFric);
  CHECK(spin_noFric > spin_hiFric,
        "friction should reduce spin: noFric=%.4f hiFric=%.4f",
        spin_noFric, spin_hiFric);

  PASS("rotational friction reduces spin");
}

// =============================================================================
// Test 69: Box resting on ground — friction prevents drift from numerical noise
//
// A box at rest with moderate friction should remain at rest (no creep).
// =============================================================================
bool test69_restingContactNoDrift() {
  printf("\n--- Test 69: Resting contact — no lateral drift ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);

  ContactCache cache;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float finalX = solver.bodies[box].position.x;
  float finalZ = solver.bodies[box].position.z;
  float lateral = sqrtf(finalX * finalX + finalZ * finalZ);

  printf("  lateral drift=%.6f\n", lateral);
  CHECK(lateral < 0.1f, "resting box drifted laterally: %.4f", lateral);

  PASS("resting contact: no lateral drift");
}

// =============================================================================
// Test 70: Tangent lambda signs — verify both tangent directions work
//
// Push box in negative X direction (opposite to typical).
// Friction should still decelerate.
// =============================================================================
bool test70_tangentDirection_negativeX() {
  printf("\n--- Test 70: Tangent friction in negative X ---\n");

  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
  solver.bodies[box].linearVelocity = {-3.0f, 0, 0}; // negative X

  ContactCache cache;
  for (int frame = 0; frame < 180; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }

  float finalX = solver.bodies[box].position.x;
  printf("  finalX=%.4f (should be negative, friction should slow it)\n",
         finalX);
  CHECK(finalX < 0, "box should have moved in -X: x=%.4f", finalX);
  // Should have decelerated (not traveled full -3*3=9 units)
  CHECK(finalX > -8.0f, "box slid too far in -X: x=%.4f", finalX);

  PASS("tangent friction works in negative direction");
}

// =============================================================================
// Test 71: Coulomb cone — tangent force bounded by normal force
//
// A very light box with large lateral velocity. Even with high friction coeff,
// tangent impulse should not exceed mu * normal impulse.
// Verify by checking no explosion or NaN.
// =============================================================================
bool test71_coulombCone_noExplosion() {
  printf("\n--- Test 71: Coulomb cone — no explosion on large lateral velocity ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 0.1f, 2.0f);
  solver.bodies[box].linearVelocity = {100.0f, 0, 0};

  ContactCache cache;
  bool hasNaN = false;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);

    Vec3 p = solver.bodies[box].position;
    if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
      hasNaN = true;
      break;
    }
  }

  CHECK(!hasNaN, "NaN detected in Coulomb cone test");
  float finalY = solver.bodies[box].position.y;
  CHECK(finalY > -10.0f && finalY < 50.0f,
        "box position unreasonable: y=%.4f", finalY);

  PASS("Coulomb cone: no explosion or NaN");
}

// =============================================================================
// Test 72: Geometric mean friction — asymmetric body friction values
//
// Two bodies with different friction coefficients. Contact friction should
// be sqrt(fricA * fricB). Verify behavior is between both extremes.
// =============================================================================
bool test72_geometricMeanFriction() {
  printf("\n--- Test 72: Geometric mean friction (asymmetric bodies) ---\n");

  // Body A (ground-like) friction=1.0, Body B (ice) friction=0.01
  // Combined = sqrt(1.0 * 0.01) = 0.1 → very low friction
  // Body A friction=1.0, Body B (rubber) friction=1.0
  // Combined = sqrt(1.0 * 1.0) = 1.0 → high friction

  auto runPair = [](float fricBot, float fricTop) -> float {
    Solver solver;
    solver.gravity = {0, -9.81f, 0};
    solver.iterations = 12;
    solver.dt = 1.0f / 60.0f;
    Vec3 halfExt(1, 1, 1);
    uint32_t bot = solver.addBody({0, 1, 0}, Quat(), halfExt, 100.0f, fricBot);
    uint32_t top = solver.addBody({0, 3, 0}, Quat(), halfExt, 10.0f, fricTop);
    solver.bodies[top].linearVelocity = {3.0f, 0, 0};
    ContactCache cache;
    for (int frame = 0; frame < 300; frame++) {
      solver.contacts.clear();
      collideAll(solver, 0.1f);
      cache.restore(solver);
      solver.step(solver.dt);
      cache.save(solver);
    }
    return solver.bodies[top].position.x;
  };

  float x_both_high = runPair(1.0f, 1.0f); // combined = 1.0
  float x_asymmetric = runPair(1.0f, 0.04f); // combined = 0.2
  float x_both_low = runPair(0.04f, 0.04f); // combined = 0.04

  printf("  both_high(1.0,1.0) x=%.4f, asym(1.0,0.04) x=%.4f, both_low(0.04,0.04) x=%.4f\n",
         x_both_high, x_asymmetric, x_both_low);

  // Geometric mean ordering: both_low > asymmetric > both_high
  CHECK(x_both_low > x_asymmetric,
        "both_low should slide more: low=%.4f asym=%.4f", x_both_low,
        x_asymmetric);
  CHECK(x_asymmetric > x_both_high,
        "asymmetric should slide more than both_high: asym=%.4f high=%.4f",
        x_asymmetric, x_both_high);

  PASS("geometric mean friction ordering correct");
}

// =============================================================================
// Test 73: Long-term friction stability — no energy gain
//
// A box resting on ground for 1200 frames. Should not gain lateral velocity
// or energy from numerical artifacts.
// =============================================================================
bool test73_longTermFrictionStability() {
  printf("\n--- Test 73: Long-term friction stability (1200 frames) ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;

  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);

  ContactCache cache;
  float maxLateral = 0;
  for (int frame = 0; frame < 1200; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);

    Vec3 p = solver.bodies[box].position;
    float lat = sqrtf(p.x * p.x + p.z * p.z);
    if (lat > maxLateral)
      maxLateral = lat;
  }

  printf("  maxLateral over 1200 frames: %.6f\n", maxLateral);
  CHECK(maxLateral < 0.1f, "energy gain or drift: maxLat=%.4f", maxLateral);

  PASS("long-term friction stability: no energy gain");
}
