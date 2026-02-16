#include "avbd_test_utils.h"
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
  for (int frame = 0; frame < 360; frame++) {
    solver.contacts.clear();
    addBoxGroundContacts(solver, boxIds[0], halfExt);
    for (int i = 1; i < N; i++)
      addBoxOnBoxContacts(solver, boxIds[i], boxIds[i - 1], halfExt, halfExt);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
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
