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

bool test11_collisionSingleBox() {
  printf("\n--- Test 11: Collision-detected single box on ground ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;
  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
  ContactCache cache;
  for (int frame = 0; frame < 120; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.05f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }
  float finalY = solver.bodies[box].position.y;
  CHECK(fabsf(finalY - 1.0f) < 0.1f, "collision box drifted: y=%.4f", finalY);
  PASS("collision single box stable");
}

bool test12_collisionThreeStack() {
  printf("\n--- Test 12: Collision-detected 3-box stack ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;
  Vec3 halfExt(1, 1, 1);
  float density = 10.0f;
  uint32_t b0 = solver.addBody({0, 1, 0}, Quat(), halfExt, density, 0.5f);
  uint32_t b1 = solver.addBody({0, 3, 0}, Quat(), halfExt, density, 0.5f);
  uint32_t b2 = solver.addBody({0, 5, 0}, Quat(), halfExt, density, 0.5f);
  ContactCache cache;
  for (int frame = 0; frame < 180; frame++) {
    solver.contacts.clear();
    collideAll(solver, 0.05f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }
  CHECK(fabsf(solver.bodies[b0].position.y - 1.0f) < 0.15f, "b0 drifted");
  CHECK(fabsf(solver.bodies[b1].position.y - 3.0f) < 0.15f, "b1 drifted");
  CHECK(fabsf(solver.bodies[b2].position.y - 5.0f) < 0.15f, "b2 drifted");
  PASS("collision-detected 3-box stack stable");
}

bool test13_collisionDrop() {
  printf("\n--- Test 13: Collision drop + settle ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;
  Vec3 halfExt(1, 1, 1);
  uint32_t box = solver.addBody({0, 4, 0}, Quat(), halfExt, 10.0f, 0.5f);
  ContactCache cache;
  for (int frame = 0; frame < 360; frame++) {
    solver.contacts.clear();
    collideBoxGround(solver, box, 0.1f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
  }
  float finalY = solver.bodies[box].position.y;
  CHECK(fabsf(finalY - 1.0f) < 0.15f, "didn't settle: y=%.4f", finalY);
  PASS("collision drop settled");
}

bool test14_collisionPhysxTower() {
  printf("\n--- Test 14: Collision PhysX-scale 5-box tower (stress) ---\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 12;
  solver.dt = 1.0f / 60.0f;
  Vec3 halfExt(2, 2, 2);
  float density = 10.0f;
  const int N = 5;
  uint32_t ids[N];
  for (int i = 0; i < N; i++)
    ids[i] =
        solver.addBody({0, 2.0f + 4.0f * i, 0}, Quat(), halfExt, density, 0.5f);
  ContactCache cache;
  bool exploded = false;
  float maxDrift = 0;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    collideAll(solver, 0.05f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    for (int i = 0; i < N; i++) {
      float expected = 2.0f + 4.0f * i;
      float drift = fabsf(solver.bodies[ids[i]].position.y - expected);
      if (drift > maxDrift)
        maxDrift = drift;
      if (fabsf(solver.bodies[ids[i]].position.y) > 200.0f)
        exploded = true;
    }
  }
  CHECK(!exploded, "EXPLOSION!");
  CHECK(maxDrift < 0.5f, "tower collapsed: maxDrift=%.4f", maxDrift);
  PASS("collision PhysX-scale 5-box tower stable");
}

bool test15_pyramidStack() {
  printf("\n--- Test 15: Pyramid stack (PhysX SnippetHelloWorld layout) ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;
  Vec3 halfExt(2, 2, 2);
  float density = 10.0f;
  const int size = 10;
  std::vector<uint32_t> ids;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size - i; j++) {
      float x = (float)(j * 2 - (size - i)) * halfExt.x;
      float y = (float)(i * 2 + 1) * halfExt.y;
      ids.push_back(solver.addBody({x, y, 0}, Quat(), halfExt, density, 0.5f));
    }
  }
  ContactCache cache;
  bool exploded = false;
  for (int frame = 0; frame < 600; frame++) {
    solver.contacts.clear();
    collideAll(solver, 0.05f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    for (size_t bi = 0; bi < ids.size(); bi++) {
      if (fabsf(solver.bodies[ids[bi]].position.y) > 200.0f)
        exploded = true;
    }
  }
  CHECK(!exploded, "EXPLOSION!");
  PASS("pyramid stack stable (PhysX layout)");
}

bool test16_pyramidNoFriction() {
  printf("\n--- Test 16: Pyramid stack (NO friction) ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.iterations = 10;
  solver.dt = 1.0f / 60.0f;
  Vec3 halfExt(2, 2, 2);
  float density = 10.0f;
  const int size = 10;
  std::vector<uint32_t> ids;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size - i; j++) {
      float x = (float)(j * 2 - (size - i)) * halfExt.x;
      float y = (float)(i * 2 + 1) * halfExt.y;
      ids.push_back(solver.addBody({x, y, 0}, Quat(), halfExt, density, 0.0f));
    }
  }
  ContactCache cache;
  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    collideAll(solver, 0.05f);
    cache.restore(solver);
    solver.step(solver.dt);
    cache.save(solver);
    for (size_t bi = 0; bi < ids.size(); bi++) {
      if (fabsf(solver.bodies[ids[bi]].position.y) > 200.0f)
        exploded = true;
    }
  }
  CHECK(!exploded, "EXPLOSION even without friction!");
  PASS("pyramid no-friction: collapsed but no explosion");
}
