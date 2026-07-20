// Deformable wavy mesh tests (SnippetDeformableMesh headless sphere-shot parity).
#include "avbd_deformable_scene.h"
#include "avbd_kinematic_shell.h"
#include "avbd_test_utils.h"
#include <cstdio>
#include <cmath>

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

bool test114_deformableSphereShot_sequential_gate() {
  printf("\n--- Test 114: Deformable sphere shot (sequential, Snippet gate) "
         "---\n");
  const DeformableSphereShotMetrics m = runDeformableSphereShot(
      BodyStaticContactSolve::SequentialPerContact, "sequential", 0.5f);
  CHECK(!m.nanDetected, "NaN in sequential run");
  CHECK(m.pass,
        "Snippet gate failed pen=%.4f (limit %.2f) lateral=%.4f (limit %.1f)",
        m.maxRaycastPen, kDeformPassPen, m.lateralDriftXZ, kDeformPassLateral);
  PASS("deformable sequential meets SnippetDeformableMesh gate");
}

bool test115_deformableAggregated_noFriction6x6() {
  printf("\n--- Test 115: Aggregated 6x6 (static normals only) ---\n");
  const DeformableSphereShotMetrics m = runDeformableSphereShot(
      BodyStaticContactSolve::Aggregated6x6, "aggregated6x6", 0.5f);
  CHECK(!m.nanDetected, "NaN in aggregated run");
  CHECK(m.maxRaycastPen <= kDeformPassPen + 0.15f,
        "aggregated pen too high (%.4f)", m.maxRaycastPen);
  CHECK(m.lateralDriftXZ <= kDeformPassLateral,
        "aggregated lateral drift (%.4f)", m.lateralDriftXZ);
  PASS("aggregated static normals-only stable");
}

bool test116_deformableFriction_dominantSequential() {
  printf("\n--- Test 116: Sequential friction (dominant contact) vs mu=0 ---\n");
  const DeformableSphereShotMetrics mMu0 = runDeformableSphereShot(
      BodyStaticContactSolve::SequentialPerContact, "sequential_mu0", 0.0f);
  const DeformableSphereShotMetrics mMuH = runDeformableSphereShot(
      BodyStaticContactSolve::SequentialPerContact, "sequential_mu05", 0.5f);

  printf("  mu=0   lateral=%.4f pen=%.4f pass=%d\n", mMu0.lateralDriftXZ,
         mMu0.maxRaycastPen, mMu0.pass ? 1 : 0);
  printf("  mu=0.5 lateral=%.4f pen=%.4f pass=%d\n", mMuH.lateralDriftXZ,
         mMuH.maxRaycastPen, mMuH.pass ? 1 : 0);

  CHECK(!mMu0.nanDetected && !mMuH.nanDetected, "NaN in friction comparison");
  CHECK(mMuH.pass, "mu=0.5 sequential failed Snippet gate");
  CHECK(mMuH.lateralDriftXZ <= kDeformPassLateral,
        "mu=0.5 lateral drift %.4f", mMuH.lateralDriftXZ);
  PASS("dominant-contact friction stable");
}

bool test117_deformableStaticAnchor_motion() {
  printf("\n--- Test 117: Deforming static anchor (no fall-through) ---\n");
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.dt = 1.0f / 60.0f;
  solver.iterations = 16;
  solver.bodyStaticContactSolve = BodyStaticContactSolve::SequentialPerContact;

  const uint32_t sphere =
      solver.addBody({0, 30.0f, 0}, Quat(), {1, 1, 1}, 10.0f, 0.5f);

  DeformableWavyMesh mesh;
  DeformableContactCache cache;
  float minY = 1e9f;

  for (int frame = 0; frame < 240; frame++) {
    mesh.waveTime += 0.01f;
    solver.contacts.clear();
    addSphereDeformableMeshContacts(solver, sphere, mesh, 0.5f);
    cache.restore(mesh, solver);
    solver.step(solver.dt);
    cache.save(mesh, solver);
    minY = std::min(minY, solver.bodies[sphere].position.y);
    if (solver.bodies[sphere].position.y != solver.bodies[sphere].position.y) {
      CHECK(false, "NaN during anchor motion test");
      return false;
    }
  }

  const float floorY = mesh.surfaceY(0, 0) - kDeformRestOffset - 1.0f;
  CHECK(minY > floorY - 5.0f, "sphere fell through wavy mesh (minY=%.2f)", minY);
  PASS("deforming static anchor stable");
}

bool test118_boxOnGround_aggregatedUnchangedBySequentialMode() {
  printf("\n--- Test 118: Single box — default aggregated vs sequential "
         "mode flag ---\n");

  auto runBox = [](BodyStaticContactSolve mode) {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 10;
    solver.dt = 1.0f / 60.0f;
    solver.bodyStaticContactSolve = mode;
    Vec3 halfExt(1, 1, 1);
    uint32_t box = solver.addBody({0, 1, 0}, Quat(), halfExt, 10.0f, 0.5f);
    for (int frame = 0; frame < 120; frame++) {
      solver.contacts.clear();
      addBoxGroundContacts(solver, box, halfExt);
      solver.step(solver.dt);
    }
    return solver.bodies[box].position.y;
  };

  const float yAgg = runBox(BodyStaticContactSolve::Aggregated6x6);
  const float ySeq = runBox(BodyStaticContactSolve::SequentialPerContact);

  printf("  aggregated finalY=%.4f sequential finalY=%.4f\n", yAgg, ySeq);

  CHECK(fabsf(yAgg - 1.0f) < 0.1f, "aggregated box drifted y=%.4f", yAgg);
  CHECK(fabsf(ySeq - 1.0f) < 0.15f, "sequential box drifted y=%.4f", ySeq);
  CHECK(fabsf(yAgg - ySeq) < 0.12f,
        "aggregated vs sequential box mismatch (agg=%.4f seq=%.4f)", yAgg, ySeq);
  PASS("default aggregated path isolated from sequential island semantics");
}

bool test119_kinematicShell_sphereShot() {
  printf("\n--- Test 119: Kinematic shell sphere shot (unified soft contact) "
         "---\n");
  const DeformableSphereShotMetrics m = runDeformableSphereShotShell();
  CHECK(!m.nanDetected, "NaN in kinematic shell run");
  CHECK(m.pass,
        "shell gate failed pen=%.4f (limit %.2f) lateral=%.4f (limit %.1f)",
        m.maxRaycastPen, kDeformPassPen, m.lateralDriftXZ, kDeformPassLateral);
  PASS("kinematic shell meets deformable sphere-shot gate");
}

bool test120_kinematicShell_stressHarness() {
  printf("\n--- Test 120: Kinematic shell stress (grid + periodic shots) ---\n");
  const DeformableShellStressMetrics m = runDeformableStressShell(600, 4);
  CHECK(m.nanEvents == 0, "NaN during shell stress");
  CHECK(m.maxPassThroughShots == 0, "sphere pass-through (%u)",
        m.maxPassThroughShots);
  CHECK(m.maxSunkBoxes == 0, "sunk boxes (%u)", m.maxSunkBoxes);
  CHECK(m.worstMinBoxBottomRel > -0.5f, "box bottom rel %.4f",
        m.worstMinBoxBottomRel);
  PASS("kinematic shell stress harness ok");
}

bool test121_kinematicShell_vs_staticAnchor_sphereShot() {
  printf("\n--- Test 121: Shell vs static-anchor sphere shot comparison ---\n");
  const DeformableSphereShotMetrics anchor = runDeformableSphereShot(
      BodyStaticContactSolve::SequentialPerContact, "anchor", 0.5f);
  const DeformableSphereShotMetrics shell = runDeformableSphereShotShell();
  printf("  anchor pass=%d pen=%.4f lateral=%.4f\n", anchor.pass ? 1 : 0,
         anchor.maxRaycastPen, anchor.lateralDriftXZ);
  printf("  shell  pass=%d pen=%.4f lateral=%.4f\n", shell.pass ? 1 : 0,
         shell.maxRaycastPen, shell.lateralDriftXZ);
  CHECK(shell.pass, "kinematic shell failed gate");
  CHECK(!shell.nanDetected && !anchor.nanDetected, "NaN in comparison");
  CHECK(shell.maxRaycastPen <= anchor.maxRaycastPen + 0.25f,
        "shell pen much worse than anchor (shell=%.4f anchor=%.4f)",
        shell.maxRaycastPen, anchor.maxRaycastPen);
  PASS("kinematic shell parity with static-anchor sphere shot");
}
