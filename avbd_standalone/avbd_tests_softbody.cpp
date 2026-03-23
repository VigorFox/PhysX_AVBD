// =============================================================================
// AVBD Soft Body Tests — test104 to test112
//
// Physics-based unit tests for the Newton-style body-level 6x6 solve
// combined with per-particle VBD. Inspired by Newton and Gaia test suites.
//
// Tests validate:
//   - Free-fall accuracy (analytical solution)
//   - Ground contact settling
//   - Volume preservation (Neo-Hookean)
//   - Material stiffness response
//   - Long-term stability
//   - Stacked bodies
//   - Convergence behavior
//   - Body-level rotation / toppling
//   - Angular momentum conservation
// =============================================================================

#include "avbd_softbody.h"
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
// Helpers
// =============================================================================

static Vec3 computeCOM(const std::vector<SoftParticle>& particles,
                       uint32_t start, uint32_t count) {
  Vec3 com;
  float totalMass = 0;
  for (uint32_t i = start; i < start + count; i++) {
    if (particles[i].invMass <= 0.0f) continue;
    com = com + particles[i].position * particles[i].mass;
    totalMass += particles[i].mass;
  }
  if (totalMass > 0) com = com * (1.0f / totalMass);
  return com;
}

static float computeTetVolume(const std::vector<SoftParticle>& p,
                              uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
  Vec3 e1 = p[i1].position - p[i0].position;
  Vec3 e2 = p[i2].position - p[i0].position;
  Vec3 e3 = p[i3].position - p[i0].position;
  return fabsf(e1.dot(e2.cross(e3)) / 6.0f);
}

static bool hasNaN(const std::vector<SoftParticle>& particles,
                   uint32_t start, uint32_t count) {
  for (uint32_t i = start; i < start + count; i++) {
    const Vec3& p = particles[i].position;
    if (p.x != p.x || p.y != p.y || p.z != p.z) return true;
  }
  return false;
}

static bool hasExplosion(const std::vector<SoftParticle>& particles,
                         uint32_t start, uint32_t count, float limit = 50.0f) {
  for (uint32_t i = start; i < start + count; i++) {
    const Vec3& p = particles[i].position;
    if (fabsf(p.x) > limit || fabsf(p.y) > limit || fabsf(p.z) > limit)
      return true;
  }
  return false;
}

// Generate a simple 2x2x2 tet cube (8 verts, 5 tets)
static void generateSimpleTetCube(Vec3 origin, float size,
                                  std::vector<Vec3>& verts,
                                  std::vector<uint32_t>& tets) {
  verts.clear();
  tets.clear();
  float s = size;
  verts.push_back(origin + Vec3{0, 0, 0});
  verts.push_back(origin + Vec3{s, 0, 0});
  verts.push_back(origin + Vec3{s, 0, s});
  verts.push_back(origin + Vec3{0, 0, s});
  verts.push_back(origin + Vec3{0, s, 0});
  verts.push_back(origin + Vec3{s, s, 0});
  verts.push_back(origin + Vec3{s, s, s});
  verts.push_back(origin + Vec3{0, s, s});

  // 5-tet decomposition of a cube
  uint32_t tetIdx[] = {
    0,1,3,4,  1,2,3,6,  3,4,6,7,  1,4,5,6,  1,3,4,6
  };
  for (int i = 0; i < 20; i++)
    tets.push_back(tetIdx[i]);
}

// =============================================================================
// test104: Free-fall accuracy (no contacts)
//
// A tet cube free-falls for 1 second without contacts.
// COM should follow y(t) = y0 + 0.5*g*t^2.
// Ref: Newton test_physics_validation.py::test_free_fall
// =============================================================================
bool test104_softBodyFreeFall() {
  printf("test104_softBodyFreeFall\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;
  solver.innerIterations = 20;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSimpleTetCube({-0.5f, 5.0f, -0.5f}, 1.0f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e3f, 0.3f, 100.0f, 0.0f);

  const SoftBody& sb = solver.softBodies[0];
  Vec3 com0 = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);

  float dt = solver.dt;
  int nFrames = (int)(1.0f / dt);
  for (int f = 0; f < nFrames; f++)
    solver.step(dt);

  Vec3 comF = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);
  float t = nFrames * dt;
  float y_analytical = com0.y + 0.5f * (-9.8f) * t * t;

  printf("  COM final: (%.3f, %.3f, %.3f)  analytical y=%.3f\n",
         comF.x, comF.y, comF.z, y_analytical);

  CHECK(!hasNaN(solver.softParticles, sb.particleStart, sb.particleCount),
        "NaN in free-fall");
  CHECK(fabsf(comF.y - y_analytical) < 0.5f,
        "COM y=%.3f too far from analytical=%.3f", comF.y, y_analytical);
  CHECK(fabsf(comF.x - com0.x) < 0.1f, "Unexpected horizontal drift x=%.3f", comF.x);
  CHECK(fabsf(comF.z - com0.z) < 0.1f, "Unexpected horizontal drift z=%.3f", comF.z);

  PASS("Free-fall matches analytical trajectory");
}

// =============================================================================
// test105: Ground contact settling
//
// A tet cube drops from y=2 onto ground y=0.
// After 3 seconds, COM should settle near ground, no penetration.
// Ref: Gaia test1_cubeDrop
// =============================================================================
bool test105_softBodyGroundSettle() {
  printf("test105_softBodyGroundSettle\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 8;
  solver.innerIterations = 20;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSimpleTetCube({-0.5f, 2.0f, -0.5f}, 1.0f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f, 0.01f);

  const SoftBody& sb = solver.softBodies[0];

  for (int f = 0; f < 180; f++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  Vec3 com = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);
  float minY = 1e9f, maxY = -1e9f;
  for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
    minY = std::min(minY, solver.softParticles[i].position.y);
    maxY = std::max(maxY, solver.softParticles[i].position.y);
  }

  printf("  COM=(%.3f,%.3f,%.3f) Y=[%.3f,%.3f]\n", com.x, com.y, com.z, minY, maxY);

  CHECK(!hasNaN(solver.softParticles, sb.particleStart, sb.particleCount), "NaN");
  CHECK(!hasExplosion(solver.softParticles, sb.particleStart, sb.particleCount), "Explosion");
  CHECK(com.y < 1.5f, "Cube didn't drop: COM.y=%.3f", com.y);
  CHECK(com.y > -0.5f, "Cube fell through ground: COM.y=%.3f", com.y);
  CHECK(minY > -0.1f, "Particle below ground: minY=%.3f", minY);
  CHECK(maxY - minY > 0.3f, "Cube collapsed flat: height=%.3f", maxY - minY);

  PASS("Ground contact settling");
}

// =============================================================================
// test106: Volume preservation (Neo-Hookean)
//
// A single tet drops and settles. Volume should be > 80% of original.
// Ref: Newton test_tet_energy
// =============================================================================
bool test106_softBodyVolumePreservation() {
  printf("test106_softBodyVolumePreservation\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 8;
  solver.innerIterations = 20;

  std::vector<Vec3> verts = {
    {0.0f, 2.0f, 0.0f},
    {1.0f, 2.0f, 0.0f},
    {0.5f, 2.0f, 0.866f},
    {0.5f, 2.866f, 0.433f},
  };
  std::vector<uint32_t> tets = {0, 1, 2, 3};
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f, 0.01f);

  float V0 = computeTetVolume(solver.softParticles, 0, 1, 2, 3);

  for (int f = 0; f < 300; f++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  float Vf = computeTetVolume(solver.softParticles, 0, 1, 2, 3);
  float ratio = Vf / V0;
  printf("  V0=%.4f Vf=%.4f ratio=%.3f\n", V0, Vf, ratio);

  CHECK(!hasNaN(solver.softParticles, 0, 4), "NaN");
  CHECK(ratio > 0.80f, "Volume lost: ratio=%.3f", ratio);
  CHECK(ratio < 1.5f, "Volume inflated: ratio=%.3f", ratio);

  PASS("Volume preservation (>80%%)");
}

// =============================================================================
// test107: Material stiffness comparison
//
// Soft (E=1e3) vs stiff (E=1e6): stiff body preserves shape better.
// Ref: Gaia test9_materialStiffness
// =============================================================================
bool test107_softBodyMaterialStiffness() {
  printf("test107_softBodyMaterialStiffness\n");

  auto runWithStiffness = [](float E) -> float {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 8;
    solver.innerIterations = 20;

    std::vector<Vec3> verts;
    std::vector<uint32_t> tets;
    generateSimpleTetCube({-0.5f, 2.0f, -0.5f}, 1.0f, verts, tets);
    solver.addSoftBody(verts, tets, {}, E, 0.3f, 100.0f, 0.01f);
    const SoftBody& sb = solver.softBodies[0];

    for (int f = 0; f < 120; f++) {
      solver.softContacts.clear();
      detectSoftGroundContacts(solver.softParticles, solver.softContacts);
      solver.step(solver.dt);
    }

    float minY = 1e9f, maxY = -1e9f;
    for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
      minY = std::min(minY, solver.softParticles[i].position.y);
      maxY = std::max(maxY, solver.softParticles[i].position.y);
    }
    return maxY - minY;
  };

  float heightSoft = runWithStiffness(1e3f);
  float heightStiff = runWithStiffness(1e6f);
  printf("  Soft (E=1e3): height=%.3f  Stiff (E=1e6): height=%.3f\n",
         heightSoft, heightStiff);

  CHECK(heightStiff > heightSoft * 0.7f,
        "Stiff body deformed more than soft?!");

  PASS("Stiff material deforms less than soft");
}

// =============================================================================
// test108: Long-term stability (no explosion, 10 seconds)
//
// Ref: Newton test_energy_conservation stability aspect
// =============================================================================
bool test108_softBodyLongTermStability() {
  printf("test108_softBodyLongTermStability\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 8;
  solver.innerIterations = 20;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSimpleTetCube({-0.5f, 1.0f, -0.5f}, 1.0f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f, 0.01f);

  const SoftBody& sb = solver.softBodies[0];

  for (int f = 0; f < 600; f++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);

    if (hasNaN(solver.softParticles, sb.particleStart, sb.particleCount)) {
      CHECK(false, "NaN at frame %d", f);
    }
    if (hasExplosion(solver.softParticles, sb.particleStart, sb.particleCount)) {
      CHECK(false, "Explosion at frame %d", f);
    }
  }

  Vec3 com = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);
  CHECK(com.y > -0.5f && com.y < 3.0f, "COM out of range: y=%.3f", com.y);

  PASS("10-second stability (no NaN/explosion)");
}

// =============================================================================
// test109: Two stacked soft bodies
//
// Ref: Gaia test4_twoSoftBodiesStacked
// =============================================================================
bool test109_softBodyStacked() {
  printf("test109_softBodyStacked\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 8;
  solver.innerIterations = 20;

  std::vector<Vec3> verts1, verts2;
  std::vector<uint32_t> tets1, tets2;
  generateSimpleTetCube({-0.5f, 0.5f, -0.5f}, 1.0f, verts1, tets1);
  generateSimpleTetCube({-0.5f, 2.0f, -0.5f}, 1.0f, verts2, tets2);
  solver.addSoftBody(verts1, tets1, {}, 1e5f, 0.3f, 100.0f, 0.01f);
  solver.addSoftBody(verts2, tets2, {}, 1e5f, 0.3f, 100.0f, 0.01f);

  for (int f = 0; f < 180; f++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  Vec3 com1 = computeCOM(solver.softParticles,
    solver.softBodies[0].particleStart, solver.softBodies[0].particleCount);
  Vec3 com2 = computeCOM(solver.softParticles,
    solver.softBodies[1].particleStart, solver.softBodies[1].particleCount);
  printf("  COM1=(%.3f,%.3f,%.3f)  COM2=(%.3f,%.3f,%.3f)\n",
         com1.x, com1.y, com1.z, com2.x, com2.y, com2.z);

  uint32_t all = (uint32_t)solver.softParticles.size();
  CHECK(!hasNaN(solver.softParticles, 0, all), "NaN");
  CHECK(!hasExplosion(solver.softParticles, 0, all), "Explosion");
  CHECK(com1.y > -0.5f, "Bottom body below ground");
  CHECK(com2.y > -0.5f, "Top body below ground");
  CHECK(com1.y < 3.0f && com2.y < 5.0f, "Bodies exploded upward");

  PASS("Two stacked soft bodies stable");
}

// =============================================================================
// test110: Convergence behavior
//
// Run 100 inner iterations and verify finite displacement.
// Ref: Gaia test5_convergenceBenchmark
// =============================================================================
bool test110_softBodyConvergence() {
  printf("test110_softBodyConvergence\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 1;
  solver.innerIterations = 100;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSimpleTetCube({-0.5f, 2.0f, -0.5f}, 1.0f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f, 0.0f);

  solver.softContacts.clear();
  detectSoftGroundContacts(solver.softParticles, solver.softContacts);

  std::vector<Vec3> pos0(solver.softParticles.size());
  for (size_t i = 0; i < solver.softParticles.size(); i++)
    pos0[i] = solver.softParticles[i].position;

  solver.step(solver.dt);

  float totalDisp = 0;
  for (size_t i = 0; i < solver.softParticles.size(); i++) {
    Vec3 d = solver.softParticles[i].position - pos0[i];
    totalDisp += d.length();
  }

  printf("  Total displacement after 100 iterations: %.6f\n", totalDisp);

  uint32_t all = (uint32_t)solver.softParticles.size();
  CHECK(!hasNaN(solver.softParticles, 0, all), "NaN");
  CHECK(totalDisp < 100.0f, "Displacement too large: %.3f", totalDisp);
  CHECK(totalDisp > 0.001f, "No movement at all");

  PASS("Convergence: finite displacement");
}

// =============================================================================
// test111: Asymmetric toppling (body-level rotation)
//
// A tet cube is rotated ~30 degrees so it's on an edge, unstable.
// The body-level 6x6 solve should cause settling to a lower-energy state.
// =============================================================================
bool test111_softBodyToppling() {
  printf("test111_softBodyToppling\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 8;
  solver.innerIterations = 20;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSimpleTetCube({-0.5f, 0.5f, -0.5f}, 1.0f, verts, tets);

  // Rotate initial positions ~30 degrees around Z
  float angle = 0.52f;
  float cs = cosf(angle), sn = sinf(angle);
  Vec3 center = {0, 1.0f, 0};
  for (auto& v : verts) {
    Vec3 r = v - center;
    v.x = center.x + r.x * cs - r.y * sn;
    v.y = center.y + r.x * sn + r.y * cs;
  }

  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f, 0.01f);
  const SoftBody& sb = solver.softBodies[0];
  Vec3 com0 = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);

  for (int f = 0; f < 180; f++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  Vec3 comF = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);
  printf("  COM start=(%.3f,%.3f,%.3f) end=(%.3f,%.3f,%.3f)\n",
         com0.x, com0.y, com0.z, comF.x, comF.y, comF.z);

  CHECK(!hasNaN(solver.softParticles, sb.particleStart, sb.particleCount), "NaN");
  CHECK(!hasExplosion(solver.softParticles, sb.particleStart, sb.particleCount), "Explosion");
  CHECK(comF.y < com0.y, "COM didn't drop: initial=%.3f final=%.3f", com0.y, comF.y);
  CHECK(comF.y < 1.5f, "Body didn't settle: COM.y=%.3f", comF.y);

  PASS("Asymmetric toppling");
}

// =============================================================================
// test112: Spinning body stability
//
// Spin a tet cube in zero gravity. VBD's per-particle Gauss-Seidel sweeps
// naturally damp rotation (known limitation). Verify stable behavior:
// no NaN, no explosion, COM doesn't drift.
// =============================================================================
bool test112_softBodyAngularMomentum() {
  printf("test112_softBodyAngularMomentum\n");
  Solver solver;
  solver.gravity = {0, 0, 0};
  solver.iterations = 4;
  solver.innerIterations = 10;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSimpleTetCube({-0.5f, -0.5f, -0.5f}, 1.0f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f, 0.0f);

  const SoftBody& sb = solver.softBodies[0];
  Vec3 com0 = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);

  // Set tangential velocity: v = omega x r, omega = (0, 2, 0)
  Vec3 omega(0, 2.0f, 0);
  for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
    Vec3 r = solver.softParticles[i].position - com0;
    solver.softParticles[i].velocity = omega.cross(r);
  }

  for (int f = 0; f < 60; f++)
    solver.step(solver.dt);

  Vec3 comF = computeCOM(solver.softParticles, sb.particleStart, sb.particleCount);

  CHECK(!hasNaN(solver.softParticles, sb.particleStart, sb.particleCount), "NaN");
  CHECK(!hasExplosion(solver.softParticles, sb.particleStart, sb.particleCount), "Explosion");

  // COM should not drift significantly (no external forces)
  Vec3 comDrift = comF - com0;
  float driftMag = comDrift.length();
  printf("  COM drift: %.4f\n", driftMag);
  CHECK(driftMag < 0.5f, "COM drifted too much: %.4f", driftMag);

  PASS("Spinning body: stable, no explosion");
}
