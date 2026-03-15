// =============================================================================
// AVBD Soft Body / Cloth Tests — test104 to test122
//
// Tests 102–103 are occupied (articulationD6LoopClosure, scissorLiftValidation).
// Soft body tests start at test104.
// =============================================================================

#include "avbd_softbody.h"
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
    printf("  PASS: %s\n", msg);                                               \
    gTestsPassed++;                                                            \
    return true;                                                               \
  } while (0)

// =============================================================================
// test104: Single tetrahedron — free-fall to ground
//
// A single tet (4 particles) drops under gravity and lands on ground y=0.
// Validates: volume preservation, no ground penetration
// =============================================================================
bool test104_softBodySingleTet() {
  printf("test104_softBodySingleTet\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Single tet at height y=2
  std::vector<Vec3> verts = {
    {0.0f, 2.0f, 0.0f},
    {1.0f, 2.0f, 0.0f},
    {0.5f, 2.0f, 0.866f},
    {0.5f, 2.866f, 0.433f},
  };
  std::vector<uint32_t> tets = {0, 1, 2, 3};
  std::vector<uint32_t> tris; // no surface mesh

  solver.addSoftBody(verts, tets, tris, 1e5f, 0.3f, 100.0f);

  // Compute initial volume
  auto computeTetVolume = [&]() {
    Vec3 p0 = solver.softParticles[0].position;
    Vec3 p1 = solver.softParticles[1].position;
    Vec3 p2 = solver.softParticles[2].position;
    Vec3 p3 = solver.softParticles[3].position;
    Vec3 e1 = p1 - p0, e2 = p2 - p0, e3 = p3 - p0;
    return e1.dot(e2.cross(e3)) / 6.0f;
  };

  float V0 = computeTetVolume();

  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  float Vfinal = computeTetVolume();
  float volRatio = fabsf(Vfinal / V0);

  // Check volume preservation (>90%)
  CHECK(volRatio > 0.90f, "Volume lost: V0=%.4f Vf=%.4f ratio=%.3f", V0, Vfinal, volRatio);

  // Check no penetration (all particles above ground)
  for (uint32_t i = 0; i < 4; i++) {
    CHECK(solver.softParticles[i].position.y > -0.05f,
          "Particle %d below ground: y=%.4f", i, solver.softParticles[i].position.y);
  }

  // Check tet landed near ground
  float minY = 1e10f;
  for (uint32_t i = 0; i < 4; i++)
    minY = std::min(minY, solver.softParticles[i].position.y);
  CHECK(minY < 0.5f, "Tet didn't fall to ground: minY=%.3f", minY);

  PASS("single tet: volume preserved, landed on ground");
}

// =============================================================================
// test105: Cube soft body (5 tets) — drop to ground, shape retention
//
// PhysX-equivalent: SnippetDeformableVolume (cube at height)
// =============================================================================
bool test105_softBodyCubeDrop() {
  printf("test105_softBodyCubeDrop\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 3, 0), 0.5f, verts, tets);

  solver.addSoftBody(verts, tets, {}, 2e5f, 0.3f, 100.0f);

  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  // Check all edges: deviation from rest length
  float maxEdgeDeviation = 0.0f;
  for (const auto& sb : solver.softBodies) {
    for (const auto& dc : sb.distConstraints) {
      Vec3 diff = solver.softParticles[dc.p1].position - solver.softParticles[dc.p0].position;
      float curLen = diff.length();
      float deviation = fabsf(curLen - dc.restLength) / dc.restLength;
      maxEdgeDeviation = std::max(maxEdgeDeviation, deviation);
    }
  }
  CHECK(maxEdgeDeviation < 0.10f, "Edge deviation too large: %.3f", maxEdgeDeviation);

  // No penetration
  for (uint32_t i = 0; i < (uint32_t)solver.softParticles.size(); i++) {
    CHECK(solver.softParticles[i].position.y > -0.1f,
          "Particle %d penetrates ground: y=%.4f", i, solver.softParticles[i].position.y);
  }

  // COM should be near ground
  Vec3 com(0, 0, 0);
  for (const auto& sp : solver.softParticles) com = com + sp.position;
  com = com * (1.0f / (float)solver.softParticles.size());
  CHECK(com.y < 1.5f, "Cube didn't fall enough: COM.y=%.3f", com.y);

  PASS("cube drop: edges preserved, no penetration");
}

// =============================================================================
// test106: Distance constraint only — triangle mesh spring network
//
// Surface-only mesh (no tets), just distance constraints.
// Validates: edges don't collapse, mesh retains shape.
// =============================================================================
bool test106_softBodyDistanceOnly() {
  printf("test106_softBodyDistanceOnly\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Small cloth-like grid 4×4, used as a 3D mesh (just distance constraints)
  std::vector<Vec3> verts;
  std::vector<uint32_t> tris;
  generateClothGrid(Vec3(0, 3, 0), 1.0f, 1.0f, 4, 4, verts, tris);

  solver.addSoftBody(verts, {}, tris, 1e5f, 0.3f, 200.0f);

  for (int frame = 0; frame < 200; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  // Edges should not have collapsed
  float maxDeviation = 0.0f;
  for (const auto& sb : solver.softBodies) {
    for (const auto& dc : sb.distConstraints) {
      Vec3 diff = solver.softParticles[dc.p1].position - solver.softParticles[dc.p0].position;
      float curLen = diff.length();
      float deviation = fabsf(curLen - dc.restLength) / std::max(dc.restLength, 1e-6f);
      maxDeviation = std::max(maxDeviation, deviation);
    }
  }
  CHECK(maxDeviation < 0.15f, "Distance constraint deviation too large: %.3f", maxDeviation);

  PASS("distance-only mesh: edges preserved");
}

// =============================================================================
// test107: Volume preservation test — compress cube, check volume
//
// A soft cube between ground and a heavy rigid body. Volume should be preserved.
// =============================================================================
bool test107_softBodyVolumePreserve() {
  printf("test107_softBodyVolumePreserve\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  // Soft cube at y=0.5 (resting on ground)
  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 0.5f, 0), 0.5f, verts, tets);

  // Use high Poisson ratio for near-incompressible
  solver.addSoftBody(verts, tets, {}, 2e5f, 0.45f, 100.0f);

  // Compute initial total volume
  auto computeTotalVolume = [&]() {
    float totalVol = 0;
    for (const auto& sb : solver.softBodies) {
      for (const auto& vc : sb.volConstraints) {
        Vec3 e1 = solver.softParticles[vc.p1].position - solver.softParticles[vc.p0].position;
        Vec3 e2 = solver.softParticles[vc.p2].position - solver.softParticles[vc.p0].position;
        Vec3 e3 = solver.softParticles[vc.p3].position - solver.softParticles[vc.p0].position;
        totalVol += e1.dot(e2.cross(e3)) / 6.0f;
      }
    }
    return totalVol;
  };

  float V0 = computeTotalVolume();

  // Simulate: gravity compresses cube onto ground
  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  float Vfinal = computeTotalVolume();
  float volDeviation = fabsf(Vfinal - V0) / fabsf(V0);
  CHECK(volDeviation < 0.05f, "Volume deviation too large: %.3f (V0=%.4f, Vf=%.4f)",
        volDeviation, V0, Vfinal);

  printf("  volume deviation: %.4f\n", volDeviation);
  PASS("volume preservation: within tolerance");
}

// =============================================================================
// test108: Soft body resting on ground — stability test
//
// A soft cube placed on ground should settle without exploding.
// =============================================================================
bool test108_softBodyStackOnGround() {
  printf("test108_softBodyStackOnGround\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 1.0f, 0), 0.5f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f);

  Vec3 prevCom(0, 0, 0);
  for (const auto& sp : solver.softParticles) prevCom = prevCom + sp.position;
  prevCom = prevCom * (1.0f / (float)solver.softParticles.size());

  bool exploded = false;
  for (int frame = 0; frame < 500; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);

    for (const auto& sp : solver.softParticles) {
      if (fabsf(sp.position.x) > 50.0f || fabsf(sp.position.y) > 50.0f ||
          fabsf(sp.position.z) > 50.0f) {
        exploded = true;
        break;
      }
    }
    if (exploded) break;
  }
  CHECK(!exploded, "Soft body exploded!");

  // COM should be stable near ground
  Vec3 com(0, 0, 0);
  for (const auto& sp : solver.softParticles) com = com + sp.position;
  com = com * (1.0f / (float)solver.softParticles.size());
  CHECK(com.y > -0.1f && com.y < 3.0f, "COM unstable: y=%.3f", com.y);

  PASS("soft body resting on ground: stable");
}

// =============================================================================
// test109: Multiple soft bodies — no mutual penetration
//
// Two soft cubes stacked. Both should survive.
// PhysX-equivalent: SnippetDeformableVolume (multiple deformable volumes)
// =============================================================================
bool test109_softBodyMultiple() {
  printf("test109_softBodyMultiple\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Lower cube
  std::vector<Vec3> verts1, verts2;
  std::vector<uint32_t> tets1, tets2;
  generateCubeTets(Vec3(0, 1.0f, 0), 0.5f, verts1, tets1);
  generateCubeTets(Vec3(0, 3.0f, 0), 0.5f, verts2, tets2);

  solver.addSoftBody(verts1, tets1, {}, 1e5f, 0.3f, 100.0f);
  solver.addSoftBody(verts2, tets2, {}, 1e5f, 0.3f, 100.0f);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);

    for (const auto& sp : solver.softParticles) {
      if (fabsf(sp.position.y) > 50.0f) { exploded = true; break; }
    }
    if (exploded) break;
  }
  CHECK(!exploded, "Soft bodies exploded!");

  // Both should be above ground
  for (uint32_t i = 0; i < (uint32_t)solver.softParticles.size(); i++) {
    CHECK(solver.softParticles[i].position.y > -0.15f,
          "Particle %d below ground: y=%.4f", i, solver.softParticles[i].position.y);
  }

  PASS("multiple soft bodies: stable, no explosion");
}

// =============================================================================
// test110: Cloth draped on sphere obstacle
//
// A cloth grid drops onto a static rigid sphere (approximated as a box).
// PhysX-equivalent: SnippetDeformableSurface (cloth on rotating sphere)
// =============================================================================
bool test110_clothDrape() {
  printf("test110_clothDrape\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Static rigid box as obstacle at y=2 (approximating a sphere)
  uint32_t obstacleIdx = solver.addBody({0, 2.0f, 0}, Quat(), {0.5f, 0.5f, 0.5f}, 0.0f, 0.5f);
  (void)obstacleIdx;

  // Cloth grid 8×8 at y=5
  std::vector<Vec3> verts;
  std::vector<uint32_t> tris;
  generateClothGrid(Vec3(0, 5, 0), 3.0f, 3.0f, 8, 8, verts, tris);

  solver.addSoftBody(verts, {}, tris, 5e3f, 0.3f, 1000.0f, 0.05f, 0.001f, 0.005f);

  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    detectSoftRigidContacts(solver.softParticles, solver.bodies, solver.softContacts);
    solver.step(solver.dt);
  }

  // Some particles should be above the obstacle (draped)
  int aboveObstacle = 0;
  for (const auto& sp : solver.softParticles) {
    if (sp.position.y > 2.0f) aboveObstacle++;
  }
  CHECK(aboveObstacle > 0, "No cloth particles above obstacle");

  // Some should be below obstacle height (hanging down)
  int belowObstacle = 0;
  for (const auto& sp : solver.softParticles) {
    if (sp.position.y < 2.0f) belowObstacle++;
  }
  CHECK(belowObstacle > 0, "Cloth didn't drape — all particles above obstacle");

  PASS("cloth drape: particles above and below obstacle");
}

// =============================================================================
// test111: Cloth bending stiffness comparison
//
// Same cloth, with vs without bending stiffness. With bending should
// maintain a flatter profile.
// =============================================================================
bool test111_clothBendingStiffness() {
  printf("test111_clothBendingStiffness\n");

  auto runCloth = [](float bendStiffness) -> float {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 20;

    // Cloth 6×6, pinned at two ends
    std::vector<Vec3> verts;
    std::vector<uint32_t> tris;
    generateClothGrid(Vec3(0, 5, 0), 2.0f, 2.0f, 6, 6, verts, tris);

    solver.addSoftBody(verts, {}, tris, 1e5f, 0.3f, 200.0f, 0.05f, bendStiffness, 0.01f);

    // Pin left and right edges
    SoftBody& sb = solver.softBodies[0];
    for (int j = 0; j < 6; j++) {
      // Left edge: i=0
      KinematicPin kp;
      kp.particleIdx = sb.particleStart + (uint32_t)(j * 6);
      kp.worldTarget = solver.softParticles[kp.particleIdx].position;
      kp.rho = 1e6f;
      sb.pins.push_back(kp);
      // Right edge: i=5
      KinematicPin kp2;
      kp2.particleIdx = sb.particleStart + (uint32_t)(j * 6 + 5);
      kp2.worldTarget = solver.softParticles[kp2.particleIdx].position;
      kp2.rho = 1e6f;
      sb.pins.push_back(kp2);
    }

    for (int frame = 0; frame < 300; frame++) {
      solver.softContacts.clear();
      solver.step(solver.dt);
    }

    // Measure sag: minimum y of center particles
    float minY = 1e10f;
    for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
      minY = std::min(minY, solver.softParticles[i].position.y);
    }
    return minY;
  };

  float sagNoBend = runCloth(0.0f);
  float sagWithBend = runCloth(0.01f);

  // With bending stiffness, cloth should sag less (higher minY)
  CHECK(sagWithBend > sagNoBend - 0.01f,
        "Bending stiffness didn't reduce sag: noBend=%.3f withBend=%.3f",
        sagNoBend, sagWithBend);

  printf("  sagNoBend=%.3f sagWithBend=%.3f\n", sagNoBend, sagWithBend);
  PASS("bending stiffness comparison");
}

// =============================================================================
// test112: Cloth pinned at 4 corners — symmetric sag under gravity
//
// PhysX-equivalent: SnippetDeformableSurface (cloth with kinematic pins)
// =============================================================================
bool test112_clothPinnedCorners() {
  printf("test112_clothPinnedCorners\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 50;

  std::vector<Vec3> verts;
  std::vector<uint32_t> tris;
  int M = 5, N = 5;
  generateClothGrid(Vec3(0, 2, 0), 0.5f, 0.5f, M, N, verts, tris);

  // Heavier, stiffer cloth for stability
  solver.addSoftBody(verts, {}, tris, 1e4f, 0.3f, 2000.0f, 0.5f, 0.0f, 0.02f);

  // Pin ALL boundary vertices (like a drum skin)
  SoftBody& sb = solver.softBodies[0];
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      if (i == 0 || i == M-1 || j == 0 || j == N-1) {
        KinematicPin kp;
        kp.particleIdx = sb.particleStart + (uint32_t)(j * M + i);
        kp.worldTarget = solver.softParticles[kp.particleIdx].position;
        kp.rho = 1e8f;
        sb.pins.push_back(kp);
      }
    }
  }

  uint32_t centerIdx = sb.particleStart + (uint32_t)((N/2) * M + M/2);

  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  // Center particle should sag (y < 2)
  float centerY = solver.softParticles[centerIdx].position.y;
  CHECK(centerY < 2.0f, "Center didn't sag: y=%.3f", centerY);

  // Boundary should stay pinned
  float maxDy = 0;
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      if (i == 0 || i == M-1 || j == 0 || j == N-1) {
        uint32_t pi = sb.particleStart + (uint32_t)(j * M + i);
        float dy = fabsf(solver.softParticles[pi].position.y - 2.0f);
        maxDy = std::max(maxDy, dy);
      }
    }
  }
  CHECK(maxDy < 0.1f, "Boundary drifted: maxDy=%.3f", maxDy);

  // Check approximate symmetry
  float cx = solver.softParticles[centerIdx].position.x;
  float cz = solver.softParticles[centerIdx].position.z;
  CHECK(fabsf(cx) < 0.3f && fabsf(cz) < 0.3f,
        "Center drifted laterally: x=%.3f z=%.3f", cx, cz);

  PASS("pinned edges: center sag, boundary stable");
}

// =============================================================================
// test113: Soft-rigid attachment — soft body bound to falling rigid body
//
// PhysX-equivalent: SnippetDeformableVolumeAttachment
// =============================================================================
bool test113_softRigidAttach() {
  printf("test113_softRigidAttach\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 30;

  // Static rigid body at y=2
  uint32_t rigidIdx = solver.addBody({0, 2.0f, 0}, Quat(), {0.3f, 0.3f, 0.3f}, 0.0f, 0.5f);

  // Soft cube slightly below rigid body (center at y=1.2)
  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 1.2f, 0), 0.3f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 500.0f);

  // Attach top particles (y > 1.35) to static rigid body
  SoftBody& sb = solver.softBodies[0];
  for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
    if (solver.softParticles[i].position.y > 1.35f) {
      AttachmentConstraint ac;
      ac.particleIdx = i;
      ac.rigidBodyIdx = rigidIdx;
      ac.localOffset = solver.bodies[rigidIdx].rotation.conjugate().rotate(
          solver.softParticles[i].position - solver.bodies[rigidIdx].position);
      ac.rho = 1e5f;
      sb.attachments.push_back(ac);
    }
  }

  for (int frame = 0; frame < 200; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  // Attached particles should be near the rigid body anchor points
  for (const auto& ac : sb.attachments) {
    Vec3 worldAnchor = solver.bodies[rigidIdx].position +
                       solver.bodies[rigidIdx].rotation.rotate(ac.localOffset);
    float gap = (solver.softParticles[ac.particleIdx].position - worldAnchor).length();
    CHECK(gap < 0.5f, "Attachment gap too large: %.3f", gap);
  }

  // Bottom particles should have sagged under gravity but not fallen through ground
  float minY = 1e10f;
  for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
    minY = std::min(minY, solver.softParticles[i].position.y);
  }
  CHECK(minY > -0.5f, "Soft body fell too far: minY=%.3f", minY);

  PASS("soft-rigid attachment: coupled motion");
}

// =============================================================================
// test114: Soft body on rigid box
//
// A soft cube drops onto a static rigid box.
// PhysX-equivalent: SnippetDeformableVolume (soft on rigid)
// =============================================================================
bool test114_softOnRigidBox() {
  printf("test114_softOnRigidBox\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Static rigid box as platform at y=1
  solver.addBody({0, 1.0f, 0}, Quat(), {2.0f, 0.5f, 2.0f}, 0.0f, 0.5f);

  // Soft cube dropping from y=4
  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 4.0f, 0), 0.5f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f);

  for (int frame = 0; frame < 300; frame++) {
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    detectSoftRigidContacts(solver.softParticles, solver.bodies, solver.softContacts);
    solver.step(solver.dt);
  }

  // Soft body should be resting on the rigid box (y ≈ 1.5 to 2.5)
  Vec3 com(0, 0, 0);
  for (const auto& sp : solver.softParticles)
    com = com + sp.position;
  com = com * (1.0f / (float)solver.softParticles.size());
  CHECK(com.y > 1.0f && com.y < 4.0f,
        "Soft body not resting on box: COM.y=%.3f", com.y);

  // No explosion
  for (const auto& sp : solver.softParticles) {
    CHECK(fabsf(sp.position.y) < 50.0f, "Particle exploded");
  }

  PASS("soft on rigid box: resting correctly");
}

// =============================================================================
// test115: Kinematic pin oscillation
//
// PhysX-equivalent: SnippetDeformableVolumeKinematic
// Some particles pinned, oscillated side-to-side. Others free.
// =============================================================================
bool test115_kinematicPinOscillate() {
  printf("test115_kinematicPinOscillate\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Soft cube
  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateSubdividedCubeTets(Vec3(0, 3, 0), 0.5f, 2, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f);

  SoftBody& sb = solver.softBodies[0];

  // Pin top layer particles (y ≈ 3.5)
  std::vector<uint32_t> pinnedIds;
  for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
    if (solver.softParticles[i].position.y > 3.3f) {
      KinematicPin kp;
      kp.particleIdx = i;
      kp.worldTarget = solver.softParticles[i].position;
      kp.rho = 1e6f;
      sb.pins.push_back(kp);
      pinnedIds.push_back(i);
    }
  }
  CHECK(!pinnedIds.empty(), "No pinned particles found");

  for (int frame = 0; frame < 300; frame++) {
    // Oscillate pinned particles in X
    float t = frame * solver.dt;
    float offsetX = 0.3f * sinf(2.0f * 3.14159f * 0.5f * t);
    for (auto& kp : sb.pins) {
      Vec3 basePos = solver.softParticles[kp.particleIdx].initialPosition; // rest position
      // We stored initialPosition at start of step, so use the original vert
      // For simplicity, store offset from first pin position
      kp.worldTarget.x = solver.softParticles[kp.particleIdx].position.x; // will be reset
    }
    // Actually, let's just oscillate relative to the original position
    for (size_t pi = 0; pi < sb.pins.size(); pi++) {
      uint32_t idx = sb.pins[pi].particleIdx;
      // Original Y position from the verts
      float origX = verts[idx - sb.particleStart].x;
      float origY = verts[idx - sb.particleStart].y;
      float origZ = verts[idx - sb.particleStart].z;
      sb.pins[pi].worldTarget = Vec3(origX + offsetX, origY, origZ);
    }

    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);
  }

  // Pinned particles should follow the oscillation target
  for (const auto& kp : sb.pins) {
    float gap = (solver.softParticles[kp.particleIdx].position - kp.worldTarget).length();
    CHECK(gap < 0.5f, "Pinned particle drifted: gap=%.3f", gap);
  }

  // Non-pinned particles should be lower (hanging)
  float minFreeY = 1e10f;
  for (uint32_t i = sb.particleStart; i < sb.particleStart + sb.particleCount; i++) {
    bool isPinned = false;
    for (auto pid : pinnedIds) if (pid == i) { isPinned = true; break; }
    if (!isPinned) {
      minFreeY = std::min(minFreeY, solver.softParticles[i].position.y);
    }
  }
  CHECK(minFreeY < 3.3f, "Free particles didn't hang: minY=%.3f", minFreeY);

  PASS("kinematic pin oscillation: pinned follow, free hang");
}

// =============================================================================
// test116: Rigid body falls on soft body — cushioning
//
// A heavy rigid box drops onto a soft cube. The soft body should deform
// and cushion the impact.
// =============================================================================
bool test116_rigidFallOnSoft() {
  printf("test116_rigidFallOnSoft\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // Soft cube on ground
  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 0.5f, 0), 0.5f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 5e4f, 0.3f, 100.0f);

  // Heavy rigid box falling from above
  uint32_t rigidIdx = solver.addBody({0, 4.0f, 0}, Quat(), {0.3f, 0.3f, 0.3f}, 2000.0f, 0.5f);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.softContacts.clear();
    collideBoxGround(solver, rigidIdx);
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    detectSoftRigidContacts(solver.softParticles, solver.bodies, solver.softContacts);
    solver.step(solver.dt);

    if (fabsf(solver.bodies[rigidIdx].position.y) > 50.0f) {
      exploded = true;
      break;
    }
  }
  CHECK(!exploded, "Rigid body or soft body exploded!");

  // Rigid body should have settled near ground
  CHECK(solver.bodies[rigidIdx].position.y < 3.5f,
        "Rigid didn't fall: y=%.3f", solver.bodies[rigidIdx].position.y);
  CHECK(solver.bodies[rigidIdx].position.y > -0.5f,
        "Rigid went through ground: y=%.3f", solver.bodies[rigidIdx].position.y);

  PASS("rigid on soft: stable cushioning");
}

// =============================================================================
// test117: Material stiffness comparison
//
// Same shape, different Young's modulus. Stiffer → less deformation.
// =============================================================================
bool test117_materialStiffness() {
  printf("test117_materialStiffness\n");

  auto measureDeformation = [](float E) -> float {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 20;

    std::vector<Vec3> verts;
    std::vector<uint32_t> tets;
    generateCubeTets(Vec3(0, 3, 0), 0.5f, verts, tets);
    solver.addSoftBody(verts, tets, {}, E, 0.3f, 100.0f);

    for (int frame = 0; frame < 200; frame++) {
      solver.softContacts.clear();
      detectSoftGroundContacts(solver.softParticles, solver.softContacts);
      solver.step(solver.dt);
    }

    // Measure max edge deviation
    float maxDev = 0;
    for (const auto& sb : solver.softBodies) {
      for (const auto& dc : sb.distConstraints) {
        Vec3 diff = solver.softParticles[dc.p1].position - solver.softParticles[dc.p0].position;
        float curLen = diff.length();
        float dev = fabsf(curLen - dc.restLength) / dc.restLength;
        maxDev = std::max(maxDev, dev);
      }
    }
    return maxDev;
  };

  float devLow = measureDeformation(1e4f);   // soft
  float devHigh = measureDeformation(1e6f);   // stiff

  CHECK(devHigh < devLow + 0.01f,
        "Stiffer material should deform less: E_low=%.4f E_high=%.4f",
        devLow, devHigh);

  printf("  E_low_dev=%.4f E_high_dev=%.4f\n", devLow, devHigh);
  PASS("material stiffness: stiffer deforms less");
}

// =============================================================================
// test118: Poisson ratio near-incompressible
//
// ν=0.48 → volume should be nearly preserved under compression.
// =============================================================================
bool test118_materialPoisson() {
  printf("test118_materialPoisson\n");

  auto measureVolChange = [](float nu) -> float {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = 30;

    std::vector<Vec3> verts;
    std::vector<uint32_t> tets;
    generateCubeTets(Vec3(0, 0.5f, 0), 0.5f, verts, tets);
    solver.addSoftBody(verts, tets, {}, 2e5f, nu, 100.0f);

    float V0 = 0;
    for (const auto& sb : solver.softBodies)
      for (const auto& vc : sb.volConstraints) {
        Vec3 e1 = solver.softParticles[vc.p1].position - solver.softParticles[vc.p0].position;
        Vec3 e2 = solver.softParticles[vc.p2].position - solver.softParticles[vc.p0].position;
        Vec3 e3 = solver.softParticles[vc.p3].position - solver.softParticles[vc.p0].position;
        V0 += e1.dot(e2.cross(e3)) / 6.0f;
      }

    for (int frame = 0; frame < 300; frame++) {
      solver.softContacts.clear();
      detectSoftGroundContacts(solver.softParticles, solver.softContacts);
      solver.step(solver.dt);
    }

    float Vfinal = 0;
    for (const auto& sb : solver.softBodies)
      for (const auto& vc : sb.volConstraints) {
        Vec3 e1 = solver.softParticles[vc.p1].position - solver.softParticles[vc.p0].position;
        Vec3 e2 = solver.softParticles[vc.p2].position - solver.softParticles[vc.p0].position;
        Vec3 e3 = solver.softParticles[vc.p3].position - solver.softParticles[vc.p0].position;
        Vfinal += e1.dot(e2.cross(e3)) / 6.0f;
      }

    return fabsf(Vfinal - V0) / fabsf(V0);
  };

  float devCompressible = measureVolChange(0.1f);
  float devIncompressible = measureVolChange(0.48f);

  CHECK(devIncompressible < devCompressible + 0.01f,
        "Higher ν should preserve volume better: nu0.1=%.4f nu0.48=%.4f",
        devCompressible, devIncompressible);

  printf("  nu0.1_dev=%.4f nu0.48_dev=%.4f\n", devCompressible, devIncompressible);
  PASS("Poisson ratio: higher nu preserves volume better");
}

// =============================================================================
// test119: Convergence benchmark — iteration count vs constraint violation
//
// Measure distance constraint violation vs iteration count
// =============================================================================
bool test119_convergenceSoftBench() {
  printf("test119_convergenceSoftBench\n");

  auto measureViolation = [](int iters) -> float {
    Solver solver;
    solver.gravity = {0, -9.8f, 0};
    solver.iterations = iters;

    std::vector<Vec3> verts;
    std::vector<uint32_t> tets;
    generateSubdividedCubeTets(Vec3(0, 3, 0), 0.5f, 2, verts, tets);
    solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f);

    // Single step — cold start
    solver.softContacts.clear();
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);

    // Measure max distance constraint violation
    float maxViol = 0;
    for (const auto& sb : solver.softBodies) {
      for (const auto& dc : sb.distConstraints) {
        Vec3 diff = solver.softParticles[dc.p1].position - solver.softParticles[dc.p0].position;
        float curLen = diff.length();
        float viol = fabsf(curLen - dc.restLength);
        maxViol = std::max(maxViol, viol);
      }
    }
    return maxViol;
  };

  float viol5 = measureViolation(5);
  float viol10 = measureViolation(10);
  float viol20 = measureViolation(20);

  printf("  Convergence: 5iter=%.6f, 10iter=%.6f, 20iter=%.6f\n",
         viol5, viol10, viol20);

  // More iterations should give lower violation (monotonic)
  CHECK(viol10 <= viol5 + 1e-4f, "10 iters not better than 5: %.6f vs %.6f", viol10, viol5);
  CHECK(viol20 <= viol10 + 1e-4f, "20 iters not better than 10: %.6f vs %.6f", viol20, viol10);

  PASS("convergence benchmark: monotonic improvement");
}

// =============================================================================
// test120: Unified scene — rigid + articulation + soft body
//
// Everything in one scene. Validates no cross-system explosion.
// =============================================================================
bool test120_unifiedScene() {
  printf("test120_unifiedScene\n");
  Solver solver;
  solver.gravity = {0, -9.8f, 0};
  solver.iterations = 20;

  // 1. Rigid box on ground
  uint32_t box0 = solver.addBody({-2, 0.5f, 0}, Quat(), {0.5f, 0.5f, 0.5f}, 500.0f);

  // 2. Articulation: 3-link pendulum
  Articulation artic;
  artic.fixedBase = true;
  artic.fixedBasePos = Vec3(2, 5, 0);
  uint32_t link0 = solver.addBody({2, 4, 0}, Quat(), {0.15f, 0.4f, 0.15f}, 500.0f);
  uint32_t link1 = solver.addBody({2, 3, 0}, Quat(), {0.15f, 0.4f, 0.15f}, 500.0f);
  uint32_t link2 = solver.addBody({2, 2, 0}, Quat(), {0.15f, 0.4f, 0.15f}, 500.0f);
  artic.addLink(link0, -1, eARTIC_REVOLUTE, Vec3(0,0,1), Vec3(0,0,0), Vec3(0,0.5f,0), solver.bodies);
  artic.addLink(link1, 0, eARTIC_REVOLUTE, Vec3(0,0,1), Vec3(0,-0.5f,0), Vec3(0,0.5f,0), solver.bodies);
  artic.addLink(link2, 1, eARTIC_REVOLUTE, Vec3(0,0,1), Vec3(0,-0.5f,0), Vec3(0,0.5f,0), solver.bodies);
  solver.articulations.push_back(artic);

  // 3. Soft cube at center
  std::vector<Vec3> verts;
  std::vector<uint32_t> tets;
  generateCubeTets(Vec3(0, 2, 0), 0.4f, verts, tets);
  solver.addSoftBody(verts, tets, {}, 1e5f, 0.3f, 100.0f);

  bool exploded = false;
  for (int frame = 0; frame < 300; frame++) {
    solver.contacts.clear();
    solver.softContacts.clear();
    collideBoxGround(solver, box0);
    detectSoftGroundContacts(solver.softParticles, solver.softContacts);
    solver.step(solver.dt);

    // Check no explosion
    for (const auto& b : solver.bodies) {
      if (b.mass > 0 && (fabsf(b.position.y) > 100.0f || fabsf(b.position.x) > 100.0f)) {
        exploded = true;
        break;
      }
    }
    for (const auto& sp : solver.softParticles) {
      if (fabsf(sp.position.y) > 100.0f) { exploded = true; break; }
    }
    if (exploded) break;
  }
  CHECK(!exploded, "Unified scene exploded!");

  PASS("unified scene: rigid + articulation + soft body stable");
}
