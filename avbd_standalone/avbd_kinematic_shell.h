#pragma once
// Kinematic soft-particle shell for deformable wavy mesh (AVBD unified contact path).
#include "avbd_deformable_scene.h"
#include "avbd_solver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace AvbdRef {

/** Phase 0 complete: direct soft-contact path only (no body-static synthesize). */
static constexpr bool kShellSynthesizeBodyStaticContacts = false;

inline uint32_t shellParticleIndex(uint32_t shellStart, int gridA, int gridB) {
  return shellStart + uint32_t(gridA * kDeformGridSize + gridB);
}

inline void synthesizeKinematicShellToContacts(Solver &solver) {
  solver.contacts.clear();
  for (const auto &sc : solver.softContacts) {
    if (sc.rigidBodyIdx == UINT32_MAX ||
        sc.rigidBodyIdx >= solver.bodies.size())
      continue;
    Contact c;
    c.bodyA = sc.rigidBodyIdx;
    c.bodyB = UINT32_MAX;
    c.normal = sc.normal;
    c.rA = sc.rigidLocalPoint;
    c.rB = sc.surfacePoint;
    c.depth = sc.depth;
    c.friction = sc.friction;
    c.staticPrevWorldPoint = sc.surfacePointPrev;
    for (int i = 0; i < 3; ++i) {
      c.lambda[i] = (i == 0) ? sc.lambda : 0.0f;
      c.penalty[i] = (i == 0) ? sc.k : PENALTY_MIN;
      c.fmin[i] = 0.0f;
      c.fmax[i] = 0.0f;
    }
    c.fmin[0] = -1e30f;
    c.fmax[0] = 0.0f;
    solver.contacts.push_back(c);
  }
}

inline int findNearestShellVertex(const DeformableWavyMesh &mesh,
                                  const Vec3 &worldPoint) {
  int bestA = 0;
  int bestB = 0;
  float bestD2 = 1e30f;
  for (int a = 0; a < kDeformGridSize; ++a) {
    for (int b = 0; b < kDeformGridSize; ++b) {
      const Vec3 d = mesh.vertex(a, b) - worldPoint;
      const float d2 = d.dot(d);
      if (d2 < bestD2) {
        bestD2 = d2;
        bestA = a;
        bestB = b;
      }
    }
  }
  return bestA * kDeformGridSize + bestB;
}

inline uint32_t buildWavyMeshKinematicShell(Solver &solver,
                                            const DeformableWavyMesh &mesh) {
  std::vector<Vec3> verts;
  verts.reserve(kDeformGridSize * kDeformGridSize);
  for (int a = 0; a < kDeformGridSize; ++a) {
    for (int b = 0; b < kDeformGridSize; ++b)
      verts.push_back(mesh.vertex(a, b));
  }
  return solver.addKinematicShell(verts);
}

inline void syncKinematicShellFromMesh(Solver &solver, uint32_t shellStart,
                                       const DeformableWavyMesh &mesh,
                                       float dt) {
  int idx = 0;
  for (int a = 0; a < kDeformGridSize; ++a) {
    for (int b = 0; b < kDeformGridSize; ++b, ++idx) {
      SoftParticle &sp = solver.softParticles[shellStart + uint32_t(idx)];
      const Vec3 newPos = mesh.vertex(a, b);
      if (dt > 0.0f)
        sp.velocity = (newPos - sp.position) * (1.0f / dt);
      sp.initialPosition = sp.position;
      sp.position = newPos;
      sp.predictedPosition = newPos;
      sp.outerPosition = newPos;
      sp.prevVelocity = sp.velocity;
    }
  }
}

struct ShellContactCache {
  struct Key {
    int32_t ia, ib;
    int32_t bodyIdx;
    bool operator<(const Key &o) const {
      if (bodyIdx != o.bodyIdx)
        return bodyIdx < o.bodyIdx;
      if (ia != o.ia)
        return ia < o.ia;
      return ib < o.ib;
    }
  };
  struct Entry {
    float lambda;
    float k;
    float lambdaTangent[2];
    float penTangent[2];
    Vec3 surfacePointPrev;
  };
  std::map<Key, Entry> data;

  static Key makeKey(uint32_t bodyIdx, int gridFlat) {
    return {gridFlat / kDeformGridSize, gridFlat % kDeformGridSize,
            (int32_t)bodyIdx};
  }

  void save(const Solver &solver, uint32_t shellStart) {
    data.clear();
    for (const auto &sc : solver.softContacts) {
      if (sc.rigidBodyIdx == UINT32_MAX)
        continue;
      if (sc.particleIdx < shellStart)
        continue;
      const int flat = int(sc.particleIdx - shellStart);
      Entry e;
      e.lambda = sc.lambda;
      e.k = sc.k;
      e.lambdaTangent[0] = sc.lambdaTangent[0];
      e.lambdaTangent[1] = sc.lambdaTangent[1];
      e.penTangent[0] = sc.penTangent[0];
      e.penTangent[1] = sc.penTangent[1];
      e.surfacePointPrev = sc.surfacePoint;
      data[makeKey(sc.rigidBodyIdx, flat)] = e;
    }
  }

  void restore(Solver &solver, uint32_t shellStart) {
    for (auto &sc : solver.softContacts) {
      if (sc.rigidBodyIdx == UINT32_MAX)
        continue;
      if (sc.particleIdx < shellStart)
        continue;
      const int flat = int(sc.particleIdx - shellStart);
      auto it = data.find(makeKey(sc.rigidBodyIdx, flat));
      if (it == data.end())
        continue;
      sc.lambda = it->second.lambda;
      sc.k = it->second.k;
      sc.lambdaTangent[0] = it->second.lambdaTangent[0];
      sc.lambdaTangent[1] = it->second.lambdaTangent[1];
      sc.penTangent[0] = it->second.penTangent[0];
      sc.penTangent[1] = it->second.penTangent[1];
      sc.surfacePointPrev = it->second.surfacePointPrev;
    }
  }
};

inline void addSphereShellContacts(Solver &solver, uint32_t sphereIdx,
                                   uint32_t shellStart,
                                   const DeformableWavyMesh &mesh,
                                   float friction) {
  Body &sph = solver.bodies[sphereIdx];
  const float R = sph.halfExtent.x;
  const float sampleStep = kDeformGridStep * 0.35f;

  for (int di = -4; di <= 4; ++di) {
    for (int dj = -4; dj <= 4; ++dj) {
      const float sx = sph.position.x + float(di) * sampleStep;
      const float sz = sph.position.z + float(dj) * sampleStep;
      const float sy = mesh.surfaceY(sx, sz);
      const Vec3 v(sx, sy, sz);

      Vec3 n = sph.position - v;
      float dist = n.length();
      if (dist < 1e-4f)
        n = Vec3(0, 1, 0);
      else
        n = n * (1.0f / dist);

      const Vec3 worldA = sph.position - n * R;
      const float sep = (worldA - v).dot(n);
      if (sep > kDeformRestOffset + 0.12f)
        continue;

      const Quat qInv = sph.rotation.conjugate();
      const Vec3 rA = qInv.rotate(worldA - sph.position);
      const int flat = findNearestShellVertex(mesh, v);
      const int ga = flat / kDeformGridSize;
      const int gb = flat % kDeformGridSize;
      n = mesh.normalAt(ga, gb);
      if (n.dot(sph.position - v) < 0.0f)
        n = n * -1.0f;
      const uint32_t pi = shellStart + uint32_t(flat);

      SoftContact sc;
      sc.particleIdx = pi;
      sc.rigidBodyIdx = sphereIdx;
      sc.normal = n;
      sc.depth = kDeformRestOffset;
      sc.friction = friction;
      sc.surfacePoint = v;
      sc.surfacePointPrev = v;
      sc.rigidLocalPoint = rA;
      sc.k = 1e4f;
      sc.ke = 1e6f;
      if (fabsf(n.y) > 0.9f)
        sc.tangent1 = n.cross(Vec3(1, 0, 0)).normalized();
      else
        sc.tangent1 = n.cross(Vec3(0, 1, 0)).normalized();
      sc.tangent2 = n.cross(sc.tangent1);
      solver.softContacts.push_back(sc);
    }
  }
}

inline void addBoxShellContacts(Solver &solver, uint32_t boxIdx, Vec3 halfExt,
                                uint32_t shellStart,
                                const DeformableWavyMesh &mesh,
                                float friction) {
  Body &box = solver.bodies[boxIdx];
  const float hx = halfExt.x, hy = halfExt.y, hz = halfExt.z;
  const Vec3 corners[4] = {{-hx, -hy, -hz},
                           {hx, -hy, -hz},
                           {hx, -hy, hz},
                           {-hx, -hy, hz}};

  for (int i = 0; i < 4; ++i) {
    const Vec3 worldCorner = box.position + box.rotation.rotate(corners[i]);
    const float sx = worldCorner.x;
    const float sz = worldCorner.z;
    const Vec3 v(sx, mesh.surfaceY(sx, sz), sz);

    Vec3 n = worldCorner - v;
    float dist = n.length();
    if (dist < 1e-4f)
      n = Vec3(0, 1, 0);
    else
      n = n * (1.0f / dist);

    const float sep = (worldCorner - v).dot(n);
    if (sep > kDeformRestOffset + 0.12f)
      continue;

    const int flat = findNearestShellVertex(mesh, v);
    const int ga = flat / kDeformGridSize;
    const int gb = flat % kDeformGridSize;
    n = mesh.normalAt(ga, gb);
    if (n.dot(worldCorner - v) < 0.0f)
      n = n * -1.0f;
    const uint32_t pi = shellStart + uint32_t(flat);

    SoftContact sc;
    sc.particleIdx = pi;
    sc.rigidBodyIdx = boxIdx;
    sc.normal = n;
    sc.depth = kDeformRestOffset;
    sc.friction = friction;
    sc.surfacePoint = v;
    sc.surfacePointPrev = v;
    sc.rigidLocalPoint = corners[i];
    sc.k = 1e4f;
    sc.ke = 1e6f;
    if (fabsf(n.y) > 0.9f)
      sc.tangent1 = n.cross(Vec3(1, 0, 0)).normalized();
    else
      sc.tangent1 = n.cross(Vec3(0, 1, 0)).normalized();
    sc.tangent2 = n.cross(sc.tangent1);
    solver.softContacts.push_back(sc);
  }
}

struct DeformableShellStressMetrics {
  float worstMinBoxBottomRel = 1e9f;
  float worstMinBodyY = 1e9f;
  uint32_t maxSunkBoxes = 0;
  uint32_t maxPassThroughShots = 0;
  uint32_t nanEvents = 0;
  bool ok = false;
};

inline DeformableSphereShotMetrics
runDeformableSphereShotShell(int substeps = kDeformSubsteps,
                             float friction = 0.5f) {
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.dt = 1.0f / 60.0f;
  solver.iterations = 16;
  solver.innerIterations = 4;
  solver.bodyStaticContactSolve = BodyStaticContactSolve::SequentialPerContact;

  const uint32_t sphere =
      solver.addBody({0, kDeformShotSpawnY, 0}, Quat(),
                     {kDeformShotRadius, kDeformShotRadius, kDeformShotRadius},
                     3.0f, friction);
  solver.bodies[sphere].linearVelocity = {0, -kDeformShotSpeedY, 0};

  DeformableWavyMesh mesh;
  const uint32_t shellStart = buildWavyMeshKinematicShell(solver, mesh);
  ShellContactCache cache;
  DeformableSphereShotMetrics m;

  const float subDt = solver.dt / float(substeps);
  const float waveStep = 0.01f / float(substeps);

  for (int frame = 0; frame < kDeformShotFrames; ++frame) {
    for (int sub = 0; sub < substeps; ++sub) {
      mesh.waveTime += waveStep;
      syncKinematicShellFromMesh(solver, shellStart, mesh, subDt);
      solver.softContacts.clear();
      addSphereShellContacts(solver, sphere, shellStart, mesh, friction);
      if (kShellSynthesizeBodyStaticContacts) {
        synthesizeKinematicShellToContacts(solver);
        solver.softContacts.clear();
      }
      cache.restore(solver, shellStart);
      solver.step(subDt);
      cache.save(solver, shellStart);
    }

    Body &sph = solver.bodies[sphere];
    if (sph.position.x != sph.position.x || sph.position.y != sph.position.y) {
      m.nanDetected = true;
      break;
    }

    const float surfaceY = mesh.surfaceY(sph.position.x, sph.position.z);
    const float rayPen =
        surfaceY - (sph.position.y - kDeformShotRadius) - kDeformRestOffset;
    if (rayPen > m.maxRaycastPen)
      m.maxRaycastPen = rayPen;

    const float lateral = sqrtf(sph.position.x * sph.position.x +
                                sph.position.z * sph.position.z);
    if (lateral > m.lateralDriftXZ)
      m.lateralDriftXZ = lateral;
  }

  m.pass = !m.nanDetected && m.maxRaycastPen <= kDeformPassPen &&
           m.lateralDriftXZ <= kDeformPassLateral;

  printf("[DeformableSphereShotShell] friction=%.2f pass=%d "
         "maxRaycastPen=%.4f lateralDriftXZ=%.4f nan=%d finalY=%.4f\n",
         friction, m.pass ? 1 : 0, m.maxRaycastPen, m.lateralDriftXZ,
         m.nanDetected ? 1 : 0, solver.bodies[sphere].position.y);

  return m;
}

inline DeformableShellStressMetrics
runDeformableStressShell(int frames = 600, int substeps = kDeformSubsteps) {
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.dt = 1.0f / 60.0f;
  solver.iterations = 16;
  solver.innerIterations = 4;

  solver.bodyStaticContactSolve = BodyStaticContactSolve::SequentialPerContact;

  DeformableWavyMesh mesh;
  const uint32_t shellStart = buildWavyMeshKinematicShell(solver, mesh);
  ShellContactCache cache;

  const int gridX = 6;
  const int gridZ = 6;
  const float halfExtent = 2.0f;
  const float spacing = halfExtent * 2.2f;
  const Vec3 halfExt(halfExtent, halfExtent, halfExtent);
  std::vector<uint32_t> boxes;
  boxes.reserve(gridX * gridZ);

  for (int iz = 0; iz < gridZ; ++iz) {
    for (int ix = 0; ix < gridX; ++ix) {
      const float x = -0.5f * float(gridX - 1) * spacing + float(ix) * spacing;
      const float z = -0.5f * float(gridZ - 1) * spacing + float(iz) * spacing;
      const float y = mesh.surfaceY(x, z) + halfExtent + 0.05f;
      const uint32_t box =
          solver.addBody({x, y, z}, Quat(), halfExt, 10.0f, 1.0f);
      boxes.push_back(box);
    }
  }

  const float subDt = solver.dt / float(substeps);
  const float waveStep = 0.01f / float(substeps);
  const int shotInterval = 24;
  int nextShot = 0;
  int shotSerial = 0;

  DeformableShellStressMetrics m;

  for (int frame = 0; frame < frames; ++frame) {
    if (frame == nextShot) {
      const float x = float((shotSerial % 7) - 3) * 7.5f;
      const float z = float(((shotSerial * 3) % 11) - 5) * 6.6f;
      const uint32_t shot =
          solver.addBody({x, 55.0f, z}, Quat(),
                         {kDeformShotRadius, kDeformShotRadius,
                          kDeformShotRadius},
                         3.0f, 0.5f);
      solver.bodies[shot].linearVelocity = {0, -kDeformShotSpeedY, 0};
      shotSerial++;
      nextShot += shotInterval;
    }

    for (int sub = 0; sub < substeps; ++sub) {
      mesh.waveTime += waveStep;
      syncKinematicShellFromMesh(solver, shellStart, mesh, subDt);
      solver.softContacts.clear();
      for (uint32_t box : boxes)
        addBoxShellContacts(solver, box, halfExt, shellStart, mesh, 1.0f);
      for (uint32_t bi = 0; bi < solver.bodies.size(); ++bi) {
        if (solver.bodies[bi].mass < 5.0f)
          addSphereShellContacts(solver, bi, shellStart, mesh, 0.5f);
      }
      if (kShellSynthesizeBodyStaticContacts) {
        synthesizeKinematicShellToContacts(solver);
        solver.softContacts.clear();
      }
      cache.restore(solver, shellStart);
      solver.step(subDt);
      cache.save(solver, shellStart);
    }

    uint32_t frameSunk = 0;
    uint32_t framePassThrough = 0;
    float frameMinRel = 1e9f;
    for (uint32_t box : boxes) {
      const Body &b = solver.bodies[box];
      if (b.position.y != b.position.y) {
        m.nanEvents++;
        continue;
      }
      const float bottomY = b.position.y - halfExtent;
      const float surf = mesh.surfaceY(b.position.x, b.position.z);
      const float rel = bottomY - surf;
      frameMinRel = std::min(frameMinRel, rel);
      if (rel < -0.5f)
        frameSunk++;
    }
    for (uint32_t bi = 0; bi < solver.bodies.size(); ++bi) {
      if (solver.bodies[bi].mass >= 5.0f)
        continue;
      const Body &s = solver.bodies[bi];
      const float surf = mesh.surfaceY(s.position.x, s.position.z);
      if (s.position.y < surf - 5.0f)
        framePassThrough++;
    }

    if (frameMinRel < 1e8f)
      m.worstMinBoxBottomRel =
          std::min(m.worstMinBoxBottomRel, frameMinRel);
    m.maxSunkBoxes = std::max(m.maxSunkBoxes, frameSunk);
    m.maxPassThroughShots =
        std::max(m.maxPassThroughShots, framePassThrough);
  }

  m.ok = (m.nanEvents == 0) && (m.maxPassThroughShots == 0) &&
         (m.maxSunkBoxes == 0) && (m.worstMinBoxBottomRel > -0.5f);

  printf("[DeformableStressShell] frames=%d worstMinBoxBottomRel=%.4f "
         "maxSunkBoxes=%u maxPassThroughShots=%u nanEvents=%u ok=%d\n",
         frames, m.worstMinBoxBottomRel, m.maxSunkBoxes, m.maxPassThroughShots,
         m.nanEvents, m.ok ? 1 : 0);

  return m;
}

} // namespace AvbdRef
