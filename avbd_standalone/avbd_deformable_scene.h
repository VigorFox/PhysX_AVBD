#pragma once
// Shared deformable wavy-mesh scene (mirrors SnippetDeformableMesh headless sphere-shot).
#include "avbd_solver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace AvbdRef {

static constexpr float kDeformPi = 3.14159265f;
static constexpr int kDeformGridSize = 8;
static constexpr float kDeformGridStep = 512.0f / float(kDeformGridSize - 1);
/** Snippet shape restOffset=-0.5 (raycast gate expects ~0.55-0.65 at rest). */
static constexpr float kDeformRestOffset = 0.5f;
static constexpr float kDeformMeshActorY = 2.0f;
static constexpr float kDeformShotRadius = 3.0f;
static constexpr float kDeformShotSpawnY = 55.0f;
static constexpr float kDeformShotSpeedY = 200.0f;
static constexpr float kDeformPassPen = 0.70f;
static constexpr float kDeformPassLateral = 15.0f;
static constexpr int kDeformShotFrames = 180;
static constexpr int kDeformSubsteps = 4;

struct DeformableWavyMesh {
  float waveTime = 0.0f;

  float amplitude() const { return sinf(waveTime) * 20.0f; }

  Vec3 vertex(int a, int b) const {
    const float coeffA = float(a) / float(kDeformGridSize - 1);
    const float coeffB = float(b) / float(kDeformGridSize - 1);
    const float y =
        kDeformMeshActorY + 20.0f +
        sinf(coeffA * 2.0f * kDeformPi) * cosf(coeffB * 2.0f * kDeformPi) *
            amplitude();
    return Vec3(-400.0f + float(b) * kDeformGridStep, y,
                -400.0f + float(a) * kDeformGridStep);
  }

  Vec3 normalAt(int a, int b) const {
    const int am = (a > 0) ? a - 1 : a;
    const int ap = (a < kDeformGridSize - 1) ? a + 1 : a;
    const int bm = (b > 0) ? b - 1 : b;
    const int bp = (b < kDeformGridSize - 1) ? b + 1 : b;
    const Vec3 dx = vertex(a, bp) - vertex(a, bm);
    const Vec3 dz = vertex(ap, b) - vertex(am, b);
    Vec3 n = dz.cross(dx);
    const float len = n.length();
    return (len > 1e-6f) ? n * (1.0f / len) : Vec3(0, 1, 0);
  }

  float surfaceY(float x, float z) const {
    const float gx = (x + 400.0f) / kDeformGridStep;
    const float gz = (z + 400.0f) / kDeformGridStep;
    const int b0 = std::max(0, std::min(kDeformGridSize - 2, (int)gx));
    const int a0 = std::max(0, std::min(kDeformGridSize - 2, (int)gz));
    const float tx = gx - float(b0);
    const float tz = gz - float(a0);
    const float y00 = vertex(a0, b0).y;
    const float y10 = vertex(a0, b0 + 1).y;
    const float y01 = vertex(a0 + 1, b0).y;
    const float y11 = vertex(a0 + 1, b0 + 1).y;
    const float y0 = y00 * (1.0f - tx) + y10 * tx;
    const float y1 = y01 * (1.0f - tx) + y11 * tx;
    return y0 * (1.0f - tz) + y1 * tz;
  }
};

inline bool isBodyStaticContact(const Contact &c) {
  return c.bodyB == UINT32_MAX;
}

struct DeformableContactCache {
  struct Key {
    int32_t ia, ib;
    bool operator<(const Key &o) const {
      if (ia != o.ia)
        return ia < o.ia;
      return ib < o.ib;
    }
  };
  struct Entry {
    float lambda[3];
    float penalty[3];
    Vec3 staticPrevWorld;
  };
  std::map<Key, Entry> data;

  static Key findGridKey(const DeformableWavyMesh &mesh, const Vec3 &worldB) {
    for (int a = 0; a < kDeformGridSize; ++a) {
      for (int b = 0; b < kDeformGridSize; ++b) {
        const Vec3 d = mesh.vertex(a, b) - worldB;
        if (d.dot(d) < 1e-3f)
          return {a, b};
      }
    }
    return {-1, -1};
  }

  void save(const DeformableWavyMesh &mesh, const Solver &solver) {
    data.clear();
    for (const auto &c : solver.contacts) {
      if (!isBodyStaticContact(c))
        continue;
      const Key k = findGridKey(mesh, c.rB);
      if (k.ia < 0)
        continue;
      Entry e;
      for (int i = 0; i < 3; ++i) {
        e.lambda[i] = c.lambda[i];
        e.penalty[i] = c.penalty[i];
      }
      e.staticPrevWorld = c.rB;
      data[k] = e;
    }
  }

  void restore(const DeformableWavyMesh &mesh, Solver &solver) {
    for (auto &c : solver.contacts) {
      if (!isBodyStaticContact(c))
        continue;
      const Key k = findGridKey(mesh, c.rB);
      if (k.ia < 0)
        continue;
      auto it = data.find(k);
      if (it == data.end())
        continue;
      for (int i = 0; i < 3; ++i) {
        c.lambda[i] = it->second.lambda[i];
        c.penalty[i] = it->second.penalty[i];
      }
      c.staticPrevWorldPoint = it->second.staticPrevWorld;
    }
  }
};

inline void addSphereDeformableMeshContacts(Solver &solver, uint32_t sphereIdx,
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
      solver.addContact(sphereIdx, UINT32_MAX, n, rA, v, kDeformRestOffset,
                        friction);
    }
  }
}

struct DeformableSphereShotMetrics {
  float maxRaycastPen = 0.0f;
  float lateralDriftXZ = 0.0f;
  bool nanDetected = false;
  bool pass = false;
};

inline DeformableSphereShotMetrics
runDeformableSphereShot(BodyStaticContactSolve mode, const char *modeLabel,
                        float friction = 0.5f, int substeps = kDeformSubsteps) {
  Solver solver;
  solver.gravity = {0, -9.81f, 0};
  solver.dt = 1.0f / 60.0f;
  solver.iterations = 16;
  solver.bodyStaticContactSolve = mode;
  solver.allowBodyStaticFrictionIn6x6LowContact = false;

  const uint32_t sphere =
      solver.addBody({0, kDeformShotSpawnY, 0}, Quat(),
                     {kDeformShotRadius, kDeformShotRadius, kDeformShotRadius},
                     3.0f, friction);
  solver.bodies[sphere].linearVelocity = {0, -kDeformShotSpeedY, 0};

  DeformableWavyMesh mesh;
  DeformableContactCache cache;
  DeformableSphereShotMetrics m;

  const float subDt = solver.dt / float(substeps);
  const float waveStep = 0.01f / float(substeps);

  for (int frame = 0; frame < kDeformShotFrames; ++frame) {
    for (int sub = 0; sub < substeps; ++sub) {
      mesh.waveTime += waveStep;
      solver.contacts.clear();
      addSphereDeformableMeshContacts(solver, sphere, mesh, friction);
      cache.restore(mesh, solver);
      solver.step(subDt);
      cache.save(mesh, solver);
    }

    Body &sph = solver.bodies[sphere];
    if (sph.position.x != sph.position.x || sph.position.y != sph.position.y) {
      m.nanDetected = true;
      break;
    }

    const float surfaceY = mesh.surfaceY(sph.position.x, sph.position.z);
    // Include restOffset like Snippet shape (restOffset=-0.5).
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

  printf("[DeformableSphereShot] mode=%s friction=%.2f pass=%d "
         "maxRaycastPen=%.4f lateralDriftXZ=%.4f nan=%d "
         "finalY=%.4f\n",
         modeLabel, friction, m.pass ? 1 : 0, m.maxRaycastPen, m.lateralDriftXZ,
         m.nanDetected ? 1 : 0, solver.bodies[sphere].position.y);

  return m;
}

} // namespace AvbdRef
