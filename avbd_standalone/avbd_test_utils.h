#pragma once
#include "avbd_solver.h"
#include <map>
#include <vector>

namespace AvbdRef {

// =============================================================================
// Helper: Generate box-on-ground contacts
// =============================================================================
inline void addBoxGroundContacts(Solver &solver, uint32_t boxIdx,
                                 Vec3 halfExt) {
  Body &box = solver.bodies[boxIdx];

  float hx = halfExt.x, hy = halfExt.y, hz = halfExt.z;
  Vec3 corners[4] = {
      {-hx, -hy, -hz},
      {hx, -hy, -hz},
      {hx, -hy, hz},
      {-hx, -hy, hz},
  };

  Vec3 normal(0, 1, 0);

  for (int i = 0; i < 4; i++) {
    Vec3 worldCorner = box.position + box.rotation.rotate(corners[i]);
    float depth = -worldCorner.y; // positive when penetrating
    Vec3 groundPoint(worldCorner.x, 0, worldCorner.z);
    solver.addContact(boxIdx, UINT32_MAX, normal, corners[i], groundPoint,
                      depth, box.friction);
  }
}

// =============================================================================
// Helper: Generate box-on-box contacts
// =============================================================================
inline void addBoxOnBoxContacts(Solver &solver, uint32_t topIdx,
                                uint32_t bottomIdx, Vec3 halfExtTop,
                                Vec3 halfExtBot) {
  Body &top = solver.bodies[topIdx];
  Body &bot = solver.bodies[bottomIdx];

  float hx = halfExtTop.x, hy = halfExtTop.y, hz = halfExtTop.z;
  Vec3 corners[4] = {
      {-hx, -hy, -hz},
      {hx, -hy, -hz},
      {hx, -hy, hz},
      {-hx, -hy, hz},
  };

  Vec3 normal(0, 1, 0); // from bottom body to top body

  for (int i = 0; i < 4; i++) {
    Vec3 worldCornerA = top.position + top.rotation.rotate(corners[i]);
    float topOfBot = bot.position.y + halfExtBot.y;
    float depth = topOfBot - worldCornerA.y;

    Vec3 rA = corners[i];
    Vec3 worldContact(worldCornerA.x, topOfBot, worldCornerA.z);
    Vec3 rB = worldContact - bot.position;

    float fric = sqrtf(top.friction * bot.friction);
    solver.addContact(topIdx, bottomIdx, normal, rA, rB, depth, fric);
  }
}

// =============================================================================
// ContactCache -- warm-start lambda / penalty across frames
// =============================================================================
struct ContactCache {
  struct Key {
    uint32_t bodyA, bodyB;
    int32_t rAx, rAy, rAz;
    bool operator<(const Key &o) const {
      if (bodyA != o.bodyA)
        return bodyA < o.bodyA;
      if (bodyB != o.bodyB)
        return bodyB < o.bodyB;
      if (rAx != o.rAx)
        return rAx < o.rAx;
      if (rAy != o.rAy)
        return rAy < o.rAy;
      return rAz < o.rAz;
    }
  };
  struct Entry {
    float lambda[3], penalty[3];
  };
  std::map<Key, Entry> data;

  static Key makeKey(const Contact &c) {
    return {c.bodyA, c.bodyB,
            (int32_t)(c.rA.x * 1000.0f + (c.rA.x >= 0 ? 0.5f : -0.5f)),
            (int32_t)(c.rA.y * 1000.0f + (c.rA.y >= 0 ? 0.5f : -0.5f)),
            (int32_t)(c.rA.z * 1000.0f + (c.rA.z >= 0 ? 0.5f : -0.5f))};
  }

  void save(const Solver &solver) {
    data.clear();
    for (const auto &c : solver.contacts) {
      Entry e;
      for (int i = 0; i < 3; i++) {
        e.lambda[i] = c.lambda[i];
        e.penalty[i] = c.penalty[i];
      }
      data[makeKey(c)] = e;
    }
  }

  void restore(Solver &solver) {
    for (auto &c : solver.contacts) {
      auto it = data.find(makeKey(c));
      if (it != data.end()) {
        for (int i = 0; i < 3; i++) {
          c.lambda[i] = it->second.lambda[i];
          c.penalty[i] = it->second.penalty[i];
        }
      }
    }
  }
};

// =============================================================================
// Dynamic contact generation (with proximity check)
// =============================================================================
inline void addBoxGroundContactsDynamic(Solver &solver, uint32_t boxIdx,
                                        Vec3 halfExt, float margin = 0.1f) {
  Body &box = solver.bodies[boxIdx];
  float hx = halfExt.x, hy = halfExt.y, hz = halfExt.z;
  Vec3 corners[4] = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz}, {-hx, -hy, hz}};
  Vec3 normal(0, 1, 0);
  for (int i = 0; i < 4; i++) {
    Vec3 wc = box.position + box.rotation.rotate(corners[i]);
    if (wc.y > margin)
      continue;
    float depth = -wc.y;
    Vec3 gp(wc.x, 0, wc.z);
    solver.addContact(boxIdx, UINT32_MAX, normal, corners[i], gp, depth,
                      box.friction);
  }
}

} // namespace AvbdRef
