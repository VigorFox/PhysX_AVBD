#pragma once
#include "avbd_types.h"
#include <vector>

namespace AvbdRef {

static constexpr float PENALTY_MIN = 1000.0f;
static constexpr float PENALTY_MAX = 1e9f;

struct Solver {
  Vec3 gravity = {0, -9.8f, 0};
  int iterations = 10;
  float alpha = 0.95f;              // stabilization
  float beta = 1000.0f;             // penalty growth rate
  float gamma = 0.99f;              // warmstart decay
  float penaltyScale = 0.25f;       // body-ground penalty floor
  float penaltyScaleDynDyn = 0.05f; // dynamic-dynamic penalty
  int propagationDepth = 4;         // graph-propagation depth
  float propagationDecay = 0.5f;    // per-edge decay factor
  float dt = 1.0f / 60.0f;
  bool use3x3Solve = false; // false=6x6 LDLT (default), true=block-elim 3x3
  bool verbose = false;     // per-iteration logging

  std::vector<Body> bodies;
  std::vector<Contact> contacts;
  std::vector<SphericalJoint> sphericalJoints;
  std::vector<FixedJoint> fixedJoints;
  std::vector<D6Joint> d6Joints;

  // Joint creation
  void addSphericalJoint(uint32_t bodyA, uint32_t bodyB, Vec3 anchorA,
                         Vec3 anchorB, float rho_ = 1e6f);

  void addFixedJoint(uint32_t bodyA, uint32_t bodyB, Vec3 anchorA, Vec3 anchorB,
                     float rho_ = 1e6f);

  void addD6Joint(uint32_t bodyA, uint32_t bodyB, Vec3 anchorA, Vec3 anchorB,
                  uint32_t linearMotion_ = 0, uint32_t angularMotion_ = 0x2A,
                  float angularDamping_ = 0.0f, float rho_ = 1e6f);

  // Body creation
  uint32_t addBody(Vec3 pos, Quat rot, Vec3 halfExtent, float density,
                   float fric = 0.5f);

  // Contact creation
  void addContact(uint32_t bodyA, uint32_t bodyB, Vec3 normal, Vec3 rA, Vec3 rB,
                  float depth, float fric = 0.5f);

  // Solver core
  void computeConstraint(Contact &c);
  void computeC0(Contact &c);
  void warmstart();
  void step(float dt_);
};

} // namespace AvbdRef
