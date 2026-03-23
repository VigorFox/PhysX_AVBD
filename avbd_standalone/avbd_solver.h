#pragma once
#include "avbd_articulation.h"
#include "avbd_softbody.h"
#include "avbd_types.h"
#include <vector>

namespace AvbdRef {

static constexpr float PENALTY_MIN = 1000.0f;
static constexpr float PENALTY_MAX = 1e9f;

struct Solver {
  Vec3 gravity = {0, -9.8f, 0};
  int iterations = 10;
  int outerIterations = 1;    // AVBD outer iterations (proximal anchor updates)
  int innerIterations = 10;   // AVBD inner iterations (VBD sweeps per outer)
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

  // Phase 3: Convergence acceleration
  bool useTreeSweep = false;           // tree-structured sweep ordering for artic chains
  bool useAndersonAccel = false;       // Anderson Acceleration on body positions
  int aaWindowSize = 3;                // AA window size (m)
  bool useChebyshev = false;           // Chebyshev semi-iterative position relaxation
  float chebyshevSpectralRadius = 0.92f; // estimated spectral radius

  // Per-step convergence history (populated if articulations present)
  std::vector<float> convergenceHistory;

  std::vector<Body> bodies;
  std::vector<Contact> contacts;
  std::vector<D6Joint> d6Joints;     // unified: all joint types
  std::vector<GearJoint> gearJoints; // kept separate (velocity constraint)
  std::vector<Articulation> articulations; // pure AVBD articulations (AL constraints)

  // Soft body system
  std::vector<SoftParticle> softParticles;
  std::vector<SoftBody> softBodies;
  std::vector<SoftContact> softContacts;

  // Joint creation (all return index into d6Joints)
  uint32_t addSphericalJoint(uint32_t bodyA, uint32_t bodyB,
                             Vec3 localAnchorA, Vec3 localAnchorB,
                             float rho_ = 1e6f);
  uint32_t addFixedJoint(uint32_t bodyA, uint32_t bodyB,
                         Vec3 localAnchorA, Vec3 localAnchorB,
                         float rho_ = 1e6f);
  uint32_t addD6Joint(uint32_t bodyA, uint32_t bodyB,
                      Vec3 anchorA, Vec3 anchorB,
                      uint32_t linearMotion_ = 0,
                      uint32_t angularMotion_ = 0x2A,
                      float angularDamping_ = 0.0f, float rho_ = 1e6f);
  uint32_t addRevoluteJoint(uint32_t bodyA, uint32_t bodyB,
                            Vec3 localAnchorA, Vec3 localAnchorB,
                            Vec3 localAxisA,
                            Vec3 localAxisB = Vec3(0, 0, 1),
                            float rho = 1e6f);
  uint32_t addPrismaticJoint(uint32_t bodyA, uint32_t bodyB,
                             Vec3 localAnchorA, Vec3 localAnchorB,
                             Vec3 localAxisA, float rho = 1e6f);

  // Cone limit (spherical joints)
  void setSphericalJointConeLimit(uint32_t jointIdx, Vec3 coneAxisA,
                                  float limitAngle);

  // Revolute limit/drive (operates on d6Joints by index)
  void setRevoluteJointLimit(uint32_t jointIdx, float lowerLimit,
                             float upperLimit);
  void setRevoluteJointDrive(uint32_t jointIdx, float targetVelocity,
                             float maxForce);

  // Prismatic limit/drive (operates on d6Joints by index)
  void setPrismaticJointLimit(uint32_t jointIdx, float lowerLimit,
                              float upperLimit);
  void setPrismaticJointDrive(uint32_t jointIdx, float targetVelocity,
                              float damping);

  // Gear joint (separate)
  void addGearJoint(uint32_t bodyA, uint32_t bodyB,
                    Vec3 axisA, Vec3 axisB,
                    float ratio = -1.f, float rho = 1e5f);

  // Body creation
  uint32_t addBody(Vec3 pos, Quat rot, Vec3 halfExtent, float density,
                   float fric = 0.5f);

  // Contact creation
  void addContact(uint32_t bodyA, uint32_t bodyB, Vec3 normal, Vec3 rA,
                  Vec3 rB, float depth, float fric = 0.5f);

  // Soft body creation
  // Returns index of first particle added
  uint32_t addSoftBody(const std::vector<Vec3>& vertices,
                       const std::vector<uint32_t>& tets,
                       const std::vector<uint32_t>& tris,
                       float youngsModulus = 1e5f,
                       float poissonsRatio = 0.3f,
                       float density = 100.0f,
                       float damping = 0.01f,
                       float bendingStiffness = 0.0f,
                       float thickness = 0.01f);

  // Solver core
  void computeConstraint(Contact &c);
  void computeC0(Contact &c);
  void warmstart();
  void step(float dt_);
};

} // namespace AvbdRef
