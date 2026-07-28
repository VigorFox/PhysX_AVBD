#pragma once
#include "avbd_articulation.h"
#include "avbd_body_static_semantics.h"
#include "avbd_island_pcg.h"
#include "avbd_softbody.h"
#include "avbd_types.h"
#include <vector>

namespace AvbdRef {

static constexpr float PENALTY_MIN = 1000.0f;
static constexpr float PENALTY_MAX = 1e9f;

/** Optional island dispatch using the same primal/dual body-static rules. */
enum class BodyStaticContactSolve {
  Aggregated6x6,       //!< PhysX default: per-body 6x6 normals; dual tangents
  SequentialPerContact //!< Standalone alias: normals-only primal pass + dual
};

/** Process-wide test harness switch, set before any Solver is constructed. */
void setContactIslandPcgSuiteProbeEnabled(bool enabled);
bool isContactIslandPcgSuiteProbeEnabled();
void setCanonicalRigidContactAuthoringSuiteProbeEnabled(bool enabled);
bool isCanonicalRigidContactAuthoringSuiteProbeEnabled();

/** Read-only diagnostics for the large dynamic-contact friction stage. */
struct DynDynFrictionPassStats {
  uint32_t dynamicContactCount = 0;
  uint32_t invocationCount = 0;
  uint32_t activeInvocationCount = 0;
  uint32_t tangentImpulseCount = 0;
  float maxNormalImpulseLimit = 0.0f;
  float totalAbsTangentImpulse = 0.0f;
  float maxLinearMomentumDelta = 0.0f;
  float maxAngularMomentumDelta = 0.0f;
};

/** Read-only diagnostics for routed force-mode linear-drive rows. */
struct LinearDriveIslandStats {
  uint32_t emittedRowCount = 0;
  uint32_t accelerationRowCount = 0;
  uint32_t unsaturatedRowCount = 0;
  uint32_t saturatedRowCount = 0;
  float maxAbsForce = 0.0f;
  float maxForceLimit = 0.0f;
  float maxAbsDual = 0.0f;
};

/** Read-only diagnostics for routed TWIST velocity-drive rows. */
struct AngularDriveIslandStats {
  uint32_t emittedRowCount = 0;
  uint32_t accelerationRowCount = 0;
  uint32_t unsaturatedRowCount = 0;
  uint32_t saturatedRowCount = 0;
  float maxAbsTorque = 0.0f;
  float maxTorqueLimit = 0.0f;
  float maxAbsDual = 0.0f;
};

/**
 * Restore the shared rigid rotation about a frictionless support normal.
 *
 * This changes only the two-body rigid null mode: both angular velocities
 * receive the same axis rotation and both COM velocities receive the
 * corresponding rotation about the instantaneous system COM.  Total linear
 * momentum, every globally rotation-invariant joint coordinate derivative,
 * and support-normal point velocity are therefore unchanged.
 */
bool restoreTwoBodySupportAxisAngularMomentum(
    Body &bodyA, Body &bodyB, const Vec3 &supportNormal,
    float expectedAxisAngularMomentum);

struct Solver {
  Vec3 gravity = {0, -9.8f, 0};
  int iterations = 10;
  int outerIterations = 1;    // AVBD outer iterations (proximal anchor updates)
  int innerIterations = 10;   // AVBD inner iterations (VBD sweeps per outer)
  float alpha = 0.95f;              // stabilization
  float beta = 1000.0f;             // penalty growth rate
  float gamma = 0.99f;              // warmstart decay
  float penaltyScaleDynDyn = kPenScaleDynDyn; // dyn-dyn floor scale (PhysX-matched)
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
  /** Probe-only routing; default off until frame-level gates accept it. */
  bool useIslandPcgProbe = false;
  IslandPcgStats islandPcgLastStats;
  /** Probe control; true preserves the production large-contact path. */
  bool enableSequentialDynDynFriction = true;
  /** Probe-only physical sign for angular impulse integration. */
  bool useFrictionAngularImpulseSignProbe = false;
  /** Probe-only statistics; disabled to avoid default runtime overhead. */
  bool enableDynDynFrictionDiagnostics = false;
  DynDynFrictionPassStats dynDynFrictionLastStats;
  /** Probe-only shared normal + 2D tangent contact-island objective. */
  bool useContactIslandPcgProbe = isContactIslandPcgSuiteProbeEnabled();
  /** Canonical same-world-point authoring for rigid test contacts only. */
  bool useCanonicalRigidContactAuthoringProbe =
      isCanonicalRigidContactAuthoringSuiteProbeEnabled();
  IslandPcgStats contactIslandPcgLastStats;
  /** True only when the current step used the complete shared contact island. */
  bool contactIslandPcgRoutedLastStep = false;
  LinearDriveIslandStats linearDriveIslandLastStats;
  AngularDriveIslandStats angularDriveIslandLastStats;

  BodyStaticContactSolve bodyStaticContactSolve =
      BodyStaticContactSolve::Aggregated6x6;
  /** If true, allow body-static tangents in 6x6 for low-contact islands (friction tests). */
  bool allowBodyStaticFrictionIn6x6LowContact = true;

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
                             float maxForce, bool freeSpin = false,
                             float gearRatio = 1.0f);

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

  // Soft body creation — returns index of first particle added
  uint32_t addSoftBody(const std::vector<Vec3> &vertices,
                       const std::vector<uint32_t> &tets,
                       const std::vector<uint32_t> &tris,
                       float youngsModulus = 1e5f,
                       float poissonsRatio = 0.3f,
                       float density = 100.0f,
                       float damping = 0.01f,
                       float bendingStiffness = 0.0f,
                       float thickness = 0.01f);

  /** Kinematic collision shell: particles with invMass=0, no internal elasticity. */
  uint32_t addKinematicShell(const std::vector<Vec3> &positions);

  // Solver core
  void computeConstraint(Contact &c);
  void computeConstraintBodyStatic(Contact &c);
  void computeC0(Contact &c);
  void warmstart();
  void step(float dt_);

private:
  bool isSequentialBodyStaticIsland() const;
  bool bodyTouchesStatic(uint32_t bodyIdx) const;
  bool bodyTouchesKinematicShell(uint32_t bodyIdx) const;
  void sequentialBodyStaticPrimalPass(float dt);
  void applyBodyStaticDepenetrationSweeps(uint32_t sweeps);
  void applyLowIslandDynDynFrictionSweeps(uint32_t sweeps);
  void sequentialDynDynFrictionPass(float dt);
  bool solveFixedD6IslandPcgProbe(float dt);
  bool solveContactIslandPcgProbe(float dt);
  float contactGeomViolation(const Contact &c) const;
};

} // namespace AvbdRef
