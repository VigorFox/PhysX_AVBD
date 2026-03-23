#include "avbd_solver.h"
#include "avbd_articulation.h"
#include "avbd_d6_core.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace AvbdRef {

// =============================================================================
// Factory methods  all create D6Joint entries in the unified d6Joints vector
// =============================================================================

uint32_t Solver::addSphericalJoint(uint32_t bodyA, uint32_t bodyB,
                                   Vec3 anchorA, Vec3 anchorB, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = 0;     // all 3 linear LOCKED
  j.angularMotion = 0x2A; // all 3 angular FREE (2|(2<<2)|(2<<4))
  j.rho = rho_;

  // Compute localFrameB from initial relative rotation (matches PhysX)
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

void Solver::setSphericalJointConeLimit(uint32_t jointIdx, Vec3 coneAxisA,
                                        float limitAngle) {
  if (jointIdx < d6Joints.size()) {
    d6Joints[jointIdx].coneAngleLimit = limitAngle;
    d6Joints[jointIdx].coneAxisA = coneAxisA;
    d6Joints[jointIdx].coneLambda = 0.0f;

    // Build localFrameA so X-axis = cone axis (matches PhysX joint frame)
    Vec3 axisNorm = coneAxisA.normalized();
    Vec3 perp;
    if (std::fabs(axisNorm.x) < 0.9f)
      perp = axisNorm.cross(Vec3(1, 0, 0)).normalized();
    else
      perp = axisNorm.cross(Vec3(0, 1, 0)).normalized();
    Vec3 perp2 = axisNorm.cross(perp);
    d6Joints[jointIdx].localFrameA = quatFromColumns(axisNorm, perp, perp2);
    d6Joints[jointIdx].localFrameB =
        d6Joints[jointIdx].relativeRotation.conjugate() *
        d6Joints[jointIdx].localFrameA;
  }
}

uint32_t Solver::addFixedJoint(uint32_t bodyA, uint32_t bodyB,
                               Vec3 anchorA, Vec3 anchorB, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = 0;  // all 3 linear LOCKED
  j.angularMotion = 0; // all 3 angular LOCKED
  j.rho = rho_;
  // Compute initial relative rotation and localFrameB
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;
  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

uint32_t Solver::addD6Joint(uint32_t bodyA, uint32_t bodyB,
                            Vec3 anchorA, Vec3 anchorB,
                            uint32_t linearMotion_, uint32_t angularMotion_,
                            float angularDamping_, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = linearMotion_;
  j.angularMotion = angularMotion_;
  j.rho = rho_;
  j.angularDriveDamping =
      Vec3(angularDamping_, angularDamping_, angularDamping_);

  // Compute relativeRotation and localFrameB (matches PhysX)
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

uint32_t Solver::addRevoluteJoint(uint32_t bodyA, uint32_t bodyB,
                                  Vec3 localAnchorA, Vec3 localAnchorB,
                                  Vec3 localAxisA, Vec3 localAxisB,
                                  float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = localAnchorA;
  j.anchorB = localAnchorB;
  j.linearMotion = 0; // all 3 linear LOCKED

  // Angular: X(twist) = FREE, Y(swing1) = LOCKED, Z(swing2) = LOCKED
  // Hinge axis maps to joint-frame X
  j.angularMotion = 2; // 2|(0<<2)|(0<<4) = axis0=FREE, axis1=LOCKED, axis2=LOCKED

  j.rho = rho_;

  // Build reference axes perpendicular to hinge axis
  Vec3 axisA = localAxisA.normalized();
  Vec3 axisB = localAxisB.normalized();

  auto buildRefAxis = [](Vec3 axis) -> Vec3 {
    Vec3 perp;
    if (fabsf(axis.x) < 0.9f)
      perp = axis.cross(Vec3(1, 0, 0)).normalized();
    else
      perp = axis.cross(Vec3(0, 1, 0)).normalized();
    return perp;
  };

  Vec3 refA = buildRefAxis(axisA);
  Vec3 z_axisA = axisA.cross(refA);

  // localFrameA: X=hingeAxis, Y=refAxisA, Z=cross
  j.localFrameA = quatFromColumns(axisA, refA, z_axisA);

  // Store relative rotation for angular error computation
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;

  // Store revolute-specific fields for hinge angle measurement
  j.hingeAxisB = axisB;
  j.refAxisA = refA;

  // Compute refAxisB: project worldRefA onto B hinge plane, transform to B local
  Vec3 worldRefA = rotA.rotate(refA);
  Vec3 worldAxisB = rotB.rotate(axisB);
  Vec3 proj = worldRefA - worldAxisB * worldRefA.dot(worldAxisB);
  float projLen = proj.length();
  if (projLen > 1e-8f) {
    j.refAxisB = rotB.conjugate().rotate(proj * (1.0f / projLen));
  } else {
    j.refAxisB = buildRefAxis(axisB);
  }

  // Compute localFrameB from relativeRotation (matches PhysX convention)
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

void Solver::setRevoluteJointLimit(uint32_t jointIdx, float lowerLimit,
                                   float upperLimit) {
  if (jointIdx < d6Joints.size()) {
    // Change angular axis 0 from FREE to LIMITED
    d6Joints[jointIdx].angularMotion =
        (d6Joints[jointIdx].angularMotion & ~0x3) | 1; // axis 0 = LIMITED
    d6Joints[jointIdx].angularLimitLower[0] = lowerLimit;
    d6Joints[jointIdx].angularLimitUpper[0] = upperLimit;
    d6Joints[jointIdx].lambdaLimitAngular[0] = 0.0f;
  }
}

void Solver::setRevoluteJointDrive(uint32_t jointIdx, float targetVelocity,
                                   float maxForce) {
  if (jointIdx < d6Joints.size()) {
    // Use post-solve motor (matches PhysX) instead of AL velocity drive.
    // This avoids ADMM oscillation when coupled with gear constraints.
    d6Joints[jointIdx].motorEnabled = true;
    d6Joints[jointIdx].motorTargetVelocity = targetVelocity;
    d6Joints[jointIdx].motorMaxForce = maxForce;
  }
}

uint32_t Solver::addPrismaticJoint(uint32_t bodyA, uint32_t bodyB,
                                   Vec3 localAnchorA, Vec3 localAnchorB,
                                   Vec3 localAxisA, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = localAnchorA;
  j.anchorB = localAnchorB;
  j.angularMotion = 0; // all 3 angular LOCKED

  // Linear: X(slide) = FREE, Y = LOCKED, Z = LOCKED
  j.linearMotion = 2; // axis0=FREE, axis1=LOCKED, axis2=LOCKED

  j.rho = rho_;

  Vec3 axisA = localAxisA.normalized();
  Vec3 helper =
      (std::abs(axisA.x) > 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
  Vec3 t1 = axisA.cross(helper).normalized();
  Vec3 t2 = axisA.cross(t1);
  j.localFrameA = quatFromColumns(axisA, t1, t2);

  // Store relative rotation and compute localFrameB
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

void Solver::setPrismaticJointLimit(uint32_t jointIdx, float lowerLimit,
                                    float upperLimit) {
  if (jointIdx < d6Joints.size()) {
    // Change linear axis 0 from FREE to LIMITED
    d6Joints[jointIdx].linearMotion =
        (d6Joints[jointIdx].linearMotion & ~0x3) | 1; // axis 0 = LIMITED
    d6Joints[jointIdx].linearLimitLower[0] = lowerLimit;
    d6Joints[jointIdx].linearLimitUpper[0] = upperLimit;
    d6Joints[jointIdx].lambdaLimitLinear[0] = 0.0f;
  }
}

void Solver::setPrismaticJointDrive(uint32_t jointIdx, float targetVelocity,
                                    float damping) {
  if (jointIdx < d6Joints.size()) {
    d6Joints[jointIdx].driveFlags |= 0x01; // linear X drive
    d6Joints[jointIdx].driveLinearVelocity = Vec3(targetVelocity, 0, 0);
    d6Joints[jointIdx].linearDriveDamping.x = damping;
    d6Joints[jointIdx].lambdaDriveLinear = Vec3();
  }
}

void Solver::addGearJoint(uint32_t bodyA, uint32_t bodyB,
                          Vec3 axisA, Vec3 axisB,
                          float ratio, float rho_) {
  GearJoint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.axisA = axisA.normalized();
  j.axisB = axisB.normalized();
  j.gearRatio = ratio;
  j.rho = rho_;
  gearJoints.push_back(j);
}

// =============================================================================
// Body / Contact creation
// =============================================================================

uint32_t Solver::addBody(Vec3 pos, Quat rot, Vec3 halfExtent, float density,
                         float fric) {
  Body b;
  b.position = pos;
  b.rotation = rot;
  b.linearVelocity = {};
  b.angularVelocity = {};
  b.prevLinearVelocity = {};
  b.friction = fric;
  b.halfExtent = halfExtent;

  float vol = 8.0f * halfExtent.x * halfExtent.y * halfExtent.z;
  if (density > 0) {
    b.mass = vol * density;
    float sx = 2 * halfExtent.x, sy = 2 * halfExtent.y, sz = 2 * halfExtent.z;
    float Ixx = b.mass / 12.0f * (sy * sy + sz * sz);
    float Iyy = b.mass / 12.0f * (sx * sx + sz * sz);
    float Izz = b.mass / 12.0f * (sx * sx + sy * sy);
    b.inertiaTensor = Mat33::diag(Ixx, Iyy, Izz);
  } else {
    b.mass = 0;
    b.inertiaTensor = Mat33::diag(0, 0, 0);
  }
  b.computeDerived();

  uint32_t idx = (uint32_t)bodies.size();
  bodies.push_back(b);
  return idx;
}

void Solver::addContact(uint32_t bodyA, uint32_t bodyB, Vec3 normal, Vec3 rA,
                        Vec3 rB, float depth, float fric) {
  Contact c;
  c.bodyA = bodyA;
  c.bodyB = bodyB;
  c.normal = normal;
  c.rA = rA;
  c.rB = rB;
  c.depth = depth;
  c.friction = fric;
  for (int i = 0; i < 3; i++) {
    c.lambda[i] = 0;
    c.penalty[i] = PENALTY_MIN;
    c.fmin[i] = 0;
    c.fmax[i] = 0;
  }
  c.fmin[0] = -1e30f;
  c.fmax[0] = 0.0f;
  contacts.push_back(c);
}

// =============================================================================
// Contact constraint computation
// =============================================================================

void Solver::computeConstraint(Contact &c) {
  Body &bA = bodies[c.bodyA];
  bool bStatic = (c.bodyB == UINT32_MAX);
  Body *pB = bStatic ? nullptr : &bodies[c.bodyB];

  Vec3 rAw = bA.rotation.rotate(c.rA);
  Vec3 rBw = bStatic ? Vec3() : pB->rotation.rotate(c.rB);

  c.JA = Vec6(c.normal, rAw.cross(c.normal));
  c.JB = bStatic ? Vec6() : Vec6(Vec3() - c.normal, Vec3() - rBw.cross(c.normal));

  Vec3 t1, t2;
  if (fabsf(c.normal.y) > 0.9f)
    t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
  else
    t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
  t2 = c.normal.cross(t1);

  c.JAt1 = Vec6(t1, rAw.cross(t1));
  c.JBt1 = bStatic ? Vec6() : Vec6(Vec3() - t1, Vec3() - rBw.cross(t1));
  c.JAt2 = Vec6(t2, rAw.cross(t2));
  c.JBt2 = bStatic ? Vec6() : Vec6(Vec3() - t2, Vec3() - rBw.cross(t2));

  Vec6 dpA(bA.position - bA.initialPosition, bA.deltaWInitial());
  Vec6 dpB;
  if (!bStatic)
    dpB = Vec6(pB->position - pB->initialPosition, pB->deltaWInitial());

  c.C[0] = c.C0[0] * (1.0f - alpha) + dot(c.JA, dpA) + dot(c.JB, dpB);
  c.C[1] = c.C0[1] * (1.0f - alpha) + dot(c.JAt1, dpA) + dot(c.JBt1, dpB);
  c.C[2] = c.C0[2] * (1.0f - alpha) + dot(c.JAt2, dpA) + dot(c.JBt2, dpB);

  float frictionBound = fabsf(c.lambda[0]) * c.friction;
  c.fmax[1] = frictionBound;
  c.fmin[1] = -frictionBound;
  c.fmax[2] = frictionBound;
  c.fmin[2] = -frictionBound;
}

void Solver::computeC0(Contact &c) {
  Body &bA = bodies[c.bodyA];
  bool bStatic = (c.bodyB == UINT32_MAX);
  Body *pB = bStatic ? nullptr : &bodies[c.bodyB];

  Vec3 wA = bA.position + bA.rotation.rotate(c.rA);
  Vec3 wB = bStatic ? c.rB : (pB->position + pB->rotation.rotate(c.rB));

  float rawC0 = (wA - wB).dot(c.normal) - c.depth;

  // Depth-adaptive C0 clamping: for deep penetrations (fast impacts),
  // reduce C0 so that alpha blending does not over-soften the correction.
  // Shallow contacts keep C0 unchanged; deep ones fade C0 toward zero.
  const float c0Threshold = 0.05f;  // 50 mm: only trigger on fast impacts
  const float c0MaxDepth  = 0.20f;  // 200 mm: full fade-out
  if (rawC0 < -c0Threshold) {
    float t = std::clamp(
        (c0MaxDepth + rawC0) / (c0MaxDepth - c0Threshold), 0.0f, 1.0f);
    rawC0 *= t;
  }

  c.C0[0] = rawC0;
  c.C0[1] = 0.0f;
  c.C0[2] = 0.0f;
}

void Solver::warmstart() {
  for (auto &c : contacts) {
    for (int i = 0; i < 3; i++) {
      c.lambda[i] = c.lambda[i] * alpha * gamma;
      c.penalty[i] =
          std::max(PENALTY_MIN, std::min(PENALTY_MAX, c.penalty[i] * gamma));
    }
  }
  // Soft body AVBD warmstart (penalty only, no elastic dual)
  for (auto &sb : softBodies) {
    for (auto &ac : sb.attachments)
      ac.k = std::max(1e3f, std::min(ac.kMax, ac.k * gamma));
    for (auto &kp : sb.pins)
      kp.k = std::max(1e3f, std::min(kp.kMax, kp.k * gamma));
  }
  for (auto &sc : softContacts) {
    sc.k = std::min(1e4f, sc.ke);
  }
}

// =============================================================================
// Soft body creation
// =============================================================================

uint32_t Solver::addSoftBody(const std::vector<Vec3>& vertices,
                             const std::vector<uint32_t>& tets,
                             const std::vector<uint32_t>& tris,
                             float youngsModulus_,
                             float poissonsRatio_,
                             float density_,
                             float damping_,
                             float bendingStiffness_,
                             float thickness_) {
  uint32_t particleStart = (uint32_t)softParticles.size();

  // Compute per-vertex mass from tet volumes (or uniform if no tets)
  std::vector<float> vertexMass(vertices.size(), 0.0f);
  if (!tets.empty()) {
    for (size_t i = 0; i + 3 < tets.size(); i += 4) {
      Vec3 e1 = vertices[tets[i+1]] - vertices[tets[i]];
      Vec3 e2 = vertices[tets[i+2]] - vertices[tets[i]];
      Vec3 e3 = vertices[tets[i+3]] - vertices[tets[i]];
      float vol = fabsf(e1.dot(e2.cross(e3)) / 6.0f);
      float tetMass = vol * density_;
      float perVertex = tetMass / 4.0f;
      vertexMass[tets[i]]   += perVertex;
      vertexMass[tets[i+1]] += perVertex;
      vertexMass[tets[i+2]] += perVertex;
      vertexMass[tets[i+3]] += perVertex;
    }
  } else if (!tris.empty()) {
    // Surface mesh: estimate mass from triangle area × thickness × density
    for (size_t i = 0; i + 2 < tris.size(); i += 3) {
      Vec3 e1 = vertices[tris[i+1]] - vertices[tris[i]];
      Vec3 e2 = vertices[tris[i+2]] - vertices[tris[i]];
      float area = e1.cross(e2).length() * 0.5f;
      float triMass = area * thickness_ * density_;
      float perVertex = triMass / 3.0f;
      vertexMass[tris[i]]   += perVertex;
      vertexMass[tris[i+1]] += perVertex;
      vertexMass[tris[i+2]] += perVertex;
    }
  }

  // Ensure minimum mass
  float minMass = 1e-4f;
  for (auto& m : vertexMass)
    m = std::max(m, minMass);

  // Create particles
  for (size_t i = 0; i < vertices.size(); i++) {
    SoftParticle sp;
    sp.position = vertices[i];
    sp.velocity = Vec3(0, 0, 0);
    sp.prevVelocity = Vec3(0, 0, 0);
    sp.initialPosition = vertices[i];
    sp.predictedPosition = vertices[i];
    sp.mass = vertexMass[i];
    sp.invMass = 1.0f / sp.mass;
    sp.damping = damping_;
    softParticles.push_back(sp);
  }

  // Create SoftBody
  SoftBody sb;
  sb.particleStart = particleStart;
  sb.particleCount = (uint32_t)vertices.size();
  sb.tetrahedra = tets;
  sb.triangles = tris;
  sb.youngsModulus = youngsModulus_;
  sb.poissonsRatio = poissonsRatio_;
  sb.density = density_;
  sb.damping = damping_;
  sb.bendingStiffness = bendingStiffness_;
  sb.thickness = thickness_;

  sb.buildElements(softParticles);

  softBodies.push_back(sb);
  return particleStart;
}

// =============================================================================
// Main solver step
// =============================================================================

void Solver::step(float dt_) {
  dt = dt_;
  float invDt = 1.0f / dt;
  float dt2 = dt * dt;

  warmstart();

  // Step 1: Build adjacency list from joints
  uint32_t nBodies = (uint32_t)bodies.size();
  std::vector<std::vector<uint32_t>> adj(nBodies);
  auto addEdge = [&](uint32_t a, uint32_t b) {
    if (a < nBodies && b < nBodies && a != UINT32_MAX && b != UINT32_MAX) {
      adj[a].push_back(b);
      adj[b].push_back(a);
    }
  };
  for (const auto &j : d6Joints)
    addEdge(j.bodyA, j.bodyB);
  for (const auto &j : gearJoints)
    addEdge(j.bodyA, j.bodyB);
  for (const auto &artic : articulations) {
    for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
      uint32_t child = artic.joints[ji].bodyIndex;
      uint32_t parent = artic.getParentBodyIndex(ji);
      addEdge(parent, child);
    }
  }

  // Step 2: Jacobi propagation of effective mass
  std::vector<float> mEff(nBodies);
  for (uint32_t i = 0; i < nBodies; i++)
    mEff[i] = bodies[i].mass;

  for (int d = 0; d < propagationDepth; d++) {
    std::vector<float> mNext(nBodies);
    for (uint32_t i = 0; i < nBodies; i++) {
      float neighborSum = 0.0f;
      for (uint32_t nb : adj[i])
        neighborSum += mEff[nb];
      mNext[i] = bodies[i].mass + propagationDecay * neighborSum;
    }
    mEff = mNext;
  }

  // Step 3: set penalty floor using propagated effective mass
  for (auto &c : contacts) {
    float augA = mEff[c.bodyA];
    float augB = (c.bodyB != UINT32_MAX) ? mEff[c.bodyB] : 0.0f;
    float massB = (c.bodyB != UINT32_MAX) ? bodies[c.bodyB].mass : 0.0f;

    float effectiveMass, scale;
    if (c.bodyB != UINT32_MAX && massB > 0.0f) {
      effectiveMass = std::max(augA, augB);
      scale = penaltyScaleDynDyn;
    } else {
      effectiveMass = augA;
      scale = penaltyScale;
    }
    float penFloor = std::max(PENALTY_MIN, scale * effectiveMass / dt2);
    for (int i = 0; i < 3; i++)
      c.penalty[i] = std::max(c.penalty[i], penFloor);
  }

  // Compute C0 for alpha blending
  for (auto &c : contacts)
    computeC0(c);

  // Warmstart bodies
  for (auto &body : bodies) {
    if (body.mass <= 0)
      continue;
    body.updateInvInertiaWorld();

    // Inertial prediction
    body.inertialPosition =
        body.position + body.linearVelocity * dt + gravity * dt2;
    Quat angVel(0, body.angularVelocity.x, body.angularVelocity.y,
                body.angularVelocity.z);
    body.inertialRotation =
        (body.rotation + angVel * body.rotation * (0.5f * dt)).normalized();

    // Adaptive warmstarting
    Vec3 accel = (body.linearVelocity - body.prevLinearVelocity) * invDt;
    float gravLen = gravity.length();
    float accelWeight = 0.0f;
    if (gravLen > 1e-6f) {
      Vec3 gravDir = gravity.normalized();
      accelWeight =
          std::max(0.0f, std::min(1.0f, accel.dot(gravDir) / gravLen));
    }

    body.initialPosition = body.position;
    body.initialRotation = body.rotation;

    body.position = body.position + body.linearVelocity * dt +
                    gravity * (accelWeight * dt2);
    body.rotation = body.inertialRotation;
  }

  // Predict soft particles
  uint32_t nSoftParticles = (uint32_t)softParticles.size();
  for (uint32_t i = 0; i < nSoftParticles; i++) {
    SoftParticle &sp = softParticles[i];
    if (sp.invMass <= 0.0f) continue;
    sp.predictedPosition = sp.position + sp.velocity * dt + gravity * dt2;
    sp.initialPosition = sp.position;
    // AVBD elastic proximal warmstart (mirrors PhysX: retain fraction from prior timestep)
    sp.elasticK = sp.elasticK * 0.5f;
    // Adaptive warmstart (same as rigid bodies)
    Vec3 accel = (sp.velocity - sp.prevVelocity) * invDt;
    float gravLen = gravity.length();
    float accelWeight = 0.0f;
    if (gravLen > 1e-6f) {
      Vec3 gravDir = gravity.normalized();
      accelWeight = std::max(0.0f, std::min(1.0f, accel.dot(gravDir) / gravLen));
    }
    sp.position = sp.position + sp.velocity * dt + gravity * (accelWeight * dt2);
  }

  // =========================================================================
  // Compute sweep order (tree-structured for articulation chains)
  // =========================================================================
  std::vector<uint32_t> sweepOrder;
  {
    std::vector<bool> isArticBody(nBodies, false);
    std::vector<uint32_t> articOrder;
    if (useTreeSweep && !articulations.empty()) {
      for (const auto &artic : articulations) {
        for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
          uint32_t bi = artic.joints[ji].bodyIndex;
          if (bi < nBodies && !isArticBody[bi]) {
            isArticBody[bi] = true;
            articOrder.push_back(bi);
          }
        }
      }
    }
    // Non-articulation bodies first (any order)
    for (uint32_t i = 0; i < nBodies; i++) {
      if (!isArticBody[i] && bodies[i].mass > 0)
        sweepOrder.push_back(i);
    }
    // Articulation bodies in tree order (root → leaves)
    for (uint32_t bi : articOrder)
      sweepOrder.push_back(bi);
    // Add remaining dynamic non-artic bodies not yet added
    // (covers all bodies with mass > 0)
  }

  // =========================================================================
  // Anderson Acceleration state (positions only — quaternion mixing is
  // ill-conditioned for AA linear extrapolation)
  // =========================================================================
  int aaDim = (int)nBodies * 3; // 3 pos per body
  int aaCount = 0;
  std::vector<std::vector<float>> aaFHistory, aaXHistory;
  if (useAndersonAccel) {
    aaFHistory.resize(aaWindowSize);
    aaXHistory.resize(aaWindowSize);
    for (int i = 0; i < aaWindowSize; i++) {
      aaFHistory[i].resize(aaDim, 0.0f);
      aaXHistory[i].resize(aaDim, 0.0f);
    }
  }

  // Pack body positions into flat vector (for AA)
  auto packState = [&](std::vector<float> &state) {
    state.resize(aaDim);
    for (uint32_t i = 0; i < nBodies; i++) {
      state[i * 3 + 0] = bodies[i].position.x;
      state[i * 3 + 1] = bodies[i].position.y;
      state[i * 3 + 2] = bodies[i].position.z;
    }
  };
  auto unpackState = [&](const std::vector<float> &state) {
    for (uint32_t i = 0; i < nBodies; i++) {
      if (bodies[i].mass <= 0) continue;
      bodies[i].position.x = state[i * 3 + 0];
      bodies[i].position.y = state[i * 3 + 1];
      bodies[i].position.z = state[i * 3 + 2];
    }
  };

  // =========================================================================
  // Chebyshev state
  // =========================================================================
  float chebyOmega = 1.0f;
  std::vector<Vec3> chebyPrevPos, chebyPrevPrevPos;
  std::vector<Quat> chebyPrevRot, chebyPrevPrevRot;
  if (useChebyshev) {
    chebyPrevPos.resize(nBodies);
    chebyPrevPrevPos.resize(nBodies);
    chebyPrevRot.resize(nBodies);
    chebyPrevPrevRot.resize(nBodies);
    for (uint32_t i = 0; i < nBodies; i++) {
      chebyPrevPos[i] = bodies[i].position;
      chebyPrevPrevPos[i] = bodies[i].position;
      chebyPrevRot[i] = bodies[i].rotation;
      chebyPrevPrevRot[i] = bodies[i].rotation;
    }
  }

  // Convergence history
  convergenceHistory.clear();

  // =========================================================================
  // Rebuild per-particle adjacency (picks up any pins/attachments added
  // after addSoftBody, mirrors PhysX buildAdjacency before solve)
  // =========================================================================
  for (auto &sb : softBodies)
    sb.buildAdjacency();

  // =========================================================================
  // Per-particle contact index (prefix-sum, mirrors PhysX)
  // Avoids O(particles * contacts) scan inside the VBD loop.
  // =========================================================================
  std::vector<uint32_t> scIdxBuf(softContacts.size());
  std::vector<uint32_t> scStart(nSoftParticles + 1, 0);
  std::vector<uint32_t> scCount(nSoftParticles, 0);
  auto buildSoftContactIndex = [&]() {
    for (uint32_t i = 0; i < nSoftParticles; i++) scCount[i] = 0;
    for (uint32_t ci = 0; ci < (uint32_t)softContacts.size(); ci++)
      scCount[softContacts[ci].particleIdx]++;
    scStart[0] = 0;
    for (uint32_t i = 0; i < nSoftParticles; i++)
      scStart[i + 1] = scStart[i] + scCount[i];
    for (uint32_t i = 0; i < nSoftParticles; i++) scCount[i] = 0;
    for (uint32_t ci = 0; ci < (uint32_t)softContacts.size(); ci++) {
      uint32_t pi = softContacts[ci].particleIdx;
      scIdxBuf[scStart[pi] + scCount[pi]] = ci;
      scCount[pi]++;
    }
  };
  buildSoftContactIndex();

  // Pre-compute body-level inertial targets for Newton-style body solve
  float invDtSq = 1.0f / dt2;
  std::vector<Vec3> bodyComPred(softBodies.size());
  std::vector<Vec3> bodyThetaPred(softBodies.size());
  std::vector<Vec3> bodyAccumTheta(softBodies.size());
  for (uint32_t si = 0; si < (uint32_t)softBodies.size(); si++)
  {
    const SoftBody& sb = softBodies[si];
    Vec3 com, comPred;
    float totalMass = 0.0f;
    Vec3 angMom;
    for (uint32_t li = 0; li < sb.particleCount; li++)
    {
      uint32_t pi = sb.particleStart + li;
      if (softParticles[pi].invMass <= 0.0f) continue;
      float m = softParticles[pi].mass;
      com = com + softParticles[pi].position * m;
      comPred = comPred + softParticles[pi].predictedPosition * m;
      totalMass += m;
    }
    if (totalMass > 0.0f)
    {
      float invM = 1.0f / totalMass;
      com = com * invM;
      comPred = comPred * invM;
    }
    bodyComPred[si] = comPred;
    Mat33 bodyI;
    for (uint32_t li = 0; li < sb.particleCount; li++)
    {
      uint32_t pi = sb.particleStart + li;
      if (softParticles[pi].invMass <= 0.0f) continue;
      float m = softParticles[pi].mass;
      Vec3 r = softParticles[pi].position - com;
      float r2 = r.dot(r);
      bodyI = bodyI + (Mat33::diag(r2, r2, r2) - outer(r, r)) * m;
      angMom = angMom + r.cross(softParticles[pi].velocity) * m;
    }
    Vec3 omega = bodyI.inverse() * angMom;
    if (omega.x != omega.x) omega = Vec3();
    bodyThetaPred[si] = omega * dt;
    bodyAccumTheta[si] = Vec3();
  }

  // =========================================================================
  // Chebyshev semi-iterative state for soft particles (mirrors PhysX)
  // =========================================================================
  float softChebyOmega = 1.0f;
  std::vector<Vec3> softChebyPrevPos(nSoftParticles);
  std::vector<Vec3> softChebyPrevPrevPos(nSoftParticles);
  if (useChebyshev) {
    for (uint32_t i = 0; i < nSoftParticles; i++) {
      softChebyPrevPos[i] = softParticles[i].position;
      softChebyPrevPrevPos[i] = softParticles[i].position;
    }
  }

  // =========================================================================
  // Main solver loop
  // =========================================================================
  for (int it = 0; it < iterations; it++) {
    // Save pre-iteration state for AA
    std::vector<float> preState;
    if (useAndersonAccel)
      packState(preState);

    // Save pre-iteration state for Chebyshev
    if (useChebyshev) {
      for (uint32_t i = 0; i < nBodies; i++) {
        chebyPrevPrevPos[i] = chebyPrevPos[i];
        chebyPrevPrevRot[i] = chebyPrevRot[i];
        chebyPrevPos[i] = bodies[i].position;
        chebyPrevRot[i] = bodies[i].rotation;
      }
    }

    // ---- Primal update (per body in sweep order) ----
    bool reverseSweep = useTreeSweep && (it % 2 == 1);
    int nSweep = (int)sweepOrder.size();
    for (int si = 0; si < nSweep; si++) {
      int idx = reverseSweep ? (nSweep - 1 - si) : si;
      uint32_t bi = sweepOrder[idx];
      Body &body = bodies[bi];
      if (body.mass <= 0)
        continue;

      // Check if this body needs full 6x6 solve
      bool bodyNeedsFull6x6 = false;
      if (use3x3Solve) {
        for (const auto &jnt : d6Joints) {
          if (jnt.bodyA == bi || jnt.bodyB == bi) {
            // Any joint with non-trivial coupling needs 6x6
            if (jnt.linearMotion != 0 || jnt.angularMotion != 0x2A) {
              // Not purely spherical -> needs 6x6 for coupled solve
              // Actually, prismatic/revolute have lin-ang coupling
              bodyNeedsFull6x6 = true;
              break;
            }
          }
        }
      }

      Mat66 lhs = body.getMassMatrix() / dt2;
      Vec6 disp(body.position - body.inertialPosition, body.deltaWInertial());
      Vec6 rhs = lhs * disp;

      const float contactBoostFraction = 0.005f;
      float boostFloor = contactBoostFraction * body.mass / dt2;

      // ---- Contact contributions ----
      for (auto &c : contacts) {
        bool isA = (c.bodyA == bi);
        bool isB = (c.bodyB == bi);
        if (!isA && !isB)
          continue;

        computeConstraint(c);

        for (int i = 0; i < 3; i++) {
          Vec6 J = isA ? (i == 0 ? c.JA : (i == 1 ? c.JAt1 : c.JAt2))
                       : (i == 0 ? c.JB : (i == 1 ? c.JBt1 : c.JBt2));
          float pen = std::max(c.penalty[i], boostFloor);
          float f = std::max(c.fmin[i],
                             std::min(c.fmax[i], pen * c.C[i] + c.lambda[i]));
          rhs += J * f;
          lhs += outer(J, J * pen);
        }
      }

      // ---- D6 Joint contributions (unified) ----
      for (const auto &jnt : d6Joints) {
        addD6Contribution(jnt, bi, bodies, dt, lhs, rhs);
      }

      // ---- Articulation contributions (pure AVBD AL constraints) ----
      for (const auto &artic : articulations) {
        for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
          addArticulationContribution(artic, ji, bi, bodies, dt, lhs, rhs);
        }
        for (int mi = 0; mi < (int)artic.mimicJoints.size(); mi++) {
          addMimicJointContribution(artic, mi, bi, bodies, dt, lhs, rhs);
        }
        for (int ti = 0; ti < (int)artic.ikTargets.size(); ti++) {
          addIKTargetContribution(artic, ti, bi, bodies, dt, lhs, rhs);
        }
      }

      // ---- Soft body attachment contributions to rigid body ----
      for (const auto &sb : softBodies) {
        for (const auto &ac : sb.attachments) {
          addAttachmentContribution_rigid(ac, bi, softParticles, bodies, dt, lhs, rhs);
        }
      }

      // ---- Gear Joint contributions ----
      for (auto &gnt : gearJoints) {
        bool isA = (gnt.bodyA == bi);
        bool isB = (gnt.bodyB == bi);
        if (!isA && !isB)
          continue;
        if (gnt.bodyA >= (uint32_t)bodies.size() ||
            gnt.bodyB >= (uint32_t)bodies.size())
          continue;

        Body &bA = bodies[gnt.bodyA];
        Body &bB = bodies[gnt.bodyB];
        if (bA.mass <= 0.f || bB.mass <= 0.f)
          continue;

        Vec3 worldAxisA = bA.rotation.rotate(gnt.axisA);
        Vec3 worldAxisB = bB.rotation.rotate(gnt.axisB);
        Vec3 dwA = bA.deltaWInitial();
        Vec3 dwB = bB.deltaWInitial();
        float C = dwA.dot(worldAxisA) * gnt.gearRatio + dwB.dot(worldAxisB);

        float effectiveRho = std::max(gnt.rho, body.mass / dt2);
        Vec3 J_ang = isA ? (worldAxisA * gnt.gearRatio) : worldAxisB;
        float f = effectiveRho * C + gnt.lambdaGear;

        for (int r = 0; r < 3; r++)
          for (int c2 = 0; c2 < 3; c2++)
            lhs.m[3 + r][3 + c2] +=
                effectiveRho * (&J_ang.x)[r] * (&J_ang.x)[c2];
        for (int r = 0; r < 3; r++)
          rhs.v[3 + r] += f * (&J_ang.x)[r];
      }

      // ---- Solve and apply ----
      bool solve3x3ForBody = use3x3Solve && !bodyNeedsFull6x6;
      if (!solve3x3ForBody) {
        Vec6 delta = solveLDLT(lhs, rhs);
        body.position -= delta.linear();
        Quat dq(0, delta[3], delta[4], delta[5]);
        body.rotation =
            (body.rotation - dq * body.rotation * 0.5f).normalized();
      } else {
        Mat33 Alin, Aang;
        Vec3 rhsLin(rhs[0], rhs[1], rhs[2]);
        Vec3 rhsAng(rhs[3], rhs[4], rhs[5]);
        for (int r = 0; r < 3; r++)
          for (int c = 0; c < 3; c++) {
            Alin.m[r][c] = lhs.m[r][c];
            Aang.m[r][c] = lhs.m[3 + r][3 + c];
          }
        Vec3 deltaPos = Alin.inverse() * rhsLin;
        Vec3 deltaTheta = Aang.inverse() * rhsAng;
        body.position -= deltaPos;
        Quat dq(0, deltaTheta.x, deltaTheta.y, deltaTheta.z);
        body.rotation =
            (body.rotation - dq * body.rotation * 0.5f).normalized();
      }
    }

    // ---- Body-level 6x6 solve for soft bodies (mirrors PhysX) ----
    for (uint32_t si = 0; si < (uint32_t)softBodies.size(); si++)
    {
      const SoftBody& sb = softBodies[si];
      Vec3 com;
      float bodyMass = 0.0f;
      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        if (softParticles[pi].invMass <= 0.0f) continue;
        com = com + softParticles[pi].position * softParticles[pi].mass;
        bodyMass += softParticles[pi].mass;
      }
      if (bodyMass <= 0.0f) continue;
      com = com * (1.0f / bodyMass);

      uint32_t bodyContactCount = 0;
      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        bodyContactCount += scStart[pi + 1] - scStart[pi];
      }
      if (bodyContactCount == 0) continue;

      Mat33 bodyInertia;
      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        if (softParticles[pi].invMass <= 0.0f) continue;
        Vec3 r = softParticles[pi].position - com;
        float r2 = r.dot(r);
        bodyInertia = bodyInertia +
          (Mat33::diag(r2, r2, r2) - outer(r, r)) * softParticles[pi].mass;
      }

      float bodyMassDtSq = bodyMass * invDtSq;
      Mat33 A_ll = Mat33::diag(bodyMassDtSq, bodyMassDtSq, bodyMassDtSq);
      Mat33 A_la, A_al;
      Mat33 A_aa = bodyInertia * invDtSq;
      float reg = 1e-4f * bodyMassDtSq;
      A_aa = A_aa + Mat33::diag(reg, reg, reg);

      Vec3 g_l = (com - bodyComPred[si]) * bodyMassDtSq;
      Vec3 g_a = (bodyInertia * invDtSq) * (bodyAccumTheta[si] - bodyThetaPred[si]);

      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        Vec3 r = softParticles[pi].position - com;
        for (uint32_t k = scStart[pi]; k < scStart[pi + 1]; k++)
        {
          const SoftContact& sc = softContacts[scIdxBuf[k]];
          Vec3 n = sc.normal;
          float violation;
          if (sc.rigidBodyIdx == UINT32_MAX)
            violation = softParticles[pi].position.dot(n);
          else
            violation = (softParticles[pi].position - sc.surfacePoint).dot(n) - sc.margin;

          float pen = sc.k;
          Vec3 rCrossN = r.cross(n);
          A_ll = A_ll + outer(n, n) * pen;
          A_la = A_la + outer(n, rCrossN) * pen;
          A_al = A_al + outer(rCrossN, n) * pen;
          A_aa = A_aa + outer(rCrossN, rCrossN) * pen;

          float f = std::min(0.0f, pen * violation);
          if (f < 0.0f)
          {
            g_l = g_l + n * f;
            g_a = g_a + rCrossN * f;
          }
        }
      }

      Mat33 A_ll_inv = A_ll.inverse();
      Mat33 S = A_aa - A_al.mul(A_ll_inv).mul(A_la);
      Vec3 deltaTheta = S.inverse() * (g_a - A_al.mul(A_ll_inv) * g_l);
      Vec3 deltaPos = A_ll_inv * (g_l - A_la * deltaTheta);

      if (deltaPos.x != deltaPos.x || deltaTheta.x != deltaTheta.x) continue;

      float thetaMag = deltaTheta.length();
      if (thetaMag > 0.5f) deltaTheta = deltaTheta * (0.5f / thetaMag);

      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        if (softParticles[pi].invMass <= 0.0f) continue;
        Vec3 r = softParticles[pi].position - com;
        softParticles[pi].position = softParticles[pi].position - deltaPos - deltaTheta.cross(r);
      }
      bodyAccumTheta[si] = bodyAccumTheta[si] - deltaTheta;
    }

    // ---- AVBD Soft particle primal update (outer/inner loop, mirrors PhysX) ----
    // Snapshot positions as proximal anchor for AVBD elastic term
    for (uint32_t i = 0; i < nSoftParticles; i++)
      softParticles[i].outerPosition = softParticles[i].position;

    // Reset Chebyshev state for each outer iteration when innerIterations > 1
    float softAdaptiveRho = chebyshevSpectralRadius;
    if (innerIterations > 1) {
      softChebyOmega = 1.0f;
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        softChebyPrevPos[i] = softParticles[i].position;
        softChebyPrevPrevPos[i] = softParticles[i].position;
      }
    }

    float softPrevMaxDxSq = 0.0f;

    for (int innerIt = 0; innerIt < innerIterations; innerIt++) {
    float softMaxDxSq = 0.0f;
    for (uint32_t si = 0; si < (uint32_t)softBodies.size(); si++) {
      const SoftBody &sb = softBodies[si];
      for (uint32_t li = 0; li < sb.particleCount; li++) {
        uint32_t spi = sb.particleStart + li;
        SoftParticle &sp = softParticles[spi];
        if (sp.invMass <= 0.0f) continue;

        float mOverDt2 = sp.mass / dt2;
        Vec3 f3 = (sp.predictedPosition - sp.position) * mOverDt2;
        Mat33 H3 = Mat33::diag(mOverDt2, mOverDt2, mOverDt2);

        const SoftBody::ParticleAdjacency& adj = sb.adjacency[li];

        // StVK triangle contributions (adjacency lookup)
        for (const auto &ref : adj.triRefs) {
          Vec3 ft; Mat33 Ht;
          evaluateStVKForceHessian(sb.triElements[ref.index], (int)ref.vOrder,
                                   sb.mu, sb.lambda, softParticles, ft, Ht);
          f3 = f3 + ft; H3 = H3 + Ht;
        }
        // Neo-Hookean tet contributions (adjacency lookup)
        for (const auto &ref : adj.tetRefs) {
          Vec3 ft; Mat33 Ht;
          evaluateNeoHookeanForceHessian(sb.tetElements[ref.index], (int)ref.vOrder,
                                         sb.mu, sb.lambda, softParticles, ft, Ht);
          f3 = f3 + ft; H3 = H3 + Ht;
        }
        // Bending contributions (adjacency lookup)
        for (const auto &ref : adj.bendRefs) {
          Vec3 fb; Mat33 Hb;
          evaluateBendingForceHessian(sb.bendElements[ref.index], (int)ref.vOrder,
                                      sb.bendingStiffness, softParticles, fb, Hb);
          f3 = f3 + fb; H3 = H3 + Hb;
        }
        // Attachment (adjacency lookup)
        for (uint32_t ai : adj.attachmentIndices) {
          Vec3 fa; Mat33 Ha;
          evaluateAttachmentForceHessian_particle(sb.attachments[ai], softParticles, bodies, fa, Ha);
          f3 = f3 + fa; H3 = H3 + Ha;
        }
        // Kinematic pin (adjacency lookup)
        for (uint32_t pi : adj.pinIndices) {
          Vec3 fp; Mat33 Hp;
          evaluatePinForceHessian(sb.pins[pi], softParticles, fp, Hp);
          f3 = f3 + fp; H3 = H3 + Hp;
        }

        // Soft contacts (indexed lookup, mirrors PhysX)
        for (uint32_t k = scStart[spi]; k < scStart[spi + 1]; k++) {
          Vec3 fc; Mat33 Hc;
          evaluateContactForceHessian(softContacts[scIdxBuf[k]], softParticles, fc, Hc);
          f3 = f3 + fc; H3 = H3 + Hc;
        }

        // Stiffness-proportional Rayleigh damping (Newton VBD style):
        // Per-axis damping proportional to elastic stiffness, clamped so no
        // axis gets less damping than mass-proportional (baseline stability).
        if (sp.damping > 0.0f) {
          float dampCoeff = sp.damping * sp.mass * invDt;
          Mat33 H_elastic = H3 - Mat33::diag(mOverDt2, mOverDt2, mOverDt2);
          float he_xx = fmaxf(H_elastic.m[0][0], 0.0f);
          float he_yy = fmaxf(H_elastic.m[1][1], 0.0f);
          float he_zz = fmaxf(H_elastic.m[2][2], 0.0f);
          float trHe = he_xx + he_yy + he_zz;
          float dx, dy, dz;
          if (trHe > 1e-10f) {
            float s = dampCoeff * 3.0f / trHe;
            dx = fmaxf(he_xx * s, dampCoeff);
            dy = fmaxf(he_yy * s, dampCoeff);
            dz = fmaxf(he_zz * s, dampCoeff);
          } else {
            dx = dy = dz = dampCoeff;
          }
          Mat33 H_damp = Mat33::diag(dx, dy, dz);
          f3 = f3 - H_damp * (sp.position - sp.initialPosition);
          H3 = H3 + H_damp;
        }

        // AVBD elastic proximal term: pulls toward outer-iteration anchor
        // to ensure convergence independent of update order (Jacobi-safe)
        if (sp.elasticK > 0.0f) {
          H3 = H3 + Mat33::diag(sp.elasticK, sp.elasticK, sp.elasticK);
          f3 = f3 + (sp.outerPosition - sp.position) * sp.elasticK;
        }

        // Solve 3x3: displacement = H^-1 * f (with clamping, mirrors PhysX)
        Vec3 displacement = H3.inverse() * f3;
        float dxLenSq = displacement.dot(displacement);
        const float maxDx = 1.0f;
        if (dxLenSq > maxDx * maxDx)
          displacement = displacement * (maxDx / sqrtf(dxLenSq));
        if (displacement.x == displacement.x) { // NaN guard
          sp.position = sp.position + displacement;
          if (dxLenSq > softMaxDxSq) softMaxDxSq = dxLenSq;
        }
      }
    }

    // Early termination for soft particles (mirrors PhysX)
    bool softConverged = (softMaxDxSq < 1e-12f);
    if (softConverged) break;

    // Adaptive spectral-radius estimation (mirrors PhysX).
    // Measure GS convergence ratio from iterations 0-1, then use
    // min(measured, user-provided) as the Chebyshev parameter.
    if (innerIt == 0) {
      softPrevMaxDxSq = softMaxDxSq;
    } else if (innerIt == 1 && useChebyshev) {
      if (softPrevMaxDxSq > 1e-20f) {
        float measuredRho = sqrtf(softMaxDxSq / softPrevMaxDxSq);
        softAdaptiveRho = std::min(measuredRho, chebyshevSpectralRadius);
        softAdaptiveRho = std::min(softAdaptiveRho, 0.95f);
      }
      softPrevMaxDxSq = softMaxDxSq;
    }

    // Chebyshev semi-iterative for soft particles (matches PhysX)
    // Use innerIt for iteration index when innerIterations > 1
    int chebyIt = (innerIterations > 1) ? innerIt : it;
    if (useChebyshev && chebyIt >= 2) {
      float rhoSq = softAdaptiveRho * softAdaptiveRho;
      // Use uniform recurrence
      softChebyOmega = 4.0f / (4.0f - rhoSq * softChebyOmega);
      softChebyOmega = std::max(1.0f, std::min(softChebyOmega, 2.0f));

      // Divergence guard: if displacement grew, disable Chebyshev
      if (softPrevMaxDxSq > 1e-20f && softMaxDxSq > softPrevMaxDxSq * 1.1f) {
        softChebyOmega = 1.0f;
        softAdaptiveRho = 0.0f;
      }

      if (softChebyOmega > 1.0f) {
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        if (softParticles[i].invMass <= 0.0f) continue;
        // Skip Chebyshev for particles with active contacts
        // (over-relaxation can push them through surfaces)
        if (scStart[i + 1] > scStart[i]) continue;
        // Also skip for particles with pins or attachments
        bool hasConstraint = false;
        for (const auto &sb : softBodies) {
          uint32_t li = i - sb.particleStart;
          if (li < sb.particleCount) {
            const auto &adj = sb.adjacency[li];
            if (!adj.pinIndices.empty() || !adj.attachmentIndices.empty())
              hasConstraint = true;
            break;
          }
        }
        if (hasConstraint) continue;
        softParticles[i].position = softChebyPrevPrevPos[i] +
            (softParticles[i].position - softChebyPrevPrevPos[i]) * softChebyOmega;
      }
      }
      softPrevMaxDxSq = softMaxDxSq;
    }
    if (useChebyshev) {
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        softChebyPrevPrevPos[i] = softChebyPrevPos[i];
        softChebyPrevPos[i] = softParticles[i].position;
      }
    }
    } // end innerIterations loop

    // Collision projection (Jolt-style hard constraint, mirrors PhysX)
    for (uint32_t ci = 0; ci < (uint32_t)softContacts.size(); ci++) {
      SoftParticle &sp = softParticles[softContacts[ci].particleIdx];
      if (sp.invMass <= 0.0f) continue;
      const SoftContact &sc = softContacts[ci];
      Vec3 n = sc.normal;
      float projPen;
      if (sc.rigidBodyIdx == UINT32_MAX)
        projPen = -(sp.position.dot(n));          // ground plane
      else
        projPen = -(sp.position - sc.surfacePoint).dot(n);  // body surface
      if (projPen > 0.0f) {
        if (sc.rigidBodyIdx != UINT32_MAX && sc.rigidBodyIdx < softBodies.size())
          projPen = std::min(projPen, 0.05f);
        sp.position = sp.position + n * projPen;
      }
    }

    // ---- Dual update ----
    // Contact dual
    for (auto &c : contacts) {
      computeConstraint(c);
      for (int i = 0; i < 3; i++) {
        float oldLambda = c.lambda[i];
        float rawLambda = c.penalty[i] * c.C[i] + oldLambda;
        c.lambda[i] = std::max(c.fmin[i], std::min(c.fmax[i], rawLambda));
        if (c.lambda[i] < c.fmax[i] && c.lambda[i] > c.fmin[i])
          c.penalty[i] =
              std::min(c.penalty[i] + beta * fabsf(c.C[i]), PENALTY_MAX);
      }
    }

    // Joint dual (unified D6)
    {
      const float lambdaDecay = 0.99f;
      for (auto &jnt : d6Joints) {
        updateD6Dual(jnt, bodies, dt, lambdaDecay);
      }
    }

    // Gear joint dual (inside iteration loop, matching PhysX)
    {
      const float lambdaDecay = 0.99f;
      for (auto &gnt : gearJoints) {
        if (gnt.bodyA >= (uint32_t)bodies.size() ||
            gnt.bodyB >= (uint32_t)bodies.size())
          continue;
        Body &bA = bodies[gnt.bodyA];
        Body &bB = bodies[gnt.bodyB];
        if (bA.mass <= 0.f || bB.mass <= 0.f)
          continue;

        float mA = bA.mass, mB = bB.mass;
        float mEff2 = (mA > 0.f && mB > 0.f) ? std::min(mA, mB) : std::max(mA, mB);
        float Mh2 = mEff2 / dt2;
        float admm_step = gnt.rho * gnt.rho / (gnt.rho + Mh2);
        float rhoDual = std::min(Mh2, admm_step);

        Vec3 worldAxisA = bA.rotation.rotate(gnt.axisA);
        Vec3 worldAxisB = bB.rotation.rotate(gnt.axisB);
        Vec3 dwA = bA.deltaWInitial();
        Vec3 dwB = bB.deltaWInitial();
        float C = dwA.dot(worldAxisA) * gnt.gearRatio + dwB.dot(worldAxisB);

        gnt.lambdaGear = gnt.lambdaGear * lambdaDecay + rhoDual * C;
      }
    }

    // Articulation dual
    {
      const float lambdaDecay = 0.99f;
      for (auto &artic : articulations) {
        for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
          updateArticulationDual(artic, ji, bodies, dt, lambdaDecay);
        }
        for (int mi = 0; mi < (int)artic.mimicJoints.size(); mi++) {
          updateMimicDual(artic, mi, bodies, dt, lambdaDecay);
        }
        for (int ti = 0; ti < (int)artic.ikTargets.size(); ti++) {
          updateIKTargetDual(artic, ti, bodies, dt, lambdaDecay);
        }
      }
    }

    // Soft body AVBD dual update (penalty growth + elastic proximal)
    {
      for (auto &sb : softBodies) {
        for (auto &ac : sb.attachments)
          updateAttachmentDual(ac, softParticles, bodies, beta);
        for (auto &kp : sb.pins)
          updatePinDual(kp, softParticles, beta);
      }
      for (auto &sc : softContacts)
        updateSoftContactDual(sc, softParticles, beta);

      // AVBD elastic proximal dual update: increase proximal weight
      // proportional to displacement from the outer-iteration anchor
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        SoftParticle &sp = softParticles[i];
        if (sp.invMass <= 0.0f) continue;
        float disp = (sp.position - sp.outerPosition).length();
        sp.elasticK = std::min(sp.elasticK + beta * disp, sp.elasticKMax);
      }
    }

    // ===================================================================
    // Anderson Acceleration (Type I, safeguarded)
    // ===================================================================
    if (useAndersonAccel && it >= 0) {
      std::vector<float> postState;
      packState(postState);

      // Compute residual f_k = g(x_k) - x_k
      std::vector<float> fk(aaDim);
      for (int i = 0; i < aaDim; i++)
        fk[i] = postState[i] - preState[i];

      // Store in circular buffer
      int slot = aaCount % aaWindowSize;
      aaXHistory[slot] = preState;
      aaFHistory[slot] = fk;
      aaCount++;

      int mk = std::min(aaCount - 1, aaWindowSize); // number of differences
      if (mk >= 1) {
        // Build ΔF matrix columns: ΔF_j = f_k - f_{k-j}
        // We have: slot = most recent, (slot-1+ws)%ws = one before, etc.
        std::vector<std::vector<float>> deltaF(mk, std::vector<float>(aaDim));
        std::vector<std::vector<float>> deltaX(mk, std::vector<float>(aaDim));
        for (int j = 0; j < mk; j++) {
          int prevSlot = (slot - 1 - j + aaWindowSize * 2) % aaWindowSize;
          for (int i = 0; i < aaDim; i++) {
            deltaF[j][i] = fk[i] - aaFHistory[prevSlot][i];
            deltaX[j][i] = preState[i] - aaXHistory[prevSlot][i];
          }
        }

        // Solve normal equations: (ΔF^T ΔF) θ = ΔF^T f_k
        std::vector<float> FTF(mk * mk, 0.0f);
        std::vector<float> FTf(mk, 0.0f);
        for (int i = 0; i < mk; i++) {
          for (int j = 0; j <= i; j++) {
            float dot = 0;
            for (int d = 0; d < aaDim; d++)
              dot += deltaF[i][d] * deltaF[j][d];
            FTF[i * mk + j] = dot;
            FTF[j * mk + i] = dot;
          }
          float dot = 0;
          for (int d = 0; d < aaDim; d++)
            dot += deltaF[i][d] * fk[d];
          FTf[i] = dot;
        }

        // Tikhonov regularization
        float maxDiag = 0;
        for (int i = 0; i < mk; i++)
          maxDiag = std::max(maxDiag, FTF[i * mk + i]);
        float reg = 1e-8f * std::max(maxDiag, 1.0f);
        for (int i = 0; i < mk; i++)
          FTF[i * mk + i] += reg;

        // Gaussian elimination (mk ≤ 3, tiny system)
        std::vector<float> theta(mk, 0.0f);
        for (int i = 0; i < mk; i++) {
          float pivot = FTF[i * mk + i];
          if (std::fabs(pivot) < 1e-15f) continue;
          for (int j = i + 1; j < mk; j++) {
            float factor = FTF[j * mk + i] / pivot;
            for (int k = i + 1; k < mk; k++)
              FTF[j * mk + k] -= factor * FTF[i * mk + k];
            FTf[j] -= factor * FTf[i];
          }
        }
        for (int i = mk - 1; i >= 0; i--) {
          float sum = FTf[i];
          for (int j = i + 1; j < mk; j++)
            sum -= FTF[i * mk + j] * theta[j];
          float pivot = FTF[i * mk + i];
          theta[i] = (std::fabs(pivot) > 1e-15f) ? (sum / pivot) : 0.0f;
        }

        // Compute AA iterate: x_{k+1} = g(x_k) - ΔG * θ
        //   where ΔG_j = (x_k + f_k) - (x_{k-j} + f_{k-j}) = ΔX_j + ΔF_j
        std::vector<float> aaState(aaDim);
        for (int i = 0; i < aaDim; i++) {
          float correction = 0;
          for (int j = 0; j < mk; j++)
            correction += theta[j] * (deltaX[j][i] + deltaF[j][i]);
          aaState[i] = postState[i] - correction;
        }

        // Safeguard: measure actual constraint violation before/after AA
        float violBefore = 0;
        for (auto &artic : articulations)
          violBefore = std::max(violBefore, artic.computeMaxPositionViolation(bodies));

        // Tentatively apply AA state
        unpackState(aaState);

        float violAfter = 0;
        for (auto &artic : articulations)
          violAfter = std::max(violAfter, artic.computeMaxPositionViolation(bodies));

        // Reject if AA increased violation
        if (violAfter > violBefore) {
          unpackState(postState);
        }
      }
    }

    // ===================================================================
    // Chebyshev semi-iterative position relaxation
    //
    // x_{k+1}^cheb = x_{k-1} + omega_k * (x_{k+1}^GS - x_{k-1})
    // omega follows the Chebyshev recurrence for spectral radius rho.
    // ===================================================================
    if (useChebyshev && it >= 2) {
      // Chebyshev omega recurrence
      float rhoSq = chebyshevSpectralRadius * chebyshevSpectralRadius;
      if (it == 2) {
        chebyOmega = 2.0f / (2.0f - rhoSq);
      } else {
        chebyOmega = 1.0f / (1.0f - rhoSq * chebyOmega / 4.0f);
      }
      chebyOmega = std::max(1.0f, std::min(chebyOmega, 2.0f)); // safety clamp

      for (uint32_t i = 0; i < nBodies; i++) {
        if (bodies[i].mass <= 0) continue;
        // Relaxed position: x_new = x_{k-1} + omega * (x_GS - x_{k-1})
        bodies[i].position = chebyPrevPrevPos[i] +
            (bodies[i].position - chebyPrevPrevPos[i]) * chebyOmega;
        // For rotation: use SLERP-like interpolation via quaternion blend
        // Approximate: q_new ≈ normalize(q_{k-1} + omega * (q_GS - q_{k-1}))
        Quat qPrev = chebyPrevPrevRot[i];
        Quat qCur = bodies[i].rotation;
        float dotQ = qPrev.w * qCur.w + qPrev.x * qCur.x +
                     qPrev.y * qCur.y + qPrev.z * qCur.z;
        if (dotQ < 0) qCur = qCur * (-1.0f);
        Quat qBlend;
        qBlend.w = qPrev.w + chebyOmega * (qCur.w - qPrev.w);
        qBlend.x = qPrev.x + chebyOmega * (qCur.x - qPrev.x);
        qBlend.y = qPrev.y + chebyOmega * (qCur.y - qPrev.y);
        qBlend.z = qPrev.z + chebyOmega * (qCur.z - qPrev.z);
        bodies[i].rotation = qBlend.normalized();
      }
    }

    // ===================================================================
    // Convergence tracking
    // ===================================================================
    if (!articulations.empty()) {
      float maxViol = 0;
      for (const auto &artic : articulations)
        maxViol = std::max(maxViol, artic.computeMaxPositionViolation(bodies));
      convergenceHistory.push_back(maxViol);
    }
  } // end iteration loop

  // =========================================================================
  // Post-solve motor drives (matches PhysX Stage 5b)
  //
  // After all constraint iterations converge, directly apply clamped torque
  // to bodies. This avoids coupling the motor with position/gear constraints
  // in the Hessian which causes ADMM oscillation.
  // =========================================================================
  for (auto &jnt : d6Joints) {
    if (!jnt.motorEnabled || jnt.motorMaxForce <= 0.0f)
      continue;

    bool isAStatic = (jnt.bodyA == UINT32_MAX || jnt.bodyA >= (uint32_t)bodies.size());
    bool isBStatic = (jnt.bodyB == UINT32_MAX || jnt.bodyB >= (uint32_t)bodies.size());
    if (isAStatic && isBStatic)
      continue;

    // Twist axis = X-axis of localFrameA in world space
    Quat rotA_m = isAStatic ? Quat() : bodies[jnt.bodyA].rotation;
    Vec3 worldAxis = (rotA_m * jnt.localFrameA).rotate(Vec3(1, 0, 0));
    float axLen = worldAxis.length();
    if (axLen > 1e-6f) worldAxis = worldAxis * (1.0f / axLen);

    // Apply motor to body B
    if (!isBStatic) {
      Body &bodyB = bodies[jnt.bodyB];

      // Current angular velocity from position-level solver
      Quat deltaQB = bodyB.rotation * bodyB.initialRotation.conjugate();
      if (deltaQB.w < 0.0f) deltaQB = deltaQB * (-1.0f);
      Vec3 currentAngVel = Vec3(deltaQB.x, deltaQB.y, deltaQB.z) * (2.0f * invDt);
      float currentAxisVel = currentAngVel.dot(worldAxis);

      float velocityError = jnt.motorTargetVelocity - currentAxisVel;

      // Effective inertia along twist axis
      Vec3 invITimesAxis = bodyB.invInertiaWorld * worldAxis;
      float effectiveInvInertia = worldAxis.dot(invITimesAxis);
      if (effectiveInvInertia < 1e-10f)
        continue;

      float effectiveInertia = 1.0f / effectiveInvInertia;
      float requiredTorque = effectiveInertia * velocityError * invDt;
      float clampedTorque = std::max(-jnt.motorMaxForce,
                                     std::min(jnt.motorMaxForce, requiredTorque));
      float angularAccel = clampedTorque * effectiveInvInertia;
      float deltaAngle = angularAccel * dt2;

      // Apply rotation to body B (axis-angle -> quaternion)
      float ha = deltaAngle * 0.5f;
      float sinHa = sinf(ha), cosHa = cosf(ha);
      Quat dRot(cosHa, worldAxis.x * sinHa, worldAxis.y * sinHa,
                worldAxis.z * sinHa);
      bodyB.rotation = (dRot * bodyB.rotation).normalized();
    }

    // Apply opposite rotation to body A if dynamic
    if (!isAStatic) {
      Body &bodyA = bodies[jnt.bodyA];

      Quat deltaQA = bodyA.rotation * bodyA.initialRotation.conjugate();
      if (deltaQA.w < 0.0f) deltaQA = deltaQA * (-1.0f);
      Vec3 currentAngVelA = Vec3(deltaQA.x, deltaQA.y, deltaQA.z) * (2.0f * invDt);
      float currentAxisVelA = currentAngVelA.dot(worldAxis);
      float velocityErrorA = -jnt.motorTargetVelocity - currentAxisVelA;

      Vec3 invITimesAxisA = bodyA.invInertiaWorld * worldAxis;
      float effectiveInvInertiaA = worldAxis.dot(invITimesAxisA);
      if (effectiveInvInertiaA > 1e-10f) {
        float effectiveInertiaA = 1.0f / effectiveInvInertiaA;
        float requiredTorqueA = effectiveInertiaA * velocityErrorA * invDt;
        float clampedTorqueA = std::max(-jnt.motorMaxForce,
                                        std::min(jnt.motorMaxForce, requiredTorqueA));
        float deltaAngleA = clampedTorqueA * effectiveInvInertiaA * dt2;
        float haA = deltaAngleA * 0.5f;
        float sinHaA = sinf(haA), cosHaA = cosf(haA);
        Quat dRotA(cosHaA, worldAxis.x * sinHaA, worldAxis.y * sinHaA,
                   worldAxis.z * sinHaA);
        bodyA.rotation = (dRotA * bodyA.rotation).normalized();
      }
    }
  }

  // Update velocities
  for (auto &body : bodies) {
    if (body.mass <= 0)
      continue;
    body.prevLinearVelocity = body.linearVelocity;
    body.linearVelocity = (body.position - body.initialPosition) * invDt;
    Quat dq = body.rotation * body.initialRotation.conjugate();
    if (dq.w < 0)
      dq = -dq;
    body.angularVelocity = Vec3(dq.x, dq.y, dq.z) * (2.0f * invDt);

    // Per-body damping
    if (body.linearDamping > 0.0f) {
      float decay = std::max(0.0f, 1.0f - body.linearDamping * dt);
      body.linearVelocity = body.linearVelocity * decay;
    }
    if (body.angularDamping > 0.0f) {
      float decay = std::max(0.0f, 1.0f - body.angularDamping * dt);
      body.angularVelocity = body.angularVelocity * decay;
    }

    // Velocity clamping
    float linSpeed = body.linearVelocity.length();
    if (linSpeed > body.maxLinearVelocity) {
      body.linearVelocity = body.linearVelocity * (body.maxLinearVelocity / linSpeed);
    }
    float angSpeed = body.angularVelocity.length();
    if (angSpeed > body.maxAngularVelocity) {
      body.angularVelocity = body.angularVelocity * (body.maxAngularVelocity / angSpeed);
    }
  }

  // Update soft particle velocities
  for (auto &sp : softParticles) {
    if (sp.invMass <= 0.0f) continue;
    sp.prevVelocity = sp.velocity;
    sp.velocity = (sp.position - sp.initialPosition) * invDt;
    if (sp.damping > 0.0f) {
      float decay = std::max(0.0f, 1.0f - sp.damping * dt);
      sp.velocity = sp.velocity * decay;
    }
  }

}

} // namespace AvbdRef
