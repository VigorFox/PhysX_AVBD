#include "avbd_solver.h"
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

  c.C0[0] = (wA - wB).dot(c.normal) - c.depth;
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

  // =========================================================================
  // Main solver loop
  // =========================================================================
  for (int it = 0; it < iterations; it++) {
    // ---- Primal update (per body) ----
    for (uint32_t bi = 0; bi < (uint32_t)bodies.size(); bi++) {
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
  }
}

} // namespace AvbdRef
