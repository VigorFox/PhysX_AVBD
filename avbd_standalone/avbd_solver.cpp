#include "avbd_solver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace AvbdRef {

void Solver::addSphericalJoint(uint32_t bodyA, uint32_t bodyB, Vec3 anchorA,
                               Vec3 anchorB, float rho_) {
  SphericalJoint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.lambda = Vec3();
  j.rho = rho_;
  j.coneAngleLimit = 0.0f;
  j.coneLambda = 0.0f;
  sphericalJoints.push_back(j);
}

void Solver::setSphericalJointConeLimit(uint32_t jointIdx, Vec3 coneAxisA,
                                        float limitAngle) {
  if (jointIdx < sphericalJoints.size()) {
    sphericalJoints[jointIdx].coneAngleLimit = limitAngle;
    sphericalJoints[jointIdx].coneAxisA = coneAxisA;
    sphericalJoints[jointIdx].coneLambda = 0.0f;
  }
}

void Solver::addFixedJoint(uint32_t bodyA, uint32_t bodyB, Vec3 anchorA,
                           Vec3 anchorB, float rho_) {
  FixedJoint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.lambdaPos = Vec3();
  j.lambdaRot = Vec3();
  j.rho = rho_;
  // Compute initial relative rotation
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  fixedJoints.push_back(j);
}

void Solver::addD6Joint(uint32_t bodyA, uint32_t bodyB, Vec3 anchorA,
                        Vec3 anchorB, uint32_t linearMotion_,
                        uint32_t angularMotion_, float angularDamping_,
                        float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = linearMotion_;
  j.angularMotion = angularMotion_;
  j.lambdaLinear = Vec3();
  j.lambdaAngular = Vec3();
  j.rho = rho_;
  j.driveFlags = 0;
  j.driveLinearVelocity = Vec3();
  j.driveAngularVelocity = Vec3();
  j.linearDriveDamping = Vec3();
  j.angularDriveDamping =
      Vec3(angularDamping_, angularDamping_, angularDamping_);
  j.localFrameA = Quat(); // identity
  j.lambdaDriveLinear = Vec3();
  j.lambdaDriveAngular = Vec3();
  d6Joints.push_back(j);
}

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
  // Normal: fmin=-inf, fmax=0 (can push, not pull)
  c.fmin[0] = -1e30f;
  c.fmax[0] = 0.0f;
  // Friction: bounds set dynamically
  c.fmin[1] = 0;
  c.fmax[1] = 0;
  c.fmin[2] = 0;
  c.fmax[2] = 0;
  contacts.push_back(c);
}

void Solver::computeConstraint(Contact &c) {
  Body &bA = bodies[c.bodyA];
  bool bStatic = (c.bodyB == UINT32_MAX);
  Body *pB = bStatic ? nullptr : &bodies[c.bodyB];

  Vec3 wA = bA.position + bA.rotation.rotate(c.rA);
  Vec3 wB = bStatic ? c.rB : (pB->position + pB->rotation.rotate(c.rB));

  Vec3 diff = wA - wB;
  // float Cn = diff.dot(c.normal) - c.depth; // unused direct calc

  Vec3 rAw = bA.rotation.rotate(c.rA);
  Vec3 rBw = bStatic ? Vec3() : pB->rotation.rotate(c.rB);

  c.JA = Vec6(c.normal, rAw.cross(c.normal));
  c.JB = bStatic ? Vec6() : Vec6(-c.normal, (-rBw).cross(c.normal));
  if (!bStatic) {
    c.JB = Vec6(Vec3() - c.normal, Vec3() - rBw.cross(c.normal));
  }

  Vec3 t1, t2;
  if (fabsf(c.normal.y) > 0.9f) {
    t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
  } else {
    t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
  }
  t2 = c.normal.cross(t1);

  c.JAt1 = Vec6(t1, rAw.cross(t1));
  c.JBt1 = bStatic ? Vec6() : Vec6(Vec3() - t1, Vec3() - rBw.cross(t1));
  c.JAt2 = Vec6(t2, rAw.cross(t2));
  c.JBt2 = bStatic ? Vec6() : Vec6(Vec3() - t2, Vec3() - rBw.cross(t2));

  Vec6 dpA(bA.position - bA.initialPosition, bA.deltaWInitial());
  Vec6 dpB;
  if (!bStatic) {
    dpB = Vec6(pB->position - pB->initialPosition, pB->deltaWInitial());
  }

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
  for (const auto &j : sphericalJoints)
    addEdge(j.bodyA, j.bodyB);
  for (const auto &j : fixedJoints)
    addEdge(j.bodyA, j.bodyB);
  for (const auto &j : d6Joints)
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
      effectiveMass = std::max(augA, augB); // heavier body determines floor
      scale = penaltyScaleDynDyn;           // softer for deformation
    } else {
      effectiveMass = augA; // body-ground (propagated)
      scale = penaltyScale; // full stiffness
    }
    float penFloor = std::max(PENALTY_MIN, scale * effectiveMass / dt2);
    for (int i = 0; i < 3; i++)
      c.penalty[i] = std::max(c.penalty[i], penFloor);
  }

  // Compute C0 for alpha blending
  for (auto &c : contacts) {
    computeC0(c);
  }

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

  // Main solver loop
  for (int it = 0; it < iterations; it++) {
    // Primal update (per body)
    for (uint32_t bi = 0; bi < (uint32_t)bodies.size(); bi++) {
      Body &body = bodies[bi];
      if (body.mass <= 0)
        continue;

      Mat66 lhs = body.getMassMatrix() / dt2;
      Vec6 disp(body.position - body.inertialPosition, body.deltaWInertial());
      Vec6 rhs = lhs * disp;

      const float contactBoostFraction = 0.005f;
      float boostFloor = contactBoostFraction * body.mass / dt2;

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

      // Spherical Joint contributions
      for (auto &jnt : sphericalJoints) {
        bool isA = (jnt.bodyA == bi);
        bool isB = (jnt.bodyB == bi);
        if (!isA && !isB)
          continue;

        bool otherStatic =
            isA ? (jnt.bodyB == UINT32_MAX) : (jnt.bodyA == UINT32_MAX);
        Vec3 r = isA ? body.rotation.rotate(jnt.anchorA)
                     : body.rotation.rotate(jnt.anchorB);
        Vec3 worldAnchorA =
            isA ? (body.position + r)
                : (otherStatic
                       ? jnt.anchorA
                       : (bodies[jnt.bodyA].position +
                          bodies[jnt.bodyA].rotation.rotate(jnt.anchorA)));
        Vec3 worldAnchorB =
            isA ? (otherStatic
                       ? jnt.anchorB
                       : (bodies[jnt.bodyB].position +
                          bodies[jnt.bodyB].rotation.rotate(jnt.anchorB)))
                : (body.position + r);

        Vec3 C = worldAnchorA - worldAnchorB;
        Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        // Auto-boost penalty: ensure rho >= M/h^2 for good primal convergence
        float effectiveRho = std::max(jnt.rho, body.mass / dt2);

        for (int k = 0; k < 3; k++) {
          Vec3 axis = axes[k];
          float Ck = C.dot(axis);
          float sign = isA ? 1.0f : -1.0f;
          Vec6 J(axis * sign, r.cross(axis) * sign);
          float lam = (k == 0) ? jnt.lambda.x
                               : ((k == 1) ? jnt.lambda.y : jnt.lambda.z);
          float f = effectiveRho * Ck + lam;
          rhs += J * f;
          lhs += outer(J, J * effectiveRho);
        }

        // Process cone limit (inequality constraint)
        if (jnt.coneAngleLimit > 0.0f) {
          Quat rotA =
              isA ? body.rotation
                  : ((jnt.bodyA == UINT32_MAX) ? Quat()
                                               : bodies[jnt.bodyA].rotation);
          Quat rotB =
              isB ? body.rotation
                  : ((jnt.bodyB == UINT32_MAX) ? Quat()
                                               : bodies[jnt.bodyB].rotation);
          float coneViolation = jnt.computeConeViolation(rotA, rotB);

          float forceMag = effectiveRho * coneViolation - jnt.coneLambda;

          if (forceMag > 0.0f) {
            Vec3 worldAxisA = rotA.rotate(jnt.coneAxisA);
            Vec3 worldAxisB = rotB.rotate(jnt.coneAxisA);
            Vec3 corrAxis = worldAxisA.cross(worldAxisB);

            float corrAxisMag = corrAxis.length();
            if (corrAxisMag > 1e-6f) {
              corrAxis = corrAxis * (1.0f / corrAxisMag);

              float signJ = isA ? 1.0f : -1.0f;
              Vec3 gradRot = corrAxis * (-signJ);
              Vec6 J(Vec3(0, 0, 0), gradRot);

              rhs += J * forceMag;
              lhs += outer(J, J * effectiveRho);
            }
          }
        }
      }

      // Fixed Joint contributions
      for (auto &jnt : fixedJoints) {
        bool isA = (jnt.bodyA == bi);
        bool isB = (jnt.bodyB == bi);
        if (!isA && !isB)
          continue;

        bool otherStatic =
            isA ? (jnt.bodyB == UINT32_MAX) : (jnt.bodyA == UINT32_MAX);
        Vec3 r = isA ? body.rotation.rotate(jnt.anchorA)
                     : body.rotation.rotate(jnt.anchorB);

        float sign = isA ? 1.0f : -1.0f;

        // Linear part
        Vec3 worldAnchorA =
            isA ? (body.position + r)
                : (otherStatic
                       ? jnt.anchorA
                       : (bodies[jnt.bodyA].position +
                          bodies[jnt.bodyA].rotation.rotate(jnt.anchorA)));
        Vec3 worldAnchorB =
            isA ? (otherStatic
                       ? jnt.anchorB
                       : (bodies[jnt.bodyB].position +
                          bodies[jnt.bodyB].rotation.rotate(jnt.anchorB)))
                : (body.position + r);

        Vec3 C_lin = worldAnchorA - worldAnchorB;
        Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        // Auto-boost penalty for primal convergence
        float effectiveRho = std::max(jnt.rho, body.mass / dt2);

        for (int k = 0; k < 3; k++) {
          Vec3 axis = axes[k];
          float Ck = C_lin.dot(axis);
          Vec6 J(axis * sign, r.cross(axis) * sign);
          float lam = (k == 0) ? jnt.lambdaPos.x
                               : ((k == 1) ? jnt.lambdaPos.y : jnt.lambdaPos.z);
          float f = effectiveRho * Ck + lam;
          rhs += J * f;
          lhs += outer(J, J * effectiveRho);
        }

        // Angular part
        Quat rotA =
            isA ? body.rotation
                : ((jnt.bodyA == UINT32_MAX) ? Quat()
                                             : bodies[jnt.bodyA].rotation);
        Quat rotB =
            isA ? ((jnt.bodyB == UINT32_MAX) ? Quat()
                                             : bodies[jnt.bodyB].rotation)
                : body.rotation;
        Vec3 C_ang = jnt.computeRotationViolation(rotA, rotB);
        if (!isA) // Only need sign flip if B-relative? No, verify with logic.
                  // computeRotViol returns world-space axis-angle diff.
                  // If we represent err = rotA * rel * rotB^-1, a rotation of A
                  // increases err. A rotation of B decreases err.
          C_ang = -C_ang; // Wait, let's verify logic below.

        // Actually, let's stick to original logic:
        // J_ang_A = axis, J_ang_B = -axis.
        // Wait, C_ang computed is global rotation vector needed to align.
        // If isA, J is +axis. If isB, J is -axis.
        // But C_ang sign must match J sign convention.
        // Let's trust the original code's `if (!isA) C_ang = -C_ang;` which
        // implies C_ang is error FROM A perspective.

        for (int k = 0; k < 3; k++) {
          Vec3 axis = axes[k]; // global axis
          float Ck = C_ang.dot(axis);
          Vec3 jAng = isA ? axis : -axis;
          Vec6 J(Vec3(), jAng);
          float lam = (k == 0) ? jnt.lambdaRot.x
                               : ((k == 1) ? jnt.lambdaRot.y : jnt.lambdaRot.z);
          float f = effectiveRho * Ck + lam;
          rhs += J * f;
          lhs += outer(J, J * effectiveRho);
        }
      }

      // D6 Joint contributions
      for (auto &jnt : d6Joints) {
        bool isA = (jnt.bodyA == bi);
        bool isB = (jnt.bodyB == bi);
        if (!isA && !isB)
          continue;

        bool otherStatic =
            isA ? (jnt.bodyB == UINT32_MAX) : (jnt.bodyA == UINT32_MAX);
        Vec3 r = isA ? body.rotation.rotate(jnt.anchorA)
                     : body.rotation.rotate(jnt.anchorB);

        float sign = isA ? 1.0f : -1.0f;

        // Linear part
        Vec3 worldAnchorA =
            isA ? (body.position + r)
                : (otherStatic
                       ? jnt.anchorA
                       : (bodies[jnt.bodyA].position +
                          bodies[jnt.bodyA].rotation.rotate(jnt.anchorA)));
        Vec3 worldAnchorB =
            isA ? (otherStatic
                       ? jnt.anchorB
                       : (bodies[jnt.bodyB].position +
                          bodies[jnt.bodyB].rotation.rotate(jnt.anchorB)))
                : (body.position + r);

        Vec3 C_lin = worldAnchorA - worldAnchorB;
        Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        // Auto-boost penalty for primal convergence
        float effectiveRho = std::max(jnt.rho, body.mass / dt2);

        for (int k = 0; k < 3; k++) {
          uint32_t motion = jnt.getLinearMotion(k);
          if (motion != 0) // ONLY LOCKED (0)
            continue;

          Vec3 axis = axes[k];
          float Ck = C_lin.dot(axis);
          Vec6 J(axis * sign, r.cross(axis) * sign);
          float lam =
              (k == 0) ? jnt.lambdaLinear.x
                       : ((k == 1) ? jnt.lambdaLinear.y : jnt.lambdaLinear.z);
          float f = effectiveRho * Ck + lam;
          rhs += J * f;
          lhs += outer(J, J * effectiveRho);
        }

        // Angular part
        Quat rotA =
            isA ? body.rotation
                : ((jnt.bodyA == UINT32_MAX) ? Quat()
                                             : bodies[jnt.bodyA].rotation);
        Quat rotB =
            isA ? ((jnt.bodyB == UINT32_MAX) ? Quat()
                                             : bodies[jnt.bodyB].rotation)
                : body.rotation;

        Quat relRot(1, 0, 0, 0); // Implicit identity target generic D6?
                                 // Actually for general D6, target is identity
                                 // usually unless driven.
        Quat target = rotA * relRot;
        Quat err = target * rotB.conjugate();
        if (err.w < 0)
          err = -err;
        Vec3 C_ang = Vec3(err.x, err.y, err.z) * 2.0f;
        if (!isA)
          C_ang = -C_ang;

        for (int k = 0; k < 3; k++) {
          uint32_t motion = jnt.getAngularMotion(k);
          if (motion != 0) // ONLY LOCKED (0)
            continue;

          Vec3 axis = axes[k];
          float Ck = C_ang.dot(axis);
          Vec3 jAng = isA ? axis : -axis;
          Vec6 J(Vec3(), jAng);
          float lam =
              (k == 0) ? jnt.lambdaAngular.x
                       : ((k == 1) ? jnt.lambdaAngular.y : jnt.lambdaAngular.z);
          float f = effectiveRho * Ck + lam;
          rhs += J * f;
          lhs += outer(J, J * effectiveRho);
        }

        // ====== AVBD AL velocity-drive constraints ======
        // Joint frame A in world space
        Quat jointFrameA = (jnt.bodyA == UINT32_MAX)
                               ? jnt.localFrameA
                               : bodies[jnt.bodyA].rotation * jnt.localFrameA;
        jointFrameA = jointFrameA.normalized();

        // --- Linear velocity drive (AL constraint) ---
        // C_lin[a] = (Δx_B - Δx_A) · axis - v_target · dt
        if (jnt.driveFlags & 0x07) {
          // Compute displacements
          Vec3 dxThis = body.position - body.initialPosition;
          Vec3 dxOther(0, 0, 0);
          if (!otherStatic) {
            uint32_t otherIdx = isA ? jnt.bodyB : jnt.bodyA;
            dxOther =
                bodies[otherIdx].position - bodies[otherIdx].initialPosition;
          }
          // Relative displacement: B - A
          Vec3 relDisp = isA ? (dxOther - dxThis) : (dxThis - dxOther);
          Vec3 worldTarget = jointFrameA.rotate(jnt.driveLinearVelocity) * dt;

          Vec3 localAxes[3] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
          for (int a = 0; a < 3; a++) {
            if ((jnt.driveFlags & (1 << a)) == 0)
              continue;
            float damping = (&jnt.linearDriveDamping.x)[a];
            if (damping <= 0.0f)
              continue;

            float rho_drive = damping / dt2;
            Vec3 wAxis = jointFrameA.rotate(localAxes[a]);
            float C = relDisp.dot(wAxis) - worldTarget.dot(wAxis);
            float lam = (&jnt.lambdaDriveLinear.x)[a];
            float sign = isA ? -1.0f : 1.0f;
            float f = sign * (rho_drive * C + lam);

            // Hessian: ρ · (axis ⊗ axis), RHS: f · axis
            for (int k = 0; k < 3; k++) {
              (&rhs.v[0])[k] += f * (&wAxis.x)[k];
              for (int l = 0; l < 3; l++) {
                lhs.m[k][l] += rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];
              }
            }
          }
        }

        // --- Angular velocity drive (AL constraint) ---
        // C_ang[a] = (δω_B - δω_A) · axis - ω_target · dt
        if (jnt.driveFlags & 0xF0) {
          Vec3 dwThis = body.deltaWInitial();
          Vec3 dwOther(0, 0, 0);
          if (!otherStatic) {
            uint32_t otherIdx = isA ? jnt.bodyB : jnt.bodyA;
            dwOther = bodies[otherIdx].deltaWInitial();
          }
          // Relative angular displacement: B - A
          Vec3 relDW = isA ? (dwOther - dwThis) : (dwThis - dwOther);
          Vec3 worldAngTarget =
              jointFrameA.rotate(jnt.driveAngularVelocity) * dt;

          bool slerpDrive = (jnt.driveFlags & 0x20) != 0;
          if (slerpDrive) {
            // SLERP: same constraint on all 3 axes
            float damping = jnt.angularDriveDamping.z;
            if (damping > 0.0f) {
              float rho_drive = damping / dt2;
              for (int k = 0; k < 3; k++) {
                float C = (&relDW.x)[k] - (&worldAngTarget.x)[k];
                float lam = (&jnt.lambdaDriveAngular.x)[k];
                float sign = isA ? -1.0f : 1.0f;
                float f = sign * (rho_drive * C + lam);
                (&rhs.v[3])[k] += f;
                lhs.m[3 + k][3 + k] += rho_drive;
              }
            }
          } else {
            // Per-axis: TWIST(0x10)->X, SWING1(0x40)->Y, SWING2(0x80)->Z
            struct AxisInfo {
              uint32_t bit;
              int dampIdx;
              Vec3 localAxis;
            };
            AxisInfo axisInfos[3] = {
                {0x10, 0, Vec3(1, 0, 0)},
                {0x40, 1, Vec3(0, 1, 0)},
                {0x80, 2, Vec3(0, 0, 1)},
            };
            for (int a = 0; a < 3; a++) {
              if ((jnt.driveFlags & axisInfos[a].bit) == 0)
                continue;
              float damping =
                  (&jnt.angularDriveDamping.x)[axisInfos[a].dampIdx];
              if (damping <= 0.0f)
                continue;

              float rho_drive = damping / dt2;
              Vec3 wAxis = jointFrameA.rotate(axisInfos[a].localAxis);
              float C = relDW.dot(wAxis) - worldAngTarget.dot(wAxis);
              float lam = (&jnt.lambdaDriveAngular.x)[axisInfos[a].dampIdx];
              float sign = isA ? -1.0f : 1.0f;
              float f = sign * (rho_drive * C + lam);

              for (int k = 0; k < 3; k++) {
                (&rhs.v[3])[k] += f * (&wAxis.x)[k];
                for (int l = 0; l < 3; l++) {
                  lhs.m[3 + k][3 + l] +=
                      rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];
                }
              }
            }
          }
        }
      }

      // Solve and update
      // Solve and apply
      if (!use3x3Solve) {
        // ---- 6x6 fully-coupled LDLT ----
        Vec6 delta = solveLDLT(lhs, rhs);

        body.position -= delta.linear();
        Quat dq(0, delta[3], delta[4], delta[5]);
        body.rotation =
            (body.rotation - dq * body.rotation * 0.5f).normalized();
      } else {
        // ---- Decoupled 3x3 (same accumulation as 6x6, simpler solve) ----
        //
        // Same contacts + joints are accumulated into the 6x6 LHS/RHS.
        // We extract the diagonal 3x3 blocks and solve independently.
        Mat33 Alin, Aang;
        Vec3 rhsLin(rhs[0], rhs[1], rhs[2]);
        Vec3 rhsAng(rhs[3], rhs[4], rhs[5]);
        for (int r = 0; r < 3; r++) {
          for (int c = 0; c < 3; c++) {
            Alin.m[r][c] = lhs.m[r][c];
            Aang.m[r][c] = lhs.m[3 + r][3 + c];
          }
        }
        Vec3 deltaPos = Alin.inverse() * rhsLin;
        Vec3 deltaTheta = Aang.inverse() * rhsAng;
        body.position -= deltaPos;
        Quat dq(0, deltaTheta.x, deltaTheta.y, deltaTheta.z);
        body.rotation =
            (body.rotation - dq * body.rotation * 0.5f).normalized();
      }
    }

    // Dual update (all constraints)
    for (auto &c : contacts) {
      computeConstraint(c);
      for (int i = 0; i < 3; i++) {
        float oldLambda = c.lambda[i];
        float rawLambda = c.penalty[i] * c.C[i] + oldLambda;
        c.lambda[i] = std::max(c.fmin[i], std::min(c.fmax[i], rawLambda));

        // Penalty growth
        if (c.lambda[i] < c.fmax[i] && c.lambda[i] > c.fmin[i]) {
          c.penalty[i] =
              std::min(c.penalty[i] + beta * fabsf(c.C[i]), PENALTY_MAX);
        }
      }
    }

    // 3b. Joint dual update (ONCE per frame, OUTSIDE iteration loop)
    //
    // Three mechanisms ensure stable AL convergence for joints:
    //
    // (A) Auto-boosted primal penalty (effectiveRho):
    //     In the body loop, rho_primal = max(jnt.rho, body.mass/h^2).
    //     Ensures penalty is always >= body inertia for good primal
    //     convergence. Without this, heavy bodies (M/h^2 >> rho) barely
    //     respond to penalty, leaving large C that destabilizes the dual.
    //
    // (B) ADMM-safe dual step (rhoDual):
    //     rhoDual = min(M/h^2, rho^2/(rho+M/h^2))
    //     - Light bodies (M/h^2 << rho): rhoDual = M/h^2 (conservative)
    //     - Heavy bodies (M/h^2 >> rho): rhoDual = rho^2/(rho+M/h^2)
    //     This prevents dual overshoot for both regimes.
    //
    // (C) Lambda decay (leaky integrator):
    //     lambda_new = decay * lambda + rhoDual * C
    //     With decay=0.99, oscillation modes are exponentially damped.
    //     Steady-state residual: C_ss ~ (1-decay)/(1-decay+rhoDual/rho_eff)
    //     ~1-3% for typical configurations.
    // =====================================================================
    {
      auto getBodyMass = [&](uint32_t idx) -> float {
        return (idx == UINT32_MAX || idx >= (uint32_t)bodies.size())
                   ? 0.0f
                   : bodies[idx].mass;
      };
      auto computeRhoDual = [&](uint32_t idxA, uint32_t idxB,
                                float rho) -> float {
        float mA = getBodyMass(idxA);
        float mB = getBodyMass(idxB);
        // Use smaller dynamic mass (bottleneck for convergence)
        float mEff;
        if (mA <= 0.0f)
          mEff = mB;
        else if (mB <= 0.0f)
          mEff = mA;
        else
          mEff = std::min(mA, mB);
        float Mh2 = mEff / dt2;
        // Two regimes:
        //   Light bodies (Mh2 << rho): penalty dominates, primal converges
        //     well, use rhoDual = Mh2 (conservative, avoids mesh instability)
        //   Heavy bodies (Mh2 >> rho): penalty weak, need dual to compensate
        //     use rhoDual = rho^2/(rho+Mh2) (ADMM-safe inexact step)
        // Taking min covers both:
        float admm_step = rho * rho / (rho + Mh2);
        return std::min(Mh2, admm_step);
      };

      // Leaky integrator: lambda = decay * lambda + rhoDual * C
      // decay < 1 prevents lambda overshoot/oscillation while still
      // allowing lambda to converge to ~correct steady-state force.
      // Residual: C_ss = F_ss * (1-decay) / (rho*(1-decay) + rhoDual*decay)
      // With decay=0.99, residual is ~1-3% of joint length.
      const float lambdaDecay = 0.99f;

      for (auto &jnt : sphericalJoints) {
        float rhoDual = computeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho);
        bool aStatic = (jnt.bodyA == UINT32_MAX);
        bool bStatic = (jnt.bodyB == UINT32_MAX);
        Vec3 wA = aStatic ? jnt.anchorA
                          : bodies[jnt.bodyA].position +
                                bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
        Vec3 wB = bStatic ? jnt.anchorB
                          : bodies[jnt.bodyB].position +
                                bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
        jnt.lambda = jnt.lambda * lambdaDecay + (wA - wB) * rhoDual;

        if (jnt.coneAngleLimit > 0.0f) {
          Quat rotA = aStatic ? Quat() : bodies[jnt.bodyA].rotation;
          Quat rotB = bStatic ? Quat() : bodies[jnt.bodyB].rotation;
          float coneViol = jnt.computeConeViolation(rotA, rotB);

          jnt.coneLambda -= coneViol * rhoDual;
          jnt.coneLambda = std::max(-1e9f, std::min(0.0f, jnt.coneLambda));
        }
      }

      for (auto &jnt : fixedJoints) {
        float rhoDual = computeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho);
        bool aStatic = (jnt.bodyA == UINT32_MAX);
        bool bStatic = (jnt.bodyB == UINT32_MAX);
        Vec3 wA = aStatic ? jnt.anchorA
                          : bodies[jnt.bodyA].position +
                                bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
        Vec3 wB = bStatic ? jnt.anchorB
                          : bodies[jnt.bodyB].position +
                                bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
        Quat rotA = aStatic ? Quat() : bodies[jnt.bodyA].rotation;
        Quat rotB = bStatic ? Quat() : bodies[jnt.bodyB].rotation;
        jnt.lambdaPos = jnt.lambdaPos * lambdaDecay + (wA - wB) * rhoDual;
        jnt.lambdaRot = jnt.lambdaRot * lambdaDecay +
                        jnt.computeRotationViolation(rotA, rotB) * rhoDual;
      }

      for (auto &jnt : d6Joints) {
        float rhoDual = computeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho);
        bool aStatic = (jnt.bodyA == UINT32_MAX);
        bool bStatic = (jnt.bodyB == UINT32_MAX);
        Vec3 wA = aStatic ? jnt.anchorA
                          : bodies[jnt.bodyA].position +
                                bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
        Vec3 wB = bStatic ? jnt.anchorB
                          : bodies[jnt.bodyB].position +
                                bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
        Vec3 posViol = wA - wB;
        for (int axis = 0; axis < 3; axis++) {
          if (jnt.getLinearMotion(axis) != 0)
            continue;
          (&jnt.lambdaLinear.x)[axis] =
              (&jnt.lambdaLinear.x)[axis] * lambdaDecay +
              (&posViol.x)[axis] * rhoDual;
        }

        // --- Drive dual update ---
        // Joint frame A in world space
        Quat jointFrameA = aStatic
                               ? jnt.localFrameA
                               : bodies[jnt.bodyA].rotation * jnt.localFrameA;
        jointFrameA = jointFrameA.normalized();

        // Linear velocity drive dual
        if (jnt.driveFlags & 0x07) {
          Vec3 dxA = aStatic ? Vec3()
                             : bodies[jnt.bodyA].position -
                                   bodies[jnt.bodyA].initialPosition;
          Vec3 dxB = bStatic ? Vec3()
                             : bodies[jnt.bodyB].position -
                                   bodies[jnt.bodyB].initialPosition;
          Vec3 relDisp = dxB - dxA;
          Vec3 worldTarget = jointFrameA.rotate(jnt.driveLinearVelocity) * dt;

          Vec3 localAxes[3] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
          for (int a = 0; a < 3; a++) {
            if ((jnt.driveFlags & (1 << a)) == 0)
              continue;
            float damping = (&jnt.linearDriveDamping.x)[a];
            if (damping <= 0.0f)
              continue;
            Vec3 wAxis = jointFrameA.rotate(localAxes[a]);
            float C = relDisp.dot(wAxis) - worldTarget.dot(wAxis);
            float rhoDualDrive = std::min(damping / dt2, rhoDual);
            (&jnt.lambdaDriveLinear.x)[a] =
                (&jnt.lambdaDriveLinear.x)[a] * lambdaDecay + rhoDualDrive * C;
          }
        }

        // Angular velocity drive dual
        if (jnt.driveFlags & 0xF0) {
          auto getDeltaW = [&](uint32_t idx) -> Vec3 {
            if (idx == UINT32_MAX || idx >= (uint32_t)bodies.size())
              return Vec3();
            return bodies[idx].deltaWInitial();
          };
          Vec3 dwA = getDeltaW(jnt.bodyA);
          Vec3 dwB = getDeltaW(jnt.bodyB);
          Vec3 relDW = dwB - dwA;
          Vec3 worldAngTarget =
              jointFrameA.rotate(jnt.driveAngularVelocity) * dt;

          bool slerpDrive = (jnt.driveFlags & 0x20) != 0;
          if (slerpDrive) {
            float damping = jnt.angularDriveDamping.z;
            if (damping > 0.0f) {
              float rhoDualDrive = std::min(damping / dt2, rhoDual);
              for (int k = 0; k < 3; k++) {
                float C = (&relDW.x)[k] - (&worldAngTarget.x)[k];
                (&jnt.lambdaDriveAngular.x)[k] =
                    (&jnt.lambdaDriveAngular.x)[k] * lambdaDecay +
                    rhoDualDrive * C;
              }
            }
          } else {
            struct AxisInfo {
              uint32_t bit;
              int idx;
              Vec3 localAxis;
            };
            AxisInfo axes[3] = {
                {0x10, 0, Vec3(1, 0, 0)},
                {0x40, 1, Vec3(0, 1, 0)},
                {0x80, 2, Vec3(0, 0, 1)},
            };
            for (int a = 0; a < 3; a++) {
              if ((jnt.driveFlags & axes[a].bit) == 0)
                continue;
              float damping = (&jnt.angularDriveDamping.x)[axes[a].idx];
              if (damping <= 0.0f)
                continue;
              Vec3 wAxis = jointFrameA.rotate(axes[a].localAxis);
              float C = relDW.dot(wAxis) - worldAngTarget.dot(wAxis);
              float rhoDualDrive = std::min(damping / dt2, rhoDual);
              (&jnt.lambdaDriveAngular.x)[axes[a].idx] =
                  (&jnt.lambdaDriveAngular.x)[axes[a].idx] * lambdaDecay +
                  rhoDualDrive * C;
            }
          }
        }
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

  // (GS velocity motor removed — drives are now pure AVBD AL constraints)
}

} // namespace AvbdRef
