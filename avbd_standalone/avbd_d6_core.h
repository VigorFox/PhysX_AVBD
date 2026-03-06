#pragma once
#include "avbd_types.h"
#include <cmath>
#include <vector>

namespace AvbdRef {

// =============================================================================
// Helper: effective mass for ADMM-safe dual step
// =============================================================================
inline float d6GetBodyMass(uint32_t idx, const std::vector<Body> &bodies) {
  if (idx == UINT32_MAX || idx >= (uint32_t)bodies.size())
    return 0.0f;
  return bodies[idx].mass;
}

inline float d6ComputeRhoDual(uint32_t idxA, uint32_t idxB, float rho,
                              const std::vector<Body> &bodies, float dt2) {
  float mA = d6GetBodyMass(idxA, bodies);
  float mB = d6GetBodyMass(idxB, bodies);
  float mEff;
  if (mA <= 0.0f)
    mEff = mB;
  else if (mB <= 0.0f)
    mEff = mA;
  else
    mEff = std::min(mA, mB);
  float Mh2 = mEff / dt2;
  float admm_step = rho * rho / (rho + Mh2);
  return std::min(Mh2, admm_step);
}

// =============================================================================
// computeAngularError -- axis-angle decomposition (matches PhysX)
//
// Computes angular error around a specific axis using axis-angle decomposition.
// This is numerically equivalent to PhysX's AvbdD6JointConstraint::computeAngularError.
// =============================================================================
inline float computeAngularError(const Quat &rotA, const Quat &rotB,
                                 const Quat &localFrameA, const Quat &localFrameB,
                                 int axis) {
  Quat worldFrameA = rotA * localFrameA;
  Quat worldFrameB = rotB * localFrameB;
  Quat relRot = worldFrameA * worldFrameB.conjugate();
  if (relRot.w < 0.0f)
    relRot = relRot * (-1.0f);

  float angle = 2.0f * acosf(std::max(-1.0f, std::min(1.0f, relRot.w)));
  if (angle < 1e-6f)
    return 0.0f;

  Vec3 axisVec(relRot.x, relRot.y, relRot.z);
  float len = axisVec.length();
  if (len < 1e-12f)
    return 0.0f;
  axisVec = axisVec * (1.0f / len);

  Vec3 localAxis(axis == 0 ? 1.0f : 0.0f, axis == 1 ? 1.0f : 0.0f,
                 axis == 2 ? 1.0f : 0.0f);
  Vec3 worldAxis = worldFrameA.rotate(localAxis);

  return angle * axisVec.dot(worldAxis);
}

// Compute angular limit violation for a given error and axis limits
inline float computeAngularLimitViolation(float error, float lower, float upper) {
  if (error < lower) return error - lower;
  if (error > upper) return error - upper;
  return 0.0f;
}

// =============================================================================
// addD6Contribution -- primal update per body
//
// Accumulates the D6Joint's Hessian and gradient contributions into the
// per-body 6x6 LHS/RHS system.
//
// Sign convention:
//   C_lin = anchorA_world - anchorB_world
//   J = axis * sign   (sign = +1 for bodyA, -1 for bodyB)
//
// Angular error:
//   Revolute pattern (twist FREE/LIMITED, swing LOCKED):
//     Cross-product axis alignment (immune to twist-angle amplification)
//   Generic: axis-angle decomposition matching PhysX computeAngularError
//
// Axis selection:
//   All-LOCKED linear => world axes (numerical stability)
//   Otherwise linear => joint-local axes from localFrameA
//   Angular: always uses worldFrameA.rotate(localAxis) (matches PhysX)
// =============================================================================
inline void addD6Contribution(const D6Joint &jnt, uint32_t bi,
                              const std::vector<Body> &bodies, float dt,
                              Mat66 &lhs, Vec6 &rhs) {
  bool isA = (jnt.bodyA == bi);
  bool isB = (jnt.bodyB == bi);
  if (!isA && !isB)
    return;

  bool otherStatic =
      isA ? (jnt.bodyB == UINT32_MAX) : (jnt.bodyA == UINT32_MAX);

  const Body &body = bodies[bi];
  float dt2 = dt * dt;
  float sign = isA ? 1.0f : -1.0f;

  Vec3 r = isA ? body.rotation.rotate(jnt.anchorA)
               : body.rotation.rotate(jnt.anchorB);

  // World-space anchor positions
  Vec3 worldAnchorA, worldAnchorB;
  if (isA) {
    worldAnchorA = body.position + r;
    worldAnchorB =
        otherStatic
            ? jnt.anchorB
            : (bodies[jnt.bodyB].position +
               bodies[jnt.bodyB].rotation.rotate(jnt.anchorB));
  } else {
    worldAnchorA =
        otherStatic
            ? jnt.anchorA
            : (bodies[jnt.bodyA].position +
               bodies[jnt.bodyA].rotation.rotate(jnt.anchorA));
    worldAnchorB = body.position + r;
  }
  Vec3 C_lin = worldAnchorA - worldAnchorB;

  // Joint frame in world space
  Quat rotA = isA ? body.rotation
                   : ((jnt.bodyA == UINT32_MAX) ? Quat()
                                                : bodies[jnt.bodyA].rotation);
  Quat rotB = isA ? ((jnt.bodyB == UINT32_MAX) ? Quat()
                                                : bodies[jnt.bodyB].rotation)
                   : body.rotation;

  bool bStatic = (jnt.bodyB == UINT32_MAX);
  Quat jointFrameA = bStatic ? jnt.localFrameA : rotA * jnt.localFrameA;
  jointFrameA = jointFrameA.normalized();

  Vec3 localAxesA[3] = {jointFrameA.rotate(Vec3(1, 0, 0)),
                        jointFrameA.rotate(Vec3(0, 1, 0)),
                        jointFrameA.rotate(Vec3(0, 0, 1))};

  // Auto-boost penalty for primal convergence
  float effectiveRho = std::max(jnt.rho, body.mass / dt2);

  // --- Linear constraints ---
  bool linAllLocked = (jnt.linearMotion == 0);
  Vec3 linearAxes[3];
  if (linAllLocked) {
    linearAxes[0] = Vec3(1, 0, 0);
    linearAxes[1] = Vec3(0, 1, 0);
    linearAxes[2] = Vec3(0, 0, 1);
  } else {
    linearAxes[0] = localAxesA[0];
    linearAxes[1] = localAxesA[1];
    linearAxes[2] = localAxesA[2];
  }

  for (int k = 0; k < 3; k++) {
    uint32_t motion = (jnt.linearMotion >> (k * 2)) & 0x3;
    if (motion == 2)
      continue; // FREE

    Vec3 axis = linearAxes[k];

    if (motion == 0) { // LOCKED
      float Ck = C_lin.dot(axis);
      Vec6 J(axis * sign, r.cross(axis) * sign);
      float lam = (&jnt.lambdaLinear.x)[k];
      float f = effectiveRho * Ck + lam;
      rhs += J * f;
      lhs += outer(J, J * effectiveRho);
    } else if (motion == 1) { // LIMITED
      // Baseline stiffness along this axis (prevents drift in free range)
      {
        float Ck = C_lin.dot(axis);
        Vec6 J(axis * sign, r.cross(axis) * sign);
        lhs += outer(J, J * effectiveRho);
      }

      // Limit violation
      float dist = -C_lin.dot(axis);
      float limitViol = 0.0f;
      if (dist < jnt.linearLimitLower[k])
        limitViol = dist - jnt.linearLimitLower[k];
      else if (dist > jnt.linearLimitUpper[k])
        limitViol = dist - jnt.linearLimitUpper[k];

      if (limitViol != 0.0f) {
        Vec3 J_xyz = -axis * sign;
        Vec6 Jlim(J_xyz, r.cross(-axis) * sign);
        float lam = jnt.lambdaLimitLinear[k];
        float f = effectiveRho * limitViol + lam;
        rhs += Jlim * f;
        lhs += outer(Jlim, Jlim * effectiveRho);
      }
    }
  }

  // --- Angular constraints (matches PhysX DyAvbdSolver.cpp) ---
  {
    // Detect revolute pattern: twist(0) FREE or LIMITED, swing(1,2) LOCKED
    uint32_t twistMotion = jnt.getAngularMotion(0);
    uint32_t swing1Motion = jnt.getAngularMotion(1);
    uint32_t swing2Motion = jnt.getAngularMotion(2);
    bool isRevolutePattern =
        (twistMotion != 0) && (swing1Motion == 0) && (swing2Motion == 0);

    Quat worldFrameB = rotB * jnt.localFrameB;
    worldFrameB = worldFrameB.normalized();

    if (isRevolutePattern) {
      // Cross-product axis alignment (2 rows) - matches PhysX revolute solver.
      // Unlike computeAngularError decomposition, this is immune to large
      // twist angles amplifying swing drift.
      Vec3 worldTwistA = jointFrameA.rotate(Vec3(1, 0, 0));
      Vec3 worldTwistB = worldFrameB.rotate(Vec3(1, 0, 0));
      Vec3 axisViolation = worldTwistA.cross(worldTwistB);

      // Build perpendicular basis from worldTwistA
      Vec3 perp1, perp2;
      if (std::fabs(worldTwistA.x) < 0.9f)
        perp1 = worldTwistA.cross(Vec3(1, 0, 0));
      else
        perp1 = worldTwistA.cross(Vec3(0, 1, 0));
      float perp1Len = perp1.length();
      if (perp1Len > 1e-6f)
        perp1 = perp1 * (1.0f / perp1Len);
      perp2 = worldTwistA.cross(perp1);
      float perp2Len = perp2.length();
      if (perp2Len > 1e-6f)
        perp2 = perp2 * (1.0f / perp2Len);

      float err1 = axisViolation.dot(perp1);
      float err2 = axisViolation.dot(perp2);

      // Row 1 (stored in lambdaAngular[1])
      {
        Vec6 J(Vec3(), perp1 * sign * (-1.0f));
        float f = effectiveRho * err1 + (&jnt.lambdaAngular.x)[1];
        rhs += J * f;
        lhs += outer(J, J * effectiveRho);
      }
      // Row 2 (stored in lambdaAngular[2])
      {
        Vec6 J(Vec3(), perp2 * sign * (-1.0f));
        float f = effectiveRho * err2 + (&jnt.lambdaAngular.x)[2];
        rhs += J * f;
        lhs += outer(J, J * effectiveRho);
      }

      // Handle twist axis (0) if LIMITED
      if (twistMotion == 1) {
        Vec3 worldAxis = jointFrameA.rotate(Vec3(1, 0, 0));
        Vec3 jAng = worldAxis * sign;
        Vec6 J(Vec3(), jAng);

        float error = computeAngularError(rotA, rotB, jnt.localFrameA,
                                          jnt.localFrameB, 0);
        float limitViol = computeAngularLimitViolation(
            error, jnt.angularLimitLower[0], jnt.angularLimitUpper[0]);
        float f = effectiveRho * limitViol + jnt.lambdaLimitAngular[0];
        float forceMag = 0.0f;

        if (jnt.angularLimitLower[0] < jnt.angularLimitUpper[0]) {
          if (limitViol > 0.0f || jnt.lambdaLimitAngular[0] > 0.0f)
            forceMag = std::max(0.0f, f);
          else if (limitViol < 0.0f || jnt.lambdaLimitAngular[0] < 0.0f)
            forceMag = std::min(0.0f, f);
        } else {
          forceMag = f;
        }

        if (std::fabs(forceMag) > 0.0f) {
          rhs += J * forceMag;
          lhs += outer(J, J * effectiveRho);
        }
      }
    } else {
      // Generic per-axis angular constraint handling (matches PhysX generic path)
      for (int k = 0; k < 3; k++) {
        uint32_t motion = jnt.getAngularMotion(k);
        if (motion == 2)
          continue; // FREE

        Vec3 localAxis(k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f,
                       k == 2 ? 1.0f : 0.0f);
        Vec3 worldAxis = jointFrameA.rotate(localAxis);
        Vec3 jAng = worldAxis * sign;
        Vec6 J(Vec3(), jAng);

        if (motion == 0) { // LOCKED
          float Ck = computeAngularError(rotA, rotB, jnt.localFrameA,
                                         jnt.localFrameB, k);
          float lam = (&jnt.lambdaAngular.x)[k];
          float f = effectiveRho * Ck + lam;
          rhs += J * f;
          lhs += outer(J, J * effectiveRho);
        } else if (motion == 1) { // LIMITED
          float error = computeAngularError(rotA, rotB, jnt.localFrameA,
                                            jnt.localFrameB, k);
          float limitViol = computeAngularLimitViolation(
              error, jnt.angularLimitLower[k], jnt.angularLimitUpper[k]);

          if (limitViol != 0.0f) {
            float lam = jnt.lambdaLimitAngular[k];
            float f = effectiveRho * limitViol + lam;
            float forceMag = 0.0f;
            if (jnt.angularLimitLower[k] < jnt.angularLimitUpper[k]) {
              if (limitViol > 0.0f || lam > 0.0f)
                forceMag = std::max(0.0f, f);
              else if (limitViol < 0.0f || lam < 0.0f)
                forceMag = std::min(0.0f, f);
            } else {
              forceMag = f;
            }

            if (std::fabs(forceMag) > 0.0f) {
              rhs += J * forceMag;
              lhs += outer(J, J * effectiveRho);
            }
          }
        }
      }
    }
  }

  // --- Cone limit (matches PhysX: uses per-body joint frame X-axis) ---
  if (jnt.coneAngleLimit > 0.0f) {
    Vec3 worldAxisA_cone = (rotA * jnt.localFrameA).rotate(Vec3(1, 0, 0));
    Vec3 worldAxisB_cone = (rotB * jnt.localFrameB).rotate(Vec3(1, 0, 0));
    float dotProd =
        std::max(-1.0f, std::min(1.0f, worldAxisA_cone.dot(worldAxisB_cone)));
    float coneViol = acosf(dotProd) - jnt.coneAngleLimit;

    if (coneViol > 0.0f || jnt.coneLambda < 0.0f) {
      Vec3 n = worldAxisA_cone.cross(worldAxisB_cone);
      float len = n.length();
      if (len > 1e-6f) {
        n = n * (1.0f / len);
        Vec3 J_ang = n * sign;
        Vec6 J(Vec3(), J_ang);
        float f = effectiveRho * coneViol + (-jnt.coneLambda);
        f = std::max(0.0f, f);
        rhs += J * f;
        lhs += outer(J, J * effectiveRho);
      }
    }
  }

  // --- Drives ---
  if (jnt.driveFlags & 0x07) {
    Vec3 dxThis = body.position - body.initialPosition;
    Vec3 dxOther = otherStatic
                       ? Vec3()
                       : (bodies[isA ? jnt.bodyB : jnt.bodyA].position -
                          bodies[isA ? jnt.bodyB : jnt.bodyA].initialPosition);
    Vec3 relDisp = isA ? (dxOther - dxThis) : (dxThis - dxOther);
    Vec3 worldTarget = jointFrameA.rotate(jnt.driveLinearVelocity) * dt;

    for (int a = 0; a < 3; a++) {
      if ((jnt.driveFlags & (1 << a)) == 0)
        continue;
      float damping = (&jnt.linearDriveDamping.x)[a];
      if (damping <= 0.0f)
        continue;

      float rho_drive = damping / dt2;
      Vec3 wAxis = localAxesA[a];
      float C = relDisp.dot(wAxis) - worldTarget.dot(wAxis);
      float lam = (&jnt.lambdaDriveLinear.x)[a];
      float sign_d = isA ? -1.0f : 1.0f;
      float f = sign_d * (rho_drive * C + lam);

      Vec6 Jd(wAxis, r.cross(wAxis));
      rhs += Jd * f;
      lhs += outer(Jd, Jd * rho_drive);
    }
  }

  if (jnt.driveFlags & 0xF0) {
    Vec3 dwThis = body.deltaWInitial();
    Vec3 dwOther = otherStatic
                       ? Vec3()
                       : bodies[isA ? jnt.bodyB : jnt.bodyA].deltaWInitial();
    Vec3 relDW = isA ? (dwOther - dwThis) : (dwThis - dwOther);
    Vec3 worldAngTarget = jointFrameA.rotate(jnt.driveAngularVelocity) * dt;

    if (jnt.driveFlags & 0x20) { // SLERP
      float damping = jnt.angularDriveDamping.z;
      if (damping > 0.0f) {
        float rho_drive = damping / dt2;
        for (int k2 = 0; k2 < 3; k2++) {
          float C = (&relDW.x)[k2] - (&worldAngTarget.x)[k2];
          float lam = (&jnt.lambdaDriveAngular.x)[k2];
          float sign_d = isA ? -1.0f : 1.0f;
          float f = sign_d * (rho_drive * C + lam);
          Vec6 Jd(Vec3(), Vec3((k2 == 0 ? 1.f : 0.f), (k2 == 1 ? 1.f : 0.f),
                               (k2 == 2 ? 1.f : 0.f)));
          rhs += Jd * f;
          lhs += outer(Jd, Jd * rho_drive);
        }
      }
    } else {
      uint32_t bits[3] = {0x10, 0x40, 0x80};
      for (int a = 0; a < 3; a++) {
        if ((jnt.driveFlags & bits[a]) == 0)
          continue;
        float damping = (&jnt.angularDriveDamping.x)[a];
        if (damping <= 0.0f)
          continue;
        float rho_drive = damping / dt2;
        Vec3 wAxis = localAxesA[a];
        float C = relDW.dot(wAxis) - worldAngTarget.dot(wAxis);
        float lam = (&jnt.lambdaDriveAngular.x)[a];
        float sign_d = isA ? -1.0f : 1.0f;
        float f = sign_d * (rho_drive * C + lam);

        Vec6 Jd(Vec3(), wAxis);
        rhs += Jd * f;
        lhs += outer(Jd, Jd * rho_drive);
      }
    }
  }
}

// =============================================================================
// updateD6Dual -- dual update (lambda update, Augmented Lagrangian)
//
// Same sign convention as primal: C_lin = wA - wB, C_ang = vec(err)*2
// =============================================================================
inline void updateD6Dual(D6Joint &jnt, const std::vector<Body> &bodies,
                         float dt, float lambdaDecay) {
  bool aStatic = (jnt.bodyA == UINT32_MAX);
  bool bStatic = (jnt.bodyB == UINT32_MAX);
  float dt2 = dt * dt;
  float rhoDual = d6ComputeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho, bodies, dt2);

  Vec3 wA = aStatic ? jnt.anchorA
                    : bodies[jnt.bodyA].position +
                          bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
  Vec3 wB = bStatic ? jnt.anchorB
                    : bodies[jnt.bodyB].position +
                          bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
  Vec3 C_lin = wA - wB;

  Quat rotA = aStatic ? Quat() : bodies[jnt.bodyA].rotation;
  Quat rotB = bStatic ? Quat() : bodies[jnt.bodyB].rotation;
  Quat jointFrameA = bStatic ? jnt.localFrameA : rotA * jnt.localFrameA;
  jointFrameA = jointFrameA.normalized();
  Vec3 localAxesA[3] = {jointFrameA.rotate(Vec3(1, 0, 0)),
                        jointFrameA.rotate(Vec3(0, 1, 0)),
                        jointFrameA.rotate(Vec3(0, 0, 1))};

  // --- Linear dual ---
  bool linAllLocked = (jnt.linearMotion == 0);
  Vec3 linearAxes[3];
  if (linAllLocked) {
    linearAxes[0] = Vec3(1, 0, 0);
    linearAxes[1] = Vec3(0, 1, 0);
    linearAxes[2] = Vec3(0, 0, 1);
  } else {
    linearAxes[0] = localAxesA[0];
    linearAxes[1] = localAxesA[1];
    linearAxes[2] = localAxesA[2];
  }

  for (int k = 0; k < 3; k++) {
    uint32_t motion = (jnt.linearMotion >> (k * 2)) & 0x3;
    if (motion == 0) { // LOCKED
      float Ck = C_lin.dot(linearAxes[k]);
      (&jnt.lambdaLinear.x)[k] =
          (&jnt.lambdaLinear.x)[k] * lambdaDecay + Ck * rhoDual;
    } else if (motion == 1) { // LIMITED
      float dist = -C_lin.dot(linearAxes[k]);
      float limitViol = 0.0f;
      if (dist < jnt.linearLimitLower[k])
        limitViol = dist - jnt.linearLimitLower[k];
      else if (dist > jnt.linearLimitUpper[k])
        limitViol = dist - jnt.linearLimitUpper[k];

      float newLam =
          jnt.lambdaLimitLinear[k] * lambdaDecay + limitViol * rhoDual;
      if (jnt.linearLimitLower[k] < jnt.linearLimitUpper[k]) {
        float signRef = (std::fabs(limitViol) > 1e-6f)
                            ? limitViol
                            : ((std::fabs(jnt.lambdaLimitLinear[k]) > 1e-6f)
                                   ? jnt.lambdaLimitLinear[k]
                                   : 0.0f);
        if (signRef > 0.0f)
          newLam = std::max(0.0f, newLam);
        else if (signRef < 0.0f)
          newLam = std::min(0.0f, newLam);
        else
          newLam = 0.0f;
      }
      jnt.lambdaLimitLinear[k] = newLam;
    }
  }

  // --- Angular dual (matches PhysX) ---
  {
    uint32_t twistMotion = jnt.getAngularMotion(0);
    uint32_t swing1Motion = jnt.getAngularMotion(1);
    uint32_t swing2Motion = jnt.getAngularMotion(2);
    bool isRevolutePattern =
        (twistMotion != 0) && (swing1Motion == 0) && (swing2Motion == 0);

    if (isRevolutePattern) {
      // Cross-product axis alignment dual (matches PhysX revolute path)
      Quat worldFrameA_d = rotA * jnt.localFrameA;
      Quat worldFrameB_d = rotB * jnt.localFrameB;
      Vec3 worldTwistA = worldFrameA_d.rotate(Vec3(1, 0, 0));
      Vec3 worldTwistB = worldFrameB_d.rotate(Vec3(1, 0, 0));
      Vec3 axisViol = worldTwistA.cross(worldTwistB);

      Vec3 perp1, perp2;
      if (std::fabs(worldTwistA.x) < 0.9f)
        perp1 = worldTwistA.cross(Vec3(1, 0, 0));
      else
        perp1 = worldTwistA.cross(Vec3(0, 1, 0));
      float p1Len = perp1.length();
      if (p1Len > 1e-6f) perp1 = perp1 * (1.0f / p1Len);
      perp2 = worldTwistA.cross(perp1);
      float p2Len = perp2.length();
      if (p2Len > 1e-6f) perp2 = perp2 * (1.0f / p2Len);

      float err1 = axisViol.dot(perp1);
      float err2 = axisViol.dot(perp2);

      (&jnt.lambdaAngular.x)[1] =
          (&jnt.lambdaAngular.x)[1] * lambdaDecay + err1 * rhoDual;
      (&jnt.lambdaAngular.x)[2] =
          (&jnt.lambdaAngular.x)[2] * lambdaDecay + err2 * rhoDual;

      // Twist axis (0) if LIMITED
      if (twistMotion == 1) {
        float angErr = computeAngularError(rotA, rotB, jnt.localFrameA,
                                           jnt.localFrameB, 0);
        float limitViol = computeAngularLimitViolation(
            angErr, jnt.angularLimitLower[0], jnt.angularLimitUpper[0]);
        float newLam =
            jnt.lambdaLimitAngular[0] * lambdaDecay + limitViol * rhoDual;

        if (jnt.angularLimitLower[0] < jnt.angularLimitUpper[0]) {
          if (limitViol > 0.0f || jnt.lambdaLimitAngular[0] > 0.0f)
            newLam = std::max(0.0f, newLam);
          else if (limitViol < 0.0f || jnt.lambdaLimitAngular[0] < 0.0f)
            newLam = std::min(0.0f, newLam);
          else
            newLam = 0.0f;
        }
        jnt.lambdaLimitAngular[0] = newLam;
      }
    } else {
      // Generic per-axis angular dual (matches PhysX generic path)
      for (int k = 0; k < 3; k++) {
        uint32_t motion = jnt.getAngularMotion(k);
        if (motion == 2) continue; // FREE

        if (motion == 0) { // LOCKED
          float angErr = computeAngularError(rotA, rotB, jnt.localFrameA,
                                             jnt.localFrameB, k);
          (&jnt.lambdaAngular.x)[k] =
              (&jnt.lambdaAngular.x)[k] * lambdaDecay + angErr * rhoDual;
        } else if (motion == 1) { // LIMITED
          float angErr = computeAngularError(rotA, rotB, jnt.localFrameA,
                                             jnt.localFrameB, k);
          float limitViol = computeAngularLimitViolation(
              angErr, jnt.angularLimitLower[k], jnt.angularLimitUpper[k]);
          float newLam =
              jnt.lambdaLimitAngular[k] * lambdaDecay + limitViol * rhoDual;

          if (jnt.angularLimitLower[k] < jnt.angularLimitUpper[k]) {
            float signRef =
                (std::fabs(limitViol) > 1e-6f)
                    ? limitViol
                    : ((std::fabs(jnt.lambdaLimitAngular[k]) > 1e-6f)
                           ? jnt.lambdaLimitAngular[k]
                           : 0.0f);
            if (signRef > 0.0f)
              newLam = std::max(0.0f, newLam);
            else if (signRef < 0.0f)
              newLam = std::min(0.0f, newLam);
            else
              newLam = 0.0f;
          }
          jnt.lambdaLimitAngular[k] = newLam;
        }
      }
    }
  }

  // --- Cone limit dual (matches PhysX: per-body joint frame X-axis) ---
  if (jnt.coneAngleLimit > 0.0f) {
    Vec3 worldAxisA_cone = (rotA * jnt.localFrameA).rotate(Vec3(1, 0, 0));
    Vec3 worldAxisB_cone = (rotB * jnt.localFrameB).rotate(Vec3(1, 0, 0));
    float dotProd =
        std::max(-1.0f, std::min(1.0f, worldAxisA_cone.dot(worldAxisB_cone)));
    float coneViol = acosf(dotProd) - jnt.coneAngleLimit;
    jnt.coneLambda -= coneViol * rhoDual;
    jnt.coneLambda = std::max(-1e9f, std::min(0.0f, jnt.coneLambda));
  }

  // --- Drive dual ---
  if (jnt.driveFlags & 0x07) {
    Vec3 dxA = aStatic ? Vec3()
                       : bodies[jnt.bodyA].position -
                             bodies[jnt.bodyA].initialPosition;
    Vec3 dxB = bStatic ? Vec3()
                       : bodies[jnt.bodyB].position -
                             bodies[jnt.bodyB].initialPosition;
    Vec3 relDisp = dxB - dxA;
    Vec3 worldTarget = jointFrameA.rotate(jnt.driveLinearVelocity) * dt;

    for (int a = 0; a < 3; a++) {
      if ((jnt.driveFlags & (1 << a)) == 0)
        continue;
      float damping = (&jnt.linearDriveDamping.x)[a];
      if (damping <= 0.0f)
        continue;

      float rhoDualDrive = std::min(damping / dt2, rhoDual);
      float C = relDisp.dot(localAxesA[a]) - worldTarget.dot(localAxesA[a]);
      (&jnt.lambdaDriveLinear.x)[a] =
          (&jnt.lambdaDriveLinear.x)[a] * lambdaDecay + rhoDualDrive * C;
    }
  }

  if (jnt.driveFlags & 0xF0) {
    Vec3 dwA = aStatic ? Vec3() : bodies[jnt.bodyA].deltaWInitial();
    Vec3 dwB = bStatic ? Vec3() : bodies[jnt.bodyB].deltaWInitial();
    Vec3 relDW = dwB - dwA;
    Vec3 worldAngTarget = jointFrameA.rotate(jnt.driveAngularVelocity) * dt;

    uint32_t bits[3] = {0x10, 0x40, 0x80};
    for (int a = 0; a < 3; a++) {
      bool active = (jnt.driveFlags & 0x20) || (jnt.driveFlags & bits[a]);
      if (!active)
        continue;
      float damping = (jnt.driveFlags & 0x20) ? jnt.angularDriveDamping.z
                                              : (&jnt.angularDriveDamping.x)[a];
      if (damping <= 0.0f)
        continue;

      float rhoDualDrive = std::min(damping / dt2, rhoDual);
      Vec3 wAxis = (jnt.driveFlags & 0x20)
                       ? Vec3(a == 0 ? 1.f : 0.f, a == 1 ? 1.f : 0.f,
                              a == 2 ? 1.f : 0.f)
                       : localAxesA[a];
      float C = relDW.dot(wAxis) - worldAngTarget.dot(wAxis);
      (&jnt.lambdaDriveAngular.x)[a] =
          (&jnt.lambdaDriveAngular.x)[a] * lambdaDecay + rhoDualDrive * C;
    }
  }
}

} // namespace AvbdRef
