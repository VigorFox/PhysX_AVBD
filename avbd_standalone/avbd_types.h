#pragma once
#include "avbd_math.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace AvbdRef {

// ====================== Rigid Body ==========================================

struct Body {
  Vec3 position;
  Quat rotation;
  Vec3 linearVelocity;
  Vec3 angularVelocity;
  Vec3 prevLinearVelocity;

  Vec3 initialPosition; // position at start of step (before warmstart)
  Quat initialRotation;
  Vec3 inertialPosition; // prediction = pos + v*dt + g*dt^2
  Quat inertialRotation;

  float mass;          // 0 = static
  Mat33 inertiaTensor; // in local frame
  float friction;
  Vec3 halfExtent; // box half-extents (for collision queries)

  // Derived (computed at init)
  float invMass;
  Mat33 invInertiaWorld; // inverse inertia in world frame

  void computeDerived() {
    invMass = (mass > 0) ? 1.0f / mass : 0.0f;
    if (mass > 0) {
      invInertiaWorld = inertiaTensor.inverse();
    }
  }

  void updateInvInertiaWorld() {
    if (mass <= 0)
      return;
    Mat33 invIlocal = inertiaTensor.inverse();
    Mat33 R;
    float qw = rotation.w, qx = rotation.x, qy = rotation.y, qz = rotation.z;
    R.m[0][0] = 1 - 2 * (qy * qy + qz * qz);
    R.m[0][1] = 2 * (qx * qy - qz * qw);
    R.m[0][2] = 2 * (qx * qz + qy * qw);
    R.m[1][0] = 2 * (qx * qy + qz * qw);
    R.m[1][1] = 1 - 2 * (qx * qx + qz * qz);
    R.m[1][2] = 2 * (qy * qz - qx * qw);
    R.m[2][0] = 2 * (qx * qz - qy * qw);
    R.m[2][1] = 2 * (qy * qz + qx * qw);
    R.m[2][2] = 1 - 2 * (qx * qx + qy * qy);

    Mat33 RI;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        RI.m[i][j] = 0;
        for (int k = 0; k < 3; k++)
          RI.m[i][j] += R.m[i][k] * invIlocal.m[k][j];
      }
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        invInertiaWorld.m[i][j] = 0;
        for (int k = 0; k < 3; k++)
          invInertiaWorld.m[i][j] += RI.m[i][k] * R.m[j][k]; // R^T
      }
  }

  Mat66 getMassMatrix() const {
    Mat66 M;
    for (int i = 0; i < 3; i++)
      M.m[i][i] = mass;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        M.m[3 + i][3 + j] = inertiaTensor.m[i][j];
    return M;
  }

  Vec3 deltaWInitial() const {
    Quat dq = rotation * initialRotation.conjugate();
    if (dq.w < 0)
      dq = dq * (-1.f);
    return Vec3(dq.x, dq.y, dq.z) * 2.0f;
  }

  Vec3 deltaWInertial() const {
    Quat dq = rotation * inertialRotation.conjugate();
    if (dq.w < 0)
      dq = dq * (-1.f);
    return Vec3(dq.x, dq.y, dq.z) * 2.0f;
  }
};

// ====================== Contact Constraint ==================================

struct Contact {
  uint32_t bodyA;
  uint32_t bodyB; // UINT32_MAX = static ground

  Vec3 normal;
  Vec3 rA;
  Vec3 rB;
  float depth;
  float friction;

  Vec6 JA;
  Vec6 JB;
  Vec6 JAt1, JBt1;
  Vec6 JAt2, JBt2;
  float C[3];
  float C0[3];

  float fmin[3], fmax[3];

  float lambda[3];
  float penalty[3];
};

// ====================== Unified D6 Joint ====================================
//
// All joint types (spherical, fixed, revolute, prismatic, generic D6) are
// represented as a single D6Joint struct. Factory methods configure the
// per-axis motion modes appropriately.
//
// Axis convention (in joint frame, defined by localFrameA):
//   Linear  axes 0,1,2 = X,Y,Z
//   Angular axes 0,1,2 = X(twist), Y(swing1), Z(swing2)
//
// Motion modes (packed 2 bits per axis):
//   0 = LOCKED  (equality constraint, lambda accumulates freely)
//   1 = LIMITED  (inequality constraint with lower/upper bounds)
//   2 = FREE     (no constraint)
//
// Axis selection for Hessian construction:
//   When all 3 linear (or angular) DOFs are LOCKED, world axes {1,0,0},
//   {0,1,0}, {0,0,1} are used for numerical stability. When any DOF is
//   FREE or LIMITED, joint-local axes (from localFrameA) are used so that
//   the correct axis is freed/limited.
// ============================================================================

struct D6Joint {
  uint32_t bodyA; // UINT32_MAX = static (anchorA is world pos)
  uint32_t bodyB;
  Vec3 anchorA; // local frame (or world if static)
  Vec3 anchorB;
  Quat localFrameA;      // joint frame orientation relative to body A
  Quat localFrameB;      // joint frame orientation relative to body B
  Quat relativeRotation; // rotA_init^-1 * rotB_init (reference)

  uint32_t linearMotion;  // packed 2 bits per axis: 0=LOCKED, 1=LIMITED, 2=FREE
  uint32_t angularMotion; // packed 2 bits per axis

  // Lambda (dual) multipliers for LOCKED axes
  Vec3 lambdaLinear;
  Vec3 lambdaAngular;

  // Per-axis limits (for LIMITED axes)
  float linearLimitLower[3];
  float linearLimitUpper[3];
  float angularLimitLower[3];
  float angularLimitUpper[3];
  float lambdaLimitLinear[3];
  float lambdaLimitAngular[3];

  // Cone limit (spherical-joint-style cone constraint)
  float coneAngleLimit; // <= 0 means disabled
  Vec3 coneAxisA;       // local cone axis on body A
  float coneLambda;

  // Drive configuration
  // driveFlags bit mask:
  //   0x01=linearX, 0x02=linearY, 0x04=linearZ
  //   0x10=TWIST, 0x20=SLERP, 0x40=SWING1, 0x80=SWING2
  uint32_t driveFlags;
  Vec3 driveLinearVelocity;  // target velocity in joint frame A space
  Vec3 driveAngularVelocity; // target angular velocity in joint frame A space
  Vec3 linearDriveDamping;   // per-axis
  Vec3 angularDriveDamping;  // (twist / swing1 / swing2)
  Vec3 lambdaDriveLinear;
  Vec3 lambdaDriveAngular;

  float rho; // penalty parameter

  // Post-solve revolute motor (matches PhysX post-solve motor approach).
  // Used instead of AL velocity drive to avoid ADMM oscillation when
  // coupled with gear constraints.
  bool motorEnabled;
  float motorTargetVelocity;
  float motorMaxForce;

  // Revolute-specific: hinge angle measurement helpers
  // (set by addRevoluteJoint, unused by other joint types)
  Vec3 hingeAxisB; // hinge axis in B's local frame (normalized)
  Vec3 refAxisA;   // reference axis in A's local frame (perp to hinge)
  Vec3 refAxisB;   // reference axis in B's local frame

  D6Joint() {
    memset(this, 0, sizeof(D6Joint));
    bodyA = UINT32_MAX;
    bodyB = 0;
    localFrameA = Quat();      // identity
    localFrameB = Quat();      // identity
    relativeRotation = Quat(); // identity
    rho = 1e6f;
    motorEnabled = false;
    motorTargetVelocity = 0.0f;
    motorMaxForce = 0.0f;
  }

  uint32_t getLinearMotion(int axis) const {
    return (linearMotion >> (axis * 2)) & 0x3;
  }
  uint32_t getAngularMotion(int axis) const {
    return (angularMotion >> (axis * 2)) & 0x3;
  }

  // Compute hinge angle using reference axes (revolute joints only).
  float computeHingeAngle(const Quat &rotA, const Quat &rotB) const {
    // World-space joint frame A
    Quat frameA = (bodyA == UINT32_MAX) ? localFrameA : rotA * localFrameA;
    Vec3 worldAxisA = frameA.rotate(Vec3(1, 0, 0)); // hinge axis

    Vec3 worldRefA = rotA.rotate(refAxisA);
    Vec3 worldRefB = rotB.rotate(refAxisB);

    // Project refB onto plane perpendicular to hinge axis
    Vec3 projB = worldRefB - worldAxisA * worldRefB.dot(worldAxisA);
    float projLen = projB.length();
    if (projLen < 1e-8f)
      return 0.0f;
    projB = projB * (1.0f / projLen);

    float cosAngle = worldRefA.dot(projB);
    cosAngle = std::max(-1.0f, std::min(1.0f, cosAngle));
    float sinAngle = worldAxisA.dot(worldRefA.cross(projB));
    return atan2f(sinAngle, cosAngle);
  }
};

// Gear Joint: constrains angular velocities of two bodies about their axes.
// Constraint: omegaA . axisA * gearRatio + omegaB . axisB = 0
// Separate from D6 because it constrains velocities, not positions.
struct GearJoint {
  uint32_t bodyA, bodyB;
  Vec3 axisA;
  Vec3 axisB;
  float gearRatio;
  float lambdaGear;
  float rho;
  GearJoint()
      : bodyA(0), bodyB(0), gearRatio(1.f), lambdaGear(0.f), rho(1e5f) {}
};

} // namespace AvbdRef
