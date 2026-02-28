#pragma once
#include "avbd_math.h"
#include <algorithm>
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
      // For box: I = diag(m/12*(h^2+d^2), m/12*(w^2+d^2), m/12*(w^2+h^2))
      // invInertiaWorld = R * I^-1 * R^T (but for axis-aligned initial, just
      // I^-1)
      invInertiaWorld = inertiaTensor.inverse();
    }
  }

  void updateInvInertiaWorld() {
    if (mass <= 0)
      return;
    Mat33 invIlocal = inertiaTensor.inverse();
    // Build rotation matrix from quaternion
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
    // Inertia tensor (not inverse!)
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        M.m[3 + i][3 + j] = inertiaTensor.m[i][j];
    return M;
  }

  // deltaW from initial: 2 * (q * q_initial^-1).xyz
  Vec3 deltaWInitial() const {
    Quat dq = rotation * initialRotation.conjugate();
    if (dq.w < 0) {
      dq = dq * (-1.f);
    }
    return Vec3(dq.x, dq.y, dq.z) * 2.0f;
  }

  // deltaW from inertial: 2 * (q * q_inertial^-1).xyz
  Vec3 deltaWInertial() const {
    Quat dq = rotation * inertialRotation.conjugate();
    if (dq.w < 0) {
      dq = dq * (-1.f);
    }
    return Vec3(dq.x, dq.y, dq.z) * 2.0f;
  }
};

// ====================== Contact Constraint ==================================

struct Contact {
  uint32_t bodyA; // index into bodies[]
  uint32_t bodyB; // UINT32_MAX = static ground

  Vec3 normal; // from B to A (points away from B surface)
  Vec3 rA;     // contact point relative to bodyA center (local frame)
  Vec3 rB;     // contact point relative to bodyB center (local frame)
  float depth; // penetration depth (>=0 means overlapping)
  float friction;

  // Jacobians (computed per step by computeConstraint)
  Vec6 JA;         // Jacobian for bodyA (normal row)
  Vec6 JB;         // Jacobian for bodyB (normal row)
  Vec6 JAt1, JBt1; // tangent 1
  Vec6 JAt2, JBt2; // tangent 2
  float C[3];      // constraint error: [normal, tangent1, tangent2]
  float C0[3]; // constraint error at initial configuration (for alpha blending)

  // Force limits
  float fmin[3], fmax[3];

  // Dual variables (persistent across iterations, warmstarted across frames)
  float lambda[3];
  float penalty[3];
};

// ====================== Joint Constraints ====================================

struct SphericalJoint {
  uint32_t bodyA; // UINT32_MAX = static (anchorA is world pos)
  uint32_t bodyB;
  Vec3 anchorA; // local frame (or world if static)
  Vec3 anchorB; // local frame (or world if static)
  Vec3 lambda;  // AL multiplier (3 components)
  float rho;    // penalty for joint

  // Cone limit
  float coneAngleLimit; // <= 0 means disabled
  Vec3 coneAxisA;       // local cone axis
  float coneLambda;     // AL multiplier for cone limit

  SphericalJoint()
      : bodyA(UINT32_MAX), bodyB(0), rho(1e6f), coneAngleLimit(0.0f),
        coneLambda(0.0f) {}

  float computeConeViolation(const Quat &rotA, const Quat &rotB) const {
    if (coneAngleLimit <= 0.0f)
      return 0.0f;
    Vec3 worldAxisA = rotA.rotate(coneAxisA);
    Vec3 worldAxisB = rotB.rotate(coneAxisA);
    float dotProd = std::max(-1.0f, std::min(1.0f, worldAxisA.dot(worldAxisB)));
    return acosf(dotProd) - coneAngleLimit;
  }
};

struct FixedJoint {
  uint32_t bodyA;
  uint32_t bodyB;
  Vec3 anchorA;
  Vec3 anchorB;
  Quat relativeRotation; // target: rotA^-1 * rotB
  Vec3 lambdaPos;
  Vec3 lambdaRot;
  float rho;

  FixedJoint() : bodyA(UINT32_MAX), bodyB(0), rho(1e6f) {}

  Vec3 computeRotationViolation(const Quat &rotA, const Quat &rotB) const {
    // error = rotA * relativeRotation * rotB^-1
    Quat target = rotA * relativeRotation;
    Quat err = target * rotB.conjugate();
    if (err.w < 0)
      err = err * (-1.f);
    return Vec3(err.x, err.y, err.z) * 2.0f;
  }
};

// D6 Joint: configurable per-axis lock/free + linear/angular drive
struct D6Joint {
  uint32_t bodyA;
  uint32_t bodyB;
  Vec3 anchorA; // local frame (or world if static)
  Vec3 anchorB;
  uint32_t linearMotion;  // packed 2 bits per axis: 0=LOCKED, 1=LIMITED, 2=FREE
  uint32_t angularMotion; // same encoding
  Vec3 lambdaLinear;      // AL multiplier for locked linear DOFs
  Vec3 lambdaAngular;     // AL multiplier for locked angular DOFs
  float rho;

  // Drive configuration
  // driveFlags bit mask:
  //   0x01=X, 0x02=Y, 0x04=Z (linear drives)
  //   0x10=TWIST, 0x20=SLERP, 0x40=SWING1, 0x80=SWING2 (angular drives)
  uint32_t driveFlags;
  Vec3 driveLinearVelocity;  // target linear velocity (joint frame A space)
  Vec3 driveAngularVelocity; // target angular velocity (joint frame A space)
  Vec3 linearDriveDamping;   // per-axis linear drive damping
  Vec3 angularDriveDamping;  // per-axis angular drive damping
                             // (twist/swing1/swing2)
  Quat localFrameA;          // joint frame orientation relative to body A

  // AL multipliers for velocity-level drive constraints
  Vec3 lambdaDriveLinear;  // λ for linear velocity drive (per-axis)
  Vec3 lambdaDriveAngular; // λ for angular velocity drive (twist/swing1/swing2)

  D6Joint()
      : bodyA(UINT32_MAX), bodyB(0), linearMotion(0), angularMotion(0x2A),
        rho(1e6f), driveFlags(0) {}

  uint32_t getLinearMotion(int axis) const {
    return (linearMotion >> (axis * 2)) & 0x3;
  }
  uint32_t getAngularMotion(int axis) const {
    return (angularMotion >> (axis * 2)) & 0x3;
  }
};

// Gear Joint: constrains angular velocities of two bodies about their axes
// Constraint: omegaA · axisA * gearRatio + omegaB · axisB = 0
// So if gearRatio = -1: A and B spin opposite directions at equal speed (meshed
// gears)
//    if gearRatio = 0.5: B spins half as fast as A (reduction gear)
struct GearJoint {
  uint32_t bodyA, bodyB;
  Vec3 axisA;       // Rotation axis in A's local frame (normalized)
  Vec3 axisB;       // Rotation axis in B's local frame (normalized)
  float gearRatio;  // Constraint: (omega_A * axisA) * gearRatio + (omega_B *
                    // axisB) = 0
  float lambdaGear; // AL multiplier
  float rho;
  GearJoint()
      : bodyA(0), bodyB(0), gearRatio(1.f), lambdaGear(0.f), rho(1e5f) {}
};

} // namespace AvbdRef
