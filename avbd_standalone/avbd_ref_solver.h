// =============================================================================
// AVBD Reference Solver -- Standalone, zero-dependency implementation
//
// Strictly follows IsaacLagoy/AVBD3D solver.cpp / solver.h
// No PhysX dependency. Only standard C++ math.
//
// Usage:
//   AvbdRef::Solver solver;
//   solver.gravity = {0, -9.8f, 0};
//   solver.addBody(pos, rot, halfExtent, density);
//   solver.addContact(bodyA, bodyB, normal, rA, rB, depth);
//   solver.step(dt);
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <vector>

namespace AvbdRef {

// ====================== Minimal math types ==================================

struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
  Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
  Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3& operator+=(const Vec3& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
  Vec3& operator-=(const Vec3& b) { x-=b.x; y-=b.y; z-=b.z; return *this; }
  float dot(const Vec3& b) const { return x*b.x + y*b.y + z*b.z; }
  Vec3 cross(const Vec3& b) const {
    return {y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x};
  }
  float length2() const { return x*x + y*y + z*z; }
  float length() const { return sqrtf(length2()); }
  Vec3 normalized() const { float l = length(); return l > 1e-12f ? *this * (1.f/l) : Vec3(); }
};
inline Vec3 operator*(float s, const Vec3& v) { return v*s; }

struct Quat {
  float w, x, y, z;
  Quat() : w(1), x(0), y(0), z(0) {}
  Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
  Quat conjugate() const { return {w, -x, -y, -z}; }
  // q * p (Hamilton product)
  Quat operator*(const Quat& p) const {
    return {w*p.w - x*p.x - y*p.y - z*p.z,
            w*p.x + x*p.w + y*p.z - z*p.y,
            w*p.y - x*p.z + y*p.w + z*p.x,
            w*p.z + x*p.y - y*p.x + z*p.w};
  }
  Quat operator+(const Quat& p) const { return {w+p.w, x+p.x, y+p.y, z+p.z}; }
  Quat operator-(const Quat& p) const { return {w-p.w, x-p.x, y-p.y, z-p.z}; }
  Quat operator*(float s) const { return {w*s, x*s, y*s, z*s}; }
  float length() const { return sqrtf(w*w + x*x + y*y + z*z); }
  Quat normalized() const { float l = length(); return {w/l, x/l, y/l, z/l}; }
  Vec3 rotate(const Vec3& v) const {
    // q * (0,v) * q^-1
    Quat qv(0, v.x, v.y, v.z);
    Quat r = *this * qv * conjugate();
    return {r.x, r.y, r.z};
  }
};

struct Mat33 {
  float m[3][3]; // m[row][col]
  Mat33() { for(int i=0;i<3;i++) for(int j=0;j<3;j++) m[i][j] = 0; }
  static Mat33 diag(float a, float b, float c) {
    Mat33 r; r.m[0][0]=a; r.m[1][1]=b; r.m[2][2]=c; return r;
  }
  Vec3 operator*(const Vec3& v) const {
    return {m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z};
  }
  Mat33 inverse() const;
  Mat33 transpose() const {
    Mat33 r;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) r.m[i][j] = m[j][i];
    return r;
  }
  Mat33 mul(const Mat33& b) const {  // matrix × matrix
    Mat33 r;
    for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
        r.m[i][j] = m[i][0]*b.m[0][j] + m[i][1]*b.m[1][j] + m[i][2]*b.m[2][j];
    return r;
  }
  Mat33 operator-(const Mat33& b) const {
    Mat33 r;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) r.m[i][j] = m[i][j] - b.m[i][j];
    return r;
  }
  Mat33 operator+(const Mat33& b) const {
    Mat33 r;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) r.m[i][j] = m[i][j] + b.m[i][j];
    return r;
  }
  Mat33 operator*(float s) const {
    Mat33 r;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) r.m[i][j] = m[i][j] * s;
    return r;
  }
};

inline Mat33 Mat33::inverse() const {
  Mat33 inv;
  float det = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
  if (fabsf(det) < 1e-20f) return Mat33::diag(0,0,0);
  float invDet = 1.f / det;
  inv.m[0][0] = (m[1][1]*m[2][2]-m[1][2]*m[2][1]) * invDet;
  inv.m[0][1] = (m[0][2]*m[2][1]-m[0][1]*m[2][2]) * invDet;
  inv.m[0][2] = (m[0][1]*m[1][2]-m[0][2]*m[1][1]) * invDet;
  inv.m[1][0] = (m[1][2]*m[2][0]-m[1][0]*m[2][2]) * invDet;
  inv.m[1][1] = (m[0][0]*m[2][2]-m[0][2]*m[2][0]) * invDet;
  inv.m[1][2] = (m[0][2]*m[1][0]-m[0][0]*m[1][2]) * invDet;
  inv.m[2][0] = (m[1][0]*m[2][1]-m[1][1]*m[2][0]) * invDet;
  inv.m[2][1] = (m[0][1]*m[2][0]-m[0][0]*m[2][1]) * invDet;
  inv.m[2][2] = (m[0][0]*m[1][1]-m[0][1]*m[1][0]) * invDet;
  return inv;
}

// ====================== 6x6 types ==========================================

struct Vec6 {
  float v[6];
  Vec6() { for(int i=0;i<6;i++) v[i]=0; }
  Vec6(const Vec3& lin, const Vec3& ang) {
    v[0]=lin.x; v[1]=lin.y; v[2]=lin.z;
    v[3]=ang.x; v[4]=ang.y; v[5]=ang.z;
  }
  float& operator[](int i) { return v[i]; }
  float operator[](int i) const { return v[i]; }
  Vec6 operator+(const Vec6& b) const { Vec6 r; for(int i=0;i<6;i++) r.v[i]=v[i]+b.v[i]; return r; }
  Vec6& operator+=(const Vec6& b) { for(int i=0;i<6;i++) v[i]+=b.v[i]; return *this; }
  Vec6 operator*(float s) const { Vec6 r; for(int i=0;i<6;i++) r.v[i]=v[i]*s; return r; }
  Vec3 linear() const { return {v[0], v[1], v[2]}; }
  Vec3 angular() const { return {v[3], v[4], v[5]}; }
};
inline float dot(const Vec6& a, const Vec6& b) {
  float s = 0; for(int i=0;i<6;i++) s += a[i]*b[i]; return s;
}

struct Mat66 {
  float m[6][6]; // m[row][col]
  Mat66() { for(int i=0;i<6;i++) for(int j=0;j<6;j++) m[i][j]=0; }
  Mat66 operator/(float s) const { Mat66 r; float inv=1.f/s; for(int i=0;i<6;i++) for(int j=0;j<6;j++) r.m[i][j]=m[i][j]*inv; return r; }
  Vec6 operator*(const Vec6& v) const {
    Vec6 r;
    for(int i=0;i<6;i++) { float s=0; for(int j=0;j<6;j++) s += m[i][j]*v[j]; r[i]=s; }
    return r;
  }
  Mat66& operator+=(const Mat66& b) { for(int i=0;i<6;i++) for(int j=0;j<6;j++) m[i][j]+=b.m[i][j]; return *this; }
};

// outer(a, b) = a * b^T (6x6 rank-1)
inline Mat66 outer(const Vec6& a, const Vec6& b) {
  Mat66 r;
  for(int i=0;i<6;i++) for(int j=0;j<6;j++) r.m[i][j] = a[i]*b[j];
  return r;
}

// ====================== LDLT solver (ref: ldlt.cpp) =========================

inline Vec6 solveLDLT(const Mat66& lhs, const Vec6& rhs) {
  // LDL^T decomposition
  float L[6][6] = {};
  float D[6] = {};

  for (int i = 0; i < 6; i++) {
    float sum = 0;
    for (int j = 0; j < i; j++) sum += L[i][j] * L[i][j] * D[j];
    D[i] = lhs.m[i][i] - sum;
    if (fabsf(D[i]) < 1e-12f) D[i] = 1e-12f; // regularize

    L[i][i] = 1.0f;
    for (int j = i+1; j < 6; j++) {
      float s = lhs.m[j][i];
      for (int k = 0; k < i; k++) s -= L[j][k] * L[i][k] * D[k];
      L[j][i] = s / D[i];
    }
  }

  // Forward: Ly = b
  Vec6 y;
  for (int i = 0; i < 6; i++) {
    float s = 0;
    for (int j = 0; j < i; j++) s += L[i][j] * y[j];
    y[i] = rhs[i] - s;
  }

  // Diagonal: Dz = y
  Vec6 z;
  for (int i = 0; i < 6; i++) z[i] = y[i] / D[i];

  // Backward: L^T x = z
  Vec6 x;
  for (int i = 5; i >= 0; i--) {
    float s = 0;
    for (int j = i+1; j < 6; j++) s += L[j][i] * x[j];
    x[i] = z[i] - s;
  }
  return x;
}

// ====================== Rigid Body ==========================================

struct Body {
  Vec3 position;
  Quat rotation;
  Vec3 linearVelocity;
  Vec3 angularVelocity;
  Vec3 prevLinearVelocity;

  Vec3 initialPosition;  // position at start of step (before warmstart)
  Quat initialRotation;
  Vec3 inertialPosition;  // prediction = pos + v*dt + g*dt^2
  Quat inertialRotation;

  float mass;       // 0 = static
  Mat33 inertiaTensor; // in local frame
  float friction;
  Vec3 halfExtent;   // box half-extents (for collision queries)

  // Derived (computed at init)
  float invMass;
  Mat33 invInertiaWorld; // inverse inertia in world frame

  void computeDerived() {
    invMass = (mass > 0) ? 1.0f / mass : 0.0f;
    if (mass > 0) {
      // For box: I = diag(m/12*(h^2+d^2), m/12*(w^2+d^2), m/12*(w^2+h^2))
      // invInertiaWorld = R * I^-1 * R^T (but for axis-aligned initial, just I^-1)
      invInertiaWorld = inertiaTensor.inverse();
    }
  }

  void updateInvInertiaWorld() {
    if (mass <= 0) return;
    // R * I_local^-1 * R^T
    // For simplicity, compute I_local^-1 then rotate
    Mat33 invIlocal = inertiaTensor.inverse();
    // R * invIlocal * R^T
    // Build rotation matrix from quaternion
    Mat33 R;
    float qw=rotation.w, qx=rotation.x, qy=rotation.y, qz=rotation.z;
    R.m[0][0] = 1-2*(qy*qy+qz*qz); R.m[0][1] = 2*(qx*qy-qz*qw);   R.m[0][2] = 2*(qx*qz+qy*qw);
    R.m[1][0] = 2*(qx*qy+qz*qw);   R.m[1][1] = 1-2*(qx*qx+qz*qz); R.m[1][2] = 2*(qy*qz-qx*qw);
    R.m[2][0] = 2*(qx*qz-qy*qw);   R.m[2][1] = 2*(qy*qz+qx*qw);   R.m[2][2] = 1-2*(qx*qx+qy*qy);
    // R * invIlocal
    Mat33 RI;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
      RI.m[i][j] = 0;
      for(int k=0;k<3;k++) RI.m[i][j] += R.m[i][k] * invIlocal.m[k][j];
    }
    // (R * invIlocal) * R^T
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
      invInertiaWorld.m[i][j] = 0;
      for(int k=0;k<3;k++) invInertiaWorld.m[i][j] += RI.m[i][k] * R.m[j][k]; // R^T
    }
  }

  Mat66 getMassMatrix() const {
    Mat66 M;
    for(int i=0;i<3;i++) M.m[i][i] = mass;
    // Inertia tensor (not inverse!)
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) M.m[3+i][3+j] = inertiaTensor.m[i][j];
    return M;
  }

  // deltaW from initial: 2 * (q * q_initial^-1).xyz
  Vec3 deltaWInitial() const {
    Quat dq = rotation * initialRotation.conjugate();
    if (dq.w < 0) { dq = dq * (-1.f); }
    return Vec3(dq.x, dq.y, dq.z) * 2.0f;
  }

  // deltaW from inertial: 2 * (q * q_inertial^-1).xyz
  Vec3 deltaWInertial() const {
    Quat dq = rotation * inertialRotation.conjugate();
    if (dq.w < 0) { dq = dq * (-1.f); }
    return Vec3(dq.x, dq.y, dq.z) * 2.0f;
  }
};

// ====================== Contact Constraint ==================================

struct Contact {
  uint32_t bodyA;  // index into bodies[]
  uint32_t bodyB;  // UINT32_MAX = static ground

  Vec3 normal;   // from B to A (points away from B surface)
  Vec3 rA;       // contact point relative to bodyA center (local frame)
  Vec3 rB;       // contact point relative to bodyB center (local frame)
  float depth;   // penetration depth (>=0 means overlapping)
  float friction;

  // Jacobians (computed per step by computeConstraint)
  Vec6 JA;       // Jacobian for bodyA (normal row)
  Vec6 JB;       // Jacobian for bodyB (normal row)
  Vec6 JAt1, JBt1; // tangent 1
  Vec6 JAt2, JBt2; // tangent 2
  float C[3];      // constraint error: [normal, tangent1, tangent2]
  float C0[3];     // constraint error at initial configuration (for alpha blending)

  // Force limits
  float fmin[3], fmax[3];

  // Dual variables (persistent across iterations, warmstarted across frames)
  float lambda[3];
  float penalty[3];
};

// ====================== Joint Constraints ====================================

struct SphericalJoint {
  uint32_t bodyA;  // UINT32_MAX = static (anchorA is world pos)
  uint32_t bodyB;
  Vec3 anchorA;    // local frame (or world if static)
  Vec3 anchorB;    // local frame (or world if static)
  Vec3 lambda;     // AL multiplier (3 components)
  float rho;       // penalty for joint

  SphericalJoint() : bodyA(UINT32_MAX), bodyB(0), rho(1e6f) {}
};

struct FixedJoint {
  uint32_t bodyA;
  uint32_t bodyB;
  Vec3 anchorA;
  Vec3 anchorB;
  Quat relativeRotation;  // target: rotA^-1 * rotB
  Vec3 lambdaPos;
  Vec3 lambdaRot;
  float rho;

  FixedJoint() : bodyA(UINT32_MAX), bodyB(0), rho(1e6f) {}

  Vec3 computeRotationViolation(const Quat& rotA, const Quat& rotB) const {
    // error = rotA * relativeRotation * rotB^-1
    Quat target = rotA * relativeRotation;
    Quat err = target * rotB.conjugate();
    if (err.w < 0) err = err * (-1.f);
    return Vec3(err.x, err.y, err.z) * 2.0f;
  }
};

// D6 Joint: configurable per-axis lock/free + angular drive
// For SnippetJoint: linearMotion=0 (all locked) + SLERP angular damping
struct D6Joint {
  uint32_t bodyA;
  uint32_t bodyB;
  Vec3 anchorA;         // local frame (or world if static)
  Vec3 anchorB;
  uint32_t linearMotion;  // packed 2 bits per axis: 0=LOCKED, 1=LIMITED, 2=FREE
  uint32_t angularMotion; // same encoding
  Vec3 lambdaLinear;    // AL multiplier for locked linear DOFs
  Vec3 lambdaAngular;   // AL multiplier for locked angular DOFs
  float rho;

  // Angular drive (SLERP)
  float angularDamping;    // uniform SLERP damping coefficient
  Vec3 driveTargetAngVel;  // target angular velocity (world frame)

  D6Joint() : bodyA(UINT32_MAX), bodyB(0), linearMotion(0), angularMotion(0x2A),
              rho(1e6f), angularDamping(0.0f) {}

  uint32_t getLinearMotion(int axis) const { return (linearMotion >> (axis * 2)) & 0x3; }
  uint32_t getAngularMotion(int axis) const { return (angularMotion >> (axis * 2)) & 0x3; }
};

// ====================== Solver ==============================================

static constexpr float PENALTY_MIN = 1000.0f;
static constexpr float PENALTY_MAX = 1e9f;

struct Solver {
  Vec3 gravity = {0, -9.8f, 0};
  int iterations = 10;
  float alpha = 0.95f;   // stabilization
  float beta = 1000.0f;  // penalty growth rate
  float gamma = 0.99f;   // warmstart decay
  float penaltyScale = 0.25f; // body-ground penalty floor
  float penaltyScaleDynDyn = 0.05f; // dynamic-dynamic penalty (5x softer for natural deformation)
  int   propagationDepth = 4;   // graph-propagation depth (Jacobi iterations for effective mass)
  float propagationDecay = 0.5f; // per-edge decay factor for mass propagation
  float dt = 1.0f / 60.0f;
  bool use3x3Solve = false;  // false=6x6 LDLT (default), true=block-elim 3x3 (PhysX fallback)
  bool verbose = false;   // per-iteration logging

  std::vector<Body> bodies;
  std::vector<Contact> contacts;
  std::vector<SphericalJoint> sphericalJoints;
  std::vector<FixedJoint> fixedJoints;
  std::vector<D6Joint> d6Joints;

  // -----------------------------------------------------------------------
  // Add a spherical (ball-socket) joint
  // -----------------------------------------------------------------------
  void addSphericalJoint(uint32_t bodyA, uint32_t bodyB,
                         Vec3 anchorA, Vec3 anchorB, float rho_ = 1e6f) {
    SphericalJoint j;
    j.bodyA = bodyA; j.bodyB = bodyB;
    j.anchorA = anchorA; j.anchorB = anchorB;
    j.lambda = Vec3(); j.rho = rho_;
    sphericalJoints.push_back(j);
  }

  // -----------------------------------------------------------------------
  // Add a fixed joint (6-DOF: position + rotation locked)
  // -----------------------------------------------------------------------
  void addFixedJoint(uint32_t bodyA, uint32_t bodyB,
                     Vec3 anchorA, Vec3 anchorB, float rho_ = 1e6f) {
    FixedJoint j;
    j.bodyA = bodyA; j.bodyB = bodyB;
    j.anchorA = anchorA; j.anchorB = anchorB;
    j.lambdaPos = Vec3(); j.lambdaRot = Vec3(); j.rho = rho_;
    // Compute initial relative rotation
    Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
    Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
    j.relativeRotation = rotA.conjugate() * rotB;
    fixedJoints.push_back(j);
  }

  // -----------------------------------------------------------------------
  // Add a D6 joint (configurable DOFs + angular drive)
  // linearMotion: 0=all locked (default), 0x2A=all free
  // angularMotion: 0x2A=all free (default), 0=all locked
  // -----------------------------------------------------------------------
  void addD6Joint(uint32_t bodyA, uint32_t bodyB,
                  Vec3 anchorA, Vec3 anchorB,
                  uint32_t linearMotion_ = 0, uint32_t angularMotion_ = 0x2A,
                  float angularDamping_ = 0.0f, float rho_ = 1e6f) {
    D6Joint j;
    j.bodyA = bodyA; j.bodyB = bodyB;
    j.anchorA = anchorA; j.anchorB = anchorB;
    j.linearMotion = linearMotion_; j.angularMotion = angularMotion_;
    j.lambdaLinear = Vec3(); j.lambdaAngular = Vec3();
    j.rho = rho_;
    j.angularDamping = angularDamping_;
    j.driveTargetAngVel = Vec3();
    d6Joints.push_back(j);
  }

  // -----------------------------------------------------------------------
  // Add a box body (halfExtent per axis)
  // density < 0 => static
  // -----------------------------------------------------------------------
  uint32_t addBody(Vec3 pos, Quat rot, Vec3 halfExtent, float density, float fric = 0.5f) {
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
      float sx = 2*halfExtent.x, sy = 2*halfExtent.y, sz = 2*halfExtent.z;
      float Ixx = b.mass / 12.0f * (sy*sy + sz*sz);
      float Iyy = b.mass / 12.0f * (sx*sx + sz*sz);
      float Izz = b.mass / 12.0f * (sx*sx + sy*sy);
      b.inertiaTensor = Mat33::diag(Ixx, Iyy, Izz);
    } else {
      b.mass = 0;
      b.inertiaTensor = Mat33::diag(0,0,0);
    }
    b.computeDerived();

    uint32_t idx = (uint32_t)bodies.size();
    bodies.push_back(b);
    return idx;
  }

  // -----------------------------------------------------------------------
  // Add a contact (call after collision detection, before step)
  // rA, rB in local body frame
  // normal points from B to A
  // depth > 0 means penetrating
  // -----------------------------------------------------------------------
  void addContact(uint32_t bodyA, uint32_t bodyB, Vec3 normal, Vec3 rA, Vec3 rB, float depth, float fric = 0.5f) {
    Contact c;
    c.bodyA = bodyA;
    c.bodyB = bodyB;
    c.normal = normal;
    c.rA = rA;
    c.rB = rB;
    c.depth = depth;
    c.friction = fric;
    for(int i=0;i<3;i++) { c.lambda[i]=0; c.penalty[i]=PENALTY_MIN; c.fmin[i]=0; c.fmax[i]=0; }
    // Normal: fmin=-inf, fmax=0 (can push, not pull)
    c.fmin[0] = -1e30f; c.fmax[0] = 0.0f;
    // Friction: bounds set dynamically
    c.fmin[1] = 0; c.fmax[1] = 0;
    c.fmin[2] = 0; c.fmax[2] = 0;
    contacts.push_back(c);
  }

  // -----------------------------------------------------------------------
  // Compute constraint values and Jacobians for a contact
  // (ref: manifold.cpp computeConstraint + computeDerivatives)
  // -----------------------------------------------------------------------
  void computeConstraint(Contact& c) {
    Body& bA = bodies[c.bodyA];
    bool bStatic = (c.bodyB == UINT32_MAX);
    Body* pB = bStatic ? nullptr : &bodies[c.bodyB];

    // World-space contact points
    Vec3 wA = bA.position + bA.rotation.rotate(c.rA);
    Vec3 wB = bStatic ? c.rB : (pB->position + pB->rotation.rotate(c.rB));

    // Normal constraint: C = dot(wA - wB, normal) - depth
    // When C < 0 => penetrating => needs correction
    // Actually in ref: C = dot(dpA*JA + dpB*JB) with alpha blending
    // But simpler direct evaluation:
    Vec3 diff = wA - wB;
    float Cn = diff.dot(c.normal) - c.depth;

    // Build Jacobians (6-vectors: [linear, angular])
    Vec3 rAw = bA.rotation.rotate(c.rA);
    Vec3 rBw = bStatic ? Vec3() : pB->rotation.rotate(c.rB);

    // Normal Jacobian for bodyA: J_A = [n, rA x n]
    c.JA = Vec6(c.normal, rAw.cross(c.normal));
    c.JB = bStatic ? Vec6() : Vec6(-c.normal, (-rBw).cross(c.normal));
    // Actually JB should be: [-n, -rBw x n] = [-n, rBw.cross(-n)]
    // Let me be precise: ref uses J for bodyA, -J for bodyB
    // JB = -JA in structure, but with rB cross term
    if (!bStatic) {
      c.JB = Vec6(Vec3()-c.normal, Vec3()-rBw.cross(c.normal));
    }

    // Tangent directions
    Vec3 t1, t2;
    if (fabsf(c.normal.y) > 0.9f) {
      t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
    } else {
      t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
    }
    t2 = c.normal.cross(t1);

    c.JAt1 = Vec6(t1, rAw.cross(t1));
    c.JBt1 = bStatic ? Vec6() : Vec6(Vec3()-t1, Vec3()-rBw.cross(t1));
    c.JAt2 = Vec6(t2, rAw.cross(t2));
    c.JBt2 = bStatic ? Vec6() : Vec6(Vec3()-t2, Vec3()-rBw.cross(t2));

    // Constraint values using alpha blending (ref: manifold.cpp L160-168)
    Vec6 dpA(bA.position - bA.initialPosition, bA.deltaWInitial());
    Vec6 dpB;
    if (!bStatic) {
      dpB = Vec6(pB->position - pB->initialPosition, pB->deltaWInitial());
    }

    // C[0] = C0*(1-alpha) + JA.dpA + JB.dpB  (normal)
    c.C[0] = c.C0[0] * (1.0f - alpha) + dot(c.JA, dpA) + dot(c.JB, dpB);
    c.C[1] = c.C0[1] * (1.0f - alpha) + dot(c.JAt1, dpA) + dot(c.JBt1, dpB);
    c.C[2] = c.C0[2] * (1.0f - alpha) + dot(c.JAt2, dpA) + dot(c.JBt2, dpB);

    // Update friction bounds from normal lambda
    float frictionBound = fabsf(c.lambda[0]) * c.friction;
    c.fmax[1] = frictionBound;
    c.fmin[1] = -frictionBound;
    c.fmax[2] = frictionBound;
    c.fmin[2] = -frictionBound;
  }

  // -----------------------------------------------------------------------
  // Compute C0 at the start of the step (before any position changes)
  // -----------------------------------------------------------------------
  void computeC0(Contact& c) {
    Body& bA = bodies[c.bodyA];
    bool bStatic = (c.bodyB == UINT32_MAX);
    Body* pB = bStatic ? nullptr : &bodies[c.bodyB];

    Vec3 wA = bA.position + bA.rotation.rotate(c.rA);
    Vec3 wB = bStatic ? c.rB : (pB->position + pB->rotation.rotate(c.rB));

    c.C0[0] = (wA - wB).dot(c.normal) - c.depth;
    // Tangent C0: relative tangential displacement (initially 0 for new contacts)
    c.C0[1] = 0.0f;
    c.C0[2] = 0.0f;
  }

  // -----------------------------------------------------------------------
  // Warmstart step: decay lambda and penalty (ref: solver.cpp L60-75)
  // -----------------------------------------------------------------------
  void warmstart() {
    for (auto& c : contacts) {
      for (int i = 0; i < 3; i++) {
        c.lambda[i] = c.lambda[i] * alpha * gamma;
        c.penalty[i] = std::clamp(c.penalty[i] * gamma, PENALTY_MIN, PENALTY_MAX);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main solver step (ref: solver.cpp L39-177)
  // -----------------------------------------------------------------------
  void step(float dt_) {
    dt = dt_;
    float invDt = 1.0f / dt;
    float dt2 = dt * dt;

    // =====================================================================
    // 1. Initialize forces / collision (warmstart)
    // =====================================================================
    warmstart();

    // Enforce adaptive penalty floor based on body mass.
    //
    // Two-tier penalty scaling:
    //   Body-ground: penaltyScale (0.25) -- full stiffness for stacking.
    //   Dynamic-dynamic: penaltyScaleDynDyn (0.05) with max effective mass
    //     -- softer scale but uses the heavier body's mass so extreme
    //        mass ratios don't cause tunneling.
    //
    // Graph-propagated effective mass (Neumann series of Schur complement):
    //   A mesh node's effective resistance to contact forces depends on
    //   the entire joint sub-graph it belongs to, not just its direct
    //   neighbors.  We propagate mass through the joint graph using
    //   Jacobi iteration:
    //     m_eff^(k+1)[i] = m[i] + decay * sum(m_eff^(k)[j], j in N(i))
    //   After D iterations, each node accumulates the mass of its D-hop
    //   neighborhood with exponential decay per hop -- a truncated
    //   Neumann series approximation of the joint-graph Schur complement.
    //
    //   2D mesh interior (valence=4, depth=4, decay=0.5): m_eff ~ 31m
    //   Chain interior  (valence=2, depth=4, decay=0.5): m_eff ~ 5m
    //   Free body       (valence=0):                     m_eff = m
    //
    // For dynamic-dynamic contacts, the penalty floor uses max(augA,augB)
    // instead of geometric mean.  Rationale: the contact must be stiff
    // enough to decelerate the HEAVIER body within one timestep.  AVBD's
    // implicit solve keeps this stable regardless of the mass ratio.

    // Step 1: Build adjacency list from joints
    uint32_t nBodies = (uint32_t)bodies.size();
    std::vector<std::vector<uint32_t>> adj(nBodies);
    auto addEdge = [&](uint32_t a, uint32_t b) {
      if (a < nBodies && b < nBodies && a != UINT32_MAX && b != UINT32_MAX) {
        adj[a].push_back(b);
        adj[b].push_back(a);
      }
    };
    for (const auto& j : sphericalJoints) addEdge(j.bodyA, j.bodyB);
    for (const auto& j : fixedJoints)     addEdge(j.bodyA, j.bodyB);
    for (const auto& j : d6Joints)        addEdge(j.bodyA, j.bodyB);

    // Step 2: Jacobi propagation of effective mass
    std::vector<float> mEff(nBodies);
    for (uint32_t i = 0; i < nBodies; i++) mEff[i] = bodies[i].mass;

    for (int d = 0; d < propagationDepth; d++) {
      std::vector<float> mNext(nBodies);
      for (uint32_t i = 0; i < nBodies; i++) {
        float neighborSum = 0.0f;
        for (uint32_t nb : adj[i]) neighborSum += mEff[nb];
        mNext[i] = bodies[i].mass + propagationDecay * neighborSum;
      }
      mEff = mNext;
    }

    // Step 3: set penalty floor using propagated effective mass
    for (auto& c : contacts) {
      float augA = mEff[c.bodyA];
      float augB = (c.bodyB != UINT32_MAX) ? mEff[c.bodyB] : 0.0f;
      float massB = (c.bodyB != UINT32_MAX) ? bodies[c.bodyB].mass : 0.0f;

      float effectiveMass, scale;
      if (c.bodyB != UINT32_MAX && massB > 0.0f) {
        effectiveMass = std::max(augA, augB);  // heavier body determines floor
        scale = penaltyScaleDynDyn;            // softer for deformation
      } else {
        effectiveMass = augA;                  // body-ground (propagated)
        scale = penaltyScale;                  // full stiffness
      }
      float penFloor = std::max(PENALTY_MIN, scale * effectiveMass / dt2);
      for (int i = 0; i < 3; i++)
        c.penalty[i] = std::max(c.penalty[i], penFloor);
    }

    // Compute C0 for alpha blending
    for (auto& c : contacts) {
      computeC0(c);
    }

    // =====================================================================
    // 2. Warmstart bodies (ref: solver.cpp L76-98)
    //    Compute inertial prediction, then adaptive warmstart position
    // =====================================================================
    for (auto& body : bodies) {
      if (body.mass <= 0) continue;
      body.updateInvInertiaWorld();

      // Inertial prediction
      body.inertialPosition = body.position + body.linearVelocity * dt + gravity * dt2;
      Quat angVel(0, body.angularVelocity.x, body.angularVelocity.y, body.angularVelocity.z);
      body.inertialRotation = (body.rotation + angVel * body.rotation * (0.5f * dt)).normalized();

      // Adaptive warmstarting
      Vec3 accel = (body.linearVelocity - body.prevLinearVelocity) * invDt;
      float gravLen = gravity.length();
      float accelWeight = 0.0f;
      if (gravLen > 1e-6f) {
        Vec3 gravDir = gravity.normalized();
        accelWeight = std::clamp(accel.dot(gravDir) / gravLen, 0.0f, 1.0f);
      }

      body.initialPosition = body.position;
      body.initialRotation = body.rotation;

      body.position = body.position + body.linearVelocity * dt + gravity * (accelWeight * dt2);
      body.rotation = body.inertialRotation;
    }

    // =====================================================================
    // 3. Main solver loop (ref: solver.cpp L103-164)
    //    Each iteration: primal update (per body) + dual update (per force)
    // =====================================================================
    for (int it = 0; it < iterations; it++) {

      // --- Primal update (per body) --- (ref: solver.cpp L107-138)
      for (uint32_t bi = 0; bi < (uint32_t)bodies.size(); bi++) {
        Body& body = bodies[bi];
        if (body.mass <= 0) continue;

        // LHS = M / h^2
        Mat66 lhs = body.getMassMatrix() / dt2;

        // RHS = (M/h^2) * [pos - inertialPos, deltaWInertial]
        Vec6 disp(body.position - body.inertialPosition, body.deltaWInertial());
        Vec6 rhs = lhs * disp;

        // Iterate over contacts on this body
        //
        // Per-body primal-only boost (0.5%) as a safety net:
        // prevents the heavy body from slowly drifting through contacts
        // during oscillations (where contact count drops temporarily).
        // With 0.5%: for a 410 kg ball, boost = 7,373 per contact.
        // The DUAL step still uses the shared (un-boosted) penalty.
        const float contactBoostFraction = 0.005f;
        float boostFloor = contactBoostFraction * body.mass / dt2;

        for (auto& c : contacts) {
          bool isA = (c.bodyA == bi);
          bool isB = (c.bodyB == bi);
          if (!isA && !isB) continue;

          // Recompute constraint
          computeConstraint(c);

          // For each constraint row (normal + 2 tangent)
          for (int i = 0; i < 3; i++) {
            Vec6 J = isA
              ? (i==0 ? c.JA : (i==1 ? c.JAt1 : c.JAt2))
              : (i==0 ? c.JB : (i==1 ? c.JBt1 : c.JBt2));

            // Primal penalty: shared geoMean + small per-body boost
            float pen = std::max(c.penalty[i], boostFloor);

            // Clamped force: f = clamp(penalty*C + lambda, fmin, fmax)
            float f = std::clamp(pen * c.C[i] + c.lambda[i], c.fmin[i], c.fmax[i]);

            // RHS += J * f   (Eq. 13)
            rhs += J * f;

            // LHS += outer(J, J * penalty)  (Eq. 17)
            lhs += outer(J, J * pen);
          }
        }

        // ---------------------------------------------------------------
        // Accumulate SPHERICAL JOINT contributions (3 rows per joint)
        //   C_k = (anchorA_w - anchorB_w) . e_k
        //   Jacobian for this body:
        //     Body A: J = [+e_k, +(r_A x e_k)]
        //     Body B: J = [-e_k, -(r_B x e_k)]
        //   Force: f = rho * C + lambda   (equality: no clamping)
        // ---------------------------------------------------------------
        for (auto& jnt : sphericalJoints) {
          bool isA = (jnt.bodyA == bi);
          bool isB = (jnt.bodyB == bi);
          if (!isA && !isB) continue;

          bool otherStatic = isA
            ? (jnt.bodyB == UINT32_MAX)
            : (jnt.bodyA == UINT32_MAX);

          Vec3 worldAnchorA, worldAnchorB, r;
          if (isA) {
            r = body.rotation.rotate(jnt.anchorA);
            worldAnchorA = body.position + r;
            worldAnchorB = otherStatic ? jnt.anchorB
              : bodies[jnt.bodyB].position + bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
          } else {
            r = body.rotation.rotate(jnt.anchorB);
            worldAnchorB = body.position + r;
            worldAnchorA = otherStatic ? jnt.anchorA
              : bodies[jnt.bodyA].position + bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
          }

          Vec3 posError = worldAnchorA - worldAnchorB;
          float sign = isA ? 1.0f : -1.0f;

          // Auto-boost penalty: ensure rho >= M/h^2 for good primal convergence
          float effectiveRho = std::max(jnt.rho, body.mass / dt2);

          for (int axis = 0; axis < 3; axis++) {
            Vec3 n(0,0,0);
            (&n.x)[axis] = 1.0f;

            Vec3 gradPos = n * sign;
            Vec3 gradRot = r.cross(n) * sign;
            Vec6 J(gradPos, gradRot);

            float C = posError.dot(n);
            float f = jnt.rho * C + jnt.lambda.dot(n) * ((&n.x)[axis]); // lambda[axis]
            // Simpler: just use component directly
            float lam = (&jnt.lambda.x)[axis];
            f = effectiveRho * C + lam;

            rhs += J * f;
            lhs += outer(J, J * effectiveRho);
          }
        }

        // ---------------------------------------------------------------
        // Accumulate FIXED JOINT contributions (6 rows: 3 pos + 3 rot)
        // ---------------------------------------------------------------
        for (auto& jnt : fixedJoints) {
          bool isA = (jnt.bodyA == bi);
          bool isB = (jnt.bodyB == bi);
          if (!isA && !isB) continue;

          bool otherStatic = isA
            ? (jnt.bodyB == UINT32_MAX)
            : (jnt.bodyA == UINT32_MAX);

          float sign = isA ? 1.0f : -1.0f;

          // --- Position rows (3 DOF) ---
          {
            Vec3 worldAnchorA, worldAnchorB, r;
            if (isA) {
              r = body.rotation.rotate(jnt.anchorA);
              worldAnchorA = body.position + r;
              worldAnchorB = otherStatic ? jnt.anchorB
                : bodies[jnt.bodyB].position + bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
            } else {
              r = body.rotation.rotate(jnt.anchorB);
              worldAnchorB = body.position + r;
              worldAnchorA = otherStatic ? jnt.anchorA
                : bodies[jnt.bodyA].position + bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
            }

            Vec3 posError = worldAnchorA - worldAnchorB;

            // Auto-boost penalty for primal convergence
            float effectiveRho = std::max(jnt.rho, body.mass / dt2);

            for (int axis = 0; axis < 3; axis++) {
              Vec3 n(0,0,0);
              (&n.x)[axis] = 1.0f;

              Vec3 gradPos = n * sign;
              Vec3 gradRot = r.cross(n) * sign;
              Vec6 J(gradPos, gradRot);

              float C = (&posError.x)[axis];
              float lam = (&jnt.lambdaPos.x)[axis];
              float f = effectiveRho * C + lam;

              rhs += J * f;
              lhs += outer(J, J * effectiveRho);
            }
          }

          // --- Rotation rows (3 DOF) ---
          {
            Quat rotA = isA ? body.rotation
              : (otherStatic ? Quat() : bodies[jnt.bodyA].rotation);
            Quat rotB = isA
              ? (otherStatic ? Quat() : bodies[jnt.bodyB].rotation)
              : body.rotation;

            Vec3 rotErr = jnt.computeRotationViolation(rotA, rotB);

            // Same effectiveRho for rotation rows
            float effectiveRhoRot = std::max(jnt.rho, body.mass / dt2);

            for (int axis = 0; axis < 3; axis++) {
              Vec3 n(0,0,0);
              (&n.x)[axis] = 1.0f;

              Vec3 gradPos(0,0,0);
              Vec3 gradRot = n * sign;
              Vec6 J(gradPos, gradRot);

              float C = (&rotErr.x)[axis];
              float lam = (&jnt.lambdaRot.x)[axis];
              float f = effectiveRhoRot * C + lam;

              rhs += J * f;
              lhs += outer(J, J * effectiveRhoRot);
            }
          }
        }

        // ---------------------------------------------------------------
        // Accumulate D6 JOINT contributions
        //   Locked linear DOFs: 3 position rows (same as spherical)
        //   Angular damping (SLERP): adds I_damp/h^2 to angular diagonal
        //     and -I_damp/h^2 * omega_target * h to angular RHS
        //
        //   3x3 path: locked position DOFs handled by GS pass.
        //   Angular DAMPING is still accumulated directly (it's per-body,
        //   no cross-coupling, and only affects angular diagonal).
        // ---------------------------------------------------------------
        for (auto& jnt : d6Joints) {
          bool isA = (jnt.bodyA == bi);
          bool isB = (jnt.bodyB == bi);
          if (!isA && !isB) continue;

          bool otherStatic = isA
            ? (jnt.bodyB == UINT32_MAX)
            : (jnt.bodyA == UINT32_MAX);

          float sign = isA ? 1.0f : -1.0f;

          // --- Locked linear DOFs (position constraint) ---
          {
            Vec3 worldAnchorA, worldAnchorB, r;
            if (isA) {
              r = body.rotation.rotate(jnt.anchorA);
              worldAnchorA = body.position + r;
              worldAnchorB = otherStatic ? jnt.anchorB
                : bodies[jnt.bodyB].position + bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
            } else {
              r = body.rotation.rotate(jnt.anchorB);
              worldAnchorB = body.position + r;
              worldAnchorA = otherStatic ? jnt.anchorA
                : bodies[jnt.bodyA].position + bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
            }

            Vec3 posError = worldAnchorA - worldAnchorB;

            // Auto-boost penalty for primal convergence
            float effectiveRho = std::max(jnt.rho, body.mass / dt2);

            for (int axis = 0; axis < 3; axis++) {
              if (jnt.getLinearMotion(axis) != 0) continue; // only LOCKED

              Vec3 n(0,0,0);
              (&n.x)[axis] = 1.0f;

              Vec3 gradPos = n * sign;
              Vec3 gradRot = r.cross(n) * sign;
              Vec6 J(gradPos, gradRot);

              float C = (&posError.x)[axis];
              float lam = (&jnt.lambdaLinear.x)[axis];
              float f = effectiveRho * C + lam;

              rhs += J * f;
              lhs += outer(J, J * effectiveRho);
            }
          }

          // --- Locked angular DOFs (rotation constraint) ---
          for (int axis = 0; axis < 3; axis++) {
            if (jnt.getAngularMotion(axis) != 0) continue; // only LOCKED
            // TODO: compute angular error for locked angular DOF
            // For now, SnippetJoint D6 has all angular DOFs free, so this is not triggered
          }

          // --- Angular velocity damping (SLERP drive) ---
          // Model: penalize angular velocity omega = deltaTheta / h.
          //
          // Energy: E_damp = 0.5 * c * |omega|^2
          //       = 0.5 * (c / h^2) * |deltaTheta|^2
          // Gradient: g_damp = (c / h^2) * deltaTheta_init
          // Hessian:  H_damp = (c / h^2) * I
          //
          // IMPORTANT: Damping is a LOCAL property of each body, not a
          // constraint between two bodies. Do NOT use the joint sign here!
          // The sign would flip the gradient for body B, making the
          // "damping" add energy instead of dissipating it.
          if (jnt.angularDamping > 0.0f) {
            float dampEff = jnt.angularDamping / (dt * dt);  // c/h^2

            // Angular displacement from initial position this frame
            Vec3 deltaW = body.deltaWInitial();

            for (int axis = 0; axis < 3; axis++) {
              // Add directly to angular Hessian diagonal (no J, no sign)
              lhs.m[3 + axis][3 + axis] += dampEff;

              // RHS: gradient of damping energy
              float dw = (&deltaW.x)[axis];
              (&rhs[3])[axis] += dampEff * dw;
            }
          }
        }

        // Solve and apply
        if (!use3x3Solve) {
          // ---- 6x6 fully-coupled LDLT ----
          Vec6 delta = solveLDLT(lhs, rhs);
          body.position -= delta.linear();
          Quat dq(0, delta[3], delta[4], delta[5]);
          body.rotation = (body.rotation - dq * body.rotation * 0.5f).normalized();
        } else {
          // ---- Decoupled 3x3 (same accumulation as 6x6, simpler solve) ----
          //
          // Same contacts + joints are accumulated into the 6x6 LHS/RHS.
          // We extract the diagonal 3x3 blocks and solve independently.
          //
          // KNOWN LIMITATION: dropping the off-diagonal B block loses the
          // linear-angular coupling from joints with offset anchors.
          // For tightly-coupled meshes (chainmail), this makes contact
          // response ~42x weaker.  Joint chains work fine; mesh impact
          // scenarios need 6x6 (enableLocal6x6Solve = true in PhysX).
          //
          Mat33 Alin, Aang;
          Vec3 rhsLin(rhs[0], rhs[1], rhs[2]);
          Vec3 rhsAng(rhs[3], rhs[4], rhs[5]);
          for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
              Alin.m[r][c] = lhs.m[r][c];
              Aang.m[r][c] = lhs.m[3+r][3+c];
            }
          }
          Vec3 deltaPos = Alin.inverse() * rhsLin;
          Vec3 deltaTheta = Aang.inverse() * rhsAng;
          body.position -= deltaPos;
          Quat dq(0, deltaTheta.x, deltaTheta.y, deltaTheta.z);
          body.rotation = (body.rotation - dq * body.rotation * 0.5f).normalized();
        }
      }

      // --- Dual update (per force/contact) --- (ref: solver.cpp L140-164)
      for (auto& c : contacts) {
        computeConstraint(c);

        for (int i = 0; i < 3; i++) {
          // Lambda update (Eq. 11)
          c.lambda[i] = std::clamp(c.penalty[i] * c.C[i] + c.lambda[i], c.fmin[i], c.fmax[i]);

          // Penalty growth (Eq. 16): if lambda within bounds, penalty grows
          if (c.lambda[i] > c.fmin[i] && c.lambda[i] < c.fmax[i]) {
            c.penalty[i] = std::min(c.penalty[i] + beta * fabsf(c.C[i]), PENALTY_MAX);
          }
        }
      }

      // Per-iteration diagnostics
      if (verbose) {
        float maxViol = 0;
        float maxPen = 0;
        float maxLam = 0;
        for (auto& c : contacts) {
          if (c.C[0] < -maxViol) maxViol = -c.C[0];
          maxPen = std::max(maxPen, c.penalty[0]);
          maxLam = std::max(maxLam, fabsf(c.lambda[0]));
        }
        printf("    it=%d maxViol=%.6f maxPen=%.0f maxLam=%.2f b0y=%.6f\n",
          it, maxViol, maxPen, maxLam,
          bodies.size() > 0 && bodies[0].mass > 0 ? bodies[0].position.y : 0.0f);
      }
    } // end iteration loop

    // =====================================================================
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
          ? 0.0f : bodies[idx].mass;
      };
      auto computeRhoDual = [&](uint32_t idxA, uint32_t idxB, float rho) -> float {
        float mA = getBodyMass(idxA);
        float mB = getBodyMass(idxB);
        // Use smaller dynamic mass (bottleneck for convergence)
        float mEff;
        if (mA <= 0.0f) mEff = mB;
        else if (mB <= 0.0f) mEff = mA;
        else mEff = std::min(mA, mB);
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

      for (auto& jnt : sphericalJoints) {
        float rhoDual = computeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho);
        bool aStatic = (jnt.bodyA == UINT32_MAX);
        bool bStatic = (jnt.bodyB == UINT32_MAX);
        Vec3 wA = aStatic ? jnt.anchorA
          : bodies[jnt.bodyA].position + bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
        Vec3 wB = bStatic ? jnt.anchorB
          : bodies[jnt.bodyB].position + bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
        jnt.lambda = jnt.lambda * lambdaDecay + (wA - wB) * rhoDual;
      }

      for (auto& jnt : fixedJoints) {
        float rhoDual = computeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho);
        bool aStatic = (jnt.bodyA == UINT32_MAX);
        bool bStatic = (jnt.bodyB == UINT32_MAX);
        Vec3 wA = aStatic ? jnt.anchorA
          : bodies[jnt.bodyA].position + bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
        Vec3 wB = bStatic ? jnt.anchorB
          : bodies[jnt.bodyB].position + bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
        Quat rotA = aStatic ? Quat() : bodies[jnt.bodyA].rotation;
        Quat rotB = bStatic ? Quat() : bodies[jnt.bodyB].rotation;
        jnt.lambdaPos = jnt.lambdaPos * lambdaDecay + (wA - wB) * rhoDual;
        jnt.lambdaRot = jnt.lambdaRot * lambdaDecay + jnt.computeRotationViolation(rotA, rotB) * rhoDual;
      }

      for (auto& jnt : d6Joints) {
        float rhoDual = computeRhoDual(jnt.bodyA, jnt.bodyB, jnt.rho);
        bool aStatic = (jnt.bodyA == UINT32_MAX);
        bool bStatic = (jnt.bodyB == UINT32_MAX);
        Vec3 wA = aStatic ? jnt.anchorA
          : bodies[jnt.bodyA].position + bodies[jnt.bodyA].rotation.rotate(jnt.anchorA);
        Vec3 wB = bStatic ? jnt.anchorB
          : bodies[jnt.bodyB].position + bodies[jnt.bodyB].rotation.rotate(jnt.anchorB);
        Vec3 posViol = wA - wB;
        for (int axis = 0; axis < 3; axis++) {
          if (jnt.getLinearMotion(axis) != 0) continue;
          (&jnt.lambdaLinear.x)[axis] = (&jnt.lambdaLinear.x)[axis] * lambdaDecay + (&posViol.x)[axis] * rhoDual;
        }
      }
    }

    // =====================================================================
    // 4. Compute velocities (BDF1) (ref: solver.cpp L166-172)
    // =====================================================================
    for (auto& body : bodies) {
      if (body.mass <= 0) continue;
      body.prevLinearVelocity = body.linearVelocity;
      Vec6 v = Vec6(body.position - body.initialPosition, body.deltaWInitial()) * invDt;
      body.linearVelocity = v.linear();
      body.angularVelocity = v.angular();
    }
  }

  // -----------------------------------------------------------------------
  // Debug print
  // -----------------------------------------------------------------------
  void printState(const char* label) const {
    printf("=== %s ===\n", label);
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
      const Body& b = bodies[i];
      if (b.mass <= 0) { printf("  body[%u] STATIC pos=(%.4f,%.4f,%.4f)\n", i, b.position.x, b.position.y, b.position.z); continue; }
      printf("  body[%u] pos=(%.6f,%.6f,%.6f) vel=(%.4f,%.4f,%.4f) mass=%.1f\n",
        i, b.position.x, b.position.y, b.position.z,
        b.linearVelocity.x, b.linearVelocity.y, b.linearVelocity.z, b.mass);
    }
  }
};

} // namespace AvbdRef
