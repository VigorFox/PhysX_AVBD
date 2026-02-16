#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace AvbdRef {

// ====================== Minimal math types ==================================

struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3 &b) const { return {x + b.x, y + b.y, z + b.z}; }
  Vec3 operator-(const Vec3 &b) const { return {x - b.x, y - b.y, z - b.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3 &operator+=(const Vec3 &b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
  Vec3 &operator-=(const Vec3 &b) {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }
  float dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }
  Vec3 cross(const Vec3 &b) const {
    return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
  }
  float length2() const { return x * x + y * y + z * z; }
  float length() const { return sqrtf(length2()); }
  Vec3 normalized() const {
    float l = length();
    return l > 1e-12f ? *this * (1.f / l) : Vec3();
  }
};
inline Vec3 operator*(float s, const Vec3 &v) { return v * s; }

struct Quat {
  float w, x, y, z;
  Quat() : w(1), x(0), y(0), z(0) {}
  Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
  Quat conjugate() const { return {w, -x, -y, -z}; }
  // q * p (Hamilton product)
  Quat operator*(const Quat &p) const {
    return {w * p.w - x * p.x - y * p.y - z * p.z,
            w * p.x + x * p.w + y * p.z - z * p.y,
            w * p.y - x * p.z + y * p.w + z * p.x,
            w * p.z + x * p.y - y * p.x + z * p.w};
  }
  Quat operator+(const Quat &p) const {
    return {w + p.w, x + p.x, y + p.y, z + p.z};
  }
  Quat operator-(const Quat &p) const {
    return {w - p.w, x - p.x, y - p.y, z - p.z};
  }
  Quat operator*(float s) const { return {w * s, x * s, y * s, z * s}; }
  float length() const { return sqrtf(w * w + x * x + y * y + z * z); }
  Quat normalized() const {
    float l = length();
    return {w / l, x / l, y / l, z / l};
  }
  Quat operator-() const { return {-w, -x, -y, -z}; }
  Vec3 rotate(const Vec3 &v) const {
    // q * (0,v) * q^-1
    Quat qv(0, v.x, v.y, v.z);
    Quat r = *this * qv * conjugate();
    return {r.x, r.y, r.z};
  }
};

struct Mat33 {
  float m[3][3]; // m[row][col]
  Mat33() {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        m[i][j] = 0;
  }
  static Mat33 diag(float a, float b, float c) {
    Mat33 r;
    r.m[0][0] = a;
    r.m[1][1] = b;
    r.m[2][2] = c;
    return r;
  }
  Vec3 operator*(const Vec3 &v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
  }
  Mat33 inverse() const;
  Mat33 transpose() const {
    Mat33 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r.m[i][j] = m[j][i];
    return r;
  }
  Mat33 mul(const Mat33 &b) const { // matrix × matrix
    Mat33 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r.m[i][j] =
            m[i][0] * b.m[0][j] + m[i][1] * b.m[1][j] + m[i][2] * b.m[2][j];
    return r;
  }
  Mat33 operator-(const Mat33 &b) const {
    Mat33 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r.m[i][j] = m[i][j] - b.m[i][j];
    return r;
  }
  Mat33 operator+(const Mat33 &b) const {
    Mat33 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r.m[i][j] = m[i][j] + b.m[i][j];
    return r;
  }
  Mat33 operator*(float s) const {
    Mat33 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r.m[i][j] = m[i][j] * s;
    return r;
  }
};

inline Mat33 Mat33::inverse() const {
  Mat33 inv;
  float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (fabsf(det) < 1e-20f)
    return Mat33::diag(0, 0, 0);
  float invDet = 1.f / det;
  inv.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
  inv.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
  inv.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
  inv.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
  inv.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
  inv.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
  inv.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
  inv.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
  inv.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;
  return inv;
}

// ====================== 6x6 types ==========================================

struct Vec6 {
  float v[6];
  Vec6() {
    for (int i = 0; i < 6; i++)
      v[i] = 0;
  }
  Vec6(const Vec3 &lin, const Vec3 &ang) {
    v[0] = lin.x;
    v[1] = lin.y;
    v[2] = lin.z;
    v[3] = ang.x;
    v[4] = ang.y;
    v[5] = ang.z;
  }
  float &operator[](int i) { return v[i]; }
  float operator[](int i) const { return v[i]; }
  Vec6 operator+(const Vec6 &b) const {
    Vec6 r;
    for (int i = 0; i < 6; i++)
      r.v[i] = v[i] + b.v[i];
    return r;
  }
  Vec6 &operator+=(const Vec6 &b) {
    for (int i = 0; i < 6; i++)
      v[i] += b.v[i];
    return *this;
  }
  Vec6 operator*(float s) const {
    Vec6 r;
    for (int i = 0; i < 6; i++)
      r.v[i] = v[i] * s;
    return r;
  }
  Vec3 linear() const { return {v[0], v[1], v[2]}; }
  Vec3 angular() const { return {v[3], v[4], v[5]}; }
};
inline float dot(const Vec6 &a, const Vec6 &b) {
  float s = 0;
  for (int i = 0; i < 6; i++)
    s += a[i] * b[i];
  return s;
}

struct Mat66 {
  float m[6][6]; // m[row][col]
  Mat66() {
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++)
        m[i][j] = 0;
  }
  Mat66 operator/(float s) const {
    Mat66 r;
    float inv = 1.f / s;
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++)
        r.m[i][j] = m[i][j] * inv;
    return r;
  }
  Vec6 operator*(const Vec6 &v) const {
    Vec6 r;
    for (int i = 0; i < 6; i++) {
      float s = 0;
      for (int j = 0; j < 6; j++)
        s += m[i][j] * v[j];
      r[i] = s;
    }
    return r;
  }
  Mat66 &operator+=(const Mat66 &b) {
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++)
        m[i][j] += b.m[i][j];
    return *this;
  }
};

// outer(a, b) = a * b^T (6x6 rank-1)
inline Mat66 outer(const Vec6 &a, const Vec6 &b) {
  Mat66 r;
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
      r.m[i][j] = a[i] * b[j];
  return r;
}

// ====================== LDLT solver (ref: ldlt.cpp) =========================

inline Vec6 solveLDLT(const Mat66 &lhs, const Vec6 &rhs) {
  // LDL^T decomposition
  float L[6][6] = {};
  float D[6] = {};

  for (int i = 0; i < 6; i++) {
    float sum = 0;
    for (int j = 0; j < i; j++)
      sum += L[i][j] * L[i][j] * D[j];
    D[i] = lhs.m[i][i] - sum;
    if (fabsf(D[i]) < 1e-12f)
      D[i] = 1e-12f; // regularize

    L[i][i] = 1.0f;
    for (int j = i + 1; j < 6; j++) {
      float s = lhs.m[j][i];
      for (int k = 0; k < i; k++)
        s -= L[j][k] * L[i][k] * D[k];
      L[j][i] = s / D[i];
    }
  }

  // Forward: Ly = b
  Vec6 y;
  for (int i = 0; i < 6; i++) {
    float s = 0;
    for (int j = 0; j < i; j++)
      s += L[i][j] * y[j];
    y[i] = rhs[i] - s;
  }

  // Diagonal: Dz = y
  Vec6 z;
  for (int i = 0; i < 6; i++)
    z[i] = y[i] / D[i];

  // Backward: L^T x = z
  Vec6 x;
  for (int i = 5; i >= 0; i--) {
    float s = 0;
    for (int j = i + 1; j < 6; j++)
      s += L[j][i] * x[j];
    x[i] = z[i] - s;
  }
  return x;
}

} // namespace AvbdRef
