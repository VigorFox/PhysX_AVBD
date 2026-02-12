// =============================================================================
// AVBD Collision Detection -- Standalone, zero-dependency implementation
//
// Box vs infinite ground plane (y=0)
// Box vs Box (SAT, Separating Axis Theorem)
//
// All contacts are generated in the format expected by AvbdRef::Solver:
//   normal: from B to A (pushes A away from B)
//   rA: contact in bodyA local frame
//   rB: contact in bodyB local frame (or world pos for ground)
//   depth > 0 means penetrating
// =============================================================================
#pragma once
#include "avbd_ref_solver.h"
#include <cmath>
#include <algorithm>

namespace AvbdRef {

// ====================== Box vs Ground Plane =================================
// Ground plane: y = 0, normal = (0, 1, 0)
// Tests all 8 corners, generates contacts for those below or near ground.
// margin: contacts generated when corner.y < margin (proximity contacts)
// =============================================================================
inline int collideBoxGround(Solver& solver, uint32_t boxIdx, float margin = 0.02f) {
  const Body& box = solver.bodies[boxIdx];
  Vec3 he = box.halfExtent;

  // 8 corners in local frame
  Vec3 localCorners[8];
  int ci = 0;
  for (int sx = -1; sx <= 1; sx += 2)
    for (int sy = -1; sy <= 1; sy += 2)
      for (int sz = -1; sz <= 1; sz += 2)
        localCorners[ci++] = {sx * he.x, sy * he.y, sz * he.z};

  Vec3 normal(0, 1, 0);
  int count = 0;

  for (int i = 0; i < 8; i++) {
    Vec3 wc = box.position + box.rotation.rotate(localCorners[i]);
    float dist = wc.y; // signed distance to ground (positive = above)
    if (dist < margin) {
      float depth = -dist; // positive when penetrating
      Vec3 groundPt(wc.x, 0, wc.z);
      solver.addContact(boxIdx, UINT32_MAX, normal, localCorners[i], groundPt,
                        depth, box.friction);
      count++;
    }
  }
  return count;
}

// ====================== Helpers for SAT Box-Box =============================

// Build 3 orthonormal axes from quaternion
inline void quatToAxes(const Quat& q, Vec3 axes[3]) {
  float qw = q.w, qx = q.x, qy = q.y, qz = q.z;
  axes[0] = {1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qz*qw), 2*(qx*qz - qy*qw)};
  axes[1] = {2*(qx*qy - qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qx*qw)};
  axes[2] = {2*(qx*qz + qy*qw), 2*(qy*qz - qx*qw), 1 - 2*(qx*qx + qy*qy)};
}

// Get face vertices most aligned with `localDir` in box local space
inline void getBoxFace(const Vec3& localDir, const Vec3& he, Vec3 face[4]) {
  float ax = fabsf(localDir.x), ay = fabsf(localDir.y), az = fabsf(localDir.z);
  if (ax >= ay && ax >= az) {
    float s = localDir.x > 0 ? he.x : -he.x;
    face[0] = {s, -he.y, -he.z}; face[1] = {s, he.y, -he.z};
    face[2] = {s, he.y, he.z};   face[3] = {s, -he.y, he.z};
  } else if (ay >= az) {
    float s = localDir.y > 0 ? he.y : -he.y;
    face[0] = {-he.x, s, -he.z}; face[1] = {he.x, s, -he.z};
    face[2] = {he.x, s, he.z};   face[3] = {-he.x, s, he.z};
  } else {
    float s = localDir.z > 0 ? he.z : -he.z;
    face[0] = {-he.x, -he.y, s}; face[1] = {he.x, -he.y, s};
    face[2] = {he.x, he.y, s};   face[3] = {-he.x, he.y, s};
  }
}

// Sutherland-Hodgman polygon clipping against half-space: dot(v, planeN) <= planeDist
inline int clipPolygon(const Vec3* poly, int count, const Vec3& planeN, float planeDist,
                       Vec3* out, int maxOut = 15) {
  if (count == 0) return 0;
  int outCount = 0;
  for (int i = 0; i < count && outCount < maxOut; i++) {
    int j = (i + 1) % count;
    float di = poly[i].dot(planeN) - planeDist;
    float dj = poly[j].dot(planeN) - planeDist;
    if (di <= 0) {
      out[outCount++] = poly[i];
      if (dj > 0 && outCount < maxOut) {
        float t = di / (di - dj);
        out[outCount++] = poly[i] + (poly[j] - poly[i]) * t;
      }
    } else if (dj <= 0 && outCount < maxOut) {
      float t = di / (di - dj);
      out[outCount++] = poly[i] + (poly[j] - poly[i]) * t;
    }
  }
  return outCount;
}

// ====================== Box vs Box (SAT) ====================================
//
// Returns number of contacts generated (0 = no collision).
// Contact normal points from B to A (pushes A away from B).
//
// margin adds a shell around each box for proximity detection.
// For perfectly-touching boxes (gap=0), the depth will be ~0.
// =============================================================================
inline int collideBoxBox(Solver& solver, uint32_t idxA, uint32_t idxB, float margin = 0.02f) {
  const Body& bA = solver.bodies[idxA];
  const Body& bB = solver.bodies[idxB];

  Vec3 axA[3], axB[3];
  quatToAxes(bA.rotation, axA);
  quatToAxes(bB.rotation, axB);

  Vec3 heA = bA.halfExtent;
  Vec3 heB = bB.halfExtent;
  Vec3 d = bB.position - bA.position; // B center - A center

  float heAf[3] = {heA.x, heA.y, heA.z};
  float heBf[3] = {heB.x, heB.y, heB.z};

  // Precompute abs dot products with epsilon for parallel edge cases
  float absAxDot[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      absAxDot[i][j] = fabsf(axA[i].dot(axB[j])) + 1e-6f;

  float minPen = 1e30f;
  Vec3 bestAxis;

  // Test a candidate separating axis. Returns false if it separates.
  // The axis does NOT need to be normalized here -- we normalize first.
  auto testAxis = [&](Vec3 axis) -> bool {
    float len2 = axis.length2();
    if (len2 < 1e-10f) return true; // degenerate axis, skip
    float invLen = 1.0f / sqrtf(len2);
    Vec3 n = axis * invLen;

    // Half-projections of each box onto axis
    float rA = fabsf(axA[0].dot(n)) * heAf[0] +
               fabsf(axA[1].dot(n)) * heAf[1] +
               fabsf(axA[2].dot(n)) * heAf[2];
    float rB = fabsf(axB[0].dot(n)) * heBf[0] +
               fabsf(axB[1].dot(n)) * heBf[1] +
               fabsf(axB[2].dot(n)) * heBf[2];

    float dist = n.dot(d); // signed distance from A center to B center along n
    float pen = rA + rB - fabsf(dist) + margin;
    if (pen < 0) return false; // separating axis!

    if (pen < minPen) {
      minPen = pen;
      // Normal from B to A: if dist > 0 (B is on the + side), we want normal
      // pointing from B toward A, i.e., in the - direction
      bestAxis = (dist > 0) ? n * (-1.0f) : n;
    }
    return true;
  };

  // --- 3 face axes of A ---
  for (int i = 0; i < 3; i++)
    if (!testAxis(axA[i])) return 0;

  // --- 3 face axes of B ---
  for (int j = 0; j < 3; j++)
    if (!testAxis(axB[j])) return 0;

  // --- 9 edge-edge cross products ---
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (!testAxis(axA[i].cross(axB[j]))) return 0;

  // =========================================================================
  // No separating axis -> collision!
  // bestAxis = contact normal, pointing from B to A.
  // =========================================================================
  Vec3 normal = bestAxis;

  // =========================================================================
  // Generate contacts via reference/incident face clipping
  // =========================================================================

  // Choose reference face: the body whose face axis is most aligned with
  // the contact normal. The reference face "receives" the incident face.
  //
  // For body A: test how much `normal` aligns with each face axis.
  //   normal points B->A, so the most aligned A face is the one facing B.
  // For body B: test alignment with `-normal` (the direction facing A).

  float bestDotA = 0; int bestFaceIdxA = 0;
  for (int i = 0; i < 3; i++) {
    float dd = fabsf(normal.dot(axA[i]));
    if (dd > bestDotA) { bestDotA = dd; bestFaceIdxA = i; }
  }
  float bestDotB = 0; int bestFaceIdxB = 0;
  for (int i = 0; i < 3; i++) {
    float dd = fabsf(normal.dot(axB[i]));
    if (dd > bestDotB) { bestDotB = dd; bestFaceIdxB = i; }
  }

  const Body* refBody;
  const Body* incBody;
  Vec3 refAx[3], incAx[3];
  Vec3 refHe, incHe;
  Vec3 refFaceN; // outward normal of the reference face, in world space

  if (bestDotA >= bestDotB) {
    // A is reference face body
    refBody = &bA; incBody = &bB;
    for (int i = 0; i < 3; i++) { refAx[i] = axA[i]; incAx[i] = axB[i]; }
    refHe = heA; incHe = heB;
    // Reference face normal: the axA direction facing toward B
    refFaceN = axA[bestFaceIdxA];
    if (refFaceN.dot(d) < 0) refFaceN = -refFaceN; // flip to face toward B
  } else {
    // B is reference face body
    refBody = &bB; incBody = &bA;
    for (int i = 0; i < 3; i++) { refAx[i] = axB[i]; incAx[i] = axA[i]; }
    refHe = heB; incHe = heA;
    refFaceN = axB[bestFaceIdxB];
    if (refFaceN.dot(d) > 0) refFaceN = -refFaceN; // flip to face toward A
  }

  // --- Reference face vertices (world space) ---
  Vec3 refLocalDir; // reference face normal in ref body's local frame
  refLocalDir.x = refFaceN.dot(refAx[0]);
  refLocalDir.y = refFaceN.dot(refAx[1]);
  refLocalDir.z = refFaceN.dot(refAx[2]);

  Vec3 refFace[4];
  getBoxFace(refLocalDir, refHe, refFace);
  Vec3 refFaceW[4];
  for (int i = 0; i < 4; i++)
    refFaceW[i] = refBody->position + refBody->rotation.rotate(refFace[i]);

  // --- Incident face vertices (world space) ---
  // The incident face is the one on incBody most facing the reference face
  Vec3 incLocalDir; // negative of refFaceN in incident body's local frame
  incLocalDir.x = -refFaceN.dot(incAx[0]);
  incLocalDir.y = -refFaceN.dot(incAx[1]);
  incLocalDir.z = -refFaceN.dot(incAx[2]);

  Vec3 incFace[4];
  getBoxFace(incLocalDir, incHe, incFace);

  Vec3 poly[16];
  for (int i = 0; i < 4; i++)
    poly[i] = incBody->position + incBody->rotation.rotate(incFace[i]);
  int polyCount = 4;

  // --- Clip incident polygon against 4 side planes of reference face ---
  for (int e = 0; e < 4; e++) {
    Vec3 edgeStart = refFaceW[e];
    Vec3 edgeEnd = refFaceW[(e + 1) % 4];
    Vec3 edgeDir = (edgeEnd - edgeStart).normalized();
    // Side plane outward normal: refFaceN cross edge direction
    // (refFaceN × edgeDir points outward when vertices are ordered correctly)
    Vec3 sideN = refFaceN.cross(edgeDir);
    float sideDist = sideN.dot(edgeStart);

    Vec3 tmp[16];
    polyCount = clipPolygon(poly, polyCount, sideN, sideDist, tmp);
    for (int i = 0; i < polyCount; i++) poly[i] = tmp[i];
    if (polyCount == 0) break;
  }

  // --- Output contacts ---
  // Reference plane: defined by refFaceN and any reference face vertex
  float refPlaneDist = refFaceN.dot(refFaceW[0]);

  int contactCount = 0;
  float fric = sqrtf(bA.friction * bB.friction);

  // Determine which body is the "incident" body (its face is clipped).
  // In our solver convention, bodyA is the one being pushed along -normal
  // by the constraint force (f <= 0). The incident body is the one whose
  // face was clipped against the reference face -- it's the one that needs
  // to be pushed out. So incident = bodyA.
  //
  // If A is reference and B is incident: swap so contact = (B, A, -normal)
  // If B is reference and A is incident: contact = (A, B, normal) as-is
  uint32_t contactBodyA, contactBodyB;
  Vec3 contactNormal;
  if (bestDotA >= bestDotB) {
    // A was reference, B was incident -> incident=B should be bodyA
    contactBodyA = idxB;
    contactBodyB = idxA;
    contactNormal = -normal; // flip: was from B->A, now from A->B (new B is old A)
  } else {
    // B was reference, A was incident -> incident=A is already bodyA
    contactBodyA = idxA;
    contactBodyB = idxB;
    contactNormal = normal;
  }

  for (int i = 0; i < polyCount; i++) {
    // How far the incident polygon point penetrates past the reference plane
    // positive separation = point is on the outward side (not penetrating)
    float separation = poly[i].dot(refFaceN) - refPlaneDist;
    float depth = -separation; // positive = penetrating

    if (depth < -margin) continue; // too far apart

    // Contact point: project the clipped incident point onto reference plane
    Vec3 contactPt = poly[i] - refFaceN * separation;

    // Convert to local frames of the CONTACT bodies (not ref/inc)
    Vec3 rA_local = solver.bodies[contactBodyA].rotation.conjugate().rotate(
      contactPt - solver.bodies[contactBodyA].position);
    Vec3 rB_local = solver.bodies[contactBodyB].rotation.conjugate().rotate(
      contactPt - solver.bodies[contactBodyB].position);

    solver.addContact(contactBodyA, contactBodyB, contactNormal,
                      rA_local, rB_local, depth, fric);
    contactCount++;
  }

  return contactCount;
}

// =============================================================================
// Convenience: detect all collisions for all body pairs + ground
// =============================================================================
inline int collideAll(Solver& solver, float margin = 0.02f) {
  int totalContacts = 0;
  uint32_t n = (uint32_t)solver.bodies.size();

  // Each dynamic body vs ground
  for (uint32_t i = 0; i < n; i++) {
    if (solver.bodies[i].mass <= 0) continue;
    totalContacts += collideBoxGround(solver, i, margin);
  }

  // Each body pair (at least one dynamic)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = i + 1; j < n; j++) {
      if (solver.bodies[i].mass <= 0 && solver.bodies[j].mass <= 0) continue;

      // Quick sphere broadphase
      Vec3 pA = solver.bodies[i].position, pB = solver.bodies[j].position;
      Vec3 hA = solver.bodies[i].halfExtent, hB = solver.bodies[j].halfExtent;
      float rA = std::max({hA.x, hA.y, hA.z}) * 1.74f; // sqrt(3)
      float rB = std::max({hB.x, hB.y, hB.z}) * 1.74f;
      Vec3 diff = pB - pA;
      if (diff.length() > rA + rB + margin) continue;

      totalContacts += collideBoxBox(solver, i, j, margin);
    }
  }

  return totalContacts;
}

} // namespace AvbdRef
