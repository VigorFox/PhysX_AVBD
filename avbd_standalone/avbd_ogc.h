#pragma once
// =============================================================================
// OGC (Offset Geometric Contact) — 4-Path Collision Detection
//
// Reference: "Offset Geometric Contact", SIGGRAPH 2025
//            Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel
//
// Path 1: Rigid-Rigid → Handled by existing SAT/GJK (avbd_collision.h)
// Path 2: Rigid-Soft → Analytical box SDF query
// Path 3: Soft-Soft → OGC simplified (§3.9: surface-only outward offset, pure quadratic)
// Path 4: Soft self-collision → OGC full (safety bubble + two-stage activation)
//
// All paths output SoftContact structs for unified solver integration.
// =============================================================================

#include "avbd_softbody.h"
#include "avbd_types.h"
#include "avbd_math.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace AvbdRef {

// =============================================================================
// OGC Parameters
// =============================================================================
struct OGCParams {
  float contactRadius = 0.05f;     // r: offset radius
  float contactStiffness = 1e5f;   // k_c: contact stiffness
  float friction = 0.3f;           // mu_c
  float safetyRelax = 0.45f;       // gamma_p: safety bound relaxation (0 < gamma_p < 0.5)
  float redetectRatio = 0.01f;     // gamma_e: redetection trigger ratio
  float tau = -1.0f;               // activation threshold; -1 means r/2 (auto)

  float getTau() const { return (tau < 0.0f) ? contactRadius * 0.5f : tau; }
};

// =============================================================================
// Surface triangle — extracted from tet mesh boundary
// =============================================================================
struct SurfaceTriangle {
  uint32_t v0, v1, v2;  // global particle indices
  uint32_t softBodyIdx;  // which soft body this belongs to
};

// =============================================================================
// AABB for broadphase
// =============================================================================
struct AABB {
  Vec3 lo, hi;

  AABB() : lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f} {}

  void expand(Vec3 p) {
    lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
  }

  void inflate(float r) {
    lo.x -= r; lo.y -= r; lo.z -= r;
    hi.x += r; hi.y += r; hi.z += r;
  }

  bool overlaps(const AABB& o) const {
    return lo.x <= o.hi.x && hi.x >= o.lo.x &&
           lo.y <= o.hi.y && hi.y >= o.lo.y &&
           lo.z <= o.hi.z && hi.z >= o.lo.z;
  }
};

// =============================================================================
// Extract boundary (surface) triangles from a tetrahedral mesh
//
// A face is on the boundary if it appears in exactly one tetrahedron.
// Returns surface triangles with outward-facing normals.
// =============================================================================
inline void extractSurfaceTriangles(
    const SoftBody& sb, uint32_t sbIdx,
    const std::vector<SoftParticle>& particles,
    std::vector<SurfaceTriangle>& outSurface)
{
  // Each tet has 4 faces. Encode each face as sorted (a,b,c).
  struct FaceKey {
    uint32_t a, b, c;
    bool operator<(const FaceKey& o) const {
      if (a != o.a) return a < o.a;
      if (b != o.b) return b < o.b;
      return c < o.c;
    }
    bool operator==(const FaceKey& o) const {
      return a == o.a && b == o.b && c == o.c;
    }
  };
  struct FaceInfo {
    FaceKey key;
    uint32_t v0, v1, v2;  // original winding (global indices)
    uint32_t oppositeVert; // vertex opposite to this face in the tet
    int count;
  };

  std::vector<FaceInfo> faces;
  faces.reserve(sb.tetrahedra.size()); // 4 faces per tet, but size/4*4 = size

  for (size_t i = 0; i + 3 < sb.tetrahedra.size(); i += 4) {
    uint32_t g[4];
    for (int j = 0; j < 4; j++)
      g[j] = sb.particleStart + sb.tetrahedra[i + j];

    // 4 faces of a tet: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    uint32_t faceIndices[4][3] = {
      {g[0], g[2], g[1]}, // opposite to g[3]
      {g[0], g[1], g[3]}, // opposite to g[2]
      {g[0], g[3], g[2]}, // opposite to g[1]
      {g[1], g[2], g[3]}, // opposite to g[0]
    };
    uint32_t oppVerts[4] = {g[3], g[2], g[1], g[0]};

    for (int f = 0; f < 4; f++) {
      uint32_t a = faceIndices[f][0], b = faceIndices[f][1], c = faceIndices[f][2];
      FaceKey key;
      key.a = std::min({a, b, c});
      key.c = std::max({a, b, c});
      key.b = a + b + c - key.a - key.c;
      faces.push_back({key, a, b, c, oppVerts[f], 1});
    }
  }

  // Sort by key then merge duplicates
  std::sort(faces.begin(), faces.end(), [](const FaceInfo& x, const FaceInfo& y) {
    return x.key < y.key;
  });

  for (size_t i = 0; i < faces.size(); ) {
    size_t j = i + 1;
    while (j < faces.size() && faces[j].key == faces[i].key)
      j++;

    if (j - i == 1) {
      // Boundary face: ensure outward normal
      // The outward normal should point away from the opposite vertex
      const FaceInfo& fi = faces[i];
      Vec3 p0 = particles[fi.v0].position;
      Vec3 p1 = particles[fi.v1].position;
      Vec3 p2 = particles[fi.v2].position;
      Vec3 po = particles[fi.oppositeVert].position;
      Vec3 n = (p1 - p0).cross(p2 - p0);
      Vec3 toOpp = po - p0;
      SurfaceTriangle st;
      st.softBodyIdx = sbIdx;
      if (n.dot(toOpp) > 0.0f) {
        // normal points inward, flip
        st.v0 = fi.v0; st.v1 = fi.v2; st.v2 = fi.v1;
      } else {
        st.v0 = fi.v0; st.v1 = fi.v1; st.v2 = fi.v2;
      }
      outSurface.push_back(st);
    }
    i = j;
  }
}

// =============================================================================
// Point-to-triangle distance and closest point
// =============================================================================
enum ClosestFeature { FEATURE_FACE, FEATURE_EDGE, FEATURE_VERTEX };

struct ClosestPointResult {
  Vec3 point;            // closest point on triangle
  Vec3 normal;           // direction from closest point to query point
  float distance;        // unsigned distance
  ClosestFeature feature;
};

inline ClosestPointResult closestPointOnTriangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
  ClosestPointResult result;

  Vec3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0.0f && d2 <= 0.0f) {
    result.point = a;
    result.feature = FEATURE_VERTEX;
    Vec3 diff = p - a;
    result.distance = diff.length();
    result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : Vec3(0, 1, 0);
    return result;
  }

  Vec3 bp = p - b;
  float d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0.0f && d4 <= d3) {
    result.point = b;
    result.feature = FEATURE_VERTEX;
    Vec3 diff = p - b;
    result.distance = diff.length();
    result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : Vec3(0, 1, 0);
    return result;
  }

  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v = d1 / (d1 - d3);
    result.point = a + ab * v;
    result.feature = FEATURE_EDGE;
    Vec3 diff = p - result.point;
    result.distance = diff.length();
    result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : Vec3(0, 1, 0);
    return result;
  }

  Vec3 cp = p - c;
  float d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0.0f && d5 <= d6) {
    result.point = c;
    result.feature = FEATURE_VERTEX;
    Vec3 diff = p - c;
    result.distance = diff.length();
    result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : Vec3(0, 1, 0);
    return result;
  }

  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w = d2 / (d2 - d6);
    result.point = a + ac * w;
    result.feature = FEATURE_EDGE;
    Vec3 diff = p - result.point;
    result.distance = diff.length();
    result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : Vec3(0, 1, 0);
    return result;
  }

  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    result.point = b + (c - b) * w;
    result.feature = FEATURE_EDGE;
    Vec3 diff = p - result.point;
    result.distance = diff.length();
    result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : Vec3(0, 1, 0);
    return result;
  }

  // Inside triangle
  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom;
  float w = vc * denom;
  result.point = a + ab * v + ac * w;
  result.feature = FEATURE_FACE;
  Vec3 diff = p - result.point;
  result.distance = diff.length();

  if (result.distance > 1e-10f) {
    result.normal = diff * (1.0f / result.distance);
  } else {
    // Point on face — use face normal
    Vec3 faceN = ab.cross(ac);
    float fLen = faceN.length();
    result.normal = fLen > 1e-10f ? faceN * (1.0f / fLen) : Vec3(0, 1, 0);
  }
  return result;
}

// =============================================================================
// OGC Block membership test
//
// Given a query point x and a surface triangle, determine if x lies in the
// face block U_t. For vertex/edge blocks, we just check the complement:
// if x is within r of the triangle but NOT in U_t, it's in a vertex or edge block.
//
// For the simplified path (§3.9, volumetric), we use the closest-point feature
// type to determine the block: FEATURE_FACE → face block, FEATURE_EDGE → edge
// block, FEATURE_VERTEX → vertex block. OGC orthogonality is guaranteed by
// construction: the contact normal is always (x - c(x,a)) / ||x - c(x,a)||.
// =============================================================================
inline bool isInFaceBlock(Vec3 x, Vec3 a, Vec3 b, Vec3 c, float r) {
  // Check if x is within distance r of the triangle face interior
  // The face block is the slab region above/below the triangle within distance r
  ClosestPointResult cp = closestPointOnTriangle(x, a, b, c);
  return cp.feature == FEATURE_FACE && cp.distance <= r;
}

// =============================================================================
// Two-stage C² activation function (Eq. 18-20)
//
// g(d, r) = { k_c/2 * (r-d)² ,  if tau <= d <= r     (quadratic)
//           { -k'_c * log(d) + b, if 0 < d < tau       (log barrier)
// =============================================================================
struct ActivationResult {
  float energy;
  float force;     // -dg/dd (positive = repulsive)
  float hessian;   // d²g/dd²
};

inline ActivationResult ogcActivationQuadratic(float d, float r, float kc) {
  // Pure quadratic: g = k_c/2 * (r-d)²
  // force = k_c * (r - d)  (positive when d < r)
  // hessian = k_c
  float pen = r - d;
  ActivationResult res;
  res.energy = 0.5f * kc * pen * pen;
  res.force = kc * pen;
  res.hessian = kc;
  return res;
}

inline ActivationResult ogcActivationFull(float d, float r, float kc, float tau) {
  ActivationResult res;
  if (d >= r) {
    res.energy = 0.0f;
    res.force = 0.0f;
    res.hessian = 0.0f;
  } else if (d >= tau) {
    // Quadratic stage
    float pen = r - d;
    res.energy = 0.5f * kc * pen * pen;
    res.force = kc * pen;
    res.hessian = kc;
  } else if (d > 1e-10f) {
    // Log barrier stage
    float rmt = r - tau;
    float kc_prime = tau * kc * rmt * rmt;
    float b = 0.5f * kc * rmt * rmt + kc_prime * logf(tau);
    res.energy = -kc_prime * logf(d) + b;
    res.force = kc_prime / d;     // -dg/dd = k'_c / d
    res.hessian = kc_prime / (d * d);
  } else {
    // Clamp at very small d to avoid singularity
    float rmt = r - tau;
    float kc_prime = tau * kc * rmt * rmt;
    float d_clamp = 1e-10f;
    res.energy = kc_prime * 10.0f; // large but finite
    res.force = kc_prime / d_clamp;
    res.hessian = kc_prime / (d_clamp * d_clamp);
  }
  return res;
}

// =============================================================================
// PATH 2: SDF Rigid-Soft Contact Detection
//
// Compute analytical signed distance from soft particle to rigid box.
// The box SDF is exact: d = max(|x_local| - halfExtent) with proper sign.
// Outputs SoftContacts with orthogonal normals.
// =============================================================================
inline void detectSoftRigidSDF(
    const std::vector<SoftParticle>& particles,
    const std::vector<Body>& rigidBodies,
    const std::vector<SoftBody>& softBodies,
    std::vector<SoftContact>& contacts,
    float margin = 0.05f)
{
  for (uint32_t bi = 0; bi < (uint32_t)rigidBodies.size(); bi++) {
    const Body& rb = rigidBodies[bi];
    if (rb.halfExtent.x <= 0 && rb.halfExtent.y <= 0 && rb.halfExtent.z <= 0)
      continue;

    Vec3 he = rb.halfExtent;

    // Build AABB for broadphase
    // Box world AABB = center ± extent_world
    float maxExt = sqrtf(he.x * he.x + he.y * he.y + he.z * he.z) + margin;
    AABB boxAABB;
    boxAABB.lo = rb.position - Vec3(maxExt, maxExt, maxExt);
    boxAABB.hi = rb.position + Vec3(maxExt, maxExt, maxExt);

    for (uint32_t pi = 0; pi < (uint32_t)particles.size(); pi++) {
      if (particles[pi].invMass <= 0.0f) continue;
      const Vec3& pp = particles[pi].position;

      // Quick AABB check
      if (pp.x < boxAABB.lo.x || pp.x > boxAABB.hi.x ||
          pp.y < boxAABB.lo.y || pp.y > boxAABB.hi.y ||
          pp.z < boxAABB.lo.z || pp.z > boxAABB.hi.z)
        continue;

      // Transform to box local space
      Vec3 localP = rb.rotation.conjugate().rotate(pp - rb.position);

      // Analytical box SDF: signed distance
      // Outside: max of (|x|-he) components > 0
      // Inside:  min of (he-|x|) components (negative)
      Vec3 q = Vec3(fabsf(localP.x) - he.x,
                     fabsf(localP.y) - he.y,
                     fabsf(localP.z) - he.z);

      bool inside = (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f);
      float sdf;
      Vec3 localNormal;

      if (inside) {
        // Inside: SDF = max(q.x, q.y, q.z) (negative)
        sdf = std::max(q.x, std::max(q.y, q.z));
        // Gradient points along the axis of minimum penetration
        if (q.x > q.y && q.x > q.z)
          localNormal = Vec3(localP.x > 0 ? 1.0f : -1.0f, 0, 0);
        else if (q.y > q.z)
          localNormal = Vec3(0, localP.y > 0 ? 1.0f : -1.0f, 0);
        else
          localNormal = Vec3(0, 0, localP.z > 0 ? 1.0f : -1.0f);
      } else {
        // Outside: SDF = length(max(q, 0))
        Vec3 clamped = Vec3(std::max(q.x, 0.0f), std::max(q.y, 0.0f), std::max(q.z, 0.0f));
        sdf = clamped.length();
        if (sdf > 1e-10f)
          localNormal = clamped * (1.0f / sdf);
        else
          localNormal = Vec3(0, 1, 0);
      }

      if (sdf < margin) {
        float depth = inside ? -sdf : std::max(0.0f, margin - sdf);
        Vec3 worldNormal = rb.rotation.rotate(localNormal).normalized();

        Vec3 t1, t2;
        if (fabsf(worldNormal.x) < 0.9f)
          t1 = worldNormal.cross(Vec3(1, 0, 0)).normalized();
        else
          t1 = worldNormal.cross(Vec3(0, 1, 0)).normalized();
        t2 = worldNormal.cross(t1);

        SoftContact sc;
        sc.particleIdx = pi;
        sc.rigidBodyIdx = bi;
        sc.normal = worldNormal;
        sc.depth = depth;
        sc.friction = sqrtf(particles[pi].damping * rb.friction);
        sc.tangent1 = t1;
        sc.tangent2 = t2;
        contacts.push_back(sc);
      }
    }
  }
}

// =============================================================================
// PATH 3: OGC Simplified Soft-Soft Contact (§3.9)
//
// For volumetric meshes:
//   - Surface-only outward offset
//   - Pure quadratic energy (only first stage of activation)
//   - Skip safety bubble (conservative bound truncation)
//   - DCD: check if particle is inside another body's volume → penetration
//
// The contact normal is always orthogonal to the closest surface feature
// by OGC construction. Force = k_c * (r - d) * n, Hessian = k_c * (n ⊗ n).
// =============================================================================

// Helper: check if a point is inside a tet mesh using ray casting (parity)
// Cast ray along +Y and count intersections with surface triangles
inline bool isPointInsideTetMesh(
    Vec3 point,
    const std::vector<SurfaceTriangle>& surface,
    const std::vector<SoftParticle>& particles)
{
  int crossings = 0;
  for (const auto& tri : surface) {
    Vec3 a = particles[tri.v0].position;
    Vec3 b = particles[tri.v1].position;
    Vec3 c = particles[tri.v2].position;

    // Ray: origin = point, direction = (0, 1, 0)
    Vec3 e1 = b - a, e2 = c - a;
    // Möller–Trumbore intersection
    Vec3 h = Vec3(0, 1, 0).cross(e2);
    float det = e1.dot(h);
    if (fabsf(det) < 1e-10f) continue;
    float invDet = 1.0f / det;
    Vec3 s = point - a;
    float u = invDet * s.dot(h);
    if (u < 0.0f || u > 1.0f) continue;
    Vec3 q = s.cross(e1);
    float v = invDet * Vec3(0, 1, 0).dot(q);
    if (v < 0.0f || u + v > 1.0f) continue;
    float t = invDet * e2.dot(q);
    if (t > 1e-6f) crossings++;
  }
  return (crossings & 1) != 0;
}

inline void detectSoftSoftOGC(
    const std::vector<SoftParticle>& particles,
    const std::vector<SoftBody>& softBodies,
    const std::vector<std::vector<SurfaceTriangle>>& surfaces,
    std::vector<SoftContact>& contacts,
    const OGCParams& params = OGCParams())
{
  float r = params.contactRadius;
  uint32_t nBodies = (uint32_t)softBodies.size();

  // For each pair of soft bodies
  for (uint32_t sbA = 0; sbA < nBodies; sbA++) {
    for (uint32_t sbB = sbA + 1; sbB < nBodies; sbB++) {
      const SoftBody& bodyA = softBodies[sbA];
      const SoftBody& bodyB = softBodies[sbB];
      const auto& surfA = surfaces[sbA];
      const auto& surfB = surfaces[sbB];

      // AABB broadphase between two soft bodies
      AABB aabbA, aabbB;
      for (uint32_t i = bodyA.particleStart; i < bodyA.particleStart + bodyA.particleCount; i++)
        aabbA.expand(particles[i].position);
      for (uint32_t i = bodyB.particleStart; i < bodyB.particleStart + bodyB.particleCount; i++)
        aabbB.expand(particles[i].position);
      aabbA.inflate(r);
      aabbB.inflate(r);
      if (!aabbA.overlaps(aabbB)) continue;

      // For each particle in A, check against B's surface
      for (uint32_t pi = bodyA.particleStart; pi < bodyA.particleStart + bodyA.particleCount; pi++) {
        if (particles[pi].invMass <= 0.0f) continue;
        Vec3 pp = particles[pi].position;

        // Quick AABB check (particle vs body B inflated AABB)
        if (pp.x < aabbB.lo.x || pp.x > aabbB.hi.x ||
            pp.y < aabbB.lo.y || pp.y > aabbB.hi.y ||
            pp.z < aabbB.lo.z || pp.z > aabbB.hi.z)
          continue;

        // Check if particle is inside body B (DCD, §3.9)
        bool insideB = isPointInsideTetMesh(pp, surfB, particles);
        if (insideB) {
          // Penetrated: find closest surface face and compute penetration depth
          float minDist = 1e30f;
          Vec3 bestNormal(0, 1, 0);
          for (const auto& tri : surfB) {
            Vec3 a = particles[tri.v0].position;
            Vec3 b = particles[tri.v1].position;
            Vec3 c = particles[tri.v2].position;
            ClosestPointResult cp = closestPointOnTriangle(pp, a, b, c);
            if (cp.distance < minDist) {
              minDist = cp.distance;
              // Normal points inward (from closest point to query point = into the body)
              // For penetrating case, we want to push OUT, so flip the normal
              Vec3 faceN = (b - a).cross(c - a);
              float fLen = faceN.length();
              if (fLen > 1e-10f)
                bestNormal = faceN * (1.0f / fLen); // outward surface normal
              else
                bestNormal = cp.normal;
            }
          }
          // Penetration depth = dist_to_surface + r (§3.9)
          float depth = minDist + r;

          SoftContact sc;
          sc.particleIdx = pi;
          sc.rigidBodyIdx = UINT32_MAX; // marker: soft-soft contact
          sc.normal = bestNormal;
          sc.depth = depth;
          sc.k = params.contactStiffness;
          sc.ke = params.contactStiffness * 10.0f;
          sc.friction = params.friction;
          Vec3 t1, t2;
          if (fabsf(bestNormal.x) < 0.9f)
            t1 = bestNormal.cross(Vec3(1, 0, 0)).normalized();
          else
            t1 = bestNormal.cross(Vec3(0, 1, 0)).normalized();
          t2 = bestNormal.cross(t1);
          sc.tangent1 = t1;
          sc.tangent2 = t2;
          contacts.push_back(sc);
          continue; // don't also check offset blocks
        }

        // Not inside: check OGC offset blocks on B's surface
        // Find all surface triangles within distance r (outward offset only)
        for (const auto& tri : surfB) {
          Vec3 a = particles[tri.v0].position;
          Vec3 b = particles[tri.v1].position;
          Vec3 c = particles[tri.v2].position;

          ClosestPointResult cp = closestPointOnTriangle(pp, a, b, c);
          if (cp.distance >= r) continue;

          // OGC orthogonality: check that the contact direction is outward
          Vec3 faceN = (b - a).cross(c - a);
          float fLen = faceN.length();
          if (fLen < 1e-10f) continue;
          faceN = faceN * (1.0f / fLen); // outward normal

          // §3.9: outward-only offset → only consider if point is on the outside
          Vec3 toPoint = pp - cp.point;
          if (toPoint.dot(faceN) < 0.0f) continue; // point on inner side

          // OGC contact normal: orthogonal to the feature
          // For face block: normal = face normal
          // For edge/vertex block: normal = (x - c(x,a)) / ||x - c(x,a)||
          Vec3 contactNormal;
          if (cp.feature == FEATURE_FACE) {
            contactNormal = faceN;
          } else {
            contactNormal = cp.normal; // already pointing from closest to query
          }

          float depth = r - cp.distance;

          SoftContact sc;
          sc.particleIdx = pi;
          sc.rigidBodyIdx = UINT32_MAX;
          sc.normal = contactNormal;
          sc.depth = depth;
          sc.k = params.contactStiffness;
          sc.ke = params.contactStiffness * 10.0f;
          sc.friction = params.friction;
          Vec3 t1, t2;
          if (fabsf(contactNormal.x) < 0.9f)
            t1 = contactNormal.cross(Vec3(1, 0, 0)).normalized();
          else
            t1 = contactNormal.cross(Vec3(0, 1, 0)).normalized();
          t2 = contactNormal.cross(t1);
          sc.tangent1 = t1;
          sc.tangent2 = t2;
          contacts.push_back(sc);
        }
      }

      // Symmetric: particles of B against A's surface
      for (uint32_t pi = bodyB.particleStart; pi < bodyB.particleStart + bodyB.particleCount; pi++) {
        if (particles[pi].invMass <= 0.0f) continue;
        Vec3 pp = particles[pi].position;

        if (pp.x < aabbA.lo.x || pp.x > aabbA.hi.x ||
            pp.y < aabbA.lo.y || pp.y > aabbA.hi.y ||
            pp.z < aabbA.lo.z || pp.z > aabbA.hi.z)
          continue;

        bool insideA = isPointInsideTetMesh(pp, surfA, particles);
        if (insideA) {
          float minDist = 1e30f;
          Vec3 bestNormal(0, 1, 0);
          for (const auto& tri : surfA) {
            Vec3 a = particles[tri.v0].position;
            Vec3 b = particles[tri.v1].position;
            Vec3 c = particles[tri.v2].position;
            ClosestPointResult cp = closestPointOnTriangle(pp, a, b, c);
            if (cp.distance < minDist) {
              minDist = cp.distance;
              Vec3 faceN = (b - a).cross(c - a);
              float fLen = faceN.length();
              if (fLen > 1e-10f)
                bestNormal = faceN * (1.0f / fLen);
              else
                bestNormal = cp.normal;
            }
          }
          float depth = minDist + r;

          SoftContact sc;
          sc.particleIdx = pi;
          sc.rigidBodyIdx = UINT32_MAX;
          sc.normal = bestNormal;
          sc.depth = depth;
          sc.k = params.contactStiffness;
          sc.ke = params.contactStiffness * 10.0f;
          sc.friction = params.friction;
          Vec3 t1, t2;
          if (fabsf(bestNormal.x) < 0.9f)
            t1 = bestNormal.cross(Vec3(1, 0, 0)).normalized();
          else
            t1 = bestNormal.cross(Vec3(0, 1, 0)).normalized();
          t2 = bestNormal.cross(t1);
          sc.tangent1 = t1;
          sc.tangent2 = t2;
          contacts.push_back(sc);
          continue;
        }

        for (const auto& tri : surfA) {
          Vec3 a = particles[tri.v0].position;
          Vec3 b = particles[tri.v1].position;
          Vec3 c = particles[tri.v2].position;

          ClosestPointResult cp = closestPointOnTriangle(pp, a, b, c);
          if (cp.distance >= r) continue;

          Vec3 faceN = (b - a).cross(c - a);
          float fLen = faceN.length();
          if (fLen < 1e-10f) continue;
          faceN = faceN * (1.0f / fLen);

          Vec3 toPoint = pp - cp.point;
          if (toPoint.dot(faceN) < 0.0f) continue;

          Vec3 contactNormal;
          if (cp.feature == FEATURE_FACE) {
            contactNormal = faceN;
          } else {
            contactNormal = cp.normal;
          }

          float depth = r - cp.distance;

          SoftContact sc;
          sc.particleIdx = pi;
          sc.rigidBodyIdx = UINT32_MAX;
          sc.normal = contactNormal;
          sc.depth = depth;
          sc.k = params.contactStiffness;
          sc.ke = params.contactStiffness * 10.0f;
          sc.friction = params.friction;
          Vec3 t1, t2;
          if (fabsf(contactNormal.x) < 0.9f)
            t1 = contactNormal.cross(Vec3(1, 0, 0)).normalized();
          else
            t1 = contactNormal.cross(Vec3(0, 1, 0)).normalized();
          t2 = contactNormal.cross(t1);
          sc.tangent1 = t1;
          sc.tangent2 = t2;
          contacts.push_back(sc);
        }
      }
    } // for sbB
  } // for sbA
}

// =============================================================================
// PATH 4: OGC Full Self-Collision Detection
//
// For self-collision within a single soft body:
//   - Safety bubble (conservative displacement bound per vertex)
//   - Two-stage C² activation (quadratic + log barrier)
//   - Displacement truncation after each solver iteration
//
// Self-collision contacts are generated between particles of the SAME body
// that are not topologically adjacent.
// =============================================================================

// Build adjacency set: for each particle, which particles are topologically
// connected (share a tet or triangle). These should NOT generate self-contacts.
inline void buildAdjacency(
    const SoftBody& sb,
    std::vector<std::vector<uint32_t>>& adj)
{
  adj.resize(sb.particleCount);
  for (auto& a : adj) a.clear();

  auto addAdj = [&](uint32_t local_a, uint32_t local_b) {
    adj[local_a].push_back(local_b);
    adj[local_b].push_back(local_a);
  };

  for (size_t i = 0; i + 3 < sb.tetrahedra.size(); i += 4) {
    uint32_t v[4];
    for (int j = 0; j < 4; j++) v[j] = sb.tetrahedra[i + j];
    for (int a = 0; a < 4; a++)
      for (int b = a + 1; b < 4; b++)
        addAdj(v[a], v[b]);
  }
  for (size_t i = 0; i + 2 < sb.triangles.size(); i += 3) {
    uint32_t v[3];
    for (int j = 0; j < 3; j++) v[j] = sb.triangles[i + j];
    for (int a = 0; a < 3; a++)
      for (int b = a + 1; b < 3; b++)
        addAdj(v[a], v[b]);
  }

  // Deduplicate
  for (auto& a : adj) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
  }
}

inline bool isAdjacent(uint32_t localA, uint32_t localB,
                       const std::vector<std::vector<uint32_t>>& adj) {
  if (localA >= adj.size()) return false;
  const auto& a = adj[localA];
  return std::binary_search(a.begin(), a.end(), localB);
}

// Compute per-vertex conservative displacement bound (Eq. 21)
// b_v = gamma_p * min(d_min,v)
// Simplified: only compute vertex-to-face minimum distance (skip edge-edge)
inline void computeSafetyBounds(
    const SoftBody& sb,
    const std::vector<SurfaceTriangle>& surface,
    const std::vector<SoftParticle>& particles,
    const std::vector<std::vector<uint32_t>>& adj,
    float gammaP,
    std::vector<float>& bounds)
{
  bounds.resize(sb.particleCount, 1e30f);

  for (uint32_t li = 0; li < sb.particleCount; li++) {
    uint32_t gi = sb.particleStart + li;
    Vec3 pi = particles[gi].position;
    float dMin = 1e30f;

    for (const auto& tri : surface) {
      // Skip faces that contain this vertex
      uint32_t lv0 = tri.v0 - sb.particleStart;
      uint32_t lv1 = tri.v1 - sb.particleStart;
      uint32_t lv2 = tri.v2 - sb.particleStart;
      if (lv0 == li || lv1 == li || lv2 == li) continue;
      // Skip topologically adjacent faces
      if (isAdjacent(li, lv0, adj) && isAdjacent(li, lv1, adj) && isAdjacent(li, lv2, adj))
        continue;

      Vec3 a = particles[tri.v0].position;
      Vec3 b = particles[tri.v1].position;
      Vec3 c = particles[tri.v2].position;
      ClosestPointResult cp = closestPointOnTriangle(pi, a, b, c);
      dMin = std::min(dMin, cp.distance);
    }

    bounds[li] = gammaP * std::max(dMin, 1e-6f);
  }
}

// Truncate displacement to safety bound
inline void truncateDisplacement(
    SoftParticle& sp,
    Vec3 prevPosition,
    float bound)
{
  Vec3 disp = sp.position - prevPosition;
  float dispMag = disp.length();
  if (dispMag > bound && dispMag > 1e-10f) {
    sp.position = prevPosition + disp * (bound / dispMag);
  }
}

// Detect self-collision contacts within a single soft body
inline void detectSelfCollisionOGC(
    const std::vector<SoftParticle>& particles,
    const SoftBody& sb,
    uint32_t sbIdx,
    const std::vector<SurfaceTriangle>& surface,
    const std::vector<std::vector<uint32_t>>& adj,
    std::vector<SoftContact>& contacts,
    const OGCParams& params = OGCParams())
{
  float r = params.contactRadius;
  float tau = params.getTau();

  // For each surface vertex, check against non-adjacent surface triangles
  for (uint32_t li = 0; li < sb.particleCount; li++) {
    uint32_t gi = sb.particleStart + li;
    if (particles[gi].invMass <= 0.0f) continue;
    Vec3 pp = particles[gi].position;

    for (const auto& tri : surface) {
      uint32_t lv0 = tri.v0 - sb.particleStart;
      uint32_t lv1 = tri.v1 - sb.particleStart;
      uint32_t lv2 = tri.v2 - sb.particleStart;

      // Skip if vertex is part of or adjacent to this triangle
      if (lv0 == li || lv1 == li || lv2 == li) continue;
      if (isAdjacent(li, lv0, adj) || isAdjacent(li, lv1, adj) || isAdjacent(li, lv2, adj))
        continue;

      Vec3 a = particles[tri.v0].position;
      Vec3 b = particles[tri.v1].position;
      Vec3 c = particles[tri.v2].position;

      ClosestPointResult cp = closestPointOnTriangle(pp, a, b, c);
      if (cp.distance >= r) continue;

      // OGC: determine contact normal based on feature type
      Vec3 faceN = (b - a).cross(c - a);
      float fLen = faceN.length();
      if (fLen < 1e-10f) continue;
      faceN = faceN * (1.0f / fLen);

      Vec3 contactNormal;
      if (cp.feature == FEATURE_FACE) {
        // Face block: use face normal, direction based on which side the point is
        contactNormal = (cp.normal.dot(faceN) >= 0.0f) ? faceN : faceN * (-1.0f);
      } else {
        contactNormal = cp.normal; // vertex/edge block: radial direction
      }

      // Full path: two-stage activation
      float d = cp.distance;
      ActivationResult act = ogcActivationFull(d, r, params.contactStiffness, tau);

      if (act.force <= 0.0f) continue;

      float depth = r - d;

      SoftContact sc;
      sc.particleIdx = gi;
      sc.rigidBodyIdx = UINT32_MAX;
      sc.normal = contactNormal;
      sc.depth = depth;
      sc.k = act.hessian;  // Full hessian from activation
      sc.ke = params.contactStiffness * 100.0f;
      sc.friction = params.friction;
      Vec3 t1, t2;
      if (fabsf(contactNormal.x) < 0.9f)
        t1 = contactNormal.cross(Vec3(1, 0, 0)).normalized();
      else
        t1 = contactNormal.cross(Vec3(0, 1, 0)).normalized();
      t2 = contactNormal.cross(t1);
      sc.tangent1 = t1;
      sc.tangent2 = t2;
      contacts.push_back(sc);
    }
  }
}

// =============================================================================
// OGC Contact Force/Hessian evaluator
//
// For solver integration. Uses the existing SoftContact format but applies
// OGC-specific force computation:
//   - Simplified (soft-soft): pure quadratic energy
//   - Full (self-collision): two-stage activation
//
// Since we set sc.k appropriately during detection, the existing
// evaluateContactForceHessian() already handles the force computation correctly:
//   f = k * penetration * n, H = k * (n ⊗ n) + friction
//
// This function provides an alternative that uses the OGC activation function
// directly for more accurate force scaling near the barrier boundary.
// =============================================================================
inline void evaluateOGCContactForceHessian(
    const SoftContact& sc,
    const std::vector<SoftParticle>& particles,
    bool useLogBarrier,  // true for self-collision, false for soft-soft
    float contactRadius,
    float contactStiffness,
    float tau,
    Vec3& outForce, Mat33& outHessian)
{
  outForce = Vec3(0, 0, 0);
  outHessian = Mat33();

  const SoftParticle& sp = particles[sc.particleIdx];
  Vec3 n = sc.normal;
  float depth = sc.depth;

  if (depth <= 0.0f) return;

  float d = contactRadius - depth;  // distance = r - penetration
  ActivationResult act;
  if (useLogBarrier) {
    act = ogcActivationFull(d, contactRadius, contactStiffness, tau);
  } else {
    act = ogcActivationQuadratic(d, contactRadius, contactStiffness);
  }

  if (act.force <= 0.0f) return;

  // Force: push particle along normal
  outForce = n * act.force;
  // Hessian: stiffness along normal direction
  outHessian = outer(n, n) * act.hessian;

  // Friction (same Coulomb model as existing contacts)
  if (sc.friction > 0.0f && act.force > 0.0f) {
    Vec3 dx = sp.position - sp.predictedPosition;
    float dot_n = n.dot(dx);
    Vec3 ut = dx - n * dot_n;
    float utNorm = ut.length();

    if (utNorm > 0.0f) {
      float eps_u = 1e-4f;
      float f1;
      if (utNorm > eps_u)
        f1 = 1.0f / utNorm;
      else
        f1 = (-utNorm / eps_u + 2.0f) / eps_u;

      float scale = sc.friction * act.force * f1;
      outForce = outForce - ut * scale;
      Mat33 P = Mat33::diag(1, 1, 1) - outer(n, n);
      outHessian = outHessian + P * scale;
    }
  }
}

// =============================================================================
// Convenience: detect all 4 paths in one call
//
// Usage in test loop:
//   solver.softContacts.clear();
//   detectAllSoftContacts(solver, surfaces, adjacencies, safetyBounds, params);
//   solver.step(dt);
// =============================================================================
inline void detectAllSoftContacts(
    Solver& solver,
    const std::vector<std::vector<SurfaceTriangle>>& surfaces,
    const std::vector<std::vector<std::vector<uint32_t>>>& adjacencies,
    const OGCParams& params = OGCParams())
{
  // Path 1: Rigid-rigid is handled by existing collideAll/collideBoxGround

  // Path 2: Rigid-soft via SDF
  detectSoftRigidSDF(solver.softParticles, solver.bodies, solver.softBodies,
                     solver.softContacts, params.contactRadius);

  // Path 3: Soft-soft via OGC simplified
  detectSoftSoftOGC(solver.softParticles, solver.softBodies, surfaces,
                    solver.softContacts, params);

  // Path 4: Self-collision via OGC full (for each soft body)
  for (uint32_t si = 0; si < (uint32_t)solver.softBodies.size(); si++) {
    if (si < surfaces.size() && si < adjacencies.size()) {
      detectSelfCollisionOGC(solver.softParticles, solver.softBodies[si], si,
                             surfaces[si], adjacencies[si],
                             solver.softContacts, params);
    }
  }

  // Also detect ground contacts (kept for compatibility)
  detectSoftGroundContacts(solver.softParticles, solver.softContacts,
                           0.0f, params.contactRadius, params.friction);
}

// =============================================================================
// Helper: build all surfaces and adjacencies for all soft bodies
// =============================================================================
inline void buildOGCSurfacesAndAdjacency(
    const std::vector<SoftBody>& softBodies,
    const std::vector<SoftParticle>& particles,
    std::vector<std::vector<SurfaceTriangle>>& surfaces,
    std::vector<std::vector<std::vector<uint32_t>>>& adjacencies)
{
  uint32_t nBodies = (uint32_t)softBodies.size();
  surfaces.resize(nBodies);
  adjacencies.resize(nBodies);

  for (uint32_t si = 0; si < nBodies; si++) {
    surfaces[si].clear();
    extractSurfaceTriangles(softBodies[si], si, particles, surfaces[si]);
    buildAdjacency(softBodies[si], adjacencies[si]);
  }
}

} // namespace AvbdRef
