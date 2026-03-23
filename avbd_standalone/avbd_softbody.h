#pragma once
// =============================================================================
// AVBD Soft Body / Cloth — VBD energy-based deformable body system
//
// Elastic forces (StVK for triangles, Neo-Hookean for tetrahedra, analytic
// dihedral bending) use pure VBD: direct force + Hessian from energy gradient.
// Contact, attachment, and pin constraints use AVBD: adaptive penalty k only,
// no explicit lambda (Lagrange multiplier).
//
// Reference: Newton VBD (SIGGRAPH 2024) — particle_vbd_kernels.py
// =============================================================================

#include "avbd_types.h"
#include "avbd_math.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace AvbdRef {

// =============================================================================
// SoftParticle — 3-DOF mass point (no rotation)
// =============================================================================
struct SoftParticle {
  Vec3 position;
  Vec3 velocity;
  Vec3 prevVelocity;
  Vec3 initialPosition;     // position at start of step
  Vec3 predictedPosition;   // inertial prediction (pos + v*dt + g*dt²)
  float mass;               // 0 = pinned/kinematic
  float invMass;
  float damping;

  // AVBD elastic proximal (mirrors PhysX AvbdSoftParticle)
  float elasticK;            // adaptive proximal weight
  Vec3 outerPosition;        // position snapshot at start of outer iteration (proximal anchor)
  float elasticKMax;         // proximal upper bound

  SoftParticle()
      : mass(1.0f), invMass(1.0f), damping(0.0f),
        elasticK(0.0f), elasticKMax(1e6f) {}
};

// =============================================================================
// VBD Element types — precomputed rest-state data
// =============================================================================

// Triangle element for cloth/shells — StVK membrane energy
struct TriElement {
  uint32_t p0, p1, p2;        // global particle indices
  float DmInv00, DmInv01;     // 2×2 inverse of reference edge matrix
  float DmInv10, DmInv11;
  float restArea;
};

// Tetrahedral element for volumetric soft bodies — Neo-Hookean energy
struct TetElement {
  uint32_t p0, p1, p2, p3;    // global particle indices
  Mat33 DmInv;                 // 3×3 inverse of reference edge matrix
  float restVolume;
};

// Bending element — analytic dihedral angle derivatives
// Vertex order follows Newton convention: [opp0, opp1, edgeStart, edgeEnd]
struct BendingElement {
  uint32_t opp0, opp1;        // wing vertices (opposite to shared edge)
  uint32_t edgeStart, edgeEnd; // shared edge vertices
  float restAngle;
  float restLength;            // rest edge length for stiffness scaling
};

// Edge info — for diagnostic edge length measurements
struct EdgeInfo {
  uint32_t p0, p1;
  float restLength;
};

// =============================================================================
// AVBD constraint types — adaptive penalty k only, NO lambda
// =============================================================================

struct AttachmentConstraint {
  uint32_t particleIdx;      // global soft particle index
  uint32_t rigidBodyIdx;     // index into solver.bodies[]
  Vec3 localOffset;          // attachment point in rigid body local frame
  float k;                   // adaptive penalty
  float kMax;

  AttachmentConstraint()
      : particleIdx(0), rigidBodyIdx(0), k(1e3f), kMax(1e5f) {}
};

struct KinematicPin {
  uint32_t particleIdx;      // global soft particle index
  Vec3 worldTarget;          // fixed world position
  float k;                   // adaptive penalty
  float kMax;

  KinematicPin()
      : particleIdx(0), k(1e4f), kMax(1e6f) {}
};

// =============================================================================
// SoftContact — particle vs ground or rigid body (AVBD penalty)
// =============================================================================
struct SoftContact {
  uint32_t particleIdx;     // global soft particle index
  uint32_t rigidBodyIdx;    // UINT32_MAX = ground
  Vec3 normal;
  float depth;

  float k;                  // adaptive penalty
  float ke;                 // material stiffness cap
  float friction;

  Vec3 tangent1, tangent2;
  Vec3 surfacePoint;        // contact point on surface (for collision projection)

  SoftContact()
      : particleIdx(0), rigidBodyIdx(UINT32_MAX),
        depth(0), k(1e4f), ke(1e6f), friction(0.5f) {}
};

// =============================================================================
// SoftBody — mesh + VBD elements + AVBD constraints
// =============================================================================
struct SoftBody {
  // Particle indices (global into solver.softParticles[])
  uint32_t particleStart;
  uint32_t particleCount;

  // Topology (local indices, offset by particleStart for global)
  std::vector<uint32_t> triangles;   // 3 indices per tri (local)
  std::vector<uint32_t> tetrahedra;  // 4 indices per tet (local)

  // Material
  float youngsModulus;
  float poissonsRatio;
  float density;
  float damping;
  float bendingStiffness;
  float thickness;

  // Lamé parameters (computed from E, nu)
  float mu, lambda;

  // VBD elements (built at setup)
  std::vector<TriElement> triElements;
  std::vector<TetElement> tetElements;
  std::vector<BendingElement> bendElements;

  // Diagnostic edge data
  std::vector<EdgeInfo> edges;

  // Per-particle element adjacency (built at setup, mirrors PhysX)
  struct ParticleElementRef {
    uint32_t index;
    uint8_t vOrder;
  };
  struct ParticleAdjacency {
    std::vector<ParticleElementRef> triRefs;
    std::vector<ParticleElementRef> tetRefs;
    std::vector<ParticleElementRef> bendRefs;
    std::vector<uint32_t> attachmentIndices;
    std::vector<uint32_t> pinIndices;
  };
  std::vector<ParticleAdjacency> adjacency;

  // AVBD constraints
  std::vector<AttachmentConstraint> attachments;
  std::vector<KinematicPin> pins;

  SoftBody()
      : particleStart(0), particleCount(0),
        youngsModulus(1e5f), poissonsRatio(0.3f),
        density(100.0f), damping(0.01f),
        bendingStiffness(0.0f), thickness(0.01f),
        mu(0.0f), lambda(0.0f) {}

  // Compute Lamé parameters from Young's modulus and Poisson ratio
  void computeLameParameters() {
    mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
    lambda = youngsModulus * poissonsRatio /
             ((1.0f + poissonsRatio) * (1.0f - 2.0f * poissonsRatio));
  }

  // Build triangle elements (StVK) from triangle mesh
  void buildTriElements(const std::vector<SoftParticle>& particles) {
    triElements.clear();
    if (triangles.empty()) return;

    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
      uint32_t gp0 = particleStart + triangles[i];
      uint32_t gp1 = particleStart + triangles[i+1];
      uint32_t gp2 = particleStart + triangles[i+2];

      Vec3 e1 = particles[gp1].position - particles[gp0].position;
      Vec3 e2 = particles[gp2].position - particles[gp0].position;

      // Rest area
      float area = e1.cross(e2).length() * 0.5f;
      if (area < 1e-12f) continue;

      // Build 2D reference frame: p0 at origin, p1 on x-axis
      float L1 = e1.length();
      if (L1 < 1e-10f) continue;
      Vec3 edir = e1 * (1.0f / L1);
      float d = e2.dot(edir);
      Vec3 e2perp = e2 - edir * d;
      float h = e2perp.length();
      if (h < 1e-10f) continue;

      // Dm = [[L1, d], [0, h]], DmInv = Dm^{-1}
      float det = L1 * h;
      float invDet = 1.0f / det;

      TriElement te;
      te.p0 = gp0;
      te.p1 = gp1;
      te.p2 = gp2;
      te.DmInv00 = h * invDet;
      te.DmInv01 = -d * invDet;
      te.DmInv10 = 0.0f;
      te.DmInv11 = L1 * invDet;
      te.restArea = area;
      triElements.push_back(te);
    }
  }

  // Build tetrahedral elements (Neo-Hookean) from tet mesh
  void buildTetElements(const std::vector<SoftParticle>& particles) {
    tetElements.clear();
    if (tetrahedra.empty()) return;

    for (size_t i = 0; i + 3 < tetrahedra.size(); i += 4) {
      uint32_t gp0 = particleStart + tetrahedra[i];
      uint32_t gp1 = particleStart + tetrahedra[i+1];
      uint32_t gp2 = particleStart + tetrahedra[i+2];
      uint32_t gp3 = particleStart + tetrahedra[i+3];

      Vec3 e1 = particles[gp1].position - particles[gp0].position;
      Vec3 e2 = particles[gp2].position - particles[gp0].position;
      Vec3 e3 = particles[gp3].position - particles[gp0].position;

      float signedVol = e1.dot(e2.cross(e3)) / 6.0f;

      // Ensure positive winding
      if (signedVol < 0.0f) {
        std::swap(gp1, gp2);
        e1 = particles[gp1].position - particles[gp0].position;
        e2 = particles[gp2].position - particles[gp0].position;
        signedVol = -signedVol;
      }

      if (signedVol < 1e-12f) continue;

      // Dm = [e1 | e2 | e3] (columns)
      Mat33 Dm;
      Dm.m[0][0] = e1.x; Dm.m[1][0] = e1.y; Dm.m[2][0] = e1.z;
      Dm.m[0][1] = e2.x; Dm.m[1][1] = e2.y; Dm.m[2][1] = e2.z;
      Dm.m[0][2] = e3.x; Dm.m[1][2] = e3.y; Dm.m[2][2] = e3.z;

      TetElement te;
      te.p0 = gp0;
      te.p1 = gp1;
      te.p2 = gp2;
      te.p3 = gp3;
      te.DmInv = Dm.inverse();
      te.restVolume = signedVol;
      tetElements.push_back(te);
    }
  }

  // Build bending elements from triangle pairs sharing an edge
  void buildBendingElements(const std::vector<SoftParticle>& particles) {
    bendElements.clear();
    if (bendingStiffness <= 0.0f || triangles.size() < 6) return;

    // Build edge → triangle adjacency
    struct EdgeKey {
      uint32_t a, b;
      bool operator<(const EdgeKey& o) const {
        if (a != o.a) return a < o.a;
        return b < o.b;
      }
    };
    struct TriInfo { uint32_t triIdx; uint32_t opposite; };
    std::map<EdgeKey, std::vector<TriInfo>> edgeToTris;

    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
      uint32_t v[3] = { triangles[i], triangles[i+1], triangles[i+2] };
      uint32_t triIdx = (uint32_t)(i / 3);
      for (int e = 0; e < 3; e++) {
        uint32_t a = particleStart + v[e];
        uint32_t b = particleStart + v[(e+1) % 3];
        uint32_t opp = particleStart + v[(e+2) % 3];
        EdgeKey ek;
        ek.a = std::min(a, b);
        ek.b = std::max(a, b);
        edgeToTris[ek].push_back({triIdx, opp});
      }
    }

    for (auto& [ek, tris] : edgeToTris) {
      if (tris.size() != 2) continue;

      BendingElement be;
      // Newton convention: [opp0, opp1, edgeStart, edgeEnd]
      be.opp0 = tris[0].opposite;
      be.opp1 = tris[1].opposite;
      be.edgeStart = ek.a;
      be.edgeEnd = ek.b;

      // Calculate rest edge length
      Vec3 edgeVec = particles[ek.b].position - particles[ek.a].position;
      be.restLength = edgeVec.length();

      // Compute rest dihedral angle
      be.restAngle = computeDihedralAngle(
          particles[be.opp0].position, particles[be.opp1].position,
          particles[be.edgeStart].position, particles[be.edgeEnd].position);
      bendElements.push_back(be);
    }
  }

  // Build edge info for diagnostics
  void buildEdges(const std::vector<SoftParticle>& particles) {
    struct Edge {
      uint32_t a, b;
      bool operator<(const Edge& o) const {
        if (a != o.a) return a < o.a;
        return b < o.b;
      }
    };
    std::vector<Edge> rawEdges;
    auto addEdge = [&](uint32_t a, uint32_t b) {
      if (a > b) std::swap(a, b);
      rawEdges.push_back({a, b});
    };

    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
      uint32_t v0 = particleStart + triangles[i];
      uint32_t v1 = particleStart + triangles[i+1];
      uint32_t v2 = particleStart + triangles[i+2];
      addEdge(v0, v1); addEdge(v1, v2); addEdge(v2, v0);
    }
    for (size_t i = 0; i + 3 < tetrahedra.size(); i += 4) {
      uint32_t v0 = particleStart + tetrahedra[i];
      uint32_t v1 = particleStart + tetrahedra[i+1];
      uint32_t v2 = particleStart + tetrahedra[i+2];
      uint32_t v3 = particleStart + tetrahedra[i+3];
      addEdge(v0, v1); addEdge(v0, v2); addEdge(v0, v3);
      addEdge(v1, v2); addEdge(v1, v3); addEdge(v2, v3);
    }

    std::sort(rawEdges.begin(), rawEdges.end());
    rawEdges.erase(std::unique(rawEdges.begin(), rawEdges.end(),
                               [](const Edge& a, const Edge& b) {
                                 return a.a == b.a && a.b == b.b;
                               }),
                   rawEdges.end());

    edges.clear();
    edges.reserve(rawEdges.size());
    for (const auto& e : rawEdges) {
      EdgeInfo ei;
      ei.p0 = e.a;
      ei.p1 = e.b;
      ei.restLength = (particles[e.a].position - particles[e.b].position).length();
      edges.push_back(ei);
    }
  }

  // Build all VBD elements from mesh topology
  void buildElements(const std::vector<SoftParticle>& particles) {
    computeLameParameters();
    // Tet mesh: use Neo-Hookean only (no StVK triangles)
    // Tri mesh: use StVK + bending (no tets)
    if (!tetrahedra.empty()) {
      buildTetElements(particles);
    }
    if (!triangles.empty()) {
      buildTriElements(particles);
      buildBendingElements(particles);
    }
    buildEdges(particles);
    // Note: buildAdjacency() is called separately after all constraints
    // (pins, attachments) are added, typically at the start of step().
  }

  void buildAdjacency() {
    adjacency.resize(particleCount);
    for (uint32_t i = 0; i < particleCount; i++) {
      adjacency[i].triRefs.clear();
      adjacency[i].tetRefs.clear();
      adjacency[i].bendRefs.clear();
      adjacency[i].attachmentIndices.clear();
      adjacency[i].pinIndices.clear();
    }
    for (uint32_t ei = 0; ei < (uint32_t)triElements.size(); ei++) {
      const TriElement& tri = triElements[ei];
      uint32_t verts[3] = { tri.p0, tri.p1, tri.p2 };
      for (uint8_t v = 0; v < 3; v++) {
        uint32_t li = verts[v] - particleStart;
        if (li < particleCount)
          adjacency[li].triRefs.push_back({ei, v});
      }
    }
    for (uint32_t ei = 0; ei < (uint32_t)tetElements.size(); ei++) {
      const TetElement& tet = tetElements[ei];
      uint32_t verts[4] = { tet.p0, tet.p1, tet.p2, tet.p3 };
      for (uint8_t v = 0; v < 4; v++) {
        uint32_t li = verts[v] - particleStart;
        if (li < particleCount)
          adjacency[li].tetRefs.push_back({ei, v});
      }
    }
    for (uint32_t ei = 0; ei < (uint32_t)bendElements.size(); ei++) {
      const BendingElement& be = bendElements[ei];
      uint32_t verts[4] = { be.opp0, be.opp1, be.edgeStart, be.edgeEnd };
      for (uint8_t v = 0; v < 4; v++) {
        uint32_t li = verts[v] - particleStart;
        if (li < particleCount)
          adjacency[li].bendRefs.push_back({ei, v});
      }
    }
    for (uint32_t ai = 0; ai < (uint32_t)attachments.size(); ai++) {
      uint32_t li = attachments[ai].particleIdx - particleStart;
      if (li < particleCount)
        adjacency[li].attachmentIndices.push_back(ai);
    }
    for (uint32_t pi = 0; pi < (uint32_t)pins.size(); pi++) {
      uint32_t li = pins[pi].particleIdx - particleStart;
      if (li < particleCount)
        adjacency[li].pinIndices.push_back(pi);
    }
  }

  // =========================================================================
  // Dihedral angle computation (Newton convention: opp0, opp1, edge0, edge1)
  // =========================================================================
  static float computeDihedralAngle(Vec3 x0, Vec3 x1, Vec3 x2, Vec3 x3) {
    // x0, x1 = opposite (wing) vertices
    // x2, x3 = shared edge vertices
    Vec3 e = x3 - x2;
    float eLen = e.length();
    if (eLen < 1e-10f) return 0.0f;
    Vec3 eHat = e * (1.0f / eLen);

    Vec3 x02 = x2 - x0, x03 = x3 - x0;
    Vec3 x13 = x3 - x1, x12 = x2 - x1;

    Vec3 n1 = x02.cross(x03);
    Vec3 n2 = x13.cross(x12);
    float n1Len = n1.length();
    float n2Len = n2.length();
    if (n1Len < 1e-10f || n2Len < 1e-10f) return 0.0f;
    Vec3 n1Hat = n1 * (1.0f / n1Len);
    Vec3 n2Hat = n2 * (1.0f / n2Len);

    float sinA = n1Hat.cross(n2Hat).dot(eHat);
    float cosA = std::max(-1.0f, std::min(1.0f, n1Hat.dot(n2Hat)));
    return atan2f(sinA, cosA);
  }
};

// =============================================================================
// Mesh generation utilities
// =============================================================================

inline void generateCubeTets(Vec3 center, float halfSize,
                             std::vector<Vec3>& outVerts,
                             std::vector<uint32_t>& outTets) {
  float h = halfSize;
  outVerts = {
    center + Vec3(-h, -h, -h),
    center + Vec3( h, -h, -h),
    center + Vec3( h,  h, -h),
    center + Vec3(-h,  h, -h),
    center + Vec3(-h, -h,  h),
    center + Vec3( h, -h,  h),
    center + Vec3( h,  h,  h),
    center + Vec3(-h,  h,  h),
  };
  outTets = {
    0, 1, 3, 4,
    1, 2, 3, 6,
    3, 4, 6, 7,
    1, 4, 5, 6,
    1, 3, 4, 6,
  };
}

inline void generateSubdividedCubeTets(Vec3 center, float halfSize, int N,
                                       std::vector<Vec3>& outVerts,
                                       std::vector<uint32_t>& outTets) {
  outVerts.clear();
  outTets.clear();
  float cellSize = 2.0f * halfSize / (float)N;
  Vec3 origin = center - Vec3(halfSize, halfSize, halfSize);

  for (int iz = 0; iz <= N; iz++)
    for (int iy = 0; iy <= N; iy++)
      for (int ix = 0; ix <= N; ix++)
        outVerts.push_back(origin + Vec3(ix * cellSize, iy * cellSize, iz * cellSize));

  auto idx = [&](int ix, int iy, int iz) -> uint32_t {
    return (uint32_t)(iz * (N+1) * (N+1) + iy * (N+1) + ix);
  };

  for (int iz = 0; iz < N; iz++)
    for (int iy = 0; iy < N; iy++)
      for (int ix = 0; ix < N; ix++) {
        uint32_t v[8] = {
          idx(ix,   iy,   iz),
          idx(ix+1, iy,   iz),
          idx(ix+1, iy+1, iz),
          idx(ix,   iy+1, iz),
          idx(ix,   iy,   iz+1),
          idx(ix+1, iy,   iz+1),
          idx(ix+1, iy+1, iz+1),
          idx(ix,   iy+1, iz+1),
        };
        outTets.insert(outTets.end(), {v[0], v[1], v[3], v[4]});
        outTets.insert(outTets.end(), {v[1], v[2], v[3], v[6]});
        outTets.insert(outTets.end(), {v[3], v[4], v[6], v[7]});
        outTets.insert(outTets.end(), {v[1], v[4], v[5], v[6]});
        outTets.insert(outTets.end(), {v[1], v[3], v[4], v[6]});
      }
}

inline void generateConeTets(Vec3 center, float radius, float height, int N,
                             std::vector<Vec3>& outVerts,
                             std::vector<uint32_t>& outTets) {
  outVerts.clear();
  outTets.clear();

  const int nLayers = std::max(N, 2);
  const int nRing = std::max(4 * N, 8);
  const float pi2 = 2.0f * 3.14159265358979f;

  for (int i = 0; i < nLayers; i++) {
    float t = (float)i / (float)nLayers;
    float h = t * height;
    float r = radius * (1.0f - t);
    outVerts.push_back(center + Vec3(0, h, 0));
    for (int j = 0; j < nRing; j++) {
      float angle = pi2 * (float)j / (float)nRing;
      outVerts.push_back(center + Vec3(r * cosf(angle), h, r * sinf(angle)));
    }
  }

  uint32_t apexIdx = (uint32_t)outVerts.size();
  outVerts.push_back(center + Vec3(0, height, 0));

  const int stride = 1 + nRing;
  auto ci = [stride](int layer) -> uint32_t { return (uint32_t)(layer * stride); };
  auto ri = [stride, nRing](int layer, int j) -> uint32_t {
    return (uint32_t)(layer * stride + 1 + ((j % nRing + nRing) % nRing));
  };

  for (int i = 0; i + 1 < nLayers; i++) {
    for (int j = 0; j < nRing; j++) {
      outTets.insert(outTets.end(), {ci(i), ri(i,j+1), ri(i,j), ci(i+1)});
      outTets.insert(outTets.end(), {ri(i,j), ri(i,j+1), ci(i+1), ri(i+1,j)});
      outTets.insert(outTets.end(), {ri(i,j+1), ci(i+1), ri(i+1,j), ri(i+1,j+1)});
    }
  }

  { // apex cap
    int top = nLayers - 1;
    for (int j = 0; j < nRing; j++)
      outTets.insert(outTets.end(), {ci(top), ri(top,j+1), ri(top,j), apexIdx});
  }

  // fix orientation
  for (size_t t = 0; t + 3 < outTets.size(); t += 4) {
    Vec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
    Vec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
    Vec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
    if (e1.dot(e2.cross(e3)) < 0.0f)
      std::swap(outTets[t+1], outTets[t+2]);
  }
}

inline void generateClothGrid(Vec3 center, float sizeX, float sizeZ,
                              int M, int N,
                              std::vector<Vec3>& outVerts,
                              std::vector<uint32_t>& outTris) {
  outVerts.clear();
  outTris.clear();
  float dx = sizeX / (float)(M - 1);
  float dz = sizeZ / (float)(N - 1);
  Vec3 origin = center - Vec3(sizeX * 0.5f, 0, sizeZ * 0.5f);

  for (int j = 0; j < N; j++)
    for (int i = 0; i < M; i++)
      outVerts.push_back(origin + Vec3(i * dx, 0, j * dz));

  for (int j = 0; j < N - 1; j++)
    for (int i = 0; i < M - 1; i++) {
      uint32_t v00 = (uint32_t)(j * M + i);
      uint32_t v10 = v00 + 1;
      uint32_t v01 = v00 + (uint32_t)M;
      uint32_t v11 = v01 + 1;
      outTris.insert(outTris.end(), {v00, v10, v01});
      outTris.insert(outTris.end(), {v10, v11, v01});
    }
}

// =============================================================================
// SDF-based voxel tet generator (Jolt-style simple grid approach)
// =============================================================================
// Voxelizes SDF on a regular grid, includes cells with any corner inside,
// and decomposes each hex cell into 5 alternating tets (Newton parity).
// No surface projection or relaxation — simple and stable.
// =============================================================================

// ---- SDF primitives --------------------------------------------------------

inline float sdfBox(Vec3 p, Vec3 halfExtents) {
  Vec3 d(fabsf(p.x) - halfExtents.x,
         fabsf(p.y) - halfExtents.y,
         fabsf(p.z) - halfExtents.z);
  Vec3 clamped(std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f));
  float outside = clamped.length();
  float inside = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
  return outside + inside;
}

inline float sdfSphere(Vec3 p, float radius) {
  return p.length() - radius;
}

inline float sdfCone(Vec3 p, float radius, float height) {
  float r = sqrtf(p.x * p.x + p.z * p.z);
  float sideLen = sqrtf(radius * radius + height * height);
  float nr = height / sideLen;

  if (p.y < 0.0f)
    return std::max(-p.y, sqrtf(r * r + p.y * p.y) - radius);
  if (p.y > height) {
    float dr = r, dy = p.y - height;
    return sqrtf(dr * dr + dy * dy);
  }
  float t = std::max(0.0f, std::min(1.0f,
    ((r - radius) * (-radius) + p.y * height) / (radius * radius + height * height)));
  float cx = radius * (1.0f - t);
  float cy = height * t;
  float dist = sqrtf((r - cx) * (r - cx) + (p.y - cy) * (p.y - cy));
  bool inside = (nr * (r - radius * (1.0f - p.y / height)) <= 0.0f && p.y >= 0.0f);
  return inside ? -dist : dist;
}

inline float sdfCylinder(Vec3 p, float radius, float halfHeight) {
  float r = sqrtf(p.x * p.x + p.z * p.z);
  float dr = r - radius;
  float dy = fabsf(p.y) - halfHeight;
  float outsideDist = Vec3(std::max(dr, 0.0f), std::max(dy, 0.0f), 0.0f).length();
  float insideDist = std::min(std::max(dr, dy), 0.0f);
  return outsideDist + insideDist;
}

// ---- Core SDF to tet generator (simple grid, no surface projection) --------

template<typename SdfFunc>
inline void generateTetsFromSdf(
    SdfFunc sdfFunc,
    Vec3 aabbMin, Vec3 aabbMax,
    Vec3 worldCenter,
    int resolution,
    std::vector<Vec3>& outVerts,
    std::vector<uint32_t>& outTets)
{
  outVerts.clear();
  outTets.clear();

  Vec3 extent = aabbMax - aabbMin;
  float longest = std::max({extent.x, extent.y, extent.z});
  float cellSize = longest / (float)resolution;
  int nx = std::max(1, (int)ceilf(extent.x / cellSize));
  int ny = std::max(1, (int)ceilf(extent.y / cellSize));
  int nz = std::max(1, (int)ceilf(extent.z / cellSize));
  Vec3 gridExtent(nx * cellSize, ny * cellSize, nz * cellSize);
  Vec3 gridOrigin = (aabbMin + aabbMax) * 0.5f - gridExtent * 0.5f;

  int vnx = nx + 1, vny = ny + 1, vnz = nz + 1;
  int totalGridVerts = vnx * vny * vnz;
  std::vector<Vec3> gridVerts(totalGridVerts);
  std::vector<float> gridSdf(totalGridVerts);

  for (int iz = 0; iz < vnz; iz++)
    for (int iy = 0; iy < vny; iy++)
      for (int ix = 0; ix < vnx; ix++) {
        int gi = iz * vny * vnx + iy * vnx + ix;
        Vec3 localP = gridOrigin + Vec3(ix * cellSize, iy * cellSize, iz * cellSize);
        gridVerts[gi] = localP;
        gridSdf[gi] = sdfFunc(localP);
      }

  auto gridIdx = [vnx, vny](int ix, int iy, int iz) -> int {
    return iz * vny * vnx + iy * vnx + ix;
  };

  // Classify cells and emit tets — include cells with any corner inside
  std::vector<uint32_t> vertRemap(totalGridVerts, UINT32_MAX);
  uint32_t nextVert = 0;

  auto emitVert = [&](int gi) -> uint32_t {
    if (vertRemap[gi] == UINT32_MAX) {
      vertRemap[gi] = nextVert++;
      outVerts.push_back(gridVerts[gi] + worldCenter);
    }
    return vertRemap[gi];
  };

  for (int iz = 0; iz < nz; iz++)
    for (int iy = 0; iy < ny; iy++)
      for (int ix = 0; ix < nx; ix++) {
        int g[8] = {
          gridIdx(ix, iy, iz),     gridIdx(ix+1, iy, iz),
          gridIdx(ix+1, iy+1, iz), gridIdx(ix, iy+1, iz),
          gridIdx(ix, iy, iz+1),   gridIdx(ix+1, iy, iz+1),
          gridIdx(ix+1, iy+1, iz+1), gridIdx(ix, iy+1, iz+1)
        };

        bool anyInside = false;
        for (int c = 0; c < 8; c++)
          if (gridSdf[g[c]] <= 0.0f) { anyInside = true; break; }
        if (!anyInside) continue;

        uint32_t v[8];
        for (int c = 0; c < 8; c++)
          v[c] = emitVert(g[c]);

        // Newton's alternating 5-tet decomposition
        if ((ix ^ iy ^ iz) & 1) {
          outTets.insert(outTets.end(), {v[0], v[1], v[4], v[3]});
          outTets.insert(outTets.end(), {v[2], v[3], v[6], v[1]});
          outTets.insert(outTets.end(), {v[5], v[4], v[1], v[6]});
          outTets.insert(outTets.end(), {v[7], v[6], v[3], v[4]});
          outTets.insert(outTets.end(), {v[4], v[1], v[6], v[3]});
        } else {
          outTets.insert(outTets.end(), {v[1], v[2], v[5], v[0]});
          outTets.insert(outTets.end(), {v[3], v[0], v[7], v[2]});
          outTets.insert(outTets.end(), {v[4], v[7], v[0], v[5]});
          outTets.insert(outTets.end(), {v[6], v[5], v[2], v[7]});
          outTets.insert(outTets.end(), {v[5], v[2], v[7], v[0]});
        }
      }
}

// ---- Convenience wrappers --------------------------------------------------

inline void generateBoxTetsSdf(Vec3 center, Vec3 halfExtents, int resolution,
                               std::vector<Vec3>& outVerts,
                               std::vector<uint32_t>& outTets) {
  Vec3 margin(halfExtents.x * 0.05f, halfExtents.y * 0.05f, halfExtents.z * 0.05f);
  generateTetsFromSdf(
    [halfExtents](Vec3 p) { return sdfBox(p, halfExtents); },
    Vec3(0,0,0) - halfExtents - margin, halfExtents + margin,
    center, resolution, outVerts, outTets);
}

inline void generateSphereTetsSdf(Vec3 center, float radius, int resolution,
                                  std::vector<Vec3>& outVerts,
                                  std::vector<uint32_t>& outTets) {
  float m = radius * 0.05f;
  generateTetsFromSdf(
    [radius](Vec3 p) { return sdfSphere(p, radius); },
    Vec3(-radius - m, -radius - m, -radius - m),
    Vec3( radius + m,  radius + m,  radius + m),
    center, resolution, outVerts, outTets);
}

inline void generateConeTetsSdf(Vec3 center, float radius, float height, int resolution,
                                std::vector<Vec3>& outVerts,
                                std::vector<uint32_t>& outTets) {
  float m = radius * 0.05f;
  generateTetsFromSdf(
    [radius, height](Vec3 p) { return sdfCone(p, radius, height); },
    Vec3(-radius - m, -m, -radius - m),
    Vec3( radius + m, height + m,  radius + m),
    center, resolution, outVerts, outTets);
}

inline void generateCylinderTetsSdf(Vec3 center, float radius, float halfHeight,
                                    int resolution,
                                    std::vector<Vec3>& outVerts,
                                    std::vector<uint32_t>& outTets) {
  float m = radius * 0.05f;
  generateTetsFromSdf(
    [radius, halfHeight](Vec3 p) { return sdfCylinder(p, radius, halfHeight); },
    Vec3(-radius - m, -halfHeight - m, -radius - m),
    Vec3( radius + m,  halfHeight + m,  radius + m),
    center, resolution, outVerts, outTets);
}

// =============================================================================
// VBD Force/Hessian evaluators — return (force, hessian) per vertex
// =============================================================================

// ---- StVK membrane energy for triangles ----
// Returns force and Hessian contribution for vertex vOrder (0,1,2) of triangle
inline void evaluateStVKForceHessian(
    const TriElement& tri, int vOrder,
    float mu, float lam,
    const std::vector<SoftParticle>& particles,
    Vec3& outForce, Mat33& outHessian)
{
  Vec3 x0 = particles[tri.p0].position;
  Vec3 x01 = particles[tri.p1].position - x0;
  Vec3 x02 = particles[tri.p2].position - x0;

  float D00 = tri.DmInv00, D01 = tri.DmInv01;
  float D10 = tri.DmInv10, D11 = tri.DmInv11;

  // Deformation gradient F = [f0, f1] (3×2 stored as two Vec3)
  Vec3 f0 = x01 * D00 + x02 * D10;
  Vec3 f1 = x01 * D01 + x02 * D11;

  // Green strain: G = 0.5 * (F^T F - I)
  float f0f0 = f0.dot(f0);
  float f1f1 = f1.dot(f1);
  float f0f1 = f0.dot(f1);

  float G00 = 0.5f * (f0f0 - 1.0f);
  float G11 = 0.5f * (f1f1 - 1.0f);
  float G01 = 0.5f * f0f1;

  float Gfro2 = G00 * G00 + G11 * G11 + 2.0f * G01 * G01;
  if (Gfro2 < 1e-20f) {
    outForce = Vec3(0, 0, 0);
    outHessian = Mat33();
    return;
  }

  float trG = G00 + G11;

  // PK1 stress: P = F * (2μG + λ·tr(G)·I)
  float ltrG = lam * trG;
  float twoMu = 2.0f * mu;
  Vec3 PK1_0 = f0 * (twoMu * G00 + ltrG) + f1 * (twoMu * G01);
  Vec3 PK1_1 = f0 * (twoMu * G01) + f1 * (twoMu * G11 + ltrG);

  // Scalar derivatives: df0/dxi, df1/dxi
  float df0, df1;
  if (vOrder == 0) {
    df0 = -D00 - D10;
    df1 = -D01 - D11;
  } else if (vOrder == 1) {
    df0 = D00;
    df1 = D01;
  } else {
    df0 = D10;
    df1 = D11;
  }

  // Force = -area * (PK1_0 * df0 + PK1_1 * df1)
  outForce = (PK1_0 * df0 + PK1_1 * df1) * (-tri.restArea);

  // Hessian via d²ψ/dF² chain rule
  float df0sq = df0 * df0;
  float df1sq = df1 * df1;
  float df0df1 = df0 * df1;

  float Ic = f0f0 + f1f1;
  float two_dpsi_dIc = -mu + (0.5f * Ic - 1.0f) * lam;
  Mat33 I33 = Mat33::diag(1, 1, 1);

  Mat33 f0f0m = outer(f0, f0);
  Mat33 f1f1m = outer(f1, f1);
  Mat33 f0f1m = outer(f0, f1);
  Mat33 f1f0m = outer(f1, f0);

  Mat33 H00 = f0f0m * lam + I33 * two_dpsi_dIc
            + (I33 * f0f0 + f0f0m * 2.0f + f1f1m) * mu;
  Mat33 H01 = f0f1m * lam + (I33 * f0f1 + f1f0m) * mu;
  Mat33 H11 = f1f1m * lam + I33 * two_dpsi_dIc
            + (I33 * f1f1 + f1f1m * 2.0f + f0f0m) * mu;

  float area = tri.restArea;
  outHessian = H00 * (df0sq * area) + H11 * (df1sq * area)
             + (H01 + H01.transpose()) * (df0df1 * area);
}

// ---- Neo-Hookean energy for tetrahedra ----
// Simplified per-vertex form (cofactor derivative vanishes after contraction)
inline void evaluateNeoHookeanForceHessian(
    const TetElement& tet, int vOrder,
    float mu, float lam,
    const std::vector<SoftParticle>& particles,
    Vec3& outForce, Mat33& outHessian)
{
  Vec3 p0 = particles[tet.p0].position;
  Vec3 e1 = particles[tet.p1].position - p0;
  Vec3 e2 = particles[tet.p2].position - p0;
  Vec3 e3 = particles[tet.p3].position - p0;

  // Ds = [e1 | e2 | e3] (columns)
  Mat33 Ds;
  Ds.m[0][0] = e1.x; Ds.m[1][0] = e1.y; Ds.m[2][0] = e1.z;
  Ds.m[0][1] = e2.x; Ds.m[1][1] = e2.y; Ds.m[2][1] = e2.z;
  Ds.m[0][2] = e3.x; Ds.m[1][2] = e3.y; Ds.m[2][2] = e3.z;

  // F = Ds * DmInv
  Mat33 F = Ds.mul(tet.DmInv);

  // Determinant of F
  float J = F.m[0][0] * (F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1])
          - F.m[0][1] * (F.m[1][0] * F.m[2][2] - F.m[1][2] * F.m[2][0])
          + F.m[0][2] * (F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0]);

  // Safe alpha = 1 + mu/lambda
  float lam_safe = (fabsf(lam) < 1e-6f) ? 1e-6f : lam;
  float alpha = 1.0f + mu / lam_safe;

  // Cofactor matrix (numerically stable, no inverse needed)
  Mat33 cof;
  cof.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
  cof.m[1][0] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
  cof.m[2][0] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];
  cof.m[0][1] = F.m[0][2] * F.m[2][1] - F.m[0][1] * F.m[2][2];
  cof.m[1][1] = F.m[0][0] * F.m[2][2] - F.m[0][2] * F.m[2][0];
  cof.m[2][1] = F.m[0][1] * F.m[2][0] - F.m[0][0] * F.m[2][1];
  cof.m[0][2] = F.m[0][1] * F.m[1][2] - F.m[0][2] * F.m[1][1];
  cof.m[1][2] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
  cof.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];

  // Vertex selector m (from DmInv rows)
  const Mat33& DI = tet.DmInv;
  Vec3 m;
  if (vOrder == 0) {
    m = Vec3(-(DI.m[0][0] + DI.m[1][0] + DI.m[2][0]),
             -(DI.m[0][1] + DI.m[1][1] + DI.m[2][1]),
             -(DI.m[0][2] + DI.m[1][2] + DI.m[2][2]));
  } else if (vOrder == 1) {
    m = Vec3(DI.m[0][0], DI.m[0][1], DI.m[0][2]);
  } else if (vOrder == 2) {
    m = Vec3(DI.m[1][0], DI.m[1][1], DI.m[1][2]);
  } else {
    m = Vec3(DI.m[2][0], DI.m[2][1], DI.m[2][2]);
  }

  // F * m and cof * m
  Vec3 Fm = F * m;
  Vec3 cofm = cof * m;

  float V0 = tet.restVolume;

  // Force = -V0 * (mu * F*m + lambda * (J - alpha) * cof*m)
  outForce = (Fm * mu + cofm * (lam * (J - alpha))) * (-V0);

  // Hessian = V0 * (mu * |m|² * I + lambda * outer(cof*m, cof*m))
  float m2 = m.dot(m);
  outHessian = Mat33::diag(mu * m2, mu * m2, mu * m2) * V0
             + outer(cofm, cofm) * (lam * V0);
}

// ---- Analytic dihedral angle bending ----
// Gauss-Newton Hessian: H ≈ k * (dθ/dx)(dθ/dx)^T
inline void evaluateBendingForceHessian(
    const BendingElement& be, int vOrder,
    float stiffness,
    const std::vector<SoftParticle>& particles,
    Vec3& outForce, Mat33& outHessian)
{
  const float eps = 1e-6f;

  Vec3 x0 = particles[be.opp0].position;     // opp0
  Vec3 x1 = particles[be.opp1].position;     // opp1
  Vec3 x2 = particles[be.edgeStart].position; // edge start
  Vec3 x3 = particles[be.edgeEnd].position;   // edge end

  Vec3 e = x3 - x2;
  Vec3 x02 = x2 - x0, x03 = x3 - x0;
  Vec3 x13 = x3 - x1, x12 = x2 - x1;

  Vec3 n1 = x02.cross(x03);
  Vec3 n2 = x13.cross(x12);

  float n1Norm = n1.length();
  float n2Norm = n2.length();
  float eNorm = e.length();

  if (n1Norm < eps || n2Norm < eps || eNorm < eps) {
    outForce = Vec3(0, 0, 0);
    outHessian = Mat33();
    return;
  }

  Vec3 n1Hat = n1 * (1.0f / n1Norm);
  Vec3 n2Hat = n2 * (1.0f / n2Norm);
  Vec3 eHat = e * (1.0f / eNorm);

  float sinTheta = n1Hat.cross(n2Hat).dot(eHat);
  float cosTheta = std::max(-1.0f, std::min(1.0f, n1Hat.dot(n2Hat)));
  float theta = atan2f(sinTheta, cosTheta);

  float k = stiffness * be.restLength;
  float dE_dtheta = k * (theta - be.restAngle);

  // Helper: compute d(n_hat)/dx given |n|, n_hat, and d(n_unnorm)/dx
  // d(n_hat)/dx = (1/|n|) * (I - n_hat * n_hat^T) * dn/dx
  auto normalizedDerivative = [](float unnormLen, const Vec3& nHat, const Mat33& dNdx) -> Mat33 {
    Mat33 P = Mat33::diag(1, 1, 1) - outer(nHat, nHat);
    return P.mul(dNdx) * (1.0f / unnormLen);
  };

  // Helper: compute dtheta/dx from dn1hat/dx, dn2hat/dx
  auto angleDerivative = [](const Vec3& n1h, const Vec3& n2h, const Vec3& eh,
                            const Mat33& dn1dx, const Mat33& dn2dx,
                            float sinT, float cosT,
                            const Mat33& skN1, const Mat33& skN2) -> Vec3 {
    // dsin/dx = ([n1]× dn2/dx - [n2]× dn1/dx)^T e_hat
    Mat33 dSinMat = skN1.mul(dn2dx) - skN2.mul(dn1dx);
    Vec3 dSin = dSinMat.transpose() * eh;
    // dcos/dx = (dn1/dx)^T n2 + (dn2/dx)^T n1
    Vec3 dCos = dn1dx.transpose() * n2h + dn2dx.transpose() * n1h;
    // dtheta = dsin * cos - dcos * sin
    return dSin * cosT - dCos * sinT;
  };

  // Skew matrices
  Mat33 skE = skew(e);
  Mat33 skX03 = skew(x03);
  Mat33 skX02 = skew(x02);
  Mat33 skX13 = skew(x13);
  Mat33 skX12 = skew(x12);
  Mat33 skN1 = skew(n1Hat);
  Mat33 skN2 = skew(n2Hat);

  // n1 = x02 × x03, dn1/dx for each vertex:
  // dn1/dx0: n1 = (x2-x0)×(x3-x0). d/dx0 = -[x03-x02]× = [e]× ... Newton uses skew(e)
  // Actually from chain rule: d(x02×x03)/dx0 = d/dx0((x2-x0)×(x3-x0))
  //   = (-I)×(x3-x0) + (x2-x0)×(-I) = -(x03)× + (x02)× ... no
  //   For cross product a×b: d(a×b)/da = -[b]×, d(a×b)/db = [a]×
  //   a = x02 = x2-x0, da/dx0 = -I
  //   b = x03 = x3-x0, db/dx0 = -I
  //   dn1/dx0 = -[b]× * (-I) + [a]× * (-I) = [x03]× - [x02]× = [x03 - x02]× = [e]×
  Mat33 dn1_dx0_unnorm = skE;
  Mat33 dn1_dx1_unnorm = Mat33(); // n1 doesn't depend on x1
  // dn1/dx2: a=x02, da/dx2=I. dn1/dx2 = -[x03]× * I = -[x03]×
  Mat33 dn1_dx2_unnorm = skX03 * (-1.0f);
  // dn1/dx3: b=x03, db/dx3=I. dn1/dx3 = [x02]× * I = [x02]×
  Mat33 dn1_dx3_unnorm = skX02;

  // n2 = x13 × x12
  // a=x13=x3-x1, b=x12=x2-x1
  // dn2/dx0 = 0
  Mat33 dn2_dx0_unnorm = Mat33();
  // dn2/dx1: da/dx1=-I, db/dx1=-I
  //   dn2/dx1 = -[x12]×*(-I) + [x13]×*(-I) = [x12]× - [x13]× = -[x13-x12]× = -[e]×
  Mat33 dn2_dx1_unnorm = skE * (-1.0f);
  // dn2/dx2: b=x12, db/dx2=I. dn2/dx2 = [x13]× * I = [x13]×
  Mat33 dn2_dx2_unnorm = skX13;
  // dn2/dx3: a=x13, da/dx3=I. dn2/dx3 = -[x12]× * I = -[x12]×
  Mat33 dn2_dx3_unnorm = skX12 * (-1.0f);

  // Normalized derivatives
  Mat33 dn1hat_dx0 = normalizedDerivative(n1Norm, n1Hat, dn1_dx0_unnorm);
  Mat33 dn1hat_dx1 = Mat33(); // zero
  Mat33 dn1hat_dx2 = normalizedDerivative(n1Norm, n1Hat, dn1_dx2_unnorm);
  Mat33 dn1hat_dx3 = normalizedDerivative(n1Norm, n1Hat, dn1_dx3_unnorm);

  Mat33 dn2hat_dx0 = Mat33(); // zero
  Mat33 dn2hat_dx1 = normalizedDerivative(n2Norm, n2Hat, dn2_dx1_unnorm);
  Mat33 dn2hat_dx2 = normalizedDerivative(n2Norm, n2Hat, dn2_dx2_unnorm);
  Mat33 dn2hat_dx3 = normalizedDerivative(n2Norm, n2Hat, dn2_dx3_unnorm);

  // Angle derivatives for all 4 vertices
  Vec3 dtheta_dx0 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx0, dn2hat_dx0,
                                     sinTheta, cosTheta, skN1, skN2);
  Vec3 dtheta_dx1 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx1, dn2hat_dx1,
                                     sinTheta, cosTheta, skN1, skN2);
  Vec3 dtheta_dx2 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx2, dn2hat_dx2,
                                     sinTheta, cosTheta, skN1, skN2);
  Vec3 dtheta_dx3 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx3, dn2hat_dx3,
                                     sinTheta, cosTheta, skN1, skN2);

  // Select derivative for current vertex
  Vec3 dtheta_dx;
  switch (vOrder) {
    case 0: dtheta_dx = dtheta_dx0; break; // opp0
    case 1: dtheta_dx = dtheta_dx1; break; // opp1
    case 2: dtheta_dx = dtheta_dx2; break; // edgeStart
    case 3: dtheta_dx = dtheta_dx3; break; // edgeEnd
    default: outForce = Vec3(0,0,0); outHessian = Mat33(); return;
  }

  // Force = -dE/dtheta * dtheta/dx
  outForce = dtheta_dx * (-dE_dtheta);
  // Gauss-Newton Hessian: k * outer(dtheta/dx, dtheta/dx)
  outHessian = outer(dtheta_dx, dtheta_dx) * k;
}

// =============================================================================
// AVBD contact/pin/attachment force evaluators (penalty only, no lambda)
// =============================================================================

// Soft contact penalty force (particle side, VBD convention)
inline void evaluateContactForceHessian(
    const SoftContact& sc,
    const std::vector<SoftParticle>& particles,
    Vec3& outForce, Mat33& outHessian)
{
  outForce = Vec3(0, 0, 0);
  outHessian = Mat33();

  const SoftParticle& sp = particles[sc.particleIdx];
  Vec3 n = sc.normal;

  float penetration;
  if (sc.rigidBodyIdx == UINT32_MAX) {
    // Ground contact: depth = -p·n
    penetration = -(sp.position.dot(n));
  } else {
    penetration = sc.depth;
  }

  if (penetration <= 0.0f) return;

  float fn = sc.k * penetration;

  // Normal force: push particle out (VBD: force toward solution)
  outForce = n * fn;
  outHessian = outer(n, n) * sc.k;

  // Friction (Coulomb penalty, IPC-style regularization)
  if (sc.friction > 0.0f && fn > 0.0f) {
    Vec3 dx = sp.position - sp.predictedPosition;
    float dot_n = n.dot(dx);
    Vec3 ut = dx - n * dot_n; // tangential slip
    float utNorm = ut.length();

    if (utNorm > 0.0f) {
      float eps_u = 1e-4f;
      float f1;
      if (utNorm > eps_u)
        f1 = 1.0f / utNorm;
      else
        f1 = (-utNorm / eps_u + 2.0f) / eps_u;

      float scale = sc.friction * fn * f1;
      outForce = outForce - ut * scale;
      Mat33 P = Mat33::diag(1, 1, 1) - outer(n, n);
      outHessian = outHessian + P * scale;
    }
  }
}

// Kinematic pin penalty force (VBD convention)
inline void evaluatePinForceHessian(
    const KinematicPin& kp,
    const std::vector<SoftParticle>& particles,
    Vec3& outForce, Mat33& outHessian)
{
  Vec3 C = particles[kp.particleIdx].position - kp.worldTarget;
  // Energy = k/2 * |C|², force = -k*C, Hessian = k*I
  outForce = C * (-kp.k);
  outHessian = Mat33::diag(kp.k, kp.k, kp.k);
}

// Attachment penalty force on particle (VBD convention)
inline void evaluateAttachmentForceHessian_particle(
    const AttachmentConstraint& ac,
    const std::vector<SoftParticle>& particles,
    const std::vector<Body>& rigidBodies,
    Vec3& outForce, Mat33& outHessian)
{
  const Body& rb = rigidBodies[ac.rigidBodyIdx];
  Vec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
  Vec3 C = particles[ac.particleIdx].position - worldAnchor;
  outForce = C * (-ac.k);
  outHessian = Mat33::diag(ac.k, ac.k, ac.k);
}

// Attachment penalty contribution to rigid body (AVBD convention for rigid solver)
inline void addAttachmentContribution_rigid(
    const AttachmentConstraint& ac,
    uint32_t bodyIdx,
    const std::vector<SoftParticle>& particles,
    const std::vector<Body>& rigidBodies,
    float dt, Mat66& lhs, Vec6& rhs)
{
  if (bodyIdx != ac.rigidBodyIdx) return;

  const Body& rb = rigidBodies[bodyIdx];
  Vec3 worldOffset = rb.rotation.rotate(ac.localOffset);
  Vec3 worldAnchor = rb.position + worldOffset;
  Vec3 C = particles[ac.particleIdx].position - worldAnchor;

  // Penalty force: f = k * C (pushes rigid toward particle)
  Vec3 fLin = C * (-ac.k);
  Vec3 fAng = worldOffset.cross(C * (-ac.k));

  for (int i = 0; i < 3; i++) rhs.v[i] += (&fLin.x)[i];
  for (int i = 0; i < 3; i++) rhs.v[3+i] += (&fAng.x)[i];

  // LHS: k * J^T J
  for (int i = 0; i < 3; i++)
    lhs.m[i][i] += ac.k;

  Mat33 sk;
  sk.m[0][0] = 0;              sk.m[0][1] = -worldOffset.z; sk.m[0][2] = worldOffset.y;
  sk.m[1][0] = worldOffset.z;  sk.m[1][1] = 0;              sk.m[1][2] = -worldOffset.x;
  sk.m[2][0] = -worldOffset.y; sk.m[2][1] = worldOffset.x;  sk.m[2][2] = 0;

  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      float val = 0;
      for (int kk = 0; kk < 3; kk++)
        val += sk.m[kk][r] * sk.m[kk][c];
      lhs.m[3+r][3+c] += ac.k * val;
    }

  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      float val = -ac.k * sk.m[c][r];
      lhs.m[r][3+c] += val;
      lhs.m[3+c][r] += val;
    }
}

// =============================================================================
// AVBD Dual updates — penalty growth only, no lambda
// =============================================================================

inline void updateAttachmentDual(AttachmentConstraint& ac,
                                 const std::vector<SoftParticle>& particles,
                                 const std::vector<Body>& rigidBodies,
                                 float beta) {
  const Body& rb = rigidBodies[ac.rigidBodyIdx];
  Vec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
  float C_lin = (particles[ac.particleIdx].position - worldAnchor).length();
  ac.k = std::min(ac.k + beta * C_lin, ac.kMax);
}

inline void updatePinDual(KinematicPin& kp,
                          const std::vector<SoftParticle>& particles,
                          float beta) {
  float C_lin = (particles[kp.particleIdx].position - kp.worldTarget).length();
  kp.k = std::min(kp.k + beta * C_lin, kp.kMax);
}

inline void updateSoftContactDual(SoftContact& sc,
                                  const std::vector<SoftParticle>& particles,
                                  float beta) {
  Vec3 n = sc.normal;
  float penetration;
  if (sc.rigidBodyIdx == UINT32_MAX) {
    penetration = -(particles[sc.particleIdx].position.dot(n));
  } else {
    penetration = sc.depth;
  }
  penetration = std::max(0.0f, penetration);
  sc.k = std::min(sc.k + beta * penetration, sc.ke);
}

// =============================================================================
// Collision detection (unchanged)
// =============================================================================

inline void detectSoftGroundContacts(const std::vector<SoftParticle>& particles,
                                     std::vector<SoftContact>& contacts,
                                     float groundY = 0.0f,
                                     float margin = 0.02f,
                                     float friction = 0.5f) {
  contacts.clear();
  Vec3 normal(0, 1, 0);
  Vec3 t1(1, 0, 0), t2(0, 0, 1);

  for (uint32_t i = 0; i < (uint32_t)particles.size(); i++) {
    if (particles[i].invMass <= 0.0f) continue;
    float dist = particles[i].position.y - groundY;
    if (dist < margin) {
      SoftContact sc;
      sc.particleIdx = i;
      sc.rigidBodyIdx = UINT32_MAX;
      sc.normal = normal;
      sc.depth = std::max(0.0f, -dist);
      sc.friction = friction;
      sc.tangent1 = t1;
      sc.tangent2 = t2;
      contacts.push_back(sc);
    }
  }
}

inline void detectSoftRigidContacts(const std::vector<SoftParticle>& particles,
                                    const std::vector<Body>& rigidBodies,
                                    std::vector<SoftContact>& contacts,
                                    float margin = 0.02f) {
  for (uint32_t pi = 0; pi < (uint32_t)particles.size(); pi++) {
    if (particles[pi].invMass <= 0.0f) continue;
    const Vec3& pp = particles[pi].position;

    for (uint32_t bi = 0; bi < (uint32_t)rigidBodies.size(); bi++) {
      const Body& rb = rigidBodies[bi];
      if (rb.halfExtent.x <= 0 && rb.halfExtent.y <= 0 && rb.halfExtent.z <= 0)
        continue;

      Vec3 localP = rb.rotation.conjugate().rotate(pp - rb.position);
      Vec3 he = rb.halfExtent;

      Vec3 closest;
      closest.x = std::max(-he.x, std::min(he.x, localP.x));
      closest.y = std::max(-he.y, std::min(he.y, localP.y));
      closest.z = std::max(-he.z, std::min(he.z, localP.z));

      Vec3 diff = localP - closest;
      float dist = diff.length();

      bool inside = (fabsf(localP.x) <= he.x &&
                     fabsf(localP.y) <= he.y &&
                     fabsf(localP.z) <= he.z);
      Vec3 normal;
      float depth;

      if (inside) {
        float dx = he.x - fabsf(localP.x);
        float dy = he.y - fabsf(localP.y);
        float dz = he.z - fabsf(localP.z);
        if (dx <= dy && dx <= dz)
          normal = Vec3(localP.x > 0 ? 1.0f : -1.0f, 0, 0);
        else if (dy <= dz)
          normal = Vec3(0, localP.y > 0 ? 1.0f : -1.0f, 0);
        else
          normal = Vec3(0, 0, localP.z > 0 ? 1.0f : -1.0f);
        depth = std::min(dx, std::min(dy, dz));
      } else {
        if (dist > margin) continue;
        normal = diff * (1.0f / std::max(dist, 1e-10f));
        depth = std::max(0.0f, margin - dist);
      }

      Vec3 worldNormal = rb.rotation.rotate(normal).normalized();

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

} // namespace AvbdRef
