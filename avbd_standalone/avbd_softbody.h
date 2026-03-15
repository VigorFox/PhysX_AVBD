#pragma once
// =============================================================================
// AVBD Soft Body / Cloth — Constraint-based deformable body system
//
// All soft body constraints (distance, volume, bending, attachment, pin) are
// formulated as Augmented Lagrangian rows in the same energy function as
// rigid body contacts and joints. Soft particles are 3-DOF (position only).
//
// Integration: In the primal sweep, each soft particle accumulates a 3×3 LHS
// and 3×1 RHS from all touching constraints, then solves a 3×3 system.
// Dual update follows the same pattern as rigid contacts/joints.
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

  SoftParticle()
      : mass(1.0f), invMass(1.0f), damping(0.0f) {}
};

// =============================================================================
// Constraint types within a SoftBody
// =============================================================================

struct DistanceConstraint {
  uint32_t p0, p1;     // global particle indices
  float restLength;
  float lambda;
  float rho;

  DistanceConstraint()
      : p0(0), p1(0), restLength(0), lambda(0), rho(1e4f) {}
};

struct VolumeConstraint {
  uint32_t p0, p1, p2, p3;  // tet vertex indices (global particle)
  float restVolume;          // signed volume (1/6 * det)
  float lambda;
  float rho;

  VolumeConstraint()
      : p0(0), p1(0), p2(0), p3(0), restVolume(0), lambda(0), rho(1e5f) {}
};

struct BendingConstraint {
  uint32_t p0, p1, p2, p3;  // shared edge (p0,p1), wing vertices (p2,p3)
  float restAngle;           // dihedral angle at rest
  float lambda;
  float rho;

  BendingConstraint()
      : p0(0), p1(0), p2(0), p3(0), restAngle(0), lambda(0), rho(1e3f) {}
};

struct AttachmentConstraint {
  uint32_t particleIdx;      // global soft particle index
  uint32_t rigidBodyIdx;     // index into solver.bodies[]
  Vec3 localOffset;          // attachment point in rigid body local frame
  Vec3 lambda;               // bilateral AL dual (3-DOF)
  float rho;

  AttachmentConstraint()
      : particleIdx(0), rigidBodyIdx(0), rho(1e5f) {}
};

struct KinematicPin {
  uint32_t particleIdx;      // global soft particle index
  Vec3 worldTarget;          // fixed world position
  Vec3 lambda;
  float rho;

  KinematicPin()
      : particleIdx(0), rho(1e6f) {}
};

// =============================================================================
// SoftContact — particle vs ground or rigid body
// =============================================================================
struct SoftContact {
  uint32_t particleIdx;     // global soft particle index
  uint32_t rigidBodyIdx;    // UINT32_MAX = ground
  Vec3 normal;
  float depth;

  float lambda;             // normal AL dual
  float rho;
  float friction;

  float lambdaT1, lambdaT2;
  Vec3 tangent1, tangent2;

  SoftContact()
      : particleIdx(0), rigidBodyIdx(UINT32_MAX),
        depth(0), lambda(0), rho(1e4f), friction(0.5f),
        lambdaT1(0), lambdaT2(0) {}
};

// =============================================================================
// SoftBody — mesh + constraints
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
  float thickness;         // for bending stiffness calculation

  // Constraints (built at setup)
  std::vector<DistanceConstraint> distConstraints;
  std::vector<VolumeConstraint> volConstraints;
  std::vector<BendingConstraint> bendConstraints;
  std::vector<AttachmentConstraint> attachments;
  std::vector<KinematicPin> pins;

  SoftBody()
      : particleStart(0), particleCount(0),
        youngsModulus(1e5f), poissonsRatio(0.3f),
        density(100.0f), damping(0.01f),
        bendingStiffness(0.0f), thickness(0.01f) {}

  // Build distance constraints from edges in triangles + tetrahedra
  void buildDistanceConstraints(const std::vector<SoftParticle>& particles) {
    // Collect unique edges
    struct Edge {
      uint32_t a, b;
      bool operator<(const Edge& o) const {
        if (a != o.a) return a < o.a;
        return b < o.b;
      }
    };
    std::vector<Edge> edges;
    auto addEdge = [&](uint32_t a, uint32_t b) {
      uint32_t ga = particleStart + a;
      uint32_t gb = particleStart + b;
      if (ga > gb) std::swap(ga, gb);
      edges.push_back({ga, gb});
    };

    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
      addEdge(triangles[i], triangles[i+1]);
      addEdge(triangles[i+1], triangles[i+2]);
      addEdge(triangles[i+2], triangles[i]);
    }
    for (size_t i = 0; i + 3 < tetrahedra.size(); i += 4) {
      uint32_t v0 = tetrahedra[i], v1 = tetrahedra[i+1];
      uint32_t v2 = tetrahedra[i+2], v3 = tetrahedra[i+3];
      addEdge(v0, v1); addEdge(v0, v2); addEdge(v0, v3);
      addEdge(v1, v2); addEdge(v1, v3); addEdge(v2, v3);
    }

    // Deduplicate
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end(),
                            [](const Edge& a, const Edge& b) {
                              return a.a == b.a && a.b == b.b;
                            }),
                edges.end());

    // Compute rest lengths and stiffness
    // rho_dist = E * A_eff / L0, with A_eff estimated from average edge length
    float avgLen = 0.0f;
    for (const auto& e : edges) {
      float L = (particles[e.a].position - particles[e.b].position).length();
      avgLen += L;
    }
    if (!edges.empty()) avgLen /= (float)edges.size();
    float Aeff = avgLen * avgLen * 0.1f; // rough cross-section estimate

    distConstraints.clear();
    distConstraints.reserve(edges.size());
    for (const auto& e : edges) {
      DistanceConstraint dc;
      dc.p0 = e.a;
      dc.p1 = e.b;
      dc.restLength = (particles[e.a].position - particles[e.b].position).length();
      dc.lambda = 0.0f;
      dc.rho = std::max(1.0f, youngsModulus * Aeff / std::max(dc.restLength, 1e-6f));
      distConstraints.push_back(dc);
    }
  }

  // Build volume constraints from tetrahedra
  void buildVolumeConstraints(const std::vector<SoftParticle>& particles) {
    volConstraints.clear();
    float bulkModulus = youngsModulus / (3.0f * (1.0f - 2.0f * poissonsRatio));
    bulkModulus = std::max(bulkModulus, 1e3f); // clamp for ν near 0.5

    for (size_t i = 0; i + 3 < tetrahedra.size(); i += 4) {
      VolumeConstraint vc;
      vc.p0 = particleStart + tetrahedra[i];
      vc.p1 = particleStart + tetrahedra[i+1];
      vc.p2 = particleStart + tetrahedra[i+2];
      vc.p3 = particleStart + tetrahedra[i+3];

      Vec3 e1 = particles[vc.p1].position - particles[vc.p0].position;
      Vec3 e2 = particles[vc.p2].position - particles[vc.p0].position;
      Vec3 e3 = particles[vc.p3].position - particles[vc.p0].position;
      float signedVol = e1.dot(e2.cross(e3)) / 6.0f;

      // Ensure positive winding: swap p1/p2 if negative
      if (signedVol < 0.0f) {
        std::swap(vc.p1, vc.p2);
        signedVol = -signedVol;
      }
      vc.restVolume = signedVol;
      vc.lambda = 0.0f;

      float V0abs = std::max(vc.restVolume, 1e-10f);
      vc.rho = bulkModulus / std::max(powf(V0abs, 1.0f / 3.0f), 1e-4f);
      volConstraints.push_back(vc);
    }
  }

  // Build bending constraints from triangle pairs sharing an edge
  void buildBendingConstraints(const std::vector<SoftParticle>& particles) {
    bendConstraints.clear();
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

    // Bending stiffness: rho_bend = k_bend * E * t^3 / (12*(1-ν²) * A_avg)
    float avgTriArea = 0.0f;
    uint32_t numTris = (uint32_t)(triangles.size() / 3);
    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
      Vec3 p0 = particles[particleStart + triangles[i]].position;
      Vec3 p1 = particles[particleStart + triangles[i+1]].position;
      Vec3 p2 = particles[particleStart + triangles[i+2]].position;
      avgTriArea += (p1 - p0).cross(p2 - p0).length() * 0.5f;
    }
    if (numTris > 0) avgTriArea /= (float)numTris;
    float bendRho = bendingStiffness * youngsModulus * thickness * thickness * thickness
                    / (12.0f * (1.0f - poissonsRatio * poissonsRatio) * std::max(avgTriArea, 1e-8f));
    bendRho = std::max(bendRho, 1.0f);

    for (auto& [ek, tris] : edgeToTris) {
      if (tris.size() != 2) continue; // boundary edge or non-manifold

      BendingConstraint bc;
      bc.p0 = ek.a;
      bc.p1 = ek.b;
      bc.p2 = tris[0].opposite;
      bc.p3 = tris[1].opposite;
      bc.lambda = 0.0f;
      bc.rho = bendRho;

      // Compute rest dihedral angle
      bc.restAngle = computeDihedralAngle(
          particles[bc.p0].position, particles[bc.p1].position,
          particles[bc.p2].position, particles[bc.p3].position);
      bendConstraints.push_back(bc);
    }
  }

  // Compute all constraints from mesh topology
  void buildConstraints(const std::vector<SoftParticle>& particles) {
    buildDistanceConstraints(particles);
    buildVolumeConstraints(particles);
    buildBendingConstraints(particles);
  }

  // =========================================================================
  // Dihedral angle computation
  // =========================================================================
  static float computeDihedralAngle(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3) {
    Vec3 edge = p1 - p0;
    float edgeLen = edge.length();
    if (edgeLen < 1e-10f) return 0.0f;
    Vec3 edgeN = edge * (1.0f / edgeLen);

    Vec3 n1 = (p2 - p0).cross(edge);
    Vec3 n2 = (p3 - p0).cross(edge);
    float n1Len = n1.length();
    float n2Len = n2.length();
    if (n1Len < 1e-10f || n2Len < 1e-10f) return 0.0f;
    n1 = n1 * (1.0f / n1Len);
    n2 = n2 * (1.0f / n2Len);

    float cosA = std::max(-1.0f, std::min(1.0f, n1.dot(n2)));
    float sinA = n1.cross(n2).dot(edgeN);
    return atan2f(sinA, cosA);
  }
};

// =============================================================================
// Mesh generation utilities
// =============================================================================

// Generate a cube tet mesh: 8 vertices, 5 tetrahedra
// Returns local-indexed vertices and tets
inline void generateCubeTets(Vec3 center, float halfSize,
                             std::vector<Vec3>& outVerts,
                             std::vector<uint32_t>& outTets) {
  float h = halfSize;
  outVerts = {
    center + Vec3(-h, -h, -h), // 0
    center + Vec3( h, -h, -h), // 1
    center + Vec3( h,  h, -h), // 2
    center + Vec3(-h,  h, -h), // 3
    center + Vec3(-h, -h,  h), // 4
    center + Vec3( h, -h,  h), // 5
    center + Vec3( h,  h,  h), // 6
    center + Vec3(-h,  h,  h), // 7
  };
  // 5-tet decomposition of a cube (consistent diagonal)
  outTets = {
    0, 1, 3, 4,
    1, 2, 3, 6,
    3, 4, 6, 7,
    1, 4, 5, 6,
    1, 3, 4, 6,
  };
}

// Generate a subdivided cube: (N+1)^3 vertices, 5*N^3 tetrahedra
inline void generateSubdividedCubeTets(Vec3 center, float halfSize, int N,
                                       std::vector<Vec3>& outVerts,
                                       std::vector<uint32_t>& outTets) {
  outVerts.clear();
  outTets.clear();
  float cellSize = 2.0f * halfSize / (float)N;
  Vec3 origin = center - Vec3(halfSize, halfSize, halfSize);

  // Vertices: (N+1)^3
  for (int iz = 0; iz <= N; iz++)
    for (int iy = 0; iy <= N; iy++)
      for (int ix = 0; ix <= N; ix++)
        outVerts.push_back(origin + Vec3(ix * cellSize, iy * cellSize, iz * cellSize));

  auto idx = [&](int ix, int iy, int iz) -> uint32_t {
    return (uint32_t)(iz * (N+1) * (N+1) + iy * (N+1) + ix);
  };

  // 5 tets per cube cell
  for (int iz = 0; iz < N; iz++)
    for (int iy = 0; iy < N; iy++)
      for (int ix = 0; ix < N; ix++) {
        uint32_t v[8] = {
          idx(ix,   iy,   iz),   // 0
          idx(ix+1, iy,   iz),   // 1
          idx(ix+1, iy+1, iz),   // 2
          idx(ix,   iy+1, iz),   // 3
          idx(ix,   iy,   iz+1), // 4
          idx(ix+1, iy,   iz+1), // 5
          idx(ix+1, iy+1, iz+1), // 6
          idx(ix,   iy+1, iz+1), // 7
        };
        outTets.insert(outTets.end(), {v[0], v[1], v[3], v[4]});
        outTets.insert(outTets.end(), {v[1], v[2], v[3], v[6]});
        outTets.insert(outTets.end(), {v[3], v[4], v[6], v[7]});
        outTets.insert(outTets.end(), {v[1], v[4], v[5], v[6]});
        outTets.insert(outTets.end(), {v[1], v[3], v[4], v[6]});
      }
}

// Generate a cloth grid: M×N vertices, 2*(M-1)*(N-1) triangles
// Grid lies in XZ plane at height y
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
// Constraint math — primal contributions (add to 3×3 LHS, 3×1 RHS)
// =============================================================================

// Distance constraint: C = ||p1 - p0|| - L0
inline void addDistanceContribution(const DistanceConstraint& dc,
                                    uint32_t pi,
                                    const std::vector<SoftParticle>& particles,
                                    Mat33& lhs, Vec3& rhs) {
  Vec3 diff = particles[dc.p1].position - particles[dc.p0].position;
  float dist = diff.length();
  if (dist < 1e-10f) return;
  Vec3 n = diff * (1.0f / dist);

  float C = dist - dc.restLength;
  float f = dc.rho * C + dc.lambda;

  Vec3 J = (pi == dc.p0) ? (n * -1.0f) : n;

  // LHS += rho * J^T J (outer product of 3-vectors)
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      lhs.m[r][c] += dc.rho * (&J.x)[r] * (&J.x)[c];

  // RHS += J * f
  rhs = rhs + J * f;
}

// Volume constraint: C = V - V0
// V = 1/6 * (p1-p0) . ((p2-p0) × (p3-p0))
inline void addVolumeContribution(const VolumeConstraint& vc,
                                  uint32_t pi,
                                  const std::vector<SoftParticle>& particles,
                                  Mat33& lhs, Vec3& rhs) {
  Vec3 e1 = particles[vc.p1].position - particles[vc.p0].position;
  Vec3 e2 = particles[vc.p2].position - particles[vc.p0].position;
  Vec3 e3 = particles[vc.p3].position - particles[vc.p0].position;

  float V = e1.dot(e2.cross(e3)) / 6.0f;
  float C = V - vc.restVolume;
  float f = vc.rho * C + vc.lambda;

  // Jacobians: dV/dp_i = 1/6 * cross products
  Vec3 J;
  if (pi == vc.p1)      J = e2.cross(e3) * (1.0f / 6.0f);
  else if (pi == vc.p2) J = e3.cross(e1) * (1.0f / 6.0f);
  else if (pi == vc.p3) J = e1.cross(e2) * (1.0f / 6.0f);
  else if (pi == vc.p0) {
    Vec3 J1 = e2.cross(e3) * (1.0f / 6.0f);
    Vec3 J2 = e3.cross(e1) * (1.0f / 6.0f);
    Vec3 J3 = e1.cross(e2) * (1.0f / 6.0f);
    J = (J1 + J2 + J3) * -1.0f;
  }
  else return;

  float Jnorm = J.length();
  if (Jnorm < 1e-12f) return;

  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      lhs.m[r][c] += vc.rho * (&J.x)[r] * (&J.x)[c];

  rhs = rhs + J * f;
}

// Bending constraint: C = theta - theta0
// Uses finite-difference Jacobian for simplicity
inline void addBendingContribution(const BendingConstraint& bc,
                                   uint32_t pi,
                                   const std::vector<SoftParticle>& particles,
                                   Mat33& lhs, Vec3& rhs) {
  float theta = SoftBody::computeDihedralAngle(
      particles[bc.p0].position, particles[bc.p1].position,
      particles[bc.p2].position, particles[bc.p3].position);
  float C = theta - bc.restAngle;
  // Wrap to [-π, π]
  while (C > 3.14159265f) C -= 2.0f * 3.14159265f;
  while (C < -3.14159265f) C += 2.0f * 3.14159265f;

  float f = bc.rho * C + bc.lambda;

  // Finite-difference Jacobian (∂θ/∂p_i)
  const float eps = 1e-4f;
  Vec3 J;
  // We need mutable copies for FD perturbation
  Vec3 pos[4] = {
    particles[bc.p0].position,
    particles[bc.p1].position,
    particles[bc.p2].position,
    particles[bc.p3].position
  };
  int whichVert = -1;
  if (pi == bc.p0) whichVert = 0;
  else if (pi == bc.p1) whichVert = 1;
  else if (pi == bc.p2) whichVert = 2;
  else if (pi == bc.p3) whichVert = 3;
  else return;

  for (int axis = 0; axis < 3; axis++) {
    Vec3 posSave = pos[whichVert];
    (&pos[whichVert].x)[axis] += eps;
    float thetaP = SoftBody::computeDihedralAngle(pos[0], pos[1], pos[2], pos[3]);
    (&pos[whichVert].x)[axis] -= 2.0f * eps;
    float thetaM = SoftBody::computeDihedralAngle(pos[0], pos[1], pos[2], pos[3]);
    pos[whichVert] = posSave;
    // Wrap difference to [-π, π] to avoid atan2 discontinuity at ±π
    float dTheta = thetaP - thetaM;
    while (dTheta > 3.14159265f) dTheta -= 2.0f * 3.14159265f;
    while (dTheta < -3.14159265f) dTheta += 2.0f * 3.14159265f;
    (&J.x)[axis] = dTheta / (2.0f * eps);
  }

  float Jnorm = J.length();
  if (Jnorm < 1e-12f || std::isnan(Jnorm) || std::isinf(Jnorm)) return;
  // Clamp Jacobian norm to prevent instability from degenerate geometry
  const float maxJnorm = 50.0f;
  if (Jnorm > maxJnorm) {
    J = J * (maxJnorm / Jnorm);
  }

  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      lhs.m[r][c] += bc.rho * (&J.x)[r] * (&J.x)[c];

  rhs = rhs + J * f;
}

// Attachment: C = p_soft - (x_rigid + R_rigid * r_local)
inline void addAttachmentContribution_particle(
    const AttachmentConstraint& ac,
    uint32_t pi,
    const std::vector<SoftParticle>& particles,
    const std::vector<Body>& rigidBodies,
    Mat33& lhs, Vec3& rhs) {
  if (pi != ac.particleIdx) return;

  const Body& rb = rigidBodies[ac.rigidBodyIdx];
  Vec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
  Vec3 C = particles[pi].position - worldAnchor;
  Vec3 f = C * ac.rho + ac.lambda;

  // J_particle = I (identity), so LHS += rho * I, RHS += f
  for (int i = 0; i < 3; i++)
    lhs.m[i][i] += ac.rho;
  rhs = rhs + f;
}

// Attachment contribution to rigid body (6×6 side)
inline void addAttachmentContribution_rigid(
    const AttachmentConstraint& ac,
    uint32_t bodyIdx,
    const std::vector<SoftParticle>& particles,
    const std::vector<Body>& rigidBodies,
    float dt, Mat66& lhs, Vec6& rhs) {
  if (bodyIdx != ac.rigidBodyIdx) return;

  const Body& rb = rigidBodies[bodyIdx];
  Vec3 worldOffset = rb.rotation.rotate(ac.localOffset);
  Vec3 worldAnchor = rb.position + worldOffset;
  Vec3 C = particles[ac.particleIdx].position - worldAnchor;
  Vec3 f = C * ac.rho + ac.lambda;

  // J_rigid_lin = -I, J_rigid_ang = [r_world]×
  // Force on rigid = -f (linear), worldOffset × (-f) (angular)
  Vec3 fLin = f * -1.0f;
  Vec3 fAng = worldOffset.cross(f * -1.0f);

  for (int i = 0; i < 3; i++) rhs.v[i] += (&fLin.x)[i];
  for (int i = 0; i < 3; i++) rhs.v[3+i] += (&fAng.x)[i];

  // LHS contribution: rho * J^T J
  // Linear-linear: rho * I
  for (int i = 0; i < 3; i++)
    lhs.m[i][i] += ac.rho;

  // Linear-angular and angular-angular: cross product terms
  // [r]× = skew(worldOffset)
  Mat33 skew;
  skew.m[0][0] = 0;              skew.m[0][1] = -worldOffset.z; skew.m[0][2] = worldOffset.y;
  skew.m[1][0] = worldOffset.z;  skew.m[1][1] = 0;              skew.m[1][2] = -worldOffset.x;
  skew.m[2][0] = -worldOffset.y; skew.m[2][1] = worldOffset.x;  skew.m[2][2] = 0;

  // ang-ang: rho * skew^T * skew
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      float val = 0;
      for (int k = 0; k < 3; k++)
        val += skew.m[k][r] * skew.m[k][c]; // skew^T * skew
      lhs.m[3+r][3+c] += ac.rho * val;
    }

  // lin-ang coupling: -rho * skew^T
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      float val = -ac.rho * skew.m[c][r]; // -skew^T
      lhs.m[r][3+c] += val;
      lhs.m[3+c][r] += val;
    }
}

// Kinematic pin: C = p_soft - p_target
inline void addPinContribution(const KinematicPin& kp,
                               uint32_t pi,
                               const std::vector<SoftParticle>& particles,
                               Mat33& lhs, Vec3& rhs) {
  if (pi != kp.particleIdx) return;

  Vec3 C = particles[pi].position - kp.worldTarget;
  Vec3 f = C * kp.rho + kp.lambda;

  for (int i = 0; i < 3; i++)
    lhs.m[i][i] += kp.rho;
  rhs = rhs + f;
}

// Soft contact: C_n = n · (p - surface_point), unilateral
inline void addSoftContactContribution(const SoftContact& sc,
                                       uint32_t pi,
                                       const std::vector<SoftParticle>& particles,
                                       Mat33& lhs, Vec3& rhs) {
  if (pi != sc.particleIdx) return;

  Vec3 n = sc.normal;

  // Contact constraint: C = depth >= 0 (positive when penetrating)
  // For ground: C = groundY - p·n = -p.y
  // Jacobian: dC/dp = -n
  float C_contact;
  if (sc.rigidBodyIdx == UINT32_MAX) {
    // Ground: depth = -(p·n - groundY) = -p.y
    C_contact = -(particles[pi].position.dot(n));
  } else {
    C_contact = sc.depth;
  }

  // Unilateral: f >= 0 (only push, no pull)
  float f_contact = std::max(0.0f, sc.rho * C_contact + sc.lambda);

  // LHS: rho * J * J^T = rho * (-n)(-n)^T = rho * n*n^T
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      lhs.m[r][c] += sc.rho * (&n.x)[r] * (&n.x)[c];

  // RHS: J * f = (-n) * f
  rhs = rhs - n * f_contact;

  // Friction (Coulomb cone)
  if (sc.friction > 0.0f && f_contact > 0.0f) {
    float fmax_t = sc.friction * f_contact;

    // Tangent 1: C_t1 = (p - p_pred)·t1, dC/dp = t1
    float C_t1 = (particles[pi].position - particles[pi].predictedPosition).dot(sc.tangent1);
    float f_t1 = std::max(-fmax_t, std::min(fmax_t, sc.rho * C_t1 + sc.lambdaT1));
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        lhs.m[r][c] += sc.rho * (&sc.tangent1.x)[r] * (&sc.tangent1.x)[c];
    rhs = rhs + sc.tangent1 * f_t1;

    // Tangent 2: C_t2 = (p - p_pred)·t2, dC/dp = t2
    float C_t2 = (particles[pi].position - particles[pi].predictedPosition).dot(sc.tangent2);
    float f_t2 = std::max(-fmax_t, std::min(fmax_t, sc.rho * C_t2 + sc.lambdaT2));
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        lhs.m[r][c] += sc.rho * (&sc.tangent2.x)[r] * (&sc.tangent2.x)[c];
    rhs = rhs + sc.tangent2 * f_t2;
  }
}

// =============================================================================
// Dual updates
// =============================================================================

inline void updateDistanceDual(DistanceConstraint& dc,
                               const std::vector<SoftParticle>& particles,
                               float beta, float lambdaDecay) {
  Vec3 diff = particles[dc.p1].position - particles[dc.p0].position;
  float dist = diff.length();
  float C = dist - dc.restLength;

  dc.lambda = dc.lambda * lambdaDecay + dc.rho * C;
  // bilateral → no clamping
  if (fabsf(C) > 1e-5f)
    dc.rho = std::min(dc.rho + beta * fabsf(C), 1e4f);
}

inline void updateVolumeDual(VolumeConstraint& vc,
                             const std::vector<SoftParticle>& particles,
                             float beta, float lambdaDecay) {
  Vec3 e1 = particles[vc.p1].position - particles[vc.p0].position;
  Vec3 e2 = particles[vc.p2].position - particles[vc.p0].position;
  Vec3 e3 = particles[vc.p3].position - particles[vc.p0].position;
  float V = e1.dot(e2.cross(e3)) / 6.0f;
  float C = V - vc.restVolume;

  vc.lambda = vc.lambda * lambdaDecay + vc.rho * C;
  if (fabsf(C) > 1e-6f)
    vc.rho = std::min(vc.rho + beta * fabsf(C), 1e9f);
}

inline void updateBendingDual(BendingConstraint& bc,
                              const std::vector<SoftParticle>& particles,
                              float beta, float lambdaDecay) {
  float theta = SoftBody::computeDihedralAngle(
      particles[bc.p0].position, particles[bc.p1].position,
      particles[bc.p2].position, particles[bc.p3].position);
  float C = theta - bc.restAngle;
  while (C > 3.14159265f) C -= 2.0f * 3.14159265f;
  while (C < -3.14159265f) C += 2.0f * 3.14159265f;

  bc.lambda = bc.lambda * lambdaDecay + bc.rho * C;
  if (fabsf(C) > 1e-4f)
    bc.rho = std::min(bc.rho + beta * fabsf(C), 1e4f);
}

inline void updateAttachmentDual(AttachmentConstraint& ac,
                                 const std::vector<SoftParticle>& particles,
                                 const std::vector<Body>& rigidBodies,
                                 float lambdaDecay) {
  const Body& rb = rigidBodies[ac.rigidBodyIdx];
  Vec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
  Vec3 C = particles[ac.particleIdx].position - worldAnchor;
  ac.lambda = ac.lambda * lambdaDecay + C * ac.rho;
}

inline void updatePinDual(KinematicPin& kp,
                          const std::vector<SoftParticle>& particles,
                          float lambdaDecay) {
  Vec3 C = particles[kp.particleIdx].position - kp.worldTarget;
  kp.lambda = kp.lambda * lambdaDecay + C * kp.rho;
}

inline void updateSoftContactDual(SoftContact& sc,
                                  const std::vector<SoftParticle>& particles,
                                  float beta) {
  Vec3 n = sc.normal;
  float C_n;
  if (sc.rigidBodyIdx == UINT32_MAX) {
    C_n = -(particles[sc.particleIdx].position.dot(n)); // depth = -p.y for ground
  } else {
    C_n = sc.depth;
  }
  // Unilateral: lambda >= 0
  sc.lambda = std::max(0.0f, sc.lambda + sc.rho * C_n);
  if (C_n > 0.0f)
    sc.rho = std::min(sc.rho + beta * C_n, 1e9f);

  // Friction tangent duals
  if (sc.friction > 0.0f) {
    float fmax_t = sc.friction * sc.lambda;
    float C_t1 = particles[sc.particleIdx].position.dot(sc.tangent1)
                 - particles[sc.particleIdx].predictedPosition.dot(sc.tangent1);
    sc.lambdaT1 = std::max(-fmax_t, std::min(fmax_t, sc.lambdaT1 + sc.rho * C_t1));
    float C_t2 = particles[sc.particleIdx].position.dot(sc.tangent2)
                 - particles[sc.particleIdx].predictedPosition.dot(sc.tangent2);
    sc.lambdaT2 = std::max(-fmax_t, std::min(fmax_t, sc.lambdaT2 + sc.rho * C_t2));
  }
}

// =============================================================================
// Soft-particle ↔ ground collision detection
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

// =============================================================================
// Soft-particle ↔ rigid box collision detection
// =============================================================================
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

      // Transform particle into rigid body local frame
      Vec3 localP = rb.rotation.conjugate().rotate(pp - rb.position);
      Vec3 he = rb.halfExtent;

      // Closest point on box surface in local frame
      Vec3 closest;
      closest.x = std::max(-he.x, std::min(he.x, localP.x));
      closest.y = std::max(-he.y, std::min(he.y, localP.y));
      closest.z = std::max(-he.z, std::min(he.z, localP.z));

      Vec3 diff = localP - closest;
      float dist = diff.length();

      // If inside the box, push to nearest face
      bool inside = (fabsf(localP.x) <= he.x &&
                     fabsf(localP.y) <= he.y &&
                     fabsf(localP.z) <= he.z);
      Vec3 normal;
      float depth;

      if (inside) {
        // Find nearest face
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
        depth = -dist; // negative outside, flipped for convention
        if (depth < -margin) continue;
        depth = std::max(0.0f, margin - dist);
      }

      // Transform normal back to world
      Vec3 worldNormal = rb.rotation.rotate(normal).normalized();

      // Build tangent basis
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
      sc.friction = sqrtf(particles[pi].damping * rb.friction); // approximate
      sc.tangent1 = t1;
      sc.tangent2 = t2;
      contacts.push_back(sc);
    }
  }
}

} // namespace AvbdRef
