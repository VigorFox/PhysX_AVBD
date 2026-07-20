// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "DyAvbdKinematicShell.h"
#include "DyAvbdConstraint.h"
#include "DyAvbdSolverBody.h"
#include "PxAvbdSoftBody.h"
#include "PxAvbdSoftBody.h"
#include <map>
#include <vector>

namespace physx {
namespace Dy {

bool AvbdKinematicShell::sActive = false;
bool AvbdKinematicShell::sBoxCornerShellEnabled = false;
PxU32 AvbdKinematicShell::sGridSize = 0;
PxReal AvbdKinematicShell::sGridStep = 0.0f;
PxVec3 AvbdKinematicShell::sWorld[AvbdKinematicShell::kMaxVerts];
PxVec3 AvbdKinematicShell::sPrevWorld[AvbdKinematicShell::kMaxVerts];

PxU32 AvbdKinematicShell::shellParticleCount() {
  if (!sActive)
    return 0;
  return sGridSize * sGridSize;
}

void AvbdKinematicShell::syncIslandSoftParticles(AvbdSoftParticle *out,
                                                 PxU32 capacity) {
  const PxU32 n = shellParticleCount();
  if (!out || capacity < n)
    return;
  for (PxU32 i = 0; i < n; ++i) {
    AvbdSoftParticle &sp = out[i];
    sp.position = sWorld[i];
    sp.predictedPosition = sWorld[i];
    sp.initialPosition = sPrevWorld[i];
    sp.outerPosition = sWorld[i];
    sp.invMass = 0.0f;
    sp.mass = 0.0f;
    sp.velocity = PxVec3(0.0f);
    sp.prevVelocity = PxVec3(0.0f);
    sp.elasticK = 0.0f;
    sp.elasticKMax = 1e6f;
    sp.damping = 0.0f;
  }
}

static AvbdSoftBody sKinematicShellSoftBody;
static bool sKinematicShellSoftBodyInit = false;

AvbdSoftBody &AvbdKinematicShell::kinematicShellSoftBody() {
  if (!sKinematicShellSoftBodyInit) {
    sKinematicShellSoftBody.particleStart = 0;
    sKinematicShellSoftBody.particleCount = kMaxVerts;
    sKinematicShellSoftBodyInit = true;
  }
  sKinematicShellSoftBody.particleCount = shellParticleCount();
  return sKinematicShellSoftBody;
}

static constexpr PxReal kMeshRestDepth = 0.5f;

static void initShellContactTangents(AvbdSoftContact &sc) {
  if (PxAbs(sc.normal.y) > 0.9f)
    sc.tangent1 = sc.normal.cross(PxVec3(1, 0, 0)).getNormalized();
  else
    sc.tangent1 = sc.normal.cross(PxVec3(0, 1, 0)).getNormalized();
  sc.tangent2 = sc.normal.cross(sc.tangent1);
}

static bool finalizeShellContact(AvbdSoftContact &sc, PxU32 bi,
                                 const AvbdSolverBody &body,
                                 const PxVec3 &worldContact,
                                 const PxVec3 &shellSurf, PxReal friction) {
  PxVec3 shellN = AvbdKinematicShell::outwardNormal(
      shellSurf.x, shellSurf.z, body.position);
  const PxReal sep = (worldContact - shellSurf).dot(shellN);
  if (sep > kMeshRestDepth + 0.12f)
    return false;

  sc.particleIdx =
      AvbdKinematicShell::nearestParticleIndex(shellSurf.x, shellSurf.z);
  sc.rigidBodyIdx = bi;
  sc.normal = shellN;
  sc.projNormal = shellN;
  sc.depth = kMeshRestDepth;
  sc.margin = 0.0f;
  sc.friction = friction;
  sc.surfacePoint = shellSurf;
  sc.surfacePointPrev =
      AvbdKinematicShell::bilinearSurfacePointPrev(shellSurf.x, shellSurf.z);
  sc.rigidLocalPoint =
      body.rotation.rotateInv(worldContact - body.position);
  sc.alLambda = 0.0f;
  sc.alLambdaTangent[0] = 0.0f;
  sc.alLambdaTangent[1] = 0.0f;
  sc.penTangent[0] = 1000.0f;
  sc.penTangent[1] = 1000.0f;
  sc.k = 1e4f;
  sc.ke = 1e6f;
  initShellContactTangents(sc);
  return true;
}

static bool appendBoxCornerShellContact(AvbdSoftContact &sc, PxU32 bi,
                                        const AvbdSolverBody &body,
                                        const PxVec3 &rigidLocal,
                                        PxReal friction) {
  const PxVec3 worldCorner = body.position + body.rotation.rotate(rigidLocal);
  const PxVec3 shellSurf = AvbdKinematicShell::bilinearSurfacePoint(
      worldCorner.x, worldCorner.z);
  return finalizeShellContact(sc, bi, body, worldCorner, shellSurf, friction);
}

static bool appendSphereSampleShellContact(AvbdSoftContact &sc, PxU32 bi,
                                           const AvbdSolverBody &body,
                                           PxReal radius, PxReal friction,
                                           PxReal sampleX, PxReal sampleZ) {
  const PxVec3 shellSurf =
      AvbdKinematicShell::bilinearSurfacePoint(sampleX, sampleZ);
  PxVec3 n = body.position - shellSurf;
  PxReal dist = n.magnitude();
  if (dist < 1e-4f)
    n = PxVec3(0.0f, 1.0f, 0.0f);
  else
    n *= (1.0f / dist);
  const PxVec3 worldA = body.position - n * radius;
  return finalizeShellContact(sc, bi, body, worldA, shellSurf, friction);
}

static void estimateBodyShapeFromDeformableContacts(
    const AvbdContactConstraint *constraints, PxU32 numConstraints,
    const AvbdSolverBody &body, PxU32 bi, PxU32 numBodies, PxVec3 &halfExt,
    PxReal &sphereRadius, PxReal &friction) {
  halfExt = PxVec3(2.0f);
  sphereRadius = 0.5f;
  friction = 0.5f;
  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdContactConstraint &c = constraints[i];
    if (!hasDeformableStaticAnchor(c))
      continue;
    const PxU32 bA = c.header.bodyIndexA;
    const PxU32 bB = c.header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies))
      continue;
    const PxU32 cbi = (bA < numBodies) ? bA : bB;
    if (cbi != bi)
      continue;
    friction = c.friction;
    const bool dynIsA = (bA == bi);
    const PxVec3 worldPt =
        dynIsA ? body.position + body.rotation.rotate(c.contactPointA)
               : body.position + body.rotation.rotate(c.contactPointB);
    const PxVec3 local = body.rotation.rotateInv(worldPt - body.position);
    halfExt.x = PxMax(halfExt.x, PxAbs(local.x));
    halfExt.y = PxMax(halfExt.y, PxAbs(local.y));
    halfExt.z = PxMax(halfExt.z, PxAbs(local.z));
    sphereRadius = PxMax(sphereRadius, (worldPt - body.position).magnitude());
  }
}

PxU32 AvbdKinematicShell::buildIslandShellContacts(
    const AvbdContactConstraint *constraints, PxU32 numConstraints,
    const AvbdSolverBody *bodies, PxU32 numBodies, AvbdSoftContact *out,
    PxU32 capacity, const PxU32 *deformAnchorCounts) {
  if (!isActive() || !bodies || !out || numBodies == 0)
    return 0;

  static constexpr PxReal kShellBoxRebuildMaxGap = 25.0f;
  const PxReal sampleStep = gridStep() * 0.35f;
  PxU32 count = 0;
  for (PxU32 bi = 0; bi < numBodies; ++bi) {
    const AvbdSolverBody &body = bodies[bi];
    if (body.invMass <= 0.0f)
      continue;
    const PxU32 deformCount =
        deformAnchorCounts ? deformAnchorCounts[bi] : 0u;
    if (!shouldEmitDominantIslandShell(body, deformCount))
      continue;

    PxVec3 halfExt(2.0f);
    PxReal sphereRadius = 3.0f;
    PxReal friction = 0.5f;
    if (constraints && numConstraints > 0) {
      estimateBodyShapeFromDeformableContacts(
          constraints, numConstraints, body, bi, numBodies, halfExt,
          sphereRadius, friction);
    }

    const PxReal mass = 1.0f / body.invMass;
    const bool sphereLike = mass < 5.0f;
    const PxReal verticalExtent = sphereLike ? sphereRadius : halfExt.y;
    const PxReal surfY =
        sampleSurfaceY(body.position.x, body.position.z);
    const PxReal bottomClearance = body.position.y - verticalExtent - surfY;
    const PxReal gapLimit = sphereLike ? 12.0f : 25.0f;
    if (bottomClearance > gapLimit)
      continue;

    if (sphereLike) {
      for (int di = -4; di <= 4 && count < capacity; ++di) {
        for (int dj = -4; dj <= 4 && count < capacity; ++dj) {
          const PxReal sx = body.position.x + PxReal(di) * sampleStep;
          const PxReal sz = body.position.z + PxReal(dj) * sampleStep;
          if (appendSphereSampleShellContact(out[count], bi, body, sphereRadius,
                                             friction, sx, sz))
            ++count;
        }
      }
    } else {
      const PxReal hx = halfExt.x;
      const PxReal hy = halfExt.y;
      const PxReal hz = halfExt.z;
      const PxVec3 corners[4] = {PxVec3(-hx, -hy, -hz), PxVec3(hx, -hy, -hz),
                                 PxVec3(hx, -hy, hz), PxVec3(-hx, -hy, hz)};
      for (PxU32 ci = 0; ci < 4u && count < capacity; ++ci) {
        if (appendBoxCornerShellContact(out[count], bi, body, corners[ci],
                                        friction))
          ++count;
      }
    }
  }
  return count;
}

static PxReal shellContactViolation(const AvbdSoftContact &sc,
                                  const AvbdSolverBody &body) {
  const PxVec3 rAw = body.rotation.rotate(sc.rigidLocalPoint);
  const PxVec3 wA = body.position + rAw;
  PxReal geom = (wA - sc.surfacePoint).dot(sc.normal) - sc.depth;
  if (geom < 0.0f)
    geom = PxMin(geom, -sc.depth);
  return geom;
}

PxU32 AvbdKinematicShell::compactDominantIslandShellContacts(
    AvbdSoftContact *contacts, PxU32 count, const AvbdSolverBody *bodies,
    PxU32 numBodies) {
  if (!contacts || count == 0 || !bodies)
    return count;

  std::vector<PxU32> bestIdx(numBodies, PX_MAX_U32);
  std::vector<PxReal> bestViol(numBodies, 1e30f);
  std::vector<bool> keep(count, false);

  for (PxU32 i = 0; i < count; ++i) {
    const AvbdSoftContact &sc = contacts[i];
    const PxU32 bi = sc.rigidBodyIdx;
    if (bi >= numBodies || bodies[bi].invMass <= 0.0f)
      continue;
    const PxReal viol = shellContactViolation(sc, bodies[bi]);
    if (viol < bestViol[bi]) {
      bestViol[bi] = viol;
      bestIdx[bi] = i;
    }
  }

  for (PxU32 bi = 0; bi < numBodies; ++bi) {
    if (bestIdx[bi] != PX_MAX_U32)
      keep[bestIdx[bi]] = true;
  }

  PxU32 out = 0;
  for (PxU32 i = 0; i < count; ++i) {
    if (!keep[i])
      continue;
    if (out != i)
      contacts[out] = contacts[i];
    ++out;
  }
  return out;
}

void AvbdKinematicShell::countDeformableAnchorsPerBody(
    const AvbdContactConstraint *constraints, PxU32 numConstraints,
    PxU32 numBodies, PxU32 *outCounts) {
  if (!outCounts || numBodies == 0)
    return;
  for (PxU32 bi = 0; bi < numBodies; ++bi)
    outCounts[bi] = 0;
  if (!constraints || numConstraints == 0)
    return;
  for (PxU32 i = 0; i < numConstraints; ++i) {
    const AvbdContactConstraint &c = constraints[i];
    if (!hasDeformableStaticAnchor(c))
      continue;
    const PxU32 bA = c.header.bodyIndexA;
    const PxU32 bB = c.header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies))
      continue;
    const PxU32 bi = (bA < numBodies) ? bA : bB;
    if (bi < numBodies)
      ++outCounts[bi];
  }
}

bool AvbdKinematicShell::shouldEmitDominantIslandShell(
    const AvbdSolverBody &body, PxU32 deformAnchorCount) {
  // Box-corner shell normals inject energy on heaving mesh stacks (spread/speed
  // blow-up vs NP-only Entry-122 path). Spheres and stress harness use NP rows;
  // shell grid sampling remains available behind AVBD_KINEMATIC_SHELL_BOX=1.
  if (body.invMass <= 0.0f)
    return false;
  if (!boxCornerShellEnabled())
    return false;
  static constexpr PxU32 kDominantSynthesizeMinContacts = 4u;
  const PxReal mass = 1.0f / body.invMass;
  if (mass < 5.0f)
    return false;
  return deformAnchorCount >= kDominantSynthesizeMinContacts;
}

PxU32 AvbdKinematicShell::stripDeformableAnchorContactsForBodies(
    AvbdContactConstraint *constraints, PxU32 count,
    const bool *replaceBodyMask, PxU32 numBodies) {
  if (!constraints || count == 0 || !replaceBodyMask)
    return count;
  PxU32 out = 0;
  for (PxU32 i = 0; i < count; ++i) {
    const AvbdContactConstraint &c = constraints[i];
    if (hasDeformableStaticAnchor(c)) {
      const PxU32 bA = c.header.bodyIndexA;
      const PxU32 bB = c.header.bodyIndexB;
      if (isBodyVsStaticContact(bA, bB, numBodies)) {
        const PxU32 bi = (bA < numBodies) ? bA : bB;
        if (bi < numBodies && replaceBodyMask[bi]) {
          continue;
        }
      }
    }
    if (out != i)
      constraints[out] = constraints[i];
    ++out;
  }
  return out;
}

namespace {
struct IslandShellCacheKey {
  PxU32 bodyIdx;
  PxU32 particleIdx;
  bool operator<(const IslandShellCacheKey &o) const {
    if (bodyIdx != o.bodyIdx)
      return bodyIdx < o.bodyIdx;
    return particleIdx < o.particleIdx;
  }
};
struct IslandShellCacheEntry {
  PxReal alLambda;
  PxReal k;
  PxReal alLambdaTangent[2];
  PxReal penTangent[2];
  PxVec3 surfacePointPrev;
};
static std::map<IslandShellCacheKey, IslandShellCacheEntry> sIslandShellCache;
} // namespace

void AvbdKinematicShell::restoreIslandShellContactCache(AvbdSoftContact *contacts,
                                                        PxU32 count) {
  // Entry 153: do NOT restore alLambda / k / penTangent across frames on a
  // heaving mesh. That dual warmstart (keyed only by body+particle index)
  // accumulates multi-minute "spring memory" ? sudden bounce / sink after
  // long SnippetDeformableMesh runs. Matches Entry 108 NP policy.
  // surfacePointPrev is always overwritten by refineShellSoftContactAnchor
  // from the published bilinear prev grid, so cache prev is unused.
  PX_UNUSED(contacts);
  PX_UNUSED(count);
  // Drop stale map so long sessions do not grow unbounded.
  sIslandShellCache.clear();
}

void AvbdKinematicShell::saveIslandShellContactCache(
    const AvbdSoftContact *contacts, PxU32 count) {
  // Dual state is not warmstarted (see restore). Keep no-op for call sites.
  PX_UNUSED(contacts);
  PX_UNUSED(count);
}

void AvbdKinematicShell::refineShellSoftContactAnchor(
    AvbdSoftContact &sc, const AvbdSolverBody &body) {
  if (!isActive())
    return;
  const PxVec3 shellSurf =
      bilinearSurfacePoint(body.position.x, body.position.z);
  const PxVec3 shellN =
      outwardNormal(body.position.x, body.position.z, body.position);
  const PxVec3 shellPrev =
      bilinearSurfacePointPrev(body.position.x, body.position.z);

  const PxVec3 worldA_np = body.position + body.rotation.rotate(sc.rigidLocalPoint);
  const PxReal radiusAlongN =
      PxMax(0.01f, -(worldA_np - body.position).dot(shellN));
  const PxVec3 worldA = body.position - shellN * radiusAlongN;

  sc.normal = shellN;
  sc.projNormal = shellN;
  sc.surfacePoint = shellSurf;
  sc.surfacePointPrev = shellPrev;
  sc.rigidLocalPoint = body.rotation.rotateInv(worldA - body.position);
  sc.depth = kMeshRestDepth;
  if (PxAbs(shellN.y) > 0.9f)
    sc.tangent1 = shellN.cross(PxVec3(1, 0, 0)).getNormalized();
  else
    sc.tangent1 = shellN.cross(PxVec3(0, 1, 0)).getNormalized();
  sc.tangent2 = shellN.cross(sc.tangent1);
}

bool AvbdKinematicShell::applyShellNormalAndPrev(
    AvbdContactConstraint &constraint, AvbdSolverBody *bodyA,
    AvbdSolverBody *bodyB, PxReal /*restDist*/, PxVec3 &tangent0,
    PxVec3 &tangent1) {
  if (!isActive())
    return false;

  AvbdSolverBody *dynBody = nullptr;
  if (bodyA && bodyA->invMass > 0.0f)
    dynBody = bodyA;
  else if (bodyB && bodyB->invMass > 0.0f)
    dynBody = bodyB;
  if (!dynBody)
    return false;

  const PxVec3 dynCenter = dynBody->position;
  const PxVec3 shellN =
      outwardNormal(dynCenter.x, dynCenter.z, dynCenter);
  // Do NOT rewrite NP contact points here: overriding static anchors caused
  // sphere launch / lateral explosions (Entry 129, headless 2026-07-13).
  constraint.contactNormal = shellN;
  // This remains an NP contact row, so pair its current static anchor with the
  // same point for the previous anchor.  Mixing the NP corner contact at t with
  // a bilinear body-centre sample at t-dt turns spatial slope into fictitious
  // surface velocity and can inflate the depenetration cap.  Direct shell rows
  // carry their own coherent surfacePoint/surfacePointPrev pair.
  constraint.staticPrevWorldPoint =
      bodyA == dynBody ? constraint.contactPointB : constraint.contactPointA;
  constraint.header.flags |= AvbdContactConstraintFlags::eKINEMATIC_SHELL_ANCHOR;

  if (PxAbs(shellN.y) > 0.9f)
    tangent0 = shellN.cross(PxVec3(1, 0, 0)).getNormalized();
  else
    tangent0 = shellN.cross(PxVec3(0, 1, 0)).getNormalized();
  tangent1 = shellN.cross(tangent0);
  constraint.tangent0 = tangent0;
  constraint.tangent1 = tangent1;
  (void)bodyA;
  (void)bodyB;
  return true;
}

} // namespace Dy
} // namespace physx
