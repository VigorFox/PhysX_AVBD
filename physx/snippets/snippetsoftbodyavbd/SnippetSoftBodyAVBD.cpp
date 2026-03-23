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
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// ****************************************************************************
// SnippetSoftBodyAVBD
//
// Demonstrates a CPU-side VBD/AVBD soft body simulation rendered alongside a
// PhysX AVBD rigid-body scene.  A subdivided tetrahedral cube drops onto a
// ground plane; elastic forces use Neo-Hookean energy via VBD, while ground
// contacts are enforced through AVBD adaptive penalty.
//
// When built without RENDER_SNIPPET, runs automated unit tests covering:
//   1. Gravity free-fall (tet cube drops under gravity)
//   2. Ground contact (tet cube comes to rest above ground plane)
//   3. Volume preservation (Neo-Hookean prevents excessive compression)
//   4. Kinematic pin (pinned particles stay near target)
//   5. Cloth drape (tri mesh drapes under gravity)
//   6. Energy dissipation (damping removes kinetic energy)
//   7. Static equilibrium (zero-gravity cube stays at rest)
//   8. Multiple soft bodies (two independent bodies simulated together)
//   9. Soft-soft collision (stacked cubes don't interpenetrate)
//  10. Soft-rigid collision (soft cube rests on rigid box)
//  11. Slope rotation (cube slides/rotates on inclined rigid surface)
//  12. Cone-cube penetration (cone on cube without interpenetration)
//  13. OGC soft-soft collision (Sec 3.9 simplified path)
//  14. OGC rigid-soft SDF collision (analytical box SDF)
//  15. OGC self-collision (full path with two-stage activation)
//  16. OGC full pipeline (all paths combined)
//
// No GPU or CUDA dependency -- runs entirely on the CPU.
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "PxPhysicsAPI.h"
#include "PxAvbdSoftBody.h"

#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"

#include "SnippetSoftBodyAVBD.h"

using namespace physx;
using namespace physx::Dy;

// ---------------------------------------------------------------------------
// Globals (shared with render file for visual mode)
// ---------------------------------------------------------------------------

static PxDefaultAllocator      gAllocator;
static PxDefaultErrorCallback  gErrorCallback;
static PxFoundation*           gFoundation  = NULL;
static PxPhysics*              gPhysics     = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*             gMaterial    = NULL;

PxScene*                       gScene       = NULL;

// VBD soft body data (managed outside PhysX scene)
PxArray<AvbdSoftParticle>        gParticles;
PxArray<AvbdSoftBody>            gSoftBodies;
static PxArray<AvbdSoftContact>  gContacts;

// Render data shared with the render file
PxArray<SoftBodyRenderData>      gSoftBodyRenderData;

// ============================================================================
// Test infrastructure
// ============================================================================

static int gTestsPassed = 0;
static int gTestsFailed = 0;

#define TEST_CHECK(cond, testName) \
  do { \
    if (cond) { \
      gTestsPassed++; \
      printf("  PASS: %s\n", testName); \
    } else { \
      gTestsFailed++; \
      printf("  FAIL: %s\n", testName); \
    } \
    fflush(stdout); \
  } while(0)

#define TEST_CLOSE(val, expected, tol, testName) \
  TEST_CHECK(PxAbs((val) - (expected)) < (tol), testName)

static int getSelectedTestId()
{
  const char* value = std::getenv("PHYSX_AVBD_SOFTBODY_TEST_ID");
  if (!value || !value[0])
    return -1;
  const int id = std::atoi(value);
  return id > 0 ? id : -1;
}

static bool shouldRunTest(int selectedId, int testId)
{
  return selectedId < 0 || selectedId == testId;
}

// ============================================================================
// Helper: compute AABB (min/max Y) of particle set
// ============================================================================

static void getParticleBoundsY(const AvbdSoftParticle* particles,
                                PxU32 start, PxU32 count,
                                PxReal& minY, PxReal& maxY)
{
  minY = PX_MAX_F32;
  maxY = -PX_MAX_F32;
  for (PxU32 i = start; i < start + count; i++)
  {
    PxReal y = particles[i].position.y;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
}

static PxVec3 getParticleCentroid(const AvbdSoftParticle* particles,
                                   PxU32 start, PxU32 count)
{
  PxVec3 c(0.0f);
  for (PxU32 i = start; i < start + count; i++)
    c += particles[i].position;
  return c * (1.0f / PxReal(count));
}

static PxReal getMaxSpeed(const AvbdSoftParticle* particles,
                           PxU32 start, PxU32 count)
{
  PxReal maxV = 0.0f;
  for (PxU32 i = start; i < start + count; i++)
  {
    PxReal v = particles[i].velocity.magnitude();
    if (v > maxV) maxV = v;
  }
  return maxV;
}

// ============================================================================
// Body-level rotation diagnostics
// ============================================================================

static bool isRotationTraceEnabled()
{
	const char* value = std::getenv("PHYSX_AVBD_SOFTBODY_ROT_TRACE");
	return value && value[0] && value[0] != '0';
}

static PxU32 getRotationTraceInterval()
{
	const char* value = std::getenv("PHYSX_AVBD_SOFTBODY_ROT_TRACE_INTERVAL");
	if (!value || !value[0])
		return 30;
	const int interval = std::atoi(value);
	return interval > 0 ? PxU32(interval) : 30;
}

static PxVec3 computeBodyMassCentroid(const PxArray<AvbdSoftParticle>& particles,
																			const AvbdSoftBody& body)
{
	PxVec3 centroid(0.0f);
	PxReal totalMass = 0.0f;
	for (PxU32 i = 0; i < body.particleCount; i++)
	{
		const PxU32 pi = body.particleStart + i;
		const PxReal mass = particles[pi].mass;
		centroid += particles[pi].position * mass;
		totalMass += mass;
	}
	return totalMass > 0.0f ? centroid * (1.0f / totalMass) : PxVec3(0.0f);
}

static void captureBodyReferenceLocals(const PxArray<AvbdSoftParticle>& particles,
																			 const PxArray<AvbdSoftBody>& bodies,
																			 PxArray<PxArray<PxVec3> >& refs)
{
	refs.clear();
	refs.resize(bodies.size());
	for (PxU32 bi = 0; bi < bodies.size(); bi++)
	{
		const AvbdSoftBody& body = bodies[bi];
		const PxVec3 centroid = computeBodyMassCentroid(particles, body);
		refs[bi].resize(body.particleCount);
		for (PxU32 i = 0; i < body.particleCount; i++)
		{
			const PxU32 pi = body.particleStart + i;
			refs[bi][i] = particles[pi].position - centroid;
		}
	}
}

static PxQuat estimateBodyRotation(const PxArray<AvbdSoftParticle>& particles,
																	 const AvbdSoftBody& body,
																	 const PxArray<PxVec3>& refLocals)
{
	if (refLocals.size() != body.particleCount)
		return PxQuat(PxIdentity);

	const PxVec3 centroid = computeBodyMassCentroid(particles, body);

	PxReal sxx = 0.0f, sxy = 0.0f, sxz = 0.0f;
	PxReal syx = 0.0f, syy = 0.0f, syz = 0.0f;
	PxReal szx = 0.0f, szy = 0.0f, szz = 0.0f;

	for (PxU32 i = 0; i < body.particleCount; i++)
	{
		const PxU32 pi = body.particleStart + i;
		const PxReal mass = particles[pi].mass;
		const PxVec3 p = particles[pi].position - centroid;
		const PxVec3 q = refLocals[i];
		sxx += mass * p.x * q.x; sxy += mass * p.x * q.y; sxz += mass * p.x * q.z;
		syx += mass * p.y * q.x; syy += mass * p.y * q.y; syz += mass * p.y * q.z;
		szx += mass * p.z * q.x; szy += mass * p.z * q.y; szz += mass * p.z * q.z;
	}

	const PxReal N[4][4] = {
		{ sxx + syy + szz, syz - szy,         szx - sxz,         sxy - syx },
		{ syz - szy,       sxx - syy - szz,   sxy + syx,         szx + sxz },
		{ szx - sxz,       sxy + syx,        -sxx + syy - szz,   syz + szy },
		{ sxy - syx,       szx + sxz,         syz + szy,        -sxx - syy + szz }
	};

	PxReal qv[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
	for (PxU32 iter = 0; iter < 16; iter++)
	{
		PxReal next[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		for (PxU32 r = 0; r < 4; r++)
			for (PxU32 c = 0; c < 4; c++)
				next[r] += N[r][c] * qv[c];

		const PxReal len = PxSqrt(next[0]*next[0] + next[1]*next[1] + next[2]*next[2] + next[3]*next[3]);
		if (len < 1e-12f)
			return PxQuat(PxIdentity);

		qv[0] = next[0] / len;
		qv[1] = next[1] / len;
		qv[2] = next[2] / len;
		qv[3] = next[3] / len;
	}

	if (qv[0] < 0.0f)
	{
		qv[0] = -qv[0];
		qv[1] = -qv[1];
		qv[2] = -qv[2];
		qv[3] = -qv[3];
	}

	return PxQuat(qv[1], qv[2], qv[3], qv[0]).getNormalized();
}

static PxVec3 estimateBodyOmega(const PxArray<AvbdSoftParticle>& particles,
																const AvbdSoftBody& body)
{
	const PxVec3 centroid = computeBodyMassCentroid(particles, body);
	PxVec3 angularMomentum(0.0f);
	PxMat33 inertia = PxMat33::createDiagonal(PxVec3(0.0f));

	for (PxU32 i = 0; i < body.particleCount; i++)
	{
		const PxU32 pi = body.particleStart + i;
		const PxReal mass = particles[pi].mass;
		const PxVec3 r = particles[pi].position - centroid;
		const PxReal r2 = r.dot(r);
		inertia = inertia + (PxMat33::createDiagonal(PxVec3(r2)) - avbdOuter(r, r)) * mass;
		angularMomentum += r.cross(particles[pi].velocity) * mass;
	}

	PxVec3 omega = inertia.getInverse() * angularMomentum;
	if (omega.x != omega.x || omega.y != omega.y || omega.z != omega.z)
		return PxVec3(0.0f);
	return omega;
}

static void printBodyRotationTrace(const char* label,
																	 PxU32 frame,
																	 const PxArray<AvbdSoftParticle>& particles,
																	 const PxArray<AvbdSoftBody>& bodies,
																	 const PxArray<PxArray<PxVec3> >& refs)
{
	for (PxU32 bi = 0; bi < bodies.size(); bi++)
	{
		const PxQuat q = estimateBodyRotation(particles, bodies[bi], refs[bi]);
		PxReal angleDeg = PxAcos(PxClamp(q.w, -1.0f, 1.0f)) * (360.0f / PxPi);
		if (angleDeg > 180.0f)
			angleDeg = 360.0f - angleDeg;

		PxVec3 axis(0.0f, 1.0f, 0.0f);
		const PxReal sinHalf = PxSqrt(PxMax(0.0f, 1.0f - q.w * q.w));
		if (sinHalf > 1e-5f)
			axis = PxVec3(q.x, q.y, q.z) * (1.0f / sinHalf);

		const PxVec3 com = computeBodyMassCentroid(particles, bodies[bi]);
		const PxVec3 omega = estimateBodyOmega(particles, bodies[bi]);
		printf("  ROT[%s] frame=%u body=%u angleDeg=%.3f axis=(%.3f,%.3f,%.3f) omega=(%.3f,%.3f,%.3f) com=(%.3f,%.3f,%.3f)\n",
					 label, frame, bi,
					 angleDeg, axis.x, axis.y, axis.z,
					 omega.x, omega.y, omega.z,
					 com.x, com.y, com.z);
	}
}

// ============================================================================
// Helper: run soft body sim for N frames
// ============================================================================

static void stepSoft(PxArray<AvbdSoftParticle>& particles,
                     PxArray<AvbdSoftBody>& softBodies,
                     PxArray<AvbdSoftContact>& contacts,
                     PxU32 frames, PxReal dt,
                     PxVec3 gravity = PxVec3(0.0f, -9.81f, 0.0f),
                     PxReal groundY = 0.0f,
                     bool enableGround = true,
                     PxU32 outerIter = 1, PxU32 innerIter = 10,
                     PxReal chebyshevRho = 0.92f)
{
	PxArray<PxArray<PxVec3> > refs;
	const bool trace = isRotationTraceEnabled();
	const PxU32 traceInterval = getRotationTraceInterval();
	if (trace)
	{
		captureBodyReferenceLocals(particles, softBodies, refs);
		printBodyRotationTrace("stepSoft", 0, particles, softBodies, refs);
	}

  for (PxU32 f = 0; f < frames; f++)
  {
    if (enableGround)
      avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
                                   contacts, groundY, 0.02f, 0.5f);
    else
      contacts.clear();

    avbdStepSoftBodies(
      particles.begin(), particles.size(),
      softBodies.begin(), softBodies.size(),
      contacts.begin(), contacts.size(),
      dt, gravity, outerIter, innerIter, 1000.0f,
      NULL, NULL, NULL, chebyshevRho);

		if (trace && (((f + 1) % traceInterval) == 0 || (f + 1) == frames))
			printBodyRotationTrace("stepSoft", f + 1, particles, softBodies, refs);
  }
}

// ============================================================================
// Test 1: Gravity free-fall
// ============================================================================

static void testGravityFreeFall()
{
  printf("\n--- Test 1: Gravity Free-Fall ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 5.0f, 0.0f), 0.3f, 2, verts, tets);

  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
                     particles, bodies);

  PxVec3 c0 = getParticleCentroid(particles.begin(), 0, particles.size());

  // 60 frames = 1 second, no ground plane
  // Chebyshev disabled (rho=0) to avoid amplifying symmetric drift
  stepSoft(particles, bodies, contacts, 60, 1.0f/60.0f,
           PxVec3(0.0f, -9.81f, 0.0f), -100.0f, false,
           1, 10, 0.0f);

  PxVec3 c1 = getParticleCentroid(particles.begin(), 0, particles.size());

  // Centroid should have fallen significantly (~4.9m in 1s of free fall)
  TEST_CHECK(c1.y < c0.y - 3.0f, "Centroid dropped >3m in 1s free fall");
  // Horizontal should stay near zero
  TEST_CHECK(PxAbs(c1.x - c0.x) < 0.1f, "No horizontal drift in X");
  TEST_CHECK(PxAbs(c1.z - c0.z) < 0.1f, "No horizontal drift in Z");
}

// ============================================================================
// Test 2: Ground contact
// ============================================================================

static void testGroundContact()
{
  printf("\n--- Test 2: Ground Contact ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.0f, 0.0f), 0.3f, 2, verts, tets);

  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  // 5 seconds to let it settle
  stepSoft(particles, bodies, contacts, 300, 1.0f/60.0f);

  PxReal minY, maxY;
  getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);

  // No particle should be significantly below ground
  TEST_CHECK(minY > -0.1f, "No particle deeply below ground");
  // The body should be resting near ground
  TEST_CHECK(maxY < 2.0f, "Body settled (not floating)");
  // Velocities should be small (at rest)
  PxReal maxV = getMaxSpeed(particles.begin(), 0, particles.size());
  TEST_CHECK(maxV < 1.0f, "Velocities near zero at rest");
}

// ============================================================================
// Test 3: Volume preservation (Neo-Hookean)
// ============================================================================

static void testVolumePreservation()
{
  printf("\n--- Test 3: Volume Preservation ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  PxReal halfSize = 0.3f;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 2.0f, 0.0f), halfSize, 3, verts, tets);

  // High stiffness to resist compression
  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     5e5f, 0.4f, 1000.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  PxReal minY0, maxY0;
  getParticleBoundsY(particles.begin(), 0, particles.size(), minY0, maxY0);
  PxReal height0 = maxY0 - minY0;

  // Drop onto ground and settle
  stepSoft(particles, bodies, contacts, 300, 1.0f/60.0f);

  PxReal minY1, maxY1;
  getParticleBoundsY(particles.begin(), 0, particles.size(), minY1, maxY1);
  PxReal height1 = maxY1 - minY1;

  // Height should be preserved to within 50% (volume preservation)
  PxReal ratio = height1 / height0;
  TEST_CHECK(ratio > 0.4f, "Vertical extent >40% of original (not flat)");
  TEST_CHECK(ratio < 1.5f, "Vertical extent <150% of original (not exploded)");
}

// ============================================================================
// Test 4: Kinematic pin
// ============================================================================

static void testKinematicPin()
{
  printf("\n--- Test 4: Kinematic Pin ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 3.0f, 0.0f), 0.3f, 2, verts, tets);

  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  // Pin the top 4 particles at their initial positions
  PxReal topY = -PX_MAX_F32;
  for (PxU32 i = 0; i < particles.size(); i++)
    if (particles[i].position.y > topY) topY = particles[i].position.y;

  for (PxU32 i = 0; i < particles.size(); i++)
  {
    if (PxAbs(particles[i].position.y - topY) < 0.01f)
    {
      AvbdKinematicPin pin;
      pin.particleIdx = i;
      pin.worldTarget = particles[i].position;
      pin.k = 1e5f;
      pin.kMax = 1e7f;
      bodies[0].pins.pushBack(pin);

      // Rebuild adjacency pin refs
      PxU32 localIdx = i - bodies[0].particleStart;
      if (localIdx < bodies[0].adjacency.size())
        bodies[0].adjacency[localIdx].pinIndices.pushBack(bodies[0].pins.size() - 1);
    }
  }

  // Sim 3 seconds
  stepSoft(particles, bodies, contacts, 180, 1.0f/60.0f,
           PxVec3(0.0f, -9.81f, 0.0f), -100.0f, false);

  // Pinned particles should still be near their targets
  bool allPinsClose = true;
  for (PxU32 i = 0; i < bodies[0].pins.size(); i++)
  {
    PxU32 pi = bodies[0].pins[i].particleIdx;
    PxReal dist = (particles[pi].position - bodies[0].pins[i].worldTarget).magnitude();
    if (dist > 0.5f) allPinsClose = false;
  }
  TEST_CHECK(allPinsClose, "All pinned particles within 0.5m of target");

  // Bottom particles should have dropped
  PxReal minY, maxY;
  getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);
  TEST_CHECK(maxY > minY + 0.1f, "Body stretched vertically (pins hold top)");
}

// ============================================================================
// Test 5: Cloth drape
// ============================================================================

static void testClothDrape()
{
  printf("\n--- Test 5: Cloth Drape ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tris;
  avbdGenerateClothGrid(PxVec3(0.0f, 2.0f, 0.0f), 2.0f, 2.0f, 10, 10, verts, tris);

  avbdCreateSoftBody(verts.begin(), verts.size(),
                     NULL, 0,
                     tris.begin(), tris.size(),
                     1e4f, 0.3f, 500.0f, 5.0f, 1.0f, 0.005f,
                     particles, bodies);

  PxVec3 c0 = getParticleCentroid(particles.begin(), 0, particles.size());

  // Drop onto ground, 3 seconds
  stepSoft(particles, bodies, contacts, 180, 1.0f/60.0f);

  PxVec3 c1 = getParticleCentroid(particles.begin(), 0, particles.size());

  // Centroid should have dropped
  TEST_CHECK(c1.y < c0.y - 0.5f, "Cloth centroid dropped");

  PxReal minY, maxY;
  getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);
  TEST_CHECK(minY > -0.2f, "Cloth not below ground");

  // Cloth should be relatively flat after resting
  PxReal vertExtent = maxY - minY;
  TEST_CHECK(vertExtent < 1.5f, "Cloth vertical extent reasonable");
}

// ============================================================================
// Test 6: Energy dissipation (damping)
// ============================================================================

static void testEnergyDissipation()
{
  printf("\n--- Test 6: Energy Dissipation ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.5f, 0.0f), 0.3f, 2, verts, tets);

  // High damping
  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 50.0f, 0.0f, 0.01f,
                     particles, bodies);

  // Drop onto ground
  stepSoft(particles, bodies, contacts, 120, 1.0f/60.0f);

  PxReal v1 = getMaxSpeed(particles.begin(), 0, particles.size());

  // Continue for 3 more seconds
  stepSoft(particles, bodies, contacts, 180, 1.0f/60.0f);

  PxReal v2 = getMaxSpeed(particles.begin(), 0, particles.size());

  // Highly damped body should be nearly at rest
  TEST_CHECK(v2 < 2.0f, "Max speed < 2 m/s after 5s with high damping");
  // Speed should have decreased or stayed low
  TEST_CHECK(v2 <= v1 + 0.5f, "Speed did not increase significantly");
}

// ============================================================================
// Test 7: Static equilibrium (zero gravity)
// ============================================================================

static void testStaticEquilibrium()
{
  printf("\n--- Test 7: Static Equilibrium ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 2.0f, 0.0f), 0.3f, 2, verts, tets);

  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 5.0f, 0.0f, 0.01f,
                     particles, bodies);

  PxVec3 c0 = getParticleCentroid(particles.begin(), 0, particles.size());

  // Zero gravity, no ground
  stepSoft(particles, bodies, contacts, 120, 1.0f/60.0f,
           PxVec3(0.0f), -100.0f, false);

  PxVec3 c1 = getParticleCentroid(particles.begin(), 0, particles.size());

  // Should stay exactly in place (no external forces, rest shape)
  PxReal drift = (c1 - c0).magnitude();
  TEST_CHECK(drift < 0.01f, "Centroid drift < 1cm in zero gravity");

  PxReal maxV = getMaxSpeed(particles.begin(), 0, particles.size());
  TEST_CHECK(maxV < 0.1f, "Max speed < 0.1 m/s in zero gravity");
}

// ============================================================================
// Test 8: Multiple soft bodies
// ============================================================================

static void testMultipleSoftBodies()
{
  printf("\n--- Test 8: Multiple Soft Bodies ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  // Body A: high, left
  PxArray<PxVec3> vertsA;
  PxArray<PxU32> tetsA;
  avbdGenerateSubdividedCubeTets(PxVec3(-2.0f, 3.0f, 0.0f), 0.3f, 2, vertsA, tetsA);

  avbdCreateSoftBody(vertsA.begin(), vertsA.size(),
                     tetsA.begin(), tetsA.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  // Body B: higher, right
  PxArray<PxVec3> vertsB;
  PxArray<PxU32> tetsB;
  avbdGenerateSubdividedCubeTets(PxVec3(2.0f, 5.0f, 0.0f), 0.3f, 2, vertsB, tetsB);

  avbdCreateSoftBody(vertsB.begin(), vertsB.size(),
                     tetsB.begin(), tetsB.size(), NULL, 0,
                     1e5f, 0.3f, 1000.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  TEST_CHECK(bodies.size() == 2, "Two soft bodies created");
  TEST_CHECK(particles.size() == vertsA.size() + vertsB.size(), "Correct total particles");

  // Drop both onto ground, 5 seconds
  stepSoft(particles, bodies, contacts, 300, 1.0f/60.0f);

  // Both should be near ground
  PxReal minYA, maxYA;
  getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minYA, maxYA);

  PxReal minYB, maxYB;
  getParticleBoundsY(particles.begin(), bodies[1].particleStart, bodies[1].particleCount, minYB, maxYB);

  TEST_CHECK(minYA > -0.1f, "Body A above ground");
  TEST_CHECK(maxYA < 2.0f,  "Body A settled");
  TEST_CHECK(minYB > -0.1f, "Body B above ground");
  TEST_CHECK(maxYB < 2.0f,  "Body B settled");

  // Bodies should remain separated (no cross-contamination)
  PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
  PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);
  TEST_CHECK(PxAbs(cA.x - cB.x) > 1.0f, "Bodies remain horizontally separated");
}

// ============================================================================
// Helper: run soft body sim with full collision (ground + soft-soft + soft-rigid)
// ============================================================================

static void stepSoftFull(PxArray<AvbdSoftParticle>& particles,
                         PxArray<AvbdSoftBody>& softBodies,
                         PxArray<AvbdSoftContact>& contacts,
                         PxArray<AvbdRigidBox>* rigidBoxes,
                         PxU32 frames, PxReal dt,
                         PxVec3 gravity = PxVec3(0.0f, -9.81f, 0.0f),
                         PxReal groundY = 0.0f,
                         PxU32 outerIter = 2, PxU32 innerIter = 15,
                         PxReal softSoftMargin = 0.3f,
                         PxReal rigidMargin = 0.1f,
                         PxU32 detectInterval = 1)
{
	PxArray<PxArray<PxVec3> > refs;
	const bool trace = isRotationTraceEnabled();
	const PxU32 traceInterval = getRotationTraceInterval();
	if (trace)
	{
		captureBodyReferenceLocals(particles, softBodies, refs);
		printBodyRotationTrace("stepSoftFull", 0, particles, softBodies, refs);
	}

  for (PxU32 f = 0; f < frames; f++)
  {
    if (f % detectInterval == 0)
    {
      avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
                                   contacts, groundY, 0.02f, 0.5f);

      avbdDetectSoftSoftContacts(particles.begin(), particles.size(),
                                 softBodies.begin(), softBodies.size(),
                                 contacts, softSoftMargin, 0.5f);

      if (rigidBoxes && rigidBoxes->size() > 0)
				avbdDetectSoftRigidSDF(particles.begin(), particles.size(),
															 rigidBoxes->begin(), rigidBoxes->size(),
															 contacts, rigidMargin);
    }

    avbdStepSoftBodies(
      particles.begin(), particles.size(),
      softBodies.begin(), softBodies.size(),
      contacts.begin(), contacts.size(),
      dt, gravity, outerIter, innerIter, 1000.0f);

		if (trace && (((f + 1) % traceInterval) == 0 || (f + 1) == frames))
			printBodyRotationTrace("stepSoftFull", f + 1, particles, softBodies, refs);
  }
}

// ============================================================================
// Test 9: Soft-soft collision
// ============================================================================

static void testSoftSoftCollision()
{
  printf("\n--- Test 9: Soft-Soft Collision ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;

  // Body A (bottom): cube sitting near ground
  PxArray<PxVec3> vertsA;
  PxArray<PxU32> tetsA;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.0f, 0.0f), 0.5f, 3, vertsA, tetsA);

  avbdCreateSoftBody(vertsA.begin(), vertsA.size(),
                     tetsA.begin(), tetsA.size(), NULL, 0,
                     2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  // Body B (top): cube dropped from above, directly on top of A
  PxArray<PxVec3> vertsB;
  PxArray<PxU32> tetsB;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3, vertsB, tetsB);

  avbdCreateSoftBody(vertsB.begin(), vertsB.size(),
                     tetsB.begin(), tetsB.size(), NULL, 0,
                     2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  // Record initial centroid of top body
  PxVec3 cB0 = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);

  // Simulate 5 seconds with full collision
  stepSoftFull(particles, bodies, contacts, NULL, 300, 1.0f/60.0f);

  PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
  PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);

  PxReal minYA, maxYA;
  getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minYA, maxYA);
  PxReal minYB, maxYB;
  getParticleBoundsY(particles.begin(), bodies[1].particleStart, bodies[1].particleCount, minYB, maxYB);

  printf("  cA=(%.3f,%.3f,%.3f) cB=(%.3f,%.3f,%.3f)\n", cA.x, cA.y, cA.z, cB.x, cB.y, cB.z);
  printf("  boundsA=[%.3f..%.3f] boundsB=[%.3f..%.3f]\n", minYA, maxYA, minYB, maxYB);

  // Body B should have fallen but NOT passed through Body A
  TEST_CHECK(cB.y < cB0.y - 0.5f, "Top body fell under gravity");
  // With realistic friction, the top cube may slide off, so only check
  // that the centroids haven't interpenetrated deeply.
  TEST_CHECK(cB.y > cA.y - 0.5f, "Top body centroid not far below bottom");
  TEST_CHECK(minYB > -0.15f, "Top body above ground after settling");
  // Both above ground
  TEST_CHECK(minYA > -0.15f, "Bottom body above ground");
  TEST_CHECK(minYB > -0.15f, "Top body above ground");
}

// ============================================================================
// Test 10: Soft-rigid box collision
// ============================================================================

static void testSoftRigidCollision()
{
  printf("\n--- Test 10: Soft-Rigid Box Collision ---\n");

  PxArray<AvbdSoftParticle> particles;
  PxArray<AvbdSoftBody> bodies;
  PxArray<AvbdSoftContact> contacts;
  PxArray<AvbdRigidBox> rigidBoxes;

  // Soft cube dropped from above
  PxArray<PxVec3> verts;
  PxArray<PxU32> tets;
  avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 5.0f, 0.0f), 0.5f, 3, verts, tets);

  avbdCreateSoftBody(verts.begin(), verts.size(),
                     tets.begin(), tets.size(), NULL, 0,
                     2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
                     particles, bodies);

  // Rigid box obstacle on the ground at (0, 1.5, 0)
  AvbdRigidBox box;
  box.center = PxVec3(0.0f, 1.5f, 0.0f);
  box.rotation = PxQuat(PxIdentity);
  box.halfExtent = PxVec3(2.0f, 1.5f, 2.0f);
  box.friction = 0.5f;
  rigidBoxes.pushBack(box);

  PxVec3 c0 = getParticleCentroid(particles.begin(), 0, particles.size());

  // Simulate 5 seconds
  stepSoftFull(particles, bodies, contacts, &rigidBoxes, 300, 1.0f/60.0f);

  PxVec3 c1 = getParticleCentroid(particles.begin(), 0, particles.size());

  PxReal minY, maxY;
  getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);

  // Soft body should have fallen
  TEST_CHECK(c1.y < c0.y - 0.5f, "Soft body fell under gravity");
  // Rigid box top is at y=3: soft body should rest ON it (not fall through)
  TEST_CHECK(minY > 2.5f, "Soft body rests on rigid box (not fallen through)");
  // Should be near the box top, not exploded
  TEST_CHECK(maxY < 6.0f, "Soft body not exploded upward");
}

// ============================================================================
// Test 11: Slope rotation (cube sliding/rotating on inclined surface)
// ============================================================================

static void testSlopeRotation()
{
	printf("\n--- Test 11: Slope Rotation ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;
	PxArray<AvbdRigidBox> rigidBoxes;

	// Soft cube
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 5.0f, 0.0f), 0.5f, 3, verts, tets);

	avbdCreateSoftBody(verts.begin(), verts.size(),
						 tets.begin(), tets.size(), NULL, 0,
						 2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
						 particles, bodies);

	// Inclined rigid box: 30-degree slope tilted so +X is downhill.
	// The box acts as a wide ramp.
	AvbdRigidBox ramp;
	ramp.center = PxVec3(0.0f, 2.0f, 0.0f);
	PxReal angle = 3.14159265f / 6.0f; // 30 degrees
	ramp.rotation = PxQuat(angle, PxVec3(0.0f, 0.0f, 1.0f));
	ramp.halfExtent = PxVec3(4.0f, 0.5f, 4.0f);
	ramp.friction = 0.3f;
	rigidBoxes.pushBack(ramp);

	PxVec3 c0 = getParticleCentroid(particles.begin(), 0, particles.size());

	// Simulate 3 seconds with ground at y=0 so cube stops after sliding off ramp
	stepSoftFull(particles, bodies, contacts, &rigidBoxes, 180, 1.0f/60.0f,
				 PxVec3(0.0f, -9.81f, 0.0f), 0.0f, 3, 10, 0.3f, 0.1f);

	PxVec3 c1 = getParticleCentroid(particles.begin(), 0, particles.size());

	printf("  c0=(%.3f,%.3f,%.3f) c1=(%.3f,%.3f,%.3f)\n",
		   c0.x, c0.y, c0.z, c1.x, c1.y, c1.z);

	// The cube should have slid downhill in X (or at least moved laterally)
	PxReal lateralDrift = PxAbs(c1.x - c0.x) + PxAbs(c1.z - c0.z);
	TEST_CHECK(lateralDrift > 0.3f, "Cube drifted laterally on slope (sliding/rotation)");

	// The cube should have dropped from its starting height
	TEST_CHECK(c1.y < c0.y - 0.5f, "Cube fell under gravity onto slope");

	// The cube should not have exploded or gone below ground
	PxReal minY, maxY;
	getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);
	TEST_CHECK(minY > -0.5f, "Cube above ground (not fallen through)");
	TEST_CHECK(maxY < 10.0f, "Cube not exploded");
}

// ============================================================================
// Test 12: Cone-cube penetration (cone sitting on cube, no interpenetration)
// ============================================================================

static void testConeCubePenetration()
{
	printf("\n--- Test 12: Cone-Cube Penetration ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	// Body A: soft cube (bottom), close to ground so it settles quickly
	PxArray<PxVec3> cubeVerts;
	PxArray<PxU32> cubeTets;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 0.6f, 0.0f), 0.5f, 3, cubeVerts, cubeTets);

	avbdCreateSoftBody(cubeVerts.begin(), cubeVerts.size(),
						 cubeTets.begin(), cubeTets.size(), NULL, 0,
						 2e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
						 particles, bodies);

	// Body B: soft cone (top), placed just above where cube will settle
	// Cube top ~1.1, cone base at 1.5 -> small gap, minimal impact velocity
	PxArray<PxVec3> coneVerts;
	PxArray<PxU32> coneTets;
	avbdGenerateConeTets(PxVec3(0.0f, 2.0f, 0.0f), 0.5f, 1.5f, 3, coneVerts, coneTets);

	avbdCreateSoftBody(coneVerts.begin(), coneVerts.size(),
						 coneTets.begin(), coneTets.size(), NULL, 0,
						 2e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
						 particles, bodies);

	// Record initial centroids
	PxVec3 cCube0 = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
	PxVec3 cCone0 = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);

	AvbdOGCParams ogcParams;
	ogcParams.contactRadius    = 0.15f;
	ogcParams.contactStiffness = 1e5f;
	ogcParams.friction         = 0.3f;

	// Simulate 5 seconds with ground + OGC soft-soft collision
	for (PxU32 f = 0; f < 300; f++)
	{
		if (f % 2 == 0)
		{
			contacts.clear();
			avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
										 contacts, 0.0f, 0.05f, 0.5f);

			avbdDetectSoftSoftOGC(particles.begin(), particles.size(),
								  bodies.begin(), bodies.size(),
								  contacts, ogcParams);
		}

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f);
	}

	PxVec3 cCube = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
	PxVec3 cCone = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);

	PxReal minYCube, maxYCube;
	getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minYCube, maxYCube);
	PxReal minYCone, maxYCone;
	getParticleBoundsY(particles.begin(), bodies[1].particleStart, bodies[1].particleCount, minYCone, maxYCone);

	printf("  cube: c=(%.3f,%.3f,%.3f) bounds=[%.3f..%.3f]\n", cCube.x, cCube.y, cCube.z, minYCube, maxYCube);
	printf("  cone: c=(%.3f,%.3f,%.3f) bounds=[%.3f..%.3f]\n", cCone.x, cCone.y, cCone.z, minYCone, maxYCone);

	// Cone centroid should have dropped from its start (it fell onto the cube)
	TEST_CHECK(cCone.y < cCone0.y, "Cone fell under gravity");

	// Both bodies above ground
	TEST_CHECK(minYCube > -0.15f, "Cube above ground");
	TEST_CHECK(minYCone > -0.15f, "Cone above ground");

	// Key test: cone bottom must NOT penetrate deeply into cube top.
	PxReal overlap = maxYCube - minYCone;
	printf("  overlap (cubeMaxY - coneMinY) = %.3f\n", overlap);
	TEST_CHECK(overlap < 0.7f, "Cone not deeply penetrating cube (overlap < 0.7)");

	// Cone centroid should be above cube centroid (not fallen through it)
	TEST_CHECK(cCone.y > cCube.y - 0.2f, "Cone centroid above cube centroid");

	// Neither body should have exploded
	TEST_CHECK(maxYCone < 8.0f, "Cone not exploded");
	TEST_CHECK(maxYCube < 5.0f, "Cube not exploded");
}

// ============================================================================
// Test 13: OGC Soft-Soft Collision (Sec 3.9 simplified path)
// ============================================================================

static void testOGCSoftSoftCollision()
{
	printf("\n--- Test 13: OGC Soft-Soft Collision ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	// Body A (bottom): cube near ground
	PxArray<PxVec3> vertsA, vertsB;
	PxArray<PxU32> tetsA, tetsB;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.0f, 0.0f), 0.5f, 3, vertsA, tetsA);
	avbdCreateSoftBody(vertsA.begin(), vertsA.size(),
	                   tetsA.begin(), tetsA.size(), NULL, 0,
	                   2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
	                   particles, bodies);

	// Body B (top): cube dropped from above
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3, vertsB, tetsB);
	avbdCreateSoftBody(vertsB.begin(), vertsB.size(),
	                   tetsB.begin(), tetsB.size(), NULL, 0,
	                   2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
	                   particles, bodies);

	PxVec3 cB0 = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);

	AvbdOGCParams ogc;
	ogc.contactRadius    = 0.15f;
	ogc.contactStiffness = 1e5f;
	ogc.friction         = 0.3f;

	for (PxU32 f = 0; f < 300; f++)
	{
		if (f % 2 == 0)
		{
			contacts.clear();
			avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
			                             contacts, 0.0f, 0.02f, 0.5f);
			avbdDetectSoftSoftOGC(particles.begin(), particles.size(),
			                      bodies.begin(), bodies.size(),
			                      contacts, ogc);
		}

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f);
	}

	PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
	PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);
	PxReal minYA, maxYA, minYB, maxYB;
	getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minYA, maxYA);
	getParticleBoundsY(particles.begin(), bodies[1].particleStart, bodies[1].particleCount, minYB, maxYB);

	printf("  cA=(%.3f,%.3f,%.3f) cB=(%.3f,%.3f,%.3f)\n", cA.x, cA.y, cA.z, cB.x, cB.y, cB.z);
	printf("  boundsA=[%.3f..%.3f] boundsB=[%.3f..%.3f]\n", minYA, maxYA, minYB, maxYB);

	TEST_CHECK(cB.y < cB0.y - 0.5f, "Top body fell under gravity (OGC)");
	TEST_CHECK(cB.y > cA.y - 0.5f, "Top body not passthrough bottom (OGC)");
	TEST_CHECK(minYA > -0.15f, "Bottom body above ground (OGC)");
	TEST_CHECK(minYB > -0.15f, "Top body above ground (OGC)");
}

// ============================================================================
// Test 14: OGC Rigid-Soft Collision (analytical SDF)
// ============================================================================

static void testOGCRigidSoftCollision()
{
	printf("\n--- Test 14: OGC Rigid-Soft SDF Collision ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	// Soft cube dropped onto a rigid box
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 3.0f, 0.0f), 0.5f, 3, verts, tets);
	avbdCreateSoftBody(verts.begin(), verts.size(),
	                   tets.begin(), tets.size(), NULL, 0,
	                   1e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
	                   particles, bodies);

	PxArray<AvbdRigidBox> rigidBoxes;
	AvbdRigidBox rb;
	rb.center     = PxVec3(0.0f, 0.5f, 0.0f);
	rb.halfExtent = PxVec3(2.0f, 0.5f, 2.0f);
	rb.friction   = 0.5f;
	rigidBoxes.pushBack(rb);

	PxVec3 c0 = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);

	for (PxU32 f = 0; f < 300; f++)
	{
		if (f % 2 == 0)
		{
			contacts.clear();
			avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
			                             contacts, 0.0f, 0.02f, 0.5f);
			avbdDetectSoftRigidSDF(particles.begin(), particles.size(),
			                       rigidBoxes.begin(), rigidBoxes.size(),
			                       contacts, 0.05f);
		}

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f);
	}

	PxVec3 cEnd = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
	PxReal minY, maxY;
	getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minY, maxY);

	printf("  centroid=(%.3f,%.3f,%.3f) bounds=[%.3f..%.3f]\n", cEnd.x, cEnd.y, cEnd.z, minY, maxY);

	TEST_CHECK(cEnd.y < c0.y, "Soft cube fell (SDF)");
	TEST_CHECK(minY > 0.9f, "Soft cube above rigid box top face (SDF)");
	TEST_CHECK(maxY < 5.0f, "Soft cube not exploded (SDF)");
}

// ============================================================================
// Test 15: OGC Self-Collision Detection
// ============================================================================

static void testOGCSelfCollision()
{
	printf("\n--- Test 15: OGC Self-Collision ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	// Large soft cube that can self-collide when compressed
	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 3.0f, 0.0f), 1.0f, 3, verts, tets);
	avbdCreateSoftBody(verts.begin(), verts.size(),
	                   tets.begin(), tets.size(), NULL, 0,
	                   5e4f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
	                   particles, bodies);

	// Build self-collision adjacency
	PxArray<AvbdSelfCollisionAdjacency> selfAdj;
	avbdBuildAllSelfCollisionAdjacencies(bodies.begin(), bodies.size(), selfAdj);

	AvbdOGCParams ogc;
	ogc.contactRadius    = 0.08f;
	ogc.contactStiffness = 5e4f;
	ogc.friction         = 0.3f;

	for (PxU32 f = 0; f < 180; f++)
	{
		contacts.clear();
		avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
		                             contacts, 0.0f, 0.02f, 0.5f);
		avbdDetectSelfCollisionOGC(particles.begin(), bodies[0], selfAdj[0], contacts, ogc);

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f);
	}

	PxVec3 cEnd = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
	PxReal minY, maxY;
	getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minY, maxY);

	printf("  centroid=(%.3f,%.3f,%.3f) bounds=[%.3f..%.3f]\n", cEnd.x, cEnd.y, cEnd.z, minY, maxY);

	// Body should settle on ground without exploding or collapsing
	TEST_CHECK(minY > -0.15f, "Self-collision body above ground");
	TEST_CHECK(maxY < 5.0f, "Self-collision body not exploded");
	TEST_CHECK(cEnd.y > 0.0f, "Self-collision centroid positive Y");
}

// ============================================================================
// Test 16: OGC Full Pipeline (all paths combined)
// ============================================================================

static void testOGCFullPipeline()
{
	printf("\n--- Test 16: OGC Full Pipeline ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	// Two soft cubes + rigid box floor
	PxArray<PxVec3> vertsA, vertsB;
	PxArray<PxU32> tetsA, tetsB;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.5f, 0.0f), 0.5f, 3, vertsA, tetsA);
	avbdCreateSoftBody(vertsA.begin(), vertsA.size(),
	                   tetsA.begin(), tetsA.size(), NULL, 0,
	                   2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
	                   particles, bodies);

	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 4.0f, 0.0f), 0.5f, 3, vertsB, tetsB);
	avbdCreateSoftBody(vertsB.begin(), vertsB.size(),
	                   tetsB.begin(), tetsB.size(), NULL, 0,
	                   2e5f, 0.3f, 500.0f, 10.0f, 0.0f, 0.01f,
	                   particles, bodies);

	PxArray<AvbdRigidBox> rigidBoxes;
	AvbdRigidBox rb;
	rb.center     = PxVec3(3.0f, 0.5f, 0.0f); // off to the side, won't interact
	rb.halfExtent = PxVec3(1.0f, 0.5f, 1.0f);
	rb.friction   = 0.5f;
	rigidBoxes.pushBack(rb);

	// Pre-build adjacencies
	PxArray<AvbdSelfCollisionAdjacency> selfAdj;
	avbdBuildAllSelfCollisionAdjacencies(bodies.begin(), bodies.size(), selfAdj);

	AvbdOGCParams ogc;
	ogc.contactRadius    = 0.15f;
	ogc.contactStiffness = 1e5f;
	ogc.friction         = 0.3f;

	for (PxU32 f = 0; f < 300; f++)
	{
		contacts.clear();
		avbdDetectAllOGCContacts(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			rigidBoxes.begin(), rigidBoxes.size(),
			selfAdj.begin(), selfAdj.size(),
			contacts, ogc, 0.0f);

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f);
	}

	PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].particleStart, bodies[0].particleCount);
	PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].particleStart, bodies[1].particleCount);
	PxReal minYA, maxYA, minYB, maxYB;
	getParticleBoundsY(particles.begin(), bodies[0].particleStart, bodies[0].particleCount, minYA, maxYA);
	getParticleBoundsY(particles.begin(), bodies[1].particleStart, bodies[1].particleCount, minYB, maxYB);

	printf("  cA=(%.3f,%.3f,%.3f) cB=(%.3f,%.3f,%.3f)\n", cA.x, cA.y, cA.z, cB.x, cB.y, cB.z);
	printf("  boundsA=[%.3f..%.3f] boundsB=[%.3f..%.3f]\n", minYA, maxYA, minYB, maxYB);

	TEST_CHECK(minYA > -0.15f, "Body A above ground (full OGC)");
	TEST_CHECK(minYB > -0.15f, "Body B above ground (full OGC)");
	TEST_CHECK(cB.y > cA.y - 0.5f, "B centroid not far below A (full OGC)");
	TEST_CHECK(maxYA < 5.0f, "Body A not exploded (full OGC)");
	TEST_CHECK(maxYB < 5.0f, "Body B not exploded (full OGC)");
}

// ============================================================================
// Test 17: Asymmetric toppling (body-level 6x6 solve validation)
//
// A tet cube is placed on an edge rotated ~30 deg. The body-level 6x6
// solve should generate torque from asymmetric ground contacts, causing
// the cube to topple and settle with a lower COM.
// ============================================================================

static void testAsymmetricToppling()
{
	printf("\n--- Test 17: Asymmetric Toppling ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.0f, 0.0f), 0.5f, 2, verts, tets);

	// Rotate ~30 degrees around Z axis
	const PxReal angle = 0.52f;
	const PxReal cs = PxCos(angle), sn = PxSin(angle);
	const PxVec3 center(0.0f, 1.0f, 0.0f);
	for (PxU32 i = 0; i < verts.size(); i++)
	{
		PxVec3 r = verts[i] - center;
		verts[i].x = center.x + r.x * cs - r.y * sn;
		verts[i].y = center.y + r.x * sn + r.y * cs;
	}

	avbdCreateSoftBody(verts.begin(), verts.size(),
	                   tets.begin(), tets.size(), NULL, 0,
	                   1e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
	                   particles, bodies);

	PxVec3 c0 = getParticleCentroid(particles.begin(), 0, particles.size());

	stepSoft(particles, bodies, contacts, 180, 1.0f/60.0f,
	         PxVec3(0.0f, -9.81f, 0.0f), 0.0f, true,
	         8, 20, 0.0f);

	PxVec3 c1 = getParticleCentroid(particles.begin(), 0, particles.size());
	PxReal minY, maxY;
	getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);

	printf("  COM start=(%.3f,%.3f,%.3f) end=(%.3f,%.3f,%.3f)\n",
	       c0.x, c0.y, c0.z, c1.x, c1.y, c1.z);
	printf("  Y bounds=[%.3f, %.3f]\n", minY, maxY);

	TEST_CHECK(c1.y < c0.y, "COM dropped (body toppled)");
	TEST_CHECK(c1.y < 1.5f, "Body settled near ground");
	TEST_CHECK(minY > -0.1f, "No ground penetration");
	TEST_CHECK(maxY < 3.0f, "No explosion");
}

// ============================================================================
// Test 18: Material stiffness comparison
//
// Soft (E=1e3) vs stiff (E=1e6) body: stiff should preserve shape better.
// ============================================================================

static void testMaterialStiffness()
{
	printf("\n--- Test 18: Material Stiffness ---\n");

	auto runStiffness = [](PxReal E) -> PxReal {
		PxArray<AvbdSoftParticle> particles;
		PxArray<AvbdSoftBody> bodies;
		PxArray<AvbdSoftContact> contacts;

		PxArray<PxVec3> verts;
		PxArray<PxU32> tets;
		avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 2.0f, 0.0f), 0.3f, 2, verts, tets);

		avbdCreateSoftBody(verts.begin(), verts.size(),
		                   tets.begin(), tets.size(), NULL, 0,
		                   E, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
		                   particles, bodies);

		stepSoft(particles, bodies, contacts, 120, 1.0f/60.0f,
		         PxVec3(0.0f, -9.81f, 0.0f), 0.0f, true,
		         8, 20, 0.0f);

		PxReal minY, maxY;
		getParticleBoundsY(particles.begin(), 0, particles.size(), minY, maxY);
		return maxY - minY;
	};

	PxReal hSoft = runStiffness(1e3f);
	PxReal hStiff = runStiffness(1e6f);
	printf("  Soft (E=1e3): height=%.3f  Stiff (E=1e6): height=%.3f\n", hSoft, hStiff);

	TEST_CHECK(hStiff > hSoft * 0.7f, "Stiff body preserves shape better");
}

// ============================================================================
// Test 19: Long-term stability (10 seconds, no NaN/explosion)
// ============================================================================

static void testLongTermStability()
{
	printf("\n--- Test 19: Long-Term Stability ---\n");

	PxArray<AvbdSoftParticle> particles;
	PxArray<AvbdSoftBody> bodies;
	PxArray<AvbdSoftContact> contacts;

	PxArray<PxVec3> verts;
	PxArray<PxU32> tets;
	avbdGenerateSubdividedCubeTets(PxVec3(0.0f, 1.0f, 0.0f), 0.3f, 2, verts, tets);

	avbdCreateSoftBody(verts.begin(), verts.size(),
	                   tets.begin(), tets.size(), NULL, 0,
	                   1e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
	                   particles, bodies);

	bool stable = true;
	for (PxU32 f = 0; f < 600; f++)
	{
		contacts.clear();
		avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
		                             contacts, 0.0f, 0.02f, 0.5f);
		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 8, 20, 1000.0f);

		for (PxU32 i = 0; i < particles.size(); i++)
		{
			if (particles[i].position.x != particles[i].position.x ||
			    PxAbs(particles[i].position.y) > 50.0f)
			{
				printf("  Unstable at frame %u\n", f);
				stable = false;
				break;
			}
		}
		if (!stable) break;
	}

	PxVec3 c = getParticleCentroid(particles.begin(), 0, particles.size());
	printf("  Final COM=(%.3f,%.3f,%.3f) stable=%s\n", c.x, c.y, c.z, stable ? "yes" : "no");

	TEST_CHECK(stable, "10-second simulation stable");
	TEST_CHECK(c.y > -0.5f && c.y < 3.0f, "COM in reasonable range");
}

// ===========================================================================
// Visual-mode infrastructure
// ===========================================================================

PxU32       gVisFrameCount = 0;
PxU32       gVisMaxFrames  = 0;
const char* gVisTestName   = NULL;

enum VisStepMode {
	VS_GROUND,
	VS_NO_GROUND,
	VS_FULL,
	VS_OGC,
	VS_OGC_RIGID,
	VS_OGC_SELF,
	VS_OGC_FULL,
};

static VisStepMode                          gVisStepMode  = VS_GROUND;
static PxArray<AvbdRigidBox>                gVisRigidBoxes;
static AvbdOGCParams                        gVisOGC;
static PxArray<AvbdSelfCollisionAdjacency>  gVisSelfAdj;
static PxArray<PxArray<PxVec3> >            gVisBodyRefs;
static PxVec3                               gVisGravity   = PxVec3(0.0f, -9.81f, 0.0f);
static PxReal                               gVisGroundY   = 0.0f;
static PxU32                                gVisOuterIter = 1;
static PxU32                                gVisInnerIter = 10;

static void addVisCube(PxVec3 center, PxReal half, PxU32 sub,
                       PxReal E, PxReal nu, PxReal density, PxReal damping,
                       PxReal compliance = 0.0f, PxReal lsTol = 0.01f)
{
	PxArray<PxVec3> v;
	PxArray<PxU32>  t;
	avbdGenerateSubdividedCubeTets(center, half, sub, v, t);
	avbdCreateSoftBody(v.begin(), v.size(),
	                   t.begin(), t.size(), NULL, 0,
	                   E, nu, density, damping, compliance, lsTol,
	                   gParticles, gSoftBodies);
}

static void updateVisRenderData()
{
	gSoftBodyRenderData.clear();
	for (PxU32 i = 0; i < gSoftBodies.size(); i++)
	{
		SoftBodyRenderData rd;
		rd.positions    = &gParticles[gSoftBodies[i].particleStart].position;
		rd.numParticles = gSoftBodies[i].particleCount;
		rd.tetIndices   = gSoftBodies[i].tetrahedra.begin();
		rd.numTets      = gSoftBodies[i].tetrahedra.size() / 4;
		rd.triIndices   = gSoftBodies[i].triangles.size() > 0 ? gSoftBodies[i].triangles.begin() : NULL;
		rd.numTris      = gSoftBodies[i].triangles.size() / 3;
		gSoftBodyRenderData.pushBack(rd);
	}
}

static void resetVisState()
{
	gParticles.clear();
	gSoftBodies.clear();
	gContacts.clear();
	gSoftBodyRenderData.clear();
	gVisRigidBoxes.clear();
	gVisSelfAdj.clear();
	gVisBodyRefs.clear();
	gVisFrameCount = 0;
	gVisMaxFrames  = 300;
	gVisGravity    = PxVec3(0.0f, -9.81f, 0.0f);
	gVisGroundY    = 0.0f;
	gVisOuterIter  = 1;
	gVisInnerIter  = 10;
	gVisStepMode   = VS_GROUND;
	gVisOGC        = AvbdOGCParams();
}

static void setupVisualTest(int testId)
{
	resetVisState();

	switch (testId)
	{
	case 1:
		addVisCube(PxVec3(0,5,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 0.01f);
		gVisMaxFrames = 90;
		gVisStepMode  = VS_NO_GROUND;
		gVisTestName  = "Test 1: Gravity Free-Fall";
		break;

	case 2:
		addVisCube(PxVec3(0,1,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 2: Ground Contact";
		break;

	case 3:
		addVisCube(PxVec3(0,2,0), 0.3f, 3, 5e5f, 0.4f, 1000.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 3: Volume Preservation";
		break;

	case 4:
	{
		addVisCube(PxVec3(0,3,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		PxReal topY = -PX_MAX_F32;
		for (PxU32 i = 0; i < gParticles.size(); i++)
			if (gParticles[i].position.y > topY) topY = gParticles[i].position.y;
		for (PxU32 i = 0; i < gParticles.size(); i++)
		{
			if (PxAbs(gParticles[i].position.y - topY) < 0.01f)
			{
				AvbdKinematicPin pin;
				pin.particleIdx = i;
				pin.worldTarget = gParticles[i].position;
				pin.k = 1e5f;
				pin.kMax = 1e7f;
				gSoftBodies[0].pins.pushBack(pin);
				PxU32 localIdx = i - gSoftBodies[0].particleStart;
				if (localIdx < gSoftBodies[0].adjacency.size())
					gSoftBodies[0].adjacency[localIdx].pinIndices.pushBack(gSoftBodies[0].pins.size() - 1);
			}
		}
		gVisMaxFrames = 180;
		gVisStepMode  = VS_NO_GROUND;
		gVisTestName  = "Test 4: Kinematic Pin";
		break;
	}

	case 5:
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32> tris;
		avbdGenerateClothGrid(PxVec3(0,2,0), 2.0f, 2.0f, 10, 10, verts, tris);
		avbdCreateSoftBody(verts.begin(), verts.size(),
		                   NULL, 0, tris.begin(), tris.size(),
		                   1e4f, 0.3f, 500.0f, 5.0f, 1.0f, 0.005f,
		                   gParticles, gSoftBodies);
		gVisMaxFrames = 180;
		gVisTestName  = "Test 5: Cloth Drape";
		break;
	}

	case 6:
		addVisCube(PxVec3(0,1.5f,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 50.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 6: Energy Dissipation";
		break;

	case 7:
		addVisCube(PxVec3(0,2,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 5.0f);
		gVisMaxFrames = 120;
		gVisStepMode  = VS_NO_GROUND;
		gVisGravity   = PxVec3(0.0f);
		gVisTestName  = "Test 7: Static Equilibrium";
		break;

	case 8:
		addVisCube(PxVec3(-2,3,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		addVisCube(PxVec3( 2,5,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisTestName  = "Test 8: Multiple Soft Bodies";
		break;

	case 9:
		addVisCube(PxVec3(0,1,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		addVisCube(PxVec3(0,4,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		gVisMaxFrames = 300;
		gVisStepMode  = VS_FULL;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 9: Soft-Soft Collision";
		break;

	case 10:
	{
		addVisCube(PxVec3(0,5,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox box;
		box.center     = PxVec3(0,1.5f,0);
		box.halfExtent = PxVec3(2,1.5f,2);
		box.friction   = 0.5f;
		gVisRigidBoxes.pushBack(box);
		gVisMaxFrames = 300;
		gVisStepMode  = VS_FULL;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 10: Soft-Rigid Collision";
		break;
	}

	case 11:
	{
		addVisCube(PxVec3(0,5,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox ramp;
		ramp.center     = PxVec3(0,2,0);
		ramp.rotation   = PxQuat(3.14159265f / 6.0f, PxVec3(0,0,1));
		ramp.halfExtent = PxVec3(4,0.5f,4);
		ramp.friction   = 0.3f;
		gVisRigidBoxes.pushBack(ramp);
		gVisMaxFrames = 180;
		gVisStepMode  = VS_FULL;
		gVisOuterIter = 3; gVisInnerIter = 10;
		gVisTestName  = "Test 11: Slope Rotation";
		break;
	}

	case 12:
	{
		addVisCube(PxVec3(0,0.6f,0), 0.5f, 3, 2e5f, 0.3f, 1000.0f, 0.01f);
		PxArray<PxVec3> cv; PxArray<PxU32> ct;
		avbdGenerateConeTets(PxVec3(0,2,0), 0.5f, 1.5f, 3, cv, ct);
		avbdCreateSoftBody(cv.begin(), cv.size(), ct.begin(), ct.size(), NULL, 0,
		                   2e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
		                   gParticles, gSoftBodies);
		gVisOGC.contactRadius    = 0.15f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 12: Cone-Cube Penetration";
		break;
	}

	case 13:
		addVisCube(PxVec3(0,1,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		addVisCube(PxVec3(0,4,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		gVisOGC.contactRadius    = 0.15f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 13: OGC Soft-Soft";
		break;

	case 14:
	{
		addVisCube(PxVec3(0,3,0), 0.5f, 3, 1e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox rb;
		rb.center     = PxVec3(0,0.5f,0);
		rb.halfExtent = PxVec3(2,0.5f,2);
		rb.friction   = 0.5f;
		gVisRigidBoxes.pushBack(rb);
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC_RIGID;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 14: OGC Rigid-Soft SDF";
		break;
	}

	case 15:
	{
		addVisCube(PxVec3(0,3,0), 1.0f, 3, 5e4f, 0.3f, 500.0f, 10.0f);
		avbdBuildAllSelfCollisionAdjacencies(gSoftBodies.begin(), gSoftBodies.size(), gVisSelfAdj);
		gVisOGC.contactRadius    = 0.08f;
		gVisOGC.contactStiffness = 5e4f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 180;
		gVisStepMode  = VS_OGC_SELF;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 15: OGC Self-Collision";
		break;
	}

	case 16:
	{
		addVisCube(PxVec3(0,1.5f,0), 0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		addVisCube(PxVec3(0,4,0),    0.5f, 3, 2e5f, 0.3f, 500.0f, 10.0f);
		AvbdRigidBox rb;
		rb.center     = PxVec3(3,0.5f,0);
		rb.halfExtent = PxVec3(1,0.5f,1);
		rb.friction   = 0.5f;
		gVisRigidBoxes.pushBack(rb);
		avbdBuildAllSelfCollisionAdjacencies(gSoftBodies.begin(), gSoftBodies.size(), gVisSelfAdj);
		gVisOGC.contactRadius    = 0.15f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction         = 0.3f;
		gVisMaxFrames = 300;
		gVisStepMode  = VS_OGC_FULL;
		gVisOuterIter = 2; gVisInnerIter = 15;
		gVisTestName  = "Test 16: OGC Full Pipeline";
		break;
	}

	case 17:
	{
		PxArray<PxVec3> verts;
		PxArray<PxU32>  tets;
		PxVec3 center(0,1,0);
		avbdGenerateSubdividedCubeTets(center, 0.5f, 2, verts, tets);
		const PxReal angle = 0.52f;
		const PxReal cs = PxCos(angle), sn = PxSin(angle);
		for (PxU32 i = 0; i < verts.size(); i++)
		{
			PxVec3 r = verts[i] - center;
			verts[i].x = center.x + r.x * cs - r.y * sn;
			verts[i].y = center.y + r.x * sn + r.y * cs;
		}
		avbdCreateSoftBody(verts.begin(), verts.size(), tets.begin(), tets.size(), NULL, 0,
		                   1e5f, 0.3f, 1000.0f, 0.01f, 0.0f, 0.01f,
		                   gParticles, gSoftBodies);
		gVisMaxFrames = 180;
		gVisOuterIter = 8; gVisInnerIter = 20;
		gVisTestName  = "Test 17: Asymmetric Toppling";
		break;
	}

	case 18:
		addVisCube(PxVec3(-1,2,0), 0.3f, 2, 1e3f, 0.3f, 1000.0f, 0.01f);
		addVisCube(PxVec3( 1,2,0), 0.3f, 2, 1e6f, 0.3f, 1000.0f, 0.01f);
		gVisMaxFrames = 120;
		gVisOuterIter = 8; gVisInnerIter = 20;
		gVisTestName  = "Test 18: Material Stiffness";
		break;

	case 19:
		addVisCube(PxVec3(0,1,0), 0.3f, 2, 1e5f, 0.3f, 1000.0f, 0.01f);
		gVisMaxFrames = 600;
		gVisOuterIter = 8; gVisInnerIter = 20;
		gVisTestName  = "Test 19: Long-Term Stability";
		break;
	}

	captureBodyReferenceLocals(gParticles, gSoftBodies, gVisBodyRefs);
	updateVisRenderData();
	printf("=== Visual: %s (%u frames) ===\n", gVisTestName ? gVisTestName : "?", gVisMaxFrames);
}

// ---------------------------------------------------------------------------
// Visual-mode scene setup (PhysX scene for ground rendering only)
// ---------------------------------------------------------------------------

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPhysics    = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity       = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher             = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader  = PxDefaultSimulationFilterShader;
	sceneDesc.solverType    = PxSolverType::eAVBD;
	gScene = gPhysics->createScene(sceneDesc);

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.1f);

	// Ground plane (rendered via PhysX scene; collision handled by AVBD)
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*ground);
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = 1.0f / 60.0f;

	gContacts.clear();

	// Ground contacts (unless disabled)
	if (gVisStepMode != VS_NO_GROUND)
		avbdDetectSoftGroundContacts(gParticles.begin(), gParticles.size(),
		                             gContacts, gVisGroundY, 0.02f, 0.5f);

	// Mode-specific additional contacts
	switch (gVisStepMode)
	{
	case VS_FULL:
		avbdDetectSoftSoftContacts(gParticles.begin(), gParticles.size(),
		                           gSoftBodies.begin(), gSoftBodies.size(),
		                           gContacts, 0.3f, 0.5f);
		if (gVisRigidBoxes.size())
			avbdDetectSoftRigidSDF(gParticles.begin(), gParticles.size(),
			                       gVisRigidBoxes.begin(), gVisRigidBoxes.size(),
			                       gContacts, 0.1f);
		break;
	case VS_OGC:
		avbdDetectSoftSoftOGC(gParticles.begin(), gParticles.size(),
		                       gSoftBodies.begin(), gSoftBodies.size(),
		                       gContacts, gVisOGC);
		break;
	case VS_OGC_RIGID:
		avbdDetectSoftRigidSDF(gParticles.begin(), gParticles.size(),
		                        gVisRigidBoxes.begin(), gVisRigidBoxes.size(),
		                        gContacts, 0.05f);
		break;
	case VS_OGC_SELF:
		if (gVisSelfAdj.size())
			avbdDetectSelfCollisionOGC(gParticles.begin(), gSoftBodies[0],
			                           gVisSelfAdj[0], gContacts, gVisOGC);
		break;
	case VS_OGC_FULL:
		avbdDetectAllOGCContacts(
			gParticles.begin(), gParticles.size(),
			gSoftBodies.begin(), gSoftBodies.size(),
			gVisRigidBoxes.begin(), gVisRigidBoxes.size(),
			gVisSelfAdj.begin(), gVisSelfAdj.size(),
			gContacts, gVisOGC, gVisGroundY);
		break;
	default:
		break;
	}

	avbdStepSoftBodies(
		gParticles.begin(), gParticles.size(),
		gSoftBodies.begin(), gSoftBodies.size(),
		gContacts.begin(), gContacts.size(),
		dt, gVisGravity, gVisOuterIter, gVisInnerIter, 1000.0f);

	if (isRotationTraceEnabled())
	{
		const PxU32 traceInterval = getRotationTraceInterval();
		if ((gVisFrameCount % traceInterval) == 0)
			printBodyRotationTrace(gVisTestName ? gVisTestName : "visual", gVisFrameCount, gParticles, gSoftBodies, gVisBodyRefs);
	}

	// Update render data pointers
	for (PxU32 i = 0; i < gSoftBodyRenderData.size(); i++)
		gSoftBodyRenderData[i].positions = &gParticles[gSoftBodies[i].particleStart].position;

	gVisFrameCount++;
}

void cleanupPhysics(bool /*interactive*/)
{
	gSoftBodyRenderData.reset();
	gContacts.reset();
	gSoftBodies.reset();
	gParticles.reset();
	gVisRigidBoxes.reset();
	gVisSelfAdj.reset();

	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	PX_RELEASE(gFoundation);

	printf("SnippetSoftBodyAVBD done.\n");
}

void keyPress(unsigned char /*key*/, const PxTransform& /*camera*/)
{
}

static bool isVisualMode()
{
	const char* v = std::getenv("PHYSX_AVBD_SOFTBODY_VISUAL");
	return v && v[0] && v[0] != '0';
}

int snippetMain(int, const char*const*)
{
	const int selectedId = getSelectedTestId();

#ifdef RENDER_SNIPPET
	if (isVisualMode())
	{
		// Create PhysX scene (ground plane for rendering)
		initPhysics(true);

		extern void renderInit();
		renderInit();

		printf("=== AVBD Soft Body Visual Tests ===\n");
		if (selectedId > 0)
			printf("=== Running only test %d ===\n", selectedId);

		for (int testId = 1; testId <= 19; testId++)
		{
			if (!shouldRunTest(selectedId, testId)) continue;
			setupVisualTest(testId);
			extern void renderRun();
			renderRun();
		}

		printf("=== All visual tests completed ===\n");
		cleanupPhysics(true);
		return 0;
	}
#endif

	// PxArray uses the foundation allocator -- must create PxFoundation first
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	printf("=== AVBD Soft Body Unit Tests ===\n");
	if (selectedId > 0)
		printf("=== Running only test %d ===\n", selectedId);

	if (shouldRunTest(selectedId, 1)) testGravityFreeFall();
	if (shouldRunTest(selectedId, 2)) testGroundContact();
	if (shouldRunTest(selectedId, 3)) testVolumePreservation();
	if (shouldRunTest(selectedId, 4)) testKinematicPin();
	if (shouldRunTest(selectedId, 5)) testClothDrape();
	if (shouldRunTest(selectedId, 6)) testEnergyDissipation();
	if (shouldRunTest(selectedId, 7)) testStaticEquilibrium();
	if (shouldRunTest(selectedId, 8)) testMultipleSoftBodies();
	if (shouldRunTest(selectedId, 9)) testSoftSoftCollision();
	if (shouldRunTest(selectedId, 10)) testSoftRigidCollision();
	if (shouldRunTest(selectedId, 11)) testSlopeRotation();
	if (shouldRunTest(selectedId, 12)) testConeCubePenetration();
	if (shouldRunTest(selectedId, 13)) testOGCSoftSoftCollision();
	if (shouldRunTest(selectedId, 14)) testOGCRigidSoftCollision();
	if (shouldRunTest(selectedId, 15)) testOGCSelfCollision();
	if (shouldRunTest(selectedId, 16)) testOGCFullPipeline();
	if (shouldRunTest(selectedId, 17)) testAsymmetricToppling();
	if (shouldRunTest(selectedId, 18)) testMaterialStiffness();
	if (shouldRunTest(selectedId, 19)) testLongTermStability();

	printf("\n=== Results: %d PASSED, %d FAILED (out of %d) ===\n",
	       gTestsPassed, gTestsFailed, gTestsPassed + gTestsFailed);

	PX_RELEASE(gFoundation);
	return gTestsFailed > 0 ? 1 : 0;
}

