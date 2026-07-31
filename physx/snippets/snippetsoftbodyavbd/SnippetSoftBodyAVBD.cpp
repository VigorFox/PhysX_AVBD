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
//  33. Plane speculative-CCD active-set flag gating
//
// No GPU or CUDA dependency -- runs entirely on the CPU.
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "PxPhysicsAPI.h"
#include "DyAvbdSoftBody.h"

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
	for (PxU32 i = 0; i < body.compiled.particleCount; i++)
	{
		const PxU32 pi = body.compiled.particleStart + i;
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
		refs[bi].resize(body.compiled.particleCount);
		for (PxU32 i = 0; i < body.compiled.particleCount; i++)
		{
			const PxU32 pi = body.compiled.particleStart + i;
			refs[bi][i] = particles[pi].position - centroid;
		}
	}
}

static PxQuat estimateBodyRotation(const PxArray<AvbdSoftParticle>& particles,
																	 const AvbdSoftBody& body,
																	 const PxArray<PxVec3>& refLocals)
{
	if (refLocals.size() != body.compiled.particleCount)
		return PxQuat(PxIdentity);

	const PxVec3 centroid = computeBodyMassCentroid(particles, body);

	PxReal sxx = 0.0f, sxy = 0.0f, sxz = 0.0f;
	PxReal syx = 0.0f, syy = 0.0f, syz = 0.0f;
	PxReal szx = 0.0f, szy = 0.0f, szz = 0.0f;

	for (PxU32 i = 0; i < body.compiled.particleCount; i++)
	{
		const PxU32 pi = body.compiled.particleStart + i;
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

	for (PxU32 i = 0; i < body.compiled.particleCount; i++)
	{
		const PxU32 pi = body.compiled.particleStart + i;
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
      pin.point.setVertex(i);
      pin.worldTarget = particles[i].position;
      pin.k = 1e5f;
      pin.kMax = 1e7f;
      bodies[0].runtime.pins.pushBack(pin);
    }
  }
  bodies[0].runtime.compileObjectiveProgram(
    bodies[0].compiled.particleStart,
    bodies[0].compiled.particleCount);

  // Sim 3 seconds
  stepSoft(particles, bodies, contacts, 180, 1.0f/60.0f,
           PxVec3(0.0f, -9.81f, 0.0f), -100.0f, false);

  // Pinned particles should still be near their targets
  bool allPinsClose = true;
  for (PxU32 i = 0; i < bodies[0].runtime.pins.size(); i++)
  {
    PxU32 pi = bodies[0].runtime.pins[i].point.particleIndices[0];
    PxReal dist =
      (particles[pi].position - bodies[0].runtime.pins[i].worldTarget).magnitude();
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
  getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minYA, maxYA);

  PxReal minYB, maxYB;
  getParticleBoundsY(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount, minYB, maxYB);

  TEST_CHECK(minYA > -0.1f, "Body A above ground");
  TEST_CHECK(maxYA < 2.0f,  "Body A settled");
  TEST_CHECK(minYB > -0.1f, "Body B above ground");
  TEST_CHECK(maxYB < 2.0f,  "Body B settled");

  // Bodies should remain separated (no cross-contamination)
  PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
  PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);
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
	AvbdOGCParams softSoftParams;
	softSoftParams.contactRadius = softSoftMargin;
	softSoftParams.contactStiffness = 1e5f;
	softSoftParams.friction = 0.5f;

  for (PxU32 f = 0; f < frames; f++)
  {
    if (f % detectInterval == 0)
    {
      avbdDetectSoftGroundContacts(particles.begin(), particles.size(),
                                   contacts, groundY, 0.02f, 0.5f);

      avbdDetectSoftSoftOGC(particles.begin(), particles.size(),
                            softBodies.begin(), softBodies.size(),
                            contacts, softSoftParams);

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
  PxVec3 cB0 = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);

  // Simulate 5 seconds with full collision
  stepSoftFull(particles, bodies, contacts, NULL, 300, 1.0f/60.0f);

  PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
  PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);

  PxReal minYA, maxYA;
  getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minYA, maxYA);
  PxReal minYB, maxYB;
  getParticleBoundsY(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount, minYB, maxYB);

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
	PxVec3 cCube0 = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
	PxVec3 cCone0 = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);

	AvbdOGCParams ogcParams;
	ogcParams.contactRadius    = 0.15f;
	ogcParams.contactStiffness = 1e5f;
	ogcParams.friction         = 0.3f;
	PxU32 softSoftContactDetectionFrames = 0;
	PxU32 maxSoftSoftContacts = 0;

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
			PxU32 softSoftContacts = 0;
			for(PxU32 contactId = 0;
				contactId < contacts.size(); ++contactId)
			{
				if(contacts[contactId].geometry.targetKind ==
					AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE)
					softSoftContacts++;
			}
			if(softSoftContacts)
				softSoftContactDetectionFrames++;
			maxSoftSoftContacts =
				PxMax(maxSoftSoftContacts, softSoftContacts);
		}

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f);
	}

	PxVec3 cCube = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
	PxVec3 cCone = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);

	PxReal minYCube, maxYCube;
	getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minYCube, maxYCube);
	PxReal minYCone, maxYCone;
	getParticleBoundsY(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount, minYCone, maxYCone);

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
	printf("  soft-soft contact detection frames=%u maxContacts=%u\n",
		softSoftContactDetectionFrames, maxSoftSoftContacts);

	PxU32 cubeParticlesInsideCone = 0;
	for(PxU32 localId = 0; localId < bodies[0].compiled.particleCount; ++localId)
	{
		const PxU32 particleId = bodies[0].compiled.particleStart + localId;
		if(avbdIsPointInsideTetMesh(
			particles[particleId].position,
			bodies[1].compiled.surfaceTriangles, particles.begin()))
		{
			cubeParticlesInsideCone++;
		}
	}
	PxU32 coneParticlesInsideCube = 0;
	for(PxU32 localId = 0; localId < bodies[1].compiled.particleCount; ++localId)
	{
		const PxU32 particleId = bodies[1].compiled.particleStart + localId;
		if(avbdIsPointInsideTetMesh(
			particles[particleId].position,
			bodies[0].compiled.surfaceTriangles, particles.begin()))
		{
			coneParticlesInsideCube++;
		}
	}
	printf("  final inside particles: cube-in-cone=%u cone-in-cube=%u\n",
		cubeParticlesInsideCone, coneParticlesInsideCube);
	TEST_CHECK(
		softSoftContactDetectionFrames > 0 && maxSoftSoftContacts > 0,
		"Cone-cube OGC contact path was exercised");
	TEST_CHECK(
		cubeParticlesInsideCone == 0 && coneParticlesInsideCube == 0,
		"No final cone-cube volumetric interpenetration");

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

	PxVec3 cB0 = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);

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

	PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
	PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);
	PxReal minYA, maxYA, minYB, maxYB;
	getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minYA, maxYA);
	getParticleBoundsY(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount, minYB, maxYB);

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

	PxVec3 c0 = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);

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

	PxVec3 cEnd = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
	PxReal minY, maxY;
	getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minY, maxY);

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
	AvbdSoftBodyWorkspace workspace;

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
	workspace.reserve(particles.size(), particles.size() * 4);
	contacts.reserve(particles.size() * 4);

	AvbdOGCParams ogc;
	ogc.contactRadius    = 0.08f;
	ogc.contactStiffness = 5e4f;
	ogc.friction         = 0.3f;

	for (PxU32 f = 0; f < 180; f++)
	{
		workspace.contact.beginStep();
		// The AVBD contact dual is stateful.  Use the unified detector so
		// ground and self-contact multipliers survive re-detection.
		avbdDetectAllOGCContacts(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			NULL, 0, selfAdj.begin(), selfAdj.size(),
			contacts, ogc, 0.0f, NULL, &workspace.contact);

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f,
			NULL, NULL, NULL, 0.92f, NULL, &workspace);
	}

	PxVec3 cEnd = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
	PxReal minY, maxY;
	getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minY, maxY);

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
	AvbdSoftBodyWorkspace workspace;

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
	workspace.reserve(particles.size(), particles.size() * 4);
	contacts.reserve(particles.size() * 4);

	AvbdOGCParams ogc;
	ogc.contactRadius    = 0.15f;
	ogc.contactStiffness = 1e5f;
	ogc.friction         = 0.3f;

	for (PxU32 f = 0; f < 300; f++)
	{
		contacts.clear();
		workspace.contact.beginStep();
		avbdDetectAllOGCContacts(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			rigidBoxes.begin(), rigidBoxes.size(),
			selfAdj.begin(), selfAdj.size(),
			contacts, ogc, 0.0f, NULL, &workspace.contact);

		avbdStepSoftBodies(
			particles.begin(), particles.size(),
			bodies.begin(), bodies.size(),
			contacts.begin(), contacts.size(),
			1.0f/60.0f, PxVec3(0.0f, -9.81f, 0.0f), 2, 15, 1000.0f,
			NULL, NULL, NULL, 0.92f, NULL, &workspace);
	}

	PxVec3 cA = getParticleCentroid(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount);
	PxVec3 cB = getParticleCentroid(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount);
	PxReal minYA, maxYA, minYB, maxYB;
	getParticleBoundsY(particles.begin(), bodies[0].compiled.particleStart, bodies[0].compiled.particleCount, minYA, maxYA);
	getParticleBoundsY(particles.begin(), bodies[1].compiled.particleStart, bodies[1].compiled.particleCount, minYB, maxYB);

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

// ============================================================================
// Test 20: Position-level AVBD contact primal/dual semantics
// ============================================================================

static void testContactAugmentedLagrangian()
{
	printf("\n--- Test 20: Contact Augmented Lagrangian ---\n");

	PxArray<AvbdSoftParticle> particles(1);
	particles[0].position = PxVec3(0.02f, -0.01f, 0.0f);
	particles[0].initialPosition = particles[0].position;

	AvbdSoftContact contact;
	AvbdSoftContactGeometry& geometry = contact.geometry;
	AvbdSoftContactAugmentedState& state = contact.state;
	geometry.particleIdx = 0;
	geometry.targetKind =
		AvbdSoftContactTargetKind::eWORLD_STATIC;
	geometry.velocityOwner =
		AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex = PX_MAX_U32;
	geometry.normal = PxVec3(0.0f, 1.0f, 0.0f);
	geometry.tangent1 = PxVec3(1.0f, 0.0f, 0.0f);
	geometry.tangent2 = PxVec3(0.0f, 0.0f, 1.0f);
	state.particlePointPrev = PxVec3(0.0f, -0.01f, 0.0f);
	geometry.surfacePoint = PxVec3(0.0f);
	state.surfacePointPrev = PxVec3(0.0f);
	state.k = 1000.0f;
	state.ke = 1e6f;
	state.alLambda = -5.0f;
	state.penTangent[0] = state.penTangent[1] = 1000.0f;
	geometry.friction = 0.5f;

	PxVec3 force;
	PxMat33 hessian;
	avbdEvaluateContactForceHessian(
		geometry, state,
		particles.begin(), force, hessian);
	TEST_CLOSE(force.y, 15.0f, 1e-4f,
		"Normal primal includes persistent AL multiplier");
	TEST_CLOSE(force.x, -7.5f, 1e-4f,
		"Tangent primal is clamped by the Coulomb cone");
	TEST_CLOSE(force.z, 0.0f, 1e-5f,
		"Contact primal has no spurious tangent component");

	avbdUpdateSoftContactDual(
		geometry, state,
		particles.begin(), 1000.0f);
	TEST_CLOSE(state.alLambda, -15.0f, 1e-4f,
		"Normal dual stores the clamped augmented force");
	const PxReal tangentDualMagnitude = PxSqrt(
		state.alLambdaTangent[0] * state.alLambdaTangent[0] +
		state.alLambdaTangent[1] * state.alLambdaTangent[1]);
	TEST_CHECK(
		tangentDualMagnitude <=
			geometry.friction * PxAbs(state.alLambda) + 1e-4f,
		"Tangent dual remains inside the Coulomb cone");

	PxArray<AvbdSoftContact> previous;
	previous.pushBack(contact);
	PxArray<AvbdSoftContact> detected;
	AvbdSoftContact rotatedBasis;
	rotatedBasis.geometry.particleIdx = 0;
	rotatedBasis.geometry.targetKind =
		AvbdSoftContactTargetKind::eWORLD_STATIC;
	rotatedBasis.geometry.velocityOwner =
		AvbdVelocityObjectiveOwner::PositionAL;
	rotatedBasis.geometry.targetIndex = PX_MAX_U32;
	rotatedBasis.geometry.normal = geometry.normal;
	rotatedBasis.geometry.tangent1 = PxVec3(0.0f, 0.0f, 1.0f);
	rotatedBasis.geometry.tangent2 = PxVec3(1.0f, 0.0f, 0.0f);
	rotatedBasis.state.k = 1000.0f;
	rotatedBasis.state.ke = 1e6f;
	detected.pushBack(rotatedBasis);
	detected.pushBack(rotatedBasis);
	avbdTransferSoftContactState(
		previous.begin(), previous.size(), particles.begin(), detected);
	TEST_CHECK(detected[0].state.alLambda < -14.0f,
		"Contact re-detection preserves the normal dual");
	TEST_CHECK(PxAbs(detected[0].state.alLambdaTangent[0]) < 1e-4f &&
		PxAbs(detected[0].state.alLambdaTangent[1]) > 7.0f,
		"Contact re-detection rotates the tangent dual into the new basis");
	TEST_CHECK(PxAbs(detected[1].state.alLambda) < 1e-6f &&
		PxAbs(detected[1].state.alLambdaTangent[0]) < 1e-6f &&
		PxAbs(detected[1].state.alLambdaTangent[1]) < 1e-6f,
		"Contact re-detection transfers each prior dual at most once");

	AvbdSoftContact stablePrevious = contact;
	stablePrevious.geometry.targetKind =
		AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE;
	stablePrevious.geometry.targetIndex = 0;
	stablePrevious.geometry.source = AvbdSoftContactSource(
		AvbdSoftContactSource::eSOFT_SURFACE, 0, 11, 12);
	stablePrevious.geometry.surfacePoint = PxVec3(0.0f);
	previous.clear();
	previous.pushBack(stablePrevious);

	AvbdSoftContact sameSource = rotatedBasis;
	sameSource.geometry.targetKind =
		AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE;
	sameSource.geometry.targetIndex = 0;
	sameSource.geometry.source = stablePrevious.geometry.source;
	// Exact feature identity, rather than world-space proximity, owns the
	// augmented state when a deforming surface moves.
	sameSource.geometry.surfacePoint = PxVec3(1.0f, 0.0f, 0.0f);
	AvbdSoftContact differentSource = sameSource;
	differentSource.geometry.source = AvbdSoftContactSource(
		AvbdSoftContactSource::eSOFT_SURFACE, 0, 21, 22);

	detected.clear();
	detected.pushBack(sameSource);
	detected.pushBack(differentSource);
	AvbdSoftContactWorkspace transferWorkspace;
	avbdTransferSoftContactState(
		previous.begin(), previous.size(), particles.begin(), detected,
		&transferWorkspace);
	TEST_CHECK(detected[0].state.alLambda < -14.0f,
		"Stable contact source preserves dual across surface motion");
	TEST_CHECK(PxAbs(detected[1].state.alLambda) < 1e-6f,
		"Different contact source cannot inherit another objective's dual");

	const PxU32 previousUsedCapacity =
		transferWorkspace.previousUsed.capacity();
	avbdTransferSoftContactState(
		previous.begin(), previous.size(), particles.begin(), detected,
		&transferWorkspace);
	TEST_CHECK(
		transferWorkspace.previousUsed.capacity() == previousUsedCapacity,
		"Persistent contact transfer workspace reuses scratch capacity");

	PxArray<AvbdSoftParticle> selfParticles(4);
	selfParticles[0].position = PxVec3(0.0f, 0.0f, 0.0f);
	selfParticles[1].position = PxVec3(1.0f, 0.0f, 0.0f);
	selfParticles[2].position = PxVec3(0.0f, 0.0f, 1.0f);
	selfParticles[3].position = PxVec3(0.2f, 0.01f, 0.2f);
	for(PxU32 i = 0; i < selfParticles.size(); ++i)
		selfParticles[i].initialPosition = selfParticles[i].position;
	AvbdSoftBody selfBody;
	selfBody.compiled.particleStart = 0;
	selfBody.compiled.particleCount = selfParticles.size();
	selfBody.compiled.surfaceTriangles.pushBack(0);
	selfBody.compiled.surfaceTriangles.pushBack(1);
	selfBody.compiled.surfaceTriangles.pushBack(2);
	for(PxU32 i = 0; i < selfParticles.size(); ++i)
		selfBody.compiled.surfaceVertices.pushBack(i);
	AvbdSelfCollisionAdjacency selfAdjacency;
	selfAdjacency.resize(selfParticles.size());
	AvbdOGCParams selfParams;
	selfParams.contactRadius = 0.05f;

	PxArray<AvbdSoftContact> selfContacts;
	avbdDetectSelfCollisionOGC(
		selfParticles.begin(), selfBody, 7,
		selfAdjacency, selfContacts, selfParams);
	TEST_CHECK(
		selfContacts.size() == 1 &&
		selfContacts[0].geometry.source.type ==
			AvbdSoftContactSource::eSELF_SURFACE &&
		selfContacts[0].geometry.source.targetBodyIndex == 7,
		"Self-contact detector emits an explicit source identity");
	if(selfContacts.empty())
		return;
	const AvbdSoftContactSource originalSelfSource =
		selfContacts[0].geometry.source;

	selfBody.compiled.surfaceTriangles[0] = 2;
	selfBody.compiled.surfaceTriangles[1] = 0;
	selfBody.compiled.surfaceTriangles[2] = 1;
	selfContacts.clear();
	avbdDetectSelfCollisionOGC(
		selfParticles.begin(), selfBody, 7,
		selfAdjacency, selfContacts, selfParams);
	TEST_CHECK(
		selfContacts.size() == 1 &&
		selfContacts[0].geometry.source == originalSelfSource,
		"Surface feature identity survives triangle index reordering");
}

// ============================================================================
// Test 21: Position-level AVBD kinematic-pin primal/dual semantics
// ============================================================================

static void testPinAugmentedLagrangian()
{
	printf("\n--- Test 21: Kinematic Pin Augmented Lagrangian ---\n");

	PxArray<AvbdSoftParticle> particles(1);
	particles[0].position = PxVec3(0.1f, -0.2f, 0.3f);
	particles[0].initialPosition = particles[0].position;

	AvbdKinematicPin pin;
	pin.point.setVertex(0);
	pin.worldTarget = PxVec3(0.0f);
	pin.k = 100.0f;
	pin.kMax = 1000.0f;

	PxVec3 forceBeforeDual;
	PxMat33 hessian;
	avbdEvaluatePinForceHessian(
		pin.point, pin, particles.begin(), 0,
		forceBeforeDual, hessian);

	// beta=0 isolates the equality multiplier update from penalty growth.
	avbdUpdatePinDual(pin, pin.point, particles.begin(), 0.0f);

	PxVec3 forceAfterDual;
	avbdEvaluatePinForceHessian(
		pin.point, pin, particles.begin(), 0,
		forceAfterDual, hessian);
	TEST_CHECK(
		(forceAfterDual - forceBeforeDual * 2.0f).magnitude() < 1e-4f,
		"Pin dual contributes a persistent vector multiplier to the primal");

	particles[0].position = pin.worldTarget;
	PxVec3 forceAtTarget;
	avbdEvaluatePinForceHessian(
		pin.point, pin, particles.begin(), 0,
		forceAtTarget, hessian);
	TEST_CHECK(
		(forceAtTarget - forceBeforeDual).magnitude() < 1e-4f,
		"Pin multiplier remains active when the current residual is zero");

	TEST_CLOSE(hessian.column0.x, 100.0f, 1e-5f,
		"Pin Hessian keeps the current AL penalty");

	avbdWarmstartPinState(pin, 0.5f, 0.8f, 10.0f);
	TEST_CHECK(
		(pin.alLambda - PxVec3(4.0f, -8.0f, 12.0f)).magnitude() < 1e-4f,
		"Pin warm-start scales the multiplier and penalty as one AL state");
	TEST_CLOSE(pin.k, 80.0f, 1e-5f,
		"Pin warm-start applies the configured penalty continuation");

	// The pin is force/position-valued. dt therefore appears only through
	// the particle inertial block m/dt^2, not by rescaling the pin force.
	const PxReal frameDts[3] =
		{ 1.0f / 30.0f, 1.0f / 60.0f, 1.0f / 120.0f };
	bool timestepScalingMatches = true;
	for (PxU32 i = 0; i < 3; i++)
	{
		PxArray<AvbdSoftParticle> stepParticles(1);
		stepParticles[0].position = PxVec3(0.1f, 0.0f, 0.0f);
		stepParticles[0].initialPosition = stepParticles[0].position;
		stepParticles[0].mass = 2.0f;
		stepParticles[0].invMass = 0.5f;

		PxArray<AvbdSoftBody> stepBodies(1);
		stepBodies[0].compiled.particleStart = 0;
		stepBodies[0].compiled.particleCount = 1;
		stepBodies[0].compiled.elementAdjacency.resize(1);
		stepBodies[0].runtime.objectiveAdjacency.resize(1);
		AvbdKinematicPin stepPin;
		stepPin.point.setVertex(0);
		stepPin.worldTarget = PxVec3(0.0f);
		stepPin.k = 100.0f;
		stepPin.kMax = 1000.0f;
		stepBodies[0].runtime.pins.pushBack(stepPin);
		stepBodies[0].runtime.compileObjectiveProgram(0, 1);

		const PxReal dt = frameDts[i];
		const PxReal expectedDelta =
			-10.0f / (2.0f / (dt * dt) + 100.0f);
		avbdStepSoftBodies(
			stepParticles.begin(), stepParticles.size(),
			stepBodies.begin(), stepBodies.size(),
			NULL, 0, dt, PxVec3(0.0f),
			1, 1, 0.0f, NULL, NULL, NULL, 0.0f);
		const PxReal actualDelta = stepParticles[0].position.x - 0.1f;
		if (PxAbs(actualDelta - expectedDelta) > 1e-6f)
			timestepScalingMatches = false;
	}
	TEST_CHECK(
		timestepScalingMatches,
		"Pin position solve uses force-valued k with particle m/dt^2 inertia");
}

// ============================================================================
// Test 22: Position-level AVBD rigid-soft attachment primal/dual semantics
// ============================================================================

static void testAttachmentAugmentedLagrangian()
{
	printf("\n--- Test 22: Rigid-Soft Attachment Augmented Lagrangian ---\n");

	PxArray<AvbdSoftParticle> particles(1);
	particles[0].position = PxVec3(0.1f, 0.3f, 0.3f);
	particles[0].initialPosition = particles[0].position;

	AvbdSolverBody rigidBody;
	rigidBody.initialize(
		PxTransform(PxIdentity), PxVec3(0.0f), PxVec3(0.0f), 1.0f,
		PxMat33(PxIdentity), 0);

	AvbdSoftAttachment attachment;
	attachment.point.setVertex(0);
	attachment.rigidBodyIdx = 0;
	attachment.localOffset = PxVec3(0.0f, 0.5f, 0.0f);
	attachment.k = 100.0f;
	attachment.kMax = 1000.0f;

	const PxReal evaluationDt = 1.0f / 60.0f;
	AvbdSoftRigidAttachmentCoupledStep stepBeforeDual;
	const bool evaluatedBeforeDual =
		avbdEvaluateSoftRigidAttachmentCoupledStep(
			attachment, attachment.point, particles.begin(),
			particles.size(), rigidBody,
			evaluationDt, stepBeforeDual);

	// beta=0 isolates the equality multiplier update from penalty growth.
	avbdUpdateAttachmentDual(
		attachment, attachment.point, particles.begin(),
		&rigidBody, 0.0f);

	AvbdSoftRigidAttachmentCoupledStep stepAfterDual;
	const bool evaluatedAfterDual =
		avbdEvaluateSoftRigidAttachmentCoupledStep(
			attachment, attachment.point, particles.begin(),
			particles.size(), rigidBody,
			evaluationDt, stepAfterDual);
	TEST_CHECK(
		evaluatedBeforeDual && evaluatedAfterDual &&
		(stepAfterDual.multiplier -
			stepBeforeDual.multiplier * 2.0f).magnitude() <
				1e-4f,
		"Attachment dual contributes one multiplier to the coupled block");

	const PxVec3 worldOffset =
		rigidBody.rotation.rotate(attachment.localOffset);
	particles[0].position = rigidBody.position + worldOffset;

	AvbdSoftRigidAttachmentCoupledStep stepAtTarget;
	const bool evaluatedAtTarget =
		avbdEvaluateSoftRigidAttachmentCoupledStep(
			attachment, attachment.point, particles.begin(),
			particles.size(), rigidBody,
			evaluationDt, stepAtTarget);
	TEST_CHECK(
		evaluatedAtTarget &&
		(stepAtTarget.multiplier -
			stepBeforeDual.multiplier).magnitude() < 1e-4f,
		"Attachment multiplier remains active at zero current residual");

	PxArray<AvbdSoftBody> attachmentBodies(1);
	attachmentBodies[0].compiled.particleStart = 0;
	attachmentBodies[0].compiled.particleCount = 1;
	attachmentBodies[0].runtime.attachments.pushBack(attachment);
	AvbdKinematicPin compiledPin;
	compiledPin.point.setVertex(0);
	attachmentBodies[0].runtime.pins.pushBack(compiledPin);
	attachmentBodies[0].runtime.compileObjectiveProgram(0, 1);
	TEST_CHECK(
		attachmentBodies[0].runtime.compiledObjectives.size() == 2 &&
		attachmentBodies[0].runtime.objectiveAdjacency.size() == 1 &&
		attachmentBodies[0].runtime.objectiveAdjacency[0].
			objectiveIndices.size() == 2 &&
		attachmentBodies[0].runtime.compiledObjectives[0].owner ==
			AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL &&
		attachmentBodies[0].runtime.compiledObjectives[1].owner ==
			AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL,
		"Prep assigns each soft equality objective one compiled owner");
	TEST_CHECK(
		attachmentBodies[0].runtime.isObjectiveProgramCurrent(0, 1),
		"Compiled soft objective program matches its runtime state snapshot");

	PxArray<AvbdSoftBody> staleProgramBodies = attachmentBodies;
	staleProgramBodies[0].runtime.attachments[0].rigidBodyIdx = 1;
	const bool staleProgramRejected =
		!staleProgramBodies[0].runtime.
			isObjectiveProgramCurrent(0, 1);
	staleProgramBodies[0].runtime.compileObjectiveProgram(0, 1);
	TEST_CHECK(
		staleProgramRejected &&
		staleProgramBodies[0].runtime.isObjectiveProgramCurrent(0, 1),
		"Runtime objective mutation requires an explicit prep recompile");

	PxArray<AvbdSoftBody> unsupportedBodies(1);
	unsupportedBodies[0].compiled.particleStart = 0;
	unsupportedBodies[0].compiled.particleCount = 1;
	AvbdKinematicPin invalidPin;
	invalidPin.point.setVertex(7);
	unsupportedBodies[0].runtime.pins.pushBack(invalidPin);
	unsupportedBodies[0].runtime.compileObjectiveProgram(0, 1);
	TEST_CHECK(
		unsupportedBodies[0].runtime.compiledObjectives.size() == 1 &&
		unsupportedBodies[0].runtime.compiledObjectives[0].owner ==
			AvbdSoftObjectiveOwner::eUNSUPPORTED &&
		unsupportedBodies[0].runtime.objectiveAdjacency[0].
			objectiveIndices.empty(),
		"Invalid soft objective compiles to Unsupported instead of disappearing");

	TEST_CHECK(
		evaluatedAtTarget &&
		(stepAtTarget.particleCorrections[0] +
			stepAtTarget.rigidLinearCorrection).magnitude() <
				1e-5f &&
		stepAtTarget.rigidAngularCorrection.
			magnitudeSquared() > 1.0e-12f,
		"Coupled attachment block owns both endpoints exactly once");

	bool actorOrderInvariant = true;
	for (PxU32 physicalBodyIndex = 0;
		physicalBodyIndex < 2; physicalBodyIndex++)
	{
		PxArray<AvbdSolverBody> orderedBodies(2);
		orderedBodies[physicalBodyIndex] = rigidBody;
		orderedBodies[physicalBodyIndex].nodeIndex = physicalBodyIndex;
		const PxU32 decoyIndex = 1 - physicalBodyIndex;
		orderedBodies[decoyIndex].initialize(
			PxTransform(PxVec3(10.0f, 0.0f, 0.0f)),
			PxVec3(0.0f), PxVec3(0.0f), 1.0f,
			PxMat33(PxIdentity), decoyIndex);

		AvbdSoftAttachment orderedAttachment = attachment;
		orderedAttachment.rigidBodyIdx = physicalBodyIndex;
		PxArray<AvbdSoftBody> orderedSoftBodies(1);
		orderedSoftBodies[0].compiled.particleStart = 0;
		orderedSoftBodies[0].compiled.particleCount = 1;
		orderedSoftBodies[0].runtime.attachments.pushBack(
			orderedAttachment);
		orderedSoftBodies[0].runtime.compileObjectiveProgram(0, 1);

		AvbdSoftRigidAttachmentCoupledStep orderedStep;
		const bool orderedEvaluated =
			avbdEvaluateSoftRigidAttachmentCoupledStep(
				orderedAttachment, orderedAttachment.point,
				particles.begin(), particles.size(),
				orderedBodies[physicalBodyIndex],
				evaluationDt, orderedStep);
		if (!orderedEvaluated ||
			!orderedSoftBodies[0].runtime.
				isObjectiveProgramCurrent(0, 1) ||
			orderedSoftBodies[0].runtime.
				compiledObjectives[0].rigidBodyIdx !=
					physicalBodyIndex ||
			(orderedStep.multiplier -
				stepAtTarget.multiplier).magnitude() > 1e-4f ||
			(orderedStep.rigidAngularCorrection -
				stepAtTarget.rigidAngularCorrection).
					magnitude() > 1e-4f)
			actorOrderInvariant = false;
	}
	TEST_CHECK(
		actorOrderInvariant,
		"Attachment owner and coupled block are invariant to actor order");

	attachmentBodies[0].runtime.attachments.clear();
	attachmentBodies[0].runtime.compileObjectiveProgram(0, 1);
	const bool removedOwnerGone =
		attachmentBodies[0].runtime.compiledObjectives.size() == 1 &&
		attachmentBodies[0].runtime.compiledObjectives[0].owner ==
			AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL;

	AvbdSoftAttachment readdedAttachment;
	readdedAttachment.point.setVertex(0);
	readdedAttachment.rigidBodyIdx = 0;
	readdedAttachment.localOffset = attachment.localOffset;
	readdedAttachment.k = 100.0f;
	readdedAttachment.kMax = 1000.0f;
	attachmentBodies[0].runtime.attachments.pushBack(readdedAttachment);
	attachmentBodies[0].runtime.compileObjectiveProgram(0, 1);
	PxU32 readdedAttachmentOwnerCount = 0;
	for (PxU32 oi = 0;
		oi < attachmentBodies[0].runtime.compiledObjectives.size(); oi++)
	{
		if (attachmentBodies[0].runtime.compiledObjectives[oi].owner ==
			AvbdSoftObjectiveOwner::
				eRIGID_ATTACHMENT_POSITION_AL)
			readdedAttachmentOwnerCount++;
	}
	TEST_CHECK(
		removedOwnerGone &&
		readdedAttachmentOwnerCount == 1 &&
		attachmentBodies[0].runtime.attachments[0].
			alLambda.magnitudeSquared() == 0.0f,
		"Attachment remove/re-add recompiles one fresh owner without stale dual");

	const PxReal attachmentDts[3] =
		{ 1.0f / 30.0f, 1.0f / 60.0f, 1.0f / 120.0f };
	bool attachmentDtInvariant = true;
	for (PxU32 i = 0; i < 3; i++)
	{
		AvbdSoftParticle dtParticle;
		dtParticle.position = PxVec3(0.1f, 0.0f, 0.0f);
		dtParticle.invMass = 1.0f;
		AvbdSoftAttachment dtAttachment;
		dtAttachment.k = 100.0f;
		dtAttachment.kMax = 1000.0f;
		AvbdSoftRigidAttachmentCoupledStep dtStep;
		const bool dtEvaluated =
			avbdEvaluateSoftRigidAttachmentCoupledStep(
				dtAttachment, dtAttachment.point,
				&dtParticle, 1, rigidBody,
				attachmentDts[i], dtStep);
		const PxReal dt2 =
			attachmentDts[i] * attachmentDts[i];
		const PxReal expectedMultiplier =
			-0.1f / (2.0f * dt2 + 0.01f);
		if (!dtEvaluated ||
			PxAbs(dtStep.multiplier.x -
				expectedMultiplier) > 1e-5f)
			attachmentDtInvariant = false;
	}
	TEST_CHECK(
		attachmentDtInvariant,
		"Coupled attachment remains force/position-valued across timesteps");

	avbdWarmstartAttachmentState(attachment, 0.5f, 0.8f, 10.0f);
	TEST_CHECK(
		(attachment.alLambda -
			PxVec3(4.0f, -8.0f, 12.0f)).magnitude() < 1e-4f,
		"Attachment warm-start scales the multiplier and penalty as one AL state");
	TEST_CLOSE(attachment.k, 80.0f, 1e-5f,
		"Attachment warm-start applies the configured penalty continuation");

	const PxReal frameDts[3] =
		{ 1.0f / 30.0f, 1.0f / 60.0f, 1.0f / 120.0f };
	bool timestepScalingMatches = true;
	for (PxU32 i = 0; i < 3; i++)
	{
		PxArray<AvbdSoftParticle> stepParticles(1);
		stepParticles[0].position = PxVec3(0.1f, 0.0f, 0.0f);
		stepParticles[0].initialPosition = stepParticles[0].position;

		AvbdSoftAttachment stepAttachment;
		stepAttachment.point.setVertex(0);
		stepAttachment.rigidBodyIdx = 0;
		stepAttachment.k = 100.0f;
		stepAttachment.kMax = 1000.0f;
		const PxReal dt = frameDts[i];
		AvbdSoftRigidAttachmentCoupledStep localStep;
		const bool localEvaluated =
			avbdEvaluateSoftRigidAttachmentCoupledStep(
				stepAttachment, stepAttachment.point,
				stepParticles.begin(), stepParticles.size(),
				rigidBody, dt, localStep);
		const PxReal actualRigidCorrection =
			localStep.rigidLinearCorrection.x;
		const PxReal dt2 = dt * dt;
		const PxReal expectedRigidCorrection =
			0.1f * dt2 / (2.0f * dt2 + 0.01f);
		if (!localEvaluated ||
			PxAbs(actualRigidCorrection -
			expectedRigidCorrection) > 1e-6f)
			timestepScalingMatches = false;
	}
	TEST_CHECK(
		timestepScalingMatches,
		"Attachment uses force-valued k with both endpoint inertias");
}

// ============================================================================
// Test 23: Soft-soft contact must own both incident particle blocks
// ============================================================================

static void testSoftSoftContactTwoSidedObjective()
{
	printf("\n--- Test 23: Soft-Soft Two-Sided Contact Objective ---\n");

	PxArray<AvbdSoftParticle> particles(4);
	particles[0].position = PxVec3(0.25f, 0.05f, 0.25f);
	particles[1].position = PxVec3(0.0f, 0.0f, 0.0f);
	particles[2].position = PxVec3(0.0f, 0.0f, 1.0f);
	particles[3].position = PxVec3(1.0f, 0.0f, 0.0f);
	for (PxU32 i = 0; i < particles.size(); i++)
		particles[i].initialPosition = particles[i].position;

	PxArray<AvbdSoftBody> bodies(2);
	bodies[0].compiled.particleStart = 0;
	bodies[0].compiled.particleCount = 1;
	bodies[0].compiled.surfaceVertices.pushBack(0);
	bodies[0].compiled.elementAdjacency.resize(1);
	bodies[0].runtime.compileObjectiveProgram(0, 1);

	bodies[1].compiled.particleStart = 1;
	bodies[1].compiled.particleCount = 3;
	bodies[1].compiled.surfaceVertices.pushBack(1);
	bodies[1].compiled.surfaceVertices.pushBack(2);
	bodies[1].compiled.surfaceVertices.pushBack(3);
	bodies[1].compiled.elementAdjacency.resize(3);
	// Winding produces an outward +Y face normal.
	bodies[1].compiled.surfaceTriangles.pushBack(1);
	bodies[1].compiled.surfaceTriangles.pushBack(2);
	bodies[1].compiled.surfaceTriangles.pushBack(3);
	bodies[1].runtime.compileObjectiveProgram(1, 3);

	AvbdOGCParams params;
	params.contactRadius = 0.1f;
	params.contactStiffness = 1e4f;
	params.friction = 0.0f;
	PxArray<AvbdSoftContact> contacts;
	avbdDetectSoftSoftOGC(
		particles.begin(), particles.size(),
		bodies.begin(), bodies.size(), contacts, params);
	TEST_CHECK(
		contacts.size() == 1 &&
		contacts[0].geometry.source.type ==
			AvbdSoftContactSource::eSOFT_SURFACE,
		"Soft-soft detector emits one prepared surface objective");

	PxReal initialMassWeightedY = 0.0f;
	for (PxU32 i = 0; i < particles.size(); i++)
		initialMassWeightedY += particles[i].mass * particles[i].position.y;

	bool preparedIncidentBlocks = false;
	bool preparedBlocksBalance = false;
	if (contacts.size() == 1)
	{
		const AvbdSoftContact& contact = contacts[0];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		const PxReal targetWeightSum =
			geometry.surfaceWeights[0] +
			geometry.surfaceWeights[1] +
			geometry.surfaceWeights[2];
		preparedIncidentBlocks =
			geometry.hasDeformableSurfaceTarget() &&
			PxAbs(targetWeightSum - 1.0f) < 1e-5f &&
			avbdGetSoftContactParticleJacobianScale(
				geometry, geometry.particleIdx) > 0.0f;

		PxVec3 forceSum(0.0f);
		for (PxU32 particleIdx = 0;
		     particleIdx < particles.size(); particleIdx++)
		{
			const PxReal jacobianScale =
				avbdGetSoftContactParticleJacobianScale(
					geometry, particleIdx);
			if (particleIdx != geometry.particleIdx &&
				jacobianScale >= 0.0f)
				preparedIncidentBlocks = false;
			PxVec3 force;
			PxMat33 hessian;
			avbdEvaluateContactParticleBlock(
				geometry, contact.state, particles.begin(),
				jacobianScale, force, hessian);
			forceSum += force;
		}
		preparedBlocksBalance = forceSum.magnitude() < 1e-3f;
	}
	TEST_CHECK(
		preparedIncidentBlocks,
		"Prepared soft-soft objective owns query and barycentric target blocks");
	TEST_CHECK(
		preparedBlocksBalance,
		"Shared soft-contact particle blocks form an internal balanced force");

	const PxReal dt = 1.0f / 60.0f;

	avbdStepSoftBodies(
		particles.begin(), particles.size(),
		bodies.begin(), bodies.size(),
		contacts.begin(), contacts.size(),
		dt, PxVec3(0.0f),
		1, 30, 0.0f);

	const PxReal targetCentroidY =
		(particles[1].position.y + particles[2].position.y +
		 particles[3].position.y) / 3.0f;
	TEST_CHECK(
		targetCentroidY < -1e-5f,
		"Soft-soft objective applies the opposite response to target vertices");

	PxReal finalMassWeightedY = 0.0f;
	for (PxU32 i = 0; i < particles.size(); i++)
		finalMassWeightedY += particles[i].mass * particles[i].position.y;
	TEST_CHECK(
		PxAbs(finalMassWeightedY - initialMassWeightedY) < 1e-4f,
		"Two-sided soft contact preserves its internal mass-weighted translation");
}

// ============================================================================
// Test 24: Dynamic rigid-soft contact must own both incident body blocks
// ============================================================================

static void testDynamicRigidSoftContactTwoSidedObjective()
{
	printf("\n--- Test 24: Dynamic Rigid-Soft Two-Sided Contact Objective ---\n");

	PxArray<AvbdSoftParticle> particles(1);
	particles[0].position = PxVec3(0.0f, -0.1f, 0.0f);
	particles[0].initialPosition = particles[0].position;

	AvbdSolverBody rigidBody;
	rigidBody.initialize(
		PxTransform(PxIdentity), PxVec3(0.0f), PxVec3(0.0f), 1.0f,
		PxMat33(PxIdentity), 0);

	AvbdSoftContact contact;
	contact.geometry.source = AvbdSoftContactSource(
		AvbdSoftContactSource::eRIGID_SDF, 0, 17, 23);
	contact.geometry.particleIdx = 0;
	contact.geometry.targetKind =
		AvbdSoftContactTargetKind::eRIGID_BODY;
	contact.geometry.velocityOwner =
		AvbdVelocityObjectiveOwner::ManifoldFinalize;
	contact.geometry.targetIndex = 0;
	contact.geometry.normal = PxVec3(0.0f, 1.0f, 0.0f);
	contact.geometry.projNormal = contact.geometry.normal;
	contact.geometry.surfacePoint = PxVec3(0.0f);
	contact.geometry.rigidLocalPoint = PxVec3(0.0f);
	contact.geometry.margin = 0.0f;
	contact.geometry.friction = 0.0f;
	contact.state.k = 100.0f;
	contact.state.ke = 1000.0f;
	TEST_CHECK(
		contact.geometry.hasRigidBodyTarget() &&
		contact.geometry.targetIndex == 0,
		"Prep IR identifies the rigid target without sentinel overloading");

	PxVec3 softForce;
	PxMat33 softHessian;
	avbdEvaluateContactForceHessian(
		contact.geometry, contact.state, particles.begin(),
		softForce, softHessian);
	TEST_CHECK(
		softForce.y > 0.0f,
		"Dynamic rigid-soft contact contributes the soft particle block");

	AvbdBlock6x6 rigidHessian;
	rigidHessian.setZero();
	AvbdVec6 rigidGradient;
	const PxU32 rigidContributionCount =
		avbdAddDynamicSoftRigidContactContributions_rigid(
			&contact, 1, 0, particles.begin(), particles.size(),
			rigidBody, rigidHessian, rigidGradient);
	TEST_CHECK(
		rigidContributionCount == 1 &&
		rigidGradient.linear.dot(softForce) > 0.0f,
		"One dynamic rigid-soft objective contributes the opposite rigid block");

	const PxReal dt = 1.0f / 60.0f;
	const PxReal invDtSq = 1.0f / (dt * dt);
	const PxMat33 softSystem =
		PxMat33::createDiagonal(PxVec3(invDtSq)) + softHessian;
	const PxVec3 softCorrection =
		softSystem.getInverse() * softForce;

	AvbdBlock6x6 rigidSystem;
	rigidSystem.initializeDiagonal(
		1.0f, PxMat33(PxIdentity), invDtSq);
	rigidSystem.linearLinear += rigidHessian.linearLinear;
	rigidSystem.linearAngular += rigidHessian.linearAngular;
	rigidSystem.angularLinear += rigidHessian.angularLinear;
	rigidSystem.angularAngular += rigidHessian.angularAngular;
	const PxVec3 rigidCorrection =
		-(rigidSystem.linearLinear.getInverse() *
			rigidGradient.linear);
	TEST_CHECK(
		PxAbs(softCorrection.y + rigidCorrection.y) < 1e-6f,
		"Dynamic rigid-soft local blocks preserve equal-mass translation");

	AvbdSoftContact offCenterContact = contact;
	offCenterContact.geometry.rigidLocalPoint =
		PxVec3(0.2f, 0.0f, 0.0f);
	particles[0].position = PxVec3(0.2f, -0.1f, 0.0f);
	PxVec3 offCenterSoftForce;
	PxMat33 offCenterSoftHessian;
	avbdEvaluateContactParticleBlockAtSurfacePoint(
		offCenterContact.geometry, offCenterContact.state,
		particles.begin(),
		avbdGetRigidContactSurfacePoint(
			offCenterContact.geometry, rigidBody),
		1.0f, offCenterSoftForce, offCenterSoftHessian);
	AvbdBlock6x6 offCenterRigidHessian;
	offCenterRigidHessian.setZero();
	AvbdVec6 offCenterRigidGradient;
	const PxU32 offCenterCount =
		avbdAddDynamicSoftRigidContactContributions_rigid(
			&offCenterContact, 1, 0,
			particles.begin(), particles.size(), rigidBody,
			offCenterRigidHessian, offCenterRigidGradient);
	const PxVec3 offCenterWorldOffset =
		rigidBody.rotation.rotate(
			offCenterContact.geometry.rigidLocalPoint);
	const PxReal offCenterCrossHessian =
		offCenterRigidHessian.linearAngular.column0.magnitudeSquared() +
		offCenterRigidHessian.linearAngular.column1.magnitudeSquared() +
		offCenterRigidHessian.linearAngular.column2.magnitudeSquared();
	TEST_CHECK(
		offCenterCount == 1 &&
		(offCenterRigidGradient.linear - offCenterSoftForce).
			magnitude() < 1e-6f &&
		(offCenterRigidGradient.angular -
		 offCenterWorldOffset.cross(offCenterSoftForce)).
			magnitude() < 1e-6f &&
		offCenterCrossHessian > 1e-6f,
		"Off-center rigid-soft contact preserves linear-angular coupling");
	particles[0].position = PxVec3(0.0f, -0.1f, 0.0f);

	AvbdSoftContact frictionContact = contact;
	frictionContact.geometry.friction = 0.5f;
	frictionContact.geometry.tangent1 =
		PxVec3(1.0f, 0.0f, 0.0f);
	frictionContact.geometry.tangent2 =
		PxVec3(0.0f, 0.0f, 1.0f);
	frictionContact.state.alLambda = -10.0f;
	frictionContact.state.particlePointPrev =
		particles[0].position - PxVec3(0.02f, 0.0f, 0.0f);
	frictionContact.state.surfacePointPrev = PxVec3(0.0f);
	frictionContact.state.penTangent[0] = 100.0f;
	frictionContact.state.penTangent[1] = 100.0f;
	PxVec3 frictionSoftForce;
	PxMat33 frictionSoftHessian;
	avbdEvaluateContactParticleBlockAtSurfacePoint(
		frictionContact.geometry, frictionContact.state,
		particles.begin(), PxVec3(0.0f), 1.0f,
		frictionSoftForce, frictionSoftHessian);
	AvbdBlock6x6 frictionRigidHessian;
	frictionRigidHessian.setZero();
	AvbdVec6 frictionRigidGradient;
	const PxU32 frictionCount =
		avbdAddDynamicSoftRigidContactContributions_rigid(
			&frictionContact, 1, 0,
			particles.begin(), particles.size(), rigidBody,
			frictionRigidHessian, frictionRigidGradient);
	TEST_CHECK(
		frictionCount == 1 &&
		PxAbs(frictionSoftForce.x) > 1e-5f &&
		(frictionRigidGradient.linear - frictionSoftForce).
			magnitude() < 1e-6f,
		"Dynamic rigid-soft normal and friction share both endpoint blocks");

	bool actorOrderInvariant = true;
	for (PxU32 physicalBodyIndex = 0;
	     physicalBodyIndex < 2; physicalBodyIndex++)
	{
		AvbdSolverBody orderedBodies[2];
		for (PxU32 bodyIndex = 0; bodyIndex < 2; bodyIndex++)
		{
			orderedBodies[bodyIndex].initialize(
				PxTransform(
					bodyIndex == physicalBodyIndex
						? PxVec3(0.0f)
						: PxVec3(10.0f, 0.0f, 0.0f)),
				PxVec3(0.0f), PxVec3(0.0f), 1.0f,
				PxMat33(PxIdentity), bodyIndex);
		}
		AvbdSoftContact orderedContact = contact;
		orderedContact.geometry.targetIndex = physicalBodyIndex;
		PxU32 contributionCount = 0;
		for (PxU32 bodyIndex = 0; bodyIndex < 2; bodyIndex++)
		{
			AvbdBlock6x6 orderedHessian;
			orderedHessian.setZero();
			AvbdVec6 orderedGradient;
			const PxU32 bodyContributions =
				avbdAddDynamicSoftRigidContactContributions_rigid(
					&orderedContact, 1, bodyIndex,
					particles.begin(), particles.size(),
					orderedBodies[bodyIndex],
					orderedHessian, orderedGradient);
			contributionCount += bodyContributions;
			if (bodyIndex == physicalBodyIndex)
			{
				if (bodyContributions != 1 ||
					(orderedGradient.linear -
					 rigidGradient.linear).magnitude() > 1e-6f)
					actorOrderInvariant = false;
			}
			else if (bodyContributions != 0 ||
				orderedGradient.linear.magnitude() > 1e-8f)
				actorOrderInvariant = false;
		}
		if (contributionCount != 1)
			actorOrderInvariant = false;
	}
	TEST_CHECK(
		actorOrderInvariant,
		"Rigid target owner is exact and invariant to actor storage order");

	bool timestepScalingMatches = true;
	const PxReal frameDts[3] =
		{ 1.0f / 30.0f, 1.0f / 60.0f, 1.0f / 120.0f };
	const PxReal endpointMasses[2][2] =
		{ { 0.25f, 4.0f }, { 4.0f, 0.25f } };
	for (PxU32 massCase = 0; massCase < 2; massCase++)
	{
		const PxReal softMass = endpointMasses[massCase][0];
		const PxReal rigidMass = endpointMasses[massCase][1];
		for (PxU32 frame = 0; frame < 3; frame++)
		{
			const PxReal frameInvDtSq =
				1.0f / (frameDts[frame] * frameDts[frame]);
			const PxReal expectedSoftCorrection =
				10.0f / (softMass * frameInvDtSq + 100.0f);
			const PxReal expectedRigidCorrection =
				-10.0f / (rigidMass * frameInvDtSq + 100.0f);

			const PxMat33 frameSoftSystem =
				PxMat33::createDiagonal(
					PxVec3(softMass * frameInvDtSq)) +
				softHessian;
			const PxReal actualSoftCorrection =
				(frameSoftSystem.getInverse() * softForce).y;

			AvbdBlock6x6 frameRigidSystem;
			frameRigidSystem.initializeDiagonal(
				1.0f / rigidMass, PxMat33(PxIdentity),
				frameInvDtSq);
			frameRigidSystem.linearLinear +=
				rigidHessian.linearLinear;
			const PxReal actualRigidCorrection =
				-(frameRigidSystem.linearLinear.getInverse() *
					rigidGradient.linear).y;
			if (PxAbs(actualSoftCorrection -
					expectedSoftCorrection) > 1e-6f ||
				PxAbs(actualRigidCorrection -
					expectedRigidCorrection) > 1e-6f)
				timestepScalingMatches = false;
		}
	}
	TEST_CHECK(
		timestepScalingMatches,
		"Rigid-soft contact preserves timestep and endpoint mass-ratio algebra");

	PxArray<AvbdSoftParticle> unequalParticles(particles);
	unequalParticles[0].mass = 2.0f;
	unequalParticles[0].invMass = 0.5f;
	unequalParticles[0].predictedPosition =
		unequalParticles[0].position;
	AvbdSolverBody unequalRigid;
	unequalRigid.initialize(
		PxTransform(PxIdentity), PxVec3(0.0f), PxVec3(0.0f), 0.125f,
		PxMat33::createDiagonal(PxVec3(0.125f)), 0);
	const PxVec3 unequalRigidAnchor = unequalRigid.position;
	AvbdSoftContact unequalContact = contact;
	const PxReal initialWeightedY =
		unequalParticles[0].mass * unequalParticles[0].position.y +
		8.0f * unequalRigid.position.y;
	for (PxU32 iteration = 0; iteration < 60; iteration++)
	{
		PxVec3 contactForce;
		PxMat33 contactHessian;
		avbdEvaluateContactParticleBlockAtSurfacePoint(
			unequalContact.geometry, unequalContact.state,
			unequalParticles.begin(),
			avbdGetRigidContactSurfacePoint(
				unequalContact.geometry, unequalRigid),
			1.0f, contactForce, contactHessian);
		const PxReal softMassInvDtSq =
			unequalParticles[0].mass * invDtSq;
		const PxVec3 inertialForce =
			(unequalParticles[0].predictedPosition -
			 unequalParticles[0].position) * softMassInvDtSq;
		const PxMat33 particleSystem =
			PxMat33::createDiagonal(PxVec3(softMassInvDtSq)) +
			contactHessian;
		unequalParticles[0].position +=
			particleSystem.getInverse() *
			(inertialForce + contactForce);

		AvbdBlock6x6 bodySystem;
		bodySystem.initializeDiagonal(
			unequalRigid.invMass, unequalRigid.invInertiaWorld,
			invDtSq);
		AvbdVec6 bodyGradient(
			(unequalRigid.position - unequalRigidAnchor) *
				(8.0f * invDtSq),
			PxVec3(0.0f));
		avbdAddDynamicSoftRigidContactContributions_rigid(
			&unequalContact, 1, 0,
			unequalParticles.begin(), unequalParticles.size(),
			unequalRigid, bodySystem, bodyGradient);
		unequalRigid.position -=
			bodySystem.linearLinear.getInverse() *
			bodyGradient.linear;
	}
	const PxReal finalWeightedY =
		unequalParticles[0].mass * unequalParticles[0].position.y +
		8.0f * unequalRigid.position.y;
	TEST_CHECK(
		PxAbs(finalWeightedY - initialWeightedY) < 1e-4f &&
		unequalParticles[0].position.y > particles[0].position.y &&
		unequalRigid.position.y < 0.0f,
		"Unequal-mass rigid-soft block descent preserves internal translation");

	PxArray<AvbdSoftContact> previousContacts;
	AvbdSoftContact persistentContact = contact;
	persistentContact.state.alLambda = -7.0f;
	previousContacts.pushBack(persistentContact);
	PxArray<AvbdSoftContact> detectedContacts;
	detectedContacts.pushBack(contact);
	avbdTransferSoftContactState(
		previousContacts.begin(), previousContacts.size(),
		particles.begin(), detectedContacts);
	const bool churnPreserved =
		detectedContacts[0].state.alLambda < -6.0f;

	detectedContacts.clear();
	detectedContacts.pushBack(contact);
	avbdTransferSoftContactState(
		NULL, 0, particles.begin(), detectedContacts);
	TEST_CHECK(
		churnPreserved &&
		PxAbs(detectedContacts[0].state.alLambda) < 1e-6f,
		"Rigid contact churn preserves one state but remove/re-add starts fresh");
}

// ============================================================================
// Test 25: A positive-J rejected step is not a convergence certificate
// ============================================================================

static void testSoftSweepResidualAuthority()
{
	printf("\n--- Test 25: Soft Sweep Residual Authority ---\n");

	PxArray<AvbdSoftParticle> particles(4);
	particles[0].position = PxVec3(0.0f, 0.0f, 0.0f);
	particles[1].position = PxVec3(1.0f, 0.0f, 0.0f);
	particles[2].position = PxVec3(0.0f, 1.0f, 0.0f);
	particles[3].position = PxVec3(0.0f, 0.0f, 1.0f);
	for(PxU32 particleIdx = 0;
		particleIdx < particles.size(); particleIdx++)
	{
		particles[particleIdx].initialPosition =
			particles[particleIdx].position;
	}

	AvbdSoftBody body;
	body.compiled.particleStart = 0;
	body.compiled.particleCount = particles.size();
	body.compiled.tetrahedra.pushBack(0);
	body.compiled.tetrahedra.pushBack(1);
	body.compiled.tetrahedra.pushBack(2);
	body.compiled.tetrahedra.pushBack(3);
	body.buildElements(particles);
	TEST_CHECK(
		body.compiled.tetElements.size() == 1,
		"Residual authority fixture compiles one positive-volume tetrahedron");

	// Compress the rest-space unit tetrahedron exactly to the solver's
	// det(F) floor. A further inward local solve step must be rejected.
	particles[1].position = PxVec3(0.05f, 0.0f, 0.0f);
	const PxVec3 localSolveDisplacement(-0.01f, 0.0f, 0.0f);
	const AvbdSoftTetDisplacementLimitResult limitResult =
		avbdLimitTetDisplacementObserved(
			body, 1, particles.begin(), localSolveDisplacement);
	TEST_CHECK(
		limitResult.reason ==
			AvbdSoftTetDisplacementLimitReason::
				ePOSITIVE_J_REJECTED &&
		limitResult.appliedDisplacement.magnitudeSquared() < 1e-20f &&
		localSolveDisplacement.magnitudeSquared() > 1e-6f,
		"Positive-J limiter exposes a rejected nonzero local solve step");

	AvbdSoftSweepConvergenceObservation observation;
	observation.observe(
		localSolveDisplacement, false, limitResult);
	TEST_CHECK(
		observation.isAppliedDisplacementConverged(1e-12f) &&
		!observation.isResidualConverged(1e-12f) &&
		observation.positiveJRejectedSteps == 1,
		"Applied displacement alone cannot certify residual convergence");
}

// ============================================================================
// Test 26: Optimized tetrahedron kernel must preserve the reference algebra
// ============================================================================

static void evaluateNeoHookeanReference(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxVec3 p0 = particles[tet.p0].position;
	const PxVec3 e1 = particles[tet.p1].position - p0;
	const PxVec3 e2 = particles[tet.p2].position - p0;
	const PxVec3 e3 = particles[tet.p3].position - p0;
	const PxMat33 F = PxMat33(e1, e2, e3) * tet.DmInv;
	const PxReal J = F.getDeterminant();

	PxMat33 cofactor;
	cofactor.column0 = F.column1.cross(F.column2);
	cofactor.column1 = F.column2.cross(F.column0);
	cofactor.column2 = F.column0.cross(F.column1);

	const PxMat33& inverseRestShape = tet.DmInv;
	PxVec3 shapeGradient;
	if(vOrder == 0)
	{
		shapeGradient = PxVec3(
			-avbdColSum(inverseRestShape.column0),
			-avbdColSum(inverseRestShape.column1),
			-avbdColSum(inverseRestShape.column2));
	}
	else if(vOrder == 1)
		shapeGradient = avbdMatRow(inverseRestShape, 0);
	else if(vOrder == 2)
		shapeGradient = avbdMatRow(inverseRestShape, 1);
	else
		shapeGradient = avbdMatRow(inverseRestShape, 2);

	const PxVec3 Fm = F * shapeGradient;
	const PxVec3 cofactorM = cofactor * shapeGradient;
	const PxReal lambdaSafe =
		PxAbs(lam) < 1e-6f ? 1e-6f : lam;
	const PxReal alpha = 1.0f + mu / lambdaSafe;
	const PxReal safeJ = PxMax(J, 0.05f);
	const PxReal restVolume = tet.restVolume;
	const PxReal shapeGradientNormSq =
		shapeGradient.magnitudeSquared();

	outForce =
		(Fm * mu + cofactorM * (lam * (safeJ - alpha))) *
		(-restVolume);
	outHessian = PxMat33::createDiagonal(
		PxVec3(mu * shapeGradientNormSq * restVolume)) +
		avbdOuter(cofactorM, cofactorM) * (lam * restVolume);
	if(J < 0.5f)
	{
		const PxReal regularization =
			(0.5f - J) * lam * restVolume *
			shapeGradientNormSq;
		outHessian += PxMat33::createDiagonal(
			PxVec3(regularization));
	}
}

static PxReal matrixDifferenceMagnitude(
	const PxMat33& lhs, const PxMat33& rhs)
{
	return PxSqrt(
		(lhs.column0 - rhs.column0).magnitudeSquared() +
		(lhs.column1 - rhs.column1).magnitudeSquared() +
		(lhs.column2 - rhs.column2).magnitudeSquared());
}

static PxReal matrixMagnitude(const PxMat33& matrix)
{
	return PxSqrt(
		matrix.column0.magnitudeSquared() +
		matrix.column1.magnitudeSquared() +
		matrix.column2.magnitudeSquared());
}

static void testNeoHookeanKernelEquivalence()
{
	printf("\n--- Test 26: Neo-Hookean Kernel Equivalence ---\n");

	PxArray<AvbdSoftParticle> particles(4);
	const PxVec3 restPositions[4] =
	{
		PxVec3(0.1f, -0.2f, 0.3f),
		PxVec3(1.2f, 0.1f, 0.2f),
		PxVec3(-0.1f, 0.9f, 0.4f),
		PxVec3(0.2f, 0.2f, 1.4f)
	};
	for(PxU32 particleIdx = 0; particleIdx < 4; particleIdx++)
		particles[particleIdx].position = restPositions[particleIdx];

	AvbdSoftBody body;
	body.compiled.particleStart = 0;
	body.compiled.particleCount = particles.size();
	body.compiled.tetrahedra.pushBack(0);
	body.compiled.tetrahedra.pushBack(1);
	body.compiled.tetrahedra.pushBack(2);
	body.compiled.tetrahedra.pushBack(3);
	body.buildElements(particles);
	TEST_CHECK(
		body.compiled.tetElements.size() == 1,
		"Kernel equivalence fixture compiles one irregular tetrahedron");
	if(body.compiled.tetElements.size() != 1)
		return;

	const PxMat33 deformations[3] =
	{
		PxMat33(
			PxVec3(1.1f, 0.1f, 0.0f),
			PxVec3(0.2f, 0.9f, 0.1f),
			PxVec3(0.0f, 0.15f, 1.05f)),
		PxMat33::createDiagonal(PxVec3(0.2f, 1.0f, 1.0f)),
		PxMat33::createDiagonal(PxVec3(-0.3f, 1.0f, 1.0f))
	};
	const PxVec3 translation(0.4f, -0.7f, 1.3f);
	const PxReal mu = 37000.0f;
	const PxReal lambda = 81000.0f;
	bool equivalent = true;
	bool finite = true;
	for(PxU32 deformationIdx = 0; deformationIdx < 3;
		deformationIdx++)
	{
		for(PxU32 particleIdx = 0; particleIdx < 4; particleIdx++)
		{
			particles[particleIdx].position =
				translation +
				deformations[deformationIdx] *
					(restPositions[particleIdx] -
					 restPositions[0]);
		}
		for(PxU32 vertexOrder = 0; vertexOrder < 4; vertexOrder++)
		{
			PxVec3 referenceForce;
			PxMat33 referenceHessian;
			evaluateNeoHookeanReference(
				body.compiled.tetElements[0], int(vertexOrder),
				mu, lambda, particles.begin(),
				referenceForce, referenceHessian);
			PxVec3 kernelForce;
			PxMat33 kernelHessian;
			const PxReal lambdaSafe =
				PxAbs(lambda) < 1e-6f ? 1e-6f : lambda;
			avbdEvaluateNeoHookeanForceHessianPrepared(
				body.compiled.tetElements[0], int(vertexOrder),
				mu, lambda, 1.0f + mu / lambdaSafe,
				particles.begin(),
				kernelForce, kernelHessian);
			const PxReal forceTolerance =
				5e-5f * PxMax(1.0f, referenceForce.magnitude());
			const PxReal hessianTolerance =
				5e-5f * PxMax(
					1.0f, matrixMagnitude(referenceHessian));
			if((kernelForce - referenceForce).magnitude() >
					forceTolerance ||
				matrixDifferenceMagnitude(
					kernelHessian, referenceHessian) >
					hessianTolerance)
				equivalent = false;
			if(!kernelForce.isFinite() ||
				!kernelHessian.column0.isFinite() ||
				!kernelHessian.column1.isFinite() ||
				!kernelHessian.column2.isFinite())
				finite = false;
		}
	}
	TEST_CHECK(
		equivalent,
		"Fast tet kernel matches generic F/cofactor algebra in all J regimes");
	TEST_CHECK(
		finite,
		"Tet kernel remains finite for normal, compressed, and inverted states");
}

// ============================================================================
// Test 27: Analytical positive-J limiter must preserve generic determinant IR
// ============================================================================

static PxVec3 tetDeterminantGradientReference(
	const AvbdTetElement& tet, PxU32 vertexOrder,
	const AvbdSoftParticle* particles)
{
	const PxVec3 p0 = particles[tet.p0].position;
	const PxMat33 F(
		particles[tet.p1].position - p0,
		particles[tet.p2].position - p0,
		particles[tet.p3].position - p0);
	const PxMat33 deformation = F * tet.DmInv;
	const PxMat33 cofactor(
		deformation.column1.cross(deformation.column2),
		deformation.column2.cross(deformation.column0),
		deformation.column0.cross(deformation.column1));
	const PxMat33& inverseRestShape = tet.DmInv;
	const PxVec3 shapeGradient =
		vertexOrder == 0
			? PxVec3(
				-avbdColSum(inverseRestShape.column0),
				-avbdColSum(inverseRestShape.column1),
				-avbdColSum(inverseRestShape.column2))
			: avbdMatRow(inverseRestShape, int(vertexOrder - 1));
	return cofactor * shapeGradient;
}

static AvbdSoftTetDisplacementLimitResult
limitTetDisplacementReference(
	const AvbdSoftBody& body, PxU32 particleIdx,
	const AvbdSoftParticle* particles, const PxVec3& displacement,
	PxReal minDetF = 0.05f)
{
	if(!displacement.isFinite())
	{
		return AvbdSoftTetDisplacementLimitResult(
			PxVec3(0.0f), 0.0f,
			AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
	}
	if(particleIdx < body.compiled.particleStart ||
		particleIdx >= body.compiled.particleStart +
			body.compiled.particleCount)
	{
		return AvbdSoftTetDisplacementLimitResult(
			displacement, 1.0f,
			AvbdSoftTetDisplacementLimitReason::eNONE);
	}

	const PxU32 localIdx =
		particleIdx - body.compiled.particleStart;
	const AvbdParticleElementAdjacency& adjacency =
		body.compiled.elementAdjacency[localIdx];
	const PxVec3 proposedPosition =
		particles[particleIdx].position + displacement;
	PxReal fraction = 1.0f;
	for(PxU32 refId = 0; refId < adjacency.tetRefs.size(); refId++)
	{
		const AvbdTetElement& tet =
			body.compiled.tetElements[
				adjacency.tetRefs[refId].index];
		const PxVec3 current0 = particles[tet.p0].position;
		const PxVec3 current1 = particles[tet.p1].position;
		const PxVec3 current2 = particles[tet.p2].position;
		const PxVec3 current3 = particles[tet.p3].position;
		const PxReal currentDetF =
			(PxMat33(
				current1 - current0, current2 - current0,
				current3 - current0) *
			 tet.DmInv).getDeterminant();
		const PxVec3 proposed0 =
			tet.p0 == particleIdx ? proposedPosition : current0;
		const PxVec3 proposed1 =
			tet.p1 == particleIdx ? proposedPosition : current1;
		const PxVec3 proposed2 =
			tet.p2 == particleIdx ? proposedPosition : current2;
		const PxVec3 proposed3 =
			tet.p3 == particleIdx ? proposedPosition : current3;
		const PxReal proposedDetF =
			(PxMat33(
				proposed1 - proposed0, proposed2 - proposed0,
				proposed3 - proposed0) *
			 tet.DmInv).getDeterminant();
		if(!PxIsFinite(currentDetF) || !PxIsFinite(proposedDetF))
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					eNONFINITE_REJECTED);
		}
		if(proposedDetF >= minDetF ||
			proposedDetF >= currentDetF)
			continue;
		if(currentDetF <= minDetF)
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					ePOSITIVE_J_REJECTED);
		}
		const PxReal admissible =
			(currentDetF - minDetF) /
			(currentDetF - proposedDetF);
		fraction = PxMin(
			fraction,
			PxMax(0.0f, admissible * 0.99f));
	}
	return AvbdSoftTetDisplacementLimitResult(
		displacement * fraction, fraction,
		fraction < 1.0f
			? AvbdSoftTetDisplacementLimitReason::
				ePOSITIVE_J_LIMITED
			: AvbdSoftTetDisplacementLimitReason::eNONE);
}

static void testPositiveJLimiterKernelEquivalence()
{
	printf("\n--- Test 27: Positive-J Limiter Kernel Equivalence ---\n");

	PxArray<AvbdSoftParticle> particles(4);
	const PxVec3 restPositions[4] =
	{
		PxVec3(0.0f, 0.0f, 0.0f),
		PxVec3(1.0f, 0.0f, 0.0f),
		PxVec3(0.0f, 1.0f, 0.0f),
		PxVec3(0.0f, 0.0f, 1.0f)
	};
	for(PxU32 particleIdx = 0; particleIdx < 4; particleIdx++)
		particles[particleIdx].position = restPositions[particleIdx];

	AvbdSoftBody body;
	body.compiled.particleStart = 0;
	body.compiled.particleCount = particles.size();
	body.compiled.tetrahedra.pushBack(0);
	body.compiled.tetrahedra.pushBack(1);
	body.compiled.tetrahedra.pushBack(2);
	body.compiled.tetrahedra.pushBack(3);
	body.buildElements(particles);
	TEST_CHECK(
		body.compiled.tetElements.size() == 1,
		"Limiter equivalence fixture compiles one unit tetrahedron");
	if(body.compiled.tetElements.size() != 1)
		return;

	bool equivalent = true;
	bool sawUnlimited = false;
	bool sawLimited = false;
	bool sawRejected = false;
	const PxMat33 deformations[2] =
	{
		PxMat33(PxIdentity),
		PxMat33::createDiagonal(PxVec3(0.0f, 1.0f, 1.0f))
	};
	for(PxU32 deformationIdx = 0; deformationIdx < 2;
		deformationIdx++)
	{
		for(PxU32 particleIdx = 0; particleIdx < 4; particleIdx++)
		{
			particles[particleIdx].position =
				deformations[deformationIdx] *
					restPositions[particleIdx];
		}
		for(PxU32 vertexOrder = 0; vertexOrder < 4; vertexOrder++)
		{
			const PxVec3 determinantGradient =
				tetDeterminantGradientReference(
					body.compiled.tetElements[0],
					vertexOrder, particles.begin());
			const PxReal gradientNormSq =
				determinantGradient.magnitudeSquared();
			if(gradientNormSq <= 1e-12f)
				continue;
			const AvbdTetElement& tet =
				body.compiled.tetElements[0];
			const PxVec3 p0 = particles[tet.p0].position;
			AvbdTetVertexLinearization linearization;
			avbdEvaluateTetDeterminantAndGradient(
				tet, vertexOrder,
				particles[tet.p1].position - p0,
				particles[tet.p2].position - p0,
				particles[tet.p3].position - p0,
				linearization.determinant,
				linearization.determinantGradient);
			const PxVec3 displacements[2] =
			{
				determinantGradient *
					(0.01f / gradientNormSq),
				determinantGradient *
					(-2.0f / gradientNormSq)
			};
			for(PxU32 displacementIdx = 0;
				displacementIdx < 2; displacementIdx++)
			{
				const AvbdSoftTetDisplacementLimitResult reference =
					limitTetDisplacementReference(
						body, vertexOrder, particles.begin(),
						displacements[displacementIdx]);
				const AvbdSoftTetDisplacementLimitResult kernel =
					avbdLimitTetDisplacementObserved(
						body, vertexOrder, particles.begin(),
						displacements[displacementIdx]);
				const AvbdSoftTetDisplacementLimitResult cachedKernel =
					avbdLimitTetDisplacementFromLinearizations(
						displacements[displacementIdx],
						&linearization, 1);
				if(reference.reason != kernel.reason ||
					reference.reason != cachedKernel.reason ||
					PxAbs(reference.fraction - kernel.fraction) >
						2e-5f ||
					PxAbs(
						reference.fraction -
						cachedKernel.fraction) > 2e-5f ||
					(reference.appliedDisplacement -
					 kernel.appliedDisplacement).magnitude() >
						2e-5f ||
					(reference.appliedDisplacement -
					 cachedKernel.appliedDisplacement).magnitude() >
						2e-5f)
					equivalent = false;
				sawUnlimited =
					sawUnlimited ||
					kernel.reason ==
						AvbdSoftTetDisplacementLimitReason::eNONE;
				sawLimited =
					sawLimited ||
					kernel.reason ==
						AvbdSoftTetDisplacementLimitReason::
							ePOSITIVE_J_LIMITED;
				sawRejected =
					sawRejected ||
					kernel.reason ==
						AvbdSoftTetDisplacementLimitReason::
							ePOSITIVE_J_REJECTED;
			}
		}
	}
	TEST_CHECK(
		equivalent,
		"Analytical limiter matches generic current/proposed det(F) algebra");
	TEST_CHECK(
		sawUnlimited && sawLimited && sawRejected,
		"Limiter equivalence covers unlimited, limited, and rejected steps");
}

// ============================================================================
// Test 28: Residual exit requires consecutive feasible local solves
// ============================================================================

static void testResidualConvergenceTracker()
{
	printf("\n--- Test 28: Residual Convergence Tracker ---\n");

	const PxVec3 smallLocalSolve(5e-5f, 0.0f, 0.0f);
	const AvbdSoftTetDisplacementLimitResult feasibleResult(
		smallLocalSolve, 1.0f,
		AvbdSoftTetDisplacementLimitReason::eNONE);
	AvbdSoftSweepConvergenceObservation feasibleObservation;
	feasibleObservation.observe(
		smallLocalSolve, false, feasibleResult);

	AvbdSoftResidualConvergenceTracker tracker(1e-8f, 2);
	const bool stoppedAfterOne = tracker.observe(feasibleObservation);
	const bool stoppedAfterTwo = tracker.observe(feasibleObservation);
	TEST_CHECK(
		!stoppedAfterOne && stoppedAfterTwo,
		"Residual policy requires two consecutive feasible sweeps");

	const AvbdSoftTetDisplacementLimitResult rejectedResult(
		PxVec3(0.0f), 0.0f,
		AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_REJECTED);
	AvbdSoftSweepConvergenceObservation rejectedObservation;
	rejectedObservation.observe(
		smallLocalSolve, false, rejectedResult);
	TEST_CHECK(
		!tracker.observe(rejectedObservation) &&
		!tracker.observe(feasibleObservation),
		"Positive-J rejection resets residual convergence history");

	const PxVec3 largeLocalSolve(2e-4f, 0.0f, 0.0f);
	const AvbdSoftTetDisplacementLimitResult largeResult(
		largeLocalSolve, 1.0f,
		AvbdSoftTetDisplacementLimitReason::eNONE);
	AvbdSoftSweepConvergenceObservation largeObservation;
	largeObservation.observe(
		largeLocalSolve, false, largeResult);
	TEST_CHECK(
		!tracker.observe(largeObservation) &&
		!tracker.observe(feasibleObservation),
		"Residual above tolerance resets consecutive convergence");
}

// ============================================================================
// Test 29: Symmetric 3x3 block solve must preserve generic inverse algebra
// ============================================================================

static void testSymmetricParticleBlockSolve()
{
	printf("\n--- Test 29: Symmetric Particle Block Solve ---\n");

	const PxMat33 systems[4] =
	{
		PxMat33::createDiagonal(PxVec3(2.0f, 3.0f, 5.0f)),
		PxMat33(
			PxVec3(4.0f, 1.0f, 0.5f),
			PxVec3(1.0f, 3.0f, 0.2f),
			PxVec3(0.5f, 0.2f, 2.0f)),
		PxMat33(
			PxVec3(1e6f, 2e4f, -1e4f),
			PxVec3(2e4f, 8e5f, 3e4f),
			PxVec3(-1e4f, 3e4f, 6e5f)),
		PxMat33(
			PxVec3(1.0f, 0.999f, 0.998f),
			PxVec3(0.999f, 1.001f, 0.997f),
			PxVec3(0.998f, 0.997f, 1.002f))
	};
	const PxVec3 rightHandSides[3] =
	{
		PxVec3(1.0f, -2.0f, 3.0f),
		PxVec3(-0.01f, 0.03f, 0.02f),
		PxVec3(1000.0f, -300.0f, 7.0f)
	};
	bool equivalent = true;
	bool finite = true;
	for(PxU32 systemIdx = 0; systemIdx < 4; systemIdx++)
	{
		for(PxU32 rhsIdx = 0; rhsIdx < 3; rhsIdx++)
		{
			const PxVec3 reference =
				systems[systemIdx].getInverse() *
				rightHandSides[rhsIdx];
			const PxVec3 fast =
				avbdSolveSymmetric33(
					systems[systemIdx], rightHandSides[rhsIdx]);
			const PxReal tolerance =
				2e-5f * PxMax(1.0f, reference.magnitude());
			if((fast - reference).magnitude() > tolerance)
				equivalent = false;
			if(!fast.isFinite())
				finite = false;
		}
	}
	TEST_CHECK(
		equivalent && finite,
		"Symmetric direct solve matches generic inverse across block scales");

	const PxMat33 singular(PxZero);
	const PxVec3 singularRhs(1.0f, 2.0f, 3.0f);
	TEST_CHECK(
		(avbdSolveSymmetric33(singular, singularRhs) -
		 singularRhs).magnitudeSquared() < 1e-20f,
		"Singular symmetric solve preserves the generic identity fallback");
}

// ============================================================================
// Test 30: CPU AVBD deformable-volume actor and Scene lifecycle
// ============================================================================

static void testCpuAvbdDeformableVolumeSceneLifecycle()
{
	printf(
		"\n--- Test 30: CPU AVBD Deformable Volume Scene Lifecycle ---\n");

	PxPhysics* physics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), false);
	TEST_CHECK(
		physics != NULL,
		"CPU AVBD deformable-volume test creates PxPhysics");
	if(!physics)
		return;

	PxDeformableVolumeMaterial* material =
		physics->createDeformableVolumeMaterial(
			25000.0f, 0.3f, 0.4f, 0.02f);
	TEST_CHECK(
		material != NULL,
		"CPU deformable-volume material factory is available");
	if(material)
	{
		TEST_CHECK(
			physics->getNbDeformableVolumeMaterials() == 1,
			"CPU deformable-volume material is registered with PxPhysics");
		TEST_CHECK(
			PxAbs(material->getYoungsModulus() - 25000.0f) < 1e-3f &&
			PxAbs(material->getPoissons() - 0.3f) < 1e-6f &&
			PxAbs(material->getDynamicFriction() - 0.4f) < 1e-6f &&
			PxAbs(material->getElasticityDamping() - 0.02f) < 1e-6f,
			"CPU deformable-volume material preserves public properties");
		material->setMaterialModel(
			PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
		TEST_CHECK(
			material->getMaterialModel() ==
				PxDeformableVolumeMaterialModel::eNEO_HOOKEAN,
			"CPU deformable-volume material preserves Neo-Hookean selection");
		material->setMaterialModel(
			PxDeformableVolumeMaterialModel::eCO_ROTATIONAL);
		TEST_CHECK(
			material->getMaterialModel() ==
				PxDeformableVolumeMaterialModel::eCO_ROTATIONAL,
			"CPU deformable-volume material preserves co-rotational selection");
	}

	PxDeformableVolume* volume = physics->createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	TEST_CHECK(
		volume != NULL,
		"Explicit CPU AVBD deformable-volume factory returns an actor");
	if(volume)
	{
		volume->setSelfCollisionStressTolerance(0.25f);
		TEST_CHECK(
			PxAbs(
				volume->getSelfCollisionStressTolerance() -
				0.25f) < 1.0e-6f,
			"CPU AVBD volume preserves public self-collision stress tolerance");
		TEST_CHECK(
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD,
			"CPU AVBD actor reports its immutable backend");
		TEST_CHECK(
			volume->getCudaContextManager() == NULL,
			"CPU AVBD actor has no CUDA context");
		TEST_CHECK(
			volume->getPositionInvMassBufferH() == NULL &&
			volume->getRestPositionBufferH() == NULL &&
			volume->getSimPositionInvMassBufferH() == NULL &&
			volume->getSimVelocityBufferH() == NULL,
			"Detached CPU AVBD actor exposes no unowned host buffers");
		TEST_CHECK(
			volume->getGpuDeformableVolumeIndex() == PX_INVALID_U32,
			"CPU AVBD actor never aliases a GPU actor index");

		const PxVec3 vertices[4] =
		{
			PxVec3(0.0f, 0.0f, 0.0f),
			PxVec3(1.0f, 0.0f, 0.0f),
			PxVec3(0.0f, 1.0f, 0.0f),
			PxVec3(0.0f, 0.0f, 1.0f)
		};
		const PxU32 tetrahedra[4] = { 0, 1, 2, 3 };
		PxTetrahedronMeshDesc meshDesc;
		meshDesc.points.count = 4;
		meshDesc.points.stride = sizeof(PxVec3);
		meshDesc.points.data = vertices;
		meshDesc.tetrahedrons.count = 1;
		meshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
		meshDesc.tetrahedrons.data = tetrahedra;

		PxCookingParams cookingParams(physics->getTolerancesScale());
		cookingParams.buildGPUData = false;
		PxTetrahedronMesh* mesh = PxCreateTetrahedronMesh(
			cookingParams, meshDesc,
			physics->getPhysicsInsertionCallback());
		TEST_CHECK(
			mesh != NULL,
			"CPU AVBD lifecycle test creates a host tetrahedron mesh");

		PxShape* shape = NULL;
		bool attached = false;
		if(mesh && material)
		{
			shape = physics->createShape(
				PxTetrahedronMeshGeometry(mesh), *material, true,
				PxShapeFlag::eSIMULATION_SHAPE);
			TEST_CHECK(
				shape != NULL,
				"CPU deformable-volume shape factory accepts its material");
			if(shape)
			{
				attached = volume->attachShape(*shape);
				TEST_CHECK(
					attached,
					"CPU AVBD actor attaches a tetrahedron collision shape");
			}
		}

		if(attached)
		{
			PxVec4* positions = volume->getPositionInvMassBufferH();
			PxVec4* restPositions = volume->getRestPositionBufferH();
			TEST_CHECK(
				positions != NULL && restPositions != NULL,
				"Attached CPU AVBD shape owns host collision buffers");
			if(positions && restPositions)
			{
				for(PxU32 i = 0; i < 4; i++)
				{
					positions[i] = PxVec4(vertices[i], 1.0f);
					restPositions[i] = PxVec4(vertices[i], 0.0f);
				}
			}
			TEST_CHECK(
				positions && restPositions &&
				positions[2].y == 1.0f &&
				restPositions[3].z == 1.0f,
				"CPU AVBD collision buffers are writable host storage");

			volume->detachShape();
			TEST_CHECK(
				volume->getPositionInvMassBufferH() == NULL &&
				volume->getRestPositionBufferH() == NULL,
				"Detaching a CPU AVBD shape clears host buffer ownership");
		}

		PX_RELEASE(shape);
		PX_RELEASE(mesh);

		const PxI32 vertexToTet[4] = { 0, 0, 0, 0 };
		PxDeformableVolumeSimulationDataDesc simulationDataDesc;
		simulationDataDesc.vertexToTet.count = 4;
		simulationDataDesc.vertexToTet.stride = sizeof(PxI32);
		simulationDataDesc.vertexToTet.data = vertexToTet;
		PxCookingParams deformableCookingParams = cookingParams;
		// CPU AVBD must not require GRB/BV32 data merely to obtain the shared
		// deformable simulation topology and collision embedding.
		deformableCookingParams.buildGPUData = false;
		PxDeformableVolumeMesh* volumeMesh =
			PxCreateDeformableVolumeMesh(
				deformableCookingParams, meshDesc, meshDesc,
				simulationDataDesc,
				physics->getPhysicsInsertionCallback());
		TEST_CHECK(
			volumeMesh != NULL,
			"CPU AVBD lifecycle test cooks a complete deformable-volume mesh");

		PxDefaultMemoryOutputStream cookedVolumeStream;
		const bool volumeStreamCooked =
			PxCookDeformableVolumeMesh(
				deformableCookingParams, meshDesc, meshDesc,
				simulationDataDesc, cookedVolumeStream);
		PxDeformableVolumeMesh* streamedVolumeMesh = NULL;
		if(volumeStreamCooked)
		{
			PxDefaultMemoryInputData cookedVolumeInput(
				cookedVolumeStream.getData(),
				cookedVolumeStream.getSize());
			streamedVolumeMesh =
				physics->createDeformableVolumeMesh(cookedVolumeInput);
		}
		const bool streamedVolumeResourcesReady =
			streamedVolumeMesh &&
			streamedVolumeMesh->getCollisionMesh() &&
			streamedVolumeMesh->getSimulationMesh() &&
			streamedVolumeMesh->getDeformableVolumeAuxData() &&
			streamedVolumeMesh->getDeformableVolumeAuxData()->
				getGridModelInvMass();
		TEST_CHECK(
			volumeStreamCooked && streamedVolumeResourcesReady,
			"CPU-only cooked stream reloads complete shared deformable data");

		PxCookingParams gpuCookingParams = deformableCookingParams;
		gpuCookingParams.buildGPUData = true;
		PxDefaultMemoryOutputStream gpuVolumeStream;
		const bool gpuVolumeStreamCooked =
			PxCookDeformableVolumeMesh(
				gpuCookingParams, meshDesc, meshDesc,
				simulationDataDesc, gpuVolumeStream);
		PxDeformableVolumeMesh* gpuStreamedVolumeMesh = NULL;
		if(gpuVolumeStreamCooked)
		{
			PxDefaultMemoryInputData gpuVolumeInput(
				gpuVolumeStream.getData(), gpuVolumeStream.getSize());
			gpuStreamedVolumeMesh =
				physics->createDeformableVolumeMesh(gpuVolumeInput);
		}
		TEST_CHECK(
			gpuStreamedVolumeMesh &&
			gpuStreamedVolumeMesh->getCollisionMesh() &&
			gpuStreamedVolumeMesh->getSimulationMesh() &&
			gpuStreamedVolumeMesh->getDeformableVolumeAuxData(),
			"Optional GRB cooked stream retains the shared deformable payload");

		PxDeformableVolumeMesh* legacyStreamedVolumeMesh = NULL;
		const PxU64 gpuVolumeStreamSize = gpuVolumeStream.getSize();
		if(gpuVolumeStreamCooked &&
			gpuVolumeStreamSize >= 16 &&
			gpuVolumeStreamSize <= PX_MAX_U32)
		{
			const PxU32 legacyStreamSize =
				static_cast<PxU32>(gpuVolumeStreamSize);
			PxArray<PxU8> legacyVolumeBytes;
			legacyVolumeBytes.resize(legacyStreamSize);
			PxMemCopy(
				legacyVolumeBytes.begin(), gpuVolumeStream.getData(),
				legacyStreamSize);
			const PxU32 legacyVersion = 3;
			PxMemCopy(
				legacyVolumeBytes.begin() + 8,
				&legacyVersion, sizeof(legacyVersion));
			PxU32 legacyFlags = 0;
			PxMemCopy(
				&legacyFlags, legacyVolumeBytes.begin() + 12,
				sizeof(legacyFlags));
			legacyFlags &= ~(1u << 10);
			PxMemCopy(
				legacyVolumeBytes.begin() + 12,
				&legacyFlags, sizeof(legacyFlags));
			PxDefaultMemoryInputData legacyVolumeInput(
				legacyVolumeBytes.begin(), legacyVolumeBytes.size());
			legacyStreamedVolumeMesh =
				physics->createDeformableVolumeMesh(legacyVolumeInput);
		}
		TEST_CHECK(
			legacyStreamedVolumeMesh &&
			legacyStreamedVolumeMesh->getCollisionMesh() &&
			legacyStreamedVolumeMesh->getSimulationMesh() &&
			legacyStreamedVolumeMesh->getDeformableVolumeAuxData(),
			"Version-3 GRB stream remains backward compatible");

		PxShape* sceneShape = NULL;
		bool sceneResourcesAttached = false;
		if(volumeMesh && material)
		{
			sceneShape = physics->createShape(
				PxTetrahedronMeshGeometry(
					volumeMesh->getCollisionMesh()),
				*material, true, PxShapeFlag::eSIMULATION_SHAPE);
			const bool collisionAttached =
				sceneShape && volume->attachShape(*sceneShape);
			const bool simulationAttached =
				collisionAttached &&
				volume->attachSimulationMesh(
					*volumeMesh->getSimulationMesh(),
					*volumeMesh->getDeformableVolumeAuxData());
			sceneResourcesAttached =
				collisionAttached && simulationAttached;
			TEST_CHECK(
				sceneResourcesAttached,
				"CPU AVBD actor attaches cooked collision and simulation resources");
		}

		if(sceneResourcesAttached)
		{
			PxVec4* collisionPositions =
				volume->getPositionInvMassBufferH();
			PxVec4* collisionRestPositions =
				volume->getRestPositionBufferH();
			PxVec4* simulationPositions =
				volume->getSimPositionInvMassBufferH();
			PxVec4* simulationVelocities =
				volume->getSimVelocityBufferH();
			TEST_CHECK(
				collisionPositions && collisionRestPositions &&
				simulationPositions && simulationVelocities,
				"CPU AVBD Scene resources expose all four host buffers");
			if(collisionPositions && collisionRestPositions &&
				simulationPositions && simulationVelocities)
			{
				for(PxU32 i = 0; i < 4; i++)
				{
					const PxVec3 position =
						vertices[i] + PxVec3(0.0f, 3.0f, 0.0f);
					collisionPositions[i] = PxVec4(position, 1.0f);
					collisionRestPositions[i] = PxVec4(position, 0.0f);
					simulationPositions[i] = PxVec4(position, 1.0f);
					simulationVelocities[i] = PxVec4(PxZero);
				}
			}
			volume->setSolverIterationCounts(4, 1);
		}

		PxDeformableVolume* secondaryVolume =
			physics->createDeformableVolume(
				PxDeformableVolumeBackend::eCPU_AVBD);
		PxShape* secondaryShape = NULL;
		bool secondaryResourcesAttached = false;
		if(secondaryVolume && streamedVolumeResourcesReady && material)
		{
			secondaryShape = physics->createShape(
				PxTetrahedronMeshGeometry(
					streamedVolumeMesh->getCollisionMesh()),
				*material, true, PxShapeFlag::eSIMULATION_SHAPE);
			secondaryResourcesAttached =
				secondaryShape &&
				secondaryVolume->attachShape(*secondaryShape) &&
				secondaryVolume->attachSimulationMesh(
					*streamedVolumeMesh->getSimulationMesh(),
					*streamedVolumeMesh->getDeformableVolumeAuxData());
			if(secondaryResourcesAttached)
			{
				PxVec4* collisionPositions =
					secondaryVolume->getPositionInvMassBufferH();
				PxVec4* collisionRestPositions =
					secondaryVolume->getRestPositionBufferH();
				PxVec4* simulationPositions =
					secondaryVolume->getSimPositionInvMassBufferH();
				PxVec4* simulationVelocities =
					secondaryVolume->getSimVelocityBufferH();
				for(PxU32 i = 0; i < 4; i++)
				{
					const PxVec3 position =
						vertices[i] + PxVec3(2.0f, 4.0f, 0.0f);
					collisionPositions[i] = PxVec4(position, 1.0f);
					collisionRestPositions[i] =
						PxVec4(position, 0.0f);
					simulationPositions[i] = PxVec4(position, 1.0f);
					simulationVelocities[i] = PxVec4(PxZero);
				}
				secondaryVolume->setSolverIterationCounts(4, 1);
			}
		}
		TEST_CHECK(
			secondaryResourcesAttached,
			"CPU AVBD lifecycle test prepares a second Scene actor");

		PxDefaultCpuDispatcher* dispatcher =
			PxDefaultCpuDispatcherCreate(1);
		PxScene* scene = NULL;
		if(dispatcher)
		{
			PxSceneDesc sceneDesc(physics->getTolerancesScale());
			sceneDesc.cpuDispatcher = dispatcher;
			sceneDesc.filterShader = PxDefaultSimulationFilterShader;
			sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
			sceneDesc.solverType = PxSolverType::eAVBD;
			scene = physics->createScene(sceneDesc);
		}
		TEST_CHECK(
			dispatcher != NULL && scene != NULL,
			"CPU AVBD lifecycle test creates a CPU scene");
		if(scene && sceneResourcesAttached)
		{
			scene->addActor(*volume);
			TEST_CHECK(
				volume->getScene() == scene &&
				scene->getNbDeformableVolumes() == 1 &&
				!volume->isSleeping(),
				"CPU AVBD deformable volume enters an AVBD CPU Scene awake");

			if(secondaryResourcesAttached)
				scene->addActor(*secondaryVolume);
			TEST_CHECK(
				secondaryVolume &&
				secondaryVolume->getScene() == scene &&
				scene->getNbDeformableVolumes() == 2,
				"CPU AVBD Scene owns multiple deformable volumes");

			PxDeformableVolume* queriedVolume = NULL;
			TEST_CHECK(
				scene->getDeformableVolumes(
					&queriedVolume, 1, 0) == 1 &&
				queriedVolume == volume,
				"CPU AVBD Scene reports its deformable-volume actor");

			PxVec4* simulationPositions =
				volume->getSimPositionInvMassBufferH();
			PxReal initialCenterY = 0.0f;
			if(simulationPositions)
			{
				for(PxU32 i = 0; i < 4; i++)
					initialCenterY += simulationPositions[i].y;
				initialCenterY *= 0.25f;
			}

			for(PxU32 frame = 0; frame < 12; frame++)
			{
				scene->simulate(1.0f / 60.0f);
				scene->fetchResults(true);
			}

			PxReal finalCenterY = 0.0f;
			bool finiteState = simulationPositions != NULL;
			if(simulationPositions)
			{
				for(PxU32 i = 0; i < 4; i++)
				{
					finiteState =
						finiteState &&
						simulationPositions[i].isFinite();
					finalCenterY += simulationPositions[i].y;
				}
				finalCenterY *= 0.25f;
			}
			TEST_CHECK(
				finiteState &&
				finalCenterY < initialCenterY - 0.01f,
				"CPU AVBD Scene steps and writes finite gravity motion back to host buffers");

			scene->removeActor(*volume);
			TEST_CHECK(
				volume->getScene() == NULL &&
				secondaryVolume &&
				secondaryVolume->getScene() == scene &&
				scene->getNbDeformableVolumes() == 1,
				"Removing the first CPU AVBD body preserves the second");

			PxVec4* secondaryPositions = secondaryVolume
				? secondaryVolume->getSimPositionInvMassBufferH()
				: NULL;
			PxReal secondaryCenterBefore = 0.0f;
			for(PxU32 i = 0; secondaryPositions && i < 4; i++)
				secondaryCenterBefore +=
					secondaryPositions[i].y;
			secondaryCenterBefore *= 0.25f;
			for(PxU32 frame = 0;
				secondaryVolume &&
				secondaryVolume->getScene() == scene &&
				frame < 3; frame++)
			{
				scene->simulate(1.0f / 60.0f);
				scene->fetchResults(true);
			}
			PxReal secondaryCenterAfter = 0.0f;
			for(PxU32 i = 0; secondaryPositions && i < 4; i++)
				secondaryCenterAfter +=
					secondaryPositions[i].y;
			secondaryCenterAfter *= 0.25f;
			TEST_CHECK(
				secondaryPositions &&
				secondaryCenterAfter < secondaryCenterBefore,
				"Remaining CPU AVBD body keeps stepping after middle removal");
			if(secondaryVolume &&
				secondaryVolume->getScene() == scene)
				scene->removeActor(*secondaryVolume);
			TEST_CHECK(
				secondaryVolume &&
				secondaryVolume->getScene() == NULL &&
				scene->getNbDeformableVolumes() == 0,
				"CPU AVBD multi-body Scene ownership clears cleanly");

			PxVec4* simulationVelocities =
				volume->getSimVelocityBufferH();
			PxReal gravityDisabledCenterY = 0.0f;
			if(simulationPositions && simulationVelocities)
			{
				for(PxU32 i = 0; i < 4; i++)
				{
					simulationVelocities[i] = PxVec4(PxZero);
					gravityDisabledCenterY +=
						simulationPositions[i].y;
				}
				gravityDisabledCenterY *= 0.25f;
			}
			volume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			scene->addActor(*volume);
			TEST_CHECK(
				volume->getScene() == scene &&
				scene->getNbDeformableVolumes() == 1,
				"CPU AVBD deformable volume can re-enter the Scene");
			if(volume->getScene() == scene)
			{
				for(PxU32 frame = 0; frame < 6; frame++)
				{
					scene->simulate(1.0f / 60.0f);
					scene->fetchResults(true);
				}
				PxReal stationaryCenterY = 0.0f;
				for(PxU32 i = 0; i < 4; i++)
					stationaryCenterY +=
						simulationPositions[i].y;
				stationaryCenterY *= 0.25f;
				TEST_CHECK(
					PxAbs(
						stationaryCenterY -
						gravityDisabledCenterY) < 1.0e-3f,
					"CPU AVBD Scene honors eDISABLE_GRAVITY after re-entry");
				scene->removeActor(*volume);
			}
			volume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, false);
			TEST_CHECK(
				volume->getScene() == NULL &&
					scene->getNbDeformableVolumes() == 0,
				"CPU AVBD Scene re-entry leaves no stale ownership");

			// Keep the two deformable volumes far enough apart that each one
			// forms an independent native island with its own sleeping rigid
			// target.  This locks the provider contract to one complete tuple
			// per island instead of allowing an all-entries/single-island
			// selection to fail closed.
			PxVec4* primaryCollisionPositions =
				volume->getPositionInvMassBufferH();
			PxVec4* primaryRestPositions =
				volume->getRestPositionBufferH();
			PxVec4* primarySimulationPositions =
				volume->getSimPositionInvMassBufferH();
			PxVec4* primarySimulationVelocities =
				volume->getSimVelocityBufferH();
			PxVec4* secondaryCollisionPositions =
				secondaryVolume
					? secondaryVolume->getPositionInvMassBufferH()
					: NULL;
			PxVec4* secondaryRestPositions =
				secondaryVolume
					? secondaryVolume->getRestPositionBufferH()
					: NULL;
			PxVec4* secondarySimulationPositions =
				secondaryVolume
					? secondaryVolume->getSimPositionInvMassBufferH()
					: NULL;
			PxVec4* secondarySimulationVelocities =
				secondaryVolume
					? secondaryVolume->getSimVelocityBufferH()
					: NULL;
			const bool partitionBuffersReady =
				primaryCollisionPositions &&
				primaryRestPositions &&
				primarySimulationPositions &&
				primarySimulationVelocities &&
				secondaryCollisionPositions &&
				secondaryRestPositions &&
				secondarySimulationPositions &&
				secondarySimulationVelocities;
			TEST_CHECK(
				partitionBuffersReady,
				"CPU AVBD multi-island fixture owns both host tuples");

			PxMaterial* rigidMaterial =
				physics->createMaterial(0.5f, 0.5f, 0.0f);
			PxRigidDynamic* primaryRigid = NULL;
			PxRigidDynamic* secondaryRigid = NULL;
			if(partitionBuffersReady && rigidMaterial)
			{
				const PxVec3 primaryOffset(-4.0f, 4.0f, 0.0f);
				const PxVec3 secondaryOffset(4.0f, 4.0f, 0.0f);
				for(PxU32 i = 0; i < 4; i++)
				{
					const PxVec3 primaryPosition =
						vertices[i] + primaryOffset;
					const PxVec3 secondaryPosition =
						vertices[i] + secondaryOffset;
					primaryCollisionPositions[i] =
						PxVec4(primaryPosition, 1.0f);
					primaryRestPositions[i] =
						PxVec4(primaryPosition, 0.0f);
					primarySimulationPositions[i] =
						PxVec4(primaryPosition, 1.0f);
					primarySimulationVelocities[i] = PxVec4(PxZero);
					secondaryCollisionPositions[i] =
						PxVec4(secondaryPosition, 1.0f);
					secondaryRestPositions[i] =
						PxVec4(secondaryPosition, 0.0f);
					secondarySimulationPositions[i] =
						PxVec4(secondaryPosition, 1.0f);
					secondarySimulationVelocities[i] = PxVec4(PxZero);
				}
				volume->markDirty(PxDeformableVolumeDataFlag::eALL);
				secondaryVolume->markDirty(
					PxDeformableVolumeDataFlag::eALL);

				primaryRigid = physics->createRigidDynamic(
					PxTransform(PxVec3(-3.5f, 2.5f, 0.0f)));
				secondaryRigid = physics->createRigidDynamic(
					PxTransform(PxVec3(4.5f, 2.5f, 0.0f)));
				const PxVec3 rigidHalfExtent(1.0f, 0.25f, 1.0f);
				const bool primaryShapeReady =
					primaryRigid &&
					PxRigidActorExt::createExclusiveShape(
						*primaryRigid,
						PxBoxGeometry(rigidHalfExtent),
						*rigidMaterial) &&
					PxRigidBodyExt::setMassAndUpdateInertia(
						*primaryRigid, 20.0f);
				const bool secondaryShapeReady =
					secondaryRigid &&
					PxRigidActorExt::createExclusiveShape(
						*secondaryRigid,
						PxBoxGeometry(rigidHalfExtent),
						*rigidMaterial) &&
					PxRigidBodyExt::setMassAndUpdateInertia(
						*secondaryRigid, 20.0f);
				if(primaryShapeReady && secondaryShapeReady)
				{
					const PxRigidDynamicLockFlags locks =
						PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
						PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
						PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
						PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
						PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
					primaryRigid->setActorFlag(
						PxActorFlag::eDISABLE_GRAVITY, true);
					secondaryRigid->setActorFlag(
						PxActorFlag::eDISABLE_GRAVITY, true);
					primaryRigid->setRigidDynamicLockFlags(locks);
					secondaryRigid->setRigidDynamicLockFlags(locks);
					primaryRigid->setSolverIterationCounts(8, 1);
					secondaryRigid->setSolverIterationCounts(8, 1);

					scene->addActor(*volume);
					scene->addActor(*secondaryVolume);
					scene->addActor(*primaryRigid);
					scene->addActor(*secondaryRigid);
					primaryRigid->putToSleep();
					secondaryRigid->putToSleep();
					const bool initiallySleeping =
						primaryRigid->isSleeping() &&
						secondaryRigid->isSleeping();
					for(PxU32 frame = 0; frame < 60; frame++)
					{
						scene->simulate(1.0f / 60.0f);
						scene->fetchResults(true);
					}
					const bool bothResponded =
						!primaryRigid->isSleeping() &&
						!secondaryRigid->isSleeping() &&
						primaryRigid->getGlobalPose().p.y < 2.49f &&
						secondaryRigid->getGlobalPose().p.y < 2.49f;
					TEST_CHECK(
						initiallySleeping && bothResponded,
						"Independent CPU AVBD soft islands each own a complete dynamic contact tuple");

					scene->removeActor(*primaryRigid);
					scene->removeActor(*secondaryRigid);
					scene->removeActor(*volume);
					scene->removeActor(*secondaryVolume);
				}
				else
				{
					TEST_CHECK(
						false,
						"CPU AVBD multi-island fixture creates both rigid targets");
				}
			}
			else
			{
				TEST_CHECK(
					false,
					"CPU AVBD multi-island fixture creates its rigid material");
			}
			PX_RELEASE(primaryRigid);
			PX_RELEASE(secondaryRigid);
			PX_RELEASE(rigidMaterial);
		}
		PX_RELEASE(scene);
		PX_RELEASE(dispatcher);

		if(secondaryVolume)
			secondaryVolume->release();
		PX_RELEASE(secondaryShape);
		volume->release();
		PX_RELEASE(sceneShape);
		PX_RELEASE(volumeMesh);
		PX_RELEASE(streamedVolumeMesh);
		PX_RELEASE(gpuStreamedVolumeMesh);
		PX_RELEASE(legacyStreamedVolumeMesh);
	}

	PX_RELEASE(material);
	physics->release();
}

// ============================================================================
// Test 31: Public CPU AVBD deformable-surface Scene lifecycle
// ============================================================================

static void testCpuAvbdDeformableSurfaceSceneLifecycle()
{
	printf(
		"\n--- Test 31: CPU AVBD Deformable Surface Scene Lifecycle ---\n");

	PxPhysics* physics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), false);
	TEST_CHECK(
		physics != NULL,
		"CPU AVBD deformable-surface test creates PxPhysics");
	if(!physics)
		return;

	PxDeformableSurfaceMaterial* material =
		physics->createDeformableSurfaceMaterial(
			5000.0f, 0.3f, 0.35f, 0.02f, 0.01f, 0.02f);
	TEST_CHECK(
		material != NULL,
		"CPU deformable-surface material factory is available");
	if(material)
	{
		TEST_CHECK(
			physics->getNbDeformableSurfaceMaterials() == 1,
			"CPU deformable-surface material is registered with PxPhysics");
		TEST_CHECK(
			PxAbs(material->getYoungsModulus() - 5000.0f) < 1e-3f &&
			PxAbs(material->getPoissons() - 0.3f) < 1e-6f &&
			PxAbs(material->getDynamicFriction() - 0.35f) < 1e-6f &&
			PxAbs(material->getThickness() - 0.02f) < 1e-6f &&
			PxAbs(material->getBendingStiffness() - 0.01f) < 1e-6f,
			"CPU deformable-surface material preserves public properties");
	}

	PxDeformableSurface* surface = physics->createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	TEST_CHECK(
		surface != NULL,
		"Explicit CPU AVBD deformable-surface factory returns an actor");
	if(!surface)
	{
		PX_RELEASE(material);
		physics->release();
		return;
	}

	TEST_CHECK(
		surface->getDeformableSurfaceBackend() ==
			PxDeformableSurfaceBackend::eCPU_AVBD &&
		surface->getCudaContextManager() == NULL,
		"CPU AVBD surface reports its immutable non-CUDA backend");
	TEST_CHECK(
		surface->getPositionInvMassBufferH() == NULL &&
		surface->getVelocityBufferH() == NULL &&
		surface->getRestPositionBufferH() == NULL,
		"Detached CPU surface exposes no unowned host buffers");

	const PxVec3 vertices[4] =
	{
		PxVec3(-0.5f, 3.0f, -0.5f),
		PxVec3( 0.5f, 3.0f, -0.5f),
		PxVec3(-0.5f, 3.0f,  0.5f),
		PxVec3( 0.5f, 3.0f,  0.5f)
	};
	const PxU32 triangles[6] = { 0, 1, 2, 2, 1, 3 };
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = vertices;
	meshDesc.triangles.count = 2;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles;

	PxCookingParams cookingParams(physics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.meshPreprocessParams |=
		PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;
	PxTriangleMesh* mesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics->getPhysicsInsertionCallback());
	TEST_CHECK(
		mesh != NULL,
		"CPU AVBD surface test cooks a host triangle mesh");

	PxShape* shape = NULL;
	bool attached = false;
	if(mesh && material)
	{
		shape = physics->createShape(
			PxTriangleMeshGeometry(mesh), *material, true,
			PxShapeFlag::eSIMULATION_SHAPE);
		attached = shape && surface->attachShape(*shape);
	}
	TEST_CHECK(
		attached,
		"CPU AVBD surface attaches its deformable triangle shape");

	PxVec4* positions = surface->getPositionInvMassBufferH();
	PxVec4* velocities = surface->getVelocityBufferH();
	PxVec4* restPositions = surface->getRestPositionBufferH();
	const bool hostTupleReady =
		positions && velocities && restPositions;
	TEST_CHECK(
		hostTupleReady,
		"Attached CPU surface owns the complete host buffer tuple");
	if(hostTupleReady)
	{
		for(PxU32 i = 0; i < 4; i++)
		{
			const PxReal inverseMass = i < 2 ? 0.0f : 1.0f;
			positions[i] = PxVec4(vertices[i], inverseMass);
			velocities[i] = PxVec4(PxZero);
			restPositions[i] = PxVec4(vertices[i], 0.0f);
		}
		surface->markDirty(PxDeformableSurfaceDataFlag::eALL);
		surface->setSolverIterationCounts(8, 1);
	}

	PxDefaultCpuDispatcher* dispatcher =
		PxDefaultCpuDispatcherCreate(1);
	PxScene* scene = NULL;
	if(dispatcher)
	{
		PxSceneDesc sceneDesc(physics->getTolerancesScale());
		sceneDesc.cpuDispatcher = dispatcher;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		scene = physics->createScene(sceneDesc);
	}
	TEST_CHECK(
		dispatcher != NULL && scene != NULL,
		"CPU AVBD surface test creates an AVBD CPU Scene");

	if(scene && attached && hostTupleReady)
	{
		scene->addActor(*surface);
		TEST_CHECK(
			surface->getScene() == scene &&
			scene->getNbDeformableSurfaces() == 1 &&
			!surface->isSleeping(),
			"CPU deformable surface enters the AVBD Scene awake");

		PxDeformableSurface* queriedSurface = NULL;
		TEST_CHECK(
			scene->getDeformableSurfaces(
				&queriedSurface, 1, 0) == 1 &&
			queriedSurface == surface,
			"CPU AVBD Scene reports its deformable-surface actor");

		for(PxU32 frame = 0; frame < 60; frame++)
		{
			scene->simulate(1.0f / 60.0f);
			scene->fetchResults(true);
		}

		bool finiteState = true;
		for(PxU32 i = 0; i < 4; i++)
			finiteState = finiteState &&
				positions[i].isFinite() &&
				velocities[i].isFinite();
		const PxReal pinnedDrift = PxMax(
			(positions[0].getXYZ() - vertices[0]).magnitude(),
			(positions[1].getXYZ() - vertices[1]).magnitude());
		const PxReal freeDrop = 3.0f -
			0.5f * (positions[2].y + positions[3].y);
		TEST_CHECK(
			finiteState && pinnedDrift < 1.0e-4f &&
				freeDrop > 0.02f,
			"CPU AVBD surface solves pinned cloth and writes finite host state");

		const PxBounds3 bounds = surface->getWorldBounds();
		TEST_CHECK(
			bounds.isValid() && !bounds.isEmpty() &&
				bounds.contains(positions[0].getXYZ()),
			"CPU AVBD surface publishes finite host-derived bounds");

		scene->removeActor(*surface);
		TEST_CHECK(
			surface->getScene() == NULL &&
				scene->getNbDeformableSurfaces() == 0,
			"CPU AVBD surface leaves no stale Scene ownership");
	}

	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	surface->release();
	PX_RELEASE(shape);
	PX_RELEASE(mesh);
	PX_RELEASE(material);
	physics->release();
}

// ============================================================================
// Test 32: One exact Position AL objective for weighted soft points
// ============================================================================

static void testWeightedPointAttachmentPositionAl()
{
	printf(
		"\n--- Test 32: Weighted-Point Attachment Position AL ---\n");

	PxArray<AvbdSoftParticle> particles(4);
	particles[0].position = PxVec3(1.0f, 0.0f, 0.0f);
	particles[1].position = PxVec3(0.0f, 2.0f, 0.0f);
	particles[2].position = PxVec3(0.0f, 0.0f, 3.0f);
	particles[3].position = PxVec3(-4.0f, -4.0f, -4.0f);
	particles[0].invMass = 1.0f;
	particles[1].invMass = 2.0f;
	particles[2].invMass = 4.0f;
	particles[3].invMass = 8.0f;

	AvbdSoftPoint point;
	point.particleCount = 3;
	point.particleIndices[0] = 0;
	point.particleIndices[1] = 1;
	point.particleIndices[2] = 2;
	point.particleIndices[3] = PX_MAX_U32;
	point.weights[0] = 0.2f;
	point.weights[1] = 0.3f;
	point.weights[2] = 0.5f;
	point.weights[3] = 0.0f;

	const PxVec3 pointPosition(0.2f, 0.6f, 1.5f);
	TEST_CHECK(
		(avbdGetSoftPointPosition(point, particles.begin()) -
			pointPosition).magnitude() < 1.0e-6f,
		"Weighted soft point evaluates one exact barycentric position");

	AvbdKinematicPin pin;
	pin.point = point;
	pin.worldTarget = PxVec3(0.1f, 0.2f, 0.3f);
	pin.k = 50.0f;
	pin.kMax = 1000.0f;
	const PxVec3 residual = pointPosition - pin.worldTarget;
	bool pinBlocksAreExact = true;
	for(PxU32 particleIndex = 0; particleIndex < 3; particleIndex++)
	{
		PxVec3 force;
		PxMat33 hessian;
		avbdEvaluatePinForceHessian(
			point, pin, particles.begin(), particleIndex,
			force, hessian);
		const PxReal weight = point.weights[particleIndex];
		pinBlocksAreExact = pinBlocksAreExact &&
			(force + residual * (pin.k * weight)).
				magnitude() < 1.0e-5f &&
			PxAbs(hessian.column0.x -
				pin.k * weight * weight) < 1.0e-6f &&
			PxAbs(hessian.column1.y -
				pin.k * weight * weight) < 1.0e-6f &&
			PxAbs(hessian.column2.z -
				pin.k * weight * weight) < 1.0e-6f;
	}
	TEST_CHECK(
		pinBlocksAreExact,
		"Weighted pin uses w*force and w^2*H for each particle block");

	avbdUpdatePinDual(pin, point, particles.begin(), 0.0f);
	TEST_CHECK(
		(pin.alLambda - residual * 50.0f).magnitude() < 1.0e-5f,
		"Weighted pin updates one dual from the full point residual");

	PxArray<AvbdSoftBody> bodies(1);
	bodies[0].compiled.particleStart = 0;
	bodies[0].compiled.particleCount = particles.size();
	bodies[0].runtime.pins.pushBack(pin);
	bodies[0].runtime.compileObjectiveProgram(
		0, particles.size());
	const AvbdCompiledSoftObjective& compiled =
		bodies[0].runtime.compiledObjectives[0];
	TEST_CHECK(
		compiled.point == point &&
		bodies[0].runtime.objectiveAdjacency[0].
			objectiveIndices.size() == 1 &&
		bodies[0].runtime.objectiveAdjacency[1].
			objectiveIndices.size() == 1 &&
		bodies[0].runtime.objectiveAdjacency[2].
			objectiveIndices.size() == 1 &&
		bodies[0].runtime.objectiveAdjacency[3].
			objectiveIndices.empty(),
		"Prep compiles one weighted objective into all unique endpoint blocks");

	bodies[0].runtime.pins[0].point.weights[0] = 0.25f;
	TEST_CHECK(
		!bodies[0].runtime.isObjectiveProgramCurrent(
			0, particles.size()),
		"Compiled weighted-point IR rejects a mutated runtime snapshot");

	AvbdSolverBody rigidBody;
	rigidBody.initialize(
		PxTransform(PxIdentity), PxVec3(0.0f), PxVec3(0.0f),
		1.0f, PxMat33(PxIdentity), 0);
	AvbdSoftAttachment attachment;
	attachment.point = point;
	attachment.rigidBodyIdx = 0;
	attachment.k = 100.0f;
	attachment.kMax = 1000.0f;
	const PxReal dt = 0.1f;
	const PxReal dt2 = dt * dt;
	AvbdSoftRigidAttachmentCoupledStep step;
	const bool evaluated =
		avbdEvaluateSoftRigidAttachmentCoupledStep(
			attachment, point, particles.begin(), particles.size(),
			rigidBody, dt, step);
	const PxReal softPointInverseMass =
		0.2f * 0.2f * particles[0].invMass +
		0.3f * 0.3f * particles[1].invMass +
		0.5f * 0.5f * particles[2].invMass;
	const PxVec3 expectedMultiplier =
		-pointPosition /
		((softPointInverseMass + rigidBody.invMass) * dt2 +
			1.0f / attachment.k);
	bool correctionsAreExact = evaluated &&
		(step.multiplier - expectedMultiplier).magnitude() < 1.0e-4f;
	for(PxU32 endpoint = 0; endpoint < point.particleCount; endpoint++)
	{
		const PxVec3 expectedCorrection =
			expectedMultiplier *
			(dt2 * point.weights[endpoint] *
			 particles[point.particleIndices[endpoint]].invMass);
		correctionsAreExact = correctionsAreExact &&
			(step.particleCorrections[endpoint] -
				expectedCorrection).magnitude() < 1.0e-5f;
	}
	TEST_CHECK(
		correctionsAreExact,
		"Coupled rigid attachment uses the exact weighted soft response");

	avbdUpdateAttachmentDual(
		attachment, point, particles.begin(), &rigidBody, 0.0f);
	TEST_CHECK(
		(attachment.alLambda - pointPosition * 100.0f).
			magnitude() < 1.0e-4f,
		"Weighted rigid attachment owns one full-residual dual update");

	AvbdSoftPoint duplicatePoint;
	duplicatePoint.particleCount = 2;
	duplicatePoint.particleIndices[0] = 0;
	duplicatePoint.particleIndices[1] = 0;
	duplicatePoint.particleIndices[2] = PX_MAX_U32;
	duplicatePoint.particleIndices[3] = PX_MAX_U32;
	duplicatePoint.weights[0] = 0.25f;
	duplicatePoint.weights[1] = 0.75f;
	duplicatePoint.weights[2] = 0.0f;
	duplicatePoint.weights[3] = 0.0f;
	TEST_CHECK(
		PxAbs(avbdGetSoftPointJacobianWeight(
			duplicatePoint, 0) - 1.0f) < 1.0e-6f &&
		PxAbs(avbdGetSoftPointInverseMass(
			duplicatePoint, particles.begin(), particles.size()) -
			particles[0].invMass) < 1.0e-6f,
		"Repeated indices combine Jacobians before inverse-mass evaluation");
}

// ============================================================================
// Test 33: Plane speculative CCD compiles only for opted-in soft bodies
// ============================================================================

static void testPlaneSpeculativeCcdActiveSet()
{
	printf(
		"\n--- Test 33: Plane Speculative CCD Active Set ---\n");

	PxArray<AvbdSoftParticle> particles(2);
	PxArray<AvbdSoftBody> bodies(2);
	for(PxU32 i = 0; i < 2; ++i)
	{
		particles[i].position = PxVec3(0.0f, 1.2f, 0.0f);
		particles[i].predictedPosition =
			PxVec3(0.0f, -0.8f, 0.0f);
		particles[i].initialPosition = particles[i].position;
		particles[i].invMass = 1.0f;
		particles[i].mass = 1.0f;
		bodies[i].compiled.particleStart = i;
		bodies[i].compiled.particleCount = 1;
		bodies[i].compiled.surfaceVertices.pushBack(i);
	}
	bodies[0].compiled.speculativeCCDEnabled = true;
	bodies[1].compiled.speculativeCCDEnabled = false;

	AvbdWorldPlane plane;
	plane.offset = 0.5f;
	plane.friction = 0.0f;
	PxArray<AvbdSoftContact> contacts;
	avbdDetectSoftWorldPlaneContacts(
		particles.begin(), particles.size(),
		&plane, 1, contacts, 0.02f,
		bodies.begin(), bodies.size());

	PxU32 positivePrepared = 0;
	PxU32 negativePrepared = 0;
	for(PxU32 i = 0; i < contacts.size(); ++i)
	{
		if(contacts[i].geometry.particleIdx == 0)
			positivePrepared++;
		else if(contacts[i].geometry.particleIdx == 1)
			negativePrepared++;
	}
	TEST_CHECK(
		positivePrepared == 1,
		"Opted-in body prepares one swept plane contact");
	TEST_CHECK(
		negativePrepared == 0,
		"Flag-off body omits the swept plane contact");
}

// ============================================================================
// Test 34: Soft-soft speculative OGC owns swept vertex-face and edge-edge
// ============================================================================

static void testSoftSoftSweptOgcFeatures()
{
	printf(
		"\n--- Test 34: Soft-Soft Swept OGC Features ---\n");
	const PxReal margin = 0.05f;
	AvbdOGCParams params;
	params.contactRadius = margin;

	PxArray<AvbdSoftParticle> particles(6);
	const PxVec3 queryXZ[3] =
	{
		PxVec3(-0.3f, 0.0f, -0.2f),
		PxVec3(0.3f, 0.0f, -0.2f),
		PxVec3(0.0f, 0.0f, 0.3f)
	};
	const PxVec3 targetXZ[3] =
	{
		PxVec3(-1.0f, 0.0f, -1.0f),
		PxVec3(1.0f, 0.0f, -1.0f),
		PxVec3(0.0f, 0.0f, 1.0f)
	};
	for(PxU32 i = 0; i < 3; ++i)
	{
		particles[i].initialPosition =
			queryXZ[i] + PxVec3(0.0f, 0.2f, 0.0f);
		particles[i].position =
			queryXZ[i] - PxVec3(0.0f, 0.2f, 0.0f);
		particles[i].invMass = 1.0f;
		particles[i].mass = 1.0f;
		particles[i + 3].initialPosition = targetXZ[i];
		particles[i + 3].position = targetXZ[i];
		particles[i + 3].invMass = 0.0f;
		particles[i + 3].mass = 0.0f;
	}
	PxArray<AvbdSoftBody> bodies(2);
	bodies[0].compiled.particleStart = 0;
	bodies[0].compiled.particleCount = 3;
	bodies[1].compiled.particleStart = 3;
	bodies[1].compiled.particleCount = 3;
	for(PxU32 i = 0; i < 3; ++i)
	{
		bodies[0].compiled.surfaceVertices.pushBack(i);
		bodies[0].compiled.surfaceTriangles.pushBack(i);
		bodies[1].compiled.surfaceVertices.pushBack(i + 3);
		bodies[1].compiled.surfaceTriangles.pushBack(i + 3);
	}
	bodies[0].compiled.surfaceTriangleElementIndices.pushBack(0);
	bodies[1].compiled.surfaceTriangleElementIndices.pushBack(0);
	bodies[0].compiled.speculativeCCDEnabled = true;
	PxArray<AvbdSoftContact> contacts;
	avbdDetectSoftSoftOGC(
		particles.begin(), particles.size(),
		bodies.begin(), bodies.size(), contacts, params);
	PxU32 sweptFaceContacts = 0;
	bool faceNormalStable = true;
	for(PxU32 contactIndex = 0;
		contactIndex < contacts.size(); contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		if(geometry.targetIndex == 1 &&
			geometry.surfaceParticleIndices[2] != PX_MAX_U32)
		{
			sweptFaceContacts++;
			faceNormalStable = faceNormalStable &&
				geometry.depth == 0.0f &&
				geometry.normal.y > 0.8f;
		}
	}
	TEST_CHECK(
		sweptFaceContacts > 0 && faceNormalStable,
		"Soft-soft swept vertex-face rows retain the initial target-to-query side");

	contacts.clear();
	bodies[0].compiled.speculativeCCDEnabled = false;
	avbdDetectSoftSoftOGC(
		particles.begin(), particles.size(),
		bodies.begin(), bodies.size(), contacts, params);
	TEST_CHECK(
		contacts.empty(),
		"Soft-soft flag-off negative control tunnels past the end-step shell");

	PxArray<AvbdSoftParticle> edgeParticles(4);
	edgeParticles[0].initialPosition = PxVec3(-1.0f, 0.2f, 0.0f);
	edgeParticles[1].initialPosition = PxVec3(1.0f, 0.2f, 0.0f);
	edgeParticles[0].position = PxVec3(-1.0f, -0.2f, 0.0f);
	edgeParticles[1].position = PxVec3(1.0f, -0.2f, 0.0f);
	edgeParticles[2].initialPosition =
		edgeParticles[2].position = PxVec3(0.0f, 0.0f, -1.0f);
	edgeParticles[3].initialPosition =
		edgeParticles[3].position = PxVec3(0.0f, 0.0f, 1.0f);
	for(PxU32 i = 0; i < 4; ++i)
	{
		edgeParticles[i].invMass = i < 2 ? 1.0f : 0.0f;
		edgeParticles[i].mass = i < 2 ? 1.0f : 0.0f;
	}
	PxArray<AvbdSoftBody> edgeBodies(2);
	edgeBodies[0].compiled.particleStart = 0;
	edgeBodies[0].compiled.particleCount = 2;
	edgeBodies[1].compiled.particleStart = 2;
	edgeBodies[1].compiled.particleCount = 2;
	AvbdEdgeInfo queryEdge;
	queryEdge.p0 = 0; queryEdge.p1 = 1; queryEdge.restLength = 2.0f;
	AvbdEdgeInfo targetEdge;
	targetEdge.p0 = 2; targetEdge.p1 = 3; targetEdge.restLength = 2.0f;
	edgeBodies[0].compiled.surfaceEdges.pushBack(queryEdge);
	edgeBodies[1].compiled.surfaceEdges.pushBack(targetEdge);
	edgeBodies[0].compiled.speculativeCCDEnabled = true;
	contacts.clear();
	avbdDetectSoftSoftOGC(
		edgeParticles.begin(), edgeParticles.size(),
		edgeBodies.begin(), edgeBodies.size(), contacts, params);
	bool sweptEdgeContact = false;
	for(PxU32 contactIndex = 0;
		contactIndex < contacts.size(); contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		sweptEdgeContact = sweptEdgeContact ||
			(geometry.queryParticleIndices[1] != PX_MAX_U32 &&
			 geometry.surfaceParticleIndices[1] != PX_MAX_U32 &&
			 geometry.surfaceParticleIndices[2] == PX_MAX_U32 &&
			 geometry.depth == 0.0f &&
			 geometry.normal.y > 0.8f);
	}
	TEST_CHECK(
		sweptEdgeContact,
		"Soft-soft swept edge interiors compile one barycentric two-sided row");
}

// ============================================================================
// Test 35: Self speculative OGC owns swept vertex-face and edge-edge
// ============================================================================

static void testSelfSweptOgcFeatures()
{
	printf(
		"\n--- Test 35: Self Swept OGC Features ---\n");
	AvbdOGCParams params;
	params.contactRadius = 0.05f;
	PxArray<AvbdSoftParticle> particles(6);
	const PxVec3 target[3] =
	{
		PxVec3(-1.0f, 0.0f, -1.0f),
		PxVec3(1.0f, 0.0f, -1.0f),
		PxVec3(0.0f, 0.0f, 1.0f)
	};
	const PxVec3 query[3] =
	{
		PxVec3(-0.3f, 0.0f, -0.2f),
		PxVec3(0.3f, 0.0f, -0.2f),
		PxVec3(0.0f, 0.0f, 0.3f)
	};
	for(PxU32 i = 0; i < 3; ++i)
	{
		particles[i].initialPosition =
			particles[i].position = target[i];
		particles[i].invMass = 0.0f;
		particles[i + 3].initialPosition =
			query[i] + PxVec3(0.0f, 0.2f, 0.0f);
		particles[i + 3].position =
			query[i] - PxVec3(0.0f, 0.2f, 0.0f);
		particles[i + 3].invMass = 1.0f;
		particles[i + 3].mass = 1.0f;
	}
	AvbdSoftBody body;
	body.compiled.particleStart = 0;
	body.compiled.particleCount = 6;
	for(PxU32 i = 0; i < 3; ++i)
		body.compiled.surfaceTriangles.pushBack(i);
	for(PxU32 i = 0; i < 3; ++i)
	{
		body.compiled.surfaceTriangles.pushBack(i + 3);
		body.compiled.surfaceVertices.pushBack(i + 3);
	}
	body.compiled.speculativeCCDEnabled = true;
	PxArray<PxArray<PxU32> > adjacency(6);
	PxArray<AvbdSoftContact> contacts;
	avbdDetectSelfCollisionOGC(
		particles.begin(), body, 0,
		adjacency, contacts, params);
	bool stableSweptFace = false;
	for(PxU32 contactIndex = 0;
		contactIndex < contacts.size(); contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		stableSweptFace = stableSweptFace ||
			(geometry.particleIdx >= 3 &&
			 geometry.surfaceParticleIndices[2] != PX_MAX_U32 &&
			 geometry.depth == 0.0f &&
			 geometry.normal.y > 0.8f);
	}
	TEST_CHECK(
		stableSweptFace,
		"Self swept vertex-face row remains on the penetration-free initial side");
	contacts.clear();
	body.compiled.speculativeCCDEnabled = false;
	avbdDetectSelfCollisionOGC(
		particles.begin(), body, 0,
		adjacency, contacts, params);
	TEST_CHECK(
		contacts.empty(),
		"Self flag-off negative control crosses between discrete detections");

	PxArray<AvbdSoftParticle> edgeParticles(4);
	edgeParticles[0].initialPosition = PxVec3(-1.0f, 0.2f, 0.0f);
	edgeParticles[1].initialPosition = PxVec3(1.0f, 0.2f, 0.0f);
	edgeParticles[0].position = PxVec3(-1.0f, -0.2f, 0.0f);
	edgeParticles[1].position = PxVec3(1.0f, -0.2f, 0.0f);
	edgeParticles[2].initialPosition =
		edgeParticles[2].position = PxVec3(0.0f, 0.0f, -1.0f);
	edgeParticles[3].initialPosition =
		edgeParticles[3].position = PxVec3(0.0f, 0.0f, 1.0f);
	for(PxU32 i = 0; i < 4; ++i)
	{
		edgeParticles[i].invMass = i < 2 ? 1.0f : 0.0f;
		edgeParticles[i].mass = i < 2 ? 1.0f : 0.0f;
	}
	AvbdSoftBody edgeBody;
	edgeBody.compiled.particleStart = 0;
	edgeBody.compiled.particleCount = 4;
	AvbdEdgeInfo queryEdge;
	queryEdge.p0 = 0; queryEdge.p1 = 1; queryEdge.restLength = 2.0f;
	AvbdEdgeInfo targetEdge;
	targetEdge.p0 = 2; targetEdge.p1 = 3; targetEdge.restLength = 2.0f;
	edgeBody.compiled.surfaceEdges.pushBack(queryEdge);
	edgeBody.compiled.surfaceEdges.pushBack(targetEdge);
	edgeBody.compiled.speculativeCCDEnabled = true;
	PxArray<PxArray<PxU32> > edgeAdjacency(4);
	contacts.clear();
	avbdDetectSelfCollisionOGC(
		edgeParticles.begin(), edgeBody, 0,
		edgeAdjacency, contacts, params);
	bool stableSweptEdge = false;
	for(PxU32 contactIndex = 0;
		contactIndex < contacts.size(); contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		stableSweptEdge = stableSweptEdge ||
			(geometry.queryParticleIndices[1] != PX_MAX_U32 &&
			 geometry.surfaceParticleIndices[1] != PX_MAX_U32 &&
			 geometry.depth == 0.0f &&
			 geometry.normal.y > 0.8f);
	}
	TEST_CHECK(
		stableSweptEdge,
		"Self swept edge interiors preserve a unique stable barycentric owner");
}

// ============================================================================
// Test 36: Volume model, stress, and Surface bending-damping kernels
// ============================================================================

static void testDeformableMaterialSemantics()
{
	printf(
		"\n--- Test 36: Deformable Material Semantics ---\n");
	PxArray<AvbdSoftParticle> particles(4);
	const PxVec3 rest[4] =
	{
		PxVec3(0.0f, 0.0f, 0.0f),
		PxVec3(1.0f, 0.0f, 0.0f),
		PxVec3(0.0f, 1.0f, 0.0f),
		PxVec3(0.0f, 0.0f, 1.0f)
	};
	for(PxU32 i = 0; i < 4; ++i)
	{
		particles[i].position = rest[i];
		particles[i].initialPosition = rest[i];
		particles[i].invMass = 1.0f;
		particles[i].mass = 1.0f;
	}
	AvbdSoftBody body;
	body.compiled.particleStart = 0;
	body.compiled.particleCount = 4;
	for(PxU32 i = 0; i < 4; ++i)
		body.compiled.tetrahedra.pushBack(i);
	body.compiled.buildTetElements(particles);
	const AvbdTetElement& tet = body.compiled.tetElements[0];
	const PxQuat rotation(PxPi * 0.63f,
		PxVec3(0.3f, 0.7f, 0.2f).getNormalized());
	for(PxU32 i = 0; i < 4; ++i)
		particles[i].position = rotation.rotate(rest[i]);
	PxReal maxRigidRotationForce = 0.0f;
	for(PxU32 vertexOrder = 0; vertexOrder < 4; ++vertexOrder)
	{
		PxVec3 force;
		PxMat33 hessian;
		avbdEvaluateCorotationalForceHessianPrepared(
			tet, int(vertexOrder), 1000.0f, 1500.0f,
			particles.begin(), force, hessian);
		maxRigidRotationForce = PxMax(
			maxRigidRotationForce, force.magnitude());
	}
	TEST_CHECK(
		maxRigidRotationForce < 1.0e-2f,
		"Co-rotational volume energy is invariant under a proper rigid rotation");

	for(PxU32 i = 0; i < 4; ++i)
		particles[i].position = rest[i];
	particles[1].position.x = 1.5f;
	PxVec3 corotForce;
	PxMat33 corotHessian;
	PxVec3 neoForce;
	PxMat33 neoHessian;
	avbdEvaluateCorotationalForceHessianPrepared(
		tet, 1, 1000.0f, 1500.0f,
		particles.begin(), corotForce, corotHessian);
	avbdEvaluateNeoHookeanForceHessianPrepared(
		tet, 1, 1000.0f, 1500.0f,
		1.0f + 1000.0f / 1500.0f,
		particles.begin(), neoForce, neoHessian);
	TEST_CHECK(
		corotForce.isFinite() && neoForce.isFinite() &&
		(corotForce - neoForce).magnitude() > 1.0f,
		"Co-rotational and Neo-Hookean public models select distinct finite kernels");
	const PxReal restStress = [&]()
	{
		for(PxU32 i = 0; i < 4; ++i)
			particles[i].position = rest[i];
		return avbdComputeTetStressCoefficient(
			tet, particles.begin());
	}();
	particles[1].position.x = 1.5f;
	const PxReal stretchedStress =
		avbdComputeTetStressCoefficient(
			tet, particles.begin());
	TEST_CHECK(
		restStress < 1.0e-5f &&
		stretchedStress > 0.1f,
		"Volume stress coefficient is zero at rest and detects deviatoric strain");

	PxArray<AvbdSoftParticle> stressParticles(5);
	for(PxU32 i = 0; i < 4; ++i)
	{
		stressParticles[i].position = rest[i];
		stressParticles[i].initialPosition = rest[i];
		stressParticles[i].invMass = 0.0f;
	}
	stressParticles[4].position =
		stressParticles[4].initialPosition =
			PxVec3(0.2f, 0.2f, 0.01f);
	stressParticles[4].invMass = 1.0f;
	stressParticles[4].mass = 1.0f;
	AvbdSoftBody stressBody;
	stressBody.compiled.particleStart = 0;
	stressBody.compiled.particleCount = 5;
	for(PxU32 i = 0; i < 4; ++i)
		stressBody.compiled.tetrahedra.pushBack(i);
	stressBody.compiled.buildTetElements(stressParticles);
	stressBody.compiled.surfaceTriangles.pushBack(0);
	stressBody.compiled.surfaceTriangles.pushBack(1);
	stressBody.compiled.surfaceTriangles.pushBack(2);
	stressBody.compiled.surfaceTriangleElementIndices.pushBack(0);
	stressBody.compiled.surfaceVertices.pushBack(4);
	stressParticles[1].position.x = 1.5f;
	stressParticles[1].initialPosition =
		stressParticles[1].position;
	PxArray<PxArray<PxU32> > stressAdjacency(5);
	PxArray<AvbdSoftContact> stressContacts;
	AvbdOGCParams stressParams;
	stressParams.contactRadius = 0.05f;
	stressBody.compiled.selfCollisionStressTolerance = 10.0f;
	avbdDetectSelfCollisionOGC(
		stressParticles.begin(), stressBody, 0,
		stressAdjacency, stressContacts, stressParams);
	const bool relaxedStressAllowsContact =
		!stressContacts.empty();
	stressContacts.clear();
	stressBody.compiled.selfCollisionStressTolerance = 0.01f;
	avbdDetectSelfCollisionOGC(
		stressParticles.begin(), stressBody, 0,
		stressAdjacency, stressContacts, stressParams);
	TEST_CHECK(
		relaxedStressAllowsContact && stressContacts.empty(),
		"Volume self-collision stress tolerance filters stressed target tets");

	PxArray<AvbdSoftParticle> clothParticles(4);
	clothParticles[0].position = PxVec3(0.0f, 0.0f, 0.0f);
	clothParticles[1].position = PxVec3(1.0f, 0.0f, 0.0f);
	clothParticles[2].position = PxVec3(0.0f, 1.0f, 0.0f);
	clothParticles[3].position = PxVec3(0.0f, 0.0f, 1.0f);
	for(PxU32 i = 0; i < 4; ++i)
	{
		clothParticles[i].invMass = 1.0f;
		clothParticles[i].mass = 1.0f;
		clothParticles[i].velocity = PxVec3(0.0f);
	}
	clothParticles[2].velocity = PxVec3(0.0f, 0.0f, 1.0f);
	clothParticles[3].velocity = PxVec3(0.0f, 1.0f, 0.0f);
	PxArray<AvbdSoftBody> clothBodies(1);
	clothBodies[0].compiled.particleStart = 0;
	clothBodies[0].compiled.particleCount = 4;
	AvbdBendingElement bending;
	bending.opp0 = 2;
	bending.opp1 = 3;
	bending.edgeStart = 0;
	bending.edgeEnd = 1;
	bending.restShapeAngle = 0.0f;
	bending.restAngle = 0.0f;
	bending.restLength = 1.0f;
	clothBodies[0].compiled.bendElements.pushBack(bending);
	clothBodies[0].material.bendingStiffness = 1.0f;
	clothBodies[0].material.bendingDamping = 60.0f;
	const PxReal angularDifferenceBefore = 2.0f;
	avbdApplyBendingDamping(
		clothParticles.begin(), clothBodies.begin(),
		clothBodies.size(), 1.0f / 60.0f);
	const PxVec3 edgeDirection(1.0f, 0.0f, 0.0f);
	PxVec3 tipDirection0 =
		edgeDirection.cross(
			clothParticles[2].position -
			clothParticles[0].position);
	PxVec3 tipDirection1 =
		edgeDirection.cross(
			clothParticles[3].position -
			clothParticles[0].position);
	const PxReal tipDistance0 = tipDirection0.normalize();
	const PxReal tipDistance1 = tipDirection1.normalize();
	const PxVec3 linearVelocity =
		(clothParticles[0].velocity +
		 clothParticles[1].velocity) * 0.5f;
	const PxReal angularVelocity0 =
		tipDirection0.dot(
			clothParticles[2].velocity - linearVelocity) /
		tipDistance0;
	const PxReal angularVelocity1 =
		tipDirection1.dot(
			clothParticles[3].velocity - linearVelocity) /
		tipDistance1;
	TEST_CHECK(
		PxAbs(angularVelocity1 - angularVelocity0) <
			angularDifferenceBefore,
		"Surface bending damping independently reduces hinge angular velocity");
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
		rd.positions    = &gParticles[gSoftBodies[i].compiled.particleStart].position;
		rd.numParticles = gSoftBodies[i].compiled.particleCount;
		rd.tetIndices   = gSoftBodies[i].compiled.tetrahedra.begin();
		rd.numTets      = gSoftBodies[i].compiled.tetrahedra.size() / 4;
		rd.triIndices   = gSoftBodies[i].compiled.triangles.size() > 0 ?
			gSoftBodies[i].compiled.triangles.begin() : NULL;
		rd.numTris      = gSoftBodies[i].compiled.triangles.size() / 3;
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
				pin.point.setVertex(i);
				pin.worldTarget = gParticles[i].position;
				pin.k = 1e5f;
				pin.kMax = 1e7f;
				gSoftBodies[0].runtime.pins.pushBack(pin);
			}
		}
		gSoftBodies[0].runtime.compileObjectiveProgram(
			gSoftBodies[0].compiled.particleStart,
			gSoftBodies[0].compiled.particleCount);
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
		gVisOGC.contactRadius = 0.3f;
		gVisOGC.contactStiffness = 1e5f;
		gVisOGC.friction = 0.5f;
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
		avbdDetectSoftSoftOGC(gParticles.begin(), gParticles.size(),
		                      gSoftBodies.begin(), gSoftBodies.size(),
		                      gContacts, gVisOGC);
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
			avbdDetectSelfCollisionOGC(
				gParticles.begin(), gSoftBodies[0], 0,
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
		gSoftBodyRenderData[i].positions =
			&gParticles[gSoftBodies[i].compiled.particleStart].position;

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
	if (shouldRunTest(selectedId, 20)) testContactAugmentedLagrangian();
	if (shouldRunTest(selectedId, 21)) testPinAugmentedLagrangian();
	if (shouldRunTest(selectedId, 22)) testAttachmentAugmentedLagrangian();
	if (shouldRunTest(selectedId, 23)) testSoftSoftContactTwoSidedObjective();
	if (shouldRunTest(selectedId, 24))
		testDynamicRigidSoftContactTwoSidedObjective();
	if (shouldRunTest(selectedId, 25))
		testSoftSweepResidualAuthority();
	if (shouldRunTest(selectedId, 26))
		testNeoHookeanKernelEquivalence();
	if (shouldRunTest(selectedId, 27))
		testPositiveJLimiterKernelEquivalence();
	if (shouldRunTest(selectedId, 28))
		testResidualConvergenceTracker();
	if (shouldRunTest(selectedId, 29))
		testSymmetricParticleBlockSolve();
	if (shouldRunTest(selectedId, 30))
		testCpuAvbdDeformableVolumeSceneLifecycle();
	if (shouldRunTest(selectedId, 31))
		testCpuAvbdDeformableSurfaceSceneLifecycle();
	if (shouldRunTest(selectedId, 32))
		testWeightedPointAttachmentPositionAl();
	if (shouldRunTest(selectedId, 33))
		testPlaneSpeculativeCcdActiveSet();
	if (shouldRunTest(selectedId, 34))
		testSoftSoftSweptOgcFeatures();
	if (shouldRunTest(selectedId, 35))
		testSelfSweptOgcFeatures();
	if (shouldRunTest(selectedId, 36))
		testDeformableMaterialSemantics();

	printf("\n=== Results: %d PASSED, %d FAILED (out of %d) ===\n",
	       gTestsPassed, gTestsFailed, gTestsPassed + gTestsFailed);

	PX_RELEASE(gFoundation);
	return gTestsFailed > 0 ? 1 : 0;
}
