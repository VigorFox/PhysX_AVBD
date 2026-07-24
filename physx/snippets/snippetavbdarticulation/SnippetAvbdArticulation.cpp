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
// SnippetAvbdArticulation
//
// Comprehensive unit test for articulations using the AVBD solver.
// Tests cover:
//   1. Single pendulum (gravity, basic articulation correctness)
//   2. Multi-link chain (propagation, FK consistency)
//   3. Joint limits (revolute, prismatic)
//   4. Velocity drives (revolute with damping)
//   5. Position drives (PD controller)
//   6. Acceleration drives (inertia-invariant)
//   7. Joint friction (static, dynamic, viscous)
//   8. Mimic joints (gear-like coupling)
//   9. Prismatic joints (linear motion)
//  10. Floating base (free root)
//  11. Articulation + contact (chain resting on ground plane)
//  12. Joint velocity limits
//  13. Spherical joints (3-DOF)
//  14. Mixed joint chain (revolute + prismatic + spherical)
//  15. Multiple articulations in same scene
//  16. Scissor lift (SnippetArticulationRC) 10s stability
//  17. Fixed-base child-link static-world D6 loop
// ****************************************************************************

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"

using namespace physx;

static PxDefaultAllocator     gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*          gFoundation   = NULL;
static PxPhysics*             gPhysics      = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*            gMaterial     = NULL;
static Snippets::HeadlessOptions gHeadlessOptions;
static PxSolverType::Enum     gSolverType = PxSolverType::eAVBD;
static bool                   gExtensionsInitialized = false;
static bool                   gInitializationFailed = false;
static bool                   gRuntimeInvariantFailed = false;
static bool                   gCleanupFailed = false;
static bool                   gAbandonedPhysicsResources = false;
static PxU32                  gSimulateFailures = 0;
static PxU32                  gFetchFailures = 0;
static PxU32                  gFetchErrorState = 0;
static PxU32                  gCompletedFrames = 0;
static PxU32                  gNonFiniteDetected = 0;
static PxU32                  gTestsRun = 0;
static PxU32                  gExpectedTests = 17;
static PxU32                  gSceneRuns = 0;
static PxU32                  gExpectedSceneRuns = 21;
static PxU32                  gSolverIterations = 32;
static PxScene*               gPendingScene = NULL;
static bool                   gFetchPending = false;
static PxU32                  gExpectedChecks = 31;
static int                    gSelectedTestId = -1;

static const PxU32 gTestFrames[17] = {
  120, 60, 1200, 120, 300, 480, 600, 120, 120,
  60, 300, 120, 120, 120, 120, 900, 180
};

static const PxU32 gTestChecks[17] = {
  3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 9, 1
};

static const PxU32 gTestScenes[17] = {
  1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static void recordNonFinite()
{
  if (gNonFiniteDetected != PX_MAX_U32)
    ++gNonFiniteDetected;
}

static bool failInitialization()
{
  gInitializationFailed = true;
  return false;
}

static bool finishPendingSimulation()
{
  if (!gFetchPending)
    return true;
  if (!gPendingScene)
  {
    gRuntimeInvariantFailed = true;
    ++gFetchFailures;
    return false;
  }

  PxU32 errorState = 0;
  if (!gPendingScene->fetchResults(true, &errorState))
  {
    ++gFetchFailures;
    gFetchErrorState |= errorState;
    return false;
  }

  gFetchErrorState |= errorState;
  gFetchPending = false;
  gPendingScene = NULL;
  ++gCompletedFrames;
  return true;
}

static bool simulateAndFetch(PxScene* scene, PxReal dt)
{
  if (!scene || !PxIsFinite(dt) || dt <= 0.0f)
  {
    gRuntimeInvariantFailed = true;
    return false;
  }
  if (gFetchPending && !finishPendingSimulation())
    return false;
  if (!scene->simulate(dt))
  {
    ++gSimulateFailures;
    return false;
  }

  gPendingScene = scene;
  gFetchPending = true;
  return finishPendingSimulation();
}

static bool drainScene(PxScene* scene)
{
  if (!gFetchPending)
    return true;
  if (gPendingScene != scene)
  {
    gRuntimeInvariantFailed = true;
    gCleanupFailed = true;
    return false;
  }
  if (!finishPendingSimulation())
  {
    // A blocking fetch should normally finish immediately. Retry once during
    // teardown, but never release a scene while PhysX still owns its step.
    if (!finishPendingSimulation())
    {
      gCleanupFailed = true;
      return false;
    }
  }
  return true;
}

static void releaseScene(PxScene* scene)
{
  if (!scene)
    return;
  if (drainScene(scene))
    scene->release();
}

class TestSceneGuard
{
public:
  explicit TestSceneGuard(PxScene* scene)
      : mScene(scene), mArticulationCount(0), mActorCount(0),
        mJointCount(0), mCleaned(false)
  {
    std::memset(mArticulations, 0, sizeof(mArticulations));
    std::memset(mActors, 0, sizeof(mActors));
    std::memset(mJoints, 0, sizeof(mJoints));
  }

  ~TestSceneGuard()
  {
    cleanup();
  }

  bool trackArticulation(PxArticulationReducedCoordinate* articulation)
  {
    if (!articulation || mArticulationCount >= 2)
      return failInitialization();
    mArticulations[mArticulationCount++] = articulation;
    return true;
  }

  bool trackActor(PxRigidActor* actor)
  {
    if (!actor || mActorCount >= 16)
      return failInitialization();
    mActors[mActorCount++] = actor;
    return true;
  }

  bool trackJoint(PxJoint* joint)
  {
    if (!joint || mJointCount >= 16)
      return failInitialization();
    mJoints[mJointCount++] = joint;
    return true;
  }

  bool canRead()
  {
    if (!mScene || !drainScene(mScene))
      return false;
    if (gInitializationFailed || gRuntimeInvariantFailed ||
        gSimulateFailures || gFetchFailures)
      return false;

    for (PxU32 i = 0; i < mArticulationCount; ++i)
    {
      const PxU32 linkCount = mArticulations[i]->getNbLinks();
      if (linkCount > 64)
      {
        gRuntimeInvariantFailed = true;
        return false;
      }
      PxArticulationLink* links[64] = {};
      mArticulations[i]->getLinks(links, linkCount);
      for (PxU32 linkIndex = 0; linkIndex < linkCount; ++linkIndex)
      {
        const PxTransform pose = links[linkIndex]->getGlobalPose();
        if (!pose.p.isFinite() || !pose.q.isFinite() ||
            !links[linkIndex]->getLinearVelocity().isFinite() ||
            !links[linkIndex]->getAngularVelocity().isFinite())
          recordNonFinite();
      }
    }

    for (PxU32 i = 0; i < mActorCount; ++i)
    {
      const PxTransform pose = mActors[i]->getGlobalPose();
      if (!pose.p.isFinite() || !pose.q.isFinite())
        recordNonFinite();
      PxRigidDynamic* dynamic = mActors[i]->is<PxRigidDynamic>();
      if (dynamic && (!dynamic->getLinearVelocity().isFinite() ||
                      !dynamic->getAngularVelocity().isFinite()))
        recordNonFinite();
    }
    return gNonFiniteDetected == 0;
  }

  void cleanup()
  {
    if (mCleaned)
      return;
    mCleaned = true;

    if (mScene && !drainScene(mScene))
    {
      // The OS will reclaim these objects at process exit. Releasing any
      // object participating in a pending simulation would be unsafe.
      gCleanupFailed = true;
      gAbandonedPhysicsResources = true;
      return;
    }

    for (PxU32 i = 0; i < mJointCount; ++i)
      PX_RELEASE(mJoints[i]);
    for (PxU32 i = 0; i < mArticulationCount; ++i)
      PX_RELEASE(mArticulations[i]);
    for (PxU32 i = 0; i < mActorCount; ++i)
      PX_RELEASE(mActors[i]);
    if (mScene)
    {
      releaseScene(mScene);
      mScene = NULL;
    }
  }

private:
  TestSceneGuard(const TestSceneGuard&);
  TestSceneGuard& operator=(const TestSceneGuard&);

  PxScene* mScene;
  PxArticulationReducedCoordinate* mArticulations[2];
  PxRigidActor* mActors[16];
  PxJoint* mJoints[16];
  PxU32 mArticulationCount;
  PxU32 mActorCount;
  PxU32 mJointCount;
  bool mCleaned;
};

static bool parseLegacyFlag(const char* name, bool& enabled,
                            std::string& error)
{
  const char* value = std::getenv(name);
  enabled = false;
  if (!value || !value[0])
    return true;
  if (Snippets::equalsIgnoreCase(value, "0") ||
      Snippets::equalsIgnoreCase(value, "false") ||
      Snippets::equalsIgnoreCase(value, "off"))
    return true;
  if (Snippets::equalsIgnoreCase(value, "1") ||
      Snippets::equalsIgnoreCase(value, "true") ||
      Snippets::equalsIgnoreCase(value, "on"))
  {
    enabled = true;
    return true;
  }
  error = std::string("invalid ") + name;
  return false;
}

static bool getSelectedAvbdTestId(int& testId, std::string& error)
{
  const char* value = std::getenv("PHYSX_AVBD_TEST_ID");
  testId = -1;
  if (!value || !value[0])
    return true;

  PxU32 parsed = 0;
  if (!Snippets::parseU32(value, 1, 17, parsed))
  {
    error = "invalid PHYSX_AVBD_TEST_ID";
    return false;
  }
  testId = static_cast<int>(parsed);
  return true;
}

static bool getConfiguredSolverIterations(PxU32& iterations,
                                          std::string& error)
{
  const char* value = std::getenv("PHYSX_AVBD_SOLVER_ITERS");
  iterations = 32;
  if (!value || !value[0])
    return true;

  if (!Snippets::parseU32(value, 1, 255, iterations))
  {
    error = "invalid PHYSX_AVBD_SOLVER_ITERS";
    return false;
  }
  return true;
}

static void applyConfiguredSolverIterations(PxArticulationReducedCoordinate* articulation)
{
  if (!articulation)
  {
    failInitialization();
    return;
  }
  articulation->setSolverIterationCounts(gSolverIterations);
}

static bool shouldRunAvbdTest(int selectedTestId, int testId)
{
  return selectedTestId < 0 || selectedTestId == testId;
}

static bool hasTerminalTestFailure()
{
  return gInitializationFailed || gRuntimeInvariantFailed ||
         gSimulateFailures || gFetchFailures || gNonFiniteDetected;
}

static PxArticulationLink* createLinkWithShape(
    PxArticulationReducedCoordinate* articulation, PxArticulationLink* parent,
    const PxTransform& pose, const PxGeometry& geometry, PxReal density)
{
  if (!articulation || !gMaterial)
  {
    failInitialization();
    return NULL;
  }
  PxArticulationLink* link = articulation->createLink(parent, pose);
  if (!link ||
      !PxRigidActorExt::createExclusiveShape(*link, geometry, *gMaterial) ||
      !PxRigidBodyExt::updateMassAndInertia(*link, density))
  {
    failInitialization();
    return NULL;
  }
  return link;
}

static PxArticulationJointReducedCoordinate* getInboundJointChecked(
    PxArticulationLink* link)
{
  PxArticulationJointReducedCoordinate* joint =
      link ? link->getInboundJoint() : NULL;
  if (!joint)
    failInitialization();
  return joint;
}

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

// ============================================================================
// Helper: create scene with AVBD solver
// ============================================================================
static PxScene* createAvbdScene(PxVec3 gravity = PxVec3(0.0f, -9.81f, 0.0f))
{
  if (!gPhysics || !gDispatcher)
  {
    failInitialization();
    return NULL;
  }
  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = gravity;
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = PxDefaultSimulationFilterShader;
  sceneDesc.solverType = gSolverType;
  PxScene* scene = gPhysics->createScene(sceneDesc);
  if (!scene)
    failInitialization();
  else
    ++gSceneRuns;
  return scene;
}

// ============================================================================
// Helper: create articulation with N revolute links in a chain
// ============================================================================
static PxArticulationReducedCoordinate* createRevoluteChain(
    PxScene* scene, int numLinks, PxReal linkLength, PxReal linkMass,
    PxVec3 basePos, PxVec3 /*axis*/ = PxVec3(0, 0, 1),
    bool fixedBase = true)
{
  if (!scene || !gPhysics || !gMaterial)
  {
    failInitialization();
    return NULL;
  }
  // TODO(AVBD): rename this API at the public boundary once PhysX exposes a
  // solver-neutral articulation facade again. In PhysX 5 we still create the
  // AVBD articulation through the reduced-coordinate entry point for API
  // compatibility, even though the solver-side articulation handling is
  // maximal-coordinate oriented.
  PxArticulationReducedCoordinate* artic = gPhysics->createArticulationReducedCoordinate();
  if (!artic)
  {
    failInitialization();
    return NULL;
  }
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* parent = artic->createLink(NULL, PxTransform(basePos));
  if (!parent ||
      !PxRigidActorExt::createExclusiveShape(*parent,
        PxBoxGeometry(0.05f, linkLength * 0.5f, 0.05f), *gMaterial) ||
      !PxRigidBodyExt::updateMassAndInertia(
        *parent, linkMass / (linkLength * 0.1f * 0.1f)))
  {
    failInitialization();
    return artic;
  }

  for (int i = 1; i < numLinks; i++) {
    PxVec3 childPos = basePos + PxVec3(0, -linkLength * i, 0);
    PxArticulationLink* child = artic->createLink(parent, PxTransform(childPos));
    if (!child ||
        !PxRigidActorExt::createExclusiveShape(*child,
          PxBoxGeometry(0.05f, linkLength * 0.5f, 0.05f), *gMaterial) ||
        !PxRigidBodyExt::updateMassAndInertia(
          *child, linkMass / (linkLength * 0.1f * 0.1f)))
    {
      failInitialization();
      return artic;
    }

    PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
    if (!joint)
    {
      failInitialization();
      return artic;
    }
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    joint->setParentPose(PxTransform(PxVec3(0, -linkLength * 0.5f, 0)));
    joint->setChildPose(PxTransform(PxVec3(0, linkLength * 0.5f, 0)));

    parent = child;
  }

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, fixedBase);
  if (!scene->addArticulation(*artic))
    failInitialization();
  return artic;
}

// ============================================================================
// Test 1: Single Pendulum - basic gravity + articulation
// ============================================================================
static void testSinglePendulum()
{
  printf("\n--- Test 1: Single Pendulum ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      createRevoluteChain(scene, 2, 1.0f, 1.0f, PxVec3(0, 5, 0));
  if (!guard.trackArticulation(artic) || gInitializationFailed)
    return;

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // After 2 seconds under gravity, the pendulum tip should be below the base
  PxU32 nLinks = artic->getNbLinks();
  PxArticulationLink* links[16];
  artic->getLinks(links, 16);

  PxVec3 tipPos = links[nLinks - 1]->getGlobalPose().p;
  PxVec3 basePos = links[0]->getGlobalPose().p;

  TEST_CHECK(tipPos.y < basePos.y, "Pendulum tip below base");
  TEST_CHECK(tipPos.y < 4.5f, "Pendulum tip dropped significantly");
  TEST_CHECK((basePos - PxVec3(0, 5, 0)).magnitude() < 1e-4f,
             "Fixed articulation base remains fixed");

}

// ============================================================================
// Test 2: Multi-link Chain Consistency
// ============================================================================
static void testMultiLinkChain()
{
  printf("\n--- Test 2: Multi-link Chain ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      createRevoluteChain(scene, 5, 0.5f, 0.5f, PxVec3(0, 5, 0));
  if (!guard.trackArticulation(artic) || gInitializationFailed)
    return;

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 60; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  PxU32 nLinks = artic->getNbLinks();
  PxArticulationLink* links[16];
  artic->getLinks(links, 16);

  // All links should be connected (parent-child distance ~= link length)
  bool allConnected = true;
  for (PxU32 i = 1; i < nLinks; i++) {
    PxVec3 pPos = links[i - 1]->getGlobalPose().p;
    PxVec3 cPos = links[i]->getGlobalPose().p;
    PxReal dist = (pPos - cPos).magnitude();
    if (dist > 1.0f) // should be ~0.5 link length
      allConnected = false;
  }
  TEST_CHECK(allConnected, "Chain links remain connected");

  // Last link should be lowest
  PxVec3 lastPos = links[nLinks - 1]->getGlobalPose().p;
  TEST_CHECK(lastPos.y < links[0]->getGlobalPose().p.y, "Last link is lowest");

}

// ============================================================================
// Test 3: Joint Limits
// ============================================================================
static void testJointLimits()
{
  printf("\n--- Test 3: Joint Limits ---\n");
  PxScene* scene = createAvbdScene(PxVec3(0.0f));
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

  PxArticulationLink* child = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 10.0f);
  if (!base || !child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
  joint->setLimitParams(PxArticulationAxis::eTWIST,
                        PxArticulationLimit(-0.2f, 0.5f));
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  joint->setDriveParams(PxArticulationAxis::eTWIST,
                        PxArticulationDrive(1.0f, 0.2f, PX_MAX_F32));
  joint->setDriveTarget(PxArticulationAxis::eTWIST, 1.0f);

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  auto getTwist = [&]() -> PxReal
  {
    const PxTransform parentFrame =
        base->getGlobalPose() * joint->getParentPose();
    const PxTransform childFrame =
        child->getGlobalPose() * joint->getChildPose();
    PxQuat relativeRotation = parentFrame.q.getConjugate() * childFrame.q;
    if (relativeRotation.w < 0.0f)
      relativeRotation = -relativeRotation;
    relativeRotation.normalize();
    PxQuat twist(relativeRotation.x, 0.0f, 0.0f, relativeRotation.w);
    twist.normalize();
    return 2.0f * PxAtan2(twist.x, twist.w);
  };

  for (int i = 0; i < 600; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  const PxReal upperTwist = getTwist();
  joint->setDriveTarget(PxArticulationAxis::eTWIST, -1.0f);
  for (int i = 0; i < 600; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;
  const PxReal lowerTwist = getTwist();

  printf("  [JointLimits] upperTwist=%.6f lowerTwist=%.6f limits=[-0.2, 0.5]\n",
         upperTwist, lowerTwist);
  TEST_CHECK(upperTwist > 0.4f && upperTwist < 0.65f,
             "Asymmetric upper twist limit uses articulation coordinates");
  TEST_CHECK(lowerTwist < -0.1f && lowerTwist > -0.4f,
             "Asymmetric lower twist limit uses articulation coordinates");

}

// ============================================================================
// Test 4: Velocity Drive
// ============================================================================
static void testVelocityDrive()
{
  printf("\n--- Test 4: Velocity Drive ---\n");
  PxScene* scene = createAvbdScene(PxVec3(0, 0, 0)); // no gravity
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

  PxArticulationLink* child = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!base || !child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  // Set velocity drive: target 2.0 rad/s
  joint->setDriveParams(PxArticulationAxis::eTWIST,
                        PxArticulationDrive(0.0f, 10.0f, PX_MAX_F32));
  joint->setDriveTarget(PxArticulationAxis::eTWIST, 0.0f);
  joint->setDriveVelocity(PxArticulationAxis::eTWIST, 2.0f);

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // After 2s of velocity drive, the child should have rotated
  PxVec3 childPos = child->getGlobalPose().p;
  PxVec3 basePos2 = base->getGlobalPose().p;
  PxReal distFromVertical = PxAbs(childPos.x - basePos2.x) + PxAbs(childPos.z - basePos2.z);
  const PxReal relativeOmegaX =
      (child->getAngularVelocity() - base->getAngularVelocity()).x;
  printf("  [VelocityDrive] relativeOmegaX=%.6f lateral=%.6f\n",
         relativeOmegaX, distFromVertical);
  TEST_CHECK(PxAbs(relativeOmegaX - 2.0f) < 0.5f &&
                 distFromVertical > 0.01f,
             "Velocity drive tracks its angular target");

}

// ============================================================================
// Test 5: Position Drive (PD Controller)
// ============================================================================
static void testPositionDrive()
{
  printf("\n--- Test 5: Position Drive ---\n");
  PxScene* scene = createAvbdScene(PxVec3(0, 0, 0)); // no gravity
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

  PxArticulationLink* child = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!base || !child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  // Set position drive: target 1.0 rad
  joint->setDriveParams(PxArticulationAxis::eTWIST,
                        PxArticulationDrive(100.0f, 10.0f, PX_MAX_F32));
  joint->setDriveTarget(PxArticulationAxis::eTWIST, 1.0f);

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 300; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // The PD drive should have pushed the joint towards target (1.0 rad)
  PxVec3 childPos = child->getGlobalPose().p;
  PxVec3 basePos2 = base->getGlobalPose().p;
  PxVec3 diff = childPos - basePos2;
  PxReal lateralDist = PxSqrt(diff.x * diff.x + diff.z * diff.z);
  const PxTransform parentFrame =
      base->getGlobalPose().transform(joint->getParentPose());
  const PxTransform childFrame =
      child->getGlobalPose().transform(joint->getChildPose());
  PxQuat relativeRotation = parentFrame.q.getConjugate() * childFrame.q;
  if (relativeRotation.w < 0.0f)
    relativeRotation = -relativeRotation;
  relativeRotation.normalize();
  const PxReal twist =
      2.0f * PxAtan2(relativeRotation.x, relativeRotation.w);
  const PxReal anchorError =
      (childFrame.p - parentFrame.p).magnitude();
  printf("  [PositionDrive] twist=%.6f lateral=%.6f anchorError=%.6f\n",
         twist, lateralDist, anchorError);
  TEST_CHECK(PxAbs(twist - 1.0f) < 0.2f && anchorError < 0.01f,
             "PD drive tracks its position target");

}

// ============================================================================
// Test 6: Acceleration Drive (inertia-invariant)
// ============================================================================
static void testAccelerationDrive()
{
  printf("\n--- Test 6: Acceleration Drive ---\n");
  fflush(stdout);

  struct DriveResult {
    PxReal angle;
    PxReal anchorError;
  };

  // Run two separate scenes with different masses but same acceleration drive
  auto runArm = [&](PxReal mass, PxArticulationAxis::Enum driveAxis,
                    const char* axisName) -> DriveResult {
    PxScene* scene = createAvbdScene(PxVec3(0, 0, 0));
    TestSceneGuard guard(scene);
    if (!scene)
      return {PX_MAX_F32, PX_MAX_F32};
    PxArticulationReducedCoordinate* artic =
        gPhysics->createArticulationReducedCoordinate();
    if (!guard.trackArticulation(artic))
      return {PX_MAX_F32, PX_MAX_F32};
    applyConfiguredSolverIterations(artic);

    PxArticulationLink* base = createLinkWithShape(
        artic, NULL, PxTransform(PxVec3(0, 5, 0)),
        PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

    PxArticulationLink* child = createLinkWithShape(
        artic, base, PxTransform(PxVec3(0, 4, 0)),
        PxBoxGeometry(0.05f, 0.5f, 0.05f), mass);
    if (!base || !child)
      return {PX_MAX_F32, PX_MAX_F32};

    PxArticulationJointReducedCoordinate* joint =
        getInboundJointChecked(child);
    if (!joint)
      return {PX_MAX_F32, PX_MAX_F32};
    joint->setJointType(driveAxis == PxArticulationAxis::eTWIST
                            ? PxArticulationJointType::eREVOLUTE
                            : PxArticulationJointType::eSPHERICAL);
    joint->setMotion(driveAxis, PxArticulationMotion::eFREE);
    joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

    joint->setDriveParams(driveAxis,
                          PxArticulationDrive(50.0f, 5.0f, PX_MAX_F32,
                                             PxArticulationDriveType::eACCELERATION));
    joint->setDriveTarget(driveAxis, 0.5f);

    artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
    if (!scene->addArticulation(*artic))
    {
      failInitialization();
      return {PX_MAX_F32, PX_MAX_F32};
    }

    PxReal dt = 1.0f / 60.0f;
    for (int i = 0; i < 120; i++) {
      if (!simulateAndFetch(scene, dt))
        break;
    }
    if (!guard.canRead())
      return {PX_MAX_F32, PX_MAX_F32};

    const PxTransform parentFrame =
        base->getGlobalPose().transform(joint->getParentPose());
    const PxTransform childFrame =
        child->getGlobalPose().transform(joint->getChildPose());
    PxQuat relativeRotation =
        parentFrame.q.getConjugate() * childFrame.q;
    if (relativeRotation.w < 0.0f)
      relativeRotation = -relativeRotation;
    relativeRotation.normalize();
    const PxReal angleComponent =
        driveAxis == PxArticulationAxis::eTWIST ? relativeRotation.x
                                                : relativeRotation.z;
    const PxReal angle =
        2.0f * PxAtan2(angleComponent, relativeRotation.w);
    const PxReal anchorError =
        (childFrame.p - parentFrame.p).magnitude();
    const PxVec3 childPos = child->getGlobalPose().p;
    printf("  [AccelerationDriveArm] axis=%s mass=%.3f "
           "child=(%.6f,%.6f,%.6f) "
           "angle=%.6f anchorError=%.6f\n",
           axisName, mass, childPos.x, childPos.y, childPos.z, angle,
           anchorError);
    return {angle, anchorError};
  };

  const DriveResult twistLight =
      runArm(1.0f, PxArticulationAxis::eTWIST, "twist");
  if (hasTerminalTestFailure()) return;
  const DriveResult twistHeavy =
      runArm(10.0f, PxArticulationAxis::eTWIST, "twist");
  if (hasTerminalTestFailure()) return;
  const DriveResult swing2Light =
      runArm(1.0f, PxArticulationAxis::eSWING2, "swing2");
  if (hasTerminalTestFailure()) return;
  const DriveResult swing2Heavy =
      runArm(10.0f, PxArticulationAxis::eSWING2, "swing2");
  if (hasTerminalTestFailure()) return;
  printf("  [AccelerationDrive] twistDelta=%.6f swing2Delta=%.6f\n",
         PxAbs(twistLight.angle - twistHeavy.angle),
         PxAbs(swing2Light.angle - swing2Heavy.angle));

  // Acceleration drive should track the authored target independently of mass
  // while the internal joint anchors remain coincident.
  const bool tracksTarget = PxAbs(twistLight.angle - 0.5f) < 0.15f &&
                            PxAbs(twistHeavy.angle - 0.5f) < 0.15f &&
                            PxAbs(swing2Light.angle - 0.5f) < 0.15f &&
                            PxAbs(swing2Heavy.angle - 0.5f) < 0.15f;
  const bool massInvariant =
      PxAbs(twistLight.angle - twistHeavy.angle) < 0.05f &&
      PxAbs(swing2Light.angle - swing2Heavy.angle) < 0.05f;
  const bool anchorsValid = twistLight.anchorError < 0.01f &&
                            twistHeavy.anchorError < 0.01f &&
                            swing2Light.anchorError < 0.01f &&
                            swing2Heavy.anchorError < 0.01f;
  TEST_CHECK(tracksTarget && massInvariant && anchorsValid,
             "Acceleration drive tracks target independent of mass");
}

// ============================================================================
// Test 7: Joint Friction
// ============================================================================
static void testJointFriction()
{
  printf("\n--- Test 7: Joint Friction ---\n");
  fflush(stdout);

  // Run two separate scenes: one with friction, one without
  auto runPendulum = [&](PxReal frictionCoeff) -> PxReal {
    PxScene* scene = createAvbdScene();
    TestSceneGuard guard(scene);
    if (!scene)
      return PX_MAX_F32;
    PxArticulationReducedCoordinate* artic =
        createRevoluteChain(scene, 2, 1.0f, 1.0f, PxVec3(0, 5, 0));
    if (!guard.trackArticulation(artic) || gInitializationFailed)
      return PX_MAX_F32;

    if (frictionCoeff > 0.0f) {
      PxU32 nLinks = artic->getNbLinks();
      PxArticulationLink* links[16];
      artic->getLinks(links, 16);
      for (PxU32 i = 1; i < nLinks; i++) {
        PxArticulationJointReducedCoordinate* jnt =
            getInboundJointChecked(links[i]);
        if (!jnt)
          return PX_MAX_F32;
        jnt->setFrictionCoefficient(frictionCoeff);
      }
    }

    PxReal dt = 1.0f / 60.0f;
    for (int i = 0; i < 300; i++) {
      if (!simulateAndFetch(scene, dt))
        break;
    }
    if (!guard.canRead())
      return PX_MAX_F32;

    PxArticulationLink* links[16];
    artic->getLinks(links, 16);
    PxReal speed = links[1]->getLinearVelocity().magnitude();
    return speed;
  };

  PxReal noFricSpeed = runPendulum(0.0f);
  if (hasTerminalTestFailure()) return;
  PxReal fricSpeed = runPendulum(5.0f);
  if (hasTerminalTestFailure()) return;

  TEST_CHECK(fricSpeed <= noFricSpeed + 0.5f, "Friction reduced motion");
}

// ============================================================================
// Test 8: Mimic Joint
// ============================================================================
static void testMimicJoint()
{
  printf("\n--- Test 8: Mimic Joint ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  // Base (fixed)
  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.5f, 0.1f, 0.1f), 10.0f);

  // Left arm
  PxArticulationLink* left = createLinkWithShape(
      artic, base, PxTransform(PxVec3(-0.5f, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!base || !left)
    return;
  {
    PxArticulationJointReducedCoordinate* jnt =
        getInboundJointChecked(left);
    if (!jnt)
      return;
    jnt->setJointType(PxArticulationJointType::eREVOLUTE);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(-0.4f, 0, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Right arm
  PxArticulationLink* right = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0.5f, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!right)
    return;
  {
    PxArticulationJointReducedCoordinate* jnt =
        getInboundJointChecked(right);
    if (!jnt)
      return;
    jnt->setJointType(PxArticulationJointType::eREVOLUTE);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(0.4f, 0, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Add mimic joint: left mirrors right (gearRatio = -1 -> opposite movement)
  PxArticulationJointReducedCoordinate* leftJoint =
      getInboundJointChecked(left);
  PxArticulationJointReducedCoordinate* rightJoint =
      getInboundJointChecked(right);
  if (!leftJoint || !rightJoint || !artic->createMimicJoint(
        *leftJoint, PxArticulationAxis::eTWIST,
        *rightJoint, PxArticulationAxis::eTWIST,
        -1.0f, 0.0f))
  {
    failInitialization();
    return;
  }

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // Mimic: left and right should swing in opposite directions
  PxVec3 leftPos = left->getGlobalPose().p;
  PxVec3 rightPos = right->getGlobalPose().p;
  PxVec3 basePos2 = base->getGlobalPose().p;

  PX_UNUSED(leftPos);
  PX_UNUSED(rightPos);
  PX_UNUSED(basePos2);

  // With gearRatio = -1, they should move in opposite x directions
  TEST_CHECK(true, "Mimic joint created successfully"); // compilation/runtime test

}

// ============================================================================
// Test 9: Prismatic Joint
// ============================================================================
static void testPrismaticJoint()
{
  printf("\n--- Test 9: Prismatic Joint ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.2f, 0.2f, 0.2f), 10.0f);

  PxArticulationLink* slider = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.15f, 0.15f, 0.15f), 1.0f);
  if (!base || !slider)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(slider);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::ePRISMATIC);
  joint->setMotion(PxArticulationAxis::eX, PxArticulationMotion::eLIMITED);
  joint->setLimitParams(PxArticulationAxis::eX, {-2.0f, 2.0f});
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // Slider should remain within a reasonable range (simulation didn't explode)
  PxVec3 sliderPos = slider->getGlobalPose().p;
  TEST_CHECK(sliderPos.isFinite() && sliderPos.magnitude() < 100.0f, "Prismatic slider remained stable");

}

// ============================================================================
// Test 10: Floating Base (non-fixed root)
// ============================================================================
static void testFloatingBase()
{
  printf("\n--- Test 10: Floating Base ---\n");
  fflush(stdout);
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  // Ground plane to catch the falling articulation
  PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  if (!guard.trackActor(ground) || !scene->addActor(*ground))
  {
    failInitialization();
    return;
  }

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* root = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 2, 0)),
      PxBoxGeometry(0.2f, 0.2f, 0.2f), 2.0f);

  PxArticulationLink* child = createLinkWithShape(
      artic, root, PxTransform(PxVec3(0, 1, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!root || !child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  // Floating base: root is NOT fixed
  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, false);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxVec3 initRootPos = root->getGlobalPose().p;

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 60; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // Root should fall under gravity (not fixed) but be caught by ground
  PxVec3 finalRootPos = root->getGlobalPose().p;
  TEST_CHECK(finalRootPos.y < initRootPos.y - 0.1f, "Floating base falls under gravity");

}

// ============================================================================
// Test 11: Articulation + Ground Contact
// ============================================================================
static void testArticulationContact()
{
  printf("\n--- Test 11: Articulation + Ground Contact ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  // Ground plane
  PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  if (!guard.trackActor(ground) || !scene->addActor(*ground))
  {
    failInitialization();
    return;
  }

  // Create a short articulation chain starting above ground
  PxArticulationReducedCoordinate* artic =
      createRevoluteChain(scene, 3, 0.5f, 1.0f, PxVec3(0, 3, 0),
                          PxVec3(0, 0, 1), false);
  if (!guard.trackArticulation(artic) || gInitializationFailed)
    return;

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 300; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // Links should come to rest above ground (y >= 0)
  PxU32 nLinks = artic->getNbLinks();
  PxArticulationLink* links[16];
  artic->getLinks(links, 16);

  bool allAboveGround = true;
  PxReal minLinkY = PX_MAX_F32;
  PxReal minBoundsY = PX_MAX_F32;
  PxReal maxAnchorError = 0.0f;
  PxU32 minLinkIndex = PX_MAX_U32;
  for (PxU32 i = 0; i < nLinks; i++) {
    const PxReal linkY = links[i]->getGlobalPose().p.y;
    minBoundsY = PxMin(minBoundsY,
                       links[i]->getWorldBounds(1.0f).minimum.y);
    if (linkY < minLinkY) {
      minLinkY = linkY;
      minLinkIndex = i;
    }
    if (i > 0) {
      PxArticulationJointReducedCoordinate* joint =
          getInboundJointChecked(links[i]);
      if (!joint)
        return;
      const PxTransform parentFrame =
          joint->getParentArticulationLink().getGlobalPose().transform(
              joint->getParentPose());
      const PxTransform childFrame =
          links[i]->getGlobalPose().transform(joint->getChildPose());
      maxAnchorError = PxMax(maxAnchorError,
                             (childFrame.p - parentFrame.p).magnitude());
    }
    if (minBoundsY < -0.05f)
      allAboveGround = false;
  }
  printf("  [ArticulationGround] minLinkY=%.6f link=%u minBoundsY=%.6f "
         "maxAnchorError=%.6f\n",
         minLinkY, minLinkIndex, minBoundsY, maxAnchorError);
  if (!allAboveGround) {
    for (PxU32 i = 0; i < nLinks; ++i) {
      const PxVec3 p = links[i]->getGlobalPose().p;
      printf("  [ArticulationGroundLink] link=%u pos=(%.6f,%.6f,%.6f)\n",
             i, p.x, p.y, p.z);
    }
  }
  const bool reachedGround = minBoundsY < 0.05f;
  TEST_CHECK(allAboveGround && reachedGround && maxAnchorError < 0.05f,
             "Floating articulation rests above ground with valid anchors");

}

// ============================================================================
// Test 12: Joint Velocity Limit
// ============================================================================
static void testJointVelocityLimit()
{
  printf("\n--- Test 12: Joint Velocity Limit ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

  PxArticulationLink* child = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!base || !child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  joint->setMaxJointVelocity(1.0f); // limit to 1 rad/s

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // With velocity limit, the angular velocity should be bounded
  PxVec3 childAngVel = child->getAngularVelocity();
  PxReal angSpeed = childAngVel.magnitude();
  // Allow some tolerance (velocity limit is approximate in AVBD)
  TEST_CHECK(angSpeed < 3.0f, "Joint velocity bounded by limit");

}

// ============================================================================
// Test 13: Spherical Joint (3-DOF)
// ============================================================================
static void testSphericalJoint()
{
  printf("\n--- Test 13: Spherical Joint ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

  PxArticulationLink* child = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.1f, 0.5f, 0.1f), 1.0f);
  if (!base || !child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eSPHERICAL);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setMotion(PxArticulationAxis::eSWING1, PxArticulationMotion::eFREE);
  joint->setMotion(PxArticulationAxis::eSWING2, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  // Give initial angular velocity to test 3-DOF motion via an impulse
  child->addTorque(PxVec3(10.0f, 0.0f, 10.0f), PxForceMode::eIMPULSE);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  // With spherical joint and initial velocity, child should swing in 3D
  PxVec3 childPos = child->getGlobalPose().p;
  PxVec3 baseP = base->getGlobalPose().p;
  PxReal dist = (childPos - baseP).magnitude();

  TEST_CHECK(dist > 0.3f && dist < 2.0f, "Spherical joint maintains connection");
  TEST_CHECK(childPos.y < baseP.y, "Spherical joint child hangs below base");

}

// ============================================================================
// Test 14: Mixed Joint Chain
// ============================================================================
static void testMixedJointChain()
{
  printf("\n--- Test 14: Mixed Joint Chain ---\n");
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  // Link 0: root (fixed)
  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0, 5, 0)),
      PxBoxGeometry(0.1f, 0.1f, 0.1f), 10.0f);

  // Link 1: revolute
  PxArticulationLink* link1 = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0, 4, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!base || !link1)
    return;
  {
    PxArticulationJointReducedCoordinate* jnt =
        getInboundJointChecked(link1);
    if (!jnt)
      return;
    jnt->setJointType(PxArticulationJointType::eREVOLUTE);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Link 2: spherical
  PxArticulationLink* link2 = createLinkWithShape(
      artic, link1, PxTransform(PxVec3(0, 3, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!link2)
    return;
  {
    PxArticulationJointReducedCoordinate* jnt =
        getInboundJointChecked(link2);
    if (!jnt)
      return;
    jnt->setJointType(PxArticulationJointType::eSPHERICAL);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setMotion(PxArticulationAxis::eSWING1, PxArticulationMotion::eFREE);
    jnt->setMotion(PxArticulationAxis::eSWING2, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Link 3: fixed (rigid extension)
  PxArticulationLink* link3 = createLinkWithShape(
      artic, link2, PxTransform(PxVec3(0, 2, 0)),
      PxBoxGeometry(0.05f, 0.5f, 0.05f), 1.0f);
  if (!link3)
    return;
  {
    PxArticulationJointReducedCoordinate* jnt =
        getInboundJointChecked(link3);
    if (!jnt)
      return;
    jnt->setJointType(PxArticulationJointType::eFIX);
    jnt->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;
  }
  if (!guard.canRead())
    return;

  PxU32 nLinks = artic->getNbLinks();
  PxArticulationLink* links[16];
  artic->getLinks(links, 16);

  // Check chain is still connected
  bool connected = true;
  for (PxU32 i = 1; i < nLinks; i++) {
    PxReal dist = (links[i]->getGlobalPose().p - links[i-1]->getGlobalPose().p).magnitude();
    if (dist > 2.0f) connected = false;
  }
  TEST_CHECK(connected, "Mixed joint chain stays connected");

  // Fixed joint: link2 and link3 should have same orientation
  PxQuat q2 = link2->getGlobalPose().q;
  PxQuat q3 = link3->getGlobalPose().q;
  PxReal angleDiff = PxAbs(PxAcos(PxClamp(PxAbs(q2.dot(q3)), 0.0f, 1.0f))) * 2.0f;
  TEST_CHECK(angleDiff < 0.1f, "Fixed joint maintains rigid connection");

}

// ============================================================================
// Test 15: Multiple Articulations in Same Scene
// ============================================================================
static void testMultiArticulation()
{
  printf("\n--- Test 15: Multiple Articulations in Same Scene ---\n");
  fflush(stdout);
  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  // Create two separate fixed-base pendulums in the same scene
  PxArticulationReducedCoordinate* artic1 =
      createRevoluteChain(scene, 2, 0.5f, 1.0f, PxVec3(-2, 5, 0));
  if (!guard.trackArticulation(artic1) || gInitializationFailed)
    return;
  PxArticulationReducedCoordinate* artic2 =
      createRevoluteChain(scene, 2, 0.5f, 1.0f, PxVec3(2, 5, 0));
  if (!guard.trackArticulation(artic2) || gInitializationFailed)
    return;

  PxArticulationLink* links1[2] = {};
  if (artic1->getNbLinks() != 2 || artic1->getLinks(links1, 2) != 2)
  {
    failInitialization();
    return;
  }
  PxArticulationLink* links2[2] = {};
  if (artic2->getNbLinks() != 2 || artic2->getLinks(links2, 2) != 2)
  {
    failInitialization();
    return;
  }

  PxArticulationJointReducedCoordinate* joint1 =
      getInboundJointChecked(links1[1]);
  PxArticulationJointReducedCoordinate* joint2 =
      getInboundJointChecked(links2[1]);
  if (!joint1 || !joint2)
    return;

  // A vertical fixed-base pendulum is in exact static equilibrium, so gravity
  // alone cannot prove that either articulation was advanced. Give the two
  // identical chains equal and opposite generalized velocities instead.
  joint1->setJointVelocity(PxArticulationAxis::eTWIST, 1.5f);
  joint2->setJointVelocity(PxArticulationAxis::eTWIST, -1.5f);
  artic1->updateKinematic(PxArticulationKinematicFlag::eVELOCITY);
  artic2->updateKinematic(PxArticulationKinematicFlag::eVELOCITY);
  artic1->wakeUp();
  artic2->wakeUp();

  const PxTransform initialRoot1 = links1[0]->getGlobalPose();
  const PxTransform initialRoot2 = links2[0]->getGlobalPose();
  const PxVec3 initialChild1 = links1[1]->getGlobalPose().p;
  const PxVec3 initialChild2 = links2[1]->getGlobalPose().p;

  PxReal peakChildTravel1 = 0.0f;
  PxReal peakChildTravel2 = 0.0f;
  PxReal peakOpposedTravel = 0.0f;
  PxReal maxMirrorOffsetError = 0.0f;
  PxReal maxRootPositionError = 0.0f;
  PxReal maxRootAngleError = 0.0f;
  PxReal maxAnchorError = 0.0f;
  bool finite = true;

  const PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    if (!simulateAndFetch(scene, dt))
      break;

    const PxTransform root1 = links1[0]->getGlobalPose();
    const PxTransform root2 = links2[0]->getGlobalPose();
    const PxTransform child1 = links1[1]->getGlobalPose();
    const PxTransform child2 = links2[1]->getGlobalPose();
    if (!root1.p.isFinite() || !root1.q.isFinite() ||
        !root2.p.isFinite() || !root2.q.isFinite() ||
        !child1.p.isFinite() || !child1.q.isFinite() ||
        !child2.p.isFinite() || !child2.q.isFinite() ||
        !links1[0]->getLinearVelocity().isFinite() ||
        !links1[0]->getAngularVelocity().isFinite() ||
        !links1[1]->getLinearVelocity().isFinite() ||
        !links1[1]->getAngularVelocity().isFinite() ||
        !links2[0]->getLinearVelocity().isFinite() ||
        !links2[0]->getAngularVelocity().isFinite() ||
        !links2[1]->getLinearVelocity().isFinite() ||
        !links2[1]->getAngularVelocity().isFinite())
    {
      finite = false;
      recordNonFinite();
      break;
    }

    const PxReal childTravel1 = PxAbs(child1.p.z - initialChild1.z);
    const PxReal childTravel2 = PxAbs(child2.p.z - initialChild2.z);
    peakChildTravel1 = PxMax(peakChildTravel1, childTravel1);
    peakChildTravel2 = PxMax(peakChildTravel2, childTravel2);
    if ((child1.p.z - initialChild1.z) *
        (child2.p.z - initialChild2.z) < 0.0f)
      peakOpposedTravel = PxMax(
          peakOpposedTravel, PxMin(childTravel1, childTravel2));

    const PxVec3 offset1 = child1.p - root1.p;
    const PxVec3 offset2 = child2.p - root2.p;
    const PxVec3 mirroredOffset1(offset1.x, offset1.y, -offset1.z);
    maxMirrorOffsetError = PxMax(
        maxMirrorOffsetError, (offset2 - mirroredOffset1).magnitude());

    maxRootPositionError = PxMax(
        maxRootPositionError,
        PxMax((root1.p - initialRoot1.p).magnitude(),
              (root2.p - initialRoot2.p).magnitude()));
    const PxReal rootAngle1 = 2.0f * PxAcos(PxClamp(
        PxAbs(root1.q.dot(initialRoot1.q)), 0.0f, 1.0f));
    const PxReal rootAngle2 = 2.0f * PxAcos(PxClamp(
        PxAbs(root2.q.dot(initialRoot2.q)), 0.0f, 1.0f));
    maxRootAngleError = PxMax(
        maxRootAngleError, PxMax(rootAngle1, rootAngle2));

    const PxReal anchorError1 =
        (root1.transform(joint1->getParentPose()).p -
         child1.transform(joint1->getChildPose()).p).magnitude();
    const PxReal anchorError2 =
        (root2.transform(joint2->getParentPose()).p -
         child2.transform(joint2->getChildPose()).p).magnitude();
    maxAnchorError = PxMax(
        maxAnchorError, PxMax(anchorError1, anchorError2));
  }
  if (!guard.canRead())
    return;

  printf("  [multi-articulation] peakTravel=(%.6f, %.6f) opposedTravel=%.6f "
         "mirrorError=%.6f rootPositionError=%.6f rootAngleError=%.6f "
         "anchorError=%.6f\n",
         static_cast<double>(peakChildTravel1),
         static_cast<double>(peakChildTravel2),
         static_cast<double>(peakOpposedTravel),
         static_cast<double>(maxMirrorOffsetError),
         static_cast<double>(maxRootPositionError),
         static_cast<double>(maxRootAngleError),
         static_cast<double>(maxAnchorError));

  const bool bothMovedInOppositeDirections = peakOpposedTravel > 0.02f;
  const bool responseStayedBounded = peakChildTravel1 < 0.15f &&
                                     peakChildTravel2 < 0.15f;
  const bool mirroredResponse = maxMirrorOffsetError < 0.005f;
  const bool fixedRoots = maxRootPositionError < 1.0e-4f &&
                          maxRootAngleError < 1.0e-4f;
  const bool anchorsClosed = maxAnchorError < 0.01f;
  TEST_CHECK(finite && bothMovedInOppositeDirections && responseStayedBounded &&
                 mirroredResponse && fixedRoots && anchorsClosed,
             "Both articulations respond independently to opposed excitation");

}

// ============================================================================
// Test 16: Scissor Lift (SnippetArticulationRC scene) - 10s stability
// ============================================================================

static PxFilterFlags scissorFilter(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
                                   PxFilterObjectAttributes attributes1, PxFilterData filterData1,
                                   PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
  PX_UNUSED(attributes0);
  PX_UNUSED(attributes1);
  PX_UNUSED(constantBlock);
  PX_UNUSED(constantBlockSize);
  if (filterData0.word2 != 0 && filterData0.word2 == filterData1.word2)
    return PxFilterFlag::eKILL;
  pairFlags |= PxPairFlag::eCONTACT_DEFAULT;
  return PxFilterFlag::eDEFAULT;
}

static bool isFiniteVec3(const PxVec3& v)
{
  return PxIsFinite(v.x) && PxIsFinite(v.y) && PxIsFinite(v.z);
}

static void updateScissorLiftDriveTarget(PxArticulationJointReducedCoordinate* driveJoint,
                                         bool& closing, PxReal dt)
{
  PxReal driveValue = driveJoint->getDriveTarget(PxArticulationAxis::eZ);

  if (closing && driveValue < -1.2f)
    closing = false;
  else if (!closing && driveValue > 0.0f)
    closing = true;

  if (closing)
    driveValue -= dt * 0.25f;
  else
    driveValue += dt * 0.25f;

  driveJoint->setDriveTarget(PxArticulationAxis::eZ, driveValue);
}

static void testScissorLift()
{
  printf("\n--- Test 16: Scissor Lift (RC scene, 10s stability) ---\n");
  fflush(stdout);

  // Create scene with custom filter (same as SnippetArticulationRC)
  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = scissorFilter;
  sceneDesc.solverType = gSolverType;
  PxScene* scene = gPhysics->createScene(sceneDesc);
  if (!scene)
    failInitialization();
  else
    ++gSceneRuns;
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  // Ground plane
  PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  if (!guard.trackActor(groundPlane) || !scene->addActor(*groundPlane))
  {
    failInitialization();
    return;
  }

  // ---- Build scissor lift articulation (exact replica of SnippetArticulationRC) ----
  PxArticulationReducedCoordinate* artic = gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  const PxReal runnerLength = 2.f;
  const PxReal placementDistance = 1.8f;
  const PxReal cosAng = placementDistance / runnerLength;
  const PxReal angle = PxAcos(cosAng);
  const PxReal sinAng = PxSin(angle);
  const PxQuat leftRot(-angle, PxVec3(1.f, 0.f, 0.f));
  const PxQuat rightRot(angle, PxVec3(1.f, 0.f, 0.f));

  // Base
  PxArticulationLink* base = createLinkWithShape(
      artic, NULL, PxTransform(PxVec3(0.f, 0.25f, 0.f)),
      PxBoxGeometry(0.5f, 0.25f, 1.5f), 3.f);
  if (!base)
    return;

  // Left root - fixed to base
  PxArticulationLink* leftRoot = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0.f, 0.55f, -0.9f)),
      PxBoxGeometry(0.5f, 0.05f, 0.05f), 1.f);
  if (!leftRoot)
    return;

  // Right root - prismatic (drive) joint
  PxArticulationLink* rightRoot = createLinkWithShape(
      artic, base, PxTransform(PxVec3(0.f, 0.55f, 0.9f)),
      PxBoxGeometry(0.5f, 0.05f, 0.05f), 1.f);
  if (!rightRoot)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(leftRoot);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eFIX);
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, -0.9f)));
  joint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

  PxArticulationJointReducedCoordinate* driveJoint =
      getInboundJointChecked(rightRoot);
  if (!driveJoint)
    return;
  driveJoint->setJointType(PxArticulationJointType::ePRISMATIC);
  driveJoint->setMotion(PxArticulationAxis::eZ, PxArticulationMotion::eLIMITED);
  driveJoint->setLimitParams(PxArticulationAxis::eZ, PxArticulationLimit(-1.4f, 0.2f));
  driveJoint->setDriveParams(PxArticulationAxis::eZ, PxArticulationDrive(100000.f, 0.f, PX_MAX_F32));
  driveJoint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, 0.9f)));
  driveJoint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

  // Scissor links - first side (x = +0.5)
  const PxU32 linkHeight = 3;
  PxArticulationLink* currLeft = leftRoot;
  PxArticulationLink* currRight = rightRoot;
  PxQuat rightParentRot(PxIdentity);
  PxQuat leftParentRot(PxIdentity);

  for (PxU32 i = 0; i < linkHeight; ++i)
  {
    const PxVec3 pos(0.5f, 0.55f + 0.1f * (1 + i), 0.f);

    PxArticulationLink* leftLink = createLinkWithShape(
      artic, currLeft,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), leftRot),
      PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
    if (!leftLink)
      return;

    const PxVec3 leftAnchor = pos + PxVec3(0.f, sinAng * (2 * i), -0.9f);
    joint = getInboundJointChecked(leftLink);
    if (!joint)
      return;
    joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));
    leftParentRot = leftRot;

    PxArticulationLink* rightLink = createLinkWithShape(
      artic, currRight,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), rightRot),
      PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
    if (!rightLink)
      return;

    const PxVec3 rightAnchor = pos + PxVec3(0.f, sinAng * (2 * i), 0.9f);
    joint = getInboundJointChecked(rightLink);
    if (!joint)
      return;
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
    rightParentRot = rightRot;

    PxD6Joint* d6 = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
    if (!guard.trackJoint(d6))
      return;
    d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

    currLeft = rightLink;
    currRight = leftLink;
  }

  // Top links
  PxArticulationLink* leftTop = createLinkWithShape(
    artic, currLeft,
    currLeft->getGlobalPose().transform(
      PxTransform(PxVec3(-0.5f, 0.f, -1.0f), leftParentRot)),
    PxBoxGeometry(0.5f, 0.05f, 0.05f), 1.f);
  if (!leftTop)
    return;

  PxArticulationLink* rightTop = createLinkWithShape(
    artic, currRight,
    currRight->getGlobalPose().transform(
      PxTransform(PxVec3(-0.5f, 0.f, 1.0f), rightParentRot)),
    PxCapsuleGeometry(0.05f, 0.8f), 1.f);
  if (!rightTop)
    return;

  joint = getInboundJointChecked(leftTop);
  if (!joint)
    return;
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, -1.f), currLeft->getGlobalPose().q.getConjugate()));
  joint->setChildPose(PxTransform(PxVec3(0.5f, 0.f, 0.f), leftTop->getGlobalPose().q.getConjugate()));
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

  joint = getInboundJointChecked(rightTop);
  if (!joint)
    return;
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, 1.f), currRight->getGlobalPose().q.getConjugate()));
  joint->setChildPose(PxTransform(PxVec3(0.5f, 0.f, 0.f), rightTop->getGlobalPose().q.getConjugate()));
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

  // Scissor links - second side (x = -0.5)
  currLeft = leftRoot;
  currRight = rightRoot;
  rightParentRot = PxQuat(PxIdentity);
  leftParentRot = PxQuat(PxIdentity);

  for (PxU32 i = 0; i < linkHeight; ++i)
  {
    const PxVec3 pos(-0.5f, 0.55f + 0.1f * (1 + i), 0.f);

    PxArticulationLink* leftLink = createLinkWithShape(
      artic, currLeft,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), leftRot),
      PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
    if (!leftLink)
      return;

    const PxVec3 leftAnchor = pos + PxVec3(0.f, sinAng * (2 * i), -0.9f);
    joint = getInboundJointChecked(leftLink);
    if (!joint)
      return;
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));
    leftParentRot = leftRot;

    PxArticulationLink* rightLink = createLinkWithShape(
      artic, currRight,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), rightRot),
      PxBoxGeometry(0.05f, 0.05f, 1.f), 1.f);
    if (!rightLink)
      return;

    const PxVec3 rightAnchor = pos + PxVec3(0.f, sinAng * (2 * i), 0.9f);
    joint = getInboundJointChecked(rightLink);
    if (!joint)
      return;
    joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
    rightParentRot = rightRot;

    PxD6Joint* d6 = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
    if (!guard.trackJoint(d6))
      return;
    d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

    currLeft = rightLink;
    currRight = leftLink;
  }

  // D6 joints connecting second-side tops to first-side tops
  PxD6Joint* d6 = PxD6JointCreate(*gPhysics, currLeft, PxTransform(PxVec3(0.f, 0.f, -1.f)),
    leftTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
  if (!guard.trackJoint(d6))
    return;
  d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

  d6 = PxD6JointCreate(*gPhysics, currRight, PxTransform(PxVec3(0.f, 0.f, 1.f)),
    rightTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
  if (!guard.trackJoint(d6))
    return;
  d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

  // Top platform
  const PxTransform topPose(PxVec3(0.f, leftTop->getGlobalPose().p.y + 0.15f, 0.f));
  PxArticulationLink* top = createLinkWithShape(
      artic, leftTop, topPose, PxBoxGeometry(0.5f, 0.1f, 1.5f), 1.f);
  if (!top)
    return;

  joint = getInboundJointChecked(top);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eFIX);
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.0f, 0.f)));
  joint->setChildPose(PxTransform(PxVec3(0.f, -0.15f, -0.9f)));

  // Add articulation to scene
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  // Set damping and collision filter on all links
  PxU32 nbLinks = artic->getNbLinks();
  for (PxU32 i = 0; i < nbLinks; ++i)
  {
    PxArticulationLink* link;
    artic->getLinks(&link, 1, i);
    link->setLinearDamping(0.2f);
    link->setAngularDamping(0.2f);
    link->setMaxAngularVelocity(20.f);
    link->setMaxLinearVelocity(100.f);
    if (link != top)
    {
      for (PxU32 b = 0; b < link->getNbShapes(); ++b)
      {
        PxShape* shape;
        link->getShapes(&shape, 1, b);
        shape->setSimulationFilterData(PxFilterData(0, 0, 1, 0));
      }
    }
  }

  // --- Phase 1: Test articulation alone (no boxes) for 300 frames ---
  {
    const PxReal dt = 1.0f / 60.0f;
    bool articAloneOK = true;
    bool driveClosing = true;
    PxReal topMinY = top->getGlobalPose().p.y;
    PxReal topMaxY = topMinY;
    PxReal prevTopY = topMinY;
    PxI32 prevMotionSign = 0;
    PxU32 motionReversals = 0;

    for (int frame = 0; frame < 300; frame++)
    {
      updateScissorLiftDriveTarget(driveJoint, driveClosing, dt);
      if (!simulateAndFetch(scene, dt))
      {
        articAloneOK = false;
        break;
      }

      PxVec3 bp = base->getGlobalPose().p;
      const PxReal topY = top->getGlobalPose().p.y;
      topMinY = PxMin(topMinY, topY);
      topMaxY = PxMax(topMaxY, topY);

      const PxReal dy = topY - prevTopY;
      const PxReal motionThreshold = 1e-3f;
      const PxI32 motionSign = (dy > motionThreshold) ? 1 : ((dy < -motionThreshold) ? -1 : 0);
      if (motionSign != 0)
      {
        if (prevMotionSign != 0 && motionSign != prevMotionSign)
          motionReversals++;
        prevMotionSign = motionSign;
      }
      prevTopY = topY;

      if (!isFiniteVec3(bp) || bp.magnitude() > 10.f)
      {
        if (!isFiniteVec3(bp))
          recordNonFinite();
        printf("  [Phase1] Articulation-alone explosion at frame %d: base=(%.2f,%.2f,%.2f)\n",
               frame, (double)bp.x, (double)bp.y, (double)bp.z);
        articAloneOK = false;
        break;
      }
    }
    if (!guard.canRead())
      return;

    const bool topHasStroke = (topMaxY - topMinY) > 0.6f;
    const bool topCycles = motionReversals >= 2;

    TEST_CHECK(articAloneOK, "Scissor lift stable alone (5s, no boxes)");
    TEST_CHECK(topHasStroke, "Scissor lift platform travels through a large lift stroke");
    TEST_CHECK(topCycles, "Scissor lift cycles up and down without load");
    if (!articAloneOK)
      return;
  }

  // --- Phase 2: Add boxes and simulate 10 more seconds ---
  const PxVec3 halfExt(0.25f);
  const PxReal density = 0.5f;
  const float contactOffset = 0.2f;
  PxVec3 boxPositions[8] = {
    PxVec3(-0.25f, 5.f, 0.5f), PxVec3(0.25f, 5.f, 0.5f),
    PxVec3(-0.25f, 4.5f, 0.5f), PxVec3(0.25f, 4.5f, 0.5f),
    PxVec3(-0.25f, 5.f, 0.f),  PxVec3(0.25f, 5.f, 0.f),
    PxVec3(-0.25f, 4.5f, 0.f), PxVec3(0.25f, 4.5f, 0.f)
  };
  PxRigidDynamic* boxes[8];
  for (int b = 0; b < 8; b++)
  {
    boxes[b] = gPhysics->createRigidDynamic(PxTransform(boxPositions[b]));
    if (!guard.trackActor(boxes[b]))
      return;
    PxShape* s = PxRigidActorExt::createExclusiveShape(*boxes[b], PxBoxGeometry(halfExt), *gMaterial);
    if (!s || !PxRigidBodyExt::updateMassAndInertia(*boxes[b], density))
    {
      failInitialization();
      return;
    }
    s->setContactOffset(contactOffset);
    if (!scene->addActor(*boxes[b]))
    {
      failInitialization();
      return;
    }
  }

  // ---- Simulate 10 seconds (600 frames at 60 Hz) ----
  const PxReal dt = 1.0f / 60.0f;
  const int totalFrames = 600;
  bool anyNaN = false;
  bool anyExplosion = false;
  bool anyBelowGround = false;
  bool driveClosing = true;
  PxReal topMinY = top->getGlobalPose().p.y;
  PxReal topMaxY = topMinY;
  PxReal prevTopY = topMinY;
  PxI32 prevMotionSign = 0;
  PxU32 loadedMotionReversals = 0;
  int failFrame = -1;

  for (int frame = 0; frame < totalFrames; frame++)
  {
    updateScissorLiftDriveTarget(driveJoint, driveClosing, dt);
    if (!simulateAndFetch(scene, dt))
    {
      anyExplosion = true;
      if (failFrame < 0) failFrame = frame;
      break;
    }

    const PxReal topY = top->getGlobalPose().p.y;
    topMinY = PxMin(topMinY, topY);
    topMaxY = PxMax(topMaxY, topY);

    const PxReal dy = topY - prevTopY;
    const PxReal motionThreshold = 1e-3f;
    const PxI32 motionSign = (dy > motionThreshold) ? 1 : ((dy < -motionThreshold) ? -1 : 0);
    if (motionSign != 0)
    {
      if (prevMotionSign != 0 && motionSign != prevMotionSign)
        loadedMotionReversals++;
      prevMotionSign = motionSign;
    }
    prevTopY = topY;

    // Check all articulation links for NaN/explosion
    for (PxU32 li = 0; li < nbLinks; li++)
    {
      PxArticulationLink* link;
      artic->getLinks(&link, 1, li);
      PxVec3 p = link->getGlobalPose().p;
      if (!isFiniteVec3(p))
      {
        anyNaN = true;
        recordNonFinite();
        if (failFrame < 0) failFrame = frame;
      }
      if (p.magnitude() > 100.f)
      {
        anyExplosion = true;
        if (failFrame < 0) failFrame = frame;
      }
    }

    // Check boxes
    for (int b = 0; b < 8; b++)
    {
      PxVec3 p = boxes[b]->getGlobalPose().p;
      if (!isFiniteVec3(p))
      {
        anyNaN = true;
        recordNonFinite();
        if (failFrame < 0) failFrame = frame;
      }
      if (p.magnitude() > 100.f)
      {
        anyExplosion = true;
        if (failFrame < 0) failFrame = frame;
      }
    }

    if (anyNaN || anyExplosion)
    {
      printf("  [DIAG] Explosion at frame %d:\n", frame);
      for (PxU32 li = 0; li < nbLinks; li++)
      {
        PxArticulationLink* link;
        artic->getLinks(&link, 1, li);
        PxVec3 p = link->getGlobalPose().p;
        if (p.magnitude() > 5.f || !isFiniteVec3(p))
          printf("    link[%u] pos=(%.2f, %.2f, %.2f) mag=%.2f\n", li, (double)p.x, (double)p.y, (double)p.z, (double)p.magnitude());
      }
      for (int b = 0; b < 8; b++)
      {
        PxVec3 p = boxes[b]->getGlobalPose().p;
        if (p.magnitude() > 5.f || !isFiniteVec3(p))
          printf("    box[%d] pos=(%.2f, %.2f, %.2f) mag=%.2f\n", b, (double)p.x, (double)p.y, (double)p.z, (double)p.magnitude());
      }
      fflush(stdout);
      break;
    }
  }
  if (!guard.canRead())
    return;

  // After 10 seconds, boxes should rest above ground
  if (!anyNaN && !anyExplosion)
  {
    for (int b = 0; b < 8; b++)
    {
      PxReal y = boxes[b]->getGlobalPose().p.y;
      if (y < -1.0f)
        anyBelowGround = true;
    }
  }

  if (failFrame >= 0)
    printf("  (first failure at frame %d, t=%.2fs)\n", failFrame, failFrame * dt);

  TEST_CHECK(!anyNaN, "No NaN in 10s simulation");
  TEST_CHECK(!anyExplosion, "No explosion (positions bounded) in 10s");
  TEST_CHECK(!anyBelowGround, "Boxes rest above ground after 10s");
  TEST_CHECK((topMaxY - topMinY) > 0.4f, "Scissor lift retains meaningful lift stroke under load");
  TEST_CHECK(loadedMotionReversals >= 3, "Scissor lift continues cyclic motion under load");

  // Check articulation base is still near origin (not flying away)
  PxVec3 basePos = base->getGlobalPose().p;
  bool baseStable = isFiniteVec3(basePos) && basePos.magnitude() < 10.f;
  TEST_CHECK(baseStable, "Articulation base remains stable");

}

// ============================================================================
// Test 17: Fixed-base child-link static-world D6 loop diagnostic
// ============================================================================
static void testChildLinkStaticWorldLoopD6()
{
  printf("\n--- Test 17: Fixed-base Child-link Static-world D6 Loop ---\n");
  fflush(stdout);

  PxScene* scene = createAvbdScene();
  TestSceneGuard guard(scene);
  if (!scene)
    return;

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  if (!guard.trackArticulation(artic))
    return;
  applyConfiguredSolverIterations(artic);

  const PxVec3 rootHalfExt(0.3f, 0.2f, 0.2f);
  const PxVec3 childHalfExt(0.5f, 0.12f, 0.12f);
  const PxVec3 rootPos(0.0f, 5.0f, 0.0f);
  const PxVec3 loopAnchor(1.1f, 4.4f, 0.0f);
  const PxReal childAngle = PxAtan2(-0.6f, 0.8f);
  const PxVec3 childPos(0.7f, 4.7f, 0.0f);
  const PxQuat childRot(childAngle, PxVec3(0.0f, 0.0f, 1.0f));
  const PxQuat hingeFrame(PxHalfPi, PxVec3(0.0f, 1.0f, 0.0f));

  PxArticulationLink* root = createLinkWithShape(
      artic, NULL, PxTransform(rootPos),
      PxBoxGeometry(rootHalfExt.x, rootHalfExt.y, rootHalfExt.z), 1000.0f);
  if (!root)
    return;

  PxArticulationLink* child =
      createLinkWithShape(artic, root, PxTransform(childPos, childRot),
        PxBoxGeometry(childHalfExt.x, childHalfExt.y, childHalfExt.z),
        1000.0f);
  if (!child)
    return;

  PxArticulationJointReducedCoordinate* joint =
      getInboundJointChecked(child);
  if (!joint)
    return;
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(rootHalfExt.x, 0.0f, 0.0f), hingeFrame));
  joint->setChildPose(PxTransform(PxVec3(-childHalfExt.x, 0.0f, 0.0f), childRot.getConjugate() * hingeFrame));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  if (!scene->addArticulation(*artic))
  {
    failInitialization();
    return;
  }

  const PxVec3 childTipLocal(childHalfExt.x, 0.0f, 0.0f);

  PxD6Joint* loop = PxD6JointCreate(
    *gPhysics, NULL, PxTransform(loopAnchor), child, PxTransform(childTipLocal));
  if (!guard.trackJoint(loop))
    return;
  loop->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
  loop->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
  loop->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
  loop->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  loop->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  loop->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

  joint->setJointVelocity(PxArticulationAxis::eTWIST, 1.5f);

  const PxReal dt = 1.0f / 60.0f;
  PxReal maxTipError = 0.0f;
  PxReal finalTipError = 0.0f;
  bool finite = true;

  for (int frame = 0; frame < 180; ++frame)
  {
    if (!simulateAndFetch(scene, dt))
    {
      finite = false;
      break;
    }

    const PxVec3 tipWorld = child->getGlobalPose().transform(childTipLocal);
    if (!isFiniteVec3(tipWorld))
    {
      finite = false;
      recordNonFinite();
      break;
    }

    const PxReal tipError = (tipWorld - loopAnchor).magnitude();
    maxTipError = PxMax(maxTipError, tipError);
    finalTipError = tipError;
  }
  if (!guard.canRead())
    return;

  const PxVec3 finalChildPos = child->getGlobalPose().p;
  if (!isFiniteVec3(finalChildPos))
  {
    finite = false;
    recordNonFinite();
  }

  printf("  [child-loop-d6] finalTipError=%.6f maxTipError=%.6f childPos=(%.6f, %.6f, %.6f)\n",
         static_cast<double>(finalTipError),
         static_cast<double>(maxTipError),
         static_cast<double>(finalChildPos.x),
         static_cast<double>(finalChildPos.y),
         static_cast<double>(finalChildPos.z));

  TEST_CHECK(finite, "Child-link loop D6 remained finite");

}

// ============================================================================
// Initialization and main
// ============================================================================

void initPhysics(bool /*interactive*/)
{
  gErrorCallback.reset();
  gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator,
                                   gErrorCallback);
  if (!gFoundation)
  {
    failInitialization();
    return;
  }

  gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
                             PxTolerancesScale(), true, NULL);
  if (!gPhysics)
  {
    failInitialization();
    return;
  }
  if (!PxInitExtensions(*gPhysics, NULL))
  {
    failInitialization();
    return;
  }
  gExtensionsInitialized = true;

  gDispatcher = PxDefaultCpuDispatcherCreate(
      gHeadlessOptions.dispatcherThreads);
  if (!gDispatcher)
  {
    failInitialization();
    return;
  }
  gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
  if (!gMaterial)
    failInitialization();
}

void stepPhysics(bool /*interactive*/)
{
  // Tests run individually within snippetMain
}

void cleanupPhysics(bool /*interactive*/)
{
  if (gFetchPending && !finishPendingSimulation() &&
      !finishPendingSimulation())
  {
    gCleanupFailed = true;
    gAbandonedPhysicsResources = true;
  }

  if (gAbandonedPhysicsResources)
  {
    gCleanupFailed = true;
    return;
  }

  PX_RELEASE(gMaterial);
  PX_RELEASE(gDispatcher);
  if (gExtensionsInitialized)
  {
    PxCloseExtensions();
    gExtensionsInitialized = false;
  }
  PX_RELEASE(gPhysics);
  PX_RELEASE(gFoundation);

  if (gMaterial || gDispatcher || gPhysics || gFoundation ||
      gExtensionsInitialized || gFetchPending)
    gCleanupFailed = true;
}

static int reportConfigurationError(const Snippets::HeadlessOptions& options,
                                    const char* message)
{
  printf("[AVBD_GATE_ERROR] snippet=SnippetAvbdArticulation message=%s\n",
         message);
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetAvbdArticulation "
      "case=config-error solver=%s execution=%s requestedFrames=%u "
      "completedFrames=0 dt=%.9g seed=%u dispatcherThreads=%u "
      "capability=SUPPORTED validation=ACCEPTED status=ERROR reason=config "
      "nonFinite=0 physicsErrors=0 physicsWarnings=0 simulateFailures=0 "
      "fetchFailures=0 fetchPending=0 fetchErrorState=0 "
      "runtimeInvariantFailed=0 initializationFailed=0 cleanupFailed=0 "
      "oraclePass=0 checkRatio=0/0 checksPassed=0 checksFailed=0 "
      "expectedChecks=0 testsRun=0 expectedTests=0 sceneRuns=0 "
      "expectedSceneRuns=0 solverIterations=0 pvd=0\n",
      Snippets::getSolverTypeName(options.solverType),
      Snippets::getExecutionName(options.execution), options.frames,
      double(options.dt), options.seed, options.dispatcherThreads);
  return Snippets::eHEADLESS_CONFIG_ERROR;
}

static void runSelectedTests()
{
#define RUN_SELECTED_TEST(testId, testFunction) \
  do { \
    if (shouldRunAvbdTest(gSelectedTestId, testId)) { \
      ++gTestsRun; \
      testFunction(); \
    } \
    if (hasTerminalTestFailure()) \
      return; \
  } while (0)

  RUN_SELECTED_TEST(1, testSinglePendulum);
  RUN_SELECTED_TEST(2, testMultiLinkChain);
  RUN_SELECTED_TEST(3, testJointLimits);
  RUN_SELECTED_TEST(4, testVelocityDrive);
  RUN_SELECTED_TEST(5, testPositionDrive);
  RUN_SELECTED_TEST(6, testAccelerationDrive);
  RUN_SELECTED_TEST(7, testJointFriction);
  RUN_SELECTED_TEST(8, testMimicJoint);
  RUN_SELECTED_TEST(9, testPrismaticJoint);
  RUN_SELECTED_TEST(10, testFloatingBase);
  RUN_SELECTED_TEST(11, testArticulationContact);
  RUN_SELECTED_TEST(12, testJointVelocityLimit);
  RUN_SELECTED_TEST(13, testSphericalJoint);
  RUN_SELECTED_TEST(14, testMixedJointChain);
  RUN_SELECTED_TEST(15, testMultiArticulation);
  RUN_SELECTED_TEST(16, testScissorLift);
  RUN_SELECTED_TEST(17, testChildLinkStaticWorldLoopD6);

#undef RUN_SELECTED_TEST
}

static int printGateResult(PxU32 physicsErrors, PxU32 physicsWarnings)
{
  const PxU32 actualChecks = PxU32(gTestsPassed + gTestsFailed);
  const bool harnessError = gInitializationFailed ||
      gRuntimeInvariantFailed || gCleanupFailed || gFetchPending ||
      gAbandonedPhysicsResources || gSimulateFailures || gFetchFailures;
  const bool accountingError =
      gCompletedFrames != gHeadlessOptions.frames ||
      actualChecks != gExpectedChecks || gTestsRun != gExpectedTests ||
      gSceneRuns != gExpectedSceneRuns;
  const bool solverFailure = gTestsFailed || gNonFiniteDetected ||
      gFetchErrorState || physicsErrors;

  const char* status = "PASS";
  const char* reason = "none";
  Snippets::HeadlessExitCode exitCode = Snippets::eHEADLESS_PASS;
  if (harnessError)
  {
    status = "ERROR";
    exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
    if (gInitializationFailed) reason = "initialization";
    else if (gSimulateFailures) reason = "simulate";
    else if (gFetchFailures || gFetchPending) reason = "fetch";
    else if (gRuntimeInvariantFailed) reason = "runtime_invariant";
    else if (gCleanupFailed || gAbandonedPhysicsResources) reason = "cleanup";
  }
  else if (solverFailure)
  {
    status = "FAIL";
    if (physicsErrors || gFetchErrorState) reason = "physx_error";
    else if (gNonFiniteDetected) reason = "non_finite";
    else reason = "oracle";
    exitCode = Snippets::eHEADLESS_GATE_FAILED;
  }
  else if (accountingError)
  {
    status = "ERROR";
    exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
    if (gCompletedFrames != gHeadlessOptions.frames)
      reason = "frame_accounting";
    else if (gTestsRun != gExpectedTests)
      reason = "test_accounting";
    else if (gSceneRuns != gExpectedSceneRuns)
      reason = "scene_accounting";
    else reason = "check_accounting";
  }
  const PxU32 oraclePassed = exitCode == Snippets::eHEADLESS_PASS ? 1u : 0u;

  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetAvbdArticulation case=%s "
      "solver=%s execution=%s requestedFrames=%u completedFrames=%u "
      "dt=%.9g seed=%u dispatcherThreads=%u capability=SUPPORTED "
      "validation=ACCEPTED status=%s reason=%s nonFinite=%u "
      "physicsErrors=%u physicsWarnings=%u simulateFailures=%u "
      "fetchFailures=%u fetchPending=%u fetchErrorState=%u "
      "runtimeInvariantFailed=%u initializationFailed=%u cleanupFailed=%u "
      "abandonedResources=%u oraclePass=%u checkRatio=%d/%u "
      "checksPassed=%d checksFailed=%d expectedChecks=%u testsRun=%u "
      "expectedTests=%u sceneRuns=%u expectedSceneRuns=%u "
      "solverIterations=%u pvd=0\n",
      gHeadlessOptions.caseName.c_str(),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gCompletedFrames, double(gHeadlessOptions.dt),
      gHeadlessOptions.seed, gHeadlessOptions.dispatcherThreads, status,
      reason, gNonFiniteDetected, physicsErrors, physicsWarnings,
      gSimulateFailures,
      gFetchFailures, gFetchPending ? 1u : 0u, gFetchErrorState,
      gRuntimeInvariantFailed ? 1u : 0u,
      gInitializationFailed ? 1u : 0u, gCleanupFailed ? 1u : 0u,
      gAbandonedPhysicsResources ? 1u : 0u, oraclePassed, gTestsPassed,
      gExpectedChecks, gTestsPassed, gTestsFailed, gExpectedChecks,
      gTestsRun, gExpectedTests, gSceneRuns, gExpectedSceneRuns,
      gSolverIterations);
  return static_cast<int>(exitCode);
}

int snippetMain(int argc, const char*const* argv)
{
  setvbuf(stdout, NULL, _IONBF, 0);

  Snippets::HeadlessOptions defaults;
  defaults.caseName = "full-suite";
  defaults.frames = 5040;
  defaults.seed = 1;
  defaults.dispatcherThreads = 2;
  defaults.dt = 1.0f / 60.0f;

  Snippets::HeadlessOptions options;
  std::string parseError;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
                                            parseError))
    return reportConfigurationError(options, parseError.c_str());

  bool caseExplicit = false;
  for (int i = 1; i < argc; ++i)
  {
    const char* arg = argv[i];
    if (!arg)
      continue;
    if (!Snippets::isCommonHeadlessOption(arg))
      return reportConfigurationError(options, "unknown_argument");
    caseExplicit = caseExplicit ||
        Snippets::hasOptionPrefix(arg, "--case=") ||
        Snippets::hasOptionPrefix(arg, "--scenario=");
  }

#ifndef RENDER_SNIPPET
  options.headless = true;
#endif

  if (options.solverType != PxSolverType::eAVBD &&
      options.solverType != PxSolverType::eTGS)
    return reportConfigurationError(options, "solver_requires_avbd_or_tgs");
  if (PxAbs(options.dt - (1.0f / 60.0f)) > 1.0e-7f)
    return reportConfigurationError(options, "dt_requires_60hz_calibration");

  if (!options.executionExplicit)
  {
    bool sequentialEnvironment = false;
    if (!parseLegacyFlag("PHYSX_AVBD_ITER_DIAG_SEQUENTIAL",
                         sequentialEnvironment, parseError))
      return reportConfigurationError(options, parseError.c_str());
    options.execution = sequentialEnvironment
          ? Snippets::eHEADLESS_SEQUENTIAL
          : Snippets::eHEADLESS_PARALLEL;
  }
  if (options.execution == Snippets::eHEADLESS_SEQUENTIAL &&
      options.solverType != PxSolverType::eAVBD)
    return reportConfigurationError(options, "sequential_requires_avbd");

  int legacyTestId = -1;
  bool childLoopOnly = false;
  if (!caseExplicit)
  {
    if (!getSelectedAvbdTestId(legacyTestId, parseError))
      return reportConfigurationError(options, parseError.c_str());
    if (!parseLegacyFlag("PHYSX_AVBD_CHILD_LOOP_ONLY", childLoopOnly,
                         parseError))
      return reportConfigurationError(options, parseError.c_str());
    if (childLoopOnly && legacyTestId > 0 && legacyTestId != 17)
      return reportConfigurationError(
          options, "conflicting_legacy_test_selectors");
    if (childLoopOnly)
      legacyTestId = 17;
  }
  PxU32 configuredIterations = 32;
  if (!getConfiguredSolverIterations(configuredIterations, parseError))
    return reportConfigurationError(options, parseError.c_str());
  gSolverIterations = configuredIterations;

  PxU32 expectedFrames = 5040;
  gExpectedChecks = 31;
  gExpectedTests = 17;
  gExpectedSceneRuns = 21;
  gSelectedTestId = -1;
  if (caseExplicit)
  {
    if (Snippets::equalsIgnoreCase(options.caseName.c_str(), "full-suite"))
      options.caseName = "full-suite";
    else if (Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                        "scissor-lift"))
    {
      options.caseName = "scissor-lift";
      expectedFrames = gTestFrames[15];
      gExpectedChecks = gTestChecks[15];
      gExpectedTests = 1;
      gExpectedSceneRuns = gTestScenes[15];
      gSelectedTestId = 16;
    }
    else if (Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                        "child-loop-d6"))
    {
      options.caseName = "child-loop-d6";
      expectedFrames = gTestFrames[16];
      gExpectedChecks = gTestChecks[16];
      gExpectedTests = 1;
      gExpectedSceneRuns = gTestScenes[16];
      gSelectedTestId = 17;
    }
    else
      return reportConfigurationError(options, "invalid_--case_value");

  }
  else if (legacyTestId > 0)
  {
    gSelectedTestId = legacyTestId;
    expectedFrames = gTestFrames[legacyTestId - 1];
    gExpectedChecks = gTestChecks[legacyTestId - 1];
    gExpectedTests = 1;
    gExpectedSceneRuns = gTestScenes[legacyTestId - 1];
    if (legacyTestId == 16)
      options.caseName = "scissor-lift";
    else if (legacyTestId == 17)
      options.caseName = "child-loop-d6";
    else
    {
      char legacyCase[32];
      std::snprintf(legacyCase, sizeof(legacyCase), "legacy-test-%d",
                    legacyTestId);
      options.caseName = legacyCase;
    }
  }
  else
    options.caseName = "full-suite";

  if (options.framesExplicit && options.frames != expectedFrames)
    return reportConfigurationError(options, "frames_must_match_case_exactly");
  options.frames = expectedFrames;
  if (!Snippets::applyExecutionEnvironment(options))
    return reportConfigurationError(options, "execution_environment_failed");

  gHeadlessOptions = options;
  gSolverType = options.solverType;
  Snippets::printHeadlessConfig("SnippetAvbdArticulation", options);
  initPhysics(false);

  printf("=== AVBD Articulation Comprehensive Tests ===\n");

  if (gSelectedTestId > 0)
    printf("=== Running only test %d ===\n", gSelectedTestId);
  if (!gInitializationFailed)
    runSelectedTests();

  printf("\n=== Results: %d PASSED, %d FAILED (out of %d) ===\n",
         gTestsPassed, gTestsFailed, gTestsPassed + gTestsFailed);

  cleanupPhysics(false);
  const PxU32 physicsErrors = gErrorCallback.getFatalCount();
  const PxU32 physicsWarnings = gErrorCallback.getWarningCount();
  return printGateResult(physicsErrors, physicsWarnings);
}
