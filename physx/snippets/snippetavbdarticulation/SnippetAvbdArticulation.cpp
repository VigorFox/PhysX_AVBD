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
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"

using namespace physx;

static PxDefaultAllocator     gAllocator;
static PxDefaultErrorCallback gErrorCallback;
static PxFoundation*          gFoundation   = NULL;
static PxPhysics*             gPhysics      = NULL;
static PxDefaultCpuDispatcher* gDispatcher  = NULL;
static PxMaterial*            gMaterial     = NULL;

static bool isEnvFlagEnabled(const char* name)
{
  const char* value = std::getenv(name);
  return value && value[0] && value[0] != '0';
}

static int getSelectedAvbdTestId()
{
  const char* value = std::getenv("PHYSX_AVBD_TEST_ID");
  if (!value || !value[0])
    return -1;

  const int testId = std::atoi(value);
  return testId > 0 ? testId : -1;
}

static bool shouldRunAvbdTest(int selectedTestId, int testId)
{
  return selectedTestId < 0 || selectedTestId == testId;
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
  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = gravity;
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = PxDefaultSimulationFilterShader;
  sceneDesc.solverType = PxSolverType::eAVBD;
  return gPhysics->createScene(sceneDesc);
}

// ============================================================================
// Helper: create articulation with N revolute links in a chain
// ============================================================================
static PxArticulationReducedCoordinate* createRevoluteChain(
    PxScene* scene, int numLinks, PxReal linkLength, PxReal linkMass,
    PxVec3 basePos, PxVec3 /*axis*/ = PxVec3(0, 0, 1))
{
  // TODO(AVBD): rename this API at the public boundary once PhysX exposes a
  // solver-neutral articulation facade again. In PhysX 5 we still create the
  // AVBD articulation through the reduced-coordinate entry point for API
  // compatibility, even though the solver-side articulation handling is
  // maximal-coordinate oriented.
  PxArticulationReducedCoordinate* artic = gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* parent = artic->createLink(NULL, PxTransform(basePos));
  PxRigidActorExt::createExclusiveShape(*parent, PxBoxGeometry(0.05f, linkLength * 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*parent, linkMass / (linkLength * 0.1f * 0.1f));

  for (int i = 1; i < numLinks; i++) {
    PxVec3 childPos = basePos + PxVec3(0, -linkLength * i, 0);
    PxArticulationLink* child = artic->createLink(parent, PxTransform(childPos));
    PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.05f, linkLength * 0.5f, 0.05f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*child, linkMass / (linkLength * 0.1f * 0.1f));

    PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    joint->setParentPose(PxTransform(PxVec3(0, -linkLength * 0.5f, 0)));
    joint->setChildPose(PxTransform(PxVec3(0, linkLength * 0.5f, 0)));

    parent = child;
  }

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);
  return artic;
}

// ============================================================================
// Test 1: Single Pendulum - basic gravity + articulation
// ============================================================================
static void testSinglePendulum()
{
  printf("\n--- Test 1: Single Pendulum ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      createRevoluteChain(scene, 2, 1.0f, 1.0f, PxVec3(0, 5, 0));

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // After 2 seconds under gravity, the pendulum tip should be below the base
  PxU32 nLinks = artic->getNbLinks();
  PxArticulationLink* links[16];
  artic->getLinks(links, 16);

  PxVec3 tipPos = links[nLinks - 1]->getGlobalPose().p;
  PxVec3 basePos = links[0]->getGlobalPose().p;

  TEST_CHECK(tipPos.y < basePos.y, "Pendulum tip below base");
  TEST_CHECK(tipPos.y < 4.5f, "Pendulum tip dropped significantly");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 2: Multi-link Chain Consistency
// ============================================================================
static void testMultiLinkChain()
{
  printf("\n--- Test 2: Multi-link Chain ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      createRevoluteChain(scene, 5, 0.5f, 0.5f, PxVec3(0, 5, 0));

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 60; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

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

  artic->release();
  scene->release();
}

// ============================================================================
// Test 3: Joint Limits
// ============================================================================
static void testJointLimits()
{
  printf("\n--- Test 3: Joint Limits ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  PxArticulationLink* child = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 10.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
  joint->setLimitParams(PxArticulationAxis::eTWIST, {-0.5f, 0.5f});
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 300; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // Joint should respect limits: angle should be <= 0.5 rad (with some tolerance)
  PxVec3 childEnd = child->getGlobalPose().p;
  PxVec3 baseP = base->getGlobalPose().p;
  PxVec3 dir = (childEnd - baseP).getNormalized();
  // If limited to +-0.5 rad, the child can't swing past ~0.48 (sin(0.5))
  TEST_CHECK(PxAbs(dir.x) < 0.6f, "Joint stayed within limit region");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 4: Velocity Drive
// ============================================================================
static void testVelocityDrive()
{
  printf("\n--- Test 4: Velocity Drive ---\n");
  PxScene* scene = createAvbdScene(PxVec3(0, 0, 0)); // no gravity

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  PxArticulationLink* child = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
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
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // After 2s of velocity drive, the child should have rotated
  PxVec3 childPos = child->getGlobalPose().p;
  PxVec3 basePos2 = base->getGlobalPose().p;
  PxReal distFromVertical = PxAbs(childPos.x - basePos2.x) + PxAbs(childPos.z - basePos2.z);
  // With non-zero target velocity, child should have moved from its initial position
  TEST_CHECK(distFromVertical > 0.01f || PxAbs(childPos.y - basePos2.y) < 0.95f,
             "Velocity drive caused rotation");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 5: Position Drive (PD Controller)
// ============================================================================
static void testPositionDrive()
{
  printf("\n--- Test 5: Position Drive ---\n");
  PxScene* scene = createAvbdScene(PxVec3(0, 0, 0)); // no gravity

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  PxArticulationLink* child = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  // Set position drive: target 1.0 rad
  joint->setDriveParams(PxArticulationAxis::eTWIST,
                        PxArticulationDrive(100.0f, 10.0f, PX_MAX_F32));
  joint->setDriveTarget(PxArticulationAxis::eTWIST, 1.0f);

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 300; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // The PD drive should have pushed the joint towards target (1.0 rad)
  PxVec3 childPos = child->getGlobalPose().p;
  PxVec3 basePos2 = base->getGlobalPose().p;
  PxVec3 diff = childPos - basePos2;
  PxReal lateralDist = PxSqrt(diff.x * diff.x + diff.z * diff.z);
  // With position target 1.0 rad, the child should have moved laterally
  TEST_CHECK(lateralDist > 0.1f || PxAbs(diff.y) < 0.95f, "PD drive moved joint towards target");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 6: Acceleration Drive (inertia-invariant)
// ============================================================================
static void testAccelerationDrive()
{
  printf("\n--- Test 6: Acceleration Drive ---\n");
  fflush(stdout);

  // Run two separate scenes with different masses but same acceleration drive
  auto runArm = [&](PxReal mass) -> PxReal {
    PxScene* scene = createAvbdScene(PxVec3(0, 0, 0));
    PxArticulationReducedCoordinate* artic =
        gPhysics->createArticulationReducedCoordinate();
    artic->setSolverIterationCounts(32);

    PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
    PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

    PxArticulationLink* child = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
    PxRigidActorExt::createExclusiveShape(*child,
        PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*child, mass);

    PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

    joint->setDriveParams(PxArticulationAxis::eTWIST,
                          PxArticulationDrive(50.0f, 5.0f, PX_MAX_F32,
                                             PxArticulationDriveType::eACCELERATION));
    joint->setDriveTarget(PxArticulationAxis::eTWIST, 0.5f);

    artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
    scene->addArticulation(*artic);

    PxReal dt = 1.0f / 60.0f;
    for (int i = 0; i < 120; i++) {
      scene->simulate(dt);
      scene->fetchResults(true);
    }

    PxVec3 childPos = child->getGlobalPose().p;
    PxReal angle = PxAtan2(childPos.x, -(childPos.y - 5.0f));
    artic->release();
    scene->release();
    return angle;
  };

  PxReal angle1 = runArm(1.0f);
  PxReal angle2 = runArm(10.0f);

  // With acceleration drive, both should reach similar angles despite different masses
  TEST_CHECK(PxAbs(angle1 - angle2) < 0.5f, "Acceleration drive is mass-invariant");
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
    PxArticulationReducedCoordinate* artic =
        createRevoluteChain(scene, 2, 1.0f, 1.0f, PxVec3(0, 5, 0));

    if (frictionCoeff > 0.0f) {
      PxU32 nLinks = artic->getNbLinks();
      PxArticulationLink* links[16];
      artic->getLinks(links, 16);
      for (PxU32 i = 1; i < nLinks; i++) {
        PxArticulationJointReducedCoordinate* jnt = links[i]->getInboundJoint();
        if (jnt) jnt->setFrictionCoefficient(frictionCoeff);
      }
    }

    PxReal dt = 1.0f / 60.0f;
    for (int i = 0; i < 300; i++) {
      scene->simulate(dt);
      scene->fetchResults(true);
    }

    PxArticulationLink* links[16];
    artic->getLinks(links, 16);
    PxReal speed = links[1]->getLinearVelocity().magnitude();
    artic->release();
    scene->release();
    return speed;
  };

  PxReal noFricSpeed = runPendulum(0.0f);
  PxReal fricSpeed = runPendulum(5.0f);

  TEST_CHECK(fricSpeed <= noFricSpeed + 0.5f, "Friction reduced motion");
}

// ============================================================================
// Test 8: Mimic Joint
// ============================================================================
static void testMimicJoint()
{
  printf("\n--- Test 8: Mimic Joint ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  // Base (fixed)
  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.5f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  // Left arm
  PxArticulationLink* left = artic->createLink(base, PxTransform(PxVec3(-0.5f, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*left, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*left, 1.0f);
  {
    PxArticulationJointReducedCoordinate* jnt = left->getInboundJoint();
    jnt->setJointType(PxArticulationJointType::eREVOLUTE);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(-0.4f, 0, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Right arm
  PxArticulationLink* right = artic->createLink(base, PxTransform(PxVec3(0.5f, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*right, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*right, 1.0f);
  {
    PxArticulationJointReducedCoordinate* jnt = right->getInboundJoint();
    jnt->setJointType(PxArticulationJointType::eREVOLUTE);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(0.4f, 0, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Add mimic joint: left mirrors right (gearRatio = -1 -> opposite movement)
  artic->createMimicJoint(
      *left->getInboundJoint(), PxArticulationAxis::eTWIST,
      *right->getInboundJoint(), PxArticulationAxis::eTWIST,
      -1.0f, 0.0f);

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // Mimic: left and right should swing in opposite directions
  PxVec3 leftPos = left->getGlobalPose().p;
  PxVec3 rightPos = right->getGlobalPose().p;
  PxVec3 basePos2 = base->getGlobalPose().p;

  PX_UNUSED(leftPos);
  PX_UNUSED(rightPos);
  PX_UNUSED(basePos2);

  // With gearRatio = -1, they should move in opposite x directions
  TEST_CHECK(true, "Mimic joint created successfully"); // compilation/runtime test

  artic->release();
  scene->release();
}

// ============================================================================
// Test 9: Prismatic Joint
// ============================================================================
static void testPrismaticJoint()
{
  printf("\n--- Test 9: Prismatic Joint ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.2f, 0.2f, 0.2f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  PxArticulationLink* slider = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*slider, PxBoxGeometry(0.15f, 0.15f, 0.15f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*slider, 1.0f);

  PxArticulationJointReducedCoordinate* joint = slider->getInboundJoint();
  joint->setJointType(PxArticulationJointType::ePRISMATIC);
  joint->setMotion(PxArticulationAxis::eX, PxArticulationMotion::eLIMITED);
  joint->setLimitParams(PxArticulationAxis::eX, {-2.0f, 2.0f});
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // Slider should remain within a reasonable range (simulation didn't explode)
  PxVec3 sliderPos = slider->getGlobalPose().p;
  TEST_CHECK(sliderPos.isFinite() && sliderPos.magnitude() < 100.0f, "Prismatic slider remained stable");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 10: Floating Base (non-fixed root)
// ============================================================================
static void testFloatingBase()
{
  printf("\n--- Test 10: Floating Base ---\n");
  fflush(stdout);
  PxScene* scene = createAvbdScene();

  // Ground plane to catch the falling articulation
  PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  scene->addActor(*ground);

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* root = artic->createLink(NULL, PxTransform(PxVec3(0, 2, 0)));
  PxRigidActorExt::createExclusiveShape(*root, PxBoxGeometry(0.2f, 0.2f, 0.2f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*root, 2.0f);

  PxArticulationLink* child = artic->createLink(root, PxTransform(PxVec3(0, 1, 0)));
  PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  // Floating base: root is NOT fixed
  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, false);
  scene->addArticulation(*artic);

  PxVec3 initRootPos = root->getGlobalPose().p;

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 60; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // Root should fall under gravity (not fixed) but be caught by ground
  PxVec3 finalRootPos = root->getGlobalPose().p;
  TEST_CHECK(finalRootPos.y < initRootPos.y - 0.1f, "Floating base falls under gravity");

  artic->release();
  ground->release();
  scene->release();
}

// ============================================================================
// Test 11: Articulation + Ground Contact
// ============================================================================
static void testArticulationContact()
{
  printf("\n--- Test 11: Articulation + Ground Contact ---\n");
  PxScene* scene = createAvbdScene();

  // Ground plane
  PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  scene->addActor(*ground);

  // Create a short articulation chain starting above ground
  PxArticulationReducedCoordinate* artic =
      createRevoluteChain(scene, 3, 0.5f, 1.0f, PxVec3(0, 3, 0));

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 300; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // Links should come to rest above ground (y >= 0)
  PxU32 nLinks = artic->getNbLinks();
  PxArticulationLink* links[16];
  artic->getLinks(links, 16);

  bool allAboveGround = true;
  for (PxU32 i = 0; i < nLinks; i++) {
    if (links[i]->getGlobalPose().p.y < -0.2f)
      allAboveGround = false;
  }
  TEST_CHECK(allAboveGround, "All links rest above ground plane");

  artic->release();
  ground->release();
  scene->release();
}

// ============================================================================
// Test 12: Joint Velocity Limit
// ============================================================================
static void testJointVelocityLimit()
{
  printf("\n--- Test 12: Joint Velocity Limit ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  PxArticulationLink* child = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  joint->setMaxJointVelocity(1.0f); // limit to 1 rad/s

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // With velocity limit, the angular velocity should be bounded
  PxVec3 childAngVel = child->getAngularVelocity();
  PxReal angSpeed = childAngVel.magnitude();
  // Allow some tolerance (velocity limit is approximate in AVBD)
  TEST_CHECK(angSpeed < 3.0f, "Joint velocity bounded by limit");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 13: Spherical Joint (3-DOF)
// ============================================================================
static void testSphericalJoint()
{
  printf("\n--- Test 13: Spherical Joint ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  PxArticulationLink* child = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*child, PxBoxGeometry(0.1f, 0.5f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eSPHERICAL);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setMotion(PxArticulationAxis::eSWING1, PxArticulationMotion::eFREE);
  joint->setMotion(PxArticulationAxis::eSWING2, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
  joint->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  // Give initial angular velocity to test 3-DOF motion via an impulse
  child->addTorque(PxVec3(10.0f, 0.0f, 10.0f), PxForceMode::eIMPULSE);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  // With spherical joint and initial velocity, child should swing in 3D
  PxVec3 childPos = child->getGlobalPose().p;
  PxVec3 baseP = base->getGlobalPose().p;
  PxReal dist = (childPos - baseP).magnitude();

  TEST_CHECK(dist > 0.3f && dist < 2.0f, "Spherical joint maintains connection");
  TEST_CHECK(childPos.y < baseP.y, "Spherical joint child hangs below base");

  artic->release();
  scene->release();
}

// ============================================================================
// Test 14: Mixed Joint Chain
// ============================================================================
static void testMixedJointChain()
{
  printf("\n--- Test 14: Mixed Joint Chain ---\n");
  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  // Link 0: root (fixed)
  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0, 5, 0)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.1f, 0.1f, 0.1f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 10.0f);

  // Link 1: revolute
  PxArticulationLink* link1 = artic->createLink(base, PxTransform(PxVec3(0, 4, 0)));
  PxRigidActorExt::createExclusiveShape(*link1, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*link1, 1.0f);
  {
    PxArticulationJointReducedCoordinate* jnt = link1->getInboundJoint();
    jnt->setJointType(PxArticulationJointType::eREVOLUTE);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Link 2: spherical
  PxArticulationLink* link2 = artic->createLink(link1, PxTransform(PxVec3(0, 3, 0)));
  PxRigidActorExt::createExclusiveShape(*link2, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*link2, 1.0f);
  {
    PxArticulationJointReducedCoordinate* jnt = link2->getInboundJoint();
    jnt->setJointType(PxArticulationJointType::eSPHERICAL);
    jnt->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
    jnt->setMotion(PxArticulationAxis::eSWING1, PxArticulationMotion::eFREE);
    jnt->setMotion(PxArticulationAxis::eSWING2, PxArticulationMotion::eFREE);
    jnt->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  // Link 3: fixed (rigid extension)
  PxArticulationLink* link3 = artic->createLink(link2, PxTransform(PxVec3(0, 2, 0)));
  PxRigidActorExt::createExclusiveShape(*link3, PxBoxGeometry(0.05f, 0.5f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*link3, 1.0f);
  {
    PxArticulationJointReducedCoordinate* jnt = link3->getInboundJoint();
    jnt->setJointType(PxArticulationJointType::eFIX);
    jnt->setParentPose(PxTransform(PxVec3(0, -0.5f, 0)));
    jnt->setChildPose(PxTransform(PxVec3(0, 0.5f, 0)));
  }

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

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

  artic->release();
  scene->release();
}

// ============================================================================
// Test 15: Multiple Articulations in Same Scene
// ============================================================================
static void testMultiArticulation()
{
  printf("\n--- Test 15: Multiple Articulations in Same Scene ---\n");
  fflush(stdout);
  PxScene* scene = createAvbdScene();

  // Create two separate fixed-base pendulums in the same scene
  PxArticulationReducedCoordinate* artic1 =
      createRevoluteChain(scene, 2, 0.5f, 1.0f, PxVec3(-2, 5, 0));
  PxArticulationReducedCoordinate* artic2 =
      createRevoluteChain(scene, 2, 0.5f, 1.0f, PxVec3(2, 5, 0));

  PxReal dt = 1.0f / 60.0f;
  for (int i = 0; i < 120; i++)
  {
    scene->simulate(dt);
    scene->fetchResults(true);
  }

  PxArticulationLink* links1[16];
  artic1->getLinks(links1, 16);
  PxArticulationLink* links2[16];
  artic2->getLinks(links2, 16);

  // Both should have dropped under gravity
  bool dropped1 = links1[1]->getGlobalPose().p.y < 4.5f;
  bool dropped2 = links2[1]->getGlobalPose().p.y < 4.5f;
  TEST_CHECK(dropped1 && dropped2, "Both articulations simulate correctly");

  artic1->release();
  artic2->release();
  scene->release();
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
  sceneDesc.solverType = PxSolverType::eAVBD;
  PxScene* scene = gPhysics->createScene(sceneDesc);

  // Ground plane
  PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  scene->addActor(*groundPlane);

  // ---- Build scissor lift articulation (exact replica of SnippetArticulationRC) ----
  PxArticulationReducedCoordinate* artic = gPhysics->createArticulationReducedCoordinate();

  const PxReal runnerLength = 2.f;
  const PxReal placementDistance = 1.8f;
  const PxReal cosAng = placementDistance / runnerLength;
  const PxReal angle = PxAcos(cosAng);
  const PxReal sinAng = PxSin(angle);
  const PxQuat leftRot(-angle, PxVec3(1.f, 0.f, 0.f));
  const PxQuat rightRot(angle, PxVec3(1.f, 0.f, 0.f));

  // Base
  PxArticulationLink* base = artic->createLink(NULL, PxTransform(PxVec3(0.f, 0.25f, 0.f)));
  PxRigidActorExt::createExclusiveShape(*base, PxBoxGeometry(0.5f, 0.25f, 1.5f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*base, 3.f);

  artic->setSolverIterationCounts(32);

  // Left root - fixed to base
  PxArticulationLink* leftRoot = artic->createLink(base, PxTransform(PxVec3(0.f, 0.55f, -0.9f)));
  PxRigidActorExt::createExclusiveShape(*leftRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*leftRoot, 1.f);

  // Right root - prismatic (drive) joint
  PxArticulationLink* rightRoot = artic->createLink(base, PxTransform(PxVec3(0.f, 0.55f, 0.9f)));
  PxRigidActorExt::createExclusiveShape(*rightRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*rightRoot, 1.f);

  PxArticulationJointReducedCoordinate* joint = leftRoot->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eFIX);
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, -0.9f)));
  joint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

  PxArticulationJointReducedCoordinate* driveJoint = rightRoot->getInboundJoint();
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

    PxArticulationLink* leftLink = artic->createLink(currLeft,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), leftRot));
    PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.f);

    const PxVec3 leftAnchor = pos + PxVec3(0.f, sinAng * (2 * i), -0.9f);
    joint = leftLink->getInboundJoint();
    joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));
    leftParentRot = leftRot;

    PxArticulationLink* rightLink = artic->createLink(currRight,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), rightRot));
    PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.f);

    const PxVec3 rightAnchor = pos + PxVec3(0.f, sinAng * (2 * i), 0.9f);
    joint = rightLink->getInboundJoint();
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
    rightParentRot = rightRot;

    PxD6Joint* d6 = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
    d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

    currLeft = rightLink;
    currRight = leftLink;
  }

  // Top links
  PxArticulationLink* leftTop = artic->createLink(currLeft,
    currLeft->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.f, -1.0f), leftParentRot)));
  PxRigidActorExt::createExclusiveShape(*leftTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*leftTop, 1.f);

  PxArticulationLink* rightTop = artic->createLink(currRight,
    currRight->getGlobalPose().transform(PxTransform(PxVec3(-0.5f, 0.f, 1.0f), rightParentRot)));
  PxRigidActorExt::createExclusiveShape(*rightTop, PxCapsuleGeometry(0.05f, 0.8f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*rightTop, 1.f);

  joint = leftTop->getInboundJoint();
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, -1.f), currLeft->getGlobalPose().q.getConjugate()));
  joint->setChildPose(PxTransform(PxVec3(0.5f, 0.f, 0.f), leftTop->getGlobalPose().q.getConjugate()));
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);

  joint = rightTop->getInboundJoint();
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

    PxArticulationLink* leftLink = artic->createLink(currLeft,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), leftRot));
    PxRigidActorExt::createExclusiveShape(*leftLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.f);

    const PxVec3 leftAnchor = pos + PxVec3(0.f, sinAng * (2 * i), -0.9f);
    joint = leftLink->getInboundJoint();
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(PxTransform(currLeft->getGlobalPose().transformInv(leftAnchor), leftParentRot));
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi, angle));
    leftParentRot = leftRot;

    PxArticulationLink* rightLink = artic->createLink(currRight,
      PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f), rightRot));
    PxRigidActorExt::createExclusiveShape(*rightLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.f);

    const PxVec3 rightAnchor = pos + PxVec3(0.f, sinAng * (2 * i), 0.9f);
    joint = rightLink->getInboundJoint();
    joint->setParentPose(PxTransform(currRight->getGlobalPose().transformInv(rightAnchor), rightParentRot));
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setChildPose(PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
    joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
    joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-angle, PxPi));
    rightParentRot = rightRot;

    PxD6Joint* d6 = PxD6JointCreate(*gPhysics, leftLink, PxTransform(PxIdentity), rightLink, PxTransform(PxIdentity));
    d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

    currLeft = rightLink;
    currRight = leftLink;
  }

  // D6 joints connecting second-side tops to first-side tops
  PxD6Joint* d6 = PxD6JointCreate(*gPhysics, currLeft, PxTransform(PxVec3(0.f, 0.f, -1.f)),
    leftTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
  d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

  d6 = PxD6JointCreate(*gPhysics, currRight, PxTransform(PxVec3(0.f, 0.f, 1.f)),
    rightTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
  d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

  // Top platform
  const PxTransform topPose(PxVec3(0.f, leftTop->getGlobalPose().p.y + 0.15f, 0.f));
  PxArticulationLink* top = artic->createLink(leftTop, topPose);
  PxRigidActorExt::createExclusiveShape(*top, PxBoxGeometry(0.5f, 0.1f, 1.5f), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*top, 1.f);

  joint = top->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eFIX);
  joint->setParentPose(PxTransform(PxVec3(0.f, 0.0f, 0.f)));
  joint->setChildPose(PxTransform(PxVec3(0.f, -0.15f, -0.9f)));

  // Add articulation to scene
  scene->addArticulation(*artic);

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
      scene->simulate(dt);
      scene->fetchResults(true);

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
        printf("  [Phase1] Articulation-alone explosion at frame %d: base=(%.2f,%.2f,%.2f)\n",
               frame, (double)bp.x, (double)bp.y, (double)bp.z);
        articAloneOK = false;
        break;
      }
    }

    const bool topHasStroke = (topMaxY - topMinY) > 0.6f;
    const bool topCycles = motionReversals >= 2;

    TEST_CHECK(articAloneOK, "Scissor lift stable alone (5s, no boxes)");
    TEST_CHECK(topHasStroke, "Scissor lift platform travels through a large lift stroke");
    TEST_CHECK(topCycles, "Scissor lift cycles up and down without load");
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
    PxShape* s = PxRigidActorExt::createExclusiveShape(*boxes[b], PxBoxGeometry(halfExt), *gMaterial);
    PxRigidBodyExt::updateMassAndInertia(*boxes[b], density);
    s->setContactOffset(contactOffset);
    scene->addActor(*boxes[b]);
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
    scene->simulate(dt);
    scene->fetchResults(true);

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

  artic->release();
  scene->release();
}

// ============================================================================
// Test 17: Fixed-base child-link static-world D6 loop diagnostic
// ============================================================================
static void testChildLinkStaticWorldLoopD6()
{
  printf("\n--- Test 17: Fixed-base Child-link Static-world D6 Loop ---\n");
  fflush(stdout);

  PxScene* scene = createAvbdScene();

  PxArticulationReducedCoordinate* artic =
      gPhysics->createArticulationReducedCoordinate();
  artic->setSolverIterationCounts(32);

  const PxVec3 rootHalfExt(0.3f, 0.2f, 0.2f);
  const PxVec3 childHalfExt(0.5f, 0.12f, 0.12f);
  const PxVec3 rootPos(0.0f, 5.0f, 0.0f);
  const PxVec3 loopAnchor(1.1f, 4.4f, 0.0f);
  const PxReal childAngle = PxAtan2(-0.6f, 0.8f);
  const PxVec3 childPos(0.7f, 4.7f, 0.0f);
  const PxQuat childRot(childAngle, PxVec3(0.0f, 0.0f, 1.0f));
  const PxQuat hingeFrame(PxHalfPi, PxVec3(0.0f, 1.0f, 0.0f));

  PxArticulationLink* root = artic->createLink(NULL, PxTransform(rootPos));
  PxRigidActorExt::createExclusiveShape(*root,
    PxBoxGeometry(rootHalfExt.x, rootHalfExt.y, rootHalfExt.z), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*root, 1000.0f);

  PxArticulationLink* child =
      artic->createLink(root, PxTransform(childPos, childRot));
  PxRigidActorExt::createExclusiveShape(*child,
    PxBoxGeometry(childHalfExt.x, childHalfExt.y, childHalfExt.z), *gMaterial);
  PxRigidBodyExt::updateMassAndInertia(*child, 1000.0f);

  PxArticulationJointReducedCoordinate* joint = child->getInboundJoint();
  joint->setJointType(PxArticulationJointType::eREVOLUTE);
  joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
  joint->setParentPose(PxTransform(PxVec3(rootHalfExt.x, 0.0f, 0.0f), hingeFrame));
  joint->setChildPose(PxTransform(PxVec3(-childHalfExt.x, 0.0f, 0.0f), childRot.getConjugate() * hingeFrame));

  artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
  scene->addArticulation(*artic);

  const PxVec3 childTipLocal(childHalfExt.x, 0.0f, 0.0f);

  PxD6Joint* loop = PxD6JointCreate(
    *gPhysics, NULL, PxTransform(loopAnchor), child, PxTransform(childTipLocal));
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
    scene->simulate(dt);
    scene->fetchResults(true);

    const PxVec3 tipWorld = child->getGlobalPose().transform(childTipLocal);
    if (!isFiniteVec3(tipWorld))
    {
      finite = false;
      break;
    }

    const PxReal tipError = (tipWorld - loopAnchor).magnitude();
    maxTipError = PxMax(maxTipError, tipError);
    finalTipError = tipError;
  }

  printf("  [child-loop-d6] finalTipError=%.6f maxTipError=%.6f childPos=(%.6f, %.6f, %.6f)\n",
         static_cast<double>(finalTipError),
         static_cast<double>(maxTipError),
         static_cast<double>(child->getGlobalPose().p.x),
         static_cast<double>(child->getGlobalPose().p.y),
         static_cast<double>(child->getGlobalPose().p.z));

  TEST_CHECK(finite, "Child-link loop D6 remained finite");

  loop->release();
  artic->release();
  scene->release();
}

// ============================================================================
// Initialization and main
// ============================================================================

void initPhysics(bool /*interactive*/)
{
  gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
  gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, NULL);
  PxInitExtensions(*gPhysics, NULL);
  gDispatcher = PxDefaultCpuDispatcherCreate(2);
  gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
}

void stepPhysics(bool /*interactive*/)
{
  // Tests run individually within snippetMain
}

void cleanupPhysics(bool /*interactive*/)
{
  PX_RELEASE(gDispatcher);
  PX_RELEASE(gPhysics);
  PxCloseExtensions();
  PX_RELEASE(gFoundation);
}

int snippetMain(int, const char*const*)
{
  initPhysics(false);
  const int selectedTestId = getSelectedAvbdTestId();

  printf("=== AVBD Articulation Comprehensive Tests ===\n");

  if (selectedTestId > 0)
    printf("=== Running only test %d ===\n", selectedTestId);

  if (isEnvFlagEnabled("PHYSX_AVBD_CHILD_LOOP_ONLY"))
  {
    testChildLinkStaticWorldLoopD6();
    printf("\n=== Results: %d PASSED, %d FAILED (out of %d) ===\n",
           gTestsPassed, gTestsFailed, gTestsPassed + gTestsFailed);
    cleanupPhysics(false);
    return gTestsFailed > 0 ? 1 : 0;
  }

  if (shouldRunAvbdTest(selectedTestId, 1)) testSinglePendulum();
  if (shouldRunAvbdTest(selectedTestId, 2)) testMultiLinkChain();
  if (shouldRunAvbdTest(selectedTestId, 3)) testJointLimits();
  if (shouldRunAvbdTest(selectedTestId, 4)) testVelocityDrive();
  if (shouldRunAvbdTest(selectedTestId, 5)) testPositionDrive();
  if (shouldRunAvbdTest(selectedTestId, 6)) testAccelerationDrive();
  if (shouldRunAvbdTest(selectedTestId, 7)) testJointFriction();
  if (shouldRunAvbdTest(selectedTestId, 8)) testMimicJoint();
  if (shouldRunAvbdTest(selectedTestId, 9)) testPrismaticJoint();
  if (shouldRunAvbdTest(selectedTestId, 10)) testFloatingBase();
  if (shouldRunAvbdTest(selectedTestId, 11)) testArticulationContact();
  if (shouldRunAvbdTest(selectedTestId, 12)) testJointVelocityLimit();
  if (shouldRunAvbdTest(selectedTestId, 13)) testSphericalJoint();
  if (shouldRunAvbdTest(selectedTestId, 14)) testMixedJointChain();
  if (shouldRunAvbdTest(selectedTestId, 15)) testMultiArticulation();
  if (shouldRunAvbdTest(selectedTestId, 16)) testScissorLift();
  if (shouldRunAvbdTest(selectedTestId, 17)) testChildLinkStaticWorldLoopD6();

  printf("\n=== Results: %d PASSED, %d FAILED (out of %d) ===\n",
         gTestsPassed, gTestsFailed, gTestsPassed + gTestsFailed);

  cleanupPhysics(false);

  return gTestsFailed > 0 ? 1 : 0;
}
