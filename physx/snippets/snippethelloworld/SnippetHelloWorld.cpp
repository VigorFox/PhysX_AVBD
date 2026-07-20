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
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

// Box stacks on a plane; optional headless ball shot knocks stacks down.

#include <ctype.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "PxPhysicsAPI.h"

#ifdef RENDER_SNIPPET
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#endif

using namespace physx;

static PxDefaultAllocator gAllocator;
static PxDefaultErrorCallback gErrorCallback;
static PxFoundation *gFoundation = NULL;
static PxPhysics *gPhysics = NULL;
static PxDefaultCpuDispatcher *gDispatcher = NULL;
static PxScene *gScene = NULL;
static PxMaterial *gMaterial = NULL;
static PxPvd *gPvd = NULL;

static PxReal stackZ = 10.0f;
static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;
static bool gHeadlessMode = false;
static bool gHeadlessBallShot = false;
static PxU32 gHeadlessFrameCount = 600;
static PxU32 gBallShotFrame = 30;
static PxU32 gSimFrame = 0;
static PxRigidDynamic *gShotBall = NULL;

static const PxReal gBallShotRadius = 3.0f;
static const PxVec3 gBallShotPos(0.0f, 22.0f, 70.0f);
static const PxVec3 gBallShotVel(0.0f, -20.0f, -220.0f);

struct HelloWorldMetrics {
  PxReal maxSpeedAll = 0.0f;
  PxReal maxBoxSpeedSettle = 0.0f;
  PxReal maxBoxSpeedDeltaSettle = 0.0f;
  PxReal maxBoxCenterY = -1e9f;
  PxReal minBoxCenterY = 1e9f;
  PxReal initialAvgBoxZ = 0.0f;
  PxReal settleAvgBoxZ = 0.0f;
  PxReal maxSettleBoxVxz = 0.0f;
  PxReal maxImpactBottomVz = 0.0f;
  PxReal maxImpactBottomZDrift = 0.0f;
  // Ball ground-bounce diagnostics (restitution e=0.6 material).
  PxReal maxBallUpVy = 0.0f;     // max +vy after shot (rebound signature)
  PxReal maxBallSpeed = 0.0f;
  PxReal ballVyAtFirstGround = 0.0f; // vy when first near plane (y < r+0.5)
  bool ballSawGround = false;
  PxU32 awakeBoxesSettle = 0;
  PxReal tailAvgBoxSpeed = 0.0f;
  PxU32 tailSpeedSamples = 0;
  bool nanDetected = false;
  bool initialSampled = false;
};

static HelloWorldMetrics gMetrics;
static PxReal gPrevMaxBoxSpeed = 0.0f;

static bool hasArg(int argc, const char *const *argv, const char *name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && std::strcmp(argv[i], name) == 0)
      return true;
  }
  return false;
}

static PxSolverType::Enum getRequestedSolverType(int argc,
                                                 const char *const *argv) {
  for (int i = 1; i < argc; ++i) {
    if (!argv[i])
      continue;
    if (std::strncmp(argv[i], "--solver=", 9) == 0) {
      const char *v = argv[i] + 9;
      if (std::strcmp(v, "tgs") == 0 || std::strcmp(v, "TGS") == 0)
        return PxSolverType::eTGS;
      if (std::strcmp(v, "avbd") == 0 || std::strcmp(v, "AVBD") == 0)
        return PxSolverType::eAVBD;
    }
  }
  return PxSolverType::eAVBD;
}

static const char *solverName(PxSolverType::Enum t) {
  return (t == PxSolverType::eTGS) ? "tgs" : "avbd";
}

static bool isHeadlessRequested(int argc, const char *const *argv) {
  if (hasArg(argc, argv, "--headless") ||
      hasArg(argc, argv, "--headless-ball-shot"))
    return true;
  const char *value = std::getenv("PHYSX_SNIPPET_HEADLESS");
  return value && value[0] && value[0] != '0';
}

static PxRigidDynamic *createDynamic(const PxTransform &t,
                                     const PxGeometry &geometry,
                                     const PxVec3 &velocity = PxVec3(0)) {
  PxRigidDynamic *dynamic =
      PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
  dynamic->setAngularDamping(0.5f);
  dynamic->setLinearVelocity(velocity);
  gScene->addActor(*dynamic);
  return dynamic;
}

static void createStack(const PxTransform &t, PxU32 size, PxReal halfExtent) {
  PxShape *shape = gPhysics->createShape(
      PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
  for (PxU32 i = 0; i < size; i++) {
    for (PxU32 j = 0; j < size - i; j++) {
      PxTransform localTm(
          PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) *
          halfExtent);
      PxRigidDynamic *body = gPhysics->createRigidDynamic(t.transform(localTm));
      body->attachShape(*shape);
      PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
      gScene->addActor(*body);
    }
  }
  shape->release();
}

static void spawnBallShot() {
  if (gShotBall)
    return;
  gShotBall = createDynamic(PxTransform(gBallShotPos),
                            PxSphereGeometry(gBallShotRadius), gBallShotVel);
  printf("[HelloWorldBallShot] spawn pos=(%.1f,%.1f,%.1f) vel=(%.1f,%.1f,%.1f) "
         "radius=%.2f\n",
         gBallShotPos.x, gBallShotPos.y, gBallShotPos.z, gBallShotVel.x,
         gBallShotVel.y, gBallShotVel.z, gBallShotRadius);
}

static void sampleDynamics(PxU32 frameIndex, PxU32 settleTailFrames) {
  const PxU32 nbDyn = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  if (nbDyn == 0)
    return;

  PxArray<PxRigidActor *> actors(nbDyn);
  gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
                    reinterpret_cast<PxActor **>(actors.begin()), nbDyn);

  PxReal maxBoxSpeed = 0.0f;
  PxReal sumBoxSpeed = 0.0f;
  PxU32 countAbove5 = 0;
  PxU32 countAbove15 = 0;
  PxReal maxBoxVxz = 0.0f;
  PxReal maxBottomVz = 0.0f;
  PxReal sumBottomZ = 0.0f;
  PxU32 bottomCount = 0;
  PxU32 awakeBoxes = 0;
  PxReal sumY = 0.0f;
  PxReal sumZ = 0.0f;
  PxU32 boxCount = 0;

  for (PxU32 a = 0; a < nbDyn; ++a) {
    PxRigidDynamic *rb = actors[a] ? actors[a]->is<PxRigidDynamic>() : NULL;
    if (!rb)
      continue;

    const PxVec3 p = rb->getGlobalPose().p;
    const PxVec3 v = rb->getLinearVelocity();
    if (!PxIsFinite(p.x) || !PxIsFinite(p.y) || !PxIsFinite(p.z) ||
        !PxIsFinite(v.x) || !PxIsFinite(v.y) || !PxIsFinite(v.z)) {
      gMetrics.nanDetected = true;
      continue;
    }

    const PxReal speed = v.magnitude();
    gMetrics.maxSpeedAll = PxMax(gMetrics.maxSpeedAll, speed);

    const bool isBall = (rb == gShotBall);
    if (isBall) {
      gMetrics.maxBallSpeed = PxMax(gMetrics.maxBallSpeed, speed);
      if (v.y > gMetrics.maxBallUpVy)
        gMetrics.maxBallUpVy = v.y;
      // First near-ground sample after shot: capture post-impact normal speed.
      if (!gMetrics.ballSawGround && p.y < (gBallShotRadius + 0.5f)) {
        gMetrics.ballSawGround = true;
        gMetrics.ballVyAtFirstGround = v.y;
      }
    } else {
      boxCount++;
      sumY += p.y;
      sumZ += p.z;
      gMetrics.maxBoxCenterY = PxMax(gMetrics.maxBoxCenterY, p.y);
      gMetrics.minBoxCenterY = PxMin(gMetrics.minBoxCenterY, p.y);
      maxBoxSpeed = PxMax(maxBoxSpeed, speed);
      sumBoxSpeed += speed;
      if (speed > 5.0f)
        countAbove5++;
      if (speed > 15.0f)
        countAbove15++;
      maxBoxVxz = PxMax(maxBoxVxz, PxSqrt(v.x * v.x + v.z * v.z));
      if (p.y < 4.0f) {
        bottomCount++;
        sumBottomZ += p.z;
        maxBottomVz = PxMax(maxBottomVz, PxAbs(v.z));
      }
      if (!rb->isSleeping())
        awakeBoxes++;
    }
  }

  const bool inImpactWindow =
      gHeadlessBallShot && frameIndex >= gBallShotFrame &&
      frameIndex <= gBallShotFrame + 70u;
  if (inImpactWindow && bottomCount > 0) {
    gMetrics.maxImpactBottomVz =
        PxMax(gMetrics.maxImpactBottomVz, maxBottomVz);
    gMetrics.maxImpactBottomZDrift =
        PxMax(gMetrics.maxImpactBottomZDrift,
              PxAbs(sumBottomZ / PxReal(bottomCount) - gMetrics.initialAvgBoxZ));
  }
  const bool inSettleWindow =
      frameIndex + settleTailFrames >= gHeadlessFrameCount;
  if (boxCount > 0 && !gMetrics.initialSampled) {
    gMetrics.initialAvgBoxZ = sumZ / PxReal(boxCount);
    gMetrics.initialSampled = true;
  }
  if (inSettleWindow && boxCount > 0) {
    gMetrics.maxBoxSpeedSettle = PxMax(gMetrics.maxBoxSpeedSettle, maxBoxSpeed);
    gMetrics.maxSettleBoxVxz = PxMax(gMetrics.maxSettleBoxVxz, maxBoxVxz);
    gMetrics.settleAvgBoxZ = sumZ / PxReal(boxCount);
    gMetrics.maxBoxSpeedDeltaSettle =
        PxMax(gMetrics.maxBoxSpeedDeltaSettle,
              PxAbs(maxBoxSpeed - gPrevMaxBoxSpeed));
    gPrevMaxBoxSpeed = maxBoxSpeed;
    gMetrics.awakeBoxesSettle = PxMax(gMetrics.awakeBoxesSettle, awakeBoxes);
    gMetrics.tailAvgBoxSpeed += maxBoxSpeed;
    gMetrics.tailSpeedSamples++;
  }

  const char *traceEnv = std::getenv("AVBD_HELLOWORLD_TRACE");
  PxReal maxSpeedBoxY = 0.0f;
  if (traceEnv && traceEnv[0] && traceEnv[0] != '0' && gHeadlessBallShot &&
      frameIndex >= gBallShotFrame && (frameIndex - gBallShotFrame) % 10 == 0) {
    for (PxU32 a = 0; a < nbDyn; ++a) {
      PxRigidDynamic *rb = actors[a] ? actors[a]->is<PxRigidDynamic>() : NULL;
      if (!rb || rb == gShotBall)
        continue;
      const PxReal speed = rb->getLinearVelocity().magnitude();
      if (speed >= maxBoxSpeed - 1e-4f)
        maxSpeedBoxY = rb->getGlobalPose().p.y;
    }
  }
  if (traceEnv && traceEnv[0] && traceEnv[0] != '0' && gHeadlessBallShot &&
      frameIndex >= gBallShotFrame && (frameIndex - gBallShotFrame) % 10 == 0) {
    const PxReal avgSpeed =
        boxCount > 0 ? sumBoxSpeed / PxReal(boxCount) : 0.0f;
    printf("[HelloWorldTrace] frame=%u maxBoxSpeed=%.3f avgBoxSpeed=%.3f "
           "above5=%u above15=%u awakeBoxes=%u maxSpeedBoxY=%.2f\n",
           frameIndex, maxBoxSpeed, avgSpeed, countAbove5, countAbove15,
           awakeBoxes, maxSpeedBoxY);
  }
}

static void printHeadlessSummary() {
  const PxReal passMaxSettleSpeed = 12.0f;
  const PxReal passMaxJitter = 3.0f;
  const bool pass =
      !gMetrics.nanDetected && gMetrics.maxBoxSpeedSettle < passMaxSettleSpeed &&
      gMetrics.maxBoxSpeedDeltaSettle < passMaxJitter;
  printf("\n[HelloWorld] SUMMARY solver=%s frames=%u ballShot=%s\n",
         solverName(gSolverType), gHeadlessFrameCount,
         gHeadlessBallShot ? "yes" : "no");
  printf("[HelloWorld] maxSpeedAll=%.4f maxBoxSpeedSettle=%.4f "
         "maxBoxSpeedDeltaSettle=%.4f maxSettleBoxVxz=%.4f awakeBoxesSettle=%u\n",
         gMetrics.maxSpeedAll, gMetrics.maxBoxSpeedSettle,
         gMetrics.maxBoxSpeedDeltaSettle, gMetrics.maxSettleBoxVxz,
         gMetrics.awakeBoxesSettle);
  printf("[HelloWorld] boxCenterY=[%.4f,%.4f] boxZ drift=%.4f "
         "(initialAvgZ=%.4f settleAvgZ=%.4f) nan=%s\n",
         gMetrics.minBoxCenterY, gMetrics.maxBoxCenterY,
         gMetrics.settleAvgBoxZ - gMetrics.initialAvgBoxZ,
         gMetrics.initialAvgBoxZ, gMetrics.settleAvgBoxZ,
         gMetrics.nanDetected ? "true" : "false");
  if (gHeadlessBallShot) {
    printf("[HelloWorld] impactBottom max|vz|=%.4f maxZDrift=%.4f\n",
           gMetrics.maxImpactBottomVz, gMetrics.maxImpactBottomZDrift);
    printf("[HelloWorld] ballBounce maxUpVy=%.4f vyAtFirstGround=%.4f "
           "maxBallSpeed=%.4f sawGround=%s\n",
           gMetrics.maxBallUpVy, gMetrics.ballVyAtFirstGround,
           gMetrics.maxBallSpeed, gMetrics.ballSawGround ? "yes" : "no");
  }
  if (gMetrics.tailSpeedSamples > 0) {
    printf("[HelloWorld] tailAvgMaxBoxSpeed=%.4f (samples=%u)\n",
           gMetrics.tailAvgBoxSpeed / PxReal(gMetrics.tailSpeedSamples),
           gMetrics.tailSpeedSamples);
  }
  printf("[HelloWorld] RESULT pass=%d (fail if nan or settleSpeed>%.1f or "
         "jitter>%.1f)\n",
         pass ? 1 : 0, passMaxSettleSpeed, passMaxJitter);
}

void initPhysics(bool interactive) {
  gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

  gPvd = PxCreatePvd(*gFoundation);
  PxPvdTransport *transport =
      PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
  gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

  gPhysics =
      PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true,
                      gPvd);

  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
  gDispatcher = PxDefaultCpuDispatcherCreate(2);
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = PxDefaultSimulationFilterShader;
  sceneDesc.solverType = gSolverType;
  gScene = gPhysics->createScene(sceneDesc);

  PxPvdSceneClient *pvdClient = gScene->getScenePvdClient();
  if (pvdClient) {
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
    pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
  }
  gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

  PxRigidStatic *groundPlane =
      PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
  gScene->addActor(*groundPlane);

  for (PxU32 i = 0; i < 5; i++)
    createStack(PxTransform(PxVec3(0, 0, stackZ -= 10.0f)), 10, 2.0f);

  if (!interactive && !gHeadlessBallShot)
    createDynamic(PxTransform(PxVec3(0, 40, 100)), PxSphereGeometry(10),
                  PxVec3(0, -50, -100));

  printf("[HelloWorld] init solver=%s stacks=5 ground=plane headless=%s\n",
         solverName(gSolverType), gHeadlessMode ? "yes" : "no");
}

void stepPhysics(bool /*interactive*/) {
  if (gHeadlessBallShot && gSimFrame == gBallShotFrame)
    spawnBallShot();
  gScene->simulate(1.0f / 60.0f);
  gScene->fetchResults(true);
  if (gHeadlessMode)
    sampleDynamics(gSimFrame, 120);
  ++gSimFrame;
}

void cleanupPhysics(bool /*interactive*/) {
  PX_RELEASE(gScene);
  PX_RELEASE(gDispatcher);
  PX_RELEASE(gPhysics);
  if (gPvd) {
    PxPvdTransport *transport = gPvd->getTransport();
    PX_RELEASE(gPvd);
    PX_RELEASE(transport);
  }
  PX_RELEASE(gFoundation);
  printf("SnippetHelloWorld done.\n");
}

#ifdef RENDER_SNIPPET
void keyPress(unsigned char key, const PxTransform &camera) {
  switch (toupper(key)) {
  case 'B':
    createStack(PxTransform(PxVec3(0, 0, stackZ -= 10.0f)), 10, 2.0f);
    break;
  case ' ':
    createDynamic(camera, PxSphereGeometry(3.0f),
                  camera.rotate(PxVec3(0, 0, -1)) * 200);
    break;
  }
}
#endif

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  gSolverType = getRequestedSolverType(argc, argv);
  gHeadlessBallShot = hasArg(argc, argv, "--headless-ball-shot");
  gHeadlessMode = isHeadlessRequested(argc, argv);

  for (int i = 1; i < argc; ++i) {
    if (argv[i] && std::strncmp(argv[i], "--frames=", 9) == 0)
      gHeadlessFrameCount = PxU32(atoi(argv[i] + 9));
    if (argv[i] && std::strncmp(argv[i], "--ball-shot-frame=", 18) == 0)
      gBallShotFrame = PxU32(atoi(argv[i] + 18));
  }

#ifdef RENDER_SNIPPET
  if (!gHeadlessMode) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  gMetrics = HelloWorldMetrics();
  gPrevMaxBoxSpeed = 0.0f;

  initPhysics(true);
  for (PxU32 frame = 0; frame < gHeadlessFrameCount; ++frame) {
    PX_UNUSED(frame);
    stepPhysics(true);
  }
  printHeadlessSummary();
  cleanupPhysics(true);
  return 0;
}
