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

// This snippet illustrates the implementation and use of a pulley joint using
// PhysX's custom constraint framework.  Its headless case is also the canonical
// AVBD gate for generic PxConstraintSolverPrep / Px1DConstraint ingestion.

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"
#include "PulleyJoint.h"
#include "PxPhysicsAPI.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace physx;

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation *gFoundation = NULL;
static PxPhysics *gPhysics = NULL;
static PxDefaultCpuDispatcher *gDispatcher = NULL;
static PxScene *gScene = NULL;
static PxMaterial *gMaterial = NULL;
static PxPvd *gPvd = NULL;
static PxPvdTransport *gPvdTransport = NULL;
static PxRigidDynamic *gBox0 = NULL;
static PxRigidDynamic *gBox1 = NULL;
static PulleyJoint *gPulley = NULL;

static Snippets::HeadlessOptions gHeadlessOptions;
static PxReal gRatio = 1.0f;
static PxReal gTargetDistance = 8.0f;
static PxU32 gImpulseFrame = 20;
static PxVec3 gInitialPosition0(0.0f);
static PxVec3 gInitialPosition1(0.0f);

enum CustomCase {
  eCASE_IMPULSE,
  eCASE_MULTI_OUTPUT,
  eCASE_SPRING,
  eCASE_RESTITUTION,
  eCASE_DRIVE_LIMIT
};

static CustomCase gCustomCase = eCASE_IMPULSE;

static const char *getCustomCaseName() {
  switch (gCustomCase) {
  case eCASE_MULTI_OUTPUT:
    return "multi-output";
  case eCASE_SPRING:
    return "spring";
  case eCASE_RESTITUTION:
    return "restitution";
  case eCASE_DRIVE_LIMIT:
    return "drive-limit";
  case eCASE_IMPULSE:
  default:
    return "impulse";
  }
}

enum BreakMode {
  eBREAK_NONE,
  eBREAK_BELOW_REACTION,
  eBREAK_ABOVE_REACTION
};

static BreakMode gBreakMode = eBREAK_NONE;
static const PxReal gBreakTorqueBelowReaction = 3000.0f;
static const PxReal gBreakTorqueAboveReaction = 10000.0f;

static const char *getBreakModeName() {
  switch (gBreakMode) {
  case eBREAK_BELOW_REACTION:
    return "below";
  case eBREAK_ABOVE_REACTION:
    return "above";
  case eBREAK_NONE:
  default:
    return "none";
  }
}

static PxReal getBreakTorqueThreshold() {
  switch (gBreakMode) {
  case eBREAK_BELOW_REACTION:
    return gBreakTorqueBelowReaction;
  case eBREAK_ABOVE_REACTION:
    return gBreakTorqueAboveReaction;
  case eBREAK_NONE:
  default:
    return PX_MAX_F32;
  }
}

struct CustomJointMetrics {
  PxU32 completedFrames;
  PxU32 simulateCalls;
  PxU32 fetchCalls;
  PxU32 fetchFailures;
  PxU32 nonFinite;
  PxU32 impulseEvents;
  PxU32 responseSamples;
  PxU32 directionSamples;
  PxU32 directionViolations;
  PxU32 forceReads;
  PxU32 nonFiniteForceReads;
  PxU32 brokenReads;
  PxU32 firstBrokenFrame;
  PxReal maxRopeError;
  PxReal maxConstraintSpeedResidual;
  PxReal maxLinearForce;
  PxReal maxAngularForce;
  PxVec3 peakLinearForce;
  PxVec3 peakAngularForce;
  PxReal residualSquaredSum;
  PxReal maxAbsPosition;
  PxReal maxLinearSpeed;
  PxReal finalDisplacement0;
  PxReal finalDisplacement1;
  PxReal peakUpwardVelocity0;
  PxReal peakDownwardVelocity1;
  PxReal maxAbsLinearForceX;
  PxReal maxAbsLinearForceZ;
  PxReal maxAbsVelocity0X;
  PxReal maxAbsVelocity0Z;
  PxReal maxAbsRelativeVelocityX;
  PxReal maxAbsRelativeVelocityZ;
  PxReal firstPostSolveRelativeVelocityX;
  PxReal finalRelativeVelocityX;
  PxReal finalRelativeVelocityZ;

  CustomJointMetrics()
      : completedFrames(0), simulateCalls(0), fetchCalls(0), fetchFailures(0),
        nonFinite(0), impulseEvents(0), responseSamples(0),
        directionSamples(0), directionViolations(0), forceReads(0),
        nonFiniteForceReads(0), brokenReads(0),
        firstBrokenFrame(PX_MAX_U32), maxRopeError(0.0f),
        maxConstraintSpeedResidual(0.0f), maxLinearForce(0.0f),
        maxAngularForce(0.0f), peakLinearForce(0.0f),
        peakAngularForce(0.0f), residualSquaredSum(0.0f),
        maxAbsPosition(0.0f), maxLinearSpeed(0.0f),
        finalDisplacement0(0.0f), finalDisplacement1(0.0f),
        peakUpwardVelocity0(0.0f), peakDownwardVelocity1(0.0f),
        maxAbsLinearForceX(0.0f), maxAbsLinearForceZ(0.0f),
        maxAbsVelocity0X(0.0f), maxAbsVelocity0Z(0.0f),
        maxAbsRelativeVelocityX(0.0f), maxAbsRelativeVelocityZ(0.0f),
        firstPostSolveRelativeVelocityX(0.0f),
        finalRelativeVelocityX(0.0f), finalRelativeVelocityZ(0.0f) {}
};

static CustomJointMetrics gMetrics;

static bool isFinite(const PxVec3 &v) {
  return v.isFinite();
}

static PxReal maxAbsComponent(const PxVec3 &v) {
  return PxMax(PxAbs(v.x), PxMax(PxAbs(v.y), PxAbs(v.z)));
}

static bool parseHeadlessOptions(int argc, const char *const *argv,
                                 std::string &error) {
  Snippets::HeadlessOptions defaults;
  defaults.frames = 180;
  defaults.caseName = "impulse";
  defaults.solverType = PxSolverType::eAVBD;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults,
                                            gHeadlessOptions, error))
    return false;

  bool ratioSeen = false;
  bool impulseFrameSeen = false;
  bool breakModeSeen = false;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (Snippets::isCommonHeadlessOption(arg))
      continue;
    if (Snippets::hasOptionPrefix(arg, "--ratio=")) {
      if (ratioSeen ||
          !Snippets::parseReal(arg + std::strlen("--ratio="), 0.1f, 10.0f,
                               gRatio)) {
        error = ratioSeen ? "duplicate --ratio" : "invalid --ratio";
        return false;
      }
      ratioSeen = true;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--impulse-frame=")) {
      if (impulseFrameSeen ||
          !Snippets::parseU32(arg + std::strlen("--impulse-frame="), 1,
                              100000000u, gImpulseFrame)) {
        error = impulseFrameSeen ? "duplicate --impulse-frame"
                                 : "invalid --impulse-frame";
        return false;
      }
      impulseFrameSeen = true;
      continue;
    }
    if (Snippets::hasOptionPrefix(arg, "--break-mode=")) {
      if (breakModeSeen) {
        error = "duplicate --break-mode";
        return false;
      }
      const char *value = arg + std::strlen("--break-mode=");
      if (Snippets::equalsIgnoreCase(value, "none"))
        gBreakMode = eBREAK_NONE;
      else if (Snippets::equalsIgnoreCase(value, "below"))
        gBreakMode = eBREAK_BELOW_REACTION;
      else if (Snippets::equalsIgnoreCase(value, "above"))
        gBreakMode = eBREAK_ABOVE_REACTION;
      else {
        error = "invalid --break-mode (expected none, below, or above)";
        return false;
      }
      breakModeSeen = true;
      continue;
    }
    error = std::string("unknown option: ") + (arg ? arg : "<null>");
    return false;
  }

  if (Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(), "impulse"))
    gCustomCase = eCASE_IMPULSE;
  else if (Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(),
                                      "multi-output"))
    gCustomCase = eCASE_MULTI_OUTPUT;
  else if (Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(),
                                      "spring"))
    gCustomCase = eCASE_SPRING;
  else if (Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(),
                                      "restitution"))
    gCustomCase = eCASE_RESTITUTION;
  else if (Snippets::equalsIgnoreCase(gHeadlessOptions.caseName.c_str(),
                                      "drive-limit"))
    gCustomCase = eCASE_DRIVE_LIMIT;
  else {
    error = "unsupported --case";
    return false;
  }
  gHeadlessOptions.caseName = getCustomCaseName();
  if (gCustomCase != eCASE_IMPULSE && (ratioSeen || breakModeSeen)) {
    error = "--ratio and --break-mode require --case=impulse";
    return false;
  }
  if (gCustomCase != eCASE_RESTITUTION &&
      gCustomCase != eCASE_DRIVE_LIMIT &&
      gImpulseFrame + 20 >= gHeadlessOptions.frames) {
    error = "--impulse-frame must leave at least 20 response frames";
    return false;
  }
  return true;
}

static void recordState(PxU32 frame) {
  if (!gBox0 || !gBox1)
    return;

  const PxTransform pose0 = gBox0->getGlobalPose();
  const PxTransform pose1 = gBox1->getGlobalPose();
  const PxVec3 velocity0 = gBox0->getLinearVelocity();
  const PxVec3 velocity1 = gBox1->getLinearVelocity();
  const PxVec3 angular0 = gBox0->getAngularVelocity();
  const PxVec3 angular1 = gBox1->getAngularVelocity();

  if (!pose0.isFinite() || !pose1.isFinite() || !isFinite(velocity0) ||
      !isFinite(velocity1) || !isFinite(angular0) || !isFinite(angular1)) {
    ++gMetrics.nonFinite;
    return;
  }

  gMetrics.maxAbsPosition =
      PxMax(gMetrics.maxAbsPosition,
            PxMax(maxAbsComponent(pose0.p), maxAbsComponent(pose1.p)));
  gMetrics.maxLinearSpeed =
      PxMax(gMetrics.maxLinearSpeed,
            PxMax(velocity0.magnitude(), velocity1.magnitude()));

  const PxVec3 relativeVelocity = velocity0 - velocity1;
  gMetrics.maxAbsVelocity0X =
      PxMax(gMetrics.maxAbsVelocity0X, PxAbs(velocity0.x));
  gMetrics.maxAbsVelocity0Z =
      PxMax(gMetrics.maxAbsVelocity0Z, PxAbs(velocity0.z));
  gMetrics.maxAbsRelativeVelocityX =
      PxMax(gMetrics.maxAbsRelativeVelocityX, PxAbs(relativeVelocity.x));
  gMetrics.maxAbsRelativeVelocityZ =
      PxMax(gMetrics.maxAbsRelativeVelocityZ, PxAbs(relativeVelocity.z));
  gMetrics.finalRelativeVelocityX = relativeVelocity.x;
  gMetrics.finalRelativeVelocityZ = relativeVelocity.z;
  const PxU32 witnessFrame =
      gCustomCase == eCASE_RESTITUTION ? 0u : gImpulseFrame;
  if (frame == witnessFrame)
    gMetrics.firstPostSolveRelativeVelocityX = relativeVelocity.x;

  if (gCustomCase != eCASE_IMPULSE)
    return;

  const PxVec3 localAttachment(0.0f, 1.0f, 0.0f);
  const PxVec3 anchor0(5.0f, 10.0f, 0.0f);
  const PxVec3 anchor1(0.0f, 10.0f, 0.0f);
  const PxVec3 point0 = pose0.transform(localAttachment);
  const PxVec3 point1 = pose1.transform(localAttachment);
  PxVec3 direction0 = anchor0 - point0;
  PxVec3 direction1 = anchor1 - point1;
  const PxReal distance0 = direction0.normalize();
  const PxReal distance1 = direction1.normalize();
  // PulleyJoint::solverPrep applies ratio to body B's Jacobian but its authored
  // geometricError is distance - (distanceA + distanceB).  Preserve that
  // public sample semantics here: ratio lanes gate the velocity coupling,
  // while only ratio=1 has an integrable rope-length position oracle.
  const PxReal ropeError = gTargetDistance - (distance0 + distance1);
  gMetrics.maxRopeError = PxMax(gMetrics.maxRopeError, PxAbs(ropeError));

  if (frame > gImpulseFrame) {
    const PxVec3 r0 = point0 - pose0.p;
    const PxVec3 r1 = point1 - pose1.p;
    const PxVec3 pointVelocity0 = velocity0 + angular0.cross(r0);
    const PxVec3 pointVelocity1 = velocity1 + angular1.cross(r1);
    const PxReal residual = pointVelocity0.dot(direction0) +
                            gRatio * pointVelocity1.dot(direction1);
    gMetrics.maxConstraintSpeedResidual =
        PxMax(gMetrics.maxConstraintSpeedResidual, PxAbs(residual));
    gMetrics.residualSquaredSum += residual * residual;
    ++gMetrics.responseSamples;

    gMetrics.peakUpwardVelocity0 =
        PxMax(gMetrics.peakUpwardVelocity0, velocity0.y);
    gMetrics.peakDownwardVelocity1 =
        PxMin(gMetrics.peakDownwardVelocity1, velocity1.y);
    if (PxAbs(velocity1.y) > 0.05f) {
      ++gMetrics.directionSamples;
      if (!(velocity0.y > 0.0f && velocity1.y < 0.0f))
        ++gMetrics.directionViolations;
    }
  }

  gMetrics.finalDisplacement0 = pose0.p.y - gInitialPosition0.y;
  gMetrics.finalDisplacement1 = pose1.p.y - gInitialPosition1.y;
}

void initPhysics(bool interactive) {
  gErrorCallback.reset();
  gFoundation =
      PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
  if (!gFoundation)
    return;

  if (interactive) {
    gPvd = PxCreatePvd(*gFoundation);
    gPvdTransport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
    if (gPvd && gPvdTransport)
      gPvd->connect(*gPvdTransport, PxPvdInstrumentationFlag::eALL);
  }

  gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
                             PxTolerancesScale(), true, gPvd);
  if (!gPhysics)
    return;

  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = interactive ? PxVec3(0.0f, -9.81f, 0.0f) : PxVec3(0.0f);
  const PxU32 dispatcherThreads =
      interactive ? 2u : gHeadlessOptions.dispatcherThreads;
  gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = PxDefaultSimulationFilterShader;
  if (!interactive)
    sceneDesc.solverType = gHeadlessOptions.solverType;
  gScene = gPhysics->createScene(sceneDesc);
  if (!gScene)
    return;

  gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
  if (!gMaterial)
    return;

  if (interactive) {
    PxRigidStatic *groundPlane =
        PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
    gScene->addActor(*groundPlane);
  }

  const PxBoxGeometry boxGeom(1.0f, 1.0f, 1.0f);
  gBox0 = PxCreateDynamic(*gPhysics, PxTransform(PxVec3(5, 5, 0)), boxGeom,
                          *gMaterial, 1.0f);
  gBox1 = PxCreateDynamic(*gPhysics, PxTransform(PxVec3(0, 5, 0)), boxGeom,
                          *gMaterial, 2.0f);
  if (!gBox0 || !gBox1)
    return;

  gPulley = new PulleyJoint(
      *gPhysics, *gBox0, PxTransform(PxVec3(0.0f, 1.0f, 0.0f)),
      PxVec3(5.0f, 10.0f, 0.0f), *gBox1,
      PxTransform(PxVec3(0.0f, 1.0f, 0.0f)),
      PxVec3(0.0f, 10.0f, 0.0f));
  gPulley->setRatio(gRatio);
  gTargetDistance = 8.0f;
  gPulley->setDistance(gTargetDistance);
  PulleyJoint::HeadlessRowMode rowMode = PulleyJoint::ePULLEY_ROW;
  if (gCustomCase == eCASE_MULTI_OUTPUT)
    rowMode = PulleyJoint::eMULTI_OUTPUT_ROW;
  else if (gCustomCase == eCASE_SPRING)
    rowMode = PulleyJoint::eSPRING_ROW;
  else if (gCustomCase == eCASE_RESTITUTION)
    rowMode = PulleyJoint::eRESTITUTION_ROW;
  else if (gCustomCase == eCASE_DRIVE_LIMIT)
    rowMode = PulleyJoint::eDRIVE_LIMIT_ROW;
  gPulley->setHeadlessRowMode(rowMode);
  if (gCustomCase == eCASE_DRIVE_LIMIT)
    gPulley->mConstraint->setFlag(
        PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
  if (gBreakMode != eBREAK_NONE)
    gPulley->mConstraint->setBreakForce(PX_MAX_F32,
                                        getBreakTorqueThreshold());

  gScene->addActor(*gBox0);
  gScene->addActor(*gBox1);
  gInitialPosition0 = gBox0->getGlobalPose().p;
  gInitialPosition1 = gBox1->getGlobalPose().p;
  if (gCustomCase == eCASE_RESTITUTION) {
    gBox0->setSolverIterationCounts(16, 8);
    gBox1->setSolverIterationCounts(16, 8);
    gBox0->setLinearVelocity(PxVec3(-2.0f, 0.0f, 0.0f));
    gBox1->setLinearVelocity(PxVec3(2.0f, 0.0f, 0.0f));
  }
}

void stepPhysics(bool interactive) {
  const PxU32 frame = gMetrics.simulateCalls;
  if (!interactive && frame == gImpulseFrame && gBox1 &&
      (gCustomCase == eCASE_IMPULSE ||
       gCustomCase == eCASE_MULTI_OUTPUT ||
       gCustomCase == eCASE_SPRING)) {
    PxVec3 impulse(0.0f, -40.0f, 0.0f);
    if (gCustomCase == eCASE_MULTI_OUTPUT)
      impulse = PxVec3(30.0f, 0.0f, 45.0f);
    else if (gCustomCase == eCASE_SPRING)
      impulse = PxVec3(40.0f, 0.0f, 0.0f);
    gBox1->addForce(impulse, PxForceMode::eIMPULSE);
    ++gMetrics.impulseEvents;
  }

  ++gMetrics.simulateCalls;
  gScene->simulate(interactive ? 1.0f / 60.0f : gHeadlessOptions.dt);
  const bool fetched = gScene->fetchResults(true);
  ++gMetrics.fetchCalls;
  if (!fetched) {
    ++gMetrics.fetchFailures;
    return;
  }
  ++gMetrics.completedFrames;
  if (!interactive) {
    PxConstraint *constraint = gPulley ? gPulley->mConstraint : NULL;
    if (constraint) {
      PxVec3 linearForce(0.0f), angularForce(0.0f);
      constraint->getForce(linearForce, angularForce);
      ++gMetrics.forceReads;
      if (!linearForce.isFinite() || !angularForce.isFinite()) {
        ++gMetrics.nonFiniteForceReads;
        ++gMetrics.nonFinite;
      } else {
        const PxReal linearMagnitude = linearForce.magnitude();
        if (linearMagnitude > gMetrics.maxLinearForce) {
          gMetrics.maxLinearForce = linearMagnitude;
          gMetrics.peakLinearForce = linearForce;
          gMetrics.peakAngularForce = angularForce;
        }
        gMetrics.maxAngularForce =
            PxMax(gMetrics.maxAngularForce, angularForce.magnitude());
        gMetrics.maxAbsLinearForceX =
            PxMax(gMetrics.maxAbsLinearForceX, PxAbs(linearForce.x));
        gMetrics.maxAbsLinearForceZ =
            PxMax(gMetrics.maxAbsLinearForceZ, PxAbs(linearForce.z));
      }
      if (constraint->getFlags().isSet(PxConstraintFlag::eBROKEN)) {
        ++gMetrics.brokenReads;
        if (gMetrics.firstBrokenFrame == PX_MAX_U32)
          gMetrics.firstBrokenFrame = frame;
      }
    }
    recordState(frame);
    if (gCustomCase == eCASE_RESTITUTION && frame == 0 && constraint)
      constraint->setFlag(PxConstraintFlag::eDISABLE_CONSTRAINT, true);
  }
}

void cleanupPhysics(bool) {
  PX_RELEASE(gScene);
  gPulley = NULL;
  gBox0 = NULL;
  gBox1 = NULL;
  PX_RELEASE(gDispatcher);
  PX_RELEASE(gPhysics);
  if (gPvd) {
    PX_RELEASE(gPvd);
  }
  PX_RELEASE(gPvdTransport);
  PX_RELEASE(gFoundation);
  gMaterial = NULL;
  std::printf("SnippetCustomJoint done.\n");
}

static int runHeadless() {
  Snippets::printHeadlessConfig("SnippetCustomJoint", gHeadlessOptions);
  initPhysics(false);
  const bool initialized =
      gFoundation && gPhysics && gDispatcher && gScene && gMaterial && gBox0 &&
      gBox1 && gPulley;

  if (initialized) {
    for (PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame) {
      stepPhysics(false);
      if (gMetrics.fetchFailures || gMetrics.nonFinite)
        break;
    }
  }

  const PxReal residualRms =
      gMetrics.responseSamples
          ? PxSqrt(gMetrics.residualSquaredSum /
                   static_cast<PxReal>(gMetrics.responseSamples))
          : PX_MAX_F32;

  const char *reason = "none";
  bool passed = true;
  const PxReal signedTorqueToLinearRatio =
      PxAbs(gMetrics.peakLinearForce.y) > 1.0e-6f
          ? gMetrics.peakAngularForce.z / gMetrics.peakLinearForce.y
          : 0.0f;
  const bool expectBreak = gBreakMode == eBREAK_BELOW_REACTION;
  if (!initialized) {
    passed = false;
    reason = "initialization_failed";
  } else if (gMetrics.completedFrames != gHeadlessOptions.frames ||
             gMetrics.simulateCalls != gHeadlessOptions.frames ||
             gMetrics.fetchCalls != gHeadlessOptions.frames ||
             gMetrics.fetchFailures != 0) {
    passed = false;
    reason = "incomplete_simulation";
  } else if (gErrorCallback.getFatalCount() != 0 || gMetrics.nonFinite != 0) {
    passed = false;
    reason = "runtime_error";
  } else if (gMetrics.forceReads != gHeadlessOptions.frames ||
             gMetrics.nonFiniteForceReads != 0) {
    passed = false;
    reason = "force_writeback_accounting";
  } else if (gCustomCase == eCASE_MULTI_OUTPUT &&
             (gMetrics.impulseEvents != 1 ||
              gMetrics.maxAbsVelocity0X < 0.1f ||
              gMetrics.maxAbsVelocity0Z < 0.1f ||
              gMetrics.maxAbsLinearForceX < 1.0f ||
              gMetrics.maxAbsLinearForceZ < 1.0f)) {
    passed = false;
    reason = "multi_row_not_consumed";
  } else if (gCustomCase == eCASE_SPRING &&
             (gMetrics.impulseEvents != 1 ||
              gMetrics.maxAbsVelocity0X < 0.05f ||
              gMetrics.maxAbsLinearForceX < 1.0f ||
              PxAbs(gMetrics.finalRelativeVelocityX) > 1.0f)) {
    passed = false;
    reason = "spring_row_not_consumed";
  } else if (gCustomCase == eCASE_RESTITUTION &&
             (gMetrics.impulseEvents != 0 ||
              gMetrics.firstPostSolveRelativeVelocityX < 2.0f ||
              gMetrics.maxAbsLinearForceX < 1.0f)) {
    passed = false;
    reason = "restitution_row_not_consumed";
  } else if (gCustomCase == eCASE_DRIVE_LIMIT &&
             (gMetrics.impulseEvents != 0 ||
              gMetrics.finalRelativeVelocityX < 0.5f ||
              gMetrics.maxAbsLinearForceX < 0.1f ||
              gMetrics.maxAbsLinearForceX > 12.5f)) {
    passed = false;
    reason = "drive_limit_row_not_consumed";
  } else if (gCustomCase == eCASE_IMPULSE &&
             (gMetrics.impulseEvents != 1 ||
             gMetrics.responseSamples < gHeadlessOptions.frames -
                                            gImpulseFrame - 2)) {
    passed = false;
    reason = "missing_impulse_or_samples";
  } else if (gCustomCase == eCASE_IMPULSE &&
             gMetrics.maxLinearForce < 100.0f) {
    passed = false;
    reason = "missing_output_force";
  } else if (gCustomCase == eCASE_IMPULSE &&
             (gMetrics.maxAngularForce < 500.0f ||
             gMetrics.maxLinearForce <= 0.0f ||
             gMetrics.maxAngularForce / gMetrics.maxLinearForce < 4.9f ||
             gMetrics.maxAngularForce / gMetrics.maxLinearForce > 5.1f ||
             signedTorqueToLinearRatio < 4.9f ||
             signedTorqueToLinearRatio > 5.1f)) {
    passed = false;
    reason = "missing_output_torque";
  } else if (gCustomCase == eCASE_IMPULSE && expectBreak &&
             gMetrics.firstBrokenFrame == PX_MAX_U32) {
    passed = false;
    reason = "missing_angular_break";
  } else if (gCustomCase == eCASE_IMPULSE && expectBreak &&
             (gMetrics.firstBrokenFrame < gImpulseFrame ||
              gMetrics.firstBrokenFrame > gImpulseFrame + 2)) {
    passed = false;
    reason = "wrong_break_timing";
  } else if (gCustomCase == eCASE_IMPULSE && !expectBreak &&
             gMetrics.firstBrokenFrame != PX_MAX_U32) {
    passed = false;
    reason = "unexpected_break";
  } else if (gCustomCase == eCASE_IMPULSE && !expectBreak &&
             (gMetrics.peakUpwardVelocity0 < 0.25f ||
             gMetrics.finalDisplacement0 < 0.05f ||
              gMetrics.peakDownwardVelocity1 > -0.25f)) {
    passed = false;
    reason = "missing_coupled_response";
  } else if (gCustomCase == eCASE_IMPULSE && !expectBreak &&
             (gMetrics.directionSamples < 10 ||
              gMetrics.directionViolations * 10 >
                  gMetrics.directionSamples)) {
    passed = false;
    reason = "wrong_response_direction";
  } else if (gCustomCase == eCASE_IMPULSE && !expectBreak &&
             ((PxAbs(gRatio - 1.0f) < 1e-5f &&
               gMetrics.maxRopeError > 0.5f) ||
              residualRms > 0.5f * (1.0f + gRatio))) {
    passed = false;
    reason = "pulley_constraint_not_enforced";
  } else if (gMetrics.maxAbsPosition > 1000.0f ||
             gMetrics.maxLinearSpeed > 200.0f) {
    passed = false;
    reason = "runaway";
  }

  cleanupPhysics(false);
  std::printf(
      "[AVBD_GATE] schema=1 snippet=SnippetCustomJoint solver=%s case=%s "
      "execution=%s frames=%u completedFrames=%u status=%s reason=%s "
      "capability=%s validation=GATED ratio=%.9g targetDistance=%.9g "
      "impulseEvents=%u responseSamples=%u directionSamples=%u "
      "directionViolations=%u finalDisplacement0=%.9g finalDisplacement1=%.9g "
      "peakUpwardVelocity0=%.9g peakDownwardVelocity1=%.9g "
      "maxRopeError=%.9g residualRms=%.9g maxConstraintSpeedResidual=%.9g "
      "forceReads=%u nonFiniteForceReads=%u maxLinearForce=%.9g "
      "maxAngularForce=%.9g outputTorqueToLinearRatio=%.9g "
      "peakLinearForceX=%.9g peakLinearForceY=%.9g peakLinearForceZ=%.9g "
      "peakAngularForceX=%.9g peakAngularForceY=%.9g "
      "peakAngularForceZ=%.9g signedTorqueToLinearYRatio=%.9g "
      "maxAbsLinearForceX=%.9g maxAbsLinearForceZ=%.9g "
      "maxAbsVelocity0X=%.9g maxAbsVelocity0Z=%.9g "
      "maxAbsRelativeVelocityX=%.9g maxAbsRelativeVelocityZ=%.9g "
      "firstPostSolveRelativeVelocityX=%.9g "
      "finalRelativeVelocityX=%.9g finalRelativeVelocityZ=%.9g "
      "outputForceMinimum=100 outputTorqueMinimum=500 "
      "outputTorqueRatioMinimum=4.9 outputTorqueRatioMaximum=5.1 "
      "breakMode=%s angularBreakForce=%.9g brokenReads=%u "
      "firstBrokenFrame=%u breakFrameMaximumOffset=2 "
      "nonFinite=%u fetchFailures=%u fatalErrors=%u\n",
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      getCustomCaseName(),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gMetrics.completedFrames,
      passed ? "PASS" : "FAIL", reason, passed ? "SUPPORTED" : "UNSUPPORTED",
      double(gRatio), double(gTargetDistance), gMetrics.impulseEvents,
      gMetrics.responseSamples, gMetrics.directionSamples,
      gMetrics.directionViolations, double(gMetrics.finalDisplacement0),
      double(gMetrics.finalDisplacement1),
      double(gMetrics.peakUpwardVelocity0),
      double(gMetrics.peakDownwardVelocity1), double(gMetrics.maxRopeError),
      double(residualRms), double(gMetrics.maxConstraintSpeedResidual),
      gMetrics.forceReads, gMetrics.nonFiniteForceReads,
      double(gMetrics.maxLinearForce), double(gMetrics.maxAngularForce),
      double(gMetrics.maxLinearForce > 0.0f
                 ? gMetrics.maxAngularForce / gMetrics.maxLinearForce
                 : 0.0f),
      double(gMetrics.peakLinearForce.x),
      double(gMetrics.peakLinearForce.y),
      double(gMetrics.peakLinearForce.z),
      double(gMetrics.peakAngularForce.x),
      double(gMetrics.peakAngularForce.y),
      double(gMetrics.peakAngularForce.z),
      double(signedTorqueToLinearRatio),
      double(gMetrics.maxAbsLinearForceX),
      double(gMetrics.maxAbsLinearForceZ),
      double(gMetrics.maxAbsVelocity0X),
      double(gMetrics.maxAbsVelocity0Z),
      double(gMetrics.maxAbsRelativeVelocityX),
      double(gMetrics.maxAbsRelativeVelocityZ),
      double(gMetrics.firstPostSolveRelativeVelocityX),
      double(gMetrics.finalRelativeVelocityX),
      double(gMetrics.finalRelativeVelocityZ),
      getBreakModeName(), double(getBreakTorqueThreshold()),
      gMetrics.brokenReads, gMetrics.firstBrokenFrame,
      gMetrics.nonFinite, gMetrics.fetchFailures,
      gErrorCallback.getFatalCount());
  return passed ? Snippets::eHEADLESS_PASS
                : Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char *const *argv) {
  std::string error;
  if (!parseHeadlessOptions(argc, argv, error)) {
    std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomJoint "
                         "reason=%s\n",
                 error.c_str());
    return Snippets::eHEADLESS_CONFIG_ERROR;
  }
  if (!Snippets::applyExecutionEnvironment(gHeadlessOptions)) {
    std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomJoint "
                         "reason=execution_environment_failed\n");
    return Snippets::eHEADLESS_CONFIG_ERROR;
  }

  if (gHeadlessOptions.headless)
    return runHeadless();

#ifdef RENDER_SNIPPET
  extern void renderLoop();
  renderLoop();
#else
  initPhysics(false);
  for (PxU32 i = 0; i < 100; ++i)
    stepPhysics(false);
  cleanupPhysics(false);
#endif
  return 0;
}
