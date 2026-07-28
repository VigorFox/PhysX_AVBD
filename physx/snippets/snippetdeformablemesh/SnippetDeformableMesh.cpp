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

// ****************************************************************************
// This snippet shows how to use deformable meshes in PhysX.
//
// AVBD vs TGS on this scene:
//   - Sphere shot and settle speed: aligned with TGS (headless sphere-shot gate).
//   - --headless-stress: flat grid of boxes + periodic sphere shots on wavy mesh.
//   - AVBD publishes the mesh grid each substep so NP deformable rows receive
//     coherent surface normal/history. Synthesized box-corner shell replacement
//     is retired: retained NP rows own the contact (P3F/P3G).
//   - Box stack on a heaving mesh: AVBD may spread wider than TGS over long runs.
//     This is a known limitation of the current position-based AVBD penalty
//     contact model on fast-moving geometry, not a snippet bug.
// ****************************************************************************

#include <ctype.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "PxPhysicsAPI.h"
#include "PxAvbdKinematicShell.h"
#include "../snippetcommon/SnippetHeadless.h"

#ifdef RENDER_SNIPPET
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetrender/SnippetRender.h"
#endif

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;
static PxTriangleMesh*			gMesh		= NULL;
static PxRigidStatic*			gActor		= NULL;
static Snippets::HeadlessOptions gHeadlessOptions;
static bool gExtensionsInitialized = false;
static bool gInitializationFailed = false;
static bool gFetchPending = false;
static bool gCleanupCompleted = false;

static const PxU32 gGridSize = 8;
static const PxReal gGridStep = 512.0f / PxReal(gGridSize-1);
static const PxReal gGridMinimum = -400.0f;
static const PxReal gGridMaximum =
    gGridMinimum + gGridStep * PxReal(gGridSize - 1);
static float gTime = 0.0f;

static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;
static bool gHeadlessMode = false;
static bool gHeadlessSphereShot = false;
static bool gHeadlessStress = false;
static bool gHeadlessOwnershipProbe = false;
static bool gOwnershipProbeHeavy = false;
static bool gShellPostOwnershipProbe = false;
static bool gSurfaceHistoryProbe = false;
static bool gNormalPostOwnershipProbe = false;
static bool gBroadAuthorityProbe = false;
static bool gCreateStack = true;
static PxU32 gHeadlessFrameCount = 180;
static PxU32 gSimFrame = 0;
static PxU32 gFastImpactSubstepFrames = 0;
static const PxU32 gDeformSubsteps = 8;
static const PxReal gFastImpactSpeedThreshold = 80.0f;
static const PxU32 gFastImpactSubstepHoldFrames = 45;

static PxRigidDynamic *gShotSphere = NULL;
static PxRigidDynamic *gOwnershipProbeBox = NULL;
static const PxReal gShotSphereRadius = 3.0f;
static const PxReal gShotSpawnY = 55.0f;
static const PxReal gShotSpeedY = 200.0f;
static const PxReal gMeshRestOffset = -0.5f;

enum DeformableFilterTag {
  eFILTER_UNTAGGED = 0,
  eFILTER_MOVING_MESH = 1,
  eFILTER_BOX = 2,
  eFILTER_SPHERE_SHOT = 3,
  eFILTER_STRESS_SHOT = 4,
  eFILTER_OWNERSHIP_PROBE = 5
};

enum DeformableHeadlessCase {
  eCASE_MOVING_MESH_STACK,
  eCASE_SPHERE_SHOT,
  eCASE_STRESS_DIAGNOSTIC,
  eCASE_SURFACE_OWNER_LIGHT,
  eCASE_SURFACE_OWNER_HEAVY,
  eCASE_SHELL_POST_OWNERSHIP,
  eCASE_SURFACE_HISTORY,
  eCASE_NORMAL_POST_OWNERSHIP,
  eCASE_BROAD_COMPONENT_AUTHORITY
};

static DeformableHeadlessCase gHeadlessCase = eCASE_MOVING_MESH_STACK;
static std::vector<PxRigidDynamic *> gStressShots;

struct RuntimeMetrics {
  PxU32 completedFrames;
  PxU32 simulateFailures;
  PxU32 fetchFailures;
  PxU32 fetchErrorState;
  PxU32 nonFinite;
  PxU32 maxFullFallThroughBodies;
  PxReal maxQuaternionNormError;
  PxReal maxAbsPosition;
  PxReal maxLinearSpeed;
  PxReal maxAngularSpeed;

  RuntimeMetrics()
      : completedFrames(0), simulateFailures(0), fetchFailures(0),
        fetchErrorState(0), nonFinite(0), maxFullFallThroughBodies(0),
        maxQuaternionNormError(0.0f), maxAbsPosition(0.0f),
        maxLinearSpeed(0.0f), maxAngularSpeed(0.0f) {}
};
static RuntimeMetrics gRuntimeMetrics;

struct SphereShotMetrics {
  PxU32 contactEvents;
  PxU32 contactPoints;
  PxU32 firstContactFrame;
  PxU32 lastContactEventFrame;
  PxU32 responseSamples;
  PxU32 firstOverlapFrame;
  PxU32 maxOverlapFrame;
  PxU32 maxRestOffsetProximityFrame;
  PxU32 proximityFrames;
  PxU32 deepProximityFrames;
  PxU32 settledProximityFrames;
  PxReal maxRestOffsetProximity;
  PxReal maxImpactRestOffsetProximity;
  PxReal maxSettledRestOffsetProximity;
  PxReal maxGeomOverlap;
  PxReal maxImpactGeomOverlap;
  PxReal maxSettledGeomOverlap;
  PxReal minSphereY;
  PxReal maxAbsSphereVel;
  PxReal maxAirborneGap;
  PxReal maxSettledAirborneGap;
  // Mesh-ride / roll diagnostics (settled window + whole run)
  PxReal maxHorizVel;
  PxReal maxSettledHorizVel;
  PxReal maxAngVel;
  PxReal maxSettledAngVel;
  PxReal maxMeshAbsDyDt;       // |d(surfaceY)/dt| at sphere xz
  PxReal maxMeshAbsSlope;      // approximate surface slope magnitude
  PxReal sumSettledHorizVel;
  PxReal sumSettledMeshAbsDyDt;
  PxReal maxContactImpulse;
  PxReal maxImpactAxisVelocityDelta;
  PxU32 settledRideSamples;
  PxU32 outOfFootprintFrames;
  bool nanDetected;
};
static SphereShotMetrics gSphereShotMetrics;
static PxReal gPrevSampleSurfaceY = 0.0f;
static bool gHavePrevSampleSurfaceY = false;
static PxVec3 gPreviousSphereVelocity(0.0f);
static PxVec3 gSphereContactBaselineVelocity(0.0f);
static bool gHavePreviousSphereVelocity = false;
static bool gHaveSphereContactBaseline = false;
static const PxU32 gSphereResponseWindowFrames = 3;
static const PxReal gMinSphereResponseFraction = 0.05f;

struct StackHeadlessMetrics {
  PxReal maxSpeed;
  PxReal maxSettledSpeed;
  PxReal maxSpreadXZ;         // farthest box from origin in XZ (whole run)
  PxReal maxSettledSpreadXZ;  // same, last 60 frames
  PxReal maxWorldY;           // highest box center (whole run)
  PxReal maxSettledWorldY;
  PxReal minRelToSurface;     // most-submerged box center vs mesh surface (<0 sunk)
  PxU32 settledSunkBoxes;     // box center clearly below mesh surface (fell through)
  PxU32 maxOutOfFootprintBoxes;
  PxU32 nanBodies;
};
static StackHeadlessMetrics gStackMetrics;
static const PxReal gStackHalfExtent = 2.0f;

// --headless-stress: flat grid of boxes on wavy mesh + periodic sphere shots.
static const PxU32 gStressGridX = 6;
static const PxU32 gStressGridZ = 6;
static const PxU32 gStressShotIntervalFrames = 24;
static const PxReal gStressShotSpawnY = 55.0f;
static const PxReal gStressShotSpeedY = 200.0f;
static PxU32 gStressNextShotFrame = 0;
static PxU32 gStressShotSerial = 0;
static PxU32 gStressActiveShots = 0;

struct StressHeadlessMetrics {
  PxU32 contactEvents;
  PxU32 contactPoints;
  PxU32 worstFrame;
  PxReal worstMinBodyY;
  PxReal worstMinRelToSurface;
  PxU32 maxSunkBoxes;
  PxU32 maxPassThroughShots;
  PxU32 maxAirborneShots;
  PxU32 maxOutOfFootprintShots;
  PxU32 maxOutOfFootprintBoxes;
  PxU32 totalShotsFired;
  PxU32 nanEvents;
};
static StressHeadlessMetrics gStressMetrics;

static const PxReal gOwnershipProbeLightMass = 39.0f;
static const PxReal gOwnershipProbeHeavyMass = 41.0f;
static const PxReal gShellPostProbeMass = 10.0f;
static const PxReal gSurfaceHistoryProbeMass = 10.0f;
static const PxU32 gSurfaceHistoryWarmupFrames = 60;
static const PxU32 gSurfaceHistoryMotionFrames = 60;
static const PxReal gSurfaceHistorySpeed = 1.0f;
static PxReal gSurfaceHistoryOffset = 0.0f;
struct OwnershipProbeMetrics {
  PxVec3 initialPosition;
  PxVec3 finalPosition;
  PxReal maxHorizontalSpeed;
  PxReal maxAngularSpeed;
  PxReal minBottomGap;
  PxReal maxBottomGap;
  PxU32 outOfFootprintFrames;

  OwnershipProbeMetrics()
      : initialPosition(0.0f), finalPosition(0.0f),
        maxHorizontalSpeed(0.0f), maxAngularSpeed(0.0f),
        minBottomGap(PX_MAX_F32), maxBottomGap(-PX_MAX_F32),
        outOfFootprintFrames(0) {}
};
static OwnershipProbeMetrics gOwnershipProbeMetrics;

struct SurfaceHistoryMetrics {
  PxU32 contactEvents;
  PxU32 contactPoints;
  PxU32 motionSamples;
  PxReal surfaceYAtMotionStart;
  PxReal surfaceYAtMotionEnd;
  PxReal bodyYAtMotionStart;
  PxReal bodyYAtMotionEnd;
  PxReal sumBodyVelocityY;
  PxReal sumAbsRelativeVelocityY;
  PxReal minBodyVelocityY;
  PxReal maxBodyVelocityY;
  PxReal maxAbsRelativeVelocityY;
  bool motionStartCaptured;

  SurfaceHistoryMetrics()
      : contactEvents(0), contactPoints(0), motionSamples(0),
        surfaceYAtMotionStart(0.0f), surfaceYAtMotionEnd(0.0f),
        bodyYAtMotionStart(0.0f), bodyYAtMotionEnd(0.0f),
        sumBodyVelocityY(0.0f), sumAbsRelativeVelocityY(0.0f),
        minBodyVelocityY(PX_MAX_F32), maxBodyVelocityY(-PX_MAX_F32),
        maxAbsRelativeVelocityY(0.0f), motionStartCaptured(false) {}
};
static SurfaceHistoryMetrics gSurfaceHistoryMetrics;

static void setShapeTag(PxShape &shape, DeformableFilterTag tag) {
  // The default interactive filter shader interprets word0 as a collision
  // group.  Gate tags are needed only by the headless custom shader.
  if (!gHeadlessMode)
    return;
  PxFilterData data;
  data.word0 = static_cast<PxU32>(tag);
  shape.setSimulationFilterData(data);
}

static DeformableFilterTag getShapeTag(const PxShape *shape) {
  return shape ? static_cast<DeformableFilterTag>(
                     shape->getSimulationFilterData().word0)
               : eFILTER_UNTAGGED;
}

static DeformableFilterTag getActorTag(PxRigidActor *actor) {
  if (!actor || actor->getNbShapes() == 0)
    return eFILTER_UNTAGGED;
  PxShape *shape = NULL;
  actor->getShapes(&shape, 1);
  return getShapeTag(shape);
}

static bool isShotMeshPair(const PxShape *shape0, const PxShape *shape1,
                           DeformableFilterTag shotTag) {
  const DeformableFilterTag tag0 = getShapeTag(shape0);
  const DeformableFilterTag tag1 = getShapeTag(shape1);
  return (tag0 == shotTag && tag1 == eFILTER_MOVING_MESH) ||
         (tag1 == shotTag && tag0 == eFILTER_MOVING_MESH);
}

class DeformableSimulationCallback : public PxSimulationEventCallback {
public:
  virtual void onConstraintBreak(PxConstraintInfo *, PxU32) PX_OVERRIDE {}
  virtual void onWake(PxActor **, PxU32) PX_OVERRIDE {}
  virtual void onSleep(PxActor **, PxU32) PX_OVERRIDE {}
  virtual void onTrigger(PxTriggerPair *, PxU32) PX_OVERRIDE {}
  virtual void onAdvance(const PxRigidBody *const *, const PxTransform *,
                         const PxU32) PX_OVERRIDE {}

  virtual void onContact(const PxContactPairHeader &pairHeader,
                         const PxContactPair *pairs,
                         PxU32 nbPairs) PX_OVERRIDE {
    if (pairHeader.flags &
        (PxContactPairHeaderFlag::eREMOVED_ACTOR_0 |
         PxContactPairHeaderFlag::eREMOVED_ACTOR_1))
      return;
    const bool sphereActorPairMatches =
        (pairHeader.actors[0] == gShotSphere &&
         pairHeader.actors[1] == gActor) ||
        (pairHeader.actors[1] == gShotSphere &&
         pairHeader.actors[0] == gActor);
    const bool containsMeshActor = pairHeader.actors[0] == gActor ||
                                   pairHeader.actors[1] == gActor;
    const bool historyActorPairMatches =
        gSurfaceHistoryProbe &&
        ((pairHeader.actors[0] == gOwnershipProbeBox &&
          pairHeader.actors[1] == gActor) ||
         (pairHeader.actors[1] == gOwnershipProbeBox &&
          pairHeader.actors[0] == gActor));
    for (PxU32 i = 0; i < nbPairs; ++i) {
      const PxContactPair &pair = pairs[i];
      if (pair.flags & (PxContactPairFlag::eREMOVED_SHAPE_0 |
                        PxContactPairFlag::eREMOVED_SHAPE_1))
        continue;
      const bool historyPair =
          historyActorPairMatches &&
          isShotMeshPair(pair.shapes[0], pair.shapes[1],
                         eFILTER_OWNERSHIP_PROBE);
      const bool historyEvent =
          historyPair &&
          (pair.events.isSet(PxPairFlag::eNOTIFY_TOUCH_FOUND) ||
           pair.events.isSet(PxPairFlag::eNOTIFY_TOUCH_PERSISTS));
      if (!pair.events.isSet(PxPairFlag::eNOTIFY_TOUCH_FOUND) &&
          !historyEvent)
        continue;
      const bool spherePair =
          sphereActorPairMatches &&
          isShotMeshPair(pair.shapes[0], pair.shapes[1],
                         eFILTER_SPHERE_SHOT);
      const bool stressPair =
          containsMeshActor &&
          isShotMeshPair(pair.shapes[0], pair.shapes[1],
                         eFILTER_STRESS_SHOT);
      if (!spherePair && !stressPair && !historyPair)
        continue;

      PxContactPairPoint points[32];
      const PxU32 pointCount = pair.extractContacts(points, 32);
      if (historyPair) {
        gSurfaceHistoryMetrics.contactEvents++;
        for (PxU32 p = 0; p < pointCount; ++p) {
          if (points[p].position.isFinite() && points[p].normal.isFinite() &&
              PxIsFinite(points[p].separation) &&
              points[p].impulse.isFinite())
            gSurfaceHistoryMetrics.contactPoints++;
          else
            gRuntimeMetrics.nonFinite++;
        }
        continue;
      }
      if (stressPair) {
        gStressMetrics.contactEvents++;
        for (PxU32 p = 0; p < pointCount; ++p) {
          if (!points[p].position.isFinite() ||
              !points[p].normal.isFinite() ||
              !PxIsFinite(points[p].separation) ||
              !points[p].impulse.isFinite() ||
              !PxIsFinite(points[p].impulse.magnitude())) {
            gStressMetrics.nanEvents++;
            continue;
          }
          gStressMetrics.contactPoints++;
        }
        continue;
      }

      if (gSphereShotMetrics.lastContactEventFrame == gSimFrame)
        continue;
      gSphereShotMetrics.lastContactEventFrame = gSimFrame;
      gSphereShotMetrics.contactEvents++;
      if (gSphereShotMetrics.firstContactFrame == PX_MAX_U32)
        gSphereShotMetrics.firstContactFrame = gSimFrame + 1;
      if (!gHaveSphereContactBaseline && gHavePreviousSphereVelocity) {
        gSphereContactBaselineVelocity = gPreviousSphereVelocity;
        gHaveSphereContactBaseline = true;
      }
      for (PxU32 p = 0; p < pointCount; ++p) {
        if (!points[p].position.isFinite() || !points[p].normal.isFinite() ||
            !PxIsFinite(points[p].separation) ||
            !points[p].impulse.isFinite()) {
          gSphereShotMetrics.nanDetected = true;
          continue;
        }
        const PxReal impulse = points[p].impulse.magnitude();
        if (!PxIsFinite(impulse)) {
          gSphereShotMetrics.nanDetected = true;
          continue;
        }
        gSphereShotMetrics.contactPoints++;
        gSphereShotMetrics.maxContactImpulse =
            PxMax(gSphereShotMetrics.maxContactImpulse, impulse);
      }
    }
  }
};
static DeformableSimulationCallback gSimulationCallback;

static PxFilterFlags deformableGateFilterShader(
    PxFilterObjectAttributes, PxFilterData filterData0,
    PxFilterObjectAttributes, PxFilterData filterData1, PxPairFlags &pairFlags,
    const void *, PxU32) {
  pairFlags = PxPairFlag::eCONTACT_DEFAULT;
  const bool sphereMesh =
      (filterData0.word0 == eFILTER_SPHERE_SHOT &&
       filterData1.word0 == eFILTER_MOVING_MESH) ||
      (filterData1.word0 == eFILTER_SPHERE_SHOT &&
       filterData0.word0 == eFILTER_MOVING_MESH);
  const bool stressMesh =
      (filterData0.word0 == eFILTER_STRESS_SHOT &&
       filterData1.word0 == eFILTER_MOVING_MESH) ||
      (filterData1.word0 == eFILTER_STRESS_SHOT &&
       filterData0.word0 == eFILTER_MOVING_MESH);
  const bool historyMesh =
      gSurfaceHistoryProbe &&
      ((filterData0.word0 == eFILTER_OWNERSHIP_PROBE &&
        filterData1.word0 == eFILTER_MOVING_MESH) ||
       (filterData1.word0 == eFILTER_OWNERSHIP_PROBE &&
        filterData0.word0 == eFILTER_MOVING_MESH));
  if (sphereMesh || stressMesh)
    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
  if (historyMesh)
    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
  return PxFilterFlag::eDEFAULT;
}

static PxVec3 getMeshLocalPoint(const PxVec3 &worldPoint) {
  return gActor ? gActor->getGlobalPose().transformInv(worldPoint)
                : worldPoint;
}

static bool isInsideMeshFootprint(const PxVec3 &worldPoint,
                                  PxReal inset = 0.0f) {
  const PxVec3 localPoint = getMeshLocalPoint(worldPoint);
  return localPoint.x >= gGridMinimum + inset &&
         localPoint.x <= gGridMaximum - inset &&
         localPoint.z >= gGridMinimum + inset &&
         localPoint.z <= gGridMaximum - inset;
}

static PxReal sampleMeshSurfaceY(const PxVec3 &p) {
  if (!gMesh)
    return 0.0f;
  const PxVec3 *verts = gMesh->getVertices();
  const PxVec3 localPoint = getMeshLocalPoint(p);
  const PxReal gx =
      PxClamp((localPoint.x - gGridMinimum) / gGridStep, 0.0f,
              PxReal(gGridSize - 1));
  const PxReal gz =
      PxClamp((localPoint.z - gGridMinimum) / gGridStep, 0.0f,
              PxReal(gGridSize - 1));
  const PxU32 b0 = PxMin(PxU32(gx), gGridSize - 2);
  const PxU32 a0 = PxMin(PxU32(gz), gGridSize - 2);
  const PxReal tx = gx - PxReal(b0);
  const PxReal tz = gz - PxReal(a0);
  const auto vert = [&](PxU32 a, PxU32 b) -> PxVec3 {
    return verts[a * gGridSize + b];
  };
  const PxReal y00 = vert(a0, b0).y;
  const PxReal y10 = vert(a0, b0 + 1).y;
  const PxReal y01 = vert(a0 + 1, b0).y;
  const PxReal y11 = vert(a0 + 1, b0 + 1).y;
  // Match createMeshGround() exactly: the cell diagonal is v00-v11.
  // tri0=(v10,v00,v11) covers tx>=tz; tri1=(v11,v00,v01) covers tx<tz.
  const PxReal localY =
      tx >= tz
          ? y00 * (1.0f - tx) + y10 * (tx - tz) + y11 * tz
          : y00 * (1.0f - tz) + y11 * tx + y01 * (tz - tx);
  if (gActor)
    return gActor->getGlobalPose()
        .transform(PxVec3(localPoint.x, localY, localPoint.z))
        .y;
  return localY + 2.0f;
}

static bool measureSphereMeshRestOffsetProximity(
    const PxVec3 &sphereCenter, PxReal radius, PxReal &outProximity) {
  outProximity = 0.0f;
  if (!gScene || !gMesh)
    return false;
  const PxReal surfaceY = sampleMeshSurfaceY(sphereCenter);
  // This is a vertical proximity proxy around the negative rest-offset
  // contact band.  It is not a geometric sphere/triangle penetration depth.
  outProximity =
      surfaceY - (sphereCenter.y - radius) - gMeshRestOffset;
  return true;
}

static void resetStressHeadlessMetrics() {
  gStressMetrics = StressHeadlessMetrics();
  gStressMetrics.worstMinRelToSurface = 1e9f;
  gStressMetrics.worstMinBodyY = 1e9f;
  gStressNextShotFrame = 0;
  gStressShotSerial = 0;
  gStressActiveShots = 0;
}

static void resetOwnershipProbeMetrics() {
  gOwnershipProbeMetrics = OwnershipProbeMetrics();
}

static PxReal getOwnershipProbeMass() {
  return (gSurfaceHistoryProbe || gNormalPostOwnershipProbe)
             ? gSurfaceHistoryProbeMass
             : gShellPostOwnershipProbe
             ? gShellPostProbeMass
             : (gOwnershipProbeHeavy ? gOwnershipProbeHeavyMass
                                     : gOwnershipProbeLightMass);
}

static bool measureBoxBottomRelToSurface(const PxTransform &pose,
                                         PxReal &relativeHeight);

static void updateOwnershipProbeMetrics() {
  if (!gHeadlessOwnershipProbe || !gOwnershipProbeBox)
    return;
  const PxTransform pose = gOwnershipProbeBox->getGlobalPose();
  const PxVec3 linearVelocity = gOwnershipProbeBox->getLinearVelocity();
  const PxVec3 angularVelocity = gOwnershipProbeBox->getAngularVelocity();
  if (!pose.p.isFinite() || !linearVelocity.isFinite() ||
      !angularVelocity.isFinite()) {
    gRuntimeMetrics.nonFinite++;
    return;
  }
  gOwnershipProbeMetrics.finalPosition = pose.p;
  gOwnershipProbeMetrics.maxHorizontalSpeed = PxMax(
      gOwnershipProbeMetrics.maxHorizontalSpeed,
      PxSqrt(linearVelocity.x * linearVelocity.x +
             linearVelocity.z * linearVelocity.z));
  gOwnershipProbeMetrics.maxAngularSpeed = PxMax(
      gOwnershipProbeMetrics.maxAngularSpeed, angularVelocity.magnitude());
  PxReal bottomGap = 0.0f;
  if (measureBoxBottomRelToSurface(pose, bottomGap)) {
    gOwnershipProbeMetrics.minBottomGap =
        PxMin(gOwnershipProbeMetrics.minBottomGap, bottomGap);
    gOwnershipProbeMetrics.maxBottomGap =
        PxMax(gOwnershipProbeMetrics.maxBottomGap, bottomGap);
  } else {
    gOwnershipProbeMetrics.outOfFootprintFrames++;
  }
}

static void updateSurfaceHistoryMetrics() {
  if (!gSurfaceHistoryProbe || !gOwnershipProbeBox)
    return;

  const PxTransform pose = gOwnershipProbeBox->getGlobalPose();
  const PxVec3 linearVelocity = gOwnershipProbeBox->getLinearVelocity();
  if (!pose.p.isFinite() || !linearVelocity.isFinite()) {
    gRuntimeMetrics.nonFinite++;
    return;
  }

  const PxReal surfaceY = sampleMeshSurfaceY(pose.p);
  if (!gSurfaceHistoryMetrics.motionStartCaptured &&
      gSimFrame == gSurfaceHistoryWarmupFrames) {
    gSurfaceHistoryMetrics.surfaceYAtMotionStart = surfaceY;
    gSurfaceHistoryMetrics.surfaceYAtMotionEnd = surfaceY;
    gSurfaceHistoryMetrics.bodyYAtMotionStart = pose.p.y;
    gSurfaceHistoryMetrics.bodyYAtMotionEnd = pose.p.y;
    gSurfaceHistoryMetrics.motionStartCaptured = true;
  }

  if (gSurfaceHistoryMetrics.motionStartCaptured &&
      gSimFrame > gSurfaceHistoryWarmupFrames &&
      gSimFrame <=
          gSurfaceHistoryWarmupFrames + gSurfaceHistoryMotionFrames) {
    const PxReal relativeVelocityY =
        linearVelocity.y - gSurfaceHistorySpeed;
    gSurfaceHistoryMetrics.surfaceYAtMotionEnd = surfaceY;
    gSurfaceHistoryMetrics.bodyYAtMotionEnd = pose.p.y;
    gSurfaceHistoryMetrics.sumBodyVelocityY += linearVelocity.y;
    gSurfaceHistoryMetrics.sumAbsRelativeVelocityY +=
        PxAbs(relativeVelocityY);
    gSurfaceHistoryMetrics.minBodyVelocityY =
        PxMin(gSurfaceHistoryMetrics.minBodyVelocityY, linearVelocity.y);
    gSurfaceHistoryMetrics.maxBodyVelocityY =
        PxMax(gSurfaceHistoryMetrics.maxBodyVelocityY, linearVelocity.y);
    gSurfaceHistoryMetrics.maxAbsRelativeVelocityY =
        PxMax(gSurfaceHistoryMetrics.maxAbsRelativeVelocityY,
              PxAbs(relativeVelocityY));
    gSurfaceHistoryMetrics.motionSamples++;
  }
}

static bool isBoxInsideMeshFootprint(const PxTransform &pose) {
  for (PxU32 corner = 0; corner < 8; ++corner) {
    const PxVec3 local((corner & 1u) ? gStackHalfExtent : -gStackHalfExtent,
                       (corner & 2u) ? gStackHalfExtent : -gStackHalfExtent,
                       (corner & 4u) ? gStackHalfExtent : -gStackHalfExtent);
    if (!isInsideMeshFootprint(pose.transform(local)))
      return false;
  }
  return true;
}

static bool measureBoxBottomRelToSurface(const PxTransform &pose,
                                         PxReal &relativeHeight) {
  if (!isBoxInsideMeshFootprint(pose))
    return false;
  relativeHeight =
      pose.p.y - gStackHalfExtent - sampleMeshSurfaceY(pose.p);
  return true;
}

static bool isCompletelyBelowMovingMesh(const PxTransform &pose,
                                        DeformableFilterTag tag) {
  const PxReal margin = 0.5f;
  if (tag == eFILTER_SPHERE_SHOT || tag == eFILTER_STRESS_SHOT) {
    if (!isInsideMeshFootprint(pose.p, gShotSphereRadius))
      return false;
    return pose.p.y + gShotSphereRadius <
           sampleMeshSurfaceY(pose.p) - margin;
  }
  if (tag != eFILTER_BOX && tag != eFILTER_OWNERSHIP_PROBE)
    return false;
  if (!isBoxInsideMeshFootprint(pose))
    return false;

  // A tilted box cannot be classified from center.y+halfExtent.  Require all
  // eight world-space OBB corners to lie below their corresponding triangle.
  for (PxU32 corner = 0; corner < 8; ++corner) {
    const PxVec3 local((corner & 1u) ? gStackHalfExtent : -gStackHalfExtent,
                       (corner & 2u) ? gStackHalfExtent : -gStackHalfExtent,
                       (corner & 4u) ? gStackHalfExtent : -gStackHalfExtent);
    const PxVec3 world = pose.transform(local);
    if (world.y >= sampleMeshSurfaceY(world) - margin)
      return false;
  }
  return true;
}

static void updateRuntimeMetrics() {
  if (!gScene)
    return;
  const PxU32 nbDyn = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  if (!nbDyn)
    return;
  PxArray<PxRigidActor *> actors(nbDyn);
  gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
                    reinterpret_cast<PxActor **>(actors.begin()), nbDyn);
  PxU32 frameFullFallThroughBodies = 0;
  for (PxU32 i = 0; i < nbDyn; ++i) {
    PxRigidDynamic *body =
        actors[i] ? actors[i]->is<PxRigidDynamic>() : NULL;
    if (!body)
      continue;
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 linearVelocity = body->getLinearVelocity();
    const PxVec3 angularVelocity = body->getAngularVelocity();
    if (!pose.p.isFinite() || !pose.q.isFinite() ||
        !linearVelocity.isFinite() || !angularVelocity.isFinite()) {
      gRuntimeMetrics.nonFinite++;
      continue;
    }
    const PxReal quaternionNorm = pose.q.magnitudeSquared();
    const PxReal linearSpeed = linearVelocity.magnitude();
    const PxReal angularSpeed = angularVelocity.magnitude();
    if (!PxIsFinite(quaternionNorm) || !PxIsFinite(linearSpeed) ||
        !PxIsFinite(angularSpeed)) {
      gRuntimeMetrics.nonFinite++;
      continue;
    }
    gRuntimeMetrics.maxQuaternionNormError =
        PxMax(gRuntimeMetrics.maxQuaternionNormError,
              PxAbs(quaternionNorm - 1.0f));
    gRuntimeMetrics.maxAbsPosition =
        PxMax(gRuntimeMetrics.maxAbsPosition,
              PxMax(PxAbs(pose.p.x),
                    PxMax(PxAbs(pose.p.y), PxAbs(pose.p.z))));
    gRuntimeMetrics.maxLinearSpeed =
        PxMax(gRuntimeMetrics.maxLinearSpeed, linearSpeed);
    gRuntimeMetrics.maxAngularSpeed =
        PxMax(gRuntimeMetrics.maxAngularSpeed, angularSpeed);

    const DeformableFilterTag tag = getActorTag(body);
    if (isCompletelyBelowMovingMesh(pose, tag))
      frameFullFallThroughBodies++;
  }
  gRuntimeMetrics.maxFullFallThroughBodies =
      PxMax(gRuntimeMetrics.maxFullFallThroughBodies,
            frameFullFallThroughBodies);
}

static void updateStressHeadlessMetrics() {
  if (!gHeadlessStress || !gScene)
    return;
  const PxU32 nbDyn = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  if (nbDyn == 0)
    return;
  PxArray<PxRigidActor *> actors(nbDyn);
  gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
                    reinterpret_cast<PxActor **>(actors.begin()), nbDyn);

  PxU32 frameSunk = 0;
  PxU32 frameAirborneShots = 0;
  PxU32 framePassThroughShots = 0;
  PxU32 frameOutOfFootprintShots = 0;
  PxU32 frameOutOfFootprintBoxes = 0;
  PxReal frameMinBodyY = 1e9f;
  PxReal frameMinRel = 1e9f;
  gStressActiveShots = 0;

  for (PxU32 a = 0; a < nbDyn; ++a) {
    PxRigidDynamic *rb = actors[a] ? actors[a]->is<PxRigidDynamic>() : NULL;
    if (!rb)
      continue;
    const PxTransform pose = rb->getGlobalPose();
    const PxVec3 p = pose.p;
    const PxVec3 v = rb->getLinearVelocity();
    if (!PxIsFinite(p.x) || !PxIsFinite(p.y) || !PxIsFinite(p.z)) {
      gStressMetrics.nanEvents++;
      continue;
    }
    const DeformableFilterTag tag = getActorTag(rb);
    const bool isShot = tag == eFILTER_STRESS_SHOT;
    if (isShot) {
      gStressActiveShots++;
      if (!isInsideMeshFootprint(p, gShotSphereRadius)) {
        frameOutOfFootprintShots++;
        continue;
      }
      const PxReal surfaceY = sampleMeshSurfaceY(p);
      const PxReal gap = (p.y - gShotSphereRadius) - surfaceY - gMeshRestOffset;
      if (gap > 2.0f)
        frameAirborneShots++;
      if (p.y < surfaceY - 5.0f)
        framePassThroughShots++;
    } else if (tag == eFILTER_BOX) {
      PxReal rel = 0.0f;
      if (!measureBoxBottomRelToSurface(pose, rel)) {
        frameOutOfFootprintBoxes++;
        continue;
      }
      frameMinRel = PxMin(frameMinRel, rel);
      if (rel < -0.5f)
        frameSunk++;
    }
    frameMinBodyY = PxMin(frameMinBodyY, p.y);
  }

  if (frameMinRel < 1e8f && frameMinRel < gStressMetrics.worstMinRelToSurface) {
    gStressMetrics.worstMinRelToSurface = frameMinRel;
    gStressMetrics.worstMinBodyY = frameMinBodyY;
    gStressMetrics.worstFrame = gSimFrame;
  }
  gStressMetrics.maxSunkBoxes = PxMax(gStressMetrics.maxSunkBoxes, frameSunk);
  gStressMetrics.maxPassThroughShots =
      PxMax(gStressMetrics.maxPassThroughShots, framePassThroughShots);
  gStressMetrics.maxAirborneShots =
      PxMax(gStressMetrics.maxAirborneShots, frameAirborneShots);
  gStressMetrics.maxOutOfFootprintShots =
      PxMax(gStressMetrics.maxOutOfFootprintShots,
            frameOutOfFootprintShots);
  gStressMetrics.maxOutOfFootprintBoxes =
      PxMax(gStressMetrics.maxOutOfFootprintBoxes,
            frameOutOfFootprintBoxes);

  if (std::getenv("AVBD_STRESS_TRACE") &&
      (frameSunk > 0 || framePassThroughShots > 0 || gSimFrame < 30 ||
       gSimFrame % 60 == 0)) {
    printf("[StressTrace] frame=%u waveAmp=%.2f minBodyY=%.2f minBoxBottomRel=%.3f "
           "sunk=%u passThroughShots=%u airborneShots=%u activeShots=%u\n",
           gSimFrame, sinf(gTime) * 20.0f, frameMinBodyY, frameMinRel, frameSunk,
           framePassThroughShots, frameAirborneShots, gStressActiveShots);
  }
}

static void createStressBoxGrid() {
  PxShape *shape = gPhysics->createShape(
      PxBoxGeometry(gStackHalfExtent, gStackHalfExtent, gStackHalfExtent),
      *gMaterial);
  if (!shape) {
    gInitializationFailed = true;
    return;
  }
  setShapeTag(*shape, eFILTER_BOX);
  const PxReal spacing = gStackHalfExtent * 2.2f;
  const PxReal originX = -0.5f * PxReal(gStressGridX - 1) * spacing;
  const PxReal originZ = -0.5f * PxReal(gStressGridZ - 1) * spacing;
  for (PxU32 iz = 0; iz < gStressGridZ; ++iz) {
    for (PxU32 ix = 0; ix < gStressGridX; ++ix) {
      const PxReal x = originX + PxReal(ix) * spacing;
      const PxReal z = originZ + PxReal(iz) * spacing;
      PxVec3 pos(x, 0.0f, z);
      pos.y = sampleMeshSurfaceY(pos) + gStackHalfExtent + 0.05f;
      PxRigidDynamic *body =
          gPhysics->createRigidDynamic(PxTransform(pos, PxQuat(PxIdentity)));
      if (!body) {
        gInitializationFailed = true;
        continue;
      }
      body->attachShape(*shape);
      PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
      gScene->addActor(*body);
    }
  }
  shape->release();
  printf("[DeformableMeshStress] grid=%ux%u boxes=%u spacing=%.2f\n",
         gStressGridX, gStressGridZ, gStressGridX * gStressGridZ, spacing);
}

static void resetStackHeadlessMetrics() {
  gStackMetrics = StackHeadlessMetrics();
  gStackMetrics.minRelToSurface = 1e9f;
  gStackMetrics.maxWorldY = -1e9f;
  gStackMetrics.maxSettledWorldY = -1e9f;
}

static void printStressHeadlessSummary() {
  // Penetration, deep sink, and pass-through are probe findings here.  The
  // stress case hardens only finite/runaway and harness lifecycle invariants.
  printf("[DeformableMeshStress] solver=%s frames=%u shots=%u "
         "shotMeshContactEvents=%u shotMeshContactPoints=%u "
         "worstFrame=%u worstMinBodyY=%.4f worstMinBoxBottomRel=%.4f "
         "maxSunkBoxes=%u maxPassThroughShots=%u maxAirborneShots=%u "
         "maxOutOfFootprintShots=%u maxOutOfFootprintBoxes=%u nanEvents=%u "
         "sunkDepthValidation=DIAGNOSTIC\n",
         Snippets::getSolverTypeName(gSolverType), gHeadlessFrameCount,
         gStressMetrics.totalShotsFired, gStressMetrics.contactEvents,
         gStressMetrics.contactPoints, gStressMetrics.worstFrame,
         gStressMetrics.worstMinBodyY, gStressMetrics.worstMinRelToSurface,
         gStressMetrics.maxSunkBoxes, gStressMetrics.maxPassThroughShots,
         gStressMetrics.maxAirborneShots,
         gStressMetrics.maxOutOfFootprintShots,
         gStressMetrics.maxOutOfFootprintBoxes, gStressMetrics.nanEvents);
}

static void updateStackHeadlessMetrics() {
  if (!gHeadlessMode || gHeadlessSphereShot || gHeadlessStress || !gScene)
    return;
  const PxU32 nbDyn = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  if (nbDyn == 0)
    return;
  PxArray<PxRigidActor *> actors(nbDyn);
  gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
                    reinterpret_cast<PxActor **>(actors.begin()), nbDyn);
  const bool settledWindow = (gSimFrame + 60 >= gHeadlessFrameCount);
  PxReal frameMaxSpeed = 0.0f;
  PxReal frameMaxSpread = 0.0f;
  PxReal frameMaxWorldY = -1e9f;
  PxReal frameMinRel = 1e9f;
  PxReal frameMaxVyUp = 0.0f;
  PxReal frameMaxVxz = 0.0f;
  PxU32 frameSunk = 0;
  PxU32 frameOutOfFootprint = 0;
  PxU32 frameAwake = 0;
  for (PxU32 a = 0; a < nbDyn; ++a) {
    PxRigidDynamic *rb = actors[a] ? actors[a]->is<PxRigidDynamic>() : NULL;
    if (!rb)
      continue;
    if (!rb->isSleeping())
      frameAwake++;
    const PxVec3 v = rb->getLinearVelocity();
    const PxTransform pose = rb->getGlobalPose();
    const PxVec3 p = pose.p;
    if (!PxIsFinite(v.x) || !PxIsFinite(v.y) || !PxIsFinite(v.z) ||
        !PxIsFinite(p.x) || !PxIsFinite(p.y) || !PxIsFinite(p.z)) {
      gStackMetrics.nanBodies++;
      continue;
    }
    frameMaxSpeed = PxMax(frameMaxSpeed, v.magnitude());
    frameMaxVyUp = PxMax(frameMaxVyUp, v.y);
    frameMaxVxz = PxMax(frameMaxVxz, PxSqrt(v.x * v.x + v.z * v.z));
    frameMaxSpread = PxMax(frameMaxSpread, PxSqrt(p.x * p.x + p.z * p.z));
    frameMaxWorldY = PxMax(frameMaxWorldY, p.y);
    // Keep the historical center-vs-surface diagnostic, but only while the
    // whole OBB is inside the finite mesh footprint.
    if (!isBoxInsideMeshFootprint(pose)) {
      frameOutOfFootprint++;
      continue;
    }
    const PxReal relToSurface = p.y - sampleMeshSurfaceY(p);
    frameMinRel = PxMin(frameMinRel, relToSurface);
    if (relToSurface < -1.0f)
      frameSunk++;
  }
  if (std::getenv("AVBD_STACK_TRACE") &&
      (gSimFrame < 30 ? (gSimFrame % 3 == 0) : (gSimFrame % 30 == 0)))
    printf("[StackTrace] frame=%u spreadXZ=%.3f maxSpeed=%.3f maxVy=%.3f "
           "maxVxz=%.3f awake=%u\n",
           gSimFrame, frameMaxSpread, frameMaxSpeed, frameMaxVyUp, frameMaxVxz,
           frameAwake);
  gStackMetrics.maxSpeed = PxMax(gStackMetrics.maxSpeed, frameMaxSpeed);
  gStackMetrics.maxSpreadXZ = PxMax(gStackMetrics.maxSpreadXZ, frameMaxSpread);
  gStackMetrics.maxWorldY = PxMax(gStackMetrics.maxWorldY, frameMaxWorldY);
  gStackMetrics.minRelToSurface =
      PxMin(gStackMetrics.minRelToSurface, frameMinRel);
  gStackMetrics.maxOutOfFootprintBoxes =
      PxMax(gStackMetrics.maxOutOfFootprintBoxes, frameOutOfFootprint);
  if (settledWindow) {
    gStackMetrics.maxSettledSpeed =
        PxMax(gStackMetrics.maxSettledSpeed, frameMaxSpeed);
    gStackMetrics.maxSettledSpreadXZ =
        PxMax(gStackMetrics.maxSettledSpreadXZ, frameMaxSpread);
    gStackMetrics.maxSettledWorldY =
        PxMax(gStackMetrics.maxSettledWorldY, frameMaxWorldY);
    gStackMetrics.settledSunkBoxes =
        PxMax(gStackMetrics.settledSunkBoxes, frameSunk);
  }
}

static void resetSphereShotMetrics() {
  gSphereShotMetrics = SphereShotMetrics();
  gSphereShotMetrics.firstContactFrame = PX_MAX_U32;
  gSphereShotMetrics.lastContactEventFrame = PX_MAX_U32;
  gSphereShotMetrics.firstOverlapFrame = PX_MAX_U32;
  gSphereShotMetrics.maxOverlapFrame = PX_MAX_U32;
  gSphereShotMetrics.maxRestOffsetProximityFrame = PX_MAX_U32;
  gSphereShotMetrics.minSphereY = 1e9f;
  gHavePrevSampleSurfaceY = false;
  gPrevSampleSurfaceY = 0.0f;
  gPreviousSphereVelocity = PxVec3(0.0f);
  gSphereContactBaselineVelocity = PxVec3(0.0f);
  gHavePreviousSphereVelocity = false;
  gHaveSphereContactBaseline = false;
}

static void updateSphereShotMetrics() {
  if (!gShotSphere)
    return;
  const PxVec3 p = gShotSphere->getGlobalPose().p;
  const PxVec3 v = gShotSphere->getLinearVelocity();
  const PxVec3 w = gShotSphere->getAngularVelocity();
  if (!p.isFinite() || !v.isFinite() || !w.isFinite()) {
    gSphereShotMetrics.nanDetected = true;
    return;
  }
  gSphereShotMetrics.minSphereY = PxMin(gSphereShotMetrics.minSphereY, p.y);
  gSphereShotMetrics.maxAbsSphereVel =
      PxMax(gSphereShotMetrics.maxAbsSphereVel, v.magnitude());
  if (gHaveSphereContactBaseline &&
      gSphereShotMetrics.firstContactFrame != PX_MAX_U32 &&
      gSimFrame >= gSphereShotMetrics.firstContactFrame &&
      gSimFrame <
          gSphereShotMetrics.firstContactFrame + gSphereResponseWindowFrames) {
    const PxReal response =
        PxAbs(v.y - gSphereContactBaselineVelocity.y);
    gSphereShotMetrics.maxImpactAxisVelocityDelta =
        PxMax(gSphereShotMetrics.maxImpactAxisVelocityDelta, response);
    gSphereShotMetrics.responseSamples++;
  }
  const PxReal horizVel = PxSqrt(v.x * v.x + v.z * v.z);
  const PxReal angVel = w.magnitude();
  gSphereShotMetrics.maxHorizVel =
      PxMax(gSphereShotMetrics.maxHorizVel, horizVel);
  gSphereShotMetrics.maxAngVel = PxMax(gSphereShotMetrics.maxAngVel, angVel);

  if (!isInsideMeshFootprint(p, gShotSphereRadius)) {
    gSphereShotMetrics.outOfFootprintFrames++;
    gHavePrevSampleSurfaceY = false;
    gPreviousSphereVelocity = v;
    gHavePreviousSphereVelocity = true;
    return;
  }

  const PxReal surfaceY = sampleMeshSurfaceY(p);
  // Approximate surface slope from finite differences (world XZ).
  const PxReal eps = 2.0f;
  const PxReal syxp = sampleMeshSurfaceY(p + PxVec3(eps, 0.0f, 0.0f));
  const PxReal syxm = sampleMeshSurfaceY(p + PxVec3(-eps, 0.0f, 0.0f));
  const PxReal syzp = sampleMeshSurfaceY(p + PxVec3(0.0f, 0.0f, eps));
  const PxReal syzm = sampleMeshSurfaceY(p + PxVec3(0.0f, 0.0f, -eps));
  const PxReal dsdx = (syxp - syxm) / (2.0f * eps);
  const PxReal dsdz = (syzp - syzm) / (2.0f * eps);
  const PxReal slope = PxSqrt(dsdx * dsdx + dsdz * dsdz);
  gSphereShotMetrics.maxMeshAbsSlope =
      PxMax(gSphereShotMetrics.maxMeshAbsSlope, slope);

  const PxReal frameDt = 1.0f / 60.0f;
  PxReal meshAbsDyDt = 0.0f;
  if (gHavePrevSampleSurfaceY)
    meshAbsDyDt = PxAbs(surfaceY - gPrevSampleSurfaceY) / frameDt;
  gPrevSampleSurfaceY = surfaceY;
  gHavePrevSampleSurfaceY = true;
  gSphereShotMetrics.maxMeshAbsDyDt =
      PxMax(gSphereShotMetrics.maxMeshAbsDyDt, meshAbsDyDt);

  const bool settledWindow = (gSimFrame + 60 >= gHeadlessFrameCount);
  if (settledWindow) {
    gSphereShotMetrics.maxSettledHorizVel =
        PxMax(gSphereShotMetrics.maxSettledHorizVel, horizVel);
    gSphereShotMetrics.maxSettledAngVel =
        PxMax(gSphereShotMetrics.maxSettledAngVel, angVel);
    gSphereShotMetrics.sumSettledHorizVel += horizVel;
    gSphereShotMetrics.sumSettledMeshAbsDyDt += meshAbsDyDt;
    gSphereShotMetrics.settledRideSamples++;
  }

  // Optional per-frame ride trace (settled window or env).
  if (std::getenv("AVBD_MESH_RIDE_TRACE") &&
      (settledWindow ? (gSimFrame % 5 == 0) : (gSimFrame % 15 == 0))) {
    printf("[MeshRideTrace] frame=%u pos=(%.2f,%.2f,%.2f) vxz=%.3f vy=%.3f "
           "w=%.3f mesh|dy/dt|=%.3f slope=%.3f gap=%.3f\n",
           gSimFrame, p.x, p.y, p.z, horizVel, v.y, angVel, meshAbsDyDt, slope,
           (p.y - gShotSphereRadius) - surfaceY - gMeshRestOffset);
  }

  const PxReal gap = (p.y - gShotSphereRadius) - surfaceY - gMeshRestOffset;
  if (gap > 0.5f) {
    gSphereShotMetrics.maxAirborneGap = PxMax(gSphereShotMetrics.maxAirborneGap, gap);
    if (gSimFrame + 60 >= gHeadlessFrameCount)
      gSphereShotMetrics.maxSettledAirborneGap =
          PxMax(gSphereShotMetrics.maxSettledAirborneGap, gap);
  }
  PxReal restOffsetProximity = 0.0f;
  const PxReal geomOverlap =
      sampleMeshSurfaceY(p) - (p.y - gShotSphereRadius);
  if (geomOverlap > gSphereShotMetrics.maxGeomOverlap)
    gSphereShotMetrics.maxGeomOverlap = geomOverlap;
  if (gSimFrame < 45 && geomOverlap > gSphereShotMetrics.maxImpactGeomOverlap)
    gSphereShotMetrics.maxImpactGeomOverlap = geomOverlap;
  if (gSimFrame + 60 >= gHeadlessFrameCount &&
      geomOverlap > gSphereShotMetrics.maxSettledGeomOverlap)
    gSphereShotMetrics.maxSettledGeomOverlap = geomOverlap;

  if (measureSphereMeshRestOffsetProximity(p, gShotSphereRadius,
                                           restOffsetProximity) &&
      restOffsetProximity > 0.0f) {
    gSphereShotMetrics.proximityFrames++;
    if (gSphereShotMetrics.firstOverlapFrame == PX_MAX_U32)
      gSphereShotMetrics.firstOverlapFrame = gSimFrame;
    if (restOffsetProximity > gSphereShotMetrics.maxRestOffsetProximity) {
      gSphereShotMetrics.maxRestOffsetProximity = restOffsetProximity;
      gSphereShotMetrics.maxRestOffsetProximityFrame = gSimFrame;
    }
    if (gSimFrame < 45 &&
        restOffsetProximity >
            gSphereShotMetrics.maxImpactRestOffsetProximity)
      gSphereShotMetrics.maxImpactRestOffsetProximity =
          restOffsetProximity;
    if (gSimFrame + 60 >= gHeadlessFrameCount &&
        restOffsetProximity >
            gSphereShotMetrics.maxSettledRestOffsetProximity)
      gSphereShotMetrics.maxSettledRestOffsetProximity =
          restOffsetProximity;
    if (restOffsetProximity > 0.55f)
      gSphereShotMetrics.deepProximityFrames++;
    if (gSimFrame + 60 >= gHeadlessFrameCount)
      gSphereShotMetrics.settledProximityFrames++;
  }
  gPreviousSphereVelocity = v;
  gHavePreviousSphereVelocity = true;
}

static void printSphereShotSummary() {
  const PxVec3 fp = gShotSphere ? gShotSphere->getGlobalPose().p : PxVec3(0.0f);
  const PxReal lateralDrift = PxSqrt(fp.x * fp.x + fp.z * fp.z);
  // These are vertical proxies against the triangle surface height, not exact
  // sphere/triangle penetration depths.  The rest-offset proximity metric is
  // diagnostic; hard depth caps use geometric vertical overlap only.
  const PxReal passImpactGeomThreshold = 1.0f;
  const PxReal passSettledGeomThreshold = 1.55f;
  const PxReal passLateralThreshold = 15.0f;
  const bool finalInsideFootprint =
      isInsideMeshFootprint(fp, gShotSphereRadius);
  const PxReal finalSurfaceY =
      finalInsideFootprint ? sampleMeshSurfaceY(fp) : 0.0f;
  const PxReal finalGap = finalInsideFootprint
                              ? (fp.y - gShotSphereRadius) - finalSurfaceY -
                                    gMeshRestOffset
                              : 0.0f;
  const PxU32 minSettledProximityFrames = 30;
  const bool notSettled =
      gSphereShotMetrics.settledProximityFrames <
      minSettledProximityFrames;
  const bool airborne = finalGap > 3.0f ||
                        gSphereShotMetrics.maxSettledAirborneGap > 8.0f;
  const bool impactPenFail =
      gSphereShotMetrics.maxImpactGeomOverlap > passImpactGeomThreshold;
  const bool settledPenFail =
      gSphereShotMetrics.maxSettledGeomOverlap > passSettledGeomThreshold;
  printf("\n[DeformableMeshSphereShot] SUMMARY solver=%s frames=%u stack=%s\n",
         Snippets::getSolverTypeName(gSolverType), gHeadlessFrameCount,
         gCreateStack ? "yes" : "no");
  printf("[DeformableMeshSphereShot] meshRestOffset=-0.5 contactOffset=0.02 "
         "waveAmp=20\n");
  printf("[DeformableMeshSphereShot] firstProximityFrame=%u "
         "maxRestOffsetProximity=%.4f "
         "maxImpactRestOffsetProximity=%.4f "
         "maxSettledRestOffsetProximity=%.4f "
         "maxVerticalGeomOverlap=%.4f maxImpactVerticalGeomOverlap=%.4f "
         "maxSettledVerticalGeomOverlap=%.4f "
         "maxRestOffsetProximityFrame=%u proximityFrames=%u "
         "deepProximityFrames=%u settledProximityFrames=%u\n",
         gSphereShotMetrics.firstOverlapFrame,
         gSphereShotMetrics.maxRestOffsetProximity,
         gSphereShotMetrics.maxImpactRestOffsetProximity,
         gSphereShotMetrics.maxSettledRestOffsetProximity,
         gSphereShotMetrics.maxGeomOverlap,
         gSphereShotMetrics.maxImpactGeomOverlap,
         gSphereShotMetrics.maxSettledGeomOverlap,
         gSphereShotMetrics.maxRestOffsetProximityFrame,
         gSphereShotMetrics.proximityFrames,
         gSphereShotMetrics.deepProximityFrames,
         gSphereShotMetrics.settledProximityFrames);
  printf("[DeformableMeshSphereShot] minSphereY=%.4f maxAbsVel=%.4f "
         "maxAirborneGap=%.4f maxSettledAirborneGap=%.4f finalGap=%.4f nan=%s\n",
         gSphereShotMetrics.minSphereY, gSphereShotMetrics.maxAbsSphereVel,
         gSphereShotMetrics.maxAirborneGap,
         gSphereShotMetrics.maxSettledAirborneGap, finalGap,
         gSphereShotMetrics.nanDetected ? "true" : "false");
  printf("[DeformableMeshSphereShot] finalSpherePos=(%.4f,%.4f,%.4f)\n", fp.x,
         fp.y, fp.z);
  printf("[DeformableMeshSphereShot] finalInsideFootprint=%u "
         "outOfFootprintFrames=%u\n",
         finalInsideFootprint ? 1u : 0u,
         gSphereShotMetrics.outOfFootprintFrames);
  printf("[DeformableMeshSphereShot] lateralDriftXZ=%.4f (limit %.1f)\n",
         lateralDrift, passLateralThreshold);
  const PxReal avgSettledHoriz =
      gSphereShotMetrics.settledRideSamples
          ? gSphereShotMetrics.sumSettledHorizVel /
                PxReal(gSphereShotMetrics.settledRideSamples)
          : 0.0f;
  const PxReal avgSettledMeshDyDt =
      gSphereShotMetrics.settledRideSamples
          ? gSphereShotMetrics.sumSettledMeshAbsDyDt /
                PxReal(gSphereShotMetrics.settledRideSamples)
          : 0.0f;
  // On a standing heave wave, mesh material points move mostly in Y; horizontal
  // ride comes from slope+gravity and tangential friction. Low settled |vxz|
  // while slope and |dy/dt| are large indicates friction not following surface.
  printf("[DeformableMeshSphereShot] ride maxHorizVel=%.4f maxSettledHorizVel=%.4f "
         "avgSettledHorizVel=%.4f maxAngVel=%.4f maxSettledAngVel=%.4f "
         "maxMesh|dy/dt|=%.4f avgSettledMesh|dy/dt|=%.4f maxSlope=%.4f "
         "settledSamples=%u\n",
         gSphereShotMetrics.maxHorizVel, gSphereShotMetrics.maxSettledHorizVel,
         avgSettledHoriz, gSphereShotMetrics.maxAngVel,
         gSphereShotMetrics.maxSettledAngVel, gSphereShotMetrics.maxMeshAbsDyDt,
         avgSettledMeshDyDt, gSphereShotMetrics.maxMeshAbsSlope,
         gSphereShotMetrics.settledRideSamples);
  const bool rideWeak =
      gSphereShotMetrics.settledRideSamples >= 30 &&
      gSphereShotMetrics.maxMeshAbsSlope > 0.15f &&
      gSphereShotMetrics.maxSettledHorizVel < 0.25f &&
      gSphereShotMetrics.maxSettledAngVel < 0.25f;
  printf("[DeformableMeshSphereShot] rideDiagnosis=%s "
         "(weak if settled on slope>0.15 but |vxz| and |w| both <0.25)\n",
         rideWeak ? "WEAK_OR_STUCK" : "ok_or_flat");
  printf("[DeformableMeshSphereShot] contactEvents=%u contactPoints=%u "
         "firstContactFrame=%u responseSamples=%u maxContactImpulse=%.4f "
         "maxImpactAxisVelocityDelta=%.4f responseFraction=%.4f\n",
         gSphereShotMetrics.contactEvents, gSphereShotMetrics.contactPoints,
         gSphereShotMetrics.firstContactFrame,
         gSphereShotMetrics.responseSamples,
         gSphereShotMetrics.maxContactImpulse,
         gSphereShotMetrics.maxImpactAxisVelocityDelta,
         gSphereShotMetrics.maxImpactAxisVelocityDelta / gShotSpeedY);
  printf("[DeformableMeshSphereShot] penetrationDiagnostic impactFail=%u "
         "settledFail=%u lateralFail=%u notSettled=%u airborne=%u "
         "(limits impactVerticalGeom=%.2f settledVerticalGeom=%.2f "
         "lateralDriftXZ=%.1f settledProximity=%u)\n",
         impactPenFail ? 1u : 0u, settledPenFail ? 1u : 0u,
         lateralDrift > passLateralThreshold ? 1u : 0u,
         notSettled ? 1u : 0u, airborne ? 1u : 0u,
         passImpactGeomThreshold, passSettledGeomThreshold,
         passLateralThreshold, minSettledProximityFrames);
}

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0), PxReal density=1.0f, DeformableFilterTag tag=eFILTER_UNTAGGED)
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, density);
	if (!dynamic) {
		gInitializationFailed = true;
		return NULL;
	}
	PxShape *shape = NULL;
	dynamic->getShapes(&shape, 1);
	if (shape)
		setShapeTag(*shape, tag);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static bool spawnStressShot() {
  if (!gScene)
    return false;
  const PxReal spread = 120.0f;
  const PxReal fx = PxReal(gStressShotSerial % 7);
  const PxReal fz = PxReal((gStressShotSerial * 3) % 11);
  const PxReal x = (fx - 3.0f) * spread * 0.25f;
  // Keep every diagnostic projectile inside the actual [-400, 112] mesh
  // footprint.  The former 0.22 factor placed two projectiles at z=132,
  // where free fall was incorrectly classified against a clamped edge height.
  const PxReal z = (fz - 5.0f) * spread * 0.16f;
  const PxVec3 spawnPosition(x, gStressShotSpawnY, z);
  if (!isInsideMeshFootprint(spawnPosition, gShotSphereRadius)) {
    gInitializationFailed = true;
    return false;
  }
  PxRigidDynamic *shot =
      createDynamic(PxTransform(spawnPosition),
                    PxSphereGeometry(gShotSphereRadius),
                    PxVec3(0.0f, -gStressShotSpeedY, 0.0f), 3.0f,
                    eFILTER_STRESS_SHOT);
  if (!shot)
    return false;
  gStressShots.push_back(shot);
  gStressShotSerial++;
  gStressActiveShots++;
  gFastImpactSubstepFrames = gFastImpactSubstepHoldFrames;
  return true;
}

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	if (!shape) {
		gInitializationFailed = true;
		return;
	}
	setShapeTag(*shape, eFILTER_BOX);
	for(PxU32 i=0; i<size;i++)
	{
		for(PxU32 j=0;j<size-i;j++)
		{
			PxTransform localTm(PxVec3(PxReal(j*2) - PxReal(size-i), PxReal(i*2+1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			if (!body) {
				gInitializationFailed = true;
				continue;
			}
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			gScene->addActor(*body);
		}
	}
	shape->release();
}

static void createBroadAuthorityCompound() {
	// Each patch independently produces roughly 80 supported surface rows.
	// Their connector shapes touch each other, so the real rigid-contact graph
	// assembles both bodies into one component above the 128-row matrix-free
	// boundary.  This validates multi-body component authority rather than a
	// synthetic wide matrix on one body.
	const PxU32 side = 9;
	const PxReal spacing = 1.1f;
	const PxReal bodyCenterX = 5.0f;
	// Keep the top of every box above the one-sided triangle surface while
	// the contact settles toward the mesh's -0.5 m rest offset.  A thinner
	// compound can pass completely through the surface before reaching that
	// authored rest depth, which invalidates both the TGS control and the AVBD
	// authority probe.
	const PxVec3 halfExtents(0.45f, 0.75f, 0.45f);
	for (PxU32 bodyIndex = 0;
	     bodyIndex < 2 && !gInitializationFailed; ++bodyIndex) {
		const PxReal centerX =
		    bodyIndex == 0 ? -bodyCenterX : bodyCenterX;
		PxVec3 position(centerX, 0.0f, 0.0f);
		position.y = sampleMeshSurfaceY(position) + halfExtents.y;
		PxRigidDynamic* body =
		    gPhysics->createRigidDynamic(PxTransform(position));
		if (!body) {
			gInitializationFailed = true;
			break;
		}
		for (PxU32 z = 0; z < side && !gInitializationFailed; ++z) {
			for (PxU32 x = 0; x < side; ++x) {
				PxShape* shape = gPhysics->createShape(
				    PxBoxGeometry(halfExtents), *gMaterial);
				if (!shape) {
					gInitializationFailed = true;
					break;
				}
				const PxReal localX =
				    (PxReal(x) - PxReal(side - 1) * 0.5f) *
				    spacing;
				const PxReal localZ =
				    (PxReal(z) - PxReal(side - 1) * 0.5f) *
				    spacing;
				const PxVec3 samplePoint(
				    centerX + localX, 0.0f, localZ);
				const PxReal localY =
				    sampleMeshSurfaceY(samplePoint) +
				    halfExtents.y - position.y;
				shape->setLocalPose(PxTransform(
				    PxVec3(localX, localY, localZ)));
				shape->setContactOffset(0.05f);
				setShapeTag(*shape, eFILTER_OWNERSHIP_PROBE);
				body->attachShape(*shape);
				shape->release();
			}
		}
		PxShape* connector = gPhysics->createShape(
		    PxBoxGeometry(0.30f, 0.25f, 0.30f), *gMaterial);
		if (!connector) {
			gInitializationFailed = true;
		} else {
			const PxReal connectorWorldX =
			    bodyIndex == 0 ? -0.30f : 0.30f;
			const PxReal connectorLocalY =
			    sampleMeshSurfaceY(PxVec3(0.0f)) + 1.50f -
			    position.y;
			connector->setLocalPose(PxTransform(PxVec3(
			    connectorWorldX - centerX, connectorLocalY, 0.0f)));
			connector->setContactOffset(0.05f);
			setShapeTag(*connector, eFILTER_OWNERSHIP_PROBE);
			body->attachShape(*connector);
			connector->release();
		}
		if (gInitializationFailed ||
		    !PxRigidBodyExt::setMassAndUpdateInertia(*body, 200.0f)) {
			body->release();
			gInitializationFailed = true;
			break;
		}
		gScene->addActor(*body);
	}
	printf("[DeformableMeshBroadAuthority] bodies=2 patch=%ux%u "
	       "shapesPerBody=%u totalMass=400\n",
	       side, side, side * side + 1);
}

struct Triangle
{
	PxU32 ind0, ind1, ind2;
};

static void updateVertices(PxVec3* verts, float amplitude=0.0f)
{
	const PxU32 gridSize = gGridSize;
	const PxReal gridStep = gGridStep;

	for(PxU32 a=0; a<gridSize; a++)
	{
		const float coeffA = float(a)/float(gridSize);
		for(PxU32 b=0; b<gridSize; b++)
		{
			const float coeffB = float(b)/float(gridSize);

			const float y = 20.0f + sinf(coeffA*PxTwoPi)*cosf(coeffB*PxTwoPi)*amplitude;

			verts[a * gridSize + b] = PxVec3(gGridMinimum + b*gridStep, y, gGridMinimum + a*gridStep);
		}
	}
}

static void updateFlatVertices(PxVec3 *verts, PxReal verticalOffset) {
	const PxReal y = 20.0f + verticalOffset;
	for (PxU32 a = 0; a < gGridSize; ++a) {
		for (PxU32 b = 0; b < gGridSize; ++b) {
			verts[a * gGridSize + b] =
			    PxVec3(gGridMinimum + b * gGridStep, y,
			           gGridMinimum + a * gGridStep);
		}
	}
}

static PxTriangleMesh* createMeshGround(const PxCookingParams& params)
{
	const PxU32 gridSize = gGridSize;

	PxVec3 verts[gridSize * gridSize];

	const PxU32 nbTriangles = 2 * (gridSize - 1) * (gridSize-1);

	Triangle indices[nbTriangles];

	updateVertices(verts);

	for (PxU32 a = 0; a < (gridSize-1); ++a)
	{
		for (PxU32 b = 0; b < (gridSize-1); ++b)
		{
			Triangle& tri0 = indices[(a * (gridSize-1) + b) * 2];
			Triangle& tri1 = indices[((a * (gridSize-1) + b) * 2) + 1];

			tri0.ind0 = a * gridSize + b + 1;
			tri0.ind1 = a * gridSize + b;
			tri0.ind2 = (a + 1) * gridSize + b + 1;

			tri1.ind0 = (a + 1) * gridSize + b + 1;
			tri1.ind1 = a * gridSize + b;
			tri1.ind2 = (a + 1) * gridSize + b;
		}
	}

	PxTriangleMeshDesc meshDesc;
	meshDesc.points.data = verts;
	meshDesc.points.count = gridSize * gridSize;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.triangles.count = nbTriangles;
	meshDesc.triangles.data = indices;
	meshDesc.triangles.stride = sizeof(Triangle);

	PxTriangleMesh* triMesh = PxCreateTriangleMesh(params, meshDesc, gPhysics->getPhysicsInsertionCallback());

	return triMesh;
}

static void resetRuntimeState() {
  gErrorCallback.reset();
  gInitializationFailed = false;
  gExtensionsInitialized = false;
  gFetchPending = false;
  gCleanupCompleted = false;
  gFoundation = NULL;
  gPhysics = NULL;
  gDispatcher = NULL;
  gScene = NULL;
  gMaterial = NULL;
  gPvd = NULL;
  gMesh = NULL;
  gActor = NULL;
  gShotSphere = NULL;
  gOwnershipProbeBox = NULL;
  gStressShots.clear();
  gTime = 0.0f;
  gSimFrame = 0;
  gFastImpactSubstepFrames = 0;
  gSurfaceHistoryOffset = 0.0f;
  gRuntimeMetrics = RuntimeMetrics();
  resetStackHeadlessMetrics();
  resetSphereShotMetrics();
  resetStressHeadlessMetrics();
  resetOwnershipProbeMetrics();
  gSurfaceHistoryMetrics = SurfaceHistoryMetrics();
}

void initPhysics(bool interactive)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if (!gFoundation) {
		gInitializationFailed = true;
		return;
	}

	if (interactive) {
		gPvd = PxCreatePvd(*gFoundation);
		if (gPvd) {
			PxPvdTransport* transport =
				PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
			if (transport)
				gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
		}
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
		PxTolerancesScale(), true, gPvd);
	if (!gPhysics) {
		gInitializationFailed = true;
		return;
	}
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	if (!gExtensionsInitialized) {
		gInitializationFailed = true;
		return;
	}

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	const PxU32 dispatcherThreads =
		interactive ? 2u : gHeadlessOptions.dispatcherThreads;
	gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
	if (!gDispatcher) {
		gInitializationFailed = true;
		return;
	}
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader =
		interactive ? PxDefaultSimulationFilterShader : deformableGateFilterShader;
	sceneDesc.simulationEventCallback = interactive ? NULL : &gSimulationCallback;
	sceneDesc.solverType = gSolverType;
	gScene = gPhysics->createScene(sceneDesc);
	if (!gScene) {
		gInitializationFailed = true;
		return;
	}

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	const PxReal staticFriction = gHeadlessSphereShot ? 0.5f : 1.0f;
	const PxReal dynamicFriction = gHeadlessSphereShot ? 0.5f : 1.0f;
	gMaterial = gPhysics->createMaterial(staticFriction, dynamicFriction, 0.0f);
	if (!gMaterial) {
		gInitializationFailed = true;
		return;
	}

	PxCookingParams cookingParams(gPhysics->getTolerancesScale());
	cookingParams.midphaseDesc.setToDefault(PxMeshMidPhase::eBVH34);
	cookingParams.midphaseDesc.mBVH34Desc.quantized = false;
	// We need to disable the mesh cleaning part so that the vertex mapping remains untouched.
	cookingParams.meshPreprocessParams = PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;
	gMesh = createMeshGround(cookingParams);
	if (!gMesh) {
		gInitializationFailed = true;
		return;
	}

	PxTriangleMeshGeometry geom(gMesh);
	gActor = gPhysics->createRigidStatic(PxTransform(PxVec3(0, 2, 0)));
	PxShape* shape = gPhysics->createShape(geom, *gMaterial);
	if (!gActor || !shape) {
		PX_RELEASE(shape);
		// The actor is not scene-owned until addActor() below.  Release a
		// partially-created actor here so the headless initialization error path
		// remains leak-free.
		PX_RELEASE(gActor);
		gInitializationFailed = true;
		return;
	}
	shape->setContactOffset(0.02f);
	// A negative rest offset helps to avoid jittering when the deformed mesh moves away from objects resting on it.
	shape->setRestOffset(gMeshRestOffset);
	setShapeTag(*shape, eFILTER_MOVING_MESH);
	gActor->attachShape(*shape);
	shape->release();
	gScene->addActor(*gActor);

	if (gHeadlessSphereShot) {
		gShotSphere = createDynamic(
		    PxTransform(PxVec3(0.0f, gShotSpawnY, 0.0f)),
		    PxSphereGeometry(gShotSphereRadius), PxVec3(0.0f, -gShotSpeedY, 0.0f),
		    3.0f, eFILTER_SPHERE_SHOT);
		gPreviousSphereVelocity = PxVec3(0.0f, -gShotSpeedY, 0.0f);
		gHavePreviousSphereVelocity = gShotSphere != NULL;
		printf("[DeformableMeshSphereShot] init solver=%s headlessShot=true stack=no\n",
		       Snippets::getSolverTypeName(gSolverType));
		printf("[DeformableMeshSphereShot] spawn frame=0 pos=(0,%.2f,0) vel=(0,-%.1f,0) "
		       "radius=%.2f\n",
		       gShotSpawnY, gShotSpeedY, gShotSphereRadius);
	} else if (gHeadlessOwnershipProbe) {
		PxVec3 position(0.0f);
		position.y =
		    sampleMeshSurfaceY(position) + gStackHalfExtent +
		    ((gShellPostOwnershipProbe || gNormalPostOwnershipProbe)
		         ? -0.20f
		         : 0.05f);
		if (gShellPostOwnershipProbe)
			PxAvbdKinematicShellSetBoxCornerShellEnabled(true);
		const PxVec3 initialVelocity =
		    (gShellPostOwnershipProbe || gNormalPostOwnershipProbe)
		        ? PxVec3(8.0f, 6.0f, 0.0f)
		        : PxVec3(0.0f);
		gOwnershipProbeBox = createDynamic(
		    PxTransform(position),
		    PxBoxGeometry(gStackHalfExtent, gStackHalfExtent, gStackHalfExtent),
		    initialVelocity, 1.0f, eFILTER_OWNERSHIP_PROBE);
		if (!gOwnershipProbeBox ||
		    !PxRigidBodyExt::setMassAndUpdateInertia(
		        *gOwnershipProbeBox, getOwnershipProbeMass())) {
			gInitializationFailed = true;
		} else {
			gOwnershipProbeMetrics.initialPosition = position;
			gOwnershipProbeMetrics.finalPosition = position;
		}
		printf("[DeformableMeshOwnershipProbe] init solver=%s mass=%.3f "
		       "position=(%.3f,%.3f,%.3f) surfaceHistory=%u\n",
		       Snippets::getSolverTypeName(gSolverType),
		       double(getOwnershipProbeMass()), double(position.x),
		       double(position.y), double(position.z),
		       gSurfaceHistoryProbe ? 1u : 0u);
	} else if (gHeadlessStress) {
		PxAvbdKinematicShellSetBoxCornerShellEnabled(true);
		createStressBoxGrid();
		printf("[DeformableMeshStress] init solver=%s wavyMesh=true "
		       "shotInterval=%u substeps=%u\n",
		       Snippets::getSolverTypeName(gSolverType), gStressShotIntervalFrames,
		       gDeformSubsteps);
	} else if (gBroadAuthorityProbe) {
		createBroadAuthorityCompound();
	} else if (gCreateStack) {
		createStack(PxTransform(PxVec3(0, 22, 0)), 10, 2.0f);
	}
}

PxBounds3 gBounds;
#ifdef RENDER_SNIPPET
void debugRender()
{
	const PxVec3 c = gBounds.getCenter();
	const PxVec3 e = gBounds.getExtents();

	PxVec3 pts[8];
	pts[0] = c + PxVec3(-e.x, -e.y, e.z);
	pts[1] = c + PxVec3(-e.x,  e.y, e.z);
	pts[2] = c + PxVec3( e.x,  e.y, e.z);
	pts[3] = c + PxVec3( e.x, -e.y, e.z);
	pts[4] = c + PxVec3(-e.x, -e.y, -e.z);
	pts[5] = c + PxVec3(-e.x,  e.y, -e.z);
	pts[6] = c + PxVec3( e.x,  e.y, -e.z);
	pts[7] = c + PxVec3( e.x, -e.y, -e.z);

	PxArray<PxVec3> gContactVertices;
	struct AddQuad
	{
		static void func(PxArray<PxVec3>& v, const PxVec3* pts_, PxU32 index0, PxU32 index1, PxU32 index2, PxU32 index3)
		{
			v.pushBack(pts_[index0]);
			v.pushBack(pts_[index1]);

			v.pushBack(pts_[index1]);
			v.pushBack(pts_[index2]);

			v.pushBack(pts_[index2]);
			v.pushBack(pts_[index3]);

			v.pushBack(pts_[index3]);
			v.pushBack(pts_[index0]);
		}
	};

	AddQuad::func(gContactVertices, pts, 0, 1, 2, 3);
	AddQuad::func(gContactVertices, pts, 4, 5, 6, 7);
	AddQuad::func(gContactVertices, pts, 0, 1, 5, 4);
	AddQuad::func(gContactVertices, pts, 2, 3, 7, 6);

	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glDisable(GL_LIGHTING);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, &gContactVertices[0]);
	glDrawArrays(GL_LINES, 0, GLint(gContactVertices.size()));
	glDisableClientState(GL_VERTEX_ARRAY);
	glEnable(GL_LIGHTING);
}
#endif

static void noteFastImpactsForSubsteps() {
  if (!gScene || gHeadlessSphereShot || gHeadlessStress || !gCreateStack)
    return;
  const PxU32 nbDyn = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  if (nbDyn == 0)
    return;
  PxArray<PxRigidActor *> actors(nbDyn);
  gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
                    reinterpret_cast<PxActor **>(actors.begin()), nbDyn);
  for (PxU32 a = 0; a < nbDyn; ++a) {
    PxRigidDynamic *rb = actors[a] ? actors[a]->is<PxRigidDynamic>() : NULL;
    if (!rb || rb->isSleeping())
      continue;
    if (rb->getLinearVelocity().magnitude() >= gFastImpactSpeedThreshold) {
      gFastImpactSubstepFrames = gFastImpactSubstepHoldFrames;
      break;
    }
  }
}

void stepPhysics(bool interactive)
{
	if (!gScene)
		return;
	const PxReal frameDt = interactive ? (1.0f / 60.0f) : gHeadlessOptions.dt;
	// Substeps for sphere-shot / stress harness. Stack-only interactive stays at 1
	// substep/frame unless a fast body is in flight (space-bar shot).
	if (gHeadlessStress && gSimFrame == gStressNextShotFrame) {
		if (spawnStressShot())
			gStressMetrics.totalShotsFired++;
		gStressNextShotFrame += gStressShotIntervalFrames;
	}
	noteFastImpactsForSubsteps();
	const PxU32 substeps =
	    (gHeadlessSphereShot || gHeadlessStress || !gCreateStack ||
	     gFastImpactSubstepFrames > 0)
	        ? gDeformSubsteps
	        : 1u;
	if (gFastImpactSubstepFrames > 0)
	  --gFastImpactSubstepFrames;
	const PxReal subDt = frameDt / PxReal(substeps);
	const PxReal waveStep = 0.01f / PxReal(substeps);
	for (PxU32 sub = 0; sub < substeps; ++sub) {
		PxVec3 *verts = gMesh->getVerticesForModification();
		if (gSurfaceHistoryProbe) {
			if (gSimFrame >= gSurfaceHistoryWarmupFrames &&
			    gSimFrame <
			        gSurfaceHistoryWarmupFrames + gSurfaceHistoryMotionFrames)
				gSurfaceHistoryOffset += gSurfaceHistorySpeed * subDt;
			updateFlatVertices(verts, gSurfaceHistoryOffset);
		} else if (gBroadAuthorityProbe) {
			// Retain the production wave/refit history that forms usable AL
			// budgets, but reduce its amplitude so the compound contacts enter
			// the rest band together and the first solvable component remains
			// above the matrix-free boundary.
			gTime += waveStep;
			updateVertices(verts, sinf(gTime) * 10.0f);
		} else {
			gTime += waveStep;
			updateVertices(verts, sinf(gTime) * 20.0f);
		}
		gBounds = gMesh->refitBVH();
		gScene->resetFiltering(*gActor);
		if (gSolverType == PxSolverType::eAVBD) {
			PxAvbdKinematicShellUpdateFromMeshGrid(gMesh->getVertices(), gGridSize,
			                                       gGridStep, gActor->getGlobalPose());
		}
		if (!gScene->simulate(subDt)) {
			if (!interactive)
				gRuntimeMetrics.simulateFailures++;
			return;
		}
		gFetchPending = true;
		PxU32 errorState = 0;
		if (!gScene->fetchResults(true, &errorState)) {
			if (!interactive)
				gRuntimeMetrics.fetchFailures++;
			return;
		}
		gFetchPending = false;
		if (!interactive)
			gRuntimeMetrics.fetchErrorState |= errorState;
	}
	++gSimFrame;
	if (!interactive) {
		gRuntimeMetrics.completedFrames++;
		updateRuntimeMetrics();
	}
	if (gHeadlessSphereShot)
		updateSphereShotMetrics();
	else if (gHeadlessStress)
		updateStressHeadlessMetrics();
	else if (gHeadlessOwnershipProbe)
	{
		updateOwnershipProbeMetrics();
		updateSurfaceHistoryMetrics();
	}
	else if (gHeadlessMode && gCreateStack)
		updateStackHeadlessMetrics();
}

static bool drainPendingFetch() {
	if (!gFetchPending)
		return true;
	if (!gScene)
		return false;
	PxU32 errorState = 0;
	if (!gScene->fetchResults(true, &errorState)) {
		gRuntimeMetrics.fetchFailures++;
		return false;
	}
	gRuntimeMetrics.fetchErrorState |= errorState;
	gFetchPending = false;
	return true;
}

void cleanupPhysics(bool interactive)
{
	if (!drainPendingFetch())
		return;
	PxAvbdKinematicShellReset();
	PX_RELEASE(gScene);
	gActor = NULL;
	gShotSphere = NULL;
	gStressShots.clear();
	PX_RELEASE(gMesh);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	if (gExtensionsInitialized) {
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gCleanupCompleted = !gScene && !gMesh && !gMaterial && !gDispatcher &&
	                    !gPhysics && !gPvd && !gFoundation &&
	                    !gFetchPending;
	if (interactive)
		printf("SnippetDeformableMesh done.\n");
}

#ifdef RENDER_SNIPPET
void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case ' ':
		createDynamic(camera, PxSphereGeometry(3.0f),
		              camera.rotate(PxVec3(0, 0, -1)) * 200, 3.0f);
		gFastImpactSubstepFrames = gFastImpactSubstepHoldFrames;
		break;
	}
}
#endif

static const char *getHeadlessCaseName(DeformableHeadlessCase headlessCase) {
  switch (headlessCase) {
  case eCASE_MOVING_MESH_STACK:
    return "moving-mesh-stack";
  case eCASE_SPHERE_SHOT:
    return "sphere-shot";
  case eCASE_STRESS_DIAGNOSTIC:
    return "stress-diagnostic";
  case eCASE_SURFACE_OWNER_LIGHT:
    return "surface-owner-light";
  case eCASE_SURFACE_OWNER_HEAVY:
    return "surface-owner-heavy";
  case eCASE_SHELL_POST_OWNERSHIP:
    return "shell-post-ownership";
  case eCASE_SURFACE_HISTORY:
    return "surface-history";
  case eCASE_NORMAL_POST_OWNERSHIP:
    return "normal-post-ownership";
  case eCASE_BROAD_COMPONENT_AUTHORITY:
    return "broad-component-authority";
  default:
    return "unknown";
  }
}

static bool tryParseHeadlessCase(const char *value,
                                 DeformableHeadlessCase &headlessCase) {
  if (Snippets::equalsIgnoreCase(value, "moving-mesh-stack")) {
    headlessCase = eCASE_MOVING_MESH_STACK;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "sphere-shot")) {
    headlessCase = eCASE_SPHERE_SHOT;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "stress-diagnostic")) {
    headlessCase = eCASE_STRESS_DIAGNOSTIC;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "surface-owner-light")) {
    headlessCase = eCASE_SURFACE_OWNER_LIGHT;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "surface-owner-heavy")) {
    headlessCase = eCASE_SURFACE_OWNER_HEAVY;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "shell-post-ownership")) {
    headlessCase = eCASE_SHELL_POST_OWNERSHIP;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "surface-history")) {
    headlessCase = eCASE_SURFACE_HISTORY;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "normal-post-ownership")) {
    headlessCase = eCASE_NORMAL_POST_OWNERSHIP;
    return true;
  }
  if (Snippets::equalsIgnoreCase(value, "broad-component-authority")) {
    headlessCase = eCASE_BROAD_COMPONENT_AUTHORITY;
    return true;
  }
  return false;
}

struct GateEvaluation {
  PxU32 exitCode;
  const char *status;
  const char *reason;
  PxU32 nonFinite;
  PxU32 dynamicBodies;
  PxU32 expectedStressShots;
  PxReal finalSphereGap;
  PxReal sphereLateralDrift;
  PxReal sphereResponseFraction;

  GateEvaluation()
      : exitCode(Snippets::eHEADLESS_PASS), status("PASS"), reason("none"),
        nonFinite(0), dynamicBodies(0), expectedStressShots(0),
        finalSphereGap(0.0f),
        sphereLateralDrift(0.0f), sphereResponseFraction(0.0f) {}
};

static void setGateError(GateEvaluation &evaluation, const char *reason) {
  if (evaluation.exitCode == Snippets::eHEADLESS_CONFIG_ERROR)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_CONFIG_ERROR;
  evaluation.status = "ERROR";
  evaluation.reason = reason;
}

static void setGateFailure(GateEvaluation &evaluation, const char *reason) {
  if (evaluation.exitCode != Snippets::eHEADLESS_PASS)
    return;
  evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
  evaluation.status = "FAIL";
  evaluation.reason = reason;
}

static GateEvaluation evaluateGate(bool sceneQueriesAllowed = true) {
  GateEvaluation evaluation;
  evaluation.nonFinite = gRuntimeMetrics.nonFinite + gStackMetrics.nanBodies +
                         gStressMetrics.nanEvents +
                         (gSphereShotMetrics.nanDetected ? 1u : 0u);
  if (gHeadlessCase == eCASE_STRESS_DIAGNOSTIC)
    evaluation.expectedStressShots =
        (gHeadlessOptions.frames - 1u) / gStressShotIntervalFrames + 1u;
  if (!sceneQueriesAllowed) {
    setGateError(evaluation, "fetch_results");
    return evaluation;
  }
  if (gScene)
    evaluation.dynamicBodies =
        gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
  if (gShotSphere) {
    const PxVec3 finalPosition = gShotSphere->getGlobalPose().p;
    if (isInsideMeshFootprint(finalPosition, gShotSphereRadius))
      evaluation.finalSphereGap =
          (finalPosition.y - gShotSphereRadius) -
          sampleMeshSurfaceY(finalPosition) - gMeshRestOffset;
    evaluation.sphereLateralDrift =
        PxSqrt(finalPosition.x * finalPosition.x +
               finalPosition.z * finalPosition.z);
  }
  evaluation.sphereResponseFraction =
      gSphereShotMetrics.maxImpactAxisVelocityDelta / gShotSpeedY;
  if (gInitializationFailed)
    setGateError(evaluation, "initialization");
  if (gRuntimeMetrics.simulateFailures ||
      gRuntimeMetrics.completedFrames != gHeadlessOptions.frames ||
      gRuntimeMetrics.fetchFailures)
    setGateError(evaluation, "incomplete_simulation");
  if (gErrorCallback.getFatalCount() || gRuntimeMetrics.fetchErrorState)
    setGateFailure(evaluation, "physx_error");
  if (evaluation.nonFinite)
    setGateFailure(evaluation, "non_finite");
  if (gRuntimeMetrics.maxQuaternionNormError > 1e-3f)
    setGateFailure(evaluation, "quaternion_norm");
  if (gRuntimeMetrics.maxAbsPosition > 100000.0f ||
      gRuntimeMetrics.maxLinearSpeed > 10000.0f ||
      gRuntimeMetrics.maxAngularSpeed > 10000.0f)
    setGateFailure(evaluation, "runaway");
  if (gHeadlessCase != eCASE_STRESS_DIAGNOSTIC &&
      gRuntimeMetrics.maxFullFallThroughBodies)
    setGateFailure(evaluation, "full_fall_through");

  if (gHeadlessCase == eCASE_MOVING_MESH_STACK) {
    if (evaluation.dynamicBodies != 55)
      setGateError(evaluation, "stack_body_count");
    if (gStackMetrics.maxSpeed >= 50.0f ||
        gStackMetrics.maxSettledSpeed >= 8.0f)
      setGateFailure(evaluation, "stack_speed");
    if (gStackMetrics.settledSunkBoxes)
      setGateFailure(evaluation, "settled_sink");
    if (gStackMetrics.maxOutOfFootprintBoxes)
      setGateFailure(evaluation, "stack_out_of_footprint");
  } else if (gHeadlessCase == eCASE_BROAD_COMPONENT_AUTHORITY) {
    if (evaluation.dynamicBodies != 2)
      setGateError(evaluation, "broad_authority_body_count");
  } else if (gHeadlessCase == eCASE_SPHERE_SHOT) {
    if (evaluation.dynamicBodies != 1 || !gShotSphere)
      setGateError(evaluation, "sphere_body_count");
    if (!gSphereShotMetrics.contactEvents ||
        !gSphereShotMetrics.contactPoints)
      setGateFailure(evaluation, "missing_mesh_contact");
    if (!gSphereShotMetrics.responseSamples ||
        evaluation.sphereResponseFraction < gMinSphereResponseFraction)
      setGateFailure(evaluation, "missing_contact_response");
    if (gSphereShotMetrics.outOfFootprintFrames)
      setGateFailure(evaluation, "sphere_out_of_footprint");
    if (gSphereShotMetrics.firstContactFrame == PX_MAX_U32 ||
        gSphereShotMetrics.firstContactFrame +
                gSphereResponseWindowFrames - 1u >
            gRuntimeMetrics.completedFrames)
      setGateFailure(evaluation, "incomplete_contact_observation");
    if (gSphereShotMetrics.maxImpactGeomOverlap > 1.0f)
      setGateFailure(evaluation, "impact_penetration");
    if (gSphereShotMetrics.maxSettledGeomOverlap > 1.55f)
      setGateFailure(evaluation, "settled_penetration");
    if (evaluation.sphereLateralDrift > 15.0f)
      setGateFailure(evaluation, "lateral_drift");
    if (gSphereShotMetrics.settledProximityFrames < 30)
      setGateFailure(evaluation, "not_settled");
    if (evaluation.finalSphereGap > 3.0f ||
        gSphereShotMetrics.maxSettledAirborneGap > 8.0f)
      setGateFailure(evaluation, "airborne");
  } else if (gHeadlessOwnershipProbe) {
    if (evaluation.dynamicBodies != 1 || !gOwnershipProbeBox)
      setGateError(evaluation, "ownership_probe_body_count");
    if (gOwnershipProbeMetrics.minBottomGap == PX_MAX_F32 ||
        gOwnershipProbeMetrics.maxBottomGap == -PX_MAX_F32)
      setGateError(evaluation, "ownership_probe_missing_sample");
    if (gOwnershipProbeMetrics.outOfFootprintFrames)
      setGateFailure(evaluation, "ownership_probe_out_of_footprint");
    if (gSurfaceHistoryProbe) {
      if (!gSurfaceHistoryMetrics.motionStartCaptured ||
          gSurfaceHistoryMetrics.motionSamples !=
              gSurfaceHistoryMotionFrames)
        setGateError(evaluation, "surface_history_motion_samples");
      if (!gSurfaceHistoryMetrics.contactEvents ||
          !gSurfaceHistoryMetrics.contactPoints)
        setGateError(evaluation, "surface_history_contact_witness");
      const PxReal surfaceRise =
          gSurfaceHistoryMetrics.surfaceYAtMotionEnd -
          gSurfaceHistoryMetrics.surfaceYAtMotionStart;
      if (!PxIsFinite(surfaceRise) || surfaceRise < 0.9f)
        setGateError(evaluation, "surface_history_mesh_motion");
    }
  } else {
    if (evaluation.dynamicBodies !=
        gStressGridX * gStressGridZ + gStressMetrics.totalShotsFired)
      setGateError(evaluation, "stress_body_count");
    if (gStressMetrics.totalShotsFired != evaluation.expectedStressShots)
      setGateError(evaluation, "stress_shot_launch");
    if (!gStressMetrics.contactEvents || !gStressMetrics.contactPoints)
      setGateError(evaluation, "stress_no_contact_witness");
  }
  return evaluation;
}

static void printGateDetails(const GateEvaluation &evaluation) {
  if (gHeadlessCase == eCASE_SPHERE_SHOT) {
    printSphereShotSummary();
  } else if (gSurfaceHistoryProbe) {
    const PxReal invSamples =
        gSurfaceHistoryMetrics.motionSamples
            ? 1.0f / PxReal(gSurfaceHistoryMetrics.motionSamples)
            : 0.0f;
    const PxReal surfaceRise =
        gSurfaceHistoryMetrics.surfaceYAtMotionEnd -
        gSurfaceHistoryMetrics.surfaceYAtMotionStart;
    const PxReal bodyRise = gSurfaceHistoryMetrics.bodyYAtMotionEnd -
                            gSurfaceHistoryMetrics.bodyYAtMotionStart;
    const PxReal meanBodyVelocityY =
        gSurfaceHistoryMetrics.sumBodyVelocityY * invSamples;
    printf("[DeformableMeshSurfaceHistory] solver=%s contactEvents=%u "
           "contactPoints=%u motionSamples=%u targetVelocityY=%.9g "
           "surfaceRise=%.9g bodyRise=%.9g poseFollowError=%.9g "
           "meanBodyVelocityY=%.9g meanAbsRelativeVelocityY=%.9g "
           "minBodyVelocityY=%.9g maxBodyVelocityY=%.9g "
           "maxAbsRelativeVelocityY=%.9g velocityFollowRatio=%.9g\n",
           Snippets::getSolverTypeName(gSolverType),
           gSurfaceHistoryMetrics.contactEvents,
           gSurfaceHistoryMetrics.contactPoints,
           gSurfaceHistoryMetrics.motionSamples,
           double(gSurfaceHistorySpeed), double(surfaceRise), double(bodyRise),
           double(PxAbs(bodyRise - surfaceRise)), double(meanBodyVelocityY),
           double(gSurfaceHistoryMetrics.sumAbsRelativeVelocityY * invSamples),
           double(gSurfaceHistoryMetrics.minBodyVelocityY),
           double(gSurfaceHistoryMetrics.maxBodyVelocityY),
           double(gSurfaceHistoryMetrics.maxAbsRelativeVelocityY),
           double(meanBodyVelocityY / gSurfaceHistorySpeed));
  } else if (gHeadlessOwnershipProbe) {
    const PxVec3 displacement =
        gOwnershipProbeMetrics.finalPosition -
        gOwnershipProbeMetrics.initialPosition;
    printf("[DeformableMeshOwnershipProbe] solver=%s mass=%.3f "
           "maxHorizontalSpeed=%.9g maxAngularSpeed=%.9g "
           "minBottomGap=%.9g maxBottomGap=%.9g "
           "finalDisplacement=(%.9g,%.9g,%.9g) outOfFootprintFrames=%u\n",
           Snippets::getSolverTypeName(gSolverType),
           double(getOwnershipProbeMass()),
           double(gOwnershipProbeMetrics.maxHorizontalSpeed),
           double(gOwnershipProbeMetrics.maxAngularSpeed),
           double(gOwnershipProbeMetrics.minBottomGap),
           double(gOwnershipProbeMetrics.maxBottomGap),
           double(displacement.x), double(displacement.y),
           double(displacement.z),
           gOwnershipProbeMetrics.outOfFootprintFrames);
  } else if (gHeadlessCase == eCASE_STRESS_DIAGNOSTIC) {
    printStressHeadlessSummary();
  } else {
    printf("[DeformableMeshStack] solver=%s frames=%u numBoxes=%u "
           "maxSpeed=%.4f maxSettledSpeed=%.4f nanBodies=%u\n",
           Snippets::getSolverTypeName(gSolverType), gHeadlessFrameCount,
           evaluation.dynamicBodies, gStackMetrics.maxSpeed,
           gStackMetrics.maxSettledSpeed, gStackMetrics.nanBodies);
    printf("[DeformableMeshStack] maxSpreadXZ=%.4f maxSettledSpreadXZ=%.4f "
           "maxWorldY=%.4f maxSettledWorldY=%.4f minRelToSurface=%.4f "
           "settledSunkBoxes=%u maxOutOfFootprintBoxes=%u "
           "spreadValidation=DIAGNOSTIC\n",
           gStackMetrics.maxSpreadXZ, gStackMetrics.maxSettledSpreadXZ,
           gStackMetrics.maxWorldY, gStackMetrics.maxSettledWorldY,
           gStackMetrics.minRelToSurface, gStackMetrics.settledSunkBoxes,
           gStackMetrics.maxOutOfFootprintBoxes);
  }
}

static PxReal printableMetric(PxReal value) {
  return PxIsFinite(value) ? value : 0.0f;
}

static void printGateResult(const GateEvaluation &evaluation,
                            PxU32 physicsErrors, PxU32 physicsWarnings) {
  const char *validation =
      (gHeadlessCase == eCASE_STRESS_DIAGNOSTIC ||
       gHeadlessOwnershipProbe || gBroadAuthorityProbe)
                               ? "PROBE"
                               : "GATED";
  const bool stressDeepSinkObserved =
      gHeadlessCase == eCASE_STRESS_DIAGNOSTIC &&
      (gStressMetrics.maxSunkBoxes ||
       gStressMetrics.maxPassThroughShots ||
       gRuntimeMetrics.maxFullFallThroughBodies);
  const char *probeFinding =
      gSurfaceHistoryProbe
          ? "moving-surface-point-history"
          : gNormalPostOwnershipProbe
          ? "normal-post-stage-overlap"
          : stressDeepSinkObserved
          ? "deep-sink-observed"
          : (gHeadlessCase == eCASE_STRESS_DIAGNOSTIC &&
                     (gStressMetrics.maxOutOfFootprintShots ||
                      gStressMetrics.maxOutOfFootprintBoxes)
                 ? "out-of-footprint-observed"
                 : "none");
  const bool sphereContactObserved =
      gSphereShotMetrics.firstContactFrame != PX_MAX_U32;
  const bool stressMetricsObserved =
      gHeadlessCase == eCASE_STRESS_DIAGNOSTIC &&
      gRuntimeMetrics.completedFrames > 0;
  const PxU32 sphereFirstContactFrame =
      sphereContactObserved ? gSphereShotMetrics.firstContactFrame : 0u;
  const PxReal stressWorstMinRel =
      stressMetricsObserved ? gStressMetrics.worstMinRelToSurface : 0.0f;
  const bool ownershipMetricsObserved =
      gHeadlessOwnershipProbe && gRuntimeMetrics.completedFrames > 0;
  const PxVec3 ownershipDisplacement =
      ownershipMetricsObserved
          ? gOwnershipProbeMetrics.finalPosition -
                gOwnershipProbeMetrics.initialPosition
          : PxVec3(0.0f);
  const PxReal historyInvSamples =
      gSurfaceHistoryMetrics.motionSamples
          ? 1.0f / PxReal(gSurfaceHistoryMetrics.motionSamples)
          : 0.0f;
  const PxReal historySurfaceRise =
      gSurfaceHistoryMetrics.surfaceYAtMotionEnd -
      gSurfaceHistoryMetrics.surfaceYAtMotionStart;
  const PxReal historyBodyRise = gSurfaceHistoryMetrics.bodyYAtMotionEnd -
                                 gSurfaceHistoryMetrics.bodyYAtMotionStart;
  const PxReal historyMeanBodyVelocityY =
      gSurfaceHistoryMetrics.sumBodyVelocityY * historyInvSamples;
  const PxReal historyMeanAbsRelativeVelocityY =
      gSurfaceHistoryMetrics.sumAbsRelativeVelocityY * historyInvSamples;
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetDeformableMesh case=%s solver=%s "
      "execution=%s requestedFrames=%u completedFrames=%u dt=%.9g seed=%u "
      "dispatcherThreads=%u "
      "capability=SUPPORTED validation=%s status=%s reason=%s "
      "sceneClass=rigid-on-moving-triangle-mesh meshActor=rigid-static "
      "meshMotion=vertex-update softBody=0 cloth=0 probeFinding=%s "
      "zeroPenetrationClaim=0 "
      "nonFinite=%u physicsErrors=%u physicsWarnings=%u simulateFailures=%u "
      "fetchFailures=%u "
      "fetchErrorState=%u cleanupCompleted=%u dynamicBodies=%u "
      "maxFullFallThroughBodies=%u maxQuaternionNormError=%.9g "
      "maxAbsPosition=%.9g maxLinearSpeed=%.9g maxAngularSpeed=%.9g "
      "maxSpeed=%.9g maxSettledSpeed=%.9g settledSunkBoxes=%u "
      "stackMaxOutOfFootprintBoxes=%u maxSettledSpreadXZ=%.9g "
      "sphereContactEvents=%u "
      "sphereContactPoints=%u sphereFirstContactObserved=%u "
      "sphereFirstContactFrame=%u sphereOutOfFootprintFrames=%u "
      "sphereResponseSamples=%u sphereResponseFraction=%.9g "
      "sphereFinalGap=%.9g sphereLateralDrift=%.9g "
      "maxImpactVerticalGeomOverlap=%.9g "
      "maxSettledVerticalGeomOverlap=%.9g "
      "maxImpactRestOffsetProximity=%.9g "
      "maxSettledRestOffsetProximity=%.9g settledProximityFrames=%u "
      "stressShots=%u expectedStressShots=%u stressContactEvents=%u "
      "stressContactPoints=%u stressMaxSunkBoxes=%u "
      "stressMetricsObserved=%u stressWorstMinRelToSurface=%.9g "
      "stressMaxPassThroughShots=%u stressMaxOutOfFootprintShots=%u "
      "stressMaxOutOfFootprintBoxes=%u "
      "ownershipMetricsObserved=%u ownershipProbeMass=%.9g "
      "ownershipMaxHorizontalSpeed=%.9g ownershipMaxAngularSpeed=%.9g "
      "ownershipMinBottomGap=%.9g ownershipMaxBottomGap=%.9g "
      "ownershipFinalDx=%.9g ownershipFinalDy=%.9g ownershipFinalDz=%.9g "
      "ownershipOutOfFootprintFrames=%u "
      "historyMetricsObserved=%u historyContactEvents=%u "
      "historyContactPoints=%u historyMotionSamples=%u "
      "historyTargetVelocityY=%.9g historySurfaceRise=%.9g "
      "historyBodyRise=%.9g historyPoseFollowError=%.9g "
      "historyMeanBodyVelocityY=%.9g "
      "historyMeanAbsRelativeVelocityY=%.9g "
      "historyMinBodyVelocityY=%.9g historyMaxBodyVelocityY=%.9g "
      "historyMaxAbsRelativeVelocityY=%.9g "
      "historyVelocityFollowRatio=%.9g "
      "minSphereResponseFraction=%.9g responseWindowFrames=%u\n",
      getHeadlessCaseName(gHeadlessCase),
      Snippets::getSolverTypeName(gHeadlessOptions.solverType),
      Snippets::getExecutionName(gHeadlessOptions.execution),
      gHeadlessOptions.frames, gRuntimeMetrics.completedFrames,
      double(gHeadlessOptions.dt), gHeadlessOptions.seed,
      gHeadlessOptions.dispatcherThreads, validation, evaluation.status,
      evaluation.reason, probeFinding,
      evaluation.nonFinite, physicsErrors, physicsWarnings,
      gRuntimeMetrics.simulateFailures, gRuntimeMetrics.fetchFailures,
      gRuntimeMetrics.fetchErrorState, gCleanupCompleted ? 1u : 0u,
      evaluation.dynamicBodies, gRuntimeMetrics.maxFullFallThroughBodies,
      double(printableMetric(gRuntimeMetrics.maxQuaternionNormError)),
      double(printableMetric(gRuntimeMetrics.maxAbsPosition)),
      double(printableMetric(gRuntimeMetrics.maxLinearSpeed)),
      double(printableMetric(gRuntimeMetrics.maxAngularSpeed)),
      double(printableMetric(gStackMetrics.maxSpeed)),
      double(printableMetric(gStackMetrics.maxSettledSpeed)),
      gStackMetrics.settledSunkBoxes,
      gStackMetrics.maxOutOfFootprintBoxes,
      double(printableMetric(gStackMetrics.maxSettledSpreadXZ)),
      gSphereShotMetrics.contactEvents, gSphereShotMetrics.contactPoints,
      sphereContactObserved ? 1u : 0u, sphereFirstContactFrame,
      gSphereShotMetrics.outOfFootprintFrames,
      gSphereShotMetrics.responseSamples,
      double(printableMetric(evaluation.sphereResponseFraction)),
      double(printableMetric(evaluation.finalSphereGap)),
      double(printableMetric(evaluation.sphereLateralDrift)),
      double(printableMetric(gSphereShotMetrics.maxImpactGeomOverlap)),
      double(printableMetric(gSphereShotMetrics.maxSettledGeomOverlap)),
      double(printableMetric(
          gSphereShotMetrics.maxImpactRestOffsetProximity)),
      double(printableMetric(
          gSphereShotMetrics.maxSettledRestOffsetProximity)),
      gSphereShotMetrics.settledProximityFrames,
      gStressMetrics.totalShotsFired, evaluation.expectedStressShots,
      gStressMetrics.contactEvents, gStressMetrics.contactPoints,
      gStressMetrics.maxSunkBoxes, stressMetricsObserved ? 1u : 0u,
      double(printableMetric(stressWorstMinRel)),
      gStressMetrics.maxPassThroughShots,
      gStressMetrics.maxOutOfFootprintShots,
      gStressMetrics.maxOutOfFootprintBoxes,
      ownershipMetricsObserved ? 1u : 0u,
      double(ownershipMetricsObserved ? getOwnershipProbeMass() : 0.0f),
      double(ownershipMetricsObserved
                 ? gOwnershipProbeMetrics.maxHorizontalSpeed
                 : 0.0f),
      double(ownershipMetricsObserved ? gOwnershipProbeMetrics.maxAngularSpeed
                                      : 0.0f),
      double(ownershipMetricsObserved ? gOwnershipProbeMetrics.minBottomGap
                                      : 0.0f),
      double(ownershipMetricsObserved ? gOwnershipProbeMetrics.maxBottomGap
                                      : 0.0f),
      double(ownershipDisplacement.x), double(ownershipDisplacement.y),
      double(ownershipDisplacement.z),
      ownershipMetricsObserved
          ? gOwnershipProbeMetrics.outOfFootprintFrames
          : 0u,
      gSurfaceHistoryProbe && gRuntimeMetrics.completedFrames > 0 ? 1u : 0u,
      gSurfaceHistoryMetrics.contactEvents,
      gSurfaceHistoryMetrics.contactPoints,
      gSurfaceHistoryMetrics.motionSamples,
      double(gSurfaceHistoryProbe ? gSurfaceHistorySpeed : 0.0f),
      double(printableMetric(historySurfaceRise)),
      double(printableMetric(historyBodyRise)),
      double(printableMetric(PxAbs(historyBodyRise - historySurfaceRise))),
      double(printableMetric(historyMeanBodyVelocityY)),
      double(printableMetric(historyMeanAbsRelativeVelocityY)),
      double(gSurfaceHistoryProbe
                 ? printableMetric(gSurfaceHistoryMetrics.minBodyVelocityY)
                 : 0.0f),
      double(gSurfaceHistoryProbe
                 ? printableMetric(gSurfaceHistoryMetrics.maxBodyVelocityY)
                 : 0.0f),
      double(printableMetric(
          gSurfaceHistoryMetrics.maxAbsRelativeVelocityY)),
      double(printableMetric(
          historyMeanBodyVelocityY / gSurfaceHistorySpeed)),
      double(gMinSphereResponseFraction), gSphereResponseWindowFrames);
}

static int reportConfigurationError(const Snippets::HeadlessOptions &options,
                                    const char *message) {
  const char *validation =
      (Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "stress-diagnostic") ||
       Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "surface-owner-light") ||
       Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "surface-owner-heavy") ||
       Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "shell-post-ownership") ||
       Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "surface-history") ||
       Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "normal-post-ownership") ||
       Snippets::equalsIgnoreCase(options.caseName.c_str(),
                                  "broad-component-authority"))
          ? "PROBE"
          : "GATED";
  printf("[AVBD_GATE_ERROR] snippet=SnippetDeformableMesh message=%s\n",
         message);
  printf(
      "[AVBD_GATE] schema=1 snippet=SnippetDeformableMesh case=config-error "
      "solver=%s execution=%s requestedFrames=%u completedFrames=0 dt=%.9g "
      "seed=%u dispatcherThreads=%u capability=SUPPORTED validation=%s "
      "status=ERROR "
      "reason=config nonFinite=0 physicsErrors=0 simulateFailures=0 "
      "physicsWarnings=0 fetchFailures=0\n",
      Snippets::getSolverTypeName(options.solverType),
      Snippets::getExecutionName(options.execution), options.frames,
      double(options.dt), options.seed, options.dispatcherThreads, validation);
  return Snippets::eHEADLESS_CONFIG_ERROR;
}

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  Snippets::HeadlessOptions defaults;
  defaults.caseName = "moving-mesh-stack";
  defaults.frames = 7200;
  defaults.seed = 1;
  defaults.dispatcherThreads = 2;
  defaults.dt = 1.0f / 60.0f;

  Snippets::HeadlessOptions options;
  std::string parseError;
  if (!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options,
                                            parseError))
    return reportConfigurationError(options, parseError.c_str());

  bool legacySphere = false;
  bool legacyStress = false;
  bool caseSeen = false;
  bool headlessOnlyOptionSeen = false;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!arg)
      continue;
    if (Snippets::isCommonHeadlessOption(arg)) {
      if (Snippets::hasOptionPrefix(arg, "--case=") ||
          Snippets::hasOptionPrefix(arg, "--scenario="))
        caseSeen = true;
      if (std::strcmp(arg, "--headless") != 0 &&
          !Snippets::hasOptionPrefix(arg, "--solver="))
        headlessOnlyOptionSeen = true;
      continue;
    }
    if (std::strcmp(arg, "--headless-sphere-shot") == 0) {
      if (legacySphere)
        return reportConfigurationError(options,
                                        "duplicate_legacy_sphere_alias");
      legacySphere = true;
      options.headless = true;
      options.caseName = "sphere-shot";
      headlessOnlyOptionSeen = true;
      continue;
    }
    if (std::strcmp(arg, "--headless-stress") == 0) {
      if (legacyStress)
        return reportConfigurationError(options,
                                        "duplicate_legacy_stress_alias");
      legacyStress = true;
      options.headless = true;
      options.caseName = "stress-diagnostic";
      headlessOnlyOptionSeen = true;
      continue;
    }
    return reportConfigurationError(options, "unknown_argument");
  }

#ifndef RENDER_SNIPPET
  options.headless = true;
#endif

  if (legacySphere && legacyStress)
    return reportConfigurationError(options, "conflicting_legacy_aliases");
  if (caseSeen && (legacySphere || legacyStress))
    return reportConfigurationError(options, "legacy_alias_conflicts_case");
  if (legacySphere)
    options.caseName = "sphere-shot";
  else if (legacyStress)
    options.caseName = "stress-diagnostic";

  DeformableHeadlessCase headlessCase = eCASE_MOVING_MESH_STACK;
  if (!tryParseHeadlessCase(options.caseName.c_str(), headlessCase))
    return reportConfigurationError(options, "invalid_--case_value");
  options.caseName = getHeadlessCaseName(headlessCase);
  if (!options.headless && headlessOnlyOptionSeen)
    return reportConfigurationError(options, "gate_option_requires_--headless");

  if (!options.framesExplicit) {
    if (headlessCase == eCASE_SPHERE_SHOT)
      options.frames = 180;
    else if (headlessCase == eCASE_STRESS_DIAGNOSTIC)
      options.frames = 600;
    else if (headlessCase == eCASE_SURFACE_OWNER_LIGHT ||
             headlessCase == eCASE_SURFACE_OWNER_HEAVY ||
             headlessCase == eCASE_SHELL_POST_OWNERSHIP)
      options.frames = 240;
    else if (headlessCase == eCASE_SURFACE_HISTORY)
      options.frames = 180;
    else if (headlessCase == eCASE_NORMAL_POST_OWNERSHIP)
      options.frames = 240;
    else if (headlessCase == eCASE_BROAD_COMPONENT_AUTHORITY)
      options.frames = 16;
    else
      options.frames = 7200;
  }
  if (headlessCase == eCASE_MOVING_MESH_STACK && options.frames < 7200)
    return reportConfigurationError(options,
                                    "stack_frames_must_be_at_least_7200");
  if (headlessCase == eCASE_SPHERE_SHOT && options.frames < 180)
    return reportConfigurationError(options,
                                    "sphere_frames_must_be_at_least_180");
  if (headlessCase == eCASE_STRESS_DIAGNOSTIC && options.frames < 600)
    return reportConfigurationError(options,
                                    "stress_frames_must_be_at_least_600");
  if ((headlessCase == eCASE_SURFACE_OWNER_LIGHT ||
       headlessCase == eCASE_SURFACE_OWNER_HEAVY ||
       headlessCase == eCASE_SHELL_POST_OWNERSHIP) &&
      options.frames < 240)
    return reportConfigurationError(
        options, "surface_owner_frames_must_be_at_least_240");
  if (headlessCase == eCASE_SURFACE_HISTORY && options.frames < 180)
    return reportConfigurationError(
        options, "surface_history_frames_must_be_at_least_180");
  if (headlessCase == eCASE_NORMAL_POST_OWNERSHIP && options.frames < 240)
    return reportConfigurationError(
        options, "normal_post_frames_must_be_at_least_240");
	if (headlessCase == eCASE_BROAD_COMPONENT_AUTHORITY &&
	    options.frames < 16)
		return reportConfigurationError(
		    options, "broad_authority_frames_must_be_at_least_16");
  if (options.execution == Snippets::eHEADLESS_SEQUENTIAL &&
      options.solverType != PxSolverType::eAVBD)
    return reportConfigurationError(options, "sequential_requires_avbd");
  if (PxAbs(options.dt - (1.0f / 60.0f)) > 1e-7f)
    return reportConfigurationError(options, "dt_requires_60hz_calibration");
  if (!Snippets::applyExecutionEnvironment(options))
    return reportConfigurationError(options, "execution_environment_failed");

  gHeadlessOptions = options;
  gHeadlessCase = headlessCase;
  gSolverType = options.solverType;
  gHeadlessMode = options.headless;
  gHeadlessSphereShot = headlessCase == eCASE_SPHERE_SHOT;
  gHeadlessStress = headlessCase == eCASE_STRESS_DIAGNOSTIC;
  gHeadlessOwnershipProbe =
      headlessCase == eCASE_SURFACE_OWNER_LIGHT ||
      headlessCase == eCASE_SURFACE_OWNER_HEAVY ||
      headlessCase == eCASE_SHELL_POST_OWNERSHIP ||
      headlessCase == eCASE_SURFACE_HISTORY ||
      headlessCase == eCASE_NORMAL_POST_OWNERSHIP;
  gOwnershipProbeHeavy = headlessCase == eCASE_SURFACE_OWNER_HEAVY;
  gShellPostOwnershipProbe = headlessCase == eCASE_SHELL_POST_OWNERSHIP;
  gSurfaceHistoryProbe = headlessCase == eCASE_SURFACE_HISTORY;
  gNormalPostOwnershipProbe =
      headlessCase == eCASE_NORMAL_POST_OWNERSHIP;
  gBroadAuthorityProbe =
      headlessCase == eCASE_BROAD_COMPONENT_AUTHORITY;
  gCreateStack = headlessCase == eCASE_MOVING_MESH_STACK;
  gHeadlessFrameCount = options.frames;
  resetRuntimeState();

#ifdef RENDER_SNIPPET
  if (!gHeadlessMode) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  Snippets::printHeadlessConfig("SnippetDeformableMesh", gHeadlessOptions);
  initPhysics(false);
  for (PxU32 i = 0; i < gHeadlessOptions.frames &&
                    !gInitializationFailed &&
                    !gRuntimeMetrics.simulateFailures &&
                    !gRuntimeMetrics.fetchFailures;
       ++i)
    stepPhysics(false);

  // A failed blocking fetch may still leave the scene in the simulate/fetch
  // interval.  Drain it before any actor or articulation query.  If that is
  // impossible, create a lifecycle-only ERROR snapshot without reading scene.
  const bool sceneQueriesSafe = drainPendingFetch();
  GateEvaluation evaluation = evaluateGate(sceneQueriesSafe);
  if (sceneQueriesSafe)
    printGateDetails(evaluation);
  cleanupPhysics(false);
  if (!gCleanupCompleted)
    setGateError(evaluation, "teardown");
  const PxU32 physicsErrors = gErrorCallback.getFatalCount();
  if (physicsErrors &&
      evaluation.exitCode != Snippets::eHEADLESS_CONFIG_ERROR) {
    evaluation.exitCode = Snippets::eHEADLESS_GATE_FAILED;
    evaluation.status = "FAIL";
    evaluation.reason = "physx_error";
  }
  printGateResult(evaluation, physicsErrors,
                  gErrorCallback.getWarningCount());
  return static_cast<int>(evaluation.exitCode);
}
