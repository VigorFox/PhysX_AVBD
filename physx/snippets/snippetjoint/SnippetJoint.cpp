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
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

// ****************************************************************************
// This snippet illustrates simple use of joints in physx
//
// It creates a chain of objects joined by limited spherical joints, a chain
// joined by fixed joints which is breakable, and a chain of damped D6 joints
// ****************************************************************************

#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"
#include "PxPhysicsAPI.h"
#include <ctype.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

static PxDefaultAllocator gAllocator;
static PxDefaultErrorCallback gErrorCallback;
static PxFoundation *gFoundation = NULL;
static PxPhysics *gPhysics = NULL;
static PxDefaultCpuDispatcher *gDispatcher = NULL;
static PxScene *gScene = NULL;
static PxMaterial *gMaterial = NULL;
static PxPvd *gPvd = NULL;
static PxSolverType::Enum gSolverType = PxSolverType::eAVBD;

static std::vector<PxRigidDynamic *> gRevoluteChainBodies;
static std::vector<PxRevoluteJoint *> gRevoluteChainJoints;
static std::vector<PxRigidDynamic *> gFixedChainBodies;
static std::vector<PxFixedJoint *> gFixedChainJoints;
static std::vector<PxU8> gFixedBreakReported;
static std::vector<PxRigidDynamic *> gPrismaticChainBodies;
static std::vector<PxPrismaticJoint *> gPrismaticChainJoints;
static std::vector<PxVec3> gPrismaticRestPositions;

static void getJointWorldAxes(PxRevoluteJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1);
static void getJointWorldAxes(PxPrismaticJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1);

struct RevoluteJitterStats {
  PxU32 frame = 0;
  PxReal maxTailLateral = 0.0f;
  PxReal sumW4Early = 0.0f, sumW5Early = 0.0f;
  PxReal sumW4Late = 0.0f, sumW5Late = 0.0f;
  PxReal sumW4PerpEarly = 0.0f, sumW5PerpEarly = 0.0f;
  PxReal sumW4PerpLate = 0.0f, sumW5PerpLate = 0.0f;
  PxReal maxW4PerpLate = 0.0f, maxW5PerpLate = 0.0f;
  PxU32 awake4Late = 0, awake5Late = 0;
  PxU32 cntEarly = 0, cntLate = 0;
  PxReal prevAngle3 = 0.0f, prevAngle4 = 0.0f;
  PxReal prevD3 = 0.0f, prevD4 = 0.0f;
  PxU32 flip3 = 0, flip4 = 0;
  PxReal maxAbsAngle3 = 0.0f, maxAbsAngle4 = 0.0f;
  PxReal maxAxisMisalign3Deg = 0.0f, maxAxisMisalign4Deg = 0.0f;
};

static RevoluteJitterStats gRevoluteStats;

struct FixedChainStats {
  PxReal maxLinearForce = 0.0f;
  PxReal maxAngularForce = 0.0f;
  PxU32 firstBrokenFrame = PX_MAX_U32;
  PxU32 brokenCount = 0;
};

static FixedChainStats gFixedStats;

struct PrismaticDriftStats {
  PxReal maxTailTransverse = 0.0f;
  PxReal sumTailEarly = 0.0f;
  PxReal sumTailLate = 0.0f;
  PxReal sumTailAngVelEarly = 0.0f;
  PxReal sumTailAngVelLate = 0.0f;
  PxReal maxJointAxisMisalignDeg = 0.0f;
  PxU32 cntEarly = 0;
  PxU32 cntLate = 0;
};

static PrismaticDriftStats gPrismaticStats;

static bool shouldDumpRevoluteState(PxU32 frame) {
  if (frame <= 180)
    return (frame % 30) == 0;
  if (frame >= 200 && frame <= 420)
    return (frame % 10) == 0;
  if (frame >= 900 && frame <= 1300)
    return (frame % 10) == 0;
  return (frame % 120) == 0;
}

static const char *getSolverTypeName(PxSolverType::Enum solverType) {
  switch (solverType) {
  case PxSolverType::ePGS:
    return "pgs";
  case PxSolverType::eTGS:
    return "tgs";
  case PxSolverType::eAVBD:
    return "avbd";
  default:
    return "unknown";
  }
}

static bool tryParseSolverType(const char *value,
                               PxSolverType::Enum &solverType) {
  if (!value || !value[0])
    return false;

  if (_stricmp(value, "pgs") == 0) {
    solverType = PxSolverType::ePGS;
    return true;
  }
  if (_stricmp(value, "tgs") == 0) {
    solverType = PxSolverType::eTGS;
    return true;
  }
  if (_stricmp(value, "avbd") == 0) {
    solverType = PxSolverType::eAVBD;
    return true;
  }
  return false;
}

static PxSolverType::Enum getRequestedSolverType(int argc,
                                                 const char *const *argv) {
  for (int i = 1; i < argc; ++i) {
    if (!argv[i])
      continue;
    static const char prefix[] = "--solver=";
    if (std::strncmp(argv[i], prefix, sizeof(prefix) - 1) == 0) {
      PxSolverType::Enum solverType = PxSolverType::eAVBD;
      if (tryParseSolverType(argv[i] + sizeof(prefix) - 1, solverType))
        return solverType;
    }
  }

  const char *value = std::getenv("PHYSX_SNIPPET_SOLVER");
  PxSolverType::Enum solverType = PxSolverType::eAVBD;
  if (tryParseSolverType(value, solverType))
    return solverType;

  return PxSolverType::eAVBD;
}

static void dumpRevoluteChainState(PxU32 frame) {
  if (gRevoluteChainBodies.size() < 5 || gRevoluteChainJoints.size() < 5)
    return;

  printf("[RevoluteNodes] frame=%u\n", frame);

  for (PxU32 i = 0; i < gRevoluteChainBodies.size(); ++i) {
    PxRigidDynamic *body = gRevoluteChainBodies[i];
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 w = body->getAngularVelocity();
    printf("  node%u p=(%.3f,%.3f,%.3f) q=(%.4f,%.4f,%.4f,%.4f) "
           "w=(%.3f,%.3f,%.3f) sleep=%d\n",
           i, pose.p.x, pose.p.y, pose.p.z, pose.q.x, pose.q.y, pose.q.z,
           pose.q.w, w.x, w.y, w.z, body->isSleeping() ? 1 : 0);
  }

  for (PxU32 j = 0; j < gRevoluteChainJoints.size(); ++j) {
    PxVec3 a0, a1;
    getJointWorldAxes(gRevoluteChainJoints[j], a0, a1);
    const PxReal dot = PxClamp(a0.dot(a1), -1.0f, 1.0f);
    const PxReal misDeg = PxAcos(dot) * 180.0f / PxPi;
    const PxReal angle = gRevoluteChainJoints[j]->getAngle();
    const PxReal vel = gRevoluteChainJoints[j]->getVelocity();
    printf("  joint%u angle=%.4f vel=%.4f axisMisalignDeg=%.3f\n", j, angle,
           vel, misDeg);
  }
}

static void dumpPrismaticChainState(PxU32 frame) {
  if (gPrismaticChainBodies.size() < 5 || gPrismaticChainJoints.size() < 5 ||
      gPrismaticRestPositions.size() < 5)
    return;

  printf("[PrismaticNodes] frame=%u\n", frame);

  for (PxU32 i = 0; i < gPrismaticChainBodies.size(); ++i) {
    PxRigidDynamic *body = gPrismaticChainBodies[i];
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 linearVelocity = body->getLinearVelocity();
    const PxVec3 angularVelocity = body->getAngularVelocity();
    const PxVec3 restDelta = pose.p - gPrismaticRestPositions[i];
    const PxReal transverse =
        PxSqrt(restDelta.y * restDelta.y + restDelta.z * restDelta.z);
    printf("  node%u p=(%.3f,%.3f,%.3f) d=(%.3f,%.3f,%.3f) trans=%.4f "
           "v=(%.3f,%.3f,%.3f) w=(%.3f,%.3f,%.3f) sleep=%d\n",
           i, pose.p.x, pose.p.y, pose.p.z, restDelta.x, restDelta.y,
           restDelta.z, transverse, linearVelocity.x, linearVelocity.y,
           linearVelocity.z, angularVelocity.x, angularVelocity.y,
           angularVelocity.z, body->isSleeping() ? 1 : 0);
  }

  for (PxU32 j = 0; j < gPrismaticChainJoints.size(); ++j) {
    PxVec3 a0, a1;
    getJointWorldAxes(gPrismaticChainJoints[j], a0, a1);
    const PxReal dot = PxClamp(a0.dot(a1), -1.0f, 1.0f);
    const PxReal misDeg = PxAcos(dot) * 180.0f / PxPi;
    printf("  joint%u pos=%.4f vel=%.4f axisMisalignDeg=%.3f\n", j,
           gPrismaticChainJoints[j]->getPosition(),
           gPrismaticChainJoints[j]->getVelocity(), misDeg);
  }
}

static void dumpFixedChainState(PxU32 frame) {
  if (gFixedChainBodies.size() < 5 || gFixedChainJoints.size() < 5)
    return;

  printf("[FixedNodes] frame=%u\n", frame);

  for (PxU32 i = 0; i < gFixedChainBodies.size(); ++i) {
    PxRigidDynamic *body = gFixedChainBodies[i];
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 linearVelocity = body->getLinearVelocity();
    const PxVec3 angularVelocity = body->getAngularVelocity();
    printf("  node%u p=(%.3f,%.3f,%.3f) v=(%.3f,%.3f,%.3f) w=(%.3f,%.3f,%.3f) sleep=%d\n",
           i, pose.p.x, pose.p.y, pose.p.z, linearVelocity.x, linearVelocity.y,
           linearVelocity.z, angularVelocity.x, angularVelocity.y,
           angularVelocity.z, body->isSleeping() ? 1 : 0);
  }

  for (PxU32 j = 0; j < gFixedChainJoints.size(); ++j) {
    PxVec3 linearForce(0.0f), angularForce(0.0f);
    PxConstraint *constraint = gFixedChainJoints[j]->getConstraint();
    if (!constraint)
      continue;
    constraint->getForce(linearForce, angularForce);
    const PxReal linMag = linearForce.magnitude();
    const PxReal angMag = angularForce.magnitude();
    const bool broken = constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
    printf("  joint%u linForce=%.4f angForce=%.4f broken=%d f=(%.3f,%.3f,%.3f) t=(%.3f,%.3f,%.3f)\n",
           j, linMag, angMag, broken ? 1 : 0, linearForce.x, linearForce.y,
           linearForce.z, angularForce.x, angularForce.y, angularForce.z);
  }
}

static void getJointWorldAxes(PxRevoluteJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1) {
  axis0 = PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = PxVec3(1.0f, 0.0f, 0.0f);
  if (!joint)
    return;

  PxRigidActor *a0 = nullptr;
  PxRigidActor *a1 = nullptr;
  joint->getActors(a0, a1);

  const PxTransform lp0 = joint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform lp1 = joint->getLocalPose(PxJointActorIndex::eACTOR1);
  const PxVec3 localAxis(1.0f, 0.0f, 0.0f);

  axis0 = a0 ? a0->getGlobalPose().q.rotate(lp0.q.rotate(localAxis))
             : lp0.q.rotate(localAxis);
  axis1 = a1 ? a1->getGlobalPose().q.rotate(lp1.q.rotate(localAxis))
             : lp1.q.rotate(localAxis);

  const PxReal l0 = axis0.magnitudeSquared();
  const PxReal l1 = axis1.magnitudeSquared();
  axis0 = (l0 > 1e-12f) ? axis0 * PxRecipSqrt(l0) : PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = (l1 > 1e-12f) ? axis1 * PxRecipSqrt(l1) : PxVec3(1.0f, 0.0f, 0.0f);
}

static void getJointWorldAxes(PxPrismaticJoint *joint, PxVec3 &axis0,
                              PxVec3 &axis1) {
  axis0 = PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = PxVec3(1.0f, 0.0f, 0.0f);
  if (!joint)
    return;

  PxRigidActor *a0 = nullptr;
  PxRigidActor *a1 = nullptr;
  joint->getActors(a0, a1);

  const PxTransform lp0 = joint->getLocalPose(PxJointActorIndex::eACTOR0);
  const PxTransform lp1 = joint->getLocalPose(PxJointActorIndex::eACTOR1);
  const PxVec3 localAxis(1.0f, 0.0f, 0.0f);

  axis0 = a0 ? a0->getGlobalPose().q.rotate(lp0.q.rotate(localAxis))
             : lp0.q.rotate(localAxis);
  axis1 = a1 ? a1->getGlobalPose().q.rotate(lp1.q.rotate(localAxis))
             : lp1.q.rotate(localAxis);

  const PxReal l0 = axis0.magnitudeSquared();
  const PxReal l1 = axis1.magnitudeSquared();
  axis0 = (l0 > 1e-12f) ? axis0 * PxRecipSqrt(l0) : PxVec3(1.0f, 0.0f, 0.0f);
  axis1 = (l1 > 1e-12f) ? axis1 * PxRecipSqrt(l1) : PxVec3(1.0f, 0.0f, 0.0f);
}

static PxVec3 getJointWorldAxis(PxRevoluteJoint *joint) {
  if (!joint)
    return PxVec3(1.0f, 0.0f, 0.0f);

  PxVec3 axis0, axis1;
  getJointWorldAxes(joint, axis0, axis1);

  PxVec3 axis = axis0 + axis1;
  const PxReal len2 = axis.magnitudeSquared();
  if (len2 > 1e-12f)
    axis *= PxRecipSqrt(len2);
  else
    axis = (axis0.magnitudeSquared() > 1e-12f) ? axis0.getNormalized()
                                               : PxVec3(1.0f, 0.0f, 0.0f);
  return axis;
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

// revolute joint limited to an angle range of ±45 degrees
static PxJoint *createLimitedRevolute(PxRigidActor *a0, const PxTransform &t0,
                                      PxRigidActor *a1,
                                      const PxTransform &t1) {
  PxRevoluteJoint *j = PxRevoluteJointCreate(*gPhysics, a0, t0, a1, t1);
  j->setLimit(PxJointAngularLimitPair(-PxPi / 4, PxPi / 4));
  j->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);
  return j;
}

// spherical joint limited to an angle of at most pi/4 radians (45 degrees)
static PxJoint *createLimitedSpherical(PxRigidActor *a0, const PxTransform &t0,
                                       PxRigidActor *a1,
                                       const PxTransform &t1) {
  PxSphericalJoint *j = PxSphericalJointCreate(*gPhysics, a0, t0, a1, t1);
  j->setLimitCone(PxJointLimitCone(PxPi / 4, PxPi / 4));
  j->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
  return j;
}

// prismatic joint limited between -2 and 2
static PxJoint *createLimitedPrismatic(PxRigidActor *a0, const PxTransform &t0,
                                       PxRigidActor *a1,
                                       const PxTransform &t1) {
  PxPrismaticJoint *j = PxPrismaticJointCreate(*gPhysics, a0, t0, a1, t1);
  j->setLimit(PxJointLinearLimitPair(-2.0f, 2.0f, PxSpring(0, 0)));
  j->setPrismaticJointFlag(PxPrismaticJointFlag::eLIMIT_ENABLED, true);
  return j;
}

// fixed, breakable joint
static PxJoint *createBreakableFixed(PxRigidActor *a0, const PxTransform &t0,
                                     PxRigidActor *a1, const PxTransform &t1) {
  PxFixedJoint *j = PxFixedJointCreate(*gPhysics, a0, t0, a1, t1);
  j->setBreakForce(1000, 100000);
  j->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
  j->setConstraintFlag(PxConstraintFlag::eDISABLE_PREPROCESSING, true);
  return j;
}

// D6 joint with a spring maintaining its position
static PxJoint *createDampedD6(PxRigidActor *a0, const PxTransform &t0,
                               PxRigidActor *a1, const PxTransform &t1) {
  PxD6Joint *j = PxD6JointCreate(*gPhysics, a0, t0, a1, t1);
  j->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
  j->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
  j->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
  j->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
  j->setDrive(PxD6Drive::eSLERP, PxD6JointDrive(0, 1000, FLT_MAX, true));
  return j;
}

typedef PxJoint *(*JointCreateFunction)(PxRigidActor *a0, const PxTransform &t0,
                                        PxRigidActor *a1,
                                        const PxTransform &t1);

// create a chain rooted at the origin and extending along the x-axis, all
// transformed by the argument t.

static void createChain(const PxTransform &t, PxU32 length, const PxGeometry &g,
                        PxReal separation, JointCreateFunction createJoint) {
  PxVec3 offset(separation / 2, 0, 0);
  PxTransform localTm(offset);
  PxRigidDynamic *prev = NULL;

  for (PxU32 i = 0; i < length; i++) {
    PxRigidDynamic *current =
        PxCreateDynamic(*gPhysics, t * localTm, g, *gMaterial, 1.0f);
    PxJoint *joint = (*createJoint)(prev, prev ? PxTransform(offset) : t,
                                    current, PxTransform(-offset));

    if (createJoint == createLimitedRevolute) {
      gRevoluteChainBodies.push_back(current);
      if (joint) {
        PxRevoluteJoint *revolute = joint->is<PxRevoluteJoint>();
        if (revolute)
          gRevoluteChainJoints.push_back(revolute);
      }
    } else if (createJoint == createBreakableFixed) {
      gFixedChainBodies.push_back(current);
      if (joint) {
        PxFixedJoint *fixed = joint->is<PxFixedJoint>();
        if (fixed) {
          gFixedChainJoints.push_back(fixed);
          gFixedBreakReported.push_back(0);
        }
      }
    } else if (createJoint == createLimitedPrismatic) {
      gPrismaticChainBodies.push_back(current);
      gPrismaticRestPositions.push_back((t * localTm).p);
      if (joint) {
        PxPrismaticJoint *prismatic = joint->is<PxPrismaticJoint>();
        if (prismatic)
          gPrismaticChainJoints.push_back(prismatic);
      }
    }

    gScene->addActor(*current);
    prev = current;
    localTm.p.x += separation;
  }
}

void initPhysics(bool /*interactive*/) {
  gFoundation =
      PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
  gPvd = PxCreatePvd(*gFoundation);
  PxPvdTransport *transport =
      PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
  gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

  gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
                             PxTolerancesScale(), true, gPvd);
  PxInitExtensions(*gPhysics, gPvd);

  PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
  sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
  gDispatcher = PxDefaultCpuDispatcherCreate(2);
  sceneDesc.cpuDispatcher = gDispatcher;
  sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    sceneDesc.solverType = gSolverType;
  gScene = gPhysics->createScene(sceneDesc);

    printf("[SnippetJointConfig] solver=%s\n",
      getSolverTypeName(sceneDesc.solverType));

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

  createChain(PxTransform(PxVec3(0.0f, 20.0f, 0.0f)), 5,
              PxBoxGeometry(2.0f, 0.5f, 0.5f), 4.0f, createLimitedSpherical);
  createChain(PxTransform(PxVec3(0.0f, 20.0f, -10.0f)), 5,
              PxBoxGeometry(2.0f, 0.5f, 0.5f), 4.0f, createBreakableFixed);
  createChain(PxTransform(PxVec3(0.0f, 20.0f, -20.0f)), 5,
              PxBoxGeometry(2.0f, 0.5f, 0.5f), 4.0f, createDampedD6);
  createChain(PxTransform(PxVec3(0.0f, 20.0f, -30.0f)), 5,
              PxBoxGeometry(2.0f, 0.5f, 0.5f), 4.0f, createLimitedPrismatic);
  createChain(PxTransform(PxVec3(0.0f, 20.0f, -40.0f)), 5,
              PxBoxGeometry(2.0f, 0.5f, 0.5f), 4.0f, createLimitedRevolute);
}

void stepPhysics(bool /*interactive*/) {
  gScene->simulate(1.0f / 60.0f);
  gScene->fetchResults(true);

  if (shouldDumpRevoluteState(gRevoluteStats.frame))
    dumpRevoluteChainState(gRevoluteStats.frame);
  if (shouldDumpRevoluteState(gRevoluteStats.frame))
    dumpFixedChainState(gRevoluteStats.frame);
  if (shouldDumpRevoluteState(gRevoluteStats.frame))
    dumpPrismaticChainState(gRevoluteStats.frame);

  for (PxU32 j = 0; j < gFixedChainJoints.size(); ++j) {
    PxVec3 linearForce(0.0f), angularForce(0.0f);
    PxConstraint *constraint = gFixedChainJoints[j]->getConstraint();
    if (!constraint)
      continue;
    constraint->getForce(linearForce, angularForce);
    gFixedStats.maxLinearForce = PxMax(gFixedStats.maxLinearForce,
                                       linearForce.magnitude());
    gFixedStats.maxAngularForce = PxMax(gFixedStats.maxAngularForce,
                                        angularForce.magnitude());
    const bool broken = constraint->getFlags().isSet(PxConstraintFlag::eBROKEN);
    if (broken && j < gFixedBreakReported.size() && !gFixedBreakReported[j]) {
      gFixedBreakReported[j] = 1;
      if (gFixedStats.firstBrokenFrame == PX_MAX_U32)
        gFixedStats.firstBrokenFrame = gRevoluteStats.frame;
      gFixedStats.brokenCount++;
      printf("[FixedBreak] frame=%u joint=%u linForce=%.4f angForce=%.4f\n",
             gRevoluteStats.frame, j, linearForce.magnitude(),
             angularForce.magnitude());
    }
  }

  const PxU32 earlyBegin = 250, earlyEnd = 550;
  const PxU32 lateBegin = 1000, lateEnd = 1300;

  if (gRevoluteChainBodies.size() >= 5 && gRevoluteChainJoints.size() >= 5) {
    const PxVec3 tailP = gRevoluteChainBodies[4]->getGlobalPose().p;
    const PxReal lateral = PxSqrt(tailP.x * tailP.x + tailP.z * tailP.z);
    gRevoluteStats.maxTailLateral = PxMax(gRevoluteStats.maxTailLateral, lateral);

    const PxReal w4 = gRevoluteChainBodies[3]->getAngularVelocity().magnitude();
    const PxReal w5 = gRevoluteChainBodies[4]->getAngularVelocity().magnitude();

    const PxVec3 axis3 = getJointWorldAxis(gRevoluteChainJoints[3]);
    const PxVec3 axis4 = getJointWorldAxis(gRevoluteChainJoints[4]);
    const PxVec3 wv4 = gRevoluteChainBodies[3]->getAngularVelocity();
    const PxVec3 wv5 = gRevoluteChainBodies[4]->getAngularVelocity();
    const PxVec3 wv4Perp = wv4 - axis3 * wv4.dot(axis3);
    const PxVec3 wv5Perp = wv5 - axis4 * wv5.dot(axis4);
    const PxReal w4Perp = wv4Perp.magnitude();
    const PxReal w5Perp = wv5Perp.magnitude();

    if (gRevoluteStats.frame >= earlyBegin && gRevoluteStats.frame < earlyEnd) {
      gRevoluteStats.sumW4Early += w4;
      gRevoluteStats.sumW5Early += w5;
      gRevoluteStats.sumW4PerpEarly += w4Perp;
      gRevoluteStats.sumW5PerpEarly += w5Perp;
      gRevoluteStats.cntEarly++;
    }
    if (gRevoluteStats.frame >= lateBegin && gRevoluteStats.frame < lateEnd) {
      gRevoluteStats.sumW4Late += w4;
      gRevoluteStats.sumW5Late += w5;
      gRevoluteStats.sumW4PerpLate += w4Perp;
      gRevoluteStats.sumW5PerpLate += w5Perp;
      gRevoluteStats.maxW4PerpLate = PxMax(gRevoluteStats.maxW4PerpLate, w4Perp);
      gRevoluteStats.maxW5PerpLate = PxMax(gRevoluteStats.maxW5PerpLate, w5Perp);
      if (!gRevoluteChainBodies[3]->isSleeping())
        gRevoluteStats.awake4Late++;
      if (!gRevoluteChainBodies[4]->isSleeping())
        gRevoluteStats.awake5Late++;
      gRevoluteStats.cntLate++;
    }

    PxVec3 j3a0, j3a1, j4a0, j4a1;
    getJointWorldAxes(gRevoluteChainJoints[3], j3a0, j3a1);
    getJointWorldAxes(gRevoluteChainJoints[4], j4a0, j4a1);
    const PxReal dot3 = PxClamp(j3a0.dot(j3a1), -1.0f, 1.0f);
    const PxReal dot4 = PxClamp(j4a0.dot(j4a1), -1.0f, 1.0f);
    const PxReal mis3 = PxAcos(dot3) * 180.0f / PxPi;
    const PxReal mis4 = PxAcos(dot4) * 180.0f / PxPi;
    gRevoluteStats.maxAxisMisalign3Deg = PxMax(gRevoluteStats.maxAxisMisalign3Deg, mis3);
    gRevoluteStats.maxAxisMisalign4Deg = PxMax(gRevoluteStats.maxAxisMisalign4Deg, mis4);

    const PxReal a3 = gRevoluteChainJoints[3]->getAngle();
    const PxReal a4 = gRevoluteChainJoints[4]->getAngle();
    gRevoluteStats.maxAbsAngle3 = PxMax(gRevoluteStats.maxAbsAngle3, PxAbs(a3));
    gRevoluteStats.maxAbsAngle4 = PxMax(gRevoluteStats.maxAbsAngle4, PxAbs(a4));
    const PxReal d3 = a3 - gRevoluteStats.prevAngle3;
    const PxReal d4 = a4 - gRevoluteStats.prevAngle4;

    if (PxAbs(d3) > 1e-5f && PxAbs(gRevoluteStats.prevD3) > 1e-5f &&
        d3 * gRevoluteStats.prevD3 < 0.0f)
      gRevoluteStats.flip3++;
    if (PxAbs(d4) > 1e-5f && PxAbs(gRevoluteStats.prevD4) > 1e-5f &&
        d4 * gRevoluteStats.prevD4 < 0.0f)
      gRevoluteStats.flip4++;

    gRevoluteStats.prevD3 = d3;
    gRevoluteStats.prevD4 = d4;
    gRevoluteStats.prevAngle3 = a3;
    gRevoluteStats.prevAngle4 = a4;
  }

  if (gPrismaticChainBodies.size() >= 5 && gPrismaticRestPositions.size() >= 5 &&
      gPrismaticChainJoints.size() >= 5) {
    const PxVec3 tailDelta =
        gPrismaticChainBodies[4]->getGlobalPose().p - gPrismaticRestPositions[4];
    const PxReal tailTransverse =
        PxSqrt(tailDelta.y * tailDelta.y + tailDelta.z * tailDelta.z);
    const PxReal tailAngVel =
        gPrismaticChainBodies[4]->getAngularVelocity().magnitude();
    gPrismaticStats.maxTailTransverse =
        PxMax(gPrismaticStats.maxTailTransverse, tailTransverse);

    if (gRevoluteStats.frame >= earlyBegin && gRevoluteStats.frame < earlyEnd) {
      gPrismaticStats.sumTailEarly += tailTransverse;
      gPrismaticStats.sumTailAngVelEarly += tailAngVel;
      gPrismaticStats.cntEarly++;
    }
    if (gRevoluteStats.frame >= lateBegin && gRevoluteStats.frame < lateEnd) {
      gPrismaticStats.sumTailLate += tailTransverse;
      gPrismaticStats.sumTailAngVelLate += tailAngVel;
      gPrismaticStats.cntLate++;
    }

    for (PxU32 j = 0; j < gPrismaticChainJoints.size(); ++j) {
      PxVec3 a0, a1;
      getJointWorldAxes(gPrismaticChainJoints[j], a0, a1);
      const PxReal dot = PxClamp(a0.dot(a1), -1.0f, 1.0f);
      const PxReal misDeg = PxAcos(dot) * 180.0f / PxPi;
      gPrismaticStats.maxJointAxisMisalignDeg =
          PxMax(gPrismaticStats.maxJointAxisMisalignDeg, misDeg);
    }
  }

  gRevoluteStats.frame++;
}

void cleanupPhysics(bool /*interactive*/) {
  const PxReal avgW4Early = gRevoluteStats.cntEarly
                                ? gRevoluteStats.sumW4Early / gRevoluteStats.cntEarly
                                : 0.0f;
  const PxReal avgW5Early = gRevoluteStats.cntEarly
                                ? gRevoluteStats.sumW5Early / gRevoluteStats.cntEarly
                                : 0.0f;
  const PxReal avgW4Late = gRevoluteStats.cntLate
                               ? gRevoluteStats.sumW4Late / gRevoluteStats.cntLate
                               : 0.0f;
  const PxReal avgW5Late = gRevoluteStats.cntLate
                               ? gRevoluteStats.sumW5Late / gRevoluteStats.cntLate
                               : 0.0f;
    const PxReal avgW4PerpEarly = gRevoluteStats.cntEarly
             ? gRevoluteStats.sumW4PerpEarly / gRevoluteStats.cntEarly
             : 0.0f;
    const PxReal avgW5PerpEarly = gRevoluteStats.cntEarly
             ? gRevoluteStats.sumW5PerpEarly / gRevoluteStats.cntEarly
             : 0.0f;
    const PxReal avgW4PerpLate = gRevoluteStats.cntLate
            ? gRevoluteStats.sumW4PerpLate / gRevoluteStats.cntLate
            : 0.0f;
    const PxReal avgW5PerpLate = gRevoluteStats.cntLate
            ? gRevoluteStats.sumW5PerpLate / gRevoluteStats.cntLate
            : 0.0f;
  const PxReal awake4LateRatio = gRevoluteStats.cntLate
                                     ? PxReal(gRevoluteStats.awake4Late) / PxReal(gRevoluteStats.cntLate)
                                     : 0.0f;
  const PxReal awake5LateRatio = gRevoluteStats.cntLate
                                     ? PxReal(gRevoluteStats.awake5Late) / PxReal(gRevoluteStats.cntLate)
                                     : 0.0f;
  const PxReal growth4 = (avgW4Early > 1e-6f) ? (avgW4Late / avgW4Early) : 0.0f;
  const PxReal growth5 = (avgW5Early > 1e-6f) ? (avgW5Late / avgW5Early) : 0.0f;
    const PxReal growth4Perp = (avgW4PerpEarly > 1e-6f) ? (avgW4PerpLate / avgW4PerpEarly) : 0.0f;
    const PxReal growth5Perp = (avgW5PerpEarly > 1e-6f) ? (avgW5PerpLate / avgW5PerpEarly) : 0.0f;
    const bool jitterReproduced = (growth4Perp > 1.10f) || (growth5Perp > 1.10f) ||
                                (gRevoluteStats.maxW4PerpLate > 2.0f && awake4LateRatio > 0.30f) ||
                                (gRevoluteStats.maxW5PerpLate > 2.0f && awake5LateRatio > 0.30f) ||
                                (gRevoluteStats.flip4 > 140);

  printf("[RevoluteDiag] tail lateral max=%.5f, avgW early=(%.5f,%.5f), "
      "avgW late=(%.5f,%.5f), growth=(%.3f,%.3f), "
      "avgWPerp early=(%.5f,%.5f), avgWPerp late=(%.5f,%.5f), "
      "growthPerp=(%.3f,%.3f), maxWPerpLate=(%.5f,%.5f), awakeLate=(%.3f,%.3f), "
      "max|angle|=(%.3f,%.3f), axisMisalignDeg=(%.3f,%.3f), flips=(%u,%u), "
      "jitter_reproduced=%s\n",
         gRevoluteStats.maxTailLateral, avgW4Early, avgW5Early, avgW4Late,
      avgW5Late, growth4, growth5, avgW4PerpEarly, avgW5PerpEarly,
      avgW4PerpLate, avgW5PerpLate, growth4Perp, growth5Perp,
      gRevoluteStats.maxW4PerpLate, gRevoluteStats.maxW5PerpLate,
      awake4LateRatio, awake5LateRatio,
      gRevoluteStats.maxAbsAngle3, gRevoluteStats.maxAbsAngle4,
      gRevoluteStats.maxAxisMisalign3Deg, gRevoluteStats.maxAxisMisalign4Deg,
      gRevoluteStats.flip3,
         gRevoluteStats.flip4, jitterReproduced ? "true" : "false");

      const PxReal avgPrismaticTailEarly =
        gPrismaticStats.cntEarly
          ? gPrismaticStats.sumTailEarly / gPrismaticStats.cntEarly
          : 0.0f;
      const PxReal avgPrismaticTailLate =
        gPrismaticStats.cntLate ? gPrismaticStats.sumTailLate / gPrismaticStats.cntLate
                    : 0.0f;
      const PxReal avgPrismaticTailAngVelEarly =
        gPrismaticStats.cntEarly
          ? gPrismaticStats.sumTailAngVelEarly / gPrismaticStats.cntEarly
          : 0.0f;
      const PxReal avgPrismaticTailAngVelLate =
        gPrismaticStats.cntLate
          ? gPrismaticStats.sumTailAngVelLate / gPrismaticStats.cntLate
          : 0.0f;
      const PxReal prismaticTailGrowth =
        (avgPrismaticTailEarly > 1e-6f)
          ? (avgPrismaticTailLate / avgPrismaticTailEarly)
          : 0.0f;
      const PxReal prismaticAngVelGrowth =
        (avgPrismaticTailAngVelEarly > 1e-6f)
          ? (avgPrismaticTailAngVelLate / avgPrismaticTailAngVelEarly)
          : 0.0f;

      printf("[PrismaticDiag] tail transverse max=%.5f, avgTail early=%.5f, "
         "avgTail late=%.5f, tailGrowth=%.3f, avgTailAngVel early=%.5f, "
         "avgTailAngVel late=%.5f, angVelGrowth=%.3f, maxAxisMisalignDeg=%.3f\n",
         gPrismaticStats.maxTailTransverse, avgPrismaticTailEarly,
         avgPrismaticTailLate, prismaticTailGrowth, avgPrismaticTailAngVelEarly,
         avgPrismaticTailAngVelLate, prismaticAngVelGrowth,
         gPrismaticStats.maxJointAxisMisalignDeg);

    printf("[FixedDiag] maxLinForce=%.5f, maxAngForce=%.5f, firstBrokenFrame=%u, brokenCount=%u\n",
        gFixedStats.maxLinearForce, gFixedStats.maxAngularForce,
        gFixedStats.firstBrokenFrame, gFixedStats.brokenCount);

  PX_RELEASE(gScene);
  PX_RELEASE(gDispatcher);
  PxCloseExtensions();
  PX_RELEASE(gPhysics);
  if (gPvd) {
    PxPvdTransport *transport = gPvd->getTransport();
    PX_RELEASE(gPvd);
    PX_RELEASE(transport);
  }
  PX_RELEASE(gFoundation);

  printf("SnippetJoint done.\n");
}

void keyPress(unsigned char key, const PxTransform &camera) {
  switch (toupper(key)) {
  case ' ':
    createDynamic(camera, PxSphereGeometry(3.0f),
                  camera.rotate(PxVec3(0, 0, -1)) * 200);
    break;
  }
}

static bool hasHeadlessArg(int argc, const char *const *argv) {
  for (int i = 1; i < argc; ++i) {
    if (!argv[i])
      continue;
    if (std::strcmp(argv[i], "--headless") == 0)
      return true;
    if (std::strncmp(argv[i], "--headless-", 11) == 0)
      return true;
  }
  return false;
}

static bool isHeadlessRequested(int argc, const char *const *argv) {
  if (hasHeadlessArg(argc, argv))
    return true;

  const char *value = std::getenv("PHYSX_SNIPPET_HEADLESS");
  return value && value[0] && value[0] != '0';
}

int snippetMain(int argc, const char *const *argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  gSolverType = getRequestedSolverType(argc, argv);
#ifdef RENDER_SNIPPET
  if (!isHeadlessRequested(argc, argv)) {
    extern void renderLoop();
    renderLoop();
    return 0;
  }
#endif

  static const PxU32 frameCount = 1400;
  initPhysics(false);
  for (PxU32 i = 0; i < frameCount; i++)
    stepPhysics(false);
  cleanupPhysics(false);

  return 0;
}
