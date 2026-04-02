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

// Contact and joint constraint preparation for the AVBD solver.
// Split from DyAvbdDynamics.cpp - contains prepareAvbdContacts() and
// prepareAvbdConstraints() member functions of AvbdDynamicsContext.

#include "DyAvbdDynamics.h"
#include "DyAvbdConstraint.h"
#include "DyConstraint.h"
#include "DyFeatherstoneArticulation.h"
#include "DyIslandManager.h"
#include "PxContact.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsIslandManagerTypes.h"
#include "PxsRigidBody.h"
#include "PxsSimpleIslandManager.h"
#include "common/PxProfileZone.h"
#include "foundation/PxMath.h"

using namespace physx;
using namespace physx::Dy;

static PxU32 findArticulationLinkIndex(FeatherstoneArticulation *articulation,
                                       const PxsRigidCore *rigidCore) {
  if (!articulation || !rigidCore)
    return PX_MAX_U32;
  ArticulationData &artData = articulation->getArticulationData();
  const PxU32 linkCount = artData.getLinkCount();
  for (PxU32 linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
    if (artData.getLink(linkIdx).bodyCore == rigidCore)
      return linkIdx;
  }
  return PX_MAX_U32;
}

#ifndef AVBD_JOINT_DEBUG
#define AVBD_JOINT_DEBUG 0
#endif
#ifndef AVBD_JOINT_DEBUG_FRAMES
#define AVBD_JOINT_DEBUG_FRAMES 2
#endif

static constexpr physx::PxU16 AVBD_PRISMATIC_LIMIT_ENABLED_FLAG = 0x0002;

// Helper struct for joint data protocol (must match SnippetAvbdDx11)
struct AvbdSnippetJointData {
  enum Type { eSPHERICAL = 0, eFIXED, eREVOLUTE, ePRISMATIC, eD6 };
  int type;
  PxVec3 pivot0;
  PxVec3 pivot1;
  PxVec3 axis0;
  PxVec3 axis1;
  float limitLow;
  float limitHigh;
  float padding[2];
};

// Standard PhysX JointData structure (mirrors ExtJointData.h)
// This is the base format for all standard PhysX joints
struct PhysXJointData {
  PxConstraintInvMassScale invMassScale;
  PxTransform32 c2b[2]; // Constraint-to-body transforms
};

// Mirror of PxJointLimitParameters (16 bytes with padding)
struct PhysXJointLimitParameters {
  PxReal restitution;
  PxReal bounceThreshold;
  PxReal stiffness;
  PxReal damping;
};

// Mirror of PxJointLimitCone
struct PhysXJointLimitCone : PhysXJointLimitParameters {
  PxReal yAngle;
  PxReal zAngle;
};

// Mirror of SphericalJointData (ExtSphericalJoint.h)
struct PhysXSphericalJointData : PhysXJointData {
  PhysXJointLimitCone limit;
  PxU16 jointFlags; // PxSphericalJointFlags
};

// Mirror of FixedJointData (ExtFixedJoint.h) - no extra members
struct PhysXFixedJointData : PhysXJointData {
  // No additional members
};

// Mirror of PxJointAngularLimitPair
struct PhysXJointAngularLimitPair : PhysXJointLimitParameters {
  PxReal upper;
  PxReal lower;
};

// Mirror of PxJointLinearLimitPair
struct PhysXJointLinearLimitPair : PhysXJointLimitParameters {
  PxReal upper;
  PxReal lower;
};

// Mirror of RevoluteJointData (ExtRevoluteJoint.h)
struct PhysXRevoluteJointData : PhysXJointData {
  PxReal driveVelocity;
  PxReal driveForceLimit;
  PxReal driveGearRatio;
  PhysXJointAngularLimitPair limit;
  PxU16 jointFlags; // PxRevoluteJointFlags
};

// Mirror of PrismaticJointData (ExtPrismaticJoint.h)
struct PhysXPrismaticJointData : PhysXJointData {
  PhysXJointLinearLimitPair limit;
  PxU16 jointFlags; // PxPrismaticJointFlags
};

// Mirror of PxD6JointDrive
struct PhysXD6JointDrive {
  PxReal stiffness;
  PxReal damping;
  PxReal forceLimit;
  PxU32 flags;
};

// Mirror of PxJointLinearLimit
struct PhysXJointLinearLimit : PhysXJointLimitParameters {
  PxReal value;
};

// Mirror of PxJointLimitPyramid
struct PhysXJointLimitPyramid : PhysXJointLimitParameters {
  PxReal yAngleMin;
  PxReal yAngleMax;
  PxReal zAngleMin;
  PxReal zAngleMax;
};

// Mirror of D6JointData (ExtD6Joint.h) - partial, enough for type detection
struct PhysXD6JointData : PhysXJointData {
  PxU32 motion[6]; // PxD6Motion::Enum
  PhysXJointLinearLimit distanceLimit;
  PhysXJointLinearLimitPair linearLimitX;
  PhysXJointLinearLimitPair linearLimitY;
  PhysXJointLinearLimitPair linearLimitZ;
  PhysXJointAngularLimitPair twistLimit;
  PhysXJointLimitCone swingLimit;
  PhysXJointLimitPyramid pyramidSwingLimit;
  PhysXD6JointDrive drive[6];
  PxTransform drivePosition;
  PxVec3 driveLinearVelocity;
  PxVec3 driveAngularVelocity;
  PxU32 locked;
  PxU32 limited;
  PxU32 driving;
  // More members follow but not needed for detection
};

// Mirror of GearJointData (ExtGearJoint.h)
struct PhysXGearJointData : PhysXJointData {
  const void *hingeJoint0; // PxBase* - either PxJoint or
                           // PxArticulationJointReducedCoordinate
  const void *hingeJoint1; // PxBase* - either PxJoint or
                           // PxArticulationJointReducedCoordinate
  float gearRatio;
  float error;
};

// Enum for detected joint types (local copy for compatibility)
enum PhysXJointType {
  eJOINT_UNKNOWN = -1,
  eJOINT_FIXED = 0,
  eJOINT_SPHERICAL = 1,
  eJOINT_REVOLUTE = 2,
  eJOINT_PRISMATIC = 3,
  eJOINT_D6 = 4,
  eJOINT_GEAR = 5
};

// Helper function to convert Dy::ConstraintJointType to local PhysXJointType
static PhysXJointType getJointTypeFromConstraint(Dy::ConstraintJointType cjt) {
  switch (cjt) {
  case Dy::eCONSTRAINT_JOINT_SPHERICAL:
    return eJOINT_SPHERICAL;
  case Dy::eCONSTRAINT_JOINT_REVOLUTE:
    return eJOINT_REVOLUTE;
  case Dy::eCONSTRAINT_JOINT_PRISMATIC:
    return eJOINT_PRISMATIC;
  case Dy::eCONSTRAINT_JOINT_FIXED:
    return eJOINT_FIXED;
  case Dy::eCONSTRAINT_JOINT_D6:
    return eJOINT_D6;
  case Dy::eCONSTRAINT_JOINT_GEAR:
    return eJOINT_GEAR;
  default:
    return eJOINT_UNKNOWN;
  }
}

PxU32 AvbdDynamicsContext::prepareAvbdContacts(
    AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
    AvbdContactConstraint *constraints, PxU32 maxConstraints,
    PxU32 startContactIdx, PxU32 numContactsToProcess, PxU32 bodyOffset) {

  PxU32 constraintIndex = 0;
  const PxU32 endContactIdx = startContactIdx + numContactsToProcess;
  const PxU32 actualMax =
      PxMin(static_cast<PxU32>(mContactList.size()), endContactIdx);
  const PxU32 bodyEnd = bodyOffset + islandBodyCount;

  // Debug counters for lambda warm-starting diagnosis
  static PxU32 sDebugHits = 0;
  static PxU32 sDebugMisses = 0;
  static PxU32 sFrameCount = 0;
  PxU32 localHits = 0;
  PxU32 localMisses = 0;

  for (PxU32 i = startContactIdx;
       i < actualMax && constraintIndex < maxConstraints; ++i) {
    const PxsIndexedContactManager &icm = mContactList[i];
    PxsContactManager *cm = icm.contactManager;

    if (!cm)
      continue;

    PxU32 globalBody0Idx = PX_MAX_U32;
    PxU32 globalBody1Idx = PX_MAX_U32;

    if (icm.indexType0 == PxsIndexedInteraction::eBODY) {
      globalBody0Idx = static_cast<PxU32>(icm.solverBody0);
    }
    if (icm.indexType1 == PxsIndexedInteraction::eBODY) {
      globalBody1Idx = static_cast<PxU32>(icm.solverBody1);
    }

    if (globalBody0Idx == PX_MAX_U32 && globalBody1Idx == PX_MAX_U32)
      continue;

    PxU32 localBody0Idx = PX_MAX_U32;
    PxU32 localBody1Idx = PX_MAX_U32;

    if (globalBody0Idx != PX_MAX_U32 && globalBody0Idx >= bodyOffset &&
        globalBody0Idx < bodyEnd) {
      localBody0Idx = globalBody0Idx - bodyOffset;
    }
    if (globalBody1Idx != PX_MAX_U32 && globalBody1Idx >= bodyOffset &&
        globalBody1Idx < bodyEnd) {
      localBody1Idx = globalBody1Idx - bodyOffset;
    }

    if (localBody0Idx == PX_MAX_U32 && localBody1Idx == PX_MAX_U32)
      continue;

    const PxU32 npIndex = cm->getWorkUnit().mNpIndex;
    PxsContactManagerOutput &output =
        mOutputIterator.getContactManagerOutput(npIndex);

    if (output.nbContacts == 0)
      continue;
    // NOTE: Do NOT filter by eHAS_TOUCH here. AVBD is a position-based solver
    // that predicts positions (gravity warmstart) before solving. Near-miss
    // contacts (separation > 0, eHAS_TOUCH=false) are essential: without them,
    // the solver has no constraint to prevent predicted positions from
    // penetrating static geometry, causing bodies to fall through the ground.

    AvbdSolverBody *bodyA = (localBody0Idx < islandBodyCount)
                                ? &avbdBodies[localBody0Idx]
                                : nullptr;
    AvbdSolverBody *bodyB = (localBody1Idx < islandBodyCount)
                                ? &avbdBodies[localBody1Idx]
                                : nullptr;

    const PxU8 *contactData = output.contactPoints;
    const PxU8 *patchData = output.contactPatches;

    if (!contactData || !patchData)
      continue;

    for (PxU8 patchIdx = 0; patchIdx < output.nbPatches; ++patchIdx) {
      const PxContactPatch *patch = reinterpret_cast<const PxContactPatch *>(
          patchData + patchIdx * sizeof(PxContactPatch));

      const PxVec3 normal = patch->normal;
      const PxU32 startContact = patch->startContactIndex;
      const PxU16 numContactsInPatch = patch->nbContacts;

      for (PxU16 c = 0;
           c < numContactsInPatch && constraintIndex < maxConstraints; ++c) {
        const PxContact *contact = reinterpret_cast<const PxContact *>(
            contactData + (startContact + c) * sizeof(PxContact));

        AvbdContactConstraint &constraint = constraints[constraintIndex];

        constraint.header.bodyIndexA = localBody0Idx;
        constraint.header.bodyIndexB = localBody1Idx;
        constraint.header.type =
            static_cast<PxU16>(AvbdConstraintType::eCONTACT);
        constraint.header.flags = 0;
        constraint.header.compliance = 0.0f;
        constraint.header.damping = AvbdConstants::AVBD_CONSTRAINT_DAMPING;
        constraint.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_LOW;
        constraint.header.penalty =
            AvbdConstants::AVBD_MIN_PENALTY_RHO; // PENALTY_MIN = 1000

        // Lambda & penalty warm-starting (ref: AVBD3D solver.cpp L64-72)
        //   lambda *= alpha * gamma
        //   penalty = clamp(penalty * gamma, PENALTY_MIN, PENALTY_MAX)
        const PxU32 cmIdx = cm->getIndex();
        const PxU32 cacheIdx = cmIdx * MAX_CONTACTS_PER_CM + c;
        constraint.cacheIndex = cacheIdx;

        // AVBD warmstart decay constants
        // alpha=0.95, gamma=0.99 => alpha*gamma=0.9405
        const PxReal wsAlpha = 0.95f;
        const PxReal wsGamma = 0.99f;
        const PxReal wsPenaltyMin = 1000.0f;
        const PxReal wsPenaltyMax = 1e9f;

        if (mEnableLambdaWarmStart && cacheIdx < mLambdaCache.size()) {
          CachedLambda &cached = mLambdaCache[cacheIdx];
          if (cached.frameAge <= LAMBDA_MAX_AGE) {
            // Apply warmstart decay (ref Eq. 19)
            constraint.header.lambda = cached.lambda * wsAlpha * wsGamma;
            constraint.tangentLambda0 =
                cached.tangentLambda0 * wsAlpha * wsGamma;
            constraint.tangentLambda1 =
                cached.tangentLambda1 * wsAlpha * wsGamma;
            // Restore and decay penalty (normal + tangent)
            constraint.header.penalty =
                PxClamp(cached.penalty * wsGamma, wsPenaltyMin, wsPenaltyMax);
            constraint.tangentPenalty0 = PxClamp(
                cached.tangentPenalty0 * wsGamma, wsPenaltyMin, wsPenaltyMax);
            constraint.tangentPenalty1 = PxClamp(
                cached.tangentPenalty1 * wsGamma, wsPenaltyMin, wsPenaltyMax);
            localHits++;
          } else {
            constraint.header.lambda = 0.0f;
            constraint.tangentLambda0 = 0.0f;
            constraint.tangentLambda1 = 0.0f;
            constraint.header.penalty = wsPenaltyMin;
            constraint.tangentPenalty0 = wsPenaltyMin;
            constraint.tangentPenalty1 = wsPenaltyMin;
            localMisses++;
          }
        } else {
          constraint.header.lambda = 0.0f;
          constraint.tangentLambda0 = 0.0f;
          constraint.tangentLambda1 = 0.0f;
          constraint.header.penalty = wsPenaltyMin;
          constraint.tangentPenalty0 = wsPenaltyMin;
          constraint.tangentPenalty1 = wsPenaltyMin;
          localMisses++;
          // Grow cache if needed (for next frame)
          if (mEnableLambdaWarmStart && cacheIdx >= mLambdaCache.size()) {
            PxU32 newSize =
                ((cacheIdx + 1) + 1023u) & ~1023u; // Round up to 1024
            mLambdaCache.resize(newSize);
          }
        }

        if (bodyA) {
          constraint.contactPointA =
              bodyA->rotation.rotateInv(contact->contact - bodyA->position);
        } else {
          constraint.contactPointA = contact->contact;
        }

        if (bodyB) {
          constraint.contactPointB =
              bodyB->rotation.rotateInv(contact->contact - bodyB->position);
        } else {
          constraint.contactPointB = contact->contact;
        }

        constraint.contactNormal = normal;
        constraint.penetrationDepth = contact->separation;
        constraint.restitution = patch->restitution;
        constraint.friction = patch->dynamicFriction;

        PxVec3 t0, t1;
        if (PxAbs(normal.y) > 0.9f) {
          t0 = normal.cross(PxVec3(1, 0, 0)).getNormalized();
        } else {
          t0 = normal.cross(PxVec3(0, 1, 0)).getNormalized();
        }
        t1 = normal.cross(t0);

        constraint.tangent0 = t0;
        constraint.tangent1 = t1;
        // NOTE: tangentLambda0/1 are already set by the warmstart block above
        // (either warmstarted values or 0). Do NOT overwrite them here.

        // Initialize C0 to 0 (will be computed by solver before iterations)
        constraint.C0 = 0.0f;

        ++constraintIndex;
      }
    }
  }

  // Update global debug counters and print statistics
  sDebugHits += localHits;
  sDebugMisses += localMisses;
  sFrameCount++;
  // DEBUG: Lambda cache stats disabled during explosion debugging
  // if (sFrameCount % 60 == 0 && constraintIndex > 0) {
  //   PxU32 total = sDebugHits + sDebugMisses;
  //   float hitRate = total > 0 ? (float)sDebugHits / total * 100.0f : 0.0f;
  //   printf(
  //       "[AVBD Lambda Cache] Frame %u: hits=%u misses=%u (%.1f%% hit
  //       rate)\n", sFrameCount, sDebugHits, sDebugMisses, hitRate);
  // }

  return constraintIndex;
}

void AvbdDynamicsContext::prepareAvbdConstraints(
    const IG::IslandSim &islandSim, AvbdSolverBody *avbdBodies,
    PxU32 islandBodyCount, PxU32 bodyOffset,
    AvbdD6JointConstraint *d6Constraints, PxU32 &numD6, PxU32 maxD6,
    AvbdGearJointConstraint *gearConstraints, PxU32 &numGear, PxU32 maxGear,
    PxU32 islandIndex, PxU32 *bodyRemapTable,
    PxU32 *articulationFirstLinkIndex,
    FeatherstoneArticulation **articulationByActiveIdx,
    PxU32 numArticulations) {

  PX_UNUSED(avbdBodies);
  PX_UNUSED(islandBodyCount);

  const PxU32 numDynamicBodies =
      islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);

  const IG::Island &island =
      islandSim.getIsland(islandSim.getActiveIslands()[islandIndex]);
  IG::EdgeIndex edgeIndex = island.mFirstEdge[IG::Edge::eCONSTRAINT];

  numD6 = 0;
  numGear = 0;

  while (edgeIndex != IG_INVALID_EDGE) {
    const IG::Edge &edge = islandSim.getEdge(edgeIndex);
    Dy::Constraint *constraint = mIslandManager.getConstraint(edgeIndex);

    if (constraint && constraint->constantBlock &&
        constraint->constantBlockSize > 0) {

      const PxNodeIndex nodeIndex0 =
          islandSim.mCpuData.getNodeIndex1(edgeIndex);
      const PxNodeIndex nodeIndex1 =
          islandSim.mCpuData.getNodeIndex2(edgeIndex);

      PxU32 localBody0 = PX_MAX_U32;
      PxU32 localBody1 = PX_MAX_U32;

      // Check if bodies are static using inverseMass (more reliable than
      // nodeIndex.isStaticBody()) invMass == 0 means infinite mass (static
      // body) Note: NULL bodyCore means connected to world, which is handled by
      // localBody staying PX_MAX_U32
      bool body0IsStatic =
          constraint->bodyCore0 && constraint->bodyCore0->inverseMass == 0.0f;
      bool body1IsStatic =
          constraint->bodyCore1 && constraint->bodyCore1->inverseMass == 0.0f;

      if (!body0IsStatic) {
        if (!nodeIndex0.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex0);
          const IG::Node &node0 = islandSim.getNode(nodeIndex0);
          const bool isArt0 =
              node0.getNodeType() == IG::Node::eARTICULATION_TYPE;
          const PxU32 remapIdx0 =
              isArt0 ? (numDynamicBodies + activeIdx) : activeIdx;
          if (bodyRemapTable[remapIdx0] != PX_MAX_U32) {
            localBody0 = bodyRemapTable[remapIdx0] - bodyOffset;
            // For articulation nodes, bodyRemapTable points to the first link.
            // Resolve the specific link via bodyCore matching.
            if (isArt0 &&
                articulationByActiveIdx && articulationFirstLinkIndex &&
                activeIdx < numArticulations + 1 &&
                articulationByActiveIdx[activeIdx] && constraint->bodyCore0) {
              PxU32 linkIdx = findArticulationLinkIndex(
                  articulationByActiveIdx[activeIdx], constraint->bodyCore0);
              if (linkIdx != PX_MAX_U32) {
                localBody0 = articulationFirstLinkIndex[activeIdx] - bodyOffset + linkIdx;
              }
            }
          }
        }
      }

      if (!body1IsStatic) {
        if (!nodeIndex1.isStaticBody()) {
          const PxU32 activeIdx = islandSim.getActiveNodeIndex(nodeIndex1);
          const IG::Node &node1 = islandSim.getNode(nodeIndex1);
          const bool isArt1 =
              node1.getNodeType() == IG::Node::eARTICULATION_TYPE;
          const PxU32 remapIdx1 =
              isArt1 ? (numDynamicBodies + activeIdx) : activeIdx;
          if (bodyRemapTable[remapIdx1] != PX_MAX_U32) {
            localBody1 = bodyRemapTable[remapIdx1] - bodyOffset;
            // For articulation nodes, resolve specific link via bodyCore.
            if (isArt1 &&
                articulationByActiveIdx && articulationFirstLinkIndex &&
                activeIdx < numArticulations + 1 &&
                articulationByActiveIdx[activeIdx] && constraint->bodyCore1) {
              PxU32 linkIdx = findArticulationLinkIndex(
                  articulationByActiveIdx[activeIdx], constraint->bodyCore1);
              if (linkIdx != PX_MAX_U32) {
                localBody1 = articulationFirstLinkIndex[activeIdx] - bodyOffset + linkIdx;
              }
            }
          }
        }
      }

      // Use the jointType field to detect joint type reliably
      PhysXJointType jointType =
          getJointTypeFromConstraint(constraint->jointType);

#if AVBD_JOINT_DEBUG
      if (getAvbdMotorFrameCounter() < AVBD_JOINT_DEBUG_FRAMES) {
        printf("[Constraint Parse] type: %d, internal type: %d, block size: "
               "%u, expected: %zu\n",
               jointType, constraint->jointType, constraint->constantBlockSize,
               sizeof(PhysXJointData));
        if (constraint->constantBlockSize >= sizeof(PhysXJointData)) {
          const PhysXJointData *physXData =
              static_cast<const PhysXJointData *>(constraint->constantBlock);
          printf("  -> invMassScale0: %f, c2b0.p.x: %f, c2b1.p.x: %f\n",
                 physXData->invMassScale.linear0, physXData->c2b[0].p.x,
                 physXData->c2b[1].p.x);
        }
      }
#endif

      if (jointType == eJOINT_GEAR) {
        // Process GearJoint
        if (numGear < maxGear && gearConstraints) {
          const PhysXGearJointData *gearData =
              static_cast<const PhysXGearJointData *>(
                  constraint->constantBlock);

          AvbdGearJointConstraint &c = gearConstraints[numGear++];
          c.initDefaults();
          c.header.bodyIndexA = localBody0;
          c.header.bodyIndexB = localBody1;
          c.gearRatio = gearData->gearRatio;

          // AVBD cleanly retrieves the unmodified geometric error from
          // ExtGearJoint
          c.geometricError = -gearData->error;
          c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;

          // Store gear axes in LOCAL body space
          // In PhysX, RevoluteJoint rotation axis is the X axis in joint local
          // frame
          PxQuat frameA = gearData->c2b[0].q;
          PxQuat frameB = gearData->c2b[1].q;

          // Joint rotation axis is X in joint local frame (PhysX convention)
          PxVec3 jointAxis = PxVec3(1.0f, 0.0f, 0.0f);

          // Transform joint axis to body-local space
          // c2b is "constraint to body" transform, so we rotate the joint axis
          // by it
          c.gearAxis0 = frameA.rotate(jointAxis); // Body A local space
          c.gearAxis1 = frameB.rotate(jointAxis); // Body B local space
        }
      } else if (jointType != eJOINT_UNKNOWN &&
                 constraint->constantBlockSize >= sizeof(PhysXJointData)) {
        // Process standard PhysX joints (Spherical, Revolute, Fixed, D6, etc.)
        const PhysXJointData *physXData =
            static_cast<const PhysXJointData *>(constraint->constantBlock);

        // Validate joint data
        const float firstFloat = physXData->invMassScale.linear0;
        if (firstFloat >= 0.5f && firstFloat <= 2.0f &&
            physXData->c2b[0].p.isFinite() && physXData->c2b[1].p.isFinite()) {

          PxVec3 anchorA = physXData->c2b[0].p;
          PxVec3 anchorB = physXData->c2b[1].p;
          PxQuat frameA = physXData->c2b[0].q;
          PxQuat frameB = physXData->c2b[1].q;

          // Handle anchor transformation based on body type:
          // - NULL bodyCore: connected to world, c2b is already in world space
          // - Static body (inverseMass == 0): transform c2b from body local to
          // world space
          // - Dynamic body: c2b stays in body local space for AVBD solver

          if (!constraint->bodyCore0) {
            // Connected to world - c2b[0] is already in world space, keep as is
            // localBody0 = PX_MAX_U32 indicates world anchor
          } else if (body0IsStatic) {
            // Static rigid body - transform anchor to world space
            PxTransform staticPose = constraint->bodyCore0->body2World;
            PxTransform jointFrame = staticPose * physXData->c2b[0];
            anchorA = jointFrame.p;
            frameA = jointFrame.q;
          }

          if (!constraint->bodyCore1) {
            // Connected to world - c2b[1] is already in world space, keep as is
            // localBody1 = PX_MAX_U32 indicates world anchor
          } else if (body1IsStatic) {
            // Static rigid body - transform anchor to world space
            PxTransform staticPose = constraint->bodyCore1->body2World;
            PxTransform jointFrame = staticPose * physXData->c2b[1];
            anchorB = jointFrame.p;
            frameB = jointFrame.q;
          }

          const PxU32 d6CountBefore = numD6;

          switch (jointType) {
          case eJOINT_SPHERICAL: {
            // Convert Spherical joint -> D6 with linear locked, angular free
            // Spherical = ball-and-socket: position locked, rotation free
            // With optional cone limit on swing axes
            if (numD6 < maxD6) {
              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.localFrameA = frameA;
              c.localFrameB = frameB;

              // Linear: all locked (position constraint)
              c.linearMotion = 0;
              c.linearLimitLower = PxVec3(0.0f);
              c.linearLimitUpper = PxVec3(0.0f);

              // Angular: default all FREE
              // 2 bits per axis: 0=LOCKED, 1=LIMITED, 2=FREE
              // ALL FREE = 2|(2<<2)|(2<<4) = 0x2A
              c.angularMotion = 0x2A;
              c.angularLimitLower = PxVec3(-PxPi);
              c.angularLimitUpper = PxVec3(PxPi);

              // Check for cone limit -- implemented as a single cone
              // constraint (not per-axis LIMITED) to match reference
              // spherical solver behavior and avoid 2-axis oscillation.
              if (constraint->constantBlockSize >=
                  sizeof(PhysXSphericalJointData)) {
                const PhysXSphericalJointData *sphericalData =
                    static_cast<const PhysXSphericalJointData *>(
                        constraint->constantBlock);

                // PxSphericalJointFlag::eLIMIT_ENABLED = 0x0002
                if (sphericalData->jointFlags & 0x0002) {
                  // Use the smaller of the two cone angles as limit
                  c.coneAngleLimit = PxMin(sphericalData->limit.yAngle,
                                           sphericalData->limit.zAngle);
                }
              }

              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }

          case eJOINT_FIXED: {
            // Convert Fixed joint -> D6 with all DOFs locked
            if (numD6 < maxD6) {
              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.localFrameA = frameA;
              c.localFrameB = frameB;
              c.linearMotion = 0;   // all position locked
              c.angularMotion = 0;  // all rotation locked
              c.linearLimitLower = PxVec3(0.0f);
              c.linearLimitUpper = PxVec3(0.0f);
              c.angularLimitLower = PxVec3(0.0f);
              c.angularLimitUpper = PxVec3(0.0f);
              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }

          case eJOINT_REVOLUTE: {
            // Convert Revolute joint -> D6 with twist free/limited, rest locked
            if (numD6 < maxD6) {
              const PhysXRevoluteJointData *revoluteData =
                  static_cast<const PhysXRevoluteJointData *>(
                      constraint->constantBlock);

              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.localFrameA = frameA;
              c.localFrameB = frameB;

              c.linearMotion = 0;   // all position locked
              // Twist axis (bit0-1): LIMITED=1 or FREE=2; swing1,swing2 = LOCKED=0
              bool hasLimit = (revoluteData->jointFlags & 0x0001) != 0;
              c.angularMotion = hasLimit ? 0x01 : 0x02; // twist only

              c.linearLimitLower = PxVec3(0.0f);
              c.linearLimitUpper = PxVec3(0.0f);
              if (hasLimit) {
                c.angularLimitLower = PxVec3(revoluteData->limit.lower, 0.0f, 0.0f);
                c.angularLimitUpper = PxVec3(revoluteData->limit.upper, 0.0f, 0.0f);
              } else {
                c.angularLimitLower = PxVec3(-PxPi, 0.0f, 0.0f);
                c.angularLimitUpper = PxVec3(PxPi, 0.0f, 0.0f);
              }

              // Motor/drive support: use post-solve motor (like reference)
              // instead of AL velocity drive (which couples with gear constraints
              // and causes instability).
              if (revoluteData->jointFlags & 0x0002) { // eDRIVE_ENABLED
                c.motorEnabled = 1;
                c.motorTargetVelocity = revoluteData->driveVelocity;
                c.motorMaxForce = revoluteData->driveForceLimit;
              }

              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }

          case eJOINT_PRISMATIC: {
            // Convert Prismatic joint -> D6 with X-axis free/limited, rest locked
            if (numD6 < maxD6) {
              const PhysXPrismaticJointData *prismaticData =
                  static_cast<const PhysXPrismaticJointData *>(
                      constraint->constantBlock);

#if AVBD_JOINT_DEBUG
              if (getAvbdMotorFrameCounter() < AVBD_JOINT_DEBUG_FRAMES) {
                printf("[Parse Prismatic->D6] sizeof(PhysXPrismaticJointData)=%zu, "
                       "flags=%u, limit(L=%f, U=%f)\n",
                       sizeof(PhysXPrismaticJointData),
                       (unsigned int)prismaticData->jointFlags,
                       prismaticData->limit.lower, prismaticData->limit.upper);
              }
#endif

              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.localFrameA = frameA;
              c.localFrameB = frameB;

              // X = LIMITED or FREE, Y&Z = LOCKED
              bool hasLimit = (prismaticData->jointFlags &
                  AVBD_PRISMATIC_LIMIT_ENABLED_FLAG) != 0;
              c.linearMotion = hasLimit ? 0x01 : 0x02; // X only
              c.angularMotion = 0;  // all rotation locked

              if (hasLimit) {
                c.linearLimitLower = PxVec3(prismaticData->limit.lower, 0.0f, 0.0f);
                c.linearLimitUpper = PxVec3(prismaticData->limit.upper, 0.0f, 0.0f);
              } else {
                c.linearLimitLower = PxVec3(-PX_MAX_F32, 0.0f, 0.0f);
                c.linearLimitUpper = PxVec3(PX_MAX_F32, 0.0f, 0.0f);
              }
              c.angularLimitLower = PxVec3(0.0f);
              c.angularLimitUpper = PxVec3(0.0f);

              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }

          case eJOINT_D6: {
            if (numD6 < maxD6) {
              const PhysXD6JointData *d6Data =
                  static_cast<const PhysXD6JointData *>(
                      constraint->constantBlock);

              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              c.localFrameA = frameA;
              c.localFrameB = frameB;

              // Set motion flags from D6 data
              // Each axis uses 2 bits: 0=LOCKED, 1=LIMITED, 2=FREE
              c.linearMotion = 0;
              c.angularMotion = 0;
              for (int i = 0; i < 3; ++i) {
                // Linear axes: X, Y, Z (indices 0, 1, 2)
                // Store motion type directly (0=LOCKED, 1=LIMITED, 2=FREE)
                c.linearMotion |= (d6Data->motion[i] << (i * 2));
                // Angular axes: TWIST, SWING1, SWING2 (indices 3, 4, 5)
                c.angularMotion |= (d6Data->motion[i + 3] << (i * 2));
              }

              // Set limits from D6 data
              c.linearLimitLower =
                  PxVec3(d6Data->linearLimitX.lower, d6Data->linearLimitY.lower,
                         d6Data->linearLimitZ.lower);
              c.linearLimitUpper =
                  PxVec3(d6Data->linearLimitX.upper, d6Data->linearLimitY.upper,
                         d6Data->linearLimitZ.upper);
              c.angularLimitLower =
                  PxVec3(d6Data->twistLimit.lower, -d6Data->swingLimit.yAngle,
                         -d6Data->swingLimit.zAngle);
              c.angularLimitUpper =
                  PxVec3(d6Data->twistLimit.upper, d6Data->swingLimit.yAngle,
                         d6Data->swingLimit.zAngle);

              // Set drive parameters if any drives are active
              c.driveFlags = 0;
              if (d6Data->driving != 0) {
                // Set stiffness and damping from drive parameters
                c.linearStiffness = PxVec3(d6Data->drive[0].stiffness,
                                           d6Data->drive[1].stiffness,
                                           d6Data->drive[2].stiffness);
                c.linearDamping =
                    PxVec3(d6Data->drive[0].damping, d6Data->drive[1].damping,
                           d6Data->drive[2].damping);
                // Angular drive data indices:
                // drive[3] = eSWING (deprecated), also used for eSWING1
                // drive[4] = eTWIST
                // drive[5] = eSLERP, also used for eSWING2
                // For AVBD: angularDamping.x = TWIST, .y = SWING1, .z =
                // SWING2/SLERP
                c.angularStiffness = PxVec3(
                    d6Data->drive[4].stiffness,  // TWIST
                    d6Data->drive[3].stiffness,  // SWING1 (uses SWING slot)
                    d6Data->drive[5].stiffness); // SWING2/SLERP
                c.angularDamping =
                    PxVec3(d6Data->drive[4].damping, // TWIST
                           d6Data->drive[3].damping, // SWING1 (uses SWING slot)
                           d6Data->drive[5].damping); // SWING2/SLERP
                // Map D6 Joint driving flags to AVBD driveFlags format
                // PhysX D6 Joint uses PxD6Drive::Enum bit positions:
                //   eX=0, eY=1, eZ=2 (linear drives - bit 0-2)
                //   eTWIST=4, eSWING1=6, eSWING2=7 (angular drives - need remap
                //   to bit 3-5)
                // AVBD expects: bit 0-2=linear X/Y/Z, bit 3-5=angular X/Y/Z
                c.driveFlags = d6Data->driving &
                               0x07; // Linear drives (eX,eY,eZ) - bit 0-2
                if (d6Data->driving & (1 << PxD6Drive::eTWIST))
                  c.driveFlags |= 1 << 3; // TWIST -> bit 3 (angular X)
                if (d6Data->driving & (1 << PxD6Drive::eSWING1))
                  c.driveFlags |= 1 << 4; // SWING1 -> bit 4 (angular Y)
                if (d6Data->driving & (1 << PxD6Drive::eSWING2))
                  c.driveFlags |= 1 << 5; // SWING2 -> bit 5 (angular Z)
                if (d6Data->driving & (1 << PxD6Drive::eSLERP))
                  c.driveFlags |=
                      1 << 5; // SLERP -> bit 5 (angular Z, reuse SWING2)
                if (d6Data->driving & (1 << PxD6Drive::eSWING))
                  c.driveFlags |= 1 << 3; // SWING (deprecated) -> bit 3
                                          // (angular X, reuse TWIST)

                // Set target drive velocities
                c.driveLinearVelocity = d6Data->driveLinearVelocity;
                c.driveAngularVelocity = d6Data->driveAngularVelocity;

                // Set max drive forces from drive[i].forceLimit
                c.driveLinearForce = PxVec3(d6Data->drive[0].forceLimit,
                                            d6Data->drive[1].forceLimit,
                                            d6Data->drive[2].forceLimit);
                c.driveAngularForce =
                    PxVec3(d6Data->drive[4].forceLimit,  // TWIST
                           d6Data->drive[3].forceLimit,  // SWING1
                           d6Data->drive[5].forceLimit); // SWING2/SLERP
              }

              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }

          } // end switch (jointType)

          // Fill breakable joint info for any D6 constraint created above
          if (numD6 > d6CountBefore) {
            AvbdD6JointConstraint &c = d6Constraints[numD6 - 1];
            restoreJointLambdaFromCache(*this, c,
                                        reinterpret_cast<PxU64>(constraint));
            c.writeBackIndex = constraint->index;
            // Convert break force to break impulse (force * dt)
            // Use the same formula as TGS: impulse = force * simDt
            // dt is not available here, so store raw force; the task will
            // compare lambda (which is impulse) against force * dt at writeback
            c.linBreakImpulse = constraint->linBreakForce;
            c.angBreakImpulse = constraint->angBreakForce;
          }
        } // end else if (jointType != eJOINT_UNKNOWN && ...)
        else if (constraint->constantBlockSize >=
                 sizeof(AvbdSnippetJointData)) {
          const AvbdSnippetJointData *data =
              static_cast<const AvbdSnippetJointData *>(
                  constraint->constantBlock);

          if (data->type == AvbdSnippetJointData::eD6 && numD6 < maxD6) {
            AvbdD6JointConstraint &c = d6Constraints[numD6++];
            c.initDefaults();
            c.header.bodyIndexA = localBody0;
            c.header.bodyIndexB = localBody1;
            c.anchorA = data->pivot0;
            c.anchorB = data->pivot1;

            PxVec3 xAxis = data->axis0.getNormalized();
            PxVec3 yAxis = data->axis1.getNormalized();
            PxVec3 zAxis = xAxis.cross(yAxis).getNormalized();
            yAxis = zAxis.cross(xAxis).getNormalized();

            c.localFrameA = PxQuat(PxMat33(xAxis, yAxis, zAxis));
            c.localFrameB = PxQuat(PxIdentity);

            c.linearLimitLower = PxVec3(data->limitLow);
            c.linearLimitUpper = PxVec3(data->limitHigh);
            c.angularLimitLower = PxVec3(-PxPi);
            c.angularLimitUpper = PxVec3(PxPi);
            c.linearMotion = 0;
            c.angularMotion = 0;

            if (data->limitLow > -PX_MAX_F32 / 2 ||
                data->limitHigh < PX_MAX_F32 / 2) {
              c.linearMotion = 0b010101;
            }

            restoreJointLambdaFromCache(*this, c,
                                        reinterpret_cast<PxU64>(constraint));
          }
        }
      } // end else if (AvbdSnippetJointData)
    } // end if (constraint && constraint->constantBlock && ...)
    edgeIndex = edge.mNextIslandEdge;
  } // end while (edgeIndex != IG_INVALID_EDGE)
} // end prepareAvbdConstraints
