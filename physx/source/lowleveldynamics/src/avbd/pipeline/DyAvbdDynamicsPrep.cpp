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

#include "avbd/pipeline/DyAvbdDynamics.h"
#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/solver/rigid/DyAvbdKinematicShell.h"
#include "DyConstraint.h"
#include "DyConstraintPrep.h"
#include "DyFeatherstoneArticulation.h"
#include "DyIslandManager.h"
#include "PxContact.h"
#include "PxsContactManager.h"
#include "PxsContactManagerState.h"
#include "PxsIslandManagerTypes.h"
#include "PxsRigidBody.h"
#include "PxsSimpleIslandManager.h"
#include "common/PxProfileZone.h"
#include "extensions/PxD6Joint.h"
#include "foundation/PxMath.h"
#include "extensions/PxJoint.h"

#include <cstdint>

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

static PxVec3 computeKinematicContactStep(
    const PxsRigidBody *kinematicBody, const PxVec3 &worldPoint, PxReal dt) {
  if (!kinematicBody || dt <= 0.0f)
    return PxVec3(0.0f);
  const PxsBodyCore &core = kinematicBody->getCore();
  const PxVec3 pointVelocity =
      core.linearVelocity +
      core.angularVelocity.cross(worldPoint - core.body2World.p);
  return pointVelocity * dt;
}

static constexpr physx::PxU16 AVBD_PRISMATIC_LIMIT_ENABLED_FLAG = 0x0002;

// Match the standalone contact cache's 1 mm local-anchor quantization.  The
// cache slot itself is still scoped by the persistent contact-manager index;
// this key validates row identity when NP patch/contact ordering changes.
static PX_FORCE_INLINE PxI32 quantizeAvbdContactAnchor(PxReal value,
                                                       PxReal lengthScale) {
  const PxReal safeLengthScale = PxMax(lengthScale, 1e-6f);
  const PxReal scaled = PxClamp(value * (1000.0f / safeLengthScale),
                                -2147483000.0f,
                                2147483000.0f);
  return static_cast<PxI32>(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
}

static PX_FORCE_INLINE void mixAvbdContactKey(PxU64 &hash, PxU32 value) {
  hash ^= static_cast<PxU64>(value);
  hash *= 1099511628211ull;
}

static PxU64 makeAvbdContactCacheKey(PxU32 globalBody0Idx,
                                     PxU32 globalBody1Idx,
                                     const PxVec3 &localPointA,
                                     const PxVec3 &localPointB,
                                     PxReal lengthScale) {
  PxU64 hash = 14695981039346656037ull;
  mixAvbdContactKey(hash, globalBody0Idx);
  mixAvbdContactKey(hash, globalBody1Idx);
  mixAvbdContactKey(
      hash, static_cast<PxU32>(
                quantizeAvbdContactAnchor(localPointA.x, lengthScale)));
  mixAvbdContactKey(
      hash, static_cast<PxU32>(
                quantizeAvbdContactAnchor(localPointA.y, lengthScale)));
  mixAvbdContactKey(
      hash, static_cast<PxU32>(
                quantizeAvbdContactAnchor(localPointA.z, lengthScale)));
  mixAvbdContactKey(
      hash, static_cast<PxU32>(
                quantizeAvbdContactAnchor(localPointB.x, lengthScale)));
  mixAvbdContactKey(
      hash, static_cast<PxU32>(
                quantizeAvbdContactAnchor(localPointB.y, lengthScale)));
  mixAvbdContactKey(
      hash, static_cast<PxU32>(
                quantizeAvbdContactAnchor(localPointB.z, lengthScale)));
  return hash != 0 ? hash : 1;
}

static PxU64 makeAvbdContactManagerKey(const PxsContactManager *manager,
                                       PxU32 globalBody0Idx,
                                       PxU32 globalBody1Idx) {
  const std::uintptr_t address =
      reinterpret_cast<std::uintptr_t>(manager);
  PxU64 hash = 14695981039346656037ull;
  mixAvbdContactKey(hash, static_cast<PxU32>(address));
  mixAvbdContactKey(
      hash, static_cast<PxU32>(static_cast<PxU64>(address) >> 32));
  mixAvbdContactKey(hash, globalBody0Idx);
  mixAvbdContactKey(hash, globalBody1Idx);
  return hash != 0 ? hash : 1;
}

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
  PxReal distanceMinDist;
  bool mUseDistanceLimit;
  bool mUseNewLinearLimits;
  bool mUseConeLimit;
  bool mUsePyramidLimits;
  PxU8 angularDriveConfig;
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

// Convert the extension concrete type stored by Sc::ConstraintSim into the
// compact joint categories used by AVBD constraint preparation.
static PhysXJointType getJointTypeFromConcreteType(PxU16 concreteType) {
  switch (concreteType) {
  case PxJointConcreteType::eSPHERICAL:
    return eJOINT_SPHERICAL;
  case PxJointConcreteType::eREVOLUTE:
    return eJOINT_REVOLUTE;
  case PxJointConcreteType::ePRISMATIC:
    return eJOINT_PRISMATIC;
  case PxJointConcreteType::eFIXED:
    return eJOINT_FIXED;
  case PxJointConcreteType::eD6:
    return eJOINT_D6;
  case PxJointConcreteType::eGEAR:
    return eJOINT_GEAR;
  default:
    return eJOINT_UNKNOWN;
  }
}

struct AvbdContactPrepSource {
  const PxArray<PxsIndexedContactManager> *contactList;
  PxsContactManagerOutputIterator *outputIterator;
  PxArray<AvbdDynamicsContext::CachedContactManagerState> *managerStateCache;
  const PxArray<AvbdDynamicsContext::CachedLambda> *lambdaCache;
  bool enableLambdaWarmStart;
  PxReal lengthScale;
  PxU16 frameStamp;
};

static PxU32 prepareAvbdContactsImpl(
    const IG::IslandSim *islandSim, const AvbdContactPrepSource &source,
    PxReal dt,
    AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
    AvbdContactConstraint *constraints, PxU32 maxConstraints,
    PxU32 startContactIdx, PxU32 numContactsToProcess, PxU32 bodyOffset,
    AvbdContactPrepSnapshot *snapshots, AvbdContactOutputToken *outputTokens,
    PxU32 outputTokenBase) {

  PxU32 constraintIndex = 0;
  const PxU32 endContactIdx = startContactIdx + numContactsToProcess;
  const PxU32 actualMax =
      snapshots ? endContactIdx
                : PxMin(static_cast<PxU32>(source.contactList->size()),
                        endContactIdx);
  const PxU32 bodyEnd = bodyOffset + islandBodyCount;
  const PxU32 numActiveKinematics =
      snapshots || !islandSim ? 0 : islandSim->getNbActiveKinematics();
  const PxNodeIndex *activeKinematics =
      snapshots || !islandSim ? nullptr : islandSim->getActiveKinematics();
  const PxU16 frameStamp = source.frameStamp;

  for (PxU32 i = startContactIdx;
       i < actualMax && constraintIndex < maxConstraints; ++i) {
    AvbdContactPrepSnapshot *snapshot =
        snapshots ? &snapshots[i] : nullptr;
    const PxsIndexedContactManager *icmPtr =
        snapshot ? nullptr : &(*source.contactList)[i];
    PxsContactManager *cm = snapshot ? nullptr : icmPtr->contactManager;

    if ((!snapshot && !cm) ||
        (snapshot && (!snapshot->eligible ||
                      snapshot->contactManagerIndex == PX_MAX_U32)))
      continue;
    if (snapshot)
      snapshot->emittedResponseRows = 0;

    PxU32 globalBody0Idx = PX_MAX_U32;
    PxU32 globalBody1Idx = PX_MAX_U32;
    PxsRigidBody *kinematicBodyA = nullptr;
    PxsRigidBody *kinematicBodyB = nullptr;

    const PxU8 indexType0 = snapshot ? snapshot->indexType0 : icmPtr->indexType0;
    const PxU8 indexType1 = snapshot ? snapshot->indexType1 : icmPtr->indexType1;
    const PxU64 solverBody0 = snapshot ? snapshot->solverBody0 : icmPtr->solverBody0;
    const PxU64 solverBody1 = snapshot ? snapshot->solverBody1 : icmPtr->solverBody1;
    if (indexType0 == PxsIndexedInteraction::eBODY) {
      globalBody0Idx = static_cast<PxU32>(solverBody0);
    } else if (indexType0 == PxsIndexedInteraction::eKINEMATIC) {
        const PxU32 kinematicIndex = static_cast<PxU32>(solverBody0);
      if (kinematicIndex < numActiveKinematics)
        kinematicBodyA =
            getRigidBodyFromIG(*islandSim, activeKinematics[kinematicIndex]);
    }
    if (indexType1 == PxsIndexedInteraction::eBODY) {
      globalBody1Idx = static_cast<PxU32>(solverBody1);
    } else if (indexType1 == PxsIndexedInteraction::eKINEMATIC) {
      const PxU32 kinematicIndex = static_cast<PxU32>(solverBody1);
      if (kinematicIndex < numActiveKinematics)
        kinematicBodyB =
            getRigidBodyFromIG(*islandSim, activeKinematics[kinematicIndex]);
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

    const PxU32 npIndex = snapshot ? snapshot->npIndex : cm->getWorkUnit().mNpIndex;
    PxsContactManagerOutput *output =
        snapshot ? nullptr
                 : &source.outputIterator->getContactManagerOutput(npIndex);

    const PxU32 nbContacts = snapshot ? snapshot->nbContacts : output->nbContacts;
    const PxU32 nbPatches = snapshot ? snapshot->nbPatches : output->nbPatches;
    if (nbContacts == 0)
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
    const PxU32 cmIdx = snapshot ? snapshot->contactManagerIndex : cm->getIndex();
    PxU8 contactManagerAge = snapshot ? snapshot->contactManagerAge : 255;
    const PxU64 contactManagerKey =
        snapshot ? snapshot->contactManagerKey
                 : makeAvbdContactManagerKey(cm, globalBody0Idx, globalBody1Idx);
    AvbdDynamicsContext::CachedContactManagerState *contactManagerState = nullptr;
    if (source.managerStateCache &&
        cmIdx < source.managerStateCache->size()) {
      contactManagerState = &(*source.managerStateCache)[cmIdx];
      if (!snapshot && contactManagerState->key == contactManagerKey &&
          contactManagerState->frameStamp != 0) {
        const PxU16 age = static_cast<PxU16>(
            frameStamp - contactManagerState->frameStamp);
        contactManagerAge = age >= 255u ? PxU8(255) : PxU8(age);
      }
    }
    const PxU32 managerConstraintStart = constraintIndex;

    const PxU8 *contactData = snapshot ? snapshot->contactPoints : output->contactPoints;
    const PxU8 *patchData = snapshot ? snapshot->contactPatches : output->contactPatches;

    if (!contactData || !patchData)
      continue;

    // Contact modification changes the point stride from PxContact to either
    // PxExtendedContact or PxModifiableContact.  Use the public stream decoder
    // to select the format; fixed sizeof(PxContact) arithmetic silently read
    // later modified points from the middle of the preceding point.
    const PxContactStreamIterator stream(
        patchData, contactData,
        snapshot ? nullptr : output->getInternalFaceIndice(), nbPatches,
        nbContacts);
    if (stream.forceNoResponse)
      continue;
    const PxU32 contactPointSize = stream.contactPointSize;
    const bool hasExtendedContact =
        stream.mStreamFormat != PxContactStreamIterator::eSIMPLE_STREAM;

    for (PxU8 patchIdx = 0; patchIdx < nbPatches; ++patchIdx) {
      const PxContactPatch *patch = reinterpret_cast<const PxContactPatch *>(
          patchData + patchIdx * sizeof(PxContactPatch));

      const PxVec3 normal = patch->normal;
      const PxU32 startContact = patch->startContactIndex;
      const PxU16 numContactsInPatch = patch->nbContacts;

      // Standard PhysX contact reports expose at most two friction anchors per
      // contact patch. The regular PGS/TGS prep fills this PxFrictionPatch
      // while building its solver rows; AVBD ingests the NP stream directly,
      // so construct the same public payload here. Use the first point and the
      // farthest point in the patch as deterministic manifold anchors, then
      // accumulate every AVBD row into its nearest anchor during writeback.
      PxFrictionPatch *reportFrictionPatch =
          output && output->frictionPatches
              ? reinterpret_cast<PxFrictionPatch *>(output->frictionPatches) +
                    patchIdx
              : nullptr;
      PxU32 reportAnchorCount = 0;
      if (reportFrictionPatch) {
        reportFrictionPatch->anchorCount = 0;
        reportFrictionPatch->anchorPositions[0] = PxVec3(0.0f);
        reportFrictionPatch->anchorPositions[1] = PxVec3(0.0f);
        reportFrictionPatch->anchorImpulses[0] = PxVec3(0.0f);
        reportFrictionPatch->anchorImpulses[1] = PxVec3(0.0f);

        if (numContactsInPatch > 0 &&
            (patch->dynamicFriction > 0.0f ||
             patch->staticFriction > 0.0f)) {
          const PxContact *firstContact =
              reinterpret_cast<const PxContact *>(
                  contactData + startContact * contactPointSize);
          reportFrictionPatch->anchorPositions[0] = firstContact->contact;
          reportAnchorCount = 1;

          PxReal farthestDistanceSq = 0.0f;
          PxVec3 farthestPosition = firstContact->contact;
          for (PxU16 anchorCandidate = 1;
               anchorCandidate < numContactsInPatch; ++anchorCandidate) {
            const PxContact *candidate =
                reinterpret_cast<const PxContact *>(
                    contactData +
                    (startContact + anchorCandidate) * contactPointSize);
            const PxReal distanceSq =
                (candidate->contact - firstContact->contact)
                    .magnitudeSquared();
            if (distanceSq > farthestDistanceSq) {
              farthestDistanceSq = distanceSq;
              farthestPosition = candidate->contact;
            }
          }
          if (farthestDistanceSq > 1.0e-12f) {
            reportFrictionPatch->anchorPositions[1] = farthestPosition;
            reportAnchorCount = 2;
          }
          reportFrictionPatch->anchorCount = reportAnchorCount;
        }
      }

      for (PxU16 c = 0;
           c < numContactsInPatch && constraintIndex < maxConstraints; ++c) {
        const PxContact *contact = reinterpret_cast<const PxContact *>(
            contactData + (startContact + c) * contactPointSize);
        const PxExtendedContact *extendedContact =
            hasExtendedContact
                ? static_cast<const PxExtendedContact *>(contact)
                : nullptr;
        const PxReal maxImpulse =
            extendedContact ? extendedContact->maxImpulse : PX_MAX_REAL;

        // Match the shared PhysX contact prep contract: a zero maximum impulse
        // removes the point from response while leaving it available to
        // notification callbacks.
        if (maxImpulse <= 0.0f)
          continue;

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

        const PxVec3 worldContact = contact->contact;
        if (bodyA) {
          constraint.contactPointA =
              bodyA->rotation.rotateInv(contact->contact - bodyA->position);
        } else {
          constraint.contactPointA =
              worldContact +
              computeKinematicContactStep(
                  kinematicBodyA, worldContact, dt);
        }

        if (bodyB) {
          constraint.contactPointB =
              bodyB->rotation.rotateInv(contact->contact - bodyB->position);
        } else {
          constraint.contactPointB =
              worldContact +
              computeKinematicContactStep(
                  kinematicBodyB, worldContact, dt);
        }

        // Lambda & penalty warm-starting (ref: AVBD3D solver.cpp L64-72)
        //   lambda *= alpha * gamma
        //   penalty = clamp(penalty * gamma, PENALTY_MIN, PENALTY_MAX)
        const PxU64 cacheKey = makeAvbdContactCacheKey(
            globalBody0Idx, globalBody1Idx, constraint.contactPointA,
            constraint.contactPointB, source.lengthScale);
        const PxU64 cacheIdx64 =
            static_cast<PxU64>(cmIdx) *
                AvbdDynamicsContext::CONTACT_CACHE_SLOTS_PER_CM +
            (cacheKey & (AvbdDynamicsContext::CONTACT_CACHE_SLOTS_PER_CM - 1u));
        const PxU32 cacheIdx = cacheIdx64 < PX_MAX_U32
                                   ? static_cast<PxU32>(cacheIdx64)
                                   : PX_MAX_U32;
        constraint.cacheIndex = cacheIdx;
        constraint.cacheKey = cacheKey;
        constraint.contactManagerEstablished =
            contactManagerAge == 1 ? 1 : 0;
        constraint.contactImpulseWriteback =
            !snapshot && output && output->contactForces
                ? output->contactForces + startContact + c
                : nullptr;
        if (snapshot && outputTokens) {
          AvbdContactOutputToken &token =
              outputTokens[outputTokenBase + constraintIndex];
          token.targetIndex = snapshot->outputTargetBase + startContact + c;
          token.flags = snapshot->outputTargetFlags;
        }
        constraint.frictionImpulseWriteback = nullptr;
        constraint.frictionSweepImpulse = PxVec3(0.0f);
        constraint.velocityNormalImpulse = -1.0f;
        resetAvbdContactObjectiveProgram(
            constraint.objectiveProgram);
        if (reportFrictionPatch && reportAnchorCount > 0) {
          PxU32 anchorIndex = 0;
          if (reportAnchorCount > 1) {
            const PxReal distance0 =
                (worldContact -
                 reportFrictionPatch->anchorPositions[0])
                    .magnitudeSquared();
            const PxReal distance1 =
                (worldContact -
                 reportFrictionPatch->anchorPositions[1])
                    .magnitudeSquared();
            if (distance1 < distance0)
              anchorIndex = 1;
          }
          constraint.frictionImpulseWriteback =
              &reportFrictionPatch->anchorImpulses[anchorIndex];
        }

        // AVBD warmstart decay constants
        // alpha=0.95, gamma=0.99 => alpha*gamma=0.9405
        const PxReal wsAlpha = 0.95f;
        const PxReal wsGamma = 0.99f;
        const PxReal wsPenaltyMin = 1000.0f;
        const PxReal wsPenaltyMax = 1e9f;

        const bool bodyVsStatic =
            (bodyA != nullptr) != (bodyB != nullptr);
        const PxReal restDist = snapshot ? snapshot->restDistance
                                         : cm->getRestDistance();
        const bool deformableStaticAnchor =
            isDeformableStaticAnchorContact(bodyVsStatic, restDist);

        const AvbdDynamicsContext::CachedLambda *cachedLambda = nullptr;
        if (snapshot) {
          const PxU64 cacheBase = static_cast<PxU64>(cmIdx) *
                                  AvbdDynamicsContext::CONTACT_CACHE_SLOTS_PER_CM;
          if (snapshot->lambdaCacheDirect && cacheIdx64 >= cacheBase) {
            const PxU64 slotOffset = cacheIdx64 - cacheBase;
            if (slotOffset < AvbdDynamicsContext::CONTACT_CACHE_SLOTS_PER_CM)
              cachedLambda = reinterpret_cast<
                  const AvbdDynamicsContext::CachedLambda *>(
                  snapshot->lambdaCacheDirect) + slotOffset;
          } else if (snapshot->lambdaCacheBytes && cacheIdx64 >= cacheBase) {
            const PxU64 slotOffset = cacheIdx64 - cacheBase;
            if (slotOffset < AvbdDynamicsContext::CONTACT_CACHE_SLOTS_PER_CM &&
                snapshot->lambdaCacheSlots) {
              const PxU8 slot = static_cast<PxU8>(slotOffset);
              for (PxU32 compactSlot = 0;
                   compactSlot < snapshot->lambdaCacheSlotCount;
                   ++compactSlot) {
                if (snapshot->lambdaCacheSlots[compactSlot] == slot) {
                  cachedLambda = reinterpret_cast<const AvbdDynamicsContext::CachedLambda *>(
                                     snapshot->lambdaCacheBytes) +
                                 compactSlot;
                  break;
                }
              }
            }
          }
        } else if (source.lambdaCache && cacheIdx < source.lambdaCache->size()) {
          cachedLambda = &(*source.lambdaCache)[cacheIdx];
        }

        if (deformableStaticAnchor) {
          // Entry 108: no lambda/penalty warmstart on deformable mesh anchors.
          // Stale multipliers from the moving surface cause impact blow-up without CCD.
          constraint.header.lambda = 0.0f;
          constraint.tangentLambda0 = 0.0f;
          constraint.tangentLambda1 = 0.0f;
          constraint.header.penalty = wsPenaltyMin;
          constraint.tangentPenalty0 = wsPenaltyMin;
          constraint.tangentPenalty1 = wsPenaltyMin;
          setFrictionStick(constraint, false);
        } else if (source.enableLambdaWarmStart && cachedLambda) {
          const AvbdDynamicsContext::CachedLambda &cached = *cachedLambda;
          const PxU16 frameAge =
              static_cast<PxU16>(frameStamp - cached.frameStamp);
          if (cached.key == cacheKey && cached.frameStamp != 0 &&
              frameAge <= AvbdDynamicsContext::LAMBDA_MAX_AGE) {
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
            setFrictionStick(constraint, cached.stick != 0);
          } else {
            constraint.header.lambda = 0.0f;
            constraint.tangentLambda0 = 0.0f;
            constraint.tangentLambda1 = 0.0f;
            constraint.header.penalty = wsPenaltyMin;
            constraint.tangentPenalty0 = wsPenaltyMin;
            constraint.tangentPenalty1 = wsPenaltyMin;
            setFrictionStick(constraint, false);
          }
        } else {
          constraint.header.lambda = 0.0f;
          constraint.tangentLambda0 = 0.0f;
          constraint.tangentLambda1 = 0.0f;
          constraint.header.penalty = wsPenaltyMin;
          constraint.tangentPenalty0 = wsPenaltyMin;
          constraint.tangentPenalty1 = wsPenaltyMin;
          setFrictionStick(constraint, false);
        }

        constraint.contactNormal = normal;
        constraint.targetVelocity =
            extendedContact ? extendedContact->targetVelocity : PxVec3(0.0f);
        constraint.maxImpulse = maxImpulse;
        constraint.invMassScaleA =
            PxMax(0.0f, patch->mMassModification.linear0);
        constraint.invMassScaleB =
            PxMax(0.0f, patch->mMassModification.linear1);
        constraint.invInertiaScaleA =
            PxMax(0.0f, patch->mMassModification.angular0);
        constraint.invInertiaScaleB =
            PxMax(0.0f, patch->mMassModification.angular1);
        constraint.header.flags = deformableStaticAnchor
                                      ? AvbdContactConstraintFlags::
                                            eDEFORMABLE_STATIC_ANCHOR
                                      : 0;
        // Deformable mesh only: TGS-style separation - restDistance. Plane/stack
        // keep baseline separation.
        constraint.penetrationDepth =
            deformableStaticAnchor ? (contact->separation - restDist)
                                   : contact->separation;
        // Material slice from NP-combined patch (PxCombineMode already applied).
        // Restitution -> post-AL material normal response; mu -> dual + friction.
        constraint.restitution = patch->restitution;
        constraint.friction = patch->dynamicFriction;
        constraint.staticFriction = patch->staticFriction;

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
        constraint.supportClass = AvbdSupportClass::eUnset;

        if (deformableStaticAnchor) {
          // Entry 153: never restore staticPrev from CM-index lambda cache.
          // Contact order within a mesh CM reorders every frame -> aliasing
          // injects multi-metre fictitious mesh steps after long runs.
          // Published-grid path: preserve the current NP anchor and carry its
          // same-xz mesh displacement back to t-dt. Non-grid deformable
          // anchors keep prev = now; CM-index history is intentionally not
          // restored because contact order within a mesh CM is not stable.
          if (AvbdKinematicShell::isActive()) {
            const bool shellApplied =
                AvbdKinematicShell::applyShellNormalAndPrev(
                    constraint, bodyA, bodyB, restDist, t0, t1);
            if (!shellApplied)
              constraint.staticPrevWorldPoint = worldContact;
          } else {
            constraint.staticPrevWorldPoint = worldContact;
          }
        } else if (bodyVsStatic) {
          // A rigid world/kinematic partner has no AVBD solver body.  Its
          // body-vs-static displacement baseline is therefore the world-space
          // contact captured by narrow phase, not the world origin.  Using
          // zero here turns the actor's absolute X/Z coordinates into a
          // fictitious tangential step and injects unbounded friction energy.
          constraint.staticPrevWorldPoint = worldContact;
        } else {
          constraint.staticPrevWorldPoint = PxVec3(0.0f);
        }

        if (snapshot)
          ++snapshot->emittedResponseRows;
        ++constraintIndex;
      }
    }

    // A manager owns established-support semantics only after it emitted at
    // least one response row in the current step.  Notification-only streams,
    // forceNoResponse pairs, and points disabled with maxImpulse=0 must leave a
    // lifecycle gap so a later response is classified as contact onset.
    if (constraintIndex > managerConstraintStart) {
      if (snapshot)
        snapshot->managerStateCommit = 1;
      else if (contactManagerState) {
        contactManagerState->key = contactManagerKey;
        contactManagerState->frameStamp = frameStamp;
      }
    }
  }

  return constraintIndex;
}

PxU32 AvbdDynamicsContext::prepareAvbdContacts(
    const IG::IslandSim &islandSim, PxReal dt,
    AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
    AvbdContactConstraint *constraints, PxU32 maxConstraints,
    PxU32 startContactIdx, PxU32 numContactsToProcess, PxU32 bodyOffset,
    AvbdContactPrepSnapshot *snapshots, AvbdContactOutputToken *outputTokens,
    PxU32 outputTokenBase) {
  if (snapshots) {
    return prepareAvbdContactSnapshots(
        dt, avbdBodies, islandBodyCount, constraints, maxConstraints,
        startContactIdx, numContactsToProcess, bodyOffset, snapshots,
        outputTokens, outputTokenBase, getLengthScale(),
        mEnableLambdaWarmStart, getAvbdFrameStamp());
  }
  const AvbdContactPrepSource source = {
      &mContactList,
      &mOutputIterator,
      &mContactManagerStateCache,
      &mLambdaCache,
      mEnableLambdaWarmStart,
      getLengthScale(),
      getAvbdFrameStamp()};
  return prepareAvbdContactsImpl(
      &islandSim, source, dt, avbdBodies, islandBodyCount, constraints,
      maxConstraints, startContactIdx, numContactsToProcess, bodyOffset,
      snapshots, outputTokens, outputTokenBase);
}

PxU32 physx::Dy::prepareAvbdContactSnapshots(
    PxReal dt, AvbdSolverBody *avbdBodies, PxU32 islandBodyCount,
    AvbdContactConstraint *constraints, PxU32 maxConstraints,
    PxU32 startContactIdx, PxU32 numContactsToProcess, PxU32 bodyOffset,
    AvbdContactPrepSnapshot *snapshots, AvbdContactOutputToken *outputTokens,
    PxU32 outputTokenBase, PxReal lengthScale, bool enableLambdaWarmStart,
    PxU16 frameStamp) {
  const AvbdContactPrepSource source = {
      nullptr, nullptr, nullptr, nullptr, enableLambdaWarmStart, lengthScale,
      frameStamp};
  return prepareAvbdContactsImpl(
      nullptr, source, dt, avbdBodies, islandBodyCount, constraints,
      maxConstraints, startContactIdx, numContactsToProcess, bodyOffset,
      snapshots, outputTokens, outputTokenBase);
}

void AvbdDynamicsContext::prepareAvbdConstraints(
    const IG::IslandSim &islandSim, PxReal dt, AvbdSolverBody *avbdBodies,
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

  numD6 = 0;
  numGear = 0;

  START_ENUMERATING_ISLAND_EDGES(IG::Edge::eCONSTRAINT) {
    GET_CURRENT_ISLAND_EDGE

    Dy::Constraint *constraint = mIslandManager.getConstraint(edgeId);

    if (constraint && constraint->constantBlock &&
        constraint->constantBlockSize > 0 &&
        (constraint->flags &
         static_cast<PxU16>(
             PxConstraintFlag::eDISABLE_CONSTRAINT)) == 0) {

      const PxNodeIndex nodeIndex0 =
          islandSim.mCpuData.getNodeIndex1(edgeId);
      const PxNodeIndex nodeIndex1 =
          islandSim.mCpuData.getNodeIndex2(edgeId);

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
      const bool body0IsKinematic =
          !nodeIndex0.isStaticBody() &&
          islandSim.getNode(nodeIndex0).isKinematic();
      const bool body1IsKinematic =
          !nodeIndex1.isStaticBody() &&
          islandSim.getNode(nodeIndex1).isKinematic();

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

      const PxU16 concreteType = getConstraintConcreteType(constraint->index);
      const PhysXJointType jointType =
          getJointTypeFromConcreteType(concreteType);

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
              // Tag the original joint type so solver-path source-detection
              // (DyAvbdSolverJointPath.cpp::getConstraintSourceType and the
              // prismatic warmstart-clear in DyAvbdDynamics.cpp) can tell a
              // native PxSphericalJoint apart from an articulation internal
              // joint that happens to share the same motion-mask shape.
              c.header.type = AvbdConstraintType::eJOINT_SPHERICAL;
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

              // Check for cone limit -- implemented as the same single
              // elliptical constraint emitted by ExtSphericalJoint (not two
              // independent per-axis LIMITED rows).
              if (constraint->constantBlockSize >=
                  sizeof(PhysXSphericalJointData)) {
                const PhysXSphericalJointData *sphericalData =
                    static_cast<const PhysXSphericalJointData *>(
                        constraint->constantBlock);

                // PxSphericalJointFlag::eLIMIT_ENABLED = 0x0002
                if (sphericalData->jointFlags & 0x0002) {
                  c.sourceFlags |= AvbdD6JointConstraint::
                      eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE;
                  c.coneAngleLimit = sphericalData->limit.yAngle;
                  c.coneAngleLimitZ = sphericalData->limit.zAngle;
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
              c.header.type = AvbdConstraintType::eJOINT_FIXED;
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
              c.header.type = AvbdConstraintType::eJOINT_REVOLUTE;
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

              // Native revolute motors are finalized by the strict velocity
              // owner. Supported centered gear topology is solved as one
              // coupled velocity objective; unsupported mixtures fail closed.
              if (revoluteData->jointFlags & 0x0002) { // eDRIVE_ENABLED
                c.motorEnabled = 1;
                c.motorTargetVelocity = revoluteData->driveVelocity;
                c.motorMaxForce = revoluteData->driveForceLimit;
                c.motorGearRatio = revoluteData->driveGearRatio;
                if (revoluteData->jointFlags & 0x0004) // eDRIVE_FREESPIN
                  c.sourceFlags |= AvbdD6JointConstraint::
                      eNATIVE_REVOLUTE_MOTOR_FREESPIN;
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

              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.type = AvbdConstraintType::eJOINT_PRISMATIC;
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.anchorA = anchorA;
              c.anchorB = anchorB;
              // Forward the authored joint frames verbatim, matching the
              // contract used by every other joint type (and avbd_standalone's
              // addD6Joint / addPrismaticJoint).
              //
              // Earlier branch-local probes rebuilt prismatic localFrameB
              // every frame as `bodyB.q^{-1} * worldFrameA` to mask transient
              // direct-hit divergence (AVBD audit Entry 025). That probe had
              // the side effect of making the prismatic generic angular-lock
              // rows blind to real violations: computeAngularError() always
              // saw the rest-relative quaternion as identity, so the dual
              // update never accumulated lambda to resist gravity-driven
              // body rotation. SnippetJoint's prismatic chain then free-fell
              // to the ground despite Y/Z linear and all-angular locks
              // (Entry 094 evidence). Keep the authored frame so the angular
              // error reflects the actual joint violation.
              c.localFrameA = frameA.getNormalized();
              c.localFrameB = frameB.getNormalized();

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
              if ((constraint->flags &
                   static_cast<PxU16>(
                       PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES)) != 0) {
                c.sourceFlags |=
                    AvbdD6JointConstraint::eD6_DRIVE_LIMITS_ARE_FORCES;
              }

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
              const bool bothSwingAxesLimited =
                  d6Data->motion[PxD6Axis::eSWING1] ==
                      PxD6Motion::eLIMITED &&
                  d6Data->motion[PxD6Axis::eSWING2] ==
                      PxD6Motion::eLIMITED;
              if (bothSwingAxesLimited && d6Data->mUseConeLimit) {
                c.sourceFlags |= AvbdD6JointConstraint::
                    eD6_LEGACY_CONE_LIMIT_ACTIVE;
                c.coneAngleLimit = d6Data->swingLimit.yAngle;
                c.coneAngleLimitZ = d6Data->swingLimit.zAngle;
              }

              // Set drive parameters if any drives are active
              c.driveFlags = 0;
              c.driveAccelerationFlags = 0;
              c.driveOutputForceFlags = 0;
              if (d6Data->driving != 0) {
                // Set stiffness and damping from drive parameters
                c.linearStiffness = PxVec3(d6Data->drive[0].stiffness,
                                           d6Data->drive[1].stiffness,
                                           d6Data->drive[2].stiffness);
                c.linearDamping =
                    PxVec3(d6Data->drive[0].damping, d6Data->drive[1].damping,
                           d6Data->drive[2].damping);
                // PhysX 5.9 angular drive data indices:
                // drive[3] = eTWIST
                // drive[4] = eSWING1
                // drive[5] = eSWING2, also used for eSLERP
                // For AVBD: angularDamping.x = TWIST, .y = SWING1, .z =
                // SWING2/SLERP
                c.angularStiffness = PxVec3(
                    d6Data->drive[3].stiffness,  // TWIST
                    d6Data->drive[4].stiffness,  // SWING1
                    d6Data->drive[5].stiffness); // SWING2/SLERP
                c.angularDamping =
                    PxVec3(d6Data->drive[3].damping, // TWIST
                           d6Data->drive[4].damping, // SWING1
                           d6Data->drive[5].damping); // SWING2/SLERP
                // Map D6 Joint driving flags to AVBD driveFlags format
                // PhysX D6 Joint uses PxD6Drive::Enum bit positions:
                //   eX=0, eY=1, eZ=2 (linear drives - bit 0-2)
                //   eTWIST=3, eSWING1=4, eSWING2=5, eSLERP=6
                // AVBD expects: bit 0-2=linear X/Y/Z, bit 3-5=angular X/Y/Z
                c.driveFlags = d6Data->driving &
                               0x07; // Linear drives (eX,eY,eZ) - bit 0-2
                const PxU32 accelerationFlag =
                    static_cast<PxU32>(PxD6JointDriveFlag::eACCELERATION);
                const PxU32 outputForceFlag =
                    static_cast<PxU32>(PxD6JointDriveFlag::eOUTPUT_FORCE);
                for (PxU32 drive = 0; drive < 3; ++drive) {
                  if ((c.driveFlags & (1u << drive)) != 0 &&
                      (d6Data->drive[drive].flags & accelerationFlag) != 0)
                    c.driveAccelerationFlags |= 1u << drive;
                  if ((c.driveFlags & (1u << drive)) != 0 &&
                      (d6Data->drive[drive].flags & outputForceFlag) != 0)
                    c.driveOutputForceFlags |= 1u << drive;
                }
                if (d6Data->driving & (1 << PxD6Drive::eTWIST))
                  c.driveFlags |= 1 << 3; // TWIST -> bit 3 (angular X)
                if (d6Data->driving & (1 << PxD6Drive::eSWING1))
                  c.driveFlags |= 1 << 4; // SWING1 -> bit 4 (angular Y)
                if (d6Data->driving & (1 << PxD6Drive::eSWING2))
                  c.driveFlags |= 1 << 5; // SWING2 -> bit 5 (angular Z)
                if (d6Data->driving & (1 << PxD6Drive::eSLERP)) {
                  c.driveFlags |=
                      1 << 5; // SLERP -> bit 5 (angular Z, reuse SWING2)
                  c.sourceFlags |= AvbdD6JointConstraint::eD6_SLERP_DRIVE;
                }

                const PxU32 angularDriveIndices[3] = {
                    PxD6Drive::eTWIST, PxD6Drive::eSWING1,
                    PxD6Drive::eSWING2};
                for (PxU32 axis = 0; axis < 3; ++axis) {
                  const PxU32 drive = angularDriveIndices[axis];
                  if ((c.driveFlags & (1u << (axis + 3))) != 0 &&
                      (d6Data->drive[drive].flags & accelerationFlag) != 0)
                    c.driveAccelerationFlags |= 1u << (axis + 3);
                  if ((c.driveFlags & (1u << (axis + 3))) != 0 &&
                      (d6Data->drive[drive].flags & outputForceFlag) != 0)
                    c.driveOutputForceFlags |= 1u << (axis + 3);
                }

                // Set target drive velocities
                c.driveLinearPosition = d6Data->drivePosition.p;
                c.driveAngularPosition = d6Data->drivePosition.q;
                c.driveLinearVelocity = d6Data->driveLinearVelocity;
                c.driveAngularVelocity = d6Data->driveAngularVelocity;

                // Set max drive forces from drive[i].forceLimit
                c.driveLinearForce = PxVec3(d6Data->drive[0].forceLimit,
                                            d6Data->drive[1].forceLimit,
                                            d6Data->drive[2].forceLimit);
                c.driveAngularForce =
                    PxVec3(d6Data->drive[3].forceLimit,  // TWIST
                           d6Data->drive[4].forceLimit,  // SWING1
                           d6Data->drive[5].forceLimit); // SWING2/SLERP
              }

              c.header.rho = AvbdConstants::AVBD_DEFAULT_PENALTY_RHO_HIGH;
            }
            break;
          }

          } // end switch (jointType)

          // Fill breakable joint info for any D6 constraint created above.
          //
          // The AVBD break threshold contract stores the authored
          // PxConstraint::setBreakForce() value verbatim (no dt scaling),
          // and DyAvbdTasks.cpp compares |lambda| > breakForce in force
          // space. Install the threshold BEFORE the warmstart restore so
          // restoreJointLambdaFromCache() can apply the breakable-only
          // lambda reseed policy that prevents stiff-chain ratcheting
          // (see DyAvbdDynamics.cpp).
          if (numD6 > d6CountBefore) {
            AvbdD6JointConstraint &c = d6Constraints[numD6 - 1];
            // A kinematic is represented by the same no-solver-body sentinel
            // as rigid static/world, but its prescribed motion remains part
            // of every velocity/damping objective.  Preserve that motion
            // explicitly instead of treating the endpoint as stationary.
            if (body0IsKinematic && constraint->bodyCore0)
              c.externalAngularStepA =
                  constraint->bodyCore0->angularVelocity * dt;
            if (body1IsKinematic && constraint->bodyCore1)
              c.externalAngularStepB =
                  constraint->bodyCore1->angularVelocity * dt;
            c.linBreakImpulse = constraint->linBreakForce;
            c.angBreakImpulse = constraint->angBreakForce;
            c.writeBackIndex = constraint->index;
            restoreJointLambdaFromCache(*this, c,
                                        reinterpret_cast<PxU64>(constraint));
          }
          } // end validated standard joint data
        } // end standard joint route
        else if (jointType == eJOINT_UNKNOWN && constraint->solverPrep &&
                 (constraint->flags &
                  static_cast<PxU16>(
                      PxConstraintFlag::eDISABLE_CONSTRAINT)) == 0 &&
                 numD6 < maxD6) {
          // Unknown extension/custom constraints are defined by their public
          // solverPrep rows. Admit the complete row set only when every row is
          // a unit-mass-scaled hard, force-spring, restitution, or pure
          // acceleration-spring damping row.  Each public mode has an
          // independent SnippetCustomJoint authority; non-unit mass scaling
          // and breakable multi-row sets remain fail-closed.
          Px1DConstraint rows[MAX_CONSTRAINT_ROWS];
          setupConstraintRows(rows, MAX_CONSTRAINT_ROWS);
          PxConstraintInvMassScale invMassScale(1.0f, 1.0f, 1.0f, 1.0f);
          PxVec3p bodyAWorldOffset(0.0f);
          PxVec3p cAtW(0.0f), cBtW(0.0f);
          const PxTransform identity(PxIdentity);
          const PxTransform &bodyFrame0 =
              constraint->body0 ? constraint->body0->getPose() : identity;
          const PxTransform &bodyFrame1 =
              constraint->body1 ? constraint->body1->getPose() : identity;
          const bool useExtendedLimits =
              (constraint->flags &
               static_cast<PxU16>(
                   PxConstraintFlag::eENABLE_EXTENDED_LIMITS)) != 0;
          const PxU32 rowCount = (*constraint->solverPrep)(
              rows, bodyAWorldOffset, MAX_CONSTRAINT_ROWS, invMassScale,
              constraint->constantBlock, bodyFrame0, bodyFrame1,
              useExtendedLimits, cAtW, cBtW);

          const bool unitMassScale =
              PxAbs(invMassScale.linear0 - 1.0f) <= 1e-6f &&
              PxAbs(invMassScale.angular0 - 1.0f) <= 1e-6f &&
              PxAbs(invMassScale.linear1 - 1.0f) <= 1e-6f &&
              PxAbs(invMassScale.angular1 - 1.0f) <= 1e-6f;

          bool supportedRows =
              rowCount > 0 && rowCount <= MAX_CONSTRAINT_ROWS &&
              rowCount <= maxD6 - numD6 && unitMassScale;
          const bool multiRow = rowCount > 1;
          const bool multiRowBreakable =
              multiRow && (constraint->linBreakForce < PX_MAX_F32 ||
                           constraint->angBreakForce < PX_MAX_F32);
          for (PxU32 rowIndex = 0;
               supportedRows && rowIndex < rowCount; ++rowIndex) {
            const Px1DConstraint &row = rows[rowIndex];
            const PxU16 springFlag =
                static_cast<PxU16>(Px1DConstraintFlag::eSPRING);
            const PxU16 accelerationSpringFlag =
                static_cast<PxU16>(
                    Px1DConstraintFlag::eACCELERATION_SPRING);
            const PxU16 restitutionFlag =
                static_cast<PxU16>(Px1DConstraintFlag::eRESTITUTION);
            const PxU16 driveLimitFlag =
                static_cast<PxU16>(Px1DConstraintFlag::eHAS_DRIVE_LIMIT);
            const bool spring = (row.flags & springFlag) != 0;
            const bool accelerationSpring =
                (row.flags & accelerationSpringFlag) != 0;
            const bool restitution = (row.flags & restitutionFlag) != 0;
            const bool driveLimit = (row.flags & driveLimitFlag) != 0;
            const bool finiteRow =
                row.linear0.isFinite() && row.angular0.isFinite() &&
                row.linear1.isFinite() && row.angular1.isFinite() &&
                PxIsFinite(row.geometricError) &&
                PxIsFinite(row.velocityTarget) &&
                PxIsFinite(row.minImpulse) &&
                PxIsFinite(row.maxImpulse);
            const bool nonzeroJacobian =
                row.linear0.magnitudeSquared() +
                    row.angular0.magnitudeSquared() +
                    row.linear1.magnitudeSquared() +
                    row.angular1.magnitudeSquared() >
                1e-12f;
            const bool pureAccelerationDamping =
                spring && accelerationSpring &&
                PxAbs(row.mods.spring.stiffness) <= 1e-6f &&
                PxIsFinite(row.mods.spring.damping) &&
                row.mods.spring.damping > 0.0f &&
                PxAbs(row.geometricError) <= 1e-6f;
            const bool forceSpring =
                spring && !accelerationSpring &&
                PxIsFinite(row.mods.spring.stiffness) &&
                row.mods.spring.stiffness >= 0.0f &&
                PxIsFinite(row.mods.spring.damping) &&
                row.mods.spring.damping >= 0.0f;
            const bool restitutionRow =
                !spring && restitution &&
                PxIsFinite(row.mods.bounce.restitution) &&
                row.mods.bounce.restitution >= 0.0f &&
                row.mods.bounce.restitution <= 1.0f &&
                PxIsFinite(row.mods.bounce.velocityThreshold) &&
                row.mods.bounce.velocityThreshold >= 0.0f;
            const bool hardRow = !spring && !restitution;
            const bool dampingOutputForce =
                pureAccelerationDamping &&
                (row.flags &
                 static_cast<PxU16>(
                     Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;

            supportedRows =
                finiteRow && nonzeroJacobian &&
                row.minImpulse <= row.maxImpulse &&
                (!driveLimit || spring) &&
                (hardRow || forceSpring || restitutionRow ||
                 pureAccelerationDamping) &&
                !dampingOutputForce &&
                !multiRowBreakable;
          }

          if (supportedRows) {
            for (PxU32 rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
              const Px1DConstraint &row = rows[rowIndex];
              const bool accelerationDamping =
                  (row.flags &
                   static_cast<PxU16>(
                       Px1DConstraintFlag::eSPRING)) != 0 &&
                  (row.flags &
                   static_cast<PxU16>(
                       Px1DConstraintFlag::eACCELERATION_SPRING)) != 0;
              const bool forceSpring =
                  (row.flags &
                   static_cast<PxU16>(Px1DConstraintFlag::eSPRING)) != 0 &&
                  !accelerationDamping;
              const bool restitution =
                  (row.flags &
                   static_cast<PxU16>(
                       Px1DConstraintFlag::eRESTITUTION)) != 0 &&
                  !forceSpring;
              AvbdD6JointConstraint &c = d6Constraints[numD6++];
              c.initDefaults();
              c.header.type = AvbdConstraintType::eJOINT_CUSTOM_1D;
              c.header.bodyIndexA = localBody0;
              c.header.bodyIndexB = localBody1;
              c.header.rho = forceSpring
                                 ? row.mods.spring.stiffness
                                 : (accelerationDamping || restitution
                                        ? 0.0f
                                        : AvbdConstants::
                                              AVBD_DEFAULT_PENALTY_RHO_HIGH);
              c.header.damping =
                  (accelerationDamping || forceSpring)
                      ? row.mods.spring.damping
                      : 0.0f;
              c.linearMotion = 0x2A;  // all standard D6 rows FREE
              c.angularMotion = 0x2A;
              if (accelerationDamping)
                c.sourceFlags |= AvbdD6JointConstraint::
                    eGENERIC_ACCELERATION_DAMPING_1D_ROW;
              else if (forceSpring)
                c.sourceFlags |=
                    AvbdD6JointConstraint::eGENERIC_FORCE_SPRING_1D_ROW;
              else if (restitution)
                c.sourceFlags |=
                    AvbdD6JointConstraint::eGENERIC_RESTITUTION_1D_ROW;
              else
                c.sourceFlags |=
                    AvbdD6JointConstraint::eGENERIC_HARD_1D_ROW;
              if (multiRow) {
                c.sourceFlags |=
                    AvbdD6JointConstraint::eGENERIC_MULTI_ROW;
                if (rowIndex == 0)
                  c.sourceFlags |=
                      AvbdD6JointConstraint::eGENERIC_MULTI_ROW_LEADER;
              }

              c.genericLinearA = row.linear0;
              c.genericAngularA = row.angular0;
              c.genericLinearB = -row.linear1;
              c.genericAngularB = -row.angular1;
              c.genericGeometricError = row.geometricError;
              c.genericVelocityTarget = row.velocityTarget;
              const bool scaleDriveLimitToImpulse =
                  (row.flags &
                   static_cast<PxU16>(
                       Px1DConstraintFlag::eHAS_DRIVE_LIMIT)) != 0 &&
                  (constraint->flags &
                   static_cast<PxU16>(
                       PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES)) != 0;
              const PxReal impulseScale =
                  scaleDriveLimitToImpulse ? dt : 1.0f;
              c.genericMinImpulse = row.minImpulse * impulseScale;
              c.genericMaxImpulse = row.maxImpulse * impulseScale;
              c.genericRestitution =
                  restitution ? row.mods.bounce.restitution : 0.0f;
              c.genericBounceThreshold =
                  restitution ? row.mods.bounce.velocityThreshold : 0.0f;
              c.genericReferencePositionA = bodyFrame0.p;
              c.genericReferenceRotationA = bodyFrame0.q;
              c.genericReferencePositionB = bodyFrame1.p;
              c.genericReferenceRotationB = bodyFrame1.q;
              c.genericRowFlags = row.flags;
              c.genericSolveHint = row.solveHint;
              // Match Dy::writeBack1DStep exactly.  PxConstraint::getForce()
              // exposes actor-0's wrench about body0WorldOffset rather than
              // the raw center-of-mass solver Jacobian:
              //   angular =
              //       (angular0 + linear0 x (cA2w - bodyFrame0.p)
              //        - body0WorldOffset x linear0) * impulse.
              // solverPrep is invoked every frame, so freezing this authored
              // writeback Jacobian here has the same lifetime as the row.
              c.genericAngularAWriteback =
                  row.angular0 +
                  row.linear0.cross(PxVec3(cAtW) - bodyFrame0.p) -
                  PxVec3(bodyAWorldOffset).cross(row.linear0);

              c.linBreakImpulse = constraint->linBreakForce;
              c.angBreakImpulse = constraint->angBreakForce;
              c.writeBackIndex = constraint->index;
              if (!multiRow) {
                restoreJointLambdaFromCache(
                    *this, c, reinterpret_cast<PxU64>(constraint));
              }
            }
          }
        } else if (!constraint->solverPrep &&
                   constraint->constantBlockSize >=
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
    } // end if (constraint && constraint->constantBlock && ...)
    GET_NEXT_ISLAND_EDGE
  } // end island constraint-edge enumeration
} // end prepareAvbdConstraints
