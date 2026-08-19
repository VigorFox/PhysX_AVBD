// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "DyAvbdKernelLabCapture.h"

#include "foundation/PxMemory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace physx {
namespace Dy {

namespace {

static const PxU64 kFnvOffset = 14695981039346656037ull;
static const PxU64 kFnvPrime = 1099511628211ull;

PX_FORCE_INLINE PxU32 readEnvironmentU32(const char* name,
                                         PxU32 defaultValue) {
  const char* value = std::getenv(name);
  if (!value || !value[0])
    return defaultValue;
  char* end = NULL;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || end == value || *end != '\0' || parsed > PX_MAX_U32)
    return defaultValue;
  return static_cast<PxU32>(parsed);
}

bool copyPath(char (&destination)[512], const char* source) {
  const char* value = source && source[0] ? source :
      "avbd_kernel_lab_trace.bin";
  PxU32 i = 0;
  for (; i + 1u < sizeof(destination) && value[i]; ++i)
    destination[i] = value[i];
  if (value[i]) {
    destination[0] = '\0';
    return false;
  }
  destination[i] = '\0';
  return true;
}

const char* captureStateName(PxU32 state) {
  switch (state) {
  case 0u:
    return "armed";
  case 1u:
    return "reserved";
  case 2u:
    return "pre-range-ready";
  case 3u:
    return "ready-to-write";
  case 4u:
    return "rejected";
  case 5u:
    return "written";
  case 6u:
    return "write-failed";
  default:
    return "unknown";
  }
}

PX_FORCE_INLINE void hashBytes(PxU64& hash, const void* data, PxU64 bytes) {
  const PxU8* source = static_cast<const PxU8*>(data);
  for (PxU64 i = 0; i < bytes; ++i) {
    hash ^= source[i];
    hash *= kFnvPrime;
  }
}

template <typename T>
PX_FORCE_INLINE void hashArray(PxU64& hash, const PxArray<T>& values) {
  if (!values.empty())
    hashBytes(hash, values.begin(), PxU64(values.size()) * sizeof(T));
}

template <typename T>
PX_FORCE_INLINE bool writeArray(FILE* file, const PxArray<T>& values) {
  if (values.empty())
    return true;
  return std::fwrite(values.begin(), sizeof(T), values.size(), file) ==
      values.size();
}

PX_FORCE_INLINE bool writeObject(FILE* file, const void* value, size_t size) {
  return std::fwrite(value, 1u, size, file) == size;
}

PX_FORCE_INLINE void storeVec3(PxF32 (&destination)[3], const PxVec3& source) {
  destination[0] = source.x;
  destination[1] = source.y;
  destination[2] = source.z;
}

PX_FORCE_INLINE void storeQuat(PxF32 (&destination)[4], const PxQuat& source) {
  destination[0] = source.x;
  destination[1] = source.y;
  destination[2] = source.z;
  destination[3] = source.w;
}

void storeBody(AvbdKernelLabTraceBody& destination,
               const AvbdSolverBody& source) {
  storeVec3(destination.position, source.position);
  destination.invMass = source.invMass;
  storeQuat(destination.rotation, source.rotation);
  storeVec3(destination.inertialPosition, source.inertialPosition);
  storeQuat(destination.inertialRotation, source.inertialRotation);
  storeVec3(destination.prevPosition, source.prevPosition);
  storeQuat(destination.prevRotation, source.prevRotation);
  destination.invInertiaWorld[0] = source.invInertiaWorld.column0.x;
  destination.invInertiaWorld[1] = source.invInertiaWorld.column0.y;
  destination.invInertiaWorld[2] = source.invInertiaWorld.column0.z;
  destination.invInertiaWorld[3] = source.invInertiaWorld.column1.x;
  destination.invInertiaWorld[4] = source.invInertiaWorld.column1.y;
  destination.invInertiaWorld[5] = source.invInertiaWorld.column1.z;
  destination.invInertiaWorld[6] = source.invInertiaWorld.column2.x;
  destination.invInertiaWorld[7] = source.invInertiaWorld.column2.y;
  destination.invInertiaWorld[8] = source.invInertiaWorld.column2.z;
  destination.nodeIndex = source.nodeIndex;
  destination.lockFlags = source.lockFlags;
}

void storeContact(AvbdKernelLabTraceContact& destination,
                  const AvbdContactConstraint& source) {
  destination.bodyIndexA = source.header.bodyIndexA;
  destination.bodyIndexB = source.header.bodyIndexB;
  destination.type = source.header.type;
  destination.flags = source.header.flags;
  destination.compliance = source.header.compliance;
  destination.damping = source.header.damping;
  destination.lambda = source.header.lambda;
  destination.rho = source.header.rho;
  destination.penalty = source.header.penalty;
  destination.colorGroup = source.header.colorGroup;
  destination.reserved0 = 0;
  storeVec3(destination.contactPointA, source.contactPointA);
  destination.penetrationDepth = source.penetrationDepth;
  storeVec3(destination.contactPointB, source.contactPointB);
  destination.restitution = source.restitution;
  storeVec3(destination.contactNormal, source.contactNormal);
  destination.friction = source.friction;
  storeVec3(destination.targetVelocity, source.targetVelocity);
  destination.maxImpulse = source.maxImpulse;
  destination.invMassScaleA = source.invMassScaleA;
  destination.invMassScaleB = source.invMassScaleB;
  destination.invInertiaScaleA = source.invInertiaScaleA;
  destination.invInertiaScaleB = source.invInertiaScaleB;
  storeVec3(destination.tangent0, source.tangent0);
  destination.tangentLambda0 = source.tangentLambda0;
  storeVec3(destination.tangent1, source.tangent1);
  destination.tangentLambda1 = source.tangentLambda1;
  destination.staticFriction = source.staticFriction;
  destination.tangentPenalty0 = source.tangentPenalty0;
  destination.tangentPenalty1 = source.tangentPenalty1;
  destination.C0 = source.C0;
}

PX_FORCE_INLINE void storeOwnerPose(AvbdKernelLabTraceOwnerPose& destination,
                                    const AvbdSolverBody& source) {
  storeVec3(destination.position, source.position);
  storeQuat(destination.rotation, source.rotation);
}

} // namespace

AvbdKernelLabCapture::AvbdKernelLabCapture()
    : mState(eARMED),
      mTargetIsland(readEnvironmentU32("PHYSX_AVBD_KERNEL_LAB_CAPTURE_ISLAND", 0u)),
      mTargetIteration(readEnvironmentU32(
          "PHYSX_AVBD_KERNEL_LAB_CAPTURE_ITERATION", 0u)),
      mTargetColor(readEnvironmentU32("PHYSX_AVBD_KERNEL_LAB_CAPTURE_COLOR", 0u)),
      mMinimumBodies(readEnvironmentU32(
          "PHYSX_AVBD_KERNEL_LAB_CAPTURE_MIN_BODIES", 512u)),
      mTicket(1u),
      mReservedBodyCount(0),
      mReservedContactCount(0),
      mReservedRefCount(0),
      mOutputPathValid(false) {
  PxMemZero(&mHeader, sizeof(mHeader));
  mOutputPathValid =
      copyPath(mOutputPath, std::getenv("PHYSX_AVBD_KERNEL_LAB_CAPTURE_PATH"));
}

AvbdKernelLabCapture::~AvbdKernelLabCapture() {
  flush();
  // This object only exists when the explicit capture environment flag is
  // enabled. Emit one cold-path terminal record so an intentionally
  // fail-closed request cannot be mistaken for a missing hook or a failed
  // file write.
  std::printf(
      "[AVBD_KERNEL_LAB_CAPTURE] schema=%u state=%s reservedBodies=%u "
      "reservedContacts=%u capturedBodies=%u capturedContacts=%u output=%s\n",
      unsigned(eAVBD_KERNEL_LAB_TRACE_SCHEMA),
      captureStateName(static_cast<PxU32>(mState)), unsigned(mReservedBodyCount),
      unsigned(mReservedContactCount), unsigned(mBodies.size()),
      unsigned(mContacts.size()), mOutputPath);
}

bool AvbdKernelLabCapture::reserveArrays(PxU32 bodyCount, PxU32 contactCount,
                                         PxU32 maxRefs) {
  return mBodies.reserve(bodyCount) && mContacts.reserve(contactCount) &&
      mMapOffsets.reserve(bodyCount + 1u) && mMapCounts.reserve(bodyCount) &&
      mMapIndices.reserve(maxRefs) && mColorOffsets.reserve(bodyCount + 1u) &&
      mColorBodies.reserve(bodyCount) && mOwnerOrder.reserve(bodyCount) &&
      mPostOwnerPoses.reserve(bodyCount);
}

void AvbdKernelLabCapture::reject() {
  if (mState != eWRITTEN && mState != eWRITE_FAILED)
    mState = eREJECTED;
}

PxU32 AvbdKernelLabCapture::reserve(PxU32 islandIndex, PxU32 bodyCount,
                                    PxU32 contactCount) {
  if (mState != eARMED ||
      !mOutputPathValid ||
      (mTargetIsland != PX_MAX_U32 && islandIndex != mTargetIsland) ||
      bodyCount < mMinimumBodies || contactCount == 0 ||
      contactCount > PX_MAX_U32 / 2u) {
    return eINVALID_TICKET;
  }
  const PxU32 maxRefs = contactCount * 2u;
  if (!reserveArrays(bodyCount, contactCount, maxRefs)) {
    reject();
    return eINVALID_TICKET;
  }
  mReservedBodyCount = bodyCount;
  mReservedContactCount = contactCount;
  mReservedRefCount = maxRefs;
  mState = eRESERVED;
  return mTicket;
}

bool AvbdKernelLabCapture::finalizePayloadHashes() {
  const PxU64 bodyBytes = PxU64(mBodies.size()) * sizeof(AvbdKernelLabTraceBody);
  const PxU64 contactBytes =
      PxU64(mContacts.size()) * sizeof(AvbdKernelLabTraceContact);
  const PxU64 mapBytes = PxU64(mMapOffsets.size() + mMapCounts.size() +
      mMapIndices.size() + mColorOffsets.size() + mColorBodies.size() +
      mOwnerOrder.size()) * sizeof(PxU32);
  const PxU64 poseBytes =
      PxU64(mPostOwnerPoses.size()) * sizeof(AvbdKernelLabTraceOwnerPose);
  const PxU64 maxU64 = ~PxU64(0);
  if (bodyBytes > maxU64 - contactBytes ||
      bodyBytes + contactBytes > maxU64 - mapBytes ||
      bodyBytes + contactBytes + mapBytes > maxU64 - poseBytes) {
    return false;
  }
  mHeader.payloadBytes = bodyBytes + contactBytes + mapBytes + poseBytes;
  PxU64 payloadHash = kFnvOffset;
  hashArray(payloadHash, mBodies);
  hashArray(payloadHash, mContacts);
  hashArray(payloadHash, mMapOffsets);
  hashArray(payloadHash, mMapCounts);
  hashArray(payloadHash, mMapIndices);
  hashArray(payloadHash, mColorOffsets);
  hashArray(payloadHash, mColorBodies);
  hashArray(payloadHash, mOwnerOrder);
  hashArray(payloadHash, mPostOwnerPoses);
  mHeader.payloadHash = payloadHash;

  PxU64 topologyHash = kFnvOffset;
  hashArray(topologyHash, mMapOffsets);
  hashArray(topologyHash, mMapCounts);
  hashArray(topologyHash, mMapIndices);
  hashArray(topologyHash, mColorOffsets);
  hashArray(topologyHash, mColorBodies);
  hashArray(topologyHash, mOwnerOrder);
  for (PxU32 i = 0; i < mContacts.size(); ++i) {
    hashBytes(topologyHash, &mContacts[i].bodyIndexA,
              sizeof(mContacts[i].bodyIndexA));
    hashBytes(topologyHash, &mContacts[i].bodyIndexB,
              sizeof(mContacts[i].bodyIndexB));
  }
  mHeader.topologyHash = topologyHash;
  return true;
}

void AvbdKernelLabCapture::capturePreRange(
    PxU32 ticket, const AvbdRigidSolveContext& context,
    const AvbdSolverConfig& config, PxU32 islandIndex, PxU32 colorIndex,
    const PxU32* ownerOrder, PxU32 begin, PxU32 end, PxU32 workerCount,
    PxU32 taskGrainBodies, PxU32 taskCount, PxU32 taskChunkBodies) {
  if (!config.enableLocal6x6Solve) {
    reject();
    return;
  }
  if (mState != eRESERVED || ticket != mTicket ||
      (mTargetIsland != PX_MAX_U32 && islandIndex != mTargetIsland) ||
      context.iteration.activeIteration != mTargetIteration ||
      (mTargetColor != PX_MAX_U32 && colorIndex != mTargetColor)) {
    return;
  }

  const AvbdRigidSolveIterationState& state = context.iteration;
  const AvbdBodyConstraintMap* map = state.contactMap;
  if (!ownerOrder || begin >= end || !state.bodies || !state.contacts || !map ||
      !map->constraintOffsets || !map->constraintCounts ||
      !map->constraintIndices || state.numBodies == 0 ||
      state.numBodies > mReservedBodyCount ||
      state.numContacts > mReservedContactCount ||
      map->numBodies != state.numBodies ||
      map->totalConstraintRefs > mReservedRefCount ||
      context.bodyColorCount > mReservedBodyCount ||
      context.bodyColorBodies.size() > mReservedBodyCount ||
      end - begin > mReservedBodyCount ||
      end > context.bodyColorBodies.size() ||
      context.bodyColorCount == 0 ||
      context.bodyColorOffsets.size() != context.bodyColorCount + 1u ||
      context.bodyColorOffsets[context.bodyColorCount] !=
          context.bodyColorBodies.size()) {
    reject();
    return;
  }

  // The trace/reference authority is intentionally broader than the compact
  // CPU candidate. A candidate tier is advertised only when every selected
  // row is either dynamic--dynamic or ordinary rigid body-static primal; all
  // deformable/kinematic/objective-owned and locked scopes remain
  // reference-only. This keeps capture selection from biasing the corpus.
  bool hasBodyStatic = false;
  bool candidateScopeSupported = true;
  bool selectedHasOrdinaryRigidStatic = false;
  for (PxU32 contactIndex = 0; contactIndex < state.numContacts;
       ++contactIndex) {
    const AvbdContactConstraint& contact = state.contacts[contactIndex];
    hasBodyStatic = hasBodyStatic ||
        contact.header.bodyIndexA >= state.numBodies ||
        contact.header.bodyIndexB >= state.numBodies;
  }
  for (PxU32 bodyIndex = 0; bodyIndex < state.numBodies; ++bodyIndex) {
    candidateScopeSupported = candidateScopeSupported &&
        state.bodies[bodyIndex].nodeIndex == bodyIndex &&
        state.bodies[bodyIndex].lockFlags == 0;
  }
  for (PxU32 ownerSlot = begin; ownerSlot < end; ++ownerSlot) {
    const PxU32 owner = ownerOrder[ownerSlot];
    if (owner >= state.numBodies || state.bodies[owner].invMass <= 0.0f) {
      reject();
      return;
    }
    const PxU32 offset = map->constraintOffsets[owner];
    const PxU32 count = map->constraintCounts[owner];
    if (offset > map->totalConstraintRefs ||
        count > map->totalConstraintRefs - offset) {
      reject();
      return;
    }
    for (PxU32 row = 0; row < count; ++row) {
      const PxU32 contactIndex = map->constraintIndices[offset + row];
      if (contactIndex >= state.numContacts) {
        reject();
        return;
      }
      const AvbdContactConstraint& contact = state.contacts[contactIndex];
      if (contact.header.bodyIndexA != owner &&
          contact.header.bodyIndexB != owner) {
        reject();
        return;
      }
      const bool dynamicA = contact.header.bodyIndexA < state.numBodies;
      const bool dynamicB = contact.header.bodyIndexB < state.numBodies;
      if (hasDeformableStaticAnchor(contact) ||
          hasKinematicShellAnchor(contact) || (!dynamicA && !dynamicB)) {
        candidateScopeSupported = false;
        continue;
      }
      if (dynamicA && dynamicB) {
        candidateScopeSupported = candidateScopeSupported &&
            state.bodies[contact.header.bodyIndexA].nodeIndex ==
                contact.header.bodyIndexA &&
            state.bodies[contact.header.bodyIndexB].nodeIndex ==
                contact.header.bodyIndexB &&
            state.bodies[contact.header.bodyIndexA].lockFlags == 0 &&
            state.bodies[contact.header.bodyIndexB].lockFlags == 0;
      } else {
        selectedHasOrdinaryRigidStatic = true;
        const PxU32 dynamicBody = dynamicA ? contact.header.bodyIndexA :
                                             contact.header.bodyIndexB;
        candidateScopeSupported = candidateScopeSupported &&
            state.bodies[dynamicBody].nodeIndex == dynamicBody &&
            state.bodies[dynamicBody].lockFlags == 0;
      }
    }
  }

  // Every resize is within storage reserved by update(), so this branch does
  // not allocate and only copies the stable parent-owned prepared state.
  if (!mBodies.resize(state.numBodies) || !mContacts.resize(state.numContacts) ||
      !mMapOffsets.resize(state.numBodies + 1u) ||
      !mMapCounts.resize(state.numBodies) ||
      !mMapIndices.resize(map->totalConstraintRefs) ||
      !mColorOffsets.resize(context.bodyColorCount + 1u) ||
      !mColorBodies.resize(context.bodyColorBodies.size()) ||
      !mOwnerOrder.resize(end - begin) ||
      !mPostOwnerPoses.resize(end - begin)) {
    reject();
    return;
  }
  for (PxU32 bodyIndex = 0; bodyIndex < state.numBodies; ++bodyIndex)
    storeBody(mBodies[bodyIndex], state.bodies[bodyIndex]);
  for (PxU32 contactIndex = 0; contactIndex < state.numContacts;
       ++contactIndex)
    storeContact(mContacts[contactIndex], state.contacts[contactIndex]);
  PxMemCopy(mMapOffsets.begin(), map->constraintOffsets,
            sizeof(PxU32) * (state.numBodies + 1u));
  PxMemCopy(mMapCounts.begin(), map->constraintCounts,
            sizeof(PxU32) * state.numBodies);
  if (map->totalConstraintRefs > 0)
    PxMemCopy(mMapIndices.begin(), map->constraintIndices,
              sizeof(PxU32) * map->totalConstraintRefs);
  PxMemCopy(mColorOffsets.begin(), context.bodyColorOffsets.begin(),
            sizeof(PxU32) * (context.bodyColorCount + 1u));
  if (!context.bodyColorBodies.empty())
    PxMemCopy(mColorBodies.begin(), context.bodyColorBodies.begin(),
              sizeof(PxU32) * context.bodyColorBodies.size());
  PxMemCopy(mOwnerOrder.begin(), ownerOrder + begin,
            sizeof(PxU32) * (end - begin));

  PxMemZero(&mHeader, sizeof(mHeader));
  static const PxU8 kMagic[8] = {'A', 'V', 'B', 'D', 'K', 'L', 'R', '1'};
  PxMemCopy(mHeader.magic, kMagic, sizeof(kMagic));
  mHeader.schema = eAVBD_KERNEL_LAB_TRACE_SCHEMA;
  mHeader.endianMarker = eAVBD_KERNEL_LAB_TRACE_ENDIAN_MARKER;
  mHeader.contractHash = eAVBD_KERNEL_LAB_TRACE_CONTRACT_HASH;
  mHeader.producerBuildHash = eAVBD_KERNEL_LAB_TRACE_PRODUCER_BUILD_HASH;
  mHeader.flags = eAVBD_KERNEL_LAB_TRACE_FAST_CPU_COLOR |
      eAVBD_KERNEL_LAB_TRACE_LOCAL_6X6 |
      eAVBD_KERNEL_LAB_TRACE_PRE_RANGE |
      eAVBD_KERNEL_LAB_TRACE_FLOAT_BITS;
  if (candidateScopeSupported) {
    if (selectedHasOrdinaryRigidStatic)
      mHeader.flags |=
          eAVBD_KERNEL_LAB_TRACE_ORDINARY_RIGID_STATIC_PRIMAL_ONLY;
    else
      mHeader.flags |= eAVBD_KERNEL_LAB_TRACE_DYNAMIC_DYNAMIC_ONLY;
  }
  if (hasBodyStatic)
    mHeader.flags |= eAVBD_KERNEL_LAB_TRACE_HAS_BODY_STATIC;
  mHeader.configFlags = config.enableLocal6x6Solve ?
      eAVBD_KERNEL_LAB_TRACE_CONFIG_LOCAL_6X6 : 0u;
  if (config.enableParallelization)
    mHeader.configFlags |= eAVBD_KERNEL_LAB_TRACE_CONFIG_PARALLEL;
  if (config.requiresOrderedBackend())
    mHeader.configFlags |= eAVBD_KERNEL_LAB_TRACE_CONFIG_ORDERED;
  mHeader.bodyRecordBytes = sizeof(AvbdKernelLabTraceBody);
  mHeader.contactRecordBytes = sizeof(AvbdKernelLabTraceContact);
  mHeader.bodyCount = state.numBodies;
  mHeader.contactCount = state.numContacts;
  mHeader.constraintRefCount = map->totalConstraintRefs;
  mHeader.colorCount = context.bodyColorCount;
  mHeader.colorBodyCount = context.bodyColorBodies.size();
  mHeader.ownerCount = end - begin;
  mHeader.islandIndex = islandIndex;
  mHeader.activeIteration = state.activeIteration;
  mHeader.selectedColor = colorIndex;
  mHeader.selectedBegin = begin;
  mHeader.selectedEnd = end;
  mHeader.workerCount = workerCount;
  mHeader.taskGrainBodies = taskGrainBodies;
  mHeader.taskCount = taskCount;
  mHeader.taskChunkBodies = taskChunkBodies;
  mHeader.effectiveIterationCount = state.iters;
#ifdef PX_PHYSICS_VERSION
  mHeader.physicsVersion = PX_PHYSICS_VERSION;
#endif
  mHeader.dt = state.dt;
  mHeader.invDt2 = context.invDt2;
  mHeader.avbdAlpha = config.avbdAlpha;
  mState = ePRE_RANGE_READY;
}

void AvbdKernelLabCapture::capturePostRange(
    PxU32 ticket, const AvbdRigidSolveContext& context, PxU32 colorIndex) {
  if (mState != ePRE_RANGE_READY || ticket != mTicket ||
      colorIndex != mHeader.selectedColor ||
      !context.iteration.bodies ||
      mOwnerOrder.size() != mPostOwnerPoses.size()) {
    return;
  }
  for (PxU32 ownerSlot = 0; ownerSlot < mOwnerOrder.size(); ++ownerSlot) {
    const PxU32 owner = mOwnerOrder[ownerSlot];
    if (owner >= context.iteration.numBodies) {
      reject();
      return;
    }
    storeOwnerPose(mPostOwnerPoses[ownerSlot],
                   context.iteration.bodies[owner]);
  }
  mHeader.flags |= eAVBD_KERNEL_LAB_TRACE_POST_RANGE_ORACLE;
  if (!finalizePayloadHashes()) {
    reject();
    return;
  }
  mState = eREADY_TO_WRITE;
}

bool AvbdKernelLabCapture::writeFile() {
  FILE* file = std::fopen(mOutputPath, "wb");
  if (!file)
    return false;
  const bool complete = writeObject(file, &mHeader, sizeof(mHeader)) &&
      writeArray(file, mBodies) && writeArray(file, mContacts) &&
      writeArray(file, mMapOffsets) && writeArray(file, mMapCounts) &&
      writeArray(file, mMapIndices) && writeArray(file, mColorOffsets) &&
      writeArray(file, mColorBodies) && writeArray(file, mOwnerOrder) &&
      writeArray(file, mPostOwnerPoses);
  const bool closed = std::fclose(file) == 0;
  return complete && closed;
}

void AvbdKernelLabCapture::flush() {
  if (mState != eREADY_TO_WRITE)
    return;
  mState = writeFile() ? eWRITTEN : eWRITE_FAILED;
}

} // namespace Dy
} // namespace physx
