// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Pointer-free wire contract shared by the opt-in AVBD CPU capture and the
// avbd_kernel_lab replay executable.  This is deliberately a local-6x6
// input record, not a serialization of the full Scene or a solver ABI.

#ifndef DY_AVBD_KERNEL_LAB_TRACE_CONTRACT_H
#define DY_AVBD_KERNEL_LAB_TRACE_CONTRACT_H

#include "foundation/PxSimpleTypes.h"

namespace physx {
namespace Dy {

// Increment this and update the contract hash whenever a record field changes.
static const PxU32 eAVBD_KERNEL_LAB_TRACE_SCHEMA = 2u;
static const PxU32 eAVBD_KERNEL_LAB_TRACE_ENDIAN_MARKER = 0x01020304u;
static const PxU64 eAVBD_KERNEL_LAB_TRACE_CONTRACT_HASH =
    0x8640e3fd2c91b70dull;
static const PxU64 eAVBD_KERNEL_LAB_TRACE_PRODUCER_BUILD_HASH =
    0x5d72d4afc91b3e62ull;

enum AvbdKernelLabTraceFlags : PxU32 {
  eAVBD_KERNEL_LAB_TRACE_GENERATED = 1u << 0,
  eAVBD_KERNEL_LAB_TRACE_FAST_CPU_COLOR = 1u << 1,
  eAVBD_KERNEL_LAB_TRACE_LOCAL_6X6 = 1u << 2,
  eAVBD_KERNEL_LAB_TRACE_HAS_BODY_STATIC = 1u << 3,
  eAVBD_KERNEL_LAB_TRACE_DYNAMIC_DYNAMIC_ONLY = 1u << 4,
  eAVBD_KERNEL_LAB_TRACE_PRE_RANGE = 1u << 5,
  eAVBD_KERNEL_LAB_TRACE_POST_RANGE_ORACLE = 1u << 6,
  eAVBD_KERNEL_LAB_TRACE_FLOAT_BITS = 1u << 7,
  // The range contains ordinary body-static rows. This tier replaces only
  // the local normal primal; dual/material/finalize ownership remains the
  // production scalar path.
  eAVBD_KERNEL_LAB_TRACE_ORDINARY_RIGID_STATIC_PRIMAL_ONLY = 1u << 8
};

enum AvbdKernelLabTraceConfigFlags : PxU32 {
  eAVBD_KERNEL_LAB_TRACE_CONFIG_LOCAL_6X6 = 1u << 0,
  eAVBD_KERNEL_LAB_TRACE_CONFIG_PARALLEL = 1u << 1,
  eAVBD_KERNEL_LAB_TRACE_CONFIG_ORDERED = 1u << 2
};

// The payload is written in this exact order:
//   header, body records, contact records, CSR offsets, CSR counts,
//   CSR indices, full color offsets, full color bodies, selected owner order.
// All values are native little-endian IEEE-754 binary32.  Readers reject a
// different marker, schema, contract hash, or record size instead of guessing.
struct AvbdKernelLabTraceHeader {
  PxU8 magic[8];
  PxU32 schema;
  PxU32 endianMarker;
  PxU64 contractHash;
  PxU64 producerBuildHash;
  PxU64 payloadHash;
  PxU64 payloadBytes;
  PxU32 flags;
  PxU32 configFlags;
  PxU32 bodyRecordBytes;
  PxU32 contactRecordBytes;
  PxU32 bodyCount;
  PxU32 contactCount;
  PxU32 constraintRefCount;
  PxU32 colorCount;
  PxU32 colorBodyCount;
  PxU32 ownerCount;
  PxU32 islandIndex;
  PxU32 activeIteration;
  PxU32 selectedColor;
  PxU32 selectedBegin;
  PxU32 selectedEnd;
  PxU32 workerCount;
  PxU32 taskGrainBodies;
  PxU32 taskCount;
  PxU32 taskChunkBodies;
  PxU32 effectiveIterationCount;
  PxU32 physicsVersion;
  PxU64 topologyHash;
  PxF32 dt;
  PxF32 invDt2;
  PxF32 avbdAlpha;
  PxU32 reserved;
};

// Fields used by AvbdSolver::solveLocalSystem and solveRigidBodyRange.  This
// intentionally excludes body padding and unrelated writeback/velocity state.
struct AvbdKernelLabTraceBody {
  PxF32 position[3];
  PxF32 invMass;
  PxF32 rotation[4];
  PxF32 inertialPosition[3];
  PxF32 inertialRotation[4];
  PxF32 prevPosition[3];
  PxF32 prevRotation[4];
  PxF32 invInertiaWorld[9];
  PxU32 nodeIndex;
  PxU32 lockFlags;
};

// Fields read by the contact-only local 6x6 primal.  In particular this omits
// contact-report pointers and the post-AL objective program. A body-static
// selected scope may be retained for production-reference diagnosis, but the
// initial compact CPU candidate must fail closed unless the trace advertises
// DYNAMIC_DYNAMIC_ONLY. Soft/deformable scopes never pass fast CPU admission.
struct AvbdKernelLabTraceContact {
  PxU32 bodyIndexA;
  PxU32 bodyIndexB;
  PxU16 type;
  PxU16 flags;
  PxF32 compliance;
  PxF32 damping;
  PxF32 lambda;
  PxF32 rho;
  PxF32 penalty;
  PxU16 colorGroup;
  PxU16 reserved0;
  PxF32 contactPointA[3];
  PxF32 penetrationDepth;
  PxF32 contactPointB[3];
  PxF32 restitution;
  PxF32 contactNormal[3];
  PxF32 friction;
  PxF32 targetVelocity[3];
  PxF32 maxImpulse;
  PxF32 invMassScaleA;
  PxF32 invMassScaleB;
  PxF32 invInertiaScaleA;
  PxF32 invInertiaScaleB;
  PxF32 tangent0[3];
  PxF32 tangentLambda0;
  PxF32 tangent1[3];
  PxF32 tangentLambda1;
  PxF32 staticFriction;
  PxF32 tangentPenalty0;
  PxF32 tangentPenalty1;
  PxF32 C0;
};

PX_COMPILE_TIME_ASSERT(sizeof(AvbdKernelLabTraceBody) == 132);
PX_COMPILE_TIME_ASSERT(sizeof(AvbdKernelLabTraceContact) == 164);

// The production color range's exact post-solve owner state.  Lab reference
// and candidate replays must both reproduce these raw bits before reporting a
// timing value for a captured trace.
struct AvbdKernelLabTraceOwnerPose {
  PxF32 position[3];
  PxF32 rotation[4];
};

PX_COMPILE_TIME_ASSERT(sizeof(AvbdKernelLabTraceOwnerPose) == 28);

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_KERNEL_LAB_TRACE_CONTRACT_H
