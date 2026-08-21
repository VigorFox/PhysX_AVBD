// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_TERMINAL_STATE_H
#define DY_AVBD_OGC_TERMINAL_STATE_H

#include "avbd/contact/DyAvbdContactRigidPrimitives.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx {
namespace Dy {

enum class AvbdTerminalOgcClosureStatus : physx::PxU8 {
  eNOT_RUN,
  eCONVERGED,
  eROLLED_BACK,
  eUNRESOLVED
};

enum class AvbdTerminalOgcProgressAction : physx::PxU8 {
  eCONVERGED,
  ePROJECT,
  eFAIL_CLOSED
};

// Pure convergence policy for one same-time terminal nonlinear epoch. The
// pass budget is only a safety ceiling; zero overlaps is the sole successful
// exit, while an attempted projection with no committed transaction is an
// immediate, explicit stall.
PX_FORCE_INLINE AvbdTerminalOgcProgressAction selectTerminalOgcProgressAction(
    physx::PxU32 overlapCount, bool previousProjectionAttempted,
    physx::PxU32 previousCommittedCorrections,
    physx::PxU32 projectionPasses, physx::PxU32 maximumProjectionPasses) {
  if (overlapCount == 0u)
    return AvbdTerminalOgcProgressAction::eCONVERGED;
  if ((previousProjectionAttempted &&
       previousCommittedCorrections == 0u) ||
      projectionPasses >= maximumProjectionPasses)
    return AvbdTerminalOgcProgressAction::eFAIL_CLOSED;
  return AvbdTerminalOgcProgressAction::ePROJECT;
}

// Frame-local owner for the terminal current-pose OGC epoch.
struct AvbdTerminalOgcState {
  physx::PxArray<AvbdSoftParticle> proxyParticles;
  physx::PxArray<AvbdSoftBody> collisionBodies;
  physx::PxArray<AvbdRigidBox> rigidBoxes;
  physx::PxArray<AvbdRigidSphere> rigidSpheres;
  physx::PxArray<AvbdRigidCapsule> rigidCapsules;
  physx::PxArray<AvbdRigidConvex> rigidConvexes;
  physx::PxArray<AvbdRigidTriangleSurface> rigidTriangleSurfaces;
  physx::PxArray<AvbdSoftContact> contacts;
  AvbdSoftContactWorkspace contactWorkspace;
  AvbdOgcGeometryEpochSidecar geometrySidecar;
  // One mutable OgcPairState registry owns both selection pairs and pairs
  // first discovered by the terminal t=dt detector. Selection state is copied
  // back only after terminal velocity handoff has consumed this registry.
  physx::PxArray<AvbdOgcPairState> pairStates;
  physx::PxArray<AvbdOgcPairState> detectedPairScratch;
  physx::PxArray<physx::PxU32> detectedPairIndexScratch;
  physx::PxArray<physx::PxU32> detectedPairToRegistryScratch;
  physx::PxArray<physx::PxU32> pairIndices;
  physx::PxArray<physx::PxVec3> velocityBasePos;
  physx::PxArray<physx::PxQuat> velocityBaseRot;
  physx::PxArray<physx::PxVec3> velocityBaseLinear;
  physx::PxArray<physx::PxVec3> velocityBaseAngular;
  // Previous accepted soft positions captured before any terminal projector
  // advances its velocity anchor. Fail-closed rollback must use this snapshot
  // rather than initialPosition, which geometric correction moves by design.
  physx::PxArray<physx::PxVec3> acceptedSoftPositions;
  physx::PxArray<physx::PxVec3> acceptedSoftVelocities;
  physx::PxArray<physx::PxU8> sourceBodyMask;
  // A stalled terminal nonlinear solve is fail-closed at the previous
  // accepted pose. These masks suppress velocity reconstruction for every
  // endpoint in the rolled-back OGC connected component.
  physx::PxArray<physx::PxU8> failClosedSoftBodyMask;
  physx::PxArray<physx::PxU8> failClosedRigidBodyMask;
  physx::PxArray<physx::PxU8> failClosedPairMask;
  // Reused nonlinear verification and fail-closed component scratch.
  physx::PxArray<physx::PxU8> overlapPairMask;
  physx::PxArray<physx::PxU32> rollbackParents;
  physx::PxArray<physx::PxU8> rollbackFailedRoots;
  physx::PxArray<physx::PxVec3> broadphaseBodyMinimum;
  physx::PxArray<physx::PxVec3> broadphaseBodyMaximum;
  physx::PxU32 selectionPairCount = 0u;
  bool pairRegistryActive = false;
  bool currentPoseEpochApplied = false;
  bool closureUnresolved = false;
  bool failClosed = false;
  bool stalled = false;
  AvbdTerminalOgcClosureStatus closureStatus =
      AvbdTerminalOgcClosureStatus::eNOT_RUN;
  physx::PxU32 detectionEpochs = 0u;
  physx::PxU32 projectionPasses = 0u;
  physx::PxU32 committedCorrections = 0u;
  physx::PxU32 lastOverlapCount = 0u;
  physx::PxU32 lastContactCount = 0u;
  physx::PxReal maximumPenetration = 0.0f;
};

} // namespace Dy
} // namespace physx

#endif
