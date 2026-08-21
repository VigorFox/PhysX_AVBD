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

#include "avbd/solver/post_al/DyAvbdPostAl.h"
#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/post_al/DyAvbdPostAlContactResponse.h"
#include "common/PxProfileZone.h"

#include <cstdio>
#include <cstdlib>

namespace physx {
namespace Dy {

namespace {

template <typename T>
PX_FORCE_INLINE physx::PxU64 avbdArrayCapacityBytes(
    const physx::PxArray<T> &array) {
  return physx::PxU64(array.capacity()) * sizeof(T);
}

} // namespace

physx::PxU64 AvbdPostAlWorkspace::capacityBytes() const {
  const AvbdTerminalOgcState &ogc = terminal.ogc;
  physx::PxU64 bytes = 0u;
#define AVBD_ACCUMULATE_CAPACITY(array_) \
  bytes += avbdArrayCapacityBytes(array_)
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.softPositionBefore);
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.rigidPositionBefore);
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.rigidRotationBefore);
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.rigidInvInertiaBefore);
  AVBD_ACCUMULATE_CAPACITY(
      poseWriteAdmission.scratch.componentParents);
  AVBD_ACCUMULATE_CAPACITY(
      poseWriteAdmission.scratch.participatingComponents);
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.scratch.contactAlphas);
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.scratch.componentAlphas);
  AVBD_ACCUMULATE_CAPACITY(poseWriteAdmission.scratch.particleAlphas);
  AVBD_ACCUMULATE_CAPACITY(pose.splitRigidDeepPoseRecovery);
  AVBD_ACCUMULATE_CAPACITY(pose.splitRigidFiniteMaterialPose);
  AVBD_ACCUMULATE_CAPACITY(pose.postBlockPos);
  AVBD_ACCUMULATE_CAPACITY(pose.postBlockRot);
  AVBD_ACCUMULATE_CAPACITY(pose.deformableNormalStageMask);
  AVBD_ACCUMULATE_CAPACITY(friction.postDepenPos);
  AVBD_ACCUMULATE_CAPACITY(friction.postDepenRot);
  AVBD_ACCUMULATE_CAPACITY(
      friction.responseWorkspace.dominantDeformable);
  AVBD_ACCUMULATE_CAPACITY(
      friction.responseWorkspace.bodyDeformRawCount);
  AVBD_ACCUMULATE_CAPACITY(friction.responseWorkspace.contactIndices);
  AVBD_ACCUMULATE_CAPACITY(friction.responseWorkspace.bodyContactCount);
  AVBD_ACCUMULATE_CAPACITY(
      friction.responseWorkspace.bodyContactNormalSum);
  AVBD_ACCUMULATE_CAPACITY(friction.responseWorkspace.linearVelocity);
  AVBD_ACCUMULATE_CAPACITY(friction.responseWorkspace.angularVelocity);
  AVBD_ACCUMULATE_CAPACITY(
      friction.responseWorkspace.initialLinearVelocity);
  AVBD_ACCUMULATE_CAPACITY(
      friction.responseWorkspace.initialAngularVelocity);
  AVBD_ACCUMULATE_CAPACITY(friction.responseWorkspace.touched);
  AVBD_ACCUMULATE_CAPACITY(friction.responseWorkspace.bodySpeed);
  AVBD_ACCUMULATE_CAPACITY(ogc.proxyParticles);
  AVBD_ACCUMULATE_CAPACITY(ogc.collisionBodies);
  AVBD_ACCUMULATE_CAPACITY(ogc.rigidBoxes);
  AVBD_ACCUMULATE_CAPACITY(ogc.rigidSpheres);
  AVBD_ACCUMULATE_CAPACITY(ogc.rigidCapsules);
  AVBD_ACCUMULATE_CAPACITY(ogc.rigidConvexes);
  AVBD_ACCUMULATE_CAPACITY(ogc.rigidTriangleSurfaces);
  AVBD_ACCUMULATE_CAPACITY(ogc.contacts);
  AVBD_ACCUMULATE_CAPACITY(ogc.geometrySidecar.triangleCoreCertificates);
  AVBD_ACCUMULATE_CAPACITY(ogc.geometrySidecar.contactTriangleCoreIndices);
  AVBD_ACCUMULATE_CAPACITY(ogc.pairStates);
  AVBD_ACCUMULATE_CAPACITY(ogc.detectedPairScratch);
  AVBD_ACCUMULATE_CAPACITY(ogc.detectedPairIndexScratch);
  AVBD_ACCUMULATE_CAPACITY(ogc.detectedPairToRegistryScratch);
  AVBD_ACCUMULATE_CAPACITY(ogc.pairIndices);
  AVBD_ACCUMULATE_CAPACITY(ogc.velocityBasePos);
  AVBD_ACCUMULATE_CAPACITY(ogc.velocityBaseRot);
  AVBD_ACCUMULATE_CAPACITY(ogc.velocityBaseLinear);
  AVBD_ACCUMULATE_CAPACITY(ogc.velocityBaseAngular);
  AVBD_ACCUMULATE_CAPACITY(ogc.acceptedSoftPositions);
  AVBD_ACCUMULATE_CAPACITY(ogc.acceptedSoftVelocities);
  AVBD_ACCUMULATE_CAPACITY(ogc.sourceBodyMask);
  AVBD_ACCUMULATE_CAPACITY(ogc.failClosedSoftBodyMask);
  AVBD_ACCUMULATE_CAPACITY(ogc.failClosedRigidBodyMask);
  AVBD_ACCUMULATE_CAPACITY(ogc.failClosedPairMask);
  AVBD_ACCUMULATE_CAPACITY(ogc.overlapPairMask);
  AVBD_ACCUMULATE_CAPACITY(ogc.rollbackParents);
  AVBD_ACCUMULATE_CAPACITY(ogc.rollbackFailedRoots);
  AVBD_ACCUMULATE_CAPACITY(ogc.broadphaseBodyMinimum);
  AVBD_ACCUMULATE_CAPACITY(ogc.broadphaseBodyMaximum);
  AVBD_ACCUMULATE_CAPACITY(velocity.physicalContactTangentOwnerIndex);
  AVBD_ACCUMULATE_CAPACITY(velocity.fastNormalImpactByBody);
#undef AVBD_ACCUMULATE_CAPACITY
  return bytes;
}

void AvbdPostAlWorkspace::beginSolve() {
  capacityAtSolveStart = capacityBytes();
  contactWorkspaceGrowthBytesAtSolveStart =
      terminal.ogc.contactWorkspace.growthBytes;

  // The workspace survives the frame, but the terminal epoch does not.
  // Invalidate only logical payloads; PxArray::clear() preserves capacity.
  AvbdTerminalOgcState &ogc = terminal.ogc;
  ogc.proxyParticles.clear();
  ogc.collisionBodies.clear();
  ogc.rigidBoxes.clear();
  ogc.rigidSpheres.clear();
  ogc.rigidCapsules.clear();
  ogc.rigidConvexes.clear();
  ogc.rigidTriangleSurfaces.clear();
  ogc.contacts.clear();
  ogc.geometrySidecar.clear();
  ogc.pairStates.clear();
  ogc.detectedPairScratch.clear();
  ogc.detectedPairIndexScratch.clear();
  ogc.detectedPairToRegistryScratch.clear();
  ogc.pairIndices.clear();
  ogc.sourceBodyMask.clear();
  ogc.failClosedSoftBodyMask.clear();
  ogc.failClosedRigidBodyMask.clear();
  ogc.failClosedPairMask.clear();
  ogc.overlapPairMask.clear();
  ogc.rollbackParents.clear();
  ogc.rollbackFailedRoots.clear();
  ogc.broadphaseBodyMinimum.clear();
  ogc.broadphaseBodyMaximum.clear();
  ogc.selectionPairCount = 0u;
  ogc.pairRegistryActive = false;
  ogc.currentPoseEpochApplied = false;
  ogc.closureUnresolved = false;
  ogc.failClosed = false;
  ogc.stalled = false;
  ogc.closureStatus = AvbdTerminalOgcClosureStatus::eNOT_RUN;
}

void AvbdPostAlWorkspace::endSolve() {
  const physx::PxU64 after = capacityBytes();
  const physx::PxU64 contactGrowth =
      terminal.ogc.contactWorkspace.growthBytes -
      contactWorkspaceGrowthBytesAtSolveStart;
  const physx::PxU64 directGrowth =
      after > capacityAtSolveStart ? after - capacityAtSolveStart : 0u;
  if (directGrowth != 0u || contactGrowth != 0u) {
    ++growthEvents;
    growthBytes += directGrowth + contactGrowth;
  }
}

void AvbdSolver::postAlStages(
    physx::PxReal dt, physx::PxReal invDt, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
    const physx::PxVec3 &gravity,
    bool hasBodyStaticContact, bool deformableFastImpactIsland,
    const physx::PxArray<bool> &touchingBodyStatic,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    bool allowRigidDeepPoseRecoverySplit,
    bool allowRigidFiniteMaterialPoseSplit,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    const physx::PxArray<bool> &touchesKinematicShell,
    const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
    const physx::PxArray<bool> *positionOwnedAngularBodies,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    bool hasJointConstraints, bool skipBodyStaticFriction,
    bool applyVelocityDamping,
    AvbdSoftParticle *softParticlesForVel,
    physx::PxU32 numSoftParticlesForVel,
    AvbdSolverStats &stats,
    const AvbdPostAlContactWorkPlan *postAlContactWork,
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan) {
  PX_PROFILE_ZONE("AVBD.postAlStages", 0);

  AvbdPostAlWorkspace localWorkspace;
  AvbdPostAlWorkspace &workspace =
      terminalSoftExecutionPlan && terminalSoftExecutionPlan->postAlWorkspace
          ? *terminalSoftExecutionPlan->postAlWorkspace
          : localWorkspace;
  workspace.beginSolve();
  const physx::PxU64 workspaceGrowthEventsBefore = workspace.growthEvents;
  const physx::PxU64 workspaceGrowthBytesBefore = workspace.growthBytes;

  const bool hasKinematicShellContacts =
      shellContacts && numShellContacts > 0 && shellParticles &&
      numShellParticles > 0;
  AvbdOgcPairTrustRegionContext postAlOgcContext;
  AvbdOgcPairTrustRegionContext *postAlOgcContextPtr =
      initializeOgcPairTrustRegionContextView(
          terminalSoftExecutionPlan, numShellContacts, postAlOgcContext)
          ? &postAlOgcContext
          : nullptr;
  AvbdOgcPoseWritePhaseState &poseWriteAdmission =
      workspace.poseWriteAdmission;
  poseWriteAdmission.capture(
      postAlOgcContextPtr, shellContacts, numShellContacts, shellParticles,
      numShellParticles, bodies, numBodies);
  AvbdPostAlPoseState &poseState = workspace.pose;
  poseState.run(
      *this, bodies, numBodies, contacts, numContacts, contactMap, gravity, dt,
      hasBodyStaticContact, deformableFastImpactIsland,
      allowRigidDeepPoseRecoverySplit, allowRigidFiniteMaterialPoseSplit,
      shellParticles, numShellParticles, shellContacts, numShellContacts,
      touchesKinematicShell, stats);
  admitOgcPoseWritePhase(
      poseWriteAdmission, postAlOgcContextPtr, bodies, numBodies,
      shellParticles, numShellParticles, softBodiesForRecovery,
      numSoftBodiesForRecovery, shellContacts, numShellContacts,
      terminalSoftExecutionPlan);
  const physx::PxArray<bool> &splitRigidDeepPoseRecovery =
      poseState.splitRigidDeepPoseRecovery;
  const physx::PxArray<bool> &splitRigidFiniteMaterialPose =
      poseState.splitRigidFiniteMaterialPose;
  const physx::PxArray<physx::PxVec3> &postBlockPos = poseState.postBlockPos;
  const physx::PxArray<physx::PxQuat> &postBlockRot = poseState.postBlockRot;
  physx::PxArray<physx::PxU8> *deformableNormalStageMaskPtr =
      poseState.normalStageMaskPtr();

  AvbdPostAlRecoveryState &recoveryState = workspace.recovery;
  recoveryState.run(
      *this, bodies, numBodies, shellParticles, numShellParticles,
      softBodiesForRecovery, numSoftBodiesForRecovery, shellContacts,
      numShellContacts, stats, terminalSoftExecutionPlan);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  poseWriteAdmission.capture(
      postAlOgcContextPtr, shellContacts, numShellContacts, shellParticles,
      numShellParticles, bodies, numBodies);
  AvbdPostAlFrictionState &frictionState = workspace.friction;
  frictionState.run(
      bodies, numBodies, contacts, numContacts, gravity, dt,
      skipBodyStaticFriction, hasKinematicShellContacts,
      touchesKinematicShell, mConfig.lengthScale, stats);
  admitOgcPoseWritePhase(
      poseWriteAdmission, postAlOgcContextPtr, bodies, numBodies,
      shellParticles, numShellParticles, softBodiesForRecovery,
      numSoftBodiesForRecovery, shellContacts, numShellContacts,
      terminalSoftExecutionPlan);
  physx::PxArray<physx::PxVec3> &postDepenPos = frictionState.postDepenPos;
  physx::PxArray<physx::PxQuat> &postDepenRot = frictionState.postDepenRot;

  AvbdPostAlTerminalState &terminalState = workspace.terminal;
  terminalState.run(
      terminalSoftExecutionPlan, bodies, numBodies, shellParticles,
      numShellParticles, softBodiesForRecovery, numSoftBodiesForRecovery,
      shellContacts, numShellContacts, mConfig.lengthScale, stats);
  AvbdTerminalOgcState &terminalOgcState = terminalState.ogc;

  if (std::getenv("PHYSX_AVBD_OGC_TERMINAL_TRACE") &&
      (terminalOgcState.currentPoseEpochApplied ||
       terminalOgcState.closureUnresolved)) {
    std::printf(
        "[AVBD_OGC_TERMINAL] applied=%u status=%u detectionEpochs=%u "
        "projectionPasses=%u committed=%u lastContacts=%u overlaps=%u "
        "maxPenetration=%.9g stalled=%u failClosed=%u unresolved=%u\n",
        terminalOgcState.currentPoseEpochApplied ? 1u : 0u,
        physx::PxU32(terminalOgcState.closureStatus),
        terminalOgcState.detectionEpochs,
        terminalOgcState.projectionPasses,
        terminalOgcState.committedCorrections,
        terminalOgcState.lastContactCount,
        terminalOgcState.lastOverlapCount,
        double(terminalOgcState.maximumPenetration),
        terminalOgcState.stalled ? 1u : 0u,
        terminalOgcState.failClosed ? 1u : 0u,
        terminalOgcState.closureUnresolved ? 1u : 0u);
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  AvbdPostAlVelocityState &velocityState = workspace.velocity;
  velocityState.build(numBodies, contacts, numContacts, touchingBodyStatic,
                      linearVelAtSolveStart, deformableFastImpactIsland);
  {
    reconstructPostAlBodyVelocities(
        dt, invDt, bodies, numBodies, contacts, numContacts, gravity,
        hasKinematicShellContacts, touchesKinematicShell,
        shellLinearVelAtSolveStart, shellContacts, numShellContacts,
        positionOwnedAngularBodies, d6Joints, numD6, applyVelocityDamping,
        splitRigidDeepPoseRecovery, splitRigidFiniteMaterialPose, postBlockPos,
        postBlockRot, postDepenPos, postDepenRot, velocityState,
        terminalOgcState.currentPoseEpochApplied,
        terminalOgcState.velocityBasePos, terminalOgcState.velocityBaseRot,
        linearVelAtSolveStart, angularVelAtSolveStart,
        mConfig.velocityDamping, mConfig.angularDamping);
    if (terminalOgcState.failClosed &&
        terminalOgcState.velocityBaseLinear.size() == numBodies &&
        terminalOgcState.velocityBaseAngular.size() == numBodies) {
      for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
        if (bodyIndex >= terminalOgcState.failClosedRigidBodyMask.size() ||
            terminalOgcState.failClosedRigidBodyMask[bodyIndex] == 0u)
          continue;
        const physx::PxVec3 linear =
            terminalOgcState.velocityBaseLinear[bodyIndex];
        const physx::PxVec3 angular =
            terminalOgcState.velocityBaseAngular[bodyIndex];
        if (!linear.isFinite() || !angular.isFinite())
          continue;
        bodies[bodyIndex].linearVelocity = linear;
        bodies[bodyIndex].prevLinearVelocity = linear;
        bodies[bodyIndex].angularVelocity = angular;
        bodies[bodyIndex].projectLockedVelocities();
      }
    }
    if (d6Joints && numD6 > 0) {
      PX_PROFILE_ZONE("AVBD.projectBodyStaticLockedD6LinearVelocity", 0);
      projectBodyStaticLockedD6LinearVelocities(bodies, numBodies, d6Joints,
                                                numD6);
    }
    if (contacts && numContacts > 0) {
      PX_PROFILE_ZONE("AVBD.materialNormalVelocity", 0);
      clampBodyStaticInelasticNormalVelocities(
          bodies, numBodies, contacts, numContacts, contactMap,
          linearVelAtSolveStart, angularVelAtSolveStart,
          &splitRigidFiniteMaterialPose, dt,
          mConfig.bounceApproachSpeedThreshold(), mConfig.lengthScale,
          hasJointConstraints,
          mConfig.enableBoundedComponentProductionProbe,
          deformableNormalStageMaskPtr, &stats);
      PX_PROFILE_ZONE("AVBD.contactTargetVelocity", 0);
      applyAvbdContactTargetVelocity(bodies, numBodies, contacts,
                                     numContacts, dt, postAlContactWork);
    }
    if (hasKinematicShellContacts) {
      PX_PROFILE_ZONE("AVBD.kinematicShellInelasticVel", 0);
      clampKinematicOgcInelasticNormalVelocities(
          bodies, numBodies, shellParticles, numShellParticles, shellContacts,
          numShellContacts, dt, &stats);
    }
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f)
        bodies[i].projectLockedVelocities();
    }

    AvbdOgcPairState *activePairStates =
        terminalOgcState.pairRegistryActive
            ? (terminalOgcState.pairStates.empty()
                   ? nullptr
                   : terminalOgcState.pairStates.begin())
            : (terminalSoftExecutionPlan
                   ? terminalSoftExecutionPlan->ogcPairStates
                   : nullptr);
    const physx::PxU32 numActivePairStates =
        terminalOgcState.pairRegistryActive
            ? terminalOgcState.pairStates.size()
            : (terminalSoftExecutionPlan
                   ? terminalSoftExecutionPlan->numOgcPairStates
                   : 0u);
    finalizePostAlSoftVelocities(
        softParticlesForVel, numSoftParticlesForVel, invDt, bodies, numBodies,
        shellParticles, numShellParticles, softBodiesForRecovery,
        numSoftBodiesForRecovery, shellContacts, numShellContacts, dt,
		terminalSoftExecutionPlan,
        activePairStates, numActivePairStates,
        terminalOgcState.contacts.empty() ? nullptr
                                          : terminalOgcState.contacts.begin(),
        terminalOgcState.contacts.size(),
        terminalOgcState.currentPoseEpochApplied,
        &terminalOgcState,
        activePairStates, numActivePairStates,
        mConfig.lengthScale, stats);

    // The terminal registry was the sole mutable owner from fresh DCD through
    // velocity handoff. Publish the selection prefix back only after every
    // terminal correction has been consumed; extension pairs have no stale
    // state to carry because the next Scene contact epoch recompiles them.
    if (terminalOgcState.pairRegistryActive && terminalSoftExecutionPlan &&
        terminalSoftExecutionPlan->ogcPairStates &&
        terminalOgcState.selectionPairCount ==
            terminalSoftExecutionPlan->numOgcPairStates &&
        terminalOgcState.selectionPairCount <=
            terminalOgcState.pairStates.size()) {
      for (physx::PxU32 pairIndex = 0;
           pairIndex < terminalOgcState.selectionPairCount; ++pairIndex)
        terminalSoftExecutionPlan->ogcPairStates[pairIndex] =
            terminalOgcState.pairStates[pairIndex];
    }
  }
  workspace.endSolve();
  if (std::getenv("PHYSX_AVBD_OGC_WORKSPACE_TRACE")) {
    std::printf(
        "[AVBD_OGC_WORKSPACE] persistent=%u capacityBytes=%llu "
        "solveGrowthEvents=%llu solveGrowthBytes=%llu "
        "totalGrowthEvents=%llu totalGrowthBytes=%llu\n",
        &workspace == &localWorkspace ? 0u : 1u,
        static_cast<unsigned long long>(workspace.capacityBytes()),
        static_cast<unsigned long long>(
            workspace.growthEvents - workspaceGrowthEventsBefore),
        static_cast<unsigned long long>(
            workspace.growthBytes - workspaceGrowthBytesBefore),
        static_cast<unsigned long long>(workspace.growthEvents),
        static_cast<unsigned long long>(workspace.growthBytes));
  }
}

} // namespace Dy
} // namespace physx
