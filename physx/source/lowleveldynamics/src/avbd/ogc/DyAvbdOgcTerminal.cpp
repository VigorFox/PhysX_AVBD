// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/ogc/DyAvbdOgcTerminal.h"
#include "avbd/ogc/DyAvbdOgcTerminalState.h"
#include "avbd/ogc/DyAvbdOgcCurrentPose.h"
#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"
#include "avbd/ogc/DyAvbdOgcPairState.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/ogc/DyAvbdOgcStaticResponse.h"
#include "avbd/ogc/DyAvbdOgcTriangleCoreGeometry.h"
#include "common/PxProfileZone.h"
#include "avbd/solver/DyAvbdSolver.h"

#include <cstdio>
#include <cstdlib>

namespace physx {
namespace Dy {

namespace {

struct AvbdTerminalOverlapReport {
  const physx::PxArray<physx::PxU8> *pairMask = nullptr;
  physx::PxU32 overlapCount = 0u;
  physx::PxU32 invalidGeometryCount = 0u;
  physx::PxReal maximumPenetration = 0.0f;
};

static AvbdTerminalOverlapReport evaluateTerminalCurrentPoseOverlaps(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const physx::PxU32 *pairIndices, physx::PxU32 numPairIndices,
    physx::PxU32 numPairStates,
    const AvbdOgcGeometryEpochView &geometryEpoch,
    const AvbdSoftParticle *particles, physx::PxU32 numParticles,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxReal tolerance, physx::PxU32 pass,
    physx::PxArray<physx::PxU8> &pairMask) {
  AvbdTerminalOverlapReport report;
  pairMask.resize(numPairStates);
  for (physx::PxU32 pairIndex = 0u;
       pairIndex < numPairStates; ++pairIndex)
    pairMask[pairIndex] = 0u;
  report.pairMask = &pairMask;
  if (!contacts || !particles)
    return report;

  auto recordOverlap = [&](physx::PxU32 contactIndex,
                           const AvbdSoftContactGeometry &geometry,
                           physx::PxReal signedGap, bool core,
                           bool invalid) {
    ++report.overlapCount;
    if (invalid)
      ++report.invalidGeometryCount;
    if (physx::PxIsFinite(signedGap))
      report.maximumPenetration = physx::PxMax(
          report.maximumPenetration, physx::PxMax(0.0f, -signedGap));
    else
      report.maximumPenetration = PX_MAX_F32;
    if (pairIndices && contactIndex < numPairIndices) {
      const physx::PxU32 pairIndex = pairIndices[contactIndex];
      if (pairIndex < pairMask.size())
        pairMask[pairIndex] = 1u;
    }
    if (std::getenv("PHYSX_AVBD_OGC_TERMINAL_TRACE"))
      std::printf(
          "[AVBD_OGC_TERMINAL_OVERLAP] pass=%u contact=%u "
          "sourceBody=%u targetKind=%u target=%u primitive=%llu "
          "core=%u invalid=%u gap=%.9g\n",
          pass, contactIndex, geometry.queryBodyIndex,
          physx::PxU32(geometry.targetKind), geometry.targetIndex,
          static_cast<unsigned long long>(geometry.source.primitiveKey),
          core ? 1u : 0u, invalid ? 1u : 0u, double(signedGap));
  };

  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const AvbdSoftContactGeometry &geometry = contacts[contactIndex].geometry;
    // Deformable/deformable contact already has one coupled Position-AL
    // nonlinear owner. M3 terminal closure owns only the rigid geometry
    // boundary; treating a dense soft-pair manifold as a second hard
    // fail-closed owner overconstrains rolling and recreates contact freeze.
    if (geometry.hasDeformableSurfaceTarget())
      continue;
    const AvbdOgcRigidBoxGeometry *rigidBox =
        geometryEpoch.getRigidBox(contactIndex, numContacts);
    const AvbdOgcTriangleCoreCertificate *triangleCore =
        geometryEpoch.getTriangleCore(contactIndex, numContacts);
    if (triangleCore) {
      const AvbdSolverBody *coreBody = nullptr;
      if (geometry.hasRigidBodyTarget()) {
        if (geometry.targetIndex >= numBodies) {
          recordOverlap(contactIndex, geometry, -PX_MAX_F32,
                        /*core=*/true, /*invalid=*/true);
          continue;
        }
        coreBody = &bodies[geometry.targetIndex];
      }
      physx::PxReal coreGap = 0.0f;
      if (!getCurrentRigidBoxTriangleCoreFaceGap(
              geometry, coreBody, particles, numParticles, coreGap,
              PX_MAX_U32, physx::PxVec3(0.0f), rigidBox, triangleCore)) {
        recordOverlap(contactIndex, geometry, -PX_MAX_F32,
                      /*core=*/true, /*invalid=*/true);
        continue;
      }
      if (coreGap < -tolerance) {
        recordOverlap(contactIndex, geometry, coreGap,
                      /*core=*/true, /*invalid=*/false);
        continue;
      }
    }

    const physx::PxVec3 queryPoint =
        avbdGetSoftContactQueryPoint(geometry, particles);
    if (!queryPoint.isFinite()) {
      recordOverlap(contactIndex, geometry, -PX_MAX_F32,
                    /*core=*/false, /*invalid=*/true);
      continue;
    }
    const bool dynamic = geometry.hasRigidBodyTarget();
    const bool validTarget = dynamic
        ? geometry.targetIndex < numBodies
        : geometry.hasWorldStaticTarget();
    if (!validTarget) {
      recordOverlap(contactIndex, geometry, -PX_MAX_F32,
                    /*core=*/false, /*invalid=*/true);
      continue;
    }

    AvbdOgcCurrentPairGeometry currentGeometry;
    bool valid = false;
    const AvbdSolverBody *dynamicTarget =
        dynamic ? &bodies[geometry.targetIndex] : nullptr;
    valid = getCurrentOgcPairGeometry(
        geometry, dynamicTarget, queryPoint, currentGeometry, rigidBox);
    if (!valid) {
      recordOverlap(contactIndex, geometry, -PX_MAX_F32,
                    /*core=*/false, /*invalid=*/true);
      continue;
    }
    if (currentGeometry.signedGap < -tolerance)
      recordOverlap(contactIndex, geometry, currentGeometry.signedGap,
                    /*core=*/false, /*invalid=*/false);
  }
  return report;
}

static physx::PxU32 findTerminalComponentRoot(
    physx::PxArray<physx::PxU32> &parents, physx::PxU32 node) {
  physx::PxU32 root = node;
  while (parents[root] != root)
    root = parents[root];
  while (parents[node] != node) {
    const physx::PxU32 next = parents[node];
    parents[node] = root;
    node = next;
  }
  return root;
}

static void joinTerminalComponents(
    physx::PxArray<physx::PxU32> &parents, physx::PxU32 lhs,
    physx::PxU32 rhs) {
  const physx::PxU32 lhsRoot = findTerminalComponentRoot(parents, lhs);
  const physx::PxU32 rhsRoot = findTerminalComponentRoot(parents, rhs);
  if (lhsRoot != rhsRoot)
    parents[rhsRoot] = lhsRoot;
}

// An unresolved nonlinear contact epoch must never publish an overlapping
// pose. Roll back every endpoint in the affected OGC connected component to
// the previous accepted state, then let the caller perform a fresh DCD
// verification at the same t=dt. This is an exceptional fail-closed path,
// not a contact response or an extra time step.
static physx::PxU32 rollbackTerminalOgcComponents(
    AvbdTerminalOgcState &terminalState,
    const AvbdTerminalOverlapReport *overlaps, bool rollbackAllPairs,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *particles, physx::PxU32 numParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies) {
  const physx::PxU32 nodeCount = numSoftBodies + numBodies;
  if (!particles || !softBodies || nodeCount == 0u)
    return 0u;

  physx::PxArray<physx::PxU32> &parents = terminalState.rollbackParents;
  parents.resize(nodeCount);
  for (physx::PxU32 node = 0; node < nodeCount; ++node)
    parents[node] = node;

  for (physx::PxU32 pairIndex = 0;
       pairIndex < terminalState.pairStates.size(); ++pairIndex) {
    const AvbdOgcPairState &pair = terminalState.pairStates[pairIndex];
    if (!pair.geometry.active || pair.key.sourceBodyIndex >= numSoftBodies)
      continue;
    if (pair.key.targetKind ==
            AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
        pair.key.targetBodyIndex < numSoftBodies) {
      joinTerminalComponents(parents, pair.key.sourceBodyIndex,
                             pair.key.targetBodyIndex);
    } else if (pair.key.targetKind ==
                   AvbdSoftContactTargetKind::eRIGID_BODY &&
               pair.key.targetBodyIndex < numBodies) {
      joinTerminalComponents(parents, pair.key.sourceBodyIndex,
                             numSoftBodies + pair.key.targetBodyIndex);
    }
  }

  physx::PxArray<physx::PxU8> &failedRoots =
      terminalState.rollbackFailedRoots;
  failedRoots.resize(nodeCount);
  for (physx::PxU32 node = 0u; node < nodeCount; ++node)
    failedRoots[node] = 0u;
  bool markedPair = false;
  for (physx::PxU32 pairIndex = 0;
       pairIndex < terminalState.pairStates.size(); ++pairIndex) {
    const AvbdOgcPairState &pair = terminalState.pairStates[pairIndex];
    const bool selected = rollbackAllPairs ||
        (overlaps && overlaps->pairMask &&
         pairIndex < overlaps->pairMask->size() &&
         (*overlaps->pairMask)[pairIndex] != 0u);
    if (!selected || !pair.geometry.active ||
        pair.key.sourceBodyIndex >= numSoftBodies)
      continue;
    failedRoots[findTerminalComponentRoot(
        parents, pair.key.sourceBodyIndex)] = 1u;
    markedPair = true;
  }
  // A malformed fresh manifold may not map to a pair. Admission already
  // restricted sourceBodyMask to the terminal island; roll that island back
  // rather than accepting geometry which could not acquire an owner.
  if (!markedPair) {
    for (physx::PxU32 sourceBody = 0;
         sourceBody < numSoftBodies; ++sourceBody) {
      if (sourceBody < terminalState.sourceBodyMask.size() &&
          terminalState.sourceBodyMask[sourceBody] != 0u)
        failedRoots[findTerminalComponentRoot(parents, sourceBody)] = 1u;
    }
  }

  terminalState.failClosedSoftBodyMask.resize(
      numSoftBodies, physx::PxU8(0));
  terminalState.failClosedRigidBodyMask.resize(numBodies, physx::PxU8(0));
  terminalState.failClosedPairMask.resize(
      terminalState.pairStates.size(), physx::PxU8(0));
  physx::PxU32 rolledBackEndpoints = 0u;
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex) {
    const physx::PxU32 root = findTerminalComponentRoot(parents, bodyIndex);
    if (failedRoots[root] == 0u)
      continue;
    const AvbdSoftBody &body = softBodies[bodyIndex];
    const physx::PxU32 start = body.compiled.particleStart;
    const physx::PxU32 count = body.compiled.particleCount;
    if (start > numParticles || count > numParticles - start)
      continue;
    terminalState.failClosedSoftBodyMask[bodyIndex] = 1u;
    ++rolledBackEndpoints;
    for (physx::PxU32 local = 0; local < count; ++local) {
      AvbdSoftParticle &particle = particles[start + local];
      const physx::PxU32 particleIndex = start + local;
      if (particleIndex >= terminalState.acceptedSoftPositions.size() ||
          !terminalState.acceptedSoftPositions[particleIndex].isFinite())
        continue;
      const physx::PxVec3 acceptedPosition =
          terminalState.acceptedSoftPositions[particleIndex];
      particle.position = acceptedPosition;
      particle.initialPosition = acceptedPosition;
      particle.predictedPosition = acceptedPosition;
      particle.outerPosition = acceptedPosition;
    }
  }
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    const physx::PxU32 node = numSoftBodies + bodyIndex;
    const physx::PxU32 root = findTerminalComponentRoot(parents, node);
    if (failedRoots[root] == 0u)
      continue;
    AvbdSolverBody &body = bodies[bodyIndex];
    if (!body.prevPosition.isFinite() || !body.prevRotation.isFinite())
      continue;
    physx::PxQuat rotationDelta =
        body.prevRotation * body.rotation.getConjugate();
    if (rotationDelta.w < 0.0f)
      rotationDelta = -rotationDelta;
    if (rotationDelta.isFinite()) {
      rotationDelta.normalize();
      const physx::PxMat33 rotationMatrix(rotationDelta);
      body.invInertiaWorld = rotationMatrix * body.invInertiaWorld *
          rotationMatrix.getTranspose();
    }
    body.position = body.prevPosition;
    body.rotation = body.prevRotation;
    body.predictedPosition = body.prevPosition;
    body.predictedRotation = body.prevRotation;
    body.inertialPosition = body.prevPosition;
    body.inertialRotation = body.prevRotation;
    terminalState.failClosedRigidBodyMask[bodyIndex] = 1u;
    ++rolledBackEndpoints;
  }

  for (physx::PxU32 pairIndex = 0;
       pairIndex < terminalState.pairStates.size(); ++pairIndex) {
    AvbdOgcPairState &pair = terminalState.pairStates[pairIndex];
    if (pair.key.sourceBodyIndex >= numSoftBodies)
      continue;
    const physx::PxU32 root = findTerminalComponentRoot(
        parents, pair.key.sourceBodyIndex);
    if (failedRoots[root] == 0u)
      continue;
    terminalState.failClosedPairMask[pairIndex] = 1u;
    pair.solve = AvbdOgcPairSolveState();
    pair.trustRegion.refreshRequested = true;
  }
  return rolledBackEndpoints;
}

} // namespace

// Member-owned terminal current-pose OGC closure. The state remains
// frame-local and is never retained after postAlStages returns.
void runTerminalCurrentPoseClosure(
    AvbdTerminalOgcState &terminalState,
    bool terminalCurrentPoseRefreshNeeded,
    const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery, physx::PxReal lengthScale,
    AvbdSolverStats &stats) {
  terminalState.closureStatus = AvbdTerminalOgcClosureStatus::eNOT_RUN;
  terminalState.closureUnresolved = false;
  terminalState.failClosed = false;
  terminalState.stalled = false;
  terminalState.detectionEpochs = 0u;
  terminalState.projectionPasses = 0u;
  terminalState.committedCorrections = 0u;
  terminalState.lastOverlapCount = 0u;
  terminalState.lastContactCount = 0u;
  terminalState.maximumPenetration = 0.0f;
  terminalState.failClosedSoftBodyMask.clear();
  terminalState.failClosedRigidBodyMask.clear();
  terminalState.failClosedPairMask.clear();

  if (!terminalCurrentPoseRefreshNeeded || !shellParticles ||
      numShellParticles == 0u || !softBodiesForRecovery ||
      numSoftBodiesForRecovery == 0u || !terminalSoftExecutionPlan)
    return;

  terminalState.velocityBasePos.resize(numBodies);
  terminalState.velocityBaseRot.resize(numBodies);
  terminalState.velocityBaseLinear.resize(numBodies);
  terminalState.velocityBaseAngular.resize(numBodies);
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    terminalState.velocityBasePos[bodyIndex] = bodies[bodyIndex].position;
    terminalState.velocityBaseRot[bodyIndex] = bodies[bodyIndex].rotation;
    terminalState.velocityBaseLinear[bodyIndex] =
        bodies[bodyIndex].linearVelocity;
    terminalState.velocityBaseAngular[bodyIndex] =
        bodies[bodyIndex].angularVelocity;
  }
  terminalState.acceptedSoftPositions.resize(numShellParticles);
  terminalState.acceptedSoftVelocities.resize(numShellParticles);
  for (physx::PxU32 particleIndex = 0;
       particleIndex < numShellParticles; ++particleIndex)
    terminalState.acceptedSoftPositions[particleIndex] =
        shellParticles[particleIndex].initialPosition;
  for (physx::PxU32 particleIndex = 0;
       particleIndex < numShellParticles; ++particleIndex)
    terminalState.acceptedSoftVelocities[particleIndex] =
        shellParticles[particleIndex].velocity;

  terminalState.selectionPairCount =
      terminalSoftExecutionPlan->numOgcPairStates;
  terminalState.pairStates.resize(terminalState.selectionPairCount);
  for (physx::PxU32 pairIndex = 0;
       pairIndex < terminalState.selectionPairCount; ++pairIndex)
    terminalState.pairStates[pairIndex] =
        terminalSoftExecutionPlan->ogcPairStates[pairIndex];
  terminalState.pairRegistryActive = true;

  const physx::PxReal overlapTolerance = physx::PxMax(
      1.0e-5f, 1.0e-4f * physx::PxMax(lengthScale, 1.0e-6f));
  // This is a safety ceiling, not a prescribed iteration count. Every
  // committed nonlinear response is followed immediately by a fresh DCD
  // epoch, and convergence/stall decisions use that new geometry.
  static const physx::PxU32 kMaximumProjectionEpochs = 12u;
  bool rollbackApplied = false;
  bool previousProjectionAttempted = false;
  physx::PxU32 previousCommittedCorrections = 0u;

  for (;;) {
    ++terminalState.detectionEpochs;
    if (!avbdBuildTerminalCurrentPoseContacts(
            terminalSoftExecutionPlan, bodies, numBodies, shellParticles,
            numShellParticles, softBodiesForRecovery,
            numSoftBodiesForRecovery, terminalState.sourceBodyMask.begin(),
            terminalState.sourceBodyMask.size(), terminalState.proxyParticles,
            terminalState.collisionBodies, terminalState.rigidBoxes,
            terminalState.rigidSpheres, terminalState.rigidCapsules,
            terminalState.rigidConvexes,
            terminalState.rigidTriangleSurfaces, terminalState.contacts,
            terminalState.contactWorkspace, terminalState.geometrySidecar)) {
      if (!rollbackApplied && rollbackTerminalOgcComponents(
              terminalState, nullptr, /*rollbackAllPairs=*/true, bodies,
              numBodies, shellParticles, numShellParticles,
              softBodiesForRecovery, numSoftBodiesForRecovery) > 0u) {
        rollbackApplied = true;
        terminalState.failClosed = true;
        terminalState.currentPoseEpochApplied = true;
        continue;
      }
      terminalState.closureUnresolved = true;
      terminalState.closureStatus =
          AvbdTerminalOgcClosureStatus::eUNRESOLVED;
      break;
    }
    terminalState.lastContactCount = terminalState.contacts.size();
    if (terminalState.contacts.empty()) {
      terminalState.closureStatus = rollbackApplied
          ? AvbdTerminalOgcClosureStatus::eROLLED_BACK
          : AvbdTerminalOgcClosureStatus::eCONVERGED;
      break;
    }

    if (!refreshCurrentOgcPairRegistry(
            terminalState.contacts.begin(), terminalState.contacts.size(),
            terminalState.rigidBoxes.begin(), terminalState.rigidBoxes.size(),
            shellParticles, numShellParticles, bodies, numBodies,
            numSoftBodiesForRecovery, terminalState.pairStates,
            terminalState.detectedPairScratch,
            terminalState.detectedPairIndexScratch,
            terminalState.detectedPairToRegistryScratch,
            terminalState.pairIndices)) {
      if (!rollbackApplied && rollbackTerminalOgcComponents(
              terminalState, nullptr, /*rollbackAllPairs=*/true, bodies,
              numBodies, shellParticles, numShellParticles,
              softBodiesForRecovery, numSoftBodiesForRecovery) > 0u) {
        rollbackApplied = true;
        terminalState.failClosed = true;
        terminalState.currentPoseEpochApplied = true;
        continue;
      }
      terminalState.closureUnresolved = true;
      terminalState.closureStatus =
          AvbdTerminalOgcClosureStatus::eUNRESOLVED;
      break;
    }

    const AvbdOgcGeometryEpochView geometryEpoch = makeOgcGeometryEpochView(
        terminalState.pairStates.empty() ? nullptr
                                         : terminalState.pairStates.begin(),
        terminalState.pairStates.size(),
        terminalState.pairIndices.empty() ? nullptr
                                          : terminalState.pairIndices.begin(),
        terminalState.pairIndices.size(), &terminalState.geometrySidecar);
    const AvbdTerminalOverlapReport overlap =
        evaluateTerminalCurrentPoseOverlaps(
            terminalState.contacts.begin(), terminalState.contacts.size(),
            terminalState.pairIndices.empty()
                ? nullptr : terminalState.pairIndices.begin(),
            terminalState.pairIndices.size(), terminalState.pairStates.size(),
            geometryEpoch, shellParticles, numShellParticles, bodies,
            numBodies, overlapTolerance, terminalState.projectionPasses,
            terminalState.overlapPairMask);
    terminalState.lastOverlapCount = overlap.overlapCount;
    terminalState.maximumPenetration = overlap.maximumPenetration;
    if (overlap.overlapCount == 0u) {
      if (rollbackApplied) {
        // The pose transaction was rejected, but tangential/free motion is
        // still valid. Publish one terminal normal owner for every rolled
        // pair so velocity handoff removes only renewed inward motion.
        for (physx::PxU32 pairIndex = 0;
             pairIndex < terminalState.pairStates.size(); ++pairIndex) {
          if (pairIndex >= terminalState.failClosedPairMask.size() ||
              terminalState.failClosedPairMask[pairIndex] == 0u)
            continue;
          AvbdOgcPairState &pair = terminalState.pairStates[pairIndex];
          const physx::PxU32 contactIndex =
              pair.geometry.representativeContact;
          if (!pair.geometry.active ||
              contactIndex >= terminalState.contacts.size() ||
              contactIndex >= terminalState.pairIndices.size() ||
              terminalState.pairIndices[contactIndex] != pairIndex)
            continue;
          pair.solve.publishLocalPositionResult(
              contactIndex, overlapTolerance,
              AvbdOgcVelocityContactDomain::eTERMINAL);
        }
      }
      consumeCurrentOgcPairRefreshRequests(
          terminalState.pairStates.empty()
              ? nullptr : terminalState.pairStates.begin(),
          terminalState.pairStates.size(),
          terminalState.pairIndices.empty()
              ? nullptr : terminalState.pairIndices.begin(),
          terminalState.pairIndices.size());
      terminalState.closureStatus = rollbackApplied
          ? AvbdTerminalOgcClosureStatus::eROLLED_BACK
          : AvbdTerminalOgcClosureStatus::eCONVERGED;
      break;
    }
    if (rollbackApplied) {
      terminalState.closureUnresolved = true;
      terminalState.closureStatus =
          AvbdTerminalOgcClosureStatus::eUNRESOLVED;
      break;
    }

    const AvbdTerminalOgcProgressAction progressAction =
        selectTerminalOgcProgressAction(
            overlap.overlapCount, previousProjectionAttempted,
            previousCommittedCorrections, terminalState.projectionPasses,
            kMaximumProjectionEpochs);
    if (progressAction ==
        AvbdTerminalOgcProgressAction::eFAIL_CLOSED) {
      terminalState.stalled = previousProjectionAttempted &&
          previousCommittedCorrections == 0u;
      const physx::PxU32 rolledBack = rollbackTerminalOgcComponents(
          terminalState, &overlap, /*rollbackAllPairs=*/false, bodies,
          numBodies, shellParticles, numShellParticles,
          softBodiesForRecovery, numSoftBodiesForRecovery);
      if (rolledBack == 0u) {
        terminalState.closureUnresolved = true;
        terminalState.closureStatus =
            AvbdTerminalOgcClosureStatus::eUNRESOLVED;
        break;
      }
      rollbackApplied = true;
      terminalState.failClosed = true;
      terminalState.currentPoseEpochApplied = true;
      continue;
    }

    PX_PROFILE_ZONE("AVBD.terminalCurrentPoseOgcProject", 0);
    physx::PxU32 committed = 0u;
    committed += applyWorldStaticTriangleCoreLocalManifold(
        shellParticles, numShellParticles, softBodiesForRecovery,
        numSoftBodiesForRecovery, terminalState.contacts.begin(),
        terminalState.contacts.size(), 1u, lengthScale, &stats,
        terminalSoftExecutionPlan, bodies, numBodies,
        terminalState.contacts.begin(), terminalState.contacts.size(),
        &geometryEpoch);
    committed += applyWorldStaticSoftNormalDepenetrationSweeps(
        shellParticles, numShellParticles, softBodiesForRecovery,
        numSoftBodiesForRecovery, terminalState.contacts.begin(),
        terminalState.contacts.size(), 1u, &stats,
        /*ogcExecutionPlan=*/nullptr, /*ogcRigidBodies=*/nullptr, 0u,
        /*ogcContacts=*/nullptr, 0u,
        terminalState.pairStates.empty()
            ? nullptr : terminalState.pairStates.begin(),
        terminalState.pairStates.size(),
        terminalState.pairIndices.empty()
            ? nullptr : terminalState.pairIndices.begin(),
        terminalState.pairIndices.size(),
        AvbdOgcVelocityContactDomain::eTERMINAL, &geometryEpoch);
    committed += applyDynamicSoftRigidNormalDepenetrationSweeps(
        bodies, numBodies, shellParticles, numShellParticles,
        softBodiesForRecovery, numSoftBodiesForRecovery,
        terminalState.contacts.begin(), terminalState.contacts.size(), 1u,
        lengthScale, &stats,
        terminalState.pairStates.empty()
            ? nullptr : terminalState.pairStates.begin(),
        terminalState.pairStates.size(),
        terminalState.pairIndices.empty()
            ? nullptr : terminalState.pairIndices.begin(),
        terminalState.pairIndices.size(),
        /*softComplianceResponseScale=*/4.0f,
        /*projectToCurrentPoseBoundary=*/true, nullptr, 0u, nullptr, 0u,
        AvbdOgcVelocityContactDomain::eTERMINAL, &geometryEpoch);
    committed += applyDynamicSoftRigidTriangleCoreLocalManifold(
        bodies, numBodies, shellParticles, numShellParticles,
        softBodiesForRecovery, numSoftBodiesForRecovery,
        terminalState.contacts.begin(), terminalState.contacts.size(), 1u,
        lengthScale, &stats,
        terminalState.pairStates.empty()
            ? nullptr : terminalState.pairStates.begin(),
        terminalState.pairStates.size(),
        terminalState.pairIndices.empty()
            ? nullptr : terminalState.pairIndices.begin(),
        terminalState.pairIndices.size(), nullptr, 0u, nullptr, 0u,
        AvbdOgcVelocityContactDomain::eTERMINAL, &geometryEpoch);
    consumeCurrentOgcPairRefreshRequests(
        terminalState.pairStates.empty()
            ? nullptr : terminalState.pairStates.begin(),
        terminalState.pairStates.size(),
        terminalState.pairIndices.empty()
            ? nullptr : terminalState.pairIndices.begin(),
        terminalState.pairIndices.size());

    previousProjectionAttempted = true;
    previousCommittedCorrections = committed;
    ++terminalState.projectionPasses;
    terminalState.committedCorrections += committed;
    terminalState.currentPoseEpochApplied = true;
  }
}

} // namespace Dy
} // namespace physx
