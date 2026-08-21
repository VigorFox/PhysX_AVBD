// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointIteration.h"
#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/joint/DyAvbdCoupledD6.h"
#include "avbd/solver/joint/DyAvbdJointPhaseState.h"
#include "avbd/solver/joint/DyAvbdJointPositionSolves.h"
#include "avbd/solver/joint/DyAvbdLinearDriveSolve.h"
#include "avbd/solver/joint/DyAvbdSpatialTendon.h"
#include "common/PxProfileZone.h"
#include <algorithm>

namespace physx {
namespace Dy {

// Position phase for soft particles and their coupled attachment owners.
// This boundary keeps the nonlinear soft-body GS pass explicit without
// changing the order of particle, rigid-attachment, and soft-pair solves.
struct AvbdJointSoftParticlePhaseInput {
  AvbdSoftParticle *softParticles;
  physx::PxU32 numSoftParticles;
  AvbdSolverBody *bodies;
  physx::PxU32 numBodies;
  AvbdSoftBody *softBodies;
  physx::PxU32 numSoftBodies;
  AvbdSoftContact *softContacts;
  physx::PxU32 numSoftContacts;
  const physx::PxU32 *softParticleBodyIndices;
  const physx::PxU32 *softContactStarts;
  const AvbdSoftContactParticleRef *softContactRefs;
  AvbdTetMaterialPacketKernels tetMaterialPacketKernels;
  const AvbdOgcPairTrustRegionContext *ogcPairContext;
  physx::PxReal dt;
  physx::PxReal invDt2;
  FeatherstoneArticulation *const *articulationForBody;
  const physx::PxU32 *linkIndexForBody;
};

void AvbdSolver::runAvbdJointSoftParticlePhase(
    const AvbdJointSoftParticlePhaseInput &input) {
  AvbdSoftParticle *const softParticles = input.softParticles;
  const physx::PxU32 numSoftParticles = input.numSoftParticles;
  AvbdSolverBody *const bodies = input.bodies;
  const physx::PxU32 numBodies = input.numBodies;
  AvbdSoftBody *const softBodies = input.softBodies;
  const physx::PxU32 numSoftBodies = input.numSoftBodies;
  AvbdSoftContact *const softContacts = input.softContacts;
  const physx::PxU32 numSoftContacts = input.numSoftContacts;
  const physx::PxU32 *const softParticleBodyIndices =
      input.softParticleBodyIndices;
  const physx::PxU32 *const softContactStarts = input.softContactStarts;
  const AvbdSoftContactParticleRef *const softContactRefs =
      input.softContactRefs;
  const AvbdTetMaterialPacketKernels &tetMaterialPacketKernels =
      input.tetMaterialPacketKernels;
  const AvbdOgcPairTrustRegionContext *const ogcPairContext =
      input.ogcPairContext;
  const physx::PxReal dt = input.dt;
  const physx::PxReal invDt2 = input.invDt2;
  FeatherstoneArticulation *const *const articulationForBody =
      input.articulationForBody;
  const physx::PxU32 *const linkIndexForBody = input.linkIndexForBody;
  if (numSoftParticles == 0 || numSoftBodies == 0)
    return;

  PX_PROFILE_ZONE("AVBD.softParticlePrimal", 0);

  // The soft-particle primal is nonlinear GS. It deliberately stays serial
  // until a conflict-colored taskgraph schedule is introduced; numBodies is
  // not a proxy for soft work and no private worker pool is entered here.
  for (physx::PxU32 spi = 0; spi < numSoftParticles; ++spi) {
    if (softParticles[spi].invMass <= 0.0f)
      continue;
    const physx::PxU32 softBodyIndex = softParticleBodyIndices[spi];
    if (softBodyIndex >= numSoftBodies)
      continue;
    solveSoftParticle(
        spi, softParticles, numSoftParticles, bodies, numBodies,
        softBodies[softBodyIndex], softContacts, numSoftContacts,
        softContactRefs, softContactStarts[spi], softContactStarts[spi + 1],
        dt, invDt2, tetMaterialPacketKernels, ogcPairContext);
  }

  solveSoftRigidAttachmentsCoupled(
      softParticles, numSoftParticles, bodies, numBodies, softBodies,
      numSoftBodies, dt, articulationForBody, linkIndexForBody);
  solveSoftPairAttachmentsCoupled(
      softParticles, numSoftParticles, softBodies, numSoftBodies, dt);
}

struct AvbdJointPrimalPhaseInput {
  const AvbdSolverConfig &config;
  AvbdSolverBody *bodies;
  physx::PxU32 numBodies;
  AvbdContactConstraint *contacts;
  physx::PxU32 numContacts;
  AvbdD6JointConstraint *d6Joints;
  physx::PxU32 numD6;
  AvbdGearJointConstraint *gearJoints;
  physx::PxU32 numGear;
  physx::PxReal dt;
  physx::PxReal invDt2;
  const physx::PxVec3 &gravity;
  const AvbdBodyConstraintMap *contactMap;
  const AvbdBodyConstraintMap *d6Map;
  const AvbdBodyConstraintMap *gearMap;
  AvbdSoftParticle *softParticles;
  physx::PxU32 numSoftParticles;
  AvbdSoftBody *softBodies;
  physx::PxU32 numSoftBodies;
  AvbdSoftContact *softContacts;
  physx::PxU32 numSoftContacts;
  const physx::PxU32 *softParticleBodyIndices;
  const physx::PxU32 *softContactStarts;
  const AvbdSoftContactParticleRef *softContactRefs;
  const physx::PxU32 *rigidTargetContactStarts;
  const physx::PxU32 *rigidTargetContactRefs;
  const AvbdOgcPairTrustRegionContext *mixedOgcPairContext;
  AvbdTetMaterialPacketKernels tetMaterialPacketKernels;
  FeatherstoneArticulation *const *articulationForBody;
  const physx::PxU32 *linkIndexForBody;
  bool &coupledFixedD6Island;
  bool &coupledSphericalConeIsland;
  bool &coupledLinearPositionDriveIsland;
  bool &coupledLinearPositionDriveFrictionPositionOwnerIsland;
  bool &coupledLinearDriveIsland;
  bool &coupledAngularPositionDriveIsland;
  bool &coupledSpatialTendonIsland;
  const physx::PxArray<physx::PxU32> &coupledSpatialTendonRowIndices;
  AvbdSolverStats &stats;
};

void AvbdSolver::runAvbdJointPrimalIteration(
    const AvbdJointPrimalPhaseInput &input) {
  const AvbdSolverConfig &config = input.config;
  AvbdSolverBody *const bodies = input.bodies;
  const physx::PxU32 numBodies = input.numBodies;
  AvbdContactConstraint *const contacts = input.contacts;
  const physx::PxU32 numContacts = input.numContacts;
  AvbdD6JointConstraint *const d6Joints = input.d6Joints;
  const physx::PxU32 numD6 = input.numD6;
  AvbdGearJointConstraint *const gearJoints = input.gearJoints;
  const physx::PxU32 numGear = input.numGear;
  const physx::PxReal dt = input.dt;
  const physx::PxReal invDt2 = input.invDt2;
  const physx::PxVec3 &gravity = input.gravity;
  const AvbdBodyConstraintMap *const contactMap = input.contactMap;
  const AvbdBodyConstraintMap *const d6Map = input.d6Map;
  const AvbdBodyConstraintMap *const gearMap = input.gearMap;
  AvbdSoftParticle *const softParticles = input.softParticles;
  const physx::PxU32 numSoftParticles = input.numSoftParticles;
  AvbdSoftBody *const softBodies = input.softBodies;
  const physx::PxU32 numSoftBodies = input.numSoftBodies;
  AvbdSoftContact *const softContacts = input.softContacts;
  const physx::PxU32 numSoftContacts = input.numSoftContacts;
  const physx::PxU32 *const softParticleBodyIndices =
      input.softParticleBodyIndices;
  const physx::PxU32 *const softContactStarts = input.softContactStarts;
  const AvbdSoftContactParticleRef *const softContactRefs =
      input.softContactRefs;
  const physx::PxU32 *const rigidTargetContactStarts =
      input.rigidTargetContactStarts;
  const physx::PxU32 *const rigidTargetContactRefs =
      input.rigidTargetContactRefs;
  const AvbdOgcPairTrustRegionContext *const mixedOgcPairContext =
      input.mixedOgcPairContext;
  const AvbdTetMaterialPacketKernels &tetMaterialPacketKernels =
      input.tetMaterialPacketKernels;
  FeatherstoneArticulation *const *const articulationForBody =
      input.articulationForBody;
  const physx::PxU32 *const linkIndexForBody = input.linkIndexForBody;
  bool &coupledFixedD6Island = input.coupledFixedD6Island;
  bool &coupledSphericalConeIsland = input.coupledSphericalConeIsland;
  bool &coupledLinearPositionDriveIsland =
      input.coupledLinearPositionDriveIsland;
  bool &coupledLinearPositionDriveFrictionPositionOwnerIsland =
      input.coupledLinearPositionDriveFrictionPositionOwnerIsland;
  bool &coupledLinearDriveIsland = input.coupledLinearDriveIsland;
  bool &coupledAngularPositionDriveIsland =
      input.coupledAngularPositionDriveIsland;
  bool &coupledSpatialTendonIsland = input.coupledSpatialTendonIsland;
  const physx::PxArray<physx::PxU32> &coupledSpatialTendonRowIndices =
      input.coupledSpatialTendonRowIndices;
  AvbdSolverStats &stats = input.stats;
  (void)stats;
  PX_PROFILE_ZONE("AVBD.blockDescentWithJoints", 0);

  bool coupledSolved = false;
  if (coupledFixedD6Island) {
    coupledSolved = solveCoupledFixedD6Island(
        bodies, numBodies, d6Joints[0], invDt2);
    if (!coupledSolved) {
      coupledFixedD6Island = false;
      fallbackAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledFixedD6);
    }
  } else if (coupledSphericalConeIsland) {
    coupledSolved = solveCoupledSphericalConeIsland(
        bodies, numBodies, d6Joints[0], invDt2);
    if (!coupledSolved) {
      coupledSphericalConeIsland = false;
      fallbackAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledSphericalCone);
    }
  } else if (coupledLinearPositionDriveIsland) {
    coupledSolved = solveCoupledLinearPositionDriveIsland(
        bodies, numBodies, contacts, numContacts, d6Joints[0], dt, invDt2,
        gravity, config);
    if (!coupledSolved) {
      coupledLinearPositionDriveIsland = false;
      coupledLinearPositionDriveFrictionPositionOwnerIsland = false;
      fallbackAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledLinearPositionDrive);
    }
  } else if (coupledLinearDriveIsland) {
    coupledSolved = solveCoupledLinearDriveIsland(
        bodies, numBodies, contacts, numContacts, d6Joints[0], dt, invDt2,
        config);
    if (!coupledSolved) {
      coupledLinearDriveIsland = false;
      fallbackAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledLinearVelocityDrive);
    }
  } else if (coupledAngularPositionDriveIsland) {
    coupledSolved = solveCoupledAngularPositionDriveIsland(
        bodies, numBodies, contacts, numContacts, d6Joints[0], dt, invDt2,
        config);
    if (!coupledSolved) {
      coupledAngularPositionDriveIsland = false;
      fallbackAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledAngularPositionDrive);
    }
  }

  const bool useDeterministicOrder =
      config.isDeterministic() &&
      (config.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);
  physx::PxArray<physx::PxU32> bodyOrder;
  if (useDeterministicOrder) {
    bodyOrder.resize(numBodies);
    for (physx::PxU32 bi = 0; bi < numBodies; ++bi)
      bodyOrder[bi] = bi;
    std::sort(bodyOrder.begin(), bodyOrder.end(),
              [&bodies](physx::PxU32 a, physx::PxU32 b) {
                return bodies[a].invMass > bodies[b].invMass;
              });
  }
  const physx::PxU32 *orderPtr =
      useDeterministicOrder ? bodyOrder.begin() : nullptr;

  auto solveBody = [&](physx::PxU32 idx) {
    const physx::PxU32 i = orderPtr ? orderPtr[idx] : idx;
    if (bodies[i].invMass <= 0.0f)
      return;
    solveLocalSystemWithJoints(
        bodies[i], bodies, numBodies, contacts, numContacts, d6Joints,
        numD6, gearJoints, numGear, dt, invDt2, contactMap, d6Map, gearMap,
        softParticles, numSoftParticles, softContacts, numSoftContacts,
        softBodies, numSoftBodies, rigidTargetContactStarts,
        rigidTargetContactRefs,
        mixedOgcPairContext &&
                mixedOgcPairContext->isComplete(numSoftContacts)
            ? mixedOgcPairContext
            : nullptr);
  };

  if (!coupledSolved) {
    for (physx::PxU32 idx = 0; idx < numBodies; ++idx)
      solveBody(idx);
  }

  if (coupledSpatialTendonIsland) {
    bool tendonSolved = true;
    for (physx::PxU32 row = 0;
         row < coupledSpatialTendonRowIndices.size(); ++row) {
      const physx::PxU32 rowIndex = coupledSpatialTendonRowIndices[row];
      tendonSolved =
          solveCoupledSpatialTendonRow(
              bodies, numBodies, d6Joints[rowIndex], dt, invDt2) &&
          tendonSolved;
    }
    if (!tendonSolved) {
      coupledSpatialTendonIsland = false;
      for (physx::PxU32 row = 0;
           row < coupledSpatialTendonRowIndices.size(); ++row) {
        fallbackAvbdJointObjective(
            d6Joints[coupledSpatialTendonRowIndices[row]].objectiveProgram,
            AvbdJointObjectiveKind::CoupledSpatialTendon);
      }
    }
  }

  AvbdJointSoftParticlePhaseInput softParticleInput{
      softParticles,
      numSoftParticles,
      bodies,
      numBodies,
      softBodies,
      numSoftBodies,
      softContacts,
      numSoftContacts,
      softParticleBodyIndices,
      softContactStarts,
      softContactRefs,
      tetMaterialPacketKernels,
      mixedOgcPairContext && mixedOgcPairContext->isComplete(numSoftContacts)
          ? mixedOgcPairContext
          : nullptr,
      dt,
      invDt2,
      articulationForBody,
      linkIndexForBody};
  runAvbdJointSoftParticlePhase(softParticleInput);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }
  PX_AVBD_PROFILE_STAT(stats.totalIterations++);
}

void AvbdSolver::runAvbdJointIterationPhase(
    const AvbdJointIterationPhaseInput &input) {
  const AvbdSolverConfig &config = input.config;
  const physx::PxReal dt = input.dt;
  const physx::PxReal invDt2 = input.invDt2;
  AvbdSolverBody *const bodies = input.bodies;
  const physx::PxU32 numBodies = input.numBodies;
  AvbdContactConstraint *const contacts = input.contacts;
  const physx::PxU32 numContacts = input.numContacts;
  AvbdD6JointConstraint *const d6Joints = input.d6Joints;
  const physx::PxU32 numD6 = input.numD6;
  AvbdGearJointConstraint *const gearJoints = input.gearJoints;
  const physx::PxU32 numGear = input.numGear;
  const physx::PxVec3 &gravity = input.gravity;
  const AvbdBodyConstraintMap *const contactMap = input.contactMap;
  const AvbdBodyConstraintMap *const d6Map = input.d6Map;
  const AvbdBodyConstraintMap *const gearMap = input.gearMap;
  const physx::PxU32 iterationOverride = input.iterationOverride;
  const bool hasCompleteSoftSelection = input.hasCompleteSoftSelection;
  AvbdSoftParticle *const softParticles = input.softParticles;
  const physx::PxU32 numSoftParticles = input.numSoftParticles;
  AvbdSoftBody *const softBodies = input.softBodies;
  const physx::PxU32 numSoftBodies = input.numSoftBodies;
  AvbdSoftContact *const softContacts = input.softContacts;
  const physx::PxU32 numSoftContacts = input.numSoftContacts;
  const physx::PxU32 *const softParticleBodyIndices =
      input.softParticleBodyIndices;
  const physx::PxU32 *const softContactStarts = input.softContactStarts;
  const AvbdSoftContactParticleRef *const softContactRefs =
      input.softContactRefs;
  const physx::PxU32 *const rigidTargetContactStarts =
      input.rigidTargetContactStarts;
  const physx::PxU32 *const rigidTargetContactRefs =
      input.rigidTargetContactRefs;
  AvbdOgcPairTrustRegionContext *const mixedOgcPairContext =
      input.mixedOgcPairContext;
  const AvbdSoftIslandExecutionPlan *const softExecutionPlan =
      input.softExecutionPlan;
  FeatherstoneArticulation *const *const articulationForBody =
      input.articulationForBody;
  const physx::PxU32 *const linkIndexForBody = input.linkIndexForBody;
  const bool slerpVelocityDriveIsland = input.slerpVelocityDriveIsland;
  bool &coupledFixedD6Island = input.coupledFixedD6Island;
  bool &coupledSphericalConeIsland = input.coupledSphericalConeIsland;
  bool &coupledLinearPositionDriveIsland =
      input.coupledLinearPositionDriveIsland;
  bool &coupledLinearPositionDriveFrictionPositionOwnerIsland =
      input.coupledLinearPositionDriveFrictionPositionOwnerIsland;
  bool &coupledLinearDriveIsland = input.coupledLinearDriveIsland;
  bool &coupledAngularPositionDriveIsland =
      input.coupledAngularPositionDriveIsland;
  bool &coupledSpatialTendonIsland = input.coupledSpatialTendonIsland;
  const physx::PxArray<physx::PxU32> &coupledSpatialTendonRowIndices =
      input.coupledSpatialTendonRowIndices;
  AvbdSolverStats &stats = input.stats;
  bool hasDynamicSoftRigidContact = false;
  for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    const AvbdSoftContactGeometry &geometry = softContacts[sci].geometry;
    if (geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numBodies &&
        bodies[geometry.targetIndex].invMass > 0.0f) {
      hasDynamicSoftRigidContact = true;
      break;
    }
  }

  AvbdJointPositionPhaseState positionPhase;
  initializeAvbdJointPositionPhaseState(
      positionPhase, config, slerpVelocityDriveIsland,
      coupledFixedD6Island, coupledSphericalConeIsland,
      coupledSpatialTendonIsland, hasDynamicSoftRigidContact, bodies,
      numBodies);

  const physx::PxU32 baseIterations =
      physx::PxMax(config.iterations, iterationOverride);
  const bool hasExplicitJointConstraints = numD6 > 0 || numGear > 0;
  const physx::PxU32 jointIterations =
      hasExplicitJointConstraints && config.jointIterationOverride > 0
          ? physx::PxMax(baseIterations, config.jointIterationOverride)
          : baseIterations;
  const physx::PxU32 minIterations =
      hasExplicitJointConstraints && config.jointIterationOverride > 0
          ? physx::PxMin(jointIterations, config.jointIterationOverride)
          : physx::PxMin(jointIterations, physx::PxU32(4));
  const bool enableEarlyStop =
      config.enableEarlyStop && !hasCompleteSoftSelection &&
      jointIterations - minIterations > 1;
  const physx::PxReal rotationTolerance =
      physx::PxMax(4.0f * config.positionTolerance /
                       physx::PxMax(config.lengthScale, 1e-6f),
                   1e-4f);
  physx::PxU32 consecutiveConvergedIterations = 0;
  physx::PxArray<physx::PxVec3> earlyStopPrevPos;
  physx::PxArray<physx::PxQuat> earlyStopPrevRot;
  if (enableEarlyStop) {
    earlyStopPrevPos.resize(numBodies);
    earlyStopPrevRot.resize(numBodies);
  }

  const AvbdTetMaterialPacketKernels tetMaterialPacketKernels =
      numSoftParticles > 0 && numSoftBodies > 0 &&
              !config.requiresOrderedBackend()
          ? avbdSelectTetMaterialPacketKernels(softBodies, numSoftBodies)
          : AvbdTetMaterialPacketKernels{NULL, NULL};
  AvbdOgcPoseWritePhaseState ogcPoseWritePhase;

  for (physx::PxU32 iter = 0; iter < jointIterations; ++iter) {
    if (positionPhase.useChebyshev) {
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        positionPhase.chebyPrevPrevPos[i] = positionPhase.chebyPrevPos[i];
        positionPhase.chebyPrevPrevRot[i] = positionPhase.chebyPrevRot[i];
        positionPhase.chebyPrevPos[i] = bodies[i].position;
        positionPhase.chebyPrevRot[i] = bodies[i].rotation;
      }
    }
    if (enableEarlyStop) {
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        earlyStopPrevPos[i] = bodies[i].position;
        earlyStopPrevRot[i] = bodies[i].rotation;
      }
    }

    ogcPoseWritePhase.capture(
        mixedOgcPairContext, softContacts, numSoftContacts,
        softParticles, numSoftParticles, bodies, numBodies);

    AvbdJointPrimalPhaseInput primalInput{
        config,
        bodies,
        numBodies,
        contacts,
        numContacts,
        d6Joints,
        numD6,
        gearJoints,
        numGear,
        dt,
        invDt2,
        gravity,
        contactMap,
        d6Map,
        gearMap,
        softParticles,
        numSoftParticles,
        softBodies,
        numSoftBodies,
        softContacts,
        numSoftContacts,
        softParticleBodyIndices,
        softContactStarts,
        softContactRefs,
        rigidTargetContactStarts,
        rigidTargetContactRefs,
        mixedOgcPairContext,
        tetMaterialPacketKernels,
        articulationForBody,
        linkIndexForBody,
        coupledFixedD6Island,
        coupledSphericalConeIsland,
        coupledLinearPositionDriveIsland,
        coupledLinearPositionDriveFrictionPositionOwnerIsland,
        coupledLinearDriveIsland,
        coupledAngularPositionDriveIsland,
        coupledSpatialTendonIsland,
        coupledSpatialTendonRowIndices,
        stats};
    runAvbdJointPrimalIteration(primalInput);

    // Admission exhausts only the affected pair component's relative-motion
    // budget.  Other material and joint degrees of freedom must keep their
    // configured nonlinear iteration budget until terminal DCD publishes the
    // next geometry epoch.
    admitOgcPoseWritePhase(
        ogcPoseWritePhase, mixedOgcPairContext, bodies, numBodies,
        softParticles, numSoftParticles, softBodies, numSoftBodies,
        softContacts, numSoftContacts, softExecutionPlan);

    runAvbdJointDualPhase(
        bodies, numBodies, contacts, numContacts, d6Joints, numD6,
        gearJoints, numGear, dt, invDt2, config, softParticles,
        numSoftParticles, softBodies, numSoftBodies, softContacts,
        numSoftContacts, stats);

    if (applyAvbdJointIterationPolicy(
            bodies, numBodies, iter, positionPhase, config, enableEarlyStop,
            minIterations, rotationTolerance, earlyStopPrevPos,
            earlyStopPrevRot, consecutiveConvergedIterations))
      break;
  }
}

} // namespace Dy
} // namespace physx
