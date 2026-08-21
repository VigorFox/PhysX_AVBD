// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointOgcPhase.h"
#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/ogc/DyAvbdOgcDynamicResponse.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcStaticResponse.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

void AvbdJointOgcAdmissionState::initialize(
    const AvbdJointOgcAdmissionInput &input) {
  AvbdSolver &solver = input.solver;
  const physx::PxReal dt = input.dt;
  AvbdSolverBody *const bodies = input.bodies;
  const physx::PxU32 numBodies = input.numBodies;
  AvbdSoftParticle *const softParticles = input.softParticles;
  const physx::PxU32 numSoftParticles = input.numSoftParticles;
  AvbdSoftBody *const softBodies = input.softBodies;
  const physx::PxU32 numSoftBodies = input.numSoftBodies;
  AvbdSoftContact *const softContacts = input.softContacts;
  const physx::PxU32 numSoftContacts = input.numSoftContacts;
  const AvbdSoftIslandExecutionPlan *const softExecutionPlan =
      input.softExecutionPlan;
  const bool useProvidedSoftExecutionPlan =
      input.useProvidedSoftExecutionPlan;
  const AvbdOgcGeometryEpochView geometryEpoch =
      makeOgcGeometryEpochView(
          useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr);
  const bool hasCompleteSoftSelection = input.hasCompleteSoftSelection;
  AvbdSolverStats &stats = input.stats;
  applyWorldStaticOgcInitialAdmission(
      softParticles, numSoftParticles, softBodies, numSoftBodies,
      softContacts, numSoftContacts,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr);
  applyOgcMixedWarmstartAdmission(
      bodies, numBodies, softParticles, numSoftParticles, softBodies,
      numSoftBodies, softContacts, numSoftContacts,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr,
      &admissionContacts, &admissionDisplacements);
  initializeMixedOgcPairEpoch(
      bodies, numBodies, softParticles, numSoftParticles, softContacts,
      numSoftContacts, softExecutionPlan, useProvidedSoftExecutionPlan,
      admissionContacts, admissionDisplacements, pairContext, pairStates,
      numPairStates);
  initializeAvbdSoftContactDepenetrationTargets(
      bodies, numBodies, softParticles, softBodies, numSoftBodies,
      softContacts, numSoftContacts, dt);

  if (!hasCompleteSoftSelection || numSoftContacts == 0)
    return;

  PX_PROFILE_ZONE("AVBD.worldStaticTriangleCoreLocalManifold", 0);
  applyWorldStaticTriangleCoreLocalManifold(
      softParticles, numSoftParticles, softBodies, numSoftBodies,
      softContacts, numSoftContacts, 1u,
      solver.getConfig().lengthScale, &stats,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr,
      bodies, numBodies, softContacts, numSoftContacts,
      useProvidedSoftExecutionPlan ? &geometryEpoch : nullptr);

  PX_PROFILE_ZONE("AVBD.dynamicSoftRigidTriangleCoreLocalManifold", 0);
  applyDynamicSoftRigidTriangleCoreLocalManifold(
      bodies, numBodies, softParticles, numSoftParticles, softBodies,
      numSoftBodies, softContacts, numSoftContacts, 4u,
      solver.getConfig().lengthScale, &stats, pairStates,
      numPairStates,
      useProvidedSoftExecutionPlan ? softExecutionPlan->ogcPairIndices
                                   : nullptr,
      useProvidedSoftExecutionPlan ? softExecutionPlan->numOgcPairIndices
                                   : 0u,
      useProvidedSoftExecutionPlan &&
              softExecutionPlan->hasMixedOgcPairContactPlan(numSoftContacts)
          ? softExecutionPlan->ogcPairContactStarts
          : nullptr,
      useProvidedSoftExecutionPlan &&
              softExecutionPlan->hasMixedOgcPairContactPlan(numSoftContacts)
          ? softExecutionPlan->numOgcPairContactStarts
          : 0u,
      useProvidedSoftExecutionPlan &&
              softExecutionPlan->hasMixedOgcPairContactPlan(numSoftContacts)
              ? softExecutionPlan->ogcPairContactRefs
          : nullptr,
      useProvidedSoftExecutionPlan &&
              softExecutionPlan->hasMixedOgcPairContactPlan(numSoftContacts)
              ? softExecutionPlan->numOgcPairContactRefs
          : 0u,
      AvbdOgcVelocityContactDomain::eSELECTION,
      useProvidedSoftExecutionPlan ? &geometryEpoch : nullptr);

  PX_PROFILE_ZONE("AVBD.worldStaticTriangleCoreLocalManifold", 0);
  applyWorldStaticTriangleCoreLocalManifold(
      softParticles, numSoftParticles, softBodies, numSoftBodies,
      softContacts, numSoftContacts, 1u,
      solver.getConfig().lengthScale, &stats,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr,
      bodies, numBodies, softContacts, numSoftContacts,
      useProvidedSoftExecutionPlan ? &geometryEpoch : nullptr);
}

} // namespace Dy
} // namespace physx

namespace physx {
namespace Dy {

void AvbdSolver::applyAvbdJointOgcVelocityHandoff(
    const AvbdJointOgcVelocityHandoffInput &input)
{
    AvbdSolverBody *const bodies = input.bodies;
    const physx::PxU32 numBodies = input.numBodies;
    AvbdSoftParticle *const softParticles = input.softParticles;
    const physx::PxU32 numSoftParticles = input.numSoftParticles;
    AvbdSoftContact *const softContacts = input.softContacts;
    const physx::PxU32 numSoftContacts = input.numSoftContacts;
    Dy::AvbdOgcPairState *const mixedOgcPairStates = input.mixedOgcPairStates;
    const physx::PxU32 numMixedOgcPairStates = input.numMixedOgcPairStates;
    const AvbdSoftIslandExecutionPlan *const softExecutionPlan =
        input.softExecutionPlan;
    const bool useProvidedSoftExecutionPlan = input.useProvidedSoftExecutionPlan;
    AvbdSolverStats &stats = input.stats;
    if (mixedOgcPairStates && numMixedOgcPairStates > 0)
    {
        PX_PROFILE_ZONE("AVBD.admittedMixedOgcPairInelasticVel", 0);
        ::physx::Dy::clampAdmittedMixedOgcPairNormalVelocities(
            bodies, numBodies, softParticles, numSoftParticles, softContacts,
            numSoftContacts, mixedOgcPairStates, numMixedOgcPairStates,
            useProvidedSoftExecutionPlan
                ? softExecutionPlan->ogcPairIndices
                : nullptr,
            useProvidedSoftExecutionPlan
                ? softExecutionPlan->numOgcPairIndices
                : 0u,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->ogcPairContactStarts
                : nullptr,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->numOgcPairContactStarts
                : 0u,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->ogcPairContactRefs
                : nullptr,
            useProvidedSoftExecutionPlan &&
                    softExecutionPlan->hasMixedOgcPairContactPlan(
                        numSoftContacts)
                ? softExecutionPlan->numOgcPairContactRefs
                : 0u,
            mConfig.lengthScale, &stats);
    }

}

} // namespace Dy
} // namespace physx
