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

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/contact/DyAvbdContactPrep.h"
#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/ogc/DyAvbdOgcGeometryQueries.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/joint/DyAvbdJointCoupledMath.h"
#include "avbd/solver/joint/DyAvbdJointCoupledSystem.h"
#include "avbd/solver/joint/DyAvbdCoupledD6.h"
#include "avbd/solver/joint/DyAvbdJointDriveMath.h"
#include "avbd/solver/joint/DyAvbdJointFinalization.h"
#include "avbd/solver/joint/DyAvbdJointGeometryPolicy.h"
#include "avbd/solver/joint/DyAvbdJointIteration.h"
#include "avbd/solver/joint/DyAvbdJointOgcPhase.h"
#include "avbd/solver/joint/DyAvbdJointObjectiveCompilation.h"
#include "avbd/solver/joint/DyAvbdJointPreparation.h"
#include "avbd/solver/joint/DyAvbdJointPositionSolves.h"
#include "avbd/solver/joint/DyAvbdJointPhaseState.h"
#include "avbd/solver/joint/DyAvbdJointProjection.h"
#include "avbd/solver/joint/DyAvbdJointSoftExecutionData.h"
#include "avbd/solver/joint/DyAvbdJointSupportPolicies.h"
#include "avbd/solver/joint/DyAvbdJointVelocityPolicies.h"
#include "avbd/solver/joint/DyAvbdLinearDriveSolve.h"
#include "avbd/solver/rigid/DyAvbdRigidPhases.h"
#include "avbd/solver/soft/DyAvbdSoftWarmstart.h"
#include "avbd/solver/joint/DyAvbdNativeMotorVelocity.h"
#include "avbd/solver/joint/DyAvbdSpatialTendon.h"
#include "DyFeatherstoneArticulation.h"
#include "CmConeLimitHelper.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "PxConstraintDesc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// Enable detailed joint solver diagnostics (first N frames)
#ifndef AVBD_JOINT_DEBUG
#define AVBD_JOINT_DEBUG 0
#endif
#ifndef AVBD_JOINT_DEBUG_FRAMES
#define AVBD_JOINT_DEBUG_FRAMES 200
#endif

// External frame counter from DyAvbdDynamics.cpp (used by motor drives)
extern physx::PxU64 getAvbdMotorFrameCounter();

namespace physx {
namespace Dy {

// Conservative single-particle part of the shared OGC trust region.  The
// rigid endpoint has already been updated for this nonlinear GS iteration, so
// evaluate the soft candidate against that *current* pose.  Every incident
// dynamic pair votes for one common alpha; the particle may still deform in
// any tangent direction, but it cannot spend more normal clearance than the
// pair owns.  This is a candidate filter, not a post-solve depenetration.
// A deferred inertial target is intentionally released only through the
// contact-admitted soft body.  Unlike the ordinary material path, it can
// increase an already-large compressive load after the boundary clip.  Keep
// that release from degrading an incident tet: if the tet is healthy retain
// the existing 5% determinant floor; if it was already below the floor,
// accept only a non-worsening candidate so the regular material/contact
// equations still have a route to repair it.
// OGC's admission point is the pair prediction, not a post-penetration
// ejection.  Restricting only a rigid (or only the four simulation vertices
// behind one collision vertex) transfers the load into the wrong endpoint:
// the former launches the rigid and the latter shears/inverts the tet mesh.
//
// This one-time same-dt line search instead advances every endpoint of the
// active mixed island to one common feasible DCD epoch.  A source volume is
// moved coherently from its accepted frame-start state to its warmstarted
// candidate, and its paired 6DOF rigid uses the same alpha.  The regular
// coupled AVBD/OGC rows then distribute subsequent pressure locally through
// the soft support.  It is neither a sweep/TOI query nor a physics substep.
//
// World-static targets use the companion routine below.  They have no rigid
// endpoint to interpolate, so their admissible initial guess must be clipped
// at the *collision support* level.  Scaling an entire falling volume here
// makes gravity look disabled; scaling just the weighted query/core supports
// is the OGC vertex trust-region operation that lets the material block form
// the visible landing deformation.
void AvbdSolver::solveWithJoints(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    const physx::PxVec3 &gravity, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors,
    physx::PxU32 iterationOverride,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    FeatherstoneArticulation *const *articulationForBody,
    const physx::PxU32 *linkIndexForBody,
    AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solveWithJoints", 0);

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  const AvbdJointPhaseAdmission phaseAdmission =
      buildAvbdJointPhaseAdmission(
          mInitialized, numBodies, bodies, softParticles, numSoftParticles,
          softBodies, numSoftBodies, softContacts, numSoftContacts,
          softExecutionPlan);
  const bool hasCompleteSoftSelection =
      phaseAdmission.hasCompleteSoftSelection;
  if (!mInitialized ||
      (numBodies == 0 && !hasCompleteSoftSelection)) {
    return;
  }
  // The admission helper is the sole boundary for provider-plan validation.
  // Later phases consume only this immutable admission result.
  const bool useProvidedSoftExecutionPlan =
      phaseAdmission.useProvidedSoftExecutionPlan;
  const bool useProvidedRigidTargetContactPlan =
      phaseAdmission.useProvidedRigidTargetContactPlan;
  const bool hasPreparedSoftPrediction =
      phaseAdmission.hasPreparedSoftPrediction;

  AvbdJointOgcAdmissionState ogcAdmission;

  PX_AVBD_PROFILE_STAT(stats.numBodies = numBodies);
  PX_AVBD_PROFILE_STAT(stats.numContacts = numContacts);
  PX_AVBD_PROFILE_STAT(stats.numJoints = numD6 + numGear);

  const physx::PxReal invDt = 1.0f / dt;
  const physx::PxReal invDt2 = invDt * invDt;

  AvbdJointObjectiveCompilationState objectiveCompilation =
      compileAvbdJointObjectives(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          numGear, gravity, numSoftParticles,
          numSoftBodies, numSoftContacts);
  bool &coupledLinearPositionDriveIsland =
      objectiveCompilation.coupledLinearPositionDriveIsland;
  bool &coupledLinearPositionDriveFrictionPositionOwnerIsland =
      objectiveCompilation.coupledLinearPositionDriveFrictionPositionOwnerIsland;
  const bool &slerpVelocityDriveIsland =
      objectiveCompilation.slerpVelocityDriveIsland;
  bool &coupledAngularPositionDriveIsland =
      objectiveCompilation.coupledAngularPositionDriveIsland;
  bool &coupledLinearDriveIsland =
      objectiveCompilation.coupledLinearDriveIsland;
  bool &coupledFixedD6Island = objectiveCompilation.coupledFixedD6Island;
  bool &coupledSphericalConeIsland =
      objectiveCompilation.coupledSphericalConeIsland;
  bool &coupledSpatialTendonIsland =
      objectiveCompilation.coupledSpatialTendonIsland;
  physx::PxArray<physx::PxU32> &coupledSpatialTendonRowIndices =
      objectiveCompilation.coupledSpatialTendonRowIndices;
  const bool passiveCenteredGearVelocityProjectionIsland =
      isPassiveCenteredGearVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, gearJoints,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts);
  AvbdJointMotorAdmissionState motorAdmission =
      buildAvbdJointMotorAdmission(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          gearJoints, numGear, gravity, numSoftParticles, numSoftBodies,
          numSoftContacts);
  const bool nativeRevoluteMotorVelocityProjectionIsland =
      motorAdmission.nativeRevoluteMotorVelocityProjectionIsland;
  const bool contactCoupledNativeRevoluteMotorVelocityProjectionIsland =
      motorAdmission.contactCoupledNativeRevoluteMotorVelocityProjectionIsland;

  // Specialized contact-coupled joint objectives have first claim on their
  // physical contact sources. Compile every remaining ordinary rigid contact
  // only after that priority decision, so later solve stages consume an
  // immutable unique-owner program rather than overriding earlier flags.
  if (numSoftParticles == 0 && numSoftBodies == 0 &&
      numSoftContacts == 0) {
    compileAvbdOrdinaryRigidContactObjectives(
        contacts, numContacts, numBodies, contactMap);
  }

  AvbdJointVelocityPhaseState velocityPhase;
  initializeAvbdJointVelocityPhaseState(
      velocityPhase, mConfig, bodies, numBodies, contacts, numContacts,
      d6Joints, numD6, numGear, gravity, dt, numSoftParticles, numSoftBodies,
      numSoftContacts, nativeRevoluteMotorVelocityProjectionIsland,
      coupledLinearDriveIsland, coupledLinearPositionDriveIsland,
      coupledLinearPositionDriveFrictionPositionOwnerIsland,
      coupledAngularPositionDriveIsland, coupledSphericalConeIsland);
  const bool &passiveGenericHard1DVelocityProjectionIsland =
      velocityPhase.passiveGenericHard1DVelocityProjectionIsland;
  AvbdJointContactPhaseState contactPhase;
  buildAvbdJointContactPhaseState(
      contactPhase, bodies, numBodies, contacts, numContacts, numD6,
      numGear, softParticles, numSoftParticles, softContacts, numSoftContacts,
      passiveCenteredGearVelocityProjectionIsland);
  const physx::PxArray<bool> &touchesKinematicShell =
      contactPhase.touchesKinematicShell;
  physx::PxArray<physx::PxVec3> genericLinearVelAtSolveStart;
  physx::PxArray<physx::PxVec3> genericAngularVelAtSolveStart;
  if (passiveGenericHard1DVelocityProjectionIsland) {
    genericLinearVelAtSolveStart.resize(numBodies);
    genericAngularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      genericLinearVelAtSolveStart[i] = bodies[i].linearVelocity;
      genericAngularVelAtSolveStart[i] = bodies[i].angularVelocity;
    }
  }

  runAvbdJointWarmstartPhase(
      mConfig, dt, invDt, invDt2, bodies, numBodies, gravity,
      touchesKinematicShell, softParticles, numSoftParticles, softBodies,
      numSoftBodies, softContacts, numSoftContacts,
      hasPreparedSoftPrediction);

  AvbdJointOgcAdmissionInput ogcAdmissionInput{
      *this,
      dt,
      bodies,
      numBodies,
      softParticles,
      numSoftParticles,
      softBodies,
      numSoftBodies,
      softContacts,
      numSoftContacts,
      softExecutionPlan,
      useProvidedSoftExecutionPlan,
      hasCompleteSoftSelection,
      stats};
  ogcAdmission.initialize(ogcAdmissionInput);
  AvbdOgcPairTrustRegionContext &mixedOgcPairContext =
      ogcAdmission.pairContext;

  // Constraint preparation establishes the contact penalty policy, C0
  // history, deterministic order, and no-contact body state before execution
  // data materialization and the same-dt iteration phase.
  prepareAvbdJointConstraintPhase(
      mConfig, invDt2, bodies, numBodies, contacts, numContacts, d6Joints,
      numD6, gearJoints, numGear, contactMap);

  // Materialize the provider-owned execution plan, or build the equivalent
  // source-ordered local fallback for direct/native callers.  All storage is
  // owned by this scope; only immutable views cross into the solver phases.
  AvbdJointExecutionPhaseState executionPhase;
  AvbdJointExecutionPhaseInput executionInput{
      softExecutionPlan,
      useProvidedSoftExecutionPlan,
      useProvidedRigidTargetContactPlan,
      softBodies,
      numSoftBodies,
      softContacts,
      numSoftContacts,
      numSoftParticles,
      mixedOgcPairContext,
      executionPhase};
  initializeAvbdJointExecutionPhaseState(executionInput);
  const AvbdSoftExecutionData &softExecutionData =
      executionPhase.softExecutionData;
  const physx::PxU32 *softParticleBodyIndices =
      softExecutionData.particleBodyIndices;
  const physx::PxU32 *softContactStarts = softExecutionData.contactStarts;
  const AvbdSoftContactParticleRef *softContactRefs =
      softExecutionData.contactRefs;
  const physx::PxU32 *rigidTargetContactStarts =
      softExecutionData.rigidTargetContactStarts;
  const physx::PxU32 *rigidTargetContactRefs =
      softExecutionData.rigidTargetContactRefs;

  // =========================================================================
  // Stage 5: Main solver loop -- primal + dual per iteration (unified AL)
  //
  // Primal: Block Coordinate Descent over bodies
  //   (A) Contact constraints: full AVBD AL local system solve (3x3 or 6x6)
  //       Only for bodies WITH contacts. Bodies without contacts keep
  //       their current position (initialized above, then refined by GS).
  //   (B) Joint constraints:   Gauss-Seidel corrections (applied immediately)
  //       Each joint correction is applied to the body before processing
  //       the next joint, so subsequent joints see the updated state.
  //       Full correction (no relaxation) for equality constraints;
  //       the PBD generalized-mass denominator w naturally prevents
  //       overcorrection.
  //
  // Dual: AL multiplier updates for both contacts and joints
  // =========================================================================
  // Stage 5 owns the complete same-dt primal/dual iteration lifecycle.
  AvbdJointIterationPhaseInput iterationInput{
      mConfig,
      dt,
      invDt2,
      bodies,
      numBodies,
      contacts,
      numContacts,
      d6Joints,
      numD6,
      gearJoints,
      numGear,
      gravity,
      contactMap,
      d6Map,
      gearMap,
      iterationOverride,
      hasCompleteSoftSelection,
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
      mixedOgcPairContext.isComplete(numSoftContacts)
          ? &mixedOgcPairContext
          : nullptr,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr,
      articulationForBody,
      linkIndexForBody,
      slerpVelocityDriveIsland,
      coupledFixedD6Island,
      coupledSphericalConeIsland,
      coupledLinearPositionDriveIsland,
      coupledLinearPositionDriveFrictionPositionOwnerIsland,
      coupledLinearDriveIsland,
      coupledAngularPositionDriveIsland,
      coupledSpatialTendonIsland,
      coupledSpatialTendonRowIndices,
      stats};
  runAvbdJointIterationPhase(iterationInput);

  // Shared post-AL stages, OGC endpoint handoff, bending damping, and joint
  // velocity finalization are one explicit phase boundary.  The phase owner
  // preserves their order while keeping the mixed-island coordinator focused
  // on admission and iteration lifecycle.
  AvbdJointPostAlPhaseState postAlPhase;
  postAlPhase.run(
      *this, dt, invDt, bodies, numBodies, contacts, numContacts, contactMap,
      gravity, contactPhase, velocityPhase, motorAdmission, ogcAdmission,
      passiveCenteredGearVelocityProjectionIsland, coupledFixedD6Island,
      coupledLinearDriveIsland, coupledLinearPositionDriveIsland,
      coupledLinearPositionDriveFrictionPositionOwnerIsland,
      coupledAngularPositionDriveIsland, coupledSphericalConeIsland,
      /*hasJointConstraints=*/true,
      contactCoupledNativeRevoluteMotorVelocityProjectionIsland ||
          coupledLinearPositionDriveFrictionPositionOwnerIsland,
      /*applyVelocityDamping=*/true, softParticles, numSoftParticles,
      softBodies, numSoftBodies, softContacts, numSoftContacts,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr,
      useProvidedSoftExecutionPlan, articulationForBody, d6Joints, numD6,
      gearJoints, genericLinearVelAtSolveStart, genericAngularVelAtSolveStart,
      stats);

}

//=============================================================================
// Soft body VBD: per-particle 3x3 block coordinate descent
//=============================================================================

} // namespace Dy
} // namespace physx
