// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointFinalization.h"
#include "avbd/solver/joint/DyAvbdCoupledD6.h"
#include "avbd/solver/joint/DyAvbdJointOgcPhase.h"
#include "avbd/solver/joint/DyAvbdJointPhaseState.h"
#include "avbd/solver/joint/DyAvbdJointSupportPolicies.h"
#include "avbd/solver/joint/DyAvbdJointVelocityPolicies.h"
#include "avbd/solver/joint/DyAvbdLinearDriveSolve.h"
#include "avbd/solver/joint/DyAvbdNativeMotorVelocity.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

void initializeAvbdJointVelocityPhaseState(
    AvbdJointVelocityPhaseState &state, const AvbdSolverConfig &config,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, const physx::PxVec3 &gravity, physx::PxReal dt,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts,
    bool nativeRevoluteMotorVelocityProjectionIsland,
    bool coupledLinearDriveIsland,
    bool coupledLinearPositionDriveIsland,
    bool coupledLinearPositionDriveFrictionPositionOwnerIsland,
    bool coupledAngularPositionDriveIsland,
    bool coupledSphericalConeIsland) {
  state.conserveNativeRevoluteMotorAngularMomentum = false;
  state.nativeRevoluteMotorExpectedAngularMomentum = 0.0f;
  state.conserveNativeRevoluteMotorAngularMomentumVector = false;
  state.nativeRevoluteMotorExpectedAngularMomentumVector =
      physx::PxVec3(0.0f);
  state.conserveNativeRevoluteMotorLinearMomentum = false;
  state.nativeRevoluteMotorExpectedLinearMomentum = physx::PxVec3(0.0f);
  state.conserveNativeRevoluteMotorSpatialMomentum = false;
  state.nativeRevoluteMotorExpectedSpatialAngularMomentum =
      physx::PxVec3(0.0f);
  state.useNativeRevoluteMotorSolveStartRelativeVelocity = false;
  state.nativeRevoluteMotorSolveStartRelativeVelocity = 0.0f;

  if (nativeRevoluteMotorVelocityProjectionIsland) {
    const AvbdD6JointConstraint &motor = d6Joints[0];
    const physx::PxU32 bodyA = motor.header.bodyIndexA;
    const physx::PxU32 bodyB = motor.header.bodyIndexB;
    const bool dynamicA =
        bodyA < numBodies && bodies[bodyA].invMass > 0.0f;
    const bool dynamicB =
        bodyB < numBodies && bodies[bodyB].invMass > 0.0f;
    physx::PxVec3 motorAxis =
        dynamicA
            ? (bodies[bodyA].rotation * motor.localFrameA)
                  .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f))
            : motor.localFrameA.rotate(
                  physx::PxVec3(1.0f, 0.0f, 0.0f));
    if (motorAxis.normalize() > 1e-6f) {
      if (dynamicA)
        state.nativeRevoluteMotorSolveStartRelativeVelocity -=
            motorAxis.dot(bodies[bodyA].angularVelocity);
      if (dynamicB)
        state.nativeRevoluteMotorSolveStartRelativeVelocity +=
            motorAxis.dot(bodies[bodyB].angularVelocity);
      state.useNativeRevoluteMotorSolveStartRelativeVelocity =
          physx::PxIsFinite(
              state.nativeRevoluteMotorSolveStartRelativeVelocity);
    }
    if (bodyA < numBodies && bodyB < numBodies && bodyA != bodyB &&
        bodies[bodyA].invMass > 0.0f &&
        bodies[bodyB].invMass > 0.0f) {
      physx::PxVec3 axis =
          (bodies[bodyA].rotation * motor.localFrameA)
              .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
      if (axis.normalize() > 1e-6f) {
        const physx::PxMat33 inertiaA =
            bodies[bodyA].invInertiaWorld.getInverse();
        const physx::PxMat33 inertiaB =
            bodies[bodyB].invInertiaWorld.getInverse();
        state.nativeRevoluteMotorExpectedAngularMomentum = axis.dot(
            inertiaA.transform(bodies[bodyA].angularVelocity) *
                    motor.motorGearRatio +
                inertiaB.transform(bodies[bodyB].angularVelocity));
        state.conserveNativeRevoluteMotorAngularMomentum =
            physx::PxIsFinite(
                state.nativeRevoluteMotorExpectedAngularMomentum);
        if (physx::PxAbs(motor.motorGearRatio - 1.0f) <= 1e-6f &&
            motor.anchorA.magnitudeSquared() <= 1e-8f &&
            motor.anchorB.magnitudeSquared() <= 1e-8f) {
          state.nativeRevoluteMotorExpectedAngularMomentumVector =
              inertiaA.transform(bodies[bodyA].angularVelocity) +
              inertiaB.transform(bodies[bodyB].angularVelocity);
          state.conserveNativeRevoluteMotorAngularMomentumVector =
              state.nativeRevoluteMotorExpectedAngularMomentumVector
                  .isFinite();
        }
        const physx::PxReal massA = 1.0f / bodies[bodyA].invMass;
        const physx::PxReal massB = 1.0f / bodies[bodyB].invMass;
        state.nativeRevoluteMotorExpectedLinearMomentum =
            bodies[bodyA].linearVelocity * massA +
            bodies[bodyB].linearVelocity * massB;
        state.conserveNativeRevoluteMotorLinearMomentum =
            physx::PxIsFinite(massA) && physx::PxIsFinite(massB) &&
            state.nativeRevoluteMotorExpectedLinearMomentum.isFinite();
        state.nativeRevoluteMotorExpectedSpatialAngularMomentum =
            bodies[bodyA].position.cross(
                bodies[bodyA].linearVelocity * massA) +
            inertiaA.transform(bodies[bodyA].angularVelocity) +
            bodies[bodyB].position.cross(
                bodies[bodyB].linearVelocity * massB) +
            inertiaB.transform(bodies[bodyB].angularVelocity);
        state.conserveNativeRevoluteMotorSpatialMomentum =
            state.conserveNativeRevoluteMotorLinearMomentum &&
            state.nativeRevoluteMotorExpectedSpatialAngularMomentum
                .isFinite();
      }
    }
  }

  state.passiveGenericHard1DIndex = PX_MAX_U32;
  state.passiveGenericHard1DVelocityProjectionIsland =
      isSinglePassiveGenericHard1DVelocityProjectionSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts,
          state.passiveGenericHard1DIndex);

  state.coupledExpectedMomentum = physx::PxVec3(0.0f);
  state.coupledExpectedAngularMomentum = physx::PxVec3(0.0f);
  state.conserveCoupledLinearPositionSupportAxisMomentum =
      coupledLinearPositionDriveIsland &&
      !coupledLinearPositionDriveFrictionPositionOwnerIsland &&
      (d6Joints[0].anchorA - d6Joints[0].anchorB).magnitudeSquared() >
          1e-12f;
  state.coupledLinearPositionSupportAxis = physx::PxVec3(0.0f);
  state.coupledExpectedLinearPositionSupportAxisAngularMomentum = 0.0f;
  if (state.conserveCoupledLinearPositionSupportAxisMomentum) {
    state.coupledLinearPositionSupportAxis = contacts[0].contactNormal;
    const physx::PxReal supportAxisLength =
        state.coupledLinearPositionSupportAxis.normalize();
    const physx::PxU32 bodyA = d6Joints[0].header.bodyIndexA;
    const physx::PxU32 bodyB = d6Joints[0].header.bodyIndexB;
    state.conserveCoupledLinearPositionSupportAxisMomentum =
        supportAxisLength > 1e-6f &&
        computeTwoBodySupportAxisAngularMomentum(
            bodies[bodyA], bodies[bodyB],
            state.coupledLinearPositionSupportAxis,
            config.angularDamping, config.angularDamping,
            state.coupledExpectedLinearPositionSupportAxisAngularMomentum);
  }
  if ((coupledLinearDriveIsland || coupledLinearPositionDriveIsland) &&
      !coupledLinearPositionDriveFrictionPositionOwnerIsland) {
    physx::PxReal totalMass = 0.0f;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal mass = 1.0f / bodies[i].invMass;
      totalMass += mass;
      state.coupledExpectedMomentum += bodies[i].linearVelocity * mass;
    }
    state.coupledExpectedMomentum =
        (state.coupledExpectedMomentum + gravity * (totalMass * dt)) *
        config.velocityDamping;
  }
  if (coupledAngularPositionDriveIsland || coupledSphericalConeIsland) {
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      state.coupledExpectedAngularMomentum +=
          bodies[i].invInertiaWorld.getInverse() *
          bodies[i].angularVelocity;
    state.coupledExpectedAngularMomentum *= config.velocityDamping;
  }
}

// Finalization for coupled joint null-space momentum.  This phase is kept
// separate from contact and drive projection so that the coordinator only
// publishes the already-captured solve-start targets and does not own the
// conservation formulas themselves.
static void finalizeAvbdCoupledJointMomentum(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdD6JointConstraint *d6Joints,
    bool coupledLinearDriveIsland, bool coupledLinearPositionDriveIsland,
    bool coupledLinearPositionDriveFrictionPositionOwnerIsland,
    bool coupledAngularPositionDriveIsland, bool coupledSphericalConeIsland,
    const physx::PxVec3 &coupledExpectedMomentum,
    const physx::PxVec3 &coupledExpectedAngularMomentum,
    bool conserveCoupledLinearPositionSupportAxisMomentum,
    const physx::PxVec3 &coupledLinearPositionSupportAxis,
    physx::PxReal coupledExpectedLinearPositionSupportAxisAngularMomentum,
    physx::PxReal dt) {
  if ((coupledLinearDriveIsland || coupledLinearPositionDriveIsland) &&
      !coupledLinearPositionDriveFrictionPositionOwnerIsland) {
    physx::PxReal totalMass = 0.0f;
    physx::PxVec3 finalMomentum(0.0f);
    bool velocityLimitActive = false;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxReal mass = 1.0f / bodies[i].invMass;
      totalMass += mass;
      finalMomentum += bodies[i].linearVelocity * mass;
      const physx::PxReal maxSpeedSquared = bodies[i].maxLinearVelocitySq;
      velocityLimitActive |=
          maxSpeedSquared > 0.0f &&
          bodies[i].linearVelocity.magnitudeSquared() >=
              maxSpeedSquared * (1.0f - 1e-5f);
    }
    if (!velocityLimitActive) {
      // Close only the translation-invariant equation after the shared
      // velocity finalize. Both velocity-drive and position-drive objectives
      // are internal, so neither may create a shared tangent momentum mode.
      // The strict frictional support owner is excluded: ground friction is
      // an external impulse, and restoring solve-start tangent momentum would
      // erase its physical effect.
      physx::PxVec3 momentumError =
          finalMomentum - coupledExpectedMomentum;
      if (numContacts > 0) {
        physx::PxVec3 supportNormal = contacts[0].contactNormal;
        supportNormal.normalize();
        momentumError -= supportNormal * momentumError.dot(supportNormal);
      }
      const physx::PxVec3 commonVelocity = momentumError / totalMass;
      for (physx::PxU32 i = 0; i < numBodies; ++i) {
        bodies[i].linearVelocity -= commonVelocity;
        bodies[i].position -= commonVelocity * dt;
      }
    }
  }

  if (coupledLinearPositionDriveIsland &&
      conserveCoupledLinearPositionSupportAxisMomentum) {
    const physx::PxU32 bodyA = d6Joints[0].header.bodyIndexA;
    const physx::PxU32 bodyB = d6Joints[0].header.bodyIndexB;
    // This is a shared rigid null-mode closure, not a second drive solve: it
    // leaves every rotating-frame D6 coordinate derivative and every
    // support-normal point velocity unchanged.
    restoreTwoBodySupportAxisAngularMomentum(
        bodies[bodyA], bodies[bodyB], coupledLinearPositionSupportAxis,
        coupledExpectedLinearPositionSupportAxisAngularMomentum);
  }

  if (coupledAngularPositionDriveIsland || coupledSphericalConeIsland) {
    physx::PxMat33 totalInertia(physx::PxZero);
    physx::PxVec3 finalAngularMomentum(0.0f);
    bool velocityLimitActive = false;
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      const physx::PxMat33 inertia =
          bodies[i].invInertiaWorld.getInverse();
      totalInertia += inertia;
      finalAngularMomentum += inertia * bodies[i].angularVelocity;
      const physx::PxReal maxSpeedSquared =
          bodies[i].maxAngularVelocitySq;
      velocityLimitActive |=
          maxSpeedSquared > 0.0f &&
          bodies[i].angularVelocity.magnitudeSquared() >=
              maxSpeedSquared * (1.0f - 1e-5f);
    }
    if (!velocityLimitActive) {
      // An internal centered angular row cannot change island angular
      // momentum. Remove only the shared world-angular mode after velocity
      // finalize; applying the same correction to both endpoints preserves
      // relative angular velocity and the D6/cone state.
      const physx::PxVec3 angularMomentumError =
          finalAngularMomentum - coupledExpectedAngularMomentum;
      const physx::PxVec3 commonAngularVelocity =
          totalInertia.getInverse() * angularMomentumError;
      const physx::PxReal correctionSpeed =
          commonAngularVelocity.magnitude();
      for (physx::PxU32 i = 0; i < numBodies; ++i)
        bodies[i].angularVelocity -= commonAngularVelocity;
      if (correctionSpeed > 1e-10f && PxIsFinite(correctionSpeed)) {
        const physx::PxQuat correction(
            -correctionSpeed * dt,
            commonAngularVelocity * (1.0f / correctionSpeed));
        for (physx::PxU32 i = 0; i < numBodies; ++i)
          bodies[i].rotation =
              (correction * bodies[i].rotation).getNormalized();
      }
    }
  }
}

static void finalizeAvbdJointVelocityPolicies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6) {
  // Mimic velocity iterations are hard even when the position equation is
  // compliant. Position ownership remains in the generic-hard AL path for
  // hard mimic rows and in the compliant position path for compliant mimic
  // rows.
  for (physx::PxU32 i = 0; i < numD6; ++i) {
    if (hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::ArticulationHardMimic) ||
        hasAvbdJointObjective(
            d6Joints[i].objectiveProgram,
            AvbdJointObjectiveKind::ArticulationCompliantMimic)) {
      PX_PROFILE_ZONE("AVBD.projectArticulationMimicVelocity1D", 0);
      projectArticulationMimicVelocity1D(
          bodies, numBodies, d6Joints[i]);
    }
  }

  // Locked pose/velocity projection is the final rigid-body policy boundary:
  // it must run after every admitted velocity objective and before the solver
  // hands state back to the caller.
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f) {
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
      bodies[i].projectLockedVelocities();
    }
  }
}

static void runAvbdJointVelocityFinalizePhase(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints,
    const physx::PxVec3 &gravity, physx::PxReal dt,
    const AvbdJointMotorAdmissionState &motorAdmission,
    const AvbdJointVelocityPhaseState &velocityPhase,
    bool passiveCenteredGearVelocityProjectionIsland,
    bool coupledFixedD6Island, bool coupledLinearDriveIsland,
    bool coupledLinearPositionDriveIsland,
    bool coupledLinearPositionDriveFrictionPositionOwnerIsland,
    bool coupledAngularPositionDriveIsland,
    bool coupledSphericalConeIsland,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &genericLinearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &genericAngularVelAtSolveStart) {
  if (motorAdmission.nativeRevoluteMotorVelocityProjectionIsland) {
    PX_PROFILE_ZONE("AVBD.projectNativeRevoluteMotorVelocity", 0);
    projectSingleNativeRevoluteMotorVelocity(
        bodies, numBodies, d6Joints[0], dt,
        velocityPhase.conserveNativeRevoluteMotorAngularMomentum,
        velocityPhase.nativeRevoluteMotorExpectedAngularMomentum,
        velocityPhase.conserveNativeRevoluteMotorAngularMomentumVector,
        velocityPhase.nativeRevoluteMotorExpectedAngularMomentumVector,
        velocityPhase.conserveNativeRevoluteMotorLinearMomentum,
        velocityPhase.nativeRevoluteMotorExpectedLinearMomentum,
        velocityPhase.conserveNativeRevoluteMotorSpatialMomentum,
        velocityPhase.nativeRevoluteMotorExpectedSpatialAngularMomentum,
        velocityPhase.useNativeRevoluteMotorSolveStartRelativeVelocity,
        velocityPhase.nativeRevoluteMotorSolveStartRelativeVelocity);
  }

  if (motorAdmission.contactCoupledNativeRevoluteMotorVelocityProjectionIsland) {
    PX_PROFILE_ZONE(
        "AVBD.projectContactCoupledNativeRevoluteMotorVelocity", 0);
    projectContactCoupledNativeRevoluteMotorVelocity(
        bodies, numBodies, contacts, numContacts, d6Joints[0], gravity, dt);
  }

  if (motorAdmission.nativeRevoluteMotorGearVelocityProjectionIsland &&
      motorAdmission.nativeRevoluteMotorGearJointIndex < numD6) {
    PX_PROFILE_ZONE("AVBD.projectNativeRevoluteMotorGearVelocity", 0);
    projectNativeRevoluteMotorGearVelocity(
        bodies, numBodies,
        d6Joints[motorAdmission.nativeRevoluteMotorGearJointIndex],
        gearJoints[0], dt);
  }

  if (coupledFixedD6Island) {
    PX_PROFILE_ZONE("AVBD.projectCoupledFixedD6Velocity", 0);
    projectCoupledFixedD6Velocity(bodies, numBodies, d6Joints[0]);
  }

  if (passiveCenteredGearVelocityProjectionIsland) {
    PX_PROFILE_ZONE("AVBD.projectPassiveCenteredGearVelocity", 0);
    projectPassiveCenteredGearVelocity(
        bodies, numBodies, gearJoints[0],
        angularVelAtSolveStart);
  }
  if (velocityPhase.passiveGenericHard1DVelocityProjectionIsland &&
      velocityPhase.passiveGenericHard1DIndex < numD6) {
    PX_PROFILE_ZONE("AVBD.projectPassiveGenericHard1DVelocity", 0);
    projectSinglePassiveGenericHard1DVelocity(
        bodies, numBodies, d6Joints[velocityPhase.passiveGenericHard1DIndex],
        genericLinearVelAtSolveStart, genericAngularVelAtSolveStart);
  }
  finalizeAvbdCoupledJointMomentum(
      bodies, numBodies, contacts, numContacts, d6Joints,
      coupledLinearDriveIsland, coupledLinearPositionDriveIsland,
      coupledLinearPositionDriveFrictionPositionOwnerIsland,
      coupledAngularPositionDriveIsland, coupledSphericalConeIsland,
      velocityPhase.coupledExpectedMomentum,
      velocityPhase.coupledExpectedAngularMomentum,
      velocityPhase.conserveCoupledLinearPositionSupportAxisMomentum,
      velocityPhase.coupledLinearPositionSupportAxis,
      velocityPhase.coupledExpectedLinearPositionSupportAxisAngularMomentum,
      dt);

  finalizeAvbdJointVelocityPolicies(bodies, numBodies, d6Joints, numD6);
}

// Joint post-AL phase boundary.  The mixed-joint coordinator owns phase
// ordering; this owner keeps post-AL recovery, OGC velocity handoff, bending
// damping, and joint velocity finalization together without duplicating their
// policy implementations.

void AvbdJointPostAlPhaseState::run(
    AvbdSolver &solver, physx::PxReal dt, physx::PxReal invDt,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, const physx::PxVec3 &gravity,
    const AvbdJointContactPhaseState &contactPhase,
    const AvbdJointVelocityPhaseState &velocityPhase,
    const AvbdJointMotorAdmissionState &motorAdmission,
    AvbdJointOgcAdmissionState &ogcAdmission,
    bool passiveCenteredGearVelocityProjectionIsland,
    bool coupledFixedD6Island, bool coupledLinearDriveIsland,
    bool coupledLinearPositionDriveIsland,
    bool coupledLinearPositionDriveFrictionPositionOwnerIsland,
    bool coupledAngularPositionDriveIsland,
    bool coupledSphericalConeIsland, bool hasJointConstraints,
    bool skipBodyStaticFriction, bool applyVelocityDamping,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    bool useProvidedSoftExecutionPlan,
    FeatherstoneArticulation *const *articulationForBody,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints,
    const physx::PxArray<physx::PxVec3> &genericLinearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &genericAngularVelAtSolveStart,
    AvbdSolverStats &stats) {
  buildAvbdJointPositionOwnedAngularBodies(
      positionOwnedAngularBodies, numBodies, softBodies, numSoftBodies,
      articulationForBody);

  solver.postAlStages(
      dt, invDt, bodies, numBodies, contacts, numContacts, contactMap, gravity,
      contactPhase.hasBodyStaticContact,
      contactPhase.deformableFastImpactIsland,
      contactPhase.touchingBodyStatic,
      numContacts > 0 ? &contactPhase.linearVelAtSolveStart : nullptr,
      numContacts > 0 ? &contactPhase.angularVelAtSolveStart : nullptr,
      /*allowRigidDeepPoseRecoverySplit=*/false,
      /*allowRigidFiniteMaterialPoseSplit=*/false, softParticles,
      numSoftParticles, softBodies, numSoftBodies, softContacts,
      numSoftContacts, contactPhase.touchesKinematicShell,
      contactPhase.hasKinematicShellContacts
          ? &contactPhase.shellLinearVelAtSolveStart
          : nullptr,
      &positionOwnedAngularBodies, d6Joints, numD6, hasJointConstraints,
      skipBodyStaticFriction, applyVelocityDamping, softParticles,
      numSoftParticles, stats, /*postAlContactWork=*/nullptr,
      useProvidedSoftExecutionPlan ? softExecutionPlan : nullptr);

  const bool mixedPairPlanComplete =
      ogcAdmission.pairContext.isComplete(numSoftContacts);
  AvbdJointOgcVelocityHandoffInput ogcVelocityInput{
      bodies,
      numBodies,
      softParticles,
      numSoftParticles,
      softBodies,
      numSoftBodies,
      softContacts,
      numSoftContacts,
      mixedPairPlanComplete ? ogcAdmission.pairStates : nullptr,
      mixedPairPlanComplete ? ogcAdmission.numPairStates : 0u,
      softExecutionPlan,
      useProvidedSoftExecutionPlan,
      stats};
  solver.applyAvbdJointOgcVelocityHandoff(ogcVelocityInput);

  avbdApplyBendingDamping(softParticles, softBodies, numSoftBodies, dt);
#if PX_CHECKED
  // The special momentum finalizers below may pair a velocity correction
  // with a rigid pose correction.  Their compilation policies require a
  // pure-rigid island; keep that ownership boundary executable so a future
  // policy expansion cannot silently add an unadmitted writer after the
  // mixed OGC terminal epoch.
  if (numSoftContacts > 0u) {
    PX_ASSERT(!coupledLinearDriveIsland);
    PX_ASSERT(!coupledLinearPositionDriveIsland);
    PX_ASSERT(!coupledAngularPositionDriveIsland);
    PX_ASSERT(!coupledSphericalConeIsland);
  }
#endif
  runAvbdJointVelocityFinalizePhase(
      bodies, numBodies, contacts, numContacts, d6Joints, numD6, gearJoints,
      gravity, dt, motorAdmission, velocityPhase,
      passiveCenteredGearVelocityProjectionIsland, coupledFixedD6Island,
      coupledLinearDriveIsland, coupledLinearPositionDriveIsland,
      coupledLinearPositionDriveFrictionPositionOwnerIsland,
      coupledAngularPositionDriveIsland, coupledSphericalConeIsland,
      contactPhase.angularVelAtSolveStart, genericLinearVelAtSolveStart,
      genericAngularVelAtSolveStart);
}

} // namespace Dy
} // namespace physx
