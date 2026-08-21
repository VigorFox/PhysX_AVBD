// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointObjectiveCompilation.h"
#include "avbd/solver/joint/DyAvbdCoupledD6.h"
#include "avbd/solver/joint/DyAvbdJointSupportPolicies.h"
#include "avbd/solver/joint/DyAvbdJointVelocityPolicies.h"

namespace physx {
namespace Dy {

AvbdJointObjectiveCompilationState compileAvbdJointObjectives(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    physx::PxU32 numGear, const physx::PxVec3 &gravity,
    physx::PxU32 numSoftParticles, physx::PxU32 numSoftBodies,
    physx::PxU32 numSoftContacts) {
  AvbdJointObjectiveCompilationState state;
  for (physx::PxU32 i = 0; i < numD6; ++i)
    resetAvbdJointObjectiveProgram(d6Joints[i].objectiveProgram);
  const auto getJointObjectiveSourceRowMask =
      [](const AvbdD6JointConstraint &joint,
         AvbdJointObjectiveKind kind) -> physx::PxU32 {
        const physx::PxU32 linearDriveRows =
            joint.driveFlags & 0x7u;
        const physx::PxU32 angularDriveRows =
            joint.driveFlags & 0x38u;
        switch (kind) {
        case AvbdJointObjectiveKind::CoupledLinearVelocityDrive:
        case AvbdJointObjectiveKind::LinearPositionDrive:
        case AvbdJointObjectiveKind::CoupledLinearPositionDrive:
        case AvbdJointObjectiveKind::OrdinaryD6LinearDrive:
          return linearDriveRows;
        case AvbdJointObjectiveKind::AngularAxisVelocityDrive:
        case AvbdJointObjectiveKind::AngularAxisPositionDrive:
        case AvbdJointObjectiveKind::SlerpVelocityDrive:
        case AvbdJointObjectiveKind::SlerpPositionDrive:
        case AvbdJointObjectiveKind::CoupledAngularPositionDrive:
        case AvbdJointObjectiveKind::OrdinaryD6AngularAxisDrive:
        case AvbdJointObjectiveKind::OrdinaryD6SlerpDrive:
          return angularDriveRows;
        case AvbdJointObjectiveKind::CoupledSpatialTendon:
        case AvbdJointObjectiveKind::GenericHard1D:
        case AvbdJointObjectiveKind::GenericAccelerationDamping1D:
        case AvbdJointObjectiveKind::GenericForceSpring1D:
        case AvbdJointObjectiveKind::GenericRestitution1D:
        case AvbdJointObjectiveKind::ArticulationHardMimic:
        case AvbdJointObjectiveKind::ArticulationCompliantMimic:
        case AvbdJointObjectiveKind::ArticulationFixedTendon:
        case AvbdJointObjectiveKind::ArticulationSpatialTendon:
          return eJOINT_SOURCE_GENERIC_ROW;
        case AvbdJointObjectiveKind::NativeRevoluteMotor:
          return eJOINT_SOURCE_NATIVE_MOTOR;
        case AvbdJointObjectiveKind::CoupledFixedD6:
          return eJOINT_SOURCE_LINEAR_MOTION_X |
                 eJOINT_SOURCE_LINEAR_MOTION_Y |
                 eJOINT_SOURCE_LINEAR_MOTION_Z |
                 eJOINT_SOURCE_ANGULAR_MOTION_X |
                 eJOINT_SOURCE_ANGULAR_MOTION_Y |
                 eJOINT_SOURCE_ANGULAR_MOTION_Z;
        case AvbdJointObjectiveKind::CoupledSphericalCone:
          return eJOINT_SOURCE_LINEAR_MOTION_X |
                 eJOINT_SOURCE_LINEAR_MOTION_Y |
                 eJOINT_SOURCE_LINEAR_MOTION_Z |
                 eJOINT_SOURCE_ANGULAR_CONE;
        case AvbdJointObjectiveKind::NativePassiveReaction: {
          physx::PxU32 sourceRows = 0;
          const physx::PxU32 linearRows[3] = {
              eJOINT_SOURCE_LINEAR_MOTION_X,
              eJOINT_SOURCE_LINEAR_MOTION_Y,
              eJOINT_SOURCE_LINEAR_MOTION_Z};
          const physx::PxU32 angularRows[3] = {
              eJOINT_SOURCE_ANGULAR_MOTION_X,
              eJOINT_SOURCE_ANGULAR_MOTION_Y,
              eJOINT_SOURCE_ANGULAR_MOTION_Z};
          for (physx::PxU32 axis = 0; axis < 3; ++axis) {
            if (joint.getLinearMotion(axis) != 2)
              sourceRows |= linearRows[axis];
            if (joint.getAngularMotion(axis) != 2)
              sourceRows |= angularRows[axis];
          }
          return sourceRows;
        }
        case AvbdJointObjectiveKind::OrdinaryD6Position: {
          physx::PxU32 sourceRows = 0;
          const physx::PxU32 linearRows[3] = {
              eJOINT_SOURCE_LINEAR_MOTION_X,
              eJOINT_SOURCE_LINEAR_MOTION_Y,
              eJOINT_SOURCE_LINEAR_MOTION_Z};
          const physx::PxU32 angularRows[3] = {
              eJOINT_SOURCE_ANGULAR_MOTION_X,
              eJOINT_SOURCE_ANGULAR_MOTION_Y,
              eJOINT_SOURCE_ANGULAR_MOTION_Z};
          const physx::PxU32 ellipticalConeFlags =
              AvbdD6JointConstraint::
                  eD6_LEGACY_CONE_LIMIT_ACTIVE |
              AvbdD6JointConstraint::
                  eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE;
          const bool ellipticalCone =
              (joint.sourceFlags & ellipticalConeFlags) != 0;
          for (physx::PxU32 axis = 0; axis < 3; ++axis) {
            if (joint.getLinearMotion(axis) != 2)
              sourceRows |= linearRows[axis];
            if (joint.getAngularMotion(axis) != 2 &&
                !(ellipticalCone && axis >= 1))
              sourceRows |= angularRows[axis];
          }
          if (joint.coneAngleLimit > 0.0f)
            sourceRows |= eJOINT_SOURCE_ANGULAR_CONE;
          return sourceRows;
        }
        case AvbdJointObjectiveKind::None:
        case AvbdJointObjectiveKind::Invalid:
          break;
        }
        return 0;
      };
  const auto countJointObjectiveSourceRows =
      [](physx::PxU32 sourceRowMask) -> physx::PxU16 {
        physx::PxU16 count = 0;
        while (sourceRowMask != 0) {
          count += physx::PxU16(sourceRowMask & 1u);
          sourceRowMask >>= 1;
        }
        return count;
      };

  const auto compileSingleJointObjective =
      [&](bool candidate, AvbdVelocityObjectiveOwner owner,
          AvbdJointObjectiveKind kind) {
        if (!candidate || numD6 == 0)
          return;
        const physx::PxU32 sourceRowMask =
            getJointObjectiveSourceRowMask(d6Joints[0], kind);
        const physx::PxU16 objectiveRowCount =
            kind == AvbdJointObjectiveKind::CoupledFixedD6
                ? 6u
                : (kind ==
                           AvbdJointObjectiveKind::
                               CoupledSphericalCone
                       ? 4u
                       : (kind ==
                                  AvbdJointObjectiveKind::
                                      NativePassiveReaction
                              ? countJointObjectiveSourceRows(
                                    sourceRowMask)
                              : 1u));
        assignAvbdJointObjective(
            d6Joints[0].objectiveProgram, owner, kind,
            objectiveRowCount,
            sourceRowMask,
            d6Joints[0].cacheKey);
      };

  const bool linearPositionDriveCandidate =
      isLinearPositionDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      linearPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::LinearPositionDrive);
  const bool coupledLinearPositionDriveCandidate =
      isCoupledLinearPositionDriveIslandSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          gravity, numGear, numSoftParticles, numSoftBodies,
          numSoftContacts);
  compileSingleJointObjective(
      coupledLinearPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledLinearPositionDrive);
  const bool angularAxisVelocityDriveCandidate =
      isAngularAxisVelocityDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      angularAxisVelocityDriveCandidate,
      AvbdVelocityObjectiveOwner::JointFinalize,
      AvbdJointObjectiveKind::AngularAxisVelocityDrive);
  const bool angularAxisPositionDriveCandidate =
      isAngularAxisPositionDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      angularAxisPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::AngularAxisPositionDrive);
  const bool slerpVelocityDriveCandidate =
      isSlerpVelocityDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      slerpVelocityDriveCandidate,
      AvbdVelocityObjectiveOwner::JointFinalize,
      AvbdJointObjectiveKind::SlerpVelocityDrive);
  const bool slerpPositionDriveCandidate =
      isSlerpPositionDriveIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      slerpPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::SlerpPositionDrive);
  const bool coupledAngularPositionDriveCandidate =
      isCoupledAngularPositionDriveIslandSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledAngularPositionDriveCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledAngularPositionDrive);
  const bool coupledLinearDriveCandidate =
      isCoupledLinearDriveIslandSupported(
          bodies, numBodies, contacts, numContacts, d6Joints, numD6,
          numGear, numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledLinearDriveCandidate,
      AvbdVelocityObjectiveOwner::JointFinalize,
      AvbdJointObjectiveKind::CoupledLinearVelocityDrive);
  const bool coupledFixedD6Candidate =
      isCoupledFixedD6IslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledFixedD6Candidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledFixedD6);
  const bool coupledSphericalConeCandidate =
      isCoupledSphericalConeIslandSupported(
          bodies, numBodies, numContacts, d6Joints, numD6, numGear,
          numSoftParticles, numSoftBodies, numSoftContacts);
  compileSingleJointObjective(
      coupledSphericalConeCandidate,
      AvbdVelocityObjectiveOwner::PositionAL,
      AvbdJointObjectiveKind::CoupledSphericalCone);
  const auto isNativePassiveReactionSource =
      [](const AvbdD6JointConstraint &joint) {
        return joint.header.type ==
                   AvbdConstraintType::eJOINT_FIXED ||
               (joint.header.type ==
                    AvbdConstraintType::eJOINT_PRISMATIC &&
                joint.linearMotion == 0x02u &&
                joint.angularMotion == 0u) ||
               (joint.header.type ==
                    AvbdConstraintType::eJOINT_REVOLUTE &&
                joint.linearMotion == 0u &&
                joint.angularMotion == 0x02u &&
                joint.motorEnabled == 0u);
      };
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    if (!isNativePassiveReactionSource(joint) ||
        hasAvbdJointObjective(
            joint.objectiveProgram,
            AvbdJointObjectiveKind::CoupledFixedD6))
      continue;
    const physx::PxU32 sourceRowMask =
        getJointObjectiveSourceRowMask(
            joint,
            AvbdJointObjectiveKind::NativePassiveReaction);
    assignAvbdJointObjective(
        joint.objectiveProgram,
        AvbdVelocityObjectiveOwner::PositionAL,
        AvbdJointObjectiveKind::NativePassiveReaction,
        countJointObjectiveSourceRows(sourceRowMask),
        sourceRowMask, joint.cacheKey);
  }

  state.coupledLinearPositionDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledLinearPositionDrive);
  state.coupledLinearPositionDriveFrictionPositionOwnerIsland =
      state.coupledLinearPositionDriveIsland &&
      areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts, gravity);
  state.slerpVelocityDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::SlerpVelocityDrive);
  state.coupledAngularPositionDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledAngularPositionDrive);
  state.coupledLinearDriveIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledLinearVelocityDrive);
  state.coupledFixedD6Island =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledFixedD6);
  state.coupledSphericalConeIsland =
      numD6 > 0 &&
      hasAvbdJointObjective(
          d6Joints[0].objectiveProgram,
          AvbdJointObjectiveKind::CoupledSphericalCone);
  const bool coupledSpatialTendonCandidate = findCoupledSpatialTendonRows(
      bodies, numBodies, numContacts, d6Joints, numD6, numGear,
      numSoftParticles, numSoftBodies, numSoftContacts,
      state.coupledSpatialTendonRowIndices);
  if (coupledSpatialTendonCandidate) {
    bool supported =
        state.coupledSpatialTendonRowIndices.size() <= PX_MAX_U16;
    physx::PxU64 objectiveKey = ~physx::PxU64(0);
    for (physx::PxU32 row = 0;
         row < state.coupledSpatialTendonRowIndices.size(); ++row) {
      objectiveKey = physx::PxMin(
          objectiveKey,
          d6Joints[state.coupledSpatialTendonRowIndices[row]].cacheKey);
    }
    for (physx::PxU32 row = 0;
         row < state.coupledSpatialTendonRowIndices.size() && supported;
         ++row) {
      supported = canAssignAvbdJointObjective(
          d6Joints[state.coupledSpatialTendonRowIndices[row]]
              .objectiveProgram,
          AvbdVelocityObjectiveOwner::PositionAL,
          AvbdJointObjectiveKind::CoupledSpatialTendon,
          physx::PxU16(state.coupledSpatialTendonRowIndices.size()),
          eJOINT_SOURCE_GENERIC_ROW,
          objectiveKey);
    }
    if (supported) {
      for (physx::PxU32 row = 0;
           row < state.coupledSpatialTendonRowIndices.size(); ++row) {
        assignAvbdJointObjective(
            d6Joints[state.coupledSpatialTendonRowIndices[row]]
                .objectiveProgram,
            AvbdVelocityObjectiveOwner::PositionAL,
            AvbdJointObjectiveKind::CoupledSpatialTendon,
            physx::PxU16(state.coupledSpatialTendonRowIndices.size()),
            eJOINT_SOURCE_GENERIC_ROW,
            objectiveKey);
      }
    } else {
      for (physx::PxU32 row = 0;
           row < state.coupledSpatialTendonRowIndices.size(); ++row) {
        invalidateAvbdJointObjective(
            d6Joints[state.coupledSpatialTendonRowIndices[row]]
                .objectiveProgram);
      }
    }
  }
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    if (hasAvbdJointObjective(
            joint.objectiveProgram,
            AvbdJointObjectiveKind::CoupledSpatialTendon))
      continue;

    const physx::PxU32 sourceFlags = joint.sourceFlags;
    AvbdJointObjectiveKind kind = AvbdJointObjectiveKind::None;
    AvbdVelocityObjectiveOwner owner =
        AvbdVelocityObjectiveOwner::PositionAL;
    if ((sourceFlags &
         AvbdD6JointConstraint::eARTICULATION_MIMIC_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationHardMimic;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eARTICULATION_COMPLIANT_MIMIC_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationCompliantMimic;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eARTICULATION_FIXED_TENDON_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationFixedTendon;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eARTICULATION_SPATIAL_TENDON_ROW) != 0)
      kind = AvbdJointObjectiveKind::ArticulationSpatialTendon;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eGENERIC_ACCELERATION_DAMPING_1D_ROW) != 0)
      kind =
          AvbdJointObjectiveKind::GenericAccelerationDamping1D;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eGENERIC_FORCE_SPRING_1D_ROW) != 0)
      kind = AvbdJointObjectiveKind::GenericForceSpring1D;
    else if ((sourceFlags &
              AvbdD6JointConstraint::
                  eGENERIC_RESTITUTION_1D_ROW) != 0) {
      kind = AvbdJointObjectiveKind::GenericRestitution1D;
      owner = AvbdVelocityObjectiveOwner::JointFinalize;
    } else if ((sourceFlags &
                AvbdD6JointConstraint::
                    eGENERIC_HARD_1D_ROW) != 0)
      kind = AvbdJointObjectiveKind::GenericHard1D;

    if (kind == AvbdJointObjectiveKind::None)
      continue;
    assignAvbdJointObjective(
        joint.objectiveProgram, owner, kind, 1u,
        eJOINT_SOURCE_GENERIC_ROW, joint.cacheKey);
  }
  const physx::PxU32 genericSourceFlags =
      AvbdD6JointConstraint::eGENERIC_HARD_1D_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_MIMIC_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_FIXED_TENDON_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_SPATIAL_TENDON_ROW |
      AvbdD6JointConstraint::
          eGENERIC_ACCELERATION_DAMPING_1D_ROW |
      AvbdD6JointConstraint::
          eGENERIC_FORCE_SPRING_1D_ROW |
      AvbdD6JointConstraint::
          eGENERIC_RESTITUTION_1D_ROW |
      AvbdD6JointConstraint::
          eARTICULATION_COMPLIANT_MIMIC_ROW;
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    AvbdCompiledJointObjectiveProgram &program =
        joint.objectiveProgram;
    if (program.invalid)
      continue;

    physx::PxU32 compiledSourceRows = 0;
    for (physx::PxU32 entryIndex = 0;
         entryIndex < program.entryCount; ++entryIndex)
      compiledSourceRows |=
          program.entries[entryIndex].sourceRowMask;

    const physx::PxU32 positionSourceRows =
        getJointObjectiveSourceRowMask(
            joint,
            AvbdJointObjectiveKind::OrdinaryD6Position);
    const physx::PxU32 remainingPositionSourceRows =
        positionSourceRows & ~compiledSourceRows;
    if (remainingPositionSourceRows != 0) {
      assignAvbdJointObjective(
          program, AvbdVelocityObjectiveOwner::PositionAL,
          AvbdJointObjectiveKind::OrdinaryD6Position,
          countJointObjectiveSourceRows(
              remainingPositionSourceRows),
          remainingPositionSourceRows, joint.cacheKey);
      if (program.invalid)
        continue;
      compiledSourceRows |= remainingPositionSourceRows;
    }

    const physx::PxU32 remainingLinearDriveSourceRows =
        (joint.driveFlags & 0x7u) & ~compiledSourceRows;
    if (remainingLinearDriveSourceRows != 0) {
      assignAvbdJointObjective(
          program, AvbdVelocityObjectiveOwner::PositionAL,
          AvbdJointObjectiveKind::OrdinaryD6LinearDrive,
          countJointObjectiveSourceRows(
              remainingLinearDriveSourceRows),
          remainingLinearDriveSourceRows, joint.cacheKey);
      if (program.invalid)
        continue;
      compiledSourceRows |= remainingLinearDriveSourceRows;
    }

    const physx::PxU32 remainingAngularDriveSourceRows =
        (joint.driveFlags & 0x38u) & ~compiledSourceRows;
    if (remainingAngularDriveSourceRows != 0) {
      const bool slerpDrive =
          (joint.sourceFlags &
           AvbdD6JointConstraint::eD6_SLERP_DRIVE) != 0;
      assignAvbdJointObjective(
          program, AvbdVelocityObjectiveOwner::PositionAL,
          slerpDrive
              ? AvbdJointObjectiveKind::OrdinaryD6SlerpDrive
              : AvbdJointObjectiveKind::OrdinaryD6AngularAxisDrive,
          countJointObjectiveSourceRows(
              remainingAngularDriveSourceRows),
          remainingAngularDriveSourceRows, joint.cacheKey);
    }
  }
  for (physx::PxU32 jointIndex = 0; jointIndex < numD6;
       ++jointIndex) {
    AvbdD6JointConstraint &joint = d6Joints[jointIndex];
    AvbdCompiledJointObjectiveProgram &program = joint.objectiveProgram;
    if (program.invalid)
      continue;

    physx::PxU32 compiledSourceRows = 0;
    for (physx::PxU32 entryIndex = 0;
         entryIndex < program.entryCount; ++entryIndex)
      compiledSourceRows |= program.entries[entryIndex].sourceRowMask;

    const physx::PxU32 positionSourceRows =
        getJointObjectiveSourceRowMask(
            joint, AvbdJointObjectiveKind::OrdinaryD6Position);
    physx::PxU32 allSourceRows =
        positionSourceRows | (joint.driveFlags & 0x3fu);
    if ((joint.sourceFlags & genericSourceFlags) != 0)
      allSourceRows |= eJOINT_SOURCE_GENERIC_ROW;
    // Native revolute motor rows are compiled immediately after this generic
    // objective pass by buildAvbdJointMotorAdmission().  Reserving the same
    // source bit as a legacy row here makes that unique owner conflict with
    // itself and invalidates the complete joint objective program.
    program.legacySourceRowMask = allSourceRows & ~compiledSourceRows;
  }

  state.coupledSpatialTendonIsland =
      coupledSpatialTendonCandidate;
  for (physx::PxU32 row = 0;
       row < state.coupledSpatialTendonRowIndices.size() &&
       state.coupledSpatialTendonIsland;
       ++row) {
    state.coupledSpatialTendonIsland = hasAvbdJointObjective(
        d6Joints[state.coupledSpatialTendonRowIndices[row]]
            .objectiveProgram,
        AvbdJointObjectiveKind::CoupledSpatialTendon);
  }

  return state;
}

} // namespace Dy
} // namespace physx
