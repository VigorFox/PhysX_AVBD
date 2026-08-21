// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/DyAvbdSolver.h"

namespace physx
{
namespace Dy
{

void AvbdSolver::updateSoftDual(
    AvbdSoftParticle *softParticles, PxU32 numSoftParticles,
    AvbdSolverBody *rigidBodies, PxU32 numRigidBodies,
    AvbdSoftBody *softBodies, PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, PxU32 numSoftContacts,
    PxReal beta)
{
  for (PxU32 sbi = 0; sbi < numSoftBodies; ++sbi)
  {
    AvbdSoftBody &sb = softBodies[sbi];
    for (PxU32 oi = 0;
         oi < sb.runtime.compiledObjectives.size(); ++oi)
    {
      const AvbdCompiledSoftObjective &objective =
          sb.runtime.compiledObjectives[oi];
      switch (objective.owner)
      {
      case AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eARTICULATION_ATTACHMENT_POSITION_AL:
        avbdUpdateAttachmentDual(
            sb.runtime.attachments[objective.runtimeStateIndex],
            objective.point, softParticles, rigidBodies, beta);
        break;
      case AvbdSoftObjectiveOwner::
          eSOFT_PAIR_ATTACHMENT_POSITION_AL:
        avbdUpdateSoftPairAttachmentDual(
            sb.runtime.attachments[objective.runtimeStateIndex],
            objective.point, objective.targetPoint,
            softParticles, beta);
        break;
      case AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eDEFORMABLE_KINEMATIC_POSITION_AL:
      case AvbdSoftObjectiveOwner::
          eKINEMATIC_ATTACHMENT_POSITION_AL:
        avbdUpdatePinDual(
            sb.runtime.pins[objective.runtimeStateIndex],
            objective.point, softParticles, beta);
        break;
      default:
        PX_ASSERT(false);
        break;
      }
    }
  }
  for (PxU32 sci = 0; sci < numSoftContacts; ++sci) {
    AvbdSoftContact &sc = softContacts[sci];
    const AvbdSoftContactGeometry& geometry = sc.geometry;
    AvbdSoftContactAugmentedState& state = sc.state;
    if (avbdIsSoftContactQueryFullyKinematic(
            geometry, softParticles, numSoftParticles) &&
        geometry.hasRigidBodyTarget() &&
        geometry.targetIndex < numRigidBodies) {
      avbdUpdateKinematicShellContactDual(
          geometry, state, rigidBodies[geometry.targetIndex],
          beta, mConfig.avbdPenaltyMax);
    } else if (geometry.hasRigidBodyTarget() &&
               geometry.targetIndex < numRigidBodies) {
      avbdUpdateSoftContactDualAtSurfacePoint(
          geometry, state, softParticles,
          avbdGetRigidContactSurfacePoint(
              geometry, rigidBodies[geometry.targetIndex]),
          beta);
    } else {
      avbdUpdateSoftContactDual(
          geometry, state, softParticles, beta);
    }
  }
}

} // namespace Dy
} // namespace physx
