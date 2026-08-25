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
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

// Rigid-body velocity reconstruction after the post-AL position stages.
// Contact policy remains in the caller; this phase only converts final pose
// deltas into body velocities.

void reconstructPostAlBodyVelocities(
    physx::PxReal dt, physx::PxReal invDt, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const physx::PxVec3 &gravity,
    bool hasKinematicShellContacts,
    const physx::PxArray<bool> &touchesKinematicShell,
    const physx::PxArray<physx::PxVec3> *shellLinearVelAtSolveStart,
    AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
    const physx::PxArray<bool> *positionOwnedAngularBodies,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    bool applyVelocityDamping,
    const physx::PxArray<bool> &splitRigidDeepPoseRecovery,
    const physx::PxArray<bool> &splitRigidFiniteMaterialPose,
    const physx::PxArray<physx::PxVec3> &postBlockPos,
    const physx::PxArray<physx::PxQuat> &postBlockRot,
    const physx::PxArray<physx::PxVec3> &postDepenPos,
    const physx::PxArray<physx::PxQuat> &postDepenRot,
    AvbdPostAlVelocityState &velocityState,
    bool terminalCurrentPoseEpochApplied,
    const physx::PxArray<physx::PxVec3> &terminalVelocityBasePos,
    const physx::PxArray<physx::PxQuat> &terminalVelocityBaseRot,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal velocityDamping, physx::PxReal angularDamping) {
  (void)gravity;
  const physx::PxArray<physx::PxU32> &physicalContactTangentOwnerIndex =
      velocityState.physicalContactTangentOwnerIndex;
  const physx::PxArray<bool> &fastNormalImpactByBody =
      velocityState.fastNormalImpactByBody;
  physx::PxArray<physx::PxReal> &linearPoseVelocityGain =
      velocityState.linearPoseVelocityGain;
  physx::PxArray<physx::PxReal> &angularPoseVelocityGain =
      velocityState.angularPoseVelocityGain;
  static const physx::PxReal kShellFastImpactSpeed =
      AvbdConstants::AVBD_SHELL_FAST_IMPACT_SPEED;

    PX_PROFILE_ZONE("AVBD.updateVelocities", 0);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f) {
        linearPoseVelocityGain[i] = 1.0f;
        angularPoseVelocityGain[i] = 1.0f;
        bodies[i].prevLinearVelocity = bodies[i].linearVelocity;
        const physx::PxU32 physicalContactTangentMaterialOwnerIndex =
            physicalContactTangentOwnerIndex[i];
        const bool physicalContactTangentMaterialOwner =
            physicalContactTangentMaterialOwnerIndex != PX_MAX_U32;

        const physx::PxVec3 blockPositionForVelocity =
            (splitRigidDeepPoseRecovery[i] ||
             splitRigidFiniteMaterialPose[i])
                ? bodies[i].inertialPosition
                : postBlockPos[i];
        const physx::PxVec3 vFromBlock =
            (blockPositionForVelocity - bodies[i].prevPosition) * invDt;
        const physx::PxVec3 frictionPositionForVelocity =
            terminalCurrentPoseEpochApplied &&
                    terminalVelocityBasePos.size() == numBodies
                ? terminalVelocityBasePos[i]
                : bodies[i].position;
        const physx::PxVec3 vFromFriction =
            (frictionPositionForVelocity - postDepenPos[i]) * invDt;
        const physx::PxVec3 vFromPose = vFromBlock + vFromFriction;
        const bool fastNormalImpact =
            i < fastNormalImpactByBody.size() && fastNormalImpactByBody[i];
        if (fastNormalImpact) {
          bodies[i].linearVelocity =
              (*linearVelAtSolveStart)[i] * 0.85f + vFromPose * 0.15f;
          linearPoseVelocityGain[i] = 0.15f;
        } else if (i < touchesKinematicShell.size() && touchesKinematicShell[i] &&
                   shellLinearVelAtSolveStart &&
                   shellLinearVelAtSolveStart->size() == numBodies) {
          bool shellFast = false;
          for (physx::PxU32 sci = 0; sci < numShellContacts; ++sci) {
            const AvbdSoftContactGeometry &geometry =
                shellContacts[sci].geometry;
            if (!geometry.hasRigidBodyTarget() ||
                geometry.targetIndex != i)
              continue;
            const physx::PxReal approach =
                -(*shellLinearVelAtSolveStart)[i].dot(geometry.normal);
            if (approach > kShellFastImpactSpeed) {
              shellFast = true;
              break;
            }
          }
          if (shellFast)
            bodies[i].linearVelocity =
                (*shellLinearVelAtSolveStart)[i] * 0.85f + vFromPose * 0.15f;
          else
            bodies[i].linearVelocity = vFromPose;
          linearPoseVelocityGain[i] = shellFast ? 0.15f : 1.0f;
        } else {
          bodies[i].linearVelocity = vFromPose;
        }

        if (applyVelocityDamping &&
            !physicalContactTangentMaterialOwner) {
          bodies[i].linearVelocity *= velocityDamping;
          linearPoseVelocityGain[i] *= velocityDamping;
        }

        const bool unconstrainedAngularMotion =
            numContacts == 0 && !hasKinematicShellContacts &&
            (!d6Joints || numD6 == 0) &&
            !(positionOwnedAngularBodies &&
              positionOwnedAngularBodies->size() == numBodies &&
              (*positionOwnedAngularBodies)[i]);
        bool physicalSlerpPositionDrive = false;
        if (d6Joints) {
          for (physx::PxU32 j = 0; j < numD6; ++j) {
            const AvbdD6JointConstraint &joint = d6Joints[j];
            if (joint.header.bodyIndexA != i &&
                joint.header.bodyIndexB != i)
              continue;
            if (hasAvbdJointObjective(
                    joint.objectiveProgram,
                    AvbdJointObjectiveKind::SlerpPositionDrive))
              physicalSlerpPositionDrive = true;
            if (physicalSlerpPositionDrive)
              break;
          }
        }
        if (!unconstrainedAngularMotion) {
          const physx::PxQuat blockRotationForVelocity =
              (splitRigidDeepPoseRecovery[i] ||
               splitRigidFiniteMaterialPose[i])
                  ? bodies[i].inertialRotation
                  : postBlockRot[i];
          physx::PxQuat deltaQBlock =
              blockRotationForVelocity *
              bodies[i].prevRotation.getConjugate();
          if (deltaQBlock.w < 0.0f)
            deltaQBlock = -deltaQBlock;
          const physx::PxVec3 wBlock =
              physx::PxVec3(deltaQBlock.x, deltaQBlock.y, deltaQBlock.z) *
              (2.0f * invDt);
          const physx::PxQuat frictionRotationForVelocity =
              terminalCurrentPoseEpochApplied &&
                      terminalVelocityBaseRot.size() == numBodies
                  ? terminalVelocityBaseRot[i]
                  : bodies[i].rotation;
          physx::PxQuat deltaQFr =
              frictionRotationForVelocity * postDepenRot[i].getConjugate();
          if (deltaQFr.w < 0.0f)
            deltaQFr = -deltaQFr;
          const physx::PxVec3 wFr =
              physx::PxVec3(deltaQFr.x, deltaQFr.y, deltaQFr.z) *
              (2.0f * invDt);
          bodies[i].angularVelocity = wBlock + wFr;
          // Explicit position/velocity targets already own their damping and
          // material semantics. Applying solver-wide stabilization decay
          // again turns a constant-speed target into a frame-rate-dependent
          // lag and changes a passive manifold's inertial baseline.
          if (!physicalSlerpPositionDrive &&
              !physicalContactTangentMaterialOwner) {
            bodies[i].angularVelocity *= angularDamping;
            angularPoseVelocityGain[i] *= angularDamping;
          }
        }

        const AvbdCompiledVelocityObjective* completeManifoldObjective =
            physicalContactTangentMaterialOwnerIndex != PX_MAX_U32
                ? findAvbdCompleteManifoldObjective(
                      contacts[physicalContactTangentMaterialOwnerIndex]
                          .objectiveProgram)
                : NULL;
        if (completeManifoldObjective &&
            completeManifoldObjective->reconstruction ==
                AvbdVelocityObjectiveReconstruction::SolveStartInertial &&
            linearVelAtSolveStart && angularVelAtSolveStart &&
            linearVelAtSolveStart->size() == numBodies &&
            angularVelAtSolveStart->size() == numBodies) {
          // The position solve owns geometry, but its pose delta and AL
          // multipliers are not material impulses. Reconstruct this strict
          // manifold from solve-start inertial velocity. Its coupled
          // post-reconstruction owner rebuilds both the nonnegative normal
          // response and the tangent target from that single baseline.
          physx::PxVec3 baselineLinear =
              (*linearVelAtSolveStart)[i];
          physx::PxVec3 baselineAngular =
              (*angularVelAtSolveStart)[i];
          bodies[i].projectLockedLinearVector(baselineLinear);
          bodies[i].projectLockedAngularVector(baselineAngular);
          bodies[i].linearVelocity = baselineLinear;
          bodies[i].angularVelocity = baselineAngular;
          linearPoseVelocityGain[i] = 0.0f;
          angularPoseVelocityGain[i] = 0.0f;
          bodies[i].projectLockedVelocities();
        } else if (
            physicalContactTangentMaterialOwnerIndex != PX_MAX_U32 &&
            hasVelocityTangentTargetNormalSpan(
                contacts[physicalContactTangentMaterialOwnerIndex]) &&
            linearVelAtSolveStart && angularVelAtSolveStart &&
            linearVelAtSolveStart->size() == numBodies &&
            angularVelAtSolveStart->size() == numBodies) {
          const AvbdContactConstraint &targetContact =
              contacts[physicalContactTangentMaterialOwnerIndex];
          const bool dynamicIsA =
              targetContact.header.bodyIndexA == i;
          const physx::PxVec3 dynamicNormal =
              targetContact.contactNormal *
              (dynamicIsA ? 1.0f : -1.0f);
          const physx::PxVec3 localPoint =
              dynamicIsA ? targetContact.contactPointA
                         : targetContact.contactPointB;
          const physx::PxVec3 contactArm =
              bodies[i].prevRotation.rotate(localPoint);
          const physx::PxVec3 angularJacobian =
              contactArm.cross(dynamicNormal);
          const physx::PxReal linearScale =
              dynamicIsA ? targetContact.invMassScaleA
                         : targetContact.invMassScaleB;
          const physx::PxReal angularScale =
              dynamicIsA ? targetContact.invInertiaScaleA
                         : targetContact.invInertiaScaleB;
          const physx::PxVec3 normalLinearResponse =
              dynamicNormal * (bodies[i].invMass * linearScale);
          const physx::PxVec3 normalAngularResponse =
              bodies[i].invInertiaWorld *
              (angularJacobian * angularScale);
          const physx::PxReal normalResponse =
              dynamicNormal.dot(normalLinearResponse) +
              angularJacobian.dot(normalAngularResponse);
          if (normalResponse > 1.0e-12f) {
            physx::PxVec3 baselineLinear =
                (*linearVelAtSolveStart)[i];
            physx::PxVec3 baselineAngular =
                (*angularVelAtSolveStart)[i];
            bodies[i].projectLockedLinearVector(baselineLinear);
            bodies[i].projectLockedAngularVector(baselineAngular);
            const physx::PxVec3 poseDeltaLinear =
                bodies[i].linearVelocity - baselineLinear;
            const physx::PxVec3 poseDeltaAngular =
                bodies[i].angularVelocity - baselineAngular;
            const physx::PxReal normalImpulse = physx::PxMax(
                0.0f,
                (dynamicNormal.dot(poseDeltaLinear) +
                 angularJacobian.dot(poseDeltaAngular)) /
                    normalResponse);
            bodies[i].linearVelocity =
                baselineLinear + normalLinearResponse * normalImpulse;
            bodies[i].angularVelocity =
                baselineAngular + normalAngularResponse * normalImpulse;
            linearPoseVelocityGain[i] = 0.0f;
            angularPoseVelocityGain[i] = 0.0f;
            bodies[i].projectLockedVelocities();
          }
        }

        if (bodies[i].linearDamping > 0.0f) {
          physx::PxReal linDecay =
              1.0f / (1.0f + bodies[i].linearDamping * dt);
          bodies[i].linearVelocity *= linDecay;
          linearPoseVelocityGain[i] *= linDecay;
        }
        if (bodies[i].angularDampingBody > 0.0f) {
          physx::PxReal angDecay =
              1.0f / (1.0f + bodies[i].angularDampingBody * dt);
          bodies[i].angularVelocity *= angDecay;
          angularPoseVelocityGain[i] *= angDecay;
        }

        physx::PxReal linVelSq =
            bodies[i].linearVelocity.magnitudeSquared();
        if (linVelSq > bodies[i].maxLinearVelocitySq &&
            bodies[i].maxLinearVelocitySq > 0.0f) {
          const physx::PxReal capScale =
              physx::PxSqrt(bodies[i].maxLinearVelocitySq / linVelSq);
          bodies[i].linearVelocity *= capScale;
          linearPoseVelocityGain[i] *= capScale;
        }
        physx::PxReal angVelSq =
            bodies[i].angularVelocity.magnitudeSquared();
        if (angVelSq > bodies[i].maxAngularVelocitySq &&
            bodies[i].maxAngularVelocitySq > 0.0f) {
          const physx::PxReal capScale =
              physx::PxSqrt(bodies[i].maxAngularVelocitySq / angVelSq);
          bodies[i].angularVelocity *= capScale;
          angularPoseVelocityGain[i] *= capScale;
        }
      }
    }

}

} // namespace Dy
} // namespace physx
