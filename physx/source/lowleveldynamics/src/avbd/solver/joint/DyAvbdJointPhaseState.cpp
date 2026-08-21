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

#include "avbd/solver/joint/DyAvbdJointPhaseState.h"
#include "avbd/ogc/DyAvbdOgcPlanValidation.h"
#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "avbd/solver/joint/DyAvbdJointDriveMath.h"

namespace physx {
namespace Dy {

AvbdJointPhaseAdmission buildAvbdJointPhaseAdmission(
    bool solverInitialized, physx::PxU32 numBodies,
    AvbdSolverBody *bodies, AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies, AvbdSoftContact *softContacts,
    physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan) {
  AvbdJointPhaseAdmission admission = {false, false, false, false};
  admission.hasCompleteSoftSelection =
      softParticles && numSoftParticles > 0 && softBodies &&
      numSoftBodies > 0 && (numSoftContacts == 0 || softContacts);
  if (!solverInitialized ||
      (numBodies == 0 && !admission.hasCompleteSoftSelection))
    return admission;

  admission.useProvidedSoftExecutionPlan =
      admission.hasCompleteSoftSelection && softExecutionPlan &&
      avbdValidateSoftIslandExecutionPlan(
          *softExecutionPlan, softBodies, numSoftBodies, numSoftParticles,
          numSoftContacts);
  admission.useProvidedRigidTargetContactPlan =
      admission.useProvidedSoftExecutionPlan && numBodies > 0 &&
      avbdCanUseRigidTargetContactPlan(
          *softExecutionPlan, bodies, numBodies, softContacts,
          numSoftContacts);
  admission.hasPreparedSoftPrediction =
      admission.useProvidedSoftExecutionPlan &&
      softExecutionPlan->softPredictionPrepared;
  return admission;
}

void buildAvbdJointContactPhaseState(
    AvbdJointContactPhaseState &state, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numD6, physx::PxU32 numGear,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    bool captureAngularVelocityForPassiveGear) {
  state.hasKinematicShellContacts = false;
  state.hasBodyStaticContact = false;
  state.hasDeformableAnchorContact = false;
  state.allBodyVsStatic = numContacts > 0;
  state.deformableFastImpactIsland = false;

  state.touchesKinematicShell.resize(numBodies);
  state.touchingBodyStatic.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    state.touchesKinematicShell[i] = false;
    state.touchingBodyStatic[i] = false;
  }

  if (softContacts && numSoftContacts > 0 && softParticles) {
    for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
      const AvbdSoftContactGeometry &geometry =
          softContacts[sci].geometry;
      const physx::PxU32 bodyIndex = geometry.targetIndex;
      if (geometry.hasRigidBodyTarget() && bodyIndex < numBodies &&
          avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles)) {
        state.touchesKinematicShell[bodyIndex] = true;
        state.hasKinematicShellContacts = true;
      }
    }
  }

  if (state.hasKinematicShellContacts) {
    state.shellLinearVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      state.shellLinearVelAtSolveStart[i] = bodies[i].linearVelocity;
  }

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bodyA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[c].header.bodyIndexB;
    if (isBodyVsStaticContact(bodyA, bodyB, numBodies))
      state.hasBodyStaticContact = true;
    else
      state.allBodyVsStatic = false;
    if (hasDeformableStaticAnchor(contacts[c]))
      state.hasDeformableAnchorContact = true;
    if (isBodyVsStaticContact(bodyA, bodyB, numBodies)) {
      if (bodyA < numBodies)
        state.touchingBodyStatic[bodyA] = true;
      if (bodyB < numBodies)
        state.touchingBodyStatic[bodyB] = true;
    }
  }
  state.deformableFastImpactIsland =
      state.allBodyVsStatic && state.hasDeformableAnchorContact &&
      numD6 == 0 && numGear == 0;

  if (numContacts > 0) {
    state.linearVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      state.linearVelAtSolveStart[i] = bodies[i].linearVelocity;
  }
  if (numContacts > 0 || captureAngularVelocityForPassiveGear) {
    state.angularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      state.angularVelAtSolveStart[i] = bodies[i].angularVelocity;
  }
}

void initializeAvbdJointPositionPhaseState(
    AvbdJointPositionPhaseState &state, const AvbdSolverConfig &config,
    bool slerpVelocityDriveIsland, bool coupledFixedD6Island,
    bool coupledSphericalConeIsland, bool coupledSpatialTendonIsland,
    bool hasDynamicSoftRigidContact, AvbdSolverBody *bodies,
    physx::PxU32 numBodies) {
  state.useChebyshev =
      !slerpVelocityDriveIsland && !coupledFixedD6Island &&
      !coupledSphericalConeIsland && !coupledSpatialTendonIsland &&
      !hasDynamicSoftRigidContact && config.chebyshevRho > 0.0f &&
      config.chebyshevRho < 1.0f;
  state.chebyOmega = 1.0f;
  if (!state.useChebyshev)
    return;

  state.chebyPrevPos.resize(numBodies);
  state.chebyPrevPrevPos.resize(numBodies);
  state.chebyPrevRot.resize(numBodies);
  state.chebyPrevPrevRot.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    state.chebyPrevPos[i] = bodies[i].position;
    state.chebyPrevPrevPos[i] = bodies[i].position;
    state.chebyPrevRot[i] = bodies[i].rotation;
    state.chebyPrevPrevRot[i] = bodies[i].rotation;
  }
}

bool applyAvbdJointIterationPolicy(
    AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 iter,
    AvbdJointPositionPhaseState &positionPhase,
    const AvbdSolverConfig &config, bool enableEarlyStop,
    physx::PxU32 minIterations, physx::PxReal rotationTolerance,
    physx::PxArray<physx::PxVec3> &earlyStopPrevPos,
    physx::PxArray<physx::PxQuat> &earlyStopPrevRot,
    physx::PxU32 &consecutiveConvergedIterations) {
  if (positionPhase.useChebyshev && iter >= 2) {
    const physx::PxReal rhoSq =
        config.chebyshevRho * config.chebyshevRho;
    if (iter == 2)
      positionPhase.chebyOmega = 2.0f / (2.0f - rhoSq);
    else
      positionPhase.chebyOmega =
          1.0f /
          (1.0f - rhoSq * positionPhase.chebyOmega / 4.0f);
    positionPhase.chebyOmega =
        physx::PxClamp(positionPhase.chebyOmega, 1.0f, 2.0f);

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass <= 0.0f)
        continue;
      bodies[i].position =
          positionPhase.chebyPrevPrevPos[i] +
          (bodies[i].position - positionPhase.chebyPrevPrevPos[i]) *
              positionPhase.chebyOmega;
      physx::PxQuat qPrev = positionPhase.chebyPrevPrevRot[i];
      physx::PxQuat qCur = bodies[i].rotation;
      if (qPrev.dot(qCur) < 0.0f)
        qCur = -qCur;
      physx::PxQuat qBlend(
          qPrev.x + positionPhase.chebyOmega * (qCur.x - qPrev.x),
          qPrev.y + positionPhase.chebyOmega * (qCur.y - qPrev.y),
          qPrev.z + positionPhase.chebyOmega * (qCur.z - qPrev.z),
          qPrev.w + positionPhase.chebyOmega * (qCur.w - qPrev.w));
      bodies[i].rotation = qBlend.getNormalized();
    }
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  if (!enableEarlyStop)
    return false;

  physx::PxReal maxPositionDelta = 0.0f;
  physx::PxReal maxRotationDelta = 0.0f;
  computeMaxPoseDeltas(bodies, numBodies, earlyStopPrevPos,
                       earlyStopPrevRot, maxPositionDelta,
                       maxRotationDelta);
  if ((iter + 1) >= minIterations &&
      maxPositionDelta <= config.positionTolerance &&
      maxRotationDelta <= rotationTolerance) {
    ++consecutiveConvergedIterations;
    return consecutiveConvergedIterations >= 2;
  }
  consecutiveConvergedIterations = 0;
  return false;
}

void initializeAvbdJointExecutionPhaseState(
    const AvbdJointExecutionPhaseInput &input) {
  const AvbdSoftIslandExecutionPlan *const softExecutionPlan =
      input.softExecutionPlan;
  const bool useProvidedSoftExecutionPlan =
      input.useProvidedSoftExecutionPlan;
  const bool useProvidedRigidTargetContactPlan =
      input.useProvidedRigidTargetContactPlan;
  AvbdSoftBody *const softBodies = input.softBodies;
  const physx::PxU32 numSoftBodies = input.numSoftBodies;
  AvbdSoftContact *const softContacts = input.softContacts;
  const physx::PxU32 numSoftContacts = input.numSoftContacts;
  const physx::PxU32 numSoftParticles = input.numSoftParticles;
  AvbdOgcPairTrustRegionContext &mixedOgcPairContext =
      input.mixedOgcPairContext;
  AvbdJointExecutionPhaseState &state = input.state;
  initializeAvbdSoftExecutionData(
      softExecutionPlan, useProvidedSoftExecutionPlan,
      useProvidedRigidTargetContactPlan, softBodies, numSoftBodies,
      softContacts, numSoftContacts, numSoftParticles,
      state.softExecutionData);

  const AvbdSoftExecutionData &data = state.softExecutionData;
  if (mixedOgcPairContext.isComplete(numSoftContacts))
    mixedOgcPairContext.publishTriangleCoreSafetyPlan(
        data.triangleCoreSafetyStarts, data.numTriangleCoreSafetyStarts,
        data.triangleCoreSafetyRefs, data.numTriangleCoreSafetyRefs,
        numSoftParticles);
}

void buildAvbdJointPositionOwnedAngularBodies(
    physx::PxArray<bool> &positionOwnedAngularBodies,
    physx::PxU32 numBodies, const AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies,
    FeatherstoneArticulation *const *articulationForBody) {
  positionOwnedAngularBodies.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    positionOwnedAngularBodies[i] = false;

  for (physx::PxU32 softBodyIndex = 0;
       softBodyIndex < numSoftBodies; ++softBodyIndex) {
    const AvbdSoftBodyRuntimeState &runtime =
        softBodies[softBodyIndex].runtime;
    for (physx::PxU32 objectiveIndex = 0;
         objectiveIndex < runtime.compiledObjectives.size();
         ++objectiveIndex) {
      const AvbdCompiledSoftObjective &objective =
          runtime.compiledObjectives[objectiveIndex];
      if (objective.rigidBodyIdx >= numBodies)
        continue;

      if (objective.owner ==
          AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL) {
        positionOwnedAngularBodies[objective.rigidBodyIdx] = true;
        continue;
      }
      if (objective.owner !=
              AvbdSoftObjectiveOwner::eARTICULATION_ATTACHMENT_POSITION_AL ||
          !articulationForBody)
        continue;

      FeatherstoneArticulation *const articulation =
          articulationForBody[objective.rigidBodyIdx];
      if (!articulation)
        continue;
      for (physx::PxU32 linkBodyIndex = 0;
           linkBodyIndex < numBodies; ++linkBodyIndex)
        if (articulationForBody[linkBodyIndex] == articulation)
          positionOwnedAngularBodies[linkBodyIndex] = true;
    }
  }
}

} // namespace Dy
} // namespace physx
