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

#ifndef DY_AVBD_JOINT_PHASE_STATE_H
#define DY_AVBD_JOINT_PHASE_STATE_H

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/solver/joint/DyAvbdJointSoftExecutionData.h"

namespace physx {
namespace Dy {

struct AvbdJointPhaseAdmission {
  bool hasCompleteSoftSelection;
  bool useProvidedSoftExecutionPlan;
  bool useProvidedRigidTargetContactPlan;
  bool hasPreparedSoftPrediction;
};

AvbdJointPhaseAdmission buildAvbdJointPhaseAdmission(
    bool solverInitialized, physx::PxU32 numBodies,
    AvbdSolverBody *bodies, AvbdSoftParticle *softParticles,
    physx::PxU32 numSoftParticles, AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies, AvbdSoftContact *softContacts,
    physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan);

struct AvbdJointContactPhaseState {
  physx::PxArray<bool> touchesKinematicShell;
  physx::PxArray<physx::PxVec3> shellLinearVelAtSolveStart;
  physx::PxArray<bool> touchingBodyStatic;
  physx::PxArray<physx::PxVec3> linearVelAtSolveStart;
  physx::PxArray<physx::PxVec3> angularVelAtSolveStart;
  bool hasKinematicShellContacts;
  bool hasBodyStaticContact;
  bool hasDeformableAnchorContact;
  bool allBodyVsStatic;
  bool deformableFastImpactIsland;
};

void buildAvbdJointContactPhaseState(
    AvbdJointContactPhaseState &state, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numD6, physx::PxU32 numGear,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt,
    bool captureAngularVelocityForPassiveGear);

struct AvbdJointPositionPhaseState {
  bool useChebyshev;
  physx::PxReal chebyOmega;
  physx::PxArray<physx::PxVec3> chebyPrevPos;
  physx::PxArray<physx::PxVec3> chebyPrevPrevPos;
  physx::PxArray<physx::PxQuat> chebyPrevRot;
  physx::PxArray<physx::PxQuat> chebyPrevPrevRot;
};

struct AvbdJointContactClosureMetrics {
  physx::PxReal maxComplementarityResidual;
  physx::PxReal maxClosingDisplacement;
  bool finite;
};

void initializeAvbdJointPositionPhaseState(
    AvbdJointPositionPhaseState &state, const AvbdSolverConfig &config,
    bool slerpVelocityDriveIsland, bool coupledFixedD6Island,
    bool coupledSphericalConeIsland, bool coupledSpatialTendonIsland,
    bool hasRigidContact, bool hasDynamicSoftRigidContact,
    AvbdSolverBody *bodies, physx::PxU32 numBodies);

AvbdJointContactClosureMetrics evaluateAvbdJointContactClosure(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal avbdAlpha);

bool applyAvbdJointIterationPolicy(
    AvbdSolverBody *bodies, physx::PxU32 numBodies, physx::PxU32 iter,
    AvbdJointPositionPhaseState &positionPhase,
    const AvbdSolverConfig &config, bool enableEarlyStop,
    physx::PxU32 minIterations, physx::PxReal rotationTolerance,
    physx::PxArray<physx::PxVec3> &earlyStopPrevPos,
    physx::PxArray<physx::PxQuat> &earlyStopPrevRot,
    physx::PxU32 &consecutiveConvergedIterations);

struct AvbdJointExecutionPhaseState {
  AvbdSoftExecutionData softExecutionData;
};

struct AvbdJointExecutionPhaseInput {
  const AvbdSoftIslandExecutionPlan *softExecutionPlan;
  bool useProvidedSoftExecutionPlan;
  bool useProvidedRigidTargetContactPlan;
  AvbdSoftBody *softBodies;
  physx::PxU32 numSoftBodies;
  AvbdSoftContact *softContacts;
  physx::PxU32 numSoftContacts;
  physx::PxU32 numSoftParticles;
  AvbdOgcPairTrustRegionContext &mixedOgcPairContext;
  AvbdJointExecutionPhaseState &state;
};

void initializeAvbdJointExecutionPhaseState(
    const AvbdJointExecutionPhaseInput &input);

void buildAvbdJointPositionOwnedAngularBodies(
    physx::PxArray<bool> &positionOwnedAngularBodies,
    physx::PxU32 numBodies, const AvbdSoftBody *softBodies,
    physx::PxU32 numSoftBodies,
    FeatherstoneArticulation *const *articulationForBody);

} // namespace Dy
} // namespace physx

#endif
