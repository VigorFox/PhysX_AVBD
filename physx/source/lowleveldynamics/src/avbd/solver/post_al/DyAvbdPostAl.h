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

#ifndef DY_AVBD_POST_AL_H
#define DY_AVBD_POST_AL_H

#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/ogc/DyAvbdOgcTerminalState.h"
#include "avbd/solver/post_al/DyAvbdPostAlContactResponse.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

void finalizePostAlSoftVelocities(
    AvbdSoftParticle *softParticlesForVel,
    physx::PxU32 numSoftParticlesForVel, physx::PxReal invDt,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
    const AvbdSoftBody *softBodiesForRecovery,
    physx::PxU32 numSoftBodiesForRecovery, AvbdSoftContact *shellContacts,
    physx::PxU32 numShellContacts, physx::PxReal dt,
	const AvbdSoftIslandExecutionPlan *geometryPlan,
    AvbdOgcPairState *selectionPairStates,
    physx::PxU32 numSelectionPairStates,
    AvbdSoftContact *terminalContacts, physx::PxU32 numTerminalContacts,
    bool terminalCurrentPoseEpochApplied,
    const AvbdTerminalOgcState *terminalOgcState,
    AvbdOgcPairState *terminalPairStates,
    physx::PxU32 numTerminalPairStates,
    physx::PxReal lengthScale, AvbdSolverStats &stats);

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
    physx::PxReal velocityDamping, physx::PxReal angularDamping);

struct AvbdPostAlPoseState {
  physx::PxArray<bool> splitRigidDeepPoseRecovery;
  physx::PxArray<bool> splitRigidFiniteMaterialPose;
  physx::PxArray<physx::PxVec3> postBlockPos;
  physx::PxArray<physx::PxQuat> postBlockRot;
  physx::PxArray<physx::PxU8> deformableNormalStageMask;
  bool hasNormalStageMask;

  AvbdPostAlPoseState() : hasNormalStageMask(false) {}

  physx::PxArray<physx::PxU8> *normalStageMaskPtr() {
    return hasNormalStageMask ? &deformableNormalStageMask : nullptr;
  }

  const physx::PxArray<physx::PxU8> *normalStageMaskPtr() const {
    return hasNormalStageMask ? &deformableNormalStageMask : nullptr;
  }

  void run(
      AvbdSolver &solver, AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      const AvbdBodyConstraintMap *contactMap,
      const physx::PxVec3 &gravity, physx::PxReal dt,
      bool hasBodyStaticContact, bool deformableFastImpactIsland,
      bool allowRigidDeepPoseRecoverySplit,
      bool allowRigidFiniteMaterialPoseSplit,
      AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
      AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
      const physx::PxArray<bool> &touchesKinematicShell,
      AvbdSolverStats &stats);
};

struct AvbdPostAlRecoveryState {
  void run(
      AvbdSolver &solver, AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
      const AvbdSoftBody *softBodiesForRecovery,
      physx::PxU32 numSoftBodiesForRecovery,
      AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
      AvbdSolverStats &stats,
      const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan);
};

struct AvbdPostAlFrictionState {
  physx::PxArray<physx::PxVec3> postDepenPos;
  physx::PxArray<physx::PxQuat> postDepenRot;
  AvbdBodyStaticFrictionWorkspace responseWorkspace;

  void run(
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdContactConstraint *contacts, physx::PxU32 numContacts,
      const physx::PxVec3 &gravity, physx::PxReal dt,
      bool skipBodyStaticFriction, bool hasKinematicShellContacts,
      const physx::PxArray<bool> &touchesKinematicShell,
      physx::PxReal lengthScale, AvbdSolverStats &stats);
};

struct AvbdPostAlTerminalState {
  AvbdTerminalOgcState ogc;

  void run(
      const AvbdSoftIslandExecutionPlan *terminalSoftExecutionPlan,
      AvbdSolverBody *bodies, physx::PxU32 numBodies,
      AvbdSoftParticle *shellParticles, physx::PxU32 numShellParticles,
      const AvbdSoftBody *softBodiesForRecovery,
      physx::PxU32 numSoftBodiesForRecovery,
      AvbdSoftContact *shellContacts, physx::PxU32 numShellContacts,
      physx::PxReal lengthScale, AvbdSolverStats &stats);
};

struct AvbdPostAlVelocityState {
  physx::PxArray<physx::PxU32> physicalContactTangentOwnerIndex;
  physx::PxArray<bool> fastNormalImpactByBody;
  /** Scalar gain actually applied to pose-derived linear contact response. */
  physx::PxArray<physx::PxReal> linearPoseVelocityGain;
  /** Scalar gain actually applied to pose-derived angular contact response. */
  physx::PxArray<physx::PxReal> angularPoseVelocityGain;
  bool haveSolveStartLinear;

  void build(
      physx::PxU32 numBodies, AvbdContactConstraint *contacts,
      physx::PxU32 numContacts,
      const physx::PxArray<bool> &touchingBodyStatic,
      const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
      bool deformableFastImpactIsland);
};

// Scene-owned scratch for one mixed soft/rigid island.  State objects keep
// their PxArray capacity across frames, while run() methods overwrite only
// the active ranges.  The execution plan guarantees exclusive ownership for
// one island solve, so no solver-global mutable scratch is introduced.
struct AvbdPostAlWorkspace {
  AvbdOgcPoseWritePhaseState poseWriteAdmission;
  AvbdPostAlPoseState pose;
  AvbdPostAlRecoveryState recovery;
  AvbdPostAlFrictionState friction;
  AvbdPostAlTerminalState terminal;
  AvbdPostAlVelocityState velocity;
  physx::PxU64 growthEvents;
  physx::PxU64 growthBytes;
  physx::PxU64 capacityAtSolveStart;
  physx::PxU64 contactWorkspaceGrowthBytesAtSolveStart;

  AvbdPostAlWorkspace()
      : growthEvents(0u), growthBytes(0u), capacityAtSolveStart(0u),
        contactWorkspaceGrowthBytesAtSolveStart(0u) {}

  physx::PxU64 capacityBytes() const;
  void beginSolve();
  void endSolve();
};

// Internal contract consumed by the post-AL coordinator.  Implementations
// stay with the rigid solver kernels; exposing this narrow seam removes the
// former include-order dependency on DyAvbdSolver.cpp's anonymous namespace.
bool bodyTouchesDeformableAnchor(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex,
    const AvbdBodyConstraintMap *contactMap = nullptr);

void projectBodyStaticLockedD6LinearVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint *joints, physx::PxU32 numJoints);

void applyAvbdContactTargetVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    const AvbdPostAlContactWorkPlan *workPlan);

bool isRigidDeepBodyStaticRecoverySplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale);

bool isRigidFiniteBodyStaticMaterialSplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale);

void clampBodyStaticInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale, bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats);

void recordBodyStaticNormalAlOwnership(
    const AvbdSolverBody *bodies, const AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numBodies,
    physx::PxReal avbdAlpha,
    const physx::PxArray<bool> *touchesKinematicShell,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats &stats);

} // namespace Dy
} // namespace physx

#endif
