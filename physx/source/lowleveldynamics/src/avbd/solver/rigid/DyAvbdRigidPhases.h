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

#ifndef DY_AVBD_RIGID_PHASES_H
#define DY_AVBD_RIGID_PHASES_H

#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

void runAvbdSoftPredictionPhase(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal dt, const physx::PxVec3 &gravity,
    bool hasPreparedSoftPrediction);

void initializeAvbdNoContactBodies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap);

void warmstartAvbdRigidBodies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<bool> &touchesKinematicShell, physx::PxReal dt,
    physx::PxReal invDt, const physx::PxVec3 &gravity,
    physx::PxReal shellFastImpactSpeed);

void applyAvbdPenaltyFloor(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    const AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    physx::PxReal invDt2);

} // namespace Dy
} // namespace physx

#endif
