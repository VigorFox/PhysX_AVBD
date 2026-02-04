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

#ifndef DY_AVBD_BODY_CONVERSION_H
#define DY_AVBD_BODY_CONVERSION_H

#include "../../common/src/CmUtils.h"
#include "DyAvbdSolverBody.h"
#include "PxsRigidBody.h"
#include "foundation/PxSIMDHelpers.h"


namespace physx {
namespace Dy {

/**
 * @brief Convert PhysX body core to AVBD solver body
 *
 * @param core Source PhysX body core data
 * @param body Target AVBD solver body
 * @param bodyIndex Index in solver body array
 */
PX_FORCE_INLINE void copyToAvbdSolverBody(const PxsBodyCore &core,
                                          AvbdSolverBody &body,
                                          PxU32 bodyIndex) {
  // Initialize AVBD body from PhysX core data
  const PxTransform &pose = core.body2World;
  const PxVec3 &linVel = core.linearVelocity;
  const PxVec3 &angVel = core.angularVelocity;
  const PxReal invMass = core.inverseMass;
  const PxVec3 &invInertia = core.inverseInertia;

  // Create world-space inverse inertia matrix
  const PxMat33Padded rotation(pose.q);
  PxMat33 invInertiaTensor;
  Cm::transformInertiaTensor(invInertia, rotation, invInertiaTensor);

  // Initialize AVBD solver body
  body.initialize(pose, linVel, angVel, invMass, invInertiaTensor, bodyIndex);
}

/**
 * @brief Write back AVBD solver results to PhysX body core
 *
 * @param body Source AVBD solver body (after solving)
 * @param core Target PhysX body core to update
 */
PX_FORCE_INLINE void writeBackAvbdSolverBody(const AvbdSolverBody &body,
                                             PxsBodyCore &core) {
  // Write position and orientation
  core.body2World.p = body.position;
  
  // Quaternion should already be normalized from solver
  PX_ASSERT(PxAbs(body.rotation.magnitudeSquared() - 1.0f) < 0.01f && 
            "Quaternion not normalized in writeBackAvbdSolverBody");
  core.body2World.q = body.rotation;

  // Write velocities
  core.linearVelocity = body.linearVelocity;
  core.angularVelocity = body.angularVelocity;
}

/**
 * @brief Initialize a static AVBD body (zero mass)
 *
 * @param pose World transform of static body
 * @param body Target AVBD solver body
 * @param bodyIndex Index in solver body array
 */
PX_FORCE_INLINE void initializeStaticAvbdBody(const PxTransform &pose,
                                              AvbdSolverBody &body,
                                              PxU32 bodyIndex) {
  body.initialize(pose, PxVec3(0.0f), PxVec3(0.0f), 0.0f, PxMat33(PxIdentity),
                  bodyIndex);
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_BODY_CONVERSION_H
