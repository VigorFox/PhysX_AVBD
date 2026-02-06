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

#ifndef DY_AVBD_SOLVER_BODY_H
#define DY_AVBD_SOLVER_BODY_H

#include "foundation/PxAllocator.h"
#include "foundation/PxAssert.h"
#include "foundation/PxMat33.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"

#pragma warning(push)
#pragma warning(                                                               \
    disable : 4324) // Structure was padded due to alignment specifier

namespace physx {

namespace Dy {

/**
 * @brief AVBD Solver Body structure for position-based dynamics
 */
struct PX_ALIGN_PREFIX(16) AvbdSolverBody {
  //-------------------------------------------------------------------------
  // Current iteration state (x_k, q_k)
  //-------------------------------------------------------------------------

  physx::PxVec3 position; //!< Current iteration position x_k
  physx::PxReal invMass;  //!< Inverse mass (1/m), 0 for static bodies

  physx::PxQuat rotation; //!< Current iteration rotation q_k (quaternion)

  //-------------------------------------------------------------------------
  // Predicted state (x~, q~) from explicit integration
  //-------------------------------------------------------------------------

  physx::PxVec3
      predictedPosition; //!< Predicted position: x~ = x_n + h*v_n + h^2*f_ext/m
  physx::PxReal padding0; //!< Padding for 16-byte alignment

  physx::PxQuat predictedRotation; //!< Predicted rotation from angular
                                    //!< velocity integration

  //-------------------------------------------------------------------------
  // Velocity state (for final velocity computation)
  //-------------------------------------------------------------------------

  physx::PxVec3 linearVelocity; //!< Linear velocity v
  physx::PxReal padding1;

  physx::PxVec3 angularVelocity; //!< Angular velocity omega
  physx::PxReal padding2;

  //-------------------------------------------------------------------------
  // Previous frame state (x_n, q_n) for velocity derivation
  //-------------------------------------------------------------------------

  physx::PxVec3 prevPosition; //!< Previous frame position x_n
  physx::PxReal padding3;

  physx::PxQuat prevRotation; //!< Previous frame rotation q_n

  //-------------------------------------------------------------------------
  // Inertia tensor (world space inverse)
  //-------------------------------------------------------------------------

  physx::PxMat33 invInertiaWorld; //!< World-space inverse inertia tensor

  //-------------------------------------------------------------------------
  // Solver metadata
  //-------------------------------------------------------------------------

  physx::PxU32 nodeIndex;  //!< Index in the island/solver body array
  physx::PxU32 lockFlags;  //!< DOF lock flags (from PxRigidDynamicLockFlag)
  physx::PxU32 colorGroup; //!< Graph coloring group for parallel processing
  physx::PxU32
      numConstraints; //!< Number of constraints connected to this body

  //-------------------------------------------------------------------------
  // Methods
  //-------------------------------------------------------------------------

  /**
   * @brief Initialize from rigid body data
   */
  PX_FORCE_INLINE void initialize(const physx::PxTransform &globalPose,
                                  const physx::PxVec3 &linVel,
                                  const physx::PxVec3 &angVel,
                                  physx::PxReal invMassIn,
                                  const physx::PxMat33 &invInertiaIn,
                                  physx::PxU32 nodeIndexIn) {
    position = globalPose.p;
    rotation = globalPose.q;
    prevPosition = globalPose.p;
    prevRotation = globalPose.q;
    predictedPosition = globalPose.p;
    predictedRotation = globalPose.q;
    linearVelocity = linVel;
    angularVelocity = angVel;
    invMass = invMassIn;
    invInertiaWorld = invInertiaIn;
    nodeIndex = nodeIndexIn;
    lockFlags = 0;
    colorGroup = 0;
    numConstraints = 0;
  }

  /**
   * @brief Compute predicted position from current velocity
   * x~ = x_n + h*v + h^2*gravity/m (if not static)
   */
  PX_FORCE_INLINE void computePrediction(physx::PxReal dt,
                                         const physx::PxVec3 &gravity) {
    if (invMass > 0.0f) {
      // Explicit Euler prediction with gravity
      physx::PxVec3 acceleration = gravity;
      linearVelocity += acceleration * dt;
      predictedPosition = position + linearVelocity * dt;

      // Quaternion integration for rotation prediction
      physx::PxVec3 angVelHalf = angularVelocity * (0.5f * dt);
      physx::PxQuat deltaQ(angVelHalf.x, angVelHalf.y, angVelHalf.z, 0.0f);
      predictedRotation = rotation + deltaQ * rotation;
      predictedRotation.normalize();
    } else {
      // Static body: prediction equals current state
      predictedPosition = position;
      predictedRotation = rotation;
    }
  }

  /**
   * @brief Update velocity from position change after constraint solve
   * v_new = (x_new - x_n) / dt
   */
  PX_FORCE_INLINE void updateVelocityFromPosition(physx::PxReal invDt) {
    linearVelocity = (position - prevPosition) * invDt;

    // Angular velocity from quaternion difference
    // DISABLED: We are updating angular velocity directly via torque impulses
    // in the solver. Overwriting it here based on position/rotation change
    // would wipe out our torque contributions because rotation isn't updated
    // until AFTER this stage.
    /*
    // omega = 2 * (q_new * q_n^-1).xyz / dt
    physx::PxQuat deltaQ = rotation * prevRotation.getConjugate();
    if (deltaQ.w < 0.0f) {
      deltaQ = -deltaQ; // Ensure shortest path
    }
    angularVelocity =
        physx::PxVec3(deltaQ.x, deltaQ.y, deltaQ.z) * (2.0f * invDt);
    */
  }

  /**
   * @brief Check if this body is static (infinite mass)
   */
  PX_FORCE_INLINE bool isStatic() const { return invMass == 0.0f; }

} PX_ALIGN_SUFFIX(16);

// Verify expected size for cache alignment
PX_COMPILE_TIME_ASSERT(sizeof(AvbdSolverBody) % 16 == 0);

/**
 * @brief SoA (Structure of Arrays) layout for AVBD solver bodies
 */
struct AvbdSolverBodySoA {
  // Positions (x, y, z separate for SIMD)
  physx::PxReal *positionX;
  physx::PxReal *positionY;
  physx::PxReal *positionZ;

  // Rotations (quaternion components)
  physx::PxReal *rotationX;
  physx::PxReal *rotationY;
  physx::PxReal *rotationZ;
  physx::PxReal *rotationW;

  // Predicted positions
  physx::PxReal *predictedPositionX;
  physx::PxReal *predictedPositionY;
  physx::PxReal *predictedPositionZ;

  // Inverse mass
  physx::PxReal *invMass;

  // Linear velocity
  physx::PxReal *linearVelocityX;
  physx::PxReal *linearVelocityY;
  physx::PxReal *linearVelocityZ;

  // Angular velocity
  physx::PxReal *angularVelocityX;
  physx::PxReal *angularVelocityY;
  physx::PxReal *angularVelocityZ;

  // Body count
  physx::PxU32 numBodies;
  physx::PxU32 capacity;

  /**
   * @brief Allocate SoA arrays
   */
  inline void allocateData(physx::PxU32 count,
                           physx::PxAllocatorCallback &allocator) {
    capacity = count;
    numBodies = 0;

    // Allocate position arrays
    positionX = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::positionX",
                           __FILE__, __LINE__));
    positionY = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::positionY",
                           __FILE__, __LINE__));
    positionZ = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::positionZ",
                           __FILE__, __LINE__));

    // Allocate rotation arrays
    rotationX = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::rotationX",
                           __FILE__, __LINE__));
    rotationY = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::rotationY",
                           __FILE__, __LINE__));
    rotationZ = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::rotationZ",
                           __FILE__, __LINE__));
    rotationW = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::rotationW",
                           __FILE__, __LINE__));

    // Allocate predicted position arrays
    predictedPositionX = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::predictedPositionX", __FILE__, __LINE__));
    predictedPositionY = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::predictedPositionY", __FILE__, __LINE__));
    predictedPositionZ = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::predictedPositionZ", __FILE__, __LINE__));

    // Allocate inverse mass
    invMass = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal), "AvbdSoA::invMass",
                           __FILE__, __LINE__));

    // Allocate linear velocity arrays
    linearVelocityX = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::linearVelocityX", __FILE__, __LINE__));
    linearVelocityY = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::linearVelocityY", __FILE__, __LINE__));
    linearVelocityZ = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::linearVelocityZ", __FILE__, __LINE__));

    // Allocate angular velocity arrays
    angularVelocityX = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::angularVelocityX", __FILE__, __LINE__));
    angularVelocityY = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::angularVelocityY", __FILE__, __LINE__));
    angularVelocityZ = static_cast<physx::PxReal *>(
        allocator.allocate(count * sizeof(physx::PxReal),
                           "AvbdSoA::angularVelocityZ", __FILE__, __LINE__));
  }

  /**
   * @brief Deallocate SoA arrays
   */
  inline void deallocateData(physx::PxAllocatorCallback &allocator) {
    if (positionX)
      allocator.deallocate(positionX);
    if (positionY)
      allocator.deallocate(positionY);
    if (positionZ)
      allocator.deallocate(positionZ);

    if (rotationX)
      allocator.deallocate(rotationX);
    if (rotationY)
      allocator.deallocate(rotationY);
    if (rotationZ)
      allocator.deallocate(rotationZ);
    if (rotationW)
      allocator.deallocate(rotationW);

    if (predictedPositionX)
      allocator.deallocate(predictedPositionX);
    if (predictedPositionY)
      allocator.deallocate(predictedPositionY);
    if (predictedPositionZ)
      allocator.deallocate(predictedPositionZ);

    if (invMass)
      allocator.deallocate(invMass);

    if (linearVelocityX)
      allocator.deallocate(linearVelocityX);
    if (linearVelocityY)
      allocator.deallocate(linearVelocityY);
    if (linearVelocityZ)
      allocator.deallocate(linearVelocityZ);

    if (angularVelocityX)
      allocator.deallocate(angularVelocityX);
    if (angularVelocityY)
      allocator.deallocate(angularVelocityY);
    if (angularVelocityZ)
      allocator.deallocate(angularVelocityZ);

    positionX = positionY = positionZ = nullptr;
    rotationX = rotationY = rotationZ = rotationW = nullptr;
    predictedPositionX = predictedPositionY = predictedPositionZ = nullptr;
    invMass = nullptr;
    linearVelocityX = linearVelocityY = linearVelocityZ = nullptr;
    angularVelocityX = angularVelocityY = angularVelocityZ = nullptr;

    numBodies = 0;
    capacity = 0;
  }

  /**
   * @brief Copy from AoS to SoA layout
   */
  inline void copyFromAoS(const AvbdSolverBody *bodies, physx::PxU32 count) {
    PX_ASSERT(count <= capacity);
    numBodies = count;

    for (physx::PxU32 i = 0; i < count; ++i) {
      const AvbdSolverBody &body = bodies[i];
      positionX[i] = body.position.x;
      positionY[i] = body.position.y;
      positionZ[i] = body.position.z;

      rotationX[i] = body.rotation.x;
      rotationY[i] = body.rotation.y;
      rotationZ[i] = body.rotation.z;
      rotationW[i] = body.rotation.w;

      predictedPositionX[i] = body.predictedPosition.x;
      predictedPositionY[i] = body.predictedPosition.y;
      predictedPositionZ[i] = body.predictedPosition.z;

      invMass[i] = body.invMass;

      linearVelocityX[i] = body.linearVelocity.x;
      linearVelocityY[i] = body.linearVelocity.y;
      linearVelocityZ[i] = body.linearVelocity.z;

      angularVelocityX[i] = body.angularVelocity.x;
      angularVelocityY[i] = body.angularVelocity.y;
      angularVelocityZ[i] = body.angularVelocity.z;
    }
  }

  /**
   * @brief Copy from SoA to AoS layout
   */
  inline void copyToAoS(AvbdSolverBody *bodies, physx::PxU32 count) const {
    PX_ASSERT(count <= numBodies);
    for (physx::PxU32 i = 0; i < count; ++i) {
      AvbdSolverBody &body = bodies[i];
      body.position.x = positionX[i];
      body.position.y = positionY[i];
      body.position.z = positionZ[i];

      body.rotation.x = rotationX[i];
      body.rotation.y = rotationY[i];
      body.rotation.z = rotationZ[i];
      body.rotation.w = rotationW[i];

      body.predictedPosition.x = predictedPositionX[i];
      body.predictedPosition.y = predictedPositionY[i];
      body.predictedPosition.z = predictedPositionZ[i];

      body.invMass = invMass[i];

      body.linearVelocity.x = linearVelocityX[i];
      body.linearVelocity.y = linearVelocityY[i];
      body.linearVelocity.z = linearVelocityZ[i];

      body.angularVelocity.x = angularVelocityX[i];
      body.angularVelocity.y = angularVelocityY[i];
      body.angularVelocity.z = angularVelocityZ[i];
    }
  }
};

} // namespace Dy

} // namespace physx

#pragma warning(pop)

#endif // DY_AVBD_SOLVER_BODY_H
