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
#include "PxRigidDynamic.h"

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
  // Inertial target state (for AVBD RHS computation)
  // This is the pure inertial prediction (gravity-only, no warmstarting).
  // The RHS of the AVBD linear system uses (position - inertialPosition)
  // so that the inertia term is non-zero when position is warm-started.
  //-------------------------------------------------------------------------

  physx::PxVec3
      inertialPosition; //!< Pure inertial target: x_n + h*v + h^2*g
  physx::PxReal padding0b;

  physx::PxQuat inertialRotation; //!< Pure inertial rotation target

  //-------------------------------------------------------------------------
  // Velocity state (for final velocity computation)
  //-------------------------------------------------------------------------

  physx::PxVec3 linearVelocity; //!< Linear velocity v
  physx::PxReal padding1;

  physx::PxVec3 prevLinearVelocity; //!< Previous frame linear velocity (for adaptive warmstart)
  physx::PxReal padding1b;

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
  // Per-body damping and velocity caps (from PxsBodyCore)
  //-------------------------------------------------------------------------

  physx::PxReal linearDamping;       //!< Per-body linear damping (0 = none)
  physx::PxReal angularDampingBody;  //!< Per-body angular damping (0 = none)
  physx::PxReal maxLinearVelocitySq; //!< Max linear velocity squared
  physx::PxReal maxAngularVelocitySq;//!< Max angular velocity squared
  physx::PxReal gravityScale;         //!< 0 when eDISABLE_GRAVITY is set

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
    inertialPosition = globalPose.p;
    inertialRotation = globalPose.q;
    linearVelocity = linVel;
    prevLinearVelocity = linVel;
    angularVelocity = angVel;
    invMass = invMassIn;
    invInertiaWorld = invInertiaIn;
    nodeIndex = nodeIndexIn;
    lockFlags = 0;
    colorGroup = 0;
    numConstraints = 0;
    linearDamping = 0.0f;
    angularDampingBody = 0.0f;
    maxLinearVelocitySq = PX_MAX_F32;
    maxAngularVelocitySq = PX_MAX_F32;
    gravityScale = 1.0f;
  }

  PX_FORCE_INLINE void projectLockedLinearVector(physx::PxVec3 &value) const {
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_X)
      value.x = 0.0f;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_Y)
      value.y = 0.0f;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_Z)
      value.z = 0.0f;
  }

  PX_FORCE_INLINE void projectLockedAngularVector(physx::PxVec3 &value) const {
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_X)
      value.x = 0.0f;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y)
      value.y = 0.0f;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z)
      value.z = 0.0f;
  }

  PX_FORCE_INLINE void projectLockedVelocities() {
    if (lockFlags == 0)
      return;
    projectLockedLinearVector(linearVelocity);
    projectLockedAngularVector(angularVelocity);
  }

  PX_FORCE_INLINE void
  projectLockedPose(const physx::PxVec3 &referencePosition,
                    const physx::PxQuat &referenceRotation) {
    if (lockFlags == 0)
      return;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_X)
      position.x = referencePosition.x;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_Y)
      position.y = referencePosition.y;
    if (lockFlags & physx::PxRigidDynamicLockFlag::eLOCK_LINEAR_Z)
      position.z = referencePosition.z;

    const physx::PxU32 angularLocks =
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
    if ((lockFlags & angularLocks) == 0)
      return;

    physx::PxQuat delta = rotation * referenceRotation.getConjugate();
    if (delta.w < 0.0f)
      delta = -delta;
    delta.normalize();
    const physx::PxReal cosine =
        physx::PxClamp(delta.w, 0.0f, 1.0f);
    const physx::PxReal sinHalf =
        physx::PxSqrt(physx::PxMax(0.0f, 1.0f - cosine * cosine));
    physx::PxVec3 rotationVector;
    if (sinHalf > 1e-6f) {
      const physx::PxReal angle = 2.0f * physx::PxAcos(cosine);
      rotationVector =
          physx::PxVec3(delta.x, delta.y, delta.z) * (angle / sinHalf);
    } else {
      rotationVector =
          physx::PxVec3(delta.x, delta.y, delta.z) * 2.0f;
    }
    projectLockedAngularVector(rotationVector);

    const physx::PxReal angle = rotationVector.magnitude();
    const physx::PxQuat projectedDelta =
        angle > 1e-8f
            ? physx::PxQuat(angle, rotationVector / angle)
            : physx::PxQuat(physx::PxIdentity);
    rotation = (projectedDelta * referenceRotation).getNormalized();
  }

  /**
   * @brief Compute predicted position from current velocity
   * x~ = x_n + h*v + h^2*gravity/m (if not static)
   */
  PX_FORCE_INLINE void computePrediction(physx::PxReal dt,
                                         const physx::PxVec3 &gravity) {
    if (invMass > 0.0f) {
      // Prediction: x_pred = x + (v + g*dt)*dt = x + v*dt + g*dt^2
      // NOTE: We do NOT modify linearVelocity here (no gravity kick).
      // linearVelocity always holds the clean post-solve velocity from the
      // previous frame. This is essential for the adaptive warmstart which
      // computes acceleration = (v_current - v_previous) / dt.
      // prevLinearVelocity is updated at end-of-solve, NOT here.
      projectLockedVelocities();
      physx::PxVec3 stepLinearVelocity =
          linearVelocity + gravity * (gravityScale * dt);
      projectLockedLinearVector(stepLinearVelocity);
      predictedPosition = position + stepLinearVelocity * dt;

      // Quaternion integration for rotation prediction
      physx::PxVec3 stepAngularVelocity = angularVelocity;
      projectLockedAngularVector(stepAngularVelocity);
      physx::PxVec3 angVelHalf = stepAngularVelocity * (0.5f * dt);
      physx::PxQuat deltaQ(angVelHalf.x, angVelHalf.y, angVelHalf.z, 0.0f);
      predictedRotation = rotation + deltaQ * rotation;
      predictedRotation.normalize();
      const physx::PxVec3 currentPosition = position;
      const physx::PxQuat currentRotation = rotation;
      position = predictedPosition;
      rotation = predictedRotation;
      projectLockedPose(currentPosition, currentRotation);
      predictedPosition = position;
      predictedRotation = rotation;
      position = currentPosition;
      rotation = currentRotation;

      // Inertial target = pure prediction (same as predicted, used as RHS anchor)
      inertialPosition = predictedPosition;
      inertialRotation = predictedRotation;
    } else {
      // Static body: prediction equals current state
      predictedPosition = position;
      predictedRotation = rotation;
      inertialPosition = position;
      inertialRotation = rotation;
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

  // One contiguous allocation keeps the arrays in a predictable cache/TLB
  // footprint and makes the view safe to construct before first use.
  physx::PxReal *storage;

  PX_FORCE_INLINE AvbdSolverBodySoA()
      : positionX(nullptr), positionY(nullptr), positionZ(nullptr),
        rotationX(nullptr), rotationY(nullptr), rotationZ(nullptr),
        rotationW(nullptr), predictedPositionX(nullptr),
        predictedPositionY(nullptr), predictedPositionZ(nullptr),
        invMass(nullptr), linearVelocityX(nullptr), linearVelocityY(nullptr),
        linearVelocityZ(nullptr), angularVelocityX(nullptr),
        angularVelocityY(nullptr), angularVelocityZ(nullptr), numBodies(0),
        capacity(0), storage(nullptr) {}

  /**
   * @brief Allocate SoA arrays
   */
  inline void allocateData(physx::PxU32 count,
                           physx::PxAllocatorCallback &allocator) {
    if (storage || positionX || positionY || positionZ || rotationX ||
        rotationY || rotationZ || rotationW || predictedPositionX ||
        predictedPositionY || predictedPositionZ || invMass ||
        linearVelocityX || linearVelocityY || linearVelocityZ ||
        angularVelocityX || angularVelocityY || angularVelocityZ)
      deallocateData(allocator);
    capacity = count;
    numBodies = 0;
    if (count == 0)
      return;

    // 3 position + 4 rotation + 3 predicted position + 1 inverse mass,
    // 3 linear velocity + 3 angular velocity arrays.
    const physx::PxU32 fieldCount = 17;
    storage = static_cast<physx::PxReal *>(allocator.allocate(
        sizeof(physx::PxReal) * count * fieldCount,
        "AvbdSolverBodySoA::storage", __FILE__, __LINE__));
    if (!storage) {
      capacity = 0;
      return;
    }

    physx::PxReal *fields = storage;
    positionX = fields + count * 0;
    positionY = fields + count * 1;
    positionZ = fields + count * 2;
    rotationX = fields + count * 3;
    rotationY = fields + count * 4;
    rotationZ = fields + count * 5;
    rotationW = fields + count * 6;
    predictedPositionX = fields + count * 7;
    predictedPositionY = fields + count * 8;
    predictedPositionZ = fields + count * 9;
    invMass = fields + count * 10;
    linearVelocityX = fields + count * 11;
    linearVelocityY = fields + count * 12;
    linearVelocityZ = fields + count * 13;
    angularVelocityX = fields + count * 14;
    angularVelocityY = fields + count * 15;
    angularVelocityZ = fields + count * 16;
  }

  /**
   * @brief Deallocate SoA arrays
   */
  inline void deallocateData(physx::PxAllocatorCallback &allocator) {
    if (storage) {
      allocator.deallocate(storage);
    } else {
      // Retain safe cleanup for views produced by older callers that may have
      // assigned individual arrays before the contiguous layout was added.
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
    }

    positionX = positionY = positionZ = nullptr;
    rotationX = rotationY = rotationZ = rotationW = nullptr;
    predictedPositionX = predictedPositionY = predictedPositionZ = nullptr;
    invMass = nullptr;
    linearVelocityX = linearVelocityY = linearVelocityZ = nullptr;
    angularVelocityX = angularVelocityY = angularVelocityZ = nullptr;
    storage = nullptr;

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

/**
 * Complete hot-state SoA storage for the producer-native rigid path.
 *
 * This is deliberately separate from AvbdSolverBodySoA.  The older view is
 * retained for compatibility and contains only the fields used by the first
 * pose-only experiments.  P61's view owns every state element consumed while
 * assembling a rigid local system, so a future packet producer can read the
 * state directly without an AoS staging copy.  It is not selected by the
 * solver until its write ownership and scalar differential gates are complete.
 */
struct AvbdSolverBodyHotSoA {
  // Field ordinals are part of the producer/consumer ABI.  Keep this list in
  // the same order as allocateData() below: a producer can bind a range to
  // this storage without constructing a second pointer-rich sidecar.
  enum FloatField : physx::PxU32 {
    ePositionX = 0,
    ePositionY,
    ePositionZ,
    eRotationX,
    eRotationY,
    eRotationZ,
    eRotationW,
    ePredictedPositionX,
    ePredictedPositionY,
    ePredictedPositionZ,
    ePredictedRotationX,
    ePredictedRotationY,
    ePredictedRotationZ,
    ePredictedRotationW,
    eInertialPositionX,
    eInertialPositionY,
    eInertialPositionZ,
    eInertialRotationX,
    eInertialRotationY,
    eInertialRotationZ,
    eInertialRotationW,
    eLinearVelocityX,
    eLinearVelocityY,
    eLinearVelocityZ,
    ePrevLinearVelocityX,
    ePrevLinearVelocityY,
    ePrevLinearVelocityZ,
    eAngularVelocityX,
    eAngularVelocityY,
    eAngularVelocityZ,
    ePrevPositionX,
    ePrevPositionY,
    ePrevPositionZ,
    ePrevRotationX,
    ePrevRotationY,
    ePrevRotationZ,
    ePrevRotationW,
    eInvInertiaWorld00,
    eInvInertiaWorld01,
    eInvInertiaWorld02,
    eInvInertiaWorld10,
    eInvInertiaWorld11,
    eInvInertiaWorld12,
    eInvInertiaWorld20,
    eInvInertiaWorld21,
    eInvInertiaWorld22,
    eInvMass,
    eLinearDamping,
    eAngularDampingBody,
    eMaxLinearVelocitySq,
    eMaxAngularVelocitySq,
    eGravityScale,
    eFloatFieldCount
  };

  enum UintField : physx::PxU32 {
    eNodeIndex = 0,
    eLockFlags,
    eColorGroup,
    eNumConstraints,
    eUintFieldCount
  };

  // Position/rotation state.  The first index is the component.
  physx::PxReal *position[3];
  physx::PxReal *rotation[4];
  physx::PxReal *predictedPosition[3];
  physx::PxReal *predictedRotation[4];
  physx::PxReal *inertialPosition[3];
  physx::PxReal *inertialRotation[4];
  physx::PxReal *linearVelocity[3];
  physx::PxReal *prevLinearVelocity[3];
  physx::PxReal *angularVelocity[3];
  physx::PxReal *prevPosition[3];
  physx::PxReal *prevRotation[4];

  // World-space inverse inertia, indexed as [column][row] to match PxMat33.
  physx::PxReal *invInertiaWorld[3][3];

  // Scalar state used by prediction, locking, damping, and local assembly.
  physx::PxReal *invMass;
  physx::PxReal *linearDamping;
  physx::PxReal *angularDampingBody;
  physx::PxReal *maxLinearVelocitySq;
  physx::PxReal *maxAngularVelocitySq;
  physx::PxReal *gravityScale;

  // Integer metadata used by dependency ownership and fallback selection.
  physx::PxU32 *nodeIndex;
  physx::PxU32 *lockFlags;
  physx::PxU32 *colorGroup;
  physx::PxU32 *numConstraints;

  physx::PxU32 numBodies;
  physx::PxU32 capacity;
  physx::PxReal *floatStorage;
  physx::PxU32 *uintStorage;

  PX_FORCE_INLINE AvbdSolverBodyHotSoA()
      : invMass(nullptr), linearDamping(nullptr),
        angularDampingBody(nullptr), maxLinearVelocitySq(nullptr),
        maxAngularVelocitySq(nullptr), gravityScale(nullptr),
        nodeIndex(nullptr), lockFlags(nullptr), colorGroup(nullptr),
        numConstraints(nullptr), numBodies(0), capacity(0),
        floatStorage(nullptr), uintStorage(nullptr) {
    for (physx::PxU32 i = 0; i < 3; ++i) {
      position[i] = predictedPosition[i] = inertialPosition[i] = nullptr;
      linearVelocity[i] = prevLinearVelocity[i] = angularVelocity[i] = nullptr;
      prevPosition[i] = nullptr;
      for (physx::PxU32 j = 0; j < 3; ++j)
        invInertiaWorld[i][j] = nullptr;
    }
    for (physx::PxU32 i = 0; i < 4; ++i) {
      rotation[i] = predictedRotation[i] = inertialRotation[i] = nullptr;
      prevRotation[i] = nullptr;
    }
  }

  PX_FORCE_INLINE void clearPointers() {
    for (physx::PxU32 i = 0; i < 3; ++i) {
      position[i] = predictedPosition[i] = inertialPosition[i] = nullptr;
      linearVelocity[i] = prevLinearVelocity[i] = angularVelocity[i] = nullptr;
      prevPosition[i] = nullptr;
      for (physx::PxU32 j = 0; j < 3; ++j)
        invInertiaWorld[i][j] = nullptr;
    }
    for (physx::PxU32 i = 0; i < 4; ++i) {
      rotation[i] = predictedRotation[i] = inertialRotation[i] = nullptr;
      prevRotation[i] = nullptr;
    }
    invMass = linearDamping = angularDampingBody = nullptr;
    maxLinearVelocitySq = maxAngularVelocitySq = gravityScale = nullptr;
    nodeIndex = lockFlags = colorGroup = numConstraints = nullptr;
  }

  inline void allocateData(physx::PxU32 count,
                           physx::PxAllocatorCallback &allocator) {
    deallocateData(allocator);
    capacity = count;
    numBodies = 0;
    if (count == 0)
      return;

    // All body state, inertia, and scalar hot fields.
    const physx::PxU32 floatFieldCount = eFloatFieldCount;
    const physx::PxU32 uintFieldCount = eUintFieldCount;
    floatStorage = static_cast<physx::PxReal *>(allocator.allocate(
        sizeof(physx::PxReal) * count * floatFieldCount,
        "AvbdSolverBodyHotSoA::floatStorage", __FILE__, __LINE__));
    uintStorage = static_cast<physx::PxU32 *>(allocator.allocate(
        sizeof(physx::PxU32) * count * uintFieldCount,
        "AvbdSolverBodyHotSoA::uintStorage", __FILE__, __LINE__));
    if (!floatStorage || !uintStorage) {
      deallocateData(allocator);
      return;
    }

    physx::PxU32 field = 0;
    const auto nextFloat = [&]() {
      return floatStorage + count * field++;
    };
    for (physx::PxU32 i = 0; i < 3; ++i)
      position[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 4; ++i)
      rotation[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 3; ++i)
      predictedPosition[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 4; ++i)
      predictedRotation[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 3; ++i)
      inertialPosition[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 4; ++i)
      inertialRotation[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 3; ++i)
      linearVelocity[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 3; ++i)
      prevLinearVelocity[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 3; ++i)
      angularVelocity[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 3; ++i)
      prevPosition[i] = nextFloat();
    for (physx::PxU32 i = 0; i < 4; ++i)
      prevRotation[i] = nextFloat();
    for (physx::PxU32 column = 0; column < 3; ++column)
      for (physx::PxU32 row = 0; row < 3; ++row)
        invInertiaWorld[column][row] = nextFloat();
    invMass = nextFloat();
    linearDamping = nextFloat();
    angularDampingBody = nextFloat();
    maxLinearVelocitySq = nextFloat();
    maxAngularVelocitySq = nextFloat();
    gravityScale = nextFloat();
    PX_ASSERT(field == floatFieldCount);

    nodeIndex = uintStorage + count * 0;
    lockFlags = uintStorage + count * 1;
    colorGroup = uintStorage + count * 2;
    numConstraints = uintStorage + count * 3;
  }

  inline void deallocateData(physx::PxAllocatorCallback &allocator) {
    if (floatStorage)
      allocator.deallocate(floatStorage);
    if (uintStorage)
      allocator.deallocate(uintStorage);
    floatStorage = nullptr;
    uintStorage = nullptr;
    clearPointers();
    numBodies = 0;
    capacity = 0;
  }

  inline void copyFromAoS(const AvbdSolverBody *bodies, physx::PxU32 count) {
    PX_ASSERT(count <= capacity);
    PX_ASSERT(floatStorage && uintStorage);
    numBodies = count;
    for (physx::PxU32 i = 0; i < count; ++i) {
      const AvbdSolverBody &body = bodies[i];
      position[0][i] = body.position.x;
      position[1][i] = body.position.y;
      position[2][i] = body.position.z;
      rotation[0][i] = body.rotation.x;
      rotation[1][i] = body.rotation.y;
      rotation[2][i] = body.rotation.z;
      rotation[3][i] = body.rotation.w;
      predictedPosition[0][i] = body.predictedPosition.x;
      predictedPosition[1][i] = body.predictedPosition.y;
      predictedPosition[2][i] = body.predictedPosition.z;
      predictedRotation[0][i] = body.predictedRotation.x;
      predictedRotation[1][i] = body.predictedRotation.y;
      predictedRotation[2][i] = body.predictedRotation.z;
      predictedRotation[3][i] = body.predictedRotation.w;
      inertialPosition[0][i] = body.inertialPosition.x;
      inertialPosition[1][i] = body.inertialPosition.y;
      inertialPosition[2][i] = body.inertialPosition.z;
      inertialRotation[0][i] = body.inertialRotation.x;
      inertialRotation[1][i] = body.inertialRotation.y;
      inertialRotation[2][i] = body.inertialRotation.z;
      inertialRotation[3][i] = body.inertialRotation.w;
      linearVelocity[0][i] = body.linearVelocity.x;
      linearVelocity[1][i] = body.linearVelocity.y;
      linearVelocity[2][i] = body.linearVelocity.z;
      prevLinearVelocity[0][i] = body.prevLinearVelocity.x;
      prevLinearVelocity[1][i] = body.prevLinearVelocity.y;
      prevLinearVelocity[2][i] = body.prevLinearVelocity.z;
      angularVelocity[0][i] = body.angularVelocity.x;
      angularVelocity[1][i] = body.angularVelocity.y;
      angularVelocity[2][i] = body.angularVelocity.z;
      prevPosition[0][i] = body.prevPosition.x;
      prevPosition[1][i] = body.prevPosition.y;
      prevPosition[2][i] = body.prevPosition.z;
      prevRotation[0][i] = body.prevRotation.x;
      prevRotation[1][i] = body.prevRotation.y;
      prevRotation[2][i] = body.prevRotation.z;
      prevRotation[3][i] = body.prevRotation.w;
      invInertiaWorld[0][0][i] = body.invInertiaWorld.column0.x;
      invInertiaWorld[0][1][i] = body.invInertiaWorld.column0.y;
      invInertiaWorld[0][2][i] = body.invInertiaWorld.column0.z;
      invInertiaWorld[1][0][i] = body.invInertiaWorld.column1.x;
      invInertiaWorld[1][1][i] = body.invInertiaWorld.column1.y;
      invInertiaWorld[1][2][i] = body.invInertiaWorld.column1.z;
      invInertiaWorld[2][0][i] = body.invInertiaWorld.column2.x;
      invInertiaWorld[2][1][i] = body.invInertiaWorld.column2.y;
      invInertiaWorld[2][2][i] = body.invInertiaWorld.column2.z;
      invMass[i] = body.invMass;
      linearDamping[i] = body.linearDamping;
      angularDampingBody[i] = body.angularDampingBody;
      maxLinearVelocitySq[i] = body.maxLinearVelocitySq;
      maxAngularVelocitySq[i] = body.maxAngularVelocitySq;
      gravityScale[i] = body.gravityScale;
      nodeIndex[i] = body.nodeIndex;
      lockFlags[i] = body.lockFlags;
      colorGroup[i] = body.colorGroup;
      numConstraints[i] = body.numConstraints;
    }
  }

  inline void copyToAoS(AvbdSolverBody *bodies, physx::PxU32 count) const {
    PX_ASSERT(count <= numBodies);
    PX_ASSERT(floatStorage && uintStorage);
    for (physx::PxU32 i = 0; i < count; ++i) {
      AvbdSolverBody &body = bodies[i];
      body.position = physx::PxVec3(position[0][i], position[1][i], position[2][i]);
      body.rotation = physx::PxQuat(rotation[0][i], rotation[1][i], rotation[2][i], rotation[3][i]);
      body.predictedPosition = physx::PxVec3(predictedPosition[0][i], predictedPosition[1][i], predictedPosition[2][i]);
      body.predictedRotation = physx::PxQuat(predictedRotation[0][i], predictedRotation[1][i], predictedRotation[2][i], predictedRotation[3][i]);
      body.inertialPosition = physx::PxVec3(inertialPosition[0][i], inertialPosition[1][i], inertialPosition[2][i]);
      body.inertialRotation = physx::PxQuat(inertialRotation[0][i], inertialRotation[1][i], inertialRotation[2][i], inertialRotation[3][i]);
      body.linearVelocity = physx::PxVec3(linearVelocity[0][i], linearVelocity[1][i], linearVelocity[2][i]);
      body.prevLinearVelocity = physx::PxVec3(prevLinearVelocity[0][i], prevLinearVelocity[1][i], prevLinearVelocity[2][i]);
      body.angularVelocity = physx::PxVec3(angularVelocity[0][i], angularVelocity[1][i], angularVelocity[2][i]);
      body.prevPosition = physx::PxVec3(prevPosition[0][i], prevPosition[1][i], prevPosition[2][i]);
      body.prevRotation = physx::PxQuat(prevRotation[0][i], prevRotation[1][i], prevRotation[2][i], prevRotation[3][i]);
      body.invInertiaWorld.column0 = physx::PxVec3(invInertiaWorld[0][0][i], invInertiaWorld[0][1][i], invInertiaWorld[0][2][i]);
      body.invInertiaWorld.column1 = physx::PxVec3(invInertiaWorld[1][0][i], invInertiaWorld[1][1][i], invInertiaWorld[1][2][i]);
      body.invInertiaWorld.column2 = physx::PxVec3(invInertiaWorld[2][0][i], invInertiaWorld[2][1][i], invInertiaWorld[2][2][i]);
      body.invMass = invMass[i];
      body.linearDamping = linearDamping[i];
      body.angularDampingBody = angularDampingBody[i];
      body.maxLinearVelocitySq = maxLinearVelocitySq[i];
      body.maxAngularVelocitySq = maxAngularVelocitySq[i];
      body.gravityScale = gravityScale[i];
      body.nodeIndex = nodeIndex[i];
      body.lockFlags = lockFlags[i];
      body.colorGroup = colorGroup[i];
      body.numConstraints = numConstraints[i];
    }
  }
};

/**
 * Non-owning island range into producer-owned hot SoA storage.
 *
 * This is intentionally a compact descriptor (storage pointer, base, count),
 * not a second array of field pointers.  Preparation may hand one range to a
 * dependency-wave producer after it has established disjoint ownership; the
 * solver must not create this range by copying AoS state.  The range is not
 * wired into the live solver until the producer differential and writeback
 * gates are complete.
 */
struct AvbdSolverBodyHotSoARange {
  AvbdSolverBodyHotSoA *storage;
  physx::PxU32 base;
  physx::PxU32 count;

  PX_FORCE_INLINE AvbdSolverBodyHotSoARange()
      : storage(nullptr), base(0), count(0) {}

  PX_FORCE_INLINE AvbdSolverBodyHotSoARange(AvbdSolverBodyHotSoA &owner,
                                            physx::PxU32 first,
                                            physx::PxU32 length)
      : storage(&owner), base(first), count(length) {
    PX_ASSERT(first <= owner.numBodies);
    PX_ASSERT(length <= owner.numBodies - first);
  }

  PX_FORCE_INLINE bool isBound() const {
    return storage && storage->floatStorage && storage->uintStorage &&
           base <= storage->numBodies && count <= storage->numBodies - base;
  }

  PX_FORCE_INLINE physx::PxReal *floatField(
      AvbdSolverBodyHotSoA::FloatField field) {
    PX_ASSERT(isBound());
    PX_ASSERT(static_cast<physx::PxU32>(field) <
              AvbdSolverBodyHotSoA::eFloatFieldCount);
    return storage->floatStorage +
           storage->capacity * static_cast<physx::PxU32>(field) + base;
  }

  PX_FORCE_INLINE const physx::PxReal *floatField(
      AvbdSolverBodyHotSoA::FloatField field) const {
    PX_ASSERT(isBound());
    PX_ASSERT(static_cast<physx::PxU32>(field) <
              AvbdSolverBodyHotSoA::eFloatFieldCount);
    return storage->floatStorage +
           storage->capacity * static_cast<physx::PxU32>(field) + base;
  }

  PX_FORCE_INLINE physx::PxU32 *uintField(
      AvbdSolverBodyHotSoA::UintField field) {
    PX_ASSERT(isBound());
    PX_ASSERT(static_cast<physx::PxU32>(field) <
              AvbdSolverBodyHotSoA::eUintFieldCount);
    return storage->uintStorage +
           storage->capacity * static_cast<physx::PxU32>(field) + base;
  }

  PX_FORCE_INLINE const physx::PxU32 *uintField(
      AvbdSolverBodyHotSoA::UintField field) const {
    PX_ASSERT(isBound());
    PX_ASSERT(static_cast<physx::PxU32>(field) <
              AvbdSolverBodyHotSoA::eUintFieldCount);
    return storage->uintStorage +
           storage->capacity * static_cast<physx::PxU32>(field) + base;
  }
};

} // namespace Dy

} // namespace physx

#pragma warning(pop)

#endif // DY_AVBD_SOLVER_BODY_H
