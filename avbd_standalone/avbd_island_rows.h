#pragma once

#include "avbd_d6_core.h"
#include "avbd_island_pcg.h"
#include "avbd_types.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace AvbdRef {

struct IslandBodyMap {
  std::vector<int32_t> bodyToSlot;
  std::vector<uint32_t> slotToBody;

  int32_t slot(uint32_t bodyId) const {
    return bodyId < bodyToSlot.size() ? bodyToSlot[bodyId] : -1;
  }
};

inline IslandBodyMap buildIslandBodyMap(uint32_t bodyCount) {
  IslandBodyMap result;
  result.bodyToSlot.assign(bodyCount, -1);
  result.slotToBody.reserve(bodyCount);
  for (uint32_t bodyId = 0; bodyId < bodyCount; ++bodyId) {
    result.bodyToSlot[bodyId] =
        static_cast<int32_t>(result.slotToBody.size());
    result.slotToBody.push_back(bodyId);
  }
  return result;
}

inline bool isFixedD6RowSet(const D6Joint &joint) {
  return joint.linearMotion == 0 && joint.angularMotion == 0 &&
         joint.driveFlags == 0 && joint.coneAngleLimit <= 0.0f &&
         !joint.motorEnabled;
}

/** Three bilateral anchor rows with all angular degrees of freedom free. */
inline bool isSphericalD6RowSet(const D6Joint &joint) {
  return joint.linearMotion == 0 && joint.angularMotion == 0x2A &&
         joint.driveFlags == 0 && joint.coneAngleLimit <= 0.0f &&
         !joint.motorEnabled;
}

/** Locked anchor plus two swing-alignment rows; twist is FREE or LIMITED. */
inline bool isRevoluteD6RowSet(const D6Joint &joint) {
  const uint32_t twist = joint.getAngularMotion(0);
  return joint.linearMotion == 0 && (twist == 1 || twist == 2) &&
         joint.getAngularMotion(1) == 0 &&
         joint.getAngularMotion(2) == 0 && joint.driveFlags == 0 &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

/** Locked orientation plus a FREE or LIMITED joint-frame X translation. */
inline bool isPrismaticD6RowSet(const D6Joint &joint) {
  const uint32_t slide = joint.getLinearMotion(0);
  return joint.angularMotion == 0 && (slide == 1 || slide == 2) &&
         joint.getLinearMotion(1) == 0 &&
         joint.getLinearMotion(2) == 0 && joint.driveFlags == 0 &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

/** One force-mode X velocity drive on an otherwise free D6. */
inline bool isLinearXVelocityDriveD6RowSet(const D6Joint &joint) {
  return joint.linearMotion == 0x2A && joint.angularMotion == 0x2A &&
         joint.driveFlags == 0x01 && joint.driveAccelerationFlags == 0 &&
         joint.linearDriveDamping.x > 0.0f &&
         std::isfinite(joint.linearDriveDamping.x) &&
         joint.driveLinearForce.x > 0.0f &&
         std::isfinite(joint.driveLinearForce.x) &&
         std::isfinite(joint.driveLinearVelocity.x) &&
         std::isfinite(joint.lambdaDriveLinear.x) &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

/** Acceleration-mode counterpart of the isolated linear-X velocity drive. */
inline bool isLinearXAccelerationVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return joint.linearMotion == 0x2A && joint.angularMotion == 0x2A &&
         joint.driveFlags == 0x01 &&
         joint.driveAccelerationFlags == 0x01 &&
         joint.linearDriveDamping.x > 0.0f &&
         std::isfinite(joint.linearDriveDamping.x) &&
         joint.driveLinearForce.x > 0.0f &&
         std::isfinite(joint.driveLinearForce.x) &&
         std::isfinite(joint.driveLinearVelocity.x) &&
         std::isfinite(joint.lambdaDriveLinear.x) &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

inline bool isSupportedLinearXVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isLinearXVelocityDriveD6RowSet(joint) ||
         isLinearXAccelerationVelocityDriveD6RowSet(joint);
}

inline bool isAngularAxisVelocityDriveD6RowSet(
    const D6Joint &joint, uint32_t driveBit, int axisIndex,
    bool acceleration) {
  const float damping = (&joint.angularDriveDamping.x)[axisIndex];
  const float forceLimit = (&joint.driveAngularForce.x)[axisIndex];
  const float targetVelocity =
      (&joint.driveAngularVelocity.x)[axisIndex];
  const float lambda = (&joint.lambdaDriveAngular.x)[axisIndex];
  return joint.linearMotion == 0x2A && joint.angularMotion == 0x2A &&
         joint.driveFlags == driveBit &&
         joint.driveAccelerationFlags == (acceleration ? driveBit : 0) &&
         damping > 0.0f && std::isfinite(damping) &&
         forceLimit > 0.0f && std::isfinite(forceLimit) &&
         std::isfinite(targetVelocity) && std::isfinite(lambda) &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

inline bool isTwistVelocityDriveD6RowSet(const D6Joint &joint) {
  return isAngularAxisVelocityDriveD6RowSet(joint, 0x10, 0, false);
}

inline bool isTwistAccelerationVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isAngularAxisVelocityDriveD6RowSet(joint, 0x10, 0, true);
}

inline bool isSupportedTwistVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isTwistVelocityDriveD6RowSet(joint) ||
         isTwistAccelerationVelocityDriveD6RowSet(joint);
}

inline bool isSwing1VelocityDriveD6RowSet(const D6Joint &joint) {
  return isAngularAxisVelocityDriveD6RowSet(joint, 0x40, 1, false);
}

inline bool isSwing1AccelerationVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isAngularAxisVelocityDriveD6RowSet(joint, 0x40, 1, true);
}

inline bool isSupportedSwing1VelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isSwing1VelocityDriveD6RowSet(joint) ||
         isSwing1AccelerationVelocityDriveD6RowSet(joint);
}

inline bool isSwing2VelocityDriveD6RowSet(const D6Joint &joint) {
  return isAngularAxisVelocityDriveD6RowSet(joint, 0x80, 2, false);
}

inline bool isSwing2AccelerationVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isAngularAxisVelocityDriveD6RowSet(joint, 0x80, 2, true);
}

inline bool isSupportedSwing2VelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isSwing2VelocityDriveD6RowSet(joint) ||
         isSwing2AccelerationVelocityDriveD6RowSet(joint);
}

inline bool isSupportedSingleAxisAngularVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isSupportedTwistVelocityDriveD6RowSet(joint) ||
         isSupportedSwing1VelocityDriveD6RowSet(joint) ||
         isSupportedSwing2VelocityDriveD6RowSet(joint);
}

inline int getSupportedSingleAxisAngularVelocityDriveIndex(
    const D6Joint &joint) {
  if (isSupportedTwistVelocityDriveD6RowSet(joint))
    return 0;
  if (isSupportedSwing1VelocityDriveD6RowSet(joint))
    return 1;
  if (isSupportedSwing2VelocityDriveD6RowSet(joint))
    return 2;
  return -1;
}

inline bool isSlerpVelocityDriveD6RowSet(const D6Joint &joint) {
  return joint.linearMotion == 0x2A && joint.angularMotion == 0x2A &&
         joint.driveFlags == 0x20 && joint.driveAccelerationFlags == 0 &&
         joint.angularDriveDamping.z > 0.0f &&
         std::isfinite(joint.angularDriveDamping.z) &&
         joint.driveAngularForce.z > 0.0f &&
         std::isfinite(joint.driveAngularForce.z) &&
         std::isfinite(joint.driveAngularVelocity.x) &&
         std::isfinite(joint.driveAngularVelocity.y) &&
         std::isfinite(joint.driveAngularVelocity.z) &&
         std::isfinite(joint.lambdaDriveAngular.x) &&
         std::isfinite(joint.lambdaDriveAngular.y) &&
         std::isfinite(joint.lambdaDriveAngular.z) &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

inline bool isSlerpAccelerationVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return joint.linearMotion == 0x2A && joint.angularMotion == 0x2A &&
         joint.driveFlags == 0x20 &&
         joint.driveAccelerationFlags == 0x20 &&
         joint.angularDriveDamping.z > 0.0f &&
         std::isfinite(joint.angularDriveDamping.z) &&
         joint.driveAngularForce.z > 0.0f &&
         std::isfinite(joint.driveAngularForce.z) &&
         std::isfinite(joint.driveAngularVelocity.x) &&
         std::isfinite(joint.driveAngularVelocity.y) &&
         std::isfinite(joint.driveAngularVelocity.z) &&
         std::isfinite(joint.lambdaDriveAngular.x) &&
         std::isfinite(joint.lambdaDriveAngular.y) &&
         std::isfinite(joint.lambdaDriveAngular.z) &&
         joint.coneAngleLimit <= 0.0f && !joint.motorEnabled;
}

inline bool isSupportedSlerpVelocityDriveD6RowSet(
    const D6Joint &joint) {
  return isSlerpVelocityDriveD6RowSet(joint) ||
         isSlerpAccelerationVelocityDriveD6RowSet(joint);
}

inline float computeLinearDriveEffectiveMass(
    const Body *endpointA, const Body *endpointB, const Vec3 &rA,
    const Vec3 &rB, const Vec3 &axis) {
  float unitResponse = 0.0f;
  if (endpointA && endpointA->mass > 0.0f) {
    const Vec3 angular = rA.cross(axis);
    unitResponse += endpointA->invMass +
                    (endpointA->invInertiaWorld * angular).dot(angular);
  }
  if (endpointB && endpointB->mass > 0.0f) {
    const Vec3 angular = rB.cross(axis);
    unitResponse += endpointB->invMass +
                    (endpointB->invInertiaWorld * angular).dot(angular);
  }
  return unitResponse > 1e-8f ? 1.0f / unitResponse : 0.0f;
}

inline float computeLinearXVelocityDrivePenalty(
    const D6Joint &joint, const Body *endpointA, const Body *endpointB,
    const Vec3 &rA, const Vec3 &rB, const Vec3 &axis, float dt) {
  if (!(dt > 0.0f))
    return 0.0f;
  // The row violation is an anchor displacement over this step.  A
  // force-mode damper therefore contributes damping/dt.  For an acceleration
  // spring, PhysX scales that force coefficient by the reciprocal row
  // response (the effective mass).  The position solve already supplies the
  // implicit 1/(1 + dt*damping) denominator through M/dt^2 + J^T*rho*J; an
  // additional implicitScale here would count that denominator twice.
  float penalty = joint.linearDriveDamping.x / dt;
  if (isLinearXAccelerationVelocityDriveD6RowSet(joint)) {
    const float effectiveMass =
        computeLinearDriveEffectiveMass(endpointA, endpointB, rA, rB, axis);
    penalty *= effectiveMass;
  }
  return std::isfinite(penalty) && penalty > 0.0f ? penalty : 0.0f;
}

inline float computeAngularDriveEffectiveInertia(
    const Body *endpointA, const Body *endpointB, const Vec3 &axis) {
  float unitResponse = 0.0f;
  if (endpointA && endpointA->mass > 0.0f)
    unitResponse += (endpointA->invInertiaWorld * axis).dot(axis);
  if (endpointB && endpointB->mass > 0.0f)
    unitResponse += (endpointB->invInertiaWorld * axis).dot(axis);
  return unitResponse > 1e-8f ? 1.0f / unitResponse : 0.0f;
}

inline float computeAngularAxisVelocityDrivePenalty(
    const D6Joint &joint, const Body *endpointA, const Body *endpointB,
    const Vec3 &axis, int axisIndex, bool acceleration, float dt) {
  if (!(dt > 0.0f))
    return 0.0f;
  const float damping = (&joint.angularDriveDamping.x)[axisIndex];
  float penalty = damping / (dt * dt);
  if (acceleration) {
    const float effectiveInertia =
        computeAngularDriveEffectiveInertia(endpointA, endpointB, axis);
    const float implicitScale =
        1.0f / (1.0f + dt * damping);
    penalty *= effectiveInertia * implicitScale;
  }
  return std::isfinite(penalty) && penalty > 0.0f ? penalty : 0.0f;
}

inline float computeTwistVelocityDrivePenalty(
    const D6Joint &joint, const Body *endpointA, const Body *endpointB,
    const Vec3 &axis, float dt) {
  return computeAngularAxisVelocityDrivePenalty(
      joint, endpointA, endpointB, axis, 0,
      isTwistAccelerationVelocityDriveD6RowSet(joint), dt);
}

inline float computeSlerpVelocityDrivePenalty(
    const D6Joint &joint, const Body *endpointA, const Body *endpointB,
    const Vec3 &worldAxis, float dt) {
  if (!(dt > 0.0f))
    return 0.0f;
  const float damping = joint.angularDriveDamping.z;
  float penalty = damping / (dt * dt);
  if (isSlerpAccelerationVelocityDriveD6RowSet(joint)) {
    const float effectiveInertia =
        computeAngularDriveEffectiveInertia(endpointA, endpointB,
                                            worldAxis);
    penalty *= effectiveInertia / (1.0f + dt * damping);
  }
  return std::isfinite(penalty) && penalty > 0.0f ? penalty : 0.0f;
}

inline Vec3 computeWorldRotationDelta(const Quat &current,
                                      const Quat &initial) {
  Quat delta = (current * initial.conjugate()).normalized();
  if (delta.w < 0.0f)
    delta = delta * -1.0f;
  const Vec3 imaginary(delta.x, delta.y, delta.z);
  const float imaginaryLength = imaginary.length();
  if (!(imaginaryLength > 1e-10f))
    return imaginary * 2.0f;
  const float angle =
      2.0f * std::atan2(imaginaryLength, delta.w);
  return imaginary * (angle / imaginaryLength);
}

inline float updateClampedLinearDriveDual(float lambda, float violation,
                                          float rhoDual, float forceLimit,
                                          float lambdaDecay) {
  const float updated = lambda * lambdaDecay + rhoDual * violation;
  return std::max(-forceLimit, std::min(forceLimit, updated));
}

inline Quat computeD6SymmetricMidFrame(const Quat &frameA,
                                       const Quat &frameB) {
  Quat alignedB = frameB;
  const float frameDot = frameA.w * frameB.w + frameA.x * frameB.x +
                         frameA.y * frameB.y + frameA.z * frameB.z;
  if (frameDot < 0.0f)
    alignedB = alignedB * -1.0f;
  return Quat(frameA.w + alignedB.w, frameA.x + alignedB.x,
              frameA.y + alignedB.y, frameA.z + alignedB.z)
      .normalized();
}

inline Vec3 computeD6SymmetricAngularError(const Quat &frameA,
                                           const Quat &frameB) {
  Quat relativeFrame = (frameA * frameB.conjugate()).normalized();
  if (relativeFrame.w < 0.0f)
    relativeFrame = relativeFrame * -1.0f;
  const Vec3 imaginary(relativeFrame.x, relativeFrame.y, relativeFrame.z);
  const float imaginaryLength = imaginary.length();
  if (!(imaginaryLength > 1e-10f))
    return imaginary * 2.0f;
  const float angle =
      2.0f * std::atan2(imaginaryLength, relativeFrame.w);
  return imaginary * (angle / imaginaryLength);
}

inline bool buildRevoluteMidAxisBasis(const Vec3 &twistA,
                                      const Vec3 &twistB, Vec3 &midAxis,
                                      Vec3 &perpendicular1,
                                      Vec3 &perpendicular2) {
  midAxis = twistA + twistB;
  float midAxisLength = midAxis.length();
  if (!(midAxisLength > 1e-6f)) {
    // The cross-product objective treats antiparallel hinge axes as aligned.
    // Canonicalize the surviving direction so the fallback basis is still
    // independent of endpoint order.
    midAxis = twistA;
    if (midAxis.x < -1e-6f ||
        (std::fabs(midAxis.x) <= 1e-6f && midAxis.y < -1e-6f) ||
        (std::fabs(midAxis.x) <= 1e-6f &&
         std::fabs(midAxis.y) <= 1e-6f && midAxis.z < 0.0f))
      midAxis = -midAxis;
    midAxisLength = midAxis.length();
  }
  if (!(midAxisLength > 1e-6f))
    return false;
  midAxis = midAxis * (1.0f / midAxisLength);
  const Vec3 axes[2] = {Vec3(1.0f, 0.0f, 0.0f),
                        Vec3(0.0f, 1.0f, 0.0f)};
  perpendicular1 = std::fabs(midAxis.x) < 0.9f
                       ? midAxis.cross(axes[0])
                       : midAxis.cross(axes[1]);
  const float perpendicularLength = perpendicular1.length();
  if (!(perpendicularLength > 1e-6f))
    return false;
  perpendicular1 = perpendicular1 * (1.0f / perpendicularLength);
  perpendicular2 = midAxis.cross(perpendicular1);
  const float perpendicular2Length = perpendicular2.length();
  if (!(perpendicular2Length > 1e-6f))
    return false;
  perpendicular2 = perpendicular2 * (1.0f / perpendicular2Length);
  return true;
}

inline float computeRevoluteSymmetricTwistError(const Quat &frameA,
                                                const Quat &frameB) {
  Quat relativeFrame = (frameA.conjugate() * frameB).normalized();
  if (relativeFrame.w < 0.0f)
    relativeFrame = relativeFrame * -1.0f;
  return -2.0f * std::atan2(relativeFrame.x, relativeFrame.w);
}

/**
 * Emit the coupled linear anchor rows of a spherical D6 constraint.
 *
 * This is the exact linear subset shared with emitFixedD6IslandRows.  Keeping
 * it endpoint-centric preserves the off-diagonal body coupling that is lost
 * by two independent per-body 6x6 solves.
 */
inline bool emitSphericalD6IslandRows(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (!isSphericalD6RowSet(joint) || !(dt > 0.0f))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Vec3 rA = endpointA ? rotationA.rotate(joint.anchorA) : Vec3();
  const Vec3 rB = endpointB ? rotationB.rotate(joint.anchorB) : Vec3();
  const Vec3 worldAnchorA =
      endpointA ? endpointA->position + rA : joint.anchorA;
  const Vec3 worldAnchorB =
      endpointB ? endpointB->position + rB : joint.anchorB;
  const Vec3 linearError = worldAnchorA - worldAnchorB;
  const float dt2 = dt * dt;
  const float massA = dynamicA ? endpointA->mass : 0.0f;
  const float massB = dynamicB ? endpointB->mass : 0.0f;
  const float penalty =
      std::max(joint.rho, std::max(massA, massB) / dt2);
  const Vec3 axes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                        Vec3(0.0f, 1.0f, 0.0f),
                        Vec3(0.0f, 0.0f, 1.0f)};

  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const Vec3 axis = axes[axisIndex];
    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = jointIndex;
    row.rowSlot = static_cast<uint16_t>(axisIndex);
    row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
    row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
    row.jacobianA = dynamicA ? Vec6(axis, rA.cross(axis)) : Vec6();
    row.jacobianB = dynamicB ? Vec6(-axis, -rB.cross(axis)) : Vec6();
    row.violation = linearError.dot(axis);
    row.penalty = penalty;
    row.force = penalty * row.violation +
                (&joint.lambdaLinear.x)[axisIndex];
    row.internalTranslationInvariant = dynamicA && dynamicB;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
  }
  return true;
}

/**
 * Emit the complete frozen positional objective of an undriven revolute D6.
 * The two swing rows use the same cross-axis linearization as addD6Contribution;
 * an optional twist-limit row is emitted only for its frozen active side.
 */
inline bool emitRevoluteD6IslandRows(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (!isRevoluteD6RowSet(joint) || !(dt > 0.0f))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Vec3 rA = endpointA ? rotationA.rotate(joint.anchorA) : Vec3();
  const Vec3 rB = endpointB ? rotationB.rotate(joint.anchorB) : Vec3();
  const Vec3 worldAnchorA =
      endpointA ? endpointA->position + rA : joint.anchorA;
  const Vec3 worldAnchorB =
      endpointB ? endpointB->position + rB : joint.anchorB;
  const Vec3 linearError = worldAnchorA - worldAnchorB;
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Quat frameB =
      (endpointB ? rotationB * joint.localFrameB : joint.localFrameB)
          .normalized();
  const float dt2 = dt * dt;
  const float massA = dynamicA ? endpointA->mass : 0.0f;
  const float massB = dynamicB ? endpointB->mass : 0.0f;
  const float penalty =
      std::max(joint.rho, std::max(massA, massB) / dt2);
  const Vec3 axes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                        Vec3(0.0f, 1.0f, 0.0f),
                        Vec3(0.0f, 0.0f, 1.0f)};
  const auto emitRow = [&](uint16_t rowSlot, const Vec6 &jacobianA,
                           const Vec6 &jacobianB, float violation,
                           float force, bool translationInvariant) {
    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = jointIndex;
    row.rowSlot = rowSlot;
    row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
    row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
    row.jacobianA = dynamicA ? jacobianA : Vec6();
    row.jacobianB = dynamicB ? jacobianB : Vec6();
    row.violation = violation;
    row.penalty = penalty;
    row.force = force;
    row.internalTranslationInvariant = translationInvariant;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
    return true;
  };

  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const Vec3 axis = axes[axisIndex];
    const float violation = linearError.dot(axis);
    const float force =
        penalty * violation + (&joint.lambdaLinear.x)[axisIndex];
    if (!emitRow(static_cast<uint16_t>(axisIndex),
                 Vec6(axis, rA.cross(axis)),
                 Vec6(-axis, -rB.cross(axis)), violation, force,
                 dynamicA && dynamicB))
      return false;
  }

  const Vec3 twistA = frameA.rotate(axes[0]);
  const Vec3 twistB = frameB.rotate(axes[0]);
  const Vec3 axisViolation = twistA.cross(twistB);

  // Build the two scalar swing rows in a basis around the normalized
  // mid-axis.  Unlike a basis built from endpoint A, the mid-axis is unchanged
  // when the joint actors are exchanged.  Use the exact first derivative of
  // C = twistA x twistB so that exchanging A/B negates the complete frozen
  // row (C and both Jacobians), preserving J^T J and J^T C.
  Vec3 midAxis, perpendicular1, perpendicular2;
  if (!buildRevoluteMidAxisBasis(twistA, twistB, midAxis, perpendicular1,
                                  perpendicular2))
    return false;
  const Vec3 swingAxes[2] = {perpendicular1, perpendicular2};
  const float twistDot = twistA.dot(twistB);
  for (int swing = 0; swing < 2; ++swing) {
    const Vec3 axis = swingAxes[swing];
    const float violation = axisViolation.dot(axis);
    const float force =
        penalty * violation + (&joint.lambdaAngular.x)[1 + swing];
    const Vec3 angularJacobianA =
        twistB * twistA.dot(axis) - axis * twistDot;
    const Vec3 angularJacobianB =
        axis * twistDot - twistA * twistB.dot(axis);
    if (!emitRow(static_cast<uint16_t>(4 + swing),
                 Vec6(Vec3(), angularJacobianA),
                 Vec6(Vec3(), angularJacobianB), violation, force, true))
      return false;
  }

  if (joint.getAngularMotion(0) == 1) {
    // Extract twist in joint coordinates.  frameA^-1*frameB becomes its exact
    // quaternion inverse when the actors are exchanged, so the signed twist
    // below changes sign even when a finite swing error is also present.  The
    // leading minus preserves computeAngularError's established A-vs-B sign.
    const float angularError =
        computeRevoluteSymmetricTwistError(frameA, frameB);
    const float violation = computeAngularLimitViolation(
        angularError, joint.angularLimitLower[0],
        joint.angularLimitUpper[0]);
    const float lambda = joint.lambdaLimitAngular[0];
    const float rawForce = penalty * violation + lambda;
    float force = 0.0f;
    if (joint.angularLimitLower[0] < joint.angularLimitUpper[0]) {
      if (violation > 0.0f || lambda > 0.0f)
        force = std::max(0.0f, rawForce);
      else if (violation < 0.0f || lambda < 0.0f)
        force = std::min(0.0f, rawForce);
    } else {
      force = rawForce;
    }
    if (force != 0.0f &&
        !emitRow(3, Vec6(Vec3(), midAxis), Vec6(Vec3(), -midAxis),
                 violation, force, true))
      return false;
  }
  return true;
}

/**
 * Emit the complete frozen positional objective of an undriven prismatic D6.
 *
 * A LIMITED slide emits a unilateral row only for its frozen active side; the
 * legacy zero-gradient "baseline stiffness" is deliberately excluded because
 * it hardens a coordinate that is physically free inside the limit interval.
 * The slide frame is the endpoint-exchange-invariant quaternion midpoint; the
 * three locked angular rows use a world rotation vector, so swapping A/B
 * negates every complete row without changing the island equation.
 */
inline bool emitPrismaticD6IslandRows(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (!isPrismaticD6RowSet(joint) || !(dt > 0.0f))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Vec3 rA = endpointA ? rotationA.rotate(joint.anchorA) : Vec3();
  const Vec3 rB = endpointB ? rotationB.rotate(joint.anchorB) : Vec3();
  const Vec3 worldAnchorA =
      endpointA ? endpointA->position + rA : joint.anchorA;
  const Vec3 worldAnchorB =
      endpointB ? endpointB->position + rB : joint.anchorB;
  const Vec3 linearError = worldAnchorA - worldAnchorB;
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Quat frameB =
      (endpointB ? rotationB * joint.localFrameB : joint.localFrameB)
          .normalized();
  const Quat midFrame = computeD6SymmetricMidFrame(frameA, frameB);
  const Vec3 localAxes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f),
                             Vec3(0.0f, 0.0f, 1.0f)};
  const Vec3 axes[3] = {midFrame.rotate(localAxes[0]),
                        midFrame.rotate(localAxes[1]),
                        midFrame.rotate(localAxes[2])};
  const Vec3 angularError = computeD6SymmetricAngularError(frameA, frameB);
  const float dt2 = dt * dt;
  const float massA = dynamicA ? endpointA->mass : 0.0f;
  const float massB = dynamicB ? endpointB->mass : 0.0f;
  const float penalty =
      std::max(joint.rho, std::max(massA, massB) / dt2);
  const auto emitRow = [&](uint16_t rowSlot, const Vec6 &jacobianA,
                           const Vec6 &jacobianB, float violation,
                           float force, bool translationInvariant) {
    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = jointIndex;
    row.rowSlot = rowSlot;
    row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
    row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
    row.jacobianA = dynamicA ? jacobianA : Vec6();
    row.jacobianB = dynamicB ? jacobianB : Vec6();
    row.violation = violation;
    row.penalty = penalty;
    row.force = force;
    row.internalTranslationInvariant = translationInvariant;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
    return true;
  };

  for (int axisIndex = 1; axisIndex < 3; ++axisIndex) {
    const Vec3 axis = axes[axisIndex];
    const float violation = linearError.dot(axis);
    const float force =
        penalty * violation + (&joint.lambdaLinear.x)[axisIndex];
    if (!emitRow(static_cast<uint16_t>(axisIndex),
                 Vec6(axis, rA.cross(axis)),
                 Vec6(-axis, -rB.cross(axis)), violation, force,
                 dynamicA && dynamicB))
      return false;
  }

  if (joint.getLinearMotion(0) == 1) {
    const Vec3 axis = axes[0];
    const float distance = -linearError.dot(axis);
    const float violation = computeAngularLimitViolation(
        distance, joint.linearLimitLower[0], joint.linearLimitUpper[0]);
    const float lambda = joint.lambdaLimitLinear[0];
    const float rawForce = penalty * violation + lambda;
    float force = 0.0f;
    if (joint.linearLimitLower[0] < joint.linearLimitUpper[0]) {
      if (violation > 0.0f || lambda > 0.0f)
        force = std::max(0.0f, rawForce);
      else if (violation < 0.0f || lambda < 0.0f)
        force = std::min(0.0f, rawForce);
    } else {
      force = rawForce;
    }
    if (force != 0.0f &&
        !emitRow(0, Vec6(-axis, rA.cross(-axis)),
                 Vec6(axis, rB.cross(axis)), violation, force,
                 dynamicA && dynamicB))
      return false;
  }

  const Vec3 worldAxes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f),
                             Vec3(0.0f, 0.0f, 1.0f)};
  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const Vec3 axis = worldAxes[axisIndex];
    const float violation = (&angularError.x)[axisIndex];
    const float force =
        penalty * violation + (&joint.lambdaAngular.x)[axisIndex];
    if (!emitRow(static_cast<uint16_t>(3 + axisIndex),
                 Vec6(Vec3(), axis), Vec6(Vec3(), -axis), violation,
                 force, true))
      return false;
  }
  return true;
}

/**
 * Emit one frozen force-mode linear X velocity-drive row.
 *
 * C is the relative world-anchor displacement over the current step minus
 * targetVelocity*dt.  This retains the full offset-anchor Jacobian instead of
 * pairing a COM-only residual with an angular gradient.  The total AL force is
 * clamped in force units.  A saturated clamp has zero generalized derivative,
 * so its frozen row contributes force but no drive Hessian.
 */
inline bool emitLinearXVelocityDriveIslandRow(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (!isSupportedLinearXVelocityDriveD6RowSet(joint) || !(dt > 0.0f))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Vec3 axis = frameA.rotate(Vec3(1.0f, 0.0f, 0.0f));
  const Vec3 rA = dynamicA ? rotationA.rotate(joint.anchorA) : Vec3();
  const Vec3 rB = dynamicB ? rotationB.rotate(joint.anchorB) : Vec3();
  const Vec3 displacementA =
      dynamicA
          ? (endpointA->position + rA) -
                (endpointA->initialPosition +
                 endpointA->initialRotation.rotate(joint.anchorA))
          : Vec3();
  const Vec3 displacementB =
      dynamicB
          ? (endpointB->position + rB) -
                (endpointB->initialPosition +
                 endpointB->initialRotation.rotate(joint.anchorB))
          : Vec3();
  const float violation =
      (displacementB - displacementA).dot(axis) -
      joint.driveLinearVelocity.x * dt;
  const float drivePenalty = computeLinearXVelocityDrivePenalty(
      joint, endpointA, endpointB, rA, rB, axis, dt);
  if (!(drivePenalty > 0.0f))
    return false;
  const float rawForce =
      drivePenalty * violation + joint.lambdaDriveLinear.x;
  const float forceLimit = joint.driveLinearForce.x;
  const float force =
      std::max(-forceLimit, std::min(forceLimit, rawForce));
  const bool saturated = std::fabs(rawForce) >= forceLimit;

  IslandPcgRow row;
  row.owner = IslandRowOwner::D6;
  row.ownerIndex = jointIndex;
  row.rowSlot = 0;
  row.activeMode = saturated ? 4 : 3;
  row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
  row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
  row.jacobianA =
      dynamicA ? Vec6(-axis, -rA.cross(axis)) : Vec6();
  row.jacobianB =
      dynamicB ? Vec6(axis, rB.cross(axis)) : Vec6();
  row.violation = violation;
  row.penalty = saturated ? 0.0f : drivePenalty;
  row.force = force;
  row.internalTranslationInvariant = dynamicA && dynamicB;
  if (!system.addRow(row))
    return false;
  emittedRows = 1;
  return true;
}

/** Emit one frozen force- or acceleration-mode TWIST/SWING axis row. */
inline bool emitSingleAxisAngularVelocityDriveIslandRow(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    int axisIndex, IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (axisIndex < 0 || axisIndex > 2 || !(dt > 0.0f))
    return false;
  const uint32_t driveBits[3] = {0x10, 0x40, 0x80};
  const uint32_t driveBit = driveBits[axisIndex];
  const bool acceleration =
      joint.driveAccelerationFlags == driveBit;
  if (!isAngularAxisVelocityDriveD6RowSet(
          joint, driveBit, axisIndex, acceleration))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Vec3 localAxes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f),
                             Vec3(0.0f, 0.0f, 1.0f)};
  const Vec3 axis = frameA.rotate(localAxes[axisIndex]);
  const Vec3 deltaA =
      dynamicA ? computeWorldRotationDelta(rotationA,
                                           endpointA->initialRotation)
               : Vec3();
  const Vec3 deltaB =
      dynamicB ? computeWorldRotationDelta(rotationB,
                                           endpointB->initialRotation)
               : Vec3();
  // PhysX TWIST/SWING convention is wA-wB=target.  Since the row Jacobian
  // uses B-A, the authored target enters C with a positive sign.
  const float violation =
      (deltaB - deltaA).dot(axis) +
      (&joint.driveAngularVelocity.x)[axisIndex] * dt;
  const float drivePenalty = computeAngularAxisVelocityDrivePenalty(
      joint, endpointA, endpointB, axis, axisIndex, acceleration, dt);
  if (!(drivePenalty > 0.0f))
    return false;
  const float rawTorque =
      drivePenalty * violation +
      (&joint.lambdaDriveAngular.x)[axisIndex];
  const float torqueLimit = (&joint.driveAngularForce.x)[axisIndex];
  const float torque =
      std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
  const bool saturated = std::fabs(rawTorque) >= torqueLimit;

  IslandPcgRow row;
  row.owner = IslandRowOwner::D6;
  row.ownerIndex = jointIndex;
  row.rowSlot = static_cast<uint16_t>(3 + axisIndex);
  row.activeMode = saturated ? 4 : 3;
  row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
  row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
  row.jacobianA = dynamicA ? Vec6(Vec3(), -axis) : Vec6();
  row.jacobianB = dynamicB ? Vec6(Vec3(), axis) : Vec6();
  row.violation = violation;
  row.penalty = saturated ? 0.0f : drivePenalty;
  row.force = torque;
  row.internalTranslationInvariant = dynamicA && dynamicB;
  if (!system.addRow(row))
    return false;
  emittedRows = 1;
  return true;
}

inline bool emitTwistVelocityDriveIslandRow(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  return emitSingleAxisAngularVelocityDriveIslandRow(
      joint, jointIndex, bodies, bodyMap, dt, 0, system, emittedRows);
}

inline bool emitSwing1VelocityDriveIslandRow(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  return emitSingleAxisAngularVelocityDriveIslandRow(
      joint, jointIndex, bodies, bodyMap, dt, 1, system, emittedRows);
}

inline bool emitSwing2VelocityDriveIslandRow(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  return emitSingleAxisAngularVelocityDriveIslandRow(
      joint, jointIndex, bodies, bodyMap, dt, 2, system, emittedRows);
}

/** Emit the three frozen world-axis rows of an isolated SLERP drive. */
inline bool emitSlerpVelocityDriveIslandRows(
    const D6Joint &joint, uint32_t jointIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap, float dt,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (!isSupportedSlerpVelocityDriveD6RowSet(joint) || !(dt > 0.0f))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Vec3 deltaA =
      dynamicA ? computeWorldRotationDelta(rotationA,
                                           endpointA->initialRotation)
               : Vec3();
  const Vec3 deltaB =
      dynamicB ? computeWorldRotationDelta(rotationB,
                                           endpointB->initialRotation)
               : Vec3();
  // PhysX SLERP convention is wB-wA=target. Unlike TWIST/SWING, the target
  // therefore subtracts from the B-A world rotation-vector residual.
  const Vec3 worldTarget =
      frameA.rotate(joint.driveAngularVelocity) * dt;
  const Vec3 violation = deltaB - deltaA - worldTarget;
  const Vec3 worldAxes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f),
                             Vec3(0.0f, 0.0f, 1.0f)};
  const float torqueLimit = joint.driveAngularForce.z;
  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const Vec3 &axis = worldAxes[axisIndex];
    const float drivePenalty = computeSlerpVelocityDrivePenalty(
        joint, endpointA, endpointB, axis, dt);
    if (!(drivePenalty > 0.0f))
      return false;
    const float componentViolation = (&violation.x)[axisIndex];
    const float rawTorque =
        drivePenalty * componentViolation +
        (&joint.lambdaDriveAngular.x)[axisIndex];
    const float torque =
        std::max(-torqueLimit, std::min(torqueLimit, rawTorque));
    const bool saturated = std::fabs(rawTorque) >= torqueLimit;

    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = jointIndex;
    row.rowSlot = static_cast<uint16_t>(3 + axisIndex);
    row.activeMode = saturated ? 4 : 3;
    row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
    row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
    row.jacobianA = dynamicA ? Vec6(Vec3(), -axis) : Vec6();
    row.jacobianB = dynamicB ? Vec6(Vec3(), axis) : Vec6();
    row.violation = componentViolation;
    row.penalty = saturated ? 0.0f : drivePenalty;
    row.force = torque;
    row.internalTranslationInvariant = dynamicA && dynamicB;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
  }
  return true;
}

/**
 * Initial row emitter slice: all six bilateral rows of a fixed D6.
 *
 * The emitter is endpoint-centric and is called once per joint. It deliberately
 * does not fall back to two per-body calls, which would lose the cross block.
 */
inline bool emitFixedD6IslandRows(const D6Joint &joint, uint32_t jointIndex,
                                  const std::vector<Body> &bodies,
                                  const IslandBodyMap &bodyMap, float dt,
                                  IslandPcgSystem &system,
                                  uint32_t &emittedRows) {
  emittedRows = 0;
  if (!isFixedD6RowSet(joint) || !(dt > 0.0f))
    return false;
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  if (!dynamicA && !dynamicB)
    return true;
  const int32_t slotA = dynamicA ? bodyMap.slot(joint.bodyA) : -1;
  const int32_t slotB = dynamicB ? bodyMap.slot(joint.bodyB) : -1;
  if ((dynamicA && slotA < 0) || (dynamicB && slotB < 0))
    return false;

  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Vec3 rA = endpointA ? rotationA.rotate(joint.anchorA) : Vec3();
  const Vec3 rB = endpointB ? rotationB.rotate(joint.anchorB) : Vec3();
  const Vec3 worldAnchorA =
      endpointA ? endpointA->position + rA : joint.anchorA;
  const Vec3 worldAnchorB =
      endpointB ? endpointB->position + rB : joint.anchorB;
  const Vec3 linearError = worldAnchorA - worldAnchorB;
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const float dt2 = dt * dt;
  const float massA = dynamicA ? endpointA->mass : 0.0f;
  const float massB = dynamicB ? endpointB->mass : 0.0f;
  const float penalty =
      std::max(joint.rho, std::max(massA, massB) / dt2);
  const Vec3 axes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                        Vec3(0.0f, 1.0f, 0.0f),
                        Vec3(0.0f, 0.0f, 1.0f)};

  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const Vec3 axis = axes[axisIndex];
    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = jointIndex;
    row.rowSlot = static_cast<uint16_t>(axisIndex);
    row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
    row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
    row.jacobianA = dynamicA ? Vec6(axis, rA.cross(axis)) : Vec6();
    row.jacobianB = dynamicB ? Vec6(-axis, -rB.cross(axis)) : Vec6();
    row.violation = linearError.dot(axis);
    row.penalty = penalty;
    row.force = penalty * row.violation +
                (&joint.lambdaLinear.x)[axisIndex];
    row.internalTranslationInvariant = dynamicA && dynamicB;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
  }

  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const Vec3 axis = frameA.rotate(axes[axisIndex]);
    IslandPcgRow row;
    row.owner = IslandRowOwner::D6;
    row.ownerIndex = jointIndex;
    row.rowSlot = static_cast<uint16_t>(3 + axisIndex);
    row.bodyA = slotA >= 0 ? uint32_t(slotA) : UINT32_MAX;
    row.bodyB = slotB >= 0 ? uint32_t(slotB) : UINT32_MAX;
    row.jacobianA = dynamicA ? Vec6(Vec3(), axis) : Vec6();
    row.jacobianB = dynamicB ? Vec6(Vec3(), -axis) : Vec6();
    row.violation = computeAngularError(rotationA, rotationB,
                                        joint.localFrameA, joint.localFrameB,
                                        axisIndex);
    row.penalty = penalty;
    row.force = penalty * row.violation +
                (&joint.lambdaAngular.x)[axisIndex];
    row.internalTranslationInvariant = dynamicA && dynamicB;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
  }
  return true;
}

struct FrozenContactIslandRowSet {
  bool active[3] = {false, false, false};
  float penalty[3] = {0.0f, 0.0f, 0.0f};
  float force[3] = {0.0f, 0.0f, 0.0f};
  uint16_t activeMode[3] = {0, 0, 0};
  /** Force-valued circular Coulomb bound for rows 1/2. */
  float tangentForceBound = 0.0f;
};

/**
 * Serialize one already-frozen contact normal/tangent row set.
 *
 * The caller owns contact generation, active-set selection, penalty/force,
 * circular-cone projection, and dual/cache sequencing. Tangents are accepted
 * only as a pair and only when their frozen force lies inside the supplied
 * force-valued Coulomb disk.
 */
inline bool emitFrozenContactIslandRows(
    const Contact &contact, uint32_t contactIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap,
    const FrozenContactIslandRowSet &rowSet,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  emittedRows = 0;
  if (!rowSet.active[0] && !rowSet.active[1] && !rowSet.active[2])
    return true;
  if (rowSet.active[1] != rowSet.active[2] ||
      (rowSet.active[1] && !rowSet.active[0]))
    return false;
  for (int row = 0; row < 3; ++row) {
    if (!rowSet.active[row])
      continue;
    if (!(rowSet.penalty[row] >= 0.0f) ||
        !std::isfinite(rowSet.penalty[row]) ||
        !std::isfinite(rowSet.force[row]) ||
        !std::isfinite(contact.C[row]))
      return false;
  }
  if (rowSet.active[1]) {
    if (!(rowSet.tangentForceBound >= 0.0f) ||
        !std::isfinite(rowSet.tangentForceBound))
      return false;
    const double tangentMagnitude =
        std::sqrt(double(rowSet.force[1]) * double(rowSet.force[1]) +
                  double(rowSet.force[2]) * double(rowSet.force[2]));
    const double tolerance =
        std::max(1e-5, double(rowSet.tangentForceBound) * 1e-5);
    if (tangentMagnitude > double(rowSet.tangentForceBound) + tolerance)
      return false;
  }
  if (contact.bodyA >= bodies.size() || bodies[contact.bodyA].mass <= 0.0f)
    return false;

  const bool dynamicB = contact.bodyB < bodies.size() &&
                        bodies[contact.bodyB].mass > 0.0f;
  const bool staticB = contact.bodyB == UINT32_MAX ||
                       (contact.bodyB < bodies.size() &&
                        bodies[contact.bodyB].mass <= 0.0f);
  if (!dynamicB && !staticB)
    return false;
  const int32_t slotA = bodyMap.slot(contact.bodyA);
  const int32_t slotB = dynamicB ? bodyMap.slot(contact.bodyB) : -1;
  if (slotA < 0 || (dynamicB && slotB < 0))
    return false;

  const Vec6 jacobianA[3] = {contact.JA, contact.JAt1, contact.JAt2};
  const Vec6 jacobianB[3] = {contact.JB, contact.JBt1, contact.JBt2};
  for (int rowIndex = 0; rowIndex < 3; ++rowIndex) {
    if (!rowSet.active[rowIndex])
      continue;
    IslandPcgRow row;
    row.owner = IslandRowOwner::Contact;
    row.ownerIndex = contactIndex;
    row.rowSlot = static_cast<uint16_t>(rowIndex);
    row.activeMode = rowSet.activeMode[rowIndex];
    row.bodyA = static_cast<uint32_t>(slotA);
    row.bodyB = dynamicB ? static_cast<uint32_t>(slotB) : UINT32_MAX;
    row.jacobianA = jacobianA[rowIndex];
    row.jacobianB = dynamicB ? jacobianB[rowIndex] : Vec6();
    row.violation = contact.C[rowIndex];
    row.penalty = rowSet.penalty[rowIndex];
    row.force = rowSet.force[rowIndex];
    row.internalTranslationInvariant = dynamicB;
    if (!system.addRow(row))
      return false;
    ++emittedRows;
  }
  return true;
}

/** Serialize one already-frozen normal; retained for the first snapshot gate. */
inline bool emitFrozenContactNormalIslandRow(
    const Contact &contact, uint32_t contactIndex,
    const std::vector<Body> &bodies, const IslandBodyMap &bodyMap,
    bool active, float penalty, float force, uint16_t activeMode,
    IslandPcgSystem &system, uint32_t &emittedRows) {
  FrozenContactIslandRowSet rowSet;
  rowSet.active[0] = active;
  rowSet.penalty[0] = penalty;
  rowSet.force[0] = force;
  rowSet.activeMode[0] = activeMode;
  return emitFrozenContactIslandRows(contact, contactIndex, bodies, bodyMap,
                                     rowSet, system, emittedRows);
}

} // namespace AvbdRef
