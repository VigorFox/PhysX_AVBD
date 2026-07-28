#include "avbd_solver.h"
// Keep body-vs-static rules aligned with the PhysX DyAvbd* path.
#include "avbd_articulation.h"
#include "avbd_d6_core.h"
#include "avbd_island_rows.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace AvbdRef {

namespace {
bool gContactIslandPcgSuiteProbeEnabled = false;
bool gCanonicalRigidContactAuthoringSuiteProbeEnabled = false;

enum class RevoluteMotorVelocityOwnerKind {
  None,
  Isolated,
  CenteredGear
};

struct RevoluteMotorVelocityOwner {
  RevoluteMotorVelocityOwnerKind kind =
      RevoluteMotorVelocityOwnerKind::None;
  uint32_t motorJointIndex = UINT32_MAX;
  float expectedDynamicPairAngularMomentum = 0.0f;
  bool conserveDynamicPairAngularMomentum = false;
  Vec3 expectedDynamicPairAngularMomentumVector;
  bool conserveDynamicPairAngularMomentumVector = false;
  Vec3 expectedDynamicPairLinearMomentum;
  bool conserveDynamicPairLinearMomentum = false;
  Vec3 expectedDynamicPairSpatialAngularMomentum;
  bool conserveDynamicPairSpatialMomentum = false;
  float solveStartRelativeVelocity = 0.0f;
  bool useSolveStartRelativeVelocity = false;
};

bool isDynamicEndpoint(const std::vector<Body> &bodies, uint32_t index) {
  return index < bodies.size() && bodies[index].mass > 0.0f;
}

bool isWorldEndpoint(uint32_t index) {
  return index == UINT32_MAX;
}

bool isStrictRevoluteMotorRow(const D6Joint &joint) {
  const uint32_t twistMotion = joint.getAngularMotion(0);
  const bool limitedTwist = twistMotion == 1u;
  return joint.motorEnabled && joint.motorMaxForce > 0.0f &&
         std::isfinite(joint.motorMaxForce) &&
         std::isfinite(joint.motorTargetVelocity) &&
         std::isfinite(joint.motorGearRatio) &&
         joint.motorGearRatio > 0.0f &&
         std::isfinite(joint.motorExternalAngularVelocityA.x) &&
         std::isfinite(joint.motorExternalAngularVelocityA.y) &&
         std::isfinite(joint.motorExternalAngularVelocityA.z) &&
         std::isfinite(joint.motorExternalAngularVelocityB.x) &&
         std::isfinite(joint.motorExternalAngularVelocityB.y) &&
         std::isfinite(joint.motorExternalAngularVelocityB.z) &&
         joint.linearMotion == 0u &&
         (twistMotion == 1u || twistMotion == 2u) &&
         joint.getAngularMotion(1) == 0u &&
         joint.getAngularMotion(2) == 0u &&
         joint.driveFlags == 0u && joint.driveAccelerationFlags == 0u &&
         joint.coneAngleLimit <= 0.0f &&
         (!limitedTwist ||
          (std::isfinite(joint.angularLimitLower[0]) &&
           std::isfinite(joint.angularLimitUpper[0]) &&
           joint.angularLimitLower[0] < joint.angularLimitUpper[0]));
}

Vec3 getRevoluteMotorWorldAxis(const std::vector<Body> &bodies,
                               const D6Joint &joint) {
  const bool dynamicA = isDynamicEndpoint(bodies, joint.bodyA);
  const Quat worldFrameA =
      dynamicA ? bodies[joint.bodyA].rotation * joint.localFrameA
               : joint.localFrameA;
  return worldFrameA.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
}

bool hasPrincipalAngularResponse(const Body &body, const Vec3 &worldAxis) {
  const Vec3 response = body.invInertiaWorld * worldAxis;
  const float responseScale = std::max(response.length(), 1e-8f);
  return response.cross(worldAxis).length() <= responseScale * 1e-4f;
}

RevoluteMotorVelocityOwner classifyRevoluteMotorVelocityOwner(
    std::vector<Body> &bodies, const std::vector<Contact> &contacts,
    const std::vector<D6Joint> &d6Joints,
    const std::vector<GearJoint> &gearJoints,
    const std::vector<Articulation> &articulations,
    const std::vector<SoftBody> &softBodies,
    const std::vector<SoftContact> &softContacts,
    const std::vector<SoftParticle> &softParticles) {
  RevoluteMotorVelocityOwner owner;
  if (!contacts.empty() || !articulations.empty() || !softBodies.empty() ||
      !softContacts.empty() || !softParticles.empty())
    return owner;

  uint32_t dynamicBodyCount = 0;
  for (Body &body : bodies) {
    if (body.mass <= 0.0f)
      continue;
    body.updateInvInertiaWorld();
    dynamicBodyCount++;
  }

  if (d6Joints.size() == 1 && gearJoints.empty()) {
    const D6Joint &motor = d6Joints[0];
    if (!isStrictRevoluteMotorRow(motor))
      return owner;
    const bool dynamicA = isDynamicEndpoint(bodies, motor.bodyA);
    const bool dynamicB = isDynamicEndpoint(bodies, motor.bodyB);
    if (dynamicA == dynamicB && !dynamicA)
      return owner;
    if ((!dynamicA && !isWorldEndpoint(motor.bodyA)) ||
        (!dynamicB && !isWorldEndpoint(motor.bodyB)) ||
        (dynamicA && dynamicB && motor.bodyA == motor.bodyB) ||
        dynamicBodyCount != uint32_t(dynamicA) + uint32_t(dynamicB))
      return owner;
    const bool nonUnitDriveRatio =
        std::fabs(motor.motorGearRatio - 1.0f) > 1e-6f;
    if (motor.getAngularMotion(0) == 1u) {
      const bool oneDynamic =
          dynamicBodyCount == 1u && dynamicA != dynamicB;
      const bool dynamicPair =
          dynamicBodyCount == 2u && dynamicA && dynamicB;
      if (!oneDynamic && !dynamicPair)
        return owner;
      if (dynamicPair) {
        if (motor.anchorA.length2() > 1e-8f ||
            motor.anchorB.length2() > 1e-8f)
          return owner;
      } else {
        const Vec3 &dynamicAnchor =
            dynamicA ? motor.anchorA : motor.anchorB;
        if (dynamicAnchor.length2() > 1e-8f)
          return owner;
      }
    }
    if (motor.motorFreeSpin) {
      const bool oneDynamic =
          dynamicBodyCount == 1u && dynamicA != dynamicB;
      const bool dynamicPair =
          dynamicBodyCount == 2u && dynamicA && dynamicB;
      if (!oneDynamic && !dynamicPair)
        return owner;
      if (dynamicPair) {
        if (motor.anchorA.length2() > 1e-8f ||
            motor.anchorB.length2() > 1e-8f)
          return owner;
      } else {
        const Vec3 &dynamicAnchor =
            dynamicA ? motor.anchorA : motor.anchorB;
        if (dynamicAnchor.length2() > 1e-8f)
          return owner;
      }
    }
    if (motor.getAngularMotion(0) == 1u && motor.motorFreeSpin)
      return owner;
    if ((motor.getAngularMotion(0) == 1u || motor.motorFreeSpin) &&
        nonUnitDriveRatio)
      return owner;
    if (nonUnitDriveRatio) {
      const Vec3 localAxisA =
          motor.localFrameA.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
      const Vec3 localAxisB =
          motor.localFrameB.rotate(Vec3(1.0f, 0.0f, 0.0f)).normalized();
      if (motor.getAngularMotion(0) != 2u || motor.motorFreeSpin ||
          dynamicBodyCount != 2u || !dynamicA || !dynamicB ||
          localAxisA.length2() <= 1e-8f ||
          localAxisB.length2() <= 1e-8f ||
          motor.anchorA.cross(localAxisA).length2() > 1e-8f ||
          motor.anchorB.cross(localAxisB).length2() > 1e-8f)
        return owner;
    }

    const Vec3 worldAxis = getRevoluteMotorWorldAxis(bodies, motor);
    if (worldAxis.length2() <= 1e-8f)
      return owner;
    const bool dynamicEndpoint[2] = {dynamicA, dynamicB};
    const uint32_t bodyIndex[2] = {motor.bodyA, motor.bodyB};
    const Vec3 anchor[2] = {motor.anchorA, motor.anchorB};
    const Vec3 localAxis[2] = {
        motor.localFrameA.rotate(Vec3(1.0f, 0.0f, 0.0f)),
        motor.localFrameB.rotate(Vec3(1.0f, 0.0f, 0.0f))};
    const Vec3 externalVelocity[2] = {
        motor.motorExternalAngularVelocityA,
        motor.motorExternalAngularVelocityB};
    const bool allowCoupledOffPrincipalResponse =
        ((dynamicBodyCount == 1u && dynamicA != dynamicB) ||
         (dynamicBodyCount == 2u && dynamicA && dynamicB)) &&
        motor.getAngularMotion(0) == 2u && !motor.motorFreeSpin &&
        !nonUnitDriveRatio &&
        motor.motorExternalAngularVelocityA.length2() <= 1e-12f &&
        motor.motorExternalAngularVelocityB.length2() <= 1e-12f;
    const bool allowCoupledOffCenterResponse =
        ((dynamicBodyCount == 1u && dynamicA != dynamicB) ||
         (dynamicBodyCount == 2u && dynamicA && dynamicB)) &&
        motor.getAngularMotion(0) == 2u && !motor.motorFreeSpin &&
        !nonUnitDriveRatio &&
        motor.motorExternalAngularVelocityA.length2() <= 1e-12f &&
        motor.motorExternalAngularVelocityB.length2() <= 1e-12f;
    for (uint32_t side = 0; side < 2; ++side) {
      if (!dynamicEndpoint[side]) {
        const float stepScale =
            std::max(externalVelocity[side].length(), 1e-8f);
        if (externalVelocity[side].cross(worldAxis).length() >
            stepScale * 1e-4f)
          return owner;
        continue;
      }
      if (externalVelocity[side].length2() > 1e-12f)
        return owner;
      if (anchor[side].cross(localAxis[side]).length2() > 1e-8f &&
          !allowCoupledOffCenterResponse)
        return owner;
      if (!hasPrincipalAngularResponse(bodies[bodyIndex[side]],
                                       worldAxis) &&
          !allowCoupledOffPrincipalResponse)
        return owner;
    }

    owner.kind = RevoluteMotorVelocityOwnerKind::Isolated;
    owner.motorJointIndex = 0;
    if (motor.motorFreeSpin) {
      if (dynamicA)
        owner.solveStartRelativeVelocity -=
            worldAxis.dot(bodies[motor.bodyA].angularVelocity);
      if (dynamicB)
        owner.solveStartRelativeVelocity +=
            worldAxis.dot(bodies[motor.bodyB].angularVelocity);
      owner.useSolveStartRelativeVelocity =
          std::isfinite(owner.solveStartRelativeVelocity);
    }
    if (dynamicA && dynamicB) {
      const Body &bodyA = bodies[motor.bodyA];
      const Body &bodyB = bodies[motor.bodyB];
      owner.expectedDynamicPairAngularMomentum =
          worldAxis.dot(bodyA.invInertiaWorld.inverse() *
                            bodyA.angularVelocity *
                            motor.motorGearRatio +
                        bodyB.invInertiaWorld.inverse() *
                            bodyB.angularVelocity);
      owner.conserveDynamicPairAngularMomentum =
          std::isfinite(owner.expectedDynamicPairAngularMomentum);
      if (std::fabs(motor.motorGearRatio - 1.0f) <= 1e-6f &&
          motor.anchorA.length2() <= 1e-8f &&
          motor.anchorB.length2() <= 1e-8f) {
        owner.expectedDynamicPairAngularMomentumVector =
            bodyA.invInertiaWorld.inverse() *
                bodyA.angularVelocity +
            bodyB.invInertiaWorld.inverse() *
                bodyB.angularVelocity;
        const Vec3 &momentum =
            owner.expectedDynamicPairAngularMomentumVector;
        owner.conserveDynamicPairAngularMomentumVector =
            std::isfinite(momentum.x) &&
            std::isfinite(momentum.y) &&
            std::isfinite(momentum.z);
      }
      owner.expectedDynamicPairLinearMomentum =
          bodyA.linearVelocity * bodyA.mass +
          bodyB.linearVelocity * bodyB.mass;
      const Vec3 &linearMomentum =
          owner.expectedDynamicPairLinearMomentum;
      owner.conserveDynamicPairLinearMomentum =
          std::isfinite(bodyA.mass) &&
          std::isfinite(bodyB.mass) &&
          std::isfinite(linearMomentum.x) &&
          std::isfinite(linearMomentum.y) &&
          std::isfinite(linearMomentum.z);
      owner.expectedDynamicPairSpatialAngularMomentum =
          bodyA.position.cross(bodyA.linearVelocity * bodyA.mass) +
          bodyA.invInertiaWorld.inverse() *
              bodyA.angularVelocity +
          bodyB.position.cross(bodyB.linearVelocity * bodyB.mass) +
          bodyB.invInertiaWorld.inverse() *
              bodyB.angularVelocity;
      const Vec3 &spatialAngularMomentum =
          owner.expectedDynamicPairSpatialAngularMomentum;
      owner.conserveDynamicPairSpatialMomentum =
          owner.conserveDynamicPairLinearMomentum &&
          std::isfinite(spatialAngularMomentum.x) &&
          std::isfinite(spatialAngularMomentum.y) &&
          std::isfinite(spatialAngularMomentum.z);
    }
    return owner;
  }

  if (dynamicBodyCount != 2 || d6Joints.size() != 2 ||
      gearJoints.size() != 1)
    return owner;
  const GearJoint &gear = gearJoints[0];
  if (gear.bodyA >= bodies.size() || gear.bodyB >= bodies.size() ||
      gear.bodyA == gear.bodyB || bodies[gear.bodyA].mass <= 0.0f ||
      bodies[gear.bodyB].mass <= 0.0f ||
      !std::isfinite(gear.gearRatio) ||
      std::fabs(gear.gearRatio) <= 1e-6f ||
      gear.axisA.length2() <= 1e-8f || gear.axisB.length2() <= 1e-8f)
    return owner;

  bool ownsGearBodyA = false;
  bool ownsGearBodyB = false;
  for (uint32_t i = 0; i < d6Joints.size(); ++i) {
    const D6Joint &joint = d6Joints[i];
    const bool dynamicA = isDynamicEndpoint(bodies, joint.bodyA);
    const bool dynamicB = isDynamicEndpoint(bodies, joint.bodyB);
    if (dynamicA == dynamicB ||
        (!dynamicA && !isWorldEndpoint(joint.bodyA)) ||
        (!dynamicB && !isWorldEndpoint(joint.bodyB)) ||
        joint.linearMotion != 0u || joint.angularMotion != 0x2u ||
        joint.driveFlags != 0u || joint.driveAccelerationFlags != 0u ||
        joint.coneAngleLimit > 0.0f)
      return RevoluteMotorVelocityOwner();

    const uint32_t dynamicIndex = dynamicA ? joint.bodyA : joint.bodyB;
    const Vec3 &dynamicAnchor = dynamicA ? joint.anchorA : joint.anchorB;
    if (dynamicAnchor.length2() > 1e-8f ||
        (dynamicIndex != gear.bodyA && dynamicIndex != gear.bodyB))
      return RevoluteMotorVelocityOwner();
    if (dynamicIndex == gear.bodyA) {
      if (ownsGearBodyA)
        return RevoluteMotorVelocityOwner();
      ownsGearBodyA = true;
    } else {
      if (ownsGearBodyB)
        return RevoluteMotorVelocityOwner();
      ownsGearBodyB = true;
    }

    const Vec3 hingeAxis =
        (dynamicA ? joint.localFrameA : joint.localFrameB)
            .rotate(Vec3(1.0f, 0.0f, 0.0f))
            .normalized();
    const Vec3 gearAxis =
        (dynamicIndex == gear.bodyA ? gear.axisA : gear.axisB).normalized();
    const Vec3 worldAxis =
        bodies[dynamicIndex].rotation.rotate(gearAxis).normalized();
    if (std::fabs(hingeAxis.dot(gearAxis)) < 0.9999f ||
        !hasPrincipalAngularResponse(bodies[dynamicIndex], worldAxis))
      return RevoluteMotorVelocityOwner();

    if (joint.motorEnabled) {
      if (owner.motorJointIndex != UINT32_MAX ||
          !isStrictRevoluteMotorRow(joint) || joint.motorFreeSpin ||
          std::fabs(joint.motorGearRatio - 1.0f) > 1e-6f)
        return RevoluteMotorVelocityOwner();
      owner.motorJointIndex = i;
    }
  }
  if (!ownsGearBodyA || !ownsGearBodyB ||
      owner.motorJointIndex == UINT32_MAX)
    return RevoluteMotorVelocityOwner();
  owner.kind = RevoluteMotorVelocityOwnerKind::CenteredGear;
  return owner;
}

bool solveNativeMotorDense6(const float response[6][6],
                            const float rhs[6],
                            bool motorImpulseClamped,
                            float clampedMotorImpulse,
                            float solution[6]) {
  float augmented[6][7] = {};
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column)
      augmented[row][column] = response[row][column];
    augmented[row][6] = rhs[row];
  }
  if (motorImpulseClamped) {
    for (int row = 0; row < 6; ++row) {
      if (row == 3)
        continue;
      augmented[row][6] -=
          augmented[row][3] * clampedMotorImpulse;
      augmented[row][3] = 0.0f;
    }
    for (int column = 0; column < 7; ++column)
      augmented[3][column] = 0.0f;
    augmented[3][3] = 1.0f;
    augmented[3][6] = clampedMotorImpulse;
  }

  for (int column = 0; column < 6; ++column) {
    int pivot = column;
    float pivotMagnitude =
        std::fabs(augmented[column][column]);
    for (int row = column + 1; row < 6; ++row) {
      const float candidate =
          std::fabs(augmented[row][column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!std::isfinite(pivotMagnitude) ||
        pivotMagnitude <= 1e-10f)
      return false;
    if (pivot != column) {
      for (int entry = column; entry < 7; ++entry)
        std::swap(augmented[column][entry],
                  augmented[pivot][entry]);
    }
    const float inversePivot =
        1.0f / augmented[column][column];
    for (int entry = column; entry < 7; ++entry)
      augmented[column][entry] *= inversePivot;
    for (int row = 0; row < 6; ++row) {
      if (row == column)
        continue;
      const float factor = augmented[row][column];
      for (int entry = column; entry < 7; ++entry)
        augmented[row][entry] -=
            factor * augmented[column][entry];
    }
  }
  for (int row = 0; row < 6; ++row) {
    solution[row] = augmented[row][6];
    if (!std::isfinite(solution[row]))
      return false;
  }
  return true;
}

void projectIsolatedRevoluteMotorVelocity(
    std::vector<Body> &bodies, const D6Joint &motor, float dt,
    const RevoluteMotorVelocityOwner &owner) {
  const bool dynamicA = isDynamicEndpoint(bodies, motor.bodyA);
  const bool dynamicB = isDynamicEndpoint(bodies, motor.bodyB);
  const Vec3 worldAxis = getRevoluteMotorWorldAxis(bodies, motor);
  if (worldAxis.length2() <= 1e-8f)
    return;

  Vec3 responseA;
  Vec3 responseB;
  float unitResponse = 0.0f;
  float motorVelocity = 0.0f;
  const float driveRatio = motor.motorGearRatio;
  if (dynamicA) {
    responseA = bodies[motor.bodyA].invInertiaWorld * worldAxis;
    unitResponse += worldAxis.dot(responseA);
    motorVelocity -=
        worldAxis.dot(bodies[motor.bodyA].angularVelocity);
  } else
    motorVelocity -=
        worldAxis.dot(motor.motorExternalAngularVelocityA);
  if (dynamicB) {
    responseB = bodies[motor.bodyB].invInertiaWorld * worldAxis;
    unitResponse +=
        driveRatio * driveRatio * worldAxis.dot(responseB);
    motorVelocity +=
        driveRatio *
        worldAxis.dot(bodies[motor.bodyB].angularVelocity);
  } else
    motorVelocity +=
        driveRatio *
        worldAxis.dot(motor.motorExternalAngularVelocityB);
  if (!std::isfinite(unitResponse) || unitResponse <= 1e-10f)
    return;

  const uint32_t dynamicBodyIndex =
      dynamicA ? motor.bodyA : motor.bodyB;
  const Vec3 dynamicAnchor =
      dynamicA ? motor.anchorA : motor.anchorB;
  const Quat dynamicFrame =
      dynamicA ? motor.localFrameA : motor.localFrameB;
  const Vec3 dynamicLocalAxis =
      dynamicFrame.rotate(Vec3(1, 0, 0)).normalized();
  const bool coupledOffCenterResponse =
      dynamicA != dynamicB && motor.getAngularMotion(0) == 2u &&
      !motor.motorFreeSpin &&
      std::fabs(driveRatio - 1.0f) <= 1e-6f &&
      dynamicLocalAxis.length2() > 1e-8f &&
      dynamicAnchor.cross(dynamicLocalAxis).length2() > 1e-8f;
  if (coupledOffCenterResponse) {
    Body &body = bodies[dynamicBodyIndex];
    if (!(body.mass > 0.0f))
      return;
    const Mat33 inertia = body.invInertiaWorld.inverse();
    const Vec3 worldLeverArm =
        body.rotation.rotate(dynamicAnchor);
    const float motorSign = dynamicA ? -1.0f : 1.0f;
    const auto computeMotorImpulse =
        [&](float angularSpeed) -> float {
      const Vec3 desiredAngular = worldAxis * angularSpeed;
      const Vec3 desiredLinear =
          -desiredAngular.cross(worldLeverArm);
      const Vec3 linearImpulse =
          (desiredLinear - body.linearVelocity) * body.mass;
      const Vec3 angularImpulse =
          inertia * (desiredAngular - body.angularVelocity);
      const Vec3 anchorTorque =
          angularImpulse - worldLeverArm.cross(linearImpulse);
      return motorSign * worldAxis.dot(anchorTorque);
    };
    const float zeroSpeedImpulse = computeMotorImpulse(0.0f);
    const float impulsePerAngularSpeed =
        computeMotorImpulse(1.0f) - zeroSpeedImpulse;
    if (!std::isfinite(zeroSpeedImpulse) ||
        !std::isfinite(impulsePerAngularSpeed) ||
        impulsePerAngularSpeed <= 1e-10f)
      return;
    const float targetAngularSpeed =
        motorSign * motor.motorTargetVelocity;
    const float requiredMotorImpulse =
        zeroSpeedImpulse +
        impulsePerAngularSpeed * targetAngularSpeed;
    const float maximumMotorImpulse = motor.motorMaxForce * dt;
    const float motorImpulse =
        std::max(-maximumMotorImpulse,
                 std::min(maximumMotorImpulse,
                          requiredMotorImpulse));
    const float ownedAngularSpeed =
        (motorImpulse - zeroSpeedImpulse) /
        impulsePerAngularSpeed;
    const Vec3 candidateAngular =
        worldAxis * ownedAngularSpeed;
    const Vec3 candidateLinear =
        -candidateAngular.cross(worldLeverArm);
    if (!std::isfinite(candidateAngular.x) ||
        !std::isfinite(candidateAngular.y) ||
        !std::isfinite(candidateAngular.z) ||
        !std::isfinite(candidateLinear.x) ||
        !std::isfinite(candidateLinear.y) ||
        !std::isfinite(candidateLinear.z) ||
        candidateAngular.length() > body.maxAngularVelocity ||
        candidateLinear.length() > body.maxLinearVelocity)
      return;
    body.angularVelocity = candidateAngular;
    body.linearVelocity = candidateLinear;
    return;
  }

  const Vec3 dynamicResponse = dynamicA ? responseA : responseB;
  const bool coupledOffPrincipalResponse =
      dynamicA != dynamicB && motor.getAngularMotion(0) == 2u &&
      !motor.motorFreeSpin &&
      std::fabs(driveRatio - 1.0f) <= 1e-6f &&
      dynamicResponse.cross(worldAxis).length() >
          std::max(dynamicResponse.length(), 1e-8f) * 1e-4f;
  if (coupledOffPrincipalResponse) {
    Body &body = bodies[dynamicA ? motor.bodyA : motor.bodyB];
    const float motorSign = dynamicA ? -1.0f : 1.0f;
    const Vec3 motorJ = worldAxis * motorSign;
    const Vec3 referenceAxis =
        std::fabs(worldAxis.x) < 0.8f ? Vec3(1, 0, 0)
                                     : Vec3(0, 1, 0);
    const Vec3 swingJ0 =
        worldAxis.cross(referenceAxis).normalized();
    const Vec3 swingJ1 =
        worldAxis.cross(swingJ0).normalized();
    if (swingJ0.length2() <= 1e-8f ||
        swingJ1.length2() <= 1e-8f)
      return;

    const Vec3 jacobian[3] = {motorJ, swingJ0, swingJ1};
    const Vec3 response[3] = {
        body.invInertiaWorld * motorJ,
        body.invInertiaWorld * swingJ0,
        body.invInertiaWorld * swingJ1};
    Mat33 responseMatrix;
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        responseMatrix.m[row][column] =
            jacobian[row].dot(response[column]);
    const float determinant =
        responseMatrix.m[0][0] *
            (responseMatrix.m[1][1] * responseMatrix.m[2][2] -
             responseMatrix.m[1][2] * responseMatrix.m[2][1]) -
        responseMatrix.m[0][1] *
            (responseMatrix.m[1][0] * responseMatrix.m[2][2] -
             responseMatrix.m[1][2] * responseMatrix.m[2][0]) +
        responseMatrix.m[0][2] *
            (responseMatrix.m[1][0] * responseMatrix.m[2][1] -
             responseMatrix.m[1][1] * responseMatrix.m[2][0]);
    if (!std::isfinite(determinant) ||
        std::fabs(determinant) <= 1e-12f)
      return;

    const Vec3 rhs(
        motor.motorTargetVelocity -
            motorJ.dot(body.angularVelocity),
        -swingJ0.dot(body.angularVelocity),
        -swingJ1.dot(body.angularVelocity));
    Vec3 impulse = responseMatrix.inverse() * rhs;
    const float maximumMotorImpulse = motor.motorMaxForce * dt;
    const float clampedMotorImpulse =
        std::max(-maximumMotorImpulse,
                 std::min(maximumMotorImpulse, impulse.x));
    if (clampedMotorImpulse != impulse.x) {
      const float k11 = jacobian[1].dot(response[1]);
      const float k12 = jacobian[1].dot(response[2]);
      const float k22 = jacobian[2].dot(response[2]);
      const float swingDeterminant = k11 * k22 - k12 * k12;
      if (!std::isfinite(swingDeterminant) ||
          std::fabs(swingDeterminant) <= 1e-12f)
        return;
      const float swingRhs0 =
          rhs.y - jacobian[1].dot(response[0]) *
                      clampedMotorImpulse;
      const float swingRhs1 =
          rhs.z - jacobian[2].dot(response[0]) *
                      clampedMotorImpulse;
      impulse.y =
          (swingRhs0 * k22 - swingRhs1 * k12) /
          swingDeterminant;
      impulse.z =
          (k11 * swingRhs1 - k12 * swingRhs0) /
          swingDeterminant;
      impulse.x = clampedMotorImpulse;
    }
    const Vec3 candidate =
        body.angularVelocity + response[0] * impulse.x +
        response[1] * impulse.y + response[2] * impulse.z;
    if (!std::isfinite(candidate.x) ||
        !std::isfinite(candidate.y) ||
        !std::isfinite(candidate.z) ||
        candidate.length() > body.maxAngularVelocity)
      return;
    body.angularVelocity = candidate;
    return;
  }

  const bool coupledDynamicPairOffPrincipalResponse =
      dynamicA && dynamicB && motor.getAngularMotion(0) == 2u &&
      !motor.motorFreeSpin &&
      std::fabs(driveRatio - 1.0f) <= 1e-6f &&
      motor.anchorA.length2() <= 1e-8f &&
      motor.anchorB.length2() <= 1e-8f &&
      (responseA.cross(worldAxis).length() >
           std::max(responseA.length(), 1e-8f) * 1e-4f ||
       responseB.cross(worldAxis).length() >
           std::max(responseB.length(), 1e-8f) * 1e-4f);
  if (coupledDynamicPairOffPrincipalResponse) {
    Body &bodyA = bodies[motor.bodyA];
    Body &bodyB = bodies[motor.bodyB];
    const Vec3 referenceAxis =
        std::fabs(worldAxis.x) < 0.8f ? Vec3(1, 0, 0)
                                     : Vec3(0, 1, 0);
    const Vec3 swingJ0 =
        worldAxis.cross(referenceAxis).normalized();
    const Vec3 swingJ1 =
        worldAxis.cross(swingJ0).normalized();
    if (swingJ0.length2() <= 1e-8f ||
        swingJ1.length2() <= 1e-8f)
      return;

    const Vec3 jacobian[3] = {worldAxis, swingJ0, swingJ1};
    Vec3 responseA3[3];
    Vec3 responseB3[3];
    Mat33 responseMatrix;
    for (int row = 0; row < 3; ++row) {
      responseA3[row] = bodyA.invInertiaWorld * jacobian[row];
      responseB3[row] = bodyB.invInertiaWorld * jacobian[row];
    }
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        responseMatrix.m[row][column] =
            jacobian[row].dot(responseA3[column] +
                              responseB3[column]);
    const float determinant =
        responseMatrix.m[0][0] *
            (responseMatrix.m[1][1] * responseMatrix.m[2][2] -
             responseMatrix.m[1][2] * responseMatrix.m[2][1]) -
        responseMatrix.m[0][1] *
            (responseMatrix.m[1][0] * responseMatrix.m[2][2] -
             responseMatrix.m[1][2] * responseMatrix.m[2][0]) +
        responseMatrix.m[0][2] *
            (responseMatrix.m[1][0] * responseMatrix.m[2][1] -
             responseMatrix.m[1][1] * responseMatrix.m[2][0]);
    if (!std::isfinite(determinant) ||
        std::fabs(determinant) <= 1e-12f)
      return;

    const Vec3 relativeAngular =
        bodyB.angularVelocity - bodyA.angularVelocity;
    const Vec3 rhs(
        motor.motorTargetVelocity -
            jacobian[0].dot(relativeAngular),
        -jacobian[1].dot(relativeAngular),
        -jacobian[2].dot(relativeAngular));
    Vec3 impulse = responseMatrix.inverse() * rhs;
    const float maximumMotorImpulse = motor.motorMaxForce * dt;
    const float clampedMotorImpulse =
        std::max(-maximumMotorImpulse,
                 std::min(maximumMotorImpulse, impulse.x));
    if (clampedMotorImpulse != impulse.x) {
      const float k11 =
          jacobian[1].dot(responseA3[1] + responseB3[1]);
      const float k12 =
          jacobian[1].dot(responseA3[2] + responseB3[2]);
      const float k22 =
          jacobian[2].dot(responseA3[2] + responseB3[2]);
      const float swingDeterminant = k11 * k22 - k12 * k12;
      if (!std::isfinite(swingDeterminant) ||
          std::fabs(swingDeterminant) <= 1e-12f)
        return;
      const float swingRhs0 =
          rhs.y - jacobian[1].dot(responseA3[0] +
                                  responseB3[0]) *
                      clampedMotorImpulse;
      const float swingRhs1 =
          rhs.z - jacobian[2].dot(responseA3[0] +
                                  responseB3[0]) *
                      clampedMotorImpulse;
      impulse.y =
          (swingRhs0 * k22 - swingRhs1 * k12) /
          swingDeterminant;
      impulse.z =
          (k11 * swingRhs1 - k12 * swingRhs0) /
          swingDeterminant;
      impulse.x = clampedMotorImpulse;
    }
    Vec3 candidateA =
        bodyA.angularVelocity - responseA3[0] * impulse.x -
        responseA3[1] * impulse.y -
        responseA3[2] * impulse.z;
    Vec3 candidateB =
        bodyB.angularVelocity + responseB3[0] * impulse.x +
        responseB3[1] * impulse.y +
        responseB3[2] * impulse.z;
    if (owner.conserveDynamicPairAngularMomentumVector) {
      const Mat33 inertiaA = bodyA.invInertiaWorld.inverse();
      const Mat33 inertiaB = bodyB.invInertiaWorld.inverse();
      Mat33 inertiaSum;
      for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
          inertiaSum.m[row][column] =
              inertiaA.m[row][column] +
              inertiaB.m[row][column];
      const Vec3 currentAngularMomentum =
          inertiaA * candidateA + inertiaB * candidateB;
      const Vec3 commonAngularVelocity =
          inertiaSum.inverse() *
          (owner.expectedDynamicPairAngularMomentumVector -
           currentAngularMomentum);
      candidateA += commonAngularVelocity;
      candidateB += commonAngularVelocity;
    }
    if (!std::isfinite(candidateA.x) ||
        !std::isfinite(candidateA.y) ||
        !std::isfinite(candidateA.z) ||
        !std::isfinite(candidateB.x) ||
        !std::isfinite(candidateB.y) ||
        !std::isfinite(candidateB.z) ||
        candidateA.length() > bodyA.maxAngularVelocity ||
        candidateB.length() > bodyB.maxAngularVelocity)
      return;
    bodyA.angularVelocity = candidateA;
    bodyB.angularVelocity = candidateB;
    return;
  }

  const Vec3 localAxisA =
      motor.localFrameA.rotate(Vec3(1, 0, 0));
  const Vec3 localAxisB =
      motor.localFrameB.rotate(Vec3(1, 0, 0));
  const bool coupledDynamicPairOffCenterResponse =
      dynamicA && dynamicB && motor.getAngularMotion(0) == 2u &&
      !motor.motorFreeSpin &&
      std::fabs(driveRatio - 1.0f) <= 1e-6f &&
      (motor.anchorA.cross(localAxisA).length2() > 1e-8f ||
       motor.anchorB.cross(localAxisB).length2() > 1e-8f);
  if (coupledDynamicPairOffCenterResponse) {
    Body &bodyA = bodies[motor.bodyA];
    Body &bodyB = bodies[motor.bodyB];
    if (!(bodyA.mass > 0.0f) || !(bodyB.mass > 0.0f))
      return;
    const float invMassA = 1.0f / bodyA.mass;
    const float invMassB = 1.0f / bodyB.mass;
    const Vec3 rA = bodyA.rotation.rotate(motor.anchorA);
    const Vec3 rB = bodyB.rotation.rotate(motor.anchorB);
    const Vec3 referenceAxis =
        std::fabs(worldAxis.x) < 0.8f ? Vec3(1, 0, 0)
                                     : Vec3(0, 1, 0);
    const Vec3 swingAxis0 =
        worldAxis.cross(referenceAxis).normalized();
    const Vec3 swingAxis1 =
        worldAxis.cross(swingAxis0).normalized();
    if (swingAxis0.length2() <= 1e-8f ||
        swingAxis1.length2() <= 1e-8f)
      return;

    const Vec3 worldAxes[3] = {
        Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
    const Vec3 angularAxes[3] = {
        worldAxis, swingAxis0, swingAxis1};
    Vec3 linearJacobianA[6];
    Vec3 angularJacobianA[6];
    Vec3 linearJacobianB[6];
    Vec3 angularJacobianB[6];
    for (int row = 0; row < 3; ++row) {
      linearJacobianA[row] = worldAxes[row];
      angularJacobianA[row] = rA.cross(worldAxes[row]);
      linearJacobianB[row] = -worldAxes[row];
      angularJacobianB[row] = -rB.cross(worldAxes[row]);
      linearJacobianA[3 + row] = Vec3();
      angularJacobianA[3 + row] = angularAxes[row];
      linearJacobianB[3 + row] = Vec3();
      angularJacobianB[3 + row] = -angularAxes[row];
    }

    float rhs[6] = {};
    float responseMatrix[6][6] = {};
    for (int row = 0; row < 6; ++row) {
      const float current =
          linearJacobianA[row].dot(bodyA.linearVelocity) +
          angularJacobianA[row].dot(bodyA.angularVelocity) +
          linearJacobianB[row].dot(bodyB.linearVelocity) +
          angularJacobianB[row].dot(bodyB.angularVelocity);
      const float target =
          row == 3 ? -motor.motorTargetVelocity : 0.0f;
      rhs[row] = target - current;
      for (int column = 0; column < 6; ++column) {
        const Vec3 linearResponseA =
            linearJacobianA[column] * invMassA;
        const Vec3 angularResponseA =
            bodyA.invInertiaWorld *
            angularJacobianA[column];
        const Vec3 linearResponseB =
            linearJacobianB[column] * invMassB;
        const Vec3 angularResponseB =
            bodyB.invInertiaWorld *
            angularJacobianB[column];
        responseMatrix[row][column] =
            linearJacobianA[row].dot(linearResponseA) +
            angularJacobianA[row].dot(angularResponseA) +
            linearJacobianB[row].dot(linearResponseB) +
            angularJacobianB[row].dot(angularResponseB);
      }
    }

    float impulse[6] = {};
    if (!solveNativeMotorDense6(
            responseMatrix, rhs, false, 0.0f, impulse))
      return;
    const float maximumMotorImpulse =
        motor.motorMaxForce * dt;
    const float clampedMotorImpulse =
        std::max(-maximumMotorImpulse,
                 std::min(maximumMotorImpulse, impulse[3]));
    if (clampedMotorImpulse != impulse[3] &&
        !solveNativeMotorDense6(
            responseMatrix, rhs, true,
            clampedMotorImpulse, impulse))
      return;

    Vec3 linearImpulseA;
    Vec3 angularImpulseA;
    Vec3 linearImpulseB;
    Vec3 angularImpulseB;
    for (int row = 0; row < 6; ++row) {
      linearImpulseA += linearJacobianA[row] * impulse[row];
      angularImpulseA += angularJacobianA[row] * impulse[row];
      linearImpulseB += linearJacobianB[row] * impulse[row];
      angularImpulseB += angularJacobianB[row] * impulse[row];
    }
    Vec3 candidateLinearA =
        bodyA.linearVelocity + linearImpulseA * invMassA;
    Vec3 candidateAngularA =
        bodyA.angularVelocity +
        bodyA.invInertiaWorld * angularImpulseA;
    Vec3 candidateLinearB =
        bodyB.linearVelocity + linearImpulseB * invMassB;
    Vec3 candidateAngularB =
        bodyB.angularVelocity +
        bodyB.invInertiaWorld * angularImpulseB;

    if (owner.conserveDynamicPairSpatialMomentum &&
        owner.conserveDynamicPairLinearMomentum) {
      const Mat33 inertiaA = bodyA.invInertiaWorld.inverse();
      const Mat33 inertiaB = bodyB.invInertiaWorld.inverse();
      const Vec3 currentLinearMomentum =
          candidateLinearA * bodyA.mass +
          candidateLinearB * bodyB.mass;
      const Vec3 currentAngularMomentum =
          bodyA.position.cross(candidateLinearA * bodyA.mass) +
          inertiaA * candidateAngularA +
          bodyB.position.cross(candidateLinearB * bodyB.mass) +
          inertiaB * candidateAngularB;
      if (!std::isfinite(currentLinearMomentum.x) ||
          !std::isfinite(currentLinearMomentum.y) ||
          !std::isfinite(currentLinearMomentum.z) ||
          !std::isfinite(currentAngularMomentum.x) ||
          !std::isfinite(currentAngularMomentum.y) ||
          !std::isfinite(currentAngularMomentum.z))
        return;

      float spatialResponse[6][6] = {};
      const Vec3 basis[3] = {
          Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
      for (int column = 0; column < 6; ++column) {
        const Vec3 commonLinear =
            column < 3 ? basis[column] : Vec3();
        const Vec3 commonAngular =
            column < 3 ? Vec3() : basis[column - 3];
        const Vec3 deltaLinearA =
            commonLinear +
            commonAngular.cross(bodyA.position);
        const Vec3 deltaLinearB =
            commonLinear +
            commonAngular.cross(bodyB.position);
        const Vec3 deltaLinearMomentum =
            deltaLinearA * bodyA.mass +
            deltaLinearB * bodyB.mass;
        const Vec3 deltaAngularMomentum =
            bodyA.position.cross(deltaLinearA * bodyA.mass) +
            inertiaA * commonAngular +
            bodyB.position.cross(deltaLinearB * bodyB.mass) +
            inertiaB * commonAngular;
        spatialResponse[0][column] = deltaLinearMomentum.x;
        spatialResponse[1][column] = deltaLinearMomentum.y;
        spatialResponse[2][column] = deltaLinearMomentum.z;
        spatialResponse[3][column] = deltaAngularMomentum.x;
        spatialResponse[4][column] = deltaAngularMomentum.y;
        spatialResponse[5][column] = deltaAngularMomentum.z;
      }
      const Vec3 linearMomentumDelta =
          owner.expectedDynamicPairLinearMomentum -
          currentLinearMomentum;
      const Vec3 angularMomentumDelta =
          owner.expectedDynamicPairSpatialAngularMomentum -
          currentAngularMomentum;
      const float spatialRhs[6] = {
          linearMomentumDelta.x,
          linearMomentumDelta.y,
          linearMomentumDelta.z,
          angularMomentumDelta.x,
          angularMomentumDelta.y,
          angularMomentumDelta.z};
      float spatialCorrection[6] = {};
      if (!solveNativeMotorDense6(
              spatialResponse, spatialRhs, false, 0.0f,
              spatialCorrection))
        return;
      const Vec3 commonLinearVelocity(
          spatialCorrection[0], spatialCorrection[1],
          spatialCorrection[2]);
      const Vec3 commonAngularVelocity(
          spatialCorrection[3], spatialCorrection[4],
          spatialCorrection[5]);
      candidateLinearA +=
          commonLinearVelocity +
          commonAngularVelocity.cross(bodyA.position);
      candidateLinearB +=
          commonLinearVelocity +
          commonAngularVelocity.cross(bodyB.position);
      candidateAngularA += commonAngularVelocity;
      candidateAngularB += commonAngularVelocity;
    } else if (owner.conserveDynamicPairLinearMomentum) {
      const float totalMass = bodyA.mass + bodyB.mass;
      const Vec3 currentLinearMomentum =
          candidateLinearA * bodyA.mass +
          candidateLinearB * bodyB.mass;
      if (!std::isfinite(totalMass) || totalMass <= 1e-10f ||
          !std::isfinite(currentLinearMomentum.x) ||
          !std::isfinite(currentLinearMomentum.y) ||
          !std::isfinite(currentLinearMomentum.z))
        return;
      const Vec3 correction =
          (owner.expectedDynamicPairLinearMomentum -
           currentLinearMomentum) *
          (1.0f / totalMass);
      candidateLinearA += correction;
      candidateLinearB += correction;
    }

    if (!std::isfinite(candidateLinearA.x) ||
        !std::isfinite(candidateLinearA.y) ||
        !std::isfinite(candidateLinearA.z) ||
        !std::isfinite(candidateAngularA.x) ||
        !std::isfinite(candidateAngularA.y) ||
        !std::isfinite(candidateAngularA.z) ||
        !std::isfinite(candidateLinearB.x) ||
        !std::isfinite(candidateLinearB.y) ||
        !std::isfinite(candidateLinearB.z) ||
        !std::isfinite(candidateAngularB.x) ||
        !std::isfinite(candidateAngularB.y) ||
        !std::isfinite(candidateAngularB.z) ||
        candidateLinearA.length() > bodyA.maxLinearVelocity ||
        candidateAngularA.length() > bodyA.maxAngularVelocity ||
        candidateLinearB.length() > bodyB.maxLinearVelocity ||
        candidateAngularB.length() > bodyB.maxAngularVelocity)
      return;
    bodyA.linearVelocity = candidateLinearA;
    bodyA.angularVelocity = candidateAngularA;
    bodyB.linearVelocity = candidateLinearB;
    bodyB.angularVelocity = candidateAngularB;
    return;
  }

  const float motorBaseVelocity =
      motor.motorFreeSpin && owner.useSolveStartRelativeVelocity
          ? owner.solveStartRelativeVelocity
          : motorVelocity;
  const float requiredMotorImpulse =
      (motor.motorTargetVelocity - motorBaseVelocity) / unitResponse;
  const float maximumImpulse = motor.motorMaxForce * dt;
  float minimumMotorImpulse = -maximumImpulse;
  float maximumMotorImpulse = maximumImpulse;
  if (motor.motorFreeSpin && motor.motorTargetVelocity > 0.0f)
    minimumMotorImpulse = 0.0f;
  else if (motor.motorFreeSpin && motor.motorTargetVelocity < 0.0f)
    maximumMotorImpulse = 0.0f;
  const float motorImpulse =
      std::max(minimumMotorImpulse,
               std::min(maximumMotorImpulse, requiredMotorImpulse));
  const float motorOwnedVelocity =
      motorBaseVelocity + unitResponse * motorImpulse;
  float impulse =
      (motorOwnedVelocity - motorVelocity) / unitResponse;
  if (motor.getAngularMotion(0) == 1u) {
    const Quat rotationA =
        dynamicA ? bodies[motor.bodyA].rotation : Quat();
    const Quat rotationB =
        dynamicB ? bodies[motor.bodyB].rotation : Quat();
    const float angle = motor.computeHingeAngle(rotationA, rotationB);
    const float limitSpan =
        motor.angularLimitUpper[0] - motor.angularLimitLower[0];
    const float activeTolerance =
        std::max(1e-5f, std::fabs(limitSpan) * 1e-5f);
    const float motorOnlyVelocity =
        motorVelocity + unitResponse * impulse;
    const float predictedAngle = angle + motorOnlyVelocity * dt;
    const bool atLower =
        angle <= motor.angularLimitLower[0] + activeTolerance ||
        predictedAngle <= motor.angularLimitLower[0];
    const bool atUpper =
        angle >= motor.angularLimitUpper[0] - activeTolerance ||
        predictedAngle >= motor.angularLimitUpper[0];
    // The public hinge angle derivative is axis dot (wB - wA). Apply the
    // bounded motor first, then add only the unilateral limit impulse needed
    // to remove an outward derivative at a current or one-step prospective
    // active bound.
    if ((atLower && motorOnlyVelocity < 0.0f) ||
        (atUpper && motorOnlyVelocity > 0.0f))
      impulse = -motorVelocity / unitResponse;
  }
  if (!std::isfinite(impulse))
    return;

  Vec3 candidateA =
      dynamicA ? bodies[motor.bodyA].angularVelocity - responseA * impulse
               : Vec3();
  Vec3 candidateB =
      dynamicB ? bodies[motor.bodyB].angularVelocity +
                     responseB * (driveRatio * impulse)
               : Vec3();
  if (owner.conserveDynamicPairAngularMomentum && dynamicA && dynamicB) {
    const Mat33 inertiaA = bodies[motor.bodyA].invInertiaWorld.inverse();
    const Mat33 inertiaB = bodies[motor.bodyB].invInertiaWorld.inverse();
    const float inertiaSum =
        driveRatio * driveRatio *
            worldAxis.dot(inertiaA * worldAxis) +
        worldAxis.dot(inertiaB * worldAxis);
    const float currentMomentum =
        worldAxis.dot(
            (inertiaA * candidateA) * driveRatio +
            inertiaB * candidateB);
    if (!std::isfinite(inertiaSum) || !std::isfinite(currentMomentum) ||
        inertiaSum <= 1e-10f)
      return;
    const float commonVelocity =
        (owner.expectedDynamicPairAngularMomentum - currentMomentum) /
        inertiaSum;
    if (!std::isfinite(commonVelocity))
      return;
    candidateA += worldAxis * (driveRatio * commonVelocity);
    candidateB += worldAxis * commonVelocity;
  }
  if ((dynamicA &&
       candidateA.length() > bodies[motor.bodyA].maxAngularVelocity) ||
      (dynamicB &&
       candidateB.length() > bodies[motor.bodyB].maxAngularVelocity))
    return;
  if (dynamicA)
    bodies[motor.bodyA].angularVelocity = candidateA;
  if (dynamicB)
    bodies[motor.bodyB].angularVelocity = candidateB;
}

void projectCenteredRevoluteMotorGearVelocity(
    std::vector<Body> &bodies, const D6Joint &motor,
    const GearJoint &gear, float dt) {
  const bool dynamicA = isDynamicEndpoint(bodies, motor.bodyA);
  const bool dynamicB = isDynamicEndpoint(bodies, motor.bodyB);
  if (dynamicA == dynamicB)
    return;
  const uint32_t motorBodyIndex = dynamicA ? motor.bodyA : motor.bodyB;
  if (gear.bodyA >= bodies.size() || gear.bodyB >= bodies.size() ||
      (motorBodyIndex != gear.bodyA && motorBodyIndex != gear.bodyB))
    return;

  Body &bodyA = bodies[gear.bodyA];
  Body &bodyB = bodies[gear.bodyB];
  Body &motorBody = bodies[motorBodyIndex];
  const Vec3 axisA = bodyA.rotation.rotate(gear.axisA).normalized();
  const Vec3 axisB = bodyB.rotation.rotate(gear.axisB).normalized();
  const Vec3 motorAxis = getRevoluteMotorWorldAxis(bodies, motor);
  if (axisA.length2() <= 1e-8f || axisB.length2() <= 1e-8f ||
      motorAxis.length2() <= 1e-8f)
    return;

  const float motorSign = dynamicA ? -1.0f : 1.0f;
  const Vec3 motorJ = motorAxis * motorSign;
  const Vec3 gearJA = axisA * gear.gearRatio;
  const Vec3 gearJB = axisB;
  const Vec3 motorResponse = motorBody.invInertiaWorld * motorJ;
  const Vec3 gearResponseA = bodyA.invInertiaWorld * gearJA;
  const Vec3 gearResponseB = bodyB.invInertiaWorld * gearJB;
  const float kMM = motorJ.dot(motorResponse);
  const float kGG =
      gearJA.dot(gearResponseA) + gearJB.dot(gearResponseB);
  const float kMG =
      motorBodyIndex == gear.bodyA ? motorJ.dot(gearResponseA)
                                   : motorJ.dot(gearResponseB);
  const float determinant = kMM * kGG - kMG * kMG;
  if (!std::isfinite(determinant) || kMM <= 1e-10f ||
      kGG <= 1e-10f || determinant <= 1e-12f)
    return;

  const float motorRhs =
      motor.motorTargetVelocity -
      motorJ.dot(motorBody.angularVelocity);
  const float gearRhs =
      -(gearJA.dot(bodyA.angularVelocity) +
        gearJB.dot(bodyB.angularVelocity));
  const float unconstrainedMotorImpulse =
      (motorRhs * kGG - gearRhs * kMG) / determinant;
  const float maximumMotorImpulse = motor.motorMaxForce * dt;
  const float motorImpulse =
      std::max(-maximumMotorImpulse,
               std::min(maximumMotorImpulse,
                        unconstrainedMotorImpulse));
  const float gearImpulse =
      (gearRhs - kMG * motorImpulse) / kGG;
  if (!std::isfinite(motorImpulse) || !std::isfinite(gearImpulse))
    return;

  Vec3 candidateA =
      bodyA.angularVelocity + gearResponseA * gearImpulse;
  Vec3 candidateB =
      bodyB.angularVelocity + gearResponseB * gearImpulse;
  if (motorBodyIndex == gear.bodyA)
    candidateA += motorResponse * motorImpulse;
  else
    candidateB += motorResponse * motorImpulse;
  if (candidateA.length() > bodyA.maxAngularVelocity ||
      candidateB.length() > bodyB.maxAngularVelocity)
    return;
  bodyA.angularVelocity = candidateA;
  bodyB.angularVelocity = candidateB;
}
}

void setContactIslandPcgSuiteProbeEnabled(bool enabled) {
  gContactIslandPcgSuiteProbeEnabled = enabled;
}

bool isContactIslandPcgSuiteProbeEnabled() {
  return gContactIslandPcgSuiteProbeEnabled;
}

bool restoreTwoBodySupportAxisAngularMomentum(
    Body &bodyA, Body &bodyB, const Vec3 &supportNormal,
    float expectedAxisAngularMomentum) {
  if (!(bodyA.mass > 0.0f) || !(bodyB.mass > 0.0f) ||
      !std::isfinite(bodyA.mass) || !std::isfinite(bodyB.mass) ||
      !std::isfinite(expectedAxisAngularMomentum))
    return false;

  const float normalLength = supportNormal.length();
  if (!(normalLength > 1e-8f) || !std::isfinite(normalLength))
    return false;
  const Vec3 axis = supportNormal * (1.0f / normalLength);
  const float totalMass = bodyA.mass + bodyB.mass;
  if (!(totalMass > 0.0f) || !std::isfinite(totalMass))
    return false;
  const Vec3 centerOfMass =
      (bodyA.position * bodyA.mass + bodyB.position * bodyB.mass) *
      (1.0f / totalMass);
  const Vec3 armA = bodyA.position - centerOfMass;
  const Vec3 armB = bodyB.position - centerOfMass;
  const Mat33 inertiaA = bodyA.invInertiaWorld.inverse();
  const Mat33 inertiaB = bodyB.invInertiaWorld.inverse();
  const Vec3 currentAngularMomentum =
      armA.cross(bodyA.linearVelocity * bodyA.mass) +
      inertiaA * bodyA.angularVelocity +
      armB.cross(bodyB.linearVelocity * bodyB.mass) +
      inertiaB * bodyB.angularVelocity;
  const float currentAxisAngularMomentum =
      currentAngularMomentum.dot(axis);
  const Vec3 tangentArmA = axis.cross(armA);
  const Vec3 tangentArmB = axis.cross(armB);
  const float axisInertia =
      bodyA.mass * tangentArmA.length2() +
      axis.dot(inertiaA * axis) +
      bodyB.mass * tangentArmB.length2() +
      axis.dot(inertiaB * axis);
  if (!(axisInertia > 1e-10f) || !std::isfinite(axisInertia) ||
      !std::isfinite(currentAxisAngularMomentum))
    return false;

  const float angularCorrection =
      (expectedAxisAngularMomentum - currentAxisAngularMomentum) /
      axisInertia;
  const Vec3 commonAngularVelocity = axis * angularCorrection;
  const Vec3 candidateLinearA =
      bodyA.linearVelocity + commonAngularVelocity.cross(armA);
  const Vec3 candidateLinearB =
      bodyB.linearVelocity + commonAngularVelocity.cross(armB);
  const Vec3 candidateAngularA =
      bodyA.angularVelocity + commonAngularVelocity;
  const Vec3 candidateAngularB =
      bodyB.angularVelocity + commonAngularVelocity;
  if (!std::isfinite(candidateLinearA.x) ||
      !std::isfinite(candidateLinearA.y) ||
      !std::isfinite(candidateLinearA.z) ||
      !std::isfinite(candidateLinearB.x) ||
      !std::isfinite(candidateLinearB.y) ||
      !std::isfinite(candidateLinearB.z) ||
      !std::isfinite(candidateAngularA.x) ||
      !std::isfinite(candidateAngularA.y) ||
      !std::isfinite(candidateAngularA.z) ||
      !std::isfinite(candidateAngularB.x) ||
      !std::isfinite(candidateAngularB.y) ||
      !std::isfinite(candidateAngularB.z) ||
      candidateLinearA.length() > bodyA.maxLinearVelocity ||
      candidateLinearB.length() > bodyB.maxLinearVelocity ||
      candidateAngularA.length() > bodyA.maxAngularVelocity ||
      candidateAngularB.length() > bodyB.maxAngularVelocity)
    return false;

  bodyA.linearVelocity = candidateLinearA;
  bodyA.angularVelocity = candidateAngularA;
  bodyB.linearVelocity = candidateLinearB;
  bodyB.angularVelocity = candidateAngularB;
  return true;
}

void setCanonicalRigidContactAuthoringSuiteProbeEnabled(bool enabled) {
  gCanonicalRigidContactAuthoringSuiteProbeEnabled = enabled;
}

bool isCanonicalRigidContactAuthoringSuiteProbeEnabled() {
  return gCanonicalRigidContactAuthoringSuiteProbeEnabled;
}

// =============================================================================
// Factory methods  all create D6Joint entries in the unified d6Joints vector
// =============================================================================

uint32_t Solver::addSphericalJoint(uint32_t bodyA, uint32_t bodyB,
                                   Vec3 anchorA, Vec3 anchorB, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = 0;     // all 3 linear LOCKED
  j.angularMotion = 0x2A; // all 3 angular FREE (2|(2<<2)|(2<<4))
  j.rho = rho_;

  // Compute localFrameB from initial relative rotation (matches PhysX)
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

void Solver::setSphericalJointConeLimit(uint32_t jointIdx, Vec3 coneAxisA,
                                        float limitAngle) {
  if (jointIdx < d6Joints.size()) {
    d6Joints[jointIdx].coneAngleLimit = limitAngle;
    d6Joints[jointIdx].coneAxisA = coneAxisA;
    d6Joints[jointIdx].coneLambda = 0.0f;

    // Build localFrameA so X-axis = cone axis (matches PhysX joint frame)
    Vec3 axisNorm = coneAxisA.normalized();
    Vec3 perp;
    if (std::fabs(axisNorm.x) < 0.9f)
      perp = axisNorm.cross(Vec3(1, 0, 0)).normalized();
    else
      perp = axisNorm.cross(Vec3(0, 1, 0)).normalized();
    Vec3 perp2 = axisNorm.cross(perp);
    d6Joints[jointIdx].localFrameA = quatFromColumns(axisNorm, perp, perp2);
    d6Joints[jointIdx].localFrameB =
        d6Joints[jointIdx].relativeRotation.conjugate() *
        d6Joints[jointIdx].localFrameA;
  }
}

uint32_t Solver::addFixedJoint(uint32_t bodyA, uint32_t bodyB,
                               Vec3 anchorA, Vec3 anchorB, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = 0;  // all 3 linear LOCKED
  j.angularMotion = 0; // all 3 angular LOCKED
  j.rho = rho_;
  // Compute initial relative rotation and localFrameB
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;
  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

uint32_t Solver::addD6Joint(uint32_t bodyA, uint32_t bodyB,
                            Vec3 anchorA, Vec3 anchorB,
                            uint32_t linearMotion_, uint32_t angularMotion_,
                            float angularDamping_, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = anchorA;
  j.anchorB = anchorB;
  j.linearMotion = linearMotion_;
  j.angularMotion = angularMotion_;
  j.rho = rho_;
  j.angularDriveDamping =
      Vec3(angularDamping_, angularDamping_, angularDamping_);

  // Compute relativeRotation and localFrameB (matches PhysX)
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

uint32_t Solver::addRevoluteJoint(uint32_t bodyA, uint32_t bodyB,
                                  Vec3 localAnchorA, Vec3 localAnchorB,
                                  Vec3 localAxisA, Vec3 localAxisB,
                                  float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = localAnchorA;
  j.anchorB = localAnchorB;
  j.linearMotion = 0; // all 3 linear LOCKED

  // Angular: X(twist) = FREE, Y(swing1) = LOCKED, Z(swing2) = LOCKED
  // Hinge axis maps to joint-frame X
  j.angularMotion = 2; // 2|(0<<2)|(0<<4) = axis0=FREE, axis1=LOCKED, axis2=LOCKED

  j.rho = rho_;

  // Build reference axes perpendicular to hinge axis
  Vec3 axisA = localAxisA.normalized();
  Vec3 axisB = localAxisB.normalized();

  auto buildRefAxis = [](Vec3 axis) -> Vec3 {
    Vec3 perp;
    if (fabsf(axis.x) < 0.9f)
      perp = axis.cross(Vec3(1, 0, 0)).normalized();
    else
      perp = axis.cross(Vec3(0, 1, 0)).normalized();
    return perp;
  };

  Vec3 refA = buildRefAxis(axisA);
  Vec3 z_axisA = axisA.cross(refA);

  // localFrameA: X=hingeAxis, Y=refAxisA, Z=cross
  j.localFrameA = quatFromColumns(axisA, refA, z_axisA);

  // Store relative rotation for angular error computation
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;

  // Store revolute-specific fields for hinge angle measurement
  j.hingeAxisB = axisB;
  j.refAxisA = refA;

  // Compute refAxisB: project worldRefA onto B hinge plane, transform to B local
  Vec3 worldRefA = rotA.rotate(refA);
  Vec3 worldAxisB = rotB.rotate(axisB);
  Vec3 proj = worldRefA - worldAxisB * worldRefA.dot(worldAxisB);
  float projLen = proj.length();
  if (projLen > 1e-8f) {
    j.refAxisB = rotB.conjugate().rotate(proj * (1.0f / projLen));
  } else {
    j.refAxisB = buildRefAxis(axisB);
  }

  // Compute localFrameB from relativeRotation (matches PhysX convention)
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

void Solver::setRevoluteJointLimit(uint32_t jointIdx, float lowerLimit,
                                   float upperLimit) {
  if (jointIdx < d6Joints.size()) {
    // Change angular axis 0 from FREE to LIMITED
    d6Joints[jointIdx].angularMotion =
        (d6Joints[jointIdx].angularMotion & ~0x3) | 1; // axis 0 = LIMITED
    d6Joints[jointIdx].angularLimitLower[0] = lowerLimit;
    d6Joints[jointIdx].angularLimitUpper[0] = upperLimit;
    d6Joints[jointIdx].lambdaLimitAngular[0] = 0.0f;
  }
}

void Solver::setRevoluteJointDrive(uint32_t jointIdx, float targetVelocity,
                                   float maxForce, bool freeSpin,
                                   float gearRatio) {
  if (jointIdx < d6Joints.size()) {
    // Native revolute motor targets are consumed only by the strict
    // post-finalize velocity owner.
    d6Joints[jointIdx].motorEnabled = true;
    d6Joints[jointIdx].motorFreeSpin = freeSpin;
    d6Joints[jointIdx].motorTargetVelocity = targetVelocity;
    d6Joints[jointIdx].motorMaxForce = maxForce;
    d6Joints[jointIdx].motorGearRatio = gearRatio;
  }
}

uint32_t Solver::addPrismaticJoint(uint32_t bodyA, uint32_t bodyB,
                                   Vec3 localAnchorA, Vec3 localAnchorB,
                                   Vec3 localAxisA, float rho_) {
  D6Joint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.anchorA = localAnchorA;
  j.anchorB = localAnchorB;
  j.angularMotion = 0; // all 3 angular LOCKED

  // Linear: X(slide) = FREE, Y = LOCKED, Z = LOCKED
  j.linearMotion = 2; // axis0=FREE, axis1=LOCKED, axis2=LOCKED

  j.rho = rho_;

  Vec3 axisA = localAxisA.normalized();
  Vec3 helper =
      (std::abs(axisA.x) > 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
  Vec3 t1 = axisA.cross(helper).normalized();
  Vec3 t2 = axisA.cross(t1);
  j.localFrameA = quatFromColumns(axisA, t1, t2);

  // Store relative rotation and compute localFrameB
  Quat rotA = (bodyA == UINT32_MAX) ? Quat() : bodies[bodyA].rotation;
  Quat rotB = (bodyB == UINT32_MAX) ? Quat() : bodies[bodyB].rotation;
  j.relativeRotation = rotA.conjugate() * rotB;
  j.localFrameB = j.relativeRotation.conjugate() * j.localFrameA;

  uint32_t idx = (uint32_t)d6Joints.size();
  d6Joints.push_back(j);
  return idx;
}

void Solver::setPrismaticJointLimit(uint32_t jointIdx, float lowerLimit,
                                    float upperLimit) {
  if (jointIdx < d6Joints.size()) {
    // Change linear axis 0 from FREE to LIMITED
    d6Joints[jointIdx].linearMotion =
        (d6Joints[jointIdx].linearMotion & ~0x3) | 1; // axis 0 = LIMITED
    d6Joints[jointIdx].linearLimitLower[0] = lowerLimit;
    d6Joints[jointIdx].linearLimitUpper[0] = upperLimit;
    d6Joints[jointIdx].lambdaLimitLinear[0] = 0.0f;
  }
}

void Solver::setPrismaticJointDrive(uint32_t jointIdx, float targetVelocity,
                                    float damping) {
  if (jointIdx < d6Joints.size()) {
    d6Joints[jointIdx].driveFlags |= 0x01; // linear X drive
    d6Joints[jointIdx].driveLinearVelocity = Vec3(targetVelocity, 0, 0);
    d6Joints[jointIdx].linearDriveDamping.x = damping;
    d6Joints[jointIdx].lambdaDriveLinear = Vec3();
  }
}

void Solver::addGearJoint(uint32_t bodyA, uint32_t bodyB,
                          Vec3 axisA, Vec3 axisB,
                          float ratio, float rho_) {
  GearJoint j;
  j.bodyA = bodyA;
  j.bodyB = bodyB;
  j.axisA = axisA.normalized();
  j.axisB = axisB.normalized();
  j.gearRatio = ratio;
  j.rho = rho_;
  gearJoints.push_back(j);
}

// =============================================================================
// Body / Contact creation
// =============================================================================

uint32_t Solver::addBody(Vec3 pos, Quat rot, Vec3 halfExtent, float density,
                         float fric) {
  Body b;
  b.position = pos;
  b.rotation = rot;
  b.linearVelocity = {};
  b.angularVelocity = {};
  b.prevLinearVelocity = {};
  b.friction = fric;
  b.halfExtent = halfExtent;

  float vol = 8.0f * halfExtent.x * halfExtent.y * halfExtent.z;
  if (density > 0) {
    b.mass = vol * density;
    float sx = 2 * halfExtent.x, sy = 2 * halfExtent.y, sz = 2 * halfExtent.z;
    float Ixx = b.mass / 12.0f * (sy * sy + sz * sz);
    float Iyy = b.mass / 12.0f * (sx * sx + sz * sz);
    float Izz = b.mass / 12.0f * (sx * sx + sy * sy);
    b.inertiaTensor = Mat33::diag(Ixx, Iyy, Izz);
  } else {
    b.mass = 0;
    b.inertiaTensor = Mat33::diag(0, 0, 0);
  }
  b.computeDerived();

  uint32_t idx = (uint32_t)bodies.size();
  bodies.push_back(b);
  return idx;
}

void Solver::addContact(uint32_t bodyA, uint32_t bodyB, Vec3 normal, Vec3 rA,
                        Vec3 rB, float depth, float fric) {
  Contact c;
  c.bodyA = bodyA;
  c.bodyB = bodyB;
  c.normal = normal;
  c.rA = rA;
  c.rB = rB;
  c.depth = depth;
  c.friction = fric;
  for (int i = 0; i < 3; i++) {
    c.lambda[i] = 0;
    c.penalty[i] = PENALTY_MIN;
    c.fmin[i] = 0;
    c.fmax[i] = 0;
  }
  c.fmin[0] = -1e30f;
  c.fmax[0] = 0.0f;
  if (bodyB == UINT32_MAX)
    c.staticPrevWorldPoint = rB;
  contacts.push_back(c);
}

static void canonicalizeSharedContactOrientation(
    Contact &contact, const std::vector<Body> &bodies) {
  if (contact.bodyB >= bodies.size())
    return;
  const bool dynamicA = contact.bodyA < bodies.size() &&
                        bodies[contact.bodyA].mass > 0.0f;
  const bool dynamicB = bodies[contact.bodyB].mass > 0.0f;
  const bool keepOrientation =
      (dynamicA && !dynamicB) ||
      (dynamicA && dynamicB && contact.bodyA < contact.bodyB) ||
      (!dynamicA && !dynamicB && contact.bodyA < contact.bodyB);
  if (keepOrientation)
    return;
  std::swap(contact.bodyA, contact.bodyB);
  std::swap(contact.rA, contact.rB);
  contact.normal = -contact.normal;
  // With the deterministic tangent basis used by computeConstraint, n -> -n
  // maps t1 -> -t1 and t2 -> t2. Swapping relative displacement therefore
  // leaves row 1 unchanged and negates row 2 exactly.
  contact.lambda[2] = -contact.lambda[2];
  contact.C0[2] = -contact.C0[2];
}

static bool sharedContactLess(const Contact &a, const Contact &b) {
  if (a.bodyA != b.bodyA)
    return a.bodyA < b.bodyA;
  if (a.bodyB != b.bodyB)
    return a.bodyB < b.bodyB;
  const auto vecLess = [](const Vec3 &x, const Vec3 &y) {
    if (x.x != y.x)
      return x.x < y.x;
    if (x.y != y.y)
      return x.y < y.y;
    return x.z < y.z;
  };
  if (a.rA.x != b.rA.x || a.rA.y != b.rA.y || a.rA.z != b.rA.z)
    return vecLess(a.rA, b.rA);
  if (a.rB.x != b.rB.x || a.rB.y != b.rB.y || a.rB.z != b.rB.z)
    return vecLess(a.rB, b.rB);
  if (a.normal.x != b.normal.x || a.normal.y != b.normal.y ||
      a.normal.z != b.normal.z)
    return vecLess(a.normal, b.normal);
  if (a.depth != b.depth)
    return a.depth < b.depth;
  return a.friction < b.friction;
}

// =============================================================================
// Contact constraint computation
// =============================================================================

void Solver::computeConstraint(Contact &c) {
  Body &bA = bodies[c.bodyA];
  bool bStatic = (c.bodyB == UINT32_MAX);
  Body *pB = bStatic ? nullptr : &bodies[c.bodyB];

  Vec3 rAw = bA.rotation.rotate(c.rA);
  Vec3 rBw = bStatic ? Vec3() : pB->rotation.rotate(c.rB);

  c.JA = Vec6(c.normal, rAw.cross(c.normal));
  c.JB = bStatic ? Vec6() : Vec6(Vec3() - c.normal, Vec3() - rBw.cross(c.normal));

  Vec3 t1, t2;
  if (fabsf(c.normal.y) > 0.9f)
    t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
  else
    t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
  t2 = c.normal.cross(t1);

  c.JAt1 = Vec6(t1, rAw.cross(t1));
  c.JBt1 = bStatic ? Vec6() : Vec6(Vec3() - t1, Vec3() - rBw.cross(t1));
  c.JAt2 = Vec6(t2, rAw.cross(t2));
  c.JBt2 = bStatic ? Vec6() : Vec6(Vec3() - t2, Vec3() - rBw.cross(t2));

  Vec6 dpA(bA.position - bA.initialPosition, bA.deltaWInitial());
  Vec6 dpB;
  if (!bStatic)
    dpB = Vec6(pB->position - pB->initialPosition, pB->deltaWInitial());

  c.C[0] =
      c.C0[0] * (1.0f - alpha) + dot(c.JA, dpA) + dot(c.JB, dpB);
  c.C[1] = c.C0[1] * (1.0f - alpha) + dot(c.JAt1, dpA) + dot(c.JBt1, dpB);
  c.C[2] = c.C0[2] * (1.0f - alpha) + dot(c.JAt2, dpA) + dot(c.JBt2, dpB);

  float frictionBound = fabsf(c.lambda[0]) * c.friction;
  c.fmax[1] = frictionBound;
  c.fmin[1] = -frictionBound;
  c.fmax[2] = frictionBound;
  c.fmin[2] = -frictionBound;
}

void Solver::computeConstraintBodyStatic(Contact &c) {
  Body &bA = bodies[c.bodyA];

  Vec3 rAw = bA.rotation.rotate(c.rA);
  c.JA = Vec6(c.normal, rAw.cross(c.normal));
  c.JB = Vec6();

  Vec3 t1, t2;
  if (fabsf(c.normal.y) > 0.9f)
    t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
  else
    t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
  t2 = c.normal.cross(t1);

  c.JAt1 = Vec6(t1, rAw.cross(t1));
  c.JBt1 = Vec6();
  c.JAt2 = Vec6(t2, rAw.cross(t2));
  c.JBt2 = Vec6();

  Vec6 dpA(bA.position - bA.initialPosition, bA.deltaWInitial());
  const Vec3 staticMotion = c.rB - c.staticPrevWorldPoint;
  const Vec3 relDisp =
      Vec3(dpA[0], dpA[1], dpA[2]) - staticMotion;

  const Vec3 wA = bA.position + rAw;
  float geom = (wA - c.rB).dot(c.normal) - c.depth;
  // Normal: geometric gap (no alpha*C0), aligned with PhysX body-vs-static.
  c.C[0] = geom;
  if (geom < 0.0f)
    c.C[0] = std::min(geom, -c.depth);

  c.C[1] = c.C0[1] * (1.0f - alpha) + relDisp.dot(t1) +
           Vec3(dpA[3], dpA[4], dpA[5]).dot(rAw.cross(t1));
  c.C[2] = c.C0[2] * (1.0f - alpha) + relDisp.dot(t2) +
           Vec3(dpA[3], dpA[4], dpA[5]).dot(rAw.cross(t2));

  const float frictionBound = fabsf(c.lambda[0]) * c.friction;
  c.fmax[1] = frictionBound;
  c.fmin[1] = -frictionBound;
  c.fmax[2] = frictionBound;
  c.fmin[2] = -frictionBound;
}

void Solver::computeC0(Contact &c) {
  Body &bA = bodies[c.bodyA];
  bool bStatic = isBodyVsStaticContact(c.bodyA, c.bodyB);
  if (bStatic) {
    c.C0[0] = 0.0f;
    c.C0[1] = 0.0f;
    c.C0[2] = 0.0f;
    return;
  }
  Body *pB = &bodies[c.bodyB];

  Vec3 wA = bA.position + bA.rotation.rotate(c.rA);
  Vec3 wB = pB->position + pB->rotation.rotate(c.rB);

  float rawC0 = (wA - wB).dot(c.normal) - c.depth;

  // Depth-adaptive C0 clamping: for deep penetrations (fast impacts),
  // reduce C0 so that alpha blending does not over-soften the correction.
  // Shallow contacts keep C0 unchanged; deep ones fade C0 toward zero.
  const float c0Threshold = 0.05f;  // 50 mm: only trigger on fast impacts
  const float c0MaxDepth  = 0.20f;  // 200 mm: full fade-out
  if (rawC0 < -c0Threshold) {
    float t = std::clamp(
        (c0MaxDepth + rawC0) / (c0MaxDepth - c0Threshold), 0.0f, 1.0f);
    rawC0 *= t;
  }

  c.C0[0] = rawC0;
  c.C0[1] = 0.0f;
  c.C0[2] = 0.0f;
}

bool Solver::bodyTouchesStatic(uint32_t bodyIdx) const {
  for (const auto &c : contacts) {
    if (c.bodyA == bodyIdx && c.bodyB == UINT32_MAX)
      return true;
  }
  return false;
}

bool Solver::bodyTouchesKinematicShell(uint32_t bodyIdx) const {
  for (const auto &sc : softContacts) {
    if (sc.rigidBodyIdx != bodyIdx)
      continue;
    if (sc.particleIdx < softParticles.size() &&
        softParticles[sc.particleIdx].invMass <= 0.0f)
      return true;
  }
  return false;
}

float Solver::contactGeomViolation(const Contact &c) const {
  const Body &bA = bodies[c.bodyA];
  const Vec3 wA = bA.position + bA.rotation.rotate(c.rA);
  float geom = (wA - c.rB).dot(c.normal) - c.depth;
  if (geom < 0.0f)
    geom = std::min(geom, -c.depth);
  return geom;
}

bool Solver::isSequentialBodyStaticIsland() const {
  if (bodyStaticContactSolve != BodyStaticContactSolve::SequentialPerContact)
    return false;
  if (!d6Joints.empty() || !gearJoints.empty() || !articulations.empty())
    return false;
  if (!softBodies.empty() || contacts.empty())
    return false;
  for (const auto &c : contacts) {
    if (c.bodyB != UINT32_MAX)
      return false;
    if (c.bodyA >= bodies.size() || bodies[c.bodyA].mass <= 0.0f)
      return false;
  }
  return true;
}

void Solver::sequentialBodyStaticPrimalPass(float dt) {
  if (contacts.empty())
    return;

  const float invDt2 = 1.0f / (dt * dt);
  const float dt2 = dt * dt;
  const uint32_t bi = contacts[0].bodyA;
  Body &body = bodies[bi];
  if (body.mass <= 0.0f)
    return;

  const float boostFloor = kContactBoostFraction * body.mass * invDt2;

  // Phase A: aggregate all normal rows (stable support, matches PhysX 108).
  Mat66 lhs = body.getMassMatrix() / dt2;
  Vec6 disp(body.position - body.inertialPosition, body.deltaWInertial());
  Vec6 rhs = lhs * disp;

  Contact *deepest = nullptr;
  float bestGeom = 0.0f;

  for (auto &c : contacts) {
    if (c.bodyA != bi || c.bodyB != UINT32_MAX)
      continue;
    computeConstraint(c);
    const float g = contactGeomViolation(c);
    if (g < bestGeom) {
      bestGeom = g;
      deepest = &c;
    }
    const Vec6 &J = c.JA;
    const float pen = std::max(c.penalty[0], boostFloor);
    const float f = std::max(c.fmin[0],
                             std::min(c.fmax[0], pen * c.C[0] + c.lambda[0]));
    rhs += J * f;
    lhs += outer(J, J * pen);
  }

  {
    Vec6 delta = solveLDLT(lhs, rhs);
    body.position -= delta.linear();
    Quat dq(0, delta[3], delta[4], delta[5]);
    body.rotation =
        (body.rotation - dq * body.rotation * 0.5f).normalized();
  }

  // Friction: dual pass only (tangent rows in aggregated 6x6 over-constrain).
  (void)deepest;
}

void Solver::applyBodyStaticDepenetrationSweeps(uint32_t sweeps) {
  if (contacts.empty() || sweeps == 0)
    return;
  for (uint32_t sweep = 0; sweep < sweeps; ++sweep) {
    bool any = false;
    for (auto &c : contacts) {
      if (!isBodyVsStaticContact(c.bodyA, c.bodyB))
        continue;
      computeConstraintBodyStatic(c);
      const float viol = c.C[0];
      if (viol >= -1e-5f)
        continue;
      Body &body = bodies[c.bodyA];
      if (body.mass <= 0.0f)
        continue;
      const float corr = std::min(-viol, 0.05f);
      body.position += c.normal * corr;
      any = true;
    }
    if (!any)
      break;
  }
}

void Solver::applyLowIslandDynDynFrictionSweeps(uint32_t sweeps) {
  if (contacts.empty() || sweeps == 0 || dt <= 0.0f)
    return;
  uint32_t numDyn = 0;
  for (const auto &c : contacts) {
    if (!isBodyVsStaticContact(c.bodyA, c.bodyB))
      numDyn++;
  }
  if (numDyn > kDynDyn6x6FrictionMaxIslandContacts)
    return;

  const float invDt = 1.0f / dt;
  std::vector<Vec3> vLin(bodies.size()), vAng(bodies.size());
  std::vector<Vec3> vLin0(bodies.size()), vAng0(bodies.size());
  std::vector<bool> touched(bodies.size(), false);
  for (uint32_t i = 0; i < bodies.size(); ++i) {
    if (bodies[i].mass <= 0.0f) {
      vLin[i] = vAng[i] = vLin0[i] = vAng0[i] = Vec3();
      continue;
    }
    vLin[i] = vLin0[i] =
        (bodies[i].position - bodies[i].inertialPosition) * invDt;
    vAng[i] = vAng0[i] = bodies[i].deltaWInertial();
  }

  for (uint32_t sweep = 0; sweep < sweeps; ++sweep) {
    for (auto &c : contacts) {
      if (isBodyVsStaticContact(c.bodyA, c.bodyB) || c.friction <= 0.0f)
        continue;
      if (c.bodyA >= bodies.size() || c.bodyB >= bodies.size())
        continue;
      Body &bA = bodies[c.bodyA];
      Body &bB = bodies[c.bodyB];
      if (bA.mass <= 0.0f || bB.mass <= 0.0f)
        continue;
      computeConstraint(c);

      const Vec3 rA = bA.rotation.rotate(c.rA);
      const Vec3 rB = bB.rotation.rotate(c.rB);
      const Vec3 vA = vLin[c.bodyA] + vAng[c.bodyA].cross(rA);
      const Vec3 vB = vLin[c.bodyB] + vAng[c.bodyB].cross(rB);
      const Vec3 relV = vA - vB;
      const float jmax = std::fabs(c.lambda[0]) * c.friction * dt;
      if (jmax <= 0.0f)
        continue;

      Vec3 t1, t2;
      if (std::fabs(c.normal.y) > 0.9f)
        t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
      else
        t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
      t2 = c.normal.cross(t1);
      const Vec3 tangents[2] = {t1, t2};

      for (int ti = 0; ti < 2; ++ti) {
        const Vec3 &t = tangents[ti];
        const float vn = relV.dot(t);
        const Vec3 rCrossT_A = rA.cross(t);
        const Vec3 rCrossT_B = rB.cross(t);
        const float kEff = bA.invMass + bB.invMass +
                           rCrossT_A.dot(bA.invInertiaWorld * rCrossT_A) +
                           rCrossT_B.dot(bB.invInertiaWorld * rCrossT_B);
        if (kEff <= 1e-10f)
          continue;
        float j = -vn / kEff;
        j = std::max(-jmax, std::min(jmax, j));
        const Vec3 impulse = t * j;
        if (bA.mass > 0.0f) {
          touched[c.bodyA] = true;
          vLin[c.bodyA] += impulse * bA.invMass;
          vAng[c.bodyA] += bA.invInertiaWorld * rCrossT_A * j;
        }
        if (bB.mass > 0.0f) {
          touched[c.bodyB] = true;
          vLin[c.bodyB] -= impulse * bB.invMass;
          vAng[c.bodyB] -= bB.invInertiaWorld * rCrossT_B * j;
        }
      }
    }
  }

  for (uint32_t i = 0; i < bodies.size(); ++i) {
    if (!touched[i] || bodies[i].mass <= 0.0f)
      continue;
    bodies[i].position += (vLin[i] - vLin0[i]) * dt;
    const Vec3 dTheta = (vAng[i] - vAng0[i]) * dt;
    if (dTheta.length2() > 1e-16f) {
      Quat dq(0, dTheta.x, dTheta.y, dTheta.z);
      bodies[i].rotation =
          (bodies[i].rotation - dq * bodies[i].rotation * 0.5f).normalized();
    }
  }
}

void Solver::sequentialDynDynFrictionPass(float dt) {
  if (contacts.empty() || dt <= 0.0f)
    return;
  uint32_t numDyn = 0;
  for (const auto &c : contacts) {
    if (!isBodyVsStaticContact(c.bodyA, c.bodyB))
      numDyn++;
  }
  if (enableDynDynFrictionDiagnostics) {
    dynDynFrictionLastStats.invocationCount++;
    dynDynFrictionLastStats.dynamicContactCount =
        std::max(dynDynFrictionLastStats.dynamicContactCount, numDyn);
  }
  if (numDyn <= kDynDyn6x6FrictionMaxIslandContacts)
    return;

  const float invDt = 1.0f / dt;
  if (enableDynDynFrictionDiagnostics)
    dynDynFrictionLastStats.activeInvocationCount++;
  const auto sampleMomentum = [&]() {
    Vec3 linear;
    Vec3 angular;
    for (const Body &body : bodies) {
      if (body.mass <= 0.0f)
        continue;
      const Vec3 velocity =
          (body.position - body.inertialPosition) * invDt;
      const Vec3 angularVelocity = body.deltaWInertial() * invDt;
      const Vec3 localAngularVelocity =
          body.rotation.conjugate().rotate(angularVelocity);
      const Vec3 spinAngularMomentum = body.rotation.rotate(
          body.inertiaTensor * localAngularVelocity);
      const Vec3 bodyLinearMomentum = velocity * body.mass;
      linear += bodyLinearMomentum;
      angular += body.position.cross(bodyLinearMomentum) +
                 spinAngularMomentum;
    }
    return std::pair<Vec3, Vec3>(linear, angular);
  };
  std::pair<Vec3, Vec3> momentumBefore;
  if (enableDynDynFrictionDiagnostics)
    momentumBefore = sampleMomentum();
  for (auto &c : contacts) {
    if (isBodyVsStaticContact(c.bodyA, c.bodyB) || c.friction <= 0.0f)
      continue;
    if (c.bodyA >= bodies.size() || c.bodyB >= bodies.size())
      continue;
    Body &bA = bodies[c.bodyA];
    Body &bB = bodies[c.bodyB];
    if (bA.mass <= 0.0f || bB.mass <= 0.0f)
      continue;
    computeConstraint(c);

    const Vec3 rA = bA.rotation.rotate(c.rA);
    const Vec3 rB = bB.rotation.rotate(c.rB);
    const Vec3 vA =
        (bA.position - bA.inertialPosition) * invDt +
        bA.deltaWInertial().cross(rA);
    const Vec3 vB =
        (bB.position - bB.inertialPosition) * invDt +
        bB.deltaWInertial().cross(rB);
    const Vec3 relV = vA - vB;
    const float jmax = std::fabs(c.lambda[0]) * c.friction * dt;
    if (enableDynDynFrictionDiagnostics) {
      dynDynFrictionLastStats.maxNormalImpulseLimit =
          std::max(dynDynFrictionLastStats.maxNormalImpulseLimit, jmax);
    }
    if (jmax <= 0.0f)
      continue;

    Vec3 t1, t2;
    if (std::fabs(c.normal.y) > 0.9f)
      t1 = c.normal.cross(Vec3(1, 0, 0)).normalized();
    else
      t1 = c.normal.cross(Vec3(0, 1, 0)).normalized();
    t2 = c.normal.cross(t1);
    const Vec3 tangents[2] = {t1, t2};

    for (int ti = 0; ti < 2; ++ti) {
      const Vec3 &t = tangents[ti];
      const float vn = relV.dot(t);
      const Vec3 rCrossT_A = rA.cross(t);
      const Vec3 rCrossT_B = rB.cross(t);
      const float kEff = bA.invMass + bB.invMass +
                         rCrossT_A.dot(bA.invInertiaWorld * rCrossT_A) +
                         rCrossT_B.dot(bB.invInertiaWorld * rCrossT_B);
      if (kEff <= 1e-10f)
        continue;
      float j = -vn / kEff;
      j = std::max(-jmax, std::min(jmax, j));
      if (enableDynDynFrictionDiagnostics && std::fabs(j) > 1e-12f) {
        dynDynFrictionLastStats.tangentImpulseCount++;
        dynDynFrictionLastStats.totalAbsTangentImpulse += std::fabs(j);
      }
      const Vec3 impulse = t * j;
      if (bA.mass > 0.0f) {
        bA.position += impulse * bA.invMass * dt;
        const Vec3 dTheta = bA.invInertiaWorld * rCrossT_A * j * dt;
        Quat dq(0, dTheta.x, dTheta.y, dTheta.z);
        bA.rotation = useFrictionAngularImpulseSignProbe
                          ? (bA.rotation + dq * bA.rotation * 0.5f).normalized()
                          : (bA.rotation - dq * bA.rotation * 0.5f).normalized();
      }
      if (bB.mass > 0.0f) {
        bB.position -= impulse * bB.invMass * dt;
        const Vec3 dTheta = bB.invInertiaWorld * rCrossT_B * (-j) * dt;
        Quat dq(0, dTheta.x, dTheta.y, dTheta.z);
        bB.rotation = useFrictionAngularImpulseSignProbe
                          ? (bB.rotation + dq * bB.rotation * 0.5f).normalized()
                          : (bB.rotation - dq * bB.rotation * 0.5f).normalized();
      }
    }
  }
  if (enableDynDynFrictionDiagnostics) {
    const std::pair<Vec3, Vec3> momentumAfter = sampleMomentum();
    dynDynFrictionLastStats.maxLinearMomentumDelta =
        std::max(dynDynFrictionLastStats.maxLinearMomentumDelta,
                 (momentumAfter.first - momentumBefore.first).length());
    dynDynFrictionLastStats.maxAngularMomentumDelta =
        std::max(dynDynFrictionLastStats.maxAngularMomentumDelta,
                 (momentumAfter.second - momentumBefore.second).length());
  }
}

void Solver::warmstart() {
  for (auto &c : contacts) {
    for (int i = 0; i < 3; i++) {
      c.lambda[i] = c.lambda[i] * alpha * gamma;
      c.penalty[i] =
          std::max(PENALTY_MIN, std::min(PENALTY_MAX, c.penalty[i] * gamma));
    }
  }
  // Soft body AVBD warmstart (penalty only, no elastic dual)
  for (auto &sb : softBodies) {
    for (auto &ac : sb.attachments)
      ac.k = std::max(1e3f, std::min(ac.kMax, ac.k * gamma));
    for (auto &kp : sb.pins)
      kp.k = std::max(1e3f, std::min(kp.kMax, kp.k * gamma));
  }
  for (auto &sc : softContacts) {
    sc.k = std::min(1e4f, sc.ke);
    if (sc.rigidBodyIdx != UINT32_MAX &&
        sc.particleIdx < softParticles.size() &&
        softParticles[sc.particleIdx].invMass <= 0.0f) {
      sc.lambda = sc.lambda * alpha * gamma;
      sc.k = std::max(1e3f, std::min(sc.ke, sc.k * gamma));
      for (int ti = 0; ti < 2; ++ti) {
        sc.lambdaTangent[ti] = sc.lambdaTangent[ti] * alpha * gamma;
        sc.penTangent[ti] = std::max(
            PENALTY_MIN, std::min(PENALTY_MAX, sc.penTangent[ti] * gamma));
      }
    }
  }
}

bool Solver::solveFixedD6IslandPcgProbe(float dt) {
  IslandBodyMap bodyMap;
  bodyMap.bodyToSlot.assign(bodies.size(), -1);
  for (const D6Joint &joint : d6Joints) {
    if (!isFixedD6RowSet(joint))
      continue;
    const uint32_t endpoints[2] = {joint.bodyA, joint.bodyB};
    for (uint32_t bodyId : endpoints) {
      if (bodyId >= bodies.size() || bodies[bodyId].mass <= 0.0f ||
          bodyMap.bodyToSlot[bodyId] >= 0)
        continue;
      bodyMap.bodyToSlot[bodyId] =
          static_cast<int32_t>(bodyMap.slotToBody.size());
      bodyMap.slotToBody.push_back(bodyId);
    }
  }
  if (bodyMap.slotToBody.empty()) {
    islandPcgLastStats = IslandPcgStats();
    islandPcgLastStats.converged = true;
    return true;
  }

  const float dt2 = dt * dt;
  std::vector<Mat66> inertialBlocks(bodyMap.slotToBody.size());
  std::vector<Vec6> inertialGradient(bodyMap.slotToBody.size());
  for (size_t slot = 0; slot < bodyMap.slotToBody.size(); ++slot) {
    const Body &body = bodies[bodyMap.slotToBody[slot]];
    inertialBlocks[slot] = body.getMassMatrix() / dt2;
    const Vec6 displacement(body.position - body.inertialPosition,
                            body.deltaWInertial());
    inertialGradient[slot] = inertialBlocks[slot] * displacement;
  }

  IslandPcgSystem system;
  system.initialize(inertialBlocks, inertialGradient);
  uint32_t totalRows = 0;
  for (uint32_t jointIndex = 0; jointIndex < d6Joints.size(); ++jointIndex) {
    if (!isFixedD6RowSet(d6Joints[jointIndex]))
      continue;
    uint32_t emittedRows = 0;
    if (!emitFixedD6IslandRows(d6Joints[jointIndex], jointIndex, bodies,
                               bodyMap, dt, system, emittedRows)) {
      islandPcgLastStats = IslandPcgStats();
      islandPcgLastStats.breakdown = true;
      return false;
    }
    totalRows += emittedRows;
  }
  if (totalRows == 0) {
    islandPcgLastStats = IslandPcgStats();
    islandPcgLastStats.converged = true;
    return true;
  }

  std::vector<Vec6> delta;
  islandPcgLastStats = system.solvePcg(
      delta, 1e-7, static_cast<int>(bodyMap.slotToBody.size()) * 6);
  if (!islandPcgLastStats.converged || islandPcgLastStats.breakdown ||
      !islandPcgLastStats.finite || delta.size() != bodyMap.slotToBody.size())
    return false;

  for (size_t slot = 0; slot < bodyMap.slotToBody.size(); ++slot) {
    Body &body = bodies[bodyMap.slotToBody[slot]];
    body.position -= delta[slot].linear();
    const Vec3 angular = delta[slot].angular();
    const Quat dq(0.0f, angular.x, angular.y, angular.z);
    body.rotation =
        (body.rotation - dq * body.rotation * 0.5f).normalized();
  }
  return true;
}

static void updateFixedD6IslandPcgProbeDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt);
static void updateRevoluteD6IslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay);
static void updatePrismaticD6IslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay);
static void updateLinearXVelocityDriveIslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay);
static void updateSingleAxisAngularVelocityDriveIslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay, int axisIndex);
static void updateSlerpVelocityDriveIslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay);

bool Solver::solveContactIslandPcgProbe(float dt) {
  contactIslandPcgLastStats = IslandPcgStats();
  if (contacts.empty()) {
    contactIslandPcgLastStats.converged = true;
    return true;
  }

  // Give the shared objective one canonical dynamic-contact orientation.
  // Algebraically equivalent endpoint swaps otherwise retain opposite row
  // directions, which is enough to seed divergent floating-point trajectories
  // in a marginal stack. Preserve the world tangent force and the corresponding
  // C0 coordinates while moving the lower body id to endpoint A.
  for (Contact &contact : contacts)
    canonicalizeSharedContactOrientation(contact, bodies);

  IslandBodyMap bodyMap;
  bodyMap.bodyToSlot.assign(bodies.size(), -1);
  std::vector<uint8_t> bodyParticipates(bodies.size(), 0);
  for (const Contact &contact : contacts) {
    const uint32_t endpoints[2] = {contact.bodyA, contact.bodyB};
    for (uint32_t bodyId : endpoints) {
      if (bodyId < bodies.size() && bodies[bodyId].mass > 0.0f)
        bodyParticipates[bodyId] = 1;
    }
  }
  // A mixed island objective must be closed over every supported D6 row it
  // owns.  Include both endpoints before slot construction; otherwise a
  // contact correction can silently discard the joint reaction on a body
  // outside the narrow-phase participant set.
  for (const D6Joint &joint : d6Joints) {
    if (!isSphericalD6RowSet(joint) && !isFixedD6RowSet(joint) &&
        !isRevoluteD6RowSet(joint) && !isPrismaticD6RowSet(joint) &&
        !isSupportedLinearXVelocityDriveD6RowSet(joint) &&
        !isSupportedSingleAxisAngularVelocityDriveD6RowSet(joint) &&
        !isSupportedSlerpVelocityDriveD6RowSet(joint))
      continue;
    const uint32_t endpoints[2] = {joint.bodyA, joint.bodyB};
    for (uint32_t bodyId : endpoints) {
      if (bodyId < bodies.size() && bodies[bodyId].mass > 0.0f)
        bodyParticipates[bodyId] = 1;
    }
  }
  // Keep island slots independent of contact traversal and endpoint order.
  for (uint32_t bodyId = 0; bodyId < bodies.size(); ++bodyId) {
    if (!bodyParticipates[bodyId])
      continue;
    bodyMap.bodyToSlot[bodyId] =
        static_cast<int32_t>(bodyMap.slotToBody.size());
    bodyMap.slotToBody.push_back(bodyId);
  }
  if (bodyMap.slotToBody.empty()) {
    contactIslandPcgLastStats.converged = true;
    return true;
  }

  const float dt2 = dt * dt;
  std::vector<Mat66> inertialBlocks(bodyMap.slotToBody.size());
  std::vector<Vec6> inertialGradient(bodyMap.slotToBody.size());
  for (size_t slot = 0; slot < bodyMap.slotToBody.size(); ++slot) {
    const Body &body = bodies[bodyMap.slotToBody[slot]];
    inertialBlocks[slot] = body.getMassMatrix() / dt2;
    const Vec6 displacement(body.position - body.inertialPosition,
                            body.deltaWInertial());
    inertialGradient[slot] = inertialBlocks[slot] * displacement;
  }

  IslandPcgSystem system;
  system.initialize(inertialBlocks, inertialGradient);
  uint32_t totalRows = 0;
  for (uint32_t jointIndex = 0; jointIndex < d6Joints.size(); ++jointIndex) {
    const D6Joint &joint = d6Joints[jointIndex];
    if (!isSphericalD6RowSet(joint) && !isFixedD6RowSet(joint) &&
        !isRevoluteD6RowSet(joint) && !isPrismaticD6RowSet(joint) &&
        !isSupportedLinearXVelocityDriveD6RowSet(joint) &&
        !isSupportedSingleAxisAngularVelocityDriveD6RowSet(joint) &&
        !isSupportedSlerpVelocityDriveD6RowSet(joint))
      continue;
    uint32_t emittedRows = 0;
    bool emitted = false;
    if (isSphericalD6RowSet(joint)) {
      emitted = emitSphericalD6IslandRows(
          joint, jointIndex, bodies, bodyMap, dt, system, emittedRows);
    } else if (isFixedD6RowSet(joint)) {
      emitted = emitFixedD6IslandRows(
          joint, jointIndex, bodies, bodyMap, dt, system, emittedRows);
    } else if (isRevoluteD6RowSet(joint)) {
      emitted = emitRevoluteD6IslandRows(
          joint, jointIndex, bodies, bodyMap, dt, system, emittedRows);
    } else if (isPrismaticD6RowSet(joint)) {
      emitted = emitPrismaticD6IslandRows(
          joint, jointIndex, bodies, bodyMap, dt, system, emittedRows);
    } else if (isSupportedLinearXVelocityDriveD6RowSet(joint)) {
      const size_t firstRow = system.rows().size();
      emitted = emitLinearXVelocityDriveIslandRow(
          joint, jointIndex, bodies, bodyMap, dt, system, emittedRows);
      if (emitted && emittedRows == 1 && firstRow < system.rows().size()) {
        const IslandPcgRow &row = system.rows()[firstRow];
        ++linearDriveIslandLastStats.emittedRowCount;
        if (isLinearXAccelerationVelocityDriveD6RowSet(joint))
          ++linearDriveIslandLastStats.accelerationRowCount;
        if (row.activeMode == 4)
          ++linearDriveIslandLastStats.saturatedRowCount;
        else
          ++linearDriveIslandLastStats.unsaturatedRowCount;
        linearDriveIslandLastStats.maxAbsForce =
            std::max(linearDriveIslandLastStats.maxAbsForce,
                     std::fabs(row.force));
        linearDriveIslandLastStats.maxForceLimit =
            std::max(linearDriveIslandLastStats.maxForceLimit,
                     joint.driveLinearForce.x);
      }
    } else {
      const size_t firstRow = system.rows().size();
      const int angularAxisIndex =
          getSupportedSingleAxisAngularVelocityDriveIndex(joint);
      if (angularAxisIndex >= 0) {
        emitted = emitSingleAxisAngularVelocityDriveIslandRow(
            joint, jointIndex, bodies, bodyMap, dt, angularAxisIndex,
            system, emittedRows);
        angularDriveIslandLastStats.maxTorqueLimit =
            std::max(angularDriveIslandLastStats.maxTorqueLimit,
                     (&joint.driveAngularForce.x)[angularAxisIndex]);
      } else {
        emitted = emitSlerpVelocityDriveIslandRows(
            joint, jointIndex, bodies, bodyMap, dt, system, emittedRows);
        angularDriveIslandLastStats.maxTorqueLimit =
            std::max(angularDriveIslandLastStats.maxTorqueLimit,
                     joint.driveAngularForce.z);
      }
      if (emitted && firstRow + emittedRows <= system.rows().size()) {
        angularDriveIslandLastStats.emittedRowCount += emittedRows;
        if (joint.driveAccelerationFlags != 0)
          angularDriveIslandLastStats.accelerationRowCount += emittedRows;
        for (size_t rowIndex = firstRow;
             rowIndex < firstRow + emittedRows; ++rowIndex) {
          const IslandPcgRow &row = system.rows()[rowIndex];
          if (row.activeMode == 4)
            ++angularDriveIslandLastStats.saturatedRowCount;
          else
            ++angularDriveIslandLastStats.unsaturatedRowCount;
          angularDriveIslandLastStats.maxAbsTorque =
              std::max(angularDriveIslandLastStats.maxAbsTorque,
                       std::fabs(row.force));
        }
      }
    }
    if (!emitted) {
      contactIslandPcgLastStats.breakdown = true;
      return false;
    }
    totalRows += emittedRows;
  }
  std::vector<FrozenContactIslandRowSet> frozenRows(contacts.size());
  struct ContactOrderKey {
    uint32_t body0;
    uint32_t body1;
    Vec3 anchor0;
    Vec3 anchor1;
    Vec3 normal0;
    float depth;
    float friction;
    uint32_t contactIndex;
  };
  std::vector<ContactOrderKey> contactOrder;
  contactOrder.reserve(contacts.size());
  for (uint32_t contactIndex = 0; contactIndex < contacts.size();
       ++contactIndex) {
    const Contact &contact = contacts[contactIndex];
    const bool dynamicPair = contact.bodyB < bodies.size();
    const bool aFirst = !dynamicPair || contact.bodyA < contact.bodyB;
    ContactOrderKey key;
    key.body0 = aFirst ? contact.bodyA : contact.bodyB;
    key.body1 = dynamicPair
                    ? (aFirst ? contact.bodyB : contact.bodyA)
                    : UINT32_MAX;
    key.anchor0 = aFirst ? contact.rA : contact.rB;
    key.anchor1 = aFirst ? contact.rB : contact.rA;
    key.normal0 = aFirst ? contact.normal : -contact.normal;
    key.depth = contact.depth;
    key.friction = contact.friction;
    key.contactIndex = contactIndex;
    contactOrder.push_back(key);
  }
  const auto vecLess = [](const Vec3 &a, const Vec3 &b) {
    if (a.x != b.x)
      return a.x < b.x;
    if (a.y != b.y)
      return a.y < b.y;
    return a.z < b.z;
  };
  std::sort(contactOrder.begin(), contactOrder.end(),
            [&vecLess](const ContactOrderKey &a,
                       const ContactOrderKey &b) {
              if (a.body0 != b.body0)
                return a.body0 < b.body0;
              if (a.body1 != b.body1)
                return a.body1 < b.body1;
              if (a.anchor0.x != b.anchor0.x ||
                  a.anchor0.y != b.anchor0.y ||
                  a.anchor0.z != b.anchor0.z)
                return vecLess(a.anchor0, b.anchor0);
              if (a.anchor1.x != b.anchor1.x ||
                  a.anchor1.y != b.anchor1.y ||
                  a.anchor1.z != b.anchor1.z)
                return vecLess(a.anchor1, b.anchor1);
              if (a.normal0.x != b.normal0.x ||
                  a.normal0.y != b.normal0.y ||
                  a.normal0.z != b.normal0.z)
                return vecLess(a.normal0, b.normal0);
              if (a.depth != b.depth)
                return a.depth < b.depth;
              return a.friction < b.friction;
            });
  for (const ContactOrderKey &key : contactOrder) {
    const uint32_t contactIndex = key.contactIndex;
    Contact &contact = contacts[contactIndex];
    if (isBodyVsStaticContact(contact.bodyA, contact.bodyB))
      computeConstraintBodyStatic(contact);
    else
      computeConstraint(contact);

    FrozenContactIslandRowSet &rowSet = frozenRows[contactIndex];
    rowSet.penalty[0] = contact.penalty[0];
    rowSet.force[0] =
        std::max(contact.fmin[0],
                 std::min(contact.fmax[0],
                          rowSet.penalty[0] * contact.C[0] +
                              contact.lambda[0]));
    rowSet.active[0] = rowSet.force[0] < 0.0f;
    rowSet.activeMode[0] = 1;

    const float tangentBound =
        std::fabs(rowSet.force[0]) * contact.friction;
    rowSet.tangentForceBound = tangentBound;
    if (rowSet.active[0] && tangentBound > 0.0f) {
      rowSet.active[1] = rowSet.active[2] = true;
      rowSet.penalty[1] = contact.penalty[1];
      rowSet.penalty[2] = contact.penalty[2];
      rowSet.force[1] =
          rowSet.penalty[1] * contact.C[1] + contact.lambda[1];
      rowSet.force[2] =
          rowSet.penalty[2] * contact.C[2] + contact.lambda[2];
      const float tangentMagnitude =
          std::sqrt(rowSet.force[1] * rowSet.force[1] +
                    rowSet.force[2] * rowSet.force[2]);
      if (tangentMagnitude > tangentBound) {
        const float scale = tangentBound / tangentMagnitude;
        rowSet.force[1] *= scale;
        rowSet.force[2] *= scale;
      }
      rowSet.activeMode[1] = rowSet.activeMode[2] = 2;
    }

    uint32_t emittedRows = 0;
    if (!emitFrozenContactIslandRows(contact, contactIndex, bodies, bodyMap,
                                     rowSet, system, emittedRows)) {
      contactIslandPcgLastStats.breakdown = true;
      return false;
    }
    totalRows += emittedRows;
  }
  if (totalRows == 0) {
    contactIslandPcgLastStats.converged = true;
    return true;
  }

  std::vector<Vec6> delta;
  contactIslandPcgLastStats = system.solvePcg(
      delta, 1e-7, static_cast<int>(bodyMap.slotToBody.size()) * 12);
  if (!contactIslandPcgLastStats.converged ||
      contactIslandPcgLastStats.breakdown ||
      !contactIslandPcgLastStats.finite ||
      delta.size() != bodyMap.slotToBody.size())
    return false;

  for (size_t slot = 0; slot < bodyMap.slotToBody.size(); ++slot) {
    Body &body = bodies[bodyMap.slotToBody[slot]];
    body.position -= delta[slot].linear();
    const Vec3 angular = delta[slot].angular();
    const Quat dq(0.0f, angular.x, angular.y, angular.z);
    body.rotation =
        (body.rotation - dq * body.rotation * 0.5f).normalized();
  }

  for (uint32_t contactIndex = 0; contactIndex < contacts.size();
       ++contactIndex) {
    Contact &contact = contacts[contactIndex];
    if (isBodyVsStaticContact(contact.bodyA, contact.bodyB))
      computeConstraintBodyStatic(contact);
    else
      computeConstraint(contact);
    const FrozenContactIslandRowSet &rowSet = frozenRows[contactIndex];

    const float normalRaw =
        contact.lambda[0] + rowSet.penalty[0] * contact.C[0];
    contact.lambda[0] =
        std::max(contact.fmin[0], std::min(contact.fmax[0], normalRaw));
    if (contact.lambda[0] > contact.fmin[0] &&
        contact.lambda[0] < contact.fmax[0]) {
      contact.penalty[0] = std::min(
          contact.penalty[0] + beta * std::fabs(contact.C[0]), PENALTY_MAX);
    }

    if (rowSet.active[1]) {
      float tangent[2] = {
          contact.lambda[1] + rowSet.penalty[1] * contact.C[1],
          contact.lambda[2] + rowSet.penalty[2] * contact.C[2]};
      const float tangentBound =
          std::fabs(contact.lambda[0]) * contact.friction;
      const float tangentMagnitude =
          std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1]);
      const bool saturated = tangentMagnitude > tangentBound;
      if (saturated && tangentMagnitude > 0.0f) {
        const float scale = tangentBound / tangentMagnitude;
        tangent[0] *= scale;
        tangent[1] *= scale;
      }
      contact.lambda[1] = tangent[0];
      contact.lambda[2] = tangent[1];
      if (!saturated) {
        for (int tangentIndex = 1; tangentIndex < 3; ++tangentIndex) {
          contact.penalty[tangentIndex] =
              std::min(contact.penalty[tangentIndex] +
                           beta * std::fabs(contact.C[tangentIndex]),
                       PENALTY_MAX);
        }
      }
    } else {
      contact.lambda[1] = contact.lambda[2] = 0.0f;
    }
  }
  return true;
}

static void updateFixedD6IslandPcgProbeDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt) {
  const bool dynamicA = joint.bodyA < bodies.size() &&
                        bodies[joint.bodyA].mass > 0.0f;
  const bool dynamicB = joint.bodyB < bodies.size() &&
                        bodies[joint.bodyB].mass > 0.0f;
  const Body *bodyA = dynamicA ? &bodies[joint.bodyA] : nullptr;
  const Body *bodyB = dynamicB ? &bodies[joint.bodyB] : nullptr;
  const Quat rotationA = bodyA ? bodyA->rotation : Quat();
  const Quat rotationB = bodyB ? bodyB->rotation : Quat();
  const Vec3 worldAnchorA =
      bodyA ? bodyA->position + rotationA.rotate(joint.anchorA)
            : joint.anchorA;
  const Vec3 worldAnchorB =
      bodyB ? bodyB->position + rotationB.rotate(joint.anchorB)
            : joint.anchorB;
  const Vec3 error = worldAnchorA - worldAnchorB;
  const float massA = bodyA ? bodyA->mass : 0.0f;
  const float massB = bodyB ? bodyB->mass : 0.0f;
  const float penalty =
      std::max(joint.rho, std::max(massA, massB) / (dt * dt));
  joint.lambdaLinear += error * penalty;
  for (int axis = 0; axis < 3; ++axis) {
    (&joint.lambdaAngular.x)[axis] +=
        penalty * computeAngularError(rotationA, rotationB,
                                      joint.localFrameA, joint.localFrameB,
                                      axis);
  }
}

static void updateRevoluteD6IslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay) {
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Vec3 worldAnchorA =
      endpointA ? endpointA->position + rotationA.rotate(joint.anchorA)
                : joint.anchorA;
  const Vec3 worldAnchorB =
      endpointB ? endpointB->position + rotationB.rotate(joint.anchorB)
                : joint.anchorB;
  const float rhoDual = d6ComputeRhoDual(
      joint.bodyA, joint.bodyB, joint.rho, bodies, dt * dt);
  joint.lambdaLinear =
      joint.lambdaLinear * lambdaDecay +
      (worldAnchorA - worldAnchorB) * rhoDual;

  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Quat frameB =
      (endpointB ? rotationB * joint.localFrameB : joint.localFrameB)
          .normalized();
  const Vec3 twistA = frameA.rotate(Vec3(1.0f, 0.0f, 0.0f));
  const Vec3 twistB = frameB.rotate(Vec3(1.0f, 0.0f, 0.0f));
  Vec3 midAxis, perpendicular1, perpendicular2;
  if (buildRevoluteMidAxisBasis(twistA, twistB, midAxis, perpendicular1,
                                 perpendicular2)) {
    const Vec3 axisViolation = twistA.cross(twistB);
    joint.lambdaAngular.y =
        joint.lambdaAngular.y * lambdaDecay +
        axisViolation.dot(perpendicular1) * rhoDual;
    joint.lambdaAngular.z =
        joint.lambdaAngular.z * lambdaDecay +
        axisViolation.dot(perpendicular2) * rhoDual;
  }

  if (joint.getAngularMotion(0) == 1) {
    const float angularError =
        computeRevoluteSymmetricTwistError(frameA, frameB);
    const float violation = computeAngularLimitViolation(
        angularError, joint.angularLimitLower[0],
        joint.angularLimitUpper[0]);
    float lambda = joint.lambdaLimitAngular[0] * lambdaDecay +
                   violation * rhoDual;
    if (joint.angularLimitLower[0] < joint.angularLimitUpper[0]) {
      if (violation > 0.0f || joint.lambdaLimitAngular[0] > 0.0f)
        lambda = std::max(0.0f, lambda);
      else if (violation < 0.0f || joint.lambdaLimitAngular[0] < 0.0f)
        lambda = std::min(0.0f, lambda);
      else
        lambda = 0.0f;
    }
    joint.lambdaLimitAngular[0] = lambda;
  }
}

static void updatePrismaticD6IslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay) {
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Vec3 worldAnchorA =
      endpointA ? endpointA->position + rotationA.rotate(joint.anchorA)
                : joint.anchorA;
  const Vec3 worldAnchorB =
      endpointB ? endpointB->position + rotationB.rotate(joint.anchorB)
                : joint.anchorB;
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
  const float rhoDual = d6ComputeRhoDual(
      joint.bodyA, joint.bodyB, joint.rho, bodies, dt * dt);
  for (int axis = 1; axis < 3; ++axis) {
    (&joint.lambdaLinear.x)[axis] =
        (&joint.lambdaLinear.x)[axis] * lambdaDecay +
        linearError.dot(axes[axis]) * rhoDual;
  }

  if (joint.getLinearMotion(0) == 1) {
    const float distance = -linearError.dot(axes[0]);
    const float violation = computeAngularLimitViolation(
        distance, joint.linearLimitLower[0], joint.linearLimitUpper[0]);
    float lambda = joint.lambdaLimitLinear[0] * lambdaDecay +
                   violation * rhoDual;
    if (joint.linearLimitLower[0] < joint.linearLimitUpper[0]) {
      const float signReference =
          std::fabs(violation) > 1e-6f
              ? violation
              : (std::fabs(joint.lambdaLimitLinear[0]) > 1e-6f
                     ? joint.lambdaLimitLinear[0]
                     : 0.0f);
      if (signReference > 0.0f)
        lambda = std::max(0.0f, lambda);
      else if (signReference < 0.0f)
        lambda = std::min(0.0f, lambda);
      else
        lambda = 0.0f;
    }
    joint.lambdaLimitLinear[0] = lambda;
  }

  const Vec3 angularError = computeD6SymmetricAngularError(frameA, frameB);
  joint.lambdaAngular =
      joint.lambdaAngular * lambdaDecay + angularError * rhoDual;
}

static void updateLinearXVelocityDriveIslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay) {
  if (isLinearXAccelerationVelocityDriveD6RowSet(joint)) {
    // The effective-mass-scaled acceleration objective is physical in the
    // primal row.  A separate AL drive dual is not response-scaled and would
    // reintroduce endpoint-mass dependence across contact-PCG iterations.
    joint.lambdaDriveLinear.x = 0.0f;
    return;
  }
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const bool dynamicA = endpointA && endpointA->mass > 0.0f;
  const bool dynamicB = endpointB && endpointB->mass > 0.0f;
  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Vec3 axis = frameA.rotate(Vec3(1.0f, 0.0f, 0.0f));
  const Vec3 rA =
      dynamicA ? rotationA.rotate(joint.anchorA) : Vec3();
  const Vec3 rB =
      dynamicB ? rotationB.rotate(joint.anchorB) : Vec3();
  const Vec3 displacementA =
      dynamicA
          ? (endpointA->position + rotationA.rotate(joint.anchorA)) -
                (endpointA->initialPosition +
                 endpointA->initialRotation.rotate(joint.anchorA))
          : Vec3();
  const Vec3 displacementB =
      dynamicB
          ? (endpointB->position + rotationB.rotate(joint.anchorB)) -
                (endpointB->initialPosition +
                 endpointB->initialRotation.rotate(joint.anchorB))
          : Vec3();
  const float violation =
      (displacementB - displacementA).dot(axis) -
      joint.driveLinearVelocity.x * dt;
  const float drivePenalty = computeLinearXVelocityDrivePenalty(
      joint, endpointA, endpointB, rA, rB, axis, dt);
  const float rhoDual = std::min(
      drivePenalty,
      d6ComputeRhoDual(joint.bodyA, joint.bodyB, joint.rho, bodies,
                       dt * dt));
  const float forceLimit = joint.driveLinearForce.x;
  joint.lambdaDriveLinear.x = updateClampedLinearDriveDual(
      joint.lambdaDriveLinear.x, violation, rhoDual, forceLimit,
      lambdaDecay);
}

static void updateSingleAxisAngularVelocityDriveIslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay, int axisIndex) {
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
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
      endpointA && endpointA->mass > 0.0f
          ? computeWorldRotationDelta(rotationA,
                                      endpointA->initialRotation)
          : Vec3();
  const Vec3 deltaB =
      endpointB && endpointB->mass > 0.0f
          ? computeWorldRotationDelta(rotationB,
                                      endpointB->initialRotation)
          : Vec3();
  const float violation =
      (deltaB - deltaA).dot(axis) +
      (&joint.driveAngularVelocity.x)[axisIndex] * dt;
  const float drivePenalty = computeAngularAxisVelocityDrivePenalty(
      joint, endpointA, endpointB, axis, axisIndex,
      joint.driveAccelerationFlags != 0, dt);
  const float rhoDual = std::min(
      drivePenalty,
      d6ComputeRhoDual(joint.bodyA, joint.bodyB, joint.rho, bodies,
                       dt * dt));
  float &lambda = (&joint.lambdaDriveAngular.x)[axisIndex];
  lambda = updateClampedLinearDriveDual(
      lambda, violation, rhoDual,
      (&joint.driveAngularForce.x)[axisIndex], lambdaDecay);
}

static void updateSlerpVelocityDriveIslandPcgDual(
    D6Joint &joint, const std::vector<Body> &bodies, float dt,
    float lambdaDecay) {
  const Body *endpointA =
      joint.bodyA < bodies.size() ? &bodies[joint.bodyA] : nullptr;
  const Body *endpointB =
      joint.bodyB < bodies.size() ? &bodies[joint.bodyB] : nullptr;
  const Quat rotationA = endpointA ? endpointA->rotation : Quat();
  const Quat rotationB = endpointB ? endpointB->rotation : Quat();
  const Quat frameA =
      (endpointA ? rotationA * joint.localFrameA : joint.localFrameA)
          .normalized();
  const Vec3 deltaA =
      endpointA && endpointA->mass > 0.0f
          ? computeWorldRotationDelta(rotationA,
                                      endpointA->initialRotation)
          : Vec3();
  const Vec3 deltaB =
      endpointB && endpointB->mass > 0.0f
          ? computeWorldRotationDelta(rotationB,
                                      endpointB->initialRotation)
          : Vec3();
  const Vec3 violation =
      deltaB - deltaA - frameA.rotate(joint.driveAngularVelocity) * dt;
  const Vec3 worldAxes[3] = {Vec3(1.0f, 0.0f, 0.0f),
                             Vec3(0.0f, 1.0f, 0.0f),
                             Vec3(0.0f, 0.0f, 1.0f)};
  const float baseRhoDual = d6ComputeRhoDual(
      joint.bodyA, joint.bodyB, joint.rho, bodies, dt * dt);
  for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
    const float drivePenalty = computeSlerpVelocityDrivePenalty(
        joint, endpointA, endpointB, worldAxes[axisIndex], dt);
    const float rhoDual = std::min(drivePenalty, baseRhoDual);
    float &lambda = (&joint.lambdaDriveAngular.x)[axisIndex];
    lambda = updateClampedLinearDriveDual(
        lambda, (&violation.x)[axisIndex], rhoDual,
        joint.driveAngularForce.z, lambdaDecay);
  }
}

// =============================================================================
// Soft body creation
// =============================================================================

uint32_t Solver::addSoftBody(const std::vector<Vec3>& vertices,
                             const std::vector<uint32_t>& tets,
                             const std::vector<uint32_t>& tris,
                             float youngsModulus_,
                             float poissonsRatio_,
                             float density_,
                             float damping_,
                             float bendingStiffness_,
                             float thickness_) {
  uint32_t particleStart = (uint32_t)softParticles.size();

  // Compute per-vertex mass from tet volumes (or uniform if no tets)
  std::vector<float> vertexMass(vertices.size(), 0.0f);
  if (!tets.empty()) {
    for (size_t i = 0; i + 3 < tets.size(); i += 4) {
      Vec3 e1 = vertices[tets[i+1]] - vertices[tets[i]];
      Vec3 e2 = vertices[tets[i+2]] - vertices[tets[i]];
      Vec3 e3 = vertices[tets[i+3]] - vertices[tets[i]];
      float vol = fabsf(e1.dot(e2.cross(e3)) / 6.0f);
      float tetMass = vol * density_;
      float perVertex = tetMass / 4.0f;
      vertexMass[tets[i]]   += perVertex;
      vertexMass[tets[i+1]] += perVertex;
      vertexMass[tets[i+2]] += perVertex;
      vertexMass[tets[i+3]] += perVertex;
    }
  } else if (!tris.empty()) {
    // Surface mesh: estimate mass from triangle area × thickness × density
    for (size_t i = 0; i + 2 < tris.size(); i += 3) {
      Vec3 e1 = vertices[tris[i+1]] - vertices[tris[i]];
      Vec3 e2 = vertices[tris[i+2]] - vertices[tris[i]];
      float area = e1.cross(e2).length() * 0.5f;
      float triMass = area * thickness_ * density_;
      float perVertex = triMass / 3.0f;
      vertexMass[tris[i]]   += perVertex;
      vertexMass[tris[i+1]] += perVertex;
      vertexMass[tris[i+2]] += perVertex;
    }
  }

  // Ensure minimum mass
  float minMass = 1e-4f;
  for (auto& m : vertexMass)
    m = std::max(m, minMass);

  // Create particles
  for (size_t i = 0; i < vertices.size(); i++) {
    SoftParticle sp;
    sp.position = vertices[i];
    sp.velocity = Vec3(0, 0, 0);
    sp.prevVelocity = Vec3(0, 0, 0);
    sp.initialPosition = vertices[i];
    sp.predictedPosition = vertices[i];
    sp.mass = vertexMass[i];
    sp.invMass = 1.0f / sp.mass;
    sp.damping = damping_;
    softParticles.push_back(sp);
  }

  // Create SoftBody
  SoftBody sb;
  sb.particleStart = particleStart;
  sb.particleCount = (uint32_t)vertices.size();
  sb.tetrahedra = tets;
  sb.triangles = tris;
  sb.youngsModulus = youngsModulus_;
  sb.poissonsRatio = poissonsRatio_;
  sb.density = density_;
  sb.damping = damping_;
  sb.bendingStiffness = bendingStiffness_;
  sb.thickness = thickness_;

  sb.buildElements(softParticles);

  softBodies.push_back(sb);
  return particleStart;
}

uint32_t Solver::addKinematicShell(const std::vector<Vec3> &positions) {
  const uint32_t start = (uint32_t)softParticles.size();
  SoftBody sb;
  sb.particleStart = start;
  sb.particleCount = (uint32_t)positions.size();
  sb.youngsModulus = 0.0f;
  sb.poissonsRatio = 0.0f;
  sb.density = 0.0f;
  sb.mu = 0.0f;
  sb.lambda = 0.0f;
  sb.adjacency.resize(sb.particleCount);
  for (const Vec3 &p : positions) {
    SoftParticle sp;
    sp.position = p;
    sp.velocity = Vec3(0, 0, 0);
    sp.prevVelocity = Vec3(0, 0, 0);
    sp.initialPosition = p;
    sp.predictedPosition = p;
    sp.outerPosition = p;
    sp.mass = 0.0f;
    sp.invMass = 0.0f;
    softParticles.push_back(sp);
  }
  softBodies.push_back(sb);
  return start;
}

// =============================================================================
// Main solver step
// =============================================================================

void Solver::step(float dt_) {
  dt = dt_;
  float invDt = 1.0f / dt;
  float dt2 = dt * dt;
  dynDynFrictionLastStats = DynDynFrictionPassStats();
  contactIslandPcgRoutedLastStep = false;
  linearDriveIslandLastStats = LinearDriveIslandStats();
  angularDriveIslandLastStats = AngularDriveIslandStats();

  // A native revolute motor owns velocity only.  Classify the complete
  // supported objective before prediction so a dynamic pair can preserve
  // its solve-start angular momentum.
  const RevoluteMotorVelocityOwner revoluteMotorVelocityOwner =
      classifyRevoluteMotorVelocityOwner(
          bodies, contacts, d6Joints, gearJoints, articulations,
          softBodies, softContacts, softParticles);

  if (useContactIslandPcgProbe) {
    for (Contact &contact : contacts)
      canonicalizeSharedContactOrientation(contact, bodies);
    std::sort(contacts.begin(), contacts.end(), sharedContactLess);
  }

  warmstart();

  // Never solve a partial mixed objective.  Until the island emitter covers
  // every rigid constraint family, route only contact-only plus the verified
  // fixed/pure-spherical/undriven-revolute/prismatic and isolated force- or
  // acceleration-mode linear-X, single-axis angular, and isolated SLERP
  // velocity-drive D6 families through the shared PCG path.
  // Unsupported rows fall back as a
  // whole so a later contact solve cannot erase their per-body correction.
  const bool routeContactIslandPcg =
      useContactIslandPcgProbe && !contacts.empty() && gearJoints.empty() &&
      articulations.empty() && softBodies.empty() && softContacts.empty() &&
      std::all_of(d6Joints.begin(), d6Joints.end(),
                  [](const D6Joint &joint) {
                    return isSphericalD6RowSet(joint) ||
                           isFixedD6RowSet(joint) ||
                           isRevoluteD6RowSet(joint) ||
                           isPrismaticD6RowSet(joint) ||
                           isSupportedLinearXVelocityDriveD6RowSet(joint) ||
                           isSupportedSingleAxisAngularVelocityDriveD6RowSet(
                               joint) ||
                           isSupportedSlerpVelocityDriveD6RowSet(joint);
                  });
  contactIslandPcgRoutedLastStep = routeContactIslandPcg;

  // Step 1: Build adjacency list from joints
  uint32_t nBodies = (uint32_t)bodies.size();
  std::vector<std::vector<uint32_t>> adj(nBodies);
  auto addEdge = [&](uint32_t a, uint32_t b) {
    if (a < nBodies && b < nBodies && a != UINT32_MAX && b != UINT32_MAX) {
      adj[a].push_back(b);
      adj[b].push_back(a);
    }
  };
  for (const auto &j : d6Joints)
    addEdge(j.bodyA, j.bodyB);
  for (const auto &j : gearJoints)
    addEdge(j.bodyA, j.bodyB);
  for (const auto &artic : articulations) {
    for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
      uint32_t child = artic.joints[ji].bodyIndex;
      uint32_t parent = artic.getParentBodyIndex(ji);
      addEdge(parent, child);
    }
  }

  // Step 2: Jacobi propagation of effective mass
  std::vector<float> mEff(nBodies);
  for (uint32_t i = 0; i < nBodies; i++)
    mEff[i] = bodies[i].mass;

  for (int d = 0; d < propagationDepth; d++) {
    std::vector<float> mNext(nBodies);
    for (uint32_t i = 0; i < nBodies; i++) {
      float neighborSum = 0.0f;
      for (uint32_t nb : adj[i])
        neighborSum += mEff[nb];
      mNext[i] = bodies[i].mass + propagationDecay * neighborSum;
    }
    mEff = mNext;
  }

  // Step 3: set penalty floor using propagated effective mass
  for (auto &c : contacts) {
    float augA = mEff[c.bodyA];
    float augB = (c.bodyB != UINT32_MAX) ? mEff[c.bodyB] : 0.0f;
    float massB = (c.bodyB != UINT32_MAX) ? bodies[c.bodyB].mass : 0.0f;

    float effectiveMass, scale;
    if (c.bodyB != UINT32_MAX && massB > 0.0f) {
      effectiveMass = std::max(augA, augB);
      scale = penaltyScaleDynDyn;
    } else {
      effectiveMass = augA;
      scale = kPenScaleBodyVsStatic;
    }
    float penFloor = std::max(PENALTY_MIN, scale * effectiveMass / dt2);
    for (int i = 0; i < 3; i++)
      c.penalty[i] = std::max(c.penalty[i], penFloor);
  }

  // Compute C0 for alpha blending
  for (auto &c : contacts)
    computeC0(c);

  // Warmstart bodies
  for (uint32_t bi = 0; bi < nBodies; ++bi) {
    Body &body = bodies[bi];
    if (body.mass <= 0)
      continue;
    body.updateInvInertiaWorld();

    // Inertial prediction
    body.inertialPosition =
        body.position + body.linearVelocity * dt + gravity * dt2;
    Quat angVel(0, body.angularVelocity.x, body.angularVelocity.y,
                body.angularVelocity.z);
    body.inertialRotation =
        (body.rotation + angVel * body.rotation * (0.5f * dt)).normalized();

    // Adaptive warmstarting
    Vec3 accel = (body.linearVelocity - body.prevLinearVelocity) * invDt;
    float gravLen = gravity.length();
    float accelWeight = 0.0f;
    if (gravLen > 1e-6f) {
      Vec3 gravDir = gravity.normalized();
      accelWeight =
          std::max(0.0f, std::min(1.0f, accel.dot(gravDir) / gravLen));
    }
    if (bodyTouchesStatic(bi) || bodyTouchesKinematicShell(bi))
      accelWeight = 0.0f;

    body.initialPosition = body.position;
    body.initialRotation = body.rotation;

    body.position = body.position + body.linearVelocity * dt +
                    gravity * (accelWeight * dt2);
    body.rotation = body.inertialRotation;
  }

  // Predict soft particles
  uint32_t nSoftParticles = (uint32_t)softParticles.size();
  for (uint32_t i = 0; i < nSoftParticles; i++) {
    SoftParticle &sp = softParticles[i];
    if (sp.invMass <= 0.0f) continue;
    sp.predictedPosition = sp.position + sp.velocity * dt + gravity * dt2;
    sp.initialPosition = sp.position;
    // AVBD elastic proximal warmstart (mirrors PhysX: retain fraction from prior timestep)
    sp.elasticK = sp.elasticK * 0.5f;
    // Adaptive warmstart (same as rigid bodies)
    Vec3 accel = (sp.velocity - sp.prevVelocity) * invDt;
    float gravLen = gravity.length();
    float accelWeight = 0.0f;
    if (gravLen > 1e-6f) {
      Vec3 gravDir = gravity.normalized();
      accelWeight = std::max(0.0f, std::min(1.0f, accel.dot(gravDir) / gravLen));
    }
    sp.position = sp.position + sp.velocity * dt + gravity * (accelWeight * dt2);
  }

  // =========================================================================
  // Compute sweep order (tree-structured for articulation chains)
  // =========================================================================
  std::vector<uint32_t> sweepOrder;
  {
    std::vector<bool> isArticBody(nBodies, false);
    std::vector<uint32_t> articOrder;
    if (useTreeSweep && !articulations.empty()) {
      for (const auto &artic : articulations) {
        for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
          uint32_t bi = artic.joints[ji].bodyIndex;
          if (bi < nBodies && !isArticBody[bi]) {
            isArticBody[bi] = true;
            articOrder.push_back(bi);
          }
        }
      }
    }
    // Non-articulation bodies first (any order)
    for (uint32_t i = 0; i < nBodies; i++) {
      if (!isArticBody[i] && bodies[i].mass > 0)
        sweepOrder.push_back(i);
    }
    // Articulation bodies in tree order (root → leaves)
    for (uint32_t bi : articOrder)
      sweepOrder.push_back(bi);
    // Add remaining dynamic non-artic bodies not yet added
    // (covers all bodies with mass > 0)
  }

  // =========================================================================
  // Anderson Acceleration state (positions only — quaternion mixing is
  // ill-conditioned for AA linear extrapolation)
  // =========================================================================
  int aaDim = (int)nBodies * 3; // 3 pos per body
  int aaCount = 0;
  std::vector<std::vector<float>> aaFHistory, aaXHistory;
  if (useAndersonAccel) {
    aaFHistory.resize(aaWindowSize);
    aaXHistory.resize(aaWindowSize);
    for (int i = 0; i < aaWindowSize; i++) {
      aaFHistory[i].resize(aaDim, 0.0f);
      aaXHistory[i].resize(aaDim, 0.0f);
    }
  }

  // Pack body positions into flat vector (for AA)
  auto packState = [&](std::vector<float> &state) {
    state.resize(aaDim);
    for (uint32_t i = 0; i < nBodies; i++) {
      state[i * 3 + 0] = bodies[i].position.x;
      state[i * 3 + 1] = bodies[i].position.y;
      state[i * 3 + 2] = bodies[i].position.z;
    }
  };
  auto unpackState = [&](const std::vector<float> &state) {
    for (uint32_t i = 0; i < nBodies; i++) {
      if (bodies[i].mass <= 0) continue;
      bodies[i].position.x = state[i * 3 + 0];
      bodies[i].position.y = state[i * 3 + 1];
      bodies[i].position.z = state[i * 3 + 2];
    }
  };

  // =========================================================================
  // Chebyshev state
  // =========================================================================
  float chebyOmega = 1.0f;
  std::vector<Vec3> chebyPrevPos, chebyPrevPrevPos;
  std::vector<Quat> chebyPrevRot, chebyPrevPrevRot;
  if (useChebyshev) {
    chebyPrevPos.resize(nBodies);
    chebyPrevPrevPos.resize(nBodies);
    chebyPrevRot.resize(nBodies);
    chebyPrevPrevRot.resize(nBodies);
    for (uint32_t i = 0; i < nBodies; i++) {
      chebyPrevPos[i] = bodies[i].position;
      chebyPrevPrevPos[i] = bodies[i].position;
      chebyPrevRot[i] = bodies[i].rotation;
      chebyPrevPrevRot[i] = bodies[i].rotation;
    }
  }

  // Convergence history
  convergenceHistory.clear();

  // =========================================================================
  // Rebuild per-particle adjacency (picks up any pins/attachments added
  // after addSoftBody, mirrors PhysX buildAdjacency before solve)
  // =========================================================================
  for (auto &sb : softBodies)
    sb.buildAdjacency();

  // =========================================================================
  // Per-particle contact index (prefix-sum, mirrors PhysX)
  // Avoids O(particles * contacts) scan inside the VBD loop.
  // =========================================================================
  std::vector<uint32_t> scIdxBuf(softContacts.size());
  std::vector<uint32_t> scStart(nSoftParticles + 1, 0);
  std::vector<uint32_t> scCount(nSoftParticles, 0);
  auto buildSoftContactIndex = [&]() {
    for (uint32_t i = 0; i < nSoftParticles; i++) scCount[i] = 0;
    for (uint32_t ci = 0; ci < (uint32_t)softContacts.size(); ci++)
      scCount[softContacts[ci].particleIdx]++;
    scStart[0] = 0;
    for (uint32_t i = 0; i < nSoftParticles; i++)
      scStart[i + 1] = scStart[i] + scCount[i];
    for (uint32_t i = 0; i < nSoftParticles; i++) scCount[i] = 0;
    for (uint32_t ci = 0; ci < (uint32_t)softContacts.size(); ci++) {
      uint32_t pi = softContacts[ci].particleIdx;
      scIdxBuf[scStart[pi] + scCount[pi]] = ci;
      scCount[pi]++;
    }
  };
  buildSoftContactIndex();

  // Pre-compute body-level inertial targets for Newton-style body solve
  float invDtSq = 1.0f / dt2;
  std::vector<Vec3> bodyComPred(softBodies.size());
  std::vector<Vec3> bodyThetaPred(softBodies.size());
  std::vector<Vec3> bodyAccumTheta(softBodies.size());
  for (uint32_t si = 0; si < (uint32_t)softBodies.size(); si++)
  {
    const SoftBody& sb = softBodies[si];
    Vec3 com, comPred;
    float totalMass = 0.0f;
    Vec3 angMom;
    for (uint32_t li = 0; li < sb.particleCount; li++)
    {
      uint32_t pi = sb.particleStart + li;
      if (softParticles[pi].invMass <= 0.0f) continue;
      float m = softParticles[pi].mass;
      com = com + softParticles[pi].position * m;
      comPred = comPred + softParticles[pi].predictedPosition * m;
      totalMass += m;
    }
    if (totalMass > 0.0f)
    {
      float invM = 1.0f / totalMass;
      com = com * invM;
      comPred = comPred * invM;
    }
    bodyComPred[si] = comPred;
    Mat33 bodyI;
    for (uint32_t li = 0; li < sb.particleCount; li++)
    {
      uint32_t pi = sb.particleStart + li;
      if (softParticles[pi].invMass <= 0.0f) continue;
      float m = softParticles[pi].mass;
      Vec3 r = softParticles[pi].position - com;
      float r2 = r.dot(r);
      bodyI = bodyI + (Mat33::diag(r2, r2, r2) - outer(r, r)) * m;
      angMom = angMom + r.cross(softParticles[pi].velocity) * m;
    }
    Vec3 omega = bodyI.inverse() * angMom;
    if (omega.x != omega.x) omega = Vec3();
    bodyThetaPred[si] = omega * dt;
    bodyAccumTheta[si] = Vec3();
  }

  // =========================================================================
  // Chebyshev semi-iterative state for soft particles (mirrors PhysX)
  // =========================================================================
  float softChebyOmega = 1.0f;
  std::vector<Vec3> softChebyPrevPos(nSoftParticles);
  std::vector<Vec3> softChebyPrevPrevPos(nSoftParticles);
  if (useChebyshev) {
    for (uint32_t i = 0; i < nSoftParticles; i++) {
      softChebyPrevPos[i] = softParticles[i].position;
      softChebyPrevPrevPos[i] = softParticles[i].position;
    }
  }

  const bool sequentialStatic =
      !routeContactIslandPcg && isSequentialBodyStaticIsland();
  const bool allBodyVsStatic =
      !contacts.empty() &&
      std::all_of(contacts.begin(), contacts.end(), [this](const Contact &c) {
        return isBodyVsStaticContact(c.bodyA, c.bodyB);
      });
  const bool allKinematicShell =
      contacts.empty() && !softContacts.empty() &&
      std::all_of(softContacts.begin(), softContacts.end(),
                  [this](const SoftContact &sc) {
                    return sc.rigidBodyIdx != UINT32_MAX &&
                           sc.particleIdx < softParticles.size() &&
                           softParticles[sc.particleIdx].invMass <= 0.0f;
                  });
  const int primalIterations =
      (sequentialStatic || allBodyVsStatic || allKinematicShell)
          ? std::max(iterations, kMinBodyVsStaticInnerIters)
          : iterations;

  // =========================================================================
  // Main solver loop
  // =========================================================================
  for (int it = 0; it < primalIterations; it++) {
    // Save pre-iteration state for AA
    std::vector<float> preState;
    if (useAndersonAccel)
      packState(preState);

    // Save pre-iteration state for Chebyshev
    if (useChebyshev) {
      for (uint32_t i = 0; i < nBodies; i++) {
        chebyPrevPrevPos[i] = chebyPrevPos[i];
        chebyPrevPrevRot[i] = chebyPrevRot[i];
        chebyPrevPos[i] = bodies[i].position;
        chebyPrevRot[i] = bodies[i].rotation;
      }
    }

    // ---- Primal update ----
    if (sequentialStatic) {
      sequentialBodyStaticPrimalPass(dt);
    } else {
    uint32_t numDynContactsInIsland = 0;
    for (const auto &cc : contacts) {
      if (!isBodyVsStaticContact(cc.bodyA, cc.bodyB))
        numDynContactsInIsland++;
    }
    bool reverseSweep = useTreeSweep && (it % 2 == 1);
    int nSweep = (int)sweepOrder.size();
    for (int si = 0; si < nSweep; si++) {
      int idx = reverseSweep ? (nSweep - 1 - si) : si;
      uint32_t bi = sweepOrder[idx];
      Body &body = bodies[bi];
      if (body.mass <= 0)
        continue;

      // Check if this body needs full 6x6 solve
      bool bodyNeedsFull6x6 = false;
      if (use3x3Solve) {
        for (const auto &jnt : d6Joints) {
          if (jnt.bodyA == bi || jnt.bodyB == bi) {
            // Any joint with non-trivial coupling needs 6x6
            if (jnt.linearMotion != 0 || jnt.angularMotion != 0x2A) {
              // Not purely spherical -> needs 6x6 for coupled solve
              // Actually, prismatic/revolute have lin-ang coupling
              bodyNeedsFull6x6 = true;
              break;
            }
          }
        }
      }

      Mat66 lhs = body.getMassMatrix() / dt2;
      Vec6 disp(body.position - body.inertialPosition, body.deltaWInertial());
      Vec6 rhs = lhs * disp;

      float boostFloor = kContactBoostFraction * body.mass / dt2;

      uint32_t staticContactCount = 0;
      uint32_t dynContactCount = 0;
      for (const auto &cc : contacts) {
        if (cc.bodyA == bi || cc.bodyB == bi) {
          if (isBodyVsStaticContact(cc.bodyA, cc.bodyB))
            staticContactCount++;
          else
            dynContactCount++;
        }
      }

      // ---- Contact contributions ----
      for (auto &c : contacts) {
        if (routeContactIslandPcg)
          continue;
        bool isA = (c.bodyA == bi);
        bool isB = (c.bodyB == bi);
        if (!isA && !isB)
          continue;

        const bool bStatic = isBodyVsStaticContact(c.bodyA, c.bodyB);
        if (bStatic)
          computeConstraintBodyStatic(c);
        else
          computeConstraint(c);

        const int nRows = contactPrimalRowCount(
            c.bodyA, c.bodyB, staticContactCount, dynContactCount,
            numDynContactsInIsland, allowBodyStaticFrictionIn6x6LowContact);
        for (int i = 0; i < nRows; i++) {
          Vec6 J = isA ? (i == 0 ? c.JA : (i == 1 ? c.JAt1 : c.JAt2))
                       : (i == 0 ? c.JB : (i == 1 ? c.JBt1 : c.JBt2));
          float pen = std::max(c.penalty[i], boostFloor);
          float f = std::max(c.fmin[i],
                             std::min(c.fmax[i], pen * c.C[i] + c.lambda[i]));
          rhs += J * f;
          lhs += outer(J, J * pen);
        }
      }

      // ---- D6 Joint contributions (unified) ----
      for (const auto &jnt : d6Joints) {
        if (routeContactIslandPcg &&
            (isSphericalD6RowSet(jnt) || isFixedD6RowSet(jnt) ||
             isRevoluteD6RowSet(jnt) || isPrismaticD6RowSet(jnt) ||
             isSupportedLinearXVelocityDriveD6RowSet(jnt) ||
             isSupportedSingleAxisAngularVelocityDriveD6RowSet(jnt) ||
             isSupportedSlerpVelocityDriveD6RowSet(jnt)))
          continue;
        if (useIslandPcgProbe && isFixedD6RowSet(jnt))
          continue;
        addD6Contribution(jnt, bi, bodies, dt, lhs, rhs);
      }

      // ---- Articulation contributions (pure AVBD AL constraints) ----
      for (const auto &artic : articulations) {
        for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
          addArticulationContribution(artic, ji, bi, bodies, dt, lhs, rhs);
        }
        for (int mi = 0; mi < (int)artic.mimicJoints.size(); mi++) {
          addMimicJointContribution(artic, mi, bi, bodies, dt, lhs, rhs);
        }
        for (int ti = 0; ti < (int)artic.ikTargets.size(); ti++) {
          addIKTargetContribution(artic, ti, bi, bodies, dt, lhs, rhs);
        }
      }

      // ---- Soft body attachment contributions to rigid body ----
      for (const auto &sb : softBodies) {
        for (const auto &ac : sb.attachments) {
          addAttachmentContribution_rigid(ac, bi, softParticles, bodies, dt, lhs, rhs);
        }
      }

      const float shellPenFloor = kPenScaleBodyVsStatic * body.mass / dt2;
      for (const auto &sc : softContacts) {
        if (sc.rigidBodyIdx != bi)
          continue;
        if (sc.particleIdx >= softParticles.size())
          continue;
        if (softParticles[sc.particleIdx].invMass > 0.0f)
          continue;
        addKinematicShellContactContribution_rigid(
            sc, bi, body, shellPenFloor, lhs, rhs);
      }

      // ---- Gear Joint contributions ----
      for (auto &gnt : gearJoints) {
        bool isA = (gnt.bodyA == bi);
        bool isB = (gnt.bodyB == bi);
        if (!isA && !isB)
          continue;
        if (gnt.bodyA >= (uint32_t)bodies.size() ||
            gnt.bodyB >= (uint32_t)bodies.size())
          continue;

        Body &bA = bodies[gnt.bodyA];
        Body &bB = bodies[gnt.bodyB];
        if (bA.mass <= 0.f || bB.mass <= 0.f)
          continue;

        Vec3 worldAxisA = bA.rotation.rotate(gnt.axisA);
        Vec3 worldAxisB = bB.rotation.rotate(gnt.axisB);
        Vec3 dwA = bA.deltaWInitial();
        Vec3 dwB = bB.deltaWInitial();
        float C = dwA.dot(worldAxisA) * gnt.gearRatio + dwB.dot(worldAxisB);

        float effectiveRho = std::max(gnt.rho, body.mass / dt2);
        Vec3 J_ang = isA ? (worldAxisA * gnt.gearRatio) : worldAxisB;
        float f = effectiveRho * C + gnt.lambdaGear;

        for (int r = 0; r < 3; r++)
          for (int c2 = 0; c2 < 3; c2++)
            lhs.m[3 + r][3 + c2] +=
                effectiveRho * (&J_ang.x)[r] * (&J_ang.x)[c2];
        for (int r = 0; r < 3; r++)
          rhs.v[3 + r] += f * (&J_ang.x)[r];
      }

      // ---- Solve and apply ----
      bool solve3x3ForBody = use3x3Solve && !bodyNeedsFull6x6;
      if (!solve3x3ForBody) {
        Vec6 delta = solveLDLT(lhs, rhs);
        body.position -= delta.linear();
        Quat dq(0, delta[3], delta[4], delta[5]);
        body.rotation =
            (body.rotation - dq * body.rotation * 0.5f).normalized();
      } else {
        Mat33 Alin, Aang;
        Vec3 rhsLin(rhs[0], rhs[1], rhs[2]);
        Vec3 rhsAng(rhs[3], rhs[4], rhs[5]);
        for (int r = 0; r < 3; r++)
          for (int c = 0; c < 3; c++) {
            Alin.m[r][c] = lhs.m[r][c];
            Aang.m[r][c] = lhs.m[3 + r][3 + c];
          }
        Vec3 deltaPos = Alin.inverse() * rhsLin;
        Vec3 deltaTheta = Aang.inverse() * rhsAng;
        body.position -= deltaPos;
        Quat dq(0, deltaTheta.x, deltaTheta.y, deltaTheta.z);
        body.rotation =
            (body.rotation - dq * body.rotation * 0.5f).normalized();
      }
    }
    } // !sequentialStatic

    if (useIslandPcgProbe && !solveFixedD6IslandPcgProbe(dt))
      return;

    if (routeContactIslandPcg && !solveContactIslandPcgProbe(dt))
      return;

    if (enableSequentialDynDynFriction && !routeContactIslandPcg)
      sequentialDynDynFrictionPass(dt);

    // ---- Body-level 6x6 solve for soft bodies (mirrors PhysX) ----
    for (uint32_t si = 0; si < (uint32_t)softBodies.size(); si++)
    {
      const SoftBody& sb = softBodies[si];
      Vec3 com;
      float bodyMass = 0.0f;
      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        if (softParticles[pi].invMass <= 0.0f) continue;
        com = com + softParticles[pi].position * softParticles[pi].mass;
        bodyMass += softParticles[pi].mass;
      }
      if (bodyMass <= 0.0f) continue;
      com = com * (1.0f / bodyMass);

      uint32_t bodyContactCount = 0;
      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        bodyContactCount += scStart[pi + 1] - scStart[pi];
      }
      if (bodyContactCount == 0) continue;

      Mat33 bodyInertia;
      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        if (softParticles[pi].invMass <= 0.0f) continue;
        Vec3 r = softParticles[pi].position - com;
        float r2 = r.dot(r);
        bodyInertia = bodyInertia +
          (Mat33::diag(r2, r2, r2) - outer(r, r)) * softParticles[pi].mass;
      }

      float bodyMassDtSq = bodyMass * invDtSq;
      Mat33 A_ll = Mat33::diag(bodyMassDtSq, bodyMassDtSq, bodyMassDtSq);
      Mat33 A_la, A_al;
      Mat33 A_aa = bodyInertia * invDtSq;
      float reg = 1e-4f * bodyMassDtSq;
      A_aa = A_aa + Mat33::diag(reg, reg, reg);

      Vec3 g_l = (com - bodyComPred[si]) * bodyMassDtSq;
      Vec3 g_a = (bodyInertia * invDtSq) * (bodyAccumTheta[si] - bodyThetaPred[si]);

      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        Vec3 r = softParticles[pi].position - com;
        for (uint32_t k = scStart[pi]; k < scStart[pi + 1]; k++)
        {
          const SoftContact& sc = softContacts[scIdxBuf[k]];
          Vec3 n = sc.normal;
          float violation;
          if (sc.rigidBodyIdx == UINT32_MAX)
            violation = softParticles[pi].position.dot(n);
          else
            violation =
                (softParticles[pi].position - sc.surfacePoint).dot(n) - sc.depth;

          float pen = sc.k;
          Vec3 rCrossN = r.cross(n);
          A_ll = A_ll + outer(n, n) * pen;
          A_la = A_la + outer(n, rCrossN) * pen;
          A_al = A_al + outer(rCrossN, n) * pen;
          A_aa = A_aa + outer(rCrossN, rCrossN) * pen;

          float f = std::min(0.0f, pen * violation);
          if (f < 0.0f)
          {
            g_l = g_l + n * f;
            g_a = g_a + rCrossN * f;
          }
        }
      }

      Mat33 A_ll_inv = A_ll.inverse();
      Mat33 S = A_aa - A_al.mul(A_ll_inv).mul(A_la);
      Vec3 deltaTheta = S.inverse() * (g_a - A_al.mul(A_ll_inv) * g_l);
      Vec3 deltaPos = A_ll_inv * (g_l - A_la * deltaTheta);

      if (deltaPos.x != deltaPos.x || deltaTheta.x != deltaTheta.x) continue;

      float thetaMag = deltaTheta.length();
      if (thetaMag > 0.5f) deltaTheta = deltaTheta * (0.5f / thetaMag);

      for (uint32_t li = 0; li < sb.particleCount; li++)
      {
        uint32_t pi = sb.particleStart + li;
        if (softParticles[pi].invMass <= 0.0f) continue;
        Vec3 r = softParticles[pi].position - com;
        softParticles[pi].position = softParticles[pi].position - deltaPos - deltaTheta.cross(r);
      }
      bodyAccumTheta[si] = bodyAccumTheta[si] - deltaTheta;
    }

    // ---- AVBD Soft particle primal update (outer/inner loop, mirrors PhysX) ----
    // Snapshot positions as proximal anchor for AVBD elastic term
    for (uint32_t i = 0; i < nSoftParticles; i++)
      softParticles[i].outerPosition = softParticles[i].position;

    // Reset Chebyshev state for each outer iteration when innerIterations > 1
    float softAdaptiveRho = chebyshevSpectralRadius;
    if (innerIterations > 1) {
      softChebyOmega = 1.0f;
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        softChebyPrevPos[i] = softParticles[i].position;
        softChebyPrevPrevPos[i] = softParticles[i].position;
      }
    }

    float softPrevMaxDxSq = 0.0f;

    for (int innerIt = 0; innerIt < innerIterations; innerIt++) {
    float softMaxDxSq = 0.0f;
    for (uint32_t si = 0; si < (uint32_t)softBodies.size(); si++) {
      const SoftBody &sb = softBodies[si];
      for (uint32_t li = 0; li < sb.particleCount; li++) {
        uint32_t spi = sb.particleStart + li;
        SoftParticle &sp = softParticles[spi];
        if (sp.invMass <= 0.0f) continue;

        float mOverDt2 = sp.mass / dt2;
        Vec3 f3 = (sp.predictedPosition - sp.position) * mOverDt2;
        Mat33 H3 = Mat33::diag(mOverDt2, mOverDt2, mOverDt2);

        const SoftBody::ParticleAdjacency& adj = sb.adjacency[li];

        // StVK triangle contributions (adjacency lookup)
        for (const auto &ref : adj.triRefs) {
          Vec3 ft; Mat33 Ht;
          evaluateStVKForceHessian(sb.triElements[ref.index], (int)ref.vOrder,
                                   sb.mu, sb.lambda, softParticles, ft, Ht);
          f3 = f3 + ft; H3 = H3 + Ht;
        }
        // Neo-Hookean tet contributions (adjacency lookup)
        for (const auto &ref : adj.tetRefs) {
          Vec3 ft; Mat33 Ht;
          evaluateNeoHookeanForceHessian(sb.tetElements[ref.index], (int)ref.vOrder,
                                         sb.mu, sb.lambda, softParticles, ft, Ht);
          f3 = f3 + ft; H3 = H3 + Ht;
        }
        // Bending contributions (adjacency lookup)
        for (const auto &ref : adj.bendRefs) {
          Vec3 fb; Mat33 Hb;
          evaluateBendingForceHessian(sb.bendElements[ref.index], (int)ref.vOrder,
                                      sb.bendingStiffness, softParticles, fb, Hb);
          f3 = f3 + fb; H3 = H3 + Hb;
        }
        // Attachment (adjacency lookup)
        for (uint32_t ai : adj.attachmentIndices) {
          Vec3 fa; Mat33 Ha;
          evaluateAttachmentForceHessian_particle(sb.attachments[ai], softParticles, bodies, fa, Ha);
          f3 = f3 + fa; H3 = H3 + Ha;
        }
        // Kinematic pin (adjacency lookup)
        for (uint32_t pi : adj.pinIndices) {
          Vec3 fp; Mat33 Hp;
          evaluatePinForceHessian(sb.pins[pi], softParticles, fp, Hp);
          f3 = f3 + fp; H3 = H3 + Hp;
        }

        // Soft contacts (indexed lookup, mirrors PhysX)
        for (uint32_t k = scStart[spi]; k < scStart[spi + 1]; k++) {
          Vec3 fc; Mat33 Hc;
          evaluateContactForceHessian(softContacts[scIdxBuf[k]], softParticles, fc, Hc);
          f3 = f3 + fc; H3 = H3 + Hc;
        }

        // Stiffness-proportional Rayleigh damping (Newton VBD style):
        // Per-axis damping proportional to elastic stiffness, clamped so no
        // axis gets less damping than mass-proportional (baseline stability).
        if (sp.damping > 0.0f) {
          float dampCoeff = sp.damping * sp.mass * invDt;
          Mat33 H_elastic = H3 - Mat33::diag(mOverDt2, mOverDt2, mOverDt2);
          float he_xx = fmaxf(H_elastic.m[0][0], 0.0f);
          float he_yy = fmaxf(H_elastic.m[1][1], 0.0f);
          float he_zz = fmaxf(H_elastic.m[2][2], 0.0f);
          float trHe = he_xx + he_yy + he_zz;
          float dx, dy, dz;
          if (trHe > 1e-10f) {
            float s = dampCoeff * 3.0f / trHe;
            dx = fmaxf(he_xx * s, dampCoeff);
            dy = fmaxf(he_yy * s, dampCoeff);
            dz = fmaxf(he_zz * s, dampCoeff);
          } else {
            dx = dy = dz = dampCoeff;
          }
          Mat33 H_damp = Mat33::diag(dx, dy, dz);
          f3 = f3 - H_damp * (sp.position - sp.initialPosition);
          H3 = H3 + H_damp;
        }

        // AVBD elastic proximal term: pulls toward outer-iteration anchor
        // to ensure convergence independent of update order (Jacobi-safe)
        if (sp.elasticK > 0.0f) {
          H3 = H3 + Mat33::diag(sp.elasticK, sp.elasticK, sp.elasticK);
          f3 = f3 + (sp.outerPosition - sp.position) * sp.elasticK;
        }

        // Solve 3x3: displacement = H^-1 * f (with clamping, mirrors PhysX)
        Vec3 displacement = H3.inverse() * f3;
        float dxLenSq = displacement.dot(displacement);
        const float maxDx = 1.0f;
        if (dxLenSq > maxDx * maxDx)
          displacement = displacement * (maxDx / sqrtf(dxLenSq));
        if (displacement.x == displacement.x) { // NaN guard
          sp.position = sp.position + displacement;
          if (dxLenSq > softMaxDxSq) softMaxDxSq = dxLenSq;
        }
      }
    }

    // Early termination for soft particles (mirrors PhysX)
    bool softConverged = (softMaxDxSq < 1e-12f);
    if (softConverged) break;

    // Adaptive spectral-radius estimation (mirrors PhysX).
    // Measure GS convergence ratio from iterations 0-1, then use
    // min(measured, user-provided) as the Chebyshev parameter.
    if (innerIt == 0) {
      softPrevMaxDxSq = softMaxDxSq;
    } else if (innerIt == 1 && useChebyshev) {
      if (softPrevMaxDxSq > 1e-20f) {
        float measuredRho = sqrtf(softMaxDxSq / softPrevMaxDxSq);
        softAdaptiveRho = std::min(measuredRho, chebyshevSpectralRadius);
        softAdaptiveRho = std::min(softAdaptiveRho, 0.95f);
      }
      softPrevMaxDxSq = softMaxDxSq;
    }

    // Chebyshev semi-iterative for soft particles (matches PhysX)
    // Use innerIt for iteration index when innerIterations > 1
    int chebyIt = (innerIterations > 1) ? innerIt : it;
    if (useChebyshev && chebyIt >= 2) {
      float rhoSq = softAdaptiveRho * softAdaptiveRho;
      // Use uniform recurrence
      softChebyOmega = 4.0f / (4.0f - rhoSq * softChebyOmega);
      softChebyOmega = std::max(1.0f, std::min(softChebyOmega, 2.0f));

      // Divergence guard: if displacement grew, disable Chebyshev
      if (softPrevMaxDxSq > 1e-20f && softMaxDxSq > softPrevMaxDxSq * 1.1f) {
        softChebyOmega = 1.0f;
        softAdaptiveRho = 0.0f;
      }

      if (softChebyOmega > 1.0f) {
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        if (softParticles[i].invMass <= 0.0f) continue;
        // Skip Chebyshev for particles with active contacts
        // (over-relaxation can push them through surfaces)
        if (scStart[i + 1] > scStart[i]) continue;
        // Also skip for particles with pins or attachments
        bool hasConstraint = false;
        for (const auto &sb : softBodies) {
          uint32_t li = i - sb.particleStart;
          if (li < sb.particleCount) {
            const auto &adj = sb.adjacency[li];
            if (!adj.pinIndices.empty() || !adj.attachmentIndices.empty())
              hasConstraint = true;
            break;
          }
        }
        if (hasConstraint) continue;
        softParticles[i].position = softChebyPrevPrevPos[i] +
            (softParticles[i].position - softChebyPrevPrevPos[i]) * softChebyOmega;
      }
      }
      softPrevMaxDxSq = softMaxDxSq;
    }
    if (useChebyshev) {
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        softChebyPrevPrevPos[i] = softChebyPrevPos[i];
        softChebyPrevPos[i] = softParticles[i].position;
      }
    }
    } // end innerIterations loop

    // Collision projection (Jolt-style hard constraint, mirrors PhysX)
    for (uint32_t ci = 0; ci < (uint32_t)softContacts.size(); ci++) {
      SoftParticle &sp = softParticles[softContacts[ci].particleIdx];
      if (sp.invMass <= 0.0f) continue;
      const SoftContact &sc = softContacts[ci];
      Vec3 n = sc.normal;
      float projPen;
      if (sc.rigidBodyIdx == UINT32_MAX)
        projPen = -(sp.position.dot(n));          // ground plane
      else
        projPen = -(sp.position - sc.surfacePoint).dot(n);  // body surface
      if (projPen > 0.0f) {
        if (sc.rigidBodyIdx != UINT32_MAX && sc.rigidBodyIdx < softBodies.size())
          projPen = std::min(projPen, 0.05f);
        sp.position = sp.position + n * projPen;
      }
    }

    // ---- Dual update ----
    // Contact dual (body-vs-static: computeConstraintBodyStatic + all 3 rows)
    for (auto &c : contacts) {
      if (routeContactIslandPcg)
        continue;
      if (isBodyVsStaticContact(c.bodyA, c.bodyB))
        computeConstraintBodyStatic(c);
      else
        computeConstraint(c);
      for (int i = 0; i < 3; i++) {
        float oldLambda = c.lambda[i];
        float rawLambda = c.penalty[i] * c.C[i] + oldLambda;
        c.lambda[i] = std::max(c.fmin[i], std::min(c.fmax[i], rawLambda));
        if (c.lambda[i] < c.fmax[i] && c.lambda[i] > c.fmin[i])
          c.penalty[i] =
              std::min(c.penalty[i] + beta * fabsf(c.C[i]), PENALTY_MAX);
      }
    }

    // Joint dual (unified D6)
    {
      const float lambdaDecay = 0.99f;
      for (auto &jnt : d6Joints) {
        if (useIslandPcgProbe && isFixedD6RowSet(jnt))
          updateFixedD6IslandPcgProbeDual(jnt, bodies, dt);
        else if (routeContactIslandPcg && isRevoluteD6RowSet(jnt))
          updateRevoluteD6IslandPcgDual(jnt, bodies, dt, lambdaDecay);
        else if (routeContactIslandPcg && isPrismaticD6RowSet(jnt))
          updatePrismaticD6IslandPcgDual(jnt, bodies, dt, lambdaDecay);
        else if (routeContactIslandPcg &&
                 isSupportedLinearXVelocityDriveD6RowSet(jnt)) {
          updateLinearXVelocityDriveIslandPcgDual(jnt, bodies, dt,
                                                  lambdaDecay);
          linearDriveIslandLastStats.maxAbsDual =
              std::max(linearDriveIslandLastStats.maxAbsDual,
                       std::fabs(jnt.lambdaDriveLinear.x));
        }
        else if (routeContactIslandPcg &&
                 isSupportedSingleAxisAngularVelocityDriveD6RowSet(jnt)) {
          const int angularAxisIndex =
              getSupportedSingleAxisAngularVelocityDriveIndex(jnt);
          updateSingleAxisAngularVelocityDriveIslandPcgDual(
              jnt, bodies, dt, lambdaDecay, angularAxisIndex);
          angularDriveIslandLastStats.maxAbsDual =
              std::max(angularDriveIslandLastStats.maxAbsDual,
                       std::fabs(
                           (&jnt.lambdaDriveAngular.x)[angularAxisIndex]));
        }
        else if (routeContactIslandPcg &&
                 isSupportedSlerpVelocityDriveD6RowSet(jnt)) {
          updateSlerpVelocityDriveIslandPcgDual(jnt, bodies, dt,
                                                lambdaDecay);
          angularDriveIslandLastStats.maxAbsDual =
              std::max(
                  angularDriveIslandLastStats.maxAbsDual,
                  std::max(std::fabs(jnt.lambdaDriveAngular.x),
                           std::max(std::fabs(jnt.lambdaDriveAngular.y),
                                    std::fabs(jnt.lambdaDriveAngular.z))));
        }
        else
          updateD6Dual(jnt, bodies, dt, lambdaDecay);
      }
    }

    // Gear joint dual (inside iteration loop, matching PhysX)
    {
      const float lambdaDecay = 0.99f;
      for (auto &gnt : gearJoints) {
        if (gnt.bodyA >= (uint32_t)bodies.size() ||
            gnt.bodyB >= (uint32_t)bodies.size())
          continue;
        Body &bA = bodies[gnt.bodyA];
        Body &bB = bodies[gnt.bodyB];
        if (bA.mass <= 0.f || bB.mass <= 0.f)
          continue;

        float mA = bA.mass, mB = bB.mass;
        float mEff2 = (mA > 0.f && mB > 0.f) ? std::min(mA, mB) : std::max(mA, mB);
        float Mh2 = mEff2 / dt2;
        float admm_step = gnt.rho * gnt.rho / (gnt.rho + Mh2);
        float rhoDual = std::min(Mh2, admm_step);

        Vec3 worldAxisA = bA.rotation.rotate(gnt.axisA);
        Vec3 worldAxisB = bB.rotation.rotate(gnt.axisB);
        Vec3 dwA = bA.deltaWInitial();
        Vec3 dwB = bB.deltaWInitial();
        float C = dwA.dot(worldAxisA) * gnt.gearRatio + dwB.dot(worldAxisB);

        gnt.lambdaGear = gnt.lambdaGear * lambdaDecay + rhoDual * C;
      }
    }

    // Articulation dual
    {
      const float lambdaDecay = 0.99f;
      for (auto &artic : articulations) {
        for (int ji = 0; ji < (int)artic.joints.size(); ji++) {
          updateArticulationDual(artic, ji, bodies, dt, lambdaDecay);
        }
        for (int mi = 0; mi < (int)artic.mimicJoints.size(); mi++) {
          updateMimicDual(artic, mi, bodies, dt, lambdaDecay);
        }
        for (int ti = 0; ti < (int)artic.ikTargets.size(); ti++) {
          updateIKTargetDual(artic, ti, bodies, dt, lambdaDecay);
        }
      }
    }

    // Soft body AVBD dual update (penalty growth + elastic proximal)
    {
      for (auto &sb : softBodies) {
        for (auto &ac : sb.attachments)
          updateAttachmentDual(ac, softParticles, bodies, beta);
        for (auto &kp : sb.pins)
          updatePinDual(kp, softParticles, beta);
      }
      for (auto &sc : softContacts) {
      if (sc.rigidBodyIdx != UINT32_MAX &&
          sc.particleIdx < softParticles.size() &&
          softParticles[sc.particleIdx].invMass <= 0.0f &&
          sc.rigidBodyIdx < bodies.size()) {
        Body &shellBody = bodies[sc.rigidBodyIdx];
        const float Cn =
            kinematicShellContactViolation(sc, shellBody);
        const float rawLambdaN = sc.k * Cn + sc.lambda;
        sc.lambda = std::min(0.0f, rawLambdaN);
        if (sc.lambda < 0.0f)
          sc.k = std::min(sc.k + beta * fabsf(Cn), sc.ke);

        float Ctangent[2];
        computeKinematicShellConstraint(sc, shellBody, alpha, Ctangent);
        const float frictionBound = fabsf(sc.lambda) * sc.friction;
        for (int ti = 0; ti < 2; ++ti) {
          const float fmin = -frictionBound;
          const float fmax = frictionBound;
          const float oldLt = sc.lambdaTangent[ti];
          const float rawLt =
              sc.penTangent[ti] * Ctangent[ti] + oldLt;
          sc.lambdaTangent[ti] =
              std::max(fmin, std::min(fmax, rawLt));
          if (sc.lambdaTangent[ti] < fmax && sc.lambdaTangent[ti] > fmin)
            sc.penTangent[ti] = std::min(sc.penTangent[ti] +
                                             beta * fabsf(Ctangent[ti]),
                                         PENALTY_MAX);
        }
        continue;
      }
      updateSoftContactDual(sc, softParticles, beta);
    }

      // AVBD elastic proximal dual update: increase proximal weight
      // proportional to displacement from the outer-iteration anchor
      for (uint32_t i = 0; i < nSoftParticles; i++) {
        SoftParticle &sp = softParticles[i];
        if (sp.invMass <= 0.0f) continue;
        float disp = (sp.position - sp.outerPosition).length();
        sp.elasticK = std::min(sp.elasticK + beta * disp, sp.elasticKMax);
      }
    }

    // ===================================================================
    // Anderson Acceleration (Type I, safeguarded)
    // ===================================================================
    if (useAndersonAccel && it >= 0) {
      std::vector<float> postState;
      packState(postState);

      // Compute residual f_k = g(x_k) - x_k
      std::vector<float> fk(aaDim);
      for (int i = 0; i < aaDim; i++)
        fk[i] = postState[i] - preState[i];

      // Store in circular buffer
      int slot = aaCount % aaWindowSize;
      aaXHistory[slot] = preState;
      aaFHistory[slot] = fk;
      aaCount++;

      int mk = std::min(aaCount - 1, aaWindowSize); // number of differences
      if (mk >= 1) {
        // Build ΔF matrix columns: ΔF_j = f_k - f_{k-j}
        // We have: slot = most recent, (slot-1+ws)%ws = one before, etc.
        std::vector<std::vector<float>> deltaF(mk, std::vector<float>(aaDim));
        std::vector<std::vector<float>> deltaX(mk, std::vector<float>(aaDim));
        for (int j = 0; j < mk; j++) {
          int prevSlot = (slot - 1 - j + aaWindowSize * 2) % aaWindowSize;
          for (int i = 0; i < aaDim; i++) {
            deltaF[j][i] = fk[i] - aaFHistory[prevSlot][i];
            deltaX[j][i] = preState[i] - aaXHistory[prevSlot][i];
          }
        }

        // Solve normal equations: (ΔF^T ΔF) θ = ΔF^T f_k
        std::vector<float> FTF(mk * mk, 0.0f);
        std::vector<float> FTf(mk, 0.0f);
        for (int i = 0; i < mk; i++) {
          for (int j = 0; j <= i; j++) {
            float dot = 0;
            for (int d = 0; d < aaDim; d++)
              dot += deltaF[i][d] * deltaF[j][d];
            FTF[i * mk + j] = dot;
            FTF[j * mk + i] = dot;
          }
          float dot = 0;
          for (int d = 0; d < aaDim; d++)
            dot += deltaF[i][d] * fk[d];
          FTf[i] = dot;
        }

        // Tikhonov regularization
        float maxDiag = 0;
        for (int i = 0; i < mk; i++)
          maxDiag = std::max(maxDiag, FTF[i * mk + i]);
        float reg = 1e-8f * std::max(maxDiag, 1.0f);
        for (int i = 0; i < mk; i++)
          FTF[i * mk + i] += reg;

        // Gaussian elimination (mk ≤ 3, tiny system)
        std::vector<float> theta(mk, 0.0f);
        for (int i = 0; i < mk; i++) {
          float pivot = FTF[i * mk + i];
          if (std::fabs(pivot) < 1e-15f) continue;
          for (int j = i + 1; j < mk; j++) {
            float factor = FTF[j * mk + i] / pivot;
            for (int k = i + 1; k < mk; k++)
              FTF[j * mk + k] -= factor * FTF[i * mk + k];
            FTf[j] -= factor * FTf[i];
          }
        }
        for (int i = mk - 1; i >= 0; i--) {
          float sum = FTf[i];
          for (int j = i + 1; j < mk; j++)
            sum -= FTF[i * mk + j] * theta[j];
          float pivot = FTF[i * mk + i];
          theta[i] = (std::fabs(pivot) > 1e-15f) ? (sum / pivot) : 0.0f;
        }

        // Compute AA iterate: x_{k+1} = g(x_k) - ΔG * θ
        //   where ΔG_j = (x_k + f_k) - (x_{k-j} + f_{k-j}) = ΔX_j + ΔF_j
        std::vector<float> aaState(aaDim);
        for (int i = 0; i < aaDim; i++) {
          float correction = 0;
          for (int j = 0; j < mk; j++)
            correction += theta[j] * (deltaX[j][i] + deltaF[j][i]);
          aaState[i] = postState[i] - correction;
        }

        // Safeguard: measure actual constraint violation before/after AA
        float violBefore = 0;
        for (auto &artic : articulations)
          violBefore = std::max(violBefore, artic.computeMaxPositionViolation(bodies));

        // Tentatively apply AA state
        unpackState(aaState);

        float violAfter = 0;
        for (auto &artic : articulations)
          violAfter = std::max(violAfter, artic.computeMaxPositionViolation(bodies));

        // Reject if AA increased violation
        if (violAfter > violBefore) {
          unpackState(postState);
        }
      }
    }

    // ===================================================================
    // Chebyshev semi-iterative position relaxation
    //
    // x_{k+1}^cheb = x_{k-1} + omega_k * (x_{k+1}^GS - x_{k-1})
    // omega follows the Chebyshev recurrence for spectral radius rho.
    // ===================================================================
    if (useChebyshev && it >= 2) {
      // Chebyshev omega recurrence
      float rhoSq = chebyshevSpectralRadius * chebyshevSpectralRadius;
      if (it == 2) {
        chebyOmega = 2.0f / (2.0f - rhoSq);
      } else {
        chebyOmega = 1.0f / (1.0f - rhoSq * chebyOmega / 4.0f);
      }
      chebyOmega = std::max(1.0f, std::min(chebyOmega, 2.0f)); // safety clamp

      for (uint32_t i = 0; i < nBodies; i++) {
        if (bodies[i].mass <= 0) continue;
        // Relaxed position: x_new = x_{k-1} + omega * (x_GS - x_{k-1})
        bodies[i].position = chebyPrevPrevPos[i] +
            (bodies[i].position - chebyPrevPrevPos[i]) * chebyOmega;
        // For rotation: use SLERP-like interpolation via quaternion blend
        // Approximate: q_new ≈ normalize(q_{k-1} + omega * (q_GS - q_{k-1}))
        Quat qPrev = chebyPrevPrevRot[i];
        Quat qCur = bodies[i].rotation;
        float dotQ = qPrev.w * qCur.w + qPrev.x * qCur.x +
                     qPrev.y * qCur.y + qPrev.z * qCur.z;
        if (dotQ < 0) qCur = qCur * (-1.0f);
        Quat qBlend;
        qBlend.w = qPrev.w + chebyOmega * (qCur.w - qPrev.w);
        qBlend.x = qPrev.x + chebyOmega * (qCur.x - qPrev.x);
        qBlend.y = qPrev.y + chebyOmega * (qCur.y - qPrev.y);
        qBlend.z = qPrev.z + chebyOmega * (qCur.z - qPrev.z);
        bodies[i].rotation = qBlend.normalized();
      }
    }

    // ===================================================================
    // Convergence tracking
    // ===================================================================
    if (!articulations.empty()) {
      float maxViol = 0;
      for (const auto &artic : articulations)
        maxViol = std::max(maxViol, artic.computeMaxPositionViolation(bodies));
      convergenceHistory.push_back(maxViol);
    }
  } // end iteration loop

  applyBodyStaticDepenetrationSweeps(4);
  if (!routeContactIslandPcg)
    applyLowIslandDynDynFrictionSweeps(2);

  // Update velocities
  for (auto &body : bodies) {
    if (body.mass <= 0)
      continue;
    body.prevLinearVelocity = body.linearVelocity;
    body.linearVelocity = (body.position - body.initialPosition) * invDt;
    Quat dq = body.rotation * body.initialRotation.conjugate();
    if (dq.w < 0)
      dq = -dq;
    body.angularVelocity = Vec3(dq.x, dq.y, dq.z) * (2.0f * invDt);

    // Per-body damping
    if (body.linearDamping > 0.0f) {
      float decay = std::max(0.0f, 1.0f - body.linearDamping * dt);
      body.linearVelocity = body.linearVelocity * decay;
    }
    if (body.angularDamping > 0.0f) {
      float decay = std::max(0.0f, 1.0f - body.angularDamping * dt);
      body.angularVelocity = body.angularVelocity * decay;
    }

    // Velocity clamping
    float linSpeed = body.linearVelocity.length();
    if (linSpeed > body.maxLinearVelocity) {
      body.linearVelocity = body.linearVelocity * (body.maxLinearVelocity / linSpeed);
    }
    float angSpeed = body.angularVelocity.length();
    if (angSpeed > body.maxAngularVelocity) {
      body.angularVelocity = body.angularVelocity * (body.maxAngularVelocity / angSpeed);
    }
  }

  projectD6BodyStaticLockedLinearVelocities(bodies, d6Joints);

  if (revoluteMotorVelocityOwner.motorJointIndex < d6Joints.size()) {
    const D6Joint &motor =
        d6Joints[revoluteMotorVelocityOwner.motorJointIndex];
    if (revoluteMotorVelocityOwner.kind ==
        RevoluteMotorVelocityOwnerKind::Isolated) {
      projectIsolatedRevoluteMotorVelocity(
          bodies, motor, dt, revoluteMotorVelocityOwner);
    } else if (revoluteMotorVelocityOwner.kind ==
                   RevoluteMotorVelocityOwnerKind::CenteredGear &&
               gearJoints.size() == 1) {
      projectCenteredRevoluteMotorGearVelocity(
          bodies, motor, gearJoints[0], dt);
    }
  }

  // Update soft particle velocities
  for (auto &sp : softParticles) {
    if (sp.invMass <= 0.0f) continue;
    sp.prevVelocity = sp.velocity;
    sp.velocity = (sp.position - sp.initialPosition) * invDt;
    if (sp.damping > 0.0f) {
      float decay = std::max(0.0f, 1.0f - sp.damping * dt);
      sp.velocity = sp.velocity * decay;
    }
  }

}

} // namespace AvbdRef
