// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/joint/DyAvbdJointCoupledSystem.h"
#include "avbd/solver/joint/DyAvbdJointSupportPolicies.h"

namespace physx {
namespace Dy {

double dotVectors(const physx::PxArray<AvbdVec6> &a,
                  const physx::PxArray<AvbdVec6> &b) {
  double result = 0.0;
  for (physx::PxU32 i = 0; i < a.size(); ++i)
    result += static_cast<double>(a[i].linear.x) * b[i].linear.x +
              static_cast<double>(a[i].linear.y) * b[i].linear.y +
              static_cast<double>(a[i].linear.z) * b[i].linear.z +
              static_cast<double>(a[i].angular.x) * b[i].angular.x +
              static_cast<double>(a[i].angular.y) * b[i].angular.y +
              static_cast<double>(a[i].angular.z) * b[i].angular.z;
  return result;
}

void addCoupledRow(const CoupledIslandRow &row,
                   physx::PxArray<CoupledIslandRow> &rows,
                   physx::PxArray<AvbdVec6> &gradient,
                   physx::PxArray<AvbdBlock6x6> &preconditioner) {
  rows.pushBack(row);
  if (row.bodyA != PX_MAX_U32) {
    addScaled(gradient[row.bodyA], row.jacobianA, row.force);
    preconditioner[row.bodyA].addConstraintContribution(
        row.jacobianA.linear, row.jacobianA.angular, row.penalty);
  }
  if (row.bodyB != PX_MAX_U32) {
    addScaled(gradient[row.bodyB], row.jacobianB, row.force);
    preconditioner[row.bodyB].addConstraintContribution(
        row.jacobianB.linear, row.jacobianB.angular, row.penalty);
  }
}

void applyCoupledOperator(
    const physx::PxArray<AvbdBlock6x6> &inertialBlocks,
    const physx::PxArray<CoupledIslandRow> &rows,
    const physx::PxArray<AvbdVec6> &input,
    physx::PxArray<AvbdVec6> &output) {
  output.resize(input.size());
  for (physx::PxU32 i = 0; i < input.size(); ++i)
    output[i] = multiplyBlock(inertialBlocks[i], input[i]);
  for (physx::PxU32 i = 0; i < rows.size(); ++i) {
    const CoupledIslandRow &row = rows[i];
    physx::PxReal projection = 0.0f;
    if (row.bodyA != PX_MAX_U32)
      projection += row.jacobianA.dot(input[row.bodyA]);
    if (row.bodyB != PX_MAX_U32)
      projection += row.jacobianB.dot(input[row.bodyB]);
    const physx::PxReal scale = row.penalty * projection;
    if (row.bodyA != PX_MAX_U32)
      addScaled(output[row.bodyA], row.jacobianA, scale);
    if (row.bodyB != PX_MAX_U32)
      addScaled(output[row.bodyB], row.jacobianB, scale);
  }
}

bool addBodyVsStaticContactNormalRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner,
    bool allowFriction) {
  if (!allowFriction) {
    if (!areFrictionlessBodyVsStaticContactsSupported(
            contacts, numContacts, numBodies))
      return false;
  } else {
    if (!contacts || numContacts == 0u)
      return false;
    for (physx::PxU32 i = 0; i < numContacts; ++i) {
      if (!isBodyVsStaticContact(
              contacts[i].header.bodyIndexA,
              contacts[i].header.bodyIndexB, numBodies) ||
          hasDeformableStaticAnchor(contacts[i]) ||
          hasKinematicShellAnchor(contacts[i]))
        return false;
    }
  }

  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const bool dynamicA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicA ? contact.header.bodyIndexA : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies || bodies[bodyIndex].invMass <= 0.0f)
      return false;

    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxVec3 localPoint =
        dynamicA ? contact.contactPointA : contact.contactPointB;
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 worldDynamic = body.position + r;
    const physx::PxVec3 worldStatic =
        dynamicA ? contact.contactPointB : contact.contactPointA;
    physx::PxReal violation =
        (dynamicA ? worldDynamic - worldStatic : worldStatic - worldDynamic)
            .dot(contact.contactNormal) +
        contact.penetrationDepth;
    violation -= config.avbdAlpha * contact.C0;
    const physx::PxReal massInvDt2 =
        (1.0f / body.invMass) * invDt2;
    const physx::PxReal penalty = physx::PxMax(
        contact.header.penalty,
        AvbdConstants::AVBD_CONTACT_BOOST_FRACTION * massInvDt2);
    const physx::PxReal force =
        physx::PxMin(0.0f, penalty * violation + contact.header.lambda);
    const physx::PxReal sign = dynamicA ? 1.0f : -1.0f;
    const physx::PxVec3 contactAxis = contact.contactNormal * sign;
    CoupledIslandRow row;
    row.bodyA = bodyIndex;
    row.bodyB = PX_MAX_U32;
    row.jacobianA =
        AvbdVec6(contactAxis, r.cross(contact.contactNormal) * sign);
    row.jacobianB = AvbdVec6();
    row.penalty = penalty;
    row.force = force;
    addCoupledRow(row, rows, gradient, preconditioner);
  }
  return true;
}

bool addFrictionlessBodyVsStaticContactRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdSolverConfig &config, physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner) {
  return addBodyVsStaticContactNormalRows(
      bodies, numBodies, contacts, numContacts, config, invDt2,
      rows, gradient, preconditioner, false);
}

bool addStrictFrictionalBodyVsStaticContactPositionRows(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, const AvbdSolverConfig &config,
    physx::PxReal invDt2,
    physx::PxArray<CoupledIslandRow> &rows,
    physx::PxArray<AvbdVec6> &gradient,
    physx::PxArray<AvbdBlock6x6> &preconditioner) {
  if (!areStrictFrictionalTorqueFreeBodyVsStaticContactsSupported(
          bodies, numBodies, contacts, numContacts, gravity))
    return false;

  physx::PxU32 contactsPerBody[2] = {0u, 0u};
  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    if (bodyIndex >= numBodies)
      return false;
    contactsPerBody[bodyIndex]++;
  }
  const physx::PxReal mass0 = 1.0f / bodies[0].invMass;
  const physx::PxReal mass1 = 1.0f / bodies[1].invMass;
  const bool unequalEndpointMasses =
      physx::PxAbs(mass0 - mass1) >
      1e-6f * physx::PxMax(mass0, mass1);

  for (physx::PxU32 i = 0; i < numContacts; ++i) {
    const AvbdContactConstraint &contact = contacts[i];
    const bool dynamicA = contact.header.bodyIndexA < numBodies;
    const physx::PxU32 bodyIndex =
        dynamicA ? contact.header.bodyIndexA
                 : contact.header.bodyIndexB;
    AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxVec3 localPoint =
        dynamicA ? contact.contactPointA
                 : contact.contactPointB;
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 previousR =
        body.prevRotation.rotate(localPoint);
    const physx::PxVec3 displacement =
        (body.position + r) -
        (body.prevPosition + previousR);
    const physx::PxReal sign = dynamicA ? 1.0f : -1.0f;
    const physx::PxReal tangentViolation0 =
        sign * displacement.dot(contact.tangent0);
    const physx::PxReal tangentViolation1 =
        sign * displacement.dot(contact.tangent1);

    const physx::PxReal mass = 1.0f / body.invMass;
    const physx::PxReal contactBoostFloor =
        AvbdConstants::AVBD_CONTACT_BOOST_FRACTION *
        mass * invDt2;
    physx::PxVec3 dynamicNormal =
        contact.contactNormal * sign;
    if (dynamicNormal.normalize() <= 1e-6f)
      return false;
    const physx::PxReal weightShare =
        mass * physx::PxAbs(gravity.dot(dynamicNormal)) /
        physx::PxReal(contactsPerBody[bodyIndex]);
    physx::PxReal normalCapacity = weightShare;
    if (!unequalEndpointMasses) {
      const physx::PxVec3 worldDynamic = body.position + r;
      const physx::PxVec3 worldStatic =
          dynamicA ? contact.contactPointB : contact.contactPointA;
      physx::PxReal normalViolation =
          (dynamicA ? worldDynamic - worldStatic
                    : worldStatic - worldDynamic)
              .dot(contact.contactNormal) +
          contact.penetrationDepth;
      normalViolation -= config.avbdAlpha * contact.C0;
      const physx::PxReal normalPenalty =
          physx::PxMax(contact.header.penalty, contactBoostFloor);
      const physx::PxReal normalForce =
          physx::PxMin(0.0f, normalPenalty * normalViolation +
                                contact.header.lambda);
      const physx::PxReal priorNormalForce =
          contact.header.lambda < 0.0f ? -contact.header.lambda : 0.0f;
      normalCapacity = physx::PxMax(
          weightShare,
          physx::PxMax(-normalForce, priorNormalForce));
    }
    // The two support normals and the locked joint-normal row are redundant.
    // With unequal endpoint masses the AL normal multiplier can transfer the
    // heavy body's reaction to the light endpoint, so only the exact
    // per-body weight share is an admissible Coulomb budget. Preserve the
    // accepted symmetric P4AB normal-history budget when both endpoint
    // masses are equal; no cross-mass attribution exists in that boundary.
    const physx::PxReal mu = contactCoulombMu(contact);
    const physx::PxReal tangentPenalty0 =
        physx::PxMax(contact.tangentPenalty0,
                     contactBoostFloor);
    const physx::PxReal tangentPenalty1 =
        physx::PxMax(contact.tangentPenalty1,
                     contactBoostFloor);
    physx::PxReal tangentForce0 =
        tangentPenalty0 * tangentViolation0 +
        contact.tangentLambda0;
    physx::PxReal tangentForce1 =
        tangentPenalty1 * tangentViolation1 +
        contact.tangentLambda1;
    const physx::PxReal unconstrainedTangentForce =
        physx::PxSqrt(tangentForce0 * tangentForce0 +
                      tangentForce1 * tangentForce1);
    const physx::PxReal tangentForceLimit =
        mu * normalCapacity;
    avbdProjectImpulseCone(tangentForceLimit,
                           tangentForce0, tangentForce1);
    const bool forceSaturated =
        unconstrainedTangentForce > tangentForceLimit;

    const physx::PxVec3 tangents[2] = {
        contact.tangent0, contact.tangent1};
    const physx::PxReal tangentPenalties[2] = {
        tangentPenalty0, tangentPenalty1};
    const physx::PxReal tangentForces[2] = {
        tangentForce0, tangentForce1};
    for (physx::PxU32 tangent = 0; tangent < 2u;
         ++tangent) {
      const physx::PxVec3 axis =
          tangents[tangent] * sign;
      CoupledIslandRow row;
      row.bodyA = bodyIndex;
      row.bodyB = PX_MAX_U32;
      row.jacobianA =
          AvbdVec6(axis, r.cross(tangents[tangent]) * sign);
      row.jacobianB = AvbdVec6();
      // Outside the Coulomb disk the projected force is locally bounded,
      // so retaining an unconstrained tangent Hessian would make the
      // saturated row artificially bilateral.
      row.penalty =
          forceSaturated ? 0.0f : tangentPenalties[tangent];
      row.force = tangentForces[tangent];
      addCoupledRow(row, rows, gradient, preconditioner);
    }
  }
  return true;
}

} // namespace Dy
} // namespace physx
