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

#include "avbd/solver/rigid/DyAvbdRigidPhases.h"
#include "common/PxProfileZone.h"

namespace physx {
namespace Dy {

void runAvbdSoftPredictionPhase(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    physx::PxReal dt, const physx::PxVec3 &gravity,
    bool hasPreparedSoftPrediction) {
  PX_PROFILE_ZONE("AVBD.prediction", 0);
  if (!hasPreparedSoftPrediction) {
    for (physx::PxU32 particleIndex = 0;
         particleIndex < numSoftParticles; ++particleIndex)
      softParticles[particleIndex].computePrediction(dt, gravity);
  }
}

void initializeAvbdNoContactBodies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap) {
  PX_PROFILE_ZONE("AVBD.initNoContactBodies", 0);
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    if (bodies[bodyIndex].invMass <= 0.0f)
      continue;

    bool hasContacts = false;
    if (contactMap && contactMap->numBodies > 0) {
      const physx::PxU32 *constraintIndices = nullptr;
      physx::PxU32 constraintCount = 0;
      contactMap->getBodyConstraints(
          bodyIndex, constraintIndices, constraintCount);
      hasContacts = constraintCount > 0;
    } else if (numContacts > 0) {
      for (physx::PxU32 contactIndex = 0;
           contactIndex < numContacts; ++contactIndex) {
        if (contacts[contactIndex].header.bodyIndexA == bodyIndex ||
            contacts[contactIndex].header.bodyIndexB == bodyIndex) {
          hasContacts = true;
          break;
        }
      }
    }

    if (!hasContacts) {
      bodies[bodyIndex].position = bodies[bodyIndex].inertialPosition;
      bodies[bodyIndex].rotation = bodies[bodyIndex].inertialRotation;
    }
  }
}

void warmstartAvbdRigidBodies(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<bool> &touchesKinematicShell, physx::PxReal dt,
    physx::PxReal invDt, const physx::PxVec3 &gravity,
    physx::PxReal shellFastImpactSpeed) {
  const physx::PxReal gravMag = gravity.magnitude();
  const physx::PxVec3 gravDir =
      (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    AvbdSolverBody &body = bodies[i];
    body.prevPosition = body.position;
    body.prevRotation = body.rotation;

    if (body.invMass <= 0.0f)
      continue;

    if (touchesKinematicShell[i]) {
      const bool fastImpact =
          body.linearVelocity.magnitude() > shellFastImpactSpeed;
      body.position = fastImpact
                          ? body.inertialPosition
                          : body.prevPosition + body.linearVelocity * dt;
      body.rotation = body.inertialRotation;
    } else {
      const physx::PxVec3 accel =
          (body.linearVelocity - body.prevLinearVelocity) * invDt;
      const physx::PxReal accelWeight =
          (gravMag > 1e-6f)
              ? physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f)
              : 0.0f;
      body.position = body.prevPosition + body.linearVelocity * dt +
                      gravity * (accelWeight * dt * dt);
      body.rotation = body.inertialRotation;
    }
    body.projectLockedPose(body.prevPosition, body.prevRotation);
  }
}

namespace {

// A hard rigid-contact row is conditioned against its complete spatial
// response w = J M^-1 J^T.  The ratio is dimensionless and therefore does not
// depend on mass, timestep, endpoint type or contact lever arm.
static const physx::PxReal kRigidContactConditioning = 2.0f;

static physx::PxReal computeContactRowSpatialResponse(
    const AvbdContactConstraint &contact, const physx::PxVec3 &axis,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies) {
  physx::PxReal response = 0.0f;
  const auto addEndpoint = [&](physx::PxU32 bodyIndex,
                               const physx::PxVec3 &localPoint,
                               physx::PxReal linearScale,
                               physx::PxReal angularScale) {
    if (bodyIndex >= numBodies || linearScale < 0.0f ||
        angularScale < 0.0f)
      return;

    const AvbdSolverBody &body = bodies[bodyIndex];
    if (body.invMass <= 0.0f)
      return;

    physx::PxVec3 linearJacobian = axis;
    body.projectLockedLinearVector(linearJacobian);
    response += linearScale * body.invMass *
                linearJacobian.magnitudeSquared();

    physx::PxVec3 angularJacobian =
        body.rotation.rotate(localPoint).cross(axis);
    body.projectLockedAngularVector(angularJacobian);
    const physx::PxReal angularResponse = angularJacobian.dot(
        body.invInertiaWorld * angularJacobian);
    if (angularResponse > 0.0f)
      response += angularScale * angularResponse;
  };

  addEndpoint(contact.header.bodyIndexA, contact.contactPointA,
              contact.invMassScaleA, contact.invInertiaScaleA);
  addEndpoint(contact.header.bodyIndexB, contact.contactPointB,
              contact.invMassScaleB, contact.invInertiaScaleB);
  return physx::PxIsFinite(response) && response > 1.0e-12f
             ? response
             : 0.0f;
}

static physx::PxReal computeContactRowPenaltyFloor(
    const AvbdContactConstraint &contact, const physx::PxVec3 &axis,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxReal invDt2) {
  const physx::PxReal response = computeContactRowSpatialResponse(
      contact, axis, bodies, numBodies);
  return response > 0.0f
             ? kRigidContactConditioning * invDt2 / response
             : 0.0f;
}

static void applyPenaltyFloor(physx::PxReal &penalty,
                              physx::PxReal penaltyFloor,
                              physx::PxReal penaltyMin) {
  if (!(penaltyFloor > 0.0f) || !physx::PxIsFinite(penaltyFloor))
    return;
  if (penalty <= penaltyMin || penalty < penaltyFloor)
    penalty = penaltyFloor;
}

static bool isColdDeformableContactLoaded(
    const AvbdContactConstraint &contact, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies) {
  if (!hasDeformableStaticAnchor(contact))
    return false;

  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  const physx::PxVec3 worldA =
      bodyA < numBodies
          ? bodies[bodyA].position +
                bodies[bodyA].rotation.rotate(contact.contactPointA)
          : contact.contactPointA;
  const physx::PxVec3 worldB =
      bodyB < numBodies
          ? bodies[bodyB].position +
                bodies[bodyB].rotation.rotate(contact.contactPointB)
          : contact.contactPointB;
  const physx::PxReal violation = finalizeBodyVsStaticViolation(
      (worldA - worldB).dot(contact.contactNormal) +
          contact.penetrationDepth,
      contact.penetrationDepth);
  return violation < 0.0f;
}

} // namespace

void applyAvbdLoadedTangentPenaltyFloor(
    AvbdContactConstraint &contact, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxReal invDt2,
    physx::PxReal penaltyMin) {
  if (!hasPositionTangentParticipation(contact) ||
      (contact.friction <= 0.0f && contact.staticFriction <= 0.0f))
    return;

  applyPenaltyFloor(
      contact.tangentPenalty0,
      computeContactRowPenaltyFloor(contact, contact.tangent0,
                                    bodies, numBodies, invDt2),
      penaltyMin);
  applyPenaltyFloor(
      contact.tangentPenalty1,
      computeContactRowPenaltyFloor(contact, contact.tangent1,
                                    bodies, numBodies, invDt2),
      penaltyMin);
}

void applyAvbdPenaltyFloor(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxReal invDt2, physx::PxReal penaltyMin) {
  if (!contacts || !bodies || numContacts == 0u || numBodies == 0u ||
      !(invDt2 > 0.0f))
    return;

  PX_PROFILE_ZONE("AVBD.penaltyFloor", 0);
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    AvbdContactConstraint &contact = contacts[contactIndex];
    applyPenaltyFloor(
        contact.header.penalty,
        computeContactRowPenaltyFloor(contact, contact.contactNormal,
                                      bodies, numBodies, invDt2),
        penaltyMin);
    // Deformable mesh points deliberately cold-start their multiplier because
    // the moving triangle anchor is not persistent.  Classify those rows from
    // the same predicted normal violation used by PositionAL; otherwise a
    // genuine mesh contact can never acquire the tangent conditioning needed
    // by the four-iteration solve.  Positive-separation speculative rows keep
    // their adaptive penalty and cannot create a load-free glue network.
    const bool loadedCoulombRow =
        contact.header.lambda < 0.0f ||
        isColdDeformableContactLoaded(contact, bodies, numBodies);
    if (loadedCoulombRow)
      applyAvbdLoadedTangentPenaltyFloor(
          contact, bodies, numBodies, invDt2, penaltyMin);
  }
}

} // namespace Dy
} // namespace physx
