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

#include "DyAvbdContactPrep.h"
#include "foundation/PxMath.h"

namespace physx {
namespace Dy {

//=============================================================================
// AvbdFrictionConstraint Implementation
//=============================================================================

void AvbdFrictionConstraint::initialize(
    physx::PxU32 bodyA, physx::PxU32 bodyB, const physx::PxVec3 &contactPtA,
    const physx::PxVec3 &contactPtB, const physx::PxVec3 &normal,
    physx::PxReal friction, physx::PxReal rho) {
  header.bodyIndexA = bodyA;
  header.bodyIndexB = bodyB;
  header.type = AvbdConstraintType::eFRICTION;
  header.flags = 0;
  header.compliance = 0.0f;
  header.damping = 0.0f;
  header.lambda = 0.0f;
  header.rho = rho;

  contactPointA = contactPtA;
  contactPointB = contactPtB;
  frictionCoeff = friction;
  lambdaTangent1 = 0.0f;
  lambdaTangent2 = 0.0f;
  maxFrictionForce = 0.0f;

  computeTangentBasis(normal);
}

void AvbdFrictionConstraint::computeTangentBasis(const physx::PxVec3 &normal) {
  // Compute orthonormal tangent basis from normal
  // Use Gram-Schmidt or robust method

  physx::PxVec3 up(0.0f, 1.0f, 0.0f);
  physx::PxVec3 right(1.0f, 0.0f, 0.0f);

  // Choose axis most different from normal
  physx::PxReal dotUp = physx::PxAbs(normal.dot(up));
  physx::PxReal dotRight = physx::PxAbs(normal.dot(right));

  if (dotUp < dotRight) {
    tangent1 = normal.cross(up);
  } else {
    tangent1 = normal.cross(right);
  }

  tangent1.normalize();
  tangent2 = normal.cross(tangent1);
  tangent2.normalize();
}

physx::PxVec3
AvbdFrictionConstraint::computeTangentVelocity(const AvbdSolverBody &bodyA,
                                               const AvbdSolverBody &bodyB,
                                               physx::PxReal invDt) const {
  // Compute relative velocity at contact point
  physx::PxVec3 velA =
      bodyA.linearVelocity + bodyA.angularVelocity.cross(contactPointA);
  physx::PxVec3 velB =
      bodyB.linearVelocity + bodyB.angularVelocity.cross(contactPointB);
  physx::PxVec3 relVel = velA - velB;

  // Project onto tangent plane
  physx::PxVec3 tangentVel;
  tangentVel.x = relVel.dot(tangent1);
  tangentVel.y = relVel.dot(tangent2);
  tangentVel.z = 0.0f;

  PX_UNUSED(invDt);
  return tangentVel;
}

//=============================================================================
// AvbdContactPrep Implementation
//=============================================================================

void AvbdContactPrep::convertContact(
    const physx::PxVec3 &contactPoint, const physx::PxVec3 &contactNormal,
    physx::PxReal penetration, physx::PxU32 bodyIndexA,
    physx::PxU32 bodyIndexB, const AvbdSolverBody &bodyA,
    const AvbdSolverBody &bodyB, physx::PxReal restitution,
    physx::PxReal friction, const AvbdSolverConfig &config,
    AvbdContactConstraint &outContact, AvbdFrictionConstraint *outFriction) {
  // Initialize contact constraint header
  outContact.header.bodyIndexA = bodyIndexA;
  outContact.header.bodyIndexB = bodyIndexB;
  outContact.header.type = AvbdConstraintType::eCONTACT;
  outContact.header.flags = 0;
  outContact.header.compliance = config.contactCompliance;
  outContact.header.damping = 0.0f;
  outContact.header.lambda = 0.0f;
  outContact.header.rho = config.initialRho;

  // Transform contact point to body local space
  physx::PxVec3 worldContactA = contactPoint;
  physx::PxVec3 worldContactB = contactPoint;

  // For body A: r_A = contact - posA (in world space, will be rotated)
  physx::PxQuat invRotA = bodyA.rotation.getConjugate();
  outContact.contactPointA = invRotA.rotate(worldContactA - bodyA.position);

  // For body B: r_B = contact - posB
  physx::PxQuat invRotB = bodyB.rotation.getConjugate();
  outContact.contactPointB = invRotB.rotate(worldContactB - bodyB.position);

  // Store normal and penetration
  outContact.contactNormal = contactNormal;
  outContact.penetrationDepth = penetration;

  // Material properties
  outContact.restitution = restitution;
  outContact.friction = friction;

  // Initialize friction constraint if requested
  if (outFriction != nullptr && friction > 0.0f) {
    outFriction->initialize(bodyIndexA, bodyIndexB, outContact.contactPointA,
                            outContact.contactPointB, contactNormal, friction,
                            config.initialRho);
  }
}

void AvbdContactPrep::buildManifold(
    const physx::PxVec3 *contactPoints, const physx::PxVec3 *contactNormals,
    const physx::PxReal *penetrations, physx::PxU32 numContacts,
    physx::PxU32 bodyIndexA, physx::PxU32 bodyIndexB, physx::PxReal friction,
    physx::PxReal restitution, AvbdContactManifold &outManifold) {
  outManifold.reset();
  outManifold.bodyIndexA = bodyIndexA;
  outManifold.bodyIndexB = bodyIndexB;
  outManifold.combinedFriction = friction;
  outManifold.combinedRestitution = restitution;

  // Add contacts to manifold (up to max)
  physx::PxU32 count =
      (numContacts < AvbdContactManifold::MAX_CONTACTS_PER_MANIFOLD)
          ? numContacts
          : AvbdContactManifold::MAX_CONTACTS_PER_MANIFOLD;

  for (physx::PxU32 i = 0; i < count; ++i) {
    outManifold.addContact(contactPoints[i], contactNormals[i],
                           penetrations[i]);
  }
}

physx::PxU32 AvbdContactPrep::manifoldToConstraints(
    const AvbdContactManifold &manifold, const AvbdSolverBody &bodyA,
    const AvbdSolverBody &bodyB, const AvbdSolverConfig &config,
    AvbdContactConstraint *outContacts, AvbdFrictionConstraint *outFriction,
    physx::PxU32 maxConstraints) {
  if (manifold.numContacts == 0) {
    return 0;
  }

  physx::PxU32 numCreated = 0;

  // Option 1: Create one constraint per contact point
  // Option 2: Create single constraint from manifold average
  // We use option 1 for better accuracy, with friction on the deepest contact

  for (physx::PxU32 i = 0;
       i < manifold.numContacts && numCreated < maxConstraints; ++i) {
    AvbdFrictionConstraint *frictionPtr = nullptr;

    // Add friction only for the deepest contact (i == 0 after sorting by
    // penetration)
    if (i == 0 && outFriction != nullptr && manifold.combinedFriction > 0.0f) {
      frictionPtr = &outFriction[numCreated];
    }

    convertContact(manifold.contactPoints[i], manifold.averageNormal,
                   manifold.penetrations[i], manifold.bodyIndexA,
                   manifold.bodyIndexB, bodyA, bodyB,
                   manifold.combinedRestitution, manifold.combinedFriction,
                   config, outContacts[numCreated], frictionPtr);

    numCreated++;
  }

  return numCreated;
}

} // namespace Dy
} // namespace physx
