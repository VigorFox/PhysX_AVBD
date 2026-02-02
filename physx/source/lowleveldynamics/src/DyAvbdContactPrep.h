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

#ifndef DY_AVBD_CONTACT_PREP_H
#define DY_AVBD_CONTACT_PREP_H

#include "DyAvbdConstraint.h"
#include "DyAvbdSolverBody.h"
#include "DyAvbdTypes.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

#pragma warning(push)
#pragma warning(disable : 4324)

namespace physx {

// Forward declarations for PhysX contact types
struct PxContactPoint;
class PxsContactManager;

namespace Dy {

/**
 * @brief Contact manifold for aggregating multiple contact points
 *
 * A manifold groups contacts between the same pair of bodies to improve
 * solver stability and reduce constraint count.
 */
struct AvbdContactManifold {
  physx::PxU32 bodyIndexA;
  physx::PxU32 bodyIndexB;
  physx::PxVec3 averageNormal;
  physx::PxVec3 averageContactPoint;
  physx::PxReal maxPenetration;
  physx::PxReal combinedFriction;
  physx::PxReal combinedRestitution;
  physx::PxU32 numContacts;

  static const physx::PxU32 MAX_CONTACTS_PER_MANIFOLD = 4;
  physx::PxVec3 contactPoints[MAX_CONTACTS_PER_MANIFOLD];
  physx::PxReal penetrations[MAX_CONTACTS_PER_MANIFOLD];

  PX_FORCE_INLINE void reset() {
    bodyIndexA = 0xFFFFFFFF;
    bodyIndexB = 0xFFFFFFFF;
    averageNormal = physx::PxVec3(0.0f, 1.0f, 0.0f);
    averageContactPoint = physx::PxVec3(0.0f);
    maxPenetration = 0.0f;
    combinedFriction = 0.5f;
    combinedRestitution = 0.0f;
    numContacts = 0;
  }

  /**
   * @brief Add a contact point to the manifold
   */
  PX_FORCE_INLINE void addContact(const physx::PxVec3 &point,
                                  const physx::PxVec3 &normal,
                                  physx::PxReal penetration) {
    if (numContacts < MAX_CONTACTS_PER_MANIFOLD) {
      contactPoints[numContacts] = point;
      penetrations[numContacts] = penetration;
      numContacts++;

      // Update averages
      averageNormal =
          (averageNormal * (physx::PxReal)(numContacts - 1) + normal) /
          (physx::PxReal)numContacts;
      averageNormal.normalize();

      averageContactPoint =
          (averageContactPoint * (physx::PxReal)(numContacts - 1) + point) /
          (physx::PxReal)numContacts;

      if (penetration < maxPenetration) {
        maxPenetration = penetration;
      }
    }
  }
};

/**
 * @brief Friction constraint for AVBD solver
 *
 * Implements Coulomb friction model in the AVBD framework.
 * Friction is treated as a pair of tangent constraints with force limits.
 */
struct PX_ALIGN_PREFIX(16) AvbdFrictionConstraint {
  AvbdConstraintHeader header;

  physx::PxVec3 contactPointA; //!< Contact point in body A local space
  physx::PxReal frictionCoeff;
  physx::PxVec3 contactPointB; //!< Contact point in body B local space
  physx::PxReal padding1;

  physx::PxVec3 tangent1;       //!< First tangent direction
  physx::PxReal lambdaTangent1; //!< Friction force in tangent1 direction
  physx::PxVec3 tangent2;       //!< Second tangent direction
  physx::PxReal lambdaTangent2; //!< Friction force in tangent2 direction

  physx::PxReal maxFrictionForce; //!< mu * normalForce limit
  physx::PxReal padding2[3];

  PX_FORCE_INLINE AvbdFrictionConstraint()
      : contactPointA(0.0f), frictionCoeff(0.5f), contactPointB(0.0f),
        padding1(0.0f), tangent1(1.0f, 0.0f, 0.0f), lambdaTangent1(0.0f),
        tangent2(0.0f, 0.0f, 1.0f), lambdaTangent2(0.0f),
        maxFrictionForce(0.0f) {
    header.type = AvbdConstraintType::eFRICTION;
    padding2[0] = padding2[1] = padding2[2] = 0.0f;
  }

  /**
   * @brief Initialize friction constraint from contact data
   */
  void initialize(physx::PxU32 bodyA, physx::PxU32 bodyB,
                  const physx::PxVec3 &contactPtA,
                  const physx::PxVec3 &contactPtB,
                  const physx::PxVec3 &normal, physx::PxReal friction,
                  physx::PxReal rho);

  /**
   * @brief Compute tangent basis from normal
   */
  void computeTangentBasis(const physx::PxVec3 &normal);

  /**
   * @brief Update friction force limit based on normal force
   */
  PX_FORCE_INLINE void updateFrictionLimit(physx::PxReal normalForce) {
    maxFrictionForce = frictionCoeff * physx::PxAbs(normalForce);
  }

  /**
   * @brief Compute friction violation (relative tangent velocity)
   */
  physx::PxVec3 computeTangentVelocity(const AvbdSolverBody &bodyA,
                                        const AvbdSolverBody &bodyB,
                                        physx::PxReal invDt) const;

} PX_ALIGN_SUFFIX(16);

/**
 * @brief Contact preparation helper class
 *
 * Converts PhysX collision detection results to AVBD constraint format.
 */
class AvbdContactPrep {
public:
  /**
   * @brief Convert a single contact point to AVBD constraint
   */
  static void
  convertContact(const physx::PxVec3 &contactPoint,
                 const physx::PxVec3 &contactNormal,
                 physx::PxReal penetration, physx::PxU32 bodyIndexA,
                 physx::PxU32 bodyIndexB, const AvbdSolverBody &bodyA,
                 const AvbdSolverBody &bodyB, physx::PxReal restitution,
                 physx::PxReal friction, const AvbdSolverConfig &config,
                 AvbdContactConstraint &outContact,
                 AvbdFrictionConstraint *outFriction = nullptr);

  /**
   * @brief Build contact manifold from multiple contact points
   */
  static void buildManifold(const physx::PxVec3 *contactPoints,
                            const physx::PxVec3 *contactNormals,
                            const physx::PxReal *penetrations,
                            physx::PxU32 numContacts, physx::PxU32 bodyIndexA,
                            physx::PxU32 bodyIndexB, physx::PxReal friction,
                            physx::PxReal restitution,
                            AvbdContactManifold &outManifold);

  /**
   * @brief Convert manifold to AVBD constraints
   * @return Number of constraints created
   */
  static physx::PxU32 manifoldToConstraints(
      const AvbdContactManifold &manifold, const AvbdSolverBody &bodyA,
      const AvbdSolverBody &bodyB, const AvbdSolverConfig &config,
      AvbdContactConstraint *outContacts, AvbdFrictionConstraint *outFriction,
      physx::PxU32 maxConstraints);

  /**
   * @brief Compute combined friction coefficient
   */
  static PX_FORCE_INLINE physx::PxReal
  combineFriction(physx::PxReal frictionA, physx::PxReal frictionB) {
    // Geometric mean for friction combining
    return physx::PxSqrt(frictionA * frictionB);
  }

  /**
   * @brief Compute combined restitution coefficient
   */
  static PX_FORCE_INLINE physx::PxReal
  combineRestitution(physx::PxReal restA, physx::PxReal restB) {
    // Maximum for restitution combining
    return (restA > restB) ? restA : restB;
  }
};

} // namespace Dy
} // namespace physx

#pragma warning(pop)

#endif // DY_AVBD_CONTACT_PREP_H
