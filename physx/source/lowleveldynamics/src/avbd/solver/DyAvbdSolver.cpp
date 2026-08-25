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

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/core/DyAvbdBoundedProjection.h"
#include "avbd/solver/rigid/DyAvbdRigidPhases.h"
#include "avbd/solver/post_al/DyAvbdPostAl.h"
#include "common/PxProfileZone.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace physx {
namespace Dy {

namespace {

struct KahanSum {
  physx::PxReal sum{0.0f};
  physx::PxReal c{0.0f};

  void add(physx::PxReal value) {
    const physx::PxReal y = value - c;
    const physx::PxReal t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

struct AvbdDoubleVec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  AvbdDoubleVec3() = default;
  AvbdDoubleVec3(double xValue, double yValue, double zValue)
      : x(xValue), y(yValue), z(zValue) {}

  explicit AvbdDoubleVec3(const physx::PxVec3 &value)
      : x(double(value.x)), y(double(value.y)), z(double(value.z)) {}

  AvbdDoubleVec3 operator+(const AvbdDoubleVec3 &other) const {
    return AvbdDoubleVec3(x + other.x, y + other.y, z + other.z);
  }

  AvbdDoubleVec3 operator-(const AvbdDoubleVec3 &other) const {
    return AvbdDoubleVec3(x - other.x, y - other.y, z - other.z);
  }

  AvbdDoubleVec3 operator*(double scale) const {
    return AvbdDoubleVec3(x * scale, y * scale, z * scale);
  }

  AvbdDoubleVec3 &operator+=(const AvbdDoubleVec3 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  bool isFinite() const {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }
};

static PX_FORCE_INLINE AvbdDoubleVec3 crossAvbdDouble(
    const AvbdDoubleVec3 &a, const AvbdDoubleVec3 &b) {
  return AvbdDoubleVec3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
}

static PX_FORCE_INLINE double magnitudeAvbdDouble(
    const AvbdDoubleVec3 &value) {
  return std::sqrt(
      value.x * value.x + value.y * value.y + value.z * value.z);
}

struct AvbdDoubleSymmetric3 {
  double value[3][3]{};

  bool isFinite() const {
    for (physx::PxU32 row = 0; row < 3; ++row) {
      for (physx::PxU32 column = 0; column < 3; ++column) {
        if (!std::isfinite(value[row][column]))
          return false;
      }
    }
    return true;
  }

  AvbdDoubleVec3 multiply(const AvbdDoubleVec3 &vector) const {
    return AvbdDoubleVec3(
        value[0][0] * vector.x + value[0][1] * vector.y +
            value[0][2] * vector.z,
        value[1][0] * vector.x + value[1][1] * vector.y +
            value[1][2] * vector.z,
        value[2][0] * vector.x + value[2][1] * vector.y +
            value[2][2] * vector.z);
  }
};

struct AvbdDoubleCholesky3 {
  double value[3][3]{};
};

static bool factorAvbdDoubleSpd3(
    const AvbdDoubleSymmetric3 &matrix,
    AvbdDoubleCholesky3 &factor) {
  if (!matrix.isFinite())
    return false;
  const double scale = std::max(
      std::fabs(matrix.value[0][0]),
      std::max(std::fabs(matrix.value[1][1]),
               std::fabs(matrix.value[2][2])));
  if (!std::isfinite(scale) || scale <= 0.0)
    return false;
  const double pivotTolerance = std::max(
      std::numeric_limits<double>::min(),
      scale * 64.0 * std::numeric_limits<double>::epsilon());

  const double pivot0 = matrix.value[0][0];
  if (!std::isfinite(pivot0) || pivot0 <= pivotTolerance)
    return false;
  factor.value[0][0] = std::sqrt(pivot0);
  factor.value[1][0] =
      matrix.value[1][0] / factor.value[0][0];
  factor.value[2][0] =
      matrix.value[2][0] / factor.value[0][0];

  const double pivot1 =
      matrix.value[1][1] -
      factor.value[1][0] * factor.value[1][0];
  if (!std::isfinite(pivot1) || pivot1 <= pivotTolerance)
    return false;
  factor.value[1][1] = std::sqrt(pivot1);
  factor.value[2][1] =
      (matrix.value[2][1] -
       factor.value[2][0] * factor.value[1][0]) /
      factor.value[1][1];

  const double pivot2 =
      matrix.value[2][2] -
      factor.value[2][0] * factor.value[2][0] -
      factor.value[2][1] * factor.value[2][1];
  if (!std::isfinite(pivot2) || pivot2 <= pivotTolerance)
    return false;
  factor.value[2][2] = std::sqrt(pivot2);
  return std::isfinite(factor.value[0][0]) &&
         std::isfinite(factor.value[1][0]) &&
         std::isfinite(factor.value[2][0]) &&
         std::isfinite(factor.value[1][1]) &&
         std::isfinite(factor.value[2][1]) &&
         std::isfinite(factor.value[2][2]);
}

static bool solveAvbdDoubleSpd3(
    const AvbdDoubleCholesky3 &factor,
    const AvbdDoubleVec3 &rhs, AvbdDoubleVec3 &solution) {
  const double y0 = rhs.x / factor.value[0][0];
  const double y1 =
      (rhs.y - factor.value[1][0] * y0) /
      factor.value[1][1];
  const double y2 =
      (rhs.z - factor.value[2][0] * y0 -
       factor.value[2][1] * y1) /
      factor.value[2][2];
  solution.z = y2 / factor.value[2][2];
  solution.y =
      (y1 - factor.value[2][1] * solution.z) /
      factor.value[1][1];
  solution.x =
      (y0 - factor.value[1][0] * solution.y -
       factor.value[2][0] * solution.z) /
      factor.value[0][0];
  return solution.isFinite();
}

static bool invertAvbdWorldInverseInertia(
    const physx::PxMat33 &worldInverseInertia,
    AvbdDoubleSymmetric3 &worldInertia) {
  AvbdDoubleSymmetric3 inverse;
  inverse.value[0][0] = double(worldInverseInertia.column0.x);
  inverse.value[1][1] = double(worldInverseInertia.column1.y);
  inverse.value[2][2] = double(worldInverseInertia.column2.z);
  inverse.value[0][1] = inverse.value[1][0] =
      0.5 * double(worldInverseInertia.column0.y +
                   worldInverseInertia.column1.x);
  inverse.value[0][2] = inverse.value[2][0] =
      0.5 * double(worldInverseInertia.column0.z +
                   worldInverseInertia.column2.x);
  inverse.value[1][2] = inverse.value[2][1] =
      0.5 * double(worldInverseInertia.column1.z +
                   worldInverseInertia.column2.y);

  AvbdDoubleCholesky3 factor;
  if (!factorAvbdDoubleSpd3(inverse, factor))
    return false;
  AvbdDoubleVec3 columns[3];
  if (!solveAvbdDoubleSpd3(
          factor, AvbdDoubleVec3(1.0, 0.0, 0.0), columns[0]) ||
      !solveAvbdDoubleSpd3(
          factor, AvbdDoubleVec3(0.0, 1.0, 0.0), columns[1]) ||
      !solveAvbdDoubleSpd3(
          factor, AvbdDoubleVec3(0.0, 0.0, 1.0), columns[2]))
    return false;
  for (physx::PxU32 row = 0; row < 3; ++row) {
    for (physx::PxU32 column = 0; column < 3; ++column) {
      const double direct =
          row == 0 ? columns[column].x
                   : (row == 1 ? columns[column].y
                               : columns[column].z);
      const double transpose =
          column == 0 ? columns[row].x
                      : (column == 1 ? columns[row].y
                                     : columns[row].z);
      worldInertia.value[row][column] =
          0.5 * (direct + transpose);
    }
  }
  return worldInertia.isFinite();
}

static physx::PxReal computeRotationDeltaMagnitude(
    const physx::PxQuat &current, const physx::PxQuat &previous) {
  physx::PxQuat deltaQ = current * previous.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  return 2.0f * physx::PxSqrt(deltaQ.x * deltaQ.x +
                              deltaQ.y * deltaQ.y +
                              deltaQ.z * deltaQ.z);
}

static PX_FORCE_INLINE bool getAvbdBodyContactRange(
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    const physx::PxU32 *&indices, physx::PxU32 &count);

static bool bodyTouchesDeformableAnchorImpl(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex, const AvbdBodyConstraintMap *contactMap) {
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
    const physx::PxU32 bA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bB = contacts[c].header.bodyIndexB;
    if (!hasDeformableStaticAnchor(contacts[c]))
      continue;
    if (bA == bodyIndex || bB == bodyIndex)
      return true;
  }
  return false;
}

// The contact map is built once per island.  Keep the fallback for callers
// that do not provide it (notably a few legacy/deformable paths), but make the
// hot per-body post-AL loops consume only incident rows when it is available.
static PX_FORCE_INLINE bool getAvbdBodyContactRange(
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    const physx::PxU32 *&indices, physx::PxU32 &count) {
  if (!contactMap || !contactMap->constraintOffsets ||
      !contactMap->constraintCounts || bodyIndex >= contactMap->numBodies) {
    indices = nullptr;
    count = 0;
    return false;
  }
  contactMap->getBodyConstraints(bodyIndex, indices, count);
  return true;
}

static PX_FORCE_INLINE void computeAvbdMaterialSolveStartVelocity(
    const AvbdSolverBody &body, const physx::PxVec3 &gravity,
    physx::PxReal dt, physx::PxVec3 &linear,
    physx::PxVec3 &angular) {
  linear = body.linearVelocity + gravity * (body.gravityScale * dt);
  angular = body.angularVelocity;
  if (body.linearDamping > 0.0f)
    linear *= 1.0f / (1.0f + body.linearDamping * dt);
  if (body.angularDampingBody > 0.0f)
    angular *= 1.0f / (1.0f + body.angularDampingBody * dt);
  const physx::PxReal linearSpeedSquared = linear.magnitudeSquared();
  if (linearSpeedSquared > body.maxLinearVelocitySq &&
      body.maxLinearVelocitySq > 0.0f)
    linear *= physx::PxSqrt(
        body.maxLinearVelocitySq / linearSpeedSquared);
  const physx::PxReal angularSpeedSquared = angular.magnitudeSquared();
  if (angularSpeedSquared > body.maxAngularVelocitySq &&
      body.maxAngularVelocitySq > 0.0f)
    angular *= physx::PxSqrt(
        body.maxAngularVelocitySq / angularSpeedSquared);
  body.projectLockedLinearVector(linear);
  body.projectLockedAngularVector(angular);
}

struct AvbdMaterialContactGeometry {
  physx::PxVec3 materialArmA{0.0f};
  physx::PxVec3 materialArmB{0.0f};
  physx::PxVec3 solveStartMaterialArmA{0.0f};
  physx::PxVec3 solveStartMaterialArmB{0.0f};
  physx::PxVec3 positionAlArmA{0.0f};
  physx::PxVec3 positionAlArmB{0.0f};
  physx::PxVec3 staticVelocity{0.0f};
};

// Material velocity rows are defined at the fresh narrow-phase point, not at
// the persistent PositionAL anchors.  The two fresh endpoints can separate
// slightly after pose solving (and also when reconstructed at solve start), so
// use their midpoint as one common world application point.  Equal/opposite
// impulses at that point cannot manufacture an internal couple.  PositionAL
// removal deliberately keeps its own persistent-anchor Jacobian below.
static PX_FORCE_INLINE physx::PxVec3 getAvbdFreshEndpointWorldPoint(
    const AvbdContactConstraint &contact, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, bool endpointA, bool solveStart) {
  const physx::PxU32 bodyIndex =
      endpointA ? contact.header.bodyIndexA : contact.header.bodyIndexB;
  const physx::PxVec3 localOrWorldPoint =
      endpointA ? contact.detectionPointA : contact.detectionPointB;
  if (bodyIndex < numBodies) {
    const AvbdSolverBody &body = bodies[bodyIndex];
    const physx::PxVec3 &position =
        solveStart ? body.prevPosition : body.position;
    const physx::PxQuat &rotation =
        solveStart ? body.prevRotation : body.rotation;
    return position + rotation.rotate(localOrWorldPoint);
  }
  return solveStart ? contact.staticPrevWorldPoint : localOrWorldPoint;
}

static PX_FORCE_INLINE AvbdMaterialContactGeometry
buildAvbdMaterialContactGeometry(
    const AvbdContactConstraint &contact, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies, physx::PxReal invDt) {
  AvbdMaterialContactGeometry geometry;
  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  const physx::PxVec3 freshA = getAvbdFreshEndpointWorldPoint(
      contact, bodies, numBodies, true, false);
  const physx::PxVec3 freshB = getAvbdFreshEndpointWorldPoint(
      contact, bodies, numBodies, false, false);
  const physx::PxVec3 materialPoint = (freshA + freshB) * 0.5f;
  const physx::PxVec3 solveStartFreshA = getAvbdFreshEndpointWorldPoint(
      contact, bodies, numBodies, true, true);
  const physx::PxVec3 solveStartFreshB = getAvbdFreshEndpointWorldPoint(
      contact, bodies, numBodies, false, true);
  const physx::PxVec3 solveStartMaterialPoint =
      (solveStartFreshA + solveStartFreshB) * 0.5f;

  if (bodyA < numBodies) {
    geometry.materialArmA = materialPoint - bodies[bodyA].position;
    geometry.solveStartMaterialArmA =
        solveStartMaterialPoint - bodies[bodyA].prevPosition;
    geometry.positionAlArmA =
        bodies[bodyA].rotation.rotate(contact.contactPointA);
  }
  if (bodyB < numBodies) {
    geometry.materialArmB = materialPoint - bodies[bodyB].position;
    geometry.solveStartMaterialArmB =
        solveStartMaterialPoint - bodies[bodyB].prevPosition;
    geometry.positionAlArmB =
        bodies[bodyB].rotation.rotate(contact.contactPointB);
  }
  if ((bodyA < numBodies) != (bodyB < numBodies)) {
    // Surface motion must be measured at one material point.  The fresh
    // detection point is allowed to migrate across a stationary plane or
    // curved shape, so comparing it with the persistent previous anchor would
    // turn contact-manifold churn into a fictitious surface velocity.  Prep
    // stores the current/captured static material point in contactPointA/B and
    // its preceding world position in staticPrevWorldPoint.
    const physx::PxVec3 staticNow =
        bodyA < numBodies ? contact.contactPointB : contact.contactPointA;
    geometry.staticVelocity =
        (staticNow - contact.staticPrevWorldPoint) * invDt;
  }
  return geometry;
}

static PX_FORCE_INLINE bool isAvbdCentralNormalIsotropicContact(
    const AvbdSolverBody &body, const AvbdContactConstraint &contact,
    physx::PxU32 bodyIndex, physx::PxReal lengthScale) {
  const bool dynamicIsA = contact.header.bodyIndexA == bodyIndex;
  if (!dynamicIsA && contact.header.bodyIndexB != bodyIndex)
    return false;
  const physx::PxVec3 dynamicNormal =
      contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
  const physx::PxVec3 localPoint =
      dynamicIsA ? contact.contactPointA : contact.contactPointB;
  const physx::PxVec3 normalAngularJacobian =
      body.rotation.rotate(localPoint).cross(dynamicNormal);
  const physx::PxMat33 &inverseInertia = body.invInertiaWorld;
  const physx::PxReal inertiaMagnitude = physx::PxMax(
      1.0f,
      physx::PxMax(
          physx::PxAbs(inverseInertia.column0.x),
          physx::PxMax(physx::PxAbs(inverseInertia.column1.y),
                       physx::PxAbs(inverseInertia.column2.z))));
  const physx::PxReal inertiaTolerance = 1.0e-5f * inertiaMagnitude;
  const bool isotropicInertia =
      physx::PxAbs(inverseInertia.column0.x -
                   inverseInertia.column1.y) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column0.x -
                   inverseInertia.column2.z) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column0.y) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column0.z) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column1.x) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column1.z) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column2.x) <= inertiaTolerance &&
      physx::PxAbs(inverseInertia.column2.y) <= inertiaTolerance;
  const physx::PxReal lengthTolerance =
      1.0e-4f * physx::PxMax(lengthScale, 1.0f);
  return isotropicInertia &&
         normalAngularJacobian.magnitudeSquared() <=
             lengthTolerance * lengthTolerance;
}

// Chebyshev acceleration assumes that every iteration applies the same smooth
// stationary operator. A closing impact changes the unilateral active set and
// may also switch the friction cone between stick and slip, so extrapolating
// that iteration creates pose overshoot which BDF turns directly into excess
// linear and angular velocity. Keep acceleration for settled contact islands,
// but use the underlying AVBD primal/dual iteration while an impact can close
// a contact during this step.
static bool hasAvbdImpactActiveSetTransition(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> &linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal approachThreshold) {
  if (!bodies || !contacts || dt <= 0.0f ||
      linearVelAtSolveStart.size() != numBodies ||
      angularVelAtSolveStart.size() != numBodies)
    return false;

  const physx::PxReal invDt = 1.0f / dt;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &contact = contacts[c];
    if (contact.maxImpulse <= 0.0f)
      continue;

    const physx::PxU32 bodyIndex[2] = {
        contact.header.bodyIndexA, contact.header.bodyIndexB};
    const physx::PxVec3 axes[2] = {
        contact.contactNormal, -contact.contactNormal};
    const AvbdMaterialContactGeometry geometry =
        buildAvbdMaterialContactGeometry(
            contact, bodies, numBodies, invDt);
    const physx::PxVec3 solveStartArms[2] = {
        geometry.solveStartMaterialArmA,
        geometry.solveStartMaterialArmB};
    physx::PxReal relativeNormalVelocity = 0.0f;
    physx::PxU32 dynamicCount = 0;
    physx::PxU32 dynamicEndpoint = 0;
    for (physx::PxU32 endpoint = 0; endpoint < 2; ++endpoint) {
      const physx::PxU32 bodyIndexValue = bodyIndex[endpoint];
      if (bodyIndexValue >= numBodies)
        continue;
      relativeNormalVelocity +=
          linearVelAtSolveStart[bodyIndexValue].dot(axes[endpoint]) +
          angularVelAtSolveStart[bodyIndexValue].dot(
              solveStartArms[endpoint].cross(axes[endpoint]));
      dynamicEndpoint = endpoint;
      ++dynamicCount;
    }

    if (dynamicCount == 1u) {
      relativeNormalVelocity -=
          geometry.staticVelocity.dot(axes[dynamicEndpoint]);
    }

    const physx::PxReal approach = -relativeNormalVelocity;
    if (approach > approachThreshold &&
        approach > contact.detectionSeparation * invDt)
      return true;
  }
  return false;
}

// Enforce the velocity counterpart of body-vs-static locked D6 linear rows.
// Position-level AL convergence can leave a small first-step pose residual;
// reconstructing velocity directly from that residual creates a velocity that
// violates an otherwise hard joint.  This is a Jacobian/effective-mass
// projection, not a magnitude dead-zone.  Dynamic-dynamic, limited/free and
// driven rows remain outside this first body-vs-static correctness slice.
static void projectBodyStaticLockedD6LinearVelocitiesImpl(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint *joints, physx::PxU32 numJoints) {
  if (!bodies || !joints)
    return;

  for (physx::PxU32 ji = 0; ji < numJoints; ++ji) {
    const AvbdD6JointConstraint &joint = joints[ji];
    const bool aDynamic = joint.header.bodyIndexA < numBodies;
    const bool bDynamic = joint.header.bodyIndexB < numBodies;
    if (aDynamic == bDynamic)
      continue;

    AvbdSolverBody &body =
        bodies[aDynamic ? joint.header.bodyIndexA : joint.header.bodyIndexB];
    if (body.invMass <= 0.0f)
      continue;

    physx::PxQuat worldFrameA =
        aDynamic ? body.rotation * joint.localFrameA : joint.localFrameA;
    const physx::PxReal frameMagnitudeSquared = worldFrameA.magnitudeSquared();
    if (frameMagnitudeSquared > 1e-8f &&
        physx::PxIsFinite(frameMagnitudeSquared))
      worldFrameA *= 1.0f / physx::PxSqrt(frameMagnitudeSquared);
    const physx::PxVec3 r = body.rotation.rotate(
        aDynamic ? joint.anchorA : joint.anchorB);
    const bool allLinearLocked = joint.linearMotion == 0;
    const physx::PxU32 linearSourceRows[3] = {
        eJOINT_SOURCE_LINEAR_MOTION_X,
        eJOINT_SOURCE_LINEAR_MOTION_Y,
        eJOINT_SOURCE_LINEAR_MOTION_Z};
    const physx::PxU32 angularSourceRows[3] = {
        eJOINT_SOURCE_ANGULAR_MOTION_X,
        eJOINT_SOURCE_ANGULAR_MOTION_Y,
        eJOINT_SOURCE_ANGULAR_MOTION_Z};
    const auto isPositionGeometrySource =
        [&](physx::PxU32 sourceRow) -> bool {
      const AvbdCompiledJointObjective *objective =
          findAvbdJointObjectiveForSourceRow(
              joint.objectiveProgram, sourceRow);
      if (!objective ||
          objective->owner !=
              AvbdVelocityObjectiveOwner::PositionAL)
        return false;
      return objective->kind ==
                 AvbdJointObjectiveKind::OrdinaryD6Position ||
             objective->kind ==
                 AvbdJointObjectiveKind::CoupledFixedD6;
    };

    for (physx::PxU32 axis = 0; axis < 3; ++axis) {
      if (joint.getLinearMotion(axis) != 0 ||
          !isPositionGeometrySource(linearSourceRows[axis]))
        continue;

      physx::PxVec3 worldAxis(0.0f);
      worldAxis[axis] = 1.0f;
      if (!allLinearLocked)
        worldAxis = worldFrameA.rotate(worldAxis);

      const physx::PxVec3 rCrossAxis = r.cross(worldAxis);
      const physx::PxReal recipResponse =
          body.invMass +
          rCrossAxis.dot(body.invInertiaWorld.transform(rCrossAxis));
      if (recipResponse <= 1e-12f || !physx::PxIsFinite(recipResponse))
        continue;

      const physx::PxReal anchorSpeed =
          (body.linearVelocity + body.angularVelocity.cross(r)).dot(worldAxis);
      if (!physx::PxIsFinite(anchorSpeed))
        continue;

      // C = anchorA-anchorB, so the dynamic-B Jacobian is -J.
      const physx::PxReal dynamicSign = aDynamic ? 1.0f : -1.0f;
      const physx::PxReal impulse =
          -dynamicSign * anchorSpeed / recipResponse;
      body.linearVelocity += worldAxis * (dynamicSign * impulse * body.invMass);
      body.angularVelocity += body.invInertiaWorld.transform(
          rCrossAxis * (dynamicSign * impulse));
    }

    // A dynamic body fixed to a static/world endpoint has no admissible
    // spatial velocity.  Project the complete six-dimensional locked
    // subspace after pose-to-velocity reconstruction; the row-wise linear
    // projection above remains responsible for partially locked joints.
    bool completeFixedPositionObjective =
        allLinearLocked && joint.angularMotion == 0;
    for (physx::PxU32 axis = 0;
         axis < 3 && completeFixedPositionObjective; ++axis) {
      completeFixedPositionObjective =
          isPositionGeometrySource(linearSourceRows[axis]) &&
          isPositionGeometrySource(angularSourceRows[axis]);
    }
    if (completeFixedPositionObjective) {
      body.linearVelocity = physx::PxVec3(0.0f);
      body.angularVelocity = physx::PxVec3(0.0f);
    }

  }
}

// Suppress pose-solve bounce only on fast normal approach (sphere shot).
static const physx::PxReal kBodyStaticFastImpactSpeed =
    AvbdConstants::AVBD_BODY_STATIC_FAST_IMPACT_SPEED;

struct AvbdPassiveMaterialComponentRow {
  physx::PxU32 bodyA{PX_MAX_U32};
  physx::PxU32 bodyB{PX_MAX_U32};
  physx::PxVec3 linearA{0.0f};
  physx::PxVec3 angularA{0.0f};
  physx::PxVec3 linearB{0.0f};
  physx::PxVec3 angularB{0.0f};
  physx::PxVec3 positionAlAngularA{0.0f};
  physx::PxVec3 positionAlAngularB{0.0f};
};

static bool projectAvbdCoulombNcpImpulse(
    double mu, double &normal, double &tangent0, double &tangent1);

struct AvbdCoulombNcpLayout {
  physx::PxU32 rowCount{0u};
  physx::PxU32 normalCount{0u};
  physx::PxU32 anchorCount{0u};
  physx::PxU32 patchCount{0u};
  const physx::PxU32 *normalRows{nullptr};
  const physx::PxU32 *normalPatches{nullptr};
  const physx::PxU32 *tangentRows{nullptr};
  const physx::PxU32 *tangentPatches{nullptr};
  const physx::PxU32 *patchAnchorCounts{nullptr};
  const double *patchFriction{nullptr};
};

static bool solveAvbdCoulombNcpFixedPoint(
    const double *response, const double *q,
    const AvbdCoulombNcpLayout &layout, double *impulses,
    double *scratch);

/**
 * Close every ordinary rigid material row in a connected component as one
 * total-impulse product-cone problem.  Geometry remains PositionAL-owned.
 * The material baseline is the final pose velocity with the complete raw AL
 * response removed, so body-static support and dynamic-dynamic transfer see
 * one simultaneous objective instead of two order-dependent consumers.
 */
static bool mayHavePostAlContactWork(
    const AvbdPostAlContactWorkPlan *workPlan, physx::PxU8 work) {
  return !workPlan || workPlan->mayHave(work);
}

// Classify one final, validated contact program for the three post-AL
// consumers below.  The point predicate deliberately mirrors its consumer's
// first three continues, including the NaN behavior of !(magnitudeSq <= eps).
static physx::PxU8 collectValidatedPostAlContactWork(
    const AvbdContactConstraint &contact, const AvbdSolverBody *bodies,
    physx::PxU32 numBodies) {
  physx::PxU8 work = 0;
  bool velocityFrictionManifoldOwner = false;
  const AvbdCompiledContactObjectiveProgram &program =
      contact.objectiveProgram;
  for (physx::PxU32 entryIndex = 0; entryIndex < program.entryCount;
       ++entryIndex) {
    const AvbdCompiledVelocityObjective &entry = program.entries[entryIndex];
    if (entry.owner == AvbdVelocityObjectiveOwner::ComponentFinalize &&
        entry.kind == AvbdVelocityObjectiveKind::PassiveFriction) {
      work = physx::PxU8(
          work | AvbdPostAlContactWorkPlan::ePASSIVE_COMPONENT);
      velocityFrictionManifoldOwner = true;
    }
    if (entry.owner != AvbdVelocityObjectiveOwner::ManifoldFinalize)
      continue;
    if (entry.span == AvbdVelocityObjectiveSpan::NormalAndTangentCone &&
        entry.reconstruction ==
            AvbdVelocityObjectiveReconstruction::PoseResidual) {
      velocityFrictionManifoldOwner = true;
      work = physx::PxU8(
          work | AvbdPostAlContactWorkPlan::eCOMPLETE_MANIFOLD);
    }
  }

  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  const bool dynamicA =
      bodyA < numBodies && bodies[bodyA].invMass > 0.0f;
  const bool dynamicB =
      bodyB < numBodies && bodies[bodyB].invMass > 0.0f;
  if (!velocityFrictionManifoldOwner &&
      !(contact.targetVelocity.magnitudeSquared() <= 1.0e-12f) &&
      (dynamicA || dynamicB))
    work = physx::PxU8(work | AvbdPostAlContactWorkPlan::ePOINT_TARGET);
  return work;
}

static void applyAvbdPassiveFrictionComponents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    const AvbdPostAlContactWorkPlan *workPlan) {
  if (!bodies || !contacts || numBodies == 0 ||
      numContacts == 0 || dt <= 0.0f || !contactMap ||
      contactMap->numBodies != numBodies ||
      !contactMap->constraintOffsets || !contactMap->constraintCounts ||
      (contactMap->totalConstraintRefs > 0 &&
       !contactMap->constraintIndices) ||
      !linearVelAtSolveStart || !angularVelAtSolveStart ||
      !linearPoseVelocityGain || !angularPoseVelocityGain ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies ||
      linearPoseVelocityGain->size() != numBodies ||
      angularPoseVelocityGain->size() != numBodies ||
      !mayHavePostAlContactWork(
          workPlan, AvbdPostAlContactWorkPlan::ePASSIVE_COMPONENT))
    return;
  const physx::PxReal invDt = 1.0f / dt;
  const physx::PxReal effectiveBounceThreshold =
      bounceThreshold > 0.0f
          ? bounceThreshold
          : AvbdConstants::AVBD_BOUNCE_THRESHOLD;

  physx::PxArray<physx::PxU8> visitedContacts(numContacts);
  physx::PxArray<physx::PxU8> visitedBodies(numBodies);
  for (physx::PxU32 c = 0; c < numContacts; ++c)
    visitedContacts[c] = 0;
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    visitedBodies[body] = 0;

  // Reused O(B) scatter/mass/gather storage.  Each multiply only clears the
  // bodies in the current CSR component, so an iteration is O(C+B_component).
  physx::PxArray<physx::PxVec3> bodyLinearImpulse(numBodies);
  physx::PxArray<physx::PxVec3> bodyAngularImpulse(numBodies);
  physx::PxArray<physx::PxVec3> bodyLinearDelta(numBodies);
  physx::PxArray<physx::PxVec3> bodyAngularDelta(numBodies);

  for (physx::PxU32 seed = 0; seed < numContacts; ++seed) {
    if (visitedContacts[seed] || hasRigidMaterialConsumed(contacts[seed]) ||
        !hasVelocityPassiveFrictionComponentOwner(contacts[seed]))
      continue;

    physx::PxArray<physx::PxU32> componentContacts;
    physx::PxArray<physx::PxU32> bodyQueue;
    const auto enqueueBody = [&](physx::PxU32 bodyIndex) {
      if (bodyIndex < numBodies && !visitedBodies[bodyIndex]) {
        visitedBodies[bodyIndex] = 1;
        bodyQueue.pushBack(bodyIndex);
      }
    };
    const AvbdCompiledVelocityObjective *seedObjective =
        findAvbdVelocityObjective(
            contacts[seed].objectiveProgram,
            AvbdVelocityObjectiveOwner::ComponentFinalize,
            AvbdVelocityObjectiveKind::PassiveFriction);
    if (!seedObjective)
      continue;
    const physx::PxU64 objectiveKey = seedObjective->objectiveKey;
    enqueueBody(contacts[seed].header.bodyIndexA);
    enqueueBody(contacts[seed].header.bodyIndexB);
    bool supported = !bodyQueue.empty();
    for (physx::PxU32 queueIndex = 0;
         queueIndex < bodyQueue.size() && supported; ++queueIndex) {
      const physx::PxU32 bodyIndex = bodyQueue[queueIndex];
      const physx::PxU32 *mapIndices = nullptr;
      physx::PxU32 mapCount = 0;
      const bool hasMapRange = getAvbdBodyContactRange(
          contactMap, bodyIndex, mapIndices, mapCount);
      if (!hasMapRange) {
        supported = false;
        break;
      }
      const physx::PxU32 loopCount = mapCount;
      for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
        const physx::PxU32 c = mapIndices[loopIndex];
        if (c >= numContacts) {
          supported = false;
          break;
        }
        const AvbdContactConstraint &contact = contacts[c];
        if (hasRigidMaterialConsumed(contact))
          continue;
        if (contact.header.bodyIndexA != bodyIndex &&
            contact.header.bodyIndexB != bodyIndex)
          continue;
        const AvbdCompiledVelocityObjective *objective =
            findAvbdVelocityObjective(
                contact.objectiveProgram,
                AvbdVelocityObjectiveOwner::ComponentFinalize,
                AvbdVelocityObjectiveKind::PassiveFriction);
        // Component compilation is all-or-nothing.  Seeing any differently
        // owned incident material row here means the published topology is
        // inconsistent, so leave the complete pose/AL state untouched.
        if (!objective ||
            !hasVelocityPassiveFrictionComponentOwner(contact) ||
            objective->objectiveKey != objectiveKey ||
            objective->objectiveRowCount !=
                seedObjective->objectiveRowCount) {
          supported = false;
          break;
        }
        if (!visitedContacts[c]) {
          visitedContacts[c] = 1;
          componentContacts.pushBack(c);
          enqueueBody(contact.header.bodyIndexA);
          enqueueBody(contact.header.bodyIndexB);
        }
      }
    }
    supported = supported && !componentContacts.empty() &&
                componentContacts.size() ==
                    seedObjective->objectiveRowCount &&
                componentContacts.size() <= PX_MAX_U32 / 3u;
    if (!supported)
      continue;
    std::sort(componentContacts.begin(), componentContacts.end());

    const physx::PxU32 contactCount = componentContacts.size();
    const physx::PxU32 rowCount = contactCount * 3;
    physx::PxArray<AvbdPassiveMaterialComponentRow> rows(rowCount);
    physx::PxArray<double> rawAlImpulse(rowCount);
    physx::PxArray<double> poseVelocityMinusTarget(rowCount);
    physx::PxArray<double> q(rowCount);
    physx::PxArray<double> impulses(rowCount);
    physx::PxArray<double> nextImpulses(rowCount);
    physx::PxArray<double> gradient(rowCount);
    physx::PxArray<double> responseVelocity(rowCount);
    physx::PxArray<double> deltaImpulse(rowCount);
    physx::PxArray<double> deltaResponse(rowCount);
    physx::PxArray<double> friction(contactCount);
    physx::PxArray<double> scaledFriction(contactCount);
    physx::PxArray<double> scaledAlImpulse(rowCount);
    physx::PxArray<double> scaledBaseQ(rowCount);
    physx::PxArray<double> sqrtDiagonal(rowCount);
    physx::PxArray<double> inverseSqrtDiagonal(rowCount);
    physx::PxArray<double> fixedPointDelta(rowCount);
    physx::PxArray<double> currentResponse(rowCount);
    physx::PxArray<double> contactCertificate(contactCount);
    physx::PxArray<double> contactCertificateScale(contactCount);
    physx::PxArray<physx::PxU8> activeContacts(contactCount);
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot)
      activeContacts[contactSlot] = 0;
    bool finite = true;
    bool hasStaticEndpoint = false;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      const AvbdContactConstraint &contact =
          contacts[componentContacts[contactSlot]];
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      const bool dynamicA = bodyA < numBodies;
      const bool dynamicB = bodyB < numBodies;
      hasStaticEndpoint = hasStaticEndpoint || (dynamicA != dynamicB);
      const AvbdMaterialContactGeometry geometry =
          buildAvbdMaterialContactGeometry(
              contact, bodies, numBodies, invDt);
      physx::PxReal staticSign = 0.0f;
      if (dynamicA != dynamicB)
        staticSign = dynamicA ? -1.0f : 1.0f;
      const physx::PxVec3 axes[3] = {
          contact.contactNormal, contact.tangent0, contact.tangent1};
      physx::PxReal solveStartNormalVelocity = 0.0f;
      for (physx::PxU32 component = 0;
           component < 3; ++component) {
        AvbdPassiveMaterialComponentRow &row =
            rows[contactSlot * 3 + component];
        row.bodyA = bodyA;
        row.bodyB = bodyB;
        if (dynamicA) {
          row.linearA = axes[component];
          row.angularA = geometry.materialArmA.cross(axes[component]);
          row.positionAlAngularA =
              geometry.positionAlArmA.cross(axes[component]);
        }
        if (dynamicB) {
          row.linearB = -axes[component];
          row.angularB = geometry.materialArmB.cross(-axes[component]);
          row.positionAlAngularB =
              geometry.positionAlArmB.cross(-axes[component]);
        }
        physx::PxReal solveStartVelocity =
            staticSign * geometry.staticVelocity.dot(axes[component]);
        physx::PxReal poseVelocity = solveStartVelocity;
        if (dynamicA) {
          solveStartVelocity +=
              (*linearVelAtSolveStart)[bodyA].dot(axes[component]) +
              (*angularVelAtSolveStart)[bodyA].dot(
                  geometry.solveStartMaterialArmA.cross(axes[component]));
          poseVelocity +=
              bodies[bodyA].linearVelocity.dot(axes[component]) +
              bodies[bodyA].angularVelocity.dot(row.angularA);
        }
        if (dynamicB) {
          solveStartVelocity -=
              (*linearVelAtSolveStart)[bodyB].dot(axes[component]) +
              (*angularVelAtSolveStart)[bodyB].dot(
                  geometry.solveStartMaterialArmB.cross(axes[component]));
          poseVelocity +=
              bodies[bodyB].linearVelocity.dot(row.linearB) +
              bodies[bodyB].angularVelocity.dot(row.angularB);
        }
        if (component == 0u)
          solveStartNormalVelocity = solveStartVelocity;
        const physx::PxReal approach = -solveStartNormalVelocity;
        const physx::PxReal normalTarget =
            component == 0u && approach > effectiveBounceThreshold &&
                    approach > contact.detectionSeparation * invDt
                ? contact.restitution * approach
                : 0.0f;
        poseVelocityMinusTarget[contactSlot * 3u + component] =
            double(poseVelocity - normalTarget);
        finite = finite &&
                 row.linearA.isFinite() &&
                  row.angularA.isFinite() &&
                  row.linearB.isFinite() &&
                  row.angularB.isFinite() &&
                  row.positionAlAngularA.isFinite() &&
                  row.positionAlAngularB.isFinite() &&
                 physx::PxIsFinite(solveStartVelocity) &&
                 physx::PxIsFinite(poseVelocity);
      }
      rawAlImpulse[contactSlot * 3u] =
          double(physx::PxMax(0.0f, -contact.header.lambda) * dt);
      rawAlImpulse[contactSlot * 3u + 1u] =
          double(-contact.tangentLambda0 * dt);
      rawAlImpulse[contactSlot * 3u + 2u] =
          double(-contact.tangentLambda1 * dt);
      friction[contactSlot] = double(contactCoulombMu(contact));
      finite = finite && std::isfinite(friction[contactSlot]) &&
               friction[contactSlot] >= 0.0;
    }
    if (!finite)
      continue;

    const auto multiplyResponse =
        [&](const physx::PxArray<double> &input,
            physx::PxArray<double> &output) -> bool {
          for (physx::PxU32 bodySlot = 0;
               bodySlot < bodyQueue.size(); ++bodySlot) {
            const physx::PxU32 body = bodyQueue[bodySlot];
            bodyLinearImpulse[body] = physx::PxVec3(0.0f);
            bodyAngularImpulse[body] = physx::PxVec3(0.0f);
          }
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            const AvbdPassiveMaterialComponentRow &materialRow =
                rows[row];
            const physx::PxReal impulse = physx::PxReal(input[row]);
            if (!physx::PxIsFinite(impulse))
              return false;
            if (materialRow.bodyA < numBodies) {
              bodyLinearImpulse[materialRow.bodyA] +=
                  materialRow.linearA * impulse;
              bodyAngularImpulse[materialRow.bodyA] +=
                  materialRow.angularA * impulse;
            }
            if (materialRow.bodyB < numBodies) {
              bodyLinearImpulse[materialRow.bodyB] +=
                  materialRow.linearB * impulse;
              bodyAngularImpulse[materialRow.bodyB] +=
                  materialRow.angularB * impulse;
            }
          }
          for (physx::PxU32 bodySlot = 0;
               bodySlot < bodyQueue.size(); ++bodySlot) {
            const physx::PxU32 body = bodyQueue[bodySlot];
            bodyLinearDelta[body] =
                bodyLinearImpulse[body] * bodies[body].invMass;
            bodyAngularDelta[body] =
                bodies[body].invInertiaWorld.transform(
                    bodyAngularImpulse[body]);
            bodies[body].projectLockedLinearVector(bodyLinearDelta[body]);
            bodies[body].projectLockedAngularVector(bodyAngularDelta[body]);
            if (!bodyLinearDelta[body].isFinite() ||
                !bodyAngularDelta[body].isFinite())
              return false;
          }
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            const AvbdPassiveMaterialComponentRow &materialRow =
                rows[row];
            physx::PxReal value = 0.0f;
            if (materialRow.bodyA < numBodies) {
              value +=
                  bodyLinearDelta[materialRow.bodyA].dot(
                      materialRow.linearA) +
                  bodyAngularDelta[materialRow.bodyA].dot(
                      materialRow.angularA);
            }
            if (materialRow.bodyB < numBodies) {
              value +=
                  bodyLinearDelta[materialRow.bodyB].dot(
                      materialRow.linearB) +
                  bodyAngularDelta[materialRow.bodyB].dot(
                      materialRow.angularB);
            }
            output[row] = double(value);
            if (!std::isfinite(output[row]))
              return false;
          }
          return true;
        };

    // Map the PositionAL impulse of the active material frontier through its
    // persistent-anchor Jacobian, then observe that velocity change through
    // the fresh material Jacobian.  Dormant contacts retain their already
    // committed PositionAL response exactly; promoting a contact atomically
    // replaces only that contact's old Jacobian by its fresh material row.
    // When addToBodyDelta is true, append
    // M^-1 (J_material^T-J_AL^T) pAL for the final active frontier.
    const auto applyPositionAlJacobianReplacement =
        [&](physx::PxArray<double> *materialResponse,
            bool addToBodyDelta) -> bool {
          for (physx::PxU32 bodySlot = 0;
               bodySlot < bodyQueue.size(); ++bodySlot) {
            const physx::PxU32 body = bodyQueue[bodySlot];
            bodyLinearImpulse[body] = physx::PxVec3(0.0f);
            bodyAngularImpulse[body] = physx::PxVec3(0.0f);
          }
          for (physx::PxU32 rowIndex = 0; rowIndex < rowCount;
               ++rowIndex) {
            if (!activeContacts[rowIndex / 3u])
              continue;
            const AvbdPassiveMaterialComponentRow &row = rows[rowIndex];
            const physx::PxReal alImpulse =
                physx::PxReal(rawAlImpulse[rowIndex]);
            if (!physx::PxIsFinite(alImpulse))
              return false;
            if (row.bodyA < numBodies) {
              const physx::PxReal linearGain =
                  (*linearPoseVelocityGain)[row.bodyA];
              const physx::PxReal angularGain =
                  (*angularPoseVelocityGain)[row.bodyA];
              if (!physx::PxIsFinite(linearGain) ||
                  !physx::PxIsFinite(angularGain) ||
                  linearGain < 0.0f || angularGain < 0.0f)
                return false;
              bodyLinearImpulse[row.bodyA] +=
                  row.linearA * ((1.0f - linearGain) * alImpulse);
              bodyAngularImpulse[row.bodyA] +=
                  (row.angularA -
                   row.positionAlAngularA * angularGain) * alImpulse;
            }
            if (row.bodyB < numBodies) {
              const physx::PxReal linearGain =
                  (*linearPoseVelocityGain)[row.bodyB];
              const physx::PxReal angularGain =
                  (*angularPoseVelocityGain)[row.bodyB];
              if (!physx::PxIsFinite(linearGain) ||
                  !physx::PxIsFinite(angularGain) ||
                  linearGain < 0.0f || angularGain < 0.0f)
                return false;
              bodyLinearImpulse[row.bodyB] +=
                  row.linearB * ((1.0f - linearGain) * alImpulse);
              bodyAngularImpulse[row.bodyB] +=
                  (row.angularB -
                   row.positionAlAngularB * angularGain) * alImpulse;
            }
          }
          for (physx::PxU32 bodySlot = 0;
               bodySlot < bodyQueue.size(); ++bodySlot) {
            const physx::PxU32 body = bodyQueue[bodySlot];
            physx::PxVec3 linearDelta =
                bodyLinearImpulse[body] * bodies[body].invMass;
            physx::PxVec3 angularDelta =
                bodies[body].invInertiaWorld.transform(
                    bodyAngularImpulse[body]);
            bodies[body].projectLockedLinearVector(linearDelta);
            bodies[body].projectLockedAngularVector(angularDelta);
            if (!linearDelta.isFinite() || !angularDelta.isFinite())
              return false;
            if (addToBodyDelta) {
              bodyLinearDelta[body] += linearDelta;
              bodyAngularDelta[body] += angularDelta;
            } else {
              bodyLinearDelta[body] = linearDelta;
              bodyAngularDelta[body] = angularDelta;
            }
          }
          if (!materialResponse)
            return true;
          for (physx::PxU32 rowIndex = 0; rowIndex < rowCount;
               ++rowIndex) {
            const AvbdPassiveMaterialComponentRow &row = rows[rowIndex];
            physx::PxReal value = 0.0f;
            if (row.bodyA < numBodies)
              value += bodyLinearDelta[row.bodyA].dot(row.linearA) +
                       bodyAngularDelta[row.bodyA].dot(row.angularA);
            if (row.bodyB < numBodies)
              value += bodyLinearDelta[row.bodyB].dot(row.linearB) +
                       bodyAngularDelta[row.bodyB].dot(row.angularB);
            (*materialResponse)[rowIndex] = double(value);
            if (!std::isfinite((*materialResponse)[rowIndex]))
              return false;
          }
          return true;
        };

    // Solve in the AL correction d = p-pAL.  A dormant block has d=0 exactly
    // and retains its PositionAL response.  Once admitted, different material
    // and AL Jacobians give the velocity
    //   u = vPose-target-Jm M^-1 Jal^T pAL + Jm M^-1 Jm^T p
    //     = vPose-target + Jm M^-1 (Jm^T-Jal^T) pAL + Wm d.
    // Keeping correction coordinates preserves deep-stack conditioning.  The
    // replacement term is added only after the NCP map admits a block, so a
    // stable support row cannot become active merely because its detection
    // point migrated relative to its persistent friction anchor.
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      q[row] = poseVelocityMinusTarget[row];
      impulses[row] = 0.0;
      if (!std::isfinite(q[row]))
        finite = false;
    }
    if (!finite)
      continue;

    // Coulomb-geometry-preserving block equilibration. Tangent scale is the
    // mean of the two scalar self responses, which is invariant to rotations
    // of the tangent basis. With x=sqrt(D)d and a=sqrt(D)pAL, the physical
    // cone maps to the affine admissible set a+x in K(mu'),
    // mu'=mu*sqrt(dt/dn), while H=D^-1/2 W D^-1/2 stays matrix-free.
    double maximumDiagonal = 0.0;
    for (physx::PxU32 rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
      const AvbdPassiveMaterialComponentRow &row = rows[rowIndex];
      double diagonal = 0.0;
      const auto addDiagonal =
          [&](physx::PxU32 bodyIndex, const physx::PxVec3 &linear,
              const physx::PxVec3 &angular) {
            if (bodyIndex >= numBodies)
              return;
            physx::PxVec3 linearDelta =
                linear * bodies[bodyIndex].invMass;
            physx::PxVec3 angularDelta =
                bodies[bodyIndex].invInertiaWorld.transform(angular);
            bodies[bodyIndex].projectLockedLinearVector(linearDelta);
            bodies[bodyIndex].projectLockedAngularVector(angularDelta);
            diagonal += double(linear.dot(linearDelta) +
                               angular.dot(angularDelta));
          };
      addDiagonal(row.bodyA, row.linearA, row.angularA);
      addDiagonal(row.bodyB, row.linearB, row.angularB);
      if (!std::isfinite(diagonal) || diagonal < 0.0) {
        finite = false;
        break;
      }
      // The pose-minus-target RHS has already been folded into q, so reuse
      // that row scratch for the diagonal instead of allocating another
      // component-sized array.
      poseVelocityMinusTarget[rowIndex] = diagonal;
      maximumDiagonal = std::max(maximumDiagonal, diagonal);
    }
    if (!finite || maximumDiagonal <= 1.0e-12)
      continue;

    const double diagonalFloor =
        std::max(1.0e-12, 1.0e-8 * maximumDiagonal);
    double maximumScaledDiagonal = 0.0;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      const physx::PxU32 row = contactSlot * 3u;
      const double normalDiagonal =
          std::max(poseVelocityMinusTarget[row], diagonalFloor);
      const double tangentDiagonal = std::max(
          0.5 * (poseVelocityMinusTarget[row + 1u] +
                 poseVelocityMinusTarget[row + 2u]),
          diagonalFloor);
      const double normalScale = std::sqrt(normalDiagonal);
      const double tangentScale = std::sqrt(tangentDiagonal);
      sqrtDiagonal[row] = normalScale;
      sqrtDiagonal[row + 1u] = tangentScale;
      sqrtDiagonal[row + 2u] = tangentScale;
      inverseSqrtDiagonal[row] = 1.0 / normalScale;
      inverseSqrtDiagonal[row + 1u] = 1.0 / tangentScale;
      inverseSqrtDiagonal[row + 2u] = 1.0 / tangentScale;
      scaledFriction[contactSlot] =
          friction[contactSlot] * tangentScale / normalScale;
      finite = finite && std::isfinite(scaledFriction[contactSlot]);
    }
    for (physx::PxU32 row = 0; row < rowCount && finite; ++row) {
      maximumScaledDiagonal = std::max(
          maximumScaledDiagonal,
          poseVelocityMinusTarget[row] * inverseSqrtDiagonal[row] *
              inverseSqrtDiagonal[row]);
      q[row] *= inverseSqrtDiagonal[row];
      scaledBaseQ[row] = q[row];
      scaledAlImpulse[row] = rawAlImpulse[row] * sqrtDiagonal[row];
      impulses[row] *= sqrtDiagonal[row];
      finite = std::isfinite(q[row]) && std::isfinite(impulses[row]) &&
               std::isfinite(scaledAlImpulse[row]);
    }
    if (!finite || maximumScaledDiagonal <= 1.0e-12)
      continue;

    const auto projectScaledCorrection =
        [&](physx::PxU32 contactSlot, double &normalCorrection,
            double &tangent0Correction,
            double &tangent1Correction) -> bool {
          const physx::PxU32 row = contactSlot * 3u;
          double totalNormal =
              scaledAlImpulse[row] + normalCorrection;
          double totalTangent0 =
              scaledAlImpulse[row + 1u] + tangent0Correction;
          double totalTangent1 =
              scaledAlImpulse[row + 2u] + tangent1Correction;
          if (!projectAvbdCoulombNcpImpulse(
                  scaledFriction[contactSlot], totalNormal,
                  totalTangent0, totalTangent1))
            return false;
          normalCorrection = totalNormal - scaledAlImpulse[row];
          tangent0Correction =
              totalTangent0 - scaledAlImpulse[row + 1u];
          tangent1Correction =
              totalTangent1 - scaledAlImpulse[row + 2u];
          return std::isfinite(normalCorrection) &&
                 std::isfinite(tangent0Correction) &&
                 std::isfinite(tangent1Correction);
        };

    const auto multiplyScaledResponse =
        [&](const physx::PxArray<double> &input,
            physx::PxArray<double> &output) -> bool {
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            gradient[row] =
                input[row] * inverseSqrtDiagonal[row];
            if (!std::isfinite(gradient[row]))
              return false;
          }
          if (!multiplyResponse(gradient, output))
            return false;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            output[row] *= inverseSqrtDiagonal[row];
            if (!std::isfinite(output[row]))
              return false;
          }
          return true;
        };

    if (!multiplyScaledResponse(impulses, currentResponse))
      continue;
    const auto computePhysicalCertificate =
        [&](const physx::PxArray<double> &state,
            const physx::PxArray<double> &response,
            double lipschitz, bool activeOnly, double &residual,
            double &velocityScale) -> bool {
          if (!std::isfinite(lipschitz) || lipschitz <= 0.0)
            return false;
          const double step = 1.0 / lipschitz;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            gradient[row] = response[row] + q[row];
            deltaImpulse[row] = state[row] - step * gradient[row];
          }
          for (physx::PxU32 contactSlot = 0;
               contactSlot < contactCount; ++contactSlot) {
            const physx::PxU32 row = contactSlot * 3u;
            if (activeOnly && !activeContacts[contactSlot]) {
              deltaImpulse[row] = state[row];
              deltaImpulse[row + 1u] = state[row + 1u];
              deltaImpulse[row + 2u] = state[row + 2u];
            } else if (!projectScaledCorrection(
                           contactSlot, deltaImpulse[row],
                           deltaImpulse[row + 1u],
                           deltaImpulse[row + 2u])) {
              return false;
            }
          }
          residual = 0.0;
          velocityScale = 1.0;
          for (physx::PxU32 contactSlot = 0;
               contactSlot < contactCount; ++contactSlot) {
            const physx::PxU32 firstRow = contactSlot * 3u;
            double localResidual = 0.0;
            double localVelocityScale = 1.0;
            for (physx::PxU32 component = 0; component < 3u;
                 ++component) {
              const physx::PxU32 row = firstRow + component;
              localResidual = std::max(
                  localResidual,
                  sqrtDiagonal[row] * lipschitz *
                      std::fabs(state[row] - deltaImpulse[row]));
              localVelocityScale = std::max(
                  localVelocityScale,
                  sqrtDiagonal[row] *
                      std::max(
                          std::fabs(q[row]),
                          std::max(std::fabs(response[row]),
                                   std::fabs(gradient[row]))));
            }
            contactCertificate[contactSlot] = localResidual;
            contactCertificateScale[contactSlot] =
                localVelocityScale;
            if (!activeOnly || activeContacts[contactSlot]) {
              residual = std::max(residual, localResidual);
              velocityScale =
                  std::max(velocityScale, localVelocityScale);
            }
          }
          return std::isfinite(residual) &&
                 std::isfinite(velocityScale);
        };

    // Synchronized non-associated Coulomb fixed point. Every contact reads
    // the same component iterate. The normal trial is clamped independently,
    // then the tangent trial is projected onto its mu*normal disk. Thus the
    // converged map is Signorini plus maximum dissipation, without a PGS row
    // order and without the artificial sliding-to-normal coupling of an
    // associated second-order-cone QP.
    static const physx::PxU32 kMaxIterations = 512u;
    static const physx::PxU32 kMaxBacktracks = 32u;
    static const physx::PxU32 kMaxRelaxations = 12u;
    static const double kRelativeResidualTolerance = 2.0e-6;
    double lipschitz = maximumScaledDiagonal;
    double certificate = 0.0;
    double certificateScale = 1.0;
    if (!computePhysicalCertificate(
            impulses, currentResponse, lipschitz, false, certificate,
            certificateScale))
      continue;
    const double initialCertificate = certificate;
    double certificateAt8 = certificate;
    double certificateAt32 = certificate;
    double certificateAt128 = certificate;
    double certificateAt256 = certificate;
    physx::PxU32 iterationsUsed = 0;
    physx::PxU32 backtracksUsed = 0;
    physx::PxU32 relaxationsUsed = 0;
    physx::PxU32 activeContactCount = 0;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      if (contactCertificate[contactSlot] >
          kRelativeResidualTolerance *
              contactCertificateScale[contactSlot]) {
        activeContacts[contactSlot] = 1;
        ++activeContactCount;
      }
    }

    const auto refreshActiveReplacementRhs = [&]() -> bool {
      if (!applyPositionAlJacobianReplacement(&deltaResponse, false))
        return false;
      for (physx::PxU32 row = 0; row < rowCount; ++row) {
        q[row] = scaledBaseQ[row] +
                 deltaResponse[row] * inverseSqrtDiagonal[row];
        if (!std::isfinite(q[row]))
          return false;
      }
      return true;
    };

    // The full dynamic graph is structural; the solve frontier is physical.
    // Begin with contacts that violate the exact natural map at the current
    // pose/AL state.  After each synchronized solve, inspect every dormant
    // cut row through the same complete response operator and promote all
    // newly violated rows in stable contact order.  The mask grows
    // monotonically, static endpoints remain leaves, and no associative
    // lookup or contact-order PGS sweep is introduced.
    bool converged = false;
    physx::PxU32 frontierRounds = 0;
    for (physx::PxU32 frontierRound = 0;
         frontierRound <= contactCount && finite && !converged;
         ++frontierRound) {
      frontierRounds = frontierRound + 1u;
      if (activeContactCount > 0u) {
        if (!refreshActiveReplacementRhs()) {
          finite = false;
          break;
        }
        if (!computePhysicalCertificate(
                impulses, currentResponse, lipschitz, true,
                certificate, certificateScale)) {
          finite = false;
          break;
        }
        bool activeConverged =
            certificate <=
                kRelativeResidualTolerance * certificateScale;
        for (physx::PxU32 iteration = 0;
             iteration < kMaxIterations && finite &&
                 !activeConverged;
             ++iteration) {
          ++iterationsUsed;
          double trialL = lipschitz;
          bool accepted = false;
          for (physx::PxU32 backtrack = 0;
               backtrack < kMaxBacktracks; ++backtrack) {
            ++backtracksUsed;
            double currentCertificate = 0.0;
            double currentCertificateScale = 1.0;
            if (!computePhysicalCertificate(
                    impulses, currentResponse, trialL, true,
                    currentCertificate,
                    currentCertificateScale)) {
              finite = false;
              break;
            }
            double deltaNormSquared = 0.0;
            for (physx::PxU32 row = 0; row < rowCount; ++row) {
              // Inactive contacts are fixed by the dense frontier mask, so
              // their synchronized natural-map delta is exactly zero.
              fixedPointDelta[row] =
                  deltaImpulse[row] - impulses[row];
              deltaNormSquared +=
                  fixedPointDelta[row] * fixedPointDelta[row];
            }
            if (!std::isfinite(deltaNormSquared)) {
              finite = false;
              break;
            }
            if (deltaNormSquared == 0.0) {
              for (physx::PxU32 row = 0; row < rowCount; ++row)
                deltaResponse[row] = 0.0;
            } else if (!multiplyScaledResponse(
                           fixedPointDelta, deltaResponse)) {
              finite = false;
              break;
            }

            double directionalCurvature = 0.0;
            for (physx::PxU32 row = 0; row < rowCount; ++row)
              directionalCurvature +=
                  fixedPointDelta[row] * deltaResponse[row];
            const double majorization =
                trialL * deltaNormSquared;
            if (!std::isfinite(directionalCurvature) ||
                directionalCurvature >
                    majorization +
                        1.0e-10 *
                            std::max(1.0,
                                     std::fabs(majorization))) {
              trialL *= 2.0;
              if (!std::isfinite(trialL))
                finite = false;
              if (!finite)
                break;
              continue;
            }
            for (physx::PxU32 relaxation = 0;
                 relaxation < kMaxRelaxations; ++relaxation) {
              const double weight =
                  std::ldexp(1.0, -int(relaxation));
              if (relaxation > 0)
                ++relaxationsUsed;
              for (physx::PxU32 row = 0; row < rowCount; ++row) {
                nextImpulses[row] =
                    impulses[row] +
                    weight * fixedPointDelta[row];
                responseVelocity[row] =
                    currentResponse[row] +
                    weight * deltaResponse[row];
              }
              double nextCertificate = 0.0;
              double nextCertificateScale = 1.0;
              if (!computePhysicalCertificate(
                      nextImpulses, responseVelocity, trialL, true,
                      nextCertificate, nextCertificateScale)) {
                finite = false;
                break;
              }
              const double residualSlack =
                  1.0e-7 *
                  std::max(1.0, currentCertificate);
              if (nextCertificate <=
                      currentCertificate + residualSlack ||
                  nextCertificate <=
                      kRelativeResidualTolerance *
                          nextCertificateScale) {
                certificate = nextCertificate;
                certificateScale = nextCertificateScale;
                accepted = true;
                break;
              }
            }
            if (!finite || accepted)
              break;
            // The common denominator changes only the synchronized proximal
            // step; it cannot alter the NCP fixed point or add row order.
            trialL *= 2.0;
            if (!std::isfinite(trialL)) {
              finite = false;
              break;
            }
          }
          if (!finite || !accepted) {
            finite = false;
            break;
          }

          lipschitz = trialL;
          activeConverged =
              certificate <=
                  kRelativeResidualTolerance * certificateScale;
          if (iterationsUsed == 8u)
            certificateAt8 = certificate;
          else if (iterationsUsed == 32u)
            certificateAt32 = certificate;
          else if (iterationsUsed == 128u)
            certificateAt128 = certificate;
          else if (iterationsUsed == 256u)
            certificateAt256 = certificate;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            impulses[row] = nextImpulses[row];
            currentResponse[row] = responseVelocity[row];
          }
        }
        if (!finite || !activeConverged) {
          finite = false;
          break;
        }
      }

      if (!multiplyScaledResponse(impulses, currentResponse) ||
          !computePhysicalCertificate(
              impulses, currentResponse, lipschitz, false,
              certificate, certificateScale)) {
        finite = false;
        break;
      }
      physx::PxU32 promoted = 0;
      for (physx::PxU32 contactSlot = 0;
           contactSlot < contactCount; ++contactSlot) {
        if (!activeContacts[contactSlot] &&
            contactCertificate[contactSlot] >
                kRelativeResidualTolerance *
                    contactCertificateScale[contactSlot]) {
          activeContacts[contactSlot] = 1;
          ++promoted;
        }
      }
      activeContactCount += promoted;
      converged = promoted == 0u &&
                  certificate <=
                      kRelativeResidualTolerance * certificateScale;
    }
    if (std::getenv("PHYSX_AVBD_RIGID_COMPONENT_TRACE")) {
      std::printf(
          "[AVBD_RIGID_COMPONENT_NCP] contacts=%u active=%u "
          "frontierRounds=%u bodies=%u static=%u "
          "iterations=%u backtracks=%u relaxations=%u "
          "converged=%u finite=%u "
          "certificate=%.17g certificateScale=%.17g lipschitz=%.17g "
          "certificateInitial=%.17g certificate8=%.17g "
          "certificate32=%.17g certificate128=%.17g "
          "certificate256=%.17g\n",
          contactCount, activeContactCount, frontierRounds,
          bodyQueue.size(), hasStaticEndpoint ? 1u : 0u,
          iterationsUsed, backtracksUsed, relaxationsUsed,
          converged ? 1u : 0u, finite ? 1u : 0u,
          certificate, certificateScale, lipschitz,
          initialCertificate, certificateAt8, certificateAt32,
          certificateAt128, certificateAt256);
    }
    if (!finite || !converged)
      continue;

    // Re-evaluate the certified synchronized NCP map, then map both d and the
    // total p=(a+x)/sqrt(D) back to physical impulse units. Coulomb feasibility
    // is checked on total p before the atomic component commit.
    if (!multiplyScaledResponse(impulses, currentResponse) ||
        !computePhysicalCertificate(
            impulses, currentResponse, lipschitz, false, certificate,
            certificateScale) ||
        certificate > kRelativeResidualTolerance * certificateScale)
      continue;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      deltaImpulse[row] = impulses[row] * inverseSqrtDiagonal[row];
      impulses[row] =
          (scaledAlImpulse[row] + impulses[row]) *
          inverseSqrtDiagonal[row];
      if (!std::isfinite(deltaImpulse[row]) ||
          !std::isfinite(impulses[row]))
        finite = false;
    }
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount && finite; ++contactSlot) {
      const physx::PxU32 row = contactSlot * 3u;
      const double normal = impulses[row];
      const double tangentMagnitude = std::sqrt(
          impulses[row + 1u] * impulses[row + 1u] +
          impulses[row + 2u] * impulses[row + 2u]);
      const double coneLimit =
          friction[contactSlot] * std::max(0.0, normal);
      const double coneTolerance =
          1.0e-9 *
          std::max(1.0,
                   std::max(std::fabs(normal), tangentMagnitude));
      if (!std::isfinite(tangentMagnitude) ||
          !std::isfinite(coneLimit) || normal < -coneTolerance ||
          tangentMagnitude > coneLimit + coneTolerance)
        finite = false;
    }
    if (!finite || !multiplyResponse(deltaImpulse, deltaResponse) ||
        !applyPositionAlJacobianReplacement(nullptr, true))
      continue;

    // A static endpoint supplies an external reaction, so this component has
    // no closed six-dimensional momentum nullspace to restore.  Validate the
    // complete prospective state first, then commit every dynamic body and
    // report row atomically.  The NCP above still includes all dynamic-static
    // and dynamic-dynamic cross response through the shared dynamic bodies.
    if (hasStaticEndpoint) {
      for (physx::PxU32 bodySlot = 0;
           bodySlot < bodyQueue.size(); ++bodySlot) {
        const physx::PxU32 body = bodyQueue[bodySlot];
        finite = finite &&
                 (bodies[body].linearVelocity +
                  bodyLinearDelta[body]).isFinite() &&
                 (bodies[body].angularVelocity +
                  bodyAngularDelta[body]).isFinite();
      }
      for (physx::PxU32 contactSlot = 0;
           contactSlot < contactCount && finite; ++contactSlot) {
        const physx::PxU32 row = contactSlot * 3u;
        finite =
            physx::PxIsFinite(
                static_cast<physx::PxReal>(impulses[row])) &&
            physx::PxIsFinite(
                static_cast<physx::PxReal>(impulses[row + 1u])) &&
            physx::PxIsFinite(
                static_cast<physx::PxReal>(impulses[row + 2u]));
      }
      if (!finite)
        continue;
      for (physx::PxU32 bodySlot = 0;
           bodySlot < bodyQueue.size(); ++bodySlot) {
        const physx::PxU32 body = bodyQueue[bodySlot];
        bodies[body].linearVelocity += bodyLinearDelta[body];
        bodies[body].angularVelocity += bodyAngularDelta[body];
      }
      for (physx::PxU32 contactSlot = 0;
           contactSlot < contactCount; ++contactSlot) {
        AvbdContactConstraint &contact =
            contacts[componentContacts[contactSlot]];
        const physx::PxU32 row = contactSlot * 3u;
        const physx::PxReal normalImpulse =
            static_cast<physx::PxReal>(impulses[row]);
        const physx::PxReal tangent0 =
            static_cast<physx::PxReal>(impulses[row + 1u]);
        const physx::PxReal tangent1 =
            static_cast<physx::PxReal>(impulses[row + 2u]);
        contact.velocityNormalImpulse = normalImpulse;
        contact.frictionSweepImpulse =
            contact.tangent0 * tangent0 +
            contact.tangent1 * tangent1;
      }
      continue;
    }

    // A closed all-dynamic contact graph has a six-dimensional rigid-twist
    // nullspace.  Fresh material rows apply equal/opposite impulses at one
    // common world point, so neither total linear nor total angular momentum
    // can change.  Pose-derived velocity can nevertheless retain an
    // actor-order-dependent rigid translation and rotation after nonlinear
    // body-coordinate descent.  Restore both invariants with the unique
    // mass-metric rigid twist; J times that twist is zero, so this cannot
    // alter the certified contact NCP.
    std::sort(bodyQueue.begin(), bodyQueue.end());
    double totalMass = 0.0;
    AvbdDoubleVec3 massWeightedCenter;
    for (physx::PxU32 bodySlot = 0;
         bodySlot < bodyQueue.size(); ++bodySlot) {
      const physx::PxU32 bodyIndex = bodyQueue[bodySlot];
      const AvbdSolverBody &body = bodies[bodyIndex];
      if (body.invMass <= 0.0f || body.lockFlags != 0 ||
          !body.position.isFinite()) {
        finite = false;
        break;
      }
      const double mass = 1.0 / double(body.invMass);
      if (!std::isfinite(mass) || mass <= 0.0) {
        finite = false;
        break;
      }
      totalMass += mass;
      massWeightedCenter += AvbdDoubleVec3(body.position) * mass;
    }
    if (!finite || !std::isfinite(totalMass) || totalMass <= 0.0)
      continue;
    const AvbdDoubleVec3 componentCenter =
        massWeightedCenter * (1.0 / totalMass);
    if (!componentCenter.isFinite())
      continue;

    AvbdDoubleVec3 desiredMomentum;
    AvbdDoubleVec3 prospectiveMomentum;
    AvbdDoubleVec3 desiredAngularMomentum;
    AvbdDoubleVec3 prospectiveAngularMomentum;
    AvbdDoubleSymmetric3 componentInertia;
    bool angularMomentumAvailable = true;
    for (physx::PxU32 bodySlot = 0;
         bodySlot < bodyQueue.size(); ++bodySlot) {
      const physx::PxU32 bodyIndex = bodyQueue[bodySlot];
      const AvbdSolverBody &body = bodies[bodyIndex];
      const double mass = 1.0 / double(body.invMass);

      // Preserve the external/inertial prediction, not the raw solve-start
      // velocity.  The latter omits gravity and also bypasses the normalized
      // quaternion integration used by computePrediction; restoring momentum
      // to that stale state would remove an external impulse and can inject
      // angular energy on every dynamic-dynamic contact frame.  Reconstruct
      // the same pose-derived baseline as the post-AL velocity stage, then
      // apply the public per-body damping and caps once.
      physx::PxVec3 desiredLinear =
          (body.inertialPosition - body.prevPosition) * invDt;
      if (body.linearDamping > 0.0f)
        desiredLinear *= 1.0f / (1.0f + body.linearDamping * dt);
      const physx::PxReal desiredLinearSpeedSq =
          desiredLinear.magnitudeSquared();
      if (desiredLinearSpeedSq > body.maxLinearVelocitySq &&
          body.maxLinearVelocitySq > 0.0f)
        desiredLinear *= physx::PxSqrt(
            body.maxLinearVelocitySq / desiredLinearSpeedSq);
      body.projectLockedLinearVector(desiredLinear);

      physx::PxQuat desiredRotationDelta =
          body.inertialRotation * body.prevRotation.getConjugate();
      if (desiredRotationDelta.w < 0.0f)
        desiredRotationDelta = -desiredRotationDelta;
      physx::PxVec3 desiredAngular(
          desiredRotationDelta.x, desiredRotationDelta.y,
          desiredRotationDelta.z);
      desiredAngular *= 2.0f * invDt;
      if (body.angularDampingBody > 0.0f)
        desiredAngular *=
            1.0f / (1.0f + body.angularDampingBody * dt);
      const physx::PxReal desiredAngularSpeedSq =
          desiredAngular.magnitudeSquared();
      if (desiredAngularSpeedSq > body.maxAngularVelocitySq &&
          body.maxAngularVelocitySq > 0.0f)
        desiredAngular *= physx::PxSqrt(
            body.maxAngularVelocitySq / desiredAngularSpeedSq);
      body.projectLockedAngularVector(desiredAngular);

      const physx::PxVec3 prospectiveLinear =
          body.linearVelocity + bodyLinearDelta[bodyIndex];
      const physx::PxVec3 prospectiveAngular =
          body.angularVelocity + bodyAngularDelta[bodyIndex];
      if (!desiredLinear.isFinite() || !desiredAngular.isFinite() ||
          !prospectiveLinear.isFinite() ||
          !prospectiveAngular.isFinite()) {
        finite = false;
        break;
      }

      const AvbdDoubleVec3 desiredLinearDouble(desiredLinear);
      const AvbdDoubleVec3 prospectiveLinearDouble(prospectiveLinear);
      desiredMomentum += desiredLinearDouble * mass;
      prospectiveMomentum += prospectiveLinearDouble * mass;

      if (angularMomentumAvailable) {
        AvbdDoubleSymmetric3 bodyInertia;
        if (!invertAvbdWorldInverseInertia(
                body.invInertiaWorld, bodyInertia)) {
          angularMomentumAvailable = false;
          continue;
        }
        const AvbdDoubleVec3 arm =
            AvbdDoubleVec3(body.position) - componentCenter;
        const double armValues[3] = {arm.x, arm.y, arm.z};
        const double armMagnitudeSq =
            arm.x * arm.x + arm.y * arm.y + arm.z * arm.z;
        for (physx::PxU32 row = 0; row < 3; ++row) {
          for (physx::PxU32 column = 0; column < 3; ++column) {
            componentInertia.value[row][column] +=
                bodyInertia.value[row][column] +
                mass * ((row == column ? armMagnitudeSq : 0.0) -
                        armValues[row] * armValues[column]);
          }
        }
        desiredAngularMomentum +=
            bodyInertia.multiply(AvbdDoubleVec3(desiredAngular)) +
            crossAvbdDouble(arm, desiredLinearDouble * mass);
        prospectiveAngularMomentum +=
            bodyInertia.multiply(AvbdDoubleVec3(prospectiveAngular)) +
            crossAvbdDouble(arm, prospectiveLinearDouble * mass);
      }
    }
    if (!finite || !desiredMomentum.isFinite() ||
        !prospectiveMomentum.isFinite())
      continue;

    const AvbdDoubleVec3 momentumError =
        desiredMomentum - prospectiveMomentum;
    const AvbdDoubleVec3 nullspaceTranslation =
        momentumError * (1.0 / totalMass);
    AvbdDoubleVec3 angularMomentumError;
    AvbdDoubleVec3 nullspaceRotation;
    bool angularMomentumRestored = false;
    if (angularMomentumAvailable && componentInertia.isFinite() &&
        desiredAngularMomentum.isFinite() &&
        prospectiveAngularMomentum.isFinite()) {
      angularMomentumError =
          desiredAngularMomentum - prospectiveAngularMomentum;
      AvbdDoubleCholesky3 componentInertiaFactor;
      angularMomentumRestored =
          factorAvbdDoubleSpd3(
              componentInertia, componentInertiaFactor) &&
          solveAvbdDoubleSpd3(
              componentInertiaFactor, angularMomentumError,
              nullspaceRotation);
    }
    if (!angularMomentumRestored)
      nullspaceRotation = AvbdDoubleVec3();
    if (!nullspaceTranslation.isFinite() ||
        !nullspaceRotation.isFinite())
      continue;

    const auto bodyTwistLinearCorrection =
        [&](physx::PxU32 bodyIndex) -> physx::PxVec3 {
          const AvbdDoubleVec3 arm =
              AvbdDoubleVec3(bodies[bodyIndex].position) -
              componentCenter;
          const AvbdDoubleVec3 correction =
              nullspaceTranslation +
              crossAvbdDouble(nullspaceRotation, arm);
          return physx::PxVec3(
              static_cast<physx::PxReal>(correction.x),
              static_cast<physx::PxReal>(correction.y),
              static_cast<physx::PxReal>(correction.z));
        };
    const auto computeTwistNullspaceCertificate =
        [&](double &certificateScale) -> double {
          double maximumResidual = 0.0;
          certificateScale = 1.0;
          const physx::PxVec3 angularCorrection(
              static_cast<physx::PxReal>(nullspaceRotation.x),
              static_cast<physx::PxReal>(nullspaceRotation.y),
              static_cast<physx::PxReal>(nullspaceRotation.z));
          for (physx::PxU32 rowIndex = 0; rowIndex < rowCount;
               ++rowIndex) {
            const AvbdPassiveMaterialComponentRow &row = rows[rowIndex];
            const physx::PxVec3 linearA =
                bodyTwistLinearCorrection(row.bodyA);
            const physx::PxVec3 linearB =
                bodyTwistLinearCorrection(row.bodyB);
            const double contributionA =
                double(linearA.dot(row.linearA) +
                       angularCorrection.dot(row.angularA));
            const double contributionB =
                double(linearB.dot(row.linearB) +
                       angularCorrection.dot(row.angularB));
            maximumResidual = std::max(
                maximumResidual,
                std::fabs(contributionA + contributionB));
            certificateScale = std::max(
                certificateScale,
                std::fabs(contributionA) + std::fabs(contributionB));
          }
          return maximumResidual;
        };

    double nullspaceCertificateScale = 1.0;
    double nullspaceCertificate =
        computeTwistNullspaceCertificate(nullspaceCertificateScale);
    const double nullspaceTolerance =
        128.0 * double(std::numeric_limits<physx::PxReal>::epsilon()) *
        nullspaceCertificateScale;
    if (angularMomentumRestored &&
        (!std::isfinite(nullspaceCertificate) ||
         nullspaceCertificate > nullspaceTolerance)) {
      // A malformed/non-common contact arm must never be hidden by changing
      // relative material velocity.  Retain the exact uniform translation
      // nullspace and safely decline the angular restoration for this
      // component.
      angularMomentumRestored = false;
      nullspaceRotation = AvbdDoubleVec3();
      nullspaceCertificate =
          computeTwistNullspaceCertificate(nullspaceCertificateScale);
    }
    if (!std::isfinite(nullspaceCertificate) ||
        nullspaceCertificate >
            128.0 *
                double(std::numeric_limits<physx::PxReal>::epsilon()) *
                nullspaceCertificateScale)
      continue;
    PX_ASSERT(
        nullspaceCertificate <=
        128.0 * double(std::numeric_limits<physx::PxReal>::epsilon()) *
            nullspaceCertificateScale);

    AvbdDoubleVec3 correctedMomentum = prospectiveMomentum;
    AvbdDoubleVec3 correctedAngularMomentum =
        prospectiveAngularMomentum;
    const physx::PxVec3 angularCorrection(
        static_cast<physx::PxReal>(nullspaceRotation.x),
        static_cast<physx::PxReal>(nullspaceRotation.y),
        static_cast<physx::PxReal>(nullspaceRotation.z));
    for (physx::PxU32 bodySlot = 0;
         bodySlot < bodyQueue.size(); ++bodySlot) {
      const physx::PxU32 bodyIndex = bodyQueue[bodySlot];
      const physx::PxVec3 linearCorrection =
          bodyTwistLinearCorrection(bodyIndex);
      if (!linearCorrection.isFinite() ||
          !angularCorrection.isFinite() ||
          !(bodies[bodyIndex].linearVelocity +
            bodyLinearDelta[bodyIndex] + linearCorrection).isFinite() ||
          !(bodies[bodyIndex].angularVelocity +
            bodyAngularDelta[bodyIndex] + angularCorrection).isFinite()) {
        finite = false;
        break;
      }
      const double mass = 1.0 / double(bodies[bodyIndex].invMass);
      correctedMomentum += AvbdDoubleVec3(linearCorrection) * mass;
      if (angularMomentumAvailable) {
        AvbdDoubleSymmetric3 bodyInertia;
        if (!invertAvbdWorldInverseInertia(
                bodies[bodyIndex].invInertiaWorld, bodyInertia)) {
          angularMomentumAvailable = false;
        } else {
          const AvbdDoubleVec3 arm =
              AvbdDoubleVec3(bodies[bodyIndex].position) -
              componentCenter;
          correctedAngularMomentum +=
              bodyInertia.multiply(AvbdDoubleVec3(angularCorrection)) +
              crossAvbdDouble(
                  arm, AvbdDoubleVec3(linearCorrection) * mass);
        }
      }
    }
    if (!finite)
      continue;

    if (std::getenv("PHYSX_AVBD_RIGID_COMPONENT_TRACE")) {
      const double angularErrorBefore =
          angularMomentumAvailable
              ? magnitudeAvbdDouble(angularMomentumError)
              : -1.0;
      const double angularErrorAfter =
          angularMomentumAvailable
              ? magnitudeAvbdDouble(
                    desiredAngularMomentum -
                    correctedAngularMomentum)
              : -1.0;
      std::printf(
          "[AVBD_RIGID_COMPONENT_TWIST] contacts=%u bodies=%u "
          "angularAvailable=%u angularRestored=%u mass=%.17g "
          "momentumBefore=%.17g momentumAfter=%.17g "
          "angularBefore=%.17g angularAfter=%.17g "
          "translationCorrection=%.17g rotationCorrection=%.17g "
          "nullspaceCertificate=%.17g nullspaceScale=%.17g\n",
          contactCount, bodyQueue.size(),
          angularMomentumAvailable ? 1u : 0u,
          angularMomentumRestored ? 1u : 0u, totalMass,
          magnitudeAvbdDouble(momentumError),
          magnitudeAvbdDouble(desiredMomentum - correctedMomentum),
          angularErrorBefore, angularErrorAfter,
          magnitudeAvbdDouble(nullspaceTranslation),
          magnitudeAvbdDouble(nullspaceRotation),
          nullspaceCertificate, nullspaceCertificateScale);
    }

    for (physx::PxU32 bodySlot = 0;
         bodySlot < bodyQueue.size(); ++bodySlot) {
      const physx::PxU32 body = bodyQueue[bodySlot];
      bodies[body].linearVelocity +=
          bodyLinearDelta[body] + bodyTwistLinearCorrection(body);
      bodies[body].angularVelocity +=
          bodyAngularDelta[body] + angularCorrection;
    }
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      AvbdContactConstraint &contact =
          contacts[componentContacts[contactSlot]];
      const physx::PxU32 row = contactSlot * 3;
      const physx::PxReal normalImpulse =
          static_cast<physx::PxReal>(impulses[row]);
      const physx::PxReal tangent0 =
          static_cast<physx::PxReal>(impulses[row + 1]);
      const physx::PxReal tangent1 =
          static_cast<physx::PxReal>(impulses[row + 2]);
      // Velocity material response is report state, never AL dual state.
      // Keeping header.lambda intact preserves the position solve's warmstart
      // for the next frame.
      contact.velocityNormalImpulse = normalImpulse;
      contact.frictionSweepImpulse =
          contact.tangent0 * tangent0 +
          contact.tangent1 * tangent1;
    }
  }
}

/**
 * Project a strict one-to-four-point rigid-static contact manifold as one
 * total-impulse material objective.  All normal and tangent rows read the same
 * NCP iterate, and state is committed only after the complete cone block has
 * a finite fixed-point certificate.
 */
static void applyAvbdContactMaterialFrictionManifolds(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    const AvbdPostAlContactWorkPlan *workPlan) {
  if (!bodies || !contacts || dt <= 0.0f)
    return;

  applyAvbdPassiveFrictionComponents(
      bodies, numBodies, contacts, numContacts,
      contactMap, linearVelAtSolveStart, angularVelAtSolveStart,
      linearPoseVelocityGain, angularPoseVelocityGain,
      dt, bounceThreshold, workPlan);

  if (!mayHavePostAlContactWork(
          workPlan, AvbdPostAlContactWorkPlan::eCOMPLETE_MANIFOLD))
    return;

  physx::PxArray<physx::PxU8> visitedManifoldRows(numContacts);
  for (physx::PxU32 c = 0; c < numContacts; ++c)
    visitedManifoldRows[c] = 0;
  for (physx::PxU32 seed = 0; seed < numContacts; ++seed) {
    const AvbdCompiledVelocityObjective *seedObjective =
        findAvbdCompleteManifoldObjective(
            contacts[seed].objectiveProgram);
    if (visitedManifoldRows[seed] ||
        hasRigidMaterialConsumed(contacts[seed]) || !seedObjective)
      continue;
    if (!isBodyVsStaticContact(
            contacts[seed].header.bodyIndexA,
            contacts[seed].header.bodyIndexB, numBodies))
      continue;
    const physx::PxU32 bodyIndex =
        contacts[seed].header.bodyIndexA < numBodies
            ? contacts[seed].header.bodyIndexA
            : contacts[seed].header.bodyIndexB;
    if (bodyIndex >= numBodies)
      continue;
    AvbdSolverBody &body = bodies[bodyIndex];
    if (body.invMass <= 0.0f)
      continue;
    if (!linearPoseVelocityGain || !angularPoseVelocityGain ||
        linearPoseVelocityGain->size() != numBodies ||
        angularPoseVelocityGain->size() != numBodies)
      continue;
    const physx::PxReal linearGain =
        (*linearPoseVelocityGain)[bodyIndex];
    const physx::PxReal angularGain =
        (*angularPoseVelocityGain)[bodyIndex];
    if (!physx::PxIsFinite(linearGain) ||
        !physx::PxIsFinite(angularGain) ||
        linearGain < 0.0f || angularGain < 0.0f)
      continue;

    physx::PxU32 contactIndices[4] = {};
    physx::PxU32 contactCount = 0;
    const physx::PxU64 objectiveKey = seedObjective->objectiveKey;
    bool supportedGroup = true;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const AvbdContactConstraint &contact = contacts[c];
      if (hasRigidMaterialConsumed(contact))
        continue;
      const AvbdCompiledVelocityObjective *objective =
          findAvbdCompleteManifoldObjective(
              contact.objectiveProgram);
      if (!objective || objective->objectiveKey != objectiveKey)
        continue;
      visitedManifoldRows[c] = 1;
      if (contact.header.bodyIndexA != bodyIndex &&
          contact.header.bodyIndexB != bodyIndex)
        supportedGroup = false;
      if (contactCount < 4)
        contactIndices[contactCount] = c;
      ++contactCount;
    }
    if (!supportedGroup || contactCount == 0 || contactCount > 4 ||
        contactCount != seedObjective->objectiveRowCount)
      continue;

    // PositionAL retains the per-contact geometry rows, while material
    // friction follows PhysX's patch model: every contact contributes one
    // normal row, but a patch contributes at most two persistent tangent
    // anchors sharing the patch's total normal load.  Keeping these layouts
    // separate is essential for spinning face contacts; four independent
    // point cones cancel their translational forces and make sliding appear
    // frictionless even though every point is saturated.
    physx::PxU32 contactPatch[4] = {};
    physx::PxU32 patchManager[4] = {};
    physx::PxU32 patchOrdinal[4] = {};
    physx::PxU32 patchAnchorCounts[4] = {};
    physx::PxU32 anchorContactSlots[4] = {};
    physx::PxU32 anchorPatches[4] = {};
    double patchFriction[4] = {};
    physx::PxU32 patchCount = 0u;
    physx::PxU32 anchorCount = 0u;
    bool finite = true;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount && finite; ++contactSlot) {
      const AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      physx::PxU32 patch = 0u;
      for (; patch < patchCount; ++patch) {
        if (patchManager[patch] == contact.contactManagerIndex &&
            patchOrdinal[patch] == contact.contactPatchIndex)
          break;
      }
      const double friction = double(contactCoulombMu(contact));
      if (!std::isfinite(friction) || friction < 0.0 ||
          contact.frictionAnchorCount > 2u) {
        finite = false;
        break;
      }
      if (patch == patchCount) {
        if (patchCount >= 4u) {
          finite = false;
          break;
        }
        patchManager[patch] = contact.contactManagerIndex;
        patchOrdinal[patch] = contact.contactPatchIndex;
        patchFriction[patch] = friction;
        patchAnchorCounts[patch] = contact.frictionAnchorCount;
        ++patchCount;
      } else if (std::fabs(patchFriction[patch] - friction) > 1.0e-9 ||
                 patchAnchorCounts[patch] !=
                     contact.frictionAnchorCount) {
        finite = false;
        break;
      }
      contactPatch[contactSlot] = patch;
      const physx::PxU8 anchorMask = contact.frictionAnchorMask;
      for (physx::PxU32 bit = 0u; bit < 2u; ++bit) {
        if ((anchorMask & (1u << bit)) == 0u)
          continue;
        if (anchorCount >= 4u) {
          finite = false;
          break;
        }
        for (physx::PxU32 prior = 0u; prior < anchorCount; ++prior) {
          if (anchorPatches[prior] == patch &&
              anchorContactSlots[prior] == contactSlot) {
            finite = false;
            break;
          }
        }
        if (!finite)
          break;
        anchorContactSlots[anchorCount] = contactSlot;
        anchorPatches[anchorCount] = patch;
        ++anchorCount;
      }
    }
    for (physx::PxU32 patch = 0u;
         patch < patchCount && finite; ++patch) {
      physx::PxU32 actualAnchorCount = 0u;
      for (physx::PxU32 anchor = 0u; anchor < anchorCount; ++anchor)
        actualAnchorCount += anchorPatches[anchor] == patch ? 1u : 0u;
      finite = actualAnchorCount == patchAnchorCounts[patch];
    }
    if (!finite || patchCount == 0u)
      continue;

    // Assign every geometric normal point to its nearest material anchor.
    // This is the same patch-pressure partition used by contact reporting:
    // it retains an asymmetric support load while still limiting velocity
    // friction to the patch's one or two actual anchors.
    physx::PxU32 solverContactPatch[4] = {};
    physx::PxU32 solverAnchorPatch[4] = {};
    physx::PxU32 solverPatchAnchorCounts[4] = {};
    double solverPatchFriction[4] = {};
    physx::PxVec3 freshMaterialPoints[4];
    for (physx::PxU32 contactSlot = 0u;
         contactSlot < contactCount; ++contactSlot) {
      const AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      const physx::PxVec3 freshA = getAvbdFreshEndpointWorldPoint(
          contact, bodies, numBodies, true, false);
      const physx::PxVec3 freshB = getAvbdFreshEndpointWorldPoint(
          contact, bodies, numBodies, false, false);
      freshMaterialPoints[contactSlot] = (freshA + freshB) * 0.5f;
      finite = finite && freshMaterialPoints[contactSlot].isFinite();
    }
    physx::PxU32 solverPatchCount = 0u;
    for (physx::PxU32 patch = 0u;
         patch < patchCount && finite; ++patch) {
      if (patchAnchorCounts[patch] == 0u) {
        if (solverPatchCount >= 4u) {
          finite = false;
          break;
        }
        const physx::PxU32 solverPatch = solverPatchCount++;
        solverPatchAnchorCounts[solverPatch] =
            patchAnchorCounts[patch];
        solverPatchFriction[solverPatch] = patchFriction[patch];
        for (physx::PxU32 contactSlot = 0u;
             contactSlot < contactCount; ++contactSlot)
          if (contactPatch[contactSlot] == patch)
            solverContactPatch[contactSlot] = solverPatch;
        continue;
      }

      for (physx::PxU32 anchor = 0u; anchor < anchorCount; ++anchor) {
        if (anchorPatches[anchor] != patch)
          continue;
        if (solverPatchCount >= 4u) {
          finite = false;
          break;
        }
        const physx::PxU32 solverPatch = solverPatchCount++;
        solverAnchorPatch[anchor] = solverPatch;
        solverPatchAnchorCounts[solverPatch] = 1u;
        solverPatchFriction[solverPatch] = patchFriction[patch];
      }
      for (physx::PxU32 contactSlot = 0u;
           contactSlot < contactCount && finite; ++contactSlot) {
        if (contactPatch[contactSlot] != patch)
          continue;
        physx::PxU32 nearestAnchor = PX_MAX_U32;
        physx::PxReal nearestDistanceSquared = PX_MAX_F32;
        for (physx::PxU32 anchor = 0u; anchor < anchorCount; ++anchor) {
          if (anchorPatches[anchor] != patch)
            continue;
          const physx::PxReal distanceSquared =
              (freshMaterialPoints[contactSlot] -
               freshMaterialPoints[anchorContactSlots[anchor]])
                  .magnitudeSquared();
          if (distanceSquared < nearestDistanceSquared) {
            nearestDistanceSquared = distanceSquared;
            nearestAnchor = anchor;
          }
        }
        if (nearestAnchor == PX_MAX_U32 ||
            !physx::PxIsFinite(nearestDistanceSquared)) {
          finite = false;
          break;
        }
        solverContactPatch[contactSlot] =
            solverAnchorPatch[nearestAnchor];
      }
    }
    if (!finite || solverPatchCount == 0u)
      continue;

    const physx::PxU32 rawRowCount = contactCount * 3u;
    const physx::PxU32 rowCount = contactCount + anchorCount * 2u;
    physx::PxVec3 materialArms[4];
    physx::PxVec3 pointVelocities[4];
    physx::PxVec3 rawAxes[12];
    physx::PxVec3 rawPositionAlAngularJacobians[12];
    physx::PxVec3 axes[12];
    physx::PxVec3 materialAngularJacobians[12];
    physx::PxU32 normalRows[4] = {};
    physx::PxU32 tangentRows[4] = {};
    double response[12u * 12u] = {};
    double q[12] = {};
    double rawAlImpulse[12] = {};
    double impulses[12] = {};
    double scratch[12] = {};

    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount && finite; ++contactSlot) {
      const AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      const bool dynamicIsA =
          contact.header.bodyIndexA == bodyIndex;
      const physx::PxReal dynamicSign = dynamicIsA ? 1.0f : -1.0f;
      const AvbdMaterialContactGeometry geometry =
          buildAvbdMaterialContactGeometry(
              contact, bodies, numBodies, 1.0f / dt);
      materialArms[contactSlot] =
          dynamicIsA ? geometry.materialArmA : geometry.materialArmB;
      const physx::PxVec3 positionAlArm =
          dynamicIsA ? geometry.positionAlArmA : geometry.positionAlArmB;
      pointVelocities[contactSlot] =
          body.linearVelocity +
          body.angularVelocity.cross(materialArms[contactSlot]) -
          geometry.staticVelocity;
      const physx::PxVec3 contactAxes[3] = {
          contact.contactNormal, contact.tangent0, contact.tangent1};
      for (physx::PxU32 component = 0u; component < 3u; ++component) {
        const physx::PxU32 rawRow = contactSlot * 3u + component;
        rawAxes[rawRow] = contactAxes[component] * dynamicSign;
        rawPositionAlAngularJacobians[rawRow] =
            positionAlArm.cross(rawAxes[rawRow]);
        finite = finite && rawAxes[rawRow].isFinite() &&
                 rawPositionAlAngularJacobians[rawRow].isFinite();
      }
      const physx::PxU32 rawRow = contactSlot * 3u;
      rawAlImpulse[rawRow] =
          double(physx::PxMax(0.0f, -contact.header.lambda) * dt);
      rawAlImpulse[rawRow + 1u] =
          double(-contact.tangentLambda0 * dt);
      rawAlImpulse[rawRow + 2u] =
          double(-contact.tangentLambda1 * dt);

      const physx::PxU32 normalRow = contactSlot;
      normalRows[contactSlot] = normalRow;
      axes[normalRow] = rawAxes[rawRow];
      materialAngularJacobians[normalRow] =
          materialArms[contactSlot].cross(axes[normalRow]);
      q[normalRow] = double(
          pointVelocities[contactSlot].dot(axes[normalRow]) -
          contact.targetVelocity.dot(contact.contactNormal));
      impulses[normalRow] = rawAlImpulse[rawRow];
      finite = finite && axes[normalRow].isFinite() &&
               materialAngularJacobians[normalRow].isFinite() &&
               std::isfinite(q[normalRow]) &&
               std::isfinite(impulses[normalRow]);
    }
    for (physx::PxU32 anchor = 0u;
         anchor < anchorCount && finite; ++anchor) {
      const physx::PxU32 contactSlot = anchorContactSlots[anchor];
      const AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      const physx::PxU32 rawRow = contactSlot * 3u;
      const physx::PxU32 tangentRow = contactCount + anchor * 2u;
      tangentRows[anchor] = tangentRow;
      for (physx::PxU32 tangent = 0u; tangent < 2u; ++tangent) {
        const physx::PxU32 row = tangentRow + tangent;
        const physx::PxU32 rawTangentRow = rawRow + tangent + 1u;
        axes[row] = rawAxes[rawTangentRow];
        materialAngularJacobians[row] =
            materialArms[contactSlot].cross(axes[row]);
        q[row] = double(
            pointVelocities[contactSlot].dot(axes[row]) -
            contact.targetVelocity.dot(
                tangent == 0u ? contact.tangent0 : contact.tangent1));
        impulses[row] = rawAlImpulse[rawTangentRow];
        finite = finite && axes[row].isFinite() &&
                 materialAngularJacobians[row].isFinite() &&
                 std::isfinite(q[row]) && std::isfinite(impulses[row]);
      }
    }

    // Remove every position-level AL row from the physical material
    // baseline, including non-anchor tangents.  The new patch rows are then
    // assembled at fresh material points and added back atomically.
    for (physx::PxU32 rawColumn = 0u;
         rawColumn < rawRowCount && finite; ++rawColumn) {
      physx::PxVec3 linearDelta =
          rawAxes[rawColumn] * (body.invMass * linearGain);
      physx::PxVec3 angularDelta = body.invInertiaWorld.transform(
          rawPositionAlAngularJacobians[rawColumn]) * angularGain;
      body.projectLockedLinearVector(linearDelta);
      body.projectLockedAngularVector(angularDelta);
      finite = linearDelta.isFinite() && angularDelta.isFinite();
      for (physx::PxU32 row = 0u; row < rowCount && finite; ++row) {
        const double crossResponse = double(
            axes[row].dot(linearDelta) +
            materialAngularJacobians[row].dot(angularDelta));
        q[row] -= crossResponse * rawAlImpulse[rawColumn];
        finite = std::isfinite(crossResponse) && std::isfinite(q[row]);
      }
    }
    for (physx::PxU32 column = 0u;
         column < rowCount && finite; ++column) {
      physx::PxVec3 linearDelta = axes[column] * body.invMass;
      physx::PxVec3 angularDelta = body.invInertiaWorld.transform(
          materialAngularJacobians[column]);
      body.projectLockedLinearVector(linearDelta);
      body.projectLockedAngularVector(angularDelta);
      finite = linearDelta.isFinite() && angularDelta.isFinite();
      for (physx::PxU32 row = 0u; row < rowCount && finite; ++row) {
        const double value = double(
            axes[row].dot(linearDelta) +
            materialAngularJacobians[row].dot(angularDelta));
        response[row * rowCount + column] = value;
        finite = std::isfinite(value);
      }
    }

    AvbdCoulombNcpLayout layout;
    layout.rowCount = rowCount;
    layout.normalCount = contactCount;
    layout.anchorCount = anchorCount;
    layout.patchCount = solverPatchCount;
    layout.normalRows = normalRows;
    layout.normalPatches = solverContactPatch;
    layout.tangentRows = tangentRows;
    layout.tangentPatches = solverAnchorPatch;
    layout.patchAnchorCounts = solverPatchAnchorCounts;
    layout.patchFriction = solverPatchFriction;
    if (!finite || !solveAvbdCoulombNcpFixedPoint(
                       response, q, layout, impulses, scratch))
      continue;

    physx::PxVec3 linearImpulse(0.0f);
    physx::PxVec3 angularImpulse(0.0f);
    for (physx::PxU32 rawRow = 0u; rawRow < rawRowCount; ++rawRow) {
      const physx::PxReal impulse =
          physx::PxReal(rawAlImpulse[rawRow]);
      linearImpulse -= rawAxes[rawRow] * (linearGain * impulse);
      angularImpulse -= rawPositionAlAngularJacobians[rawRow] *
                        (angularGain * impulse);
    }
    for (physx::PxU32 row = 0u; row < rowCount; ++row) {
      const double impulse = impulses[row];
      if (!std::isfinite(impulse)) {
        finite = false;
        break;
      }
      linearImpulse += axes[row] * physx::PxReal(impulse);
      angularImpulse +=
          materialAngularJacobians[row] * physx::PxReal(impulse);
    }
    physx::PxVec3 linearDelta = linearImpulse * body.invMass;
    physx::PxVec3 angularDelta =
        body.invInertiaWorld.transform(angularImpulse);
    body.projectLockedLinearVector(linearDelta);
    body.projectLockedAngularVector(angularDelta);
    finite = finite && linearDelta.isFinite() && angularDelta.isFinite() &&
             (body.linearVelocity + linearDelta).isFinite() &&
             (body.angularVelocity + angularDelta).isFinite();
    if (!finite)
      continue;

    body.linearVelocity += linearDelta;
    body.angularVelocity += angularDelta;
    for (physx::PxU32 contactSlot = 0;
         contactSlot < contactCount; ++contactSlot) {
      AvbdContactConstraint &contact =
          contacts[contactIndices[contactSlot]];
      contact.velocityNormalImpulse =
          physx::PxReal(impulses[normalRows[contactSlot]]);
      contact.frictionSweepImpulse = physx::PxVec3(0.0f);
    }
    for (physx::PxU32 anchor = 0u; anchor < anchorCount; ++anchor) {
      AvbdContactConstraint &contact =
          contacts[contactIndices[anchorContactSlots[anchor]]];
      const physx::PxU32 row = tangentRows[anchor];
      contact.frictionSweepImpulse +=
          contact.tangent0 * physx::PxReal(impulses[row]) +
          contact.tangent1 * physx::PxReal(impulses[row + 1u]);
    }
  }
}

/**
 * Consume PxContactModifyCallback target velocity after pose-to-velocity
 * reconstruction.  The projection uses the same contact-local inverse
 * mass/inertia scales as PhysX's impulse solvers and remains unilateral on
 * the normal row.
 */
static void applyAvbdContactTargetVelocityImpl(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    const AvbdPostAlContactWorkPlan *workPlan) {
  if (!bodies || !contacts || dt <= 0.0f)
    return;

  applyAvbdContactMaterialFrictionManifolds(
      bodies, numBodies, contacts, numContacts,
      contactMap, linearVelAtSolveStart, angularVelAtSolveStart,
      linearPoseVelocityGain, angularPoseVelocityGain,
      dt, bounceThreshold, workPlan);

  if (!mayHavePostAlContactWork(
          workPlan, AvbdPostAlContactWorkPlan::ePOINT_TARGET))
    return;

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &cc = contacts[c];
    if (hasRigidMaterialConsumed(cc))
      continue;
    if (hasVelocityFrictionManifoldOwner(cc))
      continue;
    if (cc.targetVelocity.magnitudeSquared() <= 1e-12f)
      continue;

    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    const bool dynA = bA < numBodies && bodies[bA].invMass > 0.0f;
    const bool dynB = bB < numBodies && bodies[bB].invMass > 0.0f;
    if (!dynA && !dynB)
      continue;

    const bool solveStartTangentOwner =
        hasVelocityTangentTargetNormalSpan(cc);
    const physx::PxVec3 rA =
        dynA ? (solveStartTangentOwner ? bodies[bA].prevRotation
                                      : bodies[bA].rotation)
                   .rotate(cc.contactPointA)
             : physx::PxVec3(0.0f);
    const physx::PxVec3 rB =
        dynB ? (solveStartTangentOwner ? bodies[bB].prevRotation
                                      : bodies[bB].rotation)
                   .rotate(cc.contactPointB)
             : physx::PxVec3(0.0f);
    const physx::PxReal invMassA =
        dynA ? bodies[bA].invMass * cc.invMassScaleA : 0.0f;
    const physx::PxReal invMassB =
        dynB ? bodies[bB].invMass * cc.invMassScaleB : 0.0f;
    const physx::PxMat33 invInertiaA =
        dynA ? bodies[bA].invInertiaWorld * cc.invInertiaScaleA
             : physx::PxMat33(physx::PxZero);
    const physx::PxMat33 invInertiaB =
        dynB ? bodies[bB].invInertiaWorld * cc.invInertiaScaleB
             : physx::PxMat33(physx::PxZero);

    auto pointVelocity = [&](bool bodyA) {
      if (bodyA) {
        return dynA ? bodies[bA].linearVelocity +
                          bodies[bA].angularVelocity.cross(rA)
                    : physx::PxVec3(0.0f);
      }
      return dynB ? bodies[bB].linearVelocity +
                        bodies[bB].angularVelocity.cross(rB)
                  : physx::PxVec3(0.0f);
    };
    auto response = [&](const physx::PxVec3 &axis) {
      const physx::PxVec3 rAx = rA.cross(axis);
      const physx::PxVec3 rBx = rB.cross(axis);
      return invMassA + invMassB +
             rAx.dot(invInertiaA * rAx) +
             rBx.dot(invInertiaB * rBx);
    };
    auto applyImpulse = [&](const physx::PxVec3 &axis,
                            physx::PxReal impulse) {
      if (dynA) {
        bodies[bA].linearVelocity += axis * (impulse * invMassA);
        bodies[bA].angularVelocity +=
            invInertiaA * (rA.cross(axis) * impulse);
      }
      if (dynB) {
        bodies[bB].linearVelocity -= axis * (impulse * invMassB);
        bodies[bB].angularVelocity -=
            invInertiaB * (rB.cross(axis) * impulse);
      }
    };

    const physx::PxVec3 &normal = cc.contactNormal;
    physx::PxReal normalImpulse = 0.0f;
    const physx::PxReal normalResponse = response(normal);
    const bool ownedCombinedNormalTarget =
        hasVelocityTangentTargetOwner(cc) &&
        physx::PxAbs(cc.targetVelocity.dot(normal)) > 1.0e-6f;
    if ((!hasVelocityTangentTargetOwner(cc) ||
         ownedCombinedNormalTarget) &&
        normalResponse > 1e-12f) {
      const physx::PxReal currentNormal =
          (pointVelocity(true) - pointVelocity(false)).dot(normal);
      const physx::PxReal requestedNormal =
          cc.targetVelocity.dot(normal);
      const physx::PxReal deltaNormal =
          requestedNormal - currentNormal;
      if (deltaNormal > 0.0f) {
        normalImpulse = deltaNormal / normalResponse;
        if (cc.maxImpulse < PX_MAX_REAL) {
          const physx::PxReal existingImpulse =
              physx::PxMax(0.0f, -cc.header.lambda) * dt;
          normalImpulse = physx::PxMin(
              normalImpulse,
              physx::PxMax(0.0f, cc.maxImpulse - existingImpulse));
        }
        if (normalImpulse > 0.0f) {
          applyImpulse(normal, normalImpulse);
        }
      }
    }

    const physx::PxReal targetT0 =
        cc.targetVelocity.dot(cc.tangent0);
    const physx::PxReal targetT1 =
        cc.targetVelocity.dot(cc.tangent1);
    if (physx::PxAbs(targetT0) <= 1e-6f &&
        physx::PxAbs(targetT1) <= 1e-6f)
      continue;
    const physx::PxReal mu = contactCoulombMu(cc);
    const physx::PxReal existingNormalSupport =
        physx::PxMax(0.0f, -cc.header.lambda) * dt;
    const physx::PxReal normalSupport =
        hasVelocityTangentTargetOwner(cc)
            ? existingNormalSupport + normalImpulse
            : physx::PxMax(normalImpulse, existingNormalSupport);
    const physx::PxReal tangentLimit = mu * normalSupport;
    if (tangentLimit <= 0.0f)
      continue;

    const physx::PxVec3 relativeVelocity =
        pointVelocity(true) - pointVelocity(false);
    const physx::PxReal responseT0 = response(cc.tangent0);
    const physx::PxReal responseT1 = response(cc.tangent1);
    physx::PxReal impulseT0 =
        responseT0 > 1e-12f
            ? (targetT0 - relativeVelocity.dot(cc.tangent0)) / responseT0
            : 0.0f;
    physx::PxReal impulseT1 =
        responseT1 > 1e-12f
            ? (targetT1 - relativeVelocity.dot(cc.tangent1)) / responseT1
            : 0.0f;
    avbdProjectImpulseCone(tangentLimit, impulseT0, impulseT1);
    applyImpulse(cc.tangent0, impulseT0);
    applyImpulse(cc.tangent1, impulseT1);
    cc.frictionSweepImpulse +=
        cc.tangent0 * impulseT0 + cc.tangent1 * impulseT1;
  }
}

static bool isRigidDeepBodyStaticRecoverySplitSupportedImpl(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  if (!bodies || bodyIndex >= numBodies || !contacts)
    return false;

  bool foundContact = false;
  physx::PxReal worstInitialViolation = PX_MAX_REAL;
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 contactIndex = hasMapRange ? mapIndices[loopIndex] : loopIndex;
    const AvbdContactConstraint &contact = contacts[contactIndex];
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    if (bodyA != bodyIndex && bodyB != bodyIndex)
      continue;
    foundContact = true;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
        hasDeformableStaticAnchor(contact) || contact.friction > 0.0f ||
        contact.staticFriction > 0.0f || contact.restitution > 0.0f ||
        contact.targetVelocity.magnitudeSquared() > 1e-12f ||
        contact.maxImpulse < PX_MAX_REAL)
      return false;

    const bool dynamicIsA = bodyA == bodyIndex;
    const physx::PxReal linearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal angularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (!physx::PxIsFinite(linearScale) ||
        !physx::PxIsFinite(angularScale) || linearScale < 0.0f ||
        angularScale < 0.0f ||
        physx::PxAbs(linearScale - 1.0f) > 1e-6f ||
        physx::PxAbs(angularScale - 1.0f) > 1e-6f)
      return false;

    const physx::PxVec3 initialWorldA =
        dynamicIsA
            ? bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointA)
            : contact.staticPrevWorldPoint;
    const physx::PxVec3 initialWorldB =
        dynamicIsA
            ? contact.staticPrevWorldPoint
            : bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointB);
    const physx::PxReal initialViolation =
        (initialWorldA - initialWorldB).dot(contact.contactNormal) +
        contact.penetrationDepth;
    worstInitialViolation =
        physx::PxMin(worstInitialViolation, initialViolation);
  }
  return foundContact &&
         worstInitialViolation <
             -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE *
                 physx::PxMax(lengthScale, physx::PxReal(1e-6f));
}

static bool isRigidFiniteBodyStaticMaterialSplitSupportedImpl(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  if (!bodies || bodyIndex >= numBodies || !contacts)
    return false;

  bool foundContact = false;
  physx::PxU32 contactCount = 0;
  physx::PxReal manifoldLinearScale = 0.0f;
  physx::PxReal manifoldAngularScale = 0.0f;
  const physx::PxReal deepLimit =
      -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE *
      physx::PxMax(lengthScale, physx::PxReal(1.0e-6f));
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 contactIndex = hasMapRange ? mapIndices[loopIndex] : loopIndex;
    const AvbdContactConstraint &contact = contacts[contactIndex];
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    if (bodyA != bodyIndex && bodyB != bodyIndex)
      continue;
    if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
        hasDeformableStaticAnchor(contact) || contact.friction > 0.0f ||
        contact.staticFriction > 0.0f || contact.restitution < 0.0f ||
        contact.targetVelocity.magnitudeSquared() > 1.0e-12f ||
        contact.maxImpulse >= PX_MAX_REAL ||
        !physx::PxIsFinite(contact.maxImpulse) ||
        contact.maxImpulse < 0.0f)
      return false;

    const bool dynamicIsA = bodyA == bodyIndex;
    const physx::PxReal linearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal angularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (!physx::PxIsFinite(linearScale) ||
        !physx::PxIsFinite(angularScale) || linearScale < 0.0f ||
        angularScale < 0.0f)
      return false;
    if (!foundContact) {
      manifoldLinearScale = linearScale;
      manifoldAngularScale = angularScale;
    } else if (
        physx::PxAbs(linearScale - manifoldLinearScale) > 1.0e-6f ||
        physx::PxAbs(angularScale - manifoldAngularScale) > 1.0e-6f) {
      return false;
    }
    foundContact = true;
    ++contactCount;

    const physx::PxVec3 initialWorldA =
        dynamicIsA
            ? bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointA)
            : contact.staticPrevWorldPoint;
    const physx::PxVec3 initialWorldB =
        dynamicIsA
            ? contact.staticPrevWorldPoint
            : bodies[bodyIndex].prevPosition +
                  bodies[bodyIndex].prevRotation.rotate(contact.contactPointB);
    const physx::PxReal initialViolation =
        (initialWorldA - initialWorldB).dot(contact.contactNormal) +
        contact.penetrationDepth;
    if (contact.persistentPointMatched == 0 &&
        initialViolation < deepLimit)
      return false;
  }
  return foundContact && contactCount >= 1 && contactCount <= 4;
}

/**
 * Material normal-velocity response after pose finalize (friction already applied).
 * - Deformable: mesh-relative e=0 (heave).
 * - Rigid body-static: material restitution with scene bounce threshold.
 * - Dyn-dyn: same restitution on relative normal speed (linear mass split).
 * Friction mu is consumed elsewhere (dual cone + body-static friction post-pass).
 */
static bool applyBodyStaticRestitutionSpatialRow(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    physx::PxReal &linearDeltaMagnitude) {
  linearDeltaMagnitude = 0.0f;
  if (!linearVelAtSolveStart || !angularVelAtSolveStart ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies || dt <= 0.0f)
    return false;

  AvbdSolverBody &body = bodies[bodyIndex];
  const physx::PxReal invDt = 1.0f / dt;
  physx::PxVec3 aggregateNormal(0.0f);
  physx::PxVec3 aggregateAngularJacobian(0.0f);
  physx::PxReal aggregateApproach = 0.0f;
  physx::PxReal aggregateRestitution = 0.0f;
  physx::PxReal aggregateStaticNormalVelocity = 0.0f;
  physx::PxReal aggregateLinearScale = 0.0f;
  physx::PxReal aggregateAngularScale = 0.0f;
  physx::PxU32 rowCount = 0;

  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
    const AvbdContactConstraint &cc = contacts[c];
    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies) ||
        hasDeformableStaticAnchor(cc) || (bA != bodyIndex && bB != bodyIndex) ||
        cc.restitution <= 0.0f || cc.maxImpulse < PX_MAX_REAL)
      continue;

    const bool dynIsA = bA == bodyIndex;
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);
    const physx::PxVec3 localPoint =
        dynIsA ? cc.contactPointA : cc.contactPointB;
    const physx::PxVec3 r0 = body.prevRotation.rotate(localPoint);
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 staticNow =
        dynIsA ? cc.contactPointB : cc.contactPointA;
    const physx::PxReal staticNormalVelocity =
        ((staticNow - cc.staticPrevWorldPoint) * invDt).dot(nd);
    const physx::PxReal solveStartPointVn =
        (*linearVelAtSolveStart)[bodyIndex].dot(nd) +
        (*angularVelAtSolveStart)[bodyIndex].dot(r0.cross(nd)) -
        staticNormalVelocity;
    const physx::PxReal approach =
        physx::PxMax(-solveStartPointVn, physx::PxReal(0.0f));
    if (approach <= bounceThreshold)
      continue;

    aggregateNormal += nd;
    aggregateAngularJacobian += r.cross(nd);
    aggregateApproach += approach;
    aggregateRestitution += physx::PxMin(cc.restitution, physx::PxReal(1.0f));
    aggregateStaticNormalVelocity += staticNormalVelocity;
    aggregateLinearScale += dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
    aggregateAngularScale +=
        dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
    ++rowCount;
  }

  if (rowCount == 0)
    return false;

  const physx::PxReal invRowCount = 1.0f / physx::PxReal(rowCount);
  aggregateNormal *= invRowCount;
  aggregateAngularJacobian *= invRowCount;
  aggregateApproach *= invRowCount;
  aggregateRestitution *= invRowCount;
  aggregateStaticNormalVelocity *= invRowCount;
  aggregateLinearScale *= invRowCount;
  aggregateAngularScale *= invRowCount;

  const physx::PxVec3 angularResponse =
      body.invInertiaWorld.transform(aggregateAngularJacobian) *
      aggregateAngularScale;
  const physx::PxReal response =
      body.invMass * aggregateLinearScale *
          aggregateNormal.magnitudeSquared() +
      aggregateAngularJacobian.dot(angularResponse);
  if (!physx::PxIsFinite(response) || response <= 1.0e-12f)
    return false;

  const physx::PxReal currentRelativeVn =
      body.linearVelocity.dot(aggregateNormal) +
      body.angularVelocity.dot(aggregateAngularJacobian) -
      aggregateStaticNormalVelocity;
  const physx::PxReal desiredRelativeVn =
      aggregateRestitution * aggregateApproach;
  const physx::PxReal impulse =
      (desiredRelativeVn - currentRelativeVn) / response;
  if (!physx::PxIsFinite(impulse))
    return false;
  if (impulse <= 1.0e-8f)
    return true;

  const physx::PxVec3 linearDelta =
      aggregateNormal * (impulse * body.invMass * aggregateLinearScale);
  body.linearVelocity += linearDelta;
  body.angularVelocity += angularResponse * impulse;
  linearDeltaMagnitude = linearDelta.magnitude();
  return true;
}

/**
 * Solve the free block of a finite-contact active set directly. P1I is
 * deliberately limited to at most four rows, so the whole manifold can be
 * solved as one deterministic objective instead of replaying point-wise
 * velocity Gauss-Seidel after the position solve.
 */
static bool solveFiniteContactFreeSystem(
    const physx::PxReal response[4][4], const physx::PxReal rhs[4],
    const physx::PxU32 freeRows[4], physx::PxU32 freeCount,
    physx::PxReal solution[4]) {
  physx::PxReal augmented[4][5] = {};
  for (physx::PxU32 row = 0; row < freeCount; ++row) {
    for (physx::PxU32 column = 0; column < freeCount; ++column) {
      augmented[row][column] =
          response[freeRows[row]][freeRows[column]];
    }
    augmented[row][freeCount] = rhs[row];
  }

  for (physx::PxU32 column = 0; column < freeCount; ++column) {
    physx::PxU32 pivot = column;
    physx::PxReal pivotMagnitude =
        physx::PxAbs(augmented[column][column]);
    for (physx::PxU32 row = column + 1; row < freeCount; ++row) {
      const physx::PxReal candidate =
          physx::PxAbs(augmented[row][column]);
      if (candidate > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = candidate;
      }
    }
    if (!physx::PxIsFinite(pivotMagnitude) ||
        pivotMagnitude <= 1.0e-10f)
      return false;
    if (pivot != column) {
      for (physx::PxU32 entry = column; entry <= freeCount; ++entry) {
        const physx::PxReal temporary = augmented[column][entry];
        augmented[column][entry] = augmented[pivot][entry];
        augmented[pivot][entry] = temporary;
      }
    }

    const physx::PxReal inversePivot =
        1.0f / augmented[column][column];
    for (physx::PxU32 entry = column; entry <= freeCount; ++entry)
      augmented[column][entry] *= inversePivot;
    for (physx::PxU32 row = 0; row < freeCount; ++row) {
      if (row == column)
        continue;
      const physx::PxReal factor = augmented[row][column];
      for (physx::PxU32 entry = column; entry <= freeCount; ++entry)
        augmented[row][entry] -= factor * augmented[column][entry];
    }
  }

  for (physx::PxU32 row = 0; row < freeCount; ++row) {
    solution[freeRows[row]] = augmented[row][freeCount];
    if (!physx::PxIsFinite(solution[freeRows[row]]))
      return false;
  }
  return true;
}

static bool solveFiniteContactObjective(
    const physx::PxReal response[4][4], const physx::PxReal q[4],
    const physx::PxReal caps[4], physx::PxU32 rowCount,
    physx::PxReal impulses[4]) {
  // Enumerate lower/free/upper status for the bounded convex objective.
  // At four rows this is at most 3^4 = 81 direct candidates.
  physx::PxU32 statusCount = 1;
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    statusCount *= 3;

  bool found = false;
  physx::PxReal bestObjective = PX_MAX_REAL;
  for (physx::PxU32 encoded = 0; encoded < statusCount; ++encoded) {
    physx::PxU32 code = encoded;
    physx::PxU8 status[4] = {};
    physx::PxU32 freeRows[4] = {};
    physx::PxU32 freeCount = 0;
    physx::PxReal candidate[4] = {};
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      status[row] = static_cast<physx::PxU8>(code % 3);
      code /= 3;
      if (status[row] == 1)
        freeRows[freeCount++] = row;
      else if (status[row] == 2)
        candidate[row] = caps[row];
    }

    physx::PxReal rhs[4] = {};
    for (physx::PxU32 freeIndex = 0; freeIndex < freeCount; ++freeIndex) {
      const physx::PxU32 row = freeRows[freeIndex];
      rhs[freeIndex] = -q[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column) {
        if (status[column] == 2)
          rhs[freeIndex] -= response[row][column] * caps[column];
      }
    }
    if (freeCount > 0 &&
        !solveFiniteContactFreeSystem(
            response, rhs, freeRows, freeCount, candidate))
      continue;

    physx::PxReal scale = 1.0f;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      scale = physx::PxMax(scale, physx::PxAbs(q[row]));
    const physx::PxReal tolerance = 1.0e-5f * scale;
    bool valid = true;
    for (physx::PxU32 row = 0; row < rowCount && valid; ++row) {
      if (candidate[row] < -tolerance ||
          candidate[row] > caps[row] + tolerance) {
        valid = false;
        break;
      }
      candidate[row] = physx::PxClamp(
          candidate[row], physx::PxReal(0.0f), caps[row]);
      physx::PxReal gradient = q[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column)
        gradient += response[row][column] * candidate[column];
      if ((status[row] == 0 && gradient < -tolerance) ||
          (status[row] == 1 && physx::PxAbs(gradient) > tolerance) ||
          (status[row] == 2 && gradient > tolerance))
        valid = false;
    }
    if (!valid)
      continue;

    physx::PxReal objective = 0.0f;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      objective += q[row] * candidate[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column) {
        objective += 0.5f * candidate[row] *
                     response[row][column] * candidate[column];
      }
    }
    if (!physx::PxIsFinite(objective))
      continue;
    if (!found || objective < bestObjective) {
      found = true;
      bestObjective = objective;
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        impulses[row] = candidate[row];
    }
  }
  return found;
}

static bool applyBodyStaticFiniteSpatialBudget(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    physx::PxReal &linearDeltaMagnitude) {
  linearDeltaMagnitude = 0.0f;
  if (!linearVelAtSolveStart || !angularVelAtSolveStart ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies || dt <= 0.0f)
    return false;

  AvbdSolverBody &body = bodies[bodyIndex];
  const physx::PxReal invDt = 1.0f / dt;
  physx::PxU32 rowIndices[4] = {};
  physx::PxVec3 normals[4] = {};
  physx::PxVec3 angularJacobians[4] = {};
  physx::PxReal targets[4] = {};
  physx::PxReal staticNormalVelocities[4] = {};
  physx::PxReal caps[4] = {};
  physx::PxU32 rowCount = 0;
  physx::PxReal linearScale = 0.0f;
  physx::PxReal angularScale = 0.0f;

  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  const bool hasMapRange = getAvbdBodyContactRange(
      contactMap, bodyIndex, mapIndices, mapCount);
  const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
  for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
    const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
    AvbdContactConstraint &cc = contacts[c];
    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    if (!isBodyVsStaticContact(bA, bB, numBodies) ||
        hasDeformableStaticAnchor(cc) || (bA != bodyIndex && bB != bodyIndex) ||
        cc.maxImpulse >= PX_MAX_REAL)
      continue;

    const bool dynIsA = bA == bodyIndex;
    const physx::PxReal rowLinearScale =
        dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
    const physx::PxReal rowAngularScale =
        dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);
    const physx::PxVec3 localPoint =
        dynIsA ? cc.contactPointA : cc.contactPointB;
    const physx::PxVec3 r0 = body.prevRotation.rotate(localPoint);
    const physx::PxVec3 r = body.rotation.rotate(localPoint);
    const physx::PxVec3 angularJacobian = r.cross(nd);
    const physx::PxReal cap =
        physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f));
    const physx::PxVec3 staticNow =
        dynIsA ? cc.contactPointB : cc.contactPointA;
    const physx::PxReal staticNormalVelocity =
        ((staticNow - cc.staticPrevWorldPoint) * invDt).dot(nd);
    const physx::PxReal solveStartPointVn =
        (*linearVelAtSolveStart)[bodyIndex].dot(nd) +
        (*angularVelAtSolveStart)[bodyIndex].dot(r0.cross(nd)) -
        staticNormalVelocity;
    const physx::PxReal approach =
        physx::PxMax(-solveStartPointVn, physx::PxReal(0.0f));
    const physx::PxVec3 initialWorldA =
        dynIsA
            ? body.prevPosition +
                  body.prevRotation.rotate(cc.contactPointA)
            : cc.staticPrevWorldPoint;
    const physx::PxVec3 initialWorldB =
        dynIsA
            ? cc.staticPrevWorldPoint
            : body.prevPosition +
                  body.prevRotation.rotate(cc.contactPointB);
    const physx::PxReal initialViolation =
        (initialWorldA - initialWorldB).dot(cc.contactNormal) +
        cc.penetrationDepth;
    // Match TGS impact eligibility: restitution is active only when the
    // solve-start point speed exceeds the scene threshold and the point will
    // close its current separation within this step.
    const bool collidingWithinStep =
        approach > initialViolation * invDt;
    const physx::PxReal restitution =
        cc.restitution > 0.0f && approach > bounceThreshold &&
                collidingWithinStep
            ? physx::PxMin(cc.restitution, physx::PxReal(1.0f))
            : physx::PxReal(0.0f);

    if (rowCount >= 4)
      return false;
    if (rowCount == 0) {
      linearScale = rowLinearScale;
      angularScale = rowAngularScale;
    }
    rowIndices[rowCount] = c;
    normals[rowCount] = nd;
    angularJacobians[rowCount] = angularJacobian;
    targets[rowCount] = restitution * approach;
    staticNormalVelocities[rowCount] = staticNormalVelocity;
    caps[rowCount] = cap;
    ++rowCount;
  }

  if (rowCount == 0)
    return false;

  physx::PxReal response[4][4] = {};
  physx::PxReal q[4] = {};
  physx::PxReal impulses[4] = {};
  physx::PxReal totalCap = 0.0f;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    const physx::PxReal currentRelativeVn =
        body.linearVelocity.dot(normals[row]) +
        body.angularVelocity.dot(angularJacobians[row]) -
        staticNormalVelocities[row];
    q[row] = currentRelativeVn - targets[row];
    totalCap += caps[row];
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      response[row][column] =
          body.invMass * linearScale *
              normals[row].dot(normals[column]) +
          angularJacobians[row].dot(
              body.invInertiaWorld.transform(
                  angularJacobians[column]) *
              angularScale);
    }
  }

  if (totalCap <= 1.0e-8f)
    return true;
  if (!solveFiniteContactObjective(
          response, q, caps, rowCount, impulses))
    return false;

  physx::PxVec3 linearImpulse(0.0f);
  physx::PxVec3 angularImpulse(0.0f);
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    linearImpulse += normals[row] * impulses[row];
    angularImpulse += angularJacobians[row] * impulses[row];
  }

  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    // The bounded velocity objective owns only its applied impulse.  Its
    // result must not replace the position AL multiplier that is cached for
    // the next frame.
    contacts[rowIndices[row]].velocityNormalImpulse = impulses[row];
  }
  const physx::PxVec3 linearDelta =
      linearImpulse * (body.invMass * linearScale);
  body.linearVelocity += linearDelta;
  body.angularVelocity +=
      body.invInertiaWorld.transform(angularImpulse) * angularScale;
  linearDeltaMagnitude = linearDelta.magnitude();
  return true;
}

struct SurfaceFinalizeTopologyNode {
  physx::PxU32 parent;
  physx::PxU32 bodyCount;
  physx::PxU32 rowCount;
  physx::PxReal firstLinearScale;
  physx::PxReal firstAngularScale;
  physx::PxU8 strictOwner;
  physx::PxU8 bodyStrictOwner;
  physx::PxU8 restitution;
  physx::PxU8 finiteImpulse;
  physx::PxU8 targetVelocity;
  physx::PxU8 mixedScale;
  physx::PxU8 rigidStatic;
  physx::PxU8 nonOwnerDeformable;
  physx::PxU8 scaleSeen;
  physx::PxU8 lockedDof;
  physx::PxU8 nonDynamicBody;
  physx::PxU8 fastImpact;
  physx::PxU8 snapshotUnsupported;
  physx::PxU32 budgetDiagNoCorrectionRows;
  physx::PxU32 budgetDiagZeroBudgetRequiredRows;
  physx::PxU32 budgetDiagWithinBudgetRows;
  physx::PxU32 budgetDiagOverBudgetRows;
  physx::PxU32 budgetDiagUnsupportedRows;
};

struct SurfaceFinalizeBudgetDiagSnapshot {
  physx::PxReal outwardVelocity;
  physx::PxReal maximumImpulse;
  physx::PxU8 classification;
  physx::PxU8 fastImpact;
  physx::PxU8 unsupported;

  SurfaceFinalizeBudgetDiagSnapshot()
      : outwardVelocity(0.0f), maximumImpulse(0.0f),
        classification(0), fastImpact(0),
        unsupported(0) {}
};

enum SurfaceFinalizeBudgetDiagClass {
  eBUDGET_DIAG_NOT_APPLICABLE = 0,
  eBUDGET_DIAG_NO_CORRECTION,
  eBUDGET_DIAG_ZERO_BUDGET_REQUIRED,
  eBUDGET_DIAG_WITHIN_BUDGET,
  eBUDGET_DIAG_OVER_BUDGET,
  eBUDGET_DIAG_UNSUPPORTED
};

struct SurfaceFinalizeMatrixFreeRow {
  physx::PxU32 bodies[2];
  physx::PxVec3 axes[2];
  physx::PxVec3 angularJacobians[2];
};

struct SurfaceFinalizeDoubleVec3 {
  double x;
  double y;
  double z;

  SurfaceFinalizeDoubleVec3() : x(0.0), y(0.0), z(0.0) {}
};

static SurfaceFinalizeDoubleVec3
transformSurfaceFinalizeDouble(
    const physx::PxMat33 &matrix,
    const SurfaceFinalizeDoubleVec3 &value) {
  SurfaceFinalizeDoubleVec3 result;
  result.x = double(matrix.column0.x) * value.x +
             double(matrix.column1.x) * value.y +
             double(matrix.column2.x) * value.z;
  result.y = double(matrix.column0.y) * value.x +
             double(matrix.column1.y) * value.y +
             double(matrix.column2.y) * value.z;
  result.z = double(matrix.column0.z) * value.x +
             double(matrix.column1.z) * value.y +
             double(matrix.column2.z) * value.z;
  return result;
}

// Matrix-free equivalent of the dense J M^-1 J^T bounded solve.  It is used
// only as a backend choice for broad components; capability and KKT semantics
// do not depend on row count.  Until the unbounded feasibility classifier is
// also scalable, a converged bounded optimum with residual fails closed as
// ResidualUnclassified instead of being guessed as BudgetExhausted or
// mislabeled as a numerical fault.
static AvbdBoundedProjectionResult
solveSurfaceFinalizeMatrixFreeBoundedProjection(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    physx::PxU32 root,
    const AvbdContactConstraint *contacts,
    const physx::PxArray<physx::PxU32> &orderedRows,
    const physx::PxArray<double> &outward,
    const physx::PxArray<double> &upperBounds,
    double relativeTolerance = 1.0e-6) {
  using namespace AvbdBoundedProjectionDetail;
  AvbdBoundedProjectionResult result;
  const physx::PxU32 rowCount = orderedRows.size();
  result.candidateImpulses.resize(rowCount, 0.0);
  result.commitImpulses.resize(rowCount, 0.0);
  if (!bodies || !contacts || nodes.size() != numBodies ||
      rowCount == 0 || outward.size() != rowCount ||
      upperBounds.size() != rowCount ||
      !std::isfinite(relativeTolerance) || relativeTolerance <= 0.0)
    return result;

  physx::PxArray<SurfaceFinalizeMatrixFreeRow> rows(rowCount);
  double velocityScale = 1.0;
  double impulseScale = 1.0;
  double trace = 0.0;
  double maximumDiagonal = 0.0;
  bool needsCorrection = false;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (!std::isfinite(outward[row]) ||
        !std::isfinite(upperBounds[row]) || upperBounds[row] < 0.0)
      return result;
    needsCorrection = needsCorrection || outward[row] > 0.0;
    velocityScale = std::max(velocityScale, std::fabs(outward[row]));
    impulseScale = std::max(impulseScale, upperBounds[row]);
    const AvbdContactConstraint &contact = contacts[orderedRows[row]];
    SurfaceFinalizeMatrixFreeRow &operatorRow = rows[row];
    operatorRow.bodies[0] = contact.header.bodyIndexA;
    operatorRow.bodies[1] = contact.header.bodyIndexB;
    operatorRow.axes[0] = contact.contactNormal;
    operatorRow.axes[1] = -contact.contactNormal;
    const physx::PxVec3 localPoints[2] = {
        contact.contactPointA, contact.contactPointB};
    double diagonal = 0.0;
    for (physx::PxU32 end = 0; end < 2; ++end) {
      const physx::PxU32 body = operatorRow.bodies[end];
      operatorRow.angularJacobians[end] = physx::PxVec3(0.0f);
      if (body >= numBodies)
        continue;
      if (nodes[body].parent != root)
        return result;
      const physx::PxVec3 arm =
          bodies[body].rotation.rotate(localPoints[end]);
      operatorRow.angularJacobians[end] =
          arm.cross(operatorRow.axes[end]);
      const double linearResponse =
          double(bodies[body].invMass * nodes[body].firstLinearScale);
      const double angularResponse = double(
          operatorRow.angularJacobians[end].dot(
              bodies[body].invInertiaWorld.transform(
                  operatorRow.angularJacobians[end])) *
          nodes[body].firstAngularScale);
      diagonal += linearResponse + angularResponse;
    }
    if (!std::isfinite(diagonal) || diagonal < 0.0)
      return result;
    trace += diagonal;
    maximumDiagonal = std::max(maximumDiagonal, diagonal);
  }
  if (!needsCorrection) {
    result.status = eAVBD_BOUNDED_NO_CORRECTION;
    result.lowerRows = rowCount;
    return result;
  }
  if (!std::isfinite(trace) || trace <= 1.0e-14) {
    result.status = eAVBD_BOUNDED_INFEASIBLE;
    result.maximumResidual = velocityScale;
    return result;
  }

  physx::PxArray<SurfaceFinalizeDoubleVec3> linearImpulses(
      numBodies);
  physx::PxArray<SurfaceFinalizeDoubleVec3> angularImpulses(
      numBodies);
  physx::PxArray<SurfaceFinalizeDoubleVec3> linearResponses(
      numBodies);
  physx::PxArray<SurfaceFinalizeDoubleVec3> angularResponses(
      numBodies);
  const auto applyResponse =
      [&](const physx::PxArray<double> &impulses,
          physx::PxArray<double> &values) {
        std::fill(
            linearImpulses.begin(), linearImpulses.end(),
            SurfaceFinalizeDoubleVec3());
        std::fill(
            angularImpulses.begin(), angularImpulses.end(),
            SurfaceFinalizeDoubleVec3());
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          const double impulse = impulses[row];
          for (physx::PxU32 end = 0; end < 2; ++end) {
            const physx::PxU32 body = rows[row].bodies[end];
            if (body >= numBodies)
              continue;
            linearImpulses[body].x +=
                double(rows[row].axes[end].x) * impulse;
            linearImpulses[body].y +=
                double(rows[row].axes[end].y) * impulse;
            linearImpulses[body].z +=
                double(rows[row].axes[end].z) * impulse;
            angularImpulses[body].x +=
                double(rows[row].angularJacobians[end].x) * impulse;
            angularImpulses[body].y +=
                double(rows[row].angularJacobians[end].y) * impulse;
            angularImpulses[body].z +=
                double(rows[row].angularJacobians[end].z) * impulse;
          }
        }
        for (physx::PxU32 body = 0; body < numBodies; ++body) {
          if (nodes[body].parent != root) {
            linearResponses[body] = SurfaceFinalizeDoubleVec3();
            angularResponses[body] = SurfaceFinalizeDoubleVec3();
            continue;
          }
          const double linearScale =
              double(bodies[body].invMass) *
              double(nodes[body].firstLinearScale);
          linearResponses[body].x =
              linearImpulses[body].x * linearScale;
          linearResponses[body].y =
              linearImpulses[body].y * linearScale;
          linearResponses[body].z =
              linearImpulses[body].z * linearScale;
          angularResponses[body] =
              transformSurfaceFinalizeDouble(
                  bodies[body].invInertiaWorld,
                  angularImpulses[body]);
          const double angularScale =
              double(nodes[body].firstAngularScale);
          angularResponses[body].x *= angularScale;
          angularResponses[body].y *= angularScale;
          angularResponses[body].z *= angularScale;
        }
        values.resize(rowCount);
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          double value = 0.0;
          for (physx::PxU32 end = 0; end < 2; ++end) {
            const physx::PxU32 body = rows[row].bodies[end];
            if (body >= numBodies)
              continue;
            value +=
                double(rows[row].axes[end].x) *
                    linearResponses[body].x +
                double(rows[row].axes[end].y) *
                    linearResponses[body].y +
                double(rows[row].axes[end].z) *
                    linearResponses[body].z +
                double(rows[row].angularJacobians[end].x) *
                    angularResponses[body].x +
                double(rows[row].angularJacobians[end].y) *
                    angularResponses[body].y +
                double(rows[row].angularJacobians[end].z) *
                    angularResponses[body].z;
          }
          values[row] = value;
        }
      };
  const double feasibilityTolerance =
      relativeTolerance * velocityScale;
  const double boundTolerance =
      relativeTolerance * impulseScale;
  result.projectedGradientTolerance = feasibilityTolerance;
  double lipschitzBound = maximumDiagonal;
  physx::PxArray<double> impulses(rowCount, 0.0);
  physx::PxArray<double> extrapolated(rowCount, 0.0);
  physx::PxArray<double> next(rowCount, 0.0);
  physx::PxArray<double> responseValues;
  physx::PxArray<double> gradientValues(rowCount, 0.0);
  physx::PxArray<double> baseResponse;
  double acceleration = 1.0;
  double currentObjective = 0.0;
  const physx::PxU32 iterationLimit =
      physx::PxMax(
          physx::PxU32(4096),
          physx::PxU32(1024 + 128 * nodes[root].bodyCount));
  bool converged = false;
  const auto takeProjectedStep =
      [&](const physx::PxArray<double> &base,
          physx::PxArray<double> &candidate,
          physx::PxArray<double> &candidateResponse,
          double &candidateObjective) {
        applyResponse(base, baseResponse);
        double baseObjective = 0.0;
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          gradientValues[row] =
              baseResponse[row] - outward[row];
          baseObjective +=
              0.5 * base[row] * baseResponse[row] -
              outward[row] * base[row];
        }
        if (!std::isfinite(baseObjective))
          return false;
        for (;;) {
          const double inverseLipschitz =
              1.0 / lipschitzBound;
          double gradientStep = 0.0;
          double stepNormSquared = 0.0;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            candidate[row] = std::min(
                upperBounds[row],
                std::max(
                    0.0,
                    base[row] -
                        inverseLipschitz *
                            gradientValues[row]));
            const double delta =
                candidate[row] - base[row];
            gradientStep += gradientValues[row] * delta;
            stepNormSquared += delta * delta;
          }
          applyResponse(candidate, candidateResponse);
          candidateObjective = 0.0;
          for (physx::PxU32 row = 0; row < rowCount; ++row)
            candidateObjective +=
                0.5 * candidate[row] *
                    candidateResponse[row] -
                outward[row] * candidate[row];
          const double modelObjective =
              baseObjective + gradientStep +
              0.5 * lipschitzBound * stepNormSquared;
          const double modelSlack =
              1.0e-13 *
              std::max(
                  1.0,
                  std::max(
                      std::fabs(candidateObjective),
                      std::fabs(modelObjective)));
          if (std::isfinite(candidateObjective) &&
              std::isfinite(modelObjective) &&
              candidateObjective <=
                  modelObjective + modelSlack)
            return true;
          lipschitzBound *= 2.0;
          if (!std::isfinite(lipschitzBound))
            return false;
        }
      };
  for (physx::PxU32 iteration = 0;
       iteration < iterationLimit; ++iteration) {
    double nextObjective = 0.0;
    if (!takeProjectedStep(
            extrapolated, next, responseValues,
            nextObjective))
      return result;
    const double objectiveSlack =
        1.0e-13 * std::max(1.0, std::fabs(currentObjective));
    if (nextObjective > currentObjective + objectiveSlack) {
      extrapolated = impulses;
      acceleration = 1.0;
      if (!takeProjectedStep(
              extrapolated, next, responseValues,
              nextObjective) ||
          nextObjective > currentObjective + 16.0 * objectiveSlack)
        return result;
    }
    impulses.swap(next);
    currentObjective = nextObjective;
    result.iterations = iteration + 1;
    applyResponse(impulses, responseValues);
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      gradientValues[row] = responseValues[row] - outward[row];
    if (projectedGradientViolation(
            gradientValues, impulses, upperBounds, boundTolerance) <=
        feasibilityTolerance) {
      converged = true;
      break;
    }
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(
                         1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        (acceleration - 1.0) / nextAcceleration;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      extrapolated[row] =
          impulses[row] + momentum * (impulses[row] - next[row]);
    acceleration = nextAcceleration;
    if ((iteration + 1) % 64 == 0) {
      extrapolated = impulses;
      acceleration = 1.0;
    }
  }
  if (!converged) {
    result.maximumKktViolation = projectedGradientViolation(
        gradientValues, impulses, upperBounds, boundTolerance);
    result.status = eAVBD_BOUNDED_ITERATION_LIMIT;
    return result;
  }

  result.maximumKktViolation = projectedGradientViolation(
      gradientValues, impulses, upperBounds, boundTolerance);
  double maximumResidual = 0.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (upperBounds[row] <= boundTolerance ||
        upperBounds[row] - impulses[row] <= boundTolerance)
      ++result.upperRows;
    else if (impulses[row] <= boundTolerance)
      ++result.lowerRows;
    else
      ++result.freeRows;
    maximumResidual =
        std::max(maximumResidual, -gradientValues[row]);
  }
  result.maximumResidual = maximumResidual;
  result.candidateImpulses = impulses;
  if (!std::isfinite(maximumResidual) ||
      maximumResidual > 4.0 * feasibilityTolerance) {
    result.status = std::isfinite(maximumResidual)
                        ? eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED
                        : eAVBD_BOUNDED_NUMERICAL_FAILURE;
    return result;
  }
  result.commitImpulses = result.candidateImpulses;
  result.status = eAVBD_BOUNDED_SOLVED;
  return result;
}

static bool isSurfaceFinalizeContactNear(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint &contact);

static SurfaceFinalizeBudgetDiagSnapshot
classifySurfaceFinalizeBudgetDiag(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint &contact, physx::PxReal dt,
    physx::PxReal lengthScale,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart) {
  SurfaceFinalizeBudgetDiagSnapshot snapshot;
  if (!bodies || dt <= 0.0f ||
      !isSurfaceFinalizeContactNear(bodies, numBodies, contact))
    return snapshot;
  if (contact.restitution != 0.0f ||
      contact.targetVelocity.magnitudeSquared() > 1.0e-12f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  const bool dynamicA = bodyA < numBodies;
  const bool dynamicB = bodyB < numBodies;
  if (!dynamicA && !dynamicB)
    return snapshot;
  if ((dynamicA && (bodies[bodyA].invMass <= 0.0f ||
                    bodies[bodyA].lockFlags != 0)) ||
      (dynamicB && (bodies[bodyB].invMass <= 0.0f ||
                    bodies[bodyB].lockFlags != 0))) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const auto pointVelocity =
      [&](physx::PxU32 body, const physx::PxVec3 &localPoint) {
        if (body >= numBodies)
          return physx::PxVec3(0.0f);
        const physx::PxVec3 arm =
            bodies[body].rotation.rotate(localPoint);
        return bodies[body].linearVelocity +
               bodies[body].angularVelocity.cross(arm);
      };
  physx::PxVec3 velocityA =
      pointVelocity(bodyA, contact.contactPointA);
  physx::PxVec3 velocityB =
      pointVelocity(bodyB, contact.contactPointB);
  if (isBodyVsStaticContact(bodyA, bodyB, numBodies) &&
      hasDeformableStaticAnchor(contact)) {
    const physx::PxVec3 staticNow =
        dynamicA ? contact.contactPointB : contact.contactPointA;
    const physx::PxVec3 staticStep =
        staticNow - contact.staticPrevWorldPoint;
    const physx::PxReal aliasCap =
        AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
    if (!staticStep.isFinite() ||
        staticStep.magnitudeSquared() > aliasCap * aliasCap) {
      snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
      snapshot.unsupported = 1;
      return snapshot;
    }
    const physx::PxVec3 staticVelocity = staticStep / dt;
    if (dynamicA)
      velocityB = staticVelocity;
    else
      velocityA = staticVelocity;
  }

  const physx::PxReal outwardVelocity =
      (velocityA - velocityB).dot(contact.contactNormal);
  snapshot.outwardVelocity = outwardVelocity;
  const physx::PxReal velocityTolerance =
      1.0e-5f *
      physx::PxMax(lengthScale, physx::PxReal(1.0e-6f)) / dt;
  if (!physx::PxIsFinite(outwardVelocity)) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const bool haveSolveStart =
      linearVelAtSolveStart && angularVelAtSolveStart &&
      linearVelAtSolveStart->size() == numBodies &&
      angularVelAtSolveStart->size() == numBodies;
  if (haveSolveStart) {
    const auto solveStartPointVelocity =
        [&](physx::PxU32 body, const physx::PxVec3 &localPoint) {
          if (body >= numBodies)
            return physx::PxVec3(0.0f);
          const physx::PxVec3 arm =
              bodies[body].rotation.rotate(localPoint);
          return (*linearVelAtSolveStart)[body] +
                 (*angularVelAtSolveStart)[body].cross(arm);
        };
    physx::PxVec3 solveStartA =
        solveStartPointVelocity(bodyA, contact.contactPointA);
    physx::PxVec3 solveStartB =
        solveStartPointVelocity(bodyB, contact.contactPointB);
    if (isBodyVsStaticContact(bodyA, bodyB, numBodies) &&
        hasDeformableStaticAnchor(contact)) {
      const physx::PxVec3 staticNow =
          dynamicA ? contact.contactPointB : contact.contactPointA;
      const physx::PxVec3 staticVelocity =
          (staticNow - contact.staticPrevWorldPoint) / dt;
      if (dynamicA)
        solveStartB = staticVelocity;
      else
        solveStartA = staticVelocity;
    }
    const physx::PxReal solveStartRelative =
        (solveStartA - solveStartB).dot(contact.contactNormal);
    if (!physx::PxIsFinite(solveStartRelative)) {
      snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
      snapshot.unsupported = 1;
      return snapshot;
    }
    snapshot.fastImpact =
        -solveStartRelative > kBodyStaticFastImpactSpeed ? 1 : 0;
  }

  physx::PxReal budget =
      physx::PxMax(-contact.header.lambda, physx::PxReal(0.0f)) * dt;
  if (contact.maxImpulse < PX_MAX_REAL)
    budget = physx::PxMin(
        budget, physx::PxMax(contact.maxImpulse, physx::PxReal(0.0f)));
  snapshot.maximumImpulse = budget;
  if (!physx::PxIsFinite(budget) || budget < 0.0f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }
  if (outwardVelocity <= velocityTolerance) {
    snapshot.classification = eBUDGET_DIAG_NO_CORRECTION;
    return snapshot;
  }

  physx::PxReal response = 0.0f;
  const auto addResponse =
      [&](physx::PxU32 body, const physx::PxVec3 &localPoint,
          const physx::PxVec3 &axis, physx::PxReal linearScale,
          physx::PxReal angularScale) {
        if (body >= numBodies)
          return true;
        if (!physx::PxIsFinite(linearScale) ||
            !physx::PxIsFinite(angularScale) || linearScale < 0.0f ||
            angularScale < 0.0f)
          return false;
        const physx::PxVec3 arm =
            bodies[body].rotation.rotate(localPoint);
        const physx::PxVec3 angularJacobian = arm.cross(axis);
        response += bodies[body].invMass * linearScale +
                    angularJacobian.dot(
                        bodies[body].invInertiaWorld.transform(
                            angularJacobian)) *
                        angularScale;
        return true;
      };
  if (!addResponse(bodyA, contact.contactPointA,
                   contact.contactNormal, contact.invMassScaleA,
                   contact.invInertiaScaleA) ||
      !addResponse(bodyB, contact.contactPointB,
                   -contact.contactNormal, contact.invMassScaleB,
                   contact.invInertiaScaleB) ||
      !physx::PxIsFinite(response) || response <= 1.0e-12f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }

  const physx::PxReal requiredImpulse = outwardVelocity / response;
  if (!physx::PxIsFinite(requiredImpulse) ||
      !physx::PxIsFinite(budget) || budget < 0.0f) {
    snapshot.classification = eBUDGET_DIAG_UNSUPPORTED;
    snapshot.unsupported = 1;
    return snapshot;
  }
  if (budget <= 1.0e-8f) {
    snapshot.classification = eBUDGET_DIAG_ZERO_BUDGET_REQUIRED;
    return snapshot;
  }
  const physx::PxReal impulseTolerance =
      1.0e-6f *
      physx::PxMax(physx::PxReal(1.0f),
                   physx::PxMax(requiredImpulse, budget));
  snapshot.classification = physx::PxU8(
      requiredImpulse <= budget + impulseTolerance
          ? eBUDGET_DIAG_WITHIN_BUDGET
          : eBUDGET_DIAG_OVER_BUDGET);
  return snapshot;
}

static bool isSurfaceFinalizeContactNear(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint &contact) {
  const physx::PxU32 bodyA = contact.header.bodyIndexA;
  const physx::PxU32 bodyB = contact.header.bodyIndexB;
  if (bodyA >= numBodies && bodyB >= numBodies)
    return false;
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
  physx::PxReal violation =
      (worldA - worldB).dot(contact.contactNormal) +
      contact.penetrationDepth;
  if (isBodyVsStaticContact(bodyA, bodyB, numBodies) &&
      hasDeformableStaticAnchor(contact)) {
    violation = finalizeBodyVsStaticViolation(
        violation, contact.penetrationDepth);
  }
  return physx::PxIsFinite(violation) &&
         violation < AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE;
}

// Discover the strict P3E/P3K owner before P3K mutates velocity.  This is the
// extracted control predicate of the legacy manifold diagnostic below:
// dominant deformable/static contact, near-surface capability, non-fast
// solve-start COM approach, and at least one near position-tangent-owned row.
// Keeping the legacy marker separately lets the hidden gate prove exact
// equivalence before any production owner replacement is attempted.
static void discoverSurfaceFinalizeStrictOwnersPreP3K(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    physx::PxArray<SurfaceFinalizeTopologyNode> &nodes) {
  if (!bodies || !contacts || nodes.size() != numBodies)
    return;

  const bool haveSolveStart =
      linearVelAtSolveStart &&
      linearVelAtSolveStart->size() == numBodies;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    nodes[body].bodyStrictOwner = 0;
    if (bodies[body].invMass <= 0.0f)
      continue;

    physx::PxU32 dominant = PX_MAX_U32;
    physx::PxReal worstViolation = PX_MAX_REAL;
    for (physx::PxU32 row = 0; row < numContacts; ++row) {
      const AvbdContactConstraint &contact = contacts[row];
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      if (!isBodyVsStaticContact(bodyA, bodyB, numBodies) ||
          (bodyA != body && bodyB != body))
        continue;
      const bool dynamicIsA = bodyA == body;
      const physx::PxVec3 worldA =
          dynamicIsA
              ? bodies[body].position +
                    bodies[body].rotation.rotate(contact.contactPointA)
              : contact.contactPointA;
      const physx::PxVec3 worldB =
          dynamicIsA
              ? contact.contactPointB
              : bodies[body].position +
                    bodies[body].rotation.rotate(contact.contactPointB);
      physx::PxReal violation =
          (worldA - worldB).dot(contact.contactNormal) +
          contact.penetrationDepth;
      if (hasDeformableStaticAnchor(contact))
        violation = finalizeBodyVsStaticViolation(
            violation, contact.penetrationDepth);
      if (violation < worstViolation) {
        worstViolation = violation;
        dominant = row;
      }
    }
    if (dominant == PX_MAX_U32)
      continue;

    const AvbdContactConstraint &dominantContact = contacts[dominant];
    if (!hasDeformableStaticAnchor(dominantContact) ||
        worstViolation >= AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE)
      continue;
    if (haveSolveStart) {
      const bool dynamicIsA =
          dominantContact.header.bodyIndexA == body;
      const physx::PxVec3 outwardNormal =
          dominantContact.contactNormal *
          (dynamicIsA ? 1.0f : -1.0f);
      const physx::PxReal approach =
          -(*linearVelAtSolveStart)[body].dot(outwardNormal);
      if (approach > kBodyStaticFastImpactSpeed)
        continue;
    }

    for (physx::PxU32 row = 0; row < numContacts; ++row) {
      const AvbdContactConstraint &contact = contacts[row];
      if ((contact.header.bodyIndexA != body &&
           contact.header.bodyIndexB != body) ||
          !isBodyVsStaticContact(
              contact.header.bodyIndexA, contact.header.bodyIndexB,
              numBodies) ||
          !hasDeformableStaticAnchor(contact) ||
          !hasDeformablePositionTangentOwner(contact) ||
          !isSurfaceFinalizeContactNear(bodies, numBodies, contact))
        continue;
      nodes[body].bodyStrictOwner = 1;
      break;
    }
  }
}

static physx::PxU32 findFinalizeComponentRoot(
    physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    physx::PxU32 body) {
  physx::PxU32 root = body;
  while (nodes[root].parent != root)
    root = nodes[root].parent;
  while (nodes[body].parent != body) {
    const physx::PxU32 next = nodes[body].parent;
    nodes[body].parent = root;
    body = next;
  }
  return root;
}

static void recordSurfaceDeformableFinalizeComponentTopology(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxArray<SurfaceFinalizeTopologyNode> &nodes,
    const physx::PxArray<SurfaceFinalizeBudgetDiagSnapshot>
        &budgetDiagSnapshots,
    bool hasJointConstraints, bool enableProductionProbe,
    physx::PxArray<bool> &probeOwnedBodies,
    AvbdSolverStats *stats) {
  if (!stats || nodes.size() != numBodies || numBodies == 0)
    return;

  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    nodes[body].parent = body;
    nodes[body].bodyCount = 0;
    nodes[body].rowCount = 0;
    nodes[body].firstLinearScale = 0.0f;
    nodes[body].firstAngularScale = 0.0f;
    nodes[body].restitution = 0;
    nodes[body].finiteImpulse = 0;
    nodes[body].targetVelocity = 0;
    nodes[body].mixedScale = 0;
    nodes[body].rigidStatic = 0;
    nodes[body].nonOwnerDeformable = 0;
    nodes[body].scaleSeen = 0;
    nodes[body].lockedDof = 0;
    nodes[body].nonDynamicBody = 0;
    nodes[body].fastImpact = 0;
    nodes[body].snapshotUnsupported = 0;
    nodes[body].budgetDiagNoCorrectionRows = 0;
    nodes[body].budgetDiagZeroBudgetRequiredRows = 0;
    nodes[body].budgetDiagWithinBudgetRows = 0;
    nodes[body].budgetDiagOverBudgetRows = 0;
    nodes[body].budgetDiagUnsupportedRows = 0;
  }
  physx::PxArray<physx::PxArray<physx::PxU32> > componentRows(
      numBodies);
  for (physx::PxU32 row = 0; row < numContacts; ++row) {
    if (!isSurfaceFinalizeContactNear(
            bodies, numBodies, contacts[row]))
      continue;
    const physx::PxU32 bodyA = contacts[row].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[row].header.bodyIndexB;
    if (bodyA >= numBodies || bodyB >= numBodies)
      continue;
    const physx::PxU32 rootA =
        findFinalizeComponentRoot(nodes, bodyA);
    const physx::PxU32 rootB =
        findFinalizeComponentRoot(nodes, bodyB);
    if (rootA != rootB)
      nodes[rootB].parent = rootA;
  }
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    nodes[body].parent = findFinalizeComponentRoot(nodes, body);

  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 root = nodes[body].parent;
    ++nodes[root].bodyCount;
    if (nodes[body].bodyStrictOwner)
      nodes[root].strictOwner = 1;
    if (bodies[body].lockFlags != 0)
      nodes[root].lockedDof = 1;
    if (bodies[body].invMass <= 0.0f)
      nodes[root].nonDynamicBody = 1;
  }

  for (physx::PxU32 row = 0; row < numContacts; ++row) {
    const AvbdContactConstraint &contact = contacts[row];
    if (!isSurfaceFinalizeContactNear(
            bodies, numBodies, contact))
      continue;
    const physx::PxU32 bodyA = contact.header.bodyIndexA;
    const physx::PxU32 bodyB = contact.header.bodyIndexB;
    const physx::PxU32 dynamicBody =
        bodyA < numBodies ? bodyA : bodyB;
    if (dynamicBody >= numBodies)
      continue;
    const physx::PxU32 root = nodes[dynamicBody].parent;
    if (!nodes[root].strictOwner)
      continue;

    ++nodes[root].rowCount;
    componentRows[root].pushBack(row);
    const SurfaceFinalizeBudgetDiagSnapshot snapshot =
        row < budgetDiagSnapshots.size()
            ? budgetDiagSnapshots[row]
            : SurfaceFinalizeBudgetDiagSnapshot();
    const physx::PxU8 budgetClass =
        row < budgetDiagSnapshots.size()
            ? snapshot.classification
            : physx::PxU8(eBUDGET_DIAG_UNSUPPORTED);
    if (snapshot.fastImpact)
      nodes[root].fastImpact = 1;
    if (snapshot.unsupported ||
        budgetClass == eBUDGET_DIAG_UNSUPPORTED)
      nodes[root].snapshotUnsupported = 1;
    switch (budgetClass) {
    case eBUDGET_DIAG_NO_CORRECTION:
      ++nodes[root].budgetDiagNoCorrectionRows;
      break;
    case eBUDGET_DIAG_ZERO_BUDGET_REQUIRED:
      ++nodes[root].budgetDiagZeroBudgetRequiredRows;
      break;
    case eBUDGET_DIAG_WITHIN_BUDGET:
      ++nodes[root].budgetDiagWithinBudgetRows;
      break;
    case eBUDGET_DIAG_OVER_BUDGET:
      ++nodes[root].budgetDiagOverBudgetRows;
      break;
    default:
      ++nodes[root].budgetDiagUnsupportedRows;
      break;
    }
    if (contact.restitution > 0.0f)
      nodes[root].restitution = 1;
    if (contact.maxImpulse < PX_MAX_REAL)
      nodes[root].finiteImpulse = 1;
    if (contact.targetVelocity.magnitudeSquared() > 1.0e-12f)
      nodes[root].targetVelocity = 1;
    if (isBodyVsStaticContact(bodyA, bodyB, numBodies)) {
      if (!hasDeformableStaticAnchor(contact))
        nodes[root].rigidStatic = 1;
      else if (!hasDeformablePositionTangentOwner(contact))
        nodes[root].nonOwnerDeformable = 1;
    }

    const auto recordScale =
        [&](physx::PxU32 body, physx::PxReal linearScale,
            physx::PxReal angularScale) {
          if (body >= numBodies)
            return;
          SurfaceFinalizeTopologyNode &bodyNode = nodes[body];
          if (!bodyNode.scaleSeen) {
            bodyNode.scaleSeen = 1;
            bodyNode.firstLinearScale = linearScale;
            bodyNode.firstAngularScale = angularScale;
            return;
          }
          const physx::PxReal linearTolerance =
              1.0e-6f *
              physx::PxMax(
                  physx::PxReal(1.0f),
                  physx::PxMax(
                      physx::PxAbs(bodyNode.firstLinearScale),
                               physx::PxAbs(linearScale)));
          const physx::PxReal angularTolerance =
              1.0e-6f *
              physx::PxMax(
                  physx::PxReal(1.0f),
                  physx::PxMax(
                      physx::PxAbs(bodyNode.firstAngularScale),
                               physx::PxAbs(angularScale)));
          if (physx::PxAbs(
                  linearScale - bodyNode.firstLinearScale) >
                  linearTolerance ||
              physx::PxAbs(
                  angularScale - bodyNode.firstAngularScale) >
                  angularTolerance)
            nodes[root].mixedScale = 1;
        };
    recordScale(bodyA, contact.invMassScaleA,
                contact.invInertiaScaleA);
    recordScale(bodyB, contact.invMassScaleB,
                contact.invInertiaScaleB);
  }

  for (physx::PxU32 root = 0; root < numBodies; ++root) {
    const SurfaceFinalizeTopologyNode &component = nodes[root];
    if (component.parent != root || !component.strictOwner)
      continue;
    PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowComponents);
    const bool shadowUnsupported =
        component.restitution || component.targetVelocity ||
        component.mixedScale || component.rigidStatic ||
        component.nonOwnerDeformable || hasJointConstraints ||
        component.lockedDof || component.nonDynamicBody ||
        component.fastImpact || component.snapshotUnsupported;
    if (shadowUnsupported) {
      PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowUnsupported);
      continue;
    }

    physx::PxArray<physx::PxU32> orderedRows = componentRows[root];
    std::sort(
        orderedRows.begin(), orderedRows.end(),
        [&](physx::PxU32 lhs, physx::PxU32 rhs) {
          const AvbdContactConstraint &a = contacts[lhs];
          const AvbdContactConstraint &b = contacts[rhs];
          if (a.cacheKey != b.cacheKey)
            return a.cacheKey < b.cacheKey;
          const physx::PxU32 aMin =
              physx::PxMin(a.header.bodyIndexA, a.header.bodyIndexB);
          const physx::PxU32 bMin =
              physx::PxMin(b.header.bodyIndexA, b.header.bodyIndexB);
          if (aMin != bMin)
            return aMin < bMin;
          const physx::PxU32 aMax =
              physx::PxMax(a.header.bodyIndexA, a.header.bodyIndexB);
          const physx::PxU32 bMax =
              physx::PxMax(b.header.bodyIndexA, b.header.bodyIndexB);
          if (aMax != bMax)
            return aMax < bMax;
          const physx::PxReal aValues[9] = {
              a.contactNormal.x, a.contactNormal.y, a.contactNormal.z,
              a.contactPointA.x, a.contactPointA.y, a.contactPointA.z,
              a.contactPointB.x, a.contactPointB.y, a.contactPointB.z};
          const physx::PxReal bValues[9] = {
              b.contactNormal.x, b.contactNormal.y, b.contactNormal.z,
              b.contactPointA.x, b.contactPointA.y, b.contactPointA.z,
              b.contactPointB.x, b.contactPointB.y, b.contactPointB.z};
          for (physx::PxU32 value = 0; value < 9; ++value) {
            if (aValues[value] != bValues[value])
              return aValues[value] < bValues[value];
          }
          return lhs < rhs;
        });

    const physx::PxU32 rowCount = orderedRows.size();
    physx::PxArray<double> outward(rowCount, 0.0);
    physx::PxArray<double> upperBounds(rowCount, 0.0);
    bool assemblyValid = rowCount == component.rowCount;
    for (physx::PxU32 row = 0; row < rowCount && assemblyValid; ++row) {
      const physx::PxU32 contactIndex = orderedRows[row];
      if (contactIndex >= budgetDiagSnapshots.size()) {
        assemblyValid = false;
        break;
      }
      outward[row] =
          double(budgetDiagSnapshots[contactIndex].outwardVelocity);
      upperBounds[row] =
          double(budgetDiagSnapshots[contactIndex].maximumImpulse);
      if (!std::isfinite(outward[row]) ||
          !std::isfinite(upperBounds[row]) ||
          upperBounds[row] < 0.0) {
        assemblyValid = false;
        break;
      }
    }

    if (!assemblyValid) {
      continue;
    }
    const bool useMatrixFreeBackend = rowCount > 128;
    AvbdBoundedProjectionResult shadow;
    if (useMatrixFreeBackend) {
      shadow = solveSurfaceFinalizeMatrixFreeBoundedProjection(
          bodies, numBodies, nodes, root, contacts, orderedRows,
          outward, upperBounds);
    } else {
      physx::PxArray<double> response(rowCount * rowCount, 0.0);
      for (physx::PxU32 row = 0;
           row < rowCount && assemblyValid; ++row) {
        const AvbdContactConstraint &a =
            contacts[orderedRows[row]];
        for (physx::PxU32 column = 0;
             column < rowCount; ++column) {
          const AvbdContactConstraint &b =
              contacts[orderedRows[column]];
          double value = 0.0;
          const physx::PxU32 aBodies[2] = {
              a.header.bodyIndexA, a.header.bodyIndexB};
          const physx::PxU32 bBodies[2] = {
              b.header.bodyIndexA, b.header.bodyIndexB};
          const physx::PxVec3 aPoints[2] = {
              a.contactPointA, a.contactPointB};
          const physx::PxVec3 bPoints[2] = {
              b.contactPointA, b.contactPointB};
          const physx::PxVec3 aAxes[2] = {
              a.contactNormal, -a.contactNormal};
          const physx::PxVec3 bAxes[2] = {
              b.contactNormal, -b.contactNormal};
          for (physx::PxU32 aEnd = 0; aEnd < 2; ++aEnd) {
            const physx::PxU32 body = aBodies[aEnd];
            if (body >= numBodies)
              continue;
            for (physx::PxU32 bEnd = 0; bEnd < 2; ++bEnd) {
              if (bBodies[bEnd] != body)
                continue;
              const SurfaceFinalizeTopologyNode &bodyNode =
                  nodes[body];
              const physx::PxVec3 aArm =
                  bodies[body].rotation.rotate(aPoints[aEnd]);
              const physx::PxVec3 bArm =
                  bodies[body].rotation.rotate(bPoints[bEnd]);
              const physx::PxVec3 aAngular =
                  aArm.cross(aAxes[aEnd]);
              const physx::PxVec3 bAngular =
                  bArm.cross(bAxes[bEnd]);
              value +=
                  double(bodies[body].invMass *
                         bodyNode.firstLinearScale *
                         aAxes[aEnd].dot(bAxes[bEnd])) +
                  double(aAngular.dot(
                             bodies[body].invInertiaWorld.transform(
                                 bAngular)) *
                         bodyNode.firstAngularScale);
            }
          }
          if (!std::isfinite(value)) {
            assemblyValid = false;
            break;
          }
          response[row * rowCount + column] = value;
        }
      }
      if (!assemblyValid) {
        continue;
      }
      shadow = solveAvbdBoundedProjection(
          response, outward, upperBounds, 6 * component.bodyCount);
    }
    switch (shadow.status) {
    case eAVBD_BOUNDED_SOLVED:
      PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowSolved);
      if (enableProductionProbe &&
          shadow.commitImpulses.size() == rowCount &&
          probeOwnedBodies.size() == numBodies) {
        physx::PxArray<physx::PxVec3> linearImpulses(
            numBodies, physx::PxVec3(0.0f));
        physx::PxArray<physx::PxVec3> angularImpulses(
            numBodies, physx::PxVec3(0.0f));
        bool commitValid = true;
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          const double candidate = shadow.commitImpulses[row];
          if (!std::isfinite(candidate) || candidate < 0.0 ||
              candidate > double(PX_MAX_REAL)) {
            commitValid = false;
            break;
          }
          const physx::PxReal impulse = physx::PxReal(candidate);
          const AvbdContactConstraint &contact =
              contacts[orderedRows[row]];
          const physx::PxU32 rowBodies[2] = {
              contact.header.bodyIndexA, contact.header.bodyIndexB};
          const physx::PxVec3 rowPoints[2] = {
              contact.contactPointA, contact.contactPointB};
          const physx::PxVec3 rowAxes[2] = {
              contact.contactNormal, -contact.contactNormal};
          for (physx::PxU32 end = 0; end < 2; ++end) {
            const physx::PxU32 body = rowBodies[end];
            if (body >= numBodies)
              continue;
            const physx::PxVec3 arm =
                bodies[body].rotation.rotate(rowPoints[end]);
            linearImpulses[body] += rowAxes[end] * impulse;
            angularImpulses[body] +=
                arm.cross(rowAxes[end]) * impulse;
          }
        }
        physx::PxArray<physx::PxVec3> linearDeltas(
            numBodies, physx::PxVec3(0.0f));
        physx::PxArray<physx::PxVec3> angularDeltas(
            numBodies, physx::PxVec3(0.0f));
        if (commitValid) {
          for (physx::PxU32 body = 0; body < numBodies; ++body) {
            if (nodes[body].parent != root)
              continue;
            const SurfaceFinalizeTopologyNode &bodyNode = nodes[body];
            linearDeltas[body] =
                linearImpulses[body] *
                (bodies[body].invMass * bodyNode.firstLinearScale);
            angularDeltas[body] =
                bodies[body].invInertiaWorld.transform(
                    angularImpulses[body]) *
                bodyNode.firstAngularScale;
            if (!linearDeltas[body].isFinite() ||
                !angularDeltas[body].isFinite()) {
              commitValid = false;
              break;
            }
          }
        }
        if (commitValid) {
          double velocityScale = 1.0;
          for (physx::PxU32 row = 0; row < rowCount; ++row)
            velocityScale =
                std::max(velocityScale, std::fabs(outward[row]));
          const double residualTolerance = 8.0e-6 * velocityScale;
          for (physx::PxU32 row = 0; row < rowCount; ++row) {
            const AvbdContactConstraint &contact =
                contacts[orderedRows[row]];
            const physx::PxU32 rowBodies[2] = {
                contact.header.bodyIndexA, contact.header.bodyIndexB};
            const physx::PxVec3 rowPoints[2] = {
                contact.contactPointA, contact.contactPointB};
            const physx::PxVec3 rowAxes[2] = {
                contact.contactNormal, -contact.contactNormal};
            double responseDelta = 0.0;
            for (physx::PxU32 end = 0; end < 2; ++end) {
              const physx::PxU32 body = rowBodies[end];
              if (body >= numBodies)
                continue;
              const physx::PxVec3 arm =
                  bodies[body].rotation.rotate(rowPoints[end]);
              const physx::PxVec3 pointDelta =
                  linearDeltas[body] +
                  angularDeltas[body].cross(arm);
              responseDelta +=
                  double(pointDelta.dot(rowAxes[end]));
            }
            const double postOutward =
                outward[row] - responseDelta;
            if (!std::isfinite(postOutward) ||
                postOutward > residualTolerance) {
              commitValid = false;
              break;
            }
          }
        }
        if (commitValid) {
          physx::PxU32 committedBodies = 0;
          physx::PxU32 replacedOwners = 0;
          for (physx::PxU32 body = 0; body < numBodies; ++body) {
            if (nodes[body].parent != root)
              continue;
            bodies[body].linearVelocity -= linearDeltas[body];
            bodies[body].angularVelocity -= angularDeltas[body];
            probeOwnedBodies[body] = true;
            ++committedBodies;
            if (nodes[body].bodyStrictOwner)
              ++replacedOwners;
          }
          PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeProbeCommittedComponents);
        }
      }
      break;
    case eAVBD_BOUNDED_NO_CORRECTION:
      break;
    case eAVBD_BOUNDED_BUDGET_EXHAUSTED:
      break;
    case eAVBD_BOUNDED_INFEASIBLE:
      break;
    case eAVBD_BOUNDED_RESIDUAL_UNCLASSIFIED:
      break;
    case eAVBD_BOUNDED_ITERATION_LIMIT:
      PX_AVBD_PROFILE_STAT(++stats->surfaceDeformableFinalizeShadowIterationLimit);
      break;
    default:
      break;
    }
  }
}

struct AvbdRigidRestitutionRow {
  physx::PxU32 contactIndex;
  physx::PxVec3 axis[2];
  physx::PxVec3 angularJacobian[2];
  physx::PxVec3 positionAlAngularJacobian[2];
  double alImpulse;
  double q;
};

/**
 * Apply the non-associated Coulomb proximal map to one trial impulse.
 *
 * The unilateral normal is clamped first and is therefore independent of
 * tangential demand.  The tangent pair is then projected onto the disk whose
 * radius is mu times that accepted normal impulse.  This ordering is the
 * fixed-point form of Signorini normal complementarity plus Coulomb maximum
 * dissipation.  An Euclidean projection of all three values onto a second
 * order cone is not equivalent: on a sliding contact its associated KKT law
 * creates a positive normal separation velocity proportional to slip speed.
 */
static bool projectAvbdCoulombNcpImpulse(
    double mu, double &normal, double &tangent0, double &tangent1) {
  if (!std::isfinite(mu) || mu < 0.0 || !std::isfinite(normal) ||
      !std::isfinite(tangent0) || !std::isfinite(tangent1))
    return false;

  normal = std::max(0.0, normal);
  const double tangentMagnitude =
      std::sqrt(tangent0 * tangent0 + tangent1 * tangent1);
  if (!std::isfinite(tangentMagnitude))
    return false;
  const double tangentLimit = mu * normal;
  if (!std::isfinite(tangentLimit))
    return false;
  if (tangentMagnitude <= tangentLimit)
    return true;
  if (tangentLimit <= 0.0) {
    tangent0 = 0.0;
    tangent1 = 0.0;
    return true;
  }

  const double tangentScale = tangentLimit / tangentMagnitude;
  tangent0 *= tangentScale;
  tangent1 *= tangentScale;
  return std::isfinite(normal) && std::isfinite(tangent0) &&
         std::isfinite(tangent1);
}

/**
 * Solve the non-associated rigid Coulomb NCP with synchronized Jacobi maps.
 *
 * The explicit layout owns one unilateral row per geometric contact and one
 * tangent disk per persistent friction anchor.  Each disk reads the normal
 * load assigned to its pressure partition.  Every row reads the same old
 * impulse vector, so a converged fixed point satisfies nonnegative normal
 * complementarity and tangential maximum dissipation without a contact-order
 * sweep. `impulses` is an in/out warm start and `scratch` has rowCount entries.
 */
static bool solveAvbdCoulombNcpFixedPoint(
    const double *response, const double *q,
    const AvbdCoulombNcpLayout &layout, double *impulses,
    double *scratch) {
  if (!response || !q || !impulses || !scratch ||
      layout.rowCount == 0u || layout.normalCount == 0u ||
      layout.patchCount == 0u || !layout.normalRows ||
      !layout.normalPatches ||
      (layout.anchorCount > 0u &&
       (!layout.tangentRows || !layout.tangentPatches)) ||
      !layout.patchAnchorCounts || !layout.patchFriction ||
      layout.anchorCount > PX_MAX_U32 / 2u ||
      layout.normalCount > PX_MAX_U32 - 2u * layout.anchorCount ||
      layout.rowCount != layout.normalCount + 2u * layout.anchorCount ||
      layout.rowCount > 12u || layout.patchCount > 4u)
    return false;
  const physx::PxU32 rowCount = layout.rowCount;

  physx::PxU8 rowOwned[12] = {};
  physx::PxU32 actualPatchAnchorCounts[4] = {};
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    rowOwned[row] = 0u;
  for (physx::PxU32 patch = 0; patch < layout.patchCount; ++patch) {
    actualPatchAnchorCounts[patch] = 0u;
    if (!std::isfinite(layout.patchFriction[patch]) ||
        layout.patchFriction[patch] < 0.0)
      return false;
  }
  for (physx::PxU32 normal = 0; normal < layout.normalCount; ++normal) {
    const physx::PxU32 row = layout.normalRows[normal];
    const physx::PxU32 patch = layout.normalPatches[normal];
    if (row >= rowCount || patch >= layout.patchCount || rowOwned[row])
      return false;
    rowOwned[row] = 1u;
  }
  for (physx::PxU32 anchor = 0; anchor < layout.anchorCount; ++anchor) {
    const physx::PxU32 row = layout.tangentRows[anchor];
    const physx::PxU32 patch = layout.tangentPatches[anchor];
    if (row >= rowCount || row + 1u >= rowCount ||
        patch >= layout.patchCount || rowOwned[row] ||
        rowOwned[row + 1u])
      return false;
    rowOwned[row] = 1u;
    rowOwned[row + 1u] = 1u;
    ++actualPatchAnchorCounts[patch];
  }
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    if (!rowOwned[row])
      return false;
  for (physx::PxU32 patch = 0; patch < layout.patchCount; ++patch)
    if (actualPatchAnchorCounts[patch] !=
        layout.patchAnchorCounts[patch])
      return false;

  double lipschitz = 0.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (!std::isfinite(q[row]))
      return false;
    double absoluteRowSum = 0.0;
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      const double value = response[row * rowCount + column];
      if (!std::isfinite(value))
        return false;
      absoluteRowSum += std::fabs(value);
    }
    if (!std::isfinite(absoluteRowSum))
      return false;
    lipschitz = std::max(lipschitz, absoluteRowSum);
  }
  if (!std::isfinite(lipschitz) || lipschitz <= 1.0e-12)
    return false;

  double velocity[12] = {};
  double candidate[12] = {};
  double candidateVelocity[12] = {};
  double candidateMap[12] = {};
  double patchNormal[4] = {};
  const auto projectState = [&](double *state) -> bool {
    for (physx::PxU32 patch = 0; patch < layout.patchCount; ++patch)
      patchNormal[patch] = 0.0;
    for (physx::PxU32 normal = 0; normal < layout.normalCount; ++normal) {
      const physx::PxU32 row = layout.normalRows[normal];
      if (!std::isfinite(state[row]))
        return false;
      state[row] = std::max(0.0, state[row]);
      patchNormal[layout.normalPatches[normal]] += state[row];
    }
    for (physx::PxU32 patch = 0; patch < layout.patchCount; ++patch)
      if (!std::isfinite(patchNormal[patch]) || patchNormal[patch] < 0.0)
        return false;
    for (physx::PxU32 anchor = 0; anchor < layout.anchorCount; ++anchor) {
      const physx::PxU32 row = layout.tangentRows[anchor];
      const physx::PxU32 patch = layout.tangentPatches[anchor];
      double anchorNormal =
          patchNormal[patch] / double(layout.patchAnchorCounts[patch]);
      if (!projectAvbdCoulombNcpImpulse(
              layout.patchFriction[patch], anchorNormal,
              state[row], state[row + 1u]))
        return false;
    }
    return true;
  };
  if (!projectState(impulses))
    return false;
  const auto computeVelocity = [&](const double *state, double *output,
                                   double &velocityScale) -> bool {
    velocityScale = 1.0;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      double value = q[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column)
        value += response[row * rowCount + column] * state[column];
      if (!std::isfinite(value))
        return false;
      output[row] = value;
      velocityScale = std::max(
          velocityScale,
          std::max(std::fabs(q[row]),
                   std::max(std::fabs(value), std::fabs(value - q[row]))));
    }
    return std::isfinite(velocityScale);
  };
  const auto computeMap = [&](const double *state, const double *stateVelocity,
                              double stepDenominator,
                              double *output) -> bool {
    const double step = 1.0 / stepDenominator;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      output[row] = state[row] - step * stateVelocity[row];
      if (!std::isfinite(output[row]))
        return false;
    }
    return projectState(output);
  };
  const auto computeCertificate = [&](const double *state,
                                      const double *stateVelocity,
                                      double stepDenominator, double *map,
                                      double &residual) -> bool {
    if (!computeMap(state, stateVelocity, stepDenominator, map))
      return false;
    residual = 0.0;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      residual = std::max(
          residual,
          stepDenominator * std::fabs(state[row] - map[row]));
    return std::isfinite(residual);
  };

  static const physx::PxU32 kMaxIterations = 512u;
  static const physx::PxU32 kMaxRelaxations = 12u;
  static const double kRelativeFixedPointTolerance = 2.0e-6;
  double velocityScale = 1.0;
  double residual = 0.0;
  if (!computeVelocity(impulses, velocity, velocityScale) ||
      !computeCertificate(
          impulses, velocity, lipschitz, scratch, residual))
    return false;
  if (residual <= kRelativeFixedPointTolerance * velocityScale)
    return true;

  for (physx::PxU32 iteration = 0; iteration < kMaxIterations;
       ++iteration) {
    bool accepted = false;
    double acceptedResidual = residual;
    double acceptedScale = velocityScale;
    for (physx::PxU32 relaxation = 0;
         relaxation < kMaxRelaxations; ++relaxation) {
      const double weight = std::ldexp(1.0, -int(relaxation));
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        candidate[row] =
            impulses[row] + weight * (scratch[row] - impulses[row]);
      if (!computeVelocity(candidate, candidateVelocity,
                           acceptedScale) ||
          !computeCertificate(candidate, candidateVelocity,
                              lipschitz, candidateMap,
                              acceptedResidual))
        return false;
      const double residualSlack =
          1.0e-7 * std::max(1.0, residual);
      if (acceptedResidual <= residual + residualSlack ||
          acceptedResidual <=
              kRelativeFixedPointTolerance * acceptedScale) {
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      // A smaller synchronized step is obtained by growing the common
      // denominator. This changes no contact ordering and preserves the NCP
      // fixed point. Re-evaluate the current natural map strictly fresh.
      lipschitz *= 2.0;
      if (!std::isfinite(lipschitz) ||
          !computeCertificate(impulses, velocity, lipschitz, scratch,
                              residual))
        return false;
      continue;
    }
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      impulses[row] = candidate[row];
      velocity[row] = candidateVelocity[row];
      scratch[row] = candidateMap[row];
    }
    residual = acceptedResidual;
    velocityScale = acceptedScale;
    if (residual <= kRelativeFixedPointTolerance * velocityScale) {
      // The accepted residual was evaluated from a freshly assembled W*p+q,
      // not extrapolated state. Recompute once more before publishing.
      if (!computeVelocity(impulses, velocity, velocityScale) ||
          !computeCertificate(
              impulses, velocity, lipschitz, scratch, residual))
        return false;
      return residual <=
             kRelativeFixedPointTolerance * velocityScale;
    }
  }
  return false;
}

#if 0
/**
 * Solve the symmetric positive-semidefinite Signorini normal block with a
 * diagonally equilibrated, monotonically restarted accelerated projection.
 * This is a synchronized component solve: no row observes a partially
 * updated body velocity and no contact-order Gauss--Seidel sweep is used.
 */
static bool solveAvbdNormalNcpAccelerated(
    const double *response, const double *q,
    physx::PxU32 contactCount, double *impulses) {
  if (!response || !q || !impulses || contactCount == 0u ||
      contactCount > PX_MAX_U32 / contactCount)
    return false;
  physx::PxArray<double> scale(contactCount);
  physx::PxArray<double> inverseScale(contactCount);
  physx::PxArray<double> state(contactCount);
  physx::PxArray<double> previous(contactCount);
  physx::PxArray<double> extrapolated(contactCount);
  physx::PxArray<double> candidate(contactCount);
  physx::PxArray<double> gradient(contactCount);
  physx::PxArray<physx::PxU32> rowOffsets(contactCount + 1u);
  physx::PxArray<physx::PxU32> columnIndices;
  physx::PxArray<double> scaledValues;

  double maximumDiagonal = 0.0;
  for (physx::PxU32 row = 0; row < contactCount; ++row) {
    const double diagonal = response[row * contactCount + row];
    if (!std::isfinite(diagonal) || diagonal < 0.0 ||
        !std::isfinite(q[row]))
      return false;
    maximumDiagonal = std::max(maximumDiagonal, diagonal);
  }
  if (maximumDiagonal <= 1.0e-12)
    return false;
  const double diagonalFloor =
      std::max(1.0e-12, 1.0e-10 * maximumDiagonal);
  for (physx::PxU32 row = 0; row < contactCount; ++row) {
    const double diagonal = std::max(
        response[row * contactCount + row], diagonalFloor);
    scale[row] = std::sqrt(diagonal);
    inverseScale[row] = 1.0 / scale[row];
    state[row] =
        std::max(0.0, impulses[row]) * scale[row];
    previous[row] = state[row];
  }

  // The contact response is structurally sparse: two normal rows interact
  // only when they share a dynamic body. Compress the stable dense assembly
  // once, then keep every accelerated iteration on contiguous CSR arrays.
  // No associative lookup is involved.
  double lipschitz = 0.0;
  rowOffsets[0] = 0u;
  for (physx::PxU32 row = 0; row < contactCount; ++row) {
    double rowSum = 0.0;
    for (physx::PxU32 column = 0; column < contactCount; ++column) {
      const double physicalValue =
          response[row * contactCount + column];
      if (!std::isfinite(physicalValue))
        return false;
      if (physicalValue == 0.0)
        continue;
      const double value = physicalValue * inverseScale[row] *
                           inverseScale[column];
      if (!std::isfinite(value))
        return false;
      columnIndices.pushBack(column);
      scaledValues.pushBack(value);
      rowSum += std::fabs(value);
    }
    if (!std::isfinite(rowSum))
      return false;
    lipschitz = std::max(lipschitz, rowSum);
    rowOffsets[row + 1u] = columnIndices.size();
  }
  if (!std::isfinite(lipschitz) || lipschitz <= 1.0e-12)
    return false;

  const auto evaluate = [&](const physx::PxArray<double> &input,
                            physx::PxArray<double> &outputGradient,
                            double &objective) -> bool {
    objective = 0.0;
    for (physx::PxU32 row = 0; row < contactCount; ++row) {
      double responseValue = 0.0;
      for (physx::PxU32 entry = rowOffsets[row];
           entry < rowOffsets[row + 1u]; ++entry)
        responseValue +=
            scaledValues[entry] * input[columnIndices[entry]];
      const double scaledQ = q[row] * inverseScale[row];
      outputGradient[row] = responseValue + scaledQ;
      objective += input[row] *
                   (scaledQ + 0.5 * responseValue);
      if (!std::isfinite(outputGradient[row]) ||
          !std::isfinite(objective))
        return false;
    }
    return true;
  };

  const auto certified = [&](const physx::PxArray<double> &input,
                             double &residual,
                             double &velocityScale) -> bool {
    residual = 0.0;
    velocityScale = 1.0;
    for (physx::PxU32 row = 0; row < contactCount; ++row) {
      double scaledVelocity = q[row] * inverseScale[row];
      for (physx::PxU32 entry = rowOffsets[row];
           entry < rowOffsets[row + 1u]; ++entry)
        scaledVelocity +=
            scaledValues[entry] * input[columnIndices[entry]];
      const double physicalVelocity = scaledVelocity * scale[row];
      const double physicalImpulse = input[row] * inverseScale[row];
      const double rowResidual =
          physicalImpulse > 1.0e-10
              ? std::fabs(physicalVelocity)
              : std::max(0.0, -physicalVelocity);
      residual = std::max(residual, rowResidual);
      velocityScale = std::max(
          velocityScale,
          std::max(std::fabs(q[row]),
                   std::fabs(physicalVelocity)));
      if (!std::isfinite(residual) || !std::isfinite(velocityScale))
        return false;
    }
    return true;
  };

  static const physx::PxU32 kMaxIterations = 8192u;
  static const double kRelativeTolerance = 2.0e-6;
  double acceleration = 1.0;
  double currentObjective = 0.0;
  if (!evaluate(state, gradient, currentObjective))
    return false;
  for (physx::PxU32 iteration = 0; iteration < kMaxIterations;
       ++iteration) {
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        iteration == 0u ? 0.0 : (acceleration - 1.0) / nextAcceleration;
    for (physx::PxU32 row = 0; row < contactCount; ++row)
      extrapolated[row] =
          state[row] + momentum * (state[row] - previous[row]);

    double extrapolatedObjective = 0.0;
    if (!evaluate(extrapolated, gradient, extrapolatedObjective))
      return false;
    for (physx::PxU32 row = 0; row < contactCount; ++row)
      candidate[row] =
          std::max(0.0, extrapolated[row] - gradient[row] / lipschitz);

    double candidateObjective = 0.0;
    if (!evaluate(candidate, gradient, candidateObjective))
      return false;
    if (candidateObjective >
        currentObjective +
            1.0e-12 * std::max(1.0, std::fabs(currentObjective))) {
      // Monotone restart: discard only the extrapolation, not the accepted
      // iterate or the common synchronized operator.
      if (!evaluate(state, gradient, extrapolatedObjective))
        return false;
      for (physx::PxU32 row = 0; row < contactCount; ++row)
        candidate[row] =
            std::max(0.0, state[row] - gradient[row] / lipschitz);
      if (!evaluate(candidate, gradient, candidateObjective))
        return false;
      acceleration = 1.0;
    } else {
      acceleration = nextAcceleration;
    }

    previous = state;
    state = candidate;
    currentObjective = candidateObjective;
    if ((iteration & 15u) == 15u || iteration + 1u == kMaxIterations) {
      double residual = 0.0;
      double velocityScale = 1.0;
      if (!certified(state, residual, velocityScale))
        return false;
      if (residual <= kRelativeTolerance * velocityScale) {
        for (physx::PxU32 point = 0; point < contactCount; ++point) {
          impulses[point] = state[point] * inverseScale[point];
        }
        return true;
      }
    }
  }
  return false;
}
#endif

static PX_FORCE_INLINE bool projectAvbdTangentDisk(
    double limit, double &tangent0, double &tangent1) {
  if (!std::isfinite(limit) || limit < 0.0 ||
      !std::isfinite(tangent0) || !std::isfinite(tangent1))
    return false;
  const double magnitude =
      std::sqrt(tangent0 * tangent0 + tangent1 * tangent1);
  if (!std::isfinite(magnitude))
    return false;
  if (magnitude <= limit)
    return true;
  if (limit == 0.0) {
    tangent0 = 0.0;
    tangent1 = 0.0;
    return true;
  }
  const double ratio = limit / magnitude;
  tangent0 *= ratio;
  tangent1 *= ratio;
  return std::isfinite(tangent0) && std::isfinite(tangent1);
}

#if 0
/** Solve maximum-dissipation tangent rows grouped into fixed patch disks. */
static bool solveAvbdTangentDisksAccelerated(
    const double *response, const double *q, const double *limits,
    physx::PxU32 anchorCount, double *impulses) {
  if (!response || !q || !limits || !impulses || anchorCount == 0u ||
      anchorCount > PX_MAX_U32 / 2u)
    return false;
  const physx::PxU32 rowCount = anchorCount * 2u;
  if (rowCount > PX_MAX_U32 / rowCount)
    return false;

  physx::PxArray<double> scale(rowCount);
  physx::PxArray<double> inverseScale(rowCount);
  physx::PxArray<double> scaledLimit(anchorCount);
  physx::PxArray<double> state(rowCount);
  physx::PxArray<double> previous(rowCount);
  physx::PxArray<double> extrapolated(rowCount);
  physx::PxArray<double> candidate(rowCount);
  physx::PxArray<double> gradient(rowCount);
  physx::PxArray<physx::PxU32> rowOffsets(rowCount + 1u);
  physx::PxArray<physx::PxU32> columnIndices;
  physx::PxArray<double> scaledValues;

  double maximumDiagonal = 0.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (!std::isfinite(response[row * rowCount + row]) ||
        response[row * rowCount + row] < 0.0 || !std::isfinite(q[row]))
      return false;
    maximumDiagonal =
        std::max(maximumDiagonal, response[row * rowCount + row]);
  }
  if (maximumDiagonal <= 1.0e-12)
    return false;
  const double diagonalFloor =
      std::max(1.0e-12, 1.0e-10 * maximumDiagonal);
  for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
    const physx::PxU32 row = anchor * 2u;
    const double pairDiagonal = std::max(
        0.5 * (response[row * rowCount + row] +
               response[(row + 1u) * rowCount + row + 1u]),
        diagonalFloor);
    const double pairScale = std::sqrt(pairDiagonal);
    if (!std::isfinite(limits[anchor]) || limits[anchor] < 0.0)
      return false;
    scaledLimit[anchor] = limits[anchor] * pairScale;
    for (physx::PxU32 component = 0; component < 2u; ++component) {
      scale[row + component] = pairScale;
      inverseScale[row + component] = 1.0 / pairScale;
      state[row + component] = impulses[row + component] * pairScale;
      previous[row + component] = state[row + component];
    }
    if (!projectAvbdTangentDisk(
            scaledLimit[anchor], state[row], state[row + 1u]))
      return false;
    previous[row] = state[row];
    previous[row + 1u] = state[row + 1u];
  }

  double lipschitz = 0.0;
  rowOffsets[0] = 0u;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    double rowSum = 0.0;
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      const double physicalValue = response[row * rowCount + column];
      if (!std::isfinite(physicalValue))
        return false;
      if (physicalValue == 0.0)
        continue;
      const double value = physicalValue * inverseScale[row] *
                           inverseScale[column];
      if (!std::isfinite(value))
        return false;
      columnIndices.pushBack(column);
      scaledValues.pushBack(value);
      rowSum += std::fabs(value);
    }
    if (!std::isfinite(rowSum))
      return false;
    lipschitz = std::max(lipschitz, rowSum);
    rowOffsets[row + 1u] = columnIndices.size();
  }
  if (!std::isfinite(lipschitz) || lipschitz <= 1.0e-12)
    return false;

  const auto evaluate = [&](const physx::PxArray<double> &input,
                            physx::PxArray<double> &outputGradient,
                            double &objective) -> bool {
    objective = 0.0;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      double responseValue = 0.0;
      for (physx::PxU32 entry = rowOffsets[row];
           entry < rowOffsets[row + 1u]; ++entry)
        responseValue +=
            scaledValues[entry] * input[columnIndices[entry]];
      const double scaledQ = q[row] * inverseScale[row];
      outputGradient[row] = responseValue + scaledQ;
      objective += input[row] * (scaledQ + 0.5 * responseValue);
      if (!std::isfinite(outputGradient[row]) ||
          !std::isfinite(objective))
        return false;
    }
    return true;
  };
  const auto projectState = [&](physx::PxArray<double> &value) -> bool {
    for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
      const physx::PxU32 row = anchor * 2u;
      if (!projectAvbdTangentDisk(
              scaledLimit[anchor], value[row], value[row + 1u]))
        return false;
    }
    return true;
  };
  const auto certified = [&](const physx::PxArray<double> &input,
                             double &residual,
                             double &velocityScale) -> bool {
    double unusedObjective = 0.0;
    if (!evaluate(input, gradient, unusedObjective))
      return false;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      candidate[row] = input[row] - gradient[row] / lipschitz;
    if (!projectState(candidate))
      return false;
    residual = 0.0;
    velocityScale = 1.0;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      residual = std::max(
          residual,
          scale[row] * lipschitz *
              std::fabs(input[row] - candidate[row]));
      velocityScale = std::max(
          velocityScale,
          std::max(std::fabs(q[row]),
                   std::fabs(gradient[row] * scale[row])));
    }
    return std::isfinite(residual) && std::isfinite(velocityScale);
  };

  static const physx::PxU32 kMaxIterations = 4096u;
  static const double kRelativeTolerance = 2.0e-6;
  double acceleration = 1.0;
  double currentObjective = 0.0;
  if (!evaluate(state, gradient, currentObjective))
    return false;
  for (physx::PxU32 iteration = 0; iteration < kMaxIterations;
       ++iteration) {
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        iteration == 0u ? 0.0 : (acceleration - 1.0) / nextAcceleration;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      extrapolated[row] =
          state[row] + momentum * (state[row] - previous[row]);
    double ignoredObjective = 0.0;
    if (!evaluate(extrapolated, gradient, ignoredObjective))
      return false;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      candidate[row] =
          extrapolated[row] - gradient[row] / lipschitz;
    if (!projectState(candidate))
      return false;
    double candidateObjective = 0.0;
    if (!evaluate(candidate, gradient, candidateObjective))
      return false;
    if (candidateObjective >
        currentObjective +
            1.0e-12 * std::max(1.0, std::fabs(currentObjective))) {
      if (!evaluate(state, gradient, ignoredObjective))
        return false;
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        candidate[row] = state[row] - gradient[row] / lipschitz;
      if (!projectState(candidate) ||
          !evaluate(candidate, gradient, candidateObjective))
        return false;
      acceleration = 1.0;
    } else {
      acceleration = nextAcceleration;
    }
    previous = state;
    state = candidate;
    currentObjective = candidateObjective;
    if ((iteration & 15u) == 15u || iteration + 1u == kMaxIterations) {
      double residual = 0.0;
      double velocityScale = 1.0;
      if (!certified(state, residual, velocityScale))
        return false;
      if (residual <= kRelativeTolerance * velocityScale) {
        for (physx::PxU32 row = 0; row < rowCount; ++row)
          impulses[row] = state[row] * inverseScale[row];
        return true;
      }
    }
  }
  return false;
}
#endif

/**
 * Solve a convex projected material subproblem through a matrix-free
 * response operator.  Diagonal equilibration keeps the projection in physical
 * impulse units while monotone FISTA backtracking obtains a safe spectral
 * step without assembling W, scanning an N-by-N matrix, or using a hash.
 */
template <typename Multiply, typename Project>
static bool solveAvbdProjectedQuadraticMatrixFree(
    const Multiply &multiply, const Project &project,
    const double *q, const double *diagonal, physx::PxU32 rowCount,
    double *impulses, physx::PxU32 maximumIterations = 4096u,
    double relativeTolerance = 2.0e-6) {
  if (!q || !diagonal || !impulses || rowCount == 0u ||
      maximumIterations == 0u || !std::isfinite(relativeTolerance) ||
      relativeTolerance <= 0.0)
    return false;

  physx::PxArray<double> scale(rowCount);
  physx::PxArray<double> inverseScale(rowCount);
  physx::PxArray<double> state(rowCount);
  physx::PxArray<double> previous(rowCount);
  physx::PxArray<double> extrapolated(rowCount);
  physx::PxArray<double> candidate(rowCount);
  physx::PxArray<double> gradient(rowCount);
  physx::PxArray<double> candidateGradient(rowCount);
  physx::PxArray<double> physicalInput(rowCount);
  physx::PxArray<double> physicalResponse(rowCount);

  double maximumDiagonal = 0.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    if (!std::isfinite(diagonal[row]) || diagonal[row] < 0.0 ||
        !std::isfinite(q[row]) || !std::isfinite(impulses[row]))
      return false;
    maximumDiagonal = std::max(maximumDiagonal, diagonal[row]);
  }
  if (maximumDiagonal <= 1.0e-12)
    return false;
  const double diagonalFloor =
      std::max(1.0e-12, 1.0e-10 * maximumDiagonal);
  for (physx::PxU32 row = 0; row < rowCount; ++row) {
    scale[row] = std::sqrt(std::max(diagonal[row], diagonalFloor));
    inverseScale[row] = 1.0 / scale[row];
    state[row] = impulses[row] * scale[row];
    previous[row] = state[row];
  }
  if (!project(state))
    return false;
  previous = state;

  const auto evaluate = [&](const physx::PxArray<double> &input,
                            physx::PxArray<double> &outputGradient,
                            double &objective) -> bool {
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      physicalInput[row] = input[row] * inverseScale[row];
      if (!std::isfinite(physicalInput[row]))
        return false;
    }
    if (!multiply(physicalInput, physicalResponse))
      return false;
    objective = 0.0;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      const double scaledQ = q[row] * inverseScale[row];
      const double scaledResponse =
          physicalResponse[row] * inverseScale[row];
      outputGradient[row] = scaledQ + scaledResponse;
      objective += input[row] * (scaledQ + 0.5 * scaledResponse);
      if (!std::isfinite(outputGradient[row]) ||
          !std::isfinite(objective))
        return false;
    }
    return true;
  };

  double currentObjective = 0.0;
  if (!evaluate(state, gradient, currentObjective))
    return false;
  double lipschitz = 1.0;
  double acceleration = 1.0;
  static const physx::PxU32 kMaximumBacktracks = 48u;

  for (physx::PxU32 iteration = 0; iteration < maximumIterations;
       ++iteration) {
    const double nextAcceleration =
        0.5 * (1.0 + std::sqrt(1.0 + 4.0 * acceleration * acceleration));
    const double momentum =
        iteration == 0u ? 0.0 : (acceleration - 1.0) / nextAcceleration;
    for (physx::PxU32 row = 0; row < rowCount; ++row)
      extrapolated[row] =
          state[row] + momentum * (state[row] - previous[row]);

    double extrapolatedObjective = 0.0;
    if (!evaluate(extrapolated, gradient, extrapolatedObjective))
      return false;

    bool accepted = false;
    double candidateObjective = 0.0;
    for (physx::PxU32 backtrack = 0;
         backtrack < kMaximumBacktracks; ++backtrack) {
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        candidate[row] =
            extrapolated[row] - gradient[row] / lipschitz;
      if (!project(candidate) ||
          !evaluate(candidate, candidateGradient, candidateObjective))
        return false;

      double model = extrapolatedObjective;
      double deltaMagnitudeSq = 0.0;
      for (physx::PxU32 row = 0; row < rowCount; ++row) {
        const double delta = candidate[row] - extrapolated[row];
        model += gradient[row] * delta;
        deltaMagnitudeSq += delta * delta;
      }
      model += 0.5 * lipschitz * deltaMagnitudeSq;
      const double modelTolerance =
          1.0e-12 * std::max(
              1.0, std::max(std::fabs(model),
                            std::fabs(candidateObjective)));
      if (candidateObjective <= model + modelTolerance) {
        accepted = true;
        break;
      }
      lipschitz *= 2.0;
      if (!std::isfinite(lipschitz))
        return false;
    }
    if (!accepted)
      return false;

    if (candidateObjective >
        currentObjective +
            1.0e-12 * std::max(1.0, std::fabs(currentObjective))) {
      // Monotone restart.  Re-run the same backtracking majorization from the
      // accepted iterate instead of allowing extrapolation to raise energy.
      extrapolated = state;
      if (!evaluate(extrapolated, gradient, extrapolatedObjective))
        return false;
      accepted = false;
      for (physx::PxU32 backtrack = 0;
           backtrack < kMaximumBacktracks; ++backtrack) {
        for (physx::PxU32 row = 0; row < rowCount; ++row)
          candidate[row] =
              extrapolated[row] - gradient[row] / lipschitz;
        if (!project(candidate) ||
            !evaluate(candidate, candidateGradient, candidateObjective))
          return false;
        double model = extrapolatedObjective;
        double deltaMagnitudeSq = 0.0;
        for (physx::PxU32 row = 0; row < rowCount; ++row) {
          const double delta = candidate[row] - extrapolated[row];
          model += gradient[row] * delta;
          deltaMagnitudeSq += delta * delta;
        }
        model += 0.5 * lipschitz * deltaMagnitudeSq;
        const double modelTolerance =
            1.0e-12 * std::max(
                1.0, std::max(std::fabs(model),
                              std::fabs(candidateObjective)));
        if (candidateObjective <= model + modelTolerance) {
          accepted = true;
          break;
        }
        lipschitz *= 2.0;
        if (!std::isfinite(lipschitz))
          return false;
      }
      if (!accepted)
        return false;
      acceleration = 1.0;
    } else {
      acceleration = nextAcceleration;
    }

    previous = state;
    state = candidate;
    currentObjective = candidateObjective;

    if ((iteration & 7u) == 7u || iteration + 1u == maximumIterations) {
      if (!evaluate(state, gradient, currentObjective))
        return false;
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        candidate[row] = state[row] - gradient[row] / lipschitz;
      if (!project(candidate))
        return false;
      double residual = 0.0;
      double velocityScale = 1.0;
      for (physx::PxU32 row = 0; row < rowCount; ++row) {
        residual = std::max(
            residual, scale[row] * lipschitz *
                          std::fabs(state[row] - candidate[row]));
        velocityScale = std::max(
            velocityScale,
            std::max(std::fabs(q[row]),
                     std::fabs(physicalResponse[row] + q[row])));
      }
      if (!std::isfinite(residual) || !std::isfinite(velocityScale))
        return false;
      if (residual <= relativeTolerance * velocityScale) {
        for (physx::PxU32 row = 0; row < rowCount; ++row)
          impulses[row] = state[row] * inverseScale[row];
        return true;
      }
    }
  }
  return false;
}

static bool solveAvbdRestitutionFreeSystem(
    const double response[8][8], const double q[8],
    const physx::PxU32 freeRows[8], physx::PxU32 freeCount,
    double solution[8]) {
  double augmented[8][9] = {};
  double matrixScale = 1.0;
  for (physx::PxU32 row = 0; row < freeCount; ++row) {
    const physx::PxU32 sourceRow = freeRows[row];
    for (physx::PxU32 column = 0; column < freeCount; ++column) {
      augmented[row][column] =
          response[sourceRow][freeRows[column]];
      matrixScale = std::max(
          matrixScale, std::fabs(augmented[row][column]));
    }
    augmented[row][freeCount] = -q[sourceRow];
  }

  const double pivotTolerance = 1.0e-11 * matrixScale;
  for (physx::PxU32 column = 0; column < freeCount; ++column) {
    physx::PxU32 pivot = column;
    double pivotMagnitude = std::fabs(augmented[column][column]);
    for (physx::PxU32 row = column + 1u; row < freeCount; ++row) {
      const double magnitude = std::fabs(augmented[row][column]);
      if (magnitude > pivotMagnitude) {
        pivot = row;
        pivotMagnitude = magnitude;
      }
    }
    if (!std::isfinite(pivotMagnitude) ||
        pivotMagnitude <= pivotTolerance)
      return false;
    if (pivot != column) {
      for (physx::PxU32 entry = column; entry <= freeCount; ++entry)
        std::swap(augmented[column][entry], augmented[pivot][entry]);
    }
    for (physx::PxU32 row = column + 1u; row < freeCount; ++row) {
      const double factor =
          augmented[row][column] / augmented[column][column];
      augmented[row][column] = 0.0;
      for (physx::PxU32 entry = column + 1u; entry <= freeCount; ++entry)
        augmented[row][entry] -= factor * augmented[column][entry];
    }
  }

  for (physx::PxU32 reverse = freeCount; reverse > 0; --reverse) {
    const physx::PxU32 row = reverse - 1u;
    double value = augmented[row][freeCount];
    for (physx::PxU32 column = row + 1u; column < freeCount; ++column)
      value -= augmented[row][column] * solution[column];
    value /= augmented[row][row];
    if (!std::isfinite(value))
      return false;
    solution[row] = value;
  }
  return true;
}

/**
 * Solve the small positive-semidefinite restitution complementarity block.
 * Eight points give at most 256 active sets, so deterministic enumeration is
 * cheaper and more reproducible than a point-order Gauss--Seidel replay.
 */
static bool solveAvbdRestitutionBlock(
    const double response[8][8], const double q[8],
    physx::PxU32 rowCount, double impulses[8]) {
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    impulses[row] = 0.0;

  double velocityScale = 1.0;
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    velocityScale = std::max(velocityScale, std::fabs(q[row]));
  const double kktTolerance = 2.0e-6 * velocityScale;
  const double impulseTolerance = 1.0e-10;
  bool found = true;
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    found = found && q[row] >= -kktTolerance;
  double bestObjective = found ? 0.0 : std::numeric_limits<double>::infinity();
  physx::PxU32 bestMask = 0;
  double best[8] = {};

  const physx::PxU32 maskCount = 1u << rowCount;
  for (physx::PxU32 mask = 1u; mask < maskCount; ++mask) {
    physx::PxU32 freeRows[8];
    physx::PxU32 freeCount = 0;
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      if ((mask & (1u << row)) != 0)
        freeRows[freeCount++] = row;
    }
    double freeSolution[8] = {};
    if (!solveAvbdRestitutionFreeSystem(
            response, q, freeRows, freeCount, freeSolution))
      continue;

    double candidate[8] = {};
    bool valid = true;
    for (physx::PxU32 freeIndex = 0; freeIndex < freeCount; ++freeIndex) {
      const double value = freeSolution[freeIndex];
      if (value <= impulseTolerance || !std::isfinite(value)) {
        valid = false;
        break;
      }
      candidate[freeRows[freeIndex]] = value;
    }
    if (!valid)
      continue;

    double objective = 0.0;
    for (physx::PxU32 row = 0; row < rowCount && valid; ++row) {
      double residual = q[row];
      for (physx::PxU32 column = 0; column < rowCount; ++column)
        residual += response[row][column] * candidate[column];
      if ((mask & (1u << row)) != 0) {
        if (std::fabs(residual) > 4.0 * kktTolerance)
          valid = false;
      } else if (residual < -kktTolerance) {
        valid = false;
      }
      objective += candidate[row] *
                   (q[row] + 0.5 * (residual - q[row]));
    }
    if (!valid || !std::isfinite(objective))
      continue;
    const double objectiveTolerance =
        1.0e-12 * std::max(1.0, std::fabs(bestObjective));
    if (!found || objective < bestObjective - objectiveTolerance ||
        (std::fabs(objective - bestObjective) <= objectiveTolerance &&
         mask < bestMask)) {
      found = true;
      bestObjective = objective;
      bestMask = mask;
      for (physx::PxU32 row = 0; row < rowCount; ++row)
        best[row] = candidate[row];
    }
  }
  if (!found)
    return false;
  for (physx::PxU32 row = 0; row < rowCount; ++row)
    impulses[row] = best[row];
  return true;
}

static bool isAvbdManifoldRestitutionResidual(
    const AvbdContactConstraint &contact) {
  const AvbdCompiledVelocityObjective *objective =
      findAvbdContactSourceObjective(
          contact.objectiveProgram,
          eCONTACT_SOURCE_MATERIAL_NORMAL);
  return objective &&
         objective->owner == AvbdVelocityObjectiveOwner::ManifoldFinalize &&
         objective->kind == AvbdVelocityObjectiveKind::MaterialNormal &&
         objective->span ==
             AvbdVelocityObjectiveSpan::NormalAndTangentCone &&
         objective->reconstruction ==
             AvbdVelocityObjectiveReconstruction::PoseResidual;
}

struct AvbdDynamicContactConePoint {
  physx::PxU32 contactIndex;
  physx::PxU32 bodyIndex[2];
  physx::PxVec3 axis[3][2];
  physx::PxVec3 angularJacobian[3][2];
  physx::PxVec3 positionAlAngularJacobian[3][2];
  physx::PxReal linearScale[2];
  physx::PxReal angularScale[2];
  double reportImpulse[3];
  double solveStartVelocityMinusTarget[3];
  double poseVelocityMinusTarget[3];
  double friction;
  double approach;
  double normalTarget;
  physx::PxU8 persistentPoint;
};

static const AvbdCompiledVelocityObjective *
findAvbdBodyStaticMaterialNormalManifold(
    const AvbdContactConstraint &contact);

static bool buildAvbdRigidImpactPoint(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 contactIndex,
    const physx::PxArray<physx::PxVec3> &linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> &angularVelAtSolveStart,
    physx::PxReal invDt, physx::PxReal bounceThreshold,
    bool allowRestitution, AvbdDynamicContactConePoint &point) {
  AvbdContactConstraint &contact = contacts[contactIndex];
  point.contactIndex = contactIndex;
  point.bodyIndex[0] = contact.header.bodyIndexA;
  point.bodyIndex[1] = contact.header.bodyIndexB;
  const bool dynamicEndpoint[2] = {
      point.bodyIndex[0] < numBodies,
      point.bodyIndex[1] < numBodies};
  const physx::PxU32 dynamicCount =
      physx::PxU32(dynamicEndpoint[0]) +
      physx::PxU32(dynamicEndpoint[1]);
  if (dynamicCount == 0u ||
      !physx::PxIsFinite(contact.maxImpulse) ||
      contact.maxImpulse < PX_MAX_REAL ||
      !physx::PxIsFinite(contact.restitution) ||
      contact.restitution < 0.0f || contact.restitution > 1.0f ||
      !contact.targetVelocity.isFinite() ||
      physx::PxAbs(contact.targetVelocity.dot(contact.contactNormal)) >
          1.0e-6f)
    return false;

  const physx::PxReal contactLinearScale[2] = {
      contact.invMassScaleA, contact.invMassScaleB};
  const physx::PxReal contactAngularScale[2] = {
      contact.invInertiaScaleA, contact.invInertiaScaleB};
  for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
    point.linearScale[endpoint] = 0.0f;
    point.angularScale[endpoint] = 0.0f;
    if (dynamicEndpoint[endpoint]) {
      if (bodies[point.bodyIndex[endpoint]].invMass <= 0.0f ||
          !physx::PxIsFinite(contactLinearScale[endpoint]) ||
          !physx::PxIsFinite(contactAngularScale[endpoint]) ||
          contactLinearScale[endpoint] < 0.0f ||
          contactAngularScale[endpoint] < 0.0f)
        return false;
      point.linearScale[endpoint] = contactLinearScale[endpoint];
      point.angularScale[endpoint] = contactAngularScale[endpoint];
    }
  }

  const physx::PxVec3 basis[3] = {
      contact.contactNormal, contact.tangent0, contact.tangent1};
  for (physx::PxU32 component = 0; component < 3u; ++component) {
    if (!basis[component].isFinite() ||
        basis[component].magnitudeSquared() <= 1.0e-12f)
      return false;
    point.axis[component][0] = basis[component];
    point.axis[component][1] = -basis[component];
  }

  const AvbdMaterialContactGeometry geometry =
      buildAvbdMaterialContactGeometry(contact, bodies, numBodies, invDt);
  const physx::PxVec3 solveStartArm[2] = {
      geometry.solveStartMaterialArmA,
      geometry.solveStartMaterialArmB};
  const physx::PxVec3 materialArm[2] = {
      geometry.materialArmA, geometry.materialArmB};
  const physx::PxVec3 positionAlArm[2] = {
      geometry.positionAlArmA, geometry.positionAlArmB};
  physx::PxReal solveStartRelativeVelocity[3] = {};
  physx::PxReal poseRelativeVelocity[3] = {};
  for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
    for (physx::PxU32 component = 0; component < 3u; ++component)
      point.angularJacobian[component][endpoint] = physx::PxVec3(0.0f);
    for (physx::PxU32 component = 0; component < 3u; ++component)
      point.positionAlAngularJacobian[component][endpoint] =
          physx::PxVec3(0.0f);
    if (!dynamicEndpoint[endpoint])
      continue;
    const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
    const AvbdSolverBody &body = bodies[bodyIndex];
    for (physx::PxU32 component = 0; component < 3u; ++component) {
      const physx::PxVec3 solveStartAngularJacobian =
          solveStartArm[endpoint].cross(point.axis[component][endpoint]);
      point.angularJacobian[component][endpoint] =
          materialArm[endpoint].cross(point.axis[component][endpoint]);
      point.positionAlAngularJacobian[component][endpoint] =
          positionAlArm[endpoint].cross(
              point.axis[component][endpoint]);
      solveStartRelativeVelocity[component] +=
          linearVelAtSolveStart[bodyIndex].dot(
              point.axis[component][endpoint]) +
          angularVelAtSolveStart[bodyIndex].dot(
              solveStartAngularJacobian);
      poseRelativeVelocity[component] +=
          body.linearVelocity.dot(point.axis[component][endpoint]) +
          body.angularVelocity.dot(
              point.angularJacobian[component][endpoint]);
    }
  }
  if (dynamicCount == 1u) {
    const physx::PxU32 dynamicIndex = dynamicEndpoint[0] ? 0u : 1u;
    for (physx::PxU32 component = 0; component < 3u; ++component) {
      const physx::PxReal staticMaterialVelocity =
          geometry.staticVelocity.dot(point.axis[component][dynamicIndex]);
      solveStartRelativeVelocity[component] -= staticMaterialVelocity;
      poseRelativeVelocity[component] -= staticMaterialVelocity;
    }
  }

  const physx::PxReal approach = -solveStartRelativeVelocity[0];
  const physx::PxReal normalTargetVelocity =
      allowRestitution && approach > bounceThreshold &&
              approach > contact.detectionSeparation * invDt
          ? contact.restitution * approach
          : 0.0f;
  if (dynamicCount == 2u) {
    point.reportImpulse[0] =
        double(physx::PxMax(0.0f, -contact.header.lambda) / invDt);
    point.reportImpulse[1] = double(-contact.tangentLambda0 / invDt);
    point.reportImpulse[2] = double(-contact.tangentLambda1 / invDt);
    point.friction = double(contactCoulombMu(contact));
  } else {
    // For an authoritative frozen-epoch solve, body-static rows use the AL
    // force integral only as a warm iterate.  They do not consume the result
    // of an earlier velocity owner; the final natural-map certificate decides
    // whether this iterate may remain dormant or must be re-solved.
    point.reportImpulse[0] =
        double(physx::PxMax(0.0f, -contact.header.lambda) / invDt);
    point.reportImpulse[1] = double(-contact.tangentLambda0 / invDt);
    point.reportImpulse[2] = double(-contact.tangentLambda1 / invDt);
    // Static endpoints are leaves in the material graph, not a separate
    // material model.  Once an impact component replaces the pose-derived
    // velocity it must also reconstruct the static patch's Coulomb response;
    // otherwise the replacement silently discards ground friction.
    point.friction = double(contactCoulombMu(contact));
  }
  point.approach = double(approach);
  point.normalTarget = double(normalTargetVelocity);
  point.persistentPoint = contact.persistentPointMatched;
  if (!std::isfinite(point.reportImpulse[0]) ||
      !std::isfinite(point.reportImpulse[1]) ||
      !std::isfinite(point.reportImpulse[2]) ||
      !std::isfinite(point.friction) || point.friction < 0.0)
    return false;
  point.poseVelocityMinusTarget[0] =
      double(poseRelativeVelocity[0] - normalTargetVelocity);
  point.poseVelocityMinusTarget[1] =
      double(poseRelativeVelocity[1] -
             contact.targetVelocity.dot(contact.tangent0));
  point.poseVelocityMinusTarget[2] =
      double(poseRelativeVelocity[2] -
             contact.targetVelocity.dot(contact.tangent1));
  point.solveStartVelocityMinusTarget[0] =
      double(solveStartRelativeVelocity[0] - normalTargetVelocity);
  point.solveStartVelocityMinusTarget[1] =
      double(solveStartRelativeVelocity[1] -
             contact.targetVelocity.dot(contact.tangent0));
  point.solveStartVelocityMinusTarget[2] =
      double(solveStartRelativeVelocity[2] -
             contact.targetVelocity.dot(contact.tangent1));
  return std::isfinite(point.solveStartVelocityMinusTarget[0]) &&
         std::isfinite(point.solveStartVelocityMinusTarget[1]) &&
         std::isfinite(point.solveStartVelocityMinusTarget[2]) &&
         std::isfinite(point.poseVelocityMinusTarget[0]) &&
         std::isfinite(point.poseVelocityMinusTarget[1]) &&
         std::isfinite(point.poseVelocityMinusTarget[2]);
}

/**
 * Close the impact-connected rigid contact frontier from one immutable
 * post-body-static pose-velocity baseline.
 *
 * Restitution-eligible dynamic manifolds seed the active set. After each
 * simultaneous component solve, every dormant manifold incident through the
 * body-contact CSR is evaluated at the candidate velocity. A non-zero natural
 * Coulomb-map residual promotes the complete manifold, components are merged
 * through dynamic endpoints, and the enlarged problem is rebuilt from the
 * same baseline. Consequently propagation stops on the NCP cut, not at an
 * arbitrary graph distance. Static endpoints are leaves and never connect
 * otherwise independent components.
 *
 * All identity and traversal state is dense: contact ranges, body CSR,
 * generation stamps, union/find roots and counting offsets. There is no hash
 * lookup in classification, frontier discovery, assembly or commit.
 */
static void applyRigidImpactActiveFrontiers(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold) {
  if (!bodies || !contacts || numBodies == 0u || numContacts == 0u ||
      !contactMap || contactMap->numBodies != numBodies ||
      !contactMap->constraintOffsets || !contactMap->constraintCounts ||
      (contactMap->totalConstraintRefs > 0u &&
       !contactMap->constraintIndices) ||
      !linearVelAtSolveStart || !angularVelAtSolveStart || dt <= 0.0f ||
      !linearPoseVelocityGain || !angularPoseVelocityGain ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies ||
      linearPoseVelocityGain->size() != numBodies ||
      angularPoseVelocityGain->size() != numBodies ||
      numContacts > PX_MAX_U32 / 3u)
    return;

  const physx::PxReal invDt = 1.0f / dt;
  const auto hasFrontierMaterialOwner =
      [&](const AvbdContactConstraint &contact,
          bool dynamicDynamic, bool rigidStatic) -> bool {
        if (dynamicDynamic)
          return isAvbdManifoldRestitutionResidual(contact) ||
                 hasVelocityPassiveFrictionComponentOwner(contact);
        if (!rigidStatic)
          return false;
        if (findAvbdBodyStaticMaterialNormalManifold(contact))
          return true;
        const AvbdCompiledVelocityObjective *objective =
            findAvbdCompleteManifoldObjective(contact.objectiveProgram);
        return objective &&
               objective->reconstruction ==
                   AvbdVelocityObjectiveReconstruction::PoseResidual &&
               (objective->kind ==
                    AvbdVelocityObjectiveKind::PassiveFriction ||
                objective->kind ==
                    AvbdVelocityObjectiveKind::MaterialNormal);
      };
  physx::PxArray<AvbdDynamicContactConePoint> pointCache(numContacts);
  physx::PxArray<physx::PxU32> rangeBegin(numContacts);
  physx::PxArray<physx::PxU32> rangeEnd(numContacts);
  physx::PxArray<physx::PxU8> rangeStart(numContacts);
  physx::PxArray<physx::PxU8> supportedRange(numContacts);
  physx::PxArray<physx::PxU8> activeRange(numContacts);
  physx::PxArray<physx::PxU8> promoteRange(numContacts);
  for (physx::PxU32 contact = 0; contact < numContacts; ++contact) {
    rangeBegin[contact] = contact;
    rangeEnd[contact] = contact + 1u;
    rangeStart[contact] = 0;
    supportedRange[contact] = 0;
    activeRange[contact] = 0;
    promoteRange[contact] = 0;
  }

  physx::PxArray<physx::PxU32> parent(numBodies);
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    parent[body] = body;
  const auto findRoot = [&](physx::PxU32 body) -> physx::PxU32 {
    physx::PxU32 root = body;
    while (parent[root] != root)
      root = parent[root];
    while (parent[body] != body) {
      const physx::PxU32 next = parent[body];
      parent[body] = root;
      body = next;
    }
    return root;
  };
  const auto uniteBodies = [&](physx::PxU32 bodyA,
                               physx::PxU32 bodyB) {
    physx::PxU32 rootA = findRoot(bodyA);
    physx::PxU32 rootB = findRoot(bodyB);
    if (rootA == rootB)
      return;
    if (rootB < rootA)
      std::swap(rootA, rootB);
    parent[rootB] = rootA;
  };

  physx::PxU32 supportedRangeCount = 0u;
  physx::PxU32 activeRangeCount = 0u;
  for (physx::PxU32 begin = 0; begin < numContacts;) {
    const physx::PxU32 managerIndex = contacts[begin].contactManagerIndex;
    physx::PxU32 end = begin + 1u;
    while (end < numContacts &&
           contacts[end].contactManagerIndex == managerIndex)
      ++end;
    rangeStart[begin] = 1;
    rangeEnd[begin] = end;
    for (physx::PxU32 contact = begin; contact < end; ++contact)
      rangeBegin[contact] = begin;

    const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
    const bool dynamicA = bodyA < numBodies;
    const bool dynamicB = bodyB < numBodies;
    const physx::PxU32 dynamicCount =
        physx::PxU32(dynamicA) + physx::PxU32(dynamicB);
    const bool dynamicDynamic = dynamicCount == 2u;
    const bool rigidStatic = dynamicCount == 1u;
    bool supported =
        managerIndex != PX_MAX_U32 &&
        (dynamicDynamic || rigidStatic) &&
        hasFrontierMaterialOwner(
            contacts[begin], dynamicDynamic, rigidStatic);
    bool restitutionSeed = false;
    for (physx::PxU32 contactIndex = begin;
         contactIndex < end && supported; ++contactIndex) {
      AvbdContactConstraint &contact = contacts[contactIndex];
      if (contact.header.bodyIndexA != bodyA ||
          contact.header.bodyIndexB != bodyB ||
          hasDeformableStaticAnchor(contact) ||
          hasKinematicShellAnchor(contact) ||
          !hasFrontierMaterialOwner(
              contact, dynamicDynamic, rigidStatic) ||
          (rigidStatic &&
           contact.targetVelocity.magnitudeSquared() > 1.0e-12f) ||
          !buildAvbdRigidImpactPoint(
              bodies, numBodies, contacts, contactIndex,
              *linearVelAtSolveStart, *angularVelAtSolveStart, invDt,
              bounceThreshold, true, pointCache[contactIndex])) {
        supported = false;
        break;
      }
      restitutionSeed =
          restitutionSeed ||
          pointCache[contactIndex].normalTarget > 0.0;
    }
    if (supported) {
      supportedRange[begin] = 1;
      ++supportedRangeCount;
      if (dynamicDynamic) {
        // Publishing the PositionAL force integral is report-only. It is not
        // replayed into velocity or subtracted through a different Jacobian.
        for (physx::PxU32 contactIndex = begin; contactIndex < end;
             ++contactIndex) {
          AvbdContactConstraint &contact = contacts[contactIndex];
          const AvbdDynamicContactConePoint &point =
              pointCache[contactIndex];
          contact.velocityNormalImpulse =
              physx::PxReal(point.reportImpulse[0]);
          contact.frictionSweepImpulse =
              contact.tangent0 * physx::PxReal(point.reportImpulse[1]) +
              contact.tangent1 * physx::PxReal(point.reportImpulse[2]);
        }
      }
      // A stationary/static endpoint is a leaf of the same material graph,
      // not a reason to fall back to a second velocity owner.  Seed its
      // dynamic body's root directly; only dynamic-dynamic ranges join roots.
      if (restitutionSeed) {
        activeRange[begin] = 1;
        ++activeRangeCount;
      }
      // A warm support is not an impact edge.  Keep its complete PositionAL
      // response dormant until the candidate impact actually violates that
      // manager's natural map; only a real restitution seed joins roots here.
      if (dynamicDynamic && restitutionSeed)
        uniteBodies(bodyA, bodyB);
    }
    begin = end;
  }
  if (activeRangeCount == 0u)
    return;

  physx::PxArray<physx::PxVec3> candidateLinearDelta(numBodies);
  physx::PxArray<physx::PxVec3> candidateAngularDelta(numBodies);
  physx::PxArray<double> candidateImpulse(numContacts * 3u);
  physx::PxArray<physx::PxU8> activeBody(numBodies);
  physx::PxArray<physx::PxU8> rootActive(numBodies);
  physx::PxArray<physx::PxU8> rootSolved(numBodies);
  physx::PxArray<physx::PxU32> bodySlot(numBodies);
  physx::PxArray<physx::PxU32> managerVisitStamp(numContacts);
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    bodySlot[body] = PX_MAX_U32;
  for (physx::PxU32 contact = 0; contact < numContacts; ++contact)
    managerVisitStamp[contact] = 0u;

  physx::PxU32 visitEpoch = 0u;
  physx::PxU32 closureRound = 0u;
  bool closureConverged = false;
  for (; closureRound <= supportedRangeCount; ++closureRound) {
    for (physx::PxU32 body = 0; body < numBodies; ++body)
      parent[body] = findRoot(body);
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      candidateLinearDelta[body] = physx::PxVec3(0.0f);
      candidateAngularDelta[body] = physx::PxVec3(0.0f);
      activeBody[body] = 0;
      rootActive[body] = 0;
      rootSolved[body] = 0;
    }
    for (physx::PxU32 row = 0; row < numContacts * 3u; ++row)
      candidateImpulse[row] = 0.0;
    for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
         ++contactIndex) {
      if (supportedRange[rangeBegin[contactIndex]]) {
        candidateImpulse[contactIndex * 3u] =
            std::max(0.0, pointCache[contactIndex].reportImpulse[0]);
        candidateImpulse[contactIndex * 3u + 1u] =
            pointCache[contactIndex].reportImpulse[1];
        candidateImpulse[contactIndex * 3u + 2u] =
            pointCache[contactIndex].reportImpulse[2];
      }
    }

    physx::PxArray<physx::PxU32> rootOffsets(numBodies + 1u);
    for (physx::PxU32 root = 0; root <= numBodies; ++root)
      rootOffsets[root] = 0u;
    physx::PxU32 activeContactCount = 0u;
    // First identify roots seeded by a closing impact. Dormant support rows
    // retain their complete PositionAL response and enter only through the
    // natural-map cut below; topology alone cannot flood the frontier.
    for (physx::PxU32 begin = 0; begin < numContacts;) {
      const physx::PxU32 end = rangeEnd[begin];
      if (rangeStart[begin] && activeRange[begin]) {
        const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
        const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
        const physx::PxU32 dynamicBody =
            bodyA < numBodies ? bodyA : bodyB;
        const physx::PxU32 root = parent[dynamicBody];
        rootActive[root] = 1;
      }
      begin = end;
    }
    for (physx::PxU32 begin = 0; begin < numContacts;) {
      const physx::PxU32 end = rangeEnd[begin];
      if (rangeStart[begin] && activeRange[begin]) {
        const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
        const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
        const physx::PxU32 dynamicBody =
            bodyA < numBodies ? bodyA : bodyB;
        const physx::PxU32 root = parent[dynamicBody];
        rootOffsets[root + 1u] += end - begin;
        activeContactCount += end - begin;
      }
      begin = end;
    }
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      const physx::PxU32 root = parent[body];
      if (!rootActive[root])
        continue;
      activeBody[body] = 1;
    }
    // A later writer on an unmodelled incident row would immediately
    // invalidate this root's certificate. Capability is therefore
    // all-or-nothing per active body: finite/custom/deformable/kinematic rows
    // remain on the established pipeline, and this frontier publishes
    // nothing for that solve.
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      if (!activeBody[body])
        continue;
      const physx::PxU32 *incidentContacts = nullptr;
      physx::PxU32 incidentCount = 0u;
      if (!getAvbdBodyContactRange(
              contactMap, body, incidentContacts, incidentCount))
        return;
      for (physx::PxU32 incident = 0; incident < incidentCount; ++incident) {
        const physx::PxU32 contactIndex = incidentContacts[incident];
        if (contactIndex >= numContacts ||
            !supportedRange[rangeBegin[contactIndex]])
          return;
      }
    }
    for (physx::PxU32 root = 1; root <= numBodies; ++root)
      rootOffsets[root] += rootOffsets[root - 1u];
    physx::PxArray<physx::PxU32> rootWriteOffsets(numBodies);
    for (physx::PxU32 root = 0; root < numBodies; ++root)
      rootWriteOffsets[root] = rootOffsets[root];
    physx::PxArray<physx::PxU32> packedContacts(activeContactCount);
    // Canonical contact-row order is independent of the order in which the
    // frontier discovered or merged a manager.
    for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
         ++contactIndex) {
      const physx::PxU32 begin = rangeBegin[contactIndex];
      if (!activeRange[begin])
        continue;
      const physx::PxU32 bodyA = contacts[contactIndex].header.bodyIndexA;
      const physx::PxU32 bodyB = contacts[contactIndex].header.bodyIndexB;
      const physx::PxU32 dynamicBody =
          bodyA < numBodies ? bodyA : bodyB;
      const physx::PxU32 root = parent[dynamicBody];
      packedContacts[rootWriteOffsets[root]++] = contactIndex;
    }

    for (physx::PxU32 root = 0; root < numBodies; ++root) {
      const physx::PxU32 componentBegin = rootOffsets[root];
      const physx::PxU32 componentEnd = rootOffsets[root + 1u];
      const physx::PxU32 pointCount = componentEnd - componentBegin;
      if (pointCount == 0u)
        continue;
      if (pointCount > PX_MAX_U32 / 2u)
        continue;

#if 0
      physx::PxArray<physx::PxVec3> linearResponse(pointCount * 2u);
      physx::PxArray<physx::PxVec3> angularResponse(pointCount * 2u);
      for (physx::PxU32 column = 0; column < pointCount; ++column) {
        const physx::PxU32 pointIndex = column;
        const AvbdDynamicContactConePoint &point = pointCache[
            packedContacts[componentBegin + pointIndex]];
        for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
          const physx::PxU32 responseIndex = column * 2u + endpoint;
          linearResponse[responseIndex] = physx::PxVec3(0.0f);
          angularResponse[responseIndex] = physx::PxVec3(0.0f);
          const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
          if (bodyIndex >= numBodies)
            continue;
          const AvbdSolverBody &body = bodies[bodyIndex];
          physx::PxVec3 linear =
              point.axis[0][endpoint] *
              (body.invMass * point.linearScale[endpoint]);
          physx::PxVec3 angularImpulse =
              point.angularJacobian[0][endpoint];
          body.projectLockedAngularVector(angularImpulse);
          physx::PxVec3 angular =
              body.invInertiaWorld.transform(angularImpulse) *
              point.angularScale[endpoint];
          body.projectLockedLinearVector(linear);
          body.projectLockedAngularVector(angular);
          linearResponse[responseIndex] = linear;
          angularResponse[responseIndex] = angular;
        }
      }

      physx::PxArray<double> response(pointCount * pointCount);
      for (physx::PxU32 row = 0; row < pointCount; ++row) {
        const physx::PxU32 rowPointIndex = row;
        const AvbdDynamicContactConePoint &rowPoint = pointCache[
            packedContacts[componentBegin + rowPointIndex]];
        for (physx::PxU32 column = 0; column < pointCount; ++column) {
          const physx::PxU32 columnPointIndex = column;
          const AvbdDynamicContactConePoint &columnPoint = pointCache[
              packedContacts[componentBegin + columnPointIndex]];
          double value = 0.0;
          for (physx::PxU32 rowEndpoint = 0; rowEndpoint < 2u;
               ++rowEndpoint) {
            const physx::PxU32 rowBody =
                rowPoint.bodyIndex[rowEndpoint];
            if (rowBody >= numBodies)
              continue;
            for (physx::PxU32 columnEndpoint = 0; columnEndpoint < 2u;
                 ++columnEndpoint) {
              if (columnPoint.bodyIndex[columnEndpoint] != rowBody)
                continue;
              value += double(
                  rowPoint.axis[0][rowEndpoint].dot(
                      linearResponse[column * 2u + columnEndpoint]) +
                  rowPoint.angularJacobian[0][rowEndpoint].dot(
                      angularResponse[column * 2u + columnEndpoint]));
            }
          }
          response[row * pointCount + column] = value;
        }
      }

      physx::PxArray<double> q(pointCount);
      physx::PxArray<double> impulses(pointCount);
      for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
           ++pointIndex) {
        const AvbdDynamicContactConePoint &point = pointCache[
            packedContacts[componentBegin + pointIndex]];
        q[pointIndex] = point.solveStartVelocityMinusTarget[0];
        for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
          const physx::PxU32 body = point.bodyIndex[endpoint];
          if (body >= numBodies)
            continue;
          q[pointIndex] += double(
              warmLinearDelta[body].dot(point.axis[0][endpoint]) +
              warmAngularDelta[body].dot(
                  point.angularJacobian[0][endpoint]));
        }
        impulses[pointIndex] =
            std::max(0.0, point.reportImpulse[0]);
      }
      // q currently includes the complete warm response. Active variables
      // replace, rather than add to, their warm normal impulses.
      for (physx::PxU32 rowPoint = 0; rowPoint < pointCount; ++rowPoint) {
        for (physx::PxU32 columnPoint = 0;
             columnPoint < pointCount; ++columnPoint) {
          q[rowPoint] -=
              response[rowPoint * pointCount + columnPoint] *
              impulses[columnPoint];
        }
      }
      // Geometry is PositionAL-owned, but material velocity is not derived
      // from its pose displacement or dual multiplier.  Solve the complete
      // physical cone from the immutable pre-contact velocity epoch:
      //
      //   0 <= p_n _|_ J(v^- + M^-1 J^T p) - vTarget >= 0.
      //
      // Commit the resulting body velocity, rather than replaying p on top of
      // pose-derived velocity.  This removes artificial split/depenetration
      // speed without ever admitting a tensile material impulse.
      if (!solveAvbdNormalNcpAccelerated(
              response.begin(), q.begin(), pointCount,
              impulses.begin())) {
        if (std::getenv("PHYSX_AVBD_RIGID_IMPACT_TRACE")) {
          double minimumNormalQ = std::numeric_limits<double>::infinity();
          double maximumNormalQ = -std::numeric_limits<double>::infinity();
          for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
               ++pointIndex) {
            minimumNormalQ =
                std::min(minimumNormalQ, q[pointIndex]);
            maximumNormalQ =
                std::max(maximumNormalQ, q[pointIndex]);
          }
          std::printf(
              "[AVBD_RIGID_FRONTIER_SOLVE_FAIL] points=%u rows=%u "
              "normalQ=[%.9g,%.9g]\n",
              pointCount, pointCount, minimumNormalQ, maximumNormalQ);
        }
        continue;
      }

      physx::PxArray<physx::PxU32> componentBodies;
      for (physx::PxU32 body = 0; body < numBodies; ++body) {
        if (activeBody[body] && parent[body] == root) {
          bodySlot[body] = componentBodies.size();
          componentBodies.pushBack(body);
        }
      }
      physx::PxArray<physx::PxVec3> componentLinearDelta(
          componentBodies.size());
      physx::PxArray<physx::PxVec3> componentAngularDelta(
          componentBodies.size());
      for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot) {
        const physx::PxU32 body = componentBodies[slot];
        componentLinearDelta[slot] = candidateLinearDelta[body];
        componentAngularDelta[slot] = candidateAngularDelta[body];
      }
      bool finite = true;
      for (physx::PxU32 row = 0; row < pointCount && finite; ++row) {
        finite = std::isfinite(impulses[row]);
        const physx::PxU32 pointIndex = row;
        const AvbdDynamicContactConePoint &point = pointCache[
            packedContacts[componentBegin + pointIndex]];
        const physx::PxReal impulseCorrection = physx::PxReal(
            impulses[row] -
            std::max(0.0, point.reportImpulse[0]));
        for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
          const physx::PxU32 body = point.bodyIndex[endpoint];
          if (body >= numBodies)
            continue;
          const physx::PxU32 slot = bodySlot[body];
          componentLinearDelta[slot] +=
              linearResponse[row * 2u + endpoint] *
              impulseCorrection;
          componentAngularDelta[slot] +=
              angularResponse[row * 2u + endpoint] *
              impulseCorrection;
        }
      }

      // Publish the solved normal iterate into round-local scratch so patch
      // budgets see final active normals and dormant certified warm normals.
      for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
           ++pointIndex) {
        const physx::PxU32 contactIndex =
            packedContacts[componentBegin + pointIndex];
        candidateImpulse[contactIndex * 3u] = impulses[pointIndex];
      }

      // PhysX friction is patch-owned: normals remain per contact point, but
      // each NP patch exposes at most two deterministic material anchors and
      // both share mu*sum(point normal impulse).  Solving a cone independently
      // at every point would multiply friction and manufacture angular energy.
      physx::PxArray<physx::PxU32> anchorContacts;
      for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
           ++contactIndex) {
        const physx::PxU32 begin = rangeBegin[contactIndex];
        // A dormant zero-normal dynamic edge is deliberately not part of the
        // current structural root.  Its patch therefore has zero friction
        // capacity and, more importantly, its other body has no component
        // slot.  It can enter only after the normal natural-map certificate
        // promotes the complete manager.
        if (!supportedRange[begin] || !activeRange[begin])
          continue;
        const physx::PxU32 bodyA =
            contacts[contactIndex].header.bodyIndexA;
        const physx::PxU32 bodyB =
            contacts[contactIndex].header.bodyIndexB;
        const physx::PxU32 dynamicBody =
            bodyA < numBodies ? bodyA : bodyB;
        if (parent[dynamicBody] != root)
          continue;
        const physx::PxU8 mask =
            contacts[contactIndex].frictionAnchorMask;
        if ((mask & 1u) != 0u) {
          anchorContacts.pushBack(contactIndex);
        }
        if ((mask & 2u) != 0u) {
          anchorContacts.pushBack(contactIndex);
        }
      }
      const physx::PxU32 anchorCount = anchorContacts.size();
      if (finite && anchorCount > 0u &&
          anchorCount <= PX_MAX_U32 / 2u) {
        const physx::PxU32 tangentRowCount = anchorCount * 2u;
        if (tangentRowCount > PX_MAX_U32 / tangentRowCount) {
          finite = false;
        } else {
          physx::PxArray<physx::PxVec3> tangentLinearResponse(
              tangentRowCount * 2u);
          physx::PxArray<physx::PxVec3> tangentAngularResponse(
              tangentRowCount * 2u);
          physx::PxArray<double> tangentResponse(
              tangentRowCount * tangentRowCount);
          physx::PxArray<double> tangentQ(tangentRowCount);
          physx::PxArray<double> tangentImpulse(tangentRowCount);
          physx::PxArray<double> tangentLimit(anchorCount);

          for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
            const physx::PxU32 contactIndex = anchorContacts[anchor];
            const AvbdContactConstraint &contact = contacts[contactIndex];
            const AvbdDynamicContactConePoint &point =
                pointCache[contactIndex];
            double patchNormal = 0.0;
            for (physx::PxU32 patchContact = 0;
                 patchContact < numContacts; ++patchContact) {
              if (!supportedRange[rangeBegin[patchContact]] ||
                  !activeRange[rangeBegin[patchContact]])
                continue;
              const AvbdContactConstraint &other = contacts[patchContact];
              if (other.contactManagerIndex == contact.contactManagerIndex &&
                  other.contactPatchIndex == contact.contactPatchIndex)
                patchNormal +=
                    candidateImpulse[patchContact * 3u];
            }
            const physx::PxU32 patchAnchorCount =
                physx::PxMax(physx::PxU32(1u),
                             physx::PxU32(contact.frictionAnchorCount));
            tangentLimit[anchor] =
                point.friction * std::max(0.0, patchNormal) /
                double(patchAnchorCount);

            for (physx::PxU32 tangent = 0; tangent < 2u; ++tangent) {
              const physx::PxU32 row = anchor * 2u + tangent;
              const physx::PxU32 component = tangent + 1u;
              tangentImpulse[row] = 0.0;
              tangentQ[row] = point.poseVelocityMinusTarget[component];
              for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
                const physx::PxU32 responseIndex = row * 2u + endpoint;
                tangentLinearResponse[responseIndex] =
                    physx::PxVec3(0.0f);
                tangentAngularResponse[responseIndex] =
                    physx::PxVec3(0.0f);
                const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
                if (bodyIndex >= numBodies)
                  continue;
                const physx::PxU32 slot = bodySlot[bodyIndex];
                if (slot >= componentBodies.size()) {
                  finite = false;
                  continue;
                }
                tangentQ[row] += double(
                    componentLinearDelta[slot].dot(
                        point.axis[component][endpoint]) +
                    componentAngularDelta[slot].dot(
                        point.angularJacobian[component][endpoint]));
                const AvbdSolverBody &body = bodies[bodyIndex];
                physx::PxVec3 linear =
                    point.axis[component][endpoint] *
                    (body.invMass * point.linearScale[endpoint]);
                physx::PxVec3 angularImpulse =
                    point.angularJacobian[component][endpoint];
                body.projectLockedAngularVector(angularImpulse);
                physx::PxVec3 angular =
                    body.invInertiaWorld.transform(angularImpulse) *
                    point.angularScale[endpoint];
                body.projectLockedLinearVector(linear);
                body.projectLockedAngularVector(angular);
                tangentLinearResponse[responseIndex] = linear;
                tangentAngularResponse[responseIndex] = angular;
              }
            }
          }

          for (physx::PxU32 row = 0; row < tangentRowCount; ++row) {
            const AvbdDynamicContactConePoint &rowPoint =
                pointCache[anchorContacts[row / 2u]];
            const physx::PxU32 rowComponent = row % 2u + 1u;
            for (physx::PxU32 column = 0;
                 column < tangentRowCount; ++column) {
              const AvbdDynamicContactConePoint &columnPoint =
                  pointCache[anchorContacts[column / 2u]];
              double value = 0.0;
              for (physx::PxU32 rowEndpoint = 0; rowEndpoint < 2u;
                   ++rowEndpoint) {
                const physx::PxU32 rowBody =
                    rowPoint.bodyIndex[rowEndpoint];
                if (rowBody >= numBodies)
                  continue;
                for (physx::PxU32 columnEndpoint = 0;
                     columnEndpoint < 2u; ++columnEndpoint) {
                  if (columnPoint.bodyIndex[columnEndpoint] != rowBody)
                    continue;
                  value += double(
                      rowPoint.axis[rowComponent][rowEndpoint].dot(
                          tangentLinearResponse[
                              column * 2u + columnEndpoint]) +
                      rowPoint.angularJacobian[rowComponent][rowEndpoint].dot(
                          tangentAngularResponse[
                              column * 2u + columnEndpoint]));
                }
              }
              tangentResponse[row * tangentRowCount + column] = value;
            }
          }

          if (!solveAvbdTangentDisksAccelerated(
                  tangentResponse.begin(), tangentQ.begin(),
                  tangentLimit.begin(), anchorCount,
                  tangentImpulse.begin())) {
            finite = false;
          } else {
            for (physx::PxU32 row = 0; row < tangentRowCount; ++row) {
              const physx::PxReal impulse =
                  physx::PxReal(tangentImpulse[row]);
              finite = finite && physx::PxIsFinite(impulse);
              const physx::PxU32 contactIndex =
                  anchorContacts[row / 2u];
              candidateImpulse[
                  contactIndex * 3u + row % 2u + 1u] =
                  tangentImpulse[row];
              const AvbdDynamicContactConePoint &point =
                  pointCache[contactIndex];
              for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
                const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
                if (bodyIndex >= numBodies)
                  continue;
                const physx::PxU32 slot = bodySlot[bodyIndex];
                if (slot >= componentBodies.size()) {
                  finite = false;
                  continue;
                }
                componentLinearDelta[slot] +=
                    tangentLinearResponse[row * 2u + endpoint] * impulse;
                componentAngularDelta[slot] +=
                    tangentAngularResponse[row * 2u + endpoint] * impulse;
              }
            }
          }
        }
      }
      for (physx::PxU32 slot = 0;
           slot < componentBodies.size() && finite; ++slot) {
        const physx::PxU32 body = componentBodies[slot];
        finite = componentLinearDelta[slot].isFinite() &&
                 componentAngularDelta[slot].isFinite() &&
                 (bodies[body].linearVelocity +
                  componentLinearDelta[slot]).isFinite() &&
                 (bodies[body].angularVelocity +
                  componentAngularDelta[slot]).isFinite();
      }
      if (finite) {
        rootSolved[root] = 1;
        for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot) {
          const physx::PxU32 body = componentBodies[slot];
          candidateLinearDelta[body] = componentLinearDelta[slot];
          candidateAngularDelta[body] = componentAngularDelta[slot];
        }
        for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
             ++pointIndex) {
          const physx::PxU32 contactIndex =
              packedContacts[componentBegin + pointIndex];
          candidateImpulse[contactIndex * 3u] = impulses[pointIndex];
        }
      }
      for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot)
        bodySlot[componentBodies[slot]] = PX_MAX_U32;
#endif

      physx::PxArray<physx::PxU32> componentBodies;
      for (physx::PxU32 body = 0; body < numBodies; ++body) {
        if (activeBody[body] && parent[body] == root) {
          bodySlot[body] = componentBodies.size();
          componentBodies.pushBack(body);
        }
      }
      if (componentBodies.size() == 0u)
        continue;

      bool finite = true;
      physx::PxArray<physx::PxReal> componentLinearScale(
          componentBodies.size());
      physx::PxArray<physx::PxReal> componentAngularScale(
          componentBodies.size());
      physx::PxArray<physx::PxU8> componentScaleSet(
          componentBodies.size());
      for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot)
        componentScaleSet[slot] = 0u;
      // The convex matrix-free subsolves require a symmetric mobility. A
      // contact-modification callback may assign different inverse-mass
      // scales to different managers on the same body; those components need
      // the general nonsymmetric natural-map backend and are left untouched
      // here rather than being certified against the wrong operator.
      for (physx::PxU32 row = 0; row < pointCount; ++row) {
        const AvbdDynamicContactConePoint &point = pointCache[
            packedContacts[componentBegin + row]];
        for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
          const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
          if (bodyIndex >= numBodies)
            continue;
          const physx::PxU32 slot = bodySlot[bodyIndex];
          if (slot >= componentBodies.size()) {
            finite = false;
            continue;
          }
          if (!physx::PxIsFinite((*linearPoseVelocityGain)[bodyIndex]) ||
              !physx::PxIsFinite((*angularPoseVelocityGain)[bodyIndex]) ||
              (*linearPoseVelocityGain)[bodyIndex] < 0.0f ||
              (*angularPoseVelocityGain)[bodyIndex] < 0.0f) {
            finite = false;
            continue;
          }
          if (!componentScaleSet[slot]) {
            componentScaleSet[slot] = 1u;
            componentLinearScale[slot] = point.linearScale[endpoint];
            componentAngularScale[slot] = point.angularScale[endpoint];
          } else if (
              physx::PxAbs(componentLinearScale[slot] -
                           point.linearScale[endpoint]) > 1.0e-6f ||
              physx::PxAbs(componentAngularScale[slot] -
                           point.angularScale[endpoint]) > 1.0e-6f) {
            finite = false;
          }
        }
      }
      physx::PxArray<physx::PxVec3> positionAlRemovalLinear(
          componentBodies.size());
      physx::PxArray<physx::PxVec3> positionAlRemovalAngular(
          componentBodies.size());
      for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot) {
        positionAlRemovalLinear[slot] = physx::PxVec3(0.0f);
        positionAlRemovalAngular[slot] = physx::PxVec3(0.0f);
      }
      // vPose already contains the raw PositionAL material response. Remove
      // exactly the rows owned by this frontier through their persistent AL
      // Jacobian. The solved total impulses are added later through the fresh
      // material Jacobian, preserving every unrelated velocity owner.
      for (physx::PxU32 row = 0; row < pointCount; ++row) {
        const physx::PxU32 contactIndex =
            packedContacts[componentBegin + row];
        const AvbdContactConstraint &contact = contacts[contactIndex];
        const AvbdDynamicContactConePoint &point = pointCache[contactIndex];
        for (physx::PxU32 component = 0; component < 3u; ++component) {
          if (component > 0u &&
              !hasPositionTangentParticipation(contact))
            continue;
          const physx::PxReal impulse =
              -physx::PxReal(point.reportImpulse[component]);
          if (!physx::PxIsFinite(impulse)) {
            finite = false;
            continue;
          }
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
            if (bodyIndex >= numBodies)
              continue;
            const physx::PxU32 slot = bodySlot[bodyIndex];
            if (slot >= componentBodies.size()) {
              finite = false;
              continue;
            }
            const AvbdSolverBody &body = bodies[bodyIndex];
            physx::PxVec3 linear =
                point.axis[component][endpoint] *
                (body.invMass * point.linearScale[endpoint] * impulse *
                 (*linearPoseVelocityGain)[bodyIndex]);
            physx::PxVec3 angularImpulse =
                point.positionAlAngularJacobian[component][endpoint] *
                impulse;
            body.projectLockedAngularVector(angularImpulse);
            physx::PxVec3 angular =
                body.invInertiaWorld.transform(angularImpulse) *
                (point.angularScale[endpoint] *
                 (*angularPoseVelocityGain)[bodyIndex]);
            body.projectLockedLinearVector(linear);
            body.projectLockedAngularVector(angular);
            positionAlRemovalLinear[slot] += linear;
            positionAlRemovalAngular[slot] += angular;
          }
        }
      }

      // Every response multiply is a body scatter followed by a row gather.
      // Work is O(rows+bodies) and touches only dense arrays; no N^2 matrix or
      // associative lookup is built.
      physx::PxArray<physx::PxVec3> normalLinearResponse(pointCount * 2u);
      physx::PxArray<physx::PxVec3> normalAngularResponse(pointCount * 2u);
      physx::PxArray<double> normalDiagonal(pointCount);
      physx::PxArray<double> normalBaseQ(pointCount);
      physx::PxArray<double> normalQ(pointCount);
      physx::PxArray<double> normalImpulse(pointCount);
      physx::PxArray<physx::PxU32> normalSlot(numContacts);
      for (physx::PxU32 contact = 0; contact < numContacts; ++contact)
        normalSlot[contact] = PX_MAX_U32;

      for (physx::PxU32 row = 0; row < pointCount; ++row) {
        const physx::PxU32 contactIndex =
            packedContacts[componentBegin + row];
        normalSlot[contactIndex] = row;
        const AvbdDynamicContactConePoint &point = pointCache[contactIndex];
        normalBaseQ[row] = point.poseVelocityMinusTarget[0];
        normalImpulse[row] = std::max(0.0, point.reportImpulse[0]);
        double diagonal = 0.0;
        for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
          const physx::PxU32 responseIndex = row * 2u + endpoint;
          normalLinearResponse[responseIndex] = physx::PxVec3(0.0f);
          normalAngularResponse[responseIndex] = physx::PxVec3(0.0f);
          const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
          if (bodyIndex >= numBodies)
            continue;
          const physx::PxU32 slot = bodySlot[bodyIndex];
          if (slot >= componentBodies.size()) {
            finite = false;
            continue;
          }
          normalBaseQ[row] += double(
              positionAlRemovalLinear[slot].dot(
                  point.axis[0][endpoint]) +
              positionAlRemovalAngular[slot].dot(
                  point.angularJacobian[0][endpoint]));
          const AvbdSolverBody &body = bodies[bodyIndex];
          physx::PxVec3 linear =
              point.axis[0][endpoint] *
              (body.invMass * point.linearScale[endpoint]);
          physx::PxVec3 angularImpulse =
              point.angularJacobian[0][endpoint];
          body.projectLockedAngularVector(angularImpulse);
          physx::PxVec3 angular =
              body.invInertiaWorld.transform(angularImpulse) *
              point.angularScale[endpoint];
          body.projectLockedLinearVector(linear);
          body.projectLockedAngularVector(angular);
          normalLinearResponse[responseIndex] = linear;
          normalAngularResponse[responseIndex] = angular;
          diagonal += double(
              point.axis[0][endpoint].dot(linear) +
              point.angularJacobian[0][endpoint].dot(angular));
        }
        normalDiagonal[row] = diagonal;
        finite = finite && std::isfinite(normalBaseQ[row]) &&
                 std::isfinite(normalImpulse[row]) &&
                 std::isfinite(normalDiagonal[row]) &&
                 normalDiagonal[row] > 1.0e-12;
      }

      // Build an exact stable patch view from immutable row ids. Only rows
      // admitted to this material frontier are replaced; dormant rows retain
      // their complete PositionAL response until the natural-map cut promotes
      // their manager.
      physx::PxArray<physx::PxU32> patchContacts;
      for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
           ++contactIndex) {
        const physx::PxU32 begin = rangeBegin[contactIndex];
        if (!supportedRange[begin] || !activeRange[begin])
          continue;
        const physx::PxU32 bodyA =
            contacts[contactIndex].header.bodyIndexA;
        const physx::PxU32 bodyB =
            contacts[contactIndex].header.bodyIndexB;
        const physx::PxU32 dynamicBody =
            bodyA < numBodies ? bodyA : bodyB;
        if (parent[dynamicBody] != root)
          continue;
        if ((bodyA < numBodies && parent[bodyA] != root) ||
            (bodyB < numBodies && parent[bodyB] != root)) {
          finite = false;
          continue;
        }
        if (pointCache[contactIndex].friction > 0.0 &&
            contacts[contactIndex].frictionAnchorCount > 0u)
          patchContacts.pushBack(contactIndex);
      }
      std::sort(
          patchContacts.begin(), patchContacts.end(),
          [&](physx::PxU32 lhs, physx::PxU32 rhs) {
            const AvbdContactConstraint &a = contacts[lhs];
            const AvbdContactConstraint &b = contacts[rhs];
            if (a.contactManagerIndex != b.contactManagerIndex)
              return a.contactManagerIndex < b.contactManagerIndex;
            if (a.contactPatchIndex != b.contactPatchIndex)
              return a.contactPatchIndex < b.contactPatchIndex;
            return lhs < rhs;
          });

      physx::PxArray<physx::PxU32> patchOffsets;
      physx::PxArray<double> patchFriction;
      physx::PxArray<physx::PxU32> anchorContacts;
      physx::PxArray<physx::PxU32> anchorPatch;
      patchOffsets.pushBack(0u);
      for (physx::PxU32 patchBegin = 0;
           patchBegin < patchContacts.size() && finite;) {
        const physx::PxU32 firstContact = patchContacts[patchBegin];
        const AvbdContactConstraint &first = contacts[firstContact];
        physx::PxU32 patchEnd = patchBegin + 1u;
        while (patchEnd < patchContacts.size()) {
          const AvbdContactConstraint &next =
              contacts[patchContacts[patchEnd]];
          if (next.contactManagerIndex != first.contactManagerIndex ||
              next.contactPatchIndex != first.contactPatchIndex)
            break;
          ++patchEnd;
        }
        const physx::PxU32 patchIndex = patchFriction.size();
        const physx::PxU32 firstAnchor = anchorContacts.size();
        for (physx::PxU32 index = patchBegin; index < patchEnd; ++index) {
          const physx::PxU32 contactIndex = patchContacts[index];
          const physx::PxU8 mask =
              contacts[contactIndex].frictionAnchorMask;
          if ((mask & 1u) != 0u) {
            anchorContacts.pushBack(contactIndex);
            anchorPatch.pushBack(patchIndex);
          }
          if ((mask & 2u) != 0u) {
            anchorContacts.pushBack(contactIndex);
            anchorPatch.pushBack(patchIndex);
          }
        }
        const physx::PxU32 actualAnchorCount =
            anchorContacts.size() - firstAnchor;
        const physx::PxU32 expectedAnchorCount =
            first.frictionAnchorCount;
        if (actualAnchorCount != expectedAnchorCount ||
            actualAnchorCount == 0u || actualAnchorCount > 2u ||
            (actualAnchorCount == 2u &&
             anchorContacts[firstAnchor] ==
                 anchorContacts[firstAnchor + 1u])) {
          finite = false;
          break;
        }
        patchFriction.pushBack(pointCache[firstContact].friction);
        patchOffsets.pushBack(patchEnd);
        patchBegin = patchEnd;
      }

      const physx::PxU32 anchorCount = anchorContacts.size();
      if (anchorCount > PX_MAX_U32 / 4u)
        finite = false;
      const physx::PxU32 tangentRowCount =
          finite ? anchorCount * 2u : 0u;
      physx::PxArray<physx::PxVec3> tangentLinearResponse(
          tangentRowCount * 2u);
      physx::PxArray<physx::PxVec3> tangentAngularResponse(
          tangentRowCount * 2u);
      physx::PxArray<double> tangentDiagonal(tangentRowCount);
      physx::PxArray<double> tangentBaseQ(tangentRowCount);
      physx::PxArray<double> tangentQ(tangentRowCount);
      physx::PxArray<double> tangentImpulse(tangentRowCount);
      physx::PxArray<double> tangentLimit(anchorCount);
      physx::PxArray<double> tangentScaledLimit(anchorCount);
      for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
        const physx::PxU32 contactIndex = anchorContacts[anchor];
        const AvbdDynamicContactConePoint &point = pointCache[contactIndex];
        double pairDiagonal[2] = {0.0, 0.0};
        for (physx::PxU32 tangent = 0; tangent < 2u; ++tangent) {
          const physx::PxU32 row = anchor * 2u + tangent;
          const physx::PxU32 component = tangent + 1u;
          tangentBaseQ[row] =
              point.poseVelocityMinusTarget[component];
          tangentImpulse[row] = point.reportImpulse[component];
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 responseIndex = row * 2u + endpoint;
            tangentLinearResponse[responseIndex] = physx::PxVec3(0.0f);
            tangentAngularResponse[responseIndex] = physx::PxVec3(0.0f);
            const physx::PxU32 bodyIndex = point.bodyIndex[endpoint];
            if (bodyIndex >= numBodies)
              continue;
            const physx::PxU32 slot = bodySlot[bodyIndex];
            if (slot >= componentBodies.size()) {
              finite = false;
              continue;
            }
            tangentBaseQ[row] += double(
                positionAlRemovalLinear[slot].dot(
                    point.axis[component][endpoint]) +
                positionAlRemovalAngular[slot].dot(
                    point.angularJacobian[component][endpoint]));
            const AvbdSolverBody &body = bodies[bodyIndex];
            physx::PxVec3 linear =
                point.axis[component][endpoint] *
                (body.invMass * point.linearScale[endpoint]);
            physx::PxVec3 angularImpulse =
                point.angularJacobian[component][endpoint];
            body.projectLockedAngularVector(angularImpulse);
            physx::PxVec3 angular =
                body.invInertiaWorld.transform(angularImpulse) *
                point.angularScale[endpoint];
            body.projectLockedLinearVector(linear);
            body.projectLockedAngularVector(angular);
            tangentLinearResponse[responseIndex] = linear;
            tangentAngularResponse[responseIndex] = angular;
            pairDiagonal[tangent] += double(
                point.axis[component][endpoint].dot(linear) +
                point.angularJacobian[component][endpoint].dot(angular));
          }
          finite = finite && std::isfinite(tangentBaseQ[row]) &&
                   std::isfinite(pairDiagonal[tangent]);
        }
        const double diskDiagonal =
            0.5 * (pairDiagonal[0] + pairDiagonal[1]);
        tangentDiagonal[anchor * 2u] = diskDiagonal;
        tangentDiagonal[anchor * 2u + 1u] = diskDiagonal;
        finite = finite && std::isfinite(diskDiagonal) &&
                 diskDiagonal > 1.0e-12;
      }

      physx::PxArray<AvbdDoubleVec3> responseLinear(
          componentBodies.size());
      physx::PxArray<AvbdDoubleVec3> responseAngular(
          componentBodies.size());
      const auto clearBodyResponse = [&]() {
        for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot) {
          responseLinear[slot] = AvbdDoubleVec3();
          responseAngular[slot] = AvbdDoubleVec3();
        }
      };
      const auto scatterNormal = [&](const physx::PxArray<double> &input) {
        for (physx::PxU32 row = 0; row < pointCount; ++row) {
          const AvbdDynamicContactConePoint &point = pointCache[
              packedContacts[componentBegin + row]];
          const double impulse = input[row];
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 body = point.bodyIndex[endpoint];
            if (body >= numBodies)
              continue;
            const physx::PxU32 slot = bodySlot[body];
            responseLinear[slot] += AvbdDoubleVec3(
                normalLinearResponse[row * 2u + endpoint]) * impulse;
            responseAngular[slot] += AvbdDoubleVec3(
                normalAngularResponse[row * 2u + endpoint]) * impulse;
          }
        }
      };
      const auto scatterTangent = [&](const physx::PxArray<double> &input) {
        for (physx::PxU32 row = 0; row < tangentRowCount; ++row) {
          const AvbdDynamicContactConePoint &point =
              pointCache[anchorContacts[row / 2u]];
          const double impulse = input[row];
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 body = point.bodyIndex[endpoint];
            if (body >= numBodies)
              continue;
            const physx::PxU32 slot = bodySlot[body];
            responseLinear[slot] += AvbdDoubleVec3(
                tangentLinearResponse[row * 2u + endpoint]) * impulse;
            responseAngular[slot] += AvbdDoubleVec3(
                tangentAngularResponse[row * 2u + endpoint]) * impulse;
          }
        }
      };
      const auto gatherNormal = [&](physx::PxArray<double> &output) {
        for (physx::PxU32 row = 0; row < pointCount; ++row) {
          const AvbdDynamicContactConePoint &point = pointCache[
              packedContacts[componentBegin + row]];
          double value = 0.0;
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 body = point.bodyIndex[endpoint];
            if (body >= numBodies)
              continue;
            const physx::PxU32 slot = bodySlot[body];
            const physx::PxVec3 &linearAxis = point.axis[0][endpoint];
            const physx::PxVec3 &angularAxis =
                point.angularJacobian[0][endpoint];
            value += responseLinear[slot].x * double(linearAxis.x) +
                     responseLinear[slot].y * double(linearAxis.y) +
                     responseLinear[slot].z * double(linearAxis.z) +
                     responseAngular[slot].x * double(angularAxis.x) +
                     responseAngular[slot].y * double(angularAxis.y) +
                     responseAngular[slot].z * double(angularAxis.z);
          }
          output[row] = value;
        }
      };
      const auto gatherTangent = [&](physx::PxArray<double> &output) {
        for (physx::PxU32 row = 0; row < tangentRowCount; ++row) {
          const AvbdDynamicContactConePoint &point =
              pointCache[anchorContacts[row / 2u]];
          const physx::PxU32 component = row % 2u + 1u;
          double value = 0.0;
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 body = point.bodyIndex[endpoint];
            if (body >= numBodies)
              continue;
            const physx::PxU32 slot = bodySlot[body];
            const physx::PxVec3 &linearAxis =
                point.axis[component][endpoint];
            const physx::PxVec3 &angularAxis =
                point.angularJacobian[component][endpoint];
            value += responseLinear[slot].x * double(linearAxis.x) +
                     responseLinear[slot].y * double(linearAxis.y) +
                     responseLinear[slot].z * double(linearAxis.z) +
                     responseAngular[slot].x * double(angularAxis.x) +
                     responseAngular[slot].y * double(angularAxis.y) +
                     responseAngular[slot].z * double(angularAxis.z);
          }
          output[row] = value;
        }
      };
      const auto multiplyNormal =
          [&](const physx::PxArray<double> &input,
              physx::PxArray<double> &output) -> bool {
            clearBodyResponse();
            scatterNormal(input);
            gatherNormal(output);
            for (physx::PxU32 row = 0; row < output.size(); ++row)
              if (!std::isfinite(output[row]))
                return false;
            return true;
          };
      const auto multiplyTangent =
          [&](const physx::PxArray<double> &input,
              physx::PxArray<double> &output) -> bool {
            clearBodyResponse();
            scatterTangent(input);
            gatherTangent(output);
            for (physx::PxU32 row = 0; row < output.size(); ++row)
              if (!std::isfinite(output[row]))
                return false;
            return true;
          };

      const auto projectNormals =
          [](physx::PxArray<double> &state) -> bool {
            for (physx::PxU32 row = 0; row < state.size(); ++row) {
              if (!std::isfinite(state[row]))
                return false;
              state[row] = std::max(0.0, state[row]);
            }
            return true;
          };

      bool converged = false;
      physx::PxArray<double> normalResponse(pointCount);
      physx::PxArray<double> tangentResponse(tangentRowCount);
      physx::PxArray<double> crossNormal(pointCount);
      physx::PxArray<double> crossTangent(tangentRowCount);
      static const physx::PxU32 kMaximumCouplingIterations = 64u;
      for (physx::PxU32 couplingIteration = 0;
           couplingIteration < kMaximumCouplingIterations && finite;
           ++couplingIteration) {
        if (tangentRowCount > 0u) {
          clearBodyResponse();
          scatterTangent(tangentImpulse);
          gatherNormal(crossNormal);
        }
        for (physx::PxU32 row = 0; row < pointCount; ++row)
          normalQ[row] = normalBaseQ[row] +
                         (tangentRowCount > 0u ? crossNormal[row] : 0.0);
        if (!solveAvbdProjectedQuadraticMatrixFree(
                multiplyNormal, projectNormals, normalQ.begin(),
                normalDiagonal.begin(), pointCount, normalImpulse.begin(),
                2048u, 5.0e-5)) {
          finite = false;
          break;
        }

        if (tangentRowCount > 0u) {
          physx::PxArray<double> patchNormal(patchFriction.size());
          for (physx::PxU32 patch = 0; patch < patchFriction.size(); ++patch) {
            double normalSum = 0.0;
            for (physx::PxU32 index = patchOffsets[patch];
                 index < patchOffsets[patch + 1u]; ++index) {
              const physx::PxU32 contactIndex = patchContacts[index];
              const physx::PxU32 slot = normalSlot[contactIndex];
              normalSum +=
                  slot < pointCount
                      ? normalImpulse[slot]
                      : candidateImpulse[contactIndex * 3u];
            }
            patchNormal[patch] = std::max(0.0, normalSum);
          }
          for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
            const physx::PxU32 patch = anchorPatch[anchor];
            const physx::PxU32 patchAnchorCount =
                contacts[anchorContacts[anchor]].frictionAnchorCount;
            tangentLimit[anchor] =
                patchFriction[patch] * patchNormal[patch] /
                double(patchAnchorCount);
            tangentScaledLimit[anchor] =
                tangentLimit[anchor] *
                std::sqrt(tangentDiagonal[anchor * 2u]);
          }
          const auto projectTangents =
              [&](physx::PxArray<double> &state) -> bool {
                for (physx::PxU32 anchor = 0; anchor < anchorCount;
                     ++anchor) {
                  const physx::PxU32 row = anchor * 2u;
                  if (!projectAvbdTangentDisk(
                          tangentScaledLimit[anchor], state[row],
                          state[row + 1u]))
                    return false;
                }
                return true;
              };
          clearBodyResponse();
          scatterNormal(normalImpulse);
          gatherTangent(crossTangent);
          for (physx::PxU32 row = 0; row < tangentRowCount; ++row)
            tangentQ[row] = tangentBaseQ[row] + crossTangent[row];
          if (!solveAvbdProjectedQuadraticMatrixFree(
                  multiplyTangent, projectTangents, tangentQ.begin(),
                  tangentDiagonal.begin(), tangentRowCount,
                  tangentImpulse.begin(), 2048u, 5.0e-5)) {
            finite = false;
            break;
          }
        }

        clearBodyResponse();
        scatterNormal(normalImpulse);
        if (tangentRowCount > 0u)
          scatterTangent(tangentImpulse);
        gatherNormal(normalResponse);
        if (tangentRowCount > 0u)
          gatherTangent(tangentResponse);

        // Physical non-associated Coulomb certificate: Signorini normal map
        // and fixed-radius patch tangent maps are checked independently at
        // the same final body velocity. This avoids associated-cone dilation.
        double residual = 0.0;
        double velocityScale = 1.0;
        for (physx::PxU32 row = 0; row < pointCount; ++row) {
          const double velocity = normalBaseQ[row] + normalResponse[row];
          const double rowResidual =
              normalImpulse[row] > 1.0e-10
                  ? std::fabs(velocity)
                  : std::max(0.0, -velocity);
          residual = std::max(residual, rowResidual);
          velocityScale = std::max(
              velocityScale,
              std::max(std::fabs(normalBaseQ[row]),
                       std::fabs(velocity)));
        }
        for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
          const physx::PxU32 row = anchor * 2u;
          const double velocity0 =
              tangentBaseQ[row] + tangentResponse[row];
          const double velocity1 =
              tangentBaseQ[row + 1u] + tangentResponse[row + 1u];
          const double step = 1.0 / tangentDiagonal[row];
          double projected0 = tangentImpulse[row] - step * velocity0;
          double projected1 = tangentImpulse[row + 1u] - step * velocity1;
          if (!projectAvbdTangentDisk(
                  tangentLimit[anchor], projected0, projected1)) {
            finite = false;
            break;
          }
          const double tangentResidual =
              std::sqrt(
                  (tangentImpulse[row] - projected0) *
                      (tangentImpulse[row] - projected0) +
                  (tangentImpulse[row + 1u] - projected1) *
                      (tangentImpulse[row + 1u] - projected1)) /
              step;
          residual = std::max(residual, tangentResidual);
          velocityScale = std::max(
              velocityScale,
              std::max(std::sqrt(velocity0 * velocity0 +
                                 velocity1 * velocity1),
                       std::max(std::fabs(tangentBaseQ[row]),
                                std::fabs(tangentBaseQ[row + 1u]))));
        }
        if (!finite || !std::isfinite(residual) ||
            !std::isfinite(velocityScale)) {
          finite = false;
          break;
        }
        if (residual <= 2.0e-6 * velocityScale) {
          converged = true;
          break;
        }
      }

      if (finite && converged) {
        clearBodyResponse();
        scatterNormal(normalImpulse);
        if (tangentRowCount > 0u)
          scatterTangent(tangentImpulse);

        physx::PxArray<physx::PxVec3> stagedLinearDelta(
            componentBodies.size());
        physx::PxArray<physx::PxVec3> stagedAngularDelta(
            componentBodies.size());
        for (physx::PxU32 slot = 0;
             slot < componentBodies.size(); ++slot) {
          responseLinear[slot] +=
              AvbdDoubleVec3(positionAlRemovalLinear[slot]);
          responseAngular[slot] +=
              AvbdDoubleVec3(positionAlRemovalAngular[slot]);
          const physx::PxU32 body = componentBodies[slot];
          finite = finite && responseLinear[slot].isFinite() &&
                   responseAngular[slot].isFinite() &&
                   std::fabs(responseLinear[slot].x) <= PX_MAX_REAL &&
                   std::fabs(responseLinear[slot].y) <= PX_MAX_REAL &&
                   std::fabs(responseLinear[slot].z) <= PX_MAX_REAL &&
                   std::fabs(responseAngular[slot].x) <= PX_MAX_REAL &&
                   std::fabs(responseAngular[slot].y) <= PX_MAX_REAL &&
                   std::fabs(responseAngular[slot].z) <= PX_MAX_REAL;
          if (!finite)
            continue;
          stagedLinearDelta[slot] = physx::PxVec3(
              physx::PxReal(responseLinear[slot].x),
              physx::PxReal(responseLinear[slot].y),
              physx::PxReal(responseLinear[slot].z));
          stagedAngularDelta[slot] = physx::PxVec3(
              physx::PxReal(responseAngular[slot].x),
              physx::PxReal(responseAngular[slot].y),
              physx::PxReal(responseAngular[slot].z));
          finite = (bodies[body].linearVelocity +
                    stagedLinearDelta[slot]).isFinite() &&
                   (bodies[body].angularVelocity +
                    stagedAngularDelta[slot]).isFinite();
        }

        // A closed all-dynamic contact component cannot change total linear
        // momentum. Nonlinear pose reconstruction can nevertheless leave an
        // actor-order-dependent uniform translation after the PositionAL
        // response has been removed. Restore only that exact J-nullspace from
        // the contact-free inertial prediction; never replace the bodies'
        // relative velocity or any other velocity owner's state.
        bool closedDynamicComponent = true;
        for (physx::PxU32 row = 0; row < pointCount; ++row) {
          const AvbdDynamicContactConePoint &point = pointCache[
              packedContacts[componentBegin + row]];
          closedDynamicComponent =
              closedDynamicComponent &&
              point.bodyIndex[0] < numBodies &&
              point.bodyIndex[1] < numBodies &&
              physx::PxAbs(point.linearScale[0] - 1.0f) <= 1.0e-6f &&
              physx::PxAbs(point.linearScale[1] - 1.0f) <= 1.0e-6f;
        }
        double totalMass = 0.0;
        AvbdDoubleVec3 desiredMomentum;
        AvbdDoubleVec3 prospectiveMomentum;
        if (finite && closedDynamicComponent) {
          for (physx::PxU32 slot = 0;
               slot < componentBodies.size(); ++slot) {
            const physx::PxU32 bodyIndex = componentBodies[slot];
            const AvbdSolverBody &body = bodies[bodyIndex];
            if (body.invMass <= 0.0f || body.lockFlags != 0u) {
              closedDynamicComponent = false;
              break;
            }
            const double mass = 1.0 / double(body.invMass);
            const physx::PxVec3 desiredLinear =
                (body.inertialPosition - body.prevPosition) * invDt *
                (*linearPoseVelocityGain)[bodyIndex];
            const physx::PxVec3 prospectiveLinear =
                body.linearVelocity + stagedLinearDelta[slot];
            if (!std::isfinite(mass) || mass <= 0.0 ||
                !desiredLinear.isFinite() ||
                !prospectiveLinear.isFinite()) {
              finite = false;
              break;
            }
            totalMass += mass;
            desiredMomentum += AvbdDoubleVec3(desiredLinear) * mass;
            prospectiveMomentum +=
                AvbdDoubleVec3(prospectiveLinear) * mass;
          }
        }
        if (finite && closedDynamicComponent) {
          if (!std::isfinite(totalMass) || totalMass <= 0.0 ||
              !desiredMomentum.isFinite() ||
              !prospectiveMomentum.isFinite()) {
            finite = false;
          } else {
            const AvbdDoubleVec3 translation =
                (desiredMomentum - prospectiveMomentum) *
                (1.0 / totalMass);
            const physx::PxVec3 correction(
                physx::PxReal(translation.x),
                physx::PxReal(translation.y),
                physx::PxReal(translation.z));
            if (!translation.isFinite() || !correction.isFinite()) {
              finite = false;
            } else {
              for (physx::PxU32 slot = 0;
                   slot < componentBodies.size(); ++slot) {
                stagedLinearDelta[slot] += correction;
                finite = finite &&
                    (bodies[componentBodies[slot]].linearVelocity +
                     stagedLinearDelta[slot]).isFinite();
              }
            }
          }
        }
        for (physx::PxU32 row = 0; row < pointCount && finite; ++row) {
          finite = finite && std::isfinite(normalImpulse[row]) &&
                   normalImpulse[row] >= 0.0;
        }
        for (physx::PxU32 row = 0;
             row < tangentRowCount && finite; ++row)
          finite = std::isfinite(tangentImpulse[row]);

        if (finite) {
          rootSolved[root] = 1;
          for (physx::PxU32 slot = 0;
               slot < componentBodies.size(); ++slot) {
            const physx::PxU32 body = componentBodies[slot];
            candidateLinearDelta[body] = stagedLinearDelta[slot];
            candidateAngularDelta[body] = stagedAngularDelta[slot];
          }
          for (physx::PxU32 row = 0; row < pointCount; ++row) {
            const physx::PxU32 contactIndex =
                packedContacts[componentBegin + row];
            candidateImpulse[contactIndex * 3u] = normalImpulse[row];
            candidateImpulse[contactIndex * 3u + 1u] = 0.0;
            candidateImpulse[contactIndex * 3u + 2u] = 0.0;
          }
          for (physx::PxU32 anchor = 0; anchor < anchorCount; ++anchor) {
            const physx::PxU32 contactIndex = anchorContacts[anchor];
            candidateImpulse[contactIndex * 3u + 1u] =
                tangentImpulse[anchor * 2u];
            candidateImpulse[contactIndex * 3u + 2u] =
                tangentImpulse[anchor * 2u + 1u];
          }
        }
      }
      for (physx::PxU32 slot = 0; slot < componentBodies.size(); ++slot)
        bodySlot[componentBodies[slot]] = PX_MAX_U32;
    }

    // Atomic fail-closed semantics: an unsolved active root must not be
    // mistaken for a converged cut merely because no candidate delta was
    // published for it.
    for (physx::PxU32 begin = 0; begin < numContacts;) {
      const physx::PxU32 end = rangeEnd[begin];
      if (activeRange[begin]) {
        const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
        const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
        const physx::PxU32 dynamicBody =
            bodyA < numBodies ? bodyA : bodyB;
        if (!rootSolved[parent[dynamicBody]])
          return;
      }
      begin = end;
    }

    for (physx::PxU32 begin = 0; begin < numContacts;) {
      promoteRange[begin] = 0;
      begin = rangeEnd[begin];
    }
    ++visitEpoch;
    if (visitEpoch == 0u) {
      for (physx::PxU32 contact = 0; contact < numContacts; ++contact)
        managerVisitStamp[contact] = 0u;
      visitEpoch = 1u;
    }
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      if (!activeBody[body] || !rootSolved[parent[body]])
        continue;
      const physx::PxU32 *incidentContacts = nullptr;
      physx::PxU32 incidentCount = 0u;
      if (!getAvbdBodyContactRange(
              contactMap, body, incidentContacts, incidentCount))
        continue;
      for (physx::PxU32 incident = 0; incident < incidentCount;
           ++incident) {
        const physx::PxU32 contactIndex = incidentContacts[incident];
        if (contactIndex >= numContacts)
          continue;
        const physx::PxU32 begin = rangeBegin[contactIndex];
        if (!supportedRange[begin] || activeRange[begin] ||
            managerVisitStamp[begin] == visitEpoch)
          continue;
        managerVisitStamp[begin] = visitEpoch;

        double residual = 0.0;
        double velocityScale = 1.0;
        const physx::PxU32 end = rangeEnd[begin];
        const auto evaluateCandidateVelocity =
            [&](const AvbdDynamicContactConePoint &point,
                double velocity[3]) -> bool {
          for (physx::PxU32 component = 0; component < 3u; ++component)
            velocity[component] =
                point.poseVelocityMinusTarget[component];
          for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
            const physx::PxU32 endpointBody = point.bodyIndex[endpoint];
            if (endpointBody >= numBodies)
              continue;
            for (physx::PxU32 component = 0; component < 3u;
                 ++component) {
              velocity[component] += double(
                  candidateLinearDelta[endpointBody].dot(
                      point.axis[component][endpoint]) +
                  candidateAngularDelta[endpointBody].dot(
                      point.angularJacobian[component][endpoint]));
            }
          }
          return std::isfinite(velocity[0]) &&
                 std::isfinite(velocity[1]) &&
                 std::isfinite(velocity[2]);
        };
        bool managerFinite = true;
        for (physx::PxU32 pointIndex = begin;
             pointIndex < end && managerFinite; ++pointIndex) {
          const AvbdDynamicContactConePoint &point = pointCache[pointIndex];
          double velocity[3];
          managerFinite =
              managerFinite && evaluateCandidateVelocity(point, velocity);
          const double dormantNormal =
              candidateImpulse[pointIndex * 3u];
          const double mapResidual =
              dormantNormal > 1.0e-10
                  ? std::fabs(velocity[0])
                  : std::max(0.0, -velocity[0]);
          residual = std::max(residual, mapResidual);
          velocityScale = std::max(
              velocityScale,
              std::max(std::fabs(point.poseVelocityMinusTarget[0]),
                       std::fabs(velocity[0])));
          managerFinite = managerFinite &&
                          std::isfinite(dormantNormal) &&
                          dormantNormal >= 0.0 &&
                          std::isfinite(residual) &&
                          std::isfinite(velocityScale);
        }

        // A dormant support is a valid frontier cut only when its complete
        // non-associated Coulomb map is satisfied.  Check each NP patch once
        // using the same two-dimensional disk, patch-normal budget, response
        // metric, and material velocity epoch as the active tangent solve.
        // Contact-manager manifolds are small, so this bounded local scan is
        // both deterministic and cheaper than introducing associative lookup.
        for (physx::PxU32 patchLeader = begin;
             patchLeader < end && managerFinite; ++patchLeader) {
          const AvbdContactConstraint &leader = contacts[patchLeader];
          const AvbdDynamicContactConePoint &leaderPoint =
              pointCache[patchLeader];
          if (leaderPoint.friction <= 0.0 ||
              leader.frictionAnchorCount == 0u)
            continue;

          bool patchSeen = false;
          for (physx::PxU32 prior = begin; prior < patchLeader; ++prior) {
            if (contacts[prior].contactPatchIndex ==
                leader.contactPatchIndex) {
              patchSeen = true;
              break;
            }
          }
          if (patchSeen)
            continue;

          double patchNormal = 0.0;
          physx::PxU32 anchorContacts[2] = {PX_MAX_U32, PX_MAX_U32};
          physx::PxU32 anchorCount = 0u;
          for (physx::PxU32 patchPoint = begin; patchPoint < end;
               ++patchPoint) {
            const AvbdContactConstraint &contact = contacts[patchPoint];
            if (contact.contactPatchIndex != leader.contactPatchIndex)
              continue;
            patchNormal += candidateImpulse[patchPoint * 3u];
            const physx::PxU8 mask = contact.frictionAnchorMask;
            if ((mask & 1u) != 0u) {
              if (anchorCount < 2u)
                anchorContacts[anchorCount] = patchPoint;
              ++anchorCount;
            }
            if ((mask & 2u) != 0u) {
              if (anchorCount < 2u)
                anchorContacts[anchorCount] = patchPoint;
              ++anchorCount;
            }
          }

          const physx::PxU32 expectedAnchorCount =
              leader.frictionAnchorCount;
          if (anchorCount != expectedAnchorCount || anchorCount == 0u ||
              anchorCount > 2u ||
              (anchorCount == 2u &&
               anchorContacts[0] == anchorContacts[1]) ||
              !std::isfinite(patchNormal) ||
              !std::isfinite(leaderPoint.friction)) {
            managerFinite = false;
            break;
          }

          const double tangentLimit =
              leaderPoint.friction * std::max(0.0, patchNormal) /
              double(anchorCount);
          for (physx::PxU32 anchor = 0;
               anchor < anchorCount && managerFinite; ++anchor) {
            const physx::PxU32 anchorContactIndex =
                anchorContacts[anchor];
            const AvbdDynamicContactConePoint &point =
                pointCache[anchorContactIndex];
            double velocity[3];
            managerFinite = evaluateCandidateVelocity(point, velocity);

            double pairDiagonal[2] = {0.0, 0.0};
            for (physx::PxU32 tangent = 0; tangent < 2u; ++tangent) {
              const physx::PxU32 component = tangent + 1u;
              for (physx::PxU32 endpoint = 0; endpoint < 2u;
                   ++endpoint) {
                const physx::PxU32 endpointBody =
                    point.bodyIndex[endpoint];
                if (endpointBody >= numBodies)
                  continue;
                const AvbdSolverBody &endpointSolverBody =
                    bodies[endpointBody];
                physx::PxVec3 linear =
                    point.axis[component][endpoint] *
                    (endpointSolverBody.invMass *
                     point.linearScale[endpoint]);
                physx::PxVec3 angularImpulse =
                    point.angularJacobian[component][endpoint];
                endpointSolverBody.projectLockedAngularVector(
                    angularImpulse);
                physx::PxVec3 angular =
                    endpointSolverBody.invInertiaWorld.transform(
                        angularImpulse) *
                    point.angularScale[endpoint];
                endpointSolverBody.projectLockedLinearVector(linear);
                endpointSolverBody.projectLockedAngularVector(angular);
                pairDiagonal[tangent] += double(
                    point.axis[component][endpoint].dot(linear) +
                    point.angularJacobian[component][endpoint].dot(
                        angular));
              }
            }
            const double diskDiagonal =
                0.5 * (pairDiagonal[0] + pairDiagonal[1]);
            if (!managerFinite || !std::isfinite(tangentLimit) ||
                tangentLimit < 0.0 || !std::isfinite(diskDiagonal) ||
                diskDiagonal <= 1.0e-12) {
              managerFinite = false;
              break;
            }

            const double step = 1.0 / diskDiagonal;
            const double dormantTangent0 =
                candidateImpulse[anchorContactIndex * 3u + 1u];
            const double dormantTangent1 =
                candidateImpulse[anchorContactIndex * 3u + 2u];
            double projected0 =
                dormantTangent0 - step * velocity[1];
            double projected1 =
                dormantTangent1 - step * velocity[2];
            if (!projectAvbdTangentDisk(
                    tangentLimit, projected0, projected1)) {
              managerFinite = false;
              break;
            }
            const double tangentResidual =
                std::sqrt(
                    (dormantTangent0 - projected0) *
                        (dormantTangent0 - projected0) +
                    (dormantTangent1 - projected1) *
                        (dormantTangent1 - projected1)) /
                step;
            residual = std::max(residual, tangentResidual);
            velocityScale = std::max(
                velocityScale,
                std::max(
                    std::sqrt(velocity[1] * velocity[1] +
                              velocity[2] * velocity[2]),
                    std::max(
                        std::fabs(point.poseVelocityMinusTarget[1]),
                        std::fabs(point.poseVelocityMinusTarget[2]))));
            managerFinite = std::isfinite(dormantTangent0) &&
                            std::isfinite(dormantTangent1) &&
                            std::isfinite(residual) &&
                            std::isfinite(velocityScale);
          }
        }
        const double residualTolerance =
            1.0e-7 + 2.0e-6 * velocityScale;
        if (!managerFinite || !std::isfinite(residual) ||
            residual > residualTolerance)
          promoteRange[begin] = 1;
      }
    }

    bool promoted = false;
    for (physx::PxU32 begin = 0; begin < numContacts;) {
      const physx::PxU32 end = rangeEnd[begin];
      if (promoteRange[begin]) {
        promoted = true;
        activeRange[begin] = 1;
        ++activeRangeCount;
        const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
        const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
        if (bodyA < numBodies && bodyB < numBodies)
          uniteBodies(bodyA, bodyB);
      }
      begin = end;
    }
    if (!promoted) {
      closureConverged = true;
      break;
    }
  }

  if (!closureConverged)
    return;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 root = findRoot(body);
    if (!activeBody[body] || !rootSolved[root])
      continue;
    bodies[body].linearVelocity += candidateLinearDelta[body];
    bodies[body].angularVelocity += candidateAngularDelta[body];
  }
  for (physx::PxU32 begin = 0; begin < numContacts;) {
    const physx::PxU32 end = rangeEnd[begin];
    if (supportedRange[begin] && activeRange[begin]) {
      const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
      const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
      const physx::PxU32 dynamicBody =
          bodyA < numBodies ? bodyA : bodyB;
      if (rootSolved[findRoot(dynamicBody)]) {
        for (physx::PxU32 contactIndex = begin; contactIndex < end;
             ++contactIndex) {
          AvbdContactConstraint &contact = contacts[contactIndex];
          setRigidMaterialConsumed(contact, true);
          contact.velocityNormalImpulse = physx::PxReal(
              candidateImpulse[contactIndex * 3u]);
          contact.frictionSweepImpulse =
              contact.tangent0 * physx::PxReal(
                  candidateImpulse[contactIndex * 3u + 1u]) +
              contact.tangent1 * physx::PxReal(
                  candidateImpulse[contactIndex * 3u + 2u]);
        }
      }
    }
    begin = end;
  }

  if (std::getenv("PHYSX_AVBD_RIGID_IMPACT_TRACE")) {
    double maximumTarget = 0.0;
    double normalReportBase = 0.0;
    double normalIncrement = 0.0;
    for (physx::PxU32 begin = 0; begin < numContacts;) {
      const physx::PxU32 end = rangeEnd[begin];
      if (activeRange[begin]) {
        for (physx::PxU32 contactIndex = begin; contactIndex < end;
             ++contactIndex) {
          maximumTarget = std::max(
              maximumTarget, pointCache[contactIndex].normalTarget);
          normalReportBase += pointCache[contactIndex].reportImpulse[0];
          normalIncrement += candidateImpulse[contactIndex * 3u];
        }
      }
      begin = end;
    }
    {
      double maximumLinearDelta = 0.0;
      double maximumAngularDelta = 0.0;
      AvbdDoubleVec3 momentumBefore;
      AvbdDoubleVec3 momentumDelta;
      for (physx::PxU32 body = 0; body < numBodies; ++body) {
        maximumLinearDelta = std::max(
            maximumLinearDelta,
            double(candidateLinearDelta[body].magnitude()));
        maximumAngularDelta = std::max(
            maximumAngularDelta,
            double(candidateAngularDelta[body].magnitude()));
        if (activeBody[body] && bodies[body].invMass > 0.0f) {
          const double mass = 1.0 / double(bodies[body].invMass);
          momentumDelta += AvbdDoubleVec3(candidateLinearDelta[body]) * mass;
          momentumBefore +=
              AvbdDoubleVec3(bodies[body].linearVelocity -
                             candidateLinearDelta[body]) * mass;
        }
      }
      std::printf(
          "[AVBD_RIGID_FRONTIER] rounds=%u activeManagers=%u "
          "supportedManagers=%u target=%.9g normalBase=%.9g "
          "normalTotal=%.9g maxLinearDelta=%.9g "
          "maxAngularDelta=%.9g momentumBefore=(%.9g,%.9g,%.9g) "
          "momentumDelta=(%.9g,%.9g,%.9g) closure=%s\n",
          closureRound + 1u, activeRangeCount, supportedRangeCount,
          maximumTarget, normalReportBase, normalIncrement,
          maximumLinearDelta, maximumAngularDelta,
          momentumBefore.x, momentumBefore.y, momentumBefore.z,
          momentumDelta.x, momentumDelta.y, momentumDelta.z,
          closureConverged ? "PASS" : "FAIL");
    }
  }
}

#if 0
/**
 * Solve every simultaneously detected dynamic impact from one immutable
 * velocity epoch.  Contact-manager ranges remain useful manifold metadata,
 * but are not solver components: a projectile can touch several managers in
 * one frame and all of those rows share body response.
 *
 * Components are built with dense union/find indices and a counting CSR.
 * There is no hash lookup, pointer identity, or manager-order mutation in the
 * solve.  Disconnected impact components are committed independently; every
 * component is assembled completely before any of its body velocities or
 * reports are changed.
 */
static void applyDynamicDynamicImpactComponents(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceThreshold) {
  if (!bodies || !contacts || numBodies == 0u || numContacts == 0u ||
      !linearVelAtSolveStart || !angularVelAtSolveStart || dt <= 0.0f ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies)
    return;
  const physx::PxReal invDt = 1.0f / dt;

  physx::PxArray<physx::PxU32> parent(numBodies);
  physx::PxArray<physx::PxU8> impactContact(numContacts);
  physx::PxArray<physx::PxU8> supportedContact(numContacts);
  physx::PxArray<physx::PxU8> impactBody(numBodies);
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    parent[body] = body;
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    impactBody[body] = 0;
  for (physx::PxU32 contact = 0; contact < numContacts; ++contact) {
    impactContact[contact] = 0;
    supportedContact[contact] = 0;
  }

  const auto findRoot = [&](physx::PxU32 body) -> physx::PxU32 {
    physx::PxU32 root = body;
    while (parent[root] != root)
      root = parent[root];
    while (parent[body] != body) {
      const physx::PxU32 next = parent[body];
      parent[body] = root;
      body = next;
    }
    return root;
  };
  const auto uniteBodies = [&](physx::PxU32 bodyA,
                               physx::PxU32 bodyB) {
    physx::PxU32 rootA = findRoot(bodyA);
    physx::PxU32 rootB = findRoot(bodyB);
    if (rootA == rootB)
      return;
    if (rootB < rootA)
      std::swap(rootA, rootB);
    parent[rootB] = rootA;
  };

  // Freeze impact eligibility for every manager before any velocity commit.
  // The report defaults to the PositionAL material force integrated over dt;
  // a successful impact component overwrites it with AL + impact increment.
  for (physx::PxU32 begin = 0; begin < numContacts;) {
    const physx::PxU32 managerIndex = contacts[begin].contactManagerIndex;
    physx::PxU32 end = begin + 1u;
    while (end < numContacts &&
           contacts[end].contactManagerIndex == managerIndex)
      ++end;
    if (managerIndex == PX_MAX_U32 ||
        !isAvbdManifoldRestitutionResidual(contacts[begin])) {
      begin = end;
      continue;
    }

    bool supported = true;
    bool impactEligible = false;
    physx::PxU32 bodyA = PX_MAX_U32;
    physx::PxU32 bodyB = PX_MAX_U32;
    for (physx::PxU32 contactIndex = begin; contactIndex < end;
         ++contactIndex) {
      AvbdDynamicContactConePoint point;
      if (!buildAvbdDynamicImpactPoint(
              bodies, numBodies, contacts, contactIndex,
              *linearVelAtSolveStart, *angularVelAtSolveStart,
              invDt, bounceThreshold, point)) {
        supported = false;
        break;
      }
      if (contactIndex == begin) {
        bodyA = point.bodyIndex[0];
        bodyB = point.bodyIndex[1];
      } else if (point.bodyIndex[0] != bodyA ||
                 point.bodyIndex[1] != bodyB) {
        supported = false;
        break;
      }
      AvbdContactConstraint &contact = contacts[contactIndex];
      contact.velocityNormalImpulse = physx::PxReal(point.alImpulse[0]);
      contact.frictionSweepImpulse =
          contact.tangent0 * physx::PxReal(point.alImpulse[1]) +
          contact.tangent1 * physx::PxReal(point.alImpulse[2]);
      impactEligible = impactEligible || point.normalTarget > 0.0;
    }
    if (supported) {
      for (physx::PxU32 contactIndex = begin; contactIndex < end;
           ++contactIndex)
        supportedContact[contactIndex] = 1;
      if (impactEligible) {
        uniteBodies(bodyA, bodyB);
        impactBody[bodyA] = 1;
        impactBody[bodyB] = 1;
        for (physx::PxU32 contactIndex = begin; contactIndex < end;
             ++contactIndex)
          impactContact[contactIndex] = 1;
      }
    }
    begin = end;
  }

  // First active-frontier closure: add every supported dynamic manifold
  // directly incident to an impact body.  These zero-target rows contribute
  // their cross response in the same NCP, so a struck box is not solved as a
  // free body while it is already supported by neighbouring boxes.  The
  // frontier is frozen from the pre-commit body mask; it does not flood-fill
  // a resting stack merely because contacts are topologically connected.
  for (physx::PxU32 begin = 0; begin < numContacts;) {
    const physx::PxU32 managerIndex = contacts[begin].contactManagerIndex;
    physx::PxU32 end = begin + 1u;
    while (end < numContacts &&
           contacts[end].contactManagerIndex == managerIndex)
      ++end;
    if (supportedContact[begin] && !impactContact[begin]) {
      const physx::PxU32 bodyA = contacts[begin].header.bodyIndexA;
      const physx::PxU32 bodyB = contacts[begin].header.bodyIndexB;
      if (bodyA < numBodies && bodyB < numBodies &&
          (impactBody[bodyA] || impactBody[bodyB])) {
        uniteBodies(bodyA, bodyB);
        for (physx::PxU32 contactIndex = begin; contactIndex < end;
             ++contactIndex)
          impactContact[contactIndex] = 1;
      }
    }
    begin = end;
  }

  // Stable counting CSR: root -> contact indices.  Roots are dense body
  // indices, so associative lookup would add cost and nondeterminism without
  // carrying any information.
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    parent[body] = findRoot(body);
  physx::PxArray<physx::PxU32> rootOffsets(numBodies + 1u);
  for (physx::PxU32 root = 0; root <= numBodies; ++root)
    rootOffsets[root] = 0;
  physx::PxU32 totalImpactContacts = 0;
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    if (!impactContact[contactIndex])
      continue;
    const physx::PxU32 bodyA = contacts[contactIndex].header.bodyIndexA;
    if (bodyA >= numBodies)
      return;
    ++rootOffsets[parent[bodyA] + 1u];
    ++totalImpactContacts;
  }
  if (totalImpactContacts == 0u)
    return;
  for (physx::PxU32 root = 1; root <= numBodies; ++root)
    rootOffsets[root] += rootOffsets[root - 1u];
  physx::PxArray<physx::PxU32> rootWriteOffsets(numBodies);
  for (physx::PxU32 root = 0; root < numBodies; ++root)
    rootWriteOffsets[root] = rootOffsets[root];
  physx::PxArray<physx::PxU32> packedContacts(totalImpactContacts);
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    if (!impactContact[contactIndex])
      continue;
    const physx::PxU32 root =
        parent[contacts[contactIndex].header.bodyIndexA];
    packedContacts[rootWriteOffsets[root]++] = contactIndex;
  }

  physx::PxArray<physx::PxU32> bodySlot(numBodies);
  for (physx::PxU32 body = 0; body < numBodies; ++body)
    bodySlot[body] = PX_MAX_U32;

  for (physx::PxU32 root = 0; root < numBodies; ++root) {
    const physx::PxU32 componentBegin = rootOffsets[root];
    const physx::PxU32 componentEnd = rootOffsets[root + 1u];
    const physx::PxU32 pointCount = componentEnd - componentBegin;
    if (pointCount == 0u)
      continue;
    if (pointCount > PX_MAX_U32 / 3u)
      continue;
    const physx::PxU32 rowCount = pointCount * 3u;
    if (rowCount > PX_MAX_U32 / rowCount)
      continue;

    physx::PxArray<AvbdDynamicContactConePoint> points(pointCount);
    bool finite = true;
    for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
         ++pointIndex) {
      finite = finite && buildAvbdDynamicImpactPoint(
                             bodies, numBodies, contacts,
                             packedContacts[componentBegin + pointIndex],
                             *linearVelAtSolveStart,
                             *angularVelAtSolveStart, invDt,
                             bounceThreshold, points[pointIndex]);
    }
    if (!finite)
      continue;

    physx::PxArray<physx::PxVec3> linearResponse(rowCount * 2u);
    physx::PxArray<physx::PxVec3> angularResponse(rowCount * 2u);
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      const physx::PxU32 pointIndex = column / 3u;
      const physx::PxU32 component = column % 3u;
      for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
        const AvbdSolverBody &body =
            bodies[points[pointIndex].bodyIndex[endpoint]];
        physx::PxVec3 linear =
            points[pointIndex].axis[component][endpoint] *
            (body.invMass * points[pointIndex].linearScale[endpoint]);
        physx::PxVec3 angularImpulse =
            points[pointIndex].angularJacobian[component][endpoint];
        body.projectLockedAngularVector(angularImpulse);
        physx::PxVec3 angular =
            body.invInertiaWorld.transform(angularImpulse) *
            points[pointIndex].angularScale[endpoint];
        body.projectLockedLinearVector(linear);
        body.projectLockedAngularVector(angular);
        linearResponse[column * 2u + endpoint] = linear;
        angularResponse[column * 2u + endpoint] = angular;
      }
    }

    physx::PxArray<double> response(rowCount * rowCount);
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      const physx::PxU32 rowPoint = row / 3u;
      const physx::PxU32 rowComponent = row % 3u;
      for (physx::PxU32 column = row; column < rowCount; ++column) {
        const physx::PxU32 columnPoint = column / 3u;
        double value = 0.0;
        for (physx::PxU32 rowEndpoint = 0; rowEndpoint < 2u;
             ++rowEndpoint) {
          for (physx::PxU32 columnEndpoint = 0; columnEndpoint < 2u;
               ++columnEndpoint) {
            if (points[rowPoint].bodyIndex[rowEndpoint] !=
                points[columnPoint].bodyIndex[columnEndpoint])
              continue;
            value += double(
                points[rowPoint].axis[rowComponent][rowEndpoint].dot(
                    linearResponse[column * 2u + columnEndpoint]) +
                points[rowPoint]
                    .angularJacobian[rowComponent][rowEndpoint]
                    .dot(angularResponse[column * 2u + columnEndpoint]));
          }
        }
        response[row * rowCount + column] = value;
        response[column * rowCount + row] = value;
      }
    }

    physx::PxArray<double> q(rowCount);
    physx::PxArray<double> impactImpulses(rowCount);
    physx::PxArray<double> scratch(rowCount);
    physx::PxArray<double> friction(pointCount);
    for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
         ++pointIndex) {
      friction[pointIndex] = points[pointIndex].friction;
      for (physx::PxU32 component = 0; component < 3u; ++component) {
        const physx::PxU32 row = pointIndex * 3u + component;
        q[row] = points[pointIndex].poseVelocityMinusTarget[component];
        impactImpulses[row] = 0.0;
      }
    }
    if (!solveAvbdCoulombNcpFixedPoint(
            response.begin(), q.begin(), friction.begin(), pointCount,
            impactImpulses.begin(), scratch.begin()))
      continue;

    physx::PxArray<physx::PxU32> componentBodies;
    for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
         ++pointIndex) {
      for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
        const physx::PxU32 body = points[pointIndex].bodyIndex[endpoint];
        if (bodySlot[body] == PX_MAX_U32) {
          bodySlot[body] = componentBodies.size();
          componentBodies.pushBack(body);
        }
      }
    }
    physx::PxArray<physx::PxVec3> linearDelta(componentBodies.size());
    physx::PxArray<physx::PxVec3> angularDelta(componentBodies.size());
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < componentBodies.size(); ++bodyIndex) {
      linearDelta[bodyIndex] = physx::PxVec3(0.0f);
      angularDelta[bodyIndex] = physx::PxVec3(0.0f);
    }
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      if (!std::isfinite(impactImpulses[row])) {
        finite = false;
        break;
      }
      const physx::PxU32 pointIndex = row / 3u;
      for (physx::PxU32 endpoint = 0; endpoint < 2u; ++endpoint) {
        const physx::PxU32 slot =
            bodySlot[points[pointIndex].bodyIndex[endpoint]];
        linearDelta[slot] +=
            linearResponse[row * 2u + endpoint] *
            physx::PxReal(impactImpulses[row]);
        angularDelta[slot] +=
            angularResponse[row * 2u + endpoint] *
            physx::PxReal(impactImpulses[row]);
      }
    }
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < componentBodies.size() && finite; ++bodyIndex) {
      const AvbdSolverBody &body = bodies[componentBodies[bodyIndex]];
      finite = linearDelta[bodyIndex].isFinite() &&
               angularDelta[bodyIndex].isFinite() &&
               (body.linearVelocity + linearDelta[bodyIndex]).isFinite() &&
               (body.angularVelocity + angularDelta[bodyIndex]).isFinite();
    }
    for (physx::PxU32 pointIndex = 0;
         pointIndex < pointCount && finite; ++pointIndex) {
      const double normalReport = points[pointIndex].alImpulse[0] +
                                  impactImpulses[pointIndex * 3u];
      const double tangent0Report = points[pointIndex].alImpulse[1] +
                                    impactImpulses[pointIndex * 3u + 1u];
      const double tangent1Report = points[pointIndex].alImpulse[2] +
                                    impactImpulses[pointIndex * 3u + 2u];
      finite = std::isfinite(normalReport) && normalReport >= 0.0 &&
               std::isfinite(tangent0Report) &&
               std::isfinite(tangent1Report);
    }
    if (finite) {
      for (physx::PxU32 bodyIndex = 0;
           bodyIndex < componentBodies.size(); ++bodyIndex) {
        AvbdSolverBody &body = bodies[componentBodies[bodyIndex]];
        body.linearVelocity += linearDelta[bodyIndex];
        body.angularVelocity += angularDelta[bodyIndex];
      }
      for (physx::PxU32 pointIndex = 0; pointIndex < pointCount;
           ++pointIndex) {
        AvbdContactConstraint &contact =
            contacts[points[pointIndex].contactIndex];
        contact.velocityNormalImpulse = physx::PxReal(
            points[pointIndex].alImpulse[0] +
            impactImpulses[pointIndex * 3u]);
        contact.frictionSweepImpulse =
            contact.tangent0 *
                physx::PxReal(points[pointIndex].alImpulse[1] +
                              impactImpulses[pointIndex * 3u + 1u]) +
            contact.tangent1 *
                physx::PxReal(points[pointIndex].alImpulse[2] +
                              impactImpulses[pointIndex * 3u + 2u]);
      }
    }
    for (physx::PxU32 bodyIndex = 0;
         bodyIndex < componentBodies.size(); ++bodyIndex)
      bodySlot[componentBodies[bodyIndex]] = PX_MAX_U32;
  }
}
#endif

#if 0
static void applyDynamicDynamicMaterialConeManifolds(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    physx::PxReal dt, physx::PxReal bounceThreshold) {
  if (!linearVelAtSolveStart || !angularVelAtSolveStart || dt <= 0.0f ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies)
    return;
  const physx::PxReal invDt = 1.0f / dt;

  for (physx::PxU32 begin = 0; begin < numContacts;) {
    const physx::PxU32 managerIndex =
        contacts[begin].contactManagerIndex;
    physx::PxU32 end = begin + 1u;
    while (end < numContacts &&
           contacts[end].contactManagerIndex == managerIndex)
      ++end;
    if (managerIndex == PX_MAX_U32 || end - begin > 8u ||
        !isAvbdManifoldRestitutionResidual(contacts[begin])) {
      begin = end;
      continue;
    }

    const physx::PxU32 bodyIndex[2] = {
        contacts[begin].header.bodyIndexA,
        contacts[begin].header.bodyIndexB};
    if (bodyIndex[0] >= numBodies || bodyIndex[1] >= numBodies ||
        bodies[bodyIndex[0]].invMass <= 0.0f ||
        bodies[bodyIndex[1]].invMass <= 0.0f) {
      begin = end;
      continue;
    }

    const physx::PxReal linearScale[2] = {
        contacts[begin].invMassScaleA,
        contacts[begin].invMassScaleB};
    const physx::PxReal angularScale[2] = {
        contacts[begin].invInertiaScaleA,
        contacts[begin].invInertiaScaleB};
    bool supportedManifold =
        physx::PxIsFinite(linearScale[0]) &&
        physx::PxIsFinite(linearScale[1]) &&
        physx::PxIsFinite(angularScale[0]) &&
        physx::PxIsFinite(angularScale[1]) &&
        linearScale[0] >= 0.0f && linearScale[1] >= 0.0f &&
        angularScale[0] >= 0.0f && angularScale[1] >= 0.0f;
    AvbdDynamicContactConePoint points[8];
    physx::PxU32 pointCount = 0;
    for (physx::PxU32 contactIndex = begin;
         contactIndex < end && supportedManifold; ++contactIndex) {
      AvbdContactConstraint &contact = contacts[contactIndex];
      const bool sameResponseScales =
          physx::PxAbs(contact.invMassScaleA - linearScale[0]) <= 1.0e-6f &&
          physx::PxAbs(contact.invMassScaleB - linearScale[1]) <= 1.0e-6f &&
          physx::PxAbs(contact.invInertiaScaleA - angularScale[0]) <=
              1.0e-6f &&
          physx::PxAbs(contact.invInertiaScaleB - angularScale[1]) <=
              1.0e-6f;
      if (!isAvbdManifoldRestitutionResidual(contact) ||
          contact.header.bodyIndexA != bodyIndex[0] ||
          contact.header.bodyIndexB != bodyIndex[1] ||
          !physx::PxIsFinite(contact.maxImpulse) ||
          contact.maxImpulse < PX_MAX_REAL ||
          !physx::PxIsFinite(contact.restitution) ||
          contact.restitution < 0.0f || contact.restitution > 1.0f ||
          !contact.targetVelocity.isFinite() || !sameResponseScales) {
        supportedManifold = false;
        break;
      }

      // A normal contact-modification target has different unilateral and
      // restitution composition rules and is deliberately left on the
      // legacy owner. Tangential conveyor targets fit this cone objective;
      // ordinary rigid contacts enter with a zero target.
      if (physx::PxAbs(
              contact.targetVelocity.dot(contact.contactNormal)) >
          1.0e-6f) {
        supportedManifold = false;
        break;
      }

      AvbdDynamicContactConePoint &point = points[pointCount];
      point.contactIndex = contactIndex;
      const physx::PxVec3 basis[3] = {
          contact.contactNormal, contact.tangent0, contact.tangent1};
      if (!basis[0].isFinite() || !basis[1].isFinite() ||
          !basis[2].isFinite() ||
          basis[0].magnitudeSquared() <= 1.0e-12f ||
          basis[1].magnitudeSquared() <= 1.0e-12f ||
          basis[2].magnitudeSquared() <= 1.0e-12f) {
        supportedManifold = false;
        break;
      }
      for (physx::PxU32 component = 0; component < 3u; ++component) {
        point.axis[component][0] = basis[component];
        point.axis[component][1] = -basis[component];
      }
      const AvbdMaterialContactGeometry geometry =
          buildAvbdMaterialContactGeometry(
              contact, bodies, numBodies, invDt);
      const physx::PxVec3 solveStartArm[2] = {
          geometry.solveStartMaterialArmA,
          geometry.solveStartMaterialArmB};
      const physx::PxVec3 materialArm[2] = {
          geometry.materialArmA, geometry.materialArmB};
      physx::PxReal solveStartRelativeVelocity[3] = {};
      physx::PxReal poseRelativeVelocity[3] = {};
      for (physx::PxU32 endPoint = 0; endPoint < 2; ++endPoint) {
        const AvbdSolverBody &body = bodies[bodyIndex[endPoint]];
        for (physx::PxU32 component = 0; component < 3u; ++component) {
          const physx::PxVec3 solveStartAngularJacobian =
              solveStartArm[endPoint].cross(
                  point.axis[component][endPoint]);
          point.angularJacobian[component][endPoint] =
              materialArm[endPoint].cross(
                  point.axis[component][endPoint]);
          solveStartRelativeVelocity[component] +=
              (*linearVelAtSolveStart)[bodyIndex[endPoint]].dot(
                  point.axis[component][endPoint]) +
              (*angularVelAtSolveStart)[bodyIndex[endPoint]].dot(
                  solveStartAngularJacobian);
          poseRelativeVelocity[component] +=
              body.linearVelocity.dot(point.axis[component][endPoint]) +
              body.angularVelocity.dot(
                  point.angularJacobian[component][endPoint]);
        }
      }
      // The material row remains inelastic when restitution is below either
      // TGS eligibility gate.  Dropping the row entirely would retain the
      // separating BDF velocity generated by the position AL correction.
      const physx::PxReal approach = -solveStartRelativeVelocity[0];
      const physx::PxReal normalTargetVelocity =
          approach > bounceThreshold &&
                  approach > contact.detectionSeparation * invDt
              ? contact.restitution * approach
              : 0.0f;
      point.alImpulse[0] = double(
          physx::PxMax(0.0f, -contact.header.lambda) * dt);
      point.alImpulse[1] = double(-contact.tangentLambda0 * dt);
      point.alImpulse[2] = double(-contact.tangentLambda1 * dt);
      point.friction = double(contactCoulombMu(contact));
      point.approach = double(approach);
      point.normalTarget = double(normalTargetVelocity);
      point.persistentPoint = contact.persistentPointMatched;
      if (!std::isfinite(point.alImpulse[0]) ||
          !std::isfinite(point.alImpulse[1]) ||
          !std::isfinite(point.alImpulse[2]) ||
          !std::isfinite(point.friction) || point.friction < 0.0) {
        supportedManifold = false;
        break;
      }
      point.poseVelocityMinusTarget[0] =
          double(poseRelativeVelocity[0] - normalTargetVelocity);
      point.poseVelocityMinusTarget[1] =
          double(poseRelativeVelocity[1] -
                 contact.targetVelocity.dot(contact.tangent0));
      point.poseVelocityMinusTarget[2] =
          double(poseRelativeVelocity[2] -
                 contact.targetVelocity.dot(contact.tangent1));
      ++pointCount;
    }
    if (!supportedManifold || pointCount != end - begin || pointCount == 0) {
      begin = end;
      continue;
    }

    const physx::PxU32 scalarRowCount = pointCount * 3u;
    double response[24u * 24u] = {};
    physx::PxVec3 linearResponse[24][2];
    physx::PxVec3 angularResponse[24][2];
    for (physx::PxU32 column = 0; column < scalarRowCount; ++column) {
      const physx::PxU32 pointIndex = column / 3u;
      const physx::PxU32 component = column % 3u;
      for (physx::PxU32 endPoint = 0; endPoint < 2; ++endPoint) {
        const AvbdSolverBody &body = bodies[bodyIndex[endPoint]];
        linearResponse[column][endPoint] =
            points[pointIndex].axis[component][endPoint] *
            (body.invMass * linearScale[endPoint]);
        physx::PxVec3 angularImpulse =
            points[pointIndex].angularJacobian[component][endPoint];
        body.projectLockedAngularVector(angularImpulse);
        angularResponse[column][endPoint] =
            body.invInertiaWorld.transform(angularImpulse) *
            angularScale[endPoint];
        body.projectLockedLinearVector(linearResponse[column][endPoint]);
        body.projectLockedAngularVector(
            angularResponse[column][endPoint]);
      }
    }
    for (physx::PxU32 row = 0; row < scalarRowCount; ++row) {
      const physx::PxU32 rowPoint = row / 3u;
      const physx::PxU32 rowComponent = row % 3u;
      for (physx::PxU32 column = row; column < scalarRowCount; ++column) {
        double value = 0.0;
        for (physx::PxU32 endPoint = 0; endPoint < 2; ++endPoint) {
          value += double(
              points[rowPoint].axis[rowComponent][endPoint].dot(
                  linearResponse[column][endPoint]) +
              points[rowPoint].angularJacobian[rowComponent][endPoint].dot(
                  angularResponse[column][endPoint]));
        }
        response[row * scalarRowCount + column] = value;
        response[column * scalarRowCount + row] = value;
      }
    }

    // PositionAL already owns geometric non-penetration and persistent
    // friction, exactly as in the AVBD reference algorithm.  Its nonlinear
    // final multiplier is a force certificate, not an impulse that can be
    // removed from the BDF velocity and replayed through a different
    // Jacobian.  The velocity stage therefore owns only a *new impact
    // increment*: when restitution is eligible, solve
    //
    //   0 <= d_n  _|_  uPose + W d - uRestitution >= 0,
    //   |d_t| <= mu d_n,
    //
    // and add J^T d to the pose-derived velocity.  Below the restitution
    // threshold no second physical solve is performed; the AL result remains
    // authoritative.  The sum of the AL cone impulse used for reporting and
    // the incremental impact cone remains inside the convex Coulomb cone.
    double q[24] = {};
    double impactImpulses[24] = {};
    double scratch[24] = {};
    double friction[8] = {};
    bool hasRestitutionTarget = false;
    for (physx::PxU32 point = 0; point < pointCount; ++point) {
      friction[point] = points[point].friction;
      hasRestitutionTarget =
          hasRestitutionTarget || points[point].normalTarget > 0.0;
      for (physx::PxU32 component = 0; component < 3u; ++component) {
        const physx::PxU32 row = point * 3u + component;
        q[row] = points[point].poseVelocityMinusTarget[component];
      }
    }
    if (!hasRestitutionTarget) {
      // This owner still publishes the complete material report.  Publishing
      // is deliberately side-effect free: no AL force is applied a second
      // time to body velocity.
      for (physx::PxU32 point = 0; point < pointCount; ++point) {
        AvbdContactConstraint &contact =
            contacts[points[point].contactIndex];
        contact.velocityNormalImpulse =
            physx::PxReal(points[point].alImpulse[0]);
        contact.frictionSweepImpulse =
            contact.tangent0 * physx::PxReal(points[point].alImpulse[1]) +
            contact.tangent1 * physx::PxReal(points[point].alImpulse[2]);
      }
      begin = end;
      continue;
    }
    if (!solveAvbdCoulombNcpFixedPoint(
            response, q, friction, pointCount, impactImpulses, scratch)) {
      begin = end;
      continue;
    }

    physx::PxVec3 linearImpulse[2] = {
        physx::PxVec3(0.0f), physx::PxVec3(0.0f)};
    physx::PxVec3 angularImpulse[2] = {
        physx::PxVec3(0.0f), physx::PxVec3(0.0f)};
    for (physx::PxU32 point = 0; point < pointCount; ++point) {
      AvbdContactConstraint &contact =
          contacts[points[point].contactIndex];
      contact.velocityNormalImpulse =
          physx::PxReal(points[point].alImpulse[0] +
                        impactImpulses[point * 3u]);
      contact.frictionSweepImpulse =
          contact.tangent0 *
              physx::PxReal(points[point].alImpulse[1] +
                            impactImpulses[point * 3u + 1u]) +
          contact.tangent1 *
              physx::PxReal(points[point].alImpulse[2] +
                            impactImpulses[point * 3u + 2u]);
      for (physx::PxU32 component = 0; component < 3u; ++component) {
        const physx::PxU32 row = point * 3u + component;
        const physx::PxReal impactImpulse =
            physx::PxReal(impactImpulses[row]);
        for (physx::PxU32 endPoint = 0; endPoint < 2; ++endPoint) {
          linearImpulse[endPoint] +=
              points[point].axis[component][endPoint] *
              impactImpulse;
          angularImpulse[endPoint] +=
              points[point].angularJacobian[component][endPoint] *
              impactImpulse;
        }
      }
    }
    if (std::getenv("PHYSX_AVBD_RIGID_IMPACT_TRACE")) {
      double maximumApproach = 0.0;
      double maximumTarget = 0.0;
      double normalAl = 0.0;
      double normalTotal = 0.0;
      physx::PxU32 persistentPoints = 0;
      for (physx::PxU32 point = 0; point < pointCount; ++point) {
        maximumApproach =
            std::max(maximumApproach, points[point].approach);
        maximumTarget =
            std::max(maximumTarget, points[point].normalTarget);
        normalAl += points[point].alImpulse[0];
        normalTotal += points[point].alImpulse[0] +
                       impactImpulses[point * 3u];
        persistentPoints += points[point].persistentPoint ? 1u : 0u;
      }
      if (maximumTarget > 0.0) {
        double materialEnergyDelta = 0.0;
        for (physx::PxU32 row = 0; row < scalarRowCount; ++row) {
          const physx::PxU32 point = row / 3u;
          const physx::PxU32 component = row % 3u;
          const double target =
              component == 0u ? points[point].normalTarget : 0.0;
          const double materialVelocity =
              points[point].poseVelocityMinusTarget[component] + target;
          const double correction = impactImpulses[row];
          double responseCorrection = 0.0;
          for (physx::PxU32 column = 0;
               column < scalarRowCount; ++column) {
            responseCorrection +=
                response[row * scalarRowCount + column] *
                impactImpulses[column];
          }
          materialEnergyDelta +=
              correction *
              (materialVelocity + 0.5 * responseCorrection);
        }
        std::printf(
            "[AVBD_RIGID_IMPACT] manager=%u points=%u persistent=%u "
            "approach=%.9g target=%.9g normalAl=%.9g "
            "normalTotal=%.9g linearDelta=(%.9g,%.9g) "
            "angularDelta=(%.9g,%.9g) energyDelta=%.9g "
            "normalArm=(%.9g,%.9g) tangentDelta=%.9g "
            "invMass=(%.9g,%.9g)\n",
            managerIndex, pointCount, persistentPoints,
            maximumApproach, maximumTarget, normalAl, normalTotal,
            double(linearImpulse[0].magnitude()),
            double(linearImpulse[1].magnitude()),
            double(angularImpulse[0].magnitude()),
            double(angularImpulse[1].magnitude()),
            materialEnergyDelta,
            double(points[0].angularJacobian[0][0].magnitude()),
            double(points[0].angularJacobian[0][1].magnitude()),
            std::sqrt(
                impactImpulses[1] * impactImpulses[1] +
                impactImpulses[2] * impactImpulses[2]),
            double(bodies[bodyIndex[0]].invMass),
            double(bodies[bodyIndex[1]].invMass));
      }
    }
    for (physx::PxU32 endPoint = 0; endPoint < 2; ++endPoint) {
      AvbdSolverBody &body = bodies[bodyIndex[endPoint]];
      physx::PxVec3 linearDelta =
          linearImpulse[endPoint] *
          (body.invMass * linearScale[endPoint]);
      body.projectLockedAngularVector(angularImpulse[endPoint]);
      physx::PxVec3 angularDelta =
          body.invInertiaWorld.transform(angularImpulse[endPoint]) *
          angularScale[endPoint];
      body.projectLockedLinearVector(linearDelta);
      body.projectLockedAngularVector(angularDelta);
      body.linearVelocity += linearDelta;
      body.angularVelocity += angularDelta;
    }
    begin = end;
  }
}
#endif

static const AvbdCompiledVelocityObjective *
findAvbdBodyStaticMaterialNormalManifold(
    const AvbdContactConstraint &contact) {
  const AvbdCompiledVelocityObjective *objective =
      findAvbdContactSourceObjective(
          contact.objectiveProgram,
          eCONTACT_SOURCE_MATERIAL_NORMAL);
  return objective &&
                 objective->owner ==
                     AvbdVelocityObjectiveOwner::ManifoldFinalize &&
                 objective->kind ==
                      AvbdVelocityObjectiveKind::MaterialNormal &&
                 objective->span == AvbdVelocityObjectiveSpan::Normal &&
                 objective->reconstruction ==
                     AvbdVelocityObjectiveReconstruction::PoseResidual
             ? objective
             : nullptr;
}

/**
 * Replace the position-AL normal impulse of an ordinary rigid-static
 * manifold with one total material impulse block.
 *
 * The signed correction is permitted to remove split/AL velocity, but the
 * solved total impulse remains unilateral.  This is the body-static analogue
 * of applyDynamicDynamicMaterialConeManifolds and deliberately retains every
 * point row instead of averaging a multi-point manifold into one spatial row.
 */
static void applyBodyStaticMaterialNormalManifolds(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    physx::PxArray<bool> &consumedBodies) {
  consumedBodies.resize(numBodies);
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex)
    consumedBodies[bodyIndex] = false;
  if (!bodies || !contacts || numBodies == 0 || numContacts == 0 ||
      !linearVelAtSolveStart || !angularVelAtSolveStart || dt <= 0.0f ||
      !linearPoseVelocityGain || !angularPoseVelocityGain ||
      linearVelAtSolveStart->size() != numBodies ||
      angularVelAtSolveStart->size() != numBodies ||
      linearPoseVelocityGain->size() != numBodies ||
      angularPoseVelocityGain->size() != numBodies)
    return;

  const physx::PxReal invDt = 1.0f / dt;
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    AvbdSolverBody &body = bodies[bodyIndex];
    if (body.invMass <= 0.0f)
      continue;
    const physx::PxReal linearGain =
        (*linearPoseVelocityGain)[bodyIndex];
    const physx::PxReal angularGain =
        (*angularPoseVelocityGain)[bodyIndex];
    if (!physx::PxIsFinite(linearGain) ||
        !physx::PxIsFinite(angularGain) ||
        linearGain < 0.0f || angularGain < 0.0f)
      continue;

    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    const bool hasMapRange = getAvbdBodyContactRange(
        contactMap, bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;

    physx::PxU32 rowIndices[8] = {};
    physx::PxU32 rowCount = 0;
    physx::PxU32 expectedRowCount = 0;
    physx::PxU64 objectiveKey = 0;
    physx::PxReal linearScale = 0.0f;
    physx::PxReal angularScale = 0.0f;
    bool found = false;
    bool supported = true;
    for (physx::PxU32 loopIndex = 0;
         loopIndex < loopCount && supported; ++loopIndex) {
      const physx::PxU32 contactIndex =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      if (contactIndex >= numContacts) {
        supported = false;
        break;
      }
      const AvbdContactConstraint &contact = contacts[contactIndex];
      if (hasRigidMaterialConsumed(contact))
        continue;
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      if (bodyA != bodyIndex && bodyB != bodyIndex)
        continue;
      if (!isBodyVsStaticContact(bodyA, bodyB, numBodies))
        continue;

      const AvbdCompiledVelocityObjective *objective =
          findAvbdBodyStaticMaterialNormalManifold(contact);
      const bool dynamicIsA = bodyA == bodyIndex;
      const physx::PxReal rowLinearScale =
          dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
      const physx::PxReal rowAngularScale =
          dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
      if (!objective || hasDeformableStaticAnchor(contact) ||
          hasKinematicShellAnchor(contact) ||
          !physx::PxIsFinite(contact.maxImpulse) ||
          contact.maxImpulse < PX_MAX_REAL ||
          !physx::PxIsFinite(rowLinearScale) ||
          !physx::PxIsFinite(rowAngularScale) ||
          rowLinearScale < 0.0f || rowAngularScale < 0.0f ||
          !physx::PxIsFinite(contact.restitution) ||
          contact.restitution < 0.0f || contact.restitution > 1.0f) {
        supported = false;
        break;
      }

      if (!found) {
        found = true;
        objectiveKey = objective->objectiveKey;
        expectedRowCount = objective->objectiveRowCount;
        linearScale = rowLinearScale;
        angularScale = rowAngularScale;
      } else if (objective->objectiveKey != objectiveKey ||
                 objective->objectiveRowCount != expectedRowCount ||
                 physx::PxAbs(rowLinearScale - linearScale) > 1.0e-6f ||
                 physx::PxAbs(rowAngularScale - angularScale) > 1.0e-6f) {
        supported = false;
        break;
      }
      if (rowCount >= 8u) {
        supported = false;
        break;
      }
      rowIndices[rowCount++] = contactIndex;
    }
    if (!supported || !found || rowCount == 0 ||
        rowCount != expectedRowCount)
      continue;

    AvbdRigidRestitutionRow rows[8];
    for (physx::PxU32 rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
      AvbdRigidRestitutionRow &row = rows[rowIndex];
      AvbdContactConstraint &contact = contacts[rowIndices[rowIndex]];
      const bool dynamicIsA =
          contact.header.bodyIndexA == bodyIndex;
      row.contactIndex = rowIndices[rowIndex];
      row.axis[0] = contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
      row.axis[1] = physx::PxVec3(0.0f);
      const AvbdMaterialContactGeometry geometry =
          buildAvbdMaterialContactGeometry(
              contact, bodies, numBodies, invDt);
      const physx::PxVec3 solveStartArm =
          dynamicIsA ? geometry.solveStartMaterialArmA
                     : geometry.solveStartMaterialArmB;
      const physx::PxVec3 materialArm =
          dynamicIsA ? geometry.materialArmA : geometry.materialArmB;
      const physx::PxVec3 positionAlArm =
          dynamicIsA ? geometry.positionAlArmA : geometry.positionAlArmB;
      row.angularJacobian[0] = materialArm.cross(row.axis[0]);
      row.angularJacobian[1] = physx::PxVec3(0.0f);
      row.positionAlAngularJacobian[0] =
          positionAlArm.cross(row.axis[0]);
      row.positionAlAngularJacobian[1] = physx::PxVec3(0.0f);

      const physx::PxReal staticNormalVelocity =
          geometry.staticVelocity.dot(row.axis[0]);
      const physx::PxReal solveStartRelativeVelocity =
          (*linearVelAtSolveStart)[bodyIndex].dot(row.axis[0]) +
          (*angularVelAtSolveStart)[bodyIndex].dot(
              solveStartArm.cross(row.axis[0])) -
          staticNormalVelocity;
      const physx::PxReal poseRelativeVelocity =
          body.linearVelocity.dot(row.axis[0]) +
          body.angularVelocity.dot(row.angularJacobian[0]) -
          staticNormalVelocity;
      const physx::PxReal approach = -solveStartRelativeVelocity;
      const physx::PxReal restitution =
          physx::PxMin(contact.restitution, physx::PxReal(1.0f));
      const physx::PxReal targetVelocity =
          approach > bounceThreshold &&
                  approach > contact.detectionSeparation * invDt
              ? restitution * approach
              : 0.0f;
      row.alImpulse = double(
          physx::PxMax(0.0f, -contact.header.lambda) * dt);
      row.q = double(poseRelativeVelocity - targetVelocity);
    }

    double response[8][8] = {};
    double positionAlCrossResponse[8][8] = {};
    for (physx::PxU32 column = 0; column < rowCount; ++column) {
      physx::PxVec3 linearResponse =
          rows[column].axis[0] * (body.invMass * linearScale);
      physx::PxVec3 angularResponse =
          body.invInertiaWorld.transform(
              rows[column].angularJacobian[0]) *
          angularScale;
      physx::PxVec3 positionAlAngularResponse =
          body.invInertiaWorld.transform(
              rows[column].positionAlAngularJacobian[0]) *
          (angularScale * angularGain);
      body.projectLockedLinearVector(linearResponse);
      body.projectLockedAngularVector(angularResponse);
      body.projectLockedAngularVector(positionAlAngularResponse);
      for (physx::PxU32 row = 0; row < rowCount; ++row) {
        response[row][column] = double(
            rows[row].axis[0].dot(linearResponse) +
            rows[row].angularJacobian[0].dot(angularResponse));
        positionAlCrossResponse[row][column] = double(
            rows[row].axis[0].dot(linearResponse * linearGain) +
            rows[row].angularJacobian[0].dot(
                positionAlAngularResponse));
      }
    }

    double q[8] = {};
    double totalImpulses[8] = {};
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      q[row] = rows[row].q;
      for (physx::PxU32 column = 0; column < rowCount; ++column)
        q[row] -=
            positionAlCrossResponse[row][column] *
            rows[column].alImpulse;
    }
    if (!solveAvbdRestitutionBlock(response, q, rowCount,
                                   totalImpulses))
      continue;

    physx::PxVec3 linearImpulse(0.0f);
    physx::PxVec3 angularImpulse(0.0f);
    for (physx::PxU32 row = 0; row < rowCount; ++row) {
      const physx::PxReal totalImpulse =
          physx::PxReal(totalImpulses[row]);
      const physx::PxReal alImpulse =
          physx::PxReal(rows[row].alImpulse);
      linearImpulse += rows[row].axis[0] *
                       (totalImpulse - linearGain * alImpulse);
      angularImpulse +=
          rows[row].angularJacobian[0] * totalImpulse -
          rows[row].positionAlAngularJacobian[0] *
              (angularGain * alImpulse);
      contacts[rows[row].contactIndex].velocityNormalImpulse = totalImpulse;
    }
    physx::PxVec3 linearDelta =
        linearImpulse * (body.invMass * linearScale);
    physx::PxVec3 angularDelta =
        body.invInertiaWorld.transform(angularImpulse) * angularScale;
    body.projectLockedLinearVector(linearDelta);
    body.projectLockedAngularVector(angularDelta);
    body.linearVelocity += linearDelta;
    body.angularVelocity += angularDelta;
    consumedBodies[bodyIndex] = true;
  }
}

static void applyAvbdMaterialNormalVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale,
    bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  const physx::PxReal invDt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
  const physx::PxReal bounceThreshold =
      bounceApproachThreshold > 0.0f
          ? bounceApproachThreshold
          : AvbdConstants::AVBD_BOUNCE_THRESHOLD;
  physx::PxArray<SurfaceFinalizeTopologyNode> finalizeTopologyNodes;
  physx::PxArray<SurfaceFinalizeBudgetDiagSnapshot>
      finalizeBudgetDiagSnapshots;
  physx::PxArray<bool> finalizeProbeOwnedBodies;
  physx::PxArray<bool> *finalizeProbeOwnedBodiesPtr = nullptr;
  if (stats && deformableNormalStageMask) {
    finalizeProbeOwnedBodies.resize(numBodies);
    for (physx::PxU32 body = 0; body < numBodies; ++body)
      finalizeProbeOwnedBodies[body] = false;
    finalizeProbeOwnedBodiesPtr = &finalizeProbeOwnedBodies;
    finalizeTopologyNodes.resize(numBodies);
    for (physx::PxU32 body = 0; body < numBodies; ++body) {
      finalizeTopologyNodes[body].strictOwner = 0;
      finalizeTopologyNodes[body].bodyStrictOwner = 0;
    }
    finalizeBudgetDiagSnapshots.resize(numContacts);
    for (physx::PxU32 row = 0; row < numContacts; ++row) {
      finalizeBudgetDiagSnapshots[row] =
          classifySurfaceFinalizeBudgetDiag(
              bodies, numBodies, contacts[row], dt, lengthScale,
              linearVelAtSolveStart, angularVelAtSolveStart);
    }
    discoverSurfaceFinalizeStrictOwnersPreP3K(
        bodies, numBodies, contacts, numContacts,
        linearVelAtSolveStart, finalizeTopologyNodes);
    recordSurfaceDeformableFinalizeComponentTopology(
        bodies, numBodies, contacts, numContacts,
        finalizeTopologyNodes, finalizeBudgetDiagSnapshots,
        hasJointConstraints, enableBoundedComponentProductionProbe,
        finalizeProbeOwnedBodies, stats);
  }
  // The rigid material frontier must observe the post-reconstruction pose
  // velocity before any legacy material owner modifies it.  Its atomic
  // contact-owned replacement marks every consumed row, so the remaining
  // specialized paths can handle only unclaimed material sources.
  // Hard-joint islands need the joint-nullspace mobility rather than the
  // free-body response used below. Leave them on their existing coupled path
  // until that Schur operator is published; a sequential contact projection
  // followed by joint projection would invalidate either certificate.
  if (!hasJointConstraints) {
    applyRigidImpactActiveFrontiers(
        bodies, numBodies, contacts, numContacts, contactMap,
        linearVelAtSolveStart, angularVelAtSolveStart,
        linearPoseVelocityGain, angularPoseVelocityGain,
        dt, bounceThreshold);
  }
  physx::PxArray<bool> bodyStaticMaterialNormalConsumed;
  applyBodyStaticMaterialNormalManifolds(
      bodies, numBodies, contacts, numContacts, contactMap,
      linearVelAtSolveStart, angularVelAtSolveStart,
      linearPoseVelocityGain, angularPoseVelocityGain,
      dt, bounceThreshold,
      bodyStaticMaterialNormalConsumed);
  // ---- Body-static (incl. deformable anchors) ----
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;
    if (i < bodyStaticMaterialNormalConsumed.size() &&
        bodyStaticMaterialNormalConsumed[i])
      continue;
    if (finalizeProbeOwnedBodiesPtr && (*finalizeProbeOwnedBodiesPtr)[i])
      continue;
    bool passiveMaterialComponentOwned = false;
    bool completeMaterialManifoldOwned = false;
    bool pointMaterialNormalOwned = false;
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    const bool hasMapRange = getAvbdBodyContactRange(
        contactMap, i, mapIndices, mapCount);
    const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
      if (hasRigidMaterialConsumed(contacts[c]))
        continue;
      if (hasVelocityPassiveFrictionComponentOwner(contacts[c])) {
        passiveMaterialComponentOwned = true;
        break;
      }
      if (isBodyVsStaticContact(
              contacts[c].header.bodyIndexA,
              contacts[c].header.bodyIndexB, numBodies) &&
          findAvbdCompleteManifoldObjective(
              contacts[c].objectiveProgram))
        completeMaterialManifoldOwned = true;
      const AvbdCompiledVelocityObjective *materialNormalOwner =
          findAvbdContactSourceObjective(
              contacts[c].objectiveProgram,
              eCONTACT_SOURCE_MATERIAL_NORMAL);
      if (materialNormalOwner &&
          materialNormalOwner->owner ==
              AvbdVelocityObjectiveOwner::PointFinalize)
        pointMaterialNormalOwned = true;
    }
    // A complete cone manifold owns normal and tangent material response as
    // one transaction.  Running the legacy dominant-normal sweep first would
    // consume the same normal source twice and make ground impact depend on
    // post-AL call order.
    if (passiveMaterialComponentOwned || completeMaterialManifoldOwned ||
        pointMaterialNormalOwned)
      continue;

    physx::PxU32 dominant = 0xFFFFFFFFu;
    physx::PxU32 initialDominant = 0xFFFFFFFFu;
    physx::PxReal worstViolation = 1e9f;
    physx::PxReal worstInitialViolation = 1e9f;
    physx::PxVec3 domWorldA(0.0f), domWorldB(0.0f);
    physx::PxVec3 initialDomWorldA(0.0f), initialDomWorldB(0.0f);

    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c = hasMapRange ? mapIndices[loopIndex] : loopIndex;
      if (hasRigidMaterialConsumed(contacts[c]))
        continue;
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      if (!isBodyVsStaticContact(bA, bB, numBodies))
        continue;
      if (bA != i && bB != i)
        continue;

      const bool dynIsA = (bA == i);
      physx::PxVec3 worldA, worldB;
      if (dynIsA) {
        worldA = bodies[i].position +
                 bodies[i].rotation.rotate(contacts[c].contactPointA);
        worldB = contacts[c].contactPointB;
      } else {
        worldA = contacts[c].contactPointA;
        worldB = bodies[i].position +
                 bodies[i].rotation.rotate(contacts[c].contactPointB);
      }
      physx::PxReal violation =
          (worldA - worldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      const physx::PxVec3 initialWorldA =
          dynIsA
              ? bodies[i].prevPosition +
                    bodies[i].prevRotation.rotate(contacts[c].contactPointA)
              : contacts[c].staticPrevWorldPoint;
      const physx::PxVec3 initialWorldB =
          dynIsA
              ? contacts[c].staticPrevWorldPoint
              : bodies[i].prevPosition +
                    bodies[i].prevRotation.rotate(contacts[c].contactPointB);
      const physx::PxReal initialViolation =
          (initialWorldA - initialWorldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      if (hasDeformableStaticAnchor(contacts[c]))
        violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
      if (violation < worstViolation) {
        worstViolation = violation;
        dominant = c;
        domWorldA = worldA;
        domWorldB = worldB;
      }
      if (initialViolation < worstInitialViolation) {
        worstInitialViolation = initialViolation;
        initialDominant = c;
        initialDomWorldA = worldA;
        initialDomWorldB = worldB;
      }
    }

    if (dominant == 0xFFFFFFFFu)
      continue;

    const bool splitDeepInitialDepenetration =
        initialDominant != 0xFFFFFFFFu &&
        !hasDeformableStaticAnchor(contacts[initialDominant]) &&
        worstInitialViolation <
            -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE *
                physx::PxMax(lengthScale, physx::PxReal(1e-6f));
    if (splitDeepInitialDepenetration) {
      dominant = initialDominant;
      domWorldA = initialDomWorldA;
      domWorldB = initialDomWorldB;
    }

    if (finiteMaterialPoseSplit &&
        finiteMaterialPoseSplit->size() == numBodies &&
        (*finiteMaterialPoseSplit)[i]) {
      physx::PxReal spatialLinearDelta = 0.0f;
      const bool finiteOwned = applyBodyStaticFiniteSpatialBudget(
          bodies, numBodies, contacts, numContacts, contactMap, i,
          linearVelAtSolveStart, angularVelAtSolveStart, dt, bounceThreshold,
          spatialLinearDelta);
      if (finiteOwned) {
        continue;
      }
    }

    const bool isDeform = hasDeformableStaticAnchor(contacts[dominant]);
    const AvbdContactConstraint &cc = contacts[dominant];
    const bool dynIsA = (cc.header.bodyIndexA == i);
    const physx::PxVec3 nd = cc.contactNormal * (dynIsA ? 1.0f : -1.0f);

    physx::PxReal staticNormalVelocity = 0.0f;
    if (!isDeform && invDt > 0.0f) {
      const physx::PxVec3 staticNow = dynIsA ? domWorldB : domWorldA;
      staticNormalVelocity =
          ((staticNow - cc.staticPrevWorldPoint) * invDt).dot(nd);
    }

    const bool hasSolveStartVelocity =
        linearVelAtSolveStart &&
        linearVelAtSolveStart->size() == numBodies;
    const physx::PxReal vn = bodies[i].linearVelocity.dot(nd);
    const physx::PxReal relativeVn = vn - staticNormalVelocity;
    physx::PxReal solveStartRelativeVn = relativeVn;
    physx::PxReal approach = 0.0f;
    if (hasSolveStartVelocity) {
      solveStartRelativeVn =
          (*linearVelAtSolveStart)[i].dot(nd) - staticNormalVelocity;
      approach = -solveStartRelativeVn;
      if (approach < 0.0f)
        approach = 0.0f;
    }
    const bool hasFiniteMaxImpulse = cc.maxImpulse < PX_MAX_REAL;
    const physx::PxReal maxImpulseRelativeVn =
        hasSolveStartVelocity && hasFiniteMaxImpulse
            ? solveStartRelativeVn +
                  physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)) *
                      bodies[i].invMass *
                      (dynIsA ? cc.invMassScaleA : cc.invMassScaleB)
            : PX_MAX_REAL;
    if (isDeform) {
      if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats)
        PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeBodies++);
      const physx::PxReal nearLim =
          AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE;
      if (worstViolation >= nearLim)
        continue;
      if (approach > kBodyStaticFastImpactSpeed)
        continue;

      physx::PxReal vMeshN = 0.0f;
      if (invDt > 0.0f) {
        const physx::PxVec3 staticNow = dynIsA ? domWorldB : domWorldA;
        const physx::PxVec3 meshStep = staticNow - cc.staticPrevWorldPoint;
        const physx::PxReal stepCap = AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
        if (meshStep.magnitudeSquared() <= stepCap * stepCap)
          vMeshN = (meshStep * invDt).dot(nd);
      }
      const physx::PxVec3 dynamicWorldPoint =
          dynIsA ? domWorldA : domWorldB;
      const physx::PxVec3 dynamicContactArm =
          dynamicWorldPoint - bodies[i].position;
      const physx::PxReal contactRelativeVnBefore =
          (bodies[i].linearVelocity +
           bodies[i].angularVelocity.cross(dynamicContactArm))
                  .dot(nd) -
          vMeshN;
      const bool spatialOwner = hasDeformablePositionTangentOwner(cc);
      const physx::PxReal comRelativeVn = vn - vMeshN;
      const physx::PxReal correctionRelativeVn =
          spatialOwner ? contactRelativeVnBefore : comRelativeVn;
      if (correctionRelativeVn > 0.0f) {
        physx::PxReal linearDeltaMagnitude = 0.0f;
        bool corrected = false;
        if (spatialOwner) {
          const physx::PxReal linearScale =
              dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
          const physx::PxReal angularScale =
              dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
          const physx::PxReal linearResponse =
              bodies[i].invMass * linearScale;
          const physx::PxVec3 angularJacobian =
              dynamicContactArm.cross(nd);
          const physx::PxVec3 angularResponse =
              bodies[i].invInertiaWorld.transform(angularJacobian);
          const physx::PxReal totalResponse =
              linearResponse +
              angularJacobian.dot(angularResponse) * angularScale;
          if (totalResponse > 1.0e-8f) {
            const physx::PxReal impulse =
                contactRelativeVnBefore / totalResponse;
            linearDeltaMagnitude = impulse * linearResponse;
            bodies[i].linearVelocity -=
                nd * linearDeltaMagnitude;
            bodies[i].angularVelocity -=
                angularResponse * (impulse * angularScale);
            corrected = true;
          }
        } else {
          bodies[i].linearVelocity -= nd * comRelativeVn;
          linearDeltaMagnitude = comRelativeVn;
          corrected = true;
        }
        if (corrected) {
          if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats &&
              deformableNormalStageMask &&
              dominant < deformableNormalStageMask->size()) {
            const physx::PxReal contactRelativeVnAfter =
                (bodies[i].linearVelocity +
                 bodies[i].angularVelocity.cross(dynamicContactArm))
                        .dot(nd) -
                vMeshN;
            const physx::PxReal diagnosticVelocityTolerance =
                1.0e-5f *
                physx::PxMax(lengthScale, physx::PxReal(1.0e-6f)) *
                invDt;
            if (contactRelativeVnAfter < -diagnosticVelocityTolerance) {
              PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeContactReversalCorrections++);
            }
          }
          if (deformableNormalStageMask &&
              dominant < deformableNormalStageMask->size())
            (*deformableNormalStageMask)[dominant] |= 4u;
          if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats) {
            PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeCorrections++);
            PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFinalizeDelta += linearDeltaMagnitude);
          }
        }
      }
      continue;
    }

    // Rigid body-static: material e from NP-combined patch restitution.
    // Compliant contacts (e < 0) treated as inelastic for now.
    const physx::PxReal e =
        (cc.restitution > 0.0f) ? physx::PxMin(cc.restitution, 1.0f) : 0.0f;
    physx::PxReal approachEff = approach;
    if (e > 0.0f && relativeVn < 0.0f)
      approachEff = physx::PxMax(approachEff, -relativeVn);
    bool restitutionOwned = false;
    physx::PxReal restitutionLinearDelta = 0.0f;
    if (e > 0.0f && !hasFiniteMaxImpulse) {
      restitutionOwned = applyBodyStaticRestitutionSpatialRow(
          bodies, numBodies, contacts, numContacts, contactMap, i,
          linearVelAtSolveStart, angularVelAtSolveStart, dt, bounceThreshold,
          restitutionLinearDelta);
    } else if (e > 0.0f && approachEff > bounceThreshold) {
      const physx::PxReal desiredRelativeVn =
          physx::PxMin(e * approachEff, maxImpulseRelativeVn);
      const physx::PxReal deltaV =
          staticNormalVelocity + desiredRelativeVn - vn;
      bodies[i].linearVelocity += nd * deltaV;
      restitutionOwned = true;
    }
    if (!restitutionOwned) {
      // Inelastic / resting: the position solve may clear the narrow-phase
      // overlap in this step, but that geometric correction is not impact
      // velocity. Preserve any separating velocity the body already had at
      // solve start (so an authored take-off is not cancelled), and remove
      // only the separating speed created by the contact correction.
      const physx::PxReal allowedRelativeVn =
          hasSolveStartVelocity
              ? physx::PxMin(
                    physx::PxMax(solveStartRelativeVn, physx::PxReal(0.0f)),
                    maxImpulseRelativeVn)
              : physx::PxReal(0.0f);
      const bool shouldClamp =
          hasSolveStartVelocity || worstViolation < -1e-5f ||
          splitDeepInitialDepenetration;
      if (shouldClamp && relativeVn > allowedRelativeVn) {
        const physx::PxReal deltaV = relativeVn - allowedRelativeVn;
        bodies[i].linearVelocity -= nd * deltaV;
      }
    }
  }

  // Legacy/finite dyn-dyn restitution. Ordinary unlimited manifolds were
  // consumed above as one spatial block; this point path remains only until
  // finite total-impulse reconstruction has an explicit residual budget.
  // Apply only for free rigid pairs (no deformable); e and bounce threshold
  // from material/scene. Skip if either body already handled as body-static
  // dominant this frame would double-count; dyn-dyn contacts are exclusive.
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    if (hasRigidMaterialConsumed(cc))
      continue;
    const AvbdCompiledVelocityObjective *materialNormalObjective =
        findAvbdContactSourceObjective(
            cc.objectiveProgram,
            eCONTACT_SOURCE_MATERIAL_NORMAL);
    // A compiled material-normal source is consumed only by its unique
    // owner. Legacy rows retain the historical path until their compile
    // classification is made explicit.
    if (materialNormalObjective &&
        materialNormalObjective->owner !=
            AvbdVelocityObjectiveOwner::PointFinalize)
      continue;
    if (hasDeformableStaticAnchor(cc) ||
        hasVelocityPassiveFrictionComponentOwner(cc))
      continue;
    const physx::PxU32 bA = cc.header.bodyIndexA;
    const physx::PxU32 bB = cc.header.bodyIndexB;
    if (bA >= numBodies || bB >= numBodies)
      continue;
    if (bodies[bA].invMass <= 0.0f || bodies[bB].invMass <= 0.0f)
      continue;
    const physx::PxReal e =
        (cc.restitution > 0.0f) ? physx::PxMin(cc.restitution, 1.0f) : 0.0f;
    if (e <= 1e-6f)
      continue;
    if (!linearVelAtSolveStart || linearVelAtSolveStart->size() != numBodies)
      continue;

    const physx::PxVec3 &n = cc.contactNormal;
    const physx::PxReal vrel0 =
        ((*linearVelAtSolveStart)[bA] - (*linearVelAtSolveStart)[bB]).dot(n);
    const physx::PxReal approach = (vrel0 < 0.0f) ? -vrel0 : 0.0f;
    if (approach <= bounceThreshold)
      continue;

    const physx::PxReal vrel =
        (bodies[bA].linearVelocity - bodies[bB].linearVelocity).dot(n);
    const physx::PxReal invMassA =
        bodies[bA].invMass * cc.invMassScaleA;
    const physx::PxReal invMassB =
        bodies[bB].invMass * cc.invMassScaleB;
    const physx::PxReal invSum = invMassA + invMassB;
    if (invSum < 1e-12f)
      continue;
    const physx::PxReal maxImpulseVrel =
        cc.maxImpulse < PX_MAX_REAL
            ? vrel0 + physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)) * invSum
            : PX_MAX_REAL;
    const physx::PxReal desiredVrel =
        physx::PxMin(e * approach, maxImpulseVrel);
    if (vrel >= desiredVrel)
      continue;
    physx::PxReal j = (desiredVrel - vrel) / invSum;
    if (cc.maxImpulse < PX_MAX_REAL)
      j = physx::PxMin(
          j, physx::PxMax(cc.maxImpulse, physx::PxReal(0.0f)));
    bodies[bA].linearVelocity += n * (j * invMassA);
    bodies[bB].linearVelocity -= n * (j * invMassB);
  }
}

// Backward-compatible name used by postAlStages call site.
static void clampBodyStaticInelasticNormalVelocitiesImpl(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale,
    bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  applyAvbdMaterialNormalVelocity(bodies, numBodies, contacts, numContacts,
                                  contactMap,
                                  linearVelAtSolveStart,
                                  angularVelAtSolveStart,
                                  linearPoseVelocityGain,
                                  angularPoseVelocityGain,
                                  finiteMaterialPoseSplit, dt,
                                  bounceApproachThreshold, lengthScale,
                                  hasJointConstraints,
                                  enableBoundedComponentProductionProbe,
                                  deformableNormalStageMask, stats);
}

static void recordBodyStaticNormalAlOwnershipImpl(
    const AvbdSolverBody *bodies, const AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numBodies,
    physx::PxReal /*avbdAlpha*/,
    const physx::PxArray<bool> * /*touchesKinematicShell*/,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats &stats) {
  (void)stats;
  if (!bodies || !contacts || !deformableNormalStageMask)
    return;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &contact = contacts[c];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        !hasDeformableStaticAnchor(contact))
      continue;
    PX_AVBD_PROFILE_STAT(stats.surfaceDeformableAlRows++);
    if (c < deformableNormalStageMask->size())
      (*deformableNormalStageMask)[c] |= 1u;
    if (hasDeformablePositionTangentOwner(contact))
      PX_AVBD_PROFILE_STAT(stats.surfaceDeformablePositionTangentRows += 2);
  }
}

static void computeMaxPoseDeltas(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const physx::PxArray<physx::PxVec3> &prevPos,
    const physx::PxArray<physx::PxQuat> &prevRot,
    physx::PxReal &maxPositionDelta, physx::PxReal &maxRotationDelta) {
  maxPositionDelta = 0.0f;
  maxRotationDelta = 0.0f;
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass <= 0.0f)
      continue;

    maxPositionDelta = physx::PxMax(
        maxPositionDelta, (bodies[i].position - prevPos[i]).magnitude());
    maxRotationDelta = physx::PxMax(
        maxRotationDelta,
        computeRotationDeltaMagnitude(bodies[i].rotation, prevRot[i]));
  }
}
} // namespace

bool bodyTouchesDeformableAnchor(
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxU32 bodyIndex, const AvbdBodyConstraintMap *contactMap) {
  return bodyTouchesDeformableAnchorImpl(contacts, numContacts, bodyIndex,
                                         contactMap);
}

void projectBodyStaticLockedD6LinearVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdD6JointConstraint *joints, physx::PxU32 numJoints) {
  projectBodyStaticLockedD6LinearVelocitiesImpl(bodies, numBodies, joints,
                                                numJoints);
}

void applyAvbdContactTargetVelocity(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    physx::PxReal dt, physx::PxReal bounceThreshold,
    const AvbdPostAlContactWorkPlan *workPlan) {
  applyAvbdContactTargetVelocityImpl(bodies, numBodies, contacts, numContacts,
                                     contactMap, linearVelAtSolveStart,
                                     angularVelAtSolveStart,
                                     linearPoseVelocityGain,
                                     angularPoseVelocityGain, dt,
                                     bounceThreshold, workPlan);
}

bool isRigidDeepBodyStaticRecoverySplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  return isRigidDeepBodyStaticRecoverySplitSupportedImpl(
      bodies, numBodies, contacts, numContacts, contactMap, bodyIndex,
      lengthScale);
}

bool isRigidFiniteBodyStaticMaterialSplitSupported(
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap, physx::PxU32 bodyIndex,
    physx::PxReal lengthScale) {
  return isRigidFiniteBodyStaticMaterialSplitSupportedImpl(
      bodies, numBodies, contacts, numContacts, contactMap, bodyIndex,
      lengthScale);
}

void clampBodyStaticInelasticNormalVelocities(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const AvbdBodyConstraintMap *contactMap,
    const physx::PxArray<physx::PxVec3> *linearVelAtSolveStart,
    const physx::PxArray<physx::PxVec3> *angularVelAtSolveStart,
    const physx::PxArray<physx::PxReal> *linearPoseVelocityGain,
    const physx::PxArray<physx::PxReal> *angularPoseVelocityGain,
    const physx::PxArray<bool> *finiteMaterialPoseSplit,
    physx::PxReal dt, physx::PxReal bounceApproachThreshold,
    physx::PxReal lengthScale, bool hasJointConstraints,
    bool enableBoundedComponentProductionProbe,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats *stats) {
  clampBodyStaticInelasticNormalVelocitiesImpl(
      bodies, numBodies, contacts, numContacts, contactMap,
      linearVelAtSolveStart, angularVelAtSolveStart,
      linearPoseVelocityGain, angularPoseVelocityGain,
      finiteMaterialPoseSplit, dt, bounceApproachThreshold, lengthScale,
      hasJointConstraints, enableBoundedComponentProductionProbe,
      deformableNormalStageMask, stats);
}

void recordBodyStaticNormalAlOwnership(
    const AvbdSolverBody *bodies, const AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, physx::PxU32 numBodies,
    physx::PxReal avbdAlpha,
    const physx::PxArray<bool> *touchesKinematicShell,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    AvbdSolverStats &stats) {
  recordBodyStaticNormalAlOwnershipImpl(
      bodies, contacts, numContacts, numBodies, avbdAlpha,
      touchesKinematicShell, deformableNormalStageMask, stats);
}

bool AvbdSolver::beginRigidSolveIteration(
    AvbdRigidSolveIterationState &state) {
  if (!state.bodies || !state.stats || state.iter >= state.iters ||
      state.iterationActive)
    return false;
  PX_ASSERT(!state.parallelDualComplete);

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  state.activeIteration = state.iter++;
  state.iterationActive = true;
  state.primalFinalized = false;

  // Save pre-iteration state for Chebyshev relaxation and convergence tests.
  if (state.useChebyshev) {
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      state.chebyPrevPrevPos[i] = state.chebyPrevPos[i];
      state.chebyPrevPrevRot[i] = state.chebyPrevRot[i];
      state.chebyPrevPos[i] = bodies[i].position;
      state.chebyPrevRot[i] = bodies[i].rotation;
    }
  }
  if (state.enableEarlyStop) {
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      state.earlyStopPrevPos[i] = bodies[i].position;
      state.earlyStopPrevRot[i] = bodies[i].rotation;
    }
  }
  return true;
}

bool AvbdSolver::finalizeRigidPrimalIteration(
    AvbdRigidSolveIterationState &state) {
  if (!state.bodies || !state.stats || !state.iterationActive)
    return false;
  if (state.primalFinalized)
    return true;

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  const AvbdBodyConstraintMap *contactMap = state.contactMap;
  const physx::PxU32 iter = state.activeIteration;

  PX_PROFILE_ZONE("AVBD.finalizePrimal", 0);
  // Chebyshev semi-iterative position/rotation relaxation.
  if (state.useChebyshev && iter >= 2) {
    const physx::PxReal rhoSq = mConfig.chebyshevRho * mConfig.chebyshevRho;
    if (iter == 2)
      state.chebyOmega = 2.0f / (2.0f - rhoSq);
    else
      state.chebyOmega =
          1.0f / (1.0f - rhoSq * state.chebyOmega / 4.0f);
    state.chebyOmega = physx::PxClamp(state.chebyOmega, 1.0f, 2.0f);

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass <= 0.0f)
        continue;
      const physx::PxVec3 gsPosition = bodies[i].position;
      const physx::PxQuat gsRotation = bodies[i].rotation;
      const physx::PxVec3 relaxedPosition =
          state.chebyPrevPrevPos[i] +
          (bodies[i].position - state.chebyPrevPrevPos[i]) *
              state.chebyOmega;

      physx::PxQuat qPrev = state.chebyPrevPrevRot[i];
      physx::PxQuat qCur = bodies[i].rotation;
      if (qPrev.dot(qCur) < 0.0f)
        qCur = -qCur;
      physx::PxQuat qBlend(
          qPrev.x + state.chebyOmega * (qCur.x - qPrev.x),
          qPrev.y + state.chebyOmega * (qCur.y - qPrev.y),
          qPrev.z + state.chebyOmega * (qCur.z - qPrev.z),
          qPrev.w + state.chebyOmega * (qCur.w - qPrev.w));
      const physx::PxQuat relaxedRotation = qBlend.getNormalized();

      // A unilateral body-static active set has zero energy on its satisfied
      // side.  Reject only an outward extrapolation after a deep, quasi-static
      // overlap has already been cleared by the ordinary block step.
      bool rejectBodyStaticOvershoot = false;
      if (state.hasBodyStaticContact) {
        physx::PxReal minGsViolation = PX_MAX_REAL;
        physx::PxReal minRelaxedViolation = PX_MAX_REAL;
        bool foundBodyStatic = false;
        bool deepQuasistaticInitialOverlap = false;
        const physx::PxU32 *mapIndices = nullptr;
        physx::PxU32 mapCount = 0;
        if (contactMap && contactMap->numBodies > 0)
          contactMap->getBodyConstraints(i, mapIndices, mapCount);
        const physx::PxU32 loopCount = mapIndices ? mapCount : numContacts;
        for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
          const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
          const physx::PxU32 bA = contacts[c].header.bodyIndexA;
          const physx::PxU32 bB = contacts[c].header.bodyIndexB;
          if (!isBodyVsStaticContact(bA, bB, numBodies) ||
              (bA != i && bB != i))
            continue;

          const bool dynIsA = (bA == i);
          const physx::PxVec3 gsWorldA =
              dynIsA ? gsPosition + gsRotation.rotate(contacts[c].contactPointA)
                     : contacts[c].contactPointA;
          const physx::PxVec3 gsWorldB =
              dynIsA ? contacts[c].contactPointB
                     : gsPosition + gsRotation.rotate(contacts[c].contactPointB);
          const physx::PxVec3 relaxedWorldA =
              dynIsA ? relaxedPosition +
                           relaxedRotation.rotate(contacts[c].contactPointA)
                     : contacts[c].contactPointA;
          const physx::PxVec3 relaxedWorldB =
              dynIsA ? contacts[c].contactPointB
                     : relaxedPosition +
                           relaxedRotation.rotate(contacts[c].contactPointB);
          const physx::PxReal gsViolation =
              (gsWorldA - gsWorldB).dot(contacts[c].contactNormal) +
              contacts[c].penetrationDepth;
          const physx::PxReal relaxedViolation =
              (relaxedWorldA - relaxedWorldB).dot(contacts[c].contactNormal) +
              contacts[c].penetrationDepth;
          minGsViolation = physx::PxMin(minGsViolation, gsViolation);
          minRelaxedViolation =
              physx::PxMin(minRelaxedViolation, relaxedViolation);
          foundBodyStatic = true;

          const physx::PxVec3 initialWorldA =
              dynIsA ? bodies[i].prevPosition +
                           bodies[i].prevRotation.rotate(
                               contacts[c].contactPointA)
                     : contacts[c].staticPrevWorldPoint;
          const physx::PxVec3 initialWorldB =
              dynIsA ? contacts[c].staticPrevWorldPoint
                     : bodies[i].prevPosition +
                           bodies[i].prevRotation.rotate(
                               contacts[c].contactPointB);
          const physx::PxReal initialViolation =
              (initialWorldA - initialWorldB).dot(contacts[c].contactNormal) +
              contacts[c].penetrationDepth;
          const physx::PxVec3 outwardNormal =
              contacts[c].contactNormal * (dynIsA ? 1.0f : -1.0f);
          const physx::PxReal approach =
              state.linearVelAtSolveStart &&
                      state.linearVelAtSolveStart->size() == numBodies
                  ? physx::PxMax(
                        0.0f,
                        -(*state.linearVelAtSolveStart)[i].dot(outwardNormal))
                  : 0.0f;
          const physx::PxReal deepOverlapThreshold =
              0.05f * physx::PxMax(mConfig.lengthScale, 1e-6f);
          if (initialViolation < -deepOverlapThreshold &&
              approach <= mConfig.bounceApproachSpeedThreshold())
            deepQuasistaticInitialOverlap = true;
        }
        const physx::PxReal activeSetTolerance =
            0.01f * physx::PxMax(mConfig.lengthScale, 1e-6f);
        rejectBodyStaticOvershoot =
            foundBodyStatic && deepQuasistaticInitialOverlap &&
            minGsViolation >= activeSetTolerance &&
            minRelaxedViolation > minGsViolation + activeSetTolerance;
      }

      bodies[i].position = rejectBodyStaticOvershoot ? gsPosition
                                                      : relaxedPosition;
      bodies[i].rotation = rejectBodyStaticOvershoot ? gsRotation
                                                      : relaxedRotation;
    }
  }

  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (bodies[i].invMass > 0.0f)
      bodies[i].projectLockedPose(bodies[i].prevPosition,
                                  bodies[i].prevRotation);
  }

  state.primalFinalized = true;
  return true;
}

bool AvbdSolver::completeRigidSolveIteration(
    AvbdRigidSolveIterationState &state) {
  if (!state.bodies || !state.stats || !state.iterationActive ||
      !finalizeRigidPrimalIteration(state))
    return false;

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  AvbdSolverStats &stats = *state.stats;
  const physx::PxU32 iter = state.activeIteration;

  // The dual must observe the accepted primal pose.  In particular,
  // Chebyshev relaxation changes q; updating lambda before that relaxation
  // leaves the persisted force one phase behind the pose and BDF velocity.
  PX_AVBD_PROFILE_STAT(stats.totalIterations++);
  if (!state.parallelDualComplete) {
    PX_PROFILE_ZONE("AVBD.updateLambda", 0);
    updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                state.dt, stats);
  }
  state.parallelDualComplete = false;

  if (state.enableEarlyStop) {
    physx::PxReal maxPositionDelta = 0.0f;
    physx::PxReal maxRotationDelta = 0.0f;
    computeMaxPoseDeltas(bodies, numBodies, state.earlyStopPrevPos,
                         state.earlyStopPrevRot, maxPositionDelta,
                         maxRotationDelta);
    if ((iter + 1) >= state.minIterations &&
        maxPositionDelta <= mConfig.positionTolerance &&
        maxRotationDelta <= state.rotationTolerance) {
      ++state.consecutiveConvergedIterations;
      if (state.consecutiveConvergedIterations >= 2)
        state.iter = state.iters;
    } else {
      state.consecutiveConvergedIterations = 0;
    }
  }
  state.iterationActive = false;
  state.primalFinalized = false;
  return state.iter < state.iters;
}

bool AvbdSolver::advanceRigidSolveIterations(
    AvbdRigidSolveIterationState &state) {
  if (!beginRigidSolveIteration(state))
    return false;

  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  const physx::PxReal dt = state.dt;
  const AvbdBodyConstraintMap *contactMap = state.contactMap;

  {
    PX_PROFILE_ZONE("AVBD.blockDescent", 0);
    blockDescentIteration(bodies, numBodies, contacts, numContacts, dt,
                          contactMap, state.colorBatches, state.numColors);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      if (bodies[i].invMass > 0.0f)
        bodies[i].projectLockedPose(bodies[i].prevPosition,
                                    bodies[i].prevRotation);
    }
  }
  completeRigidSolveIteration(state);
  return true;
}

void AvbdSolver::buildRigidDependencyWaves(
    AvbdRigidSolveContext &context) {
  AvbdRigidSolveIterationState &state = context.iteration;
  const physx::PxU32 numBodies = state.numBodies;
  context.dependencyWaveOffsets.clear();
  context.dependencyWaveBodies.clear();
  context.dependencyWaveCount = 0;
  if (numBodies == 0)
    return;

  physx::PxArray<physx::PxU32> bodyOrder(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    bodyOrder[i] = i;

  const bool useDeterministicOrder =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);
  if (useDeterministicOrder) {
    std::sort(bodyOrder.begin(), bodyOrder.end(),
              [&state](physx::PxU32 a, physx::PxU32 b) {
                if (state.bodies[a].invMass != state.bodies[b].invMass)
                  return state.bodies[a].invMass > state.bodies[b].invMass;
                return a < b;
              });
  }

  physx::PxArray<physx::PxU32> orderPosition(numBodies);
  physx::PxArray<physx::PxU32> bodyWave(numBodies);
  for (physx::PxU32 position = 0; position < numBodies; ++position) {
    orderPosition[bodyOrder[position]] = position;
    bodyWave[bodyOrder[position]] = 0;
  }

  // The serial body sweep is a Gauss--Seidel order.  A body depends only on
  // incident dynamic bodies that have already appeared in that order; those
  // edges are acyclic by construction and can therefore be levelized in one
  // forward pass.
  for (physx::PxU32 position = 0; position < numBodies; ++position) {
    const physx::PxU32 body = bodyOrder[position];
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (state.contactMap)
      state.contactMap->getBodyConstraints(body, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : state.numContacts;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c = mapIndices ? mapIndices[loopIndex] : loopIndex;
      const AvbdContactConstraint &contact = state.contacts[c];
      const physx::PxU32 other =
          contact.header.bodyIndexA == body ? contact.header.bodyIndexB
                                             : contact.header.bodyIndexA;
      if (other >= numBodies || other == body ||
          orderPosition[other] >= position)
        continue;
      bodyWave[body] = physx::PxMax(bodyWave[body], bodyWave[other] + 1u);
    }
  }

  physx::PxU32 maxWave = 0;
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    maxWave = physx::PxMax(maxWave, bodyWave[i]);
  context.dependencyWaveCount = maxWave + 1u;
  context.dependencyWaveOffsets.resize(context.dependencyWaveCount + 1u);
  for (physx::PxU32 wave = 0; wave <= context.dependencyWaveCount; ++wave)
    context.dependencyWaveOffsets[wave] = 0;
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    ++context.dependencyWaveOffsets[bodyWave[i] + 1u];
  for (physx::PxU32 wave = 1; wave <= context.dependencyWaveCount; ++wave)
    context.dependencyWaveOffsets[wave] +=
        context.dependencyWaveOffsets[wave - 1u];

  context.dependencyWaveBodies.resize(numBodies);
  physx::PxArray<physx::PxU32> waveWriteOffsets(
      context.dependencyWaveCount);
  for (physx::PxU32 wave = 0; wave < context.dependencyWaveCount; ++wave)
    waveWriteOffsets[wave] = context.dependencyWaveOffsets[wave];
  for (physx::PxU32 position = 0; position < numBodies; ++position) {
    const physx::PxU32 body = bodyOrder[position];
    const physx::PxU32 wave = bodyWave[body];
    context.dependencyWaveBodies[waveWriteOffsets[wave]++] = body;
  }

}

bool AvbdSolver::buildRigidBodyColorPlan(
    AvbdRigidSolveContext &context) {
  PX_PROFILE_ZONE("AVBD.buildRigidBodyColorPlan", 0);
  AvbdRigidSolveIterationState &state = context.iteration;
  AvbdSolverBody *bodies = state.bodies;
  const physx::PxU32 numBodies = state.numBodies;
  AvbdContactConstraint *contacts = state.contacts;
  const physx::PxU32 numContacts = state.numContacts;
  const AvbdBodyConstraintMap *contactMap = state.contactMap;

  context.bodyColorOffsets.clear();
  context.bodyColorBodies.clear();
  context.bodyColorCount = 0;
  context.maxBodyColorWidth = 0;

  // The fast schedule is deliberately fail-closed.  A partial map or a body
  // index that does not name its island-local slot would make two tasks read
  // and write an unproven ownership graph.
  if (!bodies || !contacts || numBodies == 0 || numContacts == 0 ||
      !contactMap || contactMap->numBodies != numBodies ||
      !contactMap->constraintOffsets || !contactMap->constraintCounts ||
      (contactMap->totalConstraintRefs > 0 &&
       !contactMap->constraintIndices) ||
      contactMap->constraintOffsets[numBodies] !=
          contactMap->totalConstraintRefs)
    return false;

  if (contactMap->constraintOffsets[0] != 0)
    return false;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 begin = contactMap->constraintOffsets[body];
    const physx::PxU32 end = contactMap->constraintOffsets[body + 1u];
    if (begin > end || end > contactMap->totalConstraintRefs ||
        contactMap->constraintCounts[body] != end - begin)
      return false;
  }
  physx::PxArray<physx::PxU32> bodyColors(numBodies);
  physx::PxArray<physx::PxU32> forbiddenColorStamp(numBodies);
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    bodyColors[body] = PX_MAX_U32;
    forbiddenColorStamp[body] = 0;
    if (bodies[body].nodeIndex != body)
      return false;
  }

  physx::PxU32 dynamicBodyCount = 0;
  physx::PxU32 colorCount = 0;
  physx::PxU32 stamp = 0;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    if (bodies[body].invMass <= 0.0f)
      continue;

    ++dynamicBodyCount;
    ++stamp;
    if (stamp == 0) {
      for (physx::PxU32 color = 0; color < numBodies; ++color)
        forbiddenColorStamp[color] = 0;
      stamp = 1;
    }

    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    contactMap->getBodyConstraints(body, mapIndices, mapCount);
    if (mapCount > 0 && !mapIndices)
      return false;
    for (physx::PxU32 ref = 0; ref < mapCount; ++ref) {
      const physx::PxU32 contactIndex = mapIndices[ref];
      if (contactIndex >= numContacts)
        return false;
      const AvbdContactConstraint &contact = contacts[contactIndex];
      const physx::PxU32 bodyA = contact.header.bodyIndexA;
      const physx::PxU32 bodyB = contact.header.bodyIndexB;
      if (bodyA != body && bodyB != body)
        return false;
      const physx::PxU32 other = bodyA == body ? bodyB : bodyA;
      if (other >= numBodies || other == body ||
          bodies[other].invMass <= 0.0f)
        continue;
      const physx::PxU32 otherColor = bodyColors[other];
      if (otherColor < colorCount)
        forbiddenColorStamp[otherColor] = stamp;
    }

    physx::PxU32 color = 0;
    while (color < colorCount && forbiddenColorStamp[color] == stamp)
      ++color;
    if (color == colorCount)
      ++colorCount;
    if (color >= numBodies)
      return false;
    bodyColors[body] = color;
  }

  if (dynamicBodyCount == 0 || colorCount == 0)
    return false;
  // Validate the strict independent-set contract against the source rows,
  // independently of the CSR traversal used to build the plan.
  for (physx::PxU32 contactIndex = 0; contactIndex < numContacts;
       ++contactIndex) {
    const physx::PxU32 bodyA = contacts[contactIndex].header.bodyIndexA;
    const physx::PxU32 bodyB = contacts[contactIndex].header.bodyIndexB;
    if (bodyA >= numBodies || bodyB >= numBodies || bodyA == bodyB ||
        bodies[bodyA].invMass <= 0.0f || bodies[bodyB].invMass <= 0.0f)
      continue;
    if (bodyColors[bodyA] == PX_MAX_U32 ||
        bodyColors[bodyB] == PX_MAX_U32 ||
        bodyColors[bodyA] == bodyColors[bodyB])
      return false;
  }
  context.bodyColorOffsets.resize(colorCount + 1u);
  for (physx::PxU32 color = 0; color <= colorCount; ++color)
    context.bodyColorOffsets[color] = 0;
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    if (bodyColors[body] < colorCount)
      ++context.bodyColorOffsets[bodyColors[body] + 1u];
  }
  for (physx::PxU32 color = 1; color <= colorCount; ++color)
    context.bodyColorOffsets[color] +=
        context.bodyColorOffsets[color - 1u];
  if (context.bodyColorOffsets[colorCount] != dynamicBodyCount)
    return false;

  context.bodyColorBodies.resize(dynamicBodyCount);
  physx::PxArray<physx::PxU32> writeOffsets(colorCount);
  for (physx::PxU32 color = 0; color < colorCount; ++color) {
    writeOffsets[color] = context.bodyColorOffsets[color];
    context.maxBodyColorWidth = physx::PxMax(
        context.maxBodyColorWidth,
        context.bodyColorOffsets[color + 1u] -
            context.bodyColorOffsets[color]);
  }
  for (physx::PxU32 body = 0; body < numBodies; ++body) {
    const physx::PxU32 color = bodyColors[body];
    if (color < colorCount)
      context.bodyColorBodies[writeOffsets[color]++] = body;
  }

  context.bodyColorCount = colorCount;
  return true;
}
//=============================================================================
// Main Solver Entry Point
//=============================================================================

bool AvbdSolver::prepareRigidSolve(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, const AvbdBodyConstraintMap *contactMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors,
    physx::PxU32 iterationOverride, AvbdSolverStats &stats,
    AvbdRigidSolveContext &context) {
  PX_PROFILE_ZONE("AVBD.prepareRigidSolve", 0);
  context.postAlContactWork.reset();
  if (!mInitialized || numBodies == 0)
    return false;

  context.invDt = 1.0f / dt;
  context.invDt2 = context.invDt * context.invDt;
  context.gravity = gravity;
  context.hasBodyStaticContact = false;
  context.deformableFastImpactIsland = false;
  context.touchingBodyStatic.clear();
  context.linearVelAtSolveStart.clear();
  context.angularVelAtSolveStart.clear();
  AvbdRigidSolveIterationState &iterationState = context.iteration;
  iterationState.bodies = bodies;
  iterationState.numBodies = numBodies;
  iterationState.contacts = contacts;
  iterationState.numContacts = numContacts;
  iterationState.dt = dt;
  iterationState.contactMap = contactMap;
  iterationState.colorBatches = colorBatches;
  iterationState.numColors = numColors;
  iterationState.stats = &stats;
  iterationState.iter = 0;
  iterationState.activeIteration = 0;
  iterationState.iterationActive = false;
  PX_AVBD_PROFILE_STAT(stats.numBodies = numBodies);
  PX_AVBD_PROFILE_STAT(stats.numContacts = numContacts);

  // Stage 1: Prediction
  {
    PX_PROFILE_ZONE("AVBD.prediction", 0);
    computePrediction(bodies, numBodies, dt, gravity);
  }

  // The contact BCD path below uses body-level Jacobi snapshots and does not
  // consume the legacy solver-owned graph coloring.  Building that shared
  // coloring here is both redundant and unsafe when independent island tasks
  // enter the same solver concurrently.

  // Adaptive position warmstarting (ref: AVBD3D solver.cpp L76-98)
  //
  // The solver's inertia term RHS = M/h^2*(x - x_pred) drives the body
  // toward its prediction. The warmstart position controls the gravity drive:
  //
  //   x_warmstart = x_n + v*dt + accelWeight * g*dt^2
  //   x_pred      = x_n + v*dt + g*dt^2
  //   RHS = M/h^2 * (accelWeight - 1) * g*dt^2
  //
  //   accelWeight=0 (supported): RHS = -M*g  (full gravity drive)
  //   accelWeight=1 (freefall):  RHS = 0     (no gravity drive)
  //
  // accelWeight = clamp(dot(acceleration, gravDir) / |g|, 0, 1)
  //   acceleration = (v_current - v_previous) / dt
  //
  // Now that computePrediction does NOT modify linearVelocity:
  //   linearVelocity     = v_{N-1, postsolve}  (clean post-solve from last
  //   frame) prevLinearVelocity  = v_{N-2, postsolve}  (saved at end of frame
  //   N-2)
  context.hasBodyStaticContact = false;
  bool hasDeformableAnchorContact = false;
  bool allBodyVsStatic = (numContacts > 0);
  context.touchingBodyStatic.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    context.touchingBodyStatic[i] = false;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const physx::PxU32 bA = contacts[c].header.bodyIndexA;
    const physx::PxU32 bB = contacts[c].header.bodyIndexB;
    if (isBodyVsStaticContact(bA, bB, numBodies)) {
      context.hasBodyStaticContact = true;
      if (bA < numBodies)
        context.touchingBodyStatic[bA] = true;
      if (bB < numBodies)
        context.touchingBodyStatic[bB] = true;
    } else {
      allBodyVsStatic = false;
    }
    if (hasDeformableStaticAnchor(contacts[c]))
      hasDeformableAnchorContact = true;
  }
  // Fast sphere-on-mesh islands: single dynamic + deformable static only.
  context.deformableFastImpactIsland =
      allBodyVsStatic && hasDeformableAnchorContact;

  // Snapshot pre-solve velocity for material restitution (incl. pure dyn-dyn
  // islands) and deformable fast-impact blend.
  context.linearVelAtSolveStart.clear();
  context.angularVelAtSolveStart.clear();
  if (numContacts > 0) {
    context.linearVelAtSolveStart.resize(numBodies);
    context.angularVelAtSolveStart.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      // Material contact reads the same public pre-contact epoch that the
      // velocity pipeline exposes: external acceleration, then per-body
      // damping and speed caps.  Capturing an earlier raw velocity and later
      // replacing the pose-derived result would erase damping on every owned
      // impact component.
      physx::PxVec3 linear, angular;
      computeAvbdMaterialSolveStartVelocity(
          bodies[i], gravity, dt, linear, angular);
      context.linearVelAtSolveStart[i] = linear;
      context.angularVelAtSolveStart[i] = angular;
    }

  }

  {
    PX_PROFILE_ZONE("AVBD.initPositions", 0);

    const physx::PxReal gravMag = gravity.magnitude();
    const physx::PxVec3 gravDir =
        (gravMag > 1e-6f) ? gravity / gravMag : physx::PxVec3(0.0f);

    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      // Save current position for velocity computation at end of solve.
      // In the reference this is "initialPosition".
      bodies[i].prevPosition = bodies[i].position;
      bodies[i].prevRotation = bodies[i].rotation;

      if (bodies[i].invMass > 0.0f) {
        // Compute acceleration from velocity change across frames
        // accel = (v_{N-1} - v_{N-2}) / dt
        physx::PxVec3 accel =
            (bodies[i].linearVelocity - bodies[i].prevLinearVelocity) * context.invDt;

        physx::PxReal accelWeight = 0.0f;
        if (!context.touchingBodyStatic[i] && gravMag > 1e-6f) {
          accelWeight =
              physx::PxClamp(accel.dot(gravDir) / gravMag, 0.0f, 1.0f);
        }

        // Warmstart position: x = x_n + v*dt + accelWeight * g*dt^2
        // Body-vs-static: start from inertial prediction only. Gravity
        // warmstart overshoots into the mesh on fast impacts without CCD;
        // the supported RHS (accelWeight=0) then fights contacts and ejects.
        if (context.touchingBodyStatic[i]) {
          const bool deformableTouch =
              bodyTouchesDeformableAnchor(contacts, numContacts, i,
                                          contactMap);
          const bool fastImpact =
              bodies[i].linearVelocity.magnitude() > kBodyStaticFastImpactSpeed;
          // Slow support on heaving mesh: inertial init pulls bodies into the
          // surface (accelWeight=0 already removes gravity drive from RHS).
          // Fast deformable impact: inertial start avoids warmstart overshoot.
          if (deformableTouch && fastImpact) {
            bodies[i].position = bodies[i].inertialPosition;
          } else {
            bodies[i].position =
                bodies[i].prevPosition + bodies[i].linearVelocity * dt;
          }
          bodies[i].rotation = bodies[i].inertialRotation;
        } else {
          bodies[i].position = bodies[i].prevPosition +
                               bodies[i].linearVelocity * dt +
                               gravity * (accelWeight * dt * dt);
          bodies[i].rotation = bodies[i].inertialRotation;
        }
        bodies[i].projectLockedPose(bodies[i].prevPosition,
                                    bodies[i].prevRotation);
      }
    }
  }

  // Condition every hard contact coordinate against its complete spatial
  // response.  Endpoint type, mass ratio and lever arm therefore select the
  // same dimensionless AVBD stiffness instead of unrelated empirical floors.
  applyAvbdPenaltyFloor(contacts, numContacts, bodies, numBodies,
                        context.invDt2, mConfig.avbdPenaltyMin);

  // =========================================================================
  // Compute C0 for alpha blending (ref: AVBD3D manifold.cpp computeC0)
  //
  // C0 = initial constraint violation at PRE-WARMSTART positions (the old
  // positions from end of previous step, saved as prevPosition/prevRotation).
  //
  // CRITICAL: C0 must be computed at old positions, NOT warmstart positions!
  // If C0 captures the gravity-induced predicted penetration, then
  // alpha blending (violation - alpha*C0) cancels 95% of the constraint
  // signal, causing bodies to fall through each other.
  //
  // At old positions, established contacts have C0 ~= 0, so alpha blending
  // is nearly a no-op (violation ~= violation - 0). For newly penetrating
  // contacts, C0 < 0 and the blending gradually corrects over frames.
  // =========================================================================
  {
    PX_PROFILE_ZONE("AVBD.computeC0", 0);
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      // Use prevPosition/prevRotation = positions from START of step
      // (saved before warmstart body positions were applied)
      physx::PxVec3 wA =
          (bA < numBodies)
              ? bodies[bA].prevPosition +
                    bodies[bA].prevRotation.rotate(contacts[c].contactPointA)
              : contacts[c].contactPointA;
      physx::PxVec3 wB =
          (bB < numBodies)
              ? bodies[bB].prevPosition +
                    bodies[bB].prevRotation.rotate(contacts[c].contactPointB)
              : contacts[c].contactPointB;
      if (hasDeformableStaticAnchor(contacts[c])) {
        // Moving mesh anchor: no alpha-soften on normals.
        contacts[c].C0 = 0.0f;
        contacts[c].tangentC0 = 0.0f;
        contacts[c].tangentC1 = 0.0f;
        continue;
      }
      const physx::PxVec3 initialDelta = wA - wB;
      physx::PxReal rawC0 = initialDelta.dot(contacts[c].contactNormal) +
                            contacts[c].penetrationDepth;

      // Depth-adaptive C0 clamping: for deep penetrations (fast impacts),
      // reduce C0 so that alpha blending does not over-soften the correction.
      const physx::PxReal c0Threshold = 0.05f * mConfig.lengthScale;
      const physx::PxReal c0MaxDepth = 0.20f * mConfig.lengthScale;
      if (contacts[c].persistentPointMatched == 0 &&
          rawC0 < -c0Threshold) {
        physx::PxReal t = PxClamp(
            (c0MaxDepth + rawC0) / (c0MaxDepth - c0Threshold), 0.0f, 1.0f);
        rawC0 *= t;
      }
      contacts[c].C0 = rawC0;
      contacts[c].tangentC0 = initialDelta.dot(contacts[c].tangent0);
      contacts[c].tangentC1 = initialDelta.dot(contacts[c].tangent1);
    }
  }

  // Contact row ids are the stable identity shared by body CSR, deferred
  // output tokens and report writeback for the complete solve epoch.  Moving
  // contact objects here would leave every already-built sidecar pointing at
  // a different physical row.  Deterministic consumers sort row-id views;
  // the authoritative storage itself is never permuted after prep.


  iterationState.hasBodyStaticContact = context.hasBodyStaticContact;
  iterationState.linearVelAtSolveStart =
      numContacts > 0 ? &context.linearVelAtSolveStart : nullptr;
  const bool impactActiveSetTransition =
      hasAvbdImpactActiveSetTransition(
          bodies, numBodies, contacts, numContacts,
          context.linearVelAtSolveStart, context.angularVelAtSolveStart,
          dt, mConfig.bounceApproachSpeedThreshold());
  const bool useChebyshev =
      !hasDeformableAnchorContact && !impactActiveSetTransition &&
      mConfig.chebyshevRho > 0.0f &&
      mConfig.chebyshevRho < 1.0f;
  iterationState.useChebyshev = useChebyshev;
  iterationState.chebyOmega = 1.0f;
  if (useChebyshev) {
    iterationState.chebyPrevPos.resize(numBodies);
    iterationState.chebyPrevPrevPos.resize(numBodies);
    iterationState.chebyPrevRot.resize(numBodies);
    iterationState.chebyPrevPrevRot.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i) {
      iterationState.chebyPrevPos[i] = bodies[i].position;
      iterationState.chebyPrevPrevPos[i] = bodies[i].position;
      iterationState.chebyPrevRot[i] = bodies[i].rotation;
      iterationState.chebyPrevPrevRot[i] = bodies[i].rotation;
    }
  }
  const physx::PxU32 iters =
      physx::PxMax(mConfig.iterations, iterationOverride);
  iterationState.iters = iters;
  iterationState.minIterations =
      physx::PxMin(iters, physx::PxU32(4));
  iterationState.enableEarlyStop =
      mConfig.enableEarlyStop &&
      iters - iterationState.minIterations > 1;
  iterationState.rotationTolerance =
      physx::PxMax(4.0f * mConfig.positionTolerance /
                       physx::PxMax(mConfig.lengthScale, 1e-6f),
                   1e-4f);
  iterationState.consecutiveConvergedIterations = 0;
  if (iterationState.enableEarlyStop) {
    iterationState.earlyStopPrevPos.resize(numBodies);
    iterationState.earlyStopPrevRot.resize(numBodies);
  }
  return true;
}

void AvbdSolver::finishRigidSolve(AvbdRigidSolveContext &context) {
  AvbdRigidSolveIterationState &iterationState = context.iteration;
  if (!iterationState.bodies || !iterationState.stats)
    return;
  const physx::PxArray<bool> touchesKinematicShell;
  AvbdSolverStats &stats = *iterationState.stats;
  postAlStages(
      iterationState.dt, context.invDt, iterationState.bodies,
      iterationState.numBodies, iterationState.contacts,
      iterationState.numContacts, iterationState.contactMap, context.gravity,
      context.hasBodyStaticContact, context.deformableFastImpactIsland,
      context.touchingBodyStatic,
      iterationState.numContacts > 0
          ? &context.linearVelAtSolveStart
          : nullptr,
      iterationState.numContacts > 0
          ? &context.angularVelAtSolveStart
          : nullptr,
      false, false, nullptr, 0, nullptr, 0, nullptr, 0,
      touchesKinematicShell, nullptr,
      nullptr, nullptr, 0, false, false, false, nullptr, 0, stats,
      &context.postAlContactWork);
}

void AvbdSolver::solve(physx::PxReal dt, AvbdSolverBody *bodies,
                       physx::PxU32 numBodies, AvbdContactConstraint *contacts,
                       physx::PxU32 numContacts, const physx::PxVec3 &gravity,
                       const AvbdBodyConstraintMap *contactMap,
                       AvbdColorBatch *colorBatches, physx::PxU32 numColors,
                       physx::PxU32 iterationOverride,
                       AvbdSolverStats &stats) {
  PX_PROFILE_ZONE("AVBD.solve", 0);
  AvbdRigidSolveContext context;
  if (!prepareRigidSolve(dt, bodies, numBodies, contacts, numContacts, gravity,
                         contactMap, colorBatches, numColors,
                         iterationOverride, stats, context))
    return;
  while (context.iteration.iter < context.iteration.iters) {
    if (!advanceRigidSolveIterations(context.iteration))
      break;
  }
  finishRigidSolve(context);
}

void AvbdSolver::solveIsland(
    physx::PxReal dt, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, AvbdD6JointConstraint *d6Joints,
    physx::PxU32 numD6, AvbdGearJointConstraint *gearJoints,
    physx::PxU32 numGear, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdColorBatch *colorBatches, physx::PxU32 numColors,
    physx::PxU32 iterationOverride,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    FeatherstoneArticulation *const *articulationForBody,
    const physx::PxU32 *linkIndexForBody,
    AvbdSolverStats &stats,
    AvbdRigidSolveContext *deferredRigidContext) {
  PX_PROFILE_ZONE("AVBD.solveIsland", 0);

  // solveIsland is the sole public island entry and owns transient
  // classification before dispatching to either internal solve module.
  stats.reset();
  if (deferredRigidContext)
    deferredRigidContext->postAlContactWork.reset();
  const bool hasJoints = (numD6 > 0 || numGear > 0);
  const bool hasDeformableSoftVbd =
      softParticles && numSoftParticles > 0 && softBodies &&
      numSoftBodies > 0 &&
      (numSoftContacts == 0 || softContacts);
  const bool contactOnlyTargetOwnership =
      !hasJoints && !hasDeformableSoftVbd;
  // This is the deferred non-ordered rigid path admitted by the task graph.
  // Keep ordered/deterministic and synchronous entries on the original
  // classification sequence even when their island data happens to match.
  const bool fastDeferredRigidClassification =
      deferredRigidContext && contactOnlyTargetOwnership &&
      mConfig.enableParallelization &&
      !mConfig.requiresOrderedBackend();
  physx::PxU8 postAlContactWorkMask = 0;
  bool postAlContactWorkKnown = fastDeferredRigidClassification;
  physx::PxArray<physx::PxU32> rigidStaticContactsPerBody(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i)
    rigidStaticContactsPerBody[i] = 0;
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    resetAvbdContactObjectiveProgram(contacts[c].objectiveProgram);
    setRigidMaterialConsumed(contacts[c], false);
    if (!assignAvbdVelocityObjective(
            contacts[c].objectiveProgram,
            AvbdVelocityObjectiveOwner::PositionAL,
            AvbdVelocityObjectiveKind::GeometryNormal,
            AvbdVelocityObjectiveSpan::Normal,
            AvbdVelocityObjectiveReconstruction::PoseDerived,
            1u,
            contacts[c].cacheKey))
      continue;
    if (!contactOnlyTargetOwnership ||
        !isBodyVsStaticContact(contacts[c].header.bodyIndexA,
                               contacts[c].header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contacts[c]))
      continue;
    const physx::PxU32 bodyIndex =
        contacts[c].header.bodyIndexA < numBodies
            ? contacts[c].header.bodyIndexA
            : contacts[c].header.bodyIndexB;
    rigidStaticContactsPerBody[bodyIndex]++;
  }
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &contact = contacts[c];
    if (!contactOnlyTargetOwnership ||
        !isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        hasDeformableStaticAnchor(contact) ||
        hasKinematicShellAnchor(contact))
      continue;
    const physx::PxU32 bodyIndex =
        contact.header.bodyIndexA < numBodies
            ? contact.header.bodyIndexA
            : contact.header.bodyIndexB;
    const physx::PxReal targetNormal =
        contact.targetVelocity.dot(contact.contactNormal);
    const physx::PxReal targetTangent0 =
        contact.targetVelocity.dot(contact.tangent0);
    const physx::PxReal targetTangent1 =
        contact.targetVelocity.dot(contact.tangent1);
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxReal dynamicLinearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal dynamicAngularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    const bool hasTangentTarget =
        physx::PxAbs(targetTangent0) > 1e-6f ||
        physx::PxAbs(targetTangent1) > 1e-6f;
    const bool defaultDynamicScales =
        physx::PxAbs(dynamicLinearScale - 1.0f) <= 1e-6f &&
        physx::PxAbs(dynamicAngularScale - 1.0f) <= 1e-6f;
    const physx::PxU32 angularLocks =
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
        physx::PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
    const bool allAngularMotionLocked =
        (bodies[bodyIndex].lockFlags & angularLocks) == angularLocks;
    const physx::PxVec3 staticPoint =
        dynamicIsA ? contact.contactPointB : contact.contactPointA;
    const physx::PxReal lengthTolerance =
        1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0f);
    const bool stationaryStatic =
        (staticPoint - contact.staticPrevWorldPoint).magnitudeSquared() <=
        lengthTolerance * lengthTolerance;
    const bool pureUnlimitedTangentTarget =
        physx::PxAbs(targetNormal) <= 1e-6f &&
        contact.maxImpulse >= PX_MAX_REAL;
    const bool strictFiniteCombinedTarget =
        targetNormal > 1e-6f && contact.maxImpulse >= 0.0f &&
        contact.maxImpulse < PX_MAX_REAL &&
        physx::PxIsFinite(contact.maxImpulse) && allAngularMotionLocked &&
        stationaryStatic;
    if (rigidStaticContactsPerBody[bodyIndex] == 1 &&
        (contact.friction > 0.0f || contact.staticFriction > 0.0f) &&
        hasTangentTarget && contact.restitution == 0.0f &&
        defaultDynamicScales &&
        (pureUnlimitedTangentTarget || strictFiniteCombinedTarget)) {
      if (!assignAvbdVelocityObjective(
              contact.objectiveProgram,
              AvbdVelocityObjectiveOwner::PointFinalize,
              AvbdVelocityObjectiveKind::TangentTarget,
              strictFiniteCombinedTarget
                  ? AvbdVelocityObjectiveSpan::NormalAndTangentCone
                  : AvbdVelocityObjectiveSpan::TangentCone,
              AvbdVelocityObjectiveReconstruction::PoseDerived,
              1u,
              contact.cacheKey))
        continue;

      // The nonlinear position solve may rotate a cached local contact point
      // while enforcing its normal row.  For a central contact on an
      // isotropic body, that row has no physical angular Jacobian and cannot
      // create tangent-space generalized velocity.  Mark this independently
      // so velocity reconstruction can retain only the normal impulse span
      // before the unique tangent target is applied.
      if (!pureUnlimitedTangentTarget)
        continue;

      if (stationaryStatic && isAvbdCentralNormalIsotropicContact(
                                  bodies[bodyIndex], contact, bodyIndex,
                                  mConfig.lengthScale)) {
        AvbdCompiledVelocityObjective *objective =
            findAvbdVelocityObjective(
                contact.objectiveProgram,
                AvbdVelocityObjectiveOwner::PointFinalize,
                AvbdVelocityObjectiveKind::TangentTarget);
        if (objective)
          objective->reconstruction =
              AvbdVelocityObjectiveReconstruction::NormalResponseSpan;
      }
  }
  }

  // Ordinary rigid material response is owned by the complete dynamic-body
  // contact component. Static endpoints are leaves (their infinite mass
  // introduces no cross-row response), while body-static and dynamic-dynamic
  // rows incident to the same dynamic graph are compiled together. Any
  // unsupported incident row rejects the complete component; no supported
  // subset is claimed by this owner.
  if (contactOnlyTargetOwnership && contacts && numContacts > 0 &&
      contactMap && contactMap->numBodies == numBodies &&
      contactMap->constraintOffsets && contactMap->constraintCounts &&
      (contactMap->totalConstraintRefs == 0 ||
       contactMap->constraintIndices)) {
    physx::PxArray<physx::PxU8> visitedBodies(numBodies);
    physx::PxArray<physx::PxU8> visitedContacts(numContacts);
    for (physx::PxU32 body = 0; body < numBodies; ++body)
      visitedBodies[body] = 0;
    for (physx::PxU32 contact = 0; contact < numContacts; ++contact)
      visitedContacts[contact] = 0;

    for (physx::PxU32 seed = 0; seed < numBodies; ++seed) {
      if (visitedBodies[seed])
        continue;
      physx::PxArray<physx::PxU32> bodyQueue;
      physx::PxArray<physx::PxU32> componentContacts;
      bodyQueue.pushBack(seed);
      visitedBodies[seed] = 1;
      bool supported = true;
      bool hasStaticEndpoint = false;

      for (physx::PxU32 queueIndex = 0;
           queueIndex < bodyQueue.size(); ++queueIndex) {
        const physx::PxU32 bodyIndex = bodyQueue[queueIndex];
        if (bodies[bodyIndex].invMass <= 0.0f ||
            bodies[bodyIndex].lockFlags != 0)
          supported = false;
        const physx::PxU32 *mapIndices = nullptr;
        physx::PxU32 mapCount = 0;
        const bool hasMapRange = getAvbdBodyContactRange(
            contactMap, bodyIndex, mapIndices, mapCount);
        if (!hasMapRange) {
          supported = false;
          break;
        }
        const physx::PxU32 loopCount = mapCount;
        for (physx::PxU32 loopIndex = 0; loopIndex < loopCount;
             ++loopIndex) {
          const physx::PxU32 c = mapIndices[loopIndex];
          if (c >= numContacts) {
            supported = false;
            break;
          }
          AvbdContactConstraint &contact = contacts[c];
          const physx::PxU32 bodyA = contact.header.bodyIndexA;
          const physx::PxU32 bodyB = contact.header.bodyIndexB;
          if (bodyA != bodyIndex && bodyB != bodyIndex)
            continue;
          if (!visitedContacts[c]) {
            visitedContacts[c] = 1;
            componentContacts.pushBack(c);
          }

          const bool dynamicA = bodyA < numBodies;
          const bool dynamicB = bodyB < numBodies;
          hasStaticEndpoint = hasStaticEndpoint || (dynamicA != dynamicB);
          if (!dynamicA && !dynamicB) {
            supported = false;
            continue;
          }
          if (hasDeformableStaticAnchor(contact) ||
              hasKinematicShellAnchor(contact) ||
              !contact.targetVelocity.isFinite() ||
              contact.targetVelocity.magnitudeSquared() > 1.0e-12f ||
              !contact.contactNormal.isFinite() ||
              !contact.tangent0.isFinite() ||
              !contact.tangent1.isFinite() ||
              contact.contactNormal.magnitudeSquared() <= 1.0e-12f ||
              contact.tangent0.magnitudeSquared() <= 1.0e-12f ||
              contact.tangent1.magnitudeSquared() <= 1.0e-12f ||
              !physx::PxIsFinite(contact.friction) ||
              !physx::PxIsFinite(contact.staticFriction) ||
              contact.friction < 0.0f ||
              contact.staticFriction < 0.0f ||
              !physx::PxIsFinite(contact.restitution) ||
              contact.restitution < 0.0f ||
              contact.restitution > 1.0f ||
              !physx::PxIsFinite(contact.maxImpulse) ||
              contact.maxImpulse < PX_MAX_REAL) {
            supported = false;
          }
          if (dynamicA &&
              (!physx::PxIsFinite(contact.invMassScaleA) ||
               !physx::PxIsFinite(contact.invInertiaScaleA) ||
               physx::PxAbs(contact.invMassScaleA - 1.0f) > 1.0e-6f ||
               physx::PxAbs(contact.invInertiaScaleA - 1.0f) > 1.0e-6f))
            supported = false;
          if (dynamicB &&
              (!physx::PxIsFinite(contact.invMassScaleB) ||
               !physx::PxIsFinite(contact.invInertiaScaleB) ||
               physx::PxAbs(contact.invMassScaleB - 1.0f) > 1.0e-6f ||
               physx::PxAbs(contact.invInertiaScaleB - 1.0f) > 1.0e-6f))
            supported = false;

          if (dynamicA && !visitedBodies[bodyA]) {
            visitedBodies[bodyA] = 1;
            bodyQueue.pushBack(bodyA);
          }
          if (dynamicB && !visitedBodies[bodyB]) {
            visitedBodies[bodyB] = 1;
            bodyQueue.pushBack(bodyB);
          }

          if (dynamicA != dynamicB) {
            const bool dynamicIsA = dynamicA;
            const physx::PxVec3 staticPoint =
                dynamicIsA ? contact.contactPointB
                           : contact.contactPointA;
            const physx::PxReal lengthTolerance =
                1.0e-4f *
                physx::PxMax(mConfig.lengthScale, 1.0f);
            if ((staticPoint - contact.staticPrevWorldPoint)
                    .magnitudeSquared() >
                lengthTolerance * lengthTolerance)
              supported = false;
          }
        }
      }

      // The passive velocity component is a closed material system.  Static
      // support is already owned by its finite PositionAL/manifold closure;
      // mixing those persistent support rows into an impact component would
      // incorrectly reinterpret the final nonlinear AL dual as a removable
      // impulse and would connect the complete resting stack.
      if (!supported || componentContacts.empty() || hasStaticEndpoint)
        continue;
      // The component seed is already a unique dense island index.  Use it
      // directly as the transient grouping id; no contact fingerprint, hash
      // table, or minimum-key reduction is needed in this hot path.
      const physx::PxU64 objectiveKey = physx::PxU64(seed) + 1u;
      for (physx::PxU32 index = 0;
           index < componentContacts.size(); ++index) {
        const AvbdContactConstraint &contact =
            contacts[componentContacts[index]];
        if (!canAssignAvbdVelocityObjective(
                contact.objectiveProgram,
                AvbdVelocityObjectiveOwner::ComponentFinalize,
                AvbdVelocityObjectiveKind::PassiveFriction,
                AvbdVelocityObjectiveSpan::NormalAndTangentCone,
                AvbdVelocityObjectiveReconstruction::
                    PoseResidual,
                componentContacts.size(),
                objectiveKey)) {
          supported = false;
          break;
        }
      }
      if (!supported)
        continue;
      for (physx::PxU32 index = 0;
           index < componentContacts.size(); ++index) {
        assignAvbdVelocityObjective(
            contacts[componentContacts[index]].objectiveProgram,
            AvbdVelocityObjectiveOwner::ComponentFinalize,
            AvbdVelocityObjectiveKind::PassiveFriction,
            AvbdVelocityObjectiveSpan::NormalAndTangentCone,
            AvbdVelocityObjectiveReconstruction::PoseResidual,
            componentContacts.size(),
            objectiveKey);
      }
    }
  }

  // A strict one-to-four-row rigid-static friction manifold has one coupled
  // material-velocity objective. This includes a shared explicit tangential
  // target or the passive zero-target case. Mark every physical row so
  // position friction and the body-static sweep cannot replay it. This owner
  // is admitted only for contact-only bodies whose complete incident set is
  // stationary rigid-static contact. A central single point on an isotropic
  // body has no normal angular response, so its material velocity can always
  // be rebuilt exactly from the captured solve-start inertial epoch. The same
  // baseline is required for a damped manifold outside a restitution-active
  // impact: position AL owns geometry, and subtracting its final multipliers
  // through a fresh material Jacobian after unequal linear/angular damping is
  // not an exact inverse and can perform positive work. Restitution-active
  // multi-point impacts retain pose-residual reconstruction so their angular
  // support distribution is finalized by the coupled impact frontier.
  for (physx::PxU32 bodyIndex = 0; bodyIndex < numBodies; ++bodyIndex) {
    if (rigidStaticContactsPerBody[bodyIndex] == 0 ||
        rigidStaticContactsPerBody[bodyIndex] > 4)
      continue;

    bool supported = true;
    bool haveReferenceTarget = false;
    bool restitutionActive = false;
    physx::PxVec3 referenceDynamicTarget(0.0f);
    const AvbdContactConstraint *singlePointContact = nullptr;
    physx::PxU64 objectiveKey = ~physx::PxU64(0);
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    const bool hasMapRange = getAvbdBodyContactRange(
        contactMap, bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = hasMapRange ? mapCount : numContacts;
    physx::PxVec3 solveStartLinear, solveStartAngular;
    computeAvbdMaterialSolveStartVelocity(
        bodies[bodyIndex], gravity, dt,
        solveStartLinear, solveStartAngular);
    bool componentOwned = false;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      if (c < numContacts &&
          (contacts[c].header.bodyIndexA == bodyIndex ||
           contacts[c].header.bodyIndexB == bodyIndex) &&
          hasVelocityPassiveFrictionComponentOwner(contacts[c])) {
        componentOwned = true;
        break;
      }
    }
    if (componentOwned)
      continue;
    for (physx::PxU32 loopIndex = 0;
         loopIndex < loopCount && supported; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      AvbdContactConstraint &contact = contacts[c];
      if (contact.header.bodyIndexA != bodyIndex &&
          contact.header.bodyIndexB != bodyIndex)
        continue;
      singlePointContact = &contact;
      if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                                 contact.header.bodyIndexB, numBodies) ||
          hasDeformableStaticAnchor(contact) ||
          hasKinematicShellAnchor(contact)) {
        supported = false;
        break;
      }
      const bool dynamicIsA = contact.header.bodyIndexA == bodyIndex;
      const physx::PxVec3 dynamicNormal =
          contact.contactNormal * (dynamicIsA ? 1.0f : -1.0f);
      const physx::PxVec3 dynamicLocalPoint =
          dynamicIsA ? contact.contactPointA : contact.contactPointB;
      const physx::PxVec3 dynamicArm =
          bodies[bodyIndex].rotation.rotate(dynamicLocalPoint);
      const physx::PxReal solveStartNormalSpeed =
          (solveStartLinear + solveStartAngular.cross(dynamicArm))
              .dot(dynamicNormal);
      restitutionActive =
          restitutionActive ||
          (contact.restitution > 0.0f &&
           solveStartNormalSpeed <
               -mConfig.bounceApproachSpeedThreshold());
      const physx::PxReal dynamicLinearScale =
          dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
      const physx::PxReal dynamicAngularScale =
          dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
      const physx::PxReal targetNormal =
          contact.targetVelocity.dot(contact.contactNormal);
      const physx::PxVec3 staticPoint =
          dynamicIsA ? contact.contactPointB : contact.contactPointA;
      const physx::PxReal lengthTolerance =
          1.0e-4f * physx::PxMax(mConfig.lengthScale, 1.0f);
      const bool stationaryStatic =
          (staticPoint - contact.staticPrevWorldPoint).magnitudeSquared() <=
          lengthTolerance * lengthTolerance;
      const physx::PxVec3 dynamicTarget =
          contact.targetVelocity * (dynamicIsA ? 1.0f : -1.0f);
      if ((contact.friction <= 0.0f &&
           contact.staticFriction <= 0.0f) ||
          physx::PxAbs(targetNormal) > 1.0e-6f ||
          contact.maxImpulse < PX_MAX_REAL ||
          !physx::PxIsFinite(contact.restitution) ||
          contact.restitution < 0.0f || contact.restitution > 1.0f ||
          physx::PxAbs(dynamicLinearScale - 1.0f) > 1.0e-6f ||
          physx::PxAbs(dynamicAngularScale - 1.0f) > 1.0e-6f ||
          !stationaryStatic) {
        supported = false;
        break;
      }
      if (!haveReferenceTarget) {
        referenceDynamicTarget = dynamicTarget;
        haveReferenceTarget = true;
      } else if ((dynamicTarget - referenceDynamicTarget).magnitudeSquared() >
                 1.0e-10f) {
        supported = false;
      }
      objectiveKey = physx::PxMin(objectiveKey, contact.cacheKey);
    }
    if (!supported || !haveReferenceTarget)
      continue;
    const bool passiveFriction =
        referenceDynamicTarget.magnitudeSquared() <= 1.0e-12f;
    const bool exactSinglePointBaseline =
        rigidStaticContactsPerBody[bodyIndex] == 1u &&
        singlePointContact && isAvbdCentralNormalIsotropicContact(
                                  bodies[bodyIndex], *singlePointContact,
                                  bodyIndex, mConfig.lengthScale);
    const bool dampedInertialBaseline =
        !restitutionActive &&
        (bodies[bodyIndex].linearDamping > 0.0f ||
         bodies[bodyIndex].angularDampingBody > 0.0f);
    const AvbdVelocityObjectiveReconstruction reconstruction =
        exactSinglePointBaseline || dampedInertialBaseline
            ? AvbdVelocityObjectiveReconstruction::SolveStartInertial
            : AvbdVelocityObjectiveReconstruction::PoseResidual;
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      const AvbdContactConstraint &contact = contacts[c];
      if ((contact.header.bodyIndexA == bodyIndex ||
           contact.header.bodyIndexB == bodyIndex) &&
          !canAssignAvbdVelocityObjective(
              contact.objectiveProgram,
              AvbdVelocityObjectiveOwner::ManifoldFinalize,
              passiveFriction
                  ? AvbdVelocityObjectiveKind::PassiveFriction
                  : AvbdVelocityObjectiveKind::TangentTarget,
              AvbdVelocityObjectiveSpan::NormalAndTangentCone,
              reconstruction,
              rigidStaticContactsPerBody[bodyIndex],
              objectiveKey)) {
        supported = false;
        break;
      }
    }
    if (!supported) {
      for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
        const physx::PxU32 c =
            hasMapRange ? mapIndices[loopIndex] : loopIndex;
        AvbdContactConstraint &contact = contacts[c];
        if (contact.header.bodyIndexA == bodyIndex ||
            contact.header.bodyIndexB == bodyIndex)
          invalidateAvbdVelocityObjective(contact.objectiveProgram);
      }
      continue;
    }
    for (physx::PxU32 loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
      const physx::PxU32 c =
          hasMapRange ? mapIndices[loopIndex] : loopIndex;
      AvbdContactConstraint &contact = contacts[c];
      if (contact.header.bodyIndexA == bodyIndex ||
          contact.header.bodyIndexB == bodyIndex) {
        assignAvbdVelocityObjective(
            contact.objectiveProgram,
            AvbdVelocityObjectiveOwner::ManifoldFinalize,
            passiveFriction
                ? AvbdVelocityObjectiveKind::PassiveFriction
                : AvbdVelocityObjectiveKind::TangentTarget,
            AvbdVelocityObjectiveSpan::NormalAndTangentCone,
            reconstruction,
            rigidStaticContactsPerBody[bodyIndex],
            objectiveKey);
      }
    }
  }

  // Specialized target/manifold/component programs have first claim.
  // Compile all remaining ordinary rigid contact sources through the same
  // helper used by joint islands, so owner classification has one entry point.
  if (contactOnlyTargetOwnership)
    compileAvbdOrdinaryRigidContactObjectives(
        contacts, numContacts, numBodies, contactMap);

  // Strict Phase-3 owner: ordinary zero-target deformable/static tangents use
  // the same position-level row in primal and dual. Joint-mixed islands remain
  // excluded until they have an independent capability fixture. NP contacts
  // cannot create a synthesized soft/direct-shell batch at this boundary.
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    AvbdContactConstraint &contact = contacts[c];
    if (!isBodyVsStaticContact(contact.header.bodyIndexA,
                               contact.header.bodyIndexB, numBodies) ||
        !hasDeformableStaticAnchor(contact) ||
        (contact.friction <= 0.0f &&
         contact.staticFriction <= 0.0f))
      continue;
    if (hasJoints || hasDeformableSoftVbd)
      continue;
    if (contact.restitution != 0.0f)
      continue;
    if (contact.maxImpulse < PX_MAX_REAL)
      continue;
    const physx::PxReal targetNormal =
        contact.targetVelocity.dot(contact.contactNormal);
    const physx::PxReal targetTangent0 =
        contact.targetVelocity.dot(contact.tangent0);
    const physx::PxReal targetTangent1 =
        contact.targetVelocity.dot(contact.tangent1);
    if (physx::PxAbs(targetNormal) > 1.0e-6f ||
        physx::PxAbs(targetTangent0) > 1.0e-6f ||
        physx::PxAbs(targetTangent1) > 1.0e-6f) {
      continue;
    }
    const bool dynamicIsA = contact.header.bodyIndexA < numBodies;
    const physx::PxReal dynamicLinearScale =
        dynamicIsA ? contact.invMassScaleA : contact.invMassScaleB;
    const physx::PxReal dynamicAngularScale =
        dynamicIsA ? contact.invInertiaScaleA : contact.invInertiaScaleB;
    if (physx::PxAbs(dynamicLinearScale - 1.0f) > 1.0e-6f ||
        physx::PxAbs(dynamicAngularScale - 1.0f) > 1.0e-6f) {
      continue;
    }
    if (!assignAvbdVelocityObjective(
            contact.objectiveProgram,
            AvbdVelocityObjectiveOwner::PositionAL,
            AvbdVelocityObjectiveKind::PassiveFriction,
            AvbdVelocityObjectiveSpan::TangentCone,
            AvbdVelocityObjectiveReconstruction::PoseDerived,
            1u,
            contact.cacheKey))
      continue;
  }

  // Publish the remaining authored source slots as an explicit migration
  // backlog. Geometry normal is already compiled independently above.
  // Material normal exists for every contact; material tangent exists only
  // when friction or an authored tangential target is present.
  //
  // On the admitted deferred non-ordered rigid path, publication and
  // validation have no cross-contact dependency: both only read/write the
  // current program.  Fuse them to avoid one complete wide-contact walk.
  // Ordered/synchronous paths retain the original two-pass sequence.
  if (fastDeferredRigidClassification) {
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      AvbdContactConstraint &contact = contacts[c];
      physx::PxU8 authoredSourceSlots =
          eCONTACT_SOURCE_GEOMETRY_NORMAL |
          eCONTACT_SOURCE_MATERIAL_NORMAL;
      const physx::PxReal targetTangent0 =
          contact.targetVelocity.dot(contact.tangent0);
      const physx::PxReal targetTangent1 =
          contact.targetVelocity.dot(contact.tangent1);
      if (contact.friction > 0.0f || contact.staticFriction > 0.0f ||
          physx::PxAbs(targetTangent0) > 1.0e-6f ||
          physx::PxAbs(targetTangent1) > 1.0e-6f ||
          hasVelocityPassiveFrictionComponentOwner(contact))
        authoredSourceSlots = physx::PxU8(
            authoredSourceSlots |
            eCONTACT_SOURCE_MATERIAL_TANGENT);
      setAvbdContactObjectiveLegacySources(
          contact.objectiveProgram, authoredSourceSlots);
      if (!isValidAvbdContactObjectiveProgram(contact.objectiveProgram)) {
        invalidateAvbdVelocityObjective(contact.objectiveProgram);
        postAlContactWorkKnown = false;
      } else {
        markAvbdContactObjectiveProgramValidated(contact.objectiveProgram);
        postAlContactWorkMask = physx::PxU8(
            postAlContactWorkMask |
            collectValidatedPostAlContactWork(contact, bodies, numBodies));
      }
    }
  } else {
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      AvbdContactConstraint &contact = contacts[c];
      physx::PxU8 authoredSourceSlots =
          eCONTACT_SOURCE_GEOMETRY_NORMAL |
          eCONTACT_SOURCE_MATERIAL_NORMAL;
      const physx::PxReal targetTangent0 =
          contact.targetVelocity.dot(contact.tangent0);
      const physx::PxReal targetTangent1 =
          contact.targetVelocity.dot(contact.tangent1);
      if (contact.friction > 0.0f || contact.staticFriction > 0.0f ||
          physx::PxAbs(targetTangent0) > 1.0e-6f ||
          physx::PxAbs(targetTangent1) > 1.0e-6f ||
          hasVelocityPassiveFrictionComponentOwner(contact))
        authoredSourceSlots = physx::PxU8(
            authoredSourceSlots |
            eCONTACT_SOURCE_MATERIAL_TANGENT);
      setAvbdContactObjectiveLegacySources(
          contact.objectiveProgram, authoredSourceSlots);
    }

    // The compiled program is the only ownership authority consumed below.
    // Any internally inconsistent program is converted to the explicit
    // fail-closed state before position or velocity stages can inspect it.
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      if (!isValidAvbdContactObjectiveProgram(
              contacts[c].objectiveProgram)) {
        invalidateAvbdVelocityObjective(
            contacts[c].objectiveProgram);
      } else {
        markAvbdContactObjectiveProgramValidated(
            contacts[c].objectiveProgram);
      }
    }
  }

  // One island entry: joint/genuine-soft module vs contact-only module. NP
  // contact data cannot synthesize soft particles or route through a second
  // primal.
  if (deferredRigidContext) {
    if (hasJoints || hasDeformableSoftVbd)
      return;
    if (prepareRigidSolve(dt, bodies, numBodies, contacts, numContacts,
                          gravity, contactMap, colorBatches, numColors,
                          iterationOverride, stats, *deferredRigidContext) &&
        postAlContactWorkKnown) {
      deferredRigidContext->postAlContactWork.publish(postAlContactWorkMask);
    }
    return;
  }
  if (hasJoints || hasDeformableSoftVbd) {
    solveWithJoints(dt, bodies, numBodies, contacts, numContacts, d6Joints,
                    numD6, gearJoints, numGear, gravity, contactMap, d6Map,
                    gearMap, colorBatches, numColors, iterationOverride,
                    softParticles, numSoftParticles, softBodies, numSoftBodies,
                    softContacts, numSoftContacts, softExecutionPlan,
                    articulationForBody,
                    linkIndexForBody, stats);
  } else {
    solve(dt, bodies, numBodies, contacts, numContacts, gravity, contactMap,
          colorBatches, numColors, iterationOverride, stats);
  }

}

//=============================================================================
// Augmented Lagrangian Multiplier Update
//
// 6x6 path (ref: AVBD3D solver.cpp L142-164):
//   lambda = clamp(penalty*C + lambda, fmin, fmax)
//   if lambda within bounds: penalty += beta * |C|
//   penalty = min(penalty, PENALTY_MAX)
//
// Fast path: XPBD formula (unchanged)
//=============================================================================

void AvbdSolver::updateLagrangianMultipliers(AvbdSolverBody *bodies,
                                             physx::PxU32 numBodies,
                                             AvbdContactConstraint *contacts,
                                             physx::PxU32 numContacts,
                                             physx::PxReal dt,
                                             AvbdSolverStats &stats) {
  (void)stats;
  physx::PxReal totalError = 0.0f;
  KahanSum totalErrorKahan;
  const bool useKahan =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eUSE_KAHAN_SUMMATION);
  const physx::PxReal invDt2 =
      dt > 0.0f ? 1.0f / (dt * dt) : 0.0f;
  physx::PxU32 numActive = 0;

  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;
    const bool deformableStaticAnchor = hasDeformableStaticAnchor(contacts[c]);

    // Compute current violation
    physx::PxReal violation = 0.0f;
    AvbdSolverBody *bodyA = nullptr;
    AvbdSolverBody *bodyB = nullptr;

    if (bodyAIdx < numBodies && bodyBIdx < numBodies) {
      bodyA = &bodies[bodyAIdx];
      bodyB = &bodies[bodyBIdx];
      violation = computeContactViolation(contacts[c], *bodyA, *bodyB);
    } else if (bodyAIdx < numBodies) {
      bodyA = &bodies[bodyAIdx];
      physx::PxVec3 worldPointA =
          bodyA->position + bodyA->rotation.rotate(contacts[c].contactPointA);
      physx::PxVec3 worldPointB = contacts[c].contactPointB;
      violation = (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
    } else if (bodyBIdx < numBodies) {
      bodyB = &bodies[bodyBIdx];
      physx::PxVec3 worldPointA = contacts[c].contactPointA;
      physx::PxVec3 worldPointB =
          bodyB->position + bodyB->rotation.rotate(contacts[c].contactPointB);
      violation = (worldPointA - worldPointB).dot(contacts[c].contactNormal) +
                  contacts[c].penetrationDepth;
    }

    // The reference beta is calibrated in unit-length coordinates.  Normalize
    // the raw world-space violation while preserving the established
    // lengthScale=1 behavior and its impact-energy/stability envelope.
    const physx::PxReal lengthScale =
        physx::PxMax(mConfig.lengthScale, 1e-6f);
    const physx::PxReal beta = mConfig.avbdBeta / lengthScale;
    const physx::PxReal penaltyMax = mConfig.avbdPenaltyMax;

    // Alpha blending (ref: AVBD3D manifold.cpp computeConstraint)
    violation -= mConfig.avbdAlpha * contacts[c].C0;
    if (deformableStaticAnchor) {
      violation = finalizeBodyVsStaticViolation(violation,
                                              contacts[c].penetrationDepth);
    }

    // =====================================================================
    // AL dual + Coulomb cone (avbd-demo3d manifold.cpp updateDual):
    //   F = K*C + ?;  Fn = min(0,F_n);  ||Ft|| <= ?|Fn|;  store F as ?.
    // Bound uses normal force F_n (not raw ? alone). Tangents are a 2D cone.
    // =====================================================================
    physx::PxReal newLambda = 0.0f;
    {
      const physx::PxReal pen = contacts[c].header.penalty;
      const physx::PxReal oldLambda = contacts[c].header.lambda;
      const bool positionTangentOwner =
          hasPositionTangentParticipation(contacts[c]);
      const physx::PxReal mu =
          positionTangentOwner ? contactCoulombMu(contacts[c]) : 0.0f;

      physx::PxReal tC0 = 0.0f, tC1 = 0.0f;
      if (positionTangentOwner &&
          (contacts[c].friction > 0.0f ||
           contacts[c].staticFriction > 0.0f)) {
        const bool bodyVsStatic =
            isBodyVsStaticContact(bodyAIdx, bodyBIdx, numBodies);
        physx::PxVec3 worldPosA, worldPosB, prevWorldPosA, prevWorldPosB;
        if (bodyAIdx < numBodies) {
          worldPosA =
              bodies[bodyAIdx].position +
              bodies[bodyAIdx].rotation.rotate(contacts[c].contactPointA);
          prevWorldPosA =
              bodies[bodyAIdx].prevPosition +
              bodies[bodyAIdx].prevRotation.rotate(contacts[c].contactPointA);
        } else {
          worldPosA = contacts[c].contactPointA;
          prevWorldPosA = deformableStaticAnchor
                              ? contacts[c].staticPrevWorldPoint
                              : contacts[c].contactPointA;
        }
        if (bodyBIdx < numBodies) {
          worldPosB =
              bodies[bodyBIdx].position +
              bodies[bodyBIdx].rotation.rotate(contacts[c].contactPointB);
          prevWorldPosB =
              bodies[bodyBIdx].prevPosition +
              bodies[bodyBIdx].prevRotation.rotate(contacts[c].contactPointB);
        } else {
          worldPosB = contacts[c].contactPointB;
          prevWorldPosB = deformableStaticAnchor
                              ? contacts[c].staticPrevWorldPoint
                              : contacts[c].contactPointB;
        }
        const physx::PxVec3 relDisp =
            bodyVsStatic
                ? computeBodyVsStaticRelDisp(worldPosA, prevWorldPosA, worldPosB,
                                             prevWorldPosB, contacts[c],
                                             numBodies)
                : (worldPosA - prevWorldPosA) - (worldPosB - prevWorldPosB);
        const physx::PxReal c0Scale = 1.0f - mConfig.avbdAlpha;
        tC0 = c0Scale * contacts[c].tangentC0 +
              relDisp.dot(contacts[c].tangent0);
        tC1 = c0Scale * contacts[c].tangentC1 +
              relDisp.dot(contacts[c].tangent1);
      }

      physx::PxReal Fn = 0.0f, Ft0 = 0.0f, Ft1 = 0.0f;
      const physx::PxReal preLen = avbdEvaluateContactForcesCone(
          pen, violation, oldLambda, contacts[c].tangentPenalty0, tC0,
          contacts[c].tangentLambda0, contacts[c].tangentPenalty1, tC1,
          contacts[c].tangentLambda1, mu, Fn, Ft0, Ft1);
      if (contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f) {
        const physx::PxReal maxNormalForce =
            physx::PxMax(0.0f, contacts[c].maxImpulse) / dt;
        Fn = physx::PxMax(Fn, -maxNormalForce);
        avbdProjectImpulseCone(maxNormalForce * mu, Ft0, Ft1);
      }
      // Coulomb capacity uses the current projected Fn (demo3d). Do not inject
      // m*g here:
      // per-contact weight floors multi-count box corners and glue HelloWorld
      // stacks under ball impact. Resting grip is the post-pass, impact-gated.
      const physx::PxReal nCap = -Fn;
      const physx::PxReal boundedNCap =
          contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f
              ? physx::PxMin(nCap, contacts[c].maxImpulse / dt)
              : nCap;
      newLambda = Fn;
      contacts[c].header.lambda = Fn;
      contacts[c].tangentLambda0 = Ft0;
      contacts[c].tangentLambda1 = Ft1;

      if (newLambda < 0.0f) {
        if (oldLambda >= 0.0f) {
          applyAvbdLoadedTangentPenaltyFloor(
              contacts[c], bodies, numBodies, invDt2,
              mConfig.avbdPenaltyMin);
        }
        physx::PxReal growthDist = physx::PxAbs(violation);
        if (deformableStaticAnchor ||
            (numContacts > 4u &&
             isBodyVsStaticContact(bodyAIdx, bodyBIdx, numBodies)))
          growthDist = physx::PxMin(growthDist, 0.15f * lengthScale);
        contacts[c].header.penalty =
            physx::PxMin(pen + beta * growthDist, penaltyMax);
      }
      const physx::PxReal bounds = boundedNCap * mu;
      if (positionTangentOwner && preLen <= bounds) {
        contacts[c].tangentPenalty0 = physx::PxMin(
            contacts[c].tangentPenalty0 + beta * physx::PxAbs(tC0),
            penaltyMax);
        contacts[c].tangentPenalty1 = physx::PxMin(
            contacts[c].tangentPenalty1 + beta * physx::PxAbs(tC1),
            penaltyMax);
      }
      if (positionTangentOwner) {
        setFrictionStick(contacts[c],
                         avbdFrictionStickFromDual(
                             boundedNCap, mu, preLen, tC0, tC1,
                             AVBD_FRICTION_STICK_THRESH * lengthScale));
      }
    }

    // Track convergence
    if (violation < 0.0f) {
      physx::PxReal err = violation * violation;
      if (useKahan) {
        totalErrorKahan.add(err);
      } else {
        totalError += err;
      }
      numActive++;
    }
  }

  if (useKahan) {
    totalError = totalErrorKahan.sum;
  }

  PX_AVBD_PROFILE_STAT(stats.constraintError =       (numActive > 0) ? sqrtf(totalError / (physx::PxReal)numActive) : 0.0f);
}

void AvbdSolver::solveRigidDualRange(
    AvbdRigidSolveIterationState &state, physx::PxU32 begin,
    physx::PxU32 end) {
  PX_ASSERT(state.bodies && state.contacts);
  PX_ASSERT(begin < end && end <= state.numContacts);
  // The fast path deliberately reuses the established per-contact kernel
  // rather than cloning its numerically sensitive /fp:fast expressions.
  // Admission keeps every range wider than four rows, preserving the only
  // physical branch in that kernel that depends on the supplied row count.
  PX_ASSERT(end - begin > 4u);
  AvbdSolverStats rangeStats;
  rangeStats.reset();
  updateLagrangianMultipliers(
      state.bodies, state.numBodies, state.contacts + begin, end - begin,
      state.dt, rangeStats);
}

//=============================================================================
// Local 6x6 System Solver -- AVBD Reference Algorithm
//
// Implements the AVBD primal update per body (ref: AVBD3D solver.cpp L107-138):
//
//   lhs = M/h^2
//   rhs = lhs * vec6{x - x_inertial, deltaW_inertial}
//   For each constraint on body:
//     f = clamp(penalty * C + lambda, fmin, fmax)
//     rhs += J * f               (Eq. 13)
//     lhs += outer(J, J*penalty)  (Eq. 17)
//   delta = solve(lhs, rhs)
//   x -= delta
//
// Key difference from old code: uses adaptive penalty (per-constraint,
// grows via beta*|C| in dual update) instead of fixed effectiveRho hack.
//=============================================================================

void AvbdSolver::accumulateBodyContactRows(
    AvbdSolverBody &body, physx::PxU32 bodyIndex, AvbdSolverBody *bodies,
    physx::PxU32 numBodies, AvbdContactConstraint *contacts,
    physx::PxU32 numContacts, const AvbdBodyConstraintMap *contactMap,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt, physx::PxReal massInvDt2, AvbdBlock6x6 &A,
    physx::PxVec3 &gLinear, physx::PxVec3 &gAngular,
    physx::PxU32 &numTouching,
    const physx::PxU32 *rigidTargetContactStarts,
    const physx::PxU32 *rigidTargetContactRefs) {

  const bool useRigidTargetContactCsr =
      softContacts && numSoftContacts > 0 &&
      bodyIndex < numBodies && rigidTargetContactStarts &&
      (rigidTargetContactRefs ||
       rigidTargetContactStarts[bodyIndex] ==
           rigidTargetContactStarts[bodyIndex + 1]);
  bool bodyUsesSoftContactNormals = false;
  if (softContacts && numSoftContacts > 0) {
    if (useRigidTargetContactCsr) {
      bodyUsesSoftContactNormals =
          rigidTargetContactStarts[bodyIndex] !=
          rigidTargetContactStarts[bodyIndex + 1];
    } else {
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci) {
        const AvbdSoftContactGeometry &geometry =
            softContacts[sci].geometry;
        if (geometry.hasRigidBodyTarget() &&
            geometry.targetIndex == bodyIndex) {
          bodyUsesSoftContactNormals = true;
          break;
        }
      }
    }
  }

  // Use contactMap for O(K) lookup if available, else O(N) scan
  const physx::PxU32 *mapIndices = nullptr;
  physx::PxU32 mapCount = 0;
  if (contactMap && contactMap->numBodies > 0) {
    contactMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
  }
  const physx::PxU32 loopCount = mapIndices ? mapCount : numContacts;

  const physx::PxReal contactBoostFloor =
      AvbdConstants::AVBD_CONTACT_BOOST_FRACTION * massInvDt2;

  for (physx::PxU32 ci = 0; ci < loopCount; ++ci) {
    const physx::PxU32 c = mapIndices ? mapIndices[ci] : ci;
    const physx::PxU32 bodyAIdx = contacts[c].header.bodyIndexA;
    const physx::PxU32 bodyBIdx = contacts[c].header.bodyIndexB;

    if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
      continue;
    }

    // A real rigid/soft contact owns the normal when both representations are
    // present. This is not reachable from the rigid NP-only solveIsland entry.
    if (bodyUsesSoftContactNormals &&
        hasDeformableStaticAnchor(contacts[c])) {
      continue;
    }

    const bool isBodyA = (bodyAIdx == bodyIndex);
    const physx::PxReal linearResponseScale =
        isBodyA ? contacts[c].invMassScaleA
                : contacts[c].invMassScaleB;
    const physx::PxReal angularResponseScale =
        isBodyA ? contacts[c].invInertiaScaleA
                : contacts[c].invInertiaScaleB;
    if (linearResponseScale <= 0.0f &&
        angularResponseScale <= 0.0f) {
      // Contact-local infinite mass/inertia: this row must not move this body,
      // but the peer still consumes the same row with its own response scales.
      continue;
    }
    AvbdSolverBody *otherBody = nullptr;
    if (isBodyA && bodyBIdx < numBodies) {
      otherBody = &bodies[bodyBIdx];
    } else if (!isBodyA && bodyAIdx < numBodies) {
      otherBody = &bodies[bodyAIdx];
    }

    physx::PxVec3 worldPosA, worldPosB;
    physx::PxVec3 r;

    if (isBodyA) {
      r = body.rotation.rotate(contacts[c].contactPointA);
      worldPosA = body.position + r;
      worldPosB =
          otherBody ? otherBody->position +
                          otherBody->rotation.rotate(contacts[c].contactPointB)
                    : contacts[c].contactPointB;
    } else {
      r = body.rotation.rotate(contacts[c].contactPointB);
      worldPosA =
          otherBody ? otherBody->position +
                          otherBody->rotation.rotate(contacts[c].contactPointA)
                    : contacts[c].contactPointA;
      worldPosB = body.position + r;
    }

    const physx::PxVec3 &normal = contacts[c].contactNormal;
    physx::PxReal violation =
        (worldPosA - worldPosB).dot(normal) + contacts[c].penetrationDepth;

    violation -= mConfig.avbdAlpha * contacts[c].C0;
    if (hasDeformableStaticAnchor(contacts[c])) {
      violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
    }

    physx::PxReal pen = contacts[c].header.penalty;
    // Per-body primal boost: small fraction of M/h^2 safety net.
    pen = physx::PxMax(pen, contactBoostFloor);
    physx::PxReal lambda = contacts[c].header.lambda;

    physx::PxReal sign = isBodyA ? 1.0f : -1.0f;
    physx::PxVec3 rCrossN = r.cross(normal);
    physx::PxVec3 gradPos = normal * sign;
    physx::PxVec3 gradRot = rCrossN * sign;

    // Normal force (unilateral) + optional Coulomb-cone tangents in 6x6.
    const physx::PxReal rawForce =
        physx::PxMin(0.0f, pen * violation + lambda);
    physx::PxReal f = rawForce;
    bool forceSaturated = false;
    if (contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f) {
      const physx::PxReal maxNormalForce =
          physx::PxMax(contacts[c].maxImpulse, physx::PxReal(0.0f)) / dt;
      f = physx::PxMax(f, -maxNormalForce);
      forceSaturated = rawForce < -maxNormalForce;
    }
    // The derivative of a clamped force is zero while saturated.  Keeping the
    // contact penalty in the local Hessian here would enforce the unilateral
    // row even though its authored impulse budget has already been exhausted.
    if (!forceSaturated) {
      A.addResponseScaledConstraintContribution(
          gradPos, gradRot, pen, linearResponseScale, angularResponseScale);
    }
    numTouching++;

    if (f < 0.0f) {
      gLinear += gradPos * (f * linearResponseScale);
      gAngular += gradRot * (f * angularResponseScale);
    }

    // Ordinary rigid-static tangents keep their dedicated material owner.
    // The strict deformable/static probe instead consumes its position dual
    // through this same body-level AVBD primal block.
    if ((contacts[c].friction > 0.0f || contacts[c].staticFriction > 0.0f) &&
        hasPositionTangentParticipation(contacts[c])) {
      physx::PxVec3 prevWorldPosA, prevWorldPosB;
      if (isBodyA) {
        prevWorldPosA = body.prevPosition +
                        body.prevRotation.rotate(contacts[c].contactPointA);
        prevWorldPosB =
            otherBody
                ? otherBody->prevPosition +
                      otherBody->prevRotation.rotate(contacts[c].contactPointB)
                : contacts[c].contactPointB;
      } else {
        prevWorldPosA =
            otherBody
                ? otherBody->prevPosition +
                      otherBody->prevRotation.rotate(contacts[c].contactPointA)
                : contacts[c].contactPointA;
        prevWorldPosB = body.prevPosition +
                        body.prevRotation.rotate(contacts[c].contactPointB);
      }
      const bool bodyVsStatic =
          isBodyVsStaticContact(bodyAIdx, bodyBIdx, numBodies);
      const physx::PxVec3 relDisp =
          bodyVsStatic
              ? computeBodyVsStaticRelDisp(
                    worldPosA, prevWorldPosA, worldPosB, prevWorldPosB,
                    contacts[c], numBodies)
              : (worldPosA - prevWorldPosA) -
                    (worldPosB - prevWorldPosB);

      const physx::PxReal tPen0 =
          physx::PxMax(contacts[c].tangentPenalty0, contactBoostFloor);
      const physx::PxReal tPen1 =
          physx::PxMax(contacts[c].tangentPenalty1, contactBoostFloor);
      const physx::PxReal c0Scale = 1.0f - mConfig.avbdAlpha;
      const physx::PxReal tC0 =
          c0Scale * contacts[c].tangentC0 +
          relDisp.dot(contacts[c].tangent0);
      const physx::PxReal tC1 =
          c0Scale * contacts[c].tangentC1 +
          relDisp.dot(contacts[c].tangent1);
      const physx::PxReal mu = contactCoulombMu(contacts[c]);

      physx::PxReal Fn = 0.0f, Ft0 = 0.0f, Ft1 = 0.0f;
      (void)avbdEvaluateContactForcesCone(
          pen, violation, lambda, tPen0, tC0, contacts[c].tangentLambda0, tPen1,
          tC1, contacts[c].tangentLambda1, mu, Fn, Ft0, Ft1);
      if (contacts[c].maxImpulse < PX_MAX_REAL && dt > 0.0f) {
        const physx::PxReal maxNormalForce =
            physx::PxMax(contacts[c].maxImpulse, physx::PxReal(0.0f)) / dt;
        Fn = physx::PxMax(Fn, -maxNormalForce);
        avbdProjectImpulseCone(maxNormalForce * mu, Ft0, Ft1);
      }

      {
        const physx::PxVec3 &t = contacts[c].tangent0;
        const physx::PxVec3 rCrossT = r.cross(t);
        const physx::PxVec3 tGradPos = t * sign;
        const physx::PxVec3 tGradRot = rCrossT * sign;
        A.addResponseScaledConstraintContribution(
            tGradPos, tGradRot, tPen0, linearResponseScale,
            angularResponseScale);
        gLinear += tGradPos * (Ft0 * linearResponseScale);
        gAngular += tGradRot * (Ft0 * angularResponseScale);
      }
      {
        const physx::PxVec3 &t = contacts[c].tangent1;
        const physx::PxVec3 rCrossT = r.cross(t);
        const physx::PxVec3 tGradPos = t * sign;
        const physx::PxVec3 tGradRot = rCrossT * sign;
        A.addResponseScaledConstraintContribution(
            tGradPos, tGradRot, tPen1, linearResponseScale,
            angularResponseScale);
        gLinear += tGradPos * (Ft1 * linearResponseScale);
        gAngular += tGradRot * (Ft1 * angularResponseScale);
      }
    }
  }

  if (softContacts && numSoftContacts > 0 && softParticles &&
      numSoftParticles > 0) {
    const physx::PxReal shellBoostFloor =
        AvbdConstants::AVBD_PEN_SCALE_BODY_VS_STATIC * massInvDt2;
    AvbdVec6 softContactRhs;
    softContactRhs.linear = physx::PxVec3(0.0f);
    softContactRhs.angular = physx::PxVec3(0.0f);
    const auto accumulateSoftContact =
        [&](physx::PxU32 sci) {
      const AvbdSoftContact &sc = softContacts[sci];
      const AvbdSoftContactGeometry &geometry = sc.geometry;
      const AvbdSoftContactAugmentedState &state = sc.state;
      if (!geometry.hasRigidBodyTarget() ||
          geometry.targetIndex != bodyIndex)
        return;
      if (avbdIsSoftContactQueryFullyKinematic(
              geometry, softParticles, numSoftParticles)) {
        avbdAddKinematicShellContactContribution_rigid(
            geometry, state, bodyIndex, body,
            shellBoostFloor, A, softContactRhs);
        numTouching++;
      } else if (
          avbdAddDynamicSoftRigidContactContribution_rigid(
              geometry, state, bodyIndex, softParticles,
              numSoftParticles, body, A, softContactRhs)) {
        numTouching++;
      }
    };
    if (useRigidTargetContactCsr) {
      for (physx::PxU32 refIndex =
               rigidTargetContactStarts[bodyIndex];
           refIndex < rigidTargetContactStarts[bodyIndex + 1];
           ++refIndex)
        accumulateSoftContact(rigidTargetContactRefs[refIndex]);
    } else {
      for (physx::PxU32 sci = 0; sci < numSoftContacts; ++sci)
        accumulateSoftContact(sci);
    }
    gLinear += softContactRhs.linear;
    gAngular += softContactRhs.angular;
  }

}

void AvbdSolver::solveLocalSystem(AvbdSolverBody &body, AvbdSolverBody *bodies,
                                  physx::PxU32 numBodies,
                                  AvbdContactConstraint *contacts,
                                  physx::PxU32 numContacts, physx::PxReal dt,
                                  physx::PxReal invDt2,
                                  const AvbdBodyConstraintMap *contactMap) {

  // Skip static bodies
  if (body.invMass <= 0.0f) {
    return;
  }
  PX_UNUSED(dt);

  const physx::PxU32 bodyIndex = body.nodeIndex;

  // =========================================================================
  // Step 1: Initialize LHS with mass matrix M/h^2
  // =========================================================================

  AvbdBlock6x6 A;
  A.initializeDiagonal(body.invMass, body.invInertiaWorld, invDt2);

  // =========================================================================
  // Step 2: Initialize RHS with inertia term
  //   rhs = (M/h^2) * vec6{x - x_inertial, deltaW_inertial}
  // =========================================================================

  physx::PxReal mass = (body.invMass > 1e-8f) ? (1.0f / body.invMass) : 0.0f;
  physx::PxReal massInvDt2 = mass * invDt2;

  physx::PxVec3 gLinear = (body.position - body.inertialPosition) * massInvDt2;

  // Angular inertia RHS: (I/h^2) * deltaW_inertial
  physx::PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
  if (deltaQ.w < 0.0f) {
    deltaQ = -deltaQ;
  }
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  physx::PxMat33 inertiaTensor = body.invInertiaWorld.getInverse();
  physx::PxVec3 gAngular = (inertiaTensor * rotError) * invDt2;

  // =========================================================================
  // Step 3: Shared rigid-contact primal accumulation (body-static contract)
  // =========================================================================

  physx::PxU32 numTouching = 0;
  accumulateBodyContactRows(
      body, bodyIndex, bodies, numBodies, contacts, numContacts, contactMap,
      nullptr, 0, nullptr, 0, dt, massInvDt2, A, gLinear, gAngular,
      numTouching);

  // No contacts: snap to inertial target
  if (numTouching == 0) {
    body.position = body.inertialPosition;
    body.rotation = body.inertialRotation;
    return;
  }

  // =========================================================================
  // Step 4: Solve A * delta = rhs via LDLT
  // =========================================================================

  AvbdLDLT ldlt;
  AvbdVec6 rhs(gLinear, gAngular);

  physx::PxVec3 deltaPos;
  physx::PxVec3 deltaTheta;

  if (ldlt.decomposeWithRegularization(A)) {
    AvbdVec6 delta = ldlt.solve(rhs);
    deltaPos = delta.linear;
    deltaTheta = delta.angular;
  } else {
    deltaPos = physx::PxVec3(0.0f);
    deltaTheta = physx::PxVec3(0.0f);
  }

  // =========================================================================
  // Step 5: Apply update  x -= delta
  //   (ref: solver.cpp L137-138)
  // =========================================================================

  body.position -= deltaPos;

  if (deltaTheta.magnitudeSquared() > 1e-12f) {
    physx::PxQuat dq(deltaTheta.x, deltaTheta.y, deltaTheta.z, 0.0f);
    body.rotation = (body.rotation - dq * body.rotation * 0.5f).getNormalized();
  }
}

void AvbdSolver::solveRigidBodyRange(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    physx::PxReal dt, physx::PxReal invDt2,
    const AvbdBodyConstraintMap *contactMap, const physx::PxU32 *bodyOrder,
    physx::PxU32 begin, physx::PxU32 end) {
  PX_PROFILE_ZONE("AVBD.solveRigidBodyRange", 0);
  PX_ASSERT(begin <= end && end <= numBodies);
  for (physx::PxU32 idx = begin; idx < end; ++idx) {
    const physx::PxU32 i = bodyOrder ? bodyOrder[idx] : idx;
    if (bodies[i].invMass <= 0.0f)
      continue;
    if (mConfig.enableLocal6x6Solve) {
      solveLocalSystem(bodies[i], bodies, numBodies, contacts, numContacts, dt,
                       invDt2, contactMap);
    } else {
      solveLocalSystemWithJoints(bodies[i], bodies, numBodies, contacts,
                                 numContacts, nullptr, 0, nullptr, 0, dt,
                                 invDt2, contactMap, nullptr, nullptr);
    }
  }
}

bool AvbdSolver::solveRigidOwnerFallback(
    AvbdRigidSolveContext &context, const physx::PxU32 *ownerBodyOrder,
    physx::PxU32 lane) {
  if (!ownerBodyOrder || !context.iteration.bodies ||
      !context.iteration.contacts || !context.iteration.contactMap ||
      lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH)
    return false;
  solveRigidBodyRange(
      context.iteration.bodies, context.iteration.numBodies,
      context.iteration.contacts, context.iteration.numContacts,
      context.iteration.dt, context.invDt2, context.iteration.contactMap,
      ownerBodyOrder, lane, lane + 1u);
  return true;
}

void AvbdSolver::blockDescentIteration(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts, physx::PxReal dt,
    const AvbdBodyConstraintMap *contactMap, AvbdColorBatch *colorBatches,
    physx::PxU32 numColors) {

  PX_UNUSED(colorBatches);
  PX_UNUSED(numColors);

  // True Block Coordinate Descent: iterate over bodies, not constraints
  // For each body, solve a local optimization problem considering all
  // constraints that affect this body.
  //
  // Parallelization uses a read-only pose snapshot for every local solve and
  // writes each result to a distinct output body.  Reading the live body array
  // here would be asynchronous Gauss-Seidel with unsynchronized neighbor
  // reads, not Jacobi, and makes both results and scale invariance depend on
  // task scheduling.

  const bool useDeterministicOrder =
      mConfig.isDeterministic() &&
      (mConfig.determinismFlags & AvbdDeterminismFlags::eSORT_BODIES);

  physx::PxArray<physx::PxU32> bodyOrder;
  if (useDeterministicOrder) {
    bodyOrder.resize(numBodies);
    for (physx::PxU32 i = 0; i < numBodies; ++i)
      bodyOrder[i] = i;
    std::sort(bodyOrder.begin(), bodyOrder.end(),
              [&bodies](physx::PxU32 a, physx::PxU32 b) {
                if (bodies[a].invMass != bodies[b].invMass)
                  return bodies[a].invMass > bodies[b].invMass;
                return a < b;
              });
  }

  const physx::PxReal invDt2 = 1.0f / (dt * dt);
  const physx::PxU32 *orderPtr =
      useDeterministicOrder ? bodyOrder.begin() : nullptr;

  // P2 removes the AVBD-private worker path.  A non-conflicting colored body
  // stage will be submitted through the Scene taskgraph in P4; until then
  // retain the authoritative Gauss-Seidel body order rather than silently
  // changing the solve to an unscheduled Jacobi variant.
  solveRigidBodyRange(bodies, numBodies, contacts, numContacts, dt, invDt2,
                      contactMap, orderPtr, 0, numBodies);
}

} // namespace Dy
} // namespace physx
