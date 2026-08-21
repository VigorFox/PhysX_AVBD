// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/ogc/DyAvbdOgcAdmission.h"
#include "avbd/solver/joint/DyAvbdJointCoupledMath.h"
#include "avbd/solver/joint/DyAvbdJointDriveMath.h"
#include "avbd/solver/joint/DyAvbdJointGeometryPolicy.h"
#include "avbd/solver/joint/DyAvbdJointProjection.h"
#include "PxConstraintDesc.h"

namespace physx {
namespace Dy {

void AvbdSolver::solveLocalSystemWithJoints(
    AvbdSolverBody &body, AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear, physx::PxReal dt,
    physx::PxReal invDt2, const AvbdBodyConstraintMap *contactMap,
    const AvbdBodyConstraintMap *d6Map, const AvbdBodyConstraintMap *gearMap,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const physx::PxU32 *rigidTargetContactStarts,
    const physx::PxU32 *rigidTargetContactRefs,
    const AvbdOgcPairTrustRegionContext *ogcPairContext) {

  if (body.invMass <= 0.0f)
    return;

  PX_UNUSED(softBodies);
  PX_UNUSED(numSoftBodies);
  PX_UNUSED(ogcPairContext);

  const physx::PxU32 bodyIndex = body.nodeIndex;

  const physx::PxReal scaledInvMass = body.invMass;
  const physx::PxMat33 scaledInvInertia = body.invInertiaWorld;

  // =========================================================================
  // Step 1: Initialize LHS with mass matrix M/h^2
  // =========================================================================
  AvbdBlock6x6 A;
  A.initializeDiagonal(scaledInvMass, scaledInvInertia, invDt2);

  // =========================================================================
  // Step 2: Initialize RHS with inertia term
  // =========================================================================
  physx::PxReal mass =
      (scaledInvMass > 1e-8f) ? (1.0f / scaledInvMass) : 0.0f;
  physx::PxReal massInvDt2 = mass * invDt2;

  physx::PxVec3 gLinear = (body.position - body.inertialPosition) * massInvDt2;

  physx::PxQuat deltaQ = body.rotation * body.inertialRotation.getConjugate();
  if (deltaQ.w < 0.0f)
    deltaQ = -deltaQ;
  physx::PxVec3 rotError(deltaQ.x, deltaQ.y, deltaQ.z);
  rotError *= 2.0f;
  physx::PxMat33 inertiaTensor = scaledInvInertia.getInverse();
  physx::PxVec3 gAngular = (inertiaTensor * rotError) * invDt2;

  physx::PxU32 numTouching = 0;
  bool hasLinearCoupling =
      false; // Force 6x6 solve for bodies touching joints with pos-rot coupling

  // =========================================================================
  // Step 3a/3b: Shared contact + shell primal (same body-static contract as
  // solveLocalSystem). Joint / gear / soft rows accumulate after this.
  // =========================================================================
  accumulateBodyContactRows(
      body, bodyIndex, bodies, numBodies, contacts, numContacts, contactMap,
      softParticles, numSoftParticles, softContacts, numSoftContacts,
      dt, massInvDt2, A, gLinear, gAngular, numTouching,
      rigidTargetContactStarts, rigidTargetContactRefs);

  // Rigid-vertex attachments are not accumulated into this one-body block.
  // Their compiled owner is the coupled rigid+particle positional block run
  // after the ordinary rigid and soft local objectives.

  // Step 3e: Accumulate D6 JOINT contributions

  //
  //   Locked linear DOFs: 3 position rows (same as spherical)
  //   Angular velocity damping (SLERP/axis drives): adds damping_eff to
  //     the angular diagonal of the Hessian, penalizing deviation from
  //     inertial rotation (which encodes current angular velocity).
  //   Locked angular DOFs: TODO (not used by SnippetJoint D6 config)
  // =========================================================================
  if (d6Joints && numD6 > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (d6Map && d6Map->numBodies > 0)
      d6Map->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numD6;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numD6)
        continue;
      const AvbdD6JointConstraint &jnt = d6Joints[j];
      const physx::PxU32 bodyAIdx = jnt.header.bodyIndexA;
      const physx::PxU32 bodyBIdx = jnt.header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;

      const bool isBodyA = (bodyAIdx == bodyIndex);
      const physx::PxU32 compiledSourceRows =
          getAvbdJointObjectiveSourceRows(jnt.objectiveProgram);
      const physx::PxU32 compiledDriveRows =
          compiledSourceRows & 0x3fu;
      const bool compiledConeObjective =
          (compiledSourceRows &
           eJOINT_SOURCE_ANGULAR_CONE) != 0;

      const bool genericHard1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::GenericHard1D) ||
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::ArticulationHardMimic);
      const bool genericAccelerationDamping1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::
                  GenericAccelerationDamping1D);
      const bool genericForceSpring1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::GenericForceSpring1D);
      const bool compliantMimic1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::
                  ArticulationCompliantMimic);
      const bool fixedTendon1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::ArticulationFixedTendon);
      const bool spatialTendon1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::ArticulationSpatialTendon);
      const bool coupledSpatialTendon1D =
          hasAvbdJointObjective(
              jnt.objectiveProgram,
              AvbdJointObjectiveKind::CoupledSpatialTendon);
      const bool compliantTendon1D =
          fixedTendon1D || spatialTendon1D;
      if (coupledSpatialTendon1D)
        continue;
      if (genericHard1D || genericAccelerationDamping1D ||
          genericForceSpring1D || compliantMimic1D ||
          compliantTendon1D) {
        const physx::PxReal effectiveMass =
            computeGeneric1DEffectiveMass(jnt, bodies, numBodies);
        if (effectiveMass <= 0.0f || dt <= 0.0f)
          continue;

        const physx::PxReal C =
            computeGeneric1DViolation(jnt, bodies, numBodies, dt);
        physx::PxReal pen = 0.0f;
        physx::PxReal unclampedForce = 0.0f;
        if (genericAccelerationDamping1D) {
          // For an acceleration spring, PhysX's implicit velocity update is
          // mass independent: dv = -v*(d*dt)/(1+d*dt).  Expressing that in
          // AVBD's position Hessian requires scaling the physical damping by
          // the row effective mass.
          pen = effectiveMass * jnt.header.damping / dt;
          unclampedForce = pen * C;
        } else if (genericForceSpring1D || compliantMimic1D ||
                   compliantTendon1D) {
          // Backward-Euler linearization of k*C + d*Cdot with respect to
          // the current position iterate.  genericGeometricError is C at the
          // start-of-step reference pose, so this uses the displacement
          // produced by this step rather than mixing the previous interval's
          // body velocity with a current-position damping Hessian.  A changed
          // public tendon offset is deliberately part of both values: the
          // reduced-coordinate tendon path damps joint motion, not an
          // externally authored offset velocity.
          physx::PxReal stiffness = jnt.header.rho;
          physx::PxReal damping = jnt.header.damping;
          if (compliantMimic1D) {
            stiffness = jnt.genericNaturalFrequency *
                        jnt.genericNaturalFrequency * effectiveMass;
            damping = 2.0f * jnt.genericNaturalFrequency *
                      jnt.genericDampingRatio * effectiveMass;
          }
          const physx::PxReal velocity =
              (C - jnt.genericGeometricError) / dt;
          pen = stiffness + damping / dt;
          unclampedForce = stiffness * C + damping * velocity;
          if (compliantTendon1D &&
              jnt.genericTendonLimitStiffness > 0.0f) {
            physx::PxReal limitViolation = 0.0f;
            if (C < jnt.genericTendonLowLimit)
              limitViolation = C - jnt.genericTendonLowLimit;
            else if (C > jnt.genericTendonHighLimit)
              limitViolation = C - jnt.genericTendonHighLimit;
            if (limitViolation != 0.0f) {
              // Tendon limit springs are additive to the rest-length spring.
              // PhysX uses the authored damping in the limit spring's implicit
              // response, but computeTendonImpulse applies the tendon-speed
              // damping force only to the rest-length spring.
              pen +=
                  jnt.genericTendonLimitStiffness + damping / dt;
              unclampedForce +=
                  jnt.genericTendonLimitStiffness * limitViolation;
            }
          }
        } else {
          pen = physx::PxMax(jnt.header.rho, effectiveMass * invDt2);
          unclampedForce = pen * C + jnt.lambdaLinear.x;
        }
        const physx::PxReal appliedImpulse = physx::PxClamp(
            -unclampedForce * dt, jnt.genericMinImpulse,
            jnt.genericMaxImpulse);
        const bool bilateral =
            jnt.genericMinImpulse < 0.0f && jnt.genericMaxImpulse > 0.0f;
        if (!bilateral && physx::PxAbs(appliedImpulse) <= 1e-12f)
          continue;

        const physx::PxReal force = -appliedImpulse / dt;
        const physx::PxVec3 &linearJacobian =
            isBodyA ? jnt.genericLinearA : jnt.genericLinearB;
        const physx::PxVec3 &angularJacobian =
            isBodyA ? jnt.genericAngularA : jnt.genericAngularB;
        A.addConstraintContribution(linearJacobian, angularJacobian, pen);
        gLinear += linearJacobian * force;
        gAngular += angularJacobian * force;
        hasLinearCoupling |=
            linearJacobian.magnitudeSquared() > 1e-12f &&
            angularJacobian.magnitudeSquared() > 1e-12f;
        ++numTouching;
        continue;
      }

      const bool otherIsStatic =
          isBodyA ? (bodyBIdx == 0xFFFFFFFF || bodyBIdx >= numBodies)
                  : (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);

      physx::PxReal mA =
          (bodyAIdx < numBodies && bodies[bodyAIdx].invMass > 1e-8f)
              ? (1.0f / bodies[bodyAIdx].invMass)
              : 0.0f;
      physx::PxReal mB =
          (bodyBIdx < numBodies && bodies[bodyBIdx].invMass > 1e-8f)
              ? (1.0f / bodies[bodyBIdx].invMass)
              : 0.0f;
      physx::PxReal mEff = physx::PxMax(mA, mB);

      // Auto-boost penalty using symmetric effective mass
      physx::PxReal pen = physx::PxMax(jnt.header.rho, mEff * invDt2);
      physx::PxReal signJ = isBodyA ? 1.0f : -1.0f;

      // Lever arm from body COM to constraint anchor (used by linear DOFs
      // AND linear drive).  Computed once and reused.
      physx::PxVec3 rArm(0.0f);

      // --- Linear DOFs (LOCKED / LIMITED / FREE) ---
      // Axis selection matches avbd_standalone:
      //   All-LOCKED => world axes (well-conditioned Hessian)
      //   Otherwise  => joint-local axes from localFrameA
      {
        physx::PxVec3 worldAnchorA, worldAnchorB;
        physx::PxVec3 r;
        if (isBodyA) {
          r = body.rotation.rotate(jnt.anchorA);
          worldAnchorA = body.position + r;
          worldAnchorB =
              otherIsStatic ? jnt.anchorB
                            : bodies[bodyBIdx].position +
                                  bodies[bodyBIdx].rotation.rotate(jnt.anchorB);
        } else {
          r = body.rotation.rotate(jnt.anchorB);
          worldAnchorB = body.position + r;
          worldAnchorA =
              otherIsStatic ? jnt.anchorA
                            : bodies[bodyAIdx].position +
                                  bodies[bodyAIdx].rotation.rotate(jnt.anchorA);
        }

        rArm = r;  // export to outer scope for drive
        physx::PxVec3 posError = worldAnchorA - worldAnchorB;

        // Compute joint-frame axes in world space
        const bool bodyAIsStatic =
            (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);
        physx::PxQuat rotA_lin =
            bodyAIsStatic
                ? physx::PxQuat(physx::PxIdentity)
                : (isBodyA ? body.rotation : bodies[bodyAIdx].rotation);
        physx::PxQuat jointFrameA_lin =
            bodyAIsStatic ? jnt.localFrameA : rotA_lin * jnt.localFrameA;
        {
          physx::PxReal qm2 = jointFrameA_lin.magnitudeSquared();
          if (qm2 > 1e-8f && PxIsFinite(qm2))
            jointFrameA_lin *= 1.0f / physx::PxSqrt(qm2);
        }

        const bool linAllLocked = (jnt.linearMotion == 0);
        physx::PxVec3 linearAxes[3];
        if (linAllLocked) {
          linearAxes[0] = physx::PxVec3(1.0f, 0.0f, 0.0f);
          linearAxes[1] = physx::PxVec3(0.0f, 1.0f, 0.0f);
          linearAxes[2] = physx::PxVec3(0.0f, 0.0f, 1.0f);
        } else {
          linearAxes[0] = jointFrameA_lin.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          linearAxes[1] = jointFrameA_lin.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f));
          linearAxes[2] = jointFrameA_lin.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f));
        }

        for (int axis = 0; axis < 3; ++axis) {
          physx::PxU32 motion = jnt.getLinearMotion(axis);
          if (motion == 2) // FREE
            continue;

          const physx::PxVec3 &n = linearAxes[axis];
          physx::PxReal C = posError.dot(n);

          physx::PxVec3 rCrossN = r.cross(n);
          physx::PxVec3 gradPos = n * signJ;
          physx::PxVec3 gradRot = rCrossN * signJ;

          if (motion == 0) { // LOCKED
            A.addConstraintContribution(gradPos, gradRot, pen);

            physx::PxReal f = pen * C + jnt.lambdaLinear[axis];
            gLinear += gradPos * f;
            gAngular += gradRot * f;
          } else if (motion == 1) { // LIMITED
            // Baseline Hessian stiffness (prevents drift in free range)
            A.addConstraintContribution(gradPos, gradRot, pen);

            physx::PxReal dist = -posError.dot(n);
            physx::PxReal limitViolation = 0.0f;
            if (dist < jnt.linearLimitLower[axis])
              limitViolation = dist - jnt.linearLimitLower[axis];
            else if (dist > jnt.linearLimitUpper[axis])
              limitViolation = dist - jnt.linearLimitUpper[axis];

            if (physx::PxAbs(limitViolation) > 0.0f) {
              physx::PxReal f = pen * limitViolation + jnt.lambdaLinear[axis];
              physx::PxReal forceMag = 0.0f;

              if (jnt.linearLimitLower[axis] < jnt.linearLimitUpper[axis]) {
                if (limitViolation > 0.0f || jnt.lambdaLinear[axis] > 0.0f) {
                  forceMag = physx::PxMax(0.0f, f);
                } else if (limitViolation < 0.0f ||
                           jnt.lambdaLinear[axis] < 0.0f) {
                  forceMag = physx::PxMin(0.0f, f);
                }
              } else {
                forceMag = f;
              }

              if (physx::PxAbs(forceMag) > 0.0f) {
                // Limit Jacobian direction: use negative axis (gradient of
                // dist)
                physx::PxVec3 nLim = n * (-1.0f);
                physx::PxVec3 gradPosLim = nLim * signJ;
                physx::PxVec3 gradRotLim = r.cross(nLim) * signJ;
                A.addConstraintContribution(gradPosLim, gradRotLim, pen);
                gLinear += gradPosLim * forceMag;
                gAngular += gradRotLim * forceMag;
              }
            }
          }
        } // End of Linear DOFs for loop
      } // End of Linear DOFs scope

      // --- Angular DOFs (LOCKED and LIMITED) ---
      {
        physx::PxQuat rotA, rotB;
        if (isBodyA) {
          rotA = body.rotation;
          rotB = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                               : bodies[bodyBIdx].rotation;
        } else {
          rotA = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                               : bodies[bodyAIdx].rotation;
          rotB = body.rotation;
        }

        // Detect revolute pattern: twist(X) FREE or LIMITED, swing(Y,Z) LOCKED
        const physx::PxU32 twistMotion = jnt.getAngularMotion(0);
        const physx::PxU32 swing1Motion = jnt.getAngularMotion(1);
        const physx::PxU32 swing2Motion = jnt.getAngularMotion(2);
        const bool isRevolutePattern =
            (twistMotion != 0) && (swing1Motion == 0) && (swing2Motion == 0);

        if (isRevolutePattern) {
          // Cross-product axis alignment (2 rows) - matches reference revolute
          // solver. Unlike computeAngularError decomposition, this is immune to
          // large twist angles amplifying swing drift.
          physx::PxQuat worldFrameA = rotA * jnt.localFrameA;
          physx::PxQuat worldFrameB = rotB * jnt.localFrameB;
          physx::PxVec3 worldTwistA =
              worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          physx::PxVec3 worldTwistB =
              worldFrameB.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          physx::PxVec3 axisViolation = worldTwistA.cross(worldTwistB);

          // Build perpendicular basis from worldTwistA
          physx::PxVec3 perp1, perp2;
          if (physx::PxAbs(worldTwistA.x) < 0.9f)
            perp1 = worldTwistA.cross(physx::PxVec3(1.0f, 0.0f, 0.0f));
          else
            perp1 = worldTwistA.cross(physx::PxVec3(0.0f, 1.0f, 0.0f));
          physx::PxReal perp1Len = perp1.magnitude();
          if (perp1Len > 1e-6f)
            perp1 *= (1.0f / perp1Len);
          perp2 = worldTwistA.cross(perp1);
          physx::PxReal perp2Len = perp2.magnitude();
          if (perp2Len > 1e-6f)
            perp2 *= (1.0f / perp2Len);

          physx::PxReal err1 = axisViolation.dot(perp1);
          physx::PxReal err2 = axisViolation.dot(perp2);

          // Row 1 (stored in lambdaAngular[1])
          {
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = -perp1 * signJ;
            A.addConstraintContribution(gradPos, gradRot, pen);
            physx::PxReal f = pen * err1 + jnt.lambdaAngular[1];
            gAngular += gradRot * f;
          }
          // Row 2 (stored in lambdaAngular[2])
          {
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = -perp2 * signJ;
            A.addConstraintContribution(gradPos, gradRot, pen);
            physx::PxReal f = pen * err2 + jnt.lambdaAngular[2];
            gAngular += gradRot * f;
          }

          // Handle twist axis (0) if LIMITED
          if (twistMotion == 1) {
            physx::PxVec3 worldAxis =
                worldFrameA.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = worldAxis * signJ;

            physx::PxReal error =
                jnt.computeAngularError(rotA, rotB, 0);
            physx::PxReal limitViolation =
                jnt.computeAngularLimitViolation(error, 0);
            physx::PxReal f =
                pen * limitViolation + jnt.lambdaAngular[0];
            physx::PxReal forceMag = 0.0f;

            if (jnt.angularLimitLower[0] < jnt.angularLimitUpper[0]) {
              if (limitViolation > 0.0f || jnt.lambdaAngular[0] > 0.0f)
                forceMag = physx::PxMax(0.0f, f);
              else if (limitViolation < 0.0f || jnt.lambdaAngular[0] < 0.0f)
                forceMag = physx::PxMin(0.0f, f);
            } else {
              forceMag = f;
            }

            if (physx::PxAbs(forceMag) > 0.0f) {
              A.addConstraintContribution(gradPos, gradRot, pen);
              gAngular += gradRot * forceMag;
            }
          }
        } else {
          // Generic per-axis angular constraint handling
          for (int axis = 0; axis < 3; ++axis) {
            if (compiledConeObjective && axis >= 1)
              continue;
            physx::PxU32 motion = jnt.getAngularMotion(axis);
            if (motion == 2) // FREE
              continue;

            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[axis] = 1.0f;
            physx::PxQuat worldFrameA = rotA * jnt.localFrameA;
            physx::PxVec3 worldAxis = worldFrameA.rotate(localAxis);

            physx::PxVec3 gradPos(0.0f);
            physx::PxVec3 gradRot = worldAxis * signJ;

            if (motion == 0) { // LOCKED
              physx::PxReal C = jnt.computeAngularError(rotA, rotB, axis);
              A.addConstraintContribution(gradPos, gradRot, pen);

              physx::PxReal f = pen * C + jnt.lambdaAngular[axis];
              gAngular += gradRot * f;
            } else if (motion == 1) { // LIMITED
              physx::PxReal error =
                  jnt.computeAngularError(rotA, rotB, axis);
              physx::PxReal limitViolation =
                  jnt.computeAngularLimitViolation(error, axis);
              physx::PxReal f =
                  pen * limitViolation + jnt.lambdaAngular[axis];
              physx::PxReal forceMag = 0.0f;

              if (jnt.angularLimitLower[axis] < jnt.angularLimitUpper[axis]) {
                if (limitViolation > 0.0f || jnt.lambdaAngular[axis] > 0.0f) {
                  forceMag = physx::PxMax(0.0f, f);
                } else if (limitViolation < 0.0f ||
                           jnt.lambdaAngular[axis] < 0.0f) {
                  forceMag = physx::PxMin(0.0f, f);
                }
              } else {
                forceMag = f;
              }

              if (physx::PxAbs(forceMag) > 0.0f) {
                A.addConstraintContribution(gradPos, gradRot, pen);
                gAngular += gradRot * forceMag;
              }
            }
          }
        }
      }

      // --- Cone limit (single angular inequality) ---
      // Public D6 legacy swing limits and native spherical limits both use
      // ConeLimitHelperTanLess so unequal Y/Z limits preserve the same
      // elliptical geometry as their Extensions solver preps.
      if (compiledConeObjective) {
        physx::PxQuat rotA_cone, rotB_cone;
        if (isBodyA) {
          rotA_cone = body.rotation;
          rotB_cone = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                                    : bodies[bodyBIdx].rotation;
        } else {
          rotA_cone = otherIsStatic ? physx::PxQuat(physx::PxIdentity)
                                    : bodies[bodyAIdx].rotation;
          rotB_cone = body.rotation;
        }

        physx::PxVec3 corrAxis(0.0f);
        physx::PxReal coneViolation = 0.0f;
        const bool ellipticalCone = computeEllipticalConeConstraint(
            jnt, rotA_cone, rotB_cone, corrAxis, coneViolation);
        if (!ellipticalCone) {
          const physx::PxVec3 worldAxisA =
              (rotA_cone * jnt.localFrameA)
                  .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          const physx::PxVec3 worldAxisB =
              (rotB_cone * jnt.localFrameB)
                  .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
          const physx::PxReal dotAB = physx::PxClamp(
              worldAxisA.dot(worldAxisB), -1.0f, 1.0f);
          const physx::PxReal coneAngle = physx::PxAcos(dotAB);
          coneViolation = coneAngle - jnt.coneAngleLimit;
          corrAxis = worldAxisA.cross(worldAxisB);
          const physx::PxReal corrAxisMag = corrAxis.magnitude();
          if (corrAxisMag > 1e-6f)
            corrAxis *= 1.0f / corrAxisMag;
          else
            corrAxis = physx::PxVec3(0.0f);
        }

        // coneLambda <= 0 (unilateral): force = pen * violation - coneLambda
        physx::PxReal forceMag = pen * coneViolation - jnt.coneLambda;

        if (forceMag > 0.0f && corrAxis.magnitudeSquared() > 1e-12f) {
          physx::PxVec3 gradPos(0.0f);
          physx::PxVec3 gradRot = -corrAxis * signJ;

          A.addConstraintContribution(gradPos, gradRot, pen);
          gAngular += gradRot * forceMag;
        }
      }

      // --- Pure AVBD AL velocity drive constraints ---
      // Replaces ad-hoc damping. Each driven axis contributes an AL
      // velocity constraint:
      //   C = (-x_B - -x_A) - axis - v_target - dt   (linear)
      //   C = (--_B - --_A) - axis - -_target - dt   (angular)
      // Hessian: -_drive - (axis - axis)
      // RHS:     sign - (-_drive - C + -)
      {
        // Joint frame A in world space
        physx::PxQuat jointFrameA =
            otherIsStatic && !isBodyA
                ? jnt.localFrameA
                : (isBodyA ? body.rotation * jnt.localFrameA
                           : (otherIsStatic ? jnt.localFrameA
                                            : bodies[bodyAIdx].rotation *
                                                  jnt.localFrameA));

        physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
        if (qMag2 > 1e-8f && PxIsFinite(qMag2))
          jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

        physx::PxReal dt2 = dt * dt;

        // Get "other body" for relative displacement
        const AvbdSolverBody *otherBody = nullptr;
        const AvbdSolverBody *bodyARef = nullptr;
        const AvbdSolverBody *bodyBRef = nullptr;
        if (isBodyA && bodyBIdx < numBodies)
          otherBody = &bodies[bodyBIdx];
        else if (!isBodyA && bodyAIdx < numBodies)
          otherBody = &bodies[bodyAIdx];
        bodyARef = (bodyAIdx < numBodies) ? &bodies[bodyAIdx] : nullptr;
        bodyBRef = (bodyBIdx < numBodies) ? &bodies[bodyBIdx] : nullptr;

        // --- Linear velocity drive (AL constraint) ---
        if ((compiledDriveRows & 0x7u) != 0) {
          for (int a = 0; a < 3; ++a) {
            if ((compiledDriveRows & (1u << a)) == 0)
              continue;
            physx::PxReal damping = (&jnt.linearDamping.x)[a];
            if (damping <= 0.0f)
              continue;

            // World-space axis
            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[a] = 1.0f;
            physx::PxVec3 wAxis = jointFrameA.rotate(localAxis);

            // Displacement of each body from start-of-step
            physx::PxVec3 dxThis = body.position - body.prevPosition;
            physx::PxVec3 dxOther =
                otherBody ? (otherBody->position - otherBody->prevPosition)
                          : physx::PxVec3(0.0f);

            // Constraint: C = (dx_B - dx_A) dot axis - v_target * dt
            physx::PxReal dxB_proj, dxA_proj;
            if (isBodyA) {
              dxA_proj = dxThis.dot(wAxis);
              dxB_proj = dxOther.dot(wAxis);
            } else {
              dxB_proj = dxThis.dot(wAxis);
              dxA_proj = dxOther.dot(wAxis);
            }
            physx::PxReal targetVel = (&jnt.driveLinearVelocity.x)[a];
            physx::PxReal C = (dxB_proj - dxA_proj) - targetVel * dt;

            const physx::PxVec3 rAWorld = bodyARef
              ? bodyARef->rotation.rotate(jnt.anchorA)
              : physx::PxVec3(0.0f);
            const physx::PxVec3 rBWorld = bodyBRef
              ? bodyBRef->rotation.rotate(jnt.anchorB)
              : physx::PxVec3(0.0f);
            const bool isAccelerationDrive =
                jnt.isLinearAccelerationDrive(a);
            const physx::PxReal stiffness = (&jnt.linearStiffness.x)[a];
            const bool usePhysicalVelocityObjective =
                stiffness <= 0.0f && (bodyARef == nullptr || bodyBRef == nullptr);
            const bool usePositionObjective =
                a == 0 &&
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::LinearPositionDrive);
            // In the verified exactly-one-dynamic, stiffness-zero subset,
            // PxD6JointDrive::damping is a force-per-velocity coefficient.
            // With C = (dxB-dxA)-targetVelocity*dt, its position objective
            // therefore uses damping/dt.  Wider dynamic-dynamic and spring
            // families retain the legacy AL path until they have independent
            // mixed-island and articulation gates.
            const physx::PxReal positionError =
                ((bodyBRef ? bodyBRef->position + rBWorld : jnt.anchorB) -
                 (bodyARef ? bodyARef->position + rAWorld : jnt.anchorA))
                    .dot(wAxis) -
                (&jnt.driveLinearPosition.x)[a];
            physx::PxReal rho_drive =
                usePositionObjective
                    ? stiffness + damping / dt
                    : (usePhysicalVelocityObjective ? damping / dt
                                                    : damping / dt2);
            if (isAccelerationDrive && usePhysicalVelocityObjective) {
              const physx::PxReal driveScale =
                computeLinearDriveRecipResponse(bodyARef, bodyBRef,
                               rAWorld, rBWorld, wAxis);
              rho_drive *= driveScale;
            } else if (isAccelerationDrive) {
              const physx::PxReal driveScale =
                computeLinearDriveRecipResponse(bodyARef, bodyBRef,
                               rAWorld, rBWorld, wAxis);
              const physx::PxReal dampingOnly =
                  physx::PxMax(0.0f, damping - stiffness);
              const physx::PxReal implicitScale =
                  1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
              rho_drive *= driveScale * implicitScale;
            }
            const physx::PxReal authoredLimit =
                (&jnt.driveLinearForce.x)[a];
            const bool limitsAreForces =
                (jnt.sourceFlags & AvbdD6JointConstraint::
                                       eD6_DRIVE_LIMITS_ARE_FORCES) != 0;
            const physx::PxReal maxForce = limitsAreForces
                ? authoredLimit
                : physx::PxMin(PX_MAX_F32, authoredLimit / dt);
            const bool usePhysicalObjective =
                usePhysicalVelocityObjective || usePositionObjective;
            const physx::PxReal lambda = usePhysicalObjective
                ? 0.0f
                : (&jnt.lambdaDriveLinear.x)[a];
            const physx::PxReal rawForce =
                usePositionObjective
                    ? stiffness * positionError + (damping / dt) * C
                    : rho_drive * C + lambda;
            const physx::PxReal driveForce = usePhysicalObjective
                ? physx::PxClamp(rawForce, -maxForce, maxForce)
                : rawForce;
            const bool saturated = usePhysicalObjective &&
                physx::PxAbs(rawForce) >= maxForce;
            physx::PxReal signAL = isBodyA ? -1.0f : 1.0f;
            physx::PxReal f = signAL * driveForce;
            if (saturated)
              rho_drive = 0.0f;

            // Full 6D Jacobian Jd = (wAxis, rArm x wAxis), matching
            // standalone.  The drive force acts at the anchor point, so
            // the lever arm produces torque.
            physx::PxVec3 rCrossW = rArm.cross(wAxis);

            // Hessian: outer(Jd, Jd * rho_drive) -> all 4 blocks
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l) {
                A.linearLinear(k, l) +=
                    rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];
                A.linearAngular(k, l) +=
                    rho_drive * (&wAxis.x)[k] * (&rCrossW.x)[l];
                A.angularLinear(k, l) +=
                    rho_drive * (&rCrossW.x)[k] * (&wAxis.x)[l];
                A.angularAngular(k, l) +=
                    rho_drive * (&rCrossW.x)[k] * (&rCrossW.x)[l];
              }

            // RHS: gradient on both linear and angular
            gLinear += physx::PxVec3(f * wAxis.x, f * wAxis.y, f * wAxis.z);
            gAngular += physx::PxVec3(f * rCrossW.x, f * rCrossW.y, f * rCrossW.z);
          }
        }

        // --- Angular velocity drive (AL constraint) ---
        if ((compiledDriveRows & 0x38u) != 0) {
          // Angular displacement from start-of-step for this body
          physx::PxQuat dqThis =
              body.rotation * body.prevRotation.getConjugate();
          if (dqThis.w < 0.0f)
            dqThis = -dqThis;
          physx::PxVec3 dThetaThis(dqThis.x, dqThis.y, dqThis.z);
          dThetaThis *= 2.0f;

          physx::PxVec3 dThetaOther =
              isBodyA ? jnt.externalAngularStepB
                      : jnt.externalAngularStepA;
          if (otherBody) {
            physx::PxQuat dqOther =
                otherBody->rotation * otherBody->prevRotation.getConjugate();
            if (dqOther.w < 0.0f)
              dqOther = -dqOther;
            dThetaOther = physx::PxVec3(dqOther.x, dqOther.y, dqOther.z) * 2.0f;
          }

          physx::PxVec3 dThetaA(0.0f), dThetaB(0.0f);
          if (isBodyA) {
            dThetaA = dThetaThis;
            dThetaB = dThetaOther;
          } else {
            dThetaA = dThetaOther;
            dThetaB = dThetaThis;
          }
          physx::PxVec3 relDW = dThetaB - dThetaA;
          physx::PxVec3 worldAngTarget =
              jointFrameA.rotate(jnt.driveAngularVelocity) * dt;
          physx::PxReal signAL = isBodyA ? -1.0f : 1.0f;

          const bool slerpDrive =
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::SlerpVelocityDrive) ||
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::SlerpPositionDrive) ||
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::
                      CoupledAngularPositionDrive) ||
              hasAvbdJointObjective(
                  jnt.objectiveProgram,
                  AvbdJointObjectiveKind::OrdinaryD6SlerpDrive);
          if (slerpDrive) {
            physx::PxReal damping =
                jnt.angularDamping.z; // SLERP uses Z damping slot
            if (damping > 0.0f) {
              const bool usePhysicalSlerpVelocityObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::SlerpVelocityDrive);
              const bool usePhysicalSlerpPositionObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::SlerpPositionDrive);
              if (usePhysicalSlerpPositionObjective) {
                physx::PxQuat worldFrameA =
                    bodyARef ? bodyARef->rotation * jnt.localFrameA
                             : jnt.localFrameA;
                physx::PxQuat worldFrameB =
                    bodyBRef ? bodyBRef->rotation * jnt.localFrameB
                             : jnt.localFrameB;
                worldFrameA.normalize();
                worldFrameB.normalize();
                physx::PxQuat currentRelative =
                    worldFrameA.getConjugate() * worldFrameB;
                currentRelative.normalize();
                physx::PxQuat targetRelative = jnt.driveAngularPosition;
                if (currentRelative.dot(targetRelative) < 0.0f)
                  targetRelative = -targetRelative;
                const physx::PxQuat delta =
                    targetRelative.getConjugate() * currentRelative;
                physx::PxVec3 rows[3];
                computeSlerpJacobianAxes(rows, worldFrameA * targetRelative,
                                         worldFrameB);
                const physx::PxReal stiffness = jnt.angularStiffness.z;
                for (int row = 0; row < 3; ++row) {
                  const physx::PxReal C = relDW.dot(rows[row]);
                  const physx::PxReal rawTorque =
                      stiffness * (&delta.x)[row] + (damping / dt) * C;
                  const physx::PxReal driveTorque = physx::PxClamp(
                      rawTorque, -jnt.driveAngularForce.z,
                      jnt.driveAngularForce.z);
                  const bool saturated =
                      physx::PxAbs(rawTorque) >= jnt.driveAngularForce.z;
                  const physx::PxReal rowTangent =
                      saturated ? 0.0f : stiffness + damping / dt;
                  for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                      A.angularAngular(k, l) +=
                          rowTangent * (&rows[row].x)[k] *
                          (&rows[row].x)[l];
                  gAngular += rows[row] * (signAL * driveTorque);
                }
              } else {
                physx::PxReal rho_drive =
                    usePhysicalSlerpVelocityObjective ? damping / dt
                                                      : damping / dt2;
                if (jnt.isAngularAccelerationDrive(2)) {
                  const physx::PxReal driveScale =
                      computeAngularDriveRecipResponse(
                          bodyARef, bodyBRef,
                          physx::PxVec3(1.0f, 0.0f, 0.0f));
                  const physx::PxReal stiffness = jnt.angularStiffness.z;
                  const physx::PxReal dampingOnly =
                      physx::PxMax(0.0f, damping - stiffness);
                  const physx::PxReal implicitScale =
                      1.0f /
                      (1.0f + dt * (dt * stiffness + dampingOnly));
                  rho_drive *= driveScale * implicitScale;
                }
                const physx::PxReal targetScale =
                    usePhysicalSlerpVelocityObjective &&
                            mConfig.angularDamping > 1e-6f
                        ? 1.0f / mConfig.angularDamping
                        : 1.0f;
                for (int k = 0; k < 3; ++k) {
                  physx::PxReal C =
                      (&relDW.x)[k] -
                      targetScale * (&worldAngTarget.x)[k];
                  const physx::PxReal lam =
                      usePhysicalSlerpVelocityObjective
                          ? 0.0f
                          : (&jnt.lambdaDriveAngular.x)[k];
                  const physx::PxReal rawTorque = rho_drive * C + lam;
                  const physx::PxReal driveTorque =
                      usePhysicalSlerpVelocityObjective
                          ? physx::PxClamp(rawTorque,
                                           -jnt.driveAngularForce.z,
                                           jnt.driveAngularForce.z)
                          : rawTorque;
                  const bool saturated =
                      usePhysicalSlerpVelocityObjective &&
                      physx::PxAbs(rawTorque) >=
                          jnt.driveAngularForce.z;
                  physx::PxReal f = signAL * driveTorque;

                  if (!saturated)
                    A.angularAngular(k, k) += rho_drive;
                  (&gAngular.x)[k] += f;
                }
              }
            }
          } else {
            // Axis mapping: bit3=twist(X), bit4=swing1(Y), bit5=swing2(Z)
            struct AxisDrive {
              int bit;
              int dampIdx;
              physx::PxVec3 localAxis;
            };
            const AxisDrive axes[3] = {
                {3, 0, physx::PxVec3(1.0f, 0.0f, 0.0f)}, // TWIST
                {4, 1, physx::PxVec3(0.0f, 1.0f, 0.0f)}, // SWING1
                {5, 2, physx::PxVec3(0.0f, 0.0f, 1.0f)}, // SWING2
            };

            for (int a = 0; a < 3; ++a) {
              if ((compiledDriveRows &
                   (1u << axes[a].bit)) == 0)
                continue;
              const physx::PxReal damping =
                  (&jnt.angularDamping.x)[axes[a].dampIdx];
              const physx::PxReal stiffness =
                  (&jnt.angularStiffness.x)[axes[a].dampIdx];
              const bool isAccelerationDrive =
                  jnt.isAngularAccelerationDrive(axes[a].dampIdx);
              const physx::PxReal effectiveRate =
                  isAccelerationDrive ? dt * stiffness + damping : damping;
              if (effectiveRate <= 0.0f)
                continue;

              physx::PxVec3 wAxis = jointFrameA.rotate(axes[a].localAxis);
              // PhysX TGS convention: Twist/Swing target velocities are
              // applied as (wA - wB), meaning wB - wA = -target. SLERP is
              // applied as wB
              // - wA = target, which is handled above.
              physx::PxReal targetOmega_dt = -worldAngTarget.dot(wAxis);
              physx::PxReal C = relDW.dot(wAxis) - targetOmega_dt;

              const bool usePhysicalAngularAxisVelocityObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::
                          AngularAxisVelocityDrive);
              const bool usePhysicalAngularPositionObjective =
                  hasAvbdJointObjective(
                      jnt.objectiveProgram,
                      AvbdJointObjectiveKind::
                          AngularAxisPositionDrive);
              physx::PxReal positionResidual = 0.0f;
              physx::PxReal positionTangent = 0.0f;
              if (usePhysicalAngularPositionObjective) {
                physx::PxQuat worldFrameA =
                    bodyARef ? bodyARef->rotation * jnt.localFrameA
                             : jnt.localFrameA;
                physx::PxQuat worldFrameB =
                    bodyBRef ? bodyBRef->rotation * jnt.localFrameB
                             : jnt.localFrameB;
                worldFrameA.normalize();
                worldFrameB.normalize();
                physx::PxQuat currentRelative =
                    worldFrameA.getConjugate() * worldFrameB;
                currentRelative.normalize();
                physx::PxQuat targetRelative = jnt.driveAngularPosition;
                if (currentRelative.dot(targetRelative) < 0.0f)
                  targetRelative = -targetRelative;
                const physx::PxQuat delta =
                    currentRelative * targetRelative.getConjugate();

                if (axes[a].dampIdx == 0) {
                  // ExtD6Joint emits geometricError=-2*delta.x for TWIST.
                  // AVBD's gradient uses current-target, so this is the
                  // opposite sign. Its local derivative is delta.w.
                  positionResidual = 2.0f * delta.x;
                  positionTangent = physx::PxAbs(delta.w);
                } else if (axes[a].dampIdx == 1) {
                  // ExtD6Joint emits delta.getBasisVector0().z for SWING1.
                  // The AVBD gradient again uses the opposite sign. In the
                  // predicate-approved isolated SWING1 row this is a
                  // full-angle sine residual with a cosine tangent.
                  positionResidual = -delta.getBasisVector0().z;
                  positionTangent = physx::PxAbs(
                      1.0f - 2.0f * delta.y * delta.y);
                } else {
                  // ExtD6Joint emits -delta.getBasisVector0().y for SWING2.
                  // AVBD uses its opposite gradient. In the predicate-approved
                  // isolated SWING2 row this is again a full-angle sine with
                  // the corresponding cosine tangent.
                  positionResidual = delta.getBasisVector0().y;
                  positionTangent = physx::PxAbs(
                      1.0f - 2.0f * delta.z * delta.z);
                }
              }
              // In the scoped force-mode TWIST/SWING1/SWING2 subset, damping
              // has the physical units torque/(angular velocity).  C is an
              // angular displacement over the step, so damping/dt maps C back
              // to a torque.  The wider angular family retains its existing AL
              // objective until SLERP and spring semantics are gated.
              physx::PxReal rho_drive =
                  usePhysicalAngularPositionObjective
                      ? stiffness * positionTangent + damping / dt
                      : (usePhysicalAngularAxisVelocityObjective
                             ? damping / dt
                             : damping / dt2);
              if (isAccelerationDrive) {
                const physx::PxReal driveScale =
                  computeAngularDriveRecipResponse(bodyARef, bodyBRef, wAxis);
                const physx::PxReal implicitScale =
                  1.0f / (1.0f + dt * effectiveRate);
                rho_drive = driveScale * implicitScale * effectiveRate;
              }
              const bool usePhysicalAngularObjective =
                  usePhysicalAngularAxisVelocityObjective ||
                  usePhysicalAngularPositionObjective;
              const physx::PxReal lambda =
                  usePhysicalAngularObjective
                      ? 0.0f
                      : (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx];
              const physx::PxReal rawTorque =
                  usePhysicalAngularPositionObjective
                      ? stiffness * positionResidual + (damping / dt) * C
                      : rho_drive * C + lambda;
              const physx::PxReal driveTorque =
                  usePhysicalAngularObjective
                      ? physx::PxClamp(rawTorque,
                                       -jnt.driveAngularForce[axes[a].dampIdx],
                                       jnt.driveAngularForce[axes[a].dampIdx])
                      : rawTorque;
              const bool saturated =
                  usePhysicalAngularObjective &&
                  physx::PxAbs(rawTorque) >=
                      jnt.driveAngularForce[axes[a].dampIdx];
              physx::PxReal f = signAL * driveTorque;
              if (saturated)
                rho_drive = 0.0f;

              // Hessian: -_drive - (wAxis - wAxis) on angular block
              for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                  A.angularAngular(k, l) +=
                      rho_drive * (&wAxis.x)[k] * (&wAxis.x)[l];

              // RHS
              gAngular += physx::PxVec3(f * wAxis.x, f * wAxis.y, f * wAxis.z);
            }
          }
        }
      }

      numTouching++;
      hasLinearCoupling = true; // D6 joints always create pos-rot coupling via lever arm
    }
  }

  // =========================================================================
  // Step 3g: Accumulate GEAR JOINT contributions (angular-only, position-level)
  //
  // Constraint: C = geometricError  (accumulated angle error, radians)
  //   Computed by GearJoint::updateError() each frame.
  //
  // Jacobians match GearJointSolverPrep (ExtGearJoint.cpp):
  //   Body A:  J_ang = +worldAxis0 * gearRatio   (con.angular0 = axis0*ratio)
  //   Body B:  J_ang = -worldAxis1               (con.angular1 = -axis1)
  //
  // gearAxis0/1 stored as BODY LOCAL vectors -> rotate to world with
  // body.rotation
  //
  //   LHS: A_ang += pen * J_ang - J_ang
  //   RHS: g_ang += J_ang * (pen*C + lambda)
  // =========================================================================
  if (gearJoints && numGear > 0) {
    const physx::PxU32 *mapIndices = nullptr;
    physx::PxU32 mapCount = 0;
    if (gearMap && gearMap->numBodies > 0)
      gearMap->getBodyConstraints(bodyIndex, mapIndices, mapCount);
    const physx::PxU32 loopCount = mapIndices ? mapCount : numGear;

    for (physx::PxU32 ji = 0; ji < loopCount; ++ji) {
      const physx::PxU32 j = mapIndices ? mapIndices[ji] : ji;
      if (j >= numGear)
        continue;
      const AvbdGearJointConstraint &gnt = gearJoints[j];
      const physx::PxU32 bodyAIdx = gnt.header.bodyIndexA;
      const physx::PxU32 bodyBIdx = gnt.header.bodyIndexB;

      if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex)
        continue;
      const bool isBodyA = (bodyAIdx == bodyIndex);
      const bool otherIsStatic =
          isBodyA ? (bodyBIdx == 0xFFFFFFFF || bodyBIdx >= numBodies)
                  : (bodyAIdx == 0xFFFFFFFF || bodyAIdx >= numBodies);

      physx::PxReal dwA = 0.0f;
      physx::PxReal dwB = 0.0f;

      auto computeDeltaW = [](const AvbdSolverBody &b,
                              const physx::PxVec3 &axis) -> physx::PxReal {
        physx::PxQuat dq = b.rotation * b.prevRotation.getConjugate();
        if (dq.w < 0.0f)
          dq = -dq;
        return physx::PxVec3(dq.x, dq.y, dq.z).dot(axis) * 2.0f;
      };

      physx::PxVec3 worldAxis0, worldAxis1;

      if (isBodyA) {
        worldAxis0 = body.rotation.rotate(gnt.gearAxis0);
        dwA = computeDeltaW(body, worldAxis0);
        // For static body B, axis is fixed in world space. (Ideally we'd use
        // the static rotation, but typically it rotates from identity)
        worldAxis1 = gnt.gearAxis1;
      } else {
        worldAxis1 = body.rotation.rotate(gnt.gearAxis1);
        dwB = computeDeltaW(body, worldAxis1);
        worldAxis0 = gnt.gearAxis0;
      }

      // If the other body IS dynamic, rotate its axis and fetch its dw
      if (!otherIsStatic) {
        if (isBodyA) {
          worldAxis1 = bodies[bodyBIdx].rotation.rotate(gnt.gearAxis1);
          dwB = computeDeltaW(bodies[bodyBIdx], worldAxis1);
        } else {
          worldAxis0 = bodies[bodyAIdx].rotation.rotate(gnt.gearAxis0);
          dwA = computeDeltaW(bodies[bodyAIdx], worldAxis0);
        }
      }

      physx::PxReal C = dwA * gnt.gearRatio + dwB + gnt.geometricError;

      const physx::PxVec3 rawAxis = isBodyA ? worldAxis0 : worldAxis1;
      const physx::PxVec3 tmpInvIAxis = body.invInertiaWorld.transform(rawAxis);
      const physx::PxReal invIaxial = rawAxis.dot(tmpInvIAxis);
      const physx::PxReal Iaxial =
          (invIaxial > 1e-10f) ? (1.0f / invIaxial) : 0.0f;
      physx::PxReal pen = physx::PxMax(gnt.header.rho, Iaxial * invDt2);

      // Jacobian for THIS body - Body B uses POSITIVE axis1 (matches TGS
      // algebraic summation)
      physx::PxVec3 J_ang =
          isBodyA ? (worldAxis0 * gnt.gearRatio) : (worldAxis1);

      // AL force: f = pen * C + gnt.lambdaGear
      physx::PxReal f = pen * C + gnt.lambdaGear;

#if AVBD_JOINT_DEBUG
      {
        static physx::PxU32 s_gearDebugCount = 0;
        if (s_gearDebugCount < 0) {
          printf("[Gear] frame=%u isA=%d body%u num=%u C=%.4f (err=%.4f "
                 "dwA=%.4f dwB=%.4f) f=%.1f pen=%.1f gearRatio=%.2f "
                 "axis==(%.1f,%.1f,%.1f)\n",
                 s_gearDebugCount, isBodyA, bodyIndex, ji, C,
                 gnt.geometricError, dwA, dwB, f, pen, gnt.gearRatio,
                 isBodyA ? worldAxis0.x : worldAxis1.x,
                 isBodyA ? worldAxis0.y : worldAxis1.y,
                 isBodyA ? worldAxis0.z : worldAxis1.z);
          if (!isBodyA)
            s_gearDebugCount++; // increment after both passes
        }
      }
#endif

      // Accumulate into 6x6 Hessian (linear part zero, angular part = J)
      A.addConstraintContribution(physx::PxVec3(0.0f), J_ang, pen);

      // RHS gradient
      gAngular += J_ang * f;

      numTouching++;
    }
  }

  // =========================================================================
  // Step 4: Handle bodies with no constraints at all
  // =========================================================================
  if (numTouching == 0) {
    body.position = body.inertialPosition;
    body.rotation = body.inertialRotation;
    return;
  }

  // =========================================================================
  // Step 5: Solve A * delta = g via LDLT
  // =========================================================================
  AvbdLDLT ldlt;
  AvbdVec6 rhs(gLinear, gAngular);

#if AVBD_JOINT_DEBUG
  {
    static physx::PxU32 s_debugSolveFrame = 0;
    bool doSolveDebug = (s_debugSolveFrame < 4);
    if (doSolveDebug &&
        (numD6 > 0 || numGear > 0)) {
      printf("  [solveUnified] body%u touching=%u gLin=(%.4f,%.4f,%.4f) "
             "gAng=(%.4f,%.4f,%.4f)\n",
             bodyIndex, numTouching, gLinear.x, gLinear.y, gLinear.z,
             gAngular.x, gAngular.y, gAngular.z);
      printf("    H_diag pos=(%.1f,%.1f,%.1f) rot=(%.1f,%.1f,%.1f)\n",
             A.linearLinear.column0.x, A.linearLinear.column1.y,
             A.linearLinear.column2.z, A.angularAngular.column0.x,
             A.angularAngular.column1.y, A.angularAngular.column2.z);
      printf("    inertialDelta pos=(%.6f,%.6f,%.6f)\n",
             body.position.x - body.inertialPosition.x,
             body.position.y - body.inertialPosition.y,
             body.position.z - body.inertialPosition.z);
      s_debugSolveFrame++;
    }
  }
#endif

  physx::PxVec3 deltaPos;
  physx::PxVec3 deltaTheta;

  // Force 6x6 solve for bodies touching Prismatic joints: the 3x3
  // decoupled solve is incompatible with Prismatic's axis-dependent
  // position projection, which creates divergent oscillation.
  const bool use6x6 = mConfig.enableLocal6x6Solve || hasLinearCoupling;
  if (use6x6) {
    if (ldlt.decomposeWithRegularization(A)) {
      AvbdVec6 delta = ldlt.solve(rhs);
      deltaPos = delta.linear;
      deltaTheta = delta.angular;
    } else {
      deltaPos = physx::PxVec3(0.0f);
      deltaTheta = physx::PxVec3(0.0f);
    }
  } else {
    // 3x3 Block-Diagonal Decoupled Solve Fallback
    physx::PxMat33 Alin = A.linearLinear;
    physx::PxMat33 Aang = A.angularAngular;

    bool linOk = (physx::PxAbs(Alin.getDeterminant()) > 1e-12f);
    bool angOk = (physx::PxAbs(Aang.getDeterminant()) > 1e-12f);

    if (linOk) {
      physx::PxMat33 AlinInv = Alin.getInverse();
      deltaPos = AlinInv * gLinear;
    } else {
      deltaPos = physx::PxVec3(0.0f);
    }

    if (angOk) {
      physx::PxMat33 AangInv = Aang.getInverse();
      deltaTheta = AangInv * gAngular;
    } else {
      deltaTheta = physx::PxVec3(0.0f);
    }
  }

  const physx::PxReal ogcAlpha = limitRigidOgcCandidate(
      ogcPairContext, bodyIndex, body, deltaPos, deltaTheta, softContacts,
      numSoftContacts, rigidTargetContactStarts, rigidTargetContactRefs,
      softParticles);
  deltaPos *= ogcAlpha;
  deltaTheta *= ogcAlpha;

#if AVBD_JOINT_DEBUG
  {
    static physx::PxU32 s_debugSolveFrame2 = 0;
    bool doSolveDebug = (s_debugSolveFrame2 < 2);
    if (doSolveDebug && (numD6 > 0 || numGear > 0)) {
      printf("    delta pos=(%.6f,%.6f,%.6f) rot=(%.6f,%.6f,%.6f)\n",
             deltaPos.x, deltaPos.y, deltaPos.z, deltaTheta.x, deltaTheta.y,
             deltaTheta.z);
      printf("    newPos=(%.4f,%.4f,%.4f)\n", body.position.x - deltaPos.x,
             body.position.y - deltaPos.y, body.position.z - deltaPos.z);
    }
    // Only increment once per full body loop (not per body)
    if (bodyIndex == 0 && (numD6 > 0 || numGear > 0)) {
      s_debugSolveFrame2++;
    }
  }
#endif

  // =========================================================================
  // Step 6: Apply update  x -= delta
  // =========================================================================
  body.position -= deltaPos;

  if (deltaTheta.magnitudeSquared() > 1e-12f) {
    physx::PxQuat dq(deltaTheta.x, deltaTheta.y, deltaTheta.z, 0.0f);
    body.rotation = (body.rotation - dq * body.rotation * 0.5f).getNormalized();
  }
}

//=============================================================================
// Block Descent Iteration - Position-Based Constraint Solving
//=============================================================================



/**
 * @brief Compute correction for D6 joint
 */
bool AvbdSolver::computeD6JointCorrection(const AvbdD6JointConstraint &joint,
                                          AvbdSolverBody *bodies,
                                          physx::PxU32 numBodies,
                                          physx::PxU32 bodyIndex,
                                          physx::PxVec3 &deltaPos,
                                          physx::PxVec3 &deltaTheta) {

  const physx::PxU32 bodyAIdx = joint.header.bodyIndexA;
  const physx::PxU32 bodyBIdx = joint.header.bodyIndexB;

  bool bodyAIsStatic = (bodyAIdx >= numBodies);

  if (bodyAIdx != bodyIndex && bodyBIdx != bodyIndex) {
    return false;
  }

  AvbdSolverBody &body = bodies[bodyIndex];
  bool isBodyA = (bodyAIdx == bodyIndex);

  AvbdSolverBody *otherBody = nullptr;
  if (isBodyA && bodyBIdx < numBodies) {
    otherBody = &bodies[bodyBIdx];
  } else if (!isBodyA && bodyAIdx < numBodies) {
    otherBody = &bodies[bodyAIdx];
  }

  deltaPos = physx::PxVec3(0.0f);
  deltaTheta = physx::PxVec3(0.0f);

  bool hasCorrection = false;

  // Check if bodies are static (index >= numBodies means static body, frame
  // already in world space) Note: bodyAIsStatic already defined above for
  // debug purposes
  bool bodyBIsStatic = (bodyBIdx >= numBodies);

  // Get rotations for frame transforms
  physx::PxQuat rotA =
      bodyAIsStatic
          ? physx::PxQuat(physx::PxIdentity)
          : (isBodyA ? body.rotation
                     : (otherBody ? otherBody->rotation
                                  : physx::PxQuat(physx::PxIdentity)));
  physx::PxQuat rotB =
      bodyBIsStatic ? physx::PxQuat(physx::PxIdentity)
                    : (isBodyA ? (otherBody ? otherBody->rotation
                                            : physx::PxQuat(physx::PxIdentity))
                               : body.rotation);

  physx::PxVec3 worldAnchorA, worldAnchorB;
  if (isBodyA) {
    worldAnchorA = body.position + body.rotation.rotate(joint.anchorA);
    worldAnchorB =
        otherBody
            ? otherBody->position + otherBody->rotation.rotate(joint.anchorB)
            : joint.anchorB; // anchorB already in world space for static
  } else {
    worldAnchorA =
        otherBody
            ? otherBody->position + otherBody->rotation.rotate(joint.anchorA)
            : joint.anchorA; // anchorA already in world space for static
    worldAnchorB = body.position + body.rotation.rotate(joint.anchorB);
  }

  physx::PxVec3 posError = worldAnchorA - worldAnchorB;

  // Position constraint (linear locked) - but skip axes with velocity drive
  // When velocity drive is active, we want the body to move, not be
  // constrained
  if (joint.linearMotion == 0) {
    // Determine which axes have velocity drive (we'll skip position
    // constraint on those)
    physx::PxU32 linearDriveAxes =
        joint.driveFlags & 0x7; // bits 0,1,2 for X,Y,Z

    // If we have velocity drive, project out the position error along driven
    // axes
    physx::PxVec3 constrainedPosError = posError;

    if (linearDriveAxes != 0 && !isBodyA) {
      // Get joint frame in world space
      physx::PxQuat jointFrameA =
          bodyAIsStatic ? joint.localFrameA : (rotA * joint.localFrameA);
      physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
      if (qMag2 > AvbdConstants::AVBD_NUMERICAL_EPSILON && PxIsFinite(qMag2)) {
        jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

        // Remove position error component along driven axes
        for (int axis = 0; axis < 3; ++axis) {
          if ((linearDriveAxes & (1 << axis)) != 0) {
            physx::PxVec3 localAxis(0.0f);
            (&localAxis.x)[axis] = 1.0f;
            physx::PxVec3 worldAxis = jointFrameA.rotate(localAxis);
            // Remove the component of position error along this driven axis
            constrainedPosError -=
                worldAxis * constrainedPosError.dot(worldAxis);
          }
        }
      }
    }

    physx::PxReal posErrorMag = constrainedPosError.magnitude();
    if (posErrorMag > AvbdConstants::AVBD_NUMERICAL_EPSILON) {
      physx::PxVec3 direction = constrainedPosError / posErrorMag;

      physx::PxVec3 r = isBodyA ? body.rotation.rotate(joint.anchorA)
                                : body.rotation.rotate(joint.anchorB);
      physx::PxVec3 rCrossD = r.cross(direction);
      physx::PxReal w =
          body.invMass + rCrossD.dot(body.invInertiaWorld * rCrossD);

      if (otherBody && otherBody->invMass > 0.0f) {
        physx::PxVec3 rOther = isBodyA
                                   ? otherBody->rotation.rotate(joint.anchorB)
                                   : otherBody->rotation.rotate(joint.anchorA);
        physx::PxVec3 rOtherCrossD = rOther.cross(direction);
        w += otherBody->invMass +
             rOtherCrossD.dot(otherBody->invInertiaWorld * rOtherCrossD);
      }

      if (w > 1e-6f) {
        physx::PxReal correctionMag = -posErrorMag / w;
        physx::PxReal sign = isBodyA ? 1.0f : -1.0f;

        deltaPos = direction * (correctionMag * body.invMass * sign);
        deltaTheta = (body.invInertiaWorld * rCrossD) * (correctionMag * sign);
      }
      hasCorrection = true;
    }
  }

  // Drive constraints now handled in AVBD Hessian
  // (solveLocalSystemWithJoints/3x3) GS fallback for drives is disabled.

  return hasCorrection;
}

//=============================================================================

} // namespace Dy
} // namespace physx
