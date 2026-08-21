// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/DyAvbdSolver.h"
#include "avbd/solver/joint/DyAvbdJointDriveMath.h"
#include "avbd/solver/joint/DyAvbdJointGeometryPolicy.h"
#include "avbd/solver/joint/DyAvbdJointProjection.h"
#include "common/PxProfileZone.h"
#include "PxConstraintDesc.h"

namespace physx {
namespace Dy {

// Joint Stage 5: keep the D6, gear, contact and soft dual updates in one
// explicit phase boundary. The numerical order is intentionally unchanged;
// this extraction only removes the coordination block from solveWithJoints.
void AvbdSolver::runAvbdJointDualPhase(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    AvbdD6JointConstraint *d6Joints, physx::PxU32 numD6,
    AvbdGearJointConstraint *gearJoints, physx::PxU32 numGear,
    physx::PxReal dt, physx::PxReal invDt2,
    const AvbdSolverConfig &config,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    AvbdSolverStats &stats) {
        PX_PROFILE_ZONE("AVBD.updateLambda", 0);
        updateLagrangianMultipliers(bodies, numBodies, contacts, numContacts,
                                    dt, stats);

        // ---------------------------------------------------------------
        // D6, Gear: ADMM-safe dual + lambda decay
        //
        // Three mechanisms ensure stable AL convergence:
        //   (A) effectiveRho = max(rho, M/h^2) in primal (above)
        //   (B) rhoDual = min(Mh2, rho^2/(rho+Mh2)) -- safe step size
        //   (C) lambda = decay*lambda + rhoDual*C -- leaky integrator
        // ---------------------------------------------------------------
        {
          const physx::PxReal lambdaDecay = 0.99f;

          auto getBodyMass = [&](physx::PxU32 idx) -> physx::PxReal {
            return (idx == 0xFFFFFFFF || idx >= numBodies)
                       ? 0.0f
                       : (bodies[idx].invMass > 1e-8f
                              ? 1.0f / bodies[idx].invMass
                              : 0.0f);
          };
          auto computeRhoDual = [&](physx::PxU32 idxA, physx::PxU32 idxB,
                                    physx::PxReal rho) -> physx::PxReal {
            physx::PxReal mA = getBodyMass(idxA);
            physx::PxReal mB = getBodyMass(idxB);
            physx::PxReal mEff;
            if (mA <= 0.0f)
              mEff = mB;
            else if (mB <= 0.0f)
              mEff = mA;
            else
              mEff = physx::PxMin(mA, mB);
            if (mEff <= 0.0f)
              return 0.0f;
            physx::PxReal Mh2 = mEff * invDt2;
            physx::PxReal admm_step = rho * rho / (rho + Mh2);
            return physx::PxMin(Mh2, admm_step);
          };

          // D6 joints
          for (physx::PxU32 j = 0; j < numD6; ++j) {
            AvbdD6JointConstraint &jnt = d6Joints[j];
            jnt.writebackLinearImpulse = physx::PxVec3(0.0f);
            jnt.writebackLinearImpulseValid = 0;
            jnt.writebackAngularImpulse = physx::PxVec3(0.0f);
            jnt.writebackAngularImpulseValid = 0;
            const physx::PxU32 compiledSourceRows =
                getAvbdJointObjectiveSourceRows(
                    jnt.objectiveProgram);
            const physx::PxU32 compiledDriveRows =
                compiledSourceRows & 0x3fu;
            const bool compiledConeObjective =
                (compiledSourceRows &
                 eJOINT_SOURCE_ANGULAR_CONE) != 0;

            const bool noDualObjective =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationFixedTendon) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationSpatialTendon) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        CoupledSpatialTendon) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationCompliantMimic) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        GenericAccelerationDamping1D) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        GenericRestitution1D);
            if (noDualObjective) {
              jnt.lambdaLinear = physx::PxVec3(0.0f);
              jnt.lambdaAngular = physx::PxVec3(0.0f);
              continue;
            }

            if (hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        GenericForceSpring1D)) {
              if (dt <= 0.0f)
                continue;
              const physx::PxReal C =
                  computeGeneric1DViolation(jnt, bodies, numBodies, dt);
              const physx::PxReal velocity =
                  (C - jnt.genericGeometricError) / dt;
              const physx::PxReal totalForce =
                  jnt.header.rho * C + jnt.header.damping * velocity;
              const physx::PxReal appliedImpulse = physx::PxClamp(
                  -totalForce * dt, jnt.genericMinImpulse,
                  jnt.genericMaxImpulse);
              const bool outputForce =
                  (jnt.genericRowFlags &
                   static_cast<physx::PxU32>(
                       Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;
              jnt.writebackLinearImpulse =
                  outputForce ? jnt.genericLinearA * appliedImpulse
                              : physx::PxVec3(0.0f);
              jnt.writebackAngularImpulse =
                  outputForce
                      ? jnt.genericAngularAWriteback * appliedImpulse
                      : physx::PxVec3(0.0f);
              jnt.writebackLinearImpulseValid = 1;
              jnt.writebackAngularImpulseValid = 1;
              jnt.lambdaLinear = physx::PxVec3(0.0f);
              jnt.lambdaAngular = physx::PxVec3(0.0f);
              continue;
            }

            if (hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::GenericHard1D) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        ArticulationHardMimic)) {
              const physx::PxReal effectiveMass =
                  computeGeneric1DEffectiveMass(jnt, bodies, numBodies);
              if (effectiveMass <= 0.0f || dt <= 0.0f)
                continue;

              const physx::PxReal Mh2 = effectiveMass * invDt2;
              const physx::PxReal rho = jnt.header.rho;
              const physx::PxReal rhoDual =
                  physx::PxMin(Mh2, rho * rho / (rho + Mh2));
              const physx::PxReal C =
                  computeGeneric1DViolation(jnt, bodies, numBodies, dt);
              const physx::PxReal pen = physx::PxMax(rho, Mh2);
              const physx::PxReal totalForce =
                  pen * C + jnt.lambdaLinear.x;
              const physx::PxReal appliedImpulse = physx::PxClamp(
                  -totalForce * dt, jnt.genericMinImpulse,
                  jnt.genericMaxImpulse);

              const bool outputForce =
                  (jnt.genericRowFlags &
                   static_cast<physx::PxU32>(
                       Px1DConstraintFlag::eOUTPUT_FORCE)) != 0;
              jnt.writebackLinearImpulse =
                  outputForce ? jnt.genericLinearA * appliedImpulse
                              : physx::PxVec3(0.0f);
              jnt.writebackAngularImpulse =
                  outputForce
                      ? jnt.genericAngularAWriteback * appliedImpulse
                              : physx::PxVec3(0.0f);
              jnt.writebackLinearImpulseValid = 1;
              jnt.writebackAngularImpulseValid = 1;

              const physx::PxReal newLambda =
                  jnt.lambdaLinear.x * lambdaDecay + C * rhoDual;
              const physx::PxReal clampedDualImpulse = physx::PxClamp(
                  -newLambda * dt, jnt.genericMinImpulse,
                  jnt.genericMaxImpulse);
              jnt.lambdaLinear =
                  physx::PxVec3(-clampedDualImpulse / dt, 0.0f, 0.0f);
              jnt.lambdaAngular = physx::PxVec3(0.0f);
              continue;
            }

            physx::PxReal rhoDual = computeRhoDual(
                jnt.header.bodyIndexA, jnt.header.bodyIndexB, jnt.header.rho);
            if (rhoDual <= 0.0f)
              continue;
            bool aStatic = (jnt.header.bodyIndexA == 0xFFFFFFFF ||
                            jnt.header.bodyIndexA >= numBodies);
            bool bStatic = (jnt.header.bodyIndexB == 0xFFFFFFFF ||
                            jnt.header.bodyIndexB >= numBodies);
            physx::PxVec3 wA =
                aStatic ? jnt.anchorA
                        : bodies[jnt.header.bodyIndexA].position +
                              bodies[jnt.header.bodyIndexA].rotation.rotate(
                                  jnt.anchorA);
            physx::PxVec3 wB =
                bStatic ? jnt.anchorB
                        : bodies[jnt.header.bodyIndexB].position +
                              bodies[jnt.header.bodyIndexB].rotation.rotate(
                                  jnt.anchorB);
            physx::PxQuat rotA = aStatic
                                     ? physx::PxQuat(physx::PxIdentity)
                                     : bodies[jnt.header.bodyIndexA].rotation;
            physx::PxQuat rotB = bStatic
                                     ? physx::PxQuat(physx::PxIdentity)
                                     : bodies[jnt.header.bodyIndexB].rotation;
            physx::PxVec3 posViol = wA - wB;

            // Compute joint-frame axes (match primal axis selection)
            physx::PxQuat jointFrameA_dual =
                aStatic
                    ? jnt.localFrameA
                    : bodies[jnt.header.bodyIndexA].rotation * jnt.localFrameA;
            {
              physx::PxReal qm2 = jointFrameA_dual.magnitudeSquared();
              if (qm2 > 1e-8f && PxIsFinite(qm2))
                jointFrameA_dual *= 1.0f / physx::PxSqrt(qm2);
            }

            const bool linAllLocked = (jnt.linearMotion == 0);
            bool hasLockedLinearRow = false;
            bool hasUnsupportedLinearRow = false;
            for (int axis = 0; axis < 3; ++axis) {
              hasLockedLinearRow |= (jnt.getLinearMotion(axis) == 0);
              hasUnsupportedLinearRow |= (jnt.getLinearMotion(axis) == 1) ||
                                         jnt.isLinearDriveEnabled(axis);
            }
            const physx::PxReal mA =
                getBodyMass(jnt.header.bodyIndexA);
            const physx::PxReal mB =
                getBodyMass(jnt.header.bodyIndexB);
            const physx::PxReal pen = physx::PxMax(
                jnt.header.rho, physx::PxMax(mA, mB) * invDt2);
            physx::PxVec3 actor0LinearForce(0.0f);
            physx::PxVec3 actor0PositionDriveForce(0.0f);
            physx::PxVec3 linearAxes[3];
            if (linAllLocked) {
              linearAxes[0] = physx::PxVec3(1.0f, 0.0f, 0.0f);
              linearAxes[1] = physx::PxVec3(0.0f, 1.0f, 0.0f);
              linearAxes[2] = physx::PxVec3(0.0f, 0.0f, 1.0f);
            } else {
              linearAxes[0] = jointFrameA_dual.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
              linearAxes[1] = jointFrameA_dual.rotate(physx::PxVec3(0.0f, 1.0f, 0.0f));
              linearAxes[2] = jointFrameA_dual.rotate(physx::PxVec3(0.0f, 0.0f, 1.0f));
            }

            for (int axis = 0; axis < 3; ++axis) {
              physx::PxU32 motion = jnt.getLinearMotion(axis);
              if (motion == 2) // FREE
                continue;

              physx::PxReal Ck = posViol.dot(linearAxes[axis]);

              if (motion == 0) { // LOCKED
                // Match the primal row force f = pen*C + lambda, using the
                // pre-update solver multiplier.  The row convention is
                // C=xA-xB, so actor0's public reaction is -axis*f.
                const physx::PxReal totalForce =
                    pen * Ck + jnt.lambdaLinear[axis];
                actor0LinearForce -= linearAxes[axis] * totalForce;
                jnt.lambdaLinear[axis] = jnt.lambdaLinear[axis] * lambdaDecay +
                                         Ck * rhoDual;
              } else if (motion == 1) { // LIMITED
                physx::PxReal dist = -posViol.dot(linearAxes[axis]);
                physx::PxReal limitViol = 0.0f;
                if (dist < jnt.linearLimitLower[axis])
                  limitViol = dist - jnt.linearLimitLower[axis];
                else if (dist > jnt.linearLimitUpper[axis])
                  limitViol = dist - jnt.linearLimitUpper[axis];

                physx::PxReal newLam =
                    jnt.lambdaLinear[axis] * lambdaDecay + limitViol * rhoDual;

                if (jnt.linearLimitLower[axis] < jnt.linearLimitUpper[axis]) {
                  physx::PxReal signRef =
                      (physx::PxAbs(limitViol) > 1e-6f)
                          ? limitViol
                          : ((physx::PxAbs(jnt.lambdaLinear[axis]) > 1e-6f)
                                 ? jnt.lambdaLinear[axis]
                                 : 0.0f);
                  if (signRef > 0.0f)
                    jnt.lambdaLinear[axis] = physx::PxMax(0.0f, newLam);
                  else if (signRef < 0.0f)
                    jnt.lambdaLinear[axis] = physx::PxMin(0.0f, newLam);
                  else
                    jnt.lambdaLinear[axis] = 0.0f;
                } else {
                  jnt.lambdaLinear[axis] = newLam;
                }
              }
            }

            const bool positionDriveActive =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::LinearPositionDrive) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        CoupledLinearPositionDrive);
            if (positionDriveActive) {
              const physx::PxVec3 axis = linearAxes[0];
              const physx::PxVec3 dxA =
                  aStatic ? physx::PxVec3(0.0f)
                          : ((bodies[jnt.header.bodyIndexA].position +
                              rotA.rotate(jnt.anchorA)) -
                             (bodies[jnt.header.bodyIndexA].prevPosition +
                              bodies[jnt.header.bodyIndexA]
                                  .prevRotation.rotate(jnt.anchorA)));
              const physx::PxVec3 dxB =
                  bStatic ? physx::PxVec3(0.0f)
                          : ((bodies[jnt.header.bodyIndexB].position +
                              rotB.rotate(jnt.anchorB)) -
                             (bodies[jnt.header.bodyIndexB].prevPosition +
                              bodies[jnt.header.bodyIndexB]
                                  .prevRotation.rotate(jnt.anchorB)));
              const physx::PxReal positionError =
                  (wB - wA).dot(axis) - jnt.driveLinearPosition.x;
              const physx::PxReal velocityError =
                  (dxB - dxA).dot(axis) / dt -
                  jnt.driveLinearVelocity.x;
              const physx::PxReal driveForce = physx::PxClamp(
                  jnt.linearStiffness.x * positionError +
                      jnt.linearDamping.x * velocityError,
                  -jnt.driveLinearForce.x, jnt.driveLinearForce.x);
              actor0PositionDriveForce = axis * driveForce;
              if ((jnt.driveOutputForceFlags & 0x1u) != 0)
                actor0LinearForce += actor0PositionDriveForce;
            }

            if ((hasLockedLinearRow && !hasUnsupportedLinearRow) ||
                positionDriveActive) {
              // ConstraintWriteback stores impulses; Sc::ConstraintSim turns
              // them back into public force by multiplying by 1/dt.
              jnt.writebackLinearImpulse = actor0LinearForce * dt;
              jnt.writebackLinearImpulseValid = 1;
            }

            // Detect revolute pattern for cross-product axis alignment
            const physx::PxU32 twistMotion_d = jnt.getAngularMotion(0);
            const physx::PxU32 swing1Motion_d = jnt.getAngularMotion(1);
            const physx::PxU32 swing2Motion_d = jnt.getAngularMotion(2);
            const bool isRevolutePattern_d =
                (twistMotion_d != 0) && (swing1Motion_d == 0) &&
                (swing2Motion_d == 0);
            physx::PxVec3 actor0AngularTorque(0.0f);

            if (isRevolutePattern_d) {
              // Cross-product axis alignment dual
              physx::PxQuat worldFrameA_d = rotA * jnt.localFrameA;
              physx::PxQuat worldFrameB_d = rotB * jnt.localFrameB;
              physx::PxVec3 worldTwistA =
                  worldFrameA_d.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
              physx::PxVec3 worldTwistB =
                  worldFrameB_d.rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
              physx::PxVec3 axisViol = worldTwistA.cross(worldTwistB);

              physx::PxVec3 perp1, perp2;
              if (physx::PxAbs(worldTwistA.x) < 0.9f)
                perp1 = worldTwistA.cross(physx::PxVec3(1.0f, 0.0f, 0.0f));
              else
                perp1 = worldTwistA.cross(physx::PxVec3(0.0f, 1.0f, 0.0f));
              physx::PxReal p1Len = perp1.magnitude();
              if (p1Len > 1e-6f) perp1 *= (1.0f / p1Len);
              perp2 = worldTwistA.cross(perp1);
              physx::PxReal p2Len = perp2.magnitude();
              if (p2Len > 1e-6f) perp2 *= (1.0f / p2Len);

              physx::PxReal err1 = axisViol.dot(perp1);
              physx::PxReal err2 = axisViol.dot(perp2);

              const physx::PxReal totalTorque1 =
                  pen * err1 + jnt.lambdaAngular[1];
              const physx::PxReal totalTorque2 =
                  pen * err2 + jnt.lambdaAngular[2];
              actor0AngularTorque +=
                  perp1 * totalTorque1 + perp2 * totalTorque2;
              jnt.lambdaAngular[1] =
                  jnt.lambdaAngular[1] * lambdaDecay + err1 * rhoDual;
              jnt.lambdaAngular[2] =
                  jnt.lambdaAngular[2] * lambdaDecay + err2 * rhoDual;

              // Twist axis (0) if LIMITED
              if (twistMotion_d == 1) {
                physx::PxReal angErr =
                    jnt.computeAngularError(rotA, rotB, 0);
                physx::PxReal limitViol =
                    jnt.computeAngularLimitViolation(angErr, 0);
                physx::PxReal newLam =
                    jnt.lambdaAngular[0] * lambdaDecay + limitViol * rhoDual;

                if (jnt.angularLimitLower[0] < jnt.angularLimitUpper[0]) {
                  if (limitViol > 0.0f || jnt.lambdaAngular[0] > 0.0f)
                    jnt.lambdaAngular[0] = physx::PxMax(0.0f, newLam);
                  else if (limitViol < 0.0f || jnt.lambdaAngular[0] < 0.0f)
                    jnt.lambdaAngular[0] = physx::PxMin(0.0f, newLam);
                  else
                    jnt.lambdaAngular[0] = 0.0f;
                } else {
                  jnt.lambdaAngular[0] = newLam;
                }
              }
            } else {
              // Generic per-axis dual
              for (int axis = 0; axis < 3; ++axis) {
                if (compiledConeObjective && axis >= 1)
                  continue;
                physx::PxU32 motion = jnt.getAngularMotion(axis);
                if (motion == 2) // FREE
                  continue;

                if (motion == 0) { // LOCKED
                  physx::PxReal angErr =
                      jnt.computeAngularError(rotA, rotB, axis);
                  physx::PxVec3 localAxis(0.0f);
                  localAxis[axis] = 1.0f;
                  const physx::PxVec3 worldAxis =
                      jointFrameA_dual.rotate(localAxis);
                  const physx::PxReal totalTorque =
                      pen * angErr + jnt.lambdaAngular[axis];
                  // C = rotation(A)-rotation(B); actor0/A's public torque is
                  // the negative world row force in this AVBD convention.
                  actor0AngularTorque -= worldAxis * totalTorque;
                  jnt.lambdaAngular[axis] =
                      jnt.lambdaAngular[axis] * lambdaDecay + angErr * rhoDual;
                } else if (motion == 1) { // LIMITED
                  physx::PxReal angErr =
                      jnt.computeAngularError(rotA, rotB, axis);
                  physx::PxReal limitViol =
                      jnt.computeAngularLimitViolation(angErr, axis);
                  physx::PxReal newLam =
                      jnt.lambdaAngular[axis] * lambdaDecay +
                      limitViol * rhoDual;

                  if (jnt.angularLimitLower[axis] <
                      jnt.angularLimitUpper[axis]) {
                    if (limitViol > 0.0f || jnt.lambdaAngular[axis] > 0.0f) {
                      jnt.lambdaAngular[axis] = physx::PxMax(0.0f, newLam);
                    } else if (limitViol < 0.0f ||
                               jnt.lambdaAngular[axis] < 0.0f) {
                      jnt.lambdaAngular[axis] = physx::PxMin(0.0f, newLam);
                    } else {
                      jnt.lambdaAngular[axis] = 0.0f;
                    }
                  } else {
                    jnt.lambdaAngular[axis] = newLam;
                  }
                }
              }
            }

            if (positionDriveActive && jnt.angularMotion == 0) {
              // The finite linear drive acts at the dynamic endpoint's
              // anchor.  All angular rows are locked in this scoped island,
              // so their public reaction is the opposite lever-arm torque.
              // PxConstraint reports linear rows about bodyAWorldOffset;
              // eOUTPUT_FORCE must therefore not add another COM moment.
              const physx::PxVec3 dynamicArm =
                  aStatic
                      ? bodies[jnt.header.bodyIndexB].rotation.rotate(
                            jnt.anchorB)
                      : bodies[jnt.header.bodyIndexA].rotation.rotate(
                            jnt.anchorA);
              actor0AngularTorque =
                  -dynamicArm.cross(actor0PositionDriveForce);
            }

            const bool angularAxisVelocityDriveActive =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        AngularAxisVelocityDrive);
            const bool slerpVelocityDriveActive =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::SlerpVelocityDrive);
            if (angularAxisVelocityDriveActive ||
                slerpVelocityDriveActive) {
              physx::PxVec3 dThetaA = aStatic
                                          ? jnt.externalAngularStepA
                                          : physx::PxVec3(0.0f);
              physx::PxVec3 dThetaB = bStatic
                                          ? jnt.externalAngularStepB
                                          : physx::PxVec3(0.0f);
              if (!aStatic) {
                physx::PxQuat dqA =
                    bodies[jnt.header.bodyIndexA].rotation *
                    bodies[jnt.header.bodyIndexA]
                        .prevRotation.getConjugate();
                if (dqA.w < 0.0f)
                  dqA = -dqA;
                dThetaA =
                    physx::PxVec3(dqA.x, dqA.y, dqA.z) * 2.0f;
              }
              if (!bStatic) {
                physx::PxQuat dqB =
                    bodies[jnt.header.bodyIndexB].rotation *
                    bodies[jnt.header.bodyIndexB]
                        .prevRotation.getConjugate();
                if (dqB.w < 0.0f)
                  dqB = -dqB;
                dThetaB =
                    physx::PxVec3(dqB.x, dqB.y, dqB.z) * 2.0f;
              }
              const physx::PxVec3 worldAngularTarget =
                  jointFrameA_dual.rotate(jnt.driveAngularVelocity) * dt;
              if (slerpVelocityDriveActive) {
                // TGS emits SLERP as three fixed world rows.  The shared
                // scalar limit clamps each row independently and actor0's
                // public torque is their aggregate, independent of which
                // endpoint owns the dynamic body.
                const physx::PxReal targetScale =
                    config.angularDamping > 1e-6f
                        ? 1.0f / config.angularDamping
                        : 1.0f;
                const physx::PxVec3 residual =
                    (dThetaB - dThetaA) - worldAngularTarget * targetScale;
                const physx::PxReal scale = jnt.angularDamping.z / dt;
                physx::PxVec3 driveTorque(0.0f);
                for (int k = 0; k < 3; ++k)
                  (&driveTorque.x)[k] = physx::PxClamp(
                      scale * (&residual.x)[k],
                      -jnt.driveAngularForce.z,
                      jnt.driveAngularForce.z);
                if ((jnt.driveOutputForceFlags & (1u << 5)) != 0)
                  actor0AngularTorque += driveTorque;
              } else {
                const physx::PxU32 driveIndex =
                    compiledDriveRows == (1u << 3)
                        ? 0u
                        : (compiledDriveRows == (1u << 4) ? 1u
                                                         : 2u);
                physx::PxVec3 localDriveAxis(0.0f);
                localDriveAxis[driveIndex] = 1.0f;
                const physx::PxVec3 worldDriveAxis =
                    jointFrameA_dual.rotate(localDriveAxis);
                // TWIST/SWING use wA-wB=target.  With the solver's
                // relDW=dThetaB-dThetaA convention this is
                // C=relDW+target*dt.  Positive physical torque acts on actor0
                // along the actor-A authored drive axis, independent of which
                // endpoint is the dynamic body.
                const physx::PxReal C =
                    (dThetaB - dThetaA).dot(worldDriveAxis) +
                    worldAngularTarget.dot(worldDriveAxis);
                const physx::PxReal driveTorque = physx::PxClamp(
                    (jnt.angularDamping[driveIndex] / dt) * C,
                    -jnt.driveAngularForce[driveIndex],
                    jnt.driveAngularForce[driveIndex]);
                if ((jnt.driveOutputForceFlags &
                     (1u << (3u + driveIndex))) != 0)
                  actor0AngularTorque += worldDriveAxis * driveTorque;
              }
            }

            const bool passiveNativeReaction =
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::
                        NativePassiveReaction) ||
                hasAvbdJointObjective(
                    jnt.objectiveProgram,
                    AvbdJointObjectiveKind::CoupledFixedD6);
            if ((jnt.angularMotion == 0 &&
                 (compiledDriveRows & 0x38u) == 0) ||
                passiveNativeReaction ||
                angularAxisVelocityDriveActive ||
                slerpVelocityDriveActive) {
              jnt.writebackAngularImpulse = actor0AngularTorque * dt;
              jnt.writebackAngularImpulseValid = 1;
            }

            // --- Cone limit dual update ---
            if (compiledConeObjective) {
              physx::PxVec3 coneAxis(0.0f);
              physx::PxReal coneViol = 0.0f;
              const bool ellipticalCone =
                  computeEllipticalConeConstraint(
                      jnt, rotA, rotB, coneAxis, coneViol);
              PX_UNUSED(coneAxis);
              if (!ellipticalCone) {
                const physx::PxVec3 worldAxisA =
                    (rotA * jnt.localFrameA)
                        .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
                const physx::PxVec3 worldAxisB =
                    (rotB * jnt.localFrameB)
                        .rotate(physx::PxVec3(1.0f, 0.0f, 0.0f));
                const physx::PxReal dotAB = physx::PxClamp(
                    worldAxisA.dot(worldAxisB), -1.0f, 1.0f);
                const physx::PxReal coneAngle =
                    physx::PxAcos(dotAB);
                coneViol = coneAngle - jnt.coneAngleLimit;
              }

              // Unilateral: coneLambda -= violation * rhoDual, clamped to <= 0
              jnt.coneLambda -= coneViol * rhoDual;
              jnt.coneLambda =
                  physx::PxMax(-1e9f, physx::PxMin(0.0f, jnt.coneLambda));
            }

            // --- Drive AL dual update ---
            physx::PxReal dt2 = dt * dt;

            // Joint frame A in world space
            physx::PxQuat jointFrameA =
                aStatic
                    ? jnt.localFrameA
                    : bodies[jnt.header.bodyIndexA].rotation * jnt.localFrameA;
            physx::PxReal qMag2 = jointFrameA.magnitudeSquared();
            if (qMag2 > 1e-8f && PxIsFinite(qMag2))
              jointFrameA *= 1.0f / physx::PxSqrt(qMag2);

            // Linear velocity drive dual
            if ((compiledDriveRows & 0x7u) != 0) {
              // Body displacements from start-of-step
              const physx::PxVec3 dxA =
                  aStatic ? physx::PxVec3(0.0f)
                          : (bodies[jnt.header.bodyIndexA].position -
                             bodies[jnt.header.bodyIndexA].prevPosition);
              const physx::PxVec3 dxB =
                  bStatic ? physx::PxVec3(0.0f)
                          : (bodies[jnt.header.bodyIndexB].position -
                             bodies[jnt.header.bodyIndexB].prevPosition);

              for (int a = 0; a < 3; ++a) {
                if ((compiledDriveRows & (1u << a)) == 0)
                  continue;
                const physx::PxReal stiffness =
                    (&jnt.linearStiffness.x)[a];
                const bool usePhysicalVelocityObjective =
                    stiffness <= 0.0f &&
                    ((aStatic || bStatic) ||
                     hasAvbdJointObjective(
                         jnt.objectiveProgram,
                         AvbdJointObjectiveKind::
                             CoupledLinearVelocityDrive));
                const bool usePositionObjective =
                    a == 0 &&
                    (hasAvbdJointObjective(
                         jnt.objectiveProgram,
                         AvbdJointObjectiveKind::
                             LinearPositionDrive) ||
                     hasAvbdJointObjective(
                         jnt.objectiveProgram,
                         AvbdJointObjectiveKind::
                             CoupledLinearPositionDrive));
                if (usePhysicalVelocityObjective || usePositionObjective) {
                  // The scoped force/acceleration objective carries its mass
                  // distinction in the primal penalty, not in an AL dual.
                  (&jnt.lambdaDriveLinear.x)[a] = 0.0f;
                  continue;
                }

                const physx::PxReal damping = (&jnt.linearDamping.x)[a];
                if (damping <= 0.0f)
                  continue;

                physx::PxVec3 localAxis(0.0f);
                (&localAxis.x)[a] = 1.0f;
                const physx::PxVec3 wAxis = jointFrameA.rotate(localAxis);
                const physx::PxVec3 worldTarget =
                    jointFrameA.rotate(jnt.driveLinearVelocity) * dt;
                const physx::PxReal C =
                    (dxB.dot(wAxis) - dxA.dot(wAxis)) -
                    worldTarget.dot(wAxis);
                const physx::PxVec3 rAWorld =
                    aStatic ? physx::PxVec3(0.0f)
                            : bodies[jnt.header.bodyIndexA].rotation.rotate(
                                  jnt.anchorA);
                const physx::PxVec3 rBWorld =
                    bStatic ? physx::PxVec3(0.0f)
                            : bodies[jnt.header.bodyIndexB].rotation.rotate(
                                  jnt.anchorB);
                const AvbdSolverBody *bodyARef =
                    aStatic ? nullptr : &bodies[jnt.header.bodyIndexA];
                const AvbdSolverBody *bodyBRef =
                    bStatic ? nullptr : &bodies[jnt.header.bodyIndexB];
                physx::PxReal rhoDualDrive =
                    physx::PxMin(damping / dt2, rhoDual);
                if (jnt.isLinearAccelerationDrive(a)) {
                  const physx::PxReal driveScale =
                      computeLinearDriveRecipResponse(
                          bodyARef, bodyBRef, rAWorld, rBWorld, wAxis);
                  const physx::PxReal dampingOnly =
                      physx::PxMax(0.0f, damping - stiffness);
                  const physx::PxReal implicitScale =
                      1.0f /
                      (1.0f + dt * (dt * stiffness + dampingOnly));
                  rhoDualDrive = physx::PxMin(
                      (damping * driveScale * implicitScale) / dt2,
                      rhoDual);
                }
                (&jnt.lambdaDriveLinear.x)[a] =
                    (&jnt.lambdaDriveLinear.x)[a] * lambdaDecay +
                    rhoDualDrive * C;
              }
            }

            // Angular velocity drive dual
            if ((compiledDriveRows & 0x38u) != 0) {
              // Angular displacements from start-of-step
              physx::PxVec3 dThetaA = aStatic
                                          ? jnt.externalAngularStepA
                                          : physx::PxVec3(0.0f);
              physx::PxVec3 dThetaB = bStatic
                                          ? jnt.externalAngularStepB
                                          : physx::PxVec3(0.0f);
              if (!aStatic) {
                physx::PxQuat dqA =
                    bodies[jnt.header.bodyIndexA].rotation *
                    bodies[jnt.header.bodyIndexA].prevRotation.getConjugate();
                if (dqA.w < 0.0f)
                  dqA = -dqA;
                dThetaA = physx::PxVec3(dqA.x, dqA.y, dqA.z) * 2.0f;
              }
              if (!bStatic) {
                physx::PxQuat dqB =
                    bodies[jnt.header.bodyIndexB].rotation *
                    bodies[jnt.header.bodyIndexB].prevRotation.getConjugate();
                if (dqB.w < 0.0f)
                  dqB = -dqB;
                dThetaB = physx::PxVec3(dqB.x, dqB.y, dqB.z) * 2.0f;
              }

              physx::PxVec3 relDW = dThetaB - dThetaA;
              physx::PxVec3 worldAngTarget =
                  jointFrameA.rotate(jnt.driveAngularVelocity) * dt;

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
                      AvbdJointObjectiveKind::
                          OrdinaryD6SlerpDrive);
              if (slerpDrive) {
                const bool usePhysicalSlerpVelocityObjective =
                    hasAvbdJointObjective(
                        jnt.objectiveProgram,
                        AvbdJointObjectiveKind::SlerpVelocityDrive);
                const bool usePhysicalSlerpPositionObjective =
                    hasAvbdJointObjective(
                        jnt.objectiveProgram,
                        AvbdJointObjectiveKind::SlerpPositionDrive);
                const bool useCoupledAngularPositionObjective =
                    hasAvbdJointObjective(
                        jnt.objectiveProgram,
                        AvbdJointObjectiveKind::
                            CoupledAngularPositionDrive);
                if (usePhysicalSlerpVelocityObjective ||
                    usePhysicalSlerpPositionObjective ||
                    useCoupledAngularPositionObjective) {
                  // The force-mode objective is solved directly in the
                  // primal path.  An AL multiplier would bypass its authored
                  // per-row torque limit on later iterations.
                  jnt.lambdaDriveAngular = physx::PxVec3(0.0f);
                } else {
                  physx::PxReal damping =
                    jnt.angularDamping.z; // SLERP uses Z damping slot
                  if (damping > 0.0f) {
                    const AvbdSolverBody *bodyARef =
                      aStatic ? nullptr : &bodies[jnt.header.bodyIndexA];
                    const AvbdSolverBody *bodyBRef =
                      bStatic ? nullptr : &bodies[jnt.header.bodyIndexB];
                    physx::PxReal rhoDualDrive = physx::PxMin(damping / dt2, rhoDual);
                    if (jnt.isAngularAccelerationDrive(2)) {
                    const physx::PxReal driveScale =
                      computeAngularDriveRecipResponse(bodyARef, bodyBRef,
                                       physx::PxVec3(1.0f, 0.0f, 0.0f));
                    const physx::PxReal stiffness = jnt.angularStiffness.z;
                    const physx::PxReal dampingOnly = physx::PxMax(0.0f, damping - stiffness);
                    const physx::PxReal implicitScale =
                      1.0f / (1.0f + dt * (dt * stiffness + dampingOnly));
                    rhoDualDrive = physx::PxMin((damping * driveScale * implicitScale) / dt2,
                                  rhoDual);
                    }
                    for (int k = 0; k < 3; ++k) {
                      physx::PxReal C = (&relDW.x)[k] - (&worldAngTarget.x)[k];
                      (&jnt.lambdaDriveAngular.x)[k] =
                          (&jnt.lambdaDriveAngular.x)[k] * lambdaDecay +
                          rhoDualDrive * C;
                    }
                  }
                }
              } else {
                struct AxisDrive {
                  int bit;
                  int dampIdx;
                  physx::PxVec3 localAxis;
                };
                const AxisDrive axes[3] = {
                    {3, 0, physx::PxVec3(1.0f, 0.0f, 0.0f)},
                    {4, 1, physx::PxVec3(0.0f, 1.0f, 0.0f)},
                    {5, 2, physx::PxVec3(0.0f, 0.0f, 1.0f)},
                };

                for (int a = 0; a < 3; ++a) {
                  if ((compiledDriveRows &
                       (1u << axes[a].bit)) == 0)
                    continue;
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
                  const bool useCoupledAngularPositionObjective =
                      hasAvbdJointObjective(
                          jnt.objectiveProgram,
                          AvbdJointObjectiveKind::
                              CoupledAngularPositionDrive);
                  if (usePhysicalAngularAxisVelocityObjective ||
                      usePhysicalAngularPositionObjective ||
                      useCoupledAngularPositionObjective) {
                    // The physical force-mode objective is solved directly
                    // in the primal path; carrying an AL multiplier would
                    // add non-physical torque on top of the authored limit.
                    (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] = 0.0f;
                    continue;
                  }
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
          const AvbdSolverBody *bodyARef =
            aStatic ? nullptr : &bodies[jnt.header.bodyIndexA];
          const AvbdSolverBody *bodyBRef =
            bStatic ? nullptr : &bodies[jnt.header.bodyIndexB];
                  // PhysX TGS convention: Twist/Swing target velocities are
                  // applied as (wA - wB), meaning wB - wA = -target. SLERP is
                  // applied as wB - wA = target, which is handled above.
                  physx::PxReal targetOmega_dt = -worldAngTarget.dot(wAxis);
                  physx::PxReal C = relDW.dot(wAxis) - targetOmega_dt;

                  physx::PxReal rhoDualDrive =
                      physx::PxMin(damping / dt2, rhoDual);
                  if (isAccelerationDrive) {
                    const physx::PxReal driveScale =
                        computeAngularDriveRecipResponse(bodyARef, bodyBRef,
                                                         wAxis);
                    const physx::PxReal implicitScale =
                        1.0f / (1.0f + dt * effectiveRate);
                    rhoDualDrive = physx::PxMin(
                        driveScale * implicitScale * effectiveRate, rhoDual);
                  }
                  (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] =
                      (&jnt.lambdaDriveAngular.x)[axes[a].dampIdx] *
                          lambdaDecay +
                      rhoDualDrive * C;
                }
              }
            }
          }
        }

        // Gear joints: AL dual
        for (physx::PxU32 j = 0; j < numGear; ++j)
          updateGearJointMultiplier(gearJoints[j], bodies, numBodies, config);

        // Soft body AVBD dual update (penalty growth only)
        if (numSoftParticles > 0 && numSoftBodies > 0) {
          PX_PROFILE_ZONE("AVBD.softDual", 0);
          updateSoftDual(softParticles, numSoftParticles, bodies, numBodies,
                         softBodies, numSoftBodies, softContacts, numSoftContacts,
                         config.avbdBeta);
        }
}

} // namespace Dy
} // namespace physx
