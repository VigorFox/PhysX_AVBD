// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#include "avbd/solver/post_al/DyAvbdPostAlContactResponse.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx {
namespace Dy {

//=============================================================================
// Body-static normal depenetration (TGS-style capped geometric projection)
//=============================================================================
void applyBodyStaticNormalDepenetrationSweeps(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdContactConstraint *contacts, physx::PxU32 numContacts,
    const physx::PxVec3 &gravity, physx::PxReal dt, physx::PxU32 sweeps,
    const physx::PxArray<bool> *skipDepenForBodies,
    physx::PxArray<physx::PxU8> *deformableNormalStageMask,
    physx::PxReal configLengthScale, AvbdSolverStats *stats) {
  (void)stats;
  if (numContacts == 0 || numBodies == 0 || dt <= 0.0f || sweeps == 0)
    return;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    bool anyCorrection = false;
    for (physx::PxU32 c = 0; c < numContacts; ++c) {
      const physx::PxU32 bA = contacts[c].header.bodyIndexA;
      const physx::PxU32 bB = contacts[c].header.bodyIndexB;
      if (!isBodyVsStaticContact(bA, bB, numBodies))
        continue;
      // Ordinary rigid-static normals are owned completely by the AVBD
      // primal/dual manifold. A geometric edit here would invalidate the
      // just-updated multiplier. Only the deformable-anchor recovery path is
      // retained until it is absorbed by the unified OGC contact solve.
      if (!hasDeformableStaticAnchor(contacts[c]))
        continue;

      const bool dynIsA = (bA < numBodies);
      const bool dynIsB = (bB < numBodies);
      if (dynIsA == dynIsB)
        continue;
      const physx::PxU32 bi = dynIsA ? bA : bB;
      const physx::PxReal linearResponseScale =
          dynIsA ? contacts[c].invMassScaleA
                 : contacts[c].invMassScaleB;
      if (linearResponseScale <= 0.0f)
        continue;
      // A finite contact impulse cannot also receive an unbounded split-pose
      // correction.  Let the capped AL force determine the pose response so
      // insufficient authored support can pass through, matching PhysX/TGS.
      if (contacts[c].maxImpulse < PX_MAX_REAL)
        continue;
      if (skipDepenForBodies && bi < skipDepenForBodies->size() &&
          (*skipDepenForBodies)[bi] &&
          hasDeformableStaticAnchor(contacts[c]))
        continue;
      AvbdSolverBody &body = bodies[bi];

      physx::PxVec3 worldA, worldB;
      if (dynIsA) {
        worldA = body.position + body.rotation.rotate(contacts[c].contactPointA);
        worldB = contacts[c].contactPointB;
      } else {
        worldA = contacts[c].contactPointA;
        worldB = body.position + body.rotation.rotate(contacts[c].contactPointB);
      }

      physx::PxReal violation =
          (worldA - worldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      const bool deformableAnchor =
          hasDeformableStaticAnchor(contacts[c]);
      if (deformableAnchor)
        violation = finalizeBodyVsStaticViolation(violation,
                                                contacts[c].penetrationDepth);
      const physx::PxReal lengthScale =
          physx::PxMax(configLengthScale, 1e-6f);
      if (violation >= -1e-5f * lengthScale)
        continue;
      const physx::PxVec3 initialWorldA =
          dynIsA
              ? body.prevPosition +
                    body.prevRotation.rotate(contacts[c].contactPointA)
              : contacts[c].staticPrevWorldPoint;
      const physx::PxVec3 initialWorldB =
          dynIsA
              ? contacts[c].staticPrevWorldPoint
              : body.prevPosition +
                    body.prevRotation.rotate(contacts[c].contactPointB);
      const physx::PxReal initialViolation =
          (initialWorldA - initialWorldB).dot(contacts[c].contactNormal) +
          contacts[c].penetrationDepth;
      const bool deepInitialViolation =
          initialViolation <
              -AvbdConstants::AVBD_BODY_STATIC_NEAR_SURFACE * lengthScale;
      // The retained normal AL row owns uninterrupted shallow support for
      // both rigid and deformable static anchors. Split-pose recovery is an
      // onset/deep-overlap emergency, never a second steady-support owner.
      if (contacts[c].persistentPointMatched &&
          !deepInitialViolation)
        continue;

      const physx::PxReal approachSpeed =
          body.linearVelocity.magnitude() + gravity.magnitude() * dt;
      physx::PxReal sweepCap =
          physx::PxMax(approachSpeed * dt * 0.5f, 0.01f * lengthScale);
      if (deformableAnchor) {
        const physx::PxVec3 staticNow = dynIsA ? worldB : worldA;
        const physx::PxVec3 meshStep =
            staticNow - contacts[c].staticPrevWorldPoint;
        // Mesh step + deeper floor: prevent multi-cycle trough sink when the
        // heaving surface rises into resting stacks (was capped too soft).
        sweepCap = physx::PxMax(sweepCap, meshStep.magnitude() * 1.5f);
        sweepCap = physx::PxMax(sweepCap, 0.04f * lengthScale);
        if (violation < -0.05f * lengthScale)
          sweepCap = physx::PxMax(sweepCap, -violation * 0.6f);
      }
      const physx::PxReal corr = physx::PxMin(-violation, sweepCap);
      if (dynIsA)
        body.position += contacts[c].contactNormal * corr;
      else
        body.position -= contacts[c].contactNormal * corr;
      if (deformableAnchor) {
        if (deformableNormalStageMask &&
            c < deformableNormalStageMask->size())
          (*deformableNormalStageMask)[c] |= 2u;
        PX_AVBD_PROFILE_STAT(stats->surfaceDeformableDepenetrationCorrections++);
      }
      anyCorrection = true;
    }
    if (!anyCorrection)
      break;
  }

}

//=============================================================================
// Sequential body-static friction fallback (rigid static partners and deformable
// rows excluded from the position-level tangent owner)
//
// TGS-style projected Gauss-Seidel friction, decoupled from the AVBD block
// solve. Rigid plane: all corner contacts per sweep. Unsupported deformable
// rows retain the legacy dominant-contact fallback; position-owned deformable
// tangents are skipped here.
//=============================================================================
void applyBodyStaticFrictionSweeps(AvbdSolverBody *bodies,
                                               physx::PxU32 numBodies,
                                               AvbdContactConstraint *contacts,
                                               physx::PxU32 numContacts,
                                               const physx::PxVec3 &gravity,
                                               physx::PxReal dt,
                                               physx::PxU32 sweeps,
                                               const physx::PxArray<physx::PxVec3> *velSeedPos,
                                               const physx::PxArray<physx::PxQuat> *velSeedRot,
                                               const physx::PxArray<bool> *skipForBodies,
                                               physx::PxReal configLengthScale,
                                               AvbdBodyStaticFrictionWorkspace &workspace,
                                               AvbdSolverStats *stats) {
  if (numContacts == 0 || numBodies == 0 || dt <= 0.0f || sweeps == 0)
    return;

  const physx::PxReal invDt = 1.0f / dt;

  // Deformable anchors: one dominant contact per body (multiple mesh rows
  // over-constrain tangential DOF). Rigid static partners: all contacts in
  // sequential GS. Raw deformable contact counts gate mesh-velocity tracking.
  physx::PxArray<physx::PxU32> &dominantDeformable =
      workspace.dominantDeformable;
  physx::PxArray<physx::PxU32> &bodyDeformRawCount =
      workspace.bodyDeformRawCount;
  dominantDeformable.resize(numBodies);
  bodyDeformRawCount.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    dominantDeformable[i] = 0xFFFFFFFFu;
    bodyDeformRawCount[i] = 0;
  }
  physx::PxArray<physx::PxU32> &frContacts = workspace.contactIndices;
  physx::PxArray<physx::PxU32> &bodyContactCount =
      workspace.bodyContactCount;
  physx::PxArray<physx::PxReal> &bodyContactNormalSum =
      workspace.bodyContactNormalSum;
  frContacts.clear();
  bodyContactCount.resize(numBodies);
  bodyContactNormalSum.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    bodyContactCount[i] = 0;
    bodyContactNormalSum[i] = 0.0f;
  }
  for (physx::PxU32 c = 0; c < numContacts; ++c) {
    const AvbdContactConstraint &cc = contacts[c];
    if (cc.friction <= 0.0f && cc.staticFriction <= 0.0f)
      continue;
    const bool dynA = cc.header.bodyIndexA < numBodies;
    const bool dynB = cc.header.bodyIndexB < numBodies;
    if (dynA == dynB)
      continue;
    if (!isBodyVsStaticContact(cc.header.bodyIndexA, cc.header.bodyIndexB,
                               numBodies))
      continue;
    // This fallback may consume only rows explicitly compiled for it. A
    // missing/invalid owner is fail-closed, never implicit permission to run a
    // second pose solver after PositionAL has finalized its multiplier.
    if (!hasVelocityBodyStaticFrictionSweepOwner(cc))
      continue;
    const physx::PxU32 bi = dynA ? cc.header.bodyIndexA : cc.header.bodyIndexB;
    if (hasDeformableStaticAnchor(cc) && skipForBodies &&
        bi < skipForBodies->size() && (*skipForBodies)[bi])
      continue;
    if (hasDeformableStaticAnchor(cc)) {
      bodyDeformRawCount[bi]++;
      const physx::PxU32 cur = dominantDeformable[bi];
      if (cur == 0xFFFFFFFFu ||
          physx::PxAbs(cc.header.lambda) >
              physx::PxAbs(contacts[cur].header.lambda))
        dominantDeformable[bi] = c;
    } else {
      frContacts.pushBack(c);
      bodyContactCount[bi]++;
      bodyContactNormalSum[bi] += physx::PxAbs(cc.header.lambda);
    }
  }
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (dominantDeformable[i] != 0xFFFFFFFFu) {
      frContacts.pushBack(dominantDeformable[i]);
      bodyContactCount[i] = 1;
      bodyContactNormalSum[i] =
          physx::PxAbs(contacts[dominantDeformable[i]].header.lambda);
    }
  }
  if (frContacts.empty())
    return;

  // Work on a separate velocity field seeded from this step's pose change, so
  // sweeps never feed position back into themselves (that caused divergence on
  // stacks where one base box carries several mesh contacts). The friction-only
  // velocity delta is converted to a tangential pose shift at the very end,
  // leaving the block solve's normal penetration resolution intact.
  physx::PxArray<physx::PxVec3> &vLin = workspace.linearVelocity;
  physx::PxArray<physx::PxVec3> &vAng = workspace.angularVelocity;
  physx::PxArray<physx::PxVec3> &vLin0 = workspace.initialLinearVelocity;
  physx::PxArray<physx::PxVec3> &vAng0 = workspace.initialAngularVelocity;
  physx::PxArray<bool> &touched = workspace.touched;
  physx::PxArray<physx::PxReal> &bodySpeed = workspace.bodySpeed;
  vLin.resize(numBodies);
  vAng.resize(numBodies);
  vLin0.resize(numBodies);
  vAng0.resize(numBodies);
  touched.resize(numBodies);
  bodySpeed.resize(numBodies);
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    touched[i] = false;
    bodySpeed[i] = 0.0f;
    if (bodies[i].invMass <= 0.0f) {
      vLin[i] = vAng[i] = vLin0[i] = vAng0[i] = physx::PxVec3(0.0f);
      continue;
    }
    const physx::PxVec3 seedPos =
        velSeedPos && i < velSeedPos->size() ? (*velSeedPos)[i] : bodies[i].position;
    const physx::PxQuat seedRot =
        velSeedRot && i < velSeedRot->size() ? (*velSeedRot)[i] : bodies[i].rotation;
    physx::PxVec3 vl = (seedPos - bodies[i].prevPosition) * invDt;
    physx::PxQuat dq = seedRot * bodies[i].prevRotation.getConjugate();
    if (dq.w < 0.0f)
      dq = -dq;
    physx::PxVec3 va = physx::PxVec3(dq.x, dq.y, dq.z) * (2.0f * invDt);
    vLin[i] = vLin0[i] = vl;
    vAng[i] = vAng0[i] = va;
    bodySpeed[i] = vl.magnitude() + va.magnitude() * 0.5f;
  }

  // Resting weight floor only when quasi-static. Impact / ball-shot must use
  // dual normal force alone - m*g floors glued HelloWorld boxes and killed ball KE.
  const physx::PxReal lengthScale =
      physx::PxMax(configLengthScale, 1e-6f);
  const physx::PxReal restSpeed = 1.5f * lengthScale;

  for (physx::PxU32 sweep = 0; sweep < sweeps; ++sweep) {
    for (physx::PxU32 fi = 0; fi < frContacts.size(); ++fi) {
      AvbdContactConstraint &cc = contacts[frContacts[fi]];
      const bool dynIsA = cc.header.bodyIndexA < numBodies;
      const physx::PxU32 bi = dynIsA ? cc.header.bodyIndexA : cc.header.bodyIndexB;
      AvbdSolverBody &body = bodies[bi];
      const physx::PxReal linearResponseScale =
          dynIsA ? cc.invMassScaleA : cc.invMassScaleB;
      const physx::PxReal angularResponseScale =
          dynIsA ? cc.invInertiaScaleA : cc.invInertiaScaleB;
      if (linearResponseScale <= 0.0f &&
          angularResponseScale <= 0.0f)
        continue;
      touched[bi] = true;

      const physx::PxVec3 cpLocal = dynIsA ? cc.contactPointA : cc.contactPointB;
      const physx::PxVec3 r = body.rotation.rotate(cpLocal);
      const physx::PxReal contactInvMass =
          body.invMass * linearResponseScale;
      const physx::PxMat33 contactInvI =
          body.invInertiaWorld * angularResponseScale;

      physx::PxVec3 worldA, worldB;
      if (dynIsA) {
        worldA = body.position + r;
        worldB = cc.contactPointB;
      } else {
        worldA = cc.contactPointA;
        worldB = body.position + r;
      }
      physx::PxReal viol =
          (worldA - worldB).dot(cc.contactNormal) + cc.penetrationDepth;
      if (hasDeformableStaticAnchor(cc))
        viol = finalizeBodyVsStaticViolation(viol, cc.penetrationDepth);

      // Mesh target velocity via SupportClass policy (solve-loop contract).
      // eRigidPlane / eDeformableMultiCorner -> vMesh=0; few-contact ride on.
      physx::PxVec3 vMesh(0.0f);
      if (cc.supportClass == AvbdSupportClass::eUnset) {
        if (hasDeformableStaticAnchor(cc)) {
          const physx::PxReal mass =
              (body.invMass > 1e-8f) ? (1.0f / body.invMass) : 1e8f;
          if (bodyDeformRawCount[bi] >=
                  AvbdConstants::AVBD_SUPPORT_MULTI_CORNER_MIN &&
              mass >= AvbdConstants::AVBD_SUPPORT_MULTI_CORNER_MASS)
            cc.supportClass = AvbdSupportClass::eDeformableMultiCorner;
          else
            cc.supportClass = AvbdSupportClass::eDeformableFewContact;
        } else {
          cc.supportClass = AvbdSupportClass::eRigidPlane;
        }
      }
      if (cc.supportClass == AvbdSupportClass::eDeformableFewContact ||
          cc.supportClass == AvbdSupportClass::eShell) {
        const physx::PxVec3 staticNow = dynIsA ? worldB : worldA;
        physx::PxVec3 vFull = (staticNow - cc.staticPrevWorldPoint) * invDt;
        const physx::PxReal stepCap = AvbdConstants::AVBD_SURFACE_STEP_ALIAS_M;
        if ((staticNow - cc.staticPrevWorldPoint).magnitudeSquared() >
            stepCap * stepCap) {
          vFull = physx::PxVec3(0.0f);
        }
        const physx::PxVec3 &n = cc.contactNormal;
        vMesh = vFull - n * vFull.dot(n);
        const physx::PxReal vCap = AvbdConstants::AVBD_SURFACE_VMESH_CAP;
        const physx::PxReal vMag2 = vMesh.magnitudeSquared();
        if (vMag2 > vCap * vCap)
          vMesh *= vCap / physx::PxSqrt(vMag2);
      }

      // Normal force from dual / penalty depth only by default.
      physx::PxReal contactN = physx::PxMax(
          physx::PxAbs(cc.header.lambda),
          cc.header.penalty * physx::PxMax(0.0f, -viol));

      // Soft shared m*g fill only when resting (not under ball impact).
      if (body.invMass > 1e-8f && bodySpeed[bi] < restSpeed &&
          viol <= 0.05f * lengthScale) {
        const physx::PxReal weight =
            (1.0f / body.invMass) * gravity.magnitude() /
            physx::PxReal(physx::PxMax(1u, bodyContactCount[bi]));
        contactN = physx::PxMax(contactN, weight);
      }

      // Velocity-level friction is dynamic ?; static ? is for dual stick only.
      const physx::PxReal mu =
          cc.friction > 0.0f ? cc.friction
                             : (cc.staticFriction > 0.0f ? cc.staticFriction
                                                         : 0.0f);
      const physx::PxReal jmax = contactN * mu * dt;
      if (jmax <= 0.0f)
        continue;

      const physx::PxVec3 tangents[2] = {cc.tangent0, cc.tangent1};
      physx::PxReal jUnc[2] = {0.0f, 0.0f};
      physx::PxReal kEff[2] = {0.0f, 0.0f};
      physx::PxVec3 rCrossT[2];
      for (physx::PxU32 a = 0; a < 2; ++a) {
        const physx::PxVec3 &t = tangents[a];
        rCrossT[a] = r.cross(t);
        kEff[a] =
            contactInvMass + rCrossT[a].dot(contactInvI * rCrossT[a]);
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxVec3 dynamicTargetVelocity =
            cc.targetVelocity * (dynIsA ? 1.0f : -1.0f);
        const physx::PxVec3 vRel =
            (vLin[bi] + vAng[bi].cross(r)) - vMesh -
            dynamicTargetVelocity;
        jUnc[a] = -vRel.dot(t) / kEff[a];
      }
      avbdProjectImpulseCone(jmax, jUnc[0], jUnc[1]);
      if (PX_AVBD_ENABLE_SOLVER_PROFILE && stats &&
          hasDeformableStaticAnchor(cc) &&
          (jUnc[0] * jUnc[0] + jUnc[1] * jUnc[1]) > 1.0e-16f)
        PX_AVBD_PROFILE_STAT(stats->surfaceDeformableFrictionCorrections++);
      for (physx::PxU32 a = 0; a < 2; ++a) {
        if (kEff[a] <= 1e-12f)
          continue;
        const physx::PxReal j = jUnc[a];
        vLin[bi] += tangents[a] * (j * contactInvMass);
        vAng[bi] += contactInvI * (rCrossT[a] * j);
        // Public PxContactPair friction impulses use the impulse applied to
        // contact body A. The sweep updates whichever endpoint is dynamic, so
        // flip the recorded direction when that endpoint is body B.
        const physx::PxReal reportSign = dynIsA ? 1.0f : -1.0f;
        cc.frictionSweepImpulse += tangents[a] * (j * reportSign);
      }
    }
  }

  // Apply only the friction-induced velocity delta as a tangential pose shift.
  for (physx::PxU32 i = 0; i < numBodies; ++i) {
    if (!touched[i] || bodies[i].invMass <= 0.0f)
      continue;
    const physx::PxVec3 dPos = (vLin[i] - vLin0[i]) * dt;
    bodies[i].position += dPos;
    const physx::PxVec3 dTheta = (vAng[i] - vAng0[i]) * dt;
    if (dTheta.magnitudeSquared() > 1e-16f) {
      physx::PxQuat dqi(dTheta.x, dTheta.y, dTheta.z, 0.0f);
      bodies[i].rotation =
          (bodies[i].rotation + dqi * bodies[i].rotation * 0.5f).getNormalized();
    }
  }
}

} // namespace Dy
} // namespace physx
