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

#ifndef DY_AVBD_SOFT_BODY_H
#define DY_AVBD_SOFT_BODY_H

// =============================================================================
// AVBD Soft Body -- Internal header
//
// Includes the public PxAvbdSoftBody.h (all portable types and evaluators)
// and adds internal-only functions that depend on AvbdSolverBody,
// AvbdBlock6x6, AvbdVec6 (from DyAvbdTypes.h).
// =============================================================================

#include "PxAvbdSoftBody.h"
#include "DyAvbdConstraint.h"
#include "DyAvbdTypes.h"

namespace physx
{
namespace Dy
{

// =============================================================================
// Internal-only: rigid-soft coupling evaluators (need AvbdSolverBody)
// =============================================================================

PX_FORCE_INLINE void avbdEvaluateAttachmentForceHessian_particle(
	const AvbdSoftAttachment& ac,
	const AvbdSoftParticle* particles,
	const AvbdSolverBody* rigidBodies,
	PxVec3& outForce, PxMat33& outHessian)
{
	const AvbdSolverBody& rb = rigidBodies[ac.rigidBodyIdx];
	PxVec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
	PxVec3 C = particles[ac.particleIdx].position - worldAnchor;
	outForce = C * (-ac.k);
	outHessian = PxMat33::createDiagonal(PxVec3(ac.k));
}

// Attachment penalty contribution to rigid body (uses AVBD 6x6 system)
PX_FORCE_INLINE void avbdAddAttachmentContribution_rigid(
	const AvbdSoftAttachment& ac,
	PxU32 bodyIdx,
	const AvbdSoftParticle* particles,
	const AvbdSolverBody* rigidBodies,
	PxReal dt,
	AvbdBlock6x6& lhs, AvbdVec6& rhs)
{
	if (bodyIdx != ac.rigidBodyIdx) return;

	PX_UNUSED(dt);

	const AvbdSolverBody& rb = rigidBodies[bodyIdx];
	PxVec3 worldOffset = rb.rotation.rotate(ac.localOffset);
	PxVec3 worldAnchor = rb.position + worldOffset;
	PxVec3 C = particles[ac.particleIdx].position - worldAnchor;

	PxVec3 fLin = C * (-ac.k);
	PxVec3 fAng = worldOffset.cross(fLin);

	rhs.linear += fLin;
	rhs.angular += fAng;

	lhs.linearLinear += PxMat33::createDiagonal(PxVec3(ac.k));

	PxMat33 sk = avbdSkew(worldOffset);
	PxMat33 skTsk = sk.getTranspose() * sk;
	lhs.angularAngular += skTsk * ac.k;

	PxMat33 offDiag = sk * (-ac.k);
	lhs.linearAngular += offDiag;
	lhs.angularLinear += offDiag.getTranspose();
}

// =============================================================================
// Internal-only: dual update for attachments (needs AvbdSolverBody)
// =============================================================================

PX_FORCE_INLINE void avbdUpdateAttachmentDual(
	AvbdSoftAttachment& ac,
	const AvbdSoftParticle* particles,
	const AvbdSolverBody* rigidBodies,
	PxReal beta)
{
	const AvbdSolverBody& rb = rigidBodies[ac.rigidBodyIdx];
	PxVec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
	PxReal C_lin = (particles[ac.particleIdx].position - worldAnchor).magnitude();
	ac.k = PxMin(ac.k + beta * C_lin, ac.kMax);
}

// =============================================================================
// Internal-only: rigid-soft contact detection stub (needs AvbdSolverBody)
// =============================================================================

inline void avbdDetectSoftRigidContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSolverBody* rigidBodies, PxU32 numRigidBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.02f)
{
	PX_UNUSED(margin);

	for (PxU32 pi = 0; pi < numParticles; pi++)
	{
		if (particles[pi].invMass <= 0.0f) continue;

		for (PxU32 bi = 0; bi < numRigidBodies; bi++)
		{
			const AvbdSolverBody& rb = rigidBodies[bi];
			if (rb.isStatic()) continue;

			// Stub -- in PhysX integration, contacts come from broadphase/narrowphase
			PX_UNUSED(rb);
		}
	}
	PX_UNUSED(contacts);
}

// =============================================================================
// Kinematic shell: rigid 6x6 row (standalone addKinematicShellContactContribution_rigid)
// =============================================================================

PX_FORCE_INLINE PxReal avbdKinematicShellContactViolation(
	const AvbdSoftContact& sc, const AvbdSolverBody& body) {
	const PxVec3 rAw = body.rotation.rotate(sc.rigidLocalPoint);
	const PxVec3 wA = body.position + rAw;
	PxReal geom = (wA - sc.surfacePoint).dot(sc.normal) - sc.depth;
	if (geom < 0.0f)
		geom = PxMin(geom, -sc.depth);
	return geom;
}

PX_FORCE_INLINE void avbdAddKinematicShellContactContribution_rigid(
	const AvbdSoftContact& sc, PxU32 bodyIdx, const AvbdSolverBody& body,
	PxReal boostFloor, AvbdBlock6x6& lhs, AvbdVec6& rhs) {
	if (sc.rigidBodyIdx != bodyIdx)
		return;

	const PxVec3 n = sc.normal;
	const PxVec3 rAw = body.rotation.rotate(sc.rigidLocalPoint);
	const PxReal geom = avbdKinematicShellContactViolation(sc, body);
	const PxVec3 gradLin = n;
	const PxVec3 gradAng = rAw.cross(n);
	const PxReal pen = PxMax(sc.k, boostFloor);
	const PxReal f = PxMin(0.0f, pen * geom + sc.alLambda);
	rhs.linear += gradLin * f;
	rhs.angular += gradAng * f;
	lhs.addConstraintContribution(gradLin, gradAng, pen);
}

PX_FORCE_INLINE void avbdComputeKinematicShellTangentViolation(
	const AvbdSoftContact& sc, const AvbdSolverBody& body, PxReal Ctangent[2])
{
	const PxVec3 rAw = body.rotation.rotate(sc.rigidLocalPoint);
	const PxVec3 worldA = body.position + rAw;
	const PxVec3 prevWorldA =
		body.prevPosition + body.prevRotation.rotate(sc.rigidLocalPoint);
	const PxVec3 staticMotion = sc.surfacePoint - sc.surfacePointPrev;
	const PxVec3 relDisp = (worldA - prevWorldA) - staticMotion;

	PxQuat deltaQ = body.rotation * body.prevRotation.getConjugate();
	if (deltaQ.w < 0.0f)
		deltaQ = -deltaQ;
	const PxVec3 dw(deltaQ.x, deltaQ.y, deltaQ.z);
	const PxVec3 angDisp = dw * 2.0f;

	Ctangent[0] = relDisp.dot(sc.tangent1) + angDisp.dot(rAw.cross(sc.tangent1));
	Ctangent[1] = relDisp.dot(sc.tangent2) + angDisp.dot(rAw.cross(sc.tangent2));
}

PX_FORCE_INLINE void avbdUpdateKinematicShellContactDual(
	AvbdSoftContact& sc, const AvbdSolverBody& body, PxReal beta, PxReal penaltyMax)
{
	const PxReal Cn = avbdKinematicShellContactViolation(sc, body);
	const PxReal rawLambdaN = sc.k * Cn + sc.alLambda;
	sc.alLambda = PxMin(0.0f, rawLambdaN);
	if (sc.alLambda < 0.0f)
		sc.k = PxMin(sc.k + beta * PxAbs(Cn), sc.ke);

	PxReal Ctangent[2];
	avbdComputeKinematicShellTangentViolation(sc, body, Ctangent);
	// Coulomb cone on shell tangents (bound from normal force F_n = alLambda).
	PxReal Fn = sc.alLambda;
	PxReal Ft0 = 0.0f, Ft1 = 0.0f;
	const PxReal preLen = avbdEvaluateContactForcesCone(
		0.0f, 0.0f, sc.alLambda, sc.penTangent[0], Ctangent[0],
		sc.alLambdaTangent[0], sc.penTangent[1], Ctangent[1],
		sc.alLambdaTangent[1], sc.friction, Fn, Ft0, Ft1);
	sc.alLambdaTangent[0] = Ft0;
	sc.alLambdaTangent[1] = Ft1;
	const PxReal bounds = PxAbs(sc.alLambda) * sc.friction;
	if (preLen <= bounds)
	{
		sc.penTangent[0] =
			PxMin(sc.penTangent[0] + beta * PxAbs(Ctangent[0]), penaltyMax);
		sc.penTangent[1] =
			PxMin(sc.penTangent[1] + beta * PxAbs(Ctangent[1]), penaltyMax);
	}
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_H
