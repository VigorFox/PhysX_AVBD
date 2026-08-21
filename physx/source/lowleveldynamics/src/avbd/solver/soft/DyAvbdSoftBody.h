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
// Includes the private portable soft-body component and adds functions that
// depend on AvbdSolverBody, AvbdBlock6x6, AvbdVec6 (from DyAvbdTypes.h).
// =============================================================================

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/solver/rigid/DyAvbdSolverBody.h"
#include "avbd/core/DyAvbdTypes.h"

namespace physx
{
namespace Dy
{

// =============================================================================
// Internal-only: dual update for attachments (needs AvbdSolverBody)
// =============================================================================

PX_FORCE_INLINE void avbdUpdateAttachmentDual(
	AvbdSoftAttachment& ac,
	const AvbdSoftPoint& point,
	const AvbdSoftParticle* particles,
	const AvbdSolverBody* rigidBodies,
	PxReal beta)
{
	const AvbdSolverBody& rb = rigidBodies[ac.rigidBodyIdx];
	PxVec3 worldAnchor = rb.position + rb.rotation.rotate(ac.localOffset);
	const PxVec3 C =
		avbdGetSoftPointPosition(point, particles) - worldAnchor;
	ac.alLambda += C * ac.k;
	const PxReal C_lin = C.magnitude();
	ac.k = PxMin(ac.k + beta * C_lin, ac.kMax);
}

struct AvbdSoftRigidAttachmentCoupledStep
{
	PxVec3 constraint;
	PxVec3 multiplier;
	PxVec3 worldOffset;
	PxVec3 particleCorrections[4];
	PxVec3 rigidLinearCorrection;
	PxVec3 rigidAngularCorrection;

	AvbdSoftRigidAttachmentCoupledStep()
		: constraint(0.0f), multiplier(0.0f),
		  worldOffset(0.0f),
		  rigidLinearCorrection(0.0f),
		  rigidAngularCorrection(0.0f)
	{
		for(PxU32 i = 0; i < 4; i++)
			particleCorrections[i] = PxVec3(0.0f);
	}
};

PX_FORCE_INLINE bool avbdEvaluateSoftRigidAttachmentCoupledStep(
	const AvbdSoftAttachment& attachment,
	const AvbdSoftPoint& point,
	const AvbdSoftParticle* particles,
	PxU32 numParticles,
	const AvbdSolverBody& rigidBody,
	PxReal dt,
	AvbdSoftRigidAttachmentCoupledStep& step)
{
	const PxReal softPointInverseMass =
		avbdGetSoftPointInverseMass(point, particles, numParticles);
	if(softPointInverseMass <= 0.0f ||
		rigidBody.invMass <= 0.0f ||
		attachment.k <= 0.0f || dt <= 0.0f)
		return false;

	PxVec3 linearInvMass(rigidBody.invMass);
	rigidBody.projectLockedLinearVector(linearInvMass);
	const PxMat33 rigidLinearInverse =
		PxMat33::createDiagonal(linearInvMass);

	const PxVec3 basis[3] = {
		PxVec3(1.0f, 0.0f, 0.0f),
		PxVec3(0.0f, 1.0f, 0.0f),
		PxVec3(0.0f, 0.0f, 1.0f)};
	PxVec3 angularColumns[3];
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		PxVec3 projectedInput = basis[axis];
		rigidBody.projectLockedAngularVector(projectedInput);
		angularColumns[axis] =
			rigidBody.invInertiaWorld * projectedInput;
		rigidBody.projectLockedAngularVector(
			angularColumns[axis]);
	}
	const PxMat33 rigidAngularInverse(
		angularColumns[0], angularColumns[1],
		angularColumns[2]);

	step.worldOffset =
		rigidBody.rotation.rotate(attachment.localOffset);
	const PxVec3 worldAnchor =
		rigidBody.position + step.worldOffset;
	step.constraint =
		avbdGetSoftPointPosition(point, particles) - worldAnchor;
	const PxMat33 skew = avbdSkew(step.worldOffset);
	const PxReal dt2 = dt * dt;
	const PxMat33 unit =
		PxMat33::createDiagonal(PxVec3(1.0f));
	const PxMat33 pointInverseMass =
		(unit * softPointInverseMass +
		 rigidLinearInverse -
		 skew * rigidAngularInverse * skew) *
		dt2;
	const PxReal compliance =
		1.0f / PxMax(attachment.k, 1.0e-6f);
	const PxMat33 effectiveMass =
		pointInverseMass + unit * compliance;
	step.multiplier = avbdSolveSymmetric33(
		effectiveMass,
		-(step.constraint +
		  attachment.alLambda * compliance));
	if(!step.multiplier.isFinite())
		return false;

	for(PxU32 i = 0; i < point.particleCount; i++)
	{
		const PxU32 particleIndex = point.particleIndices[i];
		step.particleCorrections[i] =
			step.multiplier *
			(dt2 * point.weights[i] *
			 particles[particleIndex].invMass);
		if(!step.particleCorrections[i].isFinite())
			return false;
	}
	step.rigidLinearCorrection =
		(rigidLinearInverse * step.multiplier) * (-dt2);
	step.rigidAngularCorrection =
		rigidAngularInverse *
		(skew.getTranspose() * step.multiplier);
	step.rigidAngularCorrection *= dt2;
	return step.rigidLinearCorrection.isFinite() &&
		step.rigidAngularCorrection.isFinite();
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
// Dynamic rigid-soft contact: the same prepared position objective supplies
// the soft 3x3 block and the rigid 6x6 block.
// =============================================================================

PX_FORCE_INLINE PxVec3 avbdGetRigidContactSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSolverBody& body)
{
	return
		body.position + body.rotation.rotate(geometry.rigidLocalPoint);
}

PX_FORCE_INLINE PxMat33 avbdProjectLockedContactMatrix(
	const AvbdSolverBody& body, const PxMat33& matrix,
	bool angularRows, bool angularColumns)
{
	PxMat33 projected = matrix;
	if(angularRows)
	{
		body.projectLockedAngularVector(projected.column0);
		body.projectLockedAngularVector(projected.column1);
		body.projectLockedAngularVector(projected.column2);
	}
	else
	{
		body.projectLockedLinearVector(projected.column0);
		body.projectLockedLinearVector(projected.column1);
		body.projectLockedLinearVector(projected.column2);
	}
	projected = projected.getTranspose();
	if(angularColumns)
	{
		body.projectLockedAngularVector(projected.column0);
		body.projectLockedAngularVector(projected.column1);
		body.projectLockedAngularVector(projected.column2);
	}
	else
	{
		body.projectLockedLinearVector(projected.column0);
		body.projectLockedLinearVector(projected.column1);
		body.projectLockedLinearVector(projected.column2);
	}
	return projected.getTranspose();
}

PX_FORCE_INLINE bool avbdAddDynamicSoftRigidContactContribution_rigid(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	PxU32 bodyIdx, const AvbdSoftParticle* particles,
	PxU32 numParticles, const AvbdSolverBody& body,
	AvbdBlock6x6& lhs, AvbdVec6& rhs)
{
	if(!geometry.hasRigidBodyTarget() ||
		geometry.targetIndex != bodyIdx ||
		!avbdHasSoftContactDynamicQuerySupport(
			geometry, particles, numParticles))
		return false;

	const PxVec3 worldOffset =
		body.rotation.rotate(geometry.rigidLocalPoint);
	const PxVec3 surfacePoint = body.position + worldOffset;
	PxVec3 particleForce;
	PxMat33 particleHessian;
	avbdEvaluateContactParticleBlockAtSurfacePoint(
		geometry, state, particles, surfacePoint, 1.0f,
		particleForce, particleHessian);

	PxVec3 rigidLinearForce = particleForce;
	body.projectLockedLinearVector(rigidLinearForce);
	PxVec3 rigidAngularForce = worldOffset.cross(particleForce);
	body.projectLockedAngularVector(rigidAngularForce);
	rhs.linear += rigidLinearForce;
	rhs.angular += rigidAngularForce;

	const PxMat33 skew = avbdSkew(worldOffset);
	const PxMat33 linearLinear = avbdProjectLockedContactMatrix(
		body, particleHessian, false, false);
	PxMat33 linearAngular =
		particleHessian * skew * (-1.0f);
	linearAngular = avbdProjectLockedContactMatrix(
		body, linearAngular, false, true);
	PxMat33 angularAngular =
		skew.getTranspose() * particleHessian * skew;
	angularAngular = avbdProjectLockedContactMatrix(
		body, angularAngular, true, true);
	lhs.linearLinear += linearLinear;
	lhs.linearAngular += linearAngular;
	lhs.angularLinear += linearAngular.getTranspose();
	lhs.angularAngular += angularAngular;
	return true;
}

PX_FORCE_INLINE PxU32 avbdAddDynamicSoftRigidContactContributions_rigid(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 bodyIdx, const AvbdSoftParticle* particles,
	PxU32 numParticles, const AvbdSolverBody& body,
	AvbdBlock6x6& lhs, AvbdVec6& rhs)
{
	PxU32 contributionCount = 0;
	for(PxU32 contactIdx = 0; contactIdx < numContacts; contactIdx++)
	{
		const AvbdSoftContact& contact = contacts[contactIdx];
		if(avbdAddDynamicSoftRigidContactContribution_rigid(
			contact.geometry, contact.state, bodyIdx,
			particles, numParticles, body, lhs, rhs))
			contributionCount++;
	}
	return contributionCount;
}

// =============================================================================
// Kinematic shell: rigid 6x6 row (standalone addKinematicShellContactContribution_rigid)
// =============================================================================

PX_FORCE_INLINE PxReal avbdKinematicShellContactViolation(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSolverBody& body) {
	const PxVec3 rAw = body.rotation.rotate(geometry.rigidLocalPoint);
	const PxVec3 wA = body.position + rAw;
	PxReal geom =
		(wA - geometry.surfacePoint).dot(geometry.normal) - geometry.depth;
	if (geom < 0.0f)
		geom = PxMin(geom, -geometry.depth);
	return geom;
}

PX_FORCE_INLINE void avbdAddKinematicShellContactContribution_rigid(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	PxU32 bodyIdx, const AvbdSolverBody& body,
	PxReal boostFloor, AvbdBlock6x6& lhs, AvbdVec6& rhs) {
	if (!geometry.hasRigidBodyTarget() ||
		geometry.targetIndex != bodyIdx)
		return;

	const PxVec3 n = geometry.normal;
	const PxVec3 rAw = body.rotation.rotate(geometry.rigidLocalPoint);
	const PxReal geom =
		avbdKinematicShellContactViolation(geometry, body);
	PxVec3 gradLin = n;
	body.projectLockedLinearVector(gradLin);
	PxVec3 gradAng = rAw.cross(n);
	body.projectLockedAngularVector(gradAng);
	const PxReal pen = PxMax(state.k, boostFloor);
	const PxReal f = PxMin(0.0f, pen * geom + state.alLambda);
	rhs.linear += gradLin * f;
	rhs.angular += gradAng * f;
	lhs.addConstraintContribution(gradLin, gradAng, pen);
}

PX_FORCE_INLINE void avbdComputeKinematicShellTangentViolation(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSolverBody& body, PxReal Ctangent[2])
{
	const PxVec3 rAw = body.rotation.rotate(geometry.rigidLocalPoint);
	const PxVec3 worldA = body.position + rAw;
	const PxVec3 prevWorldA =
		body.prevPosition + body.prevRotation.rotate(geometry.rigidLocalPoint);
	const PxVec3 staticMotion =
		geometry.surfacePoint - state.surfacePointPrev;
	const PxVec3 relDisp = (worldA - prevWorldA) - staticMotion;

	PxQuat deltaQ = body.rotation * body.prevRotation.getConjugate();
	if (deltaQ.w < 0.0f)
		deltaQ = -deltaQ;
	const PxVec3 dw(deltaQ.x, deltaQ.y, deltaQ.z);
	const PxVec3 angDisp = dw * 2.0f;

	Ctangent[0] = relDisp.dot(geometry.tangent1) +
		angDisp.dot(rAw.cross(geometry.tangent1));
	Ctangent[1] = relDisp.dot(geometry.tangent2) +
		angDisp.dot(rAw.cross(geometry.tangent2));
}

PX_FORCE_INLINE void avbdUpdateKinematicShellContactDual(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSolverBody& body, PxReal beta, PxReal penaltyMax)
{
	const PxReal Cn = avbdKinematicShellContactViolation(geometry, body);
	const PxReal rawLambdaN = state.k * Cn + state.alLambda;
	state.alLambda = PxMin(0.0f, rawLambdaN);
	if (state.alLambda < 0.0f)
		state.k = PxMin(state.k + beta * PxAbs(Cn), state.ke);

	PxReal Ctangent[2];
	avbdComputeKinematicShellTangentViolation(
		geometry, state, body, Ctangent);
	// Coulomb cone on shell tangents (bound from normal force F_n = alLambda).
	PxReal Fn = state.alLambda;
	PxReal Ft0 = 0.0f, Ft1 = 0.0f;
	const PxReal preLen = avbdEvaluateContactForcesCone(
		0.0f, 0.0f, state.alLambda, state.penTangent[0], Ctangent[0],
		state.alLambdaTangent[0], state.penTangent[1], Ctangent[1],
		state.alLambdaTangent[1], geometry.friction, Fn, Ft0, Ft1);
	state.alLambdaTangent[0] = Ft0;
	state.alLambdaTangent[1] = Ft1;
	const PxReal bounds = PxAbs(state.alLambda) * geometry.friction;
	if (preLen <= bounds)
	{
		state.penTangent[0] =
			PxMin(state.penTangent[0] + beta * PxAbs(Ctangent[0]), penaltyMax);
		state.penTangent[1] =
			PxMin(state.penTangent[1] + beta * PxAbs(Ctangent[1]), penaltyMax);
	}
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_H
