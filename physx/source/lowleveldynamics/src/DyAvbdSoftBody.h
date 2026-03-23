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

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_H
