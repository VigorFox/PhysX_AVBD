// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactMaterial.h"
#include "avbd/contact/DyAvbdContactPlane.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"

namespace physx
{
namespace Dy
{

// CPU AVBD world-plane contact detection.

// =============================================================================
// Ground contact detection
// =============================================================================

// A worker owns one canonical particle interval and appends only to its local
// contact array. Parent merge order remains particle-major/plane-minor.
void avbdDetectSoftWorldPlaneContactsRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 i = particleBegin; i < particleEnd; i++)
	{
		if(particles[i].invMass <= 0.0f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(softBodies, numSoftBodies, i);
		if(sourceBody && !avbdIsSoftBodySurfaceVertex(*sourceBody, i))
			continue;
		const PxVec3& position = particles[i].position;
		for(PxU32 planeIndex = 0; planeIndex < numPlanes; planeIndex++)
		{
			const AvbdWorldPlane& plane = planes[planeIndex];
			const PxReal normalMagnitudeSq =
				plane.normal.magnitudeSquared();
			if(normalMagnitudeSq <= 1e-12f ||
				!PxIsFinite(normalMagnitudeSq))
				continue;
			const PxVec3 normal =
				plane.normal * PxRecipSqrt(normalMagnitudeSq);
			const PxReal distance = normal.dot(position) - plane.offset;
			bool speculativeCandidate = false;
			if(distance >= margin)
			{
				if(!sourceBody ||
					!sourceBody->compiled.speculativeCCDEnabled ||
					!particles[i].predictedPosition.isFinite())
					continue;
				const PxReal predictedDistance =
					normal.dot(particles[i].predictedPosition) - plane.offset;
				if(predictedDistance >= margin)
					continue;
				speculativeCandidate = true;
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eGROUND, PX_MAX_U32,
				plane.primitiveKey, 0);
			geometry.particleIdx = i;
			geometry.targetKind =
				AvbdSoftContactTargetKind::eWORLD_STATIC;
			geometry.velocityOwner =
				AvbdVelocityObjectiveOwner::PositionAL;
			geometry.targetIndex = planeIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = PxMax(0.0f, -distance);
			geometry.margin = margin;
			geometry.surfacePoint = position - normal * distance;
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					plane.friction, plane.frictionCombineMode)
				: PxMax(plane.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, speculativeCandidate ? 1e6f : 1e4f,
				1e6f, particles, contacts);
		}
	}
}

void avbdDetectSoftWorldPlaneContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	avbdDetectSoftWorldPlaneContactsRange(
		particles, numParticles, 0, numParticles,
		planes, numPlanes, contacts, margin,
		softBodies, numSoftBodies);
}

void avbdDetectSoftGroundContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxArray<AvbdSoftContact>& contacts,
	PxReal groundY, PxReal margin,
	PxReal friction,
	const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies)
{
	contacts.clear();
	AvbdWorldPlane plane;
	plane.offset = groundY;
	plane.friction = friction;
	avbdDetectSoftWorldPlaneContacts(
		particles, numParticles, &plane, 1, contacts, margin,
		softBodies, numSoftBodies);
}

} // namespace Dy
} // namespace physx
