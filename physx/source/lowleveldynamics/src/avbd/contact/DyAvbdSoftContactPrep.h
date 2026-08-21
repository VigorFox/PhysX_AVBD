// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_CONTACT_PREP_H
#define DY_AVBD_SOFT_CONTACT_PREP_H

#include "avbd/solver/soft/DyAvbdSoftContactGeometry.h"
#include "foundation/PxAssert.h"

namespace physx
{
namespace Dy
{

PX_FORCE_INLINE void avbdInitializeSoftContactAnchors(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles)
{
	state.particlePointPrev =
		avbdGetSoftContactQueryPoint(geometry, particles);
	state.surfacePointPrev =
		avbdGetSoftContactSurfacePoint(geometry, particles);
}

// A velocity-owned tangent must never inherit a Position-AL spring or its
// frame anchor.  The normal AL state intentionally stays intact: this helper
// changes only the tangential owner.
PX_FORCE_INLINE void avbdResetSoftContactTangentState(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles)
{
	state.alLambdaTangent[0] = 0.0f;
	state.alLambdaTangent[1] = 0.0f;
	state.penTangent[0] = 1000.0f;
	state.penTangent[1] = 1000.0f;
	state.frictionStick = false;
	avbdInitializeSoftContactAnchors(geometry, state, particles);
}

PX_FORCE_INLINE void avbdBuildSoftContactTangents(
	AvbdSoftContactGeometry& geometry)
{
	if(PxAbs(geometry.normal.x) < 0.9f)
		geometry.tangent1 =
			geometry.normal.cross(PxVec3(1.0f, 0.0f, 0.0f)).getNormalized();
	else
		geometry.tangent1 =
			geometry.normal.cross(PxVec3(0.0f, 1.0f, 0.0f)).getNormalized();
	geometry.tangent2 = geometry.normal.cross(geometry.tangent1);
}

// The only production boundary from detection into the solver.  Geometry is
// the prepared-contact IR; this function creates and initializes its unique
// augmented state before publishing the aggregate to the solver.
PX_FORCE_INLINE void avbdAppendPreparedSoftContact(
	const AvbdSoftContactGeometry& geometry,
	PxReal k, PxReal ke,
	const AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts)
{
	PX_ASSERT(geometry.source.isValid());
	PX_ASSERT(
		geometry.targetKind !=
			AvbdSoftContactTargetKind::eUNSUPPORTED);
	PX_ASSERT(geometry.normal.isFinite());
	PX_ASSERT(geometry.tangent1.isFinite());
	PX_ASSERT(geometry.tangent2.isFinite());

	AvbdSoftContact contact;
	contact.geometry = geometry;
	AvbdSoftContactAugmentedState& state = contact.state;
	state.k = k;
	state.ke = ke;
	avbdInitializeSoftContactAnchors(geometry, state, particles);
	contacts.pushBack(contact);
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_CONTACT_PREP_H
