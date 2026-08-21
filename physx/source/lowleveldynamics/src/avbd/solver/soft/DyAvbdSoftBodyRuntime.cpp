// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPolicy.h"

namespace physx
{
namespace Dy
{

// Whole-component primal-initialization admission. Runtime storage and body
// composition live in DyAvbdSoftBodyRuntime.h.

// A speculative body relies on the full old-position -> predicted-position
// sweep to admit first-impact contacts.  This diagnostic is deliberately
// limited to an all-dynamic, unpinned and unattached component: a per-vertex
// initial guess would otherwise manufacture strain at a static or owned
// support.  Reject the entire component rather than shortening only one side
// of a soft-soft swept pair.
bool avbdCanUseSoftAdaptivePrimalInitialization(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	if(!avbdUseSoftAdaptivePrimalInitialization())
		return false;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.compiled.speculativeCCDEnabled ||
			!body.runtime.pins.empty() ||
			!body.runtime.attachments.empty() ||
			body.compiled.particleStart > numParticles ||
			body.compiled.particleCount >
				numParticles - body.compiled.particleStart)
			return false;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			if(particles[body.compiled.particleStart + localIndex].invMass <=
				0.0f)
				return false;
		}
	}
	return true;
}

// A rigid initial guess changes every dynamic particle of a body together.
// Therefore it has the same whole-component swept-contact restriction as the
// adaptive guess, and additionally needs finite positive masses for its
// mass-weighted fit.  Reject the complete component rather than applying the
// transform to only a subset of bodies or vertices.
bool avbdCanUseSoftRigidPrimalInitialization(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	if(!avbdUseSoftRigidPrimalInitialization() || !particles ||
		!softBodies || numParticles == 0 || numSoftBodies == 0)
		return false;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.compiled.speculativeCCDEnabled ||
			!body.runtime.pins.empty() ||
			!body.runtime.attachments.empty() ||
			body.compiled.particleCount == 0 ||
			body.compiled.particleStart > numParticles ||
			body.compiled.particleCount >
				numParticles - body.compiled.particleStart)
			return false;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			const AvbdSoftParticle& particle = particles[
				body.compiled.particleStart + localIndex];
			if(!PxIsFinite(particle.invMass) || particle.invMass <= 0.0f ||
				!PxIsFinite(particle.mass) || particle.mass <= 0.0f)
				return false;
		}
	}
	return true;
}

} // namespace Dy
} // namespace physx
