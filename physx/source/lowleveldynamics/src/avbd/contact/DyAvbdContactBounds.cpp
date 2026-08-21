// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactBounds.h"

namespace physx
{
namespace Dy
{

void avbdComputeSoftBodyBounds(
	const AvbdSoftParticle* particles, const AvbdSoftBody& body,
	AvbdSoftBodyBounds& bounds)
{
	bounds = AvbdSoftBodyBounds();
	for(PxU32 particleIndex = 0;
		particleIndex < body.compiled.particleCount; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[
			body.compiled.particleStart + particleIndex];
		bounds.currentMinimum = bounds.currentMinimum.minimum(
			particle.position);
		bounds.currentMaximum = bounds.currentMaximum.maximum(
			particle.position);
		// Retain the legacy swept reduction order exactly: current first,
		// then the initial position for each particle.
		bounds.sweptMinimum = bounds.sweptMinimum.minimum(
			particle.position);
		bounds.sweptMaximum = bounds.sweptMaximum.maximum(
			particle.position);
		bounds.sweptMinimum = bounds.sweptMinimum.minimum(
			particle.initialPosition);
		bounds.sweptMaximum = bounds.sweptMaximum.maximum(
			particle.initialPosition);
	}
}

} // namespace Dy
} // namespace physx
