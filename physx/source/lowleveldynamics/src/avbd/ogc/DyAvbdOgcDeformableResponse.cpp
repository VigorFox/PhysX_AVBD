// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/ogc/DyAvbdOgcResponse.h"

#include "avbd/ogc/DyAvbdOgcPairState.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftContactGeometry.h"

namespace physx
{
namespace Dy
{

PxU32 applyDeformableOgcNormalDepenetrationSweeps(
	AvbdSoftParticle* softParticles, PxU32 numSoftParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* softContacts, PxU32 numSoftContacts,
	PxU32 sweeps, AvbdOgcPairState* pairStates,
	PxU32 numPairStates, const PxU32* pairIndices,
	PxU32 numPairIndices,
	AvbdOgcVelocityContactDomain contactDomain,
	AvbdSolverStats* stats)
{
	(void)stats;
	if(!softParticles || numSoftParticles == 0 || !softBodies ||
		numSoftBodies == 0 || !softContacts || numSoftContacts == 0 ||
		sweeps == 0 || !pairStates || numPairStates == 0 || !pairIndices ||
		numPairIndices != numSoftContacts ||
		contactDomain == AvbdOgcVelocityContactDomain::eNONE)
		return 0u;

	PxU32 appliedCorrections = 0u;

	static const PxReal minimumDeterminant = 0.05f;
	static const PxReal overlapTolerance = 1.0e-6f;
	for(PxU32 sweep = 0; sweep < sweeps; ++sweep)
	{
		bool committedSweep = false;
		for(PxU32 contactIndex = 0; contactIndex < numSoftContacts;
			++contactIndex)
		{
			const AvbdSoftContactGeometry& geometry =
				softContacts[contactIndex].geometry;
			if(geometry.source.type != AvbdSoftContactSource::eSOFT_SURFACE ||
				!geometry.hasDeformableSurfaceTarget() ||
				geometry.queryBodyIndex >= numSoftBodies ||
				geometry.targetIndex >= numSoftBodies ||
				geometry.queryBodyIndex == geometry.targetIndex)
				continue;

			const PxU32 pairIndex = pairIndices[contactIndex];
			if(pairIndex >= numPairStates)
				continue;
			AvbdOgcPairState& pair = pairStates[pairIndex];
			if(!pair.geometry.active ||
				!pair.matches(geometry.source.type, geometry.targetKind,
					geometry.queryBodyIndex, geometry.targetIndex,
					geometry.source.primitiveKey))
				continue;

			const AvbdSoftBody& sourceBody =
				softBodies[geometry.queryBodyIndex];
			const AvbdSoftBody& targetBody =
				softBodies[geometry.targetIndex];
			if(sourceBody.compiled.speculativeCCDEnabled ||
				targetBody.compiled.speculativeCCDEnabled)
				continue;

			AvbdOgcNormalResponse response;
			if(!compileCurrentOgcNormalResponse(
					geometry, softParticles, numSoftParticles, NULL,
					1.0f, response) ||
				!PxIsFinite(response.current.signedGap) ||
				response.current.signedGap >= -overlapTolerance)
				continue;

			const PxReal lambda =
				-response.current.signedGap / response.effectiveResponse;
			AvbdOgcSoftPositionCandidate candidate;
			if(!PxIsFinite(lambda) || lambda <= 0.0f ||
				!buildOgcDeformablePairPositionCandidate(
					response, softParticles, numSoftParticles, sourceBody,
					targetBody, lambda, candidate))
				continue;

			PxReal alpha = 1.0f;
			bool admitted = false;
			for(PxU32 attempt = 0; attempt < 8u; ++attempt)
			{
				if(admitOgcDeformablePairPositionCandidate(
						response, candidate, softParticles, numSoftParticles,
						sourceBody, targetBody, alpha,
						minimumDeterminant))
				{
					admitted = true;
					break;
				}
				alpha *= 0.5f;
			}
			if(!admitted)
				continue;

			commitOgcSoftPositionCandidate(
				response, candidate, softParticles, numSoftParticles, alpha);
			const PxReal correction =
				-response.current.signedGap * alpha;
			publishLocalOgcPairPositionResult(
				softContacts, numSoftContacts, contactIndex, correction,
				contactDomain, pairStates, numPairStates, pairIndices,
				numPairIndices);
			committedSweep = true;
			++appliedCorrections;
		}
		if(!committedSweep)
			break;
	}
	return appliedCorrections;
}

} // namespace Dy
} // namespace physx
