// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdSoftContactPrep.h"
#include "avbd/ogc/DyAvbdOgcResponse.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTopologyQueries.h"

namespace physx
{
namespace Dy
{

// AVBD particle-primal scheduling policy and prediction boundary.
//
// This unit owns deterministic color-plan construction, dynamic conflict
// proofs, policy validation, and particle prediction setup. The numerical
// primal kernels remain in DyAvbdSoftBodyPrimalKernel.inl.
// =============================================================================

inline void avbdBuildParticlePrimalColorPlan(
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles,
	AvbdParticlePrimalSchedule particlePrimalSchedule)
{
	static const PxU32 eMAX_DYNAMIC_CONFLICT_INDICES =
		(2u * 1024u * 1024u) / sizeof(PxU32);
	workspace.particlePrimalDynamicConflictValid = false;
	workspace.particlePrimalColorPlanValid = false;
	workspace.particlePrimalColorCount = 0;
	// A future color task must never grow worker-visible scratch while it is
	// preparing an epoch.  The Scene reserves these arrays at lifecycle
	// boundaries; a smaller dynamic budget simply leaves this optional plan
	// unpublished so the authoritative serial primal remains the fallback.
	if(workspace.particlePrimalBodyIndices.capacity() < numParticles ||
		workspace.particlePrimalColors.capacity() < numParticles ||
		workspace.particlePrimalColorCounts.capacity() < numParticles ||
		workspace.particlePrimalColorOffsets.capacity() <
			numParticles + 1 ||
		workspace.particlePrimalColorParticles.capacity() < numParticles ||
		workspace.particlePrimalDynamicConflictOffsets.capacity() <
			numParticles + 1 ||
		workspace.particlePrimalDynamicConflictCounts.capacity() <
			numParticles)
		return;
	workspace.resize(workspace.particlePrimalBodyIndices, numParticles);
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalBodyIndices[particleIndex] = PX_MAX_U32;

	bool valid = true;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		if(!compiled.validateParticlePrimalStructuralAccessDescriptor() ||
			compiled.particleStart > numParticles ||
			compiled.particleCount >
				numParticles - compiled.particleStart)
		{
			valid = false;
			break;
		}
		for(PxU32 localIndex = 0;
			localIndex < compiled.particleCount; localIndex++)
		{
			const PxU32 particleIndex =
				compiled.particleStart + localIndex;
			if(workspace.particlePrimalBodyIndices[particleIndex] !=
				PX_MAX_U32)
			{
				valid = false;
				break;
			}
			workspace.particlePrimalBodyIndices[particleIndex] = bodyIndex;
		}
		if(!valid)
			break;
	}
	if(valid)
	{
		for(PxU32 particleIndex = 0; particleIndex < numParticles;
			particleIndex++)
		{
			if(workspace.particlePrimalBodyIndices[particleIndex] ==
				PX_MAX_U32)
			{
				valid = false;
				break;
			}
		}
	}
	if(!valid)
		return;

	PxU64 groupCount64 = numContacts;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const PxArray<AvbdCompiledSoftObjective>& objectives =
			softBodies[bodyIndex].runtime.compiledObjectives;
		for(PxU32 objectiveIndex = 0;
			objectiveIndex < objectives.size(); objectiveIndex++)
		{
			if(avbdIsPinPositionOwner(objectives[objectiveIndex].owner))
				groupCount64++;
		}
	}
	if(groupCount64 > PX_MAX_U32)
		return;
	if(workspace.particlePrimalDynamicAccessGroups.capacity() <
		PxU32(groupCount64))
		return;
	workspace.resize(workspace.particlePrimalDynamicAccessGroups,
		PxU32(groupCount64));

	auto writeGroup = [&workspace, numParticles](
		PxU32 groupIndex,
		AvbdParticlePrimalDynamicAccessSource source,
		const PxU32* inputIndices, PxU32 inputCount) -> bool
	{
		if(groupIndex >=
			workspace.particlePrimalDynamicAccessGroups.size() ||
			inputCount > AVBD_CONTACT_MAX_PARTICLES)
			return false;
		AvbdParticlePrimalDynamicAccessGroup& group =
			workspace.particlePrimalDynamicAccessGroups[groupIndex];
		group = AvbdParticlePrimalDynamicAccessGroup();
		group.source = source;
		for(PxU32 inputIndex = 0; inputIndex < inputCount;
			inputIndex++)
		{
			const PxU32 particleIndex = inputIndices[inputIndex];
			if(particleIndex >= numParticles)
				return false;
			bool unique = true;
			for(PxU32 previous = 0;
				previous < group.particleCount; previous++)
			{
				if(group.particleIndices[previous] == particleIndex)
				{
					unique = false;
					break;
				}
			}
			if(unique)
				group.particleIndices[group.particleCount++] = particleIndex;
		}
		PxSort(group.particleIndices, group.particleCount);
		return true;
	};

	PxU32 groupCursor = 0;
	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		contactIndex++)
	{
		PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 particleCount =
			avbdCollectSoftContactParticleIndices(
				contacts[contactIndex].geometry, particleIndices);
		if(!writeGroup(groupCursor++,
			AvbdParticlePrimalDynamicAccessSource::eCONTACT,
			particleIndices, particleCount))
			return;
	}
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const PxArray<AvbdCompiledSoftObjective>& objectives =
			softBodies[bodyIndex].runtime.compiledObjectives;
		for(PxU32 objectiveIndex = 0;
			objectiveIndex < objectives.size(); objectiveIndex++)
		{
			const AvbdCompiledSoftObjective& objective =
				objectives[objectiveIndex];
			if(!avbdIsPinPositionOwner(objective.owner))
				continue;
			PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
			const PxU32 particleCount = objective.point.particleCount;
			if(particleCount > 3)
				return;
			for(PxU32 pointIndex = 0;
				pointIndex < particleCount; pointIndex++)
				particleIndices[pointIndex] =
					objective.point.particleIndices[pointIndex];
			if(!writeGroup(groupCursor++,
				AvbdParticlePrimalDynamicAccessSource::ePIN_OBJECTIVE,
				particleIndices, particleCount))
				return;
		}
	}
	PX_ASSERT(groupCursor ==
		workspace.particlePrimalDynamicAccessGroups.size());

	workspace.resize(workspace.particlePrimalDynamicConflictCounts,
		numParticles);
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalDynamicConflictCounts[particleIndex] = 0;
	for(PxU32 groupIndex = 0;
		groupIndex < workspace.particlePrimalDynamicAccessGroups.size();
		groupIndex++)
	{
		const AvbdParticlePrimalDynamicAccessGroup& group =
			workspace.particlePrimalDynamicAccessGroups[groupIndex];
		if(group.particleCount < 2)
			continue;
		const PxU32 additions = group.particleCount - 1;
		for(PxU32 participant = 0;
			participant < group.particleCount; participant++)
		{
			const PxU32 particleIndex =
				group.particleIndices[participant];
			if(workspace.particlePrimalDynamicConflictCounts[
				particleIndex] > PX_MAX_U32 - additions)
				return;
			workspace.particlePrimalDynamicConflictCounts[
				particleIndex] += additions;
		}
	}
	workspace.resize(workspace.particlePrimalDynamicConflictOffsets,
		numParticles + 1);
	workspace.particlePrimalDynamicConflictOffsets[0] = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 begin =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
		const PxU32 count =
			workspace.particlePrimalDynamicConflictCounts[particleIndex];
		if(begin > eMAX_DYNAMIC_CONFLICT_INDICES ||
			count > eMAX_DYNAMIC_CONFLICT_INDICES - begin)
			return;
		workspace.particlePrimalDynamicConflictOffsets[particleIndex + 1] =
			begin + count;
	}
	const PxU32 dynamicConflictCount =
		workspace.particlePrimalDynamicConflictOffsets[numParticles];
	if(workspace.particlePrimalDynamicConflictIndices.capacity() <
		dynamicConflictCount)
		return;
	workspace.resize(workspace.particlePrimalDynamicConflictIndices,
		dynamicConflictCount);
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalDynamicConflictCounts[particleIndex] = 0;
	for(PxU32 groupIndex = 0;
		groupIndex < workspace.particlePrimalDynamicAccessGroups.size();
		groupIndex++)
	{
		const AvbdParticlePrimalDynamicAccessGroup& group =
			workspace.particlePrimalDynamicAccessGroups[groupIndex];
		for(PxU32 source = 0; source < group.particleCount; source++)
		{
			const PxU32 particleIndex = group.particleIndices[source];
			for(PxU32 target = 0; target < group.particleCount; target++)
			{
				if(source == target)
					continue;
				const PxU32 writeIndex =
					workspace.particlePrimalDynamicConflictOffsets[
						particleIndex] +
					workspace.particlePrimalDynamicConflictCounts[
						particleIndex]++;
				PX_ASSERT(writeIndex <
					workspace.particlePrimalDynamicConflictIndices.size());
				workspace.particlePrimalDynamicConflictIndices[writeIndex] =
					group.particleIndices[target];
			}
		}
	}

	// Sort and compact each local range in place. The write cursor never passes
	// an unread source range because compaction only removes duplicates.
	PxU32 compactWrite = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 begin =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
		const PxU32 end =
			workspace.particlePrimalDynamicConflictOffsets[
				particleIndex + 1];
		PxSort(
			workspace.particlePrimalDynamicConflictIndices.begin() + begin,
			end - begin);
		workspace.particlePrimalDynamicConflictOffsets[particleIndex] =
			compactWrite;
		for(PxU32 conflictIndex = begin; conflictIndex < end;
			conflictIndex++)
		{
			const PxU32 conflict =
				workspace.particlePrimalDynamicConflictIndices[conflictIndex];
			if(conflict == particleIndex ||
				(compactWrite >
					workspace.particlePrimalDynamicConflictOffsets[
						particleIndex] &&
					workspace.particlePrimalDynamicConflictIndices[
						compactWrite - 1] == conflict))
				continue;
			workspace.particlePrimalDynamicConflictIndices[compactWrite++] =
				conflict;
		}
	}
	workspace.particlePrimalDynamicConflictOffsets[numParticles] =
		compactWrite;
	workspace.particlePrimalDynamicConflictIndices.resize(compactWrite);

	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 begin =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
		const PxU32 end =
			workspace.particlePrimalDynamicConflictOffsets[
				particleIndex + 1];
		if(begin > end || end >
			workspace.particlePrimalDynamicConflictIndices.size())
			return;
		for(PxU32 conflictIndex = begin; conflictIndex < end;
			conflictIndex++)
		{
			const PxU32 conflict =
				workspace.particlePrimalDynamicConflictIndices[conflictIndex];
			if(conflict >= numParticles || conflict == particleIndex ||
				(conflictIndex > begin &&
					workspace.particlePrimalDynamicConflictIndices[
						conflictIndex - 1] >= conflict))
				return;
			bool reverseFound = false;
			for(PxU32 reverseIndex =
				workspace.particlePrimalDynamicConflictOffsets[conflict];
				reverseIndex <
					workspace.particlePrimalDynamicConflictOffsets[
						conflict + 1]; reverseIndex++)
			{
				if(workspace.particlePrimalDynamicConflictIndices[
					reverseIndex] == particleIndex)
				{
					reverseFound = true;
					break;
				}
			}
			if(!reverseFound)
				return;
		}
	}
	workspace.particlePrimalDynamicConflictValid = true;

	workspace.resize(workspace.particlePrimalColors, numParticles);
	// The ordered reference schedule orients every conflict by the legacy
	// particle order.  The relaxed fast schedule instead uses a standard
	// greedy coloring: it preserves the no-conflict invariant inside a color,
	// but deliberately permits a different nonlinear-GS trajectory across
	// colors.  It is therefore never selected by the ordered reference mode.
	const bool preserveLegacyCausalOrder =
		particlePrimalSchedule !=
			AvbdParticlePrimalSchedule::eRELAXED_COLOR;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalColors[particleIndex] = PX_MAX_U32;
	PxU32 colorCount = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		PxU32 color = 0;
		const PxU32 bodyIndex =
			workspace.particlePrimalBodyIndices[particleIndex];
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		const PxU32 localIndex = particleIndex - compiled.particleStart;
		auto hasEarlierNeighborWithColor = [&workspace, &compiled,
			particleIndex, localIndex](
			PxU32 candidateColor)
		{
			for(PxU32 conflictIndex =
				compiled.particlePrimalStructuralConflictOffsets[localIndex];
				conflictIndex <
				compiled.particlePrimalStructuralConflictOffsets[
					localIndex + 1]; conflictIndex++)
			{
				const PxU32 neighbor = compiled.particleStart +
					compiled.particlePrimalStructuralConflictIndices[
						conflictIndex];
				if(neighbor < particleIndex &&
					workspace.particlePrimalColors[neighbor] == candidateColor)
					return true;
			}
			for(PxU32 conflictIndex =
				workspace.particlePrimalDynamicConflictOffsets[particleIndex];
				conflictIndex <
				workspace.particlePrimalDynamicConflictOffsets[
					particleIndex + 1]; conflictIndex++)
			{
				const PxU32 neighbor =
					workspace.particlePrimalDynamicConflictIndices[
						conflictIndex];
				if(neighbor < particleIndex &&
					workspace.particlePrimalColors[neighbor] == candidateColor)
					return true;
			}
			return false;
		};
		if(preserveLegacyCausalOrder)
		{
			auto observeEarlierConflict = [&workspace, particleIndex, &color](
				PxU32 neighbor)
			{
				if(neighbor >= particleIndex)
					return true;
				const PxU32 neighborColor =
					workspace.particlePrimalColors[neighbor];
				if(neighborColor == PX_MAX_U32 || neighborColor >=
					particleIndex)
					return false;
				color = PxMax(color, neighborColor + 1);
				return true;
			};
			for(PxU32 conflictIndex =
				compiled.particlePrimalStructuralConflictOffsets[localIndex];
				conflictIndex <
					compiled.particlePrimalStructuralConflictOffsets[
						localIndex + 1]; conflictIndex++)
			{
				if(!observeEarlierConflict(compiled.particleStart +
					compiled.particlePrimalStructuralConflictIndices[conflictIndex]))
					return;
			}
			for(PxU32 conflictIndex =
				workspace.particlePrimalDynamicConflictOffsets[particleIndex];
				conflictIndex <
					workspace.particlePrimalDynamicConflictOffsets[
						particleIndex + 1]; conflictIndex++)
			{
				if(!observeEarlierConflict(
					workspace.particlePrimalDynamicConflictIndices[conflictIndex]))
					return;
			}
		}
		else
		{
			while(hasEarlierNeighborWithColor(color))
			{
				if(++color >= numParticles)
					return;
			}
		}
		if(color >= numParticles)
			return;
		workspace.particlePrimalColors[particleIndex] = color;
		colorCount = PxMax(colorCount, color + 1);
	}

	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 color = workspace.particlePrimalColors[particleIndex];
		const PxU32 bodyIndex =
			workspace.particlePrimalBodyIndices[particleIndex];
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		const PxU32 localIndex = particleIndex - compiled.particleStart;
		for(PxU32 conflictIndex =
			compiled.particlePrimalStructuralConflictOffsets[localIndex];
			conflictIndex <
				compiled.particlePrimalStructuralConflictOffsets[
					localIndex + 1]; conflictIndex++)
		{
			const PxU32 neighbor = compiled.particleStart +
				compiled.particlePrimalStructuralConflictIndices[conflictIndex];
			if(workspace.particlePrimalColors[neighbor] == color)
				return;
		}
		for(PxU32 conflictIndex =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
			conflictIndex <
				workspace.particlePrimalDynamicConflictOffsets[
					particleIndex + 1]; conflictIndex++)
		{
			const PxU32 neighbor =
				workspace.particlePrimalDynamicConflictIndices[conflictIndex];
			if(workspace.particlePrimalColors[neighbor] == color)
				return;
		}
	}

	workspace.resize(workspace.particlePrimalColorCounts, colorCount);
	workspace.resize(workspace.particlePrimalColorOffsets, colorCount + 1);
	for(PxU32 color = 0; color < colorCount; color++)
		workspace.particlePrimalColorCounts[color] = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalColorCounts[
			workspace.particlePrimalColors[particleIndex]]++;
	workspace.particlePrimalColorOffsets[0] = 0;
	for(PxU32 color = 0; color < colorCount; color++)
		workspace.particlePrimalColorOffsets[color + 1] =
			workspace.particlePrimalColorOffsets[color] +
			workspace.particlePrimalColorCounts[color];
	workspace.resize(workspace.particlePrimalColorParticles, numParticles);
	for(PxU32 color = 0; color < colorCount; color++)
		workspace.particlePrimalColorCounts[color] = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 color = workspace.particlePrimalColors[particleIndex];
		workspace.particlePrimalColorParticles[
			workspace.particlePrimalColorOffsets[color] +
			workspace.particlePrimalColorCounts[color]++] = particleIndex;
	}
	for(PxU32 color = 0; color < colorCount; color++)
	{
		const PxU32 begin = workspace.particlePrimalColorOffsets[color];
		const PxU32 end = workspace.particlePrimalColorOffsets[color + 1];
		for(PxU32 packedIndex = begin; packedIndex < end; packedIndex++)
		{
			if(workspace.particlePrimalColorParticles[packedIndex] >=
				numParticles ||
				workspace.particlePrimalColors[
					workspace.particlePrimalColorParticles[packedIndex]] != color ||
				(packedIndex > begin &&
					workspace.particlePrimalColorParticles[
						packedIndex - 1] >=
					workspace.particlePrimalColorParticles[packedIndex]))
				return;
		}
	}
	workspace.particlePrimalColorCount = colorCount;
	workspace.particlePrimalColorPlanValid = true;
}

// True-boundary collision vertices are normally expanded by Scene before they
// reach this component.  The eventual ground-patch experiment is meaningful
// only when one such vertex maps strictly inside one simulation tet.  Keep
// this qualification independent from solver ownership: it reads immutable
// contact IR and never changes a contact, particle, plan, or workspace.
PX_FORCE_INLINE bool avbdCollectGroundTetPatchFourSupport(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody& body, const AvbdSoftParticle* particles,
	PxU32 numParticles, PxU32 outSupport[4])
{
	if(!particles || geometry.queryPoint.count != 4 ||
		body.compiled.particleStart > numParticles ||
		body.compiled.particleCount >
			numParticles - body.compiled.particleStart)
		return false;

	PxReal weightSum = 0.0f;
	for(PxU32 supportIndex = 0; supportIndex < 4; ++supportIndex)
	{
		const PxU32 particleIndex =
			geometry.queryPoint.particleIndices[supportIndex];
		const PxReal weight = geometry.queryPoint.weights[supportIndex];
		if(particleIndex < body.compiled.particleStart ||
			particleIndex >= body.compiled.particleStart +
				body.compiled.particleCount ||
			!PxIsFinite(weight) || weight <= 1.0e-8f ||
			particles[particleIndex].invMass <= 0.0f ||
			!PxIsFinite(particles[particleIndex].invMass) ||
			!particles[particleIndex].position.isFinite())
			return false;
		for(PxU32 earlier = 0; earlier < supportIndex; ++earlier)
			if(outSupport[earlier] == particleIndex)
				return false;
		outSupport[supportIndex] = particleIndex;
		weightSum += weight;
	}
	return PxIsFinite(weightSum) && PxAbs(weightSum - 1.0f) <= 1.0e-3f;
}

PX_FORCE_INLINE bool avbdFindGroundTetPatchSingleTet(
	const AvbdSoftBody& body, const PxU32 support[4], PxU32& outTetIndex)
{
	outTetIndex = PX_MAX_U32;
	if(body.compiled.particleCount == 0)
		return false;

	const PxU32 firstLocalIndex =
		support[0] - body.compiled.particleStart;
	if(firstLocalIndex >= body.compiled.elementAdjacency.size())
		return false;
	const PxArray<AvbdParticleElementRef>& tetRefs =
		body.compiled.elementAdjacency[firstLocalIndex].tetRefs;
	for(PxU32 tetRefOffset = 0; tetRefOffset < tetRefs.size();
		++tetRefOffset)
	{
		const PxU32 tetIndex = tetRefs[tetRefOffset].index;
		if(tetIndex >= body.compiled.tetElements.size())
			continue;
		const AvbdTetElement& tet = body.compiled.tetElements[tetIndex];
		const PxU32 tetVertices[4] = {tet.p0, tet.p1, tet.p2, tet.p3};
		bool sameSet = true;
		for(PxU32 supportIndex = 0; supportIndex < 4 && sameSet;
			supportIndex++)
		{
			bool found = false;
			for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
				if(tetVertices[vertexIndex] == support[supportIndex])
				{
					found = true;
					break;
				}
			if(!found)
				sameSet = false;
		}
		if(sameSet)
		{
			outTetIndex = tetIndex;
			return true;
		}
	}
	return false;
}

PX_NOINLINE inline void avbdAccumulateGroundTetPatchProbe(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBodyStepStats& stepStats)
{
	if(!softBodies || !contacts || !particles)
		return;
	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		const AvbdSoftContact& contact = contacts[contactIndex];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		if(geometry.source.type != AvbdSoftContactSource::eGROUND ||
			!geometry.hasWorldStaticTarget() ||
			geometry.velocityOwner !=
				AvbdVelocityObjectiveOwner::PositionAL)
			continue;
		stepStats.groundTetPatchGroundPositionAlRows++;
		if(geometry.queryBodyIndex >= numSoftBodies ||
			geometry.queryPoint.count != 4)
			continue;
		const AvbdSoftBody& body = softBodies[geometry.queryBodyIndex];
		if(body.compiled.speculativeCCDEnabled)
			continue;
		PxU32 support[4];
		if(!avbdCollectGroundTetPatchFourSupport(
				geometry, body, particles, numParticles, support))
			continue;
		stepStats.groundTetPatchFourSupportRows++;
		PxU32 tetIndex = PX_MAX_U32;
		if(!avbdFindGroundTetPatchSingleTet(body, support, tetIndex))
			continue;
		stepStats.groundTetPatchSingleTetRows++;
		const PxVec3 surfacePoint =
			avbdGetSoftContactSurfacePoint(geometry, particles);
		if(!surfacePoint.isFinite())
			continue;
		const AvbdSoftContactRowForces rowForces =
			avbdEvaluateSoftContactRowForces(
				geometry, contact.state, particles, surfacePoint);
		if(PxIsFinite(rowForces.normal) && rowForces.normal < 0.0f)
			stepStats.groundTetPatchActiveRows++;
	}
}

// The velocity tangent path is intentionally narrower than the normal
// Position-AL path.  It is admitted only after Scene has expanded the
// collision proxy into the final simulation-particle support.  That keeps the
// impulse lever arm, mass response and ownership checks in the same domain as
// the particle solver.
bool avbdCanUseVelocityTangentOwner(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles)
{
	const bool supportedWorldStaticSource =
		geometry.source.type == AvbdSoftContactSource::eGROUND ||
		geometry.source.type == AvbdSoftContactSource::eRIGID_SDF;
	const bool supportedWorldStatic = supportedWorldStaticSource &&
		geometry.hasWorldStaticTarget();
	const bool supportedSoftSoft =
		geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
		geometry.targetKind ==
			AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
		(geometry.hasWeightedTargetPoint() ||
		 geometry.hasDeformableSurfaceTarget());
	const bool supportedDynamicRigid =
		geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
		geometry.hasRigidBodyTarget();
	const bool supportedVelocityOwner = supportedDynamicRigid
		? geometry.velocityOwner ==
			AvbdVelocityObjectiveOwner::ManifoldFinalize
		: geometry.velocityOwner ==
			AvbdVelocityObjectiveOwner::PositionAL;
	if(!softBodies || !particles ||
		(!supportedWorldStatic && !supportedSoftSoft &&
		 !supportedDynamicRigid) ||
		!supportedVelocityOwner ||
		!PxIsFinite(geometry.friction) || geometry.friction <= 0.0f)
		return false;

	const PxReal normalMagnitudeSq = geometry.normal.magnitudeSquared();
	const PxReal tangent1MagnitudeSq = geometry.tangent1.magnitudeSquared();
	const PxReal tangent2MagnitudeSq = geometry.tangent2.magnitudeSquared();
	if(!geometry.normal.isFinite() || !geometry.tangent1.isFinite() ||
		!geometry.tangent2.isFinite() || !PxIsFinite(normalMagnitudeSq) ||
		!PxIsFinite(tangent1MagnitudeSq) ||
		!PxIsFinite(tangent2MagnitudeSq) ||
		PxAbs(normalMagnitudeSq - 1.0f) > 1.0e-3f ||
		PxAbs(tangent1MagnitudeSq - 1.0f) > 1.0e-3f ||
		PxAbs(tangent2MagnitudeSq - 1.0f) > 1.0e-3f ||
		PxAbs(geometry.normal.dot(geometry.tangent1)) > 1.0e-3f ||
		PxAbs(geometry.normal.dot(geometry.tangent2)) > 1.0e-3f ||
		PxAbs(geometry.tangent1.dot(geometry.tangent2)) > 1.0e-3f)
		return false;

	PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
	const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
		geometry, supportIndices);
	if(supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
		return false;
	PxReal response = 0.0f;
	for(PxU32 supportIndex = 0; supportIndex < supportCount; ++supportIndex)
	{
		const PxU32 particleIndex = supportIndices[supportIndex];
		if(particleIndex >= numParticles)
			return false;
		const AvbdSoftBody* body = avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, particleIndex);
		if(!body || body->compiled.speculativeCCDEnabled ||
			!PxIsFinite(body->compiled.maxDepenetrationVelocity) ||
			body->compiled.maxDepenetrationVelocity < 1.0e20f)
			return false;
		const AvbdSoftParticle& particle = particles[particleIndex];
		const PxReal weight = avbdGetSoftContactParticleJacobianScale(
			geometry, particleIndex);
		if(!PxIsFinite(weight) || PxAbs(weight) <= 1.0e-8f ||
			particle.invMass < 0.0f || !PxIsFinite(particle.invMass) ||
			!particle.position.isFinite() || !particle.velocity.isFinite())
			return false;
		response += weight * weight * particle.invMass;
	}
	const bool fullyKinematicDynamicSource = supportedDynamicRigid &&
		avbdIsSoftContactQueryFullyKinematic(
			geometry, particles, numParticles);
	return PxIsFinite(response) &&
		(response > 1.0e-12f || fullyKinematicDynamicSource);
}

void avbdAssignVelocityTangentOwners(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles)
{
	if(!contacts || !particles ||
		!avbdUseVelocityTangentOwner())
		return;
	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		AvbdSoftContactGeometry& geometry = contact.geometry;
		const AvbdSoftContactTangentOwner previousOwner =
			geometry.tangentOwner;
		const bool useVelocityOwner =
			avbdCanUseVelocityTangentOwner(
				geometry, softBodies, numSoftBodies, particles,
				numParticles);
		geometry.tangentOwner = useVelocityOwner
			? AvbdSoftContactTangentOwner::eVELOCITY
			: AvbdSoftContactTangentOwner::ePOSITION_AL;
		const bool preservePrescribedSurfaceWitness = useVelocityOwner &&
			geometry.hasRigidBodyTarget() &&
			avbdIsSoftContactQueryFullyKinematic(
				geometry, particles, numParticles) &&
			contact.state.surfacePointPrev.isFinite();
		const PxVec3 previousSurfacePoint =
			contact.state.surfacePointPrev;
		// Any owner transition invalidates the old tangent state.  In
		// particular, a row that was velocity-owned in a prior epoch must not
		// later resurrect a stale Position-AL spring when its eligibility drops.
		// A prescribed source is the one exception for the geometric anchor:
		// its previous surface witness is the authoritative source velocity.
		if(useVelocityOwner || previousOwner != geometry.tangentOwner)
		{
			avbdResetSoftContactTangentState(
				geometry, contact.state, particles);
			if(preservePrescribedSurfaceWitness)
				contact.state.surfacePointPrev = previousSurfacePoint;
		}
	}
}

// Terminal tangent projection for rows whose Position-AL normal was retained
// but whose tangent was deliberately removed from the primal and dual paths.
// It uses the final simulation-particle support, applies no normal impulse,
// and does not write any AL state.  For one row the disk projection is
// non-energy-injecting in the particle inv-mass metric.
PX_NOINLINE void avbdProjectSoftContactVelocityTangents(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, AvbdSoftBodyStepStats* stepStats,
	const AvbdOgcGeometryEpochView* geometryEpoch)
{
	// The policy is process-static, so this is one cached branch per step; an
	// exact-zero diagnostic rollback keeps the legacy path free of this scan.
	if(!avbdUseVelocityTangentOwner() ||
		!particles || !contacts || !softBodies || dt <= 0.0f ||
		!PxIsFinite(dt))
		return;

	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		const AvbdSoftContactAugmentedState& state = contact.state;
		if(geometry.tangentOwner !=
			AvbdSoftContactTangentOwner::eVELOCITY ||
			geometry.hasRigidBodyTarget() ||
			!avbdCanUseVelocityTangentOwner(
				geometry, softBodies, numSoftBodies, particles,
				numParticles) ||
			!PxIsFinite(state.alLambda) || state.alLambda >= 0.0f)
			continue;

		if(stepStats)
			stepStats->worldStaticVelocityTangentOwnerRows++;
		if(geometry.hasWorldStaticTarget())
		{
			const AvbdOgcRigidBoxGeometry* rigidBox = geometryEpoch
				? geometryEpoch->getRigidBox(contactIndex, numContacts)
				: NULL;
			AvbdOgcTangentResponse response;
			const bool applied = compileCurrentOgcTangentResponse(
				geometry, particles, numParticles, NULL, response, rigidBox) &&
				applyOgcTangentVelocityResponse(
					response, contact, particles, numParticles, NULL, dt);
			if(applied && stepStats)
				stepStats->worldStaticVelocityTangentAppliedRows++;
			continue;
		}
		if(geometry.targetKind !=
			AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE)
			continue;
		PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
			geometry, supportIndices);
		PxReal response = 0.0f;
		PxVec3 relativeVelocity(0.0f);
		bool valid = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			const AvbdSoftParticle& particle = particles[particleIndex];
			if(!particle.velocity.isFinite())
			{
				valid = false;
				break;
			}
			response += weight * weight * particle.invMass;
			relativeVelocity += particle.velocity * weight;
		}
		if(!valid || !PxIsFinite(response) || response <= 1.0e-12f ||
			!relativeVelocity.isFinite())
			continue;

		const PxReal tangentVelocity0 =
			relativeVelocity.dot(geometry.tangent1);
		const PxReal tangentVelocity1 =
			relativeVelocity.dot(geometry.tangent2);
		if(!PxIsFinite(tangentVelocity0) || !PxIsFinite(tangentVelocity1))
			continue;
		PxReal tangentImpulse0 = -tangentVelocity0 / response;
		PxReal tangentImpulse1 = -tangentVelocity1 / response;
		const PxReal normalImpulseBudget =
			PxMax(-state.alLambda, 0.0f) * dt;
		const PxReal tangentImpulseLimit =
			geometry.friction * normalImpulseBudget;
		if(!PxIsFinite(normalImpulseBudget) ||
			!PxIsFinite(tangentImpulseLimit) || tangentImpulseLimit < 0.0f)
			continue;
		const PxReal tangentImpulseMagnitude = PxSqrt(
			tangentImpulse0 * tangentImpulse0 +
			tangentImpulse1 * tangentImpulse1);
		if(!PxIsFinite(tangentImpulseMagnitude))
			continue;
		if(tangentImpulseMagnitude > tangentImpulseLimit &&
			tangentImpulseMagnitude > 1.0e-12f)
		{
			const PxReal scale = tangentImpulseLimit /
				tangentImpulseMagnitude;
			tangentImpulse0 *= scale;
			tangentImpulse1 *= scale;
		}
		if(PxAbs(tangentImpulse0) <= 1.0e-12f &&
			PxAbs(tangentImpulse1) <= 1.0e-12f)
			continue;
		const PxVec3 tangentImpulse =
			geometry.tangent1 * tangentImpulse0 +
			geometry.tangent2 * tangentImpulse1;
		if(!tangentImpulse.isFinite())
			continue;

		PxVec3 updatedVelocities[AVBD_CONTACT_MAX_PARTICLES];
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			updatedVelocities[supportIndex] = particles[particleIndex].velocity +
				tangentImpulse * (particles[particleIndex].invMass * weight);
			// Match the finite-speed envelope used by
			// updateVelocityFromPosition().  This pass runs after that rebuild,
			// so it must fail closed rather than reintroduce an unbounded terminal
			// velocity through an otherwise finite multiplier.
			if(!updatedVelocities[supportIndex].isFinite() ||
				PxAbs(updatedVelocities[supportIndex].x) > 1.0e6f ||
				PxAbs(updatedVelocities[supportIndex].y) > 1.0e6f ||
				PxAbs(updatedVelocities[supportIndex].z) > 1.0e6f)
			{
				valid = false;
				break;
			}
		}
		if(!valid)
			continue;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			particles[particleIndex].velocity = updatedVelocities[supportIndex];
		}
		if(stepStats)
			stepStats->worldStaticVelocityTangentAppliedRows++;
	}
}

// Re-query one world-static contact against its authoritative current shape.
// This is intentionally a discrete endpoint query: unlike the Position-AL
// normal constraint it does not include the OGC shell margin, and it never
// tests an old-to-new segment.  Box rows retain immutable shape metadata;
// planes and legacy static rows retain their world-space surface point.
PX_FORCE_INLINE bool avbdGetCurrentWorldStaticEndpointDcdGeometry(
	const AvbdSoftContactGeometry& geometry, const PxVec3& queryPoint,
	PxVec3& normal, PxReal& trueGap,
	const AvbdOgcRigidBoxGeometry* rigidBox)
{
	if(!geometry.hasWorldStaticTarget() || !queryPoint.isFinite())
		return false;

	if(!rigidBox)
	{
		const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
		if(!geometry.normal.isFinite() || !PxIsFinite(normalLengthSq) ||
			normalLengthSq <= 1.0e-12f || !geometry.surfacePoint.isFinite())
			return false;
		normal = geometry.normal * PxRecipSqrt(normalLengthSq);
		trueGap = (queryPoint - geometry.surfacePoint).dot(normal);
		return PxIsFinite(trueGap);
	}

	const PxVec3 halfExtent = rigidBox->halfExtent;
	const PxTransform& shapeToWorld = rigidBox->shapeToTarget;
	const PxReal rotationLengthSq =
		shapeToWorld.q.magnitudeSquared();
	if(!halfExtent.isFinite() || halfExtent.x <= 0.0f ||
		halfExtent.y <= 0.0f || halfExtent.z <= 0.0f ||
		!shapeToWorld.p.isFinite() || !shapeToWorld.q.isFinite() ||
		!PxIsFinite(rotationLengthSq) || rotationLengthSq <= 1.0e-12f)
		return false;

	const PxVec3 localPoint = shapeToWorld.transformInv(queryPoint);
	if(!localPoint.isFinite())
		return false;
	const PxVec3 q(
		PxAbs(localPoint.x) - halfExtent.x,
		PxAbs(localPoint.y) - halfExtent.y,
		PxAbs(localPoint.z) - halfExtent.z);
	const bool inside = q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f;
	PxVec3 localNormal(0.0f);
	if(inside)
	{
		trueGap = PxMax(q.x, PxMax(q.y, q.z));
		if(q.x > q.y && q.x > q.z)
			localNormal = PxVec3(localPoint.x >= 0.0f ? 1.0f : -1.0f,
				0.0f, 0.0f);
		else if(q.y > q.z)
			localNormal = PxVec3(0.0f,
				localPoint.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
		else
			localNormal = PxVec3(0.0f, 0.0f,
				localPoint.z >= 0.0f ? 1.0f : -1.0f);
	}
	else
	{
		const PxVec3 outside(
			PxMax(q.x, 0.0f), PxMax(q.y, 0.0f), PxMax(q.z, 0.0f));
		trueGap = outside.magnitude();
		if(!PxIsFinite(trueGap) || trueGap <= 1.0e-12f)
			return false;
		localNormal = PxVec3(
			(localPoint.x >= 0.0f ? 1.0f : -1.0f) * outside.x,
			(localPoint.y >= 0.0f ? 1.0f : -1.0f) * outside.y,
			(localPoint.z >= 0.0f ? 1.0f : -1.0f) * outside.z) /
			trueGap;
	}

	normal = shapeToWorld.q.rotate(localNormal);
	const PxReal normalLengthSq = normal.magnitudeSquared();
	if(!normal.isFinite() || !PxIsFinite(normalLengthSq) ||
		normalLengthSq <= 1.0e-12f || !PxIsFinite(trueGap))
		return false;
	normal *= PxRecipSqrt(normalLengthSq);
	return true;
}

PX_FORCE_INLINE const AvbdSoftBody*
avbdFindWorldStaticEndpointDcdSourceBody(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 numParticles)
{
	if(!softBodies || numSoftBodies == 0)
		return NULL;
	const PxU32 representative = geometry.hasWeightedQueryPoint()
		? geometry.queryPoint.particleIndices[0]
		: geometry.hasBarycentricQueryPoint()
			? geometry.queryParticleIndices[0] : geometry.particleIdx;
	if(representative >= numParticles)
		return NULL;
	if(geometry.queryBodyIndex < numSoftBodies &&
		avbdSoftBodyContainsParticle(
			softBodies[geometry.queryBodyIndex], representative,
			numParticles))
		return &softBodies[geometry.queryBodyIndex];
	return avbdFindSoftBodyForParticle(
		softBodies, numSoftBodies, representative);
}

// Component fallback does not own a movable rigid endpoint, but it can still
// receive current-pose ground and world-static box contacts.  This must use
// the same *local* support recovery as the native mixed path.  Translating a
// complete soft body out of a static target makes a falling volume appear
// kinematic (and, after the velocity anchor is translated with it, removes
// the normal component of gravity).  Recover only the weighted query support
// instead, then let the ordinary material rows distribute that load through
// the tetrahedra.
PX_NOINLINE void avbdApplyWorldStaticComponentEndpointDcdRecovery(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	AvbdSoftBodyWorkspace& workspace, PxU32 sweeps)
{
	workspace.resize(workspace.worldStaticEndpointRecoveredBodies,
		numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		workspace.worldStaticEndpointRecoveredBodies[bodyIndex] = 0u;
	if(!particles || !softBodies || !contacts || numParticles == 0 ||
		numSoftBodies == 0 || numContacts == 0 || sweeps == 0)
		return;
	const AvbdOgcGeometryEpochView geometryEpoch =
		workspace.getComponentOgcGeometryEpochView();

	for(PxU32 sweep = 0; sweep < sweeps; ++sweep)
	{
		bool appliedAny = false;
		for(PxU32 contactIndex = 0; contactIndex < numContacts;
			++contactIndex)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[contactIndex].geometry;
			const AvbdOgcRigidBoxGeometry* rigidBox =
				geometryEpoch.getRigidBox(contactIndex, numContacts);
			if((geometry.source.type != AvbdSoftContactSource::eGROUND &&
				(geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
					!rigidBox)) ||
				!geometry.hasWorldStaticTarget() ||
				geometry.velocityOwner !=
					AvbdVelocityObjectiveOwner::PositionAL ||
				!avbdHasSoftContactDynamicQuerySupport(
					geometry, particles, numParticles))
				continue;

			const AvbdSoftBody* sourceBody =
				avbdFindWorldStaticEndpointDcdSourceBody(
					geometry, softBodies, numSoftBodies, numParticles);
			if(!sourceBody || sourceBody->compiled.speculativeCCDEnabled ||
				!PxIsFinite(sourceBody->compiled.maxDepenetrationVelocity) ||
				sourceBody->compiled.maxDepenetrationVelocity < 1.0e20f)
				continue;
			const PxU32 bodyIndex = PxU32(sourceBody - softBodies);
			if(bodyIndex >= numSoftBodies)
				continue;

			const PxVec3 queryPoint =
				avbdGetSoftContactQueryPoint(geometry, particles);
			PxVec3 normal(0.0f);
			PxReal trueGap = 0.0f;
			if(!queryPoint.isFinite() ||
				!avbdGetCurrentWorldStaticEndpointDcdGeometry(
					geometry, queryPoint, normal, trueGap, rigidBox) ||
				!(trueGap < 0.0f) || !PxIsFinite(trueGap))
				continue;

			PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
			const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
				geometry, supportIndices);
			if(supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
				continue;

			PxReal response = 0.0f;
			PxReal weights[AVBD_CONTACT_MAX_PARTICLES];
			PxVec3 deltas[AVBD_CONTACT_MAX_PARTICLES];
			bool validSupport = true;
			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const PxU32 particleIndex = supportIndices[supportIndex];
				if(particleIndex >= numParticles ||
					!avbdSoftBodyContainsParticle(
						*sourceBody, particleIndex, numParticles))
				{
					validSupport = false;
					break;
				}
				const AvbdSoftParticle& particle = particles[particleIndex];
				const PxReal weight = avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex);
				if(!PxIsFinite(weight) || !PxIsFinite(particle.invMass) ||
					particle.invMass < 0.0f || !particle.position.isFinite() ||
					!particle.initialPosition.isFinite())
				{
					validSupport = false;
					break;
				}
				weights[supportIndex] = weight;
				response += particle.invMass * weight * weight;
			}
			if(!validSupport || !PxIsFinite(response) || response <= 1.0e-12f)
				continue;

			const PxReal lambda = -trueGap / response;
			if(!PxIsFinite(lambda) || lambda <= 0.0f)
				continue;
			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const AvbdSoftParticle& particle =
					particles[supportIndices[supportIndex]];
				deltas[supportIndex] = normal *
					(particle.invMass * weights[supportIndex] * lambda);
				if(!deltas[supportIndex].isFinite() ||
					!(particle.position + deltas[supportIndex]).isFinite() ||
					!(particle.initialPosition + deltas[supportIndex]).isFinite())
				{
					validSupport = false;
					break;
				}
			}
			if(!validSupport)
				continue;

			// The support vertices move as one mass-weighted contact block.  A
			// single alpha preserves that relation and lets the exact incident-tet
			// test reject a correction before it can turn a resting contact into
			// an inversion.
			PxReal commonAlpha = 1.0f;
			bool accepted = false;
			for(PxU32 attempt = 0; attempt < 8u && !accepted; ++attempt)
			{
				bool candidateValid = true;
				auto candidatePositionFor = [&supportIndices, &deltas,
					particles, supportCount, commonAlpha](PxU32 particleIndex)
					-> PxVec3
				{
					for(PxU32 i = 0; i < supportCount; ++i)
						if(supportIndices[i] == particleIndex)
							return particles[particleIndex].position +
								deltas[i] * commonAlpha;
					return particles[particleIndex].position;
				};
				bool hasSubthresholdTet = false;
				bool improvesSubthresholdTet = false;
				for(PxU32 supportIndex = 0;
					supportIndex < supportCount && candidateValid; ++supportIndex)
				{
					const PxU32 particleIndex = supportIndices[supportIndex];
					const PxU32 localIndex = particleIndex -
						sourceBody->compiled.particleStart;
					if(localIndex >= sourceBody->compiled.elementAdjacency.size())
					{
						candidateValid = false;
						break;
					}
					const AvbdParticleElementAdjacency& adjacency =
						sourceBody->compiled.elementAdjacency[localIndex];
					for(PxU32 refIndex = 0;
						refIndex < adjacency.tetRefs.size(); ++refIndex)
					{
						const AvbdParticleElementRef& ref =
							adjacency.tetRefs[refIndex];
						if(ref.index >= sourceBody->compiled.tetElements.size())
						{
							candidateValid = false;
							break;
						}
						const AvbdTetElement& tet =
							sourceBody->compiled.tetElements[ref.index];
						if(tet.p0 >= numParticles || tet.p1 >= numParticles ||
							tet.p2 >= numParticles || tet.p3 >= numParticles)
						{
							candidateValid = false;
							break;
						}
						const PxVec3 currentP0 = particles[tet.p0].position;
						const PxVec3 currentE1 = particles[tet.p1].position - currentP0;
						const PxVec3 currentE2 = particles[tet.p2].position - currentP0;
						const PxVec3 currentE3 = particles[tet.p3].position - currentP0;
						PxReal currentDeterminant;
						PxVec3 unusedGradient;
						avbdEvaluateTetDeterminantAndGradient(
							tet, 0u, currentE1, currentE2, currentE3,
							currentDeterminant, unusedGradient);
						const PxVec3 candidateP0 = candidatePositionFor(tet.p0);
						const PxVec3 candidateE1 =
							candidatePositionFor(tet.p1) - candidateP0;
						const PxVec3 candidateE2 =
							candidatePositionFor(tet.p2) - candidateP0;
						const PxVec3 candidateE3 =
							candidatePositionFor(tet.p3) - candidateP0;
						PxReal candidateDeterminant;
						avbdEvaluateTetDeterminantAndGradient(
							tet, 0u, candidateE1, candidateE2, candidateE3,
							candidateDeterminant, unusedGradient);
						if(!PxIsFinite(currentDeterminant) ||
							!PxIsFinite(candidateDeterminant))
						{
							candidateValid = false;
							break;
						}
						if(currentDeterminant >= 0.05f)
						{
							if(candidateDeterminant < 0.05f)
							{
								candidateValid = false;
								break;
							}
						}
						else
						{
							hasSubthresholdTet = true;
							if(candidateDeterminant + 1.0e-6f <
								currentDeterminant)
							{
								candidateValid = false;
								break;
							}
							if(candidateDeterminant >
								currentDeterminant + 1.0e-6f)
								improvesSubthresholdTet = true;
						}
					}
				}
				if(candidateValid && hasSubthresholdTet &&
					!improvesSubthresholdTet)
					candidateValid = false;

				PxVec3 candidateQuery = queryPoint;
				for(PxU32 supportIndex = 0;
					supportIndex < supportCount && candidateValid; ++supportIndex)
					candidateQuery += deltas[supportIndex] *
						(commonAlpha * weights[supportIndex]);
				PxVec3 candidateNormal(0.0f);
				PxReal candidateGap = 0.0f;
				if(!candidateQuery.isFinite() ||
					!avbdGetCurrentWorldStaticEndpointDcdGeometry(
						geometry, candidateQuery, candidateNormal, candidateGap,
						rigidBox) ||
					!PxIsFinite(candidateGap) ||
					candidateGap <= trueGap + 1.0e-6f)
					candidateValid = false;
				if(candidateValid)
					accepted = true;
				else
					commonAlpha *= 0.5f;
			}
			if(!accepted)
				continue;

			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const PxVec3 delta = deltas[supportIndex] * commonAlpha;
				if(delta.magnitudeSquared() <= 0.0f)
					continue;
				AvbdSoftParticle& particle =
					particles[supportIndices[supportIndex]];
				particle.position += delta;
				// Geometric recovery is not a rebound impulse.  Moving this
				// support's reconstruction anchor prevents a one-frame launch,
				// while untouched body particles keep their gravity motion.
				particle.initialPosition += delta;
			}
			workspace.worldStaticEndpointRecoveredBodies[bodyIndex] = 1u;
			appliedAny = true;
		}
		if(!appliedAny)
			break;
	}
}

// Position recovery moves the velocity anchor along with the particles, so it
// deliberately does not manufacture a separating bounce.  Remove only the
// remaining inward normal velocity at recovered current-pose rows after the
// ordinary position-to-velocity rebuild and component finalization.
PX_NOINLINE void avbdClampWorldStaticComponentEndpointDcdVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBodyWorkspace& workspace)
{
	const PxArray<PxU8>& recoveredBodies =
		workspace.worldStaticEndpointRecoveredBodies;
	if(!particles || !softBodies || !contacts ||
		recoveredBodies.size() != numSoftBodies)
		return;
	const AvbdOgcGeometryEpochView geometryEpoch =
		workspace.getComponentOgcGeometryEpochView();

	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		const AvbdSoftContactGeometry& geometry = contacts[contactIndex].geometry;
		const AvbdOgcRigidBoxGeometry* rigidBox =
			geometryEpoch.getRigidBox(contactIndex, numContacts);
		if((geometry.source.type != AvbdSoftContactSource::eGROUND &&
			(geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
				!rigidBox)) ||
			!geometry.hasWorldStaticTarget() ||
			geometry.velocityOwner != AvbdVelocityObjectiveOwner::PositionAL ||
			!avbdHasSoftContactDynamicQuerySupport(
				geometry, particles, numParticles))
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindWorldStaticEndpointDcdSourceBody(
				geometry, softBodies, numSoftBodies, numParticles);
		if(!sourceBody || sourceBody->compiled.speculativeCCDEnabled)
			continue;
		const PxU32 bodyIndex = PxU32(sourceBody - softBodies);
		if(bodyIndex >= numSoftBodies || recoveredBodies[bodyIndex] == 0u)
			continue;

		const PxVec3 queryPoint =
			avbdGetSoftContactQueryPoint(geometry, particles);
		PxVec3 normal(0.0f);
		PxReal trueGap = 0.0f;
		if(!queryPoint.isFinite() ||
			!avbdGetCurrentWorldStaticEndpointDcdGeometry(
				geometry, queryPoint, normal, trueGap, rigidBox) ||
			!PxIsFinite(trueGap) || trueGap > 1.0e-3f)
			continue;

		PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
			geometry, particleIndices);
		if(supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
			continue;
		PxReal response = 0.0f;
		PxVec3 queryVelocity(0.0f);
		bool valid = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = particleIndices[supportIndex];
			if(particleIndex >= numParticles)
			{
				valid = false;
				break;
			}
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			const AvbdSoftParticle& particle = particles[particleIndex];
			if(!PxIsFinite(weight) || !PxIsFinite(particle.invMass) ||
				!particle.velocity.isFinite())
			{
				valid = false;
				break;
			}
			response += particle.invMass * weight * weight;
			queryVelocity += particle.velocity * weight;
		}
		if(!valid || !PxIsFinite(response) || response <= 1.0e-12f ||
			!queryVelocity.isFinite())
			continue;
		const PxReal normalVelocity = queryVelocity.dot(normal);
		if(!PxIsFinite(normalVelocity) || normalVelocity >= -1.0e-6f)
			continue;
		const PxReal impulse = -normalVelocity / response;
		if(!PxIsFinite(impulse) || impulse <= 0.0f)
			continue;

		PxVec3 candidateVelocities[AVBD_CONTACT_MAX_PARTICLES];
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = particleIndices[supportIndex];
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			const AvbdSoftParticle& particle = particles[particleIndex];
			candidateVelocities[supportIndex] = particle.velocity +
				normal * (particle.invMass * weight * impulse);
			if(!candidateVelocities[supportIndex].isFinite() ||
				PxAbs(candidateVelocities[supportIndex].x) > 1.0e6f ||
				PxAbs(candidateVelocities[supportIndex].y) > 1.0e6f ||
				PxAbs(candidateVelocities[supportIndex].z) > 1.0e6f)
			{
				valid = false;
				break;
			}
		}
		if(!valid)
			continue;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
			particles[particleIndices[supportIndex]].velocity =
				candidateVelocities[supportIndex];
	}
}

// Rebuild the redetection-epoch particle incidence and, when explicitly
// requested, its P4 causal access plan.  This has no caller-stack capture, so
// the persistent component step state can invoke it only at its serial
// redetection barriers.  It must never be called by a particle range task.
void avbdBuildSoftParticleContactIndex(
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles, AvbdSoftBodyStepStats* stepStats,
	AvbdParticlePrimalSchedule particlePrimalSchedule,
	bool validateP4AccessPlan,
	const AvbdSoftParticle* probeParticles)
{
	if(contacts && probeParticles)
		avbdAssignVelocityTangentOwners(
			contacts, numContacts, softBodies, numSoftBodies,
			probeParticles, numParticles);
	if(stepStats && probeParticles && avbdUseGroundTetPatchProbe())
		avbdAccumulateGroundTetPatchProbe(
			softBodies, numSoftBodies, contacts, numContacts,
			probeParticles, numParticles, *stepStats);
	workspace.resize(workspace.contactStarts, numParticles + 1);
	PxArray<AvbdSoftContactParticleRef>& contactIdxBuf =
		workspace.contactIndices;
	PxArray<PxU32>& contactStart = workspace.contactStarts;
	PxArray<PxU32>& contactCount = workspace.contactCounts;
	if(numContacts == 0)
	{
		std::memset(contactStart.begin(), 0,
			sizeof(PxU32) * (numParticles + 1));
		workspace.resize(contactIdxBuf, 0);
	}
	else
	{
		workspace.resize(workspace.contactCounts, numParticles);
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
			contactCount[particleIndex] = 0;
		for(PxU32 contactIndex = 0; contactIndex < numContacts; contactIndex++)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[contactIndex].geometry;
			PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
			const PxU32 particleIndexCount =
				avbdCollectSoftContactParticleIndices(
					geometry, particleIndices);
			for(PxU32 particleOffset = 0;
				particleOffset < particleIndexCount; particleOffset++)
			{
				const PxU32 particleIndex = particleIndices[particleOffset];
				if(particleIndex >= numParticles)
					continue;
				if(PxAbs(avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex)) > 1e-12f)
					contactCount[particleIndex]++;
			}
		}
		contactStart[0] = 0;
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
			contactStart[particleIndex + 1] =
				contactStart[particleIndex] + contactCount[particleIndex];
		workspace.resize(contactIdxBuf, contactStart[numParticles]);
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
			contactCount[particleIndex] = 0;
		for(PxU32 contactIndex = 0; contactIndex < numContacts; contactIndex++)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[contactIndex].geometry;
			PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
			const PxU32 particleIndexCount =
				avbdCollectSoftContactParticleIndices(
					geometry, particleIndices);
			for(PxU32 particleOffset = 0;
				particleOffset < particleIndexCount; particleOffset++)
			{
				const PxU32 particleIndex = particleIndices[particleOffset];
				if(particleIndex >= numParticles)
					continue;
				const PxReal jacobianScale =
					avbdGetSoftContactParticleJacobianScale(
						geometry, particleIndex);
				if(PxAbs(jacobianScale) <= 1e-12f)
					continue;
				contactIdxBuf[
					contactStart[particleIndex] +
						contactCount[particleIndex]++] =
					AvbdSoftContactParticleRef(
						contactIndex, jacobianScale);
			}
		}
	}
	workspace.peakContactIncidenceCount = PxMax(
		workspace.peakContactIncidenceCount, contactIdxBuf.size());
	workspace.peakContactIncidenceCapacity = PxMax(
		workspace.peakContactIncidenceCapacity, contactIdxBuf.capacity());
	const bool buildP4AccessPlan = validateP4AccessPlan ||
		avbdUsesColoredParticlePrimalSchedule(particlePrimalSchedule);
	if(buildP4AccessPlan)
	{
		avbdBuildParticlePrimalColorPlan(
			workspace, softBodies, numSoftBodies,
			contacts, numContacts, numParticles,
			particlePrimalSchedule);
		// The explicit validation mode turns a missing descriptor into a
		// Checked failure. The experimental colored schedule deliberately
		// does not: capacity or graph rejection must retain serial GS.
		if(validateP4AccessPlan)
		{
			PX_ASSERT(workspace.particlePrimalDynamicConflictValid);
			PX_ASSERT(workspace.particlePrimalColorPlanValid);
		}
	}
	if(stepStats && workspace.particlePrimalColorPlanValid)
	{
		stepStats->particlePrimalColorCount = PxMax(
			stepStats->particlePrimalColorCount,
			workspace.particlePrimalColorCount);
		stepStats->particlePrimalDynamicAccessGroupCount = PxMax(
			stepStats->particlePrimalDynamicAccessGroupCount,
			workspace.particlePrimalDynamicAccessGroups.size());
	}
}

// Rejected research prototype: the pre-sweep support takeover replaced whole
// vertex subproblems rather than the one contact row.  Keep it out of the
// build while the row-owned post-sweep formulation is developed separately.
#if 0
// Default-off, one-row PositionAL block experiment.  It intentionally takes
// over only the first stable eligible row in a sweep; that makes every
// endpoint write serial and avoids claiming that a contact-row coloring scheme
// already exists.  The owning caller skips the marked particles in the normal
// scalar traversal only after this function succeeds.
PX_NOINLINE inline bool avbdTryApplySoftSoftPositionRowBlock(
	const AvbdParticlePrimalSolveContext& solveContext,
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles,
	AvbdParticlePrimalRangeObservation& observation)
{
	if(!solveContext.particles || !solveContext.selfCollisionSafetyBounds ||
		!softBodies || !contacts || numParticles == 0)
		return false;
	workspace.resize(workspace.softSoftPositionRowBlockOwnedParticles,
		numParticles);
	std::memset(workspace.softSoftPositionRowBlockOwnedParticles.begin(),
		0, sizeof(PxU8) * numParticles);

	for(PxU32 contactIndex = 0; contactIndex < numContacts; ++contactIndex)
	{
		const AvbdSoftContact& contact = contacts[contactIndex];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		const AvbdSoftContactAugmentedState& state = contact.state;
		if(geometry.source.type != AvbdSoftContactSource::eSOFT_SURFACE ||
			geometry.targetKind !=
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE ||
			geometry.velocityOwner !=
				AvbdVelocityObjectiveOwner::PositionAL)
			continue;

		const PxReal normalMagnitudeSq = geometry.normal.magnitudeSquared();
		const PxReal tangent1MagnitudeSq = geometry.tangent1.magnitudeSquared();
		const PxReal tangent2MagnitudeSq = geometry.tangent2.magnitudeSquared();
		if(!geometry.normal.isFinite() || !geometry.tangent1.isFinite() ||
			!geometry.tangent2.isFinite() || !PxIsFinite(normalMagnitudeSq) ||
			!PxIsFinite(tangent1MagnitudeSq) ||
			!PxIsFinite(tangent2MagnitudeSq) ||
			PxAbs(normalMagnitudeSq - 1.0f) > 1.0e-3f ||
			PxAbs(tangent1MagnitudeSq - 1.0f) > 1.0e-3f ||
			PxAbs(tangent2MagnitudeSq - 1.0f) > 1.0e-3f ||
			PxAbs(geometry.normal.dot(geometry.tangent1)) > 1.0e-3f ||
			PxAbs(geometry.normal.dot(geometry.tangent2)) > 1.0e-3f ||
			PxAbs(geometry.tangent1.dot(geometry.tangent2)) > 1.0e-3f ||
			!PxIsFinite(geometry.friction) || geometry.friction <= 0.0f ||
			!PxIsFinite(state.k) || state.k <= 0.0f ||
			!PxIsFinite(state.penTangent[0]) ||
			!PxIsFinite(state.penTangent[1]) ||
			state.penTangent[0] < 0.0f || state.penTangent[1] < 0.0f)
			continue;

		PxU32 queryEndpoint[AVBD_CONTACT_POINT_MAX_SUPPORT];
		PxU32 targetEndpoint[AVBD_CONTACT_POINT_MAX_SUPPORT];
		PxU32 queryEndpointCount = 0;
		PxU32 targetEndpointCount = 0;
		if(!avbdCollectSoftContactEndpointIndices(
				geometry, true, queryEndpoint, queryEndpointCount) ||
			!avbdCollectSoftContactEndpointIndices(
				geometry, false, targetEndpoint, targetEndpointCount) ||
			queryEndpointCount == 0 || targetEndpointCount == 0)
			continue;
		const AvbdSoftBody* queryBody =
			geometry.queryBodyIndex < numSoftBodies
				? &softBodies[geometry.queryBodyIndex]
				: avbdFindSoftBodyForParticle(
					softBodies, numSoftBodies, queryEndpoint[0]);
		const AvbdSoftBody* targetBody =
			geometry.source.targetBodyIndex < numSoftBodies
				? &softBodies[geometry.source.targetBodyIndex]
				: avbdFindSoftBodyForParticle(
					softBodies, numSoftBodies, targetEndpoint[0]);
		if(!queryBody || !targetBody || queryBody == targetBody ||
			queryBody->compiled.speculativeCCDEnabled ||
			targetBody->compiled.speculativeCCDEnabled)
			continue;
		bool endpointsValid = true;
		for(PxU32 i = 0; i < queryEndpointCount; ++i)
			if(!avbdSoftBodyContainsParticle(
					*queryBody, queryEndpoint[i], numParticles))
			{
				endpointsValid = false;
				break;
			}
		for(PxU32 i = 0; i < targetEndpointCount; ++i)
			if(!avbdSoftBodyContainsParticle(
					*targetBody, targetEndpoint[i], numParticles))
			{
				endpointsValid = false;
				break;
			}
		if(!endpointsValid)
			continue;
		const PxU32 queryEnd = queryBody->compiled.particleStart +
			queryBody->compiled.particleCount;
		const PxU32 targetEnd = targetBody->compiled.particleStart +
			targetBody->compiled.particleCount;
		if(queryBody->compiled.particleStart < targetEnd &&
			targetBody->compiled.particleStart < queryEnd)
			continue;

		PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
			geometry, supportIndices);
		if(supportCount < 2 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
			continue;
		// The first diagnostic must not silently turn every other row touching
		// an endpoint into a stale Jacobi contribution.  It only takes over a
		// contact whose dynamic support has no other indexed contact row in
		// this epoch.  That keeps the scalar fallback mathematically local and
		// makes a successful A/B attributable to this one PositionAL row.
		if(!solveContext.contactStarts || !solveContext.contactIndices)
			continue;
		bool isolatedDynamicSupport = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount &&
			isolatedDynamicSupport; ++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			if(particleIndex >= numParticles)
			{
				isolatedDynamicSupport = false;
				break;
			}
			for(PxU32 refIndex = solveContext.contactStarts[particleIndex];
				refIndex < solveContext.contactStarts[particleIndex + 1];
				++refIndex)
			{
				const AvbdSoftContactParticleRef& ref =
					solveContext.contactIndices[refIndex];
				if(ref.contactIndex != contactIndex &&
					PxAbs(ref.jacobianScale) > 1.0e-12f)
				{
					isolatedDynamicSupport = false;
					break;
				}
			}
		}
		if(!isolatedDynamicSupport)
			continue;

		PxMat33 inverseBaseHessian[AVBD_CONTACT_MAX_PARTICLES];
		PxMat33 responseJacobian[AVBD_CONTACT_MAX_PARTICLES];
		PxVec3 baseDisplacement[AVBD_CONTACT_MAX_PARTICLES];
		PxVec3 displacement[AVBD_CONTACT_MAX_PARTICLES];
		const AvbdSoftBody* supportBodies[AVBD_CONTACT_MAX_PARTICLES];
		PxMat33 response(PxZero);
		PxVec3 baseRowDisplacement(0.0f);
		bool valid = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			if(particleIndex >= numParticles)
			{
				valid = false;
				break;
			}
			const PxReal signedScale =
				avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex);
			const AvbdSoftBody* body =
				avbdSoftBodyContainsParticle(*queryBody, particleIndex,
					numParticles) ? queryBody :
				(avbdSoftBodyContainsParticle(*targetBody, particleIndex,
					numParticles) ? targetBody : NULL);
			AvbdSoftParticle& particle = solveContext.particles[particleIndex];
			if(!body || !PxIsFinite(signedScale) ||
				PxAbs(signedScale) <= 1.0e-12f ||
				particle.invMass <= 0.0f || !PxIsFinite(particle.invMass) ||
				!PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
				!particle.position.isFinite() ||
				!particle.predictedPosition.isFinite() ||
				!particle.outerPosition.isFinite())
			{
				valid = false;
				break;
			}
			const PxU32 localParticleIndex = particleIndex -
				body->compiled.particleStart;
			PxVec3 baseForce;
			PxMat33 baseHessian;
			if(!avbdAssembleParticlePrimalLocalSystem(
					solveContext, *body,
					localParticleIndex,
					contactIndex, baseForce, baseHessian) ||
				!avbdInvertPositiveDefiniteSymmetric33(
					baseHessian, inverseBaseHessian[supportIndex]))
			{
				valid = false;
				break;
			}
			const PxMat33 jacobian(
				geometry.normal * signedScale,
				geometry.tangent1 * signedScale,
				geometry.tangent2 * signedScale);
			responseJacobian[supportIndex] =
				inverseBaseHessian[supportIndex] * jacobian;
			baseDisplacement[supportIndex] =
				inverseBaseHessian[supportIndex] * baseForce;
			if(!responseJacobian[supportIndex].column0.isFinite() ||
				!responseJacobian[supportIndex].column1.isFinite() ||
				!responseJacobian[supportIndex].column2.isFinite() ||
				!baseDisplacement[supportIndex].isFinite())
			{
				valid = false;
				break;
			}
			response += jacobian.getTranspose() *
				responseJacobian[supportIndex];
			baseRowDisplacement += jacobian.getTranspose() *
				baseDisplacement[supportIndex];
			supportBodies[supportIndex] = body;
		}
		if(!valid || !response.column0.isFinite() ||
			!response.column1.isFinite() || !response.column2.isFinite() ||
			!baseRowDisplacement.isFinite())
			continue;

		const PxVec3 surfacePoint = avbdGetSoftContactSurfacePoint(
			geometry, solveContext.particles);
		if(!surfacePoint.isFinite())
			continue;
		const AvbdSoftContactRowForces rowForces =
			avbdEvaluateSoftContactRowForces(
				geometry, state, solveContext.particles, surfacePoint);
		// This is a rotation-quality experiment, not an alternative way to
		// advance an inactive proximity row.  A negative normal row force is
		// the same active unilateral condition consumed by the scalar contact
		// evaluator, and makes the selected row observable in the A/B.
		if(!PxIsFinite(rowForces.normal) || rowForces.normal >= 0.0f)
			continue;
		const PxReal tangentK0 = geometry.friction > 0.0f &&
			rowForces.normal < 0.0f && !rowForces.tangentClamped
				? state.penTangent[0] : 0.0f;
		const PxReal tangentK1 = geometry.friction > 0.0f &&
			rowForces.normal < 0.0f && !rowForces.tangentClamped
				? state.penTangent[1] : 0.0f;
		const PxMat33 penalty = PxMat33::createDiagonal(
			PxVec3(state.k, tangentK0, tangentK1));
		const PxVec3 rowForce(
			rowForces.normal, rowForces.tangent[0], rowForces.tangent[1]);
		const PxMat33 system = PxMat33(PxIdentity) + penalty * response;
		PxVec3 rowSolve;
		if(!rowForce.isFinite() ||
			!avbdSolveGeneral33Checked(
				system, rowForce + penalty * baseRowDisplacement, rowSolve))
			continue;

		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			displacement[supportIndex] = baseDisplacement[supportIndex] -
				responseJacobian[supportIndex] * rowSolve;
			const PxReal displacementSq =
				displacement[supportIndex].magnitudeSquared();
			if(!displacement[supportIndex].isFinite() ||
				!PxIsFinite(displacementSq) || displacementSq > 1.0f)
			{
				valid = false;
				break;
			}
		}
		if(!valid)
			continue;

		// The support moves form one coupled block.  A per-vertex limiter would
		// destroy its Schur relation, so use one shared feasibility fraction for
		// both OGC and positive-J.  This is the multi-vertex analogue of the
		// scalar limiter: every accepted endpoint preserves the same row update.
		PxReal commonAlpha = 1.0f;
		bool acceptedCandidate = false;
		for(PxU32 attempt = 0; attempt < 8 && !acceptedCandidate;
			++attempt)
		{
			bool candidateValid = true;
			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const PxU32 particleIndex = supportIndices[supportIndex];
				const AvbdSoftParticle& particle =
					solveContext.particles[particleIndex];
				const PxReal safetyBound =
					solveContext.selfCollisionSafetyBounds[particleIndex];
				const PxVec3 candidatePosition = particle.position +
					displacement[supportIndex] * commonAlpha;
				if(!candidatePosition.isFinite() || !PxIsFinite(safetyBound) ||
					(safetyBound < 1.0e20f &&
						(candidatePosition - particle.outerPosition).magnitude() >
							safetyBound + 1.0e-6f))
				{
					candidateValid = false;
					break;
				}
			}
			if(!candidateValid)
			{
				commonAlpha *= 0.5f;
				continue;
			}

			auto candidatePositionFor = [&supportIndices, &displacement,
				&solveContext, supportCount, commonAlpha](
					PxU32 particleIndex) -> PxVec3
			{
				for(PxU32 i = 0; i < supportCount; ++i)
					if(supportIndices[i] == particleIndex)
						return solveContext.particles[particleIndex].position +
							displacement[i] * commonAlpha;
				return solveContext.particles[particleIndex].position;
			};
			for(PxU32 supportIndex = 0; supportIndex < supportCount &&
				candidateValid; ++supportIndex)
			{
				const AvbdSoftBody& body = *supportBodies[supportIndex];
				const PxU32 particleIndex = supportIndices[supportIndex];
				const PxU32 localIndex = particleIndex -
					body.compiled.particleStart;
				const AvbdParticleElementAdjacency& adjacency =
					body.compiled.elementAdjacency[localIndex];
				for(PxU32 refIndex = 0; refIndex < adjacency.tetRefs.size();
					++refIndex)
				{
					const AvbdTetElement& tet = body.compiled.tetElements[
						adjacency.tetRefs[refIndex].index];
					const PxVec3 p0 = candidatePositionFor(tet.p0);
					const PxVec3 e1 = candidatePositionFor(tet.p1) - p0;
					const PxVec3 e2 = candidatePositionFor(tet.p2) - p0;
					const PxVec3 e3 = candidatePositionFor(tet.p3) - p0;
					PxReal determinant;
					PxVec3 unusedGradient;
					avbdEvaluateTetDeterminantAndGradient(
						tet, 0, e1, e2, e3, determinant, unusedGradient);
					if(!PxIsFinite(determinant) || determinant < 0.05f)
					{
						candidateValid = false;
						break;
					}
				}
			}
			if(candidateValid)
				acceptedCandidate = true;
			else
				commonAlpha *= 0.5f;
		}
		if(!acceptedCandidate)
			continue;

		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			const PxVec3 appliedDisplacement =
				displacement[supportIndex] * commonAlpha;
			solveContext.particles[particleIndex].position += appliedDisplacement;
			workspace.softSoftPositionRowBlockOwnedParticles[particleIndex] = 1;
			observation.sweepObservation.observe(
				displacement[supportIndex], commonAlpha < 1.0f,
				AvbdSoftTetDisplacementLimitResult(
					appliedDisplacement, commonAlpha,
					commonAlpha < 1.0f
						? AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_LIMITED
						: AvbdSoftTetDisplacementLimitReason::eNONE));
		}
		return true;
	}
	return false;
}

// Keep the default particle solve unchanged.  Only an admitted row-block
// sweep calls this cold serial traversal, which skips the support particles
// already solved as one coupled local system.
PX_NOINLINE inline void avbdSolveParticlePrimalExcludingOwnedParticles(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxU8* ownedParticles,
	AvbdParticlePrimalRangeObservation& observation)
{
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 localIndex = 0; localIndex < body.compiled.particleCount;
			++localIndex)
		{
			const PxU32 particleIndex = body.compiled.particleStart + localIndex;
			if(ownedParticles[particleIndex])
				continue;
			if(solveContext.canUseTetMaterialPackets(body, localIndex))
				solveContext.solveWithTetMaterialPackets(
					body, localIndex, observation);
			else
				solveContext.solve(body, localIndex, observation);
		}
	}
}
#endif

// P3 stage-boundary contract.  ePREPARE performs every operation that must
// precede prediction (including initial-contact state setup); eRESUME consumes
// particles whose predictedPosition/elasticK have already been updated and
// immediately performs the existing predicted-position OGC redetection.
// Split execution requires caller-owned persistent workspace and one stable
// AvbdSoftBodyStepStats instance across both calls.
void avbdPredictSoftBodyParticles(
	AvbdSoftParticle* particles, PxU32 numParticles,
	PxReal dt, const PxVec3& gravity, bool useAdaptiveInitialGuess)
{
	if(!avbdUseSoftElasticProximal())
	{
		if(useAdaptiveInitialGuess)
		{
			for(PxU32 i = 0; i < numParticles; i++)
			{
				particles[i].computePredictionWithAdaptiveInitialGuess(
					dt, gravity);
				// A disabled proximal must not retain warm-start state from a
				// preceding legacy run or timestep.
				particles[i].elasticK = 0.0f;
			}
		}
		else
		{
			for(PxU32 i = 0; i < numParticles; i++)
			{
				particles[i].computePrediction(dt, gravity);
				// A disabled proximal must not retain warm-start state from a
				// preceding legacy run or timestep.
				particles[i].elasticK = 0.0f;
			}
		}
		return;
	}
	if(useAdaptiveInitialGuess)
	{
		for(PxU32 i = 0; i < numParticles; i++)
		{
			particles[i].computePredictionWithAdaptiveInitialGuess(dt, gravity);
			// Reset elastic proximal weight for new timestep
			// (warmstart: retain a fraction from prior timestep for stability)
			particles[i].elasticK = particles[i].elasticK * 0.5f;
		}
	}
	else
	{
		for(PxU32 i = 0; i < numParticles; i++)
		{
			particles[i].computePrediction(dt, gravity);
			// Reset elastic proximal weight for new timestep
			// (warmstart: retain a fraction from prior timestep for stability)
			particles[i].elasticK = particles[i].elasticK * 0.5f;
		}
	}
}

} // namespace Dy
} // namespace physx
