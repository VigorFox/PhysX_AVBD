// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdOgcPairPlan.h"
#include "avbd/ogc/DyAvbdOgcPairState.h"

namespace physx
{
namespace Sc
{

bool compileAvbdIslandOgcPairPlan(
	AvbdIslandSelectionStorage& storage)
{
	if(storage.ogcGeometrySidecar.contactTriangleCoreIndices.size() !=
			storage.contacts.size())
		return false;
	storage.ogcPairStates.clear();
	storage.ogcPairContactStarts.clear();
	storage.ogcPairContactCounts.clear();
	storage.ogcPairContactRefs.clear();
	if(!Dy::compileOgcPairProviderPlan(
		storage.contacts.begin(), storage.contacts.size(),
		storage.rigidBoxes.begin(), storage.rigidBoxes.size(),
		storage.bodies.size(), storage.bodies.size(),
		Dy::eOGC_PAIR_PROVIDER_WORLD_STATIC |
			Dy::eOGC_PAIR_PROVIDER_DYNAMIC_RIGID |
			Dy::eOGC_PAIR_PROVIDER_DEFORMABLE,
		storage.ogcPairStates, storage.ogcPairIndices))
		return false;
	if(storage.ogcPairStates.empty())
		return true;
	if(storage.ogcPairIndices.size() != storage.contacts.size())
		return false;

	// Compile the inverse once at the provider boundary. Pair-owned OGC stages
	// consume this stable source-order CSR directly instead of rescanning every
	// contact for every active pair.
	storage.ogcPairContactCounts.resize(storage.ogcPairStates.size());
	for(PxU32 pairIndex = 0;
		pairIndex < storage.ogcPairContactCounts.size(); ++pairIndex)
		storage.ogcPairContactCounts[pairIndex] = 0;
	for(PxU32 contactIndex = 0;
		contactIndex < storage.ogcPairIndices.size(); ++contactIndex)
	{
		const PxU32 pairIndex = storage.ogcPairIndices[contactIndex];
		if(pairIndex == PX_MAX_U32)
			continue;
		if(pairIndex >= storage.ogcPairStates.size())
		{
			storage.ogcPairContactCounts.clear();
			return false;
		}
		storage.ogcPairContactCounts[pairIndex]++;
	}
	storage.ogcPairContactStarts.resize(storage.ogcPairStates.size() + 1);
	storage.ogcPairContactStarts[0] = 0;
	for(PxU32 pairIndex = 0;
		pairIndex < storage.ogcPairStates.size(); ++pairIndex)
		storage.ogcPairContactStarts[pairIndex + 1] =
			storage.ogcPairContactStarts[pairIndex] +
			storage.ogcPairContactCounts[pairIndex];
	storage.ogcPairContactRefs.resize(
		storage.ogcPairContactStarts[storage.ogcPairStates.size()]);
	for(PxU32 pairIndex = 0;
		pairIndex < storage.ogcPairContactCounts.size(); ++pairIndex)
		storage.ogcPairContactCounts[pairIndex] = 0;
	for(PxU32 contactIndex = 0;
		contactIndex < storage.ogcPairIndices.size(); ++contactIndex)
	{
		const PxU32 pairIndex = storage.ogcPairIndices[contactIndex];
		if(pairIndex == PX_MAX_U32)
			continue;
		storage.ogcPairContactRefs[
			storage.ogcPairContactStarts[pairIndex] +
			storage.ogcPairContactCounts[pairIndex]++] = contactIndex;
	}
	return true;
}

} // namespace Sc
} // namespace physx
