// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions are met.

#ifndef DY_AVBD_CONTACT_TRIANGLE_SURFACE_DIAGNOSTICS_H
#define DY_AVBD_CONTACT_TRIANGLE_SURFACE_DIAGNOSTICS_H

#include "avbd/contact/DyAvbdContactRigidPrimitives.h"

namespace physx
{
namespace Dy
{

struct AvbdRigidTriangleSurfaceFeaturePlanRangeTiming
{
	PxU64 sweptEdgeNanos;
	PxU64 sweptTriangleNanos;
	PxU64 discreteEdgeNanos;
	PxU64 discreteTriangleNanos;

	AvbdRigidTriangleSurfaceFeaturePlanRangeTiming()
		: sweptEdgeNanos(0), sweptTriangleNanos(0),
		  discreteEdgeNanos(0), discreteTriangleNanos(0)
	{
	}

	void record(const AvbdRigidTriangleSurfaceFeatureWorkItem& workItem,
		PxU64 elapsedNanos)
	{
		if(workItem.phase == AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT)
		{
			if(workItem.family ==
				AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE)
				sweptEdgeNanos += elapsedNanos;
			else
				sweptTriangleNanos += elapsedNanos;
		}
		else if(workItem.family ==
			AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE)
			discreteEdgeNanos += elapsedNanos;
		else
			discreteTriangleNanos += elapsedNanos;
	}
};

struct AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming
{
	PxU64 sweptEdgeForwardOwnerNanos;
	PxU64 sweptEdgeBvhRecoveryNanos;
	PxU64 sweptEdgeNarrowPhaseNanos;
	PxU64 sweptTriangleForwardOwnerNanos;
	PxU64 sweptTriangleBvhRecoveryNanos;
	PxU64 sweptTriangleNarrowPhaseNanos;

	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming()
		: sweptEdgeForwardOwnerNanos(0), sweptEdgeBvhRecoveryNanos(0),
		  sweptEdgeNarrowPhaseNanos(0), sweptTriangleForwardOwnerNanos(0),
		  sweptTriangleBvhRecoveryNanos(0), sweptTriangleNarrowPhaseNanos(0)
	{
	}
};

struct AvbdRigidTriangleSurfaceForwardOwnerQueryStats
{
	PxArray<PxU32>* stamps;
	PxU32 numParticles;
	PxU32 numSurfaces;
	PxU32 stamp;
	PxU64 queryCalls;
	PxU64 uniqueQueries;

	AvbdRigidTriangleSurfaceForwardOwnerQueryStats()
		: stamps(NULL), numParticles(0), numSurfaces(0), stamp(0),
		  queryCalls(0), uniqueQueries(0)
	{
	}

	void configure(PxArray<PxU32>& inputStamps, PxU32 inputNumParticles,
		PxU32 inputNumSurfaces, PxU32 inputStamp)
	{
		stamps = &inputStamps;
		numParticles = inputNumParticles;
		numSurfaces = inputNumSurfaces;
		stamp = inputStamp;
		queryCalls = 0;
		uniqueQueries = 0;
		PX_ASSERT(stamp > 0 &&
			PxU64(numParticles) * numSurfaces <= inputStamps.size());
	}

	PX_FORCE_INLINE void record(PxU32 surfaceIndex, PxU32 particleIndex)
	{
		++queryCalls;
		if(!stamps || surfaceIndex >= numSurfaces ||
			particleIndex >= numParticles)
			return;
		const PxU64 index64 = PxU64(surfaceIndex) * numParticles +
			particleIndex;
		PX_ASSERT(index64 < stamps->size());
		if(index64 >= stamps->size())
			return;
		PxU32& entry = (*stamps)[PxU32(index64)];
		if(entry != stamp)
		{
			entry = stamp;
			++uniqueQueries;
		}
	}
};

struct AvbdRigidTriangleSurfaceDiscreteOGCQueryStats
{
	PxU64 edgeBvhQueries;
	PxU64 edgeBvhTriangleCandidates;
	PxU64 edgeFeatureCandidates;
	PxU64 edgeFallbackQueries;
	PxU64 triangleBvhQueries;
	PxU64 triangleBvhTriangleCandidates;
	PxU64 triangleFeatureCandidates;
	PxU64 triangleFallbackQueries;

	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats()
		: edgeBvhQueries(0), edgeBvhTriangleCandidates(0),
		  edgeFeatureCandidates(0), edgeFallbackQueries(0),
		  triangleBvhQueries(0), triangleBvhTriangleCandidates(0),
		  triangleFeatureCandidates(0), triangleFallbackQueries(0)
	{
	}

	PX_FORCE_INLINE void recordEdgeQuery(bool usedBvh,
		PxU32 triangleCandidates, PxU32 featureCandidates)
	{
		if(usedBvh)
		{
			++edgeBvhQueries;
			edgeBvhTriangleCandidates += triangleCandidates;
		}
		else
			++edgeFallbackQueries;
		edgeFeatureCandidates += featureCandidates;
	}

	PX_FORCE_INLINE void recordTriangleQuery(bool usedBvh,
		PxU32 triangleCandidates, PxU32 featureCandidates)
	{
		if(usedBvh)
		{
			++triangleBvhQueries;
			triangleBvhTriangleCandidates += triangleCandidates;
		}
		else
			++triangleFallbackQueries;
		triangleFeatureCandidates += featureCandidates;
	}
};

struct AvbdRigidTriangleSurfaceForwardOwnerResultCache
{
	PxArray<PxU32>* entries;
	const PxArray<PxU32>* surfaceSlots;
	PxU32 numParticles;
	PxU32 numSurfaces;
	PxU32 numCachedSurfaces;
	PxU32 stamp;
	PxU64 hits;
	PxU64 misses;

	AvbdRigidTriangleSurfaceForwardOwnerResultCache()
		: entries(NULL), surfaceSlots(NULL), numParticles(0),
		  numSurfaces(0), numCachedSurfaces(0), stamp(0), hits(0), misses(0)
	{
	}

	void configure(PxArray<PxU32>& inputEntries,
		const PxArray<PxU32>& inputSurfaceSlots, PxU32 inputNumParticles,
		PxU32 inputNumSurfaces, PxU32 inputNumCachedSurfaces,
		PxU32 inputStamp)
	{
		entries = &inputEntries;
		surfaceSlots = &inputSurfaceSlots;
		numParticles = inputNumParticles;
		numSurfaces = inputNumSurfaces;
		numCachedSurfaces = inputNumCachedSurfaces;
		stamp = inputStamp;
		hits = 0;
		misses = 0;
		PX_ASSERT(stamp > 0 && inputSurfaceSlots.size() >= numSurfaces &&
			PxU64(numParticles) * numCachedSurfaces <= inputEntries.size());
	}

	PX_FORCE_INLINE PxU32 getSurfaceSlot(PxU32 surfaceIndex) const
	{
		if(!surfaceSlots || surfaceIndex >= numSurfaces)
			return PX_MAX_U32;
		return (*surfaceSlots)[surfaceIndex];
	}

	PX_FORCE_INLINE bool lookup(PxU32 surfaceSlot, PxU32 particleIndex,
		bool& result)
	{
		if(!entries || surfaceSlot >= numCachedSurfaces ||
			particleIndex >= numParticles)
			return false;
		const PxU64 index64 = PxU64(surfaceSlot) * numParticles +
			particleIndex;
		PX_ASSERT(index64 < entries->size());
		if(index64 >= entries->size())
			return false;
		const PxU32 index = PxU32(index64);
		const PxU32 entry = (*entries)[index];
		if((entry >> 1) != stamp)
			return false;
		result = (entry & 1u) != 0;
		++hits;
		return true;
	}

	PX_FORCE_INLINE void store(PxU32 surfaceSlot, PxU32 particleIndex,
		bool result)
	{
		if(!entries || surfaceSlot >= numCachedSurfaces ||
			particleIndex >= numParticles)
			return;
		const PxU64 index64 = PxU64(surfaceSlot) * numParticles +
			particleIndex;
		PX_ASSERT(index64 < entries->size());
		if(index64 >= entries->size())
			return;
		const PxU32 index = PxU32(index64);
		(*entries)[index] = (stamp << 1) | (result ? 1u : 0u);
		++misses;
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_TRIANGLE_SURFACE_DIAGNOSTICS_H
