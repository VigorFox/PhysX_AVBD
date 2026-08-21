// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

#ifndef DY_AVBD_SOFT_CONTACT_WORKSPACE_H
#define DY_AVBD_SOFT_CONTACT_WORKSPACE_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdContactBounds.h"
#include "avbd/contact/DyAvbdContactWorkspace.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "foundation/PxArray.h"

namespace physx
{
namespace Dy
{

// Caller-owned scratch for contact rebuild, state transfer and OGC sweep
// refit. Keeping this separate from the contact records lets detection reuse
// capacity without making persistent solver state depend on array order.
struct AvbdSoftContactWorkspace
{
	AvbdContactEpochWorkspace epoch;
	PxArray<PxReal> selfTetStressCoefficients;
	// Safety-bound reduction state is rebuilt for every OGC outer epoch.  Keep
	// its scalar minima alongside the reusable self-contact sweep records so
	// the bound calculation does not allocate transient arrays per epoch.
	PxArray<PxReal> selfSafetyTriangleMinimums;
	PxArray<PxReal> selfSafetyEdgeMinimums;
	PxArray<AvbdSelfCollisionTriangleBounds> selfTriangleBounds;
	PxArray<AvbdSelfCollisionVertexSweepEntry> selfSortedVertices;
	PxArray<PxU32> selfActiveTriangles;
	PxArray<PxU64> selfEmittedFeatureKeys;
	PxArray<AvbdSelfCollisionEdgeBounds> selfEdgeBounds;
	PxArray<PxU32> selfEdgeCandidates;
	// P5.9a keeps self-VF candidate ownership separate from soft-pair VF
	// queries.  Both phases remain serial today, but a future soft-pair child
	// must not inherit a mutable buffer that self collision can also consume.
	PxArray<PxU32> selfTriangleCandidates;
	// One byte per particle, reused by the parent-owned analytic sphere/capsule
	// and convex swept-feature suffixes. A surface vertex's forward SDF/sweep
	// ownership depends only on one body/shape/redetection epoch, not on each
	// adjacent edge or face.
	PxArray<PxU8> rigidConvexForwardOwnerScratch;
	// Tri-state (unknown/false/true) scratch for the parent-owned rigid
	// triangle-surface swept-feature suffix.  The predicate is invariant for
	// one body/surface/redetection epoch and is evaluated lazily in canonical
	// edge/face traversal order.
	PxArray<PxU8> rigidTriangleSurfaceForwardOwnerScratch;
	AvbdSoftSoftPairQueryScratch softPairQueryScratch;
	PxArray<AvbdSoftPairBvhEpochSpans> softPairTriangleBvhEpochSpans;
	PxU32 softPairTriangleBvhEpoch;
	PxArray<AvbdSurfaceBvhNodeBounds> selfTriangleBvhBounds;
	PxArray<AvbdSurfaceBvhNodeBounds> selfEdgeBvhBounds;
	PxArray<AvbdSoftPairDetectionPlan> softPairDetectionPlan;
	PxArray<AvbdSoftContactRedetectionPhase> redetectionPhasePlan;
	PxU32 redetectionOutputCapacityBefore;
	PxArray<AvbdSoftBodyBounds> softBodyBounds;
	PxArray<PxU8> softBodyBoundsReady;
	bool softBodyBoundsValid;
	PxU64 growthEvents;
	PxU64 growthBytes;
	PxU64 sweepScratchGrowthEvents;
	PxU64 sweepScratchGrowthBytes;
	PxU64 outputGrowthEvents;
	PxU64 outputGrowthBytes;
	PxU32 peakOutputContactCount;
	PxU32 peakOutputContactCapacity;
	PxU32 peakPreviousContactCount;
	PxU32 peakPreviousContactCapacity;
	PxU32 peakPreviousUsedCapacity;

	AvbdSoftContactWorkspace()
		: softPairTriangleBvhEpoch(0), redetectionOutputCapacityBefore(0),
		  softBodyBoundsValid(false),
		  growthEvents(0), growthBytes(0),
		  sweepScratchGrowthEvents(0), sweepScratchGrowthBytes(0),
		  outputGrowthEvents(0), outputGrowthBytes(0),
		  peakOutputContactCount(0), peakOutputContactCapacity(0),
		  peakPreviousContactCount(0), peakPreviousContactCapacity(0),
		  peakPreviousUsedCapacity(0)
	{
	}

	void reserve(PxU32 contactCapacity)
	{
		epoch.reserve(contactCapacity);
	}

	template<typename T>
	void reserveSweepScratch(PxArray<T>& array, PxU32 capacity)
	{
		if(capacity > array.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(capacity - array.capacity()) * sizeof(T);
			sweepScratchGrowthEvents++;
			sweepScratchGrowthBytes +=
				PxU64(capacity - array.capacity()) * sizeof(T);
			array.reserve(capacity);
		}
	}

	void reserveSelfCollisionSweep(
		PxU32 tetCount, PxU32 triangleCount,
		PxU32 vertexCount, PxU32 edgeCount)
	{
		reserveSweepScratch(selfTetStressCoefficients, tetCount);
		reserveSweepScratch(selfSafetyTriangleMinimums, triangleCount);
		reserveSweepScratch(selfSafetyEdgeMinimums, edgeCount);
		reserveSweepScratch(selfTriangleBounds, triangleCount);
		reserveSweepScratch(selfSortedVertices, vertexCount);
		reserveSweepScratch(selfActiveTriangles, triangleCount);
		reserveSweepScratch(selfEmittedFeatureKeys, triangleCount);
		reserveSweepScratch(selfEdgeBounds, edgeCount);
		reserveSweepScratch(selfEdgeCandidates, edgeCount);
		reserveSweepScratch(selfTriangleCandidates, triangleCount);
	}

	void reserveSoftPairSweep(
		PxU32 edgeCountA, PxU32 edgeCountB,
		PxU32 triangleCandidateCapacity = 0)
	{
		reserveSweepScratch(softPairQueryScratch.edgeBoundsA, edgeCountA);
		reserveSweepScratch(softPairQueryScratch.edgeBoundsB, edgeCountB);
		reserveSweepScratch(
			softPairQueryScratch.triangleCandidates,
			triangleCandidateCapacity);
	}

	// Start a distinct soft-pair detection epoch. Callers first mark the body
	// and mode spans that a canonical pair plan needs, then refit each marked
	// span exactly once. A later pair consumer may only read a span stamped
	// with this epoch.
	void beginSoftPairTriangleBvhEpoch(PxU32 bodyCount)
	{
		reserveSweepScratch(softPairTriangleBvhEpochSpans, bodyCount);
		softPairTriangleBvhEpochSpans.resize(bodyCount);
		softPairTriangleBvhEpoch++;
		if(softPairTriangleBvhEpoch == 0)
		{
			softPairTriangleBvhEpoch = 1;
			for(PxU32 bodyIndex = 0;
				bodyIndex < softPairTriangleBvhEpochSpans.size();
				++bodyIndex)
			{
				AvbdSoftPairBvhEpochSpans& spans =
					softPairTriangleBvhEpochSpans[bodyIndex];
				spans.currentRequiredEpoch = 0;
				spans.sweptRequiredEpoch = 0;
				spans.currentRefitEpoch = 0;
				spans.sweptRefitEpoch = 0;
			}
		}
	}

	void requireSoftPairTriangleBvhBounds(
		PxU32 bodyIndex, bool swept, PxU32 nodeCount)
	{
		PX_ASSERT(bodyIndex < softPairTriangleBvhEpochSpans.size());
		PX_ASSERT(softPairTriangleBvhEpoch != 0);
		AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		PxArray<AvbdSurfaceBvhNodeBounds>& bounds = swept
			? spans.sweptBounds : spans.currentBounds;
		reserveSweepScratch(bounds, nodeCount);
		bounds.resize(nodeCount);
		if(swept)
			spans.sweptRequiredEpoch = softPairTriangleBvhEpoch;
		else
			spans.currentRequiredEpoch = softPairTriangleBvhEpoch;
	}

	bool isSoftPairTriangleBvhBoundsRequired(
		PxU32 bodyIndex, bool swept) const
	{
		PX_ASSERT(bodyIndex < softPairTriangleBvhEpochSpans.size());
		const AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		return swept ?
			spans.sweptRequiredEpoch == softPairTriangleBvhEpoch :
			spans.currentRequiredEpoch == softPairTriangleBvhEpoch;
	}

	PxArray<AvbdSurfaceBvhNodeBounds>& getSoftPairTriangleBvhBoundsForRefit(
		PxU32 bodyIndex, bool swept)
	{
		PX_ASSERT(isSoftPairTriangleBvhBoundsRequired(bodyIndex, swept));
		AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		return swept ? spans.sweptBounds : spans.currentBounds;
	}

	void markSoftPairTriangleBvhBoundsRefit(
		PxU32 bodyIndex, bool swept)
	{
		PX_ASSERT(isSoftPairTriangleBvhBoundsRequired(bodyIndex, swept));
		AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		if(swept)
			spans.sweptRefitEpoch = softPairTriangleBvhEpoch;
		else
			spans.currentRefitEpoch = softPairTriangleBvhEpoch;
	}

	const PxArray<AvbdSurfaceBvhNodeBounds>& getSoftPairTriangleBvhBounds(
		PxU32 bodyIndex, bool swept) const
	{
		PX_ASSERT(bodyIndex < softPairTriangleBvhEpochSpans.size());
		const AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		PX_ASSERT(swept ?
			spans.sweptRefitEpoch == softPairTriangleBvhEpoch :
			spans.currentRefitEpoch == softPairTriangleBvhEpoch);
		return swept ? spans.sweptBounds : spans.currentBounds;
	}

	void prepareSelfBvhBounds(
		PxU32 triangleNodeCount, PxU32 edgeNodeCount)
	{
		reserveSweepScratch(selfTriangleBvhBounds, triangleNodeCount);
		reserveSweepScratch(selfEdgeBvhBounds, edgeNodeCount);
		selfTriangleBvhBounds.resize(triangleNodeCount);
		selfEdgeBvhBounds.resize(edgeNodeCount);
	}

	void beginSoftPairDetectionPlan()
	{
		softPairDetectionPlan.clear();
	}

	void appendSoftPairDetectionPlan(
		const AvbdSoftPairDetectionPlan& plan)
	{
		if(softPairDetectionPlan.size() ==
			softPairDetectionPlan.capacity())
		{
			const PxU32 currentCapacity =
				softPairDetectionPlan.capacity();
			const PxU32 nextCapacity = currentCapacity == 0
				? 8u : currentCapacity <= PX_MAX_U32 / 2
					? currentCapacity * 2u : PX_MAX_U32;
			reserveSweepScratch(softPairDetectionPlan, nextCapacity);
		}
		softPairDetectionPlan.pushBack(plan);
	}

	bool validateSoftPairDetectionPlan(PxU32 bodyCount) const
	{
		PxU32 previousBodyA = 0;
		PxU32 previousBodyB = 0;
		for(PxU32 planIndex = 0;
			planIndex < softPairDetectionPlan.size(); ++planIndex)
		{
			const AvbdSoftPairDetectionPlan& plan =
				softPairDetectionPlan[planIndex];
			if(plan.bodyA >= plan.bodyB || plan.bodyB >= bodyCount)
				return false;
			if(planIndex > 0 &&
				(plan.bodyA < previousBodyA ||
				 (plan.bodyA == previousBodyA &&
				  plan.bodyB <= previousBodyB)))
				return false;
			previousBodyA = plan.bodyA;
			previousBodyB = plan.bodyB;
		}
		return true;
	}

	// P5.2a: phase-plan publication happens before any candidate append.  Its
	// order is the serial OGC order, and the parent retains all mutable arrays.
	void beginRedetectionPhasePlan()
	{
		redetectionPhasePlan.clear();
	}

	void appendRedetectionPhasePlan(
		AvbdSoftContactRedetectionPhase::Type type,
		PxU32 sourceBegin, PxU32 sourceEnd)
	{
		PX_ASSERT(sourceBegin < sourceEnd);
		if(redetectionPhasePlan.size() ==
			redetectionPhasePlan.capacity())
		{
			const PxU32 currentCapacity =
				redetectionPhasePlan.capacity();
			const PxU32 nextCapacity = currentCapacity == 0
				? 8u : currentCapacity <= PX_MAX_U32 / 2
					? currentCapacity * 2u : PX_MAX_U32;
			reserveSweepScratch(redetectionPhasePlan, nextCapacity);
		}
		redetectionPhasePlan.pushBack(
			AvbdSoftContactRedetectionPhase(
				type, sourceBegin, sourceEnd));
	}

	bool validateRedetectionPhasePlan() const
	{
		if(redetectionPhasePlan.empty())
			return true;
		PxU32 previousType = 0;
		for(PxU32 phaseIndex = 0;
			phaseIndex < redetectionPhasePlan.size(); ++phaseIndex)
		{
			const AvbdSoftContactRedetectionPhase& phase =
				redetectionPhasePlan[phaseIndex];
			if(phase.sourceBegin >= phase.sourceEnd ||
				PxU32(phase.type) < previousType)
				return false;
			previousType = PxU32(phase.type);
		}
		return true;
	}

	// This is prepared before prediction tasks are submitted. Child tasks then
	// write distinct body slots only; no task may resize either array.
	void prepareSoftBodyBounds(PxU32 bodyCount)
	{
		reserveSweepScratch(softBodyBounds, bodyCount);
		reserveSweepScratch(softBodyBoundsReady, bodyCount);
		softBodyBounds.resize(bodyCount);
		softBodyBoundsReady.resize(bodyCount);
		for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
			softBodyBoundsReady[bodyIndex] = 0;
		softBodyBoundsValid = false;
	}

	void markSoftBodyBoundsReady()
	{
		softBodyBoundsValid = !softBodyBounds.empty();
		for(PxU32 bodyIndex = 0;
			bodyIndex < softBodyBoundsReady.size(); ++bodyIndex)
		{
			if(!softBodyBoundsReady[bodyIndex])
			{
				softBodyBoundsValid = false;
				break;
			}
		}
	}

	void invalidateSoftBodyBounds()
	{
		softBodyBoundsValid = false;
	}

	void beginStep()
	{
		growthEvents = 0;
		growthBytes = 0;
		sweepScratchGrowthEvents = 0;
		sweepScratchGrowthBytes = 0;
		outputGrowthEvents = 0;
		outputGrowthBytes = 0;
		peakOutputContactCount = 0;
		peakOutputContactCapacity = 0;
		peakPreviousContactCount = 0;
		peakPreviousContactCapacity = 0;
		peakPreviousUsedCapacity = 0;
	}

	void recordOutputCapacityGrowth(
		PxU32 capacityBefore, PxU32 capacityAfter)
	{
		if(capacityAfter > capacityBefore)
		{
			outputGrowthEvents++;
			outputGrowthBytes +=
				PxU64(capacityAfter - capacityBefore) *
				sizeof(AvbdSoftContact);
		}
	}

	void recordOutputWatermark(PxU32 count, PxU32 capacity)
	{
		peakOutputContactCount = PxMax(peakOutputContactCount, count);
		peakOutputContactCapacity = PxMax(peakOutputContactCapacity, capacity);
	}

	void copyPreviousContacts(const PxArray<AvbdSoftContact>& contacts)
	{
		if(contacts.size() > epoch.previousContacts.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(contacts.size() - epoch.previousContacts.capacity()) *
				sizeof(AvbdSoftContact);
		}
		epoch.previousContacts.assign(contacts.begin(), contacts.end());
		peakPreviousContactCount = PxMax(
			peakPreviousContactCount, contacts.size());
		peakPreviousContactCapacity = PxMax(
			peakPreviousContactCapacity, epoch.previousContacts.capacity());
	}

	void resizePreviousUsed(PxU32 size)
	{
		if(size > epoch.previousUsed.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(size - epoch.previousUsed.capacity()) * sizeof(PxU8);
		}
		epoch.previousUsed.resize(size);
		peakPreviousUsedCapacity = PxMax(
			peakPreviousUsedCapacity, epoch.previousUsed.capacity());
	}

	void reset()
	{
		epoch.reset();
		selfTetStressCoefficients.reset();
		selfSafetyTriangleMinimums.reset();
		selfSafetyEdgeMinimums.reset();
		selfTriangleBounds.reset();
		selfSortedVertices.reset();
		selfActiveTriangles.reset();
		selfEmittedFeatureKeys.reset();
		selfEdgeBounds.reset();
		selfEdgeCandidates.reset();
		selfTriangleCandidates.reset();
		rigidConvexForwardOwnerScratch.reset();
		rigidTriangleSurfaceForwardOwnerScratch.reset();
		softPairQueryScratch.reset();
		softPairTriangleBvhEpochSpans.reset();
		softPairTriangleBvhEpoch = 0;
		selfTriangleBvhBounds.reset();
		selfEdgeBvhBounds.reset();
		softPairDetectionPlan.reset();
		redetectionPhasePlan.reset();
		softBodyBounds.reset();
		softBodyBoundsReady.reset();
		softBodyBoundsValid = false;
		beginStep();
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_CONTACT_WORKSPACE_H

