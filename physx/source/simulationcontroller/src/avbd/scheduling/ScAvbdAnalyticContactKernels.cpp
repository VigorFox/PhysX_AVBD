// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdAnalyticContactKernels.h"

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "foundation/PxTime.h"

namespace physx
{
namespace Sc
{

void executeAvbdWorldPlaneContactRange(
	const AvbdWorldPlaneContactRange& range)
{
	PX_ASSERT(range.particles && range.particleBegin < range.particleEnd);
	PX_ASSERT(range.particleEnd <= range.numParticles);
	PX_ASSERT(range.planes && range.numPlanes && range.bodies &&
		range.numBodies && range.contacts);
	Dy::avbdDetectSoftWorldPlaneContactsRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.planes, range.numPlanes, *range.contacts, range.margin,
		range.bodies, range.numBodies);
}

void executeAvbdRigidBoxSdfContactRange(
	const AvbdRigidBoxSdfContactRange& range)
{
	PX_ASSERT(range.particles && range.particleBegin < range.particleEnd);
	PX_ASSERT(range.particleEnd <= range.numParticles);
	PX_ASSERT(range.boxes && range.numBoxes && range.bodies &&
		range.numBodies && range.contacts && range.sweptContacts);
	Dy::avbdDetectSoftRigidSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.boxes, range.numBoxes, *range.contacts, range.margin,
		range.previousContacts, range.numPreviousContacts,
		range.bodies, range.numBodies);
	Dy::avbdDetectSoftRigidSweptSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.boxes, range.numBoxes, *range.sweptContacts, range.margin,
		range.bodies, range.numBodies);
}

void executeAvbdRigidSphereSdfContactRange(
	const AvbdRigidSphereSdfContactRange& range)
{
	PX_ASSERT(range.particles && range.particleBegin < range.particleEnd);
	PX_ASSERT(range.particleEnd <= range.numParticles);
	PX_ASSERT(range.spheres && range.numSpheres && range.bodies &&
		range.numBodies && range.contacts && range.sweptContacts);
	Dy::avbdDetectSoftRigidSphereSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.spheres, range.numSpheres, *range.contacts, range.margin,
		range.bodies, range.numBodies);
	Dy::avbdDetectSoftRigidSphereSweptSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.spheres, range.numSpheres, *range.sweptContacts, range.margin,
		range.bodies, range.numBodies);
}

void executeAvbdRigidCapsuleSdfContactRange(
	const AvbdRigidCapsuleSdfContactRange& range)
{
	PX_ASSERT(range.particles && range.particleBegin < range.particleEnd);
	PX_ASSERT(range.particleEnd <= range.numParticles);
	PX_ASSERT(range.capsules && range.numCapsules && range.bodies &&
		range.numBodies && range.contacts && range.sweptContacts);
	Dy::avbdDetectSoftRigidCapsuleSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.capsules, range.numCapsules, *range.contacts, range.margin,
		range.bodies, range.numBodies);
	Dy::avbdDetectSoftRigidCapsuleSweptSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.capsules, range.numCapsules, *range.sweptContacts, range.margin,
		range.bodies, range.numBodies);
}

void executeAvbdRigidConvexSdfContactRange(
	const AvbdRigidConvexSdfContactRange& range)
{
	PX_ASSERT(range.particles && range.particleBegin < range.particleEnd);
	PX_ASSERT(range.particleEnd <= range.numParticles);
	PX_ASSERT(range.convexes && range.numConvexes && range.bodies &&
		range.numBodies && range.contacts && range.sweptContacts);
	Dy::avbdDetectSoftRigidConvexSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.convexes, range.numConvexes, *range.contacts, range.margin,
		range.bodies, range.numBodies);
	Dy::avbdDetectSoftRigidConvexSweptSDFRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.convexes, range.numConvexes, *range.sweptContacts, range.margin,
		range.bodies, range.numBodies);
}

void executeAvbdRigidTriangleSurfaceContactRange(
	const AvbdRigidTriangleSurfaceContactRange& range,
	AvbdRigidTriangleSurfaceContactRangeTiming* rangeTiming,
	Dy::AvbdRigidTriangleSurfaceFeaturePlanRangeTiming* featurePlanTiming,
	Dy::AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		featureSweptSubstageTiming,
	Dy::AvbdRigidTriangleSurfaceDiscreteOGCQueryStats* discreteQueryStats)
{
	PX_ASSERT(range.particles && range.particleBegin < range.particleEnd);
	PX_ASSERT(range.particleEnd <= range.numParticles);
	PX_ASSERT(range.surfaces && range.numSurfaces && range.bodies &&
		range.numBodies && range.contacts && range.sweptContacts &&
		range.featurePlan && range.featureContacts && range.queryScratch);
	const bool detailedTelemetryEnabled = rangeTiming != NULL;
	const PxU64 currentSdfStartNanos = detailedTelemetryEnabled ?
		PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
	Dy::avbdDetectSoftRigidTriangleSurfaceRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.surfaces, range.numSurfaces, *range.contacts,
		range.queryScratch->triangleBvhQueryCandidates, range.margin,
		range.bodies, range.numBodies, range.collisionStats);
	const PxU64 sweptSdfStartNanos = detailedTelemetryEnabled ?
		PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
	Dy::avbdDetectSoftRigidTriangleSurfaceSweptRange(
		range.particles, range.numParticles,
		range.particleBegin, range.particleEnd,
		range.surfaces, range.numSurfaces, *range.sweptContacts,
		*range.queryScratch, range.margin, range.bodies, range.numBodies,
		range.collisionStats);
	const PxU64 featureStartNanos = detailedTelemetryEnabled ?
		PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
	if(range.featurePlanRowPrivateOutputs)
	{
		PX_ASSERT(range.featurePlanOutputs);
		const PxU32 featurePlanBegin = range.featurePlanRoundRobin ?
			range.featurePlanTaskIndex : range.featurePlanBegin;
		const PxU32 featurePlanEnd = range.featurePlanRoundRobin ?
			range.featurePlan->items.size() : range.featurePlanEnd;
		const PxU32 featurePlanStride = range.featurePlanRoundRobin ?
			range.featurePlanTaskCount : 1u;
		PX_ASSERT(featurePlanStride > 0);
		for(PxU32 planIndex = featurePlanBegin;
			planIndex < featurePlanEnd; planIndex += featurePlanStride)
			Dy::avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
				range.particles, range.numParticles,
				range.surfaces, range.numSurfaces,
				range.bodies, range.numBodies, *range.featurePlan,
				planIndex, planIndex + 1,
				(*range.featurePlanOutputs)[planIndex], *range.queryScratch,
				range.margin, range.collisionStats, featurePlanTiming,
				featureSweptSubstageTiming, range.forwardOwnerQueryStats,
				range.forwardOwnerResultCache, discreteQueryStats,
				range.discreteBodyLocalBoundsCullEnabled);
	}
	else if(range.featurePlanBegin < range.featurePlanEnd)
		Dy::avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
			range.particles, range.numParticles,
			range.surfaces, range.numSurfaces,
			range.bodies, range.numBodies, *range.featurePlan,
			range.featurePlanBegin, range.featurePlanEnd,
			*range.featureContacts, *range.queryScratch, range.margin,
			range.collisionStats, featurePlanTiming,
			featureSweptSubstageTiming, range.forwardOwnerQueryStats,
			range.forwardOwnerResultCache, discreteQueryStats,
			range.discreteBodyLocalBoundsCullEnabled);
	if(rangeTiming)
	{
		const PxU64 featureEndNanos =
			PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u;
		rangeTiming->currentSdfNanos =
			sweptSdfStartNanos - currentSdfStartNanos;
		rangeTiming->sweptSdfNanos =
			featureStartNanos - sweptSdfStartNanos;
		rangeTiming->featureNanos = featureEndNanos - featureStartNanos;
	}
}

void executeAvbdSelfBvhContactRange(
	const AvbdSelfBvhContactRange& range)
{
	PX_ASSERT(range.particles && range.body && range.adjacency &&
		range.parentWorkspace && range.rangeWorkspace && range.contacts &&
		range.params);
	PX_ASSERT(range.vertexBegin < range.vertexEnd ||
		range.edgeBegin < range.edgeEnd);
	Dy::avbdDetectSelfCollisionOGCBvhRange(
		range.particles, *range.body, range.softBodyIndex, *range.adjacency,
		*range.parentWorkspace, *range.rangeWorkspace,
		range.vertexBegin, range.vertexEnd,
		range.edgeBegin, range.edgeEnd, *range.contacts,
		*range.params, range.collisionStats);
}

} // namespace Sc
} // namespace physx
