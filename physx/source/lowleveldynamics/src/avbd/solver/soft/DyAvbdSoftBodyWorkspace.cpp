// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

AvbdSoftBodyWorkspace::AvbdSoftBodyWorkspace()
	: particlePrimalColorCount(0),
	  particlePrimalDynamicConflictValid(false),
	  particlePrimalColorPlanValid(false),
	  growthEvents(0), growthBytes(0), peakContactIncidenceCount(0),
	  peakContactIncidenceCapacity(0)
{
}

void AvbdSoftBodyWorkspace::reserve(
	PxU32 numParticles, PxU32 contactCapacity,
	AvbdParticlePrimalSchedule particlePrimalSchedule)
{
	contact.reserve(contactCapacity);
	contact.reserveSweepScratch(
		contact.rigidConvexForwardOwnerScratch, numParticles);
	contact.reserveSweepScratch(
		contact.rigidTriangleSurfaceForwardOwnerScratch, numParticles);
	const PxU32 contactIndexCapacity = contactCapacity <= PX_MAX_U32 / 6
		? contactCapacity * AVBD_CONTACT_MAX_PARTICLES : PX_MAX_U32;
	contactIndices.reserve(contactIndexCapacity);
	contactStarts.reserve(numParticles + 1);
	contactCounts.reserve(numParticles);
	const AvbdParticlePrimalSchedule resolvedParticlePrimalSchedule =
		particlePrimalSchedule == AvbdParticlePrimalSchedule::eDEFAULT
			? avbdGetParticlePrimalSchedule() : particlePrimalSchedule;
	if(avbdValidateParticlePrimalAccessPlan() ||
		avbdUsesColoredParticlePrimalSchedule(
			resolvedParticlePrimalSchedule))
	{
		const PxU64 dynamicGroupCapacity =
			PxU64(contactCapacity) + PxU64(numParticles);
		particlePrimalDynamicAccessGroups.reserve(
			dynamicGroupCapacity > PX_MAX_U32
				? PX_MAX_U32 : PxU32(dynamicGroupCapacity));
		particlePrimalDynamicConflictOffsets.reserve(numParticles + 1);
		particlePrimalDynamicConflictCounts.reserve(numParticles);
		const PxU64 dynamicConflictCapacity =
			PxU64(contactCapacity) *
				PxU64(AVBD_CONTACT_MAX_PARTICLES) *
				PxU64(AVBD_CONTACT_MAX_PARTICLES - 1u) +
			PxU64(numParticles) * 3u * 2u;
		const PxU32 maxDynamicConflictIndices =
			(2u * 1024u * 1024u) / sizeof(PxU32);
		particlePrimalDynamicConflictIndices.reserve(
			dynamicConflictCapacity >= maxDynamicConflictIndices
				? maxDynamicConflictIndices : PxU32(dynamicConflictCapacity));
		particlePrimalBodyIndices.reserve(numParticles);
		particlePrimalColors.reserve(numParticles);
		particlePrimalColorCounts.reserve(numParticles);
		particlePrimalColorOffsets.reserve(numParticles + 1);
		particlePrimalColorParticles.reserve(numParticles);
	}
	chebyPrevPos.reserve(numParticles);
	chebyPrevPrevPos.reserve(numParticles);
	selfCollisionSafetyBounds.reserve(numParticles);
	bodySelfCollisionSafetyBounds.reserve(numParticles);
	componentOgcGeometrySidecar.triangleCoreCertificates.reserve(
		contactCapacity);
	componentOgcGeometrySidecar.contactTriangleCoreIndices.reserve(
		contactCapacity);
	componentOgcPairStates.reserve(contactCapacity);
	componentOgcPairIndices.reserve(contactCapacity);
	componentOgcSafetyBodyMask.reserve(numParticles);
	compiledVelocityObjectives.reserve(contactCapacity);
	componentMomentumTargets.reserve(numParticles);
	componentFinalizeModes.reserve(numParticles);
}

void AvbdSoftBodyWorkspace::beginStep()
{
	growthEvents = 0;
	growthBytes = 0;
	peakContactIncidenceCount = 0;
	peakContactIncidenceCapacity = 0;
	particlePrimalColorCount = 0;
	particlePrimalDynamicConflictValid = false;
	particlePrimalColorPlanValid = false;
	contact.beginStep();
	componentOgcGeometrySidecar.clear();
}

void AvbdSoftBodyWorkspace::reset()
{
	contact.reset();
	contactIndices.reset();
	contactStarts.reset();
	contactCounts.reset();
	particlePrimalDynamicAccessGroups.reset();
	particlePrimalDynamicConflictOffsets.reset();
	particlePrimalDynamicConflictIndices.reset();
	particlePrimalDynamicConflictCounts.reset();
	particlePrimalBodyIndices.reset();
	particlePrimalColors.reset();
	particlePrimalColorCounts.reset();
	particlePrimalColorOffsets.reset();
	particlePrimalColorParticles.reset();
	chebyPrevPos.reset();
	chebyPrevPrevPos.reset();
	selfCollisionSafetyBounds.reset();
	bodySelfCollisionSafetyBounds.reset();
	componentOgcGeometrySidecar.reset();
	componentOgcPairStates.reset();
	componentOgcPairIndices.reset();
	componentOgcSafetyBodyMask.reset();
	compiledVelocityObjectives.reset();
	componentMomentumTargets.reset();
	componentFinalizeModes.reset();
	worldStaticEndpointRecoveredBodies.reset();
	beginStep();
}

AvbdOgcGeometryEpochView
AvbdSoftBodyWorkspace::getComponentOgcGeometryEpochView() const
{
	return makeOgcGeometryEpochView(
		componentOgcPairStates.empty() ? NULL :
			componentOgcPairStates.begin(),
		componentOgcPairStates.size(),
		componentOgcPairIndices.empty() ? NULL :
			componentOgcPairIndices.begin(),
		componentOgcPairIndices.size(), &componentOgcGeometrySidecar);
}

} // namespace Dy
} // namespace physx
