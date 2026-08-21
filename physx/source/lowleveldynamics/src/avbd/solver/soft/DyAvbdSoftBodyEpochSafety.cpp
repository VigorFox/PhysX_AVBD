// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Dy
{

// Component-level OGC epoch admission and trust-region ownership. Numerical
// mechanics and contact geometry remain in their dedicated units.

void avbdSnapshotOuterPositionsScalar(
	AvbdSoftParticle* particles, PxU32 numParticles,
	PxReal* selfCollisionSafetyBounds)
{
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		particles[particleIndex].outerPosition =
			particles[particleIndex].position;
		selfCollisionSafetyBounds[particleIndex] = PX_MAX_F32;
	}
}

bool avbdCanReuseComponentOgcEpoch(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles)
{
	if(!contacts || numContacts == 0 || !softBodies ||
		numSoftBodies == 0 || !particles)
		return false;
	for(PxU32 contactIndex = 0; contactIndex < numContacts; ++contactIndex)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		if(geometry.queryBodyIndex >= numSoftBodies)
			return false;
		const bool softPair =
			geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
			geometry.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
			geometry.targetIndex < numSoftBodies &&
			geometry.queryBodyIndex != geometry.targetIndex;
		const bool worldStatic =
			geometry.targetKind ==
				AvbdSoftContactTargetKind::eWORLD_STATIC &&
			(geometry.source.type == AvbdSoftContactSource::eGROUND ||
			 geometry.source.type == AvbdSoftContactSource::eRIGID_SDF);
		if(!softPair && !worldStatic)
			return false;
		const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
		if(!PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f)
			return false;
		const PxVec3 queryPoint = avbdGetSoftContactQueryPoint(
			geometry, particles);
		const PxVec3 surfacePoint = softPair
			? avbdGetSoftContactSurfacePoint(geometry, particles)
			: geometry.surfacePoint;
		const PxReal physicalGap = (queryPoint - surfacePoint).dot(
			geometry.normal * PxRecipSqrt(normalLengthSq));
		if(!queryPoint.isFinite() || !surfacePoint.isFinite() ||
			!PxIsFinite(physicalGap) || physicalGap < -1.0e-5f)
			return false;
	}
	return true;
}

bool avbdBuildComponentOgcGeometryEpoch(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles, AvbdSoftBodyWorkspace& workspace)
{
	if(workspace.componentOgcGeometrySidecar.contactTriangleCoreIndices.size()
			!= numContacts)
	{
		// Direct low-level component callers may publish an epoch containing no
		// sparse TBIX witnesses. Create only the dense empty mapping here; never
		// reconstruct geometry from contact rows or repair a partial sidecar.
		if(!workspace.componentOgcGeometrySidecar.
				contactTriangleCoreIndices.empty() ||
			!workspace.componentOgcGeometrySidecar.
				triangleCoreCertificates.empty() ||
			!workspace.componentOgcGeometrySidecar.resizeContactMapping(
				numContacts))
			return false;
	}
	const bool providerPairPlan =
		workspace.componentOgcPairIndices.size() == numContacts &&
		(numContacts == 0u ||
		 !workspace.componentOgcPairStates.empty());
	if(providerPairPlan)
	{
		for(PxU32 pairIndex = 0;
			pairIndex < workspace.componentOgcPairStates.size(); ++pairIndex)
		{
			AvbdOgcPairState& pair =
				workspace.componentOgcPairStates[pairIndex];
			pair.beginSolveEpoch();
			pair.geometry.active = true;
		}
	}
	else
	{
		workspace.componentOgcPairStates.clear();
		workspace.componentOgcPairIndices.resize(numContacts);
	}
	for(PxU32 contactIndex = 0; contactIndex < numContacts; ++contactIndex)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		PxU32 pairIndex = providerPairPlan
			? workspace.componentOgcPairIndices[contactIndex]
			: PX_MAX_U32;
		if(!providerPairPlan)
		{
			workspace.componentOgcPairIndices[contactIndex] = PX_MAX_U32;
			for(PxU32 candidateIndex = 0;
				candidateIndex < workspace.componentOgcPairStates.size();
				++candidateIndex)
			{
				const AvbdOgcPairState& candidate =
					workspace.componentOgcPairStates[candidateIndex];
				if(candidate.matches(
					geometry.source.type, geometry.targetKind,
					geometry.queryBodyIndex, geometry.targetIndex,
					geometry.source.primitiveKey))
				{
					pairIndex = candidateIndex;
					break;
				}
			}
			if(pairIndex == PX_MAX_U32)
			{
				pairIndex = workspace.componentOgcPairStates.size();
				AvbdOgcPairState pair;
				pair.initializeKey(
					geometry.source.type, geometry.targetKind,
					geometry.queryBodyIndex, geometry.targetIndex,
					geometry.source.primitiveKey);
				pair.beginGeometryEpoch();
				pair.geometry.active = true;
				workspace.componentOgcPairStates.pushBack(pair);
			}
		}
		if(pairIndex >= workspace.componentOgcPairStates.size())
			return false;

		AvbdOgcPairState& pair = workspace.componentOgcPairStates[pairIndex];
		if(!pair.matches(
				geometry.source.type, geometry.targetKind,
				geometry.queryBodyIndex, geometry.targetIndex,
				geometry.source.primitiveKey))
			return false;
		if(!providerPairPlan)
			pair.addContact();
		const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
		const PxVec3 queryPoint = avbdGetSoftContactQueryPoint(
			geometry, particles);
		const PxVec3 surfacePoint =
			geometry.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE
				? avbdGetSoftContactSurfacePoint(geometry, particles)
				: geometry.surfacePoint;
		if(normalLengthSq > 1.0e-12f && PxIsFinite(normalLengthSq) &&
			queryPoint.isFinite() && surfacePoint.isFinite())
		{
			const PxVec3 normal = geometry.normal *
				PxRecipSqrt(normalLengthSq);
			const PxReal gap = (queryPoint - surfacePoint).dot(normal);
			if(PxIsFinite(gap) &&
				(pair.geometry.representativeContact == PX_MAX_U32 ||
				 gap < pair.geometry.referenceGap))
			{
				pair.geometry.referenceGap = gap;
				pair.trustRegion.safetyGap = gap;
				pair.geometry.minimumGap = gap;
				pair.geometry.representativeContact = contactIndex;
				pair.geometry.representativeNormal = normal;
				pair.geometry.representativeGap = gap;
			}
		}
		workspace.componentOgcPairIndices[contactIndex] = pairIndex;
	}
	return workspace.componentOgcGeometrySidecar.contactTriangleCoreIndices.size()
		== numContacts;
}

bool avbdApplyComponentOgcEpochSafetyBounds(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles,
	PxReal contactRadius, PxReal safetyRelax,
	PxReal* particleSafetyBounds, PxU32 numParticles,
	AvbdSoftBodyWorkspace& workspace)
{
	if(!contacts || numContacts == 0 || !softBodies ||
		numSoftBodies == 0 || !particles || !particleSafetyBounds)
		return false;
	if(!avbdCanReuseComponentOgcEpoch(
		contacts, numContacts, softBodies, numSoftBodies, particles))
		return false;
	workspace.resize(workspace.componentOgcSafetyBodyMask, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		workspace.componentOgcSafetyBodyMask[bodyIndex] = 0u;

	if(!avbdBuildComponentOgcGeometryEpoch(
		contacts, numContacts, particles, workspace))
		return false;
	if(workspace.componentOgcPairStates.empty() ||
		workspace.componentOgcPairIndices.size() != numContacts)
		return false;
	const PxReal maximumSafetyFraction =
		PxClamp(safetyRelax, 1.0e-4f, 0.499f);
	const PxReal maximumSafetyDistance =
		PxMax(contactRadius, 1.0e-6f);
	bool appliedProximitySafetyBound = false;
	const auto applyBodySafetyDistance = [&](PxU32 bodyIndex,
		PxReal safetyDistance)
	{
		if(bodyIndex >= numSoftBodies)
			return false;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.compiled.particleStart > numParticles ||
			body.compiled.particleCount >
				numParticles - body.compiled.particleStart)
			return false;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localIndex;
			particleSafetyBounds[particleIndex] = PxMin(
				particleSafetyBounds[particleIndex], safetyDistance);
		}
		workspace.componentOgcSafetyBodyMask[bodyIndex] = 1u;
		return true;
	};
	for(PxU32 pairIndex = 0;
		pairIndex < workspace.componentOgcPairStates.size(); ++pairIndex)
	{
		AvbdOgcPairState& pair = workspace.componentOgcPairStates[pairIndex];
		if(!pair.geometry.active ||
			pair.key.sourceBodyIndex >= numSoftBodies ||
			!PxIsFinite(pair.trustRegion.safetyGap) ||
			pair.trustRegion.safetyGap < -1.0e-5f)
			return false;
		if(pair.geometry.representativeContact >= numContacts)
			return false;
		const AvbdSoftContactGeometry& representativeGeometry =
			contacts[pair.geometry.representativeContact].geometry;
		const PxVec3 representativeQueryPoint =
			avbdGetSoftContactQueryPoint(
				representativeGeometry, particles);
		const PxVec3 representativeSurfacePoint =
			avbdGetSoftContactSurfacePoint(
				representativeGeometry, particles);
		if(!representativeQueryPoint.isFinite() ||
			!representativeSurfacePoint.isFinite())
			return false;
		const PxReal normalConstraint =
			avbdEvaluateSoftContactNormalConstraint(
				representativeGeometry, representativeQueryPoint,
				representativeSurfacePoint);
		if(!PxIsFinite(normalConstraint))
			return false;

		// Active rows own the normal direction. Applying their zero gap as an
		// isotropic trust region would also freeze tangential deformation.
		const PxReal activeTolerance = PxMax(
			1.0e-5f,
			1.0e-4f * PxMax(representativeGeometry.margin, 0.0f));
		if(normalConstraint <= activeTolerance)
		{
			pair.trustRegion.remainingSafeDisplacement = 0.0f;
			pair.trustRegion.accumulatedRelativeDisplacement = 0.0f;
			pair.trustRegion.refreshRequested = false;
			continue;
		}
		const PxReal safetyDistance = PxMax(1.0e-6f,
			maximumSafetyFraction * PxMin(maximumSafetyDistance,
				normalConstraint));
		pair.trustRegion.remainingSafeDisplacement = safetyDistance;
		pair.trustRegion.accumulatedRelativeDisplacement = 0.0f;
		pair.trustRegion.refreshRequested = false;
		if(!applyBodySafetyDistance(pair.key.sourceBodyIndex, safetyDistance))
			return false;
		if(pair.key.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
			!applyBodySafetyDistance(
				pair.key.targetBodyIndex, safetyDistance))
			return false;
		appliedProximitySafetyBound = true;
	}
	return appliedProximitySafetyBound;
}

} // namespace Dy
} // namespace physx
