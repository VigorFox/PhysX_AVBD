// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactVelocityEpoch.h"

namespace physx
{
namespace Dy
{

// AVBD contact-owned velocity objective compilation and terminal refresh.
//
// This boundary resolves component finalization modes from a contact epoch,
// compiles typed velocity objectives, and refreshes the terminal DCD manifold.
// The particle-primal color planner remains in the component orchestration.
// =============================================================================

// P4.5.2b state-machine seam.  Contact redetection may change component
// finalization ownership and the compiled kinematic velocity objectives.  Keep
// that operation independent of avbdStepSoftBodies()'s stack lambdas so a
// future persistent step state invokes exactly the same canonical update at
// its initial and between-outer redetection transitions.
static PxU32 avbdFindSoftComponentBodyIndex(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 particleIndex)
{
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const PxU32 particleStart =
			softBodies[bodyIndex].compiled.particleStart;
		const PxU32 particleCount =
			softBodies[bodyIndex].compiled.particleCount;
		if(particleIndex >= particleStart &&
			particleIndex - particleStart < particleCount)
			return bodyIndex;
	}
	return PX_MAX_U32;
}

static void avbdMergeSoftComponentFinalizeMode(
	PxArray<AvbdSoftComponentFinalizeMode>& componentFinalizeModes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 particleIndex, AvbdSoftComponentFinalizeMode incoming)
{
	const PxU32 bodyIndex = avbdFindSoftComponentBodyIndex(
		softBodies, numSoftBodies, particleIndex);
	if(bodyIndex == PX_MAX_U32)
		return;
	AvbdSoftComponentFinalizeMode& current =
		componentFinalizeModes[bodyIndex];
	if(current == incoming ||
		current == AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
		return;
	if(current == AvbdSoftComponentFinalizeMode::eMOMENTUM)
		current = incoming;
	else
		current = AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
}

void avbdCompileSoftVelocityObjectives(
	PxArray<AvbdCompiledSoftVelocityObjective>& compiledVelocityObjectives,
	PxArray<AvbdSoftComponentFinalizeMode>& componentFinalizeModes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* sourceContacts, PxU32 sourceContactCount)
{
	for(PxU32 sourceIndex = 0;
		sourceIndex < sourceContactCount; sourceIndex++)
	{
		const AvbdSoftContact& source = sourceContacts[sourceIndex];
		const AvbdSoftContactGeometry& geometry = source.geometry;
		AvbdSoftComponentFinalizeMode incoming =
			AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
		if(geometry.velocityOwner ==
			AvbdVelocityObjectiveOwner::PositionAL)
		{
			incoming = geometry.hasKinematicRigidTarget()
				? AvbdSoftComponentFinalizeMode::eUNSUPPORTED
				: AvbdSoftComponentFinalizeMode::ePOSITION_OWNED;
		}
		else if(geometry.velocityOwner ==
			AvbdVelocityObjectiveOwner::ComponentFinalize)
		{
			incoming = geometry.hasKinematicRigidTarget()
				? AvbdSoftComponentFinalizeMode::eKINEMATIC_CONTACT
				: AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
		}
		if(geometry.hasWeightedQueryPoint())
		{
			for(PxU32 pointIndex = 0;
				pointIndex < geometry.queryPoint.count; pointIndex++)
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.queryPoint.particleIndices[pointIndex], incoming);
		}
		else if(geometry.hasBarycentricQueryPoint())
		{
			for(PxU32 vertexIndex = 0; vertexIndex < 3; vertexIndex++)
			{
				if(geometry.queryParticleIndices[vertexIndex] ==
					PX_MAX_U32)
					break;
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.queryParticleIndices[vertexIndex], incoming);
			}
		}
		else
			avbdMergeSoftComponentFinalizeMode(
				componentFinalizeModes, softBodies, numSoftBodies,
				geometry.particleIdx, incoming);
		if(geometry.hasWeightedTargetPoint())
		{
			for(PxU32 pointIndex = 0;
				pointIndex < geometry.targetPoint.count; pointIndex++)
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.targetPoint.particleIndices[pointIndex],
					geometry.velocityOwner ==
						AvbdVelocityObjectiveOwner::PositionAL
						? AvbdSoftComponentFinalizeMode::ePOSITION_OWNED
						: AvbdSoftComponentFinalizeMode::eUNSUPPORTED);
		}
		else if(geometry.hasDeformableSurfaceTarget())
		{
			for(PxU32 vertexIndex = 0; vertexIndex < 3; vertexIndex++)
			{
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.surfaceParticleIndices[vertexIndex],
					geometry.velocityOwner ==
						AvbdVelocityObjectiveOwner::PositionAL
						? AvbdSoftComponentFinalizeMode::ePOSITION_OWNED
						: AvbdSoftComponentFinalizeMode::eUNSUPPORTED);
			}
		}
		if(geometry.velocityOwner !=
				AvbdVelocityObjectiveOwner::ComponentFinalize ||
			!geometry.hasKinematicRigidTarget())
			continue;
		const PxU32 representativeParticle = geometry.hasWeightedQueryPoint()
			? geometry.queryPoint.particleIndices[0]
			: geometry.particleIdx;
		const PxU32 bodyIndex = geometry.queryBodyIndex < numSoftBodies
			? geometry.queryBodyIndex
			: avbdFindSoftComponentBodyIndex(
				softBodies, numSoftBodies, representativeParticle);
		if(bodyIndex == PX_MAX_U32)
			continue;
		AvbdCompiledSoftVelocityObjective objective;
		objective.owner = geometry.velocityOwner;
		objective.source = geometry.source;
		objective.bodyIndex = bodyIndex;
		objective.particleIndex = representativeParticle;
		if(geometry.hasWeightedQueryPoint())
			objective.queryPoint = geometry.queryPoint;
		else if(geometry.hasBarycentricQueryPoint())
		{
			for(PxU32 queryVertex = 0; queryVertex < 3; queryVertex++)
			{
				if(geometry.queryParticleIndices[queryVertex] == PX_MAX_U32)
					break;
				objective.queryPoint.appendMerged(
					geometry.queryParticleIndices[queryVertex],
					geometry.queryWeights[queryVertex]);
			}
		}
		else
			objective.queryPoint.setVertex(geometry.particleIdx);
		objective.normal = geometry.normal;
		objective.surfacePoint = geometry.surfacePoint;
		objective.previousSurfacePoint =
			geometry.kinematicSurfacePointPrevious;
		bool replaced = false;
		for(PxU32 compiledIndex = 0;
			compiledIndex < compiledVelocityObjectives.size();
			compiledIndex++)
		{
			AvbdCompiledSoftVelocityObjective& compiled =
				compiledVelocityObjectives[compiledIndex];
			if(compiled.particleIndex == objective.particleIndex &&
				compiled.source == objective.source)
			{
				compiled = objective;
				replaced = true;
				break;
			}
		}
		if(!replaced)
			compiledVelocityObjectives.pushBack(objective);
	}
}

// A component solve can move the simulation particles after its last outer
// redetection.  A cached contact row is not a valid final-pose OGC manifold:
// it cannot see a newly entered ground/static-box feature and it may carry a
// normal from a pose that the material solve has already left.  Refresh the
// contact stream once at the same simulation time, immediately before the
// terminal current-pose recovery and velocity reconstruction.
//
// This is deliberately a DCD-only epoch.  The callback owns proxy expansion
// and does not advance time or invoke any swept/CCD query.  Replacing the
// velocity objectives is essential: they are compiled from contact geometry,
// so retaining entries from the superseded manifold would let stale normals
// affect the final velocity phase.
bool avbdRefreshComponentTerminalOgcEpoch(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdContactRedetectFn redetectFn,
	PxArray<AvbdSoftContact>* contactsArray,
	void* redetectUserData,
	AvbdSoftContact*& contacts, PxU32& numContacts,
	AvbdSoftBodyWorkspace& workspace)
{
	if(!particles || !softBodies || numParticles == 0 ||
		numSoftBodies == 0 || !redetectFn || !contactsArray)
		return false;

	redetectFn(particles, numParticles, softBodies, numSoftBodies,
		*contactsArray, redetectUserData);
	contacts = contactsArray->begin();
	numContacts = contactsArray->size();
	avbdBuildComponentOgcGeometryEpoch(
		contacts, numContacts, particles, workspace);
	// A terminal current-pose epoch replaces the contact stream.  Re-apply
	// typed tangent ownership before compiling its velocity objectives; otherwise
	// eligible world-static rows silently fall back to a positional sticking
	// spring for the last solve phase.
	avbdAssignVelocityTangentOwners(
		contacts, numContacts, softBodies, numSoftBodies,
		particles, numParticles);
	workspace.compiledVelocityObjectives.clear();
	workspace.resize(workspace.componentFinalizeModes, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		workspace.componentFinalizeModes[bodyIndex] =
			AvbdSoftComponentFinalizeMode::eMOMENTUM;
	avbdCompileSoftVelocityObjectives(
		workspace.compiledVelocityObjectives,
		workspace.componentFinalizeModes, softBodies, numSoftBodies,
		contacts, numContacts);
	return true;
}

} // namespace Dy
} // namespace physx
