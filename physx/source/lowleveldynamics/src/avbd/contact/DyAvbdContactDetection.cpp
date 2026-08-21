// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/contact/DyAvbdContactDetection.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"

namespace physx
{
namespace Dy
{

// CPU AVBD contact orchestration for one current/swept detection epoch.
// Geometry-specific leaves live in the component's detector families; this
// file owns their stable ordering, workspace transaction, and state transfer.

void avbdDetectAllOGCContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidBox* rigidBoxes, PxU32 numRigidBoxes,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	PxReal groundY,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace* persistentWorkspace,
	const AvbdWorldPlane* worldPlanes,
	PxU32 numWorldPlanes,
	bool includeLegacyGround,
	const PxU8* selfCollisionEnabled,
	const AvbdRigidSphere* rigidSpheres,
	PxU32 numRigidSpheres,
	const AvbdRigidCapsule* rigidCapsules,
	PxU32 numRigidCapsules,
	const AvbdRigidConvex* rigidConvexes,
	PxU32 numRigidConvexes,
	const AvbdRigidTriangleSurface* rigidTriangleSurfaces,
	PxU32 numRigidTriangleSurfaces,
	AvbdOgcGeometryEpochSidecar* geometrySidecar)
{
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	PxArray<AvbdSoftContact>& previousContacts =
		workspace.epoch.previousContacts;
	workspace.copyPreviousContacts(contacts);
	contacts.clear();
	if(geometrySidecar)
		geometrySidecar->beginEpoch();
	const PxU32 outputCapacityBefore = contacts.capacity();
	if(stats)
		stats->detectionCalls++;

	const PxU32 groundStart = contacts.size();
	if(numWorldPlanes > 0 && worldPlanes)
	{
		avbdDetectSoftWorldPlaneContacts(
			particles, numParticles, worldPlanes, numWorldPlanes,
			contacts, params.contactRadius, softBodies, numSoftBodies);
	}
	else if(includeLegacyGround)
	{
		avbdDetectSoftGroundContacts(
			particles, numParticles, contacts, groundY,
			params.contactRadius, params.friction,
			softBodies, numSoftBodies);
	}
	if(stats)
		stats->generatedGroundContacts += contacts.size() - groundStart;

	if(numRigidBoxes > 0)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleBoxTests += PxU64(numParticles) * numRigidBoxes;
		avbdDetectSoftRigidSDF(
			particles, numParticles, rigidBoxes, numRigidBoxes,
			contacts, params.contactRadius, previousContacts.begin(),
			previousContacts.size(), softBodies, numSoftBodies);
		avbdDetectSoftRigidSweptSDF(
			particles, numParticles, rigidBoxes, numRigidBoxes,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidOGCFeatures(
			particles, numParticles, rigidBoxes, numRigidBoxes,
			softBodies, numSoftBodies, contacts, params.contactRadius,
			geometrySidecar);
		if(stats)
			stats->generatedRigidContacts += contacts.size() - rigidStart;
	}
	if(numRigidSpheres > 0 && rigidSpheres)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleSphereTests +=
				PxU64(numParticles) * numRigidSpheres;
		avbdDetectSoftRigidSphereSDF(
			particles, numParticles, rigidSpheres, numRigidSpheres,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidSphereSweptSDF(
			particles, numParticles, rigidSpheres, numRigidSpheres,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidSphereSweptOGCFeatures(
			particles, numParticles, rigidSpheres, numRigidSpheres,
			softBodies, numSoftBodies, contacts, params.contactRadius,
			&workspace.rigidConvexForwardOwnerScratch);
		avbdDetectSoftRigidSphereOGCFeatures(
			particles, numParticles, rigidSpheres, numRigidSpheres,
			softBodies, numSoftBodies, contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts += contacts.size() - rigidStart;
	}
	if(numRigidCapsules > 0 && rigidCapsules)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleCapsuleTests +=
				PxU64(numParticles) * numRigidCapsules;
		avbdDetectSoftRigidCapsuleSDF(
			particles, numParticles, rigidCapsules, numRigidCapsules,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidCapsuleSweptSDF(
			particles, numParticles, rigidCapsules, numRigidCapsules,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidCapsuleSweptOGCFeatures(
			particles, numParticles, rigidCapsules, numRigidCapsules,
			softBodies, numSoftBodies, contacts, params.contactRadius,
			&workspace.rigidConvexForwardOwnerScratch);
		avbdDetectSoftRigidCapsuleOGCFeatures(
			particles, numParticles, rigidCapsules, numRigidCapsules,
			softBodies, numSoftBodies, contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts += contacts.size() - rigidStart;
	}
	if(numRigidConvexes > 0 && rigidConvexes)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleConvexTests +=
				PxU64(numParticles) * numRigidConvexes;
		avbdDetectSoftRigidConvexSDF(
			particles, numParticles, rigidConvexes, numRigidConvexes,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidConvexSweptSDF(
			particles, numParticles, rigidConvexes, numRigidConvexes,
			contacts, params.contactRadius, softBodies, numSoftBodies);
		avbdDetectSoftRigidConvexSweptOGCFeatures(
			particles, numParticles, rigidConvexes, numRigidConvexes,
			softBodies, numSoftBodies, contacts, params.contactRadius,
			&workspace.rigidConvexForwardOwnerScratch);
		avbdDetectSoftRigidConvexOGCFeatures(
			particles, numParticles, rigidConvexes, numRigidConvexes,
			softBodies, numSoftBodies, contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts += contacts.size() - rigidStart;
	}
	if(numRigidTriangleSurfaces > 0 && rigidTriangleSurfaces)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleTriangleSurfaceTests +=
				PxU64(numParticles) * numRigidTriangleSurfaces;
		avbdDetectSoftRigidTriangleSurface(
			particles, numParticles, rigidTriangleSurfaces,
			numRigidTriangleSurfaces, contacts, params.contactRadius,
			softBodies, numSoftBodies, stats);
		avbdDetectSoftRigidTriangleSurfaceSwept(
			particles, numParticles, rigidTriangleSurfaces,
			numRigidTriangleSurfaces, contacts, params.contactRadius,
			softBodies, numSoftBodies, stats);
		avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
			particles, numParticles, rigidTriangleSurfaces,
			numRigidTriangleSurfaces, softBodies, numSoftBodies,
			contacts, params.contactRadius, stats, NULL,
			&workspace.rigidTriangleSurfaceForwardOwnerScratch);
		avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
			particles, numParticles, rigidTriangleSurfaces,
			numRigidTriangleSurfaces, softBodies, numSoftBodies,
			contacts, params.contactRadius, stats);
		if(stats)
			stats->generatedRigidContacts += contacts.size() - rigidStart;
	}

	if(numSoftBodies > 1)
	{
		const PxU32 softStart = contacts.size();
		avbdDetectSoftSoftOGC(
			particles, numParticles, softBodies, numSoftBodies,
			contacts, params, stats, &workspace);
		if(stats)
			stats->generatedSoftContacts += contacts.size() - softStart;
	}
	for(PxU32 si = 0; si < numSoftBodies; si++)
	{
		if(si < numAdj && perBodyAdj &&
			(!selfCollisionEnabled || selfCollisionEnabled[si]))
		{
			const PxU32 selfStart = contacts.size();
			avbdDetectSelfCollisionOGC(
				particles, softBodies[si], si, perBodyAdj[si], contacts,
				params, stats, &workspace);
			if(stats)
				stats->generatedSelfContacts += contacts.size() - selfStart;
		}
	}

	workspace.recordOutputCapacityGrowth(
		outputCapacityBefore, contacts.capacity());
	workspace.recordOutputWatermark(contacts.size(), contacts.capacity());
	avbdTransferSoftContactState(
		previousContacts.begin(), previousContacts.size(), particles,
		contacts, &workspace);
	if(geometrySidecar)
		geometrySidecar->resizeContactMapping(contacts.size());
	workspace.invalidateSoftBodyBounds();
}

void avbdDetectAllOGCContacts(
	const AvbdSoftContactDetectionView& view,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	PxReal groundY,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace* persistentWorkspace,
	AvbdOgcGeometryEpochSidecar* geometrySidecar)
{
	avbdDetectAllOGCContacts(
		view.particles, view.numParticles,
		view.softBodies, view.numSoftBodies,
		view.rigidBoxes, view.numRigidBoxes,
		view.selfCollisionAdjacencies,
		view.numSelfCollisionAdjacencies,
		contacts, params, groundY, stats, persistentWorkspace,
		view.worldPlanes, view.numWorldPlanes,
		view.includeLegacyGround, view.selfCollisionEnabled,
		view.rigidSpheres, view.numRigidSpheres,
		view.rigidCapsules, view.numRigidCapsules,
		view.rigidConvexes, view.numRigidConvexes,
		view.rigidTriangleSurfaces, view.numRigidTriangleSurfaces,
		geometrySidecar);
}

bool avbdDetectCurrentPoseOGCContacts(
	const AvbdSoftContactDetectionView& view,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace* persistentWorkspace,
	AvbdOgcGeometryEpochSidecar* geometrySidecar)
{
	if(!view.particles || view.numParticles == 0 || !view.softBodies ||
		view.numSoftBodies == 0 ||
		(view.numWorldPlanes > 0 && !view.worldPlanes) ||
		(view.numRigidBoxes > 0 && !view.rigidBoxes) ||
		(view.numRigidSpheres > 0 && !view.rigidSpheres) ||
		(view.numRigidCapsules > 0 && !view.rigidCapsules) ||
		(view.numRigidConvexes > 0 && !view.rigidConvexes) ||
		(view.numRigidTriangleSurfaces > 0 &&
		 !view.rigidTriangleSurfaces))
		return false;

	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace = persistentWorkspace
		? *persistentWorkspace : localWorkspace;
	contacts.clear();
	if(geometrySidecar)
		geometrySidecar->beginEpoch();
	if(stats)
		++stats->detectionCalls;

	if(view.numWorldPlanes > 0)
		avbdDetectSoftWorldPlaneContacts(
			view.particles, view.numParticles, view.worldPlanes,
			view.numWorldPlanes, contacts, params.contactRadius,
			view.softBodies, view.numSoftBodies);
	else if(view.includeLegacyGround)
		avbdDetectSoftGroundContacts(
			view.particles, view.numParticles, contacts, 0.0f,
			params.contactRadius, params.friction, view.softBodies,
			view.numSoftBodies);

	if(view.numRigidBoxes > 0)
	{
		avbdDetectSoftRigidSDF(
			view.particles, view.numParticles, view.rigidBoxes,
			view.numRigidBoxes, contacts, params.contactRadius, NULL, 0,
			view.softBodies, view.numSoftBodies);
		avbdDetectSoftRigidOGCFeatures(
			view.particles, view.numParticles, view.rigidBoxes,
			view.numRigidBoxes, view.softBodies, view.numSoftBodies,
			contacts, params.contactRadius, geometrySidecar);
	}
	if(view.numRigidSpheres > 0)
	{
		avbdDetectSoftRigidSphereSDF(
			view.particles, view.numParticles, view.rigidSpheres,
			view.numRigidSpheres, contacts, params.contactRadius,
			view.softBodies, view.numSoftBodies);
		avbdDetectSoftRigidSphereOGCFeatures(
			view.particles, view.numParticles, view.rigidSpheres,
			view.numRigidSpheres, view.softBodies, view.numSoftBodies,
			contacts, params.contactRadius);
	}
	if(view.numRigidCapsules > 0)
	{
		avbdDetectSoftRigidCapsuleSDF(
			view.particles, view.numParticles, view.rigidCapsules,
			view.numRigidCapsules, contacts, params.contactRadius,
			view.softBodies, view.numSoftBodies);
		avbdDetectSoftRigidCapsuleOGCFeatures(
			view.particles, view.numParticles, view.rigidCapsules,
			view.numRigidCapsules, view.softBodies, view.numSoftBodies,
			contacts, params.contactRadius);
	}
	if(view.numRigidConvexes > 0)
	{
		avbdDetectSoftRigidConvexSDF(
			view.particles, view.numParticles, view.rigidConvexes,
			view.numRigidConvexes, contacts, params.contactRadius,
			view.softBodies, view.numSoftBodies);
		avbdDetectSoftRigidConvexOGCFeatures(
			view.particles, view.numParticles, view.rigidConvexes,
			view.numRigidConvexes, view.softBodies, view.numSoftBodies,
			contacts, params.contactRadius);
	}
	if(view.numRigidTriangleSurfaces > 0)
	{
		avbdDetectSoftRigidTriangleSurface(
			view.particles, view.numParticles, view.rigidTriangleSurfaces,
			view.numRigidTriangleSurfaces, contacts, params.contactRadius,
			view.softBodies, view.numSoftBodies, stats);
		avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
			view.particles, view.numParticles, view.rigidTriangleSurfaces,
			view.numRigidTriangleSurfaces, view.softBodies,
			view.numSoftBodies, contacts, params.contactRadius, stats);
	}
	if(view.includeSoftTargets && view.numSoftBodies > 1)
		avbdDetectSoftSoftOGC(
			view.particles, view.numParticles, view.softBodies,
			view.numSoftBodies, contacts, params, stats, &workspace);

	if(geometrySidecar &&
		!geometrySidecar->resizeContactMapping(contacts.size()))
	{
		contacts.clear();
		geometrySidecar->clear();
		return false;
	}
	return true;
}

} // namespace Dy
} // namespace physx
