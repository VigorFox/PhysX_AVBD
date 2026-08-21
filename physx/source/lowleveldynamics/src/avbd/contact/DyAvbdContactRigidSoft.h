// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_RIGID_SOFT_H
#define DY_AVBD_CONTACT_RIGID_SOFT_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdContactRigidPrimitives.h"
#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/contact/DyAvbdContactTriangleSurfaceDiagnostics.h"
#include "avbd/contact/DyAvbdContactTriangleSurfaceTypes.h"
#include "foundation/PxArray.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftBody;
struct AvbdSoftParticle;
struct AvbdOgcGeometryEpochSidecar;

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_RIGID_SOFT_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_RIGID_SOFT_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_RIGID_SOFT_API
#endif

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftContact* previousContacts = NULL,
	PxU32 numPreviousContacts = 0,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftContact* previousContacts = NULL,
	PxU32 numPreviousContacts = 0,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	AvbdOgcGeometryEpochSidecar* geometrySidecar = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSphereSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSphereSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidSphereSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSphereSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidSphereSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidSphereOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidCapsuleSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidCapsuleSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidCapsuleSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidCapsuleSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidCapsuleSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidCapsuleOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidConvexSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidConvexSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidConvexSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidConvexSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidConvexSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidConvexOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdBuildRigidTriangleSurfaceOGCFeaturePlan(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 numSurfaces, AvbdRigidTriangleSurfaceFeaturePlan& plan,
	bool includeSwept = true, bool includeDiscrete = true);

DY_AVBD_CONTACT_RIGID_SOFT_API void avbdDetectSoftRigidTriangleSurface(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts, PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL, PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidTriangleSurfaceRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts,
	PxArray<PxU32>& triangleBvhQueryCandidates,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL, PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidTriangleSurfaceSwept(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts, PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL, PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidTriangleSurfaceSweptRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts,
	AvbdRigidTriangleSurfaceQueryScratch& queryScratch,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL, PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, PxReal margin = 0.05f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL,
	const AvbdRigidTriangleSurfaceFeatureWorkItem* workItem = NULL,
	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		sweptSubstageTiming = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
		forwardOwnerQueryStats = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache*
		forwardOwnerResultCache = NULL);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, PxReal margin = 0.05f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL,
	const AvbdRigidTriangleSurfaceFeatureWorkItem* workItem = NULL,
	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats* discreteQueryStats = NULL,
	bool useBodyLocalBoundsCull = false);

DY_AVBD_CONTACT_RIGID_SOFT_API void
avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidTriangleSurfaceFeaturePlan& plan,
	PxU32 planBegin, PxU32 planEnd,
	PxArray<AvbdSoftContact>& contacts,
	AvbdRigidTriangleSurfaceQueryScratch& queryScratch,
	PxReal margin = 0.05f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceFeaturePlanRangeTiming* timing = NULL,
	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		sweptSubstageTiming = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
		forwardOwnerQueryStats = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache*
		forwardOwnerResultCache = NULL,
	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats* discreteQueryStats = NULL,
	bool useDiscreteBodyLocalBoundsCull = false);

#undef DY_AVBD_CONTACT_RIGID_SOFT_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_RIGID_SOFT_H
