// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_DETECTION_H
#define DY_AVBD_CONTACT_DETECTION_H

#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/ogc/DyAvbdOgcParameters.h"
#include "foundation/PxArray.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftContact;
struct AvbdSoftContactWorkspace;
struct AvbdOgcGeometryEpochSidecar;

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_DETECTION_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_DETECTION_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_DETECTION_API
#endif

DY_AVBD_CONTACT_DETECTION_API void avbdDetectAllOGCContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidBox* rigidBoxes, PxU32 numRigidBoxes,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	PxReal groundY = 0.0f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	const AvbdWorldPlane* worldPlanes = NULL,
	PxU32 numWorldPlanes = 0,
	bool includeLegacyGround = true,
	const PxU8* selfCollisionEnabled = NULL,
	const AvbdRigidSphere* rigidSpheres = NULL,
	PxU32 numRigidSpheres = 0,
	const AvbdRigidCapsule* rigidCapsules = NULL,
	PxU32 numRigidCapsules = 0,
	const AvbdRigidConvex* rigidConvexes = NULL,
	PxU32 numRigidConvexes = 0,
	const AvbdRigidTriangleSurface* rigidTriangleSurfaces = NULL,
	PxU32 numRigidTriangleSurfaces = 0,
	AvbdOgcGeometryEpochSidecar* geometrySidecar = NULL);

DY_AVBD_CONTACT_DETECTION_API void avbdDetectAllOGCContacts(
	const AvbdSoftContactDetectionView& view,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	PxReal groundY = 0.0f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	AvbdOgcGeometryEpochSidecar* geometrySidecar = NULL);

// One same-time discrete detection epoch. Unlike avbdDetectAllOGCContacts,
// this path never evaluates a previous pose, swept segment or TOI. It is the
// geometry-provider consumer used by terminal OGC refresh.
DY_AVBD_CONTACT_DETECTION_API bool avbdDetectCurrentPoseOGCContacts(
	const AvbdSoftContactDetectionView& view,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	AvbdOgcGeometryEpochSidecar* geometrySidecar = NULL);

#undef DY_AVBD_CONTACT_DETECTION_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_DETECTION_H
