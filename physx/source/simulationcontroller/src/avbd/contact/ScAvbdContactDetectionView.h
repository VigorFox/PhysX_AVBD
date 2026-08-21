// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_CONTACT_DETECTION_VIEW_H
#define SC_AVBD_CONTACT_DETECTION_VIEW_H

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Sc
{

PX_FORCE_INLINE void configureAvbdContactDetectionView(
	Dy::AvbdSoftContactDetectionView& view,
	const Dy::AvbdRigidBox* rigidBoxes, PxU32 numRigidBoxes,
	const Dy::AvbdRigidSphere* rigidSpheres, PxU32 numRigidSpheres,
	const Dy::AvbdRigidCapsule* rigidCapsules, PxU32 numRigidCapsules,
	const Dy::AvbdRigidConvex* rigidConvexes, PxU32 numRigidConvexes,
	const Dy::AvbdRigidTriangleSurface* rigidTriangleSurfaces,
	PxU32 numRigidTriangleSurfaces,
	const Dy::AvbdWorldPlane* worldPlanes, PxU32 numWorldPlanes,
	const PxU8* selfCollisionEnabled)
{
	view.rigidBoxes = rigidBoxes;
	view.numRigidBoxes = numRigidBoxes;
	view.rigidSpheres = rigidSpheres;
	view.numRigidSpheres = numRigidSpheres;
	view.rigidCapsules = rigidCapsules;
	view.numRigidCapsules = numRigidCapsules;
	view.rigidConvexes = rigidConvexes;
	view.numRigidConvexes = numRigidConvexes;
	view.rigidTriangleSurfaces = rigidTriangleSurfaces;
	view.numRigidTriangleSurfaces = numRigidTriangleSurfaces;
	view.worldPlanes = worldPlanes;
	view.numWorldPlanes = numWorldPlanes;
	view.includeLegacyGround = false;
	view.selfCollisionEnabled = selfCollisionEnabled;
}

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_CONTACT_DETECTION_VIEW_H
