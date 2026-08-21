// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_OGC_GEOMETRY_PROVIDER_H
#define DY_AVBD_OGC_GEOMETRY_PROVIDER_H

#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

struct AvbdRigidBox;
struct AvbdRigidCapsule;
struct AvbdRigidConvex;
struct AvbdRigidSphere;
struct AvbdRigidTriangleSurface;
struct AvbdSoftBody;
struct AvbdWeightedContactPoint;
struct AvbdWorldPlane;

// Immutable collision-domain contract for a same-time current-pose OGC
// refresh.  The provider owns no solver state and exposes no previous pose,
// swept query or TOI input.  A native island task may refit these descriptors
// to its current solver pose, but it must not retain the refit beyond the
// geometry epoch.
struct AvbdOgcCurrentPoseGeometryProvider
{
	const AvbdSoftBody* collisionBodies;
	PxU32 numCollisionBodies;
	const AvbdWeightedContactPoint* collisionVertexMappings;
	PxU32 numCollisionVertexMappings;

	const AvbdWorldPlane* worldPlanes;
	PxU32 numWorldPlanes;
	const AvbdRigidBox* rigidBoxes;
	PxU32 numRigidBoxes;
	const AvbdRigidSphere* rigidSpheres;
	PxU32 numRigidSpheres;
	const AvbdRigidCapsule* rigidCapsules;
	PxU32 numRigidCapsules;
	const AvbdRigidConvex* rigidConvexes;
	PxU32 numRigidConvexes;
	const AvbdRigidTriangleSurface* rigidTriangleSurfaces;
	PxU32 numRigidTriangleSurfaces;

	PxReal contactRadius;
	bool includeSoftTargets;

	AvbdOgcCurrentPoseGeometryProvider()
		: collisionBodies(NULL), numCollisionBodies(0),
		  collisionVertexMappings(NULL), numCollisionVertexMappings(0),
		  worldPlanes(NULL), numWorldPlanes(0), rigidBoxes(NULL),
		  numRigidBoxes(0), rigidSpheres(NULL), numRigidSpheres(0),
		  rigidCapsules(NULL), numRigidCapsules(0), rigidConvexes(NULL),
		  numRigidConvexes(0), rigidTriangleSurfaces(NULL),
		  numRigidTriangleSurfaces(0), contactRadius(0.0f),
		  includeSoftTargets(true)
	{
	}

	PX_FORCE_INLINE bool hasRigidOrStaticTargets() const
	{
		return numWorldPlanes > 0 || numRigidBoxes > 0 ||
			numRigidSpheres > 0 || numRigidCapsules > 0 ||
			numRigidConvexes > 0 || numRigidTriangleSurfaces > 0;
	}

	PX_FORCE_INLINE bool isComplete(PxU32 numSimulationParticles) const
	{
		const bool validTargetViews =
			(numWorldPlanes == 0 || worldPlanes) &&
			(numRigidBoxes == 0 || rigidBoxes) &&
			(numRigidSpheres == 0 || rigidSpheres) &&
			(numRigidCapsules == 0 || rigidCapsules) &&
			(numRigidConvexes == 0 || rigidConvexes) &&
			(numRigidTriangleSurfaces == 0 || rigidTriangleSurfaces);
		return collisionBodies && numCollisionBodies > 0 &&
			collisionVertexMappings && numCollisionVertexMappings > 0 &&
			numSimulationParticles > 0 && validTargetViews &&
			(hasRigidOrStaticTargets() ||
			 (includeSoftTargets && numCollisionBodies > 1)) &&
			PxIsFinite(contactRadius) && contactRadius > 0.0f;
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_OGC_GEOMETRY_PROVIDER_H
