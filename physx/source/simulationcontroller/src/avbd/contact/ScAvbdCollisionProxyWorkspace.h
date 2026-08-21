// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_COLLISION_PROXY_WORKSPACE_H
#define SC_AVBD_COLLISION_PROXY_WORKSPACE_H

#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace physx
{
namespace Sc
{

// Scene-owned collision-domain data and rebuild scratch.  The proxy arrays
// are geometry-only; contact expansion remains the single consumer that
// converts them back to simulation supports.
struct AvbdCollisionProxyWorkspace
{
	PxArray<Dy::AvbdSoftParticle> collisionParticles;
	PxArray<Dy::AvbdSoftBody> collisionBodies;
	PxArray<Dy::AvbdWeightedContactPoint> collisionVertexMappings;
	PxArray<Dy::AvbdSelfCollisionAdjacency> collisionSelfCollisionAdjacencies;

	PxArray<Dy::AvbdSoftParticle> subsetParticles;
	PxArray<Dy::AvbdSoftBody> subsetBodies;
	PxArray<Dy::AvbdWeightedContactPoint> subsetVertexMappings;
	PxArray<Dy::AvbdSelfCollisionAdjacency> subsetSelfCollisionAdjacencies;
};

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_COLLISION_PROXY_WORKSPACE_H
