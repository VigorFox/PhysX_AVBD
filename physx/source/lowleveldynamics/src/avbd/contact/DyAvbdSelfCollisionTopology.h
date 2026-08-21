// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_SELF_COLLISION_TOPOLOGY_H
#define DY_AVBD_SELF_COLLISION_TOPOLOGY_H

#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SELF_COLLISION_TOPOLOGY_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SELF_COLLISION_TOPOLOGY_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SELF_COLLISION_TOPOLOGY_API
#endif

DY_AVBD_SELF_COLLISION_TOPOLOGY_API void
avbdBuildSelfCollisionAdjacency(
	const AvbdSoftBody& softBody, AvbdSelfCollisionAdjacency& adjacency);

DY_AVBD_SELF_COLLISION_TOPOLOGY_API void
avbdBuildAllSelfCollisionAdjacencies(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSelfCollisionAdjacency>& adjacencies);

#undef DY_AVBD_SELF_COLLISION_TOPOLOGY_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SELF_COLLISION_TOPOLOGY_H
