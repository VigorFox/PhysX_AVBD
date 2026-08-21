// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_CONTACT_EPOCH_H
#define DY_AVBD_CONTACT_EPOCH_H

#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/contact/DyAvbdSoftContactWorkspace.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_CONTACT_EPOCH_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_CONTACT_EPOCH_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_CONTACT_EPOCH_API
#endif

DY_AVBD_CONTACT_EPOCH_API void avbdResetSoftContactDepenetrationLimits(
	AvbdSoftContact* contacts, PxU32 numContacts);

DY_AVBD_CONTACT_EPOCH_API void
avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
	AvbdSoftContact& contact,
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxVec3& initialSurfacePoint, PxReal dt);

DY_AVBD_CONTACT_EPOCH_API void avbdInitializeSoftContactDepenetrationLimits(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxReal dt);

DY_AVBD_CONTACT_EPOCH_API void avbdTransferSoftContactState(
	const AvbdSoftContact* previousContacts, PxU32 numPreviousContacts,
	const AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL);

DY_AVBD_CONTACT_EPOCH_API void avbdBuildSoftContactRedetectionPhasePlan(
	AvbdSoftContactWorkspace& workspace,
	PxU32 numWorldPlanes, bool includeLegacyGround,
	PxU32 numRigidBoxes, PxU32 numRigidSpheres,
	PxU32 numRigidCapsules, PxU32 numRigidConvexes,
	PxU32 numRigidTriangleSurfaces, PxU32 numSoftBodies,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	const PxU8* selfCollisionEnabled);

DY_AVBD_CONTACT_EPOCH_API void avbdBeginSoftContactRedetection(
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace& workspace,
	AvbdSoftCollisionStats* stats = NULL);

DY_AVBD_CONTACT_EPOCH_API void avbdCompleteSoftContactRedetection(
	AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace& workspace);

#undef DY_AVBD_CONTACT_EPOCH_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_EPOCH_H
