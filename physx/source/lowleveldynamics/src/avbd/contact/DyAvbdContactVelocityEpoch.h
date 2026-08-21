// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_CONTACT_VELOCITY_EPOCH_H
#define DY_AVBD_CONTACT_VELOCITY_EPOCH_H

#include "avbd/solver/soft/DyAvbdSoftBodyStep.h"

namespace physx
{
namespace Dy
{

void avbdCompileSoftVelocityObjectives(
	PxArray<AvbdCompiledSoftVelocityObjective>& compiledVelocityObjectives,
	PxArray<AvbdSoftComponentFinalizeMode>& componentFinalizeModes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* sourceContacts, PxU32 sourceContactCount);

bool avbdRefreshComponentTerminalOgcEpoch(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdContactRedetectFn redetectFn,
	PxArray<AvbdSoftContact>* contactsArray,
	void* redetectUserData,
	AvbdSoftContact*& contacts, PxU32& numContacts,
	AvbdSoftBodyWorkspace& workspace);

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_VELOCITY_EPOCH_H
