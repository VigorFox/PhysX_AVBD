// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_JOINT_SOFT_EXECUTION_DATA_H
#define DY_AVBD_JOINT_SOFT_EXECUTION_DATA_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyWorkspace.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftIslandExecutionPlan;

// Mutable fallback storage plus immutable views consumed by one mixed-island
// joint solve. The execution plan remains the preferred owner; these arrays
// exist only when a caller cannot provide a compiled plan.
struct AvbdSoftExecutionData
{
	PxArray<PxU32> particleBodyIndicesStorage;
	PxArray<PxU8> particleBodyConflictsStorage;
	PxArray<PxU32> contactStartsStorage;
	PxArray<PxU32> contactCountsStorage;
	PxArray<AvbdSoftContactParticleRef> contactRefsStorage;
	PxArray<PxU32> triangleCoreSafetyStartsStorage;
	PxArray<PxU32> triangleCoreSafetyCountsStorage;
	PxArray<AvbdSoftContactParticleRef> triangleCoreSafetyRefsStorage;

	const PxU32* particleBodyIndices;
	const PxU32* contactStarts;
	const AvbdSoftContactParticleRef* contactRefs;
	const PxU32* triangleCoreSafetyStarts;
	const AvbdSoftContactParticleRef* triangleCoreSafetyRefs;
	PxU32 numTriangleCoreSafetyStarts;
	PxU32 numTriangleCoreSafetyRefs;
	const PxU32* rigidTargetContactStarts;
	const PxU32* rigidTargetContactRefs;

	AvbdSoftExecutionData()
		: particleBodyIndices(NULL), contactStarts(NULL), contactRefs(NULL),
		  triangleCoreSafetyStarts(NULL), triangleCoreSafetyRefs(NULL),
		  numTriangleCoreSafetyStarts(0), numTriangleCoreSafetyRefs(0),
		  rigidTargetContactStarts(NULL), rigidTargetContactRefs(NULL)
	{
	}
};

void initializeAvbdSoftExecutionData(
	const AvbdSoftIslandExecutionPlan* softExecutionPlan,
	bool useProvidedSoftExecutionPlan,
	bool useProvidedRigidTargetContactPlan,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* softContacts, PxU32 numSoftContacts,
	PxU32 numSoftParticles, AvbdSoftExecutionData& data);

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_JOINT_SOFT_EXECUTION_DATA_H
