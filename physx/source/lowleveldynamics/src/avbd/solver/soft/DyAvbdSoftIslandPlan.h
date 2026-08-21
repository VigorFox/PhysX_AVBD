// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

#ifndef DY_AVBD_SOFT_ISLAND_PLAN_H
#define DY_AVBD_SOFT_ISLAND_PLAN_H

#include "avbd/ogc/DyAvbdOgcGeometryProvider.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

struct AvbdOgcPairState;
struct AvbdOgcTriangleCoreCertificate;
struct AvbdOgcAdmissionWorkspace;
struct AvbdSoftBody;
struct AvbdSoftContactParticleRef;
struct AvbdPostAlWorkspace;
struct AvbdWeightedContactPoint;

// Provider-owned, immutable-for-one-island-solve support program.  The
// arrays are non-owning views into the provider's selection storage and must
// outlive the native soft-island task.  Keeping this view separate from the
// solver implementation makes the Scene-to-solver contract explicit without
// importing the monolithic soft-body component header.
struct AvbdSoftIslandExecutionPlan
{
	const PxU32 *particleBodyIndices;
	PxU32 numParticleBodyIndices;
	const PxU32 *contactStarts;
	PxU32 numContactStarts;
	const AvbdSoftContactParticleRef *contactRefs;
	PxU32 numContactRefs;

	// Geometry-only companion CSR for triangle/OBB core rows. It does not
	// change contact-force ownership; it is consulted by OGC admission.
	const PxU32 *triangleCoreSafetyStarts;
	PxU32 numTriangleCoreSafetyStarts;
	const AvbdSoftContactParticleRef *triangleCoreSafetyRefs;
	PxU32 numTriangleCoreSafetyRefs;

	// Inverse CSR for the rigid endpoint of dynamic soft/rigid pairs.
	const PxU32 *rigidTargetContactStarts;
	PxU32 numRigidTargetContactStarts;
	const PxU32 *rigidTargetContactRefs;
	PxU32 numRigidTargetContactRefs;

	// Immutable geometry-provider view for the optional terminal current-pose
	// OGC epoch. Scene/provider storage outlives the native island task.
	AvbdOgcCurrentPoseGeometryProvider terminalGeometryProvider;

	// Pair state is mutable only through the solver-owned OGC state; all views
	// and CSR ranges remain immutable for one island solve.
	AvbdOgcPairState *ogcPairStates;
	PxU32 numOgcPairStates;
	const PxU32 *ogcPairIndices;
	PxU32 numOgcPairIndices;
	const PxU32 *ogcPairContactStarts;
	PxU32 numOgcPairContactStarts;
	const PxU32 *ogcPairContactRefs;
	PxU32 numOgcPairContactRefs;
	const AvbdOgcTriangleCoreCertificate *ogcTriangleCoreCertificates;
	PxU32 numOgcTriangleCoreCertificates;
	const PxU32 *ogcContactTriangleCoreIndices;
	PxU32 numOgcContactTriangleCoreIndices;
	PxU32 ogcGeometryEpoch;

	// Mutable, island-exclusive scratch owned by the Scene selection storage.
	// The execution plan only borrows it for the duration of one native solve;
	// serial and task executors therefore share the same allocation contract.
	AvbdPostAlWorkspace *postAlWorkspace;
	AvbdOgcAdmissionWorkspace *ogcAdmissionWorkspace;

	bool softPredictionPrepared;

	AvbdSoftIslandExecutionPlan()
		: particleBodyIndices(NULL), numParticleBodyIndices(0),
		  contactStarts(NULL), numContactStarts(0), contactRefs(NULL),
		  numContactRefs(0), triangleCoreSafetyStarts(NULL),
		  numTriangleCoreSafetyStarts(0), triangleCoreSafetyRefs(NULL),
		  numTriangleCoreSafetyRefs(0), rigidTargetContactStarts(NULL),
		  numRigidTargetContactStarts(0), rigidTargetContactRefs(NULL),
		  numRigidTargetContactRefs(0),
		  ogcPairStates(NULL), numOgcPairStates(0), ogcPairIndices(NULL),
		  numOgcPairIndices(0), ogcPairContactStarts(NULL),
		  numOgcPairContactStarts(0), ogcPairContactRefs(NULL),
		  numOgcPairContactRefs(0), ogcTriangleCoreCertificates(NULL),
		  numOgcTriangleCoreCertificates(0),
		  ogcContactTriangleCoreIndices(NULL),
		  numOgcContactTriangleCoreIndices(0),
		  ogcGeometryEpoch(0),
		  postAlWorkspace(NULL),
		  ogcAdmissionWorkspace(NULL),
		  softPredictionPrepared(false)
	{
	}

	PX_FORCE_INLINE bool isComplete(PxU32 numParticles) const
	{
		return particleBodyIndices && numParticleBodyIndices == numParticles &&
			contactStarts && numContactStarts == numParticles + 1 &&
			(numContactRefs == 0 || contactRefs) && contactStarts[0] == 0 &&
			contactStarts[numParticles] == numContactRefs;
	}

	PX_FORCE_INLINE bool hasRigidTargetContactPlan(PxU32 numRigidBodies) const
	{
		return numRigidBodies > 0 && rigidTargetContactStarts &&
			numRigidTargetContactStarts == numRigidBodies + 1 &&
			(numRigidTargetContactRefs == 0 || rigidTargetContactRefs) &&
			rigidTargetContactStarts[0] == 0 &&
			rigidTargetContactStarts[numRigidBodies] ==
				numRigidTargetContactRefs;
	}

	PX_FORCE_INLINE bool hasTriangleCoreSafetyPlan(PxU32 numParticles) const
	{
		return triangleCoreSafetyStarts &&
			numTriangleCoreSafetyStarts == numParticles + 1 &&
			(numTriangleCoreSafetyRefs == 0 || triangleCoreSafetyRefs) &&
			triangleCoreSafetyStarts[0] == 0 &&
			triangleCoreSafetyStarts[numParticles] == numTriangleCoreSafetyRefs;
	}

	PX_FORCE_INLINE bool hasTerminalCurrentPoseGeometryPlan(
		PxU32 numSimulationParticles) const
	{
		return terminalGeometryProvider.isComplete(numSimulationParticles);
	}

	PX_FORCE_INLINE bool hasMixedOgcPairPlan(PxU32 numContacts) const
	{
		return ogcPairStates && numOgcPairStates > 0 && ogcPairIndices &&
			numOgcPairIndices == numContacts && numContacts > 0;
	}

	PX_FORCE_INLINE bool hasMixedOgcPairContactPlan(PxU32 numContacts) const
	{
		return hasMixedOgcPairPlan(numContacts) && ogcPairContactStarts &&
			numOgcPairContactStarts == numOgcPairStates + 1 &&
			(numOgcPairContactRefs == 0 || ogcPairContactRefs) &&
			ogcPairContactStarts[0] == 0 &&
			ogcPairContactStarts[numOgcPairStates] == numOgcPairContactRefs;
	}

	PX_FORCE_INLINE bool hasOgcTriangleCoreGeometryPlan(
		PxU32 numContacts) const
	{
		return ogcContactTriangleCoreIndices &&
			ogcGeometryEpoch != 0u &&
			numOgcContactTriangleCoreIndices == numContacts &&
			(numOgcTriangleCoreCertificates == 0 ||
			 ogcTriangleCoreCertificates);
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_ISLAND_PLAN_H
