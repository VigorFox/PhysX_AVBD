// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_SOFT_BODY_FINALIZATION_H
#define DY_AVBD_SOFT_BODY_FINALIZATION_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_FINALIZATION_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_FINALIZATION_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_FINALIZATION_API
#endif

enum class AvbdSoftComponentFinalizeMode : PxU8
{
	eMOMENTUM,
	eKINEMATIC_CONTACT,
	ePOSITION_OWNED,
	eUNSUPPORTED
};

struct AvbdSoftComponentMomentumTarget
{
	PxVec3 centroid;
	PxVec3 linearMomentum;
	PxVec3 angularMomentum;
	PxReal mass;
	bool valid;

	AvbdSoftComponentMomentumTarget()
		: centroid(0.0f), linearMomentum(0.0f), angularMomentum(0.0f),
		  mass(0.0f), valid(false)
	{
	}
};

struct AvbdCompiledSoftVelocityObjective
{
	AvbdVelocityObjectiveOwner owner;
	AvbdSoftContactSource source;
	PxU32 bodyIndex;
	PxU32 particleIndex;
	AvbdWeightedContactPoint queryPoint;
	PxVec3 normal;
	PxVec3 surfacePoint;
	PxVec3 previousSurfacePoint;

	AvbdCompiledSoftVelocityObjective()
		: owner(AvbdVelocityObjectiveOwner::Unsupported), source(),
		  bodyIndex(PX_MAX_U32), particleIndex(PX_MAX_U32), queryPoint(),
		  normal(0.0f, 1.0f, 0.0f), surfacePoint(0.0f),
		  previousSurfacePoint(0.0f)
	{
	}
};

DY_AVBD_SOFT_BODY_FINALIZATION_API bool avbdComputeSoftComponentMomentum(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody& body, bool usePrediction, PxReal invDt,
	PxVec3& centroid, PxVec3& linearMomentum,
	PxVec3& angularMomentum, PxMat33& inertia, PxReal& mass);

DY_AVBD_SOFT_BODY_FINALIZATION_API void avbdApplySoftComponentDampingToMomentumTarget(
	AvbdSoftComponentMomentumTarget& target,
	const AvbdSoftBody& body, PxReal dt);

DY_AVBD_SOFT_BODY_FINALIZATION_API void avbdFinalizeSoftComponentVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftComponentMomentumTarget* momentumTargets,
	const AvbdSoftComponentFinalizeMode* finalizeModes,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdCompiledSoftVelocityObjective* velocityObjectives,
	PxU32 numVelocityObjectives, PxReal invDt);

// Public world attachments are hard kinematic constraints. Position AL keeps
// them coupled to the material solve; this terminal projection removes the
// finite-penalty residual and the corresponding point velocity.
DY_AVBD_SOFT_BODY_FINALIZATION_API void avbdProjectWorldFixedPins(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies);

DY_AVBD_SOFT_BODY_FINALIZATION_API void avbdUpdateSoftContactDual(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles, PxReal beta);

#undef DY_AVBD_SOFT_BODY_FINALIZATION_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_FINALIZATION_H
