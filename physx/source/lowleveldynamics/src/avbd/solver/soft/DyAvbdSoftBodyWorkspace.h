// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#ifndef DY_AVBD_SOFT_BODY_WORKSPACE_H
#define DY_AVBD_SOFT_BODY_WORKSPACE_H

#include "foundation/PxArray.h"
#include "avbd/contact/DyAvbdSoftContactWorkspace.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "avbd/solver/soft/DyAvbdSoftBodyFinalization.h"
#include "avbd/solver/soft/DyAvbdSoftBodyScheduling.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftContactParticleRef
{
	PxU32 contactIndex;
	PxReal jacobianScale;

	AvbdSoftContactParticleRef()
		: contactIndex(PX_MAX_U32), jacobianScale(0.0f)
	{
	}

	AvbdSoftContactParticleRef(PxU32 index, PxReal scale)
		: contactIndex(index), jacobianScale(scale)
	{
	}
};

enum class AvbdParticlePrimalDynamicAccessSource : PxU8
{
	eCONTACT,
	ePIN_OBJECTIVE
};

struct AvbdParticlePrimalDynamicAccessGroup
{
	PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
	PxU8 particleCount;
	AvbdParticlePrimalDynamicAccessSource source;
	PxU16 padding;

	AvbdParticlePrimalDynamicAccessGroup()
		: particleCount(0),
		  source(AvbdParticlePrimalDynamicAccessSource::eCONTACT),
		  padding(0)
	{
		for(PxU32 i = 0; i < AVBD_CONTACT_MAX_PARTICLES; ++i)
			particleIndices[i] = PX_MAX_U32;
	}
};

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_WORKSPACE_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_WORKSPACE_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_WORKSPACE_API
#endif

struct AvbdSoftBodyWorkspace
{
	AvbdSoftContactWorkspace contact;
	PxArray<AvbdSoftContactParticleRef> contactIndices;
	PxArray<PxU32> contactStarts;
	PxArray<PxU32> contactCounts;
	PxArray<AvbdParticlePrimalDynamicAccessGroup>
		particlePrimalDynamicAccessGroups;
	PxArray<PxU32> particlePrimalDynamicConflictOffsets;
	PxArray<PxU32> particlePrimalDynamicConflictIndices;
	PxArray<PxU32> particlePrimalDynamicConflictCounts;
	PxArray<PxU32> particlePrimalBodyIndices;
	PxArray<PxU32> particlePrimalColors;
	PxArray<PxU32> particlePrimalColorCounts;
	PxArray<PxU32> particlePrimalColorOffsets;
	PxArray<PxU32> particlePrimalColorParticles;
	PxU32 particlePrimalColorCount;
	bool particlePrimalDynamicConflictValid;
	bool particlePrimalColorPlanValid;
	PxArray<PxVec3> chebyPrevPos;
	PxArray<PxVec3> chebyPrevPrevPos;
	PxArray<PxReal> selfCollisionSafetyBounds;
	PxArray<PxReal> bodySelfCollisionSafetyBounds;
	AvbdOgcGeometryEpochSidecar componentOgcGeometrySidecar;
	PxArray<AvbdOgcPairState> componentOgcPairStates;
	PxArray<PxU32> componentOgcPairIndices;
	PxArray<PxU8> componentOgcSafetyBodyMask;
	PxArray<AvbdCompiledSoftVelocityObjective> compiledVelocityObjectives;
	PxArray<AvbdSoftComponentMomentumTarget> componentMomentumTargets;
	PxArray<AvbdSoftComponentFinalizeMode> componentFinalizeModes;
	PxArray<PxU8> worldStaticEndpointRecoveredBodies;
	PxU64 growthEvents;
	PxU64 growthBytes;
	PxU32 peakContactIncidenceCount;
	PxU32 peakContactIncidenceCapacity;

	DY_AVBD_SOFT_BODY_WORKSPACE_API AvbdSoftBodyWorkspace();

	DY_AVBD_SOFT_BODY_WORKSPACE_API void reserve(
		PxU32 numParticles, PxU32 contactCapacity,
		AvbdParticlePrimalSchedule particlePrimalSchedule =
			AvbdParticlePrimalSchedule::eDEFAULT);

	template<typename T, typename Alloc>
	void resize(PxArray<T, Alloc>& array, PxU32 size)
	{
		if(size > array.capacity())
		{
			growthEvents++;
			growthBytes += PxU64(size - array.capacity()) * sizeof(T);
		}
		array.resize(size);
	}

	DY_AVBD_SOFT_BODY_WORKSPACE_API void beginStep();
	DY_AVBD_SOFT_BODY_WORKSPACE_API void reset();
	DY_AVBD_SOFT_BODY_WORKSPACE_API AvbdOgcGeometryEpochView
	getComponentOgcGeometryEpochView() const;
};

#undef DY_AVBD_SOFT_BODY_WORKSPACE_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_WORKSPACE_H
