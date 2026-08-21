// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_PAIR_STATE_H
#define DY_AVBD_OGC_PAIR_STATE_H

#include "avbd/ogc/DyAvbdOgcPair.h"
#include "foundation/PxArray.h"

namespace physx {
namespace Dy {

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
    defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
#define DY_AVBD_OGC_PAIR_STATE_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
#define DY_AVBD_OGC_PAIR_STATE_API PX_UNIX_EXPORT
#else
#define DY_AVBD_OGC_PAIR_STATE_API
#endif

struct AvbdSoftContact;
struct AvbdSoftParticle;
struct AvbdSolverBody;
struct AvbdRigidBox;

enum AvbdOgcPairProviderTarget : physx::PxU32
{
	eOGC_PAIR_PROVIDER_WORLD_STATIC = 1u << 0,
	eOGC_PAIR_PROVIDER_DYNAMIC_RIGID = 1u << 1,
	eOGC_PAIR_PROVIDER_DEFORMABLE = 1u << 2
};

// Publish the immutable pair identity and shape descriptor at the contact
// provider boundary. Derived current-pose geometry and all solve state are
// initialized later by the consuming solver epoch.
bool compileOgcPairProviderPlan(
	const AvbdSoftContact* contacts, physx::PxU32 numContacts,
	const AvbdRigidBox* rigidBoxes, physx::PxU32 numRigidBoxes,
	physx::PxU32 numSoftBodies, physx::PxU32 numDynamicRigidBodies,
	physx::PxU32 targetMask,
	physx::PxArray<AvbdOgcPairState>& pairStates,
	physx::PxArray<physx::PxU32>& pairIndices);

// Refresh the single terminal working registry from a freshly detected
// current-pose manifold. Existing selection pairs retain their solve state;
// a pair first discovered at t=dt is appended to this same registry instead
// of acquiring a second response/velocity owner. The detected arrays are
// caller-owned scratch and are reused across closure passes.
DY_AVBD_OGC_PAIR_STATE_API bool refreshCurrentOgcPairRegistry(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    const AvbdRigidBox *rigidBoxes, physx::PxU32 numRigidBoxes,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSolverBody *bodies, physx::PxU32 numBodies,
    physx::PxU32 numSoftBodies,
    physx::PxArray<AvbdOgcPairState> &pairRegistry,
    physx::PxArray<AvbdOgcPairState> &detectedPairScratch,
    physx::PxArray<physx::PxU32> &detectedPairIndexScratch,
    physx::PxArray<physx::PxU32> &detectedPairToRegistryScratch,
    physx::PxArray<physx::PxU32> &pairIndices);

void consumeCurrentOgcPairRefreshRequests(
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *pairIndices, physx::PxU32 numPairIndices);

bool publishLocalOgcPairPositionResult(
    const AvbdSoftContact *contacts, physx::PxU32 numContacts,
    physx::PxU32 contactIndex, physx::PxReal correction,
    AvbdOgcVelocityContactDomain contactDomain,
    AvbdOgcPairState *pairStates, physx::PxU32 numPairStates,
    const physx::PxU32 *pairIndices, physx::PxU32 numPairIndices);

} // namespace Dy
} // namespace physx

#endif
