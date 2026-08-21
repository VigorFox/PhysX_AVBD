// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_OGC_ADMISSION_H
#define DY_AVBD_OGC_ADMISSION_H

#include "foundation/PxArray.h"
#include "foundation/PxMat33.h"
#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"

namespace physx {
namespace Dy {

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
    defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
#define DY_AVBD_OGC_ADMISSION_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
#define DY_AVBD_OGC_ADMISSION_API PX_UNIX_EXPORT
#else
#define DY_AVBD_OGC_ADMISSION_API
#endif

struct AvbdOgcPairState;
struct AvbdOgcPairTrustRegionContext;
struct AvbdSoftBody;
struct AvbdSoftContact;
struct AvbdSoftIslandExecutionPlan;
struct AvbdSoftParticle;
struct AvbdSolverBody;

// Reusable connected-component buffers shared by initial, warm-start and
// nonlinear pose admission.  One Scene-owned instance is borrowed through
// AvbdSoftIslandExecutionPlan; local callers retain a stack fallback.
struct AvbdOgcAdmissionWorkspace {
  physx::PxArray<physx::PxU32> componentParents;
  physx::PxArray<physx::PxU8> participatingComponents;
  physx::PxArray<physx::PxReal> contactAlphas;
  physx::PxArray<physx::PxReal> componentAlphas;
  physx::PxArray<physx::PxReal> particleAlphas;
};

// Transactional boundary around one nonlinear pose-writing phase.  The
// phase may contain material, joint and attachment blocks, but OGC admission
// is performed once on their coupled endpoint rather than independently in
// each kernel.  This preserves the relative soft/rigid response of a block
// while preventing an unregistered writer from spending another pair's
// current geometry epoch.
struct AvbdOgcPoseWritePhaseState {
  physx::PxArray<physx::PxVec3> softPositionBefore;
  physx::PxArray<physx::PxVec3> rigidPositionBefore;
  physx::PxArray<physx::PxQuat> rigidRotationBefore;
  physx::PxArray<physx::PxMat33> rigidInvInertiaBefore;
  AvbdOgcAdmissionWorkspace scratch;
  bool active;

  AvbdOgcPoseWritePhaseState() : active(false) {}

  DY_AVBD_OGC_ADMISSION_API void capture(
      const AvbdOgcPairTrustRegionContext *context,
      const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
      const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
      const AvbdSolverBody *bodies, physx::PxU32 numBodies);
};

// Build the mutable pair-state / immutable contact-index view carried by one
// provider execution plan.  The view owns no storage and is valid only for
// the lifetime of the plan.
bool initializeOgcPairTrustRegionContextView(
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    physx::PxU32 numSoftContacts,
    AvbdOgcPairTrustRegionContext &context);

// Admit all pose edits made since capture() with one common alpha per
// connected soft/rigid OGC component.  The same component alpha also keeps
// every participating tet on its positive-J side.  Returns true when at least
// one component was clipped.  Contact-boundary clips request a fresh same-time
// DCD epoch; a positive-J-only clip does not invalidate contact geometry.
DY_AVBD_OGC_ADMISSION_API bool admitOgcPoseWritePhase(
    AvbdOgcPoseWritePhaseState &state,
    AvbdOgcPairTrustRegionContext *context,
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan);

void applyWorldStaticOgcInitialAdmission(
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan);

void applyOgcMixedWarmstartAdmission(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    const AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    physx::PxArray<physx::PxU8> *admittedContacts,
    physx::PxArray<physx::PxReal> *admittedNormalDisplacements);

bool initializeMixedOgcPairEpoch(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    const AvbdSoftParticle *softParticles, physx::PxU32 numSoftParticles,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const AvbdSoftIslandExecutionPlan *softExecutionPlan,
    bool useProvidedSoftExecutionPlan,
    const physx::PxArray<physx::PxU8> &admissionContacts,
    const physx::PxArray<physx::PxReal> &admissionDisplacements,
    AvbdOgcPairTrustRegionContext &context,
    AvbdOgcPairState *&pairStates, physx::PxU32 &numPairStates);

void initializeAvbdSoftContactDepenetrationTargets(
    AvbdSolverBody *bodies, physx::PxU32 numBodies,
    AvbdSoftParticle *softParticles,
    AvbdSoftBody *softBodies, physx::PxU32 numSoftBodies,
    AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    physx::PxReal dt);

physx::PxReal limitRigidOgcCandidate(
    const AvbdOgcPairTrustRegionContext *context,
    physx::PxU32 bodyIndex, const AvbdSolverBody &body,
    const physx::PxVec3 &deltaPosition,
    const physx::PxVec3 &deltaTheta,
    const AvbdSoftContact *softContacts, physx::PxU32 numSoftContacts,
    const physx::PxU32 *rigidTargetContactStarts,
    const physx::PxU32 *rigidTargetContactRefs,
    const AvbdSoftParticle *softParticles);

} // namespace Dy
} // namespace physx

#endif
