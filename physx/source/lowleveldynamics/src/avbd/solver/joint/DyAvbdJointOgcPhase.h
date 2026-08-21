// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_JOINT_OGC_PHASE_H
#define DY_AVBD_JOINT_OGC_PHASE_H

#include "avbd/ogc/DyAvbdOgcTrustRegion.h"
#include "foundation/PxArray.h"

namespace physx {
namespace Dy {

class AvbdSolver;
struct AvbdSoftBody;
struct AvbdSoftContact;
struct AvbdSoftIslandExecutionPlan;
struct AvbdSoftParticle;
struct AvbdSolverBody;
struct AvbdSolverStats;

struct AvbdJointOgcAdmissionInput {
  AvbdSolver &solver;
  physx::PxReal dt;
  AvbdSolverBody *bodies;
  physx::PxU32 numBodies;
  AvbdSoftParticle *softParticles;
  physx::PxU32 numSoftParticles;
  AvbdSoftBody *softBodies;
  physx::PxU32 numSoftBodies;
  AvbdSoftContact *softContacts;
  physx::PxU32 numSoftContacts;
  const AvbdSoftIslandExecutionPlan *softExecutionPlan;
  bool useProvidedSoftExecutionPlan;
  bool hasCompleteSoftSelection;
  AvbdSolverStats &stats;
};

struct AvbdJointOgcAdmissionState {
  physx::PxArray<physx::PxU8> admissionContacts;
  physx::PxArray<physx::PxReal> admissionDisplacements;
  AvbdOgcPairState *pairStates = nullptr;
  physx::PxU32 numPairStates = 0u;
  AvbdOgcPairTrustRegionContext pairContext;

  void initialize(const AvbdJointOgcAdmissionInput &input);
};

struct AvbdJointOgcVelocityHandoffInput {
  AvbdSolverBody *bodies;
  physx::PxU32 numBodies;
  AvbdSoftParticle *softParticles;
  physx::PxU32 numSoftParticles;
  AvbdSoftBody *softBodies;
  physx::PxU32 numSoftBodies;
  AvbdSoftContact *softContacts;
  physx::PxU32 numSoftContacts;
  AvbdOgcPairState *mixedOgcPairStates;
  physx::PxU32 numMixedOgcPairStates;
  const AvbdSoftIslandExecutionPlan *softExecutionPlan;
  bool useProvidedSoftExecutionPlan;
  AvbdSolverStats &stats;
};

} // namespace Dy
} // namespace physx

#endif
