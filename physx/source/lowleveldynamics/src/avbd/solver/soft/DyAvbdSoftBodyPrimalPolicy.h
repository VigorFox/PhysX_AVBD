// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_PRIMAL_POLICY_H
#define DY_AVBD_SOFT_BODY_PRIMAL_POLICY_H

#include "foundation/PxArray.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"
#include "avbd/contact/DyAvbdContact.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyScheduling.h"
#include "avbd/solver/soft/DyAvbdSoftBodyWorkspace.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftBodyStepStats;

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API
#endif

enum class AvbdSoftBodyStepExecutionMode : PxU8
{
	eFULL,
	ePREPARE,
	eRESUME
};

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API bool avbdCanUseVelocityTangentOwner(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles);

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API void avbdAssignVelocityTangentOwners(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles);

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API void avbdProjectSoftContactVelocityTangents(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, AvbdSoftBodyStepStats* stepStats = NULL,
	const AvbdOgcGeometryEpochView* geometryEpoch = NULL);

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API void avbdApplyWorldStaticComponentEndpointDcdRecovery(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	AvbdSoftBodyWorkspace& workspace, PxU32 sweeps = 4u);

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API void avbdClampWorldStaticComponentEndpointDcdVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBodyWorkspace& workspace);

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API void avbdBuildSoftParticleContactIndex(
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles, AvbdSoftBodyStepStats* stepStats,
	AvbdParticlePrimalSchedule particlePrimalSchedule =
		AvbdParticlePrimalSchedule::eSERIAL_LINEAR,
	bool validateP4AccessPlan = false,
	const AvbdSoftParticle* probeParticles = NULL);

DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API void avbdPredictSoftBodyParticles(
	AvbdSoftParticle* particles, PxU32 numParticles,
	PxReal dt, const PxVec3& gravity, bool useAdaptiveInitialGuess);

#undef DY_AVBD_SOFT_BODY_PRIMAL_POLICY_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_PRIMAL_POLICY_H
