// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_ISLAND_SELECTION_PLAN_H
#define SC_AVBD_ISLAND_SELECTION_PLAN_H

#include "avbd/selection/ScAvbdIslandSelectionStorage.h"

namespace physx
{
namespace Sc
{

bool compileAvbdIslandSelectionExecutionPlan(
	AvbdIslandSelectionStorage& storage,
	PxU32 numParticles,
	PxU32 numRigidBodies);

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_ISLAND_SELECTION_PLAN_H
