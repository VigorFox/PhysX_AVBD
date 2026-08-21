// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_SCHEDULING_POLICY_H
#define SC_AVBD_SCHEDULING_POLICY_H

#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Sc
{

bool useAvbdStaticWorldSelfOgcTaskFanIn();
bool useAvbdVolumeTest3x3Cadence();

static const PxU32 eAVBD_STATIC_WORLD_SELF_OGC_MIN_ITEMS_PER_TASK = 128u;

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_SCHEDULING_POLICY_H
