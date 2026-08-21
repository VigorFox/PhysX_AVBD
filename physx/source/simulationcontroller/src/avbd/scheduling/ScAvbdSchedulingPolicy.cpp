// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdSchedulingPolicy.h"

#include <cstdlib>

namespace physx
{
namespace Sc
{

bool useAvbdStaticWorldSelfOgcTaskFanIn()
{
	static const bool enabled = []()
	{
		const char* const value = std::getenv(
			"PHYSX_AVBD_P5_STATIC_WORLD_SELF_TASK_FANIN");
		return !(value && value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

bool useAvbdVolumeTest3x3Cadence()
{
	static const bool enabled = []()
	{
		const char* const value = std::getenv(
			"PHYSX_AVBD_VOLUME_TEST_3X3");
		return value && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

} // namespace Sc
} // namespace physx
