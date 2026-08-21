// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_SCENE_STATE_TYPES_H
#define SC_AVBD_SCENE_STATE_TYPES_H

#include "avbd/scheduling/ScAvbdTaskGraphTelemetry.h"

namespace physx
{
namespace Sc
{

		// P3 pre-solve preparation is deliberately stateful even while it is
		// still resumed synchronously below.  The later prediction fan-in must
		// occur after this exact prefix (initial OGC detection and the iteration
		// plan), but before avbdStepSoftBodies() performs its predicted-position
		// redetection.  Keeping the plan explicit prevents a future continuation
		// from recomputing live Scene state after child tasks have started.
		struct ComponentFallbackPlan
		{
			ComponentFallbackPlan()
				: outerIterations(1), innerIterations(1),
				  totalPositionIterations(1),
				  initialContactWorkspaceGrowthEvents(0),
				  initialContactWorkspaceGrowthBytes(0),
				  initialContactSweepScratchGrowthEvents(0),
				  initialContactSweepScratchGrowthBytes(0),
				  initialContactOutputGrowthEvents(0),
				  initialContactOutputGrowthBytes(0)
			{
			}

			PxU32	outerIterations;
			PxU32	innerIterations;
			PxU32	totalPositionIterations;
			PxU64	initialContactWorkspaceGrowthEvents;
			PxU64	initialContactWorkspaceGrowthBytes;
			PxU64	initialContactSweepScratchGrowthEvents;
			PxU64	initialContactSweepScratchGrowthBytes;
			PxU64	initialContactOutputGrowthEvents;
			PxU64	initialContactOutputGrowthBytes;
		};

		// Keep the public Scene taskgraph counters at the task boundary rather
		// than in Dy::AvbdDynamicsContext.  The solver context intentionally has
		// no profiling atomics on its hot path, while these counters let callers
		// distinguish an actually dispatched relaxed-color solve from the serial
		// component fallback.  They are reset before a new Scene graph is
		typedef AvbdStandaloneTaskGraphTelemetry StandaloneTaskGraphTelemetry;

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_SCENE_STATE_TYPES_H
