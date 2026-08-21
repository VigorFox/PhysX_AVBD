// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/solver/soft/DyAvbdSoftBodyScheduling.h"

namespace physx
{
namespace Dy
{

// AVBD scheduling-policy switches for task fan-in experiments.
//
// These are intentionally explicit opt-in controls. They carry no numerical
// solver state and keep production defaults serial/deterministic while the
// Scene-owned scheduler evolves independently.
// =============================================================================

// P4.5.2c uses this reference-only switch while the resumable lifecycle state
// is compared against the retained direct call-stack authority.  It does not
// create a worker or alter the production serial-linear default.
bool avbdUsePersistentStepStateSerial()
{
	static const bool enabled = false;
	return enabled;
}

// P4.5 task-fanin validation is deliberately opt-in.  It is consumed only
// by the Scene-owned continuation after prediction; the low-level state never
// creates a task or waits for one.  The production default remains the legacy
// serial-linear Gauss-Seidel path.
bool avbdUseCausalLayerTaskFanIn()
{
	static const bool enabled = false;
	return enabled;
}

// P6 coarse primal policy. Unlike P4's particle-color fan-in, this route
// publishes only complete, mutually independent soft bodies. Scene performs
// the collision/objective ownership proof before enabling the state, and the
// state additionally requires an empty contact epoch before every sweep. The
// rollback is sampled once and never reaches a particle loop.
bool avbdDisableIndependentBodySweepTaskFanIn()
{
	static const bool disabled = false;
	return disabled;
}

// Test-only eligibility override. It never changes the default P2/P3
// threshold, and is useful solely to exercise the P4.5 continuation on the
// small canonical-contact fixtures that intentionally fall below it.
bool avbdForceCausalLayerTaskFanIn()
{
	static const bool enabled = false;
	return enabled;
}

// This separate test-only switch forces only the P2/P3 Scene continuation for
// a small fixture.  Unlike avbdForceCausalLayerTaskFanIn(), it publishes no
// causal-layer children; its purpose is to retain the same Scene-owned serial
// oracle when comparing that continuation with a forced task fan-in route.
bool avbdForceCausalLayerTaskGraphReference()
{
	static const bool enabled = false;
	return enabled;
}

// P4.5.3b validation control.  P4.5.3a intentionally submits one whole
// published layer; splitting it into several dispatcher children remains
// opt-in until the N-worker canonical-contact and determinism matrix accepts
// it.  This is independent from the task-fanin switch so the one-range
// reference continues to be directly selectable.
bool avbdUseCausalLayerTaskPartition()
{
	static const bool enabled = false;
	return enabled;
}

// Small self/soft-contact fixtures often have causal layers with only two
// non-conflicting particles. Keep those on the one-range reference by default,
// but permit an explicit test-only override to exercise their multi-child
// parent fan-in. This must never become a production scheduling heuristic.
bool avbdForceCausalLayerTaskPartition()
{
	static const bool enabled = false;
	return enabled;
}

// Scene owns the redetection continuation whenever one of its immutable
// contact-range transactions is eligible. The Scene policy still decides
// whether a concrete component has enough work to enter this route.
bool avbdUseSceneRedetectionBridge()
{
	static const bool enabled = true;
	return enabled;
}

// Analytic contact ranges are production routes. Their Scene-side task-count
// policies keep small components serial and submit only large immutable
// particle intervals to the PhysX dispatcher.
bool avbdUseWorldPlaneContactTaskFanIn()
{
	static const bool enabled = true;
	return enabled;
}

// Static-box current and swept SDF use independent private streams. The
// parent merges complete families in canonical order before feature suffixes.
bool avbdUseRigidBoxSdfContactTaskFanIn()
{
	static const bool enabled = true;
	return enabled;
}

// Each analytic shape retains an independent eligibility proof, task pool,
// private output and parent merge despite sharing the same range policy.
bool avbdUseRigidSphereSdfContactTaskFanIn()
{
	static const bool enabled = true;
	return enabled;
}

bool avbdUseRigidCapsuleSdfContactTaskFanIn()
{
	static const bool enabled = true;
	return enabled;
}

bool avbdUseRigidConvexSdfContactTaskFanIn()
{
	static const bool enabled = true;
	return enabled;
}

} // namespace Dy
} // namespace physx
