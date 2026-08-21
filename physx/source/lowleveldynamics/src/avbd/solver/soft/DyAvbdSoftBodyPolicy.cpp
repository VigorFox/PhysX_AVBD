/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of NVIDIA CORPORATION nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "avbd/solver/soft/DyAvbdSoftBodyPolicy.h"

#include <cstdlib>
#include <cstring>

namespace physx
{
namespace Dy
{

extern "C" AvbdCpuIsaCorotationalTetPacket8Fn PX_CALL_CONV
PxAvbdCpuIsaCorotationalTetPacket8FunctionInternal();
extern "C" AvbdCpuIsaNeoHookeanTetPacket8Fn PX_CALL_CONV
PxAvbdCpuIsaNeoHookeanTetPacket8FunctionInternal();

// Internal experiment controls are process/module policy.  Every caller below
// stores this result in a function-local static before entering a simulation
// stage; never call this reader directly from an epoch, task or kernel.
static bool avbdReadProcessExactOneFlag(const char* name)
{
	const char* value = std::getenv(name);
	return value && value[0] == '1' && value[1] == '\0';
}

// Temporary physical A/B for the particle elastic proximal stabilization.
// Preserve the legacy route unless the process starts with the exact value
// "0".  This policy must only be sampled at timestep/outer-stage boundaries;
// in particular, the particle primal continues to have no new runtime branch.
bool avbdUseSoftElasticProximal()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv(
			"PHYSX_AVBD_SOFT_ELASTIC_PROXIMAL");
		return !(value && value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// Opt-in, reference-inspired A/B for adaptive primal initialization.  It is
// deliberately sampled once per module and is further
// rejected for any component that participates in speculative swept contact:
// those paths require position->predictedPosition to remain the full sweep.
bool avbdUseSoftAdaptivePrimalInitialization()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_SOFT_ADAPTIVE_INITIALIZATION");
	return enabled;
}

// Default-off A/B for a body-wide SE(3) primal initial guess.  Unlike the
// existing per-particle adaptive guess, this preserves every edge and tet of
// the accepted configuration exactly: prediction remains the inertial target
// while the primal starts at the mass-weighted rigid best fit to that target.
// Read this process policy only at a step/prediction boundary.
bool avbdUseSoftRigidPrimalInitialization()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_SOFT_RIGID_PRIMAL_INITIALIZATION");
	return enabled;
}

// Read-only qualification pass for the single-tet ground-patch experiment.
// It deliberately performs no position update: before adding any coupled
// local solve we first prove that an active world-static row really expands
// from one collision vertex to four dynamic nodes of one simulation tet.
// Keep it process-static and outside particle kernels.
bool avbdUseGroundTetPatchProbe()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_SOFT_GROUND_TET_PATCH_PROBE");
	return enabled;
}

// OGC contact uses split ownership: Position-AL owns only geometric
// non-penetration, while Coulomb friction is an end-step velocity impulse.
// Keeping a persistent positional tangent spring on an expanded collision
// proxy can constrain every simulation node behind one surface sample and make
// a deformable appear frozen at first contact.  This policy is default-on; an
// exact zero remains as a diagnostic rollback switch.
bool avbdUseVelocityTangentOwner()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv(
			"PHYSX_AVBD_CONTACT_VELOCITY_TANGENT_OWNER");
		// Preserve the previous world-static A/B switch as a compatibility
		// alias while downstream runners migrate to the target-independent name.
		if(!value)
			value = std::getenv(
				"PHYSX_AVBD_WORLD_STATIC_VELOCITY_TANGENT_OWNER");
		return !(value && value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// P8.1 begins with a passive, process-start-controlled work census. It never
// selects a packet kernel or writes shared statistics from a task: individual
// particle ranges accumulate into their existing private observation and the
// owner performs the normal deterministic merge.
bool avbdUseParticlePrimalWorkCensus()
{
	static const bool enabled = false;
	return enabled;
}

// The admitted AVX2+FMA corotational backend is on by default. Keep one
// process-start rollback switch for differential validation; ISA selection
// and rollback sampling happen once, outside every particle/tet loop.
bool avbdUseCorotationalTetPacketKernel()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv(
			"PHYSX_AVBD_DISABLE_COROTATIONAL_TET_PACKET_KERNEL");
		return !(value && value[0] == '1' && value[1] == '\0');
	}();
	return enabled;
}

AvbdCpuIsaCorotationalTetPacket8Fn
avbdGetCorotationalTetPacketKernel()
{
	static const AvbdCpuIsaCorotationalTetPacket8Fn kernel =
		avbdUseCorotationalTetPacketKernel()
			? PxAvbdCpuIsaCorotationalTetPacket8FunctionInternal()
			: NULL;
	return kernel;
}

// Neo-Hookean shares only the material-neutral packet ABI and topology IR
// with co-rotational elasticity.  Its constitutive evaluator and rollback
// switch remain independent so either backend can fall back to scalar
// authority without changing particle ownership or reduction order.
bool avbdUseNeoHookeanTetPacketKernel()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv(
			"PHYSX_AVBD_DISABLE_NEO_HOOKEAN_TET_PACKET_KERNEL");
		return !(value && value[0] == '1' && value[1] == '\0');
	}();
	return enabled;
}

AvbdCpuIsaNeoHookeanTetPacket8Fn
avbdGetNeoHookeanTetPacketKernel()
{
	static const AvbdCpuIsaNeoHookeanTetPacket8Fn kernel =
		avbdUseNeoHookeanTetPacketKernel()
			? PxAvbdCpuIsaNeoHookeanTetPacket8FunctionInternal()
			: NULL;
	return kernel;
}

// The immutable incidence program is material-neutral.  Do not compile it
// when neither selected ISA evaluator can consume it.
bool avbdUseTetMaterialPacketIr()
{
	static const bool enabled =
		avbdGetCorotationalTetPacketKernel() != NULL ||
		avbdGetNeoHookeanTetPacketKernel() != NULL;
	return enabled;
}

// The hierarchy is on by default whenever the compiled body has valid
// topology.  This intentionally remains an internal diagnostic switch: it
// permits the performance runner to compare the refit path with the exact
// retained full traversal in the same executable.  Environment selection is
// process/module policy: sampling it inside every OGC epoch serializes on the
// CRT environment lock and can dominate small self-collision workloads.
bool avbdUseSurfaceTriangleBvh()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv("PHYSX_AVBD_SURFACE_TRIANGLE_BVH");
		return !value || value[0] != '0';
	}();
	return enabled;
}

bool avbdUseSurfaceEdgeBvh()
{
	static const bool enabled = []()
	{
		if(!avbdUseSurfaceTriangleBvh())
			return false;
		const char* value = std::getenv("PHYSX_AVBD_SURFACE_EDGE_BVH");
		return !value || value[0] != '0';
	}();
	return enabled;
}

// Rigid triangle-surface topology belongs to the Scene rather than a soft
// compiled body. Keep its independent query switch so the same executable
// retains a precise full-traversal reference during P1 validation.
bool avbdUseRigidTriangleSurfaceBvh()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv("PHYSX_AVBD_RIGID_TRIANGLE_BVH");
		return !value || value[0] != '0';
	}();
	return enabled;
}

// Triangle topology, poses, feature plans and task-owned BVH scratch are all
// immutable or range-local once the Scene admits the transaction.
bool avbdUseRigidTriangleSurfaceContactTaskFanIn()
{
	return true;
}

// P5.27 is a correctness-gated feature-only repartition experiment. It never
// enables the enclosing triangle task route and, when selected, only changes
// which existing child owns a canonical feature-plan row. The parent still
// merges one private output stream per canonical row.
bool avbdUseRigidTriangleSurfaceFeatureRoundRobinTaskPlan()
{
	static const bool enabled = false;
	return enabled;
}

// P5.29 is a control for P5.27's measurement: it preserves accepted
// contiguous row ownership but uses P5.27's one-private-output-per-row
// execution and canonical parent reconstruction. It isolates that cost from
// row distribution and never enables the enclosing triangle task route.
bool avbdUseRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan()
{
	static const bool enabled = false;
	return enabled;
}

// P5.30 instruments only the already default-off triangle task route. It is
// intentionally a diagnostic switch: normal and accepted task behavior take
// no per-primitive clocks.
bool avbdUseRigidTriangleSurfaceFeatureSweptSubstageTiming()
{
	static const bool enabled = false;
	return enabled;
}

// P5.31 counts exact (surface, particle) forward-owner query multiplicity in
// the default-off triangle task route. It is diagnostic-only and does not
// cache, bypass or otherwise alter the owner predicate.
bool avbdUseRigidTriangleSurfaceFeatureForwardOwnerQueryStats()
{
	static const bool enabled = false;
	return enabled;
}

// P5.38 observes the discrete OGC query shape without a clock, allocation or
// result-cache. It is restricted to the already opt-in triangle task route so
// normal serial and accepted task execution retain no telemetry state.
bool avbdUseRigidTriangleSurfaceFeatureDiscreteQueryStats()
{
	static const bool enabled = false;
	return enabled;
}

// P5.39 is an explicitly gated task-route candidate. It rejects a complete
// discrete body/surface feature row only when the exact eight-corner local
// image of the body's world AABB misses the rigid surface's local bounds.
bool avbdUseRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull()
{
	static const bool enabled = false;
	return enabled;
}

// P5.41 promotes P5.39 only inside the already opt-in Scene triangle task
// route. The explicit disable switch retains the historical unculled task
// predicate for regression and measurement; it never changes serial/global
// behavior and must override any historical force switch.
bool avbdDisableRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull()
{
	static const bool enabled = false;
	return enabled;
}

// P5.32 introduced this result cache as a separately opt-in experiment. P5.35
// promotes it for the already opt-in triangle task route after P5.32/P5.34
// exactness and residency proof. The historical force switch remains accepted
// for reproducible P5.32 experiments; the explicit disable switch selects the
// legacy task predicate for A/B measurement and regression control.
bool avbdUseRigidTriangleSurfaceFeatureForwardOwnerResultCache()
{
	static const bool enabled = false;
	return enabled;
}

bool avbdDisableRigidTriangleSurfaceFeatureForwardOwnerResultCache()
{
	static const bool enabled = false;
	return enabled;
}

// P5.10b consumes the already-proven self-BVH range contract. The parent
// refits one self body once; children only query disjoint VF/EE outer ranges.
// It remains opt-in until the Scene two-phase merge is accepted.
bool avbdUseSelfBvhContactTaskFanIn()
{
	static const bool enabled = false;
	return enabled;
}

bool avbdForceSelfBvhContactTaskFanIn()
{
	static const bool enabled = false;
	return enabled;
}

// P4.2 builds the complete structural-plus-dynamic particle access plan only
// when explicitly requested.  Until a colored solve consumes it, keeping this
// diagnostic off preserves the authoritative serial GS hot path exactly.
bool avbdValidateParticlePrimalAccessPlan()
{
	static const bool enabled = false;
	return enabled;
}

// The ordered schedule preserves the legacy nonlinear-GS dependency order for
// reference and enhanced-determinism runs.  The relaxed schedule uses the
// same complete conflict graph, but applies a compact ordinary coloring so a
// production task graph can trade trajectory identity for throughput without
// ever admitting a shared read/write conflict.
// Returns only an explicit process policy.  Scene resolves eDEFAULT against
// its worker count and determinism contract so ordinary production callers do
// not need an environment switch merely to reach the relaxed fast path.
AvbdParticlePrimalSchedule
avbdGetConfiguredParticlePrimalSchedule()
{
	static const AvbdParticlePrimalSchedule schedule = []()
	{
		const char* value = std::getenv("PHYSX_AVBD_P4_PRIMAL_SCHEDULE");
		if(value && std::strcmp(value, "serial") == 0)
			return AvbdParticlePrimalSchedule::eSERIAL_LINEAR;
		if(value && std::strcmp(value, "colored-serial") == 0)
			return AvbdParticlePrimalSchedule::eCOLORED_SERIAL;
		if(value && std::strcmp(value, "relaxed-color") == 0)
			return AvbdParticlePrimalSchedule::eRELAXED_COLOR;
		const char* fastPath = std::getenv("PHYSX_AVBD_SOFT_FAST_PATH");
		if(fastPath && fastPath[0] != '\0')
			return avbdReadProcessExactOneFlag(
				"PHYSX_AVBD_SOFT_FAST_PATH")
				? AvbdParticlePrimalSchedule::eRELAXED_COLOR
				: AvbdParticlePrimalSchedule::eSERIAL_LINEAR;
		return AvbdParticlePrimalSchedule::eDEFAULT;
	}();
	return schedule;
}

AvbdParticlePrimalSchedule avbdGetParticlePrimalSchedule()
{
	const AvbdParticlePrimalSchedule configured =
		avbdGetConfiguredParticlePrimalSchedule();
	return configured == AvbdParticlePrimalSchedule::eDEFAULT
		? AvbdParticlePrimalSchedule::eSERIAL_LINEAR : configured;
}

// =============================================================================

} // namespace Dy
} // namespace physx
